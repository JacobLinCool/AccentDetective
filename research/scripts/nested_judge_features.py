"""Features for the nested judges: per-syllable states of every exit of the nested Accent Detective, both segmentations.

For every clip: one pass of the nested trunk gives every exit's predicted teacher layers and exit 1's CTC emissions;
  forced  the CTC head forced to the given text (as Accent Detective)
  free    no text: greedy decoding split into pinyin syllables (syllable_features.split_syllables), then forced to them;
          tone, sandhi and the Mainland-neutral flag unknown
Each layer's states have their utterance mean removed and are averaged over each syllable and its rhyme.

Writes <out>/<seg>/feat_<set>.npz (ids, lens, ctx, types) and <out>/<seg>/N<k><tag>_<set>.npy (float16), for exits
k = 1… and tags v4 v8 v12 b8 b12. Run in shards, then --merge (syllable_features.merge).

    python research/scripts/nested_judge_features.py --bundle /content/bench --nested /content/nested/student.pt \\
        --detective /content/det/model.safetensors --sets cv_train cv_dev --out /content/bench/r9f --shard 0/8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

from syllable_features import UNK, inventory, merge, split_syllables  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle")
    ap.add_argument("--nested")
    ap.add_argument("--detective", help="deployed checkpoint, for the readings and the aligner only")
    ap.add_argument("--sets", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--oracle", action="store_true", help="also the teachers' own states of the same layers, as one more "
                                                            "exit after the last (the not deployable reference O)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.merge:
        return merge(out)
    import librosa
    import torch
    from pypinyin.contrib.tone_convert import to_initials

    from twjudge.accent.compact import VOCAB, CompactJudge, context
    from twjudge.accent.nested import build_student
    from twjudge.accent.ssl_features import pool_fast
    from twjudge.lexicon.readings import TaiwanReadings

    B = Path(a.bundle)
    shard, n_shard = map(int, a.shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.nested, map_location="cpu", weights_only=False)
    net = build_student(ck["config"])
    net.load_state_dict(ck["state"])
    net = net.to(dev).eval()
    teachers = {}
    if a.oracle:
        from twjudge.accent.ssl_features import load_model

        teachers = {"v": load_model("wav2vec2").to(dev).eval(), "b": load_model("hubert").to(dev).eval()}
    rw = TaiwanReadings(ROOT / "data/lexicon/concised.json.xz")
    aligner = CompactJudge(a.detective, readings=rw, device="cpu", mode="fast").aligner
    letters = {v: k for k, v in VOCAB.items()}
    inv = inventory()
    for seg in ("forced", "free"):
        (out / seg).mkdir(parents=True, exist_ok=True)

    def free_chars(em, w):
        s, prev = [], 0
        for t in em[0].argmax(-1).tolist():
            if t != prev and t != 0:
                s.append(letters[t])
            prev = t
        sylls = [p for p, _ in split_syllables("".join(s), inv)]
        if len(sylls) < 2:
            return None
        tup = [("?", p + "5", False, p, len(to_initials(p, strict=False) or ""), p + "5") for p in sylls]
        return aligner.spans(em, VOCAB, tup, w)

    def ctx_of(chars, free):
        if not free:
            return context(chars)
        n = len(chars)
        fin = [k + 1 == n or chars[k + 1]["start"] - c["end"] >= 0.15 for k, c in enumerate(chars)]
        return [[*UNK, int(f), k / max(n - 1, 1)] for k, f in enumerate(fin)]

    def pooled(h, chars):          # h (L, T, D) → [(n, 2D)] per layer, utterance mean removed
        h = h.float()
        h = h - h.mean(1, keepdim=True)
        L, D = h.shape[0], h.shape[2]
        f = pool_fast(h, chars).reshape(len(chars), 2, L, D).permute(0, 2, 1, 3).reshape(len(chars), L, 2 * D)
        return [f[:, k].half().cpu().numpy() for k in range(L)]

    for name in a.sets:
        p = B / "sets" / f"{name}.jsonl"
        if not p.exists():
            print(f"{name}: no set file, skipped", flush=True)
            continue
        rows = [json.loads(l) for l in open(p)][shard::n_shard]
        acc = {seg: {"ids": [], "lens": [], "ctx": [], "types": [], "F": {}} for seg in ("forced", "free")}
        t0 = time.time()

        def put(seg, rid, chars, states):
            A = acc[seg]
            A["ids"].append(rid); A["lens"].append(len(chars))
            A["ctx"].append(np.array(ctx_of(chars, seg == "free"), np.float32))
            A["types"] += [c["syl"].rstrip("012345") for c in chars]
            for e, H in enumerate(states):
                for t, f in zip(net.targets, pooled(H, chars)):
                    A["F"].setdefault(f"N{e + 1}{t}", []).append(f)

        for k, r in enumerate(rows):
            w = librosa.load(str(B / r["audio"]), sr=16000, mono=True)[0]
            with torch.inference_mode():
                outs = net(torch.tensor(w, dtype=torch.float32, device=dev)[None])
                em = net.emissions(outs).float().cpu()
                states = [torch.stack([o[t][0] for t in net.targets]) for o in outs]
                if teachers:                        # the teachers' own states, same frames (wav2vec 2.0 CNN arithmetic)
                    x = torch.tensor(w, dtype=torch.float32, device=dev)[None]
                    x = (x - x.mean()) / (x.std() + 1e-7)
                    hs = {k: m(x, output_hidden_states=True).hidden_states for k, m in teachers.items()}
                    T = states[0].shape[1]
                    o = torch.stack([hs[t[0]][int(t[1:])][0, :T] for t in net.targets])
                    if o.shape[1] < T:
                        o = torch.nn.functional.pad(o, (0, 0, 0, T - o.shape[1]))
                    states.append(o)
            forced = aligner.align(w, r["text"], emissions=(em, VOCAB))
            if forced is None or len(forced) < 2:
                continue
            with torch.inference_mode():
                put("forced", r["item_id"], forced, states)
                free = free_chars(em, w)
                if free is not None:
                    put("free", r["item_id"], free, states)
            if k % 2000 == 0:
                print(f"{name}[{shard}/{n_shard}] {k}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        part = f".part{shard}"
        for seg, A in acc.items():
            if not A["ids"]:
                continue
            np.savez(out / seg / f"feat_{name}{part}.npz", ids=np.array(A["ids"]), lens=np.array(A["lens"]),
                     ctx=np.concatenate(A["ctx"]), types=np.array(A["types"]))
            for tag, F in A["F"].items():
                np.save(out / seg / f"{tag}_{name}{part}.npy", np.concatenate(F))
        print(f"{name}[{shard}/{n_shard}]: forced {len(acc['forced']['ids'])}, free {len(acc['free']['ids'])} of {len(rows)}, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
