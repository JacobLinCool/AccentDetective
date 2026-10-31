"""Syllable features: per-syllable states of every source and segmentation the unified nested judge reads.

For every clip of the given sets:
  segmentation  forced  the student's own CTC head, forced to the given text (as Accent Detective)
                free    no text: the same head decoded greedily, the letters split into pinyin syllables (fewest
                        syllables, each in the toneless inventory), then forced to that syllable sequence
  states        S  the dual student's predictions: wav2vec 2.0 layers 4, 8, 12 and Whisper layers 6, 12
                T  the teachers themselves: wav2vec 2.0 layers 4, 8, 12 and Whisper-medium layers 6, 12, 18, 24
                   (forced segmentation only)
Each layer's states have their utterance mean removed and are averaged over each syllable and over its rhyme
([syllable, rhyme] per layer, as layer_features.py). The text-free segmentation keeps only what timing gives:
phrase-final and relative position; tone, sandhi and the Mainland-neutral flag are written as unknown (6, 6, 2).

Writes <out>/<seg>/feat_<set>.npz (ids, lens, ctx (syllables × 5), types: toneless syllable per row) and
<out>/<seg>/<S|T><layer>_<set>.npy (syllables × 2·hidden, float16), layer tags v4 v8 v12 (wav2vec 2.0) h6 h12 (Whisper;
h18 h24: T only).
Run in shards, then --merge.

    python research/scripts/syllable_features.py --bundle /content/bench --detective /content/det/model.safetensors \\
        --sets cv_train cv_dev --out /content/bench/r2f --shard 0/2
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

WHISPER = "openai/whisper-medium"
T_TAGS = ["v4", "v8", "v12", "h6", "h12", "h18", "h24"]   # v: wav2vec 2.0, h: Whisper (not case-only, for case-insensitive disks)
S_TAGS = ["v4", "v8", "v12", "h6", "h12"]          # the deployed student exports Whisper 6, 12
UNK = (6, 6, 2)


def inventory():
    """Toneless pinyin syllables (letters a–z, ü as v) of the CJK block, from pypinyin."""
    from pypinyin import Style, pinyin

    inv = set()
    for c in range(0x4E00, 0xA000):
        for p in pinyin(chr(c), style=Style.NORMAL, heteronym=True)[0]:
            p = p.replace("ü", "v")
            if p.isalpha() and p.isascii():
                inv.add(p)
    return inv


def split_syllables(s: str, inv: set, maxlen=6, skip_cost=10):
    """Split a decoded letter string into inventory syllables: fewest syllables, a letter that fits no syllable
    (a decoding error) may be skipped at a cost of `skip_cost` syllables. Returns [(syllable, start index)]."""
    n = len(s)
    best = [None] * (n + 1)                  # (cost, [(syllable, start)])
    best[n] = (0, [])
    for i in range(n - 1, -1, -1):
        opts = [(best[i + 1][0] + skip_cost, best[i + 1][1])]
        for L in range(min(maxlen, n - i), 0, -1):
            if s[i:i + L] in inv:
                opts.append((best[i + L][0] + 1, [(s[i:i + L], i)] + best[i + L][1]))
        best[i] = min(opts, key=lambda o: o[0])
    return best[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle")
    ap.add_argument("--detective", default=None)
    ap.add_argument("--sets", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-teacher", action="store_true", help="student states only (no T files)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.merge:
        return merge(out)
    import librosa
    import torch
    from pypinyin.contrib.tone_convert import to_initials
    from transformers import WhisperFeatureExtractor, WhisperModel

    from twjudge.accent.compact import VOCAB, CompactJudge, context
    from twjudge.accent.ssl_features import frame_states, load_model, pool_fast
    from twjudge.lexicon.readings import TaiwanReadings

    B = Path(a.bundle)
    shard, n_shard = map(int, a.shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rw = TaiwanReadings(ROOT / "data/lexicon/concised.json.xz")
    cj = CompactJudge(a.detective, readings=rw, device=dev, mode="accurate")
    assert cj.w, "a dual (wav2vec 2.0 + Whisper) checkpoint is needed"
    letters = {v: k for k, v in VOCAB.items()}
    inv = inventory()
    if not a.no_teacher:
        ssl = load_model("wav2vec2").to(dev).eval()
        fe = WhisperFeatureExtractor.from_pretrained(WHISPER)
        wenc = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).encoder.to(dev).eval()
    for seg in ("forced", "free"):
        (out / seg).mkdir(parents=True, exist_ok=True)

    def free_chars(em, w):
        path = em[0].argmax(-1).tolist()
        s, prev = [], 0
        for t in path:
            if t != prev and t != 0:
                s.append(letters[t])
            prev = t
        sylls = [p for p, _ in split_syllables("".join(s), inv)]
        if len(sylls) < 2:
            return None
        tup = [("?", p + "5", False, p, len(to_initials(p, strict=False) or ""), p + "5") for p in sylls]
        return cj.aligner.spans(em, VOCAB, tup, w)

    def ctx_of(chars, free):
        if not free:
            return context(chars)
        n = len(chars)
        fin = [k + 1 == n or chars[k + 1]["start"] - c["end"] >= 0.15 for k, c in enumerate(chars)]
        return [[*UNK, int(f), k / max(n - 1, 1)] for k, f in enumerate(fin)]

    for name in a.sets:
        rows = [json.loads(l) for l in open(B / "sets" / f"{name}.jsonl")][shard::n_shard]
        acc = {seg: {"ids": [], "lens": [], "ctx": [], "types": [], "F": {}} for seg in ("forced", "free")}
        t0 = time.time()
        buf = []

        def put(seg, rid, chars, layers):
            A = acc[seg]
            A["ids"].append(rid); A["lens"].append(len(chars))
            A["ctx"].append(np.array(ctx_of(chars, seg == "free"), np.float32))
            A["types"] += [c["syl"].rstrip("012345") for c in chars]
            for tag, f in layers.items():
                A["F"].setdefault(tag, []).append(f)

        def pooled(h, chars):          # h (L, T, D) → {layer k: (n, 2D)} with the utterance mean removed
            h = h.float()
            h = h - h.mean(1, keepdim=True)
            L, D = h.shape[0], h.shape[2]
            f = pool_fast(h, chars).reshape(len(chars), 2, L, D).permute(0, 2, 1, 3).reshape(len(chars), L, 2 * D)
            return [f[:, k].half().cpu().numpy() for k in range(L)]

        def flush():
            if not buf:
                return
            with torch.inference_mode():
                if not a.no_teacher:
                    wavs = [b["w"] for b in buf]
                    feats = fe([w[: 30 * 16000] for w in wavs], sampling_rate=16000, return_tensors="pt").input_features
                    hs = wenc(feats.to(dev, next(wenc.parameters()).dtype), output_hidden_states=True).hidden_states
                    H = torch.stack([hs[l] for l in (6, 12, 18, 24)], 1)
                for k, b in enumerate(buf):
                    x = torch.tensor(b["w"], dtype=torch.float32, device=dev)
                    sw, sW = b["S"]
                    S = pooled(sw, b["forced"]) + pooled(sW, b["forced"])
                    layers = {f"S{t}": f for t, f in zip(S_TAGS, S)}
                    if not a.no_teacher:
                        tw = frame_states(ssl, x)
                        n = max(1, min(1500, math.ceil(min(len(b["w"]), 30 * 16000) / 320)))
                        tW = H[k, :, :n]
                        T = pooled(tw, b["forced"]) + pooled(tW, b["forced"])
                        layers.update({f"T{t}": f for t, f in zip(T_TAGS, T)})
                    put("forced", b["id"], b["forced"], layers)
                    if b["free"] is not None:
                        Sf = pooled(sw, b["free"]) + pooled(sW, b["free"])
                        put("free", b["id"], b["free"], {f"S{t}": f for t, f in zip(S_TAGS, Sf)})
            buf.clear()

        for k, r in enumerate(rows):
            w = librosa.load(str(B / r["audio"]), sr=16000, mono=True)[0]
            with torch.inference_mode():
                x = torch.tensor(w, dtype=torch.float32, device=dev)
                out_w, out_W = cj.enc.both(x[None])
                sw = torch.stack([out_w[l][0] for l in (4, 8, 12)])
                sW = torch.stack([out_W[l][0] for l in (6, 12)])
                T_ = min(sw.shape[1], sW.shape[1])
                em = torch.log_softmax(cj.head(out_w[cj.ck["ctc_layer"]][0]), dim=-1)[None].float().cpu()
            forced = cj.aligner.align(w, r["text"], emissions=(em, VOCAB))
            if forced is None or len(forced) < 2:
                continue
            free = free_chars(em, w)
            buf.append({"id": r["item_id"], "w": w, "S": (sw[:, :T_], sW[:, :T_]), "forced": forced, "free": free})
            if len(buf) >= a.batch:
                flush()
            if k % 2000 == 0:
                print(f"{name}[{shard}/{n_shard}] {k}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        flush()
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


def merge(out: Path):
    """Concatenate the shard parts of every set and segmentation into feat_<set>.npz and <tag>_<set>.npy."""
    for seg in ("forced", "free"):
        d = out / seg
        if not d.exists():
            continue
        for first in sorted(d.glob("feat_*.part0.npz")):
            name = first.name[len("feat_"):-len(".part0.npz")]
            parts = sorted(d.glob(f"feat_{name}.part*.npz"))
            fs = [np.load(p) for p in parts]
            np.savez(d / f"feat_{name}.npz", ids=np.concatenate([f["ids"] for f in fs]), lens=np.concatenate([f["lens"] for f in fs]),
                     ctx=np.concatenate([f["ctx"] for f in fs]), types=np.concatenate([f["types"] for f in fs]))
            tags = sorted({p.name.split("_", 1)[0] for p in d.glob(f"*_{name}.part0.npy")})
            for tag in tags:
                arrs = [np.load(d / f"{tag}_{name}.part{p.name.rsplit('.part', 1)[1].split('.')[0]}.npy") for p in parts]
                np.save(d / f"{tag}_{name}.npy", np.concatenate(arrs))
            for p in parts:
                p.unlink()
            for tag in tags:
                for p in d.glob(f"{tag}_{name}.part*.npy"):
                    p.unlink()
            print(f"merged {seg}/{name}", flush=True)


if __name__ == "__main__":
    main()
