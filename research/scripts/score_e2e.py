"""Score benchmark sets end to end from audio: the teacher judges and any Accent Detective checkpoint.

Every system aligns and scores each clip itself, so all of them are compared on exactly what they
would do in use, on the same machine:

  teacher     MMS CTC alignment + 12-layer SSL + each given judge checkpoint (states computed once,
              shared by the seeds); written per seed
  detective   an exported compact checkpoint (.safetensors / .pt), CUDA graphs on a GPU

Writes <out>/<system>.json: {set: {item_id: [log-odds, n_syllables]}} (null when unalignable).

    python research/scripts/score_e2e.py --bundle /content/bench --sets cv_dev cv_test cv_heldout kespeech \\
        --teacher runs_ctc/c_adv0_s0.pt runs_ctc/c_adv0_s1.pt --detective old=judge_compact.safetensors --out scores
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--teacher", nargs="*", default=[])
    ap.add_argument("--detective", nargs="*", default=[], help="name=checkpoint[@fast]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-graphs", action="store_true", help="eager execution on the GPU (no CUDA graphs)")
    a = ap.parse_args()
    import librosa
    import torch

    from twjudge.lexicon.readings import TaiwanReadings

    B, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rw = TaiwanReadings(ROOT / "data/lexicon/concised.json.xz")
    rows = {s: [json.loads(l) for l in open(B / "sets" / f"{s}.jsonl")] for s in a.sets}

    def audio(r):
        return librosa.load(str(B / r["audio"]), sr=16000, mono=True)[0]

    def save(name, res):
        json.dump(res, open(out / f"{name}.json", "w"))

    if a.teacher:
        from twjudge.accent.neural import JudgeB, utt_logits
        from twjudge.accent.ssl_features import frame_states, load_model, pool
        from twjudge.align.ctc import CTCAligner
        from twjudge.phonology import sandhi_tones, split_syllable

        judges = [JudgeB(p) for p in a.teacher]
        ssl = {k: load_model(k) for k in sorted({j.ssl_kind for j in judges})}     # one encoder per kind of judge
        dev = next(next(iter(ssl.values())).parameters()).device
        for j in judges:
            j.net.to(dev)
        aligner = CTCAligner(rw)
        names = [f"teacher_{Path(p).stem}" for p in a.teacher]
        res = {n: {} for n in names}
        for s, rs in rows.items():
            t0 = time.time()
            for n in names:
                res[n][s] = {}
            for r in rs:
                w = audio(r)
                chars = aligner.align(w, r["text"])
                if chars is None or len(chars) < 2:
                    for n in names:
                        res[n][s][r["item_id"]] = None
                    continue
                with torch.inference_mode():
                    x = torch.tensor(w, device=dev)
                    feats = {k: pool(frame_states(m_, x), chars)[None].float() for k, m_ in ssl.items()}
                    tones = [split_syllable(c["syl"])[2] for c in chars]
                    m = len(chars)
                    fin = [k + 1 == m or chars[k + 1]["start"] - c["end"] >= 0.15 for k, c in enumerate(chars)]
                    ctx = torch.tensor([[int(t), int(st), int(c["cn_neutral"]), int(f), k / max(m - 1, 1)]
                                        for k, (t, st, c, f) in enumerate(zip(tones, sandhi_tones(tones), chars, fin))],
                                       dtype=torch.float32, device=dev)[None]
                    mask = torch.zeros((1, m), dtype=torch.bool, device=dev)
                    for n, j in zip(names, judges):
                        res[n][s][r["item_id"]] = [float(utt_logits(j.net(ctx, feats[j.ssl_kind], mask), mask)[0]), m]
            print(f"teacher {s}: {len(rs)} in {time.time() - t0:.0f}s", flush=True)
            for n in names:
                save(n, res[n])
    for spec in a.detective:
        name, path = spec.split("=", 1)
        path, _, mode = path.partition("@")          # name=checkpoint@fast: a dual checkpoint's fast mode
        from twjudge.detective import _needs_eager

        if torch.cuda.is_available() and not a.no_graphs and not _needs_eager(path):
            from twjudge.accent.compact_fast import FastCompactJudge

            cj = FastCompactJudge(path, readings=rw)
        else:
            from twjudge.accent.compact import CompactJudge

            cj = CompactJudge(path, readings=rw, device="cuda" if torch.cuda.is_available() else "cpu",
                              mode=mode or "accurate")
        res = {}
        for s, rs in rows.items():
            t0 = time.time()
            res[s] = {}
            for r in rs:
                o = cj.score(audio(r), r["text"])
                res[s][r["item_id"]] = None if o is None or len(o[2]) < 2 else [float(o[0]), len(o[2])]
            print(f"{name} {s}: {len(rs)} in {time.time() - t0:.0f}s", flush=True)
            save(name, res)


if __name__ == "__main__":
    main()
