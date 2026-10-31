"""Per-syllable log-odds of a 12-layer teacher judge on per-layer feature files, for distillation.

The teacher judge (bench_train.py checkpoint) reads layers 4, 8, 12 in the `ssl_features.pool`
layout [syllable L4, L8, L12 | rhyme L4, L8, L12]; the per-layer files of compact_features.py hold
[syllable, rhyme] per layer, so the columns are reordered.

    python research/scripts/teacher_syllables.py --feat /content/bench/tf --judge runs_ctc/c_adv0_s0.pt \
        --sets cv_train --out /content/bench/teacher_syl_cv_train.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--set", default="cv_train")
    ap.add_argument("--layers", default="4,8,12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", help="the judge's stored logits json: report agreement of the utterance means")
    a = ap.parse_args()
    import torch

    from twjudge.accent import neural as N

    fd = Path(a.feat)
    f = np.load(fd / f"feat_{a.set}.npz")
    parts = [np.load(fd / f"L{l}_{a.set}.npz")["F"] for l in a.layers.split(",")]
    D = parts[0].shape[1] // 2
    F = np.concatenate([q[:, :D] for q in parts] + [q[:, D:] for q in parts], axis=1)
    ids = [str(i) for i in f["ids"]]
    cols = [N.SYL_NAMES.index(c) for c in N.CTX]
    d = {"ids": ids, "y": np.zeros(len(ids), np.float32), "lens": f["lens"], "ctx": f["X"][:, cols].astype(np.float32),
         "X": f["X"], "F": F}
    ck = torch.load(a.judge, map_location="cpu")
    net = N.SyllableNet(ck["d_in"]); net.load_state_dict(ck["state"]); net.eval().to(N.device())
    z = []
    with torch.no_grad():
        for s in range(0, len(ids), 128):
            idx = np.arange(s, min(s + 128, len(ids)))
            ctx, feat, mask = N.batches(d, idx, True, N.device())
            o = net(ctx, feat, mask).cpu().numpy()
            z += [o[k, :d["lens"][i]] for k, i in enumerate(idx)]
    np.savez(a.out, ids=np.array(ids), lens=f["lens"], z=np.concatenate(z).astype(np.float32))
    print(len(ids), "utterances")
    if a.check:
        import json

        ref = json.load(open(a.check))[a.set]
        u = np.array([v.mean() for v in z]); r = np.array([ref.get(i, np.nan) for i in ids])
        m = ~np.isnan(r)
        print("check: corr", float(np.corrcoef(u[m], r[m])[0, 1]), "max abs diff", float(np.abs(u[m] - r[m]).max()))


if __name__ == "__main__":
    main()
