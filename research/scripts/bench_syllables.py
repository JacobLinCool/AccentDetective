"""Per-syllable log-odds of trained judges on chosen sets (which syllables make an utterance "Taiwan").

    python research/scripts/bench_syllables.py --feat /content/bench/feat_abl --models runs_ctc/b_adv0_s*.pt ... \
        --sets kespeech cv_test --out /content/bench/syllables
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
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from twjudge.accent import neural as N

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for mp in a.models:
        ck = torch.load(mp, map_location="cpu")
        net = N.SyllableNet(ck["d_in"]); net.load_state_dict(ck["state"]); net.eval().to(N.device())
        for s in a.sets:
            d = N.load_pair(Path(a.feat) / f"feat_{s}.npz", Path(a.feat) / f"{ck['ssl']}_{s}.npz")
            syl = []
            with torch.no_grad():
                for st in range(0, len(d["y"]), 128):
                    idx = np.arange(st, min(st + 128, len(d["y"])))
                    ctx, feat, mask = N.batches(d, idx, True, N.device())
                    z = net(ctx, feat, mask).cpu().numpy()
                    syl += [z[k, :d["lens"][i]] for k, i in enumerate(idx)]
            np.savez(out / f"{Path(mp).stem}_{s}.npz", ids=np.array(d["ids"]), lens=d["lens"], z=np.concatenate(syl))
            print(mp, s, len(d["ids"]), flush=True)


if __name__ == "__main__":
    main()
