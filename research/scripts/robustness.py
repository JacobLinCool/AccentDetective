"""Recording conditions: per run and cell, the length-matched AUC of cv_heldout and the
sealed set under every channels.py condition, its change from clean audio, and the mean score shift per class in units
of the clean scores' standard deviation (positive = more Taiwan-like).

    python research/scripts/robustness.py --runs data/bench/runs --bundle data/bench/bundle \\
        --sealed-sets data/bench/sealed --out data/bench/released/eval
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

from twjudge.accent.prosodic import auc, length_weights  # noqa: E402

CONDS = ("mp3", "tel", "noise", "reverb")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--glob", default="r14_*")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--sealed-sets", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nchar = lambda t: len(re.sub(r"[^㐀-鿿]", "", t))
    text = {"cv_heldout": {json.loads(l)["item_id"]: json.loads(l)["text"] for l in open(Path(a.bundle) / "sets" / "cv_heldout.jsonl")},
            "r10_sealed": {json.loads(l)["item_id"]: json.loads(l)["text"] for l in open(Path(a.sealed_sets) / "r10_sealed.jsonl")}}
    res, L = {}, ["| 系統 | 格 | 集合 | 乾淨 AUC | " + " | ".join(f"{c} AUC（變化）" for c in CONDS) + " | 最大位移（SD） |",
                  "|" + "---|" * (5 + len(CONDS))]
    for run in sorted(p for p in Path(a.runs).glob(a.glob) if (p / "scores").is_dir()):
        for f in sorted((run / "scores").glob("N*.json")):
            S = json.load(open(f))
            for s in ("cv_heldout", "r10_sealed"):
                if s not in S:
                    continue
                ids = sorted(i for i, v in S[s].items() if v is not None and i in text[s]
                             and all(S.get(f"{s}@{c}", {}).get(i) is not None for c in CONDS))
                if not ids:
                    continue
                y = np.array([i.startswith("tw") for i in ids]).astype(int)
                lw = length_weights(np.array([nchar(text[s][i]) for i in ids]), y)
                z0 = np.array([S[s][i][0] for i in ids])
                r = {"clean_auc": float(auc(y, z0, lw)), "n": len(ids)}
                for c in CONDS:
                    z = np.array([S[f"{s}@{c}"][i][0] for i in ids])
                    r[c] = {"auc": float(auc(y, z, lw)), "shift_tw": float((z - z0)[y == 1].mean() / z0.std()),
                            "shift_cn": float((z - z0)[y == 0].mean() / z0.std())}
                res.setdefault(run.name, {}).setdefault(f.stem, {})[s] = r
                worst = max(max(abs(r[c]["shift_tw"]), abs(r[c]["shift_cn"])) for c in CONDS)
                L.append(f"| {run.name} | {f.stem} | {s} | {r['clean_auc']:.3f} | "
                         + " | ".join(f"{r[c]['auc']:.3f}（{r[c]['auc'] - r['clean_auc']:+.3f}）" for c in CONDS) + f" | {worst:.2f} |")
    json.dump(res, open(out / "robust.json", "w"), indent=1)
    (out / "robust.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
