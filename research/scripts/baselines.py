"""Scores of the utterance-level baselines on every v2 evaluation set, in score_e2e.py's format.

  linear heads   eGeMAPS, ECAPA, Whisper-medium encoder, pooled HuBERT / wav2vec 2.0: fitted exactly as in
                 the benchmark (bench_eval.linear_baseline: importance-weighted logistic regression on
                 cv_train, C chosen on cv_dev), applied to the old and the new sets
  fine-tuned     HuBERT / wav2vec 2.0 classifiers (bench_finetune.py logits)

Writes <out>/<system>.json: {set: {item_id: [log-odds, n_characters]}}, for evaluate.py.

    python research/scripts/baselines.py --bundle data/bench/bundle --out data/bench/scores_all \\
        --emb data/bench/results/emb data/bench/emb_local data/bench/v2/emb_v2 --ft data/bench/v2/ft_v2
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

from bench_eval import linear_baseline  # noqa: E402

SETS = ["cv_dev", "cv_test", "cv_heldout", "kespeech", "breezy", "cosy300m", "ming_b0", "synth2_breezy", "synth2_cosy"]
LINEAR = {"egemaps": "egemaps", "ecapa": "ecapa", "whisper": "whisper", "hubert_utt": "pooled_hubert", "w2v_utt": "pooled_w2v"}
FT = {"ft_hubert_s0": "ft_hubert", "ft_wav2vec2_s0": "ft_w2v"}


class SetsOnly:
    """The part of bench_eval.Bench that linear_baseline reads, for sets without feature files."""

    def __init__(self, bundle: Path, sets=SETS):
        self.sets = {s: [json.loads(l) for l in open(bundle / "sets" / f"{s}.jsonl")] for s in list(sets) + ["cv_train"]}
        self.lens = {"cv_dev": np.load(bundle / "feat" / "feat_cv_dev.npz")["lens"]}

    def ids(self, s):
        return [r["item_id"] for r in self.sets[s]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb", nargs="+", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--sets", nargs="+", default=SETS, help="sets to score (cv_dev is always needed to fit the heads)")
    a = ap.parse_args()
    sets = list(dict.fromkeys(["cv_dev", *a.sets]))
    B, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = SetsOnly(B, sets)
    nchar = {s: {r["item_id"]: len(re.sub(r"[^㐀-鿿]", "", r["text"])) for r in bench.sets[s]} for s in sets}

    def write(name, logits):
        missing = [s for s in sets if s not in logits]
        if missing:
            print(f"{name}: no scores for {missing}", flush=True)
        json.dump({s: {i: ([float(logits[s][i]), nchar[s][i]] if i in logits[s] else None) for i in nchar[s]}
                   for s in sets if s in logits}, open(out / f"{name}.json", "w"))
        print("wrote", name, flush=True)

    for emb, name in LINEAR.items():
        lg = linear_baseline(emb, [Path(d) for d in a.emb], bench)
        if lg is None:
            print("missing embeddings", emb, flush=True)
            continue
        write(name, lg)
    for tag, name in FT.items():
        p = Path(a.ft) / f"{tag}_logits.json"
        if p.exists():
            write(name, json.load(open(p)))


if __name__ == "__main__":
    main()
