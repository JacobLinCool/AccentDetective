"""Per-utterance scores of the systems the paper plots, on the common items of the benchmark.

Loads every system exactly as `research/scripts/bench_eval.py` does (so the item intersection matches the
report), keeps the ones the figures need, and also scores two ensembles the report does not contain:
the three-seed ensembles of the teacher judges, the like-for-like reference for Accent Detective's
six-seed ensemble.

    PYTHONPATH=research/src python paper/scripts/export_scores.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

import bench_eval as E  # noqa: E402

FINAL = "ensemble runs_scompact/w2v_conf_d384_L4_h4convkd1_L4-8"
KEEP = {
    "judge B": "teacher_hubert", "judge C": "teacher_w2v", FINAL: "detective",
    "runs_scompact/w2v_conf_d384_L4_h4convkd1_L4-8": "detective_single",
    "fine-tuned wav2vec 2.0": "ft_w2v", "fine-tuned HuBERT": "ft_hubert", "Whisper-medium + linear": "whisper",
    "wav2vec 2.0 utterance pooling + linear": "uttpool_w2v", "HuBERT utterance pooling + linear": "uttpool_hubert",
    "wav2vec 2.0 syllable-feature pooling + linear": "sylpool_w2v", "eGeMAPS + linear": "egemaps",
    "ECAPA + linear": "ecapa", "text-only": "text_only",
}
SETS = ("cv_test", "cv_dev", "breezy", "cosy300m", "ming_b0", "kespeech")


def main() -> None:
    E.N_BOOT = 1000
    bench = E.Bench(ROOT / "data/bench/bundle", ROOT / "data/bench/cv_cn_birthplace.json")
    systems = E.load_systems(ROOT / "data/bench/results", ROOT / "data/bench/emb_local", bench)
    out = {"meta": {s: bench.sets[s] for s in SETS}, "scores": {}}
    for name, key in KEEP.items():
        out["scores"][key] = [{s: run[s] for s in SETS if s in run} for run in systems[name]]
    to = systems["text-only"][0]["cv_test"]
    extra = {}
    for name, key in (("judge B", "teacher_hubert_ens3"), ("judge C", "teacher_w2v_ens3")):
        runs = systems[name]
        ens = [{s: np.mean([r[s] for r in runs], axis=0) for s in runs[0]}]
        out["scores"][key] = [{s: ens[0][s] for s in SETS if s in ens[0]}]
        extra[key] = E.evaluate(key, ens, bench, to)
        print(key, {k: round(extra[key][k]["mean"], 3) for k in ("auc_length_matched", "ece_raw", "voice_share",
                                                                   "synthetic_frac_breezy_first")}, flush=True)
    pickle.dump(out, open(ROOT / "paper/data/scores.pkl", "wb"))
    json.dump(extra, open(ROOT / "paper/data/teacher_ensembles.json", "w"), indent=1)


if __name__ == "__main__":
    main()
