"""Collect the per-utterance scores of every system in the paper into one directory, in score_e2e.py's format, for
evaluate.py (paper/README.md).

  baselines and syllable judges   data/bench/scores_all, plus their r10_sealed scores
                                  from the paper evaluation run (results/paper_r15.tar: score_e2e.py, bench_finetune.py,
                                  and baselines.py's linear heads), when present
  Accent Detective                the four sizes of the released checkpoint (data/bench/released, audio only and
                                  with the text), calibrated with the checkpoint's own Platt transforms
  readouts on one trunk           the same arm-C trunk read by the syllable judge (with and without text), the ASP
                                  readout and ASP plus the text readout (scores_readouts/)

Also writes <out>/sets: the bundle's set files plus the sealed set.

    python paper/scripts/collect_scores.py --model model.safetensors --out data/bench/paper
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "data/bench"
NAMES = ("nano", "tiny", "mini", "small")
DROP = {"old", "v2A", "v2B"}                       # earlier students, not in the paper


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the released checkpoint (its Platt transforms)")
    ap.add_argument("--sealed", nargs="*", default=[], help="dirs of r10_sealed score files (<system>.json) to merge")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors import safe_open

    with safe_open(a.model, framework="pt") as f:
        platt = json.loads(f.metadata()["accent_detective"])["platt"]
    out = Path(a.out)
    shutil.rmtree(out / "scores", ignore_errors=True); shutil.rmtree(out / "scores_readouts", ignore_errors=True)
    (out / "scores").mkdir(parents=True); (out / "scores_readouts").mkdir(parents=True); (out / "sets").mkdir(exist_ok=True)
    for p in (B / "sets").glob("*.jsonl"):
        shutil.copy(p, out / "sets" / p.name)
    shutil.copy(B / "sealed/r10_sealed.jsonl", out / "sets/r10_sealed.jsonl")
    sealed = {}
    for d in a.sealed:
        for p in Path(d).glob("*.json"):
            s = json.load(open(p))
            if "r10_sealed" in s:
                sealed[p.stem] = s["r10_sealed"]
    for p in sorted((B / "scores_all").glob("*.json")):
        if p.stem in DROP:
            continue
        s = json.load(open(p))
        if p.stem in sealed:
            s["r10_sealed"] = sealed[p.stem]
        json.dump(s, open(out / "scores" / p.name, "w"))
    keep = {"cv_dev", "cv_test", "cv_heldout", "kespeech", "breezy", "cosy300m", "ming_b0", "synth2_breezy", "synth2_cosy",
            "r10_sealed"}

    def calibrated(src, key, mode):
        s, b = platt[key][mode]
        S = json.load(open(src))
        return {k: {i: (None if v is None else [s * v[0] + b, v[1]]) for i, v in S[k].items()} for k in keep if k in S}

    for k, name in enumerate(NAMES, 1):
        json.dump(calibrated(B / f"released/scores/N{k}a.json", f"N{k}", "audio"), open(out / "scores" / f"detective_{name}.json", "w"))
        json.dump(calibrated(B / f"released/scores/N{k}x.json", f"N{k}", "text"),
                  open(out / "scores_readouts" / f"detective_{name}_text.json", "w"))
    # the readouts of one trunk (arm C, seed 0, 18 layers): raw log-odds (only rank-based metrics are compared)
    for src, name in ((B / "trunk_c/scores/N4t.json", "syllable_text"), (B / "trunk_c/scores/N4f.json", "syllable_audio"),
                      (B / "trunk_c/readouts/N4a.json", "asp_audio"), (B / "trunk_c/readouts/N4x.json", "asp_text")):
        S = json.load(open(src))
        json.dump({k: v for k, v in S.items() if k in keep}, open(out / "scores_readouts" / f"{name}.json", "w"))
    print(sorted(p.stem for p in (out / "scores").glob("*.json")), "sealed merged:", sorted(sealed))


if __name__ == "__main__":
    main()
