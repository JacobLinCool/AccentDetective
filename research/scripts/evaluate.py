"""Evaluate end-to-end scores (score_e2e.py) of the teacher, the current and new Accent Detectives.

Every metric is computed on the items all systems scored. Checks:
  real speech     CV test (the old test set, used to choose the current model) and CV held-out (new
                  speakers, never used for any choice): length-matched AUC with a speaker bootstrap
                  interval, EER, raw ECE
  open set        KeSpeech standard Mandarin judged Taiwan-accented at the CV dev EER threshold, per
                  home region, on two speaker halves: "select" (may guide decisions) and "final"
  synthetic       BreezyVoice vs CosyVoice-300M ranking and paired contrast (SD of real-speech scores),
                  on the original pairs and on the larger second set
  voice           Ming, texts × 4 voices: share of score variance explained by the voice

The teacher is reported per seed (mean) and as the 3-seed ensemble.

    python research/scripts/evaluate.py --bundle /content/bench --scores /content/bench/scores --out report_v2.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

from bench_eval import ctrl_auc, ece, eer, eer_threshold, speaker_bootstrap  # noqa: E402
from twjudge.accent.prosodic import auc, length_weights  # noqa: E402

REGIONS = ["none", "Beijing", "Northeastern", "Ji-Lu", "Jiao-Liao", "Zhongyuan", "Lan-Yin", "Jiang-Huai", "Southwestern"]


def half(spk: str) -> str:
    return "select" if int(hashlib.sha256(f"ke-half|{spk}".encode()).hexdigest(), 16) % 2 == 0 else "final"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--stage", choices=["all", "dev", "final"], default="all",
                    help="dev: only what may guide decisions (CV dev and development test set, the development "
                         "synthetic pairs, the voice set, the KeSpeech select half); final: the held-out speakers, "
                         "the 1,000 synthetic pairs, the KeSpeech final half")
    ap.add_argument("--textonly", help="score file of the text-only model: report each system's correlation on cv_dev")
    ap.add_argument("--propensity", help="score file of a text-only model that sees the syllables (mix_train.py "
                                         "textonly_syl): also report the content-controlled AUC (bench_eval.ctrl_auc)")
    a = ap.parse_args()
    B = Path(a.bundle)
    prop = json.load(open(a.propensity)) if a.propensity else None
    raw = {p.stem: json.load(open(p)) for p in sorted(Path(a.scores).glob("*.json"))}
    if a.stage != "all":                        # never compute anything on data the stage must not see
        keep = {"dev": {"cv_dev", "cv_test", "breezy", "cosy300m", "ming_b0", "kespeech"},
                "final": {"cv_dev", "cv_test", "cv_heldout", "synth2_breezy", "synth2_cosy", "ming_b0", "kespeech"}}[a.stage]
        raw = {n: {s: v for s, v in r.items() if s in keep} for n, r in raw.items()}
    halves = {"all": ("select", "final", "all"), "dev": ("select",), "final": ("final",)}[a.stage]
    real_sets = {"all": ("cv_test", "cv_heldout", "r10_sealed"), "dev": ("cv_dev", "cv_test"), "final": ("cv_heldout",)}[a.stage]
    # seed ensembles of the teacher judges, one per encoder (teacher_<b|c>_adv0_s<k> → teacher_<b|c>_ens)
    for enc in sorted({n.split("_")[1] for n in raw if n.startswith("teacher_")}):
        seeds = sorted(n for n in raw if n.startswith(f"teacher_{enc}_"))
        raw[f"teacher_{enc}_ens"] = {s: {i: (None if any(raw[n][s].get(i) is None for n in seeds) else
                                             [float(np.mean([raw[n][s][i][0] for n in seeds])), raw[seeds[0]][s][i][1]])
                                         for i in raw[seeds[0]][s]} for s in raw[seeds[0]]}
    sets = set.intersection(*(set(v) for v in raw.values()))
    rows = {s: [json.loads(l) for l in open(B / "sets" / f"{s}.jsonl")] for s in sets}
    common = {s: [r for r in rows[s] if all(raw[n][s].get(r["item_id"]) is not None for n in raw)] for s in sets}
    print({s: f"{len(common[s])}/{len(rows[s])}" for s in sets}, flush=True)
    z = {n: {s: np.array([raw[n][s][r["item_id"]][0] for r in common[s]]) for s in sets} for n in raw}

    def real(n, s, z_):
        rs = common[s]
        y = np.array([r["item_id"].startswith("tw") for r in rs]).astype(int)
        lens = np.array([len(re.sub(r"[^㐀-鿿]", "", r["text"])) for r in rs])
        lw = length_weights(lens, y)
        spk = np.array([r["speaker"] for r in rs])
        out = {"auc": auc(y, z_, lw), "eer": eer(y, z_), "ece": ece(y, 1 / (1 + np.exp(-z_))), "n": len(rs),
               "n_speakers": {"tw": len(set(spk[y == 1])), "cn": len(set(spk[y == 0]))}}
        if a.boot:
            out["auc_ci95"] = speaker_bootstrap(spk, lambda i: auc(y[i], z_[i], lw[i]), n=a.boot)
        if prop is not None and s in prop:
            p = np.array([prop[s].get(r["item_id"], [np.nan])[0] for r in rs])
            m = ~np.isnan(p)
            out["auc_ctrl"] = ctrl_auc(y[m], z_[m], p[m], lens[m])
            out["n_ctrl"] = int(m.sum())
            if a.boot:
                ym, zm, pm, lm = y[m], z_[m], p[m], lens[m]
                out["auc_ctrl_ci95"] = speaker_bootstrap(spk[m], lambda i: ctrl_auc(ym[i], zm[i], pm[i], lm[i]), n=a.boot)
        return out

    def pairs(z_, sa, sb):
        A = dict(zip([r["item_id"].split("__", 1)[1] for r in common[sa]], z_[sa]))
        Bz = dict(zip([r["item_id"].split("__", 1)[1] for r in common[sb]], z_[sb]))
        d = np.array([A[k] - Bz[k] for k in sorted(set(A) & set(Bz))])
        return d

    report = {}
    for n in raw:
        zn = z[n]
        r = {}
        for s in real_sets:
            if s in sets:
                r[s] = real(n, s, zn[s])
        if a.textonly and "cv_dev" in sets:
            to = json.load(open(a.textonly))["cv_dev"]
            ii = [x["item_id"] for x in common["cv_dev"]]
            r["corr_text_only_dev"] = float(np.corrcoef(zn["cv_dev"], [to[i][0] for i in ii])[0, 1])
        ydv = np.array([x["item_id"].startswith("tw") for x in common["cv_dev"]]).astype(int)
        thr = eer_threshold(ydv, zn["cv_dev"])
        # contrast in SD of real-speech scores, as in the paper (held-out speakers when a run did not score cv_test)
        sd = zn["cv_test"].std() if "cv_test" in zn else zn["cv_heldout"].std()
        for name, sa, sb in (("synthetic", "breezy", "cosy300m"), ("synthetic2", "synth2_breezy", "synth2_cosy")):
            if sa in sets and sb in sets:
                d = pairs(zn, sa, sb)
                r[name] = {"breezy_first": float(np.mean(d > 0)), "contrast_sd": float(d.mean() / sd), "n_pairs": len(d)}
        if "ming_b0" in sets:
            ids = [x["item_id"] for x in common["ming_b0"]]
            voice = np.array([i.split("_")[-1] for i in ids]); zz = zn["ming_b0"]; mu = zz.mean()
            r["voice_share"] = float(sum((voice == v).sum() * (zz[voice == v].mean() - mu) ** 2 for v in set(voice))
                                     / ((zz - mu) ** 2).sum())
        if "kespeech" in sets:
            ks = common["kespeech"]
            ke = {}
            for h in halves:
                for reg in REGIONS:
                    m = np.array([x["speech"] == "mandarin" and x["region"] == reg and (h == "all" or half(x["speaker"]) == h)
                                  for x in ks])
                    if m.any():
                        ke.setdefault(h, {})[reg] = float(np.mean(zn["kespeech"][m] > thr))
                ke[h]["non_southwestern_mean"] = float(np.mean([v for k, v in ke[h].items() if k in REGIONS[:-1]]))
            r["kespeech_judged_taiwan"] = ke
        report[n] = r
    seed_names = [n for n in report if n.startswith("teacher_c")]
    json.dump({"systems": report, "teacher_seeds": seed_names}, open(a.out, "w"), indent=1, ensure_ascii=False)
    for n, r in report.items():
        line = f"{n:28s}"
        for s in ("cv_dev", "cv_test", "cv_heldout"):
            if s in r:
                line += f" {s} AUC {r[s]['auc']:.3f}" + (f" ctrl {r[s]['auc_ctrl']:.3f}" if "auc_ctrl" in r[s] else "") \
                    + f" ECE {r[s]['ece']:.3f}"
        if "corr_text_only_dev" in r:
            line += f" text-r {r['corr_text_only_dev']:.2f}"
        for k in ("synthetic", "synthetic2"):
            if k in r:
                line += f" {k} {100 * r[k]['breezy_first']:.0f}%/{r[k]['contrast_sd']:.2f}"
        if "voice_share" in r:
            line += f" voice {100 * r['voice_share']:.1f}%"
        if "kespeech_judged_taiwan" in r:
            k = r["kespeech_judged_taiwan"]
            line += " KeS " + " ".join(f"{h} non-SW {100 * k[h]['non_southwestern_mean']:.0f}% SW {100 * k[h].get('Southwestern', float('nan')):.0f}%"
                                       for h in k)
        print(line, flush=True)


if __name__ == "__main__":
    main()
