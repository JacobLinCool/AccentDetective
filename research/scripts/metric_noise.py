"""Sampling noise of every benchmark metric, so that differences between systems are read against it.

Each metric is resampled over its own independent unit (cluster bootstrap):
  real-speech AUC, controlled AUC   speakers of cv_test / cv_heldout
  synthetic ranking and contrast    sentences (238 development pairs: 119 texts × 2 prompts; 1,000 held-out pairs:
                                    250 texts × 4 prompts)
  voice share                       the 119 texts of the voice set (4 voices each)
  KeSpeech judged Taiwan            KeSpeech speakers
For one system it reports the 95% interval; for pairs of systems scored on the same items, the 95% interval of
their difference over the same resamples (what decides whether two systems differ). The smallest difference
that is detectable is about the half-width of the paired interval.

    python research/scripts/metric_noise.py --bundle data/bench/bundle --scores <dir> \\
        --propensity data/bench/s1/textonly/textonly_syl.json --pairs A:B C:D --out noise.json
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

from bench_eval import ctrl_auc, eer_threshold  # noqa: E402
from twjudge.accent.prosodic import auc, length_weights  # noqa: E402

RNG = np.random.default_rng(0)


def clusters(keys):
    g = collections.defaultdict(list)
    for k, c in enumerate(keys):
        g[c].append(k)
    return list(g.values())


def resample(groups):
    pick = RNG.integers(0, len(groups), len(groups))
    return np.concatenate([groups[p] for p in pick])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--propensity", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B system names to compare")
    ap.add_argument("--systems", nargs="*", default=None, help="systems to report alone (default: those in --pairs)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = Path(a.bundle)
    pairs = [p.split(":") for p in a.pairs]
    names = a.systems or sorted({n for p in pairs for n in p})
    raw = {n: json.load(open(Path(a.scores) / f"{n}.json")) for n in names}
    prop = json.load(open(a.propensity))
    rows = {s: [json.loads(l) for l in open(B / "sets" / f"{s}.jsonl")] for s in
            ("cv_dev", "cv_test", "cv_heldout", "breezy", "cosy300m", "synth2_breezy", "synth2_cosy", "ming_b0", "kespeech")}

    def ok(s, i):
        return all(raw[n].get(s, {}).get(i) is not None for n in names)

    common = {s: [r for r in rs if ok(s, r["item_id"])] for s, rs in rows.items()}
    z = {n: {s: np.array([raw[n][s][r["item_id"]][0] for r in common[s]]) for s in common} for n in names}

    # ---- metric definitions on index arrays (identical to evaluate.py) ----
    def real(s):
        rs = common[s]
        y = np.array([r["item_id"].startswith("tw") for r in rs]).astype(int)
        lens = np.array([len(re.sub(r"[^㐀-鿿]", "", r["text"])) for r in rs])
        p = np.array([prop[s][r["item_id"]][0] for r in rs])
        groups = clusters([r["speaker"] for r in rs])
        f_auc = lambda zz, i: auc(y[i], zz[i], length_weights(lens[i], y[i]))
        f_ctrl = lambda zz, i: ctrl_auc(y[i], zz[i], p[i], lens[i])
        return groups, {"auc": f_auc, "ctrl_auc": f_ctrl}

    def synth(sa, sb, sd_set):
        A = {r["item_id"].split("__", 1)[1]: k for k, r in enumerate(common[sa])}
        Bk = {r["item_id"].split("__", 1)[1]: k for k, r in enumerate(common[sb])}
        keys = sorted(set(A) & set(Bk))
        tid = [common[sa][A[k]]["text_id"] for k in keys]
        ia, ib = np.array([A[k] for k in keys]), np.array([Bk[k] for k in keys])
        groups = clusters(tid)

        def diffs(n):
            return z[n][sa][ia] - z[n][sb][ib]
        return groups, diffs

    out = {}
    jobs = {}
    g_test, f_test = real("cv_test")
    g_held, f_held = real("cv_heldout")
    for m, f in f_test.items():
        jobs[f"dev_{m}"] = (g_test, (lambda f: lambda n, i: f(z[n]["cv_test"], i))(f))
    for m, f in f_held.items():
        jobs[f"heldout_{m}"] = (g_held, (lambda f: lambda n, i: f(z[n]["cv_heldout"], i))(f))
    for tag, sa, sb, sd_set in (("dev_synth", "breezy", "cosy300m", "cv_test"),
                                ("heldout_synth", "synth2_breezy", "synth2_cosy", "cv_test")):
        groups, diffs = synth(sa, sb, sd_set)
        jobs[f"{tag}_rank"] = (groups, (lambda d: lambda n, i: float(np.mean(d(n)[i] > 0)))(diffs))
        jobs[f"{tag}_contrast"] = (groups, (lambda d: lambda n, i: float(d(n)[i].mean() / z[n]["cv_test"].std()))(diffs))
    ming = common["ming_b0"]
    m_text = [r["text_id"] for r in ming]
    m_voice = np.array([r["item_id"].rsplit("_", 1)[1] for r in ming])

    def voice_share(n, i):
        zz, vv = z[n]["ming_b0"][i], m_voice[i]
        mu = zz.mean()
        return float(sum((vv == v).sum() * (zz[vv == v].mean() - mu) ** 2 for v in set(vv)) / ((zz - mu) ** 2).sum())
    jobs["voice_share"] = (clusters(m_text), voice_share)
    ks = common["kespeech"]
    ydv = np.array([r["item_id"].startswith("tw") for r in common["cv_dev"]]).astype(int)
    thr = {n: eer_threshold(ydv, z[n]["cv_dev"]) for n in names}
    std_only = np.array([r["speech"] == "mandarin" and r["region"] == "none" for r in ks])
    sw = np.array([r["speech"] == "mandarin" and r["region"] == "Southwestern" for r in ks])
    for tag, mask in (("kespeech_std", std_only), ("kespeech_sw", sw)):
        idx = np.where(mask)[0]
        groups = clusters([ks[k]["speaker"] for k in idx])
        jobs[tag] = ([idx[g] for g in groups], (lambda n, i: float(np.mean(z[n]["kespeech"][i] > thr[n]))))

    for metric, (groups, f) in jobs.items():
        res = {"point": {n: f(n, np.concatenate(groups)) for n in names}, "units": len(groups)}
        boots = collections.defaultdict(list)
        for _ in range(a.boot):
            i = resample(groups)
            vals = {n: f(n, i) for n in names}
            for n in names:
                boots[n].append(vals[n])
            for p, q in pairs:
                boots[f"{p}-{q}"].append(vals[p] - vals[q])
        res["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        res["paired_halfwidth"] = {f"{p}-{q}": float((res["ci95"][f"{p}-{q}"][1] - res["ci95"][f"{p}-{q}"][0]) / 2)
                                   for p, q in pairs}
        out[metric] = res
        print(f"{metric:24s} units {len(groups):5d}  " + "  ".join(
            f"{n}: {res['point'][n]:.3f} ±{(res['ci95'][n][1] - res['ci95'][n][0]) / 2:.3f}" for n in names[:3]) +
            "  | paired ±" + " ".join(f"{v:.3f}" for v in res["paired_halfwidth"].values()), flush=True)

    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
