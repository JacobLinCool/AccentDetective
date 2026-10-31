"""The sealed-set numbers behind the paper and the model card: every system on the original and
the rater-corrected labels.

Uses only the sealed new-speaker set (r10_sealed) and the synthetic pairs, never the held-out set. Label sets follow
the blind listening check (dataset/listening):

  original   the Common Voice locale of every clip
  round 1    the rater's four-way answer on 112 clips: Taiwan stays Taiwan, Mainland or other accent becomes
             not-Taiwan, "cannot tell" removes the clip
  round 2    (pre-registered primary) the rater's answer on all 288 pooled clips: no speech or another language
             removes the clip, leaning Taiwan / Mainland sets the label, "cannot tell" keeps the original label
  round 2s   round 2, but "cannot tell" also removes the clip (pre-registered sensitivity analysis)

Every clip the rater did not hear keeps its original label. All systems are compared on the clips every one scored.
With --out, every number is also written as JSON (paper/data/sealed.json).

    PYTHONPATH=research/src uv run --project research python research/scripts/sealed_eval.py --out paper/data/sealed.json
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
from bench_eval import ece  # noqa: E402
from twjudge.accent.prosodic import auc, length_weights  # noqa: E402

D = ROOT / "data/bench/paper"
RELEASED = ROOT / "data/bench/released/scores"           # per-condition scores of the released sizes (robustness.py)
PROBES = ROOT / "data/bench/probes/scores"               # encoder_probe.py
DOCS = ROOT / "dataset/listening"
TAIWANNESS = {"tw": 1.0, "mid": 0.0, "cn": -1.0}
SCHEMES = ("original", "round 1", "round 2", "round 2s")
CONDS = ("mp3", "tel", "noise", "reverb")
PAIRS = (("hubert_large_asp", "hubert_base_asp"), ("w2v_large_asp", "w2v_base_asp"), ("detective_small", "detective_nano"),
         ("detective_tiny", "detective_nano"), ("detective_small", "judge_hubert"), ("whisper", "detective_small"),
         ("readout_asp_audio", "readout_syllable_text"), ("readout_asp_audio", "readout_syllable_audio"),
         ("readout_asp_text", "readout_asp_audio"), ("detective_small_text", "detective_small"))


def load(p):
    return json.load(open(p))


def answers(p):
    d = load(p)
    return d.get("data", d)["answers"]


def ensemble(prefix):
    """Mean of the three seeds of a syllable judge, on every set the seeds share."""
    ss = [load(D / "scores" / f"{prefix}_s{k}.json") for k in range(3)]
    return {s: {i: [float(np.mean([x[s][i][0] for x in ss]))] for i in ss[0][s] if all(x[s].get(i) is not None for x in ss)}
            for s in ss[0]}


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x, kind="stable")).astype(float) for x in (a, b))
    for r, x in ((ra, a), (rb, b)):                                              # average ranks for ties
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def wilson(k, n):
    p, z = k / n, 1.96
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def tts_ranking(S, k=10, draws=5000):
    """How often the mean of k random BreezyVoice-CosyVoice pairs ranks BreezyVoice first: one TTS comparison."""
    b, c = S.get("synth2_breezy"), S.get("synth2_cosy")
    if not b or not c:
        return None
    diff = np.array([b[i][0] - c["synth2_cosy" + i[len("synth2_breezy"):]][0] for i in b
                     if b[i] is not None and c.get("synth2_cosy" + i[len("synth2_breezy"):]) is not None])
    pick = np.random.default_rng(0).integers(0, len(diff), (draws, k))
    return float((diff[pick].mean(1) > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="also write every number as JSON")
    a = ap.parse_args()
    out = {"schemes": SCHEMES}
    rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(D / "sets/r10_sealed.jsonl")}
    key = {k["clip"]: k for k in load(DOCS / "rating_key.json")}
    r1 = {key[c]["item_id"]: v for c, v in answers(DOCS / "ratings_pilot.json").items()}
    r2 = {key[c]["item_id"]: v for c, v in answers(DOCS / "ratings_check.json").items()}
    pool = {k["item_id"]: k["pool"] for k in key.values()}

    # ---- the ratings themselves
    print(f"round 2: {len(r2)} clips rated", dict(collections.Counter(r2.values())))
    print("\nround 2 answers by pool and original label (tw / mid / cn / no speech / other language)")
    out["ratings"] = {}
    for p in ("core", "other", "control"):
        for lab in ("tw", "cn"):
            ids = [i for i in r2 if pool[i] == p and i.startswith(lab)]
            c = collections.Counter(r2[i] for i in ids)
            out["ratings"][f"{p}_{lab}"] = {v: c[v] for v in ("tw", "mid", "cn", "silent", "lang")}
            print(f"  {p:7s} labelled {lab}: {len(ids):3d} -> " + " ".join(f"{c[v]:3d}" for v in ("tw", "mid", "cn", "silent", "lang")))
    ctrl = [i for i in r2 if pool[i] == "control"]
    flips = sum(r2[i] in ("tw", "cn") and (r2[i] == "tw") != i.startswith("tw") for i in ctrl)
    excl = sum(r2[i] in ("silent", "lang") for i in ctrl)
    for name, k in (("relabelled", flips), ("excluded", excl), ("cannot tell", sum(r2[i] == "mid" for i in ctrl))):
        p, lo, hi = wilson(k, len(ctrl))
        print(f"  control {name}: {k}/{len(ctrl)} = {p:.1%} [{lo:.1%}, {hi:.1%}]")

    both = [i for i in r1 if i in r2]
    print(f"\ntest-retest on the {len(both)} clips rated in both rounds (rows: round 1, columns: round 2 tw mid cn silent lang)")
    for v1 in ("tw", "cn", "other", "unsure"):
        c = collections.Counter(r2[i] for i in both if r1[i] == v1)
        print(f"  {v1:7s} " + " ".join(f"{c[v]:3d}" for v in ("tw", "mid", "cn", "silent", "lang")))
    dec = [i for i in both if r1[i] in ("tw", "cn") and r2[i] in ("tw", "cn")]
    out["retest"] = {"both_decided": len(dec), "same": sum(r1[i] == r2[i] for i in dec)}
    print(f"  Taiwan / Mainland in both rounds: {len(dec)} clips, same answer {out['retest']['same']}")

    # ---- label sets on the whole sealed set
    def label(i, scheme):
        orig = i.startswith("tw")
        if scheme == "original":
            return orig
        if scheme == "round 1":
            v = r1.get(i)
            return None if v == "unsure" else orig if v is None else v == "tw"
        v = r2.get(i)
        if v in ("silent", "lang") or (v == "mid" and scheme == "round 2s"):
            return None
        return v == "tw" if v in ("tw", "cn") else orig

    systems = {n: load(D / "scores" / f"{n}.json") for n in ("egemaps", "ecapa", "whisper", "pooled_hubert", "pooled_w2v",
                                                             "ft_hubert", "ft_w2v")}
    systems["judge_hubert"] = ensemble("teacher_b_adv0")
    systems["judge_w2v"] = ensemble("teacher_c_adv0")
    for k in ("nano", "tiny", "mini", "small"):
        systems[f"detective_{k}"] = load(D / "scores" / f"detective_{k}.json")
        systems[f"detective_{k}_text"] = load(D / "scores_readouts" / f"detective_{k}_text.json")
    for n in ("hubert_base", "hubert_large", "w2v_base", "w2v_large"):
        systems[f"{n}_asp"] = load(PROBES / f"{n}.json")
    for n in ("asp_audio", "asp_text", "syllable_audio", "syllable_text"):
        systems[f"readout_{n}"] = load(D / "scores_readouts" / f"{n}.json")
    sealed = {n: S.get("r10_sealed", {}) for n, S in systems.items()}
    scored = [i for i in rows if all(S.get(i) is not None for S in sealed.values())]
    Z = {n: np.array([S[i][0] for i in scored]) for n, S in sealed.items()}
    lens = np.array([len(re.sub(r"[^㐀-鿿]", "", rows[i]["text"])) for i in scored])
    spk = np.array([rows[i]["speaker"] for i in scored])
    members = {u: np.where(spk == u)[0] for u in np.unique(spk)}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[u] for u in rng.choice(list(members), len(members))]) for _ in range(500)]
    Y, keep = {}, {}
    orig = np.array([i.startswith("tw") for i in scored]).astype(int)
    print(f"\nsealed clips scored by every system: {len(scored)}")
    out["clips"] = {}
    for s in SCHEMES:
        lab = [label(i, s) for i in scored]
        keep[s] = np.array([v is not None for v in lab])
        Y[s] = np.array([bool(v) for v in lab]).astype(int)
        out["clips"][s] = {"n": int(keep[s].sum()), "relabelled": int(((Y[s] != orig) & keep[s]).sum()),
                           "removed": int((~keep[s]).sum()), "taiwan": int(Y[s][keep[s]].sum()),
                           "speakers": len(set(spk[keep[s]]))}
        print(f"  {s:9s} " + ", ".join(f"{k} {v}" for k, v in out["clips"][s].items()))

    def lm_auc(y, z, idx):
        return auc(y[idx], z[idx], length_weights(lens[idx], y[idx]))

    def stat(f, s):
        idx0 = np.where(keep[s])[0]
        bs = [f(b[keep[s][b]]) for b in boots]
        return [float(f(idx0)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    print("\nlength-matched AUC: " + " | ".join(SCHEMES) + " [95% CI, round 2] | ECE original / round 2 | 10-pair TTS comparison")
    out["systems"] = {}
    for n in Z:
        r = {"auc": {s: stat(lambda idx: lm_auc(Y[s], Z[n], idx), s) for s in SCHEMES},
             "ece": {s: ece(Y[s][keep[s]], 1 / (1 + np.exp(-Z[n][keep[s]]))) for s in SCHEMES},
             "tts10": tts_ranking(systems[n])}
        out["systems"][n] = r
        t = f"{r['tts10']:.3f}" if r["tts10"] is not None else "  -  "
        print(f"  {n:24s} " + " | ".join(f"{r['auc'][s][0]:.3f}" for s in SCHEMES) +
              f"  [{r['auc']['round 2'][1]:.3f}, {r['auc']['round 2'][2]:.3f}]  ECE {r['ece']['original']:.3f} / "
              f"{r['ece']['round 2']:.3f}  10 pairs: {t}")

    print("\npaired differences: " + " | ".join(SCHEMES))
    out["paired"] = {}
    for x, y in PAIRS:
        out["paired"][f"{x} - {y}"] = {s: stat(lambda idx: lm_auc(Y[s], Z[x], idx) - lm_auc(Y[s], Z[y], idx), s) for s in SCHEMES}
        print(f"  {x} - {y}: " + " | ".join("{:+.3f} [{:+.3f}, {:+.3f}]".format(*out["paired"][f"{x} - {y}"][s]) for s in SCHEMES))

    # ---- recording conditions (channels.py) on the sealed set, released sizes, audio only
    print("\nrecording conditions, change in length-matched AUC from clean audio: original | round 2")
    out["robust"] = {}
    for cell, size in (("N1a", "nano"), ("N2a", "tiny"), ("N3a", "mini"), ("N4a", "small")):
        S = load(RELEASED / f"{cell}.json")
        ids = [i for i in rows if all(S.get(f"r10_sealed{c}", {}).get(i) is not None for c in ("", *[f"@{c}" for c in CONDS]))]
        L = np.array([len(re.sub(r"[^㐀-鿿]", "", rows[i]["text"])) for i in ids])
        out["robust"][size] = {}
        for s in ("original", "round 2"):
            lab = [label(i, s) for i in ids]
            k = np.array([v is not None for v in lab])
            y = np.array([bool(v) for v in lab]).astype(int)[k]
            w = length_weights(L[k], y)
            base = auc(y, np.array([S["r10_sealed"][i][0] for i in ids])[k], w)
            out["robust"][size][s] = {"clean": float(base), **{c: float(auc(y, np.array([S[f"r10_sealed@{c}"][i][0] for i in ids])[k], w) - base)
                                                              for c in CONDS}}
        print(f"  {size:6s} " + "  ".join(f"{c} {out['robust'][size]['original'][c]:+.3f} | {out['robust'][size]['round 2'][c]:+.3f}" for c in CONDS))

    # ---- does a clip that sounds ambiguous get an ambiguous score? (descriptive)
    pos = {i: k for k, i in enumerate(scored)}
    heard = [i for i in r2 if r2[i] in TAIWANNESS and i in pos]
    t = np.array([TAIWANNESS[r2[i]] for i in heard])
    print(f"\nSpearman, score vs heard Taiwan-ness (lean Taiwan 1, cannot tell 0, lean Mainland -1), {len(heard)} Mandarin clips")
    out["spearman"] = {}
    for n in Z:
        z = Z[n][[pos[i] for i in heard]]
        out["spearman"][n] = spearman(z, t)
        print(f"  {n:24s} rho {out['spearman'][n]:+.2f}   score median: lean Taiwan {np.median(z[t == 1]):+.2f}, "
              f"cannot tell {np.median(z[t == 0]):+.2f}, lean Mainland {np.median(z[t == -1]):+.2f}")

    # ---- exploratory, not pre-registered: what if the listener disagreed with the labels of every clip the selector
    # systems all get right as often as on the control pool? Each unheard all-right clip draws an answer from the control
    # answers of its label, then the round 2 rule applies.
    pools = load(DOCS / "pools.json")
    right = [i for i in scored if i not in r2
             and not any((sealed[n][i][0] > pools["thresholds"][n]) != i.startswith("tw") for n in pools["systems"])]
    ctrl_answers = {lab: [r2[i] for i in ctrl if i.startswith(lab)] for lab in ("tw", "cn")}
    print(f"\nexploratory: {len(right)} unheard clips every selector system gets right, answers drawn from the control pool "
          + str({lab: dict(collections.Counter(v)) for lab, v in ctrl_answers.items()}))
    sim_rng = np.random.default_rng(1)
    show = ("whisper", "judge_hubert", "detective_nano", "detective_small", "hubert_large_asp")
    res = {n: [] for n in show}
    pos_right = [pos[i] for i in right]
    for _ in range(100):
        y, k = Y["round 2"].copy(), keep["round 2"].copy()
        for p, i in zip(pos_right, right):
            v = ctrl_answers[i[:2]][sim_rng.integers(len(ctrl_answers[i[:2]]))]
            if v in ("silent", "lang"):
                k[p] = False
            elif v in ("tw", "cn"):
                y[p] = int(v == "tw")
        idx = np.where(k)[0]
        for n in show:
            res[n].append(lm_auc(y, Z[n], idx))
    out["extrapolated"] = {n: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for n, v in res.items()}
    print("  AUC against the simulated listener (mean of 100 draws, 2.5-97.5 percentile): "
          + ", ".join(f"{n} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for n, v in out["extrapolated"].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
