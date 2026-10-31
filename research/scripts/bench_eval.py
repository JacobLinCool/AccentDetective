"""Judge benchmark report: the syllable judges against utterance-level baselines on every check.

Systems
  judges      B (HuBERT) and C (wav2vec 2.0), seeds 0–2, with speaker-adversary weight 0 / 0.3 / 1, on
              the deployed MFA-free CTC alignment (results/runs_ctc) and, as an alignment ablation
              marked [MFA], on the Montreal Forced Aligner alignment (results/runs); the deployed and
              the MFA-trained checkpoints; the text-only bias models
  baselines   linear heads on frozen utterance embeddings (eGeMAPS, ECAPA, Whisper-medium encoder,
              utterance-pooled HuBERT / wav2vec 2.0), trained with the judges' importance weights and
              the regularisation chosen on dev; fine-tuned HuBERT / wav2vec 2.0 utterance classifiers

Every check uses the items that all systems scored (an aligner can reject an item).

Checks (each with a speaker-cluster bootstrap where it applies)
  real speech       CV test: AUC (raw, length-matched), EER, macro recall, speaker-level AUC,
                    calibration (Brier, ECE) before and after Platt scaling fitted on CV dev
  content           correlation with the text-only model
  speaker           share of within-class score variance explained by the speaker (ICC); speaker
                    identification from the judges' utterance representations
  synthetic         BreezyVoice vs CosyVoice-300M, same text and voice prompt: fraction of pairs
                    ranked Taiwan-first and the standardised paired difference
  voice             Ming, 120 texts × 4 voices: share of score variance explained by the voice
  cross-domain      parliament (Taiwan), AISHELL-3 by accent, KeSpeech by region and speech type,
                    CV test Mainland speakers by birthplace: fraction judged Taiwan at the dev EER
                    threshold

    python research/scripts/bench_eval.py --results data/bench/results --out data/bench/report.json
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))

from twjudge.accent.prosodic import auc, length_weights  # noqa: E402

RNG = np.random.default_rng(0)
N_BOOT = 1000
SOUTH = re.compile(r"江苏|浙江|上海|安徽|福建|江西|湖南|湖北|广东|廣東|广西|海南|四川|重庆|贵州|云南|雲南")
SOUTHWEST = re.compile(r"四川|重庆|贵州|云南|雲南")       # Southwestern Mandarin area
NORTH = re.compile(r"北京|天津|河北|山西|内蒙|辽宁|吉林|黑龙江|山东|河南|陕西|甘肃|宁夏|青海|新疆")


# ---------------------------------------------------------------- metrics
def eer(y, z):
    order = np.argsort(-z)
    y = y[order].astype(bool)
    tp = np.cumsum(y) / max(y.sum(), 1)
    fp = np.cumsum(~y) / max((~y).sum(), 1)
    k = np.argmin(np.abs((1 - tp) - fp))
    return float(((1 - tp[k]) + fp[k]) / 2)


def eer_threshold(y, z):
    best, thr = 9, 0.0
    for t in np.quantile(z, np.linspace(0.01, 0.99, 197)):
        fnr, fpr = np.mean(z[y == 1] < t), np.mean(z[y == 0] >= t)
        if abs(fnr - fpr) < best:
            best, thr = abs(fnr - fpr), float(t)
    return thr


def ece(y, p, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def platt(y, z):
    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression(C=1e6).fit(z[:, None], y)
    return lambda x: m.predict_proba(np.asarray(x)[:, None])[:, 1]


def icc_within(z, spk, y):
    """Share of within-class variance of the scores explained by the speaker (one-way ANOVA)."""
    out = []
    for c in (0, 1):
        m = y == c
        zz, ss = z[m], spk[m]
        mu = zz.mean()
        groups = collections.defaultdict(list)
        for v, s in zip(zz, ss):
            groups[s].append(v)
        between = sum(len(g) * (np.mean(g) - mu) ** 2 for g in groups.values())
        total = ((zz - mu) ** 2).sum()
        out.append(between / total if total > 0 else np.nan)
    return float(np.mean(out))


def ctrl_auc(y, z, p, lens, strata=5):
    """Content-controlled AUC: utterances are grouped into quantile strata of a text-only model's log-odds p
    (the propensity of the text alone to be Taiwan's), the length-matched AUC is taken within each stratum,
    and the strata are averaged weighted by their number of Taiwan-Mainland pairs. What the text predicts
    cannot raise it, so it compares judges on what the audio adds."""
    y, z, p, lens = map(np.asarray, (y, z, p, lens))
    edges = np.quantile(p, np.linspace(0, 1, strata + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, strata - 1)
    num = den = 0.0
    for k in range(strata):
        m = b == k
        n1, n0 = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        if n1 == 0 or n0 == 0:
            continue
        w = length_weights(lens[m], y[m])
        if w[y[m] == 1].sum() == 0 or w[y[m] == 0].sum() == 0:
            continue
        num += n1 * n0 * auc(y[m], z[m], w)
        den += n1 * n0
    return float(num / den) if den else float("nan")


def speaker_bootstrap(spk, fn, n=None):
    """95% interval of fn(index array) over speaker-cluster resamples."""
    n = n or N_BOOT
    groups = collections.defaultdict(list)
    for i, s in enumerate(spk):
        groups[s].append(i)
    keys = list(groups)
    vals = []
    for _ in range(n):
        pick = RNG.choice(len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        vals.append(fn(idx))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------- data
class Bench:
    def __init__(self, bundle: Path, birthplace: Path | None):
        self.sets = {p.stem: [json.loads(l) for l in open(p)] for p in sorted((bundle / "sets").glob("*.jsonl"))}
        self.lens = {s: np.load(bundle / "feat" / f"feat_{s}.npz")["lens"] for s in self.sets}
        birth = json.load(open(birthplace)) if birthplace and birthplace.exists() else {}
        for r in self.sets["cv_test"]:
            r["birthplace"] = birth.get(r["speaker"])

    def ids(self, s):
        return [r["item_id"] for r in self.sets[s]]

    def restrict(self, s, keep: set):
        idx = [k for k, r in enumerate(self.sets[s]) if r["item_id"] in keep]
        self.sets[s] = [self.sets[s][k] for k in idx]
        self.lens[s] = self.lens[s][idx]


ABLATION = {"recomputed": "features recomputed (check)", "noctx": "no text context", "nobias": "no text-only model",
            "noweights": "no importance weights", "unnorm": "no per-utterance normalisation",
            "uniform": "uniform segments, no alignment"}


def judge_name(tag: str, mfa: bool) -> str | None:
    suffix = " [MFA]" if mfa else ""
    m = re.match(r"([bc])_adv([\d.]+)_s(\d+)", tag)
    if m:
        adv = float(m.group(2))
        return f"judge {m.group(1).upper()}" + (f" + adversary {adv:g}" if adv else "") + suffix
    if tag.startswith("ref_"):
        j = tag[len("ref_judge_")].upper()
        return f"judge {j} ({'MFA-trained' if mfa else 'deployed'} checkpoint)" + suffix
    if tag == "textonly":
        return "text-only" + suffix
    return None


def load_systems(res: Path, local: Path, bench: Bench):
    """name → list over seeds of {set: np.array of logits in set order}, on the items all systems scored."""
    raw = collections.defaultdict(list)

    def add(name, logits):
        raw[name].append({s: logits[s] for s in bench.sets if s in logits})

    for sub, mfa in (("runs_ctc", False), ("runs", True)):
        for f in sorted((res / sub).glob("*_logits.json")):
            name = judge_name(f.name[:-len("_logits.json")], mfa)
            if name:
                add(name, json.load(open(f)))
    for sub in ("runs_layers", "runs_compact", "runs_student", "runs_scompact", "runs_kd"):   # compact judges
        for f in sorted((res / sub).glob("*/*_logits.json")):
            if re.match(r"[bc]_adv0_s\d+", f.name):
                add(f"{sub}/{f.parent.name}", json.load(open(f)))
    for f in sorted((res / "runs_abl").glob("*/*_logits.json")):          # ablations of the CTC judges
        m = re.match(r"([bc])_adv0_s\d+_?(\w*)", f.name[:-len("_logits.json")])
        if m:
            add(f"judge {m.group(1).upper()} — {ABLATION[f.parent.name]}", json.load(open(f)))
    for f in sorted((res / "ft").glob("ft_*_logits.json")):
        model = f.name.split("_")[1]
        add({"wav2vec2": "fine-tuned wav2vec 2.0", "hubert": "fine-tuned HuBERT"}[model], json.load(open(f)))
    for emb, label in (("egemaps", "eGeMAPS + linear"), ("ecapa", "ECAPA + linear"), ("whisper", "Whisper-medium + linear"),
                       ("hubert_utt", "HuBERT utterance pooling + linear"), ("w2v_utt", "wav2vec 2.0 utterance pooling + linear"),
                       ("hubert_utt:std", "HuBERT utterance std pooling (mean removed) + linear"),
                       ("w2v_utt:std", "wav2vec 2.0 utterance std pooling (mean removed) + linear"),
                       ("sylpool_hubert", "HuBERT syllable-feature pooling + linear"),
                       ("sylpool_w2v", "wav2vec 2.0 syllable-feature pooling + linear")):
        lg = linear_baseline(emb, [res / "emb", res / "emb_abl", local], bench)
        if lg is not None:
            add(label, lg)
    coverage = {n: min(len(run[s]) / len(bench.sets[s]) for run in runs for s in run if s in bench.sets)
                for n, runs in raw.items()}
    for n, c in coverage.items():
        if c < 0.97:                      # e.g. a compact aligner that rejects many items: reported, not scored
            print(f"dropped {n}: scores only {100 * c:.1f}% of some set", flush=True)
            raw.pop(n)
    for s in list(bench.sets):
        have = [run[s] for runs in raw.values() for run in runs if s in run]
        if not have:
            continue
        keep = set.intersection(*(set(h) for h in have))
        bench.restrict(s, keep)
    bench.coverage = coverage
    systems = {n: [{s: np.array([run[s][i] for i in bench.ids(s)]) for s in run} for run in runs] for n, runs in raw.items()}
    # seed ensembles of the compact judges: the mean log-odds of their seeds, scored as one system
    for n in [n for n in systems if n.startswith(("runs_scompact/", "runs_student/", "runs_kd/")) and len(systems[n]) > 1]:
        runs = systems[n]
        common = set.intersection(*(set(r) for r in runs))
        systems[f"ensemble {n}"] = [{s: np.mean([r[s] for r in runs], axis=0) for s in common}]
    return systems


def linear_baseline(emb, dirs, bench: Bench):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from twjudge.accent.neural import utterance_weights

    emb, _, part = emb.partition(":")

    def find(s):
        for d in dirs:
            p = d / f"{emb}_{s}.npz"
            if p.exists():
                return p
        return None

    if not find("cv_train") or not find("cv_dev"):
        return None
    E = {}
    for s in bench.sets:
        p = find(s)
        if p is not None:
            d = np.load(p)
            X = np.nan_to_num(d["E"].astype(np.float32))
            if part == "std":                         # [mean, std] → std only: the mean is 0 after mean removal
                X = X[:, X.shape[1] // 2:]
            pos = {str(i): k for k, i in enumerate(d["ids"])}
            if all(i in pos for i in bench.ids(s)):
                E[s] = X[[pos[i] for i in bench.ids(s)]]
            else:                                     # an aligner rejected items: score what is there
                keep = [i for i in bench.ids(s) if i in pos]
                E[s] = (keep, X[[pos[i] for i in keep]])
    f = np.load(ROOT / "data/bench/bundle/feat/feat_cv_train.npz")
    yall = np.array([i.startswith("tw") for i in bench.ids("cv_train")], np.float32)
    wall = utterance_weights({"y": yall, "lens": f["lens"], "X": f["X"]})
    ids_of = {s: (X[0] if isinstance(X, tuple) else bench.ids(s)) for s, X in E.items()}
    E = {s: (X[1] if isinstance(X, tuple) else X) for s, X in E.items()}
    keep = {i for i in ids_of["cv_train"]}
    sel = [k for k, i in enumerate(bench.ids("cv_train")) if i in keep]
    ytr, w = yall[sel], wall[sel]
    pos_dv = {i: k for k, i in enumerate(bench.ids("cv_dev"))}
    ydv = np.array([i.startswith("tw") for i in ids_of["cv_dev"]])
    lens_dv = bench.lens["cv_dev"][[pos_dv[i] for i in ids_of["cv_dev"]]]
    sc = StandardScaler().fit(E["cv_train"])
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(E["cv_train"]), ytr, sample_weight=w)
        a = auc(ydv, m.decision_function(sc.transform(E["cv_dev"])), length_weights(lens_dv, ydv.astype(int)))
        if best is None or a > best[0]:
            best = (a, m)
    m = best[1]
    return {s: dict(zip(ids_of[s], m.decision_function(sc.transform(X)))) for s, X in E.items()}


# ---------------------------------------------------------------- checks
def evaluate(name, runs, bench: Bench, textonly):
    test = bench.sets["cv_test"]
    y = np.array([r["item_id"].startswith("tw") for r in test]).astype(int)
    spk = np.array([r["speaker"] for r in test])
    lw = length_weights(bench.lens["cv_test"], y)
    ydv = np.array([i.startswith("tw") for i in bench.ids("cv_dev")]).astype(int)
    per_seed = []
    for run in runs:
        z = run["cv_test"]
        cal = platt(ydv, run["cv_dev"])
        thr = eer_threshold(ydv, run["cv_dev"])
        p_raw, p_cal = 1 / (1 + np.exp(-z)), cal(z)
        spk_mean = {s: z[spk == s].mean() for s in set(spk)}
        spk_y = {s: y[spk == s][0] for s in set(spk)}
        sd = z.std()
        m = {"auc_raw": auc(y, z), "auc_length_matched": auc(y, z, lw), "eer": eer(y, z),
             "macro_recall_at_0": float((np.mean(z[y == 1] > 0) + np.mean(z[y == 0] <= 0)) / 2),
             "macro_recall_at_dev_eer": float((np.mean(z[y == 1] > thr) + np.mean(z[y == 0] <= thr)) / 2),
             "speaker_auc": auc(np.array([spk_y[s] for s in spk_mean]), np.array(list(spk_mean.values()))),
             "brier_raw": float(np.mean((p_raw - y) ** 2)), "ece_raw": ece(y, p_raw),
             "brier_platt": float(np.mean((p_cal - y) ** 2)), "ece_platt": ece(y, p_cal),
             "speaker_icc": icc_within(z, spk, y)}
        if textonly is not None:
            m["corr_text_only"] = float(np.corrcoef(z, textonly)[0, 1])
        if "breezy" in run and "cosy300m" in run:
            A = dict(zip([i.split("__")[-1] for i in bench.ids("breezy")], run["breezy"]))
            Bc = dict(zip([i.split("__")[-1] for i in bench.ids("cosy300m")], run["cosy300m"]))
            common = sorted(set(A) & set(Bc))
            d = np.array([A[k] - Bc[k] for k in common])
            m["synthetic_frac_breezy_first"] = float(np.mean(d > 0))
            m["synthetic_paired_diff_sd"] = float(d.mean() / sd)
        if "ming_b0" in run:
            ids = bench.ids("ming_b0")
            text = np.array([i.split("_")[-2] for i in ids]); voice = np.array([i.split("_")[-1] for i in ids])
            zz = run["ming_b0"]; mu = zz.mean(); tot = ((zz - mu) ** 2).sum()
            m["voice_share"] = float(sum((voice == v).sum() * (zz[voice == v].mean() - mu) ** 2 for v in set(voice)) / tot)
            m["text_share"] = float(sum((text == t).sum() * (zz[text == t].mean() - mu) ** 2 for t in set(text)) / tot)
        dom = {}
        for s in ("parliament", "aishell3", "kespeech", "ming_b0", "breezy", "cosy300m"):
            if s in run:
                dom[s] = float(np.mean(run[s] > thr))
        if "aishell3" in run:
            acc = np.array([r.get("accent") for r in bench.sets["aishell3"]])
            for a_ in ("north", "south", "others"):
                if (acc == a_).any():
                    dom[f"aishell3/{a_}"] = float(np.mean(run["aishell3"][acc == a_] > thr))
        if "kespeech" in run:
            ks = bench.sets["kespeech"]
            for key in sorted({(r["speech"], r["region"]) for r in ks}):
                mm = np.array([(r["speech"], r["region"]) == key for r in ks])
                dom[f"kespeech/{key[0]}/{key[1]}"] = float(np.mean(run["kespeech"][mm] > thr))
        cn = np.array([r["item_id"].startswith("cn") for r in test])
        bp = np.array([r.get("birthplace") or "" for r in test])
        for region, rx in (("south", SOUTH), ("southwest", SOUTHWEST), ("north", NORTH)):
            mm = cn & np.array([bool(rx.search(b)) for b in bp])
            if mm.any():
                dom[f"cv_test_cn/{region}"] = float(np.mean(z[mm] > thr))
                dom[f"cv_test_cn/{region}_n_speakers"] = int(len(set(spk[mm])))
        m["frac_judged_taiwan"] = dom
        per_seed.append(m)

    def agg(key):
        v = [m[key] for m in per_seed if key in m]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n_seeds": len(v)} if v else None

    keys = [k for k in per_seed[0] if k != "frac_judged_taiwan"]
    out = {k: agg(k) for k in keys}
    out["frac_judged_taiwan"] = {k: float(np.mean([m["frac_judged_taiwan"][k] for m in per_seed]))
                                 for k in per_seed[0]["frac_judged_taiwan"]}
    zs = [run["cv_test"] for run in runs]
    if not N_BOOT:
        return out
    out["auc_length_matched"]["ci95"] = speaker_bootstrap(
        spk, lambda idx: np.mean([auc(y[idx], z[idx], lw[idx]) for z in zs]))
    out["eer"]["ci95"] = speaker_bootstrap(spk, lambda idx: np.mean([eer(y[idx], z[idx]) for z in zs]), n=300)
    return out


def speaker_probe(res: Path, bench: Bench):
    """Speaker identification from the judges' utterance representations, within each class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    test = bench.sets["cv_test"]
    pos = {r["item_id"]: k for k, r in enumerate(test)}
    spk = np.array([r["speaker"] for r in test])
    lab = np.array([r["item_id"][:2] for r in test])
    out = collections.defaultdict(list)
    for sub, mfa in (("runs_ctc", False), ("runs", True)):
      for f in sorted((res / sub).glob("*_states_cv_test.npy")):
        tag = f.name[:-len("_states_cv_test.npy")]
        name = judge_name(tag, mfa)
        if not name or "checkpoint" in name:
            continue
        order = list(json.load(open(res / sub / f"{tag}_logits.json"))["cv_test"])
        Hall = np.load(f).astype(np.float32)
        H = np.zeros((len(test), Hall.shape[1]), np.float32)
        for k, i in enumerate(order):
            if i in pos:
                H[pos[i]] = Hall[k]
        accs = {}
        for c in ("tw", "cn"):
            sel = np.where(lab == c)[0]
            cnt = collections.Counter(spk[sel])
            keep = [i for i in sel if cnt[spk[i]] >= 4]
            seen, tr, te = collections.Counter(), [], []
            for i in keep:
                (tr if seen[spk[i]] % 2 == 0 else te).append(i); seen[spk[i]] += 1
            sc = StandardScaler().fit(H[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(H[tr]), spk[tr])
            accs[c] = float(np.mean(clf.predict(sc.transform(H[te])) == spk[te]))
            accs[f"{c}_chance"] = 1 / len({spk[i] for i in keep})
        out[name].append(accs)
    return {k: {kk: float(np.mean([v[kk] for v in vs])) for kk in vs[0]} for k, vs in out.items()}


def main() -> None:
    global N_BOOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="data/bench/results")
    ap.add_argument("--local", default="data/bench/emb_local")
    ap.add_argument("--bundle", default="data/bench/bundle")
    ap.add_argument("--birthplace", default="data/bench/cv_cn_birthplace.json")
    ap.add_argument("--out", default="data/bench/report.json")
    ap.add_argument("--only", help="regex: evaluate only matching systems (quick looks)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (0: no intervals)")
    a = ap.parse_args()
    N_BOOT = a.boot
    bench = Bench(Path(a.bundle), Path(a.birthplace))
    systems = load_systems(Path(a.results), Path(a.local), bench)
    to = {k: systems[k][0]["cv_test"] for k in ("text-only", "text-only [MFA]") if k in systems}
    print({s: len(bench.sets[s]) for s in bench.sets}, flush=True)
    report = {"systems": {}, "speaker_probe": speaker_probe(Path(a.results), bench), "coverage": bench.coverage,
              "n_items": {s: len(v) for s, v in bench.sets.items()}}
    for name, runs in systems.items():
        ref = to.get("text-only [MFA]" if name.endswith("[MFA]") else "text-only", next(iter(to.values()), None))
        report["systems"][name] = evaluate(name, runs, bench, None if name.startswith("text-only") else ref)
        r = report["systems"][name]
        print(f"{name:42s} AUC {r['auc_length_matched']['mean']:.3f} EER {r['eer']['mean']:.3f} "
              f"synth {r.get('synthetic_frac_breezy_first', {}) and r['synthetic_frac_breezy_first']['mean']:.2f} "
              f"voice {r.get('voice_share', {}) and r['voice_share']['mean']:.2f}", flush=True)
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
