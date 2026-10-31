"""Figures and LaTeX tables of the Accent Detective paper, from the benchmark outputs. Real speech is the sealed set only.

Inputs (paper/data/):
  sealed.json           research/scripts/sealed_eval.py: every system on the sealed speakers with the published and the
                        listener-checked labels, the listening check, paired differences, recording conditions
  report_main.json      evaluate.py on collect_scores.py's scores: the 1,000 synthetic pairs, the voice set and KeSpeech
  report_readouts.json  the same evaluation of the readouts of one trunk
  report_probes.json    the same evaluation of the frozen-encoder probes (HuBERT and wav2vec 2.0, base and large)
  teacher_ensembles.json, report_dev_ablations.json   the development benchmark (ablations)
  latency_nested.json, latency_ref_*.json   latency_nested.py on one Colab runtime

    PYTHONPATH=research/src python paper/scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
REPORT = json.load(open(PAPER / "data/report_dev_ablations.json"))["systems"]
ENS = json.load(open(PAPER / "data/teacher_ensembles.json"))
V2 = json.load(open(PAPER / "data/report_main.json"))["systems"]
RD = json.load(open(PAPER / "data/report_readouts.json"))["systems"]
PROBE_REPORT = json.load(open(PAPER / "data/report_probes.json"))["systems"]
SEALED = json.load(open(PAPER / "data/sealed.json"))
CLEAN = json.load(open(ROOT / "dataset/screening_summary_p0.5.json"))       # screen_analyze.py, release thresholds
SIZES = [("detective_nano", "Nano"), ("detective_tiny", "Tiny"), ("detective_mini", "Mini"), ("detective_small", "Small")]
CHECKED, PUBLISHED = "round 2", "original"

C_OURS, C_TEACH, C_BASE, C_PROBE, C_GREY = "#D55E00", "#0072B2", "#7F7F7F", "#009E73", "#BDBDBD"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.35                                                           # Interspeech column width, inches

# every system: key in sealed.json -> (label, family, validity record)
PROBES = {f"{e}_{s}_asp": (f"{'HuBERT' if e == 'hubert' else 'w2v'} {s}", PROBE_REPORT[f"{e}_{s}"])
          for e in ("hubert", "w2v") for s in ("base", "large")}
SYSTEMS = {
    "egemaps": ("eGeMAPS", "base", V2["egemaps"]), "ecapa": ("ECAPA", "base", V2["ecapa"]),
    "whisper": ("Whisper-med.", "base", V2["whisper"]),
    "pooled_hubert": ("Pooled HuBERT", "base", V2["pooled_hubert"]), "pooled_w2v": ("Pooled w2v", "base", V2["pooled_w2v"]),
    "ft_hubert": ("FT HuBERT", "base", V2["ft_hubert"]), "ft_w2v": ("FT w2v", "base", V2["ft_w2v"]),
    **{k: (lab + " + ASP", "probe", v) for k, (lab, v) in PROBES.items()},
    "judge_hubert": ("Syllable judge, HuBERT", "judge", V2["teacher_b_ens"]),
    "judge_w2v": ("Syllable judge, w2v", "judge", V2["teacher_c_ens"]),
    **{k: (f"Detective {lab}", "ours", V2[k]) for k, lab in SIZES},
}
COLOR = {"base": C_BASE, "probe": C_PROBE, "judge": C_TEACH, "ours": C_OURS}

# parameters of each whole system, in millions (frozen encoders and aligners included; a seed ensemble of judges
# shares its aligner and encoder): eGeMAPS has no learned front end, ECAPA-TDNN 20.77 M, Whisper-medium encoder
# 307.22 M, HuBERT / wav2vec 2.0 base 94.37 M and large 315.44 M, an ASP readout 0.5 M (base) or 0.65 M (large), MMS
# aligner 315.47 M, one syllable judge 1.49 M; Accent Detective: the parameters that run for each size, audio only
PARAMS = {"egemaps": 0.0001, "ecapa": 20.77, "whisper": 307.22, "pooled_hubert": 94.38, "pooled_w2v": 94.38,
          "ft_hubert": 94.37, "ft_w2v": 94.37, "hubert_base_asp": 94.87, "w2v_base_asp": 94.87,
          "hubert_large_asp": 316.09, "w2v_large_asp": 316.09, "judge_hubert": 315.47 + 94.37 + 3 * 1.49,
          "judge_w2v": 315.47 + 94.37 + 3 * 1.49, "detective_nano": 9.51, "detective_tiny": 16.21,
          "detective_mini": 22.92, "detective_small": 43.02}


def acc(n, scheme=CHECKED):
    return SEALED["systems"][n]["auc"][scheme][0]


def m(name, key):
    r = REPORT[name] if name in REPORT else ENS[name]
    return r[key]["mean"]


# --------------------------------------------------------------------------- figure: accuracy does not select validity
def validity():
    """(a) validity on synthetic speech: voice share against BreezyVoice-CosyVoice contrast, every system; (b) sealed AUC
    of every strong system with the published and the listener-checked labels."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COL + 0.3, 2.15), gridspec_kw={"wspace": 0.42, "width_ratios": [1.25, 1]})
    a.axvspan(0.8, 10, color="#E8F3E8", zorder=0, lw=0)
    a.text(0.9, 0.012, "voice-invariant", fontsize=6.5, color="#4C8C4A", va="bottom")
    label_at = {"egemaps": ((26.5, 0.290), "left"), "ecapa": ((62, 0.125), "center"), "whisper": ((5.1, 0.100), "right"),
                "ft_w2v": ((21, 0.050), "right"), "ft_hubert": ((43, 0.071), "left"),
                "pooled_hubert": ((22.5, 0.432), "left"), "pooled_w2v": ((28, 0.352), "left"),
                "judge_hubert": ((3.6, 0.283), "center"), "judge_w2v": ((7.6, 0.347), "center")}
    marker = {"base": "o", "judge": "^", "probe": "s"}
    for k, (lab, fam, v) in SYSTEMS.items():
        if fam == "ours":
            continue
        x, y = 100 * v["voice_share"], v["synthetic2"]["contrast_sd"]
        mk = "D" if k.startswith("pooled") else "s" if k.startswith("ft_") else marker[fam]
        a.scatter(x, y, s=22, c=COLOR[fam], marker=mk, zorder=4, edgecolors="white", linewidths=0.4)
        if k in label_at:
            (tx, ty), ha = label_at[k]
            a.text(tx, ty, lab, fontsize=6, color=COLOR[fam], ha=ha, va="center", zorder=5)
    for e, lab in (("hubert", "HuBERT"), ("w2v", "w2v")):                      # base -> large: the scale axis
        p, q = SYSTEMS[f"{e}_base_asp"][2], SYSTEMS[f"{e}_large_asp"][2]
        a.annotate("", xy=(100 * q["voice_share"], q["synthetic2"]["contrast_sd"]),
                   xytext=(100 * p["voice_share"], p["synthetic2"]["contrast_sd"]),
                   arrowprops={"arrowstyle": "-|>", "color": C_PROBE, "lw": 0.7, "mutation_scale": 6}, zorder=3)
        a.text(100 * q["voice_share"] * (0.93 if e == "hubert" else 1.08), q["synthetic2"]["contrast_sd"] - 0.022,
               f"{lab}\nbase→large", fontsize=5.6, color=C_PROBE, ha="right" if e == "hubert" else "left", va="top",
               linespacing=0.9)
    xs = [100 * V2[k]["voice_share"] for k, _ in SIZES]
    ys = [V2[k]["synthetic2"]["contrast_sd"] for k, _ in SIZES]
    a.plot(xs, ys, color=C_OURS, lw=0.6, zorder=3)
    a.scatter(xs, ys, s=[22, 30, 40, 70], c=C_OURS, marker="*", zorder=4, edgecolors="white", linewidths=0.3)
    a.text(3.0, 0.452, "Accent Detective,\nNano to Small", fontsize=6, color=C_OURS, ha="center", va="center",
           fontweight="bold", linespacing=0.9)
    a.set_xscale("log")
    a.set_xticks([1, 2, 5, 10, 20, 50])
    a.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    a.set_xlim(0.8, 90)
    a.set_ylim(0, 0.48)
    a.set_xlabel("Voice share of score variance (%)")
    a.set_ylabel("Synthetic contrast (SD)")
    a.set_title("(a) Validity on synthetic speech", loc="left", fontsize=7)
    # (b) published -> checked labels, every strong system, sorted by the published AUC
    keys = sorted((k for k, (_, fam, _) in SYSTEMS.items() if k not in ("egemaps", "ecapa")), key=lambda k: acc(k, PUBLISHED))
    for y, k in enumerate(keys):
        lab, fam, _ = SYSTEMS[k]
        p, c = acc(k, PUBLISHED), acc(k)
        b.plot([p, c], [y, y], color=COLOR[fam], lw=0.7, alpha=0.6, zorder=2)
        b.scatter([p], [y], s=12, facecolors="white", edgecolors=COLOR[fam], linewidths=0.7, zorder=3)
        b.scatter([c], [y], s=14, c=COLOR[fam], zorder=4)
    b.set_yticks(range(len(keys)))
    b.set_yticklabels([SYSTEMS[k][0].replace(" + ASP", "+ASP").replace("Syllable judge, ", "Syl. judge, ") for k in keys], fontsize=5.6)
    for t, k in zip(b.get_yticklabels(), keys):
        t.set_color(COLOR[SYSTEMS[k][1]])
    lo, hi = min(acc(k, PUBLISHED) for k in keys), max(acc(k) for k in keys)
    b.axvspan(min(acc(k) for k in keys), hi, color="#F2F2F2", zorder=0, lw=0)
    b.set_xlim(lo - 0.006, hi + 0.006)
    b.set_ylim(-0.7, len(keys) - 0.3)
    b.set_xlabel("Sealed AUC")
    b.scatter([], [], s=12, facecolors="white", edgecolors="k", linewidths=0.7, label="published labels")
    b.scatter([], [], s=14, c="k", label="listener-checked")
    b.legend(loc="upper left", frameon=False, fontsize=5.4, handletextpad=0.2, borderaxespad=0.1)
    b.set_title("(b) Accuracy, published vs. checked labels", loc="left", fontsize=7)
    fig.savefig(FIG / "validity.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- figure: what each size costs
def cost():
    """(a) sealed AUC (checked labels) against CPU latency, batch 1; (b) GPU throughput by batch size; (c) anytime scoring
    within one pass of the largest size against each size on its own (CPU, batch 1)."""
    fig, (b, c, d) = plt.subplots(1, 3, figsize=(2 * COL + 0.3, 1.75), gridspec_kw={"wspace": 0.45})
    L = json.load(open(PAPER / "data/latency_nested.json"))
    refs = {}
    for q in sorted((PAPER / "data").glob("latency_ref_*.json")):
        for r in json.load(open(q))["reference"]:
            refs[(r["device"], r["threads"], r["system"])] = r["median_ms"]
    rows = [r for r in L["nested"] if r["mode"] == "audio" and "batch_ms" in r]
    for th, ls, ms in ((1, "-", 5), (4, ":", 3.5)):
        xs = [next(r["batch_ms"] for r in rows if r["device"] == "cpu" and r["batch"] == 1 and r["threads"] == th
                   and r["size"] == k.split("_")[1]) for k, _ in SIZES]
        ys = [acc(k) for k, _ in SIZES]
        b.plot(xs, ys, color=C_OURS, lw=0.8, ls=ls, marker="*", ms=ms, zorder=3, label=f"{th} thread{'s' if th > 1 else ''}")
        if th == 1:                                   # label offsets (x factor, y shift, va) keep the labels apart
            for x, y, (_, lab), (fx, dy, va) in zip(xs, ys, SIZES, ((1.12, -0.0006, "top"), (0.8, 0.0006, "bottom"),
                                                                     (1.12, -0.0006, "top"), (1.0, 0.0007, "bottom"))):
                b.text(x * fx, y + dy, lab, fontsize=5.6, color=C_OURS, va=va, ha="left" if fx > 1 else "center")
    for th, sz in ((1, 26), (4, 14)):
        v = refs.get(("cpu", th, "judge_pipeline"))
        if v:
            b.scatter([v], [acc("judge_hubert")], marker="^", s=sz, c=C_TEACH, zorder=4, edgecolors="white", linewidths=0.4)
            if th == 1:
                b.text(v, acc("judge_hubert") + 0.0008, "judge\npipeline", fontsize=5.6, color=C_TEACH, ha="center",
                       va="bottom", linespacing=0.9)
    b.axvline(5000, color=C_GREY, lw=0.6, ls="--")
    b.text(5700, 0.5, "real time", fontsize=5.6, color=C_BASE, ha="left", va="center", rotation=90,
           transform=b.get_xaxis_transform())
    b.set_xscale("log")
    b.set_xlim(15, 8000)
    b.set_ylim(0.950, 0.970)
    b.set_xlabel("CPU latency, 5 s clip (ms)")
    b.set_ylabel("Sealed AUC (checked)")
    b.set_title("(a) Accuracy and CPU cost", loc="left", fontsize=7)
    b.legend(loc="upper left", frameon=False, fontsize=5.2, handletextpad=0.2, borderaxespad=0.1, handlelength=1.4)
    cols = {"nano": "#F4A582", "tiny": "#E07B39", "mini": "#C0501A", "small": "#7F2704"}
    for k, lab in SIZES:
        sz = k.split("_")[1]
        for prec, ls in (("fp32", "-"), ("bf16", "--")):
            pts = sorted((r["batch"], r["clips_per_s"]) for r in rows if r["device"] == "cuda" and r["size"] == sz
                         and r["precision"] == prec)
            if pts:
                c.plot(*zip(*pts), color=cols[sz], ls=ls, lw=0.9, marker="o", ms=2, label=lab if prec == "fp32" else None)
    c.plot([], [], color="k", ls="--", lw=0.9, label="bf16")
    c.set_xscale("log", base=2)
    c.set_yscale("log")
    c.set_xticks([1, 8, 32, 128, 512])
    c.set_xticklabels(["1", "8", "32", "128", "512"])
    c.set_xlabel("Batch size")
    c.set_ylabel("5 s clips per second (GPU)")
    c.set_title("(b) GPU throughput", loc="left", fontsize=7)
    c.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.2, borderaxespad=0.1, ncol=2, columnspacing=0.6)
    th = 4
    any_ = next(r for r in L["anytime"] if r["device"] == "cpu" and r["threads"] == th and r["batch"] == 1 and r["mode"] == "audio")
    t = [any_["time_to_size_ms"][k.split("_")[1]] for k, _ in SIZES]
    alone = [next(r["batch_ms"] for r in rows if r["device"] == "cpu" and r["batch"] == 1 and r["threads"] == th
                  and r["size"] == k.split("_")[1]) for k, _ in SIZES]
    ys = [acc(k) for k, _ in SIZES]
    d.step(t + [sum(alone) * 1.05], ys + [ys[-1]], where="post", color=C_OURS, lw=0.9, label="one pass")
    d.scatter(t, ys, marker="*", s=28, c=C_OURS, zorder=4)
    d.scatter(alone, ys, marker="x", s=12, c=C_TEACH, zorder=5, linewidths=0.8, label="each size alone")
    d.axvline(sum(alone), color=C_GREY, lw=0.6, ls="--")
    d.text(sum(alone) * 0.97, 0.45, "all four\nseparately", fontsize=5.4, color=C_BASE, ha="right", va="center", linespacing=0.9,
           transform=d.get_xaxis_transform())
    for x, y, (_, lab), (dx, dy, va) in zip(t, ys, SIZES, ((8, -0.0004, "top"), (-6, 0.0005, "bottom"), (8, -0.0004, "top"),
                                                            (8, 0.0004, "bottom"))):
        d.text(x + dx, y + dy, lab, fontsize=5.6, color=C_OURS, ha="left" if dx > 0 else "right", va=va)
    d.set_xlim(0, sum(alone) * 1.05)
    d.set_ylim(0.950, 0.966)
    d.set_xlabel("Time since the clip arrived (ms)")
    d.set_ylabel("Sealed AUC (checked)")
    d.set_title("(c) Anytime scoring (CPU)", loc="left", fontsize=7)
    d.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.2, borderaxespad=0.1)
    fig.savefig(FIG / "cost.pdf")
    plt.close(fig)


def fmt(v, d=3):
    return f"{v:.{d}f}".lstrip("0") if abs(v) < 1 else f"{v:.{d}f}"


def signed(v):
    return f"{v:+.3f}".replace("0.", ".").replace("-", "$-$")


def emphasize(rows, directions):
    """Bold the best and underline the second-best value of each column. rows: lists of cell strings, or None for a
    rule; directions: {column index: +1 if higher is better, -1 if lower is better}. Ties share the mark."""
    for c, d in directions.items():
        vals = sorted({float(r[c]) for r in rows if r is not None}, reverse=d > 0)
        for r in rows:
            if r is None:
                continue
            v = float(r[c])
            if v == vals[0]:
                r[c] = "\\textbf{" + r[c] + "}"
            elif len(vals) > 1 and v == vals[1]:
                r[c] = "\\underline{" + r[c] + "}"
    return "\n".join("\\midrule" if r is None else " & ".join(r) + " \\\\" for r in rows) + "\n"


def main_table():
    """Every system: sealed AUC with both label sets, synthetic pairs, 10-pair verdict, voice set, KeSpeech."""
    groups = [["egemaps", "ecapa", "whisper", "pooled_hubert", "pooled_w2v", "ft_hubert", "ft_w2v"],
              ["hubert_base_asp", "hubert_large_asp", "w2v_base_asp", "w2v_large_asp"], ["judge_hubert", "judge_w2v"],
              [k for k, _ in SIZES]]
    out = []
    for g in groups:
        if out:
            out.append(None)
        for n in g:
            lab, _, v = SYSTEMS[n]
            sy, ke, s = v["synthetic2"], v["kespeech_judged_taiwan"]["all"], SEALED["systems"][n]
            p = PARAMS[n]
            out.append([lab, "$<$0.1" if p < 0.1 else f"{p:.0f}" if p >= 10 else f"{p:.1f}",
                        fmt(s["auc"][PUBLISHED][0]), fmt(s["auc"][CHECKED][0]),
                        f"{100 * sy['breezy_first']:.0f}", fmt(sy["contrast_sd"], 2), f"{100 * s['tts10']:.0f}",
                        f"{100 * v['voice_share']:.1f}", f"{100 * ke['none']:.0f}", f"{100 * ke['Southwestern']:.0f}"])
    (PAPER / "tables/main_rows.tex").write_text(emphasize(out, {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: -1, 8: -1, 9: -1}))


def listening_table():
    """The blind listening check: answers per pool and label, with how many of the Taiwan-labelled clips the automatic
    screening keeps (in parentheses); paired AUC differences with both label sets."""
    R = SEALED["ratings"]
    out = []
    for lab, keys, pool in (("Disputed, Taiwan label", ("core_tw", "other_tw"), "disputed"),
                            ("Disputed, Mainland label", ("core_cn", "other_cn"), None),
                            ("Control, Taiwan label", ("control_tw",), "control"), ("Control, Mainland label", ("control_cn",), None)):
        c = {v: sum(R[k][v] for k in keys) for v in ("tw", "mid", "cn", "silent", "lang")}
        kept = CLEAN["listener"][pool]["clean_kept"] if pool else None
        cell = lambda v: str(c[v]) + (f" ({kept.get(v, 0)})" if kept is not None else "")
        same, other = ("tw", "cn") if "Taiwan" in lab else ("cn", "tw")
        out.append(" & ".join([lab, str(sum(c.values())), cell(same), cell(other), cell("mid"), cell("silent"), cell("lang")]) + " \\\\")
    (PAPER / "tables/listening_rows.tex").write_text("\n".join(out) + "\n")
    P = SEALED["paired"]
    out = []
    for key, lab in (("hubert_large_asp - hubert_base_asp", "HuBERT large $-$ base"),
                     ("w2v_large_asp - w2v_base_asp", "w2v large $-$ base"),
                     ("detective_small - detective_nano", "Detective Small $-$ Nano"),
                     ("whisper - detective_small", "Whisper $-$ Detective Small")):
        cells = [lab]
        for s in (PUBLISHED, CHECKED):
            v, lo, hi = P[key][s]
            cells.append(f"{signed(v)} [{signed(lo)}, {signed(hi)}]")
        out.append(" & ".join(cells) + " \\\\")
    (PAPER / "tables/paired_rows.tex").write_text("\n".join(out) + "\n")


def readout_table():
    """One trunk (the same 18 layers), four readouts: syllable judge with and without the text, ASP, ASP + text."""
    rows = [("syllable_text", "Syllables, aligned to text"), ("syllable_audio", "Syllables, decoded"),
            ("asp_audio", "Attentive pooling"), ("asp_text", "\\quad + text readout")]
    out = []
    for n, lab in rows:
        v, s = RD[n], SEALED["systems"][f"readout_{n}"]
        sy = v["synthetic2"]
        out.append([lab, fmt(s["auc"][PUBLISHED][0]), fmt(s["auc"][CHECKED][0]), f"{100 * sy['breezy_first']:.0f}",
                    fmt(sy["contrast_sd"], 2), f"{100 * v['voice_share']:.1f}",
                    f"{100 * v['kespeech_judged_taiwan']['all']['Southwestern']:.0f}"])
    (PAPER / "tables/readout_rows.tex").write_text(emphasize(out, {1: 1, 2: 1, 3: 1, 4: 1, 5: -1, 6: -1}))


def ablation_table():
    """Ablations of the two syllable judges and of the pooled baselines, per encoder: AUC, ranking, paired SD, voice."""
    rows = [("", "Syllable judge"), ("no per-utterance normalisation", "\\quad $-$ normalisation"),
            ("uniform segments, no alignment", "\\quad $-$ alignment"), ("no text context", "\\quad $-$ text context"),
            ("no text-only model", "\\quad $-$ text-only expert"), ("no importance weights", "\\quad $-$ imp. weights"),
            None, ("utterance pooling + linear", "Pooled mean+SD"), ("utterance std pooling (mean removed) + linear", "\\quad $-$ mean"),
            ("syllable-feature pooling + linear", "\\quad syl. avg.")]
    out = []
    for r in rows:
        if r is None:
            out.append(None)
            continue
        key, lab = r
        cells = [lab]
        for j, enc in (("B", "HuBERT"), ("C", "wav2vec 2.0")):
            if lab.startswith("Pooled") or "mean" in lab or "avg" in lab:
                n = f"{enc} {key}"
            else:
                n = f"judge {j}" + (f" — {key}" if key else "")
            cells += [fmt(m(n, "auc_length_matched")), f"{100 * m(n, 'synthetic_frac_breezy_first'):.0f}",
                      fmt(m(n, "synthetic_paired_diff_sd"), 2), f"{100 * m(n, 'voice_share'):.0f}"]
        out.append(cells)
    (PAPER / "tables/ablation_rows.tex").write_text(emphasize(out, {1: 1, 2: 1, 3: 1, 4: -1, 5: 1, 6: 1, 7: 1, 8: -1}))


def robust_table():
    """Recording conditions (channels.py): change in the sealed AUC of every size, listener-checked labels."""
    R = SEALED["robust"]
    out = []
    for c, lab in (("mp3", "MP3"), ("tel", "Telephone"), ("noise", "Noise"), ("reverb", "Reverb.")):
        out.append(" & ".join([lab] + [signed(R[k.split("_")[1]][CHECKED][c]) for k, _ in SIZES]) + " \\\\")
    (PAPER / "tables/robust_rows.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    (PAPER / "tables").mkdir(exist_ok=True)
    validity()
    cost()
    main_table()
    listening_table()
    readout_table()
    ablation_table()
    robust_table()
