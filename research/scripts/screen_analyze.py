"""Screening analysis: the flags on every Common Voice zh-TW clip scored by
screen_score.py, their check against the listener (the blind listening check), the cleaned list, and the sealed set re-scored with the
automatic flags.

  silent        fewer than 0.5 s of voiced audio (Silero VAD)
  not_reading   speech, but the prompt's Mandarin pinyin aligns badly: ctc_tok < 0.25, or the audio is too short for
                the prompt's letters; a prompt that cannot be turned into pinyin is left unjudged
  accent        Accent Detective Small, calibrated P(Taiwan) below --p-accent (pre-registered 0.1; the release uses 0.5,
                the user's choice after the results, for purer Taiwan-accented training speech)
  speaker       a speaker with at least 3 clips with speech whose mean calibrated log-odds is below --spk-z
                (pre-registered -2.2; the release uses 0)
  clean         none of the four

    PYTHONPATH=research/src uv run --project research python research/scripts/screen_analyze.py --scores data/bench/screen/tw \\
        --out data/bench/screen --summary dataset/screening_summary_p0.5.json --p-accent 0.5 --spk-z 0
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
from twjudge.accent.prosodic import auc, length_weights  # noqa: E402

D = ROOT / "data/bench/paper"
DOCS = ROOT / "dataset/listening"
VOICED, TOK, P_ACCENT, SPK_Z, SPK_N = 0.5, 0.25, 0.1, -2.2, 3


def logit(p):
    return float(np.log(p / (1 - p)))


def flags(rows, voiced=VOICED, tok=TOK, p_accent=P_ACCENT, spk_z=SPK_Z):
    z_acc = logit(p_accent)
    for r in rows:
        r["silent"] = r["voiced"] < voiced
        r["unjudged"] = not r["silent"] and r["n_tok"] is None
        r["not_reading"] = not r["silent"] and r["n_tok"] is not None and (r["ctc_tok"] is None or r["ctc_tok"] < tok)
        r["accent"] = r["z"] is not None and r["z"][3] < z_acc
    by = collections.defaultdict(list)
    for r in rows:
        if not r["silent"] and r["z"] is not None:
            by[r["speaker"]].append(r["z"][3])
    bad = {s for s, v in by.items() if len(v) >= SPK_N and np.mean(v) < spk_z}
    for r in rows:
        r["speaker_flag"] = r["speaker"] in bad
        r["clean"] = not (r["silent"] or r["not_reading"] or r["accent"] or r["speaker_flag"])
    return rows, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="dir of screen_score.py JSONL files")
    ap.add_argument("--out", required=True, help="dir for the cleaned list")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--p-accent", type=float, default=P_ACCENT)
    ap.add_argument("--spk-z", type=float, default=SPK_Z)
    a = ap.parse_args()
    th = {"p_accent": a.p_accent, "spk_z": a.spk_z}
    rows = [json.loads(l) for p in sorted(Path(a.scores).glob("*.jsonl")) for l in open(p)]
    rows, bad = flags(rows, **th)
    S = {"thresholds": {"voiced": VOICED, "ctc_tok": TOK, **th}, "clips": len(rows),
         "speakers": len({r["speaker"] for r in rows})}

    # ---- where each clip is used in the project
    member = {}
    for s in ("cv_train", "cv_dev", "cv_test", "cv_heldout"):
        for l in open(D / "sets" / f"{s}.jsonl"):
            i = json.loads(l)["item_id"]
            if i.startswith("tw_"):
                member[i[3:]] = s
    for s, p in (("sealed", D / "sets/r10_sealed.jsonl"), ("task", ROOT / "data/bench/sealed/r10_task.jsonl")):
        for l in open(p):
            i = json.loads(l)["item_id"]
            if i.startswith("tw_r10_"):
                member[i[7:]] = s
    group = {"cv_train": "trained", "task": "trained", "cv_dev": "development", "cv_test": "development",
             "cv_heldout": "test", "sealed": "test"}
    for r in rows:
        r["use"] = group.get(member.get(r["clip"]), "unused")

    def rate(rs, k):
        return round(float(np.mean([x[k] for x in rs])), 4) if rs else None

    keys = ("silent", "not_reading", "unjudged", "accent", "speaker_flag", "clean")
    S["flags"] = {g: {"n": len(rs), **{k: rate(rs, k) for k in keys}}
                  for g, rs in [("all", rows)] + [(g, [r for r in rows if r["use"] == g])
                                                  for g in ("trained", "development", "test", "unused")]}
    S["speakers_flagged"] = len(bad)
    print(json.dumps(S["flags"], indent=1))

    # ---- sensitivity of the clean share to each threshold
    sens = {}
    for name, kw in (("tok 0.15", {"tok": 0.15}), ("tok 0.35", {"tok": 0.35}), ("p 0.05", {"p_accent": 0.05}),
                     ("p 0.2", {"p_accent": 0.2}), ("voiced 0.3", {"voiced": 0.3}), ("voiced 1.0", {"voiced": 1.0})):
        rr, _ = flags([dict(r) for r in rows], **{**th, **kw})
        sens[name] = rate(rr, "clean")
    rows, bad = flags(rows, **th)
    S["sensitivity_clean_share"] = sens
    print("clean share under other thresholds", sens)

    # ---- against the listener: Taiwan-labelled clips of the listening check
    key = {k["clip"]: k for k in json.load(open(DOCS / "rating_key.json"))}
    ans = json.load(open(DOCS / "ratings_check.json"))["answers"]
    heard = {key[c]["item_id"].split("_r10_", 1)[1]: (v, "control" if key[c]["pool"] == "control" else "disputed")
             for c, v in ans.items() if key[c]["item_id"].startswith("tw")}
    by_clip = {r["clip"]: r for r in rows}
    V = {}
    for pool in ("disputed", "control", "all"):
        rs = [(by_clip[c], v) for c, (v, pl) in heard.items() if c in by_clip and pool in ("all", pl)]
        pos = lambda v: v in ("silent", "lang")
        v = {"n": len(rs), "listener": dict(collections.Counter(x for _, x in rs))}
        for name, f, truth in (("silent", lambda r: r["silent"], lambda x: x == "silent"),
                               ("silent_or_not_reading", lambda r: r["silent"] or r["not_reading"], pos)):
            tp = sum(f(r) and truth(x) for r, x in rs)
            v[name] = {"flagged": sum(f(r) for r, _ in rs), "true": sum(truth(x) for _, x in rs), "hit": tp,
                       "mandarin_flagged": sum(f(r) for r, x in rs if x in ("tw", "mid", "cn"))}
        removed = [x for r, x in rs if r["accent"] or r["speaker_flag"]]
        kept = [x for r, x in rs if r["clean"]]
        v["accent_removed"] = dict(collections.Counter(removed))
        v["clean_kept"] = dict(collections.Counter(kept))
        V[pool] = v
    S["listener"] = V
    print(json.dumps(V, indent=1))

    # ---- the sealed set with automatic flags (Taiwan side only; the Mainland side keeps every clip)
    sealed_rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(D / "sets/r10_sealed.jsonl")}

    def ens(prefix):
        ss = [json.load(open(D / "scores" / f"{prefix}_s{k}.json"))["r10_sealed"] for k in range(3)]
        return {i: [float(np.mean([x[i][0] for x in ss]))] for i in ss[0] if all(x.get(i) is not None for x in ss)}
    sysm = {n: json.load(open(D / "scores" / f"{n}.json"))["r10_sealed"]
            for n in ("whisper", "pooled_hubert", "ft_hubert", "detective_nano", "detective_small")}
    sysm["judge_hubert"] = ens("teacher_b_adv0")
    for n in ("hubert_base", "hubert_large"):
        sysm[f"{n}_asp"] = json.load(open(ROOT / "data/bench/probes/scores" / f"{n}.json"))["r10_sealed"]
    ids = [i for i in sealed_rows if all(s.get(i) is not None for s in sysm.values())]
    lens = np.array([len(re.sub(r"[^㐀-鿿]", "", sealed_rows[i]["text"])) for i in ids])
    y = np.array([i.startswith("tw") for i in ids]).astype(int)

    def drop(i, ks):
        r = by_clip.get(i.split("_r10_", 1)[1]) if i.startswith("tw") else None
        return r is not None and any(r[k] for k in ks)
    E = {}
    for name, ks in (("published", ()), ("auto: silent + not reading", ("silent", "not_reading")),
                     ("auto: all four (circular for Detective)", ("silent", "not_reading", "accent", "speaker_flag"))):
        keep = np.array([not drop(i, ks) for i in ids])
        E[name] = {"removed": int((~keep).sum()),
                   **{n: round(float(auc(y[keep], np.array([s[i][0] for i in ids])[keep], length_weights(lens[keep], y[keep]))), 4)
                      for n, s in sysm.items()}}
    S["sealed"] = E
    print(json.dumps(E, indent=1))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"p{a.p_accent:g}"
    with open(out / f"cv25_zh-TW_flags_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("clip", "split", "speaker", "use", "voiced", "ctc_tok", "z", "silent",
                                                   "not_reading", "unjudged", "accent", "speaker_flag", "clean")},
                               ensure_ascii=False) + "\n")
    Path(a.summary).write_text(json.dumps(S, indent=1, ensure_ascii=False))
    print("wrote", out / f"cv25_zh-TW_flags_{tag}.jsonl", a.summary)


if __name__ == "__main__":
    main()
