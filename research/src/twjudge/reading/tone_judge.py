"""F0-based tone judge for words whose Taiwan and Mainland readings differ only in tone.

The forced aligner separates segments well but not tones (it agreed with a
listener on segmental differences and was near chance on tone-only ones). This
module classifies each syllable's tone from its F0 contour instead:

1. ``reading_judge`` gives syllable boundaries.
2. F0 (Praat via parselmouth) in semitones relative to the utterance median is
   sampled at fixed points over each syllable rhyme, with the neighbouring
   syllables' edges as context.
3. A gradient-boosting classifier is trained on syllables whose reading is the
   same on both sides (no labelling needed), from real Taiwan speakers.
4. For a target word, the judge compares the classifier's probability of the
   Taiwan tone against the Mainland tone on the syllables that differ.

Tone labels apply third-tone sandhi within a word (3 3 → 2 3); 一/不 sandhi is
already applied by pypinyin on the Mainland side.

    python -m twjudge.reading.tone_judge train --judged data/judge/cv_train --model tone.pkl
    python -m twjudge.reading.tone_judge eval  --judged data/judge/cv_dev --model tone.pkl
    python -m twjudge.reading.tone_judge judge --judged <dir> --model tone.pkl --out judged.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
from pathlib import Path

import numpy as np

N_POINTS = 10
TONES = ["1", "2", "3", "4", "5"]


def sandhi(tones: list[str]) -> list[str]:
    out = list(tones)
    for k in range(len(out) - 2, -1, -1):
        if out[k] == "3" and out[k + 1] == "3":
            out[k] = "2"
    return out


def f0_track(wav: Path):
    import parselmouth

    snd = parselmouth.Sound(str(wav))
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
    t = pitch.xs()
    f = pitch.selected_array["frequency"]
    voiced = f > 0
    if voiced.sum() < 10:
        return None
    st = np.full_like(f, np.nan)
    st[voiced] = 12 * np.log2(f[voiced] / np.median(f[voiced]))
    return t, st


def contour(t, st, a: float, b: float) -> np.ndarray | None:
    m = (t >= a) & (t <= b)
    if m.sum() < 3:
        return None
    tt, ss = t[m], st[m]
    ok = ~np.isnan(ss)
    if ok.mean() < 0.5 or ok.sum() < 3:
        return None
    grid = np.linspace(tt[0], tt[-1], N_POINTS)
    return np.interp(grid, tt[ok], ss[ok])


def syllable_rows(judged_dir: Path):
    """Yield (item_id, word_index, syllable_index, features, label_or_None, word_info)."""
    wavs = {p.stem: p for c in judged_dir.glob("corpus*") for p in c.glob("*/*.wav")}
    for line in (judged_dir / "judgements.jsonl").open():
        r = json.loads(line)
        if r["status"] != "ok" or r["item_id"] not in wavs:
            continue
        tr = f0_track(wavs[r["item_id"]])
        if tr is None:
            continue
        t, st = tr
        flat = [(wi, si, s) for wi, w in enumerate(r["words"]) for si, s in enumerate(w["syllables"])]
        conts = [contour(t, st, s["rhyme_start"], s["end"]) for _, _, s in flat]
        nan = np.full(N_POINTS, np.nan)
        for k, (wi, si, s) in enumerate(flat):
            c = conts[k]
            if c is None:
                continue
            # neighbouring syllables' whole contours: tones are realised relative to context
            prev = conts[k - 1] if k > 0 and conts[k - 1] is not None else nan
            nxt = conts[k + 1] if k + 1 < len(flat) and conts[k + 1] is not None else nan
            gap_prev = s["start"] - flat[k - 1][2]["end"] if k > 0 else np.nan
            gap_next = flat[k + 1][2]["start"] - s["end"] if k + 1 < len(flat) else np.nan
            feats = np.concatenate([c, np.diff(c), prev, nxt, c - np.nanmean(prev), [
                c.mean(), c.max() - c.min(), int(np.argmin(c)), int(np.argmax(c)),
                s["end"] - s["rhyme_start"], s["end"] - s["start"], gap_prev, gap_next,
                float(k == 0), float(k == len(flat) - 1), k / max(1, len(flat) - 1)]])
            w = r["words"][wi]
            yield r["item_id"], wi, si, feats, w


def label_of(w: dict, si: int) -> str | None:
    """Tone label for training: only syllables whose reading both sides agree on."""
    cn, tw = w["cn"], w["tw"]
    if cn is None or w["target"] or w["chosen"] is None:
        return None
    if tw is not None and tw[si] != cn[si]:
        return None
    tones = sandhi([s[-1] if s[-1].isdigit() else "5" for s in cn])
    return tones[si]


def train(judged: Path, model_path: Path) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = [], []
    for _, _, si, f, w in syllable_rows(judged):
        lab = label_of(w, si)
        if lab is not None:
            X.append(f)
            y.append(lab)
    X, y = np.array(X), np.array(y)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=17)
    clf.fit(X, y)
    model_path.write_bytes(pickle.dumps(clf))
    print(json.dumps({"train_syllables": len(y), "label_counts": collections.Counter(y.tolist())}))


def evaluate(judged: Path, model_path: Path) -> dict:
    clf = pickle.loads(model_path.read_bytes())
    X, y = [], []
    for _, _, si, f, w in syllable_rows(judged):
        lab = label_of(w, si)
        if lab is not None:
            X.append(f)
            y.append(lab)
    pred = clf.predict(np.array(X))
    y = np.array(y)
    per = {t: round(float((pred[y == t] == t).mean()), 3) for t in TONES if (y == t).any()}
    # pairwise accuracy is what the judge needs: tone A vs tone B given the truth is one of them
    proba = clf.predict_proba(np.array(X))
    idx = {c: i for i, c in enumerate(clf.classes_)}
    pair = collections.defaultdict(lambda: [0, 0])
    for p, truth in zip(proba, y):
        for other in TONES:
            if other == truth or other not in idx or truth not in idx:
                continue
            key = "".join(sorted(truth + other))
            pair[key][0] += int(p[idx[truth]] > p[idx[other]])
            pair[key][1] += 1
    res = {"syllables": len(y), "accuracy": round(float((pred == y).mean()), 3), "per_tone_recall": per,
           "pairwise_accuracy": {k: round(a / n, 3) for k, (a, n) in sorted(pair.items())}}
    print(json.dumps(res))
    return res


def judge(judged: Path, model_path: Path, out: Path) -> None:
    """Judgements for target words; tone-only targets by F0, segmental ones by the aligner."""
    clf = pickle.loads(model_path.read_bytes())
    idx = {c: i for i, c in enumerate(clf.classes_)}
    probs = {}
    for item, wi, si, f, w in syllable_rows(judged):
        if w["target"]:
            probs[(item, wi, si)] = clf.predict_proba(f[None])[0]
    rows = []
    for line in (judged / "judgements.jsonl").open():
        r = json.loads(line)
        out_j = []
        for wi, w in enumerate(r.get("words", [])):
            if not w["target"]:
                continue
            j = {"word": w["word"], "tw": " ".join(w["tw"]), "cn": " ".join(w["cn"]),
                 "segmental": w["segmental"]}
            if w["segmental"]:
                j.update(heard=w["chosen"] or "other", method="aligner")
            else:
                tw_t = sandhi([s[-1] if s[-1].isdigit() else "5" for s in w["tw"]])
                cn_t = sandhi([s[-1] if s[-1].isdigit() else "5" for s in w["cn"]])
                score, used = 0.0, 0
                for si in range(len(tw_t)):
                    if tw_t[si] == cn_t[si] or (r["item_id"], wi, si) not in probs:
                        continue
                    p = probs[(r["item_id"], wi, si)]
                    score += math.log(p[idx[tw_t[si]]] + 1e-6) - math.log(p[idx[cn_t[si]]] + 1e-6)
                    used += 1
                j.update(method="f0", log_odds_tw=round(score, 3),
                         heard=("tw" if score > 0 else "cn") if used else "unjudged")
            out_j.append(j)
        for j in r.get("judgements", []):
            if j.get("heard") == "unaligned":
                out_j.append({**j, "method": "aligner"})
        rows.append({"item_id": r["item_id"], "judgements": out_j})
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(j["heard"] for r in rows for j in r["judgements"])
    tot = c["tw"] + c["cn"]
    print(json.dumps({"words": dict(c), "tw_rate": round(c["tw"] / tot, 3) if tot else None},
                     ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train", "eval", "judge"])
    ap.add_argument("--judged", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "train":
        train(Path(a.judged), Path(a.model))
    elif a.cmd == "eval":
        evaluate(Path(a.judged), Path(a.model))
    else:
        judge(Path(a.judged), Path(a.model), Path(a.out))


if __name__ == "__main__":
    main()
