"""Interpretable Taiwan-accent judge on prosodic-phonetic features (judge A).

Two gradient-boosted models: one scores each syllable from its own pitch, timing, loudness,
neighbour contrasts and realised variant given its tone context; one scores the utterance from
its rhythm and intonation statistics. The utterance score is a logistic combination of the mean
syllable log-odds and the utterance log-odds, fitted on the judge's dev split.

Content is controlled by importance weights: syllables are reweighted so that each text-context
cell (tone, sandhi tone, Mainland-neutral, phrase-final, variant opportunities) carries equal
weight in both classes, and utterances so that sentence lengths match. Weighting leaves
interactions (position × tone, utterance-level aggregation) a text-only model can still exploit,
so each syllable's log-odds is also residualised: a text-only model trained the same way is
subtracted, and only what the audio adds beyond the text context is scored. `check` reports the
text-only control before and after.

    python -m twjudge.accent.prosodic train --feat-dir data/judge_corpus --model data/judge/models/judge_a.pkl
    python -m twjudge.accent.prosodic score --model ... --feat feat.npz --out scores.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np

from twjudge.accent.features import SYL_NAMES, TEXT_COLS, UTT_NAMES

UTT_USE = [n for n in UTT_NAMES if n != "n_syl"]
LEN_BINS = np.array([0, 7, 9, 11, 14, 18, 24, 100])


def load(path: Path):
    d = np.load(path, allow_pickle=False)
    ids = [str(i) for i in d["ids"]]
    y = np.array([i.startswith("tw") for i in ids], int)
    return ids, d["X"], d["lens"], d["U"], y


def _cells(X):
    cols = [SYL_NAMES.index(c) for c in TEXT_COLS]
    return [tuple(r) for r in X[:, cols].astype(int)]


def text_weights(X, ysyl):
    cells = _cells(X)
    cnt = collections.Counter(zip(cells, ysyl))
    tot = collections.Counter(cells)
    w = np.array([tot[c] / (2 * cnt[(c, y)]) if cnt[(c, 1 - y)] > 0 else 0.0 for c, y in zip(cells, ysyl)])
    return w


def length_weights(lens, y):
    b = np.digitize(lens, LEN_BINS)
    cnt = collections.Counter(zip(b, y))
    tot = collections.Counter(b)
    return np.array([tot[k] / (2 * cnt[(k, c)]) if cnt[(k, 1 - c)] > 0 else 0.0 for k, c in zip(b, y)])


def _syl_y(lens, y):
    return np.repeat(y, lens)


class JudgeA:
    def fit(self, train, dev):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        ids, X, lens, U, y = train
        ys = _syl_y(lens, y)
        uw = length_weights(lens, y)
        sw = text_weights(X, ys) * np.repeat(uw / np.maximum(lens, 1), lens) * len(lens)
        cat = [SYL_NAMES.index(c) for c in ("tone", "sandhi")]
        self.syl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                  categorical_features=cat, l2_regularization=1.0,
                                                  early_stopping=True, validation_fraction=0.1, random_state=0)
        self.syl.fit(X, ys, sample_weight=sw)
        self.tcols = [SYL_NAMES.index(c) for c in TEXT_COLS + ["relpos"]]
        self.txt = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        self.txt.fit(X[:, self.tcols], ys, sample_weight=sw)
        ui = [UTT_NAMES.index(n) for n in UTT_USE]
        self.ui = ui
        self.utt = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                  l2_regularization=1.0, early_stopping=True,
                                                  validation_fraction=0.1, random_state=0)
        self.utt.fit(U[:, ui], y, sample_weight=uw)
        z = self.parts(dev)
        self.comb = LogisticRegression().fit(z, dev[4], sample_weight=length_weights(dev[2], dev[4]))
        return self

    def syllable_logits(self, X):
        """Log-odds of Taiwan beyond what the text context alone predicts."""
        p = np.clip(self.syl.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
        q = np.clip(self.txt.predict_proba(X[:, self.tcols])[:, 1], 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)) - np.log(q / (1 - q))

    def parts(self, data):
        ids, X, lens, U, y = data
        sl = self.syllable_logits(X)
        off = np.r_[0, np.cumsum(lens)]
        ms = np.array([sl[off[k]:off[k + 1]].mean() for k in range(len(lens))])
        pu = np.clip(self.utt.predict_proba(U[:, self.ui])[:, 1], 1e-4, 1 - 1e-4)
        return np.c_[ms, np.log(pu / (1 - pu))]

    def score(self, data):
        """Utterance P(Taiwan) and its log-odds."""
        z = self.parts(data)
        p = self.comb.predict_proba(z)[:, 1]
        return p, self.comb.decision_function(z)


def auc(y, s, w=None):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s, sample_weight=w))


def check(model: JudgeA, test, train):
    """AUC on held-out speakers, length-matched; plus the text-only control."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    ids, X, lens, U, y = test
    p, z = model.score(test)
    out = {"test_auc_length_matched": auc(y, z, length_weights(lens, y)), "test_auc_raw": auc(y, z)}
    parts = model.parts(test)
    out["syllable_part_auc"] = auc(y, parts[:, 0], length_weights(lens, y))
    out["utterance_part_auc"] = auc(y, parts[:, 1], length_weights(lens, y))
    # text-only control: the same weighting, only text-context columns as input
    cols = [SYL_NAMES.index(c) for c in TEXT_COLS + ["relpos"]]
    _, Xt, lt, _, yt = train
    yst = _syl_y(lt, yt)
    sw = text_weights(Xt, yst) * np.repeat(length_weights(lt, yt) / np.maximum(lt, 1), lt) * len(lt)
    ctl = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xt[:, cols], yst, sample_weight=sw)
    ps = np.clip(ctl.predict_proba(X[:, cols])[:, 1], 1e-4, 1 - 1e-4)
    off = np.r_[0, np.cumsum(lens)]
    ms = np.array([np.log(ps[off[k]:off[k + 1]] / (1 - ps[off[k]:off[k + 1]])).mean() for k in range(len(lens))])
    out["text_only_control_auc"] = auc(y, ms, length_weights(lens, y))
    out["judge_vs_text_only_corr"] = float(np.corrcoef(ms, z)[0, 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--feat-dir", default="data/judge_corpus")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--model", required=True)
    ap.add_argument("--feat")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "train":
        fd = Path(a.feat_dir)
        tr, dv, te = (load(fd / f"feat_{s}.npz") for s in (a.train_split, "dev", "test"))
        # pickle the class from its importable module, not from __main__
        from twjudge.accent.prosodic import JudgeA as _JudgeA

        m = _JudgeA().fit(tr, dv)
        Path(a.model).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(m, open(a.model, "wb"))
        print(json.dumps(check(m, te, tr), indent=1))
        return
    m = pickle.load(open(a.model, "rb"))
    data = load(Path(a.feat))
    p, z = m.score(data)
    with open(a.out, "w") as f:
        for i, pp, zz in zip(data[0], p, z):
            f.write(json.dumps({"item_id": i, "p_tw": float(pp), "logit": float(zz)}) + "\n")


if __name__ == "__main__":
    main()
