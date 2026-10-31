"""The judges of the nested Accent Detective, one per exit, both modes (nested_judge_features.py output).

A variant is (exit k, mode): size Nk reads exit k's predicted layers (v4 v8 v12 b8 b12) with that exit's judge
(twjudge.accent.nested.NestedJudge); mode t = forced segmentation and the text context, mode f = decoded
segmentation, context unknown. Every step trains all 8 variants on the same clips, each on its own labels (the same
judge serves both modes of its exit), losses averaged. As R5b otherwise (r2_train.py): importance weights, text-only
product-of-experts bias during training only, class-balanced syllable-type means shrunk to the global mean with a
learned share β per exit and layer, AdamW 3e-4 (β 1e-2), 12 epochs, EMA (0.999 per step) after the last epoch, no
selection on development data.

    python research/scripts/nested_judge_train.py --feat /content/bench/r9f --weights cv_train_weights.npz \\
        --propensity /content/bench/s1/textonly/textonly_syl.json --sets-dir /content/bench/sets --out /content/bench/r9
"""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from bench_eval import ctrl_auc  # noqa: E402
from twjudge.accent import neural as N  # noqa: E402
from twjudge.accent.nested import NestedJudge  # noqa: E402
from twjudge.accent.prosodic import auc, length_weights  # noqa: E402

TAGS = ["v4", "v8", "v12", "b8", "b12"]
EVAL_SETS = ["cv_dev", "cv_test", "cv_heldout", "kespeech", "breezy", "cosy300m", "ming_b0",
             "synth2_breezy", "synth2_cosy", "cv_heldout_rs_vocos", "cv_heldout_rs_dac", "r10_sealed"]
SHRINK = 20.0


class Data:
    def __init__(self, fd: Path, seg: str, name: str, n_exits: int, vocab):
        f = np.load(fd / seg / f"feat_{name}.npz")
        self.ids = [str(i) for i in f["ids"]]
        self.lens = f["lens"]
        self.off = np.r_[0, np.cumsum(self.lens)]
        self.ctx = f["ctx"].astype(np.float32)
        self.types = np.array([vocab.get(t, 0) for t in f["types"]], np.int64)
        self.y = np.array([i.startswith("tw") for i in self.ids], np.float32)
        self.F = {(e, t): np.load(fd / seg / f"N{e + 1}{t}_{name}.npy", mmap_mode="r") for e in range(n_exits) for t in TAGS}
        self.pos = {i: k for k, i in enumerate(self.ids)}

    def batch(self, idx, e, dev):
        lens = self.lens[idx]
        S = int(lens.max())
        rows = self.off[idx][:, None] + np.arange(S)[None]
        valid = np.arange(S)[None] < lens[:, None]
        flat = np.where(valid, rows, 0).reshape(-1)
        order = np.argsort(flat)                      # sorted reads from the memory-mapped arrays
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)
        feats = []
        for tag in TAGS:
            g = np.empty((len(flat), self.F[(e, tag)].shape[1]), np.float16)
            g[order] = self.F[(e, tag)][flat[order]]
            feats.append(t(g.reshape(len(idx), S, -1)))
        return {"ctx": t(self.ctx[flat].reshape(len(idx), S, -1)), "mask": t(~valid),
                "types": t(np.where(valid, self.types[flat].reshape(len(idx), S), 0)), "feats": feats}


def syllable_means(d: Data, e):
    ysyl = np.repeat(d.y, d.lens).astype(bool)
    V = int(d.types.max()) + 1
    out = {}
    for tag in TAGS:
        x = np.asarray(d.F[(e, tag)], dtype=np.float32)
        glob = 0.5 * (x[ysyl].mean(0) + x[~ysyl].mean(0))
        sums, cnts = [], []
        for c in (True, False):
            m = ysyl == c
            s = np.zeros((V, x.shape[1]), np.float32)
            np.add.at(s, d.types[m], x[m])
            sums.append(s); cnts.append(np.bincount(d.types[m], minlength=V).astype(np.float32))
        both = (cnts[0] > 0) & (cnts[1] > 0)
        mt = np.where(both[:, None], 0.5 * (sums[0] / np.maximum(cnts[0], 1)[:, None] + sums[1] / np.maximum(cnts[1], 1)[:, None]),
                      (sums[0] + sums[1]) / np.maximum(cnts[0] + cnts[1], 1)[:, None])
        n = (cnts[0] + cnts[1])[:, None]
        mu = (n * mt + SHRINK * glob) / (n + SHRINK)
        mu[0] = glob
        out[tag] = mu
        del x
    return out


def predict(net, d: Data, e, dev, bs=128):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(d.ids), bs):
            b = d.batch(np.arange(s, min(s + bs, len(d.ids))), e, dev)
            out.append(N.utt_logits(net(b, e), b["mask"]).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--propensity", required=True)
    ap.add_argument("--sets-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fd, out = Path(a.feat), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    E = a.exits
    base = np.load(fd / "forced" / "feat_cv_train.npz")
    vocab = {t: k + 1 for k, t in enumerate(sorted(set(base["types"].tolist())))}
    tr = {m: Data(fd, seg, "cv_train", E, vocab) for m, seg in (("t", "forced"), ("f", "free"))}
    dv = {m: Data(fd, seg, "cv_dev", E, vocab) for m, seg in (("t", "forced"), ("f", "free"))}
    print(f"loaded: train {len(tr['t'].ids)} forced / {len(tr['f'].ids)} free, {len(vocab)} syllable types, "
          f"{time.time() - t0:.0f}s", flush=True)
    W = np.load(a.weights)
    wmap = dict(zip([str(i) for i in W["ids"]], W["w"].astype(np.float32)))
    wt = lambda ids: np.array([wmap.get(i, 0.0) for i in ids], np.float32)

    tdat = lambda d: {"ids": d.ids, "y": d.y, "lens": d.lens, "ctx": d.ctx, "F": np.zeros((len(d.ctx), 1), np.float16)}
    bias_net = N.train_net(N.SyllableNet(None), tdat(tr["t"]), wt(tr["t"].ids), epochs=6, use_feat=False, dev_data=tdat(dv["t"]))
    bias = dict(zip(tr["t"].ids, N.predict(bias_net, tdat(tr["t"]), use_feat=False)))
    mus = {e: syllable_means(tr["t"], e) for e in range(E)}
    print(f"means {time.time() - t0:.0f}s", flush=True)
    dims = {t: tr["t"].F[(0, t)].shape[1] for t in TAGS}
    net = NestedJudge(E, TAGS, dims, mus).to(dev)
    ema = copy.deepcopy(net).eval()
    sc = {id(p) for p in net.scalars()}
    opt = torch.optim.AdamW([{"params": [p for p in net.parameters() if id(p) not in sc], "lr": 3e-4, "weight_decay": 0.01},
                             {"params": net.scalars(), "lr": 1e-2, "weight_decay": 0.0}])
    V = [(e, m) for e in range(E) for m in "tf"]
    prop = json.load(open(a.propensity))
    p_dv = np.array([prop["cv_dev"].get(i, [0.0])[0] for i in dv["t"].ids])
    log, bs = [], 64
    for ep in range(a.epochs):
        net.train()
        t1 = time.time()
        order = rng.permutation(len(tr["t"].ids))
        for s0 in range(0, len(order), bs):
            chosen = order[s0:s0 + bs]
            terms = []
            for e, m in V:
                if m == "t":
                    idx = chosen
                else:
                    idx = np.array([r for r in (tr["f"].pos.get(tr["t"].ids[j]) for j in chosen) if r is not None])
                    if len(idx) == 0:
                        continue
                d = tr[m]
                b = d.batch(idx, e, dev)
                z = N.utt_logits(net(b, e), b["mask"])
                ids = [d.ids[j] for j in idx]
                zb = z + torch.tensor([bias.get(i, 0.0) for i in ids], device=dev)
                y = torch.tensor(d.y[idx], device=dev)
                w = torch.tensor(wt(ids), device=dev)
                terms.append((nn.functional.binary_cross_entropy_with_logits(zb, y, reduction="none") * w).mean())
            loss = sum(terms) / len(terms)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
        rec = {"epoch": ep, "s": round(time.time() - t1), "beta": torch.sigmoid(net.beta_logit).round(decimals=2).tolist()}
        for e, m in V:
            d = dv[m]
            z = predict(ema, d, e, dev)
            rec[f"N{e + 1}{m}"] = round(ctrl_auc(d.y, z, np.array([prop["cv_dev"].get(i, [0.0])[0] for i in d.ids]), d.lens)
                                        if m == "t" else auc(d.y, z, length_weights(d.lens, d.y.astype(int))), 4)
        log.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(log, open(out / "train_log.json", "w"), indent=1)
    torch.save({"state": ema.state_dict(), "vocab": vocab, "tags": TAGS, "dims": dims, "exits": E, "seed": a.seed,
                "epochs": a.epochs}, out / "judge.pt")
    json.dump({f"N{e + 1}": ema.size(e) for e in range(E)}, open(out / "judge_sizes.json", "w"))

    nchar = {}

    def n_of(s, i):
        if a.sets_dir is None:
            return 0
        if s not in nchar:
            p = Path(a.sets_dir) / f"{s}.jsonl"
            nchar[s] = {json.loads(l)["item_id"]: len(re.sub(r"[^㐀-鿿]", "", json.loads(l)["text"])) for l in open(p)} if p.exists() else {}
        return nchar[s].get(i, 0)

    res = {f"N{e + 1}{m}": {} for e, m in V}
    for s in EVAL_SETS:
        for m, seg in (("t", "forced"), ("f", "free")):
            if not (fd / seg / f"feat_{s}.npz").exists():
                continue
            d = Data(fd, seg, s, E, vocab)
            for e in range(E):
                z = predict(ema, d, e, dev)
                res[f"N{e + 1}{m}"][s] = {i: [float(x), n_of(s, i)] for i, x in zip(d.ids, z)}
        print(f"scored {s} {time.time() - t0:.0f}s", flush=True)
    for k, r in res.items():
        json.dump(r, open(out / "scores" / f"{k}.json", "w"))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
