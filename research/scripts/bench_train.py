"""Train judges B / C with and without the speaker adversary, over seeds, and score every set.

Uses the benchmark folder (bench_bundle.py + bench_features.py --kind syllable): feat_/ssl_/w2v_
<set>.npz. The text-only bias model depends on the text context only, so it is trained once and
shared by all runs. For each run it writes the log-odds of every set and the utterance
representations of the speaker-probe sets (cv_test, ming_b0).

    python research/scripts/bench_train.py --feat /content/bench/feat --sets-dir /content/bench/sets \
        --out /content/bench/runs --judges b c --adv 0 0.3 1 --seeds 0 1 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))

PROBE = ("cv_test", "ming_b0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--sets-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", nargs="+", default=["b", "c"])
    ap.add_argument("--adv", nargs="+", type=float, default=[0.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--reference", nargs="*", default=[], help="existing judge checkpoints to score as well")
    # ablations
    ap.add_argument("--feat-pre", default="feat", help="prefix of the context / prosodic feature files")
    ap.add_argument("--ssl-pre", nargs=2, default=["ssl", "w2v"], help="SSL feature prefixes of judges B and C")
    ap.add_argument("--variant", default="", help="tag suffix of an ablation")
    ap.add_argument("--no-ctx", action="store_true", help="main network without the text context")
    ap.add_argument("--no-bias", action="store_true", help="no text-only product-of-experts model")
    ap.add_argument("--no-weights", action="store_true", help="no length / text-context importance weights")
    ap.add_argument("--layers", help="read per-layer feature files L<l>_<set>.npz (compact_features.py), e.g. 2,4")
    ap.add_argument("--kd", help="teacher per-syllable log-odds on cv_train (npz: ids, lens, z) for distillation")
    ap.add_argument("--kd-weight", type=float, default=0.1)
    ap.add_argument("--nested", action="store_true",
                    help="Matryoshka judge over the --layers prefixes (NestedSyllableNet); one logits file per prefix")
    ap.add_argument("--aug-feat", help="feature dir of augmented cv_train clips (compact_features.py --augment), "
                                       "added to the training set (items suffixed #aug)")
    a = ap.parse_args()
    import torch

    from twjudge.accent import neural as N

    PREFIX = dict(zip("bc", a.ssl_pre))
    fd, out = Path(a.feat), Path(a.out)

    def load(s, pre, real_ctx=False):
        return load_dir(fd, s, pre, real_ctx)

    def load_dir(fd, s, pre, real_ctx=False):
        if a.layers:
            f = np.load(fd / f"{a.feat_pre}_{s}.npz")
            parts = [np.load(fd / f"L{l}_{s}.npz") for l in a.layers.split(",")]
            assert all(list(q["ids"]) == list(f["ids"]) for q in parts), "feature files out of order"
            ids = [str(i) for i in f["ids"]]
            cols = [N.SYL_NAMES.index(c) for c in N.CTX]
            d = {"ids": ids, "y": np.array([i.startswith("tw") for i in ids], np.float32), "lens": f["lens"],
                 "ctx": f["X"][:, cols].astype(np.float32), "X": f["X"],
                 "F": np.concatenate([q["F"] for q in parts], axis=1)}
        else:
            d = N.load_pair(fd / f"{a.feat_pre}_{s}.npz", fd / f"{pre}_{s}.npz")
        if a.no_ctx and not real_ctx:
            d = {**d, "ctx": np.zeros_like(d["ctx"])}
        if s == "cv_train" and a.aug_feat and fd != Path(a.aug_feat):
            e = load_dir(Path(a.aug_feat), s, pre, real_ctx)
            d = {k: (list(d[k]) + [f"{i}#aug" for i in e[k]] if k == "ids" else np.concatenate([d[k], e[k]]))
                 for k in ("ids", "y", "lens", "ctx", "X", "F")}
        return d
    out.mkdir(parents=True, exist_ok=True)
    sets = sorted(p.stem for p in Path(a.sets_dir).glob("*.jsonl") if (fd / f"{a.feat_pre}_{p.stem}.npz").exists())
    spk_of = {}
    for l in open(Path(a.sets_dir) / "cv_train.jsonl"):
        r = json.loads(l)
        spk_of[r["item_id"]] = r["speaker"]

    def score_all(net, pre, tag):
        if getattr(net, "n_prefix", 0):                   # a nested judge: score every prefix
            for k in range(1, net.n_prefix + 1):
                score_one(net, pre, f"{tag}_P{k}", k)
            return
        score_one(net, pre, tag, None)

    def score_one(net, pre, tag, k):
        f = out / f"{tag}_logits.json"
        res = json.load(open(f)) if f.exists() else {}
        for s in sets:
            if s in res:
                continue
            d = load(s, pre)
            if s in PROBE and k is None:
                z, h = N.predict(net, d, states=True)
                np.save(out / f"{tag}_states_{s}.npy", h.astype(np.float16))
            else:
                z = N.predict(net, d, k=k)
            res[s] = dict(zip(d["ids"], map(float, z)))
        json.dump(res, open(out / f"{tag}_logits.json", "w"))

    for ref in a.reference:
        ck = torch.load(ref, map_location="cpu")
        net = N.SyllableNet(ck["d_in"]); net.load_state_dict(ck["state"])
        score_all(net, ck.get("ssl", "ssl"), "ref_" + Path(ref).stem)
        print("scored reference", ref, flush=True)

    n_nest = len(a.layers.split(",")) if a.nested else 0
    if a.nested and not a.layers:
        raise SystemExit("--nested needs --layers")

    def make_net(d_in, nested):
        return N.NestedSyllableNet(d_in // nested, nested) if nested else N.SyllableNet(d_in)

    bias_tr = None
    for j in a.judges:
        pre = PREFIX[j]
        tr, dv = load("cv_train", pre), load("cv_dev", pre)
        w = np.ones(len(tr["y"]), np.float32) if a.no_weights else N.utterance_weights(tr)
        if a.no_bias:
            bias_tr = False
        if bias_tr is None:                         # the text-only model always sees the real text context
            trb, dvb = (load("cv_train", pre, True), load("cv_dev", pre, True)) if a.no_ctx else (tr, dv)
            if (out / "textonly.pt").exists():
                bias_net = N.SyllableNet(None); bias_net.load_state_dict(torch.load(out / "textonly.pt"))
            else:
                t0 = time.time()
                bias_net = N.train_net(N.SyllableNet(None), trb, w, epochs=6, use_feat=False, dev_data=dvb)
                torch.save({k: v.cpu() for k, v in bias_net.state_dict().items()}, out / "textonly.pt")
                print(f"bias model {time.time() - t0:.0f}s", flush=True)
            bias_tr = N.predict(bias_net, trb, use_feat=False)
            if not a.no_ctx:
                score_all(bias_net, pre, "textonly")
        soft = None
        if a.kd:                                     # teacher syllable log-odds, matched by item and length
            t = np.load(a.kd)
            toff = np.r_[0, np.cumsum(t["lens"])]
            tpos = {str(i): k for k, i in enumerate(t["ids"])}
            soft = np.zeros(int(tr["lens"].sum()), np.float32)
            o = 0
            for i, n_ in zip(tr["ids"], tr["lens"]):
                k = tpos[i]
                assert t["lens"][k] == n_, f"teacher alignment differs for {i}"
                soft[o:o + n_] = t["z"][toff[k]:toff[k + 1]]
                o += n_
        names = sorted(set(spk_of[i.split("#")[0]] for i in tr["ids"]))
        spk = np.array([names.index(spk_of[i.split("#")[0]]) for i in tr["ids"]])
        for adv in a.adv:
            for seed in a.seeds:
                tag = f"{j}_adv{adv:g}_s{seed}{a.variant}"
                if (out / f"{tag}.pt").exists():          # trained earlier: only score new sets
                    ck = torch.load(out / f"{tag}.pt", map_location="cpu")
                    net = make_net(ck["d_in"], ck.get("nested", 0)); net.load_state_dict(ck["state"])
                    score_all(net, pre, tag)
                    continue
                t0 = time.time()
                net = N.train_net(make_net(tr["F"].shape[1], n_nest), tr, w, bias=bias_tr if bias_tr is not False else None,
                                  epochs=12, dev_data=dv,
                                  seed=seed, spk=spk, adv=adv, soft=soft, kd=a.kd_weight if a.kd else 0.0)
                torch.save({"d_in": tr["F"].shape[1], "nested": n_nest, "state": {k: v.cpu() for k, v in net.state_dict().items()},
                            "ctx": N.CTX, "ssl": pre, "adv": adv, "seed": seed, "variant": a.variant}, out / f"{tag}.pt")
                score_all(net, pre, tag)
                print(f"{tag} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
