"""Encoder probes: Accent Detective's audio readout on one frozen encoder (base or large), which isolates the effect of encoder scale.

The released readout (ASP, twjudge.accent.nested.ASPReadout) on one frozen encoder at a time, three layers at a third,
two thirds and all of its depth (base: 4 8 12, large: 8 16 24), everything else as train_readouts.py's audio readout:
utterance mean removed per layer, cv_train with importance weights, the text-context-only product-of-experts bias
during training only, 12 epochs, AdamW 3e-4, EMA 0.999, no selection.

The encoder runs once per clip (fp16) and its states are kept on disk, since a large encoder per training batch would be
too slow. Writes <out>/scores/<name>.json (score_e2e format) and <out>/log_<name>.json.

    python research/scripts/encoder_probe.py --encoder TencentGameMate/chinese-hubert-large --name hubert_large \\
        --layers 8 16 24 --bundle /content/bench --weights /content/cv_train_weights.npz --work /content/feat --out /content/r15
"""
import argparse
import copy
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

SETS = ["cv_train", "cv_dev", "cv_test", "cv_heldout", "kespeech", "breezy", "cosy300m", "ming_b0", "synth2_breezy",
        "synth2_cosy", "r10_sealed"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--work", required=True, help="disk for the cached states (deleted at the end)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import librosa
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, HubertModel, Wav2Vec2Model

    from twjudge.accent import neural as N
    from twjudge.accent.nested import ASPReadout, text_syllables
    from twjudge.accent.prosodic import auc
    from twjudge.align.ctc import CTCAligner
    from twjudge.lexicon.readings import TaiwanReadings

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    B, out, work = Path(a.bundle), Path(a.out), Path(a.work) / a.name
    (out / "scores").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(a.encoder)
    cls = HubertModel if cfg.model_type == "hubert" else Wav2Vec2Model
    enc = cls.from_pretrained(a.encoder).to(dev).eval()
    for q in enc.parameters():
        q.requires_grad_(False)
    dim, L = cfg.hidden_size, len(a.layers)
    log = {"encoder": a.encoder, "layers": a.layers, "params": sum(p.numel() for p in enc.parameters())}

    def load_set(name):
        p = B / "sets" / f"{name}.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(l) for l in open(p)]
        return rows[: a.limit] if a.limit else rows

    # ---- the encoder once per clip: states of the chosen layers, fp16, concatenated per set on disk
    t0 = time.time()
    rows_of, off_of = {}, {}
    for s in SETS:
        rows = load_set(s)
        if not rows:
            continue
        rows_of[s] = rows
        if (work / f"{s}.off.npy").exists():
            off_of[s] = np.load(work / f"{s}.off.npy")
            continue
        with ThreadPoolExecutor(8) as ex:
            waves = list(ex.map(lambda r: librosa.load(str(B / r["audio"]), sr=16000, mono=True)[0], rows))
        n_frames = [max(1, (len(w) - 400) // 320 + 1) for w in waves]
        off = np.r_[0, np.cumsum(n_frames)]
        mm = np.lib.format.open_memmap(work / f"{s}.npy", mode="w+", dtype=np.float16, shape=(int(off[-1]), L, dim))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            for k, w in enumerate(waves):
                x = torch.tensor(w, device=dev)[None]
                x = (x - x.mean()) / (x.std() + 1e-7)
                hs = enc(x, output_hidden_states=True).hidden_states
                H = torch.stack([hs[l][0] for l in a.layers], 1)               # (T, L, dim)
                T = int(off[k + 1] - off[k])
                H = H[:T] if H.shape[0] >= T else torch.nn.functional.pad(H, (0, 0, 0, 0, 0, T - H.shape[0]))
                mm[off[k]:off[k + 1]] = H.float().cpu().numpy().astype(np.float16)
        mm.flush(); del mm
        np.save(work / f"{s}.off.npy", off)
        off_of[s] = off
        print(f"states {s}: {len(rows)} clips, {off[-1]} frames, {time.time() - t0:.0f}s", flush=True)
    del enc
    torch.cuda.empty_cache()
    feats = {s: np.load(work / f"{s}.npy", mmap_mode="r") for s in off_of}

    def batch(s, idx):
        off = off_of[s]
        T = int(max(off[i + 1] - off[i] for i in idx))
        x = np.zeros((len(idx), T, L, dim), np.float16)
        m = np.ones((len(idx), T), bool)
        for j, i in enumerate(idx):
            h = np.asarray(feats[s][off[i]:off[i + 1]], np.float32)
            x[j, :len(h)] = h - h.mean(0, keepdims=True)                        # utterance mean removed per layer
            m[j, :len(h)] = False
        return torch.tensor(x, device=dev), torch.tensor(m, device=dev)

    # ---- the text-context-only bias (as train_readouts.py)
    aligner = CTCAligner(TaiwanReadings(ROOT / "data/lexicon/concised.json.xz"))
    vocab = {}

    def ctx_data(rows):
        ids = [r["item_id"] for r in rows]
        ctx = []
        for r in rows:
            sy = text_syllables(aligner, r["text"], vocab)
            ctx.append(np.array(sy[1], np.float32)[:, [0, 1, 2, 5, 6]] if sy else np.zeros((1, 5), np.float32))
        return {"ids": ids, "y": np.array([i.startswith("tw") for i in ids], np.float32), "lens": np.array([len(c) for c in ctx]),
                "ctx": np.concatenate(ctx), "F": np.zeros((sum(map(len, ctx)), 1), np.float16)}

    tr = rows_of["cv_train"]
    ids = [r["item_id"] for r in tr]
    y = np.array([i.startswith("tw") for i in ids], np.float32)
    W = np.load(a.weights)
    wmap = dict(zip([str(i) for i in W["ids"]], W["w"].astype(np.float32)))
    w = np.array([wmap.get(i, 0.0) for i in ids], np.float32)
    td, dd = ctx_data(tr), ctx_data(rows_of["cv_dev"])
    bias_net = N.train_net(N.SyllableNet(None), td, w, epochs=6, use_feat=False, dev_data=dd)
    bias = N.predict(bias_net, td, use_feat=False).astype(np.float32)

    # ---- ASP on the cached states
    net = ASPReadout(n_layers=L, dim=dim).to(dev)
    ema = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    lens = np.diff(off_of["cv_train"])
    order_len = np.argsort(lens)
    ep_log = []
    for ep in range(a.epochs):
        net.train()
        t1, tot, nb = time.time(), 0.0, 0
        chunks = [order_len[k:k + 32] for k in range(0, len(order_len), 32)]
        for c in [chunks[k] for k in rng.permutation(len(chunks))]:
            x, m = batch("cv_train", c)
            z = net(x, m) + torch.tensor(bias[c], device=dev)
            loss = (nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(y[c], device=dev), reduction="none")
                    * torch.tensor(w[c], device=dev)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
            tot += loss.item(); nb += 1
        ep_log.append({"epoch": ep, "s": round(time.time() - t1), "loss": round(tot / nb, 4)})
        print(json.dumps(ep_log[-1]), flush=True)
    log["epochs"] = ep_log

    # ---- scores
    res = {}
    with torch.no_grad():
        for s, rows in rows_of.items():
            if s == "cv_train":
                continue
            order = np.argsort(np.diff(off_of[s]))
            z = np.zeros(len(rows))
            for k in range(0, len(rows), 64):
                idx = order[k:k + 64]
                x, m = batch(s, idx)
                z[idx] = ema(x, m).float().cpu().numpy()
            res[s] = {r["item_id"]: [float(v), len(re.sub(r"[^㐀-鿿]", "", r["text"]))] for r, v in zip(rows, z)}
            if s in ("cv_dev", "cv_heldout", "r10_sealed"):
                yy = np.array([r["item_id"].startswith("tw") for r in rows]).astype(int)
                log[f"auc_{s}"] = float(auc(yy, z))
                print(f"{s} AUC {log[f'auc_{s}']:.4f}", flush=True)
    json.dump(res, open(out / "scores" / f"{a.name}.json", "w"))
    json.dump(log, open(out / f"log_{a.name}.json", "w"), indent=1)
    print(f"done {a.name} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
