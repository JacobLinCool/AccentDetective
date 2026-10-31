"""Readouts of one exit of a frozen nested trunk: no text (ASP) and with text, trained together.

  ASP    attentive statistics pooling: cv_train, importance weights, 12 epochs, EMA 0.999, no selection; the product-of-experts
         bias of a text-context-only model (tone, sandhi, Mainland-neutral, final, position) during training only. The
         context now comes from the text itself (no alignment), so "final" means the last syllable rather than a pause.
  text   TextReadout (twjudge.accent.nested) on ASP's frame projection, detached: its log-odds is ASP's (detached) plus the
         correction. Its product-of-experts bias is a text-only model with the same text inputs (TextOnly), fitted out of
         fold (two folds split by the text, so no sentence predicts itself), so the correction is pushed to what the audio
         adds to the text, not to what the text alone predicts.

Scores every evaluation set in both modes, N<k>a (no text) and N<k>x (text; a clip whose text gives no syllables keeps its
no-text score), plus the channels.py conditions of cv_heldout and r10_sealed as sets "<set>@<condition>".
Writes <out>/scores/N<k>{a,x}.json, <out>/readout_N<k>.pt, <out>/readout_log_N<k>.json.

    python research/scripts/train_readouts.py --nested /content/bench/r14_H_s0/student.pt --bundle /content/bench \\
        --judge /content/bench/nested/judge.pt --weights /content/cv_train_weights.npz --chan /content/bench/chan \\
        --exit 4 --out /content/bench/r14_H_s0
"""
import argparse
import copy
import json
import re
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

SETS = ["cv_dev", "cv_test", "cv_heldout", "kespeech", "breezy", "cosy300m", "ming_b0", "synth2_breezy",
        "synth2_cosy", "cv_heldout_rs_vocos", "cv_heldout_rs_dac", "r10_sealed"]
CHAN_SETS = ["cv_heldout", "r10_sealed"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nested", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--judge", required=True, help="the nested judges' checkpoint (judge.pt): the syllable-type vocabulary")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--exit", type=int, required=True, help="1-based exit of the trunk")
    ap.add_argument("--chan", help="channels.py assets dir; without it the deployment conditions are skipped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import librosa
    import torch
    import torch.nn as nn

    from twjudge.accent import neural as N
    from twjudge.accent.nested import ASPReadout, TextOnly, TextReadout, build_student, text_syllables, voiced_span
    from twjudge.accent.prosodic import auc, length_weights
    from twjudge.align.ctc import CTCAligner
    from twjudge.lexicon.readings import TaiwanReadings

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    B, out = Path(a.bundle), Path(a.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    k_exit = a.exit
    ck = torch.load(a.nested, map_location="cpu", weights_only=False)
    trunk = build_student(ck["config"]); trunk.load_state_dict(ck["state"]); trunk = trunk.to(dev).eval()
    tags = list(trunk.targets)
    for q in trunk.parameters():
        q.requires_grad_(False)
    vocab = torch.load(a.judge, map_location="cpu", weights_only=False)["vocab"]
    n_types = max(vocab.values()) + 1
    aligner = CTCAligner(TaiwanReadings(ROOT / "data/lexicon/concised.json.xz"))
    log = {"exit": k_exit, "exits": trunk.exits}

    def load_set(name):
        p = B / "sets" / f"{name}.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(l) for l in open(p)]
        return rows[: a.limit] if a.limit else rows

    def wav_of(r):
        return librosa.load(str(B / r["audio"]), sr=16000, mono=True)[0]

    def syl_of(texts):
        out_, n_ok = [], 0
        for t in texts:
            s = text_syllables(aligner, t, vocab)
            n_ok += s is not None
            out_.append(s)
        return out_, n_ok

    @torch.no_grad()
    def frames(w):
        o = trunk(torch.tensor(w, device=dev)[None], upto=k_exit)[-1]
        H = torch.stack([o[t][0].float() for t in tags])                # (L, T, 768)
        return (H - H.mean(1, keepdim=True)).transpose(0, 1).half()      # (T, L, 768), utterance mean removed

    def batch(Ws, Sy, idx):
        Fs = {i: frames(Ws[i].astype(np.float32) / 32767 if Ws[i].dtype == np.int16 else Ws[i]) for i in idx}
        T = max(Fs[i].shape[0] for i in idx)
        x = torch.zeros((len(idx), T, len(tags), 768), dtype=torch.float16, device=dev)
        m = torch.ones((len(idx), T), dtype=torch.bool, device=dev)
        span = torch.zeros((len(idx), 2), dtype=torch.long, device=dev)
        S = max(len(Sy[i][0]) if Sy[i] else 1 for i in idx)
        ty = torch.zeros((len(idx), S), dtype=torch.long, device=dev)
        cx = torch.zeros((len(idx), S, 7), dtype=torch.float32, device=dev)
        sm = torch.ones((len(idx), S), dtype=torch.bool, device=dev)
        has = torch.zeros(len(idx), device=dev)
        for j, i in enumerate(idx):
            n = Fs[i].shape[0]
            x[j, :n] = Fs[i]; m[j, :n] = False
            w = Ws[i].astype(np.float32) / 32767 if Ws[i].dtype == np.int16 else Ws[i]
            span[j] = torch.tensor(voiced_span(w, n))
            if Sy[i]:
                t_, c_ = Sy[i]
                ty[j, :len(t_)] = torch.tensor(t_); cx[j, :len(t_)] = torch.tensor(c_); sm[j, :len(t_)] = False; has[j] = 1
            else:
                sm[j, 0] = False                         # one blank syllable, so attention has a query; its output is unused
        return x, m, span, ty, cx, sm, has

    # ---- cv_train in memory (int16) and its text
    t0 = time.time()
    tr = load_set("cv_train")
    with ThreadPoolExecutor(8) as ex:
        waves = [(x * 32767).astype(np.int16) for x in ex.map(wav_of, tr)]
    ids = [r["item_id"] for r in tr]
    texts = [r["text"] for r in tr]
    y = np.array([i.startswith("tw") for i in ids], np.float32)
    W = np.load(a.weights)
    wmap = dict(zip([str(i) for i in W["ids"]], W["w"].astype(np.float32)))
    w = np.array([wmap.get(i, 0.0) for i in ids], np.float32)
    sy, n_ok = syl_of(texts)
    log["train_clips"], log["train_with_syllables"] = len(ids), n_ok
    print(f"cv_train: {len(waves)} clips, {n_ok} with syllables, {time.time() - t0:.0f}s", flush=True)

    # ---- ASP's bias: text context only (a text-only model on the text's own context)
    def ctx_data(rows_ids, syls):
        ctx = [np.array(s[1], np.float32)[:, [0, 1, 2, 5, 6]] if s else np.zeros((1, 5), np.float32) for s in syls]
        return {"ids": rows_ids, "y": np.array([i.startswith("tw") for i in rows_ids], np.float32),
                "lens": np.array([len(c) for c in ctx]), "ctx": np.concatenate(ctx), "F": np.zeros((sum(map(len, ctx)), 1), np.float16)}

    dv_rows = load_set("cv_dev")
    dv_sy, _ = syl_of([r["text"] for r in dv_rows])
    td, dd = ctx_data(ids, sy), ctx_data([r["item_id"] for r in dv_rows], dv_sy)
    bias_net = N.train_net(N.SyllableNet(None), td, w, epochs=6, use_feat=False, dev_data=dd)
    bias_a = N.predict(bias_net, td, use_feat=False).astype(np.float32)

    # ---- the text readout's bias: text only, the same inputs, out of fold (folds by text)
    fold = np.array([zlib.crc32(t.encode()) % 2 for t in texts])
    bias_x = np.zeros(len(ids), np.float32)
    has_tr = np.array([s is not None for s in sy])

    def text_batch(idx):
        S = max(len(sy[i][0]) for i in idx)
        ty = torch.zeros((len(idx), S), dtype=torch.long); cx = torch.zeros((len(idx), S, 7)); sm = torch.ones((len(idx), S), dtype=torch.bool)
        for j, i in enumerate(idx):
            t_, c_ = sy[i]
            ty[j, :len(t_)] = torch.tensor(t_); cx[j, :len(t_)] = torch.tensor(c_); sm[j, :len(t_)] = False
        return ty.to(dev), cx.to(dev), sm.to(dev)

    for f in (0, 1):
        fit = np.where(has_tr & (fold != f))[0]
        net = TextOnly(n_types).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
        for _ in range(6):
            net.train()
            perm = rng.permutation(fit)
            for s in range(0, len(perm), 64):
                idx = perm[s:s + 64]
                z = net(*text_batch(idx))
                loss = (nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(y[idx], device=dev), reduction="none")
                        * torch.tensor(w[idx], device=dev)).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        pred = np.where(has_tr & (fold == f))[0]
        with torch.no_grad():
            for s in range(0, len(pred), 256):
                idx = pred[s:s + 256]
                bias_x[idx] = net(*text_batch(idx)).float().cpu().numpy()
    k_ = has_tr
    log["text_only_oof_auc"] = float(auc(y[k_], bias_x[k_]))
    log["context_only_train_auc"] = float(auc(y, bias_a))
    print(f"biases: context-only train AUC {log['context_only_train_auc']:.3f}, text-only out-of-fold AUC "
          f"{log['text_only_oof_auc']:.3f}", flush=True)

    # ---- the two readouts
    asp, txt = ASPReadout().to(dev), TextReadout(n_types).to(dev)
    ema_a, ema_x = copy.deepcopy(asp).eval(), copy.deepcopy(txt).eval()
    opt = torch.optim.AdamW(list(asp.parameters()) + list(txt.parameters()), lr=3e-4, weight_decay=0.01)
    order_len = np.argsort([len(x) for x in waves])
    ep_log = []
    for ep in range(a.epochs):
        asp.train(); txt.train()
        t1 = time.time()
        la_s = lx_s = 0.0
        nb = 0
        chunks = [order_len[k:k + 32] for k in range(0, len(waves), 32)]
        for c in [chunks[k] for k in rng.permutation(len(chunks))]:
            x, m, span, ty, cx, sm, has = batch(waves, sy, c)
            yt, wt = torch.tensor(y[c], device=dev), torch.tensor(w[c], device=dev)
            h = asp.frames(x)
            za = asp(x, m, h=h)
            dx = txt(h.detach(), m, span, ty, cx, sm)
            la = (nn.functional.binary_cross_entropy_with_logits(za + torch.tensor(bias_a[c], device=dev), yt, reduction="none") * wt).mean()
            lx = (nn.functional.binary_cross_entropy_with_logits(za.detach() + dx + torch.tensor(bias_x[c], device=dev), yt,
                                                                 reduction="none") * wt * has).sum() / has.sum().clamp(min=1)
            opt.zero_grad(); (la + lx).backward()
            nn.utils.clip_grad_norm_(asp.parameters(), 1.0); nn.utils.clip_grad_norm_(txt.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for e_, n_ in ((ema_a, asp), (ema_x, txt)):
                    for pe, pn in zip(e_.parameters(), n_.parameters()):
                        pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
            la_s += la.item(); lx_s += lx.item(); nb += 1
        ep_log.append({"epoch": ep, "s": round(time.time() - t1), "asp": round(la_s / nb, 4), "text": round(lx_s / nb, 4),
                       "sigma": [round(float(v), 3) for v in txt.log_sigma.exp()]})
        print(json.dumps(ep_log[-1]), flush=True)
    log["epochs"] = ep_log
    del waves
    torch.save({"exit": k_exit, "exits": trunk.exits, "asp": {k: v.cpu() for k, v in ema_a.state_dict().items()},
                "text": {k: v.cpu() for k, v in ema_x.state_dict().items()}, "n_types": n_types},
               out / f"readout_N{k_exit}.pt")

    # ---- scoring, both modes
    A = None
    if a.chan:
        from channels import CONDS, load_assets, perturb

        A = load_assets(a.chan)
    res_a, res_x, delta = {}, {}, {}
    jobs = [(s, None) for s in SETS] + ([(s, c) for s in CHAN_SETS for c in CONDS] if A is not None else [])
    for s, cond in jobs:
        rows = load_set(s)
        if not rows:
            continue
        with ThreadPoolExecutor(8) as ex:
            ws = list(ex.map(wav_of, rows))
            if cond is not None:
                ws = list(ex.map(lambda p: perturb(p[0], cond, p[1]["item_id"], A), zip(ws, rows)))
        ss, _ = syl_of([r["text"] for r in rows])
        za_all, dx_all, has_all = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
        order = np.argsort([len(x) for x in ws])
        with torch.no_grad():
            for k in range(0, len(ws), 48):
                idx = order[k:k + 48].tolist()
                x, m, span, ty, cx, sm, has = batch(ws, ss, idx)
                h = ema_a.frames(x)
                za = ema_a(x, m, h=h)
                dx = ema_x(h, m, span, ty, cx, sm) * has
                za_all[idx] = za.float().cpu().numpy(); dx_all[idx] = dx.float().cpu().numpy(); has_all[idx] = has.cpu().numpy()
        name = s if cond is None else f"{s}@{cond}"
        nch = [len(re.sub(r"[^㐀-鿿]", "", r["text"])) for r in rows]
        res_a[name] = {r["item_id"]: [float(v), n] for r, v, n in zip(rows, za_all, nch)}
        res_x[name] = {r["item_id"]: [float(v + d), n] for r, v, d, n in zip(rows, za_all, dx_all, nch)}
        delta[name] = {"with_text": float(has_all.mean()), "mean_abs_correction": float(np.abs(dx_all[has_all > 0]).mean()) if has_all.any() else 0.0}
        print(f"scored {name} {time.time() - t0:.0f}s", flush=True)
    json.dump(res_a, open(out / "scores" / f"N{k_exit}a.json", "w"))
    json.dump(res_x, open(out / "scores" / f"N{k_exit}x.json", "w"))
    log["correction"] = delta
    log["params"] = {"asp": sum(p.numel() for p in asp.parameters()), "text": sum(p.numel() for p in txt.parameters())}
    json.dump(log, open(out / f"readout_log_N{k_exit}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
