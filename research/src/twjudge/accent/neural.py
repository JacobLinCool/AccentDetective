"""Neural Taiwan-accent judge on syllable-pooled HuBERT features (judge B, differentiable).

Input per syllable: the pooled, utterance-normalised HuBERT states of `ssl_features` and the
text context (tone, sandhi tone, Mainland-neutral, phrase-final, relative position) — never the
character identity. A small transformer over the syllable sequence gives each syllable a
log-odds of Taiwan; the utterance score is their mean.

Content is controlled as in judge A (length and text-context importance weights) and, in
addition, by a product-of-experts bias model: a text-only network is trained first and its
log-odds is added to the main network's during training only, so the main network learns what
the audio adds beyond the text context. At inference the main network is used alone.

Optionally (``--adv-speaker λ``) a speaker adversary reads the utterance representation (the mean
of the encoder states, which the utterance score is a linear function of) through a gradient-
reversal layer. It is label-conditional: it only chooses among the training speakers of the
utterance's own class, so it removes speaker identity beyond the accent class rather than the
accent itself.

The features it reads are differentiable functions of the waveform (`ssl_features.frame_states`
and `pool`), so `JudgeB.from_audio` scores audio end to end and passes gradients back to it.

    python -m twjudge.accent.neural train --feat-dir data/judge_corpus --model data/judge/models/judge_b.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from twjudge.accent.features import SYL_NAMES
from twjudge.accent.prosodic import length_weights, text_weights

CTX = ["tone", "sandhi", "cn_neutral", "final", "relpos"]


def device():
    # not MPS: masked attention there produces NaN gradients (torch 2.6)
    return "cuda" if torch.cuda.is_available() else "cpu"


class Context(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.tone, self.sandhi = nn.Embedding(6, d), nn.Embedding(6, d)
        self.neu, self.fin = nn.Embedding(2, d), nn.Embedding(2, d)
        self.pos = nn.Linear(1, d)

    def forward(self, c):          # c: (B, S, 5) float
        return (self.tone(c[..., 0].long()) + self.sandhi(c[..., 1].long()) + self.neu(c[..., 2].long())
                + self.fin(c[..., 3].long()) + self.pos(c[..., 4:5]))


class SyllableNet(nn.Module):
    def __init__(self, d_in: int | None, d=192, layers=2, dropout=0.2):
        super().__init__()
        self.ctx = Context(d)
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, d)) if d_in else None
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
        # no nested-tensor fast path: in eval mode it leaves NaN at padded positions
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def hidden(self, ctx, feat, mask):      # mask: (B, S) True = padding
        h = self.ctx(ctx) + (self.inp(feat) if self.inp is not None else 0)
        return self.enc(h, src_key_padding_mask=mask)                          # (B, S, d)

    def forward(self, ctx, feat, mask):
        return self.out(self.hidden(ctx, feat, mask)).squeeze(-1)             # (B, S) logits


class NestedSyllableNet(nn.Module):
    """SyllableNet over a nested (Matryoshka) set of layers: every prefix of the layer list is a working
    judge. Each layer has its own normalisation and projection, summed over the layers in use, plus an
    embedding of the prefix length, so a shorter prefix sees exactly the features it would be given alone.
    feat holds the layers' [syllable, rhyme] blocks in prefix order, `d_layer` columns each."""

    def __init__(self, d_layer: int, n_layers: int, d=192, layers=2, dropout=0.2):
        super().__init__()
        self.d_layer, self.n_prefix = d_layer, n_layers
        self.ctx = Context(d)
        self.inp = nn.ModuleList(nn.Sequential(nn.LayerNorm(d_layer), nn.Dropout(dropout), nn.Linear(d_layer, d))
                                 for _ in range(n_layers))
        self.prefix = nn.Embedding(n_layers, d)
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def hidden(self, ctx, feat, mask, k=None):
        k = k or self.n_prefix
        h = self.ctx(ctx) + self.prefix.weight[k - 1]
        for j in range(k):
            h = h + self.inp[j](feat[..., j * self.d_layer:(j + 1) * self.d_layer])
        return self.enc(h, src_key_padding_mask=mask)

    def forward(self, ctx, feat, mask, k=None):
        return self.out(self.hidden(ctx, feat, mask, k)).squeeze(-1)


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def utt_state(h, mask):
    keep = (~mask).float()[..., None]
    return (h * keep).sum(1) / keep.sum(1).clamp(min=1)


def utt_logits(syl, mask):
    keep = (~mask).float()
    return torch.where(mask, torch.zeros_like(syl), syl).sum(1) / keep.sum(1).clamp(min=1)


def load(fd: Path, split: str, ssl: str = "ssl"):
    return load_pair(fd / f"feat_{split}.npz", fd / f"{ssl}_{split}.npz")


def load_pair(feat_path: Path, ssl_path: Path):
    a = np.load(feat_path)
    b = np.load(ssl_path)
    assert list(a["ids"]) == list(b["ids"]), "feature files out of order"
    ids = [str(i) for i in a["ids"]]
    cols = [SYL_NAMES.index(c) for c in CTX]
    return {"ids": ids, "y": np.array([i.startswith("tw") for i in ids], np.float32), "lens": a["lens"],
            "ctx": a["X"][:, cols].astype(np.float32), "X": a["X"], "F": b["F"]}


def batches(d, idx, use_feat, dev):
    off = np.r_[0, np.cumsum(d["lens"])]
    S = int(d["lens"][idx].max())
    B = len(idx)
    ctx = np.zeros((B, S, len(CTX)), np.float32)
    feat = np.zeros((B, S, d["F"].shape[1]), np.float32) if use_feat else None
    mask = np.ones((B, S), bool)
    for k, i in enumerate(idx):
        n = d["lens"][i]
        ctx[k, :n] = d["ctx"][off[i]:off[i] + n]
        if use_feat:
            feat[k, :n] = d["F"][off[i]:off[i] + n]
        mask[k, :n] = False
    t = lambda x: torch.tensor(x, device=dev) if x is not None else None
    return t(ctx), t(feat), t(mask)


def train_net(net, d, w, bias=None, epochs=12, bs=64, lr=3e-4, use_feat=True, dev_data=None, seed=0,
              spk=None, adv=0.0, soft=None, kd=0.0):
    """spk: per-utterance speaker index (0..n-1) for the label-conditional speaker adversary of weight adv.
    soft: a teacher judge's per-syllable log-odds (concatenated like d["F"]) for knowledge distillation:
    adds kd × the mean squared difference of the syllable log-odds (main network, without the bias)."""
    dv = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net.to(dv)
    params = list(net.parameters())
    if adv > 0:
        n_spk = int(spk.max()) + 1
        head = nn.Sequential(nn.Linear(net.out.in_features, 256), nn.ReLU(), nn.Linear(256, n_spk)).to(dv)
        params += list(head.parameters())
        allowed = torch.zeros(2, n_spk, dtype=torch.bool, device=dv)       # speakers of each class
        allowed[torch.tensor(d["y"].astype(int), device=dv), torch.tensor(spk, device=dv)] = True
        spk_t = torch.tensor(spk, device=dv)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n = len(d["y"])
    steps, step = epochs * ((n + bs - 1) // bs), 0
    best, best_state = -1, None
    for ep in range(epochs):
        net.train()
        order = rng.permutation(n)
        hit = tot = 0
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            ctx, feat, mask = batches(d, idx, use_feat, dv)
            y = torch.tensor(d["y"][idx], device=dv)
            wt = torch.tensor(w[idx], device=dv)
            # a nested net is trained on every prefix of its layers at once (Matryoshka), losses averaged
            ks = range(1, net.n_prefix + 1) if getattr(net, "n_prefix", 0) else [None]
            loss = 0.0
            for k in ks:
                h = net.hidden(ctx, feat, mask) if k is None else net.hidden(ctx, feat, mask, k)
                z = utt_logits(net.out(h).squeeze(-1), mask)
                if bias is not None:
                    z = z + torch.tensor(bias[idx], device=dv)
                loss = loss + (nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * wt).mean() / len(ks)
            if soft is not None and kd > 0:
                off_s = np.r_[0, np.cumsum(d["lens"])]
                tgt = torch.zeros(mask.shape, device=dv)
                for k, i in enumerate(idx):
                    tgt[k, :d["lens"][i]] = torch.tensor(soft[off_s[i]:off_s[i] + d["lens"][i]], device=dv)
                diff = (net.out(h).squeeze(-1) - tgt) ** 2
                loss = loss + kd * torch.where(mask, torch.zeros_like(diff), diff).sum() / (~mask).sum()
            if adv > 0:
                lam = adv * (2 / (1 + np.exp(-10 * step / steps)) - 1)        # DANN ramp-up
                logit_s = head(_Reverse.apply(utt_state(h, mask), lam))
                logit_s = logit_s.masked_fill(~allowed[y.long()], float("-inf"))
                target = spk_t[torch.tensor(idx, device=dv)]
                loss = loss + nn.functional.cross_entropy(logit_s, target)
                hit += int((logit_s.argmax(1) == target).sum()); tot += len(idx)
            step += 1
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if dev_data is not None:
            from twjudge.accent.prosodic import auc

            zd = predict(net, dev_data, use_feat)
            a = auc(dev_data["y"], zd, length_weights(dev_data["lens"], dev_data["y"].astype(int)))
            print(f"epoch {ep} dev auc {a:.4f}" + (f" adversary train acc {hit / tot:.3f}" if tot else ""), flush=True)
            if a > best:
                best, best_state = a, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


@torch.no_grad()
def predict(net, d, use_feat=True, bs=128, states=False, k=None):
    """Utterance log-odds; with states=True also the utterance representations (n, d). k: prefix length of a
    NestedSyllableNet (default: all its layers)."""
    dv = device()
    net.eval().to(dv)
    out, hs = [], []
    for s in range(0, len(d["y"]), bs):
        idx = np.arange(s, min(s + bs, len(d["y"])))
        ctx, feat, mask = batches(d, idx, use_feat, dv)
        h = net.hidden(ctx, feat, mask) if k is None else net.hidden(ctx, feat, mask, k)
        out.append(utt_logits(net.out(h).squeeze(-1), mask).cpu().numpy())
        if states:
            hs.append(utt_state(h, mask).cpu().numpy())
    return (np.concatenate(out), np.concatenate(hs)) if states else np.concatenate(out)


def utterance_weights(d):
    y = d["y"].astype(int)
    lw = length_weights(d["lens"], y)
    sw = text_weights(d["X"], np.repeat(y, d["lens"]))
    off = np.r_[0, np.cumsum(d["lens"])]
    tw = np.array([sw[off[k]:off[k + 1]].mean() for k in range(len(y))])
    w = lw * tw
    return (w / w.mean()).astype(np.float32)


class JudgeB:
    """Trained judge B plus the frozen HuBERT front end, for scoring audio directly."""

    def __init__(self, path: str | Path):
        ck = torch.load(path, map_location="cpu")
        self.net = SyllableNet(ck["d_in"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.hubert = None
        # judge B reads HuBERT features, judge C wav2vec 2.0 ("w2v" prefix of its feature files)
        self.ssl_kind = "wav2vec2" if ck.get("ssl") == "w2v" else "hubert"

    def from_audio(self, wav16k: torch.Tensor, chars: list[dict]) -> torch.Tensor:
        """Differentiable utterance log-odds of Taiwan for a 16 kHz waveform with its alignment."""
        from twjudge.accent.ssl_features import frame_states, load_model, pool
        from twjudge.phonology import sandhi_tones, split_syllable

        if self.hubert is None:
            self.hubert = load_model(self.ssl_kind)
        dev = next(self.hubert.parameters()).device
        wav16k = wav16k.to(dev)
        self.net.to(dev)
        h = frame_states(self.hubert, wav16k)
        feat = pool(h, chars)[None].float()
        tones = [split_syllable(c["syl"])[2] for c in chars]
        n = len(chars)
        fin = [k + 1 == n or chars[k + 1]["start"] - c["end"] >= 0.15 for k, c in enumerate(chars)]
        ctx = torch.tensor([[int(t), int(s), int(c["cn_neutral"]), int(f), k / max(n - 1, 1)]
                            for k, (t, s, c, f) in enumerate(zip(tones, sandhi_tones(tones), chars, fin))],
                           dtype=torch.float32, device=wav16k.device)[None]
        mask = torch.zeros((1, n), dtype=torch.bool, device=wav16k.device)
        syl = self.net(ctx, feat, mask)
        return utt_logits(syl, mask)[0], syl[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--feat-dir", default="data/judge_corpus")
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--ssl", default="ssl", help="feature-file prefix: ssl (HuBERT, judge B) or w2v (wav2vec 2.0, judge C)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from twjudge.accent.prosodic import auc

    fd = Path(a.feat_dir)
    tr, dv, te = (load(fd, s, a.ssl) for s in ("train", "dev", "test"))
    w = utterance_weights(tr)
    print("text-only bias model")
    bias_net = train_net(SyllableNet(None), tr, w, epochs=6, use_feat=False, dev_data=dv)
    bias_tr = predict(bias_net, tr, use_feat=False)
    print("main model")
    net = train_net(SyllableNet(tr["F"].shape[1]), tr, w, bias=bias_tr, epochs=a.epochs, dev_data=dv, seed=a.seed)
    zt = predict(net, te)
    bt = predict(bias_net, te, use_feat=False)
    lw = length_weights(te["lens"], te["y"].astype(int))
    rep = {"test_auc_length_matched": auc(te["y"], zt, lw), "test_auc_raw": auc(te["y"], zt),
           "text_only_bias_auc": auc(te["y"], bt, lw), "judge_vs_text_only_corr": float(np.corrcoef(zt, bt)[0, 1])}
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"d_in": tr["F"].shape[1], "state": {k: v.cpu() for k, v in net.state_dict().items()},
                "ctx": CTX, "ssl": a.ssl, "report": rep}, a.model)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
