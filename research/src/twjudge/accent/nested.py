"""Accent Detective as one nested network.

One Conformer trunk with exits; size k is the trunk up to exit k plus that exit's parts, so every smaller size is a
part of every larger one:

  front        log-mel, two 1-D convolutions (20 ms frames), as twjudge.accent.student
  trunk        `max(exits)` Conformer layers
  exit k       LayerNorm and one linear map per distilled teacher layer (wav2vec 2.0 "v4" "v8" "v12",
               HuBERT "b8" "b12"; both Chinese base models, MIT), predicting that layer's states
  CTC head     on exit 1's predicted "v4" (in every size): forced alignment with the text, or greedy decoding
               without it
  judge k      that exit's syllable judge (NestedJudge): its own input projection per layer, learned β per layer,
               text context (unknown without the text), a 2-layer transformer, an output layer

Mode "t" uses the text (forced segmentation, tone / sandhi / Mainland-neutral context), mode "f" does not (decoded
segmentation, context unknown): the same parts, minus the text.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from twjudge.accent.unified import Context

SR, HOP, N_MELS = 16000, 160, 80
TARGETS = ("v4", "v8", "v12", "b8", "b12")


class _SE(nn.Module):
    """Squeeze-excitation over time (ECAPA): channel gates from the utterance mean of the layer output."""

    def __init__(self, d, r=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, r), nn.ReLU(), nn.Linear(r, d), nn.Sigmoid())

    def forward(self, h):                               # (T, B, d)
        return h * self.f(h.mean(0))[None]


class NestedStudent(nn.Module):
    def __init__(self, exits=(2, 4, 8, 18), targets=TARGETS, d=384, heads=6, ff=768, dropout=0.1, se=False, ctc=True):
        super().__init__()
        import torchaudio

        from twjudge.accent.compact import ctc_head

        self.exits, self.targets = tuple(exits), tuple(targets)
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, win_length=400, hop_length=HOP, n_mels=N_MELS)
        self.front = nn.Sequential(nn.Conv1d(N_MELS, d, 3, padding=1), nn.GELU(),
                                   nn.Conv1d(d, d, 3, stride=2, padding=1), nn.GELU())
        self.norm_in = nn.LayerNorm(d)
        self.enc = torchaudio.models.Conformer(input_dim=d, num_heads=heads, ffn_dim=ff, num_layers=max(exits),
                                               depthwise_conv_kernel_size=31, dropout=dropout)
        self.norm_out = nn.ModuleList([nn.LayerNorm(d) for _ in exits])
        self.pred = nn.ModuleList([nn.ModuleDict({t: nn.Linear(d, 768) for t in targets}) for _ in exits])
        self.ctc = ctc_head(arch="conv") if ctc else None          # training only (the readouts need no alignment)
        # optional: an SE gate after every Conformer layer (ECAPA's per-utterance channel gating)
        self.se = nn.ModuleList([_SE(d) for _ in range(max(exits))]) if se else None
        self.config = {"exits": list(exits), "targets": list(targets), "d": d, "heads": heads, "ff": ff, "se": bool(se)}

    @staticmethod
    def frames(n_samples: int) -> int:
        return max(0, (n_samples - 400) // 320 + 1)

    def forward(self, wav, upto: int | None = None):
        """wav (B, N) 16 kHz → [{target: (B, T, 768), "h": the exit's normalized hidden state (B, T, d)} for exits 1..upto]
        (upto = all by default)."""
        return list(self.iter_exits(wav, upto))

    def iter_exits(self, wav, upto: int | None = None):
        """forward's exits one at a time, each as soon as its layer is reached: a larger size continues the same pass
        (anytime scoring, no computation repeated)."""
        upto = upto or len(self.exits)
        x = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-7)
        m = torch.log(self.mel(x) + 1e-6)[:, :, 1:]
        h = self.front(m)
        h = self.norm_in(h.transpose(1, 2))
        T = self.frames(wav.shape[1])
        h = h[:, :T] if h.shape[1] >= T else nn.functional.pad(h, (0, 0, 0, T - h.shape[1]))
        h = h.transpose(0, 1)                                   # (T, B, d) for torchaudio's Conformer layers
        e = 0
        for i, layer in enumerate(self.enc.conformer_layers[: self.exits[upto - 1]]):
            h = layer(h, None)
            if self.se is not None:
                h = self.se[i](h)
            if i + 1 == self.exits[e]:
                z = self.norm_out[e](h.transpose(0, 1))
                yield {**{t: self.pred[e][t](z) for t in self.targets}, "h": z}
                e += 1

    def emissions(self, outs):
        """CTC log-probabilities (B, T, V) from exit 1."""
        return torch.log_softmax(self.ctc(outs[0]["v4"]), dim=-1)

    def size(self, k: int) -> int:
        """Parameters of size k: front, trunk to exit k, exit 1 (for the CTC head) and exit k, the CTC head."""
        n = sum(p.numel() for m in (self.front, self.norm_in, self.ctc) if m is not None for p in m.parameters())
        n += sum(p.numel() for layer in self.enc.conformer_layers[: self.exits[k - 1]] for p in layer.parameters())
        if self.se is not None:
            n += sum(p.numel() for g in self.se[: self.exits[k - 1]] for p in g.parameters())
        for e in sorted({0, k - 1}):
            n += sum(p.numel() for p in self.norm_out[e].parameters()) + sum(p.numel() for p in self.pred[e].parameters())
        return n

    @classmethod
    def from_detective(cls, path, exits=(2, 4, 8, 18), targets=TARGETS):
        """Warm start from a deployed dual student (Accent Detective safetensors): front, the first layers, the
        wav2vec 2.0 heads and norm of every exit, the CTC head. Layers beyond it start as the identity (their residual
        branches zeroed), HuBERT heads start untrained."""
        from twjudge.accent.compact import load_checkpoint

        ck = load_checkpoint(path)
        sc = ck["student_config"]
        net = cls(exits, targets, d=sc["d"], heads=sc["heads"], ff=sc["ff"])
        st = ck["student_state"]
        own = net.state_dict()
        n0 = sc["n_layers"]
        for k, v in st.items():
            if k.startswith(("front.", "norm_in.", "mel.")) or (k.startswith("enc.conformer_layers.")
                                                                and int(k.split(".")[2]) < n0):
                own[k] = v
        for e in range(len(exits)):
            for k in ("weight", "bias"):
                own[f"norm_out.{e}.{k}"] = st[f"norm_out.{k}"]
            for t in targets:
                if t.startswith("v") and f"pred.{t[1:]}.weight" in st:
                    own[f"pred.{e}.{t}.weight"], own[f"pred.{e}.{t}.bias"] = st[f"pred.{t[1:]}.weight"], st[f"pred.{t[1:]}.bias"]
        own.update({f"ctc.{k}": v for k, v in ck["ctc_state"].items()})
        net.load_state_dict(own)
        with torch.no_grad():
            for layer in net.enc.conformer_layers[n0:]:
                for mod in (layer.ffn1.sequential[4], layer.ffn2.sequential[4], layer.self_attn.out_proj):
                    mod.weight.zero_(); mod.bias.zero_()
                layer.conv_module.sequential[-2].weight.zero_()
        return net


class NestedJudge(nn.Module):
    """One syllable judge per exit. dims: {tag: 2 × 768}; mus: {exit index: {tag: (V, dim)}}."""

    def __init__(self, n_exits, tags, dims, mus, d=192, n_layers=2, dropout=0.2):
        super().__init__()
        self.tags = list(tags)
        self.proj = nn.ModuleList([nn.ModuleDict({t: nn.Sequential(nn.LayerNorm(dims[t]), nn.Dropout(dropout),
                                                                  nn.Linear(dims[t], d)) for t in tags})
                                   for _ in range(n_exits)])
        self.beta_logit = nn.Parameter(torch.zeros(n_exits, len(tags)))
        self.ctx = nn.ModuleList([Context(d) for _ in range(n_exits)])
        self.enc = nn.ModuleList()
        for _ in range(n_exits):
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
            self.enc.append(nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False))
        self.out = nn.ModuleList([nn.Linear(d, 1) for _ in range(n_exits)])
        for e, m in mus.items():
            for t, mu in m.items():
                self.register_buffer(f"mu_{e}_{t}", torch.as_tensor(mu, dtype=torch.float16), persistent=True)

    def forward(self, b, e):
        """b: ctx (B, S, 5), mask (B, S) True = padding, types (B, S), feats: one (B, S, dim) per tag. Syllable log-odds."""
        h = self.ctx[e](b["ctx"])
        beta = torch.sigmoid(self.beta_logit[e])
        for j, (t, x) in enumerate(zip(self.tags, b["feats"])):
            mu = getattr(self, f"mu_{e}_{t}")[b["types"]].float()
            h = h + self.proj[e][t](x.float() - beta[j] * mu)
        return self.out[e](self.enc[e](h, src_key_padding_mask=b["mask"])).squeeze(-1)

    def size(self, e) -> int:
        return sum(p.numel() for m in (self.proj[e], self.ctx[e], self.enc[e], self.out[e]) for p in m.parameters()) + len(self.tags)

    def scalars(self):
        return [self.beta_logit]


# ---------------------------------------------------------------------------------------------------- ECAPA trunk
class _Res2(nn.Module):
    """Res2Net convolution (scale groups, the first passed through, each next one convolved after adding the previous)."""

    def __init__(self, C, k, dil, scale):
        super().__init__()
        self.scale, w = scale, C // scale
        self.convs = nn.ModuleList([nn.Conv1d(w, w, k, dilation=dil, padding=dil * (k - 1) // 2) for _ in range(scale - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(w) for _ in range(scale - 1)])

    def forward(self, x):
        xs = torch.chunk(x, self.scale, 1)
        ys, y = [xs[0]], None
        for i in range(1, self.scale):
            y = torch.relu(self.bns[i - 1](self.convs[i - 1](xs[i] if i == 1 else xs[i] + y)))
            ys.append(y)
        return torch.cat(ys, 1)


class _SERes2Block(nn.Module):
    def __init__(self, C, k, dil, scale, se=128):
        super().__init__()
        self.c1, self.b1 = nn.Conv1d(C, C, 1), nn.BatchNorm1d(C)
        self.res2 = _Res2(C, k, dil, scale)
        self.c2, self.b2 = nn.Conv1d(C, C, 1), nn.BatchNorm1d(C)
        self.se = nn.Sequential(nn.Linear(C, se), nn.ReLU(), nn.Linear(se, C), nn.Sigmoid())

    def forward(self, x):
        h = self.b1(torch.relu(self.c1(x)))
        h = self.res2(h)
        h = self.b2(torch.relu(self.c2(h)))
        h = h * self.se(h.mean(2))[:, :, None]
        return x + h


class EcapaStudent(nn.Module):
    """ECAPA-TDNN-style trunk (Desplanques et al. 2020) with NestedStudent's interface and one exit: log-mel, a 5-tap
    convolution with stride 2 (20 ms frames, the teachers' rate), three SE-Res2Blocks (k 3, dilation 2, 3, 4, scale 8),
    multi-layer feature aggregation (the blocks' outputs concatenated, 1×1 convolution to d), then the same exit head
    (LayerNorm, one linear map per teacher layer) and CTC head as NestedStudent."""

    def __init__(self, exits=(1,), targets=TARGETS, d=384, C=1024, scale=8, heads=None, ff=None):
        super().__init__()
        import torchaudio

        from twjudge.accent.compact import ctc_head

        self.exits, self.targets = (1,), tuple(targets)
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, win_length=400, hop_length=HOP, n_mels=N_MELS)
        self.front = nn.Sequential(nn.Conv1d(N_MELS, C, 5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(C))
        self.blocks = nn.ModuleList([_SERes2Block(C, 3, dil, scale) for dil in (2, 3, 4)])
        self.mfa = nn.Sequential(nn.Conv1d(3 * C, d, 1), nn.ReLU())
        self.norm_out = nn.ModuleList([nn.LayerNorm(d)])
        self.pred = nn.ModuleList([nn.ModuleDict({t: nn.Linear(d, 768) for t in targets})])
        self.ctc = ctc_head(arch="conv")
        self.config = {"arch": "ecapa", "exits": [1], "targets": list(targets), "d": d, "C": C, "scale": scale}

    frames = staticmethod(NestedStudent.frames)
    emissions = NestedStudent.emissions

    def trunk_params(self):
        return sum(p.numel() for m in (self.front, self.blocks, self.mfa) for p in m.parameters())

    def forward(self, wav, upto=None):
        x = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-7)
        m = torch.log(self.mel(x) + 1e-6)[:, :, 1:]
        h = self.front(m)
        outs = []
        for b in self.blocks:
            h = b(h)
            outs.append(h)
        h = self.mfa(torch.cat(outs, 1)).transpose(1, 2)            # (B, T', d)
        T = self.frames(wav.shape[1])
        h = h[:, :T] if h.shape[1] >= T else nn.functional.pad(h, (0, 0, 0, T - h.shape[1]))
        z = self.norm_out[0](h)
        return [{**{t: self.pred[0][t](z) for t in self.targets}, "h": z}]

    def size(self, k: int = 1) -> int:
        return sum(p.numel() for m in (self.front, self.blocks, self.mfa, self.norm_out, self.pred, self.ctc) for p in m.parameters())


def conformer_trunk_params(d=384, heads=6, ff=768, layers=4):
    n = NestedStudent(exits=(layers,), d=d, heads=heads, ff=ff)
    return sum(p.numel() for m in (n.front, n.norm_in) for p in m.parameters()) + \
        sum(p.numel() for layer in n.enc.conformer_layers for p in layer.parameters())


def ecapa_channels_for(target: int, d=384, scale=8):
    """The multiple of 64 whose ECAPA trunk is closest in parameters to `target` (matched in parameters)."""
    best = None
    for C in range(256, 2049, 64):
        e = EcapaStudent(C=C, d=d, scale=scale).trunk_params()
        if best is None or abs(e - target) < best[1]:
            best = (C, abs(e - target))
    return best[0]


def build_student(config: dict):
    """The trunk a checkpoint config describes: EcapaStudent (arch "ecapa") or NestedStudent."""
    if config.get("arch") == "ecapa":
        return EcapaStudent(targets=config["targets"], d=config["d"], C=config["C"], scale=config["scale"])
    return NestedStudent(**{k: config[k] for k in ("exits", "targets", "d", "heads", "ff")}, se=config.get("se", False))


# ------------------------------------------------------------------------------------------- the readouts of an exit
class ASPReadout(nn.Module):
    """No-text readout of one exit: every predicted teacher layer, utterance mean removed, LayerNorm and a linear map
    to d, summed; channel-wise attentive statistics pooling (weighted mean and standard deviation); LayerNorm and a linear
    output, the utterance log-odds. Same parameters as r11_asp.py's ASP."""

    def __init__(self, n_layers=len(TARGETS), dim=768, d=192):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, d))
                                   for _ in range(n_layers)])
        self.att = nn.Sequential(nn.Conv1d(d, 128, 1), nn.Tanh(), nn.Conv1d(128, d, 1))
        self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(0.2), nn.Linear(2 * d, 1))

    def frames(self, x):                                 # x (B, T, L, dim) → (B, d, T)
        return sum(p(x[:, :, j].float()) for j, p in enumerate(self.proj)).transpose(1, 2)

    def forward(self, x, mask, h=None):                  # mask (B, T) True = padding
        h = self.frames(x) if h is None else h
        e = self.att(h).masked_fill(mask[:, None], -1e4)
        w = torch.softmax(e, dim=2)
        mu = (w * h).sum(2)
        sd = ((w * h * h).sum(2) - mu * mu).clamp(min=1e-5).sqrt()
        return self.out(torch.cat([mu, sd], 1)).squeeze(-1)


class _SylEmbed(nn.Module):
    """One vector per syllable of the text: toneless syllable type, Mainland tone, sandhi tone, Mainland-neutral flag,
    Taiwan tone, readings differ, utterance-final, position. ctx (B, S, 7) holds the last seven."""

    def __init__(self, n_types, d):
        super().__init__()
        self.kind = nn.Embedding(n_types, d)
        self.emb = nn.ModuleList([nn.Embedding(n, d) for n in (6, 6, 2, 6, 2, 2)])
        self.pos = nn.Linear(1, d)

    def forward(self, types, ctx):
        return self.kind(types) + self.pos(ctx[..., 6:7]) + sum(e(ctx[..., j].long()) for j, e in enumerate(self.emb))


class TextReadout(nn.Module):
    """With-text readout of one exit: a correction added to that exit's ASP log-odds.

    One query per syllable of the text (_SylEmbed); two decoder layers attend over the exit's frames (ASP's projection,
    then a linear map plus a depthwise convolution). No CTC and no alignment: the attention logit of syllable s at frame
    t gets a soft monotonic prior −(t′ − c_s)² / 2σ², with t′ the frame's position inside the voiced span, c_s = (s + ½) / S,
    one learned σ per head. Every syllable gets a log-odds and the correction is their mean. The output layer starts at
    zero, so the text mode starts equal to the no-text mode."""

    def __init__(self, n_types, d=192, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.syl = _SylEmbed(n_types, d)
        self.inp = nn.Linear(d, d)
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=d)
        layer = nn.TransformerDecoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.log_sigma = nn.Parameter(torch.full((heads,), math.log(0.08)))
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, fmask, span, types, ctx, smask):
        """h (B, d, T) ASP's frame projection; fmask (B, T) True = padding; span (B, 2) first and last voiced frame;
        types (B, S), ctx (B, S, 7), smask (B, S) True = padding. Returns the correction (B,)."""
        B, _, T = h.shape
        S = types.shape[1]
        h = h.masked_fill(fmask[:, None], 0.0)                                        # padding never enters the convolution
        m = self.inp(h.transpose(1, 2)) + self.conv(h).transpose(1, 2)
        q = self.syl(types, ctx)
        t = torch.arange(T, device=h.device, dtype=torch.float32)[None]
        a, b = span[:, :1].float(), span[:, 1:].float()
        tp = (t - a) / (b - a).clamp(min=1)                                             # (B, T)
        n = (~smask).sum(1, keepdim=True).float().clamp(min=1)
        c = (torch.arange(S, device=h.device, dtype=torch.float32)[None] + 0.5) / n      # (B, S)
        sig = self.log_sigma.exp()
        prior = -(c[:, None, :, None] - tp[:, None, None, :]) ** 2 / (2 * sig[None, :, None, None] ** 2)
        prior = prior.clamp(min=-1e4).masked_fill(fmask[:, None, None, :], -1e4)       # (B, H, S, T)
        x = self.dec(q, m, tgt_key_padding_mask=smask, memory_mask=prior.reshape(B * self.heads, S, T))
        z = self.out(x).squeeze(-1)
        keep = (~smask).float()
        return (z * keep).sum(1) / keep.sum(1).clamp(min=1)


class TextOnly(nn.Module):
    """The text alone (the product-of-experts bias of the text readout): _SylEmbed, a 2-layer encoder, the mean of the
    syllables' log-odds. Trained only, never deployed."""

    def __init__(self, n_types, d=192, layers=2, dropout=0.2):
        super().__init__()
        self.syl = _SylEmbed(n_types, d)
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def forward(self, types, ctx, smask):
        z = self.out(self.enc(self.syl(types, ctx), src_key_padding_mask=smask)).squeeze(-1)
        keep = (~smask).float()
        return (z * keep).sum(1) / keep.sum(1).clamp(min=1)


def text_syllables(aligner, text: str, vocab: dict):
    """The text's syllables for TextReadout, without audio: (type ids, ctx rows (S, 7)), or None (fewer than two Han
    syllables, or a reading outside the CTC letters). aligner: a twjudge.align.ctc.CTCAligner (only its readings are used)."""
    from twjudge.accent.compact import VOCAB
    from twjudge.phonology import sandhi_tones, split_syllable

    sy = aligner.syllables(text, VOCAB)
    if sy is None:
        return None
    cn = [split_syllable(s[1])[2] for s in sy]
    tw = [split_syllable(s[5])[2] for s in sy]
    n = len(sy)
    types = [vocab.get(s[1].rstrip("012345"), 0) for s in sy]
    ctx = [[int(a), int(b), int(s[2]), int(c), int(s[5] != s[1]), int(k == n - 1), k / max(n - 1, 1)]
           for k, (s, a, b, c) in enumerate(zip(sy, cn, sandhi_tones(cn), tw))]
    return types, ctx


def voiced_span(wav: np.ndarray, n_frames: int, floor_db: float = -35.0) -> tuple[int, int]:
    """First and last 20 ms student frame whose energy is within floor_db of the loudest (twjudge.align.ctc's rule)."""
    hop = 320
    n = min(n_frames, len(wav) // hop)
    if n < 2:
        return 0, max(n_frames - 1, 1)
    rms = np.sqrt(np.mean(wav[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
    on = np.where(20 * np.log10(rms / rms.max()) > floor_db)[0]
    return (int(on[0]), int(on[-1])) if len(on) >= 2 else (0, n_frames - 1)
