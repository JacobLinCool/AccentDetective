"""The network: one Conformer trunk with exits at several depths, and two readouts per exit.

  front      log-mel features and two 1-D convolutions, 20 ms frames
  trunk      Conformer layers; exit k sits after layer `exits[k]`
  exit k     a layer norm and one linear map per distilled teacher layer (wav2vec 2.0 layers 4, 8, 12 and HuBERT
             layers 8, 12 of the Chinese base models), predicting that layer's hidden states
  readouts   ASPReadout (audio only): every predicted layer with its utterance mean removed, projected, summed and
             pooled by channel-wise attentive statistics into a log-odds of Taiwan Mandarin;
             TextReadout (optional): one query per syllable of the text, two attention layers over the audio frames
             under a soft monotonic position prior, whose mean output corrects the audio log-odds

A smaller size is the first part of every larger one, so one pass to the last exit yields every size's score.
Module and parameter names match the released checkpoint.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

SR, HOP, N_MELS = 16000, 160, 80
TARGETS = ("v4", "v8", "v12", "b8", "b12")


class Trunk(nn.Module):
    def __init__(self, exits=(3, 6, 9, 18), targets=TARGETS, d=384, heads=6, ff=768, dropout=0.1):
        super().__init__()
        import torchaudio

        self.exits, self.targets = tuple(exits), tuple(targets)
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, win_length=400, hop_length=HOP, n_mels=N_MELS)
        self.front = nn.Sequential(nn.Conv1d(N_MELS, d, 3, padding=1), nn.GELU(),
                                   nn.Conv1d(d, d, 3, stride=2, padding=1), nn.GELU())
        self.norm_in = nn.LayerNorm(d)
        self.enc = torchaudio.models.Conformer(input_dim=d, num_heads=heads, ffn_dim=ff, num_layers=max(exits),
                                               depthwise_conv_kernel_size=31, dropout=dropout)
        self.norm_out = nn.ModuleList([nn.LayerNorm(d) for _ in exits])
        self.pred = nn.ModuleList([nn.ModuleDict({t: nn.Linear(d, 768) for t in targets}) for _ in exits])

    @staticmethod
    def frames(n_samples: int) -> int:
        """Number of 20 ms frames for a waveform of n_samples at 16 kHz."""
        return max(0, (n_samples - 400) // 320 + 1)

    def iter_exits(self, wav: torch.Tensor, upto: int | None = None):
        """wav (B, N) at 16 kHz → one dict per exit, {target: (B, T, 768), "h": (B, T, d)}, each yielded as soon as its
        layer is reached (anytime scoring: a larger size continues the same pass)."""
        upto = upto or len(self.exits)
        x = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-7)
        m = torch.log(self.mel(x) + 1e-6)[:, :, 1:]
        h = self.norm_in(self.front(m).transpose(1, 2))
        T = self.frames(wav.shape[1])
        h = h[:, :T] if h.shape[1] >= T else nn.functional.pad(h, (0, 0, 0, T - h.shape[1]))
        h = h.transpose(0, 1)                                   # (T, B, d) for torchaudio's Conformer layers
        e = 0
        for i, layer in enumerate(self.enc.conformer_layers[: self.exits[upto - 1]]):
            h = layer(h, None)
            if i + 1 == self.exits[e]:
                z = self.norm_out[e](h.transpose(0, 1))
                yield {**{t: self.pred[e][t](z) for t in self.targets}, "h": z}
                e += 1

    def forward(self, wav: torch.Tensor, upto: int | None = None) -> list[dict]:
        return list(self.iter_exits(wav, upto))


class ASPReadout(nn.Module):
    """Audio-only readout of one exit: attentive statistics pooling over the predicted teacher layers."""

    def __init__(self, n_layers=len(TARGETS), dim=768, d=192):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, d))
                                   for _ in range(n_layers)])
        self.att = nn.Sequential(nn.Conv1d(d, 128, 1), nn.Tanh(), nn.Conv1d(128, d, 1))
        self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(0.2), nn.Linear(2 * d, 1))

    def frames(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, layers, 768), utterance mean removed → frame projection (B, d, T)."""
        return sum(p(x[:, :, j].float()) for j, p in enumerate(self.proj)).transpose(1, 2)

    def forward(self, x, mask, h=None):
        """mask (B, T) True = padding → log-odds (B,)."""
        h = self.frames(x) if h is None else h
        e = self.att(h).masked_fill(mask[:, None], -1e4)
        w = torch.softmax(e, dim=2)
        mu = (w * h).sum(2)
        sd = ((w * h * h).sum(2) - mu * mu).clamp(min=1e-5).sqrt()
        return self.out(torch.cat([mu, sd], 1)).squeeze(-1)


class _SylEmbed(nn.Module):
    """One vector per syllable: toneless syllable type, Mainland tone, sandhi tone, Mainland-neutral flag, Taiwan tone,
    readings differ, utterance-final, relative position (ctx holds the last seven)."""

    def __init__(self, n_types, d):
        super().__init__()
        self.kind = nn.Embedding(n_types, d)
        self.emb = nn.ModuleList([nn.Embedding(n, d) for n in (6, 6, 2, 6, 2, 2)])
        self.pos = nn.Linear(1, d)

    def forward(self, types, ctx):
        return self.kind(types) + self.pos(ctx[..., 6:7]) + sum(e(ctx[..., j].long()) for j, e in enumerate(self.emb))


class TextReadout(nn.Module):
    """Optional text readout of one exit: a correction added to the exit's audio log-odds. No alignment: the attention
    of syllable s at frame t gets a soft monotonic prior around position (s + ½) / S of the voiced span."""

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

    def forward(self, h, fmask, span, types, ctx, smask):
        """h (B, d, T) the audio readout's frame projection; fmask (B, T) True = padding; span (B, 2) first and last
        voiced frame; types (B, S), ctx (B, S, 7), smask (B, S) True = padding → correction (B,)."""
        B, _, T = h.shape
        S = types.shape[1]
        h = h.masked_fill(fmask[:, None], 0.0)
        m = self.inp(h.transpose(1, 2)) + self.conv(h).transpose(1, 2)
        q = self.syl(types, ctx)
        t = torch.arange(T, device=h.device, dtype=torch.float32)[None]
        a, b = span[:, :1].float(), span[:, 1:].float()
        tp = (t - a) / (b - a).clamp(min=1)
        n = (~smask).sum(1, keepdim=True).float().clamp(min=1)
        c = (torch.arange(S, device=h.device, dtype=torch.float32)[None] + 0.5) / n
        sig = self.log_sigma.exp()
        prior = -(c[:, None, :, None] - tp[:, None, None, :]) ** 2 / (2 * sig[None, :, None, None] ** 2)
        prior = prior.clamp(min=-1e4).masked_fill(fmask[:, None, None, :], -1e4)
        x = self.dec(q, m, tgt_key_padding_mask=smask, memory_mask=prior.reshape(B * self.heads, S, T))
        z = self.out(x).squeeze(-1)
        keep = (~smask).float()
        return (z * keep).sum(1) / keep.sum(1).clamp(min=1)


def voiced_span(wav: np.ndarray, n_frames: int, floor_db: float = -35.0) -> tuple[int, int]:
    """First and last 20 ms frame whose energy is within floor_db of the loudest frame."""
    hop = 320
    n = min(n_frames, len(wav) // hop)
    if n < 2:
        return 0, max(n_frames - 1, 1)
    rms = np.sqrt(np.mean(wav[:n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
    on = np.where(20 * np.log10(rms / rms.max()) > floor_db)[0]
    return (int(on[0]), int(on[-1])) if len(on) >= 2 else (0, n_frames - 1)
