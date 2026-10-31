"""Recording-condition augmentation: additive noise, room reverberation, telephone band, 8-bit mu-law, gain.

Used to make the student encoder (distillation) and the judges (training) less sensitive to the
recording channel, the weakness found on KeSpeech. Every class gets the same augmentation, so the
channel never becomes a label cue. Assets are packed once by `research/scripts/augment_assets.py`:

  noise.npy / noise_off.npy   MUSAN noise and music (CC BY 4.0), int16 16 kHz, concatenated
  rir.npy                     OpenSLR 28 room impulse responses (Apache 2.0), float32 (n, L), peak-aligned

The whole batch is processed at once on the device; band limits are applied as smooth masks on the
spectrum (no recursive filters, which are slow on a GPU).

    aug = Augmenter("assets/aug", device="cuda")
    x = aug(x)          # (B, T) float waveforms on the device, each item augmented independently
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000


class Augmenter:
    def __init__(self, assets: str | Path, device: str = "cpu", p: float = 0.8, seed: int = 0):
        import torch

        d = Path(assets)
        self.noise = torch.from_numpy(np.load(d / "noise.npy").astype(np.float32) / 32768).to(device)
        self.noise_off = np.load(d / "noise_off.npy")
        self.rir = torch.from_numpy(np.load(d / "rir.npy")).to(device)
        self.dev, self.p = device, p
        self.rng = np.random.default_rng(seed)

    def _noise(self, n: int):
        k = self.rng.integers(len(self.noise_off) - 1)
        a, b = int(self.noise_off[k]), int(self.noise_off[k + 1])
        if b - a <= n:
            return self.noise[a:b].repeat(n // (b - a) + 1)[:n]
        o = self.rng.integers(a, b - n)
        return self.noise[o:o + n]

    def __call__(self, x):
        """x: (B, T) or (T,) float tensor on the device; each item is augmented with probability p."""
        import torch
        import torchaudio.functional as AF

        one = x.dim() == 1
        x = x[None] if one else x.clone()
        Bn, T = x.shape
        u = lambda: self.rng.random(Bn)
        on = u() < self.p
        rev, noi, band, mu = on & (u() < 0.35), on & (u() < 0.6), np.where(on, u(), 1.0), on & (u() < 0.15)
        if rev.any():                                                 # reverberation
            i = torch.from_numpy(np.where(rev)[0]).to(x.device)
            h = self.rir[torch.from_numpy(self.rng.integers(len(self.rir), size=len(i))).to(x.device)]
            y = AF.fftconvolve(x[i], h)[:, :T]
            x[i] = y * (x[i].abs().amax(1, keepdim=True) / (y.abs().amax(1, keepdim=True) + 1e-6))
        if noi.any():                                                 # noise at 5–25 dB SNR
            i = torch.from_numpy(np.where(noi)[0]).to(x.device)
            nz = torch.stack([self._noise(T) for _ in range(len(i))])
            snr = torch.from_numpy(self.rng.uniform(5, 25, len(i))).float().to(x.device)
            px, pn = x[i].pow(2).mean(1) + 1e-8, nz.pow(2).mean(1) + 1e-8
            x[i] = x[i] + nz * torch.sqrt(px / (pn * 10 ** (snr / 10)))[:, None]
        tel, low = band < 0.25, (band >= 0.25) & (band < 0.4)
        if tel.any() or low.any():                                    # telephone band / cheap microphone
            f = torch.fft.rfftfreq(T, 1 / SR, device=x.device)[None]
            lo = np.where(tel, 300.0, 0.0)
            hi = np.where(tel, 3400.0, np.where(low, self.rng.uniform(3000, 6000, Bn), SR))
            lo_t, hi_t = (torch.from_numpy(v).float().to(x.device)[:, None] for v in (lo, hi))
            g = torch.sigmoid((f - lo_t) / 40) * torch.sigmoid((hi_t - f) / 150)
            i = torch.from_numpy(np.where(tel | low)[0]).to(x.device)
            x[i] = torch.fft.irfft(torch.fft.rfft(x[i]) * g[i], n=T)
        if mu.any():                                                  # 8-bit mu-law
            i = torch.from_numpy(np.where(mu)[0]).to(x.device)
            peak = x[i].abs().amax(1, keepdim=True) + 1e-6
            x[i] = AF.mu_law_decoding(AF.mu_law_encoding(x[i] / peak, 256), 256) * peak
        gain = torch.from_numpy(np.where(on, 10 ** (self.rng.uniform(-10, 6, Bn) / 20), 1.0)).float().to(x.device)
        x = torch.nan_to_num(x * gain[:, None]).clamp(-1, 1)
        return x[0] if one else x
