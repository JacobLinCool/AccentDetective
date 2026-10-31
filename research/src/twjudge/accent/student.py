"""Small student encoder distilled from the judges' 12-layer Chinese SSL model (DistilHuBERT-style).

The teacher's cost on CPU is dominated by its waveform convolutions (the 7-layer CNN front end and
the 128-tap positional convolution), not by its transformer layers, so the student replaces both:

  front end   80-band log-mel spectrogram (25 ms window, 10 ms hop), two 1-D convolutions, the second
              with stride 2: 20 ms frames, the teacher's frame rate
  position    a depthwise convolution (31 taps) added to the input of the first layer (transformer)
  encoder     `n_layers` pre-norm transformer layers of width `d`, or with arch="conformer" Conformer
              layers (macaron feed-forward, self-attention, a 31-tap depthwise convolution module in
              every layer; torchaudio.models.Conformer), which carry position through the convolutions
  heads       one linear map per distilled teacher layer, predicting that layer's states (768)

The student is trained to predict several teacher layers at once (L1 + cosine loss); the judge
then reads the predicted layers exactly as it reads the teacher's (`hidden` returns them in the
teacher's indexing, so `compact.pool_layers` works unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn

SR, HOP, N_MELS = 16000, 160, 80


class Student(nn.Module):
    def __init__(self, targets=(1, 2, 3, 4, 6, 8, 12), d=384, n_layers=4, heads=6, ff=1536, dropout=0.1,
                 arch="transformer", whisper_targets=()):
        super().__init__()
        import torchaudio

        self.targets = tuple(targets)
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, win_length=400, hop_length=HOP, n_mels=N_MELS)
        self.front = nn.Sequential(nn.Conv1d(N_MELS, d, 3, padding=1), nn.GELU(),
                                   nn.Conv1d(d, d, 3, stride=2, padding=1), nn.GELU())
        self.arch = arch
        self.norm_in = nn.LayerNorm(d)
        if arch == "conformer":
            self.enc = torchaudio.models.Conformer(input_dim=d, num_heads=heads, ffn_dim=ff, num_layers=n_layers,
                                                   depthwise_conv_kernel_size=31, dropout=dropout)
        else:
            self.pos = nn.Conv1d(d, d, 31, padding=15, groups=d)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm_out = nn.LayerNorm(d)
        self.pred = nn.ModuleDict({str(t): nn.Linear(d, 768) for t in self.targets})
        # optional second teacher: Whisper-medium encoder layers (1024-d), predicted from the same trunk
        self.whisper_targets = tuple(whisper_targets)
        self.wpred = nn.ModuleDict({str(t): nn.Linear(d, 1024) for t in self.whisper_targets})
        self.config = {"targets": list(self.targets), "d": d, "n_layers": n_layers, "heads": heads, "ff": ff,
                       "arch": arch}
        if self.whisper_targets:
            self.config["whisper_targets"] = list(self.whisper_targets)

    def frames(self, n_samples: int) -> int:
        """Number of 20 ms frames the teacher produces for n_samples (wav2vec 2.0 CNN arithmetic)."""
        return max(0, (n_samples - 400) // 320 + 1)

    def forward(self, wav, lengths=None):
        """wav (B, N) at 16 kHz → {teacher layer: (B, T, 768)} with T = teacher frames of the longest input."""
        h = self.trunk(wav, lengths)
        return {t: self.pred[str(t)](h) for t in self.targets}

    def both(self, wav, lengths=None):
        """({wav2vec 2.0 layer: (B, T, 768)}, {Whisper layer: (B, T, 1024)}) from one pass of the trunk."""
        h = self.trunk(wav, lengths)
        return {t: self.pred[str(t)](h) for t in self.targets}, {t: self.wpred[str(t)](h) for t in self.whisper_targets}

    def trunk(self, wav, lengths=None):
        x = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-7)
        # drop the first mel frame: output frame j is then centred at 320 j + 160 samples, within 2.5 ms
        # of the teacher's frame centre (320 j + 200)
        m = torch.log(self.mel(x) + 1e-6)[:, :, 1:]                # (B, 80, N/160)
        h = self.front(m)                                          # (B, d, ~N/320)
        if self.arch != "conformer":
            h = h + self.pos(h)
        h = self.norm_in(h.transpose(1, 2))
        T = self.frames(wav.shape[1])
        h = h[:, :T] if h.shape[1] >= T else nn.functional.pad(h, (0, 0, 0, T - h.shape[1]))
        n = (torch.tensor([self.frames(int(l)) for l in lengths], device=wav.device) if lengths is not None
             else torch.full((wav.shape[0],), T, device=wav.device))
        if self.arch == "conformer":
            h, _ = self.enc(h, n)
        else:
            mask = torch.arange(T, device=wav.device)[None] >= n[:, None] if lengths is not None else None
            h = self.enc(h, src_key_padding_mask=mask)
        return self.norm_out(h)


def load(path, device="cpu") -> Student:
    ck = torch.load(path, map_location="cpu")
    s = Student(**ck["config"])
    s.load_state_dict(ck["state"])
    return s.to(device).eval()


def hidden(student: Student, wav16k: torch.Tensor) -> dict:
    """Predicted teacher states of one utterance, {teacher layer: (T, 768)}."""
    out = student(wav16k[None])
    return {t: v[0] for t, v in out.items()}
