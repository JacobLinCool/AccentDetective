"""The public API: load the released checkpoint and score recordings."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SR = 16000
REPO = "JacobLinCool/accent-detective"
NAMES = ("nano", "tiny", "mini", "small", "base", "medium", "large")


def index(logit: float) -> float:
    """Log-odds on a 0–1 display scale: 0.5 is undecided, each ±0.1 is about 7× the odds, clipped at ±10."""
    return min(1.0, max(0.0, 0.5 + logit / 20))


def load_audio(wav) -> np.ndarray:
    """An audio file path or a 16 kHz mono array → float32 array at 16 kHz."""
    if isinstance(wav, (str, Path)):
        import librosa

        return librosa.load(str(wav), sr=SR, mono=True)[0]
    return np.asarray(wav, dtype=np.float32)


class AccentDetective:
    """How Taiwanese a Mandarin recording sounds.

    One network cut at several depths: size k runs the trunk up to exit k and that exit's readout, so every smaller
    size is a part of every larger one and one pass to the last exit gives them all. The text is optional. Scores are
    calibrated log-odds of Taiwan Mandarin against Mainland Mandarin (Platt transform per size and mode).

        from accent_detective import AccentDetective
        det = AccentDetective.from_pretrained()
        det.score("clip.wav")                          # the largest size, audio only
        det.score("clip.wav", "這麼長的時間", size="nano")
        det.score_sizes("clip.wav")                    # every size from one pass
        det.score_recording("talk.wav")                # any length, cut at pauses
    """

    MAX_SECONDS = 15.0

    def __init__(self, model: str | Path, dictionary: str | Path, cedict: str | Path, device: str | None = None):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file

        from accent_detective.model import ASPReadout, TextReadout, Trunk
        from accent_detective.text import Readings

        with safe_open(str(model), framework="pt") as f:
            self.meta = json.loads(f.metadata()["accent_detective"])
        state = load_file(str(model))
        c = self.meta["config"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Trunk(c["exits"], c["targets"], d=c["d"], heads=c["heads"], ff=c["ff"])
        self.net.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("trunk.")})
        self.asp, self.text = torch.nn.ModuleList(), torch.nn.ModuleList()
        for k in range(1, len(c["exits"]) + 1):
            a, t = ASPReadout(), TextReadout(self.meta["n_types"])
            a.load_state_dict({n[len(f"asp.{k}."):]: v for n, v in state.items() if n.startswith(f"asp.{k}.")})
            t.load_state_dict({n[len(f"text.{k}."):]: v for n, v in state.items() if n.startswith(f"text.{k}.")})
            self.asp.append(a)
            self.text.append(t)
        for m in (self.net, self.asp, self.text):
            m.to(self.device).eval()
        self.vocab = self.meta["vocab"]
        self.readings = Readings(dictionary, cedict)
        self.keys = list(self.meta["sizes"])                         # the keys of the Platt transforms
        self.sizes = tuple(self.meta.get("names") or NAMES[: len(self.keys)])

    @classmethod
    def from_pretrained(cls, repo: str = REPO, revision: str | None = None, device: str | None = None) -> "AccentDetective":
        """Download the checkpoint and the two lexicons from the Hugging Face Hub (cached) and load them."""
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo, revision=revision,
                                      allow_patterns=["model.safetensors", "lexicon/concised.json.xz", "lexicon/cedict.txt.gz"]))
        return cls(path / "model.safetensors", path / "lexicon/concised.json.xz", path / "lexicon/cedict.txt.gz", device)

    def _k(self, size: str | None) -> int:
        if size is None:
            return len(self.keys) - 1
        if size.lower() in self.sizes:
            return self.sizes.index(size.lower())
        raise ValueError(f"unknown size {size!r}: one of {', '.join(self.sizes)}")

    def parameters(self, size: str | None = None, text: bool = False) -> int:
        """Parameters that run for one size and mode."""
        k, n = self._k(size), self.net
        mods = [n.front, n.norm_in, n.norm_out[k], n.pred[k], self.asp[k], *n.enc.conformer_layers[: n.exits[k]]]
        return sum(p.numel() for m in mods + ([self.text[k]] if text else []) for p in m.parameters())

    def _syllables(self, text):
        from accent_detective.text import text_syllables

        return text_syllables(self.readings, text, self.vocab) if text and text.strip() else None

    def _readout(self, k: int, o: dict, w: np.ndarray, syl) -> dict:
        import torch

        from accent_detective.model import voiced_span

        H = torch.stack([o[t][0] for t in self.net.targets])
        X = (H - H.mean(1, keepdim=True)).transpose(0, 1)[None]          # (1, T, layers, 768), utterance mean removed
        m = torch.zeros((1, X.shape[1]), dtype=torch.bool, device=self.device)
        h = self.asp[k].frames(X)
        z, mode = self.asp[k](X, m, h=h), "audio"
        if syl is not None:
            types, ctx = syl
            span = torch.tensor([voiced_span(w, X.shape[1])], device=self.device)
            z = z + self.text[k](h, m, span, torch.tensor([types], device=self.device),
                                 torch.tensor([ctx], dtype=torch.float32, device=self.device),
                                 torch.zeros((1, len(types)), dtype=torch.bool, device=self.device))
            mode = "text"
        raw = float(z)
        a, b = self.meta["platt"][self.keys[k]][mode]
        logit = a * raw + b
        return {"logit": logit, "probability": 1 / (1 + math.exp(-logit)), "index": index(logit), "size": self.sizes[k],
                "mode": mode, "raw_logit": raw}

    def _score(self, w: np.ndarray, text, sizes) -> dict:
        import torch

        syl = self._syllables(text)
        ks = [self._k(s) for s in sizes]
        out = {}
        with torch.inference_mode():
            outs = self.net(torch.tensor(w, dtype=torch.float32, device=self.device)[None], upto=max(ks) + 1)
            for k in ks:
                out[self.sizes[k]] = self._readout(k, outs[k], w, syl)
        return out

    def score(self, wav, text: str | None = None, size: str | None = None) -> dict:
        """One utterance (audio file or 16 kHz array), optionally the text it reads, at one size (default the largest)
        → {"logit", "probability", "index", "size", "mode", "raw_logit"}. Text without Han syllables falls back to
        the audio mode."""
        return next(iter(self._score(load_audio(wav), text, [size]).values()))

    def score_sizes(self, wav, text: str | None = None) -> dict:
        """Every size from one pass: {size name: score dict}."""
        return self._score(load_audio(wav), text, list(self.sizes))

    def score_progressive(self, wav, text: str | None = None):
        """Anytime scoring: one pass of the largest size, yielding each size's score dict, smallest first, as soon as
        its exit is reached. Nothing is computed twice."""
        import torch

        w = load_audio(wav)
        syl = self._syllables(text)
        with torch.inference_mode():
            for k, o in enumerate(self.net.iter_exits(torch.tensor(w, dtype=torch.float32, device=self.device)[None])):
                yield self._readout(k, o, w, syl)

    def score_recording(self, wav, text: str | None = None, sizes=None) -> dict:
        """A recording of any length. Up to MAX_SECONDS it is one utterance; longer audio is cut at pauses into pieces
        of 4–10 s, each scored without the text, and the log-odds are averaged weighted by duration. Returns
        {size name: score dict + "pieces": [(start s, end s, logit)]}; sizes default to the largest."""
        sizes = [self.sizes[self._k(s)] for s in (sizes or [None])]
        w = load_audio(wav)
        if len(w) <= self.MAX_SECONDS * SR:
            return {s: {**r, "pieces": [(0.0, len(w) / SR, r["logit"])]} for s, r in self._score(w, text, sizes).items()}
        hop = 320
        n = len(w) // hop
        db = 20 * np.log10(np.sqrt(np.mean(w[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12) + 1e-12)
        cuts, a = [0], 0
        while n - a > 10 * SR // hop:
            lo, hi = a + 4 * SR // hop, a + 10 * SR // hop
            a = lo + int(np.argmin(db[lo:hi]))
            cuts.append(a)
        cuts.append(n)
        pieces = [(cuts[i] * hop, cuts[i + 1] * hop) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] >= SR // hop]
        per = [(s, e, self._score(w[s:e], None, sizes)) for s, e in pieces]
        out = {}
        for sz in sizes:
            z = float(np.average([r[sz]["logit"] for _, _, r in per], weights=[e - s for s, e, _ in per]))
            out[sz] = {"logit": z, "probability": 1 / (1 + math.exp(-z)), "index": index(z), "size": sz, "mode": "audio",
                       "pieces": [(s / SR, e / SR, r[sz]["logit"]) for s, e, r in per]}
        return out


def load(repo: str = REPO, revision: str | None = None, device: str | None = None) -> AccentDetective:
    """Shorthand for AccentDetective.from_pretrained."""
    return AccentDetective.from_pretrained(repo, revision, device)
