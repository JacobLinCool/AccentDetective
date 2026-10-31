"""Accent Detective: how Taiwanese does a Mandarin recording sound?

The public entry point of the compact judge (`twjudge.accent.compact`): one small network aligns the
syllables of the given text to the audio and scores how Taiwan-Mandarin-like their realisation is.

    from twjudge.detective import Detective
    det = Detective("model.pt", lexicon="lexicon/concised.json.xz", cedict="lexicon/cedict.txt.gz")
    det.score("clip.wav", "這麼長的時間")              # one utterance
    det.score_recording("talk.wav", "第一句。第二句。")   # sentence by sentence, split at punctuation
    Detective(..., mode="fast")                        # the wav2vec 2.0 path only: faster, slightly less accurate

Scores are log-odds of Taiwan vs Mainland (0 = undecided) and `index`, the same value on a 0–1 scale
(0.5 + log-odds / 20, clipped): +0.1 is about 7× the odds. Per-syllable values say how much each
syllable pulls the judgement either way; they are not calibrated probabilities.

`NestedDetective` loads the nested checkpoint (twjudge.accent.nested): one network, four sizes,
the text optional and no alignment.

    det = NestedDetective("model.safetensors", lexicon="lexicon/concised.json.xz", cedict="lexicon/cedict.txt.gz")
    det.score("clip.wav")                                # no text, the largest size
    det.score("clip.wav", "這麼長的時間", size="tiny")     # with the text, the second size
    det.score_sizes("clip.wav")                          # all four sizes from one pass
    det.score_recording("talk.wav")                      # a long recording, cut at pauses
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

SR = 16000
SENT_END = set("。！？；!?;")
CLAUSE = set("，、,：:")


def index(logit: float) -> float:
    """Log-odds on a 0–1 display scale: 0.5 = undecided, ±0.1 = 7.4× (e²) odds, clipped at ±10."""
    return min(1.0, max(0.0, 0.5 + logit / 20))


def sentence_ids(text: str, max_len: int = 25) -> list[int]:
    """Sentence index of every Han character: split at sentence-final punctuation, and at commas once a
    sentence is longer than `max_len` characters (the judge was trained on short sentences)."""
    ids, cur, n = [], 0, 0
    for ch in text:
        if "㐀" <= ch <= "鿿":
            ids.append(cur)
            n += 1
        elif ch in SENT_END or (ch in CLAUSE and n >= max_len):
            if n:
                cur, n = cur + 1, 0
    return ids


def _needs_eager(path) -> bool:
    """CUDA graphs cover single-path, flat checkpoints; nested or dual (Whisper-path) ones run eagerly."""
    if not str(path).endswith(".safetensors"):
        return False
    import json

    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["accent_detective"])
    return bool(meta.get("wjudge") or meta.get("judge_nested"))


class Detective:
    def __init__(self, model: str | Path, lexicon: str | Path, cedict: str | Path | None = None,
                 device: str | None = None, cuda_graphs: bool = True, mode: str = "accurate"):
        if cedict is not None:
            os.environ["ACCENT_DETECTIVE_CEDICT"] = str(cedict)       # read by twjudge.lexicon.cn_neutral
        import torch

        from twjudge.lexicon.readings import TaiwanReadings

        readings = TaiwanReadings(lexicon)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and cuda_graphs and not _needs_eager(model):
            from twjudge.accent.compact_fast import FastCompactJudge

            self.judge = FastCompactJudge(model, readings=readings)
        else:
            from twjudge.accent.compact import CompactJudge

            self.judge = CompactJudge(model, readings=readings, device=device, mode=mode)

    @staticmethod
    def _audio(wav) -> np.ndarray:
        if isinstance(wav, (str, Path)):
            import librosa

            return librosa.load(str(wav), sr=SR, mono=True)[0]
        return np.asarray(wav, dtype=np.float32)

    def score(self, wav, text: str) -> dict | None:
        """One utterance (16 kHz array or audio file) and the text it reads → {"logit", "index",
        "syllables": [{"char", "logit", "start", "end"}]}, or None if the text cannot be aligned."""
        out = self.judge.score(self._audio(wav), text)
        if out is None or len(out[2]) < 2:
            return None
        z, syl, chars = out
        return {"logit": float(z), "index": index(float(z)),
                "syllables": [{"char": c["char"], "logit": float(v), "start": c["start"], "end": c["end"]}
                              for c, v in zip(chars, syl)]}

    def score_recording(self, wav, text: str) -> dict | None:
        """A longer recording: aligned once, cut at the sentence boundaries of the text, each sentence
        scored on its own. Returns {"logit", "index", "sentences": [{"text", ...score fields}]}, the overall
        log-odds weighting each sentence by its number of syllables."""
        w = self._audio(wav)
        full = self.score(w, text)
        if full is None:
            return None
        sid = sentence_ids(text)
        syls = full["syllables"]
        if len(sid) != len(syls) or len(set(sid)) == 1:
            sents = [{"text": "".join(s["char"] for s in syls), **full}]
        else:
            sents = []
            for k in sorted(set(sid)):
                part = [s for s, i in zip(syls, sid) if i == k]
                if len(part) < 2:
                    continue
                a = max(0.0, part[0]["start"] - 0.1)
                b = min(len(w) / SR, part[-1]["end"] + 0.1)
                t = "".join(s["char"] for s in part)
                r = self.score(w[int(a * SR):int(b * SR)], t)
                if r is not None:
                    sents.append({"text": t, **r})
        if not sents:
            return None
        z = float(np.average([s["logit"] for s in sents], weights=[len(s["syllables"]) for s in sents]))
        return {"logit": z, "index": index(z), "sentences": sents}


def checkpoint_format(path) -> str:
    """"nested-asp-text" for a NestedDetective checkpoint, "compact" otherwise."""
    if not str(path).endswith(".safetensors"):
        return "compact"
    import json

    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return json.loads(f.metadata()["accent_detective"]).get("format", "compact")


class NestedDetective:
    """The nested Accent Detective: a Conformer trunk with exits at layers 3, 6, 9 and 18. Size k runs the trunk up to
    exit k and that exit's readout, so every smaller size is a part of every larger one and one pass to the last exit
    gives all four. Without text the readout pools the whole utterance (ASPReadout); with the text, TextReadout adds a
    correction from the syllables the text implies. Scores are log-odds of Taiwan vs Mainland after a per-size,
    per-mode Platt transform fitted on development data; `index` is the same value on a 0–1 scale (see `index`)."""

    # public names of the exits, smallest first (the research name of exit k is N<k>); larger future models can take
    # the next names, and a checkpoint may carry its own under "names"
    NAMES = ("nano", "tiny", "mini", "small", "base", "medium", "large")
    MAX_SECONDS = 15.0                   # longer audio goes through score_recording's pause-cut chunks

    def __init__(self, model: str | Path, lexicon: str | Path, cedict: str | Path | None = None, device: str | None = None):
        if cedict is not None:
            os.environ["ACCENT_DETECTIVE_CEDICT"] = str(cedict)       # read by twjudge.lexicon.cn_neutral
        import json

        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file

        from twjudge.accent.nested import ASPReadout, NestedStudent, TextReadout
        from twjudge.align.ctc import CTCAligner
        from twjudge.lexicon.readings import TaiwanReadings

        with safe_open(str(model), framework="pt") as f:
            self.meta = json.loads(f.metadata()["accent_detective"])
        st = load_file(str(model))
        c = self.meta["config"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = NestedStudent(c["exits"], c["targets"], d=c["d"], heads=c["heads"], ff=c["ff"], ctc=False)
        self.net.load_state_dict({k[6:]: v for k, v in st.items() if k.startswith("trunk.")})
        self.asp, self.text = torch.nn.ModuleList(), torch.nn.ModuleList()
        for k in range(1, len(c["exits"]) + 1):
            a, t = ASPReadout(), TextReadout(self.meta["n_types"])
            a.load_state_dict({n[len(f"asp.{k}."):]: v for n, v in st.items() if n.startswith(f"asp.{k}.")})
            t.load_state_dict({n[len(f"text.{k}."):]: v for n, v in st.items() if n.startswith(f"text.{k}.")})
            self.asp.append(a); self.text.append(t)
        for m in (self.net, self.asp, self.text):
            m.to(self.device).eval()
        self.vocab = self.meta["vocab"]
        self.aligner = CTCAligner(TaiwanReadings(lexicon))          # its readings only: text → syllables, no audio
        self.keys = list(self.meta["sizes"])                          # N1 … : the keys of the Platt transforms
        self.SIZES = tuple(self.meta.get("names") or self.NAMES[: len(self.keys)])

    def _k(self, size: str | None) -> int:
        """Exit index of a size: its name ("nano" …), its research name ("N1" …) or None for the largest."""
        if size is None:
            return len(self.keys) - 1
        if size in self.keys:
            return self.keys.index(size)
        if size.lower() in self.SIZES:
            return self.SIZES.index(size.lower())
        raise ValueError(f"unknown size {size!r}: one of {', '.join(self.SIZES)}")

    def parameters(self, size: str | None = None, text: bool = False) -> int:
        """Parameters that run for one size and mode."""
        k = self._k(size)
        n = self.net
        mods = [n.front, n.norm_in, n.norm_out[k], n.pred[k], self.asp[k], *n.enc.conformer_layers[: n.exits[k]]]
        return sum(p.numel() for m in mods + ([self.text[k]] if text else []) for p in m.parameters())

    def _syllables(self, text):
        from twjudge.accent.nested import text_syllables

        return text_syllables(self.aligner, text, self.vocab) if text and text.strip() else None

    def _score(self, w: np.ndarray, text, sizes) -> dict:
        import torch

        syl = self._syllables(text)
        ks = [self._k(s) for s in sizes]
        out = {}
        with torch.inference_mode():
            outs = self.net(torch.tensor(w, dtype=torch.float32, device=self.device)[None], upto=max(ks) + 1)
            for k in ks:
                out[self.SIZES[k]] = self._readout(k, outs[k], w, syl)
        return out

    def score_progressive(self, wav, text: str | None = None):
        """Anytime scoring: one pass of the largest size, yielding each size's score dict (smallest first) as soon as its
        exit is reached. Each update is more accurate and later; nothing is computed twice."""
        import torch

        w = Detective._audio(wav)
        syl = self._syllables(text)
        with torch.inference_mode():
            for k, o in enumerate(self.net.iter_exits(torch.tensor(w, dtype=torch.float32, device=self.device)[None])):
                yield self._readout(k, o, w, syl)

    def _readout(self, k: int, o: dict, w: np.ndarray, syl) -> dict:
        """Exit k's outputs (batch 1) → that size's calibrated score dict."""
        import torch

        from twjudge.accent.nested import voiced_span

        H = torch.stack([o[t][0] for t in self.net.targets])
        X = (H - H.mean(1, keepdim=True)).transpose(0, 1)[None]              # (1, T, layers, 768)
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
        return {"logit": logit, "probability": 1 / (1 + math.exp(-logit)), "index": index(logit), "size": self.SIZES[k],
                "mode": mode, "raw_logit": raw}

    def score(self, wav, text: str | None = None, size: str | None = None) -> dict:
        """One utterance (16 kHz array or audio file), optionally the text it reads, at one size (default the largest)
        → {"logit", "probability", "index", "size", "mode" ("text" or "audio"), "raw_logit"}. Text without Han
        syllables falls back to the audio mode."""
        return next(iter(self._score(Detective._audio(wav), text, [size]).values()))

    def score_sizes(self, wav, text: str | None = None) -> dict:
        """Every size from one pass: {size name: score dict}."""
        return self._score(Detective._audio(wav), text, list(self.SIZES))

    def score_recording(self, wav, text: str | None = None, sizes=None) -> dict:
        """A recording of any length: up to MAX_SECONDS it is one utterance; longer audio is cut at pauses into pieces of
        4–10 s, each scored without the text (the text cannot be split without an alignment), and the log-odds are
        averaged weighted by duration. Returns {size name: score dict + "pieces": [(start s, end s, logit)]};
        sizes default to the largest."""
        sizes = [self.SIZES[self._k(s)] for s in (sizes or [None])]
        w = Detective._audio(wav)
        if len(w) <= self.MAX_SECONDS * SR:
            return {s: {**r, "pieces": [(0.0, len(w) / SR, r["logit"])]} for s, r in self._score(w, text, list(sizes)).items()}
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
        per = [(s, e, self._score(w[s:e], None, list(sizes))) for s, e in pieces]
        out = {}
        for sz in sizes:
            z = float(np.average([r[sz]["logit"] for _, _, r in per], weights=[e - s for s, e, _ in per]))
            out[sz] = {"logit": z, "probability": 1 / (1 + math.exp(-z)), "index": index(z), "size": sz, "mode": "audio",
                       "pieces": [(s / SR, e / SR, r[sz]["logit"]) for s, e, r in per]}
        return out
