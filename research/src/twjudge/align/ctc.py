"""MFA-free syllable alignment for the accent judges: CTC forced alignment with MMS.

MFA costs ~15 s per call (process start-up, dictionary compilation, feature extraction) whatever
the audio length, which dominates interactive scoring. This aligner keeps one network in memory:
torchaudio's MMS forced-alignment model (wav2vec 2.0, romanised characters, 20 ms frames).

  text      every Han character gets its Mainland reading (pypinyin, as the judges' canonical
            label) and the Mainland-neutral flag from CC-CEDICT, exactly as in `twjudge.align.mfa`
  tokens    the toneless pinyin letters of each syllable (ü as v), aligned in one CTC pass
  spans     CTC emissions are peaky, so a syllable runs from its first letter to the next
            syllable's first letter, trimmed where the energy falls into a pause; the rhyme
            starts at the first letter after the initial consonant

The emissions can come from any CTC model over the same letters: ``CTCAligner(rw, emitter=...)``
takes a function ``wav16k -> (log-probabilities (1, T, V) on CPU, letter → index)`` at 20 ms frames;
`twjudge.accent.compact` uses a small head on the judge's own SSL states instead of MMS.

Output rows have the fields of `twjudge.align.mfa` (`char`, `syl`, `tw`, `cn_neutral`, `start`, `end`,
`rhyme_start`, `variant`); the realised reading variant is not decided here and is always "cn", so
judge A's variant features are uninformative with this aligner (judges B and C do not use them).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

HOP_S = 0.02
PAUSE_DB = -35.0          # frames this far below the utterance's loudest frame count as silence


@lru_cache(maxsize=1)
def _bundle():
    import torch
    import torchaudio

    b = torchaudio.pipelines.MMS_FA
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = b.get_model(with_star=False).to(dev).eval()
    return model, b.get_dict(star=None), dev


def mms_emissions(wav16k: np.ndarray):
    import torch

    model, d, dev = _bundle()
    x = torch.tensor(wav16k, dtype=torch.float32, device=dev)[None]
    with torch.inference_mode():
        em, _ = model(x)
        em = torch.log_softmax(em, dim=-1)
    # forced_align runs on CPU (no MPS kernel); the network above uses the accelerator
    return em.float().cpu(), d


class CTCAligner:
    def __init__(self, readings, emitter=None):
        import opencc

        from twjudge.align.mfa import JudgeAligner

        # reuse the reading logic (Mainland reading, Taiwan reading, Mainland-neutral flag)
        self._readings = JudgeAligner.char_readings
        self.rw = readings
        self._s2t = opencc.OpenCC("s2t")
        self.emitter = emitter or mms_emissions

    def char_readings(self, text):
        return self._readings(self, text)

    @staticmethod
    def _letters(syl: str) -> tuple[str, int]:
        """Toneless pinyin letters and the number of initial-consonant letters."""
        from pypinyin.contrib.tone_convert import to_initials

        body = syl.rstrip("12345").replace("ü", "v").replace("u:", "v")
        ini = to_initials(body, strict=False) or ""
        return body, len(ini)

    def syllables(self, text: str, vocab) -> list[tuple] | None:
        """(char, Mainland reading, neutral flag, letters, n initial letters, Taiwan reading) per syllable."""
        sylls = []
        for ch, cn, tw, neu in self.char_readings(text):
            letters, n_ini = self._letters(cn)
            if not letters or any(c not in vocab for c in letters):
                return None
            sylls.append((ch, cn, neu, letters, n_ini, tw))
        return sylls if len(sylls) >= 2 else None

    def align(self, wav16k: np.ndarray, text: str, emissions=None) -> list[dict] | None:
        """Syllable spans; `emissions` (log-probs, vocab) can be passed when already computed."""
        em, d = emissions if emissions is not None else self.emitter(wav16k)
        sylls = self.syllables(text, d)
        if sylls is None:
            return None
        return self.spans(em, d, sylls, wav16k)

    @staticmethod
    def spans(em, d, sylls, wav16k: np.ndarray) -> list[dict] | None:
        import torch
        import torchaudio.functional as F

        targets = [d[c] for s in sylls for c in s[3]]
        T = em.shape[1]
        if T < len(targets):
            return None
        ali, scores = F.forced_align(em, torch.tensor([targets], dtype=torch.int32), blank=0)
        spans = F.merge_tokens(ali[0], scores[0].exp())       # one span per target token
        if len(spans) != len(targets):
            return None
        first = [sp.start for sp in spans]
        last = [sp.end - 1 for sp in spans]
        # energy envelope at the same 20 ms hop, to cut syllables at pauses
        frame = int(16000 * HOP_S)
        n = len(wav16k) // frame
        rms = np.sqrt(np.mean(wav16k[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
        db = 20 * np.log10(rms / rms.max())
        voiced = db > PAUSE_DB
        out, pos = [], 0
        for i, (ch, cn, neu, letters, n_ini, tw) in enumerate(sylls):
            a = first[pos]
            r = first[pos + n_ini] if n_ini < len(letters) else a
            e_tok = last[pos + len(letters) - 1]
            pos += len(letters)
            nxt = first[pos] if pos < len(first) else min(n, T)
            e = max(e_tok + 1, a + 1)
            while e < nxt and e < len(voiced) and voiced[e]:   # extend through the voiced tail
                e += 1
            out.append({"char": ch, "syl": cn, "tw": tw, "cn_neutral": neu, "start": a * HOP_S,
                        "rhyme_start": r * HOP_S, "end": e * HOP_S, "variant": "cn", "n_variants": 1})
        return out

