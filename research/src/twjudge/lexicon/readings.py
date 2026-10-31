"""Taiwan and Mainland standard readings of Chinese words.

Readings:
  Taiwan    — 教育部《國語辭典簡編本》 (CC BY-ND 3.0 TW), the modern dictionary that
              follows the 一字多音審訂表; 《重編國語辭典修訂本》 keeps literary and
              older readings (它們 tuō men, 市場 shì cháng) that would sound unnatural.
              Only multi-character words with a single reading are kept, so
              context-dependent polyphones are never guessed.
  Mainland  — pypinyin's phrase dictionary on the simplified form. pypinyin under-marks neutral
              tones; ``twjudge.lexicon.cn_neutral`` adds them from CC-CEDICT where needed.

Tone sandhi of 一 and 不 is a phonological rule both sides apply, not a lexical difference (the
dictionary lists citation tones), so ``same_reading`` ignores the tone of these characters.
``neutral_only`` marks a difference only in whether a syllable is neutral.

The aligners and reading judges of this package use ``TaiwanReadings`` for word segmentation and
both readings.
"""

from __future__ import annotations

import json
import lzma
import re
import unicodedata
from pathlib import Path

from pypinyin import Style, pinyin
from pypinyin.contrib.tone_convert import to_tone3

CJK = re.compile(r"[㐀-鿿豈-﫿]")
_PAREN = re.compile(r"[（(][^）)]*[）)]")


def tone3(syllable: str) -> str:
    """Diacritic or numbered pinyin → numbered, neutral tone as 5, ü as v."""
    s = to_tone3(syllable.strip().lower(), neutral_tone_with_five=True, v_to_u=False)
    return s.replace("ü", "v").replace("u:", "v")


def tw_syllables(raw: str, n_chars: int) -> list[str] | None:
    """Parse a moedict pinyin field; None if it does not map 1:1 onto the characters."""
    raw = _PAREN.sub(" ", raw)
    raw = unicodedata.normalize("NFC", raw).replace("-", " ").replace("　", " ")
    parts = [p for p in raw.split() if p]
    if len(parts) != n_chars or not all(re.fullmatch(r"[a-zǜ-ͯÀ-ɏ]+", p) for p in parts):
        return None
    return [tone3(p) for p in parts]


SANDHI_CHARS = set("一不")


def same_reading(t: str, c: str, char: str) -> bool:
    if t == c:
        return True
    if char in SANDHI_CHARS or char in "壹":
        return t.rstrip("12345") == c.rstrip("12345")
    return False


def neutral_only(t: str, c: str) -> bool:
    return t.rstrip("12345") == c.rstrip("12345") and ("5" in (t[-1], c[-1]))


class TaiwanReadings:
    def __init__(self, moedict_path: str | Path):
        import opencc

        self._t2s = opencc.OpenCC("tw2s")
        path = Path(moedict_path)
        opener = lzma.open if path.suffix == ".xz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            entries = json.load(f)
        self.tw: dict[str, list[str]] = {}
        self.ambiguous: set[str] = set()
        for e in entries:
            title = e.get("title", "")
            if not title or not all(CJK.fullmatch(c) for c in title):
                continue
            readings = []
            for h in e.get("heteronyms", []):
                syl = tw_syllables(h.get("pinyin", ""), len(title))
                if syl is not None and syl not in readings:
                    readings.append(syl)
            if len(readings) == 1:
                self.tw[title] = readings[0]
            elif len(readings) > 1:
                self.ambiguous.add(title)
        self.max_word = max(len(w) for w in self.tw)

    # -- Mainland side ----------------------------------------------------
    def cn_reading(self, word: str) -> list[str]:
        simp = self._t2s.convert(word)
        return [tone3(p[0]) for p in pinyin(simp, style=Style.TONE3, neutral_tone_with_five=True,
                                             errors=lambda x: ["?"] * len(x))]

    # -- segmentation ------------------------------------------------------
    def segment(self, text: str) -> list[tuple[int, str]]:
        """Word segmentation with jieba on the simplified form, mapped back onto ``text``.

        Greedy longest match over dictionary titles would join words across
        boundaries (年的歷史 → 的歷), so jieba decides the boundaries. A jieba
        token missing from the dictionary (下星期, 星期三) is split again by
        greedy matching inside the token. Falls back to greedy matching over the
        whole text only if conversion changed the length.
        """
        import jieba

        simp = self._t2s.convert(text)
        if len(simp) != len(text):
            return self._greedy(text, 0)
        out, i = [], 0
        for tok in jieba.cut(simp, HMM=False):
            piece = text[i:i + len(tok)]
            if len(piece) > 1 and piece not in self.tw and piece not in self.ambiguous:
                out.extend(self._greedy(piece, i))
            else:
                out.append((i, piece))
            i += len(tok)
        return out

    def _greedy(self, text: str, offset: int) -> list[tuple[int, str]]:
        out, i = [], 0
        while i < len(text):
            for n in range(min(self.max_word, len(text) - i), 0, -1):
                piece = text[i:i + n]
                if n == 1 or piece in self.tw or piece in self.ambiguous:
                    out.append((offset + i, piece))
                    i += n
                    break
        return out
