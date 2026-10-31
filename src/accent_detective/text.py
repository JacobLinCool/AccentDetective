"""From text to the syllable features of the optional text readout.

Every Han character gets its Mainland reading (pypinyin on the simplified form), its Taiwan reading (the Ministry of
Education's Concised Mandarin Dictionary, 教育部《國語辭典簡編本》, for multi-character words with a single reading), and
whether Mainland speech may make it a neutral tone (CC-CEDICT, including its "also pronounced" variants). Words are
segmented with jieba. No audio is needed.
"""
from __future__ import annotations

import gzip
import json
import lzma
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

CJK = re.compile(r"[㐀-鿿豈-﫿]")
_PAREN = re.compile(r"[（(][^）)]*[）)]")
LETTERS = "abcdefghijklmnopqrstuvwxyz"                 # pinyin bodies use a–z, ü written as v


def tone3(syllable: str) -> str:
    """Diacritic or numbered pinyin → numbered, neutral tone as 5, ü as v."""
    from pypinyin.contrib.tone_convert import to_tone3

    s = to_tone3(syllable.strip().lower(), neutral_tone_with_five=True, v_to_u=False)
    return s.replace("ü", "v").replace("u:", "v")


def split_syllable(syl: str) -> tuple[str, str, str]:
    """Numbered pinyin → (initial or "_", final, tone); a syllable without a digit is neutral (5)."""
    from pypinyin.contrib.tone_convert import to_finals, to_initials

    body, tone = (syl[:-1], syl[-1]) if syl[-1].isdigit() else (syl, "5")
    return to_initials(body, strict=True) or "_", to_finals(body, strict=True, v_to_u=False), tone


def sandhi_tones(tones: list[str]) -> list[str]:
    """Third-tone sandhi: a third tone before another third tone is realised as a second tone."""
    out = list(tones)
    for k in range(len(out) - 2, -1, -1):
        if out[k] == "3" and out[k + 1] == "3":
            out[k] = "2"
    return out


def _tw_syllables(raw: str, n_chars: int) -> list[str] | None:
    """A dictionary pinyin field → numbered syllables, or None if it does not map one-to-one onto the characters."""
    raw = unicodedata.normalize("NFC", _PAREN.sub(" ", raw)).replace("-", " ").replace("　", " ")
    parts = [p for p in raw.split() if p]
    if len(parts) != n_chars or not all(re.fullmatch(r"[a-zǜ-ͯÀ-ɏ]+", p) for p in parts):
        return None
    return [tone3(p) for p in parts]


class Readings:
    """Taiwan and Mainland readings of words, and the segmentation that finds the words."""

    def __init__(self, dictionary: str | Path, cedict: str | Path):
        import opencc

        self._t2s = opencc.OpenCC("tw2s")
        self._s2t = opencc.OpenCC("s2t")
        path = Path(dictionary)
        with (lzma.open if path.suffix == ".xz" else open)(path, "rt", encoding="utf-8") as f:
            entries = json.load(f)
        self.tw: dict[str, list[str]] = {}
        self.ambiguous: set[str] = set()
        for e in entries:
            title = e.get("title", "")
            if not title or not all(CJK.fullmatch(c) for c in title):
                continue
            readings = []
            for h in e.get("heteronyms", []):
                syl = _tw_syllables(h.get("pinyin", ""), len(title))
                if syl is not None and syl not in readings:
                    readings.append(syl)
            if len(readings) == 1:
                self.tw[title] = readings[0]
            elif len(readings) > 1:
                self.ambiguous.add(title)
        self.max_word = max(len(w) for w in self.tw)
        self.neutral = _neutral_tones(str(cedict))

    def cn_reading(self, word: str) -> list[str]:
        from pypinyin import Style, pinyin

        simp = self._t2s.convert(word)
        return [tone3(p[0]) for p in pinyin(simp, style=Style.TONE3, neutral_tone_with_five=True,
                                             errors=lambda x: ["?"] * len(x))]

    def segment(self, text: str) -> list[tuple[int, str]]:
        """Word segmentation with jieba on the simplified form, mapped back onto `text`; a token missing from the
        dictionary is split again by greedy longest match."""
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

    def char_readings(self, text: str) -> list[tuple[str, str, str, bool]]:
        """(character, Mainland reading, Taiwan reading, Mainland neutral?) for every Han character."""
        trad = self._s2t.convert(text)
        if len(trad) != len(text):
            trad = text
        out = []
        for off, piece in self.segment(trad):
            if not all("㐀" <= c <= "鿿" for c in piece):
                continue
            cn = self.cn_reading(piece)
            tw = self.tw.get(piece) if len(piece) > 1 else None
            tw = tw if tw and len(tw) == len(piece) else cn
            neu = self.neutral.get(piece, frozenset()) if len(piece) > 1 else frozenset()
            out += [(text[off + k] if off + k < len(text) else c, a, b, k in neu)
                    for k, (c, a, b) in enumerate(zip(piece, cn, tw))]
        return out


@lru_cache(maxsize=2)
def _neutral_tones(path: str) -> dict[str, frozenset[int]]:
    """Word → positions that are neutral in any CC-CEDICT entry or "also pronounced" variant."""
    line_re = re.compile(r"^(\S+) (\S+) \[([^\]]+)\] /(.*)/$")
    also = re.compile(r"also pr\. \[([^\]]+)\]")
    entries: dict[str, list[set[int]]] = {}
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        for line in f:
            m = line_re.match(line.rstrip("\n"))
            if not m:
                continue
            trad, simp, py, gloss = m.groups()
            if len(trad) < 2 or not all("㐀" <= c <= "鿿" for c in trad):
                continue
            for r in [py] + also.findall(gloss):
                syl = re.findall(r"[a-zA-Zü:]+[1-5]", r.replace(" ", ""))
                if len(syl) != len(trad):
                    continue
                pos = {k for k, s in enumerate(syl) if s.endswith("5")}
                for w in {trad, simp}:
                    entries.setdefault(w, []).append(pos)
    return {w: frozenset(set.union(*sets)) for w, sets in entries.items() if set.union(*sets)}


def _letters(syl: str) -> tuple[str, int]:
    """Toneless pinyin letters of a syllable and the number of its initial-consonant letters."""
    from pypinyin.contrib.tone_convert import to_initials

    body = syl.rstrip("12345").replace("ü", "v").replace("u:", "v")
    return body, len(to_initials(body, strict=False) or "")


def text_syllables(readings: Readings, text: str, vocab: dict) -> tuple[list[int], list[list[float]]] | None:
    """The text's syllables for TextReadout: (syllable type ids, context rows of 7), or None when the text has fewer
    than two Han syllables or a reading outside plain pinyin letters."""
    sy = []
    for ch, cn, tw, neu in readings.char_readings(text):
        letters, _ = _letters(cn)
        if not letters or any(c not in LETTERS for c in letters):
            return None
        sy.append((ch, cn, neu, tw))
    if len(sy) < 2:
        return None
    cn_t = [split_syllable(s[1])[2] for s in sy]
    tw_t = [split_syllable(s[3])[2] for s in sy]
    n = len(sy)
    types = [vocab.get(s[1].rstrip("012345"), 0) for s in sy]
    ctx = [[int(a), int(b), int(s[2]), int(c), int(s[3] != s[1]), int(k == n - 1), k / max(n - 1, 1)]
           for k, (s, a, b, c) in enumerate(zip(sy, cn_t, sandhi_tones(cn_t), tw_t))]
    return types, ctx
