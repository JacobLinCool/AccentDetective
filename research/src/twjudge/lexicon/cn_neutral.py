"""Mainland neutral-tone (輕聲) readings from CC-CEDICT (CC BY-SA 4.0).

pypinyin almost never marks the neutral tone (知道 → zhi1 dao4), so it cannot tell where
Mainland speech reduces a syllable. CC-CEDICT follows Mainland usage and marks it (認識
ren4 shi5, 時候 shi2 hou5), and lists "also pr." alternatives (知道 also zhi1dao5). Keyed by both
traditional and simplified forms, a multi-character word maps to its neutral-tone positions:

  mode="all"  positions neutral in every CEDICT entry, ignoring "also pr.": an unconditional
              Mainland neutral tone (excludes 不是 "is not" vs bùshi "fault",
              東西 "east and west" vs "thing", and 知道 whose main entry has dào)
  mode="any"  positions neutral in any entry or "also pr." variant: where Mainland speech may
              reduce the syllable, used as a context feature by the accent judge

    python -m twjudge.lexicon.cn_neutral data/lexicon/cedict.txt.gz   # prints a few lookups
"""

from __future__ import annotations

import gzip
import re
import sys
from functools import lru_cache
from pathlib import Path

import os

DEFAULT = os.environ.get("ACCENT_DETECTIVE_CEDICT", "data/lexicon/cedict.txt.gz")
LINE = re.compile(r"^(\S+) (\S+) \[([^\]]+)\] /(.*)/$")


def _syllables(pinyin: str) -> list[str]:
    return re.findall(r"[a-zA-Zü:]+[1-5]", pinyin.replace(" ", ""))


ALSO = re.compile(r"also pr\. \[([^\]]+)\]")


@lru_cache(maxsize=2)
def load(path: str = DEFAULT, mode: str = "all") -> dict[str, frozenset[int]]:
    entries: dict[str, list[set[int]]] = {}
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line.rstrip("\n"))
            if not m:
                continue
            trad, simp, py, gloss = m.groups()
            if len(trad) < 2 or not all("㐀" <= c <= "鿿" for c in trad):
                continue
            for r in [py] + (ALSO.findall(gloss) if mode == "any" else []):
                syl = _syllables(r)
                if len(syl) != len(trad):
                    continue
                pos = {k for k, s in enumerate(syl) if s.endswith("5")}
                for w in {trad, simp}:
                    entries.setdefault(w, []).append(pos)
    out = {}
    for w, sets in entries.items():
        keep = set.intersection(*sets) if mode == "all" else set.union(*sets)
        if keep:
            out[w] = frozenset(keep)
    return out


def neutral_positions(word: str, mode: str = "all", path: str = DEFAULT) -> frozenset[int]:
    return load(path, mode).get(word, frozenset())


if __name__ == "__main__":
    d = load(sys.argv[1] if len(sys.argv) > 1 else "data/lexicon/cedict.txt.gz")
    print(len(d), "words with a neutral tone")
    for w in ["知道", "認識", "東西", "意思", "時候", "先生", "漂亮", "企業", "衣服", "朋友"]:
        print(w, sorted(d.get(w, [])))
