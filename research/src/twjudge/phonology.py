"""Pinyin syllable helpers shared by the aligners, features and judges."""

from __future__ import annotations


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
