"""Shared helpers of the Common Voice corpus code: hashing of speaker and clip identifiers, and text normalisation.

The paper's exact splits are distributed as clip-ID lists (research/splits); the salt only matters when the corpus is
rebuilt from scratch.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

SALT = "accent-detective-cv25"
ALLOWED = re.compile(r"^[㐀-鿿豈-﫿，。、！？；：「」『』（）—…·．,.!?;:\s]+$")
STRIP = re.compile(r"[^㐀-鿿豈-﫿]")


def h(text: str) -> str:
    """Salted SHA-256 hex digest (speaker IDs use the first 16 characters)."""
    return hashlib.sha256(f"{SALT}|{text}".encode()).hexdigest()


def norm_text(s: str) -> str:
    """Han characters only, NFKC-normalised: the key that keeps a sentence in one split."""
    return STRIP.sub("", unicodedata.normalize("NFKC", s))
