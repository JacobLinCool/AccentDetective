"""Accent Detective: how Taiwanese a Mandarin recording sounds.

    from accent_detective import load
    det = load()                       # downloads the released checkpoint from the Hugging Face Hub
    det.score("clip.wav")              # {"logit", "probability", "index", "size", "mode", "raw_logit"}
"""
from accent_detective.detector import AccentDetective, index, load, load_audio

__all__ = ["AccentDetective", "index", "load", "load_audio"]
__version__ = "0.1.0"
