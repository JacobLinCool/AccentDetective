"""Command line: score audio files and print one JSON line per file.

    accent-detective clip.wav other.mp3
    accent-detective clip.wav --text "這麼長的時間" --size nano
    accent-detective talk.wav --all-sizes
"""
from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="accent-detective", description="How Taiwanese a Mandarin recording sounds.")
    ap.add_argument("audio", nargs="+", help="audio files")
    ap.add_argument("--text", help="the text the recording reads (optional; one text for all files)")
    ap.add_argument("--size", help="nano, tiny, mini or small (default: the largest)")
    ap.add_argument("--all-sizes", action="store_true", help="report every size from one pass")
    ap.add_argument("--device", help="cpu or cuda (default: cuda when available)")
    ap.add_argument("--revision", help="checkpoint revision on the Hugging Face Hub")
    a = ap.parse_args(argv)
    from accent_detective import load

    det = load(revision=a.revision, device=a.device)
    for path in a.audio:
        sizes = list(det.sizes) if a.all_sizes else [a.size]
        res = det.score_recording(path, a.text, sizes=sizes)
        out = {s: {k: v for k, v in r.items() if k != "pieces"} for s, r in res.items()}
        print(json.dumps({"file": path, **(out if a.all_sizes else next(iter(out.values())))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
