"""A second, untouched test set of real speech: Common Voice speakers the judges never saw.

  Taiwan    Common Voice 25 zh-TW validated speakers outside the judge corpus (its train / dev / test
            stopped filling before reaching them), with the corpus's own filters
  Mainland  Common Voice 21 zh-CN dev + test speakers outside the judge corpus (the corpus kept only
            part of these splits), with the corpus's own filters; never in any distillation audio

At most PER_SPK clips per speaker, the Mainland side capped at the size of the Taiwan side. The
Taiwan speakers were part of the first student's unlabelled distillation audio, so they are written
to heldout_speakers.txt for later distillation runs to exclude. Also writes synthesis items
(--synth-texts sentences × the voices of --refs) for a larger BreezyVoice / CosyVoice contrast.

Writes, in the benchmark bundle layout: sets/cv_heldout.jsonl, audio/cv_heldout/*.flac.

    python research/scripts/heldout_set.py --bundle /content/bench --synth-out /content/bench/synth2_items.jsonl
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))

TW, CN = "OpenFormosa/common_voice_25_zh-TW", "keeve101/common-voice-21.0-2025-03-14-zh-CN-split"
PER_SPK = 8


def rank(key: str) -> str:
    return hashlib.sha256(f"heldout-v1|{key}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--synth-out", required=True)
    ap.add_argument("--synth-texts", type=int, default=250)
    ap.add_argument("--voices", default="灵小棠,灵小川,v_c3be9b,v_4e2da8")
    a = ap.parse_args()
    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download, snapshot_download

    from twjudge.data.build_corpus import CN_EXCLUDE, TW_FOREIGN, _ok_text
    from twjudge.data.common import h

    B = Path(a.bundle)
    used = set()
    for s in ("cv_train", "cv_dev", "cv_test"):
        used |= {json.loads(l)["speaker"] for l in open(B / "sets" / f"{s}.jsonl")}
    # Taiwan: filter on validated.tsv metadata, audio from the validated parquets
    tsv = hf_hub_download(TW, "source/validated.tsv", repo_type="dataset")
    tw = collections.defaultdict(list)
    for r in csv.DictReader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = r["sentence"].strip()
        if int(r["up_votes"]) < 2 or int(r["down_votes"]) > 0 or TW_FOREIGN.search(r["accents"]) or not _ok_text(text):
            continue
        spk = "tw" + h(r["client_id"])[:16]
        if spk not in used:
            tw[spk].append((r["path"], text))
    want = {}
    for spk, clips in tw.items():
        for p, t in sorted(clips, key=lambda c: rank(c[0]))[:PER_SPK]:
            want[p.split("/")[-1]] = ("tw", spk, t)
    n_tw = len(want)
    # Mainland: dev + test splits, speakers outside the corpus
    src = Path(snapshot_download(CN, repo_type="dataset", allow_patterns=["data/dev-*.parquet", "data/test-*.parquet"]))
    cn = collections.defaultdict(list)
    for f in sorted(src.glob("data/*.parquet")):
        t = pq.read_table(f, columns=["client_id", "path", "sentence", "accents", "down_votes"])
        for r in t.to_pylist():
            acc = r["accents"] or ""
            text = unicodedata.normalize("NFKC", r["sentence"]).strip()
            if r["down_votes"] > 0 or CN_EXCLUDE.search(acc) or (acc and "出生地" not in acc) or not _ok_text(text):
                continue
            spk = "cn" + hashlib.sha256(r["client_id"].encode()).hexdigest()[:16]
            if spk not in used:
                cn[spk].append((r["path"], text))
    n_cn = 0
    for spk in sorted(cn, key=rank):
        for p, t in sorted(cn[spk], key=lambda c: rank(c[0]))[:PER_SPK]:
            want[p.split("/")[-1]] = ("cn", spk, t)
            n_cn += 1
        if n_cn >= n_tw:
            break
    print(f"taiwan {n_tw} clips / {len(tw)} speakers, mainland {n_cn} clips", flush=True)
    # audio
    out = B / "audio" / "cv_heldout"
    out.mkdir(parents=True, exist_ok=True)
    tw_src = Path(snapshot_download(TW, repo_type="dataset", allow_patterns=["data/validated-*.parquet"]))
    rows, k = [], 0
    for f in sorted(tw_src.glob("data/validated-*.parquet")) + sorted(src.glob("data/*.parquet")):
        for batch in pq.ParquetFile(f).iter_batches(columns=["path", "audio"], batch_size=256):
            for p, au in zip(batch.column("path").to_pylist(), batch.column("audio").to_pylist()):
                name = p.split("/")[-1]
                if name not in want:
                    continue
                lab, spk, text = want.pop(name)
                w, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32", always_2d=True)
                w = librosa.resample(w.mean(1), orig_sr=sr, target_sr=16000) if sr != 16000 else w.mean(1)
                sf.write(out / f"{k:05d}.flac", w, 16000)
                stem = name.rsplit(".", 1)[0]
                rows.append({"item_id": f"{lab}_{stem}", "audio": f"audio/cv_heldout/{k:05d}.flac", "text": text,
                             "speaker": spk, "label": lab})
                k += 1
    with open(B / "sets" / "cv_heldout.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (B / "heldout_speakers.txt").write_text("\n".join(sorted({r["speaker"] for r in rows if r["label"] == "tw"})) + "\n")
    print(f"wrote {len(rows)} clips, {len(want)} not found", flush=True)
    # synthesis items: distinct held-out Taiwan sentences of 8–20 characters
    texts, seen = [], set()
    for r in sorted((r for r in rows if r["label"] == "tw"), key=lambda r: rank(r["item_id"])):
        key = re.sub(r"[^㐀-鿿]", "", r["text"])
        if 8 <= len(key) <= 20 and key not in seen:
            seen.add(key); texts.append((r["item_id"], r["text"]))
        if len(texts) >= a.synth_texts:
            break
    with open(a.synth_out, "w") as f:
        for tid, text in texts:
            for v in a.voices.split(","):
                f.write(json.dumps({"item_id": f"{tid}__{v}", "text": text, "voice": v, "text_id": tid},
                                   ensure_ascii=False) + "\n")
    print(f"synthesis items: {len(texts)} texts", flush=True)


if __name__ == "__main__":
    main()
