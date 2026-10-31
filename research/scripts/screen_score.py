"""Screening of Common Voice zh-TW: three independent checks on every Common Voice zh-TW clip, for a cleaned list of
Taiwan-accented speech.

  speech     Silero VAD: seconds of voiced audio
  reading    CTC forced alignment of the prompt's Mandarin pinyin letters with torchaudio's MMS aligner
             (twjudge.align.ctc): mean probability of the aligned letters (ctc_tok), and the per-frame log-probability
             the forced path loses against the unconstrained best path (ctc_gap); a clip that does not read the prompt
             in Mandarin (another language, another sentence, no speech) aligns badly
  accent     Accent Detective, every size, audio only, calibrated log-odds of Taiwan (the released checkpoint's own Platt)

One JSONL per parquet file, one row per clip: clip, locale, split, speaker (the corpus hash), sentence, accents,
variant, dur, voiced, ctc_tok, ctc_gap, n_tok, z (nano, tiny, mini, small). Flags and thresholds are applied later,
locally.

    python research/scripts/screen_score.py --repo OpenFormosa/common_voice_25_zh-TW --files data/other-00000-of-00003.parquet \\
        --detective /content/det --out /content/r17
"""
import argparse
import hashlib
import io
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))
SR = 16000
_VAD = None


def decode(b: bytes):
    """Audio bytes -> (16 kHz mono float32, voiced seconds), or (None, 0.0) when unreadable."""
    global _VAD
    import librosa
    import soundfile as sf
    import torch

    torch.set_num_threads(1)
    try:
        w, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        w = w.mean(1)
    except Exception:
        try:
            w, sr = librosa.load(io.BytesIO(b), sr=None, mono=True)
        except Exception:
            return None, 0.0
    if sr != SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    w = np.clip(w, -1, 1).astype(np.float32)
    if _VAD is None:
        from silero_vad import load_silero_vad
        _VAD = load_silero_vad()
    from silero_vad import get_speech_timestamps
    ts = get_speech_timestamps(torch.from_numpy(w), _VAD, sampling_rate=SR)
    return w, float(sum(t["end"] - t["start"] for t in ts)) / SR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--files", nargs="+", required=True, help="parquet files of the repo")
    ap.add_argument("--detective", required=True, help="snapshot of the Accent Detective model repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="clips per file (smoke test)")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    import torchaudio.functional as F

    from twjudge.align.ctc import mms_emissions
    from twjudge.data.common import h
    from twjudge.detective import NestedDetective

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # CPU workers (decoding, VAD) start before CUDA is initialised, and by spawn, so they never inherit a CUDA context
    ex = ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn"))
    dev = "cuda"
    D = Path(a.detective)
    det = NestedDetective(D / "model.safetensors", D / "lexicon/concised.json.xz", D / "lexicon/cedict.txt.gz", device=dev)
    label = "tw" if "zh-TW" in a.repo else "cn"
    for f in a.files:
        dst = out / (Path(f).stem + ".jsonl")
        if dst.exists():
            print("skip", f, flush=True)
            continue
        t0 = time.time()
        local = hf_hub_download(a.repo, f, repo_type="dataset", local_dir="/content/cv")
        pf = pq.ParquetFile(local)
        cols = [c for c in ("client_id", "path", "sentence", "accents", "variant", "audio") if c in pf.schema_arrow.names]
        rows, n = [], 0
        for g in range(pf.metadata.num_row_groups):
            group = pf.read_row_group(g, columns=cols).to_pylist()
            for c0 in range(0, len(group), 512):                  # chunks keep the decoded audio in memory small
                t = group[c0:c0 + 512]
                if a.limit:
                    t = t[: max(0, a.limit - n)]
                if not t:
                    break
                dec = list(ex.map(decode, [r["audio"]["bytes"] for r in t], chunksize=16))
                for k, r in enumerate(t):
                    w, voiced = dec[k]
                    name = r["path"].replace("\\", "/").split("/")[-1].replace(".mp3", "")
                    c = r["client_id"]
                    spk = f"tw{h(c)[:16]}" if label == "tw" else f"cn{hashlib.sha256(c.encode()).hexdigest()[:16]}"
                    z, tok, gap, n_tok = None, None, None, None
                    if w is not None and len(w) >= SR // 2:
                        try:                                        # does the audio read the prompt in Mandarin?
                            em, d = mms_emissions(w)
                            sylls = det.aligner.syllables(r["sentence"], d)
                            if sylls is not None:
                                targets = [d[ch] for sy in sylls for ch in sy[3]]
                                n_tok = len(targets)
                                if em.shape[1] >= n_tok:
                                    ali, sc = F.forced_align(em, torch.tensor([targets], dtype=torch.int32), blank=0)
                                    spans = F.merge_tokens(ali[0], sc[0].exp())
                                    tok = round(float(np.mean([sp.score for sp in spans])), 4)
                                    gap = round(float((sc[0].sum() - em[0].max(-1).values.sum()) / em.shape[1]), 4)
                                else:
                                    tok, gap = 0.0, None             # too short to hold the prompt at all
                        except Exception as e:                      # noqa: BLE001  report, keep going
                            print("alignment failed", name, repr(e)[:120], flush=True)
                        try:
                            if len(w) <= det.MAX_SECONDS * SR:
                                res = det._score(w, None, list(det.SIZES))
                            else:
                                res = det.score_recording(w, None, sizes=list(det.SIZES))
                            z = [round(res[s]["logit"], 4) for s in det.SIZES]
                        except Exception as e:                      # noqa: BLE001  report, keep going
                            print("detective failed", name, repr(e)[:120], flush=True)
                    rows.append({"clip": name, "locale": label, "split": Path(f).stem.split("-")[0], "speaker": spk,
                                 "sentence": r["sentence"], "accents": r.get("accents"), "variant": r.get("variant"),
                                 "dur": None if w is None else round(len(w) / SR, 3), "voiced": round(voiced, 3),
                                 "ctc_tok": tok, "ctc_gap": gap, "n_tok": n_tok, "z": z})
                n += len(t)
                print(f"{f} group {g + 1}/{pf.metadata.num_row_groups}: {n} clips, {time.time() - t0:.0f}s", flush=True)
            if a.limit and n >= a.limit:
                break
        with open(dst, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        Path(local).unlink(missing_ok=True)
        print(f"done {f}: {len(rows)} clips, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
