"""Unlabelled distillation audio: public Common Voice Mandarin, without any judge evaluation speaker.

  Taiwan    Common Voice 25 zh-TW (validated + other), every speaker of the judge's dev and test splits
            excluded
  Mainland  Common Voice 21 zh-CN train split (the judge's Mainland dev / test speakers come from the
            dev / test splits, so they cannot occur)

  extra     (--extra name=hours) other public Mandarin read speech for more recording conditions:
            aishell1 (OpenSLR 33), thchs30 (OpenSLR 18), aishell3 (AISHELL/AISHELL-3 train), all Apache 2.0

--exclude-speakers drops further Common Voice speakers (e.g. a held-out evaluation set built later).
Clips are decoded to 16 kHz, cut to at most MAX_S seconds, and packed as int16 into one array with
offsets (`audio.npy`, `offsets.npy`, `meta.jsonl`). The judge corpus clips are kept out by speaker,
and every clip that appears in a benchmark set is excluded by file name as well.

    python research/scripts/distill_data.py --bundle /content/bench --out /content/distill --hours 100
    python research/scripts/distill_data.py --bundle /content/bench --out /content/distill2 --hours 90 \
        --exclude-speakers /content/bench/heldout_speakers.txt --extra aishell1=25 thchs30=15 aishell3=15
"""
import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))

TW, CN = "OpenFormosa/common_voice_25_zh-TW", "keeve101/common-voice-21.0-2025-03-14-zh-CN-split"
MAX_S, MIN_S = 10.0, 1.5
EXTRA = {"aishell1": "https://openslr.elda.org/resources/33/data_aishell.tgz",
         "thchs30": "https://openslr.elda.org/resources/18/data_thchs30.tgz",
         "aishell3": "AISHELL/AISHELL-3"}


def extra_files(name: str, work: Path) -> list[Path]:
    """Download one extra corpus into work/ and return its wav files in a fixed shuffled order."""
    import subprocess
    import tarfile

    d = work / name
    if not (d / ".ok").exists():
        d.mkdir(parents=True, exist_ok=True)
        if name == "aishell3":
            from huggingface_hub import snapshot_download

            snapshot_download(EXTRA[name], repo_type="dataset", local_dir=d,
                              allow_patterns=[f"train/wav/SSB0{k}*" for k in range(0, 5)])
        else:
            # whole file first (resumed and retried until it has the server's size), then unpack: a
            # streamed `curl | tar` can end early on a dropped connection without failing
            url, arc = EXTRA[name], work / f"{name}.tgz"
            size = int(subprocess.run(f"curl -sIL {url} | grep -i '^content-length' | tail -1 | tr -dc 0-9",
                                      shell=True, capture_output=True, text=True).stdout or 0)
            for _ in range(20):
                subprocess.run(["curl", "-sL", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(arc), url])
                if size and arc.exists() and arc.stat().st_size >= size:
                    break
            if not (size and arc.stat().st_size == size):
                raise RuntimeError(f"{name}: got {arc.stat().st_size if arc.exists() else 0} of {size} bytes")
            subprocess.run(["tar", "-xzf", str(arc), "-C", str(d)], check=True)
            arc.unlink()
            for t in sorted(d.rglob("*.tar.gz")):            # AISHELL-1 packs each speaker separately
                try:
                    tarfile.open(t).extractall(t.parent)
                except (EOFError, tarfile.TarError) as e:
                    print(f"{name}: skipped {t.name}: {e}", flush=True)
                t.unlink()
        (d / ".ok").touch()
    files = sorted(f for f in d.rglob("*.wav") if "/test/" not in str(f) and "/dev/" not in str(f))
    if name == "thchs30":                                   # data/ holds every clip; train/dev/test link into it
        files = [f for f in files if f.parent.name == "data"]
    np.random.default_rng(0).shuffle(files)
    return files


def decode_file(p: str):
    with open(p, "rb") as f:
        return decode(f.read())


def decode(b: bytes):
    import librosa
    import soundfile as sf

    try:
        w, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        w = w.mean(1)
        if sr != 16000:
            w = librosa.resample(w, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
    except Exception:
        return None
    w = w[: int(MAX_S * 16000)]
    if len(w) < MIN_S * 16000:
        return None
    return (np.clip(w, -1, 1) * 32767).astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--exclude-speakers", help="file of further speaker ids (tw<16 hex> / cn<16 hex>) to leave out")
    ap.add_argument("--extra", nargs="*", default=[], help="name=hours of extra corpora: aishell1, thchs30, aishell3")
    ap.add_argument("--work", default="/content/distill_src")
    ap.add_argument("--prefetch", action="store_true", help="only download and unpack the --extra corpora")
    a = ap.parse_args()
    if a.prefetch:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(len(a.extra)) as ex:
            for name, files in zip(a.extra, ex.map(lambda s: extra_files(s.split("=")[0], Path(a.work)), a.extra)):
                print(name, len(files), "files", flush=True)
        return
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    from twjudge.data.common import h

    bundle, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held, clips = set(), set()
    for s in ("cv_dev", "cv_test"):
        for l in open(bundle / "sets" / f"{s}.jsonl"):
            held.add(json.loads(l)["speaker"])
    if a.exclude_speakers:
        held |= {l.strip() for l in open(a.exclude_speakers) if l.strip()}
    for p in (bundle / "sets").glob("*.jsonl"):
        for l in open(p):
            clips.add(json.loads(l)["item_id"].split("_", 1)[-1].replace("\\", "/").split("/")[-1])
    budget = {"tw": a.hours * 0.6 * 3600, "cn": a.hours * 0.4 * 3600}
    used = {"tw": 0.0, "cn": 0.0}
    pieces, offsets, meta = [], [0], []
    for label, repo, pattern in (("tw", TW, ["data/validated-*.parquet", "data/other-*.parquet"]),
                                 ("cn", CN, ["data/train-*.parquet"])):
        src = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=pattern))
        seen = set()
        # only the requested splits: the snapshot folder may also hold other splits fetched by other scripts
        for f in sorted(q for pat in pattern for q in src.glob(pat)):
            t = pq.read_table(f, columns=["client_id", "path", "audio"])
            cid, paths = t.column("client_id").to_pylist(), t.column("path").to_pylist()
            keep = []
            for k, (c, p) in enumerate(zip(cid, paths)):
                name = p.replace("\\", "/").split("/")[-1]
                spk = (f"tw{h(c)[:16]}" if label == "tw" else f"cn{hashlib.sha256(c.encode()).hexdigest()[:16]}")
                if spk in held or name in seen or name.replace(".mp3", "") in clips or f"common_voice_{name}" in clips:
                    continue
                seen.add(name); keep.append(k)
            audio = t.column("audio")
            with ProcessPoolExecutor(a.workers) as ex:
                for k, w in zip(keep, ex.map(decode, [audio[k].as_py()["bytes"] for k in keep], chunksize=32)):
                    if w is None or used[label] >= budget[label]:
                        continue
                    pieces.append(w); offsets.append(offsets[-1] + len(w)); used[label] += len(w) / 16000
                    meta.append({"label": label, "path": paths[k]})
            print(label, f.name, f"{used[label] / 3600:.1f} h", flush=True)
            if used[label] >= budget[label]:
                break
    for spec in a.extra:
        name, hours = spec.split("=")
        files, got = extra_files(name, Path(a.work)), 0.0
        with ProcessPoolExecutor(a.workers) as ex:
            for k in range(0, len(files), 2000):
                for f, w in zip(files[k:k + 2000], ex.map(decode_file, map(str, files[k:k + 2000]), chunksize=32)):
                    if w is None or got >= float(hours) * 3600:
                        continue
                    pieces.append(w); offsets.append(offsets[-1] + len(w)); got += len(w) / 16000
                    meta.append({"label": name, "path": str(f.relative_to(Path(a.work)))})
                if got >= float(hours) * 3600:
                    break
        used[name] = got
        print(name, f"{got / 3600:.1f} h", flush=True)
    np.save(out / "audio.npy", np.concatenate(pieces))
    np.save(out / "offsets.npy", np.array(offsets, dtype=np.int64))
    with open(out / "meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print({k: round(v / 3600, 1) for k, v in used.items()}, "hours,", len(meta), "clips", flush=True)


if __name__ == "__main__":
    main()
