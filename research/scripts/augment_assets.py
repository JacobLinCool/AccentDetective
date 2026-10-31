"""Pack the augmentation assets of `twjudge.accent.augment`: MUSAN noise + music and OpenSLR 28 RIRs.

Downloads MUSAN (OpenSLR 17, CC BY 4.0) and RIRS_NOISES (OpenSLR 28, Apache 2.0), keeps up to
--noise-hours of MUSAN noise and music (no speech, so no foreign accent leaks in) and --n-rir
simulated and real room impulse responses, all at 16 kHz.

    python research/scripts/augment_assets.py --work /content/aug_src --out /content/bench/aug
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

import numpy as np

MUSAN = "https://openslr.elda.org/resources/17/musan.tar.gz"
RIRS = "https://openslr.elda.org/resources/28/rirs_noises.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-hours", type=float, default=6)
    ap.add_argument("--n-rir", type=int, default=3000)
    ap.add_argument("--rir-s", type=float, default=1.0)
    a = ap.parse_args()
    import soundfile as sf

    work, out = Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    if not (work / "musan.ok").exists():
        subprocess.run(f"curl -sL {MUSAN} | tar -xz -C {work} musan/noise musan/music", shell=True, check=True)
        (work / "musan.ok").touch()
    if not (work / "RIRS_NOISES").exists():
        subprocess.run(["curl", "-sL", "-o", str(work / "rirs.zip"), RIRS], check=True)
        zipfile.ZipFile(work / "rirs.zip").extractall(work)
    rng = np.random.default_rng(0)
    files = sorted((work / "musan").rglob("*.wav"))
    rng.shuffle(files)
    pieces, off, total = [], [0], 0
    for f in files:
        w, sr = sf.read(f, dtype="float32", always_2d=True)
        w = w.mean(1)
        if sr != 16000:
            continue
        w = w[: 16000 * 60]                                  # at most a minute per file: more variety
        pieces.append((np.clip(w, -1, 1) * 32767).astype(np.int16))
        total += len(w)
        off.append(total)
        if total > a.noise_hours * 3600 * 16000:
            break
    np.save(out / "noise.npy", np.concatenate(pieces))
    np.save(out / "noise_off.npy", np.array(off))
    rirs = sorted((work / "RIRS_NOISES").rglob("*.wav"))
    rirs = [r for r in rirs if "rir" in str(r).lower() and "noise" not in r.name.lower()]
    rng.shuffle(rirs)
    L = int(a.rir_s * 16000)
    R = []
    for f in rirs:
        h, sr = sf.read(f, dtype="float32", always_2d=True)
        if sr != 16000:
            continue
        h = h[:, 0]
        h = h[int(np.argmax(np.abs(h))):][:L]                # direct path first
        h = np.pad(h, (0, L - len(h))) / (np.abs(h).max() + 1e-6)
        R.append(h)
        if len(R) >= a.n_rir:
            break
    np.save(out / "rir.npy", np.stack(R).astype(np.float32))
    print(f"noise {total / 16000 / 3600:.1f} h from {len(pieces)} files, {len(R)} RIRs", flush=True)


if __name__ == "__main__":
    main()
