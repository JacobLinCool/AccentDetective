"""Recording conditions, evaluation only: clips re-recorded through four channels.

  mp3     MP3 at 32 kbit/s, mono (ffmpeg, libmp3lame), decoded back to 16 kHz
  tel     telephone: down to 8 kHz, 8-bit G.711 µ-law, back up to 16 kHz
  noise   a real point-source noise recording (OpenSLR 28, Apache 2.0) added at 10 dB SNR
  reverb  a real room impulse response (OpenSLR 28) convolved, peak level kept

Training only ever saw white noise, a 300–3400 Hz band, µ-law at 16 kHz and a 4 kHz low-pass (train_trunk.py's channel
consistency), so MP3, real noise and real rooms are unseen. Every clip's noise or impulse response and offset follow from a
hash of its id and the condition, so every system hears the same perturbed audio.

    python research/scripts/channels.py --work /content/rirs --out /content/bench/chan
"""
import argparse
import subprocess
import zipfile
import zlib
from pathlib import Path

import numpy as np

URL = "https://openslr.elda.org/resources/28/rirs_noises.zip"
CONDS = ("mp3", "tel", "noise", "reverb")
SR = 16000


def prepare(work: Path, out: Path, n_noise=200, n_rir=200) -> None:
    import librosa
    import soundfile as sf

    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    z = work / "rirs_noises.zip"
    if not z.exists():
        subprocess.run(["wget", "-q", "-O", str(z), URL], check=True)
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        noise = sorted(n for n in names if "/pointsource_noises/" in n and n.endswith(".wav"))
        rir = sorted(n for n in names if "/real_rirs_isotropic_noises/" in n and n.endswith(".wav")
                     and "noise" not in n.rsplit("/", 1)[-1].lower())
        rng = np.random.default_rng(0)
        noise = [noise[i] for i in sorted(rng.choice(len(noise), min(n_noise, len(noise)), replace=False))]
        rir = [rir[i] for i in sorted(rng.choice(len(rir), min(n_rir, len(rir)), replace=False))]

        def read(n):
            x, sr = sf.read(f.open(n), dtype="float32", always_2d=True)
            x = x[:, 0]
            return librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x

        ns = [read(n)[: 10 * SR] for n in noise]
        ns = [x for x in ns if len(x) > SR // 2 and np.abs(x).max() > 0]
        rs = []
        for n in rir:
            h = read(n)
            p = int(np.abs(h).argmax())
            h = h[p: p + SR]                                   # from the direct path, at most 1 s
            rs.append(h / (np.abs(h).max() + 1e-9))
    L = max(len(h) for h in rs)
    np.savez(out / "assets.npz", noise=np.concatenate(ns).astype(np.float32), noise_off=np.r_[0, np.cumsum([len(x) for x in ns])],
             rir=np.stack([np.pad(h, (0, L - len(h))) for h in rs]).astype(np.float32),
             noise_names=np.array(noise[: len(ns)]), rir_names=np.array(rir))
    print(f"{len(ns)} noises, {len(rs)} impulse responses → {out / 'assets.npz'}", flush=True)


def load_assets(d) -> dict:
    a = np.load(Path(d) / "assets.npz")
    return {k: a[k] for k in ("noise", "noise_off", "rir")}


def _mu_law(x, mu=255):
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * mu) / mu * 2 - 1                    # 8 bits
    return np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu


def perturb(w: np.ndarray, cond: str, key: str, A: dict) -> np.ndarray:
    """w float32 16 kHz → the same clip through `cond`."""
    import librosa
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(zlib.crc32(f"{key}|{cond}".encode()))
    w = np.asarray(w, np.float32)
    if cond == "mp3":
        pcm = (np.clip(w, -1, 1) * 32767).astype("<i2").tobytes()
        mp3 = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                              "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3", "pipe:1"], input=pcm, capture_output=True,
                             check=True).stdout
        dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "mp3", "-i", "pipe:0", "-f", "s16le", "-ar", str(SR),
                              "-ac", "1", "pipe:1"], input=mp3, capture_output=True, check=True).stdout
        return np.frombuffer(dec, "<i2").astype(np.float32) / 32768
    if cond == "tel":
        x = librosa.resample(w, orig_sr=SR, target_sr=8000)
        x = _mu_law(np.clip(x / (np.abs(x).max() + 1e-9), -1, 1)) * np.abs(x).max()
        return librosa.resample(x.astype(np.float32), orig_sr=8000, target_sr=SR)[: len(w)]
    if cond == "noise":
        k = int(rng.integers(len(A["noise_off"]) - 1))
        a, b = int(A["noise_off"][k]), int(A["noise_off"][k + 1])
        n = A["noise"][a:b]
        if len(n) < len(w):
            n = np.tile(n, len(w) // len(n) + 1)
        o = int(rng.integers(0, len(n) - len(w) + 1))
        n = n[o: o + len(w)]
        ps, pn = float(np.mean(w ** 2)) + 1e-12, float(np.mean(n ** 2)) + 1e-12
        return (w + n * np.sqrt(ps / pn / 10 ** (10 / 10))).astype(np.float32)
    if cond == "reverb":
        h = A["rir"][int(rng.integers(len(A["rir"])))]
        h = h[: int(np.nonzero(np.abs(h) > 1e-4)[0][-1]) + 1]
        y = fftconvolve(w, h)[: len(w)]
        return (y * (np.abs(w).max() / (np.abs(y).max() + 1e-9))).astype(np.float32)
    raise ValueError(cond)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prepare(Path(a.work), Path(a.out))


if __name__ == "__main__":
    main()
