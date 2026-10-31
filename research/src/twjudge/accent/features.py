"""Per-syllable and per-utterance prosodic-phonetic features for the interpretable accent judge.

Everything is normalised within the utterance (pitch in semitones from the utterance median,
duration and intensity relative to the utterance mean), so speaker and recording level cancel
out to first order. Per syllable:

  text context   canonical (Mainland, full) tone, tone after 3-3 sandhi, Mainland-neutral flag,
                 phrase-final flag, relative position; never the character identity
  pitch          N_POINTS-point rhyme contour, mean, slope, range
  timing         log duration, duration relative to the utterance mean, rhyme share
  loudness       mean intensity relative to the utterance mean
  neighbours     duration / pitch / intensity differences to the previous and next syllable
  realisation    which dictionary variant the aligner picked, wherever there was a choice:
                 Taiwan reading, Mainland neutral tone, de-retroflexion, -ing/-eng → -in/-en

Per utterance: speech rate, duration variability (varco, nPVI), pitch range and declination,
loudness variability, phrase-final lengthening and pause share.

    python -m twjudge.accent.features --align data/judge_corpus/align_dev --out data/judge_corpus/feat_dev.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from twjudge.reading.tone_judge import N_POINTS, contour
from twjudge.phonology import sandhi_tones, split_syllable

PAUSE_S = 0.15
SYL_NAMES = (["tone", "sandhi", "cn_neutral", "final", "relpos"]
             + [f"f0_{k}" for k in range(N_POINTS)] + ["f0_mean", "f0_slope", "f0_range", "voiced"]
             + ["logdur", "reldur", "rhyme_share", "relint"]
             + ["d_dur_prev", "d_dur_next", "d_f0_prev", "d_f0_next", "d_int_prev", "d_int_next"]
             + ["opp_tw", "chose_tw", "opp_neutral", "chose_neutral", "opp_retro", "chose_retro",
                "opp_velar", "chose_velar"])
TEXT_COLS = ["tone", "sandhi", "cn_neutral", "final", "opp_tw", "opp_neutral", "opp_retro", "opp_velar"]
UTT_NAMES = ["rate", "varco_dur", "npvi_dur", "f0_p5_95", "f0_sd", "decl", "int_sd", "npvi_int",
             "final_lengthening", "pause_share", "n_syl"]


def tracks(wav: Path):
    import parselmouth

    snd = parselmouth.Sound(str(wav))
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
    t = pitch.xs()
    f = pitch.selected_array["frequency"]
    voiced = f > 0
    if voiced.sum() < 10:
        return None
    st = np.full_like(f, np.nan)
    st[voiced] = 12 * np.log2(f[voiced] / np.median(f[voiced]))
    inten = snd.to_intensity(minimum_pitch=75, time_step=0.01)
    return t, st, inten.xs(), inten.values[0]


def _npvi(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return 100 * np.mean(np.abs(np.diff(x)) / ((x[1:] + x[:-1]) / 2 + 1e-9))


def utterance(wav: Path, chars: list[dict]):
    """(syllable feature matrix, utterance feature vector) or None."""
    tr = tracks(wav)
    if tr is None or len(chars) < 3:
        return None
    t, st, ti, iv = tr
    n = len(chars)
    tones = [split_syllable(c["syl"])[2] for c in chars]
    sand = sandhi_tones(tones)
    dur = np.array([c["end"] - c["start"] for c in chars])
    rhyme = np.array([c["end"] - c["rhyme_start"] for c in chars])
    final = np.array([k + 1 == n or chars[k + 1]["start"] - c["end"] >= PAUSE_S for k, c in enumerate(chars)])
    imean = np.array([np.mean(iv[(ti >= c["start"]) & (ti <= c["end"])]) if ((ti >= c["start"]) & (ti <= c["end"])).any()
                      else np.nan for c in chars])
    imean = imean - np.nanmean(imean)
    rows, f0m = [], []
    for k, c in enumerate(chars):
        cc = contour(t, st, c["rhyme_start"], c["end"])
        m = (t >= c["rhyme_start"]) & (t <= c["end"])
        vfrac = float(np.mean(~np.isnan(st[m]))) if m.any() else 0.0
        if cc is None:
            cc = np.full(N_POINTS, np.nan)
        f0m.append(np.nanmean(cc) if not np.all(np.isnan(cc)) else np.nan)
        v = c["variant"]
        rows.append([int(tones[k]), int(sand[k]), int(c["cn_neutral"]), int(final[k]), k / (n - 1)]
                    + list(cc) + [f0m[-1], cc[-1] - cc[0], np.nanmax(cc) - np.nanmin(cc) if not np.all(np.isnan(cc)) else np.nan, vfrac]
                    + [np.log(dur[k]), dur[k] / dur.mean(), rhyme[k] / max(dur[k], 1e-3), imean[k]]
                    + [0.0] * 6
                    + [int(c["tw"] != c["syl"]), int(v.startswith("tw")),
                       int(c["cn_neutral"]), int(v.startswith("cn5")),
                       int(split_syllable(c["syl"])[0] in ("zh", "ch", "sh")), int("retroflex" in v),
                       int(split_syllable(c["syl"])[1].endswith(("ing", "eng"))), int("velar" in v)])
    X = np.array(rows, float)
    f0m = np.array(f0m)
    i_rd, i_f0, i_int = SYL_NAMES.index("reldur"), SYL_NAMES.index("f0_mean"), SYL_NAMES.index("relint")
    j = SYL_NAMES.index("d_dur_prev")
    for k in range(n):
        p, q = max(k - 1, 0), min(k + 1, n - 1)
        X[k, j:j + 6] = [np.log(X[k, i_rd] / X[p, i_rd]), np.log(X[k, i_rd] / X[q, i_rd]),
                         X[k, i_f0] - X[p, i_f0], X[k, i_f0] - X[q, i_f0],
                         X[k, i_int] - X[p, i_int], X[k, i_int] - X[q, i_int]]
    speech = chars[-1]["end"] - chars[0]["start"]
    gaps = sum(max(0.0, chars[k + 1]["start"] - chars[k]["end"]) for k in range(n - 1))
    v = st[~np.isnan(st)]
    ok = ~np.isnan(f0m)
    decl = np.polyfit(np.arange(n)[ok], f0m[ok], 1)[0] if ok.sum() >= 3 else np.nan
    fl = dur[final].mean() / dur[~final].mean() if (~final).any() else np.nan
    U = np.array([n / speech, np.std(dur) / dur.mean(), _npvi(dur), np.percentile(v, 95) - np.percentile(v, 5),
                  np.std(v), decl, np.nanstd(imean), _npvi(np.nan_to_num(imean - np.nanmin(imean) + 1)),
                  fl, gaps / speech, n], float)
    return X, U


def text_features(chars: list[dict]) -> np.ndarray:
    """The syllable rows of `utterance` with only the text-derived columns filled (no audio analysis):
    the judges' text context and the text-context cells of the importance weights."""
    n = len(chars)
    X = np.zeros((n, len(SYL_NAMES)))
    tones = [split_syllable(c["syl"])[2] for c in chars]
    col = {k: SYL_NAMES.index(k) for k in SYL_NAMES}
    for k, (c, t, s) in enumerate(zip(chars, tones, sandhi_tones(tones))):
        ini, fin, _ = split_syllable(c["syl"])
        final = k + 1 == n or chars[k + 1]["start"] - c["end"] >= PAUSE_S
        for name, v in (("tone", int(t)), ("sandhi", int(s)), ("cn_neutral", int(c["cn_neutral"])), ("final", int(final)),
                        ("relpos", k / max(n - 1, 1)), ("opp_tw", int(c["tw"] != c["syl"])),
                        ("opp_neutral", int(c["cn_neutral"])), ("opp_retro", int(ini in ("zh", "ch", "sh"))),
                        ("opp_velar", int(fin.endswith(("ing", "eng"))))):
            X[k, col[name]] = v
    return X


def extract(align_dir: Path, items: dict | None = None):
    wavs = {p.stem: p for c in align_dir.glob("corpus*") for p in c.glob("*/*.wav")}
    Xs, Us, ids = [], [], []
    for line in (align_dir / "profile.jsonl").open():
        r = json.loads(line)
        if r["status"] != "ok" or r["item_id"] not in wavs:
            continue
        out = utterance(wavs[r["item_id"]], r["chars"])
        if out is None:
            continue
        Xs.append(out[0]); Us.append(out[1]); ids.append(r["item_id"])
    return Xs, Us, ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--align", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    align = Path(a.align)
    lines = [l for l in (align / "profile.jsonl").open()]
    wavs = {p.stem: str(p) for c in align.glob("corpus*") for p in c.glob("*/*.wav")}
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for l in lines:
        r = json.loads(l)
        if r["status"] == "ok" and r["item_id"] in wavs:
            jobs.append((r["item_id"], wavs[r["item_id"]], r["chars"]))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_one, jobs, chunksize=16))
    ids, Xs, Us = [], [], []
    for (iid, _, _), out in zip(jobs, res):
        if out is not None:
            ids.append(iid); Xs.append(out[0]); Us.append(out[1])
    lens = np.array([len(x) for x in Xs])
    np.savez_compressed(a.out, ids=np.array(ids), X=np.concatenate(Xs), lens=lens, U=np.array(Us),
                        syl_names=np.array(SYL_NAMES), utt_names=np.array(UTT_NAMES))
    print(len(ids), "utterances,", lens.sum(), "syllables")


def _one(job):
    iid, wav, chars = job
    try:
        return utterance(Path(wav), chars)
    except Exception:
        return None


if __name__ == "__main__":
    main()
