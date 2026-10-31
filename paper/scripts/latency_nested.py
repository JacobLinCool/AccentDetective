"""Latency and throughput of Accent Detective's sizes and of the reference judges, on one machine (a Colab GPU runtime:
its GPU and its host CPU), from waveform to calibrated score.

Clips: 5 s Common Voice test utterances (4.5–5.5 s long, cropped or zero-padded to exactly 5 s for the nested model;
the reference judges get the uncropped clip and its text, which their alignment needs).

  --part nested     every size, audio only and with the text:
                      GPU: batch 1, 8, 32, 128, 512, fp32 and bf16 autocast (time per batch, clips per second)
                      CPU: batch 1 and 8 at 1, 4 and all threads, fp32
                    anytime: one pass to the last exit, each size's score taken as its exit is reached (time to each
                    size), against running every size on its own
  --part reference  batch 1 on --device: the HuBERT syllable judge pipeline (MMS alignment, HuBERT, judge) and the
                    aligned compact judge (the earlier released checkpoint)

    python paper/scripts/latency_nested.py --part nested --bundle /content/bench --model model.safetensors \\
        --lexicon data/lexicon/concised.json.xz --cedict data/lexicon/cedict.txt.gz --out latency_nested.json
"""
import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))
SR, CLIP = 16000, 5 * 16000


def machine() -> dict:
    import torch

    info = {"platform": platform.platform(), "torch": torch.__version__, "cpu_count": os.cpu_count()}
    try:
        info["cpu"] = re.search(r"model name\s*:\s*(.*)", open("/proc/cpuinfo").read()).group(1)
    except OSError:
        info["cpu"] = platform.processor()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def clips(bundle: Path, n: int):
    import librosa

    rows = [json.loads(l) for l in open(bundle / "sets" / "cv_test.jsonl")]
    out = []
    for r in rows:
        w = librosa.load(str(bundle / r["audio"]), sr=SR)[0]
        if 4.5 * SR <= len(w) <= 5.5 * SR:
            out.append((w, r["text"]))
        if len(out) == n:
            break
    return out


def timed(fn, reps, sync):
    for _ in range(2):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def nested(a, res):
    import torch

    from twjudge.accent.nested import voiced_span
    from twjudge.detective import NestedDetective

    data = clips(Path(a.bundle), 32)
    fixed = [np.pad(w[:CLIP], (0, max(0, CLIP - len(w)))) for w, _ in data]
    res["clips"] = len(data)

    def run(det, k, B, text, dev, bf16=False, anytime=None):
        """One batch of B clips through size k (0-based) in one mode → scores; anytime: list collecting per-exit times."""
        idx = [i % len(data) for i in range(B)]
        x = torch.tensor(np.stack([fixed[i] for i in idx]), device=dev)
        syl = [det._syllables(data[i][1]) for i in idx] if text else None
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cpu", enabled=False)
        t0 = time.perf_counter()
        with torch.inference_mode(), ctx:
            for e, o in enumerate(det.net.iter_exits(x, upto=k + 1)):
                if e < k and anytime is None:
                    continue
                X = torch.stack([o[t] for t in det.net.targets], 2)                 # (B, T, layers, 768)
                X = X - X.mean(1, keepdim=True)
                m = torch.zeros(X.shape[:2], dtype=torch.bool, device=dev)
                h = det.asp[e].frames(X)
                z = det.asp[e](X, m, h=h)
                if text:
                    S = max(len(s[0]) for s in syl)
                    ty = torch.zeros((B, S), dtype=torch.long, device=dev)
                    cx = torch.zeros((B, S, 7), device=dev)
                    sm = torch.ones((B, S), dtype=torch.bool, device=dev)
                    for j, (t_, c_) in enumerate(syl):
                        ty[j, :len(t_)] = torch.tensor(t_); cx[j, :len(t_)] = torch.tensor(c_); sm[j, :len(t_)] = False
                    span = torch.tensor([voiced_span(fixed[i], X.shape[1]) for i in idx], device=dev)
                    z = z + det.text[e](h, m, span, ty, cx, sm)
                a_, b_ = det.meta["platt"][det.keys[e]]["text" if text else "audio"]
                p = torch.sigmoid(a_ * z.float() + b_).cpu()                        # results leave the device
                if anytime is not None:
                    anytime.append(time.perf_counter() - t0)
        return p

    for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        det = NestedDetective(a.model, lexicon=a.lexicon, cedict=a.cedict, device=dev)
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
        rows = []
        if dev == "cuda":
            settings = [(B, prec, None) for prec in ("fp32", "bf16") for B in (1, 8, 32, 128, 512)]
        else:
            settings = [(B, "fp32", th) for th in sorted({1, 4, os.cpu_count()}) for B in (1, 8)]
        for B, prec, th in settings:
            if th:
                torch.set_num_threads(th)
            for k, size in enumerate(det.SIZES):
                for text in (False, True):
                    try:
                        s = timed(lambda: run(det, k, B, text, dev, prec == "bf16"), a.reps, sync)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        s = None
                    r = {"device": dev, "batch": B, "precision": prec, "threads": th, "size": size,
                         "mode": "text" if text else "audio", "params": det.parameters(size, text)}
                    if s is not None:
                        r.update({"batch_ms": 1000 * s, "clips_per_s": B / s, "rtf": s / (B * 5.0)})
                    rows.append(r)
                    print(json.dumps(r), flush=True)
        # anytime: time to each size's score within one pass of the largest, against each size on its own
        any_rows = []
        for B, th in ([(1, None), (128, None)] if dev == "cuda" else [(1, 4), (1, os.cpu_count())]):
            if th:
                torch.set_num_threads(th)
            for text in (False, True):
                times = []
                for _ in range(2):
                    run(det, len(det.SIZES) - 1, B, text, dev, anytime=[])
                for _ in range(a.reps):
                    t = []
                    run(det, len(det.SIZES) - 1, B, text, dev, anytime=t)
                    times.append(t)
                r = {"device": dev, "batch": B, "threads": th, "mode": "text" if text else "audio",
                     "time_to_size_ms": dict(zip(det.SIZES, (1000 * np.median(np.array(times), 0)).tolist()))}
                any_rows.append(r)
                print(json.dumps(r), flush=True)
        res.setdefault("nested", []).extend(rows)
        res.setdefault("anytime", []).extend(any_rows)
        del det
        if dev == "cuda":
            torch.cuda.empty_cache()


def reference(a, res):
    import torch

    from twjudge.accent import ssl_features as S
    from twjudge.accent.compact import CompactJudge
    from twjudge.accent.neural import JudgeB
    from twjudge.align import ctc
    from twjudge.lexicon.readings import TaiwanReadings

    if a.cedict:
        os.environ["ACCENT_DETECTIVE_CEDICT"] = a.cedict
    dev = a.device
    if dev == "cpu":
        S.device = lambda: "cpu"
        ctc._bundle.cache_clear()
        import torchaudio

        b = torchaudio.pipelines.MMS_FA
        model = b.get_model(with_star=False).eval()
        ctc._bundle = lambda: (model, b.get_dict(star=None), "cpu")
    if a.threads:
        torch.set_num_threads(a.threads)
    data = clips(Path(a.bundle), 32)
    rw = TaiwanReadings(a.lexicon)
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    count = lambda *ms: sum(p.numel() for m in ms for p in m.parameters())
    out = []

    def bench(name, fn, params):
        ts = []
        for w, t in data[:3]:
            fn(w, t)
        sync()
        for w, t in data[: a.reps * 2]:
            t0 = time.perf_counter(); fn(w, t); sync(); ts.append(time.perf_counter() - t0)
        r = {"device": dev, "threads": a.threads, "system": name, "params": params, "batch": 1,
             "median_ms": 1000 * float(np.median(ts))}
        out.append(r)
        print(json.dumps(r), flush=True)

    mms, _, _ = ctc._bundle()
    aligner = ctc.CTCAligner(rw)
    judge = JudgeB(a.teacher)
    judge.hubert = S.load_model(judge.ssl_kind)

    def pipeline(w, t):
        chars = aligner.align(w, t)
        if chars is not None:
            with torch.inference_mode():
                judge.from_audio(torch.tensor(w), chars)

    bench("judge_pipeline", pipeline, count(mms, judge.hubert, judge.net))
    del judge
    cj = CompactJudge(a.compact, readings=rw, device=dev, mode="accurate")
    bench("aligned_compact", lambda w, t: cj.score(w, t), count(cj.enc, cj.head, *cj.nets))
    res.setdefault("reference", []).extend(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["nested", "reference"], required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", help="nested checkpoint")
    ap.add_argument("--teacher", help="reference: a HuBERT syllable judge (runs_ctc/b_adv0_s0.pt)")
    ap.add_argument("--compact", help="reference: the aligned compact checkpoint")
    ap.add_argument("--lexicon", default=str(ROOT / "data/lexicon/concised.json.xz"))
    ap.add_argument("--cedict", default=str(ROOT / "data/lexicon/cedict.txt.gz"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"machine": machine(), "part": a.part}
    (nested if a.part == "nested" else reference)(a, res)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
