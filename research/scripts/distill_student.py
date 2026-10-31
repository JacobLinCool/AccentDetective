"""Distil the judges' Chinese SSL encoder into a small student (twjudge.accent.student).

Each step draws a batch of clips from the packed distillation audio (distill_data.py), crops them
all to one length (the shortest clip in the batch, at most CROP_S seconds), so neither model ever
sees padding, and trains the student to predict the frozen teacher's hidden states of the target
layers: L1 + (−log σ(cosine)) per frame, summed over the targets (DistilHuBERT). Linear warm-up,
then linear decay to zero at --steps; checkpoints every --save steps. Reports the per-layer cosine
on a held-out slice of the distillation audio.

With --whisper-targets the student also predicts layers of the Whisper-medium encoder (a second teacher,
1024-d, run on the same crop padded to its 30 s window), from the same trunk; --init warm-starts from an
existing student (its Whisper heads start untrained).

With --augment the student hears the clip through a random recording channel (noise, reverberation,
telephone band, mu-law; `twjudge.accent.augment`) while the teacher hears it clean, so the student
learns to predict the clean-channel states from degraded audio.

    python research/scripts/distill_student.py --data /content/distill --teacher wav2vec2 --d 384 --layers 4 \
        --steps 40000 --out /content/students/w2v_d384_L4.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))

CROP_S = 6.0
DEV = "cuda"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", choices=["hubert", "wav2vec2"], default="wav2vec2")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--arch", choices=["transformer", "conformer"], default="transformer")
    ap.add_argument("--ff", type=int, default=0, help="feed-forward width (default 4 d; Conformer: 2 d, macaron)")
    ap.add_argument("--targets", default="1,2,3,4,6,8,12")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save", type=int, default=5000)
    ap.add_argument("--minutes", type=float, default=0, help="stop early after this wall time (0 = no limit)")
    ap.add_argument("--norm-loss", action="store_true",
                    help="remove each crop's mean from teacher and student states before the loss (what the judge reads)")
    ap.add_argument("--augment", help="augmentation assets directory (augment_assets.py)")
    ap.add_argument("--whisper-targets", default="", help="Whisper-medium encoder layers to predict too, e.g. 6,12,18")
    ap.add_argument("--whisper-weight", type=float, default=1.0)
    ap.add_argument("--init", help="student checkpoint to start from (architecture taken from it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F

    from twjudge.accent.ssl_features import load_model
    from twjudge.accent.student import Student

    global DEV
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    audio = np.load(Path(a.data) / "audio.npy", mmap_mode="r")
    off = np.load(Path(a.data) / "offsets.npy")
    n = len(off) - 1
    held = rng.choice(n, size=max(1, min(300, n // 50)), replace=False)
    train = np.setdiff1d(np.arange(n), held)
    lens = np.diff(off)
    targets = [int(t) for t in a.targets.split(",")]
    teacher = load_model(a.teacher).to(DEV).to(torch.bfloat16)
    ff = a.ff or (2 * a.d if a.arch == "conformer" else 4 * a.d)
    wt = [int(t) for t in a.whisper_targets.split(",") if t]
    if a.init:
        ck0 = torch.load(a.init, map_location="cpu")
        student = Student(**{**ck0["config"], "whisper_targets": wt}).to(DEV)
        student.load_state_dict(ck0["state"], strict=False)
        targets = list(student.targets)
    else:
        student = Student(targets=targets, d=a.d, n_layers=a.layers, heads=a.d // 64, ff=ff, arch=a.arch,
                          whisper_targets=wt).to(DEV)
    wteacher = wfe = None
    if wt:
        from transformers import WhisperFeatureExtractor, WhisperModel

        wfe = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
        wteacher = WhisperModel.from_pretrained("openai/whisper-medium", torch_dtype=torch.bfloat16).encoder
        wteacher.layers = wteacher.layers[:max(wt)]                      # deeper layers are never read
        wteacher = wteacher.to(DEV).eval()
        for q in wteacher.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = max(1, min(2000, a.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / a.steps))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if out.with_suffix(".last.pt").exists():                          # resume
        ck = torch.load(out.with_suffix(".last.pt"), map_location="cpu")
        student.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["step"]
        rng = np.random.default_rng(start)
    print(f"{n} clips, {lens.sum() / 16000 / 3600:.1f} h; student {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M",
          flush=True)

    def batch(idx):
        L = int(min(CROP_S * 16000, lens[idx].min()))
        xs = []
        for i in idx:
            o = off[i] + rng.integers(0, lens[i] - L + 1)
            xs.append(np.asarray(audio[o:o + L], dtype=np.float32) / 32768)
        return torch.tensor(np.stack(xs), device=DEV)

    def teacher_states(x):
        with torch.no_grad():
            xs = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            hs = teacher(xs.to(torch.bfloat16), output_hidden_states=True).hidden_states
        return {t: hs[t].float() for t in targets}

    def whisper_states(x):
        """Whisper encoder states of the crops (each padded to the 30 s window), {layer: (B, frames, 1024)}."""
        with torch.no_grad():
            f = wfe(list(x.float().cpu().numpy()), sampling_rate=16000, return_tensors="pt", device=DEV).input_features
            hs = wteacher(f.to(DEV, torch.bfloat16), output_hidden_states=True).hidden_states
        n = int(np.ceil(x.shape[1] / 320))
        return {t: hs[t][:, :n].float() for t in wt}

    def loss_fn(pred, tgt, targets=targets):
        tot, cos = 0.0, {}
        for t in targets:
            T = min(pred[t].shape[1], tgt[t].shape[1])
            p, q = pred[t][:, :T], tgt[t][:, :T]
            if a.norm_loss:
                p, q = p - p.mean(1, keepdim=True), q - q.mean(1, keepdim=True)
            c = F.cosine_similarity(p, q, dim=-1)
            tot = tot + (p - q).abs().mean() - F.logsigmoid(c).mean()
            cos[t] = float(c.mean())
        return tot, cos

    aug = None
    if a.augment:
        from twjudge.accent.augment import Augmenter

        aug = Augmenter(a.augment, device=DEV, seed=start)
    t0 = time.time()
    for step in range(start, a.steps):
        idx = rng.choice(train, size=a.bs)
        # bucket by length so the common crop wastes little audio
        idx = idx[np.argsort(lens[idx])][: a.bs]
        x = batch(idx)
        tgt = teacher_states(x)
        xs = aug(x) if aug is not None else x
        with torch.autocast("cuda" if DEV == "cuda" else "cpu", dtype=torch.bfloat16):
            pred, wpred = student.both(xs)
        loss, cos = loss_fn({t: v.float() for t, v in pred.items()}, tgt)
        if wt:
            wl, wcos = loss_fn({t: v.float() for t, v in wpred.items()}, whisper_states(x), targets=wt)
            loss = loss + a.whisper_weight * wl
            cos.update({f"w{t}": c for t, c in wcos.items()})
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            print(f"step {step} loss {float(loss):.3f} cos " + " ".join(f"{t}:{c:.3f}" for t, c in cos.items())
                  + f" {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.save == 0 or step + 1 == a.steps:
            torch.save({"state": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1}, out.with_suffix(".last.pt"))
        if a.minutes and time.time() - t0 > a.minutes * 60:
            print(f"time limit at step {step + 1}", flush=True)
            break
    # held-out cosine per teacher layer
    student.eval()
    cs = {t: [] for t in targets}
    with torch.no_grad():
        for i in held:
            x = torch.tensor(np.asarray(audio[off[i]:off[i + 1]], dtype=np.float32)[None] / 32768, device=DEV)
            _, cos = loss_fn({t: v.float() for t, v in student(x).items()}, teacher_states(x))
            for t, c in cos.items():
                cs[t].append(c)
    rep = {"steps": step + 1, "norm_loss": a.norm_loss, "augment": bool(a.augment), "whisper_targets": wt,
           "init": a.init, "held_out_cosine": {t: float(np.mean(v)) for t, v in cs.items()},
           "params": sum(p.numel() for p in student.parameters()), "teacher": a.teacher}
    print(json.dumps(rep), flush=True)
    torch.save({"config": student.config, "state": {k: v.cpu() for k, v in student.state_dict().items()},
                "teacher": a.teacher, "report": rep}, out)


if __name__ == "__main__":
    main()
