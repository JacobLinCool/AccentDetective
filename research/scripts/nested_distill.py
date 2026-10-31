"""Distil the teachers into the nested Accent Detective trunk (twjudge.accent.nested.NestedStudent).

Warm start from the deployed dual student (its layers, wav2vec 2.0 heads and CTC head; deeper layers start as the
identity). Every step, every exit predicts every target layer of the frozen teachers (Chinese wav2vec 2.0 base "v",
Chinese HuBERT base "b"): L1 + (−log σ(cosine)) per frame on the utterance-mean-removed states (what the judge reads,
the first student), summed over targets and exits. The CTC head on exit 1 learns the deployed CTC head's frame posteriors (KL), so
the alignment stays as precise as the deployed one (voice invariance needs it). Clips are the bundle's training
sets (unlabelled use), cropped per batch to the shortest clip (≤ 6 s). Learning rate: warm-up, then linear decay to
zero at the time limit. EMA of the weights (0.999 per step) is what is saved.

    python research/scripts/nested_distill.py --bundle /content/bench --sets cv_train --detective /content/det/model.safetensors \\
        --minutes 40 --out /content/nested/student.pt
"""
import argparse
import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

CROP_S = 6.0


def _load(path):
    import librosa

    try:
        return (librosa.load(path, sr=16000, mono=True)[0] * 32767).astype(np.int16)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--sets", nargs="+", default=["cv_train"])
    ap.add_argument("--detective", required=True)
    ap.add_argument("--exits", default="2,4,8,18")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ctc-weight", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    ap.add_argument("--init", help="nested student checkpoint to continue from (instead of the deployed student)")
    ap.add_argument("--extra", nargs="*", default=[], help="name=hours of extra unlabelled corpora (distill_data.py): "
                                                             "aishell1, thchs30")
    ap.add_argument("--work", default="/content/distill_src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F

    from twjudge.accent.compact import ctc_head, load_checkpoint
    from twjudge.accent.nested import NestedStudent
    from twjudge.accent.ssl_features import load_model
    from twjudge.accent.student import Student

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B = Path(a.bundle)
    paths = [str(B / json.loads(l)["audio"]) for s in a.sets for l in open(B / "sets" / f"{s}.jsonl")]
    paths = paths[: a.limit] if a.limit else paths
    t0 = time.time()
    with Pool(10) as p:
        clips = [c for c in p.map(_load, paths, chunksize=64) if c is not None and len(c) >= 16000]
        for spec in a.extra:
            from distill_data import decode_file, extra_files

            name, hours = spec.split("=")
            files, got = [str(f) for f in extra_files(name, Path(a.work))], 0.0
            for k in range(0, len(files), 4000):
                for w in p.map(decode_file, files[k:k + 4000], chunksize=64):
                    if w is not None and got < float(hours) * 3600:
                        clips.append(w); got += len(w) / 16000
                if got >= float(hours) * 3600:
                    break
            print(f"{name}: {got / 3600:.1f} h", flush=True)
    lens = np.array([len(c) for c in clips])
    held = rng.choice(len(clips), min(200, len(clips) // 5), replace=False)
    train = np.setdiff1d(np.arange(len(clips)), held)
    print(f"{len(clips)} clips, {lens.sum() / 16000 / 3600:.1f} h, loaded in {time.time() - t0:.0f}s", flush=True)

    exits = tuple(int(x) for x in a.exits.split(","))
    if a.init:
        ck0 = torch.load(a.init, map_location="cpu", weights_only=False)
        net = NestedStudent(**{k: ck0["config"][k] for k in ("exits", "targets", "d", "heads", "ff")})
        net.load_state_dict(ck0["state"])
        net = net.to(dev)
    else:
        net = NestedStudent.from_detective(a.detective, exits=exits).to(dev)
    exits = net.exits
    ema = copy.deepcopy(net).eval()
    ck = load_checkpoint(a.detective)
    old = Student(**ck["student_config"]); old.load_state_dict(ck["student_state"]); old = old.to(dev).eval()
    old_ctc = ctc_head(arch="conv"); old_ctc.load_state_dict(ck["ctc_state"]); old_ctc = old_ctc.to(dev).eval()
    teachers = {"v": load_model("wav2vec2").to(dev).to(torch.bfloat16).eval(),
                "b": load_model("hubert").to(dev).to(torch.bfloat16).eval()}
    for m in [old, old_ctc, *teachers.values()]:
        for q in m.parameters():
            q.requires_grad_(False)
    tg_layers = {k: sorted({int(t[1:]) for t in net.targets if t[0] == k}) for k in teachers}
    print(f"nested {sum(q.numel() for q in net.parameters()) / 1e6:.2f}M, sizes "
          + " ".join(f"{net.size(k) / 1e6:.2f}M" for k in range(1, len(exits) + 1)), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    budget = a.minutes * 60

    def batch(idx):
        L = int(min(CROP_S * 16000, lens[idx].min()))
        xs = []
        for i in idx:
            o = rng.integers(0, lens[i] - L + 1)
            xs.append(clips[i][o:o + L].astype(np.float32) / 32768)
        return torch.tensor(np.stack(xs), device=dev)

    def targets(x):
        with torch.no_grad():
            xs = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            out = {}
            for k, m in teachers.items():
                hs = m(xs.to(torch.bfloat16), output_hidden_states=True).hidden_states
                out.update({f"{k}{l}": hs[l].float() for l in tg_layers[k]})
            with torch.autocast(dev, dtype=torch.bfloat16):
                lp = torch.log_softmax(old_ctc(old(x)[4]).float(), -1)
        return out, lp

    def dist(p, q):
        T = min(p.shape[1], q.shape[1])
        p, q = p[:, :T], q[:, :T]
        p, q = p - p.mean(1, keepdim=True), q - q.mean(1, keepdim=True)
        c = F.cosine_similarity(p, q, dim=-1)
        return (p - q).abs().mean() - F.logsigmoid(c).mean(), float(c.detach().mean())

    step, t1 = 0, time.time()
    while time.time() - t1 < budget:
        frac = (time.time() - t1) / budget
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 500) * max(0.0, 1 - frac)
        idx = rng.choice(train, size=a.bs)
        x = batch(idx)
        tgt, lp_old = targets(x)
        with torch.autocast(dev, dtype=torch.bfloat16):
            outs = net(x)
            lp = net.emissions(outs).float()
        loss, cos = 0.0, {}
        for e, o in enumerate(outs):
            for t in net.targets:
                l_, c_ = dist(o[t].float(), tgt[t])
                loss = loss + l_
                cos[f"{e + 1}{t}"] = c_
        T = min(lp.shape[1], lp_old.shape[1])
        kl = (lp_old[:, :T].exp() * (lp_old[:, :T] - lp[:, :T])).sum(-1).mean()
        loss = loss + a.ctc_weight * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        if step % 250 == 0:
            print(f"step {step} {time.time() - t1:.0f}s loss {float(loss):.3f} kl {float(kl):.3f} cos "
                  + " ".join(f"{k}:{v:.3f}" for k, v in cos.items() if k.endswith(("v12", "b12"))), flush=True)
        step += 1
    # held-out cosine per exit and target, and CTC agreement with the deployed head (frame argmax)
    rep = {"steps": step, "minutes": a.minutes, "init": a.init, "extra": a.extra, "hours": float(lens.sum() / 16000 / 3600), "exits": list(exits), "sizes": [net.size(k) for k in range(1, len(exits) + 1)]}
    cs, agree = {}, []
    with torch.no_grad():
        for i in held:
            x = torch.tensor(clips[i].astype(np.float32)[None] / 32768, device=dev)
            tgt, lp_old = targets(x)
            outs = ema(x)
            lp = ema.emissions(outs).float()
            T = min(lp.shape[1], lp_old.shape[1])
            agree.append(float((lp[0, :T].argmax(-1) == lp_old[0, :T].argmax(-1)).float().mean()))
            for e, o in enumerate(outs):
                for t in ema.targets:
                    cs.setdefault(f"{e + 1}{t}", []).append(dist(o[t].float(), tgt[t])[1])
    rep["held_out_cosine"] = {k: round(float(np.mean(v)), 4) for k, v in cs.items()}
    rep["ctc_frame_agreement"] = float(np.mean(agree))
    print(json.dumps(rep), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": ema.config, "state": {k: v.cpu() for k, v in ema.state_dict().items()}, "report": rep}, a.out)


if __name__ == "__main__":
    main()
