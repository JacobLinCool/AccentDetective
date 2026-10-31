"""Train the nested trunk: continue the distilled nested student under one of several training designs (arms).

  A  exit 4's feature KD (wav2vec 2.0 v4 v8 v12, HuBERT b8 b12) + exit 1's CTC KL; the nested-distillation sampler (all clips uniform,
     24 per batch, cropped to the shortest, ≤ 6 s)
  B  A's losses; source-balanced sampling (Taiwan CV .5, Mainland CV .25, AISHELL-1 .125, THCHS-30 .125; speaker, then
     clip, uniform) and equal-length batches: L from the batch's first clip (≤ 8 s, a multiple of 0.32 s), every other
     clip ≥ L cropped to L, as many clips as fill A's mean seconds per batch (no padding anywhere)
  C  B + the accent task loss through exit 4 and its judge into the trunk, on full clips one at a time: soft targets of
     the HuBERT teacher-judge ensemble on new speakers, same-content Taiwan/Mainland pairs, channel consistency;
     weighted by the gradient-norm ratio measured at step 0, then fixed
  D  C on all four exits (feature KD and task loss per exit, summed); logs per-exit gradient norms and cosines on the
     layers every exit shares
  H  the released model: D from an arm-C trunk (--init, its judges too) with the exits moved (--exits),
     and one task-loss coefficient per exit: the ratio of that exit's feature-KD to task-loss gradient norm on the trunk,
     each averaged over 8 batches at step 0, then fixed

Every arm: the same K updates, warm-up then linear decay, EMA 0.999 saved after the last step.

    python research/scripts/train_trunk.py --arm C --seed 0 --pool /content/task_pool --bundle /content/bench \\
        --r10 /content/bench/r10 --init /content/bench/nested/student.pt --judge /content/bench/nested/judge.pt \\
        --detective /content/det/model.safetensors --out /content/bench/trunk_c/student.pt
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

SR, Q = 16000, 5120                       # length quantum 0.32 s (16 student frames)
SOURCES = {"tw_cv": 0.5, "cn_cv": 0.25, "aishell1": 0.125, "thchs30": 0.125}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list("ABCDRH"), required=True,
                    help="R: B's sampler and losses on every exit of a trunk trained from scratch (--arch)")
    ap.add_argument("--arch", choices=["conformer", "ecapa"], help="train this trunk from scratch (no --init)")
    ap.add_argument("--d", type=int, default=384, help="width of a from-scratch Conformer (heads d/64, ff 2d)")
    ap.add_argument("--layers", type=int, default=4, help="depth of a from-scratch Conformer (one exit)")
    ap.add_argument("--se", action="store_true", help="an SE gate after every Conformer layer")
    ap.add_argument("--exits", type=int, nargs="+", help="H: the exits of the trained net (the init's heads move with them)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--r10", help="align.json, pairs.json, teacher/ (arms C, D)")
    ap.add_argument("--init")
    ap.add_argument("--judge", help="the nested judges' checkpoint (judge.pt) (arms C, D: the task path's judges start from it)")
    ap.add_argument("--detective", required=True)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import librosa
    import torch
    import torch.nn.functional as F
    import torchaudio.functional as AF

    from twjudge.accent.compact import ctc_head, load_checkpoint
    from twjudge.accent.nested import (EcapaStudent, NestedJudge, NestedStudent, build_student, conformer_trunk_params,
                                       ecapa_channels_for)
    from twjudge.accent.ssl_features import load_model, pool_fast
    from twjudge.accent.student import Student

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    P = Path(a.pool)
    audio = np.load(P / "audio.npy", mmap_mode="r")
    off = np.load(P / "offsets.npy")
    meta = [json.loads(l) for l in open(P / "meta.jsonl")]
    lens = np.diff(off)
    n = len(lens)
    held = np.random.default_rng(0).choice(n, min(200, n // 10), replace=False)
    train = np.setdiff1d(np.arange(n), held)
    src = np.array([m["source"] for m in meta])
    spk = np.array([m["speaker"] for m in meta])

    # A's mean seconds per batch on this pool (fixed simulation), the exposure every arm gets per step
    g0 = np.random.default_rng(0)
    S_A = float(np.mean([24 * min(6 * SR, lens[g0.choice(train, 24)].min()) / SR for _ in range(1000)]))

    def batch_A():
        idx = rng.choice(train, 24)
        return idx, int(min(6 * SR, lens[idx].min()))

    by_src = {s: train[src[train] == s] for s in SOURCES}
    cache = {}

    def eligible(s, Lq):
        key = (s, Lq)
        if key not in cache:
            ids = by_src[s][lens[by_src[s]] >= Lq]
            sp = spk[ids]
            order = np.argsort(sp, kind="stable")
            u, start = np.unique(sp[order], return_index=True)
            cache[key] = (ids[order], np.r_[start, len(ids)]) if len(ids) else None
        return cache[key]

    def draw(s, Lq):
        e = eligible(s, Lq)
        ids, bounds = e
        k = rng.integers(len(bounds) - 1)
        return ids[rng.integers(bounds[k], bounds[k + 1])]

    names, probs = list(SOURCES), np.array(list(SOURCES.values()))

    def batch_B():
        s0 = names[rng.choice(4, p=probs)]
        c0 = draw(s0, 0)
        L = int(min(8 * SR, lens[c0]) // Q * Q)
        L = max(L, Q * 3)
        if lens[c0] < L:
            L = int(lens[c0] // Q * Q) or int(lens[c0])
        k = max(4, min(64, round(S_A * SR / L)))
        ok = [s for s in names if eligible(s, L) is not None]
        p = np.array([SOURCES[s] for s in ok]); p = p / p.sum()
        idx = [c0] + [draw(ok[rng.choice(len(ok), p=p)], L) for _ in range(k - 1)]
        return np.array(idx), L

    def crops(idx, L):
        xs = []
        for i in idx:
            o = off[i] + rng.integers(0, lens[i] - L + 1)
            xs.append(np.asarray(audio[o:o + L], dtype=np.float32) / 32768)
        return torch.tensor(np.stack(xs), device=dev)

    if a.init:
        ck0 = torch.load(a.init, map_location="cpu", weights_only=False)
        cfg = dict(ck0["config"], **({"exits": a.exits} if a.exits else {}))
        assert len(cfg["exits"]) == len(ck0["config"]["exits"]) and max(cfg["exits"]) == max(ck0["config"]["exits"])
        net = build_student(cfg)
        net.load_state_dict(ck0["state"])           # exit k's norm and heads keep their weights at the new position
    elif a.arch == "ecapa":                     # channels matched to the Conformer trunk's parameters
        net = EcapaStudent(C=ecapa_channels_for(conformer_trunk_params()))
    else:
        net = NestedStudent(exits=(a.layers,), d=a.d, heads=a.d // 64, ff=2 * a.d, se=a.se)
    net = net.to(dev)
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
    kd_exits = list(range(len(net.exits))) if a.arm == "R" else [3] if a.arm in "ABC" else [0, 1, 2, 3]
    per_exit = a.arm == "H"
    task_exits = [] if a.arm in "ABR" else kd_exits

    def targets(x):
        with torch.no_grad():
            xs = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)
            t = {}
            for k, m in teachers.items():
                hs = m(xs.to(torch.bfloat16), output_hidden_states=True).hidden_states
                t.update({f"{k}{l}": hs[l].float() for l in tg_layers[k]})
            with torch.autocast(dev, dtype=torch.bfloat16):
                lp = torch.log_softmax(old_ctc(old(x)[4]).float(), -1)
        return t, lp

    def dist(p, q):
        T = min(p.shape[1], q.shape[1])
        p, q = p[:, :T], q[:, :T]
        p, q = p - p.mean(1, keepdim=True), q - q.mean(1, keepdim=True)
        c = F.cosine_similarity(p, q, dim=-1)
        return (p - q).abs().mean() - F.logsigmoid(c).mean()

    def kd_loss(x, per_exit=False):
        tgt, lp_old = targets(x)
        with torch.autocast(dev, dtype=torch.bfloat16):
            outs = net(x, upto=max(kd_exits) + 1)
            lp = net.emissions(outs).float()
        T = min(lp.shape[1], lp_old.shape[1])
        kl = (lp_old[:, :T].exp() * (lp_old[:, :T] - lp[:, :T])).sum(-1).mean()
        terms = {e: sum(dist(outs[e][t].float(), tgt[t]) for t in net.targets) for e in kd_exits}
        return (terms if per_exit else sum(terms.values())), kl

    # ---- the task path (arms C, D)
    judge = None
    if task_exits:
        R = Path(a.r10)
        al = json.load(open(R / "align.json"))
        pairs = [p for p in json.load(open(R / "pairs.json")) if p[0] in al and p[1] in al]
        tfiles = sorted((R / "teacher").glob("*.json"))
        tsc = [json.load(open(f))["r10_task"] for f in tfiles]
        teach = {i: float(np.mean([s[i][0] for s in tsc])) for i in tsc[0]
                 if all(s.get(i) is not None for s in tsc) and i in al}
        task_rows = {json.loads(l)["item_id"]: json.loads(l) for l in open(Path(a.bundle) / "sets" / "r10_task.jsonl")}
        task_ids = sorted(teach)
        wav = {i: librosa.load(str(Path(a.bundle) / task_rows[i]["audio"]), sr=SR, mono=True)[0] for i in task_ids}
        pool_pos = {m["id"]: k for k, m in enumerate(meta)}
        pairs = [p for p in pairs if p[0] in pool_pos and p[1] in pool_pos]
        jck = torch.load(a.judge, map_location="cpu", weights_only=False)
        st = jck["state"]
        mus = {}
        for key, v in st.items():
            if key.startswith("mu_"):
                _, e, t = key.split("_", 2)
                mus.setdefault(int(e), {})[t] = v
        judge = NestedJudge(jck["exits"], jck["tags"], jck["dims"], mus)
        judge.load_state_dict(st)
        if a.arm == "H" and ck0.get("judge_state"):     # the init's judges (arm C trained exit 4's)
            judge.load_state_dict(ck0["judge_state"])
        judge = judge.to(dev).train()
        vocab = jck["vocab"]
        print(f"task: {len(task_ids)} teacher-scored clips from {len(tfiles)} teacher judges, {len(pairs)} same-content pairs",
              flush=True)

    def clip_of(i):
        if i in wav:
            return wav[i]
        k = pool_pos[i]
        return np.asarray(audio[off[k]:off[k + 1]], dtype=np.float32) / 32768

    def channel(x):
        k = rng.integers(4)
        if k == 0:
            snr = rng.uniform(10, 30)
            return x + torch.randn_like(x) * x.pow(2).mean().sqrt() / (10 ** (snr / 20))
        if k == 1:
            return AF.lowpass_biquad(AF.highpass_biquad(x, SR, 300.0), SR, 3400.0)
        if k == 2:
            return AF.mu_law_decoding(AF.mu_law_encoding(x.clamp(-1, 1), 256), 256)
        return AF.lowpass_biquad(x, SR, 4000.0)

    def utt(x, i):
        """{exit: utterance log-odds} of one full clip through the trunk, that exit's heads and judge."""
        A_ = al[i]
        chars, ctx = A_["chars"], A_["ctx"]
        with torch.autocast(dev, dtype=torch.bfloat16):
            outs = net(x[None], upto=max(task_exits) + 1)
        res = {}
        for e in task_exits:
            H = torch.stack([outs[e][t][0].float() for t in net.targets])
            H = H - H.mean(1, keepdim=True)
            nc, L_, D = len(chars), H.shape[0], H.shape[2]
            f = pool_fast(H, chars).reshape(nc, 2, L_, D).permute(0, 2, 1, 3).reshape(nc, L_, 2 * D)
            b = {"ctx": torch.tensor(ctx, dtype=torch.float32, device=dev)[None],
                 "mask": torch.zeros((1, nc), dtype=torch.bool, device=dev),
                 "types": torch.tensor([[vocab.get(c["syl"].rstrip("012345"), 0) for c in chars]], device=dev),
                 "feats": [f[None, :, j] for j in range(L_)]}
            res[e] = judge(b, e).mean()
        return res

    def task_loss(per_exit=False):
        items = [task_ids[k] for k in rng.choice(len(task_ids), 4, replace=False)]
        z = [utt(torch.tensor(clip_of(i), device=dev), i) for i in items]
        p = torch.tensor([1 / (1 + np.exp(-teach[i])) for i in items], device=dev)
        zc = [utt(channel(torch.tensor(clip_of(i), device=dev)[None])[0], i) for i in items[:2]]
        pr = [pairs[k] for k in rng.choice(len(pairs), 2, replace=False)]
        zp = [(utt(torch.tensor(clip_of(t), device=dev), t), utt(torch.tensor(clip_of(c), device=dev), c)) for t, c in pr]
        terms = {}
        for e in task_exits:
            kd = F.binary_cross_entropy_with_logits(torch.stack([d[e] for d in z]), p)
            pair = -F.logsigmoid(torch.stack([zt[e] - zcn[e] for zt, zcn in zp])).mean()
            cons = torch.stack([(z[j][e] - zc[j][e]) ** 2 for j in range(2)]).mean()
            terms[e] = (kd + pair + cons) / 3
        return terms if per_exit else sum(terms.values())

    layers_ = net.enc.conformer_layers if hasattr(net, "enc") else []
    shared = [q for layer in layers_[: net.exits[0]] for q in layer.parameters()]
    trunk = [q for layer in layers_ for q in layer.parameters()]

    def gnorm(loss, params):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)), g

    # consistency checks: equal-length batch vs single clips (eval mode); task-path features vs nested_judge_features pooling
    with torch.no_grad():
        net.eval()
        idx, L = batch_B()
        x = crops(idx[:3], L)
        ob = net(x)[-1]["b12"]
        os_ = torch.cat([net(x[j:j + 1])[-1]["b12"] for j in range(3)])
        check = {"batch_vs_single_max_abs": float((ob - os_).abs().max())}
        net.train()
    lam = 1.0
    if task_exits and per_exit:                 # H: one coefficient per exit, gradient norms averaged over 8 batches
        nk, nt = {e: [] for e in task_exits}, {e: [] for e in task_exits}
        for _ in range(8):
            idx, L = batch_B()
            terms, _ = kd_loss(crops(idx, L), per_exit=True)
            tt = task_loss(per_exit=True)
            for e in task_exits:
                nk[e].append(float(gnorm(terms[e], trunk)[0])); nt[e].append(float(gnorm(tt[e], trunk)[0]))
        lam = {e: float(np.mean(nk[e]) / max(np.mean(nt[e]), 1e-8)) for e in task_exits}
        check.update({"grad_kd": {e + 1: nk[e] for e in task_exits}, "grad_task": {e + 1: nt[e] for e in task_exits},
                      "lambda": {e + 1: v for e, v in lam.items()}})
    elif task_exits:
        idx, L = batch_B()
        kd, kl = kd_loss(crops(idx, L))
        nk, _ = gnorm(kd + kl, trunk)
        nt, _ = gnorm(task_loss(), trunk)
        lam = float(nk / nt.clamp(min=1e-8))
        check.update({"grad_kd": float(nk), "grad_task": float(nt), "lambda": lam})
    lam_of = (lambda e: lam[e]) if per_exit else (lambda e: lam)
    print(json.dumps({"arm": a.arm, "seed": a.seed, "S_A_seconds_per_batch": S_A, **check}), flush=True)

    params = [{"params": list(net.parameters()), "lr": a.lr}]
    if judge is not None:
        params.append({"params": list(judge.parameters()), "lr": 3e-4})
    base = [g["lr"] for g in params]
    opt = torch.optim.AdamW(params, weight_decay=0.01, betas=(0.9, 0.98))
    secs = {s: 0.0 for s in SOURCES}
    crop_s, conflict, log = [], [], []
    t0 = time.time()
    for step in range(a.steps):
        f = min(1.0, (step + 1) / 500) * max(0.0, 1 - step / a.steps)
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * f
        idx, L = batch_A() if a.arm == "A" else batch_B()
        for i in idx:
            secs[src[i]] += L / SR
        crop_s.append(L / SR)
        x = crops(idx, L)
        if a.arm in "DH" and step % 1000 == 0:             # per-exit gradients on the layers every exit shares
            terms, kl = kd_loss(x, per_exit=True)
            tt = task_loss(per_exit=True)
            gs = {e: gnorm(terms[e] + lam_of(e) * tt[e], shared)[1] for e in kd_exits}
            flat = {e: torch.cat([q.flatten().float() for q in gs[e] if q is not None]) for e in kd_exits}
            conflict.append({"step": step, "norm": {e + 1: float(v.norm()) for e, v in flat.items()},
                             "cos": {f"{i + 1}-{j + 1}": float(F.cosine_similarity(flat[i], flat[j], dim=0))
                                     for i in kd_exits for j in kd_exits if i < j}})
            loss = sum(terms.values()) + kl + sum(lam_of(e) * tt[e] for e in kd_exits)
        elif per_exit:
            terms, kl = kd_loss(x, per_exit=True)
            tt = task_loss(per_exit=True)
            loss = sum(terms[e] + lam_of(e) * tt[e] for e in kd_exits) + kl
            tl = sum(tt.values())
        else:
            kd, kl = kd_loss(x)
            loss = kd + kl
            if task_exits:
                tl = task_loss()
                loss = loss + lam * tl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        if judge is not None:
            torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pn.detach(), alpha=0.001)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        if step % 250 == 0:
            rec = {"step": step, "s": round(time.time() - t0), "loss": round(float(loss), 4), "kl": round(float(kl), 4),
                   "L_s": round(L / SR, 2), "clips": len(idx)}
            if task_exits and not (a.arm in "DH" and step % 1000 == 0):
                rec["task"] = round(float(tl), 4)
            if per_exit and step % 1000 != 0:
                rec["kd_e"] = [round(float(terms[e]), 4) for e in kd_exits]
                rec["task_e"] = [round(float(tt[e]), 4) for e in kd_exits]
            log.append(rec)
            print(json.dumps(rec), flush=True)
    agree = []                                  # alignment constraint: CTC frame argmax vs the deployed head, held-out clips
    with torch.no_grad():
        for i in held[:200]:
            x = torch.tensor(np.asarray(audio[off[i]:off[i + 1]], dtype=np.float32)[None] / 32768, device=dev)
            lp_old = old_ctc(old(x)[4]).float()
            lp = ema.emissions(ema(x, upto=1)).float()
            T = min(lp.shape[1], lp_old.shape[1])
            agree.append(float((lp[0, :T].argmax(-1) == lp_old[0, :T].argmax(-1)).float().mean()))
    rep = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "ctc_frame_agreement": float(np.mean(agree)), "minutes": round((time.time() - t0) / 60, 1), "S_A": S_A,
           "lambda": lam, "check": check, "seconds_by_source": secs, "crop_seconds_median": float(np.median(crop_s)),
           "crop_seconds_mean": float(np.mean(crop_s)), "conflict": conflict, "log": log,
           "sizes": [net.size(k) for k in range(1, len(net.exits) + 1)]}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("log", "conflict")}), flush=True)
    torch.save({"config": ema.config, "state": {k: v.cpu() for k, v in ema.state_dict().items()}, "report": rep,
                "judge_state": None if judge is None else {k: v.cpu() for k, v in judge.state_dict().items()}}, out)
    json.dump(rep, open(out.with_name("train_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
