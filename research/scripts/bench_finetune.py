"""Strongest baseline: fine-tune a Chinese SSL model end to end as an utterance accent classifier.

No alignment and no text input: the waveform goes through the SSL encoder (convolutional front
end frozen), the last-layer states are averaged over the utterance, and a linear layer gives the
log-odds of Taiwan. Training uses the same utterance importance weights as the judges (sentence
length × text context), random crops of at most MAX_S seconds, and keeps the epoch with the best
length-matched dev AUC. Scores every benchmark set with the full utterance (up to 20 s).

    python research/scripts/bench_finetune.py --bundle /content/bench --model wav2vec2 --out /content/bench/ft
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))
sys.path.insert(0, str(ROOT / "research/scripts"))

MAX_S, EVAL_S = 6.0, 20.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--model", choices=["hubert", "wav2vec2"], default="wav2vec2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-only", action="store_true", help="load the saved model and score sets not scored yet")
    ap.add_argument("--sets", nargs="*", help="score only these sets (default: every set of the bundle)")
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    from bench_features import audio_iter
    from transformers import HubertModel, Wav2Vec2Model

    from twjudge.accent.ssl_features import HUBERT, WAV2VEC2

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    bundle, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = a.sets or sorted(p.stem for p in (bundle / "sets").glob("*.jsonl"))

    def load_set(name, max_s):
        rows = [json.loads(l) for l in open(bundle / "sets" / f"{name}.jsonl")]
        wavs = [w[: int(max_s * 16000)].astype(np.float16) for _, w in audio_iter(bundle, rows)]
        return rows, wavs

    t0 = time.time()
    tag = f"ft_{a.model}_s{a.seed}"
    cls, (name, rev) = (HubertModel, HUBERT) if a.model == "hubert" else (Wav2Vec2Model, WAV2VEC2)
    enc = cls.from_pretrained(name, revision=rev).cuda()
    head = nn.Linear(enc.config.hidden_size, 1).cuda()
    if a.score_only:
        ck = torch.load(out / f"{tag}.pt")
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    else:
        train(a, enc, head, load_set, bundle, rng, t0, out, tag)
    res = json.load(open(out / f"{tag}_logits.json")) if (out / f"{tag}_logits.json").exists() else {}
    for s in sets:
        if s in res:
            continue
        rows, wavs = load_set(s, EVAL_S)
        res[s] = dict(zip([r["item_id"] for r in rows], map(float, predict(enc, head, wavs))))
        print(f"scored {s} {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out / f"{tag}_logits.json", "w"))


def forward(enc, head, batch):
    import torch

    n = max(len(w) for w in batch)
    x = torch.zeros(len(batch), n)
    frames = []
    for k, w in enumerate(batch):
        w = torch.tensor(w.astype(np.float32))
        x[k, :len(w)] = (w - w.mean()) / (w.std() + 1e-7)
        frames.append(max(1, len(w) // 320))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(x.cuda()).last_hidden_state
    T = h.shape[1]
    m = (torch.arange(T, device="cuda")[None] < torch.tensor(frames, device="cuda")[:, None]).float()
    pooled = (h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    return head(pooled).squeeze(-1)


def predict(enc, head, wavs):
    import torch

    enc.eval()
    order = np.argsort([len(w) for w in wavs])
    z = np.zeros(len(wavs))
    with torch.no_grad():
        for s in range(0, len(order), 8):
            idx = order[s:s + 8]
            z[idx] = forward(enc, head, [wavs[i] for i in idx]).float().cpu().numpy()
    return z


def train(a, enc, head, load_set, bundle, rng, t0, out, tag):
    import torch
    import torch.nn as nn

    from twjudge.accent.neural import utterance_weights
    from twjudge.accent.prosodic import auc, length_weights

    tr_rows, tr_w = load_set("cv_train", 30)
    dv_rows, dv_w = load_set("cv_dev", EVAL_S)
    f = np.load(bundle / "feat" / "feat_cv_train.npz")
    ytr = np.array([r["item_id"].startswith("tw") for r in tr_rows], np.float32)
    wtr = utterance_weights({"y": ytr, "lens": f["lens"], "X": f["X"]})
    fd = np.load(bundle / "feat" / "feat_cv_dev.npz")
    ydv = np.array([r["item_id"].startswith("tw") for r in dv_rows], np.float32)
    lw_dv = length_weights(fd["lens"], ydv.astype(int))
    print(f"audio loaded {time.time() - t0:.0f}s", flush=True)

    enc.feature_extractor._freeze_parameters()
    opt = torch.optim.AdamW([{"params": [p for p in enc.parameters() if p.requires_grad], "lr": 3e-5},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    bs = 16
    steps = a.epochs * (len(tr_w) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 1e-3], total_steps=steps, pct_start=0.1)
    best, best_state = -1, None
    crop = int(MAX_S * 16000)
    for ep in range(a.epochs):
        enc.train()
        perm = rng.permutation(len(tr_w))
        for s in range(0, len(perm) - bs + 1, bs):
            idx = perm[s:s + bs]
            batch = []
            for i in idx:
                w = tr_w[i]
                if len(w) > crop:
                    o = rng.integers(0, len(w) - crop)
                    w = w[o:o + crop]
                batch.append(w)
            z = forward(enc, head, batch)
            loss = (nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(ytr[idx], device="cuda"),
                                                                    reduction="none")
                    * torch.tensor(wtr[idx], device="cuda")).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
        zd = predict(enc, head, dv_w)
        aa = auc(ydv, zd, lw_dv)
        print(f"epoch {ep} dev auc {aa:.4f} {time.time() - t0:.0f}s", flush=True)
        if aa > best:
            best = aa
            best_state = ({k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    torch.save({"enc": best_state[0], "head": best_state[1], "best_dev_auc": best}, out / f"{tag}.pt")
    json.dump({"best_dev_auc": best}, open(out / f"{tag}_report.json", "w"))


if __name__ == "__main__":
    main()
