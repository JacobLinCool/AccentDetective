"""Train the compact judge's alignment head: CTC over toneless pinyin letters on frozen SSL states.

The head (LayerNorm + linear, ~22k parameters) reads layer --layer of the Chinese SSL encoder
truncated to that depth, so alignment costs almost nothing on top of the judge's own forward pass.
Targets are the Mainland readings of the Common Voice training transcripts, spelt exactly as the MMS
aligner spells them (`CTCAligner.syllables`). Reports on CV dev: greedy letter error rate and the
agreement of the forced-alignment syllable boundaries with the MMS alignment.

    python research/scripts/compact_ctc_head.py --bundle /content/bench --kind wav2vec2 --layer 4 \
        --mms-profiles /content/bench/ctc_profiles --out /content/bench/heads/w2v_L4.pt
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--kind", required=True, help="hubert, wav2vec2 or student:<checkpoint>")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mms-profiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--arch", choices=["linear", "conv"], default="linear")
    ap.add_argument("--mms-kd", type=float, default=0.0,
                    help="weight of a frame-level KL to the MMS aligner's posteriors over the same letters + blank")
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    from bench_features import audio_iter

    from twjudge.accent import compact as CJ
    from twjudge.align.ctc import CTCAligner
    from twjudge.lexicon.readings import TaiwanReadings

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    bundle = Path(a.bundle)
    aligner = CTCAligner(TaiwanReadings(ROOT / "data/lexicon/concised.json.xz"))

    def load(name, limit=None):
        rows = [json.loads(l) for l in open(bundle / "sets" / f"{name}.jsonl")][:limit]
        data = []
        for r, w in audio_iter(bundle, rows):
            sy = aligner.syllables(r["text"], CJ.VOCAB)
            if sy is None:
                continue
            data.append((r, w.astype(np.float16), [CJ.VOCAB[c] for s in sy for c in s[3]], sy))
        return data

    t0 = time.time()
    train, dev = load("cv_train"), load("cv_dev", 600)
    print(f"{len(train)} train / {len(dev)} dev utterances, {time.time() - t0:.0f}s", flush=True)
    enc = CJ.encoder(a.kind, a.layer, "cuda")
    head = CJ.ctc_head(arch=a.arch).cuda()
    mms_idx = None
    if a.mms_kd:
        from twjudge.align.ctc import _bundle

        mms, mdict, _ = _bundle()
        mms_idx = torch.tensor([0] + [mdict[c] for c in CJ.LETTERS], device="cuda")   # blank, a..z
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=0.01)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    bs = 16

    def states(batch):
        with torch.no_grad():
            hs = [CJ.hidden(enc, torch.tensor(w.astype(np.float32), device="cuda"))[a.layer] for _, w, _, _ in batch]
        return hs

    for ep in range(a.epochs):
        head.train()
        perm = rng.permutation(len(train))
        tot = 0.0
        for s in range(0, len(perm), bs):
            batch = [train[i] for i in perm[s:s + bs]]
            hs = states(batch)
            T = max(h.shape[0] for h in hs)
            x = torch.zeros(len(hs), T, hs[0].shape[1], device="cuda")
            for k, h in enumerate(hs):
                x[k, :h.shape[0]] = h
            logits = head(x)
            lp = torch.log_softmax(logits, dim=-1).transpose(0, 1)             # (T, B, V)
            tg = torch.tensor([t for _, _, tt, _ in batch for t in tt], device="cuda")
            loss = ctc(lp, tg, torch.tensor([h.shape[0] for h in hs]), torch.tensor([len(tt) for _, _, tt, _ in batch]))
            if mms_idx is not None:                  # match MMS frame posteriors (its timing) on the same letters
                kl = 0.0
                for k, (_, w, _, _) in enumerate(batch):
                    with torch.no_grad():
                        em, _ = mms(torch.tensor(w.astype(np.float32), device="cuda")[None])
                        q = torch.log_softmax(em[0][:, mms_idx], dim=-1)
                    n = min(q.shape[0], hs[k].shape[0])
                    p_ = torch.log_softmax(logits[k, :n], dim=-1)
                    kl = kl + torch.nn.functional.kl_div(p_, q[:n], log_target=True, reduction="batchmean")
                loss = loss + a.mms_kd * kl / len(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(batch)
        print(f"epoch {ep} ctc {tot / len(train):.3f} {time.time() - t0:.0f}s", flush=True)

    # dev: greedy letter error rate and boundary agreement with the MMS alignment
    head.eval()
    mms = {}
    for p in Path(a.mms_profiles).glob("profile_cv_dev*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            mms[r["item_id"]] = r["chars"]
    import torchaudio.functional as F

    errs = n_let = 0
    dstart, drhyme, dend = [], [], []
    for r, w, tg, sy in dev:
        with torch.no_grad():
            em = torch.log_softmax(head(CJ.hidden(enc, torch.tensor(w.astype(np.float32), device="cuda"))[a.layer]), -1)
        hyp = [int(t) for t in torch.unique_consecutive(em.argmax(-1)) if t != 0]
        errs += F.edit_distance(hyp, tg); n_let += len(tg)
        chars = CTCAligner.spans(em[None].float().cpu(), CJ.VOCAB, sy, w.astype(np.float32))
        ref = mms.get(r["item_id"])
        if chars and ref and len(ref) == len(chars):
            dstart += [abs(c["start"] - q["start"]) for c, q in zip(chars, ref)]
            drhyme += [abs(c["rhyme_start"] - q["rhyme_start"]) for c, q in zip(chars, ref)]
            dend += [abs(c["end"] - q["end"]) for c, q in zip(chars, ref)]
    rep = {"letter_error_rate": errs / n_let,
           "boundary_vs_mms": {k: {"median_s": float(np.median(v)), "within_40ms": float(np.mean(np.array(v) <= 0.04))}
                               for k, v in (("start", dstart), ("rhyme_start", drhyme), ("end", dend))},
           "n_dev": len(dev)}
    print(json.dumps(rep), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ssl": a.kind, "ctc_layer": a.layer, "ctc_arch": a.arch,
                "ctc_state": {k: v.cpu() for k, v in head.state_dict().items()}, "report": rep}, a.out)


if __name__ == "__main__":
    main()
