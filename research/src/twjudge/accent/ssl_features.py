"""Syllable-pooled self-supervised speech features for the neural accent judge.

A frozen Chinese HuBERT (TencentGameMate/chinese-hubert-base, MIT; judge B) or Chinese wav2vec 2.0
(TencentGameMate/chinese-wav2vec2-base, MIT; judge C, an independent second judge) encodes each utterance; the
hidden states of a few layers are normalised per utterance (mean removed, so speaker and channel
offsets cancel to first order) and mean-pooled over every aligned syllable and over its rhyme.
The same pooling is differentiable, so the trained judge can later be run end to end on audio.

    python -m twjudge.accent.ssl_features --align data/judge_corpus/align_dev --out data/judge_corpus/ssl_dev.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HUBERT = ("TencentGameMate/chinese-hubert-base", "fce0375452b1dd6c080ac3248d423d4d037bc831")
WAV2VEC2 = ("TencentGameMate/chinese-wav2vec2-base", "3991242c806928916fff4a8c0e4f76acf661b743")
LAYERS = (4, 8, 12)
HOP_S = 0.02


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(kind: str = "hubert"):
    from transformers import HubertModel, Wav2Vec2Model

    cls, (name, rev) = (HubertModel, HUBERT) if kind == "hubert" else (Wav2Vec2Model, WAV2VEC2)
    m = cls.from_pretrained(name, revision=rev).to(device()).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def frame_states(model, wav16k):
    """wav16k: 1-D float tensor at 16 kHz → (len(LAYERS), frames, 768), per-utterance mean removed."""
    import torch

    x = (wav16k - wav16k.mean()) / (wav16k.std() + 1e-7)
    out = model(x[None], output_hidden_states=True)
    h = torch.stack([out.hidden_states[k][0] for k in LAYERS])
    return h - h.mean(dim=1, keepdim=True)


def pool(h, chars):
    """Mean over each syllable and over its rhyme → (syllables, 2 * len(LAYERS) * 768)."""
    import torch

    T = h.shape[1]
    rows = []
    for c in chars:
        a, r, b = (min(T - 1, int(c[k] / HOP_S)) for k in ("start", "rhyme_start", "end"))
        b = max(b, a + 1)
        r = min(max(r, a), b - 1)
        rows.append(torch.cat([h[:, a:b].mean(1).flatten(), h[:, r:b].mean(1).flatten()]))
    return torch.stack(rows)


def pool_fast(h, chars):
    """`pool` without the per-syllable loop: one cumulative sum over time, the same output."""
    import torch

    L, T, D = h.shape
    a = torch.tensor([min(T - 1, int(c["start"] / HOP_S)) for c in chars], device=h.device)
    b = torch.tensor([min(T - 1, int(c["end"] / HOP_S)) for c in chars], device=h.device)
    r = torch.tensor([min(T - 1, int(c["rhyme_start"] / HOP_S)) for c in chars], device=h.device)
    b = torch.maximum(b, a + 1)
    r = torch.minimum(torch.maximum(r, a), b - 1)
    cs = torch.cat([torch.zeros(L, 1, D, device=h.device, dtype=torch.float32), h.float().cumsum(1)], dim=1)
    syl = (cs[:, b] - cs[:, a]) / (b - a).float()[None, :, None]          # (L, n, D)
    rhy = (cs[:, b] - cs[:, r]) / (b - r).float()[None, :, None]
    n = len(chars)
    return torch.cat([syl.permute(1, 0, 2).reshape(n, L * D), rhy.permute(1, 0, 2).reshape(n, L * D)], dim=1).to(h.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--align", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="npz whose `ids` restrict and order the utterances (feature file)")
    ap.add_argument("--model", choices=["hubert", "wav2vec2"], default="hubert")
    a = ap.parse_args()
    import librosa
    import torch

    align = Path(a.align)
    wavs = {p.stem: p for c in align.glob("corpus*") for p in c.glob("*/*.wav")}
    prof = {json.loads(l)["item_id"]: json.loads(l) for l in (align / "profile.jsonl").open()}
    ids = list(np.load(a.ids)["ids"]) if a.ids else [i for i, r in prof.items() if r["status"] == "ok" and i in wavs]
    model = load_model(a.model)
    feats, lens = [], []
    with torch.inference_mode():
        for k, iid in enumerate(ids):
            w, _ = librosa.load(str(wavs[iid]), sr=16000, mono=True)
            h = frame_states(model, torch.tensor(w, device=device()))
            feats.append(pool(h, prof[iid]["chars"]).half().cpu().numpy())
            lens.append(len(prof[iid]["chars"]))
            if k % 1000 == 0:
                print(k, len(ids), flush=True)
    np.savez(a.out, ids=np.array(ids), F=np.concatenate(feats), lens=np.array(lens))
    print(len(ids), "utterances", sum(lens), "syllables")


if __name__ == "__main__":
    main()
