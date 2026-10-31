"""The unified nested accent judge: one set of weights, many deployable variants.

A variant is (states source, layer prefix k, mode):
  source  "S"  the dual student's predictions of the teachers' layers (CPU-sized), or
          "T"  the teachers' own states (Chinese wav2vec 2.0 base and the Whisper-medium encoder, GPU)
  k       how many layers of the source's nested order are read
  mode    "t" with the text (forced segmentation, tone / sandhi / Mainland-neutral context), or
          "f" without (S only): the student's CTC head decoded and split into pinyin syllables, context unknown
Both sources are segmented by the student's own CTC head (forced to the text, or decoded), so no MMS aligner is
needed. Each layer's syllable-pooled states (utterance mean removed) have a learned share of their syllable-type
mean subtracted; the judge sums one projection per layer in use and reads the sequence with a small transformer.

    from twjudge.accent.unified import UnifiedDetective
    det = UnifiedDetective("judge.pt", "judge_compact.safetensors", device="cpu")
    det.score(wav16k, text="...", variant="S1t")      # or variant="S1f" without text, "T6t" with the teachers
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

UNK = (6, 6, 2)


class Context(nn.Module):
    """Text context with an extra 'unknown' index for tone, sandhi tone and the Mainland-neutral flag."""

    def __init__(self, d):
        super().__init__()
        self.tone, self.sandhi = nn.Embedding(7, d), nn.Embedding(7, d)
        self.neu, self.fin = nn.Embedding(3, d), nn.Embedding(2, d)
        self.pos = nn.Linear(1, d)

    def forward(self, c):
        return (self.tone(c[..., 0].long()) + self.sandhi(c[..., 1].long()) + self.neu(c[..., 2].long())
                + self.fin(c[..., 3].long()) + self.pos(c[..., 4:5]))


class UnifiedJudge(nn.Module):
    """layers: {"T": [tags], "S": [tags]} in nested order; dims: {tag: 2 × hidden}; mus: {source: {tag: (V, dim)}}."""

    def __init__(self, dims: dict, mus: dict, layers: dict, d=192, n_layers=2, dropout=0.2, beta="learned",
                 per_source=False, beta_teacher="shared"):
        super().__init__()
        self.beta_teacher = beta_teacher          # "shared" (the same β for both sources) or "zero" (teacher states unchanged)
        self.layers = {s: list(v) for s, v in layers.items()}
        self.tags = self.layers["T"]
        self.per_source = per_source
        self.beta_mode = beta
        self.register_buffer("beta_fixed", torch.tensor([float(beta == "whisper" and t.startswith("h")) for t in self.tags]))
        keys = [f"{s}_{t}" for s in self.layers for t in self.layers[s]] if per_source else self.tags
        self.proj = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(dims[k.split("_")[-1]]), nn.Dropout(dropout),
                                                    nn.Linear(dims[k.split("_")[-1]], d)) for k in keys})
        self.beta_logit = nn.Parameter(torch.zeros(len(self.tags)))
        for src, m in mus.items():
            for t, mu in m.items():
                self.register_buffer(f"mu_{src}_{t}", torch.as_tensor(mu, dtype=torch.float16), persistent=True)
        self.prefix = nn.ParameterDict({s: nn.Parameter(torch.zeros(len(self.layers[s]), d)) for s in self.layers})
        self.ctx = Context(d)
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def forward(self, b, src, k):
        """b: ctx (B, S, 5), mask (B, S) True = padding, types (B, S), feats: k tensors (B, S, dim). Syllable log-odds."""
        h = self.ctx(b["ctx"]) + self.prefix[src][k - 1]
        beta = torch.sigmoid(self.beta_logit) if self.beta_mode == "learned" else self.beta_fixed
        if src == "T" and self.beta_teacher == "zero":
            beta = torch.zeros_like(beta)
        for tag, x in zip(self.layers[src][:k], b["feats"]):
            mu = getattr(self, f"mu_{src}_{tag}")[b["types"]].float()
            h = h + self.proj[f"{src}_{tag}" if self.per_source else tag](x.float() - beta[self.tags.index(tag)] * mu)
        return self.out(self.enc(h, src_key_padding_mask=b["mask"])).squeeze(-1)

    def scalars(self):
        return [self.beta_logit] if self.beta_mode == "learned" else []

    @classmethod
    def from_checkpoint(cls, ck: dict):
        st = ck["state"]
        mus = {}
        for key, v in st.items():
            if key.startswith("mu_"):
                _, src, tag = key.split("_", 2)
                mus.setdefault(src, {})[tag] = v
        net = cls(ck["dims"], mus, ck["layers"], beta=ck.get("beta", "learned"), per_source=ck.get("proj_per_source", False),
                  beta_teacher=ck.get("beta_teacher", "shared"))
        net.load_state_dict(st)
        return net.eval()


def inventory():
    """Toneless pinyin syllables (letters a–z, ü as v) of the CJK block, from pypinyin."""
    from pypinyin import Style, pinyin

    inv = set()
    for c in range(0x4E00, 0xA000):
        for p in pinyin(chr(c), style=Style.NORMAL, heteronym=True)[0]:
            p = p.replace("ü", "v")
            if p.isalpha() and p.isascii():
                inv.add(p)
    return inv


def split_syllables(s: str, inv: set, maxlen=6, skip_cost=10):
    """Split a decoded letter string into inventory syllables: fewest syllables, a letter that fits no syllable
    (a decoding error) may be skipped at a cost of `skip_cost` syllables. Returns [(syllable, start index)]."""
    n = len(s)
    best = [None] * (n + 1)
    best[n] = (0, [])
    for i in range(n - 1, -1, -1):
        opts = [(best[i + 1][0] + skip_cost, best[i + 1][1])]
        for L in range(min(maxlen, n - i), 0, -1):
            if s[i:i + L] in inv:
                opts.append((best[i + L][0] + 1, [(s[i:i + L], i)] + best[i + L][1]))
        best[i] = min(opts, key=lambda o: o[0])
    return best[0][1]


def pooled(h, chars):
    """h (L, T, D) → [(n, 2D) per layer]: syllable and rhyme means of the utterance-mean-removed states."""
    from twjudge.accent.ssl_features import pool_fast

    h = h.float()
    h = h - h.mean(1, keepdim=True)
    L, D = h.shape[0], h.shape[2]
    f = pool_fast(h, chars).reshape(len(chars), 2, L, D).permute(0, 2, 1, 3).reshape(len(chars), L, 2 * D)
    return [f[:, j] for j in range(L)]


class UnifiedDetective:
    """End-to-end scoring with a unified judge checkpoint (r2_train.py judge.pt) and the dual student checkpoint it
    was trained on (Accent Detective v3 model.safetensors: student, CTC head). teacher=True loads the teachers for
    the "T" variants (GPU advised)."""

    W2V_LAYERS = (4, 8, 12)
    WHISPER = "openai/whisper-medium"

    def __init__(self, judge_pt, detective, device: str | None = None, teacher: bool = False, readings=None):
        from twjudge.accent.compact import CompactJudge
        from twjudge.lexicon.readings import TaiwanReadings

        ck = torch.load(judge_pt, map_location="cpu", weights_only=False)
        self.vocab = ck["vocab"]
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UnifiedJudge.from_checkpoint(ck).to(self.dev)
        root = Path(__file__).resolve().parents[4]
        rw = readings or TaiwanReadings(root / "data/lexicon/concised.json.xz")
        self.cj = CompactJudge(detective, readings=rw, device=self.dev, mode="accurate")
        self.inv = inventory()
        self.teacher = teacher
        if teacher:
            from transformers import WhisperFeatureExtractor, WhisperModel

            from twjudge.accent.ssl_features import load_model

            self.ssl = load_model("wav2vec2").to(self.dev).eval()
            self.fe = WhisperFeatureExtractor.from_pretrained(self.WHISPER)
            dt = torch.float16 if self.dev == "cuda" else torch.float32
            self.wenc = WhisperModel.from_pretrained(self.WHISPER, torch_dtype=dt).encoder.to(self.dev).eval()

    def variants(self):
        L = self.net.layers
        v = [f"S{k}{m}" for m in "tf" for k in range(1, len(L["S"]) + 1)]
        return v + ([f"T{k}t" for k in range(1, len(L["T"]) + 1)] if self.teacher else [])

    @torch.inference_mode()
    def score(self, wav16k: np.ndarray, text: str | None = None, variant: str = "S1t"):
        """→ (utterance log-odds of Taiwan, per-syllable log-odds, syllables), or None if it cannot be segmented."""
        from twjudge.accent.compact import VOCAB, context
        from pypinyin.contrib.tone_convert import to_initials

        src, k, mode = variant[0], int(variant[1:-1]), variant[-1]
        if mode == "t" and not text:
            raise ValueError(f"variant {variant} needs the text")
        if src == "T" and not self.teacher:
            raise ValueError("load with teacher=True for the T variants")
        x = torch.tensor(wav16k, dtype=torch.float32, device=self.dev)
        out_w, out_W = self.cj.enc.both(x[None])
        em = torch.log_softmax(self.cj.head(out_w[self.cj.ck["ctc_layer"]][0]), dim=-1)[None].float().cpu()
        if mode == "t":
            chars = self.cj.aligner.align(wav16k, text, emissions=(em, VOCAB))
            if chars is None or len(chars) < 2:
                return None
            ctx = context(chars)
        else:
            letters = {v: c for c, v in VOCAB.items()}
            s, prev = [], 0
            for t in em[0].argmax(-1).tolist():
                if t != prev and t != 0:
                    s.append(letters[t])
                prev = t
            sylls = [p for p, _ in split_syllables("".join(s), self.inv)]
            if len(sylls) < 2:
                return None
            chars = self.cj.aligner.spans(em, VOCAB, [("?", p + "5", False, p, len(to_initials(p, strict=False) or ""), p + "5")
                                                      for p in sylls], wav16k)
            if chars is None:
                return None
            n = len(chars)
            fin = [j + 1 == n or chars[j + 1]["start"] - c["end"] >= 0.15 for j, c in enumerate(chars)]
            ctx = [[*UNK, int(f), j / max(n - 1, 1)] for j, f in enumerate(fin)]
        tags = self.net.layers[src][:k]
        if src == "S":
            states = {f"v{l}": out_w[l][0] for l in self.W2V_LAYERS}
            states.update({f"h{l}": out_W[l][0] for l in out_W})
        else:
            from twjudge.accent.ssl_features import frame_states

            tw = frame_states(self.ssl, x)
            states = {f"v{l}": tw[j] for j, l in enumerate(self.W2V_LAYERS)}
            if any(t.startswith("h") for t in tags):
                f = self.fe([wav16k[: 30 * 16000]], sampling_rate=16000, return_tensors="pt").input_features
                hs = self.wenc(f.to(self.dev, next(self.wenc.parameters()).dtype), output_hidden_states=True).hidden_states
                nfr = max(1, min(1500, math.ceil(min(len(wav16k), 30 * 16000) / 320)))
                states.update({f"h{l}": hs[l][0, :nfr] for l in (6, 12, 18, 24)})
        feats = [pooled(states[t][None], chars)[0][None] for t in tags]
        b = {"ctx": torch.tensor(ctx, dtype=torch.float32, device=self.dev)[None],
             "mask": torch.zeros((1, len(chars)), dtype=torch.bool, device=self.dev),
             "types": torch.tensor([[self.vocab.get(c["syl"].rstrip("012345"), 0) for c in chars]], device=self.dev),
             "feats": feats}
        syl = self.net(b, src, k)[0]
        return float(syl.mean()), syl.float().cpu().numpy(), chars
