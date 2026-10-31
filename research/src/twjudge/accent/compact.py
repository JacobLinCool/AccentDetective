"""Compact accent judge: one truncated SSL encoder serves both the syllable alignment and the judge.

The deployed judge runs two large networks per utterance: MMS (wav2vec 2.0, 315M parameters) for
the CTC alignment and a 12-layer Chinese SSL model (94M) for the features. Here a single SSL
encoder truncated to its first `n_layers` layers is run once:

  alignment   a CTC head (LayerNorm + linear, toneless pinyin letters) on the states of layer
              `ctc_layer`, trained on Common Voice transcripts, feeds the same forced alignment and
              span rules as `twjudge.align.ctc` (letters, pauses from the energy envelope)
  features    the per-utterance-normalised states of `judge_layers`, pooled over every syllable
              and its rhyme, layer by layer ([syllable, rhyme] per layer)
  judge       the syllable transformer of `twjudge.accent.neural`

A checkpoint may hold several judges trained on the same features (seeds); their log-odds are averaged.
`CompactJudge(path).score(wav16k, text)` returns the utterance log-odds of Taiwan, the per-syllable
log-odds and the alignment; `encoder(kind, n_layers)` and `pool_layers` are shared with training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"          # pinyin bodies use a–z, ü written as v
VOCAB = {c: k + 1 for k, c in enumerate(LETTERS)}    # 0 = CTC blank


def encoder(kind: str, n_layers: int, device: str | None = None):
    """The first `n_layers` transformer layers of the judges' Chinese SSL model, frozen; or, for
    kind "student:<checkpoint>", a distilled student (`twjudge.accent.student`), n_layers ignored."""
    from transformers import HubertModel, Wav2Vec2Model

    from twjudge.accent.ssl_features import HUBERT, WAV2VEC2, device as default_device

    if kind.startswith("student:"):
        from twjudge.accent import student

        m = student.load(kind.split(":", 1)[1], device or default_device())
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    cls, (name, rev) = (HubertModel, HUBERT) if kind == "hubert" else (Wav2Vec2Model, WAV2VEC2)
    m = cls.from_pretrained(name, revision=rev)
    m.encoder.layers = m.encoder.layers[:n_layers]
    m.config.num_hidden_layers = n_layers
    m = m.to(device or default_device()).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def hidden(model, wav16k):
    """All hidden states (layer 0 = after the convolutional front end), (L + 1, T, 768). A student
    returns its predictions at the teacher's layer indices (other layers are zero)."""
    import torch

    from twjudge.accent.student import Student

    if isinstance(model, Student):
        out = model(wav16k[None])
        T = next(iter(out.values())).shape[1]
        hs = torch.zeros(max(out) + 1, T, 768, device=wav16k.device)
        for t, v in out.items():
            hs[t] = v[0]
        return hs
    x = (wav16k - wav16k.mean()) / (wav16k.std() + 1e-7)
    out = model(x[None], output_hidden_states=True)
    return torch.stack([h[0] for h in out.hidden_states])


def ctc_head(d: int = 768, arch: str = "linear"):
    """CTC head over toneless pinyin letters: "linear" (LayerNorm + linear, ~22k parameters) or "conv"
    (two 5-tap convolutions of width 256 before the output layer, ~1.3M, for local context)."""
    import torch.nn as nn

    if arch == "conv":
        class Conv(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = nn.LayerNorm(d)
                self.c1 = nn.Conv1d(d, 256, 5, padding=2)
                self.c2 = nn.Conv1d(256, 256, 5, padding=2)
                self.out = nn.Linear(256, len(VOCAB) + 1)
                self.act = nn.GELU()

            def forward(self, x):                     # (..., T, d), batched or not
                squeeze = x.dim() == 2
                h = self.norm(x if not squeeze else x[None]).transpose(1, 2)
                h = self.act(self.c2(self.act(self.c1(h)))).transpose(1, 2)
                o = self.out(h)
                return o[0] if squeeze else o

        return Conv()
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, len(VOCAB) + 1))


def pool_layers(hs, layers, chars):
    """Per-utterance-normalised states of `layers`, mean over each syllable and its rhyme, concatenated
    layer by layer as [syllable, rhyme] (the layout of the per-layer feature files)."""
    from twjudge.accent.ssl_features import pool_fast

    h = hs[list(layers)]
    h = h - h.mean(dim=1, keepdim=True)
    n, L, D = len(chars), len(layers), h.shape[2]
    f = pool_fast(h, chars).reshape(n, 2, L, D)                 # [syllable L…, rhyme L…] → per layer [s, r]
    return f.permute(0, 2, 1, 3).reshape(n, L * 2 * D)


def context(chars):
    """Text-context rows of the judge (tone, sandhi tone, Mainland-neutral, phrase-final, position)."""
    from twjudge.phonology import sandhi_tones, split_syllable

    tones = [split_syllable(c["syl"])[2] for c in chars]
    n = len(chars)
    fin = [k + 1 == n or chars[k + 1]["start"] - c["end"] >= 0.15 for k, c in enumerate(chars)]
    return [[int(t), int(s), int(c["cn_neutral"]), int(f), k / max(n - 1, 1)]
            for k, (t, s, c, f) in enumerate(zip(tones, sandhi_tones(tones), chars, fin))]


META_KEYS = ("ssl", "student_config", "n_layers", "ctc_layer", "ctc_arch", "judge_layers", "d_in", "teacher",
             "judge_nested", "judge_prefix", "wjudge", "syl_types")


def save_checkpoint(ck: dict, path) -> None:
    """A self-contained compact checkpoint (compact_export.py) as safetensors: tensors under the prefixes
    student. / ctc. / judge.<k>., everything else as JSON in the metadata."""
    import json

    from safetensors.torch import save_file

    states = ck.get("judge_states") or [ck["judge_state"]]
    tensors = {f"student.{k}": v for k, v in ck["student_state"].items()}
    tensors.update({f"ctc.{k}": v for k, v in ck["ctc_state"].items()})
    for i, st in enumerate(states):
        tensors.update({f"judge.{i}.{k}": v for k, v in st.items()})
    for i, st in enumerate(ck.get("wjudge_states", [])):            # the Whisper-path judges of a dual checkpoint
        tensors.update({f"wjudge.{i}.{k}": v for k, v in st.items()})
    if "syl_means" in ck:
        tensors["sylnorm.means"], tensors["sylnorm.global"] = ck["syl_means"], ck["syl_global"]
    meta = {k: ck[k] for k in META_KEYS if k in ck}
    meta["n_judges"] = len(states)
    meta["n_wjudges"] = len(ck.get("wjudge_states", []))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path),
              metadata={"accent_detective": json.dumps(meta), "reports": json.dumps(ck.get("reports", {}), default=str)})


def load_checkpoint(path) -> dict:
    """The checkpoint dict of a .pt file, or rebuilt from a .safetensors file written by save_checkpoint."""
    import torch

    if not str(path).endswith(".safetensors"):
        return torch.load(path, map_location="cpu")
    import json

    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["accent_detective"])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    ck = dict(meta)

    def part(prefix):
        return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}

    ck["student_state"] = part("student.")
    ck["ctc_state"] = part("ctc.")
    ck["judge_states"] = [part(f"judge.{i}.") for i in range(meta["n_judges"])]
    ck["judge_state"] = ck["judge_states"][0]
    ck["wjudge_states"] = [part(f"wjudge.{i}.") for i in range(meta.get("n_wjudges", 0))]
    if "sylnorm.means" in tensors:
        ck["syl_means"], ck["syl_global"] = tensors["sylnorm.means"], tensors["sylnorm.global"]
    return ck


class CompactJudge:
    """mode: "accurate" (every path of the checkpoint) or "fast" (a dual checkpoint's wav2vec 2.0 path only:
    no Whisper heads, no Whisper-path judges)."""

    def __init__(self, path: str | Path, readings=None, device: str | None = None, mode: str = "accurate"):
        import torch

        from twjudge.accent.neural import SyllableNet
        from twjudge.align.ctc import CTCAligner

        ck = load_checkpoint(path)
        self.ck = ck
        if ck["ssl"] == "student":                    # self-contained checkpoint (compact_export.py)
            from twjudge.accent.ssl_features import device as default_device
            from twjudge.accent.student import Student

            self.enc = Student(**ck["student_config"])
            self.enc.load_state_dict(ck["student_state"])
            self.enc = self.enc.to(device or default_device()).eval()
        else:
            self.enc = encoder(ck["ssl"], ck["n_layers"], device)
        self.dev = next(self.enc.parameters()).device
        self.head = ctc_head(arch=ck.get("ctc_arch", "linear")); self.head.load_state_dict(ck["ctc_state"]); self.head.to(self.dev).eval()
        from twjudge.accent.neural import NestedSyllableNet

        def make(d_in, nested):
            return NestedSyllableNet(d_in // nested, nested) if nested else SyllableNet(d_in)

        states = ck["judge_states"] if "judge_states" in ck else [ck["judge_state"]]   # a seed ensemble or one judge
        self.nets = []
        for st in states:
            net = make(ck["d_in"], ck.get("judge_nested", 0)); net.load_state_dict(st); self.nets.append(net.to(self.dev).eval())
        self.net = self.nets[0]
        self.k = ck.get("judge_prefix")                  # prefix of a nested judge (None: all its layers)
        # dual checkpoint: judges on the student's predicted Whisper layers, syllable-identity normalised
        if mode not in ("accurate", "fast"):
            raise ValueError(f"mode must be 'accurate' or 'fast', not {mode!r}")
        self.mode = mode
        self.w = ck.get("wjudge") if mode == "accurate" else None
        self.wnets = []
        if self.w:
            for st in ck["wjudge_states"]:
                net = make(self.w["d_in"], self.w.get("nested", 0)); net.load_state_dict(st); self.wnets.append(net.to(self.dev).eval())
            self.syl_index = {t: i for i, t in enumerate(ck["syl_types"])}
            self.syl_means = ck["syl_means"].float().to(self.dev)             # (types, layers, 2 D)
            self.syl_global = ck["syl_global"].float().to(self.dev)           # (layers, 2 D)
        if readings is None:
            from twjudge.lexicon.readings import TaiwanReadings

            readings = TaiwanReadings(ck.get("lexicon", "data/lexicon/concised.json.xz"))
        self.aligner = CTCAligner(readings)

    def score(self, wav16k: np.ndarray, text: str):
        import torch

        from twjudge.accent.neural import utt_logits

        with torch.inference_mode():
            x = torch.tensor(wav16k, dtype=torch.float32, device=self.dev)
            if self.w:
                out, wout = self.enc.both(x[None])
                T = next(iter(out.values())).shape[1]
                hs = torch.zeros(max(out) + 1, T, 768, device=self.dev)
                for t, v in out.items():
                    hs[t] = v[0]
            else:
                hs = hidden(self.enc, x)
            em = torch.log_softmax(self.head(hs[self.ck["ctc_layer"]]), dim=-1)[None].float().cpu()
            chars = self.aligner.align(wav16k, text, emissions=(em, VOCAB))
            if chars is None:
                return None
            feat = pool_layers(hs, self.ck["judge_layers"], chars)[None].float()
            ctx = torch.tensor(context(chars), dtype=torch.float32, device=self.dev)[None]
            mask = torch.zeros((1, len(chars)), dtype=torch.bool, device=self.dev)
            syl = torch.stack([net(ctx, feat, mask) if self.k is None else net(ctx, feat, mask, self.k)
                               for net in self.nets]).mean(0)                   # ensemble: mean log-odds
            if self.w:                                   # fuse with the Whisper path: mean of the two log-odds
                L = self.w["layers"][:self.w["prefix"]]
                W = torch.stack([wout[l][0] for l in L])
                n = len(chars)
                f = pool_layers(W, list(range(len(L))), chars).reshape(n, len(L), -1)
                idx = [self.syl_index.get(c["syl"], -1) for c in chars]
                mu = torch.stack([self.syl_means[i, :len(L)] if i >= 0 else self.syl_global[:len(L)] for i in idx])
                wfeat = (f - mu).reshape(1, n, -1)
                wsyl = torch.stack([net(ctx, wfeat, mask, self.w["prefix"]) for net in self.wnets]).mean(0)
                syl = 0.5 * (syl + wsyl)
            return float(utt_logits(syl, mask)[0]), syl[0].cpu().numpy(), chars
