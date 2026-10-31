"""CUDA-graph execution of the compact judge: the same scores, without per-kernel launch overhead.

At batch size 1 the compact judge's GPU time is almost all kernel launches. Two captured graphs remove
them, with inputs padded to buckets so each bucket is captured once:

  (GPU)     the log-mel spectrogram of the exact-length waveform (a few kernels, outside the graph:
            the STFT's edge padding must see the real end of the audio)
  graph 1   log-mel bucket (0.5 s steps, zero-padded like the front end's own convolution padding)
            → student states of the judge's layers + CTC log-probs; padding masked and dropped
  (CPU)     forced alignment on the CTC log-probs, as in `twjudge.align.ctc`
  graph 2   (frame bucket, syllable bucket of 8) → per-utterance normalisation over the real frames,
            syllable / rhyme pooling by cumulative sums, the judge network with padding masked

`FastCompactJudge(path).score(wav16k, text)` returns what `CompactJudge.score` returns. Requires CUDA.
"""

from __future__ import annotations

import numpy as np

HOP_S = 0.02


class FastCompactJudge:
    def __init__(self, path, readings=None):
        import torch

        from twjudge.accent.compact import CompactJudge

        self.cj = CompactJudge(path, readings=readings, device="cuda")
        self.dev = self.cj.dev
        self.g1, self.g2 = {}, {}
        self.layers = list(self.cj.ck["judge_layers"])
        self.ctc_layer = self.cj.ck["ctc_layer"]
        self.stream = torch.cuda.Stream()

    # -- graph 1: encoder + CTC head ------------------------------------------------------------
    def _enc(self, m, n, nf, T):
        """m: log-mel, zero-padded; n: real teacher frames; nf: real front-end frames; T: bucket frames."""
        import torch

        s = self.cj.enc
        # the front end's first convolution must also end where the real mel frames end, or the
        # stride-2 convolution reads a non-zero frame where the unpadded run reads zero padding
        h1 = s.front[1](s.front[0](m))
        h1 = h1 * (torch.arange(h1.shape[2], device=m.device)[None] < self._nm[:, None]).float()[:, None, :]
        h = s.front[3](s.front[2](h1))
        # front-end frames past the real audio must be the zeros the unpadded run is padded with
        valid = (torch.arange(h.shape[2], device=m.device)[None] < nf[:, None]).float()[:, None, :]
        h = h * valid
        if s.arch != "conformer":
            h = h + s.pos(h)
        h = s.norm_in(h.transpose(1, 2))
        h = h[:, :T] if h.shape[1] >= T else torch.nn.functional.pad(h, (0, 0, 0, T - h.shape[1]))
        # when the front end is shorter than the teacher's frame count, the unpadded run pads with exact
        # zeros after the input norm and attends to them; frames from the real teacher count on are masked
        h = h * (torch.arange(T, device=m.device)[None] < nf[:, None]).float()[..., None]
        if s.arch == "conformer":
            h, _ = s.enc(h, n)
        else:
            mask = torch.arange(T, device=m.device)[None] >= n[:, None]
            h = s.enc(h, src_key_padding_mask=mask)
        h = s.norm_out(h)
        states = {t: s.pred[str(t)](h)[0] for t in sorted(set(self.layers) | {self.ctc_layer})}
        em = torch.log_softmax(self.cj.head(states[self.ctc_layer]), dim=-1)
        return torch.stack([states[t] for t in self.layers]), em

    def _graph1(self, N):
        import torch

        if N not in self.g1:
            m = torch.zeros(1, 80, N // 160, device=self.dev)
            T = self.cj.enc.frames(N)
            n = torch.full((1,), T, device=self.dev, dtype=torch.long)
            nf = torch.full((1,), self._front_frames(N), device=self.dev, dtype=torch.long)
            self._nm = torch.full((1,), N // 160, device=self.dev, dtype=torch.long)   # real mel frames
            with torch.no_grad():
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    for _ in range(2):
                        self._enc(m, n, nf, T)
                torch.cuda.current_stream().wait_stream(self.stream)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = self._enc(m, n, nf, T)
            self.g1[N] = (g, m, (n, nf, self._nm), out)
        return self.g1[N]

    @staticmethod
    def _front_frames(n_samples: int) -> int:
        """Front-end frames of an unpadded input: mel frames N // 160 + 1, first dropped, stride 2."""
        return (n_samples // 160 + 1) // 2

    # -- graph 2: normalisation + pooling + judge --------------------------------------------------
    def _post(self, hs, tmask, a, b, r, ctx, smask):
        import torch

        from twjudge.accent.neural import utt_logits

        keep = (~tmask).float()[None, :, None]                         # (1, T, 1) real frames
        mu = (hs * keep).sum(1, keepdim=True) / keep.sum(1, keepdim=True)
        h = (hs - mu) * keep
        L, T, D = h.shape
        cs = torch.cat([torch.zeros(L, 1, D, device=h.device), h.cumsum(1)], dim=1)
        syl = (cs[:, b] - cs[:, a]) / (b - a).float()[None, :, None]
        rhy = (cs[:, b] - cs[:, r]) / (b - r).float()[None, :, None]
        S = a.shape[0]
        feat = torch.stack([syl, rhy], dim=1)                           # (L, 2, S, D) → per layer [s, r]
        feat = feat.permute(2, 0, 1, 3).reshape(S, L * 2 * D)[None]
        z = torch.stack([net(ctx, feat, smask) for net in self.cj.nets]).mean(0)
        return utt_logits(z, smask)[0], z[0]

    def _graph2(self, T, S):
        import torch

        key = (T, S)
        if key not in self.g2:
            L = len(self.layers)
            hs = torch.zeros(L, T, 768, device=self.dev)
            tmask = torch.zeros(T, dtype=torch.bool, device=self.dev)
            a = torch.zeros(S, dtype=torch.long, device=self.dev)
            b = torch.ones(S, dtype=torch.long, device=self.dev)
            r = torch.zeros(S, dtype=torch.long, device=self.dev)
            ctx = torch.zeros(1, S, 5, device=self.dev)
            smask = torch.zeros(1, S, dtype=torch.bool, device=self.dev)
            args = (hs, tmask, a, b, r, ctx, smask)
            with torch.no_grad():
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    for _ in range(2):
                        self._post(*args)
                torch.cuda.current_stream().wait_stream(self.stream)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = self._post(*args)
            self.g2[key] = (g, args, out)
        return self.g2[key]

    def score(self, wav16k: np.ndarray, text: str):
        import torch

        with torch.no_grad():
            return self._score(wav16k, text)

    def _score(self, wav16k: np.ndarray, text: str):
        import torch

        from twjudge.accent.compact import VOCAB, context
        from twjudge.align.ctc import CTCAligner

        n_samp = len(wav16k)
        N = int(np.ceil(n_samp / 8000) * 8000)
        g, mel_buf, (n, nf, nm), (hs_out, em_out) = self._graph1(N)
        w = torch.from_numpy(wav16k.astype(np.float32)).to(self.dev, non_blocking=True)
        w = (w - w.mean()) / (w.std() + 1e-7)                 # as the student normalises its input
        mel = torch.log(self.cj.enc.mel(w[None]) + 1e-6)[:, :, 1:]
        mel_buf.zero_()
        mel_buf[:, :, :mel.shape[2]] = mel
        T_real = self.cj.enc.frames(n_samp)
        n.fill_(T_real)
        nf.fill_(self._front_frames(n_samp))
        nm.fill_(mel.shape[2])
        g.replay()                                             # asynchronous: the text is read meanwhile
        sy = self.cj.aligner.syllables(text, VOCAB)
        if sy is None:
            return None
        em = em_out[:T_real][None].float().cpu()
        chars = CTCAligner.spans(em, VOCAB, sy, wav16k)
        if chars is None:
            return None
        Tb = hs_out.shape[1]
        S = len(chars)
        Sb = int(np.ceil(S / 8) * 8)
        g2, (hs, tmask, a, b, r, ctx, smask), (zu, zs) = self._graph2(Tb, Sb)
        hs.copy_(hs_out)
        tm = torch.ones(Tb, dtype=torch.bool); tm[:T_real] = False
        tmask.copy_(tm, non_blocking=True)
        ia = [min(T_real - 1, int(c["start"] / HOP_S)) for c in chars]
        ib = [max(min(T_real - 1, int(c["end"] / HOP_S)), ia[k] + 1) for k, c in enumerate(chars)]
        ir = [min(max(min(T_real - 1, int(c["rhyme_start"] / HOP_S)), ia[k]), ib[k] - 1) for k, c in enumerate(chars)]
        pad = Sb - S
        a.copy_(torch.tensor(ia + [0] * pad), non_blocking=True)
        b.copy_(torch.tensor(ib + [1] * pad), non_blocking=True)
        r.copy_(torch.tensor(ir + [0] * pad), non_blocking=True)
        c = torch.zeros(1, Sb, 5); c[0, :S] = torch.tensor(context(chars), dtype=torch.float32)
        ctx.copy_(c, non_blocking=True)
        sm = torch.ones(1, Sb, dtype=torch.bool); sm[0, :S] = False
        smask.copy_(sm, non_blocking=True)
        g2.replay()
        return float(zu), zs[:S].cpu().numpy(), chars
