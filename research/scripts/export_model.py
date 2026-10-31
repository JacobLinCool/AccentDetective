"""Export the trained trunk and its readouts as one Accent Detective checkpoint: model.safetensors.

Tensors: trunk.* (the nested trunk without its CTC head, which only trained it), asp.<k>.* and text.<k>.* (the two
readouts of exit k). Metadata "accent_detective" (JSON): the trunk config, the syllable-type vocabulary of the text
readout, and per size and mode the positive-slope Platt transform fitted on cv_dev scores (as r10_analyze.py).

    python research/scripts/export_model.py --student r14_H_s0/student.pt --readouts r14_H_s0 --vocab vocab.json \\
        --scores data/bench/released/scores --out dist/model.safetensors
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/src")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--readouts", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--scores", required=True, help="dir with N<k>{a,x}.json holding cv_dev scores")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from safetensors.torch import save_file
    from sklearn.linear_model import LogisticRegression

    ck = torch.load(a.student, map_location="cpu", weights_only=False)
    cfg = {k: ck["config"][k] for k in ("exits", "targets", "d", "heads", "ff")}
    tensors = {f"trunk.{k}": v.float().contiguous() for k, v in ck["state"].items() if not k.startswith("ctc.")}
    n_types = None
    for k in range(1, len(cfg["exits"]) + 1):
        rd = torch.load(Path(a.readouts) / f"readout_N{k}.pt", map_location="cpu", weights_only=False)
        assert list(rd["exits"]) == list(cfg["exits"]) and rd["exit"] == k
        n_types = rd["n_types"]
        tensors.update({f"asp.{k}.{n}": v.float().contiguous() for n, v in rd["asp"].items()})
        tensors.update({f"text.{k}.{n}": v.float().contiguous() for n, v in rd["text"].items()})
    vocab = json.load(open(a.vocab))
    assert max(vocab.values()) + 1 == n_types
    platt = {}
    for k in range(1, len(cfg["exits"]) + 1):
        for mode, tag in (("audio", "a"), ("text", "x")):
            S = json.load(open(Path(a.scores) / f"N{k}{tag}.json"))["cv_dev"]
            ids = sorted(i for i, v in S.items() if v is not None)
            y = np.array([i.startswith("tw") for i in ids]).astype(int)
            z = np.array([S[i][0] for i in ids])
            m = LogisticRegression(C=1e6).fit(z[:, None], y)
            s_, b_ = float(m.coef_[0, 0]), float(m.intercept_[0])
            platt.setdefault(f"N{k}", {})[mode] = [s_, b_] if s_ > 0 else [1.0, 0.0]
    meta = {"format": "nested-asp-text", "config": cfg, "n_types": n_types, "vocab": vocab, "platt": platt,
            "sizes": ["N1", "N2", "N3", "N4"]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={"accent_detective": json.dumps(meta, ensure_ascii=False)})
    print(f"{out}: {sum(v.numel() for v in tensors.values()) / 1e6:.1f} M parameters, {out.stat().st_size / 2 ** 20:.0f} MB")
    print(json.dumps(platt))


if __name__ == "__main__":
    main()
