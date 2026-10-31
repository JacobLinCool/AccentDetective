"""Assemble a compact judge checkpoint (student encoder + CTC alignment head + judge) and verify it.

The checkpoint holds everything `twjudge.accent.compact.CompactJudge` needs; only the student's
prediction heads the judge and the alignment head read are kept. Verification scores items of a
benchmark set end to end from the waveform and compares with the logits the benchmark recorded for
the same judge (which were computed from stored features of the same pipeline).

    python research/scripts/compact_export.py --student students/x.pt --head heads/x_L4.pt --judge runs/.../c_adv0_s0_x.pt \
        --layers 4,8 --out data/judge/models/judge_compact.safetensors --verify-bundle data/bench/bundle \
        --verify-logits runs/.../c_adv0_s0_x_logits.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--judge", required=True, nargs="+", help="one judge, or several (a seed ensemble)")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-bundle")
    ap.add_argument("--verify-logits", help="the judges' logits json files, comma-separated (ensemble: their mean)")
    ap.add_argument("--verify-set", default="cv_test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--judge-prefix", type=int, help="nested judges: the prefix of --layers they read")
    ap.add_argument("--wjudge", nargs="*", default=[], help="judges on the student's predicted Whisper layers (dual)")
    ap.add_argument("--wlayers", default="", help="Whisper layers of the --wjudge features, in their order")
    ap.add_argument("--wprefix", type=int, help="nested --wjudge: the prefix of --wlayers they read")
    ap.add_argument("--syl-means", help="syl_normalize.py --save-means output for the Whisper features")
    a = ap.parse_args()
    import librosa
    import torch

    from twjudge.accent.compact import CompactJudge

    st, hd = (torch.load(p, map_location="cpu") for p in (a.student, a.head))
    jds = [torch.load(p, map_location="cpu") for p in a.judge]
    jd = jds[0]
    layers = [int(x) for x in a.layers.split(",")]
    keep = sorted(set(layers) | {hd["ctc_layer"]})
    wl = [int(x) for x in a.wlayers.split(",") if x]
    wkeep = wl[:a.wprefix] if a.wjudge else []
    cfg = {**st["config"], "targets": keep}
    if wkeep:
        cfg["whisper_targets"] = wkeep
    else:
        cfg.pop("whisper_targets", None)
    state = {k: v for k, v in st["state"].items()
             if not (k.startswith("pred.") and int(k.split(".")[1]) not in keep)
             and not (k.startswith("wpred.") and int(k.split(".")[1]) not in wkeep)}
    ck = {"ssl": "student", "student_config": cfg, "student_state": state, "n_layers": cfg["n_layers"],
          "ctc_layer": hd["ctc_layer"], "ctc_state": hd["ctc_state"], "ctc_arch": hd.get("ctc_arch", "linear"), "judge_layers": layers, "d_in": jd["d_in"],
          "judge_state": jd["state"], "judge_states": [j["state"] for j in jds], "teacher": st.get("teacher"),
          "judge_nested": int(jd.get("nested", 0)), "reports": {"student": st.get("report"),
                                                                                 "head": hd.get("report")}}
    if jd.get("nested"):
        ck["judge_prefix"] = a.judge_prefix or int(jd["nested"])
    if a.wjudge:
        import numpy as np

        wjs = [torch.load(p, map_location="cpu") for p in a.wjudge]
        ck["wjudge_states"] = [j["state"] for j in wjs]
        ck["wjudge"] = {"layers": wl, "prefix": a.wprefix or len(wl), "nested": int(wjs[0].get("nested", 0)),
                        "d_in": wjs[0]["d_in"]}
        m = np.load(a.syl_means)
        assert [100 + l for l in wl] == [int(x) for x in m["layers"]], "syllable means of other layers"
        ck["syl_types"] = [str(t) for t in m["types"]]
        ck["syl_means"] = torch.tensor(m["means"][:, :ck["wjudge"]["prefix"]]).half()
        ck["syl_global"] = torch.tensor(m["glob"][:ck["wjudge"]["prefix"]]).half()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.out.endswith(".safetensors"):
        from twjudge.accent.compact import save_checkpoint

        save_checkpoint(ck, a.out)
    else:
        torch.save(ck, a.out)
    n_par = sum(v.numel() for v in state.values()) + sum(v.numel() for v in hd["ctc_state"].values()) + \
        sum(v.numel() for j in jds for v in j["state"].values()) + sum(v.numel() for st_ in ck.get("wjudge_states", []) for v in st_.values())
    print(f"saved {a.out}: {n_par / 1e6:.2f}M parameters", flush=True)
    if not a.verify_bundle:
        return
    cj = CompactJudge(a.out, device="cpu")
    refs = [json.load(open(f))[a.verify_set] for f in a.verify_logits.split(",")]
    ref = {i: float(np.mean([r[i] for r in refs])) for i in refs[0] if all(i in r for r in refs)}
    rows = [json.loads(l) for l in open(Path(a.verify_bundle) / "sets" / f"{a.verify_set}.jsonl")]
    rows = [r for r in rows if r["item_id"] in ref][:: max(1, len(ref) // a.n)][:a.n]
    z, zr = [], []
    for r in rows:
        w, _ = librosa.load(str(Path(a.verify_bundle) / r["audio"]), sr=16000)
        out = cj.score(w, r["text"])
        if out is not None:
            z.append(out[0]); zr.append(ref[r["item_id"]])
    z, zr = np.array(z), np.array(zr)
    print(f"verify on {len(z)} items: corr {np.corrcoef(z, zr)[0, 1]:.4f}, max |diff| {np.abs(z - zr).max():.3f}", flush=True)


if __name__ == "__main__":
    main()
