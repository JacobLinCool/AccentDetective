"""Features of every benchmark set (bench_bundle.py layout), for the judges and the baselines.

  --kind syllable    syllable-pooled HuBERT (ssl_) and wav2vec 2.0 (w2v_) states from the stored MFA
                     alignment, exactly as `twjudge.accent.ssl_features`; inputs of judges B and C
  --kind utterance   utterance-level embeddings for the baselines, no alignment and no text:
                       ecapa      speaker-verification ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
                       whisper    Whisper-medium encoder, layers 12 and 24, mean over the speech frames
                       hubert_utt / w2v_utt   the judges' front ends, layers 4/8/12, mean and std over
                                  all frames, without the per-utterance normalisation

Outputs <out>/<prefix>_<set>.npz with `ids` in the order of the set file.

    python research/scripts/bench_features.py --bundle /content/bench --kind syllable --out /content/bench/feat
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))

WHISPER = "openai/whisper-medium"
ECAPA = ("speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286")


def load_audio(path: str) -> np.ndarray:
    import librosa

    w, _ = librosa.load(path, sr=16000, mono=True)
    return w.astype(np.float32)


def audio_iter(bundle: Path, rows: list[dict], workers: int = 8):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        yield from zip(rows, ex.map(load_audio, [str(bundle / r["audio"]) for r in rows], chunksize=8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--kind", choices=["syllable", "utterance"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*")
    a = ap.parse_args()
    import torch

    from twjudge.accent.ssl_features import LAYERS, frame_states, load_model, pool

    bundle, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = a.sets or sorted(p.stem for p in (bundle / "sets").glob("*.jsonl"))
    dev = "cuda"
    ssl = {"hubert": load_model("hubert"), "wav2vec2": load_model("wav2vec2")}
    if a.kind == "utterance":
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import WhisperFeatureExtractor, WhisperModel

        ecapa = EncoderClassifier.from_hparams(source=ECAPA[0], revision=ECAPA[1], savedir="/content/ecapa",
                                               run_opts={"device": dev})
        wfe = WhisperFeatureExtractor.from_pretrained(WHISPER)
        whisper = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float16).encoder.to(dev).eval()
    for name in sets:
        rows = [json.loads(l) for l in open(bundle / "sets" / f"{name}.jsonl")]
        prof = [json.loads(l) for l in open(bundle / "profiles" / f"{name}.jsonl")] if a.kind == "syllable" else None
        t0 = time.time()
        acc = {}
        with torch.inference_mode():
            for k, (r, w) in enumerate(audio_iter(bundle, rows)):
                x = torch.tensor(w, device=dev)
                if a.kind == "syllable":
                    chars = prof[k]["chars"]
                    for pre, m in (("ssl", ssl["hubert"]), ("w2v", ssl["wav2vec2"])):
                        acc.setdefault(pre, []).append(pool(frame_states(m, x), chars).half().cpu().numpy())
                    acc.setdefault("lens", []).append(len(chars))
                else:
                    for pre, m in (("hubert_utt", ssl["hubert"]), ("w2v_utt", ssl["wav2vec2"])):
                        xs = (x - x.mean()) / (x.std() + 1e-7)
                        hs = m(xs[None], output_hidden_states=True).hidden_states
                        h = torch.stack([hs[j][0] for j in LAYERS])               # (L, T, 768)
                        acc.setdefault(pre, []).append(torch.cat([h.mean(1).flatten(), h.std(1).flatten()])
                                                       .half().cpu().numpy())
                    acc.setdefault("ecapa", []).append(ecapa.encode_batch(x[None]).squeeze().float().cpu().numpy())
                    seg = w[: 30 * 16000]
                    f = wfe(seg, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
                    hs = whisper(f, output_hidden_states=True).hidden_states
                    n = max(1, min(1500, int(np.ceil(len(seg) / 320))))
                    acc.setdefault("whisper", []).append(torch.cat([hs[12][0, :n].mean(0), hs[24][0, :n].mean(0)])
                                                         .float().cpu().numpy())
                if k % 2000 == 0:
                    print(f"{name} {k}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        ids = np.array([r["item_id"] for r in rows])
        if a.kind == "syllable":
            lens = np.array(acc.pop("lens"))
            for pre, F in acc.items():
                np.savez(out / f"{pre}_{name}.npz", ids=ids, F=np.concatenate(F), lens=lens)
        else:
            for pre, E in acc.items():
                np.savez(out / f"{pre}_{name}.npz", ids=ids, E=np.stack(E))
        print(f"{name}: {len(rows)} utterances, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
