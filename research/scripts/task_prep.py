"""Preparation of the accent-task corpus and the sealed set, on Colab. Three steps, each its own flag:

  --pool    the nested-distillation pool, packed: the bundle's cv_train clips (as nested_distill.py loads them) + AISHELL-1
            60 h + THCHS-30 25 h (distill_data.extra_files order and decoding), int16 in audio.npy with offsets.npy and
            meta.jsonl (source tw_cv / cn_cv / aishell1 / thchs30, speaker, id)
  --sets    new Common Voice speakers, never in any bundle set: CV25 zh-TW (validated, other) and CV21 zh-CN (train split).
            Each speaker goes to the sealed evaluation set (sha256("r10|" + speaker) % 4 == 0, ≤ 4 clips) or to the task
            pool (≤ 8 clips); quotas per class. Writes <bundle>/sets/r10_sealed.jsonl, r10_task.jsonl, audio/r10/
  --align   forced alignments (the nested student's CTC head) of the task pool and of the same-content pairs of
            cv_train (identical toneless pinyin, Taiwan × Mainland): <out>/align.part<k>.json, then --merge-align;
            pairs in <out>/pairs.json

    python research/scripts/task_prep.py --pool --bundle /content/bench --out /content/task_pool
    python research/scripts/task_prep.py --sets --bundle /content/bench
    python research/scripts/task_prep.py --align --bundle /content/bench --nested student.pt --detective model.safetensors \\
        --out /content/bench/r10 --shard 0/8
"""
import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "research/scripts"), str(ROOT / "research/src")]

TW, CN = "OpenFormosa/common_voice_25_zh-TW", "keeve101/common-voice-21.0-2025-03-14-zh-CN-split"
QUOTA = {"sealed": 1500, "task": 6000}          # clips per class
PER_SPK = {"sealed": 4, "task": 8}


def _load(path):
    import librosa

    try:
        return (librosa.load(path, sr=16000, mono=True)[0] * 32767).astype(np.int16)
    except Exception:
        return None


def build_pool(B: Path, out: Path, work: Path):
    from distill_data import decode_file, extra_files

    rows = [json.loads(l) for l in open(B / "sets" / "cv_train.jsonl")]
    out.mkdir(parents=True, exist_ok=True)
    pieces, meta = [], []
    with ProcessPoolExecutor(10) as ex:
        for r, c in zip(rows, ex.map(_load, [str(B / r["audio"]) for r in rows], chunksize=64)):
            if c is not None and len(c) >= 16000:
                pieces.append(c)
                meta.append({"source": "tw_cv" if r["item_id"].startswith("tw") else "cn_cv", "speaker": r["speaker"], "id": r["item_id"]})
        print(f"cv_train {len(pieces)} clips", flush=True)
        for name, hours in (("aishell1", 60), ("thchs30", 25)):
            files, got = [str(f) for f in extra_files(name, work)], 0.0
            for k in range(0, len(files), 4000):
                for f, w in zip(files[k:k + 4000], ex.map(decode_file, files[k:k + 4000], chunksize=64)):
                    if w is not None and got < hours * 3600:
                        pieces.append(w); got += len(w) / 16000
                        spk = Path(f).parent.name if name == "aishell1" else Path(f).stem.split("_")[0]
                        meta.append({"source": name, "speaker": f"{name}_{spk}", "id": str(Path(f).relative_to(work))})
                if got >= hours * 3600:
                    break
            print(f"{name}: {got / 3600:.1f} h", flush=True)
    offsets = np.r_[0, np.cumsum([len(p) for p in pieces])].astype(np.int64)
    np.save(out / "audio.npy", np.concatenate(pieces))
    np.save(out / "offsets.npy", offsets)
    with open(out / "meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"pool: {len(meta)} clips, {offsets[-1] / 16000 / 3600:.1f} h", flush=True)


def build_sets(B: Path, extra=False):
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    from distill_data import decode
    from twjudge.data.common import h

    bench_spk, bench_clip = set(), set()
    for p in (B / "sets").glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            bench_spk.add(r.get("speaker"))
            bench_clip.add(r["item_id"].split("_", 1)[-1].replace("\\", "/").split("/")[-1].replace(".mp3", ""))
    (B / "audio" / "r10").mkdir(parents=True, exist_ok=True)
    rows = {"sealed": [], "task": []}
    seen = set()
    api = HfApi()
    # first pass: the training splits; extra pass (--sets-extra): the other splits, minus every speaker of the first
    # pass and of the bundle, with larger per-speaker caps (few new speakers remain), written as r10_<role>_x
    # extra pass (--extra): Mainland only, which has few new speakers: every CV21 zh-CN split again, more clips per
    # speaker (counting the first pass), skipping the first pass's clips; written as r10_<role>_x
    splits = ((("cn", CN, ("data/train-", "data/dev-", "data/test-")),) if extra
              else (("tw", TW, ("data/validated-", "data/other-")), ("cn", CN, ("data/train-",))))
    per_spk = {"sealed": 12, "task": 24} if extra else PER_SPK
    first = {}
    if extra:
        for role in ("sealed", "task"):
            for l in open(B / "sets" / f"r10_{role}.jsonl"):
                r = json.loads(l)
                bench_spk.discard(r["speaker"])
                first[r["speaker"]] = first.get(r["speaker"], 0) + 1
                bench_clip.add(r["item_id"].split("_r10_", 1)[1])
    for label, repo, pats in splits:
        files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(pats) and f.endswith(".parquet"))
        n = {"sealed": 0, "task": 0}
        per = dict(first)
        for fname in files:
            if all(n[k] >= QUOTA[k] for k in n):
                break
            path = hf_hub_download(repo, fname, repo_type="dataset")
            cols = pq.read_schema(path).names
            tcol = "sentence" if "sentence" in cols else "text"
            t = pq.read_table(path, columns=["client_id", "path", tcol, "audio"])
            cid, paths, texts = t.column("client_id").to_pylist(), t.column("path").to_pylist(), t.column(tcol).to_pylist()
            pick = []
            for k, (c, p, txt) in enumerate(zip(cid, paths, texts)):
                spk = f"tw{h(c)[:16]}" if label == "tw" else f"cn{hashlib.sha256(c.encode()).hexdigest()[:16]}"
                name = p.replace("\\", "/").split("/")[-1].replace(".mp3", "")
                if spk in bench_spk or name in bench_clip or len(re.sub(r"[^㐀-鿿]", "", txt or "")) < 4:
                    continue
                role = "sealed" if int(hashlib.sha256(f"r10|{spk}".encode()).hexdigest(), 16) % 4 == 0 else "task"
                if n[role] >= QUOTA[role] or per.get(spk, 0) >= per_spk[role]:
                    continue
                per[spk] = per.get(spk, 0) + 1
                n[role] += 1
                pick.append((k, spk, name, txt, role))
            audio = t.column("audio")
            with ProcessPoolExecutor(10) as ex:
                for (k, spk, name, txt, role), w in zip(pick, ex.map(decode, [audio[k].as_py()["bytes"] for k, *_ in pick], chunksize=32)):
                    if w is None:
                        continue
                    rid = f"{label}_r10_{name}"
                    if rid in seen:
                        continue
                    seen.add(rid)
                    rel = f"audio/r10/{rid}.flac"
                    sf.write(B / rel, w.astype(np.float32) / 32768, 16000)
                    rows[role].append({"item_id": rid, "audio": rel, "text": txt, "speaker": spk, "label": None})
            print(label, fname, n, flush=True)
    for role, rs in rows.items():
        with open(B / "sets" / f"r10_{role}{'_x' if extra else ''}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(role, len(rs), "clips,", len({r["speaker"] for r in rs}), "speakers", flush=True)


def same_content_pairs(B: Path):
    """Taiwan × Mainland cv_train clips with identical toneless pinyin: each Taiwan clip with one Mainland clip of it."""
    from pypinyin import lazy_pinyin

    rows = [json.loads(l) for l in open(B / "sets" / "cv_train.jsonl")]
    key = lambda t: " ".join(lazy_pinyin(re.sub(r"[^㐀-鿿]", "", t)))
    tw, cn = {}, {}
    for r in rows:
        (tw if r["item_id"].startswith("tw") else cn).setdefault(key(r["text"]), []).append(r["item_id"])
    rng = np.random.default_rng(0)
    return [[t, cn[k][rng.integers(len(cn[k]))]] for k in sorted(set(tw) & set(cn)) for t in tw[k]]


def align(B: Path, out: Path, nested: str, detective: str, shard: str, extra=False):
    import librosa
    import torch

    from twjudge.accent.compact import VOCAB, CompactJudge, context
    from twjudge.accent.nested import NestedStudent
    from twjudge.lexicon.readings import TaiwanReadings

    k, n = map(int, shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(nested, map_location="cpu", weights_only=False)
    net = NestedStudent(**{x: ck["config"][x] for x in ("exits", "targets", "d", "heads", "ff")})
    net.load_state_dict(ck["state"])
    net = net.to(dev).eval()
    aligner = CompactJudge(detective, readings=TaiwanReadings(ROOT / "data/lexicon/concised.json.xz"), device="cpu", mode="fast").aligner
    out.mkdir(parents=True, exist_ok=True)
    if extra:
        rows = [json.loads(l) for l in open(B / "sets" / "r10_task_x.jsonl")]
    else:
        pairs = same_content_pairs(B)
        if k == 0:
            json.dump(pairs, open(out / "pairs.json", "w"))
        want = {i for p in pairs for i in p}
        rows = [json.loads(l) for l in open(B / "sets" / "r10_task.jsonl")]
        rows += [r for r in (json.loads(l) for l in open(B / "sets" / "cv_train.jsonl")) if r["item_id"] in want]
    res = {}
    for r in rows[k::n]:
        w = librosa.load(str(B / r["audio"]), sr=16000, mono=True)[0]
        with torch.inference_mode():
            em = net.emissions(net(torch.tensor(w, device=dev)[None], upto=1)).float().cpu()
        chars = aligner.align(w, r["text"], emissions=(em, VOCAB))
        if chars is None or len(chars) < 2:
            continue
        res[r["item_id"]] = {"chars": [{"start": c["start"], "end": c["end"], "rhyme_start": c["rhyme_start"], "syl": c["syl"]}
                                       for c in chars], "ctx": context(chars), "n_samples": len(w)}
    json.dump(res, open(out / f"align{'_x' if extra else ''}.part{k}.json", "w"))
    print(f"aligned {len(res)} of {len(rows[k::n])}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out")
    ap.add_argument("--work", default="/content/distill_src")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--sets", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--merge-align", action="store_true")
    ap.add_argument("--extra", action="store_true", help="with --sets / --align / --merge-align: the extra pass (*_x)")
    ap.add_argument("--merge-extra", action="store_true", help="append the extra pass to the sets, alignments, teacher scores")
    ap.add_argument("--nested")
    ap.add_argument("--detective")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    B = Path(a.bundle)
    if a.pool:
        build_pool(B, Path(a.out), Path(a.work))
    if a.sets:
        build_sets(B, a.extra)
    if a.align:
        align(B, Path(a.out), a.nested, a.detective, a.shard, a.extra)
    if a.merge_align:
        x = "_x" if a.extra else ""
        res = {}
        for p in sorted(Path(a.out).glob(f"align{x}.part*.json")):
            res.update(json.load(open(p)))
            p.unlink()
        json.dump(res, open(Path(a.out) / f"align{x}.json", "w"))
        print(f"merged {len(res)} alignments", flush=True)
    if a.merge_extra:
        R = Path(a.out)
        for role in ("sealed", "task"):
            with open(B / "sets" / f"r10_{role}.jsonl", "a") as f:
                f.write(open(B / "sets" / f"r10_{role}_x.jsonl").read())
        al = json.load(open(R / "align.json")); al.update(json.load(open(R / "align_x.json")))
        json.dump(al, open(R / "align.json", "w"))
        for p in sorted((R / "teacher_x").glob("*.json")):
            q = R / "teacher" / p.name
            base, add = json.load(open(q)), json.load(open(p))
            for s_ in ("task", "sealed"):
                base.setdefault(f"r10_{s_}", {}).update(add.get(f"r10_{s_}_x", {}))
            json.dump(base, open(q, "w"))
        n = {r: sum(1 for _ in open(B / "sets" / f"r10_{r}.jsonl")) for r in ("sealed", "task")}
        print(f"merged extra: {n}, {len(al)} alignments", flush=True)


if __name__ == "__main__":
    main()
