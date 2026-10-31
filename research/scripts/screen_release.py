"""Build the clean Taiwan-accented subset of Common Voice 25 zh-TW as a Hugging Face
dataset, one source file at a time, so the local disk never holds more than one shard.

  data/<split>-*.parquet        the clean clips with their audio, every source column, and the screening scores
  annotations/all.parquet       every clip of the source (no audio): the scores and flags, clean or not

A clip is clean when it has speech (Silero VAD, at least 0.5 s), aligns with its prompt's Mandarin pinyin (MMS CTC,
mean letter probability at least 0.25, or a prompt that cannot be romanised), sounds Taiwanese (Accent Detective Small,
calibrated P(Taiwan) at least 0.5) and comes from a speaker whose clips sound Taiwanese on average (mean log-odds at
least 0).

    uv run --project research --with pyarrow --with fsspec python research/scripts/screen_release.py \\
        --flags data/bench/screen/cv25_zh-TW_flags_p0.5.jsonl --scores data/bench/screen/tw \\
        --repo JacobLinCool/common_voice_25_zh-TW_taiwan_accent --card dataset/README.md
"""
import argparse
import json
import math
import tempfile
from pathlib import Path

SRC = "OpenFormosa/common_voice_25_zh-TW"
FILES = [f"data/other-0000{i}-of-00003.parquet" for i in range(3)] + [f"data/validated-0000{i}-of-00005.parquet" for i in range(5)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flags", required=True, help="screen_analyze.py's flag list (release thresholds)")
    ap.add_argument("--scores", required=True, help="dir of screen_score.py JSONL files (the raw scores)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    flags = {json.loads(l)["clip"]: json.loads(l) for l in open(a.flags)}
    raw = {json.loads(l)["clip"]: json.loads(l) for p in Path(a.scores).glob("*.jsonl") for l in open(p)}
    api, fs = HfApi(), HfFileSystem()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)

    def extra(clip):
        r, f = raw[clip], flags[clip]
        z = r["z"][3] if r["z"] else None
        return {"voiced_seconds": r["voiced"], "alignment_score": r["ctc_tok"],
                "taiwan_logit": z, "taiwan_probability": None if z is None else round(1 / (1 + math.exp(-z)), 4),
                "speaker_flagged": f["speaker_flag"]}

    new_features = {"voiced_seconds": {"dtype": "float64", "_type": "Value"},
                    "alignment_score": {"dtype": "float64", "_type": "Value"},
                    "taiwan_logit": {"dtype": "float64", "_type": "Value"},
                    "taiwan_probability": {"dtype": "float64", "_type": "Value"},
                    "speaker_flagged": {"dtype": "bool", "_type": "Value"}}
    counts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in FILES:
            split = Path(f).stem.split("-")[0]
            with fs.open(f"datasets/{SRC}/{f}") as fh:
                pf = pq.ParquetFile(fh)
                meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])
                meta["info"]["features"].update(new_features)          # the Audio feature stays as the source has it
                parts = []
                for g in range(pf.metadata.num_row_groups):
                    t = pf.read_row_group(g)
                    names = [p.replace("\\", "/").split("/")[-1].replace(".mp3", "") for p in t.column("path").to_pylist()]
                    keep = [k for k, n in enumerate(names) if flags.get(n, {}).get("clean")]
                    if not keep:
                        continue
                    t = t.take(pa.array(keep))
                    ex = [extra(names[k]) for k in keep]
                    for col in ex[0]:
                        t = t.append_column(col, pa.array([e[col] for e in ex]))
                    parts.append(t)
            out = Path(tmp) / f"{Path(f).stem}.parquet"
            table = pa.concat_tables(parts).replace_schema_metadata({b"huggingface": json.dumps(meta).encode()})
            pq.write_table(table, out, row_group_size=1000)
            api.upload_file(path_or_fileobj=str(out), path_in_repo=f"data/{Path(f).name}", repo_id=a.repo, repo_type="dataset")
            counts[f] = table.num_rows
            out.unlink()
            print(f, table.num_rows, "clean clips uploaded", flush=True)
        # every source clip, no audio: the scores and the flags
        rows = []
        for clip, f in flags.items():
            r = raw[clip]
            rows.append({"path": clip + ".mp3", "split": f["split"], **extra(clip), "silent": f["silent"],
                         "not_reading": f["not_reading"], "accent_flagged": f["accent"], "clean": f["clean"]})
        ann = Path(tmp) / "all.parquet"
        pq.write_table(pa.Table.from_pylist(rows), ann)
        api.upload_file(path_or_fileobj=str(ann), path_in_repo="annotations/all.parquet", repo_id=a.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
    print(json.dumps(counts, indent=1), "total", sum(counts.values()))


if __name__ == "__main__":
    main()
