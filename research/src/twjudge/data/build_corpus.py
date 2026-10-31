"""Speaker-disjoint Taiwan / Mainland corpus for the Taiwan-accent judge.

Both sides come from Common Voice, recorded by volunteers on their own devices through the same
web client, so the recording channel is similar and a classifier cannot win on channel alone.

  Taiwan:   Common Voice 25 zh-TW, self-reported birthplace,
            when given, in Taiwan.
  Mainland: Common Voice 21 zh-CN (keeve101/common-voice-21.0-2025-03-14-zh-CN-split), birthplace,
            when given, in a Mainland province (Taiwan, Hong Kong, Macau and overseas excluded).

Splits are by speaker. Taiwan speakers on a reserved list (--reserved: JSONL files with a "speaker"
field) all go to the judge's test split so the judge never trains on them; Mainland speakers follow
the CV split they come from. Each speaker contributes at most PER_SPEAKER clips.

    python -m twjudge.data.build_corpus --out data/judge_corpus
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

from twjudge.data.common import ALLOWED, h, norm_text

PER_SPEAKER = 25
MIN_CHARS, MAX_CHARS = 6, 40
TW_TARGET = {"train": 9000, "dev": 900}
TW_FOREIGN = re.compile(r"中國|中国|香港|澳門|澳门|馬來西亞|马来西亚|新加坡|美國|美国|日本|大陸|大陆")
CN_EXCLUDE = re.compile(r"台湾|臺灣|台灣|香港|澳门|澳門|马来西亚|馬來西亞|新加坡|海外")


def _rank(key: str) -> str:
    return hashlib.sha256(f"judge-v1|{key}".encode()).hexdigest()


def _ok_text(text: str) -> bool:
    n = len(norm_text(text))
    return bool(ALLOWED.match(text)) and MIN_CHARS <= n <= MAX_CHARS


def taiwan(cv_dir: Path, reserved: list[Path]) -> dict[str, list[dict]]:
    held = set()
    for p in reserved:
        held |= {json.loads(l)["speaker"] for l in p.open()}
    by_spk = collections.defaultdict(list)
    for r in csv.DictReader((cv_dir / "validated.tsv").open(), delimiter="\t", quoting=csv.QUOTE_NONE):
        if int(r["up_votes"]) < 2 or int(r["down_votes"]) > 0 or TW_FOREIGN.search(r["accents"]):
            continue
        if not _ok_text(r["sentence"].strip()):
            continue
        by_spk[h(r["client_id"])[:16]].append({"path": r["path"], "text": r["sentence"].strip()})
    splits = {"train": [], "dev": [], "test": []}
    order = sorted(set(by_spk) - held, key=_rank)
    for spk in sorted(held & set(by_spk), key=_rank):
        splits["test"] += _take(by_spk[spk], spk)
    for spk in order:
        name = "dev" if len(splits["dev"]) < TW_TARGET["dev"] else "train"
        if name == "train" and len(splits["train"]) >= TW_TARGET["train"]:
            break
        splits[name] += _take(by_spk[spk], spk)
    return splits


def mainland(cn_dir: Path) -> dict[str, list[dict]]:
    splits = {}
    for name in ("train", "dev", "test"):
        by_spk = collections.defaultdict(list)
        for f in sorted((cn_dir / "data").glob(f"{name}-*.parquet")):
            t = pq.read_table(f, columns=["client_id", "path", "sentence", "accents", "up_votes", "down_votes"])
            for r in t.to_pylist():
                acc = r["accents"] or ""
                if r["down_votes"] > 0 or CN_EXCLUDE.search(acc) or (acc and "出生地" not in acc):
                    continue
                text = unicodedata.normalize("NFKC", r["sentence"]).strip()
                if not _ok_text(text):
                    continue
                by_spk[hashlib.sha256(r["client_id"].encode()).hexdigest()[:16]].append(
                    {"path": r["path"], "text": text})
        splits[name] = [c for spk in sorted(by_spk, key=_rank) for c in _take(by_spk[spk], spk)]
    return splits


def _take(clips: list[dict], spk: str) -> list[dict]:
    return [{**c, "speaker": spk} for c in sorted(clips, key=lambda c: _rank(c["path"]))[:PER_SPEAKER]]


def extract(parquets: list[Path], wanted: set[str], dest: Path) -> set[str]:
    dest.mkdir(parents=True, exist_ok=True)
    found = set()
    for f in parquets:
        for batch in pq.ParquetFile(f).iter_batches(columns=["path", "audio"], batch_size=256):
            for path, audio in zip(batch.column("path").to_pylist(), batch.column("audio").to_pylist()):
                name = Path(path).name
                if name in wanted and name not in found:
                    (dest / name).write_bytes(audio["bytes"])
                    found.add(name)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tw-dir", default="data/cv25")
    ap.add_argument("--cn-dir", default="data/cv21cn")
    ap.add_argument("--reserved", nargs="*", default=[], help="JSONL files whose speakers go to the test split")
    ap.add_argument("--out", default="data/judge_corpus")
    a = ap.parse_args()
    out = Path(a.out)
    sets = {"tw": taiwan(Path(a.tw_dir), [Path(p) for p in a.reserved]), "cn": mainland(Path(a.cn_dir))}
    # the Mainland test split is far larger than needed; keep it about the size of the Taiwan one
    sets["cn"]["test"] = sets["cn"]["test"][:len(sets["tw"]["test"]) * 2]
    sets["cn"]["dev"] = sets["cn"]["dev"][:TW_TARGET["dev"]]
    sets["cn"]["train"] = sets["cn"]["train"][:TW_TARGET["train"]]
    found = {"tw": extract(sorted(Path(a.tw_dir, "parquet").glob("validated-*.parquet")),
                           {Path(c["path"]).name for s in sets["tw"].values() for c in s}, out / "clips/tw"),
             "cn": extract(sorted(Path(a.cn_dir, "data").glob("*.parquet")),
                           {Path(c["path"]).name for s in sets["cn"].values() for c in s}, out / "clips/cn")}
    report = {}
    for split in ("train", "dev", "test"):
        with (out / f"{split}.jsonl").open("w") as f:
            for lab in ("tw", "cn"):
                rows = [c for c in sets[lab][split] if Path(c["path"]).name in found[lab]]
                for c in rows:
                    name = Path(c["path"]).name
                    f.write(json.dumps({"item_id": f"{lab}_{Path(name).stem}", "label": lab, "split": split,
                                        "speaker": f"{lab}{c['speaker']}", "text": c["text"],
                                        "wav": str((out / "clips" / lab / name).resolve())},
                                       ensure_ascii=False) + "\n")
                report[f"{split}/{lab}"] = {"clips": len(rows), "speakers": len({c["speaker"] for c in rows})}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
