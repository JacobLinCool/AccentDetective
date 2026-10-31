"""Automatic Taiwan-accent feature profile by forced alignment, without listeners.

Taiwan Mandarin differs from Mainland Mandarin in well-described phonetic
features beyond word readings. For every syllable that carries such a
feature, the pronunciation dictionary offers the standard and the Taiwan-typical
realisation, and the aligner chooses:

  retroflex   zh ch sh  → z c s   (de-retroflexion)
  velar_nasal -ing -eng → -in -en (coda merger)

Each character is its own token so variants do not multiply across a word; the
reading of each character comes from the lexicon (Taiwan reading when the
word has one). The profile is the fraction of opportunities realised the Taiwan
way, reported next to the same profile of real Taiwan speakers (CV), which is
the reference the acoustic layer should move towards. The per-character time
spans of the same alignment are also written out; the transport layer uses
them as its phonetic conditions.

    python -m twjudge.reading.accent_profile --items items.jsonl --out out_dir
"""

from __future__ import annotations

import argparse
import hashlib
import collections
import json
import os
import shutil
import subprocess
from pathlib import Path

from twjudge.reading.reading_judge import MFA_BIN, MFA_HOME, ReadingJudge
from twjudge.lexicon.readings import TaiwanReadings

CUES = {
    "retroflex": [("zh", "z"), ("ch", "c"), ("sh", "s")],
    "velar_nasal": [("ing", "in"), ("eng", "en")],
}


def cue_alternatives(syl: str) -> dict[str, str]:
    """Taiwan-typical alternatives of a numbered syllable, keyed by cue name."""
    body, tone = (syl[:-1], syl[-1]) if syl[-1].isdigit() else (syl, "5")
    out = {}
    for a, b in CUES["retroflex"]:
        if body.startswith(a):
            out["retroflex"] = b + body[len(a):] + tone
    for a, b in CUES["velar_nasal"]:
        if body.endswith(a) and not body.endswith("ong"):
            out["velar_nasal"] = body[: -len(a)] + b + tone
    return out


class AccentProfiler(ReadingJudge):
    def char_readings(self, text: str) -> list[tuple[str, str]]:
        out = []
        for _, piece in self.rw.segment(text):
            if not all("㐀" <= c <= "鿿" for c in piece):
                continue
            reading = self.rw.tw.get(piece) if len(piece) > 1 else None
            cn = self.rw.cn_reading(piece)
            reading = reading if reading and len(reading) == len(piece) else cn
            out += list(zip(piece, reading))
        return out

    def run(self, items: list[dict], out: Path, jobs: int = 4) -> list[dict]:
        # MFA keys its temporary directory on the corpus folder name: make it unique per run
        tag = hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:10]
        corpus, aligned = out / f"corpus_{tag}", out / "aligned"
        shutil.rmtree(corpus, ignore_errors=True)
        corpus.mkdir(parents=True)
        lexicon, meta = {}, {}
        for it in items:
            toks, ok = [], True
            for k, (ch, syl) in enumerate(self.char_readings(it["text"])):
                base = self.syl.get(syl)
                if base is None:
                    ok = False
                    break
                prons = {"std": base}
                for cue, alt in cue_alternatives(syl).items():
                    p = self.syl.get(alt)
                    if p and p != base:
                        prons[cue] = p
                tok = f"c{k}_{ch}{syl}"
                toks.append({"token": tok, "syl": syl, "prons": prons})
                lexicon[tok] = sorted(set(prons.values()))
            if not ok or not toks:
                meta[it["item_id"]] = None
                continue
            spk = corpus / it.get("speaker", "s0")
            spk.mkdir(exist_ok=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", it["wav"], "-ac", "1", "-ar", "16000",
                            str(spk / f"{it['item_id']}.wav")], check=True)
            (spk / f"{it['item_id']}.lab").write_text(" ".join(t["token"] for t in toks))
            meta[it["item_id"]] = toks
        dict_path = out / "profile.dict"
        with dict_path.open("w", encoding="utf-8") as f:
            for tok, prons in sorted(lexicon.items()):
                for p in prons:
                    f.write(f"{tok}\t{p}\n")
        shutil.rmtree(aligned, ignore_errors=True)
        cmd = [str(MFA_BIN), "align", str(corpus), str(dict_path), str(MFA_HOME / "acoustic/mandarin_mfa.zip"),
               str(aligned), "--clean", "--overwrite", "-j", str(jobs), "--beam", "100", "--retry_beam", "400",
               "--output_format", "json", "--no_tokenization", "--quiet"]
        env = {**os.environ, "PATH": f"{MFA_BIN.parent}:{os.environ.get('PATH', '')}"}
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (out / "mfa.log").write_text(res.stdout + res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"mfa align failed; see {out / 'mfa.log'}")
        results = []
        for it in items:
            toks = meta[it["item_id"]]
            jpath = aligned / it.get("speaker", "s0") / f"{it['item_id']}.json"
            if toks is None or not jpath.exists():
                results.append({"item_id": it["item_id"], "status": "skipped", "cues": []})
                continue
            tiers = json.loads(jpath.read_text())["tiers"]
            spans = {e[2]: e for e in tiers["words"]["entries"]}
            phones = tiers["phones"]["entries"]
            cues, chars = [], []
            for t in toks:
                if t["token"] not in spans:
                    continue
                s = spans[t["token"]]
                ph = [p for p in phones if p[0] >= s[0] - 1e-4 and p[1] <= s[1] + 1e-4]
                got = " ".join(p[2] for p in ph)
                toned = next((p for p in ph if any(m in p[2] for m in "˥˧˨˩˦")), ph[-1] if ph else None)
                chars.append({"char": t["token"].split("_", 1)[1][0], "syl": t["syl"], "start": s[0], "end": s[1],
                              "rhyme_start": toned[0] if toned else s[0]})
                if len(t["prons"]) < 2:
                    continue
                chosen = next((k for k, v in t["prons"].items() if v == got), "other")
                for cue in t["prons"]:
                    if cue != "std":
                        cues.append({"syl": t["syl"], "cue": cue, "taiwan_variant": chosen == cue})
            results.append({"item_id": it["item_id"], "status": "ok", "cues": cues, "chars": chars})
        return results


def summarize(results: list[dict]) -> dict:
    c = collections.defaultdict(lambda: [0, 0])
    for r in results:
        for q in r["cues"]:
            c[q["cue"]][0] += int(q["taiwan_variant"])
            c[q["cue"]][1] += 1
    return {cue: {"taiwan_rate": round(a / n, 3), "opportunities": n} for cue, (a, n) in sorted(c.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon", default="data/lexicon/concised.json.xz")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(a.items)]
    res = AccentProfiler(TaiwanReadings(a.lexicon)).run(items, out, a.jobs)
    with (out / "profile.jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarize(res)
    (out / "profile_summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s))


if __name__ == "__main__":
    main()
