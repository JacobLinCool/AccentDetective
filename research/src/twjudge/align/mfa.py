"""Reading-neutral forced alignment for the Taiwan-accent judge.

The judge compares Taiwan and Mainland speakers reading different sentences, so the alignment
must not presuppose either accent. Every character is its own token and the dictionary offers
both its Mainland reading (pypinyin) and its Taiwan reading (教育部《國語辭典簡編本》 for
multi-character words), the Mainland neutral tone where CC-CEDICT marks one (pypinyin does not),
and the Taiwan-typical segmental variants (de-retroflexion, -ing/-eng → -in/-en); the aligner
picks whichever fits the audio. Each character keeps its full-tone Mainland reading as the *canonical* label, so a feature that depends on the tone category means
the same thing in both corpora, and the realised variant is recorded separately.

Simplified input is converted character by character (OpenCC s2t, not phrase-level s2twp, which
would replace words and change readings) only to look up Taiwan readings.

    python -m twjudge.align.mfa --items items.jsonl --out out_dir [--jobs 16]
    # items.jsonl: {"item_id", "wav", "text", "speaker"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from twjudge.reading.accent_profile import cue_alternatives
from twjudge.reading.reading_judge import MFA_BIN, MFA_HOME, ReadingJudge
from twjudge.lexicon.cn_neutral import neutral_positions
from twjudge.lexicon.readings import TaiwanReadings


class JudgeAligner(ReadingJudge):
    def __init__(self, rw: TaiwanReadings):
        super().__init__(rw)
        import opencc

        self._s2t = opencc.OpenCC("s2t")

    def char_readings(self, text: str) -> list[tuple[str, str, str, bool]]:
        """(character, Mainland reading, Taiwan reading, Mainland neutral?) for every Han character."""
        trad = self._s2t.convert(text)
        if len(trad) != len(text):
            trad = text
        out = []
        for off, piece in self.rw.segment(trad):
            if not all("㐀" <= c <= "鿿" for c in piece):
                continue
            cn = self.rw.cn_reading(piece)
            tw = self.rw.tw.get(piece) if len(piece) > 1 else None
            tw = tw if tw and len(tw) == len(piece) else cn
            neu = neutral_positions(piece, mode="any") if len(piece) > 1 else frozenset()
            out += [(text[off + k] if off + k < len(text) else c, a, b, k in neu)
                    for k, (c, a, b) in enumerate(zip(piece, cn, tw))]
        return out

    def run(self, items: list[dict], out: Path, jobs: int = 8) -> list[dict]:
        tag = hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:10]
        corpus, aligned = out / f"corpus_{tag}", out / "aligned"
        shutil.rmtree(corpus, ignore_errors=True)
        corpus.mkdir(parents=True)
        lexicon, meta = {}, {}
        for it in items:
            toks, ok = [], True
            for k, (ch, cn, tw, neu) in enumerate(self.char_readings(it["text"])):
                prons = {}
                cands = [("cn", cn), ("tw", tw)] + ([("cn5", cn[:-1] + "5")] if neu else [])
                for name, syl in cands:
                    p = self.syl.get(syl)
                    if p is None:
                        continue
                    prons.setdefault(p, name)
                    for cue, alt in cue_alternatives(syl).items():
                        q = self.syl.get(alt)
                        if q:
                            prons.setdefault(q, f"{name}+{cue}")
                if not prons:
                    ok = False
                    break
                tok = f"c{k}_{ch}"
                toks.append({"token": tok, "char": ch, "cn": cn, "tw": tw, "neutral": neu, "prons": prons})
                lexicon.setdefault(tok, set()).update(prons)
            if not ok or not toks:
                meta[it["item_id"]] = None
                continue
            spk = corpus / it.get("speaker", "s0")
            spk.mkdir(exist_ok=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", it["wav"], "-ac", "1", "-ar", "16000",
                            str(spk / f"{it['item_id']}.wav")], check=True)
            (spk / f"{it['item_id']}.lab").write_text(" ".join(t["token"] for t in toks))
            meta[it["item_id"]] = toks
        dict_path = out / "judge.dict"
        with dict_path.open("w", encoding="utf-8") as f:
            for tok, prons in sorted(lexicon.items()):
                for p in sorted(prons):
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
                results.append({"item_id": it["item_id"], "status": "skipped"})
                continue
            tiers = json.loads(jpath.read_text())["tiers"]
            spans = {e[2]: e for e in tiers["words"]["entries"]}
            phones = tiers["phones"]["entries"]
            chars = []
            for t in toks:
                if t["token"] not in spans:
                    continue
                s = spans[t["token"]]
                ph = [p for p in phones if p[0] >= s[0] - 1e-4 and p[1] <= s[1] + 1e-4]
                got = " ".join(p[2] for p in ph)
                toned = next((p for p in ph if any(m in p[2] for m in "˥˧˨˩˦")), ph[-1] if ph else None)
                chars.append({"char": t["char"], "syl": t["cn"], "tw": t["tw"], "cn_neutral": t["neutral"],
                              "start": s[0], "end": s[1],
                              "rhyme_start": toned[0] if toned else s[0],
                              "variant": t["prons"].get(got, "other"), "n_variants": len(t["prons"])})
            results.append({"item_id": it["item_id"], "status": "ok", "chars": chars})
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon", default="data/lexicon/concised.json.xz")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(a.items)]
    res = JudgeAligner(TaiwanReadings(a.lexicon)).run(items, out, a.jobs)
    with (out / "profile.jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(sum(r["status"] == "ok" for r in res), "aligned of", len(res))


if __name__ == "__main__":
    main()
