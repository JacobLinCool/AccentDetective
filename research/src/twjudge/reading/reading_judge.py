"""Automatic Taiwan-vs-Mainland reading judge by forced alignment.

For every target word whose Taiwan reading differs from the Mainland one, the
pronunciation dictionary offers both readings and the Montreal Forced Aligner
(``mandarin_mfa`` 3.0, tonal phones, pitch features) picks the one that fits
the audio better. All other words get their readings as well, so the
alignment of context is not forced either way.

MFA's own Taiwan dictionary is learned from data and lists e.g. 企業 as qǐ yè,
so it is not used as a reference: readings come from the lexicon
(``twjudge.lexicon.readings``) and are converted to MFA phones with
a syllable table learned from the monosyllabic entries of MFA's Mainland
dictionary.

    python -m twjudge.reading.reading_judge --items items.jsonl --out out_dir
    # items.jsonl: {"item_id", "wav", "text"}  (text = original Taiwan text)
"""

from __future__ import annotations

import argparse
import hashlib
import collections
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pypinyin.constants import PINYIN_DICT

from twjudge.lexicon.readings import (TaiwanReadings, neutral_only,
                                              same_reading, tone3)

MFA_HOME = Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents/MFA")) / "pretrained_models"
MFA_BIN = Path(os.environ.get("MFA_BIN", Path.home() / ".local/share/mamba/envs/mfa/bin/mfa"))
TONE_MARKS = "˥˧˨˩˦"
TONE_OF = {"1": "˥", "2": "˧˥", "3": "˨˩˦", "4": "˥˩", "5": ""}


def retone(phones: str, tone: str) -> str:
    """Replace the tone on a syllable's toned phone, keeping its segments unchanged."""
    out = []
    for ph in phones.split():
        if any(m in ph for m in TONE_MARKS):
            ph = "".join(ch for ch in ph if ch not in TONE_MARKS) + TONE_OF[tone]
        out.append(ph)
    return " ".join(out)


def _split(syl: str) -> tuple[str, str, str]:
    """Numbered syllable → (initial, final, tone) with pypinyin's strict finals (ju → j + v)."""
    from pypinyin.contrib.tone_convert import to_finals, to_initials

    body, tone = (syl[:-1], syl[-1]) if syl[-1].isdigit() else (syl, "5")
    return to_initials(body, strict=True), to_finals(body, strict=True, v_to_u=False), tone


def syllable_table(china_dict: Path) -> "SyllablePhones":
    """Numbered pinyin syllable → MFA phone string.

    Learned from MFA's Mainland dictionary: syllables of single-reading
    characters are taken directly; any other syllable is composed from the
    learned initial and final(+tone) pieces. Neutral tone uses the final with
    tone marks removed.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in china_dict.open(encoding="utf-8"):
        word, prob, *_, phones = line.rstrip("\n").split("\t")
        if len(word) != 1:
            continue
        readings = PINYIN_DICT.get(ord(word))
        if not readings or "," in readings:
            continue
        votes[tone3(readings)][phones] += float(prob)
    direct = {syl: c.most_common(1)[0][0] for syl, c in votes.items()}
    initials: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    finals: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for syl, phones in direct.items():
        ini, fin, tone = _split(syl)
        ph = phones.split()
        # the initial is everything before the first phone carrying a tone or a vowel quality
        k = 0
        while k < len(ph) and not any(m in ph[k] for m in TONE_MARKS) and ph[k] not in ("j", "w", "ɥ"):
            k += 1
        k = min(k, len(ph) - 1)
        if ini:
            initials[ini][" ".join(ph[:k])] += 1
            finals[(fin, tone)][" ".join(ph[k:])] += 1
        else:
            finals[(fin, tone)][" ".join(ph)] += 1
    return SyllablePhones(direct,
                          {i: c.most_common(1)[0][0] for i, c in initials.items()},
                          {f: c.most_common(1)[0][0] for f, c in finals.items()})


class SyllablePhones:
    def __init__(self, direct, initials, finals):
        self.direct, self.initials, self.finals = direct, initials, finals

    def __len__(self):
        return len(self.direct)

    def get(self, syl: str) -> str | None:
        if syl in self.direct:
            return self.direct[syl]
        try:
            ini, fin, tone = _split(syl)
        except Exception:
            return None
        rest = self.finals.get((fin, tone))
        if rest is None and tone == "5":
            for t in "1234":
                if (fin, t) in self.finals:
                    rest = "".join(ch for ch in self.finals[(fin, t)] if ch not in TONE_MARKS)
                    break
        if rest is None:
            return None
        if not ini:
            return rest
        head = self.initials.get(ini)
        if head is None:
            return None
        return f"{head} {rest}".strip()


class ReadingJudge:
    def __init__(self, readings: TaiwanReadings):
        self.rw = readings
        self.syl = syllable_table(MFA_HOME / "dictionary/mandarin_china_mfa.dict")

    def syllable_phones(self, syllables: list[str], like: list[str] | None = None) -> list[str] | None:
        """Per-syllable phones; where ``like`` has the same segments, only the tone differs."""
        out = []
        for k, s in enumerate(syllables):
            ref = like[k] if like else None
            if ref and ref != s and ref.rstrip("12345") == s.rstrip("12345"):
                base = self.syl.get(ref)
                p = None if base is None else retone(base, s[-1] if s[-1].isdigit() else "5")
            else:
                p = self.syl.get(s)
            if p is None:
                return None
            out.append(p)
        return out

    def prepare(self, text: str):
        """Tokens for the .lab file, dictionary lines, and per-token reading variants."""
        tokens, lines, words = [], {}, []
        for start, piece in self.rw.segment(text):
            if not all("\u3400" <= c <= "\u9fff" for c in piece):
                continue          # punctuation
            cn = self.rw.cn_reading(piece)
            tw = self.rw.tw.get(piece)
            variants = {}
            pc = self.syllable_phones(cn)
            if pc:
                variants["cn"] = {"syl": cn, "ph": pc}
            target = segmental = False
            if tw and len(tw) == len(cn) and len(piece) > 1:
                pt = self.syllable_phones(tw, like=cn)
                if pt:
                    variants["tw"] = {"syl": tw, "ph": pt}
                diff = [k for k, (t, c) in enumerate(zip(tw, cn)) if not same_reading(t, c, piece[k])]
                lexical = [k for k in diff if not neutral_only(tw[k], cn[k])]
                if lexical and pt and pc and pt != pc:
                    target = True
                    segmental = any(tw[k].rstrip("12345") != cn[k].rstrip("12345") for k in lexical)
                elif pt == pc:
                    variants.pop("tw", None)
            if not variants:
                return None       # unknown syllable; skip the utterance rather than mis-align
            tok = f"w{len(tokens)}_{piece}"
            tokens.append(tok)
            lines[tok] = sorted({" ".join(v["ph"]) for v in variants.values()})
            words.append({"token": tok, "word": piece, "variants": variants,
                          "target": target, "segmental": segmental})
        return tokens, lines, words

    def run(self, items: list[dict], out: Path, jobs: int = 4) -> list[dict]:
        # MFA keys its temporary directory on the corpus folder name: make it unique per run
        tag = hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:10]
        corpus, aligned = out / f"corpus_{tag}", out / "aligned"
        shutil.rmtree(corpus, ignore_errors=True)
        corpus.mkdir(parents=True)
        lexicon, meta = {}, {}
        for it in items:
            prep = self.prepare(it["text"])
            if prep is None:
                meta[it["item_id"]] = {"status": "skipped_unknown_syllable"}
                continue
            tokens, lines, words = prep
            spk = corpus / it.get("speaker", "s0")
            spk.mkdir(exist_ok=True)
            # 16 kHz mono wav, the aligner's sample rate
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", it["wav"], "-ac", "1",
                            "-ar", "16000", str(spk / f"{it['item_id']}.wav")], check=True)
            (spk / f"{it['item_id']}.lab").write_text(" ".join(tokens))
            lexicon.update(lines)
            meta[it["item_id"]] = {"status": "prepared", "words": words}
        dict_path = out / "judge.dict"
        with dict_path.open("w", encoding="utf-8") as f:
            for tok, prons in sorted(lexicon.items()):
                for p in prons:
                    f.write(f"{tok}\t{p}\n")
        shutil.rmtree(aligned, ignore_errors=True)
        cmd = [str(MFA_BIN), "align", str(corpus), str(dict_path),
               str(MFA_HOME / "acoustic/mandarin_mfa.zip"), str(aligned),
               "--clean", "--overwrite", "-j", str(jobs), "--beam", "100", "--retry_beam", "400",
               "--output_format", "json", "--no_tokenization", "--quiet"]
        env = {**os.environ, "PATH": f"{MFA_BIN.parent}:{os.environ.get('PATH', '')}"}
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (out / "mfa.log").write_text(res.stdout + res.stderr)
        if res.returncode != 0:
            raise RuntimeError(f"mfa align failed; see {out / 'mfa.log'}")

        results = []
        for it in items:
            m = meta[it["item_id"]]
            jpath = aligned / it.get("speaker", "s0") / f"{it['item_id']}.json"
            if m["status"] != "prepared" or not jpath.exists():
                results.append({"item_id": it["item_id"], "status": m["status"] if m["status"] != "prepared"
                                else "not_aligned", "words": [], "judgements": []})
                continue
            tiers = json.loads(jpath.read_text())["tiers"]
            spans = {e[2]: e for e in tiers["words"]["entries"]}
            phones = tiers["phones"]["entries"]
            words_out, judgements = [], []
            for w in m["words"]:
                span = spans.get(w["token"])
                if span is None:
                    if w["target"]:
                        judgements.append({"word": w["word"], "heard": "unaligned"})
                    continue
                ph = [p for p in phones if p[0] >= span[0] - 1e-4 and p[1] <= span[1] + 1e-4]
                got = " ".join(p[2] for p in ph)
                chosen = next((k for k, v in w["variants"].items() if " ".join(v["ph"]) == got), None)
                sylls = []
                if chosen:
                    i = 0
                    for syl, sp in zip(w["variants"][chosen]["syl"], w["variants"][chosen]["ph"]):
                        n = len(sp.split())
                        seg = ph[i:i + n]
                        i += n
                        toned = next((p for p in seg if any(mk in p[2] for mk in TONE_MARKS)), seg[-1])
                        sylls.append({"syl": syl, "start": seg[0][0], "end": seg[-1][1], "rhyme_start": toned[0]})
                words_out.append({"word": w["word"], "start": span[0], "end": span[1], "chosen": chosen,
                                  "target": w["target"], "segmental": w["segmental"],
                                  "tw": w["variants"].get("tw", {}).get("syl"),
                                  "cn": w["variants"].get("cn", {}).get("syl"), "syllables": sylls})
                if w["target"]:
                    judgements.append({"word": w["word"], "tw": " ".join(w["variants"]["tw"]["syl"]),
                                       "cn": " ".join(w["variants"]["cn"]["syl"]),
                                       "segmental": w["segmental"], "heard": chosen or "other", "phones": got})
            results.append({"item_id": it["item_id"], "status": "ok", "words": words_out,
                            "judgements": judgements})
        return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--moedict", default="data/lexicon/concised.json.xz")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    judge = ReadingJudge(TaiwanReadings(args.moedict))
    results = judge.run(items, out, args.jobs)
    with (out / "judgements.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(j["heard"] for r in results for j in r["judgements"])
    tot = sum(c[k] for k in ("tw", "cn"))
    print(json.dumps({"items": len(items), "judged_words": dict(c),
                      "tw_rate": round(c["tw"] / tot, 3) if tot else None}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
