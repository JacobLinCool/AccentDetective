---
language:
- zh
language_bcp47:
- zh-TW
- cmn-TW
license: cc0-1.0
task_categories:
- text-to-speech
- automatic-speech-recognition
- audio-classification
pretty_name: Common Voice 25 zh-TW, clean Taiwan-accented speech
tags:
- common-voice
- taiwan-mandarin
- accent
- speech
- tts
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: validated
    path: data/validated-*.parquet
  - split: other
    path: data/other-*.parquet
- config_name: annotations
  data_files:
  - split: all
    path: annotations/all.parquet
---

# Common Voice 25 zh-TW: clean Taiwan-accented speech

The Taiwan Mandarin recordings of Common Voice 25 (zh-TW), keeping only clean Taiwan-accented speech: clips that contain speech, read their prompt in Mandarin, and sound Taiwanese. It is meant as training speech for speech synthesis with a Taiwan accent.

## Why

A Common Voice locale records the language version a volunteer chose, not how the recording sounds. In a blind listening check, the zh-TW recordings that strong accent classifiers disagreed with were mostly not classifier errors: they held no speech, were in another language, or sounded Mainland. This dataset removes such recordings from the whole zh-TW release.

## What is in it

| | Clips | Hours | Speakers |
|---|---:|---:|---:|
| Common Voice 25 zh-TW (validated and other) | 135,710 | 126.5 | 2,276 |
| **This dataset** | **126,273** | **117.9** | **2,156** |

- **`default`:** the clean clips with their audio and every column of the source, in its two splits, `validated` (80,556 clips) and `other` (45,717).
- **`annotations`:** every clip of the source, clean or not, without audio: the scores and the flags below, joinable by `path`.

Added columns:

| Column | Meaning |
|---|---|
| `voiced_seconds` | Seconds of speech found by Silero VAD |
| `alignment_score` | Mean probability of the prompt's Mandarin pinyin letters under CTC forced alignment with the MMS aligner (torchaudio `MMS_FA`); empty when the prompt has characters that cannot be romanised |
| `taiwan_logit`, `taiwan_probability` | Accent Detective (Small, audio only, calibrated): log-odds and probability that the clip is Taiwan Mandarin rather than Mainland Mandarin |
| `speaker_flagged` | Whether the speaker's clips sound Mainland on average (always false in `default`) |

## How "clean" is defined

A clip is kept when all four hold:

1. **Speech:** at least 0.5 s of voiced audio.
2. **Reads the prompt in Mandarin:** alignment score at least 0.25, or a prompt that cannot be romanised.
3. **Taiwan accent:** `taiwan_probability` at least 0.5.
4. **Taiwan-accented speaker:** over a speaker's clips with speech (at least 3), the mean `taiwan_logit` is at least 0.

| Removed because | Clips |
|---|---:|
| No speech | 276 |
| Does not read the prompt in Mandarin | 336 |
| Does not sound Taiwanese | 8,695 |
| Speaker does not sound Taiwanese | 1,782 |

A clip can fail more than one check.

## Checked by listening

- **The sample:** one listener heard, blind and without the text, Common Voice zh-TW clips from an evaluation set. These were clips that at least 6 of 12 strong accent classifiers got wrong (100), plus random clips that all of them got right (20).

| | Disputed clips | Random clips all classifiers got right |
|---|---|---|
| Heard as Taiwanese | 29 → 6 kept | 13 → 13 kept |
| Heard as Mainland | 14 → 0 kept | 0 |
| Cannot tell | 21 → 0 kept | 3 → 3 kept |
| No speech | 20 → 0 kept | 0 |
| Another language | 16 → 2 kept | 4 → 4 kept |

- **On the disputed clips:** the screening removes every Mainland-sounding and every silent one, and with them most of the Taiwanese-sounding ones. It trades quantity for purity where the label is in doubt.
- **On clips every classifier gets right:** they pass through unchanged.
- **Thresholds:** the accent thresholds were set after this comparison, in favour of purity; the speech and alignment thresholds were fixed beforehand.

## Tools and licences

- **Audio and text:** Common Voice 25 zh-TW, CC0, from [OpenFormosa/common_voice_25_zh-TW](https://huggingface.co/datasets/OpenFormosa/common_voice_25_zh-TW).
- **[Accent Detective](https://huggingface.co/JacobLinCool/accent-detective):** MIT.
- **Silero VAD:** MIT.
- **MMS forced alignment:** weights under CC-BY-NC 4.0. Only the derived alignment scores are included here.
- **Licence of this dataset:** CC0, like its source.

## Intended use

- **For:** training and evaluating speech synthesis and recognition for Taiwan Mandarin, and research on Mandarin accents.
- **What the scores describe:** recordings, not people. Do not use them to infer anyone's origin or nationality.
- **Speakers:** keep Common Voice's terms, including not trying to identify the speakers.

## Citation

```bibtex
@inproceedings{ardila2020commonvoice,
  title     = {Common Voice: A Massively-Multilingual Speech Corpus},
  author    = {Ardila, Rosana and Branson, Megan and Davis, Kelly and Kohler, Michael and Meyer, Josh and Henretty, Michael and Morais, Reuben and Saunders, Lindsay and Tyers, Francis M. and Weber, Gregor},
  booktitle = {Proceedings of LREC},
  year      = {2020}
}
@misc{lin2026accentdetective,
  title        = {Accent Detective: A Compact Judge of Taiwan Mandarin Accent},
  author       = {Lin, Jhen-Ke},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/JacobLinCool/accent-detective}}
}
```
