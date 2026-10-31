# Research code

This folder holds the code behind the paper and the dataset. The `twjudge` library under `src/` contains the models, features and aligners. `scripts/` holds one script per step. Every step ran on a single NVIDIA L4 GPU. The published package in `../src/accent_detective` is a standalone reimplementation of inference only, and it gives identical scores.

```bash
cd research
uv sync                      # or: pip install -e .
export PYTHONPATH=src
```

## Data

- **Corpus.** Common Voice 25 zh-TW (Taiwan) and Common Voice 21 zh-CN (Mainland), both CC0, split by speaker.
  - `splits/` lists the exact clips of every split used in the paper: `cv_train`, `cv_dev`, `cv_test`, the sealed evaluation set (`sealed`) and the accent-task corpus (`task`).
  - `twjudge.data.build_corpus` rebuilds a corpus of the same design from scratch.
- **Unlabelled distillation audio.** Common Voice, AISHELL-1 and THCHS-30, all Apache 2.0 or CC0 (`scripts/distill_data.py`).
- **Evaluation bundle.** The scripts read one folder:
  - `sets/<set>.jsonl`: `item_id`, `audio` (a relative path), `text`, `speaker`.
  - The audio files themselves.

  The sets:
  - `cv_*` and `r10_sealed`: real speech.
  - `breezy` and `cosy300m`: the 238 development synthetic pairs.
  - `synth2_breezy` and `synth2_cosy`: the 1,000 synthetic pairs.
  - `ming_b0`: the voice set, 119 sentences in the four built-in voices of `inclusionAI/Ming-3.0-tts-0.5B`.
  - `kespeech`: KeSpeech, evaluation only.
- **Synthetic speech.** BreezyVoice and CosyVoice-300M read the same prompts with the same voice prompts, using their official inference code. `scripts/heldout_set.py` writes the 250 synthetic-pair sentences. It also builds a further test set, which the paper retired and does not use.

## Steps

| Step | Scripts |
|---|---|
| Teachers: syllable judges on frozen Chinese HuBERT and wav2vec 2.0 base | `twjudge.accent.ssl_features`, `bench_features.py`, `bench_train.py`, `teacher_syllables.py` |
| First student: distilled encoder with its own CTC alignment head | `distill_data.py`, `augment_assets.py`, `distill_student.py`, `compact_ctc_head.py`, `compact_export.py` |
| Nested student: exits at several depths, distilled at every exit | `nested_distill.py`, `syllable_features.py`, `nested_judge_features.py`, `nested_judge_train.py` |
| Accent task corpus, sealed set, alignments | `task_prep.py` |
| Nested trunk with the accent signal: arm C, then arm H, the released trunk | `train_trunk.py` |
| Readouts (attentive pooling, optional text) and the checkpoint | `train_readouts.py`, `export_model.py` |
| Baselines and end-to-end scoring | `baselines.py`, `bench_finetune.py`, `score_e2e.py`, `bench_eval.py` |
| Evaluation on the sealed set, synthetic pairs, voice set and KeSpeech | `evaluate.py`, `metric_noise.py` |
| Recording conditions (MP3, telephone, real noise, real reverberation) | `channels.py`, `robustness.py` |
| Encoder probes: one frozen encoder, base against large | `encoder_probe.py` |
| Listening check and label-corrected evaluation | `sealed_eval.py`, with `../dataset/listening/` |
| Screening Common Voice zh-TW and releasing the clean subset | `screen_score.py`, `screen_analyze.py`, `screen_release.py` |
| Figures, tables and latency of the paper | `../paper/scripts/` |

Each script's docstring gives its inputs, outputs and an example command line.

`train_trunk.py`'s arms are the training designs the paper's design choices were compared on:

| Arm | Design |
|---|---|
| A | Feature distillation only |
| B | A with source-balanced sampling |
| C | B with the accent signal through the last exit |
| D | C on every exit |
| H | The released trunk: D, started from an arm-C trunk, with exits after layers 3, 6, 9 and 18 and one loss weight per exit |

## The listening check

`../dataset/listening/` holds the blind listening check of the sealed set:

- `pools.json`: the disputed and control clips, with the 12 selector systems and their thresholds.
- `rating_key.json`: the anonymised clip key.
- `ratings_pilot.json` and `ratings_check.json`: the listener's answers.

`sealed_eval.py` turns them into the published and the listener-checked labels and writes `../paper/data/sealed.json`.
