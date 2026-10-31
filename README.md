# Accent Detective

Accent Detective is a lightweight, text-free and alignment-free neural network that measures how Taiwanese-accented Mandarin speech sounds. It offers multiple model sizes with shared weights, designed for speech data filtering and TTS accent evaluation.

- **Model:** [JacobLinCool/accent-detective](https://huggingface.co/JacobLinCool/accent-detective)
- **Demo:** [Hugging Face Space](https://huggingface.co/spaces/JacobLinCool/accent-detective)
- **Dataset:** [JacobLinCool/common_voice_25_zh-TW_taiwan_accent](https://huggingface.co/datasets/JacobLinCool/common_voice_25_zh-TW_taiwan_accent), clean Taiwan-accented speech screened from Common Voice
- **Paper:** [`paper/main.pdf`](paper/main.pdf)

## Install

```bash
pip install AccentDetective
```

## Use

```python
from accent_detective import load

det = load()                                         # downloads the checkpoint from the Hugging Face Hub
det.score("clip.wav")                                # the largest size, audio only
# {"logit": 5.25, "probability": 0.995, "index": 0.76, "size": "small", "mode": "audio", ...}
det.score("clip.wav", "這麼長的時間", size="nano")     # with the text it reads, the smallest size
det.score_sizes("clip.wav")                          # every size from one pass
det.score_recording("talk.wav")                      # any length: long audio is cut at pauses
for r in det.score_progressive("clip.wav"):          # anytime: each size's score as soon as its exit is reached
    print(r["size"], r["probability"])
```

```bash
accent-detective clip.wav other.mp3 --all-sizes
```

`probability` is the calibrated probability that the recording is Taiwan Mandarin rather than Mainland Mandarin. `logit` is the same value as log-odds. The text is optional: without it, the model judges the audio alone, and neither mode needs a forced alignment.

## Sizes

All four sizes are prefixes of one network: a smaller size is the first part of every larger one.

| Size | Parameters (audio only) | CPU latency, 5 s clip, 1 thread | Sealed AUC, published / listener-checked labels | BreezyVoice ranked above CosyVoice |
|---|---:|---:|---:|---:|
| Nano | 9.5 M | 74 ms | .888 / .956 | 79% |
| Tiny | 16.2 M | 132 ms | .903 / .961 | 79% |
| Mini | 22.9 M | 190 ms | .905 / .960 | 80% |
| Small | 43.0 M | 352 ms | .913 / .962 | 80% |

- **The evaluation:** the sealed set is Common Voice speakers held back for testing. The second column of each AUC pair uses labels corrected by a blind listening check.
- **Synthetic check:** BreezyVoice is CosyVoice-300M fine-tuned on Taiwan Mandarin, and a valid judge must score it as more Taiwanese.
- **Details:** in the paper.

## Intended use

- **For:**
  - Screening speech data for Taiwan-accented Mandarin.
  - Evaluating how Taiwanese a speech synthesiser sounds.
  - Research on Mandarin accents.
- **What the scores describe:** recordings, not people. Do not use them to infer anyone's origin or nationality.
- **Not a language detector:** the model scores any audio, including silence and other languages, so check that the input is Mandarin speech first.
- **Known weaknesses:**
  - Standard Mandarin from the Southwestern Mandarin region is judged Taiwan-accented more often than that of other Mainland regions.
  - Real noise and reverberation lower accuracy.

## Repository

| Path | Content |
|---|---|
| `src/accent_detective` | The `AccentDetective` package: inference only, published to PyPI |
| `research` | The code behind the paper and the dataset, and the exact data splits |
| `paper` | The paper: LaTeX source, figures, tables and the data behind them |
| `dataset` | The dataset card, the screening summaries and the blind listening check |
| `demo` | The Gradio demo |

## Licence

- **Code and model weights:** MIT.
- **The dataset:** CC0, like its source, Common Voice.
- **The lexicons the model downloads:**
  - 教育部《國語辭典簡編本》: CC BY-ND 3.0 TW.
  - CC-CEDICT: CC BY-SA 4.0.

## Citation

```bibtex
@misc{lin2026accentdetective,
  title        = {Accent Detective: One Small Judge to Clean Taiwan Mandarin Speech Data and to Evaluate Speech Synthesis},
  author       = {Lin, Jhen-Ke},
  year         = {2026},
  howpublished = {\url{https://github.com/JacobLinCool/AccentDetective}}
}
```
