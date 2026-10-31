# Paper

*Accent Detective: One Small Judge to Clean Taiwan Mandarin Speech Data and to Evaluate Speech Synthesis* (Interspeech format).

```bash
# figures (figures/*.pdf) and table bodies (tables/*.tex) from data/
PYTHONPATH=research/src python paper/scripts/make_figures.py      # from the repository root
# the PDF
cd paper && latexmk -pdf main.tex
```

## Data

The figures and tables read these files in `data/`:

| File | Content | Produced by |
|---|---|---|
| `sealed.json` | Every system on the sealed set: AUC with the published and the listener-checked labels, paired differences, recording conditions, the listening counts | `research/scripts/sealed_eval.py --out paper/data/sealed.json` |
| `report_main.json` | Synthetic pairs, voice set and KeSpeech for every system | `research/scripts/evaluate.py` on the scores gathered by `scripts/collect_scores.py` |
| `report_readouts.json` | The same for the four readouts of one trunk | `research/scripts/evaluate.py` |
| `report_probes.json` | The same for the frozen-encoder probes (HuBERT and wav2vec 2.0, base and large) | `research/scripts/encoder_probe.py` and `research/scripts/evaluate.py` |
| `report_dev_ablations.json`, `teacher_ensembles.json` | The development benchmark of the ablation table | `research/scripts/bench_eval.py` |
| `latency_nested.json`, `latency_ref_*.json` | Latency, throughput and anytime scoring on one machine (NVIDIA L4 and its host CPU) | `scripts/latency_nested.py` |

`../dataset/screening_summary_p0.5.json` gives the screening results of the listening table (`research/scripts/screen_analyze.py`).

`Interspeech.cls` is the Interspeech 2025 class. Replace it with the current year's official class before submission.
