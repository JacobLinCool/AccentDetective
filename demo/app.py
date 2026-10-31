"""Accent Detective demo: how Taiwanese does a Mandarin recording sound?

    pip install -r demo/requirements.txt
    python demo/app.py
"""
import os

import gradio as gr

from accent_detective import load

det = load()
SIZES = {s: det.parameters(s) for s in det.sizes}
LAYERS = dict(zip(det.sizes, det.meta["config"]["exits"]))
MAIN = det.sizes[-1]                    # the largest size
HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
.ad-score {font-size:48px; font-weight:700; line-height:1.1; margin-top:6px}
.ad-bar {height:8px; border-radius:4px; background:var(--border-color-primary); overflow:hidden; margin:6px 0 10px}
.ad-bar > div {height:100%; background:linear-gradient(90deg,#d9534f,#e0a030 50%,#2f9e5b)}
.ad-note {font-size:12px; opacity:.65; margin-top:2px}
.ad-empty {opacity:.5; font-size:14px; padding:24px 0}
.ad-sizes {width:100%; border-collapse:collapse; margin-top:14px; font-size:14px}
.ad-sizes td, .ad-sizes th {padding:4px 6px; border-top:1px solid var(--border-color-primary); text-align:right}
.ad-sizes td:first-child, .ad-sizes th:first-child {text-align:left}
.ad-sizes tr.ad-on td {font-weight:700}
"""

ABOUT = """
**The number.** The probability, from Small, the largest of the four sizes, that the recording is Taiwan Mandarin rather
than Mainland Mandarin. 50% means undecided. The probability is calibrated on development data.

**How to use it.**
- Compare renditions of the same sentence, for example two speech synthesisers reading it.
- The text is optional. With it, the model also looks at the syllables the text implies; both modes are about equally
  accurate.
- Recordings longer than 15 s are cut at pauses into 4–10 s pieces, scored without the text and averaged by length.

**Four sizes.** The table lists the four depths of the one network, Nano, Tiny, Mini and Small. A smaller size is the
first part of a larger one, so one pass gives all four.

**Known limits.**
- Standard Mandarin from the Southwestern Mandarin region (Sichuan, Chongqing, Guizhou, Yunnan) is judged
  Taiwan-accented far more often than that of other Mainland regions.
- Background noise and room reverberation lower accuracy; MP3 compression barely matters.
- The model scores any audio, including silence and other languages: check that the input is Mandarin speech.
"""

# Common Voice clips (CC0) of speakers the model never trained on
EXAMPLES = [("taiwan_1.mp3", "看到這種狀況會有點擔心"), ("taiwan_2.mp3", "入住當天早餐就隨便你吃"),
            ("mainland_1.mp3", "勇敢是指具有勇气的行为。"), ("mainland_2.mp3", "他用的是经验论心理学的方法。")]


def pct(p: float) -> str:
    """A probability as a percentage, with one decimal near 0% or 100%."""
    v = 100 * p
    return f"{v:.0f}%" if 1 <= v <= 99 else f"{v:.1f}%"


def score(audio, text):
    if audio is None:
        raise gr.Error("Please provide a recording")
    res = det.score_recording(audio, (text or "").strip() or None, sizes=det.sizes)
    r = res[MAIN]
    if len(r["pieces"]) > 1:
        note = f"A long recording, cut at pauses into {len(r['pieces'])} pieces and scored without the text"
    elif text and text.strip() and r["mode"] == "audio":
        note = "The text has no usable Han syllables; scored from the audio alone"
    else:
        note = "Scored with the text" if r["mode"] == "text" else "Scored from the audio alone"
    rows = "".join(f'<tr class="{"ad-on" if k == MAIN else ""}"><td>{k.capitalize()} ({SIZES[k] / 1e6:.1f} M parameters, '
                   f'{LAYERS[k]} layers)</td><td>{pct(v["probability"])}</td></tr>' for k, v in res.items())
    return (f'<div class="ad-score">{pct(r["probability"])}</div>'
            f'<div class="ad-note">Probability of Taiwan Mandarin ({MAIN.capitalize()}; 50% = undecided)</div>'
            f'<div class="ad-bar"><div style="width:{100 * r["probability"]:.1f}%"></div></div>'
            f'<div class="ad-note">{note}</div>'
            f'<table class="ad-sizes"><tr><th>Size</th><th>Probability of Taiwan Mandarin</th></tr>{rows}</table>')


with gr.Blocks(title="Accent Detective", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Accent Detective: how Taiwanese does a Mandarin recording sound?\n"
                "Upload or record Mandarin speech. The text is optional.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=300):
            audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Speech")
            text = gr.Textbox(label="The text it reads (optional)", lines=2)
            go = gr.Button("Score", variant="primary", size="lg")
        with gr.Column(scale=3):
            out = gr.HTML('<div class="ad-empty">Upload or record Mandarin speech</div>')
            with gr.Accordion("How to read the scores", open=False):
                gr.Markdown(ABOUT)
    gr.Examples(examples=[[os.path.join(HERE, "examples", f), t] for f, t in EXAMPLES],
                example_labels=[f"Common Voice {'zh-TW' if f.startswith('taiwan') else 'zh-CN'}: {t}" for f, t in EXAMPLES],
                inputs=[audio, text], outputs=out, fn=score, run_on_click=True, cache_examples=False, label="Examples")
    go.click(score, [audio, text], out, api_name="score")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
