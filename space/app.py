"""The Hugging Face Space: my Spanish → English Transformer next to Helsinki-NLP's opus-mt.

Type Spanish, or pick a test sentence from one of the four registers, and see three translations:
my model with beam search, my model with greedy decoding, and opus-mt-es-en, a production model
trained on far more data (the README explains why that comparison flatters it).

space/build.py assembles the Space folder (this file, the model code as the `mt` package, the
weights, the examples); run the result locally with `python space/build/app.py`.
"""

import json
import re
from pathlib import Path

import gradio as gr
import torch

from mt.decode import beam_search, greedy_batch, load_model

HERE = Path(__file__).parent
MAX_CHARS = 600  # the model was trained on sentences up to 150 characters; this allows a few
torch.set_num_threads(2)  # a free Space has 2 CPUs

model, tok = load_model(HERE / "model")
EXAMPLES = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))["registers"]
OPUS = "Helsinki-NLP/opus-mt-es-en"
try:  # loaded at startup so the first visitor doesn't wait ~20 s; the project's own venv lacks transformers
    from transformers import MarianMTModel, MarianTokenizer
    opus = MarianTokenizer.from_pretrained(OPUS), MarianMTModel.from_pretrained(OPUS).eval()
except ImportError:
    opus = None


def split_sentences(text):
    """One sentence per item: the model only ever saw single sentences in training."""
    text = " ".join(text.split())[:MAX_CHARS]
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if s]


def opus_translate(sentences):
    """opus-mt with its default settings (beam 4, as in the README's comparison). None when
    transformers isn't installed; the Space installs it."""
    if opus is None:
        return None
    otok, omodel = opus
    with torch.no_grad():
        out = omodel.generate(**otok(sentences, return_tensors="pt", padding=True))
    return otok.batch_decode(out, skip_special_tokens=True)


def translate(text):
    sentences = split_sentences(text or "")
    if not sentences:
        return "", "", ""
    beam = beam_search(model, sentences, tok, "cpu", beam=5, alpha=1.0)
    greedy = greedy_batch(model, sentences, tok, "cpu")
    opus = opus_translate(sentences)
    return (" ".join(beam), " ".join(greedy),
            " ".join(opus) if opus is not None else "(opus-mt runs on the Space; transformers isn't installed here)")


ABOUT = """
**Scores on 1,800 held-out sentences** (500 per register, minus the ~200 whose English appeared in
an earlier training set):

| Register | Greedy | **Beam 5** | opus-mt-es-en |
|---|---|---|---|
| Parliament (Europarl) | 20.3 | **22.0** | 36.1 |
| Spoken (TED talks) | 25.7 | **26.7** | 42.3 |
| Dialogue (film subtitles) | 26.7 | **28.2** | 42.6 |
| Everyday (Tatoeba) | 45.2 | **46.5** | 58.8 |
| **All: BLEU / chrF** | 26.6 / 49.5 | **27.9 / 50.5** | 42.3 / 62.6 |

opus-mt is about 14 BLEU ahead, but that gap is an upper bound: it was trained on the whole OPUS
collection, the same corpora these test sentences come from, so it has probably seen some of them.

**The model:** a 6-layer encoder-decoder Transformer (68.7M parameters) I built in PyTorch and
trained from scratch on ~830k sentence pairs from four registers (Tatoeba, TED2020, OpenSubtitles,
Europarl) for 20 epochs on a Colab T4. One shared 16k SentencePiece vocabulary for both languages.

**Where it struggles:** long or unusual sentences, names, and idioms; it was trained on sentences
of up to 150 characters, so longer input is split into sentences first. Code, training history and
every bug I fixed along the way:
[github.com/bazholifield/spanish-english-transformer](https://github.com/bazholifield/spanish-english-transformer).
"""

with gr.Blocks(title="Spanish → English Transformer") as demo:
    gr.Markdown(
        "# Spanish → English, from scratch\n"
        "A Transformer I built and trained myself, next to a production model. Type Spanish or pick "
        "a sentence from the test sets below; the reference is the human translation they come with."
    )
    with gr.Row():
        with gr.Column():
            src = gr.Textbox(label="Spanish", lines=3, max_length=MAX_CHARS,
                             placeholder="¿Qué tal el fin de semana?")
            ref = gr.Textbox(label="Reference translation (test sentences only)", interactive=False)
            go = gr.Button("Translate", variant="primary")
        with gr.Column():
            out_beam = gr.Textbox(label="My model: beam search (width 5)")
            out_greedy = gr.Textbox(label="My model: greedy decoding")
            out_opus = gr.Textbox(label="Helsinki-NLP opus-mt-es-en (production reference)")
    outputs = [out_beam, out_greedy, out_opus]

    gr.Markdown("### Test sentences, picked at random from each register")
    for register, items in EXAMPLES.items():
        gr.Examples(examples=[[x["es"], x["en"]] for x in items], inputs=[src, ref], outputs=outputs,
                    fn=lambda es, en: translate(es), run_on_click=True, cache_examples=False,
                    label=register)

    with gr.Accordion("About the model and the scores", open=False):
        gr.Markdown(ABOUT)

    go.click(translate, inputs=src, outputs=outputs)
    src.submit(translate, inputs=src, outputs=outputs)
    src.input(lambda: "", outputs=ref)  # typed text has no reference

if __name__ == "__main__":
    demo.launch()
