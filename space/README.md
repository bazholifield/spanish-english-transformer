---
title: Spanish to English Transformer
emoji: 🌎
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: A Spanish-to-English Transformer built from scratch, vs opus-mt
---

# Spanish to English, from scratch

A sequence-to-sequence Transformer for Spanish-to-English translation that I built and trained
from scratch in PyTorch, shown next to Helsinki-NLP's production model, opus-mt-es-en. I picked
Spanish and English because I speak both, which makes it easy to check the output by hand.

Try it: type a Spanish sentence, or click one of the test sentences (three per register, picked
at random, not hand-picked). You'll see my model with beam search and with greedy decoding, and
opus-mt for reference.

## Results

BLEU on 1,800 held-out sentences from four registers:

| Register | Greedy | **Beam 5** | opus-mt-es-en |
|---|---|---|---|
| Parliament (Europarl) | 20.3 | **22.0** | 36.1 |
| Spoken (TED talks) | 25.7 | **26.7** | 42.3 |
| Dialogue (film subtitles) | 26.7 | **28.2** | 42.6 |
| Everyday (Tatoeba) | 45.2 | **46.5** | 58.8 |
| **All: BLEU / chrF** | 26.6 / 49.5 | **27.9 / 50.5** | 42.3 / 62.6 |

opus-mt is about 14 BLEU ahead, and that gap is an upper bound: it was trained on the whole OPUS
collection, the same corpora these test sentences come from, so it has probably seen some of them.

## The model

- 6 encoder and 6 decoder layers, 512 dimensions, 8 attention heads: 68.7M parameters
- One shared SentencePiece vocabulary (16k pieces) for both languages
- Trained on about 830k sentence pairs mixed from four registers (Tatoeba, TED2020, OpenSubtitles,
  Europarl), 20 epochs on a Colab T4
- The weights here are stored in half precision to keep the download small; they're loaded into
  the full-precision model, so translations match the original up to rounding

## Limits

It was trained on sentences of up to 150 characters, so longer input is split into sentences
first. It's weakest on long sentences, names and idioms, and on formal parliamentary language.

## Licence

Non-commercial use only. The training data includes TED2020 (CC BY-NC-ND 4.0) and OpenSubtitles
(research use), so the weights are shared for research and demonstration, not for commercial use.

The code, the training history and every bug I fixed along the way are on GitHub:
[bazholifield/spanish-english-transformer](https://github.com/bazholifield/spanish-english-transformer).
