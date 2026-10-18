# Spanish → English Neural Machine Translator

A sequence-to-sequence Transformer model for Spanish-to-English translation, built from scratch in PyTorch. I started working on this project in 2024 while working through an online course on transformer-based MT models, since I wanted to try to build one of my own rather than just follow along. I picked Spanish and English since I speak both reasonably well, which makes it easier to check the outputs.

**Current model: 17.3 BLEU / 42.7 chrF** on 1,000 held-out Europarl sentences (October 2026 retrain). It translates parliamentary Spanish well and still struggles with everyday language, which is what the next training run targets.

---

## How it works

The model is a standard encoder-decoder Transformer (the original "Attention is All You Need" architecture) implemented using PyTorch's `nn.Transformer`. Both the source and target language share a single SentencePiece BPE tokenizer with a 16k vocabulary.

**Architecture** (68.7M parameters):
- 6 encoder layers, 6 decoder layers
- `d_model` = 512, 8 attention heads
- FFN dimension = 2048
- Sinusoidal positional encoding
- Label smoothing = 0.1, dropout = 0.1

**Training** (current model):
- Data: [Europarl v7](https://www.statmt.org/europarl/) ES-EN, the first 200,000 raw lines → ~104k pairs after cleaning (sentences ≤ 150 characters)
- Splits: 1,000 held-out test sentences (saved with the model), 3% validation, the rest for training
- Tokenizer: SentencePiece BPE, shared 16k vocab, retrained on each run's training split
- Optimizer: AdamW with the Noam warmup/decay schedule (8,000 warmup steps, peak scaled by 0.5)
- Mixed precision training (FP16) via `torch.cuda.amp`
- Early stopping after 3 epochs without a better validation loss: stopped at epoch 19, best checkpoint from epoch 16 (~1 hour on a Colab T4)
- **BLEU 17.3 / chrF 42.7** on the held-out test set (greedy decoding)

Decoding supports greedy search and beam search (beam width = 5).

---

## Results

| Model | Training data | Test BLEU / chrF |
|---|---|---|
| Original notebook checkpoint (May 2026) | Europarl, small subset | ~1 / 18.6† (couldn't stop generating) |
| Retrain 1 (Oct 2026) | ~104k pairs, end-token fix | 10.3 / 37.2* |
| **Retrain 2 (Oct 2026)** | ~104k pairs, all fixes below | **17.3 / 42.7** |

\* On 200 of the 1,000 test sentences, and only with a decoding workaround; on its own it never started a sentence (see below).
† On 50 sentences from the end of the original processed data, so not the same test set.

The original notebook reported 22.29 BLEU, but I couldn't reproduce that: the saved checkpoint scores about 1 BLEU on held-out data because of the bugs below, so that number was measured some other way.

Since everyday language is the weak spot, I also score models on four fixed 500-sentence test sets from different registers (never trained on, the same for every model). On 50 sentences from each:

| Domain | Retrain 1 | Retrain 2 |
|---|---|---|
| Parliament (Europarl) | 9.7 | **14.2** |
| Spoken (TED talks) | 4.4 | **6.7** |
| Dialogue (OpenSubtitles) | 2.9 | **5.7** |
| Everyday (Tatoeba) | 2.0 | **4.0** |

---

## What I fixed (October 2026)

Retraining with more data exposed three bugs that had been there from the start:

1. **The model never learned to stop.** The tokenizer defines `<s>`, `</s>` and `<pad>` as user-defined pieces, so SentencePiece's own `bos_id()`/`eos_id()`/`pad_id()` return -1. The dataset clamped those -1s to 0, the padding id, so every target ended in padding instead of an end token, and the loss ignores padding. That's why the old model's output looped. `src/tokenizer.py` now looks the pieces up by name (`<s>` = 1, `</s>` = 2, `<pad>` = 3).
2. **The model never started a sentence.** Targets already begin with `<s>`, and `shift_right` added another, so training taught "after `<s>`, predict `<s>`". Teacher forcing now uses the standard split: decoder input `[<s>, w1 … wn]`, labels `[w1 … wn, </s>]`.
3. **Training diverged once there was enough data.** With the original small dataset, the run never got past the 8,000-step warmup (7,659 pairs × 40 epochs ≈ 4,300 steps with the local pipeline), so the learning rate never peaked. With ~104k pairs it peaked at epoch 5, and with small batches, post-LayerNorm and FP16, the loss blew up to NaN at epoch 11. The schedule's peak is now halved (`lr_scale: 0.5`), and training stops early on a plateau or a NaN.

I also went from 7,659 training pairs to ~104k: the original `clean.py` only reads the first 20,000 lines of the 1.9M-line corpus.

---

## Project structure

```
├── src/
│   ├── architecture.py   # TransformerSeq2Seq model
│   ├── tokenizer.py      # SentencePiece wrapper
│   ├── dataset.py        # Dataset + collate
│   ├── train.py          # Training loop (config-driven)
│   └── utils.py          # Greedy/beam decode, masking
├── configs/
│   ├── retrain.yaml      # Current model: Europarl, all fixes
│   └── mix-v1.yaml       # Next run: multi-domain mix
├── data/data/
│   ├── raw/              # Europarl source files
│   ├── processed/        # Cleaned sentence pairs (original 7,659-pair set)
│   ├── mix-v1/           # Multi-domain pools + fixed eval sets (built by build_mix.py)
│   ├── load.py
│   └── clean.py
├── models/               # Saved tokenizer + checkpoint (original run)
├── notebooks/
│   └── full_notebook.ipynb   # Original training + eval pipeline (Colab)
├── main.py               # Training entry point (plain run, or --config)
├── build_mix.py          # Builds the multi-domain training mix
├── translate.py          # Run inference
├── eval.py               # BLEU evaluation
└── prepare_data.py       # Original tokenizer training script
```

---

## Running locally

```bash
pip install -r requirements.txt

# Train the current model (builds the data from data/data/raw, trains its own tokenizer,
# holds out a test set and reports test BLEU at the end)
python main.py --config configs/retrain.yaml

# Build the multi-domain mix (downloads from OPUS into data/data/opus first), then train on it
python build_mix.py
python main.py --config configs/mix-v1.yaml

# The original run (processed data + models/models tokenizer)
python main.py
```

Training needs a GPU; I run it on Colab, where checkpoints and metrics are saved to Google Drive as training goes. Each run writes its checkpoint, tokenizer and test set together into `output_dir/model/`.

`translate.py` and `eval.py` still use the decoders in `src/utils.py`; `greedy_batch` in `src/train.py` is the decoder used for the numbers above.

---

## Sample outputs

These are from the current model (retrain 2), on the same sentences I used for the original run so the two can be compared.

**Closest to training domain** — now essentially right:
```
ES: Es importante garantizar los derechos fundamentales de todos los ciudadanos.
EN: It is important to guarantee the rights of all citizens.
```
The original run gave "It is important that we need to have the rights of our citizens." The only thing missing now is "fundamental".

**Personal/colloquial — still off:**
```
ES: Llegué tarde porque se me olvidó el despertador.
EN: I was told it was the Conservative to be taken off the tail.
```
Still no idea what to do with everyday language. "Se me olvidó" (the reflexive construction) and alarm clocks basically never come up in parliament, so it reaches for parliamentary vocabulary ("the Conservative").

**Imperfect tense + personal narrative:**
```
ES: Cuando era pequeña, mi abuela me contaba historias cada noche.
EN: When I went to my night it was small, I was a little saddened.
```
Better than the original (it gets "When", "small" and the first person), but the meaning is gone. Personal narrative is still out of distribution.

**Long sentence:**
```
ES: A pesar de los esfuerzos realizados durante los últimos años para mejorar la situación
    económica de las regiones más vulnerables, los resultados han sido desiguales y aún
    queda mucho trabajo por hacer.
EN: Despite the efforts of the economy, the labour market has been much more vulnerable for
    the last few years and much more vulnerable to the working conditions.
```
It no longer loops: it produces one fluent sentence and stops. It opens correctly ("Despite the efforts…") but loses the thread halfway and repeats "much more vulnerable".

---

## Current limitations

The current model is still trained only on Europarl (EU parliamentary proceedings), so it's strong on formal, political language and weak on everyday Spanish: the domain table above shows it's about 3.5× better on parliament text than on everyday sentences. Beam search in `src/utils.py` also keeps extending beams after they've ended, so greedy decoding currently gives cleaner output.

---

## What's next

- **Multi-domain training (prepared):** `configs/mix-v1.yaml` trains on ~830k pairs from Tatoeba (everyday), TED talks (spoken), OpenSubtitles (dialogue) and Europarl, with the same fixes. Data from [OPUS](https://opus.nlpl.eu/); licenses are recorded in `data/data/mix-v1/manifest.json`.
- Fix beam search (carry finished beams forward, length normalization) and add a repetition penalty
- Experiment with a bigger model config once the data is more diverse
