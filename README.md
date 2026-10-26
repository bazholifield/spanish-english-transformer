# Spanish → English Neural Machine Translator

A sequence-to-sequence Transformer model for Spanish-to-English translation, built from scratch in PyTorch. I started working on this project in 2024 while working through an online course on transformer-based MT models, since I wanted to try to build one of my own rather than just follow along. I picked Spanish and English since I speak both reasonably well, which makes it easier to check the outputs.

**Current model: 26.6 BLEU / 49.5 chrF** across four registers (parliament, TED talks, film dialogue
and everyday sentences; 1,800 held-out sentences), up from 10.1 for the Europarl-only model and 6.7
for the first working retrain. It came from training on a multi-domain mix with stability fixes for
20 epochs (October 2026, run 5).

---

## How it works

The model is a standard encoder-decoder Transformer (the original "Attention is All You Need" architecture) implemented using PyTorch's `nn.Transformer`. Both the source and target language share a single SentencePiece BPE tokenizer with a 16k vocabulary.

**Architecture** (68.7M parameters):
- 6 encoder layers, 6 decoder layers
- `d_model` = 512, 8 attention heads
- FFN dimension = 2048
- Sinusoidal positional encoding
- Label smoothing = 0.1, dropout = 0.1

**Training** (current model, run 5):
- Data: ~830k sentence pairs from four [OPUS](https://opus.nlpl.eu/) corpora, built by `build_mix.py`:
  Tatoeba (everyday, all ~212k), TED2020 talks (200k), OpenSubtitles (dialogue, 250k sampled from ~105M),
  and Europarl (parliament, 150k). Sentences ≤ 150 characters, deduplicated across corpora, with every
  test sentence excluded by both its Spanish and its English side.
- Splits: 1% validation; testing on four fixed 500-sentence sets, one per register, never trained on
- Tokenizer: SentencePiece BPE, shared 16k vocab, retrained on each run's training split
- Pre-LayerNorm Transformer; AdamW with the Noam warmup/decay schedule (8,000 warmup steps, peak scaled by 0.5)
- Mixed precision training (FP16) via `torch.cuda.amp`, about 25 minutes per epoch on a Colab T4
- 20 epochs (~8.5 hours) with no loss spikes; validation loss fell every epoch, best at epoch 20

Decoding supports greedy search and beam search (beam width = 5).

---

## Results

The first three models were trained on Europarl only and scored on 1,000 held-out Europarl sentences:

| Model | Training data | Europarl test BLEU / chrF |
|---|---|---|
| Original notebook checkpoint (May 2026) | Europarl, small subset | ~1 / 18.6† (couldn't stop generating) |
| Retrain 1 (Oct 2026) | ~104k pairs, end-token fix | 10.3 / 37.2* |
| **Retrain 2 (Oct 2026)** | ~104k pairs, all fixes below | **17.3 / 42.7** |

\* On 200 of the 1,000 test sentences, and only with a decoding workaround; on its own it never started a sentence (see below).
† On 50 sentences from the end of the original processed data, so not the same test set.

The original notebook reported 22.29 BLEU, but I couldn't reproduce that: the saved checkpoint scores about 1 BLEU on held-out data because of the bugs below, so that number was measured some other way.

Since everyday language was the weak spot, I score every model on four fixed test sets from
different registers, 500 sentences each and the same for every model. The scores below use the
1,800 of those 2,000 sentences whose English translation never appeared in any training data
(see fix 4 below):

| Register (sentences) | Retrain 1 | Retrain 2 | Retrain 3 | Retrain 4 | **Retrain 5** |
|---|---|---|---|---|---|
| Parliament: Europarl (482) | 11.4 | 16.2 | 17.5 | 17.8 | **20.3** |
| Spoken: TED talks (488) | 4.2 | 6.4 | 21.5 | 22.5 | **25.7** |
| Dialogue: OpenSubtitles (443) | 3.3 | 5.1 | 25.2 | 24.4 | **26.7** |
| Everyday: Tatoeba (387) | 4.2 | 8.4 | 43.5 | 42.4 | **45.2** |
| **All (1,800): BLEU / chrF** | 6.7 / 29.6 | 10.1 / 33.7 | 23.8 / 47.6 | 24.0 / 47.3 | **26.6 / 49.5** |

Training on the mix barely changed parliamentary text and multiplied everything else 3–5×. Retrain 4 added
the stability fixes (fix 5 below): it scored about the same as retrain 3's best epoch, but trained for all
12 epochs without collapsing and was still improving. Retrain 5 is the same setup for 20 epochs, and it
improves on every register (+2.6 BLEU overall).
(Greedy decoding. Retrain 1 is decoded the way it was trained, starting from `<s> <s>`.)

---

## What I fixed (October 2026)

Retraining with more data exposed three bugs that had been there from the start:

1. **The model never learned to stop.** The tokenizer defines `<s>`, `</s>` and `<pad>` as user-defined pieces, so SentencePiece's own `bos_id()`/`eos_id()`/`pad_id()` return -1. The dataset clamped those -1s to 0, the padding id, so every target ended in padding instead of an end token, and the loss ignores padding. That's why the old model's output looped. `src/tokenizer.py` now looks the pieces up by name (`<s>` = 1, `</s>` = 2, `<pad>` = 3).
2. **The model never started a sentence.** Targets already begin with `<s>`, and `shift_right` added another, so training taught "after `<s>`, predict `<s>`". Teacher forcing now uses the standard split: decoder input `[<s>, w1 … wn]`, labels `[w1 … wn, </s>]`.
3. **Training diverged once there was enough data.** With the original small dataset, the run never got past the 8,000-step warmup (7,659 pairs × 40 epochs ≈ 4,300 steps with the local pipeline), so the learning rate never peaked. With ~104k pairs it peaked at epoch 5, and with small batches, post-LayerNorm and FP16, the loss blew up to NaN at epoch 11. The schedule's peak is now halved (`lr_scale: 0.5`), and training stops early on a plateau or a NaN.

4. **Some test sentences leaked into training.** `build_mix.py` held test pairs out by their Spanish
   side only, but Tatoeba often has several Spanish versions of one English sentence. About 22% of the
   Tatoeba test translations (11% for subtitles) were also training targets. It now excludes test pairs
   by both sides (`mix-v2`), and every score above leaves out the leaked sentences. The run 3 gains held
   up: Tatoeba moved from 43.9 to 43.5, the others barely changed.
5. **Run 3 collapsed in epoch 9.** After eight good epochs the validation loss jumped and never
   recovered, the classic instability of "post-LayerNorm" Transformers trained in FP16. Run 4 uses
   pre-LayerNorm (normalise before each sublayer, as most modern Transformers do), rolls back to the
   best checkpoint at half the learning rate if the loss spikes, and logs gradient size and the FP16
   loss scale every 500 steps so the next problem can be diagnosed rather than guessed.

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
│   ├── retrain.yaml      # Run 2: Europarl only, all bug fixes
│   ├── mix-v1.yaml       # Run 3 (current model): multi-domain mix
│   └── mix-v2.yaml       # Run 4 (next): leak-free mix, pre-LN, spike rollback
├── data/data/
│   ├── raw/              # Europarl source files
│   ├── processed/        # Cleaned sentence pairs (original 7,659-pair set)
│   ├── mix-v1/, mix-v2/  # Multi-domain pools + fixed eval sets (built by build_mix.py)
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

# Build the multi-domain mix (download the OPUS corpora into data/data/opus first; URLs in
# build_mix.py), then train on it. Scores the best checkpoint on the four test sets at the end.
python build_mix.py --out data/data/mix-v2
python main.py --config configs/mix-v2.yaml

# Run 2: Europarl only (builds the data from data/data/raw, holds out a 1,000-sentence test set)
python main.py --config configs/retrain.yaml

# The original run (processed data + models/models tokenizer)
python main.py
```

Training needs a GPU; I run it on Colab, where checkpoints and metrics are saved to Google Drive as training goes. Each run writes its checkpoint, tokenizer and test set together into `output_dir/model/`.

`translate.py` and `eval.py` still use the decoders in `src/utils.py`; `greedy_batch` in `src/train.py` is the decoder used for the numbers above.

---

## Sample outputs

These are from the current model (retrain 5), on the same sentences I used for the original run so
the models can be compared.

**Parliamentary:** nearly right:
```
ES: Es importante garantizar los derechos fundamentales de todos los ciudadanos.
EN: It is important to ensure fundamental rights of all citizens.
```
Only "the" is missing ("the fundamental rights"). Retrain 2 dropped "fundamental", and retrain 3
ended with the clumsy "…are given".

**Personal/colloquial:** right:
```
ES: Llegué tarde porque se me olvidó el despertador.
EN: I was late because I forgot the alarm.
```
Retrain 2 gave "I was told it was the Conservative to be taken off the tail." The reflexive *se me
olvidó* is handled correctly, which is exactly what the everyday data was for.

**Imperfect tense + personal narrative:** right:
```
ES: Cuando era pequeña, mi abuela me contaba historias cada noche.
EN: When I was little, my grandmother told me stories every night.
```
Spanish drops the subject of *era* ("when [I] was little"). Retrain 3 filled it in with the nearest
noun ("When my grandmother was little, I was telling stories"); retrain 5 gets both the dropped
subject and who told the stories. Dropped subjects (pro-drop) are still worth testing systematically
(see ROADMAP), but this classic case now works.

**Long sentence:**
```
ES: A pesar de los esfuerzos realizados durante los últimos años para mejorar la situación
    económica de las regiones más vulnerables, los resultados han sido desiguales y aún
    queda mucho trabajo por hacer.
EN: Despite the efforts made to improve the economic situation in recent years, the regions have
    been much more vulnerable, and they have still been much more vulnerable.
```
The first half is now a fluent, faithful clause, but the second half loses the thread ("the results
have been uneven and much work remains") and repeats itself. Long sentences are the clearest
remaining weakness.

---

## Current limitations

Long sentences still lose the thread and repeat themselves.
Beam search in `src/utils.py` keeps extending beams after they've ended, so greedy decoding gives
cleaner output for now. The model is trained on TED2020 (CC BY-NC-ND) and OpenSubtitles (research
use), so it's for non-commercial use.

---

## What's next

The plan is in [ROADMAP.md](ROADMAP.md): finish run 4 and fix beam search, publish the model with a
browser demo on Hugging Face, and a linguistic evaluation (a challenge set of Spanish phenomena such
as pro-drop, *se* constructions, *ser/estar* and false friends, plus an error analysis).
