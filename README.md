# Spanish → English Neural Machine Translator

A sequence-to-sequence Transformer model for Spanish-to-English translation, built from scratch in PyTorch. I started this project alongside an online course on transformer-based MT models, since I wanted to actually build one of my own rather than just follow along. I picked Spanish and English since I speak both reasonably well, which makes it easier to check the outputs.

Trained baseline achieving 22.29 BLEU. Still a work in progress, but it's producing real translations and I plan to keep improving it.

---

## How it works

The model is a standard encoder-decoder Transformer (the original "Attention is All You Need" architecture) implemented using PyTorch's `nn.Transformer`. Both the source and target language share a single SentencePiece BPE tokenizer with a 16k vocabulary.

**Architecture:**
- 6 encoder layers, 6 decoder layers
- `d_model` = 512, 8 attention heads
- FFN dimension = 2048
- Sinusoidal positional encoding
- Label smoothing = 0.1, dropout = 0.05

**Training:**
- Data: [Europarl v7](https://www.statmt.org/europarl/) ES-EN corpus (~38k sentence pairs after cleaning and filtering)
- Tokenizer: SentencePiece BPE, shared vocab, 16k tokens
- Optimizer: AdamW with a custom warmup/decay LR schedule
- Mixed precision training (FP16) via `torch.cuda.amp`
- Trained for 35 epochs on a Colab T4 GPU (~52 minutes)
- **BLEU: 22.29** on a held-out test set

Decoding supports both greedy search and beam search (beam width = 5).

---

## Project structure

```
├── src/
│   ├── architecture.py   # TransformerSeq2Seq model
│   ├── tokenizer.py      # SentencePiece wrapper
│   ├── dataset.py        # Dataset + collate
│   ├── train.py          # Training loop
│   └── utils.py          # Greedy/beam decode, masking
├── data/
│   ├── raw/              # Europarl source files
│   ├── processed/        # Cleaned sentence pairs
│   ├── load.py
│   └── clean.py
├── models/               # Saved tokenizer + best checkpoint
├── notebooks/
│   └── full_notebook.ipynb   # Full training + eval pipeline (Colab)
├── main.py               # Local training entry point
├── translate.py          # Run inference
├── eval.py               # BLEU evaluation
└── prepare_data.py       # Data prep script
```

---

## Running locally

```bash
pip install -r requirements.txt

# Prep data
python prepare_data.py

# Train
python main.py

# Translate
python translate.py

# Evaluate (BLEU)
python eval.py
```

The notebook (`notebooks/full_notebook.ipynb`) is the main training environment — it runs on a Colab GPU with Google Drive used for checkpointing.

---

## Sample outputs

These are from the most recent training run (BLEU 22.29). I picked sentences to show where it holds up and where it falls apart.

**Closest to training domain** — gets the general idea, wrong details:
```
ES: Es importante garantizar los derechos fundamentales de todos los ciudadanos.
EN: It is important that we need to have the rights of our citizens.
```
The general meaning is present, but missing specifics. It falls apart after this first sentence, but this example shows promise.

**Personal/colloquial — completely off:**
```
ES: Llegué tarde porque se me olvidó el despertador.
EN: However, I would like to say that there is a great deal with the vote on this matter.
```
The model has no idea what to do with everyday language, so it just defaults to parliamentary filler phrases. "Se me olvidó" (the reflexive construction) is basically absent from the training data.

**Imperfect tense + personal narrative:**
```
ES: Cuando era pequeña, mi abuela me contaba historias cada noche.
EN: In my opinion, I would like to say that it is a great deal of the same time.
```
Drops the actual meaning entirely and hallucinates a parliamentary opinion statement. The imperfect tense and first-person personal context are essentially out-of-distribution.

**Long sentence — degrades mid-way:**
```
ES: A pesar de los esfuerzos realizados durante los últimos años para mejorar la situación
    económica de las regiones más vulnerables, los resultados han sido desiguales y aún
    queda mucho trabajo por hacer.
EN: It is therefore important that we need to have a great deal with the EU, and more than
    the United States for example. We have already been able to be done in this debate on
    the European Union. [then loops]
```
The most Europarl-sounding of the inputs, but it still drifts, it picks up a few real content words ("EU", "work") but loses the structure of the original.

---

## Current limitations

The model was trained exclusively on Europarl (EU parliamentary proceedings), so it has a strong bias toward formal, political-sounding output regardless of input. Translations of everyday sentences work grammatically but tend to come out sounding like a European Parliament speech. More diverse training data would help a lot here.

---

## What's next

- Larger and more diverse training dataset
- Repetition penalty in beam search decoding
- Experiment with bigger model config
- Better evaluation beyond BLEU
