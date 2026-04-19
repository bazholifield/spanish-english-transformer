# Spanish-to-English Transformer Translator

A PyTorch implementation of a Seq2Seq Transformer model for machine translation, trained on the Europarl-v7 corpus.

I started putting this project together in conjunction with an online course in transformer-based MT models. I have been able to put the architecture together and the training scripts and data pipelines are implemented, but full model training is pending due to internet constraints.

## Project Structure
* `src/`: Core logic (Architecture, Tokenizer, Training loop).
* `data/`: Raw and processed Europarl data.
* `models/`: Where the SentencePiece model is stored.
* `checkpoints/`: Where the best weights are saved.
* `main.py`: The entry point for starting training.

## Steps
- [x] Transformer Architecture (Encoder/Decoder)
- [x] SentencePiece BPE Tokenization Pipeline
- [x] Data Cleaning & Filtering Scripts
- [ ] Model Training (Pending)
- [ ] BLEU Evaluation

## How to Run 
1. `python clean.py`
2. `python prepare_data.py`
3. `python main.py`