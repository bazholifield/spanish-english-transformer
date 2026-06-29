import os
import torch
import random
from src.train import run_training

def load_parallel_corpus(en_path, es_path):
    """Helper to load text files into lists."""
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]
    with open(es_path, 'r', encoding='utf-8') as f:
        es_lines = [line.strip() for line in f]
    return en_lines, es_lines

def main():
    EN_FILE = os.path.join("data", "data", "processed", "all_en.txt")
    ES_FILE = os.path.join("data", "data", "processed", "all_es.txt")
    SPM_MODEL = os.path.join("models", "models", "spm_shared.model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    if not os.path.exists(EN_FILE) or not os.path.exists(ES_FILE):
        raise FileNotFoundError(
            f"Missing data files: {EN_FILE} or {ES_FILE}. "
            "Please ensure your text data is in the 'data/data/processed/' folder."
        )
    
    full_en, full_es = load_parallel_corpus(EN_FILE, ES_FILE)
    
    # Shuffle and split data
    combined = list(zip(full_en, full_es))
    random.seed(42)
    random.shuffle(combined)
    full_en, full_es = zip(*combined)

    split_idx = int(len(full_en) * 0.9)
    train_en, val_en = list(full_en[:split_idx]), list(full_en[split_idx:])
    train_es, val_es = list(full_es[:split_idx]), list(full_es[split_idx:])
    
    print(f"Loaded {len(full_en)} pairs.")
    print(f"Training on {len(train_en)}, Validating on {len(val_en)}")

    # Training loop
    model, src_tok, tgt_tok = run_training(
        src_sentences_train=train_es,
        tgt_sentences_train=train_en,
        src_sentences_val=val_es,
        tgt_sentences_val=val_en,
        sp_model_path=SPM_MODEL,
        device=device
    )

    print("\nTraining complete.")

if __name__ == "__main__":
    main()