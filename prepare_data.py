import sentencepiece as spm
import os

def train_tokenizer():
    input_files = "data/processed/all_es.txt,data/processed/all_en.txt"
    model_prefix = "models/spm_shared"

    if not os.path.exists("data/processed/all_es.txt"):
        print("Error: data/all_es.txt not found.")
        return

    print("Training SentencePiece model...")
    
    # training run
    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=16000,
        model_type='bpe',
        character_coverage=1.0,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        user_defined_symbols="<pad>,<s>,</s>"
    )
    
    print(f"Tokenizer trained! Files saved to {model_prefix}.model and .vocab")

if __name__ == "__main__":
    train_tokenizer()