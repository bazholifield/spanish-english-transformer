import sentencepiece as spm
import os

def train_tokenizer():
    processed_dir = os.path.join("data", "data", "processed")
    es_file = os.path.join(processed_dir, "all_es.txt")
    en_file = os.path.join(processed_dir, "all_en.txt")
    model_prefix = os.path.join("models", "models", "spm_shared")

    if not os.path.exists(es_file):
        print(f"Error: {es_file} not found.")
        return

    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    print("Training SentencePiece model...")

    spm.SentencePieceTrainer.train(
        input=f"{es_file},{en_file}",
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
