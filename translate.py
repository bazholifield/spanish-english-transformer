import torch
import os
from src.architecture import TransformerSeq2Seq
from src.tokenizer import SPTokenizer
from src.utils import beam_search_decode

def translate():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    sp = SPTokenizer("models/spm_shared.model")
    src_tok = tgt_tok = sp 

    # Initialize and load model (check hyperparameters - same as training loop!)
    model = TransformerSeq2Seq(
        src_vocab_size=src_tok.vocab_size(),
        tgt_vocab_size=tgt_tok.vocab_size(),
        d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.0, # Dropout 0 for inference
        pad_idx=src_tok.id_pad
    ).to(device)

    # Load weights from training run
    checkpoint_path = "best_transformer.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found. Using unitialized model (will output gibberish).")

    model.eval()

    # translation loop
    test_sentences_es = [
        "Ella se fue a la tienda por la mañana.",
        "¿Podrías ayudarme con este ejercicio de PyTorch?",
        "Me encanta programar en Python.",
        "El sol es caliente y el cielo es azul."
    ]

    print("\n--- Translation Results ---\n")
    with torch.no_grad():
        for src in test_sentences_es:
            try:
                # encode source text to tensor
                src_ids = torch.tensor(src_tok.encode(src), dtype=torch.long).to(device)

                # beam decode
                pred = beam_search_decode(model, src_ids, src_tok, tgt_tok, beam_width=5, device=device)

                print(f"Source (ES): {src}")
                print(f"Prediction (EN): {pred}")
                print("-" * 30)

            except Exception as e:
                print(f"❌ ERROR translating: {src}. Details: {e}")

if __name__ == "__main__":
    translate()