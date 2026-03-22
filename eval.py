import torch
import sacrebleu
import os
from src.architecture import TransformerSeq2Seq
from src.tokenizer import SPTokenizer
from src.utils import greedy_decode

def load_test_data(en_path, es_path, max_samples=200):
    """Loads the parallel corpus and returns the test slice."""
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(es_path, 'r', encoding='utf-8') as f_es:
        en_lines = [line.strip() for line in f_en]
        es_lines = [line.strip() for line in f_es]
    
    test_en = en_lines[-max_samples:]
    test_es = es_lines[-max_samples:]
    return test_en, test_es

def run_evaluation():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizers
    sp_path = "models/spm_shared.model"
    if not os.path.exists(sp_path):
        raise FileNotFoundError(f"Tokenizer not found at {sp_path}")
    
    tokenizer = SPTokenizer(sp_path)
    src_tok = tgt_tok = tokenizer

    # Initialize model, load weights
    model = TransformerSeq2Seq(
        src_vocab_size=src_tok.vocab_size(),
        tgt_vocab_size=tgt_tok.vocab_size(),
        d_model=512, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6,
        dim_feedforward=2048, 
        dropout=0.0, 
        pad_idx=src_tok.id_pad
    ).to(device)

    checkpoint_path = "best_transformer.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found! BLEU score will be near zero.")

    # Load test tata
    test_en, test_es = load_test_data("data/all_en.txt", "data/all_es.txt", max_samples=200)

    # eval loop
    refs = []
    hyps = []

    print(f"Running evaluation on {len(test_es)} samples...")
    model.eval()
    with torch.no_grad():
        for i in range(len(test_es)):
            src = test_es[i]
            tgt = test_en[i]
            
            # Encode source
            src_ids = torch.tensor(src_tok.encode(src), dtype=torch.long).to(device)
            
            # greedy decode
            pred = greedy_decode(model, src_ids, src_tok, tgt_tok, device=device)[0]

            refs.append([tgt])
            hyps.append(pred)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_es)}...")

    # calculate BLEU
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    print("\n" + "="*30)
    print(f"FINAL BLEU SCORE: {bleu.score:.2f}")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()