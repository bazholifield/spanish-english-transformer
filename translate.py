"""Translate Spanish to English with a trained run (beam search by default).

    python translate.py --model runs/local/model "Llegué tarde porque se me olvidó el despertador."
    python translate.py --model runs/local/model < spanish.txt > english.txt
"""
import argparse
import sys

import torch

from src.decode import beam_search, greedy_batch, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentences", nargs="*", help="Spanish sentences (default: one per line from stdin)")
    ap.add_argument("--model", required=True, help="run folder with best_transformer.pt and spm_shared.model")
    ap.add_argument("--beam", type=int, default=5, help="beam width (1 = greedy)")
    ap.add_argument("--alpha", type=float, default=1.0, help="length normalisation for beam search (tuned on validation)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.model, device)
    sentences = args.sentences or [line.strip() for line in sys.stdin if line.strip()]
    if args.beam > 1:
        out = beam_search(model, sentences, tok, device, beam=args.beam, alpha=args.alpha)
    else:
        out = greedy_batch(model, sentences, tok, device)
    print("\n".join(out))


if __name__ == "__main__":
    main()
