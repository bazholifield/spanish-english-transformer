"""Score a trained run on the four fixed per-register test sets, with greedy decoding and beam search.

    python eval.py --model runs/local/model             # all 2,000 test sentences
    python eval.py --model runs/local/model --clean     # the 1,800 used in the README (see fix 4)
"""
import argparse
from pathlib import Path

import sacrebleu
import torch

from build_mix import norm
from src.decode import beam_search, greedy_batch, load_model

EVAL = Path("data/data/mix-v2/eval")
LEAKED_INTO = Path("data/data/mix-v1/pools")  # run 3's training data held the English of ~200 test pairs


def load_sets(clean):
    seen = set()
    if clean:
        for f in LEAKED_INTO.glob("*.en"):
            seen.update(norm(line) for line in f.read_text(encoding="utf-8").splitlines())
    sets = {}
    for es_file in sorted(EVAL.glob("*.es")):
        pairs = zip(es_file.read_text(encoding="utf-8").splitlines(),
                    es_file.with_suffix(".en").read_text(encoding="utf-8").splitlines())
        pairs = [(es, en) for es, en in pairs if norm(en) not in seen]
        sets[es_file.stem] = ([es for es, _ in pairs], [en for _, en in pairs])
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="run folder with best_transformer.pt and spm_shared.model")
    ap.add_argument("--clean", action="store_true", help="drop the ~200 test pairs whose English leaked into mix-v1")
    ap.add_argument("--beam", type=int, default=5, help="beam width (0 = greedy only)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="first N sentences per register (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.model, device)
    sets = load_sets(args.clean)
    if args.limit:
        sets = {k: (es[:args.limit], en[:args.limit]) for k, (es, en) in sets.items()}
    decoders = {"greedy": lambda s: greedy_batch(model, s, tok, device)}
    if args.beam:
        decoders[f"beam {args.beam}"] = lambda s: beam_search(model, s, tok, device, beam=args.beam, alpha=args.alpha)

    for name, decode in decoders.items():
        all_hyps, all_refs = [], []
        print(f"\n{name}:")
        for register, (es, refs) in sets.items():
            hyps = decode(es)
            print(f"  {register:10} ({len(es)}): BLEU {sacrebleu.corpus_bleu(hyps, [refs]).score:5.2f}  "
                  f"chrF {sacrebleu.corpus_chrf(hyps, [refs]).score:5.2f}", flush=True)
            all_hyps += hyps
            all_refs += refs
        print(f"  {'all':10} ({len(all_refs)}): BLEU {sacrebleu.corpus_bleu(all_hyps, [all_refs]).score:5.2f}  "
              f"chrF {sacrebleu.corpus_chrf(all_hyps, [all_refs]).score:5.2f}")


if __name__ == "__main__":
    main()
