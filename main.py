import argparse
import os
import random
from pathlib import Path

import torch

from src.train import greedy_batch, run_training


def load_parallel_corpus(en_path, es_path):
    """Helper to load text files into lists."""
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]
    with open(es_path, 'r', encoding='utf-8') as f:
        es_lines = [line.strip() for line in f]
    return en_lines, es_lines


def main():
    """Original local run: data/data/processed + models/models/spm_shared.model, 90/10 split."""
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


# ---------------------------------------------------------------------------
# Config-driven run (python main.py --config configs/retrain.yaml)
# ---------------------------------------------------------------------------

def build_pairs(cfg):
    """Read the raw Europarl files and clean them, as data/data/clean.py does, but with the
    number of lines and the length limit taken from the config (clean.py used 20,000 / 120)."""
    raw = cfg["raw_dir"]
    en_path = os.path.join(raw, "europarl-v7.es-en.en")
    es_path = os.path.join(raw, "europarl-v7.es-en.es")
    pairs = []
    with open(en_path, encoding="utf-8") as f_en, open(es_path, encoding="utf-8") as f_es:
        for i, (en, es) in enumerate(zip(f_en, f_es)):
            if i >= cfg["max_lines"]:
                break
            en, es = " ".join(en.split()), " ".join(es.split())
            if en and es and len(en) <= cfg["max_chars"] and len(es) <= cfg["max_chars"]:
                pairs.append((es, en))  # src=ES, tgt=EN
    return pairs


def train_tokenizer(cfg, train_pairs, out_dir):
    """Same settings as prepare_data.py, trained on this run's training split only."""
    import sentencepiece as spm

    corpus = os.path.join(out_dir, "spm_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for es, en in train_pairs:
            f.write(es + "\n" + en + "\n")
    prefix = os.path.join(out_dir, "spm_shared")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=prefix, vocab_size=cfg["vocab_size"], model_type="bpe",
        character_coverage=1.0, input_sentence_size=1000000, shuffle_input_sentence=True,
        user_defined_symbols="<pad>,<s>,</s>",
    )
    os.remove(corpus)
    return prefix + ".model"


def load_config(path):
    """YAML when PyYAML is installed (Colab has it); JSON otherwise. The Helm writes JSON run
    configs for this project, so local runs need no extra package."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        import json
        return json.loads(text)


def build_mix_pairs(cfg):
    """A multi-domain mix (configs/mix-v1.yaml): the first max_pairs of each pool that
    build_mix.py wrote to <data_root>/pools/. Pools are pre-shuffled, so "first N" is a sample."""
    root, pairs, counts = cfg["data_root"], [], {}
    for d in cfg["datasets"]:
        with open(os.path.join(root, "pools", d["name"] + ".es"), encoding="utf-8") as f_es, \
             open(os.path.join(root, "pools", d["name"] + ".en"), encoding="utf-8") as f_en:
            chunk = [(es.rstrip("\n"), en.rstrip("\n")) for es, en, _ in zip(f_es, f_en, range(d["max_pairs"]))]
        pairs += chunk
        counts[d["name"]] = len(chunk)
    return pairs, counts


def load_eval_sets(root):
    """The fixed per-domain evaluation sets: {domain: (spanish lines, english lines)}."""
    sets = {}
    for es_file in sorted(Path(root, "eval").glob("*.es")):
        es = es_file.read_text(encoding="utf-8").splitlines()
        en = es_file.with_suffix(".en").read_text(encoding="utf-8").splitlines()
        sets[es_file.stem] = (es, en)
    return sets


def run_from_config(path):
    cfg = load_config(path)
    out_dir = cfg["output_dir"]
    model_dir = os.path.join(out_dir, "model")  # what Sea trials loads: checkpoint + tokenizer + test set
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_sets = {}
    if cfg.get("datasets"):
        # Multi-domain mix: the test sets are the fixed per-domain eval sets, not cut from the data.
        pairs, counts = build_mix_pairs(cfg)
        eval_sets = load_eval_sets(cfg["data_root"])
        n_test = 0
        print("Mix:", ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    else:
        pairs = build_pairs(cfg)
        n_test = cfg["test_size"]
    random.seed(cfg["seed"])
    random.shuffle(pairs)
    n_val = int(len(pairs) * cfg["val_fraction"])
    test, val, train = pairs[:n_test], pairs[n_test:n_test + n_val], pairs[n_test + n_val:]
    print(f"{len(pairs)} pairs: train {len(train)}, val {len(val)}, test {len(test) or sum(len(v[0]) for v in eval_sets.values())}")
    if eval_sets:  # keep the eval sets with the model too, like test_es.txt for single-corpus runs
        os.makedirs(os.path.join(model_dir, "eval"), exist_ok=True)
        for name, (es, en) in eval_sets.items():
            Path(model_dir, "eval", name + ".es").write_text("\n".join(es) + "\n", encoding="utf-8")
            Path(model_dir, "eval", name + ".en").write_text("\n".join(en) + "\n", encoding="utf-8")

    # The test set is never trained on; it travels with the model so evaluations stay comparable.
    # (Mix runs are scored on the fixed eval sets instead, so they have no test file to write.)
    if test:
        with open(os.path.join(model_dir, "test_es.txt"), "w", encoding="utf-8") as f_es, \
             open(os.path.join(model_dir, "test_en.txt"), "w", encoding="utf-8") as f_en:
            for es, en in test:
                f_es.write(es + "\n")
                f_en.write(en + "\n")

    spm_path = train_tokenizer(cfg, train, model_dir) if cfg["retrain_tokenizer"] else cfg["tokenizer"]

    train_cfg = {k: cfg[k] for k in ("d_model", "nhead", "num_encoder_layers", "num_decoder_layers",
                                     "dim_feedforward", "dropout", "batch_size", "epochs", "warmup_steps",
                                     "label_smoothing", "bleu_every", "bleu_sentences", "lr_scale",
                                     "early_stop_patience", "norm_first", "spike_factor", "max_rollbacks",
                                     "report_every") if k in cfg}
    train_cfg["save_dir"] = model_dir
    run_training(
        src_sentences_train=[es for es, _ in train], tgt_sentences_train=[en for _, en in train],
        src_sentences_val=[es for es, _ in val], tgt_sentences_val=[en for _, en in val],
        sp_model_path=spm_path, device=device, cfg=train_cfg,
    )

    # Final score: the *best* checkpoint (lowest validation loss) on the held-out test set.
    import sacrebleu
    from src.architecture import TransformerSeq2Seq
    from src.tokenizer import SPTokenizer
    from src.train import emit

    tok = SPTokenizer(os.path.join(model_dir, "spm_shared.model"))
    ckpt = torch.load(os.path.join(model_dir, "best_transformer.pt"), map_location=device)
    model = TransformerSeq2Seq(tok.vocab_size(), tok.vocab_size(), dropout=0.0, pad_idx=tok.id_pad, **ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    if not eval_sets:
        eval_sets = {"test": ([es for es, _ in test], [en for _, en in test])}
    record, all_hyps, all_refs = {"step": ckpt["epoch"]}, [], []
    print(f"\nBest epoch {ckpt['epoch']}, greedy decoding:")
    for name, (es, refs) in eval_sets.items():
        hyps = greedy_batch(model, es, tok, device)
        bleu, chrf = sacrebleu.corpus_bleu(hyps, [refs]).score, sacrebleu.corpus_chrf(hyps, [refs]).score
        print(f"  {name:10} ({len(es)} sentences): BLEU {bleu:.2f}  chrF {chrf:.2f}")
        if len(eval_sets) > 1:
            record[f"test_bleu_{name}"], record[f"test_chrf_{name}"] = bleu, chrf
        all_hyps += hyps
        all_refs += refs
        for s, hyp, ref in list(zip(es, hyps, refs))[:2]:
            print(f"    ES  {s}\n    OUT {hyp}\n    REF {ref}\n")
    record["test_bleu"] = sacrebleu.corpus_bleu(all_hyps, [all_refs]).score
    record["test_chrf"] = sacrebleu.corpus_chrf(all_hyps, [all_refs]).score
    print(f"  overall: BLEU {record['test_bleu']:.2f}  chrF {record['test_chrf']:.2f}")
    emit(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ES→EN Transformer.")
    parser.add_argument("--config", help="YAML config (see configs/retrain.yaml); omit for the original run")
    args = parser.parse_args()
    if args.config:
        run_from_config(args.config)
    else:
        main()
