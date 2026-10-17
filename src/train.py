import os, math, shutil, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .architecture import TransformerSeq2Seq
from .tokenizer import SPTokenizer
from .dataset import Seq2SeqDataset, collate_fn

def teacher_forcing(tgt_batch):
    """Split encoded targets [<s>, w1 … wn, </s>] into decoder input [<s>, w1 … wn] and labels
    [w1 … wn, </s>]. (shift_right on the full target fed <s>, <s> and taught the model to predict
    <s> first, so it never started a sentence on its own.)"""
    return tgt_batch[:, :-1], tgt_batch[:, 1:]


def train_epoch(model, dataloader, optimizer, criterion, device, tgt_bos_id, scaler, scheduler=None):
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        optimizer.zero_grad()

        decoder_input, labels = teacher_forcing(tgt_batch)

        with torch.cuda.amp.autocast():
            logits = model(src_batch, decoder_input)
            V = logits.size(-1)
            logits_flat = logits.reshape(-1, V)
            tgt_flat = labels.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, tgt_bos_id):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            decoder_input, labels = teacher_forcing(tgt_batch)
            logits = model(src_batch, decoder_input)
            V = logits.size(-1)
            logits_flat = logits.reshape(-1, V)
            tgt_flat = labels.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)
            total_loss += loss.item()
    return total_loss / len(dataloader)

try:  # live curves when launched from the Helm (helm/training/launch.py); a no-op otherwise
    from helm_metrics import emit
except ImportError:
    def emit(record):
        pass


# Defaults reproduce the original hardcoded run; a config (configs/*.yaml) overrides any of them.
DEFAULTS = {
    "d_model": 512, "nhead": 8, "num_encoder_layers": 6, "num_decoder_layers": 6,
    "dim_feedforward": 2048, "dropout": 0.05, "batch_size": 64, "epochs": 40,
    "warmup_steps": 8000, "label_smoothing": 0.1, "bleu_every": 0, "bleu_sentences": 200,
    "lr_scale": 1.0,             # multiplies the whole Noam schedule (0.5 halves the peak)
    "early_stop_patience": 0,    # stop after N epochs without a new best val loss (0 = never)
    "save_dir": os.path.join("models", "models"),
}


@torch.no_grad()
def greedy_batch(model, sentences, tok, device, max_len=None, batch_size=64):
    """Greedy-decode a list of sentences in batches, cutting each output at its own </s>."""
    model.eval()
    out = []
    for i in range(0, len(sentences), batch_size):
        chunk = [tok.encode(s) for s in sentences[i:i + batch_size]]
        src = torch.full((len(chunk), max(map(len, chunk))), tok.id_pad, dtype=torch.long)
        for j, ids in enumerate(chunk):
            src[j, :len(ids)] = torch.tensor(ids)
        src = src.to(device)
        gen = torch.full((len(chunk), 1), tok.id_bos, dtype=torch.long, device=device)
        limit = max_len or (2 * src.size(1) + 10)
        for _ in range(limit):
            logits = model(src, gen)[:, -1]
            logits[:, [tok.id_bos, tok.id_pad]] = float("-inf")  # never valid as output tokens
            nxt = logits.argmax(-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=1)
            if (gen == tok.id_eos).any(dim=1).all():
                break
        for row in gen.tolist():
            row = row[1:]
            out.append(tok.decode(row[:row.index(tok.id_eos)] if tok.id_eos in row else row))
    return out


def run_training(src_sentences_train, tgt_sentences_train,
                 src_sentences_val, tgt_sentences_val,
                 sp_model_path, device=None, cfg=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sp = SPTokenizer(sp_model_path)
    src_tok, tgt_tok = sp, sp
    pad_idx, bos_idx, eos_idx = src_tok.id_pad, tgt_tok.id_bos, tgt_tok.id_eos

    print("Special token IDs -- PAD:", pad_idx, "BOS:", bos_idx, "EOS:", eos_idx)
    print("Tokenizer vocab size -- SRC:", src_tok.vocab_size(), "TGT:", tgt_tok.vocab_size())

    train_ds = Seq2SeqDataset(src_sentences_train, tgt_sentences_train, src_tok, tgt_tok)
    val_ds   = Seq2SeqDataset(src_sentences_val, tgt_sentences_val, src_tok, tgt_tok)
    BATCH = cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

    src_vocab_size = src_tok.vocab_size()
    tgt_vocab_size = tgt_tok.vocab_size()
    print("SRC vocab size:", src_vocab_size, "TGT vocab size:", tgt_vocab_size)

    arch = {k: cfg[k] for k in ("d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward")}
    print("Initializing model on CPU first for safety...")
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dropout=cfg["dropout"], pad_idx=pad_idx, **arch
    )

    print("Moving model to device:", device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=cfg["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scaler = torch.cuda.amp.GradScaler()

    def lr_lambda(step):
        warmup = cfg["warmup_steps"]
        step = max(step, 1)
        return cfg["lr_scale"] * (1.0 / math.sqrt(cfg["d_model"])) * min(1.0 / math.sqrt(step), step * (warmup ** -1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = cfg["save_dir"]
    save_path = os.path.join(save_dir, "best_transformer.pt")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.abspath(sp_model_path) != os.path.abspath(os.path.join(save_dir, "spm_shared.model")):
        shutil.copy(sp_model_path, os.path.join(save_dir, "spm_shared.model"))  # keep model + tokenizer together

    N_EPOCHS = cfg["epochs"]
    steps_per_epoch = len(train_loader)
    emit({"event": "start", "max_steps": N_EPOCHS * steps_per_epoch, "epochs": N_EPOCHS, "batch_size": BATCH,
          "device": str(device), "train_pairs": len(train_ds), "val_pairs": len(val_ds)})
    best_val = float('inf')
    stale = 0  # epochs since the last new best
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, bos_idx, scaler, scheduler)
        val_loss = evaluate(model, val_loader, criterion, device, bos_idx)
        t1 = time.time()
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={t1-t0:.1f}s")
        step = epoch * steps_per_epoch
        record = {"step": step, "epoch": epoch, "loss": train_loss, "eval_loss": val_loss,
                  "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": round(t1 - t0, 1)}

        if cfg["bleu_every"] and (epoch % cfg["bleu_every"] == 0 or epoch == N_EPOCHS):
            import sacrebleu
            n = cfg["bleu_sentences"]
            hyps = greedy_batch(model, list(src_sentences_val[:n]), tgt_tok, device)
            record["eval_bleu"] = sacrebleu.corpus_bleu(hyps, [list(tgt_sentences_val[:n])]).score
            print(f"  val BLEU ({n} sentences, greedy): {record['eval_bleu']:.2f}")
        emit(record)

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            # Diverged (fp16 overflow / too high a learning rate). Every later epoch would be NaN
            # too, so stop here; the best checkpoint saved so far is untouched.
            print(f" ! Loss is not finite at epoch {epoch}; stopping. Best val loss was {best_val:.4f}.")
            emit({"event": "diverged", "step": step, "epoch": epoch})
            break

        if val_loss < best_val:
            best_val = val_loss
            checkpoint_data = {
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                # Everything inference needs to rebuild this model exactly:
                'special_ids': {'pad': pad_idx, 'bos': bos_idx, 'eos': eos_idx},
                'arch': arch,
                'decoder_input': 'standard',  # [<s>, w1 …] → [w1 …, </s>]; decode from [<s>]
            }
            torch.save(checkpoint_data, save_path)
            print(f" > New best model saved to {save_path} (Val Loss: {val_loss:.4f})")
            emit({"event": "checkpoint", "step": step, "best": save_path})
            stale = 0
        else:
            stale += 1
            if cfg["early_stop_patience"] and stale >= cfg["early_stop_patience"]:
                print(f" > No improvement for {stale} epochs; stopping early (best val loss {best_val:.4f}).")
                emit({"event": "early_stop", "step": step, "epoch": epoch})
                break

    return model, src_tok, tgt_tok
