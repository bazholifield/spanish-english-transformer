import os, math, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .architecture import TransformerSeq2Seq
from .tokenizer import SPTokenizer
from .dataset import Seq2SeqDataset, collate_fn
from .utils import shift_right

def train_epoch(model, dataloader, optimizer, criterion, device, tgt_bos_id, scaler, scheduler=None):
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        optimizer.zero_grad()

        decoder_input = shift_right(tgt_batch, tgt_bos_id)

        with torch.cuda.amp.autocast():
            logits = model(src_batch, decoder_input)
            V = logits.size(-1)
            logits_flat = logits.view(-1, V)
            tgt_flat = tgt_batch.view(-1)
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
            decoder_input = shift_right(tgt_batch, tgt_bos_id)
            logits = model(src_batch, decoder_input)
            V = logits.size(-1)
            logits_flat = logits.view(-1, V)
            tgt_flat = tgt_batch.view(-1)
            loss = criterion(logits_flat, tgt_flat)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def run_training(src_sentences_train, tgt_sentences_train,
                 src_sentences_val, tgt_sentences_val,
                 sp_model_path, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sp = SPTokenizer(sp_model_path)
    src_tok, tgt_tok = sp, sp

    try:
        pad_idx = src_tok.piece_to_id("<pad>")
        bos_idx = tgt_tok.piece_to_id("<s>")
        eos_idx = tgt_tok.piece_to_id("</s>")
    except AttributeError:
        pad_idx, bos_idx, eos_idx = 0, 1, 2

    pad_idx = src_tok.id_pad if src_tok.id_pad is not None else 0
    if pad_idx < 0:
        pad_idx = 0

    print("Special token IDs -- PAD:", pad_idx, "BOS:", bos_idx, "EOS:", eos_idx)
    print("Tokenizer vocab size -- SRC:", src_tok.vocab_size(), "TGT:", tgt_tok.vocab_size())

    train_ds = Seq2SeqDataset(src_sentences_train, tgt_sentences_train, src_tok, tgt_tok)
    val_ds   = Seq2SeqDataset(src_sentences_val, tgt_sentences_val, src_tok, tgt_tok)
    BATCH = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

    src_vocab_size = src_tok.vocab_size()
    tgt_vocab_size = tgt_tok.vocab_size()
    print("SRC vocab size:", src_vocab_size, "TGT vocab size:", tgt_vocab_size)

    print("Initializing model on CPU first for safety...")
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.05,
        pad_idx=pad_idx
    )

    print("Moving model to device:", device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scaler = torch.cuda.amp.GradScaler()

    def lr_lambda(step):
        warmup = 8000
        step = max(step, 1)
        return (1.0 / math.sqrt(512)) * min(1.0 / math.sqrt(step), step * (warmup ** -1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_path = os.path.join("models", "models", "best_transformer.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    N_EPOCHS = 40
    best_val = float('inf')
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, bos_idx, scaler, scheduler)
        val_loss = evaluate(model, val_loader, criterion, device, bos_idx)
        t1 = time.time()
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={t1-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            checkpoint_data = {
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(checkpoint_data, save_path)
            print(f" > New best model saved to {save_path} (Val Loss: {val_loss:.4f})")

    return model, src_tok, tgt_tok
