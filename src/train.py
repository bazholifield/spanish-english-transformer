import os, math, random, time
from utils import shift_right
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from architecture import TransformerSeq2Seq
from tokenizer import SPTokenizer
from dataset import Seq2SeqDataset, collate_fn
from utils import create_transformer_masks, shift_right

def train_epoch(model, dataloader, optimizer, criterion, device, tgt_bos_id, scaler):
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        optimizer.zero_grad()

        decoder_input = shift_right(tgt_batch, tgt_bos_id)  # feed shifted right tgt tokens

        # Forward pass in mixed precision
        with torch.cuda.amp.autocast():
            logits = model(src_batch, decoder_input)  # (B, T, V)
            V = logits.size(-1)
            logits_flat = logits.view(-1, V)
            tgt_flat = tgt_batch.view(-1)
            loss = criterion(logits_flat, tgt_flat)

        # Backward with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

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

# Training loop
def run_training(src_sentences_train, tgt_sentences_train,
                 src_sentences_val, tgt_sentences_val,
                 sp_model_path, device=None):


    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizers
    sp = SPTokenizer(sp_model_path)
    src_tok, tgt_tok = sp, sp 

    try:
        pad_idx = src_tok.piece_to_id("<pad>")
        bos_idx = tgt_tok.piece_to_id("<s>")
        eos_idx = tgt_tok.piece_to_id("</s>")
    except AttributeError:
        pad_idx, bos_idx, eos_idx = 0, 1, 2

    pad_idx = src_tok.id_pad if src_tok.id_pad is not None else 0 # This should be 3
    pad_idx = src_tok.id_pad if src_tok.id_pad is not None else 0
    if pad_idx < 0:
       pad_idx = 0


    print("Special token IDs -- PAD:", pad_idx, "BOS:", bos_idx, "EOS:", eos_idx)
    print("Tokenizer vocab size -- SRC:", src_tok.vocab_size(), "TGT:", tgt_tok.vocab_size())

    # Dataset + Dataloader
    train_ds = Seq2SeqDataset(src_sentences_train, tgt_sentences_train, src_tok, tgt_tok)
    val_ds   = Seq2SeqDataset(src_sentences_val, tgt_sentences_val, src_tok, tgt_tok)
    BATCH = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

    # Precheck  dataset IDs
    def check_dataset(ds, vocab_size, name="Dataset"):
        for i, (src_ids, tgt_ids) in enumerate(ds):
            if src_ids.max() >= vocab_size or src_ids.min() < 0:
                print(f"{name} -- source IDs out of bounds at index {i}: min={src_ids.min()} max={src_ids.max()}")
            if tgt_ids.max() >= vocab_size or tgt_ids.min() < 0:
                print(f"{name} -- target IDs out of bounds at index {i}: min={tgt_ids.min()} max={tgt_ids.max()}")
        print(f"{name} check complete: all IDs within [0, {vocab_size-1}]")

    check_dataset(train_ds, src_tok.vocab_size(), "Train SRC")
    check_dataset(train_ds, tgt_tok.vocab_size(), "Train TGT")
    check_dataset(val_ds, src_tok.vocab_size(), "Val SRC")
    check_dataset(val_ds, tgt_tok.vocab_size(), "Val TGT")

    # CPU checks for min/max IDs
    all_src_ids = torch.cat([s for s, _ in train_ds] + [s for s, _ in val_ds])
    all_tgt_ids = torch.cat([t for _, t in train_ds] + [t for _, t in val_ds])

    print("CPU check: SRC min/max:", all_src_ids.min().item(), all_src_ids.max().item())
    print("CPU check: TGT min/max:", all_tgt_ids.min().item(), all_tgt_ids.max().item())

    src_vocab_size = src_tok.vocab_size()
    tgt_vocab_size = tgt_tok.vocab_size()
    print("Safe SRC vocab size:", src_vocab_size, "Safe TGT vocab size:", tgt_vocab_size)

    # Model
    print("Initializing model on CPU first for safety...")
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.05,
        pad_idx=pad_idx
    )

    # Move to GPU
    print("Moving model to device:", device)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scaler = torch.cuda.amp.GradScaler()

    def lr_lambda(step):
        warmup = 8000
        step = max(step, 1)
        return (1.0 / math.sqrt(512)) * min(1.0/math.sqrt(step), step*(warmup**-1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # batch check
    src_batch, tgt_batch = next(iter(train_loader))
    print("DEBUG: train batch shapes:", src_batch.shape, tgt_batch.shape)
    print("DEBUG: train batch max SRC ID:", src_batch.max().item())
    print("DEBUG: train batch max TGT ID:", tgt_batch.max().item())

    # Training loop
    N_EPOCHS = 40
    best_val = float('inf')
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, bos_idx)
        scheduler.step()
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

            save_path = os.path.join("checkpoints", "best_transformer.pt")
            torch.save(checkpoint_data, save_path)
            
            print(f" > New best model saved to {save_path} (Val Loss: {val_loss:.4f})")

    return model, src_tok, tgt_tok