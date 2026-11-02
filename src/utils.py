import torch

# Mask utilities
def create_transformer_masks(src, tgt, pad_idx):
    # src: (batch, src_len)
    # tgt: (batch, tgt_len)
    device = src.device
    src_pad_mask = (src == pad_idx)
    tgt_pad_mask = (tgt == pad_idx)
    # subsequent mask for target (prevent attending to future tokens)
    tgt_len = tgt.size(1)
    # PyTorch transformer expects mask with shape (tgt_len, tgt_len)
    tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()
    return None, tgt_mask, src_pad_mask, tgt_pad_mask

# Training + eval helper functions
def shift_right(tgt_batch, bos_id):
    # convert target [B, T] -> decoder_input [B, T] shifted right with BOS at start
    B, T = tgt_batch.size()
    dec_input = torch.full((B, T), fill_value=bos_id, dtype=torch.long, device=tgt_batch.device)
    dec_input[:, 1:] = tgt_batch[:, :-1]
    # keep pad where original pad
    return dec_input
