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

# Greedy + Beam decode
@torch.no_grad()
def greedy_decode(model, src_tensor, src_tokenizer, tgt_tokenizer, max_len=100, device='cpu'):
    model.eval()
    src = src_tensor.to(device).unsqueeze(0) if src_tensor.dim()==1 else src_tensor.to(device)

    # Initialize special token IDs and batch size
    bos = tgt_tokenizer.id_bos if tgt_tokenizer.id_bos is not None else 1
    eos = tgt_tokenizer.id_eos if tgt_tokenizer.id_eos is not None else 2
    pad = tgt_tokenizer.id_pad if tgt_tokenizer.id_pad is not None else 0
    B = src.size(0)

    # Get the max valid index (Vocab size - 1)
    V = tgt_tokenizer.vocab_size()
    max_index = V - 1

    generated = torch.full((B, 1), fill_value=bos, dtype=torch.long, device=device)

    for step in range(max_len):
        logits = model(src, generated)
        next_token_logits = logits[:, -1, :] 
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)

        next_tokens = torch.clamp(next_tokens, max=max_index)

        generated = torch.cat([generated, next_tokens], dim=1)

        if (next_tokens == eos).all():
            break

    out = []
    for i in range(B):
        ids = generated[i].tolist()
        out.append(tgt_tokenizer.decode(ids))

    return out

@torch.no_grad()
def beam_search_decode(model, src_tensor, src_tokenizer, tgt_tokenizer, beam_width=5, max_len=100, device='cpu'):
    model.eval()
    src = src_tensor.to(device).unsqueeze(0) if src_tensor.dim()==1 else src_tensor.to(device)
    bos = tgt_tokenizer.id_bos if tgt_tokenizer.id_bos is not None else 1
    eos = tgt_tokenizer.id_eos if tgt_tokenizer.id_eos is not None else 2
    pad = tgt_tokenizer.id_pad if tgt_tokenizer.id_pad is not None else 0
    vocab_size = tgt_tokenizer.vocab_size()

    beams = [([bos], 0.0)]
    for _ in range(max_len):
        new_beams = []
        for tokens, score in beams:
            cur_input = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(src, cur_input)
            probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0) 
            topk_logp, topk_idx = probs.topk(beam_width)
            for k in range(beam_width):
                nt = topk_idx[k].item()
                ns = score + topk_logp[k].item()
                new_beams.append((tokens + [nt], ns))
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
        if all(b[0][-1] == eos for b in beams):
            break
    best_tokens = beams[0][0]
    return tgt_tokenizer.decode(best_tokens)
