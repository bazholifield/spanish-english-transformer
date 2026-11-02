"""Turning Spanish sentences into English: greedy decoding and beam search, both batched.

Greedy picks the single most likely next piece at every step. Beam search keeps the `beam` best
partial translations per sentence and, at each step, extends each of them by its likeliest next
pieces, keeping the best `beam` overall. Two details decide whether beam search helps or hurts:

- A hypothesis that has produced </s> is finished: it is set aside and never extended again.
  (The old version in utils.py kept extending finished beams, so its output ran on.)
- Every extra piece multiplies in another probability below 1, so raw scores always favour
  shorter output. Finished hypotheses are compared by log-probability / length**alpha;
  alpha = 0 is no correction, alpha = 1 is the per-piece average.

Both decoders sort sentences by length before batching (less padding, much faster on CPU) and
return translations in the original order.
"""

from __future__ import annotations

import os

import torch


def load_model(model_dir, device="cpu"):
    """A run folder (best_transformer.pt + spm_shared.model, as main.py writes it) → (model, tokenizer)."""
    from .architecture import TransformerSeq2Seq
    from .tokenizer import SPTokenizer

    tok = SPTokenizer(os.path.join(model_dir, "spm_shared.model"))
    ckpt = torch.load(os.path.join(model_dir, "best_transformer.pt"), map_location=device)
    if ckpt.get("decoder_input") != "standard":
        raise ValueError(f"{model_dir} was trained before the October 2026 fixes; open it in the Helm, "
                         "which knows how each old checkpoint has to be decoded")
    tok.id_pad, tok.id_bos, tok.id_eos = (ckpt["special_ids"][k] for k in ("pad", "bos", "eos"))
    model = TransformerSeq2Seq(tok.vocab_size(), tok.vocab_size(), dropout=0.0, pad_idx=tok.id_pad,
                               **ckpt["arch"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval(), tok


def _batches(sentences, tok, batch_size):
    """Yield (original indices, padded source tensor) for length-sorted chunks."""
    encoded = [tok.encode(s) for s in sentences]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        src = torch.full((len(idx), max(len(encoded[i]) for i in idx)), tok.id_pad, dtype=torch.long)
        for row, i in enumerate(idx):
            src[row, :len(encoded[i])] = torch.tensor(encoded[i])
        yield idx, src


def _cut(tok, ids):
    return tok.decode(ids[:ids.index(tok.id_eos)] if tok.id_eos in ids else ids)


@torch.no_grad()
def greedy_batch(model, sentences, tok, device, max_len=None, batch_size=64, prefix=None):
    """Greedy-decode a list of sentences in batches, cutting each output at its own </s>.
    `prefix` is the decoder's start sequence (default [<s>])."""
    model.eval()
    prefix = prefix or [tok.id_bos]
    out = [""] * len(sentences)
    for idx, src in _batches(sentences, tok, batch_size):
        src = src.to(device)
        memory, src_pad = model.encode(src)
        gen = torch.tensor([prefix] * len(idx), dtype=torch.long, device=device)
        for _ in range(max_len or (2 * src.size(1) + 10)):
            logits = model.decode_step(gen, memory, src_pad)
            logits[:, [tok.id_bos, tok.id_pad]] = float("-inf")  # never valid as output tokens
            gen = torch.cat([gen, logits.argmax(-1, keepdim=True)], dim=1)
            if (gen == tok.id_eos).any(dim=1).all():
                break
        for i, row in zip(idx, gen.tolist()):
            out[i] = _cut(tok, row[len(prefix):])
    return out


@torch.no_grad()
def beam_search(model, sentences, tok, device, beam=5, alpha=1.0, max_len=None, batch_size=16, prefix=None):
    """Beam-search a list of sentences in batches; returns the best translation of each."""
    model.eval()
    prefix = prefix or [tok.id_bos]
    out = [""] * len(sentences)
    for idx, src in _batches(sentences, tok, batch_size):
        B, K = len(idx), beam
        src = src.to(device)
        memory, src_pad = model.encode(src)
        memory, src_pad = memory.repeat_interleave(K, 0), src_pad.repeat_interleave(K, 0)  # row b*K+k

        seqs = torch.tensor([prefix] * (B * K), dtype=torch.long, device=device)
        scores = torch.full((B, K), float("-inf"), device=device)
        scores[:, 0] = 0.0  # all K rows start identical; only one may expand at first
        finished = [[] for _ in range(B)]  # per sentence: (normalised score, token ids)
        done = [False] * B
        limit = max_len or (2 * src.size(1) + 10)

        for step in range(1, limit + 1):
            logits = model.decode_step(seqs, memory, src_pad)
            logits[:, [tok.id_bos, tok.id_pad]] = float("-inf")
            logp = torch.log_softmax(logits.float(), dim=-1).view(B, K, -1)
            V = logp.size(-1)
            # 2K candidates per sentence: enough to still have K unfinished ones if some end here
            top_scores, top_ids = (scores.unsqueeze(-1) + logp).view(B, -1).topk(2 * K, dim=-1)

            new_rows, new_tokens = [], []
            new_scores = torch.full((B, K), float("-inf"), device=device)
            for b in range(B):
                kept = 0
                for score, flat in zip(top_scores[b].tolist(), top_ids[b].tolist()):
                    if done[b] or kept == K or score == float("-inf"):
                        break
                    k, token = divmod(flat, V)
                    if token == tok.id_eos:
                        if len(finished[b]) < K:
                            hyp = seqs[b * K + k, len(prefix):].tolist()
                            finished[b].append((score / step ** alpha, hyp))
                        continue
                    new_rows.append(b * K + k)
                    new_tokens.append(token)
                    new_scores[b, kept] = score
                    kept += 1
                # pad the beam with dead rows so every sentence keeps exactly K rows
                new_rows += [b * K] * (K - kept)
                new_tokens += [tok.id_eos] * (K - kept)
                if len(finished[b]) >= K:
                    done[b] = True
            if all(done):
                break
            seqs = torch.cat([seqs[new_rows], torch.tensor(new_tokens, device=device).unsqueeze(1)], dim=1)
            scores = new_scores

        for b, i in enumerate(idx):
            hyps = finished[b]
            if not hyps:  # hit the length limit with nothing finished: take the best unfinished beam
                hyps = [(scores[b, 0].item() / limit ** alpha, seqs[b * K, len(prefix):].tolist())]
            out[i] = _cut(tok, max(hyps)[1])
    return out
