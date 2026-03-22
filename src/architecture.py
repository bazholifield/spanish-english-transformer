from utils import create_transformer_masks
import torch
import math
import torch.nn as nn

# Pytorch transformer
class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, pad_idx=0):
        super().__init__()
        self.model_type = "transformer"
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=self.pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=self.pad_idx)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):

        src = src.long()
        tgt = tgt.long()

        # create masks with single padding index
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_transformer_masks(
            src, tgt, pad_idx=self.pad_idx
        )

        # embeddings + positional encoding
        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        # transformer forward
        out = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask
        )

        logits = self.fc_out(out)  # (batch, tgt_len, vocab)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)
