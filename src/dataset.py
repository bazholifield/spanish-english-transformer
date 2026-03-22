import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SPTokenizer
from typing import List

# Dataset + collate
class Seq2SeqDataset(Dataset):
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str],
                 src_tokenizer: SPTokenizer, tgt_tokenizer: SPTokenizer, max_len=128):
        assert len(src_sentences) == len(tgt_sentences)
        self.src = src_sentences
        self.tgt = tgt_sentences
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len
        self.src_pad = self.src_tok.id_pad if self.src_tok.id_pad is not None else 0
        self.tgt_pad = self.tgt_tok.id_pad if self.tgt_tok.id_pad is not None else 0

        if self.src_pad < 0:
            self.src_pad = 0
        if self.tgt_pad < 0:
            self.tgt_pad = 0

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = self.src_tok.encode(self.src[idx])[:self.max_len]
        tgt_ids = self.tgt_tok.encode(self.tgt[idx])[:self.max_len]

        src_ids = [min(max(x, 0), self.src_tok.vocab_size()-1) for x in src_ids]
        tgt_ids = [min(max(x, 0), self.tgt_tok.vocab_size()-1) for x in tgt_ids]

        src_ids = [x if x >= 0 else self.src_pad for x in src_ids]
        tgt_ids = [x if x >= 0 else self.tgt_pad for x in tgt_ids]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, pad_idx): 
    device = batch[0][0].device 

    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    max_src = max(len(s) for s in src_batch)
    max_tgt = max(len(t) for t in tgt_batch)

    src_tensor = torch.full((len(batch), max_src), fill_value=pad_idx, dtype=torch.long)
    tgt_tensor = torch.full((len(batch), max_tgt), fill_value=pad_idx, dtype=torch.long)

    for i, (src_ids, tgt_ids) in enumerate(batch):
        src_tensor[i, :len(src_ids)] = src_ids
        tgt_tensor[i, :len(tgt_ids)] = tgt_ids

    return src_tensor.to(device), tgt_tensor.to(device)
