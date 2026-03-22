import sentencepiece as spm
from typing import List

# Tokenizer
class SPTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.id_bos = int(self.sp.bos_id())
        self.id_eos = int(self.sp.eos_id())
        self.id_pad = int(self.sp.pad_id())

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"Tokenizer expected string, got {type(text)}: {text}")

        ids = self.sp.encode(text, out_type=int)
        ids = [self.id_bos] + ids + [self.id_eos]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i not in (self.id_bos, self.id_eos, self.id_pad)]
        return self.sp.decode(ids)

    def vocab_size(self):
        return self.sp.get_piece_size()

    def piece_to_id(self, piece: str):
        return self.sp.piece_to_id(piece)
