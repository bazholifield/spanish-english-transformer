import sentencepiece as spm
from typing import List

# Tokenizer
class SPTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # prepare_data.py trains the tokenizer with <pad>, <s>, </s> as user-defined pieces, which
        # leaves SentencePiece's own bos_id()/eos_id()/pad_id() at -1. Those -1s were clamped to 0
        # in training, so the model never saw an end token. Look the pieces up by name instead.
        self.id_bos = self._special("<s>", self.sp.bos_id())
        self.id_eos = self._special("</s>", self.sp.eos_id())
        self.id_pad = self._special("<pad>", self.sp.pad_id())

    def _special(self, piece: str, builtin: int) -> int:
        if builtin >= 0:
            return int(builtin)
        piece_id = self.sp.piece_to_id(piece)
        if piece_id == self.sp.unk_id():
            raise ValueError(f"Tokenizer has no {piece} piece; retrain it with prepare_data.py")
        return int(piece_id)

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
