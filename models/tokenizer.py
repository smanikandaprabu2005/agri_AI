"""
models/tokenizer.py
===================
SentencePiece tokenizer wrapper with prompt building utilities.
Falls back gracefully if sentencepiece is not installed.
"""

import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOKENIZER_PATH, PAD_ID, BOS_ID, EOS_ID, UNK_ID


# ── Prompt templates (must match fine-tuning format) ─────────
def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return (f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Input:\n{input_text.strip()}\n\n"
                f"### Response:\n")
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def build_rag_prompt(instruction: str, context: str, input_text: str = "") -> str:
    if input_text.strip():
        return (f"### Instruction:\n{instruction.strip()}\n\n"
                f"### Input:\n{input_text.strip()}\n\n"
                f"### Context:\n{context.strip()[:600]}\n\n"
                f"### Response:\n")
    return (f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Context:\n{context.strip()[:600]}\n\n"
            f"### Response:\n")


class SageTokenizer:
    def __init__(self, model_path: str = TOKENIZER_PATH):
        self.model_path = model_path
        self._sp        = None
        self._fallback  = False
        self._load()

    def _load(self):
        try:
            import sentencepiece as spm
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Tokenizer not found: {self.model_path}")
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(self.model_path)
            print(f"[Tokenizer] Loaded: {self.model_path} (vocab={self._sp.get_piece_size()})")
        except (ImportError, FileNotFoundError) as e:
            print(f"[Tokenizer] Fallback mode — {e}")
            self._fallback = True

    def encode(self, text: str, add_bos: bool = False) -> list:
        if self._fallback:
            ids = [(hash(w) % (16000 - 10)) + 10 for w in text.lower().split()]
            return ([BOS_ID] + ids) if add_bos else ids
        ids = self._sp.encode(text, out_type=int)
        return ([BOS_ID] + ids) if add_bos else ids

    def encode_prompt(self, text: str) -> list:
        return self.encode(text, add_bos=True)

    def decode(self, ids: list, skip_special: bool = True) -> str:
        if self._fallback:
            return "[sentencepiece required for decoding]"
        if skip_special:
            ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID, UNK_ID)]
        return self._sp.decode(ids)

    def get_response_ids(self) -> list:
        """Get response marker IDs including newline to match actual prompt format.
        Must include the newline to ensure correct context-dependent tokenization.
        """
        if self._fallback:
            return []
        try:
            # Include newline as it appears in actual prompts from format_prompt()
            marker = "### Response:\n"
            ids = self._sp.encode(marker, out_type=int)
            if ids:
                return ids[:-1]  # Remove the newline token, keep just the marker
        except Exception:
            pass
        
        # Fallback: try without newline
        try:
            ids = self._sp.encode("### Response:", out_type=int)
            if ids:
                return ids
        except Exception:
            pass
        
        return []

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size() if not self._fallback else 16000

    @property
    def eos_id(self): return EOS_ID

    @property
    def bos_id(self): return BOS_ID

    @property
    def is_real(self): return not self._fallback
