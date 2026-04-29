"""
models/tokenizer.py
===================
SentencePiece tokenizer wrapper with prompt building utilities.

FIXES vs original:
  1. get_response_ids() — fully deterministic, no off-by-one
  2. encode_prompt() — explicit BOS prepend; consistent with fine-tune format
  3. Added batch_encode() for efficient dataset tokenization
  4. Added vocab_size property guard for fallback mode
  5. Prompt templates are single source of truth (imported by all pipeline steps)
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOKENIZER_PATH, PAD_ID, BOS_ID, EOS_ID, UNK_ID


# ══════════════════════════════════════════════════════════════
#  Prompt templates
#  CRITICAL: These MUST be used identically in:
#    step3 (tokenizer corpus)  |  step5 (instruction tokenisation)
#    train/finetune.py         |  chatbot/response_generator.py
#    train/evaluate.py
# ══════════════════════════════════════════════════════════════

RESPONSE_MARKER = "### Response:\n"
INST_MARKER     = "### Instruction:\n"
INPUT_MARKER    = "### Input:\n"
CONTEXT_MARKER  = "### Context:\n"


def build_prompt(instruction: str, input_text: str = "") -> str:
    """Inference-time prompt (no output appended)."""
    if input_text.strip():
        return (
            f" {INST_MARKER}{instruction.strip()}\n\n"
            f"{INPUT_MARKER}{input_text.strip()}\n\n"
            f"{RESPONSE_MARKER}"
        )
    return f" {INST_MARKER}{instruction.strip()}\n\n{RESPONSE_MARKER}"


def build_rag_prompt(instruction: str, context: str, input_text: str = "") -> str:
    """Inference-time RAG prompt (context injected, no output appended)."""
    ctx = context.strip()[:600]
    if input_text.strip():
        return (
            f"{INST_MARKER}{instruction.strip()}\n\n"
            f"{INPUT_MARKER}{input_text.strip()}\n\n"
            f"{CONTEXT_MARKER}{ctx}\n\n"
            f"{RESPONSE_MARKER}"
        )
    return (
        f" {INST_MARKER}{instruction.strip()}\n\n"
        f"{CONTEXT_MARKER}{ctx}\n\n"
        f"{RESPONSE_MARKER}"
    )


def build_training_prompt(instruction: str, input_text: str, output: str) -> str:
    """Training-time prompt (output appended so response can be masked)."""
    if input_text.strip():
        return (
            f" {INST_MARKER}{instruction.strip()}\n\n"
            f"{INPUT_MARKER}{input_text.strip()}\n\n"
            f"{RESPONSE_MARKER}{output.strip()}"
        )
    return (
        f" {INST_MARKER}{instruction.strip()}\n\n"
        f"{RESPONSE_MARKER}{output.strip()}"
    )


# ══════════════════════════════════════════════════════════════
#  SageTokenizer
# ══════════════════════════════════════════════════════════════

class SageTokenizer:
    def __init__(self, model_path: str = TOKENIZER_PATH):
        self.model_path = model_path
        self._sp        = None
        self._fallback  = False
        self._resp_ids: list[int] = []
        self._load()

    # ── Loader ────────────────────────────────────────────────
    def _load(self):
        try:
            import sentencepiece as spm
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Tokenizer not found: {self.model_path}")
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(self.model_path)
            self._resp_ids = self._compute_response_ids()
            print(
                f"[Tokenizer] Loaded: {self.model_path} "
                f"(vocab={self._sp.get_piece_size()}, "
                f"resp_ids={self._resp_ids})"
            )
        except (ImportError, FileNotFoundError) as e:
            print(f"[Tokenizer] Fallback mode — {e}")
            self._fallback = True

    # ── Response marker IDs ───────────────────────────────────
    def _compute_response_ids(self) -> list[int]:
        """
        Compute token IDs for '### Response:\n' with context-aware encoding.

        CRITICAL FIX over original:
        The original stripped the final newline token but this was fragile.
        The correct approach is to tokenise the marker as it appears at the
        END of a full prompt so the tokeniser uses the same subword splits
        it will encounter during training.

        We use the full prefix trick: encode a short prefix + marker, then
        subtract the prefix IDs to isolate only the marker tokens.
        This avoids sentence-boundary normalisation artifacts.
        """
        sp = self._sp

        # Build a minimal prefix that mimics real prompt context
        prefix    = "### Instruction:\nTest\n\n"
        full_text = prefix + RESPONSE_MARKER

        prefix_ids = sp.encode(prefix, out_type=int)
        full_ids   = sp.encode(full_text, out_type=int)

        # Marker ids = full minus prefix
        marker_ids = full_ids[len(prefix_ids):]

        # Sanity check: decoding them should round-trip
        if sp.decode(marker_ids).strip() == RESPONSE_MARKER.strip():
            return marker_ids

        # Fallback: direct encode without context
        direct = sp.encode(RESPONSE_MARKER, out_type=int)
        return direct

    # ── Encode / Decode ───────────────────────────────────────
    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        if self._fallback:
            ids = [(hash(w) % (16000 - 10)) + 10 for w in text.lower().split()]
            return ([BOS_ID] + ids) if add_bos else ids
        ids = self._sp.encode(text, out_type=int)
        return ([BOS_ID] + ids) if add_bos else ids

    def encode_prompt(self, text: str) -> list[int]:
        """Always prepends BOS — used at inference time."""
        return self.encode(text, add_bos=True)

    def batch_encode(self, texts: list[str], add_bos: bool = False) -> list[list[int]]:
        """Efficient batch encode via SentencePiece native batch API."""
        if self._fallback:
            return [self.encode(t, add_bos) for t in texts]
        results = self._sp.encode(texts, out_type=int)
        if add_bos:
            results = [[BOS_ID] + ids for ids in results]
        return results

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if self._fallback:
            return "[sentencepiece required for decoding]"
        if skip_special:
            ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID, UNK_ID)]
        return self._sp.decode(ids)

    # ── Response marker ───────────────────────────────────────
    def get_response_ids(self) -> list[int]:
        return list(self._resp_ids)

    # ── Properties ───────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        if self._fallback:
            return 16000
        return self._sp.get_piece_size()

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def is_real(self) -> bool:
        return not self._fallback