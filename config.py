"""
config.py  —  SageStorm V2.1  (production-ready hyperparameters)
=================================================================
Key changes from V2:
  - Label smoothing 0.10 → 0.15   (fixes overconfident loss)
  - Finetune LR max 5e-5 → 3e-5   (less overshooting with 337K samples)
  - Warmup steps 200 → 400        (longer ramp for big dataset)
  - Rep penalty 1.20 → 1.10       (less aggressive, less truncated answers)
  - Context max 600 → 1200 chars  (richer RAG context window)
  - Added cosine annealing min 3e-7 (V2 stopped too early)
  - Patience 8 → 10               (337K set needs more epochs to stabilise)
"""

import os
import torch

# ── Paths ────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
MODELS_DIR        = os.path.join(BASE_DIR, "saved_models")
TOKENS_DIR        = os.path.join(BASE_DIR, "tokens")
MEMORY_DIR        = os.path.join(BASE_DIR, "memory")

TOKENIZER_PATH    = os.path.join(BASE_DIR, "sage_tokenizer.model")
PRETRAIN_TOKENS   = os.path.join(TOKENS_DIR, "pretrain_tokens.jsonl")
TRAIN_TOKENS      = os.path.join(TOKENS_DIR, "train_tokens.jsonl")
VAL_TOKENS        = os.path.join(TOKENS_DIR, "val_tokens.jsonl")

PRETRAINED_CKPT   = os.path.join(MODELS_DIR, "sage_pretrained_v2.pt")
FINETUNED_CKPT    = os.path.join(MODELS_DIR, "sage_slm_v2_final.pt")
RETRIEVER_DIR     = os.path.join(MODELS_DIR, "retriever")
VOCAB_PATH        = os.path.join(MODELS_DIR, "vocab.pkl")
FARMER_PROFILE    = os.path.join(MEMORY_DIR, "farmer_profile.json")

# ── Model Architecture ───────────────────────────────────────
VOCAB_SIZE        = 16000
CONTEXT_LENGTH    = 512
EMBED_DIM         = 512
NUM_HEADS         = 8        # query heads
KV_HEADS          = 4        # key/value heads (GQA)
NUM_LAYERS        = 12
DROPOUT           = 0.1

# ── Tokenizer special IDs ────────────────────────────────────
PAD_ID            = 0
BOS_ID            = 1
EOS_ID            = 2
UNK_ID            = 3

# ── Pretraining ──────────────────────────────────────────────
PRETRAIN_BATCH    = 16
PRETRAIN_EPOCHS   = 60
PRETRAIN_LR_MAX   = 3e-4
PRETRAIN_LR_MIN   = 1e-5
PRETRAIN_WARMUP   = 500
PRETRAIN_ACCUM    = 4

# ── Fine-tuning  (V2.1 CHANGES MARKED) ──────────────────────
FINETUNE_BATCH    = 16
FINETUNE_EPOCHS   = 50
FINETUNE_LR_MAX   = 3e-5      # ← V2.1: was 5e-5 (too high for 337K samples)
FINETUNE_LR_MIN   = 3e-7      # ← V2.1: was 5e-7 (slightly warmer floor)
FINETUNE_WARMUP   = 400       # ← V2.1: was 200 (longer warmup for big set)
FINETUNE_ACCUM    = 4
FINETUNE_PATIENCE = 10        # ← V2.1: was 8
LABEL_SMOOTHING   = 0.15      # ← V2.1: was 0.10 (KEY fix for high loss)
LR_DECAY_RATE     = 0.85
WEIGHT_DECAY      = 0.05
CLIP_NORM         = 1.0

# ── Retrieval (Word2Vec + optional BM25) ─────────────────────
W2V_EMBED_SIZE    = 64
W2V_EPOCHS        = 8
W2V_WINDOW        = 3
W2V_NEG_SAMPLES   = 5
RETRIEVAL_TOP_K   = 5

# ── Generation  (V2.1 CHANGES MARKED) ───────────────────────
GEN_MAX_TOKENS    = 150
GEN_TEMPERATURE   = 0.7
GEN_TOP_K         = 50
GEN_TOP_P         = 0.9
GEN_REP_PENALTY   = 1.10      # ← V2.1: was 1.20 (less choppy, more fluent)

# ── RAG context ──────────────────────────────────────────────
RAG_CONTEXT_MAX   = 1200      # ← V2.1: was 600 (richer context = better gen)

# ── Weather ──────────────────────────────────────────────────
DEFAULT_CITY      = "Guwahati"
WEATHER_CACHE_MIN = 30

# ── UI ───────────────────────────────────────────────────────
UI_HOST           = "0.0.0.0"
UI_PORT           = 7860

# ── Device ───────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP           = (DEVICE == "cuda")

# ── Ensure directories exist ─────────────────────────────────
for d in [DATA_DIR, MODELS_DIR, TOKENS_DIR, MEMORY_DIR, RETRIEVER_DIR]:
    os.makedirs(d, exist_ok=True)
