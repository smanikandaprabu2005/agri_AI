"""
config.py
=========
SageStorm V2.2 — Regularization-tuned for 13M token pretraining budget.

Key changes vs V2.1:
  - DROPOUT 0.08 → 0.15     (critical: prevents val loss plateau at epoch 13)
  - CONTEXT_LENGTH 768 → 512 (more blocks from same data: ~25K → ~25K blocks
                               but 4x offset augmentation gives 100K+)
  - EMBED_DIM 640 → 512      (reduces params: 74M → 40M, better data/param ratio)
  - NUM_LAYERS 14 → 10       (contributes to 40M target)
  - PRETRAIN_LR_MAX 3e-4 → 2e-4  (more stable with SGDR restarts)
  - PRETRAIN_WARMUP 500 → 1000    (longer warmup for restart-aware schedule)
  - PRETRAIN_EPOCHS 60 → 80       (more epochs since augmentation changes curve)

Why 40M params?
  Your pretrain corpus = 13M tokens.
  Rule of thumb: need ~20 tokens/param for reasonable generalization.
  13M / 20 = 650K params minimum, but agriculture domain is narrow enough
  that 40M params with heavy regularization works well.
  The 337K finetune samples will fill in the agriculture knowledge anyway.
"""

import os
import torch

# ── Paths ────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
MODELS_DIR        = os.path.join(BASE_DIR, "saved_models")
TOKENS_DIR        = os.path.join(BASE_DIR, "data_pipeline", "tokens")
MEMORY_DIR        = os.path.join(BASE_DIR, "memory")
LOGS_DIR          = os.path.join(BASE_DIR, "logs")

TOKENIZER_PATH    = os.path.join(BASE_DIR, "sage_tokenizer.model")
PRETRAIN_TOKENS   = os.path.join(TOKENS_DIR, "pretrain_tokens.jsonl")
TRAIN_TOKENS      = os.path.join(TOKENS_DIR, "train_tokens.jsonl")
VAL_TOKENS        = os.path.join(TOKENS_DIR, "val_tokens.jsonl")

PRETRAINED_CKPT   = os.path.join(MODELS_DIR, "sage_pretrained_v2.pt")
FINETUNED_CKPT    = os.path.join(MODELS_DIR, "sage_slm_v2_final.pt")
RETRIEVER_DIR     = os.path.join(MODELS_DIR, "retriever")
VOCAB_PATH        = os.path.join(MODELS_DIR, "vocab.pkl")
FARMER_PROFILE    = os.path.join(MEMORY_DIR, "farmer_profile.json")

# ── Model Architecture ────────────────────────────────────────
# Tuned for 13M pretrain tokens + 337K finetune samples
#
# Parameter count: ~40M (down from 74M)
#   emb: 16000 * 512 = 8.2M
#   10 blocks * (attn + ffn): ~3.1M each = 31M
#   norm + head (tied): 0.5M
#   Total: ~40M
#
VOCAB_SIZE        = 16000
CONTEXT_LENGTH    = 512         # ← was 768 — more training blocks per epoch
EMBED_DIM         = 512         # ← was 640 — fewer params, better data/param
NUM_HEADS         = 8
KV_HEADS          = 4           # GQA ratio 2:1 maintained
NUM_LAYERS        = 10          # ← was 14
DROPOUT           = 0.15        # ← was 0.08 — CRITICAL FIX for overfitting
FFN_MULT          = 2.667       # SwiGLU hidden = 512 * 2.667 = 1,368 → 1,408 (nearest 64)

# ── Tokenizer special IDs ─────────────────────────────────────
PAD_ID            = 0
BOS_ID            = 1
EOS_ID            = 2
UNK_ID            = 3

# ── Pretraining ───────────────────────────────────────────────
PRETRAIN_BATCH    = 16
PRETRAIN_EPOCHS   = 80          # ← was 60 — more needed with SGDR restarts
PRETRAIN_LR_MAX   = 2e-4        # ← was 3e-4 — safer with restart schedule
PRETRAIN_LR_MIN   = 1e-5
PRETRAIN_WARMUP   = 1000        # ← was 500 — longer warmup for SGDR
PRETRAIN_ACCUM    = 4

# ── Fine-tuning ───────────────────────────────────────────────
# Kept same as V2.1 — 337K samples is large enough for these settings
FINETUNE_BATCH    = 16
FINETUNE_EPOCHS   = 40
FINETUNE_LR_MAX   = 5e-5        # slightly conservative with smaller model
FINETUNE_LR_MIN   = 5e-7
FINETUNE_WARMUP   = 400
FINETUNE_ACCUM    = 4
FINETUNE_PATIENCE = 8
LABEL_SMOOTHING   = 0.1
LR_DECAY_RATE     = 0.85
WEIGHT_DECAY      = 0.05
CLIP_NORM         = 1.0

# ── Training flags ────────────────────────────────────────────
USE_GRADIENT_CHECKPOINTING = True
USE_CURRICULUM_LEARNING    = True
CURRICULUM_WARMUP_EPOCHS   = 5
MIX_PRETRAIN_IN_FINETUNE   = False
PRETRAIN_MIX_RATIO         = 0.05

# ── Retrieval ─────────────────────────────────────────────────
RETRIEVER_TYPE    = "hybrid"
BM25_K1           = 1.5
BM25_B            = 0.75
TFIDF_MAX_FEATS   = 50000
HYBRID_ALPHA      = 0.6
RETRIEVAL_TOP_K   = 7
MIN_RETRIEVAL_SCORE = 0.05

# ── Word2Vec (fallback retriever) ─────────────────────────────
W2V_EMBED_SIZE    = 64
W2V_EPOCHS        = 8
W2V_WINDOW        = 3
W2V_NEG_SAMPLES   = 5

# ── Generation ────────────────────────────────────────────────
GEN_MAX_TOKENS    = 180
GEN_TEMPERATURE   = 0.65
GEN_TOP_K         = 40
GEN_TOP_P         = 0.88
GEN_REP_PENALTY   = 1.3
GEN_MIN_TOKENS    = 10

# ── Weather ───────────────────────────────────────────────────
DEFAULT_CITY      = "Guwahati"
WEATHER_CACHE_MIN = 30

# ── UI ────────────────────────────────────────────────────────
UI_HOST           = "0.0.0.0"
UI_PORT           = 7860

# ── Device ────────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP           = (DEVICE == "cuda")

# ── Ensure directories exist ──────────────────────────────────
for d in [DATA_DIR, MODELS_DIR, TOKENS_DIR, MEMORY_DIR, RETRIEVER_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)