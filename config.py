"""
config.py
=========
SageStorm V2 — Single source of truth for all hyperparameters.

CHANGES FROM V1:
  - PRETRAIN_EPOCHS: 30 → 60  (more training for small ICAR corpus)
  - FINETUNE_EPOCHS: 50 → 30  (prevents overfitting on 337K samples)
  - PRETRAIN_BATCH: 16 → 32   (better GPU utilization with 2 GPUs)
  - FINETUNE_LR_MAX: 5e-5 → 3e-5  (more conservative, less forgetting)
  - Added VAL_SPLIT constant for pretrain validation
  - NUM_WORKERS: explicit constant for reproducibility
"""

import os
import torch

# ── Paths ────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data_pipeline/data/raw")
MODELS_DIR        = os.path.join(BASE_DIR, "saved_models")
TOKENS_DIR        = os.path.join(BASE_DIR, "data_pipeline/tokens")
MEMORY_DIR        = os.path.join(BASE_DIR, "memory")

TOKENIZER_PATH    = os.path.join(BASE_DIR, "data_pipeline/sage_tokenizer.model")
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
NUM_HEADS         = 8           # query heads (GQA)
KV_HEADS          = 4           # key/value heads (GQA: 8q / 4kv = 2 groups)
NUM_LAYERS        = 12
DROPOUT           = 0.1

# ── Tokenizer Special IDs ────────────────────────────────────
PAD_ID            = 0           # NOTE: also a real vocab token — don't ignore in loss
BOS_ID            = 1
EOS_ID            = 2
UNK_ID            = 3

# ── Device ───────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP           = (DEVICE == "cuda")
NUM_WORKERS       = 4

# ── Pretraining ──────────────────────────────────────────────
# Small ICAR corpus (47MB) → needs more epochs to learn domain vocab
PRETRAIN_BATCH    = 32          # 16 → 32 for better GPU utilization
PRETRAIN_EPOCHS   = 60
PRETRAIN_LR_MAX   = 3e-4
PRETRAIN_LR_MIN   = 1e-5
PRETRAIN_WARMUP   = 500         # ~500 steps linear warmup
PRETRAIN_ACCUM    = 4           # effective batch = 32×4 = 128
PRETRAIN_VAL_SPLIT = 0.1        # 10% of tokens held out for validation

# ── Fine-tuning ──────────────────────────────────────────────
# 337K instruction samples — don't need too many epochs
FINETUNE_BATCH    = 16
FINETUNE_EPOCHS   = 30          # 50 → 30 (early stop at patience=8 anyway)
FINETUNE_LR_MAX   = 3e-5        # 5e-5 → 3e-5 (less catastrophic forgetting)
FINETUNE_LR_MIN   = 3e-7
FINETUNE_WARMUP   = 200         # linear warmup steps
FINETUNE_ACCUM    = 4           # effective batch = 16×4 = 64
FINETUNE_PATIENCE = 8           # early stop after 8 epochs no improvement
LABEL_SMOOTHING   = 0.1         # prevents overconfident predictions
LR_DECAY_RATE     = 0.85        # layer-wise decay per layer
WEIGHT_DECAY      = 0.05
CLIP_NORM         = 1.0

# ── Retrieval (Word2Vec BM25 hybrid) ─────────────────────────
W2V_EMBED_SIZE    = 64
W2V_EPOCHS        = 8
W2V_WINDOW        = 3
W2V_NEG_SAMPLES   = 5
RETRIEVAL_TOP_K   = 5

# ── Generation ───────────────────────────────────────────────
GEN_MAX_TOKENS    = 150
GEN_TEMPERATURE   = 0.7
GEN_TOP_K         = 50
GEN_TOP_P         = 0.9
GEN_REP_PENALTY   = 1.2

# ── Weather ──────────────────────────────────────────────────
DEFAULT_CITY      = "Guwahati"
WEATHER_CACHE_MIN = 30

# ── UI ───────────────────────────────────────────────────────
UI_HOST           = "0.0.0.0"
UI_PORT           = 7860

# ── Ensure directories exist ─────────────────────────────────
for _d in [DATA_DIR, MODELS_DIR, TOKENS_DIR, MEMORY_DIR, RETRIEVER_DIR]:
    os.makedirs(_d, exist_ok=True)