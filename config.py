"""
config.py
=========
Single source of truth for all project hyperparameters.
Import this in every module instead of hardcoding values.
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

# ── Fine-tuning ──────────────────────────────────────────────
FINETUNE_BATCH    = 16
FINETUNE_EPOCHS   = 50
FINETUNE_LR_MAX   = 5e-5
FINETUNE_LR_MIN   = 5e-7
FINETUNE_WARMUP   = 200
FINETUNE_ACCUM    = 4
FINETUNE_PATIENCE = 8
LABEL_SMOOTHING   = 0.1
LR_DECAY_RATE     = 0.85
WEIGHT_DECAY      = 0.05
CLIP_NORM         = 1.0

# ── Retrieval (Word2Vec) ─────────────────────────────────────
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

# ── Device ───────────────────────────────────────────────────
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP           = (DEVICE == "cuda")

# ── Ensure directories exist ─────────────────────────────────
for d in [DATA_DIR, MODELS_DIR, TOKENS_DIR, MEMORY_DIR, RETRIEVER_DIR]:
    os.makedirs(d, exist_ok=True)
