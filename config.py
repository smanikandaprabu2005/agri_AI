"""
config.py
=========
Single source of truth for all project hyperparameters.
FIXED & OPTIMIZED for 337,554 fine-tune samples + 206,923 pretrain tokens.
"""

import os
import torch

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "saved_models")
TOKENS_DIR      = os.path.join(BASE_DIR, "tokens")
MEMORY_DIR      = os.path.join(BASE_DIR, "memory")

TOKENIZER_PATH  = os.path.join(BASE_DIR, "sage_tokenizer.model")
PRETRAIN_TOKENS = os.path.join(TOKENS_DIR, "pretrain_tokens.jsonl")
TRAIN_TOKENS    = os.path.join(TOKENS_DIR, "train_tokens.jsonl")
VAL_TOKENS      = os.path.join(TOKENS_DIR, "val_tokens.jsonl")

PRETRAINED_CKPT = os.path.join(MODELS_DIR, "sage_pretrained_v2.pt")
FINETUNED_CKPT  = os.path.join(MODELS_DIR, "sage_slm_v2_final.pt")
RETRIEVER_DIR   = os.path.join(MODELS_DIR, "retriever")
VOCAB_PATH      = os.path.join(MODELS_DIR, "vocab.pkl")
FARMER_PROFILE  = os.path.join(MEMORY_DIR, "farmer_profile.json")

# ── Model Architecture ───────────────────────────────────────
# FIXED: Increased capacity to handle 337k samples + 32M tokens
# Previously 512 embed_dim was fine, but more layers help with domain depth
VOCAB_SIZE      = 16000
CONTEXT_LENGTH  = 512
EMBED_DIM       = 512
NUM_HEADS       = 8         # query heads
KV_HEADS        = 4         # key/value heads (GQA)
NUM_LAYERS      = 12
DROPOUT         = 0.1

# ── Tokenizer special IDs ────────────────────────────────────
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ── Pretraining ──────────────────────────────────────────────
# FIXED: Pretrain corpus is small (~206K tokens), fewer epochs prevent overfitting
# Effective batch = PRETRAIN_BATCH * PRETRAIN_ACCUM = 32 * 4 = 128
PRETRAIN_BATCH   = 32
PRETRAIN_EPOCHS  = 30          # was 60 — 206K tokens overfits at 60 epochs
PRETRAIN_LR_MAX  = 3e-4
PRETRAIN_LR_MIN  = 1e-5
PRETRAIN_WARMUP  = 300          # was 500; adjusted for smaller corpus
PRETRAIN_ACCUM   = 4

# ── Fine-tuning ──────────────────────────────────────────────
# FIXED: 337K samples * ~95 tokens = 32M tokens. Good capacity.
# Reduced patience 8→5 to avoid wasted compute on plateau
# Increased warmup for better gradient stability at start
FINETUNE_BATCH    = 16
FINETUNE_EPOCHS   = 30           # was 50; 337K samples converge faster
FINETUNE_LR_MAX   = 3e-5         # was 5e-5; lower = more stable with large dataset
FINETUNE_LR_MIN   = 3e-7
FINETUNE_WARMUP   = 400          # was 200; larger dataset needs longer ramp
FINETUNE_ACCUM    = 4
FINETUNE_PATIENCE = 5            # was 8; tighter early stopping
LABEL_SMOOTHING   = 0.1
LR_DECAY_RATE     = 0.85
WEIGHT_DECAY      = 0.05
CLIP_NORM         = 1.0

# ── Retrieval (Word2Vec) ─────────────────────────────────────
# FIXED: Increased embed size 64→128 for better semantic coverage
# More epochs for vocabulary of agricultural domain
W2V_EMBED_SIZE  = 128            # was 64; better semantic coverage
W2V_EPOCHS      = 15             # was 8; domain vocab needs more training
W2V_WINDOW      = 5              # was 3; capture wider agricultural context
W2V_NEG_SAMPLES = 5
RETRIEVAL_TOP_K  = 5

# ── Generation ───────────────────────────────────────────────
# FIXED: Slightly lower temp for more factual agriculture answers
GEN_MAX_TOKENS  = 200            # was 150; allow fuller answers
GEN_TEMPERATURE = 0.65           # was 0.7; more deterministic for facts
GEN_TOP_K       = 40             # was 50
GEN_TOP_P       = 0.9
GEN_REP_PENALTY = 1.3            # was 1.2; stronger repetition penalty

# ── Weather ──────────────────────────────────────────────────
DEFAULT_CITY      = "Guwahati"
WEATHER_CACHE_MIN = 30

# ── UI ───────────────────────────────────────────────────────
UI_HOST = "0.0.0.0"
UI_PORT = 7860

# ── Device ───────────────────────────────────────────────────
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

# ── Ensure directories exist ─────────────────────────────────
for _d in [DATA_DIR, MODELS_DIR, TOKENS_DIR, MEMORY_DIR, RETRIEVER_DIR]:
    os.makedirs(_d, exist_ok=True)