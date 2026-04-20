"""
config.py
=========
Single source of truth for all project hyperparameters.
Enhanced V2.1 — optimised for agriculture domain SLM.

Key changes vs V2:
  - Context length 512 → 768  (agriculture Q&A needs longer context)
  - Embed dim 512 → 640       (better capacity without blowing memory)
  - Num layers 12 → 14        (deeper = better generalisation)
  - KV heads 4 → 4            (keep GQA ratio)
  - Dropout 0.1 → 0.08        (large dataset, less regularisation needed)
  - Finetune LR raised slightly (337k samples supports higher LR)
  - Added gradient checkpointing flag
  - Added BM25 retriever toggle
  - Added curriculum learning flag
"""

import os
import torch

# ── Paths ────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
MODELS_DIR        = os.path.join(BASE_DIR, "saved_models")
TOKENS_DIR        = os.path.join(BASE_DIR, "tokens")
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

# ── Model Architecture ───────────────────────────────────────
VOCAB_SIZE        = 16000
CONTEXT_LENGTH    = 768        # ↑ from 512  (longer agri Q&A)
EMBED_DIM         = 640        # ↑ from 512  (more capacity)
NUM_HEADS         = 8          # query heads
KV_HEADS          = 4          # key/value heads (GQA ratio 2:1)
NUM_LAYERS        = 14         # ↑ from 12
DROPOUT           = 0.08       # ↓ from 0.1  (large dataset)
FFN_MULT          = 2.667      # SwiGLU hidden = embed_dim * FFN_MULT

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
FINETUNE_EPOCHS   = 40         # ↓ slightly — more data converges faster
FINETUNE_LR_MAX   = 6e-5       # ↑ from 5e-5  (337k samples supports it)
FINETUNE_LR_MIN   = 5e-7
FINETUNE_WARMUP   = 300        # ↑ from 200  (larger model needs more warmup)
FINETUNE_ACCUM    = 4
FINETUNE_PATIENCE = 8
LABEL_SMOOTHING   = 0.1
LR_DECAY_RATE     = 0.85
WEIGHT_DECAY      = 0.05
CLIP_NORM         = 1.0

# ── Training flags ───────────────────────────────────────────
USE_GRADIENT_CHECKPOINTING = True   # saves VRAM, ~20% slower
USE_CURRICULUM_LEARNING    = True   # sort by difficulty early
CURRICULUM_WARMUP_EPOCHS   = 5      # epochs before full random shuffle
MIX_PRETRAIN_IN_FINETUNE   = False  # add 5% pretrain data to finetune
PRETRAIN_MIX_RATIO         = 0.05

# ── Retrieval ────────────────────────────────────────────────
RETRIEVER_TYPE    = "bm25"     # "bm25" | "tfidf" | "hybrid" | "word2vec"
BM25_K1           = 1.5        # BM25 term saturation
BM25_B            = 0.75       # BM25 length normalisation
TFIDF_MAX_FEATS   = 50000      # max TF-IDF vocabulary
HYBRID_ALPHA      = 0.6        # weight for BM25 in hybrid (1-alpha for TF-IDF)
RETRIEVAL_TOP_K   = 7          # ↑ from 5  (retrieve more, filter later)
MIN_RETRIEVAL_SCORE = 0.05     # minimum cosine/BM25 score to include

# ── Word2Vec (fallback retriever) ────────────────────────────
W2V_EMBED_SIZE    = 64
W2V_EPOCHS        = 8
W2V_WINDOW        = 3
W2V_NEG_SAMPLES   = 5

# ── Generation ───────────────────────────────────────────────
GEN_MAX_TOKENS    = 180        # ↑ from 150  (agri answers can be detailed)
GEN_TEMPERATURE   = 0.65       # ↓ slightly for more factual outputs
GEN_TOP_K         = 40         # ↓ from 50
GEN_TOP_P         = 0.88       # ↓ from 0.9
GEN_REP_PENALTY   = 1.3        # ↑ from 1.2  (reduce repetition)
GEN_MIN_TOKENS    = 10         # prevent empty/short responses

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
for d in [DATA_DIR, MODELS_DIR, TOKENS_DIR, MEMORY_DIR, RETRIEVER_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)