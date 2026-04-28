"""
data_pipeline/step3_train_tokenizer.py
=======================================
STEP 3 of 5 — SentencePiece Tokenizer Training

FIXES:
  FIX 1: TRAIN_DATASET path: "data collection/" → "data_collection/"
  FIX 2: VOCAB_SIZE imported from config to stay in sync with model
  FIX 3: Added --vocab_size CLI arg so it can be overridden
"""

import json
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import sentencepiece as spm
except ImportError:
    raise SystemExit("sentencepiece not installed. Run: pip install sentencepiece")

# FIX 2: Import from config — single source of truth
from config import VOCAB_SIZE as CONFIG_VOCAB_SIZE

# ── Config ────────────────────────────────────────────────────
# FIX 1: Removed space from path
TRAIN_DATASET    = "data_pipeline/data collection/train_dataset.jsonl"
KNOWLEDGE_CORPUS = "data_pipeline/processed/knowledge_corpus.txt"
CORPUS_FILE      = "data_pipeline/processed/tokenizer_corpus.txt"
MODEL_PREFIX     = "sage_tokenizer"
MODEL_TYPE       = "unigram"
CHARACTER_COV    = 0.9995


def clean_markdown(text: str) -> str:
    """Remove markdown formatting noise from text."""
    text = text.replace('**', '')
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_corpus(train_path: str, knowledge_path: str, corpus_path: str) -> int:
    total = 0
    with open(corpus_path, "w", encoding="utf-8") as corpus:

        # Part 1: Instruction dataset (same format as fine-tuning)
        if Path(train_path).exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        inst = item.get("instruction", "").strip()
                        inp  = item.get("input", "").strip()
                        out  = item.get("output", "").strip()
                        if not inst or not out:
                            continue
                        # Clean markdown from all text
                        inst = clean_markdown(inst)
                        inp = clean_markdown(inp) if inp else ""
                        out = clean_markdown(out)
                        if inp:
                            text = (f"### Instruction:\n{inst}\n\n"
                                    f"### Input:\n{inp}\n\n"
                                    f"### Response:\n{out}")
                        else:
                            text = (f"### Instruction:\n{inst}\n\n"
                                    f"### Response:\n{out}")
                        corpus.write(text.strip() + "\n")
                        total += 1
                    except Exception:
                        continue
            print(f"[Step 3] Instruction samples added : {total:,}")
        else:
            print(f"[Step 3] WARNING: {train_path} not found — run step1 first")

        # Part 2: Knowledge corpus
        knowledge_count = 0
        if Path(knowledge_path).exists():
            with open(knowledge_path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if len(text) > 50:
                        # Already cleaned in step2, but apply again for safety
                        text = clean_markdown(text)
                        corpus.write(text + "\n")
                        knowledge_count += 1
            print(f"[Step 3] Knowledge sentences added : {knowledge_count:,}")
        else:
            print(f"[Step 3] WARNING: {knowledge_path} not found — run step2 first")
        total += knowledge_count

    size = Path(corpus_path).stat().st_size / 1024 / 1024
    print(f"[Step 3] Corpus size               : {size:.1f} MB ({total:,} lines)")
    return total


def train_tokenizer(corpus_path: str, model_prefix: str, vocab_size: int):
    print(f"\n[Step 3] Training SentencePiece tokenizer...")
    print(f"  Vocab size  : {vocab_size}")
    print(f"  Model type  : {MODEL_TYPE}")
    print(f"  Coverage    : {CHARACTER_COV}")

    spm.SentencePieceTrainer.train(
        input              = corpus_path,
        model_prefix       = model_prefix,
        vocab_size         = vocab_size,     # FIX 2: uses config value
        model_type         = MODEL_TYPE,
        character_coverage = CHARACTER_COV,
        pad_id             = 0,
        bos_id             = 1,
        eos_id             = 2,
        unk_id             = 3,
    )
    print(f"[Step 3] Model saved → {model_prefix}.model")
    print(f"[Step 3] Vocab saved → {model_prefix}.vocab")


def test_tokenizer(model_prefix: str):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    tests = [
        "How to control caterpillars from eating leaves?",
        "Apply urea at 18 kg per acre for rice crop.",
        "### Instruction:\nHow to control aphids?\n\n### Response:\n",
        "Spray Malathion 50EC at 2 ml/litre water. NPK 20:20:20% recommended.",
    ]
    print(f"\n[Step 3] ── Tokenizer smoke test ────────────────")
    for text in tests:
        ids     = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        print(f"  IN : {text[:60]}")
        print(f"  IDS: {ids[:8]}...")
        print(f"  OUT: {decoded[:60]}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",   default=TRAIN_DATASET)
    parser.add_argument("--knowledge",    default=KNOWLEDGE_CORPUS)
    parser.add_argument("--corpus_out",   default=CORPUS_FILE)
    parser.add_argument("--model_prefix", default=MODEL_PREFIX)
    # FIX 3: Added --vocab_size arg (defaults to config value)
    parser.add_argument("--vocab_size",   type=int, default=CONFIG_VOCAB_SIZE)
    args = parser.parse_args()

    print(f"\n[Step 3] ── Training Tokenizer ──────────────────")
    print(f"  Train data : {args.train_data}")
    print(f"  Knowledge  : {args.knowledge}")
    print(f"  Corpus     : {args.corpus_out}")
    print(f"  Model      : {args.model_prefix}.model")
    print(f"  Vocab size : {args.vocab_size}")
    print(f"─────────────────────────────────────────────────")

    build_corpus(args.train_data, args.knowledge, args.corpus_out)
    train_tokenizer(args.corpus_out, args.model_prefix, args.vocab_size)
    test_tokenizer(args.model_prefix)

    print("\n[Step 3] ✓ Complete. Run step4 next:")
    print("  python data_pipeline/step4_tokenize_pretrain.py")


if __name__ == "__main__":
    main()