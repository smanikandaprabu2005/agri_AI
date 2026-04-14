"""
data_pipeline/step3_train_tokenizer.py
=======================================
STEP 3 of 5 — SentencePiece Tokenizer Training

Adapts V1 train_sagestorm_tokenizer.py for the V2 project.

What it does:
  1. Reads train_dataset.jsonl  (instruction + input + output)
  2. Reads knowledge_corpus.txt (ICAR + research sentences)
  3. Builds a combined tokenizer corpus (tokenizer_corpus.txt)
  4. Trains SentencePiece Unigram tokenizer
       vocab_size = 16,000
       special tokens: <pad>=0  <bos>=1  <eos>=2  <unk>=3
  5. Saves sage_tokenizer.model + sage_tokenizer.vocab

Input:
  data collection/train_data/train_dataset.jsonl
  processed/knowledge_corpus.txt

Output:
  tokenizer_corpus.txt          (intermediate, can be deleted after)
  sage_tokenizer.model          ← used by ALL training and inference
  sage_tokenizer.vocab

Usage:
  python data_pipeline/step3_train_tokenizer.py
"""

import json
import argparse
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    raise SystemExit("sentencepiece not installed. Run: pip install sentencepiece")

# ── Config ────────────────────────────────────────────────────
TRAIN_DATASET    = "data collection/train_data/train_dataset.jsonl"
KNOWLEDGE_CORPUS = "processed/knowledge_corpus.txt"
CORPUS_FILE      = "tokenizer_corpus.txt"
MODEL_PREFIX     = "sage_tokenizer"
VOCAB_SIZE       = 16000
MODEL_TYPE       = "unigram"
CHARACTER_COV    = 0.9995


# ── Build corpus ──────────────────────────────────────────────
def build_corpus(train_path: str, knowledge_path: str, corpus_path: str) -> int:
    """
    Combines instruction data and knowledge corpus into one text file
    for SentencePiece training.
    The prompt format matches exactly what will be used during fine-tuning
    so the tokenizer learns the special marker tokens properly.
    """
    total = 0
    with open(corpus_path, "w", encoding="utf-8") as corpus:

        # ── Part 1: Instruction dataset ───────────────────────
        if Path(train_path).exists():
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        inst  = item.get("instruction", "").strip()
                        inp   = item.get("input", "").strip()
                        out   = item.get("output", "").strip()
                        if not inst or not out:
                            continue
                        # use the SAME prompt format as fine-tuning
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

        # ── Part 2: Knowledge corpus ──────────────────────────
        knowledge_count = 0
        if Path(knowledge_path).exists():
            with open(knowledge_path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if len(text) > 50:
                        corpus.write(text + "\n")
                        knowledge_count += 1
            print(f"[Step 3] Knowledge sentences added : {knowledge_count:,}")
        else:
            print(f"[Step 3] WARNING: {knowledge_path} not found — run step2 first")
        total += knowledge_count

    size = Path(corpus_path).stat().st_size / 1024 / 1024
    print(f"[Step 3] Corpus size               : {size:.1f} MB ({total:,} lines)")
    return total


# ── Train tokenizer ───────────────────────────────────────────
def train_tokenizer(corpus_path: str, model_prefix: str):
    print(f"\n[Step 3] Training SentencePiece tokenizer...")
    print(f"  Vocab size  : {VOCAB_SIZE}")
    print(f"  Model type  : {MODEL_TYPE}")
    print(f"  Coverage    : {CHARACTER_COV}")

    spm.SentencePieceTrainer.train(
        input              = corpus_path,
        model_prefix       = model_prefix,
        vocab_size         = VOCAB_SIZE,
        model_type         = MODEL_TYPE,
        character_coverage = CHARACTER_COV,
        pad_id             = 0,
        bos_id             = 1,
        eos_id             = 2,
        unk_id             = 3,
    )

    print(f"[Step 3] Model saved → {model_prefix}.model")
    print(f"[Step 3] Vocab saved → {model_prefix}.vocab")


# ── Test tokenizer ────────────────────────────────────────────
def test_tokenizer(model_prefix: str):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    tests = [
        "How to control caterpillars from eating leaves?",
        "Apply urea at 18 kg per acre for rice crop.",
        "### Instruction:\nHow to control aphids?\n\n### Response:\n",
    ]
    print(f"\n[Step 3] ── Tokenizer smoke test ────────────────")
    for text in tests:
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        print(f"  IN : {text[:60]}")
        print(f"  IDS: {ids[:8]}...")
        print(f"  OUT: {decoded[:60]}")
        print()


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",  default=TRAIN_DATASET)
    parser.add_argument("--knowledge",   default=KNOWLEDGE_CORPUS)
    parser.add_argument("--corpus_out",  default=CORPUS_FILE)
    parser.add_argument("--model_prefix",default=MODEL_PREFIX)
    args = parser.parse_args()

    print(f"\n[Step 3] ── Training Tokenizer ──────────────────")
    print(f"  Train data : {args.train_data}")
    print(f"  Knowledge  : {args.knowledge}")
    print(f"  Corpus     : {args.corpus_out}")
    print(f"  Model      : {args.model_prefix}.model")
    print(f"─────────────────────────────────────────────────")

    build_corpus(args.train_data, args.knowledge, args.corpus_out)
    train_tokenizer(args.corpus_out, args.model_prefix)
    test_tokenizer(args.model_prefix)

    print("\n[Step 3] ✓ Complete. Run step4 next:")
    print("  python data_pipeline/step4_tokenize_pretrain.py")


if __name__ == "__main__":
    main()
