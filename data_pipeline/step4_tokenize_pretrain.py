"""
data_pipeline/step4_tokenize_pretrain.py
=========================================
STEP 4 of 5 — Pretraining Dataset Tokenization

Adapts V1 Pretraining_dataset_tokenization.py for the V2 project.

What it does:
  1. Loads sage_tokenizer.model
  2. Reads processed/knowledge_corpus.txt  (ICAR + research sentences)
  3. Cleans each line
  4. Tokenizes each line with SentencePiece
  5. Saves token sequences to tokens/pretrain_tokens.jsonl

Input:
  sage_tokenizer.model
  processed/knowledge_corpus.txt

Output:
  tokens/pretrain_tokens.jsonl
    → Each line: {"tokens": [id1, id2, ...]}

Usage:
  python data_pipeline/step4_tokenize_pretrain.py
"""

import json
import argparse
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    raise SystemExit("sentencepiece not installed. Run: pip install sentencepiece")

# ── Config ────────────────────────────────────────────────────
TOKENIZER_MODEL = "sage_tokenizer.model"
CORPUS_FILE     = "processed/knowledge_corpus.txt"
OUTPUT_FILE     = "tokens/pretrain_tokens.jsonl"
MIN_TEXT_LEN    = 30    # skip very short sentences


# ── Cleaner ───────────────────────────────────────────────────
def clean_line(text: str) -> str:
    return " ".join(text.replace("\n","").replace("\t","").split()).strip()


# ── Tokenize corpus ───────────────────────────────────────────
def tokenize_corpus(sp, corpus_path: str, output_path: str) -> int:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(corpus_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            text = clean_line(line)
            if len(text) < MIN_TEXT_LEN:
                continue
            tokens = sp.encode(text, out_type=int)
            if len(tokens) < 5:
                continue
            f_out.write(json.dumps({"tokens": tokens}) + "\n")
            total += 1

            if total % 10000 == 0:
                print(f"  [Step 4] Tokenized {total:,} lines...")

    return total


# ── Stats ─────────────────────────────────────────────────────
def print_stats(output_path: str):
    total_toks = 0
    count      = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            total_toks += len(item["tokens"])
            count += 1
    print(f"\n[Step 4] ── Output Statistics ───────────────────")
    print(f"  Sequences tokenized : {count:,}")
    print(f"  Total tokens        : {total_toks:,}")
    print(f"  Average seq length  : {total_toks/max(count,1):.1f} tokens")
    size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  Output file size    : {size:.1f} MB")
    print(f"─────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=TOKENIZER_MODEL)
    parser.add_argument("--corpus",    default=CORPUS_FILE)
    parser.add_argument("--output",    default=OUTPUT_FILE)
    args = parser.parse_args()

    # Validate inputs
    missing = []
    if not Path(args.tokenizer).exists(): missing.append(f"{args.tokenizer}  (run step3 first)")
    if not Path(args.corpus).exists():    missing.append(f"{args.corpus}  (run step2 first)")
    if missing:
        for m in missing: print(f"[Step 4] ERROR: Not found: {m}")
        return

    print(f"\n[Step 4] ── Tokenizing Pretrain Corpus ──────────")
    print(f"  Tokenizer : {args.tokenizer}")
    print(f"  Corpus    : {args.corpus}")
    print(f"  Output    : {args.output}")
    print(f"─────────────────────────────────────────────────")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"[Step 4] Tokenizer vocab size: {sp.get_piece_size()}")

    total = tokenize_corpus(sp, args.corpus, args.output)
    print(f"[Step 4] Total sequences saved: {total:,}")
    print_stats(args.output)

    print(f"\n[Step 4] Saved → {args.output}")
    print("\n[Step 4] ✓ Complete. Run step5 next:")
    print("  python data_pipeline/step5_tokenize_instructions.py")


if __name__ == "__main__":
    main()
