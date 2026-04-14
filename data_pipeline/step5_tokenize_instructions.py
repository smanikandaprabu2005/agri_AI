"""
data_pipeline/step5_tokenize_instructions.py
=============================================
STEP 5 of 5 — Instruction Dataset Tokenization

Adapts V1 Instruction_dataset_tokenization.py for the V2 project.

What it does:
  1. Loads sage_tokenizer.model
  2. Reads train_dataset.jsonl and val_dataset.jsonl
  3. Converts each record into the structured prompt format:

       ### Instruction:
       {instruction}

       ### Input:           ← only if input field is non-empty
       {input}

       ### Response:
       {output}

  4. Tokenizes the formatted text
  5. Saves to:
       tokens/train_tokens.jsonl
       tokens/val_tokens.jsonl

This EXACT format must match what the fine-tuning script uses
for response masking (find_response_start in train/finetune.py).

Input:
  sage_tokenizer.model
  data collection/train_data/train_dataset.jsonl
  data collection/val_data/val_dataset.jsonl

Output:
  tokens/train_tokens.jsonl
  tokens/val_tokens.jsonl

Usage:
  python data_pipeline/step5_tokenize_instructions.py
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
TRAIN_DATASET   = "data collection/train_data/train_dataset.jsonl"
VAL_DATASET     = "data collection/val_data/val_dataset.jsonl"
TRAIN_TOKENS    = "tokens/train_tokens.jsonl"
VAL_TOKENS      = "tokens/val_tokens.jsonl"


# ── Format prompt ─────────────────────────────────────────────
def format_prompt(instruction: str, input_text: str, output: str) -> str:
    """
    Build the structured prompt that matches the fine-tuning format.
    CRITICAL: this format must be identical in:
      - step3 (tokenizer corpus)
      - step5 (instruction tokenization)   ← this file
      - train/finetune.py (response masking)
      - models/tokenizer.py (inference prompt building)
    """
    if input_text.strip():
        return (
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{input_text.strip()}\n\n"
            f"### Response:\n{output.strip()}"
        )
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response:\n{output.strip()}"
    )


# ── Tokenize dataset ──────────────────────────────────────────
def tokenize_dataset(sp, input_path: str, output_path: str) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total = skipped = 0
    token_counts = []

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            try:
                item = json.loads(line)
                inst  = item.get("instruction", "").strip()
                inp   = item.get("input",       "").strip()
                out   = item.get("output",      "").strip()

                if not inst or not out:
                    skipped += 1
                    continue

                text   = format_prompt(inst, inp, out)
                tokens = sp.encode(text, out_type=int)

                if len(tokens) < 5:
                    skipped += 1
                    continue

                f_out.write(json.dumps({"tokens": tokens}) + "\n")
                token_counts.append(len(tokens))
                total += 1

            except Exception:
                skipped += 1
                continue

    avg = sum(token_counts)/max(len(token_counts),1)
    mx  = max(token_counts) if token_counts else 0
    return {"total": total, "skipped": skipped, "avg_len": avg, "max_len": mx}


# ── Verify response marker ────────────────────────────────────
def verify_response_marker(sp):
    """
    Print the token IDs for '### Response:' so you can verify
    they match what find_response_start() searches for in finetune.py.
    """
    marker = "### Response:"
    ids    = sp.encode(marker, out_type=int)
    print(f"[Step 5] '### Response:' token IDs : {ids}")
    print(f"[Step 5] Decoded back               : {sp.decode(ids)!r}")
    return ids


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",  default=TOKENIZER_MODEL)
    parser.add_argument("--train_data", default=TRAIN_DATASET)
    parser.add_argument("--val_data",   default=VAL_DATASET)
    parser.add_argument("--train_out",  default=TRAIN_TOKENS)
    parser.add_argument("--val_out",    default=VAL_TOKENS)
    args = parser.parse_args()

    # Validate
    missing = []
    if not Path(args.tokenizer).exists():
        missing.append(f"{args.tokenizer}  (run step3 first)")
    if not Path(args.train_data).exists():
        missing.append(f"{args.train_data}  (run step1 first)")
    if not Path(args.val_data).exists():
        missing.append(f"{args.val_data}  (run step1 first)")
    if missing:
        for m in missing: print(f"[Step 5] ERROR: Not found: {m}")
        return

    print(f"\n[Step 5] ── Tokenizing Instruction Datasets ─────")
    print(f"  Tokenizer    : {args.tokenizer}")
    print(f"  Train data   : {args.train_data}")
    print(f"  Val data     : {args.val_data}")
    print(f"  Train tokens : {args.train_out}")
    print(f"  Val tokens   : {args.val_out}")
    print(f"─────────────────────────────────────────────────")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"[Step 5] Tokenizer vocab size: {sp.get_piece_size()}")

    # Print response marker IDs (important for finetune.py)
    verify_response_marker(sp)

    # Tokenize train
    print(f"\n[Step 5] Tokenizing training set...")
    tr = tokenize_dataset(sp, args.train_data, args.train_out)
    print(f"  Tokenized : {tr['total']:,}  |  Skipped: {tr['skipped']:,}")
    print(f"  Avg len   : {tr['avg_len']:.1f}  |  Max len: {tr['max_len']}")
    print(f"  Saved → {args.train_out}")

    # Tokenize val
    print(f"\n[Step 5] Tokenizing validation set...")
    vl = tokenize_dataset(sp, args.val_data, args.val_out)
    print(f"  Tokenized : {vl['total']:,}  |  Skipped: {vl['skipped']:,}")
    print(f"  Avg len   : {vl['avg_len']:.1f}  |  Max len: {vl['max_len']}")
    print(f"  Saved → {args.val_out}")

    print(f"\n[Step 5] ✓ Complete. All tokens ready for training!")
    print("  Run the V2 training pipeline:")
    print("    python train/pretrain.py")
    print("    python train/finetune.py")
    print("    python train/evaluate.py --generate")


if __name__ == "__main__":
    main()
