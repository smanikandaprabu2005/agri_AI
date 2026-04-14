"""
data_pipeline/step1_preprocess_dataset.py
==========================================
STEP 1 of 5 — Dataset Preprocessing

Adapts V1 strom_sage_ai_preprocessing.py for the V2 project.

What it does:
  1. Loads final_agriculture_training_dataset.jsonl  (167,993 records)
  2. Removes corrupted JSON lines
  3. Removes empty instruction / output fields
  4. Removes duplicate samples
  5. Filters by output length  (5–512 words)
  6. Prints dataset statistics
  7. Shuffles with fixed seed for reproducibility
  8. Splits 90% train / 10% validation
  9. Saves to:
       data collection/train_data/train_dataset.jsonl
       data collection/val_data/val_dataset.jsonl

Input:
  data/raw/final_agriculture_training_dataset.jsonl

Output:
  data collection/train_data/train_dataset.jsonl
  data collection/val_data/val_dataset.jsonl

Usage:
  python data_pipeline/step1_preprocess_dataset.py
  python data_pipeline/step1_preprocess_dataset.py --input path/to/dataset.jsonl
"""

import json
import random
import argparse
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
INPUT_FILE      = "data/raw/final_agriculture_training_dataset.jsonl"
TRAIN_FILE      = "data collection/train_data/train_dataset.jsonl"
VAL_FILE        = "data collection/val_data/val_dataset.jsonl"

MIN_OUTPUT_WORDS = 5
MAX_OUTPUT_WORDS = 512
TRAIN_SPLIT      = 0.9
RANDOM_SEED      = 42


# ── Loaders ───────────────────────────────────────────────────
def load_dataset(path: str) -> list:
    data      = []
    corrupted = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                corrupted += 1
    print(f"[Step 1] Loaded       : {len(data):,} records")
    print(f"[Step 1] Corrupted    : {corrupted:,} lines removed")
    return data


def remove_empty(data: list) -> list:
    clean = [
        item for item in data
        if item.get("instruction", "").strip()
        and item.get("output", "").strip()
    ]
    print(f"[Step 1] After empty  : {len(clean):,}")
    return clean


def remove_duplicates(data: list) -> list:
    seen, unique = set(), []
    for item in data:
        key = item["instruction"].strip() + "|||" + item["output"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    print(f"[Step 1] After dedup  : {len(unique):,}")
    return unique


def length_filter(data: list) -> list:
    filtered = [
        item for item in data
        if MIN_OUTPUT_WORDS
           <= len(item["output"].split())
           <= MAX_OUTPUT_WORDS
    ]
    print(f"[Step 1] After length : {len(filtered):,}  "
          f"(output {MIN_OUTPUT_WORDS}–{MAX_OUTPUT_WORDS} words)")
    return filtered


def print_stats(data: list):
    inst_lens = [len(d["instruction"].split()) for d in data]
    out_lens  = [len(d["output"].split())      for d in data]
    print(f"\n[Step 1] ── Dataset Statistics ──────────────────")
    print(f"  Total samples        : {len(data):,}")
    print(f"  Instruction  avg/max : {sum(inst_lens)/len(inst_lens):.1f} / {max(inst_lens)}")
    print(f"  Output       avg/max : {sum(out_lens)/len(out_lens):.1f} / {max(out_lens)}")
    has_input = sum(1 for d in data if d.get("input","").strip())
    print(f"  Samples with input   : {has_input:,}")
    print(f"─────────────────────────────────────────────────\n")


def split_and_save(data: list, train_path: str, val_path: str):
    random.seed(RANDOM_SEED)
    random.shuffle(data)
    n_train = int(TRAIN_SPLIT * len(data))
    train   = data[:n_train]
    val     = data[n_train:]

    for path, records in [(train_path, train), (val_path, val)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Step 1] Saved {len(records):>6,} records → {path}")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE,
                        help="Path to raw instruction dataset (.jsonl)")
    parser.add_argument("--train_out", default=TRAIN_FILE)
    parser.add_argument("--val_out",   default=VAL_FILE)
    parser.add_argument("--split",     type=float, default=TRAIN_SPLIT)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[Step 1] ERROR: Input file not found: {args.input}")
        print(f"  Place final_agriculture_training_dataset.jsonl in: data/raw/")
        return

    print(f"\n[Step 1] ── Preprocessing Dataset ───────────────")
    print(f"  Input  : {args.input}")
    print(f"  Train  : {args.train_out}")
    print(f"  Val    : {args.val_out}")
    print(f"─────────────────────────────────────────────────")

    data = load_dataset(args.input)
    data = remove_empty(data)
    data = remove_duplicates(data)
    data = length_filter(data)
    print_stats(data)
    split_and_save(data, args.train_out, args.val_out)

    print("\n[Step 1] ✓ Complete. Run step2 next:")
    print("  python data_pipeline/step2_build_knowledge_corpus.py")


if __name__ == "__main__":
    main()
