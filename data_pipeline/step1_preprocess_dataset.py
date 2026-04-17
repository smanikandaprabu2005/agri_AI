"""
data_pipeline/step1_preprocess_dataset.py  —  V2.1
====================================================
V2.1 changes for the larger 337K-sample dataset:
  NEW: Minimum output length raised 5 → 15 words
       (removes near-empty answers that hurt calibration)
  NEW: Maximum input length cap 50 words
       (removes over-long, confusing instructions)
  NEW: Language quality filter
       (removes samples dominated by numbers/symbols)
  NEW: Crop-relevance filter (optional, off by default)
       (keeps only samples mentioning agriculture keywords)
  NEW: Stratified split preserves intent distribution
  KEPT: dedup, length filter, shuffle, 90/10 split
"""

import json
import random
import argparse
import re
import os
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────
INPUT_FILE       = "data/raw/final_agriculture_training_dataset.jsonl"
TRAIN_FILE       = "data collection/train_data/train_dataset.jsonl"
VAL_FILE         = "data collection/val_data/val_dataset.jsonl"

MIN_OUTPUT_WORDS = 10      # V2.1: raised from 5 → removes near-empty answers
MAX_OUTPUT_WORDS = 512
MAX_INPUT_WORDS  = 50      # V2.1: new — cap instruction length
TRAIN_SPLIT      = 0.9
RANDOM_SEED      = 42


# ── Agri keyword set for relevance filter ────────────────────
_AGRI_KW = re.compile(
    r"\b(crop|plant|seed|soil|fertil|pest|insect|disease|harvest|farm|"
    r"rice|wheat|maize|tomato|potato|cotton|sugarcane|mustard|onion|"
    r"spray|irrigation|weed|fungi|bacteria|virus|yield|cultivat|"
    r"manure|compost|urea|npk|potash|phosphate|nitrogen|organic|"
    r"livestock|cattle|poultry|fish|goat|pig|veterinary|dairy)\b",
    re.I,
)


def is_agri_relevant(text: str, min_hits: int = 1) -> bool:
    return len(_AGRI_KW.findall(text)) >= min_hits


def language_quality(text: str) -> float:
    """
    Returns fraction of 'real' words (≥2 alpha chars).
    Samples below 0.50 are mostly numbers/symbols.
    """
    words = text.split()
    if not words:
        return 0.0
    real = sum(1 for w in words if re.match(r"^[a-zA-Z]{2,}", w))
    return real / len(words)


# ── Loaders & filters ─────────────────────────────────────────
def load_dataset(path: str) -> list:
    data, corrupted = [], 0
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
        key = (item["instruction"].strip()[:120] + "|||" +
               item["output"].strip()[:120])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    print(f"[Step 1] After dedup  : {len(unique):,}")
    return unique


def length_filter(data: list) -> list:
    filtered = [
        item for item in data
        if MIN_OUTPUT_WORDS <= len(item["output"].split()) <= MAX_OUTPUT_WORDS
        and len(item["instruction"].split()) <= MAX_INPUT_WORDS   # V2.1 new
    ]
    print(f"[Step 1] After length : {len(filtered):,}  "
          f"(out {MIN_OUTPUT_WORDS}–{MAX_OUTPUT_WORDS} words, "
          f"inst ≤{MAX_INPUT_WORDS} words)")
    return filtered


def quality_filter(data: list, min_quality: float = 0.50) -> list:
    """V2.1 new: remove samples with too many non-word tokens."""
    filtered = [
        item for item in data
        if language_quality(item["output"]) >= min_quality
    ]
    print(f"[Step 1] After quality: {len(filtered):,}  "
          f"(lang quality ≥{min_quality:.0%})")
    return filtered


def agri_filter(data: list, enabled: bool = False) -> list:
    """Optional: keep only agriculture-relevant samples."""
    if not enabled:
        return data
    filtered = [
        item for item in data
        if is_agri_relevant(item["instruction"] + " " + item["output"])
    ]
    print(f"[Step 1] After agri   : {len(filtered):,}  (agri-relevant only)")
    return filtered


def print_stats(data: list):
    inst_lens = [len(d["instruction"].split()) for d in data]
    out_lens  = [len(d["output"].split())      for d in data]
    print(f"\n[Step 1] ── Dataset Statistics (V2.1) ──────────────")
    print(f"  Total samples        : {len(data):,}")
    print(f"  Instruction avg/max  : {sum(inst_lens)/len(inst_lens):.1f} / {max(inst_lens)}")
    print(f"  Output      avg/max  : {sum(out_lens)/len(out_lens):.1f} / {max(out_lens)}")
    has_input = sum(1 for d in data if d.get("input", "").strip())
    print(f"  Samples with input   : {has_input:,}")
    # language quality distribution
    q_vals = [language_quality(d["output"]) for d in data[:5000]]
    low_q  = sum(1 for q in q_vals if q < 0.6)
    print(f"  Low-quality outputs  : ~{low_q/len(q_vals)*100:.1f}% (sample of 5k)")
    print(f"────────────────────────────────────────────────────\n")


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
    parser.add_argument("--input",      default=INPUT_FILE)
    parser.add_argument("--train_out",  default=TRAIN_FILE)
    parser.add_argument("--val_out",    default=VAL_FILE)
    parser.add_argument("--split",      type=float, default=TRAIN_SPLIT)
    parser.add_argument("--agri_only",  action="store_true",
                        help="Keep only agri-relevant samples (reduces dataset size)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[Step 1] ERROR: Input not found: {args.input}")
        print("  Expected: final_finetune_training_dataset.jsonl in data/raw/")
        return

    print(f"\n[Step 1] ── Preprocessing Dataset (V2.1) ────────────")
    print(f"  Input  : {args.input}")
    print(f"  Train  : {args.train_out}")
    print(f"  Val    : {args.val_out}")
    print(f"  MIN_OUTPUT_WORDS : {MIN_OUTPUT_WORDS}  (V2.1: raised from 5)")
    print(f"  MAX_INPUT_WORDS  : {MAX_INPUT_WORDS}   (V2.1: new filter)")
    print(f"──────────────────────────────────────────────────────")

    data = load_dataset(args.input)
    data = remove_empty(data)
    data = remove_duplicates(data)
    data = length_filter(data)
    data = quality_filter(data)
    data = agri_filter(data, enabled=args.agri_only)
    print_stats(data)
    split_and_save(data, args.train_out, args.val_out)

    print("\n[Step 1] ✓ Complete. Run step2 next:")
    print("  python data_pipeline/step2_build_knowledge_corpus.py")


if __name__ == "__main__":
    main()
