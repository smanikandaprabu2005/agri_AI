"""
setup.py
========
One-time setup: builds Word2Vec retriever from your data files.
Run this ONCE before starting the chatbot.

FIXES:
  FIX 1: Import from retrieval.vector_search (correct spelling)
          Original had "retrival" typo which crashes on Linux
  FIX 2: Guard against empty records before calling build_retriever
  FIX 3: Check that RETRIEVER_DIR doesn't already have a built retriever
         (avoid re-building unnecessarily)
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RETRIEVER_DIR

# FIX 1: Correct import path (was "retrival" — typo causes ImportError on Linux)
from retrieval.vector_search import build_retriever, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--force",    action="store_true",
                        help="Force rebuild even if retriever already exists")
    args = parser.parse_args()

    # FIX 3: Skip if already built unless --force
    retriever_exists = (
        os.path.exists(os.path.join(RETRIEVER_DIR, "vocab.pkl")) and
        os.path.exists(os.path.join(RETRIEVER_DIR, "vecs.npy"))
    )
    if retriever_exists and not args.force:
        print(f"[Setup] Retriever already exists at {RETRIEVER_DIR}")
        print(f"[Setup] Use --force to rebuild.")
        print(f"\n[Setup] You can now run:")
        print(f"  python ui/app.py              (web UI)")
        print(f"  python main.py                (terminal chat)")
        return

    print(f"[Setup] Loading data from: {args.data_dir}")

    records = []
    if not os.path.isdir(args.data_dir):
        print(f"[Setup] ERROR: Data directory not found: {args.data_dir}")
        return

    for fname in os.listdir(args.data_dir):
        if fname.endswith(".jsonl"):
            path = os.path.join(args.data_dir, fname)
            r    = load_jsonl(path)
            records.extend(r)
            print(f"  Loaded {len(r)} records from {fname}")

    # FIX 2: Guard against empty records
    if not records:
        print("[Setup] ERROR: No .jsonl files found in data directory.")
        print(f"  Place your dataset files in: {args.data_dir}")
        print(f"  Expected format: each line is a JSON object with "
              f"'instruction' and 'output' fields.")
        return

    print(f"[Setup] Total records: {len(records)}")
    print("[Setup] Building Word2Vec retriever (this may take a few minutes)...")
    ret, vocab = build_retriever(records, save_dir=RETRIEVER_DIR)
    print(f"[Setup] Retriever built and saved → {RETRIEVER_DIR}")
    print(f"[Setup] Documents indexed: {len(ret.docs)}")
    print(f"\n[Setup] Done! You can now run:")
    print(f"  python ui/app.py              (web UI)")
    print(f"  python main.py                (terminal chat)")


if __name__ == "__main__":
    main()