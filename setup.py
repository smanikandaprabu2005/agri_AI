"""
setup.py
========
One-time setup: builds Word2Vec retriever from your data files.
Run this ONCE before starting the chatbot.

Usage:
    python setup.py                         (uses data/*.jsonl)
    python setup.py --data_dir my_data/     (custom data dir)
"""

import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RETRIEVER_DIR
from retrieval.vector_search import build_retriever, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    args = parser.parse_args()

    print(f"[Setup] Loading data from: {args.data_dir}")

    records = []
    for fname in os.listdir(args.data_dir):
        if fname.endswith(".jsonl"):
            path = os.path.join(args.data_dir, fname)
            r    = load_jsonl(path)
            records.extend(r)
            print(f"  Loaded {len(r)} records from {fname}")

    if not records:
        print("[Setup] No .jsonl files found in data directory.")
        print(f"  Place your E.jsonl or dataset files in: {args.data_dir}")
        return

    print(f"[Setup] Total records: {len(records)}")
    print("[Setup] Building Word2Vec retriever (this takes a few minutes)...")
    ret, vocab = build_retriever(records, save_dir=RETRIEVER_DIR)
    print(f"[Setup] Retriever built and saved → {RETRIEVER_DIR}")
    print(f"[Setup] Documents indexed: {len(ret.docs)}")
    print("\n[Setup] Done! You can now run:")
    print("  python ui/app.py              (web UI)")
    print("  python main.py                (terminal chat)")


if __name__ == "__main__":
    main()
