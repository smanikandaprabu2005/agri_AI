"""
setup.py
========
One-time setup: builds the hybrid BM25+TF-IDF retriever from your data files.
Run this ONCE after step 1 completes (which outputs JSONL files).

Usage:
    python setup.py                                    (uses data_pipeline/data collection/)
    python setup.py --data_dir data/                   (custom data dir)
    python setup.py --retriever bm25                   (force BM25 only)
    python setup.py --retriever hybrid                 (default: BM25 + TF-IDF)
"""

import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, RETRIEVER_DIR, RETRIEVER_TYPE
from retrieval.vector_search import build_retriever, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data_pipeline/data collection/")
    parser.add_argument("--retriever", default=RETRIEVER_TYPE,
                        choices=["bm25", "tfidf", "hybrid", "word2vec"])
    args = parser.parse_args()

    # Override retriever type for this run
    os.environ["RETRIEVER_TYPE_OVERRIDE"] = args.retriever

    print(f"[Setup] Loading data from: {args.data_dir}")
    print(f"[Setup] Retriever type   : {args.retriever}")

    records = []
    for fname in sorted(os.listdir(args.data_dir)):
        if fname.endswith(".jsonl"):
            path = os.path.join(args.data_dir, fname)
            r    = load_jsonl(path)
            records.extend(r)
            print(f"  Loaded {len(r):,} records from {fname}")

    if not records:
        print("[Setup] No .jsonl files found in data directory.")
        print(f"  Place your dataset .jsonl files in: {args.data_dir}")
        return

    print(f"\n[Setup] Total records: {len(records):,}")
    print(f"[Setup] Building retriever (this may take 1-2 minutes for large datasets)...")

    import config
    config.RETRIEVER_TYPE = args.retriever  # override at runtime

    ret, _ = build_retriever(records, save_dir=RETRIEVER_DIR)
    print(f"[Setup] Retriever built and saved → {RETRIEVER_DIR}")

    n_docs = len(ret.docs) if hasattr(ret, "docs") else "?"
    print(f"[Setup] Documents indexed: {n_docs}")

    # Quick sanity test
    print("\n[Setup] Quick retrieval test:")
    test_queries = [
        "How to control aphids on tomato?",
        "Fertilizer dose for rice per acre",
        "Late blight management in potato",
    ]
    for q in test_queries:
        try:
            results = ret.retrieve(q, top_k=2)
            top_score = results[0][0] if results else 0.0
            print(f"  '{q[:50]}' → top_score={top_score:.3f}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n[Setup] Done! You can now run:")
    print("  python ui/app.py              (web UI)")
    print("  python main.py                (terminal chat)")


if __name__ == "__main__":
    main()