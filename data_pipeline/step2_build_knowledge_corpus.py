"""
data_pipeline/step2_build_knowledge_corpus.py
==============================================
STEP 2 of 5 — Knowledge Corpus Preparation

Adapts V1 preprocess_knowledge_corpus.py for the V2 project.

What it does:
  1. Loads ICAR_Text_Extracted.json  (47.5 MB — structured agricultural PDFs)
  2. Loads research_wikipedia_final_dataset.txt  (1 MB — research text)
  3. Cleans both: removes URLs, OCR noise, repeated chars, weird symbols
  4. Splits into sentences
  5. Keeps sentences longer than 80 characters
  6. Saves to processed/knowledge_corpus.txt

Input:
  data/raw/ICAR_Text_Extracted.json
  data/raw/research_wikipedia_final_dataset.txt

Output:
  processed/knowledge_corpus.txt

Usage:
  python data_pipeline/step2_build_knowledge_corpus.py
"""

import json
import re
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
ICAR_FILE     = "data/raw/ICAR_Text_Extracted.json"
RESEARCH_FILE = "data/raw/research_wikipedia_final_dataset.txt"
OUTPUT_FILE   = "processed/knowledge_corpus.txt"
MIN_SENT_LEN  = 80    # minimum characters per sentence to keep


# ── Text cleaner ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+|www\.\S+",          " ", text)   # URLs
    text = re.sub(r"--- Page \d+ ---",           " ", text)   # page markers
    text = re.sub(r"Web Url.*",                  " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Z]{6,}\b",             " ", text)   # OCR noise
    text = re.sub(r"(.)\1{5,}",                 " ", text)   # repeated chars
    text = re.sub(r"[^a-zA-Z0-9.,;:()\-\' ]+", " ", text)   # special symbols
    text = re.sub(r"\s+",                        " ", text)   # whitespace
    return text.strip()


def split_sentences(text: str) -> list:
    """Split on sentence-ending punctuation."""
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENT_LEN]


# ── ICAR processor ────────────────────────────────────────────
def process_icar(icar_path: str, out_f) -> int:
    total = 0
    with open(icar_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for category, files in data.items():
        for filename, raw_text in files.items():
            if not isinstance(raw_text, str):
                continue
            cleaned = clean_text(raw_text)
            for sent in split_sentences(cleaned):
                out_f.write(sent + "\n")
                total += 1

    print(f"[Step 2] ICAR sentences written    : {total:,}")
    return total


# ── Research/Wikipedia processor ─────────────────────────────
def process_research(research_path: str, out_f) -> int:
    total = 0
    with open(research_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = clean_text(line)
            for sent in split_sentences(cleaned):
                out_f.write(sent + "\n")
                total += 1

    print(f"[Step 2] Research sentences written: {total:,}")
    return total


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icar",     default=ICAR_FILE)
    parser.add_argument("--research", default=RESEARCH_FILE)
    parser.add_argument("--output",   default=OUTPUT_FILE)
    args = parser.parse_args()

    missing = []
    if not Path(args.icar).exists():     missing.append(args.icar)
    if not Path(args.research).exists(): missing.append(args.research)
    if missing:
        for m in missing:
            print(f"[Step 2] ERROR: File not found: {m}")
        print("  Place ICAR_Text_Extracted.json and research_wikipedia_final_dataset.txt in data/raw/")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 2] ── Building Knowledge Corpus ───────────")
    print(f"  ICAR     : {args.icar}")
    print(f"  Research : {args.research}")
    print(f"  Output   : {args.output}")
    print(f"─────────────────────────────────────────────────")

    with open(args.output, "w", encoding="utf-8") as out_f:
        icar_count     = process_icar(args.icar, out_f)
        research_count = process_research(args.research, out_f)

    total = icar_count + research_count
    size  = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\n[Step 2] Total sentences  : {total:,}")
    print(f"[Step 2] Output file size : {size:.1f} MB")
    print(f"[Step 2] Saved → {args.output}")

    print("\n[Step 2] ✓ Complete. Run step3 next:")
    print("  python data_pipeline/step3_train_tokenizer.py")


if __name__ == "__main__":
    main()
