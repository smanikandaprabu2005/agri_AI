"""
data_pipeline/step2_build_knowledge_corpus.py
==============================================
STEP 2 of 5 — Knowledge Corpus Preparation

What it does:
  1. Loads ICAR_Text_Extracted.json  (47.5 MB — structured agricultural PDFs)
  2. Loads research_wikipedia_final_dataset.txt  (1 MB — research text)
  3. Loads bsc_agri.txt  (NEW — BSc Agriculture curriculum, ~276 KB)
  4. Cleans all three, splits into sentences
  5. Keeps sentences longer than MIN_SENT_LEN characters
  6. Saves to processed/knowledge_corpus.txt

Input:
  data/raw/ICAR_Text_Extracted.json
  data/raw/research_wikipedia_final_dataset.txt
  data/raw/bsc_agri.txt                          ← NEW

Output:
  processed/knowledge_corpus.txt

Why bsc_agri.txt helps:
  - Covers crop physiology, genetics, entomology, farm management, seed tech
  - Adds ~2,000 high-quality agricultural sentences
  - Improves model's conceptual understanding beyond just Q&A patterns
  - Particularly strong for: IPM, plant pathology, crop varieties, agronomy

Usage:
  python data_pipeline/step2_build_knowledge_corpus.py
  python data_pipeline/step2_build_knowledge_corpus.py --bsc data/raw/bsc_agri.txt
"""

import json
import re
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
ICAR_FILE           = "data_pipeline/data/raw/ICAR_Text_Extracted.json"
CLEANED_ICAR_FILE   = "data_pipeline/data/raw/cleaned_ICAR_Text_Extracted.json"
RESEARCH_FILE       = "data_pipeline/data/raw/research_wikipedia_final_dataset.txt"
CLEANED_RESEARCH_FILE = "data_pipeline/data/raw/cleaned_research_wikipedia_final_dataset.txt"
BSC_AGRI_FILE       = "data_pipeline/data/raw/bsc agri.txt"          # NEW
CLEANED_BSC_FILE    = "data_pipeline/data/raw/cleaned_bsc_agri.txt"
OUTPUT_FILE         = "data_pipeline/processed/knowledge_corpus.txt"
MIN_SENT_LEN        = 40   # minimum characters per sentence to keep


# ── Text cleaner ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+|www\.\S+",          " ", text)   # URLs
    text = re.sub(r"--- Page \d+ ---",           " ", text)   # page markers
    text = re.sub(r"Web Url.*",                  " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Z]{6,}\b",             " ", text)   # OCR noise
    text = re.sub(r"(.)\1{5,}",                 " ", text)   # repeated chars
    text = re.sub(r"[^a-zA-Z0-9.,;:()\-\'%/@ ]+", " ", text)  # special symbols
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

    print(f"[Step 2] ICAR sentences written      : {total:,}")
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

    print(f"[Step 2] Research sentences written  : {total:,}")
    return total


# ── BSc Agriculture curriculum processor (NEW) ───────────────
def process_bsc_agri(bsc_path: str, out_f) -> int:
    """
    Process the BSc Agriculture curriculum text.

    This file covers:
      - Biochemistry & Biotechnology
      - Crop Science (Field Crops, Agronomy, Crop Physiology)
      - Principles of Genetics & Plant Breeding
      - Seed Technology
      - Farming Systems & Sustainable Agriculture
      - Weed Management
      - Entomology & IPM
      - Plant Pathology
      - Agricultural Economics & Marketing
      - Post Harvest Technology
      - Farm Power & Renewable Energy

    Each topic section provides dense conceptual knowledge that
    improves the model's agricultural domain understanding.
    """
    total = 0

    with open(bsc_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Split into paragraphs first (text is structured as sections)
    # Each paragraph separated by double newlines or topic headers
    paragraphs = re.split(r"\n{2,}", raw)

    for para in paragraphs:
        if len(para.strip()) < 20:
            continue
        cleaned = clean_text(para)
        for sent in split_sentences(cleaned):
            # Extra filter: skip curriculum header lines (all caps topics)
            words = sent.split()
            alpha_words = [w for w in words if re.match(r"^[a-zA-Z]+$", w)]
            if not alpha_words:
                continue
            # Skip lines that are mostly 1-2 letter words (OCR artifacts)
            avg_word_len = sum(len(w) for w in alpha_words) / len(alpha_words)
            if avg_word_len < 3.0:
                continue
            out_f.write(sent + "\n")
            total += 1

    print(f"[Step 2] BSc Agri sentences written  : {total:,}")
    return total


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icar",     default=ICAR_FILE)
    parser.add_argument("--research", default=RESEARCH_FILE)
    parser.add_argument("--bsc",      default=BSC_AGRI_FILE)
    parser.add_argument("--output",   default=OUTPUT_FILE)
    parser.add_argument("--no_bsc",   action="store_true",
                        help="Skip BSc Agriculture file if not available")
    args = parser.parse_args()

    icar_path = Path(args.icar)
    research_path = Path(args.research)
    bsc_path = Path(args.bsc)

    if args.icar == ICAR_FILE and Path(CLEANED_ICAR_FILE).exists():
        icar_path = Path(CLEANED_ICAR_FILE)
        print(f"[Step 2] Using cleaned ICAR file: {icar_path}")
    if args.research == RESEARCH_FILE and Path(CLEANED_RESEARCH_FILE).exists():
        research_path = Path(CLEANED_RESEARCH_FILE)
        print(f"[Step 2] Using cleaned research file: {research_path}")
    if args.bsc == BSC_AGRI_FILE and Path(CLEANED_BSC_FILE).exists():
        bsc_path = Path(CLEANED_BSC_FILE)
        print(f"[Step 2] Using cleaned BSc Agri file: {bsc_path}")

    # Check required files
    missing = []
    if not icar_path.exists():     missing.append(str(icar_path))
    if not research_path.exists(): missing.append(str(research_path))
    if missing:
        for m in missing:
            print(f"[Step 2] ERROR: File not found: {m}")
        print("  Place ICAR_Text_Extracted.json and research_wikipedia_final_dataset.txt in data/raw/")
        return

    # Check optional BSc file
    use_bsc = (not args.no_bsc) and bsc_path.exists()
    if not use_bsc and not args.no_bsc:
        print(f"[Step 2] INFO: BSc Agri file not found at {args.bsc}")
        print(f"         Copy bsc_agri.txt to data/raw/ to include it.")
        print(f"         Continuing without it (use --no_bsc to suppress this message).")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Step 2] ── Building Knowledge Corpus ───────────")
    print(f"  ICAR     : {icar_path}")
    print(f"  Research : {research_path}")
    print(f"  BSc Agri : {bsc_path if use_bsc else '(skipped)'}")
    print(f"  Output   : {args.output}")
    print(f"─────────────────────────────────────────────────")

    with open(args.output, "w", encoding="utf-8") as out_f:
        icar_count     = process_icar(str(icar_path), out_f)
        research_count = process_research(str(research_path), out_f)
        bsc_count      = process_bsc_agri(str(bsc_path), out_f) if use_bsc else 0

    total = icar_count + research_count + bsc_count
    size  = Path(args.output).stat().st_size / 1024 / 1024

    print(f"\n[Step 2] ── Summary ────────────────────────────")
    print(f"  ICAR sentences     : {icar_count:,}")
    print(f"  Research sentences : {research_count:,}")
    if use_bsc:
        print(f"  BSc Agri sentences : {bsc_count:,}")
    print(f"  Total sentences    : {total:,}")
    print(f"  Output file size   : {size:.1f} MB")
    print(f"  Saved → {args.output}")

    print("\n[Step 2] ✓ Complete. Run step3 next:")
    print("  python data_pipeline/step3_train_tokenizer.py")


if __name__ == "__main__":
    main()