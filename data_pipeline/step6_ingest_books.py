"""
data_pipeline/step6_ingest_books.py
=====================================
STEP 6 — Ingest Agriculture & Botany Books into Vector DB

This script:
  1. Reads books/PDFs from data/books/ directory
  2. Cleans and chunks each source
  3. Adds to the VectorDB (FAISS + BM25)
  4. Also prepares pretraining text from book content

Supported formats:
  - .txt  — plain text
  - .pdf  — via PyMuPDF (fitz) or pdfminer
  - .json — structured {"title": ..., "content": ...}

Recommended book sources:
  Agriculture / Crop Science:
    • NCERT Class 12 Biology (free PDF)
    • FAO Crop Protection Manuals (free)
    • ICAR crop production guides
    • Integrated Pest Management handbooks
    • Soil Science textbooks (Brady & Weil)
    • Plant Pathology (Agrios)

  Botany:
    • NCERT Class 11 & 12 Biology (Botany part)
    • Introductory Plant Biology (Stern)

  How to get free legal PDFs:
    - NCERT: ncert.nic.in (free download)
    - FAO:   fao.org/publications (open access)
    - ICAR:  icar.org.in/publications

Usage:
  python data_pipeline/step6_ingest_books.py
  python data_pipeline/step6_ingest_books.py --books_dir data/books --for_pretrain
"""

import os, re, json, argparse, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.vector_db import VectorDB, clean_text, chunk_text

# ── Config ────────────────────────────────────────────────────
BOOKS_DIR        = "data/books"
VECTOR_DB_DIR    = "saved_models/vector_index"
PRETRAIN_BOOKS   = "data_pipeline/processed/books_pretrain.txt"
CHUNK_SIZE       = 256    # words per chunk for RAG
OVERLAP          = 32     # overlap between chunks
PRETRAIN_CHUNK   = 512    # larger chunks for pretraining


def clean_markdown_content(text: str) -> str:
    """Clean markdown formatting from text."""
    text = text.replace('**', '')
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Source catalogue ─────────────────────────────────────────
# Add your books here. Each entry tells the system what the source is.
BOOK_CATALOGUE = {

    # New PDFs from user
    "class9.pdf":                           {"name": "Class 9 Textbook",                           "type": "school",          "use": "both"},
    "class11.pdf":                          {"name": "Class 11 Textbook",                          "type": "school",          "use": "both"},
    "Agricultural-Business-Management.pdf": {"name": "Agricultural Business Management",           "type": "agribusiness",    "use": "both"},
    "Field-Crop-Kharif.pdf":                {"name": "Field Crop Kharif",                          "type": "crop_science",    "use": "both"},
    "Introduction-to-Soil-Science.pdf":     {"name": "Introduction to Soil Science",               "type": "soil_science",    "use": "both"},
    "Soil-Chemistry-Soil-Fertility-Nutrient-Management.pdf": {"name": "Soil Chemistry Soil Fertility Nutrient Management", "type": "soil_science",    "use": "both"},
    "Livestock-Production-and-Management.pdf": {"name": "Livestock Production and Management",     "type": "livestock",       "use": "both"},
    "Crop-Pests-and-Stored-Grain-Pests.pdf": {"name": "Crop Pests and Stored Grain Pests",         "type": "pest_control",    "use": "both"},
    "Disease-of-Horticultural-Crops-and-their-Management.pdf": {"name": "Disease of Horticultural Crops and their Management", "type": "plant_disease",   "use": "both"},
    "Diseases-of-Field-Crops-and-Their-Management.pdf": {"name": "Diseases of Field Crops and Their Management", "type": "plant_disease",   "use": "both"},
    "Weed-Management.pdf":                  {"name": "Weed Management",                            "type": "weed_management", "use": "both"},
    "Manures-Fertilizers-Agrochemicals.pdf": {"name": "Manures Fertilizers Agrochemicals",         "type": "fertilizer",      "use": "both"},
    "Irrigation-Engineering.pdf":           {"name": "Irrigation Engineering",                     "type": "irrigation",      "use": "both"},
    "Micro-Irrigation-Systems-Design.pdf":  {"name": "Micro Irrigation Systems Design",            "type": "irrigation",      "use": "both"},
    "Crop-Pests-and-Stored-Grain-Pests-and-Their-Management.pdf": {"name": "Crop Pests and Stored Grain Pests and Their Management", "type": "pest_control", "use": "both"},
    "crop_production_and _management.pdf":  {"name": "Crop Production and Management",            "type": "crop_science",    "use": "both"},
    "Disease-of-Horticultural-Crops-their-Management1.pdf": {"name": "Disease of Horticultural Crops and Their Management", "type": "plant_disease",   "use": "both"},
    "FIELD-CROPS-RABI-with-Multiple-choice-questions.pdf": {"name": "Field Crops Rabi Multiple Choice Questions", "type": "crop_science", "use": "both"},
    "Fundamental-of-Soil-Water-Conservation-Eng.pdf": {"name": "Fundamental of Soil Water Conservation Engineering", "type": "water_management", "use": "both"},
    "iesc112.pdf":                           {"name": "IESC 112 Agriculture Reference",             "type": "agriculture",     "use": "both"},
    "Post-Harvest-Management-Value-Addition-of-Fruits-vegetable.pdf": {"name": "Post Harvest Management and Value Addition of Fruits and Vegetables", "type": "postharvest", "use": "both"},
    "PRINCIPLES-OF-SEED-TECHNOLOGY.pdf":     {"name": "Principles of Seed Technology",              "type": "crop_science",    "use": "both"},
    "Production-Technology-of-Fruit-Crops.pdf": {"name": "Production Technology of Fruit Crops",       "type": "horticulture",    "use": "both"},
    "Production-Technology-of-Spices-Aromatic.pdf": {"name": "Production Technology of Spices and Aromatic Crops", "type": "horticulture", "use": "both"},
    "Production-Technology-of-Vegetables.pdf": {"name": "Production Technology of Vegetables",       "type": "horticulture",    "use": "both"},
    "Protected-Cultivation-Post-Harvest-Technology.pdf": {"name": "Protected Cultivation and Post Harvest Technology", "type": "protected_cultivation", "use": "both"},
    "Water-Management-including-Micro-Irrigation.pdf": {"name": "Water Management including Micro Irrigation", "type": "irrigation", "use": "both"},
}


# ── PDF text extraction ───────────────────────────────────────
def extract_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF (best) or pdfminer fallback."""
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text as pe
        return pe(path)
    except ImportError:
        pass

    print(f"  [!] Cannot extract PDF: {path}")
    print(f"      Install: pip install PyMuPDF   OR   pip install pdfminer.six")
    return ""


# ── Source reader ─────────────────────────────────────────────
def read_source(file_path: str) -> str:
    """Read text from various file formats."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(file_path)

    elif ext == ".txt":
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".json":
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            parts = []
            for item in data:
                if isinstance(item, dict):
                    parts.append(item.get("content","") or item.get("text",""))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        elif isinstance(data, dict):
            return data.get("content","") or data.get("text","") or str(data)
        return str(data)

    elif ext in (".jsonl",):
        lines = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    lines.append(obj.get("text","") or obj.get("content",""))
                except: pass
        return "\n".join(lines)

    else:
        # Try reading as text anyway
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            return ""


# ── Quality filter ────────────────────────────────────────────
def is_quality_chunk(text: str, min_words: int = 15) -> bool:
    """Filter out low-quality chunks (headers, page numbers, etc.)."""
    words = text.split()
    if len(words) < min_words:
        return False
    # Too many numbers (likely table/figure)
    num_ratio = sum(1 for w in words if re.match(r"^\d+\.?\d*$", w)) / len(words)
    if num_ratio > 0.4:
        return False
    # Too many single chars (likely OCR noise)
    char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
    if char_ratio > 0.3:
        return False
    return True


# ── Ingest a single source ────────────────────────────────────
def ingest_source(
    file_path: str,
    catalogue_entry: dict,
    db: VectorDB,
    pretrain_f,
    for_pretrain: bool,
) -> dict:
    name = catalogue_entry["name"]
    stype = catalogue_entry["type"]
    use = catalogue_entry.get("use", "both")

    print(f"\n  📖 Processing: {name}")
    print(f"     File: {file_path}")

    raw = read_source(file_path)
    if not raw or len(raw) < 100:
        print(f"     ⚠ Empty or too short — skipped")
        return {"source": name, "chunks_rag": 0, "chunks_pretrain": 0}

    cleaned = clean_text(raw)
    # Remove markdown formatting noise
    cleaned = clean_markdown_content(cleaned)
    print(f"     Raw chars: {len(raw):,}  →  Cleaned: {len(cleaned):,}")

    # For RAG vector DB
    rag_count = 0
    if use in ("rag", "both"):
        chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        good_chunks = [c for c in chunks if is_quality_chunk(c)]
        metadatas = [{"source": name, "type": stype, "chunk_id": i,
                      "file": os.path.basename(file_path)}
                     for i in range(len(good_chunks))]
        db.add_documents(good_chunks, metadatas)
        rag_count = len(good_chunks)
        print(f"     → RAG chunks: {rag_count:,}")

    # For pretraining text
    pretrain_count = 0
    if for_pretrain and use in ("pretrain", "both") and pretrain_f:
        pt_chunks = chunk_text(cleaned, chunk_size=PRETRAIN_CHUNK, overlap=64)
        good_pt = [c for c in pt_chunks if is_quality_chunk(c, min_words=20)]
        for c in good_pt:
            pretrain_f.write(c + "\n")
        pretrain_count = len(good_pt)
        print(f"     → Pretrain chunks: {pretrain_count:,}")

    return {"source": name, "chunks_rag": rag_count, "chunks_pretrain": pretrain_count}


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books_dir",    default=BOOKS_DIR)
    parser.add_argument("--vector_dir",   default=VECTOR_DB_DIR)
    parser.add_argument("--pretrain_out", default=PRETRAIN_BOOKS)
    parser.add_argument("--for_pretrain", action="store_true",
                        help="Also extract chunks for pretraining")
    parser.add_argument("--chunk_size",   type=int, default=CHUNK_SIZE)
    parser.add_argument("--model",        default=VectorDB.DEFAULT_MODEL,
                        help="Sentence-transformers model name")
    args = parser.parse_args()

    books_dir = Path(args.books_dir)
    if not books_dir.exists():
        books_dir.mkdir(parents=True)
        print(f"[Step 6] Created books directory: {books_dir}")
        print(f"\n  ⚠ Place your book files (PDF/TXT) in: {books_dir}/")
        print(f"  Then run this script again.\n")
        print(f"  Recommended sources:")
        for fname, meta in BOOK_CATALOGUE.items():
            print(f"    {fname:40s} — {meta['name']}")
        return

    # Find available books
    available = []
    for fname, meta in BOOK_CATALOGUE.items():
        fpath = books_dir / fname
        if fpath.exists():
            available.append((str(fpath), meta))

    # Also auto-detect unlisted files
    all_files = list(books_dir.glob("**/*.pdf")) + \
                list(books_dir.glob("**/*.txt")) + \
                list(books_dir.glob("**/*.json"))
    for fpath in all_files:
        if fpath.name not in BOOK_CATALOGUE:
            print(f"  Auto-detecting: {fpath.name}")
            meta = {
                "name":  fpath.stem.replace("_", " ").title(),
                "type":  "agriculture",
                "use":   "both",
            }
            available.append((str(fpath), meta))

    if not available:
        print(f"\n[Step 6] No book files found in {books_dir}/")
        print(f"\n  Add PDF or TXT files there. Recommended:")
        for fname, meta in list(BOOK_CATALOGUE.items())[:8]:
            print(f"    {fname} — {meta['name']}")
        return

    print(f"\n[Step 6] ── Ingesting {len(available)} books ───────────────")
    print(f"  Books dir  : {books_dir}")
    print(f"  Vector DB  : {args.vector_dir}")
    print(f"  For pretrain: {args.for_pretrain}")
    print(f"─────────────────────────────────────────────────")

    # Load or create vector DB
    vector_dir = Path(args.vector_dir)
    if (vector_dir / "config.json").exists():
        print(f"\n[Step 6] Loading existing VectorDB from {vector_dir}")
        db = VectorDB.load(str(vector_dir))
    else:
        print(f"\n[Step 6] Creating new VectorDB")
        db = VectorDB(model_name=args.model)

    # Open pretrain output
    Path(args.pretrain_out).parent.mkdir(parents=True, exist_ok=True)
    pretrain_mode = "a" if Path(args.pretrain_out).exists() else "w"

    summary = []
    with open(args.pretrain_out, pretrain_mode, encoding="utf-8") as pf:
        for fpath, meta in available:
            result = ingest_source(
                fpath, meta, db,
                pf if args.for_pretrain else None,
                args.for_pretrain,
            )
            summary.append(result)

    # Save the updated DB
    db.save(args.vector_dir)

    # Summary
    total_rag = sum(r["chunks_rag"] for r in summary)
    total_pt  = sum(r["chunks_pretrain"] for r in summary)

    print(f"\n[Step 6] ── Summary ────────────────────────────────")
    print(f"  Books processed  : {len(summary):,}")
    print(f"  Total RAG chunks : {total_rag:,}")
    if args.for_pretrain:
        print(f"  Pretrain chunks  : {total_pt:,}")
    print(f"  Vector DB stats  : {db.stats()}")
    print(f"\n[Step 6] ✓ Complete!")
    print(f"\n  Next steps:")
    print(f"  1. Update context_builder to use VectorDB:")
    print(f"     from retrieval.vector_db import VectorDB")
    print(f"  2. If --for_pretrain used, add to pretraining:")
    print(f"     Merge {args.pretrain_out} with processed/knowledge_corpus.txt")
    print(f"  3. Re-tokenize if adding to pretraining.")


if __name__ == "__main__":
    main()