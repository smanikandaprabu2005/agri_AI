# SageStorm V2 — Upgraded RAG System Guide

## What Changed & Why

### Old Architecture (Word2Vec)
```
Query → Word2Vec Embeddings → Cosine Similarity → Top-5 Docs → 600-char context
```
**Problems:** Word2Vec has no sentence-level understanding, misses synonyms,
context limited to 600 chars, no source diversity.

### New Architecture (VectorDB + Hybrid)
```
Query → Sentence-Transformers (768-dim) → FAISS ANN Search ─┐
      → BM25 Keyword Search ─────────────────────────────────┤→ Rerank → 1200-char context
      → Word2Vec Fallback (if VectorDB empty) ───────────────┘
```
**Improvements:**
- Sentence-level semantic understanding
- Hybrid retrieval (dense + sparse)
- Up to 2x larger context window
- Source attribution (which book the answer came from)
- Scales to millions of chunks

---

## Installation

```bash
# Core dependencies
pip install sentence-transformers faiss-cpu PyMuPDF rank-bm25

# Optional (faster PDF reading)
pip install pdfminer.six

# Full install
pip install sentence-transformers faiss-cpu PyMuPDF rank-bm25 pdfminer.six
```

---

## Step-by-Step Setup

### Step 1: Get Book Sources

**Free legal sources:**

| Book | Source | Format |
|------|--------|--------|
| NCERT Biology 11 & 12 | ncert.nic.in → Textbooks | PDF |
| NCERT Science 9 & 10 | ncert.nic.in → Textbooks | PDF |
| FAO Crop Protection Manual | fao.org/publications | PDF |
| FAO Soil Fertility Guide | fao.org/publications | PDF |
| ICAR Publications | icar.org.in/publications | PDF |
| Organic Farming (IFOAM) | ifoam.bio | PDF |

**Download script (optional):**
```bash
# NCERT books (direct download)
wget "https://ncert.nic.in/textbook/pdf/kebo101.pdf" -O data/books/ncert_biology_11.pdf
wget "https://ncert.nic.in/textbook/pdf/lebo101.pdf" -O data/books/ncert_biology_12.pdf
```

### Step 2: Place Books
```
your_project/
├── data/
│   └── books/            ← put PDF/TXT files here
│       ├── ncert_biology_11.pdf
│       ├── ncert_biology_12.pdf
│       ├── icar_crop_production.pdf
│       ├── fao_pest_management.pdf
│       └── ...
```

### Step 3: Run Ingestion
```bash
# Basic ingestion (RAG only)
python data_pipeline/step6_ingest_books.py

# With pretraining extraction
python data_pipeline/step6_ingest_books.py --for_pretrain

# Custom directory
python data_pipeline/step6_ingest_books.py --books_dir /path/to/books
```

### Step 4: Run Chatbot
```bash
# Auto-detects VectorDB
python main.py

# Explicit VectorDB
python main.py --verbose

# Word2Vec only (no VectorDB)
python main.py --no_vector_db
```

---

## How Books Are Used

### For RAG (Vector DB)
Books go into the FAISS index as 256-word chunks with overlap.
When a farmer asks a question, the system retrieves the top-5 most relevant
chunks from ALL books combined, plus the original training data.

**Example:**
- Farmer: "My rice leaves are turning yellow at the tip"
- System retrieves: chunk from ICAR Rice Guide + chunk from NCERT Botany
- Combined with weather data → high quality answer

### For Pretraining
Books are chunked into 512-word pieces and added to the pretraining corpus.
This teaches the model domain-specific language BEFORE fine-tuning on Q&A pairs.

**Workflow:**
```bash
# 1. Ingest with pretrain flag
python data_pipeline/step6_ingest_books.py --for_pretrain

# 2. Merge with existing corpus
cat processed/knowledge_corpus.txt processed/books_pretrain.txt > processed/full_corpus.txt

# 3. Re-tokenize pretrain data
python data_pipeline/step4_tokenize_pretrain.py --corpus processed/full_corpus.txt

# 4. Retrain (or fine-tune from scratch)
python train/pretrain.py
```

---

## Architecture Decision: Which Books Go Where?

| Use Case | RAG Vector DB | Pretraining |
|----------|--------------|-------------|
| Q&A factual lookup | ✅ Best | Optional |
| Procedures & doses | ✅ Best | Optional |
| Language patterns | ❌ | ✅ Best |
| Conceptual understanding | ✅ Good | ✅ Good |
| School textbooks | ❌ (too general) | ✅ Best |

**Rule of thumb:**
- If you want the model to **cite it** → put in Vector DB
- If you want the model to **talk like it** → put in Pretraining
- For core agriculture references → **both**

---

## Configuration in step6_ingest_books.py

Each book in `BOOK_CATALOGUE` has a `"use"` field:

```python
"ncert_biology_12.pdf": {
    "name":  "NCERT Biology Class 12",
    "type":  "textbook",
    "use":   "both"      # "rag", "pretrain", or "both"
}
```

Add your own books:
```python
"my_custom_book.pdf": {
    "name":  "My Custom Guide",
    "type":  "agriculture",   # for filtering
    "use":   "both",
}
```

---

## VectorDB Performance

| Collection Size | Search Time | Memory |
|----------------|-------------|--------|
| < 10,000 chunks | < 5ms | ~50 MB |
| 10K - 100K chunks | < 20ms | ~200 MB |
| 100K - 1M chunks | < 50ms | ~2 GB |

**Expected from recommended books (~15 books):**
- ~50,000-100,000 chunks
- ~200-400 MB disk
- < 20ms search latency

---

## Troubleshooting

**"sentence-transformers not installed"**
```bash
pip install sentence-transformers
```

**"faiss-cpu not installed"**
```bash
pip install faiss-cpu
# On Mac with M1/M2:
pip install faiss-cpu --no-binary :all:
```

**"Cannot extract PDF"**
```bash
pip install PyMuPDF
```

**VectorDB returns poor results:**
- Try a better encoder: `--model sentence-transformers/all-mpnet-base-v2`
- Lower `min_score` threshold in `context_builder_v2.py`
- Reduce chunk size to 128 words for more precise retrieval

---

## Sentence-Transformers Model Options

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` (default) | 80MB | Fast | Good |
| `all-MiniLM-L12-v2` | 120MB | Medium | Better |
| `all-mpnet-base-v2` | 420MB | Slow | Best |
| `paraphrase-multilingual-MiniLM-L12-v2` | 120MB | Medium | Good (multi-lang) |

For Indian agriculture (Hindi/English mix), consider:
```bash
python data_pipeline/step6_ingest_books.py \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

---

## Full Pipeline After Books Added

```
1. EXISTING PIPELINE (steps 1-5) — keep as is

2. NEW: Book Ingestion
   python data_pipeline/step6_ingest_books.py --for_pretrain

3. IF retraining model with book content:
   cat processed/knowledge_corpus.txt processed/books_pretrain.txt \
       > processed/full_corpus.txt
   python data_pipeline/step4_tokenize_pretrain.py \
       --corpus processed/full_corpus.txt
   python train/pretrain.py
   python train/finetune.py

4. Run chatbot (VectorDB auto-loaded):
   python main.py
```

---

## Expected Quality Improvement

With NCERT + ICAR + FAO books in the Vector DB:

| Query Type | Before (Word2Vec) | After (VectorDB + Books) |
|-----------|-------------------|--------------------------|
| Pest identification | Moderate | High |
| Disease symptoms | Moderate | High |
| Dose/quantity questions | Good | Very High |
| Botanical explanations | Poor | Good |
| Scientific terminology | Poor | High |
| Crop-specific advice | Good | Very High |
