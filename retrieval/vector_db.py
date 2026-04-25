"""
retrieval/vector_db.py
======================
Dense Vector Database using FAISS + Sentence-Transformers.

Replaces the old Word2Vec retriever with a proper semantic search engine.

Features:
  - Sentence-Transformers for dense embeddings (768-dim)
  - FAISS for fast approximate nearest-neighbor search
  - Metadata storage (source, chunk_id, page_num)
  - Chunking with overlap for better retrieval
  - Hybrid search: dense + BM25 keyword fallback
  - Persistent index (save/load)

Install:
  pip install faiss-cpu sentence-transformers rank-bm25

Usage:
  from retrieval.vector_db import VectorDB
  db = VectorDB()
  db.add_documents(texts, metadatas)
  db.save("saved_models/vector_index")
  results = db.search("how to control aphids", top_k=5)
"""

import os, re, json, pickle, sys
import numpy as np
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Text chunking ─────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32,
    min_chunk: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks of roughly chunk_size words.
    Tries to break on sentence boundaries first.
    """
    # Sentence split
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        words = sent.split()
        wc = len(words)
        if current_len + wc > chunk_size and current:
            chunk = " ".join(current)
            if len(chunk) >= min_chunk:
                chunks.append(chunk)
            # overlap: keep last `overlap` words
            current = current[-overlap:] if overlap else []
            current_len = len(current)
        current.extend(words)
        current_len += wc

    if current:
        chunk = " ".join(current)
        if len(chunk) >= min_chunk:
            chunks.append(chunk)

    return chunks


def clean_text(text: str) -> str:
    """Clean raw text from PDFs/books."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"--- Page \d+ ---", " ", text)
    text = re.sub(r"\b[A-Z]{6,}\b", " ", text)       # OCR noise
    text = re.sub(r"(.)\1{4,}", " ", text)             # repeated chars
    text = re.sub(r"[^\w\s.,;:()\-\'\"/°%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── BM25 keyword search (fallback) ───────────────────────────
class BM25Index:
    """Simple BM25 implementation without external deps."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def fit(self, texts: List[str]):
        self.corpus = [self._tokenize(t) for t in texts]
        self.N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.N, 1)
        from collections import Counter
        df = Counter()
        for doc in self.corpus:
            for w in set(doc): df[w] += 1
        import math
        self.idf = {w: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
                    for w, f in df.items()}

    def score(self, query: str, top_k: int = 5) -> List[Tuple[float, int]]:
        import math
        q_terms = self._tokenize(query)
        scores = []
        for i, doc in enumerate(self.corpus):
            from collections import Counter
            tf = Counter(doc)
            dl = len(doc)
            s = 0.0
            for w in q_terms:
                if w not in self.idf: continue
                f = tf.get(w, 0)
                s += self.idf[w] * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
            scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:top_k]


# ── Main VectorDB class ───────────────────────────────────────
class VectorDB:
    """
    Dense vector database backed by FAISS.
    Falls back to BM25 if sentence-transformers not installed.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu    = use_gpu
        self.texts: List[str]           = []
        self.metadatas: List[Dict]      = []
        self.embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._bm25 = BM25Index()
        self._encoder = None
        self._dim: int = 0
        self._try_load_encoder()

    def _try_load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[VectorDB] Loading encoder: {self.model_name}")
            self._encoder = SentenceTransformer(self.model_name)
            self._dim = self._encoder.get_sentence_embedding_dimension()
            print(f"[VectorDB] Encoder ready — dim={self._dim}")
        except ImportError:
            print("[VectorDB] sentence-transformers not installed — BM25 only mode")
            print("  Install: pip install sentence-transformers")
        except Exception as e:
            print(f"[VectorDB] Encoder load failed: {e} — BM25 only mode")

    def _encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("Encoder not available")
        vecs = self._encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine via dot product
        )
        return np.array(vecs, dtype=np.float32)

    def _build_faiss(self, embeddings: np.ndarray):
        try:
            import faiss
            d = embeddings.shape[1]
            # IVF index for large collections, Flat for small
            if len(embeddings) > 50_000:
                nlist = min(256, len(embeddings) // 100)
                quantizer = faiss.IndexFlatIP(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings)
                if self.use_gpu:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
            else:
                index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            self._faiss_index = index
            print(f"[VectorDB] FAISS index built: {index.ntotal} vectors, dim={d}")
        except ImportError:
            print("[VectorDB] faiss-cpu not installed — dense search disabled")
            print("  Install: pip install faiss-cpu")

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        batch_size: int = 64,
    ):
        """Add documents to the index."""
        if not texts:
            return
        if metadatas is None:
            metadatas = [{}] * len(texts)

        print(f"[VectorDB] Adding {len(texts):,} documents...")
        start_idx = len(self.texts)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

        # Dense embeddings
        if self._encoder is not None:
            new_vecs = self._encode(texts, batch_size=batch_size)
            if self.embeddings is None:
                self.embeddings = new_vecs
            else:
                self.embeddings = np.vstack([self.embeddings, new_vecs])
            self._build_faiss(self.embeddings)

        # BM25 index (always)
        self._bm25.fit(self.texts)
        print(f"[VectorDB] Total documents: {len(self.texts):,}")

    def add_from_source(
        self,
        text: str,
        source_name: str,
        source_type: str = "book",
        chunk_size: int = 256,
        overlap: int = 32,
    ):
        """
        Chunk raw text and add to DB with source metadata.
        Use this for adding books, PDFs, manuals.
        """
        cleaned = clean_text(text)
        chunks  = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        metas   = [{"source": source_name, "type": source_type, "chunk_id": i}
                   for i in range(len(chunks))]
        print(f"[VectorDB] '{source_name}': {len(chunks)} chunks")
        self.add_documents(chunks, metas)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        use_hybrid: bool = True,
    ) -> List[Dict]:
        """
        Search for relevant passages.
        Returns list of dicts: {text, score, source, type, chunk_id}
        """
        if not self.texts:
            return []

        results = {}

        # Dense search
        if self._faiss_index is not None and self._encoder is not None:
            q_vec = self._encode([query])
            k = min(top_k * 2, len(self.texts))
            scores, indices = self._faiss_index.search(q_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0: continue
                if score < min_score: continue
                results[idx] = {
                    "text":     self.texts[idx],
                    "score":    float(score),
                    "dense":    True,
                    **self.metadatas[idx],
                }

        # BM25 hybrid search
        if use_hybrid:
            bm25_results = self._bm25.score(query, top_k=top_k * 2)
            for bm25_score, idx in bm25_results:
                if bm25_score <= 0: continue
                norm_score = min(bm25_score / 20.0, 1.0)  # normalize
                if norm_score < min_score:
                    continue
                if idx in results:
                    results[idx]["score"] = (results[idx]["score"] + norm_score) / 2
                    results[idx]["hybrid"] = True
                else:
                    results[idx] = {
                        "text":     self.texts[idx],
                        "score":    norm_score,
                        "dense":    False,
                        "hybrid":   True,
                        **self.metadatas[idx],
                    }

        # Sort by score, filter weak results, return top_k
        sorted_results = sorted(results.values(), key=lambda x: x["score"], reverse=True)
        if min_score > 0:
            sorted_results = [r for r in sorted_results if r["score"] >= min_score]
        return sorted_results[:top_k]

    def save(self, directory: str):
        """Persist the index to disk."""
        os.makedirs(directory, exist_ok=True)

        # Save texts + metadata
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadatas": self.metadatas}, f)

        # Save embeddings
        if self.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)

        # Save FAISS index
        if self._faiss_index is not None:
            try:
                import faiss
                faiss.write_index(
                    self._faiss_index,
                    os.path.join(directory, "faiss.index")
                )
            except Exception as e:
                print(f"[VectorDB] FAISS save error: {e}")

        # Save BM25
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump(self._bm25, f)

        # Save config
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump({
                "model_name": self.model_name,
                "n_docs":     len(self.texts),
                "dim":        self._dim,
            }, f, indent=2)

        print(f"[VectorDB] Saved {len(self.texts):,} docs → {directory}/")

    @classmethod
    def load(cls, directory: str, use_gpu: bool = False) -> "VectorDB":
        """Load a persisted index."""
        config_path = os.path.join(directory, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No VectorDB found at {directory}")

        with open(config_path) as f:
            config = json.load(f)

        db = cls(model_name=config["model_name"], use_gpu=use_gpu)

        # Load texts + metadata
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            data = pickle.load(f)
        db.texts     = data["texts"]
        db.metadatas = data["metadatas"]

        # Load embeddings
        emb_path = os.path.join(directory, "embeddings.npy")
        if os.path.exists(emb_path):
            db.embeddings = np.load(emb_path)
            db._build_faiss(db.embeddings)

        # Load BM25
        bm25_path = os.path.join(directory, "bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                db._bm25 = pickle.load(f)

        print(f"[VectorDB] Loaded {len(db.texts):,} docs ← {directory}/")
        return db

    def __len__(self): return len(self.texts)

    def stats(self) -> Dict:
        sources = {}
        for m in self.metadatas:
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "total_docs":   len(self.texts),
            "has_dense":    self._faiss_index is not None,
            "has_bm25":     len(self._bm25.corpus) > 0,
            "dim":          self._dim,
            "sources":      sources,
        }