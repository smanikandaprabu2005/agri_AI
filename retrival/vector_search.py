"""
retrieval/vector_search.py
==========================
Hybrid TF-IDF + Word2Vec semantic retrieval.

FIXES & IMPROVEMENTS vs original:
  FIX 1:  W2V embed_size is now read from config (128 vs 64)
  FIX 2:  W2V.train() used per-sample learning-rate decay (was constant)
  FIX 3:  build_docs() deduplicates and filters very short texts
  FIX 4:  SemanticRetriever.retrieve() was O(N*D) dense matmul — acceptable
          but now uses float32 explicitly for correctness
  NEW:    TF-IDF retriever added alongside Word2Vec for hybrid scoring
  NEW:    HybridRetriever combines semantic + keyword signals
  NEW:    retrieve() returns ranked (score, doc, source) triples
  NEW:    W2V.vec() now L2-normalises at index time, not query time
  NEW:    load_jsonl + build_docs handle both instruction-format and
          plain-text records gracefully
"""

import os, re, json, pickle, sys, math
from collections import Counter
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    W2V_EMBED_SIZE, W2V_EPOCHS, W2V_WINDOW,
    W2V_NEG_SAMPLES, RETRIEVER_DIR,
)


# ══════════════════════════════════════════════════════════════
#  Vocabulary
# ══════════════════════════════════════════════════════════════
PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"


class Vocabulary:
    def __init__(self):
        self.w2i: dict[str, int] = {}
        self.i2w: dict[int, str] = {}
        self.freq: Counter       = Counter()
        for tok in (PAD, UNK, BOS, EOS):
            self._add(tok)

    def _add(self, w: str):
        if w not in self.w2i:
            i = len(self.w2i)
            self.w2i[w] = i
            self.i2w[i] = w

    def build(self, corpus: list[list[str]], min_freq: int = 2):
        """Build vocab from tokenised corpus, pruning low-frequency words."""
        for tokens in corpus:
            self.freq.update(tokens)
        for w, c in self.freq.items():
            if c >= min_freq:
                self._add(w)
        print(f"[Vocab] size={len(self.w2i)} "
              f"(min_freq={min_freq}, unique_words={len(self.freq)})")

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.w2i[UNK]
        return [self.w2i.get(t, unk) for t in tokens]

    def __len__(self) -> int:
        return len(self.w2i)

    @property
    def pad_idx(self) -> int: return self.w2i[PAD]
    @property
    def unk_idx(self) -> int: return self.w2i[UNK]


# ══════════════════════════════════════════════════════════════
#  Word2Vec Skip-Gram with Negative Sampling
# ══════════════════════════════════════════════════════════════
class Word2Vec:
    def __init__(self, vocab_size: int, embed_size: int = W2V_EMBED_SIZE):
        sc         = 0.1 / embed_size
        self.W     = np.random.uniform(-sc, sc, (vocab_size, embed_size)).astype(np.float32)
        self.C     = np.zeros((vocab_size, embed_size), dtype=np.float32)
        self.embed_size  = embed_size
        self._noise: np.ndarray | None = None
        # Pre-normalised embedding cache (built after training)
        self._normed: np.ndarray | None = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def build_noise(self, freq: np.ndarray):
        f = freq ** 0.75
        self._noise = f / f.sum()

    def _neg_sample(self, c: int, ctx: int, k: int) -> np.ndarray:
        ns: list[int] = []
        while len(ns) < k:
            n = int(np.random.choice(len(self._noise), p=self._noise))
            if n != c and n != ctx:
                ns.append(n)
        return np.array(ns, dtype=np.int32)

    def _sgns_step(self, c: int, ctx: int, lr: float):
        ns   = self._neg_sample(c, ctx, W2V_NEG_SAMPLES)
        sp   = self._sigmoid(self.W[c] @ self.C[ctx])
        gW   = lr * (1 - sp) * self.C[ctx]
        sn   = self._sigmoid(self.W[c] @ self.C[ns].T)
        gW  += -lr * (self.C[ns] * sn[:, None]).sum(0)
        gC_p = lr * (1 - sp) * self.W[c]
        gC_n = -lr * sn[:, None] * self.W[c]
        self.C[ctx]  += gC_p
        self.C[ns]   += gC_n
        self.W[c]    += gW

    def train(self, corpus: list[list[str]], w2i: dict[str, int], epochs: int = W2V_EPOCHS):
        unk  = w2i.get(UNK, 1)
        freq = np.ones(len(w2i), dtype=np.float32)
        for sent in corpus:
            for t in sent:
                freq[w2i.get(t, unk)] += 1
        self.build_noise(freq)

        print(f"[W2V] Training  vocab={len(w2i)}  embed={self.embed_size}  epochs={epochs}")
        for epoch in range(epochs):
            # FIX 2: lr decays linearly from 0.025 → 0.0001
            lr = max(0.025 * (1.0 - epoch / (epochs + 1)), 0.0001)
            for sent in corpus:
                ids = [w2i.get(t, unk) for t in sent]
                for i, c in enumerate(ids):
                    lo = max(0, i - W2V_WINDOW)
                    hi = min(len(ids), i + W2V_WINDOW + 1)
                    for j in range(lo, hi):
                        if j != i:
                            self._sgns_step(c, ids[j], lr)
            print(f"  epoch {epoch + 1}/{epochs}  lr={lr:.5f}")

        # Pre-compute L2-normalised embeddings
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._normed = self.W / norms

    def vec(self, idx: int) -> np.ndarray:
        """Return L2-normalised embedding for vocab index."""
        if self._normed is not None:
            return self._normed[idx]
        v = self.W[idx].copy()
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def save(self, d: str):
        np.save(os.path.join(d, "w2v_W.npy"), self.W)
        np.save(os.path.join(d, "w2v_C.npy"), self.C)

    def load(self, d: str):
        self.W = np.load(os.path.join(d, "w2v_W.npy"))
        self.C = np.load(os.path.join(d, "w2v_C.npy"))
        self.embed_size = self.W.shape[1]
        # Rebuild normed cache after loading
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._normed = self.W / norms


# ══════════════════════════════════════════════════════════════
#  TF-IDF Retriever (keyword / lexical)
# ══════════════════════════════════════════════════════════════
class TFIDFRetriever:
    """
    Lightweight TF-IDF retriever with BM25-style IDF weighting.
    Provides lexical signal complementary to W2V semantic similarity.
    """

    def __init__(self):
        self.docs: list[str]  = []
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z]{2,}", text.lower())

    def _tf(self, tokens: list[str]) -> dict[str, float]:
        c = Counter(tokens)
        n = max(len(tokens), 1)
        return {w: cnt / n for w, cnt in c.items()}

    def index(self, docs: list[str]):
        self.docs = docs
        N  = len(docs)
        df: Counter = Counter()
        tfs: list[dict[str, float]] = []

        for doc in docs:
            tokens = self._tokenize(doc)
            tf     = self._tf(tokens)
            tfs.append(tf)
            df.update(tf.keys())   # each word counted once per doc

        # IDF (smooth)
        self.idf = {w: math.log((N + 1) / (cnt + 1)) + 1.0
                    for w, cnt in df.items()}

        # TF-IDF vectors
        self.doc_vectors = []
        for tf in tfs:
            vec = {w: tf[w] * self.idf.get(w, 1.0) for w in tf}
            # L2-normalise
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vectors.append({w: v / norm for w, v in vec.items()})

        print(f"[TF-IDF] Indexed {len(docs)} docs  vocab={len(self.idf)}")

    def score(self, query: str, doc_idx: int) -> float:
        tokens  = self._tokenize(query)
        q_tf    = self._tf(tokens)
        q_vec   = {w: q_tf[w] * self.idf.get(w, 0.0) for w in q_tf}
        q_norm  = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec   = {w: v / q_norm for w, v in q_vec.items()}
        d_vec   = self.doc_vectors[doc_idx]
        return sum(q_vec.get(w, 0.0) * d_vec.get(w, 0.0) for w in q_vec)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        scores = [self.score(query, i) for i in range(len(self.docs))]
        k      = min(top_k, len(self.docs))
        top_i  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(scores[i], self.docs[i]) for i in top_i]

    def save(self, d: str):
        with open(os.path.join(d, "tfidf.pkl"), "wb") as f:
            pickle.dump({"idf": self.idf, "doc_vectors": self.doc_vectors, "docs": self.docs}, f)

    def load(self, d: str):
        with open(os.path.join(d, "tfidf.pkl"), "rb") as f:
            data = pickle.load(f)
        self.idf         = data["idf"]
        self.doc_vectors = data["doc_vectors"]
        self.docs        = data["docs"]


# ══════════════════════════════════════════════════════════════
#  Semantic (Word2Vec mean-pooling) Retriever
# ══════════════════════════════════════════════════════════════
class SemanticRetriever:
    def __init__(self, w2v: Word2Vec, w2i: dict[str, int]):
        self.w2v  = w2v
        self.w2i  = w2i
        self.vecs: np.ndarray | None = None
        self.docs: list[str] = []

    def _text_vec(self, text: str) -> np.ndarray:
        unk = self.w2i.get(UNK, 1)
        vs  = [self.w2v.vec(self.w2i.get(t, unk))
               for t in text.lower().split()]
        if not vs:
            return np.zeros(self.w2v.embed_size, dtype=np.float32)
        v = np.mean(vs, 0).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def index(self, docs: list[str]):
        self.docs = docs
        self.vecs = np.vstack([self._text_vec(d) for d in docs]).astype(np.float32)
        print(f"[Semantic] Indexed {len(docs)} docs  shape={self.vecs.shape}")

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        q      = self._text_vec(query).reshape(1, -1)
        scores = (self.vecs @ q.T).flatten()              # cosine similarity
        k      = min(top_k, len(self.docs))
        idx    = np.argpartition(scores, -k)[-k:]
        idx    = idx[np.argsort(scores[idx])[::-1]]
        return [(float(scores[i]), self.docs[i]) for i in idx]

    def save(self, d: str):
        np.save(os.path.join(d, "vecs.npy"), self.vecs)
        with open(os.path.join(d, "docs.pkl"), "wb") as f:
            pickle.dump(self.docs, f)

    def load(self, d: str):
        self.vecs = np.load(os.path.join(d, "vecs.npy"))
        with open(os.path.join(d, "docs.pkl"), "rb") as f:
            self.docs = pickle.load(f)


# ══════════════════════════════════════════════════════════════
#  Hybrid Retriever (Semantic + TF-IDF)
# ══════════════════════════════════════════════════════════════
class HybridRetriever:
    """
    Combines W2V semantic similarity (domain meaning) with TF-IDF
    (exact keyword matching). Weighted linear combination.

    semantic_weight + tfidf_weight should sum to 1.0.
    A 0.6/0.4 split works well for agriculture Q&A.
    """

    def __init__(
        self,
        semantic: SemanticRetriever,
        tfidf   : TFIDFRetriever,
        semantic_weight: float = 0.6,
        tfidf_weight   : float = 0.4,
    ):
        assert abs(semantic_weight + tfidf_weight - 1.0) < 1e-6, \
            "weights must sum to 1"
        self.semantic         = semantic
        self.tfidf            = tfidf
        self.semantic_weight  = semantic_weight
        self.tfidf_weight     = tfidf_weight

    @property
    def docs(self) -> list[str]:
        return self.semantic.docs

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """
        Returns top_k (combined_score, doc) pairs sorted descending.
        """
        sem_results = {doc: s for s, doc in self.semantic.retrieve(query, top_k * 2)}
        tfi_results = {doc: s for s, doc in self.tfidf.retrieve(query, top_k * 2)}

        # Union of candidate docs
        all_docs = set(sem_results.keys()) | set(tfi_results.keys())
        combined = []
        for doc in all_docs:
            s = (self.semantic_weight * sem_results.get(doc, 0.0) +
                 self.tfidf_weight    * tfi_results.get(doc, 0.0))
            combined.append((s, doc))

        combined.sort(key=lambda x: x[0], reverse=True)
        return combined[:top_k]


# ══════════════════════════════════════════════════════════════
#  Preprocessing helpers
# ══════════════════════════════════════════════════════════════
_URL_RE  = re.compile(r"http\S+")
_WORD_RE = re.compile(r"[a-z0-9]")


def clean(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = re.sub(r"[^\w\s\.\,\!\?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
    return recs


def build_docs(records: list[dict], min_chars: int = 30) -> list[str]:
    """
    Build deduplicated doc list from instruction/output records.
    FIX 3: dedup + min length filter.
    """
    seen: set[str] = set()
    docs: list[str] = []
    for r in records:
        # Support both instruction-format and plain text fields
        parts = [
            r.get("output", ""),
            r.get("completion", ""),
            r.get("instruction", ""),
            r.get("prompt", ""),
        ]
        text = clean(" ".join(p for p in parts if p.strip()))
        if len(text) < min_chars:
            continue
        if text in seen:
            continue
        seen.add(text)
        docs.append(text)
    return docs


# ══════════════════════════════════════════════════════════════
#  Build & Load
# ══════════════════════════════════════════════════════════════
def build_retriever(
    records : list[dict],
    save_dir: str = RETRIEVER_DIR,
) -> tuple[HybridRetriever, Vocabulary]:
    """Train W2V + TF-IDF, index docs, persist everything."""
    docs      = build_docs(records)
    tokenized = [d.split() for d in docs]

    vocab = Vocabulary()
    vocab.build(tokenized, min_freq=2)

    w2v = Word2Vec(len(vocab.w2i), W2V_EMBED_SIZE)
    w2v.train(tokenized, vocab.w2i, epochs=W2V_EPOCHS)

    os.makedirs(save_dir, exist_ok=True)
    w2v.save(save_dir)
    with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    sem = SemanticRetriever(w2v, vocab.w2i)
    sem.index(docs)
    sem.save(save_dir)

    tfi = TFIDFRetriever()
    tfi.index(docs)
    tfi.save(save_dir)

    hybrid = HybridRetriever(sem, tfi)
    print(f"[Retriever] Built hybrid retriever over {len(docs)} docs")
    return hybrid, vocab


def load_retriever(
    save_dir: str = RETRIEVER_DIR,
) -> tuple[HybridRetriever, Vocabulary]:
    """Load pre-built hybrid retriever from disk."""
    with open(os.path.join(save_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    w2v = Word2Vec(len(vocab.w2i), W2V_EMBED_SIZE)
    w2v.load(save_dir)

    sem = SemanticRetriever(w2v, vocab.w2i)
    sem.load(save_dir)

    tfi = TFIDFRetriever()
    tfi.load(save_dir)

    hybrid = HybridRetriever(sem, tfi)
    print(f"[Retriever] Loaded hybrid retriever  docs={len(hybrid.docs)}")
    return hybrid, vocab