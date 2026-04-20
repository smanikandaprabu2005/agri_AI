"""
retrieval/vector_search.py
==========================
Enhanced Hybrid Retriever: BM25 + TF-IDF

Why this is better than Word2Vec:
  - BM25 is the gold standard for sparse retrieval (used in Elasticsearch, Lucene)
  - TF-IDF captures term importance without training
  - Hybrid combines lexical precision (BM25) with semantic spread (TF-IDF cosine)
  - No training needed — builds in seconds
  - Works better for agriculture domain with specific chemical/crop names

Architecture:
  BM25Retriever    — Okapi BM25 scoring
  TFIDFRetriever   — sparse TF-IDF with cosine similarity
  HybridRetriever  — weighted fusion of BM25 + TF-IDF scores
  Word2VecRetriever — kept as fallback for compatibility

Usage:
  from retrieval.vector_search import build_retriever, load_retriever
  # Same API as before — drop-in replacement
"""

import os, re, json, pickle, math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RETRIEVER_DIR, RETRIEVER_TYPE, BM25_K1, BM25_B,
    TFIDF_MAX_FEATS, HYBRID_ALPHA, RETRIEVAL_TOP_K,
)


# ── Text preprocessing ────────────────────────────────────────
_STOP = {
    "a","an","the","is","it","in","of","to","and","or","for","on","at","by",
    "from","this","that","with","be","are","was","were","as","do","does",
    "have","has","had","not","no","can","will","should","would","may","might",
    "your","my","our","their","its","what","how","which","when","where","why",
}

def tokenize(text: str, remove_stop: bool = True) -> List[str]:
    text   = text.lower()
    text   = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    if remove_stop:
        tokens = [t for t in tokens if t not in _STOP and len(t) > 1]
    return tokens


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s\.,!?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════
#  BM25 Retriever (Okapi BM25)
# ══════════════════════════════════════════════════════════════
class BM25Retriever:
    """
    Okapi BM25 — state-of-the-art sparse retrieval.
    Parameters k1=1.5, b=0.75 are well-tuned defaults for factual QA.
    """
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1   = k1
        self.b    = b
        self.docs : List[str]             = []
        self.tok  : List[List[str]]       = []
        self.df   : Dict[str, int]        = {}
        self.idf  : Dict[str, float]      = {}
        self.avgdl: float                 = 0.0
        self.N    : int                   = 0

    def index(self, docs: List[str]):
        self.docs  = docs
        self.tok   = [tokenize(d) for d in docs]
        self.N     = len(docs)
        self.avgdl = sum(len(t) for t in self.tok) / max(self.N, 1)

        # Document frequency
        self.df = defaultdict(int)
        for tok_list in self.tok:
            for term in set(tok_list):
                self.df[term] += 1

        # IDF with Robertson correction
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self.df.items()
        }
        print(f"[BM25] Indexed {self.N} docs, vocab={len(self.df)}")

    def score(self, query: str, doc_idx: int) -> float:
        q_terms = tokenize(query)
        tokens  = self.tok[doc_idx]
        dl      = len(tokens)
        tf_map  = Counter(tokens)
        s = 0.0
        for term in q_terms:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            s  += self.idf[term] * num / max(den, 1e-9)
        return s

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Tuple[float, str]]:
        scores = np.array([self.score(query, i) for i in range(self.N)])
        # Normalise to [0,1]
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        k   = min(top_k, self.N)
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(float(scores[i]), self.docs[i]) for i in idx]

    def save(self, d: str):
        with open(os.path.join(d, "bm25.pkl"), "wb") as f:
            pickle.dump({
                "docs": self.docs, "tok": self.tok, "df": dict(self.df),
                "idf": self.idf, "avgdl": self.avgdl, "N": self.N,
                "k1": self.k1, "b": self.b,
            }, f)

    def load(self, d: str):
        with open(os.path.join(d, "bm25.pkl"), "rb") as f:
            data = pickle.load(f)
        self.docs   = data["docs"]
        self.tok    = data["tok"]
        self.df     = defaultdict(int, data["df"])
        self.idf    = data["idf"]
        self.avgdl  = data["avgdl"]
        self.N      = data["N"]
        self.k1     = data["k1"]
        self.b      = data["b"]
        print(f"[BM25] Loaded {self.N} docs")


# ══════════════════════════════════════════════════════════════
#  TF-IDF Retriever (cosine similarity)
# ══════════════════════════════════════════════════════════════
class TFIDFRetriever:
    """Sparse TF-IDF with cosine similarity — fast complement to BM25."""

    def __init__(self, max_features: int = TFIDF_MAX_FEATS):
        self.max_features = max_features
        self.vocab        : Dict[str, int] = {}
        self.idf_vec      : Optional[np.ndarray] = None
        self.doc_vecs     : Optional[np.ndarray] = None
        self.docs         : List[str]             = []

    def _build_vocab(self, tokenized: List[List[str]]):
        df = defaultdict(int)
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1
        # Keep top-N by document frequency (but keep rare domain terms)
        sorted_terms = sorted(df.items(), key=lambda x: -x[1])
        self.vocab   = {t: i for i, (t, _) in enumerate(sorted_terms[:self.max_features])}

    def _tfidf_vec(self, tokens: List[str], N: int, df: Dict[str, int]) -> np.ndarray:
        tf    = Counter(tokens)
        total = max(len(tokens), 1)
        v     = np.zeros(len(self.vocab), dtype=np.float32)
        for term, cnt in tf.items():
            if term in self.vocab:
                tf_val  = cnt / total
                idf_val = math.log((N + 1) / (df.get(term, 0) + 1)) + 1.0
                v[self.vocab[term]] = tf_val * idf_val
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-9 else v

    def index(self, docs: List[str]):
        self.docs  = docs
        tokenized  = [tokenize(d) for d in docs]
        self._build_vocab(tokenized)
        N  = len(docs)
        df = defaultdict(int)
        for toks in tokenized:
            for t in set(toks):
                df[t] += 1
        self.doc_vecs = np.vstack([self._tfidf_vec(t, N, df) for t in tokenized])
        print(f"[TFIDF] Indexed {N} docs, vocab={len(self.vocab)}")

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Tuple[float, str]]:
        df = defaultdict(int)  # not needed for query; IDF already in doc_vecs
        N  = len(self.docs)
        q_toks = tokenize(query)
        q_vec  = self._tfidf_vec(q_toks, N, df)
        scores = self.doc_vecs @ q_vec
        k      = min(top_k, N)
        idx    = np.argpartition(scores, -k)[-k:]
        idx    = idx[np.argsort(scores[idx])[::-1]]
        return [(float(scores[i]), self.docs[i]) for i in idx]

    def save(self, d: str):
        np.save(os.path.join(d, "tfidf_vecs.npy"), self.doc_vecs)
        with open(os.path.join(d, "tfidf_meta.pkl"), "wb") as f:
            pickle.dump({"vocab": self.vocab, "docs": self.docs}, f)

    def load(self, d: str):
        self.doc_vecs = np.load(os.path.join(d, "tfidf_vecs.npy"))
        with open(os.path.join(d, "tfidf_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.vocab = meta["vocab"]
        self.docs  = meta["docs"]
        print(f"[TFIDF] Loaded {len(self.docs)} docs")


# ══════════════════════════════════════════════════════════════
#  Hybrid Retriever (BM25 + TF-IDF fusion)
# ══════════════════════════════════════════════════════════════
class HybridRetriever:
    """
    Reciprocal Rank Fusion (RRF) + linear interpolation.
    alpha=0.6 weights BM25 slightly higher (better for exact agri terms).
    """
    def __init__(self, alpha: float = HYBRID_ALPHA, rrf_k: int = 60):
        self.alpha  = alpha
        self.rrf_k  = rrf_k
        self.bm25   = BM25Retriever()
        self.tfidf  = TFIDFRetriever()
        self.docs   : List[str] = []

    def index(self, docs: List[str]):
        self.docs = docs
        self.bm25.index(docs)
        self.tfidf.index(docs)

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Tuple[float, str]]:
        k2   = min(top_k * 3, len(self.docs))  # retrieve more, then fuse
        bm25_res  = self.bm25.retrieve(query, k2)
        tfidf_res = self.tfidf.retrieve(query, k2)

        # Build doc -> score maps
        bm25_scores  = {doc: score for score, doc in bm25_res}
        tfidf_scores = {doc: score for score, doc in tfidf_res}

        # RRF for rank fusion
        bm25_rank    = {doc: i+1 for i, (_, doc) in enumerate(bm25_res)}
        tfidf_rank   = {doc: i+1 for i, (_, doc) in enumerate(tfidf_res)}

        all_docs = set(bm25_scores) | set(tfidf_scores)
        fused    = {}
        for doc in all_docs:
            rrf_bm25  = 1.0 / (self.rrf_k + bm25_rank.get(doc,  k2 + 1))
            rrf_tfidf = 1.0 / (self.rrf_k + tfidf_rank.get(doc, k2 + 1))
            fused[doc] = self.alpha * rrf_bm25 + (1 - self.alpha) * rrf_tfidf

        top = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
        # Normalise
        max_score = max(s for _, s in top) if top else 1.0
        return [(score / max(max_score, 1e-9), doc) for doc, score in top]

    def save(self, d: str):
        self.bm25.save(d)
        self.tfidf.save(d)
        with open(os.path.join(d, "hybrid_meta.pkl"), "wb") as f:
            pickle.dump({"alpha": self.alpha, "rrf_k": self.rrf_k}, f)

    def load(self, d: str):
        self.bm25.load(d)
        self.tfidf.load(d)
        meta_path = os.path.join(d, "hybrid_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.alpha = meta.get("alpha", HYBRID_ALPHA)
            self.rrf_k = meta.get("rrf_k", 60)
        self.docs = self.bm25.docs
        print(f"[Hybrid] Loaded {len(self.docs)} docs (BM25 + TF-IDF)")


# ══════════════════════════════════════════════════════════════
#  Backwards-compatible Word2Vec (kept for legacy)
# ══════════════════════════════════════════════════════════════
class Vocabulary:
    def __init__(self):
        self.w2i, self.i2w, self.freq = {}, {}, Counter()
        for tok in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            self._add(tok)

    def _add(self, w):
        if w not in self.w2i:
            i = len(self.w2i); self.w2i[w] = i; self.i2w[i] = w

    def build(self, corpus):
        for tokens in corpus:
            self.freq.update(tokens)
        for w, c in self.freq.items():
            if c >= 1: self._add(w)
        print(f"[Vocab] size={len(self.w2i)}")

    def encode(self, tokens): return [self.w2i.get(t, 1) for t in tokens]
    def __len__(self): return len(self.w2i)


class SemanticRetriever:
    """Legacy Word2Vec retriever — use HybridRetriever for new deployments."""
    def __init__(self, w2v, w2i):
        self.w2v  = w2v
        self.w2i  = w2i
        self.vecs = None
        self.docs = []

    def _text_vec(self, text):
        from config import W2V_EMBED_SIZE
        vs = [self.w2v.vec(self.w2i.get(t, 1)) for t in text.lower().split()]
        if not vs:
            return np.zeros(self.w2v.embed_size)
        v = np.mean(vs, 0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def index(self, docs):
        self.docs = docs
        self.vecs = np.vstack([self._text_vec(d) for d in docs])
        print(f"[Retriever] Indexed {len(docs)} docs (Word2Vec)")

    def retrieve(self, query, top_k=RETRIEVAL_TOP_K):
        q  = self._text_vec(query).reshape(1, -1)
        scores = (self.vecs @ q.T).flatten()
        k   = min(top_k, len(self.docs))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(float(scores[i]), self.docs[i]) for i in idx]

    def save(self, d):
        np.save(os.path.join(d, "vecs.npy"), self.vecs)
        with open(os.path.join(d, "docs.pkl"), "wb") as f:
            pickle.dump(self.docs, f)

    def load(self, d):
        self.vecs = np.load(os.path.join(d, "vecs.npy"))
        with open(os.path.join(d, "docs.pkl"), "rb") as f:
            self.docs = pickle.load(f)


# ── Public helpers ────────────────────────────────────────────
def load_jsonl(path: str) -> List[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except: pass
    return recs


def build_docs(records: List[dict]) -> List[str]:
    docs = []
    for r in records:
        parts = [
            r.get("instruction", ""), r.get("output", ""),
            r.get("input", ""),       r.get("prompt", ""),
            r.get("completion", ""),
        ]
        text = clean(" ".join(p for p in parts if p.strip()))
        if len(text.split()) >= 5:   # skip very short docs
            docs.append(text)
    return docs


def build_retriever(records: List[dict], save_dir: str = RETRIEVER_DIR):
    """Build and save the configured retriever. Returns the retriever."""
    docs = build_docs(records)
    print(f"[Retriever] Building {RETRIEVER_TYPE} retriever on {len(docs)} docs...")

    if RETRIEVER_TYPE == "hybrid":
        ret = HybridRetriever()
    elif RETRIEVER_TYPE == "bm25":
        ret = BM25Retriever()
    elif RETRIEVER_TYPE == "tfidf":
        ret = TFIDFRetriever()
    else:
        raise ValueError(f"Unknown retriever type: {RETRIEVER_TYPE}")

    ret.index(docs)
    os.makedirs(save_dir, exist_ok=True)
    ret.save(save_dir)
    # Save type marker
    with open(os.path.join(save_dir, "type.txt"), "w") as f:
        f.write(RETRIEVER_TYPE)
    print(f"[Retriever] Saved → {save_dir}")
    return ret, None


def load_retriever(save_dir: str = RETRIEVER_DIR):
    """Load the saved retriever (auto-detects type)."""
    type_path = os.path.join(save_dir, "type.txt")
    rtype     = RETRIEVER_TYPE
    if os.path.exists(type_path):
        with open(type_path) as f:
            rtype = f.read().strip()

    if rtype == "hybrid":
        ret = HybridRetriever()
    elif rtype == "bm25":
        ret = BM25Retriever()
    elif rtype == "tfidf":
        ret = TFIDFRetriever()
    else:
        # Legacy Word2Vec fallback
        vocab_path = os.path.join(save_dir, "vocab.pkl")
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            from config import W2V_EMBED_SIZE
            class _W2V:
                def __init__(self):
                    self.W = np.load(os.path.join(save_dir, "w2v_W.npy"))
                    self.embed_size = self.W.shape[1]
                def vec(self, idx):
                    v = self.W[idx]; n = np.linalg.norm(v)
                    return v / n if n > 1e-9 else v
            w2v = _W2V()
            ret = SemanticRetriever(w2v, vocab.w2i)
            ret.load(save_dir)
            return ret, vocab
        raise FileNotFoundError(f"No retriever found at {save_dir}")

    ret.load(save_dir)
    return ret, None