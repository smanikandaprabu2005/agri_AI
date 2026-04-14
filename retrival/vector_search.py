"""
retrieval/vector_search.py
==========================
Word2Vec (Skip-Gram + Negative Sampling) trained from scratch.
Cosine similarity document retrieval.
"""

import os, re, json, pickle, sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import W2V_EMBED_SIZE, W2V_EPOCHS, W2V_WINDOW, W2V_NEG_SAMPLES, RETRIEVER_DIR


# ── Vocabulary ───────────────────────────────────────────────
PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"

class Vocabulary:
    def __init__(self):
        self.w2i, self.i2w, self.freq = {}, {}, Counter()
        for tok in [PAD, UNK, BOS, EOS]: self._add(tok)

    def _add(self, w):
        if w not in self.w2i:
            i = len(self.w2i)
            self.w2i[w] = i; self.i2w[i] = w

    def build(self, corpus):
        for tokens in corpus: self.freq.update(tokens)
        for w, c in self.freq.items():
            if c >= 1: self._add(w)
        print(f"[Vocab] size={len(self.w2i)}")

    def encode(self, tokens): return [self.w2i.get(t, 1) for t in tokens]
    def decode(self, ids):    return [self.i2w.get(i, UNK) for i in ids]
    def __len__(self): return len(self.w2i)

    @property
    def pad_idx(self): return self.w2i[PAD]
    @property
    def unk_idx(self): return self.w2i[UNK]


# ── Word2Vec SGNS ─────────────────────────────────────────────
class Word2Vec:
    def __init__(self, vocab_size, embed_size=W2V_EMBED_SIZE):
        sc = 0.1 / embed_size
        self.W = np.random.uniform(-sc, sc, (vocab_size, embed_size))
        self.C = np.zeros((vocab_size, embed_size))
        self.embed_size = embed_size
        self._noise = None

    @staticmethod
    def _sig(x):
        return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

    def build_noise(self, freq):
        f = freq ** 0.75
        self._noise = f / f.sum()

    def _neg(self, c, ctx, k):
        ns = []
        while len(ns) < k:
            n = np.random.choice(len(self._noise), p=self._noise)
            if n != c and n != ctx: ns.append(n)
        return np.array(ns)

    def _step(self, c, ctx, lr=0.025):
        ns     = self._neg(c, ctx, W2V_NEG_SAMPLES)
        sp     = self._sig(self.W[c] @ self.C[ctx])
        gc_p   = lr * (1 - sp) * self.W[c]
        gw     = lr * (1 - sp) * self.C[ctx]
        sn     = self._sig(self.W[c] @ self.C[ns].T)
        gc_n   = -lr * sn[:, None] * self.W[c]
        gw    += -lr * (self.C[ns] * sn[:, None]).sum(0)
        self.C[ctx]  += gc_p
        self.C[ns]   += gc_n
        self.W[c]    += gw

    def train(self, corpus, w2i, epochs=W2V_EPOCHS):
        unk = w2i.get(UNK, 1)
        freq = np.ones(len(w2i))
        for s in corpus:
            for t in s: freq[w2i.get(t, unk)] += 1
        self.build_noise(freq)
        print(f"[W2V] Training epochs={epochs}")
        for epoch in range(epochs):
            lr = max(0.025 * (1 - epoch/(epochs+1)), 0.0001)
            for sent in corpus:
                ids = [w2i.get(t, unk) for t in sent]
                for i, c in enumerate(ids):
                    lo = max(0, i - W2V_WINDOW)
                    hi = min(len(ids), i + W2V_WINDOW + 1)
                    for j in range(lo, hi):
                        if j != i: self._step(c, ids[j], lr)
            print(f"  epoch {epoch+1}/{epochs} done")

    def vec(self, idx):
        v = self.W[idx]
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def save(self, d):
        np.save(os.path.join(d, "w2v_W.npy"), self.W)
        np.save(os.path.join(d, "w2v_C.npy"), self.C)

    def load(self, d):
        self.W = np.load(os.path.join(d, "w2v_W.npy"))
        self.C = np.load(os.path.join(d, "w2v_C.npy"))
        self.embed_size = self.W.shape[1]


# ── Semantic Retriever ────────────────────────────────────────
class SemanticRetriever:
    def __init__(self, w2v: Word2Vec, w2i: dict):
        self.w2v  = w2v
        self.w2i  = w2i
        self.vecs = None
        self.docs = []

    def _text_vec(self, text):
        unk = self.w2i.get(UNK, 1)
        vs  = [self.w2v.vec(self.w2i.get(t, unk)) for t in text.lower().split()]
        if not vs: return np.zeros(self.w2v.embed_size)
        v = np.mean(vs, 0)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def index(self, docs):
        self.docs = docs
        self.vecs = np.vstack([self._text_vec(d) for d in docs])
        print(f"[Retriever] Indexed {len(docs)} docs")

    def retrieve(self, query, top_k=5):
        q = self._text_vec(query).reshape(1, -1)
        scores = (self.vecs @ q.T).flatten()
        k = min(top_k, len(self.docs))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(float(scores[i]), self.docs[i]) for i in idx]

    def save(self, d):
        np.save(os.path.join(d, "vecs.npy"), self.vecs)
        with open(os.path.join(d, "docs.pkl"), "wb") as f: pickle.dump(self.docs, f)

    def load(self, d):
        self.vecs = np.load(os.path.join(d, "vecs.npy"))
        with open(os.path.join(d, "docs.pkl"), "rb") as f: self.docs = pickle.load(f)


# ── Preprocessing helpers ─────────────────────────────────────
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except: pass
    return recs


def build_docs(records):
    docs = []
    for r in records:
        parts = [r.get("prompt",""), r.get("completion",""),
                 r.get("instruction",""), r.get("output","")]
        text  = clean(" ".join(p for p in parts if p.strip()))
        if text: docs.append(text)
    return docs


def build_retriever(records, save_dir=RETRIEVER_DIR):
    docs      = build_docs(records)
    tokenized = [d.split() for d in docs]
    vocab     = Vocabulary()
    vocab.build(tokenized)
    w2v       = Word2Vec(len(vocab.w2i), W2V_EMBED_SIZE)
    w2v.train(tokenized, vocab.w2i)
    os.makedirs(save_dir, exist_ok=True)
    w2v.save(save_dir)
    with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f: pickle.dump(vocab, f)
    ret = SemanticRetriever(w2v, vocab.w2i)
    ret.index(docs)
    ret.save(save_dir)
    return ret, vocab


def load_retriever(save_dir=RETRIEVER_DIR):
    with open(os.path.join(save_dir, "vocab.pkl"), "rb") as f: vocab = pickle.load(f)
    w2v = Word2Vec(len(vocab.w2i), W2V_EMBED_SIZE)
    w2v.load(save_dir)
    ret = SemanticRetriever(w2v, vocab.w2i)
    ret.load(save_dir)
    print(f"[Retriever] Loaded {len(ret.docs)} docs")
    return ret, vocab
