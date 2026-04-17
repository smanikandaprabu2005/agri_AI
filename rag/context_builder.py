"""
rag/context_builder.py  —  SageStorm V2.1
==========================================
Key upgrades over V2:
  - Hybrid retrieval: dense (Word2Vec cosine) + sparse (BM25 keyword score)
  - Context window doubled: 600 → 1200 chars (config.RAG_CONTEXT_MAX)
  - Snippet deduplication: skip snippets overlapping with already-used text
  - Richer weather integration: includes advisory text, not just raw data
  - Intent-aware snippet selection: match snippets closer to detected intent
  - Source quality score: prefer longer, more informative snippets
"""

import re
import os
import sys
import math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weather.weather_api import needs_weather

# ── Import RAG_CONTEXT_MAX from config (new in V2.1) ────────
try:
    from config import RAG_CONTEXT_MAX
except ImportError:
    RAG_CONTEXT_MAX = 1200   # fallback if using old config.py


# ── Intent detection ─────────────────────────────────────────
_PEST    = re.compile(r"\b(pest|insect|aphid|worm|caterpillar|fly|beetle|mite|thrips|whitefly|stem borer|termite|bug|grub|larva)\b", re.I)
_DISEASE = re.compile(r"\b(disease|blight|rot|wilt|mildew|fungal|bacterial|viral|spot|rust|leaf curl|anthracnose|blast|canker|scab|mosaic)\b", re.I)
_FERT    = re.compile(r"\b(fertiliz|urea|potash|phosphate|npk|nutrient|manure|compost|dap|mop|fym|micronutrient)\b", re.I)
_SPRAY   = re.compile(r"\b(spray|pesticide|fungicide|insecticide|dose|dosage|ml|litre|concentration)\b", re.I)
_PLANT   = re.compile(r"\b(plant|sow|seed|seedling|transplant|cultivat|growing|spacing|nursery)\b", re.I)
_HARVEST = re.compile(r"\b(harvest|yield|mature|pick|cut|store|storage|post.harvest)\b", re.I)
_SOIL    = re.compile(r"\b(soil|ph|loamy|sandy|clay|drainage|tilling|plough|bed|organic matter)\b", re.I)
_WATER   = re.compile(r"\b(irrigat|water|drip|sprinkler|flood|rain|moisture|drought)\b", re.I)


def detect_intent(q: str) -> str:
    if _PEST.search(q):    return "pest_control"
    if _DISEASE.search(q): return "disease_management"
    if _FERT.search(q):    return "fertilization"
    if _SPRAY.search(q):   return "spray_advisory"
    if _HARVEST.search(q): return "harvesting"
    if _SOIL.search(q):    return "soil_management"
    if _WATER.search(q):   return "irrigation"
    if _PLANT.search(q):   return "planting"
    return "general_agriculture"


# ── BM25 lightweight scorer ───────────────────────────────────
class BM25Scorer:
    """
    Lightweight BM25 (no external deps) for keyword-based relevance.
    Used alongside dense cosine score for hybrid retrieval.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

    def _tokenize(self, text: str) -> list:
        return re.findall(r"[a-zA-Z]+", text.lower())

    def score(self, query: str, document: str) -> float:
        q_terms  = self._tokenize(query)
        d_tokens = self._tokenize(document)
        if not d_tokens or not q_terms:
            return 0.0
        dl     = len(d_tokens)
        avgdl  = 150.0   # typical agricultural sentence length
        tf     = Counter(d_tokens)
        score  = 0.0
        df_est = 0.5     # simplified: assume each term appears in ~50% of docs
        for term in set(q_terms):
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = math.log((1.0 - df_est + 0.5) / (df_est + 0.5) + 1.0)
            tf_norm = (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            score += idf * tf_norm
        return score


_bm25 = BM25Scorer()


# ── Snippet quality score ─────────────────────────────────────
def snippet_quality(text: str, query: str, intent: str) -> float:
    """
    Combines:
      - length (longer → more informative, up to 80 words)
      - overlap with query terms
      - presence of intent-specific keywords
    """
    words = text.split()
    length_score = min(len(words) / 80.0, 1.0)

    q_words    = set(query.lower().split())
    d_words    = set(text.lower().split())
    overlap    = len(q_words & d_words) / max(len(q_words), 1)

    # intent keywords boost
    intent_bonus = 0.0
    intent_kw = {
        "pest_control":       r"\b(spray|dose|ml|litre|control|apply)\b",
        "disease_management": r"\b(fungicide|spray|dose|remove|destroy)\b",
        "fertilization":      r"\b(apply|kg|acre|dose|nitrogen|phosphate|potash)\b",
        "spray_advisory":     r"\b(avoid|rain|weather|morning|evening)\b",
        "planting":           r"\b(spacing|cm|distance|row|depth|transplant)\b",
        "harvesting":         r"\b(days|mature|colour|size|stage|yield)\b",
        "soil_management":    r"\b(ph|loam|clay|organic|compost|drainage)\b",
        "irrigation":         r"\b(days|interval|drip|litre|mm|moisture)\b",
    }
    if intent in intent_kw:
        if re.search(intent_kw[intent], text, re.I):
            intent_bonus = 0.3

    return 0.4 * length_score + 0.4 * overlap + 0.2 * intent_bonus


# ── Main context builder ──────────────────────────────────────
class ContextBuilder:
    def __init__(self, retriever, memory, weather_svc, top_k: int = 5):
        self.retriever   = retriever
        self.memory      = memory
        self.weather_svc = weather_svc
        self.top_k       = top_k
        self._ctx_max    = RAG_CONTEXT_MAX

    # ── Hybrid retrieval: dense + BM25 ───────────────────────
    def _hybrid_retrieve(self, query: str, top_k: int) -> list:
        """
        Returns list of (hybrid_score, doc_text) sorted descending.
        Hybrid score = 0.65 × dense_cosine + 0.35 × normalised_bm25.
        """
        dense_results = self.retriever.retrieve(query, top_k=top_k * 2)

        # BM25 scores
        bm25_scores = [(_bm25.score(query, doc), doc) for _, doc in dense_results]
        max_bm25    = max((s for s, _ in bm25_scores), default=1.0)
        if max_bm25 == 0:
            max_bm25 = 1.0

        hybrid = []
        for (cos, doc), (bm25_raw, _) in zip(dense_results, bm25_scores):
            norm_bm25  = bm25_raw / max_bm25
            hyb        = 0.65 * cos + 0.35 * norm_bm25
            hybrid.append((hyb, doc))

        hybrid.sort(key=lambda x: x[0], reverse=True)
        return hybrid[:top_k]

    # ── Main context string ───────────────────────────────────
    def build_context_str(self, query: str) -> str:
        """
        Build RAG context string up to RAG_CONTEXT_MAX chars.
        Uses hybrid retrieval + intent-aware quality scoring.
        """
        intent   = detect_intent(query)
        results  = self._hybrid_retrieve(query, self.top_k)
        parts    = []
        used_text = set()

        for i, (score, doc) in enumerate(results[:4], 1):
            if score < 0.05:
                continue

            # intent-aware quality filtering
            qual = snippet_quality(doc, query, intent)
            if qual < 0.15:
                continue

            # deduplication: skip if >40% of words already covered
            words = set(doc.lower().split())
            overlap_ratio = len(words & used_text) / max(len(words), 1)
            if overlap_ratio > 0.40:
                continue
            used_text.update(words)

            # take up to 80 words per snippet
            snippet = " ".join(doc.split()[:80])
            parts.append(f"[{i}] {snippet}")

        # Weather integration
        if needs_weather(query) or intent == "spray_advisory":
            loc = self.memory.get_location()
            try:
                w_ctx = self.weather_svc.context_str(loc)
                parts.append("[Weather]\n" + w_ctx)
            except Exception:
                pass

        # Farmer profile context
        profile = self.memory.long.as_text()
        if profile and profile != "No profile yet.":
            parts.append(f"[Farmer] {profile}")

        full_ctx = "\n\n".join(parts)
        # Truncate to configured limit, but don't cut mid-word
        if len(full_ctx) > self._ctx_max:
            full_ctx = full_ctx[: self._ctx_max]
            last_space = full_ctx.rfind(" ")
            if last_space > self._ctx_max - 80:
                full_ctx = full_ctx[:last_space]

        return full_ctx

    # ── Helpers used by response generator ───────────────────
    def get_retrieved(self, query: str) -> list:
        """Return raw dense retrieval results (for response_generator.py)."""
        return self.retriever.retrieve(query, top_k=self.top_k)

    def get_hybrid(self, query: str) -> list:
        """Return hybrid retrieval results."""
        return self._hybrid_retrieve(query, self.top_k)

    def get_intent(self, query: str) -> str:
        return detect_intent(query)
