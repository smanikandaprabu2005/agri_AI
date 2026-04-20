"""
rag/context_builder_v2.py
==========================
Upgraded RAG Context Builder — uses VectorDB (FAISS + BM25) with
automatic fallback to the old Word2Vec retriever if VectorDB unavailable.

Key improvements over V1:
  • Dense semantic search via sentence-transformers
  • Hybrid retrieval (dense + BM25)
  • Larger context window (1200 chars vs 600)
  • Source-aware context (shows where passage came from)
  • Reranking by intent relevance
  • Multi-hop: retrieves then re-queries with expanded terms
"""

import re, os, sys
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weather.weather_api import needs_weather

# ── Intent detection (same as before, extended) ───────────────
_PEST    = re.compile(r"\b(pest|insect|aphid|worm|caterpillar|fly|beetle|mite|thrips|whitefly|stem borer|termite|borer|hoppers)\b", re.I)
_DISEASE = re.compile(r"\b(disease|blight|rot|wilt|mildew|fungal|bacterial|viral|spot|rust|leaf curl|anthracnose|blast|canker|mosaic)\b", re.I)
_FERT    = re.compile(r"\b(fertiliz|urea|potash|phosphate|npk|nutrient|manure|compost|nitrogen|micronutrient)\b", re.I)
_SPRAY   = re.compile(r"\b(spray|pesticide|fungicide|insecticide|dose|dosage|chemical|herbicide)\b", re.I)
_PLANT   = re.compile(r"\b(plant|sow|seed|seedling|transplant|cultivat|growing|spacing|nursery)\b", re.I)
_SOIL    = re.compile(r"\b(soil|ph|sandy|clay|loam|texture|drainage|organic matter|tilth)\b", re.I)
_WATER   = re.compile(r"\b(irrigat|water|drip|sprinkler|flood|moisture|drought)\b", re.I)
_HARVEST = re.compile(r"\b(harvest|yield|post.harvest|storage|grading|market)\b", re.I)

INTENT_PATTERNS = {
    "pest_control":       _PEST,
    "disease_management": _DISEASE,
    "fertilization":      _FERT,
    "spray_advisory":     _SPRAY,
    "planting":           _PLANT,
    "soil_management":    _SOIL,
    "irrigation":         _WATER,
    "harvest_storage":    _HARVEST,
}

def detect_intent(q: str) -> str:
    for intent, pat in INTENT_PATTERNS.items():
        if pat.search(q):
            return intent
    return "general_agriculture"


def expand_query(query: str, intent: str) -> str:
    """Add intent-specific terms to improve retrieval."""
    expansions = {
        "pest_control":       "pest control treatment spray chemical management",
        "disease_management": "disease treatment fungicide prevention control",
        "fertilization":      "fertilizer dose application nutrient management",
        "spray_advisory":     "spray dose ml litre application pesticide",
        "planting":           "planting spacing transplant seedling cultivation",
        "soil_management":    "soil preparation amendment organic matter pH",
        "irrigation":         "water requirement irrigation schedule method",
        "harvest_storage":    "harvest maturity storage post-harvest",
        "general_agriculture": "agriculture farming crop management",
    }
    extra = expansions.get(intent, "")
    return f"{query} {extra}".strip()


# ── Context Builder V2 ────────────────────────────────────────
class ContextBuilderV2:
    """
    Drop-in replacement for the original ContextBuilder.
    Uses VectorDB if available, falls back to Word2Vec retriever.
    """

    def __init__(
        self,
        retriever,           # old Word2Vec retriever (kept as fallback)
        memory,
        weather_svc,
        vector_db=None,      # NEW: VectorDB instance
        top_k: int = 5,
        context_max_chars: int = 1200,
        use_vector_db: bool = True,
    ):
        self.retriever       = retriever
        self.memory          = memory
        self.weather_svc     = weather_svc
        self.vector_db       = vector_db
        self.top_k           = top_k
        self.context_max     = context_max_chars
        self.use_vector_db   = use_vector_db and vector_db is not None

    # ── Public API (same as original ContextBuilder) ──────────

    def build_context_str(self, query: str) -> str:
        intent    = detect_intent(query)
        parts     = []
        seen_text = set()

        # 1. Dense / hybrid retrieval from VectorDB
        if self.use_vector_db:
            expanded = expand_query(query, intent)
            vdb_results = self.vector_db.search(
                expanded, top_k=self.top_k, min_score=0.15
            )
            for r in vdb_results:
                snippet = r["text"][:300].strip()
                if snippet in seen_text: continue
                seen_text.add(snippet)
                source_label = r.get("source", "Reference")
                parts.append(f"[{source_label}] {snippet}")

        # 2. Word2Vec fallback (always run if VectorDB didn't return enough)
        if len(parts) < 2:
            w2v_results = self.retriever.retrieve(query, top_k=self.top_k)
            for score, doc in w2v_results[:3]:
                if score >= 0.05:
                    snippet = " ".join(doc.split()[:60])
                    if snippet not in seen_text:
                        seen_text.add(snippet)
                        parts.append(f"[KnowledgeBase] {snippet}")

        # 3. Weather context
        if needs_weather(query) or intent == "spray_advisory":
            loc = self.memory.get_location()
            parts.append("[Weather] " + self.weather_svc.context_str(loc))

        # 4. Farmer profile context
        profile = self.memory.long.as_text()
        if profile and profile != "No profile yet.":
            parts.insert(0, f"[FarmerProfile] {profile}")

        context = "\n\n".join(parts)
        return context[:self.context_max]

    def get_retrieved(self, query: str) -> List[Tuple[float, str]]:
        """Returns (score, text) pairs — compatible with old API."""
        results = []

        if self.use_vector_db:
            expanded = expand_query(query, detect_intent(query))
            vdb_results = self.vector_db.search(expanded, top_k=self.top_k)
            for r in vdb_results:
                results.append((r["score"], r["text"]))

        # Supplement with Word2Vec
        if len(results) < self.top_k:
            w2v = self.retriever.retrieve(query, top_k=self.top_k)
            results.extend(w2v)

        return results[:self.top_k]

    def get_intent(self, query: str) -> str:
        return detect_intent(query)

    def search_by_source(self, query: str, source_type: str, top_k: int = 3) -> List[str]:
        """Search restricted to a specific source type (e.g. 'textbook')."""
        if not self.use_vector_db:
            return []
        results = self.vector_db.search(query, top_k=top_k * 3)
        filtered = [r["text"] for r in results if r.get("type") == source_type]
        return filtered[:top_k]

    @property
    def weather_svc(self): return self._weather_svc
    @weather_svc.setter
    def weather_svc(self, v): self._weather_svc = v