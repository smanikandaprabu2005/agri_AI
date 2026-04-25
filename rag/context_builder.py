"""
rag/context_builder.py
======================
Enhanced RAG context builder v2.1

Improvements over v2:
  + Richer intent detection (12 categories vs 6)
  + Context deduplication — avoids repeating same facts
  + Passage quality scoring (length, keyword density, coherence)
  + Context truncation respects sentence boundaries
  + Weather advisory integrated more naturally
  + Confidence score returned with context
"""

import re, os, sys
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weather.weather_api import needs_weather
from config import CONTEXT_LENGTH, MIN_RETRIEVAL_SCORE

# ── Intent patterns (order matters — first match wins) ────────
_INTENTS = [
    ("pest_control",        re.compile(
        r"\b(pest|insect|aphid|worm|caterpillar|fly|beetle|mite|thrip|"
        r"whitefly|stem.?borer|termite|leafhopper|mealy.?bug|scale|nematode)\b", re.I)),
    ("disease_management",  re.compile(
        r"\b(disease|blight|rot|wilt|mildew|fungal|bacterial|viral|spot|"
        r"rust|leaf.?curl|anthracnose|blast|canker|mosaic|yellowing|die.?back)\b", re.I)),
    ("fertilization",       re.compile(
        r"\b(fertiliz|urea|potash|phosphate|npk|nutrient|manure|compost|"
        r"dap|ssp|zinc|boron|micronutrient|top.?dress)\b", re.I)),
    ("spray_advisory",      re.compile(
        r"\b(spray|pesticide|fungicide|insecticide|dose|dosage|ml|gram|litre|"
        r"dilut|mix|knapsack|wettable)\b", re.I)),
    ("irrigation",          re.compile(
        r"\b(irrigat|water|drip|sprinkler|flood|furrow|moisture|drought|"
        r"rainwater|rain.?fed|dry)\b", re.I)),
    ("crop_selection",      re.compile(
        r"\b(best|suitable|suitable crop|recommend(?:ed)? crop|which crop|what crop|"
        r"crop choice|crop selection|dryland|drought tolerant|low rainfall|rainfed)\b", re.I)),
    ("planting",            re.compile(
        r"\b(plant|sow|seed|seedling|transplant|cultivat|growing|spacing|"
        r"density|nursery|germination|dibbling)\b", re.I)),
    ("harvesting",          re.compile(
        r"\b(harvest|mature|pick|yield|post.?harvest|storage|grade|pack)\b", re.I)),
    ("soil_management",     re.compile(
        r"\b(soil|ph|alkaline|acidic|loam|clay|sandy|alluvial|organic.?matter|"
        r"tillage|plough|compost|vermi)\b", re.I)),
    ("crop_variety",        re.compile(
        r"\b(variet|cultivar|hybrid|variety|species|breed|heirloom|dwarf|"
        r"high.?yield)\b", re.I)),
    ("government_scheme",   re.compile(
        r"\b(scheme|yojna|yojana|subsid|loan|kcc|pm.?kisan|krishi|kvk|"
        r"fasal.?bima|pmfby|nabard|insurance)\b", re.I)),
    ("livestock",           re.compile(
        r"\b(cow|buffalo|goat|pig|poultry|chicken|cattle|dairy|milch|"
        r"vaccination|deworming|veterinar|animal)\b", re.I)),
    ("general_agriculture", re.compile(r".*", re.I)),  # catch-all
]


def detect_intent(query: str) -> str:
    for intent, pattern in _INTENTS:
        if pattern.search(query):
            return intent
    return "general_agriculture"


# ── Passage quality scoring ───────────────────────────────────
def _passage_quality(passage: str, query: str) -> float:
    """
    Score passage quality (0-1) based on:
    - Length (prefer 15–80 words)
    - Query term overlap
    - Absence of noise tokens
    """
    words   = passage.split()
    n       = len(words)
    if n < 5:
        return 0.0

    # Length score (bell curve centred at 40 words)
    length_score = min(n / 40, 1.0) * max(0, 1 - max(0, n - 80) / 80)

    # Keyword overlap
    q_terms  = set(query.lower().split())
    p_terms  = set(w.lower() for w in words)
    overlap  = len(q_terms & p_terms) / max(len(q_terms), 1)

    # Penalise noisy passages (mostly numbers, special chars)
    alpha_words = sum(1 for w in words if re.match(r"^[a-zA-Z]{2,}$", w))
    coherence   = alpha_words / max(n, 1)

    return 0.3 * length_score + 0.4 * overlap + 0.3 * coherence


def _dedup_passages(passages: List[str], threshold: float = 0.7) -> List[str]:
    """Remove near-duplicate passages using Jaccard similarity."""
    if not passages:
        return passages
    unique = [passages[0]]
    for p in passages[1:]:
        p_toks = set(p.lower().split())
        is_dup = False
        for u in unique:
            u_toks = set(u.lower().split())
            inter  = len(p_toks & u_toks)
            union  = len(p_toks | u_toks)
            if union > 0 and inter / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
    return unique


def _truncate_to_sentences(text: str, max_chars: int) -> str:
    """Truncate text at a sentence boundary rather than mid-word."""
    if len(text) <= max_chars:
        return text
    # Find last sentence end before max_chars
    sub = text[:max_chars]
    for end_char in [".", "!", "?"]:
        idx = sub.rfind(end_char)
        if idx > max_chars // 2:
            return sub[: idx + 1].strip()
    # Fallback: truncate at last space
    idx = sub.rfind(" ")
    return (sub[:idx].strip() + "…") if idx > 0 else sub.strip()


# ══════════════════════════════════════════════════════════════
#  Context Builder
# ══════════════════════════════════════════════════════════════
class ContextBuilder:
    """
    Assembles the RAG prompt context from:
      1. Retrieved passages (BM25/hybrid, quality-filtered)
      2. Weather advisory (when relevant)
      3. Farmer memory profile (location, crop, soil)

    Context budget: 700 characters (fits well in 768-token context).
    """
    MAX_CONTEXT_CHARS = 700

    def __init__(self, retriever, memory, weather_svc, top_k: int = 7):
        self.retriever   = retriever
        self.memory      = memory
        self.weather_svc = weather_svc
        self.top_k       = top_k

    def build_context_str(self, query: str) -> Tuple[str, float]:
        """
        Returns (context_string, confidence_score).
        confidence_score in [0,1] — how good the retrieved context is.
        """
        intent    = detect_intent(query)
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)

        # Filter by minimum score
        retrieved = [(s, d) for s, d in retrieved if s >= MIN_RETRIEVAL_SCORE]

        # Score and rank passages by quality
        scored_passages = []
        for score, doc in retrieved[:5]:
            # Extract best sentence from doc
            sents = re.split(r"[.!\n]+", doc)
            best_sent  = ""
            best_score = 0.0
            for sent in sents:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                q_score = _passage_quality(sent, query)
                if q_score > best_score:
                    best_score = q_score
                    best_sent  = sent
            if best_sent:
                scored_passages.append((score * best_score, best_sent))

        # Sort by combined score
        scored_passages.sort(key=lambda x: -x[0])
        passages = [p for _, p in scored_passages]
        passages = _dedup_passages(passages)

        parts       = []
        used_chars  = 0
        confidence  = 0.0

        for i, (combo_score, passage) in enumerate(zip(
            [s for s, _ in scored_passages], passages
        ), 1):
            snippet = _truncate_to_sentences(passage, 180)
            if used_chars + len(snippet) > self.MAX_CONTEXT_CHARS - 100:
                break
            parts.append(f"[{i}] {snippet}")
            used_chars += len(snippet) + 6
            confidence  = max(confidence, combo_score)

        # Weather block
        weather_needed = (
            needs_weather(query)
            or intent in ("spray_advisory", "irrigation")
        )
        if weather_needed:
            loc = self.memory.get_location() or "Guwahati"
            try:
                weather_str = self.weather_svc.context_str(loc)
                parts.append(f"[Weather] {weather_str}")
            except Exception:
                pass

        # Farmer profile context
        profile = self.memory.long.as_text()
        if profile and profile != "No profile yet.":
            parts.append(f"[Farmer] {profile}")

        context = "\n\n".join(parts)
        return _truncate_to_sentences(context, self.MAX_CONTEXT_CHARS), confidence

    def get_retrieved(self, query: str):
        """Raw retrieval results (for L2 passage extraction)."""
        return self.retriever.retrieve(query, top_k=self.top_k)

    def get_intent(self, query: str) -> str:
        return detect_intent(query)

    def get_confidence(self, query: str) -> float:
        """Quick confidence check without building full context."""
        results = self.retriever.retrieve(query, top_k=3)
        if not results:
            return 0.0
        top_score = results[0][0]
        return float(top_score)