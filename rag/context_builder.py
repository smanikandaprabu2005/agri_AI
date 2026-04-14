"""
rag/context_builder.py
======================
Assembles the full RAG prompt:
  Context (retrieved docs) + Weather + Memory + Question → Answer:
"""

import re, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weather.weather_api import needs_weather

_PEST    = re.compile(r"\b(pest|insect|aphid|worm|caterpillar|fly|beetle|mite|thrips|whitefly|stem borer|termite)\b", re.I)
_DISEASE = re.compile(r"\b(disease|blight|rot|wilt|mildew|fungal|bacterial|viral|spot|rust|leaf curl|anthracnose|blast)\b", re.I)
_FERT    = re.compile(r"\b(fertiliz|urea|potash|phosphate|npk|nutrient|manure|compost)\b", re.I)
_SPRAY   = re.compile(r"\b(spray|pesticide|fungicide|insecticide|dose|dosage)\b", re.I)
_PLANT   = re.compile(r"\b(plant|sow|seed|seedling|transplant|cultivat|growing)\b", re.I)

def detect_intent(q):
    if _PEST.search(q):    return "pest_control"
    if _DISEASE.search(q): return "disease_management"
    if _FERT.search(q):    return "fertilization"
    if _SPRAY.search(q):   return "spray_advisory"
    if _PLANT.search(q):   return "planting"
    return "general_agriculture"


class ContextBuilder:
    def __init__(self, retriever, memory, weather_svc, top_k=5):
        self.retriever   = retriever
        self.memory      = memory
        self.weather_svc = weather_svc
        self.top_k       = top_k

    def build_context_str(self, query):
        """Build context string for RAG prompt (max 600 chars)."""
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        parts     = []

        for i, (score, doc) in enumerate(retrieved[:3], 1):
            if score >= 0.05:
                snippet = " ".join(doc.split()[:60])
                parts.append(f"[{i}] {snippet}")

        if needs_weather(query) or detect_intent(query) == "spray_advisory":
            loc = self.memory.get_location()
            parts.append("[Weather] " + self.weather_svc.context_str(loc))

        return "\n\n".join(parts)[:600]

    def get_retrieved(self, query):
        return self.retriever.retrieve(query, top_k=self.top_k)

    def get_intent(self, query):
        return detect_intent(query)
