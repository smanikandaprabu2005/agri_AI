"""
chatbot/response_generator.py
==============================
4-layer response strategy:
  L1: Template matching  (instant, rule-based)
  L2: Retrieval passage  (RAG, no generation needed)
  L3: SageStorm V2       (48M GPT, full RAG prompt)
  L4: Polite fallback    (always coherent)
"""

import re, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import build_prompt, build_rag_prompt
from weather.weather_api import needs_weather
from config import DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P


# ── Template answers (highest confidence) ────────────────────
_TEMPLATES = [
    (re.compile(r"\b(spray|pesticide)\b.*\b(rain|wet|raining)\b", re.I),
     "Do not spray pesticides when rain is expected — chemicals will wash off. Wait for dry weather."),
    (re.compile(r"\baph\w+.*lemon|lemon.*aph\w+", re.I),
     "For aphids on lemon trees: spray Malathion 50EC at 2 ml/litre water, every 5 days × 3 times. Avoid spraying on rainy days."),
    (re.compile(r"\bstem borer\b.*rice|rice.*stem borer", re.I),
     "To control stem borers in rice: spray Classic-20 at 2 ml/litre water. Clear field stubbles to remove breeding sites."),
    (re.compile(r"\blate blight\b.*\b(tomato|potato)|tomato.*late blight", re.I),
     "For late blight: spray Indofil M-45 at 2.5 g/litre every 7 days. Remove and destroy infected leaves immediately."),
    (re.compile(r"\b(fertiliz\w+|urea|npk)\b.*rice|rice.*(fertiliz|urea)", re.I),
     "For rice: apply Urea 18 kg/acre + Di-ammonium Phosphate 27 kg/acre + Potassium Chloride 6 kg/acre as base fertilizer."),
    (re.compile(r"\b(fertiliz\w+|urea)\b.*potato|potato.*fertiliz", re.I),
     "For potatoes: apply Urea 19 kg + Ammonium Phosphate 45 kg + Potassium Chloride 12 kg per acre. Add 2 loads of farmyard manure."),
    (re.compile(r"\b(fertiliz\w+|urea)\b.*mustard|mustard.*fertiliz", re.I),
     "For mustard: apply Urea 12 kg + Di-ammonium Phosphate 30 kg + Potassium Chloride 3 kg per large acre."),
    (re.compile(r"\bspacing|planting distance\b.*tomato|tomato.*(spacing|distance)", re.I),
     "Tomato planting spacing: 60 cm between rows × 45 cm between plants."),
    (re.compile(r"\bspacing|planting distance\b.*banana|banana.*(spacing|distance)", re.I),
     "Banana planting spacing: 1.8 m between rows × 1.8 m between plants."),
    (re.compile(r"\bwhitefly\b.*pepper|pepper.*whitefly", re.I),
     "For whitefly on pepper: spray Rogor 35EC at 2 ml/litre water, spray in morning or evening, 3 times every 5 days."),
]

def template_answer(q):
    for pat, ans in _TEMPLATES:
        if pat.search(q): return ans
    return None


# ── Retrieval passage extraction ──────────────────────────────
def best_passage(retrieved, query, min_score=0.08):
    qw = set(query.lower().split())
    best_s, best_t = -1, ""
    for cos, doc in retrieved:
        if cos < min_score: continue
        for sent in re.split(r"[.!\n]+", doc):
            s = sent.strip()
            if len(s) < 25: continue
            overlap = len(qw & set(s.lower().split())) / max(len(qw), 1)
            score   = cos * 0.5 + overlap * 0.5
            if score > best_s:
                best_s = score; best_t = s
    return best_t


# ── Post-processing ───────────────────────────────────────────
def clean_output(text):
    for m in ["### Instruction:", "### Input:", "### Response:", "### Context:",
              "<unk>", "<pad>", "<bos>", "<eos>"]:
        text = text.replace(m, "")
    # dedup repeated words (>= 3 in a row)
    words = text.split()
    out = []
    prev, cnt = None, 0
    for w in words:
        if w == prev: cnt += 1
        else: cnt = 0
        if cnt < 3: out.append(w)
        prev = w
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[0].upper() + text[1:]) if text else text


def is_coherent(text, min_words=6):
    words = text.split()
    if len(words) < min_words: return False
    real = sum(1 for w in words if re.match(r"^[a-zA-Z]{2,}$", w))
    return real / len(words) > 0.35


def truncate(text, max_sent=4):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8]
    return " ".join(sents[:max_sent])


# ── Response Generator ────────────────────────────────────────
class ResponseGenerator:
    _FALLBACK = [
        "I don't have specific information on that. Please consult your local agricultural extension officer (KVK).",
        "That's outside my current knowledge. Contact your nearest Krishi Vigyan Kendra for detailed advice.",
        "I'm not certain. Please consult an agricultural expert or your local agriculture office.",
    ]
    _fi = 0

    def __init__(self, model, tokenizer, retriever, ctx_builder,
                 max_tokens=GEN_MAX_TOKENS, temperature=GEN_TEMPERATURE):
        self.model       = model
        self.tokenizer   = tokenizer
        self.retriever   = retriever
        self.ctx_builder = ctx_builder
        self.max_tokens  = max_tokens
        self.temperature = temperature

    def _slm_generate(self, query, context=""):
        if self.model is None: return ""
        try:
            prompt  = build_rag_prompt(query, context) if context else build_prompt(query)
            ids     = self.tokenizer.encode_prompt(prompt)
            gen_ids = self.model.generate(ids, max_tokens=self.max_tokens,
                                           temperature=self.temperature, device=DEVICE)
            return self.tokenizer.decode(gen_ids, skip_special=True)
        except Exception as e:
            print(f"[Generator] SLM error: {e}")
            return ""

    def generate(self, query):
        """Returns (answer, source_label)."""

        # L1: Template
        t = template_answer(query)
        if t: return t, "template"

        # L2: Retrieval passage
        retrieved = self.ctx_builder.get_retrieved(query)
        passage   = best_passage(retrieved, query)
        if passage and len(passage.split()) >= 8:
            if needs_weather(query):
                adv = self.ctx_builder.weather_svc.advisory()
                passage = f"{passage}\n\n[Weather] {adv}"
            return passage, "retrieval"

        # L3: SageStorm generation
        context = self.ctx_builder.build_context_str(query)
        raw     = self._slm_generate(query, context)
        cleaned = truncate(clean_output(raw))
        if is_coherent(cleaned):
            return cleaned, "sagestorm"

        # L4: Fallback
        msg = self._FALLBACK[self._fi % len(self._FALLBACK)]
        self._fi += 1
        return msg, "fallback"
