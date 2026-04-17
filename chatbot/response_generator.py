"""
chatbot/response_generator.py
==============================
4-layer response strategy:
  L1: Template matching  (instant, rule-based)
  L2: Retrieval passage  (RAG, no generation needed)
  L3: SageStorm V2       (48M GPT, full RAG prompt)
  L4: Polite fallback    (always coherent)

FIXES & IMPROVEMENTS vs original:
  FIX 1:  build_rag_prompt / build_prompt imported from canonical tokenizer.py
           (was duplicating prompt format — caused response masking mismatch)
  FIX 2:  best_passage() overlap score divided by 0 when query is empty
  FIX 3:  clean_output() regex for dedup ran on words but should be on n-grams;
          replaced with a smarter trigram dedup
  FIX 4:  is_coherent() threshold 0.35 too low — raised to 0.45
  FIX 5:  _slm_generate() caught all Exceptions silently; now logs tb in debug
  NEW:    Temperature auto-adjusts based on intent (lower for spray/fertilizer)
  NEW:    Passage scoring uses query-term IDF weighting, not flat overlap
"""

import re, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# FIX 1: Import canonical prompt builders
from models.tokenizer import build_prompt, build_rag_prompt
from weather.weather_api import needs_weather
from config import DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P


# ══════════════════════════════════════════════════════════════
#  Template answers (highest confidence, instant)
# ══════════════════════════════════════════════════════════════
_TEMPLATES = [
    (re.compile(r"\b(spray|pesticide)\b.*\b(rain|wet|raining)\b", re.I),
     "Do not spray pesticides when rain is expected — chemicals will wash off. "
     "Wait for dry weather with wind speed below 10 km/h."),

    (re.compile(r"\baph\w+.*(lemon|citrus)|lemon.*aph\w+", re.I),
     "For aphids on lemon trees: spray Malathion 50EC at 2 ml/litre water, "
     "every 5 days × 3 times. Avoid spraying in rain or strong wind."),

    (re.compile(r"\bstem.?borer\b.*rice|rice.*stem.?borer", re.I),
     "For stem borers in rice: spray Chlorpyriphos 20EC at 2 ml/litre water "
     "or Carbofuran 3G at 10 kg/acre. Remove and burn infested tillers."),

    (re.compile(r"\blate.?blight\b.*(tomato|potato)|(tomato|potato).*late.?blight", re.I),
     "For late blight: spray Mancozeb 75WP at 2.5 g/litre or Metalaxyl-M every "
     "7 days. Remove and destroy infected leaves immediately."),

    (re.compile(r"\b(fertiliz\w+|urea|npk)\b.*rice|rice.*(fertiliz|urea)", re.I),
     "For rice: apply Urea 18 kg/acre + DAP 27 kg/acre + MOP 6 kg/acre as basal. "
     "Top-dress urea at tillering (21 DAT) and panicle initiation (45 DAT)."),

    (re.compile(r"\b(fertiliz\w+|urea)\b.*potato|potato.*fertiliz", re.I),
     "For potatoes: apply Urea 19 kg + DAP 45 kg + MOP 12 kg per acre as basal. "
     "Add 2 tonnes FYM per acre before planting."),

    (re.compile(r"\b(fertiliz\w+|urea)\b.*mustard|mustard.*fertiliz", re.I),
     "For mustard: apply Urea 12 kg + DAP 30 kg + MOP 3 kg per large acre at sowing."),

    (re.compile(r"\b(spacing|planting.?distance)\b.*tomato|tomato.*(spacing|distance)", re.I),
     "Tomato planting spacing: 60 cm between rows × 45 cm between plants "
     "(irrigated); 75 × 60 cm for rain-fed conditions."),

    (re.compile(r"\b(spacing|planting.?distance)\b.*banana|banana.*(spacing|distance)", re.I),
     "Banana planting spacing: 1.8 m × 1.8 m (dwarf varieties); "
     "2.1 m × 2.1 m (tall varieties)."),

    (re.compile(r"\bwhitefly\b.*(pepper|chilli)|pepper.*whitefly", re.I),
     "For whitefly on pepper: spray Imidacloprid 17.8SL at 0.5 ml/litre water. "
     "Spray in morning or evening, repeat after 10 days if needed."),

    (re.compile(r"\bpowdery.?mildew\b", re.I),
     "For powdery mildew: spray Sulphur 80WP at 2.5 g/litre or "
     "Hexaconazole 5EC at 2 ml/litre. Improve air circulation in the field."),
]


def template_answer(q: str) -> str | None:
    for pat, ans in _TEMPLATES:
        if pat.search(q):
            return ans
    return None


# ══════════════════════════════════════════════════════════════
#  Passage scoring helpers
# ══════════════════════════════════════════════════════════════
def _query_idf_weights(query: str, doc_pool: list[str]) -> dict[str, float]:
    """Light IDF over retrieved doc pool for query term weighting."""
    qwords  = set(query.lower().split())
    N       = max(len(doc_pool), 1)
    df      = Counter()
    for doc in doc_pool:
        words = set(doc.lower().split())
        for w in qwords:
            if w in words:
                df[w] += 1
    import math
    return {w: math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0 for w in qwords}


def best_passage(
    retrieved : list[tuple[float, str]],
    query     : str,
    min_score : float = 0.05,
) -> str:
    """
    Extract the single best passage from retrieved docs.
    FIX 2: guards against zero-length query.
    """
    if not query.strip() or not retrieved:
        return ""

    docs    = [doc for _, doc in retrieved]
    weights = _query_idf_weights(query, docs)
    qw      = set(query.lower().split())
    best_s, best_t = -1.0, ""

    for cos_score, doc in retrieved:
        if cos_score < min_score:
            continue
        for sent in re.split(r"[.!\n]+", doc):
            s = sent.strip()
            if len(s) < 25:
                continue
            doc_words = set(s.lower().split())
            # Weighted overlap
            overlap = sum(weights.get(w, 1.0) for w in qw if w in doc_words)
            norm_q  = sum(weights.values()) or 1.0
            score   = cos_score * 0.5 + (overlap / norm_q) * 0.5
            if score > best_s:
                best_s = score
                best_t = s

    return best_t


# ══════════════════════════════════════════════════════════════
#  Post-processing
# ══════════════════════════════════════════════════════════════
_SPECIAL_MARKERS = [
    "### Instruction:", "### Input:", "### Response:", "### Context:",
    "<unk>", "<pad>", "<bos>", "<eos>",
]


def clean_output(text: str) -> str:
    for m in _SPECIAL_MARKERS:
        text = text.replace(m, "")

    # FIX 3: Trigram dedup (original word dedup was too aggressive on sentences)
    words     = text.split()
    out_words = []
    trigrams: Counter = Counter()
    for i, w in enumerate(words):
        trigram = tuple(words[i: i + 3])
        if len(trigram) == 3 and trigrams[trigram] >= 2:
            continue           # skip this word — part of repeated trigram
        trigrams[tuple(words[max(0, i - 2): i + 1])] += 1
        out_words.append(w)

    text = " ".join(out_words)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[0].upper() + text[1:]) if text else text


def is_coherent(text: str, min_words: int = 8) -> bool:
    """
    FIX 4: Raised alpha-word ratio threshold 0.35 → 0.45.
    0.35 was accepting gibberish with too many numbers/special chars.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    real = sum(1 for w in words if re.match(r"^[a-zA-Z]{2,}$", w))
    return (real / len(words)) > 0.45


def truncate(text: str, max_sent: int = 5) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8]
    return " ".join(sents[:max_sent])


# ══════════════════════════════════════════════════════════════
#  Intent → temperature mapping (lower = more deterministic)
# ══════════════════════════════════════════════════════════════
_INTENT_TEMPS = {
    "spray_advisory":    0.50,   # exact dosage — be deterministic
    "fertilization":     0.55,
    "pest_control":      0.60,
    "disease_management":0.60,
    "government_scheme": 0.55,
    "planting":          0.65,
    "irrigation":        0.65,
    "harvest":           0.65,
    "general_agriculture": 0.70,
}


# ══════════════════════════════════════════════════════════════
#  Response Generator
# ══════════════════════════════════════════════════════════════
class ResponseGenerator:
    _FALLBACK = [
        "I don't have specific information on that. Please consult your local "
        "agricultural extension officer (KVK).",
        "That's outside my current knowledge. Contact your nearest "
        "Krishi Vigyan Kendra for detailed advice.",
        "I'm not certain about this. Please consult an agricultural expert "
        "or your local agriculture office.",
    ]
    _fi = 0

    def __init__(self, model, tokenizer, retriever, ctx_builder,
                 max_tokens: int = GEN_MAX_TOKENS,
                 temperature: float = GEN_TEMPERATURE):
        self.model       = model
        self.tokenizer   = tokenizer
        self.retriever   = retriever
        self.ctx_builder = ctx_builder
        self.max_tokens  = max_tokens
        self.temperature = temperature

    def _slm_generate(self, query: str, context: str = "", intent: str = "") -> str:
        if self.model is None:
            return ""
        # NEW: intent-based temperature
        temp = _INTENT_TEMPS.get(intent, self.temperature)
        try:
            prompt  = build_rag_prompt(query, context) if context else build_prompt(query)
            ids     = self.tokenizer.encode_prompt(prompt)
            gen_ids = self.model.generate(
                ids,
                max_tokens  = self.max_tokens,
                temperature = temp,
                device      = DEVICE,
            )
            return self.tokenizer.decode(gen_ids, skip_special=True)
        except Exception as e:
            # FIX 5: Log traceback in debug mode
            import traceback
            print(f"[Generator] SLM error: {e}")
            if os.environ.get("SAGE_DEBUG"):
                traceback.print_exc()
            return ""

    def generate(self, query: str) -> tuple[str, str]:
        """Returns (answer, source_label)."""

        # L1: Template (instant, highest precision)
        t = template_answer(query)
        if t:
            return t, "template"

        # L2: Best retrieval passage
        intent    = self.ctx_builder.get_intent(query)
        retrieved = self.ctx_builder.get_retrieved(query)
        passage   = best_passage(retrieved, query)

        if passage and len(passage.split()) >= 8:
            if needs_weather(query):
                try:
                    adv = self.ctx_builder.weather_svc.advisory()
                    passage = f"{passage}\n\n[Weather advisory] {adv}"
                except Exception:
                    pass
            return passage, "retrieval"

        # L3: SageStorm generation
        context = self.ctx_builder.build_context_str(query)
        raw     = self._slm_generate(query, context, intent)
        cleaned = truncate(clean_output(raw))
        if is_coherent(cleaned):
            return cleaned, "sagestorm"

        # L4: Polite fallback
        msg = self._FALLBACK[self._fi % len(self._FALLBACK)]
        self._fi += 1
        return msg, "fallback"