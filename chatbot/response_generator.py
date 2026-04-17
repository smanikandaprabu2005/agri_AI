"""
chatbot/response_generator.py  —  SageStorm V2.1
==================================================
Key upgrades over V2:
  - L2 passage selection uses hybrid score, not just cosine alone
  - Post-processing handles more tokeniser artefacts
  - `is_coherent` threshold tuned for agricultural vocabulary
  - Max sentences raised to 5 (was 4) for richer retrieval answers
  - Template list expanded with 8 more high-frequency crop queries
  - Fallback rotates 5 different polite messages (was 3)
  - Generation: temperature-fallback if SLM returns empty
"""

import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import build_prompt, build_rag_prompt
from weather.weather_api import needs_weather
from config import DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_TOP_K, GEN_TOP_P


# ── Template answers (highest confidence, rule-based) ────────
_TEMPLATES = [
    # Pest control
    (re.compile(r"\baph\w+.*lemon|lemon.*aph\w+", re.I),
     "Spray Malathion 50EC at 2 ml/litre water every 5 days, 3 times. "
     "Spray in early morning or evening. Avoid spraying on rainy days."),
    (re.compile(r"\bstem borer\b.*rice|rice.*stem borer", re.I),
     "Spray Classic-20 at 2 ml/litre water when dead-hearts appear. "
     "Clear field stubbles after harvest to remove egg masses."),
    (re.compile(r"\bwhitefly\b.*pepper|pepper.*whitefly", re.I),
     "Spray Rogor 35EC at 2 ml/litre water 3 times every 5 days. "
     "Spray in morning or evening for best contact."),
    (re.compile(r"\b(aphid|aphis)\b.*cotton|cotton.*aphid", re.I),
     "Spray Imidacloprid 17.8 SL at 0.5 ml/litre water. "
     "Repeat after 15 days if infestation persists."),
    (re.compile(r"\bworm.*tomato|tomato.*worm|fruit borer.*tomato", re.I),
     "Spray Indoxacarb 14.5 SC at 1 ml/litre water every 7 days. "
     "Remove and destroy infested fruits immediately."),
    (re.compile(r"\bthrips\b", re.I),
     "Spray Spinosad 45 SC at 0.3 ml/litre water. Alternate with "
     "Imidacloprid to prevent resistance. Avoid spraying in strong winds."),
    # Disease management
    (re.compile(r"\blate blight\b.*\b(tomato|potato)|tomato.*late blight", re.I),
     "Spray Indofil M-45 at 2.5 g/litre every 7 days. "
     "Remove and destroy infected leaves. Avoid overhead irrigation."),
    (re.compile(r"\bpowdery mildew\b", re.I),
     "Spray Sulphex (wettable sulphur) at 2 g/litre water or "
     "Hexaconazole 5 EC at 1 ml/litre every 10 days."),
    (re.compile(r"\bdowny mildew\b|\bdowny\b.*mildew", re.I),
     "Spray Ridomil Gold (Metalaxyl) at 2 g/litre water. "
     "Avoid excess irrigation. Remove volunteer plants."),
    (re.compile(r"\bblast\b.*rice|rice.*blast", re.I),
     "Spray Tricyclazole 75 WP at 0.6 g/litre water at tillering and "
     "booting stages. Maintain field drainage."),
    (re.compile(r"\bbacterial.*blight|blight.*rice|leaf blight", re.I),
     "Spray Streptomycin + Tetracycline mixture at 1 g/litre water "
     "every 10 days. Drain standing water from the field."),
    # Fertilization
    (re.compile(r"\b(fertiliz\w+|urea|npk)\b.*rice|rice.*(fertiliz|urea)", re.I),
     "For rice: Urea 18 kg/acre + DAP 27 kg/acre + MOP 6 kg/acre as basal. "
     "Top-dress Urea 18 kg/acre at tillering and panicle initiation."),
    (re.compile(r"\b(fertiliz\w+|urea)\b.*potato|potato.*fertiliz", re.I),
     "For potatoes: Urea 19 kg + Ammonium Phosphate 45 kg + MOP 12 kg per acre. "
     "Add 2 cart-loads of farmyard manure before planting."),
    (re.compile(r"\b(fertiliz\w+|urea)\b.*mustard|mustard.*fertiliz", re.I),
     "For mustard: Urea 12 kg + DAP 30 kg + MOP 3 kg per large acre. "
     "Apply sulphur 8 kg/acre for yield improvement."),
    (re.compile(r"\b(fertiliz\w+|npk)\b.*tomato|tomato.*fertiliz", re.I),
     "For tomatoes: Urea 25 kg + SSP 50 kg + MOP 20 kg per acre. "
     "Side-dress Urea 12 kg/acre at flowering."),
    (re.compile(r"\b(fertiliz\w+|npk)\b.*onion|onion.*fertiliz", re.I),
     "For onions: Urea 22 kg + SSP 38 kg + MOP 18 kg per acre. "
     "Apply in 2 splits: at transplanting and at 30 days after."),
    # Spray advisory
    (re.compile(r"\b(spray|pesticide)\b.*\b(rain|wet|raining|rainy)\b", re.I),
     "Do not spray pesticides when rain is expected — chemicals wash off. "
     "Wait for at least 6 dry hours after spraying before rain."),
    (re.compile(r"\bwhen.*spray|best time.*spray|spray.*morning|spray.*evening", re.I),
     "Best spraying times are early morning (6–9 AM) or evening (4–7 PM). "
     "Avoid midday (excessive evaporation) and windy conditions above 15 km/h."),
    # Spacing
    (re.compile(r"\bspacing|planting distance\b.*tomato|tomato.*(spacing|distance)", re.I),
     "Tomato planting spacing: 60 cm row-to-row × 45 cm plant-to-plant."),
    (re.compile(r"\bspacing|planting distance\b.*banana|banana.*(spacing|distance)", re.I),
     "Banana planting spacing: 1.8 m row-to-row × 1.8 m plant-to-plant. "
     "Tissue culture plants need 3 m × 3 m."),
    (re.compile(r"\bspacing\b.*onion|onion.*spacing", re.I),
     "Onion planting spacing: 15 cm row-to-row × 10 cm plant-to-plant."),
    (re.compile(r"\bspacing\b.*cabbage|cabbage.*spacing", re.I),
     "Cabbage spacing: 60 cm row-to-row × 45 cm plant-to-plant."),
]


def template_answer(q: str):
    for pat, ans in _TEMPLATES:
        if pat.search(q):
            return ans
    return None


# ── Retrieval passage extraction ──────────────────────────────
def best_passage(retrieved, query: str, min_score: float = 0.07) -> str:
    """
    Selects the best passage from dense retrieval results.
    Scores = dense_cosine × 0.5 + query_term_overlap × 0.5.
    Returns empty string if nothing passes threshold.
    """
    q_words = set(query.lower().split())
    best_score, best_text = -1.0, ""

    for cos, doc in retrieved:
        if cos < min_score:
            continue
        # split into sentences and score each
        for sent in re.split(r"[.!\n]+", doc):
            s = sent.strip()
            if len(s) < 25:
                continue
            d_words = set(s.lower().split())
            overlap = len(q_words & d_words) / max(len(q_words), 1)
            score   = cos * 0.5 + overlap * 0.5
            if score > best_score:
                best_score = score
                best_text  = s

    return best_text


# ── Post-processing helpers ───────────────────────────────────
_ARTEFACTS = [
    "### Instruction:", "### Input:", "### Response:",
    "### Context:", "<unk>", "<pad>", "<bos>", "<eos>",
    "[UNK]", "[PAD]", "[BOS]", "[EOS]",
]


def clean_output(text: str) -> str:
    for m in _ARTEFACTS:
        text = text.replace(m, "")
    # collapse repeated words (≥ 3 in a row)
    words  = text.split()
    out    = []
    prev   = None
    cnt    = 0
    for w in words:
        cnt = cnt + 1 if w == prev else 0
        if cnt < 3:
            out.append(w)
        prev = w
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    # capitalise first character
    return (text[0].upper() + text[1:]) if text else text


def is_coherent(text: str, min_words: int = 6) -> bool:
    """
    Checks that generated text is real language (not garbage tokens).
    Agricultural vocabulary has many abbreviations, so threshold is relaxed.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    # allow short abbreviations common in agri text (e.g. ml, kg, EC, WP)
    real = sum(1 for w in words if re.match(r"^[a-zA-Z]{2,}$", w))
    return (real / len(words)) > 0.30   # relaxed from 0.35 for agri vocab


def truncate(text: str, max_sent: int = 5) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8]
    return " ".join(sents[:max_sent])


# ── Response Generator ────────────────────────────────────────
class ResponseGenerator:
    _FALLBACK = [
        "I don't have specific information on that. Please consult your "
        "local agricultural extension officer (KVK).",
        "That's outside my current knowledge. Contact your nearest Krishi "
        "Vigyan Kendra for detailed advice.",
        "I'm not certain about this. Please consult an agricultural expert "
        "or your local agriculture office.",
        "For accurate advice on this, reach out to your nearest KVK or "
        "the ICAR regional research station.",
        "This requires specialist knowledge. Please contact the agriculture "
        "helpline 1800-180-1551 (toll-free in India).",
    ]
    _fi = 0

    def __init__(self, model, tokenizer, retriever, ctx_builder,
                 max_tokens=GEN_MAX_TOKENS, temperature=GEN_TEMPERATURE):
        self.model        = model
        self.tokenizer    = tokenizer
        self.retriever    = retriever
        self.ctx_builder  = ctx_builder
        self.max_tokens   = max_tokens
        self.temperature  = temperature

    def _slm_generate(self, query: str, context: str = "",
                      temperature: float = None) -> str:
        if self.model is None:
            return ""
        if temperature is None:
            temperature = self.temperature
        try:
            prompt  = (build_rag_prompt(query, context) if context
                       else build_prompt(query))
            ids     = self.tokenizer.encode_prompt(prompt)
            gen_ids = self.model.generate(
                ids,
                max_tokens  = self.max_tokens,
                temperature = temperature,
                device      = DEVICE,
            )
            return self.tokenizer.decode(gen_ids, skip_special=True)
        except Exception as e:
            print(f"[Generator] SLM error: {e}")
            return ""

    def generate(self, query: str):
        """Returns (answer_str, source_label)."""

        # ── L1: Template (highest precision) ─────────────────
        t = template_answer(query)
        if t:
            return t, "template"

        # ── L2: Retrieval passage ─────────────────────────────
        retrieved = self.ctx_builder.get_retrieved(query)
        passage   = best_passage(retrieved, query)
        if passage and len(passage.split()) >= 8:
            if needs_weather(query):
                adv     = self.ctx_builder.weather_svc.advisory()
                passage = f"{passage}\n\n[Weather] {adv}"
            return passage, "retrieval"

        # ── L3: SageStorm generation ──────────────────────────
        context = self.ctx_builder.build_context_str(query)
        raw     = self._slm_generate(query, context)
        cleaned = truncate(clean_output(raw))

        if is_coherent(cleaned):
            return cleaned, "sagestorm"

        # temperature-fallback: try with higher temperature for diversity
        raw2 = self._slm_generate(query, context, temperature=0.9)
        cleaned2 = truncate(clean_output(raw2))
        if is_coherent(cleaned2):
            return cleaned2, "sagestorm"

        # ── L4: Polite fallback ───────────────────────────────
        msg = self._FALLBACK[self._fi % len(self._FALLBACK)]
        self._fi += 1
        return msg, "fallback"
