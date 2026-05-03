"""
chatbot/response_generator.py
==============================
FIXED v2.2 — All generation bugs patched.

FIXES:
  FIX 1: encode_prompt() no longer adds BOS — tokenizer already adds it during
          fine-tuning (build_training_prompt prepends BOS via step5). Double-BOS
          caused the model to decode from offset position → "enetic..." garbage.

  FIX 2: _slm_generate() now uses build_rag_prompt() with the agriculture-specific
          instruction format that matches training, instead of a generic freeform prompt.

  FIX 3: Added missing template patterns:
          - "Will rain affect fertilizer/spray" → weather safety
          - "High yield crops" / "best crops for region" → crop selection
          - "NPK requirement for <crop>" → nutrient templates
          - "Which crop is best for <soil>" → soil-based crop advice

  FIX 4: best_passage() minimum score raised from 0.08 → 0.15, and minimum
          sentence length raised from 12 → 20 words to filter low-quality hits.

  FIX 5: score_rag_answer() passing threshold raised from 4 → 5 (out of 6).
          Added new check: answer must NOT contain hallucination markers like
          "sensor readings", "NT,", "Poultry,", "Key,", "NTESTIN".

  FIX 6: Coherence check tightened — minimum words 8 → 15, alpha ratio 0.35 → 0.50.

  FIX 7: RAG answer validation now checks the answer actually contains at least
          one agriculture keyword before accepting it.
"""

import re, os, sys
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import build_rag_prompt
from weather.weather_api import needs_weather
from config import (
    DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE,
    GEN_TOP_K, GEN_TOP_P, MIN_RETRIEVAL_SCORE,
)


# ══════════════════════════════════════════════════════════════
#  Template answers (L1) — comprehensive patterns
# ══════════════════════════════════════════════════════════════
_TEMPLATES = [
    # ── Weather + spray safety ────────────────────────────────
    (re.compile(r"\b(spray|pesticide|fertilizer|fertiliser)\b.*\b(rain|wet|raining|rainfall|weather|affect)\b", re.I),
     "Yes, rain significantly affects fertilizer and pesticide application. "
     "Do NOT spray pesticides when rain is expected within 4–6 hours — chemicals will wash off. "
     "Similarly, avoid foliar fertilizer sprays before rain. "
     "After heavy rain, wait 24–48 hours for the foliage to dry before spraying. "
     "Apply fertilizers to moist (not waterlogged) soil for best absorption."),

    (re.compile(r"\b(rain|weather)\b.*\b(affect|impact|effect)\b.*\b(fertilizer|spray|pesticide)\b", re.I),
     "Yes, rain significantly affects fertilizer and pesticide application. "
     "Do NOT spray pesticides when rain is expected within 4–6 hours — chemicals will wash off. "
     "Similarly, avoid foliar fertilizer sprays before rain. "
     "After heavy rain, wait 24–48 hours for the foliage to dry before spraying. "
     "Apply fertilizers to moist (not waterlogged) soil for best absorption."),

    (re.compile(r"\b(spray|pesticide)\b.*\b(rain|wet|raining)\b", re.I),
     "Do not spray pesticides when rain is expected — chemicals will wash off. "
     "Wait for dry weather and apply in the early morning or late evening."),

    (re.compile(r"\b(spray|pesticide).*(today|safe|when)", re.I),
     "Check local weather before spraying. Ideal spray conditions: no rain for 24 hours, "
     "wind below 10 km/h, temperature below 35°C. Spray early morning (6–9 AM) or late evening (after 5 PM). "
     "Avoid spraying on cloudy or humid days for fungicides."),

    # ── High yield / crop selection for region ────────────────
    (re.compile(r"\b(high.?yield|best.?crop|suitable.?crop|top.?crop|which.?crop|what.?crop)\b.*(sivakasi|tamil.?nadu|south)", re.I),
     "In the Sivakasi and Tamil Nadu region, high-yielding crops include:\n"
     "• Cotton — well-suited for the semi-arid climate and red loamy soil\n"
     "• Groundnut — thrives in sandy loam soil, Kharif and Rabi seasons\n"
     "• Sunflower — drought-tolerant, good for dry spells\n"
     "• Chilli — Sivakasi region is known for chilli cultivation\n"
     "• Sorghum (Jowar) — excellent for rainfed/dry land farming\n"
     "• Cowpea/Blackgram — good nitrogen-fixing legumes for rotation\n"
     "Choose based on your soil type, water availability, and current season."),

    (re.compile(r"\b(high.?yield|best.?crop|suitable.?crop|profitable.?crop|which.?crop)\b.*(region|area|location|place|district)", re.I),
     "High-yielding crop selection depends on your region's climate and soil. "
     "Please tell me your location and soil type for specific recommendations. "
     "Generally, for Kharif season: Rice (irrigated), Maize, Cotton, Groundnut, Sorghum. "
     "For Rabi season: Wheat, Mustard, Chickpea, Potato, Sunflower. "
     "For best results, contact your local KVK (Krishi Vigyan Kendra) for region-specific variety recommendations."),

    (re.compile(r"\b(crops?).*(high.?yield|give.*(high|good|more).*(yield|profit))\b", re.I),
     "High-yielding crop selection depends on your soil and season. "
     "For Kharif (June–October): Rice, Maize, Cotton, Groundnut. "
     "For Rabi (October–March): Wheat, Mustard, Chickpea, Potato. "
     "Share your location and soil type for a precise recommendation."),

    # ── Aphid management ─────────────────────────────────────
    (re.compile(r"\baph\w+.*(lemon|citrus)|lemon.*aph\w+", re.I),
     "For aphids on lemon trees: spray Malathion 50EC at 2 ml/litre water, "
     "every 5 days × 3 times. Alternatively, use Imidacloprid 17.8SL at 0.25 ml/litre. "
     "Avoid spraying on rainy or very hot days. Spray in the morning."),

    (re.compile(r"\baph\w+.*(cotton|okra|bhindi)|cotton.*aph\w+", re.I),
     "For aphids on cotton: spray Imidacloprid 17.8 SL at 0.25 ml/litre or "
     "Dimethoate 30 EC at 1.5 ml/litre. Spray in the morning."),

    (re.compile(r"\baph\w+\b", re.I),
     "For aphid control: spray Imidacloprid 17.8SL at 0.25 ml/litre or "
     "Malathion 50EC at 2 ml/litre. Repeat after 7 days if infestation persists. "
     "You can also use a neem oil spray (5 ml/litre) as an organic option."),

    # ── Stem borer ────────────────────────────────────────────
    (re.compile(r"\bstem.?borer\b.*rice|rice.*stem.?borer", re.I),
     "To control stem borers in rice: spray Chlorpyriphos 20EC at 2.5 ml/litre "
     "water or Cartap Hydrochloride 50SP at 2 g/litre. "
     "Apply at tillering stage and again at panicle initiation. "
     "Also clear field stubbles after harvest to remove breeding sites."),

    (re.compile(r"\bstem.?borer\b", re.I),
     "For stem borer control: spray Chlorpyriphos 20EC at 2.5 ml/litre or "
     "Coragen (Chlorantraniliprole) at 0.4 ml/litre. "
     "Apply at first sign of dead hearts. Clear stubbles after harvest."),

    # ── Late blight ───────────────────────────────────────────
    (re.compile(r"\blate.?blight\b.*(tomato|potato)|tomato.*late.?blight|potato.*late.?blight", re.I),
     "For late blight: spray Mancozeb 75WP at 2.5 g/litre or Metalaxyl-M + "
     "Mancozeb (Ridomil Gold) at 2.5 g/litre every 7 days. "
     "Remove and destroy infected leaves immediately. Avoid overhead irrigation."),

    # ── NPK / Fertilizer requirements ─────────────────────────
    (re.compile(r"\b(npk|nitrogen|phosphorus|potassium|nutrient).*(requirement|dose|need|for).*(groundnut|peanut)", re.I),
     "NPK requirement for groundnut per acre:\n"
     "• Nitrogen (N): 10–12 kg/acre (Urea 22–26 kg)\n"
     "• Phosphorus (P): 20–25 kg/acre (SSP 125–156 kg or DAP 43–54 kg)\n"
     "• Potassium (K): 20–25 kg/acre (MOP 33–42 kg)\n"
     "Apply the entire dose as basal before sowing. "
     "Groundnut being a legume fixes atmospheric nitrogen, so N dose is low. "
     "Calcium (gypsum 200 kg/acre) at pegging stage improves pod filling."),

    (re.compile(r"\b(npk|fertiliz\w+|nutrient).*(requirement|dose|need|for).*(rice|paddy)", re.I),
     "NPK requirement for rice (transplanted) per acre:\n"
     "• Basal: Urea 18 kg + DAP 27 kg + MOP 10 kg\n"
     "• First top-dress: Urea 18 kg at tillering (21–25 days after transplanting)\n"
     "• Second top-dress: Urea 18 kg at panicle initiation\n"
     "Total: N=54 kg, P=50 kg, K=33 kg per acre approximately."),

    (re.compile(r"\b(npk|fertiliz\w+|nutrient).*(requirement|dose|need|for).*(wheat)", re.I),
     "NPK requirement for wheat per acre:\n"
     "• Basal: Urea 25 kg + DAP 30 kg + MOP 12 kg\n"
     "• Top-dress: Urea 25 kg at Crown Root Initiation (first irrigation)\n"
     "Total: N=50 kg, P=56 kg, K=20 kg per acre approximately."),

    (re.compile(r"\b(npk|fertiliz\w+|nutrient).*(requirement|dose|need|for).*(tomato)", re.I),
     "NPK requirement for tomato per acre:\n"
     "• Basal: FYM 4 tonnes + DAP 50 kg + MOP 30 kg\n"
     "• Top-dress 1 (30 DAT): Urea 20 kg\n"
     "• Top-dress 2 (60 DAT): Urea 20 kg + MOP 15 kg\n"
     "Total: N=40 kg, P=23 kg, K=45 kg per acre approximately."),

    # ── Rice fertilizer ───────────────────────────────────────
    (re.compile(r"\b(fertiliz\w+|urea|npk)\b.*rice|rice.*(fertiliz|urea)", re.I),
     "For rice (transplanted): apply Urea 18 kg/acre + Di-ammonium Phosphate "
     "27 kg/acre + Muriate of Potash 10 kg/acre as basal dose. "
     "Apply 18 kg Urea at tillering and again at panicle initiation."),

    # ── Wheat fertilizer ──────────────────────────────────────
    (re.compile(r"\b(fertiliz\w+|urea)\b.*wheat|wheat.*fertiliz", re.I),
     "For wheat: apply Urea 25 kg/acre + DAP 30 kg/acre + MOP 12 kg/acre as basal. "
     "Top-dress with Urea 25 kg/acre at first irrigation (Crown Root Initiation stage)."),

    # ── Tomato spacing ────────────────────────────────────────
    (re.compile(r"\b(spacing|planting.?distance)\b.*tomato|tomato.*(spacing|distance)", re.I),
     "Tomato planting spacing: 60–75 cm between rows × 45–60 cm between plants. "
     "For indeterminate varieties, use wider spacing (75 × 60 cm)."),

    # ── Whitefly on pepper ────────────────────────────────────
    (re.compile(r"\bwhitefly\b.*pepper|pepper.*whitefly", re.I),
     "For whitefly on pepper: spray Imidacloprid 17.8SL at 0.3 ml/litre or "
     "Thiamethoxam 25WG at 0.3 g/litre. Spray in morning or evening, 3 times "
     "every 7 days."),

    # ── Late blight (generic) ─────────────────────────────────
    (re.compile(r"\blate.?blight\b", re.I),
     "For late blight: spray Mancozeb 75WP at 2.5 g/litre or Metalaxyl-M + "
     "Mancozeb at 2.5 g/litre every 7 days. Remove infected leaves immediately."),

    # ── Powdery mildew ────────────────────────────────────────
    (re.compile(r"\bpowdery.?mildew\b", re.I),
     "For powdery mildew: spray Wettable Sulfur 80WP at 3 g/litre or "
     "Propiconazole (Tilt) at 1 ml/litre. Apply at first sign, repeat after 10–14 days."),

    # ── Fruit fly ─────────────────────────────────────────────
    (re.compile(r"\bfruit.?fly\b", re.I),
     "For fruit fly: use protein hydrolysate bait (Malathion 50EC 1 ml + sugar "
     "10 g per litre) in bottle traps at 25 per acre. Cover fruits with newspaper bags."),

    # ── Weed control ──────────────────────────────────────────
    (re.compile(r"\b(weed|weeding|herbicide)\b.*rice|rice.*weed", re.I),
     "For weed control in rice: apply Butachlor 50EC at 2 litres/acre within "
     "3 days of transplanting. For post-emergence, use Bispyribac-sodium at 125 ml/acre "
     "at 15–20 days after transplanting."),

    # ── Soil pH correction ────────────────────────────────────
    (re.compile(r"\b(acidic.?soil|soil.?ph|lime)\b", re.I),
     "For acidic soil correction: apply Agricultural lime (CaCO₃) at 2–4 t/ha "
     "or Dolomite at 2 t/ha. Apply 6 months before sowing. Target pH 6.0–7.0."),

    # ── Seed treatment ────────────────────────────────────────
    (re.compile(r"\b(seed.?treatment|treat.?seed)\b", re.I),
     "Standard seed treatment: soak seeds in Carbendazim 50WP (2 g/kg seed) or "
     "Thiram 75WP (3 g/kg seed). For bacterial diseases, treat with Pseudomonas "
     "fluorescens (10 g/kg seed)."),

    # ── PM Kisan ──────────────────────────────────────────────
    (re.compile(r"\bpm.?kisan\b", re.I),
     "PM-KISAN scheme provides ₹6,000/year in 3 equal instalments. "
     "Register at pmkisan.gov.in or through your nearest CSC/bank. "
     "Requirements: Aadhaar card, bank account, land records."),

    # ── Crop insurance ────────────────────────────────────────
    (re.compile(r"\b(crop.?insurance|pmfby|fasal.?bima)\b", re.I),
     "Pradhan Mantri Fasal Bima Yojana (PMFBY): Premium 2% for Kharif, 1.5% for Rabi, "
     "5% for commercial crops. Enrol before cut-off date through your bank or CSC."),

    # ── Soil testing ──────────────────────────────────────────
    (re.compile(r"\b(soil.?test|soil.?card|soil.?health)\b", re.I),
     "Get your Soil Health Card from the nearest agriculture department or KVK. "
     "Test for N-P-K, pH, organic carbon, and micronutrients. Retest every 3 years."),
]


def template_answer(query: str) -> Optional[str]:
    for pat, ans in _TEMPLATES:
        if pat.search(query):
            return ans
    return None


# ══════════════════════════════════════════════════════════════
#  Passage extraction helper (L2/L3)
# ══════════════════════════════════════════════════════════════
def best_passage(retrieved, query: str, min_score: float = 0.15) -> Tuple[Optional[str], float]:
    """
    Returns (best_passage, confidence) from retrieved results.
    FIX 4: min_score raised 0.08 → 0.15, min sentence words raised 12 → 20.
    """
    q_words    = set(query.lower().split())
    best_text  = None
    best_score = 0.0

    for cos, doc in retrieved:
        if cos < min_score:
            continue
        for sent in re.split(r"[.!\n]+", doc):
            s = sent.strip()
            # FIX 4: stricter minimum length
            if len(s.split()) < 20:
                continue
            overlap = len(q_words & set(s.lower().split())) / max(len(q_words), 1)
            score   = cos * 0.5 + overlap * 0.5
            if score > best_score:
                best_score = score
                best_text  = s

    return best_text, best_score


# ══════════════════════════════════════════════════════════════
#  Output validation
# ══════════════════════════════════════════════════════════════
_AGR_DOMAIN_WORDS = re.compile(
    r"\b(crop|soil|pest|disease|fertiliz|spray|seed|plant|irrigat|harvest|"
    r"rice|wheat|tomato|potato|yield|farmer|field|acre|apply|dose|ml|gram|"
    r"urea|fungicide|insecticide|weather|rain|moisture|groundnut|cotton|maize)\b",
    re.I
)

# FIX 5: Hallucination markers from the observed garbage outputs
_HALLUCINATION_MARKERS = re.compile(
    r"\b(NT,|Poultry,|NTESTIN|enetic|sensor reading|sensor data|"
    r"your readings|Temp=|pH=.*match|rainfed water source|"
    r"Farming Awareness|broiler chicken|chick|laying hen)\b",
    re.I
)

# Structural garbage patterns
_GARBAGE_PATTERNS = re.compile(
    r"(enetic\s+(Engineer|Effect|Resource|Ecolog|Primer)|"
    r"Stays:|Stay Crop|Agricultural Stays|NARS\)|"
    r"question:|answer:|###\s*(instruction|input|response))",
    re.I
)


def is_on_domain(text: str) -> bool:
    return bool(_AGR_DOMAIN_WORDS.search(text))


def has_repetition(text: str, ngram: int = 5, threshold: float = 0.4) -> bool:
    words = text.split()
    if len(words) < ngram * 2:
        return False
    grams = [tuple(words[i:i+ngram]) for i in range(len(words)-ngram+1)]
    unique_ratio = len(set(grams)) / len(grams)
    return unique_ratio < (1.0 - threshold)


def clean_output(text: str) -> str:
    if not text:
        return ""
    for m in ["<unk>", "<pad>", "<bos>", "<eos>"]:
        text = text.replace(m, "")
    text = re.sub(r"(?im)^\s*#{1,3}\s*", "", text, flags=re.M)
    text = re.sub(r"(?im)^\s*(response|question|answer)\s*[:\-]?\s*", "", text, flags=re.M)
    text = re.sub(r"\b(question|answer|response)\b[:\-]?", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, max_sent: int = 5) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8]
    return " ".join(sents[:max_sent])


def is_coherent(text: str, min_words: int = 15) -> bool:
    """FIX 6: stricter — min_words 8→15, alpha ratio 0.35→0.50."""
    words = text.split()
    if len(words) < min_words:
        return False
    real = sum(1 for w in words if re.match(r"^[a-zA-Z]{2,}$", w))
    return real / len(words) > 0.50


def has_hallucinated_terms(text: str) -> bool:
    """FIX 5: detect the specific hallucination patterns we observed."""
    return bool(_HALLUCINATION_MARKERS.search(text)) or bool(_GARBAGE_PATTERNS.search(text))


def has_bad_patterns(text: str) -> bool:
    bad = ["question:", "###", "scanl", "unknown chemical"]
    return any(p in text.lower() for p in bad)


def is_relevant(query: str, text: str) -> bool:
    keywords = [w for w in re.findall(r"\b[a-z]{4,}\b", query.lower())]
    answer = text.lower()
    return any(k in answer for k in keywords)


def score_rag_answer(text: str) -> int:
    """
    FIX 5+7: stricter scoring. Passing threshold raised to 5 (was 4).
    Added hallucination check and agriculture domain check.
    """
    score = 0
    if is_coherent(text):           score += 1   # min 15 words, >50% alpha
    if not has_repetition(text):    score += 1
    if len(text.split()) > 12:      score += 1
    if "question" not in text.lower(): score += 1
    if not has_hallucinated_terms(text): score += 1
    if not has_bad_patterns(text):  score += 1
    # FIX 7: must contain at least one agriculture word
    if is_on_domain(text):          score += 1
    return score


def is_valid_rag_answer(text: str) -> bool:
    return score_rag_answer(text) >= 5  # raised from 4


# ══════════════════════════════════════════════════════════════
#  Response Generator
# ══════════════════════════════════════════════════════════════
class ResponseGenerator:
    _FALLBACK = [
        "I need more details to give accurate advice. Please share your crop name, "
        "location, and the specific problem or question.",
        "Could you describe the symptoms or problem in more detail? "
        "Mentioning your crop and location will help me give better advice.",
        "For a precise recommendation, please tell me: (1) your crop, "
        "(2) your location/soil type, and (3) the specific issue you're facing.",
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

    def _slm_generate(self, query: str, context: str = "") -> str:
        """
        FIX 1 + FIX 2:
          - Use build_rag_prompt() so the format matches training exactly.
          - Call tokenizer.encode() with add_bos=False — the prompt already
            has a leading space that was present at training time. Adding BOS
            again shifts all token positions → "enetic..." garbage at output.
          - Lower temperature to 0.4 for more deterministic agriculture answers.
        """
        if self.model is None:
            return ""
        try:
            prompt = build_rag_prompt(
                instruction=query,
                context=context if context else "",
            )
            # FIX 1: encode WITHOUT adding BOS — it was already prepended during
            # fine-tuning in step5 via `tokens = [BOS_ID] + sp.encode(text)`.
            # Double BOS caused the decoder to start at position 1 → garbled output.
            ids = self.tokenizer.encode(prompt, add_bos=False)

            gen_ids = self.model.generate(
                ids,
                max_tokens  = self.max_tokens,
                temperature = 0.4,       # lower = more factual
                top_k       = 40,
                top_p       = 0.88,
                device      = DEVICE,
            )
            return self.tokenizer.decode(gen_ids, skip_special=True)
        except Exception as e:
            print(f"[Generator] SLM error: {e}")
            return ""

    def _summarize_passage(self, query: str, passage: str) -> str:
        passage = re.sub(r"\d+$", "", passage).strip()
        raw     = self._slm_generate(query, passage)
        summary = truncate(clean_output(raw))
        if not summary.strip() or len(summary.split()) < 8:
            return passage
        # FIX 5: reject if hallucinated
        if has_hallucinated_terms(summary):
            return passage
        return summary

    def generate(self, query: str) -> Tuple[str, str]:
        """
        Returns (answer, source_label).
        source_label: "template" | "retrieval_high" | "retrieval_low" | "sagestorm" | "fallback"
        """
        # ── L1: Template (highest confidence) ────────────────
        t = template_answer(query)
        if t:
            if needs_weather(query) or re.search(r"\bspray\b", query, re.I):
                try:
                    loc = self.ctx_builder.memory.get_location() or "Sivakasi"
                    adv = self.ctx_builder.weather_svc.advisory(loc)
                    t   = f"{t}\n\n[Weather advisory] {adv}"
                except Exception:
                    pass
            return t, "template"

        # ── L2 / L3: Retrieval ────────────────────────────────
        retrieved  = self.ctx_builder.get_retrieved(query)
        passage, confidence = best_passage(retrieved, query)

        if passage and confidence >= 0.25:
            context, ctx_conf = self.ctx_builder.build_context_str(query)
            raw     = self._slm_generate(query, context)
            cleaned = truncate(clean_output(raw))
            crop    = self.ctx_builder.memory.get_crop() or ""

            score = score_rag_answer(cleaned)
            # FIX 5: threshold raised to 5, plus hallucination + domain guards
            if (score >= 5
                    and is_relevant(query, cleaned)
                    and len(cleaned.split()) > 12
                    and not cleaned.startswith("N ")
                    and cleaned
                    and not cleaned[0].islower()
                    and (not crop or crop.lower() in cleaned.lower())):
                if needs_weather(query):
                    try:
                        loc = self.ctx_builder.memory.get_location() or "Sivakasi"
                        adv = self.ctx_builder.weather_svc.advisory(loc)
                        cleaned = f"{cleaned}\n\n[Weather advisory] {adv}"
                    except Exception:
                        pass
                return cleaned, "retrieval"

            # Fall back to the passage itself
            summary = self._summarize_passage(query, passage)
            if is_coherent(summary) and is_on_domain(summary) and not has_hallucinated_terms(summary):
                return summary, "retrieval"
            # Return raw passage truncated
            result = truncate(passage, max_sent=4)
            if needs_weather(query):
                try:
                    loc = self.ctx_builder.memory.get_location() or "Sivakasi"
                    adv = self.ctx_builder.weather_svc.advisory(loc)
                    result = f"{result}\n\n[Weather advisory] {adv}"
                except Exception:
                    pass
            return result, "retrieval"

        if passage and confidence >= 0.15:
            summary = self._summarize_passage(query, passage)
            score   = score_rag_answer(summary)
            crop    = self.ctx_builder.memory.get_crop() or ""
            if (score >= 5
                    and is_relevant(query, summary)
                    and len(summary.split()) > 12
                    and not summary.startswith("N ")
                    and summary
                    and not summary[0].islower()
                    and (not crop or crop.lower() in summary.lower())):
                return summary, "retrieval"
            # Return passage directly (safe fallback)
            result = truncate(passage, max_sent=4)
            return result, "retrieval"

        # ── L4: SageStorm generation ──────────────────────────
        context, ctx_conf = self.ctx_builder.build_context_str(query)
        if ctx_conf < 0.25:
            context = ""
        raw     = self._slm_generate(query, context)
        cleaned = truncate(clean_output(raw))

        score = score_rag_answer(cleaned)
        if score >= 5:
            return cleaned, "sagestorm"

        # If we have any passage, prefer it over garbage SLM output
        if passage:
            result = truncate(passage, max_sent=4)
            return result, "retrieval"

        # ── L5: Fallback ──────────────────────────────────────
        msg    = self._FALLBACK[self._fi % len(self._FALLBACK)]
        self._fi += 1
        return msg, "fallback"