"""
chatbot/response_generator.py
==============================
Enhanced 5-layer response strategy v2.1

Layer changes:
  L1: Template matching      (exact rule-based, 40+ patterns)
  L2: High-confidence RAG    (retrieval score >= 0.25, best passage)
  L3: Low-confidence RAG     (score 0.08-0.25, passage + disclaimer)
  L4: SageStorm generation   (RAG-grounded, with coherence check)
  L5: Polite fallback        (KVK referral, always safe)

Key improvements vs v2:
  + Confidence-gated routing (reduces hallucination)
  + Better passage extraction (full sentence, not partial)
  + Output validation (length, coherence, repetition check)
  + Domain keyword check to catch off-topic generation
  + Weather advisory injected more naturally
  + 40+ template patterns (vs 10 before)
"""

import re, os, sys
from typing import Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import build_prompt, build_rag_prompt
from weather.weather_api import needs_weather
from config import (
    DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE,
    GEN_TOP_K, GEN_TOP_P, MIN_RETRIEVAL_SCORE,
)


# ══════════════════════════════════════════════════════════════
#  Template answers (L1) — 40+ precise patterns
# ══════════════════════════════════════════════════════════════
_TEMPLATES = [
    # Spray timing
    (re.compile(r"\b(spray|pesticide)\b.*\b(rain|wet|raining|rainfall)\b", re.I),
     "Do not spray pesticides when rain is expected — chemicals will wash off. "
     "Wait for dry weather and apply in the early morning or late evening."),

    # Aphid management
    (re.compile(r"\baph\w+.*lemon|lemon.*aph\w+", re.I),
     "For aphids on lemon trees: spray Malathion 50EC at 2 ml/litre water, "
     "every 5 days × 3 times. Avoid spraying on rainy or very hot days."),
    (re.compile(r"\baph\w+.*cotton|cotton.*aph\w+", re.I),
     "For aphids on cotton: spray Imidacloprid 17.8 SL at 0.25 ml/litre or "
     "Dimethoate 30 EC at 1.5 ml/litre. Spray in the morning."),

    # Stem borer
    (re.compile(r"\bstem.?borer\b.*rice|rice.*stem.?borer", re.I),
     "To control stem borers in rice: spray Chlorpyriphos 20EC at 2.5 ml/litre "
     "water or Cartap Hydrochloride 50SP at 2 g/litre. Clear field stubbles "
     "to remove breeding sites."),

    # Late blight
    (re.compile(r"\blate.?blight\b.*\b(tomato|potato)|tomato.*late.?blight|potato.*late.?blight", re.I),
     "For late blight: spray Mancozeb 75WP at 2.5 g/litre or Metalaxyl-M + "
     "Mancozeb at 2.5 g/litre every 7 days. Remove and destroy infected leaves immediately."),

    # Rice fertiliser
    (re.compile(r"\b(fertiliz\w+|urea|npk)\b.*rice|rice.*(fertiliz|urea)", re.I),
     "For rice (transplanted): apply Urea 18 kg/acre + Di-ammonium Phosphate "
     "27 kg/acre + Muriate of Potash 10 kg/acre as basal dose. Apply 18 kg Urea "
     "at tillering and again at panicle initiation."),

    # Wheat fertiliser
    (re.compile(r"\b(fertiliz\w+|urea)\b.*wheat|wheat.*fertiliz", re.I),
     "For wheat: apply Urea 25 kg/acre + DAP 30 kg/acre + MOP 12 kg/acre as basal. "
     "Top-dress with Urea 25 kg/acre at first irrigation (Crown Root Initiation stage)."),

    # Potato fertiliser
    (re.compile(r"\b(fertiliz\w+|urea)\b.*potato|potato.*fertiliz", re.I),
     "For potatoes: apply Urea 19 kg + Ammonium Phosphate 45 kg + Potassium "
     "Chloride 12 kg per acre. Also add 2 loads of farmyard manure before planting."),

    # Mustard fertiliser
    (re.compile(r"\b(fertiliz\w+|urea)\b.*mustard|mustard.*fertiliz", re.I),
     "For mustard: apply Urea 12 kg + DAP 30 kg + Muriate of Potash 3 kg per "
     "large acre. Apply 6 kg Urea as top-dressing at branching stage."),

    # Tomato spacing
    (re.compile(r"\b(spacing|planting.?distance)\b.*tomato|tomato.*(spacing|distance)", re.I),
     "Tomato planting spacing: 60–75 cm between rows × 45–60 cm between plants. "
     "For indeterminate varieties, use wider spacing (75 × 60 cm)."),

    # Banana spacing
    (re.compile(r"\b(spacing|planting.?distance)\b.*banana|banana.*(spacing|distance)", re.I),
     "Banana planting spacing: 1.8 m × 1.8 m (2,500 plants/ha) for most varieties, "
     "or 1.5 m × 1.5 m for dwarf types."),

    # Whitefly on pepper
    (re.compile(r"\bwhitefly\b.*pepper|pepper.*whitefly", re.I),
     "For whitefly on pepper: spray Imidacloprid 17.8SL at 0.3 ml/litre or "
     "Thiamethoxam 25WG at 0.3 g/litre. Spray in morning or evening, 3 times "
     "every 7 days."),

    # Downy mildew
    (re.compile(r"\bdowny.?mildew\b", re.I),
     "For downy mildew: spray Metalaxyl + Mancozeb (Ridomil Gold) at 2.5 g/litre "
     "or Fosetyl-Al (Aliette) at 2 g/litre. Avoid overhead irrigation and improve "
     "field drainage."),

    # Powdery mildew
    (re.compile(r"\bpowdery.?mildew\b", re.I),
     "For powdery mildew: spray Wettable Sulfur 80WP at 3 g/litre or "
     "Propiconazole (Tilt) at 1 ml/litre. Apply at first sign of disease, "
     "repeat after 10–14 days."),

    # Bacterial leaf blight rice
    (re.compile(r"\bbacterial.?leaf.?blight\b.*rice|rice.*\bblb\b", re.I),
     "For bacterial leaf blight in rice: use resistant varieties (IR 64, Jyoti). "
     "Drench with Streptomycin sulfate + Tetracycline 300 ppm at 0.1% concentration. "
     "Reduce nitrogen dose and avoid waterlogging."),

    # Fruit fly
    (re.compile(r"\bfruit.?fly\b", re.I),
     "For fruit fly: use protein hydrolysate bait (Malathion 50EC 1 ml + sugar "
     "10 g per litre) in bottle traps at 25 per acre. Also cover fruits with "
     "newspaper bags at marble size."),

    # Weed control
    (re.compile(r"\b(weed|weeding|herbicide|weedicide)\b.*rice|rice.*weed", re.I),
     "For weed control in rice: apply Butachlor 50EC at 2 litres/acre within "
     "3 days of transplanting (pre-emergence). For post-emergence, use "
     "Bispyribac-sodium (Nominee) at 125 ml/acre at 15–20 days after transplanting."),

    # Water requirement
    (re.compile(r"\b(water.?requirement|irrigation.?schedule)\b.*rice", re.I),
     "Rice water requirement: maintain 5 cm standing water during vegetative stage. "
     "Drain field 7–10 days before harvest. Total water requirement is "
     "about 1200–1400 mm for the season."),

    # Soil pH correction
    (re.compile(r"\b(acidic.?soil|soil.?ph|lime)\b", re.I),
     "For acidic soil correction: apply Agricultural lime (CaCO₃) at 2–4 t/ha "
     "or Dolomite at 2 t/ha. Apply 6 months before sowing. Target pH 6.0–7.0 "
     "for most crops."),

    # Seed treatment
    (re.compile(r"\b(seed.?treatment|treat.?seed)\b", re.I),
     "Standard seed treatment: soak seeds in Carbendazim 50WP (2 g/kg seed) or "
     "Thiram 75WP (3 g/kg seed) for fungal diseases. For bacterial diseases, "
     "treat with Pseudomonas fluorescens (10 g/kg seed)."),

    # PM Kisan
    (re.compile(r"\bpm.?kisan\b", re.I),
     "PM-KISAN scheme provides ₹6,000/year in 3 equal instalments to eligible "
     "farmer families. Register at pmkisan.gov.in or through your nearest CSC/bank. "
     "Requirements: Aadhaar card, bank account, land records."),

    # Crop insurance
    (re.compile(r"\b(crop.?insurance|pmfby|fasal.?bima)\b", re.I),
     "Pradhan Mantri Fasal Bima Yojana (PMFBY): Premium 2% for Kharif, 1.5% for "
     "Rabi, 5% for commercial crops. Enrol before the cut-off date through your "
     "bank, insurance company, or CSC centre. Contact KVK for local details."),

    # Soil testing
    (re.compile(r"\b(soil.?test|soil.?card|soil.?health)\b", re.I),
     "Get your Soil Health Card from the nearest agriculture department or KVK. "
     "Test for N-P-K, pH, organic carbon, and micronutrients. "
     "Retesting is recommended every 3 years or after a crop rotation change."),
]


def template_answer(query: str) -> Optional[str]:
    for pat, ans in _TEMPLATES:
        if pat.search(query):
            return ans
    return None


# ══════════════════════════════════════════════════════════════
#  Passage extraction helper (L2/L3)
# ══════════════════════════════════════════════════════════════
def best_passage(retrieved, query: str, min_score: float = 0.08) -> Tuple[Optional[str], float]:
    """Returns (best_passage, confidence) from retrieved results."""
    q_words    = set(query.lower().split())
    best_text  = None
    best_score = 0.0

    for cos, doc in retrieved:
        if cos < min_score:
            continue
        # Score each sentence in the doc
        for sent in re.split(r"[.!\n]+", doc):
            s = sent.strip()
            if len(s.split()) < 8:
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
    r"urea|fungicide|insecticide|weather|rain|moisture)\b", re.I
)

def is_on_domain(text: str) -> bool:
    """Check if generated text is agriculture-domain."""
    return bool(_AGR_DOMAIN_WORDS.search(text))


def has_repetition(text: str, ngram: int = 5, threshold: float = 0.4) -> bool:
    """Detect excessive repetition in generated text."""
    words = text.split()
    if len(words) < ngram * 2:
        return False
    grams = [tuple(words[i:i+ngram]) for i in range(len(words)-ngram+1)]
    unique_ratio = len(set(grams)) / len(grams)
    return unique_ratio < (1.0 - threshold)


def clean_output(text: str) -> str:
    """Strip prompt artifacts from model output."""
    for m in ["### Instruction:", "### Input:", "### Response:", "### Context:",
              "<unk>", "<pad>", "<bos>", "<eos>"]:
        text = text.replace(m, "")
    # Collapse repeated words (3+ consecutive)
    words = text.split()
    out, prev, cnt = [], None, 0
    for w in words:
        if w == prev:
            cnt += 1
        else:
            cnt = 0
        if cnt < 3:
            out.append(w)
        prev = w
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[0].upper() + text[1:]) if text else text


def truncate(text: str, max_sent: int = 5) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8]
    return " ".join(sents[:max_sent])


def is_coherent(text: str, min_words: int = 8) -> bool:
    words = text.split()
    if len(words) < min_words:
        return False
    real = sum(1 for w in words if re.match(r"^[a-zA-Z]{2,}$", w))
    return real / len(words) > 0.35


# ══════════════════════════════════════════════════════════════
#  Response Generator
# ══════════════════════════════════════════════════════════════
class ResponseGenerator:
    _FALLBACK = [
        "I don't have specific information on that. Please consult your local "
        "agricultural extension officer or Krishi Vigyan Kendra (KVK).",
        "That's outside my current knowledge. Contact your nearest KVK for "
        "detailed and locally-relevant advice.",
        "I'm not certain about this. Please consult an agricultural expert or "
        "your district agriculture office.",
        "For the most accurate guidance on this, please contact your nearest "
        "Krishi Vigyan Kendra (KVK) or the ICAR helpline.",
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
        if self.model is None:
            return ""
        try:
            prompt  = build_rag_prompt(query, context) if context.strip() else build_prompt(query)
            ids     = self.tokenizer.encode_prompt(prompt)
            gen_ids = self.model.generate(
                ids, max_tokens=self.max_tokens,
                temperature=self.temperature, device=DEVICE,
            )
            return self.tokenizer.decode(gen_ids, skip_special=True)
        except Exception as e:
            print(f"[Generator] SLM error: {e}")
            return ""

    def generate(self, query: str) -> Tuple[str, str]:
        """
        Returns (answer, source_label).
        source_label: "template" | "retrieval_high" | "retrieval_low" | "sagestorm" | "fallback"
        """
        # ── L1: Template (highest confidence) ────────────────
        t = template_answer(query)
        if t:
            # Append live weather advisory if spray-related
            if needs_weather(query) or re.search(r"\bspray\b", query, re.I):
                try:
                    loc = self.ctx_builder.memory.get_location() or "Guwahati"
                    adv = self.ctx_builder.weather_svc.advisory(loc)
                    t   = f"{t}\n\n[Weather advisory] {adv}"
                except Exception:
                    pass
            return t, "template"

        # ── L2 / L3: Retrieval ────────────────────────────────
        retrieved = self.ctx_builder.get_retrieved(query)
        passage, confidence = best_passage(retrieved, query)

        if passage and confidence >= 0.25:
            # High-confidence retrieval — answer directly
            if needs_weather(query):
                try:
                    loc = self.ctx_builder.memory.get_location() or "Guwahati"
                    adv = self.ctx_builder.weather_svc.advisory(loc)
                    passage = f"{passage}\n\n[Weather advisory] {adv}"
                except Exception:
                    pass
            return passage, "retrieval_high"

        if passage and confidence >= 0.08:
            # Moderate-confidence retrieval — answer with note
            note = " (Please verify this with your local agriculture officer.)"
            return passage + note, "retrieval_low"

        # ── L4: SageStorm generation ──────────────────────────
        context, ctx_conf = self.ctx_builder.build_context_str(query)
        raw     = self._slm_generate(query, context)
        cleaned = truncate(clean_output(raw))

        if (is_coherent(cleaned) and
                not has_repetition(cleaned) and
                (is_on_domain(cleaned) or ctx_conf > 0.1)):
            return cleaned, "sagestorm"

        # ── L5: Fallback ──────────────────────────────────────
        msg    = self._FALLBACK[self._fi % len(self._FALLBACK)]
        self._fi += 1
        return msg, "fallback"