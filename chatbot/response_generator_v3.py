"""
chatbot/response_generator_v3.py
=================================
SageStorm V3 Response Generator
================================
Upgrades the existing 5-layer pipeline with a new LAYER 0:

  Layer 0: Advisory detection → ML predict + SLM narration  (NEW)
  Layer 1: Template matching
  Layer 2: High-confidence retrieval
  Layer 3: Low-confidence retrieval
  Layer 4: SageStorm generation
  Layer 5: Fallback

Layer 0 activates when:
  - User provides soil/NPK/environmental parameters
  - User explicitly asks "advisory" or "recommend a crop"
  - Structured field data is detected in the message

When Layer 0 fires:
  1. AgriPredictor runs ML inference on the parsed inputs
  2. RAG retrieves crop-specific knowledge
  3. SageStorm generates a full advisory using the predictions as FACTS
  4. The advisory covers all 10 sections (sowing → post-harvest)

This means the SLM is not guessing — it's EXPLAINING verified ML predictions.
"""

import re
import os
import sys
from typing import Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import original response generator for fallback layers
from chatbot.response_generator import (
    ResponseGenerator, template_answer, best_passage,
    is_on_domain, has_repetition, clean_output, truncate,
    is_coherent,
)
from chatbot.advisory_engine import (
    AdvisoryEngine, AgriPredictor,
    is_advisory_query, parse_advisory_inputs,
)
from models.tokenizer import build_prompt, build_rag_prompt
from weather.weather_api import needs_weather
from config import DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE


# ── Advisory detection patterns ───────────────────────────────
_ADVISORY_PATTERNS = re.compile(
    r"""
    \b(
        soil\s+type | soil\s+ph | ph\s*[=:] | n\s*[=:]\s*\d | p\s*[=:]\s*\d |
        k\s*[=:]\s*\d | nitrogen | phosphorus | potassium |
        npk | kharif | rabi | zaid | advisory | recommend\s+crop |
        which\s+crop | what\s+crop | crop\s+recommendation |
        \d+\s*acres? | \d+\s*hectares? |
        loamy | alluvial | laterite |
        fertilizer\s+for\s+my | pest\s+management\s+for |
        sowing.*harvest | complete.*advisory | full.*advisory
    )\b
    """,
    re.I | re.VERBOSE,
)

_PARAM_PATTERN = re.compile(
    r"(?:N|P|K|pH|temp(?:erature)?|humidity|rainfall)\s*[=:]\s*[\d.]+",
    re.I
)


def needs_advisory_pipeline(query: str) -> bool:
    """
    True when the query contains field parameters and is asking
    for crop/farming advice — not just a simple pest question.
    """
    has_advisory_kw = bool(_ADVISORY_PATTERNS.search(query))
    has_param = len(_PARAM_PATTERN.findall(query)) >= 2
    has_ml_ask = bool(re.search(
        r"\b(advisory|recommend|suggest|plan|guide|complete|full|sowing|harvest)\b",
        query, re.I
    ))
    return (has_advisory_kw and (has_param or has_ml_ask)) or (has_param and has_ml_ask)


# ════════════════════════════════════════════════════════════
#  ResponseGeneratorV3
# ════════════════════════════════════════════════════════════
class ResponseGeneratorV3(ResponseGenerator):
    """
    Drop-in replacement for ResponseGenerator.
    Adds Layer 0: ML-backed advisory generation.

    Usage (in main.py / api/server.py):
        from chatbot.response_generator_v3 import ResponseGeneratorV3
        gen = ResponseGeneratorV3(model, tok, ret, ctx, predictor=predictor)
    """

    def __init__(self, model, tokenizer, retriever, ctx_builder,
                 predictor: AgriPredictor = None,
                 max_tokens: int = GEN_MAX_TOKENS,
                 temperature: float = GEN_TEMPERATURE):
        super().__init__(model, tokenizer, retriever, ctx_builder,
                         max_tokens=max_tokens, temperature=temperature)

        # Advisory engine wraps ML + RAG + SLM
        self.advisory_engine = AdvisoryEngine(
            slm_model   = model,
            tokenizer   = tokenizer,
            retriever   = retriever,
            ctx_builder = ctx_builder,
            predictor   = predictor or AgriPredictor(),
            max_tokens  = max_tokens,
            temperature = temperature,
        )

        # Keep last advisory result for multi-turn follow-up
        self._last_advisory: Optional[Dict] = None

    # ── Main entry point ─────────────────────────────────────
    def generate(self, query: str) -> Tuple[str, str]:
        """
        Returns (answer_text, source_label).

        source_label: "advisory_ml" | "template" | "retrieval" | "sagestorm" | "fallback"
        """

        # ── Layer 0: ML Advisory Pipeline ───────────────────
        if needs_advisory_pipeline(query) or self._is_followup_advisory(query):
            return self._generate_advisory(query)

        # ── Layers 1–5: Original pipeline ───────────────────
        return super().generate(query)

    def _is_followup_advisory(self, query: str) -> bool:
        """
        Detect follow-up questions about a previous advisory.
        e.g. "explain the fertilizer part more" after an advisory.
        """
        if self._last_advisory is None:
            return False
        followup_kw = re.compile(
            r"\b(explain|more about|what about|tell me about|elaborate|"
            r"why|how much|when to|which|dose|spacing|timing)\b",
            re.I
        )
        return bool(followup_kw.search(query))

    def _generate_advisory(self, query: str) -> Tuple[str, str]:
        """Route to advisory engine."""
        # Try to parse field parameters from the query
        parsed = parse_advisory_inputs(query)

        # If we have a previous advisory and this is a follow-up,
        # reuse the previous inputs but enrich with any new info
        if parsed is None and self._last_advisory is not None:
            parsed = self._last_advisory.get("predictions", {}).get("inputs", {})

        if parsed is None or len(parsed) < 3:
            # Not enough data — ask for it
            return (
                "To give you a complete farming advisory, I need your field details.\n\n"
                "Please share:\n"
                "• Soil type (Clay/Loamy/Sandy/Silt/Black/Red/Alluvial)\n"
                "• Soil pH\n"
                "• Nitrogen (N), Phosphorus (P), Potassium (K) levels in kg/ha\n"
                "• Temperature, Humidity, Rainfall\n"
                "• Season (Kharif/Rabi/Zaid)\n\n"
                "Example: 'My soil is Clay, pH 6.5, N=90, P=40, K=50, "
                "temperature 28°C, humidity 70%, rainfall 120mm, Kharif season.'",
                "fallback"
            )

        # Generate full advisory
        result = self.advisory_engine.generate_advisory(parsed)
        self._last_advisory = result

        advisory_text = result["advisory_text"]
        source = "advisory_ml"

        # Prepend a brief prediction summary banner
        pred = result["predictions"]
        banner = (
            f"[ML Prediction: {pred['crop']} ({pred['crop_confidence']}% confidence) · "
            f"Fertilizer: {pred['fertilizer']} · "
            f"Pest risk: {pred.get('pesticide_target', 'N/A')}]\n\n"
        )

        return banner + advisory_text, source

    def get_last_advisory(self) -> Optional[Dict]:
        """Return the full structured result of the last advisory generation."""
        return self._last_advisory

    def get_predictions(self) -> Optional[Dict]:
        """Return only the ML predictions from the last advisory."""
        if self._last_advisory:
            return self._last_advisory.get("predictions")
        return None