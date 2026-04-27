"""
pipeline/advisory_pipeline.py
================================
Main Advisory Pipeline Orchestrator

Implements the full architecture:

  User Input
     ↓
  ContextBuilder (crop, soil, weather, NPK, stage)
     ↓
  ──────────────────────────────────────────
  |  CropModel  |  FertilizerModel  |  PesticideModel  |  (parallel)
  ──────────────────────────────────────────
     ↓
  AggregationLayer (validate + merge + conflict-resolve)
     ↓
  LLM Explanation Layer (SageStorm V2 or API)
     ↓
  Final Advisory Report

Usage:
  from pipeline.advisory_pipeline import AdvisoryPipeline
  pipeline = AdvisoryPipeline()
  result = pipeline.run({
      "query": "What should I grow this season?",
      "location": "Guwahati",
      "soil_type": "loamy",
      "season": "kharif",
      "temperature": 32,
      "humidity": 75,
      "nitrogen": 80, "phosphorus": 40, "potassium": 30,
  })
  print(result["advisory"])
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_models.crop_model       import CropModel
from decision_models.fertilizer_model import FertilizerModel
from decision_models.pesticide_model  import PesticideModel
from pipeline.aggregation             import AggregationLayer
from prompts.advisory_prompt          import SYSTEM_PROMPT, build_advisory_prompt, build_chat_prompt


# ── LLM backend options ───────────────────────────────────────
LLM_BACKEND = os.environ.get("STROM_SAGE_LLM", "sagestorm")
# Options: "sagestorm" | "anthropic" | "openai" | "local_llama"


class AdvisoryPipeline:
    """
    End-to-end advisory pipeline.
    Runs the three decision models in parallel for minimum latency,
    then aggregates and generates the LLM explanation.
    """

    def __init__(
        self,
        llm_backend: str = LLM_BACKEND,
        sagestorm_generator=None,   # inject existing ResponseGenerator
        weather_service=None,       # inject existing WeatherService
    ):
        self.crop_model       = CropModel()
        self.fertilizer_model = FertilizerModel()
        self.pesticide_model  = PesticideModel()
        self.aggregator       = AggregationLayer()
        self.llm_backend      = llm_backend
        self.gen              = sagestorm_generator    # SageStorm ResponseGenerator
        self.weather_svc      = weather_service

        print(f"[Pipeline] AdvisoryPipeline ready — LLM backend: {llm_backend}")

    # ── Step 1: Context extraction ────────────────────────────
    def _extract_context(self, query: str, raw_input: Dict) -> Dict:
        """
        Normalise and validate the input context.
        Extracts structured fields; fills missing ones with sensible defaults.
        """
        import re

        ctx = {
            "query":       query,
            "location":    raw_input.get("location",    ""),
            "soil_type":   raw_input.get("soil_type",   ""),
            "season":      raw_input.get("season",      ""),
            "temperature": float(raw_input.get("temperature", 28)),
            "humidity":    float(raw_input.get("humidity",    65)),
            "nitrogen":    float(raw_input.get("nitrogen",    50)),
            "phosphorus":  float(raw_input.get("phosphorus",  30)),
            "potassium":   float(raw_input.get("potassium",   30)),
            "growth_stage": raw_input.get("growth_stage", ""),
            "symptoms":    raw_input.get("symptoms",    ""),
            "crop_name":   raw_input.get("crop_name",   ""),   # if user specifies
        }

        # Auto-detect season from temperature if not provided
        if not ctx["season"]:
            temp = ctx["temperature"]
            month = raw_input.get("month", 0)
            if 6 <= month <= 10 or temp >= 28:
                ctx["season"] = "kharif"
            elif 11 <= month <= 3 or temp <= 20:
                ctx["season"] = "rabi"
            else:
                ctx["season"] = "zaid"

        return ctx

    # ── Step 2: Run models in parallel ────────────────────────
    def _run_models_parallel(self, ctx: Dict, weather: Optional[Dict]):
        """Run CropModel, FertilizerModel, PesticideModel concurrently."""

        def run_crop():
            return self.crop_model.predict({
                "soil_type":   ctx["soil_type"],
                "season":      ctx["season"],
                "temperature": ctx["temperature"],
                "nitrogen":    ctx["nitrogen"],
                "phosphorus":  ctx["phosphorus"],
                "potassium":   ctx["potassium"],
            })

        def run_fertilizer(crop_name: str = ""):
            return self.fertilizer_model.predict({
                "crop_name":    crop_name or ctx.get("crop_name", ""),
                "soil_type":    ctx["soil_type"],
                "season":       ctx["season"],
                "nitrogen":     ctx["nitrogen"],
                "phosphorus":   ctx["phosphorus"],
                "potassium":    ctx["potassium"],
                "growth_stage": ctx["growth_stage"],
            })

        def run_pesticide(crop_name: str = ""):
            return self.pesticide_model.predict({
                "crop_name":   crop_name or ctx.get("crop_name", ""),
                "season":      ctx["season"],
                "temperature": ctx["temperature"],
                "humidity":    ctx["humidity"],
                "symptoms":    ctx["symptoms"],
                "growth_stage": ctx["growth_stage"],
            })

        # If crop is pre-specified, use it directly
        if ctx.get("crop_name"):
            crop_pred = run_crop()
            crop_pred.crop_name = ctx["crop_name"]  # override with user's crop
        else:
            crop_pred = run_crop()

        # Now run fertilizer and pesticide with the resolved crop — in parallel
        with ThreadPoolExecutor(max_workers=2) as ex:
            fert_future = ex.submit(run_fertilizer, crop_pred.crop_name)
            pest_future = ex.submit(run_pesticide,  crop_pred.crop_name)
            fert_pred = fert_future.result()
            pest_pred = pest_future.result()

        return crop_pred, fert_pred, pest_pred

    # ── Step 3: Get weather ───────────────────────────────────
    def _get_weather(self, location: str) -> Optional[Dict]:
        if self.weather_svc is None:
            return None
        try:
            return self.weather_svc.get(location or None)
        except Exception:
            return None

    # ── Step 4: LLM explanation ───────────────────────────────
    def _explain(self, advisory_prompt: str) -> Tuple[str, str]:
        """
        Call the selected LLM backend to generate the explanation.
        Returns (explanation_text, source_label)
        """
        if self.llm_backend == "sagestorm" and self.gen is not None:
            try:
                answer, source = self.gen.generate(advisory_prompt)
                return answer, source
            except Exception as e:
                print(f"[Pipeline] SageStorm error: {e}")

        if self.llm_backend == "anthropic":
            return self._explain_anthropic(advisory_prompt)

        # Last resort: return the structured prompt itself as the answer
        # (the aggregated data is already readable without LLM polish)
        return self._format_fallback(advisory_prompt), "structured_fallback"

    def _explain_anthropic(self, prompt: str) -> Tuple[str, str]:
        """Call Anthropic API (requires ANTHROPIC_API_KEY env var)."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text, "anthropic_claude"
        except ImportError:
            print("[Pipeline] anthropic library not installed — pip install anthropic")
        except Exception as e:
            print(f"[Pipeline] Anthropic API error: {e}")
        return self._format_fallback(prompt), "structured_fallback"

    def _format_fallback(self, prompt: str) -> str:
        """
        When no LLM is available, return a clean structured report
        by parsing the prompt sections directly.
        This ensures the pipeline ALWAYS returns useful output.
        """
        # Extract data sections from the prompt
        lines = prompt.strip().split("\n")
        output = []
        in_data = False
        for line in lines:
            if "MODEL OUTPUTS" in line:
                in_data = True
                continue
            if "TASK:" in line or "OUTPUT FORMAT" in line:
                break
            if in_data and line.strip():
                output.append(line)
        return "\n".join(output) or "Advisory data available. LLM explanation unavailable."

    # ── Main entry point ──────────────────────────────────────
    def run(self, raw_input: Dict) -> Dict:
        """
        Run the full pipeline and return the advisory.

        Returns dict with keys:
          advisory       : str   (full formatted report)
          source         : str   (which LLM backend was used)
          crop           : dict  (CropPrediction.to_dict())
          fertilizer     : dict
          pesticide      : dict
          aggregated     : dict  (AggregatedAdvisory.to_prompt_dict())
          latency_ms     : float
          warnings       : list[str]
        """
        t0 = time.perf_counter()

        query = raw_input.get("query", "") or raw_input.get("question", "")
        ctx   = self._extract_context(query, raw_input)

        # Get weather
        weather = self._get_weather(ctx["location"])
        if weather:
            ctx["temperature"] = float(weather.get("temp",  ctx["temperature"]))
            ctx["humidity"]    = float(weather.get("humid", ctx["humidity"]))

        # Run models in parallel
        crop_pred, fert_pred, pest_pred = self._run_models_parallel(ctx, weather)

        # Aggregate
        advisory_obj = self.aggregator.aggregate(
            crop_pred, fert_pred, pest_pred, ctx, weather
        )

        # Build prompt dict
        prompt_dict = advisory_obj.to_prompt_dict()

        # Generate LLM explanation
        try:
            llm_prompt = build_advisory_prompt(prompt_dict)
            explanation, source = self._explain(llm_prompt)
        except Exception as e:
            print(f"[Pipeline] Prompt build error: {e}")
            explanation = str(advisory_obj.to_prompt_dict())
            source = "fallback"

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "advisory":    explanation,
            "source":      source,
            "crop":        crop_pred.to_dict(),
            "fertilizer":  fert_pred.to_dict(),
            "pesticide":   pest_pred.to_dict(),
            "aggregated":  prompt_dict,
            "latency_ms":  latency_ms,
            "warnings":    advisory_obj.warnings,
        }

    # ── Conversational mode (Q&A) ─────────────────────────────
    def chat(self, question: str, farmer_context: str = "") -> Tuple[str, str]:
        """
        Simple chat mode — bypasses the decision models.
        Uses the LLM directly with farmer context.
        Returns (answer, source).
        """
        prompt = build_chat_prompt(farmer_context, question)
        return self._explain(prompt)


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    pipeline = AdvisoryPipeline(llm_backend="structured_fallback")

    result = pipeline.run({
        "query":       "What should I grow this kharif season?",
        "location":    "Guwahati, Assam",
        "soil_type":   "loamy",
        "season":      "kharif",
        "temperature": 32,
        "humidity":    80,
        "nitrogen":    85,
        "phosphorus":  40,
        "potassium":   35,
    })

    print("=== ADVISORY ===")
    print(result["advisory"])
    print(f"\n=== Meta ===")
    print(f"Source    : {result['source']}")
    print(f"Latency   : {result['latency_ms']} ms")
    print(f"Crop      : {result['crop']['crop_name']} ({result['crop']['confidence']*100:.0f}%)")
    print(f"Fertilizer: {result['fertilizer']['fertilizer_name']}")
    print(f"Pest      : {result['pesticide']['pest_name']}")
    print(f"Warnings  : {result['warnings']}")
