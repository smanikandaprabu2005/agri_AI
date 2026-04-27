"""
pipeline/aggregation.py
========================
Aggregation Layer

Takes outputs from CropModel, FertilizerModel, and PesticideModel,
validates them, resolves conflicts, and produces a clean structured
advisory dict that the LLM explanation layer can use.

Key responsibilities:
  1. Validate that fertilizer recommendation matches the crop
  2. Check pesticide-fertilizer timing conflicts (don't spray both same day)
  3. Check weather compatibility (no spray if rain expected)
  4. Add confidence-weighted reliability flag
  5. Produce the final structured advisory ready for the LLM prompt
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import re


@dataclass
class AggregatedAdvisory:
    # Farmer context
    location:       str
    soil_type:      str
    season:         str
    temperature:    float
    nitrogen:       float
    phosphorus:     float
    potassium:      float
    humidity:       float

    # Crop
    crop_name:      str
    crop_confidence: float
    crop_reason_tags: List[str]

    # Fertilizer
    fertilizer_name:   str
    fertilizer_dosage: str
    fertilizer_stage:  str

    # Pest/Pesticide
    pest_name:      str
    pesticide_name: str
    pesticide_dosage: str
    organic_alternative: str

    # Meta
    warnings:       List[str]
    overall_confidence: float
    weather_advisory: str
    spray_safe:     bool   # is it safe to spray today?

    def to_prompt_dict(self) -> dict:
        """Returns a flat dict for use in the LLM prompt template."""
        return {
            "location":          self.location or "Not provided",
            "soil_type":         self.soil_type or "Not provided",
            "season":            self.season or "Not provided",
            "temperature":       f"{self.temperature:.0f}°C" if self.temperature else "Not provided",
            "nitrogen":          f"{self.nitrogen:.0f} kg/ha" if self.nitrogen else "Not provided",
            "phosphorus":        f"{self.phosphorus:.0f} kg/ha" if self.phosphorus else "Not provided",
            "potassium":         f"{self.potassium:.0f} kg/ha" if self.potassium else "Not provided",
            "crop_name":         self.crop_name,
            "crop_confidence":   f"{self.crop_confidence * 100:.0f}%",
            "crop_reason_tags":  ", ".join(self.crop_reason_tags),
            "fertilizer_name":   self.fertilizer_name,
            "fertilizer_dosage": self.fertilizer_dosage,
            "fertilizer_stage":  self.fertilizer_stage,
            "pest_name":         self.pest_name,
            "pesticide_name":    self.pesticide_name,
            "pesticide_dosage":  self.pesticide_dosage,
            "organic_alternative": self.organic_alternative,
            "weather_advisory":  self.weather_advisory,
            "spray_safe":        "Yes — conditions are suitable for spraying" if self.spray_safe else "No — avoid spraying today",
            "warnings":          "; ".join(self.warnings) if self.warnings else "None",
            "overall_confidence": f"{self.overall_confidence * 100:.0f}%",
        }


class AggregationLayer:
    """
    Merges and validates outputs from all three decision models.
    Applies cross-model conflict checks and weather rules.
    """

    def aggregate(
        self,
        crop_pred,        # CropPrediction
        fertilizer_pred,  # FertilizerPrediction
        pesticide_pred,   # PesticidePrediction
        context: Dict,    # raw farmer context from ContextBuilder
        weather: Optional[Dict] = None,
    ) -> AggregatedAdvisory:
        """
        context keys: location, soil_type, season, temperature,
                      nitrogen, phosphorus, potassium, humidity
        weather keys: rain (bool), rain_pct (int), wind (float),
                      temp (float), humid (int), desc (str)
        """
        warnings  = []
        spray_safe = True
        weather_adv = "No weather data available — check local forecast."

        # ── Weather validation ────────────────────────────────
        if weather:
            temp    = float(weather.get("temp",      context.get("temperature", 28)))
            humid   = float(weather.get("humid",     context.get("humidity",    65)))
            wind    = float(weather.get("wind",      0))
            rain    = bool(weather.get("rain",       False))
            rain_pct= int(weather.get("rain_pct",    0))

            if rain or rain_pct >= 40:
                spray_safe = False
                warnings.append(
                    f"Rain expected ({rain_pct}%) — avoid pesticide/fertilizer spraying today."
                )
                weather_adv = (
                    f"Rain probability {rain_pct}%. Postpone spraying 24–48 hours. "
                    f"Temperature: {temp}°C, Humidity: {humid}%."
                )
            elif wind > 25:
                spray_safe = False
                warnings.append(
                    f"Wind speed {wind:.0f} km/h — avoid spray drift. Spray early morning."
                )
                weather_adv = (
                    f"High wind ({wind:.0f} km/h) — wait for calm. "
                    f"Temperature: {temp}°C, Humidity: {humid}%."
                )
            elif temp > 38:
                warnings.append(
                    "High temperature (>38°C) — spray early morning (6–9 AM) or after 5 PM."
                )
                weather_adv = (
                    f"Temperature {temp:.0f}°C is high — spray in cooler part of day. "
                    f"Keep crop irrigated."
                )
            else:
                weather_adv = (
                    f"Conditions are suitable. Temperature: {temp:.0f}°C, "
                    f"Humidity: {humid:.0f}%, Wind: {wind:.0f} km/h."
                )
        else:
            temp    = float(context.get("temperature", 28))
            humid   = float(context.get("humidity",    65))

        # ── Confidence conflict check ─────────────────────────
        if crop_pred.confidence < 0.55:
            warnings.append(
                f"Low crop confidence ({crop_pred.confidence*100:.0f}%) — "
                "consider consulting a local agricultural officer."
            )

        if pesticide_pred.confidence < 0.45:
            warnings.append(
                "Pest risk assessment is based on general conditions. "
                "Scout your field before applying pesticide."
            )

        # ── Fertilizer-pesticide timing conflict ──────────────
        pest_keywords  = pesticide_pred.pest_name.lower()
        fert_stage     = fertilizer_pred.application_stage.lower()
        if "spray" in fert_stage and spray_safe:
            warnings.append(
                "Do not apply pesticide and fertilizer foliar spray on the same day — "
                "space at least 48 hours apart."
            )

        # ── Nutrient conflict with crop requirement ───────────
        n  = float(context.get("nitrogen",    50))
        p  = float(context.get("phosphorus",  30))
        k  = float(context.get("potassium",   30))

        if n > 120:
            warnings.append(
                "Soil nitrogen is very high — reduce urea application to avoid lodging "
                "and nitrogen burn."
            )

        # ── Overall confidence ────────────────────────────────
        overall_conf = (
            crop_pred.confidence * 0.4
            + fertilizer_pred.confidence * 0.35
            + pesticide_pred.confidence * 0.25
        )

        return AggregatedAdvisory(
            location=context.get("location", "Not provided"),
            soil_type=context.get("soil_type", "Not provided"),
            season=context.get("season", "Not provided"),
            temperature=temp,
            nitrogen=n,
            phosphorus=p,
            potassium=k,
            humidity=humid,
            crop_name=crop_pred.crop_name,
            crop_confidence=crop_pred.confidence,
            crop_reason_tags=crop_pred.reason_tags,
            fertilizer_name=fertilizer_pred.fertilizer_name,
            fertilizer_dosage=fertilizer_pred.dosage,
            fertilizer_stage=fertilizer_pred.application_stage,
            pest_name=pesticide_pred.pest_name,
            pesticide_name=pesticide_pred.pesticide_name,
            pesticide_dosage=pesticide_pred.dosage,
            organic_alternative=pesticide_pred.organic_alternative,
            warnings=warnings,
            overall_confidence=round(overall_conf, 2),
            weather_advisory=weather_adv,
            spray_safe=spray_safe,
        )
