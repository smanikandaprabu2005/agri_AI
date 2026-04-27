"""
decision_models/fertilizer_model.py
=====================================
Fertilizer Decision Model

Given crop name, soil type, NPK levels, and growth stage, returns:
  - fertilizer_name     : str   (e.g. "Urea + DAP + MOP")
  - dosage              : str   (e.g. "Urea 18 kg/acre + DAP 27 kg/acre")
  - application_stage   : str   (e.g. "Basal at transplanting; top-dress at tillering")
  - nutrient_gap        : Dict  (how deficient each nutrient is)
  - confidence          : float

Based on ICAR crop-specific nutrient management recommendations.
"""

import os
import pickle
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "fertilizer_model.pkl")


@dataclass
class FertilizerPrediction:
    fertilizer_name: str
    dosage: str
    application_stage: str
    nutrient_gap: Dict[str, str]
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


# ── ICAR crop fertilizer database ─────────────────────────────
# {crop: {stage: {nutrient: kg_per_acre}}}
CROP_FERTILIZER_DB = {
    "Rice": {
        "basal": {"N": 18, "P": 27, "K": 10},
        "tillering": {"N": 18, "P": 0, "K": 0},
        "panicle_initiation": {"N": 9, "P": 0, "K": 5},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Basal at transplanting; 1st top-dress at active tillering (21 DAT); "
                  "2nd top-dress at panicle initiation (45 DAT)",
    },
    "Wheat": {
        "basal": {"N": 25, "P": 30, "K": 12},
        "crown_root": {"N": 25, "P": 0, "K": 0},
        "jointing": {"N": 12, "P": 0, "K": 0},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Basal at sowing; 1st top-dress at crown root initiation (21 DAS); "
                  "2nd top-dress at jointing stage (45 DAS)",
    },
    "Maize": {
        "basal": {"N": 20, "P": 25, "K": 10},
        "knee_high": {"N": 20, "P": 0, "K": 0},
        "tasseling": {"N": 10, "P": 0, "K": 5},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "Single Superphosphate (16% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Basal at sowing; top-dress at knee-high stage (25 DAS); "
                  "apply K at tasseling (50 DAS)",
    },
    "Tomato": {
        "basal": {"N": 12, "P": 18, "K": 12},
        "vegetative": {"N": 8, "P": 0, "K": 8},
        "flowering": {"N": 6, "P": 0, "K": 10},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Potassium Sulphate (50% K₂O)",
        },
        "stages": "Basal 2 weeks before transplanting; top-dress at vegetative stage "
                  "(30 DAT); foliar K at flowering",
    },
    "Potato": {
        "basal": {"N": 19, "P": 45, "K": 12},
        "tuber_initiation": {"N": 10, "P": 0, "K": 8},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "Ammonium Phosphate (11% N, 52% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "All P + K + 50% N as basal at planting; remaining N at tuber initiation "
                  "(30 DAS)",
    },
    "Cotton": {
        "basal": {"N": 10, "P": 20, "K": 10},
        "square_formation": {"N": 10, "P": 0, "K": 5},
        "boll_development": {"N": 10, "P": 0, "K": 5},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Basal at sowing; 1st top-dress at square formation (30 DAS); "
                  "2nd at boll development (60 DAS)",
    },
    "Sugarcane": {
        "basal": {"N": 25, "P": 30, "K": 15},
        "tillering": {"N": 25, "P": 0, "K": 15},
        "grand_growth": {"N": 25, "P": 0, "K": 0},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "SSP (16% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Full P + K + 1/3 N as basal; 1/3 N at tillering (60 DAP); "
                  "1/3 N at grand growth phase (120 DAP)",
    },
    "Mustard": {
        "basal": {"N": 12, "P": 30, "K": 3},
        "branching": {"N": 6, "P": 0, "K": 0},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "All P + K + 50% N at sowing; remaining N at branching stage "
                  "(25–30 DAS)",
    },
    "Groundnut": {
        "basal": {"N": 5, "P": 25, "K": 10},
        "flowering": {"N": 3, "P": 0, "K": 5},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "SSP (16% P₂O₅) — also provides Ca",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Full dose at sowing; light N top-dress at flowering if needed. "
                  "Apply gypsum 100 kg/acre at peg formation.",
    },
    "Onion": {
        "basal": {"N": 10, "P": 20, "K": 10},
        "bulb_development": {"N": 10, "P": 0, "K": 10},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Potassium Sulphate (50% K₂O)",
        },
        "stages": "50% N + full P + K as basal; 50% N at bulb development "
                  "(45 DAT)",
    },
    "Banana": {
        "basal": {"N": 18, "P": 18, "K": 36},
        "third_month": {"N": 18, "P": 0, "K": 18},
        "sixth_month": {"N": 18, "P": 0, "K": 18},
        "fertilizer_names": {
            "N": "Urea (46% N)",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Basal at planting; 2nd dose at 3rd month; 3rd dose at 6th month",
    },
    "Chickpea": {
        "basal": {"N": 4, "P": 20, "K": 10},
        "fertilizer_names": {
            "N": "Urea (46% N) — light starter dose only",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "All as basal at sowing. Use Rhizobium seed treatment to reduce "
                  "N requirement. No top-dress needed.",
    },
    "Soybean": {
        "basal": {"N": 4, "P": 25, "K": 12},
        "flowering": {"N": 0, "P": 0, "K": 5},
        "fertilizer_names": {
            "N": "Urea — minimal, Rhizobium fixes N",
            "P": "DAP (18% N, 46% P₂O₅)",
            "K": "Muriate of Potash (60% K₂O)",
        },
        "stages": "Basal at sowing with Rhizobium + PSB seed treatment. "
                  "K top-dress at flowering if deficient.",
    },
}

# Generic fallback for unknown crops
_GENERIC_FERTILIZER = {
    "fertilizer_name":     "NPK 19:19:19",
    "dosage":              "5 kg per acre dissolved in water as foliar spray, or 15 kg per acre as soil application",
    "application_stage":   "Apply at vegetative stage; repeat at flowering if needed",
    "confidence":          0.50,
}


def _calc_nutrient_gap(n: float, p: float, k: float) -> Dict[str, str]:
    """Return advisory on nutrient levels."""
    gaps = {}
    if n < 30:
        gaps["Nitrogen"] = f"Low ({n:.0f} kg/ha) — increase N application"
    elif n > 120:
        gaps["Nitrogen"] = f"High ({n:.0f} kg/ha) — reduce N dose by 20%"
    else:
        gaps["Nitrogen"] = f"Adequate ({n:.0f} kg/ha)"

    if p < 15:
        gaps["Phosphorus"] = f"Low ({p:.0f} kg/ha) — apply extra DAP/SSP"
    elif p > 80:
        gaps["Phosphorus"] = f"High ({p:.0f} kg/ha) — skip P application this season"
    else:
        gaps["Phosphorus"] = f"Adequate ({p:.0f} kg/ha)"

    if k < 15:
        gaps["Potassium"] = f"Low ({k:.0f} kg/ha) — apply MOP or SOP"
    elif k > 100:
        gaps["Potassium"] = f"High ({k:.0f} kg/ha) — reduce K application"
    else:
        gaps["Potassium"] = f"Adequate ({k:.0f} kg/ha)"

    return gaps


def _format_dosage(crop_data: dict, stage: str = "basal") -> str:
    """Build human-readable dosage string from crop DB."""
    doses = crop_data.get(stage, crop_data.get("basal", {}))
    names = crop_data.get("fertilizer_names", {})
    parts = []
    for nutrient, kg in doses.items():
        if kg > 0:
            fname = names.get(nutrient, f"{nutrient} fertilizer")
            parts.append(f"{fname}: {kg} kg/acre")
    return " + ".join(parts) if parts else "Standard NPK as per soil test"


class FertilizerModel:
    """
    Fertilizer recommendation model.
    Rule-based ICAR data — always available offline.
    """

    def __init__(self):
        pass

    def predict(self, features: Dict) -> FertilizerPrediction:
        """
        features keys:
          crop_name, soil_type, season, nitrogen, phosphorus,
          potassium, growth_stage (optional)
        """
        crop  = (features.get("crop_name", "") or "").strip().title()
        n     = float(features.get("nitrogen",    50))
        p     = float(features.get("phosphorus",  30))
        k     = float(features.get("potassium",   30))
        stage = (features.get("growth_stage", "basal") or "basal").lower()

        # Look up crop in database
        crop_data = None
        for key in CROP_FERTILIZER_DB:
            if key.lower() == crop.lower() or crop.lower() in key.lower():
                crop_data = CROP_FERTILIZER_DB[key]
                crop = key
                break

        if crop_data is None:
            return FertilizerPrediction(
                fertilizer_name=_GENERIC_FERTILIZER["fertilizer_name"],
                dosage=_GENERIC_FERTILIZER["dosage"],
                application_stage=_GENERIC_FERTILIZER["application_stage"],
                nutrient_gap=_calc_nutrient_gap(n, p, k),
                confidence=_GENERIC_FERTILIZER["confidence"],
            )

        # Adjust doses based on soil nutrient levels
        base_dose = dict(crop_data.get("basal", {}))
        if n > 80:
            base_dose["N"] = max(int(base_dose.get("N", 0) * 0.75), 0)
        if p > 60:
            base_dose["P"] = max(int(base_dose.get("P", 0) * 0.70), 0)
        if k > 80:
            base_dose["K"] = max(int(base_dose.get("K", 0) * 0.70), 0)

        # Build fertilizer name string
        names  = crop_data.get("fertilizer_names", {})
        fname  = " + ".join(
            f"{names.get(nut, nut)}"
            for nut, dose in base_dose.items() if dose > 0
        )

        # Build dosage string with adjusted values
        dosage_parts = []
        for nut, dose in base_dose.items():
            if dose > 0:
                dosage_parts.append(f"{names.get(nut, nut)}: {dose} kg/acre")
        dosage = "; ".join(dosage_parts)

        stages_str = crop_data.get("stages", "Apply as per standard schedule")

        return FertilizerPrediction(
            fertilizer_name=fname,
            dosage=dosage,
            application_stage=stages_str,
            nutrient_gap=_calc_nutrient_gap(n, p, k),
            confidence=0.90,
        )


if __name__ == "__main__":
    m = FertilizerModel()
    r = m.predict({
        "crop_name": "Rice", "soil_type": "loamy",
        "nitrogen": 45, "phosphorus": 20, "potassium": 30,
        "growth_stage": "basal",
    })
    print(r)
