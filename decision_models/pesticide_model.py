"""
decision_models/pesticide_model.py
====================================
Pesticide / Pest Control Decision Model

Given crop, season, temperature, humidity and optional symptom keywords:
  - pest_name           : str   (e.g. "Brown Plant Hopper")
  - pesticide_name      : str   (e.g. "Imidacloprid 17.8 SL")
  - dosage              : str   (e.g. "0.3 ml/litre water, 3 sprays × 7 days")
  - application_method  : str
  - safety_interval     : str   (days before harvest)
  - confidence          : float

Data source: ICAR crop protection manuals + CIB&RC registered pesticides.
All products are CIB&RC (India) registered as of 2024.
"""

import re
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class PesticidePrediction:
    pest_name: str
    pesticide_name: str
    dosage: str
    application_method: str
    safety_interval: str       # Pre-harvest interval (PHI)
    organic_alternative: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


# ── Pest database (crop → list of seasonal pests) ─────────────
# Format: {crop: [{pest, season, humidity_trigger, temp_range,
#                  pesticide, dosage, method, phi, organic}]}
PEST_DB = {
    "Rice": [
        {
            "pest": "Brown Plant Hopper (BPH)",
            "season": ["kharif"],
            "humidity_min": 70,
            "temp_range": (25, 35),
            "pesticide": "Imidacloprid 17.8 SL",
            "dosage": "0.25 ml/litre water; 3 sprays × 7 days interval",
            "method": "Directed spray at plant base / flooding; avoid foliar spray",
            "phi": "21 days",
            "organic": "Neem oil 3% (5 ml/litre) + soap solution",
        },
        {
            "pest": "Stem Borer (Yellow / White)",
            "season": ["kharif", "rabi"],
            "humidity_min": 60,
            "temp_range": (20, 40),
            "pesticide": "Chlorpyrifos 20 EC",
            "dosage": "2.5 ml/litre water; spray at egg hatching stage",
            "method": "Foliar spray — thorough coverage of leaf sheath",
            "phi": "25 days",
            "organic": "Trichogramma japonicum egg parasitoid release (50,000/acre)",
        },
        {
            "pest": "Leaf Folder",
            "season": ["kharif"],
            "humidity_min": 65,
            "temp_range": (25, 35),
            "pesticide": "Cartap Hydrochloride 50 SP",
            "dosage": "1.5 g/litre water; 2 sprays × 10 days interval",
            "method": "Foliar spray at vegetative stage",
            "phi": "14 days",
            "organic": "Neem seed kernel extract (NSKE) 5%",
        },
        {
            "pest": "Bacterial Leaf Blight (BLB)",
            "season": ["kharif"],
            "humidity_min": 75,
            "temp_range": (25, 35),
            "pesticide": "Streptomycin Sulphate + Tetracycline (Agrimycin)",
            "dosage": "0.1% solution (1 g/litre); spray at first symptom",
            "method": "Foliar spray; do NOT spray during rain",
            "phi": "7 days",
            "organic": "Copper Oxychloride 50 WP at 3 g/litre as preventive",
        },
        {
            "pest": "Neck Blast / Leaf Blast",
            "season": ["kharif"],
            "humidity_min": 80,
            "temp_range": (20, 30),
            "pesticide": "Tricyclazole 75 WP (Beam)",
            "dosage": "0.6 g/litre water; spray at boot leaf stage",
            "method": "Foliar spray — critical at heading and flowering",
            "phi": "21 days",
            "organic": "Pseudomonas fluorescens bioagent spray 10 g/litre",
        },
    ],
    "Tomato": [
        {
            "pest": "Aphids",
            "season": ["rabi", "zaid"],
            "humidity_min": 55,
            "temp_range": (15, 30),
            "pesticide": "Imidacloprid 17.8 SL",
            "dosage": "0.3 ml/litre water; 3 sprays × 5–7 days",
            "method": "Foliar spray, especially leaf undersides",
            "phi": "7 days",
            "organic": "Neem oil 3% at 5 ml/litre or soap spray (5 g/litre)",
        },
        {
            "pest": "Whitefly",
            "season": ["kharif", "rabi"],
            "humidity_min": 55,
            "temp_range": (20, 35),
            "pesticide": "Thiamethoxam 25 WG",
            "dosage": "0.3 g/litre water; 2 sprays × 7 days",
            "method": "Foliar spray morning; rotate with Spiromesifen for resistance management",
            "phi": "7 days",
            "organic": "Yellow sticky traps (25/acre) + Neem oil 3%",
        },
        {
            "pest": "Late Blight (Phytophthora infestans)",
            "season": ["rabi"],
            "humidity_min": 80,
            "temp_range": (15, 25),
            "pesticide": "Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold MZ)",
            "dosage": "2.5 g/litre water; spray every 7 days",
            "method": "Preventive foliar spray; thorough coverage including undersides",
            "phi": "10 days",
            "organic": "Copper Oxychloride 50 WP at 3 g/litre as preventive",
        },
        {
            "pest": "Fruit Borer (Helicoverpa armigera)",
            "season": ["kharif", "rabi"],
            "humidity_min": 50,
            "temp_range": (20, 38),
            "pesticide": "Emamectin Benzoate 5 SG",
            "dosage": "0.4 g/litre water; spray at egg hatching",
            "method": "Foliar spray in evening; target small larvae",
            "phi": "5 days",
            "organic": "Bt (Bacillus thuringiensis) var. kurstaki 0.5 kg/acre",
        },
    ],
    "Cotton": [
        {
            "pest": "Bollworm (American, Spotted, Pink)",
            "season": ["kharif"],
            "humidity_min": 50,
            "temp_range": (25, 40),
            "pesticide": "Profenofos 50 EC",
            "dosage": "2 ml/litre water; spray at 50% boll stage",
            "method": "Foliar spray; spray in morning for better coverage",
            "phi": "28 days",
            "organic": "Trichogramma release (1.5 lakh/acre) + pheromone traps",
        },
        {
            "pest": "Whitefly (Bemisia tabaci)",
            "season": ["kharif"],
            "humidity_min": 50,
            "temp_range": (25, 40),
            "pesticide": "Spiromesifen 22.9 SC",
            "dosage": "0.9 ml/litre water; 2 sprays × 10 days",
            "method": "Direct spray on leaf undersides",
            "phi": "7 days",
            "organic": "Neem-based formulation 0.15% EC at 3 ml/litre",
        },
        {
            "pest": "Aphids on Cotton",
            "season": ["kharif"],
            "humidity_min": 50,
            "temp_range": (20, 35),
            "pesticide": "Dimethoate 30 EC",
            "dosage": "1.5 ml/litre water; spray in early morning",
            "method": "Foliar spray — 2 sprays × 7 days if infestation high",
            "phi": "14 days",
            "organic": "Soap + Neem oil emulsion (5 ml + 5 g/litre)",
        },
    ],
    "Wheat": [
        {
            "pest": "Aphids (Sitobion avenae)",
            "season": ["rabi"],
            "humidity_min": 55,
            "temp_range": (8, 22),
            "pesticide": "Chlorpyrifos 20 EC",
            "dosage": "1.5 ml/litre water; 1–2 sprays",
            "method": "Foliar spray at flag leaf stage; avoid windy conditions",
            "phi": "21 days",
            "organic": "Ladybird beetle conservation + Neem seed extract 5%",
        },
        {
            "pest": "Yellow Rust (Puccinia striiformis)",
            "season": ["rabi"],
            "humidity_min": 75,
            "temp_range": (5, 20),
            "pesticide": "Propiconazole 25 EC (Tilt)",
            "dosage": "1 ml/litre water; spray at first symptom",
            "method": "Preventive foliar spray; 2 sprays × 14 days",
            "phi": "21 days",
            "organic": "Use resistant variety (HD 2967, PBW 343); early sowing",
        },
    ],
    "Potato": [
        {
            "pest": "Late Blight (Phytophthora infestans)",
            "season": ["rabi"],
            "humidity_min": 80,
            "temp_range": (10, 22),
            "pesticide": "Mancozeb 75 WP (Dithane M-45)",
            "dosage": "2.5 g/litre water; spray every 7 days from 30 DAS",
            "method": "Thorough foliar spray covering both leaf surfaces",
            "phi": "10 days",
            "organic": "Copper Oxychloride 50 WP 3 g/litre as preventive",
        },
        {
            "pest": "Aphids (Myzus persicae)",
            "season": ["rabi"],
            "humidity_min": 50,
            "temp_range": (12, 25),
            "pesticide": "Imidacloprid 70 WS (seed treatment)",
            "dosage": "5 g/kg seed as seed treatment OR 0.25 ml/litre foliar",
            "method": "Seed treatment preferred; foliar only if outbreak severe",
            "phi": "N/A (seed treatment)",
            "organic": "Mineral oil spray + yellow sticky traps",
        },
    ],
    "Lemon": [
        {
            "pest": "Aphids on Citrus",
            "season": ["kharif", "rabi", "zaid"],
            "humidity_min": 55,
            "temp_range": (20, 35),
            "pesticide": "Malathion 50 EC",
            "dosage": "2 ml/litre water; 3 sprays × 5 days interval",
            "method": "Foliar spray morning; target new shoots and leaf undersides",
            "phi": "7 days",
            "organic": "Neem oil 0.5% at 5 ml/litre; repeat weekly",
        },
        {
            "pest": "Citrus Psyllid",
            "season": ["kharif", "rabi"],
            "humidity_min": 50,
            "temp_range": (20, 35),
            "pesticide": "Thiamethoxam 25 WG",
            "dosage": "0.3 g/litre water; spray new leaf flush",
            "method": "Target new flush; 2 sprays per flush",
            "phi": "14 days",
            "organic": "Kaolin clay spray 5% on new growth",
        },
    ],
    "Groundnut": [
        {
            "pest": "Leaf Miner (Aproaerema modicella)",
            "season": ["kharif"],
            "humidity_min": 60,
            "temp_range": (25, 38),
            "pesticide": "Chlorpyrifos 20 EC",
            "dosage": "2 ml/litre; spray at first instar larvae",
            "method": "Foliar spray; 2 sprays × 10 days",
            "phi": "21 days",
            "organic": "Neem seed kernel extract (NSKE) 5% spray",
        },
    ],
    "Onion": [
        {
            "pest": "Thrips (Thrips tabaci)",
            "season": ["rabi"],
            "humidity_min": 45,
            "temp_range": (20, 35),
            "pesticide": "Spinosad 45 SC",
            "dosage": "0.3 ml/litre water; 3 sprays × 7 days",
            "method": "Spray in evening; target leaf axils",
            "phi": "7 days",
            "organic": "Blue sticky traps (20/acre) + Neem extract 5%",
        },
    ],
}

# Unknown crop default
_DEFAULT_PEST = PesticidePrediction(
    pest_name="General Insect Pests",
    pesticide_name="Chlorpyrifos 20 EC",
    dosage="2 ml/litre water; spray as needed",
    application_method="Foliar spray — early morning",
    safety_interval="21 days",
    organic_alternative="Neem oil 3% (5 ml/litre) as broad-spectrum organic option",
    confidence=0.35,
)


def _season_matches(pest_seasons: List[str], query_season: str) -> bool:
    if not query_season:
        return True
    q = query_season.lower().strip()
    return q in pest_seasons or not pest_seasons


def _weather_matches(pest: dict, temperature: float, humidity: float) -> float:
    """Score how much current weather favours this pest (0-1)."""
    score = 0.5  # neutral baseline

    tmin, tmax = pest.get("temp_range", (0, 50))
    if tmin <= temperature <= tmax:
        score += 0.25
    elif abs(temperature - tmin) < 5 or abs(temperature - tmax) < 5:
        score += 0.10

    hmin = pest.get("humidity_min", 0)
    if humidity >= hmin:
        score += 0.25

    return min(score, 1.0)


def _keyword_matches(pest: dict, symptoms: str) -> float:
    """Boost score if symptom keywords match pest description."""
    if not symptoms:
        return 0.0
    pest_text = f"{pest.get('pest', '')} {pest.get('pesticide', '')}".lower()
    sym_words = symptoms.lower().split()
    hits = sum(1 for w in sym_words if w in pest_text)
    return min(hits * 0.15, 0.45)


class PesticideModel:
    """
    Pest identification and pesticide recommendation model.
    Rule-based database — always available offline.
    """

    def __init__(self):
        pass

    def predict(self, features: Dict) -> PesticidePrediction:
        """
        features keys:
          crop_name, season, temperature, humidity,
          symptoms (optional text), growth_stage (optional)
        """
        crop     = (features.get("crop_name", "") or "").strip().title()
        season   = (features.get("season",    "") or "").lower().strip()
        temp     = float(features.get("temperature", 28))
        humidity = float(features.get("humidity",    65))
        symptoms = (features.get("symptoms",  "") or "").lower()

        # Try to find crop in database (fuzzy match)
        crop_pests = None
        for key in PEST_DB:
            if key.lower() == crop.lower() or crop.lower() in key.lower():
                crop_pests = PEST_DB[key]
                break

        if not crop_pests:
            # Generic fallback
            result = _DEFAULT_PEST
            result.confidence = 0.30
            return result

        # Score each pest for current conditions
        scored = []
        for pest in crop_pests:
            if not _season_matches(pest.get("season", []), season):
                continue
            weather_score  = _weather_matches(pest, temp, humidity)
            keyword_score  = _keyword_matches(pest, symptoms)
            total_score    = weather_score + keyword_score
            scored.append((total_score, pest))

        if not scored:
            # Season doesn't match anything — take most common pest
            scored = [(0.4, crop_pests[0])]

        scored.sort(key=lambda x: -x[0])
        best_score, best_pest = scored[0]
        confidence = min(best_score, 1.0)

        return PesticidePrediction(
            pest_name=best_pest["pest"],
            pesticide_name=best_pest["pesticide"],
            dosage=best_pest["dosage"],
            application_method=best_pest["method"],
            safety_interval=best_pest.get("phi", "21 days"),
            organic_alternative=best_pest.get("organic", "Neem oil 3% as general organic option"),
            confidence=round(confidence, 2),
        )

    def list_pests(self, crop: str) -> List[str]:
        """List all known pests for a crop."""
        crop = crop.title()
        pests = PEST_DB.get(crop, [])
        return [p["pest"] for p in pests]


if __name__ == "__main__":
    m = PesticideModel()
    r = m.predict({
        "crop_name": "Rice", "season": "kharif",
        "temperature": 32, "humidity": 80,
        "symptoms": "stem borer dead heart",
    })
    print(r)
