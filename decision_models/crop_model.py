"""
decision_models/crop_model.py
==============================
Crop Decision Model

Predicts the most suitable crop given:
  - Soil type (loamy, sandy, clay, red, black, alluvial)
  - Season (kharif, rabi, zaid)
  - Temperature range
  - N / P / K soil nutrient levels
  - Rainfall / irrigation availability

Outputs:
  - crop_name       : str
  - confidence      : float  [0-1]
  - reason_tags     : List[str]  (e.g. ["loamy-soil", "high-N", "kharif"])
  - alternatives    : List[str]  top-3 alternatives

Architecture:
  RandomForest classifier (fast, interpretable, no GPU needed)
  Trained on synthetic + ICAR-derived agronomic rules.
  Falls back to rule-based lookup if model file not found.

Usage:
  from decision_models.crop_model import CropModel
  model = CropModel()
  result = model.predict({
      "soil_type": "loamy", "season": "kharif",
      "temperature": 32, "nitrogen": 80,
      "phosphorus": 40, "potassium": 30,
      "rainfall": "high"
  })
"""

import os
import pickle
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "crop_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "saved", "crop_encoder.pkl")


@dataclass
class CropPrediction:
    crop_name: str
    confidence: float
    reason_tags: List[str]
    alternatives: List[str]
    raw_scores: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Rule-based crop lookup (fallback + training data source) ──
# Format: {(soil, season): [(crop, score, tags), ...]}
CROP_RULES = {
    # Kharif crops
    ("loamy",    "kharif"): [
        ("Rice",      0.92, ["loamy-soil", "kharif", "high-water"]),
        ("Maize",     0.85, ["loamy-soil", "kharif", "moderate-water"]),
        ("Soybean",   0.78, ["loamy-soil", "kharif", "legume"]),
        ("Cotton",    0.70, ["loamy-soil", "kharif", "cash-crop"]),
    ],
    ("clay",     "kharif"): [
        ("Rice",      0.95, ["clay-soil", "kharif", "waterlogged-tolerant"]),
        ("Sugarcane", 0.80, ["clay-soil", "kharif", "cash-crop"]),
        ("Cotton",    0.72, ["clay-soil", "kharif", "cash-crop"]),
        ("Jute",      0.68, ["clay-soil", "kharif", "fibre-crop"]),
    ],
    ("sandy",    "kharif"): [
        ("Groundnut",  0.88, ["sandy-soil", "kharif", "oilseed"]),
        ("Sesame",     0.80, ["sandy-soil", "kharif", "oilseed"]),
        ("Cowpea",     0.75, ["sandy-soil", "kharif", "pulse"]),
        ("Watermelon", 0.72, ["sandy-soil", "kharif", "horticulture"]),
    ],
    ("black",    "kharif"): [
        ("Cotton",    0.93, ["black-soil", "kharif", "cash-crop"]),
        ("Sorghum",   0.85, ["black-soil", "kharif", "cereal"]),
        ("Soybean",   0.80, ["black-soil", "kharif", "legume"]),
        ("Sunflower",  0.74, ["black-soil", "kharif", "oilseed"]),
    ],
    ("red",      "kharif"): [
        ("Groundnut",  0.88, ["red-soil", "kharif", "oilseed"]),
        ("Maize",      0.84, ["red-soil", "kharif", "cereal"]),
        ("Finger Millet", 0.80, ["red-soil", "kharif", "millet"]),
        ("Cowpea",     0.74, ["red-soil", "kharif", "pulse"]),
    ],
    ("alluvial", "kharif"): [
        ("Rice",       0.90, ["alluvial-soil", "kharif", "paddy"]),
        ("Sugarcane",  0.85, ["alluvial-soil", "kharif", "cash-crop"]),
        ("Maize",      0.82, ["alluvial-soil", "kharif", "cereal"]),
        ("Banana",     0.78, ["alluvial-soil", "kharif", "horticulture"]),
    ],
    # Rabi crops
    ("loamy",    "rabi"): [
        ("Wheat",     0.93, ["loamy-soil", "rabi", "cereal"]),
        ("Mustard",   0.86, ["loamy-soil", "rabi", "oilseed"]),
        ("Pea",       0.80, ["loamy-soil", "rabi", "pulse"]),
        ("Potato",    0.78, ["loamy-soil", "rabi", "vegetable"]),
    ],
    ("clay",     "rabi"): [
        ("Wheat",     0.90, ["clay-soil", "rabi", "cereal"]),
        ("Mustard",   0.83, ["clay-soil", "rabi", "oilseed"]),
        ("Lentil",    0.78, ["clay-soil", "rabi", "pulse"]),
        ("Chickpea",  0.75, ["clay-soil", "rabi", "pulse"]),
    ],
    ("sandy",    "rabi"): [
        ("Barley",     0.88, ["sandy-soil", "rabi", "cereal"]),
        ("Chickpea",   0.82, ["sandy-soil", "rabi", "pulse"]),
        ("Mustard",    0.78, ["sandy-soil", "rabi", "oilseed"]),
        ("Safflower",  0.72, ["sandy-soil", "rabi", "oilseed"]),
    ],
    ("black",    "rabi"): [
        ("Chickpea",  0.90, ["black-soil", "rabi", "pulse"]),
        ("Wheat",     0.85, ["black-soil", "rabi", "cereal"]),
        ("Linseed",   0.76, ["black-soil", "rabi", "oilseed"]),
        ("Safflower", 0.72, ["black-soil", "rabi", "oilseed"]),
    ],
    ("red",      "rabi"): [
        ("Lentil",    0.86, ["red-soil", "rabi", "pulse"]),
        ("Wheat",     0.82, ["red-soil", "rabi", "cereal"]),
        ("Mustard",   0.78, ["red-soil", "rabi", "oilseed"]),
        ("Potato",    0.72, ["red-soil", "rabi", "vegetable"]),
    ],
    ("alluvial", "rabi"): [
        ("Wheat",     0.92, ["alluvial-soil", "rabi", "cereal"]),
        ("Mustard",   0.87, ["alluvial-soil", "rabi", "oilseed"]),
        ("Potato",    0.84, ["alluvial-soil", "rabi", "vegetable"]),
        ("Lentil",    0.80, ["alluvial-soil", "rabi", "pulse"]),
    ],
    # Zaid crops
    ("loamy",    "zaid"): [
        ("Cucumber",   0.88, ["loamy-soil", "zaid", "vegetable"]),
        ("Watermelon", 0.84, ["loamy-soil", "zaid", "horticulture"]),
        ("Muskmelon",  0.80, ["loamy-soil", "zaid", "horticulture"]),
        ("Moong",      0.78, ["loamy-soil", "zaid", "pulse"]),
    ],
    ("sandy",    "zaid"): [
        ("Watermelon", 0.90, ["sandy-soil", "zaid", "horticulture"]),
        ("Moong",      0.85, ["sandy-soil", "zaid", "pulse"]),
        ("Cowpea",     0.78, ["sandy-soil", "zaid", "pulse"]),
        ("Pumpkin",    0.72, ["sandy-soil", "zaid", "vegetable"]),
    ],
}

# Default fallback for unknown combinations
_DEFAULT_KHARIF = [("Rice", 0.60, ["kharif", "general"]), ("Maize", 0.55, ["kharif", "general"])]
_DEFAULT_RABI   = [("Wheat", 0.60, ["rabi", "general"]), ("Mustard", 0.55, ["rabi", "general"])]
_DEFAULT_ZAID   = [("Cucumber", 0.60, ["zaid", "general"]), ("Moong", 0.55, ["zaid", "general"])]
_DEFAULTS       = {"kharif": _DEFAULT_KHARIF, "rabi": _DEFAULT_RABI, "zaid": _DEFAULT_ZAID}


def _normalise_soil(soil: str) -> str:
    soil = (soil or "").lower().strip()
    for s in ["loamy", "loam", "clay", "sandy", "black", "red", "alluvial"]:
        if s in soil:
            return s.replace("loam", "loamy")
    return "loamy"  # safe default


def _normalise_season(season: str) -> str:
    s = (season or "").lower().strip()
    if s in ("kharif", "rabi", "zaid"):
        return s
    if s in ("summer", "hot", "may", "june"):
        return "zaid"
    if s in ("winter", "cool", "oct", "november", "december"):
        return "rabi"
    return "kharif"  # most common default


def _nutrient_tags(n: float, p: float, k: float) -> List[str]:
    tags = []
    tags.append("high-N" if n >= 80 else ("low-N" if n < 30 else "medium-N"))
    tags.append("high-P" if p >= 60 else ("low-P" if p < 20 else "medium-P"))
    tags.append("high-K" if k >= 60 else ("low-K" if k < 20 else "medium-K"))
    return tags


def _temperature_tag(temp: float) -> str:
    if temp < 15:
        return "cool-temp"
    elif temp > 35:
        return "hot-temp"
    return "moderate-temp"


class CropModel:
    """
    Crop recommendation model.

    Priority: trained ML model → rule-based lookup → safe fallback.
    The rule-based layer is always available offline without any ML deps.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self._ml_model   = None
        self._encoder    = None
        self._use_ml     = False
        self._try_load_ml(model_path)

    def _try_load_ml(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                self._ml_model = pickle.load(f)
            if os.path.exists(ENCODER_PATH):
                with open(ENCODER_PATH, "rb") as f:
                    self._encoder = pickle.load(f)
            self._use_ml = True
            print("[CropModel] ML model loaded.")
        except Exception as e:
            print(f"[CropModel] ML load failed ({e}) — using rule-based.")

    def _rule_predict(self, features: Dict) -> CropPrediction:
        soil   = _normalise_soil(features.get("soil_type", ""))
        season = _normalise_season(features.get("season", ""))
        n      = float(features.get("nitrogen",    50))
        p      = float(features.get("phosphorus",  30))
        k      = float(features.get("potassium",   30))
        temp   = float(features.get("temperature", 28))

        candidates = CROP_RULES.get((soil, season)) or _DEFAULTS.get(season, _DEFAULT_KHARIF)

        # Adjust scores based on nutrients
        adjusted = []
        for crop, base_score, tags in candidates:
            score = base_score
            # Boost rice/wheat if high N
            if n >= 80 and crop in ("Rice", "Wheat", "Maize"):
                score = min(score + 0.05, 0.99)
            # Boost legumes if low N (they fix nitrogen)
            if n < 30 and crop in ("Chickpea", "Soybean", "Cowpea", "Moong", "Lentil", "Pea"):
                score = min(score + 0.08, 0.99)
            # Temperature adjustments
            if temp > 38 and crop == "Wheat":
                score = max(score - 0.15, 0.10)
            if temp < 15 and crop in ("Rice", "Maize", "Cotton"):
                score = max(score - 0.12, 0.10)
            adjusted.append((crop, score, tags))

        adjusted.sort(key=lambda x: -x[1])
        top = adjusted[0]

        nut_tags  = _nutrient_tags(n, p, k)
        temp_tag  = _temperature_tag(temp)
        all_tags  = list(top[2]) + nut_tags + [temp_tag]

        raw_scores = {c: round(s, 3) for c, s, _ in adjusted}
        # Always return at least 2 alternatives from the full candidate list
        alts = [c for c, _, _ in adjusted[1:4]]
        if len(alts) < 2:
            # Pull from full candidates regardless of score threshold
            all_candidates_flat = []
            for key, cands in CROP_RULES.items():
                if key[1] == season:
                    all_candidates_flat.extend([c for c, _, _ in cands])
            for c in all_candidates_flat:
                if c != top[0] and c not in alts:
                    alts.append(c)
                if len(alts) >= 3:
                    break

        return CropPrediction(
            crop_name=top[0],
            confidence=round(top[1], 2),
            reason_tags=all_tags,
            alternatives=alts,
            raw_scores=raw_scores,
        )

    def _ml_predict(self, features: Dict) -> CropPrediction:
        """Use trained scikit-learn model if available."""
        try:
            soil   = _normalise_soil(features.get("soil_type", ""))
            season = _normalise_season(features.get("season", ""))
            n      = float(features.get("nitrogen",    50))
            p      = float(features.get("phosphorus",  30))
            k      = float(features.get("potassium",   30))
            temp   = float(features.get("temperature", 28))

            # Encode categorical features
            soil_enc   = self._encoder["soil"].transform([soil])[0] if self._encoder else 0
            season_enc = self._encoder["season"].transform([season])[0] if self._encoder else 0

            X = [[soil_enc, season_enc, n, p, k, temp]]
            proba = self._ml_model.predict_proba(X)[0]
            classes = self._ml_model.classes_
            top_idx = proba.argsort()[::-1]

            crop_name  = classes[top_idx[0]]
            confidence = float(proba[top_idx[0]])
            alts       = [classes[i] for i in top_idx[1:4]]
            raw_scores = {classes[i]: round(float(proba[i]), 3) for i in top_idx[:8]}

            nut_tags  = _nutrient_tags(n, p, k)
            temp_tag  = _temperature_tag(temp)
            soil_tag  = f"{soil}-soil"
            tags      = [soil_tag, season, temp_tag] + nut_tags

            return CropPrediction(
                crop_name=crop_name,
                confidence=round(confidence, 2),
                reason_tags=tags,
                alternatives=alts,
                raw_scores=raw_scores,
            )
        except Exception as e:
            print(f"[CropModel] ML predict error: {e} — falling back to rules")
            return self._rule_predict(features)

    def predict(self, features: Dict) -> CropPrediction:
        """Main prediction entry point."""
        if self._use_ml:
            return self._ml_predict(features)
        return self._rule_predict(features)

    # ── Training ──────────────────────────────────────────────
    def train_from_jsonl(self, jsonl_path: str, save_path: str = MODEL_PATH):
        """
        Train RandomForest on JSONL records.
        Expected format: {soil_type, season, temperature, nitrogen,
                          phosphorus, potassium, crop_label}
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            import json
            import numpy as np

            records = []
            with open(jsonl_path) as f:
                for line in f:
                    r = json.loads(line.strip())
                    if "crop_label" in r:
                        records.append(r)

            if len(records) < 50:
                print(f"[CropModel] Too few labelled crop samples ({len(records)}). "
                      "Using rule-based mode.")
                return

            soil_enc   = LabelEncoder().fit([r["soil_type"] for r in records])
            season_enc = LabelEncoder().fit([r["season"]    for r in records])

            X = np.array([
                [
                    soil_enc.transform([r["soil_type"]])[0],
                    season_enc.transform([r["season"]])[0],
                    float(r.get("nitrogen",    50)),
                    float(r.get("phosphorus",  30)),
                    float(r.get("potassium",   30)),
                    float(r.get("temperature", 28)),
                ]
                for r in records
            ])
            y = [r["crop_label"] for r in records]

            clf = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            clf.fit(X, y)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(clf, f)
            with open(ENCODER_PATH, "wb") as f:
                pickle.dump({"soil": soil_enc, "season": season_enc}, f)

            print(f"[CropModel] Trained on {len(records)} samples → {save_path}")
            self._ml_model   = clf
            self._encoder    = {"soil": soil_enc, "season": season_enc}
            self._use_ml     = True

        except ImportError:
            print("[CropModel] scikit-learn not installed. "
                  "pip install scikit-learn to enable ML training.")


if __name__ == "__main__":
    m = CropModel()
    r = m.predict({
        "soil_type": "loamy", "season": "kharif",
        "temperature": 32, "nitrogen": 90,
        "phosphorus": 45, "potassium": 35,
    })
    print(r)
