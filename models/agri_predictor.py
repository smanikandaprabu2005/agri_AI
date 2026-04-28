"""
models/agri_predictor.py
========================
Agricultural ML Prediction Engine for SageStorm Advisory System

Trains on two datasets:
  1. Soil dataset  (Soil_Type, pH, Moisture, N/P/K, Temp, Humidity, Rainfall,
                    Crop_Type, Season, Irrigation, Fertilizer_Recommended ...)
  2. NPK dataset   (N, P, K, temperature, humidity, ph, rainfall, label, fertilizer)

Produces three predictions:
  - predicted_crop         (classification)
  - recommended_fertilizer (classification)
  - confidence scores      (probability)

Then builds an EXPLANATION PROMPT for SageStorm to narrate the decision.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional


# ── Try importing ML libraries (graceful fallback) ──────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[AgriPredictor] scikit-learn not found — using rule-based fallback")


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models", "agri_ml")

# ── Feature columns for each dataset ────────────────────────
SOIL_FEATURES = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Nitrogen_N", "Phosphorus_P", "Potassium_K",
    "Temperature", "Humidity", "Rainfall",
]
SOIL_CAT_FEATURES = ["Soil_Type", "Season", "Irrigation_Method", "Region"]

NPK_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ════════════════════════════════════════════════════════════
#  Encoder registry (keep all label encoders at one place)
# ════════════════════════════════════════════════════════════
class EncoderRegistry:
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit_transform(self, series: pd.Series, name: str) -> np.ndarray:
        le = LabelEncoder()
        vals = le.fit_transform(series.astype(str))
        self.encoders[name] = le
        return vals

    def transform(self, value: str, name: str) -> int:
        le = self.encoders.get(name)
        if le is None:
            return 0
        try:
            return int(le.transform([str(value)])[0])
        except ValueError:
            # Unseen label — return most common class index
            return 0

    def inverse(self, idx: int, name: str) -> str:
        le = self.encoders.get(name)
        if le is None:
            return "Unknown"
        try:
            return str(le.inverse_transform([idx])[0])
        except Exception:
            return "Unknown"

    def classes(self, name: str) -> List[str]:
        le = self.encoders.get(name)
        return list(le.classes_) if le else []


# ════════════════════════════════════════════════════════════
#  AgriPredictor
# ════════════════════════════════════════════════════════════
class AgriPredictor:
    """
    Wraps two Random Forest models:
      crop_model      — predicts crop from soil + environment features
      fert_model      — predicts fertilizer from NPK + environment features

    Also contains rule-based pesticide selector.
    """

    def __init__(self):
        self.crop_model = None
        self.fert_model = None
        self.crop_scaler = StandardScaler() if SKLEARN_OK else None
        self.fert_scaler = StandardScaler() if SKLEARN_OK else None
        self.enc = EncoderRegistry()
        self.trained = False
        self._load_if_exists()

    # ── Paths ────────────────────────────────────────────────
    @staticmethod
    def _path(name):
        os.makedirs(MODEL_DIR, exist_ok=True)
        return os.path.join(MODEL_DIR, name)

    # ── Persist / load ───────────────────────────────────────
    def save(self):
        with open(self._path("crop_model.pkl"), "wb") as f:
            pickle.dump(self.crop_model, f)
        with open(self._path("fert_model.pkl"), "wb") as f:
            pickle.dump(self.fert_model, f)
        with open(self._path("scalers.pkl"), "wb") as f:
            pickle.dump((self.crop_scaler, self.fert_scaler), f)
        with open(self._path("encoders.pkl"), "wb") as f:
            pickle.dump(self.enc, f)
        print(f"[AgriPredictor] Models saved → {MODEL_DIR}")

    def _load_if_exists(self):
        try:
            if not Path(self._path("crop_model.pkl")).exists():
                return
            with open(self._path("crop_model.pkl"), "rb") as f:
                self.crop_model = pickle.load(f)
            with open(self._path("fert_model.pkl"), "rb") as f:
                self.fert_model = pickle.load(f)
            with open(self._path("scalers.pkl"), "rb") as f:
                self.crop_scaler, self.fert_scaler = pickle.load(f)
            with open(self._path("encoders.pkl"), "rb") as f:
                self.enc = pickle.load(f)
            self.trained = True
            print("[AgriPredictor] Loaded saved models")
        except Exception as e:
            print(f"[AgriPredictor] Could not load models: {e}")

    # ── Training ─────────────────────────────────────────────
    def train_on_soil_dataset(self, df: pd.DataFrame):
        """
        Train crop prediction model from the soil CSV dataset.

        Expected columns: Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon,
        Electrical_Conductivity, Nitrogen_N, Phosphorus_P, Potassium_K,
        Temperature, Humidity, Rainfall, Crop_Type (target),
        Season, Irrigation_Method, Region
        """
        if not SKLEARN_OK:
            print("[AgriPredictor] sklearn missing — skipping soil model training")
            return

        df = df.copy().dropna(subset=["Crop_Type"])
        print(f"[AgriPredictor] Training crop model on {len(df)} samples")

        # Encode categoricals
        X_cat = np.column_stack([
            self.enc.fit_transform(df[c], c)
            for c in SOIL_CAT_FEATURES if c in df.columns
        ])

        # Numeric features (fill missing with median)
        num_cols = [c for c in SOIL_FEATURES if c in df.columns]
        X_num = df[num_cols].fillna(df[num_cols].median()).values

        X = np.hstack([X_num, X_cat])
        X = self.crop_scaler.fit_transform(X)

        y = self.enc.fit_transform(df["Crop_Type"], "crop")

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15,
                                                      random_state=42, stratify=y)

        self.crop_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
        self.crop_model.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, self.crop_model.predict(X_val))
        print(f"[AgriPredictor] Crop model accuracy: {acc*100:.1f}%")

    def train_on_npk_dataset(self, df: pd.DataFrame):
        """
        Train BOTH crop and fertilizer models from NPK dataset.

        Expected columns: N, P, K, temperature, humidity, ph, rainfall,
        label (crop target), fertilizer (target)

        FIX 2: Use NPK dataset for both models instead of soil dataset.
        This provides ~85-95% crop accuracy (vs 15% from soil dataset).
        """
        if not SKLEARN_OK:
            return

        num_cols = [c for c in NPK_FEATURES if c in df.columns]

        # ──────────────────────────────────────────────────────
        # CROP MODEL TRAINING (from NPK dataset label column)
        # ──────────────────────────────────────────────────────
        if "label" in df.columns:
            df_crop = df.copy().dropna(subset=["label"])
            print(f"[AgriPredictor] Training crop model on {len(df_crop)} samples")

            X_crop = df_crop[num_cols].fillna(df_crop[num_cols].median()).values
            X_crop = self.crop_scaler.fit_transform(X_crop)

            y_crop = self.enc.fit_transform(df_crop["label"], "crop")

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_crop, y_crop, test_size=0.15,
                random_state=42, stratify=y_crop
            )

            # FIX 6: Increase model strength for better crop predictions
            self.crop_model = RandomForestClassifier(
                n_estimators=500,       # Increased from 200
                max_depth=None,         # Allow deeper trees
                min_samples_leaf=1,     # Reduce leaf size
                n_jobs=-1,              # Use all cores
                random_state=42,
            )
            self.crop_model.fit(X_tr, y_tr)
            acc = accuracy_score(y_val, self.crop_model.predict(X_val))
            print(f"[AgriPredictor] Crop model accuracy: {acc*100:.1f}%")

        # ──────────────────────────────────────────────────────
        # FERTILIZER MODEL TRAINING (from NPK dataset)
        # ──────────────────────────────────────────────────────
        df_fert = df.copy().dropna(subset=["fertilizer"])
        print(f"[AgriPredictor] Training fertilizer model on {len(df_fert)} samples")

        X_fert = df_fert[num_cols].fillna(df_fert[num_cols].median()).values

        # Also add crop label as a feature if available
        if "label" in df_fert.columns:
            crop_enc = self.enc.fit_transform(df_fert["label"], "npk_crop")
            X_fert = np.column_stack([X_fert, crop_enc])

        X_fert = self.fert_scaler.fit_transform(X_fert)
        y_series = df_fert["fertilizer"]
        unique_count = y_series.nunique()
        if (y_series.dtype.kind in "iuf" and unique_count > 20) or unique_count > 30:
            print("[AgriPredictor] Warning: fertilizer target appears numeric/continuous or has too many distinct values.")
            print("[AgriPredictor] Skipping fertilizer classification training on this dataset.")
            return

        y_fert = self.enc.fit_transform(y_series, "fertilizer")
        class_counts = np.bincount(y_fert)
        if np.any(class_counts < 2):
            print("[AgriPredictor] Warning: fertilizer target has classes with fewer than 2 samples. "
                  "Skipping stratified split and using a plain random split.")
            stratify = None
        else:
            stratify = y_fert

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_fert, y_fert, test_size=0.15,
            random_state=42, stratify=stratify
        )

        self.fert_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        self.fert_model.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, self.fert_model.predict(X_val))
        print(f"[AgriPredictor] Fertilizer model accuracy: {acc*100:.1f}%")
        self.trained = True

    def train(self, soil_csv_path: str = None, npk_csv_path: str = None):
        """Convenience wrapper — train from file paths."""
        if soil_csv_path and Path(soil_csv_path).exists():
            df = pd.read_csv(soil_csv_path)
            self.train_on_soil_dataset(df)
        if npk_csv_path and Path(npk_csv_path).exists():
            df = pd.read_csv(npk_csv_path)
            self.train_on_npk_dataset(df)
        if self.trained:
            self.save()

    # ── Inference ─────────────────────────────────────────────
    def predict(self, inputs: Dict) -> Dict:
        """
        Main entry point.

        inputs = {
            "soil_type": "Clay",
            "ph": 6.5,
            "soil_moisture": 35.0,
            "organic_carbon": 0.8,
            "ec": 0.5,
            "N": 90, "P": 40, "K": 50,
            "temperature": 28.0,
            "humidity": 70.0,
            "rainfall": 120.0,
            "season": "Kharif",
            "irrigation": "Sprinkler",
            "region": "East",
            "previous_crop": "Rice",
        }

        Returns dict with predictions + confidence + explanation context.
        """
        if self.trained and SKLEARN_OK:
            return self._ml_predict(inputs)
        else:
            return self._rule_predict(inputs)

    def _ml_predict(self, inp: Dict) -> Dict:
        # ── Crop prediction (soil model) ─────────────────────
        soil_num = np.array([
            inp.get("ph", 6.5),
            inp.get("soil_moisture", 30.0),
            inp.get("organic_carbon", 0.8),
            inp.get("ec", 0.5),
            inp.get("N", 80),
            inp.get("P", 40),
            inp.get("K", 40),
            inp.get("temperature", 25.0),
            inp.get("humidity", 65.0),
            inp.get("rainfall", 100.0),
        ]).reshape(1, -1)

        soil_cat = np.array([
            self.enc.transform(inp.get("soil_type", "Loamy"), "Soil_Type"),
            self.enc.transform(inp.get("season", "Kharif"), "Season"),
            self.enc.transform(inp.get("irrigation", "Sprinkler"), "Irrigation_Method"),
            self.enc.transform(inp.get("region", "East"), "Region"),
        ]).reshape(1, -1)

        X_crop = np.hstack([soil_num, soil_cat])

        crop_pred, crop_conf, crop_top3 = "Rice", 0.72, []
        if self.crop_model is not None:
            try:
                X_s = self.crop_scaler.transform(X_crop)
                proba = self.crop_model.predict_proba(X_s)[0]
                top3_idx = np.argsort(proba)[::-1][:3]
                crop_pred = self.enc.inverse(top3_idx[0], "crop")
                crop_conf = float(proba[top3_idx[0]])
                crop_top3 = [
                    {"crop": self.enc.inverse(i, "crop"), "confidence": float(proba[i])}
                    for i in top3_idx
                ]
            except Exception as e:
                print(f"[AgriPredictor] Crop predict error: {e}")

        # ── Fertilizer prediction (NPK model) ────────────────
        npk_num = np.array([
            inp.get("N", 80),
            inp.get("P", 40),
            inp.get("K", 40),
            inp.get("temperature", 25.0),
            inp.get("humidity", 65.0),
            inp.get("ph", 6.5),
            inp.get("rainfall", 100.0),
        ]).reshape(1, -1)

        if "label" in self.enc.encoders:
            crop_code = self.enc.transform(crop_pred, "npk_crop")
            npk_num = np.column_stack([npk_num, [[crop_code]]])

        fert_pred, fert_conf = "Urea", 0.65
        fert_top3 = []
        if self.fert_model is not None:
            try:
                X_f = self.fert_scaler.transform(npk_num)
                proba = self.fert_model.predict_proba(X_f)[0]
                top3_idx = np.argsort(proba)[::-1][:3]
                fert_pred = self.enc.inverse(top3_idx[0], "fertilizer")
                fert_conf = float(proba[top3_idx[0]])
                fert_top3 = [
                    {"fertilizer": self.enc.inverse(i, "fertilizer"),
                     "confidence": float(proba[i])}
                    for i in top3_idx
                ]
            except Exception as e:
                print(f"[AgriPredictor] Fert predict error: {e}")

        pest = self._select_pesticide(crop_pred, inp)

        return {
            "crop": crop_pred,
            "crop_confidence": round(crop_conf * 100, 1),
            "crop_top3": crop_top3,
            "fertilizer": fert_pred,
            "fertilizer_confidence": round(fert_conf * 100, 1),
            "fertilizer_top3": fert_top3,
            "pesticide": pest["name"],
            "pesticide_target": pest["target"],
            "pesticide_dose": pest["dose"],
            "mode": "ml",
            "inputs": inp,
        }

    def _rule_predict(self, inp: Dict) -> Dict:
        """
        Fallback rule-based prediction when sklearn is not available.
        Mirrors the heuristics from the React widget but in Python.
        """
        N, P, K = inp.get("N", 80), inp.get("P", 40), inp.get("K", 40)
        ph = inp.get("ph", 6.5)
        temp = inp.get("temperature", 25.0)
        humidity = inp.get("humidity", 65.0)
        rainfall = inp.get("rainfall", 100.0)
        season = inp.get("season", "Kharif")
        soil = inp.get("soil_type", "Loamy")

        # Crop scoring
        scores = {}
        def add(crop, s): scores[crop] = scores.get(crop, 0) + s

        if season == "Kharif":
            if N > 70 and humidity > 60 and rainfall > 100: add("Rice", 30)
            if temp > 25 and humidity > 55: add("Cotton", 20)
            if N > 60 and temp > 22: add("Maize", 18)
        elif season == "Rabi":
            if temp < 25 and N > 80: add("Wheat", 30)
            if ph > 6.0 and K > 40: add("Potato", 22)
            if temp < 22: add("Mustard", 18)
        elif season == "Zaid":
            add("Maize", 20); add("Watermelon", 15)

        soil_map = {
            "Clay": [("Rice", 15), ("Wheat", 10)],
            "Alluvial": [("Rice", 15), ("Wheat", 12)],
            "Sandy": [("Groundnut", 15), ("Maize", 10)],
            "Red": [("Groundnut", 12), ("Sorghum", 10)],
            "Black": [("Cotton", 20), ("Sorghum", 10)],
            "Loamy": [("Wheat", 12), ("Tomato", 10), ("Maize", 10)],
        }
        for crop, s in soil_map.get(soil, []):
            add(crop, s)

        sorted_crops = sorted(scores.items(), key=lambda x: -x[1])
        if not sorted_crops:
            sorted_crops = [("Rice", 50), ("Maize", 30), ("Wheat", 20)]

        max_s = sorted_crops[0][1]
        top3 = [{"crop": c, "confidence": min(95, round(50 + (s/max_s)*45))}
                for c, s in sorted_crops[:3]]
        crop_pred = top3[0]["crop"]
        crop_conf = top3[0]["confidence"]

        # Fertilizer
        fert_pred, fert_reason = self._rule_fertilizer(N, P, K, ph, crop_pred)

        pest = self._select_pesticide(crop_pred, inp)

        return {
            "crop": crop_pred,
            "crop_confidence": crop_conf,
            "crop_top3": top3,
            "fertilizer": fert_pred,
            "fertilizer_confidence": 72,
            "fertilizer_top3": [{"fertilizer": fert_pred, "confidence": 72}],
            "fertilizer_reason": fert_reason,
            "pesticide": pest["name"],
            "pesticide_target": pest["target"],
            "pesticide_dose": pest["dose"],
            "mode": "rule",
            "inputs": inp,
        }

    @staticmethod
    def _rule_fertilizer(N, P, K, ph, crop) -> Tuple[str, str]:
        if ph < 5.5:
            return "Lime + DAP", "Acidic soil (pH {:.1f}) requires liming; DAP supplies N and P".format(ph)
        if ph > 8.0:
            return "Gypsum + Urea", "Alkaline soil; gypsum reclaims Na and improves Ca/S availability"
        if N < 60 and P < 30 and K < 40:
            return "NPK 17-17-17", "All three primary macronutrients are deficient"
        if N < 60 and P < 30:
            return "DAP (18-46-0)", "Both nitrogen and phosphorus are below threshold"
        if N < 60:
            return "Urea (46-0-0)", "Nitrogen is the primary limiting macronutrient"
        if P < 30:
            return "SSP (0-16-0)", "Phosphorus deficiency — SSP is the most cost-effective source"
        if K < 40:
            return "MOP (0-0-60)", "Potassium deficiency affects grain filling and disease resistance"
        crop_ferts = {
            "Rice": ("Urea + DAP", "Standard NPK split for transplanted rice"),
            "Wheat": ("Urea + DAP", "Half N basal with full P; remainder N at CRI stage"),
            "Cotton": ("NPK 12-32-16", "Cotton demands high P and K for boll development"),
            "Potato": ("NPK 10-26-26", "Tuber crops need high P+K for starch accumulation"),
            "Maize": ("Urea + DAP", "Maize is a heavy nitrogen feeder; split application reduces loss"),
            "Tomato": ("NPK 12-32-16", "Fruiting crops need balanced P and K"),
        }
        return crop_ferts.get(crop, ("NPK 20-20-20", "Balanced fertilizer for adequate soil nutrition"))

    @staticmethod
    def _select_pesticide(crop: str, inp: Dict) -> Dict:
        humidity = inp.get("humidity", 65)
        season = inp.get("season", "Kharif")

        table = {
            "Rice":     {"name": "Chlorpyrifos 20EC", "target": "Stem borer / Brown planthopper",
                         "dose": "2.5 ml/litre, spray at tillering + panicle initiation"},
            "Wheat":    {"name": "Imidacloprid 17.8SL", "target": "Aphids / Army worm",
                         "dose": "0.25 ml/litre; apply when aphid colony > 10/tiller"},
            "Maize":    {"name": "Spinosad 45SC", "target": "Fall army worm / Stem borer",
                         "dose": "0.75 ml/litre; spray in whorl at 15–20 DAS"},
            "Cotton":   {"name": "Emamectin Benzoate 5SG", "target": "Bollworm / Whitefly",
                         "dose": "0.4 g/litre; apply at squaring and boll formation"},
            "Potato":   {"name": "Thiamethoxam 25WG", "target": "Aphids / Tuber moth",
                         "dose": "0.3 g/litre; apply at 30 and 60 DAS"},
            "Tomato":   {"name": "Spinosad 45SC", "target": "Fruit borer / Whitefly",
                         "dose": "0.5 ml/litre; apply at flowering"},
            "Mustard":  {"name": "Dimethoate 30EC", "target": "Aphids / Painted bug",
                         "dose": "1.5 ml/litre; spray at flowering"},
            "Groundnut":{"name": "Chlorpyrifos 20EC", "target": "Thrips / Leaf miner",
                         "dose": "2.0 ml/litre; apply at 30 and 60 DAS"},
        }
        pest = table.get(crop, table["Rice"]).copy()

        # Add fungicide recommendation when humidity is high
        if humidity > 75:
            fungicide_map = {
                "Rice": "Tricyclazole 75WP @ 0.6 g/L for Blast disease",
                "Wheat": "Propiconazole 25EC @ 1 ml/L for Rust",
                "Potato": "Metalaxyl-M+Mancozeb @ 2.5 g/L for Late blight",
                "Tomato": "Mancozeb 75WP @ 2.5 g/L for Early blight",
                "Maize": "Mancozeb 75WP @ 2.5 g/L for Turcicum blight",
            }
            fung = fungicide_map.get(crop, "Mancozeb 75WP @ 2.5 g/L for fungal diseases")
            pest["fungicide_alert"] = f"High humidity ({humidity}%) — also apply {fung}"

        return pest


# ════════════════════════════════════════════════════════════
#  Explanation prompt builder
#  This is the KEY function — it creates the structured prompt
#  that SageStorm will use to NARRATE the advisory in its own words
# ════════════════════════════════════════════════════════════
def build_slm_explanation_prompt(predictions: Dict, rag_context: str = "") -> str:
    """
    Builds a detailed prompt so SageStorm can:
      1. Explain WHY this crop was recommended
      2. Walk through the fertilizer logic
      3. Give a complete sowing-to-harvest advisory
      4. Sound like an experienced agricultural extension officer

    The SLM receives ML predictions as FACTS and must EXPLAIN + EXPAND them.
    """
    inp = predictions.get("inputs", {})
    crop = predictions["crop"]
    fert = predictions["fertilizer"]
    pest = predictions["pesticide"]
    crop_conf = predictions["crop_confidence"]
    fert_conf = predictions["fertilizer_confidence"]

    N, P, K = inp.get("N", 80), inp.get("P", 40), inp.get("K", 40)
    ph = inp.get("ph", 6.5)
    temp = inp.get("temperature", 25.0)
    humidity = inp.get("humidity", 65.0)
    rainfall = inp.get("rainfall", 100.0)
    season = inp.get("season", "Kharif")
    soil = inp.get("soil_type", "Loamy")
    region = inp.get("region", "East")
    irrigation = inp.get("irrigation", "Sprinkler")
    farm_size = inp.get("farm_size", 2)

    # Build alternative crops string
    top3_str = ""
    if predictions.get("crop_top3"):
        alts = [f"{x['crop']} ({x['confidence']:.0f}%)"
                for x in predictions["crop_top3"][:3]]
        top3_str = ", ".join(alts)

    context_block = f"\n[Knowledge Base Context]\n{rag_context}\n" if rag_context.strip() else ""

    prompt = f"""### Instruction:
You are SageStorm, an expert agricultural advisory AI for Indian farmers. \
Using the ML predictions and farmer's field data provided below, give a \
complete, farmer-friendly advisory. Explain each decision clearly. \
Use simple language that a farmer can understand and act on.

### Input:
[ML Predictions — these are FACTS, explain them]
- Recommended crop    : {crop}  (model confidence: {crop_conf}%)
- Recommended fertilizer: {fert}  (confidence: {fert_conf}%)
- Key pest/pesticide  : {predictions.get('pesticide_target','N/A')} → {pest} @ {predictions.get('pesticide_dose','as directed')}
- Alternative crops   : {top3_str}

[Farmer's Field Data]
- Soil type     : {soil}
- Soil pH       : {ph}  {"(acidic — needs lime)" if ph < 5.8 else "(alkaline — needs gypsum)" if ph > 7.8 else "(good range)"}
- Nitrogen (N)  : {N} kg/ha  {"(low — must add N fertilizer)" if N < 60 else "(adequate)"}
- Phosphorus (P): {P} kg/ha  {"(deficient)" if P < 30 else "(adequate)"}
- Potassium (K) : {K} kg/ha  {"(low)" if K < 40 else "(adequate)"}
- Temperature   : {temp}°C
- Humidity      : {humidity}%  {"(high — fungal disease risk elevated)" if humidity > 75 else ""}
- Rainfall      : {rainfall} mm
- Season        : {season}
- Region        : {region} India
- Irrigation    : {irrigation}
- Farm size     : {farm_size} acres
{context_block}

### Response:
"""
    return prompt


# ════════════════════════════════════════════════════════════
#  Quick self-test
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    predictor = AgriPredictor()

    test_input = {
        "soil_type": "Clay", "ph": 6.5, "soil_moisture": 35.0,
        "organic_carbon": 0.8, "ec": 0.5,
        "N": 90, "P": 40, "K": 50,
        "temperature": 28.0, "humidity": 75.0, "rainfall": 120.0,
        "season": "Kharif", "irrigation": "Sprinkler",
        "region": "East", "farm_size": 3,
    }

    preds = predictor.predict(test_input)
    print(json.dumps({k: v for k, v in preds.items() if k != "inputs"}, indent=2))

    prompt = build_slm_explanation_prompt(preds, rag_context="Rice grows best in flooded conditions with 5 cm standing water.")
    print("\n--- SLM Prompt ---")
    print(prompt[:600], "...")