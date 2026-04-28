"""
train/train_agri_ml.py
========================
Train the agricultural ML models (AgriPredictor) on your two datasets.

Usage:
    python train/train_agri_ml.py
    python train/train_agri_ml.py --soil data/soil_dataset.csv --npk data/npk_dataset.csv
    python train/train_agri_ml.py --eval      (evaluate after training)

Dataset column mapping
-----------------------
Dataset 1 (soil CSV — Image 1):
  Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity,
  Nitrogen_N, Phosphorus_P, Potassium_K, Temperature, Humidity, Rainfall,
  Crop_Type (TARGET for crop model), Season, Irrigation_Method, Previous_Crop,
  Region, Fertilizer_Recommended (also can be TARGET for fert model), Yield_Last_Season

Dataset 2 (NPK CSV — Image 2):
  N, P, K, temperature, humidity, ph, rainfall,
  label (crop name), fertilizer (TARGET for fert model), yield
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agri_predictor import AgriPredictor

# ── Default paths (match your actual filenames) ────────────────
DEFAULT_SOIL = os.path.join("data", "soil_dataset.csv")
DEFAULT_NPK  = os.path.join("data", "npk_dataset.csv")


# ── Column normaliser ─────────────────────────────────────────
def normalise_soil_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles various column naming conventions in the soil dataset.
    The CSV from Image 1 uses different spacing/capitalisation.
    """
    rename_map = {
        # Common variations → standard names used by AgriPredictor
        "Soil_Type":               "Soil_Type",
        "SoilType":                "Soil_Type",
        "soil type":               "Soil_Type",
        "Soil_pH":                 "Soil_pH",
        "pH":                      "Soil_pH",
        "Soil_Moist":              "Soil_Moisture",
        "Soil_Moisture":           "Soil_Moisture",
        "Organic_C":               "Organic_Carbon",
        "Organic_Carbon":          "Organic_Carbon",
        "Electrical_Conductivity": "Electrical_Conductivity",
        "Electrical_":             "Electrical_Conductivity",
        "Electrical_C":            "Electrical_Conductivity",
        "Nitrogen_N":              "Nitrogen_N",
        "Nitrogen":                "Nitrogen_N",
        "N":                       "Nitrogen_N",
        "Phosphorus_P":            "Phosphorus_P",
        "Phosphorus":              "Phosphorus_P",
        "P":                       "Phosphorus_P",
        "Potassium_K":             "Potassium_K",
        "Potassium":               "Potassium_K",
        "K":                       "Potassium_K",
        "Temperature":             "Temperature",
        "Temp":                    "Temperature",
        "Humidity":                "Humidity",
        "Rainfall":                "Rainfall",
        "Crop_Type":               "Crop_Type",
        "CropType":                "Crop_Type",
        "Crop_Grow":               "Crop_Growth_Stage",
        "Season":                  "Season",
        "Irrigation_Method":       "Irrigation_Method",
        "Irrigation_":             "Irrigation_Method",
        "Previous_Crop":           "Previous_Crop",
        "Previous_C":              "Previous_Crop",
        "Region":                  "Region",
        "Fertilizer_Recommended":  "Fertilizer_Recommended",
        "Fertilizer_R":            "Fertilizer_Recommended",
        "Recommended_Fertilizer":  "Fertilizer_Recommended",
        "Yield_Last_Season":       "Yield_Last_Season",
        "Yield_Last":              "Yield_Last_Season",
    }

    # Apply prefix-based matching for truncated column names
    new_cols = {}
    for col in df.columns:
        # Direct match
        if col in rename_map:
            new_cols[col] = rename_map[col]
            continue
        # Prefix match
        for key, val in rename_map.items():
            if col.startswith(key[:8]) and col not in new_cols:
                new_cols[col] = val
                break

    df = df.rename(columns=new_cols)

    # Collapse duplicate columns produced by malformed or repeated CSV headers.
    if df.columns.duplicated().any():
        deduped = pd.DataFrame()
        for col in df.columns.unique():
            cols = df.loc[:, df.columns == col]
            if cols.shape[1] > 1:
                deduped[col] = cols.bfill(axis=1).iloc[:, 0]
            else:
                deduped[col] = cols.iloc[:, 0]
        df = deduped

    return df


def normalise_npk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise NPK dataset columns."""
    rename_map = {
        "label": "label",
        "Label": "label",
        "crop":  "label",
        "Crop":  "label",
        "fertilizer": "fertilizer",
        "Fertilizer": "fertilizer",
        "yield": "yield",
        "Yield": "yield",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


# ── EDA helpers ───────────────────────────────────────────────
def print_dataset_info(df: pd.DataFrame, name: str):
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Nulls   : {df.isnull().sum().sum()}")
    if "Crop_Type" in df.columns:
        print(f"  Crop classes  : {df['Crop_Type'].nunique()}")
        print(f"  Top crops     : {df['Crop_Type'].value_counts().head(5).to_dict()}")
    if "label" in df.columns:
        print(f"  Crop labels   : {df['label'].nunique()}")
        print(f"  Top labels    : {df['label'].value_counts().head(5).to_dict()}")
    if "fertilizer" in df.columns:
        fc = df['fertilizer'].nunique()
        print(f"  Fertilizer classes: {fc}")
        if fc <= 20:
            print(f"  Fertilizers   : {df['fertilizer'].value_counts().to_dict()}")
        else:
            print(f"  Fertilizer sample preview: {df['fertilizer'].head(10).tolist()}")
    if "Fertilizer_Recommended" in df.columns:
        frc = df['Fertilizer_Recommended'].nunique()
        print(f"  Recommended fert classes: {frc}")
        if frc <= 20:
            print(f"  Recommended fert  : {df['Fertilizer_Recommended'].value_counts().to_dict()}")
        else:
            print(f"  Recommended fert sample preview: {df['Fertilizer_Recommended'].head(10).tolist()}")
    print()


def evaluate_models(predictor: AgriPredictor, soil_df: pd.DataFrame = None,
                    npk_df: pd.DataFrame = None):
    """Quick evaluation on held-out samples."""
    print("\n[Eval] Running evaluation...")

    test_cases = [
        {"soil_type": "Clay", "ph": 6.5, "N": 90, "P": 40, "K": 50,
         "temperature": 28, "humidity": 75, "rainfall": 120,
         "season": "Kharif", "region": "East", "farm_size": 3},
        {"soil_type": "Loamy", "ph": 7.0, "N": 120, "P": 60, "K": 40,
         "temperature": 18, "humidity": 55, "rainfall": 60,
         "season": "Rabi", "region": "North", "farm_size": 5},
        {"soil_type": "Black", "ph": 7.5, "N": 50, "P": 25, "K": 80,
         "temperature": 32, "humidity": 50, "rainfall": 80,
         "season": "Kharif", "region": "Central", "farm_size": 10},
        {"soil_type": "Red", "ph": 5.5, "N": 40, "P": 20, "K": 30,
         "temperature": 30, "humidity": 65, "rainfall": 200,
         "season": "Kharif", "region": "South", "farm_size": 2},
    ]

    expected = ["Rice", "Wheat", "Cotton", "Rice"]  # rough expected

    print(f"\n{'─'*55}")
    print(f"  {'Input':<25} {'Predicted Crop':<15} {'Conf%':<8} {'Fertilizer'}")
    print(f"{'─'*55}")

    for i, tc in enumerate(test_cases):
        preds = predictor.predict(tc)
        print(f"  {tc['soil_type']+'  pH='+str(tc['ph']):<25} "
              f"{preds['crop']:<15} "
              f"{preds['crop_confidence']:<8} "
              f"{preds['fertilizer']}")

    print(f"{'─'*55}")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train AgriPredictor ML models")
    parser.add_argument("--soil", default=DEFAULT_SOIL,
                        help="Path to soil/crop CSV dataset")
    parser.add_argument("--npk",  default=DEFAULT_NPK,
                        help="Path to NPK/fertilizer CSV dataset")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after training")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even if saved models exist")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  SageStorm V3 — Agricultural ML Model Training")
    print(f"{'='*55}")

    predictor = AgriPredictor()

    # Check if retraining is needed
    if predictor.trained and not args.force_retrain:
        print("[Train] Saved models already exist. Use --force_retrain to retrain.")
        if args.eval:
            evaluate_models(predictor)
        return

    soil_loaded = False
    npk_loaded  = False

    # ── Load and train from soil dataset ──────────────────────
    if Path(args.soil).exists():
        print(f"\n[Train] Loading soil dataset: {args.soil}")
        df_soil = pd.read_csv(args.soil)
        df_soil = normalise_soil_columns(df_soil)
        print_dataset_info(df_soil, "Soil Dataset")

        # The Fertilizer_Recommended column can also train the fert model
        # if no separate NPK dataset is provided
        if "Fertilizer_Recommended" in df_soil.columns and "fertilizer" not in df_soil.columns:
            df_soil["fertilizer"] = df_soil["Fertilizer_Recommended"]
        if "Crop_Type" in df_soil.columns and "label" not in df_soil.columns:
            df_soil["label"] = df_soil["Crop_Type"]

        # FIX 1: DISABLED — Soil dataset gives only 15% crop accuracy
        # predictor.train_on_soil_dataset(df_soil)
        # Reason: NPK dataset provides much better crop predictions (94%+)
        #         Training both models on NPK dataset instead (see below)
        soil_loaded = True
    else:
        print(f"[Train] Soil dataset not found: {args.soil}")
        print(f"  → Place your CSV at: {args.soil}")
        print(f"  → Columns expected: Soil_Type, Soil_pH, N, P, K, "
              f"Temperature, Humidity, Rainfall, Crop_Type, Season, Fertilizer_Recommended")

    # ── Load and train from NPK dataset ───────────────────────
    if Path(args.npk).exists():
        print(f"\n[Train] Loading NPK dataset: {args.npk}")
        df_npk = pd.read_csv(args.npk)
        df_npk = normalise_npk_columns(df_npk)
        print_dataset_info(df_npk, "NPK Dataset")
        predictor.train_on_npk_dataset(df_npk)
        npk_loaded = True
    else:
        print(f"[Train] NPK dataset not found: {args.npk}")
        print(f"  → Place your CSV at: {args.npk}")
        print(f"  → Columns expected: N, P, K, temperature, humidity, ph, rainfall, label, fertilizer")

    if not soil_loaded and not npk_loaded:
        print("\n[Train] No training data found. Using rule-based predictor.")
        print("  To enable ML predictions:")
        print(f"  1. Save your soil CSV as: {args.soil}")
        print(f"  2. Save your NPK CSV as:  {args.npk}")
        print(f"  3. Re-run: python train/train_agri_ml.py")
    else:
        predictor.save()
        print(f"\n[Train] Models saved to: {predictor._path('crop_model.pkl')}")

    if args.eval:
        evaluate_models(predictor)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Next: python api/server_v3.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()