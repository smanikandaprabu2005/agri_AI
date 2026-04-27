"""
decision_models/training/generate_training_data.py
====================================================
Synthetic Training Data Generator for Decision Models

Generates labelled JSONL datasets for:
  1. crop_train.jsonl      — CropModel training data
  2. fertilizer_train.jsonl — FertilizerModel training data (optional ML)
  3. pesticide_train.jsonl  — PesticideModel training data (optional ML)

Data is generated from the ICAR rule tables in the model files,
augmented with random noise to train robust classifiers.

Usage:
  python decision_models/training/generate_training_data.py
  python decision_models/training/generate_training_data.py --n 5000
"""

import json
import random
import os
import argparse
from pathlib import Path

random.seed(42)

OUTPUT_DIR = Path(__file__).parent

SOILS   = ["loamy", "clay", "sandy", "black", "red", "alluvial"]
SEASONS = ["kharif", "rabi", "zaid"]

# Crop truth table derived from CROP_RULES in crop_model.py
CROP_TRUTH = {
    ("loamy",    "kharif"): "Rice",
    ("clay",     "kharif"): "Rice",
    ("sandy",    "kharif"): "Groundnut",
    ("black",    "kharif"): "Cotton",
    ("red",      "kharif"): "Groundnut",
    ("alluvial", "kharif"): "Rice",
    ("loamy",    "rabi"):   "Wheat",
    ("clay",     "rabi"):   "Wheat",
    ("sandy",    "rabi"):   "Barley",
    ("black",    "rabi"):   "Chickpea",
    ("red",      "rabi"):   "Lentil",
    ("alluvial", "rabi"):   "Wheat",
    ("loamy",    "zaid"):   "Cucumber",
    ("sandy",    "zaid"):   "Watermelon",
    ("clay",     "zaid"):   "Moong",
    ("black",    "zaid"):   "Moong",
    ("red",      "zaid"):   "Cowpea",
    ("alluvial", "zaid"):   "Cucumber",
}

TEMP_RANGES = {
    "kharif": (26, 38),
    "rabi":   (10, 25),
    "zaid":   (28, 42),
}


def jitter(base: float, pct: float = 0.25) -> float:
    """Add ±pct% random noise to a value."""
    delta = base * pct
    return round(base + random.uniform(-delta, delta), 1)


def generate_crop_sample(soil: str, season: str) -> dict:
    tmin, tmax = TEMP_RANGES[season]
    temp  = round(random.uniform(tmin, tmax), 1)
    n     = round(random.uniform(20, 130), 1)
    p     = round(random.uniform(10, 80),  1)
    k     = round(random.uniform(10, 90),  1)
    label = CROP_TRUTH.get((soil, season), "Rice")

    # Edge case: high N favours cereals
    if n > 100 and season == "kharif" and soil in ("loamy", "alluvial", "clay"):
        label = "Rice"
    # Low N favours legumes
    if n < 25 and season == "kharif":
        label = "Soybean" if soil in ("loamy", "black") else "Cowpea"
    if n < 25 and season == "rabi":
        label = "Chickpea"

    return {
        "soil_type":   soil,
        "season":      season,
        "temperature": temp,
        "nitrogen":    n,
        "phosphorus":  p,
        "potassium":   k,
        "crop_label":  label,
    }


def generate_crop_dataset(n_samples: int = 3000) -> list:
    samples = []
    # Ensure every soil×season combo is represented
    for soil in SOILS:
        for season in SEASONS:
            for _ in range(n_samples // (len(SOILS) * len(SEASONS))):
                samples.append(generate_crop_sample(soil, season))
    random.shuffle(samples)
    return samples


def generate_fertilizer_sample(crop: str, n: float, p: float, k: float) -> dict:
    """Generate a fertilizer recommendation label."""
    from sys import path as syspath
    syspath.insert(0, str(Path(__file__).parent.parent.parent))
    from decision_models.fertilizer_model import FertilizerModel
    model = FertilizerModel()
    pred  = model.predict({
        "crop_name":   crop,
        "nitrogen":    n,
        "phosphorus":  p,
        "potassium":   k,
    })
    return {
        "crop_name":       crop,
        "nitrogen":        n,
        "phosphorus":      p,
        "potassium":       k,
        "fertilizer_name": pred.fertilizer_name,
        "dosage":          pred.dosage,
        "stage":           pred.application_stage,
    }


def generate_pesticide_sample(crop: str, season: str, temp: float, humid: float) -> dict:
    from sys import path as syspath
    syspath.insert(0, str(Path(__file__).parent.parent.parent))
    from decision_models.pesticide_model import PesticideModel
    model = PesticideModel()
    pred  = model.predict({
        "crop_name":   crop,
        "season":      season,
        "temperature": temp,
        "humidity":    humid,
    })
    return {
        "crop_name":     crop,
        "season":        season,
        "temperature":   temp,
        "humidity":      humid,
        "pest_label":    pred.pest_name,
        "pesticide":     pred.pesticide_name,
        "dosage":        pred.pesticide_dosage if hasattr(pred, "pesticide_dosage") else pred.dosage,
    }


def save_jsonl(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} samples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=3000,
                        help="Approx samples per dataset")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    out = Path(args.output)

    print("\n[DataGen] Generating crop training data...")
    crop_data = generate_crop_dataset(args.n)
    save_jsonl(crop_data, out / "crop_train.jsonl")

    print("[DataGen] Generating fertilizer training data...")
    crops   = ["Rice", "Wheat", "Maize", "Tomato", "Potato", "Cotton",
               "Mustard", "Groundnut", "Onion", "Chickpea", "Soybean"]
    fert_data = []
    per_crop  = max(args.n // len(crops), 20)
    for crop in crops:
        for _ in range(per_crop):
            n = round(random.uniform(15, 130), 1)
            p = round(random.uniform(10, 80),  1)
            k = round(random.uniform(10, 90),  1)
            fert_data.append(generate_fertilizer_sample(crop, n, p, k))
    save_jsonl(fert_data, out / "fertilizer_train.jsonl")

    print("[DataGen] Generating pesticide training data...")
    pest_data = []
    crop_season_pairs = [
        ("Rice",    "kharif"), ("Rice",    "rabi"),
        ("Wheat",   "rabi"),   ("Tomato",  "rabi"),
        ("Potato",  "rabi"),   ("Cotton",  "kharif"),
        ("Lemon",   "kharif"), ("Lemon",   "rabi"),
        ("Groundnut","kharif"),("Onion",   "rabi"),
    ]
    per_pair = max(args.n // len(crop_season_pairs), 20)
    for crop, season in crop_season_pairs:
        tmin, tmax = TEMP_RANGES[season]
        for _ in range(per_pair):
            temp  = round(random.uniform(tmin, tmax), 1)
            humid = round(random.uniform(40, 95), 1)
            pest_data.append(generate_pesticide_sample(crop, season, temp, humid))
    save_jsonl(pest_data, out / "pesticide_train.jsonl")

    print(f"\n[DataGen] ✓ Done — {args.n} samples each in {out}/")
    print("To train the ML crop model:")
    print("  python decision_models/training/train_models.py")


if __name__ == "__main__":
    main()
