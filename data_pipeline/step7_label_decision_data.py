"""
data_pipeline/step7_label_decision_data.py
============================================
STEP 7 — Label Decision Model Training Data

Reads your existing final_agriculture_training_dataset.jsonl and extracts
labelled examples for the three decision models by parsing the
instruction/output text with heuristics and the rule databases.

Output files:
  decision_models/training/crop_train.jsonl
  decision_models/training/fertilizer_train.jsonl
  decision_models/training/pesticide_train.jsonl

Usage:
  python data_pipeline/step7_label_decision_data.py
  python data_pipeline/step7_label_decision_data.py --input path/to/dataset.jsonl
"""

import json
import re
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_FILE = "data_pipeline/data/raw/final_agriculture_training_dataset.jsonl"
OUTPUT_DIR = Path("decision_models/training")

# ── Crop name extractor from text ─────────────────────────────
_CROPS_SORTED = sorted([
    "rice","paddy","wheat","maize","corn","sorghum","jowar","bajra","millet",
    "barley","chickpea","gram","lentil","cowpea","moong","urad","arhar",
    "soybean","mustard","rapeseed","sunflower","groundnut","peanut","sesame",
    "potato","tomato","onion","garlic","chilli","chili","pepper","brinjal",
    "cauliflower","cabbage","okra","bhindi","cucumber","pumpkin",
    "mango","banana","lemon","orange","papaya","guava","coconut",
    "watermelon","muskmelon","cotton","sugarcane","jute","tea","coffee",
    "turmeric","ginger","cardamom","cashew","arecanut",
], key=len, reverse=True)

_CROP_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _CROPS_SORTED) + r")\b", re.I
)

_SOIL_RE   = re.compile(r"\b(loamy?|sandy|clay|black|red|alluvial)\s+soil", re.I)
_SEASON_RE = re.compile(r"\b(kharif|rabi|zaid)\b", re.I)
_N_RE      = re.compile(r"\b(?:nitrogen|urea)[^0-9]*(\d+)\s*kg", re.I)
_P_RE      = re.compile(r"\b(?:phosphorus|dap|phosphate)[^0-9]*(\d+)\s*kg", re.I)
_K_RE      = re.compile(r"\b(?:potassium|mop|muriate)[^0-9]*(\d+)\s*kg", re.I)
_TEMP_RE   = re.compile(r"(\d+)\s*°?C")
_PEST_RE   = re.compile(
    r"\b(aphid|stem\s*borer|boll\s*worm|whitefly|leaf\s*folder|blast|"
    r"blight|rust|mildew|caterpillar|thrip|mite|hopper|fruit\s*borer)\b", re.I
)
_PESTICIDE_RE = re.compile(
    r"\b(malathion|chlorpyrifos|imidacloprid|thiamethoxam|mancozeb|"
    r"carbendazim|propiconazole|cartap|emamectin|bt|neem|triazophos|"
    r"profenofos|spinosad|copper|streptomycin)\b", re.I
)


def extract_crop_sample(record: dict):
    text = f"{record.get('instruction','')} {record.get('output','')}"
    m_crop  = _CROP_RE.search(text)
    m_soil  = _SOIL_RE.search(text)
    m_season = _SEASON_RE.search(text)
    m_n     = _N_RE.search(text)
    m_p     = _P_RE.search(text)
    m_k     = _K_RE.search(text)
    m_temp  = _TEMP_RE.search(text)

    if not m_crop or not m_soil or not m_season:
        return None

    return {
        "crop_label":  m_crop.group(1).title(),
        "soil_type":   m_soil.group(1).lower().rstrip("y") or "loamy",
        "season":      m_season.group(1).lower(),
        "nitrogen":    float(m_n.group(1)) if m_n else 50.0,
        "phosphorus":  float(m_p.group(1)) if m_p else 30.0,
        "potassium":   float(m_k.group(1)) if m_k else 30.0,
        "temperature": float(m_temp.group(1)) if m_temp else 28.0,
    }


def extract_fertilizer_sample(record: dict):
    text = f"{record.get('instruction','')} {record.get('output','')}"
    m_crop = _CROP_RE.search(text)
    m_n    = _N_RE.search(text)
    m_p    = _P_RE.search(text)
    m_k    = _K_RE.search(text)

    if not m_crop or not (m_n or m_p or m_k):
        return None

    # Extract fertilizer name from output text
    fert_keywords = re.findall(
        r"\b(urea|dap|mop|ssp|potassium\s+sulphate|ammonium\s+phosphate|"
        r"npk\s+[\d:]+|complex\s+fertilizer)\b",
        record.get("output", ""), re.I
    )

    return {
        "crop_name":   m_crop.group(1).title(),
        "nitrogen":    float(m_n.group(1)) if m_n else 50.0,
        "phosphorus":  float(m_p.group(1)) if m_p else 30.0,
        "potassium":   float(m_k.group(1)) if m_k else 30.0,
        "fertilizer_names": list(set(f.lower() for f in fert_keywords)),
    }


def extract_pesticide_sample(record: dict):
    text = f"{record.get('instruction','')} {record.get('output','')}"
    m_crop    = _CROP_RE.search(text)
    m_pest    = _PEST_RE.search(text)
    m_chem    = _PESTICIDE_RE.search(text)
    m_season  = _SEASON_RE.search(text)
    m_temp    = _TEMP_RE.search(text)

    if not m_crop or not (m_pest or m_chem):
        return None

    return {
        "crop_name":    m_crop.group(1).title(),
        "pest_label":   m_pest.group(1).title() if m_pest else "General Pests",
        "pesticide":    m_chem.group(1).title() if m_chem else "Unknown",
        "season":       m_season.group(1).lower() if m_season else "kharif",
        "temperature":  float(m_temp.group(1)) if m_temp else 28.0,
    }


def process_dataset(input_path: str, output_dir: Path, max_records: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_samples  = []
    fert_samples  = []
    pest_samples  = []
    total         = 0
    errors        = 0

    print(f"\n[Step 7] Reading {input_path} ...")

    with open(input_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            total += 1
            if max_records and total > max_records:
                break

            cs = extract_crop_sample(record)
            if cs:
                crop_samples.append(cs)

            fs = extract_fertilizer_sample(record)
            if fs:
                fert_samples.append(fs)

            ps = extract_pesticide_sample(record)
            if ps:
                pest_samples.append(ps)

            if total % 50000 == 0:
                print(f"  Processed {total:,} records  "
                      f"(crop={len(crop_samples):,}  fert={len(fert_samples):,}  "
                      f"pest={len(pest_samples):,})")

    print(f"\n[Step 7] Total processed: {total:,}  Errors: {errors:,}")

    # Save
    def save(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {len(data):,} samples → {path}")

    save(crop_samples,  output_dir / "crop_train.jsonl")
    save(fert_samples,  output_dir / "fertilizer_train.jsonl")
    save(pest_samples,  output_dir / "pesticide_train.jsonl")

    return {
        "crop_samples":  len(crop_samples),
        "fert_samples":  len(fert_samples),
        "pest_samples":  len(pest_samples),
        "total_records": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_FILE)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--max",    type=int, default=0,
                        help="Max records to process (0=all)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[Step 7] Input not found: {args.input}")
        print("  Place your dataset at: data/raw/final_agriculture_training_dataset.jsonl")
        print("  Or use --input to specify a different path")
        return

    stats = process_dataset(args.input, Path(args.output), args.max)

    print(f"\n[Step 7] ── Summary ────────────────────────────")
    print(f"  Crop samples     : {stats['crop_samples']:,}")
    print(f"  Fertilizer samples: {stats['fert_samples']:,}")
    print(f"  Pesticide samples: {stats['pest_samples']:,}")
    print(f"\n[Step 7] ✓ Done. Next:")
    print("  python decision_models/training/train_models.py")


if __name__ == "__main__":
    main()
