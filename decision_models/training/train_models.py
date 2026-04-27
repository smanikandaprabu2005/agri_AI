"""
decision_models/training/train_models.py
==========================================
Train ML classifiers for the three decision models.

Models trained:
  1. CropModel        — RandomForest, ~200 trees
  2. (Fertilizer/Pesticide stay rule-based — DB is reliable enough)

Usage:
  python decision_models/training/train_models.py
  python decision_models/training/train_models.py --data crop_train.jsonl
  python decision_models/training/train_models.py --eval   (show evaluation)

Requirements:
  pip install scikit-learn
"""

import json
import os
import sys
import pickle
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

TRAINING_DIR = Path(__file__).parent
SAVED_DIR    = Path(__file__).parent.parent / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str) -> list:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def train_crop_model(data_path: str, save: bool = True) -> dict:
    """
    Train a RandomForestClassifier on labelled crop data.
    Returns evaluation metrics dict.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        import numpy as np
    except ImportError:
        print("[Train] scikit-learn not installed. Run: pip install scikit-learn")
        return {}

    print(f"\n[Train] Loading crop training data from {data_path}...")
    records = load_jsonl(data_path)
    if not records:
        print("[Train] No data found.")
        return {}

    # Filter records with valid labels
    records = [r for r in records if r.get("crop_label")]
    print(f"[Train] {len(records):,} labelled samples")

    # Encode categorical
    soil_enc   = LabelEncoder()
    season_enc = LabelEncoder()
    soil_enc.fit([r["soil_type"] for r in records])
    season_enc.fit([r["season"]  for r in records])

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

    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        rare_classes = [c for c, ct in class_counts.items() if ct < 2]
        print(
            f"[Train] Warning: classes with too few samples for stratified split: "
            f"{rare_classes}"
        )
        print("[Train] Falling back to a random split without stratification.")
        split_kwargs = dict(test_size=0.15, random_state=42)
    else:
        split_kwargs = dict(test_size=0.15, random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    print(f"[Train] Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"[Train] Classes: {sorted(set(y))}")

    # Train RandomForest
    print("[Train] Training RandomForest (200 trees)...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred  = clf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)

    cv_scores = None
    if min_class_count >= 2:
        cv = min(5, min_class_count)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"\n[Train] ── Evaluation ─────────────────────────")
    print(f"  Test accuracy : {acc*100:.2f}%")
    if cv_scores is not None:
        print(f"  CV accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    else:
        print("  CV accuracy   : skipped (too few samples in at least one class)")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Feature importance
    feat_names = ["soil", "season", "nitrogen", "phosphorus", "potassium", "temperature"]
    importances = clf.feature_importances_
    print("[Train] Feature importances:")
    for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 30)
        print(f"  {name:12s} {imp:.3f}  {bar}")

    if save:
        model_path   = SAVED_DIR / "crop_model.pkl"
        encoder_path = SAVED_DIR / "crop_encoder.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(encoder_path, "wb") as f:
            pickle.dump({"soil": soil_enc, "season": season_enc}, f)
        print(f"\n[Train] Model  → {model_path}")
        print(f"[Train] Encoder → {encoder_path}")

    return {
        "test_accuracy":    round(acc, 4),
        "cv_mean":          round(cv_scores.mean(), 4) if cv_scores is not None else None,
        "cv_std":           round(cv_scores.std(), 4) if cv_scores is not None else None,
        "n_classes":        len(set(y)),
        "n_train":          len(X_train),
        "n_test":           len(X_test),
    }


def evaluate_crop_model(data_path: str):
    """Quick evaluation of the saved model."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from decision_models.crop_model import CropModel

    model   = CropModel()
    records = load_jsonl(data_path)
    records = [r for r in records if r.get("crop_label")]

    correct = 0
    for r in records[:500]:
        pred = model.predict(r)
        if pred.crop_name.lower() == r["crop_label"].lower():
            correct += 1

    print(f"\n[Eval] Crop model accuracy on {min(500, len(records))} samples: "
          f"{correct/min(500, len(records))*100:.1f}%")

    # Show a few predictions
    print("\n[Eval] Sample predictions:")
    for r in records[:5]:
        pred = model.predict(r)
        ok   = "✓" if pred.crop_name.lower() == r["crop_label"].lower() else "✗"
        print(f"  {ok}  soil={r['soil_type']:8s} season={r['season']:7s} "
              f"N={r['nitrogen']:.0f}  → {pred.crop_name:12s} "
              f"(true: {r['crop_label']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=str(TRAINING_DIR / "crop_train.jsonl"))
    parser.add_argument("--eval",   action="store_true", help="Evaluate saved model")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    # Check if training data exists; if not, generate it
    if not Path(args.data).exists():
        print(f"[Train] Training data not found at {args.data}")
        print("[Train] Generating synthetic data first...")
        from generate_training_data import main as gen_main
        gen_main()

    if args.eval:
        evaluate_crop_model(args.data)
    else:
        metrics = train_crop_model(args.data, save=not args.no_save)
        if metrics:
            print(f"\n[Train] ✓ CropModel trained. Test accuracy: "
                  f"{metrics['test_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
