"""
scripts/setup_pipeline.py
==========================
One-command setup for the Strom Sage v3 pipeline.

What this does:
  1. Checks all dependencies
  2. Generates synthetic training data for decision models
  3. Trains CropModel (RandomForest) if scikit-learn is available
  4. Verifies the full pipeline runs end-to-end
  5. Prints a sample advisory report

Usage:
  python scripts/setup_pipeline.py
  python scripts/setup_pipeline.py --skip_train   (skip ML training)
  python scripts/setup_pipeline.py --from_dataset  (use real JSONL data)
"""

import sys
import os
import argparse
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  🌾  Strom Sage AI v3 — Pipeline Setup                      ║
║  Multi-Model Advisory: Crop + Fertilizer + Pesticide + LLM  ║
╚══════════════════════════════════════════════════════════════╝
"""

SAMPLE_INPUT = {
    "query":       "Give me a complete farm advisory for kharif season.",
    "location":    "Guwahati, Assam",
    "soil_type":   "loamy",
    "season":      "kharif",
    "temperature": 32,
    "humidity":    80,
    "nitrogen":    85,
    "phosphorus":  42,
    "potassium":   35,
}


def check_dep(module: str, pip_name: str = None) -> bool:
    try:
        __import__(module)
        print(f"  ✓  {module}")
        return True
    except ImportError:
        name = pip_name or module
        print(f"  ✗  {module}  (install: pip install {name})")
        return False


def step_check_dependencies():
    print("\n── Checking dependencies ──────────────────────────")
    required_ok = all([
        check_dep("json"),
        check_dep("re"),
        check_dep("pickle"),
        check_dep("concurrent.futures"),
        check_dep("dataclasses"),
    ])
    optional_ok = {
        "sklearn":   check_dep("sklearn", "scikit-learn"),
        "torch":     check_dep("torch"),
        "fastapi":   check_dep("fastapi"),
        "uvicorn":   check_dep("uvicorn"),
        "gradio":    check_dep("gradio"),
        "anthropic": check_dep("anthropic"),
    }
    print(f"\n  Core deps  : {'OK' if required_ok else 'MISSING'}")
    print(f"  ML (crop)  : {'OK' if optional_ok['sklearn'] else 'Not available — rule-based only'}")
    print(f"  SageStorm  : {'OK' if optional_ok['torch'] else 'Not available — LLM disabled'}")
    print(f"  API server : {'OK' if optional_ok['fastapi'] else 'Not available'}")
    return required_ok


def step_generate_training_data():
    print("\n── Generating synthetic training data ─────────────")
    from decision_models.training.generate_training_data import (
        generate_crop_dataset, save_jsonl
    )
    from pathlib import Path

    out = Path("decision_models/training")
    crop_data = generate_crop_dataset(3000)
    save_jsonl(crop_data, out / "crop_train.jsonl")
    print(f"  Generated {len(crop_data):,} crop training samples")


def step_train_crop_model():
    print("\n── Training CropModel (RandomForest) ───────────────")
    try:
        from decision_models.training.train_models import train_crop_model
        from pathlib import Path
        data_path = "decision_models/training/crop_train.jsonl"
        if not Path(data_path).exists():
            step_generate_training_data()
        metrics = train_crop_model(data_path, save=True)
        if metrics:
            print(f"\n  ✓  CropModel trained")
            print(f"     Test accuracy : {metrics['test_accuracy']*100:.1f}%")
            print(f"     CV accuracy   : {metrics['cv_mean']*100:.1f}% ± {metrics['cv_std']*100:.1f}%")
            print(f"     Classes       : {metrics['n_classes']}")
        return True
    except ImportError:
        print("  scikit-learn not installed — skipping ML training")
        print("  The CropModel will run in rule-based mode (still accurate)")
        return False
    except Exception as e:
        print(f"  Training error: {e}")
        return False


def step_run_pipeline_test():
    print("\n── Running full pipeline test ─────────────────────")
    from pipeline.advisory_pipeline import AdvisoryPipeline

    pipeline = AdvisoryPipeline(llm_backend="structured_fallback")
    result   = pipeline.run(SAMPLE_INPUT)

    print(f"\n  ✓  Pipeline completed in {result['latency_ms']:.0f} ms")
    print(f"     Crop      : {result['crop']['crop_name']} ({result['crop']['confidence']*100:.0f}%)")
    print(f"     Fertilizer: {result['fertilizer']['fertilizer_name'][:50]}")
    print(f"     Pest      : {result['pesticide']['pest_name']}")
    print(f"     Warnings  : {result['warnings'] or 'None'}")
    print(f"\n── Sample Advisory Output ─────────────────────────")
    print(result["advisory"][:600] + ("..." if len(result["advisory"]) > 600 else ""))
    return True


def step_run_unit_tests():
    print("\n── Running test suite ─────────────────────────────")
    result = subprocess.run(
        [sys.executable, "tests/test_pipeline.py"],
        capture_output=True, text=True
    )
    last_lines = "\n".join(result.stdout.strip().split("\n")[-6:])
    print(last_lines)
    return result.returncode == 0


def print_integration_guide():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Integration with existing SageStorm V2 project             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Copy pipeline files to your project root:               ║
║     cp -r decision_models/ ../                              ║
║     cp -r pipeline/ ../                                     ║
║     cp -r context/ ../                                      ║
║     cp -r prompts/ ../                                      ║
║                                                              ║
║  2. Replace api/server.py with api/pipeline_server.py       ║
║     cp api/pipeline_server.py ../api/server.py              ║
║                                                              ║
║  3. Replace ui/src/StromSageUI.jsx with StromSageV3.jsx     ║
║     cp ui/src/StromSageV3.jsx ../ui/src/StromSageUI.jsx     ║
║                                                              ║
║  4. Start the server:                                        ║
║     python api/pipeline_server.py                           ║
║     # or: uvicorn api.pipeline_server:app --reload          ║
║                                                              ║
║  5. (Optional) Use Anthropic Claude as LLM backend:         ║
║     export STROM_SAGE_LLM=anthropic                         ║
║     export ANTHROPIC_API_KEY=your_key_here                  ║
║     python api/pipeline_server.py                           ║
║                                                              ║
║  API endpoints:                                             ║
║    POST /advisory  — full multi-model report                ║
║    POST /chat      — conversational Q&A (existing mode)     ║
║    GET  /health    — pipeline status                        ║
║    GET  /weather   — weather data                           ║
║    GET  /profile   — farmer profile                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train",    action="store_true")
    parser.add_argument("--skip_tests",    action="store_true")
    parser.add_argument("--from_dataset",  action="store_true",
                        help="Label from real JSONL dataset")
    args = parser.parse_args()

    print(BANNER)

    ok = step_check_dependencies()
    if not ok:
        print("\n[Setup] Core dependencies missing. Exiting.")
        sys.exit(1)

    if args.from_dataset:
        print("\n── Labelling from real dataset ────────────────────")
        from data_pipeline.step7_label_decision_data import process_dataset
        from pathlib import Path
        process_dataset(
            "data/raw/final_agriculture_training_dataset.jsonl",
            Path("decision_models/training"),
        )
    else:
        step_generate_training_data()

    if not args.skip_train:
        step_train_crop_model()

    pipeline_ok = step_run_pipeline_test()

    if not args.skip_tests:
        tests_ok = step_run_unit_tests()
    else:
        tests_ok = True

    print("\n" + "=" * 55)
    print(f"  Pipeline    : {'✓ Ready' if pipeline_ok else '✗ Failed'}")
    print(f"  Tests       : {'✓ Passed' if tests_ok else '✗ Failed'}")
    print("=" * 55)

    print_integration_guide()


if __name__ == "__main__":
    main()
