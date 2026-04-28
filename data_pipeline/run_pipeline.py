"""
data_pipeline/run_pipeline.py
==============================
Master pipeline script — runs all 5 preprocessing steps in order.

Usage:
  python data_pipeline/run_pipeline.py           (run all steps)
  python data_pipeline/run_pipeline.py --from 3  (resume from step 3)
  python data_pipeline/run_pipeline.py --only 5  (run only step 5)

Steps:
  0  →  Clean raw source data files before preprocessing
  1  →  Preprocess instruction dataset (clean + split)
  2  →  Build knowledge corpus (ICAR + research)
  3  →  Train SentencePiece tokenizer
  4  →  Tokenize pretraining corpus
  5  →  Tokenize instruction dataset (train + val)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

STEPS = {
    1: ("step1_preprocess_dataset.py",      "Preprocess instruction dataset"),
    2: ("step2_build_knowledge_corpus.py",   "Build knowledge corpus"),
    3: ("step3_train_tokenizer.py",          "Train SentencePiece tokenizer"),
    4: ("step4_tokenize_pretrain.py",        "Tokenize pretrain corpus"),
    5: ("step5_tokenize_instructions.py",    "Tokenize instruction dataset"),
}

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(step_num: int):
    script, desc = STEPS[step_num]
    path = os.path.join(PIPELINE_DIR, script)
    print(f"\n{'='*55}")
    print(f"  STEP {step_num}/5 — {desc}")
    print(f"{'='*55}")
    command = [sys.executable, path]
    if step_num == 0:
        command.append("--clean-pipeline-files")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"\n[Pipeline] Step {step_num} FAILED (exit code {result.returncode})")
        return False
    print(f"\n[Pipeline] Step {step_num} ✓ Done")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full data preprocessing pipeline")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--from",   dest="from_step", type=int, default=1,
                       metavar="N", help="Start from step N (default: 1)")
    group.add_argument("--only",   dest="only_step", type=int, default=None,
                       metavar="N", help="Run only step N")
    args = parser.parse_args()

    if args.only_step is not None:
        if args.only_step not in STEPS:
            print(f"[Pipeline] Invalid step: {args.only_step}. Choose 0–5.")
            sys.exit(1)
        run_step(args.only_step)
        return

    start = args.from_step
    if start not in STEPS:
        print(f"[Pipeline] Invalid start step: {start}. Choose 0–5.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  SageStorm V2 — Data Preprocessing Pipeline")
    print(f"  Running steps {start}–5")
    print(f"{'='*55}")

    for step in range(start, 6):
        ok = run_step(step)
        if not ok:
            print(f"\n[Pipeline] Stopped at step {step}. Fix the error and resume with:")
            print(f"  python data_pipeline/run_pipeline.py --from {step}")
            sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  ✓ All pipeline steps complete!")
    print(f"  You can now train the V2 model:")
    print(f"    python train/pretrain.py")
    print(f"    python train/finetune.py")
    print(f"    python train/evaluate.py --generate")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
