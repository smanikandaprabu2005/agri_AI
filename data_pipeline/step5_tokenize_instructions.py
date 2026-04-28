"""
data_pipeline/step5_tokenize_instructions.py  — FIXED VERSION
=================================================================
FIX 1: tokenize_dataset() now takes model_path as explicit parameter
        so --tokenizer CLI arg is respected inside worker processes.
FIX 2: Path constants use the space-containing folder name that
        step1 actually creates — override with CLI args if needed.
FIX 3: Stats report token counts (not char counts).
"""

import json
import argparse
import sys
import os
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count

try:
    import sentencepiece as spm
except ImportError:
    raise SystemExit("sentencepiece not installed. Run: pip install sentencepiece")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import build_training_prompt, RESPONSE_MARKER

# ── Config — use CLI flags if your folder has a space ──────────
TOKENIZER_MODEL = "sage_tokenizer.model"
TRAIN_DATASET   = "data_pipeline/data collection/train_dataset.jsonl"
VAL_DATASET     = "data_pipeline/data collection/val_dataset.jsonl"
TRAIN_TOKENS    = "data_pipeline/tokens/train_tokens.jsonl"
VAL_TOKENS      = "data_pipeline/tokens/val_tokens.jsonl"

_SP = None


def _clean_markdown(text: str) -> str:
    """Remove markdown formatting noise from text."""
    text = text.replace('**', '')
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _init_worker(model_path: str):
    global _SP
    _SP = spm.SentencePieceProcessor()
    _SP.load(model_path)


def _tokenize_record(record_json: str):
    try:
        item   = json.loads(record_json)
        inst   = item.get("instruction", "").strip()
        inp    = item.get("input",       "").strip()
        out    = item.get("output",      "").strip()
        if not inst or not out:
            return None
        # Clean markdown from all fields
        inst = _clean_markdown(inst)
        inp = _clean_markdown(inp) if inp else ""
        out = _clean_markdown(out)
        text   = build_training_prompt(inst, inp, out)
        tokens = _SP.encode(text, out_type=int)
        if len(tokens) < 5:
            return None
        return json.dumps({"tokens": tokens}, ensure_ascii=False)
    except Exception:
        return None


def verify_response_marker(sp: spm.SentencePieceProcessor):
    prefix     = "### Instruction:\nTest\n\n"
    full       = prefix + RESPONSE_MARKER
    prefix_ids = sp.encode(prefix, out_type=int)
    full_ids   = sp.encode(full,   out_type=int)
    marker_ids = full_ids[len(prefix_ids):]
    decoded    = sp.decode(marker_ids)
    match      = decoded.strip() == RESPONSE_MARKER.strip()
    print(f"\n[Step 5] Response marker verification:")
    print(f"  Marker string : {RESPONSE_MARKER!r}")
    print(f"  Token IDs     : {marker_ids}")
    print(f"  Decoded       : {decoded!r}")
    print(f"  Round-trip OK : {'OK' if match else 'WARNING — check tokenizer!'}")
    if not match:
        print("  [WARNING] Response marker round-trip FAILED.")
        print("  Response masking during fine-tuning will be BROKEN.")
    return list(marker_ids)


def tokenize_dataset(
    sp,
    input_path:  str,
    output_path: str,
    model_path:  str,     # FIX: explicit param, not hardcoded constant
    n_workers:   int  = 4,
    dry_run:     bool = False,
) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if dry_run:
        lines = lines[:1000]
        print(f"[Step 5] DRY RUN: tokenizing first {len(lines)} samples")

    total        = 0
    skipped      = 0
    token_counts = []

    effective_workers = min(n_workers, cpu_count(), max(len(lines) // 1000, 1))
    print(f"[Step 5] Tokenizing {len(lines):,} records (workers={effective_workers})")

    with open(output_path, "w", encoding="utf-8") as fout:
        if effective_workers > 1 and len(lines) > 5000:
            with Pool(effective_workers,
                      initializer=_init_worker,
                      initargs=(model_path,)) as pool:  # FIX: pass correct model_path
                chunk = 200
                for i in range(0, len(lines), chunk):
                    batch   = lines[i : i + chunk]
                    results = pool.map(_tokenize_record, batch)
                    for res in results:
                        if res is None:
                            skipped += 1
                        else:
                            fout.write(res + "\n")
                            n_tok = len(json.loads(res)["tokens"])
                            token_counts.append(n_tok)
                            total += 1
                    if (i // chunk) % 50 == 0:
                        print(f"  [Step 5] {i:,}/{len(lines):,}  "
                              f"tokenized={total:,}  skipped={skipped:,}")
        else:
            _init_worker(model_path)
            for raw in lines:
                res = _tokenize_record(raw)
                if res is None:
                    skipped += 1
                else:
                    fout.write(res + "\n")
                    n_tok = len(json.loads(res)["tokens"])
                    token_counts.append(n_tok)
                    total += 1

    avg_tok   = sum(token_counts) / max(len(token_counts), 1)
    max_tok   = max(token_counts) if token_counts else 0
    total_tok = sum(token_counts)
    return {
        "total":        total,
        "skipped":      skipped,
        "avg_tokens":   avg_tok,
        "max_tokens":   max_tok,
        "total_tokens": total_tok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer",  default=TOKENIZER_MODEL)
    parser.add_argument("--train_data", default=TRAIN_DATASET)
    parser.add_argument("--val_data",   default=VAL_DATASET)
    parser.add_argument("--train_out",  default=TRAIN_TOKENS)
    parser.add_argument("--val_out",    default=VAL_TOKENS)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    missing = []
    if not Path(args.tokenizer).exists():
        missing.append(f"{args.tokenizer}  (run step3 first)")
    if not Path(args.train_data).exists():
        missing.append(f"{args.train_data}  (run step1 first)")
    if not Path(args.val_data).exists():
        missing.append(f"{args.val_data}  (run step1 first)")
    if missing:
        for m in missing:
            print(f"[Step 5] ERROR: Not found: {m}")
        return

    print(f"\n[Step 5] Tokenizing Instruction Datasets")
    print(f"  Tokenizer : {args.tokenizer}")
    print(f"  Train     : {args.train_data} -> {args.train_out}")
    print(f"  Val       : {args.val_data}   -> {args.val_out}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"[Step 5] Tokenizer vocab size: {sp.get_piece_size()}")

    verify_response_marker(sp)

    print(f"\n[Step 5] Tokenizing training set...")
    tr = tokenize_dataset(
        sp, args.train_data, args.train_out,
        args.tokenizer,       # FIX: pass model_path explicitly
        args.workers, args.dry_run,
    )
    print(f"  Tokenized   : {tr['total']:,}  Skipped: {tr['skipped']:,}")
    print(f"  Avg tokens  : {tr['avg_tokens']:.1f}  Max: {tr['max_tokens']}")
    print(f"  Total tokens: {tr['total_tokens']:,}")

    print(f"\n[Step 5] Tokenizing validation set...")
    vl = tokenize_dataset(
        sp, args.val_data, args.val_out,
        args.tokenizer,       # FIX: pass model_path explicitly
        args.workers, args.dry_run,
    )
    print(f"  Tokenized   : {vl['total']:,}  Skipped: {vl['skipped']:,}")
    print(f"  Avg tokens  : {vl['avg_tokens']:.1f}  Max: {vl['max_tokens']}")
    print(f"  Total tokens: {vl['total_tokens']:,}")

    print(f"\n[Step 5] Complete. Ready for training:")
    print("    python train/pretrain.py")
    print("    python train/finetune.py")


if __name__ == "__main__":
    main()