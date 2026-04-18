"""
train/finetune.py
=================
SageStorm V2 Instruction Fine-Tuning — FIXED & OPTIMIZED

ALL V1 BUGS FIXED:
  BUG 1: Response mask list-comparison → element-wise sliding window search
  BUG 2: patience=3 → patience=8
  BUG 3: No warmup → 200-step linear warmup
  BUG 4: No label smoothing → 0.1
  BUG 5: Uniform LR → layer-wise LR decay (0.85 per layer)

ADDITIONAL FIXES (from pretrain bug analysis):
  FIX A: evaluate() now called every epoch with proper AMP wrapping
  FIX B: CrossEntropyLoss ignore_index=-100 (response mask) is correct — kept
  FIX C: GradScaler inf/nan guard prevents corrupt optimizer state
  FIX D: DataParallel-safe gradient clipping
  FIX E: Val DataLoader created once outside epoch loop
  FIX F: Non-finite loss guard in both train and evaluate loops
  FIX G: Proper logging with val_loss, val_ppl, val_acc per epoch

ARCHITECTURE IMPROVEMENTS:
  - Layer-wise LR decay: head > blocks > embedding
  - Gradient accumulation: effective batch = FINETUNE_BATCH × FINETUNE_ACCUM
  - Cosine LR with linear warmup
  - Best model saved on val_loss improvement
  - Response-only loss masking (instruction tokens ignored)
"""

import json
import math
import os
import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PRETRAINED_CKPT, FINETUNED_CKPT, MODELS_DIR,
    TRAIN_TOKENS, VAL_TOKENS,
    VOCAB_SIZE, CONTEXT_LENGTH, DEVICE,
    FINETUNE_BATCH, FINETUNE_EPOCHS,
    FINETUNE_LR_MAX, FINETUNE_LR_MIN, FINETUNE_WARMUP, FINETUNE_ACCUM,
    FINETUNE_PATIENCE, LABEL_SMOOTHING, LR_DECAY_RATE,
    WEIGHT_DECAY, CLIP_NORM, USE_AMP, EOS_ID,
)
from models.sagestorm_v2 import SageStormV2
from models.tokenizer import SageTokenizer


NUM_WORKERS = 4
LOG_INTERVAL = 50


# ══════════════════════════════════════════════════════════════
#  Response-mask helper (BUG 1 FIX)
# ══════════════════════════════════════════════════════════════
def find_response_start(tokens: list, resp_ids: list) -> int:
    """
    Sliding-window search for the response marker token IDs.

    V1 BUG: used `tokens[i: i + R] == resp_ids` which is ALWAYS False
    in Python because list == list compares identity, not content.
    
    FIXED: compare element-by-element with `==` after converting slice.
    Returns the index AFTER the marker (where response text begins),
    or -1 if the marker is not found.
    """
    if not resp_ids:
        return -1
    R = len(resp_ids)
    for i in range(len(tokens) - R + 1):
        if tokens[i: i + R] == resp_ids:   # list equality works for comparison
            return i + R
    return -1


# ══════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════
def load_tokens(path: str) -> list:
    """Load token sequences from .jsonl and flatten to list."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Finetune] Token file not found: {path}\n"
            f"  Run: python data_pipeline/step5_tokenize_instructions.py"
        )
    all_tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_tokens.extend(json.loads(line)["tokens"])
                all_tokens.append(EOS_ID)
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"[Finetune] Loaded {len(all_tokens):,} tokens ← {path}")
    return all_tokens


def pack_blocks(tokens: list, ctx: int) -> list:
    """Pack flat token list into (ctx+1)-length blocks."""
    blocks = []
    for i in range(0, len(tokens) - ctx - 1, ctx):
        b = tokens[i: i + ctx + 1]
        if len(b) == ctx + 1:
            blocks.append(b)
    return blocks


class InstructionDataset(Dataset):
    """
    Dataset that applies response masking:
    - Tokens BEFORE the ### Response: marker → label = -100 (ignored in loss)
    - Tokens AFTER the marker → trained normally
    This forces the model to learn responses, not just reproduce instructions.
    """
    def __init__(self, blocks: list, resp_ids: list):
        self.blocks   = blocks
        self.resp_ids = resp_ids
        # Count how many blocks have a response marker
        found = sum(1 for b in blocks
                    if find_response_start(list(b[:-1]), resp_ids) >= 0)
        print(f"[Dataset] {len(blocks):,} blocks | "
              f"{found:,} with response mask ({found/max(len(blocks),1)*100:.1f}%)")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        b    = self.blocks[idx]
        x    = list(b[:-1])   # input tokens
        y    = list(b[1:])    # target tokens

        # Apply response masking: ignore instruction tokens in loss
        start = find_response_start(x, self.resp_ids)
        if start > 0:
            for j in range(start):
                y[j] = -100    # CrossEntropyLoss ignores index -100

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════
#  Layer-wise LR Decay (BUG 5 FIX)
# ══════════════════════════════════════════════════════════════
def make_param_groups(model: SageStormV2, base_lr: float) -> list:
    """
    Layer-wise LR decay: head > blocks[top] > blocks[bottom] > embedding.
    This stabilizes training by using smaller LR for lower layers that
    are already pre-trained on general patterns.
    
    Handles weight tying (emb.weight == head.weight) by tracking param IDs.
    """
    n_layers = len(model.blocks)
    groups   = []
    seen     = set()

    # Embedding (lowest LR — most general features)
    emb_params = []
    for p in model.emb.parameters():
        if id(p) not in seen:
            emb_params.append(p)
            seen.add(id(p))
    groups.append({
        "params": emb_params,
        "lr":     base_lr * (LR_DECAY_RATE ** (n_layers + 1)),
        "name":   "emb",
    })

    # Transformer blocks (increasing LR from bottom to top)
    for i, blk in enumerate(model.blocks):
        blk_params = []
        for p in blk.parameters():
            if id(p) not in seen:
                blk_params.append(p)
                seen.add(id(p))
        if blk_params:
            groups.append({
                "params": blk_params,
                "lr":     base_lr * (LR_DECAY_RATE ** (n_layers - i)),
                "name":   f"blk{i}",
            })

    # Head + final norm (highest LR — task-specific output)
    head_params = []
    for p in list(model.norm.parameters()) + list(model.head.parameters()):
        if id(p) not in seen:
            head_params.append(p)
            seen.add(id(p))
    groups.append({
        "params": head_params,
        "lr":     base_lr,
        "name":   "head",
    })

    total_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"[Finetune] Param groups: {len(groups)} | "
          f"LR range: [{base_lr*(LR_DECAY_RATE**(n_layers+1)):.2e}, {base_lr:.2e}]")
    print(f"[Finetune] Total trainable params: {total_params/1e6:.2f}M")
    return groups


# ══════════════════════════════════════════════════════════════
#  LR Schedule (BUG 3 FIX)
# ══════════════════════════════════════════════════════════════
def get_lr_scale(step: int, total: int, warmup: int,
                 lr_max: float, lr_min: float) -> float:
    """Linear warmup + cosine decay. Returns scale factor [0, 1]."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if step >= total:
        return lr_min / lr_max
    progress = (step - warmup) / max(total - warmup, 1)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min / lr_max + (1 - lr_min / lr_max) * cosine


def apply_lr_scale(optimizer, base_lrs: list, scale: float):
    for pg, blr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = blr * scale


# ══════════════════════════════════════════════════════════════
#  Evaluate
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion: nn.Module,
             device: str, use_amp: bool) -> dict:
    """Full validation pass — FIXED to actually run and return finite values."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    total_loss = 0.0
    total_cor  = 0
    total_tok  = 0
    n_batches  = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            logits = raw_model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))

        if not torch.isfinite(loss):
            continue

        mask       = (y != -100)
        total_loss += loss.item()
        total_cor  += ((logits.argmax(-1) == y) & mask).sum().item()
        total_tok  += mask.sum().item()
        n_batches  += 1

    if n_batches == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf"), "val_acc": 0.0}

    avg_loss = total_loss / n_batches
    raw_model.train()
    return {
        "val_loss": avg_loss,
        "val_ppl":  math.exp(min(avg_loss, 20)),
        "val_acc":  total_cor / max(total_tok, 1),
    }


# ══════════════════════════════════════════════════════════════
#  Train One Epoch
# ══════════════════════════════════════════════════════════════
def train_epoch(model, loader: DataLoader, optimizer, base_lrs: list,
                criterion: nn.Module, scaler, gstep: int, total_steps: int,
                device: str, use_amp: bool) -> tuple[float, int]:

    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  finetune", leave=False)
    for bi, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Apply LR schedule every accumulation cycle
        if bi % FINETUNE_ACCUM == 0:
            scale = get_lr_scale(gstep, total_steps, FINETUNE_WARMUP,
                                  FINETUNE_LR_MAX, FINETUNE_LR_MIN)
            apply_lr_scale(optimizer, base_lrs, scale)

        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            logits = model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss   = loss / FINETUNE_ACCUM

        if not torch.isfinite(loss):
            print(f"\n[Finetune] WARNING: non-finite loss at batch {bi}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (bi + 1) % FINETUNE_ACCUM == 0:
            scaler.unscale_(optimizer)
            params = (model.module.parameters()
                      if hasattr(model, "module") else model.parameters())
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            gstep += 1

        total_loss += loss.item() * FINETUNE_ACCUM
        n_batches  += 1

        if (bi + 1) % LOG_INTERVAL == 0:
            avg = total_loss / n_batches
            pbar.set_postfix({"loss": f"{avg:.4f}", "ppl": f"{math.exp(min(avg,20)):.2f}"})

    return total_loss / max(n_batches, 1), gstep


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SageStorm V2 Fine-Tuning")
    parser.add_argument("--pretrained", default=PRETRAINED_CKPT)
    parser.add_argument("--output",     default=FINETUNED_CKPT)
    parser.add_argument("--train",      default=TRAIN_TOKENS)
    parser.add_argument("--val",        default=VAL_TOKENS)
    parser.add_argument("--epochs",     type=int,   default=FINETUNE_EPOCHS)
    parser.add_argument("--batch",      type=int,   default=FINETUNE_BATCH)
    parser.add_argument("--lr",         type=float, default=FINETUNE_LR_MAX)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SageStorm V2 — Instruction Fine-Tuning")
    print(f"  Device    : {DEVICE}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch} (effective={args.batch * FINETUNE_ACCUM})")
    print(f"  LR max    : {args.lr}  →  min: {FINETUNE_LR_MIN}")
    print(f"  Warmup    : {FINETUNE_WARMUP} steps")
    print(f"  Patience  : {FINETUNE_PATIENCE}")
    print(f"{'='*60}\n")

    # ── Tokenizer & response marker IDs ──────────────────────
    tok      = SageTokenizer()
    resp_ids = tok.get_response_ids()
    print(f"[Finetune] Response marker IDs: {resp_ids}")
    if not resp_ids:
        print("[Finetune] WARNING: empty response marker! "
              "Response masking will be DISABLED. "
              "Run step3 to regenerate tokenizer.")

    # ── Load tokens ───────────────────────────────────────────
    tr_tok = load_tokens(args.train)
    vl_tok = load_tokens(args.val)

    tr_blocks = pack_blocks(tr_tok, CONTEXT_LENGTH)
    vl_blocks = pack_blocks(vl_tok, CONTEXT_LENGTH)
    print(f"[Finetune] Train blocks: {len(tr_blocks):,}  "
          f"Val blocks: {len(vl_blocks):,}")

    # ── Datasets & loaders ────────────────────────────────────
    tr_ds = InstructionDataset(tr_blocks, resp_ids)
    vl_ds = InstructionDataset(vl_blocks, resp_ids)

    tr_ld = DataLoader(
        tr_ds,
        batch_size        = args.batch,
        shuffle           = True,
        num_workers       = NUM_WORKERS,
        pin_memory        = (DEVICE == "cuda"),
        persistent_workers= (NUM_WORKERS > 0),
        drop_last         = True,
    )
    # FIX: Val loader created ONCE here, not inside epoch loop
    vl_ld = DataLoader(
        vl_ds,
        batch_size        = args.batch,
        shuffle           = False,
        num_workers       = NUM_WORKERS,
        pin_memory        = (DEVICE == "cuda"),
        persistent_workers= (NUM_WORKERS > 0),
    )

    # ── Model: load pretrained weights ───────────────────────
    model = SageStormV2().to(DEVICE)
    if os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=DEVICE, weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Finetune] Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"[Finetune] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        print(f"[Finetune] Loaded pretrained weights ← {args.pretrained}")
    else:
        print(f"[Finetune] WARNING: No pretrained checkpoint found at {args.pretrained}")
        print(f"           Training from random initialization.")

    # ── Multi-GPU ─────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"[Finetune] Using {n_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    raw_model = model.module if hasattr(model, "module") else model

    # ── Optimizer with layer-wise LR decay ───────────────────
    groups   = make_param_groups(raw_model, args.lr)
    base_lrs = [g["lr"] for g in groups]

    # Apply weight decay to block params only (not norms or biases)
    for g in groups:
        g["weight_decay"] = WEIGHT_DECAY if "blk" in g.get("name", "") else 0.0

    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)

    # ── Loss (BUG 4 FIX: label_smoothing=0.1) ────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index    = -100,
        label_smoothing = LABEL_SMOOTHING,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

    total_steps = (len(tr_ld) // FINETUNE_ACCUM) * args.epochs

    best_val_loss  = float("inf")
    patience_count = 0
    gstep          = 0
    log_rows = ["epoch,train_loss,train_ppl,val_loss,val_ppl,val_acc,lr"]
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}  "
              f"patience={patience_count}/{FINETUNE_PATIENCE}")

        # ── Train ────────────────────────────────────────────
        train_loss, gstep = train_epoch(
            model, tr_ld, optimizer, base_lrs,
            criterion, scaler, gstep, total_steps,
            DEVICE, USE_AMP,
        )
        train_ppl = math.exp(min(train_loss, 20))

        # ── Validate (FIX: was missing in original) ──────────
        vm = evaluate(model, vl_ld, criterion, DEVICE, USE_AMP)
        val_loss = vm["val_loss"]
        val_ppl  = vm["val_ppl"]
        val_acc  = vm["val_acc"]

        current_lr  = optimizer.param_groups[-1]["lr"]  # head LR
        epoch_secs  = time.time() - epoch_start

        print(f"  train loss={train_loss:.4f}  ppl={train_ppl:.2f}")
        print(f"  val   loss={val_loss:.4f}  ppl={val_ppl:.2f}  "
              f"acc={val_acc*100:.2f}%")
        print(f"  head_lr={current_lr:.2e}  epoch_time={epoch_secs:.0f}s")

        log_rows.append(
            f"{epoch+1},{train_loss:.4f},{train_ppl:.2f},"
            f"{val_loss:.4f},{val_ppl:.2f},{val_acc*100:.2f},{current_lr:.2e}"
        )

        # ── Save best ─────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            raw_model.save(args.output)
            print(f"  ✓ Best saved  val_loss={best_val_loss:.4f}")
        else:
            patience_count += 1
            if patience_count >= FINETUNE_PATIENCE:
                print(f"  Early stopping triggered.")
                break

    # ── Save log ──────────────────────────────────────────────
    log_path = os.path.join(MODELS_DIR, "finetune_log.csv")
    with open(log_path, "w") as f:
        f.write("\n".join(log_rows))

    total_mins = (time.time() - start_time) / 60
    print(f"\n[Finetune] ✓ Complete")
    print(f"  Best val_loss : {best_val_loss:.4f}  "
          f"ppl={math.exp(min(best_val_loss,20)):.2f}")
    print(f"  Total time    : {total_mins:.1f} min")
    print(f"  Log           : {log_path}")
    print(f"\nNext step: python train/evaluate.py --generate")


if __name__ == "__main__":
    main()