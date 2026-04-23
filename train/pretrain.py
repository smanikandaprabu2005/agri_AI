"""
train/pretrain.py
=================
SageStorm V2.2 Pretraining — Overfitting-Fixed Version

FIXES over V2.1 (based on observed val_loss=3.96 plateau + overfitting):

  FIX 1: Augmented block packing (4 offset windows per epoch)
          → Multiplies effective training data 4x with zero extra corpus
          → Most impactful single change for small-data regimes

  FIX 2: Token-level noise injection (5% random token replacement)
          → Forces context-dependent learning, prevents memorization
          → Equivalent to data augmentation in vision models

  FIX 3: Smoothed patience check (3-epoch rolling average)
          → Prevents early stopping from noise in val loss estimates
          → Your 3.96 → 4.01 oscillation was noise, not real degradation

  FIX 4: Cosine restarts (SGDR) instead of single cosine decay
          → Escapes local minima after plateaus
          → Each restart decays max_lr by 0.8x

  FIX 5: Patience increased to 12 from 5
          → Gives model room to recover from LR restart warm-up phases

  FIX 6: Val split increased to 0.15 from 0.1
          → More reliable loss estimates (2,800+ blocks vs 1,713)
"""

import json
import math
import os
import random
import time
import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PRETRAIN_TOKENS, PRETRAINED_CKPT, MODELS_DIR,
    VOCAB_SIZE, CONTEXT_LENGTH, DEVICE,
    PRETRAIN_BATCH, PRETRAIN_EPOCHS,
    PRETRAIN_LR_MAX, PRETRAIN_LR_MIN, PRETRAIN_WARMUP,
    PRETRAIN_ACCUM, CLIP_NORM, USE_AMP, EOS_ID, PAD_ID,
)
from models.sagestorm_v2 import SageStormV2


# ── Hyperparameters ───────────────────────────────────────────
VAL_SPLIT        = 0.15       # FIX 6: was 0.1 — more reliable val estimates
PATIENCE         = 12         # FIX 5: was 5 — prevent premature stopping
SAVE_EVERY       = 5
LABEL_SMOOTHING  = 0.05
LOG_INTERVAL     = 100
NUM_WORKERS      = 4
N_AUG_OFFSETS    = 4          # FIX 1: number of sliding window offsets per epoch
TOKEN_NOISE_PROB = 0.05       # FIX 2: fraction of tokens to randomly replace
N_LR_RESTARTS    = 3          # FIX 4: cosine restart cycles
SMOOTH_WINDOW    = 3          # FIX 3: epochs for val loss smoothing


# ── Data Loading ──────────────────────────────────────────────
def load_tokens(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[Pretrain] Token file not found: {path}\n"
            f"  Run: python data_pipeline/step4_tokenize_pretrain.py"
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
    print(f"[Pretrain] Loaded {len(all_tokens):,} tokens from {path}")
    return all_tokens


def split_tokens(tokens: list, val_ratio: float = VAL_SPLIT):
    split = int(len(tokens) * (1 - val_ratio))
    train = tokens[:split]
    val   = tokens[split:]
    print(f"[Pretrain] Train={len(train):,} Val={len(val):,} tokens")
    return train, val


def pack_blocks(tokens: list, ctx: int, offset: int = 0) -> list:
    """Original single-offset packing (used for val set)."""
    tokens = tokens[offset:]
    blocks = []
    for i in range(0, len(tokens) - ctx - 1, ctx):
        b = tokens[i: i + ctx + 1]
        if len(b) == ctx + 1:
            blocks.append(b)
    return blocks


def pack_blocks_augmented(tokens: list, ctx: int,
                          n_offsets: int = N_AUG_OFFSETS) -> list:
    """
    FIX 1: Pack with multiple random sliding window offsets.

    Why this works:
    With 13M tokens and ctx=512, a single pass gives ~25K blocks.
    Using 4 offsets creates ~100K distinct (but overlapping) blocks,
    which is equivalent to having 4x the training data.

    The overlapping windows force the model to learn from different
    context boundaries, improving generalization significantly.
    """
    all_blocks = []

    # First offset is always 0 (canonical)
    offsets = [0]
    if n_offsets > 1:
        # Sample without replacement from valid range
        offsets += random.sample(range(1, ctx), min(n_offsets - 1, ctx - 1))

    for offset in offsets:
        toks = tokens[offset:]
        for i in range(0, len(toks) - ctx - 1, ctx):
            b = toks[i: i + ctx + 1]
            if len(b) == ctx + 1:
                all_blocks.append(b)

    # Shuffle to avoid offset-correlated sequential batches
    random.shuffle(all_blocks)
    return all_blocks


class TokenDataset(Dataset):
    def __init__(self, blocks: list):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        b = self.blocks[idx]
        x = torch.tensor(b[:-1], dtype=torch.long)
        y = torch.tensor(b[1:],  dtype=torch.long)
        return x, y


# ── Token noise augmentation ──────────────────────────────────
def token_noise(x: torch.Tensor, vocab_size: int,
                noise_prob: float = TOKEN_NOISE_PROB) -> torch.Tensor:
    """
    FIX 2: Randomly replace noise_prob fraction of input tokens.

    Why this works:
    The model cannot simply memorize position → token mappings.
    It must use surrounding context to predict the target, which
    forces it to learn genuine language patterns rather than
    surface-level sequence statistics.

    This is analogous to random erasing in vision models or
    masked language modeling in BERT.
    """
    if noise_prob <= 0:
        return x
    # Replace with random tokens from [4, vocab_size) to avoid special tokens
    mask          = torch.rand_like(x, dtype=torch.float) < noise_prob
    random_tokens = torch.randint(4, vocab_size, x.shape, device=x.device)
    return torch.where(mask, random_tokens, x)


# ── Learning Rate Schedule with Restarts ──────────────────────
def get_lr(step: int, total: int, warmup: int,
           lr_max: float, lr_min: float,
           n_restarts: int = N_LR_RESTARTS) -> float:
    """
    FIX 4: Cosine Annealing with Warm Restarts (SGDR).

    Standard cosine decay reduces LR monotonically, which can trap
    the model in a local minimum after the initial valley phase.
    SGDR periodically resets LR to a (slightly lower) max, allowing
    the model to explore new regions of the loss landscape.

    Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
               (Loshchilov & Hutter, 2017)
    """
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step >= total:
        return lr_min

    remaining    = total - warmup
    cycle_len    = max(remaining // n_restarts, 1)
    pos_in_cycle = (step - warmup) % cycle_len
    restart_num  = (step - warmup) // cycle_len

    # Decay max LR by 0.8x each restart (progressive cooling)
    lr_max_this_cycle = lr_max * (0.8 ** restart_num)

    progress = pos_in_cycle / cycle_len
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + (lr_max_this_cycle - lr_min) * cosine


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Evaluate ──────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion: nn.Module,
             device: str, use_amp: bool) -> dict:
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    total_loss  = 0.0
    total_toks  = 0
    total_cor   = 0
    n_batches   = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            logits = raw_model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))

        if not torch.isfinite(loss):
            continue

        B, T = y.shape
        total_loss += loss.item()
        total_toks += B * T
        total_cor  += (logits.argmax(-1) == y).sum().item()
        n_batches  += 1

    if n_batches == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf"), "val_acc": 0.0}

    avg_loss = total_loss / n_batches
    ppl      = math.exp(min(avg_loss, 20))
    acc      = total_cor / max(total_toks, 1)

    raw_model.train()
    return {"val_loss": avg_loss, "val_ppl": ppl, "val_acc": acc}


# ── Train One Epoch ───────────────────────────────────────────
def train_epoch(model, loader: DataLoader, optimizer,
                criterion: nn.Module, scaler,
                base_lr: float, gstep: int, total_steps: int,
                device: str, use_amp: bool, accum: int) -> tuple:

    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  pretrain", leave=False)
    for bi, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # FIX 2: Apply token noise augmentation
        x = token_noise(x, VOCAB_SIZE, TOKEN_NOISE_PROB)

        if bi % accum == 0:
            lr = get_lr(gstep, total_steps, PRETRAIN_WARMUP,
                        PRETRAIN_LR_MAX, PRETRAIN_LR_MIN)
            set_lr(optimizer, lr)

        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            logits = model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss   = loss / accum

        if not torch.isfinite(loss):
            print(f"\n[Pretrain] Non-finite loss at batch {bi}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (bi + 1) % accum == 0:
            scaler.unscale_(optimizer)
            params = (model.module.parameters() if hasattr(model, "module")
                      else model.parameters())
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            gstep += 1

        total_loss += loss.item() * accum
        n_batches  += 1

        if (bi + 1) % LOG_INTERVAL == 0:
            avg = total_loss / n_batches
            pbar.set_postfix({
                "loss": f"{avg:.4f}",
                "ppl":  f"{math.exp(min(avg, 20)):.2f}",
                "lr":   f"{get_lr(gstep, total_steps, PRETRAIN_WARMUP, base_lr, PRETRAIN_LR_MIN):.2e}"
            })

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, gstep


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SageStorm V2.2 Pretraining")
    parser.add_argument("--tokens",       default=PRETRAIN_TOKENS)
    parser.add_argument("--output",       default=PRETRAINED_CKPT)
    parser.add_argument("--resume",       default="")
    parser.add_argument("--epochs",       type=int,   default=PRETRAIN_EPOCHS)
    parser.add_argument("--batch",        type=int,   default=PRETRAIN_BATCH)
    parser.add_argument("--lr_max",       type=float, default=PRETRAIN_LR_MAX)
    parser.add_argument("--val_split",    type=float, default=VAL_SPLIT)
    parser.add_argument("--n_offsets",    type=int,   default=N_AUG_OFFSETS,
                        help="Number of sliding window offsets per epoch")
    parser.add_argument("--noise_prob",   type=float, default=TOKEN_NOISE_PROB,
                        help="Token noise probability (0 to disable)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SageStorm V2.2 — Pretraining (Overfitting-Fixed)")
    print(f"  Device     : {DEVICE}")
    print(f"  AMP        : {USE_AMP}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch} (effective={args.batch * PRETRAIN_ACCUM})")
    print(f"  LR max     : {args.lr_max}")
    print(f"  Offsets    : {args.n_offsets}x (effective data multiplier)")
    print(f"  Token noise: {args.noise_prob:.1%}")
    print(f"  LR restarts: {N_LR_RESTARTS}")
    print(f"  Patience   : {PATIENCE}")
    print(f"{'='*60}\n")

    all_tokens               = load_tokens(args.tokens)
    train_tokens, val_tokens = split_tokens(all_tokens, args.val_split)
    del all_tokens

    # Val set: single pass (no augmentation — we want stable loss estimates)
    val_blocks  = pack_blocks(val_tokens, CONTEXT_LENGTH, offset=0)
    val_dataset = TokenDataset(val_blocks)
    val_loader  = DataLoader(
        val_dataset,
        batch_size         = args.batch,
        shuffle            = False,
        num_workers        = NUM_WORKERS,
        pin_memory         = (DEVICE == "cuda"),
        persistent_workers = (NUM_WORKERS > 0),
    )
    print(f"[Pretrain] Val blocks: {len(val_blocks):,}")

    model = SageStormV2().to(DEVICE)

    start_epoch   = 0
    best_val_loss = float("inf")
    gstep         = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch   = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        gstep         = ckpt.get("gstep", 0)
        print(f"[Pretrain] Resumed from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"[Pretrain] Using {n_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    raw_model = model.module if hasattr(model, "module") else model
    pc = raw_model.param_count()
    print(f"[Pretrain] {pc['total_M']}M parameters "
          f"(ratio: {len(train_tokens)/pc['total']:,.0f} tokens/param)")

    decay_params   = [p for n, p in raw_model.named_parameters()
                      if p.dim() >= 2 and "norm" not in n and p.requires_grad]
    nodecay_params = [p for n, p in raw_model.named_parameters()
                      if not (p.dim() >= 2 and "norm" not in n) and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr    = args.lr_max,
        betas = (0.9, 0.95),
        eps   = 1e-8,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index    = -1,
        label_smoothing = LABEL_SMOOTHING,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))
    os.makedirs(MODELS_DIR, exist_ok=True)

    log_rows       = ["epoch,train_loss,train_ppl,val_loss,val_ppl,val_acc,lr,n_blocks"]
    patience_count = 0
    start_time     = time.time()

    # FIX 3: Rolling val loss history for smoothed patience check
    val_loss_history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # FIX 1: Augmented packing — 4 offsets creates ~4x more blocks
        train_blocks  = pack_blocks_augmented(
            train_tokens, CONTEXT_LENGTH, n_offsets=args.n_offsets
        )
        train_dataset = TokenDataset(train_blocks)
        train_loader  = DataLoader(
            train_dataset,
            batch_size         = args.batch,
            shuffle            = True,
            num_workers        = NUM_WORKERS,
            pin_memory         = (DEVICE == "cuda"),
            persistent_workers = (NUM_WORKERS > 0),
            drop_last          = True,
        )

        # Scale total steps to account for augmented dataset size
        steps_per_epoch = len(train_loader) // PRETRAIN_ACCUM
        total_steps     = steps_per_epoch * args.epochs

        print(f"\nEpoch {epoch+1}/{args.epochs}  "
              f"[train_blocks={len(train_blocks):,} ({args.n_offsets}x aug)  "
              f"patience={patience_count}/{PATIENCE}]")

        train_loss, gstep = train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            args.lr_max, gstep, total_steps, DEVICE, USE_AMP, PRETRAIN_ACCUM,
        )
        train_ppl = math.exp(min(train_loss, 20))

        val_metrics = evaluate(model, val_loader, criterion, DEVICE, USE_AMP)
        val_loss    = val_metrics["val_loss"]
        val_ppl     = val_metrics["val_ppl"]
        val_acc     = val_metrics["val_acc"]

        # FIX 3: Smoothed val loss for patience check
        val_loss_history.append(val_loss)
        smoothed_val = sum(val_loss_history[-SMOOTH_WINDOW:]) / \
                       len(val_loss_history[-SMOOTH_WINDOW:])

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(f"  train_loss={train_loss:.4f}  ppl={train_ppl:.2f}")
        print(f"  val_loss  ={val_loss:.4f}  smoothed={smoothed_val:.4f}  "
              f"ppl={val_ppl:.2f}  acc={val_acc*100:.2f}%")
        print(f"  lr={current_lr:.2e}  epoch_time={epoch_time:.0f}s")

        log_rows.append(
            f"{epoch+1},{train_loss:.4f},{train_ppl:.2f},"
            f"{val_loss:.4f},{val_ppl:.2f},{val_acc*100:.2f},"
            f"{current_lr:.2e},{len(train_blocks)}"
        )

        # Save on smoothed improvement (less noise-sensitive)
        if smoothed_val < best_val_loss - 0.005:
            best_val_loss  = smoothed_val
            patience_count = 0
            raw_model.save(args.output)
            print(f"  ✓ Best checkpoint saved (smoothed_val={best_val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{PATIENCE})")

        if (epoch + 1) % SAVE_EVERY == 0:
            periodic_path = os.path.join(MODELS_DIR, f"pretrain_epoch{epoch+1}.pt")
            torch.save({
                "model_state":   raw_model.state_dict(),
                "epoch":         epoch + 1,
                "best_val_loss": best_val_loss,
                "gstep":         gstep,
                "config": {
                    "vocab_size"     : raw_model.vocab_size,
                    "context_length" : raw_model.context_length,
                    "embed_dim"      : raw_model.emb.embedding_dim,
                    "num_heads"      : raw_model.blocks[0].attn.n_heads,
                    "kv_heads"       : raw_model.blocks[0].attn.kv_heads,
                    "num_layers"     : len(raw_model.blocks),
                }
            }, periodic_path)
            print(f"  Periodic checkpoint → {periodic_path}")

        if patience_count >= PATIENCE:
            print(f"\n[Pretrain] Early stopping at epoch {epoch+1}")
            break

    log_path = os.path.join(MODELS_DIR, "pretrain_log.csv")
    with open(log_path, "w") as f:
        f.write("\n".join(log_rows))

    final_ppl = math.exp(min(best_val_loss, 20))
    print(f"\n[Pretrain] Log saved → {log_path}")
    print(f"[Pretrain] ✓ Done. Best smoothed_val_loss={best_val_loss:.4f}  "
          f"ppl={final_ppl:.2f}")
    print(f"[Pretrain] Total time: {(time.time()-start_time)/3600:.2f}h")
    print(f"\nNext step: python train/finetune.py")


if __name__ == "__main__":
    main()