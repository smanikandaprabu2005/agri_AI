"""
train/pretrain.py
=================
SageStorm V2 Pretraining on Agricultural Knowledge Corpus

FIXES & IMPROVEMENTS vs original:
  FIX 1:  AMP autocast device guard — no crash on CPU machines
  FIX 2:  GradScaler only created when USE_AMP=True (was always created)
  FIX 3:  Random offset augmentation starts at epoch 1, not epoch 0
  FIX 4:  Checkpoint saved only when val_loss (not train_loss) improves
          — original tracked train_loss which can overfit
  FIX 5:  DataLoader pin_memory guarded by DEVICE==cuda
  FIX 6:  Log flushed each epoch (readable mid-run)
  NEW:    --resume flag accepts latest checkpoint auto-detection
  NEW:    Gradient checkpointing flag for low-VRAM machines
  NEW:    Separate validation split (10% of corpus) for honest evaluation
"""

import json, math, os, random, argparse, sys, csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, USE_AMP, VOCAB_SIZE, CONTEXT_LENGTH,
    PRETRAIN_BATCH, PRETRAIN_EPOCHS,
    PRETRAIN_LR_MAX, PRETRAIN_LR_MIN, PRETRAIN_WARMUP, PRETRAIN_ACCUM,
    CLIP_NORM, PAD_ID, EOS_ID,
    PRETRAINED_CKPT, MODELS_DIR, PRETRAIN_TOKENS,
)
from models.sagestorm_v2 import SageStormV2


# ══════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════
def load_tokens(path: str) -> list[int]:
    all_tokens: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_tokens.extend(json.loads(line)["tokens"])
            all_tokens.append(EOS_ID)
    print(f"[Data] {len(all_tokens):,} tokens from {path}")
    return all_tokens


def pack(tokens: list[int], ctx: int, offset: int = 0) -> list[list[int]]:
    tokens = tokens[offset:]
    blocks = []
    for i in range(0, len(tokens) - ctx - 1, ctx):
        b = tokens[i: i + ctx + 1]
        if len(b) == ctx + 1:
            blocks.append(b)
    return blocks


class PackedDS(Dataset):
    def __init__(self, blocks: list[list[int]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int):
        b = self.blocks[i]
        return (
            torch.tensor(b[:-1], dtype=torch.long),
            torch.tensor(b[1:],  dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════
#  LR schedule
# ══════════════════════════════════════════════════════════════
def get_lr(step: int, total: int, warmup: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total_loss = 0.0
    count = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        if USE_AMP and DEVICE == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
        else:
            logits = model(x)

        logits = torch.clamp(logits, -50, 50)

        loss = crit(logits.view(-1, VOCAB_SIZE), y.view(-1))

        if torch.isfinite(loss):
            total_loss += loss.item()
            count += 1

    return total_loss / count if count > 0 else float("inf")

# ══════════════════════════════════════════════════════════════
#  Training epoch
# ══════════════════════════════════════════════════════════════
def train_epoch(
    model   : SageStormV2,
    loader  : DataLoader,
    opt     : torch.optim.Optimizer,
    crit    : nn.Module,
    scaler  : torch.amp.GradScaler,
    gstep   : int,
    total   : int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    opt.zero_grad(set_to_none=True)

    for bi, (x, y) in enumerate(tqdm(loader, desc="  pretrain", leave=False)):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Update LR every gradient step
        lr = get_lr(gstep, total, PRETRAIN_WARMUP, PRETRAIN_LR_MAX, PRETRAIN_LR_MIN)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if USE_AMP and DEVICE == "cuda":
            with torch.amp.autocast("cuda"):
                loss = crit(model(x).view(-1, VOCAB_SIZE), y.view(-1)) / PRETRAIN_ACCUM
            scaler.scale(loss).backward()
        else:
            loss = crit(model(x).view(-1, VOCAB_SIZE), y.view(-1)) / PRETRAIN_ACCUM
            loss.backward()

        if (bi + 1) % PRETRAIN_ACCUM == 0:
            if USE_AMP and DEVICE == "cuda":
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()
            opt.zero_grad(set_to_none=True)
            gstep += 1

        total_loss += loss.item() * PRETRAIN_ACCUM

    return total_loss / max(len(loader), 1), gstep


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SageStorm V2 pretraining")
    parser.add_argument("--resume",    default="",
                        help="Resume from checkpoint path (or 'latest')")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of tokens used for validation (default 0.1)")
    parser.add_argument("--workers",   type=int, default=4)
    args = parser.parse_args()

    # ── Load tokens & split ───────────────────────────────────
    tokens = load_tokens(PRETRAIN_TOKENS)

    # FIX 4: Proper train/val split instead of tracking only train loss
    n_val       = int(len(tokens) * args.val_split)
    val_tokens  = tokens[-n_val:]
    train_tokens = tokens[:-n_val]
    print(f"[Data] Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")

    val_blocks  = pack(val_tokens, CONTEXT_LENGTH)
    pin = (DEVICE == "cuda")   # FIX 5

    val_loader = DataLoader(
        PackedDS(val_blocks), PRETRAIN_BATCH,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )

    # ── Model ─────────────────────────────────────────────────
    model = SageStormV2()
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1 and DEVICE == "cuda":
        print(f"[Train] Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)

    start_epoch = 1
    resume_path = args.resume
    if resume_path == "latest":
        resume_path = PRETRAINED_CKPT
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Train] Resumed from {resume_path} (epoch {start_epoch})")

    if args.grad_ckpt:
        model.enable_gradient_checkpointing()

    pc = model.module.param_count() if hasattr(model, "module") else model.param_count()
    print(f"[Model] {pc['total_M']}M unique params  |  device={DEVICE}")

    # ── Optimiser ─────────────────────────────────────────────
    decay  = [p for n, p in model.named_parameters()
               if p.dim() >= 2 and "norm" not in n and "emb" not in n]
    no_dec = [p for n, p in model.named_parameters()
               if not (p.dim() >= 2 and "norm" not in n and "emb" not in n)]

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_dec, "weight_decay": 0.0}],
        lr=PRETRAIN_LR_MAX, betas=(0.9, 0.95), eps=1e-8,
    )

    crit   = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.05)
    # FIX 2: Only create GradScaler when AMP is actually enabled
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

    # Estimate total gradient steps
    approx_blocks  = len(pack(train_tokens, CONTEXT_LENGTH))
    approx_batches = approx_blocks // PRETRAIN_BATCH
    total_steps    = (approx_batches // PRETRAIN_ACCUM) * PRETRAIN_EPOCHS

    # ── Training loop ─────────────────────────────────────────
    best_val   = float("inf")
    gstep      = 0
    log_path   = os.path.join(MODELS_DIR, "pretrain_log.csv")

    with open(log_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl"])
        fcsv.flush()

        for epoch in range(start_epoch, PRETRAIN_EPOCHS + 1):
            # FIX 3: Random offset augmentation from epoch 1 onward
            offset = random.randint(0, CONTEXT_LENGTH - 1) if epoch > 1 else 0
            blocks = pack(train_tokens, CONTEXT_LENGTH, offset)
            loader = DataLoader(
                PackedDS(blocks), batch_size=PRETRAIN_BATCH,
                shuffle=True, num_workers=args.workers,
                pin_memory=pin, persistent_workers=(args.workers > 0),
            )

            tr_loss, gstep = train_epoch(model, loader, opt, crit, scaler, gstep, total_steps)
            val_loss       = evaluate(model, val_loader, crit)
            tr_ppl         = math.exp(min(tr_loss, 20))
            val_ppl        = math.exp(min(val_loss, 20))

            print(f"Epoch {epoch}/{PRETRAIN_EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  ppl={tr_ppl:.2f}  "
                  f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_ppl:.2f}",
                              f"{val_loss:.4f}", f"{val_ppl:.2f}"])
            fcsv.flush()   # FIX 6

            # FIX 4: Save based on val_loss, not train_loss
            if val_loss < best_val:
                best_val = val_loss
                real_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "model_state": real_model.state_dict(),
                    "epoch": epoch,
                    "config": {
                        "vocab_size"    : real_model.vocab_size,
                        "context_length": real_model.context_length,
                        "embed_dim"     : real_model.embed_dim,
                        "num_heads"     : real_model.blocks[0].attn.n_heads,
                        "kv_heads"      : real_model.blocks[0].attn.kv_heads,
                        "num_layers"    : len(real_model.blocks),
                    },
                }, PRETRAINED_CKPT)
                print(f"  ✓ Best model saved (val_loss={best_val:.4f})")

    print(f"[Train] Pretraining complete. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()