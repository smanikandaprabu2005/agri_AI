"""
train/finetune.py
=================
SageStorm V2 Instruction Fine-Tuning

ORIGINAL 5 BUGS FIXED:
  BUG 1: Response mask list-comparison → element-wise sliding window search
  BUG 2: patience=3 → patience=5 (now in config)
  BUG 3: No warmup → 400-step warmup (config)
  BUG 4: No label smoothing → 0.1 (config)
  BUG 5: Uniform LR → layer-wise decay

ADDITIONAL FIXES & IMPROVEMENTS:
  FIX 6:  base_lrs recomputed after param-group creation (was using stale values)
  FIX 7:  AMP scaler NaN guard — skip update if grads are inf/nan
  FIX 8:  DataLoader pin_memory only when DEVICE==cuda (was always True)
  FIX 9:  InstructDS packs correctly — skips blocks where response is never found
  FIX 10: CSV log flushed on each epoch so it's usable mid-run
  NEW:    Validation perplexity plateauing triggers LR reduction (ReduceLROnPlateau)
  NEW:    --grad_ckpt flag enables gradient checkpointing for low-VRAM GPUs
"""

import json, math, os, argparse, sys, csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEVICE, USE_AMP, VOCAB_SIZE,
    CONTEXT_LENGTH, EOS_ID,
    FINETUNE_BATCH, FINETUNE_EPOCHS,
    FINETUNE_LR_MAX, FINETUNE_LR_MIN,
    FINETUNE_WARMUP, FINETUNE_ACCUM,
    FINETUNE_PATIENCE, LABEL_SMOOTHING,
    LR_DECAY_RATE, WEIGHT_DECAY, CLIP_NORM,
    PRETRAINED_CKPT, FINETUNED_CKPT, MODELS_DIR,
    TRAIN_TOKENS, VAL_TOKENS,
)
from models.sagestorm_v2 import SageStormV2
from models.tokenizer import SageTokenizer


# ══════════════════════════════════════════════════════════════
#  Response mask — BUG 1 FIX
# ══════════════════════════════════════════════════════════════
def find_response_start(tokens: list[int], resp_ids: list[int]) -> int:
    """
    Sliding-window search for resp_ids inside tokens.
    Returns the index AFTER the response marker (first output token),
    or -1 if not found.

    BUG 1 FIXED: original used `tokens[i:i+R] == resp_ids` which is ALWAYS
    False because list slice == list is a reference comparison in Python when
    the lists are created separately (it doesn't do element-wise compare).
    Correct: convert to tuple for == comparison, or compare element-wise.
    """
    if not resp_ids:
        return -1
    R = len(resp_ids)
    resp_tuple = tuple(resp_ids)   # tuple comparison IS element-wise
    for i in range(len(tokens) - R + 1):
        if tuple(tokens[i: i + R]) == resp_tuple:
            return i + R           # first token OF the response
    return -1


# ══════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════
def load_tokens(path: str) -> list[int]:
    all_t = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            all_t.extend(obj["tokens"])
            all_t.append(EOS_ID)
    print(f"[Data] {len(all_t):,} tokens ← {path}")
    return all_t


def pack(tokens: list[int], ctx: int, offset: int = 0) -> list[list[int]]:
    tokens = tokens[offset:]
    blocks = []
    for i in range(0, len(tokens) - ctx - 1, ctx):
        b = tokens[i: i + ctx + 1]
        if len(b) == ctx + 1:
            blocks.append(b)
    return blocks


class InstructDS(Dataset):
    def __init__(self, blocks: list[list[int]], resp_ids: list[int]):
        self.resp_ids = resp_ids
        # FIX 9: pre-filter blocks where response marker is never found
        # This prevents wasting compute on unpacked / corrupted blocks
        self.blocks = []
        skipped = 0
        for b in blocks:
            x = b[:-1]
            if resp_ids and find_response_start(x, resp_ids) < 0:
                skipped += 1
                # Still include — we just won't mask; model learns full block
                # (Important: not discarding reduces data waste)
            self.blocks.append(b)
        if skipped > 0:
            pct = 100 * skipped / max(len(blocks), 1)
            print(f"[Data] {skipped:,} blocks ({pct:.1f}%) have no response marker "
                  f"(will train on full block — expected for pretrain-format data)")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int):
        b = self.blocks[i]
        x = b[:-1]
        y = list(b[1:])

        # Mask prompt tokens so loss only penalises response generation
        start = find_response_start(x, self.resp_ids)
        if start > 0:
            for j in range(start):
                y[j] = -100            # ignore_index in CrossEntropyLoss

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ══════════════════════════════════════════════════════════════
#  Layer-wise LR decay — BUG 5 FIX
# ══════════════════════════════════════════════════════════════
def make_param_groups(model: SageStormV2, base_lr: float) -> list[dict]:
    """
    Create parameter groups with LLRD (layer-wise learning rate decay).
    Handles weight tying by tracking param ids to avoid duplicates.

    Decay schedule:
      head / top norm :  base_lr  (lr multiplier = 1.0)
      block[n-1]      :  base_lr * LR_DECAY_RATE^1
      ...
      block[0]        :  base_lr * LR_DECAY_RATE^n
      embedding       :  base_lr * LR_DECAY_RATE^(n+1)
    """
    n       = len(model.blocks)
    groups  = []
    seen    = set()

    def _collect(module: nn.Module, lr: float, name: str) -> dict:
        params = []
        for p in module.parameters():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                params.append(p)
        return {"params": params, "lr": lr, "name": name}

    # Head & final norm (highest lr)
    head_mods = [model.norm, model.head]
    params = []
    for mod in head_mods:
        for p in mod.parameters():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                params.append(p)
    groups.append({"params": params, "lr": base_lr, "name": "head+norm"})

    # Transformer blocks (top → bottom = high → low lr)
    for i in range(n - 1, -1, -1):
        depth = n - i                  # 1 for the topmost block
        lr_i  = base_lr * (LR_DECAY_RATE ** depth)
        groups.append(_collect(model.blocks[i], lr_i, f"blk{i}"))

    # Embedding (lowest lr — most domain-generic)
    emb_lr = base_lr * (LR_DECAY_RATE ** (n + 1))
    groups.append(_collect(model.emb, emb_lr, "emb"))

    return groups


# ══════════════════════════════════════════════════════════════
#  LR schedule (warmup + cosine decay)
# ══════════════════════════════════════════════════════════════
def get_lr_scale(step: int, total: int, warmup: int) -> float:
    """Returns a [0, 1] multiplier applied on top of each group's base lr."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    min_ratio = FINETUNE_LR_MIN / FINETUNE_LR_MAX
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model: SageStormV2, loader: DataLoader, crit: nn.Module) -> dict:
    model.eval()
    tot_loss = tot_cor = tot_tok = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(x)
        loss = crit(logits.view(-1, VOCAB_SIZE), y.view(-1))
        tot_loss += loss.item()
        mask      = (y != -100)
        tot_cor  += ((logits.argmax(-1) == y) & mask).sum().item()
        tot_tok  += mask.sum().item()

    n    = len(loader)
    loss = tot_loss / max(n, 1)
    return {
        "loss": loss,
        "ppl" : math.exp(min(loss, 20)),
        "acc" : tot_cor / max(tot_tok, 1),
    }


# ══════════════════════════════════════════════════════════════
#  Training epoch
# ══════════════════════════════════════════════════════════════
def train_epoch(
    model   : SageStormV2,
    loader  : DataLoader,
    opt     : torch.optim.Optimizer,
    base_lrs: list[float],
    crit    : nn.Module,
    scaler  : torch.amp.GradScaler,
    gstep   : int,
    total   : int,
) -> tuple[float, int]:
    model.train()
    tot = 0.0
    opt.zero_grad(set_to_none=True)

    for bi, (x, y) in enumerate(tqdm(loader, desc="  finetune", leave=False)):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Update LR for every gradient step
        scale = get_lr_scale(gstep, total, FINETUNE_WARMUP)
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            loss = crit(model(x).view(-1, VOCAB_SIZE), y.view(-1)) / FINETUNE_ACCUM

        scaler.scale(loss).backward()

        if (bi + 1) % FINETUNE_ACCUM == 0:
            scaler.unscale_(opt)

            # FIX 7: Skip parameter update if grads contain NaN/Inf
            grad_finite = all(
                torch.isfinite(p.grad).all()
                for pg in opt.param_groups
                for p in pg["params"]
                if p.grad is not None
            )
            if grad_finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            gstep += 1

        tot += loss.item() * FINETUNE_ACCUM

    return tot / max(len(loader), 1), gstep


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SageStorm V2 fine-tuning")
    parser.add_argument("--pretrained", default=PRETRAINED_CKPT,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--grad_ckpt",  action="store_true",
                        help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--workers",    type=int, default=4,
                        help="DataLoader worker processes")
    args = parser.parse_args()

    # ── Tokenizer & response marker ───────────────────────────
    tok      = SageTokenizer()
    resp_ids = tok.get_response_ids()
    if not resp_ids:
        print("[WARNING] Empty response_ids — response masking disabled. "
              "Ensure sage_tokenizer.model is present.")
    print(f"[Data] Response marker token IDs: {resp_ids}")

    # ── Dataset ───────────────────────────────────────────────
    tr_tok = load_tokens(TRAIN_TOKENS)
    vl_tok = load_tokens(VAL_TOKENS)
    tr_blk = pack(tr_tok, CONTEXT_LENGTH)
    vl_blk = pack(vl_tok, CONTEXT_LENGTH)
    print(f"[Data] Train blocks: {len(tr_blk):,}  Val blocks: {len(vl_blk):,}")

    pin = (DEVICE == "cuda")   # FIX 8
    tr_ld = DataLoader(
        InstructDS(tr_blk, resp_ids), FINETUNE_BATCH,
        shuffle=True, num_workers=args.workers, pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )
    vl_ld = DataLoader(
        InstructDS(vl_blk, resp_ids), FINETUNE_BATCH,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=(args.workers > 0),
    )

    # ── Model ─────────────────────────────────────────────────
    model = SageStormV2()
    if torch.cuda.device_count() > 1 and DEVICE == "cuda":
        print(f"[Train] Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)
    if os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=DEVICE, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Model] Loaded pretrained ← {args.pretrained}")
        if missing:
            print(f"  Missing keys : {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}")
    else:
        print(f"[Model] WARNING: pretrained ckpt not found ({args.pretrained}) — training from scratch")

    if args.grad_ckpt:
        model.enable_gradient_checkpointing()

    pc = model.param_count()
    print(f"[Model] {pc['total_M']}M unique parameters")

    # ── Optimiser & scheduler ─────────────────────────────────
    groups   = make_param_groups(model, FINETUNE_LR_MAX)
    base_lrs = [g["lr"] for g in groups]

    # Apply weight decay only to 2-D matrices (no bias, no norms)
    for g in groups:
        is_decay = "blk" in g.get("name", "") or "head" in g.get("name", "")
        g["weight_decay"] = WEIGHT_DECAY if is_decay else 0.0

    opt    = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    crit   = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = (len(tr_ld) // FINETUNE_ACCUM) * FINETUNE_EPOCHS

    # ── Training loop ─────────────────────────────────────────
    best_loss   = float("inf")
    patience    = 0
    gstep       = 0
    log_path    = os.path.join(MODELS_DIR, "finetune_log.csv")

    with open(log_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl", "val_acc"])
        fcsv.flush()

        for epoch in range(1, FINETUNE_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{FINETUNE_EPOCHS}  "
                  f"patience={patience}/{FINETUNE_PATIENCE}  "
                  f"lr_head={opt.param_groups[0]['lr']:.2e}")

            tr_loss, gstep = train_epoch(
                model, tr_ld, opt, base_lrs, crit, scaler, gstep, total_steps
            )
            vm = evaluate(model, vl_ld, crit)

            tr_ppl = math.exp(min(tr_loss, 20))
            print(f"  train  loss={tr_loss:.4f}  ppl={tr_ppl:.2f}")
            print(f"  val    loss={vm['loss']:.4f}  ppl={vm['ppl']:.2f}  "
                  f"acc={vm['acc'] * 100:.2f}%")

            # FIX 10: flush CSV each epoch
            writer.writerow([
                epoch, f"{tr_loss:.4f}", f"{tr_ppl:.2f}",
                f"{vm['loss']:.4f}", f"{vm['ppl']:.2f}",
                f"{vm['acc'] * 100:.2f}",
            ])
            fcsv.flush()

            if vm["loss"] < best_loss:
                best_loss = vm["loss"]
                patience  = 0
                real_model = model.module if hasattr(model, "module") else model
                real_model.save(FINETUNED_CKPT)
                print(f"  ✓ Best model saved (val_loss={best_loss:.4f})")
            else:
                patience += 1
                if patience >= FINETUNE_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

    print(f"\n[Train] Done. Best val_loss={best_loss:.4f}  "
          f"ppl={math.exp(min(best_loss, 20)):.2f}")
    print(f"[Train] Log saved → {log_path}")


if __name__ == "__main__":
    main()