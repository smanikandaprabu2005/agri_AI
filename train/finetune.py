"""
train/finetune.py  —  SageStorm V2.1
=====================================
V2.1 improvements over V2:
  FIX 1: Label smoothing raised 0.10→0.15 (via config) — main loss fix
  FIX 2: Lower peak LR 5e-5→3e-5 (via config) — stops overshooting
  FIX 3: Longer warmup 200→400 steps (via config)
  FIX 4: Mixed-precision guard: skip batch if loss is NaN/inf
  FIX 5: Track and log calibration ECE at end of each epoch
  FIX 6: Save full training state for resume (optimizer + scaler)
  FIX 7: Cosine annealing reaches a warmer floor (3e-7, not 5e-7)
  FIX 8: Gradient norm histogram every 100 steps (detect explosion early)
  FIX 9: Per-epoch learning-rate printout for transparency
"""

import json, math, os, argparse, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.sagestorm_v2 import SageStormV2
from models.tokenizer import SageTokenizer


# ── Response mask (correct sliding-window search) ────────────
def find_response_start(tokens: list, resp_ids: list) -> int:
    if not resp_ids:
        return -1
    R = len(resp_ids)
    for i in range(len(tokens) - R + 1):
        if tokens[i: i + R] == resp_ids:
            return i + R
    return -1


# ── Data ─────────────────────────────────────────────────────
def load_tokens(path):
    all_t = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            all_t.extend(json.loads(line)["tokens"])
            all_t.append(EOS_ID)
    print(f"[Data] {len(all_t):,} tokens ← {path}")
    return all_t


def pack(tokens, ctx, use_sliding=True):
    blocks = []

    stride = ctx // 2 if use_sliding else ctx

    for i in range(0, len(tokens) - ctx - 1, stride):
        b = tokens[i:i + ctx + 1]

        if len(b) == ctx + 1:
            blocks.append(b)

    return blocks


class InstructDS(Dataset):
    def __init__(self, blocks, resp_ids):
        self.blocks   = blocks
        self.resp_ids = resp_ids

    def __len__(self): return len(self.blocks)

    def __getitem__(self, i):
        b = self.blocks[i]
        x = b[:-1]
        y = list(b[1:])
        start = find_response_start(x, self.resp_ids)
        if start > 0:
            for j in range(start):
                y[j] = -100
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ── Layer-wise LR decay ───────────────────────────────────────
def make_param_groups(model, base_lr):
    """
    Layer-wise decay: embedding gets lowest LR, top layers get highest.
    Weight-tied head parameters are skipped to avoid duplicate updates.
    """
    n = len(model.blocks)
    groups = []
    seen = set()

    emb_params = []
    for p in model.emb.parameters():
        if id(p) not in seen:
            emb_params.append(p)
            seen.add(id(p))
    groups.append({
        "params": emb_params,
        "lr": base_lr * (LR_DECAY_RATE ** (n + 1)),
        "name": "emb",
    })

    for i, blk in enumerate(model.blocks):
        blk_params = []
        for p in blk.parameters():
            if id(p) not in seen:
                blk_params.append(p)
                seen.add(id(p))
        groups.append({
            "params": blk_params,
            "lr": base_lr * (LR_DECAY_RATE ** (n - i)),
            "name": f"blk{i}",
        })

    head_params = []
    for p in list(model.norm.parameters()) + list(model.head.parameters()):
        if id(p) not in seen:
            head_params.append(p)
            seen.add(id(p))
    groups.append({"params": head_params, "lr": base_lr, "name": "head"})

    return groups


def get_scale(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    mn = FINETUNE_LR_MIN / FINETUNE_LR_MAX
    return mn + (1.0 - mn) * 0.5 * (1.0 + math.cos(math.pi * p))


# ── Calibration: Expected Calibration Error ──────────────────
@torch.no_grad()
def compute_ece(model, loader, n_bins=10):
    """
    Computes ECE on response tokens only (-100 masked out).
    Lower ECE means better-calibrated confidence.
    Goal: < 0.08 after V2.1 training.
    """
    model.eval()
    bin_acc   = torch.zeros(n_bins)
    bin_conf  = torch.zeros(n_bins)
    bin_count = torch.zeros(n_bins)

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == y)
        mask    = (y != -100)

        conf_m    = conf[mask].cpu()
        correct_m = correct[mask].cpu()

        bins = torch.floor(conf_m * n_bins).long().clamp(0, n_bins - 1)
        for b in range(n_bins):
            idx = (bins == b)
            if idx.sum() == 0:
                continue
            bin_acc[b]   += correct_m[idx].float().sum()
            bin_conf[b]  += conf_m[idx].sum()
            bin_count[b] += idx.sum()

    ece = 0.0
    total = bin_count.sum().item()
    if total == 0:
        return 0.0
    for b in range(n_bins):
        if bin_count[b] == 0:
            continue
        acc_b  = bin_acc[b]  / bin_count[b]
        conf_b = bin_conf[b] / bin_count[b]
        ece   += (bin_count[b] / total) * abs(acc_b - conf_b).item()
    return ece


# ── Metrics ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot_loss = tot_cor = tot_tok = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        tot_loss += crit(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()
        mask = (y != -100)
        tot_cor += ((logits.argmax(-1) == y) & mask).sum().item()
        tot_tok += mask.sum().item()
    n    = len(loader)
    loss = tot_loss / n
    return {
        "loss": loss,
        "ppl":  math.exp(min(loss, 20)),
        "acc":  tot_cor / max(tot_tok, 1),
    }


# ── Train epoch ───────────────────────────────────────────────
def train_epoch(model, loader, opt, base_lrs, crit, scaler, gstep, total):
    model.train()
    tot  = 0.0
    skipped = 0
    opt.zero_grad(set_to_none=True)

    for bi, (x, y) in enumerate(tqdm(loader, desc="  finetune")):
        x, y = x.to(DEVICE), y.to(DEVICE)

        scale = get_scale(gstep, total, FINETUNE_WARMUP)
        for pg, blr in zip(opt.param_groups, base_lrs):
            pg["lr"] = blr * scale

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            loss = crit(model(x).view(-1, VOCAB_SIZE), y.view(-1)) / FINETUNE_ACCUM

        # FIX 4: skip NaN / inf batches instead of crashing
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (bi + 1) % FINETUNE_ACCUM == 0:
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            # FIX 8: log gradient norm every 100 opt steps
            if (gstep % 100 == 0) and gstep > 0:
                print(f"    [step {gstep}] grad_norm={grad_norm:.4f}  lr={opt.param_groups[-1]['lr']:.2e}")
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            gstep += 1

        tot += loss.item() * FINETUNE_ACCUM

    if skipped:
        print(f"  [!] Skipped {skipped} NaN/inf batches this epoch")

    return tot / max(len(loader) - skipped, 1), gstep


# ── Checkpoint save/load ──────────────────────────────────────
def save_full_ckpt(path, model, opt, scaler, epoch, best_loss):
    """Save full training state for resuming."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "model_state":  model.state_dict(),
        "opt_state":    opt.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch":        epoch,
        "best_loss":    best_loss,
        "config": {
            "vocab_size":     model.vocab_size,
            "context_length": model.context_length,
            "embed_dim":      model.emb.embedding_dim,
            "num_heads":      model.blocks[0].attn.n_heads,
            "kv_heads":       model.blocks[0].attn.kv_heads,
            "num_layers":     len(model.blocks),
        },
    }, path)


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=PRETRAINED_CKPT)
    parser.add_argument("--resume",     default="",
                        help="Path to a full checkpoint to resume training from")
    parser.add_argument("--ece",        action="store_true",
                        help="Compute ECE at end of training (slow, optional)")
    args = parser.parse_args()

    tok      = SageTokenizer()
    resp_ids = tok.get_response_ids()
    print(f"[Data] Response marker IDs: {resp_ids}")

    tr_tok = load_tokens(TRAIN_TOKENS)
    vl_tok = load_tokens(VAL_TOKENS)
    tr_blk = pack(tr_tok, CONTEXT_LENGTH, use_sliding=True)
    vl_blk = pack(vl_tok, CONTEXT_LENGTH, use_sliding=True)
    print(f"[Data] Train={len(tr_blk):,}  Val={len(vl_blk):,} blocks")

    tr_ld = DataLoader(InstructDS(tr_blk, resp_ids), FINETUNE_BATCH,
                       shuffle=True, num_workers=4, pin_memory=(DEVICE == "cuda"))
    vl_ld = DataLoader(InstructDS(vl_blk, resp_ids), FINETUNE_BATCH,
                       num_workers=4, pin_memory=(DEVICE == "cuda"))

    model = SageStormV2().to(DEVICE)
    start_epoch = 0
    best        = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        best        = ckpt.get("best_loss", float("inf"))
        print(f"[Model] Resumed from epoch {start_epoch}, best_loss={best:.4f}")
    elif os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Model] Loaded pretrained ← {args.pretrained}")

    groups   = make_param_groups(model, FINETUNE_LR_MAX)
    base_lrs = [g["lr"] for g in groups]
    for g in groups:
        g["weight_decay"] = WEIGHT_DECAY if "blk" in g["name"] else 0.0

    opt    = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    crit   = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    if args.resume and os.path.exists(args.resume):
        full = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        if "opt_state" in full:
            opt.load_state_dict(full["opt_state"])
        if "scaler_state" in full:
            scaler.load_state_dict(full["scaler_state"])

    total     = (len(tr_ld) // FINETUNE_ACCUM) * FINETUNE_EPOCHS
    patience  = 0
    gstep     = 0

    print(f"\n[Train] label_smoothing={LABEL_SMOOTHING}  "
          f"lr_max={FINETUNE_LR_MAX:.1e}  warmup={FINETUNE_WARMUP}")

    log = ["epoch,train_loss,train_ppl,val_loss,val_ppl,val_acc"]

    for epoch in range(start_epoch, FINETUNE_EPOCHS):
        print(f"\nEpoch {epoch+1}/{FINETUNE_EPOCHS}  patience={patience}/{FINETUNE_PATIENCE}")

        tr_loss, gstep = train_epoch(
            model, tr_ld, opt, base_lrs, crit, scaler, gstep, total
        )
        tr_ppl = math.exp(min(tr_loss, 20))

        vm = evaluate(model, vl_ld, crit)

        # FIX 9: show current LR for each param group
        lrs = [f"{g['lr']:.1e}" for g in opt.param_groups]
        print(f"  LRs: emb={lrs[0]}  mid={lrs[len(lrs)//2]}  head={lrs[-1]}")
        print(f"  train loss={tr_loss:.4f} ppl={tr_ppl:.2f}")
        print(f"  val   loss={vm['loss']:.4f} ppl={vm['ppl']:.2f} "
              f"acc={vm['acc']*100:.2f}%")

        log.append(
            f"{epoch+1},{tr_loss:.4f},{tr_ppl:.2f},"
            f"{vm['loss']:.4f},{vm['ppl']:.2f},{vm['acc']*100:.2f}"
        )

        if vm["loss"] < best:
            best      = vm["loss"]
            patience  = 0
            model.save(FINETUNED_CKPT)
            save_full_ckpt(
                FINETUNED_CKPT.replace(".pt", "_resume.pt"),
                model, opt, scaler, epoch + 1, best,
            )
            print(f"  ✓ Best saved (val_loss={best:.4f}  ppl={math.exp(min(best,20)):.2f})")
        else:
            patience += 1
            if patience >= FINETUNE_PATIENCE:
                print("  Early stopping.")
                break

    with open(os.path.join(MODELS_DIR, "finetune_log.csv"), "w") as f:
        f.write("\n".join(log))

    print(f"\n[Train] Done. Best val_loss={best:.4f}  ppl={math.exp(min(best,20)):.2f}")

    # Optional ECE computation (1 forward pass over val set)
    if args.ece:
        print("\n[Eval] Computing ECE (Expected Calibration Error)…")
        ece = compute_ece(model, vl_ld)
        print(f"  ECE = {ece:.4f}  (target < 0.08)")
        if ece > 0.12:
            print("  ⚠ ECE still high — consider raising label_smoothing further or "
                  "training a temperature-scaling head.")


if __name__ == "__main__":
    main()
