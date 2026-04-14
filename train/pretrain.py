"""
train/pretrain.py
=================
SageStorm V2 Pretraining

Improvements over V1:
  Warmup + cosine LR | Gradient accumulation (eff. batch=64)
  Weight decay only on 2D params | Random offset augmentation
  Per-epoch best checkpoint saving
"""

import json, math, os, random, argparse, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.sagestorm_v2 import SageStormV2


# ── Data ─────────────────────────────────────────────────────
def load_tokens(path: str) -> list:
    all_tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            all_tokens.extend(json.loads(line)["tokens"])
            all_tokens.append(EOS_ID)
    print(f"[Data] {len(all_tokens):,} tokens from {path}")
    return all_tokens


def pack(tokens, ctx, offset=0):
    tokens = tokens[offset:]
    blocks = []
    for i in range(0, len(tokens) - ctx - 1, ctx):
        b = tokens[i: i + ctx + 1]
        if len(b) == ctx + 1:
            blocks.append(b)
    return blocks


class PackedDS(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks
    def __len__(self): return len(self.blocks)
    def __getitem__(self, i):
        b = self.blocks[i]
        return torch.tensor(b[:-1], dtype=torch.long), torch.tensor(b[1:], dtype=torch.long)


# ── Scheduler ────────────────────────────────────────────────
def get_lr(step, total, warmup, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * p))


# ── Train epoch ───────────────────────────────────────────────
def train_epoch(model, loader, opt, crit, scaler, gstep, total):
    model.train()
    total_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for bi, (x, y) in enumerate(tqdm(loader, desc="  pretrain")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        lr = get_lr(gstep, total, PRETRAIN_WARMUP, PRETRAIN_LR_MAX, PRETRAIN_LR_MIN)
        for pg in opt.param_groups: pg["lr"] = lr
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            loss = crit(model(x).view(-1, VOCAB_SIZE), y.view(-1)) / PRETRAIN_ACCUM
        scaler.scale(loss).backward()
        if (bi + 1) % PRETRAIN_ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            gstep += 1
        total_loss += loss.item() * PRETRAIN_ACCUM
    return total_loss / len(loader), gstep


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    tokens = load_tokens(PRETRAIN_TOKENS)
    est_steps = (len(pack(tokens, CONTEXT_LENGTH)) // PRETRAIN_BATCH // PRETRAIN_ACCUM) * PRETRAIN_EPOCHS

    model = SageStormV2().to(DEVICE)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Train] Resumed from {args.resume}")

    print(f"[Model] {model.param_count()['total_M']}M params")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "norm" not in n]
    no_dc = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "norm" not in n)]
    opt   = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                {"params": no_dc, "weight_decay": 0.0}],
                               lr=PRETRAIN_LR_MAX, betas=(0.9, 0.95))
    crit  = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best = float("inf")
    gstep = 0
    log = ["epoch,loss,ppl"]

    for epoch in range(PRETRAIN_EPOCHS):
        off = random.randint(0, CONTEXT_LENGTH - 1) if epoch > 0 else 0
        blocks = pack(tokens, CONTEXT_LENGTH, off)
        loader = DataLoader(PackedDS(blocks), batch_size=PRETRAIN_BATCH,
                            shuffle=True, num_workers=4, pin_memory=(DEVICE=="cuda"))
        loss, gstep = train_epoch(model, loader, opt, crit, scaler, gstep, est_steps)
        ppl = math.exp(min(loss, 20))
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS}  loss={loss:.4f}  ppl={ppl:.2f}")
        log.append(f"{epoch+1},{loss:.4f},{ppl:.2f}")
        if loss < best:
            best = loss
            model.save(PRETRAINED_CKPT)
            print(f"  ✓ Saved best (loss={best:.4f})")

    with open(os.path.join(MODELS_DIR, "pretrain_log.csv"), "w") as f:
        f.write("\n".join(log))
    print("[Train] Pretraining complete.")


if __name__ == "__main__":
    main()
