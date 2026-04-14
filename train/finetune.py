"""
train/finetune.py
=================
SageStorm V2 Instruction Fine-Tuning

All 5 V1 bugs fixed:
  BUG 1: Response mask list-comparison → element-wise search
  BUG 2: patience=3 → patience=8
  BUG 3: No warmup → 200-step warmup
  BUG 4: No label smoothing → 0.1
  BUG 5: Uniform LR → layer-wise decay
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


# ── Response mask fix ─────────────────────────────────────────
def find_response_start(tokens: list, resp_ids: list) -> int:
    """
    Correct sliding-window search.
    V1 used list-slice == list which is ALWAYS False in Python.
    Returns position after the response marker, or -1 if not found.
    """
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


def pack(tokens, ctx):
    blocks = []
    for i in range(0, len(tokens) - ctx - 1, ctx):
        b = tokens[i: i + ctx + 1]
        if len(b) == ctx + 1: blocks.append(b)
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
            for j in range(start): y[j] = -100
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ── Layer-wise LR decay ───────────────────────────────────────
def make_param_groups(model, base_lr):
    """Create parameter groups with layer-wise LR decay.
    Handles weight tying by tracking parameter IDs to avoid duplicates.
    """
    n = len(model.blocks)
    groups = []
    seen_params = set()  # Track which parameters we've already added
    
    # Embedding group
    emb_params = []
    for p in model.emb.parameters():
        pid = id(p)
        if pid not in seen_params:
            emb_params.append(p)
            seen_params.add(pid)
    groups.append({"params": emb_params,
                   "lr": base_lr * (LR_DECAY_RATE ** (n + 1)), "name": "emb"})
    
    # Block groups
    for i, blk in enumerate(model.blocks):
        blk_params = []
        for p in blk.parameters():
            pid = id(p)
            if pid not in seen_params:
                blk_params.append(p)
                seen_params.add(pid)
        groups.append({"params": blk_params,
                       "lr": base_lr * (LR_DECAY_RATE ** (n - i)), "name": f"blk{i}"})
    
    # Head group (norm + head, skip duplicates from weight tying)
    head_params = []
    for p in list(model.norm.parameters()) + list(model.head.parameters()):
        pid = id(p)
        if pid not in seen_params:
            head_params.append(p)
            seen_params.add(pid)
    groups.append({"params": head_params,
                   "lr": base_lr, "name": "head"})
    
    return groups


def get_scale(step, total, warmup):
    if step < warmup: return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    mn = FINETUNE_LR_MIN / FINETUNE_LR_MAX
    return mn + (1 - mn) * 0.5 * (1 + math.cos(math.pi * p))


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
    n = len(loader)
    loss = tot_loss / n
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "acc": tot_cor / max(tot_tok, 1)}


# ── Train epoch ───────────────────────────────────────────────
def train_epoch(model, loader, opt, base_lrs, crit, scaler, gstep, total):
    model.train()
    tot = 0.0
    opt.zero_grad(set_to_none=True)
    for bi, (x, y) in enumerate(tqdm(loader, desc="  finetune")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        scale = get_scale(gstep, total, FINETUNE_WARMUP)
        for pg, blr in zip(opt.param_groups, base_lrs): pg["lr"] = blr * scale
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            loss = crit(model(x).view(-1, VOCAB_SIZE), y.view(-1)) / FINETUNE_ACCUM
        scaler.scale(loss).backward()
        if (bi + 1) % FINETUNE_ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            gstep += 1
        tot += loss.item() * FINETUNE_ACCUM
    return tot / len(loader), gstep


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=PRETRAINED_CKPT)
    args = parser.parse_args()

    tok      = SageTokenizer()
    resp_ids = tok.get_response_ids()
    print(f"[Data] Response marker IDs: {resp_ids}")

    tr_tok = load_tokens(TRAIN_TOKENS)
    vl_tok = load_tokens(VAL_TOKENS)
    tr_blk = pack(tr_tok, CONTEXT_LENGTH)
    vl_blk = pack(vl_tok, CONTEXT_LENGTH)
    print(f"[Data] Train={len(tr_blk):,} Val={len(vl_blk):,} blocks")

    tr_ld = DataLoader(InstructDS(tr_blk, resp_ids), FINETUNE_BATCH,
                       shuffle=True, num_workers=4, pin_memory=(DEVICE=="cuda"))
    vl_ld = DataLoader(InstructDS(vl_blk, resp_ids), FINETUNE_BATCH,
                       num_workers=4, pin_memory=(DEVICE=="cuda"))

    model = SageStormV2().to(DEVICE)
    if os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Model] Loaded pretrained ← {args.pretrained}")

    groups   = make_param_groups(model, FINETUNE_LR_MAX)
    base_lrs = [g["lr"] for g in groups]
    for g in groups: g["weight_decay"] = WEIGHT_DECAY if "blk" in g["name"] else 0.0

    opt    = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    crit   = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    total  = (len(tr_ld) // FINETUNE_ACCUM) * FINETUNE_EPOCHS

    best = float("inf"); patience = 0; gstep = 0
    log  = ["epoch,train_loss,train_ppl,val_loss,val_ppl,val_acc"]

    for epoch in range(FINETUNE_EPOCHS):
        print(f"\nEpoch {epoch+1}/{FINETUNE_EPOCHS}  patience={patience}/{FINETUNE_PATIENCE}")
        tr_loss, gstep = train_epoch(model, tr_ld, opt, base_lrs, crit, scaler, gstep, total)
        vm = evaluate(model, vl_ld, crit)
        print(f"  train loss={tr_loss:.4f} ppl={math.exp(min(tr_loss,20)):.2f}")
        print(f"  val   loss={vm['loss']:.4f} ppl={vm['ppl']:.2f} acc={vm['acc']*100:.2f}%")
        log.append(f"{epoch+1},{tr_loss:.4f},{math.exp(min(tr_loss,20)):.2f},"
                   f"{vm['loss']:.4f},{vm['ppl']:.2f},{vm['acc']*100:.2f}")
        if vm["loss"] < best:
            best = vm["loss"]; patience = 0
            model.save(FINETUNED_CKPT)
            print(f"  ✓ Best saved (val_loss={best:.4f})")
        else:
            patience += 1
            if patience >= FINETUNE_PATIENCE:
                print("  Early stopping.")
                break

    with open(os.path.join(MODELS_DIR, "finetune_log.csv"), "w") as f:
        f.write("\n".join(log))
    print(f"\n[Train] Done. Best val_loss={best:.4f} ppl={math.exp(min(best,20)):.2f}")


if __name__ == "__main__":
    main()
