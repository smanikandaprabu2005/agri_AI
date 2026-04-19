"""
train/evaluate.py  —  SageStorm V2.1
======================================
Comprehensive evaluation:
  loss, PPL, token accuracy, top-5 accuracy,
  BLEU-4, repetition rate, ECE (calibration),
  response-only metrics, V1 vs V2 vs V2.1 comparison.
"""

import json, math, os, sys, collections, argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models.sagestorm_v2 import SageStormV2
from models.tokenizer import SageTokenizer


# ── Data helpers ─────────────────────────────────────────────
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
        if len(b) == ctx + 1:
            blocks.append(b)
    return blocks


def find_response_start(tokens, resp_ids):
    if not resp_ids:
        return -1
    R = len(resp_ids)
    for i in range(len(tokens) - R + 1):
        if tokens[i: i + R] == resp_ids:
            return i + R
    return -1


class InstructDS(Dataset):
    def __init__(self, blocks, resp_ids):
        self.blocks = blocks
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


# ── Text quality metrics ─────────────────────────────────────
def ngrams(t, n):
    return [tuple(t[i: i + n]) for i in range(len(t) - n + 1)]


def bleu4(hyp, ref):
    if not hyp or not ref:
        return 0.0
    bp     = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
    scores = []
    for n in range(1, 5):
        ref_ng = collections.Counter(ngrams(ref, n))
        hyp_ng = collections.Counter(ngrams(hyp, n))
        clip   = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        denom  = max(len(hyp) - n + 1, 0) + 1
        scores.append((clip + 1) / denom)
    return bp * math.exp(sum(math.log(s) for s in scores) / 4)


def rep_rate(ids, n=4):
    ng = ngrams(ids, n)
    return 0.0 if not ng else 1 - len(set(ng)) / len(ng)


def distinct_n(all_ids, n=2):
    """Corpus-level distinct-n: unique n-grams / total n-grams."""
    all_ng = []
    for ids in all_ids:
        all_ng.extend(ngrams(ids, n))
    if not all_ng:
        return 0.0
    return len(set(all_ng)) / len(all_ng)


# ── ECE (calibration) ─────────────────────────────────────────
@torch.no_grad()
def compute_ece(model, loader, n_bins=10):
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
        bins      = torch.floor(conf_m * n_bins).long().clamp(0, n_bins - 1)

        for b in range(n_bins):
            idx = (bins == b)
            if not idx.any():
                continue
            bin_acc[b]   += correct_m[idx].float().sum()
            bin_conf[b]  += conf_m[idx].sum()
            bin_count[b] += idx.sum()

    ece   = 0.0
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


# ── Core evaluation loop ──────────────────────────────────────
@torch.no_grad()
def run_eval(model, loader, crit):
    model.eval()
    tot_loss = tot_cor = tot_tok = top5_cor = resp_cor = resp_tok = 0

    for x, y in tqdm(loader, desc="  evaluating"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)

        tot_loss += crit(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

        mask     = (y != -100)
        resp_m   = mask                              # response tokens
        all_m    = (y >= 0)                          # all non-pad tokens

        tot_cor  += ((logits.argmax(-1) == y) & resp_m).sum().item()
        tot_tok  += resp_m.sum().item()
        top5_cor += ((logits.topk(5, -1).indices == y.unsqueeze(-1)).any(-1) & resp_m).sum().item()

    n    = len(loader)
    loss = tot_loss / n
    return {
        "val_loss":  loss,
        "ppl":       math.exp(min(loss, 20)),
        "tok_acc":   tot_cor  / max(tot_tok, 1),
        "top5_acc":  top5_cor / max(tot_tok, 1),
        "total_tok": tot_tok,
    }


# ── Generation samples ────────────────────────────────────────
def run_generation(model, tok):
    prompts = [
        ("Aphid control on lemon",
         "### Instruction:\nHow to control aphids on lemon tree?\n\n### Response:\n"),
        ("Rice fertilizer dose",
         "### Instruction:\nWhat fertilizer dose for rice per acre?\n\n### Response:\n"),
        ("Tomato late blight",
         "### Instruction:\nHow to prevent late blight in tomatoes?\n\n### Response:\n"),
        ("Stem borer in rice",
         "### Instruction:\nHow to control stem borers in rice?\n\n### Response:\n"),
        ("Whitefly on pepper",
         "### Instruction:\nHow to control whitefly on pepper plants?\n\n### Response:\n"),
    ]

    print("\nGeneration Samples:")
    bleus, reps, all_ids = [], [], []
    for title, p in prompts:
        ids     = tok.encode_prompt(p)
        gen_ids = model.generate(ids, max_tokens=80, device=DEVICE)
        gen_txt = tok.decode(gen_ids)
        reps.append(rep_rate(gen_ids))
        bleus.append(bleu4(gen_ids[:50], ids[:50]))
        all_ids.append(gen_ids)
        print(f"\n  Q: {title}")
        print(f"  A: {gen_txt[:250]}")
        print(f"     RepRate={reps[-1]:.3f}  BLEU={bleus[-1]:.3f}")

    d1 = distinct_n(all_ids, 1)
    d2 = distinct_n(all_ids, 2)
    print(f"\n  Avg BLEU-4     : {sum(bleus)/len(bleus):.4f}")
    print(f"  Avg Rep-Rate   : {sum(reps)/len(reps):.4f}  (lower = better)")
    print(f"  Distinct-1     : {d1:.4f}  (higher = more lexical diversity)")
    print(f"  Distinct-2     : {d2:.4f}  (higher = more phrasal diversity)")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=FINETUNED_CKPT)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--ece",      action="store_true")
    args = parser.parse_args()

    tok      = SageTokenizer()
    resp_ids = tok.get_response_ids()
    vl_blk   = pack(load_tokens(VAL_TOKENS), CONTEXT_LENGTH)
    print(f"Validation blocks: {len(vl_blk)}")

    vl_ld = DataLoader(InstructDS(vl_blk, resp_ids), 16, num_workers=4)
    model = SageStormV2.load(args.model, DEVICE)
    crit  = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)

    m  = run_eval(model, vl_ld, crit)
    V1 = {"loss": 1.8116, "ppl": 6.1203, "acc": 0.6451}
    V2 = {"loss": 2.0518, "ppl": 7.7818, "acc": 0.8378}

    print(f"""
Evaluation Results  (SageStorm V2.1)
--------------------------------------
Validation Loss  : {m['val_loss']:.7f}
Perplexity       : {m['ppl']:.6f}
Token Accuracy   : {m['tok_acc']*100:.4f}%   (response tokens only)
Top-5 Accuracy   : {m['top5_acc']*100:.4f}%

""")

    if args.ece:
        print("[Eval] Computing ECE (calibration)…")
        ece = compute_ece(model, vl_ld)
        overconf = "⚠ overconfident" if ece > 0.10 else "✓ well calibrated"
        print(f"  ECE = {ece:.4f}  ({overconf}, target < 0.08)")

    if args.generate:
        run_generation(model, tok)


if __name__ == "__main__":
    main()
