"""
train/evaluate.py
=================
Comprehensive evaluation: loss, PPL, token accuracy,
top-5 accuracy, BLEU-4, repetition rate, V1 vs V2 comparison.
All helpers inlined — no circular imports.
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
    print(f"[Data] {len(all_t):,} tokens <- {path}")
    return all_t

def pack(tokens, ctx):
    blocks = []
    for i in range(0, len(tokens) - ctx - 1, ctx):
        b = tokens[i: i + ctx + 1]
        if len(b) == ctx + 1: blocks.append(b)
    return blocks

def find_response_start(tokens, resp_ids):
    if not resp_ids: return -1
    R = len(resp_ids)
    for i in range(len(tokens) - R + 1):
        if tokens[i: i + R] == resp_ids: return i + R
    return -1

class InstructDS(Dataset):
    def __init__(self, blocks, resp_ids):
        self.blocks = blocks; self.resp_ids = resp_ids
    def __len__(self): return len(self.blocks)
    def __getitem__(self, i):
        b = self.blocks[i]; x = b[:-1]; y = list(b[1:])
        start = find_response_start(x, self.resp_ids)
        if start > 0:
            for j in range(start): y[j] = -100
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ── BLEU-4 ───────────────────────────────────────────────────
def ngrams(t, n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]

def bleu4(hyp, ref):
    if not hyp or not ref: return 0.0
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref)/len(hyp))
    scores = [(sum(min(c, collections.Counter(ngrams(ref,n))[g])
               for g,c in collections.Counter(ngrams(hyp,n)).items()) + 1) /
              (max(len(hyp)-n+1, 0) + 1) for n in range(1,5)]
    return bp * math.exp(sum(math.log(s) for s in scores)/4)

def rep_rate(ids, n=4):
    ng = ngrams(ids, n)
    return 0.0 if not ng else 1 - len(set(ng))/len(ng)

# ── Evaluation ───────────────────────────────────────────────
@torch.no_grad()
def run_eval(model, loader, crit):
    model.eval()
    tot_loss = tot_cor = tot_tok = top5_cor = 0
    for x, y in tqdm(loader, desc="  evaluating"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        tot_loss += crit(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()
        mask = (y != -100)
        tot_cor  += ((logits.argmax(-1) == y) & mask).sum().item()
        tot_tok  += mask.sum().item()
        top5_cor += ((logits.topk(5,-1).indices == y.unsqueeze(-1)).any(-1) & mask).sum().item()
    n = len(loader); loss = tot_loss / n
    return {"val_loss":loss,"ppl":math.exp(min(loss,20)),
            "tok_acc":tot_cor/max(tot_tok,1),"top5_acc":top5_cor/max(tot_tok,1),"total_tok":tot_tok}

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=FINETUNED_CKPT)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    tok      = SageTokenizer()
    resp_ids = tok.get_response_ids()
    vl_blk   = pack(load_tokens(VAL_TOKENS), CONTEXT_LENGTH)
    print(f"Validation blocks: {len(vl_blk)}")
    vl_ld    = DataLoader(InstructDS(vl_blk, resp_ids), 16, num_workers=4)
    model    = SageStormV2.load(args.model, DEVICE)
    crit     = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=LABEL_SMOOTHING)
    m        = run_eval(model, vl_ld, crit)
    V1       = {"loss":1.8116,"ppl":6.1203,"acc":0.6451}

    print(f"""
Evaluation Results
------------------
Validation Loss  : {m['val_loss']:.7f}
Perplexity       : {m['ppl']:.6f}
Token Accuracy   : {m['tok_acc']*100:.4f}%
Top-5 Accuracy   : {m['top5_acc']*100:.4f}%

V1 vs V2 comparison:
  Val Loss  : {V1['loss']} -> {m['val_loss']:.4f}  ({'improved' if m['val_loss']<V1['loss'] else 'worse'})
  PPL       : {V1['ppl']} -> {m['ppl']:.4f}   ({'improved' if m['ppl']<V1['ppl'] else 'worse'})
  Token Acc : {V1['acc']*100:.1f}%  -> {m['tok_acc']*100:.2f}%  ({'improved' if m['tok_acc']>V1['acc'] else 'worse'})""")

    if args.generate:
        prompts = [
            "### Instruction:\nHow to control aphids on lemon tree?\n\n### Response:\n",
            "### Instruction:\nWhat fertilizer dose for rice per acre?\n\n### Response:\n",
            "### Instruction:\nHow to prevent late blight in tomatoes?\n\n### Response:\n",
            "### Instruction:\nHow to control stem borers in rice?\n\n### Response:\n",
        ]
        print("\nGeneration Samples:")
        bleus, reps = [], []
        for p in prompts:
            ids = tok.encode_prompt(p)
            gen_ids = model.generate(ids, max_tokens=80, device=DEVICE)
            gen_txt = tok.decode(gen_ids)
            reps.append(rep_rate(gen_ids)); bleus.append(bleu4(gen_ids[:50], ids[:50]))
            print(f"\n  Q: {p.split(chr(10))[1]}")
            print(f"  A: {gen_txt[:200]}")
            print(f"  RepRate={reps[-1]:.3f}  BLEU={bleus[-1]:.3f}")
        print(f"\n  Avg BLEU : {sum(bleus)/len(bleus):.4f}")
        print(f"  Avg Rep  : {sum(reps)/len(reps):.4f}")

if __name__ == "__main__":
    main()
