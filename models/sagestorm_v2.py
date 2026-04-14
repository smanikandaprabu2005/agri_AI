"""
models/sagestorm_v2.py
======================
SageStorm V2 — Complete Model Architecture

Improvements over V1:
  Pre-LN RMSNorm  | RoPE  | SwiGLU FFN
  GQA (8q/4kv)   | Weight tying | 12 layers
  Repetition penalty in generation
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    VOCAB_SIZE, CONTEXT_LENGTH, EMBED_DIM,
    NUM_HEADS, KV_HEADS, NUM_LAYERS, DROPOUT,
    EOS_ID, GEN_MAX_TOKENS, GEN_TEMPERATURE,
    GEN_TOP_K, GEN_TOP_P, GEN_REP_PENALTY, DEVICE,
)


def _ffn_hidden(embed_dim: int) -> int:
    """SwiGLU hidden dim: 8/3 × embed_dim, rounded to nearest 64."""
    raw = int(embed_dim * 8 / 3)
    return ((raw + 63) // 64) * 64


# ══════════════════════════════════════════════════════════════
#  RMSNorm
# ══════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


# ══════════════════════════════════════════════════════════════
#  Rotary Position Embedding (RoPE)
# ══════════════════════════════════════════════════════════════
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq: int = CONTEXT_LENGTH, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build(max_seq)

    def _build(self, seq_len: int):
        t    = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        emb  = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        self.register_buffer("cos_c", emb.cos()[None, None])
        self.register_buffer("sin_c", emb.sin()[None, None])
        self._len = seq_len

    def forward(self, seq_len: int):
        if seq_len > self._len:
            self._build(seq_len)
        return self.cos_c[:, :, :seq_len], self.sin_c[:, :, :seq_len]


def _rot_half(x):
    a, b = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-b, a], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + _rot_half(x) * sin


# ══════════════════════════════════════════════════════════════
#  SwiGLU Feed-Forward
# ══════════════════════════════════════════════════════════════
class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float = DROPOUT):
        super().__init__()
        h = _ffn_hidden(dim)
        self.gate = nn.Linear(dim, h, bias=False)
        self.up   = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ══════════════════════════════════════════════════════════════
#  Grouped Query Attention
# ══════════════════════════════════════════════════════════════
class GQA(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_heads: int, dropout: float = DROPOUT):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % kv_heads == 0
        self.n_heads  = n_heads
        self.kv_heads = kv_heads
        self.hd       = dim // n_heads
        self.groups   = n_heads // kv_heads
        self.q  = nn.Linear(dim, n_heads  * self.hd, bias=False)
        self.k  = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.v  = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.o  = nn.Linear(dim, dim, bias=False)
        self.dp = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.hd)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads,  self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv_heads, self.hd).transpose(1, 2)
        cos, sin = self.rope(T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = mask,
            dropout_p = self.dp.p if self.training else 0.0,
            is_causal = mask is None,
        )
        return self.o(out.transpose(1, 2).contiguous().view(B, T, C))


# ══════════════════════════════════════════════════════════════
#  Transformer Block (Pre-LN)
# ══════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_heads: int):
        super().__init__()
        self.n1  = RMSNorm(dim)
        self.attn = GQA(dim, n_heads, kv_heads)
        self.n2  = RMSNorm(dim)
        self.ff  = SwiGLU(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ff(self.n2(x))
        return x


# ══════════════════════════════════════════════════════════════
#  SageStorm V2 — Full Model
# ══════════════════════════════════════════════════════════════
class SageStormV2(nn.Module):
    def __init__(
        self,
        vocab_size     = VOCAB_SIZE,
        context_length = CONTEXT_LENGTH,
        embed_dim      = EMBED_DIM,
        num_heads      = NUM_HEADS,
        kv_heads       = KV_HEADS,
        num_layers     = NUM_LAYERS,
        dropout        = DROPOUT,
    ):
        super().__init__()
        self.vocab_size     = vocab_size
        self.context_length = context_length

        self.emb    = nn.Embedding(vocab_size, embed_dim)
        self.drop   = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, kv_heads) for _ in range(num_layers)])
        self.norm   = RMSNorm(embed_dim)
        self.head   = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying — saves 8M params
        self.head.weight = self.emb.weight

        # Causal mask buffer
        mask = torch.full((context_length, context_length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, 0.0, std)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "scale" in name:
                nn.init.ones_(p)
        n = len(self.blocks)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, std=std / math.sqrt(2 * n))
            nn.init.normal_(b.ff.down.weight, std=std / math.sqrt(2 * n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h    = self.drop(self.emb(x))
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids        : list,
        max_tokens        : int   = GEN_MAX_TOKENS,
        temperature       : float = GEN_TEMPERATURE,
        top_k             : int   = GEN_TOP_K,
        top_p             : float = GEN_TOP_P,
        eos_id            : int   = EOS_ID,
        repetition_penalty: float = GEN_REP_PENALTY,
        device            : str   = DEVICE,
    ) -> list:
        self.eval()
        ids      = list(prompt_ids[-self.context_length:])
        generated = []
        seen     = set(ids)

        for _ in range(max_tokens):
            inp    = torch.tensor([ids[-self.context_length:]], dtype=torch.long, device=device)
            logits = self.forward(inp)[0, -1, :].float()

            # Repetition penalty
            for tid in seen:
                logits[tid] = logits[tid] / repetition_penalty if logits[tid] > 0 else logits[tid] * repetition_penalty

            logits /= max(temperature, 1e-6)

            # Top-k
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")

            # Top-p nucleus
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cp     = torch.cumsum(sp, 0)
                sp[cp - sp > top_p] = 0.0
                probs  = torch.zeros_like(probs).scatter_(0, si, sp)
                probs /= probs.sum().clamp(min=1e-9)

            nxt = int(torch.multinomial(probs, 1).item())
            generated.append(nxt)
            ids.append(nxt)
            seen.add(nxt)
            if nxt == eos_id:
                break

        return generated

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "config": {
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "embed_dim": self.emb.embedding_dim,
                "num_heads": self.blocks[0].attn.n_heads,
                "kv_heads": self.blocks[0].attn.kv_heads,
                "num_layers": len(self.blocks),
            }
        }, path)
        n = sum(p.numel() for p in self.parameters())
        print(f"[Model] Saved {n/1e6:.1f}M params → {path}")

    @classmethod
    def load(cls, path: str, device: str = DEVICE) -> "SageStormV2":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        print(f"[Model] Loaded {n/1e6:.1f}M params ← {path}")
        return model.to(device)

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": round(total/1e6, 2)}


if __name__ == "__main__":
    m = SageStormV2()
    c = m.param_count()
    print(f"SageStorm V2: {c['total_M']}M params")
    x = torch.randint(0, VOCAB_SIZE, (2, 32))
    out = m(x)
    print(f"Forward: {list(x.shape)} → {list(out.shape)} ✓")
