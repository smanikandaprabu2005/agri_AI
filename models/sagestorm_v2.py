"""
models/sagestorm_v2.py
======================
SageStorm V2.2 — Regularization-Enhanced Architecture

Key changes vs V2.1:
  + Stochastic depth (drop-path) per block — strongest regularizer for small data
  + Linearly increasing drop-path rate (deeper = higher rate)
  + Reduced default size (40M) to match 13M token pretrain budget
  + LayerScale initialization for more stable deep training
  + Kept all V2.1 improvements (RoPE, GQA, SwiGLU, Flash Attn)
"""

import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    VOCAB_SIZE, CONTEXT_LENGTH, EMBED_DIM, FFN_MULT,
    NUM_HEADS, KV_HEADS, NUM_LAYERS, DROPOUT,
    EOS_ID, GEN_MAX_TOKENS, GEN_TEMPERATURE, GEN_MIN_TOKENS,
    GEN_TOP_K, GEN_TOP_P, GEN_REP_PENALTY, DEVICE,
    USE_GRADIENT_CHECKPOINTING,
)

# Max stochastic depth rate — increases linearly across layers
DROP_PATH_MAX = 0.10   # 10% at the deepest layer


def _ffn_hidden(embed_dim: int, mult: float = FFN_MULT) -> int:
    """SwiGLU hidden dim rounded to nearest 64."""
    raw = int(embed_dim * mult)
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
#  Rotary Position Embedding (RoPE) — LLaMA-3 style
# ══════════════════════════════════════════════════════════════
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq: int = CONTEXT_LENGTH,
                 base: float = 500_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build(max_seq)

    def _build(self, seq_len: int):
        t   = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        self.register_buffer("cos_c", emb.cos()[None, None])
        self.register_buffer("sin_c", emb.sin()[None, None])
        self._len = seq_len

    def forward(self, seq_len: int):
        if seq_len > self._len:
            self._build(seq_len)
        return self.cos_c[:, :, :seq_len], self.sin_c[:, :, :seq_len]


def _rot_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + _rot_half(x) * sin


# ══════════════════════════════════════════════════════════════
#  SwiGLU Feed-Forward
# ══════════════════════════════════════════════════════════════
class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float = DROPOUT):
        super().__init__()
        h         = _ffn_hidden(dim)
        self.gate = nn.Linear(dim, h, bias=False)
        self.up   = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ══════════════════════════════════════════════════════════════
#  Grouped Query Attention (GQA)
# ══════════════════════════════════════════════════════════════
class GQA(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_heads: int,
                 dropout: float = DROPOUT):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % kv_heads == 0
        self.n_heads  = n_heads
        self.kv_heads = kv_heads
        self.hd       = dim // n_heads
        self.groups   = n_heads // kv_heads
        self.q        = nn.Linear(dim, n_heads  * self.hd, bias=False)
        self.k        = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.v        = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.o        = nn.Linear(dim, dim, bias=False)
        self.dp       = nn.Dropout(dropout)
        self.rope     = RotaryEmbedding(self.hd)

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
#  Transformer Block with Stochastic Depth (Drop-Path)
#
#  Drop-path randomly skips entire residual branches during training.
#  This is stronger than dropout for small-data regimes because it
#  forces different subsets of layers to learn independently.
#  Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
# ══════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, kv_heads: int,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.n1           = RMSNorm(dim)
        self.attn         = GQA(dim, n_heads, kv_heads)
        self.n2           = RMSNorm(dim)
        self.ff           = SwiGLU(dim)
        self.drop_path_p  = drop_path_rate

        # LayerScale: learn to scale residual branch magnitude
        # Initialized near zero so early training is stable
        # Reference: "Going deeper with Image Transformers" (Touvron et al., 2021)
        init_val = 1e-4
        self.ls1 = nn.Parameter(torch.ones(dim) * init_val)
        self.ls2 = nn.Parameter(torch.ones(dim) * init_val)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic depth: during training, randomly drop entire residual
        branch with probability drop_path_p. At inference, scale by
        survival probability (implicit via the division by keep_prob).
        """
        if not self.training or self.drop_path_p <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_p
        # (B, 1, 1) mask broadcasts over sequence and channel dims
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob)
        return x * mask.float() / keep_prob

    def forward(self, x, mask=None):
        # Pre-norm + LayerScale + drop-path on each branch
        x = x + self._drop_path(self.ls1 * self.attn(self.n1(x), mask))
        x = x + self._drop_path(self.ls2 * self.ff(self.n2(x)))
        return x

    def forward_with_ckpt(self, x, mask=None):
        """Gradient-checkpointed forward for VRAM savings."""
        def attn_fn(x_in):
            return self.ls1 * self.attn(self.n1(x_in), mask)
        x = x + self._drop_path(grad_ckpt(attn_fn, x, use_reentrant=False))
        x = x + self._drop_path(self.ls2 * self.ff(self.n2(x)))
        return x


# ══════════════════════════════════════════════════════════════
#  SageStorm V2.2 — Full Model
# ══════════════════════════════════════════════════════════════
class SageStormV2(nn.Module):
    def __init__(
        self,
        vocab_size      : int   = VOCAB_SIZE,
        context_length  : int   = CONTEXT_LENGTH,
        embed_dim       : int   = EMBED_DIM,
        num_heads       : int   = NUM_HEADS,
        kv_heads        : int   = KV_HEADS,
        num_layers      : int   = NUM_LAYERS,
        dropout         : float = DROPOUT,
        use_grad_ckpt   : bool  = USE_GRADIENT_CHECKPOINTING,
        drop_path_max   : float = DROP_PATH_MAX,
    ):
        super().__init__()
        self.vocab_size     = vocab_size
        self.context_length = context_length
        self.use_grad_ckpt  = use_grad_ckpt

        self.emb  = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)

        # Linearly increasing drop-path rate:
        # Layer 0 gets 0.0, last layer gets drop_path_max
        # This preserves low-level features while regularizing high-level ones
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, kv_heads,
                drop_path_rate = drop_path_max * (i / max(num_layers - 1, 1))
            )
            for i in range(num_layers)
        ])

        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying — lm_head shares embedding weights
        self.head.weight = self.emb.weight

        # Causal mask
        mask = torch.full((context_length, context_length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

        self._init_weights()
        pc = self.param_count()
        print(f"[Model] SageStorm V2.2 — {pc['total_M']}M params "
              f"({num_layers}L × {embed_dim}d × {num_heads}h, "
              f"ctx={context_length}, drop_path_max={drop_path_max})")

    def _init_weights(self):
        n   = len(self.blocks)
        std = 0.02
        for name, p in self.named_parameters():
            if "ls1" in name or "ls2" in name:
                continue   # LayerScale handled in Block.__init__
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, 0.0, std)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "scale" in name:
                nn.init.ones_(p)
        # Depth-scaled output projections (GPT-2 / LLaMA trick)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight,  std=std / math.sqrt(2 * n))
            nn.init.normal_(b.ff.down.weight,  std=std / math.sqrt(2 * n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h    = self.drop(self.emb(x))
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            if self.use_grad_ckpt and self.training:
                h = block.forward_with_ckpt(h, mask)
            else:
                h = block(h, mask)
        return self.head(self.norm(h))

    @torch.no_grad()
    def generate(
        self,
        prompt_ids         : list,
        max_tokens         : int   = GEN_MAX_TOKENS,
        min_tokens         : int   = GEN_MIN_TOKENS,
        temperature        : float = GEN_TEMPERATURE,
        top_k              : int   = GEN_TOP_K,
        top_p              : float = GEN_TOP_P,
        eos_id             : int   = EOS_ID,
        repetition_penalty : float = GEN_REP_PENALTY,
        device             : str   = DEVICE,
    ) -> list:
        self.eval()
        ids       = list(prompt_ids[-self.context_length :])
        generated = []
        seen      = {}

        for step in range(max_tokens):
            inp    = torch.tensor([ids[-self.context_length :]], dtype=torch.long,
                                  device=device)
            logits = self.forward(inp)[0, -1, :].float()

            # Frequency-scaled repetition penalty
            for tid, cnt in seen.items():
                penalty = repetition_penalty ** min(cnt, 4)
                logits[tid] = logits[tid] / penalty if logits[tid] > 0 \
                              else logits[tid] * penalty

            if step < min_tokens:
                logits[eos_id] = float("-inf")

            logits /= max(temperature, 1e-6)

            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")

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
            seen[nxt] = seen.get(nxt, 0) + 1

            if nxt == eos_id:
                break

        return generated

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "config": {
                "vocab_size"     : self.vocab_size,
                "context_length" : self.context_length,
                "embed_dim"      : self.emb.embedding_dim,
                "num_heads"      : self.blocks[0].attn.n_heads,
                "kv_heads"       : self.blocks[0].attn.kv_heads,
                "num_layers"     : len(self.blocks),
                "dropout"        : self.blocks[0].ff.drop.p,
                "drop_path_max"  : self.blocks[-1].drop_path_p,
            }
        }, path)
        print(f"[Model] Saved {self.param_count()['total_M']}M params → {path}")

    @classmethod
    def load(cls, path: str, device: str = DEVICE) -> "SageStormV2":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        cfg   = ckpt["config"]
        model = cls(**{k: v for k, v in cfg.items()}, use_grad_ckpt=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[Model] Loaded {model.param_count()['total_M']}M params ← {path}")
        return model.to(device)

    def param_count(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        unique = sum(p.numel() for p in set(self.parameters()))
        return {
            "total"   : total,
            "unique"  : unique,
            "total_M" : round(total   / 1e6, 2),
            "unique_M": round(unique  / 1e6, 2),
        }


if __name__ == "__main__":
    m = SageStormV2(use_grad_ckpt=False)
    c = m.param_count()
    print(f"Total params : {c['total_M']}M")
    print(f"Unique params: {c['unique_M']}M  "
          f"(weight tying saves {c['total_M']-c['unique_M']:.1f}M)")
    x   = torch.randint(0, VOCAB_SIZE, (2, 64))
    out = m(x)
    print(f"Forward: {list(x.shape)} → {list(out.shape)} ✓")

    # Verify drop-path rates increase with depth
    print("\nDrop-path rates per layer:")
    for i, blk in enumerate(m.blocks):
        print(f"  Layer {i:2d}: {blk.drop_path_p:.4f}")