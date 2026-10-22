"""
HCSN-LM v1.1 — causal LM with periodic HCSN measurement passes.
RoPE + SwiGLU + KV-cache + SDPA. Generation-safe. FSDP-compatible.
Developer : PAI AND Yoon A Limsuwan
Organizations : MSPS NETWORK
Licences : MIT
Year : 2026
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

# --------------------------------------------------------------- RoPE ---
class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq: int = 8192):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)                    # (max_seq, hd/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        # x: (B, H, T, head_dim)
        T = x.shape[-2]
        cos = self.cos[offset:offset + T, None, :]     # (T, 1, hd/2)
        sin = self.sin[offset:offset + T, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        out = torch.stack([out[..., ::2], out[..., 1::2]], dim=-2)
        out = out.reshape(x.shape)
        return out

# ------------------------------------------------- Causal attention ---
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, kv_cache=None, offset=0, rope=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B,H,T,dh)
        
        if rope is not None:
            q = rope(q, offset)
            k = rope(k, offset)
            
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            
        if offset == 0 and kv_cache is None and self.dropout == 0.0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            Tq, Tk = T, k.shape[2]
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=Tk - Tq)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (k, v)

# ------------------------------------------------------------- SwiGLU ---
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8 / 3):
        super().__init__()
        d_hidden = int(mult * d_model)
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w3 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ---------------------------------------------------------- HCSN pass ---
class HCSNPass(nn.Module):
    def __init__(self, d_model: int, n_scales: int = 4):
        super().__init__()
        self.n_scales = n_scales
        self.scale_emb = nn.Parameter(torch.randn(n_scales, d_model) * 0.2)
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, n_scales),
        )
        output_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.outcome_net = output_net
        self.pointer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.Tanh(),
        )
        self.log_chi = nn.Parameter(torch.tensor(-3.0))
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))
        self.lambda_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1, keepdim=True)
        w = F.softmax(self.scale_net(pooled), dim=-1)   # (B,1,S)
        scales = torch.einsum("bs,sd->bd", w, self.scale_emb)
        bias = self.outcome_net(pooled + scales)        # (B,1,d)
        chi = torch.sigmoid(self.log_chi)
        pointer = self.pointer(x)
        gamma = F.softplus(self.log_gamma)
        damped = x * torch.exp(-gamma) + (1.0 - torch.exp(-gamma)) * pooled
        gate = self.lambda_gate(x + bias)
        return damped + chi * gate * pointer

# -------------------------------------------------------------- Block ---
class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout, rope):
        super().__init__()
        self.n1 = nn.RMSNorm(d_model)
        self.n2 = nn.RMSNorm(d_model)
        self.rope = rope
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.mlp = SwiGLU(d_model)

    def forward(self, x, kv_cache=None, offset=0):
        a, kv = self.attn(self.n1(x), kv_cache, offset, self.rope)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, kv

# --------------------------------------------------------- HCSN-LM ---
class HCSNLMD(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12,
                 n_heads=12, max_seq=4096, dropout=0.0,
                 hcsn_every=4, tie_embeddings=True):
        super().__init__()
        self.max_seq = max_seq
        self.hcsn_every = hcsn_every
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model // n_heads, max_seq=max_seq)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, dropout, rope=self.rope)
            for _ in range(n_layers)
        ])
        self.hcsn = HCSNPass(d_model)
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids: Tensor, targets: Optional[Tensor] = None):
        x = self.embed(ids)
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)
            if self.hcsn_every > 0 and (i + 1) % self.hcsn_every == 0:
                x = self.hcsn(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: Tensor, max_new: int = 64,
                 temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.95) -> Tensor:
        caches = [None] * len(self.blocks)
        out = ids
        for step in range(max_new):
            x = out if step == 0 else out[:, -1:]
            offset = out.shape[1] - x.shape[1]
            h = self.embed(x)
            for i, blk in enumerate(self.blocks):
                h, kv = blk(h, caches[i], offset)
                caches[i] = kv
                if self.hcsn_every > 0 and (i + 1) % self.hcsn_every == 0:
                    h = self.hcsn(h)
            logits = self.head(self.norm(h[:, -1]))
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p:
                sorted_l, idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_l, -1)
                cum = probs.cumsum(-1)
                sorted_l[cum > top_p] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, idx, sorted_l)
            probs = torch.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
        return out

HCSN_LM_CONFIGS = {
    "debug": dict(d_model=128,  n_layers=4,  n_heads=4),
    "toy":   dict(d_model=256,  n_layers=8,  n_heads=8),
    "small": dict(d_model=512,  n_layers=16, n_heads=8),
    "base":  dict(d_model=768,  n_layers=24, n_heads=12),
    "large": dict(d_model=1024, n_layers=36, n_heads=16),
    "xl":    dict(d_model=1536, n_layers=48, n_heads=24),
}

def build_lm(name: str, vocab_size: int, **kw) -> HCSNLMD:
    return HCSNLMD(vocab_size=vocab_size, **{**HCSN_LM_CONFIGS[name], **kw})
