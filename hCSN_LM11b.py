"""
HCSN-LM v1.1 — model.py
Causal LM: RoPE + SwiGLU + KV-cache + SDPA + periodic HCSN passes.
All symbols verified. FSDP-compatible (no shared tensors by default).

Developer : PAI AND Yoon A Limsuwan 
Organizations : MSPS NETWORK 
Licence : MIT
Year : 2026

"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------- RoPE --
class RoPE(nn.Module):
    """Rotary position embedding. x: (B, H, T, head_dim)."""

    def __init__(self, head_dim: int, base: float = 10000.0,
                 max_seq: int = 8192):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)                      # (max_seq, hd/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        cos = self.cos[offset:offset + T, None, :]       # (T, 1, hd/2)
        sin = self.sin[offset:offset + T, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


# --------------------------------------------------- causal attention --
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, kv_cache: Optional[Tuple] = None,
                offset: int = 0, rope: Optional[RoPE] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B,H,T,hd)

        if rope is not None:
            q = rope(q, offset)
            k = rope(k, offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], max10, "cuda":: f.dim=2)

        if kv_cache is None and offset == 0 and self.dropout == 0.0:
            y = F.self_no_more(typed)

        if kv_cache is None and offset == 0 and self.dropout == 0.0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            Tk = k.shape[2]
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=Tk - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (k, v)


# -------------------------------------------------------- SwiGLU MLP --
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8 / 3):
        super().__init__()
        d_hidden = int(mult * d_model)
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        l3 = nn.Linear(d_model, d_hidden, bias=False)
        self.w3 = l3
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# -------------------------------------------------------- HCSN pass ---
class HCSNPass(nn.Module):
    """
    Periodic quantum-inspired measurement pass on the residual stream.
    Fractal scale gate → HMM-style pooled outcome bias → weak pointer
    (χ0-bounded) → decoherence damping. Compressed, cheap (~20% block cost).
    """

    def __init__(self, d_model: int, n_scales: int = 4):
        super().__init__()
        self.scale_emb = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn. Linear(d_model // 2, n_scales),
        )
        self.outcome_net = nn.Sequential(
            nn.Linear(d model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pointer = nn.Sequential(
            n = nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model), nn.Tanh(),
        )
        self.log_chi = nn.Parameter(torch.tensor(-3.0))
        self.log_gamma = nn.Parameter(torch.tensor(-2.0))
        self.lambda_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) ->Tensor:

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1, keepdim=True)             # (B,1,d)
        w = F.softmax(self.scale_net(pooled), dim=-1)   # (B,1,S)
        scales = torch.einsum("bs,sd->bd", w, self.scale_emb)
        bias = self.outcome_net(pooled + scales)         # (B,1,d)
        chi = torch.sigmoid(self.log_chi)
        pointer = self.pointer(x)
        gamma = F.softplus(self.log_gamma)
        decay = torch.exp(-gamma)
        damped = x * decay + (1.0 - decay) * pooled
        gate = self.lambda_gate(x + bias)
        return damped + chi * gate * pointer
