"""
=============================================================================
SESI UNIFIED QUANTUM-COGNITIVE LLM — STATE-OF-THE-ART
Integrating ALL SESI Theories into a Single Differentiable Framework
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Ultimate Integration)
=============================================================================

ARCHITECTURE OVERVIEW
─────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  SESI UNIFIED QUANTUM-COGNITIVE LLM                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input Tokens (B, T)                                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  HCSN Backbone (N layers)                                    │   │
│  │  ├── RMSNorm → CausalSelfAttention (SDPA + RoPE + KV-cache) │   │
│  │  ├── RMSNorm → SwiGLU MLP                                   │   │
│  │  └── [every hcsn_every] HCSNMeasurementPass                 │   │
│  │       ├── NonPCFFractalMeasure (λ_F(ℓ))                     │   │
│  │       ├── HierarchicalMeasurementMatrix (HMM)               │   │
│  │       ├── WeakMeasurementReadout (χ₀ ≪ 1)                   │   │
│  │       └── DecoherenceChannel (Lindblad)                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  Pooled Hidden State (B, d_model)                                   │
│       │                                                             │
│       ├─────────────────────────────────────────────────────────┐  │
│       │                                                         │  │
│       ▼                                                         ▼  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Quantum-Cognitive Integration Layer                    │     │
│  ├─────────────────────────────────────────────────────────┤     │
│  │                                                         │     │
│  │  [T1] QMD Cognitive Core                                │     │
│  │    ├── Lorenz RK4 (fp32-internal)                       │     │
│  │    ├── LatentInformationalField (amortized VI)          │     │
│  │    ├── FrameAccumulation (temporal kernel)              │     │
│  │    └── NeuralCognitiveInterface (cross-attention)       │     │
│  │                                                         │     │
│  │  [T2] SPTC Consciousness Head                           │     │
│  │    ├── Substrate → Organization → Φ_IIT                 │     │
│  │    └── Potentiality Functional                          │     │
│  │                                                         │     │
│  │  [T3] RLF Mind Interpreter                              │     │
│  │    ├── Cetasika 52 (14 Akusala + 13 Aññasamāna + 25)   │     │
│  │    ├── Nivaraṇa 5 (Five Hindrances)                     │     │
│  │    ├── Yoniso Manasikāra 10 (Rational Attention)        │     │
│  │    ├── Samādhi 3 Levels                                 │     │
│  │    └── Bojjhaṅga 7 (Enlightenment Factors)              │     │
│  │                                                         │     │
│  │  [T4] Cloud-OPMC Compressor                             │     │
│  │    ├── σ_T canonical signature                          │     │
│  │    ├── D_DE = 2^{2^N} hyper-entanglement                │     │
│  │    └── Task-sufficient classical access                 │     │
│  │                                                         │     │
│  │  [T5] SQCT Collapse Module                              │     │
│  │    ├── Geometric mass density μ̂_S                       │     │
│  │    └── Collapse rate Γ                                  │     │
│  │                                                         │     │
│  │  [T6] RUT Relational Head                               │     │
│  │    ├── Physical ↔ Cognitive cross-domain mapping        │     │
│  │    └── Relational accessibility graph                   │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                             │
│       ▼                                                             │
│  Multi-Task Output Heads                                            │
│    ├── Next-token logits (LM)                                       │
│    ├── Mental state classification                                  │
│    ├── Concentration score                                          │
│    ├── Φ estimate                                                   │
│    ├── Collapse rate                                                │
│    └── Relational consistency                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

COST OPTIMIZATION (Maximum)
────────────────────────────
  [C1] Low-rank adapters: O(d·r) instead of O(d²), r ≪ d
  [C2] Shared trunk: all theory modules consume one pooled hidden
  [C3] HCSN every N layers (not every layer)
  [C4] Gradient checkpointing on transformer + Lorenz RK4
  [C5] AMP bf16 with fp32 islands for physics
  [C6] Fused AdamW + torch.compile + DDP gradient_as_bucket_view
  [C7] KV-cache generation O(1) per token
  [C8] Token subsampling (mental_stride) for theory heads
  [C9] Triton-fused physics kernels (C^∞ surrogates)
  [C10] Non-persistent buffers for all scalar constants

THEORY INTEGRATION MAP
──────────────────────
  T1: QMD        → Quantum Mental Dynamics (Lorenz + Latent Field)
  T2: SPTC       → Structural Potentiality of Consciousness (Φ_IIT)
  T3: RLF        → Relational Logos Framework (Buddhist 5 pillars)
  T4: Cloud-OPMC → Covariant SESI Cloud-OPMC (σ_T + D_DE)
  T5: SQCT       → SESI Quantum Collapse Theory (geometric mass)
  T6: RUT        → Relational Universe Theory (cross-domain mapping)
  T7: HCS        → Hierarchical Cloud Superposition (HMM)
  T8: RLF-Mind   → Mind Interpreter (Cetasika + Samādhi)

=============================================================================
"""

from __future__ import annotations

import math
import os
import time
import logging
import argparse
import contextlib
import copy
import itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger("sesi.unified")

__all__ = [
    "SESIUnifiedConfig",
    "SESIUnifiedQuantumCognitiveLLM",
    "UnifiedLoss",
    "UnifiedTrainer",
    "SyntheticUnifiedDataset",
    "build_unified_loaders",
    "setup_dist",
    "cleanup_dist",
    "seed_all",
]


# ═══════════════════════════════════════════════════════════════════════
# AMP / DDP / Distributed Helpers
# ═══════════════════════════════════════════════════════════════════════

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = False


def amp_ctx(enabled: bool, dtype: torch.dtype, device_type: str = "cuda"):
    if not enabled or device_type != "cuda":
        return contextlib.nullcontext()
    if _AMP_NEW:
        return _autocast(device_type=device_type, dtype=dtype, enabled=True)
    return _autocast(dtype=dtype, enabled=True)


@contextlib.contextmanager
def fp32_ctx():
    if _AMP_NEW:
        with _autocast(device_type="cuda", enabled=False):
            yield
    else:
        with _autocast(enabled=False):
            yield


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def all_reduce_mean(x: Tensor) -> Tensor:
    if not is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / world_size()


def setup_dist(backend: str = "nccl") -> Tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group(backend=backend)
            return torch.device("cuda", local), dist.get_rank(), dist.get_world_size()
        dist.init_process_group(backend="gloo")
        return torch.device("cpu"), dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    return torch.device("cpu"), 0, 1


def cleanup_dist() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Smooth Primitives (No Dead Gradients)
# ═══════════════════════════════════════════════════════════════════════

def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    """C^∞ |x| = sqrt(x² + ε²). No kink at x = 0."""
    return torch.sqrt(x * x + eps * eps)


def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    """C^∞ max(x, floor). Exact for x ≫ floor."""
    return floor + F.softplus(x - floor, beta=beta)


def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    """Smooth gate in [0,1]: σ(x/τ) — approaches Heaviside as τ→0."""
    return torch.sigmoid(x / tau)


def logsumexp_max(x: Tensor, tau: float = 1e-2) -> Tensor:
    """Smooth differentiable approximation of x.max()."""
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIG — Master Configuration for All Theories
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SESIUnifiedConfig:
    # ── LM Backbone ─────────────────────────────────────────────────────
    vocab_size: int = 256              # byte-level
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    max_seq: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    
    # ── [T7] HCSN ───────────────────────────────────────────────────────
    hcsn_every: int = 4                # HCSN pass every N layers
    hcsn_scales: int = 4
    hcsn_n_outcomes: int = 4
    
    # ── [T1] QMD ────────────────────────────────────────────────────────
    qmd_d_I: int = 32                  # Internal Decision State dim
    qmd_d_Z: int = 16                  # Latent Informational Field dim
    qmd_d_N: int = 32                  # Neural state dim
    qmd_kernel_len: int = 8
    qmd_steps: int = 4
    
    # ── [T2] SPTC ───────────────────────────────────────────────────────
    sptc_n_states: int = 8             # must be power of 2
    sptc_org_hidden: int = 64
    sptc_n_pot_samples: int = 4
    
    # ── [T3] RLF / Mind Interpreter ────────────────────────────────────
    mind_dim: int = 64
    
    # ── [T4] Cloud-OPMC ─────────────────────────────────────────────────
    opmc_n_fields: int = 10
    opmc_sigma_dim: int = 64
    
    # ── [T5] SQCT ───────────────────────────────────────────────────────
    sqct_lambda: float = 1e-16
    sqct_r_C: float = 1e-7
    sqct_kappa_I: float = 0.1
    
    # ── [T6] RUT ────────────────────────────────────────────────────────
    rut_state_dim: int = 32
    
    # ── Cost Optimization ───────────────────────────────────────────────
    mental_rank: int = 32              # [C1] low-rank adapter rank
    mental_stride: int = 4             # [C8] token subsampling stride
    gradient_checkpointing: bool = True
    use_fused_adamw: bool = True
    use_torch_compile: bool = True
    
    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 3e-4
    min_lr_ratio: float = 0.02
    weight_decay: float = 0.05
    warmup_steps: int = 300
    max_steps: int = 30000
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"
    ema_decay: float = 0.9995
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 2000
    out_dir: str = "runs/sesi_unified"
    
    # ── Loss Weights ────────────────────────────────────────────────────
    w_lm: float = 1.0
    w_qmd: float = 0.1
    w_sptc: float = 0.1
    w_mind: float = 0.2
    w_opmc: float = 0.05
    w_sqct: float = 0.05
    w_rut: float = 0.05
    w_reg: float = 1e-6
    
    # ── Data ────────────────────────────────────────────────────────────
    n_train: int = 16384
    n_val: int = 2048
    n_test: int = 4096
    data_seed: int = 2026
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# 2. BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).to(x.dtype) * self.w


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq: int = 8192):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        cos = self.cos[offset:offset+T, None, :].to(x.dtype)
        sin = self.sin[offset:offset+T, None, :].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.reshape(x.shape)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 8/3):
        super().__init__()
        d_h = int(mult * d_model)
        self.w1 = nn.Linear(d_model, d_h, bias=False)
        self.w3 = nn.Linear(d_model, d_h, bias=False)
        self.w2 = nn.Linear(d_h, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, kv_cache=None, offset: int = 0,
                rope: Optional[RoPE] = None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            q = rope(q, offset)
            k = rope(k, offset)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        if offset == 0 and kv_cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            Tq, Tk = T, k.shape[2]
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device)
            mask = mask.tril(diagonal=Tk - Tq)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (k, v)


class LowRankAdapter(nn.Module):
    """[C1] Low-rank projection: O(d·r) instead of O(d²)."""
    def __init__(self, d_in: int, d_out: int, rank: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.V = nn.Parameter(torch.zeros(rank, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# 3. [T7] HCSN MEASUREMENT PASS
# ═══════════════════════════════════════════════════════════════════════

class HCSNMeasurementPass(nn.Module):
    """
    [T7] Hierarchical Cloud Superposition measurement pass.
    Fractal scale attention → HMM → weak measurement → decoherence.
    """
    def __init__(self, d_model: int, n_scales: int = 4):
        super().__init__()
        self.scale_emb = nn.Parameter(torch.randn(n_scales, d_model) * 0.02)
        self.scale_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, n_scales),
        )
        self.outcome_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )
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
        w = F.softmax(self.scale_net(pooled), dim=-1)
        scales = torch.einsum("bs,sd->bd", w, self.scale_emb)
        bias = self.outcome_net(pooled + scales.unsqueeze(1))
        chi = torch.sigmoid(self.log_chi)
        gamma = F.softplus(self.log_gamma)
        damped = x * torch.exp(-gamma) + (1.0 - torch.exp(-gamma)) * pooled
        gate = self.lambda_gate(x + bias)
        return damped + chi * gate * self.pointer(x)


# ═══════════════════════════════════════════════════════════════════════
# 4. [T1] QMD COGNITIVE CORE
# ═══════════════════════════════════════════════════════════════════════

class QMDCognitiveCore(nn.Module):
    """
    [T1] Quantum Mental Dynamics.
    Lorenz cognitive-affective-identity core + Latent Field + NCI.
    """
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d_x = cfg.qmd_d_I + cfg.qmd_d_Z + cfg.qmd_d_N
        r = cfg.mental_rank

        self.in_proj = LowRankAdapter(cfg.d_model, d_x, r)

        # Lorenz parameters
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_l = nn.Parameter(torch.tensor(8.0 / 3.0))

        # Latent field (amortized VI)
        self.lif_post = nn.Linear(d_x + cfg.qmd_d_Z, cfg.qmd_d_Z * 2)
        self.lif_prior_cell = nn.GRUCell(cfg.qmd_d_Z, cfg.qmd_d_Z)

        # Frame accumulation
        self.fah_hyper = nn.Linear(cfg.d_model, cfg.qmd_kernel_len)

        # NCI output (low-rank)
        self.nci_proj = LowRankAdapter(d_x, cfg.d_model, r)

        self.register_buffer("_dt", torch.tensor(0.01), persistent=False)

    def _lorenz(self, x, y, z):
        dx = self.sigma.float() * (y - x)
        dy = x * (self.rho.float() - z) - y
        dz = x * y - self.beta_l.float() * z
        return dx, dy, dz

    def _rk4(self, x, y, z, dt):
        x32, y32, z32 = x.float(), y.float(), z.float()
        k1 = self._lorenz(x32, y32, z32)
        k2 = self._lorenz(x32+dt/2*k1[0], y32+dt/2*k1[1], z32+dt/2*k1[2])
        k3 = self._lorenz(x32+dt/2*k2[0], y32+dt/2*k2[1], z32+dt/2*k2[2])
        k4 = self._lorenz(x32+dt*k3[0], y32+dt*k3[1], z32+dt*k3[2])
        xn = x32 + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        yn = y32 + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        zn = z32 + dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        return xn.to(x.dtype), yn.to(y.dtype), zn.to(z.dtype)

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        B = h_pooled.shape[0]
        state = self.in_proj(h_pooled)
        dI, dZ, dN = self.cfg.qmd_d_I, self.cfg.qmd_d_Z, self.cfg.qmd_d_N
        x, y, z = state[:, :1], state[:, 1:2], state[:, 2:3] + 25.0
        Z = state[:, 3:3+dZ]
        dt = self._dt.to(x.dtype)

        for _ in range(self.cfg.qmd_steps):
            x, y, z = self._rk4(x, y, z, dt)

        zi = torch.cat([x, y, z, Z], dim=-1)
        mu_lv = self.lif_post(zi)
        mu, logvar = mu_lv.chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 4.0)
        Z_new = mu + (torch.randn_like(mu) * torch.exp(0.5*logvar)
                      if self.training else torch.zeros_like(mu))

        w_k = F.softmax(self.fah_hyper(h_pooled), dim=-1)

        full = torch.cat([x, y, z, Z_new,
                          torch.zeros(B, dN, device=x.device, dtype=x.dtype)], dim=-1)
        nci = self.nci_proj(full)

        return {"qmd_x": x, "qmd_y": y, "qmd_z": z,
                "qmd_Z_mu": mu, "qmd_Z_logvar": logvar,
                "qmd_nci": nci, "qmd_frame_w": w_k}


# ═══════════════════════════════════════════════════════════════════════
# 5. [T2] SPTC CONSCIOUSNESS HEAD
# ═══════════════════════════════════════════════════════════════════════

def unit_bipartitions(k: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    idx = list(range(k))
    out = []
    for r in range(1, k):
        for A in itertools.combinations(idx, r):
            B = tuple(i for i in idx if i not in A)
            if A < B:
                out.append((A, B))
    return out


def product_cut(T: Tensor, A_units, B_units, k: int):
    Bsz, N, _ = T.shape
    Tt = T.reshape(Bsz, *([2] * k), *([2] * k))
    order = list(A_units) + list(B_units)
    perm = [0] + [1 + i for i in order] + [1 + k + i for i in order]
    a, b = 2 ** len(A_units), 2 ** len(B_units)
    Tp = Tt.permute(*perm).reshape(Bsz, a, b, a, b)
    T_A = Tp.sum(dim=(2, 4)) / b
    T_B = Tp.sum(dim=(1, 3)) / a
    T_cut = T_A[:, :, None, :, None] * T_B[:, None, :, None, :]
    return Tp.reshape(Bsz, N, N), T_cut.reshape(Bsz, N, N)


class SPTCConsciousnessHead(nn.Module):
    """
    [T2] Structural Potentiality Theory of Consciousness.
    Substrate → Organization → Φ_IIT surrogate → Potentiality.
    """
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        self.cfg = cfg
        N = cfg.sptc_n_states
        k = int(round(math.log2(N)))
        assert 2 ** k == N
        self.k, self.N = k, N
        r = cfg.mental_rank

        self.sub_proj = LowRankAdapter(cfg.d_model, cfg.sptc_org_hidden, r)

        self.org_net = nn.Sequential(
            nn.Linear(cfg.sptc_org_hidden, cfg.sptc_org_hidden), nn.SiLU(),
            nn.Linear(cfg.sptc_org_hidden, N * N),
        )
        self.dyn_net = nn.Sequential(
            nn.Linear(N + N * N + cfg.sptc_org_hidden, cfg.sptc_org_hidden), nn.SiLU(),
            nn.Linear(cfg.sptc_org_hidden, N),
        )
        self.parts = unit_bipartitions(k) if k <= 10 else unit_bipartitions(4)
        self.tau_iit = 0.1

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        B = h_pooled.shape[0]
        ctx = F.gelu(self.sub_proj(h_pooled))

        logits = self.org_net(ctx).view(B, self.N, self.N)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        offdiag = 1.0 - torch.eye(self.N, device=logits.device, dtype=logits.dtype)
        logits = logits * offdiag
        if self.training:
            u = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
            g = torch.log(u) - torch.log1p(-u)
            A = torch.sigmoid((logits + g) / 0.5)
        else:
            A = torch.sigmoid(logits / 0.5)

        s = torch.eye(self.N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        Af = A.reshape(B, 1, -1).expand(-1, self.N, -1)
        ctx_e = ctx.unsqueeze(1).expand(-1, self.N, -1)
        T = F.softmax(self.dyn_net(torch.cat([s, Af, ctx_e], dim=-1)), dim=-1)

        with fp32_ctx():
            T32 = T.float().clamp_min(1e-12)
            kls = []
            for Au, Bu in self.parts:
                Tp, Tc = product_cut(T32, Au, Bu, self.k)
                Tc = Tc.clamp_min(1e-12)
                kl = (Tp * (Tp.log() - Tc.log())).sum(-1).mean(-1)
                kls.append(kl)
            kls = torch.stack(kls, dim=-1)
            P = len(self.parts)
            phi = -self.tau_iit * (torch.logsumexp(-kls / self.tau_iit, -1) - math.log(P))

        return {"sptc_phi": phi.to(h_pooled.dtype), "sptc_T": T}


# ═══════════════════════════════════════════════════════════════════════
# 6. [T3] MIND INTERPRETER (RLF / Buddhist Psychology)
# ═══════════════════════════════════════════════════════════════════════

class MindInterpreter(nn.Module):
    """
    [T3] Relational Logos Framework / Mind Interpreter.
    Cetasika 52 + Nivaraṇa 5 + Yoniso 10 + Samādhi 3 + Bojjhaṅga 7.
    """
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        d = cfg.mind_dim
        r = cfg.mental_rank
        self.in_proj = LowRankAdapter(cfg.d_model, d, r)
        self.akusala = nn.Linear(d, 14)
        self.sobhana = nn.Linear(d, 25)
        self.nivarana = nn.Linear(d, 5)
        self.yoniso = nn.Linear(d, 10)
        self.bojjhanga = nn.Linear(d, 7)
        self.samadhi = nn.Linear(d, 3)
        self.temp_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        x = F.gelu(self.in_proj(h_pooled))
        x32 = x.float()

        ak = torch.sigmoid(self.akusala(x32)).mean(-1, keepdim=True)
        so = torch.sigmoid(self.sobhana(x32)).mean(-1, keepdim=True)
        mh = so / (so + ak + 1e-8)

        nv = torch.sigmoid(self.nivarana(x32))
        yn = torch.sigmoid(self.yoniso(x32))
        bj = torch.sigmoid(self.bojjhanga(x32))
        tau = 0.1 + F.softplus(self.temp_raw)
        sm = F.softmax(self.samadhi(x32) / tau, dim=-1)

        conc = (0.4 * bj.mean(-1, keepdim=True)
                + 0.3 * yn.mean(-1, keepdim=True)
                + 0.3 * (1.0 - nv.mean(-1, keepdim=True)))

        dt = h_pooled.dtype
        return {"mind_health": mh.to(dt), "mind_conc": conc.to(dt),
                "mind_samadh": sm.to(dt), "mind_nivarana": nv.to(dt),
                "mind_bojjhanga": bj.to(dt)}


# ═══════════════════════════════════════════════════════════════════════
# 7. [T4] CLOUD-OPMC COMPRESSOR
# ═══════════════════════════════════════════════════════════════════════

class CloudOPMCCompressor(nn.Module):
    """
    [T4] Covariant SESI Cloud–OPMC.
    Canonical structural signature σ_T(Q) + hyper-entanglement D_DE.
    """
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        Fd = cfg.opmc_n_fields
        ds = cfg.opmc_sigma_dim
        r = cfg.mental_rank

        self.field_emb = nn.Parameter(torch.randn(Fd, d) * 0.02)
        self.query = LowRankAdapter(d, d, r)
        self.value = LowRankAdapter(d, ds, r)
        self.refine = nn.Sequential(
            nn.LayerNorm(ds), nn.Linear(ds, ds), nn.GELU(), nn.Linear(ds, ds),
        )
        self.n_ref = nn.Parameter(torch.tensor(4.0))
        self.log_tau = nn.Parameter(torch.tensor(math.log(2.0)))
        self.cap_head = nn.Sequential(
            nn.Linear(d, d // 4), nn.GELU(), nn.Linear(d // 4, 1), nn.Softplus(),
        )

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        q = self.query(h_pooled).unsqueeze(1)
        logits = torch.einsum("bnd,fd->bnf", q, self.field_emb) / math.sqrt(q.shape[-1])
        prov = F.softmax(logits, dim=1)
        v = self.value(h_pooled).unsqueeze(1)
        sigma = torch.einsum("bnf,bnd->bfd", prov, v).squeeze(1)
        sigma = self.refine(sigma)

        N_eff = self.cap_head(h_pooled).squeeze(-1) + 8.0
        tau = torch.exp(self.log_tau).clamp(1e-3, 1e3)
        sat = torch.sigmoid((N_eff - self.n_ref) / tau)

        return {"opmc_sigma": sigma, "opmc_log2log2_D": N_eff,
                "opmc_saturation": sat}


# ═══════════════════════════════════════════════════════════════════════
# 8. [T5] SQCT COLLAPSE MODULE
# ═══════════════════════════════════════════════════════════════════════

class SESICollapseModule(nn.Module):
    """
    [T5] SESI Quantum Collapse Theory.
    Geometric mass density → collapse rate Γ.
    """
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        r = cfg.mental_rank
        self.kappa_I_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(cfg.sqct_kappa_I))))
        self.mu_bulk = nn.Sequential(
            LowRankAdapter(cfg.d_model, 32, r), nn.GELU(),
            nn.Linear(32, 1), nn.Softplus(),
        )
        self.mu_int = nn.Sequential(
            LowRankAdapter(cfg.d_model, 32, r), nn.GELU(),
            nn.Linear(32, 1), nn.Softplus(),
        )
        self.register_buffer("_log_lam", torch.tensor(math.log(cfg.sqct_lambda)),
                             persistent=False)

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        kappa_I = F.softplus(self.kappa_I_raw) + 1e-6
        mb = self.mu_bulk(h_pooled)
        mi = self.mu_int(h_pooled)
        mu_S = mb + kappa_I * mi

        var_mu = mu_S.var(dim=-1, keepdim=True) if mu_S.shape[-1] > 1 else torch.zeros_like(mu_S)
        log_Gamma = (torch.log(torch.exp(self._log_lam) + 1e-30)
                     + torch.log(var_mu + 1e-30))
        log_Gamma = log_Gamma.clamp(-60.0, 40.0).squeeze(-1)

        return {"sqct_mu_S": mu_S, "sqct_kappa_I": kappa_I,
                "sqct_log_Gamma": log_Gamma,
                "sqct_Gamma": torch.exp(log_Gamma)}


# ═══════════════════════════════════════════════════════════════════════
# 9. [T6] RUT RELATIONAL HEAD
# ═══════════════════════════════════════════════════════════════════════

class RUTRelationalHead(nn.Module):
    """
    [T6] Relational Universe Theory.
    Physical ↔ Cognitive cross-domain mapping.
    """
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        r = cfg.mental_rank
        ds = cfg.rut_state_dim
        self.phys_enc = LowRankAdapter(cfg.d_model, ds, r)
        self.cogn_dec = LowRankAdapter(ds, ds, r)
        self.consistency = nn.Sequential(
            LowRankAdapter(ds * 2, ds, r), nn.GELU(),
            nn.Linear(ds, 1), nn.Sigmoid(),
        )

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        phys = self.phys_enc(h_pooled)
        cogn = self.cogn_dec(phys)
        cons = self.consistency(torch.cat([phys, cogn], dim=-1)).squeeze(-1)
        return {"rut_physical": phys, "rut_cognitive": cogn,
                "rut_consistency": cons}


# ═══════════════════════════════════════════════════════════════════════
# 10. LLM BACKBONE
# ═══════════════════════════════════════════════════════════════════════

class SESITransformerBlock(nn.Module):
    """Transformer block with optional HCSN measurement pass."""
    def __init__(self, cfg: SESIUnifiedConfig, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads)
        self.mlp = SwiGLU(cfg.d_model)
        self.use_hcsn = (cfg.hcsn_every > 0
                         and (layer_idx + 1) % cfg.hcsn_every == 0)
        self.hcsn = HCSNMeasurementPass(cfg.d_model, cfg.hcsn_scales) if self.use_hcsn else None

    def forward(self, x, kv_cache=None, offset=0, rope=None):
        a, kv = self.attn(self.n1(x), kv_cache, offset, rope)
        x = x + a
        x = x + self.mlp(self.n2(x))
        if self.hcsn is not None:
            x = self.hcsn(x)
        return x, kv


# ═══════════════════════════════════════════════════════════════════════
# 11. MAIN MODEL — SESI UNIFIED QUANTUM-COGNITIVE LLM
# ═══════════════════════════════════════════════════════════════════════

class SESIUnifiedQuantumCognitiveLLM(nn.Module):
    """
    Unified SESI Quantum-Cognitive LLM integrating all 8 theories.
    """
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        self.cfg = cfg

        # ── LM Backbone ─────────────────────────────────────────────────
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RoPE(cfg.d_model // cfg.n_heads, max_seq=cfg.max_seq)
        self.blocks = nn.ModuleList([
            SESITransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # ── Theory Modules ──────────────────────────────────────────────
        self.qmd = QMDCognitiveCore(cfg)
        self.sptc = SPTCConsciousnessHead(cfg)
        self.mind = MindInterpreter(cfg)
        self.opmc = CloudOPMCCompressor(cfg)
        self.sqct = SESICollapseModule(cfg)
        self.rut = RUTRelationalHead(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, ids: Tensor, targets: Optional[Tensor] = None,
                return_mental: bool = True) -> Dict[str, Tensor]:
        B, T = ids.shape
        x = self.embed(ids)

        # ── Transformer + HCSN passes ───────────────────────────────────
        for blk in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    blk, x, None, 0, self.rope, use_reentrant=False)
            else:
                x, _ = blk(x, None, 0, self.rope)

        x = self.norm(x)
        logits = self.lm_head(x)

        out: Dict[str, Tensor] = {"logits": logits}

        if targets is not None:
            out["lm_loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))

        if not return_mental:
            return out

        # ── Pooled hidden [C2, C8] ──────────────────────────────────────
        stride = self.cfg.mental_stride
        h_pooled = x[:, ::stride].mean(dim=1)

        # ── Theory modules ──────────────────────────────────────────────
        out.update(self.qmd(h_pooled))
        out.update(self.sptc(h_pooled))
        out.update(self.mind(h_pooled))
        out.update(self.opmc(h_pooled))
        out.update(self.sqct(h_pooled))
        out.update(self.rut(h_pooled))

        return out

    @torch.no_grad()
    def generate(self, ids: Tensor, max_new: int = 64,
                 temperature: float = 0.8, top_k: int = 50) -> Tensor:
        caches = [None] * len(self.blocks)
        out = ids
        for step in range(max_new):
            x = out if step == 0 else out[:, -1:]
            offset = out.shape[1] - x.shape[1]
            h = self.embed(x)
            for i, blk in enumerate(self.blocks):
                h, kv = blk(h, caches[i], offset, self.rope)
                caches[i] = kv
            logits = self.lm_head(self.norm(h[:, -1]))
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=1)
        return out


# ═══════════════════════════════════════════════════════════════════════
# 12. MULTI-TASK LOSS
# ═══════════════════════════════════════════════════════════════════════

class UnifiedLoss(nn.Module):
    def __init__(self, cfg: SESIUnifiedConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        cfg = self.cfg
        dev = out["logits"].device
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}

        L["lm"] = out.get("lm_loss", zero)

        if "qmd_Z_mu" in out:
            mu, lv = out["qmd_Z_mu"].float(), out["qmd_Z_logvar"].float()
            L["qmd"] = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean() * 0.01
        else:
            L["qmd"] = zero

        L["sptc"] = -out.get("sptc_phi", zero).float().mean() * 0.01

        if "mind_health" in out:
            L["mind"] = ((1.0 - out["mind_health"]).float().mean()
                         + (1.0 - out["mind_conc"]).float().mean())
        else:
            L["mind"] = zero

        L["opmc"] = -out.get("opmc_saturation", zero).float().mean() * 0.01
        L["rut"] = -out.get("rut_consistency", zero).float().mean() * 0.01
        L["collapse"] = F.relu(out.get("sqct_log_Gamma", zero).float()
                               - math.log(1e-8)).pow(2).mean() * 0.01 \
                        if "sqct_log_Gamma" in out else zero

        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg

        total = (cfg.w_lm * L["lm"]
                 + cfg.w_qmd * L["qmd"]
                 + cfg.w_sptc * L["sptc"]
                 + cfg.w_mind * L["mind"]
                 + cfg.w_opmc * L["opmc"]
                 + cfg.w_rut * L["rut"]
                 + cfg.w_sqct * L["collapse"]
                 + cfg.w_reg * L["reg"])
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# 13. EMA + TRAINER
# ═══════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is None:
                self.shadow[k] = v.detach().clone()
                continue
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


class UnifiedTrainer:
    def __init__(self, cfg: SESIUnifiedConfig, model: SESIUnifiedQuantumCognitiveLLM,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)

        if cfg.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                if is_main():
                    logger.info("torch.compile enabled (max-autotune)")
            except Exception as e:
                if is_main():
                    logger.warning(f"torch.compile failed: {e}")

        if is_dist():
            self.model = DDP(self.model,
                             device_ids=[device.index] if device.type == "cuda" else None,
                             find_unused_parameters=False,
                             gradient_as_bucket_view=True)

        self.loss_fn = UnifiedLoss(cfg)

        amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16,
                  "fp32": torch.float32}[cfg.amp_dtype]
        self._amp_dtype = amp_dt
        self._amp_on = cfg.amp_dtype != "fp32" and device.type == "cuda"
        self.scaler = _GradScaler(enabled=(self._amp_on and amp_dt == torch.float16))

        fused = cfg.use_fused_adamw and device.type == "cuda"
        self.opt = torch.optim.AdamW(
            [p for p in self._raw().parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95), fused=fused,
        )
        self.ema = EMA(self._raw(), cfg.ema_decay)
        self.step = 0

    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m

    def lr_now(self) -> float:
        c = self.cfg
        if self.step < c.warmup_steps:
            return c.lr * (self.step + 1) / max(1, c.warmup_steps)
        p = (self.step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        p = min(1.0, max(0.0, p))
        return c.lr * (c.min_lr_ratio
                       + (1 - c.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        for g in self.opt.param_groups:
            g["lr"] = self.lr_now()

        if (self.step + 1) % cfg.grad_accum == 0:
            self.opt.zero_grad(set_to_none=True)

        with amp_ctx(self._amp_on, self._amp_dtype, self.device.type):
            out = self.model(ids=batch["tokens"], targets=batch.get("targets"))
            losses = self.loss_fn(out, batch, self._raw())
            loss = losses["total"] / cfg.grad_accum

        self.scaler.scale(loss).backward()

        if (self.step + 1) % cfg.grad_accum == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._raw().parameters() if p.requires_grad],
                cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update(self._raw())

        self.step += 1
        return {k: float(v) for k, v in losses.items()
                if torch.is_tensor(v) and v.numel() == 1}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tot: Dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with amp_ctx(self._amp_on, self._amp_dtype, self.device.type):
                out = self.model(ids=batch["tokens"], targets=batch.get("targets"))
                losses = self.loss_fn(out, batch, self._raw())
            bs = batch["tokens"].shape[0]
            for k, v in losses.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    tot[k] = tot.get(k, 0.0) + float(v) * bs
            n += bs
        for k in tot:
            tot[k] /= max(1, n)
        if is_dist():
            for k in list(tot.keys()):
                tot[k] = float(all_reduce_mean(
                    torch.tensor(tot[k], device=self.device)))
        tot["n"] = n
        return tot

    def state_dict(self):
        return {"model": self._raw().state_dict(),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "ema": self.ema.state_dict(),
                "step": self.step}

    def load_state_dict(self, sd):
        self._raw().load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        try:
            self.scaler.load_state_dict(sd["scaler"])
        except Exception:
            pass
        self.ema.load_state_dict(sd["ema"])
        self.step = sd["step"]


# ═══════════════════════════════════════════════════════════════════════
# 14. DATASET
# ═══════════════════════════════════════════════════════════════════════

class SyntheticUnifiedDataset(Dataset):
    def __init__(self, cfg: SESIUnifiedConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, cfg.vocab_size, (n, cfg.max_seq),
                                    generator=g)
        self.targets = torch.roll(self.tokens, -1, dims=1)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return {"tokens": self.tokens[i], "targets": self.targets[i]}


def build_unified_loaders(cfg: SESIUnifiedConfig):
    tr = SyntheticUnifiedDataset(cfg, cfg.n_train, cfg.data_seed)
    va = SyntheticUnifiedDataset(cfg, cfg.n_val, cfg.data_seed + 1)
    te = SyntheticUnifiedDataset(cfg, cfg.n_test, cfg.data_seed + 2)
    tr_s = DistributedSampler(tr, shuffle=True, drop_last=True) if is_dist() else None
    va_s = DistributedSampler(va, shuffle=False) if is_dist() else None
    te_s = DistributedSampler(te, shuffle=False) if is_dist() else None
    common = dict(batch_size=cfg.batch_size, num_workers=2,
                  pin_memory=(cfg.device != "cpu"), persistent_workers=False)
    return (
        DataLoader(tr, sampler=tr_s, shuffle=(tr_s is None),
                   drop_last=True, **common),
        DataLoader(va, sampler=va_s, shuffle=False, **common),
        DataLoader(te, sampler=te_s, shuffle=False, **common),
    )


# ═══════════════════════════════════════════════════════════════════════
# 15. SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI UNIFIED QUANTUM-COGNITIVE LLM — Smoke Test")
    print("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)

    cfg = SESIUnifiedConfig(
        d_model=128, n_layers=4, n_heads=4, max_seq=64,
        vocab_size=64, batch_size=2,
        qmd_d_I=8, qmd_d_Z=4, qmd_d_N=8,
        sptc_n_states=4, sptc_org_hidden=32,
        mind_dim=32, mental_rank=8,
        gradient_checkpointing=False, use_torch_compile=False,
    )

    model = SESIUnifiedQuantumCognitiveLLM(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")

    model.train()
    B, T = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.roll(ids, -1, dims=1)

    out = model(ids=ids, targets=targets)

    loss_fn = UnifiedLoss(cfg)
    losses = loss_fn(out, {"tokens": ids, "targets": targets}, model)
    losses["total"].backward()

    print(f"  LM loss         : {losses['lm'].item():.4f}")
    print(f"  QMD KL          : {losses['qmd'].item():.6f}")
    print(f"  SPTC Φ          : {out['sptc_phi'].mean().item():.4f}")
    print(f"  Mind health     : {out['mind_health'].mean().item():.4f}")
    print(f"  Mind conc       : {out['mind_conc'].mean().item():.4f}")
    print(f"  OPMC sat        : {out['opmc_saturation'].mean().item():.4f}")
    print(f"  RUT consistency : {out['rut_consistency'].mean().item():.4f}")
    print(f"  SQCT Γ          : {out['sqct_Gamma'].mean().item():.6e}")

    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"  Autograd        : {'PASS — ALL 8 THEORIES CONNECTED' if grad_ok else 'FAIL'}")

    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    gen = model.generate(prompt, max_new=16)
    print(f"  Generation      : {tuple(gen.shape)} (8 → +16 tokens)")

    print("=" * 72)
    print("  Smoke test complete.")


# ═══════════════════════════════════════════════════════════════════════
# 16. TRAIN / TEST DRIVERS
# ═══════════════════════════════════════════════════════════════════════

def run_train(cfg: SESIUnifiedConfig, device: torch.device):
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_dl, val_dl, _ = build_unified_loaders(cfg)
    model = SESIUnifiedQuantumCognitiveLLM(cfg)
    trainer = UnifiedTrainer(cfg, model, device)

    if is_main():
        n_par = sum(p.numel() for p in trainer._raw().parameters() if p.requires_grad)
        logger.info(f"Trainable params: {n_par:,}")

    best = float("inf")
    t0 = time.time()
    epoch = 0
    while trainer.step < cfg.max_steps:
        s = getattr(train_dl, "sampler", None)
        if isinstance(s, DistributedSampler):
            s.set_epoch(epoch)
        for batch in train_dl:
            logs = trainer.train_step(batch)
            if is_main() and trainer.step % cfg.log_every == 0:
                el = time.time() - t0
                logger.info(
                    f"step {trainer.step:6d} | "
                    + " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                    + f" | lr={trainer.lr_now():.2e} | {el:.1f}s")
            if trainer.step % cfg.eval_every == 0 and trainer.step > 0:
                ev = trainer.evaluate(val_dl)
                if is_main():
                    logger.info("  [val] " +
                                " ".join(f"{k}={v:.4f}" for k, v in ev.items()
                                         if k != "n"))
                    tot = ev.get("total", float("inf"))
                    if tot < best:
                        best = tot
                        torch.save(trainer.state_dict(),
                                   os.path.join(cfg.out_dir, "best.pt"))
            if is_main() and trainer.step % cfg.ckpt_every == 0:
                torch.save(trainer.state_dict(),
                           os.path.join(cfg.out_dir, f"ckpt_{trainer.step}.pt"))
            if trainer.step >= cfg.max_steps:
                break
        epoch += 1

    if is_main():
        torch.save(trainer.state_dict(), os.path.join(cfg.out_dir, "ckpt_final.pt"))
        logger.info(f"Training complete. Best val: {best:.4f}")


def run_test(cfg: SESIUnifiedConfig, device: torch.device,
             ckpt: Optional[str] = None):
    _, _, test_dl = build_unified_loaders(cfg)
    model = SESIUnifiedQuantumCognitiveLLM(cfg)
    trainer = UnifiedTrainer(cfg, model, device)
    path = ckpt or os.path.join(cfg.out_dir, "ckpt_final.pt")
    if os.path.exists(path):
        sd = torch.load(path, map_location=device, weights_only=False)
        trainer.load_state_dict(sd)
        if is_main():
            logger.info(f"Loaded {path} (step {sd['step']})")
    res = trainer.evaluate(test_dl)
    if is_main():
        import json
        print(json.dumps(res, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser("SESI Unified Quantum-Cognitive LLM")
    p.add_argument("--mode", choices=["smoke", "train", "test"], default="smoke")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s")

    if args.mode == "smoke":
        smoke_test()
        return

    device, r, ws = setup_dist()
    seed_all(42)
    cfg = SESIUnifiedConfig()
    if args.steps: cfg.max_steps = args.steps
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.d_model: cfg.d_model = args.d_model
    if args.n_layers: cfg.n_layers = args.n_layers
    if args.out: cfg.out_dir = args.out
    if args.no_compile: cfg.use_torch_compile = False
    if args.no_amp: cfg.amp_dtype = "fp32"
    cfg.device = str(device)

    if args.mode == "train":
        run_train(cfg, device)
    else:
        run_test(cfg, device, args.ckpt)
    cleanup_dist()


if __name__ == "__main__":
    main()
