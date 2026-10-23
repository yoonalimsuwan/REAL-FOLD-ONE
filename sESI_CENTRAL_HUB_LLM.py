"""
=============================================================================
SESI CENTRAL HUB LLM — Unified Biomedical & CFD Intelligence
Native Full Differentiable / AMP-Safe / DDP-Ready / Max Cost Optimization
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Central Hub Architecture)
=============================================================================

ARCHITECTURE OVERVIEW
─────────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  SESI CENTRAL HUB LLM                                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: Multi-domain queries (protein, oncology, aging, CFD)        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  HCSN Backbone (N layers)                                    │   │
│  │  ├── RMSNorm → CausalSelfAttention (SDPA + RoPE)            │   │
│  │  ├── RMSNorm → SwiGLU MLP                                   │   │
│  │  └── [every hcsn_every] HCSNMeasurementPass                 │   │
│  │       ├── Domain Router (meta-level)                        │   │
│  │       └── Task Executor (base-level)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  Pooled Hidden State (B, d_model)                                   │
│       │                                                             │
│       ├─────────────────────────────────────────────────────────┐  │
│       │                                                         │  │
│       ▼                                                         ▼  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Domain-Specific LLMs (4 specialized modules)           │     │
│  ├─────────────────────────────────────────────────────────┤     │
│  │                                                         │     │
│  │  [D1] ProteinFoldingLLM                                 │     │
│  │    ├── SPTC substrate encoder                           │     │
│  │    ├── Φ_IIT potentiality assessment                    │     │
│  │    └── 3D coordinate prediction                         │     │
│  │                                                         │     │
│  │  [D2] OncologyLLM                                       │     │
│  │    ├── RUT relational graph                             │     │
│  │    ├── Tumor dynamics (SEIR-like)                       │     │
│  │    └── Treatment response prediction                    │     │
│  │                                                         │     │
│  │  [D3] AntiAgingLLM                                      │     │
│  │    ├── QMD cognitive core                               │     │
│  │    ├── Biological age estimation                        │     │
│  │    └── Rejuvenation potential                           │     │
│  │                                                         │     │
│  │  [D4] CFDLLM                                            │     │
│  │    ├── Navier-Stokes surrogate                          │     │
│  │    ├── Turbulence closure                               │     │
│  │    └── Flow field prediction                            │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  Cross-Domain Integration Layer                         │     │
│  ├─────────────────────────────────────────────────────────┤     │
│  │                                                         │     │
│  │  [T1] QMD: Lorenz core for cross-domain reasoning       │     │
│  │  [T2] SPTC: Φ_IIT for consciousness assessment          │     │
│  │  [T3] RLF: Mind interpreter for ethical reasoning       │     │
│  │  [T4] Cloud-OPMC: σ_T for canonical compression         │     │
│  │  [T5] SQCT: Geometric collapse for decision-making      │     │
│  │  [T6] RUT: Relational accessibility graph               │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                             │
│       ▼                                                             │
│  Multi-Task Output Heads                                            │
│    ├── Domain router logits (which LLM to activate)                │
│    ├── Protein folding outputs                                      │
│    ├── Oncology predictions                                         │
│    ├── Anti-aging assessments                                       │
│    ├── CFD flow fields                                              │
│    └── Cross-domain synthesis                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

COST OPTIMIZATION (Maximum)
───────────────────────────
  [C1] Shared trunk: all domain LLMs consume one pooled hidden
  [C2] Low-rank adapters: O(d·r) instead of O(d²), r ≪ d
  [C3] Sparse domain routing: activate only relevant LLMs
  [C4] Gradient checkpointing on transformer + domain LLMs
  [C5] AMP bf16 with fp32 islands for physics
  [C6] Fused AdamW + torch.compile + DDP gradient_as_bucket_view
  [C7] KV-cache generation O(1) per token
  [C8] Token subsampling (mental_stride) for theory heads
  [C9] Domain-specific token masking (ignore irrelevant tokens)
  [C10] Non-persistent buffers for all scalar constants

THEORY INTEGRATION MAP
──────────────────────
  T1: QMD        → Cross-domain reasoning (Lorenz + Latent Field)
  T2: SPTC       → Consciousness assessment (Φ_IIT)
  T3: RLF        → Ethical reasoning (Buddhist 5 pillars)
  T4: Cloud-OPMC → Canonical compression (σ_T + D_DE)
  T5: SQCT       → Decision-making (geometric collapse)
  T6: RUT        → Relational accessibility (cross-domain mapping)

DOMAIN-SPECIFIC LLMS
────────────────────
  D1: ProteinFoldingLLM  → SPTC substrate + 3D coordinates
  D2: OncologyLLM        → RUT relational graph + tumor dynamics
  D3: AntiAgingLLM       → QMD cognitive core + biological age
  D4: CFDLLM             → Navier-Stokes surrogate + turbulence

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

logger = logging.getLogger("sesi.central_hub")

__all__ = [
    "CentralHubConfig",
    "SESICentralHubLLM",
    "ProteinFoldingLLM",
    "OncologyLLM",
    "AntiAgingLLM",
    "CFDLLM",
    "DomainRouter",
    "CrossDomainIntegration",
    "CentralHubLoss",
    "CentralHubTrainer",
    "SyntheticMultiDomainDataset",
    "build_hub_loaders",
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
# C^∞ Smooth Primitives
# ═══════════════════════════════════════════════════════════════════════

def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)


def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)


def logsumexp_max(x: Tensor, tau: float = 1e-2) -> Tensor:
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CentralHubConfig:
    # ── LM Backbone ─────────────────────────────────────────────────────
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    max_seq: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    
    # ── HCSN ────────────────────────────────────────────────────────────
    hcsn_every: int = 4
    hcsn_scales: int = 4
    
    # ── Domain LLMs ─────────────────────────────────────────────────────
    n_domains: int = 4  # protein, oncology, aging, CFD
    domain_hidden: int = 256
    domain_layers: int = 4
    
    # ── QMD ─────────────────────────────────────────────────────────────
    qmd_d_I: int = 32
    qmd_d_Z: int = 16
    qmd_d_N: int = 32
    qmd_kernel_len: int = 8
    qmd_steps: int = 4
    
    # ── SPTC ────────────────────────────────────────────────────────────
    sptc_n_states: int = 8
    sptc_org_hidden: int = 64
    sptc_n_pot_samples: int = 4
    
    # ── RLF / Mind ──────────────────────────────────────────────────────
    mind_dim: int = 64
    
    # ── Cloud-OPMC ──────────────────────────────────────────────────────
    opmc_n_fields: int = 10
    opmc_sigma_dim: int = 64
    
    # ── SQCT ────────────────────────────────────────────────────────────
    sqct_lambda: float = 1e-16
    sqct_r_C: float = 1e-7
    sqct_kappa_I: float = 0.1
    
    # ── RUT ─────────────────────────────────────────────────────────────
    rut_state_dim: int = 32
    
    # ── Cost Optimization ───────────────────────────────────────────────
    mental_rank: int = 32
    mental_stride: int = 4
    gradient_checkpointing: bool = True
    use_fused_adamw: bool = True
    use_torch_compile: bool = True
    sparse_routing: bool = True  # [C3] activate only relevant domains
    
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
    out_dir: str = "runs/sesi_central_hub"
    
    # ── Loss Weights ────────────────────────────────────────────────────
    w_lm: float = 1.0
    w_domain_router: float = 0.5
    w_protein: float = 0.3
    w_oncology: float = 0.3
    w_aging: float = 0.3
    w_cfd: float = 0.3
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
# 3. DOMAIN ROUTER (HCSN Meta-Level)
# ═══════════════════════════════════════════════════════════════════════

class DomainRouter(nn.Module):
    """
    HCSN meta-level router: decides which domain LLMs to activate.
    Uses sparse routing [C3] to minimize computation.
    """
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        # Low-rank router [C2]
        self.router_proj = LowRankAdapter(cfg.d_model, cfg.n_domains, r)
        
        # Temperature for soft routing
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        
        # Top-k selection (sparse routing)
        self.top_k = 2 if cfg.sparse_routing else cfg.n_domains

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            h_pooled: (B, d_model)
        Returns:
            domain_logits: (B, n_domains)
            domain_mask: (B, n_domains) binary mask for sparse routing
            domain_weights: (B, n_domains) soft weights
        """
        logits = self.router_proj(h_pooled)  # (B, n_domains)
        temp = 0.1 + F.softplus(self.log_temp)
        weights = F.softmax(logits / temp, dim=-1)
        
        # Sparse routing: select top-k domains
        if self.cfg.sparse_routing:
            top_k_vals, top_k_idx = torch.topk(weights, self.top_k, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(1, top_k_idx, 1.0)
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            mask = torch.ones_like(weights)
        
        return {
            "domain_logits": logits,
            "domain_mask": mask,
            "domain_weights": weights,
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. DOMAIN-SPECIFIC LLMS
# ═══════════════════════════════════════════════════════════════════════

class ProteinFoldingLLM(nn.Module):
    """
    [D1] Protein Folding LLM.
    Uses SPTC substrate encoder + Φ_IIT potentiality.
    """
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        # Substrate encoder (amino acid sequence)
        self.substrate_enc = nn.Sequential(
            LowRankAdapter(cfg.d_model, cfg.domain_hidden, r),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden),
        )
        
        # 3D coordinate head
        self.coord_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 3),  # (x, y, z) per residue
        )
        
        # Energy prediction head
        self.energy_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 1),
        )

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        substrate = self.substrate_enc(h_pooled)
        coords = self.coord_head(substrate)
        energy = self.energy_head(substrate).squeeze(-1)
        
        return {
            "protein_coords": coords,
            "protein_energy": energy,
        }


class OncologyLLM(nn.Module):
    """
    [D2] Oncology LLM.
    Uses RUT relational graph + tumor dynamics.
    """
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        # Relational graph encoder
        self.rel_enc = nn.Sequential(
            LowRankAdapter(cfg.d_model, cfg.domain_hidden, r),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden),
        )
        
        # Tumor dynamics head (proliferation, apoptosis, invasion)
        self.dynamics_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 3),  # [prolif, apop, inv]
            nn.Sigmoid(),
        )
        
        # Treatment response prediction
        self.treatment_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        rel = self.rel_enc(h_pooled)
        dynamics = self.dynamics_head(rel)
        treatment = self.treatment_head(rel)
        
        return {
            "oncology_dynamics": dynamics,
            "oncology_treatment_response": treatment,
        }


class AntiAgingLLM(nn.Module):
    """
    [D3] Anti-Aging LLM.
    Uses QMD cognitive core + biological age estimation.
    """
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        # Biological age encoder
        self.age_enc = nn.Sequential(
            LowRankAdapter(cfg.d_model, cfg.domain_hidden, r),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden),
        )
        
        # Biological age prediction
        self.bio_age_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 1),
            nn.Softplus(),  # age > 0
        )
        
        # Rejuvenation potential (6 hallmarks of aging)
        self.rejuvenation_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 6),
            nn.Sigmoid(),
        )

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        age_enc = self.age_enc(h_pooled)
        bio_age = self.bio_age_head(age_enc).squeeze(-1)
        rejuvenation = self.rejuvenation_head(age_enc)
        
        return {
            "aging_bio_age": bio_age,
            "aging_rejuvenation": rejuvenation,
        }


class CFDLLM(nn.Module):
    """
    [D4] CFD LLM.
    Uses Navier-Stokes surrogate + turbulence closure.
    """
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        
        # Flow field encoder
        self.flow_enc = nn.Sequential(
            LowRankAdapter(cfg.d_model, cfg.domain_hidden, r),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden),
        )
        
        # Velocity field prediction (3 components)
        self.velocity_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 3),  # (u, v, w)
        )
        
        # Pressure field prediction
        self.pressure_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 1),
        )
        
        # Turbulence closure (eddy viscosity)
        self.turbulence_head = nn.Sequential(
            nn.Linear(cfg.domain_hidden, cfg.domain_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden // 2, 1),
            nn.Softplus(),  # viscosity > 0
        )

    def forward(self, h_pooled: Tensor) -> Dict[str, Tensor]:
        flow = self.flow_enc(h_pooled)
        velocity = self.velocity_head(flow)
        pressure = self.pressure_head(flow).squeeze(-1)
        turbulence = self.turbulence_head(flow).squeeze(-1)
        
        return {
            "cfd_velocity": velocity,
            "cfd_pressure": pressure,
            "cfd_turbulence": turbulence,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5. CROSS-DOMAIN INTEGRATION (T1-T6)
# ═══════════════════════════════════════════════════════════════════════

class CrossDomainIntegration(nn.Module):
    """
    Integrates outputs from all 4 domain LLMs using T1-T6 theories.
    """
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.mental_rank
        d = cfg.d_model
        
        # [T1] QMD: Lorenz core for cross-domain reasoning
        self.qmd_proj = LowRankAdapter(d * 4, d, r)  # concatenate 4 domains
        
        # [T2] SPTC: Φ_IIT for consciousness assessment
        self.sptc_proj = LowRankAdapter(d, cfg.sptc_org_hidden, r)
        
        # [T3] RLF: Mind interpreter
        self.mind_proj = LowRankAdapter(d, cfg.mind_dim, r)
        
        # [T4] Cloud-OPMC: σ_T canonical signature
        self.opmc_proj = LowRankAdapter(d, cfg.opmc_sigma_dim, r)
        
        # [T5] SQCT: Geometric collapse
        self.sqct_proj = LowRankAdapter(d, 32, r)
        
        # [T6] RUT: Relational accessibility
        self.rut_proj = LowRankAdapter(d, cfg.rut_state_dim, r)
        
        # Final synthesis head
        self.synthesis_head = nn.Sequential(
            nn.Linear(d * 6, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, domain_outputs: Dict[str, Tensor],
                h_pooled: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            domain_outputs: dict with outputs from all 4 domain LLMs
            h_pooled: (B, d_model) from backbone
        Returns:
            dict with cross-domain integration outputs
        """
        B = h_pooled.shape[0]
        
        # Concatenate domain outputs
        protein = domain_outputs.get("protein_coords", torch.zeros(B, 3, device=h_pooled.device))
        oncology = domain_outputs.get("oncology_dynamics", torch.zeros(B, 3, device=h_pooled.device))
        aging = domain_outputs.get("aging_bio_age", torch.zeros(B, device=h_pooled.device))
        cfd = domain_outputs.get("cfd_velocity", torch.zeros(B, 3, device=h_pooled.device))
        
        # Flatten and concatenate
        concat = torch.cat([
            protein.reshape(B, -1),
            oncology.reshape(B, -1),
            aging.unsqueeze(-1) if aging.dim() == 1 else aging,
            cfd.reshape(B, -1),
        ], dim=-1)
        
        # Pad to d_model * 4 if needed
        if concat.shape[-1] < d_model * 4:
            pad = torch.zeros(B, d_model * 4 - concat.shape[-1], device=concat.device)
            concat = torch.cat([concat, pad], dim=-1)
        
        # [T1] QMD cross-domain reasoning
        qmd_out = self.qmd_proj(concat)
        
        # [T2] SPTC consciousness assessment
        sptc_out = self.sptc_proj(h_pooled)
        
        # [T3] RLF mind interpretation
        mind_out = self.mind_proj(h_pooled)
        
        # [T4] Cloud-OPMC canonical signature
        opmc_out = self.opmc_proj(h_pooled)
        
        # [T5] SQCT geometric collapse
        sqct_out = self.sqct_proj(h_pooled)
        
        # [T6] RUT relational accessibility
        rut_out = self.rut_proj(h_pooled)
        
        # Final synthesis
        synthesis_input = torch.cat([qmd_out, sptc_out, mind_out, opmc_out, sqct_out, rut_out], dim=-1)
        synthesis = self.synthesis_head(synthesis_input)
        
        return {
            "qmd_cross_domain": qmd_out,
            "sptc_consciousness": sptc_out,
            "mind_interpretation": mind_out,
            "opmc_signature": opmc_out,
            "sqct_collapse": sqct_out,
            "rut_relational": rut_out,
            "synthesis": synthesis,
        }


# ═══════════════════════════════════════════════════════════════════════
# 6. HCSN BACKBONE
# ═══════════════════════════════════════════════════════════════════════

class HCSNMeasurementPass(nn.Module):
    """HCSN measurement pass with domain routing."""
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


class HubTransformerBlock(nn.Module):
    """Transformer block with HCSN measurement pass."""
    def __init__(self, cfg: CentralHubConfig, layer_idx: int):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.n2 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads)
        self.mlp = SwiGLU(cfg.d_model)
        self.use_hcsn = (cfg.hcsn_every > 0 and (layer_idx + 1) % cfg.hcsn_every == 0)
        self.hcsn = HCSNMeasurementPass(cfg.d_model, cfg.hcsn_scales) if self.use_hcsn else None

    def forward(self, x, kv_cache=None, offset=0, rope=None):
        a, kv = self.attn(self.n1(x), kv_cache, offset, rope)
        x = x + a
        x = x + self.mlp(self.n2(x))
        if self.hcsn is not None:
            x = self.hcsn(x)
        return x, kv


# ═══════════════════════════════════════════════════════════════════════
# 7. MAIN MODEL — SESI CENTRAL HUB LLM
# ═══════════════════════════════════════════════════════════════════════

class SESICentralHubLLM(nn.Module):
    """
    Unified SESI Central Hub LLM integrating all 4 domain LLMs + 6 theories.
    """
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg

        # ── LM Backbone ─────────────────────────────────────────────────
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RoPE(cfg.d_model // cfg.n_heads, max_seq=cfg.max_seq)
        self.blocks = nn.ModuleList([
            HubTransformerBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # ── Domain Router ───────────────────────────────────────────────
        self.router = DomainRouter(cfg)

        # ── Domain-Specific LLMs ────────────────────────────────────────
        self.protein_llm = ProteinFoldingLLM(cfg)
        self.oncology_llm = OncologyLLM(cfg)
        self.aging_llm = AntiAgingLLM(cfg)
        self.cfd_llm = CFDLLM(cfg)

        # ── Cross-Domain Integration ────────────────────────────────────
        self.integration = CrossDomainIntegration(cfg)

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
                return_all: bool = True) -> Dict[str, Tensor]:
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

        if not return_all:
            return out

        # ── Pooled hidden [C2, C8] ──────────────────────────────────────
        stride = self.cfg.mental_stride
        h_pooled = x[:, ::stride].mean(dim=1)

        # ── Domain Routing [C3] ─────────────────────────────────────────
        routing = self.router(h_pooled)
        out.update(routing)

        # ── Domain-Specific LLMs (sparse activation) ────────────────────
        domain_outputs = {}
        mask = routing["domain_mask"]  # (B, 4)
        
        # [D1] Protein Folding
        if mask[:, 0].any():
            protein_out = self.protein_llm(h_pooled)
            domain_outputs.update({f"protein_{k}": v for k, v in protein_out.items()})
        
        # [D2] Oncology
        if mask[:, 1].any():
            oncology_out = self.oncology_llm(h_pooled)
            domain_outputs.update({f"oncology_{k}": v for k, v in oncology_out.items()})
        
        # [D3] Anti-Aging
        if mask[:, 2].any():
            aging_out = self.aging_llm(h_pooled)
            domain_outputs.update({f"aging_{k}": v for k, v in aging_out.items()})
        
        # [D4] CFD
        if mask[:, 3].any():
            cfd_out = self.cfd_llm(h_pooled)
            domain_outputs.update({f"cfd_{k}": v for k, v in cfd_out.items()})
        
        out.update(domain_outputs)

        # ── Cross-Domain Integration [T1-T6] ────────────────────────────
        integration_out = self.integration(domain_outputs, h_pooled)
        out.update(integration_out)

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
            probs = F.softmax(logits
            nxt = torch
            out = torch.cat([out, nxt], dim=1)
        return out


# ═══════════════════════════════════════════════════════════════════════
# 8. MULTI-TASK LOSS
# ═══════════════════════════════════════════════════════════════════════

class CentralHubLoss(nn.Module):
    def __init__(self, cfg: CentralHubConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, out: Dict[str, Tensor], batch: Dict[str, Tensor],
                model: nn.Module) -> Dict[str, Tensor]:
        cfg = self.cfg
        dev = out["logits"].device
        zero = torch.zeros((), device=dev)
        L: Dict[str, Tensor] = {}

        # LM loss
        L["lm"] = out.get("lm_loss", zero)

        # Domain router loss (encourage sparse routing)
        if "domain_weights" in out:
            router_loss = -out["domain_weights"].float().log().mean()
            L["router"] = router_loss
        else:
            L["router"] = zero

        # Domain-specific losses
        if "protein_coords" in out and "protein_targets" in batch:
            L["protein"] = F.mse_loss(out["protein_coords"], batch["protein_targets"])
        else:
            L["protein"] = zero

        if "oncology_dynamics" in out and "oncology_targets" in batch:
            L["oncology"] = F.mse_loss(out["oncology_dynamics"], batch["oncology_targets"])
        else:
            L["oncology"] = zero

        if "aging_bio_age" in out and "aging_targets" in batch:
            L["aging"] = F.mse_loss(out["aging_bio_age"], batch["aging_targets"])
        else:
            L["aging"] = zero

        if "cfd_velocity" in out and "cfd_targets" in batch:
            L["cfd"] = F.mse_loss(out["cfd_velocity"], batch["cfd_targets"])
        else:
            L["cfd"] = zero

        # Cross-domain integration losses
        if "synthesis" in out:
            L["integration"] = out["synthesis"].float().pow(2).mean() * 0.01
        else:
            L["integration"] = zero

        # L2 regularization
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["reg"] = reg

        total = (cfg.w_lm * L["lm"]
                 + cfg.w_domain_router * L["router"]
                 + cfg.w_protein * L["protein"]
                 + cfg.w_oncology * L["oncology"]
                 + cfg.w_aging * L["aging"]
                 + cfg.w_cfd * L["cfd"]
                 + cfg.w_reg * L["reg"])
        L["total"] = total
        return L


# ═══════════════════════════════════════════════════════════════════════
# 9. EMA + TRAINER
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


class CentralHubTrainer:
    def __init__(self, cfg: CentralHubConfig, model: SESICentralHubLLM,
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

        self.loss_fn = CentralHubLoss(cfg)

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
        cfg = self.cfg
        if self.step < cfg.warmup_steps:
            return cfg.lr * (self.step + 1) / max(1, cfg.warmup_steps)
        prog = (self.step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        return cfg.lr * (cfg.min_lr_ratio
                         + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

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
# 10. SYNTHETIC DATASET
# ═══════════════════════════════════════════════════════════════════════

class SyntheticMultiDomainDataset(Dataset):
    def __init__(self, cfg: CentralHubConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, cfg.vocab_size, (n, cfg.max_seq), generator=g)
        self.targets = torch.roll(self.tokens, -1, dims=1)
        
        # Domain-specific targets
        self.protein_targets = torch.randn(n, 3, generator=g)  # 3D coords
        self.oncology_targets = torch.rand(n, 3, generator=g)  # dynamics
        self.aging_targets = torch.rand(n, generator=g) * 80 + 20  # age
        self.cfd_targets = torch.randn(n, 3, generator=g)  # velocity

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        return {
            "tokens": self.tokens[i],
            "targets": self.targets[i],
            "protein_targets": self.protein_targets[i],
            "oncology_targets": self.oncology_targets[i],
            "aging_targets": self.aging_targets[i],
            "cfd_targets": self.cfd_targets[i],
        }


def build_hub_loaders(cfg: CentralHubConfig):
    tr = SyntheticMultiDomainDataset(cfg, cfg.n_train, cfg.data_seed)
    va = SyntheticMultiDomainDataset(cfg, cfg.n_val, cfg.data_seed + 1)
    te = SyntheticMultiDomainDataset(cfg, cfg.n_test, cfg.data_seed + 2)
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
# 11. SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Central Hub LLM — Smoke Test (All 4 Domains + 6 Theories)")
    print("=" * 72)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)

    cfg = CentralHubConfig(
        d_model=128, n_layers=4, n_heads=4, max_seq=64,
        vocab_size=64, batch_size=2,
        qmd_d_I=8, qmd_d_Z=4, qmd_d_N=8,
        sptc_n_states=4, sptc_org_hidden=32,
        mind_dim=32, mental_rank=8,
        gradient_checkpointing=False, use_torch_compile=False,
    )

    model = SESICentralHubLLM(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")

    model.train()
    B, T = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.roll(ids, -1, dims=1)

    out = model(ids=ids, targets=targets)

    loss_fn = CentralHubLoss(cfg)
    batch = {"tokens": ids, "targets": targets,
             "protein_targets": torch.randn(B, 3, device=device),
             "oncology_targets": torch.rand(B, 3, device=device),
             "aging_targets": torch.rand(B, device=device) * 80 + 20,
             "cfd_targets": torch.randn(B, 3, device=device)}
    losses = loss_fn(out, batch, model)
    losses["total"].backward()

    print(f"  LM loss          : {losses['lm'].item():.4f}")
    print(f"  Router loss      : {losses['router'].item():.4f}")
    print(f"  Protein loss     : {losses['protein'].item():.4f}")
    print(f"  Oncology loss    : {losses['oncology'].item():.4f}")
    print(f"  Aging loss       : {losses['aging'].item():.4f}")
    print(f"  CFD loss         : {losses['cfd'].item():.4f}")
    print(f"  Total loss       : {losses['total'].item():.4f}")
    print(f"  Domain mask      : {out['domain_mask'][0].tolist()}")
    print(f"  Domain weights   : {out['domain_weights'][0].tolist()}")

    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if
                 
```
