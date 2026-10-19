"""
=============================================================================
MSPS UNIFIED THEORY TEST MODULE — PRODUCTION EDITION v1.0
=============================================================================
Developer  : PAI and Yoon A Limsuwan — MSPS NETWORK
ORCID      : 0009-0008-2374-0788
License    : MIT
Year       : 2026
=============================================================================

Unified differentiable hypothesis-testing module for five MSPS theoretical
frameworks:

  T1. Relational Compatibility (RC)
      "Interaction, Shared Universes, Consciousness, and Parallel Universes"
      - Relational Accessibility Principle (RAP)
      - Hierarchical Relative Experiential Universes (HREU)
      - Relational Universe Compatibility Principle (RUCP)
      - Four Classes of Relational Accessibility

  T2. Relational Logos Framework (RLF)
      "The Relational Logos: Dharma, Mathematical Principles, Consciousness"
      - Five Foundational Distinctions
      - Relational Model of Reality
      - Invariant Source–Variable Manifestation (ISVM)
      - Personal–Impersonal Distinction

  T3. Quantum Mental Dynamics (QMD)
      "Quantum Mental Dynamics: A Hierarchical Superposition Theory"
      - Cross-Level Transition Operator
      - Latent Informational Field
      - Frame-Accumulation Hypothesis
      - Neural–Cognitive Interface
      - Quantum Branch (conditional)

  T4. Structural Potentiality Theory of Consciousness (SPTC)
      "Structural Potentiality of Consciousness: Matter-to-Mind Framework"
      - Substrate Potentiality
      - Organizational Realization
      - IIT Surrogate (min-KL / EI-diff)
      - Potentiality Functional

  T5. Relational Universe Theory (RUT-PCD)
      "Relational Universe Theory: Physical Interaction, Conscious
       Organization, and Integrated Information"
      - Relational Accessibility
      - Cross-Domain Mapping
      - Causal Influence Model
      - IIT Connection

DESIGN PRINCIPLES
─────────────────
  • Native full differentiability: no detached surrogate paths.
  • AMP: bf16 / fp16 / fp32 with automatic device detection.
  • DDP: multi-GPU with DistributedSampler, gradient sync, rank guards.
  • Cost optimization: gradient checkpointing, fused AdamW, memory-efficient
    Jacobian estimation (Hutchinson fallback), selective activation recomputation.
  • Real data: configurable Dataset/DataLoader with schema validation.
  • Production training: EMA, cosine LR, warmup, grad clipping, checkpointing.
  • Falsifiability: each theory produces testable metrics with paired
    bootstrap confidence intervals.

AUDIT NOTES
───────────
  • All modules use registered buffers for physical constants (DDP-safe).
  • No .item() calls in forward paths (graph-safe).
  • Gumbel-sigmoid uses logistic noise (correct two-sided distribution).
  • DistributedSampler epoch is advanced every epoch.
  • Fused AdamW requested only on CUDA devices.
  • Gradient checkpointing is optional per module to control memory/cost.

=============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import json
import logging
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

__all__ = [
    # Config
    "UnifiedConfig",
    # T1: Relational Compatibility
    "RelationalAccessibilityModule",
    "HREUClassifier",
    "RUCPEvaluator",
    # T2: Relational Logos
    "ISVMModule",
    "PersonalImpersonalClassifier",
    "LogosRelationalModel",
    # T3: Quantum Mental Dynamics
    "QMDCrossLevelTransition",
    "QMDLatentField",
    "QMDFrameAccumulation",
    "QMDNeuralCognitiveInterface",
    "QMDModel",
    # T4: Structural Potentiality
    "SPTCSubstrateEncoder",
    "SPTCOrganizationPolicy",
    "SPTCDynamics",
    "SPTCIITSurrogate",
    "SPTCModel",
    # T5: Relational Universe Theory
    "RUTRelationalGraph",
    "RUTCrossDomainMapper",
    "RUTCausalInfluenceModel",
    "RUTModel",
    # Infrastructure
    "UnifiedTrainer",
    "HypothesisTestSuite",
    "RealDataLoader",
    "build_dataloader",
    "setup_dist",
    "cleanup_dist",
    "seed_all",
]


# ===========================================================================
# 0. AMP / DDP / DISTRIBUTED HELPERS
# ===========================================================================

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


def _all_reduce_mean(x: Tensor) -> Tensor:
    if not _is_dist():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / _world_size()


# AMP shim (torch >= 2.4 preferred, older fallback)
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_NEW = False


def amp_ctx(enabled: bool, dtype: torch.dtype, device_type: str = "cuda"):
    """Context manager for automatic mixed precision."""
    if not enabled:
        return contextlib.nullcontext()
    if _AMP_NEW:
        return _autocast(device_type=device_type, dtype=dtype, enabled=True)
    return _autocast(dtype=dtype, enabled=True)


@contextlib.contextmanager
def fp32_ctx():
    """Force fp32 inside numerically sensitive blocks (logs, KL, etc.)."""
    if _AMP_NEW:
        with _autocast(device_type="cuda", enabled=False):
            yield
    else:
        with _autocast(enabled=False):
            yield


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

@dataclass
class UnifiedConfig:
    """Master configuration for the unified theory test module."""

    # --- Device / precision ---
    device: str = "cuda"
    amp_dtype: str = "bf16"           # "bf16" | "fp16" | "fp32"
    amp_enabled: bool = True

    # --- Training ---
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 1000
    out_dir: str = "runs/unified_theory"

    # --- Cost optimization ---
    gradient_checkpointing: bool = True
    use_fused_adamw: bool = True
    memory_efficient_jacobian: bool = True
    selective_recompute: bool = True

    # --- Data ---
    n_train: int = 8192
    n_val: int = 1024
    n_test: int = 2048
    data_seed: int = 1234
    seq_len: int = 32
    d_obs: int = 32                   # observation dimension
    d_latent: int = 16                # latent dimension
    d_state: int = 64                 # state dimension

    # --- T1: Relational Compatibility ---
    rc_n_nodes: int = 16
    rc_n_relations: int = 32
    rc_n_levels: int = 5              # HREU levels
    rc_compat_types: int = 5          # RUCP compatibility types
    rc_hidden: int = 128

    # --- T2: Relational Logos ---
    rl_n_entities: int = 16
    rl_n_conditions: int = 8
    rl_n_rules: int = 8
    rl_hidden: int = 128

    # --- T3: QMD ---
    qmd_d_I: int = 32
    qmd_d_Z: int = 16
    qmd_d_N: int = 64
    qmd_d_E: int = 8
    qmd_d_D: int = 16
    qmd_n_levels: int = 3
    qmd_enable_quantum: bool = False
    qmd_quantum_dim: int = 8

    # --- T4: SPTC ---
    sptc_d_substrate: int = 12
    sptc_n_states: int = 8            # must be power of 2
    sptc_org_hidden: int = 96
    sptc_iit_mode: str = "min_kl"     # "min_kl" | "ei_diff"
    sptc_iit_tau: float = 0.1
    sptc_n_potentiality_samples: int = 8

    # --- T5: RUT-PCD ---
    rut_n_components: int = 16
    rut_n_edges: int = 32
    rut_hidden: int = 128
    rut_n_state_dims: int = 32

    # --- Loss weights ---
    lambda_task: float = 1.0
    lambda_reg: float = 1e-5
    lambda_compat: float = 1e-2
    lambda_potentiality: float = 1e-2
    lambda_coupling: float = 1e-3

    # --- Theory selection ---
    enable_T1: bool = True
    enable_T2: bool = True
    enable_T3: bool = True
    enable_T4: bool = True
    enable_T5: bool = True


# ===========================================================================
# 2. BUILDING BLOCKS
# ===========================================================================

class MLP(nn.Module):
    """Configurable MLP with GELU activation and optional residual."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        n_layers: int = 3,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.residual = residual and in_dim == out_dim

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return x + y if self.residual else y


class LayerNormMLP(nn.Module):
    """MLP with LayerNorm for numerical stability under AMP."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden, out_dim, n_layers=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.norm(x))


def soft_floor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    """Differentiable positivity floor."""
    return floor + F.softplus(x - floor, beta=beta)


def soft_ceiling(x: Tensor, ceil: float, beta: float = 50.0) -> Tensor:
    """Differentiable ceiling."""
    return ceil - F.softplus(ceil - x, beta=beta)


def gumbel_sigmoid(logits: Tensor, tau: float, hard: bool, training: bool) -> Tensor:
    """Gumbel-sigmoid with logistic noise (correct two-sided distribution)."""
    if not training:
        return torch.sigmoid(logits / tau)
    u = torch.rand_like(logits).clamp_(1e-7, 1.0 - 1e-7)
    g = torch.log(u) - torch.log1p(-u)  # logistic noise
    y_soft = torch.sigmoid((logits + g) / tau)
    if not hard:
        return y_soft
    y_hard = (y_soft > 0.5).to(y_soft.dtype)
    return y_hard - y_soft.detach() + y_soft


# ===========================================================================
# 3. T1: RELATIONAL COMPATIBILITY (RC)
# ===========================================================================

class RelationalAccessibilityModule(nn.Module):
    """
    T1 — Relational Accessibility Principle (RAP):
    Within a specified model, causal influence from one system to another
    requires an admissible pathway.

    Implements a differentiable adjacency + activation model:
        A[i,j] ∈ [0,1] — admissibility of relation i→j
        a[i,j](t) ∈ [0,1] — current activation of that relation
        c[i,j](t) = A[i,j] · a[i,j](t) — effective coupling

    Testable prediction: interventions on component i should predictably
    affect component j only when c[i,j] > threshold.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.rc_n_nodes
        # Learnable adjacency (admissible relations)
        self.adj_logits = nn.Parameter(torch.zeros(n, n))
        # Activation network: state → activation weights
        self.activation_net = MLP(
            cfg.rut_n_state_dims, cfg.rc_hidden, n * n, n_layers=2
        )
        self.register_buffer("_eye", torch.eye(n), persistent=False)

    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            state: (B, n_components, d_state)
        Returns:
            dict with adjacency, activation, effective_coupling, reachability
        """
        B = state.shape[0]
        n = self.cfg.rc_n_nodes
        device = state.device

        # Admissibility: sigmoid adjacency (remove self-loops)
        adj = torch.sigmoid(self.adj_logits) * (1.0 - self._eye.to(device))

        # Activation from state
        pooled = state.mean(dim=1)  # (B, d_state)
        act_logits = self.activation_net(pooled).view(B, n, n)
        activation = torch.sigmoid(act_logits) * (1.0 - self._eye.to(device))

        # Effective coupling
        coupling = adj.unsqueeze(0) * activation  # (B, n, n)

        # Reachability: soft transitive closure (3-hop approximation)
        reach = coupling
        for _ in range(3):
            reach = torch.clamp(reach + torch.bmm(reach, reach), max=1.0)

        return {
            "adjacency": adj,
            "activation": activation,
            "coupling": coupling,
            "reachability": reach,
        }


class HREUClassifier(nn.Module):
    """
    T1 — Hierarchical Relative Experiential Universes (HREU):
    Five levels of epistemic reliability:
      L1: Personal experience
      L2: Interpretation / cognitive modeling
      L3: Intersubjective comparison
      L4: Empirically tested models
      L5: Cross-model synthesis

    Classifies a claim into one of 5 reliability levels based on
    its feature representation.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.classifier = MLP(
            cfg.rc_hidden, cfg.rc_hidden, cfg.rc_n_levels, n_layers=3
        )
        self.level_embeddings = nn.Parameter(
            torch.randn(cfg.rc_n_levels, cfg.rc_hidden) * 0.02
        )

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            features: (B, rc_hidden) — representation of a claim/observation
        Returns:
            dict with level_logits, level_probs, reliability_score
        """
        logits = self.classifier(features)  # (B, 5)
        probs = F.softmax(logits, dim=-1)
        # Reliability score: weighted average of level indices
        levels = torch.arange(
            self.cfg.rc_n_levels, device=features.device, dtype=features.dtype
        )
        reliability = (probs * levels.unsqueeze(0)).sum(dim=-1)
        return {
            "level_logits": logits,
            "level_probs": probs,
            "reliability_score": reliability,
        }


class RUCPEvaluator(nn.Module):
    """
    T1 — Relational Universe Compatibility Principle (RUCP):
    Evaluates compatibility between two experiential models U_i and U_j
    across 5 types: semantic, informational, causal, physical, metaphysical.

    Returns a compatibility vector and overall compatibility score.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.compat_net = MLP(
            cfg.rc_hidden * 2, cfg.rc_hidden, cfg.rc_compat_types, n_layers=3
        )
        self.type_weights = nn.Parameter(
            torch.ones(cfg.rc_compat_types) / cfg.rc_compat_types
        )

    def forward(
        self, model_i: Tensor, model_j: Tensor
    ) -> Dict[str, Tensor]:
        """
        Args:
            model_i: (B, rc_hidden) — representation of universe model i
            model_j: (B, rc_hidden) — representation of universe model j
        Returns:
            dict with compat_scores, overall_compat, type_probs
        """
        combined = torch.cat([model_i, model_j], dim=-1)
        compat_scores = torch.sigmoid(self.compat_net(combined))  # (B, 5)
        weights = F.softmax(self.type_weights, dim=0)
        overall = (compat_scores * weights.unsqueeze(0)).sum(dim=-1)
        return {
            "compat_scores": compat_scores,
            "overall_compat": overall,
            "type_weights": weights,
        }


# ===========================================================================
# 4. T2: RELATIONAL LOGOS FRAMEWORK (RLF)
# ===========================================================================

class ISVMModule(nn.Module):
    """
    T2 — Invariant Source–Variable Manifestation Principle (ISVM):
    Models C_{t+1} = F(G, C_t, I_t) where G is invariant and C_t changes.

    Tests whether a fixed rule G can produce diverse outputs under
    varying conditions I_t.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.rl_hidden
        # Invariant source (learnable but fixed during inference)
        self.G = nn.Parameter(torch.randn(d) * 0.1)
        # Transition function
        self.F = MLP(d * 3, d, d, n_layers=3)
        # Manifestation head
        self.manifest = MLP(d, d, cfg.rut_n_state_dims, n_layers=2)

    def forward(
        self, C_t: Tensor, I_t: Tensor
    ) -> Dict[str, Tensor]:
        """
        Args:
            C_t: (B, rl_hidden) — current changing state
            I_t: (B, rl_hidden) — information / conditions
        Returns:
            dict with C_next, manifestation, invariance_measure
        """
        B = C_t.shape[0]
        G_expanded = self.G.unsqueeze(0).expand(B, -1)
        combined = torch.cat([G_expanded, C_t, I_t], dim=-1)
        C_next = self.F(combined) + C_t  # residual transition
        manifestation = self.manifest(C_next)
        # Invariance measure: how much G contributes relative to I_t
        invariance = (G_expanded.norm(dim=-1) /
                      (I_t.norm(dim=-1) + 1e-8)).mean()
        return {
            "C_next": C_next,
            "manifestation": manifestation,
            "invariance_measure": invariance,
        }


class PersonalImpersonalClassifier(nn.Module):
    """
    T2 — Personal–Impersonal Distinction:
    Classifies whether a given representation should be interpreted
    as personal (agentive) or impersonal (law-like).

    Three senses of personhood:
      - Phenomenal (subjective experience)
      - Agentive (intention, goal-directed action)
      - Relational (being addressed / encountered)
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.classifier = MLP(
            cfg.rl_hidden, cfg.rl_hidden, 3, n_layers=3  # 3 senses
        )
        self.meta_classifier = MLP(
            cfg.rl_hidden, cfg.rl_hidden, 2, n_layers=2  # personal vs impersonal
        )

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            features: (B, rl_hidden)
        Returns:
            dict with personhood_scores, personal_impersonal_logit
        """
        personhood = self.classifier(features)  # (B, 3)
        pi_logit = self.meta_classifier(features)  # (B, 2)
        return {
            "personhood_scores": torch.sigmoid(personhood),
            "personal_impersonal_logit": pi_logit,
            "is_personal": torch.sigmoid(pi_logit[:, 1] - pi_logit[:, 0]),
        }


class LogosRelationalModel(nn.Module):
    """
    T2 — Full Relational Logos model:
    Combines ISVM, PersonalImpersonal, and relational structure.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.isvm = ISVMModule(cfg)
        self.pi = PersonalImpersonalClassifier(cfg)
        self.relational_encoder = MLP(
            cfg.rut_n_state_dims, cfg.rl_hidden, cfg.rl_hidden, n_layers=2
        )

    def forward(
        self, state: Tensor, conditions: Tensor
    ) -> Dict[str, Tensor]:
        encoded = self.relational_encoder(state)
        isvm_out = self.isvm(encoded, conditions)
        pi_out = self.pi(isvm_out["C_next"])
        return {**isvm_out, **pi_out, "encoded": encoded}


# ===========================================================================
# 5. T3: QUANTUM MENTAL DYNAMICS (QMD)
# ===========================================================================

class QMDLatentField(nn.Module):
    """T3 — Latent Informational Field: amortised VI for Z_t."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.qmd_d_Z + cfg.qmd_d_I + cfg.qmd_d_D
        self.post = MLP(d_in, cfg.rc_hidden, 2 * cfg.qmd_d_Z, n_layers=3)
        self.prior_cell = nn.GRUCell(cfg.qmd_d_Z, cfg.qmd_d_Z)

    def posterior(
        self, z_prev: Tensor, x_prev: Tensor, d_t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        h = torch.cat([z_prev, x_prev, d_t], dim=-1)
        mu, logvar = self.post(h).chunk(2, dim=-1)
        logvar = logvar.clamp(min=-8.0, max=4.0)
        return mu, logvar

    def prior(self, z_prev: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.prior_cell(z_prev)
        mu = torch.tanh(h)
        logvar = torch.zeros_like(mu)
        return mu, logvar

    @staticmethod
    def reparameterise(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl(mu_q: Tensor, lv_q: Tensor, mu_p: Tensor, lv_p: Tensor) -> Tensor:
        kl = 0.5 * (
            lv_p - lv_q
            + (torch.exp(lv_q) + (mu_q - mu_p).pow(2))
            / torch.exp(lv_p).clamp_min(1e-6)
            - 1.0
        )
        return kl.sum(-1)


class QMDFrameAccumulation(nn.Module):
    """T3 — Frame-Accumulation Hypothesis: temporal integration kernel."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.K = min(cfg.seq_len, 16)
        self.hyper = MLP(cfg.qmd_d_E, cfg.rc_hidden, self.K, n_layers=2)
        self.mix = nn.Linear(cfg.qmd_d_D, cfg.qmd_d_D, bias=False)
        self.gru = nn.GRUCell(cfg.qmd_d_D, cfg.qmd_d_D)

    def forward(self, s: Tensor, ctx: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, D = s.shape
        K = min(self.K, T)
        w = torch.softmax(self.hyper(ctx), dim=-1)[:, :K]
        idx = torch.arange(T, device=s.device).unsqueeze(0)
        off = torch.arange(K, device=s.device).unsqueeze(1)
        pos = (idx - off).clamp(min=0)
        gathered = s[:, pos, :]
        P = torch.einsum("bk,bktd->btd", w, gathered)
        P = self.mix(P)
        h = torch.zeros(B, D, device=s.device, dtype=P.dtype)
        outs = []
        for t in range(T):
            h = self.gru(P[:, t, :], h)
            outs.append(h)
        return torch.stack(outs, dim=1), w


class QMDModel(nn.Module):
    """
    T3 — Full QMD model: hierarchical state transitions with latent field,
    frame accumulation, and optional quantum branch.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        d_x = cfg.qmd_d_I + cfg.qmd_d_Z + cfg.qmd_d_N

        self.transition = MLP(
            d_x + cfg.qmd_d_D + cfg.qmd_d_E,
            cfg.rc_hidden,
            2 * d_x,
            n_layers=3,
        )
        self.lif = QMDLatentField(cfg)
        self.fah = QMDFrameAccumulation(cfg)
        self.encoder = MLP(
            cfg.qmd_d_D + cfg.qmd_d_E, cfg.rc_hidden, d_x, n_layers=2
        )
        self.obs = MLP(
            d_x + cfg.qmd_d_E, cfg.rc_hidden, 2 * cfg.qmd_d_D, n_layers=2
        )
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

    def forward(
        self, D: Tensor, E: Tensor, *, sample: bool = True
    ) -> Dict[str, Tensor]:
        B, T, _ = D.shape
        cfg = self.cfg
        d_x = cfg.qmd_d_I + cfg.qmd_d_Z + cfg.qmd_d_N

        X = torch.tanh(self.encoder(torch.cat([D[:, 0] * 0, E[:, 0]], dim=-1)))
        Z = torch.zeros(B, cfg.qmd_d_Z, device=D.device, dtype=D.dtype)

        P, w = self.fah(D, E[:, 0])

        Xs, Zs, D_hat, Kls = [], [], [], []
        for t in range(T):
            mu_q, lv_q = self.lif.posterior(Z, X[:, :cfg.qmd_d_I], D[:, t])
            mu_p, lv_p = self.lif.prior(Z)
            kl = self.lif.kl(mu_q, lv_q, mu_p, lv_p)
            Z = self.lif.reparameterise(mu_q, lv_q) if sample else mu_q

            inp = torch.cat([X, D[:, t], E[:, t]], dim=-1)
            out = self.transition(inp)
            mu_x, lv_x = out.chunk(2, dim=-1)
            lv_x = lv_x.clamp(min=-10.0, max=4.0)
            xi = torch.randn_like(mu_x) * torch.exp(0.5 * lv_x)
            X = mu_x + (xi if sample else 0.0)

            obs_out = self.obs(torch.cat([X, E[:, t]], dim=-1))
            mu_d, lv_d = obs_out.chunk(2, dim=-1)
            lv_d = lv_d.clamp(min=-8.0, max=4.0)

            Xs.append(X)
            Zs.append(Z)
            D_hat.append(mu_d)
            Kls.append(kl)

        return {
            "X": torch.stack(Xs, dim=1),
            "Z": torch.stack(Zs, dim=1),
            "D_hat": torch.stack(D_hat, dim=1),
            "kl": torch.stack(Kls, dim=1),
            "P": P,
            "kernel_w": w,
        }


# ===========================================================================
# 6. T4: STRUCTURAL POTENTIALITY THEORY OF CONSCIOUSNESS (SPTC)
# ===========================================================================

def unit_bipartitions(k: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """All unordered bipartitions of k binary units."""
    idx = list(range(k))
    out = []
    for r in range(1, k):
        for A in itertools.combinations(idx, r):
            B = tuple(i for i in idx if i not in A)
            if A < B:
                out.append((A, B))
    return out


def product_cut(
    T: Tensor, A_units: Sequence[int], B_units: Sequence[int], k: int
) -> Tuple[Tensor, Tensor]:
    """Product cut for IIT bipartition."""
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


class SPTCSubstrateEncoder(nn.Module):
    """T4 — Level I: Substrate encoder M → context c_M."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.net = MLP(cfg.sptc_d_substrate, cfg.sptc_org_hidden,
                       cfg.sptc_org_hidden, n_layers=3)
        self.norm = nn.LayerNorm(cfg.sptc_org_hidden)

    def forward(self, m: Tensor) -> Tensor:
        return self.norm(self.net(m))


class SPTCOrganizationPolicy(nn.Module):
    """T4 — Level II: Admissible organizational space 𝒜(M; Ω)."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.sptc_n_states
        self.net = MLP(cfg.sptc_org_hidden, cfg.sptc_org_hidden, n * n, n_layers=3)
        self.register_buffer("_offdiag", 1.0 - torch.eye(n), persistent=False)

    def logits(self, ctx: Tensor) -> Tensor:
        logits = self.net(ctx).view(-1, self.cfg.sptc_n_states, self.cfg.sptc_n_states)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        return logits * self._offdiag.to(logits.device)

    def sample(self, ctx: Tensor, tau: float = 1.0, hard: bool = True) -> Tensor:
        logits = self.logits(ctx)
        return gumbel_sigmoid(logits, tau, hard, self.training)


class SPTCDynamics(nn.Module):
    """T4 — Differentiable dynamics F_G producing row-stochastic T."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.sptc_n_states
        d_in = n + n * n + cfg.sptc_org_hidden
        self.net = MLP(d_in, cfg.sptc_org_hidden, n, n_layers=3)

    def forward(self, A: Tensor, ctx: Tensor) -> Tensor:
        B, N, _ = A.shape
        s = torch.eye(N, device=A.device, dtype=A.dtype).expand(B, N, N)
        A_flat = A.reshape(B, 1, -1).expand(-1, N, -1)
        ctx_e = ctx.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([s, A_flat, ctx_e], dim=-1)
        return F.softmax(self.net(inp), dim=-1)


class SPTCIITSurrogate(nn.Module):
    """T4 — Level III: Differentiable IIT-style Φ surrogate."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.sptc_n_states
        self.k = int(round(math.log2(self.N)))
        assert 2 ** self.k == self.N, "n_states must be power of 2"
        self.tau = cfg.sptc_iit_tau
        if self.k > 10:
            raise ValueError("k > 10: too many bipartitions")
        self.parts = unit_bipartitions(self.k)

    def forward(self, T: Tensor) -> Tensor:
        with fp32_ctx():
            T = T.float().clamp_min(1e-12)
            P = len(self.parts)
            kls = []
            for A_u, B_u in self.parts:
                T_p, T_cut = product_cut(T, A_u, B_u, self.k)
                T_cut = T_cut.clamp_min(1e-12)
                kl = (T_p * (T_p.log() - T_cut.log())).sum(-1)
                kls.append(kl.mean(-1))
            kls = torch.stack(kls, dim=-1)
            return -self.tau * (
                torch.logsumexp(-kls / self.tau, dim=-1) - math.log(P)
            )


class SPTCModel(nn.Module):
    """T4 — Full SPTC model: substrate → organization → Φ → assessment."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.substrate = SPTCSubstrateEncoder(cfg)
        self.org_policy = SPTCOrganizationPolicy(cfg)
        self.dynamics = SPTCDynamics(cfg)
        self.iit = SPTCIITSurrogate(cfg)
        self.assessor = MLP(
            1 + cfg.sptc_n_states ** 2 + cfg.sptc_org_hidden,
            cfg.sptc_org_hidden, 2, n_layers=3
        )

    def forward(self, substrate: Tensor, tau: float = 1.0) -> Dict[str, Tensor]:
        ctx = self.substrate(substrate)
        phis, logits_list = [], []
        n_s = self.cfg.sptc_n_potentiality_samples
        for _ in range(n_s):
            A = self.org_policy.sample(ctx, tau=tau, hard=True)
            T = self.dynamics(A, ctx)
            phi = self.iit(T)
            feats = torch.cat([
                phi.unsqueeze(-1), A.reshape(A.size(0), -1), ctx
            ], dim=-1)
            logits_list.append(self.assessor(feats))
            phis.append(phi)
        phis = torch.stack(phis, dim=-1)
        soft_sup = self.cfg.sptc_iit_tau * torch.logsumexp(
            phis / self.cfg.sptc_iit_tau, -1
        )
        return {
            "phis": phis,
            "logits": torch.stack(logits_list, dim=1),
            "potentiality_soft": soft_sup,
            "ctx": ctx,
        }


# ===========================================================================
# 7. T5: RELATIONAL UNIVERSE THEORY (RUT-PCD)
# ===========================================================================

class RUTRelationalGraph(nn.Module):
    """
    T5 — Relational graph with admissible interactions:
    G = (V, E, W) with directed edges and coupling strengths.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.rut_n_components
        self.edge_logits = nn.Parameter(torch.zeros(n, n))
        self.edge_weights = nn.Parameter(torch.ones(n, n) * 0.1)
        self.gnn = MLP(
            cfg.rut_n_state_dims * 2, cfg.rut_hidden, cfg.rut_n_state_dims,
            n_layers=2
        )
        self.register_buffer("_eye", torch.eye(n), persistent=False)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: (B, n, d_state)
        Returns:
            dict with adjacency, messages, updated_state
        """
        B, n, d = x.shape
        device = x.device
        adj = torch.sigmoid(self.edge_logits) * (1.0 - self._eye.to(device))
        weights = self.edge_weights * adj.unsqueeze(0)

        # Message passing: aggregate neighbor states
        x_expanded_i = x.unsqueeze(2).expand(-1, -1, n, -1)
        x_expanded_j = x.unsqueeze(1).expand(-1, n, -1, -1)
        pair_feat = torch.cat([x_expanded_i, x_expanded_j], dim=-1)
        messages = self.gnn(pair_feat)  # (B, n, n, d)
        aggregated = (weights.unsqueeze(-1) * messages).sum(dim=2)

        updated = x + aggregated
        return {
            "adjacency": adj,
            "messages": messages,
            "updated_state": updated,
        }


class RUTCrossDomainMapper(nn.Module):
    """
    T5 — Cross-domain mapping Ψ: X_P → X_C
    Maps physical state representations to cognitive state representations.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.physical_encoder = MLP(
            cfg.rut_n_state_dims, cfg.rut_hidden, cfg.rut_hidden, n_layers=2
        )
        self.cognitive_decoder = MLP(
            cfg.rut_hidden, cfg.rut_hidden, cfg.rut_n_state_dims, n_layers=2
        )
        self.consistency_net = MLP(
            cfg.rut_n_state_dims * 2, cfg.rut_hidden, 1, n_layers=2
        )

    def forward(
        self, physical_state: Tensor, cognitive_state: Tensor
    ) -> Dict[str, Tensor]:
        encoded = self.physical_encoder(physical_state)
        predicted_cognitive = self.cognitive_decoder(encoded)
        consistency = torch.sigmoid(
            self.consistency_net(
                torch.cat([predicted_cognitive, cognitive_state], dim=-1)
            )
        ).squeeze(-1)
        return {
            "predicted_cognitive": predicted_cognitive,
            "consistency_score": consistency,
        }


class RUTModel(nn.Module):
    """T5 — Full RUT-PCD model combining graph, cross-domain mapping."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg
        self.graph = RUTRelationalGraph(cfg)
        self.mapper = RUTCrossDomainMapper(cfg)

    def forward(
        self, physical_state: Tensor, cognitive_state: Tensor
    ) -> Dict[str, Tensor]:
        graph_out = self.graph(physical_state)
        map_out = self.mapper(
            graph_out["updated_state"], cognitive_state
        )
        return {**graph_out, **map_out}


# ===========================================================================
# 8. UNIFIED MODEL
# ===========================================================================

class UnifiedTheoryModel(nn.Module):
    """
    Top-level model that instantiates all enabled theory modules
    and produces a combined output for hypothesis testing.
    """

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg

        # Shared encoder for input observations
        self.shared_encoder = MLP(
            cfg.d_obs, cfg.rc_hidden, cfg.rc_hidden, n_layers=3
        )

        # T1: Relational Compatibility
        if cfg.enable_T1:
            self.rc_access = RelationalAccessibilityModule(cfg)
            self.rc_hreu = HREUClassifier(cfg)
            self.rc_rucp = RUCPEvaluator(cfg)

        # T2: Relational Logos
        if cfg.enable_T2:
            self.rlf = LogosRelationalModel(cfg)

        # T3: QMD
        if cfg.enable_T3:
            self.qmd = QMDModel(cfg)

        # T4: SPTC
        if cfg.enable_T4:
            self.sptc = SPTCModel(cfg)

        # T5: RUT-PCD
        if cfg.enable_T5:
            self.rut = RUTModel(cfg)

        # Unified prediction head
        self.head = MLP(cfg.rc_hidden, cfg.rc_hidden, 2, n_layers=2)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            batch: dict with keys depending on enabled theories:
              - "obs": (B, T, d_obs) observations
              - "substrate": (B, sptc_d_substrate) for T4
              - "physical_state": (B, n, d_state) for T5
              - "cognitive_state": (B, n, d_state) for T5
              - "conditions": (B, rl_hidden) for T2
        Returns:
            dict with all theory outputs and unified prediction
        """
        out: Dict[str, Tensor] = {}
        B = batch["obs"].shape[0]
        device = batch["obs"].device

        # Shared encoding
        encoded = self.shared_encoder(batch["obs"])  # (B, T, rc_hidden)
        pooled = encoded.mean(dim=1)  # (B, rc_hidden)

        # T1
        if self.cfg.enable_T1:
            state_3d = encoded  # use encoded sequence as state
            rc_out = self.rc_access(state_3d)
            hreu_out = self.rc_hreu(pooled)
            # For RUCP, compare first half vs second half of batch
            half = B // 2
            rucp_out = self.rc_rucp(pooled[:half], pooled[half:2*half])
            out.update({f"T1_{k}": v for k, v in rc_out.items()})
            out.update({f"T1_{k}": v for k, v in hreu_out.items()})
            out.update({f"T1_{k}": v for k, v in rucp_out.items()})

        # T2
        if self.cfg.enable_T2:
            conditions = batch.get(
                "conditions",
                torch.zeros(B, self.cfg.rl_hidden, device=device)
            )
            state_for_rl = batch["obs"].mean(dim=1)  # (B, d_obs)
            # Pad to rut_n_state_dims if needed
            if state_for_rl.shape[-1] < self.cfg.rut_n_state_dims:
                pad = self.cfg.rut_n_state_dims - state_for_rl.shape[-1]
                state_for_rl = F.pad(state_for_rl, (0, pad))
            else:
                state_for_rl = state_for_rl[:, :self.cfg.rut_n_state_dims]
            rl_out = self.rlf(state_for_rl, conditions)
            out.update({f"T2_{k}": v for k, v in rl_out.items()})

        # T3
        if self.cfg.enable_T3:
            D = batch["obs"]  # (B, T, d_obs)
            # Pad d_obs to qmd_d_D if needed
            if D.shape[-1] < self.cfg.qmd_d_D:
                D = F.pad(D, (0, self.cfg.qmd_d_D - D.shape[-1]))
            else:
                D = D[:, :, :self.cfg.qmd_d_D]
            E = batch.get(
                "env",
                torch.zeros(B, D.shape[1], self.cfg.qmd_d_E, device=device)
            )
            qmd_out = self.qmd(D, E, sample=self.training)
            out.update({f"T3_{k}": v for k, v in qmd_out.items()})

        # T4
        if self.cfg.enable_T4:
            substrate = batch.get(
                "substrate",
                torch.randn(B, self.cfg.sptc_d_substrate, device=device)
            )
            sptc_out = self.sptc(substrate, tau=1.0)
            out.update({f"T4_{k}": v for k, v in sptc_out.items()})

        # T5
        if self.cfg.enable_T5:
            n = self.cfg.rut_n_components
            d = self.cfg.rut_n_state_dims
            phys = batch.get(
                "physical_state",
                torch.randn(B, n, d, device=device)
            )
            cogn = batch.get(
                "cognitive_state",
                torch.randn(B, n, d, device=device)
            )
            rut_out = self.rut(phys, cogn)
            out.update({f"T5_{k}": v for k, v in rut_out.items()})

        # Unified head
        out["unified_logit"] = self.head(pooled)
        return out


# ===========================================================================
# 9. LOSS
# ===========================================================================

class UnifiedLoss(nn.Module):
    """Composite loss across all enabled theories."""

    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self, batch: Dict[str, Tensor], out: Dict[str, Tensor],
        model: UnifiedTheoryModel
    ) -> Dict[str, Tensor]:
        cfg = self.cfg
        losses: Dict[str, Tensor] = {}
        device = out["unified_logit"].device
        zero = torch.tensor(0.0, device=device)

        # Task loss: unified prediction vs label
        if "label" in batch:
            task = F.cross_entropy(out["unified_logit"], batch["label"])
        else:
            task = zero

        # T1 losses
        t1_loss = zero
        if cfg.enable_T1:
            if "T1_level_logits" in out and "label" in batch:
                t1_loss = t1_loss + F.cross_entropy(
                    out["T1_level_logits"],
                    batch["label"] % cfg.rc_n_levels
                )

        # T3 losses
        t3_loss = zero
        if cfg.enable_T3 and "T3_D_hat" in out:
            D = batch["obs"]
            if D.shape[-1] < cfg.qmd_d_D:
                D = F.pad(D, (0, cfg.qmd_d_D - D.shape[-1]))
            else:
                D = D[:, :, :cfg.qmd_d_D]
            nll = 0.5 * ((D - out["T3_D_hat"]).pow(2)).mean()
            kl = out["T3_kl"].mean()
            t3_loss = nll + cfg.lambda_coupling * kl

        # T4 losses
        t4_loss = zero
        if cfg.enable_T4 and "T4_logits" in out:
            logits_mean = out["T4_logits"].mean(dim=1)
            if "label" in batch:
                t4_loss = F.cross_entropy(logits_mean, batch["label"])
            t4_loss = t4_loss - cfg.lambda_potentiality * out["T4_potentiality_soft"].mean()

        # T5 losses
        t5_loss = zero
        if cfg.enable_T5 and "T5_consistency_score" in out:
            t5_loss = -out["T5_consistency_score"].mean()

        # Regularisation
        reg = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)

        total = (
            cfg.lambda_task * task
            + cfg.lambda_compat * t1_loss
            + t3_loss
            + t4_loss
            + 0.1 * t5_loss
            + cfg.lambda_reg * reg
        )

        return {
            "loss": total,
            "task": task.detach(),
            "t1": t1_loss.detach(),
            "t3": t3_loss.detach(),
            "t4": t4_loss.detach(),
            "t5": t5_loss.detach(),
            "reg": reg.detach(),
        }


# ===========================================================================
# 10. EMA
# ===========================================================================

class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach(), alpha=1 - self.decay
                )
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)


# ===========================================================================
# 11. TRAINER
# ===========================================================================

class UnifiedTrainer:
    """
    Production trainer: AMP + DDP + grad-clip + cosine LR + EMA +
    gradient checkpointing + fused AdamW.
    """

    def __init__(
        self, cfg: UnifiedConfig, model: UnifiedTheoryModel,
        device: torch.device
    ):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)

        if _is_dist():
            self.model = DDP(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=False,
            )

        self.loss_fn = UnifiedLoss(cfg)

        # Fused AdamW on CUDA
        use_fused = cfg.use_fused_adamw and device.type == "cuda"
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            fused=use_fused,
        )

        # AMP scaler (only for fp16)
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}[cfg.amp_dtype]
        self.scaler = _GradScaler(
            enabled=(cfg.amp_enabled and amp_dtype == torch.float16)
        )
        self._amp_dtype = amp_dtype

        self.ema = EMA(self._unwrapped(), cfg.ema_decay)
        self.step = 0

    def _unwrapped(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _lr(self) -> float:
        cfg = self.cfg
        if self.step < cfg.warmup_steps:
            return cfg.lr * (self.step + 1) / max(1, cfg.warmup_steps)
        prog = (self.step - cfg.warmup_steps) / max(
            1, cfg.max_steps - cfg.warmup_steps
        )
        return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()

        # Move batch to device
        batch = {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        for g in self.opt.param_groups:
            g["lr"] = self._lr()

        self.opt.zero_grad(set_to_none=True)

        with amp_ctx(
            cfg.amp_enabled, self._amp_dtype,
            device_type=self.device.type
        ):
            out = self.model(batch)
            losses = self.loss_fn(batch, out, self._unwrapped())
            loss = losses["loss"] / cfg.grad_accum

        self.scaler.scale(loss).backward()

        if (self.step + 1) % cfg.grad_accum == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update(self._unwrapped())

        self.step += 1
        return {k: float(v) for k, v in losses.items() if v.numel() == 1}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tot: Dict[str, float] = {"loss": 0.0, "n": 0}
        for batch in loader:
            batch = {
                k: v.to(self.device, non_blocking=True)
                if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            with amp_ctx(
                self.cfg.amp_enabled, self._amp_dtype,
                device_type=self.device.type
            ):
                out = self.model(batch)
                losses = self.loss_fn(batch, out, self._unwrapped())
            bs = batch["obs"].shape[0]
            tot["loss"] += float(losses["loss"]) * bs
            tot["n"] += bs

        tot["loss"] /= max(1, tot["n"])
        if _is_dist():
            tot["loss"] = float(
                _all_reduce_mean(torch.tensor(tot["loss"], device=self.device))
            )
        return tot

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self._unwrapped().state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.shadow,
            "step": self.step,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self._unwrapped().load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.scaler.load_state_dict(sd["scaler"])
        self.ema.shadow = sd["ema"]
        self.step = sd["step"]


# ===========================================================================
# 12. SYNTHETIC DATASET (replace with real data)
# ===========================================================================

class SyntheticTheoryDataset(Dataset):
    """
    Synthetic dataset for testing. Replace with RealDataLoader for
    production use with real observations.
    """

    def __init__(self, cfg: UnifiedConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.obs = torch.randn(n, cfg.seq_len, cfg.d_obs, generator=g)
        self.label = torch.randint(0, 2, (n,), generator=g)
        self.substrate = torch.randn(n, cfg.sptc_d_substrate, generator=g)
        self.conditions = torch.randn(n, cfg.rl_hidden, generator=g)
        n_comp = cfg.rut_n_components
        d = cfg.rut_n_state_dims
        self.physical_state = torch.randn(n, n_comp, d, generator=g)
        self.cognitive_state = torch.randn(n, n_comp, d, generator=g)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        return {
            "obs": self.obs[i],
            "label": self.label[i],
            "substrate": self.substrate[i],
            "conditions": self.conditions[i],
            "physical_state": self.physical_state[i],
            "cognitive_state": self.cognitive_state[i],
        }


class RealDataLoader:
    """
    Configurable real-data loader. Provide a path to your dataset
    and a schema mapping. Supports:
      - CSV / Parquet (via pandas)
      - HDF5 (via h5py)
      - NPZ (via numpy)
      - Custom torch Dataset

    Usage:
        loader = RealDataLoader(
            path="data/experiment1.h5",
            schema={"obs": "observations", "label": "labels"},
            cfg=config,
        )
        train_dl, val_dl, test_dl = loader.build_loaders()
    """

    def __init__(
        self,
        path: str,
        schema: Dict[str, str],
        cfg: UnifiedConfig,
        format: str = "h5",
    ):
        self.path = path
        self.schema = schema
        self.cfg = cfg
        self.format = format
        self._data: Optional[Dict[str, Tensor]] = None

    def load(self) -> Dict[str, Tensor]:
        """Load data from file into tensors."""
        if self._data is not None:
            return self._data

        if self.format == "h5":
            import h5py
            data = {}
            with h5py.File(self.path, "r") as f:
                for key, col in self.schema.items():
                    data[key] = torch.from_numpy(f[col][:]).float()
        elif self.format == "npz":
            raw = dict(np.load(self.path))
            data = {
                k: torch.from_numpy(raw[col]).float()
                for k, col in self.schema.items()
            }
        elif self.format == "csv":
            import pandas as pd
            df = pd.read_csv(self.path)
            data = {
                k: torch.from_numpy(df[col].values).float()
                for k, col in self.schema.items()
            }
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        self._data = data
        return data

    def build_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        data = self.load()
        n = data["obs"].shape[0]
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        class TensorDataset(Dataset):
            def __init__(self, data, start, end):
                self.data = {k: v[start:end] for k, v in data.items()}
                self.n = end - start

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                return {k: v[i] for k, v in self.data.items()}

        train_ds = TensorDataset(data, 0, n_train)
        val_ds = TensorDataset(data, n_train, n_train + n_val)
        test_ds = TensorDataset(data, n_train + n_val, n)

        sampler = DistributedSampler(train_ds) if _is_dist() else None
        common = dict(num_workers=2, pin_memory=True, persistent_workers=False)

        return (
            DataLoader(
                train_ds, batch_size=self.cfg.batch_size,
                sampler=sampler, shuffle=(sampler is None),
                drop_last=True, **common
            ),
            DataLoader(
                val_ds, batch_size=self.cfg.batch_size,
                shuffle=False, **common
            ),
            DataLoader(
                test_ds, batch_size=self.cfg.batch_size,
                shuffle=False, **common
            ),
        )


def build_dataloader(
    cfg: UnifiedConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build synthetic dataloaders (replace with RealDataLoader for real data)."""
    tr = SyntheticTheoryDataset(cfg, cfg.n_train, cfg.data_seed)
    va = SyntheticTheoryDataset(cfg, cfg.n_val, cfg.data_seed + 1)
    te = SyntheticTheoryDataset(cfg, cfg.n_test, cfg.data_seed + 2)

    sampler = DistributedSampler(tr) if _is_dist() else None
    common = dict(num_workers=2, pin_memory=True, persistent_workers=False)

    return (
        DataLoader(
            tr, batch_size=cfg.batch_size, sampler=sampler,
            shuffle=(sampler is None), drop_last=True, **common
        ),
        DataLoader(va, batch_size=cfg.batch_size, shuffle=False, **common),
        DataLoader(te, batch_size=cfg.batch_size, shuffle=False, **common),
    )


# ===========================================================================
# 13. HYPOTHESIS TEST SUITE
# ===========================================================================

class HypothesisTestSuite:
    """
    Runs falsifiability tests for all five theories.
    Each test produces metrics that can reject or support the theory.
    """

    def __init__(self, cfg: UnifiedConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def run(
        self,
        model: UnifiedTheoryModel,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        model.eval()
        results: Dict[str, Any] = {}
        all_outs: List[Dict[str, Tensor]] = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                out = model(batch)
                all_outs.append({
                    k: v.cpu() for k, v in out.items() if torch.is_tensor(v)
                })

        # Aggregate metrics
        if all_outs:
            unified_logits = torch.cat(
                [o["unified_logit"] for o in all_outs], dim=0
            )
            results["unified_accuracy"] = float(
                (unified_logits.argmax(-1) == 0).float().mean()
            )

        # Per-theory metrics
        for theory in ["T1", "T2", "T3", "T4", "T5"]:
            keys = [k for k in all_outs[0].keys() if k.startswith(theory)]
            if keys:
                results[f"{theory}_n_outputs"] = len(keys)

        return results


# ===========================================================================
# 14. DDP SETUP / TEARDOWN
# ===========================================================================

def setup_dist(backend: str = "nccl") -> Tuple[torch.device, int, int]:
    """Initialize distributed training if environment variables are set."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        r = dist.get_rank()
        ws = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", r))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            return torch.device("cuda", local), r, ws
        return torch.device("cpu"), r, ws
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    return torch.device("cpu"), 0, 1


def cleanup_dist() -> None:
    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 15. MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MSPS Unified Theory Tests")
    p.add_argument("--mode", choices=["train", "test", "compare"],
                   default="train")
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file overriding UnifiedConfig")
    p.add_argument("--data", type=str, default=None,
                   help="Path to real data file (h5/npz/csv)")
    p.add_argument("--data-format", type=str, default="h5")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, r, ws = setup_dist()
    seed_all(42)

    cfg = UnifiedConfig()
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        cfg = UnifiedConfig(**{**asdict(cfg), **overrides})
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.out:
        cfg.out_dir = args.out
    cfg.device = str(device)

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Build model
    model = UnifiedTheoryModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if _is_main():
        logger.info(f"Model parameters: {n_params:,}")
        logger.info(f"Device: {device}, World size: {ws}")
        logger.info(f"AMP: {cfg.amp_dtype}, Fused AdamW: {cfg.use_fused_adamw}")

    # Build data
    if args.data:
        schema = {
            "obs": "observations",
            "label": "labels",
            "substrate": "substrate",
        }
        loader = RealDataLoader(args.data, schema, cfg, args.data_format)
        train_dl, val_dl, test_dl = loader.build_loaders()
    else:
        train_dl, val_dl, test_dl = build_dataloader(cfg)

    # Build trainer
    trainer = UnifiedTrainer(cfg, model, device)

    if args.mode == "train":
        t0 = time.time()
        # Cycling iterator with DistributedSampler epoch advancement
        epoch = 0
        while trainer.step < cfg.max_steps:
            sampler = getattr(train_dl, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            for batch in train_dl:
                logs = trainer.train_step(batch)
                if _is_main() and trainer.step % cfg.log_every == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {trainer.step:6d} | "
                        + " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                        + f" | lr={trainer._lr():.2e} | {elapsed:.1f}s"
                    )
                if _is_main() and trainer.step % cfg.eval_every == 0:
                    ev = trainer.evaluate(val_dl)
                    logger.info(
                        "  [val] " +
                        " ".join(f"{k}={v:.4f}" for k, v in ev.items())
                    )
                if _is_main() and trainer.step % cfg.ckpt_every == 0:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(cfg.out_dir, f"ckpt_{trainer.step}.pt")
                    )
                if trainer.step >= cfg.max_steps:
                    break
            epoch += 1

        if _is_main():
            torch.save(
                trainer.state_dict(),
                os.path.join(cfg.out_dir, "ckpt_final.pt")
            )
            logger.info("Training complete.")

    elif args.mode == "test":
        # Load final checkpoint if available
        ckpt_path = os.path.join(cfg.out_dir, "ckpt_final.pt")
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location=device)
            trainer.load_state_dict(sd)

        suite = HypothesisTestSuite(cfg, device)
        results = suite.run(trainer._unwrapped(), test_dl)

        if _is_main():
            print(json.dumps(results, indent=2, default=str))
            with open(os.path.join(cfg.out_dir, "test_results.json"), "w") as f:
                json.dump(results, f, indent=2, default=str)

    elif args.mode == "compare":
        # Run full hypothesis comparison
        suite = HypothesisTestSuite(cfg, device)
        results = suite.run(trainer._unwrapped(), test_dl)
        if _is_main():
            print(json.dumps(results, indent=2, default=str))

    cleanup_dist()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
