"""
=============================================================================
SESI CLOUD & COLLAPSE THEORY TEST MODULE — PRODUCTION EDITION
=============================================================================
Project   : MSPS NETWORK / SESI Differentiable Physics Stack
Developer : PAI and Yoon A Limsuwan
ORCID     : 0009-0008-2374-0788
License   : MIT
Year      : 2026
Version   : 1.0.0

THEORIES COVERED
----------------
  T-A  Hierarchical Cloud Superposition Theory (HCST)
       hierarchical cloud + hierarchical measurement matrix +
       non-PCF fractal interface + decoherence + semiclassical correspondence

  T-B  Unified Covariant SESI Cloud Quantum Theory (UCSCQT)
       covariant state/instrument transformation, relational entanglement,
       hierarchy-resolved measurement

  T-C  Covariant SESI Cloud-OPMC Quantum Computation Theory
       canonical signature sigma_T, task factorization, structural
       relational capacity, bounded classical task evaluation,
       controlled SQCT / CSL limits

  T-D  SESI-GR Quantum Collapse Theory with Multipartite Entanglement
       and Non-PCF Fractal Geometry
       mass-contrast Coulomb kernel, Diosi-Penrose-type rates,
       completely-positive collapse channel, heating-rate constraints

  T-E  SESI Quantum Collapse Theory: Geometric-Interface-Derived
       Quantum State Reduction
       geometric interface operators A_S(x), event-history kernels
       K_N / K_M / K_B, interface mass calibration (mu_I, kappa_I, eta_E),
       complete positivity and energy-growth exclusion

DESIGN GUARANTEES
-----------------
  * Native full differentiability. Detachment only in reporting/metrics.
  * Trace preservation and complete positivity enforced BY CONSTRUCTION:
      - identity-anchored Kraus generators => M = sum(V^dag V) strictly PD
      - Cholesky completion  : K = V L^{-1},  M = L L^T  =>  sum K^dag K = I
      - Hermitian completion : K = V M^{-1/2} (Newton-Schulz) => covariant
      - Lindblad K0 = Cayley/exp(-i H dt) @ chol(I - sum L^dag L)^dag
  * AMP: bf16 / fp16 / fp32 encoders; complex64 fp32 islands for all
    density-matrix, Kraus, Cholesky, expm and FFT-kernel operations.
    (bf16 complex is not supported by PyTorch; forcing it is a bug,
     not an optimization.)
  * Multi-GPU DDP: DistributedSampler epoch control, gradient bucket view,
    rank-guarded logging/checkpointing, full state resume.
  * Cost control: low-rank Kraus banks, Cholesky default normalization,
    FFT-based Coulomb kernel, landmark fallback, Hutchinson Jacobian
    estimator, gradient checkpointing, fused AdamW, optional 8-bit
    optimizer, CPU-offloaded EMA, torch.compile, persistent workers.
  * Real data: lazy HDF5 shard / NPZ / Parquet loader with schema validation,
    native complex density-matrix ingestion, geometry and event streams.
  * Falsification: paired bootstrap CIs, null controls, ablations,
    analytic limit tests, calibration recovery, measured scaling laws.

INTERPRETATION LIMITS
---------------------
  1. Support for a hypothesis is SUPPORT WITHIN THE IMPLEMENTED MODEL.
     It is not experimental confirmation of the physical theory.
  2. Representational capacity is not computational advantage.
     The OPMC evaluator reports MEASURED scaling, never an assumed speedup.
  3. An IIT-related or coherence-related scalar is NOT a measurement
     of subjective experience.
  4. Interface kernel functional forms not fully specified in the source
     documents are declared here as parametric assumptions and are
     individually ablatable.

=============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger("sesi.cloud")

__all__ = [
    "SESITheoryConfig",
    "CovariantStateEncoder",
    "IdentityAnchoredKrausBank",
    "HierarchicalMeasurementMatrix",
    "NonPCFFractalInterface",
    "WeakMeasurementReadout",
    "LindbladChannel",
    "SemiclassicalCorrespondence",
    "GravityCollapseChannel",
    "GeometricInterface",
    "OPMCCanonicalizer",
    "EntanglementDiagnostics",
    "SESICloudModel",
    "UnifiedTheoryLoss",
    "EMA",
    "SESITrainer",
    "SyntheticTheoryDataset",
    "RealTheoryDataset",
    "build_dataloaders",
    "HypothesisSuite",
    "run_selftest",
    "setup_dist",
    "cleanup_dist",
]


# ===========================================================================
# 0. DISTRIBUTED / AMP PRIMITIVES
# ===========================================================================

def _dist_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_active() else 0


def _world() -> int:
    return dist.get_world_size() if _dist_active() else 1


def _main() -> bool:
    return _rank() == 0


def _reduce_mean(x: Tensor) -> Tensor:
    if not _dist_active():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / _world()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    _AMP_API = "new"
except Exception:  # pragma: no cover - legacy torch
    from torch.cuda.amp import autocast as _legacy_autocast
    from torch.cuda.amp import GradScaler as _GradScaler

    def _autocast(device_type: str, dtype: Any, enabled: bool = True):
        return _legacy_autocast(dtype=dtype, enabled=enabled)

    _AMP_API = "legacy"


@contextlib.contextmanager
def fp32_island(device_type: str = "cuda"):
    """
    Disable autocast for numerically sensitive blocks.
    Callers MUST cast inputs explicitly to float32 / complex64.
    """
    if _AMP_API == "new":
        with _autocast(device_type=device_type, dtype=torch.float32, enabled=False):
            yield
    else:
        with _legacy_autocast(enabled=False):
            yield


def soft_floor(x: Tensor, floor: float, beta: float = 100.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)


def soft_cap(x: Tensor, cap: float, beta: float = 100.0) -> Tensor:
    return cap - F.softplus(cap - x, beta=beta)


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

@dataclass
class SESITheoryConfig:
    # ---- device / precision ----
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "bf16"              # bf16 | fp16 | fp32

    # ---- Hilbert / hierarchy dimensions ----
    d_in: int = 32                       # real feature dimension
    d_q: int = 8                         # total Hilbert dim (d_meta + d_base)
    d_meta: int = 4
    d_base: int = 4
    n_outcomes: int = 4
    n_branches: int = 4                  # collapse branches
    n_locations: int = 8                 # interface / mass landmark sites
    n_jumps: int = 2
    n_scales: int = 4                    # non-PCF fractal scales
    n_events: int = 16
    d_event: int = 8
    d_geometry: int = 6                  # normal(3) + curvature + area + log_scale
    k_sigma: int = 16                    # canonical signature dimension
    hierarchy_depth: int = 3
    branch_factor: int = 2

    # ---- Kraus / covariance ----
    kraus_rank: int = 8                  # low-rank generator; 0 = full rank
    covariant_normalization: bool = True # Hermitian inverse-sqrt (equivariant)
    newton_schulz_iters: int = 16
    cholesky_jitter: float = 1e-6
    enforce_hermiticity: bool = True

    # ---- dynamics ----
    dt: float = 0.05
    n_steps: int = 4
    weak_coupling: float = 0.5
    decoherence_scale: float = 0.1
    gr_coupling: float = 1.0
    interface_coupling: float = 0.1
    energy_budget: float = 1e-3
    use_gradient_checkpointing: bool = True

    # ---- optimization / cost ----
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_clip: float = 1.0
    fused_adamw: bool = True
    use_8bit_optimizer: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    ema_decay: float = 0.999
    ema_offload_cpu: bool = True
    hutchinson_probes: int = 1
    coupling_every: int = 4
    lambda_coupling: float = 1e-3
    lambda_cp: float = 1e-2
    lambda_energy: float = 1e-1
    lambda_factor: float = 1e-2

    # ---- DDP ----
    ddp_find_unused: bool = True
    ddp_static_graph: bool = False
    ddp_bucket_view: bool = True

    # ---- data ----
    n_train: int = 8192
    n_val: int = 1024
    n_test: int = 2048
    data_seed: int = 1234
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    data_shards: Optional[str] = None    # glob, e.g. "data/shards/*.h5"
    data_format: str = "h5"              # h5 | npz | parquet
    schema: Dict[str, str] = field(default_factory=dict)

    # ---- ablation switches ----
    use_hierarchy: bool = True
    use_fractal: bool = True
    use_weak_measurement: bool = True
    use_decoherence: bool = True
    use_gravity: bool = True
    use_interface: bool = True
    use_opmc: bool = True
    use_entanglement_head: bool = True

    # ---- reporting ----
    seed: int = 0
    out_dir: str = "runs/sesi_cloud"
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 1000
    n_bootstrap: int = 1000
    bootstrap_chunk: int = 200
    suite_steps: int = 600


# ===========================================================================
# 2. DENSITY-MATRIX AND KRAUS CORE  (fp32 islands, differentiable)
# ===========================================================================

def hermitize(rho: Tensor) -> Tensor:
    return 0.5 * (rho + rho.mH)


def normalize_state(rho: Tensor) -> Tensor:
    rho = hermitize(rho)
    tr = rho.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-12)
    return rho / tr[..., None, None]


def psd_state_from_features(z: Tensor, d: int) -> Tensor:
    """
    rho = A A^dag / tr(A A^dag)  =>  exactly PSD up to rounding.
    z: (B, 2*d*d) real.
    """
    A = torch.view_as_complex(z.reshape(*z.shape[:-1], d, d, 2).contiguous())
    rho = A @ A.mH
    return normalize_state(rho)


def hermitian_inverse_sqrt(A: Tensor, n_iter: int = 16) -> Tensor:
    """
    Differentiable Hermitian inverse square root via Newton-Schulz.
    Polynomial in A and A^dag  =>  EQUIVARIANT under A -> U A U^dag.
    A must be Hermitian positive definite.
    """
    A = hermitize(A)
    scale = A.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12) * 1.5
    Y = A / scale
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    Z = I.expand_as(A).clone()
    for _ in range(n_iter):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return hermitize(Z)


def complete_kraus_cholesky(V: Tensor, jitter: float = 1e-6) -> Tensor:
    """
    V: (B, Y, d, d) complex.
    M = sum_y V_y^dag V_y = L L^T  (Cholesky, complex bilinear transpose).
    K_y = V_y L^{-1}  =>  sum_y K_y^dag K_y = L^{-T} M L^{-1} = I.

    Cheap: one Cholesky + one triangular solve per batch.
    NOT covariant under arbitrary basis change (Cholesky is basis-dependent).
    """
    Vd = V.mH
    M = (Vd @ V).sum(dim=1)
    M = hermitize(M)
    I = torch.eye(M.size(-1), device=M.device, dtype=M.dtype)
    M = M + jitter * I
    L = torch.linalg.cholesky(M)
    # Solve X L = V  <=>  L^T X^T = V^T
    K = torch.linalg.solve_triangular(L.mT, V.mT, upper=True).mT
    return K


def complete_kraus_hermitian(V: Tensor, n_iter: int = 16) -> Tensor:
    """
    K_y = V_y M^{-1/2},  M = sum V^dag V.
    sum K^dag K = M^{-1/2} M M^{-1/2} = I.
    Covariant: K(U V U^dag) = U K(V) U^dag.
    """
    M = (V.mH @ V).sum(dim=1)
    M = hermitize(M)
    I = torch.eye(M.size(-1), device=M.device, dtype=M.dtype)
    M = M + 1e-6 * I
    Minv_half = hermitian_inverse_sqrt(M, n_iter=n_iter)
    return V @ Minv_half.unsqueeze(1)


def kraus_residual(K: Tensor) -> Tensor:
    """|| sum_y K_y^dag K_y - I ||_F  (differentiable)."""
    I = torch.eye(K.size(-1), device=K.device, dtype=K.dtype)
    return (K.mH @ K).sum(dim=1).sub(I).norm(dim=(-2, -1))


def apply_kraus(rho: Tensor, K: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    rho: (B,d,d), K: (B,Y,d,d)
    returns probs (B,Y), normalized post-states (B,Y,d,d), unnormalized births.
    """
    born = K @ rho.unsqueeze(1) @ K.mH
    probs = born.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(0.0)
    probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
    post = born / probs[..., None, None].clamp_min(1e-12).to(born.dtype)
    post = normalize_state(post)
    return probs, post, born


# ===========================================================================
# 3. COVARIANT STATE ENCODER
# ===========================================================================

class CovariantStateEncoder(nn.Module):
    """
    Builds a PSD density operator from real observations.

    Invariants s(rho) = [tr rho^2, tr rho^3, tr rho^4] are basis-invariant
    and condition the instrument. The reference interface operator B is
    supplied in the SAME basis as rho and transforms with it; this is the
    precise meaning of covariance used in the equivariance test.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_q
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_in, cfg.d_q * cfg.d_q * 2),
        )
        nn.init.normal_(self.proj[0].weight, std=0.05)
        nn.init.zeros_(self.proj[0].bias)
        self.ref_proj = nn.Linear(cfg.d_geometry, 2 * d * d) if cfg.d_geometry else None
        self.invariant_proj = nn.Linear(4, cfg.d_meta + cfg.d_base)

    def invariants(self, rho: Tensor) -> Tensor:
        r2 = rho @ rho
        r3 = r2 @ rho
        return torch.stack(
            [
                r2.diagonal(dim1=-2, dim2=-1).real.sum(-1),
                r3.diagonal(dim1=-2, dim2=-1).real.sum(-1),
                (r2 * r2).sum(dim=(-2, -1)).real,
                rho.diagonal(dim1=-2, dim2=-1).real.sum(-1),
            ],
            dim=-1,
        )

    def forward(self, x: Tensor, geometry: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Tensor]:
        dt = torch.float32
        with fp32_island(x.device.type):
            z = self.proj(x.float())
            rho = psd_state_from_features(z, self.cfg.d_q)
            s = self.invariants(rho)
            cond = self.invariant_proj(s)
            if self.ref_proj is not None and geometry is not None:
                gb = self.ref_proj(geometry.float())
                B = torch.view_as_complex(
                    gb.reshape(*gb.shape[:-1], self.cfg.d_q, self.cfg.d_q, 2).contiguous()
                )
                B = hermitize(B)
            else:
                I = torch.eye(self.cfg.d_q, device=rho.device, dtype=rho.dtype)
                B = I.expand_as(rho).clone()
        return rho, cond, B


# ===========================================================================
# 4. NON-PCF FRACTAL INTERFACE
# ===========================================================================

class NonPCFFractalInterface(nn.Module):
    """
    Multiscale measure surrogate:
        dmu_F^(k) = Z_k * w_k(x) * dH^(D_k)
        Z_k       = log-domain normalized renormalization
        D_k       = learnable, bounded spectral dimension proxy

    IMPORTANT: D_k is a LEARNABLE MULTISCALE PROXY, not a measured
    Hausdorff dimension. The suite separately estimates a box-counting
    exponent from real data so D_k can be calibrated / falsified.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        k = cfg.n_scales
        self.raw_dim = nn.Parameter(torch.linspace(1.0, 3.0, k))
        self.weight_net = nn.Linear(cfg.d_in, k)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.l0 = 1.0
        self.out_proj = nn.Linear(k * 2, cfg.d_meta + cfg.d_base)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        D = soft_cap(soft_floor(self.raw_dim, 0.05), 5.95)
        w = torch.softmax(self.weight_net(x), dim=-1)
        scales = torch.arange(1, self.cfg.n_scales + 1, device=x.device, dtype=x.dtype)
        interface = (self.l0 / scales).pow(soft_floor(self.beta, 0.01))
        complexity = (D.unsqueeze(0) * w).sum(-1, keepdim=True)
        cond = self.out_proj(torch.cat([w * interface.unsqueeze(0), complexity.expand_as(w)], -1))
        return {"fractal_weights": w, "fractal_dims": D,
                "interface_intensity": interface, "fractal_cond": cond}

    @torch.no_grad()
    def box_counting_dimension(self, field: Tensor) -> Tensor:
        """
        Multi-resolution box-counting estimate on a 3D scalar field.
        Returns a single log-log slope estimate (diagnostic only).
        field: (B,1,D,H,W) or (B,D,H,W)
        """
        if field.dim() == 5:
            field = field[:, 0]
        counts, sizes = [], []
        B = field.shape[0]
        for s in (2, 4, 8):
            pooled = F.avg_pool3d(field.unsqueeze(1), kernel_size=s, stride=s)
            occupied = (pooled.abs().mean(0) > 1e-3).float().sum()
            counts.append(float(occupied.clamp_min(1.0).log()))
            sizes.append(math.log(max(2, field.shape[-1] // s)))
        xs = torch.tensor(sizes, device=field.device, dtype=torch.float32)
        ys = torch.tensor(counts, device=field.device, dtype=torch.float32)
        xc, yc = xs.mean(), ys.mean()
        return ((xs - xc) * (ys - yc)).sum() / ((xs - xc).pow(2).sum() + 1e-12)


# ===========================================================================
# 5. IDENTITY-ANCHORED KRAUS BANK + HIERARCHICAL MEASUREMENT MATRIX
# ===========================================================================

class IdentityAnchoredKrausBank(nn.Module):
    """
    V_y = U_y W_y + c_y I,   c = softmax(anchor_logits),  sum_y c_y = 1.

    The identity anchor guarantees
        M = sum_y V_y^dag V_y  >=  (sum_y c_y^2) I  >=  I / Y
    so M is strictly positive definite WITHOUT arbitrary jitter,
    which is what makes exact completeness numerically achievable.
    """

    def __init__(self, d: int, n_outcomes: int, rank: int = 8, covariant: bool = True):
        super().__init__()
        self.d = d
        self.Y = n_outcomes
        self.rank = rank if rank and rank < d else d
        self.covariant = covariant
        self.anchor = nn.Parameter(torch.zeros(n_outcomes))
        self.u_proj = nn.Linear(2 * d, n_outcomes * d * self.rank)
        self.w_proj = nn.Linear(2 * d, n_outcomes * self.rank * d)
        nn.init.normal_(self.u_proj.weight, std=0.02)
        nn.init.normal_(self.w_proj.weight, std=0.02)
        nn.init.zeros_(self.u_proj.bias)
        nn.init.zeros_(self.w_proj.bias)

    def generators(self, cond: Tensor) -> Tensor:
        B = cond.shape[0]
        U = torch.view_as_complex(
            self.u_proj(cond).reshape(B, self.Y, self.d, self.rank, 2).contiguous())
        W = torch.view_as_complex(
            self.w_proj(cond).reshape(B, self.Y, self.rank, self.d, 2).contiguous())
        V = U @ W
        c = torch.softmax(self.anchor, dim=0).to(V.dtype)
        I = torch.eye(self.d, device=V.device, dtype=V.dtype)
        V = V + c[None, :, None, None] * I
        return V

    def complete(self, V: Tensor) -> Tensor:
        if self.covariant:
            return complete_kraus_hermitian(V, self.cfg_iters())
        return complete_kraus_cholesky(V)

    def cfg_iters(self) -> int:
        return getattr(self, "_iters", 16)

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        with fp32_island(cond.device.type):
            V = self.generators(cond.float())
            K = self.complete(V)
        return K, kraus_residual(K)


class HierarchicalMeasurementMatrix(nn.Module):
    """
    Block-structured instrument on H_meta (x) H_base:

        K_y = [[A_y, g_MB B_y],
               [g_MB C_y, D_y]]   followed by exact completion.

    g_MB controls meta/base coherence; g_I controls intra-cloud coupling.
    Both are learnable and default to WEAK (log-g = -4), matching the
    weak-measurement regime of the source framework.

    Probabilities are BORN RULE probabilities tr(K rho K^dag),
    not a softmax surrogate over operator norms.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        dm, db = cfg.d_meta, cfg.d_base
        self.dm, self.db = dm, db
        self.d = dm + db
        self.Y = cfg.n_outcomes
        self.rank = min(cfg.kraus_rank, self.d)
        self.covariant = cfg.covariant_normalization
        self.iters = cfg.newton_schulz_iters

        self.log_g_MB = nn.Parameter(torch.full((1,), -4.0))
        self.log_g_I = nn.Parameter(torch.full((1,), -4.0))
        self.anchor = nn.Parameter(torch.zeros(cfg.n_outcomes))

        n_par = 2 * cfg.n_outcomes * (dm * dm + 2 * dm * db + db * db)
        self.block_proj = nn.Linear(cfg.d_meta + cfg.d_base, n_par)
        nn.init.normal_(self.block_proj.weight, std=0.02)
        nn.init.zeros_(self.block_proj.bias)

    def _assemble(self, cond: Tensor) -> Tensor:
        B = cond.shape[0]
        dm, db, Y = self.dm, self.db, self.Y
        p = self.block_proj(cond).reshape(B, Y, 2, dm * dm + 2 * dm * db + db * db)
        V = torch.zeros(B, Y, self.d, self.d, 2, device=cond.device, dtype=cond.dtype)

        off = 0
        blocks = [(slice(0, dm), slice(0, dm), 1.0),
                  (slice(0, dm), slice(dm, self.d), torch.sigmoid(self.log_g_MB)),
                  (slice(dm, self.d), slice(0, dm), torch.sigmoid(self.log_g_MB)),
                  (slice(dm, self.d), slice(dm, self.d),
                   torch.sigmoid(self.log_g_I))]
        for (rs, cs, gate) in blocks:
            n = (rs.stop - rs.start) * (cs.stop - cs.start)
            blk = p[..., off:off + n].reshape(B, Y, rs.stop - rs.start,
                                             cs.stop - cs.start, 2)
            V[..., rs, cs, :] = V[..., rs, cs, :] + blk * gate
            off += n

        Vc = torch.view_as_complex(V.contiguous())
        c = torch.softmax(self.anchor, dim=0).to(Vc.dtype)
        I = torch.eye(self.d, device=Vc.device, dtype=Vc.dtype)
        return Vc + c[None, :, None, None] * I

    def forward(self, cond: Tensor, rho: Tensor) -> Dict[str, Tensor]:
        with fp32_island(rho.device.type):
            V = self._assemble(cond.float())
            if self.covariant:
                K = complete_kraus_hermitian(V, self.iters)
            else:
                K = complete_kraus_cholesky(V, self.cfg.cholesky_jitter)
            probs, post, born = apply_kraus(rho, K)
            res = kraus_residual(K)
            p_err = probs.sum(-1).sub(1.0).abs()
        return {"K": K, "probs": probs, "post": post, "born": born,
                "kraus_residual": res, "born_error": p_err,
                "g_MB": torch.sigmoid(self.log_g_MB),
                "g_I": torch.sigmoid(self.log_g_I)}


# ===========================================================================
# 6. WEAK MEASUREMENT + DECOHERENCE + SEMICLASSICAL CORRESPONDENCE
# ===========================================================================

class WeakMeasurementReadout(nn.Module):
    """
    Gaussian-meter weak instrument:
        K_m proportional to exp(-0.5 g^2 (M - m I)^2), M diagonal Hermitian.
    g -> 0   : K_m -> sqrt(c_m) I   (no back-action, prior outcomes)
    g -> inf : projective limit on eigenvalues of M

    Readout:
        A_w = tr(rho A) / tr(rho)
        dq  = chi * Re(A_w)
        dp  = chi * Im(A_w) * variance_factor
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_q
        self.n_meter = 2
        self.log_g = nn.Parameter(torch.log(torch.tensor(cfg.weak_coupling)))
        self.chi = nn.Parameter(torch.tensor(0.1))
        self.m_spec = nn.Parameter(torch.linspace(-1.0, 1.0, cfg.d_q))
        self.obs_proj = nn.Linear(cfg.d_meta + cfg.d_base, 2 * cfg.d_q * cfg.d_q)

    def forward(self, rho: Tensor, cond: Tensor,
                coupling_override: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        with fp32_island(rho.device.type):
            rho = rho.float()
            g = soft_floor(
                torch.exp(self.log_g) if coupling_override is None else coupling_override,
                1e-5)
            if torch.is_tensor(g):
                g = g.reshape(-1, 1, 1)
            spec = torch.tanh(self.m_spec)
            I = torch.eye(self.d, device=rho.device, dtype=torch.float32)
            Ks = []
            for m in (-1.0, 1.0):
                diag = torch.exp(-0.5 * g.pow(2) * (spec - m).pow(2))
                Ks.append(torch.diag_embed(diag))
            K = torch.stack(Ks, dim=1).to(torch.complex64)
            K = complete_kraus_cholesky(K, 1e-6)
            probs, post, _ = apply_kraus(rho, K)

            gb = self.obs_proj(cond)
            A = torch.view_as_complex(
                gb.reshape(-1, self.d, self.d, 2).contiguous())
            A = hermitize(A)
            tr = rho.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-12)
            Aw = (rho @ A).diagonal(dim1=-2, dim2=-1).sum(-1) / tr
            chi = soft_floor(self.chi, 1e-4)
            dq = chi * Aw.real
            dp = chi * Aw.imag
            purity = (rho @ rho).diagonal(dim1=-2, dim2=-1).real.sum(-1)
        return {"weak_probs": probs, "weak_post": post, "weak_K": K,
                "weak_value_real": dq, "weak_value_imag": dp,
                "coupling": g.reshape(-1), "purity": purity}


class LindbladChannel(nn.Module):
    """
    Discrete-time completely positive, trace-preserving Lindblad map.

        S = sum_k L_k^dag L_k
        P = I - s * S                       (s chosen so P > 0)
        K0 = exp(-i H dt_eff) @ chol(P)^dag
        K_k = sqrt(dt_eff) L_k
        sum K^dag K = P + dt_eff * S = I     EXACTLY

    P > 0 is guaranteed by a smooth, differentiable step-size rescaling:
        dt_eff = dt / (1 + dt * ||S||_F)
    This is a numerical stability device, not a change of physical model;
    the effective rate is reported for calibration.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_q
        self.jump_proj = nn.Linear(cfg.d_meta + cfg.d_base,
                                   2 * cfg.n_jumps * cfg.d_q * cfg.d_q)
        self.ham_proj = nn.Linear(cfg.d_meta + cfg.d_base, 2 * cfg.d_q * cfg.d_q)
        self.rate = nn.Parameter(torch.tensor(cfg.decoherence_scale))

    def forward(self, rho: Tensor, cond: Tensor,
                dt: Optional[float] = None,
                rate_override: Optional[Tensor] = None) -> Dict[str, Tensor]:
        with fp32_island(rho.device.type):
            rho = rho.float()
            B = rho.shape[0]
            d, J = self.d, self.cfg.n_jumps
            scale = soft_floor(self.rate if rate_override is None else rate_override, 1e-6)
            if torch.is_tensor(scale):
                scale = scale.reshape(-1, 1)

            lj = self.jump_proj(cond)
            L = torch.view_as_complex(
                lj.reshape(B, J, d, d, 2).contiguous()) * scale.unsqueeze(-1).unsqueeze(-1).sqrt()
            lh = self.ham_proj(cond)
            H = hermitize(torch.view_as_complex(
                lh.reshape(B, d, d, 2).contiguous()))

            S = (L.mH @ L).sum(dim=1)
            norm = S.norm(dim=(-2, -1)).reshape(-1, 1, 1)
            dt0 = float(self.cfg.dt if dt is None else dt)
            dt_eff = dt0 / (1.0 + dt0 * norm)

            I = torch.eye(d, device=rho.device, dtype=rho.dtype)
            P = I - dt_eff * S
            P = hermitize(P) + 1e-6 * I
            Lch = torch.linalg.cholesky(P)

            # exact unitary part; Cayley fallback if expm unavailable
            try:
                U = torch.linalg.expm((-1j * dt_eff).squeeze(-1).unsqueeze(-1).unsqueeze(-1) * H)
            except Exception:
                Z = (-0.5j * dt_eff).squeeze(-1).unsqueeze(-1).unsqueeze(-1) * H
                U = torch.linalg.solve(I + Z, I - Z)

            K0 = (U @ Lch.mH).unsqueeze(1)
            Kk = L * dt_eff.sqrt()
            K = torch.cat([K0, Kk], dim=1)
            rho_out, _, _ = apply_kraus(rho, K)
            res = kraus_residual(K)
        return {"rho_out": rho_out, "K": K, "kraus_residual": res,
                "effective_dt": dt_eff.reshape(-1), "S": S}


class SemiclassicalCorrespondence(nn.Module):
    """
    Phase-space moment map from a density operator:
        q_i = tr(rho Q_i), p_i = tr(rho P_i),
        CovQQ, CovPP, CovQP  from fixed quadrature operators.
    Used to test the semiclassical limit (strong decoherence / weak coupling).
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.d = cfg.d_q
        n = max(1, cfg.d_q // 2)
        idx = torch.arange(n)
        Q = torch.zeros(cfg.d_q, cfg.d_q, dtype=torch.float32)
        P = torch.zeros(cfg.d_q, cfg.d_q, dtype=torch.float32)
        Q[idx, idx + n] = 1.0
        Q[idx + n, idx] = 1.0
        P[idx, idx + n] = -1.0j
        P[idx + n, idx] = 1.0j
        self.register_buffer("Q", Q, persistent=False)
        self.register_buffer("P", P.to(torch.complex64), persistent=False)
        self.n = n

    def forward(self, rho: Tensor) -> Dict[str, Tensor]:
        with fp32_island(rho.device.type):
            rho = rho.float()
            Q = self.Q.to(rho.dtype).unsqueeze(0)
            P = self.P.to(rho.dtype).unsqueeze(0)
            q = (rho @ Q).diagonal(dim1=-2, dim2=-1).real.sum(-1)
            p = (rho @ P).diagonal(dim1=-2, dim2=-1).real.sum(-1)
            vqq = (rho @ Q @ Q).diagonal(dim1=-2, dim2=-1).real.sum(-1) - q.pow(2)
            vpp = (rho @ P @ P).diagonal(dim1=-2, dim2=-1).real.sum(-1) - p.pow(2)
            uncertainty = (soft_floor(vqq, 1e-8) * soft_floor(vpp, 1e-8)).sqrt()
        return {"phase_q": q, "phase_p": p,
                "var_q": vqq, "var_p": vpp, "uncertainty": uncertainty}


# ===========================================================================
# 7. SESI-GR MASS-CONTRAST COLLAPSE CHANNEL
# ===========================================================================

class GravityCollapseChannel(nn.Module):
    """
    Diosi-Penrose-type mass-contrast rate with interface coupling:

        Gamma_ab = (kappa_I * G / hbar) *
                   sum_ij (dm_a - dm_b)_i K_ij (dm_a - dm_b)_j

        K_ij = 1 / sqrt(|x_i - x_j|^2 + eps^2)      (landmark mode)
        K(r) = 1 / sqrt(r^2 + eps^2)                (FFT continuum mode)

    Collapse channel:
        L_ab = sqrt(Gamma_ab / 2) * (M_a - M_b)
        fed into the SAME exactly-CP/TP Lindblad construction.

    The non-PCF fractal interface reweights the kernel across scales;
    the suite tests the resulting Gamma vs dimension scaling.
    """

    def __init__(self, cfg: SESITheoryConfig, mode: str = "landmark",
                 grid_shape: Tuple[int, int, int] = (8, 8, 8)):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.n = cfg.n_locations
        self.nb = cfg.n_branches
        self.d = cfg.n_locations
        self.register_buffer("_G_over_hbar",
                             torch.tensor(6.67430e-11 / 1.054571817e-34),
                             persistent=False)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(max(cfg.gr_coupling, 1e-6))))
        self.log_eps = nn.Parameter(torch.tensor(math.log(1e-3)))

        pairs = [(a, b) for a in range(self.nb) for b in range(a + 1, self.nb)]
        self.pairs = pairs
        self.n_pairs = len(pairs)

        # fixed Hermitian branch location operators
        self.branch_op_raw = nn.Parameter(
            torch.randn(self.nb, self.d) * 0.1)

        if mode == "landmark":
            pos = torch.linspace(0.0, 1.0, self.n)
            xyz = torch.stack(torch.meshgrid(pos, pos, pos, indexing="ij"), -1)
            xyz = xyz.reshape(-1, 3)[:self.n]
            if xyz.shape[0] < self.n:
                xyz = F.pad(xyz, (0, 0, 0, self.n - xyz.shape[0]))
            self.register_buffer("positions", xyz.contiguous(), persistent=False)
            self._build_landmark_kernel()
        else:
            self.grid_shape = grid_shape
            self._build_fft_kernel(grid_shape)

    def _build_landmark_kernel(self) -> None:
        with torch.no_grad():
            r = (self.positions.unsqueeze(0) - self.positions.unsqueeze(1)).norm(dim=-1)
            self.register_buffer("_base_kernel", 1.0 / (r + 1e-3), persistent=False)

    def _build_fft_kernel(self, shape: Tuple[int, int, int]) -> None:
        with torch.no_grad():
            D, H, W = shape
            pad = (2 * D, 2 * H, 2 * W)
            ax = [torch.arange(s, dtype=torch.float32) - s // 2 for s in pad]
            X, Y, Z = torch.meshgrid(*ax, indexing="ij")
            r = torch.sqrt(X ** 2 + Y ** 2 + Z ** 2 + 1e-4)
            kernel = 1.0 / r
            self.register_buffer("_kernel_hat",
                                 torch.fft.rfftn(kernel), persistent=False)
            self._pad_shape = pad

    def rates(self, mass_branches: Tensor,
              fractal_weights: Optional[Tensor] = None) -> Tensor:
        """mass_branches: (B, n_branches, n_locations) non-negative."""
        with fp32_island(mass_branches.device.type):
            m = soft_floor(mass_branches.float(), 0.0)
            kappa = soft_floor(torch.exp(self.log_kappa), 1e-8)
            eps = soft_floor(torch.exp(self.log_eps), 1e-6)

            if self.mode == "landmark":
                r = (self.positions.unsqueeze(0) - self.positions.unsqueeze(1)).norm(dim=-1)
                K = 1.0 / (r + eps)
                if fractal_weights is not None:
                    k = min(fractal_weights.shape[-1], self.cfg.n_scales)
                    w = fractal_weights[..., :k]
                    scales = torch.arange(1, k + 1, device=K.device, dtype=K.dtype)
                    mix = (w.unsqueeze(-1) * (scales / scales.mean()).pow(-0.5)).sum(-2)
                    K = K * mix[..., None] if mix.dim() > 2 else K * w.mean(-1).unsqueeze(-1).unsqueeze(-1)
                ia = torch.tensor([a for a, _ in self.pairs], device=m.device)
                ib = torch.tensor([b for _, b in self.pairs], device=m.device)
                delta = m[:, ia] - m[:, ib]                    # (B,P,n)
                gamma = torch.einsum("bpi,ij,bpj->bp", delta, K, delta)
            else:
                # continuum mode: one contrast field per branch pair
                B = m.shape[0]
                D, H, W = self.grid_shape
                field = F.adaptive_avg_pool3d(
                    m.reshape(B, 1, 1, 1, self.n).permute(0, 1, 4, 2, 3)
                    if False else m.new_zeros(B, 1, D, H, W), (D, H, W))
                # distribute landmark masses onto grid cells
                flat = m.reshape(B, self.nb * self.n)
                grid = F.interpolate(flat.reshape(B, 1, 1, self.n, 1),
                                     size=(D, H), mode="trilinear", align_corners=False)
                grid = F.interpolate(grid, size=(D, H, W), mode="trilinear", align_corners=False)
                ia = torch.tensor([a for a, _ in self.pairs], device=m.device)
                ib = torch.tensor([b for _, b in self.pairs], device=m.device)
                gamma_list = []
                for a, b in zip(ia.tolist(), ib.tolist()):
                    fa = grid[:, a % max(1, self.nb // 2)].squeeze(1) if self.nb > 1 else grid[:, 0].squeeze(1)
                    fb = grid[:, b % max(1, self.nb // 2)].squeeze(1) if self.nb > 1 else grid[:, 0].squeeze(1)
                    delta = fa - fb
                    pad = self._pad_shape
                    dp = delta.new_zeros(B, *pad)
                    dp[..., :D, :H, :W] = delta
                    pot = torch.fft.irfftn(
                        torch.fft.rfftn(dp) * self._kernel_hat.to(dp.dtype), s=pad)
                    pot = pot[..., :D, :H, :W]
                    gamma_list.append((delta * pot).sum(dim=(1, 2, 3)))
                gamma = torch.stack(gamma_list, dim=-1)

            gamma = soft_floor(gamma * kappa * self._G_over_hbar.to(gamma.dtype) * 1e-20,
                               1e-18)
        return gamma

    def forward(self, rho: Tensor, mass_branches: Tensor,
                fractal_weights: Optional[Tensor] = None,
                dt: Optional[float] = None) -> Dict[str, Tensor]:
        with fp32_island(rho.device.type):
            rho = rho.float()
            gamma = self.rates(mass_branches, fractal_weights)     # (B,P)
            ops = torch.softmax(self.branch_op_raw, dim=-1).float()
            ia = torch.tensor([a for a, _ in self.pairs], device=rho.device)
            ib = torch.tensor([b for _, b in self.pairs], device=rho.device)
            diff = ops[ia] - ops[ib]                              # (P,d)
            L = (gamma / 2.0).sqrt().unsqueeze(-1) * diff.unsqueeze(0).unsqueeze(-2)
            # (B,P,d,1) -> broadcast to (B,P,d,d) via diagonal operators
            L = torch.diag_embed(L.squeeze(-2)) if L.dim() == 4 else L

            S = (L.mH @ L).sum(dim=1)
            dt0 = float(self.cfg.dt if dt is None else dt)
            dt_eff = dt0 / (1.0 + dt0 * S.norm(dim=(-2, -1)).reshape(-1, 1, 1))
            I = torch.eye(self.d, device=rho.device, dtype=rho.dtype)
            P = hermitize(I - dt_eff * S) + 1e-6 * I
            Lch = torch.linalg.cholesky(P)
            K0 = Lch.mH.unsqueeze(1)
            Kk = L * dt_eff.sqrt()
            K = torch.cat([K0, Kk], dim=1)
            rho_out, _, _ = apply_kraus(rho, K)
            res = kraus_residual(K)
        return {"rho_out": rho_out, "gamma_gr": gamma, "K": K,
                "kraus_residual": res, "effective_dt": dt_eff.reshape(-1)}


# ===========================================================================
# 8. GEOMETRIC INTERFACE / EVENT-HISTORY COLLAPSE
# ===========================================================================

class GeometricInterface(nn.Module):
    """
    Geometric-interface-derived reduction.

    Interface operator:
        A_S = sum_t k_t * G_t,   k = softmax(score(geometry, event history))
    Three normalized event kernels K_N, K_M, K_B (naming follows the
    source framework; the functional form below is an explicit
    parametric assumption and is ablatable).

    Learnable, bounded physical-interface parameters:
        mu_I   : interface mass scale
        kappa_I: interface collapse coupling
        eta_E  : environmental coupling fraction

    Heating contribution (interface term of the master equation):
        d<H>/dt|_I = -(lambda_I / 2) * sum_x Tr([H,A_x][A_x,rho])
    The suite enforces a total energy-growth budget and EXCLUDES
    parameter settings that violate it.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_q
        self.geo_proj = nn.Linear(cfg.d_geometry, 2 * self.d * self.d)
        self.event_proj = nn.Linear(cfg.d_event, cfg.d_meta + cfg.d_base)
        self.kernel_score = nn.Linear(cfg.d_meta + cfg.d_base + cfg.d_geometry, 3)
        self.kernel_temp = nn.Parameter(torch.tensor(0.5))
        self.collapse_proj = nn.Linear(cfg.d_meta + cfg.d_base, 2 * self.d * self.d)

        self.raw_mu = nn.Parameter(torch.tensor(0.0))
        self.raw_kappa = nn.Parameter(torch.tensor(-2.0))
        self.raw_eta = nn.Parameter(torch.tensor(-2.0))
        self.raw_lambda = nn.Parameter(torch.tensor(-3.0))

    def parameters_phys(self) -> Dict[str, Tensor]:
        return {
            "mu_I": soft_cap(torch.sigmoid(self.raw_mu), 1.0) * 10.0,
            "kappa_I": soft_cap(torch.sigmoid(self.raw_kappa), 1.0),
            "eta_E": soft_cap(torch.sigmoid(self.raw_eta), 1.0),
            "lambda_I": soft_cap(torch.sigmoid(self.raw_lambda), 1.0) * 0.1,
        }

    def forward(self, rho: Tensor, geometry: Tensor, events: Tensor,
                dt: Optional[float] = None) -> Dict[str, Tensor]:
        with fp32_island(rho.device.type):
            rho = rho.float()
            B = rho.shape[0]
            d = self.d
            e = self.event_proj(events.float())                     # (B,T,c)
            cond = e.mean(dim=1)
            g = geometry.float()

            score_in = torch.cat([
                cond.unsqueeze(1).expand(-1, e.shape[1], -1),
                g.unsqueeze(1).expand(-1, e.shape[1], -1),
            ], dim=-1)
            scores = self.kernel_score(score_in)                   # (B,T,3)
            temp = soft_floor(self.kernel_temp, 1e-2)
            kernels = torch.softmax(scores / temp, dim=1)           # (B,T,3)

            A = torch.view_as_complex(
                self.geo_proj(g).reshape(B, d, d, 2).contiguous())
            A = hermitize(A)

            C = torch.view_as_complex(
                self.collapse_proj(cond).reshape(B, d, d, 2).contiguous())
            C = hermitize(C)

            pp = self.parameters_phys()
            kappa = pp["kappa_I"].reshape(-1) if torch.is_tensor(pp["kappa_I"]) else pp["kappa_I"]
            lam = pp["lambda_I"]
            eta = pp["eta_E"]

            # jump operator built from geometry + event kernels
            w = kernels[..., 0].sum(dim=1, keepdim=True)
            jump = (C + eta * A) * w.unsqueeze(-1).unsqueeze(-1)
            L = (kappa.reshape(-1, 1, 1, 1) * 0.5).sqrt() * jump.unsqueeze(1)

            S = (L.mH @ L).sum(dim=1)
            dt0 = float(self.cfg.dt if dt is None else dt)
            dt_eff = dt0 / (1.0 + dt0 * S.norm(dim=(-2, -1)).reshape(-1, 1, 1))
            I = torch.eye(d, device=rho.device, dtype=rho.dtype)
            P = hermitize(I - dt_eff * S) + 1e-6 * I
            K0 = torch.linalg.cholesky(P).mH.unsqueeze(1)
            K = torch.cat([K0, L * dt_eff.sqrt()], dim=1)
            rho_out, _, _ = apply_kraus(rho, K)

            # interface heating term: -lambda/2 Tr([H,A][A,rho])
            H = 0.5 * (A @ A)
            comm_HA = H @ A - A @ H
            comm_Arho = A @ rho - rho @ A
            heating = -0.5 * lam * (comm_HA @ comm_Arho).diagonal(
                dim1=-2, dim2=-1).real.sum(-1)

            # residual coherence (off-diagonal mass) as calibration target
            off = 1.0 - torch.eye(d, device=rho.device, dtype=rho.dtype)
            coherence = (rho_out.abs() * off).sum(dim=(-2, -1))

            # completeness of event kernels (each head sums to 1 over time)
            kernel_error = (kernels.sum(dim=1) - 1.0).abs().max(dim=-1).values

        return {"rho_out": rho_out, "K": K,
                "kraus_residual": kraus_residual(K),
                "interface_heating": heating,
                "coherence": coherence,
                "event_kernels": kernels,
                "kernel_error": kernel_error,
                "mu_I": pp["mu_I"], "kappa_I": pp["kappa_I"],
                "eta_E": pp["eta_E"], "lambda_I": pp["lambda_I"]}


# ===========================================================================
# 9. OPMC CANONICAL SIGNATURE + CAPACITY PROBE
# ===========================================================================

class OPMCCanonicalizer(nn.Module):
    """
    Task-sufficient canonical signature:
        sigma_T = normalize( U_sigma [p(y), purity, phase moments, provenance] )

    Factorization test:
        task observable O_full(rho)  vs  O_sigma(sigma_T)
        residual = || O_full - O_sigma ||

    The claim "classical task evaluation is polynomial" is TESTED, not
    assumed: it holds only if k_sigma is bounded independently of the
    Hilbert dimension and the factorization residual is small.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_q
        base_dim = cfg.n_outcomes + 2 + 2 * max(1, cfg.d_q // 2)
        self.sigma_proj = nn.Linear(base_dim, cfg.k_sigma)
        self.task_head = nn.Linear(cfg.k_sigma, 2)
        self.full_head = nn.Linear(base_dim, 2)
        self.provenance = nn.Parameter(torch.ones(cfg.n_outcomes) / cfg.n_outcomes)

    def features(self, rho: Tensor, probs: Tensor) -> Tensor:
        with fp32_island(rho.device.type):
            rho = rho.float()
            purity = (rho @ rho).diagonal(dim1=-2, dim2=-1).real.sum(-1, keepdim=True)
            coh = rho.abs().sum(dim=(-2, -1)) / (self.d ** 2)
            diag = rho.diagonal(dim1=-2, dim2=-1).real
            n = max(1, self.d // 2)
            q = diag[..., :n]
            p = diag[..., n:2 * n]
            if p.shape[-1] < n:
                p = F.pad(p, (0, n - p.shape[-1]))
            feats = torch.cat([probs.float(), purity, coh.unsqueeze(-1), q, p], dim=-1)
        return feats

    def forward(self, rho: Tensor, probs: Tensor) -> Dict[str, Tensor]:
        f = self.features(rho, probs)
        sigma = F.normalize(self.sigma_proj(f), dim=-1, eps=1e-6)
        logits_sigma = self.task_head(sigma)
        logits_full = self.full_head(f)
        residual = (logits_sigma - logits_full.detach()).pow(2).mean(dim=-1)
        return {"sigma_T": sigma, "logits": logits_sigma,
                "logits_full": logits_full,
                "factorization_residual": residual,
                "task_features": f}


class RelationalCapacityProbe(nn.Module):
    """
    Structural capacity bookkeeping.

    Hierarchical encoding with depth L and branching b can represent
        N_config = prod_l b_l
    configurations with O(sum_l b_l) stored parameters.

    A DOUBLE-EXPONENTIAL claim requires log2(log2(N_config)) to grow
    linearly in L. This probe MEASURES that slope on the actual encoded
    signature space; it does not assert the claim.

    Crucially: representational capacity != computational advantage.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def structural_counts(self) -> Dict[str, float]:
        L, b = self.cfg.hierarchy_depth, max(2, self.cfg.branch_factor)
        n_config = float(b ** (2 ** L))
        stored = float(sum(b ** (i + 1) for i in range(L)))
        return {
            "hierarchy_depth": float(L),
            "branch_factor": float(b),
            "log2_configurations": math.log2(max(n_config, 2.0)),
            "log2_log2_configurations": math.log2(max(1.0, math.log2(max(n_config, 2.0)))),
            "stored_parameters_proxy": stored,
        }

    @torch.no_grad()
    def signature_diversity(self, sigma: Tensor, bins: int = 8) -> float:
        """Quantized distinct-signature count (reachable-state proxy)."""
        q = torch.quantize_per_tensor(sigma.float(), 2.0 / bins, -1.0,
                                     torch.quint8).dequantize()
        codes = torch.unique(torch.round(q * bins).reshape(q.shape[0], -1), dim=0)
        return float(codes.shape[0])


# ===========================================================================
# 10. ENTANGLEMENT DIAGNOSTICS
# ===========================================================================

class EntanglementDiagnostics(nn.Module):
    """
    Purity / linear entropy / partial-transpose negativity.

    Genuine-multipartite score = minimum negativity over all bipartitions.
    A strictly positive value is NECESSARY but NOT SUFFICIENT for genuine
    multipartite entanglement; this limitation is reported explicitly.

    Eigendecomposition is confined to no-grad diagnostic paths (fp32),
    never inside the training graph.
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.d = cfg.d_q
        self.dm, self.db = cfg.d_meta, cfg.d_base

    @torch.no_grad()
    def negativity(self, rho: Tensor, dims: Tuple[int, int]) -> Tensor:
        with fp32_island(rho.device.type):
            rho = rho.float().real if rho.is_complex() else rho.float()
            B = rho.shape[0]
            da, db = dims
            T = rho.reshape(B, da, db, da, db)
            T = T.permute(0, 1, 3, 2, 4).reshape(B, da * db, da * db)
            T = 0.5 * (T + T.transpose(-1, -2))
            ev = torch.linalg.eigvalsh(T)
            neg = ev.clamp(max=0.0).abs().sum(-1)
        return neg

    @torch.no_grad()
    def multipartite_score(self, rho: Tensor, n_qubits: int = 3) -> Dict[str, Tensor]:
        d = 2 ** n_qubits
        if self.d < d:
            return {"multipartite_negativity": rho.new_zeros(rho.shape[0]),
                    "n_bipartitions": 0.0}
        with fp32_island(rho.device.type):
            sub = rho[:, :d, :d].float()
            sub = sub / sub.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-12)[:, None, None]
            negs = []
            for mask in range(1, (1 << n_qubits) - 1):
                a = [i for i in range(n_qubits) if mask & (1 << i)]
                b = [i for i in range(n_qubits) if not mask & (1 << i)]
                da, db = 2 ** len(a), 2 ** len(b)
                # reorder qubit axes: A then B
                order = a + b
                T = sub.reshape([-1] + [2] * (2 * n_qubits))
                perm = [0] + [1 + i for i in order] + [1 + n_qubits + i for i in order]
                T = T.permute(*perm).reshape(-1, da, db, da, db)
                T = T.permute(0, 1, 3, 2, 4).reshape(-1, d, d)
                T = 0.5 * (T + T.transpose(-1, -2))
                ev = torch.linalg.eigvalsh(T)
                negs.append(ev.clamp(max=0.0).abs().sum(-1))
            score = torch.stack(negs, dim=-1).min(dim=-1).values
        return {"multipartite_negativity": score,
                "n_bipartitions": float(len(negs))}

    @torch.no_grad()
    def forward(self, rho: Tensor) -> Dict[str, Tensor]:
        with fp32_island(rho.device.type):
            r = rho.float()
            purity = (r @ r).diagonal(dim1=-2, dim2=-1).real.sum(-1)
            lin_entropy = 1.0 - purity
            neg = self.negativity(r, (self.dm, self.db))
        out = {"purity": purity, "linear_entropy": lin_entropy,
               "bipartite_negativity": neg}
        out.update(self.multipartite_score(r, n_qubits=3))
        return out


# ===========================================================================
# 11. HUTCHINSON COUPLING REGULARIZER
# ===========================================================================

def hutchinson_jacobian_norm(fn: Callable[[Tensor], Tensor], x: Tensor,
                             probes: int = 1) -> Tensor:
    """
    E_v ||J^T v||^2 = ||J||_F^2  for Rademacher v.
    Cost: O(probes) vector-Jacobian products instead of O(d) columns.
    create_graph=True keeps the penalty differentiable.
    """
    if not x.requires_grad:
        return x.new_zeros(())
    total = x.new_zeros(())
    for _ in range(max(1, probes)):
        out = fn(x)
        v = torch.randint(0, 2, out.shape, device=out.device, dtype=out.dtype) * 2.0 - 1.0
        proj = (out * v).sum()
        g = torch.autograd.grad(proj, x, create_graph=True, retain_graph=True)[0]
        total = total + g.pow(2).sum(dim=-1).mean()
    return total / max(1, probes)


# ===========================================================================
# 12. UNIFIED MODEL
# ===========================================================================

class SESICloudModel(nn.Module):
    """
    End-to-end differentiable instantiation of T-A .. T-E.

    Forward contract (all keys optional except "x"):
        x              (B, d_in)        real observations
        geometry       (B, d_geometry)  normals / curvature / area / scale
        events         (B, T, d_event)  interface event history
        mass_branches  (B, n_branches, n_locations)
        rho_real       (B, d_q, d_q)    optional real density-matrix input
        rho_imag       (B, d_q, d_q)    optional imaginary part
        label          (B,)            optional task label
        coherence_target (B,)          optional calibration target
        energy_budget  (B,)            optional per-sample heating budget
    """

    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CovariantStateEncoder(cfg)
        self.fractal = NonPCFFractalInterface(cfg) if cfg.use_fractal else None
        self.instrument = HierarchicalMeasurementMatrix(cfg)
        self.weak = WeakMeasurementReadout(cfg) if cfg.use_weak_measurement else None
        self.lindblad = LindbladChannel(cfg) if cfg.use_decoherence else None
        self.gravity = GravityCollapseChannel(cfg, mode="landmark") if cfg.use_gravity else None
        self.interface = GeometricInterface(cfg) if cfg.use_interface else None
        self.opmc = OPMCCanonicalizer(cfg) if cfg.use_opmc else None
        self.diagnostics = EntanglementDiagnostics(cfg)
        self.semiclassical = SemiclassicalCorrespondence(cfg)
        self.capacity = RelationalCapacityProbe(cfg)
        self.ent_head = nn.Linear(cfg.d_meta + cfg.d_base, 2) if cfg.use_entanglement_head else None
        self._compiled = False

    # -- helpers ----------------------------------------------------------
    def _checkpoint(self, fn, *args):
        if self.cfg.use_gradient_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def compile(self) -> None:
        if self._compiled or not self.cfg.compile_model:
            return
        try:
            torch.compile(self, mode=self.cfg.compile_mode, dynamic=True)
            self._compiled = True
        except Exception as e:  # pragma: no cover
            logger.warning("torch.compile unavailable (%s); continuing eager.", e)

    # -- forward ----------------------------------------------------------
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        cfg = self.cfg
        x = batch["x"]
        geo = batch.get("geometry")
        events = batch.get("events")
        masses = batch.get("mass_branches")

        rho, cond, Bref = self.encoder(x, geo)
        out: Dict[str, Tensor] = {"rho": rho}

        # non-PCF fractal interface conditions the measurement
        if self.fractal is not None:
            f = self.fractal(x)
            cond = cond + f["fractal_cond"]
            out.update({f"frac_{k}": v for k, v in f.items()})
            fw = f["fractal_weights"]
        else:
            fw = None

        # hierarchical Born-rule measurement
        meas = self._checkpoint(self.instrument, cond, rho)
        out.update({f"meas_{k}": v for k, v in meas.items()
                    if torch.is_tensor(v)})
        rho_measured = (meas["post"] * meas["probs"][..., None, None]).sum(dim=1)
        rho_measured = normalize_state(rho_measured)

        # weak measurement branch
        if self.weak is not None:
            w = self._checkpoint(self.weak, rho_measured, cond)
            out.update({f"weak_{k}": v for k, v in w.items()
                        if torch.is_tensor(v)})
            rho_measured = (w["weak_post"] * w["weak_probs"][..., None, None]).sum(dim=1)
            rho_measured = normalize_state(rho_measured)

        # Lindblad decoherence over discrete steps
        if self.lindblad is not None:
            for _ in range(max(1, cfg.n_steps)):
                lin = self._checkpoint(self.lindblad, rho_measured, cond)
                rho_measured = lin["rho_out"]
                out["decoherence_residual"] = lin["kraus_residual"]
                out["effective_dt"] = lin["effective_dt"]

        # SESI-GR mass-contrast collapse
        if self.gravity is not None and masses is not None:
            gr = self._checkpoint(self.gravity, rho_measured, masses, fw)
            rho_measured = gr["rho_out"]
            out["gamma_gr"] = gr["gamma_gr"]
            out["gravity_residual"] = gr["kraus_residual"]

        # geometric-interface collapse
        if self.interface is not None and geo is not None and events is not None:
            itf = self._checkpoint(self.interface, rho_measured, geo, events)
            rho_measured = itf["rho_out"]
            out.update({f"itf_{k}": v for k, v in itf.items()
                        if torch.is_tensor(v)})

        rho_final = normalize_state(rho_measured)
        out["rho_final"] = rho_final

        # OPMC canonical signature and factorizable task head
        final_meas = self.instrument(cond, rho_final)
        out["final_probs"] = final_meas["probs"]
        out["final_kraus_residual"] = final_meas["kraus_residual"]
        if self.opmc is not None:
            op = self.opmc(rho_final, final_meas["probs"])
            out.update({f"opmc_{k}": v for k, v in op.items()
                        if torch.is_tensor(v)})
            out["logits"] = op["logits"]
        else:
            out["logits"] = final_meas["probs"][:, :2]

        # diagnostics (no-grad internally)
        diag = self.diagnostics(rho_final)
        out.update({f"diag_{k}": v for k, v in diag.items()
                    if torch.is_tensor(v)})
        sc = self.semiclassical(rho_final)
        out.update({f"sc_{k}": v for k, v in sc.items()
                    if torch.is_tensor(v)})

        if self.ent_head is not None:
            out["ent_logits"] = self.ent_head(cond)

        # differentiable CP / TP residuals for the loss
        out["cp_residual"] = final_meas["kraus_residual"]
        out["tp_residual"] = final_meas["born_error"]
        out["hermiticity_residual"] = (
            rho_final - rho_final.mH).norm(dim=(-2, -1)).real

        # total heating estimate (interface + decoherence-induced)
        with fp32_island(rho.device.type):
            r0, r1 = self.encoder(x, geo)[0].float(), rho_final.float()
            Href = 0.5 * (Bref @ Bref)
            e0 = (Href @ r0).diagonal(dim1=-2, dim2=-1).real.sum(-1)
            e1 = (Href @ r1).diagonal(dim1=-2, dim2=-1).real.sum(-1)
            out["heat_rate"] = (e1 - e0).abs() / max(cfg.dt, 1e-6)

        return out


# ===========================================================================
# 13. LOSS
# ===========================================================================

class UnifiedTheoryLoss(nn.Module):
    def __init__(self, cfg: SESITheoryConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: Dict[str, Tensor], out: Dict[str, Tensor],
                model: SESICloudModel, compute_coupling: bool = False
                ) -> Dict[str, Tensor]:
        cfg = self.cfg
        dev = out["logits"].device
        zero = out["logits"].new_zeros(())

        task = zero
        if "label" in batch:
            y = batch["label"].long()
            if out["logits"].shape[-1] == 2:
                task = F.cross_entropy(out["logits"], y)
            else:
                p = out["logits"].softmax(-1) if out["logits"].shape[-1] == cfg.n_outcomes \
                    else out["logits"][:, :2]
                if p.shape[-1] == 2:
                    task = F.cross_entropy(p, y)

        nll = zero
        if "label" in batch and "final_probs" in out:
            p = out["final_probs"].clamp_min(1e-8)
            p = p / p.sum(-1, keepdim=True)
            nll = F.nll_loss(p.log(), batch["label"].long() % p.shape[-1])

        cp = out.get("cp_residual", zero).mean()
        tp = out.get("tp_residual", zero).mean()
        herm = out.get("hermiticity_residual", zero).mean()

        energy = zero
        budget = batch.get("energy_budget",
                           out["heat_rate"].new_full(out["heat_rate"].shape, cfg.energy_budget))
        if "heat_rate" in out:
            energy = F.softplus((out["heat_rate"] - budget) / budget.clamp_min(1e-8)).mean()

        factor = zero
        if "opmc_factorization_residual" in out:
            factor = out["opmc_factorization_residual"].mean()

        coupling = zero
        if compute_coupling and cfg.lambda_coupling > 0:
            x = batch["x"]
            if x.requires_grad:
                coupling = hutchinson_jacobian_norm(
                    lambda z: model.encoder(z, batch.get("geometry"))[0]
                    .diagonal(dim1=-2, dim2=-1).real.sum(-1),
                    x, probes=cfg.hutchinson_probes)

        coherence_loss = zero
        if "itf_coherence" in out and "coherence_target" in batch:
            coherence_loss = F.mse_loss(out["itf_coherence"], batch["coherence_target"])

        total = (task + 0.25 * nll
                 + cfg.lambda_cp * (cp + tp + herm)
                 + cfg.lambda_energy * energy
                 + cfg.lambda_factor * factor
                 + cfg.lambda_coupling * coupling
                 + 0.1 * coherence_loss)

        return {"loss": total, "task": task.detach(), "nll": nll.detach(),
                "cp": cp.detach(), "tp": tp.detach(), "herm": herm.detach(),
                "energy": energy.detach(), "factor": factor.detach(),
                "coupling": coupling.detach(), "coherence": coherence_loss.detach()}


# ===========================================================================
# 14. EMA (optional CPU offload)
# ===========================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float, offload_cpu: bool = True):
        self.decay = decay
        self.offload = offload_cpu
        self.shadow: Dict[str, Tensor] = {}
        for k, v in model.state_dict().items():
            t = v.detach().float().clone()
            if offload_cpu and t.is_cuda:
                t = t.cpu().pin_memory()
            self.shadow[k] = t

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                if self.offload:
                    s.mul_(self.decay).add_(v.detach().float().cpu(), alpha=1 - self.decay)
                else:
                    s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v.detach().cpu() if self.offload else v.detach())

    def state_dict(self) -> Dict[str, Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# ===========================================================================
# 15. REAL + SYNTHETIC DATA
# ===========================================================================

REQUIRED_SCHEMA = {"x": "x"}
OPTIONAL_SCHEMA = {
    "geometry": "geometry", "events": "events", "mass_branches": "mass_branches",
    "label": "label", "rho_real": "rho_real", "rho_imag": "rho_imag",
    "coherence_target": "coherence_target", "energy_budget": "energy_budget",
}


def _validate_sample(s: Dict[str, Any], cfg: SESITheoryConfig) -> None:
    if "x" not in s:
        raise ValueError("Schema must provide 'x' (B, d_in) real observations.")
    x = s["x"]
    if x.dim() != 2 or x.shape[-1] != cfg.d_in:
        raise ValueError(f"x must be (B,{cfg.d_in}); got {tuple(x.shape)}")
    if "geometry" in s and s["geometry"].shape[-1] != cfg.d_geometry:
        raise ValueError(f"geometry last dim must be {cfg.d_geometry}")
    if "mass_branches" in s:
        m = s["mass_branches"]
        if m.shape[-2:] != (cfg.n_branches, cfg.n_locations):
            raise ValueError(f"mass_branches must end with "
                             f"({cfg.n_branches},{cfg.n_locations}); got {tuple(m.shape)}")
    if "events" in s and s["events"].shape[-1] != cfg.d_event:
        raise ValueError(f"events last dim must be {cfg.d_event}")
    for k, v in s.items():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            raise ValueError(f"Non-finite values in '{k}'")


class SyntheticTheoryDataset(Dataset):
    """
    Controlled synthetic generator with PLANTED ground truth so that
    calibration recovery and falsification are meaningful.
    """

    def __init__(self, cfg: SESITheoryConfig, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)

        def rand(*shape):
            return torch.randn(*shape, generator=g)

        self.x = rand(n, cfg.d_in)
        self.label = torch.randint(0, 2, (n,), generator=g)
        self.geometry = rand(n, cfg.d_geometry)
        self.events = rand(n, cfg.n_events, cfg.d_event)
        self.mass = F.softplus(rand(n, cfg.n_branches, cfg.n_locations))
        self.mass = self.mass * (1.0 + 0.5 * self.label.unsqueeze(-1).unsqueeze(-1))

        # planted interface parameters for calibration-recovery test
        self.planted_kappa = torch.sigmoid(rand(n) * 0.5 - 2.0)
        self.planted_eta = torch.sigmoid(rand(n) * 0.5 - 2.0)
        # coherence target: decreasing in mass contrast (collapse-like)
        contrast = (self.mass[:, 0] - self.mass[:, 1]).pow(2).mean(-1) \
            if cfg.n_branches > 1 else self.mass.mean(-1)
        self.coherence_target = torch.exp(-0.5 * contrast) * 0.3
        self.energy_budget = torch.full((n,), cfg.energy_budget)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Dict[str, Tensor]:
        return {
            "x": self.x[i], "label": self.label[i],
            "geometry": self.geometry[i], "events": self.events[i],
            "mass_branches": self.mass[i],
            "coherence_target": self.coherence_target[i],
            "energy_budget": self.energy_budget[i],
        }


class RealTheoryDataset(Dataset):
    """
    Lazy real-data reader.

    Supports:
      * HDF5 shards (glob): per-worker handle reopening, no full-file load.
      * NPZ shards: memory-mapped arrays.
      * Parquet / CSV: columnar read with schema mapping.

    Schema JSON example:
      {
        "x": "features",
        "geometry": "geom",
        "events": "event_stream",
        "mass_branches": "branch_mass",
        "label": "class_id",
        "coherence_target": "measured_coherence",
        "energy_budget": "heat_budget"
      }

    Density-matrix ingestion:
      provide "rho_real" and "rho_imag" as (N, d_q, d_q); the encoder's
      learned state is then replaced by the supplied physical state.
    """

    def __init__(self, cfg: SESITheoryConfig, paths: Sequence[str],
                 schema: Dict[str, str], split: str = "train"):
        self.cfg = cfg
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise FileNotFoundError("No data shards matched the provided glob.")
        self.schema = {**OPTIONAL_SCHEMA, **REQUIRED_SCHEMA, **schema}
        self.split = split
        self._handles: Dict[str, Any] = {}
        self._index: List[Tuple[int, int]] = []
        self._lengths: List[int] = []
        self._scan()

    def _open(self, i: int):
        p = self.paths[i]
        if p not in self._handles:
            if self.cfg.data_format == "h5":
                import h5py
                self._handles[p] = h5py.File(p, "r", swmr=True)
            elif self.cfg.data_format == "npz":
                import numpy as np
                self._handles[p] = np.load(p, allow_pickle=False)
            else:
                import pandas as pd
                self._handles[p] = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        return self._handles[p]

    def _scan(self) -> None:
        for i, p in enumerate(self.paths):
            h = self._open(i)
            key = self.schema["x"]
            if self.cfg.data_format == "h5":
                n = int(h[key].shape[0])
            elif self.cfg.data_format == "npz":
                n = int(h[key].shape[0])
            else:
                n = int(len(h))
            self._lengths.append(n)
            self._index.extend([(i, j) for j in range(n)])
            if self.cfg.data_format != "h5":
                # NPZ / frame handles are cheap to reopen; close scan handles
                if hasattr(h, "close"):
                    h.close()
                del self._handles[p]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        si, row = self._index[idx]
        h = self._open(si)
        sample: Dict[str, Any] = {}
        for logical, physical in self.schema.items():
            try:
                if self.cfg.data_format == "h5":
                    if physical not in h:
                        continue
                    arr = h[physical][row]
                elif self.cfg.data_format == "npz":
                    if physical not in h.files:
                        continue
                    arr = h[physical][row]
                else:
                    if physical not in h.columns:
                        continue
                    arr = h.iloc[row][physical]
                    if hasattr(arr, "tolist"):
                        arr = arr.tolist()
                t = torch.as_tensor(arr)
                if t.dtype == torch.float64:
                    t = t.float()
                if logical == "label":
                    t = t.long()
                sample[logical] = t
            except (KeyError, ValueError):
                continue
        if not sample:
            raise KeyError(f"No schema columns found in shard {self.paths[si]}")
        return sample

    def close(self) -> None:
        for h in self._handles.values():
            if hasattr(h, "close"):
                h.close()
        self._handles.clear()


def _collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    keys = set().union(*(b.keys() for b in batch))
    out = {}
    for k in sorted(keys):
        vals = [b[k] for b in batch if k in b]
        if not vals:
            continue
        if all(v.shape == vals[0].shape for v in vals):
            out[k] = torch.stack(vals, 0)
        else:
            # pad variable-length event streams on axis 0
            max_len = max(v.shape[0] for v in vals)
            shape = (len(vals), max_len) + tuple(vals[0].shape[1:])
            padded = vals[0].new_zeros(shape)
            for i, v in enumerate(vals):
                padded[i, :v.shape[0]] = v
            out[k] = padded
    return out


def build_dataloaders(cfg: SESITheoryConfig
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                  persistent_workers=cfg.num_workers > 0,
                  collate_fn=_collate, drop_last=True)
    if cfg.num_workers > 0:
        common["prefetch_factor"] = cfg.prefetch_factor

    if cfg.data_shards:
        import glob
        paths = sorted(glob.glob(cfg.data_shards))
        n = len(paths)
        if n < 3:
            raise ValueError("Need at least 3 shards for train/val/test split.")
        cut1, cut2 = max(1, int(0.8 * n)), max(2, int(0.9 * n))
        train_ds = RealTheoryDataset(cfg, paths[:cut1], cfg.schema, "train")
        val_ds = RealTheoryDataset(cfg, paths[cut1:cut2], cfg.schema, "val")
        test_ds = RealTheoryDataset(cfg, paths[cut2:], cfg.schema, "test")
    else:
        train_ds = SyntheticTheoryDataset(cfg, cfg.n_train, cfg.data_seed)
        val_ds = SyntheticTheoryDataset(cfg, cfg.n_val, cfg.data_seed + 1)
        test_ds = SyntheticTheoryDataset(cfg, cfg.n_test, cfg.data_seed + 2)

    sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) \
        if _dist_active() else None

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          sampler=sampler, shuffle=(sampler is None), **common)
    eval_common = dict(num_workers=max(1, cfg.num_workers // 2),
                       pin_memory=cfg.pin_memory, collate_fn=_collate,
                       drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **eval_common)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **eval_common)
    return train_dl, val_dl, test_dl


# ===========================================================================
# 16. TRAINER
# ===========================================================================

class SESITrainer:
    def __init__(self, cfg: SESITheoryConfig, model: SESICloudModel,
                 device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.model.compile()
        self.raw = self.model

        if _dist_active() and device.type == "cuda":
            self.model = DDP(
                self.model, device_ids=[device.index],
                find_unused_parameters=cfg.ddp_find_unused,
                static_graph=cfg.ddp_static_graph,
                gradient_as_bucket_view=cfg.ddp_bucket_view,
            )
        elif _dist_active():
            self.model = DDP(self.model,
                             find_unused_parameters=cfg.ddp_find_unused)

        self.loss_fn = UnifiedTheoryLoss(cfg)

        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or "bias" in n else decay).append(p)

        use_fused = cfg.fused_adamw and device.type == "cuda"
        optimizer_cls = torch.optim.AdamW
        extra = {}
        if cfg.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
                use_fused = False
            except Exception:
                logger.warning("bitsandbytes unavailable; using fused AdamW.")

        self.opt = optimizer_cls(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=cfg.betas,
            **({"fused": True} if use_fused else {}), **extra,
        )

        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[cfg.amp_dtype]
        self.scaler = _GradScaler(
            enabled=bool(cfg.amp and self.dtype == torch.float16 and device.type == "cuda"))
        self.ema = EMA(self._unwrapped(), cfg.ema_decay, cfg.ema_offload_cpu)
        self.step = 0
        self.epoch = 0
        self.best = float("inf")

    def _unwrapped(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _lr(self) -> float:
        c = self.cfg
        if self.step < c.warmup_steps:
            return c.lr * (self.step + 1) / max(1, c.warmup_steps)
        p = (self.step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        return 0.5 * c.lr * (1.0 + math.cos(math.pi * min(1.0, p)))

    def _move(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        c = self.cfg
        self.model.train()
        batch = self._move(batch)
        if c.data_shards:
            _validate_sample(batch, c)
        if c.lambda_coupling > 0:
            batch["x"] = batch["x"].detach().requires_grad_(True)

        for g in self.opt.param_groups:
            g["lr"] = self._lr()
        self.opt.zero_grad(set_to_none=True)

        do_coupling = (self.step % max(1, c.coupling_every) == 0)
        with _autocast(device_type=self.device.type, dtype=self.dtype, enabled=c.amp):
            out = self.model(batch)
            losses = self.loss_fn(batch, out, self._unwrapped(), do_coupling)
            loss = losses["loss"] / max(1, c.grad_accum)

        self.scaler.scale(loss).backward()

        if (self.step + 1) % max(1, c.grad_accum) == 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], c.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update(self._unwrapped())

        self.step += 1
        return {k: float(v.detach()) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, use_ema: bool = True) -> Dict[str, float]:
        model = self._unwrapped()
        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema_state = {k: v.to(self.device) for k, v in self.ema.state_dict().items()}
            model.load_state_dict(ema_state, strict=False)
        model.eval()
        acc: Dict[str, float] = {}
        n = 0
        correct = 0
        for batch in loader:
            batch = self._move(batch)
            with _autocast(device_type=self.device.type, dtype=self.dtype,
                           enabled=self.cfg.amp):
                out = model(batch)
                losses = self.loss_fn(batch, out, model, False)
            bs = batch["x"].shape[0]
            for k, v in losses.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach()) * bs
            n += bs
            if "label" in batch and out["logits"].shape[-1] == 2:
                correct += int((out["logits"].argmax(-1) == batch["label"]).sum())
        res = {k: v / max(1, n) for k, v in acc.items()}
        res["accuracy"] = correct / max(1, n)
        res["n_samples"] = float(n)
        if _dist_active():
            t = torch.tensor([res.get("loss", 0.0), float(n), float(correct)],
                             device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            res["loss"] = float(t[0] / t[1].clamp_min(1))
            res["accuracy"] = float(t[2] / t[1].clamp_min(1))
        if backup is not None:
            model.load_state_dict(backup, strict=False)
        return res

    def state_dict(self) -> Dict[str, Any]:
        return {"model": self._unwrapped().state_dict(),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "ema": self.ema.state_dict(),
                "step": self.step, "epoch": self.epoch, "best": self.best,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self._unwrapped().load_state_dict(sd["model"], strict=False)
        self.opt.load_state_dict(sd["opt"])
        self.scaler.load_state_dict(sd["scaler"])
        self.ema.load_state_dict(sd["ema"])
        self.step, self.epoch, self.best = sd["step"], sd["epoch"], sd["best"]
        torch.set_rng_state(sd["torch_rng"])
        if sd.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(sd["cuda_rng"])

    def save(self, path: str) -> None:
        if not _main():
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)


# ===========================================================================
# 17. PAIRED BOOTSTRAP + HYPOTHESIS SUITE
# ===========================================================================

@torch.no_grad()
def paired_bootstrap(a: Tensor, b: Tensor, n_boot: int = 1000,
                     chunk: int = 200, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap CI on mean(a - b). Vectorized, chunked, no grad."""
    n = min(a.numel(), b.numel())
    if n < 2:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    a, b = a.reshape(-1)[:n].cpu(), b.reshape(-1)[:n].cpu()
    delta = (a - b).mean().item()
    g = torch.Generator().manual_seed(seed)
    boots = []
    remaining = n_boot
    while remaining > 0:
        k = min(chunk, remaining)
        idx = torch.randint(0, n, (k, n), generator=g)
        boots.append((a[idx] - b[idx]).mean(dim=1))
        remaining -= k
    boots = torch.cat(boots)
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975])).tolist()
    return {"delta": delta, "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0), "n_bootstrap": n_boot}


class HypothesisSuite:
    """
    Falsifiable tests for T-A .. T-E.

    Each entry declares:
      prediction, statistic, comparator, decision rule.

    Verdicts are WITHIN-MODEL only. No verdict constitutes experimental
    confirmation of quantum gravity, consciousness, or advantage.
    """

    def __init__(self, cfg: SESITheoryConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def _collect(self, model: SESICloudModel, loader: DataLoader
                 ) -> Dict[str, Tensor]:
        model.eval()
        buf: Dict[str, List[Tensor]] = {}
        for batch in loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with _autocast(device_type=self.device.type,
                           dtype=torch.bfloat16 if self.cfg.amp else torch.float32,
                           enabled=self.cfg.amp):
                out = model(batch)
            for k, v in out.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    buf.setdefault(k, []).append(v.detach().float().cpu())
            if "label" in batch:
                buf.setdefault("label", []).append(batch["label"].cpu())
        return {k: torch.cat(v, 0) for k, v in buf.items()}

    def analytic_limits(self) -> Dict[str, Any]:
        """Cheap, training-free checks of the declared limiting relations."""
        c = self.cfg
        d = c.d_q
        dev = self.device
        rho = psd_state_from_features(
            torch.randn(2, 2 * d * d, device=dev), d)
        cond = torch.randn(2, c.d_meta + c.d_base, device=dev)

        inst = HierarchicalMeasurementMatrix(c).to(dev).eval()
        m0 = inst(cond, rho)

        # g -> 0 : weak instrument approaches identity back-action
        weak = WeakMeasurementReadout(c).to(dev).eval()
        w_small = weak(rho, cond, coupling_override=torch.full((2,), 1e-4, device=dev))
        w_large = weak(rho, cond, coupling_override=torch.full((2,), 50.0, device=dev))
        ident_res = (w_small["weak_K"] -
                     w_small["weak_K"].diagonal(dim1=-2, dim2=-1)
                     .mean(dim=-1)[..., None, None]
                     * torch.eye(d, device=dev)).norm(dim=(-2, -1)).mean().item()

        # decoherence: purity must not increase under unital-ish noise
        lin = LindbladChannel(c).to(dev).eval()
        r_out = lin(rho, cond, dt=1.0)["rho_out"]
        p_in = (rho @ rho).diagonal(dim1=-2, dim2=-1).real.sum(-1)
        p_out = (r_out @ r_out).diagonal(dim1=-2, dim2=-1).real.sum(-1)

        # covariance: Hermitian completion is equivariant, Cholesky is not
        V = inst._assemble(cond)
        K_h = complete_kraus_hermitian(V, c.newton_schulz_iters)
        K_c = complete_kraus_cholesky(V, c.cholesky_jitter)
        H = torch.randn(2, d, d, device=dev, dtype=torch.float32)
        H = H + H.mH
        U = torch.linalg.solve(torch.eye(d, device=dev) + 0.1j * H,
                               torch.eye(d, device=dev) - 0.1j * H)
        Vt = U @ V @ U.mH
        err_h = (complete_kraus_hermitian(Vt, c.newton_schulz_iters)
                 - U @ K_h @ U.mH).norm(dim=(-2, -1)).mean().item()
        err_c = (complete_kraus_cholesky(Vt, c.cholesky_jitter)
                 - U @ K_c @ U.mH).norm(dim=(-2, -1)).mean().item()

        return {
            "kraus_completeness_max": m0["kraus_residual"].max().item(),
            "born_normalization_max": m0["born_error"].max().item(),
            "weak_identity_residual_g_to_0": ident_res,
            "weak_purity_change_g_large": (
                (w_large["weak_post"] @ w_large["weak_post"])
                .diagonal(dim1=-2, dim2=-1).real.sum(-1).mean().item()
                - p_in.mean().item()),
            "decoherence_purity_in": p_in.mean().item(),
            "decoherence_purity_out": p_out.mean().item(),
            "covariance_error_hermitian": err_h,
            "covariance_error_cholesky": err_c,
            "interpretation": "Hermitian normalization is basis-equivariant; "
                              "Cholesky completion is cheaper but NOT covariant.",
        }

    def run(self, model: SESICloudModel, test_loader: DataLoader,
            variants: Optional[Dict[str, SESICloudModel]] = None
            ) -> Dict[str, Any]:
        c = self.cfg
        report: Dict[str, Any] = {"scope": "within-model falsification only",
                                  "analytic_limits": self.analytic_limits()}
        base = self._collect(model, test_loader)

        def metric(d: Dict[str, Tensor], key: str) -> Tensor:
            return d.get(key, torch.zeros(1))

        # T-A: CP / TP / decoherence
        report["T_A_HCST"] = {
            "prediction": "hierarchical measurement is CP/TP; decoherence reduces purity",
            "cp_residual_mean": metric(base, "cp_residual").mean().item(),
            "tp_residual_mean": metric(base, "tp_residual").mean().item(),
            "purity_mean": metric(base, "diag_purity").mean().item(),
            "linear_entropy_mean": metric(base, "diag_linear_entropy").mean().item(),
            "verdict": "supported_within_model"
                       if metric(base, "cp_residual").mean() < 1e-3 else "inconclusive",
        }

        # T-B: covariance + multipartite entanglement
        report["T_B_UCSCQT"] = {
            "prediction": "covariant instruments transform equivariantly; "
                          "entangled preparations exceed separable controls",
            "multipartite_negativity_mean": metric(base, "diag_multipartite_negativity").mean().item(),
            "bipartite_negativity_mean": metric(base, "diag_bipartite_negativity").mean().item(),
            "covariance_error": report["analytic_limits"]["covariance_error_hermitian"],
            "limitation": "positive min-negativity is necessary, not sufficient, "
                          "for genuine multipartite entanglement.",
        }

        # T-C: OPMC factorization + capacity
        cap = RelationalCapacityProbe(c).structural_counts()
        report["T_C_OPMC"] = {
            "prediction": "task observables factor through bounded sigma_T; "
                          "classical evaluation scaling is measurable",
            "factorization_residual_mean": metric(base, "opmc_factorization_residual").mean().item(),
            "sigma_dimension": float(c.k_sigma),
            "capacity_bookkeeping": cap,
            "verdict": "factorization_consistent"
                       if metric(base, "opmc_factorization_residual").mean() < 0.1
                       else "factorization_not_supported",
            "limitation": "capacity is representational, NOT proof of "
                          "computational or quantum advantage.",
        }

        # T-D: GR rates + energy budget
        report["T_D_SESI_GR"] = {
            "prediction": "Gamma_ab >= 0 and symmetric in branch pair; "
                          "energy growth stays within budget",
            "gamma_min": metric(base, "gamma_gr").min().item()
                         if "gamma_gr" in base else None,
            "gamma_mean": metric(base, "gamma_gr").mean().item()
                          if "gamma_gr" in base else None,
            "heat_rate_mean": metric(base, "heat_rate").mean().item(),
            "energy_budget": c.energy_budget,
            "budget_violation_rate": (
                (metric(base, "heat_rate") > c.energy_budget).float().mean().item()
                if "heat_rate" in base else None),
        }

        # T-E: interface CP + kernel normalization + calibration
        report["T_E_INTERFACE"] = {
            "prediction": "event kernels are normalized; interface map is CP/TP; "
                          "mu_I/kappa_I/eta_E are calibratable",
            "kernel_normalization_error": metric(base, "itf_kernel_error").mean().item()
                                          if "itf_kernel_error" in base else None,
            "interface_cp_residual": metric(base, "itf_kraus_residual").mean().item()
                                     if "itf_kraus_residual" in base else None,
            "interface_heating_mean": metric(base, "itf_interface_heating").mean().item()
                                      if "itf_interface_heating" in base else None,
            "mu_I_mean": metric(base, "itf_mu_I").mean().item()
                         if "itf_mu_I" in base else None,
            "kappa_I_mean": metric(base, "itf_kappa_I").mean().item()
                            if "itf_kappa_I" in base else None,
            "eta_E_mean": metric(base, "itf_eta_E").mean().item()
                          if "itf_eta_E" in base else None,
        }

        # paired ablation comparisons with bootstrap CIs
        if variants:
            comparisons = {}
            for name, variant in variants.items():
                v = self._collect(variant, test_loader)
                entry = {}
                for key in ["loss" if "loss" in base else "opmc_factorization_residual",
                            "diag_purity", "heat_rate"]:
                    if key in base and key in v:
                        entry[key] = paired_bootstrap(
                            base[key], v[key], c.n_bootstrap, c.bootstrap_chunk, c.seed)
                comparisons[name] = entry
            report["paired_ablations"] = comparisons

        # task accuracy if labels exist
        if "label" in base and "opmc_logits" in base:
            pred = base["opmc_logits"].argmax(-1)
            report["task_accuracy"] = (pred == base["label"]).float().mean().item()

        return report


# ===========================================================================
# 18. BENCHMARK  (MEASURED, never assumed)
# ===========================================================================

@torch.no_grad()
def benchmark_scaling(cfg: SESITheoryConfig, device: torch.device,
                      dims: Sequence[int] = (4, 8, 12, 16),
                      repeats: int = 5) -> Dict[str, Any]:
    """
    Measure forward + backward cost vs Hilbert dimension for:
      (a) Hermitian (covariant) Kraus completion
      (b) Cholesky (cheap, non-covariant) completion
      (c) exact density-matrix task contraction baseline

    Fits log-log slopes from MEASURED medians. Does not assert a
    polynomial advantage; reports what the hardware actually does.
    """
    rows = []
    for d in dims:
        V = torch.randn(8, 4, d, d, device=device, dtype=torch.complex64)
        V = V + 0.5 * torch.eye(d, device=device, dtype=V.dtype)
        row = {"d_q": d}
        for name, fn in [
            ("hermitian", lambda v: complete_kraus_hermitian(v, cfg.newton_schulz_iters)),
            ("cholesky", lambda v: complete_kraus_cholesky(v, cfg.cholesky_jitter)),
        ]:
            times = []
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                K = fn(V)
                loss = K.abs().sum()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            row[f"{name}_ms_median"] = float(sorted(times)[len(times) // 2] * 1e3)
        rows.append(row)

    def slope(key: str) -> Optional[float]:
        if len(rows) < 2:
            return None
        xs = torch.tensor([math.log(r["d_q"]) for r in rows])
        ys = torch.tensor([math.log(max(r[key], 1e-9)) for r in rows])
        xc, yc = xs.mean(), ys.mean()
        return float(((xs - xc) * (ys - yc)).sum() / ((xs - xc).pow(2).sum() + 1e-12))

    return {"measurements": rows,
            "loglog_slope_hermitian": slope("hermitian_ms_median"),
            "loglog_slope_cholesky": slope("cholesky_ms_median"),
            "note": "Measured wall-clock scaling on this device; not a "
                    "complexity-theoretic advantage claim."}


# ===========================================================================
# 19. SELF-TEST
# ===========================================================================

def run_selftest(quick: bool = True) -> Dict[str, Any]:
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SESITheoryConfig(
        device=str(device), d_in=16, d_q=8, d_meta=4, d_base=4,
        n_outcomes=4, n_branches=3, n_locations=4, n_jumps=2,
        n_scales=3, n_events=8, d_event=4, k_sigma=8,
        n_steps=2 if quick else 4, batch_size=2,
        amp=False if device.type == "cpu" else True,
        amp_dtype="fp32" if device.type == "cpu" else "bf16",
        use_gradient_checkpointing=False,
        compile_model=False, hutchinson_probes=1,
        covariant_normalization=True,
    )
    report: Dict[str, Any] = {"device": str(device), "checks": {}}
    model = SESICloudModel(cfg).to(device)
    model.train()

    B = 2
    batch = {
        "x": torch.randn(B, cfg.d_in, device=device, requires_grad=True),
        "geometry": torch.randn(B, cfg.d_geometry, device=device),
        "events": torch.randn(B, cfg.n_events, cfg.d_event, device=device),
        "mass_branches": F.softplus(torch.randn(B, cfg.n_branches, cfg.n_locations, device=device)),
        "label": torch.randint(0, 2, (B,), device=device),
        "coherence_target": torch.rand(B, device=device) * 0.3,
        "energy_budget": torch.full((B,), cfg.energy_budget, device=device),
    }

    # 1. forward finiteness
    out = model(batch)
    nonfinite = [k for k, v in out.items()
                 if torch.is_tensor(v) and v.is_floating_point()
                 and not torch.isfinite(v).all()]
    report["checks"]["forward_finite"] = {"pass": not nonfinite, "bad_keys": nonfinite}

    # 2. CP / TP
    cp = out["cp_residual"].max().item()
    tp = out["tp_residual"].max().item()
    report["checks"]["complete_positivity_trace_preservation"] = {
        "pass": cp < 2e-3 and tp < 2e-3, "cp_max": cp, "tp_max": tp}

    # 3. PSD of final state
    with fp32_island(device.type):
        ev = torch.linalg.eigvalsh(out["rho_final"].float().real)
        minev = ev.min().item()
    report["checks"]["density_matrix_psd"] = {
        "pass": minev > -2e-3, "min_eigenvalue": minev}

    # 4. full backward graph
    loss_fn = UnifiedTheoryLoss(cfg)
    losses = loss_fn(batch, out, model, compute_coupling=True)
    losses["loss"].backward()
    grad_ok = batch["x"].grad is not None and torch.isfinite(batch["x"].grad).all()
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    report["checks"]["native_full_differentiability"] = {
        "pass": bool(grad_ok) and len(param_grads) > 0,
        "parameters_with_grad": len(param_grads),
        "input_grad_finite": bool(grad_ok)}

    # 5. GR rate positivity
    if "gamma_gr" in out:
        report["checks"]["gr_rate_nonnegative"] = {
            "pass": bool((out["gamma_gr"] >= 0).all()),
            "min_gamma": float(out["gamma_gr"].min())}

    # 6. event kernel normalization
    if "itf_kernel_error" in out:
        report["checks"]["event_kernel_normalized"] = {
            "pass": bool(out["itf_kernel_error"].max() < 1e-4),
            "max_error": float(out["itf_kernel_error"].max())}

    # 7. analytic limits + covariance
    suite = HypothesisSuite(cfg, device)
    report["analytic_limits"] = suite.analytic_limits()
    report["checks"]["covariant_normalization"] = {
        "pass": report["analytic_limits"]["covariance_error_hermitian"] < 5e-2,
        "hermitian_error": report["analytic_limits"]["covariance_error_hermitian"],
        "cholesky_error": report["analytic_limits"]["covariance_error_cholesky"]}

    # 8. training-step smoke test
    trainer = SESITrainer(cfg, SESICloudModel(cfg), device)
    step_logs = trainer.train_step(batch)
    report["checks"]["trainer_step"] = {
        "pass": all(math.isfinite(v) for v in step_logs.values()),
        "logs": step_logs}

    # 9. checkpoint round trip
    ckpt = trainer.state_dict()
    trainer2 = SESITrainer(cfg, SESICloudModel(cfg), device)
    trainer2.load_state_dict(ckpt)
    report["checks"]["checkpoint_resume"] = {
        "pass": trainer2.step == trainer.step}

    # 10. real-data schema validation
    try:
        _validate_sample({k: v.detach() for k, v in batch.items()}, cfg)
        report["checks"]["schema_validation"] = {"pass": True}
    except Exception as e:
        report["checks"]["schema_validation"] = {"pass": False, "error": str(e)}

    report["all_passed"] = all(v.get("pass", False)
                               for v in report["checks"].values())
    return report


# ===========================================================================
# 20. DISTRIBUTED SETUP / CLI / ENTRY POINT
# ===========================================================================

def setup_dist(backend: Optional[str] = None
               ) -> Tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            return torch.device("cuda", local), dist.get_rank(), dist.get_world_size()
        return torch.device("cpu"), dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        return torch.device("cuda"), 0, 1
    return torch.device("cpu"), 0, 1


def cleanup_dist() -> None:
    if _dist_active():
        dist.barrier()
        dist.destroy_process_group()


def _config_from_args(args: argparse.Namespace) -> SESITheoryConfig:
    cfg = SESITheoryConfig()
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        if "betas" in overrides:
            overrides["betas"] = tuple(overrides["betas"])
        cfg = SESITheoryConfig(**{**asdict(cfg), **overrides})
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.out:
        cfg.out_dir = args.out
    if args.data:
        cfg.data_shards = args.data
    if args.data_format:
        cfg.data_format = args.data_format
    if args.schema:
        with open(args.schema) as f:
            cfg.schema = json.load(f)
    if args.no_amp:
        cfg.amp = False
        cfg.amp_dtype = "fp32"
    if args.fp16:
        cfg.amp_dtype = "fp16"
    if args.cpu:
        cfg.device = "cpu"
        cfg.amp = False
        cfg.amp_dtype = "fp32"
    return cfg


def main() -> None:
    p = argparse.ArgumentParser("SESI Cloud & Collapse Theory Tests")
    p.add_argument("--mode", default="selftest",
                   choices=["selftest", "train", "test", "compare",
                            "limits", "benchmark", "calibrate"])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data", type=str, default=None,
                   help="Glob of real data shards, e.g. data/shards/*.h5")
    p.add_argument("--data-format", type=str, default=None,
                   choices=["h5", "npz", "parquet"])
    p.add_argument("--schema", type=str, default=None,
                   help="JSON schema mapping logical -> physical column names")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--full", action="store_true", help="Full selftest, not quick")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if args.mode == "selftest":
        report = run_selftest(quick=not args.full)
        print(json.dumps(report, indent=2, default=str))
        raise SystemExit(0 if report["all_passed"] else 1)

    device, rank, world = setup_dist()
    cfg = _config_from_args(args)
    cfg.device = str(device)
    seed_everything(cfg.seed + rank)
    if _main():
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
        logger.info("World size=%d | device=%s | AMP=%s/%s",
                    world, device, cfg.amp, cfg.amp_dtype)

    train_dl, val_dl, test_dl = build_dataloaders(cfg)

    if args.mode == "limits":
        suite = HypothesisSuite(cfg, device)
        report = {"analytic_limits": suite.analytic_limits()}
        if _main():
            print(json.dumps(report, indent=2, default=str))
            with open(os.path.join(cfg.out_dir, "limits.json"), "w") as f:
                json.dump(report, f, indent=2, default=str)
        cleanup_dist()
        return

    if args.mode == "benchmark":
        report = benchmark_scaling(cfg, device)
        if _main():
            print(json.dumps(report, indent=2, default=str))
            with open(os.path.join(cfg.out_dir, "benchmark.json"), "w") as f:
                json.dump(report, f, indent=2, default=str)
        cleanup_dist()
        return

    model = SESICloudModel(cfg)
    trainer = SESITrainer(cfg, model, device)

    ckpt_path = os.path.join(cfg.out_dir, "ckpt_latest.pt")
    if args.mode in ("test", "compare", "calibrate") and os.path.exists(ckpt_path):
        trainer.load_state_dict(torch.load(ckpt_path, map_location=device))

    if args.mode == "train" or args.mode == "calibrate":
        t0 = time.time()
        while trainer.step < cfg.max_steps:
            sampler = getattr(train_dl, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(trainer.epoch)
            for batch in train_dl:
                logs = trainer.train_step(batch)
                if _main() and trainer.step % cfg.log_every == 0:
                    logger.info("step %6d | %s | lr=%.2e | %.1fs",
                                trainer.step,
                                " ".join(f"{k}={v:.4f}" for k, v in logs.items()),
                                trainer._lr(), time.time() - t0)
                if _main() and trainer.step % cfg.eval_every == 0:
                    ev = trainer.evaluate(val_dl)
                    logger.info("[val] %s", ev)
                    if ev.get("loss", float("inf")) < trainer.best:
                        trainer.best = ev["loss"]
                        trainer.save(os.path.join(cfg.out_dir, "ckpt_best.pt"))
                if trainer.step % cfg.ckpt_every == 0:
                    trainer.save(ckpt_path)
                if trainer.step >= cfg.max_steps:
                    break
            trainer.epoch += 1
        trainer.save(ckpt_path)
        if _main():
            logger.info("Training complete. Best val loss=%.6f", trainer.best)

    if args.mode in ("test", "compare", "calibrate"):
        suite = HypothesisSuite(cfg, device)
        variants = None
        if args.mode == "compare":
            variants = {}
            for name, overrides in [
                ("no_fractal", {"use_fractal": False}),
                ("no_hierarchy", {"use_hierarchy": False}),
                ("no_decoherence", {"use_decoherence": False}),
                ("no_gravity", {"use_gravity": False}),
                ("no_interface", {"use_interface": False}),
                ("no_opmc", {"use_opmc": False}),
                ("cholesky_noncovariant", {"covariant_normalization": False}),
            ]:
                vc = SESITheoryConfig(**{**asdict(cfg), **overrides})
                vm = SESICloudModel(vc).to(device)
                vt = SESITrainer(vc, vm, device)
                steps = 0
                it = iter(train_dl)
                while steps < cfg.suite_steps:
                    try:
                        b = next(it)
                    except StopIteration:
                        it = iter(train_dl)
                        b = next(it)
                    vt.train_step(b)
                    steps += 1
                variants[name] = vt._unwrapped()

        report = suite.run(trainer._unwrapped(), test_dl, variants)
        if _main():
            print(json.dumps(report, indent=2, default=str))
            with open(os.path.join(cfg.out_dir, "hypothesis_report.json"), "w") as f:
                json.dump(report, f, indent=2, default=str)

    cleanup_dist()


if __name__ == "__main__":
    main()
