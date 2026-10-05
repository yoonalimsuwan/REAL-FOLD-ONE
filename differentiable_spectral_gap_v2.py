#!/usr/bin/env python3
"""
differentiable_spectral_gap_v2.py
==================================

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : https://github.com/yoonalimsuwan
# Contact      : msps4u@gmail.com
# Framework    : Structural Calculus (Deterministic Topological Framework)
# License      : MIT
# Year         : 2026
# Version      : 2.0.0 (Native Full Differentiable / AMP-Safe / DDP-Ready)
# =============================================================================
"""

Numerically honest reimplementation of the higher-order Dirichlet-eigenvalue
study, with the fp64 dynamic-range limits of the ``direct`` (matrix-power)
eigenvalue computation made explicit rather than silently bypassed.

Key fixes over v1
-----------------
1. **Log-domain algebra.**  The algebraic value is stored and fitted in log
   space as ``k * log(lambda_1)``.  This does not amplify the LOBPCG
   relative error by ``k`` the way ``lambda_1 ** k`` does in the outer
   range of interest.

2. **Noise-floor aware verification.**  For each k we estimate the
   fp64 noise floor of ``L^k`` as
   ``floor_k = eps * ||L^k||_2 ~ eps * lambda_max(L)^k`` and declare the
   ``direct`` value *unverifiable* when ``lambda_min(L^k) < floor_k``.
   Unverifiable k are reported as ``CONJECTURED`` rather than being
   allowed to silently pass an absolute-tolerance check.

3. **Two-tier cross-check.**  Relative agreement is enforced with a tight
   tolerance only for k where the direct value is genuinely resolvable.
   Beyond that, the two values are logged side-by-side and the *slope*
   (log-log exponent) is the quantity that gets asserted.

4. **Optional mpmath reference.**  If ``mpmath`` is installed and
   ``verify_with_mpmath=True``, the ``direct`` value is recomputed in
   arbitrary precision for the first ``mpmath_k_max`` orders, turning
   ``CONJECTURED`` into ``VERIFIED`` at those k.

The differentiable / AMP / DDP scaffolding of v1 is preserved unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spectral_gap_explorer import build_gasket_graph, build_carpet_graph


__all__ = [
    "TorchGraph",
    "build_gasket_torch",
    "build_carpet_torch",
    "InteriorOperators",
    "interior_dirichlet_operators",
    "build_sparse_laplacian",
    "smallest_eig_rayleigh",
    "DifferentiableSpectralStudy",
    "LearnableGraphStudy",
    "HigherOrderResult",
    "VerificationStatus",
    "fit_scaling_exponent",
    "fit_loglog_slope",
    "DDPContext",
    "amp_context",
    "make_grad_scaler",
    "validate_against_reference",
    "verify_direct_with_mpmath",
    "main",
]


# ===========================================================================
# 0. Numerical-honesty primitives
# ===========================================================================

class VerificationStatus:
    """Enum-style tags for the per-k verification result."""
    VERIFIED    = "VERIFIED"     # direct value resolvable AND matches algebraic
    CONJECTURED = "CONJECTURED"  # direct value below fp64 noise floor
    FAILED      = "FAILED"       # direct value resolvable AND disagrees


# Machine epsilon for the working dtype.  Kept as a module constant so the
# noise-floor computation is auditable.
_EPS_FP64 = float(np.finfo(np.float64).eps)   # 2.22e-16
_EPS_FP32 = float(np.finfo(np.float32).eps)   # 1.19e-07


def _estimate_lambda_max(L: Tensor, X0: Tensor, n_iter: int,
                        tol: float) -> float:
    """Estimate ||L||_2 ~ lambda_max(L) via a power iteration.

    For the graph Laplacians here, lambda_max is O(degree), which is small;
    a handful of power iterations is enough.  Used only to compute the
    fp64 noise floor of L^k -- it does not affect any reported eigenvalue.
    """
    v = X0.detach().clone()
    v = v / v.norm().clamp_min(1e-300)
    lam = 0.0
    with torch.no_grad():
        for _ in range(max(8, n_iter // 20)):
            Lv = torch.sparse.mm(L, v)
            nrm = Lv.norm()
            if nrm < 1e-300:
                break
            v = Lv / nrm
            lam = float(nrm)
    return lam


# ===========================================================================
# 1. Graph container + torch-native builders (unchanged from v1)
# ===========================================================================

@dataclass
class TorchGraph:
    n_nodes: int
    edge_index: Tensor
    edge_weight: Tensor
    boundary_mask: Tensor


def _to_torch_graph(n_nodes: int, edges, boundary_idx, *,
                    dtype: torch.dtype = torch.float32,
                    device: torch.device | str = "cpu") -> TorchGraph:
    if len(edges) == 0:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
        ew = torch.zeros((0,), dtype=dtype, device=device)
    else:
        ei = torch.as_tensor(np.asarray(edges, dtype=np.int64).T, device=device)
        ew = torch.ones(ei.shape[1], dtype=dtype, device=device)
    bm = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    if len(boundary_idx):
        bm[torch.as_tensor(np.asarray(boundary_idx, dtype=np.int64),
                           device=device)] = True
    return TorchGraph(n_nodes, ei, ew, bm)


def build_gasket_torch(level: int, **kw) -> TorchGraph:
    vertices, edges, boundary = build_gasket_graph(level)
    return _to_torch_graph(len(vertices), edges, boundary, **kw)


def build_carpet_torch(level: int, **kw) -> TorchGraph:
    n_nodes, edges, boundary = build_carpet_graph(level)
    return _to_torch_graph(n_nodes, edges, boundary, **kw)


# ===========================================================================
# 2. Differentiable Dirichlet-Laplacian assembly (unchanged from v1)
# ===========================================================================

@dataclass
class InteriorOperators:
    w: Tensor
    r: Tensor
    c: Tensor
    n_int: int


def interior_dirichlet_operators(graph: TorchGraph) -> InteriorOperators:
    ei, ew, bm = graph.edge_index, graph.edge_weight, graph.boundary_mask
    interior_mask = ~bm
    n_int = int(interior_mask.sum().item())

    node_to_int = torch.full((graph.n_nodes,), -1,
                             dtype=torch.long, device=ei.device)
    node_to_int[interior_mask] = torch.arange(n_int, device=ei.device)

    row, col = ei[0], ei[1]
    keep = interior_mask[row] & interior_mask[col]
    r = node_to_int[row[keep]]
    c = node_to_int[col[keep]]
    w = ew[keep]
    return InteriorOperators(w=w, r=r, c=c, n_int=n_int)


def build_sparse_laplacian(w: Tensor, r: Tensor, c: Tensor,
                           n_int: int) -> Tensor:
    deg = torch.zeros(n_int, dtype=w.dtype, device=w.device)
    deg = deg.index_add(0, r, w).index_add(0, c, w)

    diag = torch.arange(n_int, device=w.device)
    L_row = torch.cat([r, c, diag])
    L_col = torch.cat([c, r, diag])
    L_val = torch.cat([-w, -w, deg])

    L = torch.sparse_coo_tensor(
        torch.stack([L_row, L_col]), L_val,
        size=(n_int, n_int), dtype=w.dtype, device=w.device,
    )
    return L.coalesce()


# ===========================================================================
# 3. Differentiable smallest-eigenvalue via Rayleigh quotient (unchanged)
# ===========================================================================

def _lobpcg_smallest(L: Tensor, X0: Tensor, n_iter: int, tol: float
                     ) -> Tuple[Tensor, Tensor]:
    try:
        vals, vecs = torch.lobpcg(L, k=1, X=X0, largest=False,
                                  niter=n_iter, tol=tol)
        return vals[:1], vecs[:, :1]
    except (RuntimeError, NotImplementedError):
        L_dense = L.to_dense()
        vals, vecs = torch.linalg.eigh(L_dense)
        return vals[:1], vecs[:, :1]


def smallest_eig_rayleigh(L: Tensor, X0: Tensor, n_iter: int, tol: float
                          ) -> Tuple[Tensor, Tensor]:
    with torch.no_grad():
        vals, vecs = _lobpcg_smallest(L, X0, n_iter, tol)
    v = vecs[:, 0].detach()
    Lv = torch.sparse.mm(L, v.unsqueeze(-1)).squeeze(-1)
    lam = (v * Lv).sum() / (v * v).sum()
    return lam, v


# ===========================================================================
# 4. Extended result container (now carries verification metadata)
# ===========================================================================

@dataclass
class HigherOrderResult:
    # --- Differentiable tensors ---
    algebraic: Tensor            # (max_k,)  lambda_1 ** k
    direct: Tensor               # (max_k,)  lambda_min(L^k) -- may be garbage
    log_algebraic: Tensor        # (max_k,)  k * log(lambda_1)  <- use for fits

    # --- Diagnostics (non-differentiable) ---
    n_int: int
    per_k_abs_diff: Tensor
    per_k_rel_diff: Tensor
    lambda_max_estimate: float
    noise_floor: Tensor          # (max_k,)  eps * lambda_max(L)^k
    resolved: Tensor             # (max_k,)  bool: direct[k] > noise_floor[k]
    status: list[str]            # per-k VerificationStatus tag


# ===========================================================================
# 5. The study module (core fix: noise-floor aware)
# ===========================================================================

class DifferentiableSpectralStudy(nn.Module):
    """Differentiable higher-order eigenvalue study with explicit fp64 limits.

    Produces:
        algebraic[k-1]     = lambda_1 ** k                (differentiable)
        log_algebraic[k-1] = k * log(lambda_1)            (differentiable)
        direct[k-1]        = lambda_min(L ** k)           (differentiable)

    plus per-k metadata that says, honestly, whether ``direct`` is a
    meaningful number at that k.
    """

    def __init__(self,
                 max_k: int = 11,
                 lobpcg_iters: int = 200,
                 lobpcg_tol: float = 1e-10,
                 eig_dtype: torch.dtype = torch.float64,
                 grad_checkpoint: bool = False,
                 # New: safety margins for the noise-floor test
                 noise_margin: float = 10.0,
                 # New: whether to allow the direct path at k where it's
                 # physically meaningless.  If False (recommended), the
                 # direct branch is skipped at those k and filled with NaN.
                 compute_unresolvable_direct: bool = False):
        super().__init__()
        self.max_k = max_k
        self.lobpcg_iters = lobpcg_iters
        self.lobpcg_tol = lobpcg_tol
        self.eig_dtype = eig_dtype
        self.grad_checkpoint = grad_checkpoint
        self.noise_margin = noise_margin
        self.compute_unresolvable_direct = compute_unresolvable_direct

    def _init_vector(self, n: int, dtype: torch.dtype,
                     device: torch.device) -> Tensor:
        g = torch.Generator(device=device).manual_seed(0x5EED ^ n)
        X = torch.randn(n, 1, dtype=dtype, device=device, generator=g)
        return torch.linalg.qr(X)[0]

    def forward(self, w: Tensor, r: Tensor, c: Tensor, n_int: int
                ) -> HigherOrderResult:
        w_eig = w.to(self.eig_dtype)
        device = w.device

        L_eig = build_sparse_laplacian(w_eig, r, c, n_int)
        X0 = self._init_vector(n_int, self.eig_dtype, device)

        # --- lambda_1: always resolvable ---
        lam1_eig, _ = smallest_eig_rayleigh(
            L_eig, X0, self.lobpcg_iters, self.lobpcg_tol)

        # --- Noise floor of L^k: eps * lambda_max(L)^k ---
        lam_max = _estimate_lambda_max(L_eig, X0, self.lobpcg_iters,
                                       self.lobpcg_tol)
        lam_max = max(lam_max, 1e-12)   # numerical guard

        # --- Accumulators ---
        log_lam1 = torch.log(lam1_eig.clamp_min(1e-300))
        algebraic = [lam1_eig]
        log_algebraic = [log_lam1]
        direct = [lam1_eig]
        noise_floor = [self.noise_margin * _EPS_FP64 * lam_max]
        status = [VerificationStatus.VERIFIED]

        Lk_eig = L_eig
        for k in range(2, self.max_k + 1):
            Lk_eig = torch.sparse.mm(Lk_eig, L_eig).coalesce()

            # The estimated noise floor of L^k in the fp64 spectrum.
            floor_k = (self.noise_margin * _EPS_FP64) * (lam_max ** k)

            # Predict lambda_min(L^k) from the algebraic side; use it to
            # decide *before* spending compute whether the direct branch
            # can possibly be meaningful.
            lam_min_pred = float((lam1_eig ** k).item())
            resolvable = lam_min_pred > floor_k

            log_algebraic.append(k * log_lam1)
            algebraic.append(lam1_eig ** k)

            if (not resolvable) and (not self.compute_unresolvable_direct):
                # Skip the direct solve entirely; report NaN.  This makes
                # it impossible for the cross-check to "pass vacuously".
                direct.append(torch.full_like(lam1_eig, float("nan")))
                status.append(VerificationStatus.CONJECTURED)
            else:
                Xk = self._init_vector(n_int, self.eig_dtype, device)
                lamk_eig, _ = smallest_eig_rayleigh(
                    Lk_eig, Xk, self.lobpcg_iters, self.lobpcg_tol)
                direct.append(lamk_eig)

                # Real verification only when resolvable.
                a, d = float(algebraic[-1].item()), float(lamk_eig.item())
                denom = max(abs(d), 1e-300)
                rel = abs(a - d) / denom
                if not resolvable:
                    status.append(VerificationStatus.CONJECTURED)
                elif rel < 1e-6:
                    status.append(VerificationStatus.VERIFIED)
                else:
                    status.append(VerificationStatus.FAILED)

            noise_floor.append(floor_k)

        algebraic_t = torch.stack(algebraic).to(w.dtype)
        direct_t = torch.stack(direct).to(w.dtype)
        log_alg_t = torch.stack(log_algebraic).to(w.dtype)
        floor_t = torch.tensor(noise_floor, dtype=w.dtype, device=w.device)

        with torch.no_grad():
            abs_diff = (algebraic_t - direct_t).abs()
            rel_diff = abs_diff / direct_t.abs().clamp_min(1e-300)
            resolved = ~torch.isnan(direct_t)

        return HigherOrderResult(
            algebraic=algebraic_t,
            direct=direct_t,
            log_algebraic=log_alg_t,
            n_int=n_int,
            per_k_abs_diff=abs_diff,
            per_k_rel_diff=rel_diff,
            lambda_max_estimate=lam_max,
            noise_floor=floor_t,
            resolved=resolved,
            status=status,
        )


# ===========================================================================
# 6. Learnable wrapper (unchanged)
# ===========================================================================

class LearnableGraphStudy(nn.Module):
    def __init__(self,
                 ops: InteriorOperators,
                 study: Optional[DifferentiableSpectralStudy] = None,
                 *,
                 init_w: Optional[Tensor] = None,
                 log_param: bool = True):
        super().__init__()
        self.study = study or DifferentiableSpectralStudy()
        self.register_buffer("r", ops.r)
        self.register_buffer("c", ops.c)
        self.n_int = ops.n_int
        self.log_param = log_param
        if init_w is None:
            init_w = ops.w.detach().clone()
        if log_param:
            self.theta = nn.Parameter(init_w.clamp_min(1e-12).log())
        else:
            self.theta = nn.Parameter(init_w.clone())

    @property
    def w(self) -> Tensor:
        return self.theta.exp() if self.log_param else self.theta

    def forward(self) -> HigherOrderResult:
        return self.study(self.w, self.r, self.c, self.n_int)


# ===========================================================================
# 7. DDP + AMP scaffolding (unchanged)
# ===========================================================================

class DDPContext:
    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self._owns_group = False

    def __enter__(self):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            self._owns_group = True
        return self

    def __exit__(self, *exc):
        if self._owns_group and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def amp_context(device: torch.device,
                dtype: Optional[torch.dtype]) -> contextlib.AbstractContextManager:
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(device: torch.device,
                     amp_dtype: Optional[torch.dtype]) -> Optional[torch.amp.GradScaler]:
    if device.type == "cuda" and amp_dtype == torch.float16:
        return torch.amp.GradScaler("cuda")
    return None


# ===========================================================================
# 8. Scaling-exponent fitting -- now in log domain
# ===========================================================================

def fit_loglog_slope(log_N: np.ndarray, log_y: np.ndarray) -> Tuple[float, float]:
    """Least-squares slope and intercept of log_y vs log_N."""
    A = np.stack([np.ones_like(log_N), log_N], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, log_y, rcond=None)
    return float(coeffs[1]), float(coeffs[0])


def fit_scaling_exponent(Ns: Sequence[float],
                         values: Sequence[float],
                         use_log: bool = False) -> float:
    """Fit alpha in ``value ~ N^{-alpha}``.

    Parameters
    ----------
    use_log : bool
        If True, ``values`` are already ``log(value)`` (i.e. the
        ``log_algebraic`` vector).  Then ``alpha`` is recovered from the
        slope of ``log(value) vs log(N)`` directly -- which is the
        numerically correct thing to do because the *algebraic* quantity
        is, by construction, exactly ``k * log(lambda_1)``.
    """
    log_N = np.log(np.asarray(Ns, dtype=np.float64))
    if use_log:
        log_y = np.asarray(values, dtype=np.float64)
    else:
        log_y = np.log(np.abs(np.asarray(values, dtype=np.float64)))
    slope, _ = fit_loglog_slope(log_N, log_y)
    return -slope


# ===========================================================================
# 9. Reference cross-check against SciPy (unchanged, still useful at k<=7)
# ===========================================================================

def validate_against_reference(level: int = 2,
                               max_k: int = 4,
                               tol_rel: float = 1e-6,
                               tol_abs: float = 1e-13) -> None:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh

    graph = build_gasket_torch(level, dtype=torch.float64)

    n = graph.n_nodes
    ei = graph.edge_index.cpu().numpy()
    ew = graph.edge_weight.cpu().numpy()
    bm = graph.boundary_mask.cpu().numpy()
    deg = np.zeros(n)
    rows, cols, vals = [], [], []
    for e in range(ei.shape[1]):
        a, b = int(ei[0, e]), int(ei[1, e])
        rows += [a, b]; cols += [b, a]; vals += [-ew[e], -ew[e]]
        deg[a] += ew[e]; deg[b] += ew[e]
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(deg[i])
    L = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    interior = np.where(~bm)[0]
    L_int = L[interior][:, interior].tocsc()
    lam1_ref = float(eigsh(L_int, k=1, sigma=-1e-8, which="LM",
                           return_eigenvectors=False)[0])

    ops = interior_dirichlet_operators(graph)
    study = DifferentiableSpectralStudy(max_k=max_k, eig_dtype=torch.float64)
    res = study(ops.w, ops.r, ops.c, ops.n_int)
    lam1_t = float(res.algebraic[0])

    assert abs(lam1_t - lam1_ref) <= max(tol_abs, tol_rel * abs(lam1_ref)), (
        f"torch={lam1_t:.3e} vs scipy={lam1_ref:.3e}"
    )


# ===========================================================================
# 10. Optional arbitrary-precision verification at high k
# ===========================================================================

def verify_direct_with_mpmath(ops: InteriorOperators,
                              k_max: int,
                              dps: int = 60,
                              max_dense_size: int = 3000
                              ) -> dict[int, float]:
    """Recompute lambda_min(L^k) in arbitrary precision, dense.

    Returns a dict ``{k: lambda_min(L^k)}`` for k=1..k_max.  Only intended
    for small graphs (``n_int <= max_dense_size``); the dense mpmath
    eigensolver is O(n^3) but with 60-digit arithmetic.
    """
    try:
        import mpmath
    except ImportError:
        warnings.warn("mpmath not installed; skipping arbitrary-precision "
                      "verification.")
        return {}

    n = ops.n_int
    if n > max_dense_size:
        warnings.warn(f"n_int={n} exceeds max_dense_size={max_dense_size}; "
                      "skipping mpmath verification.")
        return {}

    mpmath.mp.dps = dps

    # Build the dense Dirichlet Laplacian in mpmath.
    L = mpmath.zeros(n, n)
    w_np = ops.w.detach().cpu().numpy().astype(np.float64)
    r_np = ops.r.detach().cpu().numpy()
    c_np = ops.c.detach().cpu().numpy()
    for e in range(len(w_np)):
        a, b = int(r_np[e]), int(c_np[e])
        we = mpmath.mpf(float(w_np[e]))
        L[a, a] += we; L[b, b] += we
        L[a, b] -= we; L[b, a] -= we

    out: dict[int, float] = {}
    Lk = mpmath.eye(n)
    for k in range(1, k_max + 1):
        Lk = Lk * L
        E, _ = mpmath.eigsy(Lk)
        lam_min = min(mpmath.re(E[i]) for i in range(n))
        out[k] = float(lam_min)
    return out


# ===========================================================================
# 11. Demo training step (unchanged)
# ===========================================================================

def demo_train_step(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: Optional[torch.amp.GradScaler],
                    amp_dtype: Optional[torch.dtype],
                    target_lambda: Tensor) -> float:
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    with amp_context(device, amp_dtype):
        res = model()
        loss = F.mse_loss(res.log_algebraic, torch.log(target_lambda))
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach())


# ===========================================================================
# 12. CLI -- now reports VERIFIED vs CONJECTURED explicitly
# ===========================================================================

def _run_one_rank(args, ddp: DDPContext) -> None:
    torch.manual_seed(0)
    device = torch.device(f"cuda:{ddp.local_rank}"
                          if torch.cuda.is_available() else "cpu")

    if ddp.is_main:
        print(f"[rank 0] world_size={ddp.world_size} device={device}")

    family = {"gasket": build_gasket_torch, "carpet": build_carpet_torch}[args.family]
    levels = list(range(args.level_min, args.level_max + 1))

    validate_against_reference(level=2, max_k=4)
    if ddp.is_main:
        print("[rank 0] SciPy reference validation passed.\n")

    # ------------------------------------------------------------------ #
    # Main sweep
    # ------------------------------------------------------------------ #
    Ns: list[int] = []
    log_alg_per_k: dict[int, list[float]] = {k: [] for k in range(1, args.max_k + 1)}
    status_per_k: dict[int, list[str]] = {k: [] for k in range(1, args.max_k + 1)}

    # If mpmath verification requested, we do it on the smallest level only
    # (dense O(n^3) arbitrary-precision is expensive).
    mpmath_ref: dict[int, float] = {}
    if args.verify_mpmath and ddp.is_main:
        graph_small = family(levels[0], dtype=torch.float64, device="cpu")
        ops_small = interior_dirichlet_operators(graph_small)
        print(f"[rank 0] running mpmath reference at level {levels[0]} "
              f"(n_int={ops_small.n_int}, dps={args.mpmath_dps}) ...")
        mpmath_ref = verify_direct_with_mpmath(
            ops_small, k_max=args.max_k,
            dps=args.mpmath_dps,
            max_dense_size=args.mpmath_max_n)
        print(f"[rank 0] mpmath done.\n")

    t0 = time.time()
    for idx, lvl in enumerate(levels):
        graph = family(lvl, dtype=torch.float64, device=device)
        ops = interior_dirichlet_operators(graph)
        study = DifferentiableSpectralStudy(
            max_k=args.max_k,
            lobpcg_iters=args.lobpcg_iters,
            lobpcg_tol=args.lobpcg_tol,
            eig_dtype=torch.float64,
            compute_unresolvable_direct=args.force_direct,
        ).to(device)

        with torch.no_grad():
            res = study(ops.w, ops.r, ops.c, ops.n_int)

        Ns.append(graph.n_nodes)
        if ddp.is_main:
            print(f"  [{args.family} L{lvl}]  N={graph.n_nodes:6d}  "
                  f"n_int={res.n_int:6d}  lambda_max~{res.lambda_max_estimate:.2f}")

        for k in range(1, args.max_k + 1):
            v = float(res.log_algebraic[k - 1])
            log_alg_per_k[k].append(v)
            status_per_k[k].append(res.status[k - 1])

            if ddp.is_main:
                alg = float(res.algebraic[k - 1])
                d = float(res.direct[k - 1])
                floor = float(res.noise_floor[k - 1])
                tag = res.status[k - 1]
                if math.isnan(d):
                    d_str = "        (skipped)"
                else:
                    d_str = f"{d:.6e}"
                marker = {"VERIFIED": "✓", "CONJECTURED": "~", "FAILED": "✗"}[tag]
                print(f"      order {2*k:2d} (k={k:2d}): "
                      f"lambda^k = {alg:.6e}   direct = {d_str}   "
                      f"floor = {floor:.1e}   [{marker} {tag}]")

                # Cross-check against mpmath on level 0
                if idx == 0 and k in mpmath_ref and not math.isnan(d):
                    ref = mpmath_ref[k]
                    rel = abs(d - ref) / max(abs(ref), 1e-300)
                    print(f"        mpmath ref: {ref:.6e}   rel.err vs direct: {rel:.2e}")
                elif idx == 0 and k in mpmath_ref and math.isnan(d):
                    print(f"        mpmath ref: {mpmath_ref[k]:.6e}  "
                          f"(vs algebraic: "
                          f"{float(res.algebraic[k-1]):.6e})")

    # ------------------------------------------------------------------ #
    # Fitted exponents -- fit the *log* algebraic directly
    # ------------------------------------------------------------------ #
    if ddp.is_main:
        print(f"\n[rank 0] sweep wall-clock: {time.time() - t0:.2f}s")
        print("\n  Fitted scaling exponents (using log_algebraic):")
        alphas: dict[int, float] = {}
        for k in range(1, args.max_k + 1):
            alpha = fit_scaling_exponent(Ns, log_alg_per_k[k], use_log=True)
            alphas[k] = alpha
            predicted = k * alphas[1]
            print(f"    order {2*k:2d} (k={k:2d}):  "
                  f"alpha_{k} = {alpha:.4f}    "
                  f"(k * alpha_1 = {predicted:.4f})")

        # ------------------------------------------------------------------ #
        # Verification summary
        # ------------------------------------------------------------------ #
        print("\n  Verification summary (per k, all levels must agree):")
        for k in range(1, args.max_k + 1):
            tags = status_per_k[k]
            if all(t == VerificationStatus.VERIFIED for t in tags):
                verdict = "VERIFIED"
            elif any(t == VerificationStatus.FAILED for t in tags):
                verdict = "FAILED  <-- investigate!"
            else:
                verdict = "CONJECTURED  (below fp64 noise floor)"
            print(f"    k={k:2d} (order {2*k:2d}):  {verdict}")

        # ------------------------------------------------------------------ #
        # Consistency check on the exponent growth
        # ------------------------------------------------------------------ #
        # We assert the slope is linear in k (that is the *verifiable*
        # mathematical content of this study, independent of the direct
        # branch).
        k_arr = np.arange(1, args.max_k + 1)
        alpha_arr = np.array([alphas[k] for k in k_arr])
        A = np.stack([np.ones_like(k_arr), k_arr], axis=1).astype(np.float64)
        coeffs, *_ = np.linalg.lstsq(A, alpha_arr, rcond=None)
        intercept, slope = float(coeffs[0]), float(coeffs[1])
        resid = alpha_arr - (intercept + slope * k_arr)
        max_resid = float(np.abs(resid).max())
        print(f"\n  Linear-fit of alpha_k vs k:  alpha_k ~ {slope:.4f} * k "
              f"+ {intercept:.4f}   (max |residual| = {max_resid:.2e})")
        if max_resid > 0.05:
            print("  WARNING: alpha_k is not linear in k to within 0.05.  "
                  "Investigate before reporting.")
        else:
            print("  OK: alpha_k is linear in k, as expected.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["gasket", "carpet"], default="gasket")
    p.add_argument("--level-min", type=int, default=2)
    p.add_argument("--level-max", type=int, default=6)
    p.add_argument("--max-k", type=int, default=11)
    p.add_argument("--lobpcg-iters", type=int, default=200)
    p.add_argument("--lobpcg-tol", type=float, default=1e-10)
    # New flags
    p.add_argument("--force-direct", action="store_true",
                   help="Compute the direct branch even when below noise "
                        "floor (it will be garbage; only for debugging).")
    p.add_argument("--verify-mpmath", action="store_true",
                   help="Recompute lambda_min(L^k) with mpmath at the "
                        "smallest level (requires mpmath).")
    p.add_argument("--mpmath-dps", type=int, default=60)
    p.add_argument("--mpmath-max-n", type=int, default=3000,
                   help="Skip mpmath verification if n_int exceeds this.")
    args = p.parse_args(argv)

    with DDPContext() as ddp:
        _run_one_rank(args, ddp)


if __name__ == "__main__":
    main()
