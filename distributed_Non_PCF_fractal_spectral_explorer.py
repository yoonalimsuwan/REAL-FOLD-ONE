"""
Distributed Non-P.C.F. Fractal Spectral Explorer — Production Build (v3.0).
Integrates:
  1. DDP-Ready & AMP-Safe mechanics from the consolidated library.
  2. Higher-order Laplacian spectral gap exploration for non-p.c.f. fractals.

# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : https://github.com/yoonalimsuwan
# Contact      : msps4u@gmail.com
# Framework    : Structural Calculus (Deterministic Topological Framework)
# License      : MIT
# Year         : 2026
# Version      : 3.0.0 (Native Full Differentiable / AMP-Safe / DDP-Ready)
# =============================================================================


"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, csr_matrix

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None  # type: ignore[assignment]

__all__ = [
    "DistributedHigherOrderLaplacian",
    "build_torch_dirichlet_laplacian",
    "run_distributed_spectral_study",
]


# =============================================================================
# Shared helpers — AMP-safe numerics, DDP collectives
# =============================================================================
_FP32_UNSTABLE = (torch.float16, torch.bfloat16)


def _ddp_active() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _ddp_active() else 1


def _rank() -> int:
    return dist.get_rank() if _ddp_active() else 0


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Non-destructive SUM all-reduce; returns a detached clone."""
    out = x.detach().clone()
    if _ddp_active():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _all_reduce_max(x: torch.Tensor) -> torch.Tensor:
    out = x.detach().clone()
    if _ddp_active():
        dist.all_reduce(out, op=dist.ReduceOp.MAX)
    return out


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp16/bf16 → fp32 for LAPACK-class and long-chain kernels."""
    return torch.float32 if dtype in _FP32_UNSTABLE else dtype


# =============================================================================
# Sparse graph builder — fully vectorized, direct COO→CSR
# =============================================================================
def build_torch_dirichlet_laplacian(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    boundary_idx: Sequence[int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Vectorized Dirichlet graph Laplacian as a PyTorch CSR tensor.

    Optimizations vs. the reference implementation
    ----------------------------------------------
    * No Python per-edge loop — a single `np.bincount` + array concat.
    * `scipy` builds the CSR; we transfer `indptr/indices/data` directly to
      `torch.sparse_csr_tensor`, skipping the intermediate COO tensor (which
      the reference version materialized via `torch.sparse_coo_tensor` and
      then `.to_sparse_csr()` — two conversions, twice the memory).
    * `sort_indices()` called once so downstream SpMM hits the sorted fast
      path.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    m = E.shape[0]

    # Symmetric off-diagonals (−1) + diagonal (degree)
    src = np.concatenate([E[:, 0], E[:, 1], np.arange(n_nodes, dtype=np.int64)])
    dst = np.concatenate([E[:, 1], E[:, 0], np.arange(n_nodes, dtype=np.int64)])
    deg = np.bincount(E.reshape(-1), minlength=n_nodes).astype(np.float64)
    vals = np.concatenate([-np.ones(2 * m, dtype=np.float64), deg])

    L_full = coo_matrix((vals, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()

    # Dirichlet restriction to interior nodes
    mask = np.ones(n_nodes, dtype=bool)
    if len(boundary_idx) > 0:
        mask[np.asarray(boundary_idx, dtype=np.int64)] = False
    interior = np.flatnonzero(mask)
    if interior.size == 0:
        raise ValueError("Empty interior — every node is on the boundary.")

    L_int = L_full[interior, :][:, interior].tocsr()
    L_int.sort_indices()

    dev = device or torch.device("cpu")
    return torch.sparse_csr_tensor(
        torch.from_numpy(L_int.indptr.astype(np.int64)),
        torch.from_numpy(L_int.indices.astype(np.int64)),
        torch.from_numpy(L_int.data.astype(np.float64)).to(dtype),
        size=L_int.shape,
        dtype=dtype,
        device=dev,
    )


# ------------------------------------------------------------------ #
# CSR ↔ scipy bridges (fallback path for older PyTorch)
# ------------------------------------------------------------------ #
def _torch_csr_to_scipy(T: torch.Tensor) -> csr_matrix:
    T = T.to_sparse_csr()
    return csr_matrix(
        (
            T.values().detach().cpu().numpy(),
            T.col_indices().detach().cpu().numpy(),
            T.crow_indices().detach().cpu().numpy(),
        ),
        shape=tuple(T.shape),
    )


def _scipy_csr_to_torch(C: csr_matrix, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    C = C.tocsr()
    C.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(C.indptr.astype(np.int64)),
        torch.from_numpy(C.indices.astype(np.int64)),
        torch.from_numpy(C.data.astype(np.float64)).to(dtype),
        size=C.shape,
        dtype=dtype,
        device=device,
    )


def _sparse_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CSR @ CSR → CSR, with a scipy fallback for older PyTorch."""
    try:
        C = torch.sparse.mm(A, B)
        return C if C.layout == torch.sparse_csr else C.to_sparse_csr()
    except (RuntimeError, NotImplementedError):
        return _scipy_csr_to_torch(
            _torch_csr_to_scipy(A) @ _torch_csr_to_scipy(B),
            device=A.device,
            dtype=A.dtype,
        )


# =============================================================================
# DDP-Ready, AMP-Safe, Differentiable Higher-Order Laplacian
# =============================================================================
class DistributedHigherOrderLaplacian(nn.Module):
    """
    Differentiable + distributed higher-order Laplacian evaluator.

    What actually changed vs. the reference
    ---------------------------------------
    * **Precomputed L^k sparse cache.** LOBPCG convergence needs many
      matvecs; without the cache each iteration costs `k · SpMM`. With it,
      each iteration costs `1 · SpMM` after a one-shot precompute of
      `k − 1` sparse-sparse products. For k = 4 this is a ~4× speedup on
      the dominant inner loop.
    * **Differentiable Rayleigh quotient.** LOBPCG runs under `no_grad`;
      the returned eigenvalue is *recomputed* as `vᵀ Lᵏ v / vᵀ v` outside
      `no_grad`, yielding `∂λ/∂L = v vᵀ / (vᵀv)` exactly — the true
      eigenvector outer product — with no extra matvecs.
    * **Deterministic LOBPCG.** Seeded generator so runs are bit-reproducible
      given fixed backends.
    * **No DDP wrap.** The module has zero parameters. Wrapping in
      `DistributedDataParallel` used to incur per-step NCCL overhead in
      exchange for nothing. The correct distribution axis here is *spectral
      order* — see `run_distributed_spectral_study`.

    AMP safety
    ----------
    All matmuls and LOBPCG work run under `autocast(enabled=False)` in
    fp32/fp64. fp16 `cumsum` inside LOBPCG's reorthogonalization and fp16
    SpMM accumulations are both real overflow risks.
    """

    def __init__(
        self,
        laplacian_csr: torch.Tensor,
        max_k: int = 4,
        *,
        precompute_powers: bool = True,
    ) -> None:
        super().__init__()
        if laplacian_csr.layout != torch.sparse_csr:
            laplacian_csr = laplacian_csr.to_sparse_csr()
        self.register_buffer("lap", laplacian_csr, persistent=False)
        self.max_k = int(max_k)
        self.precompute_powers = bool(precompute_powers)

        self._powers: Dict[int, torch.Tensor] = {1: laplacian_csr}
        if precompute_powers:
            self._precompute_powers()

    # ------------------------------------------------------------------ #
    # Sparse-power cache
    # ------------------------------------------------------------------ #
    def _precompute_powers(self) -> None:
        with torch.autocast(self.lap.device.type, enabled=False):
            L = self.lap
            for k in range(2, self.max_k + 1):
                L = _sparse_matmul(L, self.lap)
                self._powers[k] = L

    def _get_Lk(self, k: int) -> torch.Tensor:
        if k not in self._powers:
            with torch.autocast(self.lap.device.type, enabled=False):
                self._powers[k] = _sparse_matmul(self._powers[k - 1], self.lap)
        return self._powers[k]

    # ------------------------------------------------------------------ #
    # Forward: L^k u  (matrix-free, differentiable w.r.t. u)
    # ------------------------------------------------------------------ #
    def forward(self, u: torch.Tensor, k: int = 1) -> torch.Tensor:
        return self.apply_laplacian(u, k)

    def apply_laplacian(self, u: torch.Tensor, k: int) -> torch.Tensor:
        if k < 1:
            raise ValueError("k must be >= 1")
        out_dtype = u.dtype
        with torch.autocast(u.device.type, enabled=False):
            x = u.to(self.lap.dtype)
            L = self.lap
            for _ in range(k):
                x = torch.sparse.mm(L, x.unsqueeze(-1)).squeeze(-1)
        return x.to(out_dtype)

    # ------------------------------------------------------------------ #
    # Eigenvalue solver
    # ------------------------------------------------------------------ #
    def smallest_eigenvalue(
        self,
        k: int = 1,
        *,
        differentiable: bool = True,
        seed: int = 0,
        maxiter: int = 1000,
        tol: float = 1e-9,
    ) -> torch.Tensor:
        """
        Smallest eigenvalue of L^k, differentiated via the Rayleigh
        quotient of the LOBPCG-converged eigenvector.
        """
        Lk = self._get_Lk(k)
        dim = Lk.size(0)
        dev = Lk.device
        dt = _compute_dtype(Lk.dtype)

        # ---- 1. LOBPCG under no_grad (deterministic) -------------------
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            gen = torch.Generator(device=dev).manual_seed(int(seed))
            X = torch.randn(dim, 1, dtype=dt, device=dev, generator=gen)
            Lk_c = Lk if Lk.dtype == dt else Lk.to(dt)

            def op(V: torch.Tensor) -> torch.Tensor:
                return torch.sparse.mm(Lk_c, V)

            try:
                _, eigvecs = torch.lobpcg(
                    A=op, X=X, largest=False, maxiter=maxiter, tol=tol
                )
            except (RuntimeError, AttributeError):
                if dim <= 8192:
                    _, eigvecs = torch.linalg.eigh(Lk_c.to_dense())
                    eigvecs = eigvecs[:, :1]
                else:
                    raise
            v = eigvecs[:, 0].detach().to(dt)

        # ---- 2. Differentiable Rayleigh quotient ----------------------
        if not differentiable:
            with torch.no_grad(), torch.autocast(dev.type, enabled=False):
                Lv = torch.sparse.mm(Lk_c, v.unsqueeze(-1)).squeeze(-1)
                return torch.dot(v, Lv) / torch.dot(v, v)

        with torch.autocast(dev.type, enabled=False):
            Lv = torch.sparse.mm(Lk_c, v.unsqueeze(-1)).squeeze(-1)
            lam = torch.dot(v, Lv) / torch.dot(v, v)
        return lam


# =============================================================================
# Distributed driver — spectral-order parallelism (the correct DDP axis)
# =============================================================================
def run_distributed_spectral_study(
    name: str,
    build_fn: Callable[[int], Tuple[int, Sequence[Tuple[int, int]], Sequence[int]]],
    levels: Sequence[int],
    max_k: int = 4,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-13,
    deterministic: bool = True,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], List[int]]:
    """
    Distributed higher-order spectral study.

    Rank partition
    --------------
    For each level, rank `r` owns the k values `{(k) : (k-1) % world == r}`.
    Rank 0 always owns `k = 1` and broadcasts `λ_1` to the world before the
    per-rank solves begin — this makes the algebraic comparison
    `(λ_1)^k` well-defined on every rank without redundant work.

    Determinism
    -----------
    LOBPCG seeds are derived from `(level, k)` so repeated runs match
    bit-for-bit given identical hardware and cuBLAS/cuSPARSE versions.
    """
    rank = _rank()
    world = _world_size()
    is_master = (rank == 0)

    if device is None:
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", rank))
            device = torch.device(f"cuda:{local}")
        else:
            device = torch.device("cpu")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)

    if is_master:
        print(f"\n=== Distributed Study: {name} (orders 2..{2 * max_k}) ===")
        print(f"    world_size={world}  device={device}  dtype={dtype}")

    Ns: List[int] = []
    algebraic: Dict[int, List[float]] = {k: [] for k in range(1, max_k + 1)}
    direct: Dict[int, List[float]] = {k: [] for k in range(1, max_k + 1)}

    for lvl in levels:
        n_nodes, edges, boundary_idx = build_fn(lvl)
        Ns.append(int(n_nodes))

        L_csr = build_torch_dirichlet_laplacian(
            n_nodes, edges, boundary_idx, device=device, dtype=dtype
        )
        module = DistributedHigherOrderLaplacian(L_csr, max_k=max_k).to(device)

        # ---- k = 1: rank 0 computes, broadcasts to world ---------------
        lam1_holder = torch.zeros(1, dtype=dtype, device=device)
        if rank == 0:
            lam1_holder[0] = module.smallest_eigenvalue(
                k=1, seed=lvl * 1009 + 1
            )
        if _ddp_active():
            dist.broadcast(lam1_holder, src=0)
        lam1 = float(lam1_holder.item())

        # ---- k ≥ 2: shard across ranks ---------------------------------
        lam_buf = torch.zeros(max_k, dtype=dtype, device=device)
        lam_buf[0] = lam1
        for k in range(2, max_k + 1):
            if (k - 1) % world != rank:
                continue
            lam_buf[k - 1] = module.smallest_eigenvalue(
                k=k, seed=lvl * 1009 + k
            )
        lam_buf = _all_reduce_sum(lam_buf)
        lam_list = lam_buf.tolist()

        for k in range(1, max_k + 1):
            alg = lam1 ** k
            dr = lam_list[k - 1]
            algebraic[k].append(alg)
            direct[k].append(dr)

            abs_d = abs(alg - dr)
            rel_d = abs_d / max(abs(dr), 1e-300)
            ok = (rel_d < rel_tol) or (abs_d < abs_tol)

            if is_master:
                flag = "OK" if ok else "MISMATCH"
                print(
                    f"  Level {lvl}: N={n_nodes}  k={k} (order {2 * k}):  "
                    f"λ_1^k = {alg:.6e}  direct = {dr:.6e}  "
                    f"rel.diff = {rel_d:.2e}  [{flag}]"
                )
                if not ok:
                    raise AssertionError(
                        f"Cross-check FAILED at level={lvl}, k={k}: "
                        f"λ_1^k={alg}, direct={dr}"
                    )

    # ---- Optional log-log regression on the master node ----------------
    if is_master:
        Ns_arr = np.asarray(Ns, dtype=np.float64)
        print("\n  Spectral scaling exponents (log-log slope vs N):")
        for k in range(1, max_k + 1):
            ys = np.log(np.abs(np.asarray(direct[k])) + 1e-300)
            xs = np.log(Ns_arr)
            slope = float(np.polyfit(xs, ys, 1)[0])
            print(f"    k={k} (order {2 * k}): exponent ≈ {slope:+.4f}")

    return algebraic, direct, Ns
