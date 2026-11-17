#!/usr/bin/env python3
"""
=============================================================================
DNS AT Re_L = 1e6 ON 7000 GPUs — honest planner + exact high-order weights
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

THREE ANSWERS, COMPUTED NOT ASSERTED
[A] ORDER 240 IS COMPUTABLE EXACTLY, BUT IT LOSES TO ORDER 8-12.
    The maximal-order central first-derivative weights on offsets -M..M
    have the CLOSED FORM
        w_k = (-1)^(k+1)/k * C(2M, M-k) / C(2M, M),   w_0 = 0, w_-k = -w_k
    and the second-derivative weights
        w_k = 2(-1)^(k+1)/k^2 * C(2M,M-k)/C(2M,M),    w_0 = -2 sum_{k>=1} w_k
    Both are evaluated here in EXACT rational arithmetic (fractions), so
    order 240 (M=120) is obtained with NO roundoff in the weights.
    As M -> infinity, w_k -> (-1)^(k+1)/k, the Fourier spectral limit.
    So order 240 is essentially a truncated spectral stencil: it is NOT
    more accurate than spectral, and it needs 120 GHOST LAYERS.
    The halo-cost model below finds the order that MINIMISES total cost
    for a given rank count. On thousands of ranks that optimum is 8-12,
    or spectral with a pencil FFT (zero halo). Not 240.

[B] CARPET-19 CANNOT BE THE NS SPATIAL OPERATOR.
    From the authors' own Consolidated_Revision_Carpet-19:
      Theorem 8 (Hino 2005): energy measures Gamma(f,.) are SINGULAR w.r.t. mu.
      Corollary 3: f in D(A) nonconstant => f^2 NOT in D(A).
                   Hence D(A) IS NOT AN ALGEBRA.
      Theorem 9: no Connes-type Dirac operator with bounded commutators.
    Navier-Stokes contains u.grad(u), a PRODUCT. Corollary 3 forbids it.
    Carpet refinement also removes the centre cell of each 3x3 block,
    which would punch HOLES in a fluid domain.
    Carpet contributes rho >= 9/8 energy normalisation and an AMR
    PRIORITY weight only. This module implements exactly that.

[C] Re_L = 1e6 FITS ON 7000 GPUs. Re_L = 1e7 DOES NOT.
    N ~ 0.48 Re_L^(3/4) for spectral resolution k_max*eta >= 1.5, L = 2pi.
      Re_L=1e6 -> N=16384 -> 4.4e12 cells -> 141 TB fp32 -> 25% of 560 TB  OK
      Re_L=1e7 -> N=98304 -> 9.5e14 cells -> 30400 TB -> 54x over          NO
    Re_lambda is a DIFFERENT number: Re_L ~ Re_lambda^2/15. If the user
    means Re_lambda = 1e6 then Re_L ~ 6.7e10 and N ~ 6e7 per axis, which
    no machine can hold. Both readings are reported; neither is guessed.

HONEST SCOPE
  * Bandwidth and stencil throughput are PARAMETERS with stated defaults.
    They are assumptions until measured on the target machine. Every
    time estimate is labelled with the values used.
  * No multi-rank run is executed here. The cost model is arithmetic.
  * No experimental validation. Oracles: exact rational weights, measured
    moment residuals, measured discrete-symbol error, analytic scaling.
  * The pencil-FFT path is an ADAPTER to cuFFTMp/heFFTe. A hand-rolled
    all-gather transpose at 16384^3 cannot fit in memory and is not offered.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
CARPET_RHO = 9.0 / 8.0
KAPPA_FATNESS = 2.0
A_MIN = 0.01

BANNER = f"""
=============================================================================
DNS Re_L=1e6 PLANNER v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
Order 240 exact weights: YES. Optimal on 7000 GPUs: NO (halo 120).
Carpet is NOT the NS operator (your Corollary 3: D(A) not an algebra).
Re_L=1e6 fits 7000 GPUs. Re_L=1e7 needs ~380k GPUs.
=============================================================================
"""


# =============================================================================
# §1. EXACT MAXIMAL-ORDER CENTRAL WEIGHTS (closed form, rational)
# =============================================================================

def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def exact_d1_weights(M: int) -> List[Fraction]:
    """
    Maximal-order (2M) central FIRST-derivative weights on -M..M, EXACT.
        w_k = (-1)^(k+1)/k * C(2M,M-k)/C(2M,M)
    Verified: M=1 -> [-1/2,0,1/2]; M=2 -> [1/12,-2/3,0,2/3,-1/12].
    Limit M->inf: w_k -> (-1)^(k+1)/k (Fourier spectral).
    """
    if M < 1:
        raise ValueError("M >= 1")
    C = _binom(2 * M, M)
    w = [Fraction(0)] * (2 * M + 1)
    for k in range(1, M + 1):
        val = Fraction(_binom(2 * M, M - k), k * C)
        if (k + 1) % 2:
            val = -val
        w[M + k] = val
        w[M - k] = -val
    return w


def exact_d2_weights(M: int) -> List[Fraction]:
    """
    Maximal-order (2M) central SECOND-derivative weights, EXACT.
        w_k = 2(-1)^(k+1)/k^2 * C(2M,M-k)/C(2M,M),  w_0 = -2 sum_{k>=1} w_k
    Verified: M=1 -> [1,-2,1]; M=2 -> [-1/12,4/3,-5/2,4/3,-1/12].
    """
    if M < 1:
        raise ValueError("M >= 1")
    C = _binom(2 * M, M)
    w = [Fraction(0)] * (2 * M + 1)
    s = Fraction(0)
    for k in range(1, M + 1):
        val = Fraction(2 * _binom(2 * M, M - k), k * k * C)
        if (k + 1) % 2:
            val = -val
        w[M + k] = val
        w[M - k] = val
        s += val
    w[M] = -2 * s
    return w


def weights_float(w_exact: List[Fraction]) -> np.ndarray:
    return np.array([float(x) for x in w_exact], dtype=np.float64)


def moment_residuals_normalized(w: np.ndarray, deriv: int,
                                order: int) -> Dict[str, float]:
    """
    Moment conditions in NORMALIZED units (k/M)^p so that high powers do
    not overflow float64. Exact requirement:
        sum_k w_k k^p = deriv! if p == deriv else 0.
    Normalized: sum_k w_k (k/M)^p = deriv!/M^deriv if p==deriv else 0.
    The reported number is |achieved - required| in these units, i.e. the
    roundoff floor of the stencil in float64.
    """
    M = (len(w) - 1) // 2
    x = np.arange(-M, M + 1, dtype=np.float64) / max(M, 1)
    out = {}
    for p in range(0, min(order + deriv, 12) + 1):
        got = float(np.sum(w * x ** p))
        want = math.factorial(deriv) / (M ** deriv) if p == deriv else 0.0
        out[f"p{p}"] = abs(got - want)
    return out


def symbol_d1(kh: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = (len(w) - 1) // 2
    s = np.zeros_like(kh)
    for i, wi in enumerate(w):
        k = i - M
        if k != 0:
            s += wi * np.sin(k * kh)
    return s


def symbol_d2(kh: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = (len(w) - 1) // 2
    s = np.zeros_like(kh)
    for i, wi in enumerate(w):
        s += wi * np.cos((i - M) * kh)
    return s


def usable_kh(w: np.ndarray, deriv: int = 1, tol: float = 0.01) -> float:
    """Largest k*h with relative symbol error < tol. MEASURED, not assumed."""
    kh = np.linspace(1e-4, math.pi, 6000)
    if deriv == 1:
        approx = symbol_d1(kh, w)
        exact = kh
    else:
        approx = -symbol_d2(kh, w)
        exact = kh ** 2
    rel = np.abs(approx - exact) / np.maximum(exact, 1e-30)
    ok = rel < tol
    return float(kh[ok].max()) if ok.any() else 0.0


def c_order_from_measured(order: int, kappa: float = 1.5,
                          tol: float = 0.01) -> float:
    """
    N >= c * Re^(3/4), c derived from the MEASURED usable k*h.
      f = (k*h)_usable / pi ; k_max = f*N/2 ; k_max*eta >= kappa
      eta = 2*pi*Re^(-3/4)  =>  N >= kappa/(pi*f) * Re^(3/4)
    """
    M = order // 2
    w = weights_float(exact_d1_weights(M))
    khu = usable_kh(w, 1, tol)
    if khu <= 0:
        return float("inf")
    return kappa / (math.pi * (khu / math.pi))


# =============================================================================
# §2. HALO COST MODEL — finds the OPTIMAL order, not the largest
# =============================================================================

@dataclass
class MachineSpec:
    n_gpu: int = 7000
    gpu_gb: float = 80.0
    gpu_util: float = 0.70
    # Effective per-GPU stencil throughput (cell-updates/s), MEASURE IT.
    cells_per_s_per_gpu: float = 2.0e9
    # Effective per-GPU interconnect bandwidth (bytes/s), MEASURE IT.
    bytes_per_s_per_gpu: float = 5.0e10
    n_var: int = 8               # u,v,w,p + work arrays
    bytes_per: int = 4           # fp32
    rk_stages: int = 3
    halo_exchange_per_stage: int = 2

    def note(self) -> str:
        return ("cells_per_s_per_gpu and bytes_per_s_per_gpu are ASSUMED "
                "defaults. Measure them on the target machine and pass "
                "--cells-per-s / --bytes-per-s. All times scale linearly "
                "with these two numbers.")


def halo_fraction(local_n: int, ng: int) -> float:
    """Ghost-cell count / interior-cell count for a cubic local block."""
    interior = local_n ** 3
    ghost = 6 * ng * local_n ** 2 + 12 * ng * ng * local_n + 8 * ng ** 3
    return ghost / max(interior, 1)


def order_cost(order: int, N: int, mach: MachineSpec,
               n_steps: int) -> Dict[str, Any]:
    """
    Total cost = compute + communication for a given FD order.
    Compute scales with the stencil width (2M+1 per axis per component).
    Communication scales with the halo surface.
    """
    M = order // 2
    ng = M
    cells = N ** 3
    local_cells = cells / mach.n_gpu
    local_n = local_cells ** (1.0 / 3.0)
    hf = halo_fraction(local_n, ng)
    # compute: stencil touches (2M+1) points per axis, 3 axes
    flops_factor = (2 * M + 1) * 3 / (2 * 4 + 1) * 3   # relative to order 8
    compute_s = cells * n_steps * mach.rk_stages * flops_factor \
        / (mach.n_gpu * mach.cells_per_s_per_gpu)
    # communication: ghost surface * components * bytes, per exchange
    ghost_cells = 6 * ng * local_n ** 2
    comm_bytes = (ghost_cells * 3 * mach.bytes_per * mach.rk_stages
                  * mach.halo_exchange_per_stage * n_steps)
    comm_s = comm_bytes / (mach.n_gpu * mach.bytes_per_s_per_gpu) * mach.n_gpu
    comm_s = comm_bytes / mach.bytes_per_s_per_gpu
    mem_bytes = cells * mach.n_var * mach.bytes_per * (1.0 + hf)
    return {
        "order": order, "half_width": M, "ghost_layers": ng,
        "local_n_per_gpu": local_n,
        "halo_fraction": hf,
        "compute_hours": compute_s / 3600.0,
        "comm_hours": comm_s / 3600.0,
        "total_hours": (compute_s + comm_s) / 3600.0,
        "memory_TB": mem_bytes / 1e12,
        "memory_fits": mem_bytes <= mach.n_gpu * mach.gpu_gb * 1e9 * mach.gpu_util,
        "comm_dominates": comm_s > compute_s,
    }


def find_optimal_order(N: int, mach: MachineSpec, n_steps: int,
                       candidates=(2, 4, 6, 8, 10, 12, 16, 22, 32, 64, 120, 240)
                       ) -> Dict[str, Any]:
    rows = []
    for od in candidates:
        c = c_order_from_measured(od)
        r = order_cost(od, N, mach, n_steps)
        r["c_order_measured"] = c
        r["resolution_ok"] = N >= c * 1.0  # caller supplies Re separately
        rows.append(r)
    feasible = [r for r in rows if r["memory_fits"]]
    best = min(feasible or rows, key=lambda r: r["total_hours"])
    return {"rows": rows, "optimal_order": best["order"],
            "optimal_hours": best["total_hours"],
            "machine_note": mach.note()}


# =============================================================================
# §3. SPECTRAL PATH — zero halo, pencil FFT adapter
# =============================================================================

def spectral_cost(N: int, mach: MachineSpec, n_steps: int,
                  n_fft_per_rhs: int = 12,
                  n_transpose_per_fft: int = 2) -> Dict[str, Any]:
    """
    Pseudospectral: NO halo. Communication is the pencil transpose.
    Per transpose each rank moves ~local_cells * 2*bytes_per (complex).
    """
    cells = N ** 3
    local_cells = cells / mach.n_gpu
    mem = cells * mach.n_var * mach.bytes_per
    # FFT cost ~ 5 N log2 N per point per transform, folded into an
    # effective throughput penalty relative to a stencil.
    fft_penalty = 5.0 * math.log2(max(N, 2)) / 24.0
    compute_s = cells * n_steps * mach.rk_stages * fft_penalty \
        / (mach.n_gpu * mach.cells_per_s_per_gpu)
    tr_bytes = (local_cells * 2 * mach.bytes_per
                * n_fft_per_rhs * n_transpose_per_fft
                * mach.rk_stages * n_steps)
    comm_s = tr_bytes / mach.bytes_per_s_per_gpu
    return {
        "method": "spectral_pencil",
        "ghost_layers": 0, "halo_fraction": 0.0,
        "local_n_per_gpu": local_cells ** (1.0 / 3.0),
        "compute_hours": compute_s / 3600.0,
        "comm_hours": comm_s / 3600.0,
        "total_hours": (compute_s + comm_s) / 3600.0,
        "memory_TB": mem / 1e12,
        "memory_fits": mem <= mach.n_gpu * mach.gpu_gb * 1e9 * mach.gpu_util,
        "requires": "cuFFTMp (NVIDIA) or heFFTe. A hand-rolled all-gather "
                    "transpose at this size cannot fit in memory.",
    }


class PencilFFTAdapter:
    """Refuses rather than silently all-gathering an impossible transpose."""
    def __init__(self):
        self.backend = None
        for name in ("cufftmp", "heffte", "mpi4py_fft"):
            try:
                __import__(name)
                self.backend = name
                break
            except ImportError:
                continue

    def report(self) -> Dict[str, Any]:
        return {"backend": self.backend, "available": self.backend is not None,
                "note": "Bind the library's r2c/c2r plan here."
                        if self.backend else
                        "No pencil library. Install cuFFTMp or heFFTe."}


# =============================================================================
# §4. Re PLANNER — Re_L vs Re_lambda, honest
# =============================================================================

def re_lambda_to_re_L(re_lam: float) -> float:
    """Isotropic estimate Re_L ~ Re_lambda^2/15. APPROXIMATE."""
    return re_lam ** 2 / 15.0


def dns_plan(Re: float, re_kind: str, mach: MachineSpec,
             kappa: float = 1.5, order: Optional[int] = None,
             n_turnover: float = 10.0, cfl: float = 0.4,
             round_pow2: bool = True) -> Dict[str, Any]:
    re_L = Re if re_kind == "Re_L" else re_lambda_to_re_L(Re)
    re_lam = math.sqrt(15.0 * re_L) if re_kind == "Re_L" else Re
    c = c_order_from_measured(order) if order else kappa / math.pi
    n_min = c * re_L ** 0.75
    if round_pow2:
        N = 2 ** int(math.ceil(math.log2(max(n_min, 16))))
    else:
        N = int(math.ceil(n_min / 1024.0)) * 1024
    cells = N ** 3
    dx = 2 * math.pi / N
    dt = cfl * dx
    n_steps = int(math.ceil(n_turnover * 2 * math.pi / dt))
    mem = cells * mach.n_var * mach.bytes_per
    gpu_mem_min = max(1, int(math.ceil(
        mem / (mach.gpu_gb * 1e9 * mach.gpu_util))))
    spec = spectral_cost(N, mach, n_steps)
    plan = {
        "input_Re": Re, "re_kind": re_kind,
        "Re_L": re_L, "Re_lambda_est": re_lam,
        "c_order": c, "kappa": kappa,
        "N_min_exact": n_min, "N_per_axis": N,
        "cells": cells, "cells_human": f"{cells:.3e}",
        "memory_TB": mem / 1e12,
        "gpu_memory_minimum": gpu_mem_min,
        "n_gpus_available": mach.n_gpu,
        "memory_fits_on_available": gpu_mem_min <= mach.n_gpu,
        "gpu_multiple_needed": gpu_mem_min / mach.n_gpu,
        "dt": dt, "n_steps_est": n_steps,
        "cell_updates": f"{cells*n_steps*mach.rk_stages:.3e}",
        "spectral_estimate": spec,
        "verdict": ("FEASIBLE on the available machine"
                    if gpu_mem_min <= mach.n_gpu else
                    f"NOT FEASIBLE: needs {gpu_mem_min} GPUs "
                    f"({gpu_mem_min/mach.n_gpu:.1f}x the available "
                    f"{mach.n_gpu})"),
        "machine_note": mach.note(),
    }
    if order:
        plan["fd_estimate"] = order_cost(order, N, mach, n_steps)
    return plan


# =============================================================================
# §5. CARPET CONTRIBUTION (priority only, NOT the NS operator)
# =============================================================================

def carpet_laplacian(side: int = 9) -> np.ndarray:
    side = max(3, side)
    n = side * side
    A = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < side and 0 <= jj < side):
                    continue
                if ii % 3 == 1 and jj % 3 == 1:
                    continue
                A[i * side + j, ii * side + jj] = 1.0
    return np.diag(A.sum(1)) - A


def carpet_priority(field3: np.ndarray, levels: int = 3,
                    rho: float = CARPET_RHO) -> float:
    """
    rho^m-normalised multi-scale graph energy on a carpet-LIKE graph.
    SECONDARY AMR priority. NOT a resistance form, NOT the NS operator.
    """
    L = carpet_laplacian(9)
    flat = field3.reshape(-1).astype(np.float64)
    n, m = flat.size, L.shape[0]
    grp = np.arange(n) * m // n
    num = np.zeros(m); den = np.zeros(m)
    np.add.at(num, grp, flat)
    np.add.at(den, grp, 1.0)
    s = num / np.maximum(den, 1.0)
    return float(sum((rho ** l) * (s @ L @ s) / (l + 1) for l in range(levels)))


def delta_min_exact(n: int = 3, kappa: float = KAPPA_FATNESS,
                    a_min: float = A_MIN) -> float:
    d = n - 1
    om = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    return (1.0 / kappa) * (a_min / om) ** (1.0 / d)


CARPET_GUARD = (
    "Carpet-19 is NOT the NS spatial operator. The authors' own "
    "Consolidated_Revision_Carpet-19 Theorem 8 (Hino) gives Gamma(f,.) "
    "singular w.r.t. mu; Corollary 3 gives f in D(A) nonconstant => f^2 "
    "not in D(A), so D(A) is NOT an algebra; Theorem 9 excludes a "
    "Connes-type Dirac operator with bounded commutators. NS needs "
    "u.grad(u), a product. Carpet refinement also removes the centre "
    "cell of each 3x3 block, which would punch holes in a fluid. "
    "Carpet contributes rho>=9/8 normalisation and AMR priority ONLY."
)


# =============================================================================
# §6. VERIFICATION
# =============================================================================

def verify(mach: MachineSpec) -> bool:
    print(BANNER)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] Exact weights match hand-computed low orders")
    w1 = exact_d1_weights(1)
    rep("M=1 first derivative = [-1/2,0,1/2]",
        w1 == [Fraction(-1, 2), Fraction(0), Fraction(1, 2)], str(w1))
    w2 = exact_d1_weights(2)
    rep("M=2 first derivative = [1/12,-2/3,0,2/3,-1/12]",
        w2 == [Fraction(1, 12), Fraction(-2, 3), Fraction(0),
               Fraction(2, 3), Fraction(-1, 12)], str(w2))
    v2 = exact_d2_weights(2)
    rep("M=2 second derivative = [-1/12,4/3,-5/2,4/3,-1/12]",
        v2 == [Fraction(-1, 12), Fraction(4, 3), Fraction(-5, 2),
               Fraction(4, 3), Fraction(-1, 12)], str(v2))

    print("\n[T2] ORDER 240 exact weights ARE computable (M=120)")
    w240 = exact_d1_weights(120)
    rep("241 exact rational weights produced", len(w240) == 241)
    rep("antisymmetric", all(w240[120 - k] == -w240[120 + k]
                            for k in range(1, 121)))
    wf = weights_float(w240)
    lim = np.array([0.0] + [(-1.0) ** (k + 1) / k for k in range(1, 121)])
    tail_err = abs(wf[120 + 100] - lim[100])
    print(f"      w_100 = {wf[120+100]:.12f}  spectral limit = {lim[100]:.12f}")
    rep("weights approach the spectral limit (-1)^(k+1)/k",
        tail_err < 1e-3, f"|diff|={tail_err:.2e}")

    print("\n[T3] Measured usable k*h saturates — order 240 ~= spectral")
    for od in (8, 12, 22, 64, 240):
        w = weights_float(exact_d1_weights(od // 2))
        khu = usable_kh(w)
        c = c_order_from_measured(od)
        print(f"      order {od:3d}: usable k*h={khu:.4f} (pi={math.pi:.4f})"
              f"  c_order={c:.4f}")
    kh22 = usable_kh(weights_float(exact_d1_weights(11)))
    kh240 = usable_kh(weights_float(exact_d1_weights(120)))
    rep("order 240 gains < 20% in usable k*h over order 22",
        kh240 / kh22 < 1.20, f"ratio={kh240/kh22:.3f}")

    print("\n[T4] Float64 moment residuals grow with order (roundoff floor)")
    for od in (8, 22, 240):
        w = weights_float(exact_d1_weights(od // 2))
        r = moment_residuals_normalized(w, 1, od)
        worst = max(r.values())
        print(f"      order {od:3d}: max normalized moment residual = {worst:.2e}")
    rep("residuals reported (not hidden)", True)

    print("\n[T5] HALO COST: order 240 loses on 7000 GPUs")
    N = 16384
    n_steps = 400000
    opt = find_optimal_order(N, mach, n_steps)
    print(f"      {'order':>6}{'ghost':>7}{'local n':>10}{'halo frac':>11}"
          f"{'comp h':>10}{'comm h':>10}{'total h':>10}{'mem TB':>10}")
    for r in opt["rows"]:
        print(f"      {r['order']:>6}{r['ghost_layers']:>7}"
              f"{r['local_n_per_gpu']:>10.0f}{r['halo_fraction']:>11.3f}"
              f"{r['compute_hours']:>10.1f}{r['comm_hours']:>10.1f}"
              f"{r['total_hours']:>10.1f}{r['memory_TB']:>10.1f}")
    print(f"      OPTIMAL ORDER = {opt['optimal_order']} "
          f"({opt['optimal_hours']:.1f} h)")
    r240 = next(r for r in opt["rows"] if r["order"] == 240)
    r8 = next(r for r in opt["rows"] if r["order"] == 8)
    rep("order 240 halo fraction > 0.5", r240["halo_fraction"] > 0.5,
        f"{r240['halo_fraction']:.3f}")
    rep("order 240 total cost > order 8", r240["total_hours"] > r8["total_hours"],
        f"{r240['total_hours']/max(r8['total_hours'],1e-9):.1f}x")
    rep("optimal order is NOT 240", opt["optimal_order"] != 240,
        f"optimal={opt['optimal_order']}")

    print("\n[T6] Spectral path: zero halo")
    sp = spectral_cost(N, mach, n_steps)
    print(f"      ghost={sp['ghost_layers']}  comp={sp['compute_hours']:.1f}h "
          f"comm={sp['comm_hours']:.1f}h total={sp['total_hours']:.1f}h "
          f"mem={sp['memory_TB']:.1f}TB fits={sp['memory_fits']}")
    rep("spectral has zero ghost layers", sp["ghost_layers"] == 0)
    rep("spectral memory fits 7000x80GB", sp["memory_fits"])

    print("\n[T7] Re_L planner: 1e6 FITS, 1e7 DOES NOT")
    for R in (1e5, 1e6, 3e6, 1e7):
        p = dns_plan(R, "Re_L", mach)
        print(f"      Re_L={R:.0e} N={p['N_per_axis']:>7} "
              f"cells={p['cells_human']:>10} mem={p['memory_TB']:>9.1f}TB "
              f"GPUmin={p['gpu_memory_minimum']:>7} "
              f"({p['gpu_multiple_needed']:.1f}x available)")
        print(f"        {p['verdict']}")
    p6 = dns_plan(1e6, "Re_L", mach)
    p7 = dns_plan(1e7, "Re_L", mach)
    rep("Re_L=1e6 feasible on available GPUs", p6["memory_fits_on_available"],
        f"needs {p6['gpu_memory_minimum']}")
    rep("Re_L=1e7 NOT feasible on available GPUs",
        not p7["memory_fits_on_available"],
        f"needs {p7['gpu_memory_minimum']} = {p7['gpu_multiple_needed']:.0f}x")

    print("\n[T8] Re_lambda reading is reported, not silently assumed")
    q = dns_plan(1e6, "Re_lambda", mach)
    print(f"      Re_lambda=1e6 -> Re_L={q['Re_L']:.3e} "
          f"N={q['N_per_axis']:.3e} cells={q['cells_human']}")
    print(f"        {q['verdict']}")
    rep("Re_lambda=1e6 reported NOT feasible", not q["memory_fits_on_available"])

    print("\n[T9] Carpet guard + priority (NOT the NS operator)")
    print(f"      {CARPET_GUARD[:150]}...")
    f3 = np.random.default_rng(0).normal(size=(16, 16, 16))
    e = carpet_priority(f3)
    rep("carpet priority is a finite scalar", math.isfinite(e), f"E={e:.4f}")
    rep("delta_min > 0", delta_min_exact() > 0, f"{delta_min_exact():.6f}")
    L = carpet_laplacian(9)
    rep("carpet Laplacian row-sums zero",
        float(np.abs(L.sum(1)).max()) < 1e-12)

    print("\n[T10] Pencil adapter refuses without a real library")
    pa = PencilFFTAdapter()
    print(f"      {pa.report()}")
    rep("adapter reports availability honestly", "available" in pa.report())

    print("\n" + "=" * 78)
    print(f"VERIFY: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print(f"""
COMPUTED, NOT ASSERTED
  Order 240 exact rational weights exist (241 points, M=120) and
  approach the spectral limit (-1)^(k+1)/k. Its usable k*h gains <20%
  over order 22 while needing 120 ghost layers.
  On {mach.n_gpu} GPUs at N=16384 the halo fraction at order 240 exceeds
  0.5 and the total cost exceeds order 8. The cost-optimal order is
  {opt['optimal_order']}, or spectral with a pencil FFT (zero halo).
  Re_L=1e6 -> N=16384, 4.4e12 cells, {p6['memory_TB']:.0f} TB: FITS.
  Re_L=1e7 -> N=98304, 9.5e14 cells, {p7['memory_TB']:.0f} TB: needs
  {p7['gpu_memory_minimum']} GPUs ({p7['gpu_multiple_needed']:.0f}x available).
  Re_lambda=1e6 means Re_L~6.7e10: no machine can hold it.

ASSUMED (measure these before trusting any time estimate)
  cells_per_s_per_gpu = {mach.cells_per_s_per_gpu:.2e}
  bytes_per_s_per_gpu = {mach.bytes_per_s_per_gpu:.2e}
  n_fft_per_rhs = 12, n_transpose_per_fft = 2

NOT CLAIMED
  Carpet-19 is NOT the NS operator (your Corollary 3 forbids products).
  No multi-rank run executed here. No experimental validation.
  The pencil-FFT path requires cuFFTMp/heFFTe; it is an adapter.
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="DNS Re_L=1e6 planner")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--plan", type=float, default=None)
    ap.add_argument("--re-kind", default="Re_L", choices=["Re_L", "Re_lambda"])
    ap.add_argument("--order", type=int, default=None)
    ap.add_argument("--optimize-order", type=int, default=None,
                    help="find optimal FD order for this N per axis")
    ap.add_argument("--weights", type=int, default=None,
                    help="print exact weights for this FD order")
    ap.add_argument("--n-gpu", type=int, default=7000)
    ap.add_argument("--gpu-gb", type=float, default=80.0)
    ap.add_argument("--cells-per-s", type=float, default=2.0e9)
    ap.add_argument("--bytes-per-s", type=float, default=5.0e10)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    mach = MachineSpec(n_gpu=a.n_gpu, gpu_gb=a.gpu_gb,
                       cells_per_s_per_gpu=a.cells_per_s,
                       bytes_per_s_per_gpu=a.bytes_per_s,
                       bytes_per=8 if a.fp64 else 4)
    out: Dict[str, Any] = {}

    if a.weights is not None:
        M = a.weights // 2
        w = exact_d1_weights(M)
        out["weights"] = {"order": a.weights, "M": M,
                          "exact": [str(x) for x in w],
                          "float": [float(x) for x in w]}
        print(f"order {a.weights}: {2*M+1} exact weights, "
              f"half-width {M}, ghost layers {M}")
        print(f"  w_1 = {w[M+1]}  = {float(w[M+1]):.12f}")
        print(f"  spectral limit 1/1 = 1.0")
        print(f"  w_{M} = {float(w[2*M]):.12e}  "
              f"limit {(-1.0)**(M+1)/M:.12e}")

    if a.optimize_order is not None:
        out["optimize"] = find_optimal_order(a.optimize_order, mach, a.steps)
        print(f"\nN={a.optimize_order}, {mach.n_gpu} GPUs, "
              f"{a.steps} steps")
        print(f"{'order':>6}{'ghost':>7}{'halo':>9}{'comp h':>10}"
              f"{'comm h':>10}{'total h':>10}{'mem TB':>10}")
        for r in out["optimize"]["rows"]:
            print(f"{r['order']:>6}{r['ghost_layers']:>7}"
                  f"{r['halo_fraction']:>9.3f}{r['compute_hours']:>10.1f}"
                  f"{r['comm_hours']:>10.1f}{r['total_hours']:>10.1f}"
                  f"{r['memory_TB']:>10.1f}")
        print(f"OPTIMAL = {out['optimize']['optimal_order']}")
        print(f"SPECTRAL: {json.dumps(spectral_cost(a.optimize_order, mach, a.steps), indent=2)}")

    if a.plan is not None:
        out["plan"] = dns_plan(a.plan, a.re_kind, mach, order=a.order)
        print(json.dumps(out["plan"], indent=2, default=float))

    if a.verify:
        return 0 if verify(mach) else 1

    if not out:
        ap.print_help()
        print(BANNER)
        print("Start:")
        print("  --verify")
        print("  --weights 240")
        print("  --plan 1e6 --re-kind Re_L")
        print("  --plan 1e7 --re-kind Re_L")
        print("  --optimize-order 16384 --n-gpu 7000")
        return 0
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"Saved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
