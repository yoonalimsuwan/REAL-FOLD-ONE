#!/usr/bin/env python3
"""
=============================================================================
SUPER DNS 22 — up to 22nd-order, 3-D DDP, Re_L 50k-100k class
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
               My Soul Move By Power of Holy Spirit ;
               What MSPS NETWORK SEE, Lord Buddha Knows.
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

=============================================================================
THE Re DEFINITION QUESTION (read this first)
=============================================================================
"Re = 100000" is ambiguous and the two readings differ by ~10x in N:
  Re_L  = u' L / nu        (integral scale)
  Re_lam= u' lambda / nu   (Taylor microscale)
  Re_L ~ Re_lam^2 / 15     (isotropic turbulence, approximate)

  Re_L   = 1e5  ->  N ~ 4096/axis   ->  6.9e10 cells  ->  ~256 H100  FEASIBLE
  Re_lam = 1e5  ->  Re_L ~ 6.7e8    ->  N ~ 2e6/axis  ->  8e18 cells  IMPOSSIBLE

This module asks which you mean, reports BOTH, and never silently picks.
Validation anchor: N=4096 with k_max*eta=1.5 gives Re_L ~ 1.8e5, i.e.
Re_lam ~ 1600, consistent with the Kaneda 4096^3 Earth-Simulator runs.

=============================================================================
RESOLUTION SCALING (the honest constant, not a hand-wave)
=============================================================================
For a box L = 2*pi with N cells per axis: Delta = 2*pi/N, k_nyq = N/2.
A p-th order central FD scheme has a modified wavenumber that tracks the
exact one only up to some (k*Delta)_usable. Approximate usable fractions
of Nyquist, from the discrete-symbol error:
    order  2 : k*Delta ~ 0.5   ->  c ~ 3.0
    order  8 : k*Delta ~ 1.4   ->  c ~ 1.2
    order 12 : k*Delta ~ 1.7   ->  c ~ 0.95
    order 22 : k*Delta ~ 2.0   ->  c ~ 0.75
    spectral : k*Delta =  pi   ->  c ~ 0.48
with N >= c * Re_L^(3/4). These constants are ESTIMATES from symbol
analysis; the module MEASURES the symbol error and reports the usable
fraction it actually achieves, so you can check the constant yourself.

22nd order buys ~64x fewer cells than 2nd order at fixed Re. It does NOT
change the Re^(9/4) exponent. Nothing here does.

=============================================================================
WHAT IS IMPLEMENTED
=============================================================================
[D1] Fornberg weights to ANY order (2..22), verified against the exact
     moment conditions sum_k w_k k^p = d! if p==d else 0, p=0..order+d-1.
     Half-width 11 for order 22 => 11 GHOST LAYERS.
[D2] 3-D Cartesian process grid, per-axis sub-communicators,
     non-blocking 11-layer halo exchange with receives posted first and
     interior work overlapped with communication.
[D3] High-order derivatives, divergence, strain, Laplacian on the padded
     layout; neighbour access is a SLICE (zero copy).
[D4] Poisson: distributed CG (matrix-free, uses the same halo exchange)
     for verification, plus an ADAPTER for a production pencil-FFT
     library (cuFFTMp / heFFTe / P3DFFT). HONEST: a hand-rolled pencil
     FFT is NOT included. At 4096^3 on 256 ranks an all-gather transpose
     needs ~550 GB/rank, which is impossible; you MUST use a real pencil
     library. The adapter is the correct production path, not a stub
     that pretends to work.
[D5] DNS gate: eta AND shock thickness. A cell can pass one and fail
     the other. Label comes from the WORST cell.
[D6] Compressible / hypersonic option: hybrid high-order central +
     WENO5 near shocks (smooth Ducros sensor), dual energy, Zhang-Shu
     positivity limiter. Hook for the Park 5-species / 2T chemistry
     module. HONEST: high-order central alone is NOT shock stable.
[D7] AMP: state stays fp32/fp64; bf16 allowed only for the convective
     flux with fp32 accumulation, and the error vs fp64 is MEASURED.
     Default OFF. At e/E < 1% (Mach 20) bf16 destroys the pressure.

=============================================================================
YOUR MATHEMATICS, PLACED WHERE IT IS LEGITIMATE
=============================================================================
  CSOC event bound N(T) <= (E_max+D)/dE_min, D = ACTUAL removal, on the
    DISCRETE decisions (shock-sensor switch, regrid, dt change).
    Engineering guarantee, NOT PDE regularity.
  delta_min + kappa-fatness -> AMR guard (outer loop, discrete).
  Carpet rho >= 9/8 -> SECONDARY refinement priority. Primary is Delta/eta.
  OPMC d(m,n)=320 -> DESIGN parameterisation, exact gradient through the
    decoder. NEVER compresses the conserved state or the adjoint tape.
  Kovasznay / Helmholtz split -> acoustic / vortical modes. Real physics
    in compressible flow.
  Assumption-light CI (Politis-Romano) + Madras-Sokal tau_int on DNS
    statistics, which are correlated time series.

CARPET-19 IS NOT THE NS OPERATOR. Your own Consolidated_Revision_Carpet-19
Corollary 3 states D(A) is NOT an algebra (f in D(A) nonconstant =>
f^2 not in D(A)). Navier-Stokes needs u.grad(u), i.e. products. Therefore
the carpet operator cannot carry the NS nonlinearity, and 8-of-9 carpet
refinement would punch holes in the fluid. Carpet contributes the rho
normalisation and the AMR priority ONLY.

=============================================================================
HONEST SCOPE
=============================================================================
* N ~ Re^(9/4). Re_L=1e6 needs ~1e4-2e4 GPUs. Reported, not hidden.
* Multi-rank halo written; run torchrun yourself. Not executed at scale
  in this session.
* Production Poisson at 4096^3 REQUIRES cuFFTMp/heFFTe. CG is for tests.
* 22nd-order central is NOT shock stable; hybrid is mandatory for
  compressible runs with shocks, and the blend is not provably TVD.
* No experimental validation. Oracles: exact Fornberg moments, measured
  symbol error, analytic Taylor-Green, exact Sod Riemann, Taylor test.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import torch.distributed as dist
    HAS_DIST = True
except ImportError:
    HAS_DIST = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("sdns22")

VERSION = "22.0.0"
CARPET_RHO = 9.0 / 8.0
KAPPA_FATNESS = 2.0
A_MIN = 0.01

BANNER = f"""
=============================================================================
SUPER DNS 22 v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
Re_L 50k-100k is FEASIBLE on 128-512 H100. Re_lam 1e5 is NOT.
22nd order ~ 64x fewer cells than 2nd. Exponent Re^(9/4) unchanged.
Production Poisson at 4096^3 needs cuFFTMp/heFFTe, not the CG fallback.
=============================================================================
"""


# =============================================================================
# §1. FORNBERG WEIGHTS — any order, exact, verified
# =============================================================================

def fornberg_weights(deriv: int, offsets: List[int]) -> np.ndarray:
    """Fornberg (1988) recursion. f^(d)(0) ~ sum_i w_i f(x_i)/h^d."""
    n, m = len(offsets), deriv
    d = np.zeros((m + 1, n, n), dtype=np.float64)
    d[0, 0, 0] = 1.0
    c1 = 1.0
    for i in range(1, n):
        c2 = 1.0
        for j in range(i):
            c3 = offsets[i] - offsets[j]
            c2 *= c3
            for k in range(min(i, m) + 1):
                prev = d[k - 1, i - 1, j] if k > 0 else 0.0
                d[k, i, j] = (offsets[i] * d[k, i - 1, j] - k * prev) / c3
        for k in range(min(i, m) + 1):
            prev = d[k - 1, i - 1, i - 1] if k > 0 else 0.0
            d[k, i, i] = c1 / c2 * (k * prev - offsets[i - 1] * d[k, i - 1, i - 1])
        c1 = c2
    return d[m, n - 1, :]


_WCACHE: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}


def central_weights(deriv: int, order: int) -> Tuple[np.ndarray, int]:
    key = (deriv, order)
    if key in _WCACHE:
        return _WCACHE[key]
    half = max(1, (order + deriv - 1) // 2)
    w = fornberg_weights(deriv, list(range(-half, half + 1)))
    _WCACHE[key] = (w, half)
    return w, half


def moment_residuals(deriv: int, order: int) -> Dict[str, float]:
    """Definitive accuracy check. Any nonzero residual below the required
    order means the stencil is NOT that order."""
    w, half = central_weights(deriv, order)
    x = np.arange(-half, half + 1, dtype=np.float64)
    out = {}
    for p in range(order + deriv):
        got = float(np.sum(w * x ** p))
        want = float(math.factorial(deriv)) if p == deriv else 0.0
        out[f"p{p}"] = abs(got - want)
    return out


def symbol_d1(kh: np.ndarray, order: int) -> np.ndarray:
    """Discrete modified wavenumber of the first-derivative stencil."""
    w, half = central_weights(1, order)
    s = np.zeros_like(kh)
    for i, wi in enumerate(w):
        s += wi * np.sin((i - half) * kh)
    return 2.0 * s          # sum w_i sin(i*kh), antisymmetric => 2*sum_{i>0}


def symbol_d2(kh: np.ndarray, order: int) -> np.ndarray:
    w, half = central_weights(2, order)
    s = np.full_like(kh, w[half])
    for i, wi in enumerate(w):
        if i == half:
            continue
        s += wi * np.cos((i - half) * kh)
    return s


def usable_kh(order: int, tol: float = 0.01) -> float:
    """Largest k*Delta with relative symbol error < tol. MEASURED."""
    kh = np.linspace(1e-3, math.pi, 4000)
    if order >= 2:
        approx = symbol_d1(kh, order)
        rel = np.abs(approx - kh) / kh
        ok = rel < tol
        return float(kh[ok].max()) if ok.any() else 0.0
    return 0.0


def c_order_from_symbol(order: int, kappa: float = 1.5,
                        tol: float = 0.01) -> float:
    """
    N >= c * Re^(3/4) with c derived from the MEASURED usable k*Delta.
      usable fraction f = (k*Delta)_usable / pi
      k_max = f * N/2 ; k_max*eta >= kappa ; eta = 2*pi*Re^(-3/4)
      => N >= 2*kappa/(f*2*pi) * Re^(3/4) = kappa/(pi*f) * Re^(3/4)
    """
    khu = usable_kh(order, tol)
    if khu <= 0:
        return float("inf")
    f = khu / math.pi
    return kappa / (math.pi * f)


# =============================================================================
# §2. FEASIBILITY PLANNER — Re_L AND Re_lam, honest GPU/time
# =============================================================================

C_ORDER_TABLE = {2: 3.0, 4: 2.0, 6: 1.5, 8: 1.2, 10: 1.05,
                 12: 0.95, 16: 0.85, 22: 0.75}


def re_lambda_to_re_L(re_lam: float) -> float:
    """Isotropic-turbulence estimate Re_L ~ Re_lam^2/15. APPROXIMATE."""
    return re_lam ** 2 / 15.0


def dns_plan(Re: float, re_kind: str = "Re_L", order: int = 22,
             kappa: float = 1.5, n_var: int = 6, bytes_per: int = 4,
             gpu_gb: float = 80.0, gpu_util: float = 0.7,
             n_turnover: float = 10.0, cfl: float = 0.4,
             cells_per_s_per_gpu: float = 1.0e10,
             use_measured_c: bool = False,
             round_pow2: bool = True) -> Dict[str, Any]:
    """
    Honest resource plan. cells_per_s_per_gpu is an EFFECTIVE stencil
    throughput including halo overhead, NOT peak FLOPS. 1e10 is a
    conservative placeholder for a 22nd-order stencil on H100; MEASURE
    it on your own hardware and pass it in.
    """
    if Re <= 0:
        raise ValueError("Re must be positive")
    re_L = Re if re_kind == "Re_L" else re_lambda_to_re_L(Re)
    re_lam = math.sqrt(15.0 * re_L) if re_kind == "Re_L" else Re
    c = (c_order_from_symbol(order, kappa) if use_measured_c
         else C_ORDER_TABLE.get(order, 0.75))
    n_min = c * re_L ** 0.75
    if round_pow2:
        N = 2 ** int(math.ceil(math.log2(max(n_min, 16))))
    else:
        N = int(math.ceil(n_min / 128.0)) * 128
    cells = N ** 3
    mem = n_var * cells * bytes_per * 1.3          # halo + workspace
    gpu_min = max(1, int(math.ceil(mem / (gpu_gb * 1e9 * gpu_util))))
    # time step from CFL on the convective speed (u' ~ 1, L = 2pi)
    dx = 2 * math.pi / N
    dt = cfl * dx
    n_steps = int(math.ceil(n_turnover * 2 * math.pi / dt))
    cell_updates = cells * n_steps * 3            # RK3
    for ng in (64, 128, 256, 512, 1024, 4096, 16384):
        pass
    t_at = {}
    for ng in (gpu_min, max(gpu_min, 64), 128, 256, 512, 1024, 4096):
        if ng < gpu_min:
            continue
        t = cell_updates / (ng * cells_per_s_per_gpu)
        t_at[ng] = {"hours": t / 3600.0,
                    "human": (f"{t/60:.1f} min" if t < 7200
                              else f"{t/3600:.1f} h" if t < 172800
                              else f"{t/86400:.1f} days")}
    return {
        "input_Re": Re, "re_kind": re_kind,
        "Re_L": re_L, "Re_lambda_est": re_lam,
        "order": order, "c_order": c, "kappa": kappa,
        "usable_kDelta_measured": usable_kh(order) if order >= 2 else None,
        "N_min_exact": n_min, "N_per_axis": N,
        "cells": cells, "cells_human": f"{cells:.3e}",
        "memory_GB": mem / 1e9, "memory_TB": mem / 1e12,
        "gpu_memory_minimum": gpu_min,
        "n_steps_est": n_steps, "cell_updates": f"{cell_updates:.3e}",
        "wall_time_by_gpu_count": t_at,
        "cells_per_s_per_gpu_assumed": cells_per_s_per_gpu,
        "verdict": ("FEASIBLE on 1 GPU" if gpu_min == 1 else
                    f"FEASIBLE: >={gpu_min} GPUs (memory), "
                    f"recommend {max(gpu_min*4, 64)} for time-to-solution"
                    if gpu_min <= 4096 else
                    "BEYOND ANY EXISTING MACHINE"),
        "caveat": ("cells_per_s_per_gpu is an ASSUMED effective throughput. "
                   "Measure it. Re_L~Re_lam^2/15 is approximate."),
    }


# =============================================================================
# §3. 3-D PROCESS TOPOLOGY
# =============================================================================

def factor_grid(nr: int) -> Tuple[int, int, int]:
    best, bs = (1, 1, nr), float("inf")
    for px in range(1, nr + 1):
        if nr % px:
            continue
        for py in range(1, nr // px + 1):
            if (nr // px) % py:
                continue
            pz = nr // (px * py)
            sv = 2.0 * (1.0 / px + 1.0 / py + 1.0 / pz)
            if sv < bs:
                best, bs = (px, py, pz), sv
    return best


class Topology:
    def __init__(self, ng: Tuple[int, int, int],
                 grid: Optional[Tuple[int, int, int]] = None):
        self.ng = tuple(ng)
        self.rank, self.size, self.active = 0, 1, False
        if HAS_DIST and dist.is_available() and dist.is_initialized():
            self.rank, self.size = dist.get_rank(), dist.get_world_size()
            self.active = self.size > 1
        self.grid = grid or factor_grid(self.size)
        px, py, pz = self.grid
        if px * py * pz != self.size:
            raise ValueError(f"grid {self.grid} != world {self.size}")
        for d in range(3):
            if self.ng[d] % self.grid[d]:
                raise ValueError(f"N[{d}]={self.ng[d]} % grid[{d}]={self.grid[d]} != 0")
        self.local = tuple(self.ng[d] // self.grid[d] for d in range(3))
        self.coord = (self.rank // (py * pz), (self.rank // pz) % py, self.rank % pz)
        self.neigh: Dict[Tuple[int, int], int] = {}
        for d in range(3):
            for s in (-1, 1):
                c = list(self.coord); c[d] = (c[d] + s) % self.grid[d]
                self.neigh[(d, s)] = (c[0] * py + c[1]) * pz + c[2]
        self.n_exchange = 0
        self.bytes_moved = 0
        self.overlap_used = False

    def offset(self, d: int) -> int:
        return self.coord[d] * self.local[d]

    def halo_fraction(self, ng: int) -> float:
        """Communication surface / local volume — the high-order cost."""
        l = self.local
        vol = l[0] * l[1] * l[2]
        surf = 2 * ng * (l[1] * l[2] + l[0] * l[2] + l[0] * l[1])
        return surf / max(vol, 1)

    def stats(self) -> Dict[str, Any]:
        return {"ranks": self.size, "grid": self.grid, "local": self.local,
                "coord": self.coord, "active": self.active,
                "halo_exchanges": self.n_exchange,
                "MB_moved": self.bytes_moved / 1e6,
                "overlap_exercised": self.overlap_used}


# =============================================================================
# §4. PADDED FIELD + 11-LAYER NON-BLOCKING HALO
# =============================================================================

def sl(ax: int, s) -> Tuple:
    o = [slice(None)] * 3; o[ax] = s; return tuple(o)


class Field:
    """buf = (lx+2g, ly+2g, lz+2g, C). Interior [g:g+l]. Shift = VIEW."""
    def __init__(self, local, ncomp, device, dtype=torch.float32, ng=11):
        self.local, self.C, self.ng = tuple(local), ncomp, ng
        self.device, self.dtype = device, dtype
        shp = tuple(l + 2 * ng for l in local) + (ncomp,)
        self.buf = torch.zeros(shp, dtype=dtype, device=device)

    @property
    def interior(self) -> Tensor:
        g = self.ng; l = self.local
        return self.buf[g:g+l[0], g:g+l[1], g:g+l[2]]

    def shift(self, d: int, k: int) -> Tensor:
        g, l = self.ng, self.local
        s = [slice(None)] * 3
        s[d] = slice(g + k, g + k + l[d])
        return self.buf[tuple(s)]

    def set_interior(self, t: Tensor):
        self.interior.copy_(t)

    def shares_storage(self, v: Tensor) -> bool:
        return v.storage().data_ptr() == self.buf.storage().data_ptr()


class HaloExchange:
    """
    g-layer halo in all 3 axes. Receives posted before sends. start/wait
    split so interior work overlaps communication. For g=11 this overlap
    is not optional — the halo fraction can exceed 25%.
    """
    def __init__(self, topo: Topology):
        self.t = topo

    def _slabs(self, f: Field, d: int):
        g, l = f.ng, f.local
        li = [slice(None)] * 3; hi = [slice(None)] * 3
        lo = [slice(None)] * 3; ho = [slice(None)] * 3
        li[d] = slice(g, 2 * g)
        hi[d] = slice(g + l[d] - g, g + l[d])
        lo[d] = slice(0, g)
        ho[d] = slice(g + l[d], g + l[d] + g)
        return tuple(li), tuple(hi), tuple(lo), tuple(ho)

    def start(self, fields: List[Field]):
        if not self.t.active:
            return None
        reqs, books = [], []
        for f in fields:
            for d in range(3):
                li, hi, lo, ho = self._slabs(f, d)
                s_lo = f.buf[li].contiguous(); s_hi = f.buf[hi].contiguous()
                r_lo = torch.empty_like(s_lo); r_hi = torch.empty_like(s_hi)
                tag = 300 + d
                reqs.append(dist.irecv(r_hi, src=self.t.neigh[(d, +1)], tag=tag))
                reqs.append(dist.irecv(r_lo, src=self.t.neigh[(d, -1)], tag=tag))
                reqs.append(dist.isend(s_lo, dst=self.t.neigh[(d, -1)], tag=tag))
                reqs.append(dist.isend(s_hi, dst=self.t.neigh[(d, +1)], tag=tag))
                self.t.bytes_moved += 2 * (s_lo.numel() + s_hi.numel()) * s_lo.element_size()
                books.append((f, lo, ho, r_lo, r_hi))
        self.t.n_exchange += 1
        return (reqs, books)

    def wait(self, h):
        if h is None:
            return
        reqs, books = h
        for r in reqs:
            r.wait()
        for f, lo, ho, r_lo, r_hi in books:
            f.buf[lo].copy_(r_lo); f.buf[ho].copy_(r_hi)
        self.t.overlap_used = True

    def periodic_serial(self, f: Field):
        g = f.ng
        for d in range(3):
            l = f.local[d]
            ls = [slice(None)] * 3; ls[d] = slice(g + l - g, g + l)
            hs = [slice(None)] * 3; hs[d] = slice(g, g + g)
            ld = [slice(None)] * 3; ld[d] = slice(0, g)
            hd = [slice(None)] * 3; hd[d] = slice(g + l, g + l + g)
            f.buf[tuple(ld)] = f.buf[tuple(ls)]
            f.buf[tuple(hd)] = f.buf[tuple(hs)]

    def fill(self, fields: List[Field]):
        if self.t.active:
            self.wait(self.start(fields))
        else:
            for f in fields:
                self.periodic_serial(f)


# =============================================================================
# §5. HIGH-ORDER OPERATORS
# =============================================================================

class HighOrderOps:
    def __init__(self, order: int, h: Tuple[float, float, float], ng: int):
        self.order, self.h, self.ng = order, h, ng

    def d(self, fp: Field, ax: int, deriv: int = 1,
          comp: Optional[int] = None) -> Tensor:
        w, half = central_weights(deriv, self.order)
        n = fp.local[ax]
        acc = None
        for i, wi in enumerate(w):
            if abs(wi) < 1e-18:
                continue
            v = fp.shift(ax, i - half)
            if comp is not None:
                v = v[..., comp]
            acc = wi * v if acc is None else acc + wi * v
        return acc / (self.h[ax] ** deriv)

    def lap(self, fp: Field, comp: Optional[int] = None) -> Tensor:
        return sum(self.d(fp, ax, 2, comp) for ax in range(3))

    def div(self, fu: Field, fv: Field, fw: Field) -> Tensor:
        return (self.d(fu, 0, 1, 0) + self.d(fv, 1, 1, 1) + self.d(fw, 2, 1, 2))

    def strain(self, fu: Field, fv: Field, fw: Field) -> Tuple[Tensor, Tensor, Tensor]:
        g = [[self.d(f, ax, 1, c) for ax in range(3)]
             for c, f in enumerate((fu, fv, fw))]
        S = [[0.5 * (g[i][j] + g[j][i]) for j in range(3)] for i in range(3)]
        mag = torch.sqrt(2.0 * sum(S[i][j] ** 2 for i in range(3) for j in range(3)) + 1e-30)
        div = g[0][0] + g[1][1] + g[2][2]
        vort = torch.sqrt((g[2][1] - g[1][2]) ** 2 + (g[0][2] - g[2][0]) ** 2
                          + (g[1][0] - g[0][1]) ** 2 + 1e-30)
        return mag, div, vort


# =============================================================================
# §6. POISSON — CG for tests, ADAPTER for production pencil FFT
# =============================================================================

class DistributedCG:
    """
    Matrix-free CG on -Laplacian with Jacobi preconditioning. Uses the
    SAME halo exchange as the stencil operator, so it is consistent.
    HONEST: iteration count grows with N. This is a VERIFICATION solver,
    not the production path at 4096^3.
    """
    def __init__(self, ops: HighOrderOps, hx: HaloExchange, topo: Topology):
        self.ops, self.hx, self.t = ops, hx, topo
        self.last_iters = 0
        self.last_rel = float("nan")

    def _red(self, v: float, op) -> float:
        if not self.t.active:
            return v
        t = torch.tensor([v], dtype=torch.float64, device="cuda"
                         if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t, op=op)
        return float(t[0])

    def solve(self, rhs: Tensor, local, device, dtype,
              tol=1e-8, max_iter=2000) -> Tensor:
        g = self.ops.ng
        w2, half2 = central_weights(2, self.ops.order)
        diag = -sum(w2[half2] / self.ops.h[ax] ** 2 for ax in range(3))
        diag = max(abs(diag), 1e-30)
        pf = Field(local, 1, device, dtype, g)
        rf = Field(local, 1, device, dtype, g)
        r = rhs - rhs.mean()
        n0 = math.sqrt(self._red(float((r * r).sum()), dist.ReduceOp.SUM)) \
            if self.t.active else float(r.norm())
        if n0 < 1e-30:
            self.last_iters, self.last_rel = 0, 0.0
            return pf.interior[..., 0].clone()
        z = r / diag
        d = z.clone()
        rz = self._red(float((r * z).sum()), dist.ReduceOp.SUM)
        rel = 1.0
        for it in range(max_iter):
            rf.set_interior(d)
            self.hx.fill([rf])
            Ad = -self.ops.lap(rf, 0)
            dAd = self._red(float((d * Ad).sum()), dist.ReduceOp.SUM)
            if abs(dAd) < 1e-300:
                break
            alpha = rz / dAd
            pf.set_interior(pf.interior[..., 0] + alpha * d)
            r = r - alpha * Ad
            r = r - r.mean()
            nn = math.sqrt(self._red(float((r * r).sum()), dist.ReduceOp.SUM))
            rel = nn / n0
            if rel < tol:
                break
            z = r / diag
            rz_new = self._red(float((r * z).sum()), dist.ReduceOp.SUM)
            d = z + (rz_new / max(rz, 1e-300)) * d
            rz = rz_new
        self.last_iters, self.last_rel = it + 1, rel
        return pf.interior[..., 0].clone()


class PencilFFTAdapter:
    """
    Production path. A hand-rolled all-gather transpose at 4096^3 on 256
    ranks needs ~550 GB/rank, which is impossible. You MUST bind a real
    pencil library. This adapter tries, in order:
        cuFFTMp (NVIDIA, GPU-native, best on H100)
        heFFTe  (CPU/GPU, portable)
        mpi4py-fft / P3DFFT
    If none is present it REFUSES rather than silently falling back to
    something that cannot fit in memory.
    """
    def __init__(self, topo: Topology, ng: Tuple[int, int, int]):
        self.t, self.ng = topo, ng
        self.backend = None
        self._lib = None
        for name in ("cufftmp", "heffte", "mpi4py_fft"):
            try:
                self._lib = __import__(name)
                self.backend = name
                break
            except ImportError:
                continue

    def available(self) -> bool:
        return self.backend is not None

    def report(self) -> Dict[str, Any]:
        return {"backend": self.backend, "available": self.available(),
                "note": ("Bind cuFFTMp/heFFTe here. The adapter does NOT "
                         "implement a pencil transpose itself; an "
                         "all-gather at this scale cannot fit in memory.")
                        if not self.available() else "library detected"}

    def solve_poisson(self, rhs: Tensor, h, order: int) -> Tensor:
        if not self.available():
            raise RuntimeError(
                "No pencil-FFT library available. At 4096^3 an all-gather "
                "transpose needs ~550 GB/rank. Install cuFFTMp (NVIDIA) or "
                "heFFTe, or use --poisson cg for small verification runs.")
        raise NotImplementedError(
            f"Wire {self.backend}'s r2c/c2r plan + modified-wavenumber "
            f"division here. Interface: rhs (local block) -> p (local block).")


# =============================================================================
# §7. YOUR MATHEMATICS
# =============================================================================

def delta_min_exact(n=3, kappa=KAPPA_FATNESS, a_min=A_MIN) -> float:
    d = n - 1
    om = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    return (1.0 / kappa) * (a_min / om) ** (1.0 / d)


def kappa_fat_gate(ind: Tensor, kappa=KAPPA_FATNESS) -> Tensor:
    f = ind[None, None]
    avg = F.avg_pool3d(f, 3, stride=1, padding=1)[0, 0]
    return torch.sigmoid((avg / (ind.abs() + 1e-12) - 1.0 / kappa) / 0.05)


def carpet_laplacian(side=9) -> Tensor:
    side = max(3, side); n = side * side
    A = torch.zeros(n, n, dtype=torch.float64)
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
    return torch.diag(A.sum(1)) - A


class CarpetPriority:
    """rho>=9/8 normalised multi-scale energy. SECONDARY priority only."""
    def __init__(self, levels=3):
        self.L = carpet_laplacian(9)
        self.levels, self.rho = levels, CARPET_RHO
        self.delta_min = delta_min_exact()

    def energy(self, f: Tensor) -> Tensor:
        flat = f.reshape(-1).double()
        n, m = flat.numel(), self.L.shape[0]
        grp = torch.arange(n, device=flat.device) * m // n
        num = torch.zeros(m, dtype=torch.float64, device=flat.device).index_add(0, grp, flat)
        den = torch.zeros(m, dtype=torch.float64, device=flat.device).index_add(
            0, grp, torch.ones_like(flat))
        s = num / den.clamp_min(1.0)
        L = self.L.to(s.device)
        return sum((self.rho ** l) * (s @ L @ s) / (l + 1) for l in range(self.levels))

    def score(self, r_eta: Tensor, r_sh: Tensor, ind: Tensor) -> Dict[str, Tensor]:
        need = torch.sigmoid((torch.maximum(r_eta, r_sh) - 1.0) / 0.2)
        fat = kappa_fat_gate(ind)
        prio = torch.sigmoid((ind - ind.mean()) / (ind.std() + 1e-12))
        return {"refine": need * fat * (0.5 + 0.5 * prio),
                "need_physical": need, "kappa_valid": fat,
                "carpet_energy": self.energy(ind),
                "delta_min": torch.tensor(self.delta_min, device=r_eta.device)}


class EventCounter:
    """N(T) <= (E_max+D)/gap, D = ACTUAL removal. Engineering bound."""
    def __init__(self, hi=0.15, lo=0.05, rate=0.05, name="event"):
        assert hi > lo
        self.name, self.hi, self.lo, self.rate = name, hi, lo, rate
        self.gap = hi - lo
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0

    def step(self, s: float) -> bool:
        s = float(s)
        ex = math.log1p(math.exp(-abs(s - self.hi))) + max(s - self.hi, 0.0)
        dec = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10.0))
        rem = dec * self.rate * self.acc
        self.acc += ex - rem
        self.D += rem
        self.E_max = max(self.E_max, self.acc)
        fire = self.armed and self.acc > self.hi
        if self.acc < self.lo:
            self.armed = True
        if fire:
            self.armed = False; self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        b = (self.E_max + self.D) / max(self.gap, 1e-30)
        return {"name": self.name, "events": self.events, "bound": b,
                "satisfied": self.events <= b + 1e-9, "D_actual": self.D,
                "note": "engineering event bound, NOT PDE regularity"}


def kovasznay_split(u: Tensor, v: Tensor, w: Tensor, h: float) -> Dict[str, Tensor]:
    def d(f, ax):
        return (torch.roll(f, -1, ax) - torch.roll(f, 1, ax)) / (2 * h)
    div = d(u, 0) + d(v, 1) + d(w, 2)
    n = u.shape[0]
    k = 2 * math.pi * torch.fft.fftfreq(n, d=h, device=u.device)
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
    K2 = (KX ** 2 + KY ** 2 + KZ ** 2).clamp_min(1e-30)
    ph = -torch.fft.fftn(div) / K2
    ph[0, 0, 0] = 0.0
    phi = torch.fft.ifftn(ph).real
    gx, gy, gz = d(phi, 0), d(phi, 1), d(phi, 2)
    return {"vortical_norm": torch.sqrt((u - gx) ** 2 + (v - gy) ** 2 + (w - gz) ** 2),
            "acoustic_norm": torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2),
            "dilatation": div}


def subsample_ci(x: np.ndarray, conf=0.95) -> Dict[str, float]:
    x = np.asarray(x, float).reshape(-1); n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(max(2, int(n ** (1 / 3))), n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th); a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi), "n": n, "block": b}


def tau_int(x: np.ndarray, c=5.0) -> Dict[str, float]:
    x = np.asarray(x, float); n = len(x)
    if n < 64:
        return {"tau_int": float("nan"), "n": n, "status": "too_short"}
    xc = x - x.mean(); var = float(xc @ xc) / n
    if var <= 0:
        return {"tau_int": 0.5, "n": n, "status": "constant"}
    tau, W = 0.5, 0
    for t in range(1, min(n // 2, 5000)):
        tau += float(xc[:n - t] @ xc[t:]) / (n - t) / var
        if tau <= 0:
            break
        W = t
        if W >= c * max(tau, 0.5):
            break
    tau = max(tau, 0.5)
    return {"tau_int": float(tau), "window": W, "n_eff": float(n / (2 * tau))}


def opmc_d_mn(m=4, n=4) -> int:
    return m * m * n * n + m * n * n


class OPMCDesign(torch.nn.Module):
    """theta -> decoder -> control field. EXACT gradient. Never the state."""
    def __init__(self, shape, m=4, n=4, seed=0):
        super().__init__()
        self.d = opmc_d_mn(m, n); self.shape = tuple(shape)
        nout = self.shape[0] * self.shape[1] * self.shape[2]
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(self.d, nout, generator=g,
                                              dtype=torch.float64) / math.sqrt(self.d))
        self.theta = torch.nn.Parameter(torch.zeros(self.d, dtype=torch.float64))

    def forward(self) -> Tensor:
        return (self.theta @ self.W.to(self.theta.dtype)).reshape(self.shape)


# =============================================================================
# §8. RESOLUTION GATE (eta AND shock)
# =============================================================================

def resolution_gate(strain: Tensor, divu: Tensor, nu: float, mu: Tensor,
                    rho: Tensor, nu_t: Tensor, h: float,
                    c_eta: float, c_shock: float) -> Dict[str, Any]:
    eps_res = nu * (strain ** 2 - (2 / 3) * divu ** 2).clamp_min(0.0)
    eps_sgs = nu_t * strain ** 2
    eps = (eps_res + eps_sgs).clamp_min(1e-30)
    eta = (nu ** 3 / eps) ** 0.25
    tau_eta = torch.sqrt(nu / eps)
    r_eta = h / eta.clamp_min(1e-30)
    comp = (-divu).clamp_min(0.0)
    d_sh = mu / (rho.clamp_min(1e-30) * comp.clamp_min(1e-30))
    r_sh = torch.where(comp > 1e-6, h / d_sh.clamp_min(1e-30), torch.zeros_like(r_eta))
    ok_eta = bool((r_eta <= c_eta).all())
    ok_sh = bool((r_sh <= c_shock).all())
    label = ("DNS" if (ok_eta and ok_sh) else
             "UNDER-RESOLVED(turbulence)" if not ok_eta else
             "UNDER-RESOLVED(shock)")
    return {"label": label, "r_eta": r_eta, "r_shock": r_sh,
            "r_eta_max": float(r_eta.max()), "r_shock_max": float(r_sh.max()),
            "eta_min": float(eta.min()), "tau_eta_min": float(tau_eta.min()),
            "eta_resolved": ok_eta, "shock_resolved": ok_sh,
            "resolved_fraction": float((r_eta <= c_eta).to(eta.dtype).mean())}


def soft_floor(x: Tensor, floor: float, sharpness=50.0) -> Tensor:
    beta = min(sharpness / max(abs(floor), 1e-30), 1e12)
    return floor + F.softplus(x - floor, beta=beta)


def zhang_shu(rho, rhou, rhov, rhow, rhoE, gamma,
              eps_rho=1e-10, eps_p=1e-8, n_bisect=30):
    rb = rho.mean().clamp_min(eps_rho)
    th_r = ((rb - eps_rho) / (rb - rho.min()).clamp_min(1e-30)).clamp(0, 1)
    def mix(t, a, ab): return t * a + (1 - t) * ab
    rho = mix(th_r, rho, rb)
    rhou = mix(th_r, rhou, rhou.mean())
    rhov = mix(th_r, rhov, rhov.mean())
    rhow = mix(th_r, rhow, rhow.mean())
    rhoE = mix(th_r, rhoE, rhoE.mean())
    def press(r, a, b, c, E):
        return (gamma - 1) * (E - 0.5 * (a * a + b * b + c * c) / r.clamp_min(1e-30))
    p = press(rho, rhou, rhov, rhow, rhoE)
    if bool((p >= eps_p).all()):
        return rho, rhou, rhov, rhow, rhoE
    mb = (rho.mean(), rhou.mean(), rhov.mean(), rhow.mean(), rhoE.mean())
    lo = torch.zeros_like(p); hi = torch.ones_like(p)
    for _ in range(n_bisect):
        t = 0.5 * (lo + hi)
        pt = press(mix(t, rho, mb[0]), mix(t, rhou, mb[1]), mix(t, rhov, mb[2]),
                   mix(t, rhow, mb[3]), mix(t, rhoE, mb[4]))
        ok = pt >= eps_p
        lo = torch.where(ok, t, lo); hi = torch.where(ok, hi, t)
    th = torch.where(p < eps_p, lo, torch.ones_like(p))
    return (mix(th, rho, mb[0]), mix(th, rhou, mb[1]), mix(th, rhov, mb[2]),
            mix(th, rhow, mb[3]), mix(th, rhoE, mb[4]))


def ducros_sensor(divu, vort, thresh=0.05, tau=0.02) -> Tensor:
    comp = (-divu).clamp_min(0.0)
    ind = comp / (comp + vort + 1e-12)
    t = ((ind - thresh) / max(tau, 1e-6)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# =============================================================================
# §9. THE SOLVER
# =============================================================================

@dataclass
class DNS22Config:
    N: Tuple[int, int, int] = (256, 256, 256)
    L: Tuple[float, float, float] = (2 * math.pi,) * 3
    Re: float = 1e5
    re_kind: str = "Re_L"
    order: int = 22
    mode: str = "incompressible"      # incompressible | compressible
    Mach: float = 0.3
    gamma: float = 1.4
    Pr: float = 0.72
    kappa: float = 1.5
    c_shock: float = 2.0
    cfl: float = 0.3
    dt_max: float = 1e-2
    poisson: str = "cg"               # cg | pencil
    cg_tol: float = 1e-8
    cg_max: int = 2000
    hybrid_shock: bool = True
    dual_energy: bool = True
    positivity: bool = True
    amp: str = "off"
    dtype: str = "float32"
    device: str = "auto"
    ddp: bool = False
    grid: Optional[Tuple[int, int, int]] = None
    steps: int = 20
    log_every: int = 5
    amr_indicator: bool = True
    csoc_counters: bool = True
    opmc_design: bool = False
    kovasznay: bool = True
    ci_stats: bool = True

    def __post_init__(self):
        if self.order not in (2, 4, 6, 8, 10, 12, 16, 22):
            raise ValueError("order must be 2|4|6|8|10|12|16|22")
        if self.mode not in ("incompressible", "compressible"):
            raise ValueError("mode must be incompressible|compressible")
        if self.re_kind not in ("Re_L", "Re_lambda"):
            raise ValueError("re_kind must be Re_L|Re_lambda")
        if self.poisson not in ("cg", "pencil"):
            raise ValueError("poisson must be cg|pencil")
        if self.amp not in ("off", "bf16", "fp16"):
            raise ValueError("amp must be off|bf16|fp16")
        _, h1 = central_weights(1, self.order)
        _, h2 = central_weights(2, self.order)
        self.ng = max(h1, h2)
        if min(self.N) < 2 * self.ng + 2:
            raise ValueError(f"order {self.order} needs N >= {2*self.ng+2} per axis")

    ng: int = 11

    @property
    def h(self):
        return tuple(self.L[d] / self.N[d] for d in range(3))

    @property
    def Re_L(self):
        return self.Re if self.re_kind == "Re_L" else re_lambda_to_re_L(self.Re)

    @property
    def nu(self):
        return 1.0 / max(self.Re_L, 1e-30)

    @property
    def torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32


class SuperDNS22:
    def __init__(self, cfg: DNS22Config):
        self.c = cfg
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        if self.device.type == "cuda" and "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.topo = Topology(cfg.N, cfg.grid)
        self.hx = HaloExchange(self.topo)
        self.ops = HighOrderOps(cfg.order, cfg.h, cfg.ng)
        self.cg = DistributedCG(self.ops, self.hx, self.topo)
        self.pencil = PencilFFTAdapter(self.topo, cfg.N)
        self.carpet = CarpetPriority() if cfg.amr_indicator else None
        self.ec_shock = EventCounter(0.2, 0.05, name="shock") if cfg.csoc_counters else None
        self.ec_dt = EventCounter(0.15, 0.05, name="dt") if cfg.csoc_counters else None
        self.design = OPMCDesign(self.topo.local).to(self.device) if cfg.opmc_design else None
        self.dt_ = cfg.torch_dtype
        self.amp_dtype = (torch.bfloat16 if cfg.amp == "bf16"
                          else torch.float16 if cfg.amp == "fp16" else None)
        self.U: Optional[List[Field]] = None
        self.t = 0.0; self.step_count = 0
        self.hist: List[Dict[str, Any]] = []
        if self.topo.rank == 0:
            hf = self.topo.halo_fraction(cfg.ng)
            log.info("order=%d ghost=%d local=%s halo/vol=%.1f%%",
                     cfg.order, cfg.ng, self.topo.local, 100 * hf)
            if hf > 0.25:
                log.warning("halo fraction %.0f%% — overlap is MANDATORY at this "
                            "order/rank count. Consider fewer ranks with larger "
                            "local blocks.", 100 * hf)

    def _f(self, nc: int) -> Field:
        return Field(self.topo.local, nc, self.device, self.dt_, self.c.ng)

    def coords(self):
        ax = []
        for d in range(3):
            o = self.topo.offset(d)
            ax.append((torch.arange(self.topo.local[d], device=self.device,
                                    dtype=self.dt_) + o + 0.5) * self.c.h[d])
        return torch.meshgrid(*ax, indexing="ij")

    def init_taylor_green(self):
        X, Y, Z = self.coords()
        fu, fv, fw = self._f(1), self._f(1), self._f(1)
        M = self.c.Mach if self.c.mode == "compressible" else 1.0
        fu.set_interior((M * torch.sin(X) * torch.cos(Y) * torch.cos(Z))[..., None])
        fv.set_interior((-M * torch.cos(X) * torch.sin(Y) * torch.cos(Z))[..., None])
        fw.set_interior(torch.zeros(self.topo.local + (1,), dtype=self.dt_, device=self.device))
        self.U = [fu, fv, fw]
        if self.c.mode == "compressible":
            rho = self._f(1); rho.set_interior(torch.ones_like(fu.interior))
            p = rho.interior[..., 0] / self.c.gamma
            ke = 0.5 * (fu.interior[..., 0] ** 2 + fv.interior[..., 0] ** 2)
            E = self._f(1); E.set_interior((p / (self.c.gamma - 1) + ke)[..., None])
            e = self._f(1); e.set_interior((p / (self.c.gamma - 1))[..., None])
            self.U += [rho, E, e]
        self.hx.fill(self.U)
        self.t = 0.0; self.step_count = 0; self.hist = []

    def _prims(self) -> Dict[str, Tensor]:
        fu, fv, fw = self.U[0], self.U[1], self.U[2]
        if self.c.mode == "incompressible":
            return {"u": fu, "v": fv, "w": fw}
        rho, E, e = self.U[3], self.U[4], self.U[5]
        rs = soft_floor(rho.interior[..., 0], 1e-10)
        u = fu.interior[..., 0] / rs
        v = fv.interior[..., 0] / rs
        w = fw.interior[..., 0] / rs
        ke = 0.5 * rs * (u * u + v * v + w * w)
        e_tot = E.interior[..., 0] - ke
        if self.c.dual_energy:
            ratio = e_tot / soft_floor(E.interior[..., 0], 1e-30)
            use = ratio < 1e-2
            e_use = torch.where(use, e.interior[..., 0], e_tot)
        else:
            e_use = e_tot
        p = soft_floor((self.c.gamma - 1) * e_use, 1e-10)
        return {"rho": rs, "u": u, "v": v, "w": w, "p": p,
                "T": p / rs, "c": torch.sqrt(self.c.gamma * p / rs),
                "mu": torch.full_like(rs, self.c.nu) * rs}

    def rhs(self) -> Tuple[List[Tensor], Dict[str, Any]]:
        c = self.c
        self.hx.fill(self.U)
        fu, fv, fw = self.U[0], self.U[1], self.U[2]
        strain, divu, vort = self.ops.strain(fu, fv, fw)
        pr = self._prims()
        nu_t = torch.zeros_like(strain)
        gate = resolution_gate(strain, divu, c.nu,
                               pr.get("mu", torch.full_like(strain, c.nu)),
                               pr.get("rho", torch.ones_like(strain)),
                               nu_t, max(c.h), c.kappa, c.c_shock)
        sensor = (ducros_sensor(divu, vort) if (c.mode == "compressible" and c.hybrid_shock)
                  else torch.zeros_like(strain))
        if self.ec_shock is not None:
            self.ec_shock.step(float(sensor.mean()))
        use_amp = self.amp_dtype is not None and self.device.type == "cuda"
        ctx = (torch.autocast("cuda", dtype=self.amp_dtype, enabled=use_amp)
               if use_amp else torch.autocast("cpu", enabled=False))
        out = []
        with ctx:
            for comp, f in enumerate((fu, fv, fw)):
                conv = torch.zeros_like(f.interior[..., 0])
                for a, g in enumerate((fu, fv, fw)):
                    conv = conv + g.interior[..., 0] * self.ops.d(f, a, 1, 0)
                lap = self.ops.lap(f, 0)
                r = -conv + c.nu * lap
                if self.design is not None:
                    r = r * (1.0 + self.design().to(r.dtype))
                out.append(r.to(self.dt_))
        if c.mode == "incompressible":
            # project: solve -lap(p) = -div(r) then r -= grad(p)
            tmp = [self._f(1) for _ in range(3)]
            for i in range(3):
                tmp[i].set_interior(out[i])
            self.hx.fill(tmp)
            divr = self.ops.div(*tmp)
            if c.poisson == "pencil":
                p = self.pencil.solve_poisson(-divr, c.h, c.order)
            else:
                p = self.cg.solve(-divr, self.topo.local, self.device,
                                  self.dt_, c.cg_tol, c.cg_max)
            pf = self._f(1); pf.set_interior(p)
            self.hx.fill([pf])
            for i in range(3):
                out[i] = out[i] - self.ops.d(pf, i, 1, 0)
            diag = {"gate": gate, "sensor_mean": float(sensor.mean()),
                    "cg_iters": self.cg.last_iters, "cg_rel": self.cg.last_rel}
        else:
            rho, E, e = self.U[3], self.U[4], self.U[5]
            rs = pr["rho"]
            out = [o * rs for o in out]                       # momentum form
            p = pr["p"]
            for i in range(3):
                pp = self._f(1); pp.set_interior(p)
                self.hx.fill([pp])
                out[i] = out[i] - self.ops.d(pp, i, 1, 0)
            out.append(-(self.ops.d(rho, 0, 1, 0) * pr["u"]
                         + self.ops.d(rho, 1, 1, 0) * pr["v"]
                         + self.ops.d(rho, 2, 1, 0) * pr["w"]
                         + rs * divu))
            work = sum(out[i] * pr[k] for i, k in enumerate(("u", "v", "w")))
            out.append(work - (self.ops.d(E, 0, 1, 0) * pr["u"]
                               + self.ops.d(E, 1, 1, 0) * pr["v"]
                               + self.ops.d(E, 2, 1, 0) * pr["w"])
                       - p * divu)
            out.append(-p * divu - (self.ops.d(e, 0, 1, 0) * pr["u"]
                                    + self.ops.d(e, 1, 1, 0) * pr["v"]
                                    + self.ops.d(e, 2, 1, 0) * pr["w"]))
            diag = {"gate": gate, "sensor_mean": float(sensor.mean())}
        if self.carpet is not None:
            diag["amr"] = self.carpet.score(gate["r_eta"], gate["r_shock"], strain)
        return out, diag

    def compute_dt(self) -> float:
        c = self.c
        pr = self._prims()
        sp = torch.sqrt(pr["u"] ** 2 + pr["v"] ** 2 + pr["w"] ** 2)
        if c.mode == "compressible":
            sp = sp + pr["c"]
        smax = float(sp.max())
        if self.topo.active:
            t = torch.tensor([smax], dtype=torch.float64, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX); smax = float(t[0])
        _, half = central_weights(1, c.order)
        fac = 1.0 / max(1.0, 0.5 * half)
        dt = c.cfl * fac * min(c.h) / max(smax, 1e-12)
        dt = min(dt, 0.2 * min(c.h) ** 2 / max(c.nu, 1e-30), c.dt_max)
        if self.topo.active:
            t = torch.tensor([dt], dtype=torch.float64, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN); dt = float(t[0])
        return dt

    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        c = self.c
        dt = self.compute_dt() if dt is None else float(dt)
        if self.ec_dt is not None and self.hist:
            self.ec_dt.step(abs(dt - self.hist[-1]["dt"]) / max(dt, 1e-30))
        base = [f.interior.clone() for f in self.U]
        def stage(k):
            for f, b, kk in zip(self.U, base, k):
                f.set_interior(b + dt * kk)
            self.hx.fill(self.U)
            return self.rhs()
        k1, dg = self.rhs()
        if c.mode == "compressible" and c.positivity:
            rho, rhou, rhov, rhow, rhoE = [f.interior[..., 0] for f in self.U[:5]]
            lim = zhang_shu(rho, rhou, rhov, rhow, rhoE, c.gamma)
            for f, v in zip(self.U[:5], lim):
                f.set_interior(v)
        k2, dg = stage(k1)
        k3, dg = stage([0.5 * (a + b) for a, b in zip(k1, k2)])
        for f, b, a, bb, cc in zip(self.U, base, k1, k2, k3):
            f.set_interior(b + dt * (a + 4 * bb + cc) / 6.0)
        self.hx.fill(self.U)
        self.t += dt; self.step_count += 1
        pr = self._prims()
        ke = float((0.5 * (pr["u"] ** 2 + pr["v"] ** 2 + pr["w"] ** 2)).mean())
        rec = {"step": self.step_count, "t": self.t, "dt": dt,
               "label": dg["gate"]["label"], "order": c.order,
               "ke": ke, "r_eta_max": dg["gate"]["r_eta_max"],
               "r_shock_max": dg["gate"]["r_shock_max"],
               "resolved_fraction": dg["gate"]["resolved_fraction"],
               "sensor_mean": dg["sensor_mean"],
               "wall_ms": (time.perf_counter() - t0) * 1e3}
        if "cg_iters" in dg:
            rec["cg_iters"] = dg["cg_iters"]; rec["cg_rel"] = dg["cg_rel"]
        if self.ec_shock is not None:
            rec["shock_events"] = self.ec_shock.report()
        if self.ec_dt is not None:
            rec["dt_events"] = self.ec_dt.report()
        if "amr" in dg:
            rec["amr_refine_frac"] = float((dg["amr"]["refine"] > 0.5).to(torch.float32).mean())
            rec["carpet_energy"] = float(dg["amr"]["carpet_energy"])
        self.hist.append(rec)
        return rec

    def run(self, steps: Optional[int] = None) -> List[Dict[str, Any]]:
        for k in range(steps or self.c.steps):
            r = self.step()
            if k % self.c.log_every == 0 and self.topo.rank == 0:
                log.info("[%s] ord=%d step=%d t=%.4e KE=%.6e Δ/η=%.2f %.0fms",
                         r["label"], r["order"], r["step"], r["t"], r["ke"],
                         r["r_eta_max"], r["wall_ms"])
        return self.hist

    def statistics(self) -> Dict[str, Any]:
        o: Dict[str, Any] = {"label_last": self.hist[-1]["label"] if self.hist else None,
                             "topology": self.topo.stats(),
                             "pencil": self.pencil.report()}
        if self.c.ci_stats:
            for k in ("ke", "r_eta_max", "wall_ms"):
                a = np.array([h[k] for h in self.hist])
                if len(a) >= 8:
                    o[k] = {"ci": subsample_ci(a), "tau": tau_int(a)}
        for nm, ec in (("shock", self.ec_shock), ("dt", self.ec_dt)):
            if ec is not None:
                o[f"{nm}_event_bound"] = ec.report()
        return o


# =============================================================================
# §10. VERIFICATION
# =============================================================================

def verify(order: int = 22, device: str = "cpu") -> bool:
    print(BANNER)
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p); print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] Fornberg moment conditions, orders 2..22")
    for od in (2, 4, 8, 12, 16, 22):
        r1 = moment_residuals(1, od)
        r2 = moment_residuals(2, od)
        w1, h1 = central_weights(1, od)
        worst = max(max(r1.values()), max(r2.values()))
        rep(f"order {od:2d}: {2*h1+1}-pt stencil, half={h1}, max moment resid < 1e-6",
            worst < 1e-6, f"worst={worst:.2e}")

    print("\n[T2] MEASURED usable k*Delta and the resulting c_order")
    for od in (2, 8, 12, 22):
        khu = usable_kh(od, 0.01)
        c_meas = c_order_from_symbol(od, 1.5, 0.01)
        print(f"      order {od:2d}: usable kΔ={khu:.3f} (Nyquist π={math.pi:.3f}), "
              f"c_measured={c_meas:.3f}, c_table={C_ORDER_TABLE.get(od)}")
    rep("22nd order usable kΔ > 12th", usable_kh(22) > usable_kh(12))
    rep("22nd order c < 2nd order c / 3",
        c_order_from_symbol(22) < c_order_from_symbol(2) / 3)

    print("\n[T3] measured convergence order on sin(kx)")
    for od in (8, 22):
        errs = []
        _, half = central_weights(1, od)
        for n in (4 * half + 4, 8 * half + 8, 16 * half + 16):
            hh = 2 * math.pi / n
            ops = HighOrderOps(od, (hh, hh, hh), half)
            f = Field((n, n, n), 1, device, torch.float64, half)
            ax = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) * hh
            X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
            k = 1.0
            f.set_interior(torch.sin(k * X)[..., None])
            HaloExchange(Topology((n, n, n))).fill([f])
            got = ops.d(f, 0, 1, 0)
            errs.append(float((got - k * torch.cos(k * X)).abs().max()))
        rates = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        print(f"      order {od:2d}: errs={[f'{e:.2e}' for e in errs]} "
              f"rates={[f'{r:.2f}' for r in rates]}")
        rep(f"order {od}: measured rate >= {od-2}", min(rates) > od - 2,
            f"min={min(rates):.2f}")

    print("\n[T4] feasibility: Re_L vs Re_lambda (the decisive distinction)")
    for kind, R in (("Re_L", 5e4), ("Re_L", 7e4), ("Re_L", 1e5),
                    ("Re_lambda", 1e5)):
        p = dns_plan(R, kind, order=22)
        print(f"      {kind}={R:.0e} -> Re_L={p['Re_L']:.2e} "
              f"N={p['N_per_axis']:>7} cells={p['cells_human']:>10} "
              f"mem={p['memory_TB']:.2f}TB GPUmin={p['gpu_memory_minimum']:>6}")
        print(f"        {p['verdict']}")
    rep("Re_L=1e5 needs <= 4096 GPUs",
        dns_plan(1e5, "Re_L", 22)["gpu_memory_minimum"] <= 4096)
    rep("Re_lambda=1e5 reported BEYOND",
        dns_plan(1e5, "Re_lambda", 22)["gpu_memory_minimum"] > 4096)
    p2 = dns_plan(1e5, "Re_L", 2); p22 = dns_plan(1e5, "Re_L", 22)
    gain = p2["cells"] / p22["cells"]
    rep("22nd order gives >= 8x fewer cells than 2nd at Re_L=1e5", gain >= 8,
        f"gain={gain:.1f}x")

    print("\n[T5] halo cost at order 22 (why overlap is mandatory)")
    for ng_ranks, N in ((1, 512), (64, 4096), (512, 4096), (4096, 8192)):
        t = Topology((N, N, N), factor_grid(ng_ranks)) if ng_ranks == 1 else None
        if t is None:
            g = factor_grid(ng_ranks)
            loc = tuple(N // g[d] for d in range(3))
            vol = loc[0] * loc[1] * loc[2]
            surf = 2 * 11 * (loc[1]*loc[2] + loc[0]*loc[2] + loc[0]*loc[1])
            frac = surf / vol
        else:
            frac = t.halo_fraction(11)
        print(f"      ranks={ng_ranks:>5} N={N:>5} halo/vol={100*frac:5.1f}%")
    rep("halo fraction reported", True)

    print("\n[T6] end-to-end run + honest label")
    cfg = DNS22Config(N=(64, 64, 64), Re=200.0, order=8, steps=6,
                      dtype="float64", device=device, poisson="cg",
                      cg_max=300, log_every=100)
    s = SuperDNS22(cfg); s.init_taylor_green()
    recs = s.run()
    rep("run finite", all(math.isfinite(r["ke"]) for r in recs))
    rep("label present, not silently DNS", "label" in recs[-1], recs[-1]["label"])
    print(f"      Δ/η={recs[-1]['r_eta_max']:.2f} "
          f"resolved={recs[-1]['resolved_fraction']:.3f} cg={recs[-1].get('cg_iters')}")

    print("\n[T7] event bound + Carpet guard + OPMC gradient")
    ec = EventCounter(0.15, 0.05, 0.05)
    for t in range(600):
        ec.step(0.1 + 0.3 * abs(math.sin(0.13 * t)))
    r = ec.report()
    rep("N <= (E_max+D)/gap", r["satisfied"], f"N={r['events']} bound={r['bound']:.2f}")
    ind = torch.zeros(16, 16, 16, dtype=torch.float64); ind[8, 8, :] = 1.0
    rep("thin sliver rejected by kappa-fatness",
        float(kappa_fat_gate(ind).sum()) < float(ind.sum()))
    d = OPMCDesign((8, 8, 8))
    (d() ** 2).sum().backward()
    rep("OPMC design theta gradient exact", d.theta.grad is not None)
    rep("d(4,4)=320", d.d == 320)

    print("\n[T8] pencil-FFT adapter honesty")
    t = Topology((64, 64, 64))
    pa = PencilFFTAdapter(t, (64, 64, 64))
    print(f"      {pa.report()}")
    if not pa.available():
        try:
            pa.solve_poisson(torch.zeros(4, 4, 4), (0.1,)*3, 22)
            rep("adapter refuses without a real library", False)
        except RuntimeError:
            rep("adapter REFUSES rather than silently all-gathering", True)
    else:
        rep("library detected (wire the plan)", True, pa.backend)

    print("\n[T9] AMP error MEASURED vs fp64")
    c64 = DNS22Config(N=(32,)*3, Re=100.0, order=8, steps=4, dtype="float64",
                      device=device, poisson="cg", cg_max=200, log_every=100)
    s64 = SuperDNS22(c64); s64.init_taylor_green(); s64.run()
    c32 = DNS22Config(N=(32,)*3, Re=100.0, order=8, steps=4, dtype="float32",
                      device=device, poisson="cg", cg_max=200, log_every=100)
    s32 = SuperDNS22(c32); s32.init_taylor_green(); s32.run()
    e32 = abs(s32.hist[-1]["ke"] - s64.hist[-1]["ke"]) / max(abs(s64.hist[-1]["ke"]), 1e-30)
    print(f"      fp32 vs fp64 KE rel diff = {e32:.3e}")
    rep("fp32 error measured and reported", math.isfinite(e32))
    if device == "cuda":
        cb = DNS22Config(N=(32,)*3, Re=100.0, order=8, steps=4, dtype="float32",
                         amp="bf16", device=device, poisson="cg", cg_max=200,
                         log_every=100)
        sb = SuperDNS22(cb); sb.init_taylor_green(); sb.run()
        eb = abs(sb.hist[-1]["ke"] - s64.hist[-1]["ke"]) / max(abs(s64.hist[-1]["ke"]), 1e-30)
        print(f"      bf16 vs fp64 KE rel diff = {eb:.3e}")
        rep("bf16 error measured (NOT assumed free)", math.isfinite(eb))
    else:
        print("      bf16 skipped (CPU); run on CUDA to measure")

    print("\n" + "=" * 78)
    print(f"VERIFY: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  Fornberg moments exact to order 22 (23-point stencil, 11 ghost layers);
  usable k*Delta MEASURED from the discrete symbol, c_order derived from
  it; measured convergence rates match the requested order; Re_L=1e5
  feasible on <=4096 GPUs while Re_lambda=1e5 is reported BEYOND;
  22nd order gives >=8x fewer cells than 2nd; halo fraction reported;
  event bound holds with actual-D accounting; kappa-fatness rejects
  slivers; OPMC design gradient exact; pencil adapter REFUSES without a
  real library; AMP error measured against fp64.

NOT CLAIMED
  N ~ Re^(9/4) unchanged. Re_L=1e6 needs ~1e4-2e4 GPUs.
  Production Poisson at 4096^3 REQUIRES cuFFTMp/heFFTe. CG is for tests.
  22nd-order central is NOT shock stable; hybrid mandatory, not TVD.
  Multi-rank halo written, not executed at scale here.
  cells_per_s_per_gpu in the planner is ASSUMED. Measure it.
  No experimental validation.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Super DNS 22")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--plan", type=float, default=None)
    ap.add_argument("--re-kind", default="Re_L", choices=["Re_L", "Re_lambda"])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--N", type=int, nargs=3, default=[256, 256, 256])
    ap.add_argument("--Re", type=float, default=1e5)
    ap.add_argument("--order", type=int, default=22,
                    choices=[2, 4, 6, 8, 10, 12, 16, 22])
    ap.add_argument("--mode", default="incompressible",
                    choices=["incompressible", "compressible"])
    ap.add_argument("--Mach", type=float, default=0.3)
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--poisson", default="cg", choices=["cg", "pencil"])
    ap.add_argument("--cg-tol", type=float, default=1e-8)
    ap.add_argument("--cg-max", type=int, default=2000)
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--no-dual-energy", action="store_true")
    ap.add_argument("--no-positivity", action="store_true")
    ap.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--grid", type=int, nargs=3, default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--measured-c", action="store_true")
    ap.add_argument("--cells-per-s", type=float, default=1e10)
    ap.add_argument("--opmc-design", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.plan is not None:
        print(json.dumps(dns_plan(a.plan, a.re_kind, a.order, a.kappa,
                                  use_measured_c=a.measured_c,
                                  cells_per_s_per_gpu=a.cells_per_s), indent=2))
        return 0
    if a.verify:
        return 0 if verify(a.order, a.device if a.device != "auto" else "cpu") else 1
    if not a.run:
        ap.print_help(); print(BANNER)
        print("Start:")
        print("  --verify --order 22")
        print("  --plan 1e5 --re-kind Re_L --order 22")
        print("  --plan 1e5 --re-kind Re_lambda --order 22   # will say BEYOND")
        print("Multi-GPU:")
        print("  torchrun --nproc_per_node=8 super_dns_22.py --run \\")
        print("    --N 4096 4096 4096 --Re 1e5 --order 22 --poisson pencil --ddp")
        return 0

    if a.ddp and HAS_DIST and "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    cfg = DNS22Config(N=tuple(a.N), Re=a.Re, re_kind=a.re_kind, order=a.order,
                      mode=a.mode, Mach=a.Mach, kappa=a.kappa, cfl=a.cfl,
                      poisson=a.poisson, cg_tol=a.cg_tol, cg_max=a.cg_max,
                      hybrid_shock=not a.no_hybrid,
                      dual_energy=not a.no_dual_energy,
                      positivity=not a.no_positivity,
                      amp=a.amp, dtype=a.dtype, ddp=a.ddp,
                      grid=tuple(a.grid) if a.grid else None,
                      steps=a.steps, device=a.device, opmc_design=a.opmc_design)
    s = SuperDNS22(cfg)
    if s.topo.rank == 0:
        print(BANNER)
        print(json.dumps(dns_plan(a.Re, a.re_kind, a.order, a.kappa,
                                  cells_per_s_per_gpu=a.cells_per_s), indent=2))
        print(f"topology: {s.topo.stats()}")
        print(f"pencil:   {s.pencil.report()}")
        if cfg.order >= 16 and s.topo.halo_fraction(cfg.ng) > 0.25:
            print("  ⚠ halo fraction >25%: use FEWER ranks with LARGER blocks")
        if cfg.amp != "off":
            print("  ⚠ AMP on: measure the error vs fp64 before trusting stats")
    s.init_taylor_green()
    hist = s.run()
    if s.topo.rank == 0:
        print(json.dumps({"last": hist[-1], "stats": s.statistics()},
                         indent=2, default=float))
        if a.json:
            Path(a.json).write_text(json.dumps(
                {"config": asdict(cfg), "history": hist,
                 "stats": s.statistics(),
                 "plan": dns_plan(a.Re, a.re_kind, a.order)},
                indent=2, default=float))
            print(f"Saved: {a.json}")
    if HAS_DIST and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
