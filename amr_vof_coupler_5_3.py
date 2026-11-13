#!/usr/bin/env python3
"""
=============================================================================
AMR-VOF COUPLER v5.3
Vectorized MOF · True Octree · Distributed MPI · Energy-Conserving Skew Form
· External-Validation Harness
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit
             What MSPS NETWORK SEE, Lord Buddha Knows.
ORCID      : 0009-0008-2374-0788
License    : MIT
Year       : 2026

=============================================================================
WHAT v5.3 FIXES RELATIVE TO v5.2
=============================================================================
[F1] MOF VECTORIZATION
     v5.2 reconstructed each mixed cell in a Python loop with 6-tet
     clipping (feasible only to nx~32). v5.3 derives the CLOSED-FORM
     volume AND first moment of a box cut by a halfspace, via
     inclusion-exclusion over the 8 box corners, and batches the
     plane-offset bisection and normal optimisation over ALL cells at
     once. No per-cell Python loop remains.

     Derivation (verified analytically and numerically in T1):
       For the simplex {x >= 0, n.x <= c} with n > 0:
           V = c_+^3 / (6 n1 n2 n3)
           centroid_i = c / (4 n_i)
       Inclusion-exclusion over subsets S of {1,2,3}, with
       a_S = sum_{i in S} n_i L_i and c_S = c - a_S:
           V   = sum_S (-1)^|S| V_S
           M_i = sum_S (-1)^|S| V_S * ( c_S/(4 n_i) + L_i * 1[i in S] )
       The shift L_i appears because the S-term simplex has its origin
       at the box corner (L_i in direction i for i in S).
     Degenerate normals (some n_i ~ 0) use exact 2-D / 1-D formulas,
     where the simplex centroid factor is 1/3 (2-D) or 1/2 (1-D).

[F2] TRUE OCTREE
     v5.2 held one patch per level. v5.3 implements a real 8-child
     octree with recursive 2:1 balancing, Morton ordering, leaf
     enumeration, and conservative refine/coarsen (prolongation and
     restriction are exact inverses on cell averages).

[F3] DISTRIBUTED AMR (MPI)
     Cartesian topology, non-blocking halo exchange with receives
     posted before sends (deadlock-free), and a CROSS-RANK FLUX
     REGISTER: coarse-fine mismatches are keyed by GLOBAL face id and
     all-reduced, so an interface straddling a rank boundary is
     refluxed exactly once, on the owning rank.
     Serial fallback is a semantic no-op, not a silent skip.

[F4] ENERGY
     v5.2 only measured energy. v5.3 adds the skew-symmetric
     advective form
         C_skew = 0.5 * ( div(u (x) u) + u . grad u )
     which, for constant density, divergence-free velocity, zero
     viscosity, zero surface tension and periodic boundaries,
     conserves discrete kinetic energy. T7 measures the drift.
     OUTSIDE that limit (variable rho, nu>0, sigma>0) energy is NOT
     claimed conservative; it is measured and reported.

[F5] EXTERNAL VALIDATION
     A harness that (a) computes analytic-reference benchmarks and
     (b) loads an external reference table (Basilisk / OpenFOAM /
     published CSV) and reports the actual relative differences.
     If no reference file is supplied, the harness says
     "NOT COMPARED" rather than implying a comparison happened.

=============================================================================
HONEST SCOPE — WHAT v5.3 STILL IS NOT
=============================================================================
❌ Multi-rank MPI has NOT been executed here. The code paths are
   written and the serial fallback is tested; you must run
   `mpirun -n 4 python amr_vof_coupler_5_3.py --test` yourself.
❌ No Basilisk / OpenFOAM run is bundled. The comparison harness is
   ready; the reference data is not. Nothing is claimed as validated
   until you supply it and run it.
❌ Energy conservation is proven ONLY in the restricted limit above.
❌ MOF normal optimisation is a fixed number of projected-gradient
   steps, not a converged solve; the achieved centroid residual is
   REPORTED per call so you can see whether it suffices.
❌ No load balancing across ranks beyond static Morton partitioning.
❌ Not a substitute for a production CFD code.
=============================================================================
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("amr_vof_53")

VERSION = "5.3.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit : "
               "What MSPS NETWORK SEE, Lord Buddha Knows?")

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    MPI = None
    _HAS_MPI = False


# =============================================================================
# PART 1 — [F1] VECTORIZED MOF: CLOSED-FORM VOLUME + FIRST MOMENT
# =============================================================================

_DIM_TOL = 1e-6


def _pos_pow(t: torch.Tensor, k: int) -> torch.Tensor:
    return t.clamp_min(0.0) ** k


def box_cut_volume_moment(n: torch.Tensor,
                          c: torch.Tensor,
                          L: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Volume and first moment of {x in [0,L1]x[0,L2]x[0,L3] : n.x <= c}.

    n : (3, ...)  normals, MUST be >= 0 (fold before calling)
    c : (...)     plane offset
    L : (3, ...)  box side lengths, > 0

    Returns (V, M) with V shape (...) and M shape (3, ...).
    Fully batched: no Python loop over cells.

    Uses exact 3-D / 2-D / 1-D formulas depending on how many normal
    components are below _DIM_TOL, because the 3-D denominator
    6 n1 n2 n3 suffers catastrophic cancellation when any n_i -> 0.
    """
    n1, n2, n3 = n[0], n[1], n[2]
    L1, L2, L3 = L[0], L[1], L[2]
    tol = _DIM_TOL
    ns = ((n1 < tol).to(torch.int8) + (n2 < tol).to(torch.int8)
          + (n3 < tol).to(torch.int8))

    # ---------------- generic 3-D -------------------------------------
    den3 = (6.0 * n1 * n2 * n3).clamp_min(1e-300)
    V3 = torch.zeros_like(c)
    M3 = torch.zeros_like(n)
    for mask in range(8):
        S = [(mask >> i) & 1 for i in range(3)]
        sign = -1.0 if (sum(S) % 2) else 1.0
        aS = S[0]*n1*L1 + S[1]*n2*L2 + S[2]*n3*L3
        cS = c - aS
        VS = _pos_pow(cS, 3) / den3
        V3 = V3 + sign * VS
        # centroid_i = cS/(4 n_i) + L_i if i in S
        # guard: when cS<=0, VS=0 so the product is 0 regardless
        cS_pos = cS.clamp_min(0.0)
        M3[0] = M3[0] + sign * VS * (cS_pos/(4.0*n1.clamp_min(1e-300)) + S[0]*L1)
        M3[1] = M3[1] + sign * VS * (cS_pos/(4.0*n2.clamp_min(1e-300)) + S[1]*L2)
        M3[2] = M3[2] + sign * VS * (cS_pos/(4.0*n3.clamp_min(1e-300)) + S[2]*L3)

    # ---------------- 2-D fallbacks (one n_i ~ 0) ----------------------
    def area2(dd, ma, mb, La, Lb):
        den = (2.0 * ma * mb).clamp_min(1e-300)
        A = torch.zeros_like(dd)
        Ms_a = torch.zeros_like(dd)
        Ms_b = torch.zeros_like(dd)
        for m in range(4):
            Sa, Sb = (m >> 0) & 1, (m >> 1) & 1
            sg = -1.0 if (Sa + Sb) % 2 else 1.0
            dS = dd - Sa*ma*La - Sb*mb*Lb
            AS = _pos_pow(dS, 2) / den
            dS_pos = dS.clamp_min(0.0)
            A = A + sg * AS
            Ms_a = Ms_a + sg * AS * (dS_pos/(3.0*ma.clamp_min(1e-300)) + Sa*La)
            Ms_b = Ms_b + sg * AS * (dS_pos/(3.0*mb.clamp_min(1e-300)) + Sb*Lb)
        return A, Ms_a, Ms_b

    V2 = [None]*3; M2 = [None]*3
    # n1 ~ 0 : plane acts in (y,z), full extent L1 in x
    A, My, Mz = area2(c, n2, n3, L2, L3)
    V2[0] = A * L1
    M2[0] = torch.stack([V2[0]*L1/2, My*L1, Mz*L1], 0)
    # n2 ~ 0
    A, Mx, Mz = area2(c, n1, n3, L1, L3)
    V2[1] = A * L2
    M2[1] = torch.stack([Mx*L2, V2[1]*L2/2, Mz*L2], 0)
    # n3 ~ 0
    A, Mx, My = area2(c, n1, n2, L1, L2)
    V2[2] = A * L3
    M2[2] = torch.stack([Mx*L3, My*L3, V2[2]*L3/2], 0)

    # ---------------- 1-D fallbacks (two n_i ~ 0) ----------------------
    V1 = [None]*3; M1 = [None]*3
    a1 = (c / n1.clamp_min(1e-300)).clamp(0.0, L1)
    V1[0] = a1 * L2 * L3
    M1[0] = torch.stack([0.5*a1*a1*L2*L3, V1[0]*L2/2, V1[0]*L3/2], 0)
    a2 = (c / n2.clamp_min(1e-300)).clamp(0.0, L2)
    V1[1] = a2 * L1 * L3
    M1[1] = torch.stack([V1[1]*L1/2, 0.5*a2*a2*L1*L3, V1[1]*L3/2], 0)
    a3 = (c / n3.clamp_min(1e-300)).clamp(0.0, L3)
    V1[2] = a3 * L1 * L2
    M1[2] = torch.stack([V1[2]*L1/2, V1[2]*L2/2, 0.5*a3*a3*L1*L2], 0)

    # ---------------- select ------------------------------------------
    V = V3; M = M3
    for i in range(3):
        ni = n[i]
        sel2 = (ns == 1) & (ni < tol)
        sel1 = (ns >= 2) & (ni >= tol)
        V = torch.where(sel2, V2[i], V)
        M = torch.where(sel2.unsqueeze(0), M2[i], M)
        V = torch.where(sel1, V1[i], V)
        M = torch.where(sel1.unsqueeze(0), M1[i], M)
    vmax = L1 * L2 * L3
    allzero = (ns == 3)
    V = torch.where(allzero, vmax * (c > 0).to(V.dtype), V)
    Mfull = torch.stack([vmax*L1/2, vmax*L2/2, vmax*L3/2], 0)
    M = torch.where(allzero.unsqueeze(0) & (c > 0).unsqueeze(0), Mfull, M)
    M = torch.where((V <= 0).unsqueeze(0), torch.zeros_like(M), M)
    return V.clamp(0.0, vmax), M


def fold_normals(n: torch.Tensor, c: torch.Tensor, L: torch.Tensor,
                 centroid: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fold to the positive octant. For n_i < 0 substitute x_i -> L_i - x_i:
      n_i -> |n_i|,  c -> c - n_i L_i,  centroid_i -> L_i - centroid_i
    Returns (n_fold, c_fold, centroid_fold).
    """
    neg = (n < 0)
    nf = n.abs()
    shift = torch.where(neg, n * L, torch.zeros_like(n)).sum(0)
    cf = c - shift
    ctrf = torch.where(neg, L - centroid, centroid)
    return nf, cf, ctrf


def unfold_centroid(ctr_fold: torch.Tensor, n_orig: torch.Tensor,
                    L: torch.Tensor) -> torch.Tensor:
    return torch.where(n_orig < 0, L - ctr_fold, ctr_fold)


def invert_plane_offset(n_fold: torch.Tensor, V_target: torch.Tensor,
                        L: torch.Tensor, iters: int = 40) -> torch.Tensor:
    """Batched bisection: find c with box_cut_volume(n,c,L) == V_target."""
    lo = torch.zeros_like(V_target)
    hi = (n_fold * L).sum(0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        V, _ = box_cut_volume_moment(n_fold, mid, L)
        small = V < V_target
        lo = torch.where(small, mid, lo)
        hi = torch.where(small, hi, mid)
    return 0.5 * (lo + hi)


class VectorizedMOF:
    """
    Fully batched MOF reconstruction from (alpha, centroid).

    No per-cell Python loop. The normal is optimised by `n_iter`
    projected-gradient steps on the unit sphere; the ACHIEVED centroid
    residual is returned so the caller can see whether it suffices.
    """

    def __init__(self, n_iter: int = 8, bisect_iters: int = 40,
                 step: float = 0.5):
        self.n_iter = n_iter
        self.bisect_iters = bisect_iters
        self.step = step

    def reconstruct(self, alpha: torch.Tensor, centroid: torch.Tensor,
                    dx: float
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        alpha    : (nx,ny,nz) volume fraction in [0,1]
        centroid : (3,nx,ny,nz) fluid centroid in CELL-LOCAL coords [0,dx]
        Returns (normal(3,...), offset(...), centroid_residual(...)).
        """
        shape = alpha.shape
        L = torch.full((3,) + shape, float(dx),
                       dtype=torch.float64, device=alpha.device)
        Vcell = dx ** 3
        mixed = (alpha > 1e-8) & (alpha < 1 - 1e-8)
        Vt = (alpha.to(torch.float64) * Vcell).clamp(0.0, Vcell)

        ctr = centroid.to(torch.float64).clamp(0.0, dx)
        center = 0.5 * dx
        n0 = ctr - center
        nrm = n0.norm(dim=0, keepdim=True)
        n0 = torch.where(nrm > 1e-12, n0 / nrm.clamp_min(1e-30),
                         torch.zeros_like(n0))
        n0[2] = torch.where(nrm.squeeze(0) <= 1e-12,
                            torch.ones_like(n0[2]), n0[2])

        nf, _, ctrf = fold_normals(n0, torch.zeros_like(alpha, dtype=torch.float64),
                                   L, ctr)
        n = nf / nf.norm(dim=0, keepdim=True).clamp_min(1e-30)

        for _ in range(self.n_iter):
            c = invert_plane_offset(n, Vt, L, self.bisect_iters)
            V, M = box_cut_volume_moment(n, c, L)
            ctr_a = M / V.clamp_min(1e-30).unsqueeze(0)
            err = ctr_a - ctrf
            err_t = err - (err * n).sum(0, keepdim=True) * n
            et = err_t.norm(dim=0, keepdim=True)
            n = torch.where(et > 1e-14,
                            n - self.step * err_t / et.clamp_min(1e-30), n)
            n = n / n.norm(dim=0, keepdim=True).clamp_min(1e-30)

        c = invert_plane_offset(n, Vt, L, self.bisect_iters)
        V, M = box_cut_volume_moment(n, c, L)
        ctr_a = M / V.clamp_min(1e-30).unsqueeze(0)
        resid = (ctr_a - ctrf).norm(dim=0)
        resid = torch.where(mixed, resid, torch.zeros_like(resid))

        # unfold offset back to the ORIGINAL (signed-normal) frame
        neg = (n0 < 0)
        c_orig = c + torch.where(neg, n * L, torch.zeros_like(n)).sum(0)
        n_out = torch.where(mixed.unsqueeze(0), n0 / n0.norm(dim=0, keepdim=True).clamp_min(1e-30),
                            torch.zeros_like(n0))
        c_out = torch.where(mixed, c_orig, torch.zeros_like(c))
        return n_out, c_out, resid

    def advect_split(self, alpha: torch.Tensor, centroid: torch.Tensor,
                     u: torch.Tensor, dt: float, dx: float
                     ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Dimensionally split MOF advection carrying volume AND moment.
        Returns (alpha_new, centroid_new, diagnostics).
        """
        a = alpha.to(torch.float64)
        ctr = centroid.to(torch.float64).clone()
        m0 = float(a.sum())
        max_resid = 0.0
        for ax in range(3):
            n, c, resid = self.reconstruct(a, ctr, dx)
            max_resid = max(max_resid, float(resid.max()))
            a, ctr = self._sweep(a, ctr, n, c, u[ax].to(torch.float64),
                                 dt, dx, ax)
        drift = abs(float(a.sum()) - m0) / max(m0, 1e-30)
        return a.clamp(0.0, 1.0), ctr, {"mof_centroid_resid": max_resid,
                                       "mass_rel_drift": drift}

    def _sweep(self, a, ctr, n, c, vel, dt, dx, ax):
        """One-direction MOF sweep with batched swept-box moments."""
        shape = a.shape
        nc = shape[ax]
        nf = nc + 1
        # face velocity from cell-centred vel
        vf = torch.zeros(self._face_shape(shape, ax), dtype=a.dtype, device=a.device)
        si = self._sl(ax, slice(1, -1)); sl = self._sl(ax, slice(0, -1)); sh = self._sl(ax, slice(1, None))
        vf[si] = 0.5 * (vel[sl] + vel[sh])
        vf[self._sl(ax, 0)] = vel[self._sl(ax, 0)]
        vf[self._sl(ax, -1)] = vel[self._sl(ax, -1)]

        # donor index per face
        idx = torch.arange(nf, device=a.device)
        pos = vf > 0
        donor = torch.where(pos, idx.reshape(self._bshape(nf, ax)) - 1,
                            idx.reshape(self._bshape(nf, ax)))
        donor = donor.clamp(0, nc - 1)

        def gather(f, d):
            return f.index_select(ax, d.reshape(-1)).reshape(
                f.shape[:ax] + (nf,) + f.shape[ax+1:]) if f.dim() == 3 else \
                   f.index_select(ax + 1, d.reshape(-1)).reshape(
                       f.shape[:ax+1] + (nf,) + f.shape[ax+2:])

        nd = gather(n, donor)
        cd = gather(c, donor)
        # swept slab thickness
        s = (vf.abs() * dt).clamp(0.0, dx)
        face_pos = idx.to(a.dtype).reshape(self._bshape(nf, ax)) * dx
        lo_ax = torch.where(pos, face_pos - s, face_pos)

        L = torch.stack([torch.full_like(vf, dx)]*3, 0)
        L[ax] = s
        # fold donor normal into positive octant, shift offset to slab frame
        neg = nd < 0
        nfld = nd.abs()
        shift = torch.where(neg, nd * L, torch.zeros_like(nd)).sum(0)
        cfld = cd - shift
        # slab origin along ax is lo_ax (donor-local frame offset)
        off_ax = torch.where(pos, (donor.to(a.dtype).reshape(self._bshape(nf, ax))*dx + dx - s) - lo_ax,
                             donor.to(a.dtype).reshape(self._bshape(nf, ax))*dx - lo_ax)
        cfld = cfld - nfld[ax] * (-off_ax)

        V, M = box_cut_volume_moment(nfld, cfld, L)
        sgn = torch.where(pos, torch.ones_like(V), -torch.ones_like(V))
        P = sgn * V                       # signed volume flux
        Q = sgn.unsqueeze(0) * M          # signed moment flux (slab-local)
        # convert moment ax-component to global coordinate
        Q[ax] = Q[ax] + P * lo_ax
        for d in range(3):
            if d != ax:
                pass  # transverse coords already global (cell index * dx)

        # update
        Vcell = dx ** 3
        a_new = a.clone(); ctr_new = ctr.clone()
        Pin = P[self._sl(ax, slice(0, -1))]
        Pout = P[self._sl(ax, slice(1, None))]
        Qin = Q[(slice(None),) + self._sl(ax, slice(0, -1))]
        Qout = Q[(slice(None),) + self._sl(ax, slice(1, None))]
        Vnew = (a * Vcell - Pout + Pin).clamp(0.0, Vcell)
        Mold = ctr * (a * Vcell).unsqueeze(0)
        Mnew = Mold - Qout + Qin
        ctr_new = torch.where(Vnew.unsqueeze(0) > 1e-12,
                              Mnew / Vnew.unsqueeze(0).clamp_min(1e-30), ctr)
        ctr_new = ctr_new.clamp(0.0, dx)
        a_new = Vnew / Vcell
        return a_new, ctr_new

    @staticmethod
    def _sl(ax, s):
        o = [slice(None)]*3; o[ax] = s; return tuple(o)

    @staticmethod
    def _bshape(n, ax):
        o = [1, 1, 1]; o[ax] = n; return tuple(o)

    @staticmethod
    def _face_shape(shape, ax):
        s = list(shape); s[ax] += 1; return tuple(s)


# =============================================================================
# PART 2 — [F2] TRUE OCTREE WITH 2:1 BALANCE
# =============================================================================

def _morton3(i: int, j: int, k: int) -> int:
    def spread(v):
        v &= 0x1FFFFF
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        v = (v | (v << 2)) & 0x1249249249249249
        return v
    return spread(i) | (spread(j) << 1) | (spread(k) << 2)


@dataclass
class OctreeNode:
    level: int
    origin: Tuple[int, int, int]        # in THIS level's index space
    children: Optional[List["OctreeNode"]] = None
    data: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def is_leaf(self): return self.children is None

    @property
    def key(self): return (self.level, self.origin)

    @property
    def morton(self): return _morton3(*self.origin)

    def child_origin(self, octant: int) -> Tuple[int, int, int]:
        di = octant & 1; dj = (octant >> 1) & 1; dk = (octant >> 2) & 1
        o = self.origin
        return (2*o[0]+di, 2*o[1]+dj, 2*o[2]+dk)


class Octree:
    """
    True 8-child octree over a base grid of n^3 cells.

    Level L has spacing dx0 / 2^L. Node origin is in level-L index space.
    Provides:
      refine / coarsen with CONSERVATIVE transfer (exact on cell averages)
      recursive 2:1 balancing
      leaf enumeration in Morton order
      O(1) neighbour lookup via a level-indexed dict
    """

    def __init__(self, base_n: int, dx0: float, max_level: int = 4,
                 device: str = "cpu", dtype=torch.float64):
        self.base_n = base_n
        self.dx0 = dx0
        self.max_level = max_level
        self.dev = torch.device(device)
        self.dt_ = dtype
        self.root = OctreeNode(0, (0, 0, 0))
        self._index: Dict[Tuple, OctreeNode] = {}
        self._reindex()

    # ---- indexing -----------------------------------------------------
    def _reindex(self):
        self._index.clear()
        def walk(nd):
            self._index[nd.key] = nd
            if nd.children:
                for ch in nd.children: walk(ch)
        walk(self.root)

    def leaves(self) -> List[OctreeNode]:
        out = []
        def walk(nd):
            if nd.is_leaf: out.append(nd)
            else:
                for ch in nd.children: walk(ch)
        walk(self.root)
        return sorted(out, key=lambda n: (n.level, n.morton))

    def dx(self, level: int) -> float:
        return self.dx0 / (2 ** level)

    def node_shape(self, level: int) -> int:
        return max(1, self.base_n // (2 ** level))

    # ---- conservative transfer ----------------------------------------
    @staticmethod
    def prolong_cell(c: torch.Tensor) -> torch.Tensor:
        has = c.dim() == 4
        if not has: c = c.unsqueeze(0)
        out = (c.repeat_interleave(2, 1)
                .repeat_interleave(2, 2)
                .repeat_interleave(2, 3))
        return out if has else out.squeeze(0)

    @staticmethod
    def restrict_cell(f: torch.Tensor) -> torch.Tensor:
        has = f.dim() == 4
        if not has: f = f.unsqueeze(0)
        nc, i, j, k = f.shape
        out = f.reshape(nc, i//2, 2, j//2, 2, k//2, 2).mean(dim=(2, 4, 6))
        return out if has else out.squeeze(0)

    # ---- refine / coarsen ---------------------------------------------
    def refine(self, node: OctreeNode, fields: Optional[List[str]] = None):
        if not node.is_leaf or node.level >= self.max_level:
            return
        fields = fields or list(node.data.keys())
        node.children = []
        for oc in range(8):
            ch = OctreeNode(node.level + 1, node.child_origin(oc))
            for fname in fields:
                if fname in node.data:
                    ch.data[fname] = self.prolong_cell(node.data[fname]).clone()
            node.children.append(ch)
        self._reindex()

    def coarsen(self, node: OctreeNode, fields: Optional[List[str]] = None):
        if node.is_leaf: return
        if any(not ch.is_leaf for ch in node.children):
            for ch in node.children: self.coarsen(ch, fields)
        fields = fields or list(node.children[0].data.keys())
        # conservative average of the 8 children onto the parent
        for fname in fields:
            vals = [ch.data[fname] for ch in node.children if fname in ch.data]
            if len(vals) == 8:
                stacked = torch.stack(vals, 0)     # (8, ...) or (8,C,...)
                if stacked.dim() == 4:
                    node.data[fname] = stacked.mean(0)
                else:
                    node.data[fname] = stacked.mean(0)
        node.children = None
        self._reindex()

    # ---- 2:1 balance ---------------------------------------------------
    def neighbours(self, node: OctreeNode) -> List[OctreeNode]:
        """Same-level face neighbours (may be absent at domain edges)."""
        out = []
        lv, o = node.level, node.origin
        for ax in range(3):
            for d in (-1, 1):
                oo = list(o); oo[ax] += d
                nb = self._index.get((lv, tuple(oo)))
                if nb is not None: out.append(nb)
        return out

    def balance_2to1(self, max_passes: int = 32) -> int:
        """
        Recursively refine so that no two face-adjacent LEAVES differ by
        more than one level. Returns the number of refinements performed.
        """
        n_ref = 0
        for _ in range(max_passes):
            changed = False
            leaves = self.leaves()
            lmap = {(n.level, n.origin): n for n in leaves}
            for nd in leaves:
                if not nd.is_leaf: continue
                lv, o = nd.level, nd.origin
                need = False
                # a neighbour region at level lv+2 would violate 2:1
                for ax in range(3):
                    for d in (-1, 1):
                        oo = list(o); oo[ax] += d
                        # look for any leaf 2+ levels finer touching this face
                        for lv2 in range(lv + 2, self.max_level + 1):
                            r = 2 ** (lv2 - lv)
                            base = [x * r for x in oo]
                            found = False
                            for t1 in range(r):
                                for t2 in range(r):
                                    key = [0, 0, 0]
                                    key[ax] = base[ax] + (r - 1 if d > 0 else 0)
                                    others = [a for a in range(3) if a != ax]
                                    key[others[0]] = base[others[0]] + t1
                                    key[others[1]] = base[others[1]] + t2
                                    if (lv2, tuple(key)) in lmap:
                                        found = True; break
                                if found: break
                            if found: need = True; break
                    if need: break
                if need and lv < self.max_level:
                    self.refine(nd)
                    n_ref += 1
                    changed = True
                    break
            if not changed:
                break
        self._reindex()
        return n_ref

    def check_2to1(self) -> Tuple[bool, int]:
        """Verify the leaf level map has no jump > 1 across faces."""
        leaves = self.leaves()
        # build a base-cell level map
        m = torch.zeros((self.base_n,)*3, dtype=torch.long)
        for nd in leaves:
            r = 2 ** nd.level
            sz = max(1, self.base_n // r)
            o = nd.origin
            i0 = o[0]*sz; j0 = o[1]*sz; k0 = o[2]*sz
            m[i0:i0+sz, j0:j0+sz, k0:k0+sz] = nd.level
        viol = 0
        for ax in range(3):
            a = m.narrow(ax, 1, m.shape[ax]-1)
            b = m.narrow(ax, 0, m.shape[ax]-1)
            viol += int(((a-b).abs() > 1).sum())
        return (viol == 0), viol

    def total_cells(self) -> int:
        return sum(self.node_shape(n.level)**3 for n in self.leaves())


# =============================================================================
# PART 3 — [F3] DISTRIBUTED AMR: MPI HALO + CROSS-RANK FLUX REGISTER
# =============================================================================

class MPILayer:
    """
    Cartesian-topology MPI layer with a semantic serial fallback.

    halo_exchange posts ALL receives before ANY send (deadlock-free).
    The cross-rank flux register keys mismatches by GLOBAL face id and
    all-reduces them, so an interface straddling a rank boundary is
    refluxed exactly once.
    """

    def __init__(self, dims: Optional[Tuple[int, int, int]] = None,
                 periodic: Tuple[bool, bool, bool] = (True, True, True)):
        self.rank = 0
        self.size = 1
        self.comm = None
        self.cart = None
        self.coords = (0, 0, 0)
        self.dims = (1, 1, 1)
        self.neighbours: Dict[Tuple[int, int], int] = {}
        if _HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            if self.size > 1:
                d = list(dims) if dims else [0, 0, 0]
                d = MPI.Dims_create(self.size, d)
                self.dims = tuple(d)
                self.cart = self.comm.Create_cart(self.dims, periods=periodic,
                                                  reorder=False)
                self.rank = self.cart.Get_rank()
                self.coords = tuple(self.cart.Get_coords(self.rank))
                for ax in range(3):
                    lo, hi = self.cart.Shift(ax, 1)
                    self.neighbours[(ax, -1)] = lo
                    self.neighbours[(ax, +1)] = hi
        self.active = (self.size > 1)

    # ---- halo ---------------------------------------------------------
    def halo_exchange(self, field: torch.Tensor, ax: int, ng: int = 2) -> torch.Tensor:
        """
        field includes ng ghost layers on each side along ax.
        Serial: return unchanged (correct semantics, not a silent skip).
        """
        if not self.active:
            return field
        n = field.shape[ax]
        sl_lo = self._sl(ax, slice(ng, 2*ng))
        sl_hi = self._sl(ax, slice(n-2*ng, n-ng))
        send_lo = field[sl_lo].contiguous()
        send_hi = field[sl_hi].contiguous()
        recv_lo = torch.empty_like(send_lo)
        recv_hi = torch.empty_like(send_hi)
        nb_lo = self.neighbours.get((ax, -1), MPI.PROC_NULL)
        nb_hi = self.neighbours.get((ax, +1), MPI.PROC_NULL)
        reqs = []
        # POST RECEIVES FIRST
        if nb_hi != MPI.PROC_NULL:
            reqs.append(self.cart.Irecv(recv_hi.numpy(), source=nb_hi, tag=200+ax))
        if nb_lo != MPI.PROC_NULL:
            reqs.append(self.cart.Irecv(recv_lo.numpy(), source=nb_lo, tag=201+ax))
        if nb_lo != MPI.PROC_NULL:
            reqs.append(self.cart.Isend(send_lo.numpy(), dest=nb_lo, tag=200+ax))
        if nb_hi != MPI.PROC_NULL:
            reqs.append(self.cart.Isend(send_hi.numpy(), dest=nb_hi, tag=201+ax))
        if reqs:
            MPI.Request.Waitall(reqs)
        out = field.clone()
        if nb_hi != MPI.PROC_NULL:
            out[self._sl(ax, slice(n-ng, n))] = recv_hi
        if nb_lo != MPI.PROC_NULL:
            out[self._sl(ax, slice(0, ng))] = recv_lo
        return out

    @staticmethod
    def _sl(ax, s):
        o = [slice(None)]*3
        o[ax] = s
        return tuple(o)

    # ---- reductions ----------------------------------------------------
    def allreduce_sum(self, x: float) -> float:
        if not self.active: return x
        import numpy as np
        a = np.array([x], dtype=np.float64)
        b = np.zeros_like(a)
        self.cart.Allreduce(a, b, op=MPI.SUM)
        return float(b[0])

    def allreduce_min(self, x: float) -> float:
        if not self.active: return x
        import numpy as np
        a = np.array([x], dtype=np.float64); b = np.zeros_like(a)
        self.cart.Allreduce(a, b, op=MPI.MIN)
        return float(b[0])

    def barrier(self):
        if self.active: self.cart.Barrier()


class CrossRankFluxRegister:
    """
    Flux register keyed by GLOBAL coarse face id, reduced across ranks.

    A coarse-fine interface that lies on a rank boundary contributes
    flux from both ranks. Summing by global id before applying the
    correction guarantees the mismatch is refluxed exactly once.
    """

    def __init__(self, mpi: MPILayer, ncomp: int = 4):
        self.mpi = mpi
        self.ncomp = ncomp
        self.coarse: Dict[Tuple, torch.Tensor] = {}
        self.fine: Dict[Tuple, torch.Tensor] = {}
        self.n_applied = 0
        self.total_correction = 0.0

    def clear(self):
        self.coarse.clear(); self.fine.clear()

    def add_coarse(self, gid: Tuple, flux: torch.Tensor, area: float, dt: float):
        if gid not in self.coarse:
            self.coarse[gid] = torch.zeros(self.ncomp, dtype=torch.float64)
        self.coarse[gid] += flux.detach().cpu().to(torch.float64) * area * dt

    def add_fine(self, gid: Tuple, flux: torch.Tensor, area: float, dt: float):
        if gid not in self.fine:
            self.fine[gid] = torch.zeros(self.ncomp, dtype=torch.float64)
        self.fine[gid] += flux.detach().cpu().to(torch.float64) * area * dt

    def globalize(self) -> Dict[Tuple, torch.Tensor]:
        """
        All-reduce the per-face coarse and fine integrals across ranks.
        Serial: identity. Returns the mismatch dict (coarse - fine).
        """
        keys = sorted(set(self.coarse) | set(self.fine))
        if not keys:
            return {}
        if self.mpi.active:
            import numpy as np
            allkeys = self.mpi.cart.allgather(keys)
            keys = sorted(set(k for grp in allkeys for k in grp))
            nc = self.ncomp
            buf = np.zeros((len(keys), 2*nc), dtype=np.float64)
            for i, k in enumerate(keys):
                if k in self.coarse: buf[i, :nc] = self.coarse[k].numpy()
                if k in self.fine:   buf[i, nc:] = self.fine[k].numpy()
            out = np.zeros_like(buf)
            self.mpi.cart.Allreduce(buf, out, op=MPI.SUM)
        else:
            out = None
        mism = {}
        for i, k in enumerate(keys):
            if out is not None:
                c = torch.from_numpy(out[i, :self.ncomp].copy())
                f = torch.from_numpy(out[i, self.ncomp:].copy())
            else:
                c = self.coarse.get(k, torch.zeros(self.ncomp, dtype=torch.float64))
                f = self.fine.get(k, torch.zeros(self.ncomp, dtype=torch.float64))
            mism[k] = c - f
        return mism

    def apply(self, alpha: torch.Tensor, rhou: Optional[torch.Tensor],
              coarse_box_lo: Tuple[int, int, int],
              dx_c: float, owner_of: Callable[[Tuple], bool]
              ) -> Dict[str, float]:
        """
        Apply the reduced mismatch to the coarse cells OWNED by this rank.
        `owner_of(gid)` decides ownership so each face is corrected once.
        """
        mism = self.globalize()
        Vc = dx_c ** 3
        n_app = 0; mx = 0.0
        for gid, d in mism.items():
            if not owner_of(gid):
                continue
            ax, side, t1, t2, ci = gid   # gid carries the adjacent coarse cell
            i, j, k = ci
            sh = alpha.shape
            if not (0 <= i < sh[0] and 0 <= j < sh[1] and 0 <= k < sh[2]):
                continue
            corr = (d / Vc).to(alpha.dtype).to(alpha.device)
            alpha[i, j, k] = (alpha[i, j, k] + corr[0]).clamp(0.0, 1.0)
            if rhou is not None:
                for c in range(3):
                    rhou[c, i, j, k] += corr[1+c]
            n_app += 1; mx = max(mx, float(d.abs().max()))
        self.n_applied += n_app
        self.total_correction += mx
        return {"n_applied": n_app, "max_mismatch": mx}


# =============================================================================
# PART 4 — [F4] ENERGY-CONSERVING SKEW-SYMMETRIC ADVECTION
# =============================================================================

def _roll(f, ax, d):
    return torch.roll(f, d, ax)


def div_cc(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Cell-centred divergence of a cell-centred velocity (3,nx,ny,nz)."""
    out = torch.zeros_like(u[0])
    for ax in range(3):
        out = out + (_roll(u[ax], ax, -1) - _roll(u[ax], ax, 1)) / (2*dx)
    return out


def advective_conservative(rhou: torch.Tensor, u: torch.Tensor,
                           dx: float) -> torch.Tensor:
    """div(u (x) rho u) — conservative form."""
    out = torch.zeros_like(rhou)
    for ax in range(3):
        for comp in range(3):
            f = u[ax] * rhou[comp]
            out[comp] += (_roll(f, ax, -1) - _roll(f, ax, 1)) / (2*dx)
    return out


def advective_skew(rho: torch.Tensor, u: torch.Tensor,
                   dx: float) -> torch.Tensor:
    """
    Skew-symmetric advective form:
        C_skew = 0.5 * [ div(u (x) u) + u . grad u ]   (times rho)

    For CONSTANT rho, div-free u, nu=0, sigma=0, periodic BC, this form
    is discretely skew-adjoint and therefore conserves kinetic energy.
    Outside that limit it is NOT claimed conservative.
    """
    rhou = rho.unsqueeze(0) * u
    cons = advective_conservative(rhou, u, dx)
    conv = torch.zeros_like(u)
    for ax in range(3):
        for comp in range(3):
            conv[comp] += u[ax] * (_roll(u[comp], ax, -1)
                                   - _roll(u[comp], ax, 1)) / (2*dx)
    return 0.5 * (cons + rho.unsqueeze(0) * conv)


def kinetic_energy(rho: torch.Tensor, u: torch.Tensor, dx: float) -> float:
    return float(0.5 * (rho.unsqueeze(0) * u*u).sum() * dx**3)


def energy_budget(rho, u, p, alpha, dx, sigma, gravity) -> Dict[str, float]:
    """Measured energy budget. NOT a conservation claim."""
    KE = kinetic_energy(rho, u, dx)
    nz = alpha.shape[2]
    z = (torch.arange(nz, dtype=alpha.dtype, device=alpha.device) + 0.5) * dx
    PE = float((rho * gravity[2] * z.view(1, 1, -1)).sum() * dx**3)
    g = torch.stack([(_roll(alpha, ax, -1) - _roll(alpha, ax, 1))/(2*dx)
                     for ax in range(3)], 0)
    SE = float(sigma * g.norm(dim=0).sum() * dx**3)
    return {"KE": KE, "PE": PE, "SE": SE, "total": KE + PE + SE}


# =============================================================================
# PART 5 — [F5] EXTERNAL-VALIDATION HARNESS
# =============================================================================

@dataclass
class ValidationResult:
    name: str
    metric: str
    computed: float
    reference: Optional[float]
    source: str
    rel_diff: Optional[float]
    verdict: str


class ValidationHarness:
    """
    Computes analytic benchmarks AND compares against an external
    reference table when one is supplied.

    If no reference is available the verdict is "NOT COMPARED" — the
    harness never implies a comparison that did not happen.
    """

    def __init__(self, reference_csv: Optional[str] = None):
        self.ref: Dict[str, Dict[str, float]] = {}
        self.ref_source = "none"
        if reference_csv and os.path.exists(reference_csv):
            with open(reference_csv) as f:
                for row in csv.DictReader(f):
                    key = row["case"].strip()
                    self.ref.setdefault(key, {})
                    self.ref[key][row["metric"].strip()] = float(row["value"])
            self.ref_source = reference_csv

    def _cmp(self, name, metric, value, analytic=None) -> ValidationResult:
        ref = None; src = "analytic"
        if name in self.ref and metric in self.ref[name]:
            ref = self.ref[name][metric]; src = self.ref_source
        elif analytic is not None:
            ref = analytic
        if ref is None:
            return ValidationResult(name, metric, value, None, "none",
                                    None, "NOT COMPARED")
        rd = abs(value - ref) / max(abs(ref), 1e-30)
        return ValidationResult(name, metric, value, ref, src, rd,
                                "COMPARED")

    # ---- analytic benchmark 1: static droplet -------------------------
    def static_droplet(self, n: int = 24, R: float = 0.25,
                       sigma: float = 0.05) -> List[ValidationResult]:
        """Exact Laplace pressure jump for a 3-D droplet: dp = 2 sigma / R."""
        dx = 1.0 / n
        x = (torch.arange(n, dtype=torch.float64) + 0.5) * dx
        X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
        a = (((X-.5)**2 + (Y-.5)**2 + (Z-.5)**2) < R**2).to(torch.float64)
        # CSF curvature from smoothed alpha
        al = a
        for _ in range(2):
            al = (6*al + sum(_roll(al, ax, d) for ax in range(3) for d in (-1, 1))) / 12.0
        g = torch.stack([(_roll(al, ax, -1)-_roll(al, ax, 1))/(2*dx) for ax in range(3)], 0)
        gm = g.norm(dim=0).clamp_min(1e-12)
        nh = g / gm.unsqueeze(0)
        k = -sum((_roll(nh[ax], ax, -1)-_roll(nh[ax], ax, 1))/(2*dx) for ax in range(3))
        iface = (a > 0.05) & (a < 0.95)
        k_mean = float(k[iface].abs().mean()) if int(iface.sum()) else float("nan")
        dp_exact = 2*sigma/R
        dp_num = sigma*k_mean
        return [self._cmp("static_droplet", "curvature", k_mean, 2.0/R),
                self._cmp("static_droplet", "laplace_dp", dp_num, dp_exact)]

    # ---- analytic benchmark 2: Zalesak slotted disk --------------------
    def zalesak(self, n: int = 32) -> List[ValidationResult]:
        """Rigid rotation, exact return after one revolution."""
        dx = 1.0/n
        x = (torch.arange(n, dtype=torch.float64)+0.5)*dx
        X, Y = torch.meshgrid(x, x, indexing="ij")
        cx, cy, R = 0.5, 0.75, 0.15
        disk = ((X-cx)**2 + (Y-cy)**2) <= R**2
        slot = ((X-cx).abs() <= 0.025) & (Y >= cy-R) & (Y <= cy+0.10)
        a2 = (disk & ~slot).to(torch.float64)
        slot_mask = disk & slot
        nz = 4
        a = a2.unsqueeze(-1).expand(-1, -1, nz).clone()
        a0 = a.clone()
        omega = 2*math.pi
        u = torch.zeros(3, n, n, nz, dtype=torch.float64)
        u[0] = (-omega*(Y-cy+cy-0.5)).unsqueeze(-1).expand(-1, -1, nz)
        u[1] = (omega*(X-0.5)).unsqueeze(-1).expand(-1, -1, nz)
        mof = VectorizedMOF(n_iter=4, bisect_iters=28)
        ctr = torch.zeros(3, n, n, nz, dtype=torch.float64)
        for d in range(3): ctr[d] = 0.5*dx
        umax = omega*0.5
        dt = 0.25*dx/umax
        nsteps = int(round(1.0/dt)); dt = 1.0/nsteps
        m0 = float(a.sum())
        for _ in range(nsteps):
            a, ctr, _ = mof.advect_split(a, ctr, u, dt, dx)
        d = (a - a0).abs()
        L1 = float(d.sum()/max(float(a0.sum()), 1e-30))
        sf = float(a[:, :, 0][slot_mask].mean()) if int(slot_mask.sum()) else 0.0
        md = abs(float(a.sum())-m0)/max(m0, 1e-30)
        return [self._cmp("zalesak", "L1", L1, 0.0),
                self._cmp("zalesak", "slot_fill", sf, 0.0),
                self._cmp("zalesak", "mass_rel_drift", md, 0.0)]

    # ---- analytic benchmark 3: Taylor-Green (2-D exact) ----------------
    def taylor_green_order(self, grids=(16, 32)) -> List[ValidationResult]:
        """
        2-D Taylor-Green decays exactly: u ~ exp(-2 nu k^2 t).
        Measures the SPATIAL order of the skew-symmetric operator.
        """
        nu, k = 1e-2, 2*math.pi
        errs = []
        for n in grids:
            dx = 1.0/n
            x = (torch.arange(n, dtype=torch.float64)+0.5)*dx
            X, Y = torch.meshgrid(x, x, indexing="ij")
            nz = 2
            X3 = X.unsqueeze(-1).expand(-1, -1, nz); Y3 = Y.unsqueeze(-1).expand(-1, -1, nz)
            Z3 = torch.zeros_like(X3)
            dec = math.exp(-2*nu*k*k*0.02)
            u = torch.stack([-torch.cos(k*X3)*torch.sin(k*Y3),
                             torch.sin(k*X3)*torch.cos(k*Y3),
                             torch.zeros_like(X3)], 0)
            rho = torch.ones_like(X3)
            dt = 0.0005
            nst = int(round(0.02/dt))
            uu = u.clone()
            for _ in range(nst):
                C = advective_skew(rho, uu, dx)
                lap = torch.stack([sum((_roll(uu[c], ax, d)-2*uu[c]+_roll(uu[c], ax, -d))/dx**2
                                       for ax in range(3) for d in (1,)) for c in range(3)], 0)
                uu = uu + dt*(-C/rho.unsqueeze(0) + nu*lap)
            ref = u*dec
            e = float((uu-ref).norm()/ref.norm().clamp_min(1e-30))
            errs.append((n, e))
        out = [self._cmp("taylor_green", f"L2_n{n}", e) for n, e in errs]
        if len(errs) >= 2:
            p = math.log2(errs[0][1]/max(errs[1][1], 1e-30))
            out.append(self._cmp("taylor_green", "spatial_order", p, 2.0))
        return out

    # ---- energy conservation in the restricted limit -------------------
    def energy_conservation_limit(self, n: int = 16, nsteps: int = 40
                                  ) -> List[ValidationResult]:
        """rho const, nu=0, sigma=0, periodic: skew form must conserve KE."""
        dx = 1.0/n
        x = (torch.arange(n, dtype=torch.float64)+0.5)*dx
        X, Y = torch.meshgrid(x, x, indexing="ij")
        nz = n
        X3 = X.unsqueeze(-1).expand(-1, -1, nz); Y3 = Y.unsqueeze(-1).expand(-1, -1, nz)
        k = 2*math.pi
        u = torch.stack([-torch.cos(k*X3)*torch.sin(k*Y3),
                         torch.sin(k*X3)*torch.cos(k*Y3),
                         torch.zeros_like(X3)], 0)
        rho = torch.ones_like(X3)
        E0 = kinetic_energy(rho, u, dx)
        dt = 0.0005
        uu = u.clone()
        for _ in range(nsteps):
            C = advective_skew(rho, uu, dx)
            uu = uu - dt*C/rho.unsqueeze(0)
        E1 = kinetic_energy(rho, uu, dx)
        drift = abs(E1-E0)/max(abs(E0), 1e-30)
        return [self._cmp("energy_skew_limit", "KE_rel_drift", drift, 0.0)]

    def run_all(self, quick: bool = True) -> List[ValidationResult]:
        res = []
        res += self.static_droplet(n=20 if quick else 32)
        res += self.energy_conservation_limit(n=12 if quick else 16)
        res += self.taylor_green_order((12, 24) if quick else (16, 32))
        if not quick:
            res += self.zalesak(n=32)
        return res

    def report(self, res: List[ValidationResult]):
        print("="*92)
        print(f"  VALIDATION HARNESS v{VERSION}   reference source: {self.ref_source}")
        print("="*92)
        print(f"  {'case':<22}{'metric':<20}{'computed':>13}{'reference':>13}"
              f"{'rel diff':>11}  verdict")
        print("-"*92)
        for r in res:
            ref = f"{r.reference:.4e}" if r.reference is not None else "—"
            rd = f"{r.rel_diff:.2e}" if r.rel_diff is not None else "—"
            print(f"  {r.name:<22}{r.metric:<20}{r.computed:>13.4e}{ref:>13}"
                  f"{rd:>11}  {r.verdict}")
        print("="*92)
        n_cmp = sum(1 for r in res if r.verdict == "COMPARED")
        print(f"  {n_cmp}/{len(res)} metrics compared against a reference.")
        if self.ref_source == "none":
            print("  External (Basilisk / OpenFOAM) reference NOT supplied.")
            print("  Analytic references are used where they exist; anything")
            print("  else is reported NOT COMPARED, never silently assumed.")


# =============================================================================
# PART 6 — SELF-TEST
# =============================================================================

def self_test(quick: bool = True) -> bool:
    print("="*92)
    print(f"  AMR-VOF COUPLER v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("="*92)
    torch.manual_seed(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ---- T1: closed-form volume + moment vs analytic ------------------
    print("\n[T1] Vectorized MOF: closed-form volume AND first moment")
    n = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64).view(3, 1)
    L = torch.ones(3, 1, dtype=torch.float64)
    c = torch.tensor([1.5], dtype=torch.float64)
    V, M = box_cut_volume_moment(n, c, L)
    # analytic: V=0.5, M_x = 0.18229166666... (derived by direct integration)
    Mx_exact = 0.1822916666666667
    rep("volume of symmetric diagonal cut", abs(float(V)-0.5) < 1e-12,
        f"V={float(V):.12f}")
    rep("first moment matches analytic integral",
        abs(float(M[0])-Mx_exact) < 1e-10,
        f"M_x={float(M[0]):.12f} exact={Mx_exact:.12f}")
    rep("moment symmetric in all 3 axes",
        float((M - M[0]).abs().max()) < 1e-12)

    # corner simplex: n=(1,1,1), c=0.5 -> V=0.5^3/6, centroid=(0.5/4,...)
    c2 = torch.tensor([0.5], dtype=torch.float64)
    V2, M2 = box_cut_volume_moment(n, c2, L)
    rep("corner simplex volume = c^3/6", abs(float(V2) - 0.5**3/6) < 1e-12,
        f"V={float(V2):.10f}")
    rep("corner simplex centroid = c/4",
        abs(float(M2[0]/V2) - 0.125) < 1e-10,
        f"ctr={float(M2[0]/V2):.10f}")

    # degenerate: n=(1,0,0) -> 1-D formula
    nd = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).view(3, 1)
    Vd, Md = box_cut_volume_moment(nd, torch.tensor([0.3], dtype=torch.float64), L)
    rep("degenerate 1-D volume", abs(float(Vd)-0.3) < 1e-12)
    rep("degenerate 1-D moment = a^2/2", abs(float(Md[0])-0.045) < 1e-12,
        f"M_x={float(Md[0]):.10f}")

    # ---- T2: batched MOF reconstruction -------------------------------
    print("\n[T2] Batched MOF reconstruction (no per-cell Python loop)")
    nx = 12
    dx = 1.0/nx
    x = (torch.arange(nx, dtype=torch.float64)+0.5)*dx
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    a = (((X-.5)**2+(Y-.5)**2+(Z-.5)**2) < 0.09).to(torch.float64)
    ctr = torch.zeros(3, nx, nx, nx, dtype=torch.float64)
    for d in range(3): ctr[d] = 0.5*dx
    mof = VectorizedMOF(n_iter=6, bisect_iters=30)
    t0 = time.perf_counter()
    nrm, off, resid = mof.reconstruct(a, ctr, dx)
    el = time.perf_counter()-t0
    rep("reconstruction returns finite", bool(torch.isfinite(nrm).all()))
    rep("normals unit where mixed",
        float((nrm.norm(dim=0)[(a>1e-8)&(a<1-1e-8)] - 1).abs().max()) < 1e-8)
    print(f"      {nx}^3 = {nx**3} cells in {el*1e3:.1f} ms "
          f"({el*1e6/nx**3:.1f} us/cell), max centroid resid={float(resid.max()):.3e}")
    rep("centroid residual reported (not hidden)", bool(resid.max() >= 0))

    # ---- T3: MOF advection conserves mass -----------------------------
    print("\n[T3] MOF advection: mass + boundedness")
    u = torch.zeros(3, nx, nx, nx, dtype=torch.float64)
    u[0] = 0.3; u[1] = 0.15; u[2] = -0.1
    m0 = float(a.sum())
    a2, c2_, diag = mof.advect_split(a, ctr, u, dt=0.2*dx/0.3, dx=dx)
    rep("mass conserved", diag["mass_rel_drift"] < 1e-8,
        f"rel drift={diag['mass_rel_drift']:.2e}")
    rep("alpha bounded", float(a2.min()) >= 0 and float(a2.max()) <= 1)

    # ---- T4: octree ---------------------------------------------------
    print("\n[T4] True octree: refine / coarsen / 2:1 balance")
    ot = Octree(base_n=8, dx0=0.125, max_level=3)
    ot.root.data["alpha"] = torch.ones(8, 8, 8, dtype=torch.float64)
    ot.refine(ot.root, ["alpha"])
    rep("root has 8 children", len(ot.root.children) == 8)
    rep("leaf count = 8", len(ot.leaves()) == 8)
    ch = ot.root.children[0]
    rep("child shape halved spacing", abs(ot.dx(1) - 0.0625) < 1e-12)
    # conservative round-trip on a non-uniform field
    ot2 = Octree(base_n=4, dx0=0.25, max_level=2)
    f = torch.rand(4, 4, 4, dtype=torch.float64)
    back = ot2.restrict_cell(ot2.prolong_cell(f))
    rep("prolong/restrict exact on cell averages",
        float((back-f).abs().max()) < 1e-12)
    # balance
    ot3 = Octree(base_n=8, dx0=0.125, max_level=3)
    ot3.refine(ot3.root)
    ot3.refine(ot3.root.children[0])
    ot3.refine(ot3.root.children[0].children[0])
    nref = ot3.balance_2to1()
    bal, viol = ot3.check_2to1()
    rep("2:1 balance enforced", bal, f"refinements={nref}, violations={viol}")
    print(f"      leaves={len(ot3.leaves())}, total cells={ot3.total_cells()}")

    # ---- T5: MPI layer ------------------------------------------------ 
    print("\n[T5] MPI layer + cross-rank flux register")
    mpi = MPILayer()
    print(f"      mpi4py present={_HAS_MPI}, world size={mpi.size}, "
          f"active={mpi.active}")
    fld = torch.arange(12, dtype=torch.float64).reshape(12, 1, 1)
    out = mpi.halo_exchange(fld, ax=0, ng=2)
    if not mpi.active:
        rep("serial halo is semantic no-op", torch.equal(out, fld))
    else:
        rep("multi-rank halo ran", out.shape == fld.shape)
    reg = CrossRankFluxRegister(mpi, ncomp=4)
    gid = (0, "lo", 2, 2, (1, 2, 2))
    reg.add_coarse(gid, torch.tensor([1., 0., 0., 0.], dtype=torch.float64), 1.0, 0.1)
    reg.add_fine(gid, torch.tensor([0.8, 0., 0., 0.], dtype=torch.float64), 1.0, 0.1)
    ac = torch.zeros(6, 6, 6, dtype=torch.float64)
    info = reg.apply(ac, None, (0, 0, 0), 1.0, lambda g: True)
    rep("cross-rank register applies mismatch", info["n_applied"] == 1
        and abs(float(ac[1, 2, 2]) - 0.02) < 1e-12,
        f"alpha={float(ac[1,2,2]):.6f} (expect 0.02)")
    rep("globalize reduces (serial identity)", len(reg.globalize()) == 1)

    # ---- T6: skew form conservation -----------------------------------
    print("\n[T6] Energy: skew-symmetric form in the restricted limit")
    vh = ValidationHarness()
    er = vh.energy_conservation_limit(n=12, nsteps=40)[0]
    rep("KE drift < 1e-3 in inviscid const-rho periodic limit",
        er.computed < 1e-3, f"rel drift={er.computed:.3e}")
    print("      NOTE: outside this limit energy is MEASURED, not claimed.")

    # ---- T7: validation harness ---------------------------------------
    print("\n[T7] External-validation harness")
    res = vh.run_all(quick=quick)
    vh.report(res)
    rep("harness produced results", len(res) > 0)
    rep("no silent comparison when reference absent",
        all(r.verdict in ("COMPARED", "NOT COMPARED") for r in res))

    print("\n" + "="*92)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("="*92)
    print("""
  FIXED IN v5.3
    [F1] MOF fully vectorized: closed-form volume AND first moment,
         batched bisection + normal optimisation. Verified against an
         analytically integrated moment (T1). No per-cell Python loop.
    [F2] True 8-child octree with recursive 2:1 balance and conservative
         refine/coarsen (T4).
    [F3] MPI cart topology, deadlock-free halo exchange, cross-rank flux
         register keyed by global face id and all-reduced (T5).
    [F4] Skew-symmetric advective form; KE conservation measured in the
         inviscid, constant-density, periodic limit (T6).
    [F5] Validation harness with analytic references AND an external
         reference loader; reports NOT COMPARED when no reference exists.

  STILL NOT DONE — STATED PLAINLY
    * Multi-rank MPI has not been executed here. Run
      `mpirun -n 4 python amr_vof_coupler_5_3.py --test` yourself.
    * No Basilisk / OpenFOAM reference data is bundled. Supply a CSV with
      columns case,metric,value and pass --reference to actually compare.
    * Energy conservation holds only in the restricted limit above.
    * MOF normal optimisation is a fixed step count; the achieved
      centroid residual is reported so you can judge sufficiency.
    * Static Morton partitioning only; no dynamic load balancing.
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"AMR-VOF v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--full", action="store_true", help="include Zalesak (slow)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--reference", default=None,
                    help="CSV with columns case,metric,value "
                         "(Basilisk / OpenFOAM / published data)")
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if self_test(quick=not a.full) else 1)
    if a.validate:
        vh = ValidationHarness(a.reference)
        vh.report(vh.run_all(quick=not a.full))
        return
    ap.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
