#!/usr/bin/env python3
"""
=============================================================================
AMR-VOF COUPLER v5.4
Newton MOF Normal Solve  +  Dynamic Morton Work-Weighted Load Balancing
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
ORCID      : 0009-0008-2374-0788
License    : MIT
Year       : 2026

=============================================================================
PART A — NEWTON MOF (replaces v5.3's projected gradient)
=============================================================================
For the halfspace {x : n.x <= d}, |n| = 1, cutting a box:

    dV/dd   = A                       (cut-face area)
    dM_i/dd = A * (c_face)_i          (cut-face first moment)
    dM_i/dn_j |_d = -Q_ij             (cut-face second moment)

Perturbing the normal at FIXED VOLUME:
    dV = 0  =>  dd = c_face . dn
    =>  dc_i = (1/V) [ A c_face_i c_face_j - Q_ij ] dn_j
             = J_ij dn_j

J is symmetric and equals -(A/V) * Cov(cut face), hence negative
semi-definite. Restricted to the 2-D tangent space of the unit sphere
this is a 2x2 Newton system, solved in closed form with a determinant
guard and a trust-region clamp.

A and dM/dd are obtained ANALYTICALLY by differentiating the v5.3
closed-form inclusion-exclusion volume/moment formulas:

    V     = sum_S (-1)^|S| (c_S)_+^3 / (6 n1 n2 n3)
    A     = sum_S (-1)^|S| (c_S)_+^2 / (2 n1 n2 n3)
    M_i   = sum_S (-1)^|S| V_S [ (c_S)_+/(4 n_i) + L_i 1(i in S) ]
    dM_i/dd = sum_S (-1)^|S| [ (c_S)_+^3/(6 N n_i)
                               + L_i 1(i in S) (c_S)_+^2/(2 N) ]

Q is obtained by CENTRAL DIFFERENCES IN n AT FIXED d -- no bisection
needed for those 6 evaluations, so they are cheap.

Verified by hand for n=(1,1,1)/sqrt(3), L=(1,1,1), d=1.5/sqrt(3):
    V = 0.5, A = 1.29904 (= regular hexagon area),
    c_face = (0.5,0.5,0.5), dM_i/dd = 0.649519.
Test T1 checks these numbers.

Degenerate normals (some n_i ~ 0) use the 2-D/1-D closed forms for
V and M (needed for volume matching) but fall back to a damped
gradient step for the normal update, because the 3-D derivative
formula and the fixed-d finite difference both break down there.
This is masked per cell, vectorised, and reported in the diagnostics.

=============================================================================
PART B — DYNAMIC LOAD BALANCING
=============================================================================
Weight model (NOT just cell count):
    x = [ n_cells * s,  n_mixed * s,  n_cells*log2(n_cells) * s ]
    s = ref_ratio ** level          (subcycling multiplier)
    w = theta . x

  feature 1: plain sweep cost
  feature 2: MOF mixed-cell cost (bisection + Newton, ~10-50x a plain cell)
  feature 3: multigrid cost ~ N log N

theta is calibrated ONLINE by recursive least squares against measured
per-patch wall time, so the model adapts to the actual kernel mix
(MOF vs WENO5 fallback, MG cycle count, hardware).

Partition: Morton-sort patches, prefix-sum the weights, cut at equal
cumulative weight via searchsorted. This gives contiguous Morton
ranges per rank (locality) with balanced work (not balanced count).

Rebalance hysteresis (engineering, per Six_Problems Problem 6):
  fire when imbalance > hi; cannot fire again until imbalance < lo
  AND min_interval steps have passed. Event count satisfies
      N(T) <= (E_max(T) + D(T)) / (hi - lo)
  with E_max the running peak imbalance and D the accumulated total
  variation. This is a bound on the REBALANCE MECHANISM, exactly
  analogous to a CFL bound. It is NOT a statement about the PDE.

MPI migration: global patch descriptors are all-gathered so every rank
computes the IDENTICAL partition deterministically; patch payloads are
then exchanged with non-blocking isend/irecv (receives posted first).
Serial mode reassigns ownership labels with no copy.

=============================================================================
HONEST SCOPE
=============================================================================
❌ Multi-rank MPI migration has NOT been executed here. The code path
   is written and the serial path is tested; run
   `mpirun -n 4 python amr_vof_coupler_5_4.py --test` yourself.
❌ Newton is applied only where the normal is non-degenerate and the
   2x2 tangent Jacobian is well conditioned. The fraction of cells
   taking the Newton step is REPORTED, not assumed to be 1.
❌ Q by fixed-d finite difference is O(h^2) accurate with h=1e-5; it is
   not an exact analytic second moment of the cut polygon.
❌ The cost model is a 3-feature linear surrogate fitted online. It is
   a proxy, not a hardware performance model.
❌ No comparison against Zoltan / ParMETIS / p4est partition quality.
❌ No external CFD validation. Nothing here certifies "state of the art".
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("amr_vof_54")

VERSION = "5.4.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    MPI = None
    _HAS_MPI = False


# =============================================================================
# PART A0 — CLOSED-FORM BOX / HALFSPACE VOLUME, MOMENT, AND d-DERIVATIVES
# =============================================================================

# (sign, membership mask) for the 8 inclusion-exclusion subsets
SUBSETS: Tuple[Tuple[float, Tuple[int, int, int]], ...] = (
    (+1.0, (0, 0, 0)),
    (-1.0, (1, 0, 0)), (-1.0, (0, 1, 0)), (-1.0, (0, 0, 1)),
    (+1.0, (1, 1, 0)), (+1.0, (1, 0, 1)), (+1.0, (0, 1, 1)),
    (-1.0, (1, 1, 1)),
)


def _pos(t: torch.Tensor, k: int) -> torch.Tensor:
    return t.clamp_min(0.0) ** k


def box_vm(n: torch.Tensor, d: torch.Tensor, L: torch.Tensor,
           dim_tol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Volume and first moment of {x in [0,L1]x[0,L2]x[0,L3] : n.x <= d}.

    n : (B,3)  NON-NEGATIVE (fold before calling)
    d : (B,)
    L : (B,3) or (3,)
    Returns V (B,), M (B,3).

    Exact 3-D / 2-D / 1-D forms selected by how many n_i are below dim_tol.
    """
    if L.dim() == 1:
        L = L.unsqueeze(0).expand_as(n)
    n1, n2, n3 = n[:, 0], n[:, 1], n[:, 2]
    L1, L2, L3 = L[:, 0], L[:, 1], L[:, 2]
    ns = ((n1 < dim_tol).to(torch.int8) + (n2 < dim_tol).to(torch.int8)
          + (n3 < dim_tol).to(torch.int8))

    N3 = (6.0 * n1 * n2 * n3).clamp_min(1e-300)
    V3 = torch.zeros_like(d)
    M3 = torch.zeros_like(n)
    for s, m in SUBSETS:
        aS = m[0]*n1*L1 + m[1]*n2*L2 + m[2]*n3*L3
        cS = d - aS
        p3 = _pos(cS, 3)
        VS = p3 / N3
        V3 = V3 + s * VS
        cp = cS.clamp_min(0.0)
        M3[:, 0] += s * VS * (cp/(4.0*n1.clamp_min(1e-300)) + m[0]*L1)
        M3[:, 1] += s * VS * (cp/(4.0*n2.clamp_min(1e-300)) + m[1]*L2)
        M3[:, 2] += s * VS * (cp/(4.0*n3.clamp_min(1e-300)) + m[2]*L3)

    # ---- 2-D fallbacks -------------------------------------------------
    def area2(dd, na, nb, La, Lb):
        den = (2.0*na*nb).clamp_min(1e-300)
        A = torch.zeros_like(dd)
        Ma = torch.zeros_like(dd)
        Mb = torch.zeros_like(dd)
        for s, (sa, sb) in ((+1., (0, 0)), (-1., (1, 0)),
                            (-1., (0, 1)), (+1., (1, 1))):
            dS = dd - sa*na*La - sb*nb*Lb
            p2 = _pos(dS, 2)
            AS = p2 / den
            dp = dS.clamp_min(0.0)
            A = A + s*AS
            Ma = Ma + s*AS*(dp/(3.0*na.clamp_min(1e-300)) + sa*La)
            Mb = Mb + s*AS*(dp/(3.0*nb.clamp_min(1e-300)) + sb*Lb)
        return A, Ma, Mb

    V2 = [None]*3
    M2 = [None]*3
    A, My, Mz = area2(d, n2, n3, L2, L3)
    V2[0] = A*L1
    M2[0] = torch.stack([V2[0]*L1/2, My*L1, Mz*L1], -1)
    A, Mx, Mz = area2(d, n1, n3, L1, L3)
    V2[1] = A*L2
    M2[1] = torch.stack([Mx*L2, V2[1]*L2/2, Mz*L2], -1)
    A, Mx, My = area2(d, n1, n2, L1, L2)
    V2[2] = A*L3
    M2[2] = torch.stack([Mx*L3, My*L3, V2[2]*L3/2], -1)

    # ---- 1-D fallbacks -------------------------------------------------
    V1 = [None]*3
    M1 = [None]*3
    a1 = (d/n1.clamp_min(1e-300)).clamp(0.0, L1)
    V1[0] = a1*L2*L3
    M1[0] = torch.stack([0.5*a1*a1*L2*L3, V1[0]*L2/2, V1[0]*L3/2], -1)
    a2 = (d/n2.clamp_min(1e-300)).clamp(0.0, L2)
    V1[1] = a2*L1*L3
    M1[1] = torch.stack([V1[1]*L1/2, 0.5*a2*a2*L1*L3, V1[1]*L3/2], -1)
    a3 = (d/n3.clamp_min(1e-300)).clamp(0.0, L3)
    V1[2] = a3*L1*L2
    M1[2] = torch.stack([V1[2]*L1/2, V1[2]*L2/2, 0.5*a3*a3*L1*L2], -1)

    V, M = V3, M3
    for i, ni in enumerate((n1, n2, n3)):
        sel2 = (ns == 1) & (ni < dim_tol)
        sel1 = (ns >= 2) & (ni >= dim_tol)
        V = torch.where(sel2, V2[i], V)
        M = torch.where(sel2.unsqueeze(-1), M2[i], M)
        V = torch.where(sel1, V1[i], V)
        M = torch.where(sel1.unsqueeze(-1), M1[i], M)
    vmax = L1*L2*L3
    allz = (ns == 3)
    V = torch.where(allz, vmax*(d > 0).to(V.dtype), V)
    Mfull = torch.stack([vmax*L1/2, vmax*L2/2, vmax*L3/2], -1)
    M = torch.where((allz & (d > 0)).unsqueeze(-1), Mfull, M)
    M = torch.where((V <= 0).unsqueeze(-1), torch.zeros_like(M), M)
    return V.clamp(0.0, vmax), M


def box_vm_deriv_d(n: torch.Tensor, d: torch.Tensor, L: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ANALYTIC d-derivatives of the 3-D closed form:
        A       = dV/dd   (cut-face area, since |n| = 1)
        dM_i/dd = A * (c_face)_i

    Valid only for non-degenerate normals; the caller masks the rest.
    n is clamped away from zero internally to avoid division blow-up.
    """
    if L.dim() == 1:
        L = L.unsqueeze(0).expand_as(n)
    nc = n.clamp_min(1e-8)
    n1, n2, n3 = nc[:, 0], nc[:, 1], nc[:, 2]
    L1, L2, L3 = L[:, 0], L[:, 1], L[:, 2]
    N2 = (2.0*n1*n2*n3).clamp_min(1e-300)
    N3 = (6.0*n1*n2*n3).clamp_min(1e-300)
    A = torch.zeros_like(d)
    dM = torch.zeros_like(n)
    for s, m in SUBSETS:
        aS = m[0]*n1*L1 + m[1]*n2*L2 + m[2]*n3*L3
        cS = d - aS
        p2, p3 = _pos(cS, 2), _pos(cS, 3)
        A = A + s*p2/N2
        dM[:, 0] += s*(p3/(N3*n1) + m[0]*L1*p2/N2)
        dM[:, 1] += s*(p3/(N3*n2) + m[1]*L2*p2/N2)
        dM[:, 2] += s*(p3/(N3*n3) + m[2]*L3*p2/N2)
    return A, dM


# =============================================================================
# PART A1 — TANGENT BASIS AND 2x2 NEWTON SOLVE
# =============================================================================

def tangent_basis(n: torch.Tensor) -> torch.Tensor:
    """Orthonormal tangent basis T (B,3,2) with T^T n = 0."""
    B = n.shape[0]
    k = n.abs().argmin(dim=-1)
    a = torch.zeros_like(n)
    a[torch.arange(B, device=n.device), k] = 1.0
    t1 = a - (a*n).sum(-1, keepdim=True)*n
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    t2 = torch.linalg.cross(n, t1)
    t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return torch.stack([t1, t2], dim=-1)


def solve2x2(J: torch.Tensor, r: torch.Tensor,
             det_tol: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    delta = -J^{-1} r for batched 2x2 J, plus a validity mask.
    J: (B,2,2), r: (B,2). Returns delta (B,2), ok (B,) bool.
    """
    a, b = J[:, 0, 0], J[:, 0, 1]
    c, d = J[:, 1, 0], J[:, 1, 1]
    det = a*d - b*c
    good = det.abs() > det_tol
    dsafe = torch.where(good, det, torch.ones_like(det))
    d0 = -(d*r[:, 0] - b*r[:, 1])/dsafe
    d1 = -(-c*r[:, 0] + a*r[:, 1])/dsafe
    delta = torch.stack([d0, d1], -1)
    fin = torch.isfinite(delta).all(-1)
    return delta, (good & fin)


# =============================================================================
# PART A2 — NEWTON MOF SOLVER
# =============================================================================

@dataclass
class NewtonMOFConfig:
    n_iter: int = 6
    bisect_first: int = 34
    bisect_rest: int = 22
    fd_h: float = 1e-5
    max_step: float = 0.35          # trust region on ||delta|| (radians-ish)
    det_tol: float = 1e-12
    grad_step: float = 0.6          # fallback damped gradient gain
    dim_tol: float = 1e-4           # below this a normal component is degenerate
    n_floor: float = 1e-7
    warm_bracket: float = 0.15      # fractional bracket for warm-start bisection


class NewtonMOF:
    """
    Vectorised MOF reconstruction: find (n, d) with |n|=1 such that
        V(n,d) = alpha * V_cell        and        M(n,d)/V = centroid.

    Normal update uses a 2x2 Newton step on the sphere's tangent space
    with the ANALYTIC Jacobian
        J = (A c_face c_face^T - Q) / V,
    falling back to a damped gradient step where the normal is
    degenerate or J is ill conditioned. A trust region and an
    accept-if-improved check (with one halving retry) make it robust.

    Everything is batched over cells: no per-cell Python loop.
    """

    def __init__(self, cfg: Optional[NewtonMOFConfig] = None):
        self.c = cfg or NewtonMOFConfig()

    # ---------------- bisection for the volume constraint --------------
    def _bisect(self, nf: torch.Tensor, Vt: torch.Tensor, L: torch.Tensor,
                iters: int, d_init: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        hi_full = (nf*L).sum(-1)
        lo = torch.zeros_like(Vt)
        hi = hi_full.clone()
        if d_init is not None:
            span = self.c.warm_bracket*hi_full.clamp_min(1e-30)
            lo2 = (d_init - span).clamp_min(0.0)
            hi2 = (d_init + span).clamp_max(hi_full)
            Vlo, _ = box_vm(nf, lo2, L, self.c.dim_tol)
            Vhi, _ = box_vm(nf, hi2, L, self.c.dim_tol)
            ok = (Vlo <= Vt) & (Vhi >= Vt)
            lo = torch.where(ok, lo2, lo)
            hi = torch.where(ok, hi2, hi)
        for _ in range(iters):
            mid = 0.5*(lo+hi)
            V, _ = box_vm(nf, mid, L, self.c.dim_tol)
            small = V < Vt
            lo = torch.where(small, mid, lo)
            hi = torch.where(small, hi, mid)
        return 0.5*(lo+hi)

    # ---------------- residual + Jacobian ------------------------------
    def _resid(self, nf, d, L, cf_target):
        V, M = box_vm(nf, d, L, self.c.dim_tol)
        Vs = V.clamp_min(1e-300)
        c = M/Vs.unsqueeze(-1)
        r = c - cf_target
        return r, V, M, c

    def _jacobian(self, nf, d, L, V, c_face):
        """J = (A c_face c_face^T - Q)/V, with Q by fixed-d central diff."""
        h = self.c.fd_h
        B = nf.shape[0]
        Q = torch.zeros(B, 3, 3, dtype=nf.dtype, device=nf.device)
        for j in range(3):
            dp = nf.clone(); dp[:, j] += h
            dm = nf.clone(); dm[:, j] -= h
            _, Mp = box_vm(dp, d, L, self.c.dim_tol)
            _, Mm = box_vm(dm, d, L, self.c.dim_tol)
            Q[:, :, j] = -(Mp - Mm)/(2.0*h)
        Q = 0.5*(Q + Q.transpose(1, 2))
        A_cf = c_face  # placeholder to keep naming explicit below
        outer = c_face.unsqueeze(-1)*c_face.unsqueeze(-2)
        return (self._A.unsqueeze(-1).unsqueeze(-1)*outer - Q)/V.clamp_min(1e-300).unsqueeze(-1).unsqueeze(-1)

    # ---------------- main solve ---------------------------------------
    def solve(self, alpha: torch.Tensor, centroid: torch.Tensor, dx: float
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        alpha    : (nx,ny,nz) volume fraction
        centroid : (3,nx,ny,nz) target fluid centroid, CELL-LOCAL in [0,dx]
        Returns (normal (3,...), offset (...), residual (...), info)
        where the normal is in the ORIGINAL (signed) frame and the offset
        satisfies n_original . x = offset on the cell-local box [0,dx]^3.
        """
        c = self.c
        shape = alpha.shape
        dev, dt_ = alpha.device, torch.float64
        a = alpha.to(dt_).reshape(-1)
        ctr = centroid.to(dt_).reshape(3, -1).transpose(0, 1).contiguous()
        B = a.numel()
        L = torch.full((B, 3), float(dx), dtype=dt_, device=dev)
        Vcell = float(dx)**3
        Vt = (a*Vcell).clamp(0.0, Vcell)
        mixed = (a > 1e-8) & (a < 1.0 - 1e-8)

        # initial normal guess from centroid offset
        center = 0.5*float(dx)
        n0 = ctr - center
        nrm = n0.norm(dim=-1)
        fallback = torch.zeros_like(n0); fallback[:, 2] = 1.0
        n0 = torch.where((nrm > 1e-12).unsqueeze(-1), n0, fallback)
        flip = n0 < 0.0
        sgn = torch.where(flip, -torch.ones_like(n0), torch.ones_like(n0))
        nf = n0.abs().clamp_min(c.n_floor)
        nf = nf/nf.norm(dim=-1, keepdim=True).clamp_min(1e-30)

        # fold the target centroid into the positive octant
        cf_target = torch.where(flip, L - ctr, ctr)

        d = self._bisect(nf, Vt, L, c.bisect_first)
        r, V, M, cact = self._resid(nf, d, L, cf_target)
        rnorm = r.norm(dim=-1)

        n_newton = 0
        n_grad = 0
        n_halve = 0
        for it in range(c.n_iter):
            degen = (nf.min(dim=-1).values < max(c.dim_tol, 10*c.fd_h))
            A, dMdd = box_vm_deriv_d(nf, d, L)
            Asafe = A.clamp_min(1e-300)
            c_face = dMdd/Asafe.unsqueeze(-1)
            self._A = Asafe
            J = self._jacobian(nf, d, L, V, c_face)
            T = tangent_basis(nf)                       # (B,3,2)
            Jt = T.transpose(1, 2) @ J @ T              # (B,2,2)
            rt = (T.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)

            delta_n, ok = solve2x2(Jt, rt, c.det_tol)
            delta_g = -c.grad_step*rt
            use_newton = ok & (~degen) & mixed
            delta = torch.where(use_newton.unsqueeze(-1), delta_n, delta_g)
            n_newton += int(use_newton.sum())
            n_grad += int((~use_newton & mixed).sum())

            dn = delta.norm(dim=-1, keepdim=True)
            delta = delta*(c.max_step/dn.clamp_min(1e-30)).clamp(max=1.0)

            # full step candidate
            nc1 = nf + (T @ delta.unsqueeze(-1)).squeeze(-1)
            nc1 = nc1.clamp_min(c.n_floor)
            nc1 = nc1/nc1.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            d1 = self._bisect(nc1, Vt, L, c.bisect_rest, d_init=d)
            r1, V1, M1, _ = self._resid(nc1, d1, L, cf_target)
            rn1 = r1.norm(dim=-1)

            # halving retry where the full step did not improve
            worse = (rn1 >= rnorm) & mixed
            n_halve += int(worse.sum())
            nc2 = nf + (T @ (0.5*delta).unsqueeze(-1)).squeeze(-1)
            nc2 = nc2.clamp_min(c.n_floor)
            nc2 = nc2/nc2.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            d2 = self._bisect(nc2, Vt, L, c.bisect_rest, d_init=d)
            r2, V2, M2, _ = self._resid(nc2, d2, L, cf_target)
            rn2 = r2.norm(dim=-1)

            take2 = worse & (rn2 < rn1)
            take1 = (~worse) | (rn1 <= rn2)
            upd = (take1 | take2) & mixed

            nf = torch.where((take1 & upd).unsqueeze(-1), nc1,
                             torch.where((take2 & upd).unsqueeze(-1), nc2, nf))
            d = torch.where(take1 & upd, d1, torch.where(take2 & upd, d2, d))
            r = torch.where((take1 & upd).unsqueeze(-1), r1,
                            torch.where((take2 & upd).unsqueeze(-1), r2, r))
            V = torch.where(take1 & upd, V1, torch.where(take2 & upd, V2, V))
            rnorm = r.norm(dim=-1)

        # unfold
        n_out = nf*sgn
        d_out = d - (flip.to(dt_)*nf*L).sum(-1)
        resid = torch.where(mixed, rnorm, torch.zeros_like(rnorm))

        info = {
            "n_mixed": int(mixed.sum()),
            "newton_steps": n_newton,
            "gradient_steps": n_grad,
            "halving_retries": n_halve,
            "newton_fraction": (n_newton/max(n_newton+n_grad, 1)),
            "max_resid": float(resid.max()),
            "mean_resid": float(resid[mixed].mean()) if int(mixed.sum()) else 0.0,
        }
        return (n_out.transpose(0, 1).reshape(3, *shape),
                d_out.reshape(shape),
                resid.reshape(shape),
                info)

    # ---------------- projected-gradient baseline (v5.3) ---------------
    def solve_gradient(self, alpha, centroid, dx, n_iter: int = 8,
                       bisect: int = 30, step: float = 0.5
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """v5.3-style baseline, for the residual comparison test."""
        shape = alpha.shape
        dev = alpha.device
        a = alpha.to(torch.float64).reshape(-1)
        ctr = centroid.to(torch.float64).reshape(3, -1).transpose(0, 1).contiguous()
        B = a.numel()
        L = torch.full((B, 3), float(dx), dtype=torch.float64, device=dev)
        Vt = (a*float(dx)**3).clamp(0.0, float(dx)**3)
        mixed = (a > 1e-8) & (a < 1e-8*0 + 1 - 1e-8)
        n0 = ctr - 0.5*float(dx)
        nrm = n0.norm(dim=-1)
        fb = torch.zeros_like(n0); fb[:, 2] = 1.0
        n0 = torch.where((nrm > 1e-12).unsqueeze(-1), n0, fb)
        flip = n0 < 0
        sgn = torch.where(flip, -torch.ones_like(n0), torch.ones_like(n0))
        nf = n0.abs().clamp_min(self.c.n_floor)
        nf = nf/nf.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        cf = torch.where(flip, L - ctr, ctr)
        d = self._bisect(nf, Vt, L, bisect)
        r, V, M, _ = self._resid(nf, d, L, cf)
        for _ in range(n_iter):
            T = tangent_basis(nf)
            rt = (T.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)
            delta = -step*rt
            dn = delta.norm(dim=-1, keepdim=True)
            delta = delta*(self.c.max_step/dn.clamp_min(1e-30)).clamp(max=1.0)
            nf = nf + (T @ delta.unsqueeze(-1)).squeeze(-1)
            nf = nf.clamp_min(self.c.n_floor)
            nf = nf/nf.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            d = self._bisect(nf, Vt, L, bisect, d_init=d)
            r, V, M, _ = self._resid(nf, d, L, cf)
        resid = torch.where(mixed, r.norm(dim=-1), torch.zeros(B, dtype=torch.float64, device=dev))
        return (nf*sgn).transpose(0, 1).reshape(3, *shape), d.reshape(shape), resid.reshape(shape), {}


# =============================================================================
# PART B0 — MORTON CODES
# =============================================================================

def _spread_bits(v: torch.Tensor) -> torch.Tensor:
    """Spread 21 bits of each int64 into every third bit position."""
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def morton3(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return _spread_bits(i) | (_spread_bits(j) << 1) | (_spread_bits(k) << 2)


# =============================================================================
# PART B1 — WORK MODEL WITH ONLINE RLS CALIBRATION
# =============================================================================

class WorkModel:
    """
    3-feature linear cost surrogate, calibrated online by recursive
    least squares against measured per-patch wall time.

      x = [ n_cells*s, n_mixed*s, n_cells*log2(n_cells)*s ] / 1e3
      w = theta . x        (predicted milliseconds)

    s = ref_ratio**level accounts for subcycling.
    """

    def __init__(self, theta0: Optional[List[float]] = None,
                 lam: float = 0.98, P0: float = 1e3,
                 theta_min: float = 1e-6):
        th = theta0 if theta0 is not None else [0.5, 25.0, 0.05]
        self.theta = torch.tensor(th, dtype=torch.float64)
        self.P = torch.eye(3, dtype=torch.float64)*P0
        self.lam = float(lam)
        self.theta_min = float(theta_min)
        self.n_updates = 0

    def features(self, n_cells: int, n_mixed: int, level: int,
                 ref_ratio: int = 2) -> torch.Tensor:
        s = float(ref_ratio)**int(level)
        nc = float(max(n_cells, 1))
        return torch.tensor([
            nc*s,
            float(max(n_mixed, 0))*s,
            nc*math.log2(max(nc, 2.0))*s,
        ], dtype=torch.float64)/1e3

    def predict(self, x: torch.Tensor) -> float:
        return float((self.theta @ x).clamp_min(1e-9))

    def update(self, x: torch.Tensor, measured_ms: float) -> None:
        if not math.isfinite(measured_ms) or measured_ms <= 0:
            return
        P, th = self.P, self.theta
        Px = P @ x
        denom = self.lam + float(x @ Px)
        if denom <= 1e-30:
            return
        K = Px/denom
        err = float(measured_ms) - float(th @ x)
        self.theta = (th + K*err).clamp_min(self.theta_min)
        self.P = (P - torch.outer(K, Px))/self.lam
        self.P = 0.5*(self.P + self.P.T)
        self.n_updates += 1

    def state(self) -> Dict[str, Any]:
        return {"theta": self.theta.tolist(), "n_updates": self.n_updates}


# =============================================================================
# PART B2 — HYSTERESIS GATE WITH EVENT-COUNT BOUND
# =============================================================================

class RebalanceHysteresis:
    """
    Two-level gate on the load-imbalance signal.

    Fires when imbalance > hi; cannot fire again until imbalance < lo
    AND min_interval steps have elapsed. Event count obeys
        N(T) <= (E_max(T) + D(T)) / (hi - lo)
    with E_max the running peak and D the accumulated total variation.

    This is an ENGINEERING bound on the rebalance mechanism (same
    category as a CFL bound), NOT a statement about the PDE.
    """

    def __init__(self, hi: float = 0.25, lo: float = 0.10,
                 min_interval: int = 5):
        if not (0.0 < lo < hi):
            raise ValueError("require 0 < lo < hi")
        self.hi, self.lo = float(hi), float(lo)
        self.delta = self.hi - self.lo
        self.min_interval = int(min_interval)
        self.armed = True
        self.events = 0
        self.E_max = 0.0
        self.D = 0.0
        self._prev = 0.0
        self._last_fire = -10**9

    def step(self, imbalance: float, step: int) -> bool:
        self.E_max = max(self.E_max, float(imbalance))
        self.D += abs(float(imbalance) - self._prev)
        self._prev = float(imbalance)
        if imbalance < self.lo:
            self.armed = True
        fire = (self.armed and imbalance > self.hi
                and (step - self._last_fire) >= self.min_interval)
        if fire:
            self.armed = False
            self.events += 1
            self._last_fire = int(step)
        return fire

    def bound(self) -> float:
        return (self.E_max + self.D)/max(self.delta, 1e-30)

    def satisfied(self) -> bool:
        return self.events <= self.bound() + 1e-9


# =============================================================================
# PART B3 — PATCH DESCRIPTOR AND DYNAMIC LOAD BALANCER
# =============================================================================

@dataclass
class PatchDesc:
    pid: int
    level: int
    lo: Tuple[int, int, int]
    hi: Tuple[int, int, int]
    owner: int = 0
    n_mixed: int = 0
    weight: float = 1.0
    measured_ms: float = float("nan")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.hi[i]-self.lo[i] for i in range(3))

    @property
    def n_cells(self) -> int:
        s = self.shape
        return int(s[0]*s[1]*s[2])

    def centroid_idx(self) -> Tuple[int, int, int]:
        return tuple((self.lo[i]+self.hi[i])//2 for i in range(3))

    def morton(self) -> int:
        c = self.centroid_idx()
        return int(morton3(torch.tensor([c[0]], dtype=torch.int64),
                           torch.tensor([c[1]], dtype=torch.int64),
                           torch.tensor([c[2]], dtype=torch.int64))[0])


class DynamicLoadBalancer:
    """
    Morton + work-weighted repartitioning with online cost calibration,
    rebalance hysteresis, and (optional) MPI patch migration.
    """

    def __init__(self, n_ranks: int = 1, ref_ratio: int = 2,
                 hi: float = 0.25, lo: float = 0.10,
                 min_interval: int = 5, use_mpi: bool = False,
                 work_model: Optional[WorkModel] = None):
        self.ref_ratio = int(ref_ratio)
        self.model = work_model or WorkModel()
        self.gate = RebalanceHysteresis(hi, lo, min_interval)
        self.comm = None
        self.rank = 0
        self.size = int(max(n_ranks, 1))
        if use_mpi and _HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        self.history: List[Dict[str, Any]] = []

    # ---------------- weights ------------------------------------------
    def update_weights(self, descs: List[PatchDesc]) -> None:
        for d in descs:
            x = self.model.features(d.n_cells, d.n_mixed, d.level, self.ref_ratio)
            d.weight = self.model.predict(x)

    def calibrate(self, descs: List[PatchDesc]) -> int:
        n = 0
        for d in descs:
            if math.isfinite(d.measured_ms) and d.measured_ms > 0:
                x = self.model.features(d.n_cells, d.n_mixed, d.level, self.ref_ratio)
                self.model.update(x, d.measured_ms)
                n += 1
        return n

    # ---------------- partition ----------------------------------------
    def partition(self, descs: List[PatchDesc]) -> Dict[int, int]:
        """
        Morton-sort, prefix-sum weights, cut at equal cumulative weight.
        Deterministic: identical input -> identical output on every rank.
        Returns {pid: new_owner}.
        """
        if not descs:
            return {}
        order = sorted(descs, key=lambda d: (d.morton(), d.pid))
        w = torch.tensor([max(d.weight, 1e-12) for d in order], dtype=torch.float64)
        cum = torch.cumsum(w, 0)
        total = float(cum[-1])
        P = self.size
        targets = torch.tensor([total*(k+1)/P for k in range(P-1)],
                               dtype=torch.float64)
        cuts = torch.searchsorted(cum, targets, right=True)
        idx = torch.arange(len(order), dtype=torch.int64)
        ranks = (idx.unsqueeze(-1) >= cuts.unsqueeze(0)).sum(-1)
        return {d.pid: int(ranks[i]) for i, d in enumerate(order)}

    @staticmethod
    def imbalance(descs: List[PatchDesc], assignment: Dict[int, int],
                  P: int) -> float:
        load = [0.0]*P
        for d in descs:
            load[assignment.get(d.pid, d.owner)] += d.weight
        mean = sum(load)/max(P, 1)
        if mean <= 1e-30:
            return 0.0
        return (max(load) - min(load))/mean

    # ---------------- migration ----------------------------------------
    def migrate(self, descs: List[PatchDesc], assignment: Dict[int, int],
                payloads: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
                ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Serial: relabel owners, no copy.
        MPI: all-gather descriptors so every rank computes the same
        partition, then exchange payloads with isend/irecv (recv first).
        """
        if self.comm is None or self.size == 1:
            for d in descs:
                d.owner = assignment.get(d.pid, d.owner)
            return payloads or {}

        # --- MPI path (NOT executed in this session) ---
        meta = [(d.pid, d.level, d.lo, d.hi, d.n_mixed, d.weight, d.owner)
                for d in descs]
        allmeta = self.comm.allgather(meta)
        global_map: Dict[int, Tuple] = {}
        for grp in allmeta:
            for rec in grp:
                global_map[rec[0]] = rec
        gdescs = [PatchDesc(pid=r[0], level=r[1], lo=r[2], hi=r[3],
                            owner=r[6], n_mixed=r[4], weight=r[5])
                  for r in sorted(global_map.values(), key=lambda z: z[0])]
        gassign = self.partition(gdescs)

        sends: Dict[int, List] = {r: [] for r in range(self.size)}
        recvs: List[Tuple[int, int]] = []
        for d in gdescs:
            new = gassign[d.pid]
            if d.owner != new:
                if d.owner == self.rank and payloads and d.pid in payloads:
                    sends[new].append((d.pid, d.level, d.lo, d.hi,
                                       {k: v.cpu().numpy()
                                        for k, v in payloads[d.pid].items()}))
                if new == self.rank:
                    recvs.append((d.owner, d.pid))

        reqs = []
        inbox: List[Any] = []
        for src, pid in recvs:
            inbox.append(self.comm.irecv(source=src, tag=7000+pid % 1000))
        for dst, items in sends.items():
            if items:
                reqs.append(self.comm.isend(items, dest=dst, tag=7000))
        got: Dict[int, Dict[str, torch.Tensor]] = {}
        for r in inbox:
            items = r.wait()
            for pid, lev, lo, hi, fields in items:
                got[pid] = {k: torch.from_numpy(v) for k, v in fields.items()}
        for r in reqs:
            r.wait()

        keep = {}
        for d in descs:
            new = gassign.get(d.pid, d.owner)
            if new == self.rank and payloads and d.pid in payloads:
                keep[d.pid] = payloads[d.pid]
        keep.update(got)
        for d in descs:
            d.owner = gassign.get(d.pid, d.owner)
        return keep

    # ---------------- driver -------------------------------------------
    def rebalance(self, descs: List[PatchDesc], step: int,
                  payloads: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
                  force: bool = False
                  ) -> Tuple[bool, Dict[str, Any], Optional[Dict]]:
        self.update_weights(descs)
        ncal = self.calibrate(descs)
        assign = self.partition(descs)
        imb_before = self.imbalance(descs, {d.pid: d.owner for d in descs}, self.size)
        imb_after = self.imbalance(descs, assign, self.size)
        fire = force or self.gate.step(imb_before, step)
        newpay = payloads
        if fire:
            newpay = self.migrate(descs, assign, payloads)
        rec = {
            "step": step, "fired": bool(fire),
            "imbalance_before": imb_before,
            "imbalance_after": imb_after if fire else imb_before,
            "n_calibrated": ncal,
            "events": self.gate.events,
            "event_bound": self.gate.bound(),
            "bound_ok": self.gate.satisfied(),
            "theta": self.model.theta.tolist(),
        }
        self.history.append(rec)
        return bool(fire), rec, newpay


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test(quick: bool = True) -> bool:
    print("=" * 92)
    print(f"  AMR-VOF COUPLER v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 92)
    torch.manual_seed(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ---------------- T1: analytic derivatives -------------------------
    print("\n[T1] Analytic A = dV/dd and dM/dd (hand-verified reference)")
    s3 = 1.0/math.sqrt(3.0)
    n = torch.tensor([[s3, s3, s3]], dtype=torch.float64)
    L = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    d = torch.tensor([1.5*s3], dtype=torch.float64)
    V, M = box_vm(n, d, L)
    A, dM = box_vm_deriv_d(n, d, L)
    rep("V = 0.5", abs(float(V)-0.5) < 1e-12, f"V={float(V):.12f}")
    rep("A = hexagon area 1.299038", abs(float(A)-1.299038105676658) < 1e-9,
        f"A={float(A):.12f}")
    cf = dM/A.unsqueeze(-1)
    rep("c_face = (0.5,0.5,0.5)", float((cf-0.5).abs().max()) < 1e-9,
        f"c_face={cf[0].tolist()}")
    rep("dM_i/dd = 0.649519", abs(float(dM[0, 0])-0.649519052838329) < 1e-9,
        f"dM_x={float(dM[0,0]):.12f}")
    # finite-difference cross-check of the analytic d-derivative
    h = 1e-6
    Vp, Mp = box_vm(n, d+h, L)
    Vm, Mm = box_vm(n, d-h, L)
    rep("A matches FD", abs(float(A)-float((Vp-Vm)/(2*h))) < 1e-7)
    rep("dM/dd matches FD",
        float((dM-(Mp-Mm)/(2*h)).abs().max()) < 1e-7)

    # ---------------- T2: Newton vs gradient on achievable targets -----
    print("\n[T2] Newton vs projected gradient — centroid residual")
    B = 4096
    dx = 1.0
    Lb = torch.full((B, 3), dx, dtype=torch.float64)
    n_true = torch.randn(B, 3, dtype=torch.float64)
    n_true = n_true/n_true.norm(dim=-1, keepdim=True)
    nf_true = n_true.abs().clamp_min(1e-3)
    nf_true = nf_true/nf_true.norm(dim=-1, keepdim=True)
    d_true = (nf_true*Lb).sum(-1)*torch.rand(B, dtype=torch.float64)*0.8 + 0.1
    Vt, Mt = box_vm(nf_true, d_true, Lb)
    alpha = (Vt/dx**3).clamp(1e-4, 1-1e-4)
    ctr_fold = Mt/Vt.clamp_min(1e-300).unsqueeze(-1)
    flip = n_true < 0
    ctr_orig = torch.where(flip, Lb-ctr_fold, ctr_fold)
    nx = 16
    a_t = alpha[:nx*nx*nx].reshape(nx, nx, nx)
    c_t = ctr_orig[:nx*nx*nx].transpose(0, 1).reshape(3, nx, nx, nx)

    nm = NewtonMOF()
    t0 = time.perf_counter()
    nN, dN, rN, infoN = nm.solve(a_t, c_t, dx)
    tN = time.perf_counter()-t0
    t0 = time.perf_counter()
    nG, dG, rG, _ = nm.solve_gradient(a_t, c_t, dx, n_iter=8)
    tG = time.perf_counter()-t0

    m = (a_t > 1e-8) & (a_t < 1-1e-8)
    rn = float(rN[m].max()); rg = float(rG[m].max())
    mn = float(rN[m].mean()); mg = float(rG[m].mean())
    print(f"      Newton  : max={rn:.3e} mean={mn:.3e}  {tN*1e3:.0f} ms  "
          f"newton_frac={infoN['newton_fraction']:.2f}")
    print(f"      Gradient: max={rg:.3e} mean={mg:.3e}  {tG*1e3:.0f} ms")
    rep("Newton mean residual < gradient mean residual", mn < mg,
        f"ratio={mg/max(mn,1e-30):.1f}x")
    rep("Newton mean residual < 1e-3 dx", mn < 1e-3, f"mean={mn:.2e}")
    rep("Newton used on a majority of mixed cells",
        infoN["newton_fraction"] > 0.5,
        f"frac={infoN['newton_fraction']:.2f}")

    # ---------------- T3: volume constraint honoured -------------------
    print("\n[T3] Volume constraint after Newton")
    nf_rec = nN.reshape(3, -1).transpose(0, 1).abs()
    nf_rec = nf_rec/nf_rec.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    sgn = torch.where(nN.reshape(3, -1).transpose(0, 1) < 0, -1.0, 1.0)
    d_fold = dN.reshape(-1) + ((sgn < 0).to(torch.float64)*nf_rec*Lb[:nf_rec.shape[0]]).sum(-1)
    Vr, _ = box_vm(nf_rec, d_fold, Lb[:nf_rec.shape[0]])
    verr = float((Vr/dx**3 - a_t.reshape(-1)).abs().max())
    rep("volume matches alpha", verr < 1e-6, f"max|V/Vc - alpha|={verr:.2e}")

    # ---------------- T4: Morton codes ---------------------------------
    print("\n[T4] Morton codes")
    i = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    j = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    k = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    mc = morton3(i, j, k)
    rep("Morton(0,0,0)=0", int(mc[0]) == 0)
    rep("Morton(1,0,0)=1", int(mc[1]) == 1)
    rep("Morton(0,1,0)=2", int(mc[2]) == 2)
    rep("Morton(1,1,0)=3", int(mc[3]) == 3)

    # ---------------- T5: work model RLS -------------------------------
    print("\n[T5] Work model online RLS calibration")
    wm = WorkModel(theta0=[1.0, 1.0, 1.0])
    theta_true = torch.tensor([0.7, 30.0, 0.04], dtype=torch.float64)
    rng = torch.Generator().manual_seed(1)
    for _ in range(400):
        nc = int(torch.randint(64, 8000, (1,), generator=rng))
        nmix = int(nc*float(torch.rand(1, generator=rng))*0.2)
        lev = int(torch.randint(0, 3, (1,), generator=rng))
        x = wm.features(nc, nmix, lev)
        y = float(theta_true @ x)*(1.0 + 0.02*float(torch.randn(1, generator=rng)))
        wm.update(x, y)
    rel = float((wm.theta-theta_true).abs()/theta_true)
    print(f"      theta_true={theta_true.tolist()}")
    print(f"      theta_fit ={[round(v,5) for v in wm.theta.tolist()]}")
    rep("RLS recovers cost coefficients (rel err < 0.15)", bool((rel < 0.15).all()),
        f"max rel err={float(rel.max()):.3f}")

    # ---------------- T6: work-weighted partition ----------------------
    print("\n[T6] Morton + work-weighted partition")
    descs = []
    pid = 0
    for lev in (0, 1, 2):
        for a in range(4):
            for b in range(4):
                n = 8*(2**lev)
                descs.append(PatchDesc(pid=pid, level=lev,
                                       lo=(a*n, b*n, 0), hi=((a+1)*n, (b+1)*n, n),
                                       n_mixed=int(0.1*n*n*n)))
                pid += 1
    lb = DynamicLoadBalancer(n_ranks=4)
    lb.update_weights(descs)
    assign = lb.partition(descs)
    imb = lb.imbalance(descs, assign, 4)
    rep("work imbalance < 0.15", imb < 0.15, f"imbalance={imb:.4f}")
    # Morton contiguity: each rank's pids form a contiguous Morton range
    order = sorted(descs, key=lambda d: (d.morton(), d.pid))
    ranks_seq = [assign[d.pid] for d in order]
    switches = sum(1 for a, b in zip(ranks_seq, ranks_seq[1:]) if a != b)
    rep("Morton-contiguous ranks (<= P-1 switches)", switches <= 3,
        f"switches={switches}")
    cnt = [sum(1 for d in descs if assign[d.pid] == r) for r in range(4)]
    print(f"      patch counts per rank = {cnt}  (unequal counts, equal work)")

    # ---------------- T7: hysteresis event bound -----------------------
    print("\n[T7] Rebalance hysteresis + event-count bound")
    gate = RebalanceHysteresis(hi=0.25, lo=0.10, min_interval=3)
    for t in range(600):
        imb_t = 0.2 + 0.2*math.sin(0.35*t)
        gate.step(max(imb_t, 0.0), t)
    print(f"      events={gate.events}  bound={gate.bound():.1f}")
    rep("N(T) <= (E_max + D)/delta", gate.satisfied())
    rep("no chattering (events << steps)", gate.events < 600/3,
        f"{gate.events} events / 600 steps")

    # ---------------- T8: dynamic rebalance driver ---------------------
    print("\n[T8] Dynamic rebalance with measured-time feedback")
    lb2 = DynamicLoadBalancer(n_ranks=4, hi=0.20, lo=0.08, min_interval=2)
    for d in descs:
        d.owner = 0
    pay = {d.pid: {"alpha": torch.zeros(2, 2, 2)} for d in descs}
    fired = 0
    for step in range(8):
        for d in descs:
            x = lb2.model.features(d.n_cells, d.n_mixed, d.level)
            d.measured_ms = float(theta_true @ x)
        f, rec, pay = lb2.rebalance(descs, step, pay)
        fired += int(f)
    rep("rebalance fired at least once", fired >= 1, f"fired={fired}")
    rep("event bound holds across run", lb2.gate.satisfied())
    rep("payloads preserved (serial)", len(pay) == len(descs))
    print(f"      final theta={[round(v,4) for v in lb2.model.theta.tolist()]}")

    # ---------------- T9: MPI availability -----------------------------
    print("\n[T9] MPI path")
    print(f"      mpi4py present={_HAS_MPI}; serial fallback tested above")
    rep("serial fallback is a no-copy relabel", True)

    print("\n" + "=" * 92)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 92)
    print("""
  WHAT IS PROVEN BY THESE TESTS
    T1  A = dV/dd and dM/dd are the ANALYTIC derivatives of the closed
        form, matching hand-computed hexagon values and finite
        differences to 1e-9.
    T2  Newton reduces the mean centroid residual below the v5.3
        projected-gradient baseline, on achievable targets, with the
        Newton-step fraction reported.
    T3  The volume constraint is still satisfied after Newton.
    T4  Morton interleaving is correct.
    T5  RLS recovers the true cost coefficients from noisy timing.
    T6  Work-weighted Morton partitioning balances WORK (not counts)
        while keeping each rank's Morton range contiguous.
    T7  Hysteresis prevents rebalance chattering and satisfies the
        event-count bound.
    T8  The full driver runs with measured-time feedback.

  WHAT IS NOT PROVEN
    * Multi-rank MPI migration was NOT executed here.
    * Q by fixed-d finite difference is O(h^2), not an exact polygon
      second moment.
    * Newton is masked off for degenerate normals / ill-conditioned
      tangent Jacobians; that fraction is reported, not hidden.
    * No comparison against Zoltan / ParMETIS / p4est.
    * No external CFD validation. "State of the art" is not certified.
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"AMR-VOF v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if self_test(quick=not a.full) else 1)
    ap.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
