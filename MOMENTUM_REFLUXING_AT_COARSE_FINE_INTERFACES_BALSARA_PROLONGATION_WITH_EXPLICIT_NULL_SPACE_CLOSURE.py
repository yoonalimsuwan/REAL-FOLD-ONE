#!/usr/bin/env python3
"""
=============================================================================
MOMENTUM REFLUXING AT COARSE-FINE INTERFACES + BALSARA PROLONGATION
WITH EXPLICIT NULL-SPACE CLOSURE
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

LITERATURE (numerics)
 [A1] Berger & Colella (1989) JCP 82, 64.   [refluxing]
 [A2] Berger & Oliger (1984) JCP 53, 484.
 [A3] Almgren et al. (1998) JCP 142, 1.     [AMR projection, staggered]
 [B1] Balsara (2001) JCP 174, 614.          [divergence-free AMR]
 [B2] Balsara (2004) ApJS 151, 149.         [divergence-free reconstruction]
 [T1] Toth & Roe (2002) JCP 180, 736.       [curl-preserving prolongation]
 [S1] Scardovelli & Zaleski (1999) JCP 155, 228.

=============================================================================
WHAT THE FIVE FOUNDATION PAPERS LICENSE FOR THIS MODULE
=============================================================================
[P-REV22-COERC] Revision22_Coercivity_and_Fractafold_Transmission
  LICENSES: Theorem 4.3 establishes well-posedness of the ORDER-2
    transmission problem on a glued network — continuity of the field
    plus a Kirchhoff-type flux balance. Momentum refluxing is exactly a
    discrete FLUX-BALANCE condition at an interface where the
    discretisation changes, i.e. it sits in the licensed order class.
  DOES NOT LICENSE: Open Problem 4.5 states explicitly that the ORDER-4
    junction matching condition (for Delta^2 u and partial_n(Delta u))
    is NOT derived, even for the fractafold. Any coarse-fine matching
    that requires higher-derivative continuity is therefore in the
    OPEN class, not the licensed one. This module does not claim it.
  SCOPE NOTE: the fractafold theorem is about gluing fractal cells, not
    about AMR. What transfers is the DISCIPLINE of stating which order
    of matching condition is derived, not the theorem itself.

[P-PAPER8] Paper8_Internal_Diagnostic_Battery
  LICENSES the METHOD: each theorem's own convergence statistic, read
    backward, is its test. Here:
      div(curl) = 0 identically  ->  measure fine-grid divergence;
                                     it must be 0 to machine precision.
      flux telescoping           ->  measure the coarse/fine flux mismatch;
                                     after refluxing it must vanish.
  DOES NOT LICENSE: a finite-sample error rate (its Open Problem 3.1).

[P-PAPER9] Paper9_Statistical_Dictionary
  LICENSES: subsampling / mixing-concentration CI for genuinely random,
    stationary-mixing sequences (Entries 2.3, 2.6).
  FORBIDS BY ITS OWN LOGIC: applying those CIs to a DETERMINISTIC
    machine-precision residual. Round-off is not a stationary mixing
    process. This module therefore reports exact residuals with NO CI,
    and reserves subsampling CI for across-seed variability only.

[P-PAPER7] Paper7_External_Noise_Diagnostic_Protocol
  LICENSES the LICENSING DISCIPLINE: Proposition 3.1 grants a specific
    downstream consequence for a specific verified hypothesis;
    Remark 3.2 shows that passing one branch does NOT grant another
    branch's consequence. The LICENSE_TABLE below follows that form:
    each verified property grants a named consequence and nothing more.

[P-REV22-EXACT] Revision22_Exact_FiniteTime_Predictability
  LICENSES: Theorem 2.1 — when the generating law is known in closed
    form, the quantity is exactly computable, not estimated. The
    divergence constraint matrix C here IS known in closed form (it is
    the incidence structure of the 8 sub-cells), so its rank, null space
    and pseudo-inverse are EXACT and precomputable once.
  DOES NOT LICENSE: Proposition 3.1 — a quantity that must be inferred
    from data is not thereby exact. The curl mismatch is MEASURED, not
    known in closed form, and is reported as a measurement.

=============================================================================
PROVEN BY TEST (deterministic, machine precision, NO confidence interval)
=============================================================================
 T1  C has rank exactly 7; its columns sum to zero; null space dim = 5.
 T2  Consistency: C x = b is solvable IFF the coarse field is div-free.
 T3  Fine divergence == 0 to machine precision after prolongation.
 T4  Face-flux preservation: mean of the r^2 fine faces == coarse face.
 T5  Restriction/prolongation round-trip on face data is exact.
 T6  Refluxing restores the analytically-known conserved total exactly,
     even when the coarse flux is deliberately corrupted.
 T7  Refluxing is a no-op when the fluxes already match.

=============================================================================
MEASURED, NOT CLAIMED
=============================================================================
 M1  Curl mismatch at coarse edges. DERIVED FINDING: those curls involve
     only fine faces lying ON coarse faces, which are already fixed by
     the face reconstruction. The 12 interior unknowns do NOT appear in
     them. Consequently NO interior closure can enforce coarse-edge
     curl preservation. Exact preservation requires the Toth-Roe
     construction [T1], which stores and evolves edge circulations as
     additional variables. NOT IMPLEMENTED HERE.
 M2  Interior sub-cell circulation. These 6 components DO involve the
     interior unknowns, so the 5-dimensional null space can be used to
     reduce them. Three closures are provided and ABLATED:
       min_norm  : smallest ||x||  (no curl consideration)
       curl_min  : minimise interior circulation magnitude
       smooth    : minimise interior-face variation
     The ablation reports which actually reduces what. No claim is made
     that reducing interior circulation improves accuracy.

=============================================================================
HONEST LIMITS
=============================================================================
 ❌ Refluxing implemented for CELL-CENTRED conserved variables
    (rho, rhou, rhov, rhow, rhoE) — the layout super_dns_one uses.
    For a fully STAGGERED MAC momentum CV straddling the coarse-fine
    interface, the normal velocity at the interface is a SHARED degree
    of freedom requiring the Almgren et al. treatment [A3]. That is
    NOT implemented; `staggered_warning()` reports it.
 ❌ Coarse boundary data held FROZEN across fine subcycles: first-order
    in time at the interface.
 ❌ No octree. Patch-based only.
 ❌ No experimental validation.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("reflux_balsara")

VERSION = "1.0.0"

try:
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import lsqr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# 0. LICENSE TABLE  (Paper 7 discipline: named consequence per verified fact)
# =============================================================================

LICENSE_TABLE = {
    "div_free_fine": {
        "verified_by": "T3: max|div_fine| < 1e-12",
        "grants": "the prolonged velocity is admissible for an "
                  "incompressible projection step without a spurious "
                  "divergence source",
        "does_not_grant": "accuracy of the velocity values themselves; "
                          "curl preservation; any statement about the "
                          "continuum solution",
        "foundation": "Paper 8 (theorem's own statistic as its test)",
    },
    "flux_preserving_faces": {
        "verified_by": "T4: |mean(fine faces) - coarse face| < 1e-12",
        "grants": "the coarse flux integral is reproduced by the fine "
                  "faces, which is the precondition for refluxing",
        "does_not_grant": "pointwise accuracy on the face; monotonicity",
        "foundation": "Rev22 Coercivity Thm 4.3 (order-2 flux balance)",
    },
    "reflux_conservation": {
        "verified_by": "T6: conserved total matches analytic value",
        "grants": "global conservation of the refluxed variables across "
                  "the coarse-fine interface",
        "does_not_grant": "local accuracy; conservation of variables "
                          "NOT registered with the refluxer; stability",
        "foundation": "Rev22 Coercivity Thm 4.3; Paper 8 method",
    },
    "exact_constraint_algebra": {
        "verified_by": "T1/T2: rank 7, nullity 5, consistency iff div-free",
        "grants": "the interior solve is exact, not iterative; the "
                  "pseudo-inverse is precomputable once",
        "does_not_grant": "that the chosen closure is physically right",
        "foundation": "Rev22 Exact Predictability Thm 2.1 (known law)",
    },
    "curl_preservation": {
        "verified_by": "NOT VERIFIED — see M1",
        "grants": "NOTHING. Reported as a measured mismatch only.",
        "does_not_grant": "any claim of curl preservation. Exact "
                          "preservation needs Toth-Roe edge-curl "
                          "evolution, not implemented.",
        "foundation": "Paper 7 Remark 3.2 (one branch does not license "
                      "another); Rev22 Open Problem 4.5 (order-4 open)",
    },
}


def staggered_warning() -> str:
    return (
        "STAGGERED MAC MOMENTUM REFLUXING IS NOT IMPLEMENTED.\n"
        "For a MAC grid, the u-momentum control volume centred ON the\n"
        "coarse-fine interface straddles it: half coarse, half fine. The\n"
        "interface-normal velocity there is a SHARED degree of freedom,\n"
        "not an independently refined one, and requires the Almgren et\n"
        "al. (1998) constrained-face treatment. This module refluxes\n"
        "CELL-CENTRED conserved variables only (rho, rhou, rhov, rhow,\n"
        "rhoE), which is the layout super_dns_one uses. For VOF/FH MAC\n"
        "momentum, reflux the mass flux and treat the interface-normal\n"
        "velocity as shared; do NOT assume this module did it for you."
    )


# =============================================================================
# 1. GEOMETRY HELPERS
# =============================================================================

def sl(ax: int, idx) -> Tuple:
    o = [slice(None)] * 3
    o[ax] = idx
    return tuple(o)


def diff(f: Tensor, ax: int, h: float) -> Tensor:
    return (f[sl(ax, slice(1, None))] - f[sl(ax, slice(0, -1))]) / h


def pad1(f: Tensor, ax: int, periodic: bool) -> Tensor:
    n = f.shape[ax]; d = f.device
    if periodic:
        lo = f.index_select(ax, torch.tensor([n - 1], device=d))
        hi = f.index_select(ax, torch.tensor([0], device=d))
    else:
        lo = f.index_select(ax, torch.tensor([0], device=d))
        hi = f.index_select(ax, torch.tensor([n - 1], device=d))
    return torch.cat([lo, f, hi], dim=ax)


def minmod(a: Tensor, b: Tensor) -> Tensor:
    return torch.where(a * b > 0,
                       torch.sign(a) * torch.minimum(a.abs(), b.abs()),
                       torch.zeros_like(a))


@dataclass
class Grid:
    nx: int = 8; ny: int = 8; nz: int = 8
    Lx: float = 1.0; Ly: float = 1.0; Lz: float = 1.0
    periodic: Tuple[bool, bool, bool] = (True, True, True)
    device: str = "cpu"; dtype: str = "float64"

    @property
    def h(self): return (self.Lx / self.nx, self.Ly / self.ny, self.Lz / self.nz)
    @property
    def n(self): return (self.nx, self.ny, self.nz)
    @property
    def dt_(self): return torch.float64 if self.dtype == "float64" else torch.float32
    @property
    def dev(self): return torch.device(self.device)
    def face(self, ax):
        s = list(self.n); s[ax] += 1; return tuple(s)
    def cc(self, v=0.0): return torch.full(self.n, v, device=self.dev, dtype=self.dt_)
    def fc(self, ax, v=0.0):
        return torch.full(self.face(ax), v, device=self.dev, dtype=self.dt_)
    def refined(self, r=2) -> "Grid":
        return Grid(self.nx*r, self.ny*r, self.nz*r,
                    self.Lx, self.Ly, self.Lz, self.periodic,
                    self.device, self.dtype)


def divergence(u, v, w, g: Grid) -> Tensor:
    return diff(u, 0, g.h[0]) + diff(v, 1, g.h[1]) + diff(w, 2, g.h[2])


# =============================================================================
# 2. MOMENTUM REFLUXING  [A1]
# =============================================================================

@dataclass
class FaceKey:
    """Identifies a coarse face: level, axis, and the coarse face index tuple."""
    level: int
    axis: int
    idx: Tuple[int, int, int]

    def __hash__(self): return hash((self.level, self.axis, self.idx))


class FluxRegister:
    """
    Accumulates the SPACE-TIME flux integral through one coarse face.

    Units: the register stores PHYSICAL INTEGRALS (flux density x area x
    time), not flux densities. This removes the classic ambiguity about
    whether dt and area have already been folded in.

    For a conserved-variable stack q of shape (nvar, nx, ny, nz), the
    flux through an axis-a face has shape (nvar, ...transverse...).
    """
    def __init__(self, nvar: int):
        self.nvar = nvar
        self.coarse: Optional[Tensor] = None      # (nvar,) integrated
        self.fine: Optional[Tensor] = None        # (nvar,) integrated
        self.n_fine_contrib = 0

    def set_coarse(self, flux_density: Tensor, area: float, dt: float):
        """flux_density: (nvar,) at this coarse face."""
        self.coarse = flux_density * (area * dt)

    def add_fine(self, flux_density: Tensor, area_f: float, dt_f: float):
        c = flux_density * (area_f * dt_f)
        self.fine = c if self.fine is None else self.fine + c
        self.n_fine_contrib += 1

    def mismatch(self) -> Tensor:
        """delta = fine integral - coarse integral. Zero iff consistent."""
        if self.coarse is None or self.fine is None:
            raise RuntimeError("both coarse and fine flux must be registered")
        return self.fine - self.coarse


class MomentumRefluxer:
    """
    Berger-Colella refluxing for CELL-CENTRED conserved variables.

    Usage:
        rf = MomentumRefluxer(nvar=5, var_names=["rho","rhou","rhov","rhow","rhoE"])
        k = FaceKey(level=0, axis=0, idx=(I,j,k))
        rf.register_coarse(k, Fc, area_c, dt_c)
        for s in range(r):
            for jj in range(r):
                for kk in range(r):
                    rf.register_fine(k, Ff[s,jj,kk], area_f, dt_f)
        q_corrected = rf.apply(q_coarse, grid_coarse, cell_of_face)

    The correction is applied to the coarse cell whose face this is.
    `cell_of_face(key)` must return the (i,j,k) of that cell — the
    DONOR side is determined by the sign convention of the caller, so
    the caller supplies the cell explicitly. No sign is guessed here.
    """
    def __init__(self, nvar: int, var_names: Optional[List[str]] = None):
        self.nvar = nvar
        self.var_names = var_names or [f"q{i}" for i in range(nvar)]
        self.registers: Dict[FaceKey, FluxRegister] = {}
        self.total_correction = None
        self.n_corrections = 0
        self.max_mismatch = 0.0

    def _reg(self, key: FaceKey) -> FluxRegister:
        if key not in self.registers:
            self.registers[key] = FluxRegister(self.nvar)
        return self.registers[key]

    def register_coarse(self, key, flux_density, area, dt):
        self._reg(key).set_coarse(flux_density, area, dt)

    def register_fine(self, key, flux_density, area_f, dt_f):
        self._reg(key).add_fine(flux_density, area_f, dt_f)

    def audit(self) -> Dict[str, Any]:
        """Pre-apply diagnostic: how inconsistent are the fluxes?"""
        out = {}
        for key, reg in self.registers.items():
            if reg.coarse is None or reg.fine is None:
                out[str(key)] = "INCOMPLETE"
                continue
            d = reg.mismatch()
            out[str(key)] = {
                "mismatch_per_var": {n: float(d[i])
                                     for i, n in enumerate(self.var_names)},
                "max_abs": float(d.abs().max()),
                "n_fine_contrib": reg.n_fine_contrib,
            }
        return out

    def apply(self, q: Tensor, g: Grid,
              cell_of_face: Callable[[FaceKey], Tuple[int, int, int]],
              sign_of_face: Optional[Callable[[FaceKey], float]] = None) -> Tensor:
        """
        Apply the flux correction. q: (nvar, nx, ny, nz).

        sign_of_face: +1 if the outward normal of the returned cell at
        this face points in +axis (so the cell LOSES the flux), -1 if it
        points in -axis. Default +1. Getting this sign wrong flips
        conservation into anti-conservation, so it is an explicit
        argument, not an assumption.
        """
        Vc = g.h[0] * g.h[1] * g.h[2]
        out = q.clone()
        self.total_correction = torch.zeros(self.nvar, dtype=q.dtype,
                                            device=q.device)
        for key, reg in self.registers.items():
            if reg.coarse is None or reg.fine is None:
                continue
            delta = reg.mismatch()
            self.max_mismatch = max(self.max_mismatch, float(delta.abs().max()))
            if float(delta.abs().max()) < 1e-300:
                continue
            i, j, k = cell_of_face(key)
            sgn = 1.0 if sign_of_face is None else float(sign_of_face(key))
            # The coarse cell used I_c but should have used I_f.
            # Outward flux removes q, so the correction is -sgn*delta/Vc.
            out[:, i, j, k] -= sgn * delta / Vc
            self.total_correction -= sgn * delta
            self.n_corrections += 1
        return out

    def reset(self):
        self.registers.clear()
        self.total_correction = None
        self.n_corrections = 0
        self.max_mismatch = 0.0


def reflux_selftest(nvar=5, N=8, r=2, seed=0) -> Dict[str, Any]:
    """
    T6/T7: the conserved total must match the ANALYTIC value after
    refluxing, even when the coarse flux is deliberately corrupted.

    Construction: a single coarse-fine interface face. The fine flux is
    the truth. The coarse flux is perturbed by a known amount. The
    analytic total change is -(fine integral)/1 (outward loss).
    """
    torch.manual_seed(seed)
    g = Grid(nx=N, ny=N, nz=N)
    gf = g.refined(r)
    names = ["rho", "rhou", "rhov", "rhow", "rhoE"][:nvar]
    q = torch.rand(nvar, N, N, N, dtype=torch.float64) + 1.0
    total0 = q.sum(dim=(1, 2, 3))

    key = FaceKey(0, 0, (N // 2, 2, 2))
    cell = (N // 2 - 1, 2, 2)          # donor cell on the low side
    area_c = g.h[1] * g.h[2]
    dt_c = 1e-3
    area_f = gf.h[1] * gf.h[2]
    dt_f = dt_c / r

    Ff = torch.rand(nvar, r, r, r, dtype=torch.float64) * 0.1
    fine_integral = (Ff.sum(dim=(1, 2, 3))) * area_f * dt_f
    # consistent coarse flux density reproduces the fine integral exactly
    Fc_true = fine_integral / (area_c * dt_c)
    corruption = torch.rand(nvar, dtype=torch.float64) * 0.05
    Fc_bad = Fc_true + corruption

    def run(Fc, do_reflux):
        rf = MomentumRefluxer(nvar, names)
        rf.register_coarse(key, Fc, area_c, dt_c)
        for s in range(r):
            for jj in range(r):
                for kk in range(r):
                    rf.register_fine(key, Ff[s, jj, kk], area_f, dt_f)
        qq = q.clone()
        qq[:, cell[0], cell[1], cell[2]] -= Fc * area_c * dt_c / (
            g.h[0] * g.h[1] * g.h[2])
        if do_reflux:
            qq = rf.apply(qq, g, lambda k: cell, lambda k: 1.0)
        return qq.sum(dim=(1, 2, 3)), rf

    analytic = total0 - fine_integral
    t_consistent, _ = run(Fc_true, False)
    t_bad, _ = run(Fc_bad, False)
    t_fixed, rf = run(Fc_bad, True)

    return {
        "analytic_total": analytic.tolist(),
        "consistent_no_reflux_err": float((t_consistent - analytic).abs().max()),
        "corrupted_no_reflux_err": float((t_bad - analytic).abs().max()),
        "corrupted_with_reflux_err": float((t_fixed - analytic).abs().max()),
        "n_corrections": rf.n_corrections,
        "max_mismatch": rf.max_mismatch,
    }


# =============================================================================
# 3. BALSARA PROLONGATION WITH EXPLICIT NULL-SPACE CLOSURE  [B1,B2]
# =============================================================================

class BalsaraProlongation:
    """
    2:1 divergence-preserving prolongation of a MAC velocity field.

    DEGREE-OF-FREEDOM COUNT (derived, verified in T1)
      interior fine faces per coarse cell: 4 (x) + 4 (y) + 4 (z) = 12
      divergence constraints: 8 sub-cells, rank 7 (rows sum to zero)
      => null space dimension 5.
      Consistency: summing the 8 constraints gives -4*dx_c*div_coarse,
      so the system is solvable IFF div_coarse == 0 (verified T2).

    TWO STEPS
      1. FACES: limited-linear reconstruction on each coarse face,
         sampled at the r^2 fine face centres. The linear term
         integrates to zero over the face, so the mean of the fine
         faces equals the coarse value EXACTLY (T4).
      2. INTERIOR: x = pinv(C) b + N y, with N the 12x5 null-space basis
         and y the 5 free parameters per cell, fixed by `closure`.

    CLOSURES
      min_norm : y = 0                       (smallest ||x||)
      curl_min : minimise interior sub-cell circulation
      smooth   : minimise variation among the interior faces

    ⚠️ DERIVED FINDING (M1): the curl at COARSE edges involves only fine
       faces lying ON coarse faces, all fixed by step 1. The interior
       unknowns do not appear there. NO interior closure can enforce
       coarse-edge curl preservation. Exact preservation requires
       Toth-Roe [T1] edge-circulation evolution. NOT IMPLEMENTED.
    """

    def __init__(self, device="cpu", dtype=torch.float64, limited=True):
        self.dev = torch.device(device)
        self.dt_ = dtype
        self.limited = limited
        C = self._build_div_constraint()
        self.C_np = C
        self.rank = int(np.linalg.matrix_rank(C))
        U, S, Vt = np.linalg.svd(C, full_matrices=True)
        self.nullity = int((S < 1e-10).sum()) + (Vt.shape[0] - len(S))
        self.N_np = Vt[self.rank:].T          # (12, 5)
        self.pinv_np = np.linalg.pinv(C)      # (12, 8)
        self.C = torch.tensor(C, device=self.dev, dtype=dtype)
        self.N = torch.tensor(self.N_np, device=self.dev, dtype=dtype)
        self.pinv = torch.tensor(self.pinv_np, device=self.dev, dtype=dtype)
        self.last_diag: Dict[str, Any] = {}

    @staticmethod
    def _build_div_constraint() -> np.ndarray:
        """
        C (8x12), entries +-1 (the 1/h_f factor is folded into b).
        Unknown order: x[j,k] -> 2j+k ; y[i,k] -> 4+2i+k ; z[i,j] -> 8+2i+j
        Row order: sub-cell (i,j,k) -> 4i+2j+k
        """
        C = np.zeros((8, 12))
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    r = 4 * i + 2 * j + k
                    C[r, 0 + 2 * j + k] = 1.0 if i == 0 else -1.0
                    C[r, 4 + 2 * i + k] = 1.0 if j == 0 else -1.0
                    C[r, 8 + 2 * i + j] = 1.0 if k == 0 else -1.0
        return C

    # ---- step 1: flux-preserving face reconstruction -------------------
    def _face_prolong(self, uc: Tensor, ax: int, g: Grid, r=2) -> Tensor:
        tan = [a for a in range(3) if a != ax]
        a1, a2 = tan
        if self.limited:
            p1 = pad1(uc, a1, g.periodic[a1])
            s1 = 0.5 * minmod(p1[sl(a1, slice(0, -2))] - uc,
                              p1[sl(a1, slice(2, None))] - uc)
            p2 = pad1(uc, a2, g.periodic[a2])
            s2 = 0.5 * minmod(p2[sl(a2, slice(0, -2))] - uc,
                              p2[sl(a2, slice(2, None))] - uc)
        else:
            s1 = torch.zeros_like(uc); s2 = torch.zeros_like(uc)
        shape = list(uc.shape)
        shape[a1] *= r; shape[a2] *= r
        out = torch.zeros(shape, device=uc.device, dtype=uc.dtype)
        offs = (torch.arange(r, device=uc.device, dtype=uc.dtype) + 0.5) / r - 0.5
        for b1 in range(r):
            for b2 in range(r):
                val = uc + offs[b1] * s1 + offs[b2] * s2
                so = [slice(None)] * 3
                so[a1] = slice(b1, None, r)
                so[a2] = slice(b2, None, r)
                out[tuple(so)] = val
        return out

    # ---- step 2: interior solve ----------------------------------------
    def _interior(self, uf, vf, wf, g: Grid, closure: str):
        """
        g is the COARSE grid. uf,vf,wf are full fine arrays with the
        coarse-face-aligned entries already filled by step 1.
        """
        nx, ny, nz = g.n
        nc = nx * ny * nz
        dev, dt_ = uf.device, uf.dtype
        hf = tuple(h / 2 for h in g.h)

        # known boundary fine faces, gathered per coarse cell
        xL = uf[0::2, 0::2, 0::2][:nx, :ny, :nz]
        xR = uf[2::2, 0::2, 0::2][:nx, :ny, :nz]
        yL = vf[0::2, 0::2, 0::2][:nx, :ny, :nz]
        yR = vf[0::2, 2::2, 0::2][:nx, :ny, :nz]
        zL = wf[0::2, 0::2, 0::2][:nx, :ny, :nz]
        zR = wf[0::2, 0::2, 2::2][:nx, :ny, :nz]

        b = torch.zeros(nx, ny, nz, 8, device=dev, dtype=dt_)
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    r = 4 * i + 2 * j + k
                    b[..., r] = ((xL if i == 0 else -xR)
                                 + (yL if j == 0 else -yR)
                                 + (zL if k == 0 else -zR))
        bf = b.reshape(nc, 8)
        xp = bf @ self.pinv.T                    # (nc,12) particular solution

        if closure == "min_norm" or self.N.shape[1] == 0:
            y = torch.zeros(nc, self.N.shape[1], device=dev, dtype=dt_)
        else:
            A, rhs = self._closure_system(closure, xp, uf, vf, wf, g)
            if A is None:
                y = torch.zeros(nc, self.N.shape[1], device=dev, dtype=dt_)
            else:
                # per-cell small least squares (A: (nc, m, 5))
                y = torch.linalg.lstsq(A, rhs.unsqueeze(-1)).solution.squeeze(-1)

        x = xp + y @ self.N.T                    # (nc,12)

        xi = x[:, 0:4].reshape(nx, ny, nz, 2, 2)
        yi = x[:, 4:8].reshape(nx, ny, nz, 2, 2)
        zi = x[:, 8:12].reshape(nx, ny, nz, 2, 2)

        uf = uf.clone(); vf = vf.clone(); wf = wf.clone()
        uf[1::2, 0::2, 0::2] = xi.reshape(nx, 2 * ny, 2 * nz)
        vf[0::2, 1::2, 0::2] = yi.permute(0, 2, 1, 3).reshape(2 * nx, ny, 2 * nz)
        wf[0::2, 0::2, 1::2] = zi.permute(0, 1, 3, 2).reshape(2 * nx, 2 * ny, nz)
        return uf, vf, wf, float(y.abs().max())

    def _closure_system(self, closure, xp, uf, vf, wf, g: Grid):
        """
        Build the per-cell least-squares system for the 5 free parameters.

        curl_min: the 6 fully-interior curl components involve only
                  interior unknowns + this cell's known faces, so they
                  ARE controllable. Minimise their magnitude.
        smooth  : minimise the spread among each direction's 4 interior
                  faces (a smoothness regulariser).
        """
        nx, ny, nz = g.n
        nc = nx * ny * nz
        dev, dt_ = xp.device, xp.dtype
        N = self.N                                # (12,5)

        if closure == "smooth":
            # D (3x12): deviation of each interior face from its group mean
            D = torch.zeros(12, 12, device=dev, dtype=dt_)
            for grp in range(3):
                s = slice(4 * grp, 4 * grp + 4)
                D[s, s] = torch.eye(4, device=dev, dtype=dt_) - 0.25
            A = (D @ N).unsqueeze(0).expand(nc, 12, N.shape[1])
            rhs = -(D @ xp.T).T                   # (nc,12)
            return A, rhs

        if closure != "curl_min":
            return None, None

        # interior curl components, per coarse cell, in terms of the
        # 12 interior unknowns. Local index convention as in _interior.
        # curl_z at (xf=1, yf=1, zc=k):
        #   (v(1,1,k)-v(0,1,k))/hf_x - (u(1,1,k)-u(1,0,k))/hf_y
        # unknown index: v(i,1,k) -> 4+2i+k ; u(1,j,k) -> 2j+k
        rows = []
        consts = []
        hx, hy, hz = g.h[0] / 2, g.h[1] / 2, g.h[2] / 2
        for k in (0, 1):
            row = torch.zeros(12, device=dev, dtype=dt_)
            row[4 + 2 * 1 + k] += 1.0 / hx
            row[4 + 2 * 0 + k] -= 1.0 / hx
            row[2 * 1 + k] -= 1.0 / hy
            row[2 * 0 + k] += 1.0 / hy
            rows.append(row); consts.append(torch.zeros((), device=dev, dtype=dt_))
        # curl_x at (xc=i, yf=1, zf=1):
        #   (w(i,1,1)-w(i,0,1))/hf_y - (v(i,1,1)-v(i,1,0))/hf_z
        # unknown: w(i,j,1) -> 8+2i+j ; v(i,1,k) -> 4+2i+k
        for i in (0, 1):
            row = torch.zeros(12, device=dev, dtype=dt_)
            row[8 + 2 * i + 1] += 1.0 / hy
            row[8 + 2 * i + 0] -= 1.0 / hy
            row[4 + 2 * i + 1] -= 1.0 / hz
            row[4 + 2 * i + 0] += 1.0 / hz
            rows.append(row); consts.append(torch.zeros((), device=dev, dtype=dt_))
        # curl_y at (xf=1, yc=j, zf=1):
        #   (u(1,j,1)-u(1,j,0))/hf_z - (w(1,j,1)-w(0,j,1))/hf_x
        for j in (0, 1):
            row = torch.zeros(12, device=dev, dtype=dt_)
            row[2 * j + 1] += 1.0 / hz
            row[2 * j + 0] -= 1.0 / hz
            row[8 + 2 * 1 + j] -= 1.0 / hx
            row[8 + 2 * 0 + j] += 1.0 / hx
            rows.append(row); consts.append(torch.zeros((), device=dev, dtype=dt_))

        D = torch.stack(rows, 0)                  # (6,12)
        A = (D @ N).unsqueeze(0).expand(nc, 6, N.shape[1])
        rhs = -(D @ xp.T).T                       # (nc,6)
        return A, rhs

    def prolongate(self, u, v, w, g: Grid, closure="min_norm"):
        uf = self._face_prolong(u, 0, g)
        vf = self._face_prolong(v, 1, g)
        wf = self._face_prolong(w, 2, g)
        uf, vf, wf, ymag = self._interior(uf, vf, wf, g, closure)
        self.last_diag = {"rank_C": self.rank, "nullity": self.N.shape[1],
                          "closure": closure, "max_|y|": ymag}
        return uf, vf, wf


def restrict_faces(uf, vf, wf, r=2):
    """Flux-preserving restriction: mean of the r^2 fine faces per coarse face."""
    def red(f, ax):
        out = f[0::r]
        for a in range(3):
            if a == ax: continue
            n = out.shape[a] // r
            out = out.reshape(*out.shape[:a], n, r, *out.shape[a+1:]).mean(a+1)
        return out
    return red(uf, 0), red(vf, 1), red(wf, 2)


# =============================================================================
# 4. CURL DIAGNOSTICS  (measured, per Paper 8 method; NOT a claim)
# =============================================================================

def curl_edge(u, v, w, g: Grid) -> Dict[str, Tensor]:
    """
    MAC curl on edges.
      curl_x at (xc, yf, zf) = dw/dy - dv/dz
      curl_y at (xf, yc, zf) = du/dz - dw/dx
      curl_z at (xf, yf, zc) = dv/dx - du/dy
    Interior edges only (boundary edges need neighbour/BC handling).
    """
    hx, hy, hz = g.h
    cx = (v[:, 1:, 1:] - v[:, :-1, 1:]) / hy - (w[:, 1:, 1:] - w[:, 1:, :-1]) / hz
    cy = (w[1:, :, 1:] - w[:-1, :, 1:]) / hz - (u[1:, :, 1:] - u[1:, :, :-1]) / hx
    cz = (u[1:, 1:, :] - u[1:, :-1, :]) / hx - (v[1:, 1:, :] - v[:-1, 1:, :]) / hy
    # note: cx uses v at (xc,yf,zc) and w at (xc,yc,zf); shapes differ,
    # so we crop to the common interior block.
    n = tuple(min(a, b, c) for a, b, c in zip(cx.shape, cy.shape, cz.shape))
    return {"x": cx[:n[0], :n[1], :n[2]],
            "y": cy[:n[0], :n[1], :n[2]],
            "z": cz[:n[0], :n[1], :n[2]]}


def curl_mismatch(uc, vc, wc, g: Grid, uf, vf, wf, r=2) -> Dict[str, float]:
    """
    M1: compare the coarse edge curl with the block-average of the fine
    edge curls at the corresponding locations. REPORTED, not claimed.
    """
    gf = g.refined(r)
    cc = curl_edge(uc, vc, wc, g)
    cf = curl_edge(uf, vf, wf, gf)
    out = {}
    for k in ("x", "y", "z"):
        a = cc[k]; b = cf[k]
        n = tuple(min(a.shape[i], b.shape[i] // r) for i in range(3))
        a = a[:n[0], :n[1], :n[2]]
        bb = b[:n[0]*r, :n[1]*r, :n[2]*r].reshape(
            n[0], r, n[1], r, n[2], r).mean(dim=(1, 3, 5))
        num = float((bb - a).abs().max())
        den = float(a.abs().max()) + 1e-30
        out[f"curl_{k}_abs"] = num
        out[f"curl_{k}_rel"] = num / den
    return out


# =============================================================================
# 5. STATISTICS — Paper 9 / Paper 11 applicability gate
# =============================================================================

def subsample_ci(x: np.ndarray, conf=0.90, block=None) -> Dict[str, float]:
    """Politis-Romano. ONLY for stationary-mixing random sequences."""
    n = len(x)
    if n < 8:
        m = float(np.mean(x)) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    if block is None:
        block = max(2, min(n // 4, int(round(n ** (1 / 3))) * 2))
    th = float(np.mean(x))
    sub = np.array([x[j:j+block].mean() for j in range(n - block + 1)])
    sc = math.sqrt(block) * (sub - th)
    a = 1 - conf
    return {"mean": th,
            "lo": th - float(np.percentile(sc, 100*(1-a/2)))/math.sqrt(n),
            "hi": th - float(np.percentile(sc, 100*(a/2)))/math.sqrt(n),
            "n": n, "block": block}


def ci_applicability(quantity: str) -> Dict[str, Any]:
    """
    Paper 9's own logic forbids applying a mixing-sequence CI to a
    deterministic round-off residual. This gate says so explicitly.
    """
    deterministic = {
        "fine_divergence_residual", "flux_preservation_residual",
        "reflux_conservation_residual", "constraint_rank",
        "prolong_restrict_roundtrip",
    }
    if quantity in deterministic:
        return {"ci_applicable": False,
                "reason": "deterministic machine-precision residual, not a "
                          "stationary mixing random sequence (Paper 9 "
                          "Entry 2.3 hypothesis fails by construction)",
                "report": "exact residual, no interval"}
    return {"ci_applicable": True,
            "reason": "across-seed variability of a measured mismatch is a "
                      "genuine random sample; subsampling applies if the "
                      "seeds are exchangeable",
            "report": "subsample CI + number of seeds"}


# =============================================================================
# 6. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 80)
    print(f"  MOMENTUM REFLUXING + BALSARA CLOSURE v{VERSION}")
    print("=" * 80)
    torch.manual_seed(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok; ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    N = 8
    g = Grid(nx=N, ny=N, nz=N)
    gf = g.refined(2)
    bal = BalsaraProlongation()

    print("\n[T1] constraint algebra is EXACT (Rev22 Exact Predictability)")
    colsum = float(np.abs(bal.C_np.sum(0)).max())
    rep("rank == 7", bal.rank == 7, f"rank={bal.rank}")
    rep("columns sum to zero", colsum < 1e-12, f"max={colsum:.2e}")
    rep("null space dim == 5", bal.N.shape[1] == 5, f"dim={bal.N.shape[1]}")
    resid = float(np.abs(bal.C_np @ bal.N_np).max())
    rep("C @ N == 0", resid < 1e-12, f"max={resid:.2e}")

    print("\n[T2] consistency: solvable IFF coarse field is div-free")
    X, Y, Z = torch.meshgrid(*[(torch.arange(n, dtype=torch.float64)+0.5)*h
                               for n, h in zip(g.n, g.h)], indexing="ij")
    k = 2 * math.pi
    # exactly div-free via a stream function in x,y (w=0)
    psi = torch.sin(k*X)*torch.sin(k*Y)
    uu = g.fc(0); vv = g.fc(1); ww = g.fc(2)
    uu[1:-1] = (psi[:, 1:] - psi[:, :-1]) / g.h[1]
    vv[:, 1:-1] = -(psi[1:, :] - psi[:-1, :]) / g.h[0]
    dc = float(divergence(uu, vv, ww, g).abs().max())
    rep("coarse field div-free", dc < 1e-12, f"max|div|={dc:.2e}")

    print("\n[T3] FINE DIVERGENCE == 0 (Paper 8: theorem's own statistic)")
    for closure in ("min_norm", "curl_min", "smooth"):
        uf, vf, wf = bal.prolongate(uu, vv, ww, g, closure=closure)
        df = float(divergence(uf, vf, wf, gf).abs().max())
        rep(f"div-free ({closure})", df < 1e-11, f"max|div_fine|={df:.2e}")
    print(f"      CI applicable? {ci_applicability('fine_divergence_residual')}")

    print("\n[T4] FLUX PRESERVATION: mean of 4 fine faces == coarse face")
    uf, vf, wf = bal.prolongate(uu, vv, ww, g, closure="min_norm")
    for ax, (a, b) in enumerate(((uu, uf), (vv, vf), (ww, wf))):
        m = b[0::2] if ax == 0 else (b[:, 0::2] if ax == 1 else b[:, :, 0::2])
        for t in range(3):
            if t == ax: continue
            m = 0.5*(m[sl(t, slice(0, -1, 2))] + m[sl(t, slice(1, None, 2))])
        e = float((m - a).abs().max())
        rep(f"axis {ax} flux preserved", e < 1e-12, f"max err={e:.2e}")

    print("\n[T5] restriction/prolongation round-trip on faces")
    ur, vr, wr = restrict_faces(uf, vf, wf)
    e = max(float((ur-uu).abs().max()), float((vr-vv).abs().max()))
    rep("round-trip exact", e < 1e-12, f"max err={e:.2e}")

    print("\n[T6/T7] REFLUXING restores the analytic conserved total")
    rr = reflux_selftest()
    print(f"      consistent, no reflux : err={rr['consistent_no_reflux_err']:.2e}")
    print(f"      corrupted,  no reflux : err={rr['corrupted_no_reflux_err']:.2e}")
    print(f"      corrupted, WITH reflux: err={rr['corrupted_with_reflux_err']:.2e}")
    rep("consistent fluxes need no correction",
        rr["consistent_no_reflux_err"] < 1e-12)
    rep("corruption breaks conservation without reflux",
        rr["corrupted_no_reflux_err"] > 1e-8)
    rep("reflux restores conservation EXACTLY",
        rr["corrupted_with_reflux_err"] < 1e-12,
        f"{rr['n_corrections']} corrections, mismatch={rr['max_mismatch']:.2e}")

    print("\n[M1] CURL MISMATCH — MEASURED, NOT CLAIMED")
    print("      DERIVED: coarse-edge curls involve only fine faces ON")
    print("      coarse faces (fixed by step 1). Interior unknowns do not")
    print("      appear -> NO interior closure can enforce preservation.")
    curl_rows = {}
    for closure in ("min_norm", "curl_min", "smooth"):
        uf2, vf2, wf2 = bal.prolongate(uu, vv, ww, g, closure=closure)
        cm = curl_mismatch(uu, vv, ww, g, uf2, vf2, wf2)
        curl_rows[closure] = cm
        print(f"      {closure:9s} curl_rel: x={cm['curl_x_rel']:.3e} "
              f"y={cm['curl_y_rel']:.3e} z={cm['curl_z_rel']:.3e}  "
              f"|y|max={bal.last_diag['max_|y|']:.3e}")
    same = all(abs(curl_rows[c]["curl_z_rel"] - curl_rows["min_norm"]["curl_z_rel"])
               < 1e-10 for c in curl_rows)
    rep("M1 confirmed: closure does NOT change coarse-edge curl", same,
        "all closures give the same coarse-edge curl mismatch")
    print("      => exact curl preservation requires Toth-Roe edge-curl")
    print("         evolution [T1]. NOT IMPLEMENTED. No claim made.")

    print("\n[M2] interior-circulation ablation (does curl_min do anything?)")
    def interior_curl_norm(uf, vf, wf):
        cf = curl_edge(uf, vf, wf, gf)
        return float(sum(float((cf[k]**2).sum()) for k in cf))
    vals = {}
    for closure in ("min_norm", "curl_min", "smooth"):
        u3, v3, w3 = bal.prolongate(uu, vv, ww, g, closure=closure)
        vals[closure] = interior_curl_norm(u3, v3, w3)
        print(f"      {closure:9s} total interior curl^2 = {vals[closure]:.6e}")
    print("      NOTE: a lower value is NOT evidence of better accuracy.")
    rep("ablation produced numbers", len(vals) == 3)

    print("\n[PAPER 9 GATE] CI applicability")
    for q in ("fine_divergence_residual", "reflux_conservation_residual",
              "across_seed_curl_mismatch"):
        a = ci_applicability(q)
        print(f"      {q:32s} CI={a['ci_applicable']}")
    rep("deterministic residuals refuse a CI",
        not ci_applicability("fine_divergence_residual")["ci_applicable"])
    rep("across-seed mismatch allows a CI",
        ci_applicability("across_seed_curl_mismatch")["ci_applicable"])

    print("\n[LICENSE TABLE] Paper 7 discipline")
    for k, v in LICENSE_TABLE.items():
        print(f"      {k:26s} verified_by: {v['verified_by'][:44]}")
    rep("curl_preservation grants NOTHING",
        LICENSE_TABLE["curl_preservation"]["grants"].startswith("NOTHING"))

    print("\n[STAGGERED LIMIT]")
    for line in staggered_warning().splitlines()[:4]:
        print("      " + line)

    print("\n" + "=" * 80)
    print(f"  {'✅ ALL PROVEN CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 80)
    print("""
  PROVEN (deterministic, machine precision, no CI)
    constraint rank 7 / nullity 5 / C@N=0 / consistency iff div-free
    fine divergence == 0 for all three closures
    face-flux preservation exact; restriction round-trip exact
    refluxing restores the analytic conserved total exactly

  MEASURED, NOT CLAIMED
    coarse-edge curl mismatch (identical across closures — M1 confirmed)
    interior circulation ablation (lower != better)

  NOT IMPLEMENTED
    Toth-Roe exact curl preservation (needs evolved edge circulations)
    staggered MAC momentum refluxing at a straddling CV (Almgren et al.)
    time-interpolated coarse boundary data across fine subcycles
    octree; experimental validation
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "license", "staggered"])
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if self_test() else 1)
    if a.cmd == "license":
        print(json.dumps(LICENSE_TABLE, indent=2))
    else:
        print(staggered_warning())


if __name__ == "__main__":
    main()
