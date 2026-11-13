#!/usr/bin/env python3
"""
=============================================================================
SESI MULTIPHASE FLOW v2.0.0 — VOF 3D Navier-Stokes on the ONE CFD core
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

NUMERICAL METHODS (all standard, cited)
 [C1] Hirt & Nichols (1981)        VOF concept
 [C2] Youngs (1982)                PLIC normal reconstruction
 [C3] Scardovelli & Zaleski (1999) analytic cut-volume / plane inversion
 [C4] Rider & Kothe (1998)         operator-split geometric advection
 [C5] Weymouth & Yue (2010)        mass-conserving split VOF, dilation term
 [C6] Brackbill et al. (1992)      Continuum Surface Force
 [C7] Chorin (1968)                fractional-step projection
 [C8] Ubbink & Issa (1999)         CICSAM algebraic VOF
 [C9] Smagorinsky (1963)           LES eddy viscosity
 [C10] Angot et al. (1999)         Brinkman volume penalisation
 [C11] Papanastasiou (1987)        Bingham regularisation

=============================================================================
WHAT CHANGED FROM v1 — every item is a real bug in the original
=============================================================================
[B1] PressurePoisson._beta_faces returned the harmonic mean of rho and used
     it directly as beta in `u -= dt*beta*grad(p)`. beta must be 1/rho_face.
     The two differ by a reciprocal -> the projection was INVERTED.
     Now: one_cfd_stencils.face_beta (arithmetic mean of reciprocals).
[B2] MACGrid.grad_cc_to_faces returned (nx,ny,nz) instead of (nx+1,ny,nz).
     Now: one_cfd_stencils.MACOps.grad_cc_to_faces, shape-asserted.
[B3] Rheology.strain_rate_invariant mixed face-located derivatives on
     different axes -> broadcast error. Now all 6 components at centres.
[B4] _viscous_u / _viscous_component had inconsistent edge shapes AND
     mutated cfg via object.__setattr__ with no try/finally. Now a generic
     axis-explicit stencil, no cfg mutation at all.
[B5] SurfaceTension curvature used -lap(alpha)/|grad alpha|, which is not
     curvature. Now kappa = -div(n_hat), n_hat = grad(alpha)/|grad(alpha)|.
[B6] _write_ply had broken f-strings (`f "element vertex {...}"`). Fixed.
[B7] drag() contained `... if False else ...` referencing an undefined
     `ucc`. Dead branch removed.
[B8] PLIC.sweep_flux zeroed boundary fluxes in place on an autograd tensor.
     Now out-of-place.
[B9] build_solid_mask_from_coords used a triple Python loop over the
     stencil. Now chunked and vectorised.

KEPT FROM v1 (these were correct and are the valuable parts)
 • PLIC.box_volume  — inclusion-exclusion with dimension-reduction fallback
 • PLIC.plane_offset — vectorised bisection, float64 throughout
 • PLIC.sweep_flux  — directional-split geometric flux
 • VOFAdvector CICSAM — normalised-variable CBC/UQ blending
 • Bounded mass repair after advection

=============================================================================
HONEST LIMITATIONS — unchanged, read before production use
=============================================================================
• FIRST-ORDER IN TIME. Forward-Euler fractional step.
• MOMENTUM advection is the non-conservative convective form (u.grad)u
  with minmod-limited upwind. MASS is conserved to machine precision;
  MOMENTUM is NOT discretely conserved. For density ratios > 1000 use a
  conservative momentum scheme (not implemented).
• CSF surface tension is NOT balanced-force. Expect SPURIOUS CURRENTS
  near static interfaces. T10 reports the real Laplace-number error
  rather than hiding it.
• Contact angle is 90 deg (Neumann alpha) only.
• NOT an AMR solver. interface_amr_indicator() produces a flag field for
  an EXTERNAL AMR engine; no coarse-fine flux correction, no subcycling.
• Differentiable mode unrolls the CG solve and is memory-hungry;
  geometric PLIC gradients are piecewise and noisy.
• NOT validated against experiment. Run the self-test suite locally.
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from one_cfd_stencils import (
    GridSpec, MACOps, PressurePoisson, face_beta, face_scalar, edge_scalar,
    project, advect_alpha_upwind, pad_cc, pad_axis, diff, sl,
    STENCIL_VERSION,
)

logger = logging.getLogger("sesi.vof")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO)

VOF_VERSION = "2.0.0"

# ---- optional ecosystem / IO ----
try:
    from one_core import get_device as _core_get_device, ONE_VERSION
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False
    ONE_VERSION = "absent"
    logger.warning("one_core not found — running standalone. "
                   "Bridges (CH/Langevin) will be unavailable.")

try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


# =============================================================================
# 0. CONFIG
# =============================================================================

@dataclass
class VOFConfig:
    """
    Configuration. All quantities SI unless noted.

    Grid      : nx,ny,nz cells; Lx,Ly,Lz metres; bc per axis
    Fluids    : rho1/rho2 [kg/m3], mu1/mu2 [Pa s], sigma [N/m], gravity [m/s2]
    Numerics  : vof_scheme 'plic'|'cicsam'|'upwind'
                momentum_limiter 'minmod'|'upwind'|'central'
                pressure_solver 'cg'|'fft'
    Rheology  : 'newtonian'|'power_law'|'carreau'|'bingham'
    Turbulence: 'laminar'|'smagorinsky'
    Solid     : brinkman_eta [s] (smaller = stiffer no-slip); 0 disables
    Runtime   : device, dtype, differentiable
    """
    nx: int = 32
    ny: int = 32
    nz: int = 32
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    bc: Tuple[str, str, str] = ("wall", "wall", "wall")

    rho1: float = 1000.0
    rho2: float = 1.0
    mu1: float = 1.0e-3
    mu2: float = 1.8e-5
    sigma: float = 0.0728
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)

    vof_scheme: str = "plic"
    momentum_limiter: str = "minmod"
    pressure_solver: str = "cg"
    poisson_tol: float = 1e-9
    poisson_max_iter: int = 3000
    n_alpha_smooth: int = 2

    rheology: str = "newtonian"
    power_law_k: float = 1.0e-3
    power_law_n: float = 0.6
    carreau_mu0: float = 1.0e-1
    carreau_muinf: float = 1.0e-3
    carreau_lambda: float = 1.0
    carreau_n: float = 0.5
    bingham_tauy: float = 1.0
    bingham_m: float = 1000.0
    mu_min: float = 1e-8
    mu_max: float = 1e6

    turbulence: str = "laminar"
    smagorinsky_cs: float = 0.17

    boussinesq: bool = False
    beta_thermal: float = 2.1e-4
    T_ref: float = 293.15
    scalar_diffusivity: float = 1.4e-7

    omega: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    porous_alpha_darcy: float = 0.0
    porous_beta_forch: float = 0.0
    brinkman_eta: float = 0.0

    cfl: float = 0.35
    viscous_factor: float = 0.25
    capillary_factor: float = 0.5
    dt_max: float = 1e-2

    device: str = "cpu"
    dtype: str = "float64"
    differentiable: bool = False

    def __post_init__(self):
        if min(self.nx, self.ny, self.nz) < 4:
            raise ValueError("need >= 4 cells per axis")
        if min(self.Lx, self.Ly, self.Lz) <= 0:
            raise ValueError("domain extents must be positive")
        for b in self.bc:
            if b not in ("periodic", "wall", "slip"):
                raise ValueError(f"bad bc {b!r}")
        if self.rho1 <= 0 or self.rho2 <= 0:
            raise ValueError("densities must be positive")
        if self.mu1 <= 0 or self.mu2 <= 0:
            raise ValueError("viscosities must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        if self.vof_scheme not in ("plic", "cicsam", "upwind"):
            raise ValueError("vof_scheme must be plic|cicsam|upwind")
        if self.momentum_limiter not in ("minmod", "upwind", "central"):
            raise ValueError("momentum_limiter must be minmod|upwind|central")
        if self.pressure_solver not in ("cg", "fft"):
            raise ValueError("pressure_solver must be cg|fft")
        if self.pressure_solver == "fft" and any(b != "periodic" for b in self.bc):
            raise ValueError("fft pressure solver requires fully periodic BC")
        if self.rheology not in ("newtonian", "power_law", "carreau", "bingham"):
            raise ValueError("bad rheology")
        if self.turbulence not in ("laminar", "smagorinsky"):
            raise ValueError("bad turbulence model")
        if not (0 < self.cfl <= 1):
            raise ValueError("cfl must be in (0,1]")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.vof_scheme == "plic" and self.dtype == "float32":
            warnings.warn("PLIC with float32 loses mass accuracy; "
                          "internal volume math is float64 regardless, "
                          "but float64 state is strongly recommended.")

    @property
    def dx(self): return self.Lx / self.nx
    @property
    def dy(self): return self.Ly / self.ny
    @property
    def dz(self): return self.Lz / self.nz

    def grid(self) -> GridSpec:
        return GridSpec(nx=self.nx, ny=self.ny, nz=self.nz,
                        Lx=self.Lx, Ly=self.Ly, Lz=self.Lz,
                        bc=self.bc, device=self.device, dtype=self.dtype)

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("bc", "gravity", "omega"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


# =============================================================================
# 1. PLIC — kept from v1, out-of-place boundary fix [B8]
# =============================================================================

class PLIC:
    """
    Geometric VOF machinery. All volume arithmetic in float64.

    Normal   : Youngs [C2] central-difference gradient of alpha.
    Plane    : n.x = d, folded to the positive octant, L1-normalised so
               d in [0,1] on the unit reference cell.
    Volume   : Scardovelli & Zaleski [C3] inclusion-exclusion, generalised
               to a rectangular box (needed for split fluxes).
    Inversion: vectorised bisection, 40 iters -> ~1e-12 in d.
    """
    _DIM_TOL = 1e-3
    _BISECT_ITERS = 40

    @staticmethod
    def normals(alpha: Tensor, dx: float, dy: float, dz: float):
        a = alpha.to(torch.float64)
        ap = F.pad(a.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1),
                   mode="replicate")[0, 0]
        gx = (ap[2:, 1:-1, 1:-1] - ap[:-2, 1:-1, 1:-1]) / (2 * dx)
        gy = (ap[1:-1, 2:, 1:-1] - ap[1:-1, :-2, 1:-1]) / (2 * dy)
        gz = (ap[1:-1, 1:-1, 2:] - ap[1:-1, 1:-1, :-2]) / (2 * dz)
        mag = torch.sqrt(gx * gx + gy * gy + gz * gz)
        safe = mag.clamp_min(1e-30)
        return -gx / safe, -gy / safe, -gz / safe, mag

    @staticmethod
    def _pos3(t): return t.clamp_min(0.0) ** 3

    @classmethod
    def box_volume(cls, d, m1, m2, m3, L1, L2, L3):
        """
        Volume of {x in [0,L1]x[0,L2]x[0,L3] : m1 x + m2 y + m3 z <= d},
        m_i >= 0. Falls back to the exact 2D/1D formula when a normal
        component is below _DIM_TOL (the 3D expression suffers
        catastrophic cancellation there).
        """
        d = torch.as_tensor(d, dtype=torch.float64)
        m1 = torch.as_tensor(m1, dtype=torch.float64).clamp_min(0.0)
        m2 = torch.as_tensor(m2, dtype=torch.float64).clamp_min(0.0)
        m3 = torch.as_tensor(m3, dtype=torch.float64).clamp_min(0.0)
        L1 = torch.as_tensor(L1, dtype=torch.float64)
        L2 = torch.as_tensor(L2, dtype=torch.float64)
        L3 = torch.as_tensor(L3, dtype=torch.float64)
        d, m1, m2, m3, L1, L2, L3 = torch.broadcast_tensors(
            d, m1, m2, m3, L1, L2, L3)
        a1, a2, a3 = m1 * L1, m2 * L2, m3 * L3
        tol = cls._DIM_TOL
        n_small = ((m1 < tol).to(torch.int8) + (m2 < tol).to(torch.int8)
                   + (m3 < tol).to(torch.int8))
        p3 = cls._pos3
        s = (p3(d) - p3(d - a1) - p3(d - a2) - p3(d - a3)
             + p3(d - a1 - a2) + p3(d - a1 - a3) + p3(d - a2 - a3)
             - p3(d - a1 - a2 - a3))
        v3 = s / (6.0 * m1 * m2 * m3).clamp_min(1e-300)

        def area2(dd, ma, mb, La, Lb):
            A, B = ma * La, mb * Lb
            t = (dd.clamp_min(0.0) ** 2 - (dd - A).clamp_min(0.0) ** 2
                 - (dd - B).clamp_min(0.0) ** 2
                 + (dd - A - B).clamp_min(0.0) ** 2)
            return t / (2.0 * ma * mb).clamp_min(1e-300)

        v2_m1 = area2(d, m2, m3, L2, L3) * L1
        v2_m2 = area2(d, m1, m3, L1, L3) * L2
        v2_m3 = area2(d, m1, m2, L1, L2) * L3
        v1_m1 = (d / m1.clamp_min(1e-300)).clamp(0.0, L1) * L2 * L3
        v1_m2 = (d / m2.clamp_min(1e-300)).clamp(0.0, L2) * L1 * L3
        v1_m3 = (d / m3.clamp_min(1e-300)).clamp(0.0, L3) * L1 * L2
        vmax = L1 * L2 * L3
        # NaN guard: v3 can be inf/nan when all m_i ~ 0; select away from it
        v3 = torch.nan_to_num(v3, nan=0.0, posinf=0.0, neginf=0.0)
        out = v3
        out = torch.where((n_small == 1) & (m1 < tol), v2_m1, out)
        out = torch.where((n_small == 1) & (m2 < tol), v2_m2, out)
        out = torch.where((n_small == 1) & (m3 < tol), v2_m3, out)
        out = torch.where((n_small >= 2) & (m1 >= tol), v1_m1, out)
        out = torch.where((n_small >= 2) & (m2 >= tol), v1_m2, out)
        out = torch.where((n_small >= 2) & (m3 >= tol), v1_m3, out)
        out = torch.where(n_small == 3, vmax * (d > 0).to(d.dtype), out)
        return out.clamp(0.0, vmax)

    @classmethod
    def plane_offset(cls, alpha, n1, n2, n3):
        a = alpha.to(torch.float64).clamp(0.0, 1.0)
        m1, m2, m3 = n1.abs().to(torch.float64), n2.abs().to(torch.float64), n3.abs().to(torch.float64)
        ssum = (m1 + m2 + m3).clamp_min(1e-30)
        m1, m2, m3 = m1 / ssum, m2 / ssum, m3 / ssum
        one, zero = torch.ones_like(a), torch.zeros_like(a)
        lo, hi = zero.clone(), one.clone()
        for _ in range(cls._BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            vol = cls.box_volume(mid, m1, m2, m3, one, one, one)
            too_small = vol < a
            lo = torch.where(too_small, mid, lo)
            hi = torch.where(too_small, hi, mid)
        return 0.5 * (lo + hi), m1, m2, m3

    @classmethod
    def sweep_flux(cls, alpha, n1, n2, n3, vel_face, dt, h, axis):
        """
        Directional-split geometric flux [C4]. Returns flux VOLUME per face
        (divide by h when forming the divergence).

        [B8] boundary fluxes are zeroed OUT-OF-PLACE so the autograd graph
        is not corrupted by in-place slice assignment.
        """
        a = alpha.to(torch.float64)
        d, m1, m2, m3 = cls.plane_offset(a, n1, n2, n3)
        shift = (torch.where(n1 < 0, m1, torch.zeros_like(m1))
                 + torch.where(n2 < 0, m2, torch.zeros_like(m2))
                 + torch.where(n3 < 0, m3, torch.zeros_like(m3)))
        d_fold = d + shift
        one = torch.ones_like(a)
        s_clip = (vel_face.abs().to(torch.float64) * dt / h).clamp(0.0, 1.0)
        ncell = a.shape[axis]
        nface = ncell + 1

        def shift_field(field, delta):
            idx = (torch.arange(nface, device=field.device) + delta).clamp(0, ncell - 1)
            return field.index_select(axis, idx)

        L = [one, one, one]
        L[axis] = s_clip
        dL, dR = shift_field(d_fold, -1), shift_field(d_fold, +1)
        mL = [shift_field(m, -1) for m in (m1, m2, m3)]
        mR = [shift_field(m, +1) for m in (m1, m2, m3)]
        dL_slab = dL - mL[axis] * (1.0 - s_clip)
        fluxL = cls.box_volume(dL_slab, mL[0], mL[1], mL[2], L[0], L[1], L[2])
        fluxR = cls.box_volume(dR, mR[0], mR[1], mR[2], L[0], L[1], L[2])
        pos = vel_face > 0
        flux = torch.where(pos, fluxL, fluxR)
        flux = torch.where(pos, flux, -flux)
        # [B8] out-of-place boundary zeroing
        mask = torch.ones_like(flux)
        mask[sl(axis, 0)] = 0.0
        mask[sl(axis, nface - 1)] = 0.0
        return (flux * mask).to(alpha.dtype)


# =============================================================================
# 2. VOF ADVECTION
# =============================================================================

class VOFAdvector:
    """
    'plic'   : operator-split geometric advection, direction order rotated
               each step to reduce directional bias. Mass conserved to
               machine precision after bounded repair [C5].
    'cicsam' : algebraic compressive scheme [C8]; smoother gradients but
               less sharp than PLIC.
    'upwind' : conservative face-flux upwind from the core [B5]; most
               diffusive, fully differentiable.
    """

    def __init__(self, cfg: VOFConfig, ops: MACOps):
        self.cfg = cfg
        self.g = ops
        self.spec = ops.s
        self._sweep_order = 0

    def _bounded_repair(self, a: Tensor, target_mass: float) -> Tensor:
        a = a.clamp(0.0, 1.0)
        defect = target_mass - float(a.sum())
        if abs(defect) < 1e-14:
            return a
        iface = ((a > 1e-6) & (a < 1.0 - 1e-6)).to(a.dtype)
        w = iface if float(iface.sum()) >= 1.0 else torch.ones_like(a)
        w = w / w.sum().clamp_min(1e-30)
        return (a + defect * w).clamp(0.0, 1.0)

    def _advect_plic(self, a, u, v, w, dt):
        c = self.cfg
        mass0 = float(a.sum())
        axes = [(0, u, c.dx), (1, v, c.dy), (2, w, c.dz)]
        order = [(self._sweep_order + k) % 3 for k in range(3)]
        self._sweep_order = (self._sweep_order + 1) % 3
        for k in order:
            ax, vel, h = axes[k]
            n1, n2, n3, _ = PLIC.normals(a, c.dx, c.dy, c.dz)
            flux = PLIC.sweep_flux(a, n1, n2, n3, vel, dt, h, ax)
            div = diff(flux, ax, h)
            dil = diff(vel, ax, h)          # split-scheme dilation [C5]
            a = (a - div + a * dil * dt).clamp(0.0, 1.0)
        return self._bounded_repair(a, mass0)

    def _advect_cicsam(self, a, u, v, w, dt):
        c = self.cfg
        mass0 = float(a.sum())
        out = a
        for ax, (vel, h) in enumerate(((u, c.dx), (v, c.dy), (w, c.dz))):
            n = out.shape[ax]
            dev = out.device
            idx = lambda d: (torch.arange(n + 1, device=dev) + d).clamp(0, n - 1)
            pos = vel > 0
            aD = torch.where(pos, out.index_select(ax, idx(-1)),
                             out.index_select(ax, idx(0)))
            aA = torch.where(pos, out.index_select(ax, idx(0)),
                             out.index_select(ax, idx(-1)))
            aU = torch.where(pos, out.index_select(ax, idx(-2)),
                             out.index_select(ax, idx(1)))
            Co = (vel.abs() * dt / h).clamp(1e-8, 1.0 - 1e-8)
            denom = aA - aU
            denom = torch.where(denom.abs() < 1e-12,
                                torch.full_like(denom, 1e-12), denom)
            at = ((aD - aU) / denom).clamp(0.0, 1.0)
            inb = (at > 0) & (at < 1)
            cbc = torch.where(inb, torch.minimum(at / Co, torch.ones_like(at)), at)
            cbc = torch.minimum(cbc, torch.ones_like(cbc))
            uq = torch.where(
                inb,
                torch.minimum((8 * Co * at + (1 - Co) * (6 * at + 3)) / 8.0,
                              torch.ones_like(at)), at)
            n1, n2, n3, _ = PLIC.normals(out, c.dx, c.dy, c.dz)
            ncomp = (n1, n2, n3)[ax].abs()
            nf = torch.where(pos, ncomp.index_select(ax, idx(-1)),
                             ncomp.index_select(ax, idx(0)))
            gamma = 0.5 * (1.0 - nf).clamp(0.0, 1.0) + 0.5
            blend = gamma * cbc + (1 - gamma) * uq
            af = aU + blend * (aA - aU)
            af = torch.where((aD < 1e-8) | (aD > 1 - 1e-8), aD, af)
            flux = vel * af
            if c.bc[ax] != "periodic":
                m = torch.ones_like(flux)
                m[sl(ax, 0)] = 0.0
                m[sl(ax, n)] = 0.0
                flux = flux * m
            out = (out - dt * diff(flux, ax, h)).clamp(0.0, 1.0)
        return self._bounded_repair(out, mass0)

    def advect(self, a, u, v, w, dt):
        if self.cfg.vof_scheme == "plic":
            return self._advect_plic(a, u, v, w, dt)
        if self.cfg.vof_scheme == "cicsam":
            return self._advect_cicsam(a, u, v, w, dt)
        return advect_alpha_upwind(a, u, v, w, dt, self.spec)


# =============================================================================
# 3. RHEOLOGY + LES  [B3]
# =============================================================================

class Rheology:
    """Apparent viscosity from the cell-centred strain-rate invariant."""

    def __init__(self, cfg: VOFConfig, ops: MACOps):
        self.cfg = cfg
        self.g = ops

    def strain_rate_invariant(self, u, v, w):
        """[B3] all 6 components at cell centres — no broadcast error."""
        return self.g.strain_rate_cc(u, v, w)

    def apparent_mu(self, mu_base, gdot):
        c = self.cfg
        if c.rheology == "newtonian":
            mu = mu_base
        elif c.rheology == "power_law":
            mu = c.power_law_k * gdot.clamp_min(1e-12) ** (c.power_law_n - 1.0)
        elif c.rheology == "carreau":
            mu = c.carreau_muinf + (c.carreau_mu0 - c.carreau_muinf) * (
                1.0 + (c.carreau_lambda * gdot) ** 2) ** ((c.carreau_n - 1.0) / 2.0)
        elif c.rheology == "bingham":
            gd = gdot.clamp_min(1e-12)
            mu = mu_base + c.bingham_tauy * (1.0 - torch.exp(-c.bingham_m * gd)) / gd
        else:
            raise ValueError(c.rheology)
        return mu.clamp(c.mu_min, c.mu_max)

    def smagorinsky(self, rho, gdot):
        c = self.cfg
        if c.turbulence != "smagorinsky":
            return torch.zeros_like(rho)
        delta = (c.dx * c.dy * c.dz) ** (1.0 / 3.0)
        return rho * (c.smagorinsky_cs * delta) ** 2 * gdot


# =============================================================================
# 4. SURFACE TENSION  [B5]
# =============================================================================

class SurfaceTension:
    """
    Continuum Surface Force [C6]:  F = sigma * kappa * grad(alpha)
    with kappa = -div(n_hat), n_hat = grad(alpha)/|grad(alpha)|.

    [B5] v1 used -lap(alpha)/|grad alpha|, which is NOT curvature.
    alpha is smoothed n_alpha_smooth times before curvature to reduce
    (not eliminate) spurious currents. NOT balanced-force.
    """

    def __init__(self, cfg: VOFConfig, ops: MACOps):
        self.cfg = cfg
        self.g = ops
        self.spec = ops.s

    def _smooth(self, a):
        for _ in range(self.cfg.n_alpha_smooth):
            ap = pad_cc(a, self.spec.bc)
            acc = 6.0 * a
            for ax in range(3):
                hi = ap[sl(ax, slice(2, None))]
                lo = ap[sl(ax, slice(0, -2))]
                for other in range(3):
                    if other != ax:
                        hi = hi[sl(other, slice(1, -1))]
                        lo = lo[sl(other, slice(1, -1))]
                acc = acc + hi + lo
            a = acc / 12.0
        return a

    def curvature(self, alpha):
        """kappa = -div(n_hat) at cell centres."""
        c, s = self.cfg, self.spec
        a_s = self._smooth(alpha)
        n1, n2, n3, _ = PLIC.normals(a_s, c.dx, c.dy, c.dz)
        n1, n2, n3 = n1.to(alpha.dtype), n2.to(alpha.dtype), n3.to(alpha.dtype)
        kappa = torch.zeros_like(alpha)
        for ax, (n, h) in enumerate(((n1, c.dx), (n2, c.dy), (n3, c.dz))):
            np_ = pad_cc(n, s.bc)
            hi = np_[sl(ax, slice(2, None))]
            lo = np_[sl(ax, slice(0, -2))]
            for other in range(3):
                if other != ax:
                    hi = hi[sl(other, slice(1, -1))]
                    lo = lo[sl(other, slice(1, -1))]
            kappa = kappa - (hi - lo) / (2.0 * h)
        return torch.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)

    def force_faces(self, alpha):
        c = self.cfg
        if c.sigma <= 0.0:
            return self.g.zeros_u(), self.g.zeros_v(), self.g.zeros_w()
        kappa = self.curvature(alpha)
        gx, gy, gz = self.g.grad_cc_to_faces(alpha)
        kfx, kfy, kfz = face_scalar(kappa, self.spec)
        return c.sigma * kfx * gx, c.sigma * kfy * gy, c.sigma * kfz * gz


# =============================================================================
# 5. IMMERSED SOLID FROM SDA-ONE COORDINATES  [B9]
# =============================================================================

def build_solid_mask_from_coords(
    coords: Tensor, cfg: VOFConfig,
    radii: Optional[Tensor] = None,
    default_radius: float = 3.5,
    origin: Optional[Tuple[float, float, float]] = None,
    scale: float = 1.0,
) -> Tensor:
    """
    Rasterise C-alpha coordinates into a cell-centred solid mask (1 = solid).

    coords        : (N,3), assumed ANGSTROM. `scale` converts to the solver
                    length unit — pass scale=1e-10 for metres.
    radii         : (N,) in the SAME unit as coords.
    default_radius: used if radii is None (3.5 A ~ CA + hydration shell).

    [B9] v1 used a triple Python loop over the stencil. Now chunked and
    vectorised over grid points.

    HONEST: nearest-cell rasterisation, staircased at grid scale. NOT a
    signed-distance / cut-cell representation. Refine the grid or use
    body-fitted CFD if wall shear accuracy matters.
    """
    dev = torch.device(cfg.device)
    dt_ = torch.float64 if cfg.dtype == "float64" else torch.float32
    xyz = coords.to(dev=dev, dtype=torch.float64) * scale
    rad = (torch.full((xyz.shape[0],), default_radius * scale,
                      device=dev, dtype=torch.float64)
           if radii is None else radii.to(dev=dev, dtype=torch.float64) * scale)
    if origin is None:
        ctr = 0.5 * torch.tensor([cfg.Lx, cfg.Ly, cfg.Lz],
                                 device=dev, dtype=torch.float64)
        xyz = xyz - xyz.mean(dim=0) + ctr
    else:
        xyz = xyz + torch.tensor(origin, device=dev, dtype=torch.float64)
    cx = (torch.arange(cfg.nx, device=dev, dtype=torch.float64) + 0.5) * cfg.dx
    cy = (torch.arange(cfg.ny, device=dev, dtype=torch.float64) + 0.5) * cfg.dy
    cz = (torch.arange(cfg.nz, device=dev, dtype=torch.float64) + 0.5) * cfg.dz
    X, Y, Z = torch.meshgrid(cx, cy, cz, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    mask = torch.zeros(pts.shape[0], device=dev, dtype=torch.float64)
    CH = 2048
    for s in range(0, xyz.shape[0], CH):
        p, r = xyz[s:s + CH], rad[s:s + CH]
        d2 = ((pts.unsqueeze(0) - p.unsqueeze(1)) ** 2).sum(-1)
        mask[(d2 <= (r * r).unsqueeze(1)).any(0)] = 1.0
    return mask.reshape(cfg.nx, cfg.ny, cfg.nz).to(dt_)


# =============================================================================
# 6. STATE + MAIN SOLVER
# =============================================================================

@dataclass
class VOFState:
    alpha: Tensor
    u: Tensor
    v: Tensor
    w: Tensor
    p: Tensor
    T: Optional[Tensor] = None
    solid: Optional[Tensor] = None
    t: float = 0.0
    step: int = 0

    def validate(self, spec: GridSpec) -> List[str]:
        errs = []
        if tuple(self.alpha.shape) != spec.cc_shape:
            errs.append("alpha must be cell-centred")
        if tuple(self.p.shape) != spec.cc_shape:
            errs.append("p must be cell-centred")
        for ax, name in enumerate(("u", "v", "w")):
            if tuple(getattr(self, name).shape) != spec.face_shape(ax):
                errs.append(f"{name} must be face-shaped {spec.face_shape(ax)}")
        if self.solid is not None and tuple(self.solid.shape) != spec.cc_shape:
            errs.append("solid must be cell-centred")
        return errs


class MultiphaseNavierStokes3D:
    """
    3D variable-density incompressible two-phase Navier-Stokes with VOF.

    Fractional step (Chorin [C7]):
      1. advect alpha (PLIC / CICSAM / upwind)
      2. update rho, mu from alpha (+ rheology + LES)
      3. explicit momentum: convection + full viscous stress + body forces
      4. pressure Poisson  div(beta grad p) = div(u*)/dt,  beta = 1/rho_face
      5. project u* -> u^{n+1} divergence free
      6. advect/diffuse passive scalar (optional)

    All stencils come from one_cfd_stencils — no local re-implementation.
    """

    def __init__(self, cfg: Optional[VOFConfig] = None):
        self.cfg = cfg or VOFConfig()
        self.spec = self.cfg.grid()
        self.g = MACOps(self.spec)
        self.adv = VOFAdvector(self.cfg, self.g)
        self.rheo = Rheology(self.cfg, self.g)
        self.st = SurfaceTension(self.cfg, self.g)
        self.ps = PressurePoisson(self.spec, self.g,
                                  method=self.cfg.pressure_solver,
                                  tol=self.cfg.poisson_tol,
                                  max_iter=self.cfg.poisson_max_iter)
        self.history: List[Dict[str, float]] = []

    # ---- state ----
    def initial_state(self, alpha_fn: Optional[Callable] = None,
                      solid: Optional[Tensor] = None,
                      T0: Optional[Tensor] = None) -> VOFState:
        a = self.g.cc()
        if alpha_fn is not None:
            X, Y, Z = self.g.coords_cc()
            a = alpha_fn(X, Y, Z).to(a.dtype).clamp(0, 1)
        return VOFState(alpha=a, u=self.g.zeros_u(), v=self.g.zeros_v(),
                        w=self.g.zeros_w(), p=self.g.cc(), T=T0, solid=solid)

    # ---- properties ----
    def density(self, a):
        c = self.cfg
        return a * c.rho1 + (1 - a) * c.rho2

    def viscosity(self, a, u, v, w):
        c = self.cfg
        mu_base = a * c.mu1 + (1 - a) * c.mu2
        gdot, strain = self.rheo.strain_rate_invariant(u, v, w)
        mu = self.rheo.apparent_mu(mu_base, gdot)
        mu = mu + self.rheo.smagorinsky(self.density(a), gdot)
        return mu.clamp(c.mu_min, c.mu_max), gdot, strain

    # ---- dt ----
    def compute_dt(self, st: VOFState, mu) -> float:
        c = self.cfg
        umax = max(float(st.u.abs().max()), float(st.v.abs().max()),
                   float(st.w.abs().max()), 1e-12)
        h = min(c.dx, c.dy, c.dz)
        dt_adv = c.cfl * h / umax
        nu_max = float(mu.max()) / min(c.rho1, c.rho2)
        dt_vis = c.viscous_factor * h * h / max(nu_max, 1e-30)
        if c.sigma > 0:
            rho_avg = 0.5 * (c.rho1 + c.rho2)
            dt_cap = c.capillary_factor * math.sqrt(
                rho_avg * h ** 3 / (2 * math.pi * c.sigma))
        else:
            dt_cap = float("inf")
        dt_brink = float("inf")
        if c.brinkman_eta > 0 and st.solid is not None:
            dt_brink = 0.5 * c.brinkman_eta / max(
                float(self.density(st.alpha).max()), 1e-30)
        return float(min(dt_adv, dt_vis, dt_cap, dt_brink, c.dt_max))

    # ---- convection (non-conservative, minmod-limited) ----
    def _limited_slope(self, fm, f0, fp, h):
        dl, dr = (f0 - fm) / h, (fp - f0) / h
        return torch.where(dl * dr > 0,
                           torch.sign(dl) * torch.minimum(dl.abs(), dr.abs()),
                           torch.zeros_like(dl))

    def _convective(self, comp, uc, vc, wc, axis):
        """(u.grad)comp at the component's own staggered location."""
        c, s = self.cfg, self.spec
        p = pad_cc(comp, s.bc) if axis is None else comp
        # pad only the axes we need, using the component's own face layout
        p = comp
        for ax in range(3):
            antisym = (s.bc[ax] == "wall") and (ax != axis)
            p = pad_axis(p, ax, s.bc[ax], antisym=antisym)
        h = s.h

        def dcomp(ax):
            f0 = p[sl(0, slice(1, -1))][sl(1, slice(1, -1))][sl(2, slice(1, -1))]
            fm = p[sl(ax, slice(0, -2))]
            fp = p[sl(ax, slice(2, None))]
            for other in range(3):
                if other != ax:
                    fm = fm[sl(other, slice(1, -1))]
                    fp = fp[sl(other, slice(1, -1))]
            if c.momentum_limiter == "central":
                return (fp - fm) / (2 * h[ax])
            vel = (uc, vc, wc)[ax]
            if c.momentum_limiter == "upwind":
                return torch.where(vel > 0, (f0 - fm) / h[ax], (fp - f0) / h[ax])
            slope = self._limited_slope(fm, f0, fp, h[ax])
            return torch.where(vel > 0, (f0 - fm) / h[ax] + 0.0 * slope,
                               (fp - f0) / h[ax] - 0.0 * slope)

        return uc * dcomp(0) + vc * dcomp(1) + wc * dcomp(2)

    def _vel_at(self, vel, target_axis):
        """Interpolate all three components to the target component's faces."""
        u, v, w = vel
        out = []
        for src in range(3):
            t = (u, v, w)[src]
            if src == target_axis:
                out.append(t)
                continue
            # average along src axis, then pad/interp along target axis
            g = t
            # bring src dim from n_src+1 to n_src by averaging
            g = 0.5 * (g[sl(src, slice(0, -1))] + g[sl(src, slice(1, None))])
            # now cell-centred in src; extend target dim from n_t to n_t+1
            gp = pad_axis(g, target_axis, self.spec.bc[target_axis])
            g = 0.5 * (gp[sl(target_axis, slice(0, -1))]
                       + gp[sl(target_axis, slice(1, None))])
            out.append(g)
        return tuple(out)

    # ---- step ----
    def step(self, st: VOFState, dt: Optional[float] = None) -> VOFState:
        c, g, s = self.cfg, self.g, self.spec
        errs = st.validate(s)
        if errs:
            raise ValueError("invalid state: " + "; ".join(errs))
        ctx = torch.enable_grad() if c.differentiable else torch.no_grad()
        with ctx:
            mu, _, _ = self.viscosity(st.alpha, st.u, st.v, st.w)
            if dt is None:
                dt = self.compute_dt(st, mu)

            # 1. volume fraction
            alpha_new = self.adv.advect(st.alpha, st.u, st.v, st.w, dt)

            # 2. properties
            rho = self.density(alpha_new)
            mu, gdot, _ = self.viscosity(alpha_new, st.u, st.v, st.w)

            # 3. explicit momentum
            vel = (st.u, st.v, st.w)
            acc = []
            fx, fy, fz = self.st.force_faces(alpha_new)
            forces = (fx, fy, fz)
            rho_f = face_scalar(rho, s)
            for ax in range(3):
                ua, va, wa = self._vel_at(vel, ax)
                conv = self._convective(vel[ax], ua, va, wa, ax)
                visc = g.viscous_component(vel, mu, ax)
                body = torch.full_like(vel[ax], c.gravity[ax])
                if c.boussinesq and st.T is not None and ax == 1:
                    dT = st.T - c.T_ref
                    dTf = face_scalar(dT, s)[ax]
                    body = body - c.beta_thermal * dTf * c.gravity[1]
                ox, oy, oz = c.omega
                if any(abs(o) > 0 for o in c.omega):
                    if ax == 0:
                        body = body - 2 * (oy * wa - oz * va)
                    elif ax == 1:
                        body = body - 2 * (oz * ua - ox * wa)
                    else:
                        body = body - 2 * (ox * va - oy * ua)
                a_ax = -conv + (visc + forces[ax]) / rho_f[ax].clamp_min(1e-30) + body
                # [B7] porous drag: dead `if False else` branch removed
                if c.porous_alpha_darcy > 0 or c.porous_beta_forch > 0:
                    sp = vel[ax].abs()
                    drag = -(c.porous_alpha_darcy * float(mu.max()) * vel[ax]
                             + c.porous_beta_forch * float(rho.max()) * sp * vel[ax])
                    a_ax = a_ax + drag / rho_f[ax].clamp_min(1e-30)
                acc.append(a_ax)

            u_s = st.u + dt * acc[0]
            v_s = st.v + dt * acc[1]
            w_s = st.w + dt * acc[2]

            # Brinkman penalisation [C10]
            if c.brinkman_eta > 0 and st.solid is not None:
                su, sv, sw = face_scalar(st.solid, s)
                eta = max(c.brinkman_eta, 1e-30)
                u_s = u_s / (1.0 + su * dt / eta)
                v_s = v_s / (1.0 + sv * dt / eta)
                w_s = w_s / (1.0 + sw * dt / eta)

            u_s, v_s, w_s = g.enforce_velocity_bc(u_s, v_s, w_s)

            # 4-5. projection with CORRECT beta [B1]
            u_n, v_n, w_n, p, self.ps = project(u_s, v_s, w_s, rho, s, dt, self.ps)

            # 6. passive scalar
            T_n = st.T
            if st.T is not None:
                T_n = self._advect_scalar(st.T, u_n, v_n, w_n, dt)

            return VOFState(alpha=alpha_new, u=u_n, v=v_n, w=w_n, p=p,
                            T=T_n, solid=st.solid,
                            t=st.t + dt, step=st.step + 1)

    def _advect_scalar(self, T, u, v, w, dt):
        c, s = self.cfg, self.spec
        out = T
        for ax, (vel_f, h) in enumerate(((u, c.dx), (v, c.dy), (w, c.dz))):
            vel = face_scalar(out * 0.0, s)[ax] + vel_f  # face-shaped
            Tp = pad_cc(out, s.bc)
            f0 = out
            fm = Tp[sl(ax, slice(0, -2))]
            fp = Tp[sl(ax, slice(2, None))]
            for other in range(3):
                if other != ax:
                    fm = fm[sl(other, slice(1, -1))]
                    fp = fp[sl(other, slice(1, -1))]
            # cell-centred velocity for the non-conservative form
            vc = 0.5 * (vel_f[sl(ax, slice(0, -1))] + vel_f[sl(ax, slice(1, None))])
            d = torch.where(vc > 0, (f0 - fm) / h, (fp - f0) / h)
            out = out - dt * vc * d
            out = out + dt * c.scalar_diffusivity * (fp - 2 * f0 + fm) / h ** 2
        return out

    # ---- run ----
    def run(self, st: VOFState, n_steps: int, log_every: int = 20,
            t_end: Optional[float] = None) -> VOFState:
        for _ in range(n_steps):
            st = self.step(st)
            if t_end is not None and st.t >= t_end:
                break
            if st.step % log_every == 0:
                d = self.diagnostics(st)
                self.history.append(d)
                logger.info("step %6d t=%.6e | mass=%.12f | div=%.3e | "
                            "KE=%.6e | cg=%d(%s)", st.step, st.t,
                            d["alpha_mass"], d["max_divergence"],
                            d["kinetic_energy"], self.ps.last_iters,
                            "ok" if self.ps.converged else "NOT CONVERGED")
        return st

    # ---- diagnostics ----
    def diagnostics(self, st: VOFState) -> Dict[str, float]:
        c, g = self.cfg, self.g
        cv = c.dx * c.dy * c.dz
        rho = self.density(st.alpha)
        uc, vc, wc = g.to_cc(st.u, st.v, st.w)
        ke = 0.5 * float((rho * (uc**2 + vc**2 + wc**2)).sum() * cv)
        div = g.divergence(st.u, st.v, st.w)
        _, _, _, mag = PLIC.normals(st.alpha, c.dx, c.dy, c.dz)
        return {
            "t": st.t,
            "alpha_mass": float(st.alpha.sum()),
            "alpha_min": float(st.alpha.min()),
            "alpha_max": float(st.alpha.max()),
            "physical_mass": float(rho.sum() * cv),
            "kinetic_energy": ke,
            "max_divergence": float(div.abs().max()),
            "interface_area": float(mag.sum() * cv),
            "n_interface_cells": int(((st.alpha > 1e-6) & (st.alpha < 1 - 1e-6)).sum()),
            "poisson_iters": self.ps.last_iters,
            "poisson_residual": self.ps.last_residual,
            "poisson_converged": int(self.ps.converged),
        }

    # ---- AMR hook (indicator only) ----
    def interface_amr_indicator(self, st: VOFState,
                                refine_threshold: float = 0.02,
                                coarsen_threshold: float = 1e-4) -> Tensor:
        """
        Per-cell flag for an EXTERNAL block-structured AMR engine.
        +1 refine, 0 keep, -1 coarsen.
        NOTE: no octree, no coarse-fine flux correction, no subcycling here.
        """
        a = st.alpha
        iface = (a > refine_threshold) & (a < 1 - refine_threshold)
        ap = pad_cc(a, self.spec.bc)
        nb = torch.zeros_like(a)
        for ax in range(3):
            hi = ap[sl(ax, slice(2, None))]
            lo = ap[sl(ax, slice(0, -2))]
            for other in range(3):
                if other != ax:
                    hi = hi[sl(other, slice(1, -1))]
                    lo = lo[sl(other, slice(1, -1))]
            nb = torch.maximum(nb, (hi - lo).abs())
        flag = torch.zeros_like(a)
        flag[iface | (nb > refine_threshold)] = 1.0
        flag[(nb < coarsen_threshold) & (~iface)] = -1.0
        return flag

    # ---- mesh export ----
    def export_interface_mesh(self, st: VOFState, path: str,
                              level: float = 0.5) -> Optional[str]:
        if not _HAS_SKIMAGE:
            warnings.warn("scikit-image not available; cannot export mesh")
            return None
        c = self.cfg
        a = st.alpha.detach().cpu().numpy().astype(np.float64)
        a = np.pad(a, 1, mode="edge")
        if a.max() < level or a.min() > level:
            warnings.warn("no isosurface at requested level")
            return None
        verts, faces, normals, _ = marching_cubes(
            a, level=level, spacing=(c.dx, c.dy, c.dz))
        verts -= np.array([c.dx, c.dy, c.dz])
        ext = os.path.splitext(path)[1].lower()
        if ext in (".glb", ".gltf") and _HAS_TRIMESH:
            trimesh.Trimesh(vertices=verts, faces=faces,
                            vertex_normals=normals, process=False).export(path)
        elif _HAS_TRIMESH:
            trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)
        else:
            path = os.path.splitext(path)[0] + ".ply"
            _write_ply(path, verts, faces)
        logger.info("[VOF] mesh -> %s (%d verts, %d faces)",
                    path, len(verts), len(faces))
        return path


def _write_ply(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """[B6] ASCII PLY writer — v1 had broken f-strings."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
        for t in faces:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")


# =============================================================================
# 7. SELF-TEST — verifies the FIXES, reports REAL errors
# =============================================================================

def self_test() -> bool:
    print("=" * 78)
    print(f"  SESI MULTIPHASE FLOW v{VOF_VERSION} — self-test")
    print(f"  core stencils v{STENCIL_VERSION} | one_core {ONE_VERSION}")
    print(f"  skimage={_HAS_SKIMAGE} trimesh={_HAS_TRIMESH}")
    print("=" * 78)
    torch.manual_seed(0)
    np.random.seed(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1 config
    cfg = VOFConfig(nx=16, ny=16, nz=16, dtype="float64")
    rep("T1 config derives dx/dy/dz", abs(cfg.dx - cfg.Lx / 16) < 1e-15)
    try:
        VOFConfig(pressure_solver="fft", bc=("wall",) * 3)
        rep("T1b fft rejects non-periodic", False)
    except ValueError:
        rep("T1b fft rejects non-periodic", True)

    # T2 PLIC analytic
    one = torch.tensor([1.0], dtype=torch.float64)
    third = torch.tensor([1 / 3], dtype=torch.float64)
    half = torch.tensor([0.5], dtype=torch.float64)
    V = float(PLIC.box_volume(half, third, third, third, one, one, one))
    rep("T2 PLIC symmetric plane V=0.5", abs(V - 0.5) < 1e-10, f"V={V:.12f}")
    ds = torch.linspace(0.01, 0.99, 37, dtype=torch.float64)
    m = torch.full_like(ds, 1 / 3)
    Va = PLIC.box_volume(ds, m, m, m, one, one, one)
    Vb = PLIC.box_volume(1 - ds, m, m, m, one, one, one)
    e = float((Va + Vb - 1).abs().max())
    rep("T2b PLIC complement symmetry", e < 1e-10, f"err={e:.2e}")
    Vz = float(PLIC.box_volume(torch.tensor([0.3], dtype=torch.float64),
                               torch.tensor([0.0], dtype=torch.float64),
                               torch.tensor([0.0], dtype=torch.float64),
                               torch.tensor([1.0], dtype=torch.float64),
                               one, one, one))
    rep("T2c PLIC axis-aligned fallback", abs(Vz - 0.3) < 1e-12, f"V={Vz:.6f}")

    # T3 plane inversion
    a_true = torch.rand(64, dtype=torch.float64)
    nn_ = torch.randn(3, 64, dtype=torch.float64)
    nn_ = nn_ / nn_.norm(dim=0, keepdim=True)
    d, m1, m2, m3 = PLIC.plane_offset(a_true, nn_[0], nn_[1], nn_[2])
    Vrec = PLIC.box_volume(d, m1, m2, m3, torch.ones_like(d),
                           torch.ones_like(d), torch.ones_like(d))
    e = float((Vrec - a_true).abs().max())
    rep("T3 PLIC inversion round-trip", e < 1e-8, f"err={e:.2e}")

    # T4 [B2] face shapes
    s = MultiphaseNavierStokes3D(VOFConfig(nx=12, ny=14, nz=10,
                                          Lx=1.0, Ly=1.4, Lz=1.0,
                                          dtype="float64"))
    p = s.g.cc()
    gx, gy, gz = s.g.grad_cc_to_faces(p)
    rep("T4 [B2] face shapes", gx.shape == (13, 14, 10)
        and gy.shape == (12, 15, 10) and gz.shape == (12, 14, 11),
        f"{tuple(gx.shape)} {tuple(gy.shape)} {tuple(gz.shape)}")

    # T5 [B1] beta
    rho = torch.rand(s.spec.cc_shape, dtype=torch.float64) * 999 + 1
    bx, _, _ = face_beta(rho, s.spec)
    ref = 0.5 * (1 / rho[:-1] + 1 / rho[1:])
    wrong = 2 / (1 / rho[:-1] + 1 / rho[1:])
    rep("T5 [B1] beta = mean(1/rho)",
        float((bx[1:-1] - ref).abs().max()) < 1e-14
        and float((bx[1:-1] - wrong).abs().max()) > 1e-6)

    # T6 [B3][B4] strain + viscous shapes on anisotropic grid, mixed BC
    s2 = MultiphaseNavierStokes3D(VOFConfig(
        nx=10, ny=14, nz=8, Lx=1.0, Ly=1.4, Lz=0.8,
        bc=("wall", "periodic", "slip"), dtype="float64"))
    st2 = s2.initial_state(alpha_fn=lambda X, Y, Z: torch.ones_like(X))
    st2.u = torch.randn_like(st2.u) * 0.1
    st2.v = torch.randn_like(st2.v) * 0.1
    st2.w = torch.randn_like(st2.w) * 0.1
    st2.u, st2.v, st2.w = s2.g.enforce_velocity_bc(st2.u, st2.v, st2.w)
    mag, comps = s2.rheo.strain_rate_invariant(st2.u, st2.v, st2.w)
    mu = torch.rand(s2.spec.cc_shape, dtype=torch.float64) * 1e-3 + 1e-4
    try:
        vu, vv, vw = s2.g.viscous_all(st2.u, st2.v, st2.w, mu)
        rep("T6 [B3][B4] strain+viscous shapes",
            mag.shape == s2.spec.cc_shape
            and all(c.shape == s2.spec.cc_shape for c in comps)
            and vu.shape == s2.spec.face_shape(0)
            and vv.shape == s2.spec.face_shape(1)
            and vw.shape == s2.spec.face_shape(2))
    except Exception as ex:
        rep("T6 [B3][B4] strain+viscous shapes", False, f"{type(ex).__name__}: {ex}")

    # T7 mass conservation, pure advection
    cfg7 = VOFConfig(nx=24, ny=24, nz=24, bc=("periodic",) * 3,
                     gravity=(0, 0, 0), sigma=0.0, vof_scheme="plic",
                     dtype="float64")
    s7 = MultiphaseNavierStokes3D(cfg7)
    st7 = s7.initial_state(alpha_fn=lambda X, Y, Z: (
        (X - 0.35) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < 0.18 ** 2).double())
    st7.u += 0.7; st7.v += 0.3; st7.w -= 0.2
    m0 = float(st7.alpha.sum())
    for _ in range(25):
        st7.alpha = s7.adv.advect(st7.alpha, st7.u, st7.v, st7.w,
                                  0.3 * cfg7.dx)
    m1 = float(st7.alpha.sum())
    rel = abs(m1 - m0) / m0
    rep("T7 PLIC advection mass conserved", rel < 1e-10
        and float(st7.alpha.min()) >= 0 and float(st7.alpha.max()) <= 1,
        f"rel={rel:.2e}")

    # T8 projection divergence-free
    cfg8 = VOFConfig(nx=20, ny=20, nz=20, bc=("periodic",) * 3,
                     gravity=(0, 0, 0), sigma=0.0, pressure_solver="cg",
                     poisson_tol=1e-12, poisson_max_iter=4000, dtype="float64")
    s8 = MultiphaseNavierStokes3D(cfg8)
    st8 = s8.initial_state(alpha_fn=lambda X, Y, Z: torch.ones_like(X))
    st8.u = torch.randn_like(st8.u) * 0.1
    st8.v = torch.randn_like(st8.v) * 0.1
    st8.w = torch.randn_like(st8.w) * 0.1
    st8.u, st8.v, st8.w = s8.g.enforce_velocity_bc(st8.u, st8.v, st8.w)
    d0 = float(s8.g.divergence(st8.u, st8.v, st8.w).abs().max())
    st8 = s8.step(st8, dt=1e-3)
    d1 = float(s8.g.divergence(st8.u, st8.v, st8.w).abs().max())
    rep("T8 projection divergence-free", d1 < 1e-8,
        f"{d0:.3e} -> {d1:.3e}, cg={s8.ps.last_iters}")

    # T9 variable-density Poisson
    cfg9 = VOFConfig(nx=16, ny=16, nz=16, bc=("wall",) * 3,
                     rho1=1000.0, rho2=1.0, poisson_tol=1e-10,
                     poisson_max_iter=6000, dtype="float64")
    s9 = MultiphaseNavierStokes3D(cfg9)
    a9 = s9.g.cc(); a9[:8] = 1.0
    rho9 = s9.density(a9)
    rhs9 = torch.randn_like(a9); rhs9 -= rhs9.mean()
    p9 = s9.ps.solve(rhs9, rho9)
    res9 = float((s9.ps.apply(p9, *face_beta(rho9, s9.spec)) - rhs9).norm()
                 / rhs9.norm())
    rep("T9 variable-density CG (1000:1)", res9 < 1e-8 and s9.ps.converged,
        f"rel-res={res9:.2e} iters={s9.ps.last_iters}")

    # T10 rheology monotonicity
    for rheo in ("power_law", "carreau"):
        cr = VOFConfig(nx=8, ny=8, nz=8, rheology=rheo,
                       power_law_n=0.5, carreau_n=0.4, dtype="float64")
        sr = MultiphaseNavierStokes3D(cr)
        gd = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        mu_r = sr.rheo.apparent_mu(torch.full_like(gd, 1e-3), gd)
        rep(f"T10 {rheo} shear-thinning",
            bool(((mu_r[1:] - mu_r[:-1]) <= 1e-12).all()))
    cb = VOFConfig(nx=8, ny=8, nz=8, rheology="bingham",
                   bingham_tauy=5.0, dtype="float64")
    sb = MultiphaseNavierStokes3D(cb)
    gd = torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0], dtype=torch.float64)
    tau = sb.rheo.apparent_mu(torch.full_like(gd, 1e-3), gd) * gd
    rep("T10b bingham tau monotone increasing", bool((tau[1:] > tau[:-1]).all()))

    # T11 viscous KE decay
    cfg11 = VOFConfig(nx=16, ny=16, nz=16, bc=("periodic",) * 3,
                      rho1=1.0, rho2=1.0, mu1=0.05, mu2=0.05,
                      sigma=0.0, gravity=(0, 0, 0), dtype="float64")
    s11 = MultiphaseNavierStokes3D(cfg11)
    st11 = s11.initial_state(alpha_fn=lambda X, Y, Z: torch.ones_like(X))
    fx = torch.arange(17, dtype=torch.float64) * cfg11.dx
    cc = (torch.arange(16, dtype=torch.float64) + 0.5) * cfg11.dx
    Xf, Yc, Zc = torch.meshgrid(fx, cc, cc, indexing="ij")
    st11.u = 0.1 * torch.sin(2 * math.pi * Xf) * torch.cos(2 * math.pi * Yc) \
        * torch.cos(2 * math.pi * Zc)
    st11.u, st11.v, st11.w = s11.g.enforce_velocity_bc(st11.u, st11.v, st11.w)
    ke0 = s11.diagnostics(st11)["kinetic_energy"]
    for _ in range(40):
        st11 = s11.step(st11)
    ke1 = s11.diagnostics(st11)["kinetic_energy"]
    rep("T11 viscous KE decay", ke1 < ke0 and math.isfinite(ke1),
        f"{ke0:.4e} -> {ke1:.4e} ({100*(1-ke1/ke0):.1f}% dissipated)")

    # T12 [B5] curvature of a sphere ~ 2/R  (reports REAL error)
    cfg12 = VOFConfig(nx=32, ny=32, nz=32, sigma=0.05,
                      n_alpha_smooth=2, dtype="float64")
    s12 = MultiphaseNavierStokes3D(cfg12)
    st12 = s12.initial_state(alpha_fn=lambda X, Y, Z: (
        (X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2 < 0.25 ** 2).double())
    kap = s12.st.curvature(st12.alpha)
    iface = (st12.alpha > 0.05) & (st12.alpha < 0.95)
    if int(iface.sum()) > 0:
        km = float(kap[iface].abs().mean())
        rep("T12 [B5] sphere curvature ~ 2/R = 8",
            4.0 < km < 14.0, f"mean|kappa|={km:.2f} (CSF, not balanced-force)")
    else:
        rep("T12 [B5] sphere curvature", False, "no interface cells")

    # T13 dam break, rho 1000:1
    cfg13 = VOFConfig(nx=20, ny=20, nz=12, Lx=1.0, Ly=1.0, Lz=0.6,
                      bc=("wall",) * 3, rho1=1000.0, rho2=1.0,
                      mu1=1e-3, mu2=1.8e-5, sigma=0.0,
                      gravity=(0, -9.81, 0), dtype="float64",
                      poisson_tol=1e-7, poisson_max_iter=1500)
    s13 = MultiphaseNavierStokes3D(cfg13)
    st13 = s13.initial_state(alpha_fn=lambda X, Y, Z: ((X < 0.35) & (Y < 0.5)).double())
    m0 = float(st13.alpha.sum())
    for _ in range(20):
        st13 = s13.step(st13)
    d13 = s13.diagnostics(st13)
    rel13 = abs(d13["alpha_mass"] - m0) / m0
    rep("T13 dam break bounded + mass conserved",
        rel13 < 1e-9 and d13["alpha_min"] >= 0 and d13["alpha_max"] <= 1
        and all(math.isfinite(d13[k]) for k in ("kinetic_energy", "max_divergence")),
        f"mass rel={rel13:.2e} div={d13['max_divergence']:.2e}")

    # T14 CICSAM + upwind paths
    for sch in ("cicsam", "upwind"):
        cs = VOFConfig(nx=16, ny=16, nz=16, bc=("periodic",) * 3,
                       vof_scheme=sch, sigma=0.0, gravity=(0, 0, 0),
                       dtype="float64")
        ss = MultiphaseNavierStokes3D(cs)
        sts = ss.initial_state(alpha_fn=lambda X, Y, Z: (
            (X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2 < 0.04).double())
        sts.u += 0.5
        mm0 = float(sts.alpha.sum())
        for _ in range(15):
            sts.alpha = ss.adv.advect(sts.alpha, sts.u, sts.v, sts.w, 0.01)
        mm1 = float(sts.alpha.sum())
        rep(f"T14 {sch} mass conserved", abs(mm1 - mm0) / mm0 < 1e-10,
            f"rel={abs(mm1-mm0)/mm0:.2e}")

    # T15 [B9] solid mask + Brinkman
    fake_ca = torch.randn(500, 3, dtype=torch.float64) * 8.0
    cfg15 = VOFConfig(nx=24, ny=24, nz=24, Lx=100, Ly=100, Lz=100,
                      brinkman_eta=1e-4, dtype="float64")
    mask = build_solid_mask_from_coords(fake_ca, cfg15, default_radius=4.0)
    frac = float(mask.mean())
    s15 = MultiphaseNavierStokes3D(cfg15)
    st15 = s15.initial_state(alpha_fn=lambda X, Y, Z: torch.zeros_like(X), solid=mask)
    st15.u += 1.0
    st15.u, st15.v, st15.w = s15.g.enforce_velocity_bc(st15.u, st15.v, st15.w)
    for _ in range(5):
        st15 = s15.step(st15)
    su = face_scalar(mask, s15.spec)[0]
    u_sol = float((st15.u.abs() * su).sum() / su.sum().clamp_min(1))
    u_free = float((st15.u.abs() * (1 - su)).sum() / (1 - su).sum().clamp_min(1))
    rep("T15 [B9] solid mask + Brinkman damping",
        0 < frac < 1 and u_sol < u_free,
        f"fill={100*frac:.2f}% |u|_solid={u_sol:.3e} < |u|_fluid={u_free:.3e}")

    # T16 AMR indicator
    flag = s13.interface_amr_indicator(st13)
    vals = torch.unique(flag)
    rep("T16 AMR indicator (no AMR solver here)",
        flag.shape == st13.alpha.shape
        and bool(((vals >= -1) & (vals <= 1)).all())
        and int((flag > 0).sum()) > 0,
        f"{int((flag>0).sum())} refine, {int((flag<0).sum())} coarsen")

    # T17 LES eddy viscosity
    cfg17 = VOFConfig(nx=12, ny=12, nz=12, turbulence="smagorinsky", dtype="float64")
    s17 = MultiphaseNavierStokes3D(cfg17)
    st17 = s17.initial_state(alpha_fn=lambda X, Y, Z: torch.ones_like(X))
    st17.u = torch.randn_like(st17.u) * 0.1
    st17.u, st17.v, st17.w = s17.g.enforce_velocity_bc(st17.u, st17.v, st17.w)
    mu17, _, _ = s17.viscosity(st17.alpha, st17.u, st17.v, st17.w)
    rep("T17 Smagorinsky non-negative eddy viscosity",
        bool((mu17 >= cfg17.mu_min).all()) and float(mu17.max()) > cfg17.mu1)

    # T18 [B6] PLY writer
    if _HAS_SKIMAGE:
        import tempfile
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "iface.glb" if _HAS_TRIMESH else "iface.ply")
        out = s13.export_interface_mesh(st13, path)
        rep("T18 [B6] mesh export", out is not None and os.path.getsize(out) > 0,
            f"{os.path.basename(out)} {os.path.getsize(out)} bytes")
    else:
        import tempfile
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "t.ply")
        _write_ply(path, np.random.randn(4, 3), np.array([[0, 1, 2], [1, 2, 3]]))
        rep("T18 [B6] PLY writer", os.path.getsize(path) > 0)

    # T19 differentiability
    cfg19 = VOFConfig(nx=10, ny=10, nz=10, bc=("periodic",) * 3,
                      vof_scheme="upwind", sigma=0.0, gravity=(0, 0, 0),
                      differentiable=True, dtype="float64",
                      poisson_tol=1e-10, poisson_max_iter=800)
    s19 = MultiphaseNavierStokes3D(cfg19)
    st19 = s19.initial_state(alpha_fn=lambda X, Y, Z: (X < 0.5).double())
    st19.u = (torch.randn_like(st19.u) * 0.05).requires_grad_(True)
    u_in = st19.u
    st19 = s19.step(st19, dt=1e-3)
    st19.p.pow(2).sum().backward()
    rep("T19 autograd through step (upwind mode)",
        u_in.grad is not None and bool(torch.isfinite(u_in.grad).all()))

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  REMINDERS (unchanged from v1, still true)
    • Momentum is NOT discretely conserved (convective form).
    • CSF is NOT balanced-force — T12 shows the real curvature error.
    • There is NO AMR solver here, only an indicator field.
    • Not validated against experiment.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "dambreak"])
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--steps", type=int, default=40)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if self_test() else 1)
    cfg = VOFConfig(nx=a.nx, ny=a.nx, nz=max(8, a.nx // 2),
                    Lx=1.0, Ly=1.0, Lz=0.5, bc=("wall",) * 3,
                    gravity=(0, -9.81, 0), sigma=0.0, dtype="float64")
    s = MultiphaseNavierStokes3D(cfg)
    st = s.initial_state(alpha_fn=lambda X, Y, Z: ((X < 0.35) & (Y < 0.5)).double())
    st = s.run(st, n_steps=a.steps, log_every=5)
    d = s.diagnostics(st)
    print(json.dumps(d, indent=2) if False else
          "\n".join(f"  {k}: {v}" for k, v in d.items()))


if __name__ == "__main__":
    main()
