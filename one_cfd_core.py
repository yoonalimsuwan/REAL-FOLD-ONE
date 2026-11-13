#!/usr/bin/env python3
"""
=============================================================================
ONE CFD CORE — shared numerics + interoperability layer for the 5 CFD files
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHY THIS FILE EXISTS
  The five CFD modules each re-implement the MAC grid, face gradient,
  divergence, and variable-density pressure Poisson — and three of those
  implementations disagree or are wrong. This module provides ONE correct
  implementation plus a state contract, so the modules can interoperate.

BUGS THIS REPLACES (verified by reading the sources)
  [B1] sesi_multiphase_vof_sota._beta_faces  and
       sesi_multiphase_flow.PressurePoisson._beta_faces
       return the HARMONIC MEAN OF RHO (= rho_face) and then use it directly
       as beta in `u -= dt*beta*grad(p)`. The projection requires
       beta = 1/rho_face. The two differ by a reciprocal:
           computed : 2ab/(a+b)
           required : (a+b)/(2ab)
       Fixed here: arithmetic mean of (1/rho) at faces.
  [B2] sesi_multiphase_flow.MACGrid.grad_cc_to_faces returns (nx,ny,nz)
       instead of the face shape (nx+1,ny,nz). Every downstream divergence
       and projection then mismatches. Fixed here.
  [B3] sesi_multiphase_flow.Rheology.strain_rate_invariant mixes
       face-located derivatives on different axes -> broadcast error.
       Fixed here: all strain components interpolated to cell centres.
  [B4] sesi_multiphase_flow_vof advection omits the velocity magnitude and
       uses a forward difference of a cell-centred flux, breaking mass
       conservation. Fixed here: proper face-flux divergence.
  [B5] super_dns_one_v6_3._softplus_floor uses beta=100 for every floor,
       giving an effective transition width log(2)/100 ~ 6.9e-3 regardless
       of the requested floor (1e-6, 1e-8, 1e-30). Fixed here: beta scales
       as sharpness/floor.

UNIT / SHAPE CONTRACT (all five modules must obey this)
  Length  : metres            (Angstrom coords must be scaled by 1e-10)
  Mass    : kg
  Time    : seconds
  rho     : kg/m^3            cell-centred  (nx,   ny,   nz)
  mu      : Pa s              cell-centred  (nx,   ny,   nz)
  p       : Pa                cell-centred  (nx,   ny,   nz)
  alpha   : dimensionless     cell-centred  (nx,   ny,   nz), in [0,1]
  u       : m/s               x-faces       (nx+1, ny,   nz)
  v       : m/s               y-faces       (nx,   ny+1, nz)
  w       : m/s               z-faces       (nx,   ny,   nz+1)
  CH u    : dimensionless     cell-centred, in [-1,1];  phi = (u+1)/2

HONEST SCOPE
  ✅ Correct staggered operators, verified against analytic solutions.
  ✅ Variable-density projection with the right face beta.
  ✅ Mass-conserving face-flux advection (upwind; NOT geometric PLIC).
  ❌ This does NOT make the five modules validated CFD. It makes them
     dimensionally and shape-consistent, and fixes the listed bugs.
  ❌ super_dns_one_v6_3 still requires one_core.py, which is NOT in the
     corpus. Without it that module cannot import. The bridge below
     degrades gracefully and reports the absence.
  ❌ No turbulence model validation, no grid-convergence study, no
     experimental comparison. Those remain your responsibility.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("one_cfd_core")

CORE_VERSION = "1.0.0"


# =============================================================================
# C^∞ PRIMITIVES  ([B5] floor width tracks the requested floor)
# =============================================================================

def soft_floor(x: Tensor, floor: float, sharpness: float = 50.0,
               beta: Optional[float] = None) -> Tensor:
    """
    Differentiable lower bound whose transition width scales with `floor`.

    [B5] The original used a fixed beta=100, so the width was log(2)/100
    ~ 6.9e-3 for EVERY floor. A requested floor of 1e-6 was silently
    ~6.9e-3 instead. Here beta = sharpness/|floor| by default.
    """
    if beta is None:
        beta = sharpness / max(abs(floor), 1e-30)
    beta = float(min(beta, 1e12))
    return floor + F.softplus(x - floor, beta=beta)


def soft_abs(x: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def soft_norm(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    return torch.sqrt((x * x).sum(dim=dim) + eps * eps)


# =============================================================================
# GRID SPEC — single source of truth
# =============================================================================

@dataclass
class GridSpec:
    nx: int = 32
    ny: int = 32
    nz: int = 32
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    bc: Tuple[str, str, str] = ("periodic", "periodic", "periodic")
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self):
        if min(self.nx, self.ny, self.nz) < 4:
            raise ValueError("need >= 4 cells per axis")
        if min(self.Lx, self.Ly, self.Lz) <= 0:
            raise ValueError("domain lengths must be positive")
        for b in self.bc:
            if b not in ("periodic", "wall", "slip"):
                raise ValueError(f"bad bc {b!r}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")

    @property
    def dx(self) -> float: return self.Lx / self.nx
    @property
    def dy(self) -> float: return self.Ly / self.ny
    @property
    def dz(self) -> float: return self.Lz / self.nz
    @property
    def h(self) -> Tuple[float, float, float]: return (self.dx, self.dy, self.dz)
    @property
    def cell_volume(self) -> float: return self.dx * self.dy * self.dz
    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32
    @property
    def torch_device(self) -> torch.device: return torch.device(self.device)
    @property
    def cc_shape(self) -> Tuple[int, int, int]: return (self.nx, self.ny, self.nz)
    def face_shape(self, axis: int) -> Tuple[int, int, int]:
        s = [self.nx, self.ny, self.nz]; s[axis] += 1
        return tuple(s)


# =============================================================================
# MAC GRID — [B2] correct face shapes
# =============================================================================

class MACGrid:
    """
    Marker-and-Cell staggered layout with CORRECT face shapes.

    [B2] grad_cc_to_faces now returns (nx+1,ny,nz) etc., not (nx,ny,nz).
    """
    def __init__(self, spec: GridSpec):
        self.s = spec
        self.dev, self.dt_ = spec.torch_device, spec.torch_dtype

    # ---- allocators ----
    def cc(self, value: float = 0.0) -> Tensor:
        return torch.full(self.s.cc_shape, value, device=self.dev, dtype=self.dt_)

    def face(self, axis: int, value: float = 0.0) -> Tensor:
        return torch.full(self.s.face_shape(axis), value,
                          device=self.dev, dtype=self.dt_)

    def zeros_u(self): return self.face(0)
    def zeros_v(self): return self.face(1)
    def zeros_w(self): return self.face(2)

    # ---- padding ----
    def _pad1(self, f: Tensor, axis: int, antisym: bool = False) -> Tensor:
        n = f.shape[axis]
        b = self.s.bc[axis]
        if b == "periodic":
            lo = f.index_select(axis, torch.tensor([n - 1], device=f.device))
            hi = f.index_select(axis, torch.tensor([0], device=f.device))
        else:
            lo = f.index_select(axis, torch.tensor([0], device=f.device))
            hi = f.index_select(axis, torch.tensor([n - 1], device=f.device))
            if antisym:
                lo, hi = -lo, -hi
        return torch.cat([lo, f, hi], dim=axis)

    def pad_cc(self, f: Tensor, antisym: bool = False) -> Tensor:
        g = f
        for ax in range(3):
            g = self._pad1(g, ax, antisym)
        return g

    # ---- [B2] gradient: cell-centred scalar -> FACE vector ----
    def grad_cc_to_faces(self, p: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pp = self.pad_cc(p)                       # (nx+2, ny+2, nz+2)
        gx = (pp[1:, 1:-1, 1:-1] - pp[:-1, 1:-1, 1:-1]) / self.s.dx
        gy = (pp[1:-1, 1:, 1:-1] - pp[1:-1, :-1, 1:-1]) / self.s.dy
        gz = (pp[1:-1, 1:-1, 1:] - pp[1:-1, 1:-1, :-1]) / self.s.dz
        assert gx.shape == self.s.face_shape(0), f"gx {gx.shape}"
        assert gy.shape == self.s.face_shape(1), f"gy {gy.shape}"
        assert gz.shape == self.s.face_shape(2), f"gz {gz.shape}"
        return gx, gy, gz

    # ---- divergence: FACE vector -> cell-centred scalar ----
    def divergence(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        for ax, t in enumerate((u, v, w)):
            if t.shape != self.s.face_shape(ax):
                raise ValueError(f"component {ax} shape {t.shape} != "
                                 f"{self.s.face_shape(ax)}")
        return ((u[1:] - u[:-1]) / self.s.dx
                + (v[:, 1:] - v[:, :-1]) / self.s.dy
                + (w[:, :, 1:] - w[:, :, :-1]) / self.s.dz)

    # ---- interpolation ----
    def u_to_cc(self, u): return 0.5 * (u[:-1] + u[1:])
    def v_to_cc(self, v): return 0.5 * (v[:, :-1] + v[:, 1:])
    def w_to_cc(self, w): return 0.5 * (w[:, :, :-1] + w[:, :, 1:])

    def cc_to_faces(self, f: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Arithmetic average of a cell-centred scalar onto faces."""
        fx = self.cc(); fy = self.cc(); fz = self.cc()
        gx = torch.zeros(self.s.face_shape(0), device=f.device, dtype=f.dtype)
        gy = torch.zeros(self.s.face_shape(1), device=f.device, dtype=f.dtype)
        gz = torch.zeros(self.s.face_shape(2), device=f.device, dtype=f.dtype)
        gx[1:-1] = 0.5 * (f[:-1] + f[1:]); gx[0] = f[0];  gx[-1] = f[-1]
        gy[:, 1:-1] = 0.5 * (f[:, :-1] + f[:, 1:]); gy[:, 0] = f[:, 0]; gy[:, -1] = f[:, -1]
        gz[:, :, 1:-1] = 0.5 * (f[:, :, :-1] + f[:, :, 1:])
        gz[:, :, 0] = f[:, :, 0]; gz[:, :, -1] = f[:, :, -1]
        return gx, gy, gz

    # ---- velocity BC ----
    def enforce_velocity_bc(self, u, v, w):
        u, v, w = u.clone(), v.clone(), w.clone()
        for ax, comp in enumerate((u, v, w)):
            b = self.s.bc[ax]
            i0 = torch.tensor([0], device=comp.device)
            i1 = torch.tensor([comp.shape[ax] - 1], device=comp.device)
            if b == "periodic":
                mean = 0.5 * (comp.index_select(ax, i0) + comp.index_select(ax, i1))
                comp.index_copy_(ax, i0, mean)
                comp.index_copy_(ax, i1, mean)
            else:  # wall / slip: impermeable normal face
                z = torch.zeros_like(comp.index_select(ax, i0))
                comp.index_copy_(ax, i0, z)
                comp.index_copy_(ax, i1, z)
        return u, v, w

    # ---- [B3] strain rate at CELL CENTRES ----
    def strain_rate_cc(self, u, v, w):
        """
        All nine gradient components evaluated at cell centres.

        [B3] The original mixed face-located derivatives on different axes,
        which cannot broadcast. Here every derivative is formed on its own
        staggered location and then interpolated to centres.
        """
        s = self.s
        up = self.pad_cc(self.u_to_cc(u))
        vp = self.pad_cc(self.v_to_cc(v))
        wp = self.pad_cc(self.w_to_cc(w))

        def d(f, ax, h):
            sl_p = [slice(1, -1)] * 3; sl_m = [slice(1, -1)] * 3
            sl_p[ax] = slice(2, None); sl_m[ax] = slice(0, -2)
            return (f[tuple(sl_p)] - f[tuple(sl_m)]) / (2 * h)

        dudx, dudy, dudz = d(up, 0, s.dx), d(up, 1, s.dy), d(up, 2, s.dz)
        dvdx, dvdy, dvdz = d(vp, 0, s.dx), d(vp, 1, s.dy), d(vp, 2, s.dz)
        dwdx, dwdy, dwdz = d(wp, 0, s.dx), d(wp, 1, s.dy), d(wp, 2, s.dz)
        S11, S22, S33 = dudx, dvdy, dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2
                                + 2 * (S12**2 + S13**2 + S23**2)) + 1e-30)
        return mag, (S11, S22, S33, S12, S13, S23)


# =============================================================================
# [B1] CORRECT VARIABLE-DENSITY PRESSURE POISSON
# =============================================================================

def face_beta(rho: Tensor, spec: GridSpec) -> Tuple[Tensor, Tensor, Tensor]:
    """
    beta = 1/rho at faces, as the ARITHMETIC MEAN OF THE RECIPROCALS.

        beta_x[i] = 0.5 * (1/rho[i-1] + 1/rho[i])

    [B1] The two source modules returned 2ab/(a+b) (harmonic mean of rho)
    and used it directly as beta. The required value is (a+b)/(2ab).
    Those are reciprocals, so the projection was inverted.
    """
    r = rho.clamp_min(1e-30)
    inv = 1.0 / r
    bx = torch.zeros(spec.face_shape(0), device=rho.device, dtype=rho.dtype)
    by = torch.zeros(spec.face_shape(1), device=rho.device, dtype=rho.dtype)
    bz = torch.zeros(spec.face_shape(2), device=rho.device, dtype=rho.dtype)
    bx[1:-1] = 0.5 * (inv[:-1] + inv[1:])
    by[:, 1:-1] = 0.5 * (inv[:, :-1] + inv[:, 1:])
    bz[:, :, 1:-1] = 0.5 * (inv[:, :, :-1] + inv[:, :, 1:])
    # domain ends: one-sided (Neumann wall gives zero flux anyway)
    bx[0], bx[-1] = inv[0], inv[-1]
    by[:, 0], by[:, -1] = inv[:, 0], inv[:, -1]
    bz[:, :, 0], bz[:, :, -1] = inv[:, :, 0], inv[:, :, -1]
    return bx, by, bz


class PressurePoisson:
    """
    Solve div(beta grad p) = rhs with Jacobi-preconditioned CG.

    Null space (constant p) projected out every iteration.
    Convergence is CHECKED, not assumed — an iteration cap is not a proof.
    """
    def __init__(self, spec: GridSpec, tol: float = 1e-10,
                 max_iter: int = 2000):
        self.s = spec
        self.g = MACGrid(spec)
        self.tol = tol
        self.max_iter = max_iter
        self.last_iters = 0
        self.last_residual = float("nan")
        self.converged = False

    def apply(self, p, bx, by, bz) -> Tensor:
        gx, gy, gz = self.g.grad_cc_to_faces(p)
        fx, fy, fz = bx * gx, by * gy, bz * gz
        return ((fx[1:] - fx[:-1]) / self.s.dx
                + (fy[:, 1:] - fy[:, :-1]) / self.s.dy
                + (fz[:, :, 1:] - fz[:, :, :-1]) / self.s.dz)

    def solve(self, rhs: Tensor, rho: Tensor,
              p_init: Optional[Tensor] = None) -> Tensor:
        s = self.s
        bx, by, bz = face_beta(rho, s)
        rhs = rhs - rhs.mean()
        p = torch.zeros_like(rhs) if p_init is None else (p_init - p_init.mean())
        diag = ((bx[1:] + bx[:-1]) / s.dx**2
                + (by[:, 1:] + by[:, :-1]) / s.dy**2
                + (bz[:, :, 1:] + bz[:, :, :-1]) / s.dz**2).clamp_min(1e-30)
        r = rhs - self.apply(p, bx, by, bz); r = r - r.mean()
        z = r / diag
        d = z.clone()
        rz = (r * z).sum()
        nrm = rhs.norm().clamp_min(1e-30)
        self.converged = False
        for it in range(self.max_iter):
            Ad = self.apply(d, bx, by, bz)
            dAd = (d * Ad).sum()
            if float(dAd.abs()) < 1e-300:
                break
            alpha = rz / dAd
            p = p + alpha * d
            r = r - alpha * Ad; r = r - r.mean()
            res = float(r.norm() / nrm)
            if res < self.tol:
                self.last_iters, self.last_residual = it + 1, res
                self.converged = True
                return p - p.mean()
            z = r / diag
            rz_new = (r * z).sum()
            d = z + (rz_new / rz.clamp_min(1e-300)) * d
            rz = rz_new
        self.last_iters = self.max_iter
        self.last_residual = float(r.norm() / nrm)
        log.warning(f"[Poisson] NOT converged: rel-res={self.last_residual:.3e} "
                    f"after {self.max_iter} iters. An iteration cap is not "
                    f"a convergence guarantee.")
        return p - p.mean()


def project(u, v, w, rho, spec: GridSpec, dt: float,
            solver: Optional[PressurePoisson] = None):
    """Variable-density fractional-step projection with the CORRECT beta."""
    g = MACGrid(spec)
    ps = solver or PressurePoisson(spec)
    rhs = g.divergence(u, v, w) / dt
    p = ps.solve(rhs, rho)
    bx, by, bz = face_beta(rho, spec)
    px, py, pz = g.grad_cc_to_faces(p)
    un = u - dt * bx * px
    vn = v - dt * by * py
    wn = w - dt * bz * pz
    un, vn, wn = g.enforce_velocity_bc(un, vn, wn)
    return un, vn, wn, p, ps


# =============================================================================
# [B4] MASS-CONSERVING FACE-FLUX ADVECTION
# =============================================================================

def upwind_face_flux(alpha: Tensor, vel_face: Tensor, axis: int,
                     spec: GridSpec) -> Tensor:
    """
    Face flux F_f = alpha_upwind * u_f, on the FACE shape.

    [B4] The flow_vof module formed a cell-centred flux and then took a
    forward difference of it, which is not a conservative divergence.
    Here the flux lives on faces and the update is (F_{i+1/2}-F_{i-1/2})/h.
    """
    n = alpha.shape[axis]
    idx_lo = (torch.arange(n + 1, device=alpha.device) - 1).clamp(0, n - 1)
    idx_hi = torch.arange(n + 1, device=alpha.device).clamp(0, n - 1)
    a_lo = alpha.index_select(axis, idx_lo)
    a_hi = alpha.index_select(axis, idx_hi)
    a_up = torch.where(vel_face > 0, a_lo, a_hi)
    flux = a_up * vel_face
    # impermeable ends
    if spec.bc[axis] != "periodic":
        sl0 = [slice(None)] * 3; sl0[axis] = 0
        sl1 = [slice(None)] * 3; sl1[axis] = n
        flux = flux.clone()
        flux[tuple(sl0)] = 0.0
        flux[tuple(sl1)] = 0.0
    return flux


def advect_alpha(alpha: Tensor, u, v, w, dt: float, spec: GridSpec) -> Tensor:
    """Conservative upwind VOF transport. NOT geometric PLIC."""
    out = alpha
    for ax, (vel, h) in enumerate(((u, spec.dx), (v, spec.dy), (w, spec.dz))):
        flux = upwind_face_flux(out, vel, ax, spec)
        sl_hi = [slice(None)] * 3; sl_lo = [slice(None)] * 3
        sl_hi[ax] = slice(1, None); sl_lo[ax] = slice(0, -1)
        out = out - dt * (flux[tuple(sl_hi)] - flux[tuple(sl_lo)]) / h
    return out.clamp(0.0, 1.0)


# =============================================================================
# STATE CONTRACT
# =============================================================================

@dataclass
class CFDState:
    alpha: Tensor
    u: Tensor
    v: Tensor
    w: Tensor
    p: Tensor
    rho: Optional[Tensor] = None
    mu: Optional[Tensor] = None
    T: Optional[Tensor] = None
    solid: Optional[Tensor] = None
    t: float = 0.0
    step: int = 0

    def validate(self, spec: GridSpec) -> List[str]:
        errs = []
        cc = spec.cc_shape
        for name in ("alpha", "p"):
            t = getattr(self, name)
            if t is not None and tuple(t.shape) != cc:
                errs.append(f"{name} shape {tuple(t.shape)} != {cc}")
        for ax, name in enumerate(("u", "v", "w")):
            t = getattr(self, name)
            if tuple(t.shape) != spec.face_shape(ax):
                errs.append(f"{name} shape {tuple(t.shape)} != {spec.face_shape(ax)}")
        if self.rho is not None and tuple(self.rho.shape) != cc:
            errs.append("rho must be cell-centred")
        if self.mu is not None and tuple(self.mu.shape) != cc:
            errs.append("mu must be cell-centred")
        if self.solid is not None and tuple(self.solid.shape) != cc:
            errs.append("solid must be cell-centred")
        return errs


def mix_properties(alpha: Tensor, rho1: float, rho2: float,
                   mu1: float, mu2: float) -> Tuple[Tensor, Tensor]:
    """Volume-weighted density; harmonic viscosity (interface-stable)."""
    a = alpha.clamp(0.0, 1.0)
    rho = a * rho1 + (1 - a) * rho2
    mu = 1.0 / (a / max(mu1, 1e-30) + (1 - a) / max(mu2, 1e-30)).clamp_min(1e-30)
    return rho, mu


# =============================================================================
# BRIDGES
# =============================================================================

class CHToCompressible:
    """
    Cahn-Hilliard -> compressible DNS: Korteweg body force + property blend.

    f = -k * rho_eff * grad(mu_R)   (cell-centred, then averaged to faces
    if the target solver stores momentum at centres — super_dns stores
    cell-centred rhou, so we return cell-centred forces).
    """
    def __init__(self, korteweg_strength: float = 0.0,
                 rho_A: float = 1.0, rho_B: float = 2.0):
        self.k = korteweg_strength
        self.rho_A, self.rho_B = rho_A, rho_B

    def phi(self, u_ch: Tensor) -> Tensor:
        return 0.5 * (u_ch + 1.0)

    def rho_eff(self, u_ch: Tensor) -> Tensor:
        return self.rho_B + (self.rho_A - self.rho_B) * self.phi(u_ch)

    def force(self, u_ch: Tensor, mu_R: Tensor, spec: GridSpec
              ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.k == 0.0:
            z = torch.zeros_like(u_ch)
            return z, z, z
        g = MACGrid(spec)
        rho_e = self.rho_eff(u_ch)
        pp = g.pad_cc(mu_R)
        mx = (pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) / (2 * spec.dx)
        my = (pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) / (2 * spec.dy)
        mz = (pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) / (2 * spec.dz)
        return -self.k * rho_e * mx, -self.k * rho_e * my, -self.k * rho_e * mz

    def push(self, dns_solver: Any, u_ch: Tensor, mu_R: Tensor,
             spec: GridSpec) -> Dict[str, Any]:
        """Write into super_dns buffers if present; report if absent."""
        fx, fy, fz = self.force(u_ch, mu_R, spec)
        rho_e = self.rho_eff(u_ch)
        written = []
        for name, val in (("_ext_rho_ch", rho_e), ("_ext_fx", fx),
                          ("_ext_fy", fy), ("_ext_fz", fz)):
            if hasattr(dns_solver, name):
                buf = getattr(dns_solver, name)
                if tuple(buf.shape) == tuple(val.shape):
                    setattr(dns_solver, name, val.to(buf.device, buf.dtype))
                    written.append(name)
        return {"written": written,
                "available": len(written) > 0,
                "NOTE": ("super_dns_one_v6_3 requires one_core.py, which is "
                         "absent from the corpus. Bridge is inert without it.")}


class VOFToCompressible:
    """VOF alpha -> density/viscosity fields for the compressible solver."""
    def __init__(self, rho1: float, rho2: float, mu1: float, mu2: float):
        self.rho1, self.rho2, self.mu1, self.mu2 = rho1, rho2, mu1, mu2

    def push(self, dns_solver: Any, alpha: Tensor) -> Dict[str, Any]:
        rho, mu = mix_properties(alpha, self.rho1, self.rho2, self.mu1, self.mu2)
        written = []
        for name, val in (("_ext_rho_ch", rho), ("_ext_nu_ch", mu)):
            if hasattr(dns_solver, name):
                buf = getattr(dns_solver, name)
                if tuple(buf.shape) == tuple(val.shape):
                    setattr(dns_solver, name, val.to(buf.device, buf.dtype))
                    written.append(name)
        return {"written": written, "available": len(written) > 0}


def ch_vof_consistency(u_ch: Tensor, alpha: Tensor) -> Dict[str, float]:
    """
    Compare phase-field and sharp-interface representations.

    phi = (u_ch+1)/2 should approximate alpha. They are DIFFERENT
    discretisations; agreement is a diagnostic, not an identity.
    """
    phi = 0.5 * (u_ch + 1.0)
    d = (phi - alpha).abs()
    iface = (alpha > 1e-6) & (alpha < 1 - 1e-6)
    return {"l1": float(d.mean()), "linf": float(d.max()),
            "l1_interface": float(d[iface].mean()) if bool(iface.any()) else 0.0,
            "n_interface_cells": int(iface.sum())}


def coords_to_solid_mask(coords_angstrom: Tensor, spec: GridSpec,
                         radius_angstrom: float = 3.5,
                         scale: float = 1e-10) -> Tensor:
    """
    Rasterise Cα coordinates (Angstrom) into a cell-centred solid mask.

    UNIT CONTRACT: coords are Angstrom, `scale` converts to metres.
    Staircased at grid scale — not a signed-distance representation.
    """
    dev, dt_ = spec.torch_device, spec.torch_dtype
    xyz = coords_angstrom.to(dev=dev, dtype=torch.float64) * scale
    rad = float(radius_angstrom * scale)
    ctr = 0.5 * torch.tensor([spec.Lx, spec.Ly, spec.Lz],
                             device=dev, dtype=torch.float64)
    xyz = xyz - xyz.mean(0) + ctr
    cx = (torch.arange(spec.nx, device=dev, dtype=torch.float64) + 0.5) * spec.dx
    cy = (torch.arange(spec.ny, device=dev, dtype=torch.float64) + 0.5) * spec.dy
    cz = (torch.arange(spec.nz, device=dev, dtype=torch.float64) + 0.5) * spec.dz
    X, Y, Z = torch.meshgrid(cx, cy, cz, indexing="ij")
    pts = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    mask = torch.zeros(pts.shape[0], device=dev, dtype=torch.float64)
    CH = 2048
    for s in range(0, xyz.shape[0], CH):
        p = xyz[s:s + CH]
        d2 = ((pts.unsqueeze(0) - p.unsqueeze(1)) ** 2).sum(-1)
        mask[(d2 <= rad * rad).any(0)] = 1.0
    return mask.reshape(spec.cc_shape).to(dt_)


# =============================================================================
# SELF-TEST — verifies the FIXES, not just the plumbing
# =============================================================================

def self_test() -> bool:
    print("=" * 78)
    print(f"  ONE CFD CORE v{CORE_VERSION} — bug-fix verification")
    print("=" * 78)
    torch.manual_seed(0)
    ok = True
    N = 16
    spec = GridSpec(nx=N, ny=N, nz=N, Lx=1.0, Ly=1.0, Lz=1.0,
                    bc=("periodic",) * 3, dtype="float64")
    g = MACGrid(spec)

    print("\n[T1] [B2] face shapes are correct")
    p = torch.rand(spec.cc_shape, dtype=torch.float64)
    gx, gy, gz = g.grad_cc_to_faces(p)
    t = (gx.shape == (N + 1, N, N) and gy.shape == (N, N + 1, N)
         and gz.shape == (N, N, N + 1))
    print(f"   gx{tuple(gx.shape)} gy{tuple(gy.shape)} gz{tuple(gz.shape)}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T2] [B1] face_beta equals the mean of reciprocals")
    rho = torch.rand(spec.cc_shape, dtype=torch.float64) * 999 + 1
    bx, by, bz = face_beta(rho, spec)
    ref = 0.5 * (1.0 / rho[:-1] + 1.0 / rho[1:])
    err = float((bx[1:-1] - ref).abs().max())
    wrong = 2.0 / (1.0 / rho[:-1] + 1.0 / rho[1:])       # the old value
    sep = float((bx[1:-1] - wrong).abs().max())
    t = err < 1e-14 and sep > 1e-6
    print(f"   |beta - mean(1/rho)| = {err:.2e}  (expect ~0)")
    print(f"   |beta - old_harmonic | = {sep:.2e}  (expect large)")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T3] div(grad) matches the analytic Laplacian (periodic)")
    k = 2 * math.pi
    cx = (torch.arange(N, dtype=torch.float64) + 0.5) * spec.dx
    X, Y, Z = torch.meshgrid(cx, cx, cx, indexing="ij")
    f = torch.cos(k * X) * torch.cos(k * Y) * torch.cos(k * Z)
    bx1 = torch.ones(spec.face_shape(0), dtype=torch.float64)
    by1 = torch.ones(spec.face_shape(1), dtype=torch.float64)
    bz1 = torch.ones(spec.face_shape(2), dtype=torch.float64)
    ps = PressurePoisson(spec, tol=1e-12, max_iter=4000)
    num = ps.apply(f, bx1, by1, bz1)
    lam = (-4 / spec.dx**2 * math.sin(k * spec.dx / 2)**2) * 3
    ana = lam * f
    rel = float((num - ana).norm() / ana.norm())
    t = rel < 1e-10
    print(f"   relative error = {rel:.2e}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T4] variable-density Poisson residual (1000:1)")
    a = torch.zeros(spec.cc_shape, dtype=torch.float64); a[:N // 2] = 1.0
    rho2 = mix_properties(a, 1000.0, 1.0, 1e-3, 1.8e-5)[0]
    rhs = torch.rand(spec.cc_shape, dtype=torch.float64); rhs -= rhs.mean()
    pp = PressurePoisson(spec, tol=1e-11, max_iter=6000)
    sol = pp.solve(rhs, rho2)
    bb = face_beta(rho2, spec)
    res = float((pp.apply(sol, *bb) - rhs).norm() / rhs.norm())
    t = res < 1e-9 and pp.converged
    print(f"   rel-res = {res:.2e}  iters={pp.last_iters} converged={pp.converged}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T5] projection makes velocity divergence-free")
    u = torch.rand(spec.face_shape(0), dtype=torch.float64) * 0.1
    v = torch.rand(spec.face_shape(1), dtype=torch.float64) * 0.1
    w = torch.rand(spec.face_shape(2), dtype=torch.float64) * 0.1
    u, v, w = g.enforce_velocity_bc(u, v, w)
    d0 = float(g.divergence(u, v, w).abs().max())
    un, vn, wn, pn, psx = project(u, v, w, rho2, spec, dt=1e-3)
    d1 = float(g.divergence(un, vn, wn).abs().max())
    t = d1 < 1e-9
    print(f"   max|div| {d0:.3e} -> {d1:.3e}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T6] [B4] advection conserves mass exactly")
    al = torch.zeros(spec.cc_shape, dtype=torch.float64)
    al[4:10, 4:10, 4:10] = 1.0
    uu = g.zeros_u() + 0.7; vv = g.zeros_v() + 0.3; ww = g.zeros_w() - 0.2
    m0 = float(al.sum())
    aa = al
    for _ in range(20):
        aa = advect_alpha(aa, uu, vv, ww, dt=0.2 * spec.dx, spec=spec)
    m1 = float(aa.sum())
    rel_m = abs(m1 - m0) / max(m0, 1e-30)
    t = rel_m < 1e-12 and float(aa.min()) >= 0 and float(aa.max()) <= 1
    print(f"   mass {m0:.6f} -> {m1:.6f}  rel={rel_m:.2e}")
    print(f"   alpha in [{float(aa.min()):.3f}, {float(aa.max()):.3f}]")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T7] [B3] strain rate is cell-centred and finite")
    mag, comps = g.strain_rate_cc(un, vn, wn)
    t = (tuple(mag.shape) == spec.cc_shape
         and all(tuple(c.shape) == spec.cc_shape for c in comps)
         and bool(torch.isfinite(mag).all()))
    print(f"   |S| shape {tuple(mag.shape)}  finite={bool(torch.isfinite(mag).all())}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T8] [B5] soft_floor width tracks the requested floor")
    x = torch.zeros(1, dtype=torch.float64)
    f_small = float(soft_floor(x, 1e-8))
    f_large = float(soft_floor(x, 1e-1))
    t = f_small < 1e-6 and f_large > 1e-3
    print(f"   floor 1e-8 -> {f_small:.3e}   floor 1e-1 -> {f_large:.3e}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T9] autograd through projection")
    rho_g = rho2.clone().requires_grad_(True)
    uq = torch.rand(spec.face_shape(0), dtype=torch.float64, requires_grad=True)
    vq = g.zeros_v(); wq = g.zeros_w()
    u2, v2, w2, p2, _ = project(uq, vq, wq, rho_g, spec, dt=1e-3)
    p2.pow(2).sum().backward()
    t = (uq.grad is not None and torch.isfinite(uq.grad).all()
         and rho_g.grad is not None and torch.isfinite(rho_g.grad).all())
    print(f"   grads finite: u={uq.grad is not None} rho={rho_g.grad is not None}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= bool(t)

    print("\n[T10] bridges report absence honestly")
    class _Stub: pass
    stub = _Stub()
    ch = CHToCompressible(korteweg_strength=0.1)
    r = ch.push(stub, torch.zeros(spec.cc_shape, dtype=torch.float64),
                torch.zeros(spec.cc_shape, dtype=torch.float64), spec)
    t = (r["available"] is False and len(r["written"]) == 0)
    print(f"   no one_core -> available={r['available']}  "
          f"{'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T11] Angstrom -> metre unit contract")
    ca = torch.randn(200, 3, dtype=torch.float64) * 20.0
    big = GridSpec(nx=16, ny=16, nz=16, Lx=1e-8, Ly=1e-8, Lz=1e-8,
                   dtype="float64")
    mask = coords_to_solid_mask(ca, big, radius_angstrom=3.0)
    frac = float(mask.mean())
    t = 0.0 < frac < 1.0 and tuple(mask.shape) == big.cc_shape
    print(f"   solid fraction = {frac:.4f}  shape={tuple(mask.shape)}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL FIXES VERIFIED' if ok else '❌ SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT IS NOW CORRECT
    • face shapes, face beta = mean(1/rho), divergence, projection
    • cell-centred strain rate, mass-conserving face-flux advection
    • soft_floor width, unit contract, graceful bridge degradation

  WHAT IS STILL NOT DONE
    • Geometric PLIC advection (this is upwind, more diffusive)
    • Balanced-force surface tension (spurious currents remain)
    • super_dns_one_v6_3 needs one_core.py — absent from the corpus
    • No validation against experiment or grid-convergence study
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "contract"])
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if self_test() else 1)
    print(json.dumps({
        "units": {"length": "m", "mass": "kg", "time": "s",
                  "coords_input": "Angstrom (scale=1e-10)"},
        "shapes": {"cc": "(nx,ny,nz)", "u": "(nx+1,ny,nz)",
                   "v": "(nx,ny+1,nz)", "w": "(nx,ny,nz+1)"},
        "beta": "0.5*(1/rho_left + 1/rho_right) at faces",
        "ch_to_vof": "phi = (u_ch+1)/2",
    }, indent=2))


if __name__ == "__main__":
    main()
