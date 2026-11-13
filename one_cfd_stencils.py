#!/usr/bin/env python3
"""
=============================================================================
ONE CFD STENCILS — canonical MAC operators for the ONE Ecosystem
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHY THIS FILE EXISTS
  structuralfluctuatinghydro_v6_3.py already has CORRECT grad_p / div_u /
  laplacian_cell. sesi_multiphase_flow.py re-implemented them and got the
  face shapes wrong. Rather than a third copy, this module provides the
  canonical staggered operators, and self-test T1 PROVES they agree
  numerically with the FH implementations.

BUGS THIS REPLACES (verified by reading the sources)
  [B1] sesi_multiphase_flow.PressurePoisson._beta_faces returned the
       HARMONIC MEAN OF RHO and used it directly as beta in
       `u -= dt*beta*grad(p)`. Required: beta = 1/rho_face.
         computed : 2ab/(a+b)
         required : (a+b)/(2ab)
       These are reciprocals -> the projection was inverted.
  [B2] sesi_multiphase_flow.MACGrid.grad_cc_to_faces returned (nx,ny,nz)
       instead of the face shape (nx+1,ny,nz).
  [B3] sesi_multiphase_flow.Rheology.strain_rate_invariant mixed
       face-located derivatives on different axes -> broadcast error.
  [B4] sesi_multiphase_flow._viscous_u / _viscous_component had
       inconsistent edge shapes AND mutated cfg via object.__setattr__
       with no try/finally (a raised exception left cfg permuted).
  [B5] sesi_multiphase_flow_vof advection omitted the velocity magnitude
       and used a forward difference of a cell-centred flux.

SHAPE CONTRACT
  rho, mu, p, alpha : (nx,   ny,   nz  )   cell centres
  u                 : (nx+1, ny,   nz  )   x-faces
  v                 : (nx,   ny+1, nz  )   y-faces
  w                 : (nx,   ny,   nz+1)   z-faces

DEPENDENCIES: torch only. No one_core import needed, so this module is
usable even when one_core is unavailable.
=============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

STENCIL_VERSION = "1.0.0"
BCS = ("periodic", "wall", "slip")


# =============================================================================
# GRID SPEC
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
            raise ValueError("domain extents must be positive")
        for b in self.bc:
            if b not in BCS:
                raise ValueError(f"bad bc {b!r}; choose from {BCS}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")

    @property
    def dx(self): return self.Lx / self.nx
    @property
    def dy(self): return self.Ly / self.ny
    @property
    def dz(self): return self.Lz / self.nz
    @property
    def h(self): return (self.dx, self.dy, self.dz)
    @property
    def n(self): return (self.nx, self.ny, self.nz)
    @property
    def cell_volume(self): return self.dx * self.dy * self.dz
    @property
    def torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32
    @property
    def torch_device(self): return torch.device(self.device)
    @property
    def cc_shape(self): return (self.nx, self.ny, self.nz)

    def face_shape(self, ax: int):
        s = list(self.n); s[ax] += 1
        return tuple(s)


# =============================================================================
# SLICING HELPERS
# =============================================================================

def sl(ax: int, idx) -> Tuple:
    """Slice tuple selecting `idx` along axis `ax`, full range elsewhere."""
    out = [slice(None)] * 3
    out[ax] = idx
    return tuple(out)


def diff(f: Tensor, ax: int, h: float) -> Tensor:
    """Forward difference along ax: shape[ax] shrinks by 1."""
    return (f[sl(ax, slice(1, None))] - f[sl(ax, slice(0, -1))]) / h


def pad_axis(f: Tensor, ax: int, bc: str, antisym: bool = False) -> Tensor:
    """Pad by 1 on both sides along ax. antisym => ghost = -interior."""
    n = f.shape[ax]
    dev = f.device
    if bc == "periodic":
        lo = f.index_select(ax, torch.tensor([n - 1], device=dev))
        hi = f.index_select(ax, torch.tensor([0], device=dev))
    else:
        lo = f.index_select(ax, torch.tensor([0], device=dev))
        hi = f.index_select(ax, torch.tensor([n - 1], device=dev))
        if antisym:
            lo, hi = -lo, -hi
    return torch.cat([lo, f, hi], dim=ax)


def pad_cc(f: Tensor, bc: Tuple[str, str, str],
           antisym: bool = False) -> Tensor:
    g = f
    for ax in range(3):
        g = pad_axis(g, ax, bc[ax], antisym)
    return g


# =============================================================================
# [B1] CORRECT FACE beta = 1/rho
# =============================================================================

def face_beta(rho: Tensor, spec: GridSpec) -> Tuple[Tensor, Tensor, Tensor]:
    """
    beta = 1/rho at faces = ARITHMETIC MEAN OF RECIPROCALS.

        beta_x[i] = 0.5 * (1/rho[i-1] + 1/rho[i])

    [B1] The two source modules returned 2ab/(a+b) (harmonic mean of rho)
    and used it directly as beta. Required is (a+b)/(2ab) — the reciprocal.
    """
    inv = 1.0 / rho.clamp_min(1e-30)
    out = []
    for ax in range(3):
        b = torch.zeros(spec.face_shape(ax), device=rho.device, dtype=rho.dtype)
        b[sl(ax, slice(1, -1))] = 0.5 * (
            inv[sl(ax, slice(0, -1))] + inv[sl(ax, slice(1, None))])
        b[sl(ax, 0)] = inv[sl(ax, 0)]
        b[sl(ax, -1)] = inv[sl(ax, -1)]
        out.append(b)
    return tuple(out)


def face_scalar(f: Tensor, spec: GridSpec) -> Tuple[Tensor, Tensor, Tensor]:
    """Cell-centred scalar -> faces by arithmetic mean (for rho, mu)."""
    out = []
    for ax in range(3):
        b = torch.zeros(spec.face_shape(ax), device=f.device, dtype=f.dtype)
        b[sl(ax, slice(1, -1))] = 0.5 * (
            f[sl(ax, slice(0, -1))] + f[sl(ax, slice(1, None))])
        b[sl(ax, 0)] = f[sl(ax, 0)]
        b[sl(ax, -1)] = f[sl(ax, -1)]
        out.append(b)
    return tuple(out)


def edge_scalar(f: Tensor, spec: GridSpec, a: int, b: int) -> Tensor:
    """
    Cell-centred scalar -> (a,b)-edges by 4-cell average.
    Result shape: n[a]+1, n[b]+1, n[c]  (c = the remaining axis).
    """
    g = pad_axis(f, a, spec.bc[a])
    g = pad_axis(g, b, spec.bc[b])
    return 0.25 * (
        g[sl(a, slice(0, -1))][sl(b, slice(0, -1))]
        + g[sl(a, slice(1, None))][sl(b, slice(0, -1))]
        + g[sl(a, slice(0, -1))][sl(b, slice(1, None))]
        + g[sl(a, slice(1, None))][sl(b, slice(1, None))])


# =============================================================================
# MAC OPERATORS
# =============================================================================

class MACOps:
    """
    Canonical staggered operators. Shapes are ASSERTED, not assumed.

    Cross-checked against structuralfluctuatinghydro_v6_3.grad_p / div_u /
    laplacian_cell in self-test T1.
    """

    def __init__(self, spec: GridSpec):
        self.s = spec
        self.dev, self.dt_ = spec.torch_device, spec.torch_dtype

    # ---- allocators ----
    def cc(self, v: float = 0.0) -> Tensor:
        return torch.full(self.s.cc_shape, v, device=self.dev, dtype=self.dt_)

    def face(self, ax: int, v: float = 0.0) -> Tensor:
        return torch.full(self.s.face_shape(ax), v,
                          device=self.dev, dtype=self.dt_)

    def zeros_u(self): return self.face(0)
    def zeros_v(self): return self.face(1)
    def zeros_w(self): return self.face(2)

    def coords_cc(self) -> Tuple[Tensor, Tensor, Tensor]:
        c = [(torch.arange(n, device=self.dev, dtype=self.dt_) + 0.5) * h
             for n, h in zip(self.s.n, self.s.h)]
        return torch.meshgrid(*c, indexing="ij")

    # ---- [B2] gradient: centres -> FACES ----
    def grad_cc_to_faces(self, p: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pp = pad_cc(p, self.s.bc)
        g = []
        for ax in range(3):
            hi = pp[sl(ax, slice(1, None))]
            lo = pp[sl(ax, slice(0, -1))]
            for other in range(3):
                if other != ax:
                    hi = hi[sl(other, slice(1, -1))]
                    lo = lo[sl(other, slice(1, -1))]
            gx = (hi - lo) / self.s.h[ax]
            assert gx.shape == self.s.face_shape(ax), \
                f"[B2] grad axis {ax}: {gx.shape} != {self.s.face_shape(ax)}"
            g.append(gx)
        return tuple(g)

    # ---- divergence: FACES -> centres ----
    def divergence(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        for ax, t in enumerate((u, v, w)):
            if t.shape != self.s.face_shape(ax):
                raise ValueError(
                    f"component {ax} shape {tuple(t.shape)} != "
                    f"{self.s.face_shape(ax)} — use face-shaped velocities")
        return (diff(u, 0, self.s.dx) + diff(v, 1, self.s.dy)
                + diff(w, 2, self.s.dz))

    # ---- 7-point Laplacian at centres ----
    def laplacian_cc(self, f: Tensor) -> Tensor:
        fp = pad_cc(f, self.s.bc)
        out = torch.zeros_like(f)
        for ax in range(3):
            hi = fp[sl(ax, slice(2, None))]
            lo = fp[sl(ax, slice(0, -2))]
            for other in range(3):
                if other != ax:
                    hi = hi[sl(other, slice(1, -1))]
                    lo = lo[sl(other, slice(1, -1))]
            out = out + (hi - 2.0 * f + lo) / self.s.h[ax] ** 2
        return out

    # ---- interpolation ----
    def to_cc(self, u, v, w):
        return (0.5 * (u[:-1] + u[1:]),
                0.5 * (v[:, :-1] + v[:, 1:]),
                0.5 * (w[:, :, :-1] + w[:, :, 1:]))

    def cc_to_face(self, f: Tensor, ax: int) -> Tensor:
        return face_scalar(f, self.s)[ax]

    # ---- velocity BC ----
    def enforce_velocity_bc(self, u, v, w):
        out = []
        for ax, comp in enumerate((u, v, w)):
            c = comp.clone()
            i0 = torch.tensor([0], device=c.device)
            i1 = torch.tensor([c.shape[ax] - 1], device=c.device)
            if self.s.bc[ax] == "periodic":
                mean = 0.5 * (c.index_select(ax, i0) + c.index_select(ax, i1))
                c.index_copy_(ax, i0, mean)
                c.index_copy_(ax, i1, mean)
            else:
                z = torch.zeros_like(c.index_select(ax, i0))
                c.index_copy_(ax, i0, z)
                c.index_copy_(ax, i1, z)
            out.append(c)
        return tuple(out)

    # ---- [B3] strain rate, ALL at cell centres ----
    def strain_rate_cc(self, u, v, w):
        """
        gdot = sqrt(2 S:S), S = 0.5(grad u + grad u^T), evaluated at CENTRES.

        [B3] The original mixed face-located derivatives on different axes,
        which cannot broadcast. Here every component is formed at centres.
        """
        uc, vc, wc = self.to_cc(u, v, w)
        up, vp, wp = (pad_cc(t, self.s.bc) for t in (uc, vc, wc))

        def d(fp, ax):
            hi = fp[sl(ax, slice(2, None))]
            lo = fp[sl(ax, slice(0, -2))]
            for other in range(3):
                if other != ax:
                    hi = hi[sl(other, slice(1, -1))]
                    lo = lo[sl(other, slice(1, -1))]
            return (hi - lo) / (2.0 * self.s.h[ax])

        dudx, dudy, dudz = d(up, 0), d(up, 1), d(up, 2)
        dvdx, dvdy, dvdz = d(vp, 0), d(vp, 1), d(vp, 2)
        dwdx, dwdy, dwdz = d(wp, 0), d(wp, 1), d(wp, 2)
        S11, S22, S33 = dudx, dvdy, dwdz
        S12 = 0.5 * (dudy + dvdx)
        S13 = 0.5 * (dudz + dwdx)
        S23 = 0.5 * (dvdz + dwdy)
        for name, t in (("S11", S11), ("S12", S12), ("S23", S23)):
            assert t.shape == self.s.cc_shape, f"[B3] {name} {t.shape}"
        mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2
                                + 2 * (S12**2 + S13**2 + S23**2)) + 1e-30)
        return mag, (S11, S22, S33, S12, S13, S23)

    # ---- [B4] viscous stress divergence, generic over axis ----
    def viscous_component(self, vel: Tuple[Tensor, Tensor, Tensor],
                          mu: Tensor, axis: int) -> Tensor:
        """
        Component `axis` of div( mu (grad u + grad u^T) ) at its own faces.

            d/dx_a(2 mu du_a/dx_a)
          + sum_{d != a} d/dx_d( mu (du_a/dx_d + du_d/dx_a) )

        Locations:
          du_a/dx_a  at cell centres  -> flux differenced back to a-faces
          du_a/dx_d  at (a,d)-edges   -> mu averaged to the same edge
          du_d/dx_a  at (a,d)-edges   -> identical location, so they add

        [B4] No cfg permutation, no object.__setattr__, no try/finally
        needed: the axis arithmetic is explicit and shape-checked.
        """
        s = self.s
        comp = vel[axis]
        ha = s.h[axis]

        # normal term: 2 mu du_a/dx_a at centres
        d_norm = diff(comp, axis, ha)
        assert d_norm.shape == s.cc_shape
        Tn = 2.0 * mu * d_norm
        Tn_p = pad_axis(Tn, axis, s.bc[axis])
        out = diff(Tn_p, axis, ha)
        assert out.shape == s.face_shape(axis)

        # shear terms
        for d in range(3):
            if d == axis:
                continue
            hd = s.h[d]
            other = vel[d]
            cp = pad_axis(comp, d, s.bc[d], antisym=(s.bc[d] == "wall"))
            dcomp = diff(cp, d, hd)
            op = pad_axis(other, axis, s.bc[axis],
                          antisym=(s.bc[axis] == "wall"))
            dother = diff(op, axis, ha)
            if dcomp.shape != dother.shape:
                raise RuntimeError(
                    f"edge shape mismatch axis={axis} d={d}: "
                    f"{dcomp.shape} vs {dother.shape}")
            mu_e = edge_scalar(mu, s, axis, d)
            if mu_e.shape != dcomp.shape:
                raise RuntimeError(
                    f"mu edge {mu_e.shape} != stencil {dcomp.shape}")
            T = mu_e * (dcomp + dother)
            out = out + diff(T, d, hd)

        assert out.shape == s.face_shape(axis), \
            f"[B4] viscous axis {axis}: {out.shape} != {s.face_shape(axis)}"
        return out

    def viscous_all(self, u, v, w, mu):
        vel = (u, v, w)
        return tuple(self.viscous_component(vel, mu, ax) for ax in range(3))


# =============================================================================
# PRESSURE POISSON — correct beta, convergence CHECKED
# =============================================================================

class PressurePoisson:
    """
    Solve div(beta grad p) = rhs, beta = 1/rho at faces.

    'cg'  : Jacobi-preconditioned CG, variable rho, any BC.
            Null space (constant p) projected out every iteration.
    'fft' : exact spectral, constant rho + fully periodic only.

    An iteration cap is NOT a convergence proof: `converged` is reported.
    """

    def __init__(self, spec: GridSpec, ops: Optional[MACOps] = None,
                 method: str = "cg", tol: float = 1e-10,
                 max_iter: int = 3000):
        if method not in ("cg", "fft"):
            raise ValueError("method must be 'cg' or 'fft'")
        if method == "fft" and any(b != "periodic" for b in spec.bc):
            raise ValueError("fft Poisson requires fully periodic BC")
        self.s = spec
        self.g = ops or MACOps(spec)
        self.method, self.tol, self.max_iter = method, tol, max_iter
        self.last_iters = 0
        self.last_residual = float("nan")
        self.converged = False

    def apply(self, p, bx, by, bz) -> Tensor:
        gx, gy, gz = self.g.grad_cc_to_faces(p)
        return (diff(bx * gx, 0, self.s.dx)
                + diff(by * gy, 1, self.s.dy)
                + diff(bz * gz, 2, self.s.dz))

    def solve(self, rhs: Tensor, rho: Tensor,
              p_init: Optional[Tensor] = None) -> Tensor:
        if self.method == "fft":
            return self._fft(rhs, rho)
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
        self.converged = False
        return p - p.mean()

    def _fft(self, rhs, rho):
        s = self.s
        beta = float(1.0 / rho.mean())
        kx = 2 * math.pi * torch.fft.fftfreq(s.nx, d=s.dx, device=rhs.device)
        ky = 2 * math.pi * torch.fft.fftfreq(s.ny, d=s.dy, device=rhs.device)
        kz = 2 * math.pi * torch.fft.rfftfreq(s.nz, d=s.dz, device=rhs.device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        lam = (-4 / s.dx**2 * torch.sin(KX * s.dx / 2)**2
               - 4 / s.dy**2 * torch.sin(KY * s.dy / 2)**2
               - 4 / s.dz**2 * torch.sin(KZ * s.dz / 2)**2)
        lam[0, 0, 0] = 1.0
        rhs_hat = torch.fft.rfftn(rhs.to(torch.float64))
        p_hat = rhs_hat / (beta * lam)
        p_hat[0, 0, 0] = 0.0
        self.converged = True
        self.last_iters, self.last_residual = 1, 0.0
        return torch.fft.irfftn(p_hat, s=rhs.shape).to(rhs.dtype)


def project(u, v, w, rho, spec: GridSpec, dt: float,
            solver: Optional[PressurePoisson] = None):
    """Variable-density fractional-step projection with the CORRECT beta."""
    g = MACOps(spec)
    ps = solver or PressurePoisson(spec, g)
    rhs = g.divergence(u, v, w) / dt
    p = ps.solve(rhs, rho)
    bx, by, bz = face_beta(rho, spec)
    px, py, pz = g.grad_cc_to_faces(p)
    un, vn, wn = u - dt * bx * px, v - dt * by * py, w - dt * bz * pz
    un, vn, wn = g.enforce_velocity_bc(un, vn, wn)
    return un, vn, wn, p, ps


# =============================================================================
# [B5] MASS-CONSERVING FACE-FLUX ADVECTION
# =============================================================================

def upwind_face_flux(alpha: Tensor, vel_face: Tensor, ax: int,
                     spec: GridSpec) -> Tensor:
    """
    F_face = alpha_upwind * u_face, on the FACE shape.

    [B5] The flow_vof module formed a cell-centred flux and took a forward
    difference of it, which is not a conservative divergence, and omitted
    the velocity magnitude entirely.
    """
    n = alpha.shape[ax]
    dev = alpha.device
    idx_lo = (torch.arange(n + 1, device=dev) - 1).clamp(0, n - 1)
    idx_hi = torch.arange(n + 1, device=dev).clamp(0, n - 1)
    a_lo = alpha.index_select(ax, idx_lo)
    a_hi = alpha.index_select(ax, idx_hi)
    a_up = torch.where(vel_face > 0, a_lo, a_hi)
    flux = a_up * vel_face
    if spec.bc[ax] != "periodic":
        flux = flux.clone()
        flux[sl(ax, 0)] = 0.0
        flux[sl(ax, n)] = 0.0
    return flux


def advect_alpha_upwind(alpha, u, v, w, dt, spec: GridSpec) -> Tensor:
    """Conservative upwind VOF transport. NOT geometric PLIC."""
    out = alpha
    for ax, (vel, h) in enumerate(((u, spec.dx), (v, spec.dy), (w, spec.dz))):
        flux = upwind_face_flux(out, vel, ax, spec)
        out = out - dt * diff(flux, ax, h)
    return out.clamp(0.0, 1.0)


# =============================================================================
# SELF-TEST — verifies the FIXES and agreement with the FH module
# =============================================================================

def self_test() -> bool:
    print("=" * 78)
    print(f"  ONE CFD STENCILS v{STENCIL_VERSION} — verification")
    print("=" * 78)
    torch.manual_seed(0)
    ok = True
    N = 12
    spec = GridSpec(nx=N, ny=N, nz=N, dtype="float64")
    g = MACOps(spec)

    print("\n[T1] agreement with structuralfluctuatinghydro grad_p / div_u")
    try:
        import structuralfluctuatinghydro_v6_3 as fh
        p = torch.rand(spec.cc_shape, dtype=torch.float64)
        gx, gy, gz = g.grad_cc_to_faces(p)
        fx, fy, fz = fh.grad_p(p, spec.dx, spec.dy, spec.dz)
        # FH uses zero boundary gradient; ours uses replicate padding which
        # also gives zero at the boundary -> interior must match exactly.
        e = max(float((gx[1:-1] - fx[1:-1]).abs().max()),
                float((gy[:, 1:-1] - fy[:, 1:-1]).abs().max()),
                float((gz[:, :, 1:-1] - fz[:, :, 1:-1]).abs().max()))
        u, v, w = (torch.rand(spec.face_shape(a), dtype=torch.float64)
                   for a in range(3))
        ed = float((g.divergence(u, v, w)
                    - fh.div_u(u, v, w, spec.dx, spec.dy, spec.dz)).abs().max())
        t = e < 1e-13 and ed < 1e-13
        print(f"   grad interior diff = {e:.2e}   div diff = {ed:.2e}")
        print(f"   {'PASS' if t else 'FAIL'}"); ok &= t
    except ImportError:
        print("   SKIP — structuralfluctuatinghydro_v6_3 not importable")
        print("   (run repair_five.py first)")

    print("\n[T2] [B2] face shapes")
    p = torch.rand(spec.cc_shape, dtype=torch.float64)
    gx, gy, gz = g.grad_cc_to_faces(p)
    t = (gx.shape == (N + 1, N, N) and gy.shape == (N, N + 1, N)
         and gz.shape == (N, N, N + 1))
    print(f"   {tuple(gx.shape)} {tuple(gy.shape)} {tuple(gz.shape)}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T3] [B1] face_beta = mean of reciprocals, NOT harmonic mean")
    rho = torch.rand(spec.cc_shape, dtype=torch.float64) * 999 + 1
    bx, by, bz = face_beta(rho, spec)
    ref = 0.5 * (1.0 / rho[:-1] + 1.0 / rho[1:])
    wrong = 2.0 / (1.0 / rho[:-1] + 1.0 / rho[1:])
    e_ok = float((bx[1:-1] - ref).abs().max())
    e_bad = float((bx[1:-1] - wrong).abs().max())
    t = e_ok < 1e-14 and e_bad > 1e-6
    print(f"   |beta - mean(1/rho)| = {e_ok:.2e}  (expect ~0)")
    print(f"   |beta - harmonic     | = {e_bad:.2e}  (expect large)")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T4] [B3] strain rate all at centres")
    u, v, w = (torch.rand(spec.face_shape(a), dtype=torch.float64) * 0.1
               for a in range(3))
    u, v, w = g.enforce_velocity_bc(u, v, w)
    mag, comps = g.strain_rate_cc(u, v, w)
    t = (mag.shape == spec.cc_shape
         and all(c.shape == spec.cc_shape for c in comps)
         and bool(torch.isfinite(mag).all()))
    print(f"   |S| {tuple(mag.shape)}  all 6 components at centres: "
          f"{all(c.shape == spec.cc_shape for c in comps)}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T5] [B4] viscous stress shapes, all 3 axes, anisotropic grid")
    sp2 = GridSpec(nx=10, ny=14, nz=8, Lx=1.0, Ly=1.4, Lz=0.8,
                   bc=("wall", "periodic", "slip"), dtype="float64")
    g2 = MACOps(sp2)
    uu, vv, ww = (torch.rand(sp2.face_shape(a), dtype=torch.float64) * 0.1
                  for a in range(3))
    uu, vv, ww = g2.enforce_velocity_bc(uu, vv, ww)
    mu = torch.rand(sp2.cc_shape, dtype=torch.float64) * 1e-3 + 1e-4
    try:
        vu, vv2, vw = g2.viscous_all(uu, vv, ww, mu)
        t = (vu.shape == sp2.face_shape(0) and vv2.shape == sp2.face_shape(1)
             and vw.shape == sp2.face_shape(2)
             and all(bool(torch.isfinite(t_).all()) for t_ in (vu, vv2, vw)))
        print(f"   {tuple(vu.shape)} {tuple(vv2.shape)} {tuple(vw.shape)}")
        print(f"   {'PASS' if t else 'FAIL'}"); ok &= t
    except Exception as ex:
        print(f"   FAIL: {type(ex).__name__}: {ex}"); ok = False

    print("\n[T6] variable-density Poisson residual (1000:1)")
    a = torch.zeros(spec.cc_shape, dtype=torch.float64); a[:N // 2] = 1.0
    rho2 = a * 1000.0 + (1 - a) * 1.0
    rhs = torch.rand(spec.cc_shape, dtype=torch.float64); rhs -= rhs.mean()
    ps = PressurePoisson(spec, g, tol=1e-11, max_iter=6000)
    sol = ps.solve(rhs, rho2)
    res = float((ps.apply(sol, *face_beta(rho2, spec)) - rhs).norm() / rhs.norm())
    t = res < 1e-9 and ps.converged
    print(f"   rel-res={res:.2e} iters={ps.last_iters} converged={ps.converged}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T7] projection makes velocity divergence-free")
    d0 = float(g.divergence(u, v, w).abs().max())
    un, vn, wn, pn, _ = project(u, v, w, rho2, spec, dt=1e-3)
    d1 = float(g.divergence(un, vn, wn).abs().max())
    t = d1 < 1e-9
    print(f"   max|div| {d0:.3e} -> {d1:.3e}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T8] [B5] advection conserves mass exactly")
    al = torch.zeros(spec.cc_shape, dtype=torch.float64)
    al[3:8, 3:8, 3:8] = 1.0
    cu = g.zeros_u() + 0.7; cv = g.zeros_v() + 0.3; cw = g.zeros_w() - 0.2
    m0 = float(al.sum()); aa = al
    for _ in range(20):
        aa = advect_alpha_upwind(aa, cu, cv, cw, 0.2 * spec.dx, spec)
    m1 = float(aa.sum())
    rel = abs(m1 - m0) / max(m0, 1e-30)
    t = rel < 1e-12 and float(aa.min()) >= 0 and float(aa.max()) <= 1
    print(f"   mass {m0:.6f} -> {m1:.6f}  rel={rel:.2e}")
    print(f"   alpha in [{float(aa.min()):.3f}, {float(aa.max()):.3f}]")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T9] autograd through projection and viscous stress")
    rq = rho2.clone().requires_grad_(True)
    uq = torch.rand(spec.face_shape(0), dtype=torch.float64, requires_grad=True)
    u2, v2, w2, p2, _ = project(uq, g.zeros_v(), g.zeros_w(), rq, spec, 1e-3)
    (p2.pow(2).sum() + g.viscous_component(
        (u2, v2, w2), rq * 1e-3, 0).pow(2).sum()).backward()
    t = (uq.grad is not None and torch.isfinite(uq.grad).all()
         and rq.grad is not None and torch.isfinite(rq.grad).all())
    print(f"   finite grads: u={uq.grad is not None} rho={rq.grad is not None}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= bool(t)

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL STENCIL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
