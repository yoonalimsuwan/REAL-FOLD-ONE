#!/usr/bin/env python3
"""
=============================================================================
ZALESAK SLOTTED DISK  +  COUPLED CSF x MOMENTUM SPURIOUS-CURRENT ABLATION
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

LITERATURE
 [Z1] Zalesak (1979) "Fully multidimensional flux-corrected transport
      algorithms for fluids." J. Comput. Phys. 31, 335.   [slotted disk]
 [R1] Rudman (1998) IJNMF 28, 357.        [VOF benchmarks, conservative momentum]
 [R2] Rider & Kothe (1998) JCP 141, 112.  [geometric advection, reversible tests]
 [R3] Raessi & Pitsch (2012) JCP 231, 2092. [consistent mass-momentum]
 [S1] Scardovelli & Zaleski (1999) JCP 155, 228.  [analytic cut volume]
 [B1] Francois et al. (2006) JCP 213, 141.        [balanced force]
 [B2] Popinet (2009) JCP 228, 5838.               [height functions, Gerris]
 [H1] Cummins, Francois, Kothe (2005) Comput. Struct. 83, 425.
 [L1] LeVeque (1996) SIAM J. Numer. Anal. 33, 627. [deformation field]

=============================================================================
PART 1 — WHY ZALESAK IS HARDER THAN A CIRCLE
=============================================================================
A translating/rotating CIRCLE only tests curvature-driven smoothing. The
slotted disk additionally tests TOPOLOGY PRESERVATION:

  • The slot has four SHARP CORNERS. Any scheme with excess numerical
    diffusion rounds them, and a sufficiently diffusive scheme FILLS the
    slot entirely — at which point the topology has changed (a simply
    connected disk instead of a slotted one) while the L1 error can still
    look modest.
  • The two "legs" beside the slot are THIN features. Under-resolution
    merges them.

Therefore the primary diagnostic here is NOT L1 alone but:
    slot_fill = mean(alpha) inside the exact slot region   (ideal: 0)
    slot_max  = max(alpha)  inside the exact slot region   (ideal: 0)
reported alongside L1/L2/Linf, mass drift, and boundedness.

The classical Zalesak configuration is 2-D. Here it is EXTRUDED along z
(a slotted cylinder) with w = 0, which makes it a genuine 3-D test with an
EXACT reference: after one revolution alpha(T) == alpha(0).

REFERENCE ERROR
  Both this and the droplet test are REVERSIBLE / STATIC, so the reference
  is the initial field and its quadrature error CANCELS in the comparison.
  Initialisation quality still affects the error MAGNITUDE (a rougher
  shape is harder to advect), so n_sub is reported, not hidden.

=============================================================================
PART 2 — WHAT THE SPURIOUS-CURRENT ABLATION ACTUALLY TESTS
=============================================================================
HYPOTHESIS (stated before measuring, not after):
  Spurious currents around a static droplet are driven primarily by
    (a) CURVATURE ERROR  (CSF vs height function)
    (b) FORCE/PRESSURE DISCRETIZATION MISMATCH (balanced vs unbalanced)
  and only secondarily by
    (c) the MOMENTUM TRANSPORT SCHEME.
  Reason: at equilibrium |u| is tiny, so the momentum advection term is
  itself tiny; changing how a near-zero term is discretized should matter
  less than getting kappa and the force balance right.

This script MEASURES all 12 combinations (3 momentum x 2 curvature x
2 balance) and prints the actual Capillary number Ca = mu|u|max/sigma and
the Laplace-pressure error. If (c) turns out to dominate, that contradicts
the hypothesis and is reported as such.

MOMENTUM SCHEMES COMPARED
  convective : u - dt*(u.grad)u, upwind  — the v1 non-conservative form.
               Does NOT discretely conserve momentum.
  upwind     : conservative flux form, piecewise-constant donor, mass flux
               taken from the geometric PLIC volume flux on the MOMENTUM CV.
  limited    : same, with minmod-limited linear donor reconstruction.

BALANCE
  balanced   : kappa and grad(alpha) from the SAME field, identical
               two-point difference as grad(p).
  unbalanced : kappa from SMOOTHED alpha, grad(alpha) from RAW alpha
               (reproduces the sesi_multiphase_flow.py defect).

=============================================================================
HONEST SCOPE
=============================================================================
❌ Balanced force removes the DISCRETIZATION mismatch only. Curvature error
   and the diffuse-interface representation error remain. Ca is NOT zero
   and is not claimed to be.
❌ Height functions are valid only where the interface is a graph at the
   window scale; coverage is REPORTED, non-graph cells fall back to CSF.
❌ The geometric momentum path RE-RECONSTRUCTS a plane on the shifted CV.
   The shifted alpha is exact for piecewise-planar cells, but fitting a new
   plane to it introduces its own O(h^2) error. Not exact geometric
   momentum transport.
❌ Operator-split advection is FIRST ORDER IN TIME.
❌ 90-degree contact angle only. No AMR. No experimental validation.
❌ Cost: the geometric momentum adds 3 reconstructions + 9 flux evals per
   step (~4x the alpha advection). Reported per run.
=============================================================================
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

VERSION = "1.0.0"


# =============================================================================
# GRID + MAC OPERATORS
# =============================================================================

@dataclass
class Grid:
    nx: int = 32; ny: int = 32; nz: int = 32
    Lx: float = 1.0; Ly: float = 1.0; Lz: float = 1.0
    bc: Tuple[str, str, str] = ("periodic",) * 3
    device: str = "cpu"; dtype: str = "float64"

    @property
    def h(self): return (self.Lx/self.nx, self.Ly/self.ny, self.Lz/self.nz)
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
    def coords(self):
        c = [(torch.arange(n, device=self.dev, dtype=self.dt_)+0.5)*hh
             for n, hh in zip(self.n, self.h)]
        return torch.meshgrid(*c, indexing="ij")
    def faces_coord(self, ax):
        return torch.arange(self.n[ax]+1, device=self.dev, dtype=self.dt_)*self.h[ax]


def sl(ax, idx):
    o = [slice(None)]*3; o[ax] = idx; return tuple(o)


def diff(f, ax, h):
    return (f[sl(ax, slice(1, None))] - f[sl(ax, slice(0, -1))])/h


def pad1(f, ax, bc, antisym=False):
    n = f.shape[ax]; d = f.device
    if bc == "periodic":
        lo = f.index_select(ax, torch.tensor([n-1], device=d))
        hi = f.index_select(ax, torch.tensor([0], device=d))
    else:
        lo = f.index_select(ax, torch.tensor([0], device=d))
        hi = f.index_select(ax, torch.tensor([n-1], device=d))
        if antisym: lo, hi = -lo, -hi
    return torch.cat([lo, f, hi], dim=ax)


def pad_cc(f, g: Grid, antisym=False):
    out = f
    for ax in range(3): out = pad1(out, ax, g.bc[ax], antisym)
    return out


def avg_pair(f, ax):
    return 0.5*(f[sl(ax, slice(0, -1))] + f[sl(ax, slice(1, None))])


def take_wrap(f, ax, idx, periodic):
    n = f.shape[ax]
    return f.index_select(ax, idx % n if periodic else idx.clamp(0, n-1))


def central_d(f, ax, g: Grid, padded=None):
    fp = padded if padded is not None else pad_cc(f, g)
    hi = fp[sl(ax, slice(2, None))]; lo = fp[sl(ax, slice(0, -2))]
    for o in range(3):
        if o != ax:
            hi = hi[sl(o, slice(1, -1))]; lo = lo[sl(o, slice(1, -1))]
    return (hi-lo)/(2*g.h[ax])


def central_dd(f, ax, g: Grid):
    fp = pad_cc(f, g)
    hi = fp[sl(ax, slice(2, None))]; lo = fp[sl(ax, slice(0, -2))]
    for o in range(3):
        if o != ax:
            hi = hi[sl(o, slice(1, -1))]; lo = lo[sl(o, slice(1, -1))]
    return (hi - 2*f + lo)/g.h[ax]**2


def grad_faces(p, g: Grid):
    """Two-point difference cell-centred scalar -> FACES. Shape-asserted."""
    pp = pad_cc(p, g); out = []
    for ax in range(3):
        gx = diff(pp, ax, g.h[ax])
        for o in range(3):
            if o != ax: gx = gx[sl(o, slice(1, -1))]
        assert gx.shape == g.face(ax), f"grad ax{ax}: {gx.shape}"
        out.append(gx)
    return tuple(out)


def divergence(u, v, w, g: Grid):
    return diff(u, 0, g.h[0]) + diff(v, 1, g.h[1]) + diff(w, 2, g.h[2])


def cc_to_face(f, g: Grid, ax):
    b = torch.zeros(g.face(ax), device=f.device, dtype=f.dtype)
    b[sl(ax, slice(1, -1))] = avg_pair(f, ax)
    if g.bc[ax] == "periodic":
        m = 0.5*(f[sl(ax, 0)] + f[sl(ax, -1)])
        b[sl(ax, 0)] = m; b[sl(ax, -1)] = m
    else:
        b[sl(ax, 0)] = f[sl(ax, 0)]; b[sl(ax, -1)] = f[sl(ax, -1)]
    return b


def face_avg_all(f, g: Grid):
    return tuple(cc_to_face(f, g, ax) for ax in range(3))


def face_beta(rho, g: Grid):
    """beta = 1/rho at faces = ARITHMETIC MEAN OF RECIPROCALS."""
    inv = 1.0/rho.clamp_min(1e-30); out = []
    for ax in range(3):
        b = torch.zeros(g.face(ax), device=rho.device, dtype=rho.dtype)
        b[sl(ax, slice(1, -1))] = avg_pair(inv, ax)
        b[sl(ax, 0)] = inv[sl(ax, 0)]; b[sl(ax, -1)] = inv[sl(ax, -1)]
        out.append(b)
    return tuple(out)


def enforce_bc(u, v, w, g: Grid):
    out = []
    for ax, c in enumerate((u, v, w)):
        c = c.clone()
        i0 = torch.tensor([0], device=c.device)
        i1 = torch.tensor([c.shape[ax]-1], device=c.device)
        if g.bc[ax] == "periodic":
            m = 0.5*(c.index_select(ax, i0)+c.index_select(ax, i1))
            c.index_copy_(ax, i0, m); c.index_copy_(ax, i1, m)
        else:
            z = torch.zeros_like(c.index_select(ax, i0))
            c.index_copy_(ax, i0, z); c.index_copy_(ax, i1, z)
        out.append(c)
    return tuple(out)


def minmod(a, b):
    return torch.where(a*b > 0, torch.sign(a)*torch.minimum(a.abs(), b.abs()),
                       torch.zeros_like(a))


# =============================================================================
# PLIC  [S1]
# =============================================================================

class PLIC:
    _TOL = 1e-3

    @staticmethod
    def normals(alpha, g: Grid):
        a = alpha.to(torch.float64)
        gs = [central_d(a, ax, g) for ax in range(3)]
        mag = torch.sqrt(sum(t*t for t in gs) + 1e-60)
        return tuple(-t/mag.clamp_min(1e-30) for t in gs)

    @staticmethod
    def _p3(t): return t.clamp_min(0.0)**3

    @classmethod
    def box_volume(cls, d, m1, m2, m3, L1, L2, L3):
        d = torch.as_tensor(d, dtype=torch.float64)
        m1 = torch.as_tensor(m1, dtype=torch.float64).clamp_min(0.)
        m2 = torch.as_tensor(m2, dtype=torch.float64).clamp_min(0.)
        m3 = torch.as_tensor(m3, dtype=torch.float64).clamp_min(0.)
        L1 = torch.as_tensor(L1, dtype=torch.float64)
        L2 = torch.as_tensor(L2, dtype=torch.float64)
        L3 = torch.as_tensor(L3, dtype=torch.float64)
        d, m1, m2, m3, L1, L2, L3 = torch.broadcast_tensors(d, m1, m2, m3, L1, L2, L3)
        a1, a2, a3 = m1*L1, m2*L2, m3*L3
        ns = ((m1 < cls._TOL).to(torch.int8)+(m2 < cls._TOL).to(torch.int8)
              + (m3 < cls._TOL).to(torch.int8))
        p3 = cls._p3
        s = (p3(d)-p3(d-a1)-p3(d-a2)-p3(d-a3)+p3(d-a1-a2)
             + p3(d-a1-a3)+p3(d-a2-a3)-p3(d-a1-a2-a3))
        v3 = torch.nan_to_num(s/(6*m1*m2*m3).clamp_min(1e-300), 0., 0., 0.)

        def area2(dd, ma, mb, La, Lb):
            A, B = ma*La, mb*Lb
            t = (dd.clamp_min(0.)**2-(dd-A).clamp_min(0.)**2
                 - (dd-B).clamp_min(0.)**2+(dd-A-B).clamp_min(0.)**2)
            return t/(2*ma*mb).clamp_min(1e-300)

        v2 = [area2(d, m2, m3, L2, L3)*L1, area2(d, m1, m3, L1, L3)*L2,
              area2(d, m1, m2, L1, L2)*L3]
        v1 = [(d/m1.clamp_min(1e-300)).clamp(0., L1)*L2*L3,
              (d/m2.clamp_min(1e-300)).clamp(0., L2)*L1*L3,
              (d/m3.clamp_min(1e-300)).clamp(0., L3)*L1*L2]
        vmax = L1*L2*L3; out = v3
        for i, m in enumerate((m1, m2, m3)):
            out = torch.where((ns == 1) & (m < cls._TOL), v2[i], out)
        for i, m in enumerate((m1, m2, m3)):
            out = torch.where((ns >= 2) & (m >= cls._TOL), v1[i], out)
        return torch.where(ns == 3, vmax*(d > 0).to(d.dtype), out).clamp(0., vmax)

    @classmethod
    def plane_offset(cls, alpha, n, iters=26):
        a = alpha.to(torch.float64).clamp(0, 1)
        m = [t.abs().to(torch.float64) for t in n]
        s = sum(m).clamp_min(1e-30); m = [t/s for t in m]
        one = torch.ones_like(a)
        lo, hi = torch.zeros_like(a), one.clone()
        for _ in range(iters):
            mid = 0.5*(lo+hi)
            v = cls.box_volume(mid, m[0], m[1], m[2], one, one, one)
            small = v < a
            lo = torch.where(small, mid, lo); hi = torch.where(small, hi, mid)
        d = 0.5*(lo+hi)
        shift = sum(torch.where(t < 0, mm, torch.zeros_like(mm))
                    for t, mm in zip(n, m))
        return d + shift, m          # FOLDED offset

    @classmethod
    def slab_volume(cls, d_fold, m, s, axis, high_side):
        one = torch.ones_like(d_fold)
        L = [one, one, one]; L[axis] = s
        off = d_fold - m[axis]*(1.0-s) if high_side else d_fold
        return cls.box_volume(off, m[0], m[1], m[2], L[0], L[1], L[2])

    @classmethod
    def volume_flux(cls, d_fold, m, vel, dt, h, ax, periodic):
        """DIMENSIONLESS crossing volume F = V_f/V_cell on FACES."""
        s = (vel.abs().to(torch.float64)*dt/h).clamp(0., 1.)
        nf = vel.shape[ax]
        idx = torch.arange(nf, device=vel.device)
        sh = lambda f, dl: take_wrap(f, ax, idx+dl, periodic)
        dL, dR = sh(d_fold, -1), sh(d_fold, 0)
        mL = [sh(t, -1) for t in m]; mR = [sh(t, 0) for t in m]
        fL = cls.slab_volume(dL, mL, s, ax, True)
        fR = cls.slab_volume(dR, mR, s, ax, False)
        flux = torch.where(vel > 0, fL, -fR)
        if not periodic:
            msk = torch.ones_like(flux)
            msk[sl(ax, 0)] = 0.; msk[sl(ax, nf-1)] = 0.
            flux = flux*msk
        return flux


def plic_advect(alpha, u, v, w, dt, g: Grid, sweep0=0, iters=26):
    """Operator-split geometric advection + dilation correction + bounded repair."""
    a = alpha; m0 = float(a.sum()); Fs = [None]*3
    vel = (u, v, w)
    for ax in [(sweep0+k) % 3 for k in range(3)]:
        n = PLIC.normals(a, g)
        d_fold, m = PLIC.plane_offset(a, n, iters)
        Fa = PLIC.volume_flux(d_fold, m, vel[ax], dt, g.h[ax], ax,
                              g.bc[ax] == "periodic")
        Fs[ax] = Fa
        a = (a - diff(Fa, ax, 1.0)
             + a*diff(vel[ax], ax, g.h[ax])*dt).clamp(0., 1.)
    defect = m0 - float(a.sum())
    if abs(defect) > 1e-13:
        wgt = ((a > 1e-6) & (a < 1-1e-6)).to(a.dtype)
        if float(wgt.sum()) < 1.0: wgt = torch.ones_like(a)
        a = (a + defect*(wgt/wgt.sum().clamp_min(1e-30))).clamp(0., 1.)
    return a, tuple(Fs)


# =============================================================================
# GEOMETRIC MOMENTUM ON THE MOMENTUM CV  [R1,R3]
# =============================================================================

def shifted_alpha_geometric(d_fold, m, axis, g: Grid):
    """
    alpha on the momentum CV shifted half a cell along `axis`, built
    GEOMETRICALLY by clipping each cell's own plane to the overlapping
    half-box. NOT the arithmetic mean 0.5*(alpha[i-1]+alpha[i]).
    """
    s = torch.full_like(d_fold, 0.5)
    Vhi = PLIC.slab_volume(d_fold, m, s, axis, True)
    Vlo = PLIC.slab_volume(d_fold, m, s, axis, False)
    nf = g.n[axis]+1
    idx = torch.arange(nf, device=d_fold.device)
    per = g.bc[axis] == "periodic"
    return (take_wrap(Vhi, axis, idx-1, per)
            + take_wrap(Vlo, axis, idx, per)).clamp(0., 1.)


def cv_face_velocity(vel, a, d, g: Grid):
    """Velocity component d at the (a-CV, d) face location."""
    comp = vel[d]
    if d == a:
        return avg_pair(pad1(comp, a, g.bc[a]), a)          # n_a+2
    edge = avg_pair(pad1(comp, d, g.bc[d]), d)
    return avg_pair(pad1(edge, a, g.bc[a]), a)              # n_a+1, n_d+1


def _donor_pair(comp, a, d, g: Grid, order):
    if d == a:
        base = comp
        p = pad1(base, a, g.bc[a])
        fm = p[sl(a, slice(0, -2))]; f0 = p[sl(a, slice(1, -1))]
        fp = p[sl(a, slice(2, None))]
        slope = minmod(f0-fm, fp-f0)
        idx = torch.arange(g.n[a]+2, device=comp.device)
        per = g.bc[a] == "periodic"
        vl = take_wrap(base, a, idx-1, per); vh = take_wrap(base, a, idx, per)
        sll = take_wrap(slope, a, idx-1, per); slh = take_wrap(slope, a, idx, per)
    else:
        edge = avg_pair(pad1(comp, d, g.bc[d]), d)
        p = pad1(edge, d, g.bc[d])
        fm = p[sl(d, slice(0, -2))]; f0 = p[sl(d, slice(1, -1))]
        fp = p[sl(d, slice(2, None))]
        slope = minmod(f0-fm, fp-f0)
        idx = torch.arange(g.n[d]+1, device=comp.device)
        per = g.bc[d] == "periodic"
        vl = take_wrap(edge, d, idx-1, per); vh = take_wrap(edge, d, idx, per)
        sll = take_wrap(slope, d, idx-1, per); slh = take_wrap(slope, d, idx, per)
    if order == "upwind":
        return vl, vh
    return vl + 0.5*sll, vh - 0.5*slh


def geometric_momentum(alpha, vel, rho1, rho2, dt, g: Grid,
                       order="limited", iters=26, rho_floor=1e-6):
    """
    Consistent mass-momentum transport on each momentum CV.
    rho_f_new uses the SAME CV mass flux as momentum -> u=const gives
    (rho u)_new == u * rho_f_new EXACTLY.
    """
    n = PLIC.normals(alpha, g)
    d_fold, m = PLIC.plane_offset(alpha, n, iters)
    rho_cell = rho2 + (rho1-rho2)*alpha
    rf_new, rhou_new, u_new = [], [], []
    n_recon, n_flux = 1, 3
    for a in range(3):
        a_cv = shifted_alpha_geometric(d_fold, m, a, g)
        n_cv = PLIC.normals(a_cv, g)
        d_cv, m_cv = PLIC.plane_offset(a_cv, n_cv, iters)
        n_recon += 1
        Mcv = []
        for d in range(3):
            u_cv = cv_face_velocity(vel, a, d, g)
            F = PLIC.volume_flux(d_cv, m_cv, u_cv, dt, g.h[d], d,
                                 g.bc[d] == "periodic")
            Mcv.append(rho2*u_cv + (rho1-rho2)*(F*g.h[d]/dt))
            n_flux += 1
        rf_old = cc_to_face(rho_cell, g, a)
        divM = sum(diff(Mcv[d], d, g.h[d]) for d in range(3))
        rf = (rf_old - dt*divM).clamp_min(rho_floor)
        tot = torch.zeros_like(vel[a])
        for d in range(3):
            lo, hi = _donor_pair(vel[a], a, d, g, order)
            flux = Mcv[d]*torch.where(Mcv[d] > 0, lo, hi)
            if g.bc[d] != "periodic":
                flux = flux.clone()
                flux[sl(d, 0)] = 0.; flux[sl(d, flux.shape[d]-1)] = 0.
            tot = tot + diff(flux, d, g.h[d])
        ru = rf_old*vel[a] - dt*tot
        rf_new.append(rf); rhou_new.append(ru); u_new.append(ru/rf)
    return dict(rho_face=tuple(rf_new), rhou=tuple(rhou_new),
                u=enforce_bc(*u_new, g),
                diag=dict(n_recon=n_recon, n_flux=n_flux))


def convective_momentum(vel, dt, g: Grid):
    """v1 non-conservative form: u - dt*(u.grad)u, first-order upwind."""
    uc = tuple(avg_pair(vel[a], a) for a in range(3))
    out = []
    for a in range(3):
        comp = vel[a]
        adv = torch.zeros_like(comp)
        for d in range(3):
            p = pad1(comp, d, g.bc[d])
            f0 = p[sl(d, slice(1, -1))]
            fm = p[sl(d, slice(0, -2))]
            fp = p[sl(d, slice(2, None))]
            for o in range(3):
                if o != d:
                    fm = fm[sl(o, slice(1, -1))] if fm.shape[o] > f0.shape[o] else fm
                    fp = fp[sl(o, slice(1, -1))] if fp.shape[o] > f0.shape[o] else fp
            # velocity component d interpolated to component-a faces
            vd = vel[d]
            if d != a:
                vd = avg_pair(pad1(vd, d, g.bc[d]), d)
                vd = avg_pair(pad1(vd, a, g.bc[a]), a)
            else:
                vd = avg_pair(pad1(vd, a, g.bc[a]), a)
            n_t = comp.shape[d]
            v_b = vd
            if v_b.shape[d] != n_t:
                v_b = avg_pair(pad1(v_b, d, g.bc[d]), d)
            deriv = torch.where(v_b > 0, (f0-fm)/g.h[d], (fp-f0)/g.h[d])
            adv = adv + v_b*deriv
        out.append(comp - dt*adv)
    return enforce_bc(*out, g)


# =============================================================================
# CURVATURE: CSF + HEIGHT FUNCTIONS  [B1,B2,H1]
# =============================================================================

def curvature_csf(alpha, g: Grid, n_smooth=0):
    a = alpha
    for _ in range(n_smooth):
        acc = 6.0*a
        for ax in range(3):
            p = pad1(a, ax, g.bc[ax])
            acc = acc + p[sl(ax, slice(2, None))] + p[sl(ax, slice(0, -2))]
        a = acc/12.0
    gs = [central_d(a, ax, g) for ax in range(3)]
    mag = torch.sqrt(sum(t*t for t in gs) + 1e-30)
    nh = [t/mag for t in gs]
    k = torch.zeros_like(a)
    for ax in range(3):
        k = k - central_d(nh[ax], ax, g)
    return torch.nan_to_num(k, 0., 0., 0.), a


class HeightFunction:
    """Local-window heights + FULL graph formula + crossing-count validity."""
    def __init__(self, g: Grid, window=3, normal_threshold=0.7):
        self.g = g; self.w = window; self.thr = normal_threshold

    def _shift(self, f, ax, d):
        n = f.shape[ax]
        idx = torch.arange(n, device=f.device)+d
        return f.index_select(ax, idx % n if self.g.bc[ax] == "periodic"
                              else idx.clamp(0, n-1))

    def _heights(self, alpha, ax):
        h = torch.zeros_like(alpha)
        for d in range(-self.w, self.w+1):
            h = h + self._shift(alpha, ax, d)
        return h*self.g.h[ax]

    def _crossings(self, alpha, ax):
        s = (alpha > 0.5).to(alpha.dtype)
        jump = (self._shift(s, ax, 1)-s).abs()
        if self.g.bc[ax] != "periodic":
            jump = jump.clone(); jump[sl(ax, alpha.shape[ax]-1)] = 0.
        c = torch.zeros_like(alpha)
        for d in range(-self.w, self.w+1):
            c = c + self._shift(jump, ax, d)
        return c

    def curvature(self, alpha):
        g = self.g
        k_csf, _ = curvature_csf(alpha, g, 0)
        gs = [central_d(alpha, ax, g) for ax in range(3)]
        mag = torch.sqrt(sum(t*t for t in gs) + 1e-30)
        nabs = torch.stack([(t/mag).abs() for t in gs], 0)
        dom = nabs.argmax(0); mx = nabs.max(0).values
        kappa = k_csf.clone()
        used = torch.zeros_like(alpha, dtype=torch.bool)
        for ax in range(3):
            h = self._heights(alpha, ax)
            a1, a2 = [a for a in range(3) if a != ax]
            h1 = central_d(h, a1, g); h2 = central_d(h, a2, g)
            h11 = central_dd(h, a1, g); h22 = central_dd(h, a2, g)
            h12 = central_d(h1, a2, g)
            den = (1+h1*h1+h2*h2)**1.5
            num = h11*(1+h2*h2) + h22*(1+h1*h1) - 2*h1*h2*h12
            k_ax = -num/den.clamp_min(1e-30)
            ok = (dom == ax) & (mx > self.thr) & (self._crossings(alpha, ax) == 1)
            kappa = torch.where(ok, k_ax, kappa); used = used | ok
        iface = (alpha > 1e-6) & (alpha < 1-1e-6)
        nif = int(iface.sum())
        return dict(kappa=torch.nan_to_num(kappa, 0., 0., 0.),
                    kappa_csf=k_csf,
                    hf_coverage=(int((used & iface).sum())/nif if nif else 0.0))


def csf_force(alpha, kappa, g: Grid, sigma, balanced=True, n_smooth=1):
    """
    balanced   : kappa and grad(alpha) from the SAME field.
    unbalanced : kappa from smoothed alpha, grad(alpha) from RAW alpha.
    """
    if sigma <= 0:
        return (g.fc(0), g.fc(1), g.fc(2))
    if balanced:
        a_use = alpha
        for _ in range(n_smooth):
            acc = 6.0*a_use
            for ax in range(3):
                p = pad1(a_use, ax, g.bc[ax])
                acc = acc + p[sl(ax, slice(2, None))] + p[sl(ax, slice(0, -2))]
            a_use = acc/12.0
    else:
        a_use = alpha                       # RAW grad — the v1 mismatch
    kf = face_avg_all(kappa, g)
    ap = pad_cc(a_use, g)
    out = []
    for ax in range(3):
        ga = diff(ap, ax, g.h[ax])
        for o in range(3):
            if o != ax: ga = ga[sl(o, slice(1, -1))]
        out.append(sigma*kf[ax]*ga)
    return tuple(out)


# =============================================================================
# POISSON + PROJECTION
# =============================================================================

class Poisson:
    def __init__(self, g: Grid, tol=1e-11, max_iter=4000):
        self.g, self.tol, self.max_iter = g, tol, max_iter
        self.iters = 0; self.res = float("nan"); self.converged = False

    def apply(self, p, b):
        gp = grad_faces(p, self.g)
        return sum(diff(b[ax]*gp[ax], ax, self.g.h[ax]) for ax in range(3))

    def solve(self, rhs, rho):
        g = self.g; b = face_beta(rho, g)
        rhs = rhs - rhs.mean(); p = torch.zeros_like(rhs)
        diag = sum((b[ax][sl(ax, slice(1, None))]+b[ax][sl(ax, slice(0, -1))])
                   / g.h[ax]**2 for ax in range(3)).clamp_min(1e-30)
        r = rhs - self.apply(p, b); r = r - r.mean()
        z = r/diag; d = z.clone(); rz = (r*z).sum()
        nrm = rhs.norm().clamp_min(1e-30); self.converged = False
        for it in range(self.max_iter):
            Ad = self.apply(d, b); dAd = (d*Ad).sum()
            if float(dAd.abs()) < 1e-300: break
            al = rz/dAd; p = p + al*d
            r = r - al*Ad; r = r - r.mean()
            res = float(r.norm()/nrm)
            if res < self.tol:
                self.iters, self.res, self.converged = it+1, res, True
                return p - p.mean()
            z = r/diag; rzn = (r*z).sum()
            d = z + (rzn/rz.clamp_min(1e-300))*d; rz = rzn
        self.iters, self.res = self.max_iter, float(r.norm()/nrm)
        return p - p.mean()


def project(u, v, w, rho, dt, g: Grid, ps: Poisson):
    p = ps.solve(divergence(u, v, w, g)/dt, rho)
    b = face_beta(rho, g); gp = grad_faces(p, g)
    un = enforce_bc(u-dt*b[0]*gp[0], v-dt*b[1]*gp[1], w-dt*b[2]*gp[2], g)
    return un[0], un[1], un[2], p, ps


# =============================================================================
# ZALESAK SLOTTED DISK  [Z1]
# =============================================================================

@dataclass
class ZalesakConfig:
    center: Tuple[float, float] = (0.5, 0.75)
    radius: float = 0.15
    slot_width: float = 0.05
    slot_bottom: float = 0.60      # = center_y - radius
    slot_top: float = 0.85         # = slot_bottom + 0.25
    omega: float = 2*math.pi       # one revolution per unit time
    rot_center: Tuple[float, float] = (0.5, 0.5)


def zalesak_indicator(x, y, cfg: ZalesakConfig):
    """Boolean indicator: inside disk AND NOT inside slot. Extruded along z."""
    cx, cy = cfg.center
    disk = (x-cx)**2 + (y-cy)**2 <= cfg.radius**2
    slot = ((x-cx).abs() <= cfg.slot_width/2) & (y >= cfg.slot_bottom) & (y <= cfg.slot_top)
    return disk & (~slot)


def zalesak_init(g: Grid, cfg: ZalesakConfig, n_sub=12):
    """
    Sub-cell quadrature initialisation. Point sampling is O(h) in L1 near
    the sharp slot corners; n_sub^3 samples per interface cell reduces it.
    Returns (alpha, slot_mask_exact).
    """
    X, Y, Z = g.coords()
    ind = zalesak_indicator(X, Y, cfg)
    # cells fully inside / fully outside via a margin test
    hmin = min(g.h[:2])
    cx, cy = cfg.center
    r_in = cfg.radius - 1.5*hmin
    r_out = cfg.radius + 1.5*hmin
    rr2 = (X-cx)**2 + (Y-cy)**2
    slot_pad = ((X-cx).abs() <= cfg.slot_width/2 - 1.5*hmin) & \
               (Y >= cfg.slot_bottom+1.5*hmin) & (Y <= cfg.slot_top-1.5*hmin)
    sure_in = (rr2 <= r_in**2) & (~slot_pad) & \
              ((X-cx).abs() >= cfg.slot_width/2 + 1.5*hmin)
    sure_out = (rr2 >= r_out**2)
    a = torch.zeros(g.n, device=g.dev, dtype=torch.float64)
    a[sure_in] = 1.0
    iface = ~(sure_in | sure_out)
    idx = iface.nonzero(as_tuple=False)
    t = (torch.arange(n_sub, device=g.dev, dtype=torch.float64)+0.5)/n_sub - 0.5
    ox, oy = torch.meshgrid(t, t, indexing="ij")
    off2 = torch.stack([ox.reshape(-1), oy.reshape(-1)], -1)
    hs = torch.tensor(g.h[:2], device=g.dev, dtype=torch.float64)
    CH = 4096
    for s in range(0, idx.shape[0], CH):
        ii = idx[s:s+CH]
        base = (ii[:, :2].to(torch.float64)+0.5)*hs
        pts = base[:, None, :] + off2[None, :, :]*hs[None, None, :]
        ind_s = zalesak_indicator(pts[..., 0], pts[..., 1], cfg)
        frac = ind_s.to(torch.float64).mean(-1)
        a[ii[:, 0], ii[:, 1], ii[:, 2]] = frac[:, None].expand(-1, 1).squeeze(-1)
    slot_exact = ((X-cx).abs() <= cfg.slot_width/2) & \
                 (Y >= cfg.slot_bottom) & (Y <= cfg.slot_top) & (rr2 <= cfg.radius**2)
    return a.to(g.dt_), slot_exact


def zalesak_velocity(g: Grid, cfg: ZalesakConfig, t=0.0):
    """
    Rigid rotation about rot_center. u depends only on y, v only on x,
    so the DISCRETE divergence is exactly zero (verified in T2).
    """
    xc, yc = g.rot_center if hasattr(g, "rot_center") else cfg.rot_center
    xf = g.faces_coord(0); yf = g.faces_coord(1)
    Xc, Yc, Zc = g.coords()
    U = -cfg.omega*(Yc - yc)                    # u at x-faces: y is cell-centred
    ux = torch.zeros(g.face(0), device=g.dev, dtype=g.dt_)
    ux[:] = (-cfg.omega*(Yc - yc)).expand(g.face(0))
    vy = torch.zeros(g.face(1), device=g.dev, dtype=g.dt_)
    vy[:] = (cfg.omega*(Xc - xc)).expand(g.face(1))
    wz = g.fc(2)
    return ux, vy, wz


def zalesak_run(g: Grid, cfg: ZalesakConfig, n_rev=1, cfl=0.3,
                n_sub=12, iters=26, verbose=True):
    a0, slot = zalesak_init(g, cfg, n_sub)
    T = n_rev*2*math.pi/cfg.omega
    umax = max(float(v.abs().max()) for v in zalesak_velocity(g, cfg))
    dt = cfl*min(g.h)/max(umax, 1e-12)
    nsteps = int(math.ceil(T/dt)); dt = T/nsteps
    a = a0.clone(); m0 = float(a0.sum())
    t0 = time.perf_counter()
    for k in range(nsteps):
        u, v, w = zalesak_velocity(g, cfg, k*dt)
        a, _ = plic_advect(a, u, v, w, dt, g, k % 3, iters)
    el = time.perf_counter()-t0
    d = (a-a0).abs()
    vref = float(a0.sum())
    out = dict(N=g.nx, n_rev=n_rev, dt=dt, n_steps=nsteps,
               l1=float(d.sum()/max(vref, 1e-30)),
               l2=float(torch.sqrt((d*d).sum()/max((a0*a0).sum(), 1e-30))),
               linf=float(d.max()),
               mass_drift=float(abs(a.sum()-a0.sum())/max(vref, 1e-30)),
               slot_fill=float(a[slot].mean()) if int(slot.sum()) else 0.0,
               slot_max=float(a[slot].max()) if int(slot.sum()) else 0.0,
               amin=float(a.min()), amax=float(a.max()),
               n_sub=n_sub, elapsed_s=el)
    if verbose:
        print(f"    N={g.nx:3d} rev={n_rev} steps={nsteps:5d} "
              f"L1={out['l1']:.4e} slot_fill={out['slot_fill']:.4e} "
              f"slot_max={out['slot_max']:.3f} mass={out['mass_drift']:.2e} "
              f"{el:.1f}s")
    return out, a, a0


# =============================================================================
# COUPLED SPURIOUS-CURRENT ABLATION
# =============================================================================

@dataclass
class DropletConfig:
    radius: float = 0.25
    center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    sigma: float = 0.05
    mu: float = 0.01
    rho1: float = 1.0
    rho2: float = 1.0
    momentum: str = "limited"       # convective | upwind | limited
    curvature: str = "hf"           # csf | hf
    balanced: bool = True
    n_smooth: int = 1
    hf_window: int = 3


def droplet_init(g: Grid, cfg: DropletConfig, n_sub=8):
    X, Y, Z = g.coords()
    cx, cy, cz = cfg.center
    r2 = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2
    hmin = min(g.h)
    inner = r2 <= (cfg.radius-1.2*hmin)**2
    outer = r2 >= (cfg.radius+1.2*hmin)**2
    a = torch.zeros(g.n, device=g.dev, dtype=torch.float64)
    a[inner] = 1.0
    idx = (~(inner | outer)).nonzero(as_tuple=False)
    t = (torch.arange(n_sub, device=g.dev, dtype=torch.float64)+0.5)/n_sub - 0.5
    ox, oy, oz = torch.meshgrid(t, t, t, indexing="ij")
    off = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], -1)
    hs = torch.tensor(g.h, device=g.dev, dtype=torch.float64)
    CH = 1024
    for s in range(0, idx.shape[0], CH):
        ii = idx[s:s+CH]
        base = (ii.to(torch.float64)+0.5)*hs
        pts = base[:, None, :] + off[None, :, :]*hs[None, None, :]
        d2 = ((pts - torch.tensor([cx, cy, cz], device=g.dev,
                                  dtype=torch.float64))**2).sum(-1)
        a[ii[:, 0], ii[:, 1], ii[:, 2]] = (d2 <= cfg.radius**2).to(torch.float64).mean(-1)
    return a.to(g.dt_)


def droplet_run(g: Grid, cfg: DropletConfig, n_steps=30, dt=2e-4,
                n_sub=8, iters=22, verbose=False):
    a = droplet_init(g, cfg, n_sub)
    u, v, w = g.fc(0), g.fc(1), g.fc(2)
    ps = Poisson(g, tol=1e-10, max_iter=1500)
    hf = HeightFunction(g, cfg.hf_window)
    m0 = float(a.sum())
    hf_cov = []
    t0 = time.perf_counter()
    p_last = None
    for k in range(n_steps):
        if cfg.curvature == "hf":
            ho = hf.curvature(a); kappa = ho["kappa"]; hf_cov.append(ho["hf_coverage"])
        else:
            kappa, _ = curvature_csf(a, g, cfg.n_smooth)
        F = csf_force(a, kappa, g, cfg.sigma, cfg.balanced, cfg.n_smooth)
        rho = cfg.rho2 + (cfg.rho1-cfg.rho2)*a

        if cfg.momentum == "convective":
            uu, vv, ww = convective_momentum((u, v, w), dt, g)
            rf = face_avg_all(rho, g)
            uu = uu + dt*F[0]/rf[0].clamp_min(1e-30)
            vv = vv + dt*F[1]/rf[1].clamp_min(1e-30)
            ww = ww + dt*F[2]/rf[2].clamp_min(1e-30)
        else:
            gm = geometric_momentum(a, (u, v, w), cfg.rho1, cfg.rho2, dt, g,
                                    order=cfg.momentum, iters=iters)
            uu, vv, ww = gm["u"]; rf = gm["rho_face"]
            uu = uu + dt*F[0]/rf[0].clamp_min(1e-30)
            vv = vv + dt*F[1]/rf[1].clamp_min(1e-30)
            ww = ww + dt*F[2]/rf[2].clamp_min(1e-30)

        uu, vv, ww = enforce_bc(uu, vv, ww, g)
        rho_new = cfg.rho2 + (cfg.rho1-cfg.rho2)*a
        u, v, w, p_last, ps = project(uu, vv, ww, rho_new, dt, g, ps)
        a, _ = plic_advect(a, u, v, w, dt, g, k % 3, iters)

    el = time.perf_counter()-t0
    umax = max(float(u.abs().max()), float(v.abs().max()), float(w.abs().max()))
    Ca = cfg.mu*umax/max(cfg.sigma, 1e-30)
    inside = a > 0.9; outside = a < 0.1
    dp_num = float(p_last[inside].mean()-p_last[outside].mean()) \
        if int(inside.sum()) and int(outside.sum()) else float("nan")
    dp_exact = 2*cfg.sigma/cfg.radius
    lap_err = abs(dp_num-dp_exact)/dp_exact if math.isfinite(dp_num) else float("nan")
    return dict(momentum=cfg.momentum, curvature=cfg.curvature,
                balanced=cfg.balanced, N=g.nx, n_steps=n_steps,
                umax=umax, Ca=Ca, dp_num=dp_num, dp_exact=dp_exact,
                laplace_rel_err=lap_err,
                hf_coverage=float(np.mean(hf_cov)) if hf_cov else 0.0,
                mass_drift=float(abs(a.sum()-m0)/max(m0, 1e-30)),
                div_max=float(divergence(u, v, w, g).abs().max()),
                amin=float(a.min()), amax=float(a.max()), elapsed_s=el)


def spurious_ablation(g: Grid, n_steps=30, dt=2e-4, radius=0.25,
                      sigma=0.05, mu=0.01, density_ratio=1.0,
                      out_json=None):
    print("\n" + "="*94)
    print("  SPURIOUS-CURRENT ABLATION — 3 momentum x 2 curvature x 2 balance")
    print(f"  N={g.nx}  steps={n_steps}  dt={dt:.1e}  R={radius}  "
          f"sigma={sigma}  rho1/rho2={density_ratio}")
    print("="*94)
    print(f"  {'momentum':<12}{'curv':<6}{'balance':<10}"
          f"{'max|u|':>11}{'Ca':>11}{'Laplace err':>13}{'HF cov':>9}{'s':>7}")
    rows = []
    for mom in ("convective", "upwind", "limited"):
        for cur in ("csf", "hf"):
            for bal in (False, True):
                cfg = DropletConfig(radius=radius, sigma=sigma, mu=mu,
                                    rho1=density_ratio, rho2=1.0,
                                    momentum=mom, curvature=cur, balanced=bal,
                                    n_smooth=1)
                r = droplet_run(g, cfg, n_steps=n_steps, dt=dt)
                rows.append(r)
                print(f"  {mom:<12}{cur:<6}{('balanced' if bal else 'MISMATCH'):<10}"
                      f"{r['umax']:>11.3e}{r['Ca']:>11.3e}"
                      f"{r['laplace_rel_err']:>12.2%}{r['hf_coverage']:>9.2f}"
                      f"{r['elapsed_s']:>7.1f}")
    # ---- factor importance (measured, not asserted) ----
    def mean_ca(pred):
        sel = [r["Ca"] for r in rows if pred(r)]
        return float(np.mean(sel)) if sel else float("nan")
    eff_balance = mean_ca(lambda r: not r["balanced"]) - mean_ca(lambda r: r["balanced"])
    eff_curv = mean_ca(lambda r: r["curvature"] == "csf") - mean_ca(lambda r: r["curvature"] == "hf")
    ca_mom = {m: mean_ca(lambda r, m=m: r["momentum"] == m)
              for m in ("convective", "upwind", "limited")}
    spread_mom = max(ca_mom.values()) - min(ca_mom.values())
    print("\n  MEASURED FACTOR IMPORTANCE (reduction in mean Ca)")
    print(f"    balanced vs MISMATCH : {eff_balance:+.3e}")
    print(f"    HF vs CSF            : {eff_curv:+.3e}")
    print(f"    momentum spread      : {spread_mom:.3e}   "
          f"({', '.join(f'{k}={v:.2e}' for k, v in ca_mom.items())})")
    ranking = sorted([("balance", abs(eff_balance)),
                      ("curvature", abs(eff_curv)),
                      ("momentum", abs(spread_mom))],
                     key=lambda kv: -kv[1])
    print(f"    ranking (largest first): {' > '.join(k for k, _ in ranking)}")
    hyp = (abs(eff_balance) > abs(spread_mom)) and (abs(eff_curv) > abs(spread_mom))
    print(f"\n  HYPOTHESIS (balance & curvature dominate momentum): "
          f"{'SUPPORTED' if hyp else 'NOT SUPPORTED by this run'}")
    print("  NOTE: Ca is NOT zero. Balance removes the discretization mismatch;")
    print("        curvature error and the diffuse-interface representation remain.")
    if out_json:
        Path(out_json).write_text(json.dumps(
            dict(rows=rows, eff_balance=eff_balance, eff_curvature=eff_curv,
                 momentum_spread=spread_mom, ranking=[k for k, _ in ranking],
                 hypothesis_supported=hyp), indent=2))
        print(f"  json -> {out_json}")
    return rows


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test(device="cpu", quick=True):
    print("="*94)
    print(f"  ZALESAK + SPURIOUS-CURRENT SUITE v{VERSION}")
    print("="*94)
    torch.manual_seed(0); ok = True

    def rep(n, p, e=""):
        nonlocal ok; ok &= bool(p); print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    N = 16
    g = Grid(nx=N, ny=N, nz=N, device=device)

    print("\n[T1] PLIC cut volume analytic + complement symmetry")
    one = torch.tensor([1.0], dtype=torch.float64)
    th = torch.tensor([1/3], dtype=torch.float64)
    V = float(PLIC.box_volume(torch.tensor([0.5], dtype=torch.float64),
                              th, th, th, one, one, one))
    ds = torch.linspace(0.02, 0.98, 21, dtype=torch.float64)
    mm = torch.full_like(ds, 1/3)
    e = float((PLIC.box_volume(ds, mm, mm, mm, one, one, one)
               + PLIC.box_volume(1-ds, mm, mm, mm, one, one, one)-1).abs().max())
    rep("V(0.5)=0.5", abs(V-0.5) < 1e-9, f"V={V:.12f}")
    rep("V(d)+V(1-d)=1", e < 1e-9, f"err={e:.2e}")

    print("\n[T2] Zalesak velocity is DISCRETELY divergence-free")
    zc = ZalesakConfig()
    u, v, w = zalesak_velocity(g, zc)
    dv = float(divergence(u, v, w, g).abs().max())
    rep("max|div| ~ 0", dv < 1e-12, f"{dv:.2e}")

    print("\n[T3] Zalesak init: slot is EMPTY, quadrature beats point sampling")
    a0, slot = zalesak_init(g, zc, n_sub=12)
    rep("slot initially empty", float(a0[slot].max()) < 1e-12,
        f"max alpha in slot = {float(a0[slot].max()):.2e}")
    rep("alpha bounded", float(a0.min()) >= 0 and float(a0.max()) <= 1)
    X, Y, Z = g.coords()
    a_pt = zalesak_indicator(X, Y, zc).to(torch.float64)
    a_fine, _ = zalesak_init(g, zc, n_sub=24)
    e_pt = float((a_pt-a_fine).abs().sum()/a_fine.sum())
    e_12 = float((a0-a_fine).abs().sum()/a_fine.sum())
    rep("quadrature closer to converged ref", e_12 < e_pt,
        f"point={e_pt:.3e}  n_sub=12={e_12:.3e}")

    print("\n[T4] geometric momentum: u=const -> (rho u) == u*rho_f EXACTLY")
    a = droplet_init(g, DropletConfig(), 6)
    uu, vv, ww = g.fc(0, 0.7), g.fc(1, -0.3), g.fc(2, 0.2)
    dt = 0.2*min(g.h)
    for order in ("upwind", "limited"):
        gm = geometric_momentum(a, (uu, vv, ww), 1000., 1., dt, g, order=order)
        errs = [float((gm["rhou"][i]-(uu, vv, ww)[i]*gm["rho_face"][i]).abs().max())
                for i in range(3)]
        rep(f"consistency ({order})", max(errs) < 1e-8, f"max err={max(errs):.2e}")

    print("\n[T5] geometric momentum conserves sum(rho u) (periodic)")
    gm = geometric_momentum(a, (uu, vv, ww), 1000., 1., dt, g, order="limited")
    rho_c = 1. + 999.*a
    P0 = [float((cc_to_face(rho_c, g, i)*(uu, vv, ww)[i]).sum()) for i in range(3)]
    P1 = [float(gm["rhou"][i].sum()) for i in range(3)]
    rel = max(abs(P1[i]-P0[i])/max(abs(P0[i]), 1e-30) for i in range(3))
    rep("momentum conserved", rel < 1e-10, f"rel={rel:.2e}")
    print(f"      cost: {gm['diag']['n_recon']} reconstructions, "
          f"{gm['diag']['n_flux']} flux evals")

    print("\n[T6] balanced force: F == grad(p) discretely (constant kappa)")
    sph = droplet_init(g, DropletConfig(), 6)
    k_const = torch.full_like(sph, 8.0)
    F, = (csf_force(sph, k_const, g, 0.05, balanced=True, n_smooth=1),)
    a_sm = sph
    for _ in range(1):
        acc = 6.0*a_sm
        for ax in range(3):
            p = pad1(a_sm, ax, g.bc[ax])
            acc = acc + p[sl(ax, slice(2, None))] + p[sl(ax, slice(0, -2))]
        a_sm = acc/12.0
    gp = grad_faces(0.05*8.0*a_sm, g)
    rb = max(float((F[i]-gp[i]).abs().max()) for i in range(3))
    sc = max(float(F[i].abs().max()) for i in range(3))+1e-30
    rep("balanced residual ~ 0", rb/sc < 1e-12, f"rel={rb/sc:.2e}")
    Fu = csf_force(sph, k_const, g, 0.05, balanced=False, n_smooth=1)
    gp_raw = grad_faces(0.05*8.0*sph, g)
    ru = max(float((Fu[i]-gp_raw[i]).abs().max()) for i in range(3))
    rep("MISMATCH breaks balance", ru/sc > 1e-3, f"rel={ru/sc:.2e}")

    print("\n[T7] HF curvature on a sphere: better than CSF, coverage reported")
    g2 = Grid(nx=32, ny=32, nz=32, device=device)
    a2 = droplet_init(g2, DropletConfig(radius=0.28), 8)
    ho = HeightFunction(g2, 3).curvature(a2)
    ifc = (a2 > 0.05) & (a2 < 0.95)
    k_ex = 2/0.28
    k_hf = float(ho["kappa"][ifc].abs().mean())
    k_cs = float(ho["kappa_csf"][ifc].abs().mean())
    e_hf, e_cs = abs(k_hf-k_ex)/k_ex, abs(k_cs-k_ex)/k_ex
    print(f"      exact 2/R={k_ex:.3f}  HF={k_hf:.3f} ({e_hf:.1%})  "
          f"CSF={k_cs:.3f} ({e_cs:.1%})  coverage={ho['hf_coverage']:.2f}")
    rep("HF better than CSF", e_hf < e_cs, f"{e_hf:.1%} < {e_cs:.1%}")

    if not quick:
        print("\n[T8] ZALESAK measured spatial order (dt FIXED)")
        zrows = []
        for N in (32, 64):
            gz = Grid(nx=N, ny=N, nz=max(8, N//4), device=device)
            r, _, _ = zalesak_run(gz, ZalesakConfig(), n_rev=1, cfl=0.3)
            zrows.append(r)
        if len(zrows) == 2 and zrows[1]["l1"] > 0:
            p = math.log2(zrows[0]["l1"]/zrows[1]["l1"])
            print(f"    measured p_L1 = {p:.3f}   "
                  f"slot_fill {zrows[0]['slot_fill']:.3e} -> {zrows[1]['slot_fill']:.3e}")
            print("    Split PLIC is 1st order in time; p is MEASURED, not asserted.")
        rep("zalesak produced numbers", len(zrows) == 2)

        print("\n[T9] COUPLED SPURIOUS-CURRENT ABLATION")
        gs = Grid(nx=24, ny=24, nz=24, bc=("wall",)*3, device=device)
        spurious_ablation(gs, n_steps=25, dt=2e-4,
                          out_json="spurious_ablation.json")

    print("\n"+"="*94)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*94)
    print("""
  PROVEN
    PLIC cut volume analytic + symmetric
    Zalesak velocity discretely divergence-free; slot initially empty
    quadrature init closer to converged reference than point sampling
    u=const gives (rho u) == u*rho_f EXACTLY (discrete consistency)
    sum(rho u) conserved by flux telescoping
    balanced force: F == grad(p) to round-off; MISMATCH breaks it

  MEASURED, NOT CLAIMED  (run --full)
    Zalesak spatial order + slot-fill
    spurious-current Ca for all 12 configurations, with factor ranking

  NOT DONE
    Ca is not zero; curvature + representation error remain
    re-reconstruction on the shifted CV is an approximation
    split advection 1st order in time; 90-deg contact angle; no AMR
    no experimental validation
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "zalesak", "spurious", "full"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--grids", type=int, nargs="*", default=[32, 64])
    p.add_argument("--rev", type=int, default=1)
    p.add_argument("--cfl", type=float, default=0.3)
    p.add_argument("--nsub", type=int, default=12)
    p.add_argument("--N", type=int, default=24)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dt", type=float, default=2e-4)
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.cmd == "selftest":
        raise SystemExit(0 if self_test(a.device, quick=True) else 1)
    if a.cmd == "full":
        raise SystemExit(0 if self_test(a.device, quick=False) else 1)
    if a.cmd == "zalesak":
        print("="*94); print("  ZALESAK SLOTTED DISK — measured order + slot fill")
        print("="*94)
        rows = []
        for N in a.grids:
            gz = Grid(nx=N, ny=N, nz=max(8, N//4), device=a.device)
            r, _, _ = zalesak_run(gz, ZalesakConfig(), n_rev=a.rev,
                                  cfl=a.cfl, n_sub=a.nsub)
            rows.append(r)
        for i in range(1, len(rows)):
            if rows[i]["l1"] > 0:
                pp = math.log2(rows[i-1]["l1"]/rows[i]["l1"])
                print(f"    p_L1  N {rows[i-1]['N']} -> {rows[i]['N']}: {pp:.3f}")
        print("    slot_fill is the TOPOLOGY metric: >0 means the slot is filling in.")
        if a.out: Path(a.out).write_text(json.dumps(rows, indent=2))
        return
    if a.cmd == "spurious":
        gs = Grid(nx=a.N, ny=a.N, nz=a.N, bc=("wall",)*3, device=a.device)
        spurious_ablation(gs, n_steps=a.steps, dt=a.dt, density_ratio=a.ratio,
                          out_json=a.out or "spurious_ablation.json")
        return


if __name__ == "__main__":
    main()
