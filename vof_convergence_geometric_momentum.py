#!/usr/bin/env python3
"""
=============================================================================
VOF CONVERGENCE STUDY + FULL GEOMETRIC SUB-CELL MOMENTUM ON THE MOMENTUM CV
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

LITERATURE
 [Z1] Zalesak (1979) "Fully multidimensional flux-corrected transport
      algorithms for fluids." J. Comput. Phys. 31, 335.   [slotted disk]
 [R1] Rider & Kothe (1998) "Reconstructing volume tracking."
      J. Comput. Phys. 141, 112.                          [reversible vortex]
 [R2] Rudman (1998) IJNMF 28, 357.                        [conservative momentum]
 [R3] Raessi & Pitsch (2012) JCP 231, 2092.               [consistent transport]
 [L1] LeVeque (1996) "High-resolution conservative algorithms for advection
      in incompressible flow." SIAM J. Numer. Anal. 33, 627.  [3D deformation]
 [S1] Scardovelli & Zaleski (1999) JCP 155, 228.          [analytic cut volume]
 [N1] Nangia, Griffiths, Patankar (2019) JCP 390, 582.    [moving CV momentum]

=============================================================================
PART 1 — WHY A NAIVE CONVERGENCE STUDY IS MEANINGLESS
=============================================================================
TWO independent error sources contaminate a measured "spatial order":

(a) REFERENCE ERROR.
    Initialising alpha by point-sampling the indicator, (r < R).float(),
    gives an O(h) L1 error in the initial volume fractions for a CURVED
    interface. If that field is also the reference, the measured error
    saturates at the reference error and the "order" is meaningless.
    FIX: sub-sampled (quadrature) initialisation, with the reference error
    MEASURED by comparing n_sub against 2*n_sub.
    NOTE: both tests here are REVERSIBLE (alpha(T) == alpha(0)), so the
    reference error CANCELS EXACTLY in the comparison. That is precisely
    why these are the standard VOF verification cases. Initialisation
    quality still affects the error MAGNITUDE (a rougher shape is harder
    to advect), so it is reported, not ignored.

(b) TEMPORAL ERROR.
    Refining h at fixed CFL makes dt ~ h, so the first-order-in-time
    advection error also scales as h. The measured "spatial" order then
    just reports the temporal order (~1).
    FIX: two SEPARATE sweeps.
      spatial  : dt FIXED across grids (small enough to be stable at the
                 finest grid). If the error saturates, that floor is the
                 temporal error and is reported as such.
      temporal : h FIXED, dt halved. Reports the temporal order.

ANOTHER TRAP: axis-aligned uniform translation makes the PLIC flux EXACT
for a planar interface normal to the flow, producing artificial
super-convergence. The translation velocity here is OBLIQUE (1, 0.5, 0.5).

=============================================================================
PART 2 — FULL GEOMETRIC SUB-CELL MOMENTUM
=============================================================================
The previous scheme interpolated the CELL-based mass flux onto the
momentum-CV faces. That is consistent but not geometric: it never
reconstructs the interface on the shifted control volume.

This module does the full construction [R2,R3,N1]:

  1. Reconstruct the PLIC plane in every CELL (normal + offset).
  2. Build alpha on the MOMENTUM CV GEOMETRICALLY, by clipping each cell's
     own plane to the half-box that overlaps the shifted CV:
        alpha_u[i] = V_plane(cell i-1, HIGH half) + V_plane(cell i, LOW half)
     Both terms are analytic cut volumes in a box of side 1/2 along the
     shift axis, so alpha_u is in [0,1] by construction.
     This is NOT the arithmetic mean 0.5*(alpha[i-1]+alpha[i]): that is
     exact only for piecewise-CONSTANT alpha and discards the sub-cell
     interface position that PLIC carries.
  3. Reconstruct PLIC AGAIN on that shifted field (its own normals/offsets).
  4. Compute geometric swept-volume fluxes on the MOMENTUM-CV faces, using
     the velocity interpolated to those face locations:
        x-faces of the u-CV sit at CELL CENTRES  -> 0.5*(u[i]+u[i+1])
        y-faces of the u-CV sit at XY-EDGES      -> 0.5*(v[i-1,j]+v[i,j])
  5. Mass flux from THAT geometric flux; momentum and face density both
     advanced with the SAME flux:
        rho_f_new   = rho_f_old - dt * div_CV(M_CV)
        (rho u)_new = rho_f_old*u - dt * div_CV(M_CV * u_donor)
        u_new       = (rho u)_new / rho_f_new
     which makes u == const give (rho u)_new == u * rho_f_new EXACTLY.

WHERE CONSERVATION COMES FROM
  The flux-DIFFERENCE form telescopes. Not from PLIC. PLIC improves the
  accuracy of the flux value; the difference form gives conservation.

=============================================================================
HONEST SCOPE
=============================================================================
❌ Step 3 RE-RECONSTRUCTS a plane from the shifted alpha. The shifted alpha
   is exact for the piecewise-planar cell reconstruction, but fitting a NEW
   plane to it introduces its own O(h^2) interface-position error. This is
   an approximation, not an exact geometric transport of momentum.
❌ Operator-split advection is FIRST ORDER IN TIME. Measured temporal order
   is expected near 1, and the script PRINTS it rather than claiming 2.
❌ Measured spatial order for PLIC + split advection is typically between
   1 and 2 and is often NOT clean. The script reports the number and flags
   saturation instead of asserting a design order.
❌ Cost: the geometric momentum path adds 3 shifted reconstructions and 9
   flux evaluations per step (~4x the alpha-advection cost). Reported.
❌ No surface tension, no viscosity in this module. No experimental data.
=============================================================================
"""
from __future__ import annotations
import argparse, math, time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

VERSION = "1.0.0"


# =============================================================================
# GRID + OPERATORS
# =============================================================================

@dataclass
class Grid:
    nx: int = 32; ny: int = 32; nz: int = 32
    Lx: float = 1.0; Ly: float = 1.0; Lz: float = 1.0
    periodic: Tuple[bool, bool, bool] = (True, True, True)
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
    def bounds(self, ax):
        """Cell face coordinates along ax, length n_ax+1."""
        return torch.arange(self.n[ax]+1, device=self.dev, dtype=self.dt_)*self.h[ax]


def sl(ax, idx):
    o = [slice(None)]*3; o[ax] = idx; return tuple(o)


def diff(f, ax, h):
    return (f[sl(ax, slice(1, None))] - f[sl(ax, slice(0, -1))])/h


def pad1(f, ax, periodic, antisym=False):
    n = f.shape[ax]; d = f.device
    if periodic:
        lo = f.index_select(ax, torch.tensor([n-1], device=d))
        hi = f.index_select(ax, torch.tensor([0], device=d))
    else:
        lo = f.index_select(ax, torch.tensor([0], device=d))
        hi = f.index_select(ax, torch.tensor([n-1], device=d))
        if antisym: lo, hi = -lo, -hi
    return torch.cat([lo, f, hi], dim=ax)


def pad_cc(f, g: Grid, antisym=False):
    out = f
    for ax in range(3): out = pad1(out, ax, g.periodic[ax], antisym)
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


def grad_faces(p, g: Grid):
    pp = pad_cc(p, g); out = []
    for ax in range(3):
        gx = diff(pp, ax, g.h[ax])
        for o in range(3):
            if o != ax: gx = gx[sl(o, slice(1, -1))]
        assert gx.shape == g.face(ax)
        out.append(gx)
    return tuple(out)


def divergence(u, v, w, g: Grid):
    return diff(u, 0, g.h[0]) + diff(v, 1, g.h[1]) + diff(w, 2, g.h[2])


def cc_to_face(f, g: Grid, ax):
    b = torch.zeros(g.face(ax), device=f.device, dtype=f.dtype)
    b[sl(ax, slice(1, -1))] = avg_pair(f, ax)
    if g.periodic[ax]:
        m = 0.5*(f[sl(ax, 0)] + f[sl(ax, -1)])
        b[sl(ax, 0)] = m; b[sl(ax, -1)] = m
    else:
        b[sl(ax, 0)] = f[sl(ax, 0)]; b[sl(ax, -1)] = f[sl(ax, -1)]
    return b


def face_beta(rho, g: Grid):
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
        if g.periodic[ax]:
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
        ns = ((m1 < cls._TOL).to(torch.int8) + (m2 < cls._TOL).to(torch.int8)
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
    def plane_offset(cls, alpha, n, iters: int = 24):
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
        return d + shift, m          # FOLDED offset, ready for slab clipping

    @classmethod
    def slab_volume(cls, d_fold, m, s, axis, high_side: bool):
        """
        Cut volume inside a slab of thickness s along `axis`.
        high_side=True  : slab occupies [1-s, 1]  (donor's high side)
        high_side=False : slab occupies [0, s]    (donor's low side)
        """
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
        fL = cls.slab_volume(dL, mL, s, ax, high_side=True)
        fR = cls.slab_volume(dR, mR, s, ax, high_side=False)
        flux = torch.where(vel > 0, fL, -fR)
        if not periodic:
            msk = torch.ones_like(flux)
            msk[sl(ax, 0)] = 0.; msk[sl(ax, nf-1)] = 0.
            flux = flux*msk
        return flux


def plic_advect(alpha, u, v, w, dt, g: Grid, sweep0=0, iters=24):
    """Operator-split geometric advection with bounded mass repair."""
    a = alpha; m0 = float(a.sum()); Fs = [None]*3
    vel = (u, v, w)
    for ax in [(sweep0+k) % 3 for k in range(3)]:
        n = PLIC.normals(a, g)
        d_fold, m = PLIC.plane_offset(a, n, iters)
        Fa = PLIC.volume_flux(d_fold, m, vel[ax], dt, g.h[ax], ax, g.periodic[ax])
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
# PART 2 — FULL GEOMETRIC SUB-CELL MOMENTUM ON THE MOMENTUM CV
# =============================================================================

def shifted_alpha_geometric(d_fold, m, axis, g: Grid, iters=24):
    """
    alpha on the momentum CV shifted by half a cell along `axis`,
    built GEOMETRICALLY from each cell's own PLIC plane.

        alpha_cv[i] = V_plane(cell i-1, HIGH half) + V_plane(cell i, LOW half)

    Both terms are cut volumes in a box of side 1/2 along `axis`, so the
    sum lies in [0,1]. This is NOT 0.5*(alpha[i-1]+alpha[i]) -- that is
    exact only for piecewise-constant alpha and throws away the sub-cell
    interface position that PLIC carries.
    """
    s = torch.full_like(d_fold, 0.5)
    Vhi = PLIC.slab_volume(d_fold, m, s, axis, high_side=True)
    Vlo = PLIC.slab_volume(d_fold, m, s, axis, high_side=False)
    nf = g.n[axis]+1
    idx = torch.arange(nf, device=d_fold.device)
    hi_d = take_wrap(Vhi, axis, idx-1, g.periodic[axis])   # donor cell i-1
    lo_d = take_wrap(Vlo, axis, idx,   g.periodic[axis])   # donor cell i
    return (hi_d + lo_d).clamp(0., 1.)


def cv_face_velocity(vel, a, d, g: Grid):
    """
    Velocity component `d` at the (a-CV, d) face location.

    d == a : the a-CV's d-faces sit at CELL CENTRES along a
             -> 0.5*(comp[i] + comp[i+1])   (n_a+2 values incl. ends)
    d != a : they sit at (a-face, d-face) EDGES
             -> average comp along d, then along a
    """
    comp = vel[d]
    if d == a:
        p = pad1(comp, a, g.periodic[a])
        return avg_pair(p, a)                     # n_a+2
    edge = avg_pair(pad1(comp, d, g.periodic[d]), d)
    p = pad1(edge, a, g.periodic[a])
    return avg_pair(p, a)                          # n_a+1 along a, n_d+1 along d


def geometric_momentum_step(rho_cell, vel, dt, g: Grid,
                            order: str = "limited", iters: int = 24,
                            rho_floor: float = 1e-6):
    """
    FULL geometric sub-cell momentum transport on each momentum CV.

    Per component a:
      1. cell PLIC -> shifted alpha_a (geometric half-box clipping)
      2. PLIC AGAIN on alpha_a (own normals/offsets)
      3. geometric swept-volume flux on the a-CV faces, with cv_face_velocity
      4. M_CV = rho2*u_cv + (rho1-rho2)*(F*h/dt)   [handled by caller via
         the returned flux; here we take rho1/rho2 through `rho_cell` mix]
      5. rho_f and (rho u) advanced with the SAME M_CV -> exact consistency

    Returns (rho_cell_new, rho_face_new, rhou_new, u_new, diagnostics).
    """
    # cell-level advection supplies the reference cell density update
    n_cell = PLIC.normals(rho_cell*0.0 + 1.0, g)   # placeholder, unused
    raise RuntimeError("use geometric_momentum_from_alpha()")


def geometric_momentum_from_alpha(alpha, vel, rho1, rho2, dt, g: Grid,
                                  order="limited", iters=24,
                                  rho_floor=1e-6):
    """
    Full geometric momentum transport driven by the CELL alpha field.

    alpha : cell volume fraction at time n
    vel   : MAC face velocities at time n
    Returns dict with alpha_new, rho_cell_new, rho_face_new, rhou_new, u_new,
    and a cost/consistency diagnostic.
    """
    t0 = time.perf_counter()
    # ---- cell-level PLIC + alpha advection (reference mass transport) ----
    alpha_new, Fs_cell = plic_advect(alpha, *vel, dt, g, 0, iters)
    rho_cell = rho2 + (rho1-rho2)*alpha
    rho_cell_new = rho2 + (rho1-rho2)*alpha_new

    n = PLIC.normals(alpha, g)
    d_fold, m = PLIC.plane_offset(alpha, n, iters)

    n_recon = 1
    n_flux = 3
    rho_f_new, rhou_new, u_new = [], [], []
    for a in range(3):
        # 1-2. geometric shifted alpha + its OWN reconstruction
        a_cv = shifted_alpha_geometric(d_fold, m, a, g, iters)
        n_cv = PLIC.normals(a_cv, g)
        d_cv, m_cv = PLIC.plane_offset(a_cv, n_cv, iters)
        n_recon += 1

        # 3-4. geometric flux on the a-CV faces + mass flux
        Mcv = []
        for d in range(3):
            u_cv = cv_face_velocity(vel, a, d, g)
            hd = g.h[d]
            F = PLIC.volume_flux(d_cv, m_cv, u_cv, dt, hd, d, g.periodic[d])
            Mcv.append(rho2*u_cv + (rho1-rho2)*(F*hd/dt))
            n_flux += 1

        rho_f_old = cc_to_face(rho_cell, g, a)
        divM = sum(diff(Mcv[d], d, g.h[d]) for d in range(3))
        rf_new = (rho_f_old - dt*divM).clamp_min(rho_floor)

        # 5. momentum with the SAME flux
        tot = torch.zeros_like(vel[a])
        for d in range(3):
            lo, hi = _donor_pair(vel[a], a, d, g, order)
            donor = torch.where(Mcv[d] > 0, lo, hi)
            flux = Mcv[d]*donor
            if not g.periodic[d]:
                flux = flux.clone()
                flux[sl(d, 0)] = 0.; flux[sl(d, flux.shape[d]-1)] = 0.
            tot = tot + diff(flux, d, g.h[d])
        ru = rho_f_old*vel[a] - dt*tot
        rhou_new.append(ru); rho_f_new.append(rf_new)
        u_new.append(ru/rf_new)

    u_new = enforce_bc(*u_new, g)
    diag = {"n_plic_reconstructions": n_recon, "n_flux_evaluations": n_flux,
            "elapsed_ms": (time.perf_counter()-t0)*1e3}
    return dict(alpha=alpha_new, rho_cell=rho_cell_new,
                rho_face=tuple(rho_f_new), rhou=tuple(rhou_new),
                u=tuple(u_new), diag=diag)


def _donor_pair(comp, a, d, g: Grid, order):
    """Donor velocity at the (a-CV, d) face; upwind or minmod-limited linear."""
    if d == a:
        base = comp
        p = pad1(base, a, g.periodic[a])
        fm = p[sl(a, slice(0, -2))]; f0 = p[sl(a, slice(1, -1))]
        fp = p[sl(a, slice(2, None))]
        slope = minmod(f0-fm, fp-f0)
        idx = torch.arange(g.n[a]+2, device=comp.device)
        vl = take_wrap(base, a, idx-1, g.periodic[a])
        vh = take_wrap(base, a, idx,   g.periodic[a])
        sll = take_wrap(slope, a, idx-1, g.periodic[a])
        slh = take_wrap(slope, a, idx,   g.periodic[a])
    else:
        edge = avg_pair(pad1(comp, d, g.periodic[d]), d)
        p = pad1(edge, d, g.periodic[d])
        fm = p[sl(d, slice(0, -2))]; f0 = p[sl(d, slice(1, -1))]
        fp = p[sl(d, slice(2, None))]
        slope = minmod(f0-fm, fp-f0)
        idx = torch.arange(g.n[d]+1, device=comp.device)
        vl = take_wrap(edge, d, idx-1, g.periodic[d])
        vh = take_wrap(edge, d, idx,   g.periodic[d])
        sll = take_wrap(slope, d, idx-1, g.periodic[d])
        slh = take_wrap(slope, d, idx,   g.periodic[d])
    if order == "upwind":
        return vl, vh
    return vl + 0.5*sll, vh - 0.5*slh


# =============================================================================
# ACCURATE REFERENCE INITIALISATION (quadrature, not point sampling)
# =============================================================================

def sphere_vf(g: Grid, center, R, n_sub: int = 16) -> Tensor:
    """
    Volume fraction of a sphere per cell by uniform sub-cell quadrature.

    Point sampling ((r<R).float()) has O(h) L1 error for a curved interface.
    With n_sub^3 samples per INTERFACE cell the initialisation error is
    reduced by ~n_sub; it is MEASURED by comparing n_sub vs 2*n_sub.
    Only interface cells are sub-sampled (interior/exterior are exact).
    """
    X, Y, Z = g.coords()
    r2 = (X-center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2
    hmin = min(g.h)
    inner = (r2 < (R-1.0*hmin)**2)
    outer = (r2 > (R+1.0*hmin)**2)
    a = torch.zeros(g.n, device=g.dev, dtype=torch.float64)
    a[inner] = 1.0
    iface = ~(inner | outer)
    idx = iface.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return a
    # sub-cell offsets in [-0.5, 0.5)
    t = (torch.arange(n_sub, device=g.dev, dtype=torch.float64)+0.5)/n_sub - 0.5
    ox, oy, oz = torch.meshgrid(t, t, t, indexing="ij")
    off = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], -1)
    hs = torch.tensor(g.h, device=g.dev, dtype=torch.float64)
    base = (idx.to(torch.float64)+0.5)*hs
    CH = 512
    for s in range(0, idx.shape[0], CH):
        b = base[s:s+CH]
        pts = b[:, None, :] + off[None, :, :]*hs[None, None, :]
        d2 = ((pts - torch.tensor(center, device=g.dev, dtype=torch.float64))**2).sum(-1)
        frac = (d2 < R*R).to(torch.float64).mean(-1)
        ii = idx[s:s+CH]
        a[ii[:, 0], ii[:, 1], ii[:, 2]] = frac
    return a


def reference_error(g: Grid, center, R, n_sub=16) -> Dict[str, float]:
    a1 = sphere_vf(g, center, R, n_sub)
    a2 = sphere_vf(g, center, R, 2*n_sub)
    d = (a1-a2).abs()
    return {"l1": float(d.sum()/a2.sum().clamp_min(1e-30)),
            "linf": float(d.max()), "n_sub": n_sub}


# =============================================================================
# TEST CASES (both REVERSIBLE: alpha(T) == alpha(0) exactly)
# =============================================================================

@dataclass
class TestCase:
    name: str
    T: float
    velocity: Callable           # (g, t) -> (u,v,w) MAC faces
    center: Tuple[float, float, float] = (0.35, 0.5, 0.5)
    R: float = 0.15


def oblique_translation(g: Grid, t: float):
    """u = (1, 0.5, 0.5); T=2 -> displacement (2,1,1), all integer periods."""
    return (g.fc(0, 1.0), g.fc(1, 0.5), g.fc(2, 0.5))


def leveque_deformation(g: Grid, t: float, T: float = 1.0):
    """
    LeVeque (1996) 3D divergence-free deformation field, time-reversed:
      u =  2 sin^2(pi x) sin(2pi y) sin(2pi z) cos(pi t/T)
      v = -  sin(2pi x) sin^2(pi y) sin(2pi z) cos(pi t/T)
      w = -  sin(2pi x) sin(2pi y) sin^2(pi z) cos(pi t/T)
    div = 2pi - pi - pi = 0 exactly. Returns to alpha(0) at t=T.
    """
    c = math.cos(math.pi*t/T)
    xf = g.bounds(0); yf = g.bounds(1); zf = g.bounds(2)
    xc, yc, zc = g.coords()
    s2 = lambda v: torch.sin(math.pi*v)**2
    sn = lambda v: torch.sin(2*math.pi*v)
    # u on x-faces
    Xf, Yc, Zc = torch.meshgrid(xf, yc, zc, indexing="ij")
    u = 2*s2(Xf)*sn(Yc)*sn(Zc)*c
    Xc, Yf, Zc2 = torch.meshgrid(xc, yf, zc, indexing="ij")
    v = -sn(Xc)*s2(Yf)*sn(Zc2)*c
    Xc2, Yc2, Zf = torch.meshgrid(xc, yc, zf, indexing="ij")
    w = -sn(Xc2)*sn(Yc2)*s2(Zf)*c
    return (u.to(g.dt_), v.to(g.dt_), w.to(g.dt_))


def make_case(name: str) -> TestCase:
    if name == "translation":
        return TestCase("oblique_translation", 2.0, oblique_translation)
    if name == "deformation":
        return TestCase("leveque_deformation", 1.0,
                        lambda g, t: leveque_deformation(g, t, 1.0),
                        center=(0.35, 0.35, 0.35), R=0.15)
    raise ValueError(name)


# =============================================================================
# ERROR NORMS
# =============================================================================

def error_norms(a_num: Tensor, a_ref: Tensor) -> Dict[str, float]:
    d = (a_num - a_ref).abs()
    vref = float(a_ref.sum())
    return {
        "l1": float(d.sum()/max(vref, 1e-30)),
        "l2": float(torch.sqrt((d*d).sum()/max((a_ref*a_ref).sum(), 1e-30))),
        "linf": float(d.max()),
        "vol_drift": float(abs(a_num.sum()-a_ref.sum())/max(vref, 1e-30)),
        "amin": float(a_num.min()), "amax": float(a_num.max()),
    }


def order_from(e_coarse: float, e_fine: float, ratio: float = 2.0) -> float:
    if e_fine <= 0 or e_coarse <= 0: return float("nan")
    return math.log(e_coarse/e_fine)/math.log(ratio)


# =============================================================================
# RUNNERS
# =============================================================================

def run_case(g: Grid, case: TestCase, dt: float, n_sub=16,
             iters=24, verbose=False) -> Dict:
    a0 = sphere_vf(g, case.center, case.R, n_sub)
    a = a0.clone()
    nsteps = int(round(case.T/dt))
    dt_eff = case.T/nsteps
    t0 = time.perf_counter()
    for k in range(nsteps):
        t = k*dt_eff
        vel = case.velocity(g, t)
        a, _ = plic_advect(a, *vel, dt_eff, g, k % 3, iters)
    el = time.perf_counter()-t0
    en = error_norms(a, a0)
    en.update(n_steps=nsteps, dt=dt_eff, N=g.nx, elapsed_s=el)
    if verbose:
        print(f"      N={g.nx:3d} dt={dt_eff:.2e} steps={nsteps:5d} "
              f"L1={en['l1']:.4e} L2={en['l2']:.4e} "
              f"vol={en['vol_drift']:.2e} {el:.1f}s")
    return en


def spatial_sweep(case_name="translation", grids=(16, 32), dt=5e-3,
                  n_sub=16, iters=24, device="cpu") -> Dict:
    """
    SPATIAL order: dt FIXED across grids so the temporal error is a constant
    floor. If the error stops decreasing, that floor is reported explicitly
    rather than being mislabelled as a spatial order.
    """
    case = make_case(case_name)
    print(f"\n  SPATIAL SWEEP — {case.name}, dt FIXED = {dt:.2e}")
    ref = {}
    rows = []
    for N in grids:
        g = Grid(nx=N, ny=N, nz=N, device=device)
        cfl = max(float(v.abs().max()) for v in case.velocity(g, 0.0))*dt/min(g.h)
        if cfl > 0.5:
            print(f"    [SKIP] N={N}: CFL={cfl:.2f} > 0.5 at this dt. "
                  f"Reduce --dt or drop this grid.")
            continue
        ref[N] = reference_error(g, case.center, case.R, n_sub)
        print(f"    reference init error (n_sub={n_sub} vs {2*n_sub}): "
              f"L1={ref[N]['l1']:.2e}")
        rows.append(run_case(g, case, dt, n_sub, iters, verbose=True))
    orders = []
    for i in range(1, len(rows)):
        p = order_from(rows[i-1]["l1"], rows[i]["l1"],
                       rows[i]["N"]/rows[i-1]["N"])
        orders.append({"from": rows[i-1]["N"], "to": rows[i]["N"], "p_l1": p})
        print(f"    order L1  N {rows[i-1]['N']} -> {rows[i]['N']}: p = {p:.3f}")
    # saturation detection
    saturated = False
    if len(rows) >= 2:
        finest_ref = ref[rows[-1]["N"]]["l1"]
        if rows[-1]["l1"] < 5*finest_ref:
            saturated = True
            print(f"    ⚠️ finest-grid error {rows[-1]['l1']:.2e} is within 5x of "
                  f"the reference error {finest_ref:.2e} — order may be "
                  f"reference-limited, not scheme-limited.")
        if len(rows) >= 3 and orders[-1]["p_l1"] < 0.3*orders[0]["p_l1"]:
            saturated = True
            print("    ⚠️ order collapsed at the finest pair — likely the FIXED-dt "
                  "temporal error floor. Run the temporal sweep to confirm.")
    return {"case": case.name, "dt": dt, "rows": rows, "orders": orders,
            "reference_error": ref, "saturated": saturated}


def temporal_sweep(case_name="translation", N=32, dts=(2e-2, 1e-2, 5e-3),
                   n_sub=16, iters=24, device="cpu") -> Dict:
    """TEMPORAL order: h FIXED, dt halved."""
    case = make_case(case_name)
    g = Grid(nx=N, ny=N, nz=N, device=device)
    print(f"\n  TEMPORAL SWEEP — {case.name}, N FIXED = {N}")
    rows = []
    for dt in dts:
        cfl = max(float(v.abs().max()) for v in case.velocity(g, 0.0))*dt/min(g.h)
        if cfl > 0.5:
            print(f"    [SKIP] dt={dt:.2e}: CFL={cfl:.2f} > 0.5")
            continue
        rows.append(run_case(g, case, dt, n_sub, iters, verbose=True))
    orders = []
    for i in range(1, len(rows)):
        p = order_from(rows[i-1]["l1"], rows[i]["l1"],
                       rows[i-1]["dt"]/rows[i]["dt"])
        orders.append({"from": rows[i-1]["dt"], "to": rows[i]["dt"], "p_l1": p})
        print(f"    order L1  dt {rows[i-1]['dt']:.1e} -> {rows[i]['dt']:.1e}: "
              f"p = {p:.3f}")
    return {"case": case.name, "N": N, "rows": rows, "orders": orders}


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test(device="cpu", quick=True) -> bool:
    print("="*78)
    print(f"  VOF CONVERGENCE + GEOMETRIC MOMENTUM v{VERSION}")
    print("="*78)
    torch.manual_seed(0); ok = True

    def rep(n, p, e=""):
        nonlocal ok; ok &= bool(p); print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    N = 16
    g = Grid(nx=N, ny=N, nz=N, device=device)

    print("\n[T1] PLIC analytic cut volume + complement symmetry")
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

    print("\n[T2] quadrature init beats point sampling")
    a_pt = ((g.coords()[0]-0.35)**2 + (g.coords()[1]-0.5)**2
            + (g.coords()[2]-0.5)**2 < 0.15**2).double()
    a16 = sphere_vf(g, (0.35, 0.5, 0.5), 0.15, 16)
    a32 = sphere_vf(g, (0.35, 0.5, 0.5), 0.15, 32)
    e_pt = float((a_pt-a32).abs().sum()/a32.sum())
    e_16 = float((a16-a32).abs().sum()/a32.sum())
    rep("sub-sampled closer to converged reference", e_16 < e_pt,
        f"point-sampled L1={e_pt:.3e}  n_sub=16 L1={e_16:.3e}")

    print("\n[T3] shifted geometric alpha != arithmetic mean, and in [0,1]")
    a = sphere_vf(g, (0.35, 0.5, 0.5), 0.15, 16)
    n = PLIC.normals(a, g); d_fold, m = PLIC.plane_offset(a, n)
    a_cv = shifted_alpha_geometric(d_fold, m, 0, g)
    a_ar = cc_to_face(a, g, 0)
    dd = float((a_cv-a_ar).abs().max())
    rep("geometric differs from arithmetic", dd > 1e-6, f"max diff={dd:.3e}")
    rep("alpha_cv in [0,1]", float(a_cv.min()) >= 0 and float(a_cv.max()) <= 1,
        f"[{float(a_cv.min()):.3f},{float(a_cv.max()):.3f}]")
    rep("alpha_cv shape", a_cv.shape == g.face(0), f"{tuple(a_cv.shape)}")

    print("\n[T4] geometric momentum: u=const -> (rho u) == u * rho_f EXACTLY")
    rho1, rho2 = 1000.0, 1.0
    uu, vv, ww = g.fc(0, 0.7), g.fc(1, -0.3), g.fc(2, 0.2)
    dt = 0.2*min(g.h)
    for order in ("upwind", "limited"):
        out = geometric_momentum_from_alpha(a, (uu, vv, ww), rho1, rho2, dt, g,
                                            order=order)
        errs = [float((out["rhou"][i] - (uu, vv, ww)[i]*out["rho_face"][i]).abs().max())
                for i in range(3)]
        rep(f"consistency ({order})", max(errs) < 1e-8,
            f"max|(rho u) - u*rho_f| = {max(errs):.2e}")

    print("\n[T5] geometric momentum conserves sum(rho u) (periodic)")
    out = geometric_momentum_from_alpha(a, (uu, vv, ww), rho1, rho2, dt, g)
    rho_c = rho2 + (rho1-rho2)*a
    P0 = [float((cc_to_face(rho_c, g, i)*(uu, vv, ww)[i]).sum()) for i in range(3)]
    P1 = [float(out["rhou"][i].sum()) for i in range(3)]
    rel = max(abs(P1[i]-P0[i])/max(abs(P0[i]), 1e-30) for i in range(3))
    rep("momentum conserved", rel < 1e-10, f"max rel change={rel:.2e}")
    m0 = float(a.sum()); m1 = float(out["alpha"].sum())
    rep("alpha mass conserved", abs(m1-m0)/m0 < 1e-11, f"rel={abs(m1-m0)/m0:.2e}")
    print(f"      cost: {out['diag']['n_plic_reconstructions']} reconstructions, "
          f"{out['diag']['n_flux_evaluations']} flux evals, "
          f"{out['diag']['elapsed_ms']:.1f} ms")

    print("\n[T6] LeVeque deformation field is divergence-free")
    for t in (0.0, 0.25, 0.5):
        u_, v_, w_ = leveque_deformation(g, t, 1.0)
        dv = float(divergence(u_, v_, w_, g).abs().max())
        rep(f"div-free at t={t}", dv < 1e-12, f"max|div|={dv:.2e}")

    print("\n[T7] gradient flows through the geometric momentum path")
    ag = a.clone().requires_grad_(True)
    og = geometric_momentum_from_alpha(ag, (uu, vv, ww), rho1, rho2, dt, g)
    (og["u"][0].pow(2).sum() + og["alpha"].sum()).backward()
    rep("d/d(alpha) finite and non-zero",
        ag.grad is not None and bool(torch.isfinite(ag.grad).all())
        and float(ag.grad.abs().sum()) > 0)

    if not quick:
        print("\n[T8] MEASURED convergence (this is the actual study)")
        sp = spatial_sweep("translation", grids=(16, 32), dt=5e-3, device=device)
        tp = temporal_sweep("translation", N=16, dts=(2e-2, 1e-2, 5e-3),
                            device=device)
        sp2 = spatial_sweep("deformation", grids=(16, 32), dt=5e-3, device=device)
        print("\n  HONEST SUMMARY")
        for tag, s in (("translation spatial", sp), ("deformation spatial", sp2)):
            if s["orders"]:
                ps = ", ".join(f"{o['p_l1']:.2f}" for o in s["orders"])
                print(f"    {tag:<24} measured p_L1 = {ps}"
                      f"{'  [SATURATED]' if s['saturated'] else ''}")
        if tp["orders"]:
            ps = ", ".join(f"{o['p_l1']:.2f}" for o in tp["orders"])
            print(f"    {'translation temporal':<24} measured p_L1 = {ps}")
        print("    Split PLIC advection is FIRST order in time; a temporal")
        print("    order near 1 is EXPECTED, not a defect.")
        rep("convergence study produced numbers",
            bool(sp["orders"] or tp["orders"]))

    print("\n"+"="*78)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*78)
    print("""
  PROVEN
    PLIC cut volume analytic + symmetric
    quadrature init measurably closer to the converged reference than
      point sampling (which is O(h) for a curved interface)
    shifted alpha is GEOMETRIC, differs from the arithmetic mean, in [0,1]
    u=const gives (rho u) == u*rho_f EXACTLY  <- discrete consistency
    sum(rho u) conserved by flux telescoping
    LeVeque deformation field divergence-free to round-off

  MEASURED, NOT CLAIMED  (run with --study)
    spatial order at FIXED dt, with reference-error saturation flagged
    temporal order at FIXED h

  NOT DONE
    re-reconstructing a plane on the shifted CV is itself an approximation
    split advection is first order in time
    no surface tension, no viscosity, no experimental validation
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "study"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--grids", type=int, nargs="*", default=[16, 32])
    p.add_argument("--dt", type=float, default=5e-3)
    p.add_argument("--case", default="translation",
                   choices=["translation", "deformation"])
    p.add_argument("--n-sub", dest="n_sub", type=int, default=16)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if self_test(a.device, quick=True) else 1)
    print("="*78)
    print("  CONVERGENCE STUDY — measured orders, reference error reported")
    print("="*78)
    if max(a.grids) >= 64:
        print("  ⚠️ N>=64 with PLIC bisection is expensive (minutes to tens of")
        print("     minutes on CPU). Cost scales ~ N^3 * n_steps * iters.")
    sp = spatial_sweep(a.case, tuple(a.grids), a.dt, a.n_sub, device=a.device)
    tp = temporal_sweep(a.case, N=min(a.grids), dts=(4*a.dt, 2*a.dt, a.dt),
                        n_sub=a.n_sub, device=a.device)
    ok = self_test(a.device, quick=True)
    print("\n" + "="*78)
    print("  FINAL HONEST REPORT")
    print("="*78)
    if sp["orders"]:
        print("  spatial  p_L1: " + ", ".join(f"{o['p_l1']:.3f}" for o in sp["orders"])
              + ("   [SATURATED — see flags above]" if sp["saturated"] else ""))
    if tp["orders"]:
        print("  temporal p_L1: " + ", ".join(f"{o['p_l1']:.3f}" for o in tp["orders"]))
    print("  Split PLIC is first order in time. A temporal order near 1 is")
    print("  expected. A spatial order between 1 and 2 is typical and is")
    print("  REPORTED here, not asserted from the scheme's design.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
