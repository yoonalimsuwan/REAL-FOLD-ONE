#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
real_cfd_adapter.py — REAL CFD SOLVE via MACOps + project()
Replaces the analytic Poiseuille ESTIMATE in med_world_model_connectors.py
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died on the Cross to teach: when struck on the left cheek, turn also
the right, so that we may listen to the voice of the enemy who wishes us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT IS REAL HERE
=============================================================================
[R1] Incompressible Navier-Stokes, fractional-step (Chorin 1968):
       u* = u + dt(-conv + visc/rho + body)
       div(beta grad p) = div(u*)/dt,   beta = 1/rho_face
       u^{n+1} = u* - dt*beta*grad(p)
     solved with one_cfd_stencils.MACOps + PressurePoisson + project().
     NO analytic formula is used to produce the reported metrics.

[R2] Correct staggered layout, ASSERTED by MACOps:
       u:(nx+1,ny,nz)  v:(nx,ny+1,nz)  w:(nx,ny,nz+1)  p,rho:(nx,ny,nz)

[R3] Correct face beta = ARITHMETIC MEAN OF RECIPROCALS (one_cfd_stencils
     fixes the inverted-projection bug in the older modules).

[R4] Real diagnostics from the solved field:
       flow rate, pressure drop, WSS vector + magnitude, OSI, residence time.

[R5] Implicit Brinkman penalisation for the immersed solid:
       u <- u / (1 + dt*solid_face/eta)   (unconditionally stable)

=============================================================================
HONEST LIMITATIONS — READ BEFORE QUANTITATIVE USE
=============================================================================
[L1] MACOps is STRUCTURED CARTESIAN. It cannot do body-fitted unstructured
     meshes. Geometry is therefore VOXELISED onto the MAC grid and imposed
     by Brinkman penalisation. Walls are staircased at grid scale and
     diffuse over ~2 cells. Effective slip length is O(sqrt(eta)).
     => WSS and pressure drop carry O(10-40%) error on coarse grids.
        MESH CONVERGENCE IS MANDATORY. Body-fitted CFD is required for
        clinical-grade wall shear stress.

[L2] FIRST-ORDER IN TIME (forward-Euler fractional step). CFL-limited.

[L3] Momentum advection is the CONVECTIVE form (u.grad)u with first-order
     upwind at the component's own staggered location. MASS is conserved
     by the projection; MOMENTUM is NOT discretely conserved.

[L4] WSS uses a voxelised wall-normal estimate with wall distance ~ dx/2.
     This is a first-order geometric approximation, not a body-fitted
     wall gradient.

[L5] NOT a medical device. NOT for diagnosis, treatment, or surgical
     decisions. Every output is research_only and requires clinician,
     IRB, and regulatory review, plus validation against 4D-flow MRI,
     Doppler, or invasive pressure.

[L6] Governance is ENFORCED: patient-derived geometry requires an
     authorised decision (IRB + de-identification) or the solve is refused.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("real_cfd")

VERSION = "1.0.0"

# ---- the REAL backend: one_cfd_stencils ----
_BACKEND_ERR = ""
try:
    from one_cfd_stencils import (
        GridSpec, MACOps, PressurePoisson, project,
        face_beta, face_scalar, pad_axis, diff, sl,
        advect_alpha_upwind, STENCIL_VERSION,
    )
    HAS_BACKEND = True
except Exception as _e:                      # noqa: BLE001
    HAS_BACKEND = False
    _BACKEND_ERR = f"{type(_e).__name__}: {_e}"
    STENCIL_VERSION = "unavailable"

BANNER = f"""
=============================================================================
REAL CFD ADAPTER v{VERSION}   (one_cfd_stencils {STENCIL_VERSION})
MACOps + PressurePoisson + project()  —  NO analytic estimate.
Voxelised Brinkman geometry: WSS error O(10-40%) on coarse grids.
RESEARCH ONLY. Not a medical device. Mesh convergence mandatory.
=============================================================================
"""


# =============================================================================
# §1. GEOMETRY / BC / REQUEST RECORDS
# =============================================================================

@dataclass
class SurfaceMesh:
    """Compatible with med_world_model_connectors.SurfaceMesh."""
    vertices_mm: np.ndarray            # (N,3)
    faces: np.ndarray                  # (M,3) int
    normals: Optional[np.ndarray] = None
    watertight: Optional[bool] = None
    quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class BoundaryConditionSet:
    """Inlet/outlet identified by PLANES (axis, coordinate_mm)."""
    inlet_plane: Tuple[int, float]
    outlet_planes: List[Tuple[int, float]]
    inlet_area_mm2: float = 0.0
    outlet_area_mm2: float = 0.0
    flow_split: Dict[str, float] = field(default_factory=dict)
    waveform: Optional[np.ndarray] = None      # (T,) m/s, optional
    waveform_dt_s: float = 0.0
    assumptions: List[str] = field(default_factory=list)
    validation_required: List[str] = field(default_factory=list)


@dataclass
class RealCFDRequest:
    organ: str
    research_question: str
    mesh: Optional[SurfaceMesh] = None
    bc: Optional[BoundaryConditionSet] = None
    # fluid
    rho: float = 1060.0                # kg/m^3 (blood)
    mu: float = 3.5e-3                 # Pa s
    inlet_velocity_m_s: float = 0.2
    flow_rate_ml_s: Optional[float] = None
    # grid / numerics
    nx: int = 32
    ny: int = 32
    nz: int = 64
    axis_flow: int = 2                 # mesh axis mapped to grid z
    brinkman_eta: float = 1e-4
    cfl: float = 0.35
    viscous_factor: float = 0.25
    dt_max: float = 5e-3
    poisson_tol: float = 1e-7
    poisson_max_iter: int = 2000
    n_steps: int = 400
    steady_tol: float = 1e-6
    dtype: str = "float64"
    device: str = "cpu"
    compute_osi: bool = True
    compute_residence: bool = True
    # governance
    uses_patient_data: bool = False
    governance_decision: Optional[Dict[str, Any]] = None


@dataclass
class CFDResult:
    status: str
    backend: str
    metrics: Dict[str, float]
    fields_available: List[str]
    convergence: Dict[str, Any]
    assumptions: List[str]
    limitations: List[str]
    validation_required: List[str]
    clinical_actionability: str = "research_only"
    clinician_review_required: bool = True


# =============================================================================
# §2. VOXELISATION  (mesh -> MAC solid mask)
# =============================================================================

def voxelize_closed_mesh(mesh: SurfaceMesh,
                         gx: np.ndarray, gy: np.ndarray, gz: np.ndarray
                         ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ray-parity voxelisation along z. For each (i,j) column, collect the z
    crossings of all triangles whose xy-projection contains the column
    centre, sort them, and fill between consecutive pairs.

    Requires a CLOSED (watertight) mesh. Non-watertight input yields an
    odd crossing count in some columns; those columns are reported and
    left fluid (conservative: do not invent solid).
    """
    V = np.asarray(mesh.vertices_mm, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    nx, ny, nz = len(gx), len(gy), len(gz)
    dx = gx[1] - gx[0] if nx > 1 else 1.0
    dy = gy[1] - gy[0] if ny > 1 else 1.0

    crossings: List[List[float]] = [[] for _ in range(nx * ny)]
    degen = 0

    for tri in F:
        p0, p1, p2 = V[tri[0]], V[tri[1]], V[tri[2]]
        ax, ay = p0[0], p0[1]
        bx, by = p1[0], p1[1]
        cx, cy = p2[0], p2[1]
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-18:
            degen += 1
            continue                      # vertical triangle: no z-crossing
        i0 = max(0, int(math.floor((min(ax, bx, cx) - gx[0]) / dx)))
        i1 = min(nx - 1, int(math.ceil((max(ax, bx, cx) - gx[0]) / dx)))
        j0 = max(0, int(math.floor((min(ay, by, cy) - gy[0]) / dy)))
        j1 = min(ny - 1, int(math.ceil((max(ay, by, cy) - gy[0]) / dy)))
        for i in range(i0, i1 + 1):
            px = gx[i]
            for j in range(j0, j1 + 1):
                py = gy[j]
                w1 = ((px - ax) * (cy - ay) - (cx - ax) * (py - ay)) / det
                w2 = ((bx - ax) * (py - ay) - (px - ax) * (by - ay)) / det
                w0 = 1.0 - w1 - w2
                if w0 < -1e-12 or w1 < -1e-12 or w2 < -1e-12:
                    continue
                z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
                crossings[i * ny + j].append(float(z))

    solid = np.zeros((nx, ny, nz), dtype=np.float64)
    odd_columns = 0
    for i in range(nx):
        for j in range(ny):
            zs = sorted(crossings[i * ny + j])
            if len(zs) % 2 != 0:
                odd_columns += 1
                continue
            for k in range(0, len(zs), 2):
                zlo, zhi = zs[k], zs[k + 1]
                m = (gz > zlo) & (gz < zhi)
                solid[i, j, m] = 1.0

    info = {
        "n_triangles": int(len(F)),
        "degenerate_triangles": int(degen),
        "odd_crossing_columns": int(odd_columns),
        "watertight_assumed": mesh.watertight is not False,
        "solid_fraction": float(solid.mean()),
        "method": "ray-parity along z + Brinkman penalisation",
    }
    return solid, info


def cylinder_phantom(nx: int, ny: int, nz: int,
                     Lx: float, Ly: float, Lz: float,
                     radius: float) -> Tuple[SurfaceMesh, np.ndarray]:
    """Analytic cylinder mesh + exact solid mask, for verification."""
    nseg, nzseg = 48, 24
    verts, faces = [], []
    for k in range(nzseg + 1):
        z = Lz * k / nzseg
        for s in range(nseg):
            th = 2 * math.pi * s / nseg
            verts.append([Lx / 2 + radius * math.cos(th),
                          Ly / 2 + radius * math.sin(th), z])
    for k in range(nzseg):
        for s in range(nseg):
            a = k * nseg + s
            b = k * nseg + (s + 1) % nseg
            c = (k + 1) * nseg + (s + 1) % nseg
            d = (k + 1) * nseg + s
            faces.append([a, b, c]); faces.append([a, c, d])
    # caps (fan)
    for cap, zoff in ((0, 0), (1, nzseg)):
        ctr = len(verts)
        verts.append([Lx / 2, Ly / 2, Lz * zoff / nzseg])
        base = zoff * nseg
        for s in range(nseg):
            a = base + s
            b = base + (s + 1) % nseg
            faces.append([ctr, b, a] if cap == 0 else [ctr, a, b])
    mesh = SurfaceMesh(np.array(verts), np.array(faces, dtype=np.int64),
                       watertight=True)
    gx = (np.arange(nx) + 0.5) * (Lx / nx)
    gy = (np.arange(ny) + 0.5) * (Ly / ny)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    inside = ((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) < radius ** 2
    solid = np.repeat((~inside)[:, :, None], nz, axis=2).astype(np.float64)
    return mesh, solid


# =============================================================================
# §3. THE REAL SOLVER
# =============================================================================

class RealMACSolver:
    """
    Incompressible NS on a MAC grid using one_cfd_stencils.
    Solid imposed by IMPLICIT Brinkman penalisation.
    """

    def __init__(self, spec: "GridSpec", solid: Tensor,
                 inlet_faces: Tensor, outlet_faces: Tensor,
                 rho: float, mu: float, brinkman_eta: float,
                 poisson_tol: float = 1e-7, poisson_max_iter: int = 2000):
        if not HAS_BACKEND:
            raise ImportError(f"one_cfd_stencils unavailable: {_BACKEND_ERR}")
        self.s = spec
        self.g = MACOps(spec)
        self.ps = PressurePoisson(spec, self.g, method="cg",
                                  tol=poisson_tol, max_iter=poisson_max_iter)
        self.solid = solid
        self.su, self.sv, self.sw = face_scalar(solid, spec)
        self.inlet = inlet_faces          # bool on x-faces (axis 0) OR z-faces
        self.outlet = outlet_faces
        self.rho0, self.mu0 = float(rho), float(mu)
        self.eta = max(float(brinkman_eta), 1e-30)
        self.u = self.g.zeros_u()
        self.v = self.g.zeros_v()
        self.w = self.g.zeros_w()
        self.p = self.g.cc()
        self.age = self.g.cc()
        self.t = 0.0
        self.step_count = 0
        self.wss_series: List[Tensor] = []

    # ---- helpers ----
    def _rho(self) -> Tensor:
        return self.g.cc(self.rho0)

    def _mu(self) -> Tensor:
        return self.g.cc(self.mu0)

    def _pad_own(self, f: Tensor, own_axis: int) -> Tensor:
        """Ghost halo at the component's OWN staggered location."""
        out = f
        for ax in range(3):
            antisym = (self.s.bc[ax] == "wall") and (ax != own_axis)
            out = pad_axis(out, ax, self.s.bc[ax], antisym=antisym)
        return out

    def _convective(self, comp: Tensor, own_axis: int,
                    vel_at_c: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """(u.grad)comp, first-order upwind, at the component's own faces."""
        p = self._pad_own(comp, own_axis)
        h = self.s.h
        out = torch.zeros_like(comp)
        for ax in range(3):
            f0 = p[sl(0, slice(1, -1))][sl(1, slice(1, -1))][sl(2, slice(1, -1))]
            fm = p[sl(ax, slice(0, -2))]
            fp = p[sl(ax, slice(2, None))]
            for o in range(3):
                if o != ax:
                    fm = fm[sl(o, slice(1, -1))]
                    fp = fp[sl(o, slice(1, -1))]
            d = torch.where(vel_at_c[ax] > 0,
                            (f0 - fm) / h[ax], (fp - f0) / h[ax])
            out = out + vel_at_c[ax] * d
        return out

    def _apply_inlet_outlet(self, u, v, w, u_in: float, axis_flow: int):
        """Prescribe inlet normal velocity; zero-gradient at outlet."""
        comps = [u, v, w]
        c = comps[axis_flow]
        c = torch.where(self.inlet, torch.full_like(c, float(u_in)), c)
        # outlet: copy nearest interior plane (zero normal gradient)
        n = c.shape[axis_flow]
        idx_last = torch.tensor([n - 1], device=c.device)
        idx_int = torch.tensor([n - 2], device=c.device)
        interior = c.index_select(axis_flow, idx_int)
        c = c.index_copy(axis_flow, idx_last, interior)
        comps[axis_flow] = c
        return comps[0], comps[1], comps[2]

    def compute_dt(self) -> float:
        umax = max(float(self.u.abs().max()), float(self.v.abs().max()),
                   float(self.w.abs().max()), 1e-12)
        hmin = min(self.s.h)
        nu = self.mu0 / self.rho0
        dt_adv = 0.35 * hmin / umax
        dt_vis = 0.25 * hmin ** 2 / max(nu, 1e-30)
        return float(min(dt_adv, dt_vis, 5e-3))

    def step(self, dt: Optional[float] = None, u_in: float = 0.2,
             axis_flow: int = 2, store_wss: bool = False) -> Dict[str, float]:
        g, s = self.g, self.s
        dt = self.compute_dt() if dt is None else float(dt)
        rho, mu = self._rho(), self._mu()

        # 1. velocities interpolated to each component's own faces
        uc, vc, wc = g.to_cc(self.u, self.v, self.w)
        vel_at = []
        for c in range(3):
            vel_at.append(tuple(g.cc_to_face(t, c) for t in (uc, vc, wc)))

        # 2. explicit momentum: convection + viscous + body
        acc = []
        for c, comp in enumerate((self.u, self.v, self.w)):
            conv = self._convective(comp, c, vel_at[c])
            visc = g.viscous_component((self.u, self.v, self.w), mu, c)
            rf = face_scalar(rho, s)[c]
            acc.append(-conv + visc / rf.clamp_min(1e-30))
        us = self.u + dt * acc[0]
        vs = self.v + dt * acc[1]
        ws = self.w + dt * acc[2]

        # 3. IMPLICIT Brinkman penalisation (unconditionally stable)
        us = us / (1.0 + self.su * dt / self.eta)
        vs = vs / (1.0 + self.sv * dt / self.eta)
        ws = ws / (1.0 + self.sw * dt / self.eta)

        # 4. inlet / outlet
        us, vs, ws = self._apply_inlet_outlet(us, vs, ws, u_in, axis_flow)
        us, vs, ws = g.enforce_velocity_bc(us, vs, ws)

        # 5. REAL pressure projection (MACOps + project)
        un, vn, wn, p, self.ps = project(us, vs, ws, rho, s, dt, self.ps)
        un, vn, wn = self._apply_inlet_outlet(un, vn, wn, u_in, axis_flow)
        self.u, self.v, self.w, self.p = un, vn, wn, p

        # 6. residence-time (age) scalar: da/dt + u.grad a = 1
        ap = self._pad_own(self.age, 0)
        f0 = self.age
        adv = torch.zeros_like(self.age)
        for ax in range(3):
            fm = ap[sl(ax, slice(0, -2))]
            fp = ap[sl(ax, slice(2, None))]
            for o in range(3):
                if o != ax:
                    fm = fm[sl(o, slice(1, -1))]
                    fp = fp[sl(o, slice(1, -1))]
            vcc = (uc, vc, wc)[ax]
            d = torch.where(vcc > 0, (f0 - fm) / s.h[ax], (fp - f0) / s.h[ax])
            adv = adv + vcc * d
        self.age = (self.age + dt * (1.0 - adv)) * (1.0 - self.solid)

        if store_wss:
            self.wss_series.append(self.wss_vector()[0].detach().clone())

        self.t += dt
        self.step_count += 1
        div = g.divergence(self.u, self.v, self.w)
        return {"t": self.t, "dt": dt,
                "max_div": float(div.abs().max()),
                "cg_iters": self.ps.last_iters,
                "cg_residual": float(self.ps.last_residual),
                "cg_converged": bool(self.ps.converged),
                "umax": float(max(self.u.abs().max(), self.v.abs().max(),
                                  self.w.abs().max()))}

    # ---- REAL diagnostics ----
    def wall_normal(self) -> Tuple[Tensor, Tensor]:
        """n = -grad(solid)/|grad(solid)|, points solid -> fluid."""
        sp = self.solid
        for ax in range(3):
            sp = pad_axis(sp, ax, self.s.bc[ax])
        gs = []
        for ax in range(3):
            hi = sp[sl(ax, slice(2, None))]
            lo = sp[sl(ax, slice(0, -2))]
            for o in range(3):
                if o != ax:
                    hi = hi[sl(o, slice(1, -1))]
                    lo = lo[sl(o, slice(1, -1))]
            gs.append((hi - lo) / (2 * self.s.h[ax]))
        gvec = torch.stack(gs, 0)
        mag = gvec.norm(dim=0).clamp_min(1e-30)
        return -gvec / mag, mag

    def interface_fluid_mask(self) -> Tensor:
        """Fluid cells adjacent to solid."""
        s = self.solid
        nb = torch.zeros_like(s)
        for ax in range(3):
            nb = torch.maximum(nb, (torch.roll(s, -1, ax) + torch.roll(s, 1, ax)))
        return (s < 0.5) & (nb > 0.5)

    def wss_vector(self) -> Tuple[Tensor, Tensor]:
        """
        tau_vec = mu * u_tangential / d_wall,  d_wall ~ dx/2 (voxelised).
        Returns (tau (3,nx,ny,nz), mask).
        """
        n, _ = self.wall_normal()
        uc, vc, wc = self.g.to_cc(self.u, self.v, self.w)
        ucc = torch.stack([uc, vc, wc], 0)
        un = (ucc * n).sum(0, keepdim=True)
        ut = ucc - un * n
        dwall = 0.5 * min(self.s.h)
        tau = self.mu0 * ut / max(dwall, 1e-30)
        mask = self.interface_fluid_mask()
        tau = tau * mask.unsqueeze(0)
        return tau, mask

    def metrics(self, axis_flow: int = 2) -> Dict[str, float]:
        g, s = self.g, self.s
        uc, vc, wc = g.to_cc(self.u, self.v, self.w)
        cv = s.dx * s.dy * s.dz
        ke = 0.5 * self.rho0 * float((uc**2 + vc**2 + wc**2).sum() * cv)
        div = g.divergence(self.u, self.v, self.w)
        tau, mask = self.wss_vector()
        tmag = tau.norm(dim=0)
        n_if = int(mask.sum())
        # flow rate through the inlet plane
        comp = (self.u, self.v, self.w)[axis_flow]
        dA = s.h[(axis_flow + 1) % 3] * s.h[(axis_flow + 2) % 3]
        Q_in = float((comp * self.inlet).sum() * dA)
        Q_out = float((comp * self.outlet).sum() * dA)
        # pressure drop between inlet-adjacent and outlet-adjacent cells
        p = self.p
        n = p.shape[axis_flow]
        p_in = float(p.narrow(axis_flow, 0, max(1, n // 10)).mean())
        p_out = float(p.narrow(axis_flow, n - max(1, n // 10), max(1, n // 10)).mean())
        out = {
            "kinetic_energy_J": ke,
            "max_divergence_1_s": float(div.abs().max()),
            "flow_rate_in_m3_s": Q_in,
            "flow_rate_out_m3_s": Q_out,
            "mass_imbalance_rel": abs(Q_in - Q_out) / max(abs(Q_in), 1e-30),
            "pressure_drop_Pa": p_in - p_out,
            "n_interface_cells": n_if,
            "wss_mean_Pa": float(tmag[mask].mean()) if n_if else 0.0,
            "wss_max_Pa": float(tmag[mask].max()) if n_if else 0.0,
            "residence_time_mean_s": float(self.age[mask].mean()) if n_if else 0.0,
            "residence_time_max_s": float(self.age.max()),
            "poisson_iters": self.ps.last_iters,
            "poisson_residual": float(self.ps.last_residual),
            "poisson_converged": int(bool(self.ps.converged)),
            "n_steps": self.step_count,
            "sim_time_s": self.t,
        }
        if self.wss_series:
            T = torch.stack(self.wss_series, 0)          # (T,3,nx,ny,nz)
            num = T.sum(0).norm(dim=0)
            den = T.norm(dim=1).sum(0).clamp_min(1e-30)
            osi = 0.5 * (1.0 - num / den)
            out["osi_mean"] = float(osi[mask].mean()) if n_if else 0.0
            out["osi_max"] = float(osi[mask].max()) if n_if else 0.0
            out["n_wss_samples"] = int(T.shape[0])
        return out


# =============================================================================
# §4. THE ADAPTER  (drop-in replacement for the analytic estimate)
# =============================================================================

class RealCFDAdapter:
    """
    Replaces med_world_model_connectors.OrganCFDAdapter._stub / analytic path.
    Reports UNAVAILABLE (never a fake number) when the backend or geometry
    is missing, and BLOCKED when governance is not satisfied.
    """

    def __init__(self, module_dir: str = "."):
        self.backend = "one_cfd_stencils" if HAS_BACKEND else "unavailable"
        self.backend_error = _BACKEND_ERR

    def available(self) -> Dict[str, Any]:
        return {"backend": self.backend, "available": HAS_BACKEND,
                "stencil_version": STENCIL_VERSION,
                "error": self.backend_error}

    # ---- governance ----
    @staticmethod
    def _governance_ok(req: RealCFDRequest) -> Tuple[bool, str]:
        if not req.uses_patient_data:
            return True, "non-patient geometry (phantom / synthetic)"
        d = req.governance_decision or {}
        if not d.get("authorized", False):
            return False, ("BLOCKED: patient-derived geometry requires an "
                           "authorised governance decision (IRB + de-identification)")
        return True, "governance authorised"

    # ---- main entry ----
    def simulate(self, req: RealCFDRequest,
                 progress: Optional[Callable[[int, Dict], None]] = None
                 ) -> CFDResult:
        assumptions = list(req.bc.assumptions) if req.bc else []
        limitations = [
            "voxelised Brinkman geometry: staircased walls, diffuse ~2 cells",
            "WSS wall distance ~ dx/2 (first-order geometric approximation)",
            "first-order in time; convective (non-conservative) momentum form",
            "structured Cartesian MAC grid: NOT body-fitted",
        ]
        validation = ["mesh convergence study", "4D-flow MRI", "Doppler peak velocity",
                      "invasive pressure if available"]

        ok, why = self._governance_ok(req)
        if not ok:
            return CFDResult("BLOCKED", self.backend, {}, [], {},
                             assumptions, limitations + [why], validation)
        if not HAS_BACKEND:
            return CFDResult("UNAVAILABLE", self.backend, {}, [], {},
                             assumptions,
                             limitations + [f"one_cfd_stencils: {_BACKEND_ERR}"],
                             validation)
        if req.mesh is None or req.bc is None:
            return CFDResult("UNAVAILABLE", self.backend, {}, [], {},
                             assumptions,
                             limitations + ["mesh and boundary conditions are required; "
                                            "no geometry -> no solve, no fabricated number"],
                             validation)

        dev = torch.device(req.device)
        dt_ = torch.float64 if req.dtype == "float64" else torch.float32

        # ---- geometry: mesh bounding box -> grid ----
        V = np.asarray(req.mesh.vertices_mm, dtype=np.float64) * 1e-3   # mm -> m
        lo, hi = V.min(0), V.max(0)
        span = np.maximum(hi - lo, 1e-9)
        pad = 0.05 * span
        Lx, Ly, Lz = (span + 2 * pad).tolist()
        origin = lo - pad
        Vm = V - origin

        spec = GridSpec(nx=req.nx, ny=req.ny, nz=req.nz,
                        Lx=Lx, Ly=Ly, Lz=Lz,
                        bc=("wall", "wall", "wall"),
                        device=req.device, dtype=req.dtype)
        g = MACOps(spec)
        gx = (np.arange(req.nx) + 0.5) * spec.dx
        gy = (np.arange(req.ny) + 0.5) * spec.dy
        gz = (np.arange(req.nz) + 0.5) * spec.dz

        mesh_m = SurfaceMesh(Vm, req.mesh.faces, watertight=req.mesh.watertight)
        solid_np, vinfo = voxelize_closed_mesh(mesh_m, gx, gy, gz)
        solid = torch.tensor(solid_np, dtype=dt_, device=dev)

        # ---- inlet / outlet face masks on the flow axis ----
        ax = int(req.axis_flow)
        h = spec.h
        n_f = (req.nx, req.ny, req.nz)[ax] + 1
        face_coord = np.arange(n_f) * h[ax]

        def plane_mask(plane: Tuple[int, float]) -> Tensor:
            pax, pcoord = plane
            coord_m = pcoord * 1e-3 - origin[pax]
            k = int(np.argmin(np.abs(face_coord - coord_m)))
            m = torch.zeros(spec.face_shape(ax), dtype=torch.bool, device=dev)
            idx = [slice(None)] * 3
            idx[ax] = k
            m[tuple(idx)] = True
            return m

        inlet = plane_mask(req.bc.inlet_plane)
        outlet = torch.zeros_like(inlet)
        for pl in req.bc.outlet_planes:
            outlet = outlet | plane_mask(pl)
        # do not drive flow into solid
        sf = face_scalar(solid, spec)[ax] > 0.5
        inlet = inlet & (~sf)
        outlet = outlet & (~sf)

        # ---- inlet velocity from flow rate if supplied ----
        dA = h[(ax + 1) % 3] * h[(ax + 2) % 3]
        A_in = float(inlet.sum()) * dA
        if req.flow_rate_ml_s is not None and A_in > 0:
            u_in = req.flow_rate_ml_s * 1e-6 / A_in
        else:
            u_in = float(req.inlet_velocity_m_s)

        solver = RealMACSolver(spec, solid, inlet, outlet,
                               req.rho, req.mu, req.brinkman_eta,
                               req.poisson_tol, req.poisson_max_iter)

        # ---- time stepping to (quasi-)steady state ----
        t0 = time.perf_counter()
        prev_umax = 0.0
        hist: List[Dict[str, float]] = []
        converged = False
        for k in range(req.n_steps):
            if req.bc.waveform is not None and req.bc.waveform_dt_s > 0:
                ti = int(solver.t / req.bc.waveform_dt_s) % len(req.bc.waveform)
                uu = float(req.bc.waveform[ti])
            else:
                uu = u_in
            rec = solver.step(u_in=uu, axis_flow=ax,
                              store_wss=req.compute_osi and k % 5 == 0)
            hist.append(rec)
            if progress and k % 50 == 0:
                progress(k, rec)
            if k > 20 and abs(rec["umax"] - prev_umax) < req.steady_tol * max(rec["umax"], 1e-12):
                converged = True
                break
            prev_umax = rec["umax"]
        wall_s = time.perf_counter() - t0

        m = solver.metrics(axis_flow=ax)
        m["inlet_velocity_m_s"] = u_in
        m["inlet_area_m2"] = A_in
        m["wall_time_s"] = wall_s
        m["grid_cells"] = req.nx * req.ny * req.nz
        m["solid_fraction"] = vinfo["solid_fraction"]

        return CFDResult(
            status="OK",
            backend=f"one_cfd_stencils {STENCIL_VERSION} :: MACOps + project()",
            metrics=m,
            fields_available=["u", "v", "w", "p", "age", "wss_vector", "solid"],
            convergence={"steady_converged": converged,
                         "n_steps": solver.step_count,
                         "last_cg_residual": hist[-1]["cg_residual"],
                         "last_cg_converged": hist[-1]["cg_converged"],
                         "voxelisation": vinfo},
            assumptions=assumptions + [
                f"rho={req.rho} kg/m^3, mu={req.mu} Pa s (Newtonian)",
                f"Brinkman eta={req.brinkman_eta} (implicit, O(sqrt(eta)) slip)",
                "inlet plug profile unless a waveform is supplied",
            ],
            limitations=limitations,
            validation_required=validation,
        )


# =============================================================================
# §5. MESH CONVERGENCE
# =============================================================================

def mesh_convergence(req: RealCFDRequest, refine: int = 2,
                     metric: str = "pressure_drop_Pa") -> Dict[str, Any]:
    """Richardson-style convergence on the REAL solve. Reports the observed order."""
    ad = RealCFDAdapter()
    rows = []
    for f in (1, refine):
        r = RealCFDRequest(**{**asdict(req),
                              "nx": req.nx * f, "ny": req.ny * f,
                              "nz": req.nz * f,
                              "n_steps": req.n_steps})
        res = ad.simulate(r)
        rows.append({"refine": f, "nx": r.nx, "ny": r.ny, "nz": r.nz,
                     "status": res.status,
                     "value": res.metrics.get(metric, float("nan")),
                     "wss_mean": res.metrics.get("wss_mean_Pa", float("nan"))})
    out: Dict[str, Any] = {"metric": metric, "rows": rows}
    if len(rows) == 2 and all(math.isfinite(x["value"]) for x in rows):
        d = abs(rows[1]["value"] - rows[0]["value"])
        rel = d / max(abs(rows[1]["value"]), 1e-30)
        out["relative_change"] = rel
        out["verdict"] = ("CONVERGING (<10%)" if rel < 0.10 else
                          "NOT CONVERGED — refine further or use body-fitted CFD")
    else:
        out["verdict"] = "NOT EVALUATED"
    return out


# =============================================================================
# §6. SELF-TEST
# =============================================================================

def self_test(quick: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    print("\n[T1] backend availability")
    rep("one_cfd_stencils importable", HAS_BACKEND, _BACKEND_ERR or STENCIL_VERSION)
    if not HAS_BACKEND:
        print("      Run repair_five.py --apply on one_cfd_stencils.py first.")
        print("      NOT fabricating results. Exiting.")
        return False

    torch.manual_seed(0)
    n = 16 if quick else 24
    L = 1.0
    spec = GridSpec(nx=n, ny=n, nz=n, Lx=L, Ly=L, Lz=L,
                    bc=("periodic",) * 3, dtype="float64")
    g = MACOps(spec)

    # ---- T2: Taylor-Green is DISCRETELY divergence-free on MAC ----
    print("\n[T2] MAC staggered layout + discrete divergence")
    xf = torch.arange(n + 1, dtype=torch.float64) * spec.dx
    xc = (torch.arange(n, dtype=torch.float64) + 0.5) * spec.dx
    Xf, Yc, Zc = torch.meshgrid(xf, xc, xc, indexing="ij")
    Xc, Yf, Zc2 = torch.meshgrid(xc, xf, xc, indexing="ij")
    k = 2 * math.pi / L
    u0 = torch.sin(k * Xf) * torch.cos(k * Yc) * torch.cos(k * Zc)
    v0 = -torch.cos(k * Xc) * torch.sin(k * Yf) * torch.cos(k * Zc2)
    w0 = g.zeros_w()
    d0 = float(g.divergence(u0, v0, w0).abs().max())
    rep("TGV exactly div-free on MAC", d0 < 1e-12, f"max|div|={d0:.2e}")
    gx_, gy_, gz_ = g.grad_cc_to_faces(g.cc())
    rep("face shapes asserted",
        gx_.shape == (n + 1, n, n) and gy_.shape == (n, n + 1, n)
        and gz_.shape == (n, n, n + 1))

    # ---- T3: REAL project() preserves div-free + correct beta ----
    print("\n[T3] REAL project(): divergence and face beta")
    rho = g.cc(1.0)
    up, vp, wp, pp, ps = project(u0, v0, w0, rho, spec, 1e-3)
    d1 = float(g.divergence(up, vp, wp).abs().max())
    rep("project keeps div-free", d1 < 1e-10, f"{d0:.2e} -> {d1:.2e}")
    rho_v = torch.rand(spec.cc_shape, dtype=torch.float64) * 999 + 1
    bx, _, _ = face_beta(rho_v, spec)
    ref = 0.5 * (1 / rho_v[:-1] + 1 / rho_v[1:])
    wrong = 2 / (1 / rho_v[:-1] + 1 / rho_v[1:])
    rep("beta = mean(1/rho), NOT harmonic mean of rho",
        float((bx[1:-1] - ref).abs().max()) < 1e-14
        and float((bx[1:-1] - wrong).abs().max()) > 1e-6)

    # ---- T4: REAL viscous decay vs DISCRETE analytic eigenvalue ----
    print("\n[T4] REAL solve: TGV viscous decay vs discrete eigenvalue")
    nu = 0.05
    solid0 = g.cc(0.0)
    nof = torch.zeros(spec.face_shape(2), dtype=torch.bool)
    sv = RealMACSolver(spec, solid0, nof, nof, rho=1.0, mu=nu,
                       brinkman_eta=1e30, poisson_tol=1e-10)
    sv.u, sv.v, sv.w = u0.clone(), v0.clone(), w0.clone()
    lam_h = (4 / spec.dx**2 * math.sin(k * spec.dx / 2)**2
             + 4 / spec.dy**2 * math.sin(k * spec.dy / 2)**2)
    ke0 = 0.5 * float((g.to_cc(sv.u, sv.v, sv.w)[0]**2
                       + g.to_cc(sv.u, sv.v, sv.w)[1]**2).sum())
    nst = 40 if quick else 80
    dt = 0.2 * spec.dx**2 / nu
    for _ in range(nst):
        sv.step(dt=dt, u_in=0.0, axis_flow=2)
    ke1 = 0.5 * float((g.to_cc(sv.u, sv.v, sv.w)[0]**2
                       + g.to_cc(sv.u, sv.v, sv.w)[1]**2).sum())
    T = nst * dt
    num = ke1 / max(ke0, 1e-30)
    ana = math.exp(-2 * nu * lam_h * T)
    rel = abs(num - ana) / max(ana, 1e-30)
    rep("KE decay matches discrete analytic rate", rel < 0.15,
        f"num={num:.5f} ana={ana:.5f} rel={rel:.3f}")

    # ---- T5: voxelisation of a closed cylinder ----
    print("\n[T5] voxelisation (closed cylinder phantom)")
    nc = 16 if quick else 24
    Lx = Ly = 0.4
    Lz = 1.2
    R = 0.12
    mesh, exact_solid = cylinder_phantom(nc, nc, nc * 3, Lx, Ly, Lz, R)
    gxc = (np.arange(nc) + 0.5) * (Lx / nc)
    gyc = (np.arange(nc) + 0.5) * (Ly / nc)
    gzc = (np.arange(nc * 3) + 0.5) * (Lz / (nc * 3))
    vox, vinfo = voxelize_closed_mesh(mesh, gxc, gyc, gzc)
    err = float(np.abs(vox - exact_solid).mean())
    rep("voxelisation matches analytic cylinder", err < 0.02,
        f"mismatch={err:.4f}, odd_columns={vinfo['odd_crossing_columns']}")
    rep("no odd-crossing columns (watertight)", vinfo["odd_crossing_columns"] == 0)

    # ---- T6: REAL pipe solve vs Hagen-Poiseuille (honest error) ----
    print("\n[T6] REAL pipe solve vs analytic — error REPORTED, not hidden")
    req = RealCFDRequest(
        organ="vascular", research_question="phantom pipe verification",
        mesh=mesh,
        bc=BoundaryConditionSet(inlet_plane=(2, 0.0),
                                outlet_planes=[(2, Lz * 1e3)]),
        rho=1.0, mu=0.1, inlet_velocity_m_s=0.1,
        nx=nc, ny=nc, nz=nc * 3, axis_flow=2,
        brinkman_eta=1e-4, n_steps=200 if quick else 500,
        poisson_tol=1e-6, compute_osi=True,
    )
    # mesh coords are mm; phantom built in metres -> convert back
    req.mesh = SurfaceMesh(mesh.vertices_mm * 1e3, mesh.faces, watertight=True)
    req.bc = BoundaryConditionSet(inlet_plane=(2, 0.0),
                                  outlet_planes=[(2, Lz * 1e3)])
    ad = RealCFDAdapter()
    res = ad.simulate(req)
    rep("real solve returned OK", res.status == "OK", res.status)
    if res.status == "OK":
        m = res.metrics
        A_vox = (1.0 - vinfo["solid_fraction"]) * Lx * Ly
        R_eff = math.sqrt(A_vox / math.pi)
        Q = m["flow_rate_in_m3_s"]
        dpdx_ana = 8 * req.mu * Q / (math.pi * R_eff**4)
        dpdx_num = m["pressure_drop_Pa"] / Lz
        wss_ana = 4 * req.mu * Q / (math.pi * R_eff**3)
        e_dp = abs(dpdx_num - dpdx_ana) / max(abs(dpdx_ana), 1e-30)
        e_wss = abs(m["wss_mean_Pa"] - wss_ana) / max(wss_ana, 1e-30)
        print(f"      Q={Q:.4e} m^3/s  R_eff={R_eff:.4f} m")
        print(f"      dp/dx num={dpdx_num:.4e}  ana={dpdx_ana:.4e}  rel={e_dp:.3f}")
        print(f"      WSS   num={m['wss_mean_Pa']:.4e}  ana={wss_ana:.4e}  rel={e_wss:.3f}")
        print(f"      max|div|={m['max_divergence_1_s']:.2e}  "
              f"mass_imbalance={m['mass_imbalance_rel']:.2e}")
        rep("pressure drop is the right ORDER (voxelised Brinkman)", e_dp < 0.60,
            f"rel={e_dp:.3f}  (NOT clinical accuracy)")
        rep("projection converged", m["poisson_converged"] == 1,
            f"res={m['poisson_residual']:.2e}")
        rep("WSS finite and non-negative",
            math.isfinite(m["wss_mean_Pa"]) and m["wss_mean_Pa"] >= 0)
        rep("OSI computed from time series", "osi_mean" in m,
            f"OSI={m.get('osi_mean', float('nan')):.3f}")
        print("      HONEST: voxelised Brinkman WSS/dp carry O(10-40%) error.")
        print("              Mesh convergence + body-fitted CFD are required")
        print("              before any quantitative or clinical use.")

    # ---- T7: governance blocks patient geometry ----
    print("\n[T7] governance gate")
    req_p = RealCFDRequest(**{**asdict(req), "uses_patient_data": True,
                              "governance_decision": None})
    rp = ad.simulate(req_p)
    rep("patient geometry BLOCKED without authorisation", rp.status == "BLOCKED",
        rp.status)
    req_ok = RealCFDRequest(**{**asdict(req), "uses_patient_data": True,
                               "governance_decision": {"authorized": True},
                               "n_steps": 20})
    rok = ad.simulate(req_ok)
    rep("authorised patient geometry runs", rok.status == "OK", rok.status)

    # ---- T8: missing geometry -> UNAVAILABLE, never a fake number ----
    print("\n[T8] no geometry -> UNAVAILABLE")
    r_none = ad.simulate(RealCFDRequest(organ="brain",
                                        research_question="no mesh"))
    rep("no mesh -> UNAVAILABLE (no fabricated metric)",
        r_none.status == "UNAVAILABLE" and not r_none.metrics, r_none.status)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
REAL NOW
  MACOps staggered operators, PressurePoisson, project() with beta=1/rho_face.
  Real velocity/pressure fields, real WSS vector, OSI, residence time.
  NO analytic formula produces the reported metrics.

STILL LIMITED
  Voxelised Brinkman geometry (structured Cartesian cannot be body-fitted).
  WSS wall distance ~ dx/2; staircased walls; O(10-40%) error on coarse grids.
  First-order time; convective (non-conservative) momentum form.
  NOT a medical device. Mesh convergence + 4D-flow/Doppler/pressure validation
  are mandatory before any quantitative use.
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Real CFD adapter (MACOps + project)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.status:
        print(json.dumps(RealCFDAdapter().available(), indent=2))
        return 0
    if a.selftest:
        return 0 if self_test(quick=not a.full) else 1
    if a.convergence:
        nc = 12
        mesh, _ = cylinder_phantom(nc, nc, nc * 3, 0.4, 0.4, 1.2, 0.12)
        req = RealCFDRequest(
            organ="vascular", research_question="mesh convergence",
            mesh=SurfaceMesh(mesh.vertices_mm * 1e3, mesh.faces, watertight=True),
            bc=BoundaryConditionSet(inlet_plane=(2, 0.0),
                                    outlet_planes=[(2, 1200.0)]),
            rho=1.0, mu=0.1, inlet_velocity_m_s=0.1,
            nx=nc, ny=nc, nz=nc * 3, n_steps=120, poisson_tol=1e-6)
        out = mesh_convergence(req, refine=2)
        print(json.dumps(out, indent=2, default=float))
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        return 0

    ap.print_help()
    print(BANNER)
    print("Start: --status | --selftest | --selftest --full | --convergence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
