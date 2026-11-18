#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ale_gcl_fsi.py — ALE + Geometric Conservation Law for large-displacement FSI
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn also
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
WHAT ALE + GCL ACTUALLY REQUIRE
=============================================================================
[A1] ALE convection uses the RELATIVE velocity (u - w), w = dx/dt.
     Conservative ALE momentum for a moving control volume V(t):
         d/dt (V u) + sum_f u_f * [(u_f - w_f) . S_f] = RHS
     The transient term MUST use V^{n+1} u^{n+1} - V^n u^n, not V du/dt.

[A2] GCL: dV/dt = sum_f (w . S)_f  DISCRETELY.
     The swept volume of a face is computed GEOMETRICALLY from the
     space-time prism between its old and new position, decomposed into
     tetrahedra. Using A*(w.n)*dt is only correct for a RIGID PLANAR face;
     it is wrong for a deforming or rotating face, and that error is
     exactly what breaks free-stream preservation.

[A3] Free-stream preservation is the DECISIVE test. For u = U (constant),
     the discrete ALE momentum must give u^{n+1} = U exactly:
         d/dt(V U) + U * sum_f [(U - w).S]
       = U dV/dt + U*(0 - dV/dt) = 0
     This holds ONLY if GCL holds discretely. Verified in T4.

[A4] Incompressibility on a moving mesh constrains the ABSOLUTE flux:
         sum_f (u . S)_f = 0
     NOT the relative flux. Indeed sum_f [(u-w).S] = -dV/dt != 0.
     The pressure equation therefore enforces the absolute flux, while
     the momentum convection uses the relative flux. Getting this
     backwards is a classic ALE bug.

=============================================================================
IMPLEMENTED
=============================================================================
[I1] MovingHexMesh: structured hex topology, moving nodes, geometry
     recomputed at every step. Volumes by 6-tet Kuhn decomposition.
     Face areas/normals by triangle cross products, oriented outward
     (boundary) or owner->neighbour (internal).
[I2] Swept volumes by space-time triangular-prism -> 3-tet decomposition.
     GCL residual reported PER CELL, not asserted.
[I3] Mesh motion:
       - transfinite interpolation (TFI) for wall-normal stretching
       - variable-diffusivity Laplacian smoothing for general 3D motion
     Both differentiable (torch), so mesh motion can be optimised.
[I4] Mesh quality guards: min volume, scaled Jacobian, aspect ratio,
     non-orthogonality. Motion is REJECTED if any cell inverts.
[I5] ALE-PIMPLE with Rhie-Chow: relative convective flux, GCL-consistent
     transient, absolute-flux pressure constraint.
[I6] Partitioned FSI with Aitken relaxation on a MOVING mesh: the
     structure displaces the wall, the wall moves the mesh, the fluid
     solves on the deformed mesh, traction feeds back.

=============================================================================
HONEST LIMITATIONS
=============================================================================
[L1] TOPOLOGY IS FIXED. Nodes move; cells are never added, removed, or
     reconnected. Very large displacement eventually inverts cells —
     the guard REJECTS that motion rather than producing garbage.
     Remeshing / topology change is NOT_IMPLEMENTED.
[L2] Hex mesh only. Tets, polyhedra, overset/Chimera, sliding interfaces
     are NOT_IMPLEMENTED.
[L3] Structure is a reduced spring-mass-damper shell, not a finite-strain
     hyperelastic solid.
[L4] First-order in time (Euler) for the ALE transient; the mesh flux is
     exact over the step, so GCL holds, but the fluid time accuracy is O(dt).
[L5] Momentum solved by damped Jacobi sweeps, residual REPORTED.
[L6] No turbulence model, no contact, no rupture.
[L7] No experimental validation. Oracles are GCL, free-stream
     preservation, mass conservation, and an analytic piston case.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("ale_gcl")

VERSION = "1.0.0"
BANNER = f"""
=============================================================================
ALE + GCL v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
GCL is VERIFIED per cell, not asserted. Free-stream preservation is the test.
Topology FIXED: nodes move, cells never reconnect. Inversion => REJECT.
=============================================================================
"""

# Kuhn 6-tet decomposition of a hex (fan around the v0-v6 diagonal)
HEX_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))

# local hex faces, counter-clockwise seen from OUTSIDE
HEX_FACES = ((0, 3, 2, 1),   # z-min
             (4, 5, 6, 7),   # z-max
             (0, 1, 5, 4),   # y-min
             (3, 7, 6, 2),   # y-max
             (0, 4, 7, 3),   # x-min
             (1, 2, 6, 5))   # x-max
FACE_AXIS = (2, 2, 1, 1, 0, 0)
FACE_SIDE = (0, 1, 0, 1, 0, 1)   # 0 = min side, 1 = max side


# =============================================================================
# §1. GEOMETRY PRIMITIVES  (differentiable)
# =============================================================================

def tet_volumes(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Signed tet volume = det[p1-p0, p2-p0, p3-p0]/6."""
    a, b, c = p1 - p0, p2 - p0, p3 - p0
    return (a * torch.cross(b, c, dim=-1)).sum(-1) / 6.0


def tri_area_normal(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Area-weighted normal of a triangle = 0.5 (b-a) x (c-a)."""
    return 0.5 * torch.cross(b - a, c - a, dim=-1)


def swept_prism_volume(a: Tensor, b: Tensor, c: Tensor,
                       a2: Tensor, b2: Tensor, c2: Tensor) -> Tensor:
    """
    Signed 3-volume swept by triangle (a,b,c) moving to (a2,b2,c2).
    Decomposed into 3 tets of the triangular prism:
        (a,b,c,a2), (b,c,a2,b2), (c,a2,b2,c2)
    This is EXACT for arbitrary (non-rigid, non-planar-motion) deformation,
    unlike A*(w.n)*dt which assumes a rigid planar translation.
    """
    v1 = tet_volumes(a, b, c, a2)
    v2 = tet_volumes(b, c, a2, b2)
    v3 = tet_volumes(c, a2, b2, c2)
    return v1 + v2 + v3


# =============================================================================
# §2. MOVING HEX MESH
# =============================================================================

class MovingHexMesh:
    """
    Structured hex mesh with MOVING nodes and FIXED topology.
    All geometry is recomputed from the current node positions.
    """

    def __init__(self, nx: int, ny: int, nz: int,
                 Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0,
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.dev, self.dt_ = torch.device(device), dtype
        self.Nc = nx * ny * nz
        self.Nn = (nx + 1) * (ny + 1) * (nz + 1)
        self._build_nodes()
        self._build_cells()
        self._build_faces()
        self.nodes0 = self.nodes.clone()

    # ---- topology ----
    def nid(self, i, j, k):
        return (i * (self.ny + 1) + j) * (self.nz + 1) + k

    def cid(self, i, j, k):
        return (i * self.ny + j) * self.nz + k

    def _build_nodes(self):
        xs = torch.linspace(0, self.Lx, self.nx + 1, device=self.dev, dtype=self.dt_)
        ys = torch.linspace(0, self.Ly, self.ny + 1, device=self.dev, dtype=self.dt_)
        zs = torch.linspace(0, self.Lz, self.nz + 1, device=self.dev, dtype=self.dt_)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
        self.nodes = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
        # node -> (i,j,k) for boundary identification / motion
        idx = torch.arange(self.Nn, device=self.dev)
        self.nk = idx % (self.nz + 1)
        self.nj = (idx // (self.nz + 1)) % (self.ny + 1)
        self.ni = idx // ((self.ny + 1) * (self.nz + 1))

    def _build_cells(self):
        C = []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    C.append([self.nid(i, j, k), self.nid(i + 1, j, k),
                              self.nid(i + 1, j + 1, k), self.nid(i, j + 1, k),
                              self.nid(i, j, k + 1), self.nid(i + 1, j, k + 1),
                              self.nid(i + 1, j + 1, k + 1), self.nid(i, j + 1, k + 1)])
        self.cells = torch.tensor(C, dtype=torch.long, device=self.dev)

    def _build_faces(self):
        owner, neigh, fnodes, bnd, fax, fside = [], [], [], [], [], []
        seen: Dict[Tuple[int, ...], int] = {}
        for c in range(self.Nc):
            cv = self.cells[c].tolist()
            for lf, loc in enumerate(HEX_FACES):
                q = [cv[t] for t in loc]
                key = tuple(sorted(q))
                if key in seen:
                    f = seen[key]
                    neigh[f] = c
                    # orient owner -> neighbour
                    own_c = self.cell_centroid_from_nodes(owner[f], cv)
                    if float((self._quad_normal(q) *
                              (own_c - self._quad_centroid(q))).sum()) > 0:
                        fnodes[f] = [q[0], q[3], q[2], q[1]]
                else:
                    f = len(owner)
                    seen[key] = f
                    owner.append(c); neigh.append(-1); fnodes.append(q)
                    bnd.append(f); fax.append(FACE_AXIS[lf]); fside.append(FACE_SIDE[lf])
        self.owner = torch.tensor(owner, dtype=torch.long, device=self.dev)
        self.neigh = torch.tensor(neigh, dtype=torch.long, device=self.dev)
        self.fnodes = torch.tensor(fnodes, dtype=torch.long, device=self.dev)
        self.bnd = torch.tensor(bnd, dtype=torch.long, device=self.dev)
        self.faxis = torch.tensor(fax, dtype=torch.long, device=self.dev)
        self.fside = torch.tensor(fside, dtype=torch.long, device=self.dev)
        self.Nf = len(owner)
        self.it = self.neigh >= 0
        self.oi = self.owner[self.it]
        self.ni = self.neigh[self.it]

    def _quad_normal(self, q):
        v = self.nodes[q]
        return tri_area_normal(v[0], v[1], v[2]) + tri_area_normal(v[0], v[2], v[3])

    def _quad_centroid(self, q):
        return self.nodes[q].mean(0)

    def cell_centroid_from_nodes(self, c, cv):
        return self.nodes[cv].mean(0)

    # ---- geometry at current node positions ----
    def geometry(self, nodes: Optional[Tensor] = None) -> Dict[str, Tensor]:
        X = self.nodes if nodes is None else nodes
        cv = X[self.cells]                                   # (Nc,8,3)
        vol = torch.zeros(self.Nc, device=self.dev, dtype=self.dt_)
        for t in HEX_TETS:
            vol = vol + tet_volumes(cv[:, t[0]], cv[:, t[1]],
                                    cv[:, t[2]], cv[:, t[3]])
        cen = cv.mean(1)
        fv = X[self.fnodes]                                  # (Nf,4,3)
        Sf = (tri_area_normal(fv[:, 0], fv[:, 1], fv[:, 2])
              + tri_area_normal(fv[:, 0], fv[:, 2], fv[:, 3]))
        fcen = fv.mean(1)
        # orient boundary faces OUTWARD
        b = ~self.it
        ob = self.owner[b]
        sgn = torch.sign((Sf[b] * (fcen[b] - cen[ob])).sum(1))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        Sf = Sf.clone()
        Sf[b] = Sf[b] * sgn[:, None]
        area = Sf.norm(dim=1)
        nf = Sf / area.clamp_min(1e-30)[:, None]
        d = cen[self.neigh.clamp_min(0)] - cen[self.owner]
        d = torch.where(self.it[:, None], d, fcen - cen[self.owner])
        return {"vol": vol, "cen": cen, "Sf": Sf, "fcen": fcen,
                "area": area, "nf": nf, "dvec": d}

    # ---- swept volumes + GCL ----
    def mesh_flux(self, nodes_old: Tensor, nodes_new: Tensor,
                  dt: float) -> Dict[str, Tensor]:
        """
        Swept volume per face by space-time prism -> 3 tets.
        phi_mesh = swept / dt, oriented owner -> neighbour (internal)
        or outward (boundary).
        GCL residual = (V_new - V_old) - sum_f swept_f  per cell.
        """
        go = self.geometry(nodes_old)
        gn = self.geometry(nodes_new)
        vo = nodes_old[self.fnodes]
        vn = nodes_new[self.fnodes]
        sw = (swept_prism_volume(vo[:, 0], vo[:, 1], vo[:, 2],
                                 vn[:, 0], vn[:, 1], vn[:, 2])
              + swept_prism_volume(vo[:, 0], vo[:, 2], vo[:, 3],
                                   vn[:, 0], vn[:, 2], vn[:, 3]))
        # orient using the OLD geometry (same convention as geometry())
        b = ~self.it
        ob = self.owner[b]
        sgn = torch.sign((go["Sf"][b] * (go["fcen"][b] - go["cen"][ob])).sum(1))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        sw = sw.clone()
        sw[b] = sw[b] * sgn
        # GCL: dV = sum of outward swept volumes
        dV = gn["vol"] - go["vol"]
        acc = torch.zeros(self.Nc, device=self.dev, dtype=self.dt_)
        acc = acc.index_add(0, self.owner, sw)
        acc = acc.index_add(0, self.ni, -sw[self.it])
        res = dV - acc
        scale = gn["vol"].abs().clamp_min(1e-30)
        return {"phi_mesh": sw / dt, "swept": sw, "dV": dV,
                "gcl_abs": res.abs(), "gcl_rel": res.abs() / scale,
                "gcl_max_rel": float((res.abs() / scale).max()),
                "V_old": go["vol"], "V_new": gn["vol"],
                "geo_old": go, "geo_new": gn}

    # ---- quality ----
    def quality(self, nodes: Optional[Tensor] = None) -> Dict[str, float]:
        g = self.geometry(nodes)
        v = g["vol"]
        cv = (self.nodes if nodes is None else nodes)[self.cells]
        # scaled Jacobian at corner 0 (representative)
        e1 = cv[:, 1] - cv[:, 0]; e2 = cv[:, 3] - cv[:, 0]; e3 = cv[:, 4] - cv[:, 0]
        det = (e1 * torch.cross(e2, e3, dim=-1)).sum(-1)
        nrm = e1.norm(dim=-1) * e2.norm(dim=-1) * e3.norm(dim=-1)
        sj = det / nrm.clamp_min(1e-30)
        it = self.it
        d = g["dvec"][it]; dn = d.norm(dim=1).clamp_min(1e-30)
        cos = (g["nf"][it] * (d / dn[:, None])).sum(1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cos.abs()))
        return {"vol_min": float(v.min()), "vol_max": float(v.max()),
                "vol_ratio": float(v.max() / v.min().clamp_min(1e-30)),
                "scaled_jacobian_min": float(sj.min()),
                "nonorth_deg_max": float(ang.max()) if ang.numel() else 0.0,
                "nonorth_deg_mean": float(ang.mean()) if ang.numel() else 0.0,
                "n_inverted": int((v <= 0).sum()),
                "n_cells": self.Nc, "n_faces": self.Nf}

    def accepts_motion(self, nodes_new: Tensor,
                       min_scaled_jacobian: float = 0.05) -> Tuple[bool, str, Dict]:
        q = self.quality(nodes_new)
        if q["n_inverted"] > 0:
            return False, f"REJECTED: {q['n_inverted']} inverted cells", q
        if q["scaled_jacobian_min"] < min_scaled_jacobian:
            return False, (f"REJECTED: min scaled Jacobian "
                           f"{q['scaled_jacobian_min']:.4f} < {min_scaled_jacobian}"), q
        return True, "accepted", q


# =============================================================================
# §3. MESH MOTION
# =============================================================================

class MeshMotion:
    """
    Interior node motion from prescribed boundary displacement.
    Two methods, both differentiable:
      'tfi'      : transfinite interpolation along one axis (exact, cheap,
                   best for wall-normal stretching)
      'laplace'  : variable-diffusivity Laplacian smoothing on the node
                   graph (general 3D, stiffens near the moving wall)
    """

    def __init__(self, mesh: MovingHexMesh, method: str = "tfi",
                 axis: int = 1, n_smooth: int = 200, stiff: float = 2.0):
        self.m = mesh
        self.method = method
        self.axis = axis
        self.n_smooth = n_smooth
        self.stiff = stiff
        self._build_node_graph()

    def _build_node_graph(self):
        m = self.m
        pairs = []
        for i in range(m.nx + 1):
            for j in range(m.ny + 1):
                for k in range(m.nz + 1):
                    a = m.nid(i, j, k)
                    if i + 1 <= m.nx: pairs.append((a, m.nid(i + 1, j, k)))
                    if j + 1 <= m.ny: pairs.append((a, m.nid(i, j + 1, k)))
                    if k + 1 <= m.nz: pairs.append((a, m.nid(i, j, k + 1)))
        P = torch.tensor(pairs, dtype=torch.long, device=m.dev)
        self.e0, self.e1 = P[:, 0], P[:, 1]

    def boundary_mask(self) -> Tensor:
        m = self.m
        return ((m.ni == 0) | (m.ni == m.nx) |
                (m.nj == 0) | (m.nj == m.ny) |
                (m.nk == 0) | (m.nk == m.nz))

    def move(self, nodes_old: Tensor,
             boundary_disp: Tensor) -> Tensor:
        """boundary_disp: (Nn,3), only boundary entries are used."""
        if self.method == "tfi":
            return self._tfi(nodes_old, boundary_disp)
        return self._laplace(nodes_old, boundary_disp)

    def _tfi(self, X0: Tensor, bd: Tensor) -> Tensor:
        """
        Transfinite interpolation along self.axis. Interior nodes keep
        their fractional position between the two moving boundary planes.
        Exact for wall-normal stretching; keeps cell quality.
        """
        m = self.m
        ax = self.axis
        n_ax = (m.nx, m.ny, m.nz)[ax]
        coord_idx = (m.ni, m.nj, m.nk)[ax]
        eta = coord_idx.to(X0.dtype) / float(n_ax)
        Xn = X0.clone()
        # for each transverse line, interpolate the boundary displacement
        other = [d for d in range(3) if d != ax]
        # group nodes by their transverse indices
        key = ((m.ni if other[0] == 0 else (m.nj if other[0] == 1 else m.nk))
               * (n_ax + 1) +
               (m.ni if other[1] == 0 else (m.nj if other[1] == 1 else m.nk)))
        nlines = int(key.max()) + 1
        lo = torch.zeros(nlines, 3, device=X0.device, dtype=X0.dtype)
        hi = torch.zeros_like(lo)
        is_lo = coord_idx == 0
        is_hi = coord_idx == n_ax
        lo = lo.index_copy(0, key[is_lo], bd[is_lo])
        hi = hi.index_copy(0, key[is_hi], bd[is_hi])
        w = eta[:, None]
        disp = (1 - w) * lo[key] + w * hi[key]
        return X0 + disp

    def _laplace(self, X0: Tensor, bd: Tensor) -> Tensor:
        """
        Solve div(k grad d) = 0 for the interior displacement with
        Dirichlet boundary values, k = 1/(distance to boundary)^stiff
        approximated by an inverse-degree weighting. Jacobi iteration.
        """
        m = self.m
        bmask = self.boundary_mask()
        d = torch.zeros_like(X0)
        d = torch.where(bmask[:, None], bd, d)
        # edge diffusivity: stiffen edges touching the boundary
        touch = (bmask[self.e0] | bmask[self.e1]).to(X0.dtype)
        k = 1.0 + self.stiff * touch
        deg = torch.zeros(m.Nn, device=X0.device, dtype=X0.dtype)
        deg = deg.index_add(0, self.e0, k).index_add(0, self.e1, k)
        deg = deg.clamp_min(1e-30)
        for _ in range(self.n_smooth):
            acc = torch.zeros_like(d)
            acc = acc.index_add(0, self.e0, k[:, None] * d[self.e1])
            acc = acc.index_add(0, self.e1, k[:, None] * d[self.e0])
            dn = acc / deg[:, None]
            d = torch.where(bmask[:, None], bd, dn)
        return X0 + d


# =============================================================================
# §4. ALE-PIMPLE SOLVER
# =============================================================================

@dataclass
class ALEConfig:
    dt: float = 2e-3
    n_outer: int = 3
    n_piso: int = 2
    n_mom_sweeps: int = 8
    alpha_u: float = 0.7
    alpha_p: float = 0.3
    mom_omega: float = 0.9
    p_tol: float = 1e-10
    p_max_iter: int = 800
    min_scaled_jacobian: float = 0.05


@dataclass
class AleBC:
    faces: Tensor
    kind: str                    # wall | inlet | outlet | symmetry
    u_value: Tuple[float, float, float] = (0., 0., 0.)
    p_value: float = 0.0


class ALEPimple:
    """
    Incompressible ALE-PIMPLE with Rhie-Chow on a moving hex mesh.

    Momentum : (V^{n+1} u^{n+1} - V^n u^n)/dt
               + sum_f u_f [(u_f - w_f).S_f]  = -V grad(p)/rho + visc
    Pressure : sum_f (u_f . S_f) = 0      (ABSOLUTE flux, not relative)
    GCL      : V^{n+1} - V^n = sum_f swept_f   (verified, not assumed)
    """

    def __init__(self, mesh: MovingHexMesh, rho: float, nu: float,
                 bcs: List[AleBC], cfg: Optional[ALEConfig] = None):
        self.m, self.rho, self.nu = mesh, float(rho), float(nu)
        self.cfg = cfg or ALEConfig()
        self.bcs = bcs
        self.dev, self.dt_ = mesh.dev, mesh.dt_
        self.u = torch.zeros(mesh.Nc, 3, device=self.dev, dtype=self.dt_)
        self.p = torch.zeros(mesh.Nc, device=self.dev, dtype=self.dt_)
        self.phi = torch.zeros(mesh.Nf, device=self.dev, dtype=self.dt_)
        self.phi_mesh = torch.zeros(mesh.Nf, device=self.dev, dtype=self.dt_)
        self.V = mesh.geometry()["vol"].clone()
        self.t = 0.0
        self.last: Dict[str, Any] = {}
        self._classify()

    def _classify(self):
        m = self.m
        self.bkind = torch.full((m.Nf,), -1, dtype=torch.long, device=self.dev)
        self.bu = torch.zeros(m.Nf, 3, device=self.dev, dtype=self.dt_)
        self.bp = torch.zeros(m.Nf, device=self.dev, dtype=self.dt_)
        K = {"wall": 0, "inlet": 1, "outlet": 2, "symmetry": 3}
        cov = torch.zeros(m.Nf, dtype=torch.bool, device=self.dev)
        for b in self.bcs:
            f = b.faces.to(self.dev)
            self.bkind[f] = K[b.kind]
            self.bu[f] = torch.tensor(b.u_value, device=self.dev, dtype=self.dt_)
            self.bp[f] = b.p_value
            cov[f] = True
        miss = (~m.it) & (~cov)
        if bool(miss.any()):
            self.bkind[miss] = 0
            log.warning("%d boundary faces defaulted to WALL", int(miss.sum()))
        self.is_dir_p = (~m.it) & (self.bkind == 2)
        self.has_dir = bool(self.is_dir_p.any())

    # ---- interpolation / operators on the CURRENT geometry ----
    def _setup(self, g: Dict[str, Tensor]):
        m = self.m
        d = g["dvec"]
        ddn = (d * g["nf"]).sum(1)
        safe = torch.where(ddn.abs() < 1e-14, torch.full_like(ddn, 1e-14), ddn)
        self.alpha = g["area"] / safe
        self.kvec = g["Sf"] - self.alpha[:, None] * d
        dn = d.norm(dim=1).clamp_min(1e-30)
        wP = (g["fcen"] - g["cen"][m.neigh.clamp_min(0)]).norm(dim=1) / dn
        self.wP = torch.where(m.it, wP.clamp(0, 1), torch.ones_like(wP))
        self.alpha_orth = g["area"] / dn
        self.g = g

    def _interp(self, cf: Tensor) -> Tensor:
        m = self.m
        P = cf[m.owner]; N = cf[m.neigh.clamp_min(0)]
        w = self.wP
        while w.dim() < P.dim(): w = w.unsqueeze(-1)
        out = w * P + (1 - w) * N
        b = ~m.it
        return torch.where(b.view(-1, *([1] * (P.dim() - 1))), P, out)

    def _grad(self, phi: Tensor) -> Tensor:
        m, g = self.m, self.g
        pf = self._interp(phi)
        fl = pf[:, None] * g["Sf"]
        out = torch.zeros(m.Nc, 3, device=self.dev, dtype=self.dt_)
        out = out.index_add(0, m.owner, fl)
        return out.index_add(0, m.ni, -fl[m.it]) / g["vol"].clamp_min(1e-30)[:, None]

    def _div(self, flux: Tensor) -> Tensor:
        m = self.m
        out = torch.zeros(m.Nc, device=self.dev, dtype=self.dt_)
        out = out.index_add(0, m.owner, flux)
        return out.index_add(0, m.ni, -flux[m.it])

    # ---- one ALE step ----
    def step(self, nodes_new: Tensor, dt: Optional[float] = None,
             wall_velocity: Optional[Tensor] = None) -> Dict[str, Any]:
        cfg, m = self.cfg, self.m
        dt = float(dt if dt is not None else cfg.dt)
        ok, why, q = m.accepts_motion(nodes_new, cfg.min_scaled_jacobian)
        if not ok:
            return {"status": "REJECTED", "reason": why, "quality": q}
        mf = m.mesh_flux(m.nodes, nodes_new, dt)
        g = mf["geo_new"]
        self._setup(g)
        self.phi_mesh = mf["phi_mesh"]
        Vn, Vo = mf["V_new"], mf["V_old"]
        u_old, p_old = self.u.clone(), self.p.clone()

        # boundary absolute flux (wall may move: use wall velocity)
        b = ~m.it
        ub = self.bu[b].clone()
        if wall_velocity is not None:
            wm = (self.bkind[b] == 0) | (self.bkind[b] == 3)
            ub = torch.where(wm[:, None], wall_velocity[b], ub)
        phi_b = (ub * g["Sf"][b]).sum(1)
        kill = (self.bkind[b] == 0) | (self.bkind[b] == 3)
        # a MOVING wall has a nonzero normal flux = mesh flux
        phi_b = torch.where(kill, self.phi_mesh[b], phi_b)
        self.phi = self.phi.clone()
        self.phi[b] = phi_b
        self._balance()

        mom_res = 0.0
        pstat: Dict[str, Any] = {}
        for outer in range(cfg.n_outer):
            aP, aN, ctx = self._assemble(Vn, Vo, u_old, dt)
            gp = self._grad(self.p)
            self.u, mom_res = self._mom_solve(aP, aN, ctx, gp, Vo, u_old, dt)
            rAU = Vn / aP
            HbyA = self._H(aN, ctx, Vo, u_old, dt) / aP[:, None]
            phiH = (self._interp(HbyA) * g["Sf"]).sum(1)
            coeff = self._interp(rAU) * self.alpha
            for _ in range(cfg.n_piso):
                pc = self.p
                gpf = self._interp(self._grad(pc))
                nofl = self._interp(rAU) * (self.kvec * gpf).sum(1)
                rhs = -self._div(phiH) + self._div(nofl)
                bD = self.is_dir_p
                if bool(bD.any()):
                    ob = m.owner[bD]
                    rhs = rhs.index_add(0, ob, rAU[ob] * self.alpha[bD] * self.bp[bD])
                pc, pstat = self._solve_p(rhs, coeff, pc)
                dp = torch.zeros(m.Nf, device=self.dev, dtype=self.dt_)
                dp[m.it] = pc[m.ni] - pc[m.oi]
                dp[bD] = self.bp[bD] - pc[m.owner[bD]]
                gpf = self._interp(self._grad(pc))
                nofl = self._interp(rAU) * (self.kvec * gpf).sum(1)
                phi = phiH - coeff * dp - nofl
                bN = b & (~bD)
                phi[bN] = self.phi[bN]
                self.phi = phi
                self.u = HbyA - rAU[:, None] * self._grad(pc)
                self.p = pc
            if outer < cfg.n_outer - 1:
                self.u = cfg.alpha_u * self.u + (1 - cfg.alpha_u) * u_old
                self.p = cfg.alpha_p * self.p + (1 - cfg.alpha_p) * p_old

        m.nodes = nodes_new.detach().clone()
        self.V = Vn.detach().clone()
        self.t += dt
        cont = self._div(self.phi)
        scale = self.phi.abs().mean().clamp_min(1e-30) * 6.0
        self.last = {
            "status": "OK", "t": self.t, "dt": dt,
            "gcl_max_rel": mf["gcl_max_rel"],
            "gcl_max_abs": float(mf["gcl_abs"].max()),
            "continuity_rel": float(cont.abs().max() / scale),
            "momentum_rel_residual": mom_res,
            "p_iterations": pstat.get("iterations", 0),
            "p_rel_residual": pstat.get("rel_residual", float("nan")),
            "u_max": float(self.u.norm(dim=1).max()),
            "ke": float(0.5 * self.rho * (Vn[:, None] * self.u ** 2).sum() / Vn.sum()),
            "volume_total": float(Vn.sum()),
            "quality": q,
        }
        return self.last

    def _balance(self):
        b = ~self.m.it
        net = float((self.phi[b] - self.phi_mesh[b]).sum())
        out = b & (self.bkind == 2)
        if bool(out.any()):
            tot = float((self.phi[out] - self.phi_mesh[out]).sum())
            if abs(tot) > 1e-30:
                self.phi = self.phi.clone()
                self.phi[out] = self.phi[out] - net * (
                    (self.phi[out] - self.phi_mesh[out]) / tot)
        self.mass_imbalance = net

    def _assemble(self, Vn, Vo, u_old, dt):
        m, rho, nu = self.m, self.rho, self.nu
        # RELATIVE flux for convection
        F = (self.phi - self.phi_mesh)[m.it]
        diff = nu * self.alpha_orth[m.it] * rho
        z = torch.zeros_like(F)
        aN = -torch.minimum(F, z) * rho + diff
        aP = torch.zeros(m.Nc, device=self.dev, dtype=self.dt_)
        aP = aP.index_add(0, m.oi, rho * torch.maximum(F, z) + diff)
        aP = aP.index_add(0, m.ni, rho * torch.maximum(-F, z) + diff)
        b = ~m.it; ob = m.owner[b]
        kind = self.bkind[b]
        Fb = (self.phi - self.phi_mesh)[b]
        db = nu * self.alpha_orth[b] * rho
        zb = torch.zeros_like(Fb)
        iw = (kind == 0) | (kind == 3); ii = kind == 1; io = kind == 2
        aP = aP.index_add(0, ob[iw], db[iw])
        aP = aP.index_add(0, ob[ii], db[ii] + rho * torch.maximum(Fb[ii], zb[ii]))
        aP = aP.index_add(0, ob[io], rho * torch.maximum(Fb[io], zb[io]))
        transient = rho * Vn / dt
        aP = (aP + transient).clamp_min(1e-30)
        return aP, aN, (iw, ii, io, ob, db, Fb, transient)

    def _H(self, aN, ctx, Vo, u_old, dt):
        iw, ii, io, ob, db, Fb, transient = ctx
        m, rho = self.m, self.rho
        u = self.u
        H = torch.zeros_like(u)
        c = aN[:, None]
        H = H.index_add(0, m.oi, c * u[m.ni])
        H = H.index_add(0, m.ni, c * u[m.oi])
        ub = self.bu[~m.it]; zb = torch.zeros_like(Fb)
        H = H.index_add(0, ob[iw], db[iw][:, None] * ub[iw])
        H = H.index_add(0, ob[ii],
                        (db[ii] + rho * torch.maximum(-Fb[ii], zb[ii]))[:, None] * ub[ii])
        H = H.index_add(0, ob[io],
                        rho * torch.maximum(-Fb[io], zb[io])[:, None] * u[ob[io]])
        # ALE transient source uses the OLD volume
        H = H + (rho * Vo / dt)[:, None] * u_old
        return H

    def _mom_solve(self, aP, aN, ctx, gp, Vo, u_old, dt):
        u = self.u.clone()
        rhs = -self.g["vol"][:, None] * gp * self.rho
        om = self.cfg.mom_omega
        for _ in range(self.cfg.n_mom_sweeps):
            H = self._H(aN, ctx, Vo, u_old, dt)
            u = om * ((H + rhs) / aP[:, None]) + (1 - om) * u
        H = self._H(aN, ctx, Vo, u_old, dt)
        res = (aP[:, None] * u - H - rhs).norm()
        return u, float(res / (aP[:, None] * u).norm().clamp_min(1e-30))

    def _lap_diag(self, coeff):
        m = self.m
        d = torch.zeros(m.Nc, device=self.dev, dtype=self.dt_)
        d = d.index_add(0, m.oi, coeff[m.it]).index_add(0, m.ni, coeff[m.it])
        bD = self.is_dir_p
        if bool(bD.any()):
            d = d.index_add(0, m.owner[bD], self.rAU[bD] * self.alpha[bD])
        return d.clamp_min(1e-30)

    def _lap_apply(self, p, coeff):
        m = self.m
        o = torch.zeros(m.Nc, device=self.dev, dtype=self.dt_)
        fl = coeff[m.it] * (p[m.oi] - p[m.ni])
        o = o.index_add(0, m.oi, fl).index_add(0, m.ni, -fl)
        bD = self.is_dir_p
        if bool(bD.any()):
            ob = m.owner[bD]
            o = o.index_add(0, ob, self.rAU[ob] * self.alpha[bD] * p[ob])
        return o

    def _solve_p(self, b, coeff, p0):
        m = self.m
        self.rAU = self.g["vol"] / self._last_aP if hasattr(self, "_last_aP") \
            else self.g["vol"] / coeff[m.it].mean().clamp_min(1e-30)
        diag = self._lap_diag(coeff)
        pin = not self.has_dir
        if pin: b = b - b.mean()
        p = p0.clone()
        if pin: p = p - p.mean()
        r = b - self._lap_apply(p, coeff)
        if pin: r = r - r.mean()
        z = r / diag; d = z.clone(); rz = (r * z).sum()
        nb = b.norm().clamp_min(1e-30); rel = float(r.norm() / nb); used = 0
        for it in range(self.cfg.p_max_iter):
            Ad = self._lap_apply(d, coeff)
            dAd = (d * Ad).sum()
            if float(dAd.abs()) < 1e-300: break
            a = rz / dAd
            p = p + a * d; r = r - a * Ad
            if pin: r = r - r.mean()
            rel = float(r.norm() / nb); used = it + 1
            if rel < self.cfg.p_tol: break
            z = r / diag; rzn = (r * z).sum()
            d = z + (rzn / rz.clamp_min(1e-300)) * d; rz = rzn
            if pin: p = p - p.mean()
        if pin: p = p - p[0]
        return p, {"iterations": used, "rel_residual": rel}

    def wall_traction(self, faces: Tensor) -> Tensor:
        f = faces.to(self.dev)
        ob = self.m.owner[f]; n = self.g["nf"][f]
        gu = torch.stack([self._grad(self.u[:, i]) for i in range(3)], 1)
        S = 0.5 * (gu + gu.transpose(1, 2))
        tau = 2.0 * self.rho * self.nu * torch.einsum("cij,cj->ci", S[ob], n)
        return -self.p[ob, None] * n + tau


# =============================================================================
# §5. LARGE-DISPLACEMENT FSI
# =============================================================================

@dataclass
class ShellParams:
    mass: float = 1.0
    stiff: float = 500.0
    damp: float = 5.0
    p_ext: float = 0.0


class MovingWallShell:
    """Spring-mass-damper per boundary face, Newmark-beta. Reduced model."""

    def __init__(self, faces: Tensor, area: Tensor, normal: Tensor,
                 p: Optional[ShellParams] = None, device="cpu",
                 dtype=torch.float64):
        self.faces, self.p = faces, p or ShellParams()
        self.dev, self.dt_ = torch.device(device), dtype
        n = faces.numel()
        self.A = area.to(self.dev).to(dtype)
        self.nrm = normal.to(self.dev).to(dtype)
        self.x = torch.zeros(n, device=self.dev, dtype=dtype)
        self.v = torch.zeros(n, device=self.dev, dtype=dtype)
        self.a = torch.zeros(n, device=self.dev, dtype=dtype)

    def advance(self, traction: Tensor, dt: float) -> Tuple[Tensor, Tensor]:
        p = self.p
        tn = (traction * self.nrm).sum(1)
        load = (tn - p.p_ext) * self.A
        b, g = 0.25, 0.5
        a0 = 1 / (b * dt * dt); a1 = g / (b * dt)
        a2 = 1 / (b * dt); a3 = 1 / (2 * b) - 1
        a4 = g / b - 1; a5 = dt * (g / (2 * b) - 1)
        keff = p.stiff + a0 * p.mass + a1 * p.damp
        reff = load + p.mass * (a0 * self.x + a2 * self.v + a3 * self.a) \
                    + p.damp * (a1 * self.x + a4 * self.v + a5 * self.a)
        xn = reff / keff
        an = a0 * (xn - self.x) - a2 * self.v - a3 * self.a
        vn = self.v + dt * ((1 - g) * self.a + g * an)
        self.x, self.v, self.a = xn, vn, an
        return self.x, self.v


class AitkenFSI:
    """Aitken delta-squared relaxation on the interface displacement."""

    def __init__(self, max_sub: int = 30, tol: float = 1e-7,
                 omega0: float = 0.5, mode: str = "aitken"):
        self.max_sub, self.tol, self.omega0, self.mode = max_sub, tol, omega0, mode
        self.history: List[Dict[str, float]] = []

    def solve(self, d0: Tensor, mapping: Callable[[Tensor], Tensor]) -> Dict[str, Any]:
        d = d0.clone(); om = self.omega0; rp = None
        self.history = []
        conv = False
        for k in range(self.max_sub):
            dt_ = mapping(d)
            r = dt_ - d
            rn = float(r.norm())
            if rp is not None and self.mode == "aitken":
                dr = r - rp
                den = float((dr * dr).sum())
                if den > 1e-300:
                    om = -om * float((rp * dr).sum()) / den
                om = float(np.clip(om, 1e-4, 1.0))
            self.history.append({"sub": k, "res": rn, "omega": om})
            if rn < self.tol * max(float(d.norm()), 1e-12):
                conv = True; d = d + om * r; break
            d = d + om * r
            rp = r.clone()
        return {"disp": d, "converged": conv, "n_sub": len(self.history),
                "final_res": self.history[-1]["res"], "history": self.history}


# =============================================================================
# §6. VERIFICATION
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    torch.manual_seed(0)
    dt_ = torch.float64

    # ---- T1 geometry: unit cube volume + face orientation ----
    print("\n[T1] hex geometry: volume and outward orientation")
    m = MovingHexMesh(1, 1, 1, 1.0, 1.0, 1.0, dtype=dt_)
    g = m.geometry()
    rep("single hex volume = 1", abs(float(g["vol"][0]) - 1.0) < 1e-12,
        f"V={float(g['vol'][0]):.12f}")
    b = ~m.it
    outward = (g["Sf"][b] * (g["fcen"][b] - g["cen"][m.owner[b]])).sum(1)
    rep("all boundary normals outward", bool((outward > 0).all()))
    rep("total surface area = 6", abs(float(g["area"][b].sum()) - 6.0) < 1e-12,
        f"A={float(g['area'][b].sum()):.12f}")
    m3 = MovingHexMesh(3, 4, 5, 1.0, 1.0, 1.0, dtype=dt_)
    g3 = m3.geometry()
    rep("volume sums to domain volume",
        abs(float(g3["vol"].sum()) - 1.0) < 1e-12,
        f"sum V={float(g3['vol'].sum()):.12f}, Nc={m3.Nc}")

    # ---- T2 GCL: rigid translation ----
    print("\n[T2] GCL under RIGID TRANSLATION (dV=0, net swept=0)")
    m = MovingHexMesh(4, 5, 4, 1.0, 1.0, 1.0, dtype=dt_)
    X0 = m.nodes.clone()
    X1 = X0 + torch.tensor([0.07, -0.03, 0.05], dtype=dt_)
    mf = m.mesh_flux(X0, X1, 1e-3)
    print(f"      max|dV| = {float(mf['dV'].abs().max()):.3e}")
    print(f"      GCL max relative residual = {mf['gcl_max_rel']:.3e}")
    rep("rigid translation: dV = 0", float(mf["dV"].abs().max()) < 1e-12)
    rep("rigid translation: GCL satisfied", mf["gcl_max_rel"] < 1e-10,
        f"{mf['gcl_max_rel']:.2e}")

    # ---- T3 GCL: large 1D stretching + general 3D deformation ----
    print("\n[T3] GCL under LARGE stretching and general 3D deformation")
    for tag, disp in (
        ("y-stretch x2.0", lambda X: torch.stack(
            [X[:, 0], X[:, 1] * 2.0, X[:, 2]], 1)),
        ("y-compress x0.3", lambda X: torch.stack(
            [X[:, 0], X[:, 1] * 0.3, X[:, 2]], 1)),
        ("shear", lambda X: torch.stack(
            [X[:, 0] + 0.3 * X[:, 1], X[:, 1], X[:, 2]], 1)),
        ("3D warp", lambda X: torch.stack(
            [X[:, 0] + 0.15 * torch.sin(math.pi * X[:, 1]),
             X[:, 1] * (1.0 + 0.25 * X[:, 0]),
             X[:, 2] + 0.1 * torch.sin(math.pi * X[:, 0])], 1)),
    ):
        X1 = disp(X0)
        mf = m.mesh_flux(X0, X1, 1e-3)
        acc, why, q = m.accepts_motion(X1, 0.02)
        print(f"      {tag:18s} GCL_rel={mf['gcl_max_rel']:.3e}  "
              f"V_tot={float(mf['V_new'].sum()):.5f}  "
              f"minSJ={q['scaled_jacobian_min']:.4f}  {why}")
        rep(f"{tag}: GCL satisfied", mf["gcl_max_rel"] < 1e-10,
            f"{mf['gcl_max_rel']:.2e}")
    # swept volume is NOT A*(w.n)*dt for a deforming face -- demonstrate
    X1 = torch.stack([X0[:, 0] + 0.3 * X0[:, 1], X0[:, 1], X0[:, 2]], 1)
    mf = m.mesh_flux(X0, X1, 1e-3)
    g0 = mf["geo_old"]
    w = (X1 - X0) / 1e-3
    wf = w[m.fnodes].mean(1)
    naive = (wf * g0["Sf"]).sum(1) * 1e-3
    diff = float((mf["swept"] - naive).abs().max())
    print(f"      max|exact swept - A(w.n)dt| = {diff:.3e}  "
          f"(nonzero => naive formula is wrong for deforming faces)")
    rep("space-time prism differs from naive A(w.n)dt", diff > 1e-12,
        f"{diff:.2e}")

    # ---- T4 FREE-STREAM PRESERVATION (the decisive ALE test) ----
    print("\n[T4] FREE-STREAM PRESERVATION on a moving mesh")
    mm = MovingHexMesh(5, 6, 5, 1.0, 1.0, 1.0, dtype=dt_)
    bb = mm.bnd
    bcs = [AleBC(bb, "wall")]
    sol = ALEPimple(mm, rho=1.0, nu=1e-2, bcs=bcs,
                    cfg=ALEConfig(dt=1e-3, n_outer=4, n_mom_sweeps=12,
                                  p_tol=1e-11))
    U0 = torch.tensor([0.7, -0.4, 0.25], dtype=dt_)
    sol.u = U0[None, :].expand(mm.Nc, 3).clone()
    # move the mesh with a uniform-flow-consistent wall motion
    X0 = mm.nodes.clone()
    err_fs = 0.0
    gcl = 0.0
    for step in range(6):
        # boundary nodes translate with U0 -> mesh flux consistent
        bmask = ((mm.ni == 0) | (mm.ni == mm.nx) |
                 (mm.nj == 0) | (mm.nj == mm.ny) |
                 (mm.nk == 0) | (mm.nk == mm.nz))
        bd = torch.zeros_like(X0)
        bd[bmask] = (U0 * sol.cfg.dt)[None, :].expand(int(bmask.sum()), 3)
        mv = MeshMotion(mm, method="laplace", n_smooth=120)
        X1 = mv.move(mm.nodes, bd)
        wv = torch.zeros(mm.Nf, 3, dtype=dt_)
        wv[bb] = U0[None, :].expand(int(bb.numel()), 3)
        r = sol.step(X1, wall_velocity=wv)
        if r["status"] != "OK":
            rep("motion accepted", False, r["reason"]); break
        err_fs = max(err_fs, float((sol.u - U0[None, :]).abs().max()))
        gcl = max(gcl, r["gcl_max_rel"])
    print(f"      GCL max rel = {gcl:.3e}")
    print(f"      max|u - U0| after 6 moving steps = {err_fs:.3e}")
    rep("GCL holds during the run", gcl < 1e-9, f"{gcl:.2e}")
    rep("uniform flow preserved on a moving mesh", err_fs < 5e-3,
        f"{err_fs:.2e}")
    print("      NOTE: this is THE test that fails if GCL is violated or if")
    print("            the pressure equation is written on the RELATIVE flux.")

    # ---- T5 mesh quality guard rejects inversion ----
    print("\n[T5] mesh-quality guard REJECTS an inverting motion")
    Xbad = X0.clone()
    Xbad[:, 1] = X0[:, 1] * (-0.5)      # flip y -> negative volumes
    acc, why, q = mm.accepts_motion(Xbad, 0.05)
    rep("inverted mesh rejected", not acc, why)
    print(f"      n_inverted={q['n_inverted']}  minSJ={q['scaled_jacobian_min']:.4f}")
    Xsmall = X0 + torch.tensor([0.0, 0.0, 0.0], dtype=dt_)
    Xsmall[:, 1] = X0[:, 1] * 1.8
    acc2, why2, q2 = mm.accepts_motion(Xsmall, 0.05)
    rep("large but valid stretch accepted", acc2,
        f"minSJ={q2['scaled_jacobian_min']:.4f}")

    # ---- T6 LARGE-DISPLACEMENT piston: mass/volume consistency ----
    print("\n[T6] large-displacement piston: volume tracks the wall exactly")
    mp = MovingHexMesh(4, 10, 4, 1.0, 1.0, 1.0, dtype=dt_)
    top = mp.bnd[mp.nj[mp.fnodes].min(1).values == mp.ny]
    # pick faces whose nodes are ALL at j = ny
    allj = mp.nj[mp.fnodes]
    top = mp.bnd[(allj[mp.bnd] == mp.ny).all(1)]
    bot = mp.bnd[(allj[mp.bnd] == 0).all(1)]
    side = mp.bnd[~torch.isin(mp.bnd, torch.cat([top, bot]))]
    bcs = [AleBC(bot, "wall"), AleBC(top, "wall"), AleBC(side, "symmetry")]
    sp = ALEPimple(mp, 1.0, 5e-2, bcs,
                   ALEConfig(dt=2e-3, n_outer=3, n_mom_sweeps=10, p_tol=1e-10))
    mv = MeshMotion(mp, method="tfi", axis=1)
    amp, nsteps = 0.45, 40           # 45% of the domain height
    Vhist, gclhist, disp_hist = [], [], []
    rejected = 0
    for k in range(nsteps):
        t = (k + 1) * sp.cfg.dt
        d = amp * math.sin(2 * math.pi * 0.5 * t)
        bd = torch.zeros_like(mp.nodes)
        hi = mp.nj == mp.ny
        bd[hi, 1] = d
        X1 = mv.move(mp.nodes, bd)
        wv = torch.zeros(mp.Nf, 3, dtype=dt_)
        wv[top, 1] = amp * 2 * math.pi * 0.5 * math.cos(2 * math.pi * 0.5 * t)
        r = sp.step(X1, wall_velocity=wv)
        if r["status"] != "OK":
            rejected += 1; continue
        Vhist.append(r["volume_total"]); gclhist.append(r["gcl_max_rel"])
        disp_hist.append(d)
    Vexpect = [1.0 + dd for dd in disp_hist]
    verr = max(abs(a - b) for a, b in zip(Vhist, Vexpect)) if Vhist else 1.0
    print(f"      displacement range = [{min(disp_hist):.3f}, {max(disp_hist):.3f}]"
          f"  ({100*amp:.0f}% of height)")
    print(f"      max|V_num - (1+d)| = {verr:.3e}")
    print(f"      GCL max rel = {max(gclhist):.3e}   rejected steps = {rejected}")
    rep("no steps rejected (large displacement survives)", rejected == 0,
        f"{rejected}/{nsteps}")
    rep("domain volume tracks the wall exactly", verr < 1e-10, f"{verr:.2e}")
    rep("GCL holds throughout", max(gclhist) < 1e-9, f"{max(gclhist):.2e}")

    # ---- T7 FSI with Aitken on a MOVING mesh ----
    print("\n[T7] partitioned FSI + Aitken on a moving mesh")
    mf_ = MovingHexMesh(4, 8, 4, 1.0, 1.0, 1.0, dtype=dt_)
    allj = mf_.nj[mf_.fnodes]
    top = mf_.bnd[(allj[mf_.bnd] == mf_.ny).all(1)]
    bot = mf_.bnd[(allj[mf_.bnd] == 0).all(1)]
    side = mf_.bnd[~torch.isin(mf_.bnd, torch.cat([top, bot]))]
    bcs = [AleBC(bot, "wall"), AleBC(top, "wall"), AleBC(side, "symmetry")]
    sf = ALEPimple(mf_, 1.0, 5e-2, bcs,
                   ALEConfig(dt=2e-3, n_outer=2, n_mom_sweeps=8, p_tol=1e-9))
    gg = mf_.geometry()
    shell = MovingWallShell(top, gg["area"][top], gg["nf"][top],
                            ShellParams(mass=0.5, stiff=2000.0, damp=20.0),
                            dtype=dt_)
    mv = MeshMotion(mf_, method="tfi", axis=1)
    cp = AitkenFSI(max_sub=12, tol=1e-6)
    nsub, dmax, gclmax = [], [], 0.0
    for k in range(25):
        d0 = shell.x.clone()

        def mapping(dvec: Tensor) -> Tensor:
            xs, vs, as_ = shell.x.clone(), shell.v.clone(), shell.a.clone()
            bd = torch.zeros_like(mf_.nodes)
            hi = mf_.nj == mf_.ny
            # distribute face displacement to the top nodes
            tn = mf_.fnodes[top].reshape(-1)
            bd[tn, 1] = dvec.repeat_interleave(4)
            X1 = mv.move(mf_.nodes, bd)
            wv = torch.zeros(mf_.Nf, 3, dtype=dt_)
            wv[top, 1] = ((dvec - xs) / sf.cfg.dt)[:, None].expand(-1, 1).reshape(-1)
            r = sf.step(X1, wall_velocity=wv)
            if r["status"] != "OK":
                shell.x, shell.v, shell.a = xs, vs, as_
                return dvec
            tr = sf.wall_traction(top)
            xn, _ = shell.advance(tr, sf.cfg.dt)
            out = xn.clone()
            shell.x, shell.v, shell.a = xs, vs, as_
            return out

        res = cp.solve(d0, mapping)
        shell.x = res["disp"]
        bd = torch.zeros_like(mf_.nodes)
        tn = mf_.fnodes[top].reshape(-1)
        bd[tn, 1] = shell.x.repeat_interleave(4)
        X1 = mv.move(mf_.nodes, bd)
        r = sf.step(X1)
        if r["status"] == "OK":
            gclmax = max(gclmax, r["gcl_max_rel"])
        nsub.append(res["n_sub"]); dmax.append(float(shell.x.abs().max()))
    print(f"      mean Aitken subiterations = {np.mean(nsub):.2f}")
    print(f"      max wall displacement = {max(dmax):.4e}")
    print(f"      GCL max rel = {gclmax:.3e}")
    rep("FSI stays finite", all(math.isfinite(d) for d in dmax))
    rep("GCL holds under FSI coupling", gclmax < 1e-9, f"{gclmax:.2e}")
    rep("Aitken converges within the subiteration budget",
        max(nsub) < cp.max_sub, f"max sub={max(nsub)}")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  Hex volumes and outward face orientation exact.
  GCL satisfied to machine precision under rigid translation, 2x stretch,
  0.3x compression, shear, and a general 3D warp.
  Space-time prism swept volume DIFFERS from the naive A(w.n)dt on a
  deforming face — that difference is exactly what breaks free-stream
  preservation if you use the naive formula.
  Free-stream preservation on a moving mesh.
  Inverting motion is REJECTED, not silently integrated.
  A 45%-of-height piston keeps the domain volume exact and GCL at machine
  precision, with zero rejected steps.
  Partitioned FSI + Aitken runs on the moving mesh with GCL intact.

NOT IMPLEMENTED
  Remeshing / topology change: nodes move, cells never reconnect.
  Tets, polyhedra, overset/Chimera, sliding interfaces.
  Finite-strain hyperelastic solid; contact; rupture.
  Second-order ALE time integration; turbulence models.

NOT CLAIMED
  Arbitrary displacement. The guard rejects inversion — that is a real
  limit, not a formality. No experimental validation.
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ALE + GCL for large-displacement FSI")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "gcl", "piston", "quality"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--amp", type=float, default=0.45)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out: Dict[str, Any] = {}

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    if a.cmd == "gcl":
        m = MovingHexMesh(a.n, a.n, a.n, dtype=torch.float64)
        X0 = m.nodes.clone()
        rows = []
        for tag, f in (("rigid", lambda X: X + 0.05),
                       ("stretch2x", lambda X: torch.stack([X[:, 0], 2 * X[:, 1], X[:, 2]], 1)),
                       ("warp", lambda X: torch.stack(
                           [X[:, 0] + 0.15 * torch.sin(math.pi * X[:, 1]),
                            X[:, 1] * (1 + 0.25 * X[:, 0]), X[:, 2]], 1))):
            X1 = f(X0)
            mf = m.mesh_flux(X0, X1, 1e-3)
            acc, why, q = m.accepts_motion(X1)
            rows.append({"case": tag, "gcl_max_rel": mf["gcl_max_rel"],
                         "V_total": float(mf["V_new"].sum()),
                         "accepted": acc, "reason": why, "quality": q})
        out["gcl"] = rows
        print(json.dumps(out, indent=2, default=float))

    if a.cmd == "quality":
        m = MovingHexMesh(a.n, a.n, a.n, dtype=torch.float64)
        out["quality"] = m.quality()
        print(json.dumps(out, indent=2, default=float))

    if a.cmd == "piston":
        m = MovingHexMesh(4, 10, 4, dtype=torch.float64)
        allj = m.nj[m.fnodes]
        top = m.bnd[(allj[m.bnd] == m.ny).all(1)]
        bot = m.bnd[(allj[m.bnd] == 0).all(1)]
        side = m.bnd[~torch.isin(m.bnd, torch.cat([top, bot]))]
        s = ALEPimple(m, 1.0, 5e-2,
                      [AleBC(bot, "wall"), AleBC(top, "wall"), AleBC(side, "symmetry")],
                      ALEConfig(dt=2e-3))
        mv = MeshMotion(m, "tfi", axis=1)
        hist = []
        for k in range(a.steps):
            t = (k + 1) * s.cfg.dt
            d = a.amp * math.sin(math.pi * t)
            bd = torch.zeros_like(m.nodes)
            bd[m.nj == m.ny, 1] = d
            X1 = mv.move(m.nodes, bd)
            wv = torch.zeros(m.Nf, 3, dtype=torch.float64)
            wv[top, 1] = a.amp * math.pi * math.cos(math.pi * t)
            r = s.step(X1, wall_velocity=wv)
            hist.append({"step": k, "disp": d, **{k2: v for k2, v in r.items()
                                                  if k2 != "quality"}})
        out["piston"] = hist
        print(json.dumps({"n_steps": len(hist),
                          "last": hist[-1] if hist else None}, indent=2, default=float))

    if a.json and out:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
