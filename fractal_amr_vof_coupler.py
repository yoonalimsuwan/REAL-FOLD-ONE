#!/usr/bin/env python3
"""
=============================================================================
FRACTAL AMR ↔ SESI MULTIPHASE FLOW COUPLER
Energy-measure AMR error indicator + real coarse-fine reflux + subcycling
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT THIS FILE DOES
  1. Connects the fractal AMR engine to sesi_multiphase_flow.py.
  2. Uses a finite-level ENERGY MEASURE diagnostic as the AMR error
     indicator, following the Carpet-19 martingale density idea:
        X_m(S) = Gamma_m(S) / mu_m(S)
     where Gamma_m is coarse-grained edge energy and mu_m is the
     fractal Hausdorff-type cell measure.
  3. Implements a real, though simplified, patch AMR cycle:
       - tag cells from indicator
       - build fine patch (ratio 2)
       - prolongate alpha and velocity to fine patch
       - subcycle fine patch r times
       - accumulate fine/coarse boundary fluxes
       - reflux coarse cells adjacent to fine patch
       - restrict fine alpha back to coarse cells
  4. Provides fractal backends:
       - Sierpinski gasket: p.c.f., easier
       - Sierpinski carpet: non-p.c.f., supported by Carpet-19
       - Menger sponge: non-p.c.f., I1-I4 must be checked
       - Vicsek: variant-dependent; some are finitely ramified
       - random fractals: require Paper 4/5 stochastic homogenization
         diagnostics before licensing

WHAT THIS IS NOT
  ❌ Not a full NS AMR solver. Momentum reflux, pressure projection on
     patches, and coarse-fine Maxwell/velocity constraints are not included.
  ❌ Not validated against OpenFOAM/Basilisk/analytic AMR benchmarks yet.
  ❌ Not a proof that Menger/Vicsek/random cases satisfy Carpet-19 I1-I4.
=============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

VERSION = "1.0.0"


# =============================================================================
# 0. FRACTAL BACKENDS
# =============================================================================

@dataclass
class FractalBackend:
    name: str
    pcf: bool
    natural_ratio: int
    cells_per_generator: int
    dimension: float
    resistance_rho: Optional[float]
    status: str
    required_paper_basis: str

    def block_sizes(self, min_grid: int, max_levels: int = 3) -> List[int]:
        sizes = []
        b = self.natural_ratio
        for lev in range(1, max_levels + 1):
            bs = b ** lev
            if bs <= min_grid:
                sizes.append(bs)
        if not sizes:
            # fallback: use dyadic blocks if the grid is too small for the
            # natural ratio. This is a numerical surrogate, not the fractal's
            # exact scaling.
            b = 2
            for lev in range(1, max_levels + 1):
                bs = b ** lev
                if bs <= min_grid:
                    sizes.append(bs)
        return sizes or [1]


def sg_backend() -> FractalBackend:
    return FractalBackend(
        name="SierpinskiGasket",
        pcf=True,
        natural_ratio=2,
        cells_per_generator=3,
        dimension=math.log(3) / math.log(2),
        resistance_rho=5.0 / 3.0,
        status="p.c.f.; easier; Kigami theory available",
        required_paper_basis="Kigami; not Carpet-19 dependent",
    )


def carpet_backend() -> FractalBackend:
    return FractalBackend(
        name="SierpinskiCarpet",
        pcf=False,
        natural_ratio=3,
        cells_per_generator=8,
        dimension=math.log(8) / math.log(3),
        resistance_rho=1.25,  # numerical lower-bound proxy; Carpet-19 proves rho >= 9/8
        status="non-p.c.f.; Carpet-19 gives conditional theorem under I1-I4",
        required_paper_basis="Carpet-19 Theorem 10, Prop 1, Thm 1, Thm 4",
    )


def menger_backend() -> FractalBackend:
    return FractalBackend(
        name="MengerSponge",
        pcf=False,
        natural_ratio=3,
        cells_per_generator=20,
        dimension=math.log(20) / math.log(3),
        resistance_rho=None,
        status="non-p.c.f.; I1-I4 NOT verified here",
        required_paper_basis="requires new resistance/Mosco analysis",
    )


def vicsek_backend() -> FractalBackend:
    return FractalBackend(
        name="Vicsek",
        pcf=False,  # variant-dependent; standard cross may be finitely ramified
        natural_ratio=3,
        cells_per_generator=5,
        dimension=math.log(5) / math.log(3),
        resistance_rho=None,
        status="variant-dependent; literature partial; verify I1-I4 for your variant",
        required_paper_basis="partial literature; Carpet-19 does not automatically apply",
    )


def random_backend() -> FractalBackend:
    return FractalBackend(
        name="RandomFractal",
        pcf=False,
        natural_ratio=2,
        cells_per_generator=0,
        dimension=float("nan"),
        resistance_rho=None,
        status="requires Paper 4/5 stochastic homogenization diagnostics",
        required_paper_basis="Paper 4 Hyp(SA); Paper 5 AMS/log-scale/residual",
    )


BACKENDS = {
    "sg": sg_backend,
    "carpet": carpet_backend,
    "menger": menger_backend,
    "vicsek": vicsek_backend,
    "random": random_backend,
}


def backend_report() -> List[Dict[str, Any]]:
    out = []
    for k, fn in BACKENDS.items():
        b = fn()
        out.append({
            "key": k,
            "name": b.name,
            "pcf": b.pcf,
            "dimension": b.dimension,
            "rho": b.resistance_rho,
            "status": b.status,
            "paper_basis": b.required_paper_basis,
        })
    return out


# =============================================================================
# 1. ENERGY-MEASURE AMR INDICATOR
# =============================================================================

def fractal_energy_indicator(
    q: torch.Tensor,
    backend: FractalBackend,
    max_levels: int = 3,
    eps: float = 1e-20,
) -> torch.Tensor:
    """
    Finite-level energy-measure indicator.

    For each block size b, compute
        Gamma_b(S) = sum of squared discrete edge differences inside block S
        mu_b(S)    ~ b^{-D}  (equal blocks => uniform normalized measure)
        X_b(S)     = Gamma_b(S) / mu_b(S)

    Large X_b means energy concentrates more than the reference fractal
    measure predicts. This is the finite-level analogue of the Carpet-19
    martingale diagnostic X_m = dGamma/dmu.

    Returns a normalized field in [0,1] on the same grid as q.
    """
    if q.dim() != 3:
        raise ValueError("q must be (Nx, Ny, Nz)")

    dev = q.device
    dtype = q.dtype
    indicator = torch.zeros_like(q, dtype=dtype)
    count = torch.zeros_like(q, dtype=dtype)

    # discrete edge energy
    e = torch.zeros_like(q)
    for ax in range(3):
        d = torch.roll(q, -1, ax) - q
        e = e + d * d

    min_grid = min(q.shape)
    sizes = backend.block_sizes(min_grid, max_levels)
    D = backend.dimension if math.isfinite(backend.dimension) else 2.0

    for b in sizes:
        if b <= 1:
            continue
        nx = (q.shape[0] // b) * b
        ny = (q.shape[1] // b) * b
        nz = (q.shape[2] // b) * b
        if nx == 0 or ny == 0 or nz == 0:
            continue

        eb = e[:nx, :ny, :nz]
        nbx, nby, nbz = nx // b, ny // b, nz // b
        blocks = eb.reshape(nbx, b, nby, b, nbz, b)
        gamma = blocks.sum(dim=(1, 3, 5))  # (nbx,nby,nbz)

        total = float(gamma.sum())
        if total <= eps:
            continue

        nblocks = gamma.numel()
        # mu is uniform over equal blocks; X = gamma / (total/nblocks)
        X = gamma * float(nblocks) / (total + eps)
        score = torch.log1p(X)

        score_full = score.repeat_interleave(b, 0) \
                          .repeat_interleave(b, 1) \
                          .repeat_interleave(b, 2)

        indicator[:nx, :ny, :nz] += score_full
        count[:nx, :ny, :nz] += 1.0

    count = count.clamp_min(1.0)
    indicator = indicator / count

    imax = float(indicator.max())
    if imax > eps:
        indicator = indicator / imax
    return indicator


# =============================================================================
# 2. SIMPLE SCALAR FINITE-VOLUME ADVECTION (for AMR transport)
# =============================================================================

def advect_periodic(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    dx: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    First-order upwind finite-volume advection on a periodic grid.
    Velocity is cell-centred. Returns q_new and face flux densities.
    """
    ux = 0.5 * (u + torch.roll(u, -1, 0))
    uy = 0.5 * (v + torch.roll(v, -1, 1))
    uz = 0.5 * (w + torch.roll(w, -1, 2))

    qx = torch.where(ux > 0, q, torch.roll(q, -1, 0))
    qy = torch.where(uy > 0, q, torch.roll(q, -1, 1))
    qz = torch.where(uz > 0, q, torch.roll(q, -1, 2))

    Fx = ux * qx
    Fy = uy * qy
    Fz = uz * qz

    divF = ((Fx - torch.roll(Fx, 1, 0))
            + (Fy - torch.roll(Fy, 1, 1))
            + (Fz - torch.roll(Fz, 1, 2))) / dx

    q_new = q - dt * divF
    return q_new, Fx, Fy, Fz


# =============================================================================
# 3. PATCH AMR WITH SUBCYCLING AND REFLUX
# =============================================================================

@dataclass
class AMRBox:
    lo: Tuple[int, int, int]
    hi: Tuple[int, int, int]

    @property
    def size(self) -> Tuple[int, int, int]:
        return (self.hi[0] - self.lo[0],
                self.hi[1] - self.lo[1],
                self.hi[2] - self.lo[2])


def bounding_box(mask: torch.Tensor, pad: int = 1) -> Optional[AMRBox]:
    idx = mask.nonzero()
    if idx.numel() == 0:
        return None
    lo = idx.min(dim=0).values.tolist()
    hi = (idx.max(dim=0).values + 1).tolist()
    shape = mask.shape
    lo = tuple(max(0, lo[i] - pad) for i in range(3))
    hi = tuple(min(shape[i], hi[i] + pad) for i in range(3))
    if any(hi[i] <= lo[i] for i in range(3)):
        return None
    return AMRBox(lo=tuple(lo), hi=tuple(hi))


def prolongate_block(field: torch.Tensor, box: AMRBox, r: int) -> torch.Tensor:
    sub = field[box.lo[0]:box.hi[0],
                box.lo[1]:box.hi[1],
                box.lo[2]:box.hi[2]]
    return (sub.repeat_interleave(r, 0)
               .repeat_interleave(r, 1)
               .repeat_interleave(r, 2))


def restrict_block(fine: torch.Tensor, r: int) -> torch.Tensor:
    fx, fy, fz = fine.shape
    sx, sy, sz = fx // r, fy // r, fz // r
    return fine.reshape(sx, r, sy, r, sz, r).mean(dim=(1, 3, 5))


def coarse_indices_for_ext(
    box: AMRBox,
    ext_shape: Tuple[int, int, int],
    r: int,
    grid_shape: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = []
    for ax in range(3):
        ext_len = ext_shape[ax]
        fine_cell = torch.arange(ext_len, device=device) - 1
        off = torch.div(fine_cell, r, rounding_mode="floor")
        ci = box.lo[ax] + off
        ci = ci % grid_shape[ax]
        idx.append(ci)
    return idx[0], idx[1], idx[2]


def make_ext_fields(
    q_f: torch.Tensor,
    uf: torch.Tensor,
    vf: torch.Tensor,
    wf: torch.Tensor,
    cq: torch.Tensor,
    cu: torch.Tensor,
    cv: torch.Tensor,
    cw: torch.Tensor,
    box: AMRBox,
    r: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fx, fy, fz = q_f.shape
    ext_shape = (fx + 2, fy + 2, fz + 2)
    ix, iy, iz = coarse_indices_for_ext(box, ext_shape, r, cq.shape, q_f.device)

    q_ext = cq[ix[:, None, None], iy[None, :, None], iz[None, None, :]].clone()
    u_ext = cu[ix[:, None, None], iy[None, :, None], iz[None, None, :]].clone()
    v_ext = cv[ix[:, None, None], iy[None, :, None], iz[None, None, :]].clone()
    w_ext = cw[ix[:, None, None], iy[None, :, None], iz[None, None, :]].clone()

    q_ext[1:-1, 1:-1, 1:-1] = q_f
    u_ext[1:-1, 1:-1, 1:-1] = uf
    v_ext[1:-1, 1:-1, 1:-1] = vf
    w_ext[1:-1, 1:-1, 1:-1] = wf
    return q_ext, u_ext, v_ext, w_ext


def fluxes_ext(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Face flux densities on an extended patch grid.
    Fx[i] is the flux between ext cell i and i+1.
    """
    ux = 0.5 * (u[:-1] + u[1:])
    uy = 0.5 * (v[:, :-1] + v[:, 1:])
    uz = 0.5 * (w[:, :, :-1] + w[:, :, 1:])

    qx = torch.where(ux > 0, q[:-1], q[1:])
    qy = torch.where(uy > 0, q[:, :-1], q[:, 1:])
    qz = torch.where(uz > 0, q[:, :, :-1], q[:, :, 1:])

    Fx = ux * qx
    Fy = uy * qy
    Fz = uz * qz
    return Fx, Fy, Fz


def advance_fine_patch(
    q_f: torch.Tensor,
    uf: torch.Tensor,
    vf: torch.Tensor,
    wf: torch.Tensor,
    cq: torch.Tensor,
    cu: torch.Tensor,
    cv: torch.Tensor,
    cw: torch.Tensor,
    box: AMRBox,
    dx_c: float,
    r: int,
    dt_f: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Advance fine patch by one fine time step.
    Returns updated fine field and boundary flux densities.
    """
    q_ext, u_ext, v_ext, w_ext = make_ext_fields(
        q_f, uf, vf, wf, cq, cu, cv, cw, box, r)

    dx_f = dx_c / r
    Fx, Fy, Fz = fluxes_ext(q_ext, u_ext, v_ext, w_ext)

    q_new_ext = q_ext.clone()
    q_new_ext[1:-1, 1:-1, 1:-1] -= dt_f / dx_f * (
        (Fx[1:, 1:-1, 1:-1] - Fx[:-1, 1:-1, 1:-1])
        + (Fy[1:-1, 1:, 1:-1] - Fy[1:-1, :-1, 1:-1])
        + (Fz[1:-1, 1:-1, 1:] - Fz[1:-1, 1:-1, :-1])
    )

    bflux = {
        "x_low":  Fx[0, 1:-1, 1:-1].clone(),
        "x_high": Fx[-1, 1:-1, 1:-1].clone(),
        "y_low":  Fy[1:-1, 0, 1:-1].clone(),
        "y_high": Fy[1:-1, -1, 1:-1].clone(),
        "z_low":  Fz[1:-1, 1:-1, 0].clone(),
        "z_high": Fz[1:-1, 1:-1, -1].clone(),
    }
    return q_new_ext[1:-1, 1:-1, 1:-1], bflux


class AMRScalarSolver:
    """
    Real patch-AMR cycle for a scalar conserved field (alpha).

    This is not a controller-only wrapper:
      • fine patches are created,
      • fine patches subcycle with dt_f = dt_c / r,
      • fine/coarse boundary fluxes are registered,
      • coarse cells adjacent to fine patches are refluxed,
      • fine fields are restricted back to coarse cells.
    """

    def __init__(
        self,
        backend: Optional[FractalBackend] = None,
        refine_threshold: float = 0.20,
        coarsen_threshold: float = 0.02,
        ratio: int = 2,
        pad: int = 1,
    ):
        self.backend = backend or carpet_backend()
        self.refine_threshold = refine_threshold
        self.coarsen_threshold = coarsen_threshold
        self.ratio = ratio
        self.pad = pad

    def step(
        self,
        q: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        dx: float,
        dt: float,
        force_box: Optional[AMRBox] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        r = self.ratio
        shape = q.shape

        indicator = fractal_energy_indicator(q, self.backend)
        if force_box is None:
            tags = indicator > self.refine_threshold
            box = bounding_box(tags, pad=self.pad)
        else:
            box = force_box

        # coarse advance everywhere
        q_c, Fx_c, Fy_c, Fz_c = advect_periodic(q, u, v, w, dx, dt)

        diag: Dict[str, Any] = {
            "has_patch": box is not None,
            "backend": self.backend.name,
            "indicator_max": float(indicator.max()),
            "indicator_mean": float(indicator.mean()),
        }

        if box is None:
            return q_c, diag

        sx, sy, sz = box.size
        if min(sx, sy, sz) < 1:
            return q_c, diag

        # prolongate to fine patch
        q_f = prolongate_block(q, box, r)
        uf = prolongate_block(u, box, r)
        vf = prolongate_block(v, box, r)
        wf = prolongate_block(w, box, r)

        # flux registers (time-integrated fine boundary fluxes)
        regs = {
            "x_low": torch.zeros(sy, sz, device=q.device, dtype=q.dtype),
            "x_high": torch.zeros(sy, sz, device=q.device, dtype=q.dtype),
            "y_low": torch.zeros(sx, sz, device=q.device, dtype=q.dtype),
            "y_high": torch.zeros(sx, sz, device=q.device, dtype=q.dtype),
            "z_low": torch.zeros(sx, sy, device=q.device, dtype=q.dtype),
            "z_high": torch.zeros(sx, sy, device=q.device, dtype=q.dtype),
        }

        dt_f = dt / r

        # subcycle fine patch
        for _ in range(r):
            q_f, bflux = advance_fine_patch(
                q_f, uf, vf, wf,
                q, u, v, w,
                box, dx, r, dt_f,
            )

            regs["x_low"] += dt_f * bflux["x_low"].reshape(sy, r, sz, r).mean(dim=(1, 3))
            regs["x_high"] += dt_f * bflux["x_high"].reshape(sy, r, sz, r).mean(dim=(1, 3))
            regs["y_low"] += dt_f * bflux["y_low"].reshape(sx, r, sz, r).mean(dim=(1, 3))
            regs["y_high"] += dt_f * bflux["y_high"].reshape(sx, r, sz, r).mean(dim=(1, 3))
            regs["z_low"] += dt_f * bflux["z_low"].reshape(sx, r, sy, r).mean(dim=(1, 3))
            regs["z_high"] += dt_f * bflux["z_high"].reshape(sx, r, sy, r).mean(dim=(1, 3))

        # restrict fine patch back onto coarse field
        q_fine_coarse = restrict_block(q_f, r)
        q_c[box.lo[0]:box.hi[0],
            box.lo[1]:box.hi[1],
            box.lo[2]:box.hi[2]] = q_fine_coarse

        # reflux coarse cells adjacent to patch boundaries
        Nx, Ny, Nz = shape

        # x-low: outside cell = lo[0]-1, face index = lo[0]-1
        i_out_low = (box.lo[0] - 1) % Nx
        i_face_low = i_out_low
        # x-high: outside cell = hi[0], face index = hi[0]-1
        i_out_high = box.hi[0] % Nx
        i_face_high = (box.hi[0] - 1) % Nx

        for jj, j in enumerate(range(box.lo[1], box.hi[1])):
            for kk, k in enumerate(range(box.lo[2], box.hi[2])):
                q_c[i_out_low, j, k] += (
                    dt * Fx_c[i_face_low, j, k] - regs["x_low"][jj, kk]
                ) / dx
                q_c[i_out_high, j, k] += (
                    regs["x_high"][jj, kk] - dt * Fx_c[i_face_high, j, k]
                ) / dx

        # y boundaries
        j_out_low = (box.lo[1] - 1) % Ny
        j_face_low = j_out_low
        j_out_high = box.hi[1] % Ny
        j_face_high = (box.hi[1] - 1) % Ny

        for ii, i in enumerate(range(box.lo[0], box.hi[0])):
            for kk, k in enumerate(range(box.lo[2], box.hi[2])):
                q_c[i, j_out_low, k] += (
                    dt * Fy_c[i, j_face_low, k] - regs["y_low"][ii, kk]
                ) / dx
                q_c[i, j_out_high, k] += (
                    regs["y_high"][ii, kk] - dt * Fy_c[i, j_face_high, k]
                ) / dx

        # z boundaries
        k_out_low = (box.lo[2] - 1) % Nz
        k_face_low = k_out_low
        k_out_high = box.hi[2] % Nz
        k_face_high = (box.hi[2] - 1) % Nz

        for ii, i in enumerate(range(box.lo[0], box.hi[0])):
            for jj, j in enumerate(range(box.lo[1], box.hi[1])):
                q_c[i, j, k_out_low] += (
                    dt * Fz_c[i, j, k_face_low] - regs["z_low"][ii, jj]
                ) / dx
                q_c[i, j, k_out_high] += (
                    regs["z_high"][ii, jj] - dt * Fz_c[i, j, k_face_high]
                ) / dx

        diag["box_lo"] = box.lo
        diag["box_hi"] = box.hi
        diag["fine_cells"] = sx * sy * sz * (r ** 3)
        return q_c, diag


# =============================================================================
# 4. SESI MULTIPHASE FLOW ADAPTER
# =============================================================================

class VOFAMRCoupler:
    """
    Adapter between the fractal AMR scalar solver and sesi_multiphase_flow.

    This uses:
      • alpha = VOF volume fraction as the conserved scalar
      • cell-centred velocity from the MAC fields
      • fractal energy-measure indicator as AMR error estimator

    It advances ALPHA through the AMR cycle. It does NOT replace the full
    Navier-Stokes step. For a full solver, patch-local NS, momentum reflux,
    and pressure projection must be added.
    """

    def __init__(
        self,
        backend_key: str = "carpet",
        refine_threshold: float = 0.20,
        coarsen_threshold: float = 0.02,
        ratio: int = 2,
        pad: int = 1,
    ):
        if backend_key not in BACKENDS:
            raise ValueError(f"unknown backend {backend_key!r}")
        self.backend = BACKENDS[backend_key]()
        self.amr = AMRScalarSolver(
            backend=self.backend,
            refine_threshold=refine_threshold,
            coarsen_threshold=coarsen_threshold,
            ratio=ratio,
            pad=pad,
        )

    @staticmethod
    def cell_centred_velocity_from_sesi(st, solver) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert MAC velocities from sesi_multiphase_flow.VOFState to
        cell-centred velocities.
        """
        g = solver.g
        uc = g.u_to_cc(st.u)
        vc = g.v_to_cc(st.v)
        wc = g.w_to_cc(st.w)
        return uc, vc, wc

    def step_alpha(
        self,
        alpha: torch.Tensor,
        uc: torch.Tensor,
        vc: torch.Tensor,
        wc: torch.Tensor,
        dx: float,
        dt: float,
        force_box: Optional[AMRBox] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.amr.step(alpha, uc, vc, wc, dx, dt, force_box=force_box)

    def step_alpha_from_sesi_state(self, st, solver, dt: float,
                                   force_box: Optional[AMRBox] = None):
        """
        Convenience wrapper for sesi_multiphase_flow.

        Example:
            coupler = VOFAMRCoupler("carpet")
            st.alpha, diag = coupler.step_alpha_from_sesi_state(st, solver, dt)
        """
        uc, vc, wc = self.cell_centred_velocity_from_sesi(st, solver)
        dx = solver.cfg.dx  # assumes isotropic dx=dy=dz
        return self.step_alpha(st.alpha, uc, vc, wc, dx, dt, force_box=force_box)


# =============================================================================
# 5. FRACTAL LICENSING / I1-I4 CHECKLIST
# =============================================================================

def fractal_licensing_report(backend_key: str) -> Dict[str, Any]:
    """
    Honest licensing report for using a fractal backend as AMR mathematics.

    Carpet-19 standing inputs:
      I1 uniform oscillation
      I2 subsequential Mosco convergence
      I3 uniqueness/proportionality of cluster forms
      I4 point separation

    For carpet, these are supported by cited literature in Carpet-19.
    For Menger/Vicsek/random, they are NOT automatically licensed.
    """
    if backend_key not in BACKENDS:
        raise ValueError(f"unknown backend {backend_key!r}")
    b = BACKENDS[backend_key]()

    report = {
        "backend": b.name,
        "pcf": b.pcf,
        "dimension": b.dimension,
        "rho": b.resistance_rho,
        "status": b.status,
        "paper_basis": b.required_paper_basis,
        "I1": "unknown",
        "I2": "unknown",
        "I3": "unknown",
        "I4": "unknown",
        "licensed_for_amr_math": False,
    }

    if backend_key == "sg":
        report.update({
            "I1": "pcf resistance-form theory available",
            "I2": "Kigami construction available",
            "I3": "standard pcf uniqueness context",
            "I4": "point separation available",
            "licensed_for_amr_math": True,
            "note": "SG is p.c.f.; easier than carpet, but not a non-p.c.f. test.",
        })
    elif backend_key == "carpet":
        report.update({
            "I1": "Carpet-19 derived under (RU)+(G)",
            "I2": "Carpet-19 cites KZ/Mosco subsequential convergence",
            "I3": "Carpet-19 derives from BBKT uniqueness",
            "I4": "Carpet-19 cites KZ point separation",
            "licensed_for_amr_math": True,
            "note": "Carpet-19 Theorem 10 is conditional on I1-I4; for standard carpet those inputs are literature-supported.",
        })
    elif backend_key == "menger":
        report.update({
            "I1": "NOT verified",
            "I2": "NOT verified",
            "I3": "NOT verified",
            "I4": "NOT verified",
            "licensed_for_amr_math": False,
            "note": "Menger sponge requires new resistance/Mosco/uniqueness/separation analysis before Carpet-19 results can be claimed.",
        })
    elif backend_key == "vicsek":
        report.update({
            "I1": "variant-dependent",
            "I2": "variant-dependent",
            "I3": "variant-dependent",
            "I4": "variant-dependent",
            "licensed_for_amr_math": False,
            "note": "Standard Vicsek may be finitely ramified; generalized variants need separate verification.",
        })
    elif backend_key == "random":
        report.update({
            "I1": "requires Paper 4 Hyp(SA) or Paper 5 generalizations",
            "I2": "requires stochastic homogenization diagnostics",
            "I3": "not available without deterministic structure",
            "I4": "not available without deterministic structure",
            "licensed_for_amr_math": False,
            "note": "Use Paper 7 hierarchy: SA -> ergodic decomposition -> AMS -> log-scale -> residual.",
        })
    return report


# =============================================================================
# 6. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 88)
    print(f"  FRACTAL AMR VOF COUPLER v{VERSION} — self-test")
    print("=" * 88)
    torch.manual_seed(0)
    ok = True

    def rep(name: str, passed: bool, extra: str = ""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1 backend reports
    for key in ("sg", "carpet", "menger", "vicsek", "random"):
        r = fractal_licensing_report(key)
        print(f"      {key:<8} licensed={r['licensed_for_amr_math']}  {r['status']}")
    rep("backend reports generated", True)

    # T2 indicator shape and range
    Nx, Ny, Nz = 18, 18, 10
    x = torch.linspace(0, 1, Nx).view(-1, 1, 1)
    y = torch.linspace(0, 1, Ny).view(1, -1, 1)
    z = torch.linspace(0, 1, Nz).view(1, 1, -1)
    alpha = ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2 < 0.16).double()

    backend = carpet_backend()
    ind = fractal_energy_indicator(alpha, backend)
    rep("indicator shape", ind.shape == alpha.shape)
    rep("indicator finite", bool(torch.isfinite(ind).all()))
    rep("indicator normalized", float(ind.min()) >= 0 and float(ind.max()) <= 1.0 + 1e-12)

    # T3 AMR step with reflux/subcycling
    u = torch.full((Nx, Ny, Nz), 0.15, dtype=torch.float64)
    v = torch.zeros_like(u)
    w = torch.zeros_like(u)
    dx = 1.0 / Nx
    dt = 0.25 * dx / 0.15

    amr = AMRScalarSolver(backend=backend, refine_threshold=0.05, ratio=2, pad=1)
    m0 = float(alpha.sum())
    alpha_new, diag = amr.step(alpha, u, v, w, dx, dt)
    m1 = float(alpha_new.sum())
    drift = abs(m1 - m0) / max(abs(m0), 1e-30)

    rep("AMR step finite", bool(torch.isfinite(alpha_new).all()))
    rep("AMR created patch", diag.get("has_patch", False),
        f"indicator_max={diag.get('indicator_max', 0):.3f}")
    print(f"      mass drift = {drift:.3e}")
    rep("mass drift bounded", drift < 1e-2,
        "if not, reflux sign/time interpolation needs tighter treatment")

    # T4 VOFAMRCoupler standalone array API
    coupler = VOFAMRCoupler("carpet", refine_threshold=0.05)
    alpha_new2, diag2 = coupler.step_alpha(alpha, u, v, w, dx, dt)
    rep("coupler step finite", bool(torch.isfinite(alpha_new2).all()))

    # T5 SG backend works too
    sg = sg_backend()
    ind_sg = fractal_energy_indicator(alpha, sg)
    rep("SG indicator finite", bool(torch.isfinite(ind_sg).all()))

    print("\n" + "=" * 88)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 88)
    print("""
  WHAT IS REAL HERE
    • energy-measure indicator from finite-level Gamma/mu diagnostics
    • patch creation, prolongation, restriction
    • subcycling with dt_f = dt_c / r
    • coarse-fine flux registration and reflux for the scalar alpha field

  WHAT IS NOT REAL YET
    • full momentum/pressure AMR inside sesi_multiphase_flow
    • Balsara divergence-preserving velocity prolongation in this wrapper
    • non-periodic coarse-fine boundaries
    • Menger/Vicsek/random fractal licensing

  TO CONNECT TO sesi_multiphase_flow.py AFTER REPAIR:
      solver = MultiphaseNavierStokes3D(cfg)
      st = solver.initial_state(alpha_fn=...)
      coupler = VOFAMRCoupler("carpet")
      # replace or augment alpha advection:
      st.alpha, diag = coupler.step_alpha_from_sesi_state(st, solver, dt)
""")
    return ok


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "backends", "license"])
    p.add_argument("--backend", default="carpet")
    a = p.parse_args()

    if a.cmd == "selftest":
        raise SystemExit(0 if self_test() else 1)
    if a.cmd == "backends":
        for r in backend_report():
            print(r)
    if a.cmd == "license":
        print(fractal_licensing_report(a.backend))


if __name__ == "__main__":
    main()
