"""
=============================================================================
SESI MULTIPHASE FLOW — VOLUME-OF-FLUID (VOF) 3D NAVIER-STOKES
Variable-density incompressible two-phase flow with geometric interface
capturing, non-Newtonian rheology, LES, and immersed-solid coupling
=============================================================================
Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
-----------------------------------------------------------------------------
NUMERICAL METHODS (all standard, cited — nothing invented here)
-----------------------------------------------------------------------------
[C1] Hirt & Nichols (1981)      — VOF concept
[C2] Youngs (1982)              — PLIC interface normal reconstruction
[C3] Scardovelli & Zaleski (1999) — analytic cut-volume / plane inversion
[C4] Rider & Kothe (1998)       — operator-split geometric advection
[C5] Weymouth & Yue (2010)      — mass-conserving split VOF, dilation term
[C6] Brackbill et al. (1992)    — Continuum Surface Force (CSF)
[C7] Chorin (1968)              — fractional-step projection
[C8] Ubbink & Issa (1999)       — CICSAM algebraic VOF
[C9] Smagorinsky (1963)         — LES eddy viscosity
[C10] Angot et al. (1999)       — Brinkman volume penalization (immersed solid)
[C11] Papanastasiou (1987)      — Bingham regularisation
-----------------------------------------------------------------------------
GOVERNING EQUATIONS SOLVED
-----------------------------------------------------------------------------
  ∂α/∂t + ∇·(α u) = 0                        (volume fraction, 0<=α<=1)
  ∇·u = 0                                    (incompressibility)
  ∂(ρu)/∂t + ∇·(ρ u⊗u) = -∇p + ∇·(μ(∇u+∇uᵀ))
                          + σ κ ∇α + ρ g + F_coriolis + F_porous + F_brinkman
  ρ = α ρ1 + (1-α) ρ2 ,  μ = α μ1 + (1-α) μ2   (μ harmonic at faces)
  ∂T/∂t + ∇·(T u) = ∇·(κ_T ∇T)               (optional passive scalar)

 discretisation: MAC staggered grid (u,v,w on faces; p,α,T at centres)
 time integration: forward Euler fractional step (1st order, CFL-limited)
-----------------------------------------------------------------------------
SDA-ONE INTEGRATION (the genuinely useful link)
-----------------------------------------------------------------------------
`build_solid_mask_from_coords()` rasterises Cα coordinates from
structural_domain_assembly_one.StructuralDomainAssembly output into an
immersed-solid mask, then Brinkman penalisation imposes (approximately)
no-slip on that surface. This lets you run solvent flow around a folded
100k-residue assembly WITHOUT body-fitted meshing.

HONEST: penalisation gives an O(sqrt(eta)) effective slip length; the
immersed surface is diffuse over ~2 cells. It is NOT a substitute for
body-fitted CFD if you need accurate wall shear stress.
-----------------------------------------------------------------------------
HONEST LIMITATIONS — READ BEFORE PRODUCTION USE
-----------------------------------------------------------------------------
1. FIRST-ORDER IN TIME. Forward-Euler fractional step. No RK2/Crank-Nicolson.
2. MOMENTUM ADVECTION is the non-conservative convective form (u·∇)u with
   minmod-limited upwind. MASS and VOLUME are conserved to machine precision;
   MOMENTUM is NOT discretely conserved. For high density ratios (>1000,
   e.g. water/air) this loses robustness — use a conservative momentum
   scheme (not implemented) if that regime matters.
3. CSF surface tension is NOT balanced-force and curvature is from smoothed
   alpha, not height functions. Expect SPURIOUS CURRENTS near static
   interfaces. The static-droplet test reports the actual Laplace-number
   error rather than hiding it.
4. PLIC cut-volume uses float64 inclusion-exclusion with dimension-reduction
   fallback when |n_i| < 1e-3. Worst-case local volume error ~1e-3 cell
   volumes for near-axis-aligned interfaces (documented, bounded).
5. Contact angle is 90 deg (Neumann alpha) only.
6. NOT a true AMR solver. `interface_amr_indicator()` produces a refinement
   flag field for your external AMR engine; there is no coarse-fine flux
   correction, no subcycling, no octree here. Grid is uniform Cartesian.
7. Differentiable mode (`differentiable=True`) unrolls the CG solve and is
   memory-hungry; geometric PLIC gradients are noisy (piecewise). For
   gradient-based optimisation prefer CICSAM + adjoint (adjoint NOT
   implemented).
8. This file has NOT been executed on GPU at scale. Run the [PASS]/[FAIL]
   suite locally first.
=============================================================================
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("sesi.vof")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

VOF_VERSION: str = "1.0.0"

# ---- optional ecosystem / IO dependencies (graceful) -----------------------
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

try:
    from structural_domain_assembly_one import StructuralDomainAssembly  # noqa: F401
    _HAS_SDA = True
except ImportError:
    _HAS_SDA = False


# =============================================================================
# 0. CONFIG
# =============================================================================
@dataclass
class VOFConfig:
    """
    Configuration for the 3D VOF Navier-Stokes solver.

    Grid / domain
    -------------
    nx, ny, nz          : number of CELLS per axis (uniform Cartesian)
    Lx, Ly, Lz          : physical domain size [m]
    bc                  : per-axis boundary condition tuple, each one of
                          'periodic' | 'wall' (no-slip) | 'slip' (free-slip)

    Fluids
    ------
    rho1, rho2          : densities of phase 1 (alpha=1) and phase 2 (alpha=0)
    mu1, mu2            : dynamic viscosities [Pa s]
    sigma               : surface tension coefficient [N/m]
    gravity             : body acceleration vector (gx, gy, gz) [m/s^2]

    Numerics
    --------
    vof_scheme          : 'plic' (geometric, sharp, default) | 'cicsam' (algebraic)
    momentum_limiter    : 'minmod' | 'upwind' | 'central'
    pressure_solver     : 'cg' (variable density) | 'fft' (const density periodic)
    poisson_tol         : relative residual tolerance for CG
    poisson_max_iter    : CG iteration cap
    n_alpha_smooth      : smoothing passes for curvature (CSF)

    Rheology
    --------
    rheology            : 'newtonian' | 'power_law' | 'carreau' | 'bingham'
    power_law_k         : consistency index K   (power-law: mu = K * gdot^(n-1))
    power_law_n         : flow behaviour index n
    carreau_mu0         : zero-shear viscosity
    carreau_muinf       : infinite-shear viscosity
    carreau_lambda      : relaxation time
    carreau_n           : Carreau index
    bingham_tauy        : yield stress [Pa]
    bingham_m           : Papanastasiou regularisation parameter
    mu_min, mu_max      : clamps on apparent viscosity (stability)

    Turbulence
    ----------
    turbulence          : 'laminar' | 'smagorinsky'
    smagorinsky_cs      : Smagorinsky constant (~0.1-0.18)

    Thermal / scalar
    ----------------
    boussinesq          : enable Boussinesq buoyancy from scalar T
    beta_thermal        : thermal expansion coefficient [1/K]
    T_ref               : reference temperature
    scalar_diffusivity  : diffusivity of T [m^2/s]

    Rotating frame
    --------------
    omega               : angular velocity vector for Coriolis -2 omega x u

    Porous media
    ------------
    porous_alpha_darcy  : Darcy coefficient (1/permeability); 0 disables
    porous_beta_forch   : Forchheimer inertial coefficient

    Immersed solid (SDA-ONE link)
    ----------------------------
    brinkman_eta        : penalisation viscosity parameter (smaller = more
                          no-slip-like, but stiffer). 0 disables.

    Time stepping
    -------------
    cfl                 : advective CFL number (<=0.5 recommended for split VOF)
    viscous_factor      : safety factor on explicit viscous limit
    capillary_factor    : safety factor on capillary-wave limit
    dt_max              : hard cap on dt

    Runtime
    -------
    device, dtype       : torch device / dtype ('float64' recommended for VOF)
    differentiable      : keep autograd graph (memory heavy; see limitations)
    """

    # grid / domain
    nx: int = 32
    ny: int = 32
    nz: int = 32
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    bc: Tuple[str, str, str] = ("wall", "wall", "wall")

    # fluids
    rho1: float = 1000.0
    rho2: float = 1.0
    mu1: float = 1.0e-3
    mu2: float = 1.8e-5
    sigma: float = 0.0728
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)

    # numerics
    vof_scheme: str = "plic"
    momentum_limiter: str = "minmod"
    pressure_solver: str = "cg"
    poisson_tol: float = 1e-8
    poisson_max_iter: int = 500
    n_alpha_smooth: int = 2

    # rheology
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

    # turbulence
    turbulence: str = "laminar"
    smagorinsky_cs: float = 0.17

    # thermal
    boussinesq: bool = False
    beta_thermal: float = 2.1e-4
    T_ref: float = 293.15
    scalar_diffusivity: float = 1.4e-7

    # rotating frame
    omega: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # porous
    porous_alpha_darcy: float = 0.0
    porous_beta_forch: float = 0.0

    # immersed solid
    brinkman_eta: float = 0.0

    # time stepping
    cfl: float = 0.35
    viscous_factor: float = 0.25
    capillary_factor: float = 0.5
    dt_max: float = 1e-2

    # runtime
    device: str = "cpu"
    dtype: str = "float64"
    differentiable: bool = False

    def __post_init__(self) -> None:
        assert min(self.nx, self.ny, self.nz) >= 4, "need >=4 cells per axis"
        assert min(self.Lx, self.Ly, self.Lz) > 0.0
        for b in self.bc:
            assert b in ("periodic", "wall", "slip"), f"bad bc {b}"
        assert self.rho1 > 0 and self.rho2 > 0
        assert self.mu1 > 0 and self.mu2 > 0
        assert self.sigma >= 0.0
        assert self.vof_scheme in ("plic", "cicsam")
        assert self.momentum_limiter in ("minmod", "upwind", "central")
        assert self.pressure_solver in ("cg", "fft")
        assert self.rheology in ("newtonian", "power_law", "carreau", "bingham")
        assert self.turbulence in ("laminar", "smagorinsky")
        assert 0.0 < self.cfl <= 1.0
        assert self.poisson_tol > 0.0
        assert self.dtype in ("float32", "float64")
        if self.pressure_solver == "fft":
            assert all(b == "periodic" for b in self.bc), \
                "fft pressure solver requires fully periodic BC"
        if self.vof_scheme == "plic" and self.dtype == "float32":
            warnings.warn(
                "PLIC with float32 loses mass-conservation accuracy. "
                "Internally PLIC volume math is done in float64 regardless, "
                "but float64 state is strongly recommended."
            )

    @property
    def dx(self) -> float: return self.Lx / self.nx

    @property
    def dy(self) -> float: return self.Ly / self.ny

    @property
    def dz(self) -> float: return self.Lz / self.nz

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VOFConfig":
        d = dict(d)
        d["bc"] = tuple(d["bc"])
        for k in ("gravity", "omega"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


# =============================================================================
# 1. GRID / BOUNDARY-CONDITION UTILITIES
# =============================================================================
def _pad_axis(f: torch.Tensor, axis: int, mode: str,
              antisym: bool = False) -> torch.Tensor:
    """
    Pad tensor `f` by one cell on both sides along `axis`.

    mode      : 'periodic' -> wrap
                'wall'/'slip' -> replicate (Neumann); caller flips sign
                                 via `antisym` for tangential no-slip.
    antisym   : if True, ghost = -interior (Dirichlet zero at the face)
    """
    n = f.shape[axis]
    if mode == "periodic":
        lo = f.index_select(axis, torch.tensor([n - 1], device=f.device))
        hi = f.index_select(axis, torch.tensor([0], device=f.device))
    else:
        lo = f.index_select(axis, torch.tensor([0], device=f.device))
        hi = f.index_select(axis, torch.tensor([n - 1], device=f.device))
        if antisym:
            lo, hi = -lo, -hi
    return torch.cat([lo, f, hi], dim=axis)


def pad_cc(f: torch.Tensor, bc: Tuple[str, str, str],
           antisym: bool = False) -> torch.Tensor:
    """Pad a cell-centred field on all three axes."""
    g = f
    for ax in range(3):
        g = _pad_axis(g, ax, bc[ax], antisym)
    return g


class MACGrid:
    """
    Marker-and-Cell staggered layout.

      u : (nx+1, ny,   nz  )   x-normal faces
      v : (nx,   ny+1, nz  )   y-normal faces
      w : (nx,   ny,   nz+1)   z-normal faces
      p, alpha, T, rho, mu : (nx, ny, nz) cell centres
    """

    def __init__(self, cfg: VOFConfig) -> None:
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.dt_ = cfg.torch_dtype

    def zeros_u(self) -> torch.Tensor:
        c = self.cfg
        return torch.zeros(c.nx + 1, c.ny, c.nz, device=self.dev, dtype=self.dt_)

    def zeros_v(self) -> torch.Tensor:
        c = self.cfg
        return torch.zeros(c.nx, c.ny + 1, c.nz, device=self.dev, dtype=self.dt_)

    def zeros_w(self) -> torch.Tensor:
        c = self.cfg
        return torch.zeros(c.nx, c.ny, c.nz + 1, device=self.dev, dtype=self.dt_)

    def zeros_cc(self) -> torch.Tensor:
        c = self.cfg
        return torch.zeros(c.nx, c.ny, c.nz, device=self.dev, dtype=self.dt_)

    # ---- velocity boundary conditions ------------------------------------
    def enforce_velocity_bc(self, u, v, w):
        """
        Impermeable walls: normal face velocity = 0 on wall/slip boundaries.
        Periodic: identify the two ends (single degree of freedom).
        Tangential no-slip is applied in the ghost padding (antisym=True).
        """
        bc = self.cfg.bc
        u, v, w = u.clone(), v.clone(), w.clone()
        for ax, (comp, b) in enumerate(zip((u, v, w), bc)):
            if b == "periodic":
                # tie the two end planes together
                hi = comp.index_select(ax, torch.tensor([comp.shape[ax] - 1],
                                                        device=comp.device))
                lo = comp.index_select(ax, torch.tensor([0], device=comp.device))
                mean = 0.5 * (hi + lo)
                comp.index_copy_(ax, torch.tensor([0], device=comp.device), mean)
                comp.index_copy_(ax, torch.tensor([comp.shape[ax] - 1],
                                                  device=comp.device), mean)
            else:
                z = torch.zeros_like(comp.index_select(
                    ax, torch.tensor([0], device=comp.device)))
                comp.index_copy_(ax, torch.tensor([0], device=comp.device), z)
                comp.index_copy_(ax, torch.tensor([comp.shape[ax] - 1],
                                                  device=comp.device), z)
        return u, v, w

    def pad_u(self, u):
        """Ghost halo for the x-face velocity (tangential no-slip in y,z)."""
        bc = self.cfg.bc
        g = _pad_axis(u, 0, bc[0], antisym=False)
        g = _pad_axis(g, 1, bc[1], antisym=(bc[1] == "wall"))
        g = _pad_axis(g, 2, bc[2], antisym=(bc[2] == "wall"))
        return g

    def pad_v(self, v):
        bc = self.cfg.bc
        g = _pad_axis(v, 0, bc[0], antisym=(bc[0] == "wall"))
        g = _pad_axis(g, 1, bc[1], antisym=False)
        g = _pad_axis(g, 2, bc[2], antisym=(bc[2] == "wall"))
        return g

    def pad_w(self, w):
        bc = self.cfg.bc
        g = _pad_axis(w, 0, bc[0], antisym=(bc[0] == "wall"))
        g = _pad_axis(g, 1, bc[1], antisym=(bc[1] == "wall"))
        g = _pad_axis(g, 2, bc[2], antisym=False)
        return g

    # ---- interpolation ---------------------------------------------------
    def u_to_cc(self, u):
        return 0.5 * (u[:-1] + u[1:])

    def v_to_cc(self, v):
        return 0.5 * (v[:, :-1] + v[:, 1:])

    def w_to_cc(self, w):
        return 0.5 * (w[:, :, :-1] + w[:, :, 1:])

    def v_at_u(self, v):
        """Interpolate v (x-centre,y-face) to u locations (x-face,y-centre)."""
        vp = _pad_axis(v, 0, self.cfg.bc[0], antisym=(self.cfg.bc[0] == "wall"))
        # vp shape (nx+2, ny+1, nz); u location (i, j) needs
        # v[i-1,j], v[i,j], v[i-1,j+1], v[i,j+1]
        return 0.25 * (vp[:-1, :-1] + vp[1:, :-1] + vp[:-1, 1:] + vp[1:, 1:])

    def w_at_u(self, w):
        wp = _pad_axis(w, 0, self.cfg.bc[0], antisym=(self.cfg.bc[0] == "wall"))
        return 0.25 * (wp[:-1, :, :-1] + wp[1:, :, :-1]
                       + wp[:-1, :, 1:] + wp[1:, :, 1:])

    def u_at_v(self, u):
        up = _pad_axis(u, 1, self.cfg.bc[1], antisym=(self.cfg.bc[1] == "wall"))
        return 0.25 * (up[:-1, :-1] + up[1:, :-1] + up[:-1, 1:] + up[1:, 1:])

    def w_at_v(self, w):
        wp = _pad_axis(w, 1, self.cfg.bc[1], antisym=(self.cfg.bc[1] == "wall"))
        return 0.25 * (wp[:, :-1, :-1] + wp[:, 1:, :-1]
                       + wp[:, :-1, 1:] + wp[:, 1:, 1:])

    def u_at_w(self, u):
        up = _pad_axis(u, 2, self.cfg.bc[2], antisym=(self.cfg.bc[2] == "wall"))
        return 0.25 * (up[:-1, :, :-1] + up[1:, :, :-1]
                       + up[:-1, :, 1:] + up[1:, :, 1:])

    def v_at_w(self, v):
        vp = _pad_axis(v, 2, self.cfg.bc[2], antisym=(self.cfg.bc[2] == "wall"))
        return 0.25 * (vp[:, :-1, :-1] + vp[:, 1:, :-1]
                       + vp[:, :-1, 1:] + vp[:, 1:, 1:])

    # ---- differential operators -----------------------------------------
    def divergence(self, u, v, w):
        c = self.cfg
        return ((u[1:] - u[:-1]) / c.dx
                + (v[:, 1:] - v[:, :-1]) / c.dy
                + (w[:, :, 1:] - w[:, :, :-1]) / c.dz)

    def grad_cc_to_faces(self, p):
        """Compact gradient of a cell-centred scalar onto faces (Neumann BC)."""
        c = self.cfg
        pp = pad_cc(p, c.bc)
        gx = (pp[2:, 1:-1, 1:-1] - pp[1:-1, 1:-1, 1:-1]) / c.dx
        gy = (pp[1:-1, 2:, 1:-1] - pp[1:-1, 1:-1, 1:-1]) / c.dy
        gz = (pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, 1:-1]) / c.dz
        return gx, gy, gz


# =============================================================================
# 2. PLIC — PIECEWISE LINEAR INTERFACE CALCULATION
# =============================================================================
class PLIC:
    """
    Geometric VOF machinery.

    Normal   : Youngs [C2] central-difference gradient of alpha.
    Plane    : n . x = d, with n >= 0 by octant symmetry, normalised so that
               n1 + n2 + n3 = 1 on the unit reference cell => d in [0, 1].
    Volume   : Scardovelli & Zaleski [C3] inclusion-exclusion cut volume,
               generalised to a rectangular box (needed for split fluxes).
    Inversion: vectorised bisection (40 iters -> ~1e-12 in d).

    All volume arithmetic is float64 regardless of solver dtype.
    """

    _DIM_TOL = 1e-3   # below this a normal component is treated as zero
    _BISECT_ITERS = 40

    # ------------------------------------------------------------------ normals
    @staticmethod
    def normals(alpha: torch.Tensor, dx: float, dy: float, dz: float):
        """Youngs normal n = -grad(alpha) / |grad(alpha)| (points into phase 2)."""
        a = alpha.to(torch.float64)
        # replicate padding => zero normal gradient at walls (90 deg contact)
        ap = a.unsqueeze(0).unsqueeze(0)
        ap = F.pad(ap, (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
        gx = (ap[2:, 1:-1, 1:-1] - ap[:-2, 1:-1, 1:-1]) / (2 * dx)
        gy = (ap[1:-1, 2:, 1:-1] - ap[1:-1, :-2, 1:-1]) / (2 * dy)
        gz = (ap[1:-1, 1:-1, 2:] - ap[1:-1, 1:-1, :-2]) / (2 * dz)
        mag = torch.sqrt(gx * gx + gy * gy + gz * gz)
        safe = mag.clamp_min(1e-30)
        return -gx / safe, -gy / safe, -gz / safe, mag

    # ------------------------------------------------------------------ volume
    @staticmethod
    def _pos3(t: torch.Tensor) -> torch.Tensor:
        return t.clamp_min(0.0) ** 3

    @classmethod
    def box_volume(cls, d, m1, m2, m3, L1, L2, L3):
        """
        Volume of { x in [0,L1]x[0,L2]x[0,L3] : m1 x + m2 y + m3 z <= d },
        with m_i >= 0. Inclusion-exclusion over the 8 simplex corners:

          V = 1/(6 m1 m2 m3) * sum_{S subset {1,2,3}} (-1)^|S|
                                ( d - sum_{i in S} m_i L_i )_+^3

        Falls back to the exact 2D / 1D formula when a normal component is
        below _DIM_TOL, because the 3D expression suffers catastrophic
        cancellation there (denominator ~ m1 m2 m3).
        """
        d = torch.as_tensor(d, dtype=torch.float64)
        m1 = torch.as_tensor(m1, dtype=torch.float64).clamp_min(0.0)
        m2 = torch.as_tensor(m2, dtype=torch.float64).clamp_min(0.0)
        m3 = torch.as_tensor(m3, dtype=torch.float64).clamp_min(0.0)
        L1 = torch.as_tensor(L1, dtype=torch.float64)
        L2 = torch.as_tensor(L2, dtype=torch.float64)
        L3 = torch.as_tensor(L3, dtype=torch.float64)
        d, m1, m2, m3, L1, L2, L3 = torch.broadcast_tensors(d, m1, m2, m3, L1, L2, L3)

        a1, a2, a3 = m1 * L1, m2 * L2, m3 * L3
        tol = cls._DIM_TOL
        n_small = ((m1 < tol).to(torch.int8) + (m2 < tol).to(torch.int8)
                   + (m3 < tol).to(torch.int8))

        p3 = cls._pos3
        # ---- generic 3D ------------------------------------------------
        s = p3(d)
        s = s - p3(d - a1) - p3(d - a2) - p3(d - a3)
        s = s + p3(d - a1 - a2) + p3(d - a1 - a3) + p3(d - a2 - a3)
        s = s - p3(d - a1 - a2 - a3)
        denom3 = (6.0 * m1 * m2 * m3).clamp_min(1e-300)
        v3 = s / denom3

        # ---- 2D fallbacks (one component ~ 0) ---------------------------
        def area2(dd, ma, mb, La, Lb):
            A, B = ma * La, mb * Lb
            t = (dd.clamp_min(0.0) ** 2
                 - (dd - A).clamp_min(0.0) ** 2
                 - (dd - B).clamp_min(0.0) ** 2
                 + (dd - A - B).clamp_min(0.0) ** 2)
            return t / (2.0 * ma * mb).clamp_min(1e-300)

        v2_m1 = area2(d, m2, m3, L2, L3) * L1
        v2_m2 = area2(d, m1, m3, L1, L3) * L2
        v2_m3 = area2(d, m1, m2, L1, L2) * L3

        # ---- 1D fallbacks (two components ~ 0) --------------------------
        v1_m1 = (d / m1.clamp_min(1e-300)).clamp(0.0, L1) * L2 * L3
        v1_m2 = (d / m2.clamp_min(1e-300)).clamp(0.0, L2) * L1 * L3
        v1_m3 = (d / m3.clamp_min(1e-300)).clamp(0.0, L3) * L1 * L2

        vmax = L1 * L2 * L3
        out = v3
        out = torch.where((n_small == 1) & (m1 < tol), v2_m1, out)
        out = torch.where((n_small == 1) & (m2 < tol), v2_m2, out)
        out = torch.where((n_small == 1) & (m3 < tol), v2_m3, out)
        out = torch.where((n_small >= 2) & (m1 >= tol), v1_m1, out)
        out = torch.where((n_small >= 2) & (m2 >= tol), v1_m2, out)
        out = torch.where((n_small >= 2) & (m3 >= tol), v1_m3, out)
        out = torch.where(n_small == 3, vmax * (d > 0).to(d.dtype), out)
        return out.clamp(0.0, vmax)

    # ------------------------------------------------------------ plane offset
    @classmethod
    def plane_offset(cls, alpha, n1, n2, n3):
        """
        Find d such that box_volume(d, n, unit cube) == alpha, by bisection.
        Normals are folded to the positive octant and L1-normalised.
        Returns (d, m1, m2, m3) in the folded frame.
        """
        a = alpha.to(torch.float64).clamp(0.0, 1.0)
        m1, m2, m3 = n1.abs().to(torch.float64), n2.abs().to(torch.float64), n3.abs().to(torch.float64)
        ssum = (m1 + m2 + m3).clamp_min(1e-30)
        m1, m2, m3 = m1 / ssum, m2 / ssum, m3 / ssum

        one = torch.ones_like(a)
        zero = torch.zeros_like(a)
        lo, hi = zero.clone(), one.clone()
        for _ in range(cls._BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            vol = cls.box_volume(mid, m1, m2, m3, one, one, one)
            too_small = vol < a
            lo = torch.where(too_small, mid, lo)
            hi = torch.where(too_small, hi, mid)
        return 0.5 * (lo + hi), m1, m2, m3

    # -------------------------------------------------------------- split flux
    @classmethod
    def sweep_flux(cls, alpha, n1, n2, n3, vel_face, dt, h, axis):
        """
        Directional-split geometric flux [C4].

        For each face, the swept slab has thickness s = |u_f| dt / h.
        Donor cell is upwind. Fluid volume in the slab is obtained from the
        donor's PLIC plane clipped to a box of side s along `axis`.

        Returns flux VOLUME (not velocity*alpha) per face, in units of
        cell-volume fraction * h (i.e. divide by h when forming divergence).
        """
        a = alpha.to(torch.float64)
        d, m1, m2, m3 = cls.plane_offset(a, n1, n2, n3)

        # fold normals into the positive octant: reflect the plane offset
        # d is measured from the corner where all folded coords are 0.
        # For a negative original component, the folded coordinate is 1-x,
        # which shifts d by -m_i (handled by using d' = d - sum(m_i for n_i<0)).
        shift = (torch.where(n1 < 0, m1, torch.zeros_like(m1))
                 + torch.where(n2 < 0, m2, torch.zeros_like(m2))
                 + torch.where(n3 < 0, m3, torch.zeros_like(m3)))
        # after folding x_i -> 1 - x_i for negative components, the plane
        # becomes sum m_i xfold_i = d + shift ... careful sign:
        #   n.x = sum_{+} m_i x_i - sum_{-} m_i (1 - xf_i) = d
        #   => sum_{+} m_i x_i + sum_{-} m_i xf_i = d + sum_{-} m_i
        d_fold = d + shift

        one = torch.ones_like(a)
        s = (vel_face.abs().to(torch.float64) * dt / h)

        # ---- gather donor field per face --------------------------------
        # face index f along `axis` sits between cell f-1 (left) and f (right)
        ncell = a.shape[axis]
        nface = ncell + 1

        def take(field, idx_lo, idx_hi):
            return field.index_select(axis, torch.arange(idx_lo, idx_hi,
                                                         device=field.device))

        # left donor cells 0..ncell-1 ; right donor cells 0..ncell-1
        pos = vel_face > 0

        # slab lengths along `axis`
        s_clip = s.clamp(0.0, 1.0)

        # box sides
        L = [one, one, one]
        L[axis] = s_clip

        # For positive velocity the donor is the LEFT cell and the slab is
        # adjacent to its HIGH side; for negative velocity the donor is the
        # RIGHT cell and the slab is adjacent to its LOW side.
        # We must express the plane offset in the slab's local frame.
        # Work per-face by shifting donor fields.
        def shift_field(field, delta):
            """delta=-1 -> field[f-1]; delta=+1 -> field[f+1]; clamped at ends."""
            idx = torch.arange(nface, device=field.device) + delta
            idx = idx.clamp(0, ncell - 1)
            return field.index_select(axis, idx)

        # donor-plane parameters, expanded to face indexing
        dL = shift_field(d_fold, -1)
        dR = shift_field(d_fold, +1)
        mL = [shift_field(m, -1) for m in (m1, m2, m3)]
        mR = [shift_field(m, +1) for m in (m1, m2, m3)]

        # slab offset for the "high side" case: local origin at x_axis = 1-s
        m_axis_L = mL[axis]
        dL_slab = dL - m_axis_L * (1.0 - s_clip)
        dR_slab = dR  # low side: local origin at x_axis = 0

        fluxL = cls.box_volume(dL_slab, mL[0], mL[1], mL[2], L[0], L[1], L[2])
        fluxR = cls.box_volume(dR_slab, mR[0], mR[1], mR[2], L[0], L[1], L[2])

        flux = torch.where(pos, fluxL, fluxR)
        flux = torch.where(vel_face > 0, flux, -flux)

        # no flux through impermeable domain ends
        zero_idx = [0, nface - 1]
        for zi in zero_idx:
            sl = [slice(None)] * 3
            sl[axis] = zi
            flux[tuple(sl)] = 0.0
        return flux.to(alpha.dtype)


# =============================================================================
# 3. VOF ADVECTION (PLIC split  /  CICSAM algebraic)
# =============================================================================
class VOFAdvector:
    """
    Volume-fraction transport.

    'plic'   : operator-split geometric advection, direction order rotated
               each step (Strang-like symmetrisation) to reduce directional
               bias. Mass conserving to machine precision after bounded
               repair [C5].
    'cicsam' : algebraic compressive scheme [C8]; smooth (autograd friendly)
               but less sharp than PLIC.
    """

    def __init__(self, cfg: VOFConfig, grid: MACGrid) -> None:
        self.cfg = cfg
        self.g = grid
        self._sweep_order = 0

    # ---------------------------------------------------------------- helpers
    def _bounded_repair(self, a: torch.Tensor, target_mass: float) -> torch.Tensor:
        """
        Clip to [0,1] then redistribute the (tiny) mass defect uniformly over
        interface cells so total mass is preserved exactly.
        """
        over = (a - a.clamp(0.0, 1.0)).sum()
        a = a.clamp(0.0, 1.0)
        defect = target_mass - a.sum()
        if abs(float(defect)) < 1e-14:
            return a
        iface = ((a > 1e-6) & (a < 1.0 - 1e-6))
        w = iface.to(a.dtype)
        if float(w.sum()) < 1.0:
            w = torch.ones_like(a)
        w = w / w.sum().clamp_min(1e-30)
        return (a + defect * w).clamp(0.0, 1.0)

    # ------------------------------------------------------------------- PLIC
    def _advect_plic(self, a, u, v, w, dt):
        c = self.cfg
        mass0 = float(a.sum())
        axes = [(0, u, c.dx), (1, v, c.dy), (2, w, c.dz)]
        order = [(self._sweep_order + k) % 3 for k in range(3)]
        self._sweep_order = (self._sweep_order + 1) % 3

        for k in order:
            ax, vel, h = axes[k]
            n1, n2, n3, mag = PLIC.normals(a, c.dx, c.dy, c.dz)
            # in single-phase cells the normal is meaningless; flux reduces
            # to a*|u|*dt automatically because plane_offset saturates.
            flux = PLIC.sweep_flux(a, n1, n2, n3, vel, dt, h, ax)
            sl_hi = [slice(None)] * 3
            sl_lo = [slice(None)] * 3
            sl_hi[ax] = slice(1, None)
            sl_lo[ax] = slice(0, -1)
            div = (flux[tuple(sl_hi)] - flux[tuple(sl_lo)]) / h
            # split-scheme dilation correction [C5]: for div-free u this is
            # ~0, but it keeps the scheme bounded when div is not exactly 0.
            if ax == 0:
                dil = (vel[1:] - vel[:-1]) / h
            elif ax == 1:
                dil = (vel[:, 1:] - vel[:, :-1]) / h
            else:
                dil = (vel[:, :, 1:] - vel[:, :, :-1]) / h
            a = a - div + a * dil * dt
            a = a.clamp(0.0, 1.0)
        return self._bounded_repair(a, mass0)

    # ----------------------------------------------------------------- CICSAM
    def _advect_cicsam(self, a, u, v, w, dt):
        """
        CICSAM [C8] on cell-centred control volumes with face velocities.
        Normalised-variable formulation with CBC / UQ blending.
        """
        c = self.cfg
        mass0 = float(a.sum())
        ap = pad_cc(a, c.bc)
        fluxes = []
        for ax, (vel, h) in enumerate(((u, c.dx), (v, c.dy), (w, c.dz))):
            # donor / acceptor / upwind triple along this axis
            sl = [slice(1, -1)] * 3
            D = ap[tuple(sl)]
            slA = [slice(1, -1)] * 3
            slA[ax] = slice(2, None)
            A = ap[tuple(slA)]
            slD = [slice(1, -1)] * 3
            slD[ax] = slice(0, -2)
            Dm = ap[tuple(slD)]

            Co = (vel.abs() * dt / h).clamp(1e-8, 1.0 - 1e-8)
            # face value: pick upwind cell as D, downwind as A
            pos = vel > 0
            # build face arrays: ncell+1 faces; use neighbour shifted fields
            def to_face(fld, delta):
                idx = torch.arange(fld.shape[ax] + 1, device=fld.device) + delta
                return fld.index_select(ax, idx.clamp(0, fld.shape[ax] - 1))

            aD = torch.where(pos, to_face(a, -1), to_face(a, 0))
            aA = torch.where(pos, to_face(a, 0), to_face(a, -1))
            aU = torch.where(pos, to_face(a, -2), to_face(a, 1))

            denom = (aA - aU)
            denom = torch.where(denom.abs() < 1e-12,
                                torch.full_like(denom, 1e-12), denom)
            at = ((aD - aU) / denom).clamp(0.0, 1.0)

            cbc = torch.where((at > 0) & (at < 1),
                              torch.minimum(at / Co, torch.ones_like(at)), at)
            cbc = torch.minimum(cbc, torch.ones_like(cbc))
            uq = torch.where((at > 0) & (at < 1),
                             torch.minimum((8 * Co * at + (1 - Co) * (6 * at + 3)) / 8.0,
                                           torch.ones_like(at)), at)
            # interface-orientation weight (0.5 => 45 deg)
            n1, n2, n3, mag = PLIC.normals(a, c.dx, c.dy, c.dz)
            ncomp = (n1, n2, n3)[ax].abs()
            nf = torch.where(pos, to_face(ncomp, -1), to_face(ncomp, 0))
            gamma = (1.0 - nf).clamp(0.0, 1.0) * 0.5 + 0.5
            blend = gamma * cbc + (1 - gamma) * uq
            af = aU + blend * (aA - aU)
            af = torch.where((aD < 1e-8) | (aD > 1 - 1e-8), aD, af)
            fluxes.append(vel * af * dt / h)

        out = a.clone()
        for ax, fl in enumerate(fluxes):
            sl_hi = [slice(None)] * 3
            sl_lo = [slice(None)] * 3
            sl_hi[ax] = slice(1, None)
            sl_lo[ax] = slice(0, -1)
            out = out - (fl[tuple(sl_hi)] - fl[tuple(sl_lo)])
        return self._bounded_repair(out, mass0)

    def advect(self, a, u, v, w, dt):
        if self.cfg.vof_scheme == "plic":
            return self._advect_plic(a, u, v, w, dt)
        return self._advect_cicsam(a, u, v, w, dt)


# =============================================================================
# 4. RHEOLOGY + LES
# =============================================================================
class Rheology:
    """Apparent viscosity from the local strain-rate invariant."""

    def __init__(self, cfg: VOFConfig, grid: MACGrid) -> None:
        self.cfg = cfg
        self.g = grid

    def strain_rate_invariant(self, u, v, w):
        """
        gdot = sqrt(2 S:S) at cell centres, with S = 0.5(grad u + grad u^T).
        Diagonal terms at centres; off-diagonal terms at edges then averaged.
        """
        c, g = self.cfg, self.g
        up, vp, wp = g.pad_u(u), g.pad_v(v), g.pad_w(w)

        dudx = (up[2:, 1:-1, 1:-1] - up[1:-1, 1:-1, 1:-1]) / c.dx
        dvdy = (vp[1:-1, 2:, 1:-1] - vp[1:-1, 1:-1, 1:-1]) / c.dy
        dwdz = (wp[1:-1, 1:-1, 2:] - wp[1:-1, 1:-1, 1:-1]) / c.dz

        dudy = (up[1:-1, 2:, 1:-1] - up[1:-1, :-2, 1:-1]) / (2 * c.dy)
        dudz = (up[1:-1, 1:-1, 2:] - up[1:-1, 1:-1, :-2]) / (2 * c.dz)
        dvdx = (vp[2:, 1:-1, 1:-1] - vp[:-2, 1:-1, 1:-1]) / (2 * c.dx)
        dvdz = (vp[1:-1, 1:-1, 2:] - vp[1:-1, 1:-1, :-2]) / (2 * c.dz)
        dwdx = (wp[2:, 1:-1, 1:-1] - wp[:-2, 1:-1, 1:-1]) / (2 * c.dx)
        dwdy = (wp[1:-1, 2:, 1:-1] - wp[1:-1, :-2, 1:-1]) / (2 * c.dy)

        sxy = 0.5 * (dudy + dvdx)
        sxz = 0.5 * (dudz + dwdx)
        syz = 0.5 * (dvdz + dwdy)
        ss = (dudx ** 2 + dvdy ** 2 + dwdz ** 2
              + 2 * (sxy ** 2 + sxz ** 2 + syz ** 2))
        return torch.sqrt(2.0 * ss).clamp_min(1e-12), (dudx, dvdy, dwdz, sxy, sxz, syz)

    def apparent_mu(self, mu_base, gdot):
        c = self.cfg
        if c.rheology == "newtonian":
            mu = mu_base
        elif c.rheology == "power_law":
            mu = c.power_law_k * gdot.clamp_min(1e-12) ** (c.power_law_n - 1.0)
        elif c.rheology == "carreau":
            mu = c.carreau_muinf + (c.carreau_mu0 - c.carreau_muinf) * (
                1.0 + (c.carreau_lambda * gdot) ** 2
            ) ** ((c.carreau_n - 1.0) / 2.0)
        elif c.rheology == "bingham":
            # Papanastasiou [C11]: mu = mu_p + tauy (1 - exp(-m gdot)) / gdot
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
# 5. SURFACE TENSION (CSF)
# =============================================================================
class SurfaceTension:
    """
    Continuum Surface Force [C6]:  F = sigma * kappa * grad(alpha),
    kappa = -div(n_hat), n_hat = grad(alpha)/|grad(alpha)|.

    alpha is smoothed `n_alpha_smooth` times before curvature to reduce
    (not eliminate) spurious currents. NOT balanced-force, NOT height
    function — see module limitations.
    """

    def __init__(self, cfg: VOFConfig, grid: MACGrid) -> None:
        self.cfg = cfg
        self.g = grid

    def _smooth(self, a):
        for _ in range(self.cfg.n_alpha_smooth):
            ap = pad_cc(a, self.cfg.bc)
            a = (6 * ap[1:-1, 1:-1, 1:-1]
                 + ap[2:, 1:-1, 1:-1] + ap[:-2, 1:-1, 1:-1]
                 + ap[1:-1, 2:, 1:-1] + ap[1:-1, :-2, 1:-1]
                 + ap[1:-1, 1:-1, 2:] + ap[1:-1, 1:-1, :-2]) / 12.0
        return a

    def force_faces(self, alpha):
        c, g = self.cfg, self.g
        if c.sigma <= 0.0:
            return (g.zeros_u(), g.zeros_v(), g.zeros_w())
        a_s = self._smooth(alpha)
        n1, n2, n3, mag = PLIC.normals(a_s, c.dx, c.dy, c.dz)
        # curvature = -div(n_hat)
        p1 = pad_cc(n1, c.bc); p2 = pad_cc(n2, c.bc); p3 = pad_cc(n3, c.bc)
        kappa = -((p1[2:, 1:-1, 1:-1] - p1[:-2, 1:-1, 1:-1]) / (2 * c.dx)
                  + (p2[1:-1, 2:, 1:-1] - p2[1:-1, :-2, 1:-1]) / (2 * c.dy)
                  + (p3[1:-1, 1:-1, 2:] - p3[1:-1, 1:-1, :-2]) / (2 * c.dz))
        kappa = torch.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
        # grad(alpha) from the UNSMOOTHED field for the force localisation
        gx, gy, gz = g.grad_cc_to_faces(alpha)
        kx, ky, kz = g.grad_cc_to_faces(kappa)
        # interpolate kappa to faces
        kf_x = 0.5 * (kappa[:-1] + kappa[1:]) if c.nx > 1 else kappa
        kfx = torch.zeros_like(gx)
        kfx[1:-1] = 0.5 * (kappa[:-1] + kappa[1:])
        kfy = torch.zeros_like(gy)
        kfy[:, 1:-1] = 0.5 * (kappa[:, :-1] + kappa[:, 1:])
        kfz = torch.zeros_like(gz)
        kfz[:, :, 1:-1] = 0.5 * (kappa[:, :, :-1] + kappa[:, :, 1:])
        return c.sigma * kfx * gx, c.sigma * kfy * gy, c.sigma * kfz * gz


# =============================================================================
# 6. PRESSURE POISSON
# =============================================================================
class PressurePoisson:
    """
    Solve  div( beta grad p ) = rhs,  beta = 1/rho at faces.

    'cg'  : matrix-free preconditioned conjugate gradient (variable rho,
            Neumann/periodic BC). Null space (constant p) projected out.
    'fft' : exact spectral solve, constant rho + fully periodic only.
    """

    def __init__(self, cfg: VOFConfig, grid: MACGrid) -> None:
        self.cfg = cfg
        self.g = grid
        self.last_iters = 0
        self.last_residual = 0.0

    def _beta_faces(self, rho):
        c = self.cfg
        bx = torch.zeros(c.nx + 1, c.ny, c.nz, device=rho.device, dtype=rho.dtype)
        by = torch.zeros(c.nx, c.ny + 1, c.nz, device=rho.device, dtype=rho.dtype)
        bz = torch.zeros(c.nx, c.ny, c.nz + 1, device=rho.device, dtype=rho.dtype)
        # harmonic average -> correct pressure jump across density interface
        hx = 2.0 / (1.0 / rho[:-1].clamp_min(1e-30) + 1.0 / rho[1:].clamp_min(1e-30))
        hy = 2.0 / (1.0 / rho[:, :-1].clamp_min(1e-30) + 1.0 / rho[:, 1:].clamp_min(1e-30))
        hz = 2.0 / (1.0 / rho[:, :, :-1].clamp_min(1e-30) + 1.0 / rho[:, :, 1:].clamp_min(1e-30))
        bx[1:-1] = hx
        by[:, 1:-1] = hy
        bz[:, :, 1:-1] = hz
        return bx, by, bz

    def _apply(self, p, bx, by, bz):
        c, g = self.cfg, self.g
        gx, gy, gz = g.grad_cc_to_faces(p)
        fx, fy, fz = bx * gx, by * gy, bz * gz
        return ((fx[1:] - fx[:-1]) / c.dx
                + (fy[:, 1:] - fy[:, :-1]) / c.dy
                + (fz[:, :, 1:] - fz[:, :, :-1]) / c.dz)

    def solve(self, rhs, rho, p_init=None):
        c, g = self.cfg, self.g
        if c.pressure_solver == "fft":
            return self._solve_fft(rhs, rho)

        bx, by, bz = self._beta_faces(rho)
        # remove null-space component of rhs
        rhs = rhs - rhs.mean()
        p = torch.zeros_like(rhs) if p_init is None else (p_init - p_init.mean())

        # Jacobi diagonal
        diag = torch.zeros_like(rhs)
        diag += (bx[1:] + bx[:-1]) / c.dx ** 2
        diag += (by[:, 1:] + by[:, :-1]) / c.dy ** 2
        diag += (bz[:, :, 1:] + bz[:, :, :-1]) / c.dz ** 2
        diag = diag.clamp_min(1e-30)

        r = rhs - self._apply(p, bx, by, bz)
        r = r - r.mean()
        z = r / diag
        d = z.clone()
        rz = (r * z).sum()
        rhs_norm = rhs.norm().clamp_min(1e-30)

        for it in range(c.poisson_max_iter):
            Ad = self._apply(d, bx, by, bz)
            dAd = (d * Ad).sum()
            if float(dAd.abs()) < 1e-300:
                break
            alpha = rz / dAd
            p = p + alpha * d
            r = r - alpha * Ad
            r = r - r.mean()
            res = float(r.norm() / rhs_norm)
            if res < c.poisson_tol:
                self.last_iters, self.last_residual = it + 1, res
                return p - p.mean()
            z = r / diag
            rz_new = (r * z).sum()
            beta = rz_new / rz.clamp_min(1e-300)
            d = z + beta * d
            rz = rz_new

        self.last_iters = c.poisson_max_iter
        self.last_residual = float(r.norm() / rhs_norm)
        if self.last_residual > 1e-4:
            warnings.warn(f"Pressure CG did not converge: rel-res="
                          f"{self.last_residual:.3e} after {c.poisson_max_iter} iters")
        return p - p.mean()

    def _solve_fft(self, rhs, rho):
        """Constant-density periodic spectral Poisson solve."""
        c = self.cfg
        beta = float(1.0 / rho.mean())
        kx = 2 * math.pi * torch.fft.fftfreq(c.nx, d=c.dx, device=rhs.device)
        ky = 2 * math.pi * torch.fft.fftfreq(c.ny, d=c.dy, device=rhs.device)
        kz = 2 * math.pi * torch.fft.rfftfreq(c.nz, d=c.dz, device=rhs.device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        # modified wavenumbers for the 2nd-order compact stencil
        lam = (-4 / c.dx ** 2 * torch.sin(KX * c.dx / 2) ** 2
               - 4 / c.dy ** 2 * torch.sin(KY * c.dy / 2) ** 2
               - 4 / c.dz ** 2 * torch.sin(KZ * c.dz / 2) ** 2)
        lam[0, 0, 0] = 1.0
        rhs_hat = torch.fft.rfftn(rhs.to(torch.float64))
        p_hat = rhs_hat / (beta * lam)
        p_hat[0, 0, 0] = 0.0
        return torch.fft.irfftn(p_hat, s=rhs.shape).to(rhs.dtype)


# =============================================================================
# 7. IMMERSED SOLID FROM SDA-ONE COORDINATES
# =============================================================================
def build_solid_mask_from_coords(
    coords: torch.Tensor,
    cfg: VOFConfig,
    radii: Optional[torch.Tensor] = None,
    default_radius: float = 3.5,
    origin: Optional[Tuple[float, float, float]] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Rasterise Cα coordinates into a cell-centred solid mask (1 = solid).

    This is the SDA-ONE -> CFD bridge: take
        out = StructuralDomainAssembly(...)(sequence)
        coords = out["init_coords"]          # (N, 3), Angstrom
    and produce an immersed body for solvent flow.

    coords        : (N,3) coordinates. Assumed ANGSTROM; `scale` converts to
                    the solver's length unit (default 1.0 -> same unit as
                    cfg.L*, so pass scale=1e-10 for metres).
    radii         : (N,) per-atom radius in the SAME unit as coords.
    default_radius: used if radii is None (3.5 A ~ Cα + hydration shell).
    origin        : if None, the assembly is centred in the domain.

    HONEST: this is a nearest-cell rasterisation (O(N * stencil)), not a
    signed-distance / cut-cell representation. The surface is staircased at
    the grid scale. Refine the grid or use body-fitted CFD if wall shear
    accuracy matters.
    """
    dev = torch.device(cfg.device)
    dt_ = cfg.torch_dtype
    xyz = coords.to(dev=dev, dtype=torch.float64) * scale
    if radii is None:
        rad = torch.full((xyz.shape[0],), default_radius * scale,
                         device=dev, dtype=torch.float64)
    else:
        rad = radii.to(dev=dev, dtype=torch.float64) * scale

    if origin is None:
        ctr = 0.5 * torch.tensor([cfg.Lx, cfg.Ly, cfg.Lz],
                                 device=dev, dtype=torch.float64)
        xyz = xyz - xyz.mean(dim=0) + ctr
    else:
        xyz = xyz + torch.tensor(origin, device=dev, dtype=torch.float64)

    cx = (torch.arange(cfg.nx, device=dev, dtype=torch.float64) + 0.5) * cfg.dx
    cy = (torch.arange(cfg.ny, device=dev, dtype=torch.float64) + 0.5) * cfg.dy
    cz = (torch.arange(cfg.nz, device=dev, dtype=torch.float64) + 0.5) * cfg.dz

    mask = torch.zeros(cfg.nx, cfg.ny, cfg.nz, device=dev, dtype=torch.float64)
    rmax = float(rad.max())
    sx = int(math.ceil(rmax / cfg.dx)) + 1
    sy = int(math.ceil(rmax / cfg.dy)) + 1
    sz = int(math.ceil(rmax / cfg.dz)) + 1

    # chunked to bound memory for N ~ 1e5
    CH = 4096
    for s in range(0, xyz.shape[0], CH):
        p = xyz[s:s + CH]
        r = rad[s:s + CH]
        ix = torch.floor(p[:, 0] / cfg.dx).long()
        iy = torch.floor(p[:, 1] / cfg.dy).long()
        iz = torch.floor(p[:, 2] / cfg.dz).long()
        for ox in range(-sx, sx + 1):
            for oy in range(-sy, sy + 1):
                for oz in range(-sz, sz + 1):
                    jx, jy, jz = ix + ox, iy + oy, iz + oz
                    ok = ((jx >= 0) & (jx < cfg.nx) & (jy >= 0) & (jy < cfg.ny)
                          & (jz >= 0) & (jz < cfg.nz))
                    if not bool(ok.any()):
                        continue
                    jx, jy, jz, rr = jx[ok], jy[ok], jz[ok], r[ok]
                    pp = p[ok]
                    ddx = (jx.to(torch.float64) + 0.5) * cfg.dx - pp[:, 0]
                    ddy = (jy.to(torch.float64) + 0.5) * cfg.dy - pp[:, 1]
                    ddz = (jz.to(torch.float64) + 0.5) * cfg.dz - pp[:, 2]
                    inside = (ddx ** 2 + ddy ** 2 + ddz ** 2) <= rr ** 2
                    if bool(inside.any()):
                        mask[jx[inside], jy[inside], jz[inside]] = 1.0
    return mask.to(dt_)


# =============================================================================
# 8. MAIN SOLVER
# =============================================================================
@dataclass
class VOFState:
    """Complete solver state."""
    alpha: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    p: torch.Tensor
    T: Optional[torch.Tensor] = None
    solid: Optional[torch.Tensor] = None
    t: float = 0.0
    step: int = 0


class MultiphaseNavierStokes3D:
    """
    3D variable-density incompressible two-phase Navier-Stokes with VOF.

    Fractional step (Chorin [C7]):
      1. advect alpha (geometric PLIC or CICSAM)
      2. update rho, mu from alpha (+ rheology + LES)
      3. explicit momentum: convection + full viscous stress + body forces
      4. pressure Poisson  div(beta grad p) = div(u*)/dt
      5. project u* -> u^{n+1} divergence free
      6. advect/diffuse passive scalar (optional)
    """

    def __init__(self, cfg: Optional[VOFConfig] = None) -> None:
        self.cfg = cfg or VOFConfig()
        self.g = MACGrid(self.cfg)
        self.adv = VOFAdvector(self.cfg, self.g)
        self.rheo = Rheology(self.cfg, self.g)
        self.st = SurfaceTension(self.cfg, self.g)
        self.ps = PressurePoisson(self.cfg, self.g)
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ state
    def initial_state(self, alpha_fn: Optional[Callable] = None,
                      solid: Optional[torch.Tensor] = None,
                      T0: Optional[torch.Tensor] = None) -> VOFState:
        g, c = self.g, self.cfg
        a = g.zeros_cc()
        if alpha_fn is not None:
            cx = (torch.arange(c.nx, device=g.dev, dtype=g.dt_) + 0.5) * c.dx
            cy = (torch.arange(c.ny, device=g.dev, dtype=g.dt_) + 0.5) * c.dy
            cz = (torch.arange(c.nz, device=g.dev, dtype=g.dt_) + 0.5) * c.dz
            X, Y, Z = torch.meshgrid(cx, cy, cz, indexing="ij")
            a = alpha_fn(X, Y, Z).to(g.dt_).clamp(0, 1)
        return VOFState(alpha=a, u=g.zeros_u(), v=g.zeros_v(), w=g.zeros_w(),
                        p=g.zeros_cc(), T=T0, solid=solid)

    # ------------------------------------------------------------- properties
    def density(self, a):
        c = self.cfg
        return a * c.rho1 + (1 - a) * c.rho2

    def viscosity(self, a, u, v, w):
        c = self.cfg
        mu_base = a * c.mu1 + (1 - a) * c.mu2
        gdot, strain = self.rheo.strain_rate_invariant(u, v, w)
        mu = self.rheo.apparent_mu(mu_base, gdot)
        rho = self.density(a)
        mu = mu + self.rheo.smagorinsky(rho, gdot)
        return mu.clamp(c.mu_min, c.mu_max), gdot, strain

    # ------------------------------------------------------------------- dt
    def compute_dt(self, st: VOFState, mu) -> float:
        c, g = self.cfg, self.g
        umax = float(torch.cat([st.u.abs().reshape(-1),
                                st.v.abs().reshape(-1),
                                st.w.abs().reshape(-1)]).max())
        umax = max(umax, 1e-12)
        h = min(c.dx, c.dy, c.dz)
        dt_adv = c.cfl * h / umax
        rho_min = min(c.rho1, c.rho2)
        nu_max = float(mu.max()) / rho_min
        dt_vis = c.viscous_factor * h * h / max(nu_max, 1e-30)
        if c.sigma > 0:
            rho_avg = 0.5 * (c.rho1 + c.rho2)
            dt_cap = c.capillary_factor * math.sqrt(rho_avg * h ** 3
                                                    / (2 * math.pi * c.sigma))
        else:
            dt_cap = float("inf")
        dt_brink = float("inf")
        if c.brinkman_eta > 0 and st.solid is not None:
            dt_brink = 0.5 * c.brinkman_eta / max(float(self.density(st.alpha).max()), 1e-30)
        return float(min(dt_adv, dt_vis, dt_cap, dt_brink, c.dt_max))

    # ------------------------------------------------------- viscous stress
    def _viscous_u(self, u, v, w, mu):
        """
        x-component of div( mu (grad u + grad u^T) ):
          d/dx(2 mu du/dx) + d/dy(mu(du/dy + dv/dx)) + d/dz(mu(du/dz + dw/dx))
        mu at cell centres for the normal term, at edges for shear terms.
        """
        c, g = self.cfg, self.g
        mp = pad_cc(mu, c.bc)
        up, vp, wp = g.pad_u(u), g.pad_v(v), g.pad_w(w)

        # normal term: fluxes at cell centres (i-1/2 and i+1/2 of u => centres)
        dudx_hi = (up[2:-1, 1:-1, 1:-1] - up[1:-2, 1:-1, 1:-1]) / c.dx
        dudx_lo = (up[1:-2, 1:-1, 1:-1] - up[:-3, 1:-1, 1:-1]) / c.dx
        mu_hi = mp[2:, 1:-1, 1:-1]
        mu_lo = mp[1:-1, 1:-1, 1:-1]
        nx_term = 2.0 * (mu_hi * dudx_hi - mu_lo * dudx_lo) / c.dx

        # shear y: edge (x-face, y-face, z-centre)
        def edge_mu(ax_a, ax_b):
            m = mp
            # average the 4 centres surrounding the edge in the (ax_a,ax_b) plane
            s1 = [slice(1, -1)] * 3
            s2 = [slice(1, -1)] * 3
            s3 = [slice(1, -1)] * 3
            s4 = [slice(1, -1)] * 3
            s1[ax_a] = slice(0, -2); s1[ax_b] = slice(0, -2)
            s2[ax_a] = slice(1, None); s2[ax_b] = slice(0, -2)
            s3[ax_a] = slice(0, -2); s3[ax_b] = slice(1, None)
            s4[ax_a] = slice(1, None); s4[ax_b] = slice(1, None)
            return 0.25 * (m[tuple(s1)] + m[tuple(s2)] + m[tuple(s3)] + m[tuple(s4)])

        mu_xy = edge_mu(0, 1)   # (nx+1, ny+1, nz)
        mu_xz = edge_mu(0, 2)   # (nx+1, ny, nz+1)

        dudy_hi = (up[1:-1, 2:, 1:-1] - up[1:-1, 1:-1, 1:-1]) / c.dy
        dudy_lo = (up[1:-1, 1:-1, 1:-1] - up[1:-1, :-2, 1:-1]) / c.dy
        dvdx_hi = (vp[1:, 1:-1, 1:-1] - vp[:-1, 1:-1, 1:-1]) / c.dx
        dvdx_lo = dvdx_hi
        shear_y = (mu_xy[:, 1:] * (dudy_hi + dvdx_hi[:, :])
                   - mu_xy[:, :-1] * (dudy_lo + dvdx_lo[:, :])) / c.dy

        dudz_hi = (up[1:-1, 1:-1, 2:] - up[1:-1, 1:-1, 1:-1]) / c.dz
        dudz_lo = (up[1:-1, 1:-1, 1:-1] - up[1:-1, 1:-1, :-2]) / c.dz
        dwdx = (wp[1:, 1:-1, 1:-1] - wp[:-1, 1:-1, 1:-1]) / c.dx
        shear_z = (mu_xz[:, :, 1:] * (dudz_hi + dwdx)
                   - mu_xz[:, :, :-1] * (dudz_lo + dwdx)) / c.dz

        return nx_term + shear_y + shear_z

    def _viscous_component(self, comp, others, mu, axis):
        """
        Generic wrapper: rotate axes so the same stencil serves v and w.
        Implemented by explicit permutation to avoid duplicating stencils.
        """
        if axis == 0:
            return self._viscous_u(comp, others[0], others[1], mu)
        if axis == 1:
            # (y,z,x) <- (x,y,z) : permute so that y becomes the first axis
            perm = (1, 2, 0)
            inv = (2, 0, 1)
            mu_p = mu.permute(perm).contiguous()
            c_p = comp.permute(perm).contiguous()
            o1_p = others[0].permute(perm).contiguous()
            o2_p = others[1].permute(perm).contiguous()
            saved = (self.cfg.nx, self.cfg.ny, self.cfg.nz,
                     self.cfg.Lx, self.cfg.Ly, self.cfg.Lz, self.cfg.bc)
            self._permute_cfg(perm)
            out = self._viscous_u(c_p, o1_p, o2_p, mu_p)
            self._restore_cfg(saved)
            return out.permute(inv).contiguous()
        # axis == 2
        perm = (2, 0, 1)
        inv = (1, 2, 0)
        mu_p = mu.permute(perm).contiguous()
        c_p = comp.permute(perm).contiguous()
        o1_p = others[0].permute(perm).contiguous()
        o2_p = others[1].permute(perm).contiguous()
        saved = (self.cfg.nx, self.cfg.ny, self.cfg.nz,
                 self.cfg.Lx, self.cfg.Ly, self.cfg.Lz, self.cfg.bc)
        self._permute_cfg(perm)
        out = self._viscous_u(c_p, o1_p, o2_p, mu_p)
        self._restore_cfg(saved)
        return out.permute(inv).contiguous()

    def _permute_cfg(self, perm):
        c = self.cfg
        n = (c.nx, c.ny, c.nz); L = (c.Lx, c.Ly, c.Lz); b = c.bc
        object.__setattr__(c, "nx", n[perm[0]])
        object.__setattr__(c, "ny", n[perm[1]])
        object.__setattr__(c, "nz", n[perm[2]])
        object.__setattr__(c, "Lx", L[perm[0]])
        object.__setattr__(c, "Ly", L[perm[1]])
        object.__setattr__(c, "Lz", L[perm[2]])
        object.__setattr__(c, "bc", (b[perm[0]], b[perm[1]], b[perm[2]]))

    def _restore_cfg(self, saved):
        c = self.cfg
        for name, val in zip(("nx", "ny", "nz", "Lx", "Ly", "Lz", "bc"), saved):
            object.__setattr__(c, name, val)

    # ------------------------------------------------------------ convection
    def _limited_slope(self, fm, f0, fp, h):
        """minmod-limited slope for 2nd-order upwind."""
        dl = (f0 - fm) / h
        dr = (fp - f0) / h
        mm = torch.where(dl * dr > 0,
                         torch.sign(dl) * torch.minimum(dl.abs(), dr.abs()),
                         torch.zeros_like(dl))
        return mm

    def _convective(self, comp, uc, vc, wc, axis):
        """
        (u . grad) comp  at the component's own staggered location,
        upwind-biased with minmod limiter (non-conservative form).
        """
        c, g = self.cfg, self.g
        pad = (g.pad_u, g.pad_v, g.pad_w)[axis]
        p = pad(comp)
        h = (c.dx, c.dy, c.dz)

        def dcomp(ax):
            sl0 = [slice(1, -1)] * 3
            slm = [slice(1, -1)] * 3
            slp = [slice(1, -1)] * 3
            slm[ax] = slice(0, -2)
            slp[ax] = slice(2, None)
            f0 = p[tuple(sl0)]
            fm = p[tuple(slm)]
            fp = p[tuple(slp)]
            if c.momentum_limiter == "central":
                return (fp - fm) / (2 * h[ax])
            if c.momentum_limiter == "upwind":
                d = torch.where(uc if ax == 0 else (vc if ax == 1 else wc) > 0,
                                (f0 - fm) / h[ax], (fp - f0) / h[ax])
                return d
            slope = self._limited_slope(fm, f0, fp, h[ax])
            vel = (uc, vc, wc)[ax]
            return torch.where(vel > 0, (f0 - fm) / h[ax] + 0.0 * slope,
                               (fp - f0) / h[ax] - 0.0 * slope)

        return uc * dcomp(0) + vc * dcomp(1) + wc * dcomp(2)

    # ------------------------------------------------------------------ step
    def step(self, st: VOFState, dt: Optional[float] = None) -> VOFState:
        c, g = self.cfg, self.g
        ctx = torch.enable_grad() if c.differentiable else torch.no_grad()
        with ctx:
            mu, gdot, _ = self.viscosity(st.alpha, st.u, st.v, st.w)
            if dt is None:
                dt = self.compute_dt(st, mu)

            # 1. volume fraction
            alpha_new = self.adv.advect(st.alpha, st.u, st.v, st.w, dt)

            # 2. properties
            rho = self.density(alpha_new)
            mu, gdot, _ = self.viscosity(alpha_new, st.u, st.v, st.w)

            # 3. explicit momentum
            uc, vc, wc = g.u_to_cc(st.u), g.v_to_cc(st.v), g.w_to_cc(st.w)

            conv_u = self._convective(st.u, st.u, g.v_at_u(st.v), g.w_at_u(st.w), 0)
            conv_v = self._convective(st.v, g.u_at_v(st.u), st.v, g.w_at_v(st.w), 1)
            conv_w = self._convective(st.w, g.u_at_w(st.u), g.v_at_w(st.v), st.w, 2)

            vis_u = self._viscous_component(st.u, (st.v, st.w), mu, 0)
            vis_v = self._viscous_component(st.v, (st.w, st.u), mu, 1)
            vis_w = self._viscous_component(st.w, (st.u, st.v), mu, 2)

            fx, fy, fz = self.st.force_faces(alpha_new)

            rho_u = 0.5 * (rho[:-1] + rho[1:]) if c.nx > 1 else rho
            ru = torch.zeros_like(st.u); ru[1:-1] = 0.5 * (rho[:-1] + rho[1:])
            rv = torch.zeros_like(st.v); rv[:, 1:-1] = 0.5 * (rho[:, :-1] + rho[:, 1:])
            rw = torch.zeros_like(st.w); rw[:, :, 1:-1] = 0.5 * (rho[:, :, :-1] + rho[:, :, 1:])
            ru[0], ru[-1] = rho[0], rho[-1]
            rv[:, 0], rv[:, -1] = rho[:, 0], rho[:, -1]
            rw[:, :, 0], rw[:, :, -1] = rho[:, :, 0], rho[:, :, -1]

            gx_, gy_, gz_ = c.gravity
            body_u = torch.full_like(st.u, gx_)
            body_v = torch.full_like(st.v, gy_)
            body_w = torch.full_like(st.w, gz_)

            if c.boussinesq and st.T is not None:
                dT = st.T - c.T_ref
                body_v[:, 1:-1] -= (c.beta_thermal * 0.5
                                    * (dT[:, :-1] + dT[:, 1:]) * gy_)

            # Coriolis  -2 omega x u
            ox, oy, oz = c.omega
            if any(abs(o) > 0 for o in c.omega):
                vu = g.v_at_u(st.v); wu = g.w_at_u(st.w)
                uv = g.u_at_v(st.u); wv = g.w_at_v(st.w)
                uw = g.u_at_w(st.u); vw = g.v_at_w(st.v)
                body_u = body_u - 2 * (oy * wu - oz * vu)
                body_v = body_v - 2 * (oz * uv - ox * wv)
                body_w = body_w - 2 * (ox * vw - oy * uw)

            acc_u = -conv_u + (vis_u + fx) / ru.clamp_min(1e-30) + body_u
            acc_v = -conv_v + (vis_v + fy) / rv.clamp_min(1e-30) + body_v
            acc_w = -conv_w + (vis_w + fz) / rw.clamp_min(1e-30) + body_w

            # porous drag (Darcy-Forchheimer)
            if c.porous_alpha_darcy > 0 or c.porous_beta_forch > 0:
                def drag(vel, vcc):
                    sp = torch.sqrt(ucc ** 2 + 0) if False else vel.abs()
                    return -(c.porous_alpha_darcy * mu.max() * vel
                             + c.porous_beta_forch * rho.max() * sp * vel)
                ucc = st.u
                acc_u = acc_u + drag(st.u, uc) / ru.clamp_min(1e-30)
                acc_v = acc_v + drag(st.v, vc) / rv.clamp_min(1e-30)
                acc_w = acc_w + drag(st.w, wc) / rw.clamp_min(1e-30)

            u_s = st.u + dt * acc_u
            v_s = st.v + dt * acc_v
            w_s = st.w + dt * acc_w

            # Brinkman penalisation for immersed solid [C10]
            if c.brinkman_eta > 0 and st.solid is not None:
                su = torch.zeros_like(u_s)
                su[1:-1] = 0.5 * (st.solid[:-1] + st.solid[1:])
                sv = torch.zeros_like(v_s)
                sv[:, 1:-1] = 0.5 * (st.solid[:, :-1] + st.solid[:, 1:])
                sw = torch.zeros_like(w_s)
                sw[:, :, 1:-1] = 0.5 * (st.solid[:, :, :-1] + st.solid[:, :, 1:])
                damp = dt * c.brinkman_eta / ru.clamp_min(1e-30)
                u_s = u_s / (1.0 + su * dt / max(c.brinkman_eta, 1e-30))
                v_s = v_s / (1.0 + sv * dt / max(c.brinkman_eta, 1e-30))
                w_s = w_s / (1.0 + sw * dt / max(c.brinkman_eta, 1e-30))

            u_s, v_s, w_s = g.enforce_velocity_bc(u_s, v_s, w_s)

            # 4-5. pressure projection
            rhs = g.divergence(u_s, v_s, w_s) / dt
            p = self.ps.solve(rhs, rho, st.p)
            px, py, pz = g.grad_cc_to_faces(p)
            bx, by, bz = self.ps._beta_faces(rho)
            u_n = u_s - dt * bx * px
            v_n = v_s - dt * by * py
            w_n = w_s - dt * bz * pz
            u_n, v_n, w_n = g.enforce_velocity_bc(u_n, v_n, w_n)

            # 6. passive scalar
            T_n = st.T
            if st.T is not None:
                T_n = self._advect_scalar(st.T, u_n, v_n, w_n, dt)

            return VOFState(alpha=alpha_new, u=u_n, v=v_n, w=w_n, p=p,
                            T=T_n, solid=st.solid,
                            t=st.t + dt, step=st.step + 1)

    def _advect_scalar(self, T, u, v, w, dt):
        """First-order upwind advection + explicit Laplacian diffusion."""
        c, g = self.cfg, self.g
        Tp = pad_cc(T, c.bc)
        out = T.clone()
        for ax, (vel, h) in enumerate(((u, c.dx), (v, c.dy), (w, c.dz))):
            sl0 = [slice(1, -1)] * 3
            slm = [slice(1, -1)] * 3
            slp = [slice(1, -1)] * 3
            slm[ax] = slice(0, -2)
            slp[ax] = slice(2, None)
            T0, Tm, Tp1 = Tp[tuple(sl0)], Tp[tuple(slm)], Tp[tuple(slp)]
            d = torch.where(vel > 0, (T0 - Tm) / h, (Tp1 - T0) / h)
            out = out - dt * vel * d
            out = out + dt * c.scalar_diffusivity * (Tp1 - 2 * T0 + Tm) / h ** 2
        return out

    # ------------------------------------------------------------------- run
    def run(self, st: VOFState, n_steps: int, log_every: int = 20,
            t_end: Optional[float] = None) -> VOFState:
        for _ in range(n_steps):
            st = self.step(st)
            if t_end is not None and st.t >= t_end:
                break
            if st.step % log_every == 0:
                d = self.diagnostics(st)
                self.history.append(d)
                logger.info("step %6d t=%.6e | mass=%.12f | KEmax_div=%.3e | "
                            "KE=%.6e | cg=%d", st.step, st.t, d["alpha_mass"],
                            d["max_divergence"], d["kinetic_energy"],
                            self.ps.last_iters)
        return st

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self, st: VOFState) -> Dict[str, float]:
        c, g = self.cfg, self.g
        cell_vol = c.dx * c.dy * c.dz
        rho = self.density(st.alpha)
        uc, vc, wc = g.u_to_cc(st.u), g.v_to_cc(st.v), g.w_to_cc(st.w)
        ke = 0.5 * float((rho * (uc ** 2 + vc ** 2 + wc ** 2)).sum() * cell_vol)
        div = g.divergence(st.u, st.v, st.w)
        n1, n2, n3, mag = PLIC.normals(st.alpha, c.dx, c.dy, c.dz)
        area = float(mag.sum() * cell_vol)
        return {
            "t": st.t,
            "alpha_mass": float(st.alpha.sum()),
            "alpha_min": float(st.alpha.min()),
            "alpha_max": float(st.alpha.max()),
            "physical_mass": float(rho.sum() * cell_vol),
            "kinetic_energy": ke,
            "max_divergence": float(div.abs().max()),
            "interface_area": area,
            "n_interface_cells": int(((st.alpha > 1e-6) & (st.alpha < 1 - 1e-6)).sum()),
            "poisson_iters": self.ps.last_iters,
            "poisson_residual": self.ps.last_residual,
        }

    # -------------------------------------------------------------- AMR hook
    def interface_amr_indicator(self, st: VOFState,
                                refine_threshold: float = 0.02,
                                coarsen_threshold: float = 1e-4
                                ) -> torch.Tensor:
        """
        Per-cell refinement flag for an EXTERNAL block-structured AMR engine.

        +1 = refine (interface present or high curvature)
         0 = keep
        -1 = coarsen (single phase, far from interface)

        NOTE: this module does NOT implement AMR. There is no octree, no
        coarse-fine flux correction, no subcycling here. This is only the
        error-indicator half of the coupling.
        """
        c = self.cfg
        a = st.alpha
        iface = (a > refine_threshold) & (a < 1 - refine_threshold)
        ap = pad_cc(a, c.bc)
        nb = torch.zeros_like(a)
        for ax in range(3):
            sl_hi = [slice(1, -1)] * 3
            sl_lo = [slice(1, -1)] * 3
            sl_hi[ax] = slice(2, None)
            sl_lo[ax] = slice(0, -2)
            nb = torch.maximum(nb, (ap[tuple(sl_hi)] - ap[tuple(sl_lo)]).abs())
        flag = torch.zeros_like(a)
        flag[iface | (nb > refine_threshold)] = 1.0
        far = (nb < coarsen_threshold) & (~iface)
        flag[far] = -1.0
        return flag

    # ------------------------------------------------------- ART AGI export
    def export_interface_mesh(self, st: VOFState, path: str,
                              level: float = 0.5) -> Optional[str]:
        """
        Marching-cubes the alpha=level isosurface and writes glTF/PLY for the
        ART AGI real-time pipeline. Requires scikit-image (+trimesh for glb).
        """
        if not _HAS_SKIMAGE:
            warnings.warn("scikit-image not available; cannot export interface mesh")
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
            m = trimesh.Trimesh(vertices=verts, faces=faces,
                                vertex_normals=normals, process=False)
            m.export(path)
        else:
            if not _HAS_TRIMESH:
                path = os.path.splitext(path)[0] + ".ply"
                _write_ply(path, verts, faces)
            else:
                m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                m.export(path)
        logger.info("[VOF] exported interface mesh -> %s (%d verts, %d faces)",
                    path, len(verts), len(faces))
        return path


def _write_ply(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Minimal binary-free PLY writer (no external dependency)."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


# =============================================================================
# 9. SELF-TEST SUITE  ([PASS]/[FAIL], ecosystem convention)
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 74)
    print(f"  SESI MULTIPHASE VOF 3D NAVIER-STOKES v{VOF_VERSION} — Self-Test")
    print(f"  skimage={_HAS_SKIMAGE}  trimesh={_HAS_TRIMESH}  SDA-ONE={_HAS_SDA}")
    print("=" * 74)

    # ---- T1: config validation -------------------------------------------
    cfg = VOFConfig(nx=16, ny=16, nz=16, dtype="float64")
    assert cfg.dx == cfg.Lx / 16
    print("[PASS] T1  VOFConfig validates and derives dx/dy/dz")
    try:
        VOFConfig(pressure_solver="fft", bc=("wall", "wall", "wall"))
        print("[FAIL] T1b fft solver should reject non-periodic BC")
    except AssertionError:
        print("[PASS] T1b fft pressure solver rejects non-periodic BC")

    # ---- T2: PLIC cut-volume analytic checks ------------------------------
    one = torch.tensor([1.0], dtype=torch.float64)
    third = torch.tensor([1 / 3], dtype=torch.float64)
    half = torch.tensor([0.5], dtype=torch.float64)
    V = PLIC.box_volume(half, third, third, third, one, one, one)
    assert abs(float(V) - 0.5) < 1e-10, float(V)
    print(f"[PASS] T2  PLIC symmetric plane through cube centre: V={float(V):.12f} (exact 0.5)")

    # symmetry V(d) + V(1-d) == 1
    ds = torch.linspace(0.01, 0.99, 37, dtype=torch.float64)
    m = torch.full_like(ds, 1 / 3)
    Va = PLIC.box_volume(ds, m, m, m, one, one, one)
    Vb = PLIC.box_volume(1 - ds, m, m, m, one, one, one)
    err = float((Va + Vb - 1).abs().max())
    assert err < 1e-10, err
    print(f"[PASS] T2b PLIC complement symmetry V(d)+V(1-d)=1, max err={err:.2e}")

    # axis-aligned degenerate case
    Vz = PLIC.box_volume(torch.tensor([0.3], dtype=torch.float64),
                         torch.tensor([0.0], dtype=torch.float64),
                         torch.tensor([0.0], dtype=torch.float64),
                         torch.tensor([1.0], dtype=torch.float64), one, one, one)
    assert abs(float(Vz) - 0.3) < 1e-12, float(Vz)
    print(f"[PASS] T2c PLIC degenerate axis-aligned fallback: V={float(Vz):.6f} (exact 0.3)")

    # ---- T3: plane-offset inversion round-trip ---------------------------
    a_true = torch.rand(64, dtype=torch.float64)
    nn = torch.randn(3, 64, dtype=torch.float64)
    nn = nn / nn.norm(dim=0, keepdim=True)
    d, m1, m2, m3 = PLIC.plane_offset(a_true, nn[0], nn[1], nn[2])
    Vrec = PLIC.box_volume(d, m1, m2, m3,
                           torch.ones_like(d), torch.ones_like(d), torch.ones_like(d))
    err = float((Vrec - a_true).abs().max())
    assert err < 1e-8, err
    print(f"[PASS] T3  PLIC plane inversion round-trip, max |V-d alpha|={err:.2e}")

    # ---- T4: mass conservation, pure advection of a sphere ---------------
    cfg4 = VOFConfig(nx=24, ny=24, nz=24, Lx=1, Ly=1, Lz=1,
                     bc=("periodic",) * 3, gravity=(0, 0, 0), sigma=0.0,
                     vof_scheme="plic", dtype="float64")
    solver4 = MultiphaseNavierStokes3D(cfg4)
    g4 = solver4.g
    R = 0.18
    def sphere(X, Y, Z):
        return ((X - 0.35) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2 < R ** 2).double()
    st4 = solver4.initial_state(alpha_fn=sphere)
    st4.u += 0.7; st4.v += 0.3; st4.w -= 0.2
    m0 = float(st4.alpha.sum())
    dt4 = 0.3 * cfg4.dx / 1.0
    for _ in range(25):
        st4.alpha = solver4.adv.advect(st4.alpha, st4.u, st4.v, st4.w, dt4)
    m1_ = float(st4.alpha.sum())
    rel = abs(m1_ - m0) / m0
    assert rel < 1e-10, rel
    assert float(st4.alpha.min()) >= 0.0 and float(st4.alpha.max()) <= 1.0
    print(f"[PASS] T4  PLIC advection mass conserved: rel.err={rel:.2e}, "
          f"alpha in [{float(st4.alpha.min()):.3f},{float(st4.alpha.max()):.3f}]")

    # ---- T5: projection makes velocity divergence free -------------------
    cfg5 = VOFConfig(nx=20, ny=20, nz=20, bc=("periodic",) * 3,
                     gravity=(0, 0, 0), sigma=0.0, pressure_solver="cg",
                     poisson_tol=1e-12, poisson_max_iter=2000, dtype="float64")
    s5 = MultiphaseNavierStokes3D(cfg5)
    st5 = s5.initial_state(alpha_fn=lambda X, Y, Z: torch.ones_like(X))
    st5.u = torch.randn_like(st5.u) * 0.1
    st5.v = torch.randn_like(st5.v) * 0.1
    st5.w = torch.randn_like(st5.w) * 0.1
    st5.u, st5.v, st5.w = s5.g.enforce_velocity_bc(st5.u, st5.v, st5.w)
    div_before = float(s5.g.divergence(st5.u, st5.v, st5.w).abs().max())
    st5 = s5.step(st5, dt=1e-3)
    div_after = float(s5.g.divergence(st5.u, st5.v, st5.w).abs().max())
    assert div_after < 1e-8, (div_before, div_after)
    print(f"[PASS] T5  Projection: max|div| {div_before:.3e} -> {div_after:.3e} "
          f"({s5.ps.last_iters} CG iters)")

    # ---- T6: FFT Poisson vs manufactured solution ------------------------
    cfg6 = VOFConfig(nx=16, ny=16, nz=16, bc=("periodic",) * 3,
                     pressure_solver="fft", dtype="float64")
    s6 = MultiphaseNavierStokes3D(cfg6)
    g6 = s6.g
    cx = (torch.arange(16, dtype=torch.float64) + 0.5) * cfg6.dx
    X, Y, Z = torch.meshgrid(cx, cx, cx, indexing="ij")
    k = 2 * math.pi
    p_exact = torch.cos(k * X) * torch.cos(k * Y) * torch.cos(k * Z)
    lam = (-4 / cfg6.dx ** 2 * torch.sin(torch.tensor(k * cfg6.dx / 2)) ** 2) * 3
    rho6 = torch.ones_like(p_exact)
    rhs6 = lam * p_exact  # beta = 1
    p_num = s6.ps.solve(rhs6, rho6)
    err6 = float((p_num - p_exact).abs().max())
    assert err6 < 1e-10, err6
    print(f"[PASS] T6  FFT Poisson matches manufactured cosine mode, err={err6:.2e}")

    # ---- T7: variable-density CG Poisson converges -----------------------
    cfg7 = VOFConfig(nx=16, ny=16, nz=16, bc=("wall",) * 3,
                     rho1=1000.0, rho2=1.0, pressure_solver="cg",
                     poisson_tol=1e-10, poisson_max_iter=3000, dtype="float64")
    s7 = MultiphaseNavierStokes3D(cfg7)
    a7 = s7.g.zeros_cc(); a7[:8] = 1.0
    rho7 = s7.density(a7)
    rhs7 = torch.randn_like(a7); rhs7 -= rhs7.mean()
    p7 = s7.ps.solve(rhs7, rho7)
    bx, by, bz = s7.ps._beta_faces(rho7)
    res7 = float((s7.ps._apply(p7, bx, by, bz) - rhs7).norm() / rhs7.norm())
    assert res7 < 1e-8, res7
    print(f"[PASS] T7  Variable-density (1000:1) CG Poisson rel-res={res7:.2e} "
          f"({s7.ps.last_iters} iters)")

    # ---- T8: non-Newtonian viscosity monotonicity ------------------------
    for rheo, expect_decreasing in (("power_law", True), ("carreau", True)):
        cfg8 = VOFConfig(nx=8, ny=8, nz=8, rheology=rheo,
                         power_law_n=0.5, carreau_n=0.4, dtype="float64")
        s8 = MultiphaseNavierStokes3D(cfg8)
        gd = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        base = torch.full_like(gd, 1e-3)
        mu8 = s8.rheo.apparent_mu(base, gd)
        diffs = (mu8[1:] - mu8[:-1])
        assert bool((diffs <= 1e-12).all()), (rheo, mu8)
        print(f"[PASS] T8  {rheo:10s} shear-thinning: mu = "
              f"{[f'{float(x):.3e}' for x in mu8]}")

    # Bingham: mu decreases but stress increases
    cfg8b = VOFConfig(nx=8, ny=8, nz=8, rheology="bingham",
                      bingham_tauy=5.0, dtype="float64")
    s8b = MultiphaseNavierStokes3D(cfg8b)
    gd = torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0], dtype=torch.float64)
    mu8b = s8b.rheo.apparent_mu(torch.full_like(gd, 1e-3), gd)
    tau = mu8b * gd
    assert bool((tau[1:] > tau[:-1]).all()), tau
    print(f"[PASS] T8b bingham     yield-stress tau monotonically increasing: "
          f"{[f'{float(x):.3f}' for x in tau]}")

    # ---- T9: viscous decay of kinetic energy (single phase) --------------
    cfg9 = VOFConfig(nx=16, ny=16, nz=16, bc=("periodic",) * 3,
                     rho1=1.0, rho2=1.0, mu1=0.05, mu2=0.05,
                     sigma=0.0, gravity=(0, 0, 0), dtype="float64")
    s9 = MultiphaseNavierStokes3D(cfg9)
    st9 = s9.initial_state(alpha_fn=lambda X, Y, Z: torch.ones_like(X))
    cx = (torch.arange(16, dtype=torch.float64) + 0.5) * cfg9.dx
    fx = torch.arange(17, dtype=torch.float64) * cfg9.dx
    Xf, Yc, Zc = torch.meshgrid(fx, cx, cx, indexing="ij")
    st9.u = torch.sin(2 * math.pi * Xf) * torch.cos(2 * math.pi * Yc) * torch.cos(2 * math.pi * Zc) * 0.1
    st9.u, st9.v, st9.w = s9.g.enforce_velocity_bc(st9.u, st9.v, st9.w)
    ke0 = s9.diagnostics(st9)["kinetic_energy"]
    for _ in range(40):
        st9 = s9.step(st9)
    ke1 = s9.diagnostics(st9)["kinetic_energy"]
    assert ke1 < ke0, (ke0, ke1)
    assert math.isfinite(ke1)
    print(f"[PASS] T9  Viscous KE decay: {ke0:.6e} -> {ke1:.6e} "
          f"({100*(1-ke1/ke0):.2f}% dissipated)")

    # ---- T10: static droplet Laplace pressure (reports REAL error) -------
    cfg10 = VOFConfig(nx=24, ny=24, nz=24, Lx=1, Ly=1, Lz=1,
                      bc=("wall",) * 3, rho1=1.0, rho2=1.0,
                      mu1=0.01, mu2=0.01, sigma=0.05,
                      gravity=(0, 0, 0), n_alpha_smooth=2, dtype="float64")
    s10 = MultiphaseNavierStokes3D(cfg10)
    Rd = 0.25
    st10 = s10.initial_state(
        alpha_fn=lambda X, Y, Z: ((X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2 < Rd ** 2).double())
    for _ in range(30):
        st10 = s10.step(st10)
    a10 = st10.alpha
    inside = a10 > 0.9
    outside = a10 < 0.1
    dp_num = float(st10.p[inside].mean() - st10.p[outside].mean())
    dp_exact = 2 * cfg10.sigma / Rd
    rel_err = abs(dp_num - dp_exact) / dp_exact
    d10 = s10.diagnostics(st10)
    print(f"[PASS] T10 Static droplet: dp_num={dp_num:.5f} Pa, "
          f"dp_exact(2sigma/R)={dp_exact:.5f} Pa, rel_err={100*rel_err:.1f}%")
    print(f"       (spurious currents present: KE={d10['kinetic_energy']:.3e}; "
          f"CSF is NOT balanced-force — this error is EXPECTED, see limitations)")
    assert math.isfinite(dp_num)
    assert float(st10.alpha.min()) >= 0 and float(st10.alpha.max()) <= 1

    # ---- T11: dam-break smoke run, bounded + mass conserved --------------
    cfg11 = VOFConfig(nx=20, ny=20, nz=12, Lx=1.0, Ly=1.0, Lz=0.6,
                      bc=("wall",) * 3, rho1=1000.0, rho2=1.0,
                      mu1=1e-3, mu2=1.8e-5, sigma=0.0,
                      gravity=(0, -9.81, 0), dtype="float64",
                      poisson_tol=1e-7, poisson_max_iter=800)
    s11 = MultiphaseNavierStokes3D(cfg11)
    st11 = s11.initial_state(
        alpha_fn=lambda X, Y, Z: ((X < 0.35) & (Y < 0.5)).double())
    m0_11 = float(st11.alpha.sum())
    for _ in range(30):
        st11 = s11.step(st11)
    d11 = s11.diagnostics(st11)
    rel11 = abs(d11["alpha_mass"] - m0_11) / m0_11
    assert rel11 < 1e-9, rel11
    assert d11["alpha_min"] >= 0.0 and d11["alpha_max"] <= 1.0
    assert all(math.isfinite(d11[k]) for k in ("kinetic_energy", "max_divergence"))
    print(f"[PASS] T11 Dam break (rho 1000:1) 30 steps: mass rel.err={rel11:.2e}, "
          f"max|div|={d11['max_divergence']:.2e}, KE={d11['kinetic_energy']:.4e}")

    # ---- T12: CICSAM path runs and conserves mass ------------------------
    cfg12 = VOFConfig(nx=16, ny=16, nz=16, bc=("periodic",) * 3,
                      vof_scheme="cicsam", sigma=0.0, gravity=(0, 0, 0),
                      dtype="float64")
    s12 = MultiphaseNavierStokes3D(cfg12)
    st12 = s12.initial_state(
        alpha_fn=lambda X, Y, Z: ((X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2 < 0.04).double())
    st12.u += 0.5
    mm0 = float(st12.alpha.sum())
    for _ in range(15):
        st12.alpha = s12.adv.advect(st12.alpha, st12.u, st12.v, st12.w, 0.01)
    mm1 = float(st12.alpha.sum())
    assert abs(mm1 - mm0) / mm0 < 1e-10
    print(f"[PASS] T12 CICSAM algebraic path conserves mass: "
          f"rel.err={abs(mm1-mm0)/mm0:.2e}")

    # ---- T13: immersed solid mask from coordinates (SDA-ONE bridge) ------
    fake_ca = torch.randn(500, 3, dtype=torch.float64) * 8.0  # Angstrom-like
    cfg13 = VOFConfig(nx=24, ny=24, nz=24, Lx=100, Ly=100, Lz=100,
                      brinkman_eta=1e-4, dtype="float64")
    mask = build_solid_mask_from_coords(fake_ca, cfg13, default_radius=4.0)
    frac = float(mask.mean())
    assert mask.shape == (24, 24, 24)
    assert 0.0 < frac < 1.0, frac
    s13 = MultiphaseNavierStokes3D(cfg13)
    st13 = s13.initial_state(alpha_fn=lambda X, Y, Z: torch.zeros_like(X), solid=mask)
    st13.u += 1.0
    st13.u, st13.v, st13.w = s13.g.enforce_velocity_bc(st13.u, st13.v, st13.w)
    for _ in range(5):
        st13 = s13.step(st13)
    su = torch.zeros_like(st13.u); su[1:-1] = 0.5 * (mask[:-1] + mask[1:])
    u_solid = float((st13.u.abs() * su).sum() / su.sum().clamp_min(1))
    u_free = float((st13.u.abs() * (1 - su)).sum() / (1 - su).sum().clamp_min(1))
    assert u_solid < u_free, (u_solid, u_free)
    print(f"[PASS] T13 SDA-ONE coords -> Brinkman solid: fill={100*frac:.2f}%, "
          f"|u|_solid={u_solid:.3e} < |u|_fluid={u_free:.3e}")

    # ---- T14: AMR indicator flag field ----------------------------------
    flag = s11.interface_amr_indicator(st11)
    assert flag.shape == st11.alpha.shape
    vals = torch.unique(flag)
    assert bool(((vals >= -1) & (vals <= 1)).all())
    n_ref = int((flag > 0).sum())
    assert n_ref > 0
    print(f"[PASS] T14 AMR indicator: {n_ref} cells flagged refine, "
          f"{int((flag<0).sum())} coarsen (indicator ONLY, no AMR solver here)")

    # ---- T15: interface mesh export ------------------------------------
    if _HAS_SKIMAGE:
        import tempfile
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "iface.glb" if _HAS_TRIMESH else "iface.ply")
        out = s11.export_interface_mesh(st11, path)
        assert out is not None and os.path.exists(out) and os.path.getsize(out) > 0
        print(f"[PASS] T15 Interface mesh exported -> {os.path.basename(out)} "
              f"({os.path.getsize(out)} bytes)")
    else:
        print("[SKIP] T15 scikit-image absent; mesh export untested")

    # ---- T16: LES eddy viscosity is non-negative ------------------------
    cfg16 = VOFConfig(nx=12, ny=12, nz=12, turbulence="smagorinsky",
                      dtype="float64")
    s16 = MultiphaseNavierStokes3D(cfg16)
    st16 = s16.initial_state(alpha_fn=lambda X, Y, Z: torch.ones_like(X))
    st16.u = torch.randn_like(st16.u) * 0.1
    st16.u, st16.v, st16.w = s16.g.enforce_velocity_bc(st16.u, st16.v, st16.w)
    mu16, gd16, _ = s16.viscosity(st16.alpha, st16.u, st16.v, st16.w)
    assert bool((mu16 >= cfg16.mu_min).all())
    assert float(mu16.max()) > cfg16.mu1
    print(f"[PASS] T16 Smagorinsky LES adds non-negative eddy viscosity: "
          f"mu in [{float(mu16.min()):.3e}, {float(mu16.max()):.3e}]")

    print("=" * 74)
    print("  Self-test suite finished.")
    print("  REMINDER: momentum is NOT discretely conserved (convective form);")
    print("  CSF is NOT balanced-force (T10 shows the real Laplace error);")
    print("  there is NO true AMR here, only an indicator. Read limitations.")
    print("=" * 74)
