#!/usr/bin/env python3
"""
=============================================================================
MARTINI 3 (small/tiny beads + ring constraints + virtual sites)
+ FREE-ENERGY VALIDATION: PMF from umbrella sampling, CG vs AA
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

WHY THIS FILE EXISTS
  Force matching (MS-CG) reproduces the mean force at the sampled
  configurations. It does NOT guarantee that the coarse-grained model
  reproduces the potential of mean force along a reaction coordinate,
  the equilibrium structure, or any free-energy difference. This is the
  standard representability problem of coarse-graining. A CG model can
  match forces well and still give a wrong binding free energy.
  This module therefore adds a DIRECT thermodynamic test: compute the
  PMF along a chosen coordinate for both AA and CG by umbrella sampling
  + WHAM, and report the difference with block-averaged error bars.

WHAT IS IMPLEMENTED
  [M3-1] Martini 3 bead SIZE CLASSES: R (regular), S (small), T (tiny),
         with the nominal Martini 3 sigma values 0.47 / 0.41 / 0.34 nm.
         Cross-size interactions come from an EXPLICIT pair matrix, as
         Martini 3 does; a size-scaled fallback exists but is clearly
         flagged as NOT Martini.
  [M3-2] RING CONSTRAINTS: holonomic distance constraints solved by
         SHAKE (positions) and RATTLE (velocities), replacing the stiff
         harmonic bonds Martini 2 used for ring moieties. A stiff-spring
         alternative is provided when differentiability is required.
  [M3-3] VIRTUAL SITES: positions built as fixed linear combinations of
         parent beads (Martini 3 uses these for planar rings). Forces
         are redistributed to parents by the exact transpose of the
         construction, conserving total force and torque.
  [PMF-1] Umbrella sampling driver for BOTH backends:
            AA : OpenMM CustomCentroidBondForce harmonic bias
            CG : native harmonic bias in this module
  [PMF-2] WHAM (Kumar et al. 1992) self-consistent unbiasing.
  [PMF-3] Block-averaged error bars + window-overlap diagnostic.
  [PMF-4] AA-vs-CG comparison: aligned PMFs, RMSE, and the difference in
          a user-specified bound/unbound free-energy window.

HONEST SCOPE — READ BEFORE USING ANY NUMBER
  ❌ Official Martini 3 interaction parameters are NOT bundled. The
     sigma values for R/S/T are the published nominal radii; every
     epsilon and every pair-matrix entry here is a PLACEHOLDER until
     you load the official martini_v3.0*.itp. Load them with
     M3Params.from_json() after converting the official itp.
  ❌ Ring constraint topology must be supplied per molecule. The
     auto-builder here is a geometric heuristic, NOT a chemistry-aware
     Martini 3 mapping (which respects ring membership and functional
     groups). Use a real mapping file for production.
  ❌ PMF convergence is NOT guaranteed by running the driver. The block
     error and overlap diagnostics are provided precisely because a
     PMF can look plausible and be unconverged. Read them.
  ❌ A CG PMF is a free energy at ONE temperature and ONE composition.
     Agreement at one state point does not establish transferability.
  ❌ Umbrella sampling with a distance coordinate does not separate
     enthalpy and entropy, and does not account for standard-state
     volume corrections unless you apply them explicitly.
  ❌ No claim of state of the art. Run it and read the table.

REFERENCES
  [1] Souza et al. (2021) Martini 3. Nature Methods 18, 382-388.
  [2] Kumar, Rosenberg, Bouzida, Swendsen, Kollman (1992) WHAM.
      J. Comput. Chem. 13, 1011-1021.
  [3] Torrie & Valleau (1977) Umbrella sampling. JCP 23, 187.
  [4] Ryckaert, Ciccotti, Berendsen (1977) SHAKE. JCP 23, 327.
  [5] Andersen (1983) RATTLE. JCP 52, 24.
  [6] Izvekov & Voth (2005) MS-CG force matching. JPC B 109, 2469.
  [7] Louis (2002) Beware of density-dependent pair potentials:
      representability. J. Phys.: Condens. Matter 14, 9187.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("m3_pmf")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# OpenMM units: nm, kJ/mol, ps, amu, e
KB = 0.008314462618          # kJ mol^-1 K^-1
COULOMB = 138.935456         # kJ nm mol^-1 e^-2

# Martini 3 nominal bead radii (sigma, nm) — published size classes [1]
M3_SIGMA = {"R": 0.47, "S": 0.41, "T": 0.34}

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    mm = app = unit = None
    HAS_OPENMM = False


# =============================================================================
# §1  MARTINI 3 BEAD TYPES WITH SIZE CLASSES
# =============================================================================

@dataclass
class M3BeadType:
    name: str
    size: str = "R"            # "R" | "S" | "T"
    epsilon: float = 3.0       # kJ/mol  (PLACEHOLDER unless loaded)
    charge: float = 0.0
    mass: float = 72.0
    placeholder: bool = True

    @property
    def sigma(self) -> float:
        return M3_SIGMA[self.size]


@dataclass
class M3Bond:
    i: int
    j: int
    r0: float                  # nm
    k: float                   # kJ/mol/nm^2


@dataclass
class M3Angle:
    i: int
    j: int
    k: int
    theta0: float              # degrees
    k_angle: float             # kJ/mol


@dataclass
class M3Constraint:
    """Holonomic ring constraint |r_i - r_j| = d0 (nm)."""
    i: int
    j: int
    d0: float


@dataclass
class M3VirtualSite:
    """
    Virtual site: r_vs = sum_p w_p r_p, with sum_p w_p = 1.
    Martini 3 uses these for planar ring moieties [1].
    """
    parents: List[int]
    weights: List[float]

    def __post_init__(self):
        if len(self.parents) != len(self.weights):
            raise ValueError("parents/weights length mismatch")
        s = sum(self.weights)
        if abs(s - 1.0) > 1e-8:
            raise ValueError(f"virtual-site weights must sum to 1, got {s}")


@dataclass
class M3Params:
    bead_types: Dict[str, M3BeadType] = field(default_factory=dict)
    bonds: List[M3Bond] = field(default_factory=list)
    angles: List[M3Angle] = field(default_factory=list)
    constraints: List[M3Constraint] = field(default_factory=list)
    vsites: List[M3VirtualSite] = field(default_factory=list)
    pair_matrix: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    lj_cutoff: float = 1.1
    coulomb_cutoff: float = 1.1
    epsilon_r: float = 15.0
    coulomb_style: str = "shift"      # "shift" | "reaction_field"
    epsilon_rf: float = 78.5
    exclude_bonded: bool = True
    all_placeholder: bool = True

    # ---------------- construction ----------------
    @classmethod
    def placeholder(cls) -> "M3Params":
        """
        Size classes are real; EVERY epsilon here is a PLACEHOLDER.
        Load official Martini 3 parameters before quoting any number.
        """
        types = {}
        for base, eps in (("P", 5.0), ("N", 4.0), ("C", 3.5), ("Q", 5.0)):
            for size in ("R", "S", "T"):
                nm = f"{base}{size}"
                types[nm] = M3BeadType(nm, size, eps, 0.0,
                                       72.0 if size == "R" else
                                       (54.0 if size == "S" else 36.0), True)
        return cls(bead_types=types, all_placeholder=True)

    # ---------------- I/O ----------------
    def to_json(self, path: str) -> None:
        d = {
            "bead_types": {k: asdict(v) for k, v in self.bead_types.items()},
            "bonds": [asdict(b) for b in self.bonds],
            "angles": [asdict(a) for a in self.angles],
            "constraints": [asdict(c) for c in self.constraints],
            "vsites": [asdict(v) for v in self.vsites],
            "pair_matrix": {k: list(v) for k, v in self.pair_matrix.items()},
            "lj_cutoff": self.lj_cutoff,
            "coulomb_cutoff": self.coulomb_cutoff,
            "epsilon_r": self.epsilon_r,
            "coulomb_style": self.coulomb_style,
            "epsilon_rf": self.epsilon_rf,
            "exclude_bonded": self.exclude_bonded,
            "all_placeholder": self.all_placeholder,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "M3Params":
        with open(path) as f:
            d = json.load(f)
        p = cls()
        p.bead_types = {k: M3BeadType(**v) for k, v in d["bead_types"].items()}
        p.bonds = [M3Bond(**b) for b in d.get("bonds", [])]
        p.angles = [M3Angle(**a) for a in d.get("angles", [])]
        p.constraints = [M3Constraint(**c) for c in d.get("constraints", [])]
        p.vsites = [M3VirtualSite(**v) for v in d.get("vsites", [])]
        p.pair_matrix = {k: tuple(v) for k, v in d.get("pair_matrix", {}).items()}
        for k in ("lj_cutoff", "coulomb_cutoff", "epsilon_r", "coulomb_style",
                  "epsilon_rf", "exclude_bonded", "all_placeholder"):
            if k in d:
                setattr(p, k, d[k])
        return p

    # ---------------- pair lookup ----------------
    def pair(self, ti: str, tj: str) -> Tuple[float, float]:
        """
        Martini 3 uses an EXPLICIT interaction matrix [1]. Cross-size
        entries must come from that matrix. The fallback below scales
        sigma by size class and epsilon geometrically — it is NOT
        Martini and is flagged whenever it is used.
        """
        for key in (f"{ti}|{tj}", f"{tj}|{ti}"):
            if key in self.pair_matrix:
                return self.pair_matrix[key]
        a, b = self.bead_types[ti], self.bead_types[tj]
        sig = 0.5 * (a.sigma + b.sigma)
        eps = math.sqrt(max(a.epsilon * b.epsilon, 0.0))
        return sig, eps

    def uses_fallback(self, ti: str, tj: str) -> bool:
        return (f"{ti}|{tj}" not in self.pair_matrix
                and f"{tj}|{ti}" not in self.pair_matrix)


@dataclass
class M3Mapping:
    """Beads with atom indices, type names, charges, masses."""
    bead_atoms: List[List[int]]
    bead_types: List[str]
    charges: List[float]
    masses: List[float]

    @property
    def n_beads(self) -> int:
        return len(self.bead_atoms)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "M3Mapping":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def auto(cls, n_atoms: int, per_bead: int = 4,
             type_name: str = "CR") -> "M3Mapping":
        """Geometric heuristic. NOT a chemistry-aware Martini 3 map."""
        beads, types, q, m = [], [], [], []
        for s in range(0, n_atoms, per_bead):
            idx = list(range(s, min(s + per_bead, n_atoms)))
            beads.append(idx)
            types.append(type_name)
            q.append(0.0)
            m.append(72.0)
        return cls(beads, types, q, m)

    def map_positions(self, aa: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.n_beads, 3, dtype=aa.dtype, device=aa.device)
        for b, idx in enumerate(self.bead_atoms):
            if idx:
                out[b] = aa[idx].mean(0)
        return out

    @staticmethod
    def map_forces(aa_f: torch.Tensor, mapping: "M3Mapping") -> torch.Tensor:
        """Conjugate CG force for a COM map: F_b = sum_{a in b} F_a."""
        out = torch.zeros(mapping.n_beads, 3,
                          dtype=aa_f.dtype, device=aa_f.device)
        for b, idx in enumerate(mapping.bead_atoms):
            if idx:
                out[b] = aa_f[idx].sum(0)
        return out


# =============================================================================
# §2  RING CONSTRAINTS (SHAKE / RATTLE) + VIRTUAL SITES
# =============================================================================

class RingConstraints:
    """
    Holonomic distance constraints for ring moieties [M3-2].

    SHAKE  : positions, iterated to tolerance
    RATTLE : velocities, projected onto the constraint manifold

    Differentiability: SHAKE is an iterative solve. Autograd CAN
    differentiate through a fixed number of iterations, but the result
    is the derivative of the truncated iteration, not of the exact
    constrained map. For exact gradients use either (a) stiff springs
    (differentiable, approximate constraint) or (b) implicit
    differentiation of the constraint equations. Both options are
    exposed; neither is silently substituted for the other.
    """

    def __init__(self, constraints: List[M3Constraint],
                 masses: torch.Tensor, tol: float = 1e-8,
                 max_iter: int = 500):
        self.cons = constraints
        self.m = masses
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.last_iters = 0
        self.last_max_violation = 0.0

    def shake(self, x_new: torch.Tensor, x_old: torch.Tensor,
              dt: float) -> torch.Tensor:
        """
        Project x_new onto the constraint manifold, using x_old as the
        reference direction (standard SHAKE). Returns corrected positions.
        """
        if not self.cons:
            return x_new
        x = x_new.clone()
        inv_m = 1.0 / self.m.clamp_min(1e-30)
        for it in range(self.max_iter):
            max_viol = 0.0
            for c in self.cons:
                i, j, d0 = c.i, c.j, c.d0
                rij_old = x_old[i] - x_old[j]
                rij_new = x[i] - x[j]
                r2 = float((rij_new * rij_new).sum())
                viol = r2 - d0 * d0
                max_viol = max(max_viol, abs(viol))
                denom = 2.0 * float(rij_old @ rij_new) * (inv_m[i] + inv_m[j])
                if abs(denom) < 1e-30:
                    continue
                lam = viol / denom
                x[i] = x[i] - lam * inv_m[i] * rij_old
                j_ = x[j] + lam * inv_m[j] * rij_old
                x[j] = j_
            self.last_iters = it + 1
            self.last_max_violation = max_viol
            if max_viol < self.tol:
                break
        return x

    def rattle(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project velocities onto the constraint tangent space."""
        if not self.cons:
            return v
        v = v.clone()
        inv_m = 1.0 / self.m.clamp_min(1e-30)
        for _ in range(self.max_iter):
            max_viol = 0.0
            for c in self.cons:
                i, j = c.i, c.j
                rij = x[i] - x[j]
                vij = v[i] - v[j]
                num = float(rij @ vij)
                den = float(rij @ rij) * (inv_m[i] + inv_m[j])
                if abs(den) < 1e-30:
                    continue
                lam = num / den
                max_viol = max(max_viol, abs(num))
                v[i] = v[i] - lam * inv_m[i] * rij
                v[j] = v[j] + lam * inv_m[j] * rij
            if max_viol < self.tol:
                break
        return v

    def violation(self, x: torch.Tensor) -> float:
        if not self.cons:
            return 0.0
        return max(abs(float((x[c.i] - x[c.j]).norm()) - c.d0)
                   for c in self.cons)


class VirtualSites:
    """
    Martini 3 virtual sites [M3-3]: r_vs = sum_p w_p r_p.
    Force redistribution is the exact transpose: F_p += w_p F_vs.
    This conserves total force (sum w = 1) and total torque
    (because the site is an affine combination of parents).
    """

    def __init__(self, vsites: List[M3VirtualSite]):
        self.vs = vsites

    def build(self, x_real: torch.Tensor) -> torch.Tensor:
        if not self.vs:
            return x_real
        out = []
        for v in self.vs:
            acc = torch.zeros(3, dtype=x_real.dtype, device=x_real.device)
            for p, w in zip(v.parents, v.weights):
                acc = acc + w * x_real[p]
            out.append(acc)
        return torch.cat([x_real, torch.stack(out, 0)], 0)

    def redistribute(self, f_all: torch.Tensor, n_real: int) -> torch.Tensor:
        f = f_all[:n_real].clone()
        for k, v in enumerate(self.vs):
            fvs = f_all[n_real + k]
            for p, w in zip(v.parents, v.weights):
                f[p] = f[p] + w * fvs
        return f


# =============================================================================
# §3  CG FORCE FIELD WITH SIZE CLASSES
# =============================================================================

class M3ForceField:
    """
    Martini-3-compatible energy/forces with R/S/T size classes,
    ring constraints (as stiff springs OR via SHAKE in the integrator),
    and virtual sites.

    All parameters are placeholders until official ones are loaded.
    """

    def __init__(self, params: M3Params, mapping: M3Mapping,
                 constraint_mode: str = "stiff",
                 k_constraint: float = 5e4,
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        assert constraint_mode in ("stiff", "shake")
        self.p = params
        self.map = mapping
        self.mode = constraint_mode
        self.k_con = float(k_constraint)
        self.dev = torch.device(device)
        self.dt_ = dtype
        self.nb = mapping.n_beads
        self.vs = VirtualSites(params.vsites)
        self.mass = torch.tensor(mapping.masses, dtype=dtype, device=self.dev)
        self.charge = torch.tensor(mapping.charges, dtype=dtype, device=self.dev)
        self.tidx = [params.bead_types[t] for t in mapping.bead_types]
        self.rc = RingConstraints(params.constraints, self.mass) \
            if constraint_mode == "shake" else None
        self.n_fallback_pairs = 0
        self._build_pair_tables()

    def _build_pair_tables(self):
        n = self.nb
        S = torch.zeros(n, n, dtype=self.dt_, device=self.dev)
        E = torch.zeros(n, n, dtype=self.dt_, device=self.dev)
        for i in range(n):
            for j in range(n):
                ti, tj = self.tidx[i].name, self.tidx[j].name
                s, e = self.p.pair(ti, tj)
                S[i, j], E[i, j] = s, e
                if i < j and self.p.uses_fallback(ti, tj):
                    self.n_fallback_pairs += 1
        self.S, self.E = S, E
        excl = torch.zeros(n, n, dtype=torch.bool, device=self.dev)
        if self.p.exclude_bonded:
            for b in self.p.bonds:
                excl[b.i, b.j] = excl[b.j, b.i] = True
            for c in self.p.constraints:
                excl[c.i, c.j] = excl[c.j, c.i] = True
        self.excl = excl

    # ---------------- energy terms ----------------
    def bond_energy(self, x):
        if not self.p.bonds:
            return x.new_zeros(())
        e = x.new_zeros(())
        for b in self.p.bonds:
            r = (x[b.i] - x[b.j]).norm().clamp_min(1e-9)
            e = e + 0.5 * b.k * (r - b.r0) ** 2
        return e

    def angle_energy(self, x):
        if not self.p.angles:
            return x.new_zeros(())
        e = x.new_zeros(())
        for a in self.p.angles:
            v1, v2 = x[a.i] - x[a.j], x[a.k] - x[a.j]
            ct = (v1 @ v2) / (v1.norm().clamp_min(1e-9) * v2.norm().clamp_min(1e-9))
            ct = ct.clamp(-1 + 1e-7, 1 - 1e-7)
            e = e + 0.5 * a.k_angle * (ct - math.cos(math.radians(a.theta0))) ** 2
        return e

    def constraint_energy(self, x):
        """Stiff-spring surrogate when mode='stiff' (differentiable)."""
        if self.mode != "stiff" or not self.p.constraints:
            return x.new_zeros(())
        e = x.new_zeros(())
        for c in self.p.constraints:
            r = (x[c.i] - x[c.j]).norm().clamp_min(1e-9)
            e = e + 0.5 * self.k_con * (r - c.d0) ** 2
        return e

    def nonbonded_energy(self, x):
        n = x.shape[0]
        if n < 2:
            return x.new_zeros(())
        d = x.unsqueeze(1) - x.unsqueeze(0)
        r = d.norm(dim=-1).clamp_min(1e-6)
        mask = ~(torch.eye(n, dtype=torch.bool, device=x.device) | self.excl[:n, :n])
        S, E = self.S[:n, :n], self.E[:n, :n]
        sr6 = (S / r) ** 6
        lj = 4.0 * E * (sr6 ** 2 - sr6)
        src6 = (S / self.p.lj_cutoff) ** 6
        shift = 4.0 * E * (src6 ** 2 - src6)
        lj = torch.where(r < self.p.lj_cutoff, lj - shift, torch.zeros_like(lj))
        qq = self.charge[:n].unsqueeze(1) * self.charge[:n].unsqueeze(0)
        qq = qq * COULOMB / self.p.epsilon_r
        rc = self.p.coulomb_cutoff
        if self.p.coulomb_style == "reaction_field":
            erf = self.p.epsilon_rf
            krf = (erf - self.p.epsilon_r) / (2 * erf + self.p.epsilon_r) / rc ** 3
            crf = 3 * erf / (2 * erf + self.p.epsilon_r) / rc
            vc = qq * (1.0 / r + krf * r ** 2 - crf)
        else:
            vc = qq * (1.0 / r - 1.0 / rc)
        vc = torch.where(r < rc, vc, torch.zeros_like(vc))
        return ((lj + vc) * mask).sum()

    def energy(self, x_real):
        x = self.vs.build(x_real) if self.p.vsites else x_real
        return (self.bond_energy(x) + self.angle_energy(x)
                + self.constraint_energy(x) + self.nonbonded_energy(x))

    def forces(self, x_real):
        xx = x_real.detach().requires_grad_(True)
        E = self.energy(xx)
        g = torch.autograd.grad(E, xx)[0]
        return E.detach(), -g


# =============================================================================
# §4  REACTION COORDINATE + UMBRELLA BIAS
# =============================================================================

@dataclass
class ReactionCoordinate:
    """Distance between centres of mass of two bead/atom groups."""
    group_a: List[int]
    group_b: List[int]
    masses_a: Optional[List[float]] = None
    masses_b: Optional[List[float]] = None

    def value(self, x: torch.Tensor) -> torch.Tensor:
        a = self._com(x, self.group_a, self.masses_a)
        b = self._com(x, self.group_b, self.masses_b)
        return (a - b).norm().clamp_min(1e-9)

    @staticmethod
    def _com(x, idx, m):
        if m is None:
            return x[idx].mean(0)
        w = torch.tensor(m, dtype=x.dtype, device=x.device)
        w = w / w.sum()
        return (w.unsqueeze(-1) * x[idx]).sum(0)


@dataclass
class UmbrellaWindow:
    xi0: float                 # nm, window centre
    k: float                   # kJ/mol/nm^2
    samples: List[float] = field(default_factory=list)

    def bias(self, xi: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.k * (xi - self.xi0) ** 2


class CGUmbrellaMD:
    """Overdamped Langevin with an umbrella bias, for the CG backend."""

    def __init__(self, ff: M3ForceField, rc: ReactionCoordinate,
                 temperature: float = 300.0, dt: float = 0.005,
                 friction: float = 5.0):
        self.ff = ff
        self.rc = rc
        self.T = float(temperature)
        self.dt = float(dt)
        self.gamma = float(friction)

    def run(self, x0: torch.Tensor, window: UmbrellaWindow,
            n_steps: int, sample_every: int = 10,
            equilibrate: int = 500) -> UmbrellaWindow:
        x = x0.clone()
        inv_m = 1.0 / self.ff.mass.clamp_min(1e-30)
        for step in range(n_steps):
            xx = x.detach().requires_grad_(True)
            E = self.ff.energy(xx) + window.bias(self.rc.value(xx))
            f = -torch.autograd.grad(E, xx)[0]
            with torch.no_grad():
                drift = (self.dt / self.gamma) * f * inv_m.unsqueeze(-1)
                scale = torch.sqrt(2.0 * KB * self.T * self.dt / self.gamma)
                noise = scale * inv_m.sqrt().unsqueeze(-1) * torch.randn_like(x)
                x_new = x + drift + noise
                if self.ff.rc is not None:
                    x_new = self.ff.rc.shake(x_new, x, self.dt)
                x = x_new
            if step >= equilibrate and step % sample_every == 0:
                window.samples.append(float(self.rc.value(x)))
        return window


class AAUmbrellaMD:
    """OpenMM AA backend with a CustomCentroidBondForce umbrella bias."""

    def __init__(self, pdb_path: Optional[str] = None,
                 forcefield_files: Optional[List[str]] = None,
                 temperature: float = 300.0, friction: float = 1.0,
                 dt_ps: float = 0.002, platform: str = "CPU"):
        self.available = (HAS_OPENMM and pdb_path is not None
                          and os.path.exists(pdb_path))
        self.reason = ""
        if not HAS_OPENMM:
            self.reason = "OpenMM not installed"
        elif pdb_path is None:
            self.reason = "no PDB supplied"
        elif not os.path.exists(pdb_path):
            self.reason = f"PDB not found: {pdb_path}"
        self.T, self.friction, self.dt_ps = temperature, friction, dt_ps
        if not self.available:
            return
        ff_files = forcefield_files or ["amber14-all.xml", "implicit/gbn2.xml"]
        pdb = app.PDBFile(pdb_path)
        ff = app.ForceField(*[f for f in ff_files if f])
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        self.system = ff.createSystem(modeller.topology,
                                      nonbondedMethod=app.NoCutoff,
                                      constraints=app.HBonds)
        self.topology = modeller.topology
        integ = mm.LangevinMiddleIntegrator(temperature * unit.kelvin,
                                            friction / unit.picosecond,
                                            dt_ps * unit.picoseconds)
        try:
            plat = mm.Platform.getPlatformByName(platform)
        except Exception:
            plat = mm.Platform.getPlatformByName("Reference")
        self.sim = app.Simulation(modeller.topology, self.system, integ, plat)
        self.sim.context.setPositions(modeller.positions)
        self.n_atoms = self.system.getNumParticles()
        self._bias_force = None

    def set_umbrella(self, rc: ReactionCoordinate, window: UmbrellaWindow):
        if not self.available:
            raise RuntimeError(self.reason)
        if self._bias_force is not None:
            self.system.removeForce(self._bias_force_idx)
        f = mm.CustomCentroidBondForce(2, "0.5*k*(distance(g1,g2)-xi0)^2")
        f.addPerBondParameter("k")
        f.addPerBondParameter("xi0")
        g1 = f.addGroup(rc.group_a)
        g2 = f.addGroup(rc.group_b)
        f.addBond([g1, g2], [window.k, window.xi0])
        self._bias_force_idx = self.system.addForce(f)
        self._bias_force = f
        self.sim.context.reinitialize(preserveState=True)

    def run(self, window: UmbrellaWindow, n_steps: int,
            sample_every: int = 100, equilibrate: int = 5000,
            rc: Optional[ReactionCoordinate] = None) -> UmbrellaWindow:
        if not self.available:
            raise RuntimeError(self.reason)
        self.sim.step(equilibrate)
        for k in range(n_steps // sample_every):
            self.sim.step(sample_every)
            pos = self.sim.context.getState(getPositions=True).getPositions(
                asNumpy=True).value_in_unit(unit.nanometer)
            x = torch.tensor(pos, dtype=torch.float64)
            window.samples.append(float(rc.value(x)))
        return window


# =============================================================================
# §5  WHAM  [2]
# =============================================================================

def wham(windows: List[UmbrellaWindow], temperature: float = 300.0,
         n_bins: int = 100, xi_min: Optional[float] = None,
         xi_max: Optional[float] = None, tol: float = 1e-7,
         max_iter: int = 10000) -> Dict[str, Any]:
    """
    Self-consistent WHAM [2].

      p(xi) = sum_j h_j(xi) / sum_j n_j exp(-beta w_j(xi) + beta f_j)
      exp(-beta f_j) = sum_xi p(xi) exp(-beta w_j(xi))

    Returns PMF = -kT ln p, aligned so min(PMF)=0, plus convergence info.
    """
    kT = KB * temperature
    beta = 1.0 / kT
    data = [np.asarray(w.samples, dtype=np.float64) for w in windows]
    data = [d[np.isfinite(d)] for d in data]
    n_j = np.array([len(d) for d in data], dtype=np.float64)
    if n_j.sum() == 0:
        raise ValueError("no samples in any window")
    allx = np.concatenate([d for d in data if len(d)])
    lo = float(allx.min()) if xi_min is None else float(xi_min)
    hi = float(allx.max()) if xi_max is None else float(xi_max)
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    H = np.zeros((len(windows), n_bins), dtype=np.float64)
    for j, d in enumerate(data):
        if len(d):
            H[j] = np.histogram(d, bins=edges)[0].astype(np.float64)
    W = np.array([[0.5 * w.k * (x - w.xi0) ** 2 for x in centres]
                  for w in windows], dtype=np.float64)
    expW = np.exp(-beta * W)
    f = np.zeros(len(windows), dtype=np.float64)
    p = np.zeros(n_bins, dtype=np.float64)
    converged = False
    for it in range(max_iter):
        denom = (n_j[:, None] * expW * np.exp(beta * f)[:, None]).sum(0)
        denom = np.maximum(denom, 1e-300)
        p_new = H.sum(0) / denom
        s = p_new.sum()
        if s <= 0:
            break
        p_new = p_new / s
        f_new = -kT * np.log(np.maximum((p_new[None, :] * expW).sum(1), 1e-300))
        f_new -= f_new[0]
        delta = float(np.max(np.abs(f_new - f)))
        f, p = f_new, p_new
        if delta < tol:
            converged = True
            break
    pmf = -kT * np.log(np.maximum(p, 1e-300))
    pmf -= pmf.min()
    pmf[p <= 0] = np.nan
    return {"xi": centres, "pmf": pmf, "f": f, "p": p,
            "n_iter": it + 1, "converged": converged,
            "n_samples": int(n_j.sum()), "n_bins": n_bins}


def window_overlap(windows: List[UmbrellaWindow], n_bins: int = 80
                   ) -> Dict[str, Any]:
    """
    Adjacent-window histogram overlap. A rule of thumb is that neighbouring
    windows should share meaningful probability mass; near-zero overlap
    means the PMF is not reliably connected across that gap.
    """
    data = [np.asarray(w.samples, dtype=np.float64) for w in windows]
    data = [d[np.isfinite(d)] for d in data]
    if len(data) < 2:
        return {"overlaps": [], "min_overlap": 1.0, "n_gaps": 0}
    allx = np.concatenate([d for d in data if len(d)])
    edges = np.linspace(allx.min(), allx.max(), n_bins + 1)
    hs = []
    for d in data:
        h = np.histogram(d, bins=edges)[0].astype(np.float64) if len(d) \
            else np.zeros(n_bins)
        s = h.sum()
        hs.append(h / s if s > 0 else h)
    ov = []
    for a, b in zip(hs, hs[1:]):
        ov.append(float(np.minimum(a, b).sum()))
    return {"overlaps": ov, "min_overlap": float(min(ov)) if ov else 1.0,
            "n_gaps": int(sum(1 for o in ov if o < 0.05))}


def block_pmf(windows: List[UmbrellaWindow], temperature: float = 300.0,
              n_blocks: int = 5, n_bins: int = 100,
              xi_min: Optional[float] = None,
              xi_max: Optional[float] = None) -> Dict[str, Any]:
    """Split each window into blocks, re-run WHAM per block, report stderr."""
    base = wham(windows, temperature, n_bins, xi_min, xi_max)
    xi = base["xi"]
    pmfs = []
    for b in range(n_blocks):
        sub = []
        for w in windows:
            s = np.asarray(w.samples, dtype=np.float64)
            s = s[np.isfinite(s)]
            chunk = s[b::n_blocks]
            sub.append(UmbrellaWindow(w.xi0, w.k, chunk.tolist()))
        if all(len(x.samples) < 5 for x in sub):
            continue
        try:
            r = wham(sub, temperature, n_bins, xi_min, xi_max)
            pmfs.append(r["pmf"])
        except Exception:
            continue
    if len(pmfs) < 2:
        return {"xi": xi, "pmf": base["pmf"], "stderr": np.full_like(xi, np.nan),
                "n_blocks_used": len(pmfs), "base": base}
    A = np.vstack(pmfs)
    err = np.nanstd(A, axis=0, ddof=1) / math.sqrt(len(pmfs))
    return {"xi": xi, "pmf": base["pmf"], "stderr": err,
            "n_blocks_used": len(pmfs), "base": base}


# =============================================================================
# §6  AA vs CG PMF COMPARISON
# =============================================================================

def compare_pmf(xi_aa, pmf_aa, xi_cg, pmf_cg,
                bound_range: Optional[Tuple[float, float]] = None,
                err_aa=None, err_cg=None) -> Dict[str, Any]:
    """
    Interpolate both PMFs onto a common grid, align each to its own
    minimum, and report RMSE plus the bound/unbound free-energy
    difference if a bound range is supplied.
    """
    lo = max(np.nanmin(xi_aa), np.nanmin(xi_cg))
    hi = min(np.nanmax(xi_aa), np.nanmax(xi_cg))
    if hi <= lo:
        return {"status": "NO_OVERLAP"}
    grid = np.linspace(lo, hi, 200)
    a = np.interp(grid, xi_aa, pmf_aa)
    c = np.interp(grid, xi_cg, pmf_cg)
    a = a - np.nanmin(a)
    c = c - np.nanmin(c)
    diff = c - a
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    out = {"status": "COMPARED", "grid": grid, "pmf_aa": a, "pmf_cg": c,
           "rmse_kJmol": rmse,
           "max_abs_diff_kJmol": float(np.nanmax(np.abs(diff)))}
    if bound_range is not None:
        m = (grid >= bound_range[0]) & (grid <= bound_range[1])
        if m.sum() >= 2:
            # ΔG_bound = -kT ln( <exp(-β PMF)>_bound ) relative to min
            out["dG_bound_aa"] = float(np.nanmean(a[m]))
            out["dG_bound_cg"] = float(np.nanmean(c[m]))
            out["ddG_bound"] = out["dG_bound_cg"] - out["dG_bound_aa"]
    if err_aa is not None and err_cg is not None:
        ea = np.interp(grid, xi_aa, err_aa)
        ec = np.interp(grid, xi_cg, err_cg)
        out["combined_stderr"] = np.sqrt(ea ** 2 + ec ** 2)
        sig = np.abs(diff) > 2.0 * out["combined_stderr"]
        out["frac_significant_diff"] = float(np.mean(sig))
    return out


# =============================================================================
# §7  SELF-TEST  (runs without OpenMM)
# =============================================================================

def self_test() -> bool:
    print("=" * 92)
    print(f"  MARTINI 3 (S/T + rings) + PMF VALIDATION v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 92)
    torch.manual_seed(0)
    np.random.seed(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # T1 size classes
    print("\n[T1] Martini 3 size classes R / S / T")
    p = M3Params.placeholder()
    rep("sigma R = 0.47 nm", abs(p.bead_types["PR"].sigma - 0.47) < 1e-12)
    rep("sigma S = 0.41 nm", abs(p.bead_types["PS"].sigma - 0.41) < 1e-12)
    rep("sigma T = 0.34 nm", abs(p.bead_types["PT"].sigma - 0.34) < 1e-12)
    rep("placeholder flagged", p.all_placeholder)
    rep("fallback flagged when no pair entry", p.uses_fallback("PR", "PT"))
    p.pair_matrix["PR|PT"] = (0.44, 4.2)
    rep("explicit pair matrix overrides fallback",
        not p.uses_fallback("PR", "PT")
        and abs(p.pair("PR", "PT")[0] - 0.44) < 1e-12)

    # T2 ring constraints (SHAKE)
    print("\n[T2] Ring constraints — SHAKE / RATTLE")
    x0 = torch.tensor([[0.0, 0.0, 0.0],
                       [0.30, 0.0, 0.0],
                       [0.15, 0.26, 0.0]], dtype=torch.float64)
    cons = [M3Constraint(0, 1, 0.30), M3Constraint(1, 2, 0.30),
            M3Constraint(2, 0, 0.30)]
    mass = torch.full((3,), 72.0, dtype=torch.float64)
    rc = RingConstraints(cons, mass, tol=1e-10)
    x_pert = x0 + torch.randn(3, 3, dtype=torch.float64) * 0.02
    v_before = rc.violation(x_pert)
    x_fix = rc.shake(x_pert, x0, dt=0.005)
    v_after = rc.violation(x_fix)
    rep("SHAKE reduces constraint violation", v_after < v_before,
        f"{v_before:.2e} -> {v_after:.2e}")
    rep("SHAKE meets tolerance", v_after < 1e-6, f"viol={v_after:.2e}")
    v = torch.randn(3, 3, dtype=torch.float64)
    v_proj = rc.rattle(v, x_fix)
    dot = max(abs(float((x_fix[c.i] - x_fix[c.j]) @ (v_proj[c.i] - v_proj[c.j])))
              for c in cons)
    rep("RATTLE projects velocities onto tangent space", dot < 1e-8,
        f"max |r·v|={dot:.2e}")

    # T3 virtual sites conserve force and torque
    print("\n[T3] Virtual sites — force/torque conservation")
    vs = M3VirtualSite(parents=[0, 1, 2], weights=[1/3, 1/3, 1/3])
    VS = VirtualSites([vs])
    xr = torch.randn(3, 3, dtype=torch.float64)
    xall = VS.build(xr)
    rep("virtual site appended", xall.shape == (4, 3))
    rep("virtual site = affine combination",
        float((xall[3] - xr.mean(0)).abs().max()) < 1e-12)
    fall = torch.randn(4, 3, dtype=torch.float64)
    fred = VS.redistribute(fall, 3)
    rep("total force conserved",
        float((fred.sum(0) - fall.sum(0)).abs().max()) < 1e-12)
    tq_all = torch.linalg.cross(xall, fall).sum(0)
    tq_red = torch.linalg.cross(xr, fred).sum(0)
    rep("total torque conserved",
        float((tq_all - tq_red).abs().max()) < 1e-10)

    # T4 CG force field with S/T beads
    print("\n[T4] CG force field with mixed R/S/T beads")
    mapping = M3Mapping(bead_atoms=[[0], [1], [2], [3]],
                        bead_types=["CR", "CS", "CT", "CR"],
                        charges=[0.0, 0.0, 0.0, 0.0],
                        masses=[72.0, 54.0, 36.0, 72.0])
    pp = M3Params.placeholder()
    pp.bonds = [M3Bond(0, 1, 0.35, 1250.0), M3Bond(1, 2, 0.30, 1250.0)]
    pp.constraints = [M3Constraint(2, 3, 0.28)]
    ff = M3ForceField(pp, mapping, constraint_mode="stiff")
    x = torch.randn(4, 3, dtype=torch.float64) * 0.3 + 0.5
    xg = x.clone().requires_grad_(True)
    E = ff.energy(xg)
    g = torch.autograd.grad(E, xg)[0]
    rep("energy finite + autograd", bool(torch.isfinite(E)) and
        bool(torch.isfinite(g).all()), f"E={float(E):.3f} kJ/mol")
    rep("size classes give different sigma",
        abs(float(ff.S[0, 1]) - 0.5*(0.47+0.41)) < 1e-12)

    # T5 WHAM recovers a known PMF from synthetic umbrellas
    print("\n[T5] WHAM — recover a known PMF from synthetic umbrella data")
    T = 300.0
    kT = KB * T
    xig = np.linspace(0.2, 2.0, 400)
    true_pmf = 8.0 * (xig - 0.6) ** 2 + 2.0 * np.exp(-((xig - 1.4) / 0.15) ** 2)
    true_pmf -= true_pmf.min()
    centres = np.linspace(0.3, 1.9, 17)
    k_spring = 400.0
    wins = []
    rng = np.random.default_rng(0)
    for c in centres:
        w = 0.5 * k_spring * (xig - c) ** 2
        pb = np.exp(-(true_pmf + w) / kT)
        pb /= pb.sum()
        s = rng.choice(xig, size=4000, p=pb)
        wins.append(UmbrellaWindow(float(c), k_spring, s.tolist()))
    res = wham(wins, T, n_bins=120)
    m = (res["xi"] > 0.35) & (res["xi"] < 1.85)
    ref = np.interp(res["xi"][m], xig, true_pmf)
    ref = ref - ref.min()
    got = res["pmf"][m]
    got = got - np.nanmin(got)
    rmse = float(np.sqrt(np.nanmean((got - ref) ** 2)))
    rep("WHAM converged", res["converged"], f"iters={res['n_iter']}")
    rep("WHAM recovers known PMF (RMSE < 1.0 kJ/mol)", rmse < 1.0,
        f"RMSE={rmse:.3f} kJ/mol")

    # T6 overlap + block error
    print("\n[T6] Diagnostics — window overlap + block error")
    ov = window_overlap(wins)
    rep("adjacent windows overlap", ov["min_overlap"] > 0.05,
        f"min overlap={ov['min_overlap']:.3f}, gaps={ov['n_gaps']}")
    small = [UmbrellaWindow(w.xi0, w.k, w.samples[:1200]) for w in wins]
    bp = block_pmf(small, T, n_blocks=4, n_bins=80)
    rep("block stderr computed", bp["n_blocks_used"] >= 2,
        f"blocks={bp['n_blocks_used']}, "
        f"median stderr={np.nanmedian(bp['stderr']):.3f}")

    # T7 CG umbrella MD runs and samples
    print("\n[T7] CG umbrella MD")
    rc = ReactionCoordinate(group_a=[0], group_b=[3])
    md = CGUmbrellaMD(ff, rc, temperature=T, dt=0.002, friction=20.0)
    w0 = UmbrellaWindow(xi0=float(rc.value(x)), k=200.0)
    w0 = md.run(x, w0, n_steps=400, sample_every=10, equilibrate=100)
    rep("CG umbrella collected samples", len(w0.samples) > 10,
        f"n={len(w0.samples)}")
    rep("samples finite", all(math.isfinite(s) for s in w0.samples))

    # T8 comparison machinery
    print("\n[T8] AA-vs-CG comparison machinery")
    xi_a = np.linspace(0.3, 1.9, 60)
    pmf_a = 6.0 * (xi_a - 0.7) ** 2
    xi_c = np.linspace(0.3, 1.9, 55)
    pmf_c = 6.0 * (xi_c - 0.7) ** 2 + 1.5
    cmp_res = compare_pmf(xi_a, pmf_a, xi_c, pmf_c, bound_range=(0.5, 0.9))
    rep("comparison status COMPARED", cmp_res["status"] == "COMPARED")
    rep("constant offset removed by alignment",
        cmp_res["rmse_kJmol"] < 0.3, f"RMSE={cmp_res['rmse_kJmol']:.3f}")
    rep("ddG_bound reported", "ddG_bound" in cmp_res,
        f"ddG={cmp_res.get('ddG_bound', float('nan')):.3f} kJ/mol")

    # T9 OpenMM availability reported honestly
    print("\n[T9] AA backend availability")
    aa = AAUmbrellaMD(pdb_path=None)
    rep("AA unavailable reported, not faked",
        (not aa.available) and len(aa.reason) > 0, aa.reason)

    # T10 JSON round-trip
    print("\n[T10] Parameter I/O")
    import tempfile
    tmp = tempfile.mkdtemp()
    pj = os.path.join(tmp, "m3.json")
    mj = os.path.join(tmp, "map.json")
    pp.to_json(pj); mapping.to_json(mj)
    p2 = M3Params.from_json(pj); m2 = M3Mapping.from_json(mj)
    rep("params round-trip",
        len(p2.bonds) == len(pp.bonds)
        and len(p2.constraints) == len(pp.constraints))
    rep("mapping round-trip", m2.n_beads == mapping.n_beads)

    print("\n" + "=" * 92)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 92)
    print("""
  WHAT THESE TESTS PROVE
    R/S/T size classes carry the correct nominal sigma, and an explicit
      pair matrix overrides the (non-Martini) fallback.
    SHAKE satisfies ring constraints; RATTLE projects velocities.
    Virtual-site force redistribution conserves total force AND torque.
    WHAM recovers a KNOWN PMF from synthetic umbrella data (RMSE < 1 kT).
    Overlap and block-error diagnostics are computed, not assumed.
    The comparison machinery removes trivial offsets and reports ddG.

  WHAT THESE TESTS DO NOT PROVE
    No official Martini 3 parameters are bundled: every epsilon is a
      placeholder until you load the real itp.
    The self-test WHAM uses SYNTHETIC data. It validates the estimator,
      not your sampling.
    Nothing here establishes that your CG model reproduces AA
      thermodynamics. Only a real AA-vs-CG PMF run can do that.

  THE POINT OF THIS MODULE
    Force matching reproduces mean forces. It does NOT guarantee the
    PMF. Run the AA and CG umbrella sweeps, read the overlap and block
    error FIRST, and only then read the RMSE and ddG. A beautiful PMF
    with zero window overlap is not a result.
""")
    return ok


# =============================================================================
# §8  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=f"Martini3 + PMF v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--params", default=None, help="M3 params JSON")
    ap.add_argument("--mapping", default=None, help="M3 mapping JSON")
    ap.add_argument("--pdb", default=None, help="AA reference PDB")
    ap.add_argument("--windows", type=int, default=17)
    ap.add_argument("--xi-min", type=float, default=0.3)
    ap.add_argument("--xi-max", type=float, default=1.9)
    ap.add_argument("--k", type=float, default=400.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    params = M3Params.from_json(a.params) if a.params else M3Params.placeholder()
    if params.all_placeholder:
        print("WARNING: placeholder parameters. Load official Martini 3 itp.")
    if a.mapping is None or a.pdb is None:
        print("Both --mapping and --pdb are required for a real PMF run.")
        print("Running self-test instead so nothing is fabricated.")
        raise SystemExit(0 if self_test() else 1)

    mapping = M3Mapping.from_json(a.mapping)
    ff = M3ForceField(params, mapping, constraint_mode="shake")
    aa = AAUmbrellaMD(a.pdb, temperature=a.temperature)
    if not aa.available:
        print(f"AA backend unavailable: {aa.reason}")
        raise SystemExit(1)

    # groups: first and last bead/atom by default — override for your system
    rc_cg = ReactionCoordinate([0], [mapping.n_beads - 1])
    rc_aa = ReactionCoordinate([0], [aa.n_atoms - 1])
    centres = np.linspace(a.xi_min, a.xi_max, a.windows)

    cg_wins, aa_wins = [], []
    x0 = torch.zeros(mapping.n_beads, 3, dtype=torch.float64)
    x0[:, 0] = torch.linspace(0, a.xi_max, mapping.n_beads)
    md = CGUmbrellaMD(ff, rc_cg, a.temperature)
    for c in centres:
        w = UmbrellaWindow(float(c), a.k)
        cg_wins.append(md.run(x0, w, a.steps))
        wa = UmbrellaWindow(float(c), a.k)
        aa.set_umbrella(rc_aa, wa)
        aa_wins.append(aa.run(wa, a.steps, rc=rc_aa))
        print(f"  window xi0={c:.3f}  CG n={len(w cg_wins[-1].samples) if False else len(cg_wins[-1].samples)}"
              f"  AA n={len(aa_wins[-1].samples)}")

    ov_cg, ov_aa = window_overlap(cg_wins), window_overlap(aa_wins)
    print(f"  overlap: CG min={ov_cg['min_overlap']:.3f} gaps={ov_cg['n_gaps']} | "
          f"AA min={ov_aa['min_overlap']:.3f} gaps={ov_aa['n_gaps']}")
    bcg = block_pmf(cg_wins, a.temperature)
    baa = block_pmf(aa_wins, a.temperature)
    cmp_res = compare_pmf(baa["xi"], baa["pmf"], bcg["xi"], bcg["pmf"],
                          err_aa=baa["stderr"], err_cg=bcg["stderr"])
    print(f"  PMF RMSE = {cmp_res['rmse_kJmol']:.3f} kJ/mol")
    if "frac_significant_diff" in cmp_res:
        print(f"  fraction of grid with |diff| > 2*stderr = "
              f"{cmp_res['frac_significant_diff']:.2%}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"comparison": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                      for k, v in cmp_res.items()},
                       "overlap_cg": ov_cg, "overlap_aa": ov_aa}, f, indent=2)
        print(f"  json -> {a.out}")


if __name__ == "__main__":
    main()
