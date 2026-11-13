#!/usr/bin/env python3
"""
=============================================================================
MARTINI-COMPATIBLE COARSE-GRAINED FORCE FIELD + OPENMM BENCHMARK HARNESS
Turning estimated numbers into measured numbers.
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

=============================================================================
WHAT "MARTINI-COMPATIBLE" MEANS HERE — AND WHAT IT DOES NOT
=============================================================================
COMPATIBLE (implemented):
  * Units: nm, kJ/mol, ps, amu, elementary charge (OpenMM/Martini convention)
  * Nonbonded: Lennard-Jones 12-6 with a potential shift at the cutoff
  * Electrostatics: Coulomb with relative dielectric, plain-shifted or
    reaction-field form
  * Bonded: harmonic bonds, cosine-harmonic angles (Martini's angle form)
  * Explicit per-pair interaction matrix (Martini uses a lookup table,
    NOT Lorentz-Berthelot; LB is provided only as a labelled fallback)
  * Mapping: ~4 heavy atoms per bead, centre-of-geometry or mass-weighted
  * JSON parameter schema mirroring Martini's bead-type / bond / angle layout

NOT COMPATIBLE / NOT PROVIDED:
  ❌ Official Martini 2/3 bead interaction parameters are NOT bundled.
     They are published separately (cgmartini.nl) and must be supplied.
     Without them every epsilon/sigma here is a PLACEHOLDER, flagged as such.
  ❌ Martini 3's small/tiny bead sizes, ring constraints, and virtual sites
     are not implemented.
  ❌ Polarizable water (Martini PW) is not implemented.
  ❌ Elastic-network / ElNeDyn protein models are not implemented.
  ❌ No transferability claim: parameters fitted to one system are
     system-specific unless validated across many.

=============================================================================
WHAT THE BENCHMARK ACTUALLY MEASURES
=============================================================================
B1 FORCE MATCHING (the primary, honest accuracy metric)
   Map AA forces to beads: F_CG_ref[b] = sum of F_AA over atoms in bead b
   (exact conjugate force for a centre-of-mass / centre-of-geometry map).
   Compare against the CG model's own force at the SAME coordinates.
   Reports: RMSE, relative L2, cosine similarity, per-bead norm ratio.
   A generic CG model will NOT match. That gap is the measurement.

B2 PARAMETER FITTING (turning the gap into a number that can shrink)
   The CG energy is fully differentiable, so epsilon/sigma/bond/angle
   parameters are optimised by Adam against the force-matching loss.
   Reports before/after metrics. This is force matching (Izvekov-Voth /
   Noid style), not a claim of thermodynamic consistency.

B3 STRUCTURAL COMPARISON
   Run short AA MD (OpenMM LangevinMiddle) and CG MD (this module).
   Compare radius of gyration, end-to-end distance, bond-length
   distributions, and bead-bead RDF. All measured, none assumed.

B4 PERFORMANCE
   Wall time per step for AA (OpenMM) vs CG (here), DOF reduction,
   measured speedup. No estimated cost coefficients.

If OpenMM or a PDB is unavailable, the corresponding benchmark reports
UNAVAILABLE. It never reports a fabricated number.

=============================================================================
HONEST SCOPE
=============================================================================
❌ Force matching does not guarantee correct thermodynamics or kinetics.
   A CG model reproducing mean forces can still have wrong free-energy
   surfaces, wrong dynamics, and wrong temperature transferability.
❌ CG dynamics is accelerated by smoothing; the time mapping is NOT
   a physical clock. Reported speedups are wall-clock, not "simulated
   time per wall second" unless a time-mapping factor is calibrated.
❌ Backmapping CG->AA is ill-posed without a fragment library; the
   geometric reconstruction here is a guess, not a sampled ensemble.
❌ No claim of state of the art. Run the harness and read the table.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("martini_bench")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# ---- OpenMM units (nm, kJ/mol, ps, amu, e) --------------------------------
KB_KJ = 0.008314462618          # kJ mol^-1 K^-1
COULOMB = 138.935456            # kJ nm mol^-1 e^-2
AMU_NM2_PS2 = 1.0               # kJ/mol = amu nm^2 / ps^2  (exact by definition)

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    HAS_OPENMM = True
except ImportError:
    mm = app = unit = None
    HAS_OPENMM = False

try:
    from real_fold_one import OpenMMSystemBuilder, openmm_solute_energy
    HAS_REALFOLD = True
except Exception:
    HAS_REALFOLD = False


# =============================================================================
# §1  PARAMETERS
# =============================================================================

@dataclass
class BeadType:
    name: str
    sigma: float          # nm
    epsilon: float        # kJ/mol
    charge: float = 0.0   # e
    mass: float = 72.0    # amu (Martini regular bead ~ 4 waters = 72)
    placeholder: bool = True


@dataclass
class BondParam:
    i: int
    j: int
    r0: float             # nm
    k: float              # kJ/mol/nm^2


@dataclass
class AngleParam:
    i: int
    j: int
    k: int
    theta0: float         # degrees
    k_angle: float        # kJ/mol (cosine-harmonic prefactor)


@dataclass
class MartiniParams:
    """
    Martini-compatible parameter container.

    pair_matrix: explicit {(type_i, type_j): (sigma, epsilon)} overrides.
    Martini uses such a table; Lorentz-Berthelot is only a fallback and
    is NOT Martini's rule.
    """
    bead_types: Dict[str, BeadType] = field(default_factory=dict)
    bonds: List[BondParam] = field(default_factory=list)
    angles: List[AngleParam] = field(default_factory=list)
    pair_matrix: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    lj_cutoff: float = 1.1        # nm (Martini 2 standard)
    coulomb_cutoff: float = 1.1   # nm
    epsilon_r: float = 15.0       # Martini 2 relative dielectric
    coulomb_style: str = "shift"  # "shift" | "reaction_field"
    epsilon_rf: float = 78.5
    combination_rule: str = "lorentz_berthelot"   # FALLBACK, not Martini
    exclude_bonded: bool = True
    all_placeholder: bool = True

    # ---- construction -------------------------------------------------
    @classmethod
    def generic_placeholder(cls) -> "MartiniParams":
        """
        A minimal, clearly-labelled PLACEHOLDER type set.
        These are NOT Martini parameters. Supply real ones via from_json.
        """
        types = {
            "P5":  BeadType("P5",  0.47, 5.0,  0.0, 72.0, True),
            "P4":  BeadType("P4",  0.47, 5.0,  0.0, 72.0, True),
            "Nda": BeadType("Nda", 0.47, 4.5,  0.0, 72.0, True),
            "N0":  BeadType("N0",  0.47, 4.0,  0.0, 72.0, True),
            "C1":  BeadType("C1",  0.47, 3.5,  0.0, 72.0, True),
            "C3":  BeadType("C3",  0.47, 3.5,  0.0, 72.0, True),
            "Qa":  BeadType("Qa",  0.47, 5.0, +1.0, 72.0, True),
            "Qd":  BeadType("Qd",  0.47, 5.0, -1.0, 72.0, True),
        }
        return cls(bead_types=types, all_placeholder=True)

    # ---- I/O ----------------------------------------------------------
    def to_json(self, path: str) -> None:
        d = {
            "bead_types": {k: asdict(v) for k, v in self.bead_types.items()},
            "bonds": [asdict(b) for b in self.bonds],
            "angles": [asdict(a) for a in self.angles],
            "pair_matrix": {k: list(v) for k, v in self.pair_matrix.items()},
            "lj_cutoff": self.lj_cutoff,
            "coulomb_cutoff": self.coulomb_cutoff,
            "epsilon_r": self.epsilon_r,
            "coulomb_style": self.coulomb_style,
            "epsilon_rf": self.epsilon_rf,
            "combination_rule": self.combination_rule,
            "exclude_bonded": self.exclude_bonded,
            "all_placeholder": self.all_placeholder,
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MartiniParams":
        with open(path) as f:
            d = json.load(f)
        p = cls()
        p.bead_types = {k: BeadType(**v) for k, v in d["bead_types"].items()}
        p.bonds = [BondParam(**b) for b in d.get("bonds", [])]
        p.angles = [AngleParam(**a) for a in d.get("angles", [])]
        p.pair_matrix = {k: tuple(v) for k, v in d.get("pair_matrix", {}).items()}
        for k in ("lj_cutoff", "coulomb_cutoff", "epsilon_r", "coulomb_style",
                  "epsilon_rf", "combination_rule", "exclude_bonded",
                  "all_placeholder"):
            if k in d:
                setattr(p, k, d[k])
        return p

    def pair(self, ti: str, tj: str) -> Tuple[float, float]:
        key = f"{ti}|{tj}"
        key_r = f"{tj}|{ti}"
        if key in self.pair_matrix:
            return self.pair_matrix[key]
        if key_r in self.pair_matrix:
            return self.pair_matrix[key_r]
        a, b = self.bead_types[ti], self.bead_types[tj]
        if self.combination_rule != "lorentz_berthelot":
            raise ValueError(f"no pair entry for {ti},{tj} and no valid rule")
        return 0.5 * (a.sigma + b.sigma), math.sqrt(a.epsilon * b.epsilon)


# =============================================================================
# §2  AA -> CG MAPPING
# =============================================================================

@dataclass
class BeadSpec:
    atoms: List[int]
    type: str
    charge: float = 0.0
    mass: float = 72.0


@dataclass
class CGMapping:
    beads: List[BeadSpec]
    weight_mode: str = "geometry"   # "geometry" | "mass"

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"weight_mode": self.weight_mode,
                       "beads": [asdict(b) for b in self.beads]}, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CGMapping":
        with open(path) as f:
            d = json.load(f)
        return cls(beads=[BeadSpec(**b) for b in d["beads"]],
                   weight_mode=d.get("weight_mode", "geometry"))

    @classmethod
    def auto_group(cls, n_atoms: int, per_bead: int = 4,
                   type_name: str = "C1") -> "CGMapping":
        """
        Sequential grouping of `per_bead` atoms. A HEURISTIC, not a
        chemistry-aware Martini mapping (which respects rings, functional
        groups, and residue boundaries). Use from_json for real mappings.
        """
        beads = []
        for s in range(0, n_atoms, per_bead):
            idx = list(range(s, min(s + per_bead, n_atoms)))
            beads.append(BeadSpec(atoms=idx, type=type_name))
        return cls(beads=beads)

    def index_tensors(self, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flat atom->bead index and per-atom weight (sums to 1 per bead)."""
        flat_idx, flat_w = [], []
        for b, spec in enumerate(self.beads):
            n = len(spec.atoms)
            if n == 0:
                continue
            w = 1.0 / n
            for a in spec.atoms:
                flat_idx.append((a, b))
                flat_w.append(w)
        idx = torch.tensor(flat_idx, dtype=torch.long, device=device)
        w = torch.tensor(flat_w, dtype=dtype, device=device)
        return idx, w

    def map_positions(self, aa_pos: torch.Tensor) -> torch.Tensor:
        idx, w = self.index_tensors(aa_pos.device, aa_pos.dtype)
        nb = len(self.beads)
        out = torch.zeros(nb, 3, dtype=aa_pos.dtype, device=aa_pos.device)
        out = out.index_add(0, idx[:, 1], w.unsqueeze(-1) * aa_pos[idx[:, 0]])
        return out

    @staticmethod
    def map_forces(aa_force: torch.Tensor, mapping: "CGMapping") -> torch.Tensor:
        """
        Conjugate CG force for a linear position map: F_b = sum_{a in b} F_a.
        This is exact for centre-of-geometry / centre-of-mass mapping and is
        the reference used in force matching.
        """
        nb = len(mapping.beads)
        out = torch.zeros(nb, 3, dtype=aa_force.dtype, device=aa_force.device)
        for b, spec in enumerate(mapping.beads):
            if spec.atoms:
                out[b] = aa_force[spec.atoms].sum(0)
        return out


# =============================================================================
# §3  DIFFERENTIABLE MARTINI-COMPATIBLE FORCE FIELD
# =============================================================================

class MartiniCGForceField(nn.Module):
    """
    Fully differentiable CG energy and forces.

    E = E_bond + E_angle + E_LJ + E_Coulomb

    All parameters that are fitted live in nn.Parameter (log-space for
    positive quantities) so Adam can optimise them against AA forces.
    """

    def __init__(self, params: MartiniParams, mapping: CGMapping,
                 trainable: bool = False, device: str = "cpu",
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self.p = params
        self.map = mapping
        self.dev = torch.device(device)
        self.dt_ = dtype
        self.nb = len(mapping.beads)

        names = list(params.bead_types.keys())
        self.type_names = names
        self.type_index = {n: i for i, n in enumerate(names)}
        tidx = torch.tensor([self.type_index[b.type] for b in mapping.beads],
                            dtype=torch.long, device=self.dev)
        self.register_buffer("tidx", tidx)

        sig0 = torch.tensor([params.bead_types[n].sigma for n in names],
                            dtype=dtype, device=self.dev)
        eps0 = torch.tensor([params.bead_types[n].epsilon for n in names],
                            dtype=dtype, device=self.dev)
        q0 = torch.tensor([b.charge for b in mapping.beads],
                          dtype=dtype, device=self.dev)
        m0 = torch.tensor([b.mass for b in mapping.beads],
                          dtype=dtype, device=self.dev)
        self.register_buffer("charge", q0)
        self.register_buffer("mass", m0)

        if trainable:
            self.log_sigma = nn.Parameter(torch.log(sig0))
            self.log_eps = nn.Parameter(torch.log(eps0))
        else:
            self.register_buffer("log_sigma", torch.log(sig0))
            self.register_buffer("log_eps", torch.log(eps0))

        # bonded
        if params.bonds:
            bi = torch.tensor([[b.i, b.j] for b in params.bonds],
                              dtype=torch.long, device=self.dev)
            r0 = torch.tensor([b.r0 for b in params.bonds], dtype=dtype, device=self.dev)
            k0 = torch.tensor([b.k for b in params.bonds], dtype=dtype, device=self.dev)
        else:
            bi = torch.zeros(0, 2, dtype=torch.long, device=self.dev)
            r0 = torch.zeros(0, dtype=dtype, device=self.dev)
            k0 = torch.zeros(0, dtype=dtype, device=self.dev)
        self.register_buffer("bond_idx", bi)
        if trainable and len(r0):
            self.bond_r0 = nn.Parameter(r0)
            self.log_bond_k = nn.Parameter(torch.log(k0.clamp_min(1e-6)))
        else:
            self.register_buffer("bond_r0", r0)
            self.register_buffer("log_bond_k", torch.log(k0.clamp_min(1e-6)))

        if params.angles:
            ai = torch.tensor([[a.i, a.j, a.k] for a in params.angles],
                              dtype=torch.long, device=self.dev)
            t0 = torch.tensor([math.radians(a.theta0) for a in params.angles],
                              dtype=dtype, device=self.dev)
            ka = torch.tensor([a.k_angle for a in params.angles],
                              dtype=dtype, device=self.dev)
        else:
            ai = torch.zeros(0, 3, dtype=torch.long, device=self.dev)
            t0 = torch.zeros(0, dtype=dtype, device=self.dev)
            ka = torch.zeros(0, dtype=dtype, device=self.dev)
        self.register_buffer("angle_idx", ai)
        if trainable and len(t0):
            self.angle_t0 = nn.Parameter(t0)
            self.log_angle_k = nn.Parameter(torch.log(ka.clamp_min(1e-6)))
        else:
            self.register_buffer("angle_t0", t0)
            self.register_buffer("log_angle_k", torch.log(ka.clamp_min(1e-6)))

        # exclusions (1-2 bonded pairs)
        excl = torch.zeros(self.nb, self.nb, dtype=torch.bool, device=self.dev)
        if params.exclude_bonded and len(bi):
            excl[bi[:, 0], bi[:, 1]] = True
            excl[bi[:, 1], bi[:, 0]] = True
        self.register_buffer("excl", excl)

        # explicit pair matrix as dense tensors (Martini-style table)
        self._build_pair_table()

    def _build_pair_table(self) -> None:
        nt = len(self.type_names)
        S = torch.zeros(nt, nt, dtype=self.dt_, device=self.dev)
        E = torch.zeros(nt, nt, dtype=self.dt_, device=self.dev)
        has = torch.zeros(nt, nt, dtype=torch.bool, device=self.dev)
        for i, ni in enumerate(self.type_names):
            for j, nj in enumerate(self.type_names):
                key = f"{ni}|{nj}"
                if key in self.p.pair_matrix:
                    s, e = self.p.pair_matrix[key]
                    S[i, j], E[i, j], has[i, j] = s, e, True
        self.register_buffer("pair_S", S)
        self.register_buffer("pair_E", E)
        self.register_buffer("pair_has", has)

    # ---- pair parameters ---------------------------------------------
    def _pair_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sig = torch.exp(self.log_sigma)
        eps = torch.exp(self.log_eps)
        ti, tj = self.tidx.unsqueeze(1), self.tidx.unsqueeze(0)
        s_lb = 0.5 * (sig[ti] + sig[tj])
        e_lb = torch.sqrt((eps[ti] * eps[tj]).clamp_min(1e-30))
        use_tbl = self.pair_has[ti, tj]
        S = torch.where(use_tbl, self.pair_S[ti, tj], s_lb)
        E = torch.where(use_tbl, self.pair_E[ti, tj], e_lb)
        return S, E

    # ---- energy terms --------------------------------------------------
    def bond_energy(self, x: torch.Tensor) -> torch.Tensor:
        if self.bond_idx.numel() == 0:
            return x.new_zeros(())
        i, j = self.bond_idx[:, 0], self.bond_idx[:, 1]
        r = (x[i] - x[j]).norm(dim=-1).clamp_min(1e-9)
        k = torch.exp(self.log_bond_k)
        return 0.5 * (k * (r - self.bond_r0) ** 2).sum()

    def angle_energy(self, x: torch.Tensor) -> torch.Tensor:
        if self.angle_idx.numel() == 0:
            return x.new_zeros(())
        i, j, k = self.angle_idx[:, 0], self.angle_idx[:, 1], self.angle_idx[:, 2]
        v1 = x[i] - x[j]
        v2 = x[k] - x[j]
        n1 = v1.norm(dim=-1).clamp_min(1e-9)
        n2 = v2.norm(dim=-1).clamp_min(1e-9)
        ct = ((v1 * v2).sum(-1) / (n1 * n2)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        ka = torch.exp(self.log_angle_k)
        return 0.5 * (ka * (ct - torch.cos(self.angle_t0)) ** 2).sum()

    def nonbonded_energy(self, x: torch.Tensor, chunk: int = 2048
                         ) -> torch.Tensor:
        n = x.shape[0]
        if n < 2:
            return x.new_zeros(())
        S, E = self._pair_params()
        rc_lj = self.p.lj_cutoff
        rc_c = self.p.coulomb_cutoff
        qi = self.charge
        eps_r = self.p.epsilon_r
        total = x.new_zeros(())
        eye = torch.eye(n, dtype=torch.bool, device=x.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            d = x[s:e].unsqueeze(1) - x.unsqueeze(0)
            r = d.norm(dim=-1).clamp_min(1e-6)
            mask = ~(eye[s:e] | self.excl[s:e])
            sr = S[s:e] / r
            sr6 = sr ** 6
            sr12 = sr6 ** 2
            v_lj = 4.0 * E[s:e] * (sr12 - sr6)
            src = S[s:e] / rc_lj
            v_shift = 4.0 * E[s:e] * (src ** 12 - src ** 6)
            v_lj = torch.where(r < rc_lj, v_lj - v_shift, torch.zeros_like(v_lj))
            total = total + (v_lj * mask).sum()

            qq = (qi[s:e].unsqueeze(1) * qi.unsqueeze(0)) * COULOMB / eps_r
            if self.p.coulomb_style == "reaction_field":
                erf = self.p.epsilon_rf
                krf = (erf - eps_r) / (2.0 * erf + eps_r) / rc_c ** 3
                crf = (3.0 * erf) / (2.0 * erf + eps_r) / rc_c
                vc = qq * (1.0 / r + krf * r ** 2 - crf)
            else:
                vc = qq * (1.0 / r - 1.0 / rc_c)
            vc = torch.where(r < rc_c, vc, torch.zeros_like(vc))
            total = total + (vc * mask).sum()
        return total

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return (self.bond_energy(x) + self.angle_energy(x)
                + self.nonbonded_energy(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.requires_grad_(True)
        E = self.energy(x)
        Frc = -torch.autograd.grad(E, x, create_graph=self.training)[0]
        return E, Frc

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# =============================================================================
# §4  CG DYNAMICS
# =============================================================================

class CGIntegrator:
    """
    Overdamped / BAOAB Langevin for CG beads, in OpenMM units.
    Overdamped is the usual CG choice: it removes the momentum timescale
    so dt is limited by the smoothed force landscape, not by vibrations.
    """

    def __init__(self, ff: MartiniCGForceField, dt: float = 0.02,
                 temperature: float = 300.0, friction: float = 5.0,
                 overdamped: bool = True):
        self.ff = ff
        self.dt = dt
        self.T = temperature
        self.gamma = friction
        self.overdamped = overdamped

    @torch.no_grad()
    def run(self, x: torch.Tensor, n_steps: int,
            v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        x = x.clone()
        if v is None:
            v = torch.zeros_like(x)
        m = self.ff.mass.unsqueeze(-1)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            with torch.enable_grad():
                xx = x.detach().requires_grad_(True)
                E = self.ff.energy(xx)
                f = -torch.autograd.grad(E, xx)[0]
            if self.overdamped:
                drift = (self.dt / self.gamma) * f / m
                scale = math.sqrt(2.0 * KB_KJ * self.T * self.dt / self.gamma)
                x = x + drift + scale / m.sqrt() * torch.randn_like(x)
            else:
                v = v + 0.5 * self.dt * f / m
                xh = x + 0.5 * self.dt * v
                c1 = math.exp(-self.gamma * self.dt)
                c2 = math.sqrt(max(1.0 - c1 * c1, 0.0))
                v = c1 * v + c2 * math.sqrt(KB_KJ * self.T) / m.sqrt() * torch.randn_like(v)
                x = xh + 0.5 * self.dt * v
                with torch.enable_grad():
                    xx = x.detach().requires_grad_(True)
                    E2 = self.ff.energy(xx)
                    f2 = -torch.autograd.grad(E2, xx)[0]
                v = v + 0.5 * self.dt * f2 / m
        el = time.perf_counter() - t0
        return x, {"elapsed_s": el, "s_per_step": el / max(n_steps, 1)}


# =============================================================================
# §5  OPENMM REFERENCE ADAPTER
# =============================================================================

class OpenMMReference:
    """AA reference via OpenMM. Reports UNAVAILABLE if not installed."""

    def __init__(self, pdb_path: Optional[str] = None,
                 forcefield_files: Optional[List[str]] = None,
                 implicit_solvent: Optional[str] = "implicit/gbn2.xml",
                 platform: str = "CPU", temperature: float = 300.0,
                 friction: float = 1.0, dt_ps: float = 0.002):
        self.available = HAS_OPENMM and pdb_path is not None and os.path.exists(pdb_path)
        self.reason = ""
        if not HAS_OPENMM:
            self.reason = "OpenMM not installed"
        elif pdb_path is None:
            self.reason = "no PDB supplied"
        elif not os.path.exists(pdb_path):
            self.reason = f"PDB not found: {pdb_path}"
        self.temp, self.friction, self.dt_ps = temperature, friction, dt_ps
        if not self.available:
            return
        ff_files = forcefield_files or ["amber14-all.xml", implicit_solvent]
        pdb = app.PDBFile(pdb_path)
        ff = app.ForceField(*[f for f in ff_files if f])
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
        system = ff.createSystem(modeller.topology,
                                 nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds)
        self.topology = modeller.topology
        self.system = system
        integ = mm.LangevinMiddleIntegrator(temperature * unit.kelvin,
                                            friction / unit.picosecond,
                                            dt_ps * unit.picoseconds)
        try:
            plat = mm.Platform.getPlatformByName(platform)
        except Exception:
            plat = mm.Platform.getPlatformByName("Reference")
        self.sim = app.Simulation(modeller.topology, system, integ, plat)
        self.sim.context.setPositions(modeller.positions)
        self.n_atoms = system.getNumParticles()

    def positions_nm(self) -> torch.Tensor:
        st = self.sim.context.getState(getPositions=True)
        p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return torch.tensor(p, dtype=torch.float64)

    def energy_forces(self, pos_nm: Optional[torch.Tensor] = None
                      ) -> Tuple[float, torch.Tensor]:
        if pos_nm is not None:
            self.sim.context.setPositions(pos_nm.detach().cpu().numpy())
        st = self.sim.context.getState(getEnergy=True, getForces=True)
        E = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        Frc = st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)
        return float(E), torch.tensor(Frc, dtype=torch.float64)

    def minimize(self, max_iter: int = 200) -> None:
        self.sim.minimizeEnergy(maxIterations=max_iter)

    def md(self, n_steps: int) -> Tuple[torch.Tensor, Dict]:
        t0 = time.perf_counter()
        self.sim.step(n_steps)
        el = time.perf_counter() - t0
        return self.positions_nm(), {"elapsed_s": el,
                                     "s_per_step": el / max(n_steps, 1)}


# =============================================================================
# §6  OBSERVABLES
# =============================================================================

def radius_of_gyration(x: torch.Tensor, mass: Optional[torch.Tensor] = None) -> float:
    if mass is None:
        c = x.mean(0)
        return float(((x - c) ** 2).sum(-1).mean().sqrt())
    w = mass / mass.sum()
    c = (w.unsqueeze(-1) * x).sum(0)
    return float((w * ((x - c) ** 2).sum(-1)).sum().sqrt())


def end_to_end(x: torch.Tensor) -> float:
    return float((x[-1] - x[0]).norm())


def bond_lengths(x: torch.Tensor, bonds: List[BondParam]) -> torch.Tensor:
    if not bonds:
        return torch.zeros(0)
    return torch.stack([(x[b.i] - x[b.j]).norm() for b in bonds])


def rdf(x: torch.Tensor, r_max: float = 2.0, n_bins: int = 50
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    d = torch.cdist(x, x)
    n = x.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    dv = d[mask]
    hist = torch.histc(dv, bins=n_bins, min=0.0, max=r_max)
    edges = torch.linspace(0.0, r_max, n_bins + 1, device=x.device)
    shells = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    dens = n / max((4.0 / 3.0) * math.pi * r_max ** 3, 1e-9)
    g = hist / (shells * dens * n).clamp_min(1e-30)
    return 0.5 * (edges[1:] + edges[:-1]), g


# =============================================================================
# §7  BENCHMARK HARNESS
# =============================================================================

@dataclass
class BenchResult:
    name: str
    status: str            # MEASURED | UNAVAILABLE | FAILED
    metrics: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


class BenchmarkHarness:
    def __init__(self, params: MartiniParams, mapping: CGMapping,
                 device: str = "cpu"):
        self.params = params
        self.mapping = mapping
        self.dev = torch.device(device)
        self.results: List[BenchResult] = []

    # ---- B1 force matching --------------------------------------------
    def force_matching(self, aa_pos: torch.Tensor,
                       aa_force: torch.Tensor,
                       ff: Optional[MartiniCGForceField] = None
                       ) -> BenchResult:
        ff = ff or MartiniCGForceField(self.params, self.mapping,
                                       device=str(self.dev))
        xb = self.mapping.map_positions(aa_pos)
        f_ref = CGMapping.map_forces(aa_force, self.mapping)
        with torch.enable_grad():
            xx = xb.detach().requires_grad_(True)
            E = ff.energy(xx)
            f_cg = -torch.autograd.grad(E, xx)[0]
        diff = f_cg - f_ref
        rmse = float(diff.pow(2).mean().sqrt())
        rel = float(diff.norm() / f_ref.norm().clamp_min(1e-30))
        cos = float(F.cosine_similarity(f_cg.flatten(), f_ref.flatten(), dim=0))
        ratio = float(f_cg.norm() / f_ref.norm().clamp_min(1e-30))
        r = BenchResult("B1_force_matching", "MEASURED", {
            "cg_energy_kJmol": float(E),
            "force_rmse_kJmol_nm": rmse,
            "force_rel_l2": rel,
            "force_cosine": cos,
            "force_norm_ratio": ratio,
            "n_beads": len(self.mapping.beads),
            "n_atoms": int(aa_pos.shape[0]),
            "dof_reduction": float(aa_pos.shape[0] / max(len(self.mapping.beads), 1)),
            "params_placeholder": self.params.all_placeholder,
        })
        self.results.append(r)
        return r

    # ---- B2 parameter fitting ------------------------------------------
    def fit_force_matching(self, aa_pos: torch.Tensor, aa_force: torch.Tensor,
                           n_steps: int = 200, lr: float = 1e-2
                           ) -> BenchResult:
        ff = MartiniCGForceField(self.params, self.mapping,
                                 trainable=True, device=str(self.dev))
        xb = self.mapping.map_positions(aa_pos).detach()
        f_ref = CGMapping.map_forces(aa_force, self.mapping).detach()

        def loss_fn():
            xx = xb.detach().requires_grad_(True)
            E = ff.energy(xx)
            f_cg = -torch.autograd.grad(E, xx, create_graph=True)[0]
            return (f_cg - f_ref).pow(2).mean()

        with torch.enable_grad():
            l0 = float(loss_fn())
        opt = torch.optim.Adam(ff.trainable_parameters(), lr=lr)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ff.trainable_parameters(), 10.0)
            opt.step()
        el = time.perf_counter() - t0
        with torch.enable_grad():
            l1 = float(loss_fn())
        r = BenchResult("B2_force_fit", "MEASURED", {
            "loss_before": l0, "loss_after": l1,
            "loss_reduction": l0 / max(l1, 1e-30),
            "n_steps": n_steps, "elapsed_s": el,
            "n_trainable": sum(p.numel() for p in ff.trainable_parameters()),
        }, note="force matching only; NOT thermodynamic consistency")
        self.results.append(r)
        return r, ff

    # ---- B3 structural --------------------------------------------------
    def structural(self, aa_traj: torch.Tensor, cg_traj: torch.Tensor,
                   ff: MartiniCGForceField) -> BenchResult:
        m_aa = self.mapping.map_positions(aa_traj)
        rg_aa, rg_cg = radius_of_gyration(m_aa), radius_of_gyration(cg_traj)
        e2e_aa, e2e_cg = end_to_end(m_aa), end_to_end(cg_traj)
        bl_aa = bond_lengths(m_aa, self.params.bonds)
        bl_cg = bond_lengths(cg_traj, self.params.bonds)
        ra, ga = rdf(m_aa)
        rc, gc = rdf(cg_traj)
        rdf_l2 = float((ga - gc).norm() / ga.norm().clamp_min(1e-30))
        r = BenchResult("B3_structural", "MEASURED", {
            "Rg_aa_nm": rg_aa, "Rg_cg_nm": rg_cg,
            "Rg_rel_err": abs(rg_cg - rg_aa) / max(rg_aa, 1e-9),
            "end2end_aa_nm": e2e_aa, "end2end_cg_nm": e2e_cg,
            "bond_mean_aa_nm": float(bl_aa.mean()) if bl_aa.numel() else None,
            "bond_mean_cg_nm": float(bl_cg.mean()) if bl_cg.numel() else None,
            "rdf_rel_l2": rdf_l2,
        })
        self.results.append(r)
        return r

    # ---- B4 performance --------------------------------------------------
    def performance(self, aa_s_per_step: Optional[float],
                    cg_s_per_step: float, n_atoms: int) -> BenchResult:
        nb = len(self.mapping.beads)
        m = {"cg_s_per_step": cg_s_per_step,
             "n_atoms": n_atoms, "n_beads": nb,
             "dof_reduction": n_atoms / max(nb, 1)}
        if aa_s_per_step is None:
            m["aa_s_per_step"] = None
            m["wall_speedup"] = None
            note = "AA timing UNAVAILABLE (OpenMM/PDB missing)"
            status = "MEASURED"
        else:
            m["aa_s_per_step"] = aa_s_per_step
            m["wall_speedup"] = aa_s_per_step / max(cg_s_per_step, 1e-12)
            note = "wall-clock speedup; NOT a physical time-mapping factor"
            status = "MEASURED"
        r = BenchResult("B4_performance", status, m, note=note)
        self.results.append(r)
        return r

    # ---- report ----------------------------------------------------------
    def report(self) -> None:
        print("=" * 92)
        print(f"  MARTINI-CG ↔ OPENMM BENCHMARK v{VERSION}")
        print(f"  {ATTRIBUTION}")
        print("=" * 92)
        if self.params.all_placeholder:
            print("  ⚠️  PARAMETERS ARE PLACEHOLDERS — not official Martini.")
            print("     Supply real parameters via MartiniParams.from_json().")
            print("-" * 92)
        for r in self.results:
            print(f"  [{r.status}] {r.name}")
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    print(f"      {k:<28} {v:.6e}")
                else:
                    print(f"      {k:<28} {v}")
            if r.note:
                print(f"      note: {r.note}")
            print("-" * 92)
        n_meas = sum(1 for r in self.results if r.status == "MEASURED")
        print(f"  {n_meas}/{len(self.results)} benchmarks MEASURED.")
        print("""
  WHAT THESE NUMBERS MEAN
    B1 force_rel_l2 / cosine  : how far the CG model is from AA mean forces.
                                With placeholder params this will be large.
    B2 loss_reduction         : how much force matching improved the fit.
                                This is NOT transferability or thermodynamics.
    B3 Rg / RDF errors        : structural agreement over short trajectories.
    B4 wall_speedup           : wall-clock only. CG time is not physical time.

  WHAT IS NOT MEASURED HERE
    free-energy surfaces, temperature transferability, kinetics,
    official-Martini reproduction, or any claim of state of the art.
""")


# =============================================================================
# §8  SELF-TEST (runs without OpenMM)
# =============================================================================

def self_test() -> bool:
    print("=" * 92)
    print(f"  MARTINI-CG + OPENMM HARNESS v{VERSION} — self-test")
    print(f"  OpenMM available: {HAS_OPENMM} | real_fold_one: {HAS_REALFOLD}")
    print("=" * 92)
    torch.manual_seed(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # synthetic AA system: 40 atoms -> 10 beads
    n_aa = 40
    aa = torch.randn(n_aa, 3, dtype=torch.float64) * 0.3 + 1.0
    mapping = CGMapping.auto_group(n_aa, per_bead=4, type_name="C1")
    params = MartiniParams.generic_placeholder()
    for b in range(len(mapping.beads) - 1):
        params.bonds.append(BondParam(b, b + 1, r0=0.35, k=1250.0))
    for b in range(len(mapping.beads) - 2):
        params.angles.append(AngleParam(b, b + 1, b + 2, 130.0, 25.0))

    # T1 mapping conserves centre of geometry
    xb = mapping.map_positions(aa)
    rep("T1 map shape", xb.shape == (10, 3), f"{tuple(xb.shape)}")
    rep("T1 mean preserved",
        float((xb.mean(0) - aa.mean(0)).abs().max()) < 1e-12)

    # T2 force mapping is exact conjugate
    fa = torch.randn(n_aa, 3, dtype=torch.float64)
    fb = CGMapping.map_forces(fa, mapping)
    rep("T2 total force preserved",
        float((fb.sum(0) - fa.sum(0)).abs().max()) < 1e-12)

    # T3 energy differentiable
    ff = MartiniCGForceField(params, mapping)
    xx = xb.clone().requires_grad_(True)
    E = ff.energy(xx)
    g = torch.autograd.grad(E, xx)[0]
    rep("T3 energy finite + grad finite",
        bool(torch.isfinite(E)) and bool(torch.isfinite(g).all()),
        f"E={float(E):.4f} kJ/mol")

    # T4 force = -grad E consistency
    E2, F2 = ff(xb.clone())
    rep("T4 forward force = -grad",
        float((F2 + g).abs().max()) < 1e-8)

    # T5 bond energy exact at r0
    xb0 = torch.zeros(len(mapping.beads), 3, dtype=torch.float64)
    for b in range(len(mapping.beads)):
        xb0[b, 0] = b * 0.35
    Eb = ff.bond_energy(xb0)
    rep("T5 bond energy zero at r0", float(Eb) < 1e-18, f"E_bond={float(Eb):.2e}")

    # T6 LJ shift: energy zero beyond cutoff
    far = torch.zeros(2, 3, dtype=torch.float64)
    far[1, 0] = params.lj_cutoff + 0.5
    m2 = CGMapping(beads=[BeadSpec([0], "C1"), BeadSpec([1], "C1")])
    ff2 = MartiniCGForceField(MartiniParams.generic_placeholder(), m2)
    Efar = ff2.nonbonded_energy(far)
    rep("T6 LJ shifted to zero beyond cutoff", float(Efar.abs()) < 1e-12,
        f"E={float(Efar):.2e}")

    # T7 exclusions honoured
    pex = MartiniParams.generic_placeholder()
    pex.bonds = [BondParam(0, 1, 0.35, 1000.0)]
    ffe = MartiniCGForceField(pex, m2)
    near = torch.zeros(2, 3, dtype=torch.float64)
    near[1, 0] = 0.4
    rep("T7 bonded pair excluded from LJ",
        float(ffe.nonbonded_energy(near).abs()) < 1e-12)

    # T8 harness force matching runs
    h = BenchmarkHarness(params, mapping)
    r1 = h.force_matching(aa, fa)
    rep("T8 B1 force matching MEASURED", r1.status == "MEASURED",
        f"rel_l2={r1.metrics['force_rel_l2']:.3f} cos={r1.metrics['force_cosine']:.3f}")

    # T9 fitting reduces loss
    r2, ff_fit = h.fit_force_matching(aa, fa, n_steps=60, lr=5e-2)
    rep("T9 B2 fitting reduces loss",
        r2.metrics["loss_after"] <= r2.metrics["loss_before"] + 1e-9,
        f"{r2.metrics['loss_before']:.3e} -> {r2.metrics['loss_after']:.3e}")

    # T10 CG dynamics stable
    integ = CGIntegrator(ff, dt=0.005, friction=20.0)
    xfin, perf = integ.run(xb, n_steps=30)
    rep("T10 CG MD finite", bool(torch.isfinite(xfin).all()),
        f"{perf['s_per_step']*1e3:.2f} ms/step")

    # T11 structural + performance
    h.structural(aa, xfin, ff)
    h.performance(None, perf["s_per_step"], n_aa)
    rep("T11 B3/B4 recorded", len(h.results) >= 4)

    # T12 OpenMM adapter reports honestly
    ref = OpenMMReference(pdb_path=None)
    rep("T12 OpenMM unavailable reported honestly",
        (not ref.available) and len(ref.reason) > 0, ref.reason)

    # T13 JSON round-trip
    import tempfile
    tmp = tempfile.mkdtemp()
    pj = os.path.join(tmp, "params.json")
    mj = os.path.join(tmp, "map.json")
    params.to_json(pj); mapping.to_json(mj)
    p2 = MartiniParams.from_json(pj); m2r = CGMapping.from_json(mj)
    rep("T13 JSON round-trip",
        len(p2.bonds) == len(params.bonds) and len(m2r.beads) == len(mapping.beads))

    h.report()
    print("=" * 92)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 92)
    return ok


# =============================================================================
# §9  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=f"Martini-CG + OpenMM bench v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--pdb", default=None)
    ap.add_argument("--params", default=None, help="Martini params JSON")
    ap.add_argument("--mapping", default=None, help="CG mapping JSON")
    ap.add_argument("--per-bead", type=int, default=4)
    ap.add_argument("--fit-steps", type=int, default=200)
    ap.add_argument("--md-steps", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    params = (MartiniParams.from_json(a.params) if a.params
              else MartiniParams.generic_placeholder())
    ref = OpenMMReference(a.pdb)
    if not ref.available:
        print(f"OpenMM reference UNAVAILABLE: {ref.reason}")
        print("Running self-test instead so nothing is fabricated.")
        raise SystemExit(0 if self_test() else 1)

    ref.minimize(200)
    aa_pos = ref.positions_nm()
    aa_E, aa_F = ref.energy_forces(aa_pos)
    mapping = (CGMapping.from_json(a.mapping) if a.mapping
               else CGMapping.auto_group(aa_pos.shape[0], a.per_bead))

    h = BenchmarkHarness(params, mapping, device=a.device)
    h.force_matching(aa_pos, aa_F)
    _, ff_fit = h.fit_force_matching(aa_pos, aa_F, n_steps=a.fit_steps)

    aa_fin, aa_perf = ref.md(a.md_steps)
    ff_run = MartiniCGForceField(params, mapping, device=a.device)
    integ = CGIntegrator(ff_run)
    cg_fin, cg_perf = integ.run(mapping.map_positions(aa_pos), a.md_steps)
    h.structural(aa_fin, cg_fin, ff_run)
    h.performance(aa_perf["s_per_step"], cg_perf["s_per_step"], aa_pos.shape[0])
    h.report()

    if a.out:
        with open(a.out, "w") as f:
            json.dump([asdict(r) for r in h.results], f, indent=2)
        print(f"json -> {a.out}")


if __name__ == "__main__":
    main()
