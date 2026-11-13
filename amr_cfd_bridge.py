#!/usr/bin/env python3
"""
AMR ↔ CFD BRIDGE — plugs one_amr_core into every ONE-ecosystem solver.
Developer: PAI and Yoon A Limsuwan / MSPS NETWORK | MIT | 2026

Each adapter converts a solver's native state layout into the AMR field
dict {alpha|rho|u|v|w|...}, runs the AMR cycle, and writes back.

Solvers supported (state layouts taken from the actual sources):
  sesi_multiphase_flow.MultiphaseNavierStokes3D  : VOFState(alpha,u,v,w,p)
  structuralfluctuatinghydro.StructuralFluctuatingHydro : (rho,ux,uy,uz,p)
  super_dns_one.CompressibleSolver : (rho,rhou,rhov,rhow,rhoE) cell-centred
  structural_cahn_hilliard.StructuralCahnHilliard3D : u order parameter

HONEST: the compressible DNS stores momentum at CELL CENTRES, not faces.
Its adapter therefore treats rhou/rhov/rhow as cell-centred scalars for
transfer (conservative averaging is still exact) and does NOT use the
face-velocity path. That is a real layout difference, not a bug.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from one_amr_core import AMRConfig, AMREngine, restrict_cc, prolongate_cc

log = logging.getLogger("amr_bridge")


class VOFAMRAdapter:
    """sesi_multiphase_flow.MultiphaseNavierStokes3D"""
    def __init__(self, solver, cfg: Optional[AMRConfig] = None):
        c = solver.cfg
        self.cfg = cfg or AMRConfig(nx=c.nx, ny=c.ny, nz=c.nz,
                                    Lx=c.Lx, Ly=c.Ly, Lz=c.Lz,
                                    device=c.device, dtype=c.dtype)
        self.amr = AMREngine(self.cfg)
        self.solver = solver

    def to_fields(self, st) -> Dict[str, Tensor]:
        return {"alpha": st.alpha, "u": st.u, "v": st.v, "w": st.w, "p": st.p}

    def step_fn(self, f: Dict[str, Tensor], dt: float) -> Dict[str, Tensor]:
        from sesi_multiphase_flow import VOFState
        st = VOFState(alpha=f["alpha"], u=f["u"], v=f["v"], w=f["w"], p=f["p"])
        out = self.solver.step(st, dt=dt)
        return self.to_fields(out)

    def advance(self, st, dt: float):
        from sesi_multiphase_flow import VOFState
        f = self.amr.advance(self.to_fields(st), self.step_fn, dt)
        return VOFState(alpha=f["alpha"], u=f["u"], v=f["v"], w=f["w"],
                        p=f["p"], T=st.T, solid=st.solid,
                        t=st.t + dt, step=st.step + 1)


class FHAMRAdapter:
    """structuralfluctuatinghydro.StructuralFluctuatingHydro"""
    def __init__(self, solver, cfg: Optional[AMRConfig] = None):
        c = solver.cfg
        self.cfg = cfg or AMRConfig(nx=c.Nx, ny=c.Ny, nz=c.Nz,
                                    Lx=c.Lx, Ly=c.Ly, Lz=c.Lz,
                                    dtype="float64" if c.dtype == torch.float64 else "float32")
        self.amr = AMREngine(self.cfg)
        self.solver = solver

    def step_fn(self, f, dt):
        old_dt = self.solver.cfg.dt
        self.solver.cfg.dt = dt
        try:
            r, ux, uy, uz, p, diag = self.solver.step(
                f["rho"], f["u"], f["v"], f["w"], f["p"])
        finally:
            self.solver.cfg.dt = old_dt
        return {"rho": r, "u": ux, "v": uy, "w": uz, "p": p}

    def advance(self, rho, ux, uy, uz, p, dt):
        f = self.amr.advance({"rho": rho, "u": ux, "v": uy, "w": uz, "p": p},
                             self.step_fn, dt)
        return f["rho"], f["u"], f["v"], f["w"], f["p"]


class CompressibleAMRAdapter:
    """
    super_dns_one.CompressibleSolver — CELL-CENTRED conserved variables.
    Transfer uses conservative cell averaging (exact), NOT face interpolation.
    """
    KEYS = ("rho", "rhou", "rhov", "rhow", "rhoE")

    def __init__(self, solver, cfg: Optional[AMRConfig] = None):
        c = solver.cfg
        self.cfg = cfg or AMRConfig(nx=c.nx, ny=c.ny, nz=c.nz,
                                    Lx=c.Lx, Ly=c.Ly, Lz=c.Lz,
                                    device=str(solver.device),
                                    dtype="float64" if solver.dtype == torch.float64 else "float32")
        self.amr = AMREngine(self.cfg)
        self.solver = solver

    def to_fields(self):
        return {k: getattr(self.solver, k) for k in self.KEYS}

    def step_fn(self, f, dt):
        for k in self.KEYS:
            setattr(self.solver, k, f[k])
        self.solver.step(dt=dt)
        return self.to_fields()

    def advance(self, dt):
        f = self.amr.advance(self.to_fields(), self.step_fn, dt)
        for k in self.KEYS:
            setattr(self.solver, k, f[k])
        return f


class CHAMRAdapter:
    """structural_cahn_hilliard.StructuralCahnHilliard3D — order parameter u."""
    def __init__(self, solver, cfg: Optional[AMRConfig] = None):
        c = solver.cfg
        n = 32
        self.cfg = cfg or AMRConfig(nx=n, ny=n, nz=n,
                                    Lx=n*c.dx, Ly=n*c.dx, Lz=n*c.dx,
                                    device=c.device,
                                    dtype="float64" if c.dtype == torch.float64 else "float32")
        self.amr = AMREngine(self.cfg)
        self.solver = solver

    def step_fn(self, f, dt):
        old = self.solver.cfg.dt
        self.solver.cfg.dt = dt
        try:
            u_new = self.solver.step(f["u"], f.get("sigma"))
        finally:
            self.solver.cfg.dt = old
        return {**f, "u": u_new}

    def advance(self, u, sigma=None, dt=1e-5):
        f = {"u": u}
        if sigma is not None:
            f["sigma"] = sigma
        return self.amr.advance(f, self.step_fn, dt)["u"]


def make_adapter(solver, kind: str, cfg: Optional[AMRConfig] = None):
    kind = kind.lower()
    if kind == "vof":
        return VOFAMRAdapter(solver, cfg)
    if kind == "fh":
        return FHAMRAdapter(solver, cfg)
    if kind in ("dns", "compressible"):
        return CompressibleAMRAdapter(solver, cfg)
    if kind in ("ch", "cahn_hilliard"):
        return CHAMRAdapter(solver, cfg)
    raise ValueError(f"unknown solver kind {kind!r}; "
                     f"choose vof|fh|dns|ch")


if __name__ == "__main__":
    print(__doc__)
    print("\nAdapters: vof | fh | dns | ch")
    print("Usage: adapter = make_adapter(solver, 'vof'); st = adapter.advance(st, dt)")
