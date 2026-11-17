"""
SOCTurbulenceWrapper — model-agnostic control layer.

Pipeline:
  1. nu_t_base = base(state, S, Delta, nu, grid)     [standard closure]
  2. resolution monitor: eps_tot -> eta -> resolved  [DNS criterion, T7]
  3. CSOC: stress -> gate + nu_soc, event bound      [T1]
  4. learned gain (+ OPMC design field)              [T4, exact grad]
  5. nu_t_final = (1 - resolved) * nu_t_corrected    [DNS bridging]
  6. Carpet AMR + kappa-fatness                      [T2, T3]
  7. Hodge-inspired split + assumption-light CI      [T5, T6, diagnostics]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .primitives import soft_pos, KAPPA_FATNESS
from .grid import Grid, strain_rate
from .resolution import ResolutionMonitor
from .csoc import CSOCController
from .carpet import CarpetAMRIndicator
from .design import OPMCDesign
from .diagnostics import HodgeInspiredSplit, subsample_ci
from .models import TurbBase


@dataclass
class WrapperConfig:
    c_eta: float = 2.0
    res_width: float = 0.5
    theta_hi: float = 0.15
    theta_lo: float = 0.05
    nu_collapse: float = 1e-3
    decay_rate: float = 0.05
    carpet_side: int = 9
    carpet_levels: int = 3
    kappa: float = KAPPA_FATNESS
    use_opmc_design: bool = True
    use_hodge: bool = True
    dns_mode: str = "adaptive"      # "off" | "adaptive" | "force"

    def __post_init__(self):
        if self.dns_mode not in ("off", "adaptive", "force"):
            raise ValueError("dns_mode must be off|adaptive|force")
        if not self.theta_hi > self.theta_lo:
            raise ValueError("require theta_hi > theta_lo")


class SOCTurbulenceWrapper(nn.Module):
    def __init__(self, base: TurbBase, grid: Grid, nu: float,
                 cfg: Optional[WrapperConfig] = None):
        super().__init__()
        self.cfg = cfg or WrapperConfig()
        self.base = base
        self.grid = grid
        self.nu = float(nu)
        self.res = ResolutionMonitor(self.cfg.c_eta, self.cfg.res_width)
        self.csoc = CSOCController(self.cfg.theta_hi, self.cfg.theta_lo,
                                   self.cfg.nu_collapse, self.cfg.decay_rate)
        self.carpet = CarpetAMRIndicator(self.cfg.carpet_side,
                                         self.cfg.carpet_levels, self.cfg.kappa)
        self.log_gain = nn.Parameter(torch.tensor(0.0))
        self.design = OPMCDesign(grid.shape) if self.cfg.use_opmc_design else None
        self.hodge = HodgeInspiredSplit() if self.cfg.use_hodge else None
        self._hist: List[float] = []

    def reset(self) -> None:
        self.csoc.reset()
        self._hist.clear()

    def forward(self, state: Dict[str, Tensor], u: Optional[Tensor] = None
                ) -> Dict[str, Any]:
        u = state["u"] if u is None else u
        miss = self.base.missing(state)
        if miss:
            raise KeyError(f"{self.base.of_name} requires {miss} in state; "
                           f"these must be TRANSPORTED by the host solver "
                           f"(local_equilibrium_state() is a DEMO estimate).")
        S, _ = strain_rate(u, self.grid)
        delta = self.grid.delta

        nu_t_base = self.base(state, S, delta, self.nu, self.grid)   # [B1] works
        rm = self.res(S, self.nu, nu_t_base, delta, state.get("epsilon"))
        stress = (nu_t_base / max(self.nu, 1e-30)).mean()
        soc = self.csoc(stress)

        gain = torch.exp(self.log_gain)
        design = self.design() if self.design is not None else 0.0
        nu_t_corr = soft_pos(nu_t_base * (1.0 + gain * soc["gate"] + design)
                             + soc["nu_soc"], beta=80.0)

        if self.cfg.dns_mode == "off":
            resolved = torch.zeros_like(S)
        elif self.cfg.dns_mode == "force":
            resolved = torch.ones_like(S)
        else:
            resolved = rm["resolved"]
        nu_t_final = (1.0 - resolved) * nu_t_corr

        amr = self.carpet(rm["delta_over_eta"], S, self.cfg.c_eta)
        out: Dict[str, Any] = {
            "nu_t": nu_t_final, "nu_t_base": nu_t_base,
            "nu_t_corrected": nu_t_corr, "strain": S,
            "resolution": rm, "csoc": soc, "amr": amr,
            "base_model": self.base.of_name, "dns_mode": self.cfg.dns_mode,
            "state_source": state.get("_source", "host_solver"),
        }
        if self.base.of_name == "k_omega_sst":
            out["sst_wall_distance_supplied"] = "d_wall" in state
            if "d_wall" not in state:
                out["sst_warning"] = ("F2 = 1 (far-field limit). "
                                      "NOT valid for wall-bounded flow.")
        if self.hodge is not None:
            out["hodge"] = self.hodge(nu_t_final)
        rf = float(rm["resolved_fraction"])
        out["regime"] = ("DNS-resolved" if rf > 0.9 else
                         "hybrid LES/DNS" if rf > 0.1 else "modelled (RANS/LES)")
        self._hist.append(rf)
        return out

    def diagnostics_ci(self, conf: float = 0.95) -> Dict[str, float]:
        return subsample_ci(np.asarray(self._hist), conf) if self._hist else {}
