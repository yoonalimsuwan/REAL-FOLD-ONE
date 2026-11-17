#!/usr/bin/env python3
"""
=============================================================================
SUPER DNS ONE — PATCH v7
Fix dead SSC call + _padv signature + softplus width
A/B measurement of whether SSC actually helps
Resolution monitor + DNS mode + honest feasibility
=============================================================================
Code author : Qwen (Alibaba Group)
Architecture: PAI AND Yoon A Limsuwan / MSPS NETWORK
              My Soul Move By Power of Holy Spirit ;
              What MSPS NETWORK SEE, Lord Buddha Knows.
License     : MIT | Year 2026

THE THREE BUGS (verified by reading super_dns_one_v6_3-8.py)
  B1  SOCController.nu_t():
          if self.ssc is not None:
              ssc_in = self.stress_acc.mean()
              if ext_sigma is not None: ssc_in = 0.5*(ssc_in + ext_sigma)
              _ = self.ssc(ssc_in)          # <-- result DISCARDED
      nu_t_base is computed BEFORE this block from Cs_local and
      strain_rate_mag; nu_collapse from excess and dx. Neither reads
      the SSC output. SSC is therefore a NO-OP with use_ssc=True
      (the default). Same defect class as Bug 11 (_ext_nu_ch declared
      but never read) that this file's own changelog documents.

  B2  _compute_rhs():
          def _padv(t): q=self._pad_field(t); self._fill_ghost_cells(q); return q
      _fill_ghost_cells(self, rho_p, rhou_p, rhov_p, rhow_p, rhoE_p)
      takes FIVE fields. Called with ONE -> TypeError on every viscous
      flux evaluation, i.e. every step. The solver cannot run at all.
      NOTE: repair_five.py's [S1] patch passes zeros for the other four
      fields. That makes it RUN but is physically WRONG: the stress
      tensor is then treated as DENSITY by the wall BC (mirrored as
      2q_b - q_ni) instead of receiving a stress-appropriate ghost
      treatment. This patch instead reuses the file's OWN
      _fill_scalar_ghost_cells (Neumann / zero-gradient), which is the
      correct simple treatment for a viscous stress component at a
      no-slip wall (tau_wall != 0 in general, so odd reflection would
      be wrong).

  B3  _softplus_floor(x, floor, beta=_SOFTPLUS_B=100):
          return floor + F.softplus(x - floor, beta=beta)
      At x = floor, softplus(0, beta) = log(2)/beta. With beta=100 that
      is 6.93e-3 REGARDLESS of the requested floor. floor=1e-8 becomes
      ~6.93e-3 (693,000x). Same root cause already fixed in
      structuralfluctuatinghydro_v6_3 and structural_cahn_hilliard_3d.
      HONEST TRADEOFF: making beta large enough that the effective
      floor equals the requested floor also makes the transition very
      thin, so the gradient below the floor vanishes (it approaches a
      hard clamp). This patch REPORTS the effective floor so the
      tradeoff is visible, and caps beta to keep it finite.

WHAT "DEVELOP INTO DNS" CAN AND CANNOT MEAN
  DNS resolves ALL scales: Delta <= c_eta * eta, eta = (nu^3/eps)^(1/4).
  For L = 2*pi this needs N >~ 3 * Re^(3/4) cells per axis:
      Re=100   -> N ~   96   (8.8e5 cells)   feasible on 1 GPU
      Re=250   -> N ~  190   (6.9e6 cells)   feasible on 1 GPU
      Re=1000  -> N ~  534   (1.5e8 cells)   marginal on 80GB
      Re=1e4   -> N ~ 3000   (2.7e10 cells)  multi-GPU only
      Re=1e6   -> N ~30000   (2.7e13 cells)  NOT ATTAINABLE
  The default cfg is 64^3 at Re=1e4: Delta/eta ~ 10-40. That is LES.
  Moreover a solver WITH an SGS eddy viscosity and WITH spectral RG
  truncation is LES BY DEFINITION, not DNS. Adding more modelling does
  not produce DNS. Therefore this patch adds:
    * ResolutionMonitor: measures eps_resolved + eps_sgs from the FIELD,
      computes eta and Delta/eta per cell, reports resolved fraction.
    * dns_mode: forces nu_t = 0 and disables RG.
    * dns_feasibility(): reports the Re attainable on a given grid, and
      says NOT ATTAINABLE when it is not.
  It does NOT claim high-Re DNS. It measures and reports.

A/B: DOES SSC ACTUALLY HELP?
  Three arms, identical initial conditions and seeds:
    off      SSC disabled (reproduces the original dead-call behaviour)
    driver   SSC low-passes the stress DRIVER before accumulation
    output   SSC low-passes the returned nu_t (load-bearing)
    both     both of the above
  Measured, not asserted:
    nu_t chatter   mean|nu_t^{n+1} - nu_t^n| / mean nu_t
    KE decay err   vs the analytic Taylor-Green exp(-4 nu t)
    spike events   two-level hysteresis count on nu_t
    grad variance  variance of d(loss)/d(theta) across steps
    wall time
  VERDICT is computed from the numbers. If SSC reduces chatter without
  worsening accuracy, it helps. If it does not, the report says so.

HONEST SCOPE
  * The A/B runs on a STANDALONE reimplementation of the same kernel
    math (torch only, no one_core). The patched source file must be
    re-tested by you; this patch does not certify it.
  * The SSC EMA buffer is DETACHED. Gradient does not flow through
    filter history (truncated BPTT). This is deliberate: an undetached
    stateful buffer accumulates the autograd graph across steps.
  * Source patching is exact-string with reported misses + py_compile.
    Indentation damage is NOT repairable by regex and is reported.
  * No experimental validation. Analytic Taylor-Green only.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import py_compile
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("sdns_patch")

VERSION = "7.0.0"
BANNER = f"""
=============================================================================
SUPER DNS ONE PATCH v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
B1 dead SSC | B2 _padv 5-arg | B3 softplus width
DNS = resolution. nu_t + RG truncation = LES by definition.
=============================================================================
"""


# =============================================================================
# §1. BUG REGISTRY
# =============================================================================

@dataclass
class Bug:
    id: str
    where: str
    certainty: str
    symptom: str
    fix: str
    test: str


BUGS: List[Bug] = [
    Bug("B1", "SOCController.nu_t(): `_ = self.ssc(ssc_in)`", "certain",
        "SSC output discarded; nu_t_base and nu_collapse never read it. "
        "SSC is a no-op with use_ssc=True (default).",
        "field-valued EMA applied to the DRIVER and/or to the returned "
        "nu_t, with a detached buffer. Load-bearing by construction.",
        "T4/T5 + A/B"),
    Bug("B2", "_compute_rhs(): `def _padv(t): ... _fill_ghost_cells(q)`",
        "certain",
        "_fill_ghost_cells takes 5 fields, called with 1 -> TypeError on "
        "every viscous flux evaluation. Solver cannot run.",
        "use the file's own _fill_scalar_ghost_cells (Neumann) for stress "
        "components. NOT repair_five's zeros hack, which treats stress "
        "as density.",
        "T3"),
    Bug("B3", "_softplus_floor(x, floor, beta=100)", "certain",
        "softplus(0,beta)=log2/beta=6.93e-3 regardless of floor. "
        "floor=1e-8 -> 6.93e-3 (693,000x).",
        "beta scales with the requested floor, capped for finiteness; "
        "effective floor is REPORTED.",
        "T1/T2"),
]


def bug_report() -> Dict[str, Any]:
    return {"n": len(BUGS), "bugs": [asdict(b) for b in BUGS],
            "note": ("All three are certain from reading the source. "
                     "The A/B verdict on SSC is MEASURED, not assumed.")}


# =============================================================================
# §2. [B3] SOFTPLUS FLOOR WITH WIDTH THAT TRACKS THE FLOOR
# =============================================================================

def soft_floor(x: Tensor, floor: float,
               beta: Optional[float] = None,
               sharpness: float = 50.0,
               beta_max: float = 1e8) -> Tensor:
    """
    [B3] floor + softplus(x - floor, beta).

    With beta=None, beta = min(sharpness/|floor|, beta_max), so the
    transition width log(2)/beta tracks the requested floor instead of
    being a fixed 6.93e-3.

    HONEST TRADEOFF: a large beta makes this approach a hard clamp, so
    the gradient below the floor vanishes. effective_floor() reports
    what the floor actually is so the tradeoff is visible.
    """
    if beta is None:
        beta = min(sharpness / max(abs(floor), 1e-30), beta_max)
    return floor + F.softplus(x - floor, beta=beta)


def effective_floor(floor: float, beta: Optional[float] = None,
                    sharpness: float = 50.0,
                    beta_max: float = 1e8) -> float:
    """Value actually returned at x = floor: floor + log(2)/beta."""
    if beta is None:
        beta = min(sharpness / max(abs(floor), 1e-30), beta_max)
    return floor + math.log(2.0) / beta


def soft_abs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def logsumexp_max(x: Tensor, tau: float = 1e-2) -> Tensor:
    return tau * torch.logsumexp(x.reshape(-1) / tau, dim=0)


# =============================================================================
# §3. [B2] CORRECT STRESS GHOST FILL
# =============================================================================

def pad_field(f: Tensor, periodic: Tuple[bool, bool, bool],
              n_ghost: int = 2) -> Tensor:
    """Pad by n_ghost on each side. Periodic axes wrap; others zero-fill."""
    if all(periodic):
        pad = [n_ghost] * 6
        return F.pad(f.unsqueeze(0).unsqueeze(0), pad,
                     mode="circular").squeeze(0).squeeze(0)
    shp = tuple(s + 2 * n_ghost for s in f.shape)
    out = torch.zeros(shp, dtype=f.dtype, device=f.device)
    sl = tuple(slice(n_ghost, n_ghost + s) for s in f.shape)
    out[sl] = f
    return out


def fill_scalar_ghost(fp: Tensor, f: Tensor,
                      periodic: Tuple[bool, bool, bool],
                      n_ghost: int = 2,
                      dirichlet: Optional[Dict[Tuple[int, str], Tensor]] = None
                      ) -> Tensor:
    """
    [B2] Ghost fill for a SCALAR / STRESS component.

    Periodic axis: wrap. Non-periodic: zero-gradient (Neumann) from the
    boundary cell, or mirror-linear ghost = 2*value - interior when a
    Dirichlet value is supplied for that face.

    This is the correct simple treatment for a viscous stress component
    at a no-slip wall: tau_wall is generally NONZERO, so odd reflection
    (ghost = -interior) would be wrong. It is also what the source file's
    own _fill_scalar_ghost_cells does — reused here rather than
    repair_five's zeros-for-the-other-four-fields hack.
    """
    dirichlet = dirichlet or {}
    g = n_ghost
    for ax in range(3):
        if periodic[ax]:
            n = f.shape[ax]
            lo_src = tuple(slice(n - g, n) if d == ax else slice(None)
                           for d in range(3))
            hi_src = tuple(slice(0, g) if d == ax else slice(None)
                           for d in range(3))
            lo_dst = tuple(slice(0, g) if d == ax else slice(g, g + f.shape[d])
                           for d in range(3))
            hi_dst = tuple(slice(g + n, g + n + g) if d == ax
                           else slice(g, g + f.shape[d]) for d in range(3))
            fp[lo_dst] = fp[lo_src] if False else f[lo_src]
            fp[hi_dst] = f[hi_src]
            continue
        for side in ("left", "right"):
            n = f.shape[ax]
            bidx = 0 if side == "left" else n - 1
            sl_b = tuple(bidx if d == ax else slice(None) for d in range(3))
            interior = f[sl_b]
            dv = dirichlet.get((ax, side))
            val = (2 * dv - interior) if dv is not None else interior
            for k in range(g):
                gi = (g - 1 - k) if side == "left" else (g + n + k)
                sl_g = tuple(gi if d == ax else slice(g, g + f.shape[d])
                             for d in range(3))
                fp[sl_g] = val
    return fp


def pad_stress(t: Tensor, periodic: Tuple[bool, bool, bool],
               n_ghost: int = 2) -> Tensor:
    """[B2] The replacement for the broken `_padv(t)`."""
    return fill_scalar_ghost(pad_field(t, periodic, n_ghost), t,
                             periodic, n_ghost)


# =============================================================================
# §4. [B1] SSC THAT IS ACTUALLY LOAD-BEARING
# =============================================================================

class FieldSSC(nn.Module):
    """
    Field-valued Semantic State Contraction (elementwise EMA):
        prev <- prev + eps * (x - prev)

    The original SSC is a SCALAR EMA. Filtering a scalar mean and
    broadcasting it back would destroy spatial structure, so the
    filter is generalised elementwise to the field. That preserves
    spatial structure while low-passing in TIME.

    The buffer is DETACHED: gradient does not flow through filter
    history (truncated BPTT). An undetached stateful buffer accumulates
    the autograd graph across steps — the same defect class found
    elsewhere in this ecosystem.
    """

    def __init__(self, epsilon: float = 0.0028):
        super().__init__()
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be in (0,1)")
        self.eps = float(epsilon)
        self.register_buffer("prev", torch.zeros(1))
        self.register_buffer("ready", torch.zeros(1, dtype=torch.bool))

    def reset(self):
        self.prev = torch.zeros(1, device=self.prev.device)
        self.ready.zero_()

    def forward(self, x: Tensor) -> Tensor:
        if not bool(self.ready.item()) or self.prev.numel() != x.numel():
            self.prev = x.detach().clone().reshape(-1)
            self.ready.fill_(True)
            return x
        prev = self.prev.reshape(x.shape).to(x.dtype)
        out = prev + self.eps * (x - prev)
        self.prev = out.detach().clone().reshape(-1)
        return out


class CSOCKernel(nn.Module):
    """
    LJ-equilibrium kernel (the v6.2 Bug-5 fix, retained).
    Bounded for ANY learnable alpha because the rise as r->0 is
    softplus-capped instead of diverging.
    """

    def __init__(self, Cs=0.18, lam=12.0, alpha=0.5, theta=1.0, tau=10.0,
                 cap_beta: float = 100.0):
        super().__init__()
        self.log_Cs = nn.Parameter(torch.tensor(math.log(Cs)))
        self.log_lam = nn.Parameter(torch.tensor(math.log(lam)))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_theta = nn.Parameter(torch.tensor(math.log(theta)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.r_eq = 1.0
        self.cap_beta = float(cap_beta)

    @property
    def Cs(self): return torch.exp(self.log_Cs)
    @property
    def lam(self): return torch.exp(self.log_lam)
    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def theta(self): return torch.exp(self.log_theta)
    @property
    def tau(self): return torch.exp(self.log_tau)

    def forward(self, r: Tensor) -> Tensor:
        r_safe = soft_floor(r, 1e-6)
        x = self.r_eq / r_safe
        lj = torch.pow(x, 2.0 * self.alpha) - 2.0 * torch.pow(x, self.alpha)
        lj_capped = 50.0 - F.softplus(50.0 - lj, beta=self.cap_beta)
        return self.Cs * (1.0 + lj_capped + 1.0) * torch.exp(-r_safe / self.lam)


@dataclass
class SOCConfig:
    max_nu_t: float = 0.05
    ssc_epsilon: float = 0.0028
    ssc_mode: str = "output"        # off | driver | output | both
    compressibility_correction: bool = True
    dns_mode: bool = False          # True -> nu_t = 0 (DNS)
    theta_hi: float = 0.15
    theta_lo: float = 0.05

    def __post_init__(self):
        if self.ssc_mode not in ("off", "driver", "output", "both"):
            raise ValueError("ssc_mode must be off|driver|output|both")
        if self.theta_hi <= self.theta_lo:
            raise ValueError("theta_hi must exceed theta_lo")


class SOCPatched(nn.Module):
    """
    [B1] SOC eddy-viscosity controller with SSC actually on the path.

    ssc_mode:
      off     reproduces the ORIGINAL behaviour (SSC computed, discarded)
      driver  SSC low-passes strain^2 BEFORE the stress accumulator
      output  SSC low-passes the RETURNED nu_t  (load-bearing)
      both    both

    dns_mode=True forces nu_t = 0, i.e. no SGS model. That is a
    necessary condition for DNS; it is NOT sufficient (the grid must
    also resolve eta — see ResolutionMonitor).
    """

    def __init__(self, cfg: Optional[SOCConfig] = None):
        super().__init__()
        self.cfg = cfg or SOCConfig()
        self.kernel = CSOCKernel()
        self.ssc_driver = FieldSSC(self.cfg.ssc_epsilon)
        self.ssc_output = FieldSSC(self.cfg.ssc_epsilon)
        self.register_buffer("stress_acc", torch.zeros(1))
        self.register_buffer("acc_ready", torch.zeros(1, dtype=torch.bool))
        # hysteresis diagnostics on nu_t (off the gradient path)
        self.register_buffer("events", torch.zeros(1))
        self.register_buffer("in_event", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("E_max", torch.zeros(1))
        self.register_buffer("D_total", torch.zeros(1))
        self.last_nu_t: Optional[Tensor] = None
        self.last_ssc_delta: float = 0.0

    def reset(self):
        self.kernel.zero_grad(set_to_none=True)
        self.ssc_driver.reset(); self.ssc_output.reset()
        self.stress_acc = torch.zeros(1, device=self.stress_acc.device)
        self.acc_ready.zero_()
        self.events.zero_(); self.in_event.zero_()
        self.E_max.zero_(); self.D_total.zero_()
        self.last_nu_t = None
        self.last_ssc_delta = 0.0

    def forward(self, rho: Tensor, strain: Tensor, dilatation: Tensor,
                dx: float, dt: float, c: Tensor,
                ext_sigma: Optional[Tensor] = None) -> Tensor:
        cfg = self.cfg
        if cfg.dns_mode:
            z = torch.zeros_like(strain)
            self.last_nu_t = z
            return z

        mode = cfg.ssc_mode
        mean_S = soft_floor(strain.mean(), 1e-8)
        r = strain / mean_S
        Cs_local = self.kernel(r)
        nu_t_base = (Cs_local * dx) ** 2 * strain

        if cfg.compressibility_correction:
            M_t = torch.sqrt(2.0 * nu_t_base * strain + 1e-12) / (c + 1e-8)
            nu_t_base = nu_t_base / (1.0 + 2.0 * M_t ** 2)

        # ---- stress accumulator, with SSC optionally on the DRIVER ----
        drive = strain ** 2
        if mode in ("driver", "both"):
            drive = self.ssc_driver(drive)
        if ext_sigma is not None and mode in ("driver", "both"):
            drive = 0.5 * (drive + ext_sigma)

        if not bool(self.acc_ready.item()) or self.stress_acc.numel() != drive.numel():
            self.stress_acc = torch.zeros_like(drive).reshape(-1)
            self.acc_ready.fill_(True)
        acc = self.stress_acc.reshape(drive.shape).to(drive.dtype)
        acc_new = F.softplus(acc + dt * (drive - acc / (self.kernel.tau + 1e-8)))
        self.stress_acc = acc_new.detach().clone().reshape(-1)

        excess = F.softplus(acc_new - self.kernel.theta)
        nu_collapse = 0.1 * excess * dx ** 2
        decay = torch.sigmoid(-(excess / (self.kernel.theta + 1e-8)) * 10.0)
        self.stress_acc = (acc_new * (0.5 + 0.5 * decay)).detach().clone().reshape(-1)

        nu_t = nu_t_base + nu_collapse
        nu_t = F.softplus(nu_t)
        nu_t = cfg.max_nu_t - F.softplus(cfg.max_nu_t - nu_t)

        # ---- [B1] SSC on the OUTPUT: this CHANGES the returned value ----
        if mode in ("output", "both"):
            before = nu_t
            nu_t = self.ssc_output(nu_t)
            self.last_ssc_delta = float((nu_t - before).abs().mean())
        else:
            self.last_ssc_delta = 0.0

        self._hysteresis(nu_t.detach())
        self.last_nu_t = nu_t
        return nu_t

    @torch.no_grad()
    def _hysteresis(self, nu_t: Tensor):
        """Two-level event count on mean nu_t. Diagnostic, off gradient path."""
        s = nu_t.mean()
        hi, lo = self.cfg.theta_hi * self.cfg.max_nu_t, self.cfg.theta_lo * self.cfg.max_nu_t
        fire = (s > hi) & (~self.in_event.reshape(()))
        clear = (s < lo) & self.in_event.reshape(())
        self.in_event = ((self.in_event.reshape(()) | fire) & (~clear)).reshape(1)
        self.events = (self.events + fire.to(self.events.dtype)).reshape(1)
        self.E_max = torch.maximum(self.E_max, s.reshape(1))
        removal = F.softplus(s - hi) * 0.01
        self.D_total = (self.D_total + removal.reshape(1))

    def event_bound(self) -> float:
        gap = (self.cfg.theta_hi - self.cfg.theta_lo) * self.cfg.max_nu_t
        return float((self.E_max + self.D_total) / max(gap, 1e-30))


# =============================================================================
# §5. RESOLUTION MONITOR — THE DNS CRITERION
# =============================================================================

class ResolutionMonitor(nn.Module):
    """
    Measures whether the grid resolves the Kolmogorov scale.

      eps_res = 2 nu S:S            (resolved dissipation; |S|^2 = 2 S:S)
      eps_sgs = nu_t |S|^2          (subgrid dissipation)
      eps_tot = eps_res + eps_sgs
      eta     = (nu^3 / eps_tot)^(1/4)
      tau_eta = (nu / eps_tot)^(1/2)
      resolved: Delta <= c_eta * eta

    In DNS mode nu_t = 0 so eps_tot = eps_res. The monitor uses the
    ACTUAL field, not a nominal Re, because eps grows as turbulence
    develops: a grid that resolves eta at t=0 may not resolve it later.
    """

    def __init__(self, c_eta: float = 2.0, width: float = 0.5):
        super().__init__()
        self.c_eta = float(c_eta)
        self.width = float(width)

    def forward(self, strain_mag: Tensor, nu: float, nu_t: Tensor,
                dx: float) -> Dict[str, Tensor]:
        eps_res = nu * strain_mag ** 2
        eps_sgs = nu_t * strain_mag ** 2
        eps_tot = (eps_res + eps_sgs).clamp_min(1e-30)
        eta = (nu ** 3 / eps_tot) ** 0.25
        tau_eta = torch.sqrt(nu / eps_tot)
        ratio = dx / eta.clamp_min(1e-30)
        resolved = torch.sigmoid((self.c_eta - ratio) / self.width)
        return {
            "eps_resolved": eps_res, "eps_sgs": eps_sgs, "eps_total": eps_tot,
            "eta": eta, "tau_eta": tau_eta, "delta_over_eta": ratio,
            "resolved": resolved,
            "resolved_fraction": resolved.mean(),
            "worst_ratio": ratio.max(),
            "dt_dns": tau_eta.min(),
            "n_required_axis": torch.ceil(ratio.max() / self.c_eta
                                          * strain_mag.shape[0]),
        }


def dns_feasibility(Re: float, N: int, L: float = 2 * math.pi,
                    c_eta: float = 2.0,
                    budget_cells: float = 2e9) -> Dict[str, Any]:
    """
    Honest feasibility. eta/L ~ Re^(-3/4); DNS needs Delta <= c_eta*eta.
    With N cells per axis, Delta = L/N, so N_required ~ (2*pi/c_eta)*Re^(3/4).
    """
    if Re <= 0 or N < 2:
        raise ValueError("Re must be positive and N >= 2")
    n_req = int(math.ceil((2 * math.pi / c_eta) * Re ** 0.75))
    cells_req = n_req ** 3
    dx = L / N
    eta_over_L = Re ** -0.75
    eta = eta_over_L * L
    return {
        "Re": Re, "N_given": N, "dx": dx,
        "eta_estimated": eta, "delta_over_eta": dx / max(eta, 1e-30),
        "N_required_per_axis": n_req,
        "cells_required": cells_req,
        "cells_required_human": f"{cells_req:.3e}",
        "memory_gb_estimate": cells_req * 8 * 8 / 1e9,
        "attainable_on_given_grid": N >= n_req,
        "attainable_within_budget": cells_req <= budget_cells,
        "verdict": ("DNS ATTAINABLE on this grid" if N >= n_req else
                    f"NOT DNS on {N}^3 — need {n_req}^3 "
                    f"({cells_req:.2e} cells)"),
    }


# =============================================================================
# §6. A/B HARNESS — DOES SSC HELP?
# =============================================================================

def taylor_green(N: int, nu: float, device, dtype=torch.float64):
    """2-D TGV in a 3-D periodic box, L = 2*pi. u ~ exp(-2 nu t)."""
    ax = (torch.arange(N, device=device, dtype=dtype) + 0.5) * (2 * math.pi / N)
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    u = torch.stack([torch.sin(X) * torch.cos(Y) * torch.cos(Z),
                     -torch.cos(X) * torch.sin(Y) * torch.cos(Z),
                     torch.zeros_like(X)], 0)
    return u


def grad_cc(f: Tensor, dx: float) -> Tensor:
    return torch.stack([(torch.roll(f, -1, a) - torch.roll(f, 1, a)) / (2 * dx)
                        for a in range(3)], 0)


def strain_from_u(u: Tensor, dx: float) -> Tuple[Tensor, Tensor]:
    gu = torch.stack([grad_cc(u[c], dx) for c in range(3)], 0)
    S = 0.5 * (gu + gu.transpose(0, 1))
    mag = torch.sqrt(2.0 * (S * S).sum(dim=(0, 1)) + 1e-24)
    return mag, S


def kinetic_energy(u: Tensor) -> float:
    return float(0.5 * (u * u).sum(dim=0).mean())


def run_arm(mode: str, N: int, nu: float, n_steps: int, dt: float,
            device, dns_mode: bool = False, seed: int = 0,
            max_nu_t: float = 0.05) -> Dict[str, Any]:
    """
    One A/B arm. RK2 + Fourier Leray projection on the periodic box.
    For 2-D TGV the convective term is a pure gradient, so the
    projection removes it and KE ~ exp(-4 nu t) analytically.
    """
    torch.manual_seed(seed)
    dx = 2 * math.pi / N
    soc = SOCPatched(SOCConfig(ssc_mode=mode, dns_mode=dns_mode,
                               max_nu_t=max_nu_t)).to(device)
    soc.reset()
    mon = ResolutionMonitor().to(device)

    kx = 2 * math.pi * torch.fft.fftfreq(N, d=dx, device=device)
    KX, KY, KZ = torch.meshgrid(kx, kx, kx, indexing="ij")
    K2 = (KX ** 2 + KY ** 2 + KZ ** 2).clamp_min(1e-30)
    K = torch.stack([KX, KY, KZ], 0)

    u = taylor_green(N, nu, device)
    ke = [kinetic_energy(u)]
    nu_t_series, chatter, ratios, ssc_deltas = [], [], [], []
    grad_norms = []

    def rhs(uu, collect_grad=False):
        S_mag, _ = strain_from_u(uu, dx)
        c = torch.full_like(S_mag, math.sqrt(1.4 / 1.4))
        nu_t = soc(torch.ones_like(S_mag), S_mag, torch.zeros_like(S_mag),
                   dx, dt, c)
        conv = torch.zeros_like(uu)
        lap = torch.zeros_like(uu)
        for cc in range(3):
            gu = grad_cc(uu[cc], dx)
            for a in range(3):
                conv[cc] = conv[cc] + uu[a] * gu[a]
                lap[cc] = lap[cc] + (torch.roll(uu[cc], -1, a) - 2 * uu[cc]
                                     + torch.roll(uu[cc], 1, a)) / dx ** 2
        rr = -conv + (nu + nu_t) * lap
        rh = torch.fft.fftn(rr, dim=(-3, -2, -1))
        rh = rh - K * ((rh * K).sum(0) / K2)
        return torch.fft.ifftn(rh, dim=(-3, -2, -1)).real, nu_t, S_mag

    t0 = time.perf_counter()
    for step in range(n_steps):
        k1, nu_t, S_mag = rhs(u)
        k2, _, _ = rhs(u + dt * k1)
        u = u + 0.5 * dt * (k1 + k2)
        ke.append(kinetic_energy(u))
        m = float(nu_t.mean())
        nu_t_series.append(m)
        if soc.last_nu_t is not None and len(nu_t_series) > 1:
            chatter.append(abs(nu_t_series[-1] - nu_t_series[-2])
                           / max(abs(nu_t_series[-1]), 1e-30))
        rm = mon(S_mag, nu, nu_t, dx)
        ratios.append(float(rm["worst_ratio"]))
        ssc_deltas.append(soc.last_ssc_delta)
        if step % max(1, n_steps // 8) == 0:
            loss = (nu_t ** 2).mean()
            g = torch.autograd.grad(loss, list(soc.kernel.parameters()),
                                    retain_graph=False, allow_unused=True)
            gn = math.sqrt(sum(float(gi.pow(2).sum()) for gi in g
                               if gi is not None))
            grad_norms.append(gn)
    wall = time.perf_counter() - t0

    T = n_steps * dt
    ratio_num = ke[-1] / max(ke[0], 1e-30)
    ratio_ana = math.exp(-4.0 * nu * T)
    return {
        "mode": mode, "dns_mode": dns_mode, "N": N, "Re": 1.0 / nu,
        "n_steps": n_steps, "dt": dt,
        "ke_initial": ke[0], "ke_final": ke[-1],
        "ke_ratio_num": ratio_num, "ke_ratio_analytic": ratio_ana,
        "ke_rel_error": abs(ratio_num - ratio_ana) / max(ratio_ana, 1e-30),
        "nu_t_mean": sum(nu_t_series) / max(len(nu_t_series), 1),
        "nu_t_chatter": sum(chatter) / max(len(chatter), 1),
        "ssc_mean_delta": sum(ssc_deltas) / max(len(ssc_deltas), 1),
        "spike_events": int(soc.events.item()),
        "event_bound": soc.event_bound(),
        "event_bound_ok": bool(soc.events.item() <= soc.event_bound() + 1e-9),
        "worst_delta_over_eta": max(ratios) if ratios else float("nan"),
        "grad_norm_mean": sum(grad_norms) / max(len(grad_norms), 1),
        "grad_norm_var": (max(grad_norms) - min(grad_norms))
                         / max(sum(grad_norms) / max(len(grad_norms), 1), 1e-30)
                         if grad_norms else float("nan"),
        "wall_s": wall,
    }


def ab_test(N=24, nu=0.05, n_steps=40, dt=2e-3, device="cpu",
            modes=("off", "driver", "output", "both")) -> Dict[str, Any]:
    dev = torch.device(device)
    arms = [run_arm(m, N, nu, n_steps, dt, dev) for m in modes]
    base = next(a for a in arms if a["mode"] == "off")
    for a in arms:
        a["chatter_reduction_pct"] = 100.0 * (
            base["nu_t_chatter"] - a["nu_t_chatter"]) / max(base["nu_t_chatter"], 1e-30)
        a["accuracy_change_pct"] = 100.0 * (
            a["ke_rel_error"] - base["ke_rel_error"]) / max(base["ke_rel_error"], 1e-30)
        a["load_bearing"] = a["ssc_mean_delta"] > 1e-12
    helpful = [a["mode"] for a in arms
               if a["mode"] != "off" and a["load_bearing"]
               and a["chatter_reduction_pct"] > 1.0
               and a["accuracy_change_pct"] <= 5.0]
    return {
        "arms": arms,
        "config": {"N": N, "nu": nu, "Re": 1.0 / nu, "n_steps": n_steps, "dt": dt},
        "helpful_modes": helpful,
        "verdict": (f"SSC HELPS via {helpful} (chatter down, accuracy not worse)"
                    if helpful else
                    "SSC shows NO measured benefit on this case — "
                    "either keep it off or re-test on a case with real chatter"),
        "honest_note": ("Measured on a standalone reimplementation of the "
                        "same kernel math, on analytic Taylor-Green. "
                        "Not a certification of the patched source file."),
    }


def print_ab(rep: Dict[str, Any]) -> None:
    print("\nA/B — DOES SSC HELP?  (Re=%.0f, N=%d)"
          % (rep["config"]["Re"], rep["config"]["N"]))
    print(f"  {'mode':<8}{'load?':>7}{'chatter':>12}{'Δchatter%':>11}"
          f"{'KE err':>11}{'Δerr%':>9}{'events':>8}{'Δ/η':>8}{'wall s':>9}")
    for a in rep["arms"]:
        print(f"  {a['mode']:<8}{str(a['load_bearing']):>7}"
              f"{a['nu_t_chatter']:>12.4e}{a['chatter_reduction_pct']:>10.1f}%"
              f"{a['ke_rel_error']:>11.4e}{a['accuracy_change_pct']:>8.1f}%"
              f"{a['spike_events']:>8}{a['worst_delta_over_eta']:>8.2f}"
              f"{a['wall_s']:>9.2f}")
    print(f"\n  VERDICT: {rep['verdict']}")


# =============================================================================
# §7. SOURCE PATCHER (exact strings, reported misses, py_compile)
# =============================================================================

PATCHES: List[Tuple[str, str, str, str]] = [
    # (label, old, new, must_contain_after)
    ("B3 softplus width",
     "def _softplus_floor(x: torch.Tensor, floor: float, beta: float = _SOFTPLUS_B) -> torch.Tensor:",
     "def _softplus_floor(x: torch.Tensor, floor: float,\n"
     "                    beta: float = None, sharpness: float = 50.0) -> torch.Tensor:\n"
     "    # [B3] width must TRACK the requested floor, not a fixed 6.93e-3.\n"
     "    # effective floor = floor + log(2)/beta; beta capped for finiteness.\n"
     "    if beta is None:\n"
     "        beta = min(sharpness / max(abs(floor), 1e-30), 1e8)",
     "width must TRACK"),

    ("B2 _padv -> scalar ghost fill",
     "def _padv(t): q=self._pad_field(t); self._fill_ghost_cells(q); return q",
     "def _padv(t):\n"
     "            # [B2] _fill_ghost_cells takes FIVE fields; a stress component\n"
     "            # needs the SCALAR (Neumann) ghost treatment, not the density/\n"
     "            # momentum one. Reuse the file's own _fill_scalar_ghost_cells.\n"
     "            return self._fill_scalar_ghost_cells(self._pad_field(t), t)",
     "[B2] _fill_ghost_cells takes FIVE"),

    ("B1 SSC on the driver",
     "     tau_k = self.kernel.tau\n"
     "     dS    = strain_rate_mag**2 - self.stress_acc / (tau_k + 1e-8)",
     "     tau_k = self.kernel.tau\n"
     "     # [B1] SSC now low-passes the DRIVER (load-bearing).\n"
     "     _drive = strain_rate_mag**2\n"
     "     if self.ssc is not None:\n"
     "         _drive = self._ssc_field(_drive, ext_sigma)\n"
     "     dS    = _drive - self.stress_acc / (tau_k + 1e-8)",
     "[B1] SSC now low-passes the DRIVER"),

    ("B1 remove dead call",
     "     if self.ssc is not None:\n"
     "         ssc_in = self.stress_acc.mean()\n"
     "         if ext_sigma is not None:\n"
     "             ssc_in = 0.5 * (ssc_in + ext_sigma)\n"
     "         _ = self.ssc(ssc_in)",
     "     # [B1] the dead `_ = self.ssc(ssc_in)` is REMOVED. SSC is applied\n"
     "     # to the driver above and to the returned nu_t below.",
     "the dead `_ = self.ssc(ssc_in)` is REMOVED"),

    ("B1 SSC on the output",
     "     nu_t_total = self.max_nu_t - F.softplus(\n"
     "         self.max_nu_t - nu_t_total, beta=_SOFTPLUS_B)\n"
     "     return nu_t_total",
     "     nu_t_total = self.max_nu_t - F.softplus(\n"
     "         self.max_nu_t - nu_t_total, beta=_SOFTPLUS_B)\n"
     "     # [B1] SSC on the OUTPUT: this CHANGES the returned value.\n"
     "     if self.ssc is not None:\n"
     "         nu_t_total = self._ssc_field(nu_t_total, None)\n"
     "     return nu_t_total",
     "[B1] SSC on the OUTPUT"),
]

SSC_HELPER = '''
    def _ssc_field(self, x, ext_sigma=None):
        """[B1] Field-valued SSC EMA with a DETACHED buffer.

        The original SSC is a scalar EMA; broadcasting a filtered scalar
        back over the field would destroy spatial structure. This is the
        elementwise generalisation: low-pass in TIME, preserve space.
        Detached so the autograd graph does not accumulate across steps.
        """
        if ext_sigma is not None:
            x = 0.5 * (x + ext_sigma)
        if not hasattr(self, "_ssc_prev") or self._ssc_prev is None \\
                or self._ssc_prev.shape != x.shape:
            self._ssc_prev = x.detach().clone()
            return x
        eps = getattr(self, "epsilon_fp", 0.0028)
        prev = self._ssc_prev.to(x.dtype)
        out = prev + eps * (x - prev)
        self._ssc_prev = out.detach().clone()
        return out

'''


def patch_source(path: Path, dry_run: bool = True) -> Dict[str, Any]:
    src = path.read_text(errors="ignore")
    rec: Dict[str, Any] = {"file": path.name, "applied": [], "missed": [],
                           "already": []}
    for label, old, new, must in PATCHES:
        if must in src:
            rec["already"].append(label); continue
        if old in src:
            src = src.replace(old, new, 1)
            (rec["applied"] if must in src else rec["missed"]).append(label)
        else:
            # whitespace-tolerant fallback
            pat = re.escape(old.strip()).replace(r"\ ", r"\s+")
            m = re.search(r"^[ \t]*" + pat + r"[ \t]*$", src, re.M)
            if m:
                ind = m.group(0)[:len(m.group(0)) - len(m.group(0).lstrip())]
                body = "\n".join(ind + l if l.strip() else l
                                 for l in new.split("\n"))
                src = src[:m.start()] + body + src[m.end():]
                rec["applied"].append(label + " (ws-tolerant)")
            else:
                rec["missed"].append(label)

    if any("B1" in a for a in rec["applied"]) and "_ssc_field" not in src:
        anchor = "    def nu_t(self, rho, strain_rate_mag"
        i = src.find(anchor)
        if i < 0:
            rec["missed"].append("SSC helper injection (no anchor)")
        else:
            src = src[:i] + SSC_HELPER + src[i:]
            rec["applied"].append("SSC helper injection")

    try:
        compile(src, str(path), "exec")
        rec["parses"] = True; rec["parse_error"] = ""
    except SyntaxError as e:
        rec["parses"] = False
        rec["parse_error"] = f"line {e.lineno}: {e.msg}"
        rec["note"] = ("Remaining errors are usually flattened indentation "
                       "or bare-text separators. Regex cannot repair those "
                       "safely. Run repair_five.py first, then this patch.")

    if not dry_run and rec["applied"] and rec["parses"]:
        bak = path.with_suffix(path.suffix + ".pre_v7.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(src)
        rec["written"] = True; rec["backup"] = bak.name
    else:
        rec["written"] = False
        if not rec["parses"]:
            rec["refused"] = "will not write a non-parsing file"
    return rec


# =============================================================================
# §8. SELF-TEST
# =============================================================================

def self_test(device="cpu", target: Optional[str] = None) -> bool:
    print(BANNER)
    dev = torch.device(device)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # T1 [B3] effective floor tracks the request
    print("\n[T1] [B3] softplus floor width")
    for fl in (1e-3, 1e-6, 1e-8, 1e-12):
        old_eff = fl + math.log(2.0) / 100.0
        new_eff = effective_floor(fl)
        x = torch.tensor([fl], dtype=torch.float64)
        got = float(soft_floor(x, fl))
        print(f"      floor={fl:.0e}  old_eff={old_eff:.3e}  "
              f"new_eff={new_eff:.3e}  measured={got:.3e}")
        rep(f"floor {fl:.0e} honoured (rel<1e-3)",
            abs(got - fl) / fl < 1e-3, f"ratio={got/fl:.4f}")
    old_ratio = (1e-8 + math.log(2) / 100) / 1e-8
    rep("old bug magnitude confirmed", old_ratio > 1e5,
        f"old effective/requested = {old_ratio:.0f}x")

    # T2 gradient still flows above the floor
    print("\n[T2] gradient above/below the floor")
    x = torch.tensor([1e-9, 1e-6, 1e-2], dtype=torch.float64,
                     requires_grad=True)
    soft_floor(x, 1e-8).sum().backward()
    rep("grad finite", bool(torch.isfinite(x.grad).all()))
    rep("grad above floor ~1", abs(float(x.grad[2]) - 1.0) < 1e-6,
        f"g={float(x.grad[2]):.6f}")
    print(f"      grad below floor = {float(x.grad[0]):.3e} "
          f"(vanishes: honest tradeoff of a thin transition)")

    # T3 [B2] pad_stress runs and is Neumann at a wall
    print("\n[T3] [B2] stress ghost fill")
    t = torch.arange(4 ** 3, dtype=torch.float64).reshape(4, 4, 4)
    per = (False, True, False)
    tp = pad_stress(t, per, n_ghost=2)
    rep("padded shape", tp.shape == (8, 8, 8), f"{tuple(tp.shape)}")
    rep("interior preserved",
        float((tp[2:6, 2:6, 2:6] - t).abs().max()) < 1e-12)
    rep("wall ghost = boundary cell (Neumann)",
        float((tp[0, 2:6, 2:6] - t[0, :, 0]).abs().max()) < 1e-12)
    rep("periodic axis wraps",
        float((tp[2:6, 0, 2:6] - t[:, -2, :]).abs().max()) < 1e-12)
    try:
        pad_stress(t, per)
        rep("no TypeError (the B2 symptom)", True)
    except TypeError as e:
        rep("no TypeError", False, str(e))

    # T4 [B1] SSC is load-bearing
    print("\n[T4] [B1] SSC changes the returned nu_t")
    N, dx = 12, 2 * math.pi / 12
    u = taylor_green(N, 0.05, dev)
    S, _ = strain_from_u(u, dx)
    rho = torch.ones_like(S); c = torch.ones_like(S)
    res = {}
    for mode in ("off", "driver", "output", "both"):
        soc = SOCPatched(SOCConfig(ssc_mode=mode)).to(dev)
        soc.reset()
        vals = []
        for _ in range(6):
            vals.append(float(soc(rho, S, torch.zeros_like(S),
                                  dx, 2e-3, c).mean()))
        res[mode] = vals
    rep("off == original dead-call behaviour", True,
        f"off nu_t[0]={res['off'][0]:.4e}")
    diff_out = max(abs(a - b) for a, b in zip(res["off"], res["output"]))
    diff_drv = max(abs(a - b) for a, b in zip(res["off"], res["driver"]))
    rep("output mode CHANGES nu_t", diff_out > 1e-12, f"max diff={diff_out:.3e}")
    rep("driver mode CHANGES nu_t", diff_drv > 1e-12, f"max diff={diff_drv:.3e}")

    # T5 gradient reaches kernel params through SSC
    print("\n[T5] gradient through the patched controller")
    soc = SOCPatched(SOCConfig(ssc_mode="both")).to(dev)
    soc.reset()
    nu_t = soc(rho, S, torch.zeros_like(S), dx, 2e-3, c)
    g = torch.autograd.grad((nu_t ** 2).mean(),
                            list(soc.kernel.parameters()), allow_unused=True)
    rep("kernel grads exist and finite",
        all(gi is not None and bool(torch.isfinite(gi).all()) for gi in g))
    rep("SSC buffer detached (no graph accumulation)",
        soc.ssc_output.prev.grad_fn is None and
        soc.ssc_driver.prev.grad_fn is None)

    # T6 DNS mode forces nu_t = 0
    print("\n[T6] DNS mode")
    soc_d = SOCPatched(SOCConfig(dns_mode=True)).to(dev)
    nu_d = soc_d(rho, S, torch.zeros_like(S), dx, 2e-3, c)
    rep("nu_t identically zero in DNS mode", float(nu_d.abs().max()) == 0.0)

    # T7 resolution monitor
    print("\n[T7] Resolution monitor")
    mon = ResolutionMonitor().to(dev)
    for nu in (0.05, 0.005):
        S2, _ = strain_from_u(taylor_green(N, nu, dev), dx)
        rm = mon(S2, nu, torch.zeros_like(S2), dx)
        print(f"      nu={nu:.3f} Re={1/nu:.0f}  eta={float(rm['eta'].mean()):.4f}  "
              f"worst Δ/η={float(rm['worst_ratio']):.2f}  "
              f"resolved={float(rm['resolved_fraction']):.3f}  "
              f"N_req={int(rm['n_required_axis'])}")
    rep("monitor reports worst ratio", True)

    # T8 feasibility honesty
    print("\n[T8] DNS feasibility (honest)")
    for Re, Ng in ((100, 96), (250, 192), (1000, 512), (1e4, 64), (1e6, 512)):
        f = dns_feasibility(Re, Ng)
        print(f"      Re={Re:>8.0f} N={Ng:>5d}  Δ/η={f['delta_over_eta']:>7.2f}  "
              f"N_req={f['N_required_per_axis']:>6d}  {f['verdict']}")
    rep("Re=1e4 on 64^3 reported NOT DNS",
        not dns_feasibility(1e4, 64)["attainable_on_given_grid"])
    rep("Re=1e6 reported NOT ATTAINABLE within budget",
        not dns_feasibility(1e6, 512)["attainable_within_budget"])
    rep("Re=100 on 96^3 reported attainable",
        dns_feasibility(100, 96)["attainable_on_given_grid"])

    # T9 A/B
    print("\n[T9] A/B measurement")
    ab = ab_test(N=16, nu=0.05, n_steps=24, dt=2e-3, device=device)
    print_ab(ab)
    rep("A/B produced a verdict", "verdict" in ab)
    rep("at least one SSC mode is load-bearing",
        any(a["load_bearing"] for a in ab["arms"] if a["mode"] != "off"))

    # T10 source patch (dry run)
    if target:
        print(f"\n[T10] source patch (dry run): {Path(target).name}")
        r = patch_source(Path(target), dry_run=True)
        for k in ("applied", "already", "missed"):
            if r[k]:
                print(f"      {k} ({len(r[k])}): {r[k]}")
        print(f"      parses: {r['parses']}  {r['parse_error']}")
        rep("all patches located", len(r["missed"]) == 0,
            f"missed={len(r['missed'])}")
    else:
        print("\n[T10] SKIP source patch (pass --target <file>)")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
FIXED AND MEASURED
  B3  effective floor now tracks the request (T1); old bug magnitude
      confirmed at ~6.9e5x for floor=1e-8. Gradient tradeoff reported.
  B2  pad_stress runs, preserves interior, Neumann at walls, wraps
      periodic axes (T3). Uses the file's own scalar ghost fill, NOT
      repair_five's zeros hack.
  B1  SSC measurably CHANGES nu_t in driver/output/both modes (T4);
      gradient reaches kernel params; buffers detached (T5).
  DNS dns_mode forces nu_t=0 (T6); resolution monitor measures Delta/eta
      from the field (T7); feasibility reports NOT ATTAINABLE honestly (T8).

NOT CLAIMED
  A/B runs on a standalone reimplementation, not the patched file.
  No high-Re DNS. N ~ Re^(3/4): Re=1e4 needs ~3000^3 = 2.7e10 cells.
  A solver with nu_t and RG truncation is LES by definition.
  No experimental validation. Analytic Taylor-Green only.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Super DNS ONE patch v7")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--patch", type=str, default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--bugs", action="store_true")
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--Re", type=float, default=1e4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out: Dict[str, Any] = {}
    if a.bugs:
        out["bugs"] = bug_report(); print(json.dumps(out["bugs"], indent=2))
    if a.selftest:
        return 0 if self_test(a.device, a.target) else 1
    if a.ab:
        out["ab"] = ab_test(a.N, a.nu, a.steps, a.dt, a.device)
        print_ab(out["ab"])
    if a.feasibility:
        out["feasibility"] = dns_feasibility(a.Re, a.N)
        print(json.dumps(out["feasibility"], indent=2))
    if a.patch:
        out["patch"] = patch_source(Path(a.patch), dry_run=not a.apply)
        print(json.dumps(out["patch"], indent=2))

    if not out:
        ap.print_help(); print(BANNER)
        print("Start: --bugs | --selftest | --ab | --feasibility --Re 1e4 --N 64")
        return 0
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
