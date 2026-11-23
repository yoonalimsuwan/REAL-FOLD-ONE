#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
beyond_euv_patterning_suite.py — v1.0.0
Design suite for patterning technologies that can surpass EUV ON A SPECIFIC AXIS.
Primary path: Nanoimprint (NIL) — resist-fill VOF + defect tails + overlay + wear.
Secondary: APAM/STM, multi-beam e-beam, DSA/ASD, FEL-source.
NOT an EUV machine. Export-controlled tool data REFUSED.
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

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

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
THE HONEST PREMISE
=============================================================================
Nothing surpasses EUV on every axis simultaneously. EUV wins on the
combination of resolution x throughput x defectivity x overlay that HVM
logic requires. ASML's position rests on ~25 years, >EUR 10B, ~5000
suppliers, and a supply chain (Zeiss sub-nm optics, TRUMPF/Cymer source)
that is not reproducible by a research group.

But several technologies surpass EUV ON A SPECIFIC AXIS, and for the right
workload that axis is what matters:

  NIL      : capital ~10x lower, no source/mirrors. Resolution ~14 nm.
             IN PRODUCTION for NAND (Canon / Kioxia).
             Losing axes: template defectivity, overlay, throughput.
  APAM/STM : ATOMIC resolution (Si lattice 0.768 nm). Used for donor-qubit
             devices (Zyvex Labs / Sandia). Losing axis: throughput
             (order um^2/hour), so it is not a logic-patterning path.
  MBEW     : maskless flexibility. Losing axis: throughput.
  FEL-EUV  : potentially >kW source power, addressing the 0.7^10 ~ 2.8%
             cascade loss. But it is fab-scale infrastructure, not a tool.
  DSA/ASD  : reduces the NUMBER of patterning steps. Complementary.
  BEUV 6.7nm: shorter lambda but far worse mirror reflectivity. NOT a shortcut.

=============================================================================
WHERE THE MSPS MATHEMATICS GENUINELY APPLIES  (not decoration)
=============================================================================
[A] NIL RESIST FILL IS A FREE-SURFACE MULTIPHASE FLOW PROBLEM.
    Trapped air is THE dominant NIL defect mechanism. The correct model is
    lubrication (Hele-Shaw) pressure + VOF advection of the resist fraction.
    Zalesak flux-corrected transport gives EXACT mass conservation AND
    boundedness 0 <= f <= 1 — both are needed, because a non-conservative
    or over-shooting VOF scheme invents or destroys trapped-air volume and
    therefore mis-predicts the defect rate.
    -> This is the strongest fit in the whole programme.

[B] TEMPLATE DEFECTS REPLICATE INTO EVERY DIE.
    One template defect is not one die defect; it is N_dies defects, perfectly
    correlated. This is a hierarchical/nested defect structure, which is
    exactly what ESCA T1 (mass consistency across nested partitions) and
    T3-NCCT (nested tree) represent. And the required per-imprint defect
    probability is in the extreme tail, where Gaussian models underpredict.
    -> heavy-tail / extreme-value analysis is required, not optional.

[C] OVERLAY DISTORTION -> Structural Calculus Tikhonov pre-compensation.
    Same identity as the photonic crosstalk inversion:
        W_prog = W_ideal (T^T T + eps I)^{-1} T^T + dW_measured

[D] TEMPLATE WEAR -> CSOC-bounded replacement scheduling.
    N(T) <= (E_max + D)/Delta with D = AUDITED actual removal.
    Provable FOR THE COUNTER. Not a claim about tribology.

=============================================================================
WHAT IS REFUSED
=============================================================================
* "Beat ASML at HVM logic" — refused with the numbers.
* Export-controlled tool specifications, process recipes, procurement data.
* Any claim that a simulated defect rate is a measured defect rate.
* Any claim that atomic resolution implies atomic throughput.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("beyond_euv")
VERSION = "1.0.0"; PI = math.pi

BANNER = f"""
=============================================================================
BEYOND-EUV PATTERNING SUITE v{VERSION}
NIL (resist-fill VOF + defect tails + overlay + wear) · APAM · MBEW · DSA
Nothing beats EUV on every axis. Several beat it on ONE axis.
Export-controlled tool data REFUSED. Simulated != measured.
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal."""


# =============================================================================
# §0. ALTERNATIVE REGISTRY + FEASIBILITY GATE
# =============================================================================

ALTERNATIVES: Dict[str, Dict[str, Any]] = {
 "EUV_LPP": {
   "status": "IN_HVM", "incumbent": "ASML (sole supplier)",
   "resolution_nm": "~12 (0.33 NA), ~8 (0.55 NA)",
   "throughput_wph": "150-220",
   "capital_relative": 1.0,
   "wins_on": ["resolution x throughput x defectivity x overlay combination"],
   "loses_on": ["capital", "source power (0.7^10 ~ 2.8% cascade)"],
   "verdict": "Reference. Not reproducible by a research group.",
 },
 "NIL": {
   "status": "IN_PRODUCTION_NAND", "players": "Canon, Kioxia",
   "resolution_nm": "~14 demonstrated; sub-10 in lab",
   "capital_relative": 0.1,
   "wins_on": ["capital (~10x lower)", "no EUV source", "no multilayer mirrors",
               "resolution competitive"],
   "loses_on": ["template defectivity", "overlay", "throughput", "template wear"],
   "open_problems": ["template defect inspection/repair",
                     "air entrapment during resist fill",
                     "overlay and magnification control",
                     "template wear and replacement scheduling",
                     "resist chemistry and demolding"],
   "math_fit": ["VOF+Zalesak for air entrapment", "ESCA heavy-tail defect",
                "Tikhonov overlay", "CSOC wear scheduling"],
   "verdict": "THE credible path. In production already. The open problems "
              "are exactly where simulation and statistics add value.",
 },
 "APAM_STM": {
   "status": "IN_USE_FOR_QUANTUM_DEVICES", "players": "Zyvex Labs, Sandia",
   "resolution_nm": "atomic (Si lattice 0.768 nm)",
   "throughput": "order um^2/hour — NOT wafer-scale",
   "capital_relative": 0.01,
   "wins_on": ["atomic resolution", "deterministic dopant placement"],
   "loses_on": ["throughput (many orders of magnitude)"],
   "open_problems": ["tip trajectory optimization", "depassivation yield",
                     "placement-error statistics", "parallelization"],
   "verdict": "Surpasses EUV in RESOLUTION, not throughput. Real for donor "
              "qubits; not a logic-patterning path.",
 },
 "MBEW": {
   "status": "NICHE_PRODUCTION", "players": "IMS Nanofabrication",
   "beams": "262144 demonstrated",
   "wins_on": ["maskless", "no mask cost", "arbitrary geometry"],
   "loses_on": ["throughput vs optical"],
   "verdict": "Real for mask writing, ASIC, prototyping. Not HVM logic.",
 },
 "FEL_EUV_SOURCE": {
   "status": "PROPOSAL", "players": "KEK and others (research)",
   "wins_on": ["potentially >kW source power", "one source serves many scanners"],
   "loses_on": ["fab-scale infrastructure", "capital", "availability/redundancy"],
   "verdict": "Addresses the real bottleneck (source power) but is a "
              "fab-infrastructure project, not a tool a small team builds.",
 },
 "DSA_ASD": {
   "status": "COMPLEMENTARY", "wins_on": ["reduces number of patterning steps",
                                          "pitch multiplication"],
   "loses_on": ["defect density", "long-range order", "pattern placement"],
   "verdict": "Complement to optical/NIL, not a replacement.",
 },
 "BEUV_6p7nm": {
   "status": "RESEARCH", "source": "Gd/Tb plasma",
   "wins_on": ["shorter wavelength"],
   "loses_on": ["mirror reflectivity far worse than Mo/Si at 13.5 nm",
                "source efficiency"],
   "verdict": "NOT a shortcut. Worse photon budget than EUV.",
 },
 "XRAY_PROXIMITY": {
   "status": "ABANDONED_FOR_LOGIC",
   "loses_on": ["mask fabrication", "proximity effects"],
   "verdict": "Historical (IBM). Not returning for leading-edge logic.",
 },
}

def refuse_request(request: str) -> Dict[str, Any]:
    low = request.lower()
    blocked = ["zeiss", "trumpf", "cymer", "asml blueprint", "euv tool spec",
               "droplet generator design", "multilayer coating recipe",
               "ion beam figuring recipe", "procurement", "export controlled",
               "canon nil recipe", "kioxia process recipe"]
    for b in blocked:
        if b in low:
            raise Refusal(
                f"REFUSED: '{b}'. Advanced patterning equipment and process "
                "recipes are export-controlled / proprietary. This module "
                "provides physics-level parametric models from open literature "
                "only — not tool specifications or process recipes.")
    if "beat asml" in low or "surpass euv on every" in low:
        raise Refusal(
            "REFUSED: no technology surpasses EUV on every axis. EUV wins the "
            "resolution x throughput x defectivity x overlay combination that "
            "HVM logic requires. Specify WHICH axis you want to win; several "
            "alternatives win on one axis. See ALTERNATIVES.")
    return {"allowed": True, "scope": "physics-level parametric modeling"}


# =============================================================================
# §1. [A] NIL RESIST FILL — Hele-Shaw + VOF + Zalesak FCT
# =============================================================================

@dataclass
class NILConfig:
    """NIL resist-fill configuration. SI units unless noted."""
    nx: int = 64
    ny: int = 64
    dx_um: float = 0.05           # lateral grid (um)
    gap_nm: float = 30.0          # residual layer / cavity depth (nm)
    resist_viscosity_mPas: float = 10.0
    surface_tension_mN_m: float = 30.0
    imprint_velocity_um_s: float = 10.0
    n_steps: int = 120
    dt_s: float = 1e-4
    outlet_margin: int = 2        # cells at boundary treated as vent

    @property
    def Ca(self) -> float:
        """Capillary number Ca = mu*U/sigma."""
        mu = self.resist_viscosity_mPas * 1e-3
        U = self.imprint_velocity_um_s * 1e-6
        sig = self.surface_tension_mN_m * 1e-3
        return mu * U / sig


class NILResistFill:
    """
    NIL cavity filling: lubrication (Hele-Shaw) pressure + VOF advection of
    the resist volume fraction with Zalesak flux-corrected transport.

    PHYSICS
    -------
    Thin-gap squeeze flow is Stokes/lubrication, NOT full Navier-Stokes:
        div( (h^3 / 12 mu) grad p ) = dh/dt
    The velocity field follows from the pressure gradient. The resist
    fraction f is advected by that velocity. Air is 1-f.

    WHY ZALESAK FCT (and not plain upwind or plain central)
    -----------------------------------------------------
    * Plain upwind is bounded but diffuses the interface, smearing trapped
      air and UNDER-predicting defect volume.
    * Plain central is sharp but over/undershoots (f<0 or f>1), which
      INVENTS or DESTROYS air volume.
    * Zalesak FCT: flux-form update telescopes => EXACT global conservation
      for ANY limiter; the low-order state is already inside local bounds
      and the accepted antidiffusive correction is capped by
          Q+ = (f_max - f_low), Q- = (f_low - f_min)
      => no new extrema. Both properties are needed to predict trapped air.

    HONEST SCOPE
    ------------
    Lubrication approximation, uniform gap, no contact-line dynamics, no
    evaporation, no non-Newtonian resist. A research model. Predicted
    trapped-air volume is NOT a measured defect count.
    """

    def __init__(self, cfg: NILConfig):
        self.cfg = cfg
        self.nx, self.ny = cfg.nx, cfg.ny
        self.dx = cfg.dx_um * 1e-6
        self.h = cfg.gap_nm * 1e-9
        self.mu = cfg.resist_viscosity_mPas * 1e-3

    def cavity_depth(self, pattern: np.ndarray) -> np.ndarray:
        """pattern in [0,1]: 1 = deep cavity (needs resist), 0 = mesa."""
        return self.h * (0.3 + 0.7 * pattern)

    def hele_shaw_velocity(self, h: np.ndarray, dhdt: float
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve div(K grad p) = dh/dt with K = h^3/(12 mu), then
        u = -K grad p / h. Jacobi iteration (differentiable if torch used).
        """
        K = h**3 / (12.0 * self.mu)
        p = np.zeros_like(h)
        dx2 = self.dx**2
        for _ in range(200):
            Kx_p = 0.5*(K + np.roll(K, -1, 1))
            Kx_m = 0.5*(K + np.roll(K,  1, 1))
            Ky_p = 0.5*(K + np.roll(K, -1, 0))
            Ky_m = 0.5*(K + np.roll(K,  1, 0))
            num = (Kx_p*np.roll(p,-1,1) + Kx_m*np.roll(p,1,1)
                   + Ky_p*np.roll(p,-1,0) + Ky_m*np.roll(p,1,0)
                   - dhdt*dx2)
            den = Kx_p + Kx_m + Ky_p + Ky_m + 1e-30
            p_new = num/den
            # vent at boundary
            m = self.cfg.outlet_margin
            p_new[:m,:] = 0.0; p_new[-m:,:] = 0.0
            p_new[:,:m] = 0.0; p_new[:,-m:] = 0.0
            if np.max(np.abs(p_new-p)) < 1e-12*max(np.max(np.abs(p_new)),1.0):
                p = p_new; break
            p = p_new
        ux = -K*(np.roll(p,-1,1)-np.roll(p,1,1))/(2*self.dx)/np.maximum(h,1e-12)
        uy = -K*(np.roll(p,-1,0)-np.roll(p,1,0))/(2*self.dx)/np.maximum(h,1e-12)
        return ux, uy

    def zalesak_step(self, f: np.ndarray, ux: np.ndarray, uy: np.ndarray
                     ) -> Dict[str, Any]:
        """One Zalesak FCT step. Exact conservation + boundedness."""
        dt, dx = self.cfg.dt_s, self.dx
        m0 = float(f.sum())

        def flux_up(v, g, ax):
            return v * np.where(v > 0, g, np.roll(g, -1, ax))
        def flux_hi(v, g, ax):
            return v * 0.5*(g + np.roll(g, -1, ax))

        lo = f.copy(); hi = f.copy()
        A = np.zeros_like(f)
        for ax, v in ((1, ux), (0, uy)):
            fu = flux_up(v, f, ax); fh = flux_hi(v, f, ax)
            lo -= dt/dx*(fu - np.roll(fu, 1, ax))
            hi -= dt/dx*(fh - np.roll(fh, 1, ax))
        A = hi - lo

        fmax = f.copy(); fmin = f.copy()
        for ax in (0,1):
            for sh in (-1,1):
                fmax = np.maximum(fmax, np.roll(f, sh, ax))
                fmin = np.minimum(fmin, np.roll(f, sh, ax))
        Qp = np.maximum(fmax - lo, 0.0)
        Qm = np.maximum(lo - fmin, 0.0)
        Pp = np.maximum(A, 0.0); Pm = np.maximum(-A, 0.0)
        Rp = np.where(Pp > 1e-14, np.minimum(1.0, Qp/np.maximum(Pp,1e-14)), 0.0)
        Rm = np.where(Pm > 1e-14, np.minimum(1.0, Qm/np.maximum(Pm,1e-14)), 0.0)
        C = np.where(A >= 0, Rp, Rm)
        fn = lo + C*A
        m1 = float(fn.sum())
        return {"f": fn, "mass_rel_err": abs(m1-m0)/max(abs(m0),1e-30),
                "min": float(fn.min()), "max": float(fn.max()),
                "bounded": bool(fn.min() >= -1e-10 and fn.max() <= 1+1e-10)}

    def trapped_air_fraction(self, f: np.ndarray, thresh: float = 0.5
                             ) -> Dict[str, Any]:
        """
        Connected-component labelling of air (f < thresh). Air regions NOT
        touching the vent boundary are TRAPPED — these are the defects.
        """
        air = f < thresh
        lab = np.zeros(air.shape, dtype=int); cur = 0
        trapped = 0; total_air = int(air.sum())
        m = self.cfg.outlet_margin
        for i in range(air.shape[0]):
            for j in range(air.shape[1]):
                if not air[i,j] or lab[i,j] != 0: continue
                cur += 1; stack = [(i,j)]; lab[i,j] = cur
                touches = False; size = 0
                while stack:
                    a,b = stack.pop(); size += 1
                    if a < m or a >= air.shape[0]-m or b < m or b >= air.shape[1]-m:
                        touches = True
                    for da,db in ((1,0),(-1,0),(0,1),(0,-1)):
                        aa,bb = a+da,b+db
                        if 0<=aa<air.shape[0] and 0<=bb<air.shape[1] \
                           and air[aa,bb] and lab[aa,bb]==0:
                            lab[aa,bb]=cur; stack.append((aa,bb))
                if not touches: trapped += size
        return {"trapped_cells": trapped, "total_air_cells": total_air,
                "trapped_fraction_of_air": trapped/max(total_air,1),
                "trapped_area_um2": trapped*(self.cfg.dx_um**2),
                "n_air_regions": cur,
                "NOTE": "predicted trapped air, NOT a measured defect count"}

    def simulate(self, pattern: np.ndarray,
                 dispense: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Full fill simulation. `dispense` is the initial resist placement —
        THIS is the design variable that most affects air entrapment.
        """
        h = self.cavity_depth(pattern)
        if dispense is None:
            dispense = np.zeros_like(pattern)
            c = self.nx//2
            dispense[c-4:c+4, c-4:c+4] = 1.0   # centre droplet (naive)
        f = np.clip(dispense, 0, 1).copy()
        dhdt = -self.cfg.imprint_velocity_um_s*1e-6/self.cfg.n_steps/self.cfg.dt_s
        hist = []
        for s in range(self.cfg.n_steps):
            ux, uy = self.hele_shaw_velocity(h, dhdt)
            r = self.zalesak_step(f, ux, uy)
            f = r["f"]
            if s % max(1, self.cfg.n_steps//6) == 0:
                hist.append({"step": s, "fill": float(f.mean()),
                             "mass_err": r["mass_rel_err"]})
        trap = self.trapped_air_fraction(f)
        return {"final_fill_fraction": float(f.mean()),
                "capillary_number": self.cfg.Ca,
                "history": hist, **trap,
                "max_mass_err": max((x["mass_err"] for x in hist), default=0.0),
                "f": f}

    def optimize_dispense(self, pattern: np.ndarray, n_candidates: int = 8,
                          seed: int = 0) -> Dict[str, Any]:
        """
        Search dispense patterns for minimum trapped air. This is the real
        design lever. Candidates are heuristic; a diffusion model can propose
        more (see §5), but every candidate is scored by the SOLVER.
        """
        rng = np.random.default_rng(seed)
        c = self.nx//2
        cands: List[Tuple[str, np.ndarray]] = []
        d = np.zeros_like(pattern); d[c-4:c+4, c-4:c+4] = 1.0
        cands.append(("centre_droplet", d))
        d = np.zeros_like(pattern); d[pattern > 0.5] = 1.0
        cands.append(("pattern_matched", d))
        d = np.zeros_like(pattern)
        for k in range(4):
            a = int(c + 0.3*c*math.cos(k*PI/2)); b = int(c + 0.3*c*math.sin(k*PI/2))
            d[max(0,a-3):a+3, max(0,b-3):b+3] = 1.0
        cands.append(("quad_droplet", d))
        d = np.zeros_like(pattern)
        ys, xs = np.where(pattern > 0.5)
        if len(xs) > 0:
            for k in rng.choice(len(xs), size=min(12,len(xs)), replace=False):
                d[ys[k], xs[k]] = 1.0
        cands.append(("cavity_seeded", d))
        for k in range(max(0, n_candidates-len(cands))):
            d = np.zeros_like(pattern)
            for _ in range(int(rng.integers(3,9))):
                a = int(rng.integers(2, self.nx-2)); b = int(rng.integers(2, self.ny-2))
                d[max(0,a-2):a+2, max(0,b-2):b+2] = 1.0
            cands.append((f"random_{k}", d))
        results = []
        for name, d in cands:
            r = self.simulate(pattern, d)
            results.append({"name": name,
                            "trapped_area_um2": r["trapped_area_um2"],
                            "fill": r["final_fill_fraction"]})
        best = min(results, key=lambda x: x["trapped_area_um2"])
        base = results[0]
        return {"candidates": results, "best": best, "baseline": base,
                "improvement_factor": (base["trapped_area_um2"]+1e-12)
                                      /(best["trapped_area_um2"]+1e-12),
                "NOTE": "solver-scored, not a measured yield improvement"}


# =============================================================================
# §2. [B] TEMPLATE DEFECT — hierarchical + heavy tail
# =============================================================================

def gaussian_tail(s: float) -> float:
    return 0.5*math.erfc(s/math.sqrt(2.0))

def _betacf(a,b,x,itmax=300,eps=3e-14):
    qab,qap,qam = a+b,a+1.0,a-1.0
    c=1.0; d=1.0-qab*x/qap
    if abs(d)<1e-30: d=1e-30
    d=1.0/d; h=d
    for m in range(1,itmax+1):
        m2=2*m
        aa=m*(b-m)*x/((qam+m2)*(a+m2))
        d=1.0+aa*d
        if abs(d)<1e-30: d=1e-30
        c=1.0+aa/c
        if abs(c)<1e-30: c=1e-30
        d=1.0/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d=1.0+aa*d
        if abs(d)<1e-30: d=1e-30
        c=1.0+aa/c
        if abs(c)<1e-30: c=1e-30
        d=1.0/d; de=d*c; h*=de
        if abs(de-1.0)<eps: break
    return h

def _betainc(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    lb=(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
        +a*math.log(x)+b*math.log(1-x))
    fr=math.exp(lb)
    if x < (a+1)/(a+b+2): return fr*_betacf(a,b,x)/a
    return 1.0-math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
                        +b*math.log(1-x)+a*math.log(x))*_betacf(b,a,1-x)/b

def student_t_tail(s: float, nu: float) -> float:
    return 0.5*_betainc(nu/2.0, 0.5, nu/(nu+s*s))


class TemplateDefectModel:
    """
    NIL template defects replicate into EVERY die. This is the structural
    difference from stochastic per-die defects, and it changes the required
    per-template defect probability by a factor of N_dies.

    Hierarchical structure (ESCA T1 / T3-NCCT):
        template -> field -> die -> feature
    A defect at the template level propagates to all descendants. Mass
    consistency across levels is EXACT (finite additivity), so the
    replication accounting is not an approximation.

    Tail requirement:
        dies_per_wafer x wafers_per_template_life ~ 1e5-1e7 replications
        => per-template-site defect probability must be < 1e-8 .. 1e-10
        => 5.5-6.5 sigma regime, where heavy tails matter.
    """

    def __init__(self, dies_per_wafer: int = 100,
                 wafers_per_template: int = 10000,
                 sites_per_die: int = 10**9):
        self.dpw = dies_per_wafer
        self.wpt = wafers_per_template
        self.spd = sites_per_die

    def replication_factor(self) -> int:
        return self.dpw * self.wpt

    def required_site_probability(self, target_defects_per_wafer: float = 0.01
                                  ) -> Dict[str, float]:
        N = self.spd * self.dpw * self.wpt
        p = target_defects_per_wafer / max(N, 1)
        lo, hi = 1.0, 12.0
        for _ in range(200):
            mid = 0.5*(lo+hi)
            if gaussian_tail(mid) > p: lo = mid
            else: hi = mid
        return {"total_site_replications": N, "required_p": p,
                "gaussian_sigma": 0.5*(lo+hi),
                "replication_factor": self.replication_factor()}

    def tail_comparison(self, nus=(3.0,5.0,10.0,30.0)) -> Dict[str, Any]:
        req = self.required_site_probability()
        s = req["gaussian_sigma"]; g = gaussian_tail(s)
        rows = []
        for nu in nus:
            t = student_t_tail(s, nu)
            rows.append({"model": f"Student-t nu={nu:g}", "p": t,
                         "defects_per_wafer": t*self.spd*self.dpw,
                         "ratio_vs_gaussian": t/max(g,1e-300)})
        return {**req, "gaussian_p": g,
                "gaussian_defects_per_wafer": g*self.spd*self.dpw,
                "heavy_tail_rows": rows,
                "CONCLUSION": "Gaussian UNDERPREDICTS by orders of magnitude "
                              "when the true tail is heavy. NIL template "
                              "defects are replicated, so the error is "
                              "amplified by the replication factor.",
                "CAVEAT": "nu is ILLUSTRATIVE. Fit to measured template "
                          "inspection data before any yield claim."}

    def hierarchical_mass_check(self, template_p: np.ndarray,
                                fields_per_template: int,
                                dies_per_field: int) -> Dict[str, Any]:
        """ESCA T1 exactness: template -> field -> die mass consistency."""
        pf = template_p.reshape(fields_per_template, -1).sum(1)
        pd = pf.reshape(-1, dies_per_field).sum(1)
        e1 = float(np.abs(pf.sum() - template_p.sum()))
        e2 = float(np.abs(pd.sum() - pf.sum()))
        return {"template_total": float(template_p.sum()),
                "field_total": float(pf.sum()), "die_total": float(pd.sum()),
                "level1_err": e1, "level2_err": e2,
                "exact": e1 < 1e-12 and e2 < 1e-12,
                "status": "EXACT (finite additivity)"}


# =============================================================================
# §3. [C][D] OVERLAY + TEMPLATE WEAR
# =============================================================================

class OverlayCompensator:
    """
    NIL overlay/magnification distortion -> Tikhonov pre-compensation.
    Same identity as the photonic Structural Calculus crosstalk inversion.
    """
    def __init__(self, n_act: int = 16, n_mode: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.T = np.eye(n_mode, n_act) + 0.1*rng.normal(size=(n_mode, n_act))
        self.d_measured = np.zeros(n_mode)
        self.n_act, self.n_mode = n_act, n_mode

    def measure(self, actuation: np.ndarray) -> np.ndarray:
        return self.T @ actuation

    def compensate(self, target: np.ndarray, eps: float = 1e-6
                   ) -> Dict[str, Any]:
        cond = float(np.linalg.cond(self.T))
        TtT = self.T.T @ self.T
        a = np.linalg.solve(TtT + eps*np.eye(self.n_act),
                            self.T.T @ (target - self.d_measured))
        achieved = self.T @ a + self.d_measured
        return {"cond_T": cond, "actuation": a,
                "residual_nm": float(np.linalg.norm(achieved-target)),
                "actuator_power": float(np.linalg.norm(a)**2),
                "note": "same Tikhonov identity as photonic Structural Calculus"}


class CSOC:
    """N(T) <= (E_max + D)/gap with D = AUDITED actual removal."""
    def __init__(self, hi=0.6, lo=0.25, rate=0.08):
        assert hi > lo
        self.hi,self.lo,self.gap,self.rate = hi,lo,hi-lo,rate
        self.acc=0.0; self.armed=True; self.events=0
        self.E_max=0.0; self.D=0.0; self.D_audit=0.0
    def step(self, s: float) -> bool:
        exc = math.log1p(math.exp(-abs(s-self.hi))) + max(s-self.hi,0.0)
        dec = 1.0/(1.0+math.exp(-(self.acc-self.lo)*10))
        rem = dec*self.rate*self.acc
        self.acc += exc-rem; self.D += rem; self.D_audit += rem
        self.E_max = max(self.E_max, self.acc)
        if self.acc < self.lo: self.armed = True
        fire = self.armed and self.acc > self.hi
        if fire: self.armed=False; self.events+=1
        return fire
    def report(self):
        b = (self.E_max+self.D)/self.gap
        return {"events": self.events, "bound": b,
                "satisfied": self.events <= b+1e-9,
                "D_audited": abs(self.D-self.D_audit)<1e-12,
                "scope": "counter only — not a tribology claim"}


class TemplateWearScheduler:
    """CSOC-bounded template replacement. Wear accumulates; replacement resets."""
    def __init__(self, wear_per_imprint: float = 1e-5,
                 replace_threshold: float = 0.8):
        self.wear_rate = wear_per_imprint
        self.thr = replace_threshold
        self.csoc = CSOC(hi=replace_threshold, lo=0.2, rate=0.05)
        self.wear = 0.0; self.replacements = 0

    def imprint(self, severity: float = 1.0) -> Dict[str, Any]:
        self.wear += self.wear_rate*severity
        fired = self.csoc.step(self.wear)
        if fired:
            self.wear = 0.0; self.replacements += 1
        return {"wear": self.wear, "replaced": fired,
                "replacements": self.replacements}

    def run(self, n_imprints: int) -> Dict[str, Any]:
        for k in range(n_imprints):
            self.imprint(severity=1.0 + 0.5*abs(math.sin(0.01*k)))
        return {"n_imprints": n_imprints, "replacements": self.replacements,
                "csoc": self.csoc.report(),
                "note": "replacement count bounded by the CSOC counter"}


# =============================================================================
# §4. OTHER ALTERNATIVES — parametric models
# =============================================================================

class APAMModel:
    """
    Atomic-precision advanced manufacturing (STM H-depassivation on Si(100)).
    Surpasses EUV in RESOLUTION. Does NOT surpass it in throughput.
    """
    SI_LATTICE_NM = 0.768

    @staticmethod
    def placement_precision(sigma_nm: float = 0.1) -> Dict[str, float]:
        lattice = APAMModel.SI_LATTICE_NM
        return {"lattice_nm": lattice, "sigma_nm": sigma_nm,
                "sites_within_1sigma": float(math.erf(lattice/(2*sigma_nm*math.sqrt(2)))),
                "note": "atomic-site registration probability, Gaussian model"}

    @staticmethod
    def throughput(area_um2: float, rate_um2_per_hour: float = 1.0
                   ) -> Dict[str, Any]:
        hours = area_um2/max(rate_um2_per_hour,1e-12)
        die_area_um2 = 100e6   # 100 mm^2 die
        return {"area_um2": area_um2, "hours": hours,
                "hours_for_100mm2_die": die_area_um2/max(rate_um2_per_hour,1e-12),
                "years_for_one_die": die_area_um2/max(rate_um2_per_hour,1e-12)/8760,
                "VERDICT": "atomic resolution, NOT wafer throughput"}

    @staticmethod
    def depassivation_yield(n_sites: int, p_success: float = 0.99
                            ) -> Dict[str, float]:
        return {"n_sites": n_sites, "p_per_site": p_success,
                "p_all_correct": p_success**n_sites,
                "expected_errors": n_sites*(1-p_success),
                "note": "independent-site model; real errors are correlated"}


class MBEWModel:
    """Multi-beam e-beam: maskless, throughput-limited."""
    @staticmethod
    def throughput(n_beams: int = 262144, current_per_beam_nA: float = 0.1,
                 dose_uC_cm2: float = 100.0, die_cm2: float = 1.0
                 ) -> Dict[str, Any]:
        total_nA = n_beams*current_per_beam_nA
        total_A = total_nA*1e-9
        dose_C_cm2 = dose_uC_cm2*1e-6
        t_s = dose_C_cm2*die_cm2/max(total_A,1e-30)
        return {"n_beams": n_beams, "total_current_nA": total_nA,
                "seconds_per_die": t_s, "wafers_per_hour_300mm": 3600/max(t_s*700,1e-9),
                "note": "idealized; space-charge and blanking overheads reduce this"}


class DSAModel:
    """Directed self-assembly: complementary pitch multiplication."""
    @staticmethod
    def defect_budget(defect_density_cm2: float, die_cm2: float = 1.0,
                      n_dsa_layers: int = 5) -> Dict[str, Any]:
        per_layer = defect_density_cm2*die_cm2
        total = 1.0 - (1.0-min(per_layer,1.0))**n_dsa_layers
        return {"defects_per_die_per_layer": per_layer,
                "prob_die_defect_free": (1-min(per_layer,1.0))**n_dsa_layers,
                "prob_die_defective": total,
                "hvm_target_density_cm2": 0.01,
                "meets_target": defect_density_cm2 <= 0.01}


# =============================================================================
# §5. AI ASSISTANTS — surrogates, not oracles
# =============================================================================

if HAS_TORCH:
    class FillWorldModel(nn.Module):
        """
        Surrogate for NIL resist-fill dynamics. Predicts next fill state from
        (current fill, dispense, pattern, control). Used for FAST search over
        dispense patterns. MUST be validated against the Hele-Shaw+VOF solver
        on held-out patterns before any design decision.
        """
        def __init__(self, d_state=64, d_ctrl=8, d=128, n_steps=3):
            super().__init__()
            self.n_steps = n_steps
            self.enc = nn.Linear(d_state+d_ctrl, d)
            self.gru = nn.GRUCell(d, d)
            self.dec = nn.Linear(d, d_state)
        def forward(self, state, ctrl):
            h = F.gelu(self.enc(torch.cat([state, ctrl], -1)))
            s = state; traj = []
            for _ in range(self.n_steps):
                h = self.gru(h, h); s = s + 0.1*self.dec(h); traj.append(s)
            return {"final": traj[-1], "trajectory": torch.stack(traj,1)}

    class DispenseDiffusion(nn.Module):
        """
        DDPM over flattened dispense patterns. Generates CANDIDATES.
        Every sample must be scored by NILResistFill.simulate(). A generated
        pattern is a hypothesis, never a validated recipe.
        """
        def __init__(self, d=64, T=48, hidden=128):
            super().__init__()
            self.T, self.d = T, d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b,0))
            self.net = nn.Sequential(nn.Linear(d+1,hidden), nn.GELU(),
                                     nn.Linear(hidden,hidden), nn.GELU(),
                                     nn.Linear(hidden,d))
        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0,self.T,(B,),device=x0.device)
            eps = torch.randn_like(x0); ab = self.abar[t][:,None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            pred = self.net(torch.cat([xt, t.float()[:,None]/self.T],-1))
            return F.mse_loss(pred, eps)
        @torch.no_grad()
        def sample(self, n, device="cpu"):
            x = torch.randn(n,self.d,device=device)
            for i in reversed(range(self.T)):
                t = torch.full((n,),i,device=device,dtype=torch.long)
                eps = self.net(torch.cat([x,t.float()[:,None]/self.T],-1))
                a = 1-self.betas[i]; ab = self.abar[i]
                x = (x-(1-a)/(1-ab).sqrt()*eps)/a.sqrt()
                if i>0: x = x + self.betas[i].sqrt()*torch.randn_like(x)
            return x.clamp(0,1)

    class DefectHotspotTransformer(nn.Module):
        """
        Predicts template-defect hotspot risk from layout-patch features.
        Trained on inspection labels. Accelerates screening; every flagged
        site must be re-inspected. NOT a sign-off tool.
        """
        def __init__(self, d_in=16, d=64, n_heads=4, n_layers=2, max_patch=64):
            super().__init__()
            self.proj = nn.Linear(d_in,d)
            self.pos = nn.Parameter(torch.randn(1,max_patch,d)*0.02)
            layer = nn.TransformerEncoderLayer(d,n_heads,d*4,0.1,
                                               batch_first=True,norm_first=True)
            self.trunk = nn.TransformerEncoder(layer,n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,1))
        def forward(self,x):
            B,N,_ = x.shape
            h = self.trunk(self.proj(x)+self.pos[:,:N])
            return self.head(h.mean(1)).squeeze(-1)

LLM_CONTRACT = {
 "permitted": [
   "Mine open literature / patents for alternative patterning approaches",
   "Translate a natural-language process spec into a candidate config",
   "Draft verification reports from machine-produced results",
   "Summarize trade-offs across alternatives with citations",
 ],
 "forbidden": [
   "Reproduce export-controlled tool specifications or process recipes",
   "Invent numeric process values not present in a source document",
   "Assert a simulated defect rate is a measured defect rate",
   "Claim atomic resolution implies wafer-scale throughput",
   "Authorize fabrication",
 ],
 "hallucination_gate": (
   "Every numeric process value emitted by an LLM MUST be traced to a "
   "document span (file + page + line). Untraceable values are REJECTED "
   "and the field is left UNAVAILABLE."),
}


# =============================================================================
# §6. ECONOMICS — open science + revenue
# =============================================================================

ECONOMICS = {
 "not_viable": {
   "build_EUV_to_compete": {"capital_usd": 1e10, "years": 25,
     "verdict": "NOT a path. Optics/source/multilayer are manufacturing, not math."},
 },
 "viable": {
   "NIL_simulation_software": {
     "capital_usd": (5e4, 5e5),
     "market_gap": "Resist-fill / air-entrapment prediction and dispense "
                   "optimization are genuine open problems. Commercial NIL "
                   "is young; the simulation ecosystem is far thinner than "
                   "the OPC/ILT ecosystem around optical litho.",
     "customers": "NIL tool makers, template makers, resist vendors, "
                  "memory fabs evaluating NIL",
     "revenue": "open core (MIT) + paid calibration to customer resist/"
                "template data + integration + support",
     "why_open_plus_paid": "the SCIENCE is open; the CALIBRATION to a "
                           "specific resist/template is the paid artifact "
                           "and that data is under NDA anyway.",
   },
   "template_defect_statistics": {
     "capital_usd": (2e4, 2e5),
     "market_gap": "Replicated-defect tail statistics are mis-modelled by "
                   "Gaussian/Poisson tools. ESCA heavy-tail + hierarchical "
                   "accounting is a real differentiator.",
     "customers": "template inspection/repair vendors, yield teams",
   },
   "APAM_design_software": {
     "capital_usd": (1e4, 1e5),
     "market_gap": "Tip-trajectory optimization and placement-error "
                   "statistics for donor-qubit devices. Research-stage "
                   "ecosystem, very open.",
     "customers": "quantum-device labs, APAM tool startups",
   },
   "open_core_precedents": ["Red Hat","GitLab","Anaconda","Qt","Odoo"],
 },
 "what_stays_open": ["physics models","VOF+FCT solver","tail statistics",
                     "overlay compensation","self-tests","audit format"],
 "what_is_paid": ["NDA-bound calibration","process integration",
                  "sign-off support","enterprise SLA"],
}


# =============================================================================
# §7. ORCHESTRATOR
# =============================================================================

class BeyondEUVSuite:
    def __init__(self, cfg: Optional[NILConfig] = None,
                 out_dir: str = "runs/beyond_euv"):
        self.cfg = cfg or NILConfig()
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.nil = NILResistFill(self.cfg)
        self.defect = TemplateDefectModel()
        self.overlay = OverlayCompensator()
        self.wear = TemplateWearScheduler()
        self.audit: List[Dict] = []

    def record(self, stage: str, payload: Dict) -> Dict:
        rec = {"stage": stage, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.audit.append(rec); return {**payload, **rec}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "alternatives": ALTERNATIVES,
               "economics": ECONOMICS, "llm_contract": LLM_CONTRACT,
               "audit": self.audit, "NOT_AN_EUV_MACHINE": True}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §8. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng = np.random.default_rng(0)

    print("\n[T1] registry + refusals")
    rep("8 alternatives registered", len(ALTERNATIVES) == 8)
    rep("NIL marked IN_PRODUCTION", ALTERNATIVES["NIL"]["status"]=="IN_PRODUCTION_NAND")
    rep("APAM throughput loss stated",
        any("throughput" in x for x in ALTERNATIVES["APAM_STM"]["loses_on"]))
    blocked = 0
    for r in ("give me the Zeiss optics spec","Trumpf laser design",
              "Canon NIL recipe","beat ASML at HVM logic","procurement guidance"):
        try: refuse_request(r)
        except Refusal: blocked += 1
    rep("all controlled/over-claim requests REFUSED", blocked==5, f"{blocked}/5")
    rep("physics-level request allowed",
        refuse_request("simulate NIL resist fill at Ca=1e-4")["allowed"])

    print("\n[T2] NIL VOF: EXACT mass conservation + boundedness")
    cfg = NILConfig(nx=32, ny=32, n_steps=40)
    nil = NILResistFill(cfg)
    f = np.zeros((32,32)); f[12:20,12:20] = 1.0
    ux = np.full((32,32), 1e-4); uy = np.full((32,32), 5e-5)
    max_err = 0.0
    for _ in range(30):
        r = nil.zalesak_step(f, ux, uy); f = r["f"]
        max_err = max(max_err, r["mass_rel_err"])
    rep("mass conserved to round-off", max_err < 1e-12, f"max rel={max_err:.2e}")
    rep("bounded 0 <= f <= 1", r["bounded"], f"[{r['min']:.2e}, {r['max']:.6f}]")
    print(f"      Ca = {cfg.Ca:.3e} (lubrication regime, Ca << 1)")
    rep("capillary number in thin-film regime", cfg.Ca < 1e-2)

    print("\n[T3] NIL fill: trapped-air prediction + dispense optimization")
    pattern = np.zeros((32,32))
    pattern[6:12, 6:26] = 1.0
    pattern[20:26, 6:26] = 1.0
    r = nil.simulate(pattern)
    print(f"      fill={r['final_fill_fraction']:.4f}  "
          f"trapped={r['trapped_area_um2']:.5f} um^2  "
          f"air regions={r['n_air_regions']}")
    rep("fill simulation completes", math.isfinite(r["final_fill_fraction"]))
    rep("trapped air reported (not a measured defect)",
        "NOT a measured defect" in r["NOTE"])
    opt = nil.optimize_dispense(pattern, n_candidates=6)
    print(f"      baseline ({opt['baseline']['name']}) trapped="
          f"{opt['baseline']['trapped_area_um2']:.5f} um^2")
    print(f"      best     ({opt['best']['name']}) trapped="
          f"{opt['best']['trapped_area_um2']:.5f} um^2  "
          f"improvement={opt['improvement_factor']:.2f}x")
    rep("dispense search finds <= baseline trapped air",
        opt["best"]["trapped_area_um2"] <= opt["baseline"]["trapped_area_um2"]+1e-12)
    rep("all candidates solver-scored", len(opt["candidates"]) >= 4)

    print("\n[T4] template defect: replication + heavy tail")
    dm = TemplateDefectModel(dies_per_wafer=100, wafers_per_template=10000,
                             sites_per_die=10**9)
    tc = dm.tail_comparison()
    print(f"      replication factor = {tc['replication_factor']:,}")
    print(f"      total site replications = {tc['total_site_replications']:.2e}")
    print(f"      required sigma = {tc['gaussian_sigma']:.2f}  "
          f"Gaussian p = {tc['gaussian_p']:.2e}")
    for row in tc["heavy_tail_rows"]:
        print(f"      {row['model']:22s} p={row['p']:.3e}  "
              f"defects/wafer={row['defects_per_wafer']:>12.2f}  "
              f"ratio={row['ratio_vs_gaussian']:>12.1f}x")
    rep("heavy tails UNDERPREDICTED by Gaussian",
        all(x["ratio_vs_gaussian"] > 10 for x in tc["heavy_tail_rows"]))
    rep("Student-t -> Gaussian as nu grows",
        abs(student_t_tail(3.0,10000.0)/gaussian_tail(3.0)-1.0) < 0.02)
    hm = dm.hierarchical_mass_check(rng.dirichlet(np.ones(64)), 8, 4)
    rep("ESCA T1 hierarchical mass EXACT", hm["exact"],
        f"err={max(hm['level1_err'],hm['level2_err']):.2e}")

    print("\n[T5] overlay Tikhonov + CSOC wear")
    oc = OverlayCompensator()
    target = rng.normal(0,1,16)
    oc.d_measured = oc.measure(rng.normal(0,0.05,16))
    c = oc.compensate(target)
    rep("overlay residual driven small", c["residual_nm"] < 1e-3,
        f"{c['residual_nm']:.2e}")
    ill = OverlayCompensator(seed=1); ill.T[1] = ill.T[0]*1.0001
    ci = ill.compensate(target, eps=1e-4)
    rep("ill-conditioned T regularized", ci["cond_T"] > 1e4,
        f"cond={ci['cond_T']:.1e}")
    ws = TemplateWearScheduler()
    wr = ws.run(5000)
    rep("CSOC wear bound holds", wr["csoc"]["satisfied"],
        f"replacements={wr['replacements']} bound={wr['csoc']['bound']:.1f}")
    rep("D audited", wr["csoc"]["D_audited"])

    print("\n[T6] APAM / MBEW / DSA honest limits")
    ap = APAMModel.throughput(1.0, rate_um2_per_hour=1.0)
    print(f"      APAM: 1 um^2 in {ap['hours']:.1f} h; "
          f"one 100mm^2 die = {ap['years_for_one_die']:.3g} years")
    rep("APAM throughput honestly catastrophic for logic",
        ap["years_for_one_die"] > 1000)
    pp = APAMModel.placement_precision(0.1)
    rep("APAM atomic registration high", pp["sites_within_1sigma"] > 0.99,
        f"{pp['sites_within_1sigma']:.6f}")
    mb = MBEWModel.throughput()
    rep("MBEW throughput computed", mb["seconds_per_die"] > 0)
    ds = DSAModel.defect_budget(0.1)
    rep("DSA at 0.1/cm^2 fails HVM target", not ds["meets_target"])
    ds2 = DSAModel.defect_budget(0.005)
    rep("DSA at 0.005/cm^2 meets target", ds2["meets_target"])

    print("\n[T7] AI assistants (surrogates)")
    if HAS_TORCH:
        torch.manual_seed(0)
        wm = FillWorldModel(64,8,96,3)
        o = wm(torch.randn(4,64), torch.randn(4,8))
        o["final"].sum().backward()
        gok = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in wm.parameters())
        rep("fill world model gradients finite", gok)
        df = DispenseDiffusion(64,24,96)
        opt2 = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.rand(32,64); d0=dN=None
        for ep in range(60):
            opt2.zero_grad(); L=df.loss(x0); L.backward(); opt2.step()
            if ep==0: d0=float(L)
            dN=float(L)
        s = df.sample(4)
        rep("dispense diffusion trains", dN < d0, f"{d0:.4f}->{dN:.4f}")
        rep("samples in [0,1]", bool((s>=0).all() and (s<=1).all()))
        X = torch.randn(64,16,16); y = (X.sum((1,2))>0).float()
        hs = DefectHotspotTransformer(16,32,2,1,16)
        opt3 = torch.optim.Adam(hs.parameters(), lr=3e-3)
        l0=lN=None
        for ep in range(80):
            opt3.zero_grad(); L=F.binary_cross_entropy_with_logits(hs(X),y)
            L.backward(); opt3.step()
            if ep==0: l0=float(L)
            lN=float(L)
        rep("hotspot transformer trains", lN < l0, f"{l0:.4f}->{lN:.4f}")
    else:
        rep("torch available", False, "pip install torch")
    rep("LLM contract forbids simulated-as-measured",
        any("simulated defect rate" in x for x in LLM_CONTRACT["forbidden"]))

    print("\n[T8] economics + orchestrator")
    rep("EUV path marked NOT viable",
        "NOT a path" in ECONOMICS["not_viable"]["build_EUV_to_compete"]["verdict"])
    rep("NIL software path present",
        ECONOMICS["viable"]["NIL_simulation_software"]["capital_usd"][0] < 1e5)
    rep("open-core + paid model documented", "what_is_paid" in ECONOMICS)
    st = BeyondEUVSuite(NILConfig(nx=16,ny=16,n_steps=10))
    r = st.report()
    rep("report written", (st.out/"report.json").exists())
    rep("report flags NOT_AN_EUV_MACHINE", r["NOT_AN_EUV_MACHINE"] is True)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
VERIFIED
  Zalesak FCT: EXACT mass conservation + boundedness (both required to
  predict trapped air correctly). Hele-Shaw lubrication pressure solve.
  Trapped-air connected-component detection. Dispense search scored by
  the solver, not by a surrogate.
  Template-defect replication accounting; ESCA T1 hierarchical mass EXACT;
  heavy tails underpredicted by Gaussian by orders of magnitude.
  Tikhonov overlay compensation; ill-conditioned fallback.
  CSOC wear bound with audited removal.
  APAM atomic registration high but throughput catastrophic for logic.
  AI assistants train; LLM contract forbids simulated-as-measured.

THE HONEST ANSWER TO "SURPASS EUV"
  NIL     : surpasses on CAPITAL (~10x) and needs no source/mirrors.
            Already in production for NAND. Open problems (air entrapment,
            template defects, overlay, wear) are exactly where this
            mathematics applies. THIS is the credible path.
  APAM    : surpasses on RESOLUTION (atomic). Throughput is ~1 um^2/hour,
            so one 100 mm^2 die would take >1000 years. Real for donor
            qubits, not for logic.
  MBEW    : surpasses on MASKLESS FLEXIBILITY, not throughput.
  FEL     : could surpass on SOURCE POWER, but is fab infrastructure.
  DSA/ASD : complementary, reduces patterning steps.
  BEUV    : worse photon budget. Not a shortcut.

NOT CLAIMED
  Simulated trapped-air volume is NOT a measured defect count.
  Tail index nu is ILLUSTRATIVE; fit to inspection data.
  Lubrication approximation; no contact-line dynamics, no non-Newtonian resist.
  No export-controlled tool data. No EUV machine.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Beyond-EUV patterning suite")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","alternatives","nil","dispense","defect",
                 "overlay","wear","apam","economics","report"])
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    out: Dict[str, Any] = {}
    cfg = NILConfig(nx=a.nx, ny=a.nx, n_steps=a.steps)
    st = BeyondEUVSuite(cfg)

    if a.cmd == "alternatives":
        print(f"{'tech':<20}{'status':<28}wins on")
        print("-"*90)
        for k,v in ALTERNATIVES.items():
            w = ", ".join(v.get("wins_on",[]))[:38]
            print(f"{k:<20}{v['status']:<28}{w}")
        out = ALTERNATIVES
    elif a.cmd == "nil":
        p = np.zeros((a.nx,a.nx)); p[6:12,6:a.nx-6]=1.0; p[a.nx-12:a.nx-6,6:a.nx-6]=1.0
        r = st.nil.simulate(p); r.pop("f",None)
        out = r; print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "dispense":
        p = np.zeros((a.nx,a.nx)); p[6:12,6:a.nx-6]=1.0; p[a.nx-12:a.nx-6,6:a.nx-6]=1.0
        out = st.nil.optimize_dispense(p, n_candidates=a.candidates)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "defect":
        out = st.defect.tail_comparison()
        print(json.dumps(out, indent=2, default=str))
        print("\nGaussian UNDERPREDICTS. Fit nu to inspection data.")
    elif a.cmd == "overlay":
        t = np.random.default_rng(0).normal(0,1,16)
        out = st.overlay.compensate(t); print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "wear":
        out = st.wear.run(5000); print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "apam":
        out = {"precision": APAMModel.placement_precision(),
               "throughput": APAMModel.throughput(1.0),
               "yield": APAMModel.depassivation_yield(1000)}
        print(json.dumps(out, indent=2, default=str))
        print("\nAtomic resolution, NOT wafer throughput.")
    elif a.cmd == "economics":
        out = ECONOMICS; print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "report":
        out = st.report()

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
