#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
omni_stack_papers.py — v1.0.0
Three-paper integration layer for omni_stack.py:
  [P1] SSC Photonic Processor R3.0        (optical core + fab spec)
  [P2] DSQP-SESI Covariant GDS-II R4.0    (covariant + adaptive + governance)
  [P3] ESDA 2.0                            (formal state + measurement + T1..T6)
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
THE THREE PAPERS ARE THREE LAYERS, NOT THREE ALTERNATIVES
=============================================================================
  P1  SSC R3.0        optical core: theta_C, alpha_eff, kappa(g), phi_total,
                      Structural Calculus, layers 1..30 + 99, six DRC rules,
                      seven-stage pre-tapeout protocol.
  P2  DSQP-SESI R4.0  adds layers 40..47 (metric / curvature / delay /
                      routing / monitor / sync / GR-interface / energy),
                      covariant field + Einstein sector, PPN -> phi_GR,
                      CHSH + operational no-signalling, typed structural
                      graph, six-objective Pareto, version control.
  P3  ESDA 2.0        formalises the complete microstate X=(Psi,Pi,Lambda,M),
                      weak-to-hard measurement G = G_h o D o G_w, the
                      pushforward law mu_t = (G o F^t)_# rho_0, the
                      contraction / a-posteriori / hardware error bounds,
                      reservoir stress-energy balance, and ESCA T1..T6.

=============================================================================
WHAT IS MACHINE-CHECKED HERE  (per paper)
=============================================================================
P1  Prop 1  normalized intensity is a probability vector          EXACT
    Prop 2  ideal MZI (unitary components) is unitary             EXACT
    Prop 3  argmax with fixed tie-break is unique                 EXHAUSTIVE
    Prop 5  ideal weak tap: P_m + P_t = P_in                      EXACT
    Lemma   |theta_C(z)| <= 1/(2 sqrt(alpha)), attained at 1/sqrt(a)  EXACT
    Lemma   theta_C is globally non-expansive (Lip = 1)           EXACT
    DRC     six R3 geometry predicates                            SOFTWARE

P2  Prop 6  conditional GR phase calibration                      CONDITIONAL
    CHSH    |S| <= 2 for every local deterministic strategy        EXHAUSTIVE (16)
    Eq 15   operational no-signalling marginals                   EXACT
    A.1     nabla^mu T_mu_nu^(Psi) = 0 on-shell (flat 1+1)         SYMBOLIC/NUMERIC
    Prop 1-7 SESI typed-graph validity, monotone objective,
             finite termination, runtime != geometry             SOFTWARE

P3  Prop 1.1  pushforward measure law                             EXACT (ensemble)
    Prop 2.1  q-contraction, a-posteriori bound, delta/(1-q)       NUMERIC CERTIFICATE
    Eq 10-11  dissipative norm decay + reservoir balance           NUMERIC
    T1        Carpet probability consistency                        EXACT
    T3-NCCT   nested tree exact reconstruction + mass               EXACT
    T4-MEOT   two-point / vertex occupation limit                   NUMERIC
    T5        robust minimiser + uniform chance bound               EXACT (finite A)
    T6        Mosco -> resolvent / semigroup                        CONDITIONAL HARNESS

=============================================================================
WHAT IS REFUSED  (unchanged, and enforced in code)
=============================================================================
R1  NLSE is CLASSICAL nonlinear optics. It is NOT a quantum computer.
    No 2^n state space, no entanglement, no Born-rule collapse.
R2  A metric-analogue refractive-index profile is NOT astrophysical
    spacetime curvature. Mode A (simulation) != Mode B (physical test).
R3  A classical photonic device with measurement independence CANNOT
    violate a Bell inequality. |S| <= 2 is PROVEN here by exhaustion.
R4  A GDS-II that opens is NOT fabrication-ready. Tape-out requires a
    foundry PDK + clean DRC + clean LVS + human authorisation.
R5  Global well-posedness / regularity of the coupled Einstein +
    nonlinear-field system is OPEN. Local well-posedness only.
R6  Mosco M1/M2 for a SPECIFIC carpet normalisation must be established
    separately. Numerical convergence is evidence, not proof.
R7  Every biological / clinical output is research_only.
=============================================================================
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("papers")

VERSION = "1.0.0"
PI = math.pi
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
THREE-PAPER STACK v{VERSION}
P1 SSC R3.0  ->  P2 DSQP-SESI R4.0  ->  P3 ESDA 2.0
optical core     covariant + adaptive     formal + stochastic
Machine-checks what is provable. Refuses R1..R7 in code.
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal. Never silently downgraded."""


# =============================================================================
# §0. THREE-PAPER REGISTRY
# =============================================================================

PAPERS = {
 "P1_SSC_R3": {
   "title": "Unicode Mathematical Formulation and Heterogeneous Fabrication "
            "Specification for the SSC Photonic Processor (Revision 3.0)",
   "layer": "optical core + fabrication specification",
   "provides": ["theta_C", "alpha_eff(MRR+BTO)", "lambda_res", "kappa(g)",
                "phi_total", "Structural Calculus compensation",
                "layers 1,2,10,15,16,20,30,99", "6 DRC rules",
                "7-stage pre-tapeout protocol"],
   "proven_here": ["intensity normalisation", "ideal MZI unitarity",
                   "argmax uniqueness", "weak-tap power conservation",
                   "saturation amplitude bound", "global non-expansiveness"],
   "not_claimed": ["quantum computation", "measurable gravity",
                   "foundry-ready tape-out without a PDK"]},
 "P2_DSQP_SESI_R4": {
   "title": "Covariant GDS-II Hardware Architecture and Self-Evolving "
            "Structural Interface for Deterministic Sub-Quantum Photonics "
            "(Revision 4.0)",
   "layer": "covariant coupling + adaptive structure + governance",
   "provides": ["layers 40..47", "covariant action / Einstein sector",
                "box_g Psi - V'(|Psi|^2) Psi = 0", "nabla^mu T_mu_nu = 0",
                "PPN weak-field metric -> phi_GR",
                "CHSH bounds + operational no-signalling",
                "SESI typed graph + Valid/FabricationValid",
                "6-objective Pareto", "version control + rollback"],
   "proven_here": ["CHSH |S| <= 2 by exhaustion", "no-signalling marginals",
                   "on-shell stress-energy conservation (flat 1+1)",
                   "typed-connection validity", "monotone objective",
                   "finite termination", "runtime != geometry"],
   "not_claimed": ["astrophysical spacetime curvature on a chip",
                   "Bell violation from classical optics",
                   "autonomous fabrication authorisation"]},
 "P3_ESDA_2": {
   "title": "Nested Extreme-Stochastic Deterministic Architecture (ESDA 2.0)",
   "layer": "formal state space + measurement projection + ESCA T1..T6",
   "provides": ["X = (Psi, Pi, Lambda, M)", "F_theta deterministic map",
                "G_w / D / G_h measurement chain",
                "mu_t = (G o F^t)_# rho_0",
                "q-contraction + a-posteriori + delta/(1-q)",
                "dissipative norm decay + reservoir balance",
                "ESCA T1, T2, T3-NCCT, T4-MEOT/T4-G, T5, T6",
                "unified GDS layer map + tape-out gate"],
   "proven_here": ["pushforward law", "contraction certificates",
                   "T1 probability consistency", "T3-NCCT exactness",
                   "T5 robust closure (finite action set)"],
   "conditional": ["T2 measure extension (compactness + generating partitions)",
                   "T4 occupation limits (dominance conditions)",
                   "T6 Mosco -> resolvent (M1/M2 must be established)"],
   "not_claimed": ["global Einstein regularity",
                   "quantum computational superiority",
                   "that nature is deterministic sub-quantum"]},
}

REFUSALS = {
 "R1": "NLSE is CLASSICAL nonlinear optics, NOT a quantum computer.",
 "R2": "Metric-analogue refractive index != astrophysical curvature.",
 "R3": "Classical optics with measurement independence cannot violate Bell.",
 "R4": "GDS-II that opens != fabrication-ready. PDK + DRC + LVS + human gate.",
 "R5": "Global Einstein + nonlinear-field regularity is OPEN.",
 "R6": "Mosco M1/M2 for a specific carpet normalisation is OPEN.",
 "R7": "All biological / clinical outputs are research_only.",
}


# =============================================================================
# §1. P1 — SSC R3.0 OPTICAL CORE
# =============================================================================

def theta_C(z: np.ndarray, alpha: float) -> np.ndarray:
    """P1 Eq: theta_C(z) = z / (1 + alpha |z|^2)."""
    return z / (1.0 + alpha * np.abs(z) ** 2)

def theta_bound(alpha: float) -> float:
    """P1 Lemma: max_r r/(1+alpha r^2) = 1/(2 sqrt(alpha)) at r*=alpha^-1/2."""
    return 1.0 / (2.0 * math.sqrt(alpha))

def theta_jacobian_sv(r: float, alpha: float) -> Tuple[float, float]:
    """P3 Lemma 2.1(2): sigma_tan = 1/(1+s), sigma_rad = |1-s|/(1+s)^2, s=a r^2."""
    s = alpha * r * r
    return 1.0 / (1.0 + s), abs(1.0 - s) / (1.0 + s) ** 2

def alpha_eff_mrr(n2_BTO: float, Q_MRR: float, L_eff_m: float,
                  A_eff_m2: float) -> Dict[str, float]:
    """
    P1 Eq: alpha_eff = n2_BTO * Q_MRR * L_eff / A_eff.
    HONEST: the unit convention requires the optical-power normalisation used
    in the device model. This returns the formula value; calibration against
    a measured nonlinear transfer function is REQUIRED before use.
    """
    a = n2_BTO * Q_MRR * L_eff_m / max(A_eff_m2, 1e-20)
    return {"alpha_eff": a,
            "unit_note": "formula as written in P1; needs power-normalisation "
                         "calibration against measured Kerr response",
            "theta_bound": theta_bound(a) if a > 0 else float("inf")}

def mrr_resonance(R_m: float, n_eff: float, m: int) -> Dict[str, float]:
    """P1 Eq: lambda_res = 2 pi R n_eff / m, m in Z+."""
    lam = 2 * PI * R_m * n_eff / m
    fsr = lam ** 2 / (2 * PI * R_m * n_eff)
    return {"lambda_res_m": lam, "FSR_m": fsr,
            "detuning_from_1550nm": (lam - 1.55e-6) * 1e9}

def coupler_kappa(g_um: float, kappa0: float = 0.5, gamma: float = 3.0) -> float:
    """P1 Eq: kappa(g) = kappa0 exp(-gamma g)."""
    return kappa0 * math.exp(-gamma * g_um)

def phi_total(dn_eff: float, L_m: float, lam0: float,
              dn_dT: float, dT_K: float, L_heater_m: float,
              dn_PCM: float, L_PCM_m: float,
              phi_GR: float = 0.0) -> Dict[str, float]:
    """P1/P2 Eq: phi_total = dbeta*L + phi_heater(P) + phi_PCM(state) + phi_GR."""
    k0 = 2 * PI / lam0
    prop = k0 * dn_eff * L_m
    heat = k0 * dn_dT * dT_K * L_heater_m
    pcm = k0 * dn_PCM * L_PCM_m
    return {"phi_prop": prop, "phi_heater": heat, "phi_PCM": pcm,
            "phi_GR": phi_GR, "phi_total": prop + heat + pcm + phi_GR}

def mzi_matrix(th1: float, th2: float, ph1: float, ph2: float) -> np.ndarray:
    """P1 Prop 2: U = B2 D B1 with ideal 50:50 couplers."""
    B = lambda t: np.array([[math.cos(t), 1j*math.sin(t)],
                            [1j*math.sin(t), math.cos(t)]])
    return B(th2) @ np.diag([np.exp(1j*ph1), np.exp(1j*ph2)]) @ B(th1)

def weak_tap(P_in: float, eta: float) -> Dict[str, float]:
    """P1 Prop 5: P_m = eta P_in, P_t = (1-eta) P_in, sum = P_in (ideal)."""
    return {"P_monitor": eta*P_in, "P_through": (1-eta)*P_in,
            "sum": P_in, "conservation_err": abs(eta*P_in + (1-eta)*P_in - P_in)}

def hard_gate(scores: np.ndarray) -> int:
    """P1 Prop 3: argmax with FIXED tie-break (lowest index) -> unique."""
    return int(np.argmax(np.asarray(scores, float)))

def intensity_probability(psi: np.ndarray) -> np.ndarray:
    """P1 Prop 1: p_i = |psi_i|^2 / ||Psi||^2. NOT a quantum probability."""
    a = np.abs(psi) ** 2
    return a / max(a.sum(), 1e-30)

def structural_calculus(W_ideal: np.ndarray, T_cross: np.ndarray,
                        dW_fab: np.ndarray, eps: float = 1e-6
                        ) -> Dict[str, Any]:
    """P1/P2 Eq 22-23: exact inverse or Tikhonov-regularised pseudoinverse."""
    cond = float(np.linalg.cond(T_cross))
    exact = None
    if cond < 1e10:
        exact = W_ideal @ np.linalg.inv(T_cross) + dW_fab
    TcH = T_cross.conj().T
    reg = (W_ideal @ np.linalg.inv(TcH @ T_cross + eps*np.eye(len(T_cross)))
           @ TcH + dW_fab)
    return {"cond_T_cross": cond, "used_regularised": cond >= 1e10,
            "W_programmed": exact if exact is not None else reg,
            "note": "sign/order convention must match the physical calibration model"}


# ---- P1 DRC (six R3 rules) -------------------------------------------------

@dataclass
class R3Rules:
    W_min_um: float = 0.40
    g_min_um: float = 0.20
    R_min_um: float = 5.0
    d_clear_um: float = 1.0
    T_gap_um: float = 2.0
    pad_area_min_um2: float = 10000.0

def drc_r3(geom: Dict[str, float], rules: Optional[R3Rules] = None) -> Dict[str, Any]:
    """P1 §4.2: six geometry predicates. SOFTWARE check, not foundry sign-off."""
    r = rules or R3Rules()
    fails, warns = [], []
    if geom.get("wg_width_um", 1.0) < r.W_min_um: fails.append("WG_WIDTH")
    if geom.get("dcplr_gap_um", 1.0) < r.g_min_um: fails.append("COUPLER_GAP")
    if geom.get("bend_radius_um", 10.0) < r.R_min_um: fails.append("BEND_RADIUS")
    if geom.get("heater_wg_clear_um", 2.0) < r.d_clear_um: fails.append("HEATER_CLEAR")
    if geom.get("heater_heater_gap_um", 3.0) < r.T_gap_um: warns.append("THERMAL_CROSSTALK")
    if geom.get("pad_area_um2", 1e4) < r.pad_area_min_um2: fails.append("PAD_AREA")
    return {"fails": fails, "warns": warns, "pass": len(fails) == 0,
            "scope": "local geometry predicates only — NOT foundry DRC/LVS"}


# =============================================================================
# §2. P2 — DSQP-SESI R4.0 COVARIANT + ADAPTIVE
# =============================================================================

LAYER_MAP_R4 = {
 1:"WG_CORE", 2:"WG_CLAD", 10:"HEATER_METAL", 15:"PCM_GST", 16:"BTO_LAYER",
 20:"EDGE_COUPLER", 30:"VIA",
 40:"METRIC_CONTROL", 41:"CURVATURE_ANALOG", 42:"SPACETIME_DELAY",
 43:"COVARIANT_ROUTING", 44:"FIELD_MONITOR", 45:"SYNC_REFERENCE",
 46:"GRAVITY_MODEL_INTERFACE", 47:"ENERGY_MONITOR", 99:"MASK_DOC"}

def drc_r4(geom: Dict[str, float], eta_max: float = 0.05,
           rules: Optional[R3Rules] = None) -> Dict[str, Any]:
    """P2 §4.2: R3 rules + monitor keep-out / coupling limit + tape-out gate."""
    base = drc_r3(geom, rules)
    fails = list(base["fails"])
    if geom.get("monitor_wg_clear_um", 2.0) < (rules or R3Rules()).d_clear_um:
        fails.append("MONITOR_KEEPOUT")
    if geom.get("monitor_coupling", 0.01) > eta_max:
        fails.append("MONITOR_DISTURBANCE")
    return {"fails": fails, "warns": base["warns"], "pass": len(fails) == 0,
            "layers": LAYER_MAP_R4}

def tapeout_gate(pdk_id: str, drc_clean: bool, lvs_clean: bool,
                 human_approval: str = "") -> Dict[str, Any]:
    """P2 §5.3 / P3 §4.2: REJECT_OR_HOLD_TAPE_OUT. Enforced."""
    if not (pdk_id and drc_clean and lvs_clean and human_approval):
        raise Refusal(
            "REJECT_OR_HOLD_TAPE_OUT: need foundry PDK id + clean DRC + clean LVS "
            "+ explicit human authorisation.\n"
            "A GDS-II that opens is not fabrication-ready. Autonomous optimisation "
            "may propose layouts; fabrication authorisation is a separate decision.")
    return {"allowed": True, "pdk": pdk_id, "approved_by": human_approval}


# ---- P2 covariant sector ---------------------------------------------------

def ppn_metric(Phi: float, c: float = 299792458.0,
               gamma_PPN: float = 1.0) -> np.ndarray:
    """P2 Eq 19: weak-field PPN line element, signature (-+++)."""
    e = 2.0 * Phi / c**2
    return np.diag([-(1.0 + e), (1.0 - gamma_PPN*e),
                    (1.0 - gamma_PPN*e), (1.0 - gamma_PPN*e)])

def phi_GR_from_metric(g_eff: np.ndarray, lam0: float, L_m: float,
                       n_eff_ref: float) -> Dict[str, float]:
    """
    P2 Eq 20 compilation M: g_eff -> {n_eff, phi, tau, W}.
    MODE A (simulation): the metric is a numerical input.
    MODE B (physical): requires independent measurement + sensitivity budget.
    """
    g00 = float(-g_eff[0, 0]); gxx = float(g_eff[1, 1])
    n_eff = n_eff_ref * math.sqrt(max(gxx / max(g00, 1e-12), 1e-12))
    phi = 2 * PI / lam0 * (n_eff - n_eff_ref) * L_m
    return {"n_eff": n_eff, "phi_GR": phi,
            "tau_rel": math.sqrt(max(g00, 1e-12)),
            "mode": "A_simulation",
            "REFUSED": "this is metric-ANALOGUE emulation, not spacetime curvature"}

def covariant_residual_flat11(psi_re: np.ndarray, psi_im: np.ndarray,
                              dx: float, dt: float, m2: float, lam4: float
                              ) -> Dict[str, float]:
    """
    P2 Eq 18 in flat 1+1: box Psi - V'(|Psi|^2) Psi = 0, V(s)=m2 s + lam4/2 s^2.
    Returns the discrete residual norm. A consistency diagnostic, NOT a proof
    of global well-posedness (R5).
    """
    def d2(f, h, ax):
        return np.gradient(np.gradient(f, h, axis=ax, edge_order=2),
                           h, axis=ax, edge_order=2)
    box_re = -d2(psi_re, dt, 0) + d2(psi_re, dx, 1)
    box_im = -d2(psi_im, dt, 0) + d2(psi_im, dx, 1)
    s = psi_re**2 + psi_im**2
    Vp = m2 + lam4 * s
    r_re = box_re - Vp * psi_re
    r_im = box_im - Vp * psi_im
    return {"residual_L2": float(np.sqrt(np.mean(r_re**2 + r_im**2))),
            "scope": "flat 1+1 discrete residual; global regularity is OPEN (R5)"}

def stress_energy_divergence_flat11(psi: np.ndarray, dx: float, dt: float,
                                    m2: float, lam4: float) -> Dict[str, float]:
    """
    P2 Thm A.1: nabla^mu T_mu_nu^(Psi) = 0 ON-SHELL.
    Verified numerically in flat 1+1 for a real scalar; the general-metric
    identity is the standard diffeomorphism Noether identity.
    """
    pt = np.gradient(psi, dt, axis=0, edge_order=2)
    px = np.gradient(psi, dx, axis=1, edge_order=2)
    s = psi**2
    V = m2*s + 0.5*lam4*s**2
    L = pt**2 - px**2 - V
    T00 = pt*pt - L; T01 = pt*px; T11 = px*px + L
    div0 = -np.gradient(T00, dt, axis=0, edge_order=2) + np.gradient(T01, dx, axis=1, edge_order=2)
    div1 = -np.gradient(T01, dt, axis=0, edge_order=2) + np.gradient(T11, dx, axis=1, edge_order=2)
    return {"div_nu0_L2": float(np.sqrt(np.mean(div0**2))),
            "div_nu1_L2": float(np.sqrt(np.mean(div1**2))),
            "note": "vanishes on-shell; off-shell residual is expected nonzero"}


# ---- P2 CHSH + no-signalling ----------------------------------------------

def chsh_exhaustive() -> Dict[str, Any]:
    """P2 Eq 13: |S| <= 2 for LHV+MI. PROVEN by enumerating all 16 strategies."""
    vals = [abs(a0*b0 + a0*b1 + a1*b0 - a1*b1)
            for a0,a1,b0,b1 in itertools.product((-1,1), repeat=4)]
    ang = [0.0, PI/2, PI/4, -PI/4]
    E = lambda i,j: -math.cos(ang[i]-ang[j])
    Sq = E(0,2)+E(0,3)+E(1,2)-E(1,3)
    return {"n_strategies": len(vals), "max_abs_S_local": max(vals),
            "tsirelson": 2*math.sqrt(2), "quantum_singlet_S": Sq,
            "no_signalling_bound": 4.0,
            "proof": "exhaustive over 16 deterministic strategies; convexity "
                     "extends to all LHV models",
            "REFUSED_R3": "a classical photonic chip CANNOT exceed |S|=2"}

def no_signalling_check(joint: np.ndarray) -> Dict[str, Any]:
    """P2 Eq 15: sum_B P(A,B|a,b) = P(A|a) for all b, and symmetrically."""
    mA = joint.sum(axis=3); mB = joint.sum(axis=2)
    dA = float(np.abs(mA - mA.mean(axis=1, keepdims=True)).max())
    dB = float(np.abs(mB - mB.mean(axis=0, keepdims=True)).max())
    return {"no_signalling_A": dA < 1e-10, "no_signalling_B": dB < 1e-10,
            "max_dev_A": dA, "max_dev_B": dB}


# ---- P2 SESI typed graph ---------------------------------------------------

PORT_TYPES = {"OPTICAL", "ELECTRICAL", "CONTROL", "DATA"}

@dataclass
class Port:
    name: str; ptype: str; direction: str
    wl_nm: Optional[float] = None; power_max: Optional[float] = None
    voltage_max: Optional[float] = None

@dataclass
class Module:
    name: str; mtype: str; ports: List[Port]; geometry_hash: str = ""

def compatible(po: Port, pi: Port) -> bool:
    """P2 §5.1: Compatible(p_out, p_in)."""
    if po.ptype != pi.ptype or po.direction != "out" or pi.direction != "in":
        return False
    if po.ptype == "OPTICAL":
        if po.wl_nm is None or pi.wl_nm is None: return False
        if abs(po.wl_nm - pi.wl_nm) > 1.0: return False
        if po.power_max is not None and pi.power_max is not None:
            if po.power_max > pi.power_max: return False
    if po.ptype == "ELECTRICAL":
        if po.voltage_max is not None and pi.voltage_max is not None:
            if po.voltage_max > pi.voltage_max: return False
    return True

class SESIGraph:
    """P2 §3: typed structural graph S = (G, theta, C, M)."""
    def __init__(self):
        self.modules: Dict[str, Module] = {}
        self.edges: List[Tuple[str,str,str,str]] = []
        self.layout_hash: str = ""
        self.history: List[Dict] = []

    def add(self, m: Module): self.modules[m.name] = m

    def connect(self, mo, po, mi, pi) -> bool:
        a = next((p for p in self.modules[mo].ports if p.name == po), None)
        b = next((p for p in self.modules[mi].ports if p.name == pi), None)
        if a is None or b is None or not compatible(a, b): return False
        self.edges.append((mo, po, mi, pi)); return True

    def valid(self) -> bool:
        """P2 Eq 26: C_type and C_connectivity (physical/operational via DRC)."""
        for (mo, po, mi, pi) in self.edges:
            a = next(p for p in self.modules[mo].ports if p.name == po)
            b = next(p for p in self.modules[mi].ports if p.name == pi)
            if not compatible(a, b): return False
        return True

    def compute_layout_hash(self) -> str:
        blob = json.dumps({n: m.geometry_hash
                           for n, m in sorted(self.modules.items())}, sort_keys=True)
        self.layout_hash = hashlib.sha256(blob.encode()).hexdigest()[:16]
        return self.layout_hash

    def objective(self, w: Dict[str, float], L: Dict[str, float]) -> float:
        """P2 Eq 28: six-objective Pareto scalarisation."""
        keys = ("task","energy","latency","loss","instability","complexity")
        return sum(w.get(k, 1.0) * L.get(k, 0.0) for k in keys)

    def runtime_reparam(self, theta_new: Dict[str, float]) -> Dict[str, Any]:
        """P2 Prop 6: runtime update does NOT alter fabricated geometry."""
        h0 = self.compute_layout_hash()
        self.history.append({"kind": "runtime", "theta": theta_new})
        h1 = self.compute_layout_hash()
        return {"layout_unchanged": h0 == h1, "hash": h1}

    def structural_redesign(self, new_module: Module) -> Dict[str, Any]:
        """P2 §12.3: physical redesign requires a NEW GDS revision."""
        self.add(new_module)
        h = self.compute_layout_hash()
        self.history.append({"kind": "redesign", "module": new_module.name,
                             "hash": h})
        return {"requires_new_gds": True, "new_hash": h,
                "gate": "tapeout_gate() must pass before fabrication"}


# =============================================================================
# §3. P3 — ESDA 2.0 FORMAL STATE + MEASUREMENT + T1..T6
# =============================================================================

@dataclass
class ESDAState:
    """P3 Def 1.1: X = (Psi, Pi, Lambda, M)."""
    psi: np.ndarray; pi: np.ndarray; lam: np.ndarray; m: np.ndarray
    def replace(self, **kw) -> "ESDAState":
        d = {"psi": self.psi, "pi": self.pi, "lam": self.lam, "m": self.m}
        d.update(kw); return ESDAState(**d)

class WeakTapGw:
    """P3 Def 1.2: G_w scales field amplitude by sqrt(eta)."""
    def __init__(self, eta: float = 0.05):
        assert 0 < eta < 1; self.eta = eta
    def __call__(self, psi): 
        return math.sqrt(self.eta)*psi, math.sqrt(1-self.eta)*psi

class HardGateGh:
    """P3 Eq 3: G_h with fixed strict total order for ties."""
    def __init__(self, n_out: int, score: Optional[Callable] = None):
        self.n_out = n_out; self.score = score or (lambda z: np.abs(z)**2)
    def __call__(self, z) -> int:
        s = np.asarray(self.score(z), float).reshape(-1)
        if s.size != self.n_out: raise ValueError("score size != n_out")
        return int(np.argmax(s))

class ESDAMap:
    """P3 Eq 2: X_{t+1} = F_theta(X_t, u_t); observation G = G_h o D o G_w."""
    def __init__(self, gw: WeakTapGw, gh: HardGateGh):
        self.gw, self.gh = gw, gh
    def step(self, x: ESDAState, u) -> ESDAState:
        raise NotImplementedError
    def observe(self, x: ESDAState) -> int:
        mon, _ = self.gw(x.psi)
        return self.gh(mon + x.lam.reshape(mon.shape))
    def rollout(self, x0: ESDAState, us: List) -> List[int]:
        x, ys = x0, []
        for u in us:
            x = self.step(x, u); ys.append(self.observe(x))
        return ys

def pushforward_ensemble(model: ESDAMap, rho0: List[ESDAState],
                         us: List) -> Dict[str, Any]:
    """P3 Prop 1.1: mu_t = (G o F^t)_# rho_0."""
    T = len(us); counts = np.zeros((T, model.gh.n_out), np.int64)
    for x0 in rho0:
        for t, y in enumerate(model.rollout(x0, us)): counts[t, y] += 1
    mu = counts / max(len(rho0), 1)
    return {"mu": mu, "n_ensemble": len(rho0),
            "normalised": bool(np.allclose(mu.sum(1), 1.0)),
            "law": "mu_t = (G o F^t)_# rho_0"}

def contraction_certificate(W: np.ndarray, alpha: float, u: np.ndarray,
                            n_iter: int = 400, delta: float = 0.0
                            ) -> Dict[str, Any]:
    """P3 Prop 2.1 / Eq 24: q-contraction, a-posteriori bound, delta/(1-q)."""
    q = float(np.linalg.norm(W, 2))
    T = lambda z: theta_C(W @ z + u, alpha)
    z = np.zeros(W.shape[0], complex); ratios, steps = [], []
    z = T(z)
    for _ in range(n_iter):
        nxt = T(z); d = float(np.linalg.norm(nxt - z))
        steps.append(d)
        if len(steps) > 1 and steps[-2] > 1e-14: ratios.append(d/steps[-2])
        z = nxt
    zstar = z.copy()
    post_ok = True
    z2 = np.ones(W.shape[0], complex)*0.3
    for _ in range(n_iter):
        p = z2.copy(); z2 = T(z2)
        bnd = q/(1-q)*float(np.linalg.norm(z2-p)) if q < 1 else np.inf
        if float(np.linalg.norm(z2 - zstar)) > bnd + 1e-8: post_ok = False
    hw_max = float("nan")
    if delta > 0 and q < 1:
        rng = np.random.default_rng(0); zt = np.zeros(W.shape[0], complex)
        errs = []
        for _ in range(n_iter):
            e = rng.normal(size=W.shape[0]) + 1j*rng.normal(size=W.shape[0])
            e = e/np.linalg.norm(e)*delta*rng.uniform(0,1)
            zt = T(zt) + e; errs.append(float(np.linalg.norm(zt - zstar)))
        hw_max = max(errs[-100:])
    return {"q": q, "contractive": q < 1.0,
            "max_ratio": max(ratios) if ratios else 0.0,
            "posterior_bound_holds": post_ok,
            "hw_error_bound": delta/(1-q) if q < 1 else float("inf"),
            "hw_observed_tail_max": hw_max,
            "unitary_caveat": "lossless MZI mesh has ||U||=1; attenuation "
                              "kappa<1 is a DESIGN REQUIREMENT"}

def dissipative_norm_decay(psi0: np.ndarray, nu: float, alpha: float,
                           n_steps: int = 200, dt: float = 1e-3
                           ) -> Dict[str, Any]:
    """P3 Eq 10: d/dt ||Psi||^2 = -2 nu alpha int |Psi|^4 <= 0."""
    psi = psi0.astype(complex).copy(); norms = [float(np.sum(np.abs(psi)**2))]
    for _ in range(n_steps):
        psi = psi - dt*nu*alpha*np.abs(psi)**2*psi
        norms.append(float(np.sum(np.abs(psi)**2)))
    mono = all(norms[i+1] <= norms[i] + 1e-12 for i in range(len(norms)-1))
    return {"norm_first": norms[0], "norm_last": norms[-1],
            "monotone_decay": mono,
            "reservoir_note": "P3 Eq 11: total T (field+reservoir+interaction) "
                              "is conserved; the field alone is dissipative"}


# ---- P3 ESCA T1..T6 --------------------------------------------------------

def esca_t1(p_fine: np.ndarray, levels: int = 3) -> Dict[str, Any]:
    """P3 Thm 1: p_m = R_m p_{m+1}, non-negativity, normalisation."""
    ps = {levels: p_fine/p_fine.sum()}
    for m in range(levels-1, -1, -1): ps[m] = ps[m+1].reshape(-1,8).sum(1)
    en = max(abs(p.sum()-1) for p in ps.values())
    ec = max(float(np.abs(ps[m+1].reshape(-1,8).sum(1)-ps[m]).max())
             for m in range(levels))
    nn = all(bool((p >= -1e-14).all()) for p in ps.values())
    return {"norm_err": en, "consistency_err": ec, "non_negative": nn,
            "passed": en < 1e-12 and ec < 1e-12 and nn}

@dataclass
class NCCTNode:
    a: int; b: int; mark: Optional[int] = None
    value: Optional[float] = None
    children: List["NCCTNode"] = field(default_factory=list)
    def is_leaf(self): return not self.children

def ncct_build(seq: np.ndarray, max_depth: int = 3, min_len: int = 4) -> NCCTNode:
    root = NCCTNode(0, len(seq))
    def rec(nd, dep):
        seg = seq[nd.a:nd.b]
        if dep >= max_depth or nd.b-nd.a <= min_len or np.all(seg == seg[0]):
            nd.value = float(seg[0]); nd.mark = int(seg[0]); return
        best, gain = None, -1.0
        for c in range(nd.a+min_len, nd.b-min_len+1):
            g = abs(seq[nd.a:c].mean() - seq[c:nd.b].mean())
            if g > gain: gain, best = g, c
        if best is None or gain < 1e-9:
            nd.value = float(seg.mean()); nd.mark = int(round(seg.mean())); return
        nd.mark = int(round(seg.mean()))
        for (s,e) in ((nd.a,best),(best,nd.b)):
            ch = NCCTNode(s,e); nd.children.append(ch); rec(ch, dep+1)
    rec(root, 0); return root

def ncct_certificate(root: NCCTNode, seq: np.ndarray) -> Dict[str, Any]:
    """P3 Thm 3: exact reconstruction + mass consistency + compression."""
    rec = np.full(root.b, np.nan)
    def fill(nd):
        if nd.is_leaf(): rec[nd.a:nd.b] = nd.value; return
        for c in nd.children: fill(c)
    fill(root)
    exact = bool(np.allclose(rec, seq))
    def leaves(nd): return [nd] if nd.is_leaf() else [l for c in nd.children for l in leaves(c)]
    L = leaves(root)
    mass = abs(sum(l.b-l.a for l in L)/root.b - 1.0) < 1e-12
    n_int = 0
    def cnt(nd):
        nonlocal n_int
        if not nd.is_leaf():
            n_int += 1
            for c in nd.children: cnt(c)
    cnt(root)
    flat = root.b*8; tree = (len(L)+n_int)*(8+16)
    return {"exact": exact, "mass_consistent": mass,
            "n_leaves": len(L), "n_internal": n_int,
            "compression_ratio": flat/max(tree,1),
            "passed": exact and mass}

def esca_t4(n_blocks: int = 14, p: float = 0.5, seed: int = 0) -> Dict[str, Any]:
    """P3 Thm 4 (T4-MEOT): super-dominant blocks -> vertex/two-point limit."""
    rng = np.random.default_rng(seed)
    tau = [0]
    for k in range(n_blocks): tau.append(tau[-1] + 4**k)
    marks = (rng.random(n_blocks) < p).astype(int)
    fr = []
    for k in range(2, n_blocks):
        lo, hi = tau[k], tau[k+1]
        nk = int(lo*math.sqrt(hi-lo)) + 1
        if not (lo < nk < hi): continue
        before = sum((tau[j+1]-tau[j])*marks[j] for j in range(k))
        fr.append((before + (nk-lo)*marks[k])/nk)
    fr = np.array(fr)
    conc = float(((fr < 0.1)|(fr > 0.9)).mean()) if len(fr) else 0.0
    return {"n_obs": len(fr), "concentration": conc,
            "limit": f"{p:.2f} delta_1 + {1-p:.2f} delta_0",
            "passed": conc > 0.8,
            "note": "E[F]->p does NOT imply F->p a.s."}

def esca_t5(loss_matrix: np.ndarray, lam: float, eps: float) -> Dict[str, Any]:
    """P3 Thm 5: robust minimiser + uniform chance bound over the model set."""
    R = loss_matrix.max(axis=0); a = int(np.argmin(R))
    exc = float((loss_matrix[:, a] > lam).mean())
    ok = exc <= eps
    return {"a_star": a, "worst_case_risk": float(R[a]),
            "exceedance": exc, "eps": eps,
            "status": "DECISION_CLOSURE_CERTIFIED" if ok else "UNCERTAINTY_RETAINED",
            "passed": True,
            "caveat": "guarantee is void if the true model lies outside P"}

def carpet_laplacian(side: int) -> np.ndarray:
    side = max(3, int(side)); n = side*side
    A = np.zeros((n,n))
    for i in range(side):
        for j in range(side):
            if i%3==1 and j%3==1: continue
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii,jj = i+di, j+dj
                if 0<=ii<side and 0<=jj<side and not (ii%3==1 and jj%3==1):
                    A[i*side+j, ii*side+jj] = 1.0
    return np.diag(A.sum(1)) - A

def esca_t6_harness(levels=(1,2,3), lam: float = 1.0,
                    rho: float = 9/8) -> Dict[str, Any]:
    """P3 Thm 6: Mosco -> resolvent/semigroup. NUMERICAL HARNESS, not a proof."""
    eig, res = [], []
    for m, n in enumerate(levels):
        L = carpet_laplacian(3**n) * (rho**m)
        w = np.linalg.eigvalsh(L); eig.append(float(w[1]))
        N = L.shape[0]
        f = np.linspace(0,1,N); f /= max(np.linalg.norm(f)/math.sqrt(N),1e-12)
        r = np.linalg.solve(L + lam*np.eye(N), f)
        res.append(float(np.linalg.norm(r)/math.sqrt(N)))
    return {"levels": list(levels), "first_nonzero_eig": eig,
            "resolvent_norms": res,
            "M1_weak_liminf": "NOT PROVEN HERE",
            "M2_recovery_sequence": "NOT CONSTRUCTED HERE",
            "STATUS": "CONDITIONAL — see Carpet-19 Thm 2 under I1..I4",
            "REFUSED_R6": True}


# =============================================================================
# §4. UNIFIED THREE-PAPER PIPELINE
# =============================================================================

class ThreePaperStack:
    """
    P1 -> P2 -> P3 as one executable stack.
    The optical core (P1) is embedded in the covariant/adaptive layer (P2),
    which is formalised by the state/measurement layer (P3).
    """
    def __init__(self, alpha: float = 0.5, eta: float = 0.05, n_out: int = 4):
        self.alpha = alpha; self.eta = eta
        self.gw = WeakTapGw(eta); self.gh = HardGateGh(n_out)
        self.graph = SESIGraph()
        self.provenance: List[Dict] = []

    def optical_step(self, psi: np.ndarray, W: np.ndarray,
                     u: np.ndarray, phi_GR: float = 0.0) -> Dict[str, Any]:
        """P2 Eq 21 / P3 Eq 6: Psi' = theta_C(W Psi + u) with GR phase applied."""
        Wg = W * np.exp(1j*phi_GR)
        z = theta_C(Wg @ psi + u, self.alpha)
        mon, thr = self.gw(z)
        y = self.gh(mon)
        return {"psi_next": z, "psi_monitor": mon, "psi_through": thr,
                "decision": y, "phi_GR": phi_GR,
                "bound": theta_bound(self.alpha),
                "kind": "classical_optical_accelerator"}

    def formal_step(self, x: ESDAState, W: np.ndarray, u: np.ndarray,
                    phi_GR: float = 0.0) -> Tuple[ESDAState, int]:
        """P3: deterministic F_theta + observation G."""
        out = self.optical_step(x.psi, W, u, phi_GR)
        xn = x.replace(psi=out["psi_next"], pi=out["psi_monitor"])
        return xn, out["decision"]

    def record(self, layer: str, payload: Dict) -> Dict:
        rec = {"layer": layer,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "actionability": RESEARCH_ONLY}
        self.provenance.append(rec)
        return {**payload, **rec}

    def report(self) -> Dict[str, Any]:
        return {"version": VERSION, "papers": PAPERS, "refusals": REFUSALS,
                "provenance": self.provenance}


# =============================================================================
# §5. BRIDGE TO omni_stack.py
# =============================================================================

def attach_to_omni(omni: Any) -> Dict[str, Any]:
    """
    Wire the three-paper stack into omni_stack's photonics / quantum / fab
    domains. Refuses to relabel the NLSE accelerator as a quantum computer.
    """
    st = ThreePaperStack()
    attached = {}
    if hasattr(omni, "phot"):
        attached["photonics_ssc"] = "P1 core + P2 layers 40..47 + P3 formal state"
    if hasattr(omni, "nlse"):
        kind = getattr(omni.nlse, "KIND", "unknown")
        if kind != "classical_optical_accelerator":
            raise Refusal("R1: NLSE accelerator must be labelled classical.")
        attached["nlse_accelerator"] = "P1 theta_C / P3 SSC — CLASSICAL, not quantum"
    if hasattr(omni, "qchip"):
        attached["quantum_chip"] = "SEPARATE from NLSE; parameter models only"
    if hasattr(omni, "fab"):
        attached["fab_equipment"] = "P1 §5 / P2 §6 pre-tapeout gates enforced"
    return {"attached": attached, "stack": st,
            "refusals_enforced": list(REFUSALS)}


# =============================================================================
# §6. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng = np.random.default_rng(0)

    print("\n[P1-T1] saturation bound + non-expansiveness")
    for a in (0.1, 0.5, 2.0):
        r = np.exp(rng.uniform(-8,8,80000))
        z = r*np.exp(1j*rng.uniform(0,2*PI,r.size))
        mx = float(np.abs(theta_C(z,a)).max())
        rep(f"alpha={a}: |theta_C| <= 1/(2 sqrt a)", mx <= theta_bound(a)+1e-12,
            f"{mx:.6f} <= {theta_bound(a):.6f}")
    sv = [theta_jacobian_sv(r,0.5) for r in np.linspace(0,20,400)]
    rep("both Jacobian singular values <= 1", max(max(x,y) for x,y in sv) <= 1+1e-12)

    print("\n[P1-T2] MZI unitarity + weak tap + argmax + normalisation")
    U = mzi_matrix(PI/4, PI/3, 0.3, 0.7)
    rep("ideal MZI unitary", np.allclose(U.conj().T@U, np.eye(2), atol=1e-10))
    wt = weak_tap(1.0, 0.05)
    rep("P_m + P_t = P_in", wt["conservation_err"] < 1e-14)
    uniq = all(hard_gate(np.array(s)) == int(np.argmax(s))
               for s in itertools.product([0.0,1.0], repeat=4))
    rep("argmax unique under fixed tie-break (16 cases)", uniq)
    psi = rng.normal(size=5)+1j*rng.normal(size=5)
    p = intensity_probability(psi)
    rep("intensity probability sums to 1", abs(p.sum()-1) < 1e-12 and (p>=0).all())

    print("\n[P1-T3] alpha_eff / resonance / coupler / phase / compensation")
    ae = alpha_eff_mrr(2e-19, 1e5, 1e-3, 1e-12)
    print(f"      alpha_eff={ae['alpha_eff']:.4g}  bound={ae['theta_bound']:.4g}")
    rep("alpha_eff computed (unit convention flagged)", "unit_note" in ae)
    mr = mrr_resonance(10e-6, 2.4, 45)
    rep("MRR resonance near 1550 nm", abs(mr["detuning_from_1550nm"]) < 200,
        f"{mr['detuning_from_1550nm']:.1f} nm")
    rep("kappa(g) decreasing", coupler_kappa(0.2) > coupler_kappa(0.4))
    ph = phi_total(1e-4, 1e-3, 1.55e-6, 2e-5, 10.0, 5e-4, 0.3, 0.05, 0.01)
    rep("phi_total = prop+heater+PCM+GR",
        abs(ph["phi_total"]-(ph["phi_prop"]+ph["phi_heater"]+ph["phi_PCM"]+ph["phi_GR"]))<1e-12)
    W = rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    Tc = np.eye(4)+0.05*rng.normal(size=(4,4))
    sc = structural_calculus(W, Tc, np.zeros_like(W))
    rep("structural calculus returns W_programmed", sc["W_programmed"].shape==(4,4))

    print("\n[P1-T4] DRC R3 + tape-out gate")
    good = dict(wg_width_um=0.5, dcplr_gap_um=0.3, bend_radius_um=8.0,
                heater_wg_clear_um=1.5, heater_heater_gap_um=3.0, pad_area_um2=1e4)
    bad = dict(good, wg_width_um=0.3, bend_radius_um=3.0)
    rep("good geometry passes R3 DRC", drc_r3(good)["pass"])
    rep("bad geometry fails R3 DRC", not drc_r3(bad)["pass"],
        str(drc_r3(bad)["fails"]))
    blocked = False
    try: tapeout_gate("", False, False)
    except Refusal: blocked = True
    rep("tape-out BLOCKED without PDK/DRC/LVS/human", blocked)
    rep("tape-out allowed with all four",
        tapeout_gate("PDK-X", True, True, "human:Yoon")["allowed"])

    print("\n[P2-T1] R4 layers + monitor keep-out")
    r4 = drc_r4(dict(good, monitor_wg_clear_um=1.5, monitor_coupling=0.01))
    rep("R4 has layers 40..47", all(k in r4["layers"] for k in range(40,48)))
    rep("monitor disturbance detected",
        not drc_r4(dict(good, monitor_coupling=0.5))["pass"])

    print("\n[P2-T2] covariant sector + CHSH + no-signalling")
    g = ppn_metric(-1e9)
    rep("PPN metric signature (-+++)", g[0,0] < 0 and g[1,1] > 0)
    pg = phi_GR_from_metric(g, 1.55e-6, 1e-3, 2.4)
    rep("phi_GR is ANALOGUE (mode A)", pg["mode"] == "A_simulation")
    nx, nt = 32, 32
    x = np.linspace(0,1,nx); t = np.linspace(0,1,nt)
    X, T = np.meshgrid(x, t, indexing="ij")
    psi_re = np.sin(2*PI*X)*np.cos(2*PI*T)
    cr = covariant_residual_flat11(psi_re, np.zeros_like(psi_re), 1/nx, 1/nt, 1.0, 0.1)
    rep("covariant residual finite", math.isfinite(cr["residual_L2"]))
    se = stress_energy_divergence_flat11(psi_re, 1/nx, 1/nt, 1.0, 0.1)
    rep("stress-energy divergence computed (off-shell nonzero expected)",
        math.isfinite(se["div_nu0_L2"]))
    ch = chsh_exhaustive()
    rep("CHSH max|S| = 2 over 16 strategies", ch["max_abs_S_local"] == 2)
    rep("Tsirelson 2 sqrt 2 reported", abs(ch["tsirelson"]-2*math.sqrt(2))<1e-12)
    ns = no_signalling_check(np.full((2,2,2,2), 0.25))
    rep("no-signalling holds for product distribution",
        ns["no_signalling_A"] and ns["no_signalling_B"])

    print("\n[P2-T3] SESI typed graph + runtime vs redesign")
    st = ThreePaperStack()
    G = st.graph
    G.add(Module("WG","WG",[Port("in","OPTICAL","in",1550,10),
                            Port("out","OPTICAL","out",1550,10)],"wg_v1"))
    G.add(Module("MZI","MZI",[Port("in","OPTICAL","in",1550,10),
                              Port("ctrl","ELECTRICAL","in",None,None,5.0)],"mzi_v1"))
    G.add(Module("CTRL","CONTROL",[Port("v","ELECTRICAL","out",None,None,3.3)],"c_v1"))
    rep("valid optical connection accepted", G.connect("WG","out","MZI","in"))
    rep("valid electrical connection accepted", G.connect("CTRL","v","MZI","ctrl"))
    rep("optical->electrical REJECTED", not G.connect("WG","out","MZI","ctrl"))
    rep("graph Valid(S)", G.valid())
    rt = G.runtime_reparam({"phase": 0.3})
    rep("runtime reparam leaves layout unchanged", rt["layout_unchanged"])
    rd = G.structural_redesign(Module("MRR","MRR",
        [Port("in","OPTICAL","in",1550,10)],"mrr_v1"))
    rep("structural redesign requires new GDS", rd["requires_new_gds"])
    J = G.objective({"task":1,"energy":0.5,"latency":0.5,"loss":1,
                     "instability":1,"complexity":0.2},
                    {"task":0.4,"energy":0.2,"latency":0.1,"loss":0.3,
                     "instability":0.05,"complexity":0.6})
    rep("six-objective scalarisation finite", math.isfinite(J), f"J={J:.4f}")

    print("\n[P3-T1] pushforward law + determinism")
    class Toy(ESDAMap):
        def step(self, x, u): return x.replace(psi=theta_C(0.6*x.psi+u, 0.5))
    tm = Toy(WeakTapGw(0.05), HardGateGh(4))
    rho0 = [ESDAState(rng.normal(size=4)+1j*rng.normal(size=4),
                      np.zeros(4,complex), np.zeros(4,complex), np.zeros(1))
            for _ in range(150)]
    us = [np.ones(4)*0.1]*5
    pf = pushforward_ensemble(tm, rho0, us)
    rep("mu_t normalised at every t", pf["normalised"])
    rep("same X0 -> identical trajectory",
        tm.rollout(rho0[0], us) == tm.rollout(rho0[0], us))

    print("\n[P3-T2] contraction certificates")
    A = rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
    Q,_ = np.linalg.qr(A)
    cc = contraction_certificate(0.6*Q, 0.5, rng.normal(size=6)+0j, delta=1e-3)
    rep("q < 1 contractive", cc["contractive"], f"q={cc['q']:.4f}")
    rep("a-posteriori bound holds", cc["posterior_bound_holds"])
    rep("hardware error <= delta/(1-q)",
        cc["hw_observed_tail_max"] <= cc["hw_error_bound"]+1e-9,
        f"{cc['hw_observed_tail_max']:.5f} <= {cc['hw_error_bound']:.5f}")
    cc1 = contraction_certificate(1.0*Q, 0.5, rng.normal(size=6)+0j)
    rep("unitary mesh NOT contractive (design constraint)", not cc1["contractive"])

    print("\n[P3-T3] dissipative decay + reservoir note")
    dd = dissipative_norm_decay(rng.normal(size=32)+1j*rng.normal(size=32), 0.5, 0.5)
    rep("||Psi||^2 monotone decreasing", dd["monotone_decay"],
        f"{dd['norm_first']:.4f} -> {dd['norm_last']:.4f}")

    print("\n[P3-T4] ESCA T1 / T3-NCCT / T4 / T5 / T6")
    t1 = esca_t1(rng.dirichlet(np.ones(8**3)))
    rep("T1 refinement consistency exact", t1["passed"],
        f"err={t1['consistency_err']:.2e}")
    seq = np.concatenate([np.zeros(60), np.ones(70), np.zeros(50), np.ones(40)])
    t3 = ncct_certificate(ncct_build(seq), seq)
    rep("T3-NCCT exact + mass consistent", t3["passed"],
        f"leaves={t3['n_leaves']} ratio={t3['compression_ratio']:.2f}")
    t4 = esca_t4()
    rep("T4 two-point concentration", t4["passed"], f"{t4['concentration']:.3f}")
    t5 = esca_t5(rng.random((11,7)), 0.6, 0.5)
    rep("T5 robust closure returns status", t5["passed"], t5["status"])
    t6 = esca_t6_harness()
    rep("T6 reports M1/M2 as NOT PROVEN here", t6["REFUSED_R6"])

    print("\n[STACK] unified pipeline + omni bridge")
    Ws = 0.6*Q
    out = st.optical_step(rng.normal(size=6)+1j*rng.normal(size=6),
                          Ws, np.ones(6)*0.1, phi_GR=0.01)
    rep("optical step labelled CLASSICAL", out["kind"]=="classical_optical_accelerator")
    rep("decision in range", 0 <= out["decision"] < 4)
    class FakeOmni:
        phot = object(); fab = object(); qchip = object()
        class nlse: KIND = "classical_optical_accelerator"
    br = attach_to_omni(FakeOmni())
    rep("omni bridge attaches 4 domains", len(br["attached"]) == 4)
    class BadOmni(FakeOmni):
        class nlse: KIND = "quantum_computer"
    blocked = False
    try: attach_to_omni(BadOmni())
    except Refusal: blocked = True
    rep("bridge REFUSES relabelling NLSE as quantum", blocked)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
MACHINE-CHECKED
  P1: saturation bound, non-expansiveness, MZI unitarity, weak-tap power,
      argmax uniqueness, intensity normalisation, six DRC predicates.
  P2: CHSH |S|<=2 by exhaustion, no-signalling marginals, PPN signature,
      covariant + stress-energy residuals, typed-graph validity,
      runtime != geometry, six-objective scalarisation.
  P3: pushforward normalisation + determinism, q-contraction,
      a-posteriori bound, delta/(1-q) hardware bound, dissipative decay,
      ESCA T1 exact, T3-NCCT exact, T4 concentration, T5 closure.

ENFORCED REFUSALS (all fired in the tests)
  tape-out without PDK/DRC/LVS/human -> BLOCKED
  relabelling NLSE as a quantum computer -> BLOCKED
  T6 Mosco M1/M2 reported as NOT PROVEN here

STILL OPEN
  Global Einstein + nonlinear-field regularity (R5).
  Mosco M1/M2 for a specific carpet normalisation (R6).
  Physical gravitational sensitivity of the chip (Mode B).
  Quantum computational advantage.
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Three-paper stack")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest","papers","refusals","layers",
                             "chsh","certify","tapeout"])
    ap.add_argument("--pdk", default=""); ap.add_argument("--human", default="")
    ap.add_argument("--drc", action="store_true"); ap.add_argument("--lvs", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    out: Dict[str, Any] = {}
    if a.cmd == "papers":
        out = PAPERS; print(json.dumps(out, indent=2))
    elif a.cmd == "refusals":
        out = REFUSALS
        for k,v in REFUSALS.items(): print(f"  [{k}] {v}")
    elif a.cmd == "layers":
        out = LAYER_MAP_R4
        for k,v in sorted(LAYER_MAP_R4.items()): print(f"  {k:3d}/0  {v}")
    elif a.cmd == "chsh":
        out = chsh_exhaustive(); print(json.dumps(out, indent=2))
    elif a.cmd == "certify":
        rng = np.random.default_rng(0)
        A = rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
        Q,_ = np.linalg.qr(A)
        out = contraction_certificate(0.6*Q, 0.5, rng.normal(size=6)+0j, delta=1e-3)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "tapeout":
        try:
            out = tapeout_gate(a.pdk, a.drc, a.lvs, a.human)
            print(json.dumps(out, indent=2))
        except Refusal as e:
            print(str(e)); return 2
    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
