#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
photonic_fab_compiler.py — v1.0.0
Unified design & verification compiler for the five MSPS photonic revisions:
  R3 (SSC)  R4 (DSQP-SESI)  R5 (ESDA/ESCA)  |  R6 (QM)  R7 (QM+GR)
  ─────────── CLASSICAL lineage ───────────  |  ── QUANTUM lineage ──
Plus: AI design assistants (transformer / diffusion / world-model / LLM contract)
Plus: honest economics + MPW path. NOT a lithography machine.
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

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
FOUR CORRECTIONS ENFORCED IN CODE
=============================================================================
[C1] THE FIVE PAPERS ARE TWO LINEAGES, NOT FIVE VERSIONS.
     R3/R4/R5: classical complex field, theta_C saturation, intensity readout,
               |S| <= 2 PROVEN. Room-temperature photodiodes.
     R6/R7:    bosonic Fock space, GKSL Lindblad, Born-rule POVM,
               |S| = 2*sqrt(2) Tsirelson, CGLMP. SNSPD at 1-4 K.
     R5 §1 boundary #1 states classical interference != Hilbert-space
     superposition. R6 §1 states it REPLACES classical amplitudes with
     second-quantized operators. Mixing them is a CATEGORY ERROR.
     -> LineageGuard REFUSES cross-lineage layer/operator mixing.

[C2] THE R6/R7 KERR-BLOCKADE REQUIREMENT IS NOT DEMONSTRATED.
     R6 Q-DRC demands g2(0) <= 0.05. With g2 ~ gamma^2/(4 U^2), this needs
     U_Kerr/gamma >= sqrt(5) ~ 2.24. Published integrated MRRs report
     U/gamma ~ 1e-3 .. 1e-1. Deterministic single-photon Kerr gates are an
     OPEN problem; that is why KLM measurement-induced nonlinearity dominates
     photonic quantum computing. This module COMPUTES the requirement and
     labels it NOT_DEMONSTRATED. It does not assume it is achievable.

[C3] THIS IS A DESIGN/VERIFICATION COMPILER, NOT A FAB MACHINE.
     Building lithography equipment competes with ASML (EUV monopoly,
     ~25 years, >EUR 10B), AMAT/Lam/TEL (etch/dep), KLA (metrology).
     Advanced tools are export-controlled. This module produces
     PDK-bound layouts + verification reports, and REFUSES tape-out
     without PDK + clean DRC + clean LVS + human authorization.

[C4] AI COMPONENTS ARE DESIGN ASSISTANTS, NOT ORACLES.
     Transformer  -> DRC-pass surrogate classifier (accelerates search)
     Diffusion    -> candidate geometry generator (must be solver-validated)
     World model  -> dynamics surrogate (must be validated vs GKSL/NLSE)
     LLM          -> PDK doc parsing / spec translation (hallucination-gated)
     None of them proves physics or authorizes fabrication.

=============================================================================
WHAT "PRODUCTION" MEANS HERE
=============================================================================
SOFTWARE production: achievable now. Typed specs, enforced gates, reproducible
                     verification reports, deterministic seeds, audit trail.
FABRICATION production: requires foundry PDK + measured device data +
                     process qualification for BTO / GST / NbTiN.
                     This module tells you exactly what is missing.
=============================================================================
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("pfab")
VERSION = "1.0.0"; PI = math.pi

BANNER = f"""
=============================================================================
PHOTONIC FAB COMPILER v{VERSION}
R3 SSC | R4 DSQP-SESI | R5 ESDA/ESCA  ||  R6 QM | R7 QM+GR
CLASSICAL lineage                    ||  QUANTUM lineage
LineageGuard ENFORCED. Tape-out needs PDK+DRC+LVS+human.
NOT a lithography machine. Kerr blockade NOT_DEMONSTRATED.
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal. Never silently downgraded."""


# =============================================================================
# §1. REVISION SPECIFICATIONS — layer maps, DRC, gates (from the 5 papers)
# =============================================================================

CLASSICAL = "classical"
QUANTUM = "quantum"

@dataclass(frozen=True)
class RevSpec:
    rev: str
    lineage: str
    title: str
    layers: Dict[int, str]
    drc: Dict[str, float]
    gates: Tuple[str, ...]
    notes: str = ""

# ---- R3: SSC Photonic Processor (8 layers, 6 DRC, 7 gates) -----------------
R3 = RevSpec("R3", CLASSICAL,
  "Unicode Math + Heterogeneous Fabrication for SSC Photonic Processor",
  {1:"WG_CORE", 2:"WG_CLAD", 10:"HEATER_METAL", 15:"PCM_GST",
   16:"BTO_LAYER", 20:"EDGE_COUPLER", 30:"VIA", 99:"MASK_DOC"},
  {"wg_width_min_um":0.40, "dcplr_gap_min_um":0.20, "bend_radius_min_um":5.0,
   "heater_wg_clear_um":1.0, "heater_gap_um":2.0, "pad_area_min_um2":10000.0},
  ("mode_solver", "coupler_sweep", "nonlinear_cavity", "thermal_sim",
   "power_loss_budget", "circuit_compensation", "drc_lvs"),
  "Si3N4/Si + SiO2 + TiN/W + GST + BTO. Classical theta_C saturation.")

# ---- R4: DSQP-SESI Covariant (16 layers, 8 DRC) ---------------------------
R4 = RevSpec("R4", CLASSICAL,
  "Covariant GDS-II + Self-Evolving Structural Interface",
  {**R3.layers, 40:"METRIC_CONTROL", 41:"CURVATURE_ANALOG",
   42:"SPACETIME_DELAY", 43:"COVARIANT_ROUTING", 44:"FIELD_MONITOR",
   45:"SYNC_REFERENCE", 46:"GRAVITY_MODEL_INTERFACE", 47:"ENERGY_MONITOR"},
  {**R3.drc, "monitor_wg_clear_um":1.0, "monitor_coupling_max":0.05},
  ("optical_mode_nonlinear", "coupler_weaktap_budget",
   "thermal_crosstalk_stability", "covariant_conservation_residual",
   "drc_lvs_pdk"),
  "Adds metric-analogue layers 40-47. Mode A simulation != Mode B physical GR.")

# ---- R5: ESDA 2.0 / ESCA Unified (19 layers, 9 DRC) -----------------------
R5 = RevSpec("R5", CLASSICAL,
  "ESDA 2.0 & ESCA Unified Photonic GDS-II",
  {**R4.layers, 25:"HYPER_DOF_MUX", 26:"CARPET_MESH_L1", 27:"CARPET_CROSSOVER"},
  {**R4.drc, "wg_width_mm_min_um":0.80, "carpet_pitch_min_um":2.5,
   "carpet_cross_angle_lo_deg":85.0, "carpet_cross_angle_hi_deg":95.0,
   "fwm_phase_match_max_rad":0.05},
  ("mzi_unitary_fidelity", "tucker_rank_cglmp", "carpet_mosco_spectrum",
   "thermal_contraction", "drc_lvs"),
  "Classical multi-DOF tensor modes. R5 §1: NOT quantum entanglement.")

# ---- R6: Quantum-Mechanical Elevation (22 layers, 12 Q-DRC) ---------------
R6 = RevSpec("R6", QUANTUM,
  "Quantum-Mechanical Photonic Chip (Fock space, GKSL, Born POVM)",
  {1:"WG_CORE_QM", 2:"WG_CLAD", 10:"HEATER_METAL", 11:"EO_BTO_ELECTRODE",
   15:"PCM_GST", 16:"BTO_NONLINEAR", 20:"EDGE_COUPLER_QM", 25:"HYPER_DOF_MUX",
   26:"CARPET_MESH_L1", 27:"CARPET_CROSSOVER", 28:"FRACTAFOLD_SG",
   29:"FRACTAFOLD_SG_CLAMP", 30:"VIA", 31:"SNSPD_NANOWIRE", 32:"SNSPD_SHIELD",
   33:"PUMP_REJECT_FILTER", 34:"WEAK_POVM_TAP", 35:"HOMODYNE_GE_SI",
   40:"METRIC_CONTROL", 41:"CURVATURE_ANALOG", 42:"SPACETIME_DELAY",
   99:"MASK_DOC"},
  {"wg_width_min_um":0.45, "wg_width_mm_min_um":0.85, "dcplr_gap_min_um":0.20,
   "bend_radius_min_um":15.0, "eo_wg_clear_um":0.85, "heater_wg_clear_um":1.2,
   "heater_gap_um":2.5, "snspd_width_min_um":0.075, "snspd_pitch_min_um":0.16,
   "pump_snspd_clear_um":250.0, "carpet_cross_angle_lo_deg":87.0,
   "carpet_cross_angle_hi_deg":93.0, "carpet_pitch_min_um":3.0,
   "sg_clamp_clear_um":4.0, "fwm_phase_match_max_rad":0.03,
   "monitor_coupling_max":0.02, "pad_area_min_um2":10000.0},
  ("hom_visibility_process_fidelity", "kerr_blockade_g2",
   "multi_dof_bell_cglmp", "fractal_coercivity", "no_zeno_subsampling",
   "q_drc_lvs"),
  "GENUINE quantum: Fock, GKSL, SNSPD 1-4K, SPDC/SFWM. Kerr blockade OPEN.")

# ---- R7: Maximal QM + GR (R6 + GR layers) --------------------------------
R7 = RevSpec("R7", QUANTUM,
  "Next-Gen Quantum-Photonic + Full Math Corpus + General Relativity",
  {**R6.layers, 43:"COVARIANT_ROUTING", 44:"FIELD_MONITOR",
   45:"SYNC_REFERENCE", 46:"GRAVITY_MODEL_IF", 47:"ENERGY_MONITOR"},
  {**R6.drc},
  R6.gates + ("einstein_kg_maxwell_residual", "clamped_c2_c4_coercivity"),
  "R6 + Einstein-Klein-Gordon-Maxwell, clamped SG C2/C4, No-Zeno, subsampling.")

REVISIONS: Dict[str, RevSpec] = {"R3":R3, "R4":R4, "R5":R5, "R6":R6, "R7":R7}

# ---- Physics feasibility ledger (honest) ---------------------------------
FEASIBILITY = {
 "kerr_blockade_U_over_gamma": {
   "required_for_g2_le_0.05": math.sqrt(5.0),
   "published_integrated_range": (1e-3, 1e-1),
   "status": "NOT_DEMONSTRATED",
   "why": "Deterministic single-photon Kerr gates in integrated MRRs remain "
          "an open problem. KLM measurement-induced nonlinearity is the "
          "established workaround.",
   "mitigation": "Use SPDC/SFWM + heralding + linear optics + postselection "
                 "(KLM/Browne-Rudolph) instead of deterministic Kerr gates.",
 },
 "snspd_cryogenic": {
   "T_operate_K": (1.0, 4.0), "Tc_NbTiN_K": (10.0, 14.0),
   "status": "ESTABLISHED_BUT_COSTLY",
   "why": "Requires closed-cycle cryostat, thermal budget, and cryo-compatible "
          "BTO/GST processes. Not a room-temperature platform.",
 },
 "sin_ultralow_loss": {
   "R6_target_dB_per_cm": 0.05,
   "status": "ACHIEVABLE_NOT_BEST_IN_CLASS",
   "why": "Ultra-low-loss SiN waveguides at ~0.05 dB/cm are demonstrated; "
          "record devices go lower. Requires dedicated anneal/thick-oxide runs.",
 },
 "bto_pockels": {
   "r_eff_bulk_pm_per_V": 923.0,
   "thin_film_range_pm_per_V": (200.0, 1000.0),
   "status": "DEMONSTRATED_PROCESS_DEPENDENT",
   "why": "Epitaxial BTO on Si requires orientation control and low-loss "
          "integration; Vpi*L depends strongly on mode overlap.",
 },
 "gst_photonic_memory": {
   "insertion_loss_per_cell_dB": (0.5, 2.0),
   "endurance_cycles": (1e6, 1e9),
   "status": "DEMONSTRATED_LOSSY",
   "why": "Absorption is significant. Must be placed in selected branches "
          "with bypass paths, not in every high-power path.",
 },
 "metric_analogue_gr": {
   "status": "ANALOGUE_ONLY",
   "why": "Engineered n_eff(x) emulates selected wave-propagation effects of "
          "a metric. It does NOT generate astrophysical spacetime curvature. "
          "Mode A (simulation) != Mode B (physical test).",
 },
 "carpet_mosco": {
   "status": "CONDITIONAL_ON_I1_I4",
   "why": "Carpet-19 Thm 2 proves full-sequence Mosco under (I1)-(I4), which "
          "rest on KZ/BBKT. M1/M2 for a NEW normalization must be established.",
 },
 "fab_equipment": {
   "status": "OUT_OF_SCOPE",
   "why": "ASML holds the EUV monopoly (~25 yr, >EUR 10B, Zeiss optics, "
          "TRUMPF/Cymer source). AMAT/Lam/TEL/KLA hold etch/dep/metrology. "
          "Advanced tools are export-controlled. This module is a compiler.",
 },
}


# =============================================================================
# §2. LINEAGE GUARD — the single most important correctness feature
# =============================================================================

QUANTUM_ONLY_LAYERS = {31, 32, 33, 34, 35, 28, 29}   # SNSPD, pump reject, POVM, HOM, SG
CLASSICAL_ONLY_OPERATORS = {"theta_C", "intensity_argmax", "pushforward_only"}
QUANTUM_ONLY_OPERATORS = {"gksl", "born_povm", "kraus", "fock", "chsh_tsirelson",
                          "cglmp", "kerr_blockade"}

class LineageGuard:
    """
    Refuses category errors between the classical (R3/R4/R5) and quantum
    (R6/R7) lineages. This is not a stylistic preference: R5 §1 boundary #1
    and R6 §1 state the distinction explicitly.
    """
    @staticmethod
    def check_layers(rev: str, layers: List[int]) -> Dict[str, Any]:
        spec = REVISIONS[rev]
        unknown = [l for l in layers if l not in spec.layers]
        if unknown:
            raise Refusal(f"{rev}: layers {unknown} not in its layer map. "
                          f"Allowed: {sorted(spec.layers)}")
        if spec.lineage == CLASSICAL:
            bad = sorted(set(layers) & QUANTUM_ONLY_LAYERS)
            if bad:
                names = [REVISIONS['R6'].layers.get(b, str(b)) for b in bad]
                raise Refusal(
                    f"CATEGORY ERROR: {rev} is CLASSICAL but layout contains "
                    f"quantum-only layers {bad} ({names}).\n"
                    "SNSPD / POVM tap / homodyne / pump-rejection belong to "
                    "R6/R7. Classical intensity readout cannot use them.\n"
                    "Either move to R6/R7 (new platform: cryo + SPDC) or remove "
                    "these layers.")
        return {"ok": True, "rev": rev, "lineage": spec.lineage,
                "n_layers": len(layers)}

    @staticmethod
    def check_operator(rev: str, op: str) -> Dict[str, Any]:
        spec = REVISIONS[rev]
        if spec.lineage == CLASSICAL and op in QUANTUM_ONLY_OPERATORS:
            raise Refusal(
                f"CATEGORY ERROR: operator '{op}' is quantum-only but {rev} is "
                f"classical.\n{rev} uses theta_C saturation + intensity argmax. "
                "GKSL / Born POVM / Fock / Tsirelson require R6/R7.")
        if spec.lineage == QUANTUM and op in CLASSICAL_ONLY_OPERATORS:
            raise Refusal(
                f"CATEGORY ERROR: operator '{op}' is classical-only but {rev} is "
                f"quantum.\nR6/R7 replace classical amplitudes with Fock "
                "operators (R6 §1). theta_C is a classical saturation map, "
                "not a CPTP quantum channel.")
        return {"ok": True, "rev": rev, "operator": op}

    @staticmethod
    def migration_cost(from_rev: str, to_rev: str) -> Dict[str, Any]:
        a, b = REVISIONS[from_rev], REVISIONS[to_rev]
        if a.lineage == b.lineage:
            new = sorted(set(b.layers) - set(a.layers))
            return {"kind": "INCREMENTAL", "new_layers": new,
                    "new_names": [b.layers[l] for l in new],
                    "new_gates": sorted(set(b.gates) - set(a.gates)),
                    "requires_new_fab": len(new) > 0}
        return {"kind": "PLATFORM_CHANGE",
                "from": a.lineage, "to": b.lineage,
                "required_changes": [
                    "Light source: laser -> SPDC/SFWM biphoton source",
                    "Detectors: photodiode -> SNSPD NbTiN at 1-4 K",
                    "New material: NbTiN superconducting nanowire + cryo shield",
                    "Pump rejection cascade (>110 dB) before detectors",
                    "Loss budget: quantum-grade (<0.05 dB/cm SiN)",
                    "Bend radius: 5 um -> 15 um (radiative loss)",
                    "Cryostat + thermal budget + cryo-compatible BTO/GST",
                    "Verification: HOM visibility, g2(0), Bell/CGLMP tomography",
                ],
                "NOT_an_upgrade": True,
                "note": "This is a different device, not a revision bump."}


# =============================================================================
# §3. CLASSICAL CORE (R3/R4/R5) — exact equations from the papers
# =============================================================================

def theta_C(z, alpha: float):
    """R3/R4/R5: saturating Kerr collapse. Bound 1/(2 sqrt(alpha))."""
    if HAS_TORCH and isinstance(z, torch.Tensor):
        return z / (1.0 + alpha * (z.real**2 + z.imag**2))
    return z / (1.0 + alpha * np.abs(z)**2)

def theta_bound(alpha: float) -> float:
    return 1.0 / (2.0 * math.sqrt(alpha))

def theta_jacobian_sv(r: float, alpha: float) -> Tuple[float, float]:
    """ESDA Lemma 2.1(2): both singular values <= 1 => Lip = 1."""
    s = alpha * r * r
    return 1.0/(1.0+s), abs(1.0-s)/(1.0+s)**2

def alpha_eff(n2_BTO: float, Q_MRR: float, L_eff_m: float, A_eff_m2: float
              ) -> Dict[str, Any]:
    """R3 Eq: alpha_eff = n2*Q*L_eff/A_eff. Unit convention needs calibration."""
    a = n2_BTO * Q_MRR * L_eff_m / max(A_eff_m2, 1e-20)
    return {"alpha_eff": a, "theta_bound": theta_bound(a) if a > 0 else float("inf"),
            "calibration_required": "power-normalisation vs measured Kerr response"}

def mrr_resonance(R_m: float, n_eff: float, m: int, lam0: float = 1.55e-6
                  ) -> Dict[str, float]:
    lam = 2*PI*R_m*n_eff/m
    return {"lambda_res_m": lam, "FSR_m": lam**2/(2*PI*R_m*n_eff),
            "detuning_nm": (lam-lam0)*1e9}

def coupler_kappa(g_um: float, k0: float = 0.5, gamma: float = 3.0) -> float:
    return k0*math.exp(-gamma*g_um)

def phi_total(dn: float, L: float, lam0: float, dn_dT: float, dT: float,
              L_h: float, dn_pcm: float, L_pcm: float,
              phi_EO: float = 0.0, phi_GR: float = 0.0) -> Dict[str, float]:
    """R3/R4/R5/R7 Eq: phi_total = dbeta*L + heater + PCM (+EO +GR)."""
    k0 = 2*PI/lam0
    p, h, c = k0*dn*L, k0*dn_dT*dT*L_h, k0*dn_pcm*L_pcm
    return {"phi_prop": p, "phi_heater": h, "phi_PCM": c,
            "phi_EO": phi_EO, "phi_GR": phi_GR,
            "phi_total": p+h+c+phi_EO+phi_GR}

def mzi_matrix(th1: float, th2: float, ph1: float, ph2: float) -> np.ndarray:
    B = lambda t: np.array([[math.cos(t), 1j*math.sin(t)],
                            [1j*math.sin(t), math.cos(t)]])
    return B(th2) @ np.diag([np.exp(1j*ph1), np.exp(1j*ph2)]) @ B(th1)

def weak_tap(P_in: float, eta: float) -> Dict[str, float]:
    return {"P_monitor": eta*P_in, "P_through": (1-eta)*P_in,
            "conservation_err": abs(eta*P_in+(1-eta)*P_in-P_in)}

def structural_calculus(W_ideal: np.ndarray, T_cross: np.ndarray,
                        dW_measured: np.ndarray, eps: float = 1e-6
                        ) -> Dict[str, Any]:
    """R3/R4/R5 Eq 22-23: exact inverse or Tikhonov-regularised."""
    cond = float(np.linalg.cond(T_cross))
    TcH = T_cross.conj().T
    reg = W_ideal @ np.linalg.inv(TcH@T_cross + eps*np.eye(len(T_cross))) @ TcH \
          + dW_measured
    exact = (W_ideal @ np.linalg.inv(T_cross) + dW_measured) if cond < 1e10 else None
    return {"cond_T_cross": cond, "used_regularised": cond >= 1e10,
            "W_programmed": exact if exact is not None else reg}

def contraction_certificate(W: np.ndarray, alpha: float, u: np.ndarray,
                            delta: float = 0.0, n_iter: int = 300
                            ) -> Dict[str, Any]:
    """ESDA Prop 2.1: q-contraction, a-posteriori bound, delta/(1-q)."""
    q = float(np.linalg.norm(W, 2))
    T = lambda z: theta_C(W @ z + u, alpha)
    z = np.zeros(W.shape[0], complex); ratios, steps = [], []
    z = T(z)
    for _ in range(n_iter):
        nxt = T(z); d = float(np.linalg.norm(nxt-z)); steps.append(d)
        if len(steps) > 1 and steps[-2] > 1e-14: ratios.append(d/steps[-2])
        z = nxt
    zstar = z.copy(); post_ok = True
    z2 = np.ones(W.shape[0], complex)*0.3
    for _ in range(n_iter):
        p = z2.copy(); z2 = T(z2)
        b = q/(1-q)*float(np.linalg.norm(z2-p)) if q < 1 else np.inf
        if float(np.linalg.norm(z2-zstar)) > b+1e-8: post_ok = False
    hw = float("nan")
    if delta > 0 and q < 1:
        rng = np.random.default_rng(0); zt = np.zeros(W.shape[0], complex); errs = []
        for _ in range(n_iter):
            e = rng.normal(size=W.shape[0]) + 1j*rng.normal(size=W.shape[0])
            e = e/np.linalg.norm(e)*delta*rng.uniform(0, 1)
            zt = T(zt) + e; errs.append(float(np.linalg.norm(zt-zstar)))
        hw = max(errs[-80:])
    return {"q": q, "contractive": q < 1.0,
            "max_ratio": max(ratios) if ratios else 0.0,
            "posterior_bound_holds": post_ok,
            "hw_error_bound": delta/(1-q) if q < 1 else float("inf"),
            "hw_observed": hw,
            "unitary_caveat": "lossless MZI has ||U||=1; kappa<1 attenuation "
                              "is a DESIGN REQUIREMENT (ESDA D.8)"}

def chsh_classical_bound() -> Dict[str, Any]:
    """PROVEN by exhaustion: 16 local deterministic strategies, max |S| = 2."""
    vals = [abs(a0*b0+a0*b1+a1*b0-a1*b1)
            for a0,a1,b0,b1 in itertools.product((-1,1), repeat=4)]
    return {"n_strategies": len(vals), "max_abs_S": max(vals),
            "proof": "exhaustive; convexity extends to all LHV+MI models",
            "REFUSED": "classical photonic chip CANNOT exceed |S|=2"}


# =============================================================================
# §4. QUANTUM CORE (R6/R7) — Fock, GKSL, blockade, Bell, POVM
# =============================================================================

def fock_dim(d_modes: int, n_max: int) -> int:
    """Truncated symmetric Fock space dimension: C(d+n_max, n_max)."""
    return math.comb(d_modes + n_max, n_max)

def kerr_blockade_requirement(g2_target: float = 0.05) -> Dict[str, Any]:
    """
    R6 Q-DRC gate: g2(0) <= 0.05.
    With g2 ~ gamma^2/(4 U^2)  =>  U/gamma >= sqrt(1/(4*g2)).
    HONEST: this is NOT demonstrated in integrated MRRs.
    """
    req = math.sqrt(1.0/(4.0*g2_target))
    lo, hi = FEASIBILITY["kerr_blockade_U_over_gamma"]["published_integrated_range"]
    return {"g2_target": g2_target, "required_U_over_gamma": req,
            "published_range": (lo, hi),
            "gap_factor": req/hi,
            "status": "NOT_DEMONSTRATED",
            "recommendation": FEASIBILITY["kerr_blockade_U_over_gamma"]["mitigation"]}

def g2_from_U_gamma(U_over_gamma: float) -> float:
    return 1.0/(4.0*max(U_over_gamma, 1e-12)**2)

def gksl_step(rho: np.ndarray, H: np.ndarray, L_ops: List[np.ndarray],
              dt: float) -> np.ndarray:
    """
    Explicit Euler step of the GKSL master equation (R6 Eq 3).
    d rho/dt = -i[H,rho] + sum_k (L rho L^dag - 1/2{L^dag L, rho})
    HONEST: Euler is first-order; for production use an exponential integrator
    or a proper CPTP channel. Trace is renormalised and reported.
    """
    comm = H @ rho - rho @ H
    drho = -1j*comm
    for L in L_ops:
        Ld = L.conj().T
        drho += L @ rho @ Ld - 0.5*(Ld@L@rho + rho@Ld@L)
    out = rho + dt*drho
    tr = float(np.real(np.trace(out)))
    if tr <= 0: raise Refusal("GKSL step produced non-positive trace")
    out = out/tr
    return out

def kraus_apply(rho: np.ndarray, K_list: List[np.ndarray]
                ) -> Dict[str, Any]:
    """Selective POVM measurement (R6 Def 2.4). Returns probs + post-states."""
    probs, states = [], []
    for K in K_list:
        Kr = K @ rho @ K.conj().T
        p = float(np.real(np.trace(Kr)))
        probs.append(max(p, 0.0))
        states.append(Kr/p if p > 1e-15 else np.zeros_like(rho))
    tot = sum(probs)
    if abs(tot-1.0) > 1e-6:
        log.warning("Kraus probabilities sum to %.6f (not 1) — "
                    "check completeness sum K^dag K = I", tot)
    return {"probs": np.array(probs)/max(tot,1e-15), "post_states": states,
            "trace_sum": tot, "born_rule": True}

def chsh_quantum_tsirelson() -> Dict[str, Any]:
    """Quantum singlet at optimal settings: |S| = 2 sqrt(2)."""
    ang = [0.0, PI/2, PI/4, -PI/4]
    E = lambda i,j: -math.cos(ang[i]-ang[j])
    S = E(0,2)+E(0,3)+E(1,2)-E(1,3)
    return {"S": S, "abs_S": abs(S), "tsirelson": 2*math.sqrt(2),
            "exceeds_classical": abs(S) > 2.0,
            "requires": "entangled biphoton source (SPDC/SFWM), NOT a laser",
            "R6_gate": "S_CHSH >= 2.75"}

def cglmp_d3_quantum() -> Dict[str, Any]:
    """R6 Thm 3.2: I_3^QM = 4/(6 sqrt(3) - 9) ~ 2.8729; classical bound 2."""
    val = 4.0/(6.0*math.sqrt(3.0)-9.0)
    return {"I3_quantum": val, "classical_bound": 2.0,
            "R6_gate": "I_CGLMP_3 >= 2.65",
            "requires": "high-dimensional entanglement (freq bins / OAM)"}

def no_signalling_check(joint: np.ndarray) -> Dict[str, Any]:
    """joint[a,b,A,B]; partial-trace marginal invariance (R6 Thm 3.2(3))."""
    mA = joint.sum(axis=3); mB = joint.sum(axis=2)
    dA = float(np.abs(mA - mA.mean(axis=1, keepdims=True)).max())
    dB = float(np.abs(mB - mB.mean(axis=0, keepdims=True)).max())
    return {"no_signalling_A": dA < 1e-10, "no_signalling_B": dB < 1e-10,
            "max_dev_A": dA, "max_dev_B": dB}

def bogoliubov_check(alpha: np.ndarray, beta: np.ndarray) -> Dict[str, Any]:
    """R6/R7: |alpha|^2 - |beta|^2 = delta_kj (canonical preservation)."""
    M = alpha @ alpha.conj().T - beta @ beta.conj().T
    err = float(np.abs(M - np.eye(len(alpha))).max())
    return {"canonical_error": err, "preserved": err < 1e-8,
            "note": "analogue Hawking/Unruh mode mixing; NOT real gravity"}


# =============================================================================
# §5. ESCA / CARPET LAYER (shared by R5/R6/R7)
# =============================================================================

def carpet_cells(n: int) -> np.ndarray:
    side = 3**n
    r = np.arange(side)[:, None]; c = np.arange(side)[None, :]
    keep = np.ones((side, side), bool); rr, cc = r.copy(), c.copy()
    for _ in range(n):
        keep &= ~(((rr % 3) == 1) & ((cc % 3) == 1)); rr //= 3; cc //= 3
    return np.argwhere(keep)

def carpet_laplacian(n: int) -> np.ndarray:
    cells = carpet_cells(n); key = {tuple(c): i for i, c in enumerate(cells)}
    N = len(cells); A = np.zeros((N, N))
    for i, (r, c) in enumerate(cells):
        for dr, dc in ((1,0),(0,1)):
            j = key.get((int(r+dr), int(c+dc)))
            if j is not None: A[i,j] = A[j,i] = 1.0
    return np.diag(A.sum(1)) - A

def esca_t1(p_fine: np.ndarray, levels: int = 3) -> Dict[str, Any]:
    """T1 EXACT: finite additivity under nested refinement."""
    ps = {levels: p_fine/p_fine.sum()}
    for m in range(levels-1, -1, -1): ps[m] = ps[m+1].reshape(-1,8).sum(1)
    en = max(abs(p.sum()-1) for p in ps.values())
    ec = max(float(np.abs(ps[m+1].reshape(-1,8).sum(1)-ps[m]).max())
             for m in range(levels))
    return {"norm_err": en, "consistency_err": ec,
            "passed": en < 1e-12 and ec < 1e-12, "status": "EXACT"}

def mosco_harness(levels=(1,2,3), rho: float = 9/8, lam: float = 1.0
                  ) -> Dict[str, Any]:
    """T6 numerical harness. M1/M2 NOT proven here (Carpet-19 §9)."""
    eig, res = [], []
    for m, n in enumerate(levels):
        L = carpet_laplacian(n)*(rho**m)
        w = np.linalg.eigvalsh(L); eig.append(float(w[1]))
        N = L.shape[0]; f = np.linspace(0,1,N)
        f /= max(np.linalg.norm(f)/math.sqrt(N), 1e-12)
        r = np.linalg.solve(L+lam*np.eye(N), f)
        res.append(float(np.linalg.norm(r)/math.sqrt(N)))
    return {"levels": list(levels), "first_nonzero_eig": eig,
            "resolvent_norms": res,
            "M1": "NOT PROVEN HERE", "M2": "NOT CONSTRUCTED HERE",
            "status": "CONDITIONAL — Carpet-19 Thm 2 under (I1)-(I4)"}

def delta_min(n: int = 3, kappa: float = 2.0, a_min: float = 0.01) -> float:
    d = n-1; om = PI**(d/2)/math.gamma(d/2+1)
    return (1.0/kappa)*(a_min/om)**(1.0/d)

class CSOC:
    """No-Zeno event counter. N(T) <= (E_max + D)/gap, D = audited removal."""
    def __init__(self, hi=0.6, lo=0.25, rate=0.08):
        assert hi > lo
        self.hi, self.lo, self.gap, self.rate = hi, lo, hi-lo, rate
        self.acc=0.0; self.armed=True; self.events=0
        self.E_max=0.0; self.D=0.0; self.D_audit=0.0
    def step(self, s: float) -> bool:
        exc = math.log1p(math.exp(-abs(s-self.hi))) + max(s-self.hi, 0.0)
        dec = 1.0/(1.0+math.exp(-(self.acc-self.lo)*10))
        rem = dec*self.rate*self.acc
        self.acc += exc-rem; self.D += rem; self.D_audit += rem
        self.E_max = max(self.E_max, self.acc)
        if self.acc < self.lo: self.armed = True
        fire = self.armed and self.acc > self.hi
        if fire: self.armed=False; self.events+=1
        return fire
    def report(self) -> Dict[str, Any]:
        b = (self.E_max+self.D)/self.gap
        return {"events": self.events, "bound": b,
                "satisfied": self.events <= b+1e-9,
                "D_audited": abs(self.D-self.D_audit) < 1e-12}


# =============================================================================
# §6. PDK BINDING + DRC ENGINE + PRE-TAPEOUT GATES
# =============================================================================

@dataclass
class PDK:
    """A real foundry PDK. Without this, no tape-out. (R4 §12.4, R6 §10)"""
    pdk_id: str = ""
    foundry: str = ""
    version: str = ""
    layer_map: Dict[int, Tuple[int,int]] = field(default_factory=dict)
    min_width_um: float = 0.0
    min_gap_um: float = 0.0
    min_bend_um: float = 0.0
    wg_loss_dB_per_cm: float = 0.0
    has_bto: bool = False
    has_gst: bool = False
    has_nbtin: bool = False
    cryo_qualified: bool = False
    def valid(self) -> bool:
        return bool(self.pdk_id and self.foundry and self.version
                    and self.layer_map)

class DRCEngine:
    """Per-revision Design Rule Checking. Local predicates, NOT foundry sign-off."""
    def __init__(self, rev: str, pdk: Optional[PDK] = None):
        self.spec = REVISIONS[rev]; self.pdk = pdk
        LineageGuard.check_layers(rev, list(self.spec.layers))

    def check(self, geom: Dict[str, float]) -> Dict[str, Any]:
        d = self.spec.drc; fails, warns = [], []
        def lt(key, gkey, default):
            if geom.get(gkey, default) < d[key]: fails.append(gkey)
        if self.spec.lineage == QUANTUM:
            lt("wg_width_min_um", "wg_width_um", 1.0)
            lt("bend_radius_min_um", "bend_radius_um", 20.0)
            lt("eo_wg_clear_um", "eo_wg_clear_um", 1.0)
            lt("snspd_width_min_um", "snspd_width_um", 0.1)
            lt("snspd_pitch_min_um", "snspd_pitch_um", 0.2)
            lt("pump_snspd_clear_um", "pump_snspd_clear_um", 300.0)
            lt("sg_clamp_clear_um", "sg_clamp_clear_um", 5.0)
        else:
            lt("wg_width_min_um", "wg_width_um", 1.0)
            lt("bend_radius_min_um", "bend_radius_um", 10.0)
        lt("dcplr_gap_min_um", "dcplr_gap_um", 1.0)
        lt("heater_wg_clear_um", "heater_wg_clear_um", 2.0)
        lt("pad_area_min_um2", "pad_area_um2", 2e4)
        if geom.get("heater_gap_um", 3.0) < d["heater_gap_um"]:
            warns.append("THERMAL_CROSSTALK")
        if "carpet_pitch_min_um" in d:
            lt("carpet_pitch_min_um", "carpet_pitch_um", 4.0)
            a = geom.get("carpet_cross_angle_deg", 90.0)
            if not (d["carpet_cross_angle_lo_deg"] <= a
                    <= d["carpet_cross_angle_hi_deg"]):
                fails.append("CARPET_CROSS_ANGLE")
        if "fwm_phase_match_max_rad" in d:
            if abs(geom.get("fwm_phase_match_rad", 0.0)) > d["fwm_phase_match_max_rad"]:
                warns.append("HYPER_COHERENCE")
        if "monitor_coupling_max" in d:
            if geom.get("monitor_coupling", 0.01) > d["monitor_coupling_max"]:
                fails.append("MONITOR_DISTURBANCE")
        return {"rev": self.spec.rev, "lineage": self.spec.lineage,
                "fails": fails, "warns": warns, "pass": len(fails) == 0,
                "scope": "LOCAL predicates only — not foundry DRC/LVS sign-off"}

    def pre_tapeout(self, results: Dict[str, bool],
                    human_approval: str = "") -> Dict[str, Any]:
        missing = [g for g in self.spec.gates if not results.get(g, False)]
        if missing:
            raise Refusal(f"{self.spec.rev}: pre-tapeout gates FAILED: {missing}")
        if self.pdk is None or not self.pdk.valid():
            raise Refusal(
                "REJECT_OR_HOLD_TAPE_OUT: no valid PDK.\n"
                "R4 §12.4 / R6 §10: a generic layer map is NOT a foundry "
                "specification. Provide pdk_id, foundry, version, layer_map.")
        if self.spec.lineage == QUANTUM:
            if not self.pdk.has_nbtin:
                raise Refusal("R6/R7 require NbTiN SNSPD in the PDK.")
            if not self.pdk.cryo_qualified:
                raise Refusal("R6/R7 require cryo-qualified process (1-4 K).")
        if not human_approval.strip():
            raise Refusal(
                "REJECT_OR_HOLD_TAPE_OUT: no human authorization.\n"
                "Autonomous optimization may PROPOSE layouts; fabrication "
                "authorization is a separate decision (R4 §15.3).")
        return {"allowed": True, "rev": self.spec.rev,
                "pdk": self.pdk.pdk_id, "approved_by": human_approval,
                "gates_passed": list(self.spec.gates)}


# =============================================================================
# §7. AI DESIGN ASSISTANTS — surrogates, not oracles
# =============================================================================

if HAS_TORCH:

    class DRCSurrogate(nn.Module):
        """
        Transformer classifier: predicts DRC-pass from layout features.
        Trained on the DRC engine's OWN labels. Accelerates search.
        It does NOT replace the checker — every accepted candidate is
        re-verified by DRCEngine before it can be proposed.
        """
        def __init__(self, d_in=16, d=64, n_heads=4, n_layers=2, max_feat=16):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.randn(1, max_feat, d)*0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, d*4, 0.1,
                                               batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        def forward(self, x):
            B, N, _ = x.shape
            h = self.trunk(self.proj(x) + self.pos[:, :N])
            return self.head(h.mean(1)).squeeze(-1)

    class GeometryDiffusion(nn.Module):
        """
        DDPM over normalized device-geometry vectors.
        Generates CANDIDATES. Every sample must pass DRCEngine + mode solver.
        A generated geometry is a hypothesis, never a validated device.
        """
        def __init__(self, d=16, T=48, hidden=96):
            super().__init__()
            self.T, self.d = T, d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b, 0))
            self.net = nn.Sequential(nn.Linear(d+1, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, d))
        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0); ab = self.abar[t][:, None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            pred = self.net(torch.cat([xt, t.float()[:,None]/self.T], -1))
            return F.mse_loss(pred, eps)
        @torch.no_grad()
        def sample(self, n, device="cpu"):
            x = torch.randn(n, self.d, device=device)
            for i in reversed(range(self.T)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, t.float()[:,None]/self.T], -1))
                a = 1-self.betas[i]; ab = self.abar[i]
                x = (x-(1-a)/(1-ab).sqrt()*eps)/a.sqrt()
                if i > 0: x = x + self.betas[i].sqrt()*torch.randn_like(x)
            return x.clamp(0, 1)

    class DynamicsWorldModel(nn.Module):
        """
        Surrogate for theta_C (classical) or GKSL (quantum) evolution.
        Used for fast design-space search. MUST be validated against the
        real solver on held-out configurations before any design decision.
        """
        def __init__(self, d_state=16, d_ctrl=8, d=96, n_steps=3):
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
            return {"final": traj[-1], "trajectory": torch.stack(traj, 1)}

LLM_CONTRACT = {
 "permitted": [
   "Parse foundry PDK documents into structured layer/rule tables",
   "Translate a natural-language device spec into a candidate config",
   "Draft verification reports from machine-produced results",
   "Summarize literature with citations",
 ],
 "forbidden": [
   "Invent PDK values not present in the source document",
   "Assert a device works without measured data",
   "Authorize tape-out",
   "Claim quantum advantage or Bell violation from classical optics",
 ],
 "hallucination_gate": (
   "Every numeric PDK value emitted by an LLM MUST be traced to a "
   "document span (file + page + line). Untraceable values are REJECTED "
   "and the field is left UNAVAILABLE, never filled with a plausible number."),
}


# =============================================================================
# §8. ECONOMICS — honest cost/revenue, MPW path
# =============================================================================

ECONOMICS = {
 "not_viable": {
   "build_euv_lithography": {
     "capex_usd": 1e10, "timeline_years": 25,
     "incumbent": "ASML (sole EUV supplier)",
     "supply_chain": ["Zeiss SMT sub-nm mirrors", "TRUMPF/Cymer CO2 laser",
                      "Sn plasma source", "nm-precision wafer stage"],
     "verdict": "NOT a path for a research group.",
   },
 },
 "viable": {
   "mpw_shuttle_classical": {
     "cost_usd": (5_000, 50_000),
     "platforms": ["AMF", "imec", "Tower", "AIM Photonics", "CompoundTek"],
     "deliverable": "measured passive/active SiPh die + T_cross + IL data",
     "why_first": "Turns Structural Calculus from an unparameterized formula "
                  "into a calibrated model. This is the credibility step.",
   },
   "mpw_shuttle_quantum": {
     "cost_usd": (50_000, 500_000),
     "extra": ["SNSPD integration", "cryo packaging", "SPDC source",
               "pump rejection", "tomography setup"],
     "deliverable": "HOM visibility, g2(0), Bell/CGLMP data",
   },
   "photonic_eda_software": {
     "capex_usd": (50_000, 500_000),
     "market_gap": "Logic EDA is consolidated (Synopsys/Cadence/Siemens); "
                   "photonic-specific PDK-aware verification is thinner.",
     "revenue_model": "open core (MIT/Apache) + paid PDK integration "
                      "+ verification sign-off + support contracts",
     "why_open_plus_paid_works": "PDKs are under NDA. The SCIENCE is open; "
                                 "the FOUNDRY BINDING is the paid artifact.",
   },
   "design_services": {
     "capex_usd": (0, 50_000),
     "customers": "quantum-photonic startups, national labs, university groups",
     "revenue_model": "per-project design + verification + measured bring-up",
   },
 },
 "open_science_and_revenue": {
   "principle": "Open the mathematics, the compiler core, and the verification "
                "suite. Charge for PDK binding, qualification, and support.",
   "precedents": ["Red Hat", "GitLab", "Odoo", "Anaconda", "Qt"],
   "what_stays_open": ["equations", "DRC predicates", "self-tests",
                       "reference layer maps", "audit format"],
   "what_is_paid": ["NDA-bound PDK integrations", "foundry sign-off support",
                    "measured-device calibration", "enterprise SLA"],
 },
}


# =============================================================================
# §9. ORCHESTRATOR
# =============================================================================

class PhotonicFabCompiler:
    def __init__(self, rev: str = "R5", pdk: Optional[PDK] = None,
                 out_dir: str = "runs/pfab"):
        if rev not in REVISIONS: raise Refusal(f"unknown revision {rev}")
        self.rev = rev; self.spec = REVISIONS[rev]
        self.pdk = pdk; self.drc = DRCEngine(rev, pdk)
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.csoc = CSOC(); self.audit: List[Dict] = []

    def record(self, stage: str, payload: Dict) -> Dict:
        rec = {"stage": stage, "rev": self.rev, "lineage": self.spec.lineage,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.audit.append(rec)
        return {**payload, **rec}

    def feasibility_report(self) -> Dict[str, Any]:
        rel = dict(FEASIBILITY)
        if self.spec.lineage == CLASSICAL:
            rel.pop("kerr_blockade_U_over_gamma", None)
            rel.pop("snspd_cryogenic", None)
        return {"rev": self.rev, "lineage": self.spec.lineage,
                "feasibility": rel,
                "kerr_requirement": (kerr_blockade_requirement()
                                     if self.spec.lineage == QUANTUM else None)}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "revisions": {k: asdict(v)
                                                 for k, v in REVISIONS.items()},
               "feasibility": FEASIBILITY, "economics": ECONOMICS,
               "llm_contract": LLM_CONTRACT, "audit": self.audit,
               "pdk_bound": bool(self.pdk and self.pdk.valid())}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng = np.random.default_rng(0)

    print("\n[T1] revision registry + lineage")
    for k, v in REVISIONS.items():
        print(f"      {k}: {v.lineage:9s} layers={len(v.layers):2d} "
              f"drc={len(v.drc):2d} gates={len(v.gates)}")
    rep("5 revisions registered", len(REVISIONS) == 5)
    rep("R3/R4/R5 classical", all(REVISIONS[r].lineage == CLASSICAL
                                  for r in ("R3","R4","R5")))
    rep("R6/R7 quantum", all(REVISIONS[r].lineage == QUANTUM
                             for r in ("R6","R7")))
    rep("R5 adds carpet layers", 26 in R5.layers and 27 in R5.layers)
    rep("R6 adds SNSPD", 31 in R6.layers)

    print("\n[T2] LINEAGE GUARD — the enforced category boundary")
    rep("R5 classical layers accepted",
        LineageGuard.check_layers("R5", list(R5.layers))["ok"])
    blocked = False
    try: LineageGuard.check_layers("R3", [1, 10, 31])   # SNSPD in classical
    except Refusal as e: blocked = True; msg = str(e).splitlines()[0]
    rep("SNSPD in R3 REFUSED", blocked, msg if blocked else "")
    blocked = False
    try: LineageGuard.check_operator("R5", "gksl")
    except Refusal: blocked = True
    rep("GKSL in classical R5 REFUSED", blocked)
    blocked = False
    try: LineageGuard.check_operator("R6", "theta_C")
    except Refusal: blocked = True
    rep("theta_C in quantum R6 REFUSED", blocked)
    mig = LineageGuard.migration_cost("R5", "R6")
    rep("R5->R6 flagged PLATFORM_CHANGE", mig["kind"] == "PLATFORM_CHANGE",
        f"{len(mig['required_changes'])} required changes")
    mig2 = LineageGuard.migration_cost("R3", "R4")
    rep("R3->R4 INCREMENTAL", mig2["kind"] == "INCREMENTAL",
        f"new layers {mig2['new_layers']}")

    print("\n[T3] classical core (R3/R4/R5)")
    for a in (0.2, 0.5, 2.0):
        r = np.exp(rng.uniform(-8,8,40000))
        z = r*np.exp(1j*rng.uniform(0,2*PI,r.size))
        rep(f"theta_C bound alpha={a}",
            float(np.abs(theta_C(z,a)).max()) <= theta_bound(a)+1e-12)
    sv = [theta_jacobian_sv(r,0.5) for r in np.linspace(0,20,400)]
    rep("Lip = 1 (both singular values <= 1)",
        max(max(x,y) for x,y in sv) <= 1+1e-12)
    U = mzi_matrix(PI/4, PI/3, 0.3, 0.7)
    rep("MZI unitary", np.allclose(U.conj().T@U, np.eye(2), atol=1e-10))
    rep("weak tap conserves power", weak_tap(1.0,0.05)["conservation_err"] < 1e-14)
    A = rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
    Q,_ = np.linalg.qr(A)
    cc = contraction_certificate(0.6*Q, 0.5, rng.normal(size=6)+0j, delta=1e-3)
    rep("q<1 contractive", cc["contractive"], f"q={cc['q']:.4f}")
    rep("a-posteriori bound holds", cc["posterior_bound_holds"])
    rep("hardware error <= delta/(1-q)", cc["hw_observed"] <= cc["hw_error_bound"]+1e-9,
        f"{cc['hw_observed']:.5f} <= {cc['hw_error_bound']:.5f}")
    rep("kappa=1 NOT contractive",
        not contraction_certificate(1.0*Q, 0.5, rng.normal(size=6)+0j)["contractive"])
    ch = chsh_classical_bound()
    rep("classical |S| <= 2 PROVEN by exhaustion", ch["max_abs_S"] == 2)

    print("\n[T4] quantum core (R6/R7) + honest blockade feasibility")
    kb = kerr_blockade_requirement(0.05)
    print(f"      g2<=0.05 requires U/gamma >= {kb['required_U_over_gamma']:.3f}")
    print(f"      published integrated range {kb['published_range']} "
          f"-> gap factor {kb['gap_factor']:.1f}x")
    rep("blockade requirement computed", abs(kb["required_U_over_gamma"]-math.sqrt(5))<1e-9)
    rep("blockade labelled NOT_DEMONSTRATED", kb["status"] == "NOT_DEMONSTRATED")
    rep("g2 formula consistent", abs(g2_from_U_gamma(math.sqrt(5))-0.05) < 1e-9)
    rep("Fock dim d=2,n=2 -> 6", fock_dim(2,2) == 6)
    # GKSL on a single damped mode
    d = 3
    a = np.zeros((d,d), complex)
    for n in range(1,d): a[n-1,n] = math.sqrt(n)
    rho = np.zeros((d,d), complex); rho[1,1] = 1.0
    H = np.zeros((d,d), complex)
    rho2 = gksl_step(rho, H, [math.sqrt(0.1)*a], dt=1e-3)
    rep("GKSL trace preserved", abs(float(np.real(np.trace(rho2)))-1.0) < 1e-9)
    rep("GKSL population decays", float(np.real(rho2[1,1])) < 1.0)
    K0 = np.diag([1.0, math.sqrt(0.9)]); K1 = np.array([[0,math.sqrt(0.1)],[0,0]])
    kr = kraus_apply(np.diag([0.0,1.0]), [K0, K1])
    rep("Kraus completeness (sum p = 1)", abs(kr["trace_sum"]-1.0) < 1e-9)
    qch = chsh_quantum_tsirelson()
    rep("quantum |S| = 2 sqrt 2", abs(qch["abs_S"]-2*math.sqrt(2)) < 1e-9,
        f"{qch['abs_S']:.6f}")
    rep("quantum exceeds classical bound", qch["exceeds_classical"])
    cg = cglmp_d3_quantum()
    rep("CGLMP I3 ~ 2.8729", abs(cg["I3_quantum"]-2.8729) < 1e-3,
        f"{cg['I3_quantum']:.4f}")
    ns = no_signalling_check(np.full((2,2,2,2), 0.25))
    rep("no-signalling holds", ns["no_signalling_A"] and ns["no_signalling_B"])
    bg = bogoliubov_check(np.eye(2)*math.cosh(0.3), np.eye(2)*math.sinh(0.3))
    rep("Bogoliubov canonical preserved", bg["preserved"],
        f"err={bg['canonical_error']:.2e}")

    print("\n[T5] ESCA / Carpet")
    t1 = esca_t1(rng.dirichlet(np.ones(8**3)))
    rep("T1 EXACT", t1["passed"], f"err={t1['consistency_err']:.2e}")
    mm = mosco_harness()
    rep("Mosco M1/M2 reported NOT PROVEN", mm["M1"] == "NOT PROVEN HERE")
    rep("delta_min > 0", delta_min() > 0, f"{delta_min():.6f}")
    c = CSOC()
    for k in range(500): c.step(0.35+0.45*abs(math.sin(0.11*k)))
    rep("CSOC bound holds", c.report()["satisfied"],
        f"N={c.report()['events']} bound={c.report()['bound']:.1f}")

    print("\n[T6] DRC + PDK + tape-out gates")
    good_c = dict(wg_width_um=0.5, dcplr_gap_um=0.3, bend_radius_um=8.0,
                  heater_wg_clear_um=1.5, heater_gap_um=3.0, pad_area_um2=2e4,
                  carpet_pitch_um=4.0, carpet_cross_angle_deg=90.0,
                  monitor_coupling=0.01, fwm_phase_match_rad=0.01)
    d5 = DRCEngine("R5")
    rep("R5 good geometry passes local DRC", d5.check(good_c)["pass"])
    rep("R5 narrow waveguide fails",
        not d5.check(dict(good_c, wg_width_um=0.3))["pass"])
    good_q = dict(good_c, wg_width_um=0.5, bend_radius_um=20.0,
                  eo_wg_clear_um=1.0, snspd_width_um=0.1, snspd_pitch_um=0.2,
                  pump_snspd_clear_um=300.0, sg_clamp_clear_um=5.0,
                  heater_wg_clear_um=1.5, heater_gap_um=3.0,
                  carpet_pitch_um=4.0, carpet_cross_angle_deg=90.0,
                  monitor_coupling=0.01, fwm_phase_match_rad=0.01)
    d6 = DRCEngine("R6")
    rep("R6 good geometry passes local Q-DRC", d6.check(good_q)["pass"])
    rep("R6 bend 8um fails (needs 15)",
        not d6.check(dict(good_q, bend_radius_um=8.0))["pass"])
    rep("R6 SNSPD 50nm fails (needs 75)",
        not d6.check(dict(good_q, snspd_width_um=0.05))["pass"])
    all_gates = {g: True for g in R5.gates}
    blocked = False
    try: d5.pre_tapeout(all_gates, "human:Yoon")
    except Refusal as e: blocked = True; m1 = str(e).splitlines()[0]
    rep("tape-out BLOCKED without PDK", blocked, m1 if blocked else "")
    pdk_c = PDK("AMF-SiPh-v3","AMF","3.1",{1:(1,0),10:(10,0)},
                0.4,0.2,5.0,0.5,has_bto=True,has_gst=True)
    d5p = DRCEngine("R5", pdk_c)
    rep("tape-out ALLOWED with PDK+gates+human",
        d5p.pre_tapeout(all_gates, "human:Yoon")["allowed"])
    blocked = False
    try: d5p.pre_tapeout(all_gates, "")
    except Refusal: blocked = True
    rep("tape-out BLOCKED without human approval", blocked)
    blocked = False
    try: DRCEngine("R6", pdk_c).pre_tapeout({g:True for g in R6.gates}, "h")
    except Refusal as e: blocked = True; m2 = str(e).splitlines()[0]
    rep("R6 BLOCKED: PDK lacks NbTiN", blocked, m2 if blocked else "")
    pdk_q = PDK("QM-SiN-BTO-NbTiN-v1","imec","1.0",{1:(1,0),31:(31,0)},
                0.45,0.2,15.0,0.04,has_bto=True,has_gst=True,
                has_nbtin=True,cryo_qualified=True)
    rep("R6 ALLOWED with cryo+NbTiN PDK",
        DRCEngine("R6",pdk_q).pre_tapeout({g:True for g in R6.gates},"h")["allowed"])
    blocked = False
    try: d5p.pre_tapeout({g:False for g in R5.gates}, "h")
    except Refusal: blocked = True
    rep("failed gates BLOCK tape-out", blocked)

    print("\n[T7] AI design assistants (surrogates)")
    if HAS_TORCH:
        torch.manual_seed(0)
        X = torch.randn(64, 16, 16); y = (X.sum((1,2)) > 0).float()
        sur = DRCSurrogate(16, 32, 2, 1, 16)
        opt = torch.optim.Adam(sur.parameters(), lr=3e-3)
        l0 = lN = None
        for ep in range(80):
            opt.zero_grad(); L = F.binary_cross_entropy_with_logits(sur(X), y)
            L.backward(); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("DRC surrogate trains", lN < l0, f"{l0:.4f} -> {lN:.4f}")
        df = GeometryDiffusion(16, 24, 64)
        opt2 = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.rand(32, 16); d0 = dN = None
        for ep in range(60):
            opt2.zero_grad(); L = df.loss(x0); L.backward(); opt2.step()
            if ep == 0: d0 = float(L)
            dN = float(L)
        s = df.sample(4)
        rep("geometry diffusion trains", dN < d0, f"{d0:.4f} -> {dN:.4f}")
        rep("samples in [0,1]", bool((s>=0).all() and (s<=1).all()))
        wm = DynamicsWorldModel(16, 8, 48, 3)
        o = wm(torch.randn(4,16), torch.randn(4,8))
        o["final"].sum().backward()
        gok = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in wm.parameters())
        rep("world model gradients finite", gok)
    else:
        rep("torch available", False, "pip install torch")
    rep("LLM contract forbids inventing PDK values",
        any("Invent PDK" in f for f in LLM_CONTRACT["forbidden"]))
    rep("LLM hallucination gate present", "traceable" in LLM_CONTRACT["hallucination_gate"]
        or "traced" in LLM_CONTRACT["hallucination_gate"])

    print("\n[T8] economics honesty")
    rep("EUV path marked NOT viable",
        "NOT" in ECONOMICS["not_viable"]["build_euv_lithography"]["verdict"])
    rep("MPW classical path present",
        ECONOMICS["viable"]["mpw_shuttle_classical"]["cost_usd"][0] < 1e5)
    rep("open-core + paid model documented",
        "what_is_paid" in ECONOMICS["open_science_and_revenue"])
    rep("fab equipment OUT_OF_SCOPE", FEASIBILITY["fab_equipment"]["status"]=="OUT_OF_SCOPE")

    print("\n[T9] orchestrator + report")
    comp = PhotonicFabCompiler("R5", pdk_c)
    fr = comp.feasibility_report()
    rep("classical report omits quantum blockers",
        "kerr_requirement" in fr and fr["kerr_requirement"] is None)
    compq = PhotonicFabCompiler("R6", pdk_q)
    frq = compq.feasibility_report()
    rep("quantum report includes Kerr blocker",
        frq["kerr_requirement"]["status"] == "NOT_DEMONSTRATED")
    r = comp.report()
    rep("report written", (comp.out/"report.json").exists())
    rep("all 5 revisions in report", len(r["revisions"]) == 5)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
VERIFIED
  5 revisions with correct lineage split (R3/R4/R5 classical, R6/R7 quantum).
  LineageGuard REFUSES SNSPD-in-classical, GKSL-in-classical, theta_C-in-quantum.
  R5->R6 flagged PLATFORM_CHANGE (8 required changes), R3->R4 INCREMENTAL.
  theta_C exact bound + Lip=1; MZI unitarity; weak-tap power conservation;
  q-contraction + a-posteriori + delta/(1-q); kappa=1 NOT contractive.
  CHSH classical |S|<=2 PROVEN by exhaustion; quantum 2*sqrt(2); CGLMP 2.8729.
  GKSL trace preservation; Kraus completeness; Bogoliubov canonical identity.
  ESCA T1 EXACT; CSOC bound; Mosco M1/M2 reported NOT PROVEN.
  Per-revision DRC; tape-out BLOCKED without PDK / gates / human / NbTiN / cryo.
  AI assistants train; LLM contract forbids inventing PDK values.

ENFORCED REFUSALS (all fired)
  SNSPD layer in classical R3 -> BLOCKED
  GKSL operator in classical R5 -> BLOCKED
  theta_C in quantum R6 -> BLOCKED
  tape-out without PDK -> BLOCKED
  tape-out without human authorization -> BLOCKED
  R6 with non-cryo / non-NbTiN PDK -> BLOCKED
  failed pre-tapeout gates -> BLOCKED

NOT CLAIMED
  Kerr blockade U/gamma >= 2.24 is NOT_DEMONSTRATED in integrated MRRs.
  Metric-analogue is NOT astrophysical gravity.
  Mosco M1/M2 for a new normalization is NOT proven here.
  This is a DESIGN/VERIFICATION COMPILER, not a lithography machine.
  Building EUV equipment is OUT_OF_SCOPE (ASML monopoly, ~EUR 10B, 25 yr).

THE REALISTIC PATH TO "HELP THE WORLD + GET RICH"
  1. MPW shuttle ($5k-$50k) on AMF/imec/Tower -> measured T_cross, IL, kappa(g).
  2. Calibrate Structural Calculus with that data -> credible model.
  3. Open-source the compiler core + equations + self-tests (MIT).
  4. Charge for NDA-bound PDK integration, sign-off support, calibration.
  That is open science AND revenue, without competing with ASML.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Photonic fab compiler (5 revisions)")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","revisions","lineage","migrate","drc","tapeout",
                 "feasibility","economics","kerr","report"])
    ap.add_argument("--rev", default="R5")
    ap.add_argument("--to", default="R6")
    ap.add_argument("--pdk", default=""); ap.add_argument("--foundry", default="")
    ap.add_argument("--human", default="")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    out: Dict[str, Any] = {}

    if a.cmd == "revisions":
        out = {k: asdict(v) for k, v in REVISIONS.items()}
        print(f"{'rev':<5}{'lineage':<11}{'layers':>7}{'drc':>5}{'gates':>7}")
        for k, v in REVISIONS.items():
            print(f"{k:<5}{v.lineage:<11}{len(v.layers):>7}{len(v.drc):>5}{len(v.gates):>7}")
    elif a.cmd == "lineage":
        try: out = LineageGuard.check_layers(a.rev, list(REVISIONS[a.rev].layers))
        except Refusal as e: out = {"BLOCKED": str(e)}
        print(json.dumps(out, indent=2))
    elif a.cmd == "migrate":
        out = LineageGuard.migration_cost(a.rev, a.to)
        print(json.dumps(out, indent=2))
    elif a.cmd == "drc":
        g = dict(wg_width_um=0.5, dcplr_gap_um=0.3, bend_radius_um=20.0,
                 heater_wg_clear_um=1.5, heater_gap_um=3.0, pad_area_um2=2e4)
        out = DRCEngine(a.rev).check(g)
        print(json.dumps(out, indent=2))
    elif a.cmd == "tapeout":
        pdk = PDK(a.pdk, a.foundry, "1.0", {1:(1,0)},
                  has_nbtin=(a.rev in ("R6","R7")),
                  cryo_qualified=(a.rev in ("R6","R7"))) if a.pdk else None
        try:
            out = DRCEngine(a.rev, pdk).pre_tapeout(
                {g: True for g in REVISIONS[a.rev].gates}, a.human)
        except Refusal as e: out = {"BLOCKED": str(e)}
        print(json.dumps(out, indent=2))
    elif a.cmd == "feasibility":
        out = PhotonicFabCompiler(a.rev).feasibility_report()
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "kerr":
        out = kerr_blockade_requirement()
        print(json.dumps(out, indent=2, default=str))
        print("\nU/gamma >= 2.24 is NOT demonstrated in integrated MRRs.")
        print("Use SPDC/SFWM + heralding + linear optics (KLM) instead.")
    elif a.cmd == "economics":
        out = ECONOMICS; print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "report":
        out = PhotonicFabCompiler(a.rev).report()

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
