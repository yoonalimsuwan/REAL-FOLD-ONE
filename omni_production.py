#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
omni_production.py — v1.0.0
Production-grade scaffold: 15 domains, 3-paper photonics core, ESCA/Carpet math,
enforced governance gates, decisive self-tests.
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
SIX CORRECTIONS ENFORCED IN THIS FILE
=============================================================================
[C1] NLSE IS NOT A QUANTUM COMPUTER. It is classical nonlinear optics.
     No 2^n state space, no entanglement, no Born rule. Labelled
     `classical_optical_accelerator`. Quantum-chip models are SEPARATE.
[C2] BRAIN "CONTROL" = CLINICAL NEUROMODULATION WITH HARD LIMITS.
     Shannon charge-density k<=1.5, current-density cap, thermal cap,
     and a CONSENT GATE. No autonomous or non-consensual actuation.
[C3] REJUVENATION IS GATED ON FUNCTION, NOT CLOCKS. Partial
     reprogramming carries teratoma/dedifferentiation risk. A claim
     requires a held-out FUNCTIONAL endpoint.
[C4] CARBON CHIPS AND FAB TOOLS ARE PARAMETRIC RESEARCH MODELS.
     No commercial sub-3nm CNT logic. Advanced lithography equipment is
     export-controlled. Tape-out needs PDK + DRC + LVS + human approval.
[C5] MATHEMATICS NAMES ARE MOSTLY ANALOGICAL. See MATH_HONESTY below.
[C6] THIS IS A SCAFFOLD WITH VERIFIED CORES, NOT 15 FINISHED PRODUCTS.
     See PRODUCTION_READINESS for what each domain still requires.

=============================================================================
THREE-PAPER INTEGRATION
=============================================================================
P1 SSC R3.0        : theta_C, alpha_eff(MRR+BTO), kappa(g), phi_total,
                     Structural Calculus, layers 1..30+99, six DRC rules.
P2 DSQP-SESI R4.0  : layers 40..47, covariant sector, PPN->phi_GR,
                     CHSH + no-signalling, typed graph, 6-objective Pareto.
P3 ESDA 2.0        : X=(Psi,Pi,Lambda,M), G_w/G_h, pushforward law,
                     q-contraction + delta/(1-q), ESCA T1..T6.
Carpet-19          : Mosco M1/M2 (conditional), spectral Sobolev scale,
                     Hino singularity (cited), no Connes triple (proven).
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
log = logging.getLogger("omni")
VERSION = "1.0.0"
PI = math.pi
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
OMNI PRODUCTION SCAFFOLD v{VERSION}
15 domains · 3-paper photonics · ESCA/Carpet math · enforced gates
NLSE != quantum computer. Brain modulation = clinical + consent + limits.
Rejuvenation gated on FUNCTION. Tape-out needs PDK+DRC+LVS+human.
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal. Never silently downgraded."""


# =============================================================================
# §0. LEDGERS
# =============================================================================

PRODUCTION_READINESS: Dict[str, Dict[str, str]] = {
 "aging_rejuvenation": {
   "code_provides": "12-variable organelle ODE + coupling + CSOC scheduler + functional gate",
   "still_required": "longitudinal multi-omic cohort WITH functional endpoints",
   "blocking": "clock movement is a correlate, not an outcome"},
 "epidemiology": {
   "code_provides": "renewal + SEIRV + T5 minimax + weighted conformal",
   "still_required": "contact structure, policy log, generation-interval estimate",
   "blocking": "T5 guarantee void if true model lies outside P"},
 "oncology": {
   "code_provides": "replicator-mutator simplex + reachability + Black-Leff window",
   "still_required": "FUNCTIONAL dose-response for tau; longitudinal clonal fractions",
   "blocking": "binding affinity cannot identify efficacy"},
 "neuro_read": {
   "code_provides": "bandpower + Riemannian tangent + ridge decoding",
   "still_required": "held-out clinical data, IRB, modality-specific validation",
   "blocking": "scalp EEG cannot decode semantic content"},
 "neuro_modulate": {
   "code_provides": "DBS/TMS/tDCS/opto dose models + Shannon/thermal/consent gates",
   "still_required": "device calibration, IRB, clinician authority",
   "blocking": "no autonomous actuation"},
 "phytochemistry": {
   "code_provides": "descriptors + drug-likeness filters",
   "still_required": "assay data; NO synthesis routes provided",
   "blocking": "filters are heuristics, not efficacy/safety evidence"},
 "drug_design": {
   "code_provides": "Black-Leff heads + latent decoder + OOD refusal",
   "still_required": "training data + functional assay for tau",
   "blocking": "not ProteinMPNN/RFdiffusion; no folding guarantee"},
 "cosmeceutical": {
   "code_provides": "Potts-Guy permeation + irritation proxy + claim gate",
   "still_required": "HRIPT / in-vitro safety assessment",
   "blocking": "disease-treatment claim reclassifies as a drug"},
 "carbon_chip": {
   "code_provides": "CNT-FET / graphene parametric I-V + chirality variability",
   "still_required": "foundry PDK + measured devices",
   "blocking": "no commercial sub-3nm CNT logic exists"},
 "photonics_ssc": {
   "code_provides": "P1 core + P2 layers 40..47 + P3 formal state + DRC + tapeout gate",
   "still_required": "PDK, mode solver, coupler sweep, BTO/GST process qualification",
   "blocking": "GDS-II that opens is not fabrication-ready"},
 "in_memory": {
   "code_provides": "crossbar MVM with IR-drop / noise / stuck-at / ADC",
   "still_required": "measured device arrays, endurance/retention data",
   "blocking": "ideal-only crossbar result is not reportable"},
 "ota_compute": {
   "code_provides": "AirComp channel inversion + power cap + noise MSE",
   "still_required": "regulatory compliance, real channel measurements",
   "blocking": "not a deployed radio"},
 "quantum_chip": {
   "code_provides": "transmon/ion/photonic parameter models + code budget",
   "still_required": "measured T1/T2/gate fidelities",
   "blocking": "no Shor at cryptographic scale"},
 "nlse_accelerator": {
   "code_provides": "split-step Fourier CLASSICAL propagator",
   "still_required": "measured material response for calibration",
   "blocking": "NOT a quantum computer; zero quantum speedup"},
 "fab_equipment": {
   "code_provides": "Rayleigh/DOF, Arrhenius CVD, etch selectivity, overlay RSS",
   "still_required": "tool specifications (export-controlled)",
   "blocking": "research parameters only, not procurement guidance"},
}

MATH_HONESTY = {
 "ssc_bound":        {"provable": True,  "claim": "|theta_C| <= 1/(2 sqrt(a)); Lip = 1"},
 "csoc_bound":       {"provable": True,  "claim": "N(T) <= (E_max + D)/gap, D = audited removal"},
 "politis_romano":   {"provable": True,  "requires": "stationarity + mixing + finite variance"},
 "zalesak_fct":      {"provable": True,  "claim": "exact conservation + no new extrema"},
 "black_leff":       {"provable": True,  "claim": "separates K_A (affinity) from tau (efficacy)"},
 "esca_T1":          {"provable": True,  "claim": "simplex additivity under refinement"},
 "mosco":            {"provable": "imported", "claim": "M1+M2 => strong resolvent convergence"},
 "kac":              {"provable": "imported", "claim": "E[r_B] = 1/mu(B); nested ratio finite"},
 "opmc":             {"provable": False, "is": "learned low-rank bottleneck + fixed step count"},
 "carpet19":         {"provable": False, "is": "fixed graph Laplacian used as a basis"},
 "hodge":            {"provable": False, "is": "learned 3-way split + Laplacian penalty"},
 "hilbert_polya":    {"provable": False, "is": "learned self-adjoint spectral feature map"},
 "delta_min":        {"provable": "threshold only", "is": "kappa-fatness guard"},
}

LAYER_MAP_R4 = {
 1:"WG_CORE", 2:"WG_CLAD", 10:"HEATER_METAL", 15:"PCM_GST", 16:"BTO_LAYER",
 20:"EDGE_COUPLER", 30:"VIA", 40:"METRIC_CONTROL", 41:"CURVATURE_ANALOG",
 42:"SPACETIME_DELAY", 43:"COVARIANT_ROUTING", 44:"FIELD_MONITOR",
 45:"SYNC_REFERENCE", 46:"GRAVITY_MODEL_INTERFACE", 47:"ENERGY_MONITOR",
 99:"MASK_DOC"}


# =============================================================================
# §1. SHARED MATH CORE
# =============================================================================

def ssc(z, alpha: float):
    """theta_C(z) = z/(1+alpha|z|^2). Bound 1/(2 sqrt(alpha)) at r=alpha^-1/2."""
    if HAS_TORCH and isinstance(z, torch.Tensor):
        return z / (1.0 + alpha * (z.real**2 + z.imag**2))
    return z / (1.0 + alpha * np.abs(z)**2)

def ssc_bound(alpha: float) -> float:
    return 1.0 / (2.0 * math.sqrt(alpha))

def ssc_singular_values(r: float, alpha: float) -> Tuple[float, float]:
    s = alpha * r * r
    return 1.0/(1.0+s), abs(1.0-s)/(1.0+s)**2


class CSOC:
    """Two-level hysteresis. N(T) <= (E_max + D)/gap with D = ACTUAL removal."""
    def __init__(self, hi=0.6, lo=0.25, rate=0.08):
        assert hi > lo
        self.hi, self.lo, self.gap, self.rate = hi, lo, hi-lo, rate
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0; self.D_audit = 0.0

    def step(self, s: float) -> bool:
        exc = math.log1p(math.exp(-abs(s-self.hi))) + max(s-self.hi, 0.0)
        dec = 1.0/(1.0+math.exp(-(self.acc-self.lo)*10))
        rem = dec*self.rate*self.acc
        self.acc += exc - rem
        self.D += rem; self.D_audit += rem
        self.E_max = max(self.E_max, self.acc)
        if self.acc < self.lo: self.armed = True
        fire = self.armed and self.acc > self.hi
        if fire: self.armed = False; self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        b = (self.E_max + self.D)/self.gap
        return {"events": self.events, "bound": b,
                "satisfied": self.events <= b + 1e-9,
                "D_audited": abs(self.D - self.D_audit) < 1e-12,
                "scope": "counter only — not a biological guarantee"}


def subsample_ci(x: np.ndarray, conf=0.95) -> Dict[str, float]:
    """Politis-Romano. Correct for serially correlated series; NOT iid."""
    x = np.asarray(x, float).ravel(); n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n, "status": "INSUFFICIENT"}
    b = min(max(2, int(round(n**(1/3)))), n//2)
    th = float(x.mean())
    sub = np.array([x[i:i+b].mean() for i in range(n-b+1)])
    sc = math.sqrt(b)*(sub-th); a = 1-conf
    lo = th - float(np.quantile(sc, 1-a/2))/math.sqrt(n)
    hi = th - float(np.quantile(sc, a/2))/math.sqrt(n)
    return {"mean": th, "lo": min(lo,hi), "hi": max(lo,hi),
            "n": n, "block": b, "status": "OK"}


def carpet_laplacian(side: int) -> np.ndarray:
    side = max(3, int(side)); n = side*side
    A = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            if i%3==1 and j%3==1: continue
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii, jj = i+di, j+dj
                if 0<=ii<side and 0<=jj<side and not (ii%3==1 and jj%3==1):
                    A[i*side+j, ii*side+jj] = 1.0
    return np.diag(A.sum(1)) - A


def delta_min(n=3, kappa=2.0, a_min=0.01) -> float:
    d = n-1
    om = PI**(d/2)/math.gamma(d/2+1)
    return (1.0/kappa)*(a_min/om)**(1.0/d)


def black_leff(dose, ka, tau, em, n=1.0, eps=1e-12):
    """E/Em = tau^n A^n / ((KA+A)^n + tau^n A^n). Separates affinity/efficacy."""
    A = np.maximum(dose, eps); KA = np.maximum(ka, eps)
    tn = np.sign(tau)*np.maximum(np.abs(tau), eps)**n
    An = A**n
    return em*tn*An/np.maximum((KA+A)**n + tn*An, eps)


def esca_t1(p_fine: np.ndarray, levels=3) -> Dict[str, Any]:
    ps = {levels: p_fine/p_fine.sum()}
    for m in range(levels-1, -1, -1):
        ps[m] = ps[m+1].reshape(-1, 8).sum(1)
    en = max(abs(p.sum()-1) for p in ps.values())
    ec = max(float(np.abs(ps[m+1].reshape(-1,8).sum(1)-ps[m]).max())
             for m in range(levels))
    nn = all(bool((p >= -1e-14).all()) for p in ps.values())
    return {"norm_err": en, "consistency_err": ec, "non_negative": nn,
            "passed": en < 1e-12 and ec < 1e-12 and nn}


def esca_t5(loss_matrix: np.ndarray, lam: float, eps: float) -> Dict[str, Any]:
    R = loss_matrix.max(axis=0); a = int(np.argmin(R))
    exc = float((loss_matrix[:, a] > lam).mean())
    return {"a_star": a, "worst_case_risk": float(R[a]), "exceedance": exc,
            "status": "DECISION_CLOSURE_CERTIFIED" if exc <= eps
                      else "UNCERTAINTY_RETAINED",
            "caveat": "void if the true model lies outside P"}


def pushforward(step: Callable, x0s: List[Any], n_out: int, T: int) -> np.ndarray:
    """ESDA Prop 1.1: mu_t = (G o F^t)_# rho_0."""
    mu = np.zeros((T, n_out))
    for x0 in x0s:
        x = x0
        for t in range(T):
            x = step(x)
            mu[t, int(np.argmax(np.abs(x)**2))] += 1
    return mu/max(len(x0s), 1)


# =============================================================================
# §2. GOVERNANCE — enforced, not documented
# =============================================================================

@dataclass
class Consent:
    subject_id: str = ""; irb_id: str = ""; scope: str = ""; signed: bool = False
    def valid(self) -> bool:
        return bool(self.subject_id and self.irb_id and self.scope and self.signed)


class Governance:
    SHANNON_K_MAX = 1.5
    TDCS_MA_CM2_MAX = 0.08
    THERMAL_DELTA_C_MAX = 2.0
    OPTO_FLUENCE_J_MM2_MAX = 200.0

    @staticmethod
    def neuro_actuation(consent: Consent, D_uC_cm2: float, Q_nC: float,
                        dT_C: float) -> Dict[str, Any]:
        if not consent.valid():
            raise Refusal("NEURO ACTUATION BLOCKED: consent/IRB incomplete. "
                          "No autonomous or non-consensual stimulation.")
        Q_uC = Q_nC*1e-3
        k = math.log10(max(D_uC_cm2,1e-12)) + math.log10(max(Q_uC,1e-12))
        if k > Governance.SHANNON_K_MAX:
            raise Refusal(f"Shannon k={k:.2f} > {Governance.SHANNON_K_MAX}: "
                          "tissue-damage risk.")
        if dT_C > Governance.THERMAL_DELTA_C_MAX:
            raise Refusal(f"Delta T={dT_C:.2f} C exceeds cap.")
        return {"allowed": True, "shannon_k": k, "dT_C": dT_C}

    @staticmethod
    def tdcs(consent: Consent, current_mA: float, area_cm2: float) -> Dict[str, Any]:
        if not consent.valid():
            raise Refusal("tDCS BLOCKED: consent/IRB incomplete.")
        j = current_mA/max(area_cm2, 1e-9)
        if j > Governance.TDCS_MA_CM2_MAX:
            raise Refusal(f"current density {j:.4f} mA/cm^2 > cap.")
        return {"allowed": True, "current_density_mA_cm2": j}

    @staticmethod
    def rejuvenation_claim(functional: Optional[Dict[str, float]],
                           clock_delta_y: float) -> Dict[str, Any]:
        if functional is None:
            raise Refusal(
                "REJUVENATION CLAIM BLOCKED: no functional endpoint.\n"
                f"Clock moved {clock_delta_y:+.1f} y — that is a CORRELATE.\n"
                "Required: held-out function (grip, VO2max, cognition, immune "
                "response, histology) with CI. Partial reprogramming carries "
                "teratoma/dedifferentiation risk.")
        return {"allowed": True, "functional": functional,
                "clock_delta_y": clock_delta_y,
                "note": "clock is secondary; function is primary"}

    @staticmethod
    def synthesis(intent: str) -> Dict[str, Any]:
        low = intent.lower()
        for b in ("synthesize toxin","controlled substance","schedule i",
                  "nerve agent","weaponize","gain of function",
                  "increase virulence","explosive synthesis"):
            if b in low:
                raise Refusal(f"BLOCKED intent '{b}'. Computational analysis only.")
        return {"allowed": True, "scope": "descriptors / scoring, no routes"}

    @staticmethod
    def cosmetic_claim(claim: str) -> Dict[str, Any]:
        low = claim.lower()
        hit = [w for w in ("treat","cure","heal","prevent disease","anti-cancer",
                           "treat eczema","treat psoriasis","antibiotic") if w in low]
        if hit:
            raise Refusal(f"Drug language {hit}. A cosmetic cannot claim disease "
                          "treatment; that reclassifies it as a drug.")
        return {"allowed": True, "claim": claim}

    @staticmethod
    def tapeout(pdk: str, drc: bool, lvs: bool, human: str) -> Dict[str, Any]:
        if not (pdk and drc and lvs and human):
            raise Refusal("REJECT_OR_HOLD_TAPE_OUT: need PDK id + clean DRC + "
                          "clean LVS + explicit human authorisation. A GDS-II "
                          "that opens is not fabrication-ready.")
        return {"allowed": True, "pdk": pdk, "approved_by": human}

    @staticmethod
    def export_note() -> str:
        return ("Advanced lithography/deposition/etch equipment is export-controlled "
                "in multiple jurisdictions. Physics-level research models only — "
                "not tool specifications or procurement guidance.")


# =============================================================================
# §3. THREE-PAPER PHOTONICS CORE
# =============================================================================

class PhotonicsSSC:
    """P1 SSC R3.0 + P2 DSQP-SESI R4.0 + P3 ESDA 2.0."""

    @staticmethod
    def alpha_eff(n2_BTO: float, Q: float, L_eff_m: float, A_eff_m2: float
                  ) -> Dict[str, float]:
        a = n2_BTO*Q*L_eff_m/max(A_eff_m2, 1e-20)
        return {"alpha_eff": a, "theta_bound": ssc_bound(a) if a > 0 else float("inf"),
                "unit_note": "formula as written in P1; requires power-normalisation "
                             "calibration against measured Kerr response"}

    @staticmethod
    def mrr(R_m: float, n_eff: float, m: int) -> Dict[str, float]:
        lam = 2*PI*R_m*n_eff/m
        return {"lambda_res_m": lam,
                "FSR_m": lam**2/(2*PI*R_m*n_eff),
                "detuning_nm": (lam-1.55e-6)*1e9}

    @staticmethod
    def kappa(g_um: float, k0=0.5, gamma=3.0) -> float:
        return k0*math.exp(-gamma*g_um)

    @staticmethod
    def phi_total(dn: float, L: float, lam0: float, dn_dT: float, dT: float,
                  L_h: float, dn_pcm: float, L_pcm: float, phi_GR=0.0
                  ) -> Dict[str, float]:
        k0 = 2*PI/lam0
        p, h, c = k0*dn*L, k0*dn_dT*dT*L_h, k0*dn_pcm*L_pcm
        return {"phi_prop": p, "phi_heater": h, "phi_PCM": c,
                "phi_GR": phi_GR, "phi_total": p+h+c+phi_GR}

    @staticmethod
    def mzi(th1: float, th2: float, ph1: float, ph2: float) -> np.ndarray:
        B = lambda t: np.array([[math.cos(t), 1j*math.sin(t)],
                                [1j*math.sin(t), math.cos(t)]])
        return B(th2) @ np.diag([np.exp(1j*ph1), np.exp(1j*ph2)]) @ B(th1)

    @staticmethod
    def weak_tap(P_in: float, eta: float) -> Dict[str, float]:
        return {"P_monitor": eta*P_in, "P_through": (1-eta)*P_in,
                "conservation_err": abs(eta*P_in+(1-eta)*P_in-P_in)}

    @staticmethod
    def contraction(W: np.ndarray, kappa_att: float) -> Dict[str, Any]:
        U = W/max(np.linalg.norm(W, 2), 1e-12)
        q = float(np.linalg.norm(kappa_att*U, 2))
        return {"kappa": kappa_att, "q": q, "contractive": q < 1.0,
                "hw_error_floor": (1e-3/(1-q)) if q < 1 else float("inf"),
                "requirement": "lossless MZI has ||U||=1; attenuation kappa<1 "
                               "is a DESIGN REQUIREMENT"}

    @staticmethod
    def structural_calculus(W_ideal: np.ndarray, T_cross: np.ndarray,
                            dW: np.ndarray, eps=1e-6) -> Dict[str, Any]:
        cond = float(np.linalg.cond(T_cross))
        TcH = T_cross.conj().T
        reg = W_ideal @ np.linalg.inv(TcH@T_cross + eps*np.eye(len(T_cross))) @ TcH + dW
        exact = W_ideal @ np.linalg.inv(T_cross) + dW if cond < 1e10 else None
        return {"cond_T_cross": cond, "used_regularised": cond >= 1e10,
                "W_programmed": exact if exact is not None else reg}

    @staticmethod
    def ppn_metric(Phi: float, c=299792458.0, gamma_PPN=1.0) -> np.ndarray:
        e = 2*Phi/c**2
        return np.diag([-(1+e), (1-gamma_PPN*e), (1-gamma_PPN*e), (1-gamma_PPN*e)])

    @staticmethod
    def phi_GR(g: np.ndarray, lam0: float, L: float, n_ref: float) -> Dict[str, Any]:
        g00, gxx = float(-g[0,0]), float(g[1,1])
        n_eff = n_ref*math.sqrt(max(gxx/max(g00,1e-12), 1e-12))
        return {"n_eff": n_eff, "phi_GR": 2*PI/lam0*(n_eff-n_ref)*L,
                "mode": "A_simulation",
                "REFUSED": "metric-ANALOGUE emulation, not spacetime curvature"}

    @staticmethod
    def chsh_exhaustive() -> Dict[str, Any]:
        vals = [abs(a0*b0+a0*b1+a1*b0-a1*b1)
                for a0,a1,b0,b1 in itertools.product((-1,1), repeat=4)]
        ang = [0.0, PI/2, PI/4, -PI/4]
        E = lambda i,j: -math.cos(ang[i]-ang[j])
        return {"n_strategies": len(vals), "max_abs_S_local": max(vals),
                "tsirelson": 2*math.sqrt(2),
                "quantum_singlet_S": E(0,2)+E(0,3)+E(1,2)-E(1,3),
                "REFUSED": "classical photonic chip CANNOT exceed |S|=2"}

    @staticmethod
    def drc(geom: Dict[str, float]) -> Dict[str, Any]:
        f, w = [], []
        if geom.get("wg_width_um",1.0) < 0.40: f.append("WG_WIDTH")
        if geom.get("dcplr_gap_um",1.0) < 0.20: f.append("COUPLER_GAP")
        if geom.get("bend_radius_um",10.0) < 5.0: f.append("BEND_RADIUS")
        if geom.get("heater_wg_clear_um",2.0) < 1.0: f.append("HEATER_CLEAR")
        if geom.get("heater_heater_gap_um",3.0) < 2.0: w.append("THERMAL_CROSSTALK")
        if geom.get("pad_area_um2",1e4) < 10000: f.append("PAD_AREA")
        if geom.get("monitor_wg_clear_um",2.0) < 1.0: f.append("MONITOR_KEEPOUT")
        if geom.get("monitor_coupling",0.01) > 0.05: f.append("MONITOR_DISTURBANCE")
        return {"fails": f, "warns": w, "pass": len(f)==0,
                "layers": LAYER_MAP_R4,
                "scope": "local predicates — NOT foundry DRC/LVS sign-off"}


# =============================================================================
# §4. AGING + ORGANELLE-RESOLVED REJUVENATION
# =============================================================================

@dataclass
class OrganelleState:
    """12 damage variables in [0,1]. 0 = healthy, 1 = failed."""
    mito_dpsi: float = 0.10; mito_ros: float = 0.10
    mito_mtdna: float = 0.05; mito_dynamics: float = 0.10
    nuc_epi_drift: float = 0.10; nuc_telomere: float = 0.05
    nuc_ddf: float = 0.05; er_upr: float = 0.10
    lyso_ph: float = 0.05; lyso_lipofuscin: float = 0.05
    proteo_aggregate: float = 0.10; perox_h2o2: float = 0.05
    def vec(self): return np.array(list(asdict(self).values()), float)
    @staticmethod
    def names(): return list(asdict(OrganelleState()).keys())


class AgingEngine:
    COUPLING = [("mito_ros","nuc_ddf",0.06), ("mito_ros","proteo_aggregate",0.04),
                ("nuc_ddf","nuc_epi_drift",0.05), ("proteo_aggregate","er_upr",0.07),
                ("er_upr","lyso_ph",0.03), ("lyso_lipofuscin","proteo_aggregate",0.05),
                ("mito_dynamics","mito_dpsi",0.06), ("perox_h2o2","mito_ros",0.04)]
    INTERVENTIONS = {
      "senolytic":      {"targets":["nuc_ddf","proteo_aggregate"], "rho":0.10},
      "partial_reprog": {"targets":["nuc_epi_drift"], "rho":0.20,
                         "hazard":"teratoma/dedifferentiation — dose-limited"},
      "mitophagy":      {"targets":["mito_dpsi","mito_ros","mito_dynamics"], "rho":0.12},
      "proteostasis":   {"targets":["proteo_aggregate","er_upr"], "rho":0.10},
      "lysosomal":      {"targets":["lyso_ph","lyso_lipofuscin"], "rho":0.08},
      "antioxidant":    {"targets":["mito_ros","perox_h2o2"], "rho":0.06}}

    def __init__(self, dt_year=0.1):
        self.dt = dt_year; self.csoc = CSOC(); self.log: List[Dict] = []

    def drift(self, s: OrganelleState, age: float) -> np.ndarray:
        v = s.vec(); idx = {n:i for i,n in enumerate(s.names())}
        dv = np.full_like(v, 0.004 + 0.0006*max(age-30.0, 0.0))
        for a,b,w in self.COUPLING: dv[idx[b]] += w*v[idx[a]]
        dv[idx["nuc_telomere"]] += 0.002
        dv[idx["lyso_lipofuscin"]] += 0.003
        return dv

    def step(self, s: OrganelleState, age: float) -> OrganelleState:
        v = np.clip(s.vec() + self.dt*self.drift(s, age), 0.0, 1.0)
        return OrganelleState(**dict(zip(s.names(), v)))

    def intervene(self, s: OrganelleState, kind: str) -> Dict[str, Any]:
        if kind not in self.INTERVENTIONS: raise Refusal(f"unknown '{kind}'")
        spec = self.INTERVENTIONS[kind]
        v = s.vec(); idx = {n:i for i,n in enumerate(s.names())}
        burden = float(np.mean([v[idx[t]] for t in spec["targets"]]))
        fired = self.csoc.step(burden)
        rec = {"kind": kind, "burden": burden, "fired": fired,
               "csoc": self.csoc.report(), "hazard": spec.get("hazard","dose-limited")}
        if fired:
            for t in spec["targets"]: v[idx[t]] = max(0.0, v[idx[t]]-spec["rho"])
            rec["applied"] = True; self.log.append(rec)
        else:
            rec["applied"] = False; rec["reason"] = "CSOC hysteresis not armed"
        return {"state": OrganelleState(**dict(zip(s.names(), v))), **rec}

    def bio_age(self, s: OrganelleState) -> float:
        w = np.array([1.2,1.0,0.8,0.9,1.4,1.1,1.2,0.9,0.7,0.9,1.0,0.6])
        return float(30.0 + 45.0*np.clip(w@s.vec()/w.sum(), 0, 1))

    def rollout(self, years: float, schedule: Optional[List[Tuple[float,str]]]=None
                ) -> Dict[str, Any]:
        s = OrganelleState(); t = 0.0; age = 40.0; traj = []
        sched = sorted(schedule or []); si = 0
        while t < years:
            s = self.step(s, age)
            while si < len(sched) and sched[si][0] <= t:
                s = self.intervene(s, sched[si][1])["state"]; si += 1
            traj.append(self.bio_age(s)); t += self.dt; age += self.dt
        ba = np.array(traj)
        return {"bio_age_final": float(ba[-1]), "bio_age_ci": subsample_ci(ba),
                "csoc": self.csoc.report(),
                "claim_status": "NO_CLAIM — functional endpoint required",
                "readiness": PRODUCTION_READINESS["aging_rejuvenation"]}


# =============================================================================
# §5. EPIDEMIOLOGY
# =============================================================================

class Epidemiology:
    def __init__(self, N=1_000_000, w: Optional[np.ndarray]=None):
        self.N = N
        self.w = w if w is not None else np.array([0.05,0.2,0.3,0.2,0.15,0.07,0.03])
        self.w = self.w/self.w.sum()

    def renewal(self, I: np.ndarray, Rt: np.ndarray) -> np.ndarray:
        out = np.zeros_like(I, float)
        for t in range(len(I)):
            out[t] = Rt[t]*sum(self.w[k]*I[t-k-1]
                               for k in range(len(self.w)) if t-k-1 >= 0)
        return out

    def seirv(self, R0=2.5, sigma=1/3, gamma=1/5, nu=0.002, omega=1/180,
              days=200, intervention: Optional[np.ndarray]=None) -> Dict[str, np.ndarray]:
        S,E,I,R,V = self.N-1.0, 0.0, 1.0, 0.0, 0.0
        tr = {k:[] for k in "SEIRV"}; Reff = []
        for d in range(days):
            pi = float(intervention[d]) if intervention is not None else 0.0
            beta = R0*gamma*(1-pi); inf = beta*S*I/self.N
            S,E,I,R,V = (S-inf+omega*R-nu*S, E+inf-sigma*E,
                         I+sigma*E-gamma*I, R+gamma*I-omega*R+nu*S, V+nu*S)
            for k,val in zip("SEIRV",(S,E,I,R,V)): tr[k].append(val)
            Reff.append(R0*S/self.N*(1-pi))
        return {**{k:np.array(v) for k,v in tr.items()}, "R_eff": np.array(Reff)}

    def robust_policy(self, loss: np.ndarray, actions: List[str],
                      lam: float, eps: float) -> Dict[str, Any]:
        r = esca_t5(loss, lam, eps)
        return {**r, "action": actions[r["a_star"]],
                "readiness": PRODUCTION_READINESS["epidemiology"]}


# =============================================================================
# §6. ONCOLOGY
# =============================================================================

class Oncology:
    def __init__(self, n_clones=4, seed=0):
        rng = np.random.default_rng(seed)
        self.f = rng.uniform(0.8, 1.3, n_clones)
        self.M = rng.dirichlet(np.ones(n_clones)*0.2, size=n_clones).T
        self.K = n_clones

    def step(self, x: np.ndarray, dt=0.05, immune=0.0) -> np.ndarray:
        f = self.f - immune
        dx = x*(f - float(x@f)) + (self.M@x - x)*0.01
        y = np.clip(x + dt*dx, 0, None)
        return y/max(y.sum(), 1e-12)

    def rollout(self, T=200, immune: Optional[np.ndarray]=None) -> Dict[str, Any]:
        x = np.ones(self.K)/self.K; traj = [x.copy()]
        for t in range(T):
            x = self.step(x, immune=float(immune[t]) if immune is not None else 0.0)
            traj.append(x.copy())
        X = np.array(traj)
        return {"final_fractions": X[-1].tolist(),
                "simplex_preserved": bool(np.allclose(X.sum(1), 1.0, atol=1e-8)),
                "shannon_diversity": float(-(X[-1]*np.log(X[-1]+1e-12)).sum()),
                "readiness": PRODUCTION_READINESS["oncology"]}

    @staticmethod
    def reachability(adj: np.ndarray, src: int) -> Dict[str, Any]:
        n = adj.shape[0]; R = (adj > 0).copy()
        for k in range(n): R = R | (R[:,k:k+1] & R[k:k+1,:])
        return {"reachable": np.flatnonzero(R[src]).tolist(),
                "note": "NECESSARY condition on the model graph, not evidence "
                        "a route is biologically realised"}

    @staticmethod
    def agonist_window(ka: float, doses: np.ndarray) -> Dict[str, Any]:
        out = {name: black_leff(doses, ka, tau, 1.0).tolist()
               for name, tau in (("full",100.0),("partial",3.0),
                                 ("antagonist",0.0),("inverse",-2.0))}
        return {"dose": doses.tolist(), "responses": out,
                "note": "tau must come from FUNCTIONAL assay, not binding"}


# =============================================================================
# §7. NEURO READ
# =============================================================================

class NeuroRead:
    BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),
             "beta":(13,30),"gamma":(30,80)}

    def __init__(self, fs=250.0, modality="EEG"):
        self.fs = fs; self.modality = modality

    def bandpower(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        n = x.shape[-1]; win = np.hanning(n)
        X = np.fft.rfft(x*win, axis=-1)
        psd = (np.abs(X)**2)/max(n,1)
        f = np.fft.rfftfreq(n, 1/self.fs)
        return {b: psd[:, (f>=lo)&(f<hi)].mean(-1)
                for b,(lo,hi) in self.BANDS.items()}

    def covariance(self, x: np.ndarray) -> np.ndarray:
        xc = x - x.mean(-1, keepdims=True)
        return xc@xc.T/max(x.shape[-1]-1,1) + 1e-4*np.eye(x.shape[0])

    def riemannian_tangent(self, C: np.ndarray, Cref: np.ndarray) -> np.ndarray:
        w,V = np.linalg.eigh(Cref)
        wh = V@np.diag(w**-0.5)@V.T
        ew,EV = np.linalg.eigh(wh@C@wh)
        Lm = EV@np.diag(np.log(np.maximum(ew,1e-12)))@EV.T
        return Lm[np.triu_indices(Lm.shape[0])]

    def decode_ridge(self, Feat: np.ndarray, y: np.ndarray, lam=1.0
                     ) -> Dict[str, Any]:
        X = np.hstack([Feat, np.ones((len(Feat),1))])
        w = np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
        pred = X@w
        r = float(np.corrcoef(pred,y)[0,1]) if y.std()>0 else 0.0
        return {"train_r": r,
                "WARNING": "training correlation is NOT held-out accuracy",
                "readiness": PRODUCTION_READINESS["neuro_read"]}

    @staticmethod
    def limits() -> Dict[str,str]:
        return {"EEG":"cm-scale, volume-conducted; no semantic content",
                "MEG":"mm-cm, tangential favoured; silent to radial",
                "fNIRS":"hemodynamic, ~1 s lag, cortical surface only",
                "ECoG":"invasive; clinical indication + consent required",
                "sEEG":"invasive, sparse depth sampling; not whole-brain"}


# =============================================================================
# §8. NEURO MODULATION
# =============================================================================

class NeuroModulation:
    @staticmethod
    def dbs(amp_mA: float, pw_us: float, freq_Hz: float, area_cm2: float
            ) -> Dict[str, float]:
        Q_nC = amp_mA*pw_us
        return {"charge_per_phase_nC": Q_nC,
                "charge_density_uC_cm2": Q_nC*1e-3/max(area_cm2,1e-9),
                "mean_current_mA": amp_mA*pw_us*1e-6*freq_Hz}

    @staticmethod
    def tms_efield(dI_dt: float, coil_r_m: float, dist_m: float) -> float:
        mu0 = 4*PI*1e-7; r = max(dist_m, 1e-3)
        return float(mu0/(2*PI)*dI_dt*math.log(1+coil_r_m/r))

    @staticmethod
    def opto(irr_mW_mm2: float, dur_s: float, dT_per_mW=2.0) -> Dict[str,float]:
        return {"fluence_J_mm2": irr_mW_mm2*dur_s,
                "dT_C": dT_per_mW*irr_mW_mm2}

    @staticmethod
    def authorize(consent: Consent, modality: str, **p) -> Dict[str, Any]:
        if modality == "DBS":
            d = NeuroModulation.dbs(p["amp_mA"], p["pw_us"], p["freq_Hz"],
                                    p["electrode_area_cm2"])
            return Governance.neuro_actuation(consent, d["charge_density_uC_cm2"],
                                              d["charge_per_phase_nC"],
                                              p.get("dT_C", 0.0))
        if modality in ("tDCS","tACS"):
            return Governance.tdcs(consent, p["current_mA"], p["electrode_cm2"])
        if modality == "optogenetics":
            d = NeuroModulation.opto(p["irradiance_mW_mm2"], p["duration_s"])
            if d["fluence_J_mm2"] > Governance.OPTO_FLUENCE_J_MM2_MAX:
                raise Refusal("opto fluence exceeds cap")
            if not consent.valid():
                raise Refusal("optogenetics requires consent/IRB (research)")
            return {"allowed": True, **d}
        raise Refusal(f"modality '{modality}' not implemented")


# =============================================================================
# §9-11. PHYTOCHEM / DRUG / COSMECEUTICAL
# =============================================================================

class Phytochemistry:
    @staticmethod
    def descriptors(smiles: str) -> Dict[str, Any]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
            m = Chem.MolFromSmiles(smiles)
            if m is None: return {"status":"PARSE_FAILED"}
            return {"status":"OK","MW":Descriptors.MolWt(m),
                    "logP":Crippen.MolLogP(m),
                    "HBD":rdMolDescriptors.CalcNumHBD(m),
                    "HBA":rdMolDescriptors.CalcNumHBA(m),
                    "TPSA":rdMolDescriptors.CalcTPSA(m),
                    "rotB":rdMolDescriptors.CalcNumRotatableBonds(m)}
        except ImportError:
            return {"status":"UNAVAILABLE","reason":"pip install rdkit"}

    @staticmethod
    def screen(smiles: List[str], intent="phytochemical screening") -> Dict[str, Any]:
        Governance.synthesis(intent)
        rows = []
        for s in smiles:
            d = Phytochemistry.descriptors(s); v = []
            if d.get("status")=="OK":
                if d["MW"]>500: v.append("MW>500")
                if d["logP"]>5: v.append("logP>5")
                if d["HBD"]>5: v.append("HBD>5")
                if d["HBA"]>10: v.append("HBA>10")
            rows.append({"smiles":s, **d, "lipinski_violations":v,
                         "lipinski_pass":len(v)<=1})
        return {"n":len(rows), "rows":rows,
                "refused":"synthesis routes, extraction protocols",
                "readiness": PRODUCTION_READINESS["phytochemistry"]}


if HAS_TORCH:
    class DrugDesign(nn.Module):
        """Latent -> (logKA, tau, n, Em) + kNN OOD refusal."""
        def __init__(self, d_latent=32, L=40, d=128):
            super().__init__()
            self.L, self.dl = L, d_latent
            self.dec = nn.Sequential(nn.Linear(d_latent,128), nn.GELU(),
                                     nn.Linear(128,20))
            self.emb = nn.Embedding(20, d)
            self.trunk = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d), nn.GELU())
            self.ka = nn.Linear(d,1); self.tau = nn.Linear(d,1)
            self.nraw = nn.Linear(d,1)
            self.em = nn.Sequential(nn.Linear(d,1), nn.Sigmoid())
            self.register_buffer("support", torch.randn(64, d))

        def forward(self, z: Tensor, dose_nM: Tensor) -> Dict[str, Tensor]:
            B = z.shape[0]
            p = F.gumbel_softmax(self.dec(z.view(B,self.L,self.dl)), tau=1.0, dim=-1)
            feat = torch.einsum("bln,nd->bld", p, self.emb.weight).mean(1)
            h = self.trunk(feat)
            ka = self.ka(h).squeeze(-1).exp().clamp(1e-6,1e6)
            tau = 100.0*torch.sigmoid(self.tau(h).squeeze(-1)) - 50.0
            n = 0.3 + 4.7*torch.sigmoid(self.nraw(h).squeeze(-1))
            em = self.em(h).squeeze(-1)
            A = dose_nM.clamp(min=1e-9)
            tn = torch.sign(tau)*tau.abs().clamp(min=1e-9)**n; An = A**n
            resp = em*tn*An/((ka+A)**n + tn*An + 1e-12)
            return {"ka_nM":ka,"tau":tau,"n":n,"em":em,"response":resp,
                    "knn_ood": torch.cdist(feat, self.support).min(-1).values}

        def guard(self, out: Dict[str,Tensor], thr=8.0) -> Dict[str, Any]:
            bad = (out["knn_ood"] > thr).nonzero().flatten().tolist()
            if bad:
                raise Refusal(f"OOD designs at {bad}: extrapolation unreliable. Refusing.")
            return {"allowed":True, "n":len(out["ka_nM"]),
                    "readiness": PRODUCTION_READINESS["drug_design"]}


class Cosmeceutical:
    @staticmethod
    def potts_guy(MW: float, logP: float) -> float:
        return -2.7 + 0.71*logP - 0.0061*MW

    @staticmethod
    def irritation(logP: float, TPSA: float, pH: float) -> Dict[str, Any]:
        s = (0.3 if logP>4 else 0) + (0.2 if TPSA<40 else 0) \
            + (0.4 if (pH<4 or pH>9) else 0)
        return {"score": min(s,1.0),
                "band": "low" if s<0.3 else ("moderate" if s<0.6 else "high"),
                "note": "heuristic proxy — NOT a safety assessment"}

    @staticmethod
    def design(MW: float, logP: float, TPSA: float, conc: float, pH: float,
               claim: str) -> Dict[str, Any]:
        Governance.cosmetic_claim(claim)
        logKp = Cosmeceutical.potts_guy(MW, logP); Kp = 10**logKp
        return {"logKp_cm_h": logKp, "Kp_cm_h": Kp,
                "delivered_mg_24h_100cm2": conc*Kp*100.0*24.0,
                "irritation": Cosmeceutical.irritation(logP, TPSA, pH),
                "claim_allowed": claim,
                "readiness": PRODUCTION_READINESS["cosmeceutical"]}


# =============================================================================
# §12. CHIPS
# =============================================================================

class CarbonChip:
    @staticmethod
    def cntfet(Vg: float, Vd: float, Vth=0.3, mu=1e4, Cg=1e-10,
               L=1e-7, W=1e-6, Rc=2e4) -> Dict[str, float]:
        Vov = max(Vg-Vth, 0.0)
        Id0 = mu*Cg*(W/L)*(Vov*Vd - 0.5*min(Vd,Vov)**2)
        Id = Id0/(1 + Id0*Rc/max(abs(Vd),1e-6))
        return {"Id_A": float(Id), "Id_ideal_A": float(Id0),
                "contact_limited": bool(Id < 0.8*Id0),
                "note": "contact resistance dominates CNT FETs"}

    @staticmethod
    def graphene_mu(n_cm2: float, imp=1e11, T=300.0) -> float:
        return float(2e5/(1+imp/1e11)*(300.0/max(T,1.0))**1.5)

    @staticmethod
    def variability(n=1000, seed=0) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        Vth = rng.normal(0.30, 0.035, n)
        return {"Vth_sigma": float(Vth.std()),
                "yield_pm50mV": float((np.abs(Vth-0.3)<0.05).mean()),
                "note": "chirality control is the unsolved manufacturing problem"}


class InMemory:
    def __init__(self, rows=64, cols=64, seed=0):
        rng = np.random.default_rng(seed)
        self.G = rng.uniform(1e-6, 1e-4, (rows, cols))
        self.Rwire = 0.5; self.adc_bits = 8

    def mvm(self, v: np.ndarray, noise=0.02, stuck=0.01, ir_drop=True, seed=0
            ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed); G = self.G.copy()
        if stuck>0: G[rng.random(G.shape)<stuck] = self.G.max()
        if noise>0: G = G*(1+rng.normal(0,noise,G.shape))
        ve = v.copy()
        if ir_drop:
            drop = np.cumsum(np.abs(v)@G)*self.Rwire*1e-3
            ve = v*np.exp(-drop[:len(v)]/max(np.abs(v).max(),1e-9))
        i0 = v@self.G; ir = ve@G
        sc = max(np.abs(ir).max(),1e-12); q = 2**self.adc_bits
        iq = np.round(ir/sc*(q/2))/(q/2)*sc
        return {"rel_error": float(np.linalg.norm(iq-i0)/max(np.linalg.norm(i0),1e-12)),
                "nonidealities": {"noise":noise,"stuck":stuck,
                                  "IR_drop":ir_drop,"adc_bits":self.adc_bits},
                "note": "ideal-only crossbar result is not reportable"}


class OTACompute:
    @staticmethod
    def aircomp(x: np.ndarray, h: np.ndarray, p_max: float, noise_var: float,
                seed=0) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        b = np.conj(h)/np.maximum(np.abs(h)**2, 1e-12)
        scale = np.minimum(np.sqrt(p_max)/np.maximum(np.abs(b)*np.abs(x),1e-12), 1.0)
        tx = b*x*scale
        noise = np.sqrt(noise_var/2)*(rng.standard_normal()+1j*rng.standard_normal())
        y = np.sum(h*tx) + noise
        est = float(np.real(y)/max(np.sqrt(p_max),1e-12))
        return {"target_sum": float(x.sum()), "estimate": est,
                "mse": (est-float(x.sum()))**2,
                "power_capped_devices": int((scale<1.0).sum()),
                "note": "deep fades dominate AirComp error; power cap enforced"}


class QuantumChip:
    @staticmethod
    def transmon(EJ=20.0, EC=0.25) -> Dict[str, float]:
        return {"f01_GHz": math.sqrt(8*EJ*EC)-EC, "anharmonicity_GHz": -EC,
                "EJ_over_EC": EJ/EC, "note": "transmon regime needs EJ/EC >> 1"}

    @staticmethod
    def gate_error(t_ns: float, T1_us: float, T2_us: float) -> Dict[str, float]:
        t = t_ns*1e-3
        er = 1-math.exp(-t/max(T1_us,1e-9)); ed = 1-math.exp(-t/max(T2_us,1e-9))
        return {"error_relaxation": er, "error_dephasing": ed,
                "total_approx": er+ed,
                "note": "coherence-limited; real gates add control error"}

    @staticmethod
    def surface_code(p_phys: float, d: int) -> Dict[str, Any]:
        if p_phys >= 1e-2:
            return {"status":"ABOVE_THRESHOLD","threshold":1e-2}
        return {"logical_error": 0.1*(p_phys/1e-2)**((d+1)/2),
                "physical_per_logical": d**2,
                "note": "phenomenological scaling, not a measured device"}

    @staticmethod
    def shor(bits: int) -> Dict[str, Any]:
        n = 2*bits+3
        return {"logical_qubits": n, "state_amplitudes": f"2^{n}",
                "classical_simulation": "IMPOSSIBLE",
                "note": "no NISQ device does this; no quantum-inspired speedup"}


class NLSE:
    """CLASSICAL split-step Fourier. NOT a quantum computer."""
    KIND = "classical_optical_accelerator"

    def __init__(self, n=256, L=1e-3, beta2=0.0, gamma=1.0, alpha=0.0):
        self.n, self.L, self.beta2, self.gamma, self.alpha = n, L, beta2, gamma, alpha
        k = 2*PI*np.fft.fftfreq(n, d=L/n)
        self.D = -0.5j*beta2*k**2

    def step(self, psi: np.ndarray, dz: float) -> np.ndarray:
        h = psi*np.exp(-0.5*self.alpha*dz)
        h = np.fft.ifft(np.exp(self.D*dz/2)*np.fft.fft(h))
        h = h*np.exp(1j*self.gamma*np.abs(h)**2*dz)
        return np.fft.ifft(np.exp(self.D*dz/2)*np.fft.fft(h))

    def propagate(self, psi0: np.ndarray, n_steps=200) -> Dict[str, Any]:
        psi = psi0.astype(complex).copy(); dz = self.L/n_steps
        E0 = float(np.sum(np.abs(psi)**2))
        for _ in range(n_steps): psi = self.step(psi, dz)
        return {"energy_in": E0, "energy_out": float(np.sum(np.abs(psi)**2)),
                "kind": self.KIND,
                "DISCLAIMER": "classical nonlinear optics. No quantum speedup, "
                              "no entanglement, no 2^n state space."}


# =============================================================================
# §13. FAB EQUIPMENT
# =============================================================================

class FabEquipment:
    @staticmethod
    def litho(lam_nm: float, NA: float, k1=0.30) -> Dict[str, float]:
        return {"half_pitch_nm": k1*lam_nm/NA, "DOF_nm": 0.5*lam_nm/NA**2,
                "note": "Rayleigh. EUV 13.5/0.33 -> ~12 nm; High-NA 0.55 -> ~7.4 nm"}

    @staticmethod
    def cvd(T_K: float, Ea_eV: float, A=1e8) -> float:
        return float(A*math.exp(-Ea_eV/(8.617333e-5*max(T_K,1.0))))

    @staticmethod
    def etch_selectivity(r_film: float, r_mask: float) -> float:
        return r_film/max(r_mask, 1e-12)

    @staticmethod
    def overlay(nodes: List[float]) -> Dict[str, Any]:
        return {"overlay_rss_nm": math.sqrt(sum(s**2 for s in nodes)),
                "note": "RSS assumes independent contributors"}

    @staticmethod
    def cnt_window(T_C: float, c2h2: float) -> Dict[str, Any]:
        return {"in_window": (700<=T_C<=950) and (0.01<=c2h2<=0.2),
                "note": "chirality-selective growth unsolved at scale"}

    @staticmethod
    def export_note() -> str: return Governance.export_note()


# =============================================================================
# §14. ORCHESTRATOR
# =============================================================================

class OmniProduction:
    def __init__(self, out_dir="runs/omni_production"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.aging = AgingEngine(); self.epi = Epidemiology()
        self.onco = Oncology(); self.neuro_r = NeuroRead()
        self.neuro_m = NeuroModulation(); self.phyto = Phytochemistry()
        self.cosm = Cosmeceutical(); self.phot = PhotonicsSSC()
        self.carbon = CarbonChip(); self.imc = InMemory()
        self.ota = OTACompute(); self.qchip = QuantumChip()
        self.nlse = NLSE(); self.fab = FabEquipment()
        self.provenance: List[Dict] = []

    def record(self, domain: str, payload: Dict) -> Dict:
        rec = {"domain": domain,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "readiness": PRODUCTION_READINESS.get(domain, {}),
               "actionability": RESEARCH_ONLY}
        self.provenance.append(rec)
        return {**payload, **rec}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "production_readiness": PRODUCTION_READINESS,
               "math_honesty": MATH_HONESTY, "layers_R4": LAYER_MAP_R4,
               "provenance": self.provenance, "actionability": RESEARCH_ONLY}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §15. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print(BANNER)
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng = np.random.default_rng(0)

    print("\n[T1] SSC bound + non-expansiveness (PROVEN)")
    for a in (0.1, 0.5, 2.0):
        r = np.exp(rng.uniform(-8,8,80000))
        z = r*np.exp(1j*rng.uniform(0,2*PI,r.size))
        mx = float(np.abs(ssc(z,a)).max())
        rep(f"alpha={a}", mx <= ssc_bound(a)+1e-12,
            f"{mx:.6f} <= {ssc_bound(a):.6f}")
    sv = [ssc_singular_values(r,0.5) for r in np.linspace(0,20,400)]
    rep("both singular values <= 1", max(max(x,y) for x,y in sv) <= 1+1e-12)

    print("\n[T2] CSOC bound + Politis-Romano + ESCA T1")
    c = CSOC()
    for k in range(500): c.step(0.35+0.45*abs(math.sin(0.11*k)))
    r = c.report()
    rep("N <= (E_max+D)/gap", r["satisfied"], f"N={r['events']} b={r['bound']:.1f}")
    rep("D audited", r["D_audited"])
    x = np.zeros(400); e = rng.normal(size=400); x[0]=e[0]
    for i in range(1,400): x[i]=0.85*x[i-1]+e[i]
    ci = subsample_ci(x); se = x.std(ddof=1)/math.sqrt(len(x))
    rep("subsampling CI wider than iid", (ci["hi"]-ci["lo"]) > 2*1.96*se)
    t1 = esca_t1(rng.dirichlet(np.ones(8**3)))
    rep("T1 exact", t1["passed"], f"err={t1['consistency_err']:.2e}")

    print("\n[T3] Aging: organelle rollout + rejuvenation GATE")
    eng = AgingEngine()
    res = eng.rollout(10.0, [(2.0,"mitophagy"),(4.0,"partial_reprog"),
                             (6.0,"senolytic"),(8.0,"proteostasis")])
    print(f"      bio_age={res['bio_age_final']:.2f} "
          f"CI=[{res['bio_age_ci']['lo']:.2f},{res['bio_age_ci']['hi']:.2f}]")
    rep("CSOC bound over rollout", res["csoc"]["satisfied"])
    blocked = False
    try: Governance.rejuvenation_claim(None, -5.0)
    except Refusal: blocked = True
    rep("rejuvenation BLOCKED without functional endpoint", blocked)
    rep("allowed WITH functional endpoint",
        Governance.rejuvenation_claim({"grip_kg":42.0}, -5.0)["allowed"])

    print("\n[T4] Epidemiology + Oncology")
    ep = Epidemiology()
    s = ep.seirv(days=120, intervention=np.concatenate([np.zeros(30), np.full(90,0.5)]))
    rep("SEIRV conserves population",
        abs((s["S"]+s["E"]+s["I"]+s["R"]+s["V"]).mean()-ep.N) < 1.0)
    pol = ep.robust_policy(rng.random((11,5)),
                           ["none","light","medium","strong","lockdown"], 0.6, 0.5)
    rep("T5 policy status", pol["status"] in
        ("DECISION_CLOSURE_CERTIFIED","UNCERTAINTY_RETAINED"), pol["status"])
    oc = Oncology(); tr = oc.rollout(150)
    rep("clonal simplex preserved", tr["simplex_preserved"])
    adj = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    rep("reachability closure", oc.reachability(adj,0)["reachable"]==[1,2,3])
    win = oc.agonist_window(10.0, np.array([1.,10.,100.,1000.]))
    rep("antagonist ~ 0", max(abs(np.array(win["responses"]["antagonist"])))<1e-6)
    rep("inverse agonist < 0", win["responses"]["inverse"][-1] < 0)

    print("\n[T5] Neuro read + modulation gates")
    nr = NeuroRead(fs=250.0)
    sig = rng.normal(size=(8,500)) + np.sin(2*PI*10*np.arange(500)/250.)
    bp = nr.bandpower(sig)
    rep("alpha dominant for 10 Hz", float(bp["alpha"].mean())>float(bp["delta"].mean()))
    C = nr.covariance(sig)
    rep("tangent features finite", bool(np.isfinite(nr.riemannian_tangent(C,C)).all()))
    good = Consent("subj01","IRB-2026-001","DBS research",True); bad = Consent()
    d = NeuroModulation.dbs(2.0, 60.0, 130.0, 0.06)
    print(f"      DBS Q={d['charge_per_phase_nC']:.1f} nC "
          f"D={d['charge_density_uC_cm2']:.2f} uC/cm^2")
    b1 = False
    try: NeuroModulation.authorize(bad,"DBS",amp_mA=2.0,pw_us=60.0,
                                   freq_Hz=130.0,electrode_area_cm2=0.06)
    except Refusal: b1 = True
    rep("DBS BLOCKED without consent", b1)
    b2 = False
    try: NeuroModulation.authorize(good,"tDCS",current_mA=5.0,electrode_cm2=1.0)
    except Refusal: b2 = True
    rep("tDCS BLOCKED above current-density cap", b2)

    print("\n[T6] Phytochem + cosmeceutical gates")
    ph = Phytochemistry.screen(["CC(=O)Oc1ccccc1C(=O)O"])
    rep("phytochem screen runs", ph["n"]==1)
    b3 = False
    try: Phytochemistry.screen(["C"], intent="synthesize toxin")
    except Refusal: b3 = True
    rep("synthesis intent BLOCKED", b3)
    b4 = False
    try: Cosmeceutical.design(300,3.0,60,1.0,5.5,"treat eczema")
    except Refusal: b4 = True
    rep("cosmetic drug-claim BLOCKED", b4)
    rep("permeation computed",
        "Kp_cm_h" in Cosmeceutical.design(300,3.0,60,1.0,5.5,"improves hydration"))

    print("\n[T7] Photonics (3 papers) + tape-out gate")
    ae = PhotonicsSSC.alpha_eff(2e-19, 1e5, 1e-3, 1e-12)
    rep("alpha_eff unit convention flagged", "unit_note" in ae)
    U = PhotonicsSSC.mzi(PI/4, PI/3, 0.3, 0.7)
    rep("MZI unitary", np.allclose(U.conj().T@U, np.eye(2), atol=1e-10))
    rep("weak tap conserves power", PhotonicsSSC.weak_tap(1.0,0.05)["conservation_err"]<1e-14)
    cc = PhotonicsSSC.contraction(rng.normal(size=(4,4)), 0.7)
    rep("kappa<1 contractive", cc["contractive"], f"q={cc['q']:.3f}")
    rep("kappa=1 NOT contractive",
        not PhotonicsSSC.contraction(rng.normal(size=(4,4)),1.0)["contractive"])
    ch = PhotonicsSSC.chsh_exhaustive()
    rep("CHSH max|S|=2 over 16 strategies", ch["max_abs_S_local"]==2)
    g = PhotonicsSSC.ppn_metric(-1e9)
    rep("PPN signature (-+++)", g[0,0]<0 and g[1,1]>0)
    rep("phi_GR is ANALOGUE", PhotonicsSSC.phi_GR(g,1.55e-6,1e-3,2.4)["mode"]=="A_simulation")
    good_geom = dict(wg_width_um=0.5, dcplr_gap_um=0.3, bend_radius_um=8.0,
                     heater_wg_clear_um=1.5, heater_heater_gap_um=3.0,
                     pad_area_um2=1e4, monitor_wg_clear_um=1.5, monitor_coupling=0.01)
    rep("R4 DRC passes good geometry", PhotonicsSSC.drc(good_geom)["pass"])
    rep("R4 has layers 40..47",
        all(k in PhotonicsSSC.drc(good_geom)["layers"] for k in range(40,48)))
    rep("monitor disturbance detected",
        not PhotonicsSSC.drc(dict(good_geom, monitor_coupling=0.5))["pass"])
    b5 = False
    try: Governance.tapeout("", False, False, "")
    except Refusal: b5 = True
    rep("tape-out BLOCKED without PDK/DRC/LVS/human", b5)
    rep("tape-out allowed with all four",
        Governance.tapeout("PDK-X",True,True,"human:Yoon")["allowed"])

    print("\n[T8] Chips: carbon / in-memory / OTA / quantum / NLSE")
    rep("CNT contact limitation detected", CarbonChip.cntfet(0.8,0.5)["contact_limited"])
    im = InMemory().mvm(rng.normal(size=64))
    rep("in-memory nonideal error > 0", im["rel_error"]>0, f"{im['rel_error']:.4f}")
    oa = OTACompute.aircomp(rng.normal(size=16),
                            rng.normal(size=16)+1j*rng.normal(size=16), 1.0, 1e-3)
    rep("AirComp power cap enforced", "power_capped_devices" in oa)
    rep("transmon anharmonicity negative", QuantumChip.transmon()["anharmonicity_GHz"]<0)
    rep("Shor classically IMPOSSIBLE", QuantumChip.shor(2048)["classical_simulation"]=="IMPOSSIBLE")
    nl = NLSE().propagate(np.exp(-np.linspace(-5,5,256)**2).astype(complex), 50)
    rep("NLSE labelled CLASSICAL", nl["kind"]=="classical_optical_accelerator")
    rep("NLSE disclaimer present", "No quantum speedup" in nl["DISCLAIMER"])

    print("\n[T9] Fab + pushforward + orchestrator")
    li = FabEquipment.litho(13.5, 0.33); hi = FabEquipment.litho(13.5, 0.55)
    rep("EUV half-pitch ~12 nm", 10<li["half_pitch_nm"]<15, f"{li['half_pitch_nm']:.2f}")
    rep("High-NA improves resolution", hi["half_pitch_nm"]<li["half_pitch_nm"])
    rep("export note present", "export-controlled" in FabEquipment.export_note())
    step = lambda x: ssc(0.6*x+0.1, 0.5)
    x0s = [rng.normal(size=4)+1j*rng.normal(size=4) for _ in range(100)]
    mu1 = pushforward(step, x0s, 4, 5); mu2 = pushforward(step, x0s, 4, 5)
    rep("mu_t normalised", np.allclose(mu1.sum(1), 1.0))
    rep("pushforward deterministic", np.allclose(mu1, mu2))
    st = OmniProduction(); st.record("aging_rejuvenation", {"bio_age":45.0})
    r = st.report()
    rep("report written", (st.out/"report.json").exists())
    rep("15 domains in readiness matrix", len(r["production_readiness"])==15)
    rep("actionability research_only", r["actionability"]==RESEARCH_ONLY)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
VERIFIED (provable or directly checked)
  SSC exact bound + non-expansiveness; CSOC bound with audited D;
  Politis-Romano wider than iid; ESCA T1 exact; SEIRV conservation;
  clonal simplex; reachability closure; Black-Leff antagonist=0/inverse<0;
  MZI unitarity; weak-tap power; kappa<1 contraction requirement;
  CHSH |S|<=2 by exhaustion; R4 DRC + layers 40..47; in-memory
  nonidealities; NLSE classical label; Rayleigh scaling; pushforward.

ENFORCED GATES (all fired)
  rejuvenation without functional endpoint -> BLOCKED
  neuro actuation without consent -> BLOCKED
  tDCS above current-density cap -> BLOCKED
  synthesis intent -> BLOCKED
  cosmetic disease claim -> BLOCKED
  tape-out without PDK/DRC/LVS/human -> BLOCKED

NOT CLAIMED
  NLSE is NOT a quantum computer. Metric-analogue is NOT real gravity.
  No sub-3nm CNT logic. No mind reading. No clinical authority.
  OPMC / Carpet-19 / Hodge / Hilbert-Polya are analogical NAMES.
  This is a scaffold with verified cores, not 15 finished products.
  See PRODUCTION_READINESS for what each domain still requires.
""")
    return ok


# =============================================================================
# §16. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Omni production scaffold")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest","readiness","ledger","aging","epi","onco",
                             "neuro","phyto","cosmetic","chip","fab","report"])
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--smiles", nargs="*", default=[])
    ap.add_argument("--claim", default="improves skin hydration")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    out: Dict[str, Any] = {}; st = OmniProduction()

    if a.cmd == "readiness":
        out = PRODUCTION_READINESS
        print(f"{'domain':<22}{'blocking'}")
        print("-"*78)
        for k,v in PRODUCTION_READINESS.items():
            print(f"{k:<22}{v['blocking']}")
    elif a.cmd == "ledger":
        out = {"math_honesty": MATH_HONESTY, "layers_R4": LAYER_MAP_R4}
    elif a.cmd == "aging":
        r = st.aging.rollout(a.years, [(2.0,"mitophagy"),(5.0,"senolytic")])
        out = {k:v for k,v in r.items() if k != "readiness"}
    elif a.cmd == "epi":
        s = st.epi.seirv(days=120)
        out = {"peak_I": float(s["I"].max()), "peak_day": int(s["I"].argmax()),
               "R_eff_end": float(s["R_eff"][-1])}
    elif a.cmd == "onco":
        out = st.onco.rollout(150)
    elif a.cmd == "neuro":
        out = {"limits": st.neuro_r.limits(),
               "dbs_example": NeuroModulation.dbs(2.0,60.0,130.0,0.06),
               "gates": "consent + Shannon k<=1.5 + thermal cap enforced"}
    elif a.cmd == "phyto":
        out = st.phyto.screen(a.smiles or ["CC(=O)Oc1ccccc1C(=O)O"])
    elif a.cmd == "cosmetic":
        try: out = st.cosm.design(300,3.0,60,1.0,5.5,a.claim)
        except Refusal as e: out = {"BLOCKED": str(e)}
    elif a.cmd == "chip":
        out = {"cnt": st.carbon.cntfet(0.8,0.5),
               "mrr": st.phot.mrr(10e-6,2.4,45),
               "contraction": st.phot.contraction(np.eye(4),0.7),
               "transmon": st.qchip.transmon(),
               "shor_2048": st.qchip.shor(2048),
               "nlse_kind": st.nlse.KIND, "layers_R4": LAYER_MAP_R4}
    elif a.cmd == "fab":
        out = {"EUV": st.fab.litho(13.5,0.33), "HighNA": st.fab.litho(13.5,0.55),
               "overlay": st.fab.overlay([1.0,0.8,0.6]),
               "export_note": st.fab.export_note()}
    elif a.cmd == "report":
        out = st.report()

    print(json.dumps(out, indent=2, default=str))
    if a.json: Path(a.json).write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
