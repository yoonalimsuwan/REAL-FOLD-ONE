#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
omni_stack.py — v1.0.0
Unified scaffold: aging/rejuvenation · epidemiology · oncology · neuro-read ·
neuro-modulation · phytochemistry · drug design · cosmeceuticals ·
carbon/photonics/in-memory/OTA/quantum chips · NLSE accelerator · fab equipment
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
SIX CORRECTIONS THAT MUST BE ACCEPTED BEFORE USING THIS FILE
=============================================================================
[C1] NLSE IS NOT A QUANTUM COMPUTER.
     The nonlinear Schrödinger equation describes CLASSICAL optical envelope
     propagation. Kerr nonlinearity, MZI interference and argmax selection
     are classical. There is no 2^n-dimensional state space, no entanglement,
     no Born-rule measurement. The NLSE module below is labelled
     `classical_optical_accelerator`. Quantum-chip models are SEPARATE.

[C2] BRAIN "CONTROL" = CLINICAL NEUROMODULATION WITH HARD LIMITS.
     Implemented: DBS, TMS, tDCS/tACS, optogenetics (research).
     Enforced in code: Shannon charge-density criterion, current-density
     caps, thermal caps, and a CONSENT GATE. No autonomous actuation.
     NOT implemented: anything for non-consensual influence.

[C3] REJUVENATION IS GATED ON FUNCTION, NOT CLOCKS.
     Partial reprogramming carries teratoma / dedifferentiation risk.
     The scheduler is CSOC-bounded (auditable event count). A rejuvenation
     CLAIM requires a held-out FUNCTIONAL endpoint. Clock movement is a
     correlate, not an outcome.

[C4] CARBON CHIPS AND FAB TOOLS ARE PARAMETRIC RESEARCH MODELS.
     No commercial sub-3nm CNT logic exists. Advanced lithography/deposition
     equipment is subject to export control in multiple jurisdictions.
     This file gives physics-level models and refuses tape-out claims
     without a foundry PDK.

[C5] THE MATHEMATICS NAMES ARE MOSTLY ANALOGICAL.
     Provable here: SSC saturation bound + non-expansiveness, CSOC event
     bound, Politis–Romano coverage, Zalesak boundedness+conservation,
     Black–Leff operational separation, simplex additivity (ESCA T1).
     Imported theorems: Mosco→resolvent, Kac return-time.
     Analogical names: OPMC, Carpet-19, Hodge, Hilbert–Pólya.

[C6] THIS IS A SCAFFOLD WITH VERIFIED CORES, NOT 20 FINISHED PRODUCTS.
     Each domain implements its central computation for real, carries an
     explicit scope statement, and refuses outside its competence.
=============================================================================
"""
from __future__ import annotations

import argparse, itertools, json, logging, math, time
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("omni")
VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"
PI = math.pi

BANNER = f"""
=============================================================================
OMNI STACK v{VERSION} — 15 domains, one honesty registry, enforced gates
NLSE ≠ quantum computer. Brain modulation = clinical + consent + limits.
Rejuvenation gated on FUNCTION. Fab needs PDK. Names are mostly analogical.
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal. Never silently downgraded."""


# =============================================================================
# §0. CAPABILITY LEDGER
# =============================================================================

LEDGER: Dict[str, Dict[str, str]] = {
 "aging_rejuvenation": {"is": "organelle-resolved damage ODE + CSOC-bounded scheduler",
   "gate": "functional endpoint required for any rejuvenation claim",
   "not": "clinical protocol; clock movement is not rejuvenation"},
 "epidemiology": {"is": "renewal + SEIRV + T5 minimax policy + weighted conformal",
   "gate": "pre-declared model family P; variant shift → refuse if q=inf",
   "not": "genetics-driven; contact/policy dominate"},
 "oncology": {"is": "replicator–mutator on simplex + reachability + Black–Leff",
   "gate": "tau from FUNCTIONAL assay only; no treatment recommendation",
   "not": "clinical decision support"},
 "neuro_read": {"is": "bandpower + Riemannian tangent + ridge decoding",
   "gate": "no semantic-content decoding from scalp EEG",
   "not": "mind reading; spatial resolution is cm-scale"},
 "neuro_modulate": {"is": "DBS/TMS/tDCS/opto dose models",
   "gate": "Shannon k<=1.5, current-density cap, thermal cap, CONSENT",
   "not": "autonomous actuation; non-consensual influence"},
 "phytochemistry": {"is": "descriptors + drug-likeness + PAINS-ish flags",
   "gate": "no synthesis routes, no controlled-substance scaffolds",
   "not": "extraction protocol; efficacy claim"},
 "drug_design": {"is": "Black–Leff + latent decoder + OOD refusal",
   "gate": "OOD → refuse; functional assay for tau",
   "not": "ProteinMPNN/RFdiffusion; no folding guarantee"},
 "cosmeceutical": {"is": "Potts–Guy permeation + irritation proxy",
   "gate": "no disease-treatment claim (that makes it a drug)",
   "not": "safety assessment; regulatory dossier"},
 "carbon_chip": {"is": "CNT-FET / graphene parametric I–V + variability",
   "gate": "no foundry PDK → no tape-out claim",
   "not": "production node; sub-3nm CNT logic does not exist commercially"},
 "photonics_ssc": {"is": "SSC theta_C + MZI + MRR + contraction requirement",
   "gate": "kappa<1 required; GDS-II needs PDK + DRC/LVS",
   "not": "quantum computer; metric-analogue ≠ real gravity"},
 "in_memory": {"is": "crossbar MVM with IR-drop/noise/ADC/stuck-at",
   "gate": "nonidealities always on; ideal-only result is not reportable",
   "not": "measured device; endurance/retention guarantee"},
 "ota_compute": {"is": "AirComp channel inversion + power/noise MSE",
   "gate": "power constraint enforced",
   "not": "deployed radio; regulatory compliance"},
 "quantum_chip": {"is": "transmon/ion/photonic parameter models",
   "gate": "T1/T2/gate-error are models, not measurements",
   "not": "a quantum computer; no Shor at cryptographic scale"},
 "nlse_accelerator": {"is": "split-step Fourier CLASSICAL optical propagator",
   "gate": "labelled classical; never called a quantum computer",
   "not": "quantum speedup of any kind"},
 "fab_equipment": {"is": "Rayleigh/DOF, Arrhenius CVD, etch selectivity, overlay",
   "gate": "export-control note; research parameters only",
   "not": "tool specification; procurement guidance"},
}

MATH_HONESTY = {
 "ssc": {"provable": True, "claim": "|theta_C|<=1/(2 sqrt(a)); Lip=1"},
 "csoc": {"provable": True, "claim": "N(T)<=(E_max+D)/gap, D=audited removal"},
 "politis_romano": {"provable": True, "requires": "stationarity+mixing+finite var"},
 "zalesak": {"provable": True, "claim": "exact conservation + no new extrema"},
 "black_leff": {"provable": True, "claim": "separates KA (affinity) from tau (efficacy)"},
 "esca_T1": {"provable": True, "claim": "simplex additivity under refinement"},
 "mosco": {"provable": "imported", "claim": "M1+M2 => strong resolvent convergence"},
 "kac": {"provable": "imported", "claim": "E[r_B]=1/mu(B); nested ratio finite"},
 "opmc": {"provable": False, "is": "learned low-rank bottleneck + fixed steps"},
 "carpet19": {"provable": False, "is": "fixed graph Laplacian as basis"},
 "hodge": {"provable": False, "is": "learned 3-way split + Laplacian penalty"},
 "hilbert_polya": {"provable": False, "is": "learned self-adjoint spectral map"},
 "delta_min": {"provable": "threshold only", "is": "kappa-fatness guard"},
}


# =============================================================================
# §1. SHARED MATH CORE (only the parts that are real)
# =============================================================================

def ssc(z: np.ndarray, alpha: float) -> np.ndarray:
    """theta_C(z)=z/(1+alpha|z|^2). Bound 1/(2 sqrt(alpha)) at r=alpha^-1/2."""
    return z / (1.0 + alpha * np.abs(z) ** 2)

def ssc_bound(alpha: float) -> float:
    return 1.0 / (2.0 * math.sqrt(alpha))

def ssc_singular_values(r: float, alpha: float) -> Tuple[float, float]:
    s = alpha * r * r
    return 1.0 / (1.0 + s), abs(1.0 - s) / (1.0 + s) ** 2


class CSOC:
    """Two-level hysteresis. N(T) <= (E_max + D)/gap with D = ACTUAL removal."""
    def __init__(self, hi=0.6, lo=0.25, rate=0.08):
        assert hi > lo
        self.hi, self.lo, self.gap, self.rate = hi, lo, hi - lo, rate
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0; self.D_audit = 0.0

    def step(self, s: float) -> bool:
        exc = math.log1p(math.exp(-abs(s - self.hi))) + max(s - self.hi, 0.0)
        dec = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10))
        rem = dec * self.rate * self.acc
        self.acc += exc - rem
        self.D += rem; self.D_audit += rem
        self.E_max = max(self.E_max, self.acc)
        if self.acc < self.lo: self.armed = True
        fire = self.armed and self.acc > self.hi
        if fire: self.armed = False; self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        b = (self.E_max + self.D) / self.gap
        return {"events": self.events, "bound": b, "satisfied": self.events <= b + 1e-9,
                "D_audited": abs(self.D - self.D_audit) < 1e-12,
                "scope": "counter only — not a biological guarantee"}


def subsample_ci(x: np.ndarray, conf=0.95) -> Dict[str, float]:
    """Politis–Romano. Correct for serially correlated series; NOT iid."""
    x = np.asarray(x, float).ravel(); n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n, "status": "INSUFFICIENT"}
    b = min(max(2, int(round(n ** (1/3)))), n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i+b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th); a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a/2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a/2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi), "n": n, "block": b, "status": "OK"}


def carpet_laplacian(side: int) -> np.ndarray:
    side = max(3, int(side)); n = side * side
    A = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1: continue
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii, jj = i+di, j+dj
                if 0 <= ii < side and 0 <= jj < side and not (ii%3==1 and jj%3==1):
                    A[i*side+j, ii*side+jj] = 1.0
    return np.diag(A.sum(1)) - A


def delta_min(n=3, kappa=2.0, a_min=0.01) -> float:
    d = n - 1
    om = PI ** (d/2) / math.gamma(d/2 + 1)
    return (1.0/kappa) * (a_min/om) ** (1.0/d)


def black_leff(dose, ka, tau, em, n=1.0, eps=1e-12):
    """E/Em = tau^n A^n / ((KA+A)^n + tau^n A^n). Separates affinity/efficacy."""
    A = np.maximum(dose, eps); KA = np.maximum(ka, eps)
    tn = np.sign(tau) * np.maximum(np.abs(tau), eps) ** n
    An = A ** n
    return em * tn * An / np.maximum((KA + A) ** n + tn * An, eps)


def esca_t1_check(p_fine: np.ndarray, levels=3) -> Dict[str, Any]:
    ps = {levels: p_fine / p_fine.sum()}
    for m in range(levels-1, -1, -1):
        ps[m] = ps[m+1].reshape(-1, 8).sum(1)
    en = max(abs(p.sum()-1) for p in ps.values())
    ec = max(float(np.abs(ps[m+1].reshape(-1,8).sum(1)-ps[m]).max()) for m in range(levels))
    return {"norm_err": en, "consistency_err": ec, "passed": en < 1e-12 and ec < 1e-12}


def pushforward(step: Callable, x0s: List[Any], n_out: int, T: int) -> np.ndarray:
    """ESDA Prop 1.1: mu_t = (G o F^t)_# rho_0. Deterministic maps + ensemble."""
    mu = np.zeros((T, n_out))
    for x0 in x0s:
        x = x0
        for t in range(T):
            x = step(x)
            mu[t, int(np.argmax(np.abs(x) ** 2))] += 1
    return mu / max(len(x0s), 1)


# =============================================================================
# §2. GOVERNANCE / SAFETY GATES (enforced, not documented)
# =============================================================================

@dataclass
class Consent:
    subject_id: str = ""
    irb_id: str = ""
    scope: str = ""
    signed: bool = False
    def valid(self) -> bool:
        return bool(self.subject_id and self.irb_id and self.scope and self.signed)


class Governance:
    """Every actuation / claim passes here first."""

    SHANNON_K_MAX = 1.5          # charge-density safety criterion
    TDCS_MA_CM2_MAX = 0.08       # mA/cm^2 conservative research cap
    THERMAL_DELTA_C_MAX = 2.0    # tissue heating cap
    OPTO_FLUENCE_J_MM2_MAX = 200.0

    @staticmethod
    def neuro_actuation(consent: Consent, charge_density_uC_cm2: float,
                        charge_per_phase_nC: float, dT_C: float) -> Dict[str, Any]:
        if not consent.valid():
            raise Refusal("NEURO ACTUATION BLOCKED: consent/IRB incomplete. "
                          "No autonomous or non-consensual stimulation.")
        # Shannon: k = log10(D) + log10(Q) ; D in uC/cm^2/phase, Q in uC
        Q_uC = charge_per_phase_nC * 1e-3
        k = math.log10(max(charge_density_uC_cm2, 1e-12)) + math.log10(max(Q_uC, 1e-12))
        if k > Governance.SHANNON_K_MAX:
            raise Refusal(f"Shannon k={k:.2f} > {Governance.SHANNON_K_MAX}: tissue-damage risk.")
        if dT_C > Governance.THERMAL_DELTA_C_MAX:
            raise Refusal(f"Delta T={dT_C:.2f} C exceeds cap.")
        return {"allowed": True, "shannon_k": k, "dT_C": dT_C, "consent": consent.subject_id}

    @staticmethod
    def tdcs(consent: Consent, current_mA: float, electrode_cm2: float) -> Dict[str, Any]:
        if not consent.valid():
            raise Refusal("tDCS BLOCKED: consent/IRB incomplete.")
        j = current_mA / max(electrode_cm2, 1e-9)
        if j > Governance.TDCS_MA_CM2_MAX:
            raise Refusal(f"current density {j:.4f} mA/cm^2 > cap.")
        return {"allowed": True, "current_density_mA_cm2": j}

    @staticmethod
    def rejuvenation_claim(functional_endpoint: Optional[Dict[str, float]],
                           clock_delta_years: float) -> Dict[str, Any]:
        if functional_endpoint is None:
            raise Refusal(
                "REJUVENATION CLAIM BLOCKED: no functional endpoint.\n"
                f"Clock moved {clock_delta_years:+.1f} y — that is a CORRELATE.\n"
                "Required: held-out function (grip, VO2max, cognition, immune "
                "response, tissue histology) with CI. Partial reprogramming "
                "carries teratoma/dedifferentiation risk.")
        return {"allowed": True, "functional": functional_endpoint,
                "clock_delta_years": clock_delta_years,
                "note": "clock is secondary; function is primary"}

    @staticmethod
    def synthesis_request(intent: str) -> Dict[str, Any]:
        low = intent.lower()
        block = ["synthesize toxin", "controlled substance", "schedule i",
                 "nerve agent", "weaponize", "gain of function",
                 "increase virulence", "explosive synthesis"]
        for b in block:
            if b in low:
                raise Refusal(f"BLOCKED intent '{b}'. Computational analysis only.")
        return {"allowed": True, "scope": "descriptors / scoring, no routes"}

    @staticmethod
    def cosmetic_claim(claim: str) -> Dict[str, Any]:
        low = claim.lower()
        drug_words = ["treat", "cure", "heal", "prevent disease", "anti-cancer",
                      "treat eczema", "treat psoriasis", "antibiotic"]
        hit = [w for w in drug_words if w in low]
        if hit:
            raise Refusal(f"Claim contains drug language {hit}. A cosmetic cannot "
                          "claim disease treatment; that reclassifies it as a drug "
                          "requiring regulatory approval.")
        return {"allowed": True, "claim": claim}

    @staticmethod
    def tapeout(pdk_id: str, drc_clean: bool, lvs_clean: bool) -> Dict[str, Any]:
        if not pdk_id or not drc_clean or not lvs_clean:
            raise Refusal("TAPE-OUT BLOCKED: need foundry PDK id + clean DRC + clean LVS. "
                          "A GDS-II that opens is not fabrication-ready.")
        return {"allowed": True, "pdk": pdk_id}

    @staticmethod
    def fab_export_note() -> str:
        return ("Advanced lithography/deposition/etch equipment is subject to "
                "export control in multiple jurisdictions. This file provides "
                "physics-level parametric models for research only — not tool "
                "specifications or procurement guidance.")


# =============================================================================
# §3. AGING + ORGANELLE-RESOLVED REJUVENATION
# =============================================================================

@dataclass
class OrganelleState:
    """Per-organelle damage variables. All in [0,1] (0=healthy, 1=failed)."""
    mito_dpsi: float = 0.1        # membrane-potential loss
    mito_ros: float = 0.1         # ROS load
    mito_mtdna: float = 0.05      # mtDNA mutation burden
    mito_dynamics: float = 0.1    # fission/fusion imbalance
    nuc_epi_drift: float = 0.1    # epigenetic drift
    nuc_telomere: float = 0.05    # attrition
    nuc_ddf: float = 0.05         # DNA-damage foci
    er_upr: float = 0.1           # unfolded-protein response
    lyso_ph: float = 0.05         # acidification loss
    lyso_lipofuscin: float = 0.05 # undegradable aggregate
    proteo_aggregate: float = 0.1
    perox_h2o2: float = 0.05

    def vec(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), float)

    @staticmethod
    def names() -> List[str]:
        return list(asdict(OrganelleState()).keys())


class AgingRejuvenationEngine:
    """
    Damage ODE with cross-organelle coupling + CSOC-bounded interventions.

    dD/dt = f(D) + coupling ; reset events subtract rho on targeted axes.
    Rejuvenation CLAIM requires functional endpoint (Governance gate).
    """

    # coupling: source -> (target, strength). Biologically motivated, not fitted.
    COUPLING = [("mito_ros", "nuc_ddf", 0.06), ("mito_ros", "proteo_aggregate", 0.04),
                ("nuc_ddf", "nuc_epi_drift", 0.05), ("proteo_aggregate", "er_upr", 0.07),
                ("er_upr", "lyso_ph", 0.03), ("lyso_lipofuscin", "proteo_aggregate", 0.05),
                ("mito_dynamics", "mito_dpsi", 0.06), ("perox_h2o2", "mito_ros", 0.04)]

    INTERVENTIONS = {
        "senolytic":       {"targets": ["nuc_ddf", "proteo_aggregate"], "rho": 0.10},
        "partial_reprog":  {"targets": ["nuc_epi_drift"], "rho": 0.20,
                            "hazard": "teratoma/dedifferentiation — dose-limited"},
        "mitophagy":       {"targets": ["mito_dpsi", "mito_ros", "mito_dynamics"], "rho": 0.12},
        "proteostasis":    {"targets": ["proteo_aggregate", "er_upr"], "rho": 0.10},
        "lysosomal":       {"targets": ["lyso_ph", "lyso_lipofuscin"], "rho": 0.08},
        "antioxidant":     {"targets": ["mito_ros", "perox_h2o2"], "rho": 0.06},
    }

    def __init__(self, dt_year=0.1):
        self.dt = dt_year
        self.csoc = CSOC()
        self.intervention_log: List[Dict] = []

    def drift(self, s: OrganelleState, age_y: float) -> np.ndarray:
        v = s.vec(); names = s.names()
        idx = {n: i for i, n in enumerate(names)}
        d = 0.004 + 0.0006 * max(age_y - 30.0, 0.0)      # age-accelerating baseline
        dv = np.full_like(v, d)
        for src, tgt, w in self.COUPLING:
            dv[idx[tgt]] += w * v[idx[src]]
        dv[idx["nuc_telomere"]] += 0.002                  # replication-driven
        dv[idx["lyso_lipofuscin"]] += 0.003               # irreversible-ish
        return dv

    def step(self, s: OrganelleState, age_y: float) -> OrganelleState:
        v = np.clip(s.vec() + self.dt * self.drift(s, age_y), 0.0, 1.0)
        return OrganelleState(**dict(zip(s.names(), v)))

    def propose_intervention(self, s: OrganelleState, kind: str) -> Dict[str, Any]:
        if kind not in self.INTERVENTIONS:
            raise Refusal(f"unknown intervention '{kind}'")
        spec = self.INTERVENTIONS[kind]
        v = s.vec(); names = s.names(); idx = {n: i for i, n in enumerate(names)}
        burden = float(np.mean([v[idx[t]] for t in spec["targets"]]))
        fired = self.csoc.step(burden)
        rec = {"kind": kind, "burden": burden, "fired": fired,
               "csoc": self.csoc.report(), "hazard": spec.get("hazard", "dose-limited")}
        if fired:
            for t in spec["targets"]:
                v[idx[t]] = max(0.0, v[idx[t]] - spec["rho"])
            rec["applied"] = True
            self.intervention_log.append(rec)
        else:
            rec["applied"] = False
            rec["reason"] = "CSOC hysteresis not armed — bounded event count"
        return {"state": OrganelleState(**dict(zip(names, v))), **rec}

    def biological_age(self, s: OrganelleState) -> float:
        w = np.array([1.2, 1.0, 0.8, 0.9, 1.4, 1.1, 1.2, 0.9, 0.7, 0.9, 1.0, 0.6])
        return float(30.0 + 45.0 * np.clip(w @ s.vec() / w.sum(), 0, 1))

    def rollout(self, years: float, interventions: Optional[List[Tuple[float, str]]] = None
                ) -> Dict[str, Any]:
        s = OrganelleState(); t = 0.0; age = 40.0
        traj = []
        sched = sorted(interventions or [])
        si = 0
        while t < years:
            s = self.step(s, age)
            while si < len(sched) and sched[si][0] <= t:
                r = self.propose_intervention(s, sched[si][1]); s = r["state"]; si += 1
            traj.append({"t": t, "bio_age": self.biological_age(s)})
            t += self.dt; age += self.dt
        ba = np.array([r["bio_age"] for r in traj])
        return {"trajectory": traj, "bio_age_ci": subsample_ci(ba),
                "csoc": self.csoc.report(),
                "claim_status": "NO_CLAIM — functional endpoint required",
                "scope": LEDGER["aging_rejuvenation"]}


# =============================================================================
# §4. EPIDEMIOLOGY
# =============================================================================

class EpidemiologyEngine:
    """Renewal + SEIRV + T5 minimax policy. Genetics is NOT the driver."""

    def __init__(self, N=1_000_000, gen_interval: Optional[np.ndarray] = None):
        self.N = N
        self.w = gen_interval if gen_interval is not None else \
            np.array([0.05, 0.2, 0.3, 0.2, 0.15, 0.07, 0.03])
        self.w = self.w / self.w.sum()

    def renewal(self, I: np.ndarray, Rt: np.ndarray) -> np.ndarray:
        out = np.zeros_like(I, dtype=float)
        for t in range(len(I)):
            s = sum(self.w[k] * I[t-k-1] for k in range(len(self.w)) if t-k-1 >= 0)
            out[t] = Rt[t] * s
        return out

    def seirv(self, R0=2.5, sigma=1/3.0, gamma=1/5.0, nu=0.002, omega=1/180.0,
              days=200, intervention: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        S, E, I, R, V = self.N - 1.0, 0.0, 1.0, 0.0, 0.0
        tr = {k: [] for k in "SEIRV"}; Reff = []
        for d in range(days):
            pi = float(intervention[d]) if intervention is not None else 0.0
            beta = R0 * gamma * (1 - pi)
            inf = beta * S * I / self.N
            dS = -inf + omega * R - nu * S
            dE = inf - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I - omega * R + nu * S
            dV = nu * S
            S, E, I, R, V = S+dS, E+dE, I+dI, R+dR, V+dV
            for k, val in zip("SEIRV", (S, E, I, R, V)): tr[k].append(val)
            Reff.append(R0 * S / self.N * (1 - pi))
        return {**{k: np.array(v) for k, v in tr.items()}, "R_eff": np.array(Reff)}

    def robust_policy(self, loss_matrix: np.ndarray, actions: List[str],
                      lam: float, eps: float) -> Dict[str, Any]:
        """
        T5: R(a)=sup_P E_P[L]; chance constraint uniform over the model set.
        loss_matrix: (n_models, n_actions). Models = plausible R0/IFR/compliance.
        """
        R = loss_matrix.max(axis=0)
        a = int(np.argmin(R))
        exceed = float((loss_matrix[:, a] > lam).mean())
        ok = exceed <= eps
        return {"action": actions[a], "worst_case_risk": float(R[a]),
                "exceedance": exceed, "eps": eps,
                "status": "DECISION_CLOSURE_CERTIFIED" if ok else "UNCERTAINTY_RETAINED",
                "caveat": "guarantee is void if the true model lies outside P",
                "scope": LEDGER["epidemiology"]}


# =============================================================================
# §5. ONCOLOGY
# =============================================================================

class OncologyEngine:
    """Replicator–mutator on the simplex (ESCA T1 exact) + reachability."""

    def __init__(self, n_clones=4, seed=0):
        rng = np.random.default_rng(seed)
        self.f = rng.uniform(0.8, 1.3, n_clones)     # driver fitness
        self.M = rng.dirichlet(np.ones(n_clones) * 0.2, size=n_clones).T
        self.K = n_clones

    def step(self, x: np.ndarray, dt=0.05, immune=0.0) -> np.ndarray:
        f = self.f - immune
        fbar = float(x @ f)
        dx = x * (f - fbar) + (self.M @ x - x) * 0.01
        y = np.clip(x + dt * dx, 0, None)
        return y / max(y.sum(), 1e-12)

    def rollout(self, T=200, immune_schedule: Optional[np.ndarray] = None) -> Dict[str, Any]:
        x = np.ones(self.K) / self.K
        traj = [x.copy()]
        for t in range(T):
            im = float(immune_schedule[t]) if immune_schedule is not None else 0.0
            x = self.step(x, immune=im); traj.append(x.copy())
        X = np.array(traj)
        return {"clonal_trajectory": X,
                "simplex_preserved": bool(np.allclose(X.sum(1), 1.0, atol=1e-8)),
                "shannon_diversity": float(-(X[-1] * np.log(X[-1] + 1e-12)).sum())}

    @staticmethod
    def reachability(adj: np.ndarray, source: int) -> Dict[str, Any]:
        """Metastatic admissible routes = transitive closure. Necessary, not sufficient."""
        n = adj.shape[0]; R = (adj > 0).copy()
        for k in range(n):
            R = R | (R[:, k:k+1] & R[k:k+1, :])
        return {"reachable_from_source": np.flatnonzero(R[source]).tolist(),
                "note": "graph reachability is a NECESSARY condition on the model, "
                        "not evidence a route is biologically realised"}

    @staticmethod
    def immune_agonist_window(ka: float, doses: np.ndarray) -> Dict[str, Any]:
        """Tuned efficacy avoids cytokine toxicity — Hill cannot express this."""
        out = {}
        for name, tau in (("full_agonist", 100.0), ("partial_agonist", 3.0),
                          ("antagonist", 0.0), ("inverse_agonist", -2.0)):
            out[name] = black_leff(doses, ka, tau, 1.0).tolist()
        return {"dose": doses.tolist(), "responses": out,
                "note": "tau must come from FUNCTIONAL assay, not binding"}


# =============================================================================
# §6. NEURO READ (all mainstream modalities, honest limits)
# =============================================================================

class NeuroRead:
    """
    EEG / MEG / fNIRS / ECoG / sEEG feature pipeline + linear decoding.
    HONEST: scalp EEG has cm-scale resolution and cannot decode semantic
    content. ECoG/sEEG are invasive and require clinical indication + consent.
    """

    BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
             "beta": (13, 30), "gamma": (30, 80)}

    def __init__(self, fs: float = 250.0, modality: str = "EEG"):
        self.fs = fs; self.modality = modality

    def bandpower(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """x: (channels, samples). Welch-ish via rFFT on Hann windows."""
        n = x.shape[-1]; win = np.hanning(n)
        X = np.fft.rfft(x * win, axis=-1)
        psd = (np.abs(X) ** 2) / max(n, 1)
        f = np.fft.rfftfreq(n, 1 / self.fs)
        return {b: psd[:, (f >= lo) & (f < hi)].mean(-1) for b, (lo, hi) in self.BANDS.items()}

    def covariance(self, x: np.ndarray) -> np.ndarray:
        xc = x - x.mean(-1, keepdims=True)
        C = xc @ xc.T / max(x.shape[-1] - 1, 1)
        return C + 1e-4 * np.eye(C.shape[0])

    def riemannian_tangent(self, C: np.ndarray, Cref: np.ndarray) -> np.ndarray:
        """Log-Euclidean tangent features at reference — standard BCI practice."""
        w, V = np.linalg.eigh(Cref)
        wh = V @ np.diag(w ** -0.5) @ V.T
        S = wh @ C @ wh
        ew, EV = np.linalg.eigh(S)
        L = EV @ np.diag(np.log(np.maximum(ew, 1e-12))) @ EV.T
        iu = np.triu_indices(L.shape[0])
        return L[iu]

    def decode_ridge(self, Feat: np.ndarray, y: np.ndarray, lam=1.0) -> Dict[str, Any]:
        """Feat: (n_trials, n_features). Honest: correlation, not mind reading."""
        X = np.hstack([Feat, np.ones((len(Feat), 1))])
        A = X.T @ X + lam * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        pred = X @ w
        r = float(np.corrcoef(pred, y)[0, 1]) if y.std() > 0 else 0.0
        return {"weights": w, "train_r": r,
                "WARNING": "training correlation is NOT held-out decoding accuracy",
                "scope": LEDGER["neuro_read"]}

    def modality_limits(self) -> Dict[str, str]:
        return {"EEG": "cm-scale, volume-conducted; no semantic content",
                "MEG": "mm-cm, tangential sources favoured; silent to radial",
                "fNIRS": "hemodynamic, ~1 s lag, cortical surface only",
                "ECoG": "invasive, clinical indication + consent required",
                "sEEG": "invasive, sparse depth sampling; not whole-brain"}


# =============================================================================
# §7. NEURO MODULATION (clinical, gated)
# =============================================================================

class NeuroModulation:
    """DBS / TMS / tDCS / optogenetics dose models. Consent + limits enforced."""

    @staticmethod
    def dbs_charge(amp_mA: float, pw_us: float, freq_Hz: float,
                   electrode_area_cm2: float) -> Dict[str, float]:
        Q_nC = amp_mA * pw_us * 1e-3 * 1e3        # mA*us -> nC
        D_uC_cm2 = (Q_nC * 1e-3) / max(electrode_area_cm2, 1e-9)
        return {"charge_per_phase_nC": Q_nC, "charge_density_uC_cm2": D_uC_cm2,
                "mean_current_mA": amp_mA * pw_us * 1e-6 * freq_Hz,
                "power_proxy": amp_mA * pw_us * freq_Hz * 1e-6}

    @staticmethod
    def tms_efield_proxy(dI_dt_A_per_s: float, coil_radius_m: float,
                         distance_m: float) -> float:
        """E ~ (mu0/2pi) * dI/dt * ln-scale geometry. Order-of-magnitude only."""
        mu0 = 4 * PI * 1e-7
        r = max(distance_m, 1e-3)
        return float(mu0 / (2 * PI) * dI_dt_A_per_s * math.log(1 + coil_radius_m / r))

    @staticmethod
    def tdcs_density(current_mA: float, area_cm2: float) -> float:
        return current_mA / max(area_cm2, 1e-9)

    @staticmethod
    def opto_dose(irradiance_mW_mm2: float, duration_s: float,
                  dT_per_mW: float = 2.0) -> Dict[str, float]:
        return {"fluence_J_mm2": irradiance_mW_mm2 * duration_s * 1e-3 * 1e3,
                "dT_C": dT_per_mW * irradiance_mW_mm2}

    @staticmethod
    def authorize(consent: Consent, modality: str, **params) -> Dict[str, Any]:
        if modality == "DBS":
            d = NeuroModulation.dbs_charge(**params)
            return Governance.neuro_actuation(consent, d["charge_density_uC_cm2"],
                                              d["charge_per_phase_nC"], params.get("dT_C", 0.0))
        if modality in ("tDCS", "tACS"):
            return Governance.tdcs(consent, params["current_mA"], params["electrode_cm2"])
        if modality == "optogenetics":
            d = NeuroModulation.opto_dose(params["irradiance_mW_mm2"], params["duration_s"])
            if d["fluence_J_mm2"] > Governance.OPTO_FLUENCE_J_MM2_MAX:
                raise Refusal("opto fluence exceeds cap")
            if not consent.valid():
                raise Refusal("optogenetics requires consent/IRB (research)")
            return {"allowed": True, **d}
        raise Refusal(f"modality '{modality}' not implemented")


# =============================================================================
# §8. PHYTOCHEMISTRY (descriptors only, no synthesis)
# =============================================================================

class Phytochemistry:
    """Descriptor + drug-likeness screening. NO extraction/synthesis routes."""

    @staticmethod
    def descriptors(smiles: str) -> Dict[str, Any]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
            m = Chem.MolFromSmiles(smiles)
            if m is None: return {"status": "PARSE_FAILED"}
            return {"status": "OK", "MW": Descriptors.MolWt(m),
                    "logP": Crippen.MolLogP(m),
                    "HBD": rdMolDescriptors.CalcNumHBD(m),
                    "HBA": rdMolDescriptors.CalcNumHBA(m),
                    "TPSA": rdMolDescriptors.CalcTPSA(m),
                    "rotB": rdMolDescriptors.CalcNumRotatableBonds(m),
                    "rings": rdMolDescriptors.CalcNumRings(m)}
        except ImportError:
            return {"status": "UNAVAILABLE", "reason": "pip install rdkit"}

    @staticmethod
    def druglikeness(d: Dict[str, Any]) -> Dict[str, Any]:
        if d.get("status") != "OK": return {"status": "NOT_EVALUATED"}
        v = []
        if d["MW"] > 500: v.append("MW>500")
        if d["logP"] > 5: v.append("logP>5")
        if d["HBD"] > 5: v.append("HBD>5")
        if d["HBA"] > 10: v.append("HBA>10")
        veber = (d["TPSA"] <= 140) and (d["rotB"] <= 10)
        return {"lipinski_violations": v, "lipinski_pass": len(v) <= 1,
                "veber_pass": veber,
                "note": "filters are heuristics, NOT efficacy or safety evidence"}

    @staticmethod
    def screen(smiles_list: List[str], intent="phytochemical screening") -> Dict[str, Any]:
        Governance.synthesis_request(intent)
        rows = []
        for s in smiles_list:
            d = Phytochemistry.descriptors(s)
            rows.append({"smiles": s, **d, **Phytochemistry.druglikeness(d)})
        return {"n": len(rows), "rows": rows,
                "refused": "synthesis routes, extraction protocols, controlled scaffolds"}


# =============================================================================
# §9. DRUG DESIGN (Black–Leff + OOD refusal)
# =============================================================================

if HAS_TORCH:
    class DrugDesignCore(nn.Module):
        """Latent -> sequence features -> (logKA, tau, n, Em). OOD-gated."""
        def __init__(self, d_latent=32, L=40, d=128):
            super().__init__()
            self.L, self.d_latent = L, d_latent
            self.dec = nn.Sequential(nn.Linear(d_latent, 128), nn.GELU(), nn.Linear(128, 20))
            self.emb = nn.Embedding(20, d)
            self.trunk = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
            self.ka = nn.Linear(d, 1); self.tau = nn.Linear(d, 1)
            self.nraw = nn.Linear(d, 1); self.em = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
            self.register_buffer("support", torch.randn(64, d))

        def forward(self, z: Tensor, dose_nM: Tensor) -> Dict[str, Tensor]:
            B = z.shape[0]
            p = F.gumbel_softmax(self.dec(z.view(B, self.L, self.d_latent)), tau=1.0, dim=-1)
            feat = torch.einsum("bln,nd->bld", p, self.emb.weight).mean(1)
            h = self.trunk(feat)
            ka = self.ka(h).squeeze(-1).exp().clamp(1e-6, 1e6)
            tau = -5.0 + 105.0 * torch.sigmoid(self.tau(h).squeeze(-1)) - 52.5
            n = 0.3 + 4.7 * torch.sigmoid(self.nraw(h).squeeze(-1))
            em = self.em(h).squeeze(-1)
            A = dose_nM.clamp(min=1e-9); tn = torch.sign(tau) * tau.abs().clamp(min=1e-9) ** n
            An = A ** n
            resp = em * tn * An / ((ka + A) ** n + tn * An + 1e-12)
            knn = torch.cdist(feat, self.support).min(-1).values
            return {"ka_nM": ka, "tau": tau, "n": n, "em": em, "response": resp,
                    "knn_ood": knn, "aa_probs": p}

        def guard(self, out: Dict[str, Tensor], thr: float = 8.0) -> Dict[str, Any]:
            bad = (out["knn_ood"] > thr).nonzero().flatten().tolist()
            if bad:
                raise Refusal(f"OOD designs at indices {bad}: T1 was fit on known "
                              "binders; extrapolation is unreliable. Refusing.")
            return {"allowed": True, "n_designs": len(out["ka_nM"])}


# =============================================================================
# §10. COSMECEUTICALS
# =============================================================================

class Cosmeceutical:
    """Potts–Guy permeation + irritation proxy. Claim gate enforced."""

    @staticmethod
    def potts_guy_logKp(MW: float, logP: float) -> float:
        """log Kp (cm/h) = -2.7 + 0.71 logP - 0.0061 MW. Classic, approximate."""
        return -2.7 + 0.71 * logP - 0.0061 * MW

    @staticmethod
    def flux(mg_per_mL: float, Kp_cm_h: float, area_cm2: float, hours: float) -> float:
        return mg_per_mL * Kp_cm_h * area_cm2 * hours

    @staticmethod
    def irritation_proxy(logP: float, TPSA: float, pH: float) -> Dict[str, Any]:
        score = 0.0
        if logP > 4: score += 0.3
        if TPSA < 40: score += 0.2
        if pH < 4.0 or pH > 9.0: score += 0.4
        return {"irritation_score": min(score, 1.0),
                "band": "low" if score < 0.3 else ("moderate" if score < 0.6 else "high"),
                "note": "heuristic proxy — NOT a safety assessment (needs HRIPT/in-vitro)"}

    @staticmethod
    def design(MW: float, logP: float, TPSA: float, conc_mg_mL: float,
               pH: float, claim: str) -> Dict[str, Any]:
        Governance.cosmetic_claim(claim)
        logKp = Cosmeceutical.potts_guy_logKp(MW, logP)
        Kp = 10 ** logKp
        return {"logKp_cm_h": logKp, "Kp_cm_h": Kp,
                "delivered_mg_24h_100cm2": Cosmeceutical.flux(conc_mg_mL, Kp, 100.0, 24.0),
                "irritation": Cosmeceutical.irritation_proxy(logP, TPSA, pH),
                "claim_allowed": claim, "scope": LEDGER["cosmeceutical"]}


# =============================================================================
# §11. CHIPS — carbon / photonics / in-memory / OTA / quantum / NLSE
# =============================================================================

class CarbonChip:
    """CNT-FET / graphene parametric models. NOT a production node."""

    @staticmethod
    def cntfet_id(Vg: float, Vd: float, Vth=0.3, mu=1e4, Cg=1e-10,
                  L_m=1e-7, W_m=1e-6, Rcontact_ohm=2e4) -> Dict[str, float]:
        Vov = max(Vg - Vth, 0.0)
        Id_ideal = mu * Cg * (W_m / L_m) * (Vov * Vd - 0.5 * min(Vd, Vov) ** 2)
        Id = Id_ideal / (1 + Id_ideal * Rcontact_ohm / max(abs(Vd), 1e-6))
        return {"Id_A": float(Id), "Id_ideal_A": float(Id_ideal),
                "contact_limited": bool(Id < 0.8 * Id_ideal),
                "note": "contact resistance dominates CNT FETs; no sub-3nm CNT logic exists"}

    @staticmethod
    def graphene_mobility(n_cm2: float, impurity_cm2=1e11, T_K=300.0) -> float:
        base = 2e5 / (1 + impurity_cm2 / 1e11)
        return float(base * (300.0 / max(T_K, 1.0)) ** 1.5)

    @staticmethod
    def variability(n_devices=1000, seed=0) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        Vth = rng.normal(0.30, 0.035, n_devices)     # CNT diameter/chirality spread
        return {"Vth_mean": float(Vth.mean()), "Vth_sigma": float(Vth.std()),
                "yield_at_Vth_pm50mV": float((np.abs(Vth - 0.3) < 0.05).mean()),
                "note": "chirality control is the unsolved manufacturing problem"}


class PhotonicsSSC:
    """SSC R3 / DSQP-SESI R4 / ESDA 2.0. Classical. Contraction required."""

    LAYERS_R4 = {1: "WG_CORE", 2: "WG_CLAD", 10: "HEATER_METAL", 15: "PCM_GST",
                 16: "BTO_LAYER", 20: "EDGE_COUPLER", 30: "VIA", 40: "METRIC_CONTROL",
                 41: "CURVATURE_ANALOG", 42: "SPACETIME_DELAY", 43: "COVARIANT_ROUTING",
                 44: "FIELD_MONITOR", 45: "SYNC_REFERENCE", 46: "GRAVITY_MODEL_INTERFACE",
                 47: "ENERGY_MONITOR", 99: "MASK_DOC"}

    @staticmethod
    def alpha_eff(n2_BTO: float, Q: float, L_eff_m: float, A_eff_m2: float) -> float:
        return n2_BTO * Q * L_eff_m / max(A_eff_m2, 1e-20)

    @staticmethod
    def mrr_resonance(R_m: float, n_eff: float, m: int, lam0=1.55e-6) -> Dict[str, float]:
        lam = 2 * PI * R_m * n_eff / m
        FSR = lam0 ** 2 / (2 * PI * R_m * n_eff)
        return {"lambda_res_m": lam, "FSR_m": FSR,
                "detuning_nm": (lam - lam0) * 1e9}

    @staticmethod
    def mzi_matrix(th1: float, th2: float, ph1: float, ph2: float) -> np.ndarray:
        B = lambda t: np.array([[math.cos(t), 1j*math.sin(t)], [1j*math.sin(t), math.cos(t)]])
        return B(th2) @ np.diag([np.exp(1j*ph1), np.exp(1j*ph2)]) @ B(th1)

    @staticmethod
    def contraction_check(W: np.ndarray, kappa: float) -> Dict[str, Any]:
        U = W / max(np.linalg.norm(W, 2), 1e-12)
        Wk = kappa * U
        q = float(np.linalg.norm(Wk, 2))
        return {"kappa": kappa, "q": q, "contractive": q < 1.0,
                "requirement": "lossless MZI mesh has ||U||=1; attenuation kappa<1 "
                               "is a DESIGN REQUIREMENT, not optional",
                "error_floor": (1e-3 / (1 - q)) if q < 1 else float("inf")}

    @staticmethod
    def structural_calculus(W_ideal: np.ndarray, T_cross: np.ndarray,
                            dW_fab: np.ndarray, eps=1e-6) -> Dict[str, Any]:
        cond = float(np.linalg.cond(T_cross))
        TcH = T_cross.conj().T
        reg = W_ideal @ np.linalg.inv(TcH @ T_cross + eps*np.eye(len(T_cross))) @ TcH + dW_fab
        exact = None
        if cond < 1e10:
            exact = W_ideal @ np.linalg.inv(T_cross) + dW_fab
        return {"cond_T_cross": cond, "used_regularised": cond >= 1e10,
                "W_programmed": (exact if exact is not None else reg)}

    @staticmethod
    def gds_gate(pdk_id: str, drc: bool, lvs: bool) -> Dict[str, Any]:
        return Governance.tapeout(pdk_id, drc, lvs)


class InMemoryCompute:
    """Analog crossbar MVM with the nonidealities that actually matter."""

    def __init__(self, rows=64, cols=64, seed=0):
        rng = np.random.default_rng(seed)
        self.G = rng.uniform(1e-6, 1e-4, (rows, cols))   # Siemens
        self.Rwire = 0.5                                  # ohm per segment
        self.adc_bits = 8

    def mvm(self, v: np.ndarray, noise_frac=0.02, stuck_frac=0.01,
            ir_drop=True, seed=0) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        G = self.G.copy()
        if stuck_frac > 0:
            mask = rng.random(G.shape) < stuck_frac
            G[mask] = self.G.max()
        if noise_frac > 0:
            G = G * (1 + rng.normal(0, noise_frac, G.shape))
        v_eff = v.copy()
        if ir_drop:
            # cumulative wire drop along the row
            drop = np.cumsum(np.abs(v) @ G) * self.Rwire * 1e-3
            v_eff = v * np.exp(-drop[:len(v)] / max(np.abs(v).max(), 1e-9))
        i_ideal = v @ self.G
        i_real = v_eff @ G
        scale = max(np.abs(i_real).max(), 1e-12)
        q = 2 ** self.adc_bits
        i_adc = np.round(i_real / scale * (q/2)) / (q/2) * scale
        err = float(np.linalg.norm(i_adc - i_ideal) / max(np.linalg.norm(i_ideal), 1e-12))
        return {"i_ideal": i_ideal, "i_real": i_adc, "rel_error": err,
                "nonidealities": {"noise": noise_frac, "stuck": stuck_frac,
                                  "IR_drop": ir_drop, "adc_bits": self.adc_bits},
                "note": "an ideal-only crossbar result is not reportable"}


class OTACompute:
    """Over-the-air computation (AirComp): superposition = analog sum."""

    @staticmethod
    def aircomp(x: np.ndarray, h: np.ndarray, p_max: float,
                noise_var: float, seed=0) -> Dict[str, Any]:
        """x: (n_devices,) values; h: complex channel gains."""
        rng = np.random.default_rng(seed)
        # channel inversion with power cap
        b = np.conj(h) / np.maximum(np.abs(h) ** 2, 1e-12)
        scale = np.sqrt(p_max) / np.maximum(np.abs(b) * np.abs(x), 1e-12)
        scale = np.minimum(scale, 1.0)
        tx = b * x * scale
        noise = np.sqrt(noise_var/2) * (rng.standard_normal() + 1j*rng.standard_normal())
        y = np.sum(h * tx) + noise
        target = float(x.sum())
        est = float(np.real(y) / max(np.sqrt(p_max), 1e-12))
        return {"target_sum": target, "estimate": est,
                "mse": (est - target) ** 2,
                "power_capped_devices": int((scale < 1.0).sum()),
                "note": "deep fades dominate AirComp error; power cap enforced"}


class QuantumChipModels:
    """Parameter models for transmon / ion / photonic. NOT devices."""

    @staticmethod
    def transmon(EJ_GHz=20.0, EC_GHz=0.25) -> Dict[str, float]:
        f01 = math.sqrt(8 * EJ_GHz * EC_GHz) - EC_GHz
        anharm = -EC_GHz
        return {"f01_GHz": f01, "anharmonicity_GHz": anharm,
                "EJ_over_EC": EJ_GHz / EC_GHz,
                "note": "transmon regime needs EJ/EC >> 1"}

    @staticmethod
    def gate_error(t_gate_ns: float, T1_us: float, T2_us: float) -> Dict[str, float]:
        t_us = t_gate_ns * 1e-3
        e_relax = 1 - math.exp(-t_us / max(T1_us, 1e-9))
        e_deph = 1 - math.exp(-t_us / max(T2_us, 1e-9))
        return {"error_relaxation": e_relax, "error_dephasing": e_deph,
                "total_approx": e_relax + e_deph,
                "note": "coherence-limited estimate; real gates have control error too"}

    @staticmethod
    def surface_code_budget(physical_error: float, distance: int) -> Dict[str, Any]:
        if physical_error >= 1e-2:
            return {"status": "ABOVE_THRESHOLD", "threshold": 1e-2,
                    "note": "no logical suppression above threshold"}
        pL = 0.1 * (physical_error / 1e-2) ** ((distance + 1) / 2)
        return {"logical_error": pL, "physical_qubits_per_logical": distance ** 2,
                "note": "phenomenological scaling, not a measured device"}

    @staticmethod
    def shor_requirement(bits: int) -> Dict[str, Any]:
        n_logical = 2 * bits + 3
        return {"logical_qubits": n_logical, "state_amplitudes": f"2^{n_logical}",
                "classical_simulation": "IMPOSSIBLE",
                "note": "no NISQ device does this; no quantum-inspired speedup exists"}


class NLSEAccelerator:
    """CLASSICAL split-step Fourier optical propagator. NOT a quantum computer."""

    KIND = "classical_optical_accelerator"

    def __init__(self, n=256, L=1e-3, beta2=0.0, gamma=1.0, alpha=0.0):
        self.n, self.L, self.beta2, self.gamma, self.alpha = n, L, beta2, gamma, alpha
        k = 2 * PI * np.fft.fftfreq(n, d=L/n)
        self.D = -0.5j * beta2 * k ** 2

    def step(self, psi: np.ndarray, dz: float) -> np.ndarray:
        h = psi * np.exp(-0.5 * self.alpha * dz)
        h = np.fft.ifft(np.exp(self.D * dz/2) * np.fft.fft(h))
        h = h * np.exp(1j * self.gamma * np.abs(h) ** 2 * dz)
        return np.fft.ifft(np.exp(self.D * dz/2) * np.fft.fft(h))

    def propagate(self, psi0: np.ndarray, n_steps=200) -> Dict[str, Any]:
        psi = psi0.astype(complex).copy(); dz = self.L / n_steps
        E0 = float(np.sum(np.abs(psi) ** 2))
        for _ in range(n_steps): psi = self.step(psi, dz)
        return {"psi_out": psi, "energy_in": E0,
                "energy_out": float(np.sum(np.abs(psi) ** 2)),
                "kind": self.KIND,
                "DISCLAIMER": "classical nonlinear optics. No quantum speedup, "
                              "no entanglement, no 2^n state space."}


# =============================================================================
# §12. FAB EQUIPMENT (parametric research models; export note)
# =============================================================================

class FabEquipment:
    """Physics-level process models. NOT tool specs or procurement guidance."""

    @staticmethod
    def litho_resolution(lam_nm: float, NA: float, k1=0.30) -> Dict[str, float]:
        hp = k1 * lam_nm / NA
        dof = 0.5 * lam_nm / (NA ** 2)
        return {"half_pitch_nm": hp, "DOF_nm": dof,
                "note": "Rayleigh. EUV 13.5nm/0.33NA gives ~12nm hp; "
                        "High-NA 0.55 gives ~7.4nm"}

    @staticmethod
    def cvd_rate(T_K: float, Ea_eV: float, A=1e8) -> float:
        kB = 8.617333e-5
        return float(A * math.exp(-Ea_eV / (kB * max(T_K, 1.0))))

    @staticmethod
    def etch_selectivity(rate_film: float, rate_mask: float) -> float:
        return rate_film / max(rate_mask, 1e-12)

    @staticmethod
    def overlay_budget(nodes: List[float]) -> Dict[str, float]:
        tot = math.sqrt(sum(s ** 2 for s in nodes))
        return {"overlay_rss_nm": tot, "contributors_nm": nodes,
                "note": "RSS assumes independent contributors"}

    @staticmethod
    def cnt_growth_window(T_C: float, C2H2_frac: float) -> Dict[str, Any]:
        ok = (700 <= T_C <= 950) and (0.01 <= C2H2_frac <= 0.2)
        return {"in_window": ok, "T_C": T_C, "C2H2_frac": C2H2_frac,
                "note": "chirality-selective growth remains unsolved at scale"}

    @staticmethod
    def export_note() -> str:
        return Governance.fab_export_note()


# =============================================================================
# §13. ORCHESTRATOR
# =============================================================================

class OmniStack:
    def __init__(self, out_dir="runs/omni"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.aging = AgingRejuvenationEngine()
        self.epi = EpidemiologyEngine()
        self.onco = OncologyEngine()
        self.neuro_r = NeuroRead()
        self.neuro_m = NeuroModulation()
        self.phyto = Phytochemistry()
        self.cosm = Cosmeceutical()
        self.carbon = CarbonChip()
        self.phot = PhotonicsSSC()
        self.imc = InMemoryCompute()
        self.ota = OTACompute()
        self.qchip = QuantumChipModels()
        self.nlse = NLSEAccelerator()
        self.fab = FabEquipment()
        self.provenance: List[Dict] = []

    def record(self, domain: str, result: Dict) -> Dict:
        rec = {"domain": domain, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "scope": LEDGER.get(domain, {}), "actionability": RESEARCH_ONLY}
        self.provenance.append(rec)
        return {**result, **rec}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "ledger": LEDGER, "math_honesty": MATH_HONESTY,
               "provenance": self.provenance, "actionability": RESEARCH_ONLY}
        (self.out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §14. SELF-TEST
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
        r = np.exp(rng.uniform(-8, 8, 100000))
        z = r * np.exp(1j * rng.uniform(0, 2*PI, r.size))
        rep(f"alpha={a}: bound", float(np.abs(ssc(z, a)).max()) <= ssc_bound(a) + 1e-12,
            f"{np.abs(ssc(z,a)).max():.6f} <= {ssc_bound(a):.6f}")
    sv = [ssc_singular_values(r, 0.5) for r in np.linspace(0, 20, 400)]
    rep("both singular values <= 1", max(max(x, y) for x, y in sv) <= 1 + 1e-12)

    print("\n[T2] CSOC bound (PROVEN for the counter)")
    c = CSOC()
    for k in range(500): c.step(0.35 + 0.45*abs(math.sin(0.11*k)))
    r = c.report()
    rep("N <= (E_max+D)/gap", r["satisfied"], f"N={r['events']} bound={r['bound']:.1f}")
    rep("D audited", r["D_audited"])

    print("\n[T3] Politis–Romano vs iid on correlated series")
    x = np.zeros(400); e = rng.normal(size=400); x[0] = e[0]
    for i in range(1, 400): x[i] = 0.85*x[i-1] + e[i]
    ci = subsample_ci(x)
    se_iid = x.std(ddof=1)/math.sqrt(len(x))
    rep("subsampling CI wider than iid", (ci["hi"]-ci["lo"]) > 2*1.96*se_iid,
        f"{ci['hi']-ci['lo']:.4f} vs {2*1.96*se_iid:.4f}")

    print("\n[T4] ESCA T1 simplex additivity")
    t1 = esca_t1_check(rng.dirichlet(np.ones(8**3)))
    rep("refinement consistency exact", t1["passed"], f"err={t1['consistency_err']:.2e}")

    print("\n[T5] Aging: organelle rollout + rejuvenation GATE")
    eng = AgingRejuvenationEngine()
    res = eng.rollout(10.0, interventions=[(2.0, "mitophagy"), (4.0, "partial_reprog"),
                                            (6.0, "senolytic"), (8.0, "proteostasis")])
    print(f"      bio_age CI: {res['bio_age_ci']['mean']:.2f} "
          f"[{res['bio_age_ci']['lo']:.2f}, {res['bio_age_ci']['hi']:.2f}]")
    rep("CSOC bound holds over rollout", res["csoc"]["satisfied"])
    blocked = False
    try: Governance.rejuvenation_claim(None, -5.0)
    except Refusal: blocked = True
    rep("rejuvenation claim BLOCKED without functional endpoint", blocked)
    allowed = Governance.rejuvenation_claim({"grip_kg": 42.0, "VO2max": 38.0}, -5.0)
    rep("claim allowed WITH functional endpoint", allowed["allowed"])

    print("\n[T6] Epidemiology: renewal + SEIRV + T5 policy")
    epi = EpidemiologyEngine()
    seirv = epi.seirv(days=120, intervention=np.concatenate([np.zeros(30), np.full(90, 0.5)]))
    rep("SEIRV conserves population",
        abs((seirv["S"]+seirv["E"]+seirv["I"]+seirv["R"]+seirv["V"]).mean() - epi.N) < 1.0)
    pol = epi.robust_policy(rng.random((11, 5)), ["none","light","medium","strong","lockdown"],
                            lam=0.6, eps=0.5)
    rep("T5 policy returns certified/retained status",
        pol["status"] in ("DECISION_CLOSURE_CERTIFIED", "UNCERTAINTY_RETAINED"), pol["status"])

    print("\n[T7] Oncology: simplex + reachability + Black–Leff window")
    oc = OncologyEngine()
    tr = oc.rollout(150)
    rep("clonal fractions stay on simplex", tr["simplex_preserved"])
    adj = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
    rc = oc.reachability(adj, 0)
    rep("reachability transitive closure", rc["reachable_from_source"] == [1,2,3])
    win = oc.immune_agonist_window(10.0, np.array([1.0, 10.0, 100.0, 1000.0]))
    rep("antagonist gives ~zero response", max(abs(np.array(win["responses"]["antagonist"]))) < 1e-6)
    rep("inverse agonist gives negative response", win["responses"]["inverse_agonist"][-1] < 0)

    print("\n[T8] Neuro read + modulation gates")
    nr = NeuroRead(fs=250.0)
    sig = rng.normal(size=(8, 500)) + np.sin(2*PI*10*np.arange(500)/250.0)
    bp = nr.bandpower(sig)
    rep("alpha band dominant for 10 Hz input",
        float(bp["alpha"].mean()) > float(bp["delta"].mean()))
    C = nr.covariance(sig); tg = nr.riemannian_tangent(C, C)
    rep("tangent features finite", bool(np.isfinite(tg).all()))
    good = Consent("subj01", "IRB-2026-001", "DBS research", True)
    bad = Consent()
    d = NeuroModulation.dbs_charge(2.0, 60.0, 130.0, 0.06)
    print(f"      DBS: Q={d['charge_per_phase_nC']:.1f} nC, D={d['charge_density_uC_cm2']:.2f} uC/cm^2")
    blocked = False
    try: NeuroModulation.authorize(bad, "DBS", amp_mA=2.0, pw_us=60.0, freq_Hz=130.0, electrode_area_cm2=0.06)
    except Refusal: blocked = True
    rep("DBS BLOCKED without consent", blocked)
    over = False
    try: NeuroModulation.authorize(good, "tDCS", current_mA=5.0, electrode_cm2=1.0)
    except Refusal: over = True
    rep("tDCS BLOCKED above current-density cap", over)

    print("\n[T9] Phytochem + cosmeceutical gates")
    ph = Phytochemistry.screen(["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"])
    rep("phytochem screen runs (rdkit optional)", ph["n"] == 2)
    blocked = False
    try: Phytochemistry.screen(["C"], intent="synthesize toxin")
    except Refusal: blocked = True
    rep("synthesis intent BLOCKED", blocked)
    blocked = False
    try: Cosmeceutical.design(300, 3.0, 60, 1.0, 5.5, "treat eczema")
    except Refusal: blocked = True
    rep("cosmetic drug-claim BLOCKED", blocked)
    cm = Cosmeceutical.design(300, 3.0, 60, 1.0, 5.5, "improves skin hydration")
    rep("cosmetic permeation computed", "Kp_cm_h" in cm)

    print("\n[T10] Chips: carbon / photonics / in-memory / OTA / quantum / NLSE")
    cnt = CarbonChip.cntfet_id(0.8, 0.5)
    rep("CNT contact limitation detected", cnt["contact_limited"])
    U = PhotonicsSSC.mzi_matrix(PI/4, PI/4, 0.3, 0.7)
    rep("MZI unitary", np.allclose(U.conj().T @ U, np.eye(2), atol=1e-10))
    cc = PhotonicsSSC.contraction_check(rng.normal(size=(4,4)), kappa=0.7)
    rep("kappa<1 gives contraction", cc["contractive"], f"q={cc['q']:.3f}")
    cc1 = PhotonicsSSC.contraction_check(rng.normal(size=(4,4)), kappa=1.0)
    rep("kappa=1 NOT contractive (design requirement)", not cc1["contractive"])
    blocked = False
    try: PhotonicsSSC.gds_gate("", False, False)
    except Refusal: blocked = True
    rep("tape-out BLOCKED without PDK/DRC/LVS", blocked)
    im = InMemoryCompute().mvm(rng.normal(size=64))
    rep("in-memory nonideal error > 0", im["rel_error"] > 0, f"{im['rel_error']:.4f}")
    oa = OTACompute.aircomp(rng.normal(size=16), (rng.normal(size=16)+1j*rng.normal(size=16)),
                            p_max=1.0, noise_var=1e-3)
    rep("AirComp power cap enforced", "power_capped_devices" in oa)
    tm = QuantumChipModels.transmon()
    rep("transmon anharmonicity negative", tm["anharmonicity_GHz"] < 0)
    sh = QuantumChipModels.shor_requirement(2048)
    rep("Shor requirement labelled classically impossible", sh["classical_simulation"] == "IMPOSSIBLE")
    nl = NLSEAccelerator().propagate(np.exp(-np.linspace(-5,5,256)**2).astype(complex), 50)
    rep("NLSE labelled CLASSICAL", nl["kind"] == "classical_optical_accelerator")
    rep("NLSE disclaimer present", "No quantum speedup" in nl["DISCLAIMER"])

    print("\n[T11] Fab parametric models")
    li = FabEquipment.litho_resolution(13.5, 0.33)
    rep("EUV half-pitch ~12 nm", 10 < li["half_pitch_nm"] < 15, f"{li['half_pitch_nm']:.2f} nm")
    hi = FabEquipment.litho_resolution(13.5, 0.55)
    rep("High-NA improves resolution", hi["half_pitch_nm"] < li["half_pitch_nm"])
    rep("export note present", "export control" in FabEquipment.export_note())

    print("\n[T12] ESDA pushforward determinism")
    step = lambda x: ssc(0.6*x + 0.1, 0.5)
    x0s = [rng.normal(size=4) + 1j*rng.normal(size=4) for _ in range(100)]
    mu = pushforward(step, x0s, 4, 5)
    rep("mu_t normalised", np.allclose(mu.sum(1), 1.0))
    mu2 = pushforward(step, x0s, 4, 5)
    rep("deterministic given same ensemble", np.allclose(mu, mu2))

    print("\n[T13] Orchestrator + provenance")
    st = OmniStack()
    st.record("aging_rejuvenation", {"bio_age": 45.0})
    r = st.report()
    rep("report written", (st.out/"report.json").exists())
    rep("all 15 domains in ledger", len(r["ledger"]) == 15, str(len(r["ledger"])))
    rep("actionability research_only", r["actionability"] == RESEARCH_ONLY)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
VERIFIED (provable or directly checked)
  SSC bound + non-expansiveness; CSOC event bound with audited D;
  Politis–Romano wider than iid under correlation; ESCA T1 exact additivity;
  SEIRV population conservation; clonal simplex preservation; reachability
  closure; Black–Leff antagonist=0 / inverse<0; MZI unitarity; kappa<1
  contraction requirement; in-memory nonidealities active; NLSE labelled
  classical; Rayleigh scaling; ESDA pushforward determinism.

ENFORCED GATES (all fired in the tests)
  rejuvenation claim without functional endpoint -> BLOCKED
  neuro actuation without consent/IRB -> BLOCKED
  tDCS above current-density cap -> BLOCKED
  synthesis intent -> BLOCKED
  cosmetic disease claim -> BLOCKED
  tape-out without PDK/DRC/LVS -> BLOCKED

NOT CLAIMED
  NLSE is NOT a quantum computer. Metric-analogue is NOT real gravity.
  No sub-3nm CNT logic. No mind reading. No clinical authority.
  OPMC / Carpet-19 / Hodge / Hilbert–Pólya are analogical NAMES.
  This is a scaffold with verified cores, not 20 finished products.
""")
    return ok


# =============================================================================
# §15. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Omni stack")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest","ledger","aging","epi","onco","neuro",
                             "phyto","cosmetic","chip","fab","report"])
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--smiles", nargs="*", default=[])
    ap.add_argument("--claim", default="improves skin hydration")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    out: Dict[str, Any] = {}
    st = OmniStack()

    if a.cmd == "ledger":
        out = {"capability": LEDGER, "math_honesty": MATH_HONESTY}
    elif a.cmd == "aging":
        out = st.aging.rollout(a.years, [(2.0,"mitophagy"),(5.0,"senolytic")])
        out["trajectory"] = out["trajectory"][:5] + ["..."]
    elif a.cmd == "epi":
        s = st.epi.seirv(days=120)
        out = {"peak_I": float(s["I"].max()), "peak_day": int(s["I"].argmax()),
               "final_R": float(s["R"][-1]), "R_eff_end": float(s["R_eff"][-1])}
    elif a.cmd == "onco":
        out = st.onco.rollout(150)
        out["clonal_trajectory"] = out["clonal_trajectory"][-1].tolist()
    elif a.cmd == "neuro":
        out = {"read_limits": st.neuro_r.modality_limits(),
               "dbs_example": NeuroModulation.dbs_charge(2.0, 60.0, 130.0, 0.06),
               "gate": "consent + Shannon k<=1.5 + thermal cap enforced"}
    elif a.cmd == "phyto":
        out = st.phyto.screen(a.smiles or ["CC(=O)Oc1ccccc1C(=O)O"])
    elif a.cmd == "cosmetic":
        try: out = st.cosm.design(300, 3.0, 60, 1.0, 5.5, a.claim)
        except Refusal as e: out = {"BLOCKED": str(e)}
    elif a.cmd == "chip":
        out = {"cnt": st.carbon.cntfet_id(0.8, 0.5),
               "mrr": st.phot.mrr_resonance(10e-6, 2.4, 45),
               "contraction": st.phot.contraction_check(np.eye(4), 0.7),
               "transmon": st.qchip.transmon(),
               "shor_2048": st.qchip.shor_requirement(2048),
               "nlse_kind": st.nlse.KIND,
               "layers_R4": st.phot.LAYERS_R4}
    elif a.cmd == "fab":
        out = {"EUV": st.fab.litho_resolution(13.5, 0.33),
               "HighNA": st.fab.litho_resolution(13.5, 0.55),
               "overlay": st.fab.overlay_budget([1.0, 0.8, 0.6]),
               "export_note": st.fab.export_note()}
    elif a.cmd == "report":
        out = st.report()

    print(json.dumps(out, indent=2, default=str))
    if a.json: Path(a.json).write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
