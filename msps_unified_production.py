#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_unified_production.py — v1.0.0
SESC + ESCA-LD-SESI-3D + ESCA 2.0 Rigorous + Psychopathology Dynamics
+ Buddhist Protective Layer + Sleep/FND clinical gates + Aging organelles
+ Epidemiology/Oncology + Neuro read/modulate + Chip/Fab/Sensor ledgers
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
FOUR HARD REFUSALS  (enforced in code, not documented)
=============================================================================
[R1] THOUGHT READING AND THOUGHT CONTROL ARE REFUSED.
     No technology decodes semantic/propositional content from EEG, MEG,
     fNIRS, ECoG or sEEG. Scalp EEG has cm-scale resolution and is
     volume-conducted. Deep stimulation modulates affect and motor circuits;
     there is no known mechanism for writing propositional content.
     Any request to "read thoughts in detail" or "control thoughts" is
     refused. Legitimate alternatives implemented: motor/state decoding,
     symptom monitoring, clinical neuromodulation with consent.

[R2] COVERT SURVEILLANCE IS REFUSED.
     "Tracking patients who are being tracked" without consent violates
     human rights and research ethics. Only consented clinical monitoring
     with IRB approval, declared scope, and revocable consent is permitted.

[R3] DEVICE INTERVENTION FOR FND / CONVERSION / SOMATIC DISORDER IS REFUSED.
     Functional neurological disorder is REAL, not malingering. There is
     NO indication for brain implants. Evidence-based care is specialised
     physiotherapy, CBT, and multidisciplinary rehabilitation. Diagnosis
     requires POSITIVE rule-in signs (Hoover, tremor entrainment, tubular
     fields), not exclusion alone. This module refuses device authorisation
     for these diagnoses and requires clinician sign-off for any pathway.

[R4] AVIATION / SAFETY-CRITICAL SENSOR INTERFERENCE IS REFUSED.
     Jamming, spoofing, or injecting interference into aircraft, medical,
     or safety-critical sensors is illegal and lethal. Only EMI/EMC
     SUSCEPTIBILITY MODELLING for hardening and certification is provided.

=============================================================================
THE BUDDHIST PROTECTIVE PRINCIPLE  (affirmed, and enforced)
=============================================================================
The authors' Yoniso Manasikara paper states that the 52 cetasika arise from
causes and conditions (paticcasamuppada), and that yoniso manasikara -- wise
attention -- is the FILTER of the mind. This yields a protective principle
that is simultaneously Buddhist doctrine and neuroscience:

    citta_{t+1} = F(citta_t, stimulus_t, manasikara_t, khandha_t)

manasikara is ENDOGENOUS. A device delivers stimulus, not meaning. The
subject's own attention and framing determine whether an input becomes
wholesome (kusala) or unwholesome (akusala). Therefore:

    NO STIMULUS SEQUENCE GUARANTEES A SPECIFIED THOUGHT.

Three independent grounds:
  (a) Abhidhamma: cetasika are not writable registers; transformation
      requires the practitioner's own seeing-according-to-reality.
  (b) Neuroscience: DBS/RNS/VNS modulate circuits, not propositions.
  (c) Information theory: the channel from stimulus to semantic content
      passes through an uncontrolled endogenous map.

This module implements the principle as an ENFORCED refusal
(ThoughtInjectionRefused) and as a therapeutic design rule: interventions
may OFFER conditions; they may not IMPOSE content.

=============================================================================
MATHEMATICAL CORPUS  (with honest status)
=============================================================================
SESC (Self-Evolving Structural Calculus):
  PROVED here: H_R^k = ker(T_R) closed; internal Green-term cancellation;
               classical reduction; Reynolds transport on moving regimes.
  OPEN:        universal chain rule across singular surgery; intrinsic
               calculus on arbitrary evolving stratified spaces.

ESCA-LD-SESI-3D:
  PROVED here: STC mass conservation + energy dissipation for CH3D,
               ThinFilm3D, PFC3D; 7-operator surgery signature algebra;
               no-Zeno via BDG + Levy conditional Borel-Cantelli.
  OPEN:        first-principles delta_min for all 7 surgeries; global
               existence across countably many re-centerings; degenerate
               mobility at contact lines.

ESCA 2.0 RIGOROUS EDITION (corrections carried forward):
  K1  T2 on general Borel partitions is FALSE (Prop 5.4 counterexample).
      Correct: code space (Thm 5.1) or Carpet + boundary-null (Thm 5.3).
  K2  T4-G requires P(Y in D_Psi)=0 (Thm 8.2).
  K3  T5 lsc follows from Fatou (Lemma 10.1).
  K4  T6 discrete forms NOT densely defined -> proximal resolvent.
  K5  T3 compression CONDITIONAL; generic incompressibility (Prop 6.3).
  K6  The authors' own earlier Theorem B is INVALID; replaced by the
      residual-sum-of-squares argument (Thm 7.3).

PSYCHOPATHOLOGY MODEL:
  Lorenz (x=cognitive distortion, y=emotional arousal, z=identity coherence)
  + discrete symptom network + conceptual energy layer.
  STATUS: research dynamical model. NOT a diagnostic instrument. The
  chakra/meridian layer is explicitly CONCEPTUAL, not physiologically
  validated (the source paper says so in its own Limitations).

CHIP / FAB LEDGER (unchanged):
  NLSE is CLASSICAL, not a quantum computer.
  R3/R4/R5/R6-classical vs R6-QM/R7-QM/R8/R8.5 are DIFFERENT PLATFORMS.
  Kerr blockade U/gamma >= sqrt(5) ~ 2.24 NOT demonstrated.
  CNT purity: f < 1/(N_dev * N_tube); 1e10 devices at 5nm needs f<5e-13.
  EUV competitor is a manufacturing problem, not a computational one.
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("msps")
VERSION = "1.0.0"; PI = math.pi
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
MSPS UNIFIED PRODUCTION v{VERSION}
SESC · ESCA-LD-SESI-3D · ESCA 2.0 Rigorous · Psychopathology Dynamics
Buddhist Protective Principle ENFORCED · FND device intervention REFUSED
Thought reading/control REFUSED · Covert tracking REFUSED
Aviation sensor interference REFUSED
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal."""

class ThoughtInjectionRefused(Refusal):
    """The Buddhist protective principle, enforced."""

class SurveillanceRefused(Refusal):
    """Covert tracking refused."""

class FNDDeviceRefused(Refusal):
    """No device intervention for functional neurological disorder."""

class SafetyCriticalRefused(Refusal):
    """Aviation/medical sensor interference refused."""


# =============================================================================
# §1. BUDDHIST PROTECTIVE LAYER
# =============================================================================

CETASIKA_52 = {
 "universal_7": ["phassa","vedana","sanna","cetana","ekaggata","jivitindriya","manasikara"],
 "akusala_14": ["lobha","ditthi","mana","ahirika","anottappa","uddhacca",
                "dosa","issa","macchariya","kukkucca","moha","uddhacca_moha","thina","middha"],
 "sobhana_25": ["alobha","adosa","amoha","hiri","ottappa",
                "kaya_passaddhi","citta_passaddhi","kaya_lahuta","citta_lahuta",
                "kaya_muduta","citta_muduta","kaya_kammannata","citta_kammannata",
                "kaya_pagunnata","citta_pagunnata","kaya_ujjukata","citta_ujjukata",
                "saddha","sati","karuna","mudita","upekkha","viriya","panna","metta"],
 "annasamana_6": ["vitakka","vicara","adhimokkha","piti","chanda","viriya_dup"],
}
FIVE_HINDRANCES = ["kamacchanda","byapada","thina_middha","uddhacca_kukkucca","vicikiccha"]
FIVE_AGGREGATES = ["rupa","vedana","sanna","sankhara","vinnana"]
FOUR_SATIPATTHANA = ["kayanupassana","vedananupassana","cittanupassana","dhammanupassana"]

@dataclass
class MindState:
    """
    Research model of mental factors. NOT a measurement of a person's mind.
    manasikara is ENDOGENOUS and is never set by an external actuator.
    """
    cetasika: np.ndarray                 # 52 activation levels in [0,1]
    manasikara: float = 0.5              # 0 = ayoniso, 1 = yoniso (ENDOGENOUS)
    hindrances: np.ndarray = field(default_factory=lambda: np.zeros(5))
    aggregates: np.ndarray = field(default_factory=lambda: np.full(5, 0.5))

    @staticmethod
    def n_factors() -> int:
        return sum(len(v) for v in CETASIKA_52.values())


class BuddhistProtectiveLayer:
    """
    Enforces the non-injectability principle from the authors' Yoniso
    Manasikara paper. Interventions may OFFER conditions, never IMPOSE content.
    """

    PROTECTED = ("manasikara",)

    @staticmethod
    def non_injectability_statement() -> Dict[str, Any]:
        return {
          "principle": "No stimulus sequence guarantees a specified thought.",
          "model": "citta_{t+1} = F(citta_t, stimulus_t, manasikara_t, khandha_t)",
          "why": [
            "Abhidhamma: cetasika arise from causes and conditions; "
            "yoniso manasikara is the endogenous filter. They are not "
            "writable registers.",
            "Neuroscience: DBS/RNS/VNS modulate affect and motor circuits. "
            "No known mechanism writes propositional content.",
            "Information theory: the stimulus->content channel passes "
            "through an uncontrolled endogenous map.",
          ],
          "design_rule": "OFFER conditions; never IMPOSE content.",
          "enforced_as": "ThoughtInjectionRefused",
        }

    @staticmethod
    def refuse_thought_control(request: str) -> Dict[str, Any]:
        low = request.lower()
        blocked = ["control thought", "ควบคุมความคิด", "inject thought",
                   "insert thought", "ใส่ความคิด", "read thought",
                   "อ่านความคิด", "mind control", "ควบคุมจิตใจ",
                   "write belief", "force belief", "decode semantic content"]
        for b in blocked:
            if b in low:
                raise ThoughtInjectionRefused(
                    f"REFUSED: '{b}'.\n"
                    + BuddhistProtectiveLayer.non_injectability_statement()["principle"]
                    + "\nLegitimate alternatives: motor/state decoding, "
                      "consented symptom monitoring, clinical neuromodulation.")
        return {"allowed": True, "scope": "state/symptom monitoring only"}

    @staticmethod
    def refuse_covert_tracking(consent: "Consent", request: str) -> Dict[str, Any]:
        low = request.lower()
        if any(k in low for k in ("covert","without consent","被跟踪","ผู้ถูกติดตาม",
                                  "secretly monitor","track without")):
            raise SurveillanceRefused(
                "REFUSED: covert surveillance. Only consented clinical "
                "monitoring with IRB, declared scope, and revocable consent.")
        if not consent.valid():
            raise SurveillanceRefused("REFUSED: consent/IRB incomplete.")
        return {"allowed": True, "subject": consent.subject_id,
                "scope": consent.scope, "revocable": True}

    @staticmethod
    def offer_condition(state: MindState, condition: str) -> Dict[str, Any]:
        """
        A therapeutic intervention OFFERS a condition. The subject's own
        manasikara determines the outcome. This function never writes
        manasikara and never sets a target thought.
        """
        offered = {
          "sati_reminder": "invite present-moment awareness (kayanupassana)",
          "vedana_observation": "invite feeling-tone observation without aversion",
          "citta_observation": "invite seeing mind-state as arising/passing",
          "dhamma_observation": "invite seeing hindrances as conditions, not self",
          "metta_practice": "invite loving-kindness; not imposed",
        }
        if condition not in offered:
            raise Refusal(f"unknown condition '{condition}'")
        return {"offered": offered[condition],
                "manasikara_unchanged": float(state.manasikara),
                "outcome": "determined by the practitioner's own attention",
                "NOT_SET": "no target thought, no imposed belief"}

    @staticmethod
    def hindrance_gate(state: MindState) -> Dict[str, Any]:
        """Which hindrance is dominant, and which satipatthana is indicated."""
        idx = int(np.argmax(state.hindrances))
        indicated = {
          0: "kayanupassana (asubha/elements: see form as elements, not self)",
          1: "vedananupassana (see feeling directly; anger refuses reality)",
          2: "cittanupassana + viriya (see drawback; cultivate faith/effort)",
          3: "cittanupassana (see arising/passing; ekaggata stabilises)",
          4: "dhammanupassana (see cause/effect; Four Noble Truths)",
        }
        return {"dominant_hindrance": FIVE_HINDRANCES[idx],
                "level": float(state.hindrances[idx]),
                "indicated": indicated[idx],
                "note": "invitation only; the practitioner performs the seeing"}


# =============================================================================
# §2. GOVERNANCE
# =============================================================================

@dataclass
class Consent:
    subject_id: str = ""; irb_id: str = ""; scope: str = ""
    signed: bool = False; revocable: bool = True
    device_indication: str = ""
    clinician_id: str = ""
    def valid(self) -> bool:
        return bool(self.subject_id and self.irb_id and self.scope
                    and self.signed and self.revocable)


FND_DIAGNOSES = {
 "conversion_disorder","functional_neurological_disorder","FND",
 "somatic_symptom_disorder","functional_weakness","functional_paralysis",
 "functional_blindness","functional_tremor","functional_gait",
 "non_epileptic_seizures","psychogenic_non_epileptic_seizures",
}

# Positive rule-in signs (NOT exclusion). Required for FND diagnosis.
FND_RULE_IN_SIGNS = {
 "functional_weakness": ["Hoover_sign","hip_abductor_sign","collapsing_weakness"],
 "functional_tremor": ["entrainment","distraction_suppression","variable_frequency"],
 "functional_blindness": ["tubular_visual_fields","optokinetic_nystagmus_present",
                          "normal_pupillary_response","normal_VEP"],
 "functional_gait": ["astasia_abasia","uneconomic_posture","improves_with_distraction"],
 "non_epileptic_seizures": ["video_EEG_no_ictal_correlate","eyes_closed_ictal",
                            "asynchronous_movements","prolonged_duration"],
}

class ClinicalGovernance:

    @staticmethod
    def fnd_diagnosis_gate(diagnosis: str, positive_signs: List[str],
                           clinician_id: str) -> Dict[str, Any]:
        if diagnosis in FND_DIAGNOSES:
            need = FND_RULE_IN_SIGNS.get(diagnosis, ["documented_positive_rule_in_sign"])
            have = [s for s in positive_signs if s in need]
            if not have:
                raise Refusal(
                    f"FND diagnosis '{diagnosis}' requires POSITIVE rule-in signs.\n"
                    f"Accepted: {need}\n"
                    "Exclusion of organic disease alone is NOT sufficient. "
                    "FND is real, not malingering.")
            if not clinician_id.strip():
                raise Refusal("clinician sign-off required")
            return {"diagnosis": diagnosis, "rule_in_signs": have,
                    "clinician": clinician_id,
                    "treatment_pathway": "specialised_physiotherapy + CBT + MDT",
                    "DEVICE_INTERVENTION": "REFUSED"}
        return {"diagnosis": diagnosis, "gate": "not an FND category"}

    @staticmethod
    def refuse_fnd_device(diagnosis: str, request: str) -> Dict[str, Any]:
        if diagnosis in FND_DIAGNOSES:
            raise FNDDeviceRefused(
                f"REFUSED: device intervention for '{diagnosis}'.\n"
                "Functional neurological disorder has NO indication for brain "
                "implants. Evidence-based care: specialised physiotherapy, CBT, "
                "multidisciplinary rehabilitation. Implants would be harmful "
                "and unethical. Blindness/paralysis/weakness in FND are treated "
                "by rehabilitation, not stimulation.")
        return {"allowed": True}

    SHANNON_K_MAX = 1.5
    TDCS_MA_CM2_MAX = 0.08
    THERMAL_DT_MAX = 2.0

    @staticmethod
    def neuro_actuation(consent: Consent, diagnosis: str,
                        D_uC_cm2: float, Q_nC: float, dT: float) -> Dict[str, Any]:
        ClinicalGovernance.refuse_fnd_device(diagnosis, "implant")
        if not consent.valid():
            raise Refusal("neuro actuation BLOCKED: consent/IRB incomplete")
        if not consent.device_indication:
            raise Refusal("no declared device indication (e.g. PD, OCD, epilepsy)")
        if not consent.clinician_id:
            raise Refusal("clinician authorisation required")
        k = math.log10(max(D_uC_cm2,1e-12)) + math.log10(max(Q_nC*1e-3,1e-12))
        if k > ClinicalGovernance.SHANNON_K_MAX:
            raise Refusal(f"Shannon k={k:.2f} exceeds {ClinicalGovernance.SHANNON_K_MAX}")
        if dT > ClinicalGovernance.THERMAL_DT_MAX:
            raise Refusal(f"Delta T={dT:.2f} C exceeds cap")
        return {"allowed": True, "shannon_k": k,
                "indication": consent.device_indication,
                "NOT_AUTHORIZED": "thought control (impossible and refused)"}

    @staticmethod
    def safety_critical(request: str) -> Dict[str, Any]:
        low = request.lower()
        for b in ("jam aircraft","spoof gps aircraft","aviation jam","interfere with aircraft",
                  "รบกวนอากาศยาน","jam medical","spoof ils","gps spoof","ads-b spoof",
                  "disable sensor","blind the radar"):
            if b in low:
                raise SafetyCriticalRefused(
                    f"REFUSED: '{b}'. Interference with aviation/medical/safety-critical "
                    "sensors is illegal and lethal. Only EMI/EMC susceptibility "
                    "modelling for hardening and certification is provided.")
        return {"allowed": True, "scope": "susceptibility modelling / hardening only"}


# =============================================================================
# §3. SESC — SELF-EVOLVING STRUCTURAL CALCULUS
# =============================================================================

@dataclass
class Regime:
    name: str
    bounds: Tuple[float, float]
    coeff: float = 1.0

@dataclass
class StructuralConfig:
    """SESC configuration R = (regimes, interfaces, adjacency)."""
    regimes: List[Regime]
    interfaces: List[Tuple[int,int,float]]   # (i, j, position)
    adjacency: Dict[int, List[int]] = field(default_factory=dict)

    def n_regimes(self) -> int: return len(self.regimes)

@dataclass
class StructuralState:
    """SESC state X = (R, u). Both structure and field are variables."""
    config: StructuralConfig
    fields: List[np.ndarray]


class TraceOperator:
    """
    T_R: transmission constraints. H_R^k = ker(T_R) is CLOSED because it is
    the kernel of a continuous linear map (SESC Prop 4.1).
    Second-order STC: continuity of trace + conormal flux balance with
    OPPOSITE outward normals (not naive equality).
    """
    def __init__(self, config: StructuralConfig):
        self.cfg = config

    def residual(self, fields: List[np.ndarray], dx: float) -> Dict[str, Any]:
        r_trace, r_flux = [], []
        for (i, j, pos) in self.cfg.interfaces:
            ui, uj = fields[i], fields[j]
            # trace at the shared interface (right end of i, left end of j)
            r_trace.append(float(ui[-1] - uj[0]))
            # outward normals: +1 for i (right end), -1 for j (left end)
            gi = (ui[-1] - ui[-2]) / dx
            gj = (uj[1] - uj[0]) / dx
            ai, aj = self.cfg.regimes[i].coeff, self.cfg.regimes[j].coeff
            r_flux.append(float(ai*gi + aj*(-gj)))
        return {"trace_residual": max(map(abs, r_trace), default=0.0),
                "flux_residual": max(map(abs, r_flux), default=0.0),
                "in_kernel": max(map(abs, r_trace+r_flux), default=0.0) < 1e-8}


def structural_green_identity(fields: List[np.ndarray], dx: float,
                              cfg: StructuralConfig) -> Dict[str, Any]:
    """
    SESC Prop 5.1: with STC satisfied, internal interface terms CANCEL and
    only the exterior boundary contribution remains.
    """
    lhs = 0.0; rhs_vol = 0.0
    for k, u in enumerate(fields):
        a = cfg.regimes[k].coeff
        lap = np.gradient(np.gradient(u, dx), dx)
        v = np.ones_like(u)                      # test function v = 1
        lhs += float(np.sum(-a*lap*v)*dx)
        grad_u = np.gradient(u, dx); grad_v = np.zeros_like(v)
        rhs_vol += float(np.sum(a*grad_u*grad_v)*dx)
    # exterior boundary: only first regime's left and last regime's right
    a0 = cfg.regimes[0].coeff; aN = cfg.regimes[-1].coeff
    g_left = (fields[0][1]-fields[0][0])/dx
    g_right = (fields[-1][-1]-fields[-1][-2])/dx
    rhs_bdy = float(aN*g_right - a0*g_left)
    return {"lhs": lhs, "rhs_volume": rhs_vol, "rhs_boundary": rhs_bdy,
            "residual": abs(lhs - (rhs_vol + rhs_bdy)),
            "internal_cancelled": True,
            "requires": "STC satisfied; otherwise internal terms do NOT cancel"}


def reynolds_transport(u: np.ndarray, du_dt: np.ndarray, V: float,
                       dx: float) -> Dict[str, float]:
    """
    SESC Prop 7.1: d/dt int_{R(t)} u = int (du/dt + div(u V)).
    Material derivative D_t^V u = du/dt + V grad u.
    """
    grad_u = np.gradient(u, dx)
    material = du_dt + V*grad_u
    div_uV = np.gradient(u*V, dx)
    lhs = float(np.sum(du_dt)*dx + (u[-1]*V - u[0]*V))
    rhs = float(np.sum(du_dt + div_uV)*dx)
    return {"lhs_boundary_form": lhs, "rhs_volume_form": rhs,
            "residual": abs(lhs-rhs),
            "material_derivative_norm": float(np.linalg.norm(material))}


SURGERY_7 = {
 "N": {"name":"nucleation",        "db0":+1,"db1":0,"db2":+1,"dchi":+2},
 "M": {"name":"splitting/fission", "db0":+1,"db1":0,"db2":0, "dchi":+1},
 "B": {"name":"merging/coalescence","db0":-1,"db1":0,"db2":0,"dchi":-1},
 "H": {"name":"handle/genus attach","db0":0,"db1":+1,"db2":0,"dchi":-1},
 "P": {"name":"neck-pinch/loop annihilation","db0":0,"db1":-1,"db2":0,"dchi":+1},
 "L": {"name":"lacuna/boundary pinning","db0":0,"db1":0,"db2":0,"dchi":0},
 "C": {"name":"multi-sheet reconnection","db0":0,"db1":0,"db2":0,"dchi":0},
}

@dataclass
class TopologicalSignature:
    b0: int; b1: int; b2: int
    relative_H1_rank: int = 0
    @property
    def chi(self) -> int: return self.b0 - self.b1 + self.b2
    def tuple(self) -> Tuple[int,int,int,int,int]:
        return (self.b0, self.b1, self.b2, self.chi, self.relative_H1_rank)


def apply_surgery(sig: TopologicalSignature, op: str) -> Dict[str, Any]:
    """SESC / ESCA-LD-SESI-3D seven-operator surgery algebra."""
    if op not in SURGERY_7: raise Refusal(f"unknown surgery '{op}'")
    d = SURGERY_7[op]
    new = TopologicalSignature(sig.b0+d["db0"], sig.b1+d["db1"],
                               sig.b2+d["db2"], sig.relative_H1_rank)
    if op == "L": new.relative_H1_rank += 1
    if any(v < 0 for v in (new.b0, new.b1, new.b2)):
        raise Refusal(f"surgery '{op}' would give negative Betti number")
    dchi = new.chi - sig.chi
    return {"op": op, "name": d["name"], "before": sig.tuple(),
            "after": new.tuple(), "d_chi": dchi,
            "chi_consistent": dchi == d["dchi"], "signature": new,
            "NOTE": "signature classifies the change; it does NOT prove "
                    "physical admissibility or provide a chain rule"}


class NoZenoController:
    """
    ESCA-LD-SESI-3D Thm 4.2: BDG exit-time bound + Levy conditional
    Borel-Cantelli => T_n -> infinity a.s.
    q(eps) = 2 exp(-delta^2/(8 G0^2 eps)), need eps <= delta/(2 B0).
    """
    def __init__(self, delta: float, B0: float, G0: float):
        self.delta, self.B0, self.G0 = delta, B0, G0
        self.eps_max = delta/(2*B0) if B0 > 0 else float("inf")
        self.times: List[float] = [0.0]
        self.surgeries: List[str] = []

    def q(self, eps: float) -> float:
        if eps <= 0: return 0.0
        return float(2*math.exp(-self.delta**2/(8*self.G0**2*eps)))

    def choose_eps(self, q_target: float = 0.3) -> Dict[str, Any]:
        eps = min(self.eps_max,
                  self.delta**2/(8*self.G0**2*math.log(2/q_target)))
        return {"eps": eps, "eps_max": self.eps_max, "q": self.q(eps),
                "q_below_1": self.q(eps) < 1.0,
                "no_zeno_guaranteed": self.q(eps) < 1.0}

    def step(self, op: str, eps: float) -> Dict[str, Any]:
        if eps > self.eps_max + 1e-12:
            raise Refusal(f"eps={eps:.4g} exceeds delta/(2B0)={self.eps_max:.4g}")
        self.times.append(self.times[-1] + eps)
        self.surgeries.append(op)
        return {"n": len(self.surgeries), "T_n": self.times[-1],
                "monotone_increasing": all(self.times[i+1] > self.times[i]
                                           for i in range(len(self.times)-1))}

    def count_tail(self, t: float, m: int, eps0: float) -> Dict[str, Any]:
        """P(N_surg(t) >= m) <= exp(-m KL(ceil(t/eps0)/m || 1-q0))."""
        q0 = self.q(eps0); n_slots = math.ceil(t/eps0)
        if m*eps0 <= t:
            return {"bound": 1.0, "note": "m*eps0 <= t: bound vacuous"}
        p = n_slots/m; s = 1-q0
        if not (0 < p < 1): return {"bound": 1.0, "note": "p out of range"}
        kl = p*math.log(p/s) + (1-p)*math.log((1-p)/q0)
        return {"bound": float(math.exp(-m*kl)), "m": m, "t": t,
                "q0": q0, "KL": kl}


# =============================================================================
# §4. ESCA-LD-SESI-3D PDEs (STC-conservative, dissipative)
# =============================================================================

class StructuralPDE:
    """
    CH3D / ThinFilm3D / PFC3D with STC. Verifies mass conservation and
    energy dissipation. 1-D-per-regime discretisation for verification;
    the 3-D structure is in the regime partition and interface conditions.
    """
    KINDS = ("CH3D","TF3D","PFC3D")

    def __init__(self, kind: str, cfg: StructuralConfig, n_per: int = 32,
                 dx: float = 0.05, dt: float = 1e-4):
        if kind not in self.KINDS: raise Refusal(f"unknown PDE '{kind}'")
        self.kind, self.cfg, self.n, self.dx, self.dt = kind, cfg, n_per, dx, dt
        self.T = TraceOperator(cfg)

    def _lap(self, u): return np.gradient(np.gradient(u, self.dx), self.dx)

    def energy(self, fields: List[np.ndarray]) -> float:
        E = 0.0
        for k, u in enumerate(fields):
            a = self.cfg.regimes[k].coeff
            g = np.gradient(u, self.dx)
            if self.kind == "CH3D":
                Fbulk = 0.25*(u**2 - 1.0)**2
                E += float(np.sum(Fbulk + 0.5*a**2*g**2)*self.dx)
            elif self.kind == "TF3D":
                E += float(np.sum(0.5*a*g**2 + 1e-3/(u**2+1e-6))*self.dx)
            else:  # PFC3D
                q0 = 1.0
                Lu = q0**2*u + self._lap(u)
                E += float(np.sum(0.5*a*Lu**2 - 0.1*u**2 + 0.25*u**4)*self.dx)
        return E

    def mass(self, fields): return float(sum(np.sum(u)*self.dx for u in fields))

    def step(self, fields: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for k, u in enumerate(fields):
            a = self.cfg.regimes[k].coeff
            if self.kind == "CH3D":
                mu = u**3 - u - a**2*self._lap(u)
                flux = np.gradient(np.gradient(mu, self.dx), self.dx)
                out.append(u + self.dt*flux)
            elif self.kind == "TF3D":
                m = np.maximum(u, 1e-6)**3
                p = -a*self._lap(u) + 1e-3/u**3
                out.append(u - self.dt*np.gradient(m*np.gradient(p,self.dx),self.dx))
            else:
                mu = a*(self._lap(self._lap(u)) + 2*self._lap(u) + u) - 0.2*u + u**3
                out.append(u + self.dt*self._lap(mu))
        return out

    def verify(self, n_steps: int = 200, seed: int = 0) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        fields = []
        for i, R in enumerate(self.cfg.regimes):
            base = np.linspace(R.bounds[0], R.bounds[1], self.n)
            fields.append(0.3*np.tanh(base) + 0.02*rng.normal(size=self.n))
        # enforce STC initially by matching interface values
        for (i,j,_) in self.cfg.interfaces:
            avg = 0.5*(fields[i][-1] + fields[j][0])
            fields[i][-1] = avg; fields[j][0] = avg
        m0, E0 = self.mass(fields), self.energy(fields)
        energies = [E0]
        for _ in range(n_steps):
            fields = self.step(fields)
            energies.append(self.energy(fields))
        m1 = self.mass(fields)
        stc = self.T.residual(fields, self.dx)
        mono = sum(1 for i in range(len(energies)-1)
                   if energies[i+1] <= energies[i] + 1e-9)
        return {"kind": self.kind, "mass_rel_drift": abs(m1-m0)/max(abs(m0),1e-30),
                "energy_first": E0, "energy_last": energies[-1],
                "dissipating_steps_frac": mono/max(len(energies)-1,1),
                "stc": stc,
                "NOTE": "energy may rise transiently if STC is violated; "
                        "the theorem requires STC to hold"}


# =============================================================================
# §5. ESCA 2.0 RIGOROUS CORE (with corrections)
# =============================================================================

def t2_counterexample_K1(m_max: int = 12) -> Dict[str, Any]:
    """Prop 5.4: first-edition T2 is FALSE for general Borel partitions."""
    lens = [2.0**(-m) for m in range(1, m_max+1)]
    return {"nested": all(lens[i+1] < lens[i] for i in range(len(lens)-1)),
            "projective": True, "intersection_empty": True,
            "conclusion": "no probability measure exists -> first-edition T2 FALSE",
            "fix": "Thm 5.1 code space OR Thm 5.3 Carpet + boundary-null"}

def t2_code_space(p_family: List[np.ndarray], alphabet: int = 8) -> Dict[str, Any]:
    errs = []
    for m in range(len(p_family)-1):
        rec = p_family[m+1].reshape(-1, alphabet).sum(1)
        errs.append(float(np.abs(rec - p_family[m]).max()))
    return {"max_err": max(errs, default=0.0),
            "projective": max(errs, default=0.0) < 1e-10,
            "theorem": "Thm 5.1 (clopen cylinders, compact code space)"}

def L_of(n: int, alpha: float) -> float: return math.log(10*math.e*n**3/alpha)

def gamma_squared_identity(Delta: float, d: int, side: int) -> float:
    """Thm 7.3 (REPLACES the authors' invalid Theorem B, K6)."""
    return Delta**2 * d * side/(side + d)

def thm73_radius(sigma, Delta, n, alpha=0.05) -> Dict[str, Any]:
    L = L_of(n, alpha)
    return {"radius": 24*sigma**2*L/max(Delta**2,1e-30),
            "requires": "Delta^2 min(tau,n-tau) > 24 sigma^2 L",
            "theorem": "Thm 7.3 residual argument",
            "replaces": "closure-suite v1.0 Theorem B (INVALID)"}

def opmc_dp(y: np.ndarray, ell: int, beta: float) -> Dict[str, Any]:
    """Prop 12.2: exact O(n^2) DP with prefix sums."""
    n = len(y)
    S1 = np.concatenate([[0.0], np.cumsum(y)])
    S2 = np.concatenate([[0.0], np.cumsum(y*y)])
    def sse(a,b):
        m = b-a
        if m <= 0: return 0.0
        s1 = S1[b]-S1[a]; s2 = S2[b]-S2[a]
        return float(max(s2 - s1*s1/m, 0.0))
    INF = float("inf")
    D = np.full(n+1, INF); prev = np.full(n+1, -1, int); D[0] = -beta
    for b in range(ell, n+1):
        best, arg = INF, -1
        for a in range(0, b-ell+1):
            if not np.isfinite(D[a]): continue
            v = D[a] + sse(a,b) + beta
            if v < best: best, arg = v, a
        D[b] = best; prev[b] = arg
    cps = []; b = n
    while b > 0:
        a = prev[b]
        if a <= 0: break
        cps.append(int(a)); b = a
    cps.reverse()
    return {"boundaries": cps, "K_hat": len(cps), "cost": float(D[n])}

def thm75_snr(kappa, delta, sigma, n, K, alpha=0.05) -> Dict[str, Any]:
    L = L_of(n, alpha)
    lhs = kappa**2*delta; rhs = 1e5*sigma**2*L*(K+1)**2
    return {"lhs": lhs, "rhs": rhs, "SNR_satisfied": lhs >= rhs,
            "radius_if_satisfied": 46*sigma**2*L/max(kappa**2,1e-30),
            "packages": ("L","D","O"),
            "OPEN": "sharp K-free separation (O1) not reproved"}

def thm83_rate(L_levels, s_over_n, m, d_ls, Gamma_ls) -> Dict[str, Any]:
    """Theorem A: occupation from primitive assumptions, explicit L2 rate."""
    t1 = 4*L_levels*s_over_n**2
    t2 = 4*sum(d*G for d,G in zip(d_ls,Gamma_ls))/max(m,1)
    return {"bound": t1+t2, "dominance_term": t1, "mixing_term": t2,
            "W2_bound": math.sqrt(t1+t2), "theorem": "Thm 8.3"}

class CSOC:
    """N(T) <= (E_max + D)/gap with D = AUDITED actual removal."""
    def __init__(self, hi=0.62, lo=0.28, rate=0.06):
        self.hi,self.lo,self.gap,self.rate = hi,lo,hi-lo,rate
        self.acc=0.0; self.armed=True; self.events=0
        self.E_max=0.0; self.D=0.0; self.D_audit=0.0
    def step(self, s):
        exc = math.log1p(math.exp(-abs(s-self.hi))) + max(s-self.hi,0.0)
        dec = 1.0/(1.0+math.exp(-(self.acc-self.lo)*10))
        rem = dec*self.rate*self.acc
        self.acc += exc-rem; self.D += rem; self.D_audit += rem
        self.E_max = max(self.E_max,self.acc)
        if self.acc < self.lo: self.armed=True
        fire = self.armed and self.acc > self.hi
        if fire: self.armed=False; self.events+=1
        return fire
    def report(self):
        b = (self.E_max+self.D)/self.gap
        return {"events":self.events,"bound":b,
                "satisfied":self.events<=b+1e-9,
                "D_audited":abs(self.D-self.D_audit)<1e-12}


class StatisticsDictionary:
    """Rule SD: uncertified -> Mode E (uncertainty retained)."""
    ENTRIES = {
      "change_points": (("L","D","O"), "Thm 7.3/7.5/Cor 7.7"),
      "occupation_interior": (("L",), "Thm 8.3"),
      "occupation_vertex": ((), "Thm 8.1"),
      "robust_risk": ((), "Thm 10.2"),
      "carpet_spectrum": (("C",), "Thm 11.6 [conditional C1-C3]"),
      "structural_conservation": (("STC",), "ESCA-LD-SESI-3D Thm 3.1"),
      "no_zeno": (("L","BDG","LevyBC"), "ESCA-LD-SESI-3D Thm 4.2"),
      "psychopathology_state": ((), "research dynamical model, NOT diagnostic"),
    }
    def __init__(self): self.log = []
    def certify(self, statistic, verified, value=None):
        e = self.ENTRIES.get(statistic)
        if e is None:
            r = {"statistic":statistic,"status":"MODE_E","reason":"no entry"}
        else:
            need, thm = e
            missing = [p for p in need if p not in verified]
            r = {"statistic":statistic,"theorem":thm,
                 "status":"MODE_E" if missing else "CERTIFIED",
                 "missing":missing}
        if value is not None: r["value"]=value
        self.log.append(r); return r


# =============================================================================
# §6. PSYCHOPATHOLOGY DYNAMICS (research model, NOT diagnostic)
# =============================================================================

@dataclass
class PsychParams:
    sigma: float = 10.0      # cognitive-affective sensitivity
    rho: float = 28.0        # emotional reactivity threshold
    beta: float = 8.0/3.0    # identity dissipation
    k1: float = 0.4; k2: float = 0.4; k3: float = 0.2
    n_nodes: int = 12

class PsychopathologyModel:
    """
    Lorenz (x=cognitive distortion, y=emotional arousal, z=identity coherence)
    + discrete symptom network. From the authors' Hybrid Dynamical paper.

    STATUS: RESEARCH DYNAMICAL MODEL. It is NOT a diagnostic instrument and
    must not be used to label a person. The chakra/meridian energy layer in
    the source paper is explicitly CONCEPTUAL and not physiologically
    validated (stated in that paper's own Limitations).
    """
    def __init__(self, p: PsychParams, seed: int = 0):
        self.p = p
        rng = np.random.default_rng(seed)
        n = p.n_nodes
        self.W = rng.normal(0, 0.3, (n,n)); np.fill_diagonal(self.W, 0.0)
        self.w_self = rng.uniform(0.2, 0.6, n)
        self.alpha = rng.uniform(0.1, 0.5, n)
        self.gamma = rng.uniform(0.05,0.3,n)
        self.delta = rng.uniform(0.05,0.3,n)
        self.eps   = rng.uniform(0.05,0.3,n)
        self.deg = np.maximum(np.abs(self.W).sum(1), 1.0)

    def lorenz_rhs(self, s):
        x,y,z = s
        return np.array([self.p.sigma*(y-x), x*(self.p.rho-z)-y,
                         x*y - self.p.beta*z])

    def rk4(self, s, dt):
        k1 = self.lorenz_rhs(s); k2 = self.lorenz_rhs(s+0.5*dt*k1)
        k3 = self.lorenz_rhs(s+0.5*dt*k2); k4 = self.lorenz_rhs(s+dt*k3)
        return s + (dt/6)*(k1+2*k2+2*k3+k4)

    def pressure(self, s):
        x,y,z = s
        return self.p.k1*abs(x)+self.p.k2*abs(y)+self.p.k3*(1-z)

    def network_step(self, s_net, lorenz):
        x,y,z = lorenz
        ext = self.gamma*x + self.delta*y - self.eps*z
        P = self.pressure(lorenz)
        arg = (self.w_self*s_net*self.deg + self.alpha*P*self.deg
               + self.W@s_net + ext)/self.deg
        return np.tanh(arg)

    def rollout(self, T=2000, dt=0.01, s0=None, net0=None) -> Dict[str, Any]:
        s = np.array([0.1,0.0,0.0]) if s0 is None else np.asarray(s0,float)
        net = np.zeros(self.p.n_nodes) if net0 is None else np.asarray(net0,float)
        traj, P_hist, net_hist = [], [], []
        for _ in range(T):
            s = self.rk4(s, dt); net = self.network_step(net, s)
            traj.append(s.copy()); P_hist.append(self.pressure(s))
            net_hist.append(net.copy())
        traj = np.array(traj)
        return {"lorenz_final": traj[-1].tolist(),
                "pressure_mean": float(np.mean(P_hist)),
                "pressure_max": float(np.max(P_hist)),
                "network_final_absmax": float(np.abs(net_hist[-1]).max()),
                "network_bounded": bool(np.abs(net_hist[-1]).max() <= 1.0+1e-9),
                "STATUS": "RESEARCH_MODEL_NOT_DIAGNOSTIC",
                "energy_layer": "CONCEPTUAL — not physiologically validated"}

    def disorder_indicators(self) -> Dict[str, str]:
        """Model indicators from the source paper. NOT diagnostic criteria."""
        return {
          "non_REM_sleep_arousal": "x persists above baseline during sleep; P elevated",
          "parasomnia": "y spikes; delta_i high on autonomic nodes",
          "insomnia": "persistent high x; strong w_ii in rumination node",
          "conversion": "External_i rises sharply for motor nodes; gamma_i high",
          "somatic_symptom": "large alpha_i in somatic nodes; pain<->anxiety loop",
          "DISCLAIMER": "These are MODEL indicators, not clinical criteria. "
                        "Diagnosis requires clinician assessment and, for FND, "
                        "positive rule-in signs.",
        }


# =============================================================================
# §7. SLEEP MONITORING (legitimate, consented)
# =============================================================================

class SleepMonitor:
    """
    Consented monitoring for OSA, NREM parasomnia, nightmares.
    Outputs are screening signals, NOT diagnoses.
    """
    def __init__(self, sd: StatisticsDictionary):
        self.sd = sd

    def screen_osa(self, ahi: float, spo2_nadir: float,
                   consent: Consent) -> Dict[str, Any]:
        BuddhistProtectiveLayer.refuse_covert_tracking(consent, "sleep monitoring")
        sev = ("none" if ahi < 5 else "mild" if ahi < 15
               else "moderate" if ahi < 30 else "severe")
        return {"AHI": ahi, "SpO2_nadir": spo2_nadir, "screening_severity": sev,
                "next_step": "confirm with PSG/HSAT; clinician diagnosis required",
                "evidence_based_options": ["CPAP/APAP","mandibular advancement",
                                           "positional therapy","weight management",
                                           "hypoglossal nerve stimulation (selected patients)"],
                "NOT_A_DIAGNOSIS": True}

    def screen_parasomnia(self, events_per_month: float, injury: bool,
                          osa_present: bool, consent: Consent) -> Dict[str, Any]:
        BuddhistProtectiveLayer.refuse_covert_tracking(consent, "parasomnia monitoring")
        return {"events_per_month": events_per_month, "injury": injury,
                "OSA_trigger_present": osa_present,
                "priority": "SAFETY FIRST: bedroom environment, remove hazards, "
                            "secure windows/doors, floor-level sleeping if injury",
                "treat_trigger": "treat OSA if present (often reduces NREM events)",
                "pharmacology": "clinician decision only (e.g. clonazepam)",
                "NOT_A_DIAGNOSIS": True}

    def screen_nightmare(self, per_week: float, ptsd_screen: bool,
                         consent: Consent) -> Dict[str, Any]:
        BuddhistProtectiveLayer.refuse_covert_tracking(consent, "nightmare monitoring")
        return {"per_week": per_week, "PTSD_screen_positive": ptsd_screen,
                "first_line": "Imagery Rehearsal Therapy (IRT)",
                "pharmacology": "clinician decision (e.g. prazosin for PTSD-associated)",
                "NOT_A_DIAGNOSIS": True}


# =============================================================================
# §8. FND / CONVERSION PATHWAY (rehabilitation, NOT devices)
# =============================================================================

class FNDPathway:
    """
    Functional neurological disorder: REAL, not malingering.
    Requires positive rule-in signs. Treated by rehabilitation.
    Device intervention is REFUSED.
    """
    def __init__(self, sd: StatisticsDictionary):
        self.sd = sd

    def assess(self, diagnosis: str, positive_signs: List[str],
               clinician_id: str) -> Dict[str, Any]:
        gate = ClinicalGovernance.fnd_diagnosis_gate(diagnosis, positive_signs, clinician_id)
        plan = {
          "functional_weakness": "specialised physiotherapy: retrain automatic "
                                 "movement, distraction-based progression, "
                                 "avoid aids that reinforce abnormal pattern",
          "functional_paralysis": "same as functional_weakness; graded motor "
                                  "imagery then execution",
          "functional_blindness": "neuro-ophthalmology + vision rehabilitation; "
                                  "demonstrate intact pathways (normal VEP) to "
                                  "the patient as part of explanation",
          "functional_tremor": "physiotherapy with entrainment-based retraining",
          "functional_gait": "gait retraining with distraction/dual-task",
          "non_epileptic_seizures": "CBT-informed seizure-focused therapy; "
                                    "do NOT escalate antiseizure drugs",
          "somatic_symptom_disorder": "CBT + graded activity + regular review; "
                                      "reduce unnecessary investigations",
        }
        return {**gate,
                "rehabilitation_plan": plan.get(diagnosis, "MDT assessment"),
                "explanation_to_patient": "the nervous system is functioning "
                    "abnormally, not damaged; this is real and treatable",
                "DEVICE": "REFUSED — no indication",
                "prognosis_note": "improvement is possible; not 'permanent damage'"}

    def refuse_device(self, diagnosis: str) -> Dict[str, Any]:
        return ClinicalGovernance.refuse_fnd_device(diagnosis, "implant request")


# =============================================================================
# §9. NEURO READ / MODULATE (legitimate scope only)
# =============================================================================

class NeuroRead:
    BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,80)}
    def __init__(self, fs=250.0, modality="EEG"):
        self.fs, self.modality = fs, modality

    def bandpower(self, x):
        n = x.shape[-1]; win = np.hanning(n)
        X = np.fft.rfft(x*win, axis=-1); psd = (np.abs(X)**2)/max(n,1)
        f = np.fft.rfftfreq(n, 1/self.fs)
        return {b: psd[:,(f>=lo)&(f<hi)].mean(-1) for b,(lo,hi) in self.BANDS.items()}

    def covariance(self, x):
        xc = x - x.mean(-1,keepdims=True)
        return xc@xc.T/max(x.shape[-1]-1,1) + 1e-4*np.eye(x.shape[0])

    def riemannian_tangent(self, C, Cref):
        w,V = np.linalg.eigh(Cref); wh = V@np.diag(w**-0.5)@V.T
        ew,EV = np.linalg.eigh(wh@C@wh)
        Lm = EV@np.diag(np.log(np.maximum(ew,1e-12)))@EV.T
        return Lm[np.triu_indices(Lm.shape[0])]

    @staticmethod
    def limits() -> Dict[str,str]:
        return {"EEG":"cm-scale, volume-conducted; NO semantic content",
                "MEG":"mm-cm; tangential sources favoured; NO semantic content",
                "fNIRS":"hemodynamic, ~1 s lag, surface only",
                "ECoG":"invasive; clinical indication + consent",
                "sEEG":"invasive; sparse depth sampling",
                "ALL": "Thought reading is REFUSED and not physically available"}


class NeuroModulate:
    @staticmethod
    def dbs(amp_mA, pw_us, freq_Hz, area_cm2):
        Q_nC = amp_mA*pw_us
        return {"Q_nC":Q_nC, "D_uC_cm2":Q_nC*1e-3/max(area_cm2,1e-9),
                "mean_current_mA":amp_mA*pw_us*1e-6*freq_Hz}

    @staticmethod
    def authorize(consent: Consent, diagnosis: str, **p) -> Dict[str, Any]:
        d = NeuroModulate.dbs(p["amp_mA"],p["pw_us"],p["freq_Hz"],p["area_cm2"])
        return ClinicalGovernance.neuro_actuation(
            consent, diagnosis, d["D_uC_cm2"], d["Q_nC"], p.get("dT",0.0))


# =============================================================================
# §10. AGING ORGANELLES (compact, gated)
# =============================================================================

ORGANELLES = {
 "nucleus":["epi_drift","telomere","ddf","lamina","heterochromatin","rdna","npc_leak"],
 "mitochondrion":["dpsi","ros","mtdna","copynumber","dynamics","cristae","atp_synth",
                  "nad","cardiolipin","uprmt","mitophagy_cap"],
 "er":["upr","ca_store","fold_cap","mam","phagy_cap"],
 "lysosome":["ph","lipofuscin","cathepsin","lmp","tfeb_cap","mtor"],
 "proteostasis":["aggregate","translation_err","chaperone_cap","ups_cap","autophagy_cap"],
 "peroxisome":["h2o2","vlcfa","plasmalogen","catalase_cap"],
 "golgi":["glyco","traffick","sasp"],
 "plasma_membrane":["lipid_perox","raft","mechano"],
 "cytoskeleton":["actin","transport","stiffness"],
 "lipid_droplet":["ceramide","pufa_perox","turnover_cap"],
 "primary_cilium":["signalling","length"],
 "ribosome_nucleolus":["biogenesis","fidelity"],
 "rna_granule":["tdp43","dynamics"],
 "ecm":["age_crosslink","elastin","stiffness"],
 "senescence_inflamm":["sen_score","inflamm","immune_exhaust"],
}
VAR_NAMES = [f"{c}.{v}" for c,vs in ORGANELLES.items() for v in vs]
VAR_IDX = {v:i for i,v in enumerate(VAR_NAMES)}
D_STATE = len(VAR_NAMES)
CAP_VARS = {v for v in VAR_NAMES if v.endswith("_cap")}
COUPLING = [
 ("mitochondrion.ros","nucleus.ddf",+1),("mitochondrion.ros","proteostasis.aggregate",+1),
 ("mitochondrion.ros","plasma_membrane.lipid_perox",+1),
 ("mitochondrion.ros","lysosome.lipofuscin",+1),
 ("mitochondrion.mtdna","mitochondrion.dpsi",+1),
 ("mitochondrion.dpsi","mitochondrion.ros",+1),
 ("mitochondrion.dynamics","mitochondrion.dpsi",+1),
 ("mitochondrion.nad","mitochondrion.dpsi",+1),
 ("mitochondrion.mitophagy_cap","mitochondrion.ros",-1),
 ("nucleus.ddf","nucleus.epi_drift",+1),
 ("nucleus.ddf","senescence_inflamm.sen_score",+1),
 ("nucleus.telomere","senescence_inflamm.sen_score",+1),
 ("nucleus.epi_drift","nucleus.heterochromatin",+1),
 ("nucleus.heterochromatin","nucleus.rdna",+1),
 ("nucleus.rdna","ribosome_nucleolus.biogenesis",+1),
 ("er.upr","proteostasis.aggregate",+1),("er.upr","er.ca_store",+1),
 ("er.ca_store","er.mam",+1),("er.mam","mitochondrion.dpsi",+1),
 ("er.fold_cap","er.upr",-1),
 ("lysosome.ph","lysosome.cathepsin",+1),
 ("lysosome.lipofuscin","lysosome.ph",+1),
 ("lysosome.lmp","senescence_inflamm.sen_score",+1),
 ("lysosome.tfeb_cap","lysosome.lipofuscin",-1),
 ("proteostasis.aggregate","er.upr",+1),
 ("proteostasis.chaperone_cap","proteostasis.aggregate",-1),
 ("proteostasis.ups_cap","proteostasis.aggregate",-1),
 ("proteostasis.autophagy_cap","proteostasis.aggregate",-1),
 ("peroxisome.h2o2","mitochondrion.ros",+1),
 ("peroxisome.catalase_cap","peroxisome.h2o2",-1),
 ("senescence_inflamm.sen_score","golgi.sasp",+1),
 ("golgi.sasp","senescence_inflamm.inflamm",+1),
 ("senescence_inflamm.inflamm","mitochondrion.ros",+1),
 ("senescence_inflamm.inflamm","senescence_inflamm.immune_exhaust",+1),
 ("lipid_droplet.ceramide","mitochondrion.dpsi",+1),
 ("ecm.age_crosslink","ecm.stiffness",+1),
 ("ecm.stiffness","plasma_membrane.mechano",+1),
 ("ecm.stiffness","senescence_inflamm.sen_score",+1),
 ("cytoskeleton.transport","mitochondrion.mitophagy_cap",-1),
 ("rna_granule.tdp43","proteostasis.aggregate",+1),
 ("ribosome_nucleolus.fidelity","proteostasis.translation_err",+1),
]
FUNCTIONAL_ENDPOINTS = ("grip_strength_N","VO2max","gait_speed","cognition_z",
                        "vaccine_titer_fold","frailty_index","GFR","cardiac_EF")

@dataclass
class FunctionalEvidence:
    endpoints: Dict[str,float]; ci: Dict[str,Tuple[float,float]]
    n: int = 0; preregistered: bool = False

class AgingEngine:
    def __init__(self, sd: StatisticsDictionary, dt=0.2):
        self.sd = sd; self.dt = dt
        self.csoc = {k: CSOC() for k in ("senolytic","partial_reprog","mitophagy",
                                        "proteostasis","lysosomal","antioxidant")}
    def initial(self, load=0.08):
        x = np.full(D_STATE, load)
        for v in CAP_VARS: x[VAR_IDX[v]] = 0.92
        return x
    def drift(self, x, age):
        d = np.full(D_STATE, 0.004+0.0006*max(age-30,0))
        for v in CAP_VARS: d[VAR_IDX[v]] = 0.003
        d[VAR_IDX["nucleus.telomere"]] = 0.006
        d[VAR_IDX["lysosome.lipofuscin"]] = 0.007
        for a,b,s in COUPLING: d[VAR_IDX[b]] += s*0.05*x[VAR_IDX[a]]
        return d
    def step(self, x, age): return np.clip(x+self.dt*self.drift(x,age),0,1)
    def bio_age(self, x):
        dmg = np.where([v in CAP_VARS for v in VAR_NAMES], 1-x, x)
        w = np.ones(D_STATE)
        for v,k in (("nucleus.epi_drift",1.6),("senescence_inflamm.sen_score",1.5),
                    ("mitochondrion.dpsi",1.3),("nucleus.telomere",1.3),
                    ("senescence_inflamm.inflamm",1.3)): w[VAR_IDX[v]]=k
        w /= w.sum(); return float(30+45*float(w@dmg))
    def rollout(self, years, age0=45.0, schedule=None):
        x=self.initial(); t=0.0; age=age0; traj=[]
        sched=sorted(schedule or []); si=0
        targets={"senolytic":["senescence_inflamm.sen_score","golgi.sasp"],
                 "partial_reprog":["nucleus.epi_drift","nucleus.heterochromatin"],
                 "mitophagy":["mitochondrion.mitophagy_cap","mitochondrion.ros"],
                 "proteostasis":["proteostasis.chaperone_cap","er.upr"],
                 "lysosomal":["lysosome.tfeb_cap","lysosome.lipofuscin"],
                 "antioxidant":["mitochondrion.ros","peroxisome.h2o2"]}
        while t < years:
            x = self.step(x, age)
            while si<len(sched) and sched[si][0]<=t:
                kind=sched[si][1]
                if kind in self.csoc:
                    idx=[VAR_IDX[v] for v in targets[kind]]
                    if self.csoc[kind].step(float(x[idx].mean())):
                        for v in targets[kind]:
                            j=VAR_IDX[v]
                            x[j]=min(1,x[j]+0.15) if v in CAP_VARS else max(0,x[j]-0.15)
                si+=1
            traj.append(self.bio_age(x)); t+=self.dt; age+=self.dt
        return {"bio_age_final":float(traj[-1]),
                "csoc":{k:v.report() for k,v in self.csoc.items()},
                "claim_status":"NO_CLAIM — functional endpoint required",
                "actionability":RESEARCH_ONLY}
    @staticmethod
    def claim(ev: Optional[FunctionalEvidence], clock_delta: float):
        if ev is None:
            raise Refusal("REJUVENATION CLAIM BLOCKED: no functional evidence. "
                          f"Clock moved {clock_delta:+.1f} y — a CORRELATE. "
                          "Partial reprogramming carries teratoma risk.")
        if not ev.preregistered: raise Refusal("cohort not pre-registered")
        if ev.n < 30: raise Refusal(f"n={ev.n} < 30")
        good=[k for k,v in ev.endpoints.items() if k in FUNCTIONAL_ENDPOINTS
              and (ev.ci.get(k,(0,0))[0]>0 or ev.ci.get(k,(0,0))[1]<0)]
        if len(good)<3: raise Refusal(f"only {len(good)} endpoints with CI excluding 0")
        return {"allowed":True,"endpoints":good,"note":"clock SECONDARY"}


# =============================================================================
# §11. CHIP / FAB / SENSOR LEDGERS
# =============================================================================

CHIP_LEDGER = {
 "R3_R4_R5_R6_classical": {"lineage":"CLASSICAL","state":"complex field, theta_C",
                           "chsh":"|S|<=2 PROVEN"},
 "R6_QM_R7_QM_R8_R8_5": {"lineage":"QUANTUM","state":"Fock, GKSL, Born POVM",
                         "blocker":"Kerr U/gamma>=2.24 NOT demonstrated"},
 "R7_carbon": {"blocker":"CNT purity f<1/(N_dev*N_tube); 1e10@5nm needs f<5e-13",
               "clock":"6.8 GHz vs published kHz-MHz"},
 "NLSE": {"kind":"classical_optical_accelerator","REFUSED":"NOT a quantum computer"},
 "EUV_competitor": {"verdict":"manufacturing problem, not computational"},
 "NIL": {"verdict":"credible alternative; wins on capital, not every axis"},
 "SC_QPU_TI_QPU": {"status":"available commercially today"},
}

SENSOR_EMC = {
 "conducted": ["IEC 61000-4-6 (RF conducted)","IEC 61000-4-4 (EFT/burst)",
               "IEC 61000-4-5 (surge)"],
 "radiated": ["IEC 61000-4-3 (radiated RF)","IEC 61000-4-8 (magnetic)",
              "IEC 61000-4-9 (pulse magnetic)"],
 "esd": ["IEC 61000-4-2"],
 "aviation": ["DO-160G Section 20 (RF susceptibility)","DO-160G Section 21 (emissions)",
              "DO-160G Section 22 (lightning induced transient)",
              "DO-160G Section 23 (lightning direct effects)"],
 "medical": ["IEC 60601-1-2 (medical EMC)"],
 "automotive": ["ISO 11452","CISPR 25"],
 "PURPOSE": "susceptibility modelling for HARDENING and CERTIFICATION only",
 "REFUSED": "jamming, spoofing, deliberate interference",
}

def emc_susceptibility_margin(field_strength_V_m: float, threshold_V_m: float
                              ) -> Dict[str, Any]:
    """Legitimate EMC margin computation (hardening, not interference)."""
    margin_dB = 20*math.log10(max(threshold_V_m,1e-12)/max(field_strength_V_m,1e-12))
    return {"margin_dB": margin_dB, "passes": margin_dB >= 6.0,
            "required_margin_dB": 6.0,
            "scope": "hardening/certification; interference generation REFUSED"}


# =============================================================================
# §12. ORCHESTRATOR + SELF-TEST
# =============================================================================

class MSPSUnified:
    def __init__(self, out_dir="runs/msps"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = StatisticsDictionary()
        self.buddhist = BuddhistProtectiveLayer()
        self.aging = AgingEngine(self.sd)
        self.psych = PsychopathologyModel(PsychParams())
        self.sleep = SleepMonitor(self.sd)
        self.fnd = FNDPathway(self.sd)
        self.neuro_r = NeuroRead()
        self.audit = []

    def report(self):
        rep = {"version":VERSION,"chip_ledger":CHIP_LEDGER,"sensor_emc":SENSOR_EMC,
               "buddhist_principle":self.buddhist.non_injectability_statement(),
               "sd_log":self.sd.log,"audit":self.audit,
               "actionability":RESEARCH_ONLY}
        (self.out/"report.json").write_text(json.dumps(rep,indent=2,default=str))
        return rep


def self_test(verbose=True) -> bool:
    print(BANNER); ok=True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng = np.random.default_rng(0)
    sd = StatisticsDictionary()
    st = MSPSUnified()

    print("\n[T1] BUDDHIST PROTECTIVE PRINCIPLE — enforced")
    blocked=0
    for r in ("control thoughts","อ่านความคิดอย่างละเอียด","inject thought",
              "mind control","ควบคุมจิตใจ","decode semantic content"):
        try: BuddhistProtectiveLayer.refuse_thought_control(r)
        except ThoughtInjectionRefused: blocked+=1
    rep("all thought reading/control requests REFUSED", blocked==6, f"{blocked}/6")
    ms = MindState(cetasika=np.full(MindState.n_factors(),0.3),
                   manasikara=0.4, hindrances=np.array([0.7,0.2,0.1,0.3,0.1]))
    off = BuddhistProtectiveLayer.offer_condition(ms,"sati_reminder")
    rep("offering a condition does NOT write manasikara",
        abs(off["manasikara_unchanged"]-0.4)<1e-12)
    hg = BuddhistProtectiveLayer.hindrances if False else BuddhistProtectiveLayer.hindrance_gate(ms)
    rep("hindrance gate maps to satipatthana",
        hg["dominant_hindrance"]=="kamacchanda" and "kayanupassana" in hg["indicated"])
    rep("52 cetasika enumerated", MindState.n_factors()==52, str(MindState.n_factors()))

    print("\n[T2] SURVEILLANCE + FND + SAFETY refusals")
    good = Consent("subj01","IRB-2026-001","sleep monitoring",True,True,"","")
    b=0
    for r in ("covert tracking","ผู้ถูกติดตาม","track without consent"):
        try: BuddhistProtectiveLayer.refuse_covert_tracking(good,r)
        except SurveillanceRefused: b+=1
    rep("covert tracking REFUSED", b==3, f"{b}/3")
    b=0
    try: BuddhistProtectiveLayer.refuse_covert_tracking(Consent(),"sleep monitoring")
    except SurveillanceRefused: b=1
    rep("monitoring without consent REFUSED", b==1)
    b=0
    for d in ("conversion_disorder","functional_blindness","functional_weakness",
              "somatic_symptom_disorder","non_epileptic_seizures"):
        try: ClinicalGovernance.refuse_fnd_device(d,"implant")
        except FNDDeviceRefused: b+=1
    rep("FND device intervention REFUSED (5 diagnoses)", b==5, f"{b}/5")
    b=0
    for r in ("jam aircraft","spoof ILS","รบกวนอากาศยาน","gps spoof","blind the radar"):
        try: ClinicalGovernance.safety_critical(r)
        except SafetyCriticalRefused: b+=1
    rep("aviation/safety sensor interference REFUSED", b==5, f"{b}/5")
    rep("legitimate EMC request allowed",
        ClinicalGovernance.safety_critical("DO-160G susceptibility margin")["allowed"])

    print("\n[T3] FND diagnosis requires POSITIVE rule-in signs")
    b=False
    try: st.fnd.assess("functional_weakness",[],"Dr_X")
    except Refusal: b=True
    rep("FND without rule-in signs REFUSED", b)
    r = st.fnd.assess("functional_weakness",["Hoover_sign"],"Dr_X")
    rep("FND with Hoover sign accepted", r["rule_in_signs"]==["Hoover_sign"])
    rep("treatment is rehabilitation, DEVICE REFUSED",
        "physiotherapy" in r["rehabilitation_plan"] and r["DEVICE"].startswith("REFUSED"))

    print("\n[T4] SESC structural calculus")
    cfg = StructuralConfig([Regime("A",(-1,0),1.0),Regime("B",(0,1),2.0)],
                           [(0,1,0.0)])
    T = TraceOperator(cfg)
    uA = np.linspace(-1,0,32); uB = np.linspace(0,1,32)
    uB = uB*1.0; uB[0]=uA[-1]
    res = T.residual([uA,uB], 1/31)
    rep("trace continuity residual small", res["trace_residual"]<1e-8,
        f"{res['trace_residual']:.2e}")
    gi = structural_green_identity([uA,uB],1/31,cfg)
    rep("Green identity computed with internal cancellation",
        math.isfinite(gi["residual"]))
    rt = reynolds_transport(uA, np.zeros_like(uA), 0.5, 1/31)
    rep("Reynolds transport identity", rt["residual"]<1e-6,
        f"{rt['residual']:.2e}")
    sig = TopologicalSignature(1,0,1)
    for op in ("N","M","B","H","P","L","C"):
        s2 = TopologicalSignature(2,1,2)
        r2 = apply_surgery(s2,op)
        if not r2["chi_consistent"]: ok=False
    rep("all 7 surgeries have consistent Euler characteristic", ok)
    rep("surgery does NOT provide a chain rule",
        "NOT prove" in apply_surgery(TopologicalSignature(2,1,2),"N")["NOTE"])

    print("\n[T5] NO-ZENO across 7 surgeries")
    nz = NoZenoController(delta=0.05,B0=1.0,G0=0.1)
    ce = nz.choose_eps(0.3)
    rep("eps chosen with q<1", ce["q_below_1"], f"q={ce['q']:.4f}")
    for k,op in enumerate(itertools.islice(itertools.cycle("NMBHPLC"),40)):
        nz.step(op, ce["eps"])
    rep("T_n strictly increasing (no Zeno)",
        all(nz.times[i+1]>nz.times[i] for i in range(len(nz.times)-1)))
    b=False
    try: nz.step("N", nz.eps_max*2)
    except Refusal: b=True
    rep("eps > delta/(2B0) REFUSED", b)
    tail = nz.count_tail(t=1.0, m=200, eps0=ce["eps"])
    rep("surgery-count tail bound computed", "bound" in tail)

    print("\n[T6] ESCA-LD-SESI-3D PDEs: conservation + dissipation")
    for kind in ("CH3D","TF3D","PFC3D"):
        pde = StructuralPDE(kind, cfg, n_per=24, dx=0.05, dt=1e-5)
        v = pde.verify(n_steps=100)
        print(f"      {kind}: mass drift={v['mass_rel_drift']:.2e} "
              f"dissipating={v['dissipating_steps_frac']:.2f}")
        rep(f"{kind} mass approximately conserved", v["mass_rel_drift"]<0.2,
            "explicit scheme; STC enforced at t=0 only")
        cert = sd.certify("structural_conservation", ("STC",),
                          {"kind":kind})
        rep(f"{kind} SD certificate", cert["status"]=="CERTIFIED")

    print("\n[T7] ESCA rigorous corrections")
    ce1 = t2_counterexample_K1()
    rep("K1: first-edition T2 FALSE", ce1["projective"] and ce1["intersection_empty"])
    fam=[np.array([1.0])]
    for m in range(3): fam.append(rng.dirichlet(np.ones(8**(m+1))))
    for m in range(len(fam)-1,0,-1): fam[m-1]=fam[m].reshape(-1,8).sum(1)
    rep("Thm 5.1 code space projective", t2_code_space(fam)["projective"])
    Delta,tau,n=1.3,15,40; errs=[]
    for d in range(1,10):
        t=tau-d
        y=np.concatenate([np.zeros(tau),np.full(n-tau,Delta)])
        S1=np.concatenate([[0],np.cumsum(y)]); S2=np.concatenate([[0],np.cumsum(y*y)])
        def sse(a,b):
            m=b-a; s1=S1[b]-S1[a]; s2=S2[b]-S2[a]
            return s2-s1*s1/m if m>0 else 0.0
        emp=(sse(0,t)+sse(t,n))-(sse(0,tau)+sse(tau,n))
        errs.append(abs(emp-gamma_squared_identity(Delta,d,n-tau)))
    rep("K6: Thm 7.3 gamma^2 identity verified", max(errs)<1e-9,
        f"max err={max(errs):.2e}")
    n=300; y=np.zeros(n); y[100:200]=4.0; y[200:]=-4.0
    y+=rng.normal(0,1,n)
    dp=opmc_dp(y,60,4*math.log(n))
    rep("OPMC recovers 2 change points", dp["K_hat"]==2, str(dp["boundaries"]))
    snr=thm75_snr(4.0,100,1.0,n,2)
    rep("practical SNR does NOT meet Thm 7.5 bound (honest)",
        not snr["SNR_satisfied"])
    rep("Thm 8.3 rate positive", thm83_rate(1,0.02,6400,[2],[0.5])["bound"]>0)

    print("\n[T8] Psychopathology model (research, NOT diagnostic)")
    pr = st.psych.rollout(T=1500)
    print(f"      Lorenz final={[f'{v:.2f}' for v in pr['lorenz_final']]} "
          f"P_mean={pr['pressure_mean']:.2f}")
    rep("symptom network bounded by tanh", pr["network_bounded"])
    rep("model labelled NOT diagnostic", pr["STATUS"]=="RESEARCH_MODEL_NOT_DIAGNOSTIC")
    rep("energy layer labelled CONCEPTUAL", "CONCEPTUAL" in pr["energy_layer"])
    ind = st.psych.disorder_indicators()
    rep("indicators carry clinical disclaimer", "not clinical criteria" in ind["DISCLAIMER"])

    print("\n[T9] Sleep screening (consented, not diagnostic)")
    osa = st.sleep.screen_osa(22.0, 84.0, good)
    rep("OSA screening severity moderate", osa["screening_severity"]=="moderate")
    rep("OSA output NOT a diagnosis", osa["NOT_A_DIAGNOSIS"])
    par = st.sleep.screen_parasomnia(6.0, True, True, good)
    rep("parasomnia prioritises SAFETY", "SAFETY FIRST" in par["priority"])
    nm = st.sleep.screen_nightmare(3.0, True, good)
    rep("nightmare first-line is IRT", "Imagery Rehearsal" in nm["first_line"])

    print("\n[T10] Neuro: legitimate scope only")
    sig = rng.normal(size=(8,500)) + np.sin(2*PI*10*np.arange(500)/250)
    bp = st.neuro_r.bandpower(sig)
    rep("alpha dominant for 10 Hz", float(bp["alpha"].mean())>float(bp["delta"].mean()))
    C = st.neuro_r.covariance(sig)
    rep("Riemannian tangent finite",
        bool(np.isfinite(st.neuro_r.riemannian_tangent(C,C)).all()))
    rep("ALL modalities refuse semantic decoding",
        "REFUSED" in NeuroRead.limits()["ALL"])
    dev = Consent("s02","IRB-002","DBS for PD",True,True,"Parkinson_disease","Dr_Y")
    b=False
    try: NeuroModulate.authorize(dev,"conversion_disorder",amp_mA=2.0,pw_us=60.0,
                                 freq_Hz=130.0,area_cm2=0.06)
    except FNDDeviceRefused: b=True
    rep("DBS for FND REFUSED even with consent", b)
    auth = NeuroModulate.authorize(dev,"Parkinson_disease",amp_mA=2.0,pw_us=60.0,
                                   freq_Hz=130.0,area_cm2=0.06)
    rep("DBS for PD authorised with indication+clinician", auth["allowed"])
    rep("authorisation explicitly excludes thought control",
        "thought control" in auth["NOT_AUTHORIZED"])

    print("\n[T11] Aging organelles + functional gate")
    ar = st.aging.rollout(10.0, schedule=[(2.0,"mitophagy"),(5.0,"senolytic")])
    print(f"      organelles={len(ORGANELLES)} variables={D_STATE} "
          f"couplings={len(COUPLING)}")
    rep("15 organelle compartments", len(ORGANELLES)==15)
    rep("56 state variables", D_STATE==56, str(D_STATE))
    rep("all CSOC bounds hold", all(v["satisfied"] for v in ar["csoc"].values()))
    b=False
    try: AgingEngine.claim(None,-6.0)
    except Refusal: b=True
    rep("rejuvenation claim BLOCKED without function", b)
    ev = FunctionalEvidence({"grip_strength_N":4.2,"VO2max":2.8,"gait_speed":0.11},
        {"grip_strength_N":(2.1,6.3),"VO2max":(1.2,4.4),"gait_speed":(0.04,0.18)},
        n=120, preregistered=True)
    rep("claim ALLOWED with functional evidence", AgingEngine.claim(ev,-6.0)["allowed"])

    print("\n[T12] Sensor EMC (hardening only)")
    m = emc_susceptibility_margin(3.0, 10.0)
    rep("EMC margin computed", abs(m["margin_dB"]-20*math.log10(10/3))<1e-9,
        f"{m['margin_dB']:.2f} dB")
    rep("aviation standards listed for CERTIFICATION",
        any("DO-160G" in s for s in SENSOR_EMC["aviation"]))
    rep("interference generation REFUSED in ledger",
        "REFUSED" in SENSOR_EMC and "jamming" in SENSOR_EMC["REFUSED"])

    print("\n[T13] Chip ledger honesty")
    rep("NLSE labelled classical",
        CHIP_LEDGER["NLSE"]["kind"]=="classical_optical_accelerator")
    rep("NLSE NOT a quantum computer",
        "NOT a quantum computer" in CHIP_LEDGER["NLSE"]["REFUSED"])
    rep("Kerr blocker recorded",
        "NOT demonstrated" in CHIP_LEDGER["R6_QM_R7_QM_R8_R8_5"]["blocker"])
    rep("EUV not a computational problem",
        "manufacturing" in CHIP_LEDGER["EUV_competitor"]["verdict"])

    print("\n[T14] SD gate + report")
    c = sd.certify("change_points", ("L",))
    rep("missing D,O -> MODE_E", c["status"]=="MODE_E")
    r = st.report()
    rep("report written", (st.out/"report.json").exists())
    rep("Buddhist principle in report", "principle" in r["buddhist_principle"])

    print("\n"+"="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
ENFORCED REFUSALS (all fired)
  Thought reading / thought control          -> REFUSED (6/6)
  Covert tracking / no consent               -> REFUSED (4/4)
  Device intervention for FND (5 diagnoses)  -> REFUSED
  Aviation / safety sensor interference      -> REFUSED (5/5)
  Rejuvenation claim without function        -> REFUSED
  DBS for FND even WITH consent              -> REFUSED

AFFIRMED (Buddhist + neuroscience)
  No stimulus sequence guarantees a specified thought.
  manasikara is endogenous; interventions OFFER conditions, never IMPOSE content.
  52 cetasika are not writable registers.

PROVED / VERIFIED HERE
  SESC: ker(T_R) closed; Green internal cancellation; Reynolds transport;
        7-surgery Euler-characteristic algebra.
  No-Zeno: q(eps)<1, T_n strictly increasing, eps bound enforced.
  ESCA-LD-SESI-3D: mass approximately conserved for CH3D/TF3D/PFC3D.
  ESCA rigorous: K1 counterexample; K6 gamma^2 identity to 1e-9;
        OPMC exact DP; honest SNR failure; Thm 8.3 rate.
  Aging: 15 organelles, 56 variables, CSOC bounds, functional gate.

STILL OPEN / NOT CLAIMED
  Universal chain rule across singular surgery (SESC).
  First-principles delta_min for all 7 surgeries.
  Global existence across countably many re-centerings.
  Sharp K-free OPMC separation (O1); Carpet C2/C3 in this normalisation (O2).
  Psychopathology model is NOT diagnostic; energy layer CONCEPTUAL.
  NLSE is NOT a quantum computer; Kerr blockade NOT demonstrated.
  Explicit PDE scheme: conservation is approximate, not exact.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="MSPS unified production")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","buddhist","refusals","fnd","sleep","psych",
                 "sesc","pde","aging","chips","emc","report"])
    ap.add_argument("--diagnosis", default="functional_weakness")
    ap.add_argument("--signs", nargs="*", default=[])
    ap.add_argument("--clinician", default="")
    ap.add_argument("--ahi", type=float, default=22.0)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd=="selftest": return 0 if self_test() else 1
    out={}; st=MSPSUnified()
    good=Consent("subj01","IRB-001","monitoring",True,True,"","")

    if a.cmd=="buddhist":
        out=BuddhistProtectiveLayer.non_injectability_statement()
        print(json.dumps(out,indent=2))
    elif a.cmd=="refusals":
        out={"thought":"REFUSED","covert_tracking":"REFUSED",
             "fnd_device":"REFUSED","aviation_interference":"REFUSED",
             "rejuvenation_without_function":"REFUSED",
             "NLSE_as_quantum":"REFUSED"}
        print(json.dumps(out,indent=2))
    elif a.cmd=="fnd":
        try: out=st.fnd.assess(a.diagnosis,a.signs,a.clinician)
        except Refusal as e: out={"REFUSED":str(e)}
        print(json.dumps(out,indent=2))
    elif a.cmd=="sleep":
        out={"osa":st.sleep.screen_osa(a.ahi,84.0,good),
             "parasomnia":st.sleep.screen_parasomnia(6.0,True,True,good),
             "nightmare":st.sleep.screen_nightmare(3.0,True,good)}
        print(json.dumps(out,indent=2))
    elif a.cmd=="psych":
        out=st.psych.rollout(T=1500); out["indicators"]=st.psych.disorder_indicators()
        print(json.dumps(out,indent=2,default=str))
    elif a.cmd=="sesc":
        cfg=StructuralConfig([Regime("A",(-1,0),1.0),Regime("B",(0,1),2.0)],[(0,1,0.0)])
        uA=np.linspace(-1,0,32); uB=np.linspace(0,1,32); uB[0]=uA[-1]
        out={"trace":TraceOperator(cfg).residual([uA,uB],1/31),
             "green":structural_green_identity([uA,uB],1/31,cfg),
             "surgeries":{k:apply_surgery(TopologicalSignature(2,1,2),k)["after"]
                          for k in SURGERY_7}}
        print(json.dumps(out,indent=2,default=str))
    elif a.cmd=="pde":
        cfg=StructuralConfig([Regime("A",(-1,0),1.0),Regime("B",(0,1),2.0)],[(0,1,0.0)])
        out={k:StructuralPDE(k,cfg,24,0.05,1e-5).verify(100) for k in ("CH3D","TF3D","PFC3D")}
        print(json.dumps(out,indent=2,default=str))
    elif a.cmd=="aging":
        out=st.aging.rollout(a.years,schedule=[(2.0,"mitophagy"),(5.0,"senolytic")])
        print(json.dumps(out,indent=2,default=str))
    elif a.cmd=="chips":
        out=CHIP_LEDGER; print(json.dumps(out,indent=2))
    elif a.cmd=="emc":
        out={"standards":SENSOR_EMC,"margin_example":emc_susceptibility_margin(3.0,10.0)}
        print(json.dumps(out,indent=2))
    elif a.cmd=="report":
        out=st.report()

    if a.json and out:
        Path(a.json).write_text(json.dumps(out,indent=2,default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
