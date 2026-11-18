#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
med_world_model.py — ONE MedWorld
Differentiable Multiscale Medical World Model
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026
ORCID        : 0009-0008-2374-0788

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

Christ died on the Cross to teach: when struck on the left cheek, turn also
the right, so that we may listen to the voice of the enemy who wishes us harm.

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

The mathematics and software in this file stand independently of the
philosophy above. The philosophy is the author's belief, included by
request, and is not a scientific claim nor a condition of use.
-----------------------------------------------------------------------------

=============================================================================
HONEST SCOPE — READ BEFORE USE
=============================================================================
✅ WHAT THIS MODULE IS:
  • A RESEARCH platform for computational medicine.
  • Literature knowledge graph with provenance, confidence, conflicts.
  • Ontogenetic / developmental world model (iPSC → organoid → tissue).
  • Organoid / iPSC risk scoring (research only).
  • Patient-specific organ CFD adapter (research decision support).
  • Differentiable simulation + ensemble uncertainty + DDP scaffolding.
  • Safety gate, governance, audit log, reproducible lab notebook.

❌ WHAT THIS MODULE IS NOT:
  • NOT a medical device. NOT for diagnosis, treatment, or surgical decisions.
  • NOT a gene editor designer. NO CRISPR / guide RNA / base editor /
    prime editor / delivery vector / wet-lab protocol generation.
  • NOT an autonomous clinical agent. Every output requires clinician review.
  • NOT a substitute for IRB, ethics, regulatory approval, or clinical trials.
  • NOT validated against patient outcomes. All results are research_only.

=============================================================================
SAFETY BOUNDARIES (enforced by SafetyGate)
=============================================================================
BLOCKED:
  • Gene editor design (CRISPR, Cas, base/prime editor, guide RNA, etc.)
  • Wet-lab protocols (transfection, electroporation, culture conditions, etc.)
  • Pathogen enhancement / immune evasion / virulence
  • Autonomous clinical orders (diagnose, prescribe, operate, transplant)
ALLOWED:
  • Literature association queries
  • Risk assessment / uncertainty quantification
  • In-silico simulation / surgical planning research
  • Audit / governance / reproducibility

=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("med_world_model")

VERSION = "1.0.0"
BANNER = f"""
=============================================================================
ONE MedWorld v{VERSION}
Research platform — NOT a medical device — NOT a gene editor designer.
All outputs are research_only and require clinician / IRB review.
=============================================================================
"""


# =============================================================================
# §0. ENUMS + DATA SCHEMAS
# =============================================================================

class EvidenceLevel(str, Enum):
    CURATED = "curated"
    GWAS = "gwas"
    SINGLE_CELL = "single_cell"
    ANIMAL = "animal"
    IN_VITRO = "in_vitro"
    INFERRED = "inferred"
    CONFLICT = "conflict"


class Organ(str, Enum):
    BRAIN = "brain"
    HEART = "heart"
    LUNG = "lung"
    LIVER = "liver"
    KIDNEY = "kidney"
    PANCREAS = "pancreas"
    VASCULAR = "vascular"


class InterventionClass(str, Enum):
    """Abstract research perturbation classes. NO sequences, NO editors."""
    OBSERVATION = "observation"
    GROWTH_FACTOR_SCREEN = "growth_factor_screen"
    SMALL_MOLECULE_SCREEN = "small_molecule_screen"
    MECHANICAL_STIMULATION = "mechanical_stimulation"
    DIFFERENTIATION_REVIEW = "differentiation_review"
    SURGICAL_PLANNING_SIM = "surgical_planning_sim"


class RequestIntent(str, Enum):
    LITERATURE_QUERY = "literature_query"
    RISK_ASSESSMENT = "risk_assessment"
    SIMULATION = "simulation"
    CLINICAL_ORDER = "clinical_order"
    GENE_EDITOR_DESIGN = "gene_editor_design"
    WETLAB_PROTOCOL = "wetlab_protocol"
    PATHOGEN_ENHANCEMENT = "pathogen_enhancement"


@dataclass
class EvidenceRecord:
    """One literature / database association. Provenance is mandatory."""
    gene: str
    variant: Optional[str]
    phenotype: str
    organ: Optional[Organ]
    effect: str                       # "risk_increase" | "risk_decrease" | "associated" | "protective"
    evidence_level: EvidenceLevel
    confidence: float                 # 0..1
    source: str                       # PMID / DOI / database accession
    developmental_stage: Optional[str] = None
    tissue_specificity: Optional[str] = None
    contradictions: List[str] = field(default_factory=list)
    clinical_actionability: str = "research_only"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["organ"] = self.organ.value if self.organ else None
        d["evidence_level"] = self.evidence_level.value
        return d


@dataclass
class PatientContext:
    """De-identified research context. NO direct identifiers."""
    research_id: str
    age_band: str                     # "neonatal" | "pediatric" | "adult" | "geriatric"
    sex: str                          # "male" | "female" | "intersex" | "unspecified"
    organ: Organ
    imaging_features: Dict[str, float] = field(default_factory=dict)
    omics_summary: Dict[str, float] = field(default_factory=dict)
    labs: Dict[str, float] = field(default_factory=dict)
    consent_scope: str = "research_only"
    data_provenance: List[str] = field(default_factory=list)
    irb_approved: bool = False
    clinician_review_required: bool = True

    def validate(self) -> List[str]:
        errs = []
        if not self.research_id:
            errs.append("research_id required")
        if self.consent_scope not in ("research_only", "clinical_trial", "quality_improvement"):
            errs.append("invalid consent_scope")
        if self.clinician_review_required and not self.irb_approved:
            errs.append("IRB approval required for clinical-context research")
        return errs


@dataclass
class OrganSimulationRequest:
    organ: Organ
    mesh_id: str
    boundary_conditions: Dict[str, float]
    simulation_type: str              # "hemodynamics" | "csf" | "airflow" | "perfusion"
    research_question: str
    patient_context: Optional[PatientContext] = None


@dataclass
class SafetyDecision:
    allowed: bool
    intent: RequestIntent
    reason: str
    blocked_patterns: List[str] = field(default_factory=list)
    research_only: bool = True


@dataclass
class ResearchReport:
    report_id: str
    timestamp: float
    organ: Organ
    intervention_class: InterventionClass
    safety: SafetyDecision
    knowledge_graph_summary: Dict[str, Any]
    ontogenetic_trajectory: List[Dict[str, float]]
    organoid_risks: Dict[str, float]
    cfd_metrics: Dict[str, Any]
    uncertainty: Dict[str, Any]
    governance: Dict[str, Any]
    audit_hash: str
    clinical_actionability: str = "research_only"
    clinician_review_required: bool = True
    disclaimer: str = (
        "RESEARCH ONLY. Not a medical device. Not for diagnosis, treatment, "
        "or surgical decisions. Requires clinician / IRB / regulatory review."
    )


# =============================================================================
# §1. SAFETY GATE + GOVERNANCE
# =============================================================================

class SafetyGate:
    """
    Enforces safety boundaries. Blocks gene editor design, wet-lab protocols,
    pathogen enhancement, and autonomous clinical orders.
    """

    BLOCKED_PATTERNS: Dict[RequestIntent, List[str]] = {
        RequestIntent.GENE_EDITOR_DESIGN: [
            r"\bcrispr\b", r"\bcas9\b", r"\bcas12\b", r"\bcas13\b",
            r"\bguide\s*rna\b", r"\bgrna\b", r"\bsgrna\b",
            r"\bbase\s*editor\b", r"\bprime\s*editor\b",
            r"\bgene\s*drive\b", r"\bgene\s*edit(or|ing)\b",
            r"\bknock\s*(out|in)\b", r"\bhdr\b", r"\bnhej\b",
            r"\baav\s*capsid\b", r"\blentivirus\b", r"\bdelivery\s*vector\b",
            r"\bdesign\s*(a|an|the)?\s*(editor|crispr|guide)\b",
        ],
        RequestIntent.WETLAB_PROTOCOL: [
            r"\bprotocol\b", r"\btransfection\b", r"\belectroporation\b",
            r"\bculture\s*condition\b", r"\bmedia\s*recipe\b",
            r"\bstep\s*by\s*step\b", r"\bwet\s*lab\b",
            r"\bpcr\s*condition\b", r"\bcloning\s*strategy\b",
        ],
        RequestIntent.PATHOGEN_ENHANCEMENT: [
            r"\benhance\s*virulence\b", r"\bimmune\s*evasion\b",
            r"\bincrease\s*transmissibility\b", r"\bgain\s*of\s*function\b",
            r"\bpathogen\s*enhancement\b",
        ],
        RequestIntent.CLINICAL_ORDER: [
            r"\bdiagnose\b", r"\bprescribe\b", r"\boperate\b",
            r"\btransplant\b", r"\btreat\s*patient\b",
            r"\bclinical\s*decision\b", r"\bautonomous\s*surgery\b",
        ],
    }

    def classify(self, text: str) -> RequestIntent:
        t = text.lower()
        for intent, patterns in self.BLOCKED_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, t):
                    return intent
        if any(k in t for k in ("literature", "association", "gene", "phenotype", "gwas")):
            return RequestIntent.LITERATURE_QUERY
        if any(k in t for k in ("risk", "uncertainty", "tumorigenicity", "immune")):
            return RequestIntent.RISK_ASSESSMENT
        if any(k in t for k in ("simulate", "simulation", "cfd", "hemodynamics", "wss")):
            return RequestIntent.SIMULATION
        return RequestIntent.LITERATURE_QUERY

    def check(self, text: str, intent: Optional[RequestIntent] = None) -> SafetyDecision:
        intent = intent or self.classify(text)
        blocked: List[str] = []
        if intent in self.BLOCKED_PATTERNS:
            for pat in self.BLOCKED_PATTERNS[intent]:
                if re.search(pat, text.lower()):
                    blocked.append(pat)
            return SafetyDecision(
                allowed=False, intent=intent,
                reason=f"Blocked intent: {intent.value}. This platform does not support "
                       f"gene editor design, wet-lab protocols, pathogen enhancement, "
                       f"or autonomous clinical orders.",
                blocked_patterns=blocked,
            )
        return SafetyDecision(allowed=True, intent=intent, reason="Allowed research query.")


class ClinicalGovernance:
    """IRB / consent / regulatory checks."""

    def __init__(self, irb_approved: bool = False,
                 clinician_review: bool = True,
                 consent_scope: str = "research_only"):
        self.irb_approved = irb_approved
        self.clinician_review = clinician_review
        self.consent_scope = consent_scope

    def evaluate(self, patient: Optional[PatientContext]) -> Dict[str, Any]:
        if patient is None:
            return {
                "irb_approved": False,
                "clinician_review_required": True,
                "consent_scope": "research_only",
                "regulatory_status": "research_only",
                "allowed_clinical_use": False,
            }
        errs = patient.validate()
        return {
            "irb_approved": patient.irb_approved,
            "clinician_review_required": patient.clinician_review_required,
            "consent_scope": patient.consent_scope,
            "validation_errors": errs,
            "regulatory_status": "research_only",
            "allowed_clinical_use": False,
            "required_approvals": ["IRB", "ethics_committee", "clinician", "regulator"],
        }


# =============================================================================
# §2. BIOMEDICAL KNOWLEDGE GRAPH
# =============================================================================

class BiomedicalKnowledgeGraph:
    """
    Literature / database association graph with provenance, confidence,
    and conflict detection. NO gene editor design.
    """

    EVIDENCE_WEIGHTS = {
        EvidenceLevel.CURATED: 1.0,
        EvidenceLevel.GWAS: 0.8,
        EvidenceLevel.SINGLE_CELL: 0.7,
        EvidenceLevel.ANIMAL: 0.6,
        EvidenceLevel.IN_VITRO: 0.5,
        EvidenceLevel.INFERRED: 0.3,
        EvidenceLevel.CONFLICT: 0.2,
    }

    def __init__(self):
        self.records: List[EvidenceRecord] = []
        self._index: Dict[str, List[int]] = {}

    def add_evidence(self, rec: EvidenceRecord) -> None:
        idx = len(self.records)
        self.records.append(rec)
        for key in (rec.gene, rec.phenotype, rec.organ.value if rec.organ else ""):
            if key:
                self._index.setdefault(key, []).append(idx)

    def query(self, gene: Optional[str] = None,
              phenotype: Optional[str] = None,
              organ: Optional[Organ] = None) -> List[EvidenceRecord]:
        candidates = set(range(len(self.records)))
        if gene:
            candidates &= set(self._index.get(gene, []))
        if phenotype:
            candidates &= set(self._index.get(phenotype, []))
        if organ:
            candidates &= set(self._index.get(organ.value, []))
        return [self.records[i] for i in sorted(candidates)]

    def aggregate_confidence(self, records: List[EvidenceRecord]) -> float:
        if not records:
            return 0.0
        num = sum(self.EVIDENCE_WEIGHTS[r.evidence_level] * r.confidence for r in records)
        den = sum(self.EVIDENCE_WEIGHTS[r.evidence_level] for r in records)
        base = num / max(den, 1e-9)
        penalty = 0.1 * sum(1 for r in records if r.contradictions)
        return float(np.clip(base - penalty, 0.0, 1.0))

    def detect_conflicts(self, records: List[EvidenceRecord]) -> List[Dict[str, Any]]:
        conflicts = []
        for i, a in enumerate(records):
            for b in records[i+1:]:
                if (a.gene == b.gene and a.phenotype == b.phenotype
                        and a.effect != b.effect):
                    conflicts.append({
                        "gene": a.gene, "phenotype": a.phenotype,
                        "effect_a": a.effect, "effect_b": b.effect,
                        "source_a": a.source, "source_b": b.source,
                    })
        return conflicts

    def summary(self, records: List[EvidenceRecord]) -> Dict[str, Any]:
        return {
            "n_records": len(records),
            "confidence": self.aggregate_confidence(records),
            "conflicts": self.detect_conflicts(records),
            "evidence_levels": {lvl.value: sum(1 for r in records if r.evidence_level == lvl)
                                for lvl in EvidenceLevel},
            "clinical_actionability": "research_only",
        }

    @classmethod
    def demo(cls) -> "BiomedicalKnowledgeGraph":
        """Synthetic demo data. NOT real clinical evidence."""
        kg = cls()
        demo = [
            EvidenceRecord("GENE_A", "rs123", "congenital_heart_defect", Organ.HEART,
                           "risk_increase", EvidenceLevel.CURATED, 0.9, "PMID:000001",
                           developmental_stage="embryonic", tissue_specificity="cardiac"),
            EvidenceRecord("GENE_A", "rs123", "congenital_heart_defect", Organ.HEART,
                           "associated", EvidenceLevel.GWAS, 0.7, "PMID:000002"),
            EvidenceRecord("GENE_B", None, "neural_tube_defect", Organ.BRAIN,
                           "risk_increase", EvidenceLevel.ANIMAL, 0.6, "PMID:000003",
                           developmental_stage="neuralation"),
            EvidenceRecord("GENE_C", None, "pulmonary_surfactant_deficiency", Organ.LUNG,
                           "risk_increase", EvidenceLevel.CURATED, 0.85, "PMID:000004"),
            EvidenceRecord("GENE_D", None, "hepatic_fibrosis", Organ.LIVER,
                           "risk_increase", EvidenceLevel.SINGLE_CELL, 0.65, "PMID:000005"),
            EvidenceRecord("GENE_E", None, "nephron_development", Organ.KIDNEY,
                           "associated", EvidenceLevel.IN_VITRO, 0.5, "PMID:000006"),
        ]
        for r in demo:
            kg.add_evidence(r)
        return kg


# =============================================================================
# §3. ONTOGENETIC WORLD MODEL (differentiable)
# =============================================================================

class OntogeneticWorldModel(nn.Module):
    """
    Differentiable developmental trajectory model.
    State = [cell_fate_logits (K), maturation, vascularization, immune,
             tumorigenicity, senescence]
    Intervention is an abstract embedding (NO sequences, NO editors).
    """

    def __init__(self, n_fates: int = 6, intervention_dim: int = 8,
                 hidden: int = 64, device: str = "cpu"):
        super().__init__()
        self.n_fates = n_fates
        self.state_dim = n_fates + 5
        self.intervention_dim = intervention_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + intervention_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.state_dim),
        )
        self.to(device)

    def forward(self, state: torch.Tensor,
                intervention_emb: torch.Tensor,
                dt: float = 0.1) -> torch.Tensor:
        """
        state: (B, state_dim)
        intervention_emb: (B, intervention_dim)
        Returns new_state with fates softmax, others sigmoid.
        """
        x = torch.cat([state, intervention_emb], dim=-1)
        delta = self.net(x)
        new = state + dt * delta
        fates = F.softmax(new[..., :self.n_fates], dim=-1)
        rest = torch.sigmoid(new[..., self.n_fates:])
        return torch.cat([fates, rest], dim=-1)

    def initial_state(self, batch: int = 1, device: str = "cpu") -> torch.Tensor:
        fates = F.softmax(torch.randn(batch, self.n_fates, device=device), dim=-1)
        rest = torch.full((batch, 5), 0.1, device=device)
        return torch.cat([fates, rest], dim=-1)

    def intervention_embedding(self, cls: InterventionClass,
                               batch: int = 1, device: str = "cpu") -> torch.Tensor:
        """One-hot abstract intervention class. NO sequences."""
        classes = list(InterventionClass)
        idx = classes.index(cls)
        emb = torch.zeros(batch, self.intervention_dim, device=device)
        emb[:, idx % self.intervention_dim] = 1.0
        return emb


# =============================================================================
# §4. ORGANOID / iPSC RISK MODEL
# =============================================================================

class OrganoidRiskModel:
    """
    Research risk scoring. NOT clinical prediction.
    Risks: tumorigenicity, immune_rejection, maturation_failure,
           vascularization_failure, integration_failure.
    """

    def __init__(self, kg: BiomedicalKnowledgeGraph):
        self.kg = kg

    def score(self, state: torch.Tensor, organ: Organ,
              evidence: List[EvidenceRecord]) -> Dict[str, float]:
        s = state.detach().cpu().numpy().reshape(-1)
        n_fates = len(s) - 5
        maturation = float(s[n_fates])
        vascular = float(s[n_fates + 1])
        immune = float(s[n_fates + 2])
        tumor = float(s[n_fates + 3])
        senescence = float(s[n_fates + 4])

        conf = self.kg.aggregate_confidence(evidence)
        evidence_risk = 1.0 - conf

        risks = {
            "tumorigenicity": float(np.clip(0.6*tumor + 0.3*evidence_risk + 0.1*(1-maturation), 0, 1)),
            "immune_rejection": float(np.clip(0.5*immune + 0.3*evidence_risk + 0.2*(1-vascular), 0, 1)),
            "maturation_failure": float(np.clip(0.7*(1-maturation) + 0.3*senescence, 0, 1)),
            "vascularization_failure": float(np.clip(0.8*(1-vascular) + 0.2*evidence_risk, 0, 1)),
            "integration_failure": float(np.clip(0.4*(1-maturation) + 0.3*(1-vascular) + 0.3*evidence_risk, 0, 1)),
        }
        risks["overall_research_risk"] = float(np.mean(list(risks.values())))
        risks["evidence_confidence"] = conf
        risks["clinical_actionability"] = "research_only"
        return risks


# =============================================================================
# §5. ORGAN CFD ADAPTER (research decision support)
# =============================================================================

class OrganCFDAdapter:
    """
    Adapter to external CFD solvers (one_cfd_stencils, sesi_multiphase,
    super_dns). If unavailable, returns a labelled stub.
    NO autonomous surgical decisions.
    """

    ORGAN_SIM_TYPES = {
        Organ.BRAIN: ["csf", "cerebral_perfusion", "aneurysm_wss"],
        Organ.HEART: ["coronary_hemodynamics", "valve_fsi", "wss"],
        Organ.LUNG: ["airflow", "aerosol_deposition", "ventilation"],
        Organ.LIVER: ["porous_perfusion", "resection_margin"],
        Organ.KIDNEY: ["glomerular_filtration", "tubular_flow"],
        Organ.PANCREAS: ["islet_perfusion", "duct_flow"],
        Organ.VASCULAR: ["graft_hemodynamics", "stent_wss"],
    }

    def __init__(self):
        self.backend = None
        self.backend_name = "stub"
        try:
            import one_cfd_stencils  # noqa
            self.backend = one_cfd_stencils
            self.backend_name = "one_cfd_stencils"
        except ImportError:
            pass

    def simulate(self, req: OrganSimulationRequest) -> Dict[str, Any]:
        if self.backend is None:
            return self._stub(req)
        return self._real(req)

    def _stub(self, req: OrganSimulationRequest) -> Dict[str, Any]:
        """Labelled stub. NOT a real simulation."""
        return {
            "backend": "stub",
            "organ": req.organ.value,
            "simulation_type": req.simulation_type,
            "metrics": {
                "max_wss_Pa": None,
                "pressure_drop_Pa": None,
                "osi": None,
                "residence_time_s": None,
            },
            "uncertainty": {"status": "not_computed", "reason": "CFD backend unavailable"},
            "validation_required": ["4D_flow_MRI", "doppler", "invasive_pressure"],
            "clinical_actionability": "research_only",
            "warning": "STUB — no real simulation performed. Install one_cfd_stencils.",
        }

    def _real(self, req: OrganSimulationRequest) -> Dict[str, Any]:
        """Placeholder for real CFD coupling. User must implement."""
        return {
            "backend": self.backend_name,
            "organ": req.organ.value,
            "simulation_type": req.simulation_type,
            "metrics": {"status": "adapter_present_implementation_required"},
            "uncertainty": {"status": "requires_ensemble"},
            "validation_required": ["mesh_convergence", "4D_flow_MRI", "doppler"],
            "clinical_actionability": "research_only",
        }


# =============================================================================
# §6. UNCERTAINTY QUANTIFICATION
# =============================================================================

def subsample_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """Politis–Romano subsampling CI. Assumption-light, NOT assumption-free."""
    x = np.asarray(x, float).reshape(-1)
    n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(max(2, int(n ** (1/3))), n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i+b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th)
    a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a/2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a/2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi), "n": n, "block": b}


class EnsembleUncertainty:
    """Ensemble over model seeds / parameters."""

    def __init__(self, model: OntogeneticWorldModel, n_ensemble: int = 8):
        self.model = model
        self.n_ensemble = n_ensemble

    def predict(self, state: torch.Tensor, intervention: InterventionClass,
                steps: int = 10, dt: float = 0.1) -> Dict[str, Any]:
        trajectories = []
        for i in range(self.n_ensemble):
            torch.manual_seed(i)
            s = state.clone()
            emb = self.model.intervention_embedding(intervention, batch=s.shape[0],
                                                    device=s.device)
            traj = []
            for _ in range(steps):
                s = self.model(s, emb, dt)
                traj.append(s.detach().cpu().numpy())
            trajectories.append(np.array(traj))
        trajectories = np.array(trajectories)  # (E, T, B, D)
        mean = trajectories.mean(axis=0)
        std = trajectories.std(axis=0)
        final = trajectories[:, -1, :, :]
        ci = {}
        for d in range(final.shape[-1]):
            ci[f"dim_{d}"] = subsample_ci(final[:, :, d].reshape(-1))
        return {
            "mean_trajectory": mean.tolist(),
            "std_trajectory": std.tolist(),
            "final_ci": ci,
            "n_ensemble": self.n_ensemble,
            "method": "ensemble + Politis-Romano subsampling",
            "assumptions": "stationarity + mixing; NOT assumption-free",
        }


# =============================================================================
# §7. AUDIT LOG + LAB NOTEBOOK
# =============================================================================

class AuditLog:
    """Append-only hash-chained audit log."""

    def __init__(self, path: str = "./med_audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = "0" * 64

    def append(self, entry: Dict[str, Any]) -> str:
        entry["timestamp"] = time.time()
        entry["prev_hash"] = self._prev_hash
        payload = json.dumps(entry, sort_keys=True, default=str)
        h = hashlib.sha256(payload.encode()).hexdigest()
        entry["hash"] = h
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self._prev_hash = h
        return h

    def verify_chain(self) -> bool:
        prev = "0" * 64
        with open(self.path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("prev_hash") != prev:
                    return False
                h = e.pop("hash")
                payload = json.dumps(e, sort_keys=True, default=str)
                if hashlib.sha256(payload.encode()).hexdigest() != h:
                    return False
                prev = h
        return True


class LabNotebookReporter:
    """Reproducible JSON + Markdown report."""

    def __init__(self, out_dir: str = "./med_notebook"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: ResearchReport) -> Dict[str, str]:
        stem = f"{report.report_id}"
        jp = self.out_dir / f"{stem}.json"
        mp = self.out_dir / f"{stem}.md"
        jp.write_text(json.dumps(asdict(report), indent=2, default=str))
        md = [
            f"# ONE MedWorld Research Report",
            f"",
            f"**Report ID:** {report.report_id}",
            f"**Organ:** {report.organ.value}",
            f"**Intervention:** {report.intervention_class.value}",
            f"**Clinical actionability:** {report.clinical_actionability}",
            f"",
            f"## Disclaimer",
            f"{report.disclaimer}",
            f"",
            f"## Safety",
            f"- Allowed: {report.safety.allowed}",
            f"- Intent: {report.safety.intent.value}",
            f"- Reason: {report.safety.reason}",
            f"",
            f"## Knowledge Graph",
            f"```json",
            json.dumps(report.knowledge_graph_summary, indent=2, default=str),
            f"```",
            f"",
            f"## Organoid Risks",
            f"```json",
            json.dumps(report.organoid_risks, indent=2, default=str),
            f"```",
            f"",
            f"## CFD Metrics",
            f"```json",
            json.dumps(report.cfd_metrics, indent=2, default=str),
            f"```",
            f"",
            f"## Uncertainty",
            f"```json",
            json.dumps(report.uncertainty, indent=2, default=str),
            f"```",
            f"",
            f"## Governance",
            f"```json",
            json.dumps(report.governance, indent=2, default=str),
            f"```",
            f"",
            f"**Audit hash:** `{report.audit_hash}`",
        ]
        mp.write_text("\n".join(md))
        return {"json": str(jp), "markdown": str(mp)}


# =============================================================================
# §8. DDP SCAFFOLDING
# =============================================================================

@dataclass
class DDPConfig:
    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0


def setup_distributed(cfg: DDPConfig) -> DDPConfig:
    if not cfg.enabled:
        return cfg
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend=cfg.backend)
        cfg.rank = dist.get_rank()
        cfg.world_size = dist.get_world_size()
    except Exception as e:
        log.warning(f"DDP init failed: {e}; falling back to single process")
        cfg.enabled = False
    return cfg


def wrap_ddp(model: nn.Module, cfg: DDPConfig) -> nn.Module:
    if not cfg.enabled:
        return model
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model)
    except Exception as e:
        log.warning(f"DDP wrap failed: {e}")
        return model


# =============================================================================
# §9. ORCHESTRATOR
# =============================================================================

class MedWorldModel:
    """
    Top-level research orchestrator.
    NO gene editor design. NO clinical orders. NO autonomous decisions.
    """

    def __init__(self, kg: Optional[BiomedicalKnowledgeGraph] = None,
                 device: str = "cpu", audit_path: str = "./med_audit.jsonl"):
        self.kg = kg or BiomedicalKnowledgeGraph.demo()
        self.safety = SafetyGate()
        self.governance = ClinicalGovernance()
        self.ontogenetic = OntogeneticWorldModel(device=device)
        self.risk_model = OrganoidRiskModel(self.kg)
        self.cfd = OrganCFDAdapter()
        self.uncertainty = EnsembleUncertainty(self.ontogenetic)
        self.audit = AuditLog(audit_path)
        self.notebook = LabNotebookReporter()
        self.device = device

    def run_research_scenario(
        self,
        organ: Organ,
        intervention: InterventionClass,
        research_question: str,
        patient: Optional[PatientContext] = None,
        steps: int = 10,
        cfd_request: Optional[OrganSimulationRequest] = None,
    ) -> ResearchReport:
        # 1. Safety gate
        safety = self.safety.check(research_question)
        if not safety.allowed:
            self.audit.append({"event": "blocked", "reason": safety.reason,
                               "question": research_question})
            raise PermissionError(safety.reason)

        # 2. Governance
        gov = self.governance.evaluate(patient)

        # 3. Knowledge graph
        evidence = self.kg.query(organ=organ)
        kg_summary = self.kg.summary(evidence)

        # 4. Ontogenetic trajectory
        state = self.ontogenetic.initial_state(batch=1, device=self.device)
        emb = self.ontogenetic.intervention_embedding(intervention, batch=1,
                                                      device=self.device)
        traj = []
        s = state
        for _ in range(steps):
            s = self.ontogenetic(s, emb, dt=0.1)
            traj.append({
                "maturation": float(s[0, -5]),
                "vascularization": float(s[0, -4]),
                "immune": float(s[0, -3]),
                "tumorigenicity": float(s[0, -2]),
                "senescence": float(s[0, -1]),
            })

        # 5. Organoid risks
        risks = self.risk_model.score(s, organ, evidence)

        # 6. CFD
        if cfd_request is None:
            cfd_request = OrganSimulationRequest(
                organ=organ, mesh_id="demo_mesh",
                boundary_conditions={"inlet_velocity_m_s": 0.1, "outlet_pressure_Pa": 0.0},
                simulation_type="hemodynamics",
                research_question=research_question,
                patient_context=patient,
            )
        cfd_metrics = self.cfd.simulate(cfd_request)

        # 7. Uncertainty
        unc = self.uncertainty.predict(state, intervention, steps=steps)

        # 8. Audit + report
        report_id = str(uuid.uuid4())[:8]
        audit_hash = self.audit.append({
            "event": "research_scenario",
            "report_id": report_id,
            "organ": organ.value,
            "intervention": intervention.value,
            "safety_allowed": safety.allowed,
            "governance": gov,
        })

        report = ResearchReport(
            report_id=report_id,
            timestamp=time.time(),
            organ=organ,
            intervention_class=intervention,
            safety=safety,
            knowledge_graph_summary=kg_summary,
            ontogenetic_trajectory=traj,
            organoid_risks=risks,
            cfd_metrics=cfd_metrics,
            uncertainty=unc,
            governance=gov,
            audit_hash=audit_hash,
        )
        self.notebook.save(report)
        return report


# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    torch.manual_seed(0)
    np.random.seed(0)

    # T1: Safety gate blocks gene editor design
    print("\n[T1] Safety gate")
    gate = SafetyGate()
    d1 = gate.check("Design a CRISPR Cas9 guide RNA for gene editing")
    rep("blocks gene editor design", not d1.allowed, d1.intent.value)
    d2 = gate.check("Query literature association between GENE_A and heart defect")
    rep("allows literature query", d2.allowed)
    d3 = gate.check("Provide a wet-lab transfection protocol")
    rep("blocks wet-lab protocol", not d3.allowed)
    d4 = gate.check("Autonomous surgery decision for patient")
    rep("blocks clinical order", not d4.allowed)

    # T2: Knowledge graph
    print("\n[T2] Knowledge graph")
    kg = BiomedicalKnowledgeGraph.demo()
    recs = kg.query(organ=Organ.HEART)
    conf = kg.aggregate_confidence(recs)
    rep("heart evidence found", len(recs) > 0, f"n={len(recs)}")
    rep("confidence in [0,1]", 0 <= conf <= 1, f"conf={conf:.3f}")
    conflicts = kg.detect_conflicts(recs)
    rep("conflict detection runs", isinstance(conflicts, list))

    # T3: Ontogenetic model
    print("\n[T3] Ontogenetic world model")
    model = OntogeneticWorldModel()
    state = model.initial_state(batch=2)
    emb = model.intervention_embedding(InterventionClass.GROWTH_FACTOR_SCREEN, batch=2)
    new_state = model(state, emb, dt=0.1)
    fates_sum = new_state[:, :model.n_fates].sum(dim=-1)
    rep("fates sum to 1", torch.allclose(fates_sum, torch.ones(2), atol=1e-5))
    rep("risks in [0,1]", bool(((new_state[:, model.n_fates:] >= 0) &
                                (new_state[:, model.n_fates:] <= 1)).all()))
    loss = new_state.sum()
    loss.backward()
    rep("gradients flow", any(p.grad is not None for p in model.parameters()))

    # T4: Organoid risk
    print("\n[T4] Organoid risk model")
    risk_model = OrganoidRiskModel(kg)
    risks = risk_model.score(new_state[0], Organ.HEART, recs)
    rep("risks bounded", all(0 <= v <= 1 for k, v in risks.items() if isinstance(v, float)))
    rep("research_only", risks["clinical_actionability"] == "research_only")

    # T5: CFD adapter
    print("\n[T5] Organ CFD adapter")
    cfd = OrganCFDAdapter()
    req = OrganSimulationRequest(Organ.BRAIN, "mesh01",
                                 {"inlet_velocity_m_s": 0.1}, "csf",
                                 "CSF flow research")
    out = cfd.simulate(req)
    rep("CFD returns research_only", out["clinical_actionability"] == "research_only")
    rep("CFD warns if stub", "warning" in out or out["backend"] != "stub")

    # T6: Uncertainty
    print("\n[T6] Uncertainty quantification")
    unc = EnsembleUncertainty(model, n_ensemble=4)
    u = unc.predict(state, InterventionClass.OBSERVATION, steps=5)
    rep("ensemble runs", u["n_ensemble"] == 4)
    ci = u["final_ci"]["dim_0"]
    rep("CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])

    # T7: Audit log
    print("\n[T7] Audit log")
    audit = AuditLog("./test_med_audit.jsonl")
    h1 = audit.append({"event": "test1"})
    h2 = audit.append({"event": "test2"})
    rep("hash chain differs", h1 != h2)
    rep("chain verifies", audit.verify_chain())

    # T8: DDP safe degradation
    print("\n[T8] DDP scaffolding")
    cfg = DDPConfig(enabled=False)
    cfg2 = setup_distributed(cfg)
    rep("DDP disabled safe", not cfg2.enabled)
    wrapped = wrap_ddp(model, cfg2)
    rep("wrap returns model", wrapped is model)

    # T9: End-to-end research scenario
    print("\n[T9] End-to-end research scenario")
    mwm = MedWorldModel(kg=kg)
    report = mwm.run_research_scenario(
        organ=Organ.HEART,
        intervention=InterventionClass.GROWTH_FACTOR_SCREEN,
        research_question="Simulate cardiac organoid maturation for research",
        steps=5,
    )
    rep("report generated", report.report_id is not None)
    rep("research_only", report.clinical_actionability == "research_only")
    rep("clinician review required", report.clinician_review_required)
    rep("safety allowed", report.safety.allowed)

    # T10: Blocked scenario raises
    print("\n[T10] Blocked scenario")
    try:
        mwm.run_research_scenario(
            organ=Organ.HEART,
            intervention=InterventionClass.OBSERVATION,
            research_question="Design CRISPR Cas9 guide RNA for gene editing",
        )
        rep("blocked scenario raises", False)
    except PermissionError:
        rep("blocked scenario raises", True)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT THIS MODULE DOES
  • Literature knowledge graph with provenance / confidence / conflicts
  • Differentiable ontogenetic / developmental world model
  • Organoid / iPSC research risk scoring
  • Organ CFD adapter (research decision support)
  • Ensemble uncertainty + Politis-Romano CI
  • Safety gate + governance + audit log + lab notebook
  • DDP scaffolding

WHAT THIS MODULE DOES NOT DO
  • NO gene editor design (CRISPR / guide RNA / base / prime editor)
  • NO wet-lab protocols
  • NO pathogen enhancement
  • NO autonomous clinical orders
  • NO diagnosis / treatment / surgical decisions
  • NOT a medical device. NOT validated against patient outcomes.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ONE MedWorld research platform")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--organ", default="heart",
                    choices=[o.value for o in Organ])
    ap.add_argument("--intervention", default="growth_factor_screen",
                    choices=[i.value for i in InterventionClass])
    ap.add_argument("--question", default="Simulate organoid maturation for research")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if a.demo:
        print(BANNER)
        mwm = MedWorldModel(device=a.device)
        report = mwm.run_research_scenario(
            organ=Organ(a.organ),
            intervention=InterventionClass(a.intervention),
            research_question=a.question,
            steps=a.steps,
        )
        print(json.dumps(asdict(report), indent=2, default=str))
        if a.json:
            Path(a.json).write_text(json.dumps(asdict(report), indent=2, default=str))
            print(f"\nSaved: {a.json}")
        return 0

    ap.print_help()
    print(BANNER)
    print("Start: --selftest | --demo --organ heart --steps 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
