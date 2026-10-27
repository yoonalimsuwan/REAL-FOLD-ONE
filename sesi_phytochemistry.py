"""
=============================================================================
SESI PHYTOCHEMISTRY MODULE v1.0
Rigorous Plant-Derived Compound Analysis — Anti-Marketing-Hype Framework
โมดูลพฤกษเคมีเชิงเข้มงวด ต้านการตลาดแบบคาดเดา
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Evidence-Based Phytochemistry)

WHY THIS MODULE EXISTS (ทำไมต้องมีโมดูลนี้)
────────────────────────────────────────────
Phytochemistry is systematically misrepresented in three ways:

  [P1] OVERALL-OUTCOME FALLACY
       Marketing reports only "the herb works" without identifying
       which compounds act, how, and at what dose. This module forces
       mixture decomposition down to individual compounds.

  [P2] COMPLEXITY UNDERESTIMATION
       A synthetic drug = 1 molecule. A herbal extract = 100-1000+
       compounds with synergy, antagonism, and matrix effects.
       Complexity is QUANTIFIED here, never ignored.

  [P3] MARKETING-DRIVEN SPECULATION
       Claims like "natural = safe" or "cures all diseases" harm
       consumers. This module grades evidence rigorously and flags
       dangerous overclaims. A PhytoSafetyGate blocks approval
       predictions when toxicity signals are present.

CORE PRINCIPLE
──────────────
  Natural ≠ Safe. In-vitro ≠ In-vivo. Traditional ≠ Proven.
  Every prediction carries an explicit evidence grade.

ARCHITECTURE
────────────
  PhytoCompoundEncoder ──► single molecule embedding
        │
  PhytoMixtureEncoder ──► mixture + concentration matrix
        │
        ├─► MolecularComplexityScorer   [P2]
        ├─► PhytoToxicityPredictor      [P3] SAFETY FIRST
        ├─► PhytoDrugInteractionPredictor [P3]
        ├─► PhytoADMETModel             [bioavailability reality check]
        ├─► PhytoSynergyModel           [combination effects]
        └─► EvidenceQualityGrader       [anti-marketing rigor]
                    │
                    ▼
              PhytoSafetyGate ──► blocks approval if toxic
                    │
                    ▼
              PhytochemOrchestrator ──► graded, honest report

HONEST SCOPE
────────────
• This is a COMPUTATIONAL RESEARCH FRAMEWORK, not a clinical tool.
• Toxicity/interaction predictions are probabilistic estimates —
  they do NOT replace wet-lab toxicology or clinical trials.
• Evidence grading is a structured heuristic; it does not replace
  systematic review methodology (Cochrane, etc.).
• "Natural product" claims must still pass regulatory review
  (Thai FDA, EMA, US FDA) before any health claim is made.
• This module can be used for legitimate Thai herbal medicine
  RESEARCH and CONSUMER PROTECTION, not for making medical claims.

=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.phytochemistry")

__all__ = [
    "PhytochemConfig",
    "EvidenceLevel",
    "ClaimType",
    # Encoders
    "PhytoCompoundEncoder",
    "PhytoMixtureEncoder",
    # Analysis
    "MolecularComplexityScorer",
    "PhytoToxicityPredictor",
    "PhytoDrugInteractionPredictor",
    "PhytoADMETModel",
    "PhytoSynergyModel",
    # Rigor
    "EvidenceQualityGrader",
    "PhytoSafetyGate",
    # Unified
    "PhytochemOrchestrator",
]


# ═══════════════════════════════════════════════════════════════════════
# C^∞ Primitives (consistent with SESI ecosystem)
# ═══════════════════════════════════════════════════════════════════════

def sabs(x: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.sqrt(x * x + eps * eps)

def sfloor(x: Tensor, floor: float, beta: float = 50.0) -> Tensor:
    return floor + F.softplus(x - floor, beta=beta)

def sgate(x: Tensor, tau: float = 1e-2) -> Tensor:
    return torch.sigmoid(x / tau)

class LowRank(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 32):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_in, r) * 0.02)
        self.V = nn.Parameter(torch.zeros(r, d_out))
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.U.to(x.dtype)) @ self.V.to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════
# ENUMS — Evidence & Claim Classification
# ═══════════════════════════════════════════════════════════════════════

class EvidenceLevel(Enum):
    """
    Evidence grading scale (adapted from EBM hierarchy).
    Higher value = stronger evidence.
    """
    NONE = 0            # No evidence, pure speculation
    TRADITIONAL = 1     # Traditional use only (ethnobotany)
    INVITRO = 2         # Cell culture / biochemical assay
    ANIMAL = 3          # Animal models
    PHASE1 = 4          # Human safety (Phase I)
    PHASE2 = 5          # Human efficacy small (Phase II)
    RCT = 6             # Randomized controlled trial
    METAANALYSIS = 7    # Systematic review / meta-analysis


class ClaimType(Enum):
    """
    Types of health claims commonly seen in phytochemical marketing.
    Used to detect and grade overclaims.
    """
    GENERAL_WELLNESS = auto()     # "supports health" (low risk)
    SPECIFIC_BENEFIT = auto()     # "may help with X" (moderate)
    DISEASE_TREATMENT = auto()    # "treats disease Y" (HIGH risk)
    DISEASE_CURE = auto()         # "cures disease Y" (DANGEROUS)
    CURES_ALL = auto()            # "panacea" (FRAUDULENT)


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PhytochemConfig:
    # ── Dimensions ──────────────────────────────────────────────────────
    latent_dim: int = 256
    compound_dim: int = 128        # per-compound embedding
    fingerprint_dim: int = 1024    # molecular fingerprint size
    mixture_max_compounds: int = 32  # max compounds per mixture

    # ── Toxicity endpoints ──────────────────────────────────────────────
    tox_endpoints: List[str] = field(default_factory=lambda: [
        "herg",            # cardiotoxicity
        "hepatotoxicity",  # liver
        "nephrotoxicity",  # kidney
        "genotoxicity",    # DNA damage
        "carcinogenicity", # cancer-causing
        "neurotoxicity",   # nervous system
        "immunotoxicity",  # immune
        "endocrine",       # hormone disruption
    ])

    # ── ADMET ───────────────────────────────────────────────────────────
    admet_tasks: List[str] = field(default_factory=lambda: [
        "absorption", "distribution", "metabolism",
        "excretion", "bioavailability",
    ])

    # ── CYP450 interaction enzymes ──────────────────────────────────────
    cyp_enzymes: List[str] = field(default_factory=lambda: [
        "CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4",
    ])

    # ── Training ────────────────────────────────────────────────────────
    batch_size: int = 8
    lr: float = 3e-4
    device: str = "cuda"

    # ── Safety thresholds (tunable, defaults are conservative) ──────────
    tox_alarm_threshold: float = 0.5      # probability above = alarm
    interaction_alarm_threshold: float = 0.5
    bioavailability_min: float = 0.05     # <5% = unlikely effective


# ═══════════════════════════════════════════════════════════════════════
# 1. PHYTO COMPOUND ENCODER
# ═══════════════════════════════════════════════════════════════════════

class PhytoCompoundEncoder(nn.Module):
    """
    Encode a single phytochemical compound into a latent vector.

    In production, this consumes molecular fingerprints (ECFP/MACCS)
    or graph neural network embeddings. For the research framework,
    it operates on a fingerprint-sized input.

    The encoder also extracts structural descriptors relevant to
    complexity: ring count, heteroatoms, stereocenters, flexibility.
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        # Fingerprint → compound embedding
        self.fp_encoder = nn.Sequential(
            nn.Linear(cfg.fingerprint_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.compound_dim),
        )

        # Structural descriptor head
        # (rings, heteroatoms, stereocenters, rotatable bonds, MW proxy)
        self.descriptor_head = nn.Sequential(
            nn.Linear(cfg.compound_dim, 32),
            nn.GELU(),
            nn.Linear(32, 5),
            nn.Softplus(),  # descriptors are positive
        )

    def forward(self, fingerprint: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            fingerprint: (B, fingerprint_dim) molecular fingerprint
        Returns:
            compound_embedding, structural_descriptors
        """
        embedding = self.fp_encoder(fingerprint)
        descriptors = self.descriptor_head(embedding)
        return {
            "compound_embedding": embedding,
            "structural_descriptors": descriptors,
        }


# ═══════════════════════════════════════════════════════════════════════
# 2. PHYTO MIXTURE ENCODER (the key differentiator)
# ═══════════════════════════════════════════════════════════════════════

class PhytoMixtureEncoder(nn.Module):
    """
    Encode an entire herbal extract as a MIXTURE of compounds with
    concentrations. This is the critical anti-"overall-outcome" component.

    A mixture is represented as:
        compounds:  (B, K, compound_dim)  — K compounds
        concentrations: (B, K)            — relative concentration

    Attention over compounds produces a mixture-level representation,
    weighted by concentration (abundant compounds matter more, but
    potent trace compounds are not ignored — attention handles this).
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        # Concentration-aware attention
        self.mixture_attn = nn.MultiheadAttention(
            embed_dim=cfg.compound_dim,
            num_heads=4,
            batch_first=True,
        )

        # Concentration embedding
        self.conc_embed = nn.Sequential(
            nn.Linear(1, cfg.compound_dim),
            nn.Tanh(),
        )

        # Mixture pooling → global representation
        self.mixture_pool = nn.Sequential(
            nn.Linear(cfg.compound_dim * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )

        # Number-of-active-compounds estimator
        self.active_counter = nn.Sequential(
            nn.Linear(cfg.compound_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        compound_embeddings: Tensor,  # (B, K, compound_dim)
        concentrations: Tensor,       # (B, K)
    ) -> Dict[str, Tensor]:
        B, K, D = compound_embeddings.shape

        # Embed concentration and add to compound features
        conc_feat = self.conc_embed(concentrations.unsqueeze(-1))  # (B, K, D)
        conc_aware = compound_embeddings + conc_feat

        # Self-attention over compounds (mixture interactions)
        attn_out, attn_weights = self.mixture_attn(
            conc_aware, conc_aware, conc_aware
        )

        # Concentration-weighted pooling
        conc_norm = concentrations / (concentrations.sum(dim=-1, keepdim=True) + 1e-8)
        weighted_pool = (attn_out * conc_norm.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Concatenate attention mean and weighted pool
        attn_mean = attn_out.mean(dim=1)
        mixture_repr = self.mixture_pool(torch.cat([attn_mean, weighted_pool], dim=-1))

        # Estimate number of pharmacologically active compounds
        active_scores = self.active_counter(attn_out).squeeze(-1)  # (B, K)
        active_count = (active_scores * concentrations).sum(dim=-1)

        return {
            "mixture_representation": mixture_repr,
            "compound_attention": attn_weights,
            "active_scores": active_scores,
            "estimated_active_count": active_count,
            "diversity_index": self._shannon(concentrations),
        }

    @staticmethod
    def _shannon(conc: Tensor) -> Tensor:
        """Shannon diversity of the concentration distribution."""
        p = conc / (conc.sum(dim=-1, keepdim=True) + 1e-8)
        return -(p * torch.log(p + 1e-8)).sum(dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# 3. MOLECULAR COMPLEXITY SCORER
# ═══════════════════════════════════════════════════════════════════════

class MolecularComplexityScorer(nn.Module):
    """
    Quantify the complexity of a phytochemical mixture — directly
    countering the "it's just one herb" oversimplification.

    Outputs a multi-dimensional complexity profile:
      • compound_count_effective
      • interaction_potential (pairwise)
      • structural_diversity
      • concentration_dispersion

    HONEST NOTE: high complexity does NOT automatically mean "doesn't work"
    — it means "requires more rigorous study before claims are made."
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        self.complexity_net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),   # 4 complexity dimensions
            nn.Softplus(),
        )

    def forward(self, mixture_repr: Tensor,
                diversity_index: Tensor,
                active_count: Tensor) -> Dict[str, Tensor]:
        B = mixture_repr.shape[0]

        raw = self.complexity_net(mixture_repr)  # (B, 4)

        complexity_dims = {
            "compound_complexity": raw[:, 0],
            "interaction_potential": raw[:, 1] * (active_count / 10.0).clamp(min=0.1),
            "structural_diversity": raw[:, 2] + diversity_index,
            "concentration_dispersion": raw[:, 3],
        }

        # Aggregate complexity score (log-scale to avoid blowup)
        total = torch.stack(list(complexity_dims.values()), dim=-1)
        aggregate = torch.log1p(total.sum(dim=-1))

        # Complexity category (for reporting)
        # low < 2, moderate 2-4, high > 4 (heuristic thresholds)
        category = torch.where(
            aggregate < 2.0,
            torch.zeros_like(aggregate),
            torch.where(aggregate < 4.0, torch.ones_like(aggregate),
                        torch.full_like(aggregate, 2.0))
        )

        return {
            **complexity_dims,
            "total_complexity": aggregate,
            "complexity_category": category,  # 0=low,1=moderate,2=high
        }


# ═══════════════════════════════════════════════════════════════════════
# 4. PHYTO TOXICITY PREDICTOR (SAFETY FIRST)
# ═══════════════════════════════════════════════════════════════════════

class PhytoToxicityPredictor(nn.Module):
    """
    Multi-endpoint toxicity prediction.

    This is the ANTI-"natural=safe" component. It predicts probability
    of toxicity across multiple organ systems and mechanisms.

    Endpoints: hERG, hepatotoxicity, nephrotoxicity, genotoxicity,
    carcinogenicity, neurotoxicity, immunotoxicity, endocrine disruption.

    HONEST NOTE: These are computational estimates. Known toxic plant
    compounds (aristolochic acid, pyrrolizidine alkaloids, aconitine,
    atropine) MUST be caught by wet-lab and regulatory review. This
    model supports but does not replace experimental toxicology.
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg
        n_endpoints = len(cfg.tox_endpoints)

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 2, cfg.latent_dim // 2),
        )

        # One head per toxicity endpoint
        self.tox_heads = nn.ModuleDict({
            endpoint: nn.Sequential(
                nn.Linear(cfg.latent_dim // 2, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ) for endpoint in cfg.tox_endpoints
        })

    def forward(self, mixture_repr: Tensor) -> Dict[str, Tensor]:
        B = mixture_repr.shape[0]
        features = self.backbone(mixture_repr)

        tox_probs = {}
        for endpoint, head in self.tox_heads.items():
            tox_probs[endpoint] = head(features).squeeze(-1)  # (B,)

        # Stack for aggregation
        tox_stack = torch.stack(list(tox_probs.values()), dim=-1)  # (B, n_endpoints)

        # Overall toxicity risk = max-ish (pessimistic, safety-first)
        # Use log-sum-exp to approximate max smoothly
        overall_risk = torch.logsumexp(tox_stack, dim=-1) - math.log(len(self.cfg.tox_endpoints))

        # Flag any endpoint above alarm threshold
        alarm_mask = (tox_stack > self.cfg.tox_alarm_threshold).float()
        n_alarms = alarm_mask.sum(dim=-1)
        has_alarm = (n_alarms > 0).float()

        return {
            **{f"tox_{k}": v for k, v in tox_probs.items()},
            "tox_overall_risk": overall_risk,
            "tox_n_alarms": n_alarms,
            "tox_has_alarm": has_alarm,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5. PHYTO DRUG INTERACTION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════

class PhytoDrugInteractionPredictor(nn.Module):
    """
    Predict herb-drug interactions, primarily via CYP450 enzyme
    inhibition/induction and P-glycoprotein effects.

    Famous real examples this must catch:
      • St. John's Wort → induces CYP3A4 → reduces efficacy of many drugs
      • Grapefruit furanocoumarins → inhibit CYP3A4 → drug overdose risk
      • Ginkgo → antiplatelet interaction with warfarin

    HONEST NOTE: Interaction prediction is inherently uncertain;
    clinical pharmacology review is required before co-prescription.
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
        )

        # CYP inhibition/induction per enzyme
        self.cyp_heads = nn.ModuleDict({
            enzyme: nn.Sequential(
                nn.Linear(cfg.latent_dim // 2, 32),
                nn.GELU(),
                nn.Linear(32, 2),  # [inhibition_prob, induction_prob]
                nn.Sigmoid(),
            ) for enzyme in cfg.cyp_enzymes
        })

        # P-glycoprotein effect
        self.pgp_head = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, mixture_repr: Tensor) -> Dict[str, Tensor]:
        B = mixture_repr.shape[0]
        features = self.backbone(mixture_repr)

        cyp_results = {}
        all_cyp_probs = []
        for enzyme, head in self.cyp_heads.items():
            out = head(features)  # (B, 2)
            cyp_results[f"{enzyme}_inhibition"] = out[:, 0]
            cyp_results[f"{enzyme}_induction"] = out[:, 1]
            all_cyp_probs.append(out)

        pgp = self.pgp_head(features).squeeze(-1)

        # Aggregate interaction risk
        all_probs = torch.cat([p.max(dim=-1, keepdim=True).values for p in all_cyp_probs], dim=-1)
        interaction_risk = torch.cat([all_probs, pgp.unsqueeze(-1)], dim=-1).max(dim=-1).values

        has_interaction_alarm = (
            interaction_risk > self.cfg.interaction_alarm_threshold
        ).float()

        return {
            **{f"ddi_{k}": v for k, v in cyp_results.items()},
            "ddi_pgp": pgp,
            "ddi_overall_risk": interaction_risk,
            "ddi_has_alarm": has_interaction_alarm,
        }


# ═══════════════════════════════════════════════════════════════════════
# 6. PHYTO ADMET / BIOAVAILABILITY MODEL
# ═══════════════════════════════════════════════════════════════════════

class PhytoADMETModel(nn.Module):
    """
    Predict Absorption, Distribution, Metabolism, Excretion, and
    Bioavailability — the REALITY CHECK against in-vitro hype.

    Many phytochemicals show activity in cell culture but have
    negligible oral bioavailability, meaning they cannot reach
    their target in the body at relevant concentrations.
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
        )

        self.admet_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(cfg.latent_dim // 2, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ) for task in cfg.admet_tasks
        })

    def forward(self, mixture_repr: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(mixture_repr)

        admet = {}
        for task, head in self.admet_heads.items():
            admet[task] = head(features).squeeze(-1)

        bioavailability = admet["bioavailability"]
        low_bioavailability_flag = (
            bioavailability < self.cfg.bioavailability_min
        ).float()

        return {
            **{f"admet_{k}": v for k, v in admet.items()},
            "admet_low_bioavailability_flag": low_bioavailability_flag,
        }


# ═══════════════════════════════════════════════════════════════════════
# 7. PHYTO SYNERGY MODEL
# ═══════════════════════════════════════════════════════════════════════

class PhytoSynergyModel(nn.Module):
    """
    Model synergy and antagonism between mixture compounds.

    Uses two classical pharmacological reference models:
      • Bliss independence  — expected if compounds act independently
      • Loewe additivity    — expected if compounds act on same target

    Deviation from these references indicates true synergy/antagonism.
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        # Pairwise interaction scorer
        self.pair_scorer = nn.Sequential(
            nn.Linear(cfg.compound_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # -1 (antagonism) to +1 (synergy)
        )

        # Mixture-level synergy estimate
        self.synergy_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [synergy, additivity, antagonism]
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        compound_embeddings: Tensor,  # (B, K, D)
        concentrations: Tensor,       # (B, K)
        mixture_repr: Tensor,         # (B, latent_dim)
    ) -> Dict[str, Tensor]:
        B, K, D = compound_embeddings.shape

        # Pairwise synergy scores (sample of pairs for efficiency)
        # For each batch, compute mean pairwise interaction
        pair_scores = []
        for i in range(min(K, 8)):
            for j in range(i + 1, min(K, 8)):
                pair = torch.cat([
                    compound_embeddings[:, i],
                    compound_embeddings[:, j],
                ], dim=-1)
                score = self.pair_scorer(pair).squeeze(-1)
                # Weight by concentrations
                weight = concentrations[:, i] * concentrations[:, j]
                pair_scores.append(score * weight)

        if pair_scores:
            pairwise = torch.stack(pair_scores, dim=-1).sum(dim=-1)  # (B,)
        else:
            pairwise = torch.zeros(B, device=mixture_repr.device)

        # Mixture-level classification
        synergy_probs = self.synergy_head(mixture_repr)  # (B, 3)

        return {
            "synergy_pairwise_score": pairwise,
            "synergy_class_probs": synergy_probs,
            "synergy_class": synergy_probs.argmax(dim=-1),  # 0=syn,1=add,2=ant
        }


# ═══════════════════════════════════════════════════════════════════════
# 8. EVIDENCE QUALITY GRADER (ANTI-MARKETING CORE)
# ═══════════════════════════════════════════════════════════════════════

class EvidenceQualityGrader(nn.Module):
    """
    Grade health claims by evidence level and detect marketing overclaims.

    This is the ANTI-MARKETING-HYPE component. It takes a claim's
    feature representation and:
      • Assigns an evidence level (0-7)
      • Classifies claim type
      • Detects overclaim risk (claim strength >> evidence strength)
      • Flags dangerous patterns (cure-all, disease-cure without RCT)

    The overclaim detector compares the IMPLIED efficacy of the claim
    against the AVAILABLE evidence, producing a mismatch score.
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        # Claim encoder (in production, from text; here from features)
        self.claim_encoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim // 2),
            nn.GELU(),
        )

        # Evidence level estimator (0-7, continuous proxy)
        self.evidence_estimator = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # scaled to [0,1], then mapped to 0-7
        )

        # Claim type classifier
        self.claim_classifier = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, len(ClaimType)),
            nn.Softmax(dim=-1),
        )

        # Implied efficacy of the claim (how strong the claim sounds)
        self.claim_strength = nn.Sequential(
            nn.Linear(cfg.latent_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, claim_features: Tensor) -> Dict[str, Tensor]:
        encoded = self.claim_encoder(claim_features)

        # Evidence level (0-7)
        evidence_raw = self.evidence_estimator(encoded).squeeze(-1)
        evidence_level = evidence_raw * 7.0  # scale to 0-7

        # Claim type
        claim_type_probs = self.claim_classifier(encoded)
        claim_type = claim_type_probs.argmax(dim=-1)

        # Claim strength (how strong the claim sounds)
        strength = self.claim_strength(encoded).squeeze(-1) * 7.0

        # OVERCLAIM DETECTION
        # Mismatch = claim strength - evidence level (both 0-7 scale)
        overclaim_gap = F.relu(strength - evidence_level)
        is_overclaim = (overclaim_gap > 2.0).float()  # significant gap

        # DANGEROUS CLAIM detection:
        # disease-cure or cure-all claims with low evidence
        claim_type_float = claim_type.float()
        is_dangerous_claim_type = (claim_type_float >= 3.0).float()  # CURE, CURES_ALL
        low_evidence = (evidence_level < 4.0).float()  # below Phase I
        is_dangerous = is_dangerous_claim_type * low_evidence

        return {
            "evidence_level": evidence_level,
            "evidence_level_int": evidence_level.round(),
            "claim_type_probs": claim_type_probs,
            "claim_type": claim_type,
            "claim_strength": strength,
            "overclaim_gap": overclaim_gap,
            "is_overclaim": is_overclaim,
            "is_dangerous_claim": is_dangerous,
        }


# ═══════════════════════════════════════════════════════════════════════
# 9. PHYTO SAFETY GATE
# ═══════════════════════════════════════════════════════════════════════

class PhytoSafetyGate(nn.Module):
    """
    Final safety gate that integrates all signals and decides whether
    a phytochemical product should proceed toward any approval pathway.

    The gate is PESSIMISTIC by design (safety-first):
      • Any toxicity alarm → BLOCK
      • Any drug interaction alarm → WARN/BLOCK
      • Low bioavailability → FLAG (likely ineffective)
      • Overclaim / dangerous claim → BLOCK from marketing approval
      • High complexity without sufficient evidence → REQUIRE_MORE_STUDY

    Output is a decision + detailed reasoning, not just a pass/fail.
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        tox_out: Dict[str, Tensor],
        ddi_out: Dict[str, Tensor],
        admet_out: Dict[str, Tensor],
        evidence_out: Dict[str, Tensor],
        complexity_out: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        B = tox_out["tox_overall_risk"].shape[0]
        device = tox_out["tox_overall_risk"].device

        # Gather boolean signals
        tox_alarm = tox_out["tox_has_alarm"]              # (B,)
        ddi_alarm = ddi_out["ddi_has_alarm"]              # (B,)
        low_bio = admet_out["admet_low_bioavailability_flag"]  # (B,)
        overclaim = evidence_out["is_overclaim"]          # (B,)
        dangerous_claim = evidence_out["is_dangerous_claim"]  # (B,)

        # Decision logic (pessimistic / safety-first)
        # BLOCK if toxicity or dangerous claim
        block = ((tox_alarm > 0.5) | (dangerous_claim > 0.5)).float()
        # WARN if interaction or overclaim
        warn = ((ddi_alarm > 0.5) | (overclaim > 0.5) | (low_bio > 0.5)).float()
        # warn should not override block
        warn = warn * (1.0 - block)
        # APPROVE_FOR_FURTHER_STUDY only if neither block nor warn
        proceed = (1.0 - block) * (1.0 - warn)

        # Encode decision as one-hot-like: [block, warn, proceed]
        decision = torch.stack([block, warn, proceed], dim=-1)  # (B, 3)
        decision_label = decision.argmax(dim=-1)  # 0=block,1=warn,2=proceed

        # Complexity vs evidence check
        complexity = complexity_out["total_complexity"]
        evidence = evidence_out["evidence_level"]
        # High complexity needs higher evidence bar
        requires_more_study = (
            (complexity > 4.0) & (evidence < 5.0)
        ).float()

        return {
            "gate_decision": decision,
            "gate_label": decision_label,  # 0=BLOCK,1=WARN,2=PROCEED
            "gate_block": block,
            "gate_warn": warn,
            "gate_proceed": proceed,
            "gate_requires_more_study": requires_more_study,
            "gate_n_safety_flags": (
                tox_alarm + ddi_alarm + low_bio + overclaim + dangerous_claim
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# 10. UNIFIED PHYTOCHEM ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class PhytochemOrchestrator(nn.Module):
    """
    Complete phytochemistry analysis pipeline.

    Flow:
      fingerprint+concentrations
        → CompoundEncoder → MixtureEncoder
        → ComplexityScorer
        → ToxicityPredictor      (safety)
        → InteractionPredictor   (safety)
        → ADMETModel             (reality check)
        → SynergyModel           (combination)
        → EvidenceGrader         (anti-marketing)
        → SafetyGate             (final decision)
    """

    def __init__(self, cfg: PhytochemConfig):
        super().__init__()
        self.cfg = cfg

        self.compound_encoder = PhytoCompoundEncoder(cfg)
        self.mixture_encoder = PhytoMixtureEncoder(cfg)
        self.complexity_scorer = MolecularComplexityScorer(cfg)
        self.toxicity = PhytoToxicityPredictor(cfg)
        self.interaction = PhytoDrugInteractionPredictor(cfg)
        self.admet = PhytoADMETModel(cfg)
        self.synergy = PhytoSynergyModel(cfg)
        self.evidence_grader = EvidenceQualityGrader(cfg)
        self.safety_gate = PhytoSafetyGate(cfg)

        # Claim feature projector (maps mixture repr to claim space)
        self.claim_projector = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
        )

    def forward(
        self,
        fingerprints: Tensor,      # (B, K, fingerprint_dim)
        concentrations: Tensor,    # (B, K)
    ) -> Dict[str, Any]:
        B, K, _ = fingerprints.shape
        out: Dict[str, Any] = {}

        # 1. Encode each compound
        fp_flat = fingerprints.reshape(B * K, -1)
        comp_out = self.compound_encoder(fp_flat)
        compound_emb = comp_out["compound_embedding"].reshape(B, K, -1)

        # 2. Encode mixture
        mix_out = self.mixture_encoder(compound_emb, concentrations)
        mixture_repr = mix_out["mixture_representation"]
        out.update({f"mix_{k}": v for k, v in mix_out.items()})

        # 3. Complexity
        comp_out2 = self.complexity_scorer(
            mixture_repr, mix_out["diversity_index"], mix_out["estimated_active_count"]
        )
        out.update({f"complexity_{k}": v for k, v in comp_out2.items()})

        # 4. Toxicity (SAFETY FIRST)
        tox_out = self.toxicity(mixture_repr)
        out.update(tox_out)

        # 5. Drug interactions
        ddi_out = self.interaction(mixture_repr)
        out.update(ddi_out)

        # 6. ADMET / bioavailability
        admet_out = self.admet(mixture_repr)
        out.update(admet_out)

        # 7. Synergy
        syn_out = self.synergy(compound_emb, concentrations, mixture_repr)
        out.update({f"synergy_{k}": v for k, v in syn_out.items()})

        # 8. Evidence grading (anti-marketing)
        claim_features = self.claim_projector(mixture_repr)
        evidence_out = self.evidence_grader(claim_features)
        out.update({f"evidence_{k}": v for k, v in evidence_out.items()})

        # 9. Safety gate (final decision)
        gate_out = self.safety_gate(
            tox_out, ddi_out, admet_out, evidence_out, comp_out2
        )
        out.update(gate_out)

        return out


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Phytochemistry Module — Smoke Test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)

    cfg = PhytochemConfig(
        latent_dim=128,
        compound_dim=64,
        fingerprint_dim=256,
        mixture_max_compounds=8,
        device=str(device),
    )

    model = PhytochemOrchestrator(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_par:,}")

    model.train()
    B, K = 2, 8  # batch, compounds per mixture

    # Dummy molecular fingerprints + concentrations
    fingerprints = torch.rand(B, K, cfg.fingerprint_dim, device=device)
    concentrations = torch.rand(B, K, device=device) + 0.1

    # Forward
    out = model(fingerprints, concentrations)

    # Backward (use a composite objective)
    loss = (
        out["tox_overall_risk"].mean()
        - out["evidence_level"].mean()
        + out["complexity_total_complexity"].mean()
    )
    loss.backward()

    print(f"\n  ── Mixture Analysis ──")
    print(f"    Estimated active compounds : {out['mix_estimated_active_count'].mean().item():.2f}")
    print(f"    Diversity index (Shannon)  : {out['mix_diversity_index'].mean().item():.3f}")

    print(f"\n  ── Molecular Complexity ──")
    print(f"    Total complexity           : {out['complexity_total_complexity'].mean().item():.3f}")
    print(f"    Complexity category        : {out['complexity_complexity_category'].mean().item():.1f} (0=low,1=mod,2=high)")

    print(f"\n  ── TOXICITY (Safety First) ──")
    print(f"    Overall tox risk           : {out['tox_overall_risk'].mean().item():.4f}")
    print(f"    hERG risk                  : {out['tox_herg'].mean().item():.4f}")
    print(f"    Hepatotoxicity risk        : {out['tox_hepatotoxicity'].mean().item():.4f}")
    print(f"    Genotoxicity risk          : {out['tox_genotoxicity'].mean().item():.4f}")
    print(f"    Toxicity alarms            : {out['tox_n_alarms'].mean().item():.1f}")

    print(f"\n  ── Drug Interactions ──")
    print(f"    CYP3A4 inhibition          : {out['ddi_CYP3A4_inhibition'].mean().item():.4f}")
    print(f"    CYP3A4 induction           : {out['ddi_CYP3A4_induction'].mean().item():.4f}")
    print(f"    Interaction risk           : {out['ddi_overall_risk'].mean().item():.4f}")

    print(f"\n  ── ADMET / Bioavailability ──")
    print(f"    Bioavailability            : {out['admet_bioavailability'].mean().item():.4f}")
    print(f"    Low-bioavailability flag   : {out['admet_low_bioavailability_flag'].mean().item():.2f}")

    print(f"\n  ── Synergy ──")
    print(f"    Pairwise synergy score     : {out['synergy_synergy_pairwise_score'].mean().item():.4f}")
    print(f"    Synergy class              : {out['synergy_synergy_class'].tolist()} (0=syn,1=add,2=ant)")

    print(f"\n  ── Evidence Grading (Anti-Marketing) ──")
    print(f"    Evidence level             : {out['evidence_evidence_level'].mean().item():.2f} / 7")
    print(f"    Claim strength             : {out['evidence_claim_strength'].mean().item():.2f} / 7")
    print(f"    Overclaim gap              : {out['evidence_overclaim_gap'].mean().item():.3f}")
    print(f"    Is overclaim               : {out['evidence_is_overclaim'].mean().item():.2f}")
    print(f"    Is dangerous claim         : {out['evidence_is_dangerous_claim'].mean().item():.2f}")

    print(f"\n  ── SAFETY GATE (Final Decision) ──")
    print(f"    Gate label                 : {out['gate_label'].tolist()} (0=BLOCK,1=WARN,2=PROCEED)")
    print(f"    Requires more study        : {out['gate_requires_more_study'].mean().item():.2f}")
    print(f"    Num safety flags           : {out['gate_n_safety_flags'].mean().item():.1f}")

    print(f"\n  ── Loss ──")
    print(f"    Loss                       : {loss.item():.6f}")

    grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    print(f"\n  Autograd: {'PASS — FULL PHYTOCHEM PIPELINE' if grad_ok else 'FAIL'}")

    print("\n" + "=" * 72)
    print("  ⚠️  HONEST REMINDERS:")
    print("  • Natural ≠ Safe. Toxicity prediction is mandatory, not optional.")
    print("  • In-vitro activity ≠ human efficacy (bioavailability check).")
    print("  • Evidence grading is a heuristic — not a replacement for")
    print("    systematic review or regulatory evaluation.")
    print("  • This is a RESEARCH framework, not a clinical decision tool.")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
