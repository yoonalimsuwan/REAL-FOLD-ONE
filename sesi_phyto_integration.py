"""
=============================================================================
SESI PHYTOCHEMISTRY INTEGRATION MODULE v1.0
Target Prediction × Known Toxin Database × Advertising Claim Analysis
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Phyto-Target-Toxin-Claim Integration)

WHAT THIS MODULE PROVIDES
─────────────────────────
[I1] PhytoTargetPredictor
     Connects phytochemical compounds to ProteinFoldingOrchestrator
     to predict molecular targets and binding affinities.

[I2] KnownPlantToxinDatabase
     Curated database of known plant toxins with:
     • Chemical structure (SMILES)
     • Mechanism of toxicity
     • Severity level
     • Associated plant families
     • Regulatory status
     Used as HARD BLOCK in the Safety Gate.

[I3] AdvertisingClaimAnalyzer
     Analyzes marketing/label text to detect:
     • Disease-cure claims without evidence
     • "Natural = safe" fallacy
     • Unsubstantiated superiority claims
     • Misleading scientific language
     Generates regulatory warning reports.

[I4] PhytoIntegratedSafetySystem
     Unified pipeline connecting all three components.

HONEST SCOPE
────────────
• Target prediction is computational; wet-lab validation required.
• Toxin database covers well-documented toxins but is NOT exhaustive.
• Claim analysis uses pattern matching + evidence grading; it does not
  replace legal/regulatory review.
• This is a RESEARCH + CONSUMER PROTECTION tool, not a clinical system.
=============================================================================
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.phyto_integration")

__all__ = [
    "PhytoIntegrationConfig",
    # [I1] Target prediction
    "PhytoTargetPredictor",
    # [I2] Toxin database
    "KnownPlantToxinDatabase",
    "ToxinRecord",
    "ToxinSeverity",
    # [I3] Claim analysis
    "AdvertisingClaimAnalyzer",
    "ClaimVerdict",
    # [I4] Integrated system
    "PhtoIntegratedSafetySystem",
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
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PhytoIntegrationConfig:
    # ── Dimensions ──────────────────────────────────────────────────────
    latent_dim: int = 256
    compound_dim: int = 128
    protein_dim: int = 256
    fingerprint_dim: int = 1024

    # ── Target prediction ───────────────────────────────────────────────
    max_targets: int = 20              # number of candidate targets
    binding_threshold: float = 0.3     # minimum binding score to report

    # ── Toxin database ──────────────────────────────────────────────────
    # (built-in, no external files needed)

    # ── Claim analysis ──────────────────────────────────────────────────
    language: str = "en"               # "en" | "th" | "auto"

    # ── Device ──────────────────────────────────────────────────────────
    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════
# [I2] KNOWN PLANT TOXIN DATABASE
# ═══════════════════════════════════════════════════════════════════════

class ToxinSeverity(Enum):
    LOW = 1          # mild symptoms, reversible
    MODERATE = 2     # significant symptoms, may need treatment
    HIGH = 3         # serious organ damage
    CRITICAL = 4     # potentially fatal
    BANNED = 5       # banned in food/supplements by most regulators


@dataclass
class ToxinRecord:
    """A single known plant toxin."""
    name: str
    aliases: List[str]
    smiles: str                          # simplified structure identifier
    mechanism: str                       # mechanism of toxicity
    severity: ToxinSeverity
    target_organs: List[str]
    associated_plants: List[str]
    regulatory_status: str
    antidote: Optional[str] = None
    notes: str = ""


class KnownPlantToxinDatabase:
    """
    Curated database of well-documented plant toxins.

    HONEST SCOPE:
    • This covers KNOWN, well-documented toxins. It is NOT exhaustive.
    • Absence from this list does NOT mean a compound is safe.
    • New toxins are discovered regularly; this must be updated.
    • Always combine with computational toxicity prediction AND wet-lab testing.

    Sources: WHO, EMA Herbal Monographs, FDA warnings, published toxicology.
    """

    def __init__(self):
        self._toxins: Dict[str, ToxinRecord] = {}
        self._build_database()

    def _build_database(self):
        """Populate the toxin database with well-known entries."""
        records = [
            ToxinRecord(
                name="Aristolochic acid",
                aliases=["aristolochic acid I", "aristolochic acid II", "AA-I", "AA-II"],
                smiles="O=C(O)c1cc2c(cc1[N+](=O)[O-])c1ccccc1C2",
                mechanism="DNA adduct formation → mutations in TP53; "
                          "mitochondrial damage in renal tubular cells",
                severity=ToxinSeverity.BANNED,
                target_organs=["kidney", "urothelium"],
                associated_plants=["Aristolochia", "Asarum", "Bragantia"],
                regulatory_status="BANNED worldwide (WHO, FDA, EMA). "
                                  "Causes aristolochic acid nephropathy + urothelial cancer.",
                notes="Found in some traditional medicines (e.g., certain Aristolochia species). "
                      "Zero-tolerance in food and supplements.",
            ),
            ToxinRecord(
                name="Pyrrolizidine alkaloids",
                aliases=["PA", "retrorsine", "senecionine", "monocrotaline", "lasiocarpine"],
                smiles="C1CC2C(C1)N(C)CC2",  # simplified core
                mechanism="Metabolic activation by CYP3A4 → dehydropyrrolizidine "
                          "alkaloids → DNA cross-linking → hepatic sinusoidal "
                          "obstruction syndrome (veno-occlusive disease)",
                severity=ToxinSeverity.HIGH,
                target_organs=["liver", "lungs"],
                associated_plants=["Senecio", "Heliotropium", "Crotalaria", "Symphytum (comfrey)", "Borago"],
                regulatory_status="Restricted. EMA: limit 1 µg PA/day for herbal products. "
                                  "Comfrey banned for internal use in many countries.",
                notes="Cumulative hepatotoxicity. Often contaminates herbal teas and honey.",
            ),
            ToxinRecord(
                name="Aconitine",
                aliases=["aconitine", "mesaconitine", "hypaconitine"],
                smiles="CC(=O)Oc1c(C2CN3CC(C2)C(C3)(O)C(OC)OC)",
                mechanism="Persistent activation of voltage-gated sodium channels → "
                          "cardiac arrhythmia (ventricular tachycardia/fibrillation), "
                          "neurotoxicity",
                severity=ToxinSeverity.CRITICAL,
                target_organs=["heart", "nervous system"],
                associated_plants=["Aconitum (monkshood, wolfsbane)", "Delphinium"],
                regulatory_status="Highly restricted. Fatal dose ~2-6 mg. "
                                  "Requires processing (boiling) to reduce toxicity.",
                antidote="Supportive care; atropine for bradycardia; "
                         "antiarrhythmics. No specific antidote.",
                notes="Used in some traditional medicines after proper processing. "
                      "Narrow therapeutic window.",
            ),
            ToxinRecord(
                name="Atropine / Scopolamine",
                aliases=["atropine", "scopolamine", "hyoscyamine", "tropane alkaloids"],
                smiles="CN1C2CCC1CC(OC(=O)C(CO)c3ccccc3)C2",
                mechanism="Competitive antagonism of muscarinic acetylcholine receptors → "
                          "anticholinergic syndrome (delirium, tachycardia, hyperthermia)",
                severity=ToxinSeverity.HIGH,
                target_organs=["nervous system", "heart", "eyes"],
                associated_plants=["Datura (jimsonweed)", "Atropa belladonna",
                                  "Hyoscyamus", "Brugmansia"],
                regulatory_status="Controlled. Atropine is a medicine but toxic in overdose.",
                antidote="Physostigmine (for severe central anticholinergic syndrome).",
                notes="'Blind as a bat, mad as a hatter, red as a beet, "
                      "hot as a hare, dry as a bone.'",
            ),
            ToxinRecord(
                name="Ricin",
                aliases=["ricin", "RCA"],
                smiles="protein_toxin",  # protein, not small molecule
                mechanism="Ribosome-inactivating protein (N-glycosidase) → "
                          "inhibits protein synthesis → cell death",
                severity=ToxinSeverity.CRITICAL,
                target_organs=["liver", "kidney", "GI tract", "respiratory"],
                associated_plants=["Ricinus communis (castor bean)"],
                regulatory_status="Controlled substance (biological weapon). "
                                  "Castor OIL is safe (ricin is water-soluble, not in oil).",
                notes="No antidote. Supportive care only. "
                      "Extremely toxic: LD50 ~1-10 µg/kg (parenteral).",
            ),
            ToxinRecord(
                name="Strychnine",
                aliases=["strychnine", "brucine"],
                smiles="O=C1Nc2ccccc2C2C3CCc4cc5c(c4C3C1C2)[nH]c1ccccc15",
                mechanism="Antagonism of glycine receptors in spinal cord → "
                          "unopposed excitatory neurotransmission → severe convulsions",
                severity=ToxinSeverity.CRITICAL,
                target_organs=["nervous system", "respiratory muscles"],
                associated_plants=["Strychnos nux-vomica", "Strychnos ignatii"],
                regulatory_status="Controlled. Historically used as rodenticide.",
                notes="Death by asphyxiation due to respiratory muscle spasm. "
                      "No specific antidote; supportive care with benzodiazepines.",
            ),
            ToxinRecord(
                name="Cardiac glycosides (Digitalis)",
                aliases=["digoxin", "digitoxin", "ouabain", "oleandrin"],
                smiles="CC1C2CC(O)C3(C)C(CCC4C3CCC(O)C4)C2CC1",
                mechanism="Inhibition of Na+/K+-ATPase → increased intracellular Na+ → "
                          "increased Ca2+ → cardiac arrhythmia",
                severity=ToxinSeverity.HIGH,
                target_organs=["heart"],
                associated_plants=["Digitalis (foxglove)", "Nerium oleander",
                                  "Strophanthus", "Convallaria (lily of the valley)"],
                regulatory_status="Digoxin is a medicine (narrow therapeutic index). "
                                  "Oleander is toxic and not for self-medication.",
                antidote="Digoxin-specific antibody fragments (Digibind/DigiFab).",
                notes="Very narrow therapeutic window. Drug interactions common.",
            ),
            ToxinRecord(
                name="Colchicine",
                aliases=["colchicine", "colchiceine"],
                smiles="CC(=O)Nc1cccc2cc(OC)c(OC)cc12",
                mechanism="Binds tubulin → inhibits microtubule polymerization → "
                          "arrests mitosis; also affects neutrophil function",
                severity=ToxinSeverity.HIGH,
                target_organs=["GI tract", "bone marrow", "multi-organ"],
                associated_plants=["Colchicum autumnale (autumn crocus)", "Gloriosa superba"],
                regulatory_status="Medicine for gout/FMF but highly toxic in overdose. "
                                  "No antidote.",
                notes="Narrow therapeutic window. Fatal in overdose (>0.5 mg/kg). "
                      "Multi-organ failure.",
            ),
            ToxinRecord(
                name="Aflatoxin",
                aliases=["aflatoxin B1", "AFB1", "aflatoxin B2", "aflatoxin G1"],
                smiles="O=C1Oc2c3c(c1)C1OC=C(C)C1c1c3oc2-c2ccccc12",
                mechanism="Metabolic activation by CYP3A4/1A2 → aflatoxin-8,9-epoxide → "
                          "DNA adducts (N7-guanine) → TP53 mutations → hepatocellular carcinoma",
                severity=ToxinSeverity.CRITICAL,
                target_organs=["liver"],
                associated_plants=["Contaminated peanuts, corn, tree nuts, dried spices "
                                  "(produced by Aspergillus flavus/parasiticus fungi)"],
                regulatory_status="Strict limits in food (e.g., EU: 2 µg/kg AFB1, "
                                  "4 µg/kg total). IARC Group 1 carcinogen.",
                notes="Not a plant toxin per se, but a common contaminant in herbal products.",
            ),
            ToxinRecord(
                name="Thujone",
                aliases=["alpha-thujone", "beta-thujone"],
                smiles="CC1CC2C1(C)C2(C)C(=O)C",
                mechanism="GABA-A receptor antagonist → CNS excitation, seizures at high dose",
                severity=ToxinSeverity.MODERATE,
                target_organs=["nervous system"],
                associated_plants=["Artemisia absinthium (wormwood)", "Thuja", "Salvia officinalis (sage)"],
                regulatory_status="EU limits: 35 mg/kg in absinthe, 10 mg/kg in other alcoholic drinks.",
                notes="Historically blamed for 'absinthism' but modern research "
                      "shows alcohol was the main culprit.",
            ),
            ToxinRecord(
                name="Safrole",
                aliases=["safrole", "isosafrole"],
                smiles="C=CCc1ccc2OCOc2c1",
                mechanism="Metabolic activation → 1'-hydroxysafrole → DNA adducts → "
                          "hepatocarcinogenicity",
                severity=ToxinSeverity.HIGH,
                target_organs=["liver"],
                associated_plants=["Sassafras", "Ocotea pretiosa (sassafras oil)",
                                  "Piper betle (betel leaf, trace amounts)"],
                regulatory_status="Banned as food additive by FDA (1960). "
                                  "IARC Group 2B (possibly carcinogenic).",
                notes="Also a precursor for illicit drug synthesis (MDMA), hence controlled.",
            ),
            ToxinRecord(
                name="Coumarin",
                aliases=["coumarin", "1,2-benzopyrone"],
                smiles="O=c1ccc2ccccc2o1",
                mechanism="Hepatotoxicity in susceptible individuals (CYP2A6 poor metabolizers); "
                          "NOT the same as warfarin-type anticoagulants",
                severity=ToxinSeverity.MODERATE,
                target_organs=["liver"],
                associated_plants=["Cinnamomum cassia (cassia cinnamon)", "Dipteryx odorata (tonka bean)"],
                regulatory_status="EU TDI: 0.1 mg/kg body weight. "
                                  "Cassia cinnamon has ~1% coumarin vs Ceylon ~0.004%.",
                notes="Common in 'cinnamon' supplements. Cassia vs Ceylon distinction matters.",
            ),
            ToxinRecord(
                name="Hydrocyanic acid / Cyanogenic glycosides",
                aliases=["amygdalin", "laetrile", "linamarin", "dhurrin", "HCN"],
                smiles="N#CC(O)c1ccccc1",  # mandelonitrile (amygdalin aglycone)
                mechanism="Beta-glucosidase releases HCN → inhibits cytochrome c oxidase → "
                          "cellular hypoxia",
                severity=ToxinSeverity.HIGH,
                target_organs=["all organs (cellular hypoxia)", "heart", "brain"],
                associated_plants=["Prunus (apricot/bitter almond kernels)", "Manihot esculenta (cassava)",
                                  "Sorghum", "Linum (flax/linseed)", "Sambucus (elderberry, raw)"],
                regulatory_status="Amygdalin/laetrile NOT approved as cancer treatment "
                                  "(FDA, EMA). Cassava must be properly processed.",
                antidote="Hydroxocobalamin, sodium thiosulfate, or "
                         "sodium nitrite + sodium thiosulfate.",
                notes="Bitter almonds contain ~5% amygdalin. 'Vitamin B17' is a misnomer.",
            ),
            ToxinRecord(
                name="Heavy metals (contamination)",
                aliases=["arsenic", "lead", "mercury", "cadmium"],
                smiles="inorganic",
                mechanism="Varies: arsenic (oxidative stress, enzyme inhibition), "
                          "lead (neurotoxicity, anemia), mercury (neuro/nephrotoxicity), "
                          "cadmium (renal tubular damage)",
                severity=ToxinSeverity.HIGH,
                target_organs=["kidney", "nervous system", "liver", "bone"],
                associated_plants=["Contamination of ANY herbal product from soil, "
                                  "processing, or intentional adulteration"],
                regulatory_status="Strict limits in pharmacopoeias (USP, EP, Thai Herbal Pharmacopoeia). "
                                  "WHO guidelines for heavy metals in herbal medicines.",
                notes="Not plant-derived toxins but common contaminants. "
                      "Testing for heavy metals is mandatory in quality control.",
            ),
            ToxinRecord(
                name="Berberine (high dose)",
                aliases=["berberine", "berberine sulfate"],
                smiles="COc1ccc2cc3[n+](cc4ccc(OC)c(OC)c4c3OC)c2c1",
                mechanism="At high doses: GI upset, potential drug interactions (CYP3A4, P-gp); "
                          "contraindicated in pregnancy (uterine stimulant) and neonates (kernicterus)",
                severity=ToxinSeverity.LOW,
                target_organs=["GI tract", "liver (interactions)"],
                associated_plants=["Berberis", "Coptis (goldthread)", "Phellodendron", "Hydrastis (goldenseal)"],
                regulatory_status="Sold as supplement in many countries. "
                                  "Contraindicated in pregnancy and neonates.",
                notes="Generally safe at recommended doses but significant drug interactions.",
            ),
        ]

        for rec in records:
            self._toxins[rec.name.lower()] = rec
            for alias in rec.aliases:
                self._toxins[alias.lower()] = rec

        logger.info(f"KnownPlantToxinDatabase loaded: {len(self.unique_toxins)} unique toxins")

    @property
    def unique_toxins(self) -> List[ToxinRecord]:
        seen = set()
        result = []
        for rec in self._toxins.values():
            if rec.name not in seen:
                seen.add(rec.name)
                result.append(rec)
        return result

    def search(self, query: str) -> Optional[ToxinRecord]:
        """Search for a toxin by name or alias (case-insensitive)."""
        return self._toxins.get(query.lower())

    def screen_compounds(self, compound_names: List[str]) -> List[Dict[str, Any]]:
        """
        Screen a list of compound names against the toxin database.

        Returns list of matches with severity info.
        """
        matches = []
        for name in compound_names:
            rec = self.search(name)
            if rec is not None:
                matches.append({
                    "query": name,
                    "toxin_name": rec.name,
                    "severity": rec.severity.name,
                    "severity_value": rec.severity.value,
                    "mechanism": rec.mechanism,
                    "target_organs": rec.target_organs,
                    "regulatory_status": rec.regulatory_status,
                    "is_banned": rec.severity == ToxinSeverity.BANNED,
                    "is_critical_or_above": rec.severity.value >= ToxinSeverity.CRITICAL.value,
                })
        return matches

    def get_hard_block_toxins(self) -> List[str]:
        """Return names of toxins that should trigger an automatic BLOCK."""
        return [
            rec.name for rec in self.unique_toxins
            if rec.severity.value >= ToxinSeverity.CRITICAL.value
        ]

    def summary(self) -> Dict[str, Any]:
        """Summary statistics of the database."""
        by_severity = {}
        for rec in self.unique_toxins:
            key = rec.severity.name
            by_severity[key] = by_severity.get(key, 0) + 1
        return {
            "total_unique_toxins": len(self.unique_toxins),
            "by_severity": by_severity,
            "hard_block_count": len(self.get_hard_block_toxins()),
        }


# ═══════════════════════════════════════════════════════════════════════
# [I1] PHYTO TARGET PREDICTOR
# ═══════════════════════════════════════════════════════════════════════

class PhytoTargetPredictor(nn.Module):
    """
    Predicts protein targets for phytochemical compounds.

    This connects the phytochemistry module to ProteinFoldingOrchestrator
    for structure-based target prediction.

    Pipeline:
      1. Encode phytochemical compound (fingerprint → latent)
      2. Query against a panel of protein targets
      3. Predict binding affinity for each target
      4. Rank targets by predicted affinity
      5. Optionally invoke ProteinFoldingOrchestrator to model
         the target protein structure (for structure-based refinement)

    HONEST SCOPE:
    • Binding prediction is approximate; real docking requires
      tools like AutoDock Vina, GOLD, or FEP calculations.
    • The protein target panel must be populated with real protein
      embeddings for meaningful predictions.
    • ProteinFoldingOrchestrator integration is structural (optional).
    """

    def __init__(self, cfg: PhytoIntegrationConfig):
        super().__init__()
        self.cfg = cfg

        # Compound encoder
        self.compound_encoder = nn.Sequential(
            nn.Linear(cfg.fingerprint_dim, cfg.compound_dim),
            nn.GELU(),
            nn.Linear(cfg.compound_dim, cfg.latent_dim),
        )

        # Protein target encoder (shared for all targets)
        self.target_encoder = nn.Sequential(
            nn.Linear(cfg.protein_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )

        # Binding affinity predictor (compound × target)
        self.binding_net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # binding score in [0, 1]
        )

        # Interaction type classifier
        self.interaction_classifier = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 5),  # [H-bond, hydrophobic, ionic, pi-stacking, covalent]
            nn.Softmax(dim=-1),
        )

    def encode_compound(self, fingerprint: Tensor) -> Tensor:
        """Encode compound fingerprint to latent space."""
        return self.compound_encoder(fingerprint)

    def encode_target(self, target_features: Tensor) -> Tensor:
        """Encode protein target features."""
        return self.target_encoder(target_features)

    def predict_binding(
        self,
        compound_latent: Tensor,  # (B, latent_dim)
        target_latents: Tensor,   # (B, n_targets, latent_dim)
    ) -> Dict[str, Tensor]:
        """
        Predict binding affinity between compound and multiple targets.

        Args:
            compound_latent: (B, latent_dim)
            target_latents: (B, n_targets, latent_dim)

        Returns:
            dict with binding_scores, interaction_types, ranked_targets
        """
        B, T, D = target_latents.shape

        # Expand compound to match targets
        compound_expanded = compound_latent.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)

        # Concatenate compound and target features
        pair_features = torch.cat([compound_expanded, target_latents], dim=-1)  # (B, T, 2D)

        # Predict binding scores
        binding_scores = self.binding_net(pair_features).squeeze(-1)  # (B, T)

        # Predict interaction types
        interaction_types = self.interaction_classifier(pair_features)  # (B, T, 5)

        # Rank targets by binding score (descending)
        ranked_indices = binding_scores.argsort(dim=-1, descending=True)

        # Filter by threshold
        significant_mask = binding_scores > self.cfg.binding_threshold

        return {
            "binding_scores": binding_scores,
            "interaction_types": interaction_types,
            "ranked_target_indices": ranked_indices,
            "significant_targets_mask": significant_mask,
            "n_significant_targets": significant_mask.sum(dim=-1),
        }

    def forward(
        self,
        fingerprint: Tensor,         # (B, fingerprint_dim)
        target_features: Tensor,     # (B, n_targets, protein_dim)
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full target prediction pipeline.

        Args:
            fingerprint: (B, fingerprint_dim) compound fingerprint
            target_features: (B, n_targets, protein_dim) protein target features
            target_names: optional list of target names for reporting

        Returns:
            dict with binding predictions and ranked targets
        """
        compound_latent = self.encode_compound(fingerprint)
        target_latents = self.encode_target(target_features)

        binding_out = self.predict_binding(compound_latent, target_latents)

        # Build ranked target report (for the first batch item)
        ranked_report = []
        if target_names is not None:
            for idx in binding_out["ranked_target_indices"][0].tolist():
                if idx < len(target_names):
                    score = binding_out["binding_scores"][0, idx].item()
                    ranked_report.append({
                        "target": target_names[idx],
                        "binding_score": score,
                        "significant": score > self.cfg.binding_threshold,
                    })

        binding_out["ranked_report"] = ranked_report
        return binding_out


# ═══════════════════════════════════════════════════════════════════════
# [I3] ADVERTISING CLAIM ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class ClaimVerdict(Enum):
    """Verdict for an advertising claim."""
    ACCEPTABLE = auto()           # within regulatory norms
    NEEDS_QUALIFIER = auto()      # needs disclaimer/qualification
    MISLEADING = auto()           # misleading but not dangerous
    DANGEROUS = auto()            # potentially harmful to consumers
    FRAUDULENT = auto()           # clear fraud / illegal claim


class AdvertisingClaimAnalyzer:
    """
    Analyzes advertising/marketing text for problematic claims
    about phytochemical/herbal products.

    Detection categories:
      1. Disease cure/treatment claims (illegal without drug approval)
      2. "Natural = safe" fallacy
      3. Unsubstantiated superiority claims
      4. False scientific authority
      5. "No side effects" claims
      6. Panacea / cure-all claims
      7. Urgency / fear-based marketing

    HONEST SCOPE:
    • Pattern-based detection; not a full NLP system.
    • Supports English and basic Thai patterns.
    • Does not replace legal/regulatory review.
    • Designed as a CONSUMER PROTECTION screening tool.
    """

    # ── English patterns ────────────────────────────────────────────────
    EN_CURE_PATTERNS = [
        r"\bcures?\b.*\b(cancer|diabetes|HIV|AIDS|heart\s*disease|alzheimer|parkinson)",
        r"\bheals?\b.*\b(cancer|tumor|tumour)",
        r"\beliminates?\b.*\b(cancer|disease|virus)",
        r"\b100\s*%\s*(effective|cure|guarantee)",
        r"\bproven\s*(cure|remedy)\b",
    ]

    EN_NATURAL_SAFE_PATTERNS = [
        r"\bnatural\b.*\bsafe\b",
        r"\bnatural\b.*\bno\s*side\s*effects?\b",
        r"\b100\s*%\s*natural\b.*\b(safe|harmless)",
        r"\bchemical[- ]?free\b",
        r"\bno\s*chemicals?\b",
    ]

    EN_SUPERIORITY_PATTERNS = [
        r"\b(most|best|only|first|unique)\b.*\b(cure|treatment|remedy|solution)",
        r"\bbetter\s*than\b.*\b(chemo|radiation|surgery|medicine|drug)",
        r"\bsuperior\s*to\b.*\b(conventional|modern|western)\b",
        r"\breplaces?\b.*\b(chemo|chemotherapy|medicine|drug)",
    ]

    EN_NO_SIDE_EFFECT_PATTERNS = [
        r"\bno\s*side\s*effects?\b",
        r"\bzero\s*side\s*effects?\b",
        r"\bwithout\s*any\s*side\s*effects?\b",
        r"\bcompletely\s*safe\b",
        r"\babsolutely\s*safe\b",
        r"\bno\s*risks?\b",
    ]

    EN_PANACEA_PATTERNS = [
        r"\bcures?\s*(all|every)\b",
        r"\btreats?\s*(all|every|any)\s*(disease|condition|illness)",
        r"\bmiracle\s*(cure|remedy|drug|supplement)",
        r"\bwonder\s*(drug|cure|supplement)",
        r"\bpanacea\b",
        r"\bcure[- ]?all\b",
    ]

    EN_FALSE_AUTHORITY_PATTERNS = [
        r"\bdoctors?\s*(recommend|prescribe|approve)s?\b.*\b(this|our|the)\b",
        r"\bclinically\s*proven\b(?!\s*(in|by|through)\s+a\s+(peer[- ]reviewed|published))",
        r"\bscientifically\s*proven\b(?!\s*(in|by|through)\s+a\s+(peer[- ]reviewed|published))",
        r"\bFDA\s*approved\b",  # if not actually true
        r"\bWHO\s*(approved|recommended|endorsed)\b",
    ]

    EN_URGENCY_PATTERNS = [
        r"\blimited\s*time\b",
        r"\bact\s*now\b",
        r"\bbefore\s*it'?s\s*too\s*late\b",
        r"\b(doctors|pharma|government)\s*(don'?t\s*want|hide|hate)\b",
        r"\bsecret\b.*\b(cure|remedy|formula)",
        r"\bthey\s*don'?t\s*want\s*you\s*to\s*know\b",
    ]

    # ── Thai patterns ───────────────────────────────────────────────────
    TH_CURE_PATTERNS = [
        r"รักษา(ให้)?หาย(ขาด)?.*(มะเร็ง|เบาหวาน|เอดส์|หัวใจ|อัลไซเมอร์)",
        r"หายขาด.*มะเร็ง",
        r"มะเร็ง.*หาย",
        r"กำจัด.*(มะเร็ง|เนื้องอก|ไวรัส)",
    ]

    TH_NATURAL_SAFE_PATTERNS = [
        r"ธรรมชาติ.*ปลอดภัย",
        r"สมุนไพร.*ไม่มีผลข้างเคียง",
        r"จากธรรมชาติ.*100",
        r"ไร้สารเคมี",
        r"ไม่มีสารเคมี",
    ]

    TH_PANACEA_PATTERNS = [
        r"รักษา(ได้)?ทุก(โรค|อาการ)",
        r"ยาเทวดา",
        r"ยาวิเศษ",
        r"ปาฏิหาริย์",
    ]

    TH_NO_SIDE_EFFECT_PATTERNS = [
        r"ไม่มีผลข้างเคียง",
        r"ปลอดภัย\s*100\s*%",
        r"ไม่มีความเสี่ยง",
        r"ปลอดภัย(อย่าง)?(แน่นอน|สมบูรณ์)",
    ]

    TH_URGENCY_PATTERNS = [
        r"หมดเขต",
        r"จำนวนจำกัด",
        r"ก่อนจะสาย",
        r"หมอ(ไม่)?อยากให้(คุณ)?รู้",
    ]

    def __init__(self, cfg: PhytoIntegrationConfig):
        self.cfg = cfg
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile all regex patterns for efficiency."""
        def compile_all(patterns):
            return [re.compile(p, re.IGNORECASE) for p in patterns]

        self.en_cure = compile_all(self.EN_CURE_PATTERNS)
        self.en_natural_safe = compile_all(self.EN_NATURAL_SAFE_PATTERNS)
        self.en_superiority = compile_all(self.EN_SUPERIORITY_PATTERNS)
        self.en_no_side_effect = compile_all(self.EN_NO_SIDE_EFFECT_PATTERNS)
        self.en_panacea = compile_all(self.EN_PANACEA_PATTERNS)
        self.en_false_authority = compile_all(self.EN_FALSE_AUTHORITY_PATTERNS)
        self.en_urgency = compile_all(self.EN_URGENCY_PATTERNS)

        self.th_cure = compile_all(self.TH_CURE_PATTERNS)
        self.th_natural_safe = compile_all(self.TH_NATURAL_SAFE_PATTERNS)
        self.th_panacea = compile_all(self.TH_PANACEA_PATTERNS)
        self.th_no_side_effect = compile_all(self.TH_NO_SIDE_EFFECT_PATTERNS)
        self.th_urgency = compile_all(self.TH_URGENCY_PATTERNS)

    def _match_any(self, text: str, patterns: List[re.Pattern]) -> List[str]:
        """Return all matching pattern descriptions."""
        matches = []
        for pat in patterns:
            m = pat.search(text)
            if m:
                matches.append(m.group(0))
        return matches

    def analyze(self, claim_text: str) -> Dict[str, Any]:
        """
        Analyze a claim text for problematic content.

        Args:
            claim_text: the advertising/marketing text to analyze

        Returns:
            dict with verdict, detected issues, and severity
        """
        text = claim_text.lower().strip()

        issues = []
        severity_score = 0.0

        # ── Check each category ─────────────────────────────────────────
        # 1. Cure claims (most serious)
        cure_matches = self._match_any(text, self.en_cure + self.th_cure)
        if cure_matches:
            issues.append({
                "category": "DISEASE_CURE_CLAIM",
                "severity": 1.0,
                "matches": cure_matches,
                "explanation": "Claims to cure/treat a specific disease. "
                               "Requires drug registration, not supplement marketing.",
            })
            severity_score += 1.0

        # 2. Natural = safe fallacy
        natural_matches = self._match_any(text, self.en_natural_safe + self.th_natural_safe)
        if natural_matches:
            issues.append({
                "category": "NATURAL_SAFE_FALLACY",
                "severity": 0.7,
                "matches": natural_matches,
                "explanation": "Implies 'natural' automatically means 'safe'. "
                               "Many natural substances are highly toxic "
                               "(aristolochic acid, ricin, aconitine, etc.).",
            })
            severity_score += 0.7

        # 3. No side effects
        no_se_matches = self._match_any(text, self.en_no_side_effect + self.th_no_side_effect)
        if no_se_matches:
            issues.append({
                "category": "NO_SIDE_EFFECTS_CLAIM",
                "severity": 0.8,
                "matches": no_se_matches,
                "explanation": "All biologically active substances have potential side effects. "
                               "This claim is misleading and potentially dangerous.",
            })
            severity_score += 0.8

        # 4. Panacea / cure-all
        panacea_matches = self._match_any(text, self.en_panacea + self.th_panacea)
        if panacea_matches:
            issues.append({
                "category": "PANACEA_CLAIM",
                "severity": 0.95,
                "matches": panacea_matches,
                "explanation": "Claims to cure all diseases. This is a hallmark of fraud.",
            })
            severity_score += 0.95

        # 5. Unsubstantiated superiority
        sup_matches = self._match_any(text, self.en_superiority)
        if sup_matches:
            issues.append({
                "category": "UNSUBSTANTIATED_SUPERIORITY",
                "severity": 0.6,
                "matches": sup_matches,
                "explanation": "Claims superiority over established treatments "
                               "without comparative clinical evidence.",
            })
            severity_score += 0.6

        # 6. False authority
        auth_matches = self._match_any(text, self.en_false_authority)
        if auth_matches:
            issues.append({
                "category": "FALSE_AUTHORITY",
                "severity": 0.7,
                "matches": auth_matches,
                "explanation": "Invokes scientific or regulatory authority "
                               "without verifiable evidence.",
            })
            severity_score += 0.7

        # 7. Urgency / conspiracy
        urg_matches = self._match_any(text, self.en_urgency + self.th_urgency)
        if urg_matches:
            issues.append({
                "category": "URGENCY_OR_CONSPIRACY",
                "severity": 0.5,
                "matches": urg_matches,
                "explanation": "Uses urgency, scarcity, or conspiracy framing "
                               "to bypass rational evaluation.",
            })
            severity_score += 0.5

        # ── Determine verdict ────────────────────────────────────────────
        if not issues:
            verdict = ClaimVerdict.ACCEPTABLE
        elif severity_score >= 2.0:
            verdict = ClaimVerdict.FRAUDULENT
        elif severity_score >= 1.5:
            verdict = ClaimVerdict.DANGEROUS
        elif severity_score >= 0.8:
            verdict = ClaimVerdict.MISLEADING
        else:
            verdict = ClaimVerdict.NEEDS_QUALIFIER

        # Normalize severity to [0, 1]
        normalized_severity = min(severity_score / 3.0, 1.0)

        return {
            "verdict": verdict.name,
            "verdict_enum": verdict,
            "severity_score": normalized_severity,
            "n_issues": len(issues),
            "issues": issues,
            "requires_regulatory_review": verdict in (
                ClaimVerdict.MISLEADING,
                ClaimVerdict.DANGEROUS,
                ClaimVerdict.FRAUDULENT,
            ),
            "analyzed_text_length": len(claim_text),
        }

    def analyze_batch(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple claims."""
        return [self.analyze(c) for c in claims]


# ═══════════════════════════════════════════════════════════════════════
# [I4] INTEGRATED SAFETY SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class PhytoIntegratedSafetySystem(nn.Module):
    """
    Unified system connecting:
      • PhytoTargetPredictor (target prediction)
      • KnownPlantToxinDatabase (hard block)
      • AdvertisingClaimAnalyzer (marketing oversight)

    Produces a single, actionable safety verdict.
    """

    def __init__(self, cfg: PhytoIntegrationConfig):
        super().__init__()
        self.cfg = cfg
        self.target_predictor = PhytoTargetPredictor(cfg)
        self.toxin_db = KnownPlantToxinDatabase()
        self.claim_analyzer = AdvertisingClaimAnalyzer(cfg)

    def screen_for_known_toxins(
        self,
        compound_names: List[str],
    ) -> Dict[str, Any]:
        """
        Screen compound list against known toxin database.

        Returns hard-block recommendation if critical/banned toxins found.
        """
        matches = self.toxin_db.screen_compounds(compound_names)

        hard_block_toxins = [m for m in matches if m["is_banned"] or m["is_critical_or_above"]]
        warnings = [m for m in matches if not m["is_banned"] and not m["is_critical_or_above"]]

        return {
            "n_matches": len(matches),
            "hard_block": len(hard_block_toxins) > 0,
            "hard_block_toxins": hard_block_toxins,
            "warnings": warnings,
            "all_matches": matches,
        }

    def forward(
        self,
        fingerprint: Tensor,              # (B, fingerprint_dim)
        target_features: Tensor,          # (B, n_targets, protein_dim)
        compound_names: List[str],        # names for toxin screening
        claim_text: Optional[str] = None, # advertising text to analyze
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full integrated safety analysis.

        Args:
            fingerprint: compound fingerprint
            target_features: protein target panel features
            compound_names: list of compound names for toxin screening
            claim_text: optional advertising/label text
            target_names: optional target names for reporting

        Returns:
            dict with target predictions, toxin screen, claim analysis,
            and final safety verdict
        """
        out = {}

        # ── 1. Target prediction ────────────────────────────────────────
        target_out = self.target_predictor(
            fingerprint, target_features, target_names
        )
        out.update({f"target_{k}": v for k, v in target_out.items()})

        # ── 2. Toxin screening ──────────────────────────────────────────
        toxin_screen = self.screen_for_known_toxins(compound_names)
        out["toxin_screen"] = toxin_screen

        # ── 3. Claim analysis (if provided) ─────────────────────────────
        if claim_text is not None:
            claim_analysis = self.claim_analyzer.analyze(claim_text)
            out["claim_analysis"] = claim_analysis
        else:
            out["claim_analysis"] = None

        # ── 4. Final safety verdict ─────────────────────────────────────
        # Decision logic (pessimistic / safety-first)
        final_verdict = "PROCEED"
        reasons = []

        # Hard block: known critical/banned toxin
        if toxin_screen["hard_block"]:
            final_verdict = "BLOCK"
            for t in toxin_screen["hard_block_toxins"]:
                reasons.append(
                    f"KNOWN TOXIN ({t['toxin_name']}, severity={t['severity']}): "
                    f"{t['mechanism']}"
                )

        # Dangerous/fraudulent claim
        if out["claim_analysis"] is not None:
            ca = out["claim_analysis"]
            if ca["verdict"] in ("DANGEROUS", "FRAUDULENT"):
                if final_verdict == "PROCEED":
                    final_verdict = "BLOCK"
                reasons.append(
                    f"CLAIM VERDICT ({ca['verdict']}): "
                    f"{ca['n_issues']} issues detected"
                )
            elif ca["verdict"] == "MISLEADING":
                if final_verdict == "PROCEED":
                    final_verdict = "WARN"
                reasons.append("MISLEADING CLAIMS detected")

        # Toxin warnings (not hard block but concerning)
        if toxin_screen["warnings"]:
            if final_verdict == "PROCEED":
                final_verdict = "WARN"
            for w in toxin_screen["warnings"]:
                reasons.append(
                    f"Toxin warning ({w['toxin_name']}, severity={w['severity']})"
                )

        out["final_verdict"] = final_verdict
        out["final_reasons"] = reasons
        out["requires_regulatory_review"] = final_verdict in ("BLOCK", "WARN")

        return out


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Phyto Integration — Smoke Test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    torch.manual_seed(42)

    cfg = PhytoIntegrationConfig(
        latent_dim=64,
        compound_dim=32,
        protein_dim=64,
        fingerprint_dim=128,
        max_targets=5,
        device=str(device),
    )

    system = PhytoIntegratedSafetySystem(cfg).to(device)

    # ── Test 1: Toxin database ──────────────────────────────────────────
    print("\n  ── [1] Known Plant Toxin Database ──")
    db = system.toxin_db
    summary = db.summary()
    print(f"    Unique toxins      : {summary['total_unique_toxins']}")
    print(f"    Severity breakdown : {summary['by_severity']}")
    print(f"    Hard-block toxins  : {summary['hard_block_count']}")

    # Search test
    aa = db.search("aristolochic acid")
    print(f"\n    Search 'aristolochic acid':")
    print(f"      Severity : {aa.severity.name}")
    print(f"      Mechanism: {aa.mechanism[:80]}...")
    print(f"      Status   : {aa.regulatory_status[:80]}...")

    # Screening test
    test_compounds = ["curcumin", "aristolochic acid", "berberine", "unknown_compound_x"]
    screen = db.screen_compounds(test_compounds)
    print(f"\n    Screen {test_compounds}:")
    for m in screen:
        print(f"      • {m['query']} → {m['toxin_name']} ({m['severity']})")

    # ── Test 2: Claim analyzer ──────────────────────────────────────────
    print("\n  ── [2] Advertising Claim Analyzer ──")
    analyzer = system.claim_analyzer

    test_claims = [
        # Claim 1: Dangerous cure claim
        "This herbal supplement cures cancer in just 30 days! "
        "100% natural with no side effects. Doctors don't want you to know!",
        # Claim 2: Acceptable wellness claim
        "This turmeric extract may support joint health and comfort. "
        "Consult your healthcare provider before use.",
        # Claim 3: Misleading
        "The best natural remedy, superior to chemotherapy. "
        "Clinically proven to eliminate tumors.",
    ]

    for i, claim in enumerate(test_claims):
        result = analyzer.analyze(claim)
        print(f"\n    Claim {i+1}: \"{claim[:60]}...\"")
        print(f"      Verdict : {result['verdict']}")
        print(f"      Severity: {result['severity_score']:.2f}")
        print(f"      Issues  : {result['n_issues']}")
        for issue in result["issues"]:
            print(f"        • {issue['category']}")

    # ── Test 3: Target predictor ────────────────────────────────────────
    print("\n  ── [3] Phyto Target Predictor ──")
    B = 2
    fingerprint = torch.rand(B, cfg.fingerprint_dim, device=device)
    n_targets = 5
    target_features = torch.randn(B, n_targets, cfg.protein_dim, device=device)
    target_names = ["EGFR", "VEGFR", "COX-2", "BRAF", "PARP"]

    system.eval()
    with torch.no_grad():
        target_out = system.target_predictor(fingerprint, target_features, target_names)

    print(f"    Binding scores shape : {tuple(target_out['binding_scores'].shape)}")
    print(f"    Ranked targets       : {[r['target'] for r in target_out['ranked_report'][:3]]}")
    print(f"    Significant targets  : {target_out['n_significant_targets'].tolist()}")

    # ── Test 4: Full integrated pipeline ────────────────────────────────
    print("\n  ── [4] Full Integrated Pipeline ──")

    # Scenario A: Safe compound
    safe_compounds = ["curcumin", "quercetin"]
    safe_claim = "May support general wellness. Not intended to diagnose or treat any disease."

    with torch.no_grad():
        safe_out = system(
            fingerprint=fingerprint[:1],
            target_features=target_features[:1],
            compound_names=safe_compounds,
            claim_text=safe_claim,
            target_names=target_names,
        )
    print(f"\n    Scenario A (safe compounds + honest claim):")
    print(f"      Final verdict : {safe_out['final_verdict']}")
    print(f"      Toxin matches : {safe_out['toxin_screen']['n_matches']}")
    print(f"      Claim verdict : {safe_out['claim_analysis']['verdict']}")

    # Scenario B: Dangerous compound + fraudulent claim
    danger_compounds = ["aristolochic acid", "aconitine"]
    danger_claim = "Cures all cancers! 100% natural, absolutely no side effects. " \
                   "FDA approved. Better than chemotherapy!"

    with torch.no_grad():
        danger_out = system(
            fingerprint=fingerprint[:1],
            target_features=target_features[:1],
            compound_names=danger_compounds,
            claim_text=danger_claim,
            target_names=target_names,
        )
    print(f"\n    Scenario B (toxic compounds + fraudulent claim):")
    print(f"      Final verdict : {danger_out['final_verdict']}")
    print(f"      Toxin matches : {danger_out['toxin_screen']['n_matches']}")
    for t in danger_out["toxin_screen"]["hard_block_toxins"]:
        print(f"        🚫 {t['toxin_name']} ({t['severity']})")
    print(f"      Claim verdict : {danger_out['claim_analysis']['verdict']}")
    print(f"      Reasons:")
    for r in danger_out["final_reasons"]:
        print(f"        • {r}")

    print("\n" + "=" * 72)
    print("  Smoke test complete.")
    print("=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Toxin database covers known toxins but is NOT exhaustive.")
    print("  • Target prediction is approximate; docking/FEP needed for accuracy.")
    print("  • Claim analysis is pattern-based; not a legal review.")
    print("  • Always combine with wet-lab testing and regulatory review.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
