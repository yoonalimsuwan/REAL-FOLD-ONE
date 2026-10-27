"""
=============================================================================
SESI ML CLAIM ANALYZER
Machine Learning-Based Analysis of Phytochemical Marketing Claims
Replaces/augments pattern-based detection with semantic understanding
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (ML-Powered Claim Analysis)

WHY ML INSTEAD OF PATTERNS?
───────────────────────────
Pattern-based detection (regex) catches obvious keywords but misses:
  • Paraphrased claims ("melts away tumors" instead of "cures cancer")
  • Multilingual marketing (Thai, English, Chinese)
  • Implicit claims ("doctors are amazed by this discovery")
  • Evasive language ("supports the body's natural ability to fight...")

ML-based analysis uses semantic embeddings to understand MEANING,
not just keywords.

ARCHITECTURE
────────────
  claim_text
      │
      ▼
  [Sentence Encoder] — multilingual transformer embeddings
      │
      ├──► Disease Claim Detector (multi-label classification)
      ├──► Evidence Level Regressor (0-7 scale)
      ├──► Claim Severity Classifier (acceptable→fraudulent)
      ├──► Credibility Scorer (composite)
      └──► Red Flag Detector (anomaly detection)
      │
      ▼
  ClaimAnalysisReport (structured verdict)

INTEGRATION
───────────
  • Replaces AdvertisingClaimAnalyzer pattern matching
  • Outputs compatible with EvidenceQualityGrader
  • Feeds into PhytoSafetyGate for final decision

HONEST SCOPE
────────────
• Requires a pre-trained sentence encoder (e.g., multilingual-e5,
  paraphrase-multilingual-MiniLM). Without it, falls back to pattern mode.
• The classifier heads must be TRAINED on labeled claim data.
  This module provides the architecture; training data is needed.
• ML analysis is probabilistic — not a legal/regulatory verdict.
• Human review is ALWAYS required for enforcement decisions.
=============================================================================
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.ml_claim")

# ── Optional: sentence-transformers ───────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

__all__ = [
    "MLClaimAnalyzerConfig",
    "ClaimSeverityLevel",
    "MLClaimAnalyzer",
    "ClaimAnalysisReport",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS & CONFIG
# ═══════════════════════════════════════════════════════════════════════

class ClaimSeverityLevel(Enum):
    """Severity levels for marketing claims."""
    ACCEPTABLE = 0
    NEEDS_QUALIFIER = 1
    MISLEADING = 2
    DANGEROUS = 3
    FRAUDULENT = 4


# Disease categories for multi-label detection
DISEASE_CATEGORIES = [
    "cancer",
    "diabetes",
    "cardiovascular",
    "neurodegenerative",
    "autoimmune",
    "infectious",
    "respiratory",
    "digestive",
    "mental_health",
    "pain_inflammation",
    "weight_obesity",
    "skin_aging",
]

# Red flag linguistic patterns (for feature engineering)
RED_FLAG_KEYWORDS = {
    "cure_all": ["cures all", "รักษาทุกโรค", "治百病"],
    "no_side_effects": ["no side effects", "ไม่มีผลข้างเคียง", "无副作用"],
    "natural_safe": ["natural and safe", "ธรรมชาติปลอดภัย", "天然安全"],
    "miracle": ["miracle", "ปาฏิหาริย์", "奇迹"],
    "secret": ["secret cure", "ความลับ", "秘方"],
    "doctors_hate": ["doctors don't want", "หมอไม่บอก", "医生不想让你知道"],
    "guaranteed": ["guaranteed results", "รับประกันผล", "保证有效"],
    "quick_fix": ["overnight", "ชั่วข้ามคืน", "一夜之间"],
}


@dataclass
class MLClaimAnalyzerConfig:
    """Configuration for ML claim analyzer."""
    # Sentence encoder
    encoder_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    encoder_dim: int = 384  # MiniLM-L12 dimension
    use_gpu: bool = True

    # Classification heads
    n_disease_categories: int = len(DISEASE_CATEGORIES)
    n_severity_levels: int = len(ClaimSeverityLevel)
    n_evidence_levels: int = 8  # 0-7 scale

    # Feature engineering
    use_red_flag_features: bool = True

    # Device
    device: str = "cuda"


@dataclass
class ClaimAnalysisReport:
    """Structured output of claim analysis."""
    text: str
    text_length: int
    # Disease claims
    disease_claims: Dict[str, float]  # category → probability
    primary_disease_claim: Optional[str]
    # Evidence
    estimated_evidence_level: float  # 0-7
    evidence_level_name: str
    # Severity
    severity_level: ClaimSeverityLevel
    severity_name: str
    severity_score: float  # 0-1
    # Credibility
    credibility_score: float  # 0-1
    # Red flags
    red_flags: List[str]
    # Verdict
    verdict: str
    requires_human_review: bool
    # Embedding (for downstream use)
    embedding: Optional[List[float]] = None


# ═══════════════════════════════════════════════════════════════════════
# ML CLAIM ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class MLClaimAnalyzer(nn.Module):
    """
    Machine learning-based analysis of phytochemical marketing claims.

    Uses sentence embeddings + learned classification heads.

    Components:
      1. Sentence encoder (frozen, pre-trained)
      2. Disease claim detector (multi-label)
      3. Evidence level regressor
      4. Severity classifier
      5. Credibility scorer
      6. Red flag detector (rule-based features)
    """

    def __init__(self, cfg: MLClaimAnalyzerConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Sentence encoder (optional, may not be installed)
        self.encoder = None
        self.encoder_available = False
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.encoder = SentenceTransformer(cfg.encoder_model)
                self.encoder.to(self.device)
                self.encoder.eval()
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder_available = True
                logger.info(f"Sentence encoder loaded: {cfg.encoder_model}")
            except Exception as e:
                logger.warning(f"Failed to load encoder: {e}")
        else:
            logger.warning("sentence-transformers not installed; "
                          "ML analysis will use fallback features")

        # Classification heads
        input_dim = cfg.encoder_dim
        if cfg.use_red_flag_features:
            input_dim += len(RED_FLAG_KEYWORDS)  # + red flag features

        # 1. Disease claim detector (multi-label)
        self.disease_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, cfg.n_disease_categories),
        )

        # 2. Evidence level regressor (0-7)
        self.evidence_regressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 3. Severity classifier
        self.severity_classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_severity_levels),
        )

        # 4. Credibility scorer
        self.credibility_scorer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.to(self.device)

    # ── Feature extraction ──────────────────────────────────────────────
    def _encode_text(self, text: str) -> Tensor:
        """Encode text to embedding."""
        if self.encoder_available and self.encoder is not None:
            embedding = self.encoder.encode([text], convert_to_tensor=True)
            return embedding.to(self.device)
        else:
            # Fallback: simple character-based features
            # (not semantic, but allows the pipeline to run)
            logger.warning("Using fallback text features (no semantic understanding)")
            chars = [ord(c) % 100 for c in text[:self.cfg.encoder_dim]]
            # Pad or truncate
            while len(chars) < self.cfg.encoder_dim:
                chars.append(0)
            chars = chars[:self.cfg.encoder_dim]
            return torch.tensor([chars], dtype=torch.float32, device=self.device)

    def _extract_red_flags(self, text: str) -> Tuple[List[str], Tensor]:
        """
        Extract red flag features from text.

        Returns:
            list of detected red flag names
            tensor of red flag indicators (0/1)
        """
        text_lower = text.lower()
        detected = []
        flags = torch.zeros(len(RED_FLAG_KEYWORDS), device=self.device)

        for i, (flag_name, keywords) in enumerate(RED_FLAG_KEYWORDS.items()):
            for kw in keywords:
                if kw.lower() in text_lower:
                    detected.append(flag_name)
                    flags[i] = 1.0
                    break  # one match per category is enough

        return detected, flags

    # ── Analysis ────────────────────────────────────────────────────────
    def analyze(self, text: str) -> ClaimAnalysisReport:
        """
        Analyze a marketing claim text.

        Args:
            text: the claim text to analyze

        Returns:
            ClaimAnalysisReport with all analysis results
        """
        self.eval()

        with torch.no_grad():
            # 1. Encode text
            embedding = self._encode_text(text)  # (1, encoder_dim)

            # 2. Extract red flags
            red_flags, red_flag_tensor = self._extract_red_flags(text)

            # 3. Combine features
            if self.cfg.use_red_flag_features:
                combined = torch.cat([embedding, red_flag_tensor.unsqueeze(0)], dim=-1)
            else:
                combined = embedding

            # 4. Disease detection
            disease_logits = self.disease_detector(combined)
            disease_probs = torch.sigmoid(disease_logits).squeeze(0)  # (n_categories,)
            disease_claims = {
                DISEASE_CATEGORIES[i]: float(disease_probs[i])
                for i in range(len(DISEASE_CATEGORIES))
            }
            # Primary disease claim (if any above threshold)
            primary_disease = None
            max_prob = max(disease_claims.values())
            if max_prob > 0.5:
                primary_disease = max(disease_claims, key=disease_claims.get)

            # 5. Evidence level
            evidence_raw = self.evidence_regressor(combined).squeeze()
            evidence_level = float(torch.clamp(evidence_raw, 0, 7))
            evidence_names = [
                "NONE", "TRADITIONAL", "IN_VITRO", "ANIMAL",
                "PHASE_1", "PHASE_2", "RCT", "META_ANALYSIS"
            ]
            evidence_name = evidence_names[int(round(evidence_level))]

            # 6. Severity
            severity_logits = self.severity_classifier(combined)
            severity_probs = F.softmax(severity_logits, dim=-1).squeeze(0)
            severity_idx = int(severity_probs.argmax())
            severity_level = ClaimSeverityLevel(severity_idx)
            severity_score = float(severity_probs.max())

            # 7. Credibility
            credibility = float(self.credibility_scorer(combined).squeeze())

            # 8. Verdict
            verdict_map = {
                ClaimSeverityLevel.ACCEPTABLE: "ACCEPTABLE",
                ClaimSeverityLevel.NEEDS_QUALIFIER: "NEEDS_QUALIFIER",
                ClaimSeverityLevel.MISLEADING: "MISLEADING",
                ClaimSeverityLevel.DANGEROUS: "DANGEROUS",
                ClaimSeverityLevel.FRAUDULENT: "FRAUDULENT",
            }
            verdict = verdict_map[severity_level]

            # Human review needed for dangerous/fraudulent
            requires_review = severity_level in (
                ClaimSeverityLevel.DANGEROUS,
                ClaimSeverityLevel.FRAUDULENT,
            )

        return ClaimAnalysisReport(
            text=text,
            text_length=len(text),
            disease_claims=disease_claims,
            primary_disease_claim=primary_disease,
            estimated_evidence_level=evidence_level,
            evidence_level_name=evidence_name,
            severity_level=severity_level,
            severity_name=severity_level.name,
            severity_score=severity_score,
            credibility_score=credibility,
            red_flags=red_flags,
            verdict=verdict,
            requires_human_review=requires_review,
            embedding=embedding.squeeze(0).cpu().tolist() if embedding is not None else None,
        )

    def analyze_batch(self, texts: List[str]) -> List[ClaimAnalysisReport]:
        """Analyze multiple claims."""
        return [self.analyze(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════════
# TRAINING DATA SCHEMA (for fine-tuning the heads)
# ═══════════════════════════════════════════════════════════════════════

def get_training_data_schema() -> Dict[str, Any]:
    """
    Schema for labeled training data.

    To train the classification heads, you need labeled examples:

    {
        "text": "This herbal supplement cures cancer in 30 days!",
        "labels": {
            "disease_categories": ["cancer"],       # multi-label
            "evidence_level": 0,                     # 0-7
            "severity_level": 4,                     # 0-4 (fraudulent)
            "credibility": 0.1,                      # 0-1
            "red_flags": ["cure_all", "quick_fix"]
        }
    }

    HONEST NOTE: Building a quality labeled dataset requires:
      • Regulatory experts (Thai FDA, etc.)
      • Consumer protection specialists
      • Linguists (for multilingual claims)
      • At least 1000+ labeled examples for reasonable performance
    """
    return {
        "fields": {
            "text": "string — the claim text",
            "disease_categories": f"list — subset of {DISEASE_CATEGORIES}",
            "evidence_level": "int 0-7 — NONE to META_ANALYSIS",
            "severity_level": "int 0-4 — ACCEPTABLE to FRAUDULENT",
            "credibility": "float 0-1",
            "red_flags": "list of red flag categories",
        },
        "recommended_dataset_size": "1000+ labeled examples",
        "labeling_experts": [
            "regulatory affairs (FDA)",
            "consumer protection",
            "linguistics (multilingual)",
            "clinical medicine",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI ML Claim Analyzer — Smoke Test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Sentence-transformers available: {HAS_SENTENCE_TRANSFORMERS}")

    cfg = MLClaimAnalyzerConfig(
        device=str(device),
        use_gpu=torch.cuda.is_available(),
    )

    analyzer = MLClaimAnalyzer(cfg)
    print(f"  Encoder available: {analyzer.encoder_available}")

    # Test claims
    test_claims = [
        # Dangerous cure claim
        "This herbal supplement cures cancer in just 30 days! "
        "100% natural with absolutely no side effects. "
        "Doctors don't want you to know about this miracle cure!",
        # Acceptable wellness claim
        "This turmeric extract may support joint health and comfort. "
        "Consult your healthcare provider before use.",
        # Misleading superiority claim
        "The most powerful natural remedy ever discovered. "
        "Scientifically proven to eliminate tumors and replace chemotherapy.",
        # Thai language claim
        "สมุนไพรตัวนี้รักษาโรคมะเร็งให้หายขาดได้ภายใน 1 เดือน "
        "ปลอดภัย 100% ไม่มีผลข้างเคียง",
    ]

    print(f"\n  ── Analyzing {len(test_claims)} test claims ──")

    for i, claim in enumerate(test_claims):
        report = analyzer.analyze(claim)
        print(f"\n  Claim {i+1}: \"{claim[:60]}...\"")
        print(f"    Verdict            : {report.verdict}")
        print(f"    Severity           : {report.severity_name} ({report.severity_score:.2f})")
        print(f"    Evidence level     : {report.estimated_evidence_level:.1f} ({report.evidence_level_name})")
        print(f"    Credibility        : {report.credibility_score:.3f}")
        print(f"    Primary disease    : {report.primary_disease_claim}")
        print(f"    Red flags          : {report.red_flags}")
        print(f"    Human review needed: {report.requires_human_review}")

    # Test batch analysis
    print(f"\n  ── Batch Analysis ──")
    reports = analyzer.analyze_batch(test_claims)
    print(f"    Analyzed {len(reports)} claims")
    print(f"    Verdicts: {[r.verdict for r in reports]}")

    # Show training data schema
    print(f"\n  ── Training Data Schema ──")
    schema = get_training_data_schema()
    print(f"    Recommended dataset size: {schema['recommended_dataset_size']}")
    print(f"    Fields: {list(schema['fields'].keys())}")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • Classification heads need TRAINING on labeled data")
    print("  • Without sentence-transformers, falls back to weak features")
    print("  • ML analysis is probabilistic — not a legal verdict")
    print("  • Human review required for enforcement decisions")
    print("  • Multilingual support depends on the encoder model")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
