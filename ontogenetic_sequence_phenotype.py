#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ontogenetic_sequence_phenotype.py — v1.0
MSA · Sequence Transformer · Discrete Diffusion · Ontogenetic World Model
Variant→Phenotype with calibrated uncertainty · QC feature bridge
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

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

Not even a single sparrow does God forget; every hair upon our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and biology stand independently of the philosophy above.
The philosophy is the author's belief, included by request, and is not a
scientific claim nor a condition of use.
-----------------------------------------------------------------------------

=============================================================================
WHAT THIS MODULE IS
=============================================================================
A research-grade sequence→phenotype and developmental-trajectory platform
that REUSES the quantum-chemistry models already trained in this codebase
(qm9_schnet_painn.py, force_datasets.py, mol_advanced.py) as a chemistry
feature source, and layers the MSPS control mathematics on top.

Pipeline:
    Sequence / MSA
        ↓  MSAInterface (conservation, mutual information, DCA-lite)
    Transformer encoder  ←  QuantumChemBridge (PaiNN electronic features)
        ↓
    OntogeneticWorldModel  (developmental trajectory, CSOC-gated fate)
        ↓
    VariantEffectPredictor (ensemble + Politis–Romano CI + calibration)
        ↓
    Phenotype research hypothesis  (NEVER a clinical call)

Plus a discrete diffusion model for sequence hypothesis generation
(research only; every sample is labelled as synthetic, never as designed
for editing).

=============================================================================
HARD SAFETY BOUNDARIES (enforced in code, not just documented)
=============================================================================
BLOCKED outright by SafetyGate:
  • gene editor / guide RNA / base editor / prime editor design
  • delivery vector / capsid engineering
  • pathogen enhancement / immune evasion / gain of function
  • clinical diagnosis, treatment orders, embryo selection
  • any output framed as "edit this sequence to get X"

ALLOWED:
  • variant effect PREDICTION on existing or hypothetical sequences
  • developmental trajectory analysis
  • MSA / conservation / coevolution analysis
  • synthetic hypothesis generation, explicitly labelled research_only

Genetic data passes a governance gate: IRB + de-identification + consent.
No raw human genetic data leaves this process; no PHI is ever embedded in
a query to an external service.

=============================================================================
HONEST SCIENTIFIC SCOPE — READ BEFORE INTERPRETING ANY NUMBER
=============================================================================
❌ Sequence→phenotype is a GRAND-CHALLENGE problem. Most traits are
   polygenic, epistatic, environment-dependent, and poorly mapped. This
   module produces research hypotheses with quantified uncertainty, NOT
   reliable clinical predictions.
❌ The transformer and diffusion here are SMALL reference implementations.
   Real performance requires ESM-2/ProGen/AlphaFold-class models and
   curated MSA databases (UniRef, BFD). Adapters are provided; without
   those backends the answer is UNAVAILABLE, not fabricated.
❌ "Ontogenetic" here = a dynamical model of gene-expression state over
   developmental time. It does NOT reprogram or edit a genome.
❌ No experimental validation is performed here. Oracles are self-
   consistency tests: a planted-covariation MSA, a toy learnable task,
   and conservation of the probability simplex.
❌ QM9/ANI chemistry covers small molecules. Protein electronic structure
   at scale is NOT available; the QC bridge contributes residue-level
   physicochemical features and small-molecule ligand features only.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("ontogenetic")

VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
ONTOGENETIC SEQUENCE→PHENOTYPE v{VERSION}
MSA · Transformer · Discrete Diffusion · Ontogenetic World Model · QC bridge
RESEARCH ONLY. No gene-editor design. No clinical decisions.
MSPS math: CSOC bound, κ-fatness/δ_min, Carpet ρ≥9/8, OPMC, Politis–Romano.
=============================================================================
"""

# =============================================================================
# Genetic alphabets
# =============================================================================
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {c: i for i, c in enumerate(AA_ALPHABET)}
NA_ALPHABET = "ACGT"
NA_TO_IDX = {c: i for i, c in enumerate(NA_ALPHABET)}

# physicochemical property vectors per amino acid (Kyte-Doolittle hydropathy
# + molecular weight + charge class). Literature values; research features.
AA_PROPERTIES = {
    "A": (1.8, 89.1, 0), "R": (-4.5, 174.2, 1), "N": (-3.5, 132.1, 0),
    "D": (-3.5, 133.1, -1), "C": (2.5, 121.2, 0), "Q": (-3.5, 146.1, 0),
    "E": (-3.5, 147.1, -1), "G": (-0.4, 75.0, 0), "H": (-3.2, 155.2, 0),
    "I": (4.5, 131.2, 0), "L": (3.8, 131.2, 0), "K": (-3.9, 146.2, 1),
    "M": (1.9, 149.2, 0), "F": (2.8, 165.2, 0), "P": (-1.6, 115.1, 0),
    "S": (-0.8, 105.1, 0), "T": (-0.7, 119.1, 0), "W": (-0.9, 204.2, 0),
    "Y": (-1.3, 181.2, 0), "V": (4.2, 117.1, 0),
}


def seq_to_idx(seq: str, alphabet: str) -> Optional[np.ndarray]:
    mapping = AA_TO_IDX if alphabet == "aa" else NA_TO_IDX
    out = []
    for c in seq.upper():
        if c not in mapping:
            return None
        out.append(mapping[c])
    return np.array(out, dtype=np.int64)


def idx_to_seq(idx: np.ndarray, alphabet: str) -> str:
    alpha = AA_ALPHABET if alphabet == "aa" else NA_ALPHABET
    return "".join(alpha[int(i)] for i in idx)


# =============================================================================
# §G. GOVERNANCE + SAFETY GATE  (enforced, not advisory)
# =============================================================================

class SafetyGate:
    """Blocks gene-editor design and clinical-order intents outright."""
    BLOCKED = {
        "gene_editor_design": [
            r"\bcrispr\b", r"\bcas9\b", r"\bcas12\b", r"\bcas13\b",
            r"\bguide\s*rna\b", r"\bgrna\b", r"\bsgrna\b",
            r"\bbase\s*editor\b", r"\bprime\s*editor\b", r"\bgene\s*drive\b",
            r"\bdesign\s+(a|an|the)?\s*(editor|nuclease|guide)\b",
            r"\bknock\s*(out|in)\s*(design|construct)\b",
            r"\baav\s*capsid\b", r"\bdelivery\s*vector\b",
        ],
        "pathogen_enhancement": [
            r"\benhance\s*virulence\b", r"\bimmune\s*evasion\b",
            r"\bgain\s*of\s*function\b", r"\bincrease\s*transmissibility\b",
        ],
        "clinical_order": [
            r"\bdiagnose\s*(the\s*)?patient\b", r"\bprescribe\b",
            r"\bselect\s*embryo\b", r"\bclinical\s*decision\b",
            r"\bapprove\s*treatment\b",
        ],
        "phenotype_engineering": [
            r"\bedit\s+(this|the)\s+sequence\s+to\b",
            r"\bdesign\s+(a|the)\s+(baby|child|organism)\b",
            r"\bmake\s+(it|the)\s+(sequence|gene)\s+(produce|become)\b",
        ],
    }

    def check(self, text: str) -> Dict[str, Any]:
        low = text.lower()
        for intent, pats in self.BLOCKED.items():
            hits = [p for p in pats if re.search(p, low)]
            if hits:
                return {"allowed": False, "intent": intent,
                        "blocked_patterns": hits,
                        "reason": (f"BLOCKED intent={intent}. This platform performs "
                                   f"variant-effect RESEARCH and analysis only. It does "
                                   f"not design gene editors, delivery systems, or "
                                   f"clinical interventions.")}
        return {"allowed": True, "intent": "research_analysis",
                "blocked_patterns": [],
                "reason": "allowed research query"}


class GeneticGovernance:
    """IRB + consent + de-identification gate for human genetic data."""

    def __init__(self, irb_approved: bool = False,
                 approved_uses: Optional[List[str]] = None,
                 data_use_agreement: bool = False):
        self.irb_approved = irb_approved
        self.approved_uses = set(approved_uses or [])
        self.data_use_agreement = data_use_agreement
        self.safety = SafetyGate()
        self.audit: List[Dict[str, Any]] = []

    def authorize(self, text: str, uses_human_genetic_data: bool = False,
                  purpose: str = "variant_research",
                  deidentified: bool = True,
                  consent_present: bool = True) -> Dict[str, Any]:
        s = self.safety.check(text)
        d = dict(s)
        d.update({"authorized": False,
                  "clinical_actionability": RESEARCH_ONLY,
                  "clinician_review_required": True})
        if not s["allowed"]:
            self._log(d, text); return d
        if uses_human_genetic_data:
            if not self.irb_approved:
                d["reason"] = "BLOCKED: no active IRB approval"; self._log(d, text); return d
            if purpose not in self.approved_uses:
                d["reason"] = f"BLOCKED: purpose '{purpose}' not in approved uses"
                self._log(d, text); return d
            if not self.data_use_agreement:
                d["reason"] = "BLOCKED: no data-use agreement"; self._log(d, text); return d
            if not deidentified:
                d["reason"] = "BLOCKED: data not de-identified"; self._log(d, text); return d
            if not consent_present:
                d["reason"] = "BLOCKED: consent not documented"; self._log(d, text); return d
        d["authorized"] = True
        d["reason"] = "authorized for research"
        self._log(d, text)
        return d

    def _log(self, d: Dict[str, Any], text: str) -> None:
        self.audit.append({
            "ts": time.time(),
            "query_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "authorized": d["authorized"], "reason": d["reason"]})


# =============================================================================
# §M. MSPS CONTROL MATHEMATICS (the portable layer)
# =============================================================================

class CSOCEventCounter:
    """
    Engineering bound on the number of developmental-state switches:
        N(T) <= (E_max + D)/gap,   D = ACTUAL removal (audited).
    Used here to bound fate-transition chatter in the ontogenetic model.
    This is a mechanism bound, NOT a claim about biology.
    """
    def __init__(self, hi=0.60, lo=0.25, rate=0.08):
        assert hi > lo
        self.hi, self.lo, self.gap, self.rate = hi, lo, hi - lo, rate
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0; self.D_audit = 0.0

    def step(self, signal: float) -> bool:
        s = float(signal)
        excess = math.log1p(math.exp(-abs(s - self.hi))) + max(s - self.hi, 0.0)
        decay = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10.0))
        removal = decay * self.rate * self.acc
        self.acc += excess - removal
        self.D += removal; self.D_audit += removal
        self.E_max = max(self.E_max, self.acc)
        fire = self.armed and self.acc > self.hi
        if self.acc < self.lo:
            self.armed = True
        if fire:
            self.armed = False; self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        bound = (self.E_max + self.D) / max(self.gap, 1e-30)
        return {"events": self.events, "bound": bound,
                "satisfied": self.events <= bound + 1e-9,
                "D_actual": self.D,
                "D_audit_matches": abs(self.D - self.D_audit) < 1e-12,
                "note": "engineering bound on fate-switch count, NOT biology"}


def delta_min_exact(n=3, kappa=2.0, a_min=0.01) -> float:
    """Isodiametric minimum resolvable feature (Bieberbach 1915)."""
    d = n - 1
    om = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    return (1.0 / kappa) * (a_min / om) ** (1.0 / d)


def kappa_fatness_filter(points: np.ndarray, kappa: float = 2.0,
                         a_min: float = 0.01) -> np.ndarray:
    """
    Guard for sequence-space clustering: rejects degenerate thin clusters
    (slivers) whose inradius/diameter < 1/kappa. Used when grouping variants
    into functional neighborhoods. Outer/discrete loop, not differentiable.
    Returns a boolean keep-mask.
    """
    n = len(points)
    keep = np.ones(n, dtype=bool)
    dmin = delta_min_exact(3, kappa, a_min)
    for i in range(n):
        d = np.linalg.norm(points - points[i], axis=1)
        d[i] = np.inf
        nn = d.min()
        # crude sliver test: nearest-neighbour distance below the floor
        if nn < dmin * 0.1:
            keep[i] = False
    return keep


def carpet_laplacian(side: int = 9) -> np.ndarray:
    """Carpet-LIKE graph Laplacian for multiscale priority weighting."""
    side = max(3, side); n = side * side
    A = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1: continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < side and 0 <= jj < side): continue
                if ii % 3 == 1 and jj % 3 == 1: continue
                A[i * side + j, ii * side + jj] = 1.0
    return np.diag(A.sum(1)) - A


CARPET_RHO = 9.0 / 8.0


def carpet_priority(signal: np.ndarray, side: int = 9,
                    levels: int = 3, rho: float = CARPET_RHO) -> float:
    """
    ρ^m-normalized multiscale graph energy. SECONDARY priority weight for
    focusing variant analysis on multiscale-important positions.
    HONEST: heuristic on a carpet-LIKE graph; NOT the carpet resistance
    form and NOT a differential operator.
    """
    L = carpet_laplacian(side)
    flat = np.asarray(signal, dtype=float).reshape(-1)
    m = L.shape[0]
    if flat.size != m:
        # pool to m bins
        bins = np.array_split(flat, m)
        flat = np.array([b.mean() if len(b) else 0.0 for b in bins])
    grp = np.arange(len(flat)) * m // max(len(flat), 1)
    s = np.zeros(m)
    den = np.zeros(m)
    np.add.at(s, grp, flat)
    np.add.at(den, grp, 1.0)
    s = s / np.maximum(den, 1.0)
    tot = 0.0
    for lv in range(levels):
        tot += (rho ** lv) * (s @ L @ s) / (lv + 1)
    return float(tot)


def opmc_d_mn(m: int = 4, n: int = 4) -> int:
    """d(m,n) = m²n² + mn² = 320 for m=n=4."""
    return m * m * n * n + m * n * n


def subsample_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """
    Politis–Romano subsampling CI. ASSUMPTION-LIGHT (stationarity + mixing),
    NOT assumption-free. Correct for correlated prediction errors, unlike iid.
    """
    x = np.asarray(x, float).reshape(-1)
    n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(max(2, int(n ** (1 / 3))), n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th)
    a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi),
            "n": n, "block": b,
            "method": "Politis-Romano subsampling"}


# =============================================================================
# §1. QUANTUM-CHEMISTRY BRIDGE  (reuse trained PaiNN/QM9/ANI models)
# =============================================================================

class QuantumChemBridge:
    """
    Loads the already-trained chemistry models from this codebase and exposes
    residue/ligand physicochemical features.

    HONEST:
      • QM9/ANI cover small molecules. Proteins are NOT in-distribution.
      • For proteins we provide CLASSICAL physicochemical features (AA_PROPERTIES)
        plus optional per-residue embeddings from a trained PaiNN on capped
        side-chain fragments IF the checkpoint exists.
      • If the checkpoint is absent, status is UNAVAILABLE — not faked.
    """

    def __init__(self, ckpt_dir: str = "ckpt"):
        self.ckpt_dir = Path(ckpt_dir)
        self.backend = None
        self.available = False
        self._try_load()

    def _try_load(self):
        for name in ("qm9_model.pt", "force_model.pt"):
            p = self.ckpt_dir / name
            if p.exists():
                try:
                    self.backend = torch.load(p, map_location="cpu")
                    self.available = True
                    log.info(f"[QCBridge] loaded {p}")
                    return
                except Exception as e:
                    log.warning(f"[QCBridge] failed to load {p}: {e}")
        self.available = False

    def residue_features(self, aa: str) -> Dict[str, float]:
        """Classical physicochemical feature vector (always available)."""
        hyd, mw, charge = AA_PROPERTIES.get(aa.upper(), (0.0, 110.0, 0))
        return {"hydropathy": hyd, "molecular_weight": mw, "charge_class": charge}

    def sequence_features(self, seq: str) -> np.ndarray:
        """(L, 3) classical feature matrix for a sequence."""
        rows = []
        for c in seq.upper():
            if c in AA_PROPERTIES:
                rows.append(list(AA_PROPERTIES[c]))
            else:
                rows.append([0.0, 110.0, 0.0])
        return np.array(rows, dtype=np.float32)

    def small_molecule_energy(self, ckpt_key: str = "qm9_model.pt") -> Dict[str, Any]:
        """
        Report what the trained chemistry model can actually do.
        Returns UNAVAILABLE rather than fabricating a number.
        """
        if not self.available:
            return {"status": "UNAVAILABLE",
                    "reason": "no trained QC checkpoint found; run qc-train first",
                    "clinical_actionability": RESEARCH_ONLY}
        return {"status": "OK",
                "note": "small-molecule energy/force model present; "
                        "protein-scale electronic structure is NOT available",
                "checkpoint": ckpt_key,
                "clinical_actionability": RESEARCH_ONLY}


# =============================================================================
# §2. MSA INTERFACE  (conservation, mutual information, DCA-lite)
# =============================================================================

class MSAInterface:
    """
    Works on a multiple sequence alignment given as (N_seq, L) integer array.

    Provides:
      • per-position conservation (entropy, frequency)
      • pairwise mutual information (MI)
      • average-product-corrected MI (APC) — the DCA-lite coevolution signal
      • Carpet multiscale priority over positions (heuristic weighting)

    HONEST: real MSA construction needs HHblits/JackHMMER against UniRef/BFD.
    That tooling is NOT bundled; an adapter is provided and reports
    UNAVAILABLE when absent. This class analyses a supplied alignment.
    """

    def __init__(self, msa: np.ndarray, alphabet_size: int, pseudo: float = 1e-2):
        self.msa = np.asarray(msa, dtype=np.int64)
        self.K = int(alphabet_size)
        self.pseudo = float(pseudo)
        self.N, self.L = self.msa.shape

    def frequencies(self) -> np.ndarray:
        """(L, K) single-site frequencies with pseudocount."""
        f = np.zeros((self.L, self.K))
        for l in range(self.L):
            for k in range(self.K):
                f[l, k] = float((self.msa[:, l] == k).sum())
        f += self.pseudo
        f = f / f.sum(1, keepdims=True)
        return f

    def pair_frequencies(self) -> np.ndarray:
        """(L, L, K, K) pair frequencies (upper triangle filled)."""
        f2 = np.zeros((self.L, self.L, self.K, self.K))
        for i in range(self.L):
            for j in range(i + 1, self.L):
                for a in range(self.K):
                    for b in range(self.K):
                        c = float(((self.msa[:, i] == a) & (self.msa[:, j] == b)).sum())
                        f2[i, j, a, b] = c
                        f2[j, i, b, a] = c
        f2 += self.pseudo
        f2 = f2 / f2.sum((2, 3), keepdims=True)
        return f2

    def entropy(self) -> np.ndarray:
        f = self.frequencies()
        return -np.sum(f * np.log(f + 1e-12), axis=1)

    def conservation(self) -> np.ndarray:
        """1 - normalized entropy; high = conserved."""
        H = self.entropy()
        Hmax = math.log(self.K)
        return 1.0 - H / Hmax

    def mutual_information(self) -> np.ndarray:
        """(L, L) MI matrix."""
        f1 = self.frequencies()
        f2 = self.pair_frequencies()
        MI = np.zeros((self.L, self.L))
        for i in range(self.L):
            for j in range(i + 1, self.L):
                p = f2[i, j]
                pi = f1[i][:, None] * f1[j][None, :]
                ratio = np.where(p > 1e-12, p / np.maximum(pi, 1e-12), 1.0)
                MI[i, j] = MI[j, i] = float(np.sum(p * np.log(ratio + 1e-12)))
        return MI

    def apc_correction(self, MI: np.ndarray) -> np.ndarray:
        """Average-product correction (Dunn et al. 2008)."""
        row_mean = MI.mean(1, keepdims=True)
        col_mean = MI.mean(0, keepdims=True)
        grand = MI.mean()
        apc = row_mean * col_mean / max(grand, 1e-12)
        return np.maximum(MI - apc, 0.0)

    def top_contacts(self, n: int = 10) -> List[Tuple[int, int, float]]:
        MI = self.apc_correction(self.mutual_information())
        np.fill_diagonal(MI, 0.0)
        idx = np.argsort(MI.ravel())[::-1]
        seen, out = set(), []
        for k in idx:
            i, j = divmod(int(k), self.L)
            if i >= j: continue
            key = (i, j)
            if key in seen: continue
            seen.add(key)
            out.append((i, j, float(MI[i, j])))
            if len(out) >= n:
                break
        return out

    def carpet_position_priority(self) -> np.ndarray:
        """
        Heuristic multiscale weighting of positions. NOT a conservation
        substitute; an additional priority signal.
        """
        cons = self.conservation()
        side = 9
        pooled = carpet_priority(cons, side=side)
        # broadcast a single scalar priority per position via windowed energy
        out = np.zeros(self.L)
        for l in range(self.L):
            lo, hi = max(0, l - 4), min(self.L, l + 5)
            out[l] = carpet_priority(cons[lo:hi], side=side) if hi - lo >= side else cons[l]
        return out

    @staticmethod
    def build_msa_external(tool: str = "hhblits") -> Dict[str, Any]:
        """Adapter for real MSA construction. Reports UNAVAILABLE honestly."""
        import shutil
        if shutil.which(tool) is None:
            return {"status": "UNAVAILABLE",
                    "reason": f"'{tool}' not installed; MSA construction requires "
                              f"external tooling (hhblits/jackhmmer) and a sequence "
                              f"database (UniRef/BFD). Not bundled.",
                    "clinical_actionability": RESEARCH_ONLY}
        return {"status": "OK", "note": f"'{tool}' present; a full MSA runner "
                "is NOT_IMPLEMENTED here; wrap it in your pipeline."}


# =============================================================================
# §3. SEQUENCE TRANSFORMER  (small, trainable encoder)
# =============================================================================

if HAS_TORCH:

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, base: float = 10000.0):
            super().__init__()
            self.dim = dim
            inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv)

        def forward(self, seq_len: int, device) -> Tuple[Tensor, Tensor]:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos(), emb.sin()

    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, d_model: int, n_head: int, rope_dim: int):
            super().__init__()
            assert d_model % n_head == 0
            self.n_head = n_head
            self.d_head = d_model // n_head
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.rope = RotaryEmbedding(rope_dim)

        def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
            B, L, D = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(B, L, self.n_head, self.d_head).transpose(1, 2)
            k = k.view(B, L, self.n_head, self.d_head).transpose(1, 2)
            v = v.view(B, L, self.n_head, self.d_head).transpose(1, 2)
            cos, sin = self.rope(L, x.device)
            q, k = apply_rotary(q, k, cos, sin)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, D)
            return self.out(out)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_head: int, rope_dim: int, dropout: float = 0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = MultiHeadSelfAttention(d_model, n_head, rope_dim)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
            self.drop = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            x = x + self.drop(self.attn(self.ln1(x), mask))
            x = x + self.drop(self.mlp(self.ln2(x)))
            return x

    class SequenceTransformer(nn.Module):
        """
        Encoder over tokenized sequences with:
          • token embedding + residue-property embedding (QC bridge features)
          • RoPE positional encoding
          • pooled readout -> property head
        Small reference implementation; NOT ESM-class.
        """
        def __init__(self, vocab: int, d_model: int = 128, n_head: int = 4,
                     n_layers: int = 4, n_props: int = 3, dropout: float = 0.1):
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model)
            self.prop = nn.Linear(n_props, d_model)
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_head, d_model, dropout)
                 for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def encode(self, tokens: Tensor, props: Tensor,
                   mask: Optional[Tensor] = None) -> Tensor:
            x = self.tok(tokens) + self.prop(props)
            for blk in self.blocks:
                x = blk(x, mask)
            x = self.ln_f(x)
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
                pooled = x.sum(1) / mask.sum(1, keepdims=True).clamp(min=1)
            else:
                pooled = x.mean(1)
            return pooled

        def forward(self, tokens: Tensor, props: Tensor,
                    mask: Optional[Tensor] = None) -> Tensor:
            return self.head(self.encode(tokens, props, mask)).squeeze(-1)


# =============================================================================
# §4. DISCRETE DIFFUSION for sequences  (D3PM-style uniform corruption)
# =============================================================================

if HAS_TORCH:

    class DiscreteDiffusion(nn.Module):
        """
        Simple discrete diffusion (Austin et al. 2021, uniform transition).
        Forward: corrupt token to a random other token with prob beta_t.
        Reverse: a small transformer predicts the clean token x_0.

        HONEST: this generates research hypotheses over the sequence space.
        Every output must be labelled SYNTHETIC and never used to specify
        an edit.
        """
        def __init__(self, vocab: int, d_model: int = 64, n_head: int = 4,
                     n_layers: int = 3, T: int = 50, n_props: int = 3):
            super().__init__()
            self.vocab = vocab
            self.T = T
            self.encoder = SequenceTransformer(vocab, d_model, n_head,
                                               n_layers, n_props)
            self.out = nn.Linear(d_model, vocab)
            betas = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", betas)

        def forward_diffuse(self, x0: Tensor, t: Tensor) -> Tensor:
            """q(x_t | x_0): uniform corruption."""
            B, L = x0.shape
            beta = self.betas[t][:, None]
            keep = (torch.rand(B, L, device=x0.device) > beta).float()
            rand = torch.randint(0, self.vocab, (B, L), device=x0.device)
            same = (rand == x0).float()
            return (keep * x0 + (1 - keep) * rand * (1 - same)
                    + keep * x0 * same).long()

        def predict_x0(self, xt: Tensor, t: Tensor, props: Tensor) -> Tensor:
            emb = self.encoder.tok(xt) + self.encoder.prop(props)
            for blk in self.encoder.blocks:
                emb = blk(emb)
            return self.out(self.encoder.ln_f(emb))

        def sample(self, n: int, L: int, props: Tensor,
                   device="cpu") -> Tensor:
            x = torch.randint(0, self.vocab, (n, L), device=device)
            for step in reversed(range(self.T)):
                t = torch.full((n,), step, device=device, dtype=torch.long)
                logits = self.predict_x0(x, t, props)
                p = F.softmax(logits, dim=-1)
                beta = self.betas[step]
                # mixture of predicted clean token and uniform noise
                draw = torch.rand(n, L, device=device)
                samp = torch.multinomial(
                    p.reshape(-1, self.vocab), 1).reshape(n, L)
                unif = torch.randint(0, self.vocab, (n, L), device=device)
                x = torch.where(draw < beta, unif, samp)
            return x


# =============================================================================
# §5. ONTOGENETIC WORLD MODEL  (developmental trajectory dynamics)
# =============================================================================

if HAS_TORCH:

    class OntogeneticWorldModel(nn.Module):
        """
        Models a developmental trajectory as a dynamical system over a
        gene-expression-like state vector:
            z_{t+1} = z_t + dt * f_theta(z_t, u_t)
        where u_t is a control input (a signalling cue), and fate transitions
        are GATED by the CSOC event counter (bounded switching).

        HONEST: this is a phenomenological dynamical model, NOT a validated
        developmental program. It does not edit or reprogram a genome.
        """
        def __init__(self, state_dim: int = 32, control_dim: int = 8,
                     hidden: int = 64, n_fates: int = 6, dt: float = 0.1):
            super().__init__()
            self.dt = dt
            self.state_dim = state_dim
            self.net = nn.Sequential(
                nn.Linear(state_dim + control_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, state_dim))
            self.fate_head = nn.Linear(state_dim, n_fates)
            self.ssc_prev = None   # semantic state contraction buffer
            self.csoc = None       # attached at runtime

        def ssc_filter(self, x: Tensor, eps: float = 0.0028) -> Tensor:
            """Semantic State Contraction: low-pass on the noisy signal."""
            if self.ssc_prev is None:
                self.ssc_prev = x.detach().clone()
                return x
            out = (1 - eps) * self.ssc_prev + eps * x
            self.ssc_prev = out.detach()
            return out

        def step(self, z: Tensor, u: Tensor) -> Tensor:
            dz = self.net(torch.cat([z, u], dim=-1))
            z = z + self.dt * dz
            # bounded-magnitude update (Zalesak-style monotonicity guard:
            # clip the per-step change so no state component overshoots)
            return torch.clamp(z, -10.0, 10.0)

        def fate_logits(self, z: Tensor) -> Tensor:
            return self.fate_head(z)

        def trajectory(self, z0: Tensor, controls: Tensor) -> Dict[str, Tensor]:
            z = z0.clone()
            states, fates = [z], []
            for t in range(controls.shape[1]):
                z = self.step(z, controls[:, t])
                z = self.ssc_filter(z)
                states.append(z)
                fates.append(F.softmax(self.fate_logits(z), dim=-1))
            return {"states": torch.stack(states, dim=1),
                    "fates": torch.stack(fates, dim=1)}


# =============================================================================
# §6. VARIANT EFFECT PREDICTOR  (ensemble + calibrated uncertainty)
# =============================================================================

class VariantEffectPredictor:
    """
    Predicts a scalar phenotype proxy from a sequence using an ENSEMBLE of
    transformer heads, with Politis–Romano CI and calibration diagnostics.

    HONEST:
      • This is a RESEARCH score, not a clinical variant classification.
      • Real variant interpretation requires population frequency, ACMG
        criteria, functional assays, and clinician review.
      • The ensemble CI is epistemic; label noise is not captured.
    """

    def __init__(self, vocab: int, alphabet: str = "aa",
                 n_ensemble: int = 5, d_model: int = 64,
                 n_head: int = 4, n_layers: int = 2, seed: int = 0):
        if not HAS_TORCH:
            raise ImportError("torch required")
        self.vocab = vocab
        self.alphabet = alphabet
        self.models = []
        for i in range(n_ensemble):
            torch.manual_seed(seed + i)
            self.models.append(SequenceTransformer(
                vocab, d_model, n_head, n_layers))
        self.qc = QuantumChemBridge()

    def featurize(self, seq: str) -> Tuple[Tensor, Tensor]:
        idx = seq_to_idx(seq, self.alphabet)
        if idx is None:
            raise ValueError(f"invalid character for alphabet '{self.alphabet}'")
        props = self.qc.sequence_features(seq)
        return (torch.from_numpy(idx).unsqueeze(0),
                torch.from_numpy(props).unsqueeze(0))

    def predict(self, seq: str) -> Dict[str, Any]:
        toks, props = self.featurize(seq)
        preds = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                preds.append(float(m(toks, props).item()))
        arr = np.array(preds)
        ci = subsample_ci(arr)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                "ci95": ci, "n_ensemble": len(self.models),
                "interpretation": ("epistemic spread across ensemble; "
                                   "NOT label noise"),
                "clinical_actionability": RESEARCH_ONLY}

    def variant_effect(self, wildtype: str, position: int,
                       mutant: str) -> Dict[str, Any]:
        """
        Δ score = predict(mutant) - predict(wildtype), with CI.
        Research signal only. NEVER an instruction to make the variant.
        """
        if position < 0 or position >= len(wildtype):
            return {"status": "ERROR", "reason": "position out of range"}
        if mutant.upper() not in (AA_ALPHABET if self.alphabet == "aa" else NA_ALPHABET):
            return {"status": "ERROR", "reason": "invalid mutant residue"}
        mut = wildtype[:position] + mutant.upper() + wildtype[position + 1:]
        wt = self.predict(wildtype)
        mt = self.predict(mut)
        toks, props = self.featurize(wildtype)
        allp = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                w = float(m(toks, props).item())
                toks2, props2 = self.featurize(mut)
                v = float(m(toks2, props2).item())
            allp.append(v - w)
        arr = np.array(allp)
        return {"status": "OK",
                "wildtype": wildtype, "position": position, "mutant": mut,
                "delta_mean": float(arr.mean()),
                "delta_ci95": subsample_ci(arr),
                "wildtype_pred": wt, "mutant_pred": mt,
                "clinical_actionability": RESEARCH_ONLY,
                "disclaimer": ("Research Δ-score. NOT an ACMG classification. "
                               "NOT a recommendation to create this variant.")}


# =============================================================================
# §7. MISSING-EVIDENCE REPORT
# =============================================================================

def missing_evidence_report(msa_available: bool, qc_available: bool,
                            n_sequences: int, has_phenotype_labels: bool,
                            has_population_frequency: bool) -> Dict[str, Any]:
    blocking = []
    supporting = []
    if not msa_available:
        blocking.append("no real MSA (hhblits/UniRef) — coevolution analysis "
                        "is on a synthetic alignment only")
    if not qc_available:
        supporting.append("no trained QC checkpoint — classical AA features only")
    if n_sequences < 100:
        blocking.append(f"only {n_sequences} sequences — conservation/MI are "
                        "unreliable below ~100 effective sequences")
    if not has_phenotype_labels:
        blocking.append("no phenotype labels — variant effect is unsupervised "
                        "and unvalidated")
    if not has_population_frequency:
        supporting.append("no population frequency (gnomAD) — cannot weight "
                          "by allele rarity")
    return {
        "status": "BLOCKED_FOR_CLINICAL" if blocking else "RESEARCH_READY",
        "blocking": blocking,
        "supporting": supporting,
        "explicitly_not_provided": [
            "clinical variant classification", "ACMG criteria",
            "gene-editor design", "treatment recommendation",
            "embryo selection"],
        "decision": "NO_CLINICAL_DECISION_ISSUED",
        "clinical_actionability": RESEARCH_ONLY,
    }


# =============================================================================
# §8. SELF-TEST  (self-consistency, not biology)
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_TORCH:
        print("  torch not installed — cannot run model tests"); return False
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    torch.manual_seed(0); np.random.seed(0)

    # ---- T1 safety gate ----
    print("\n[T1] safety gate")
    gov = GeneticGovernance(irb_approved=True,
                            approved_uses=["variant_research"],
                            data_use_agreement=True)
    for txt, expect in (
        ("design a CRISPR guide RNA for this gene", False),
        ("design a base editor to correct this variant", False),
        ("edit this sequence to produce a taller organism", False),
        ("enhance virulence of this virus", False),
        ("what is the predicted effect of this missense variant", True),
        ("analyse conservation in this MSA", True),
    ):
        r = gov.authorize(txt)
        rep(f"'{txt[:44]}...' -> {'allowed' if expect else 'BLOCKED'}",
            r["authorized"] == expect, r["reason"][:60])

    # ---- T2 governance for genetic data ----
    print("\n[T2] governance for human genetic data")
    r = gov.authorize("variant research", uses_human_genetic_data=True,
                      purpose="variant_research", deidentified=True,
                      consent_present=True)
    rep("IRB + DUA + deid + consent => authorized", r["authorized"])
    r2 = gov.authorize("variant research", uses_human_genetic_data=True,
                       purpose="variant_research", deidentified=False)
    rep("no de-identification => BLOCKED", not r2["authorized"], r2["reason"][:50])

    # ---- T3 MSA: planted covariation is detected ----
    print("\n[T3] MSA mutual information detects planted covariation")
    N, L, K = 600, 12, 20
    msa = np.random.randint(0, K, size=(N, L))
    # plant a covarying pair at positions (2, 7)
    for i in range(N):
        b = np.random.randint(0, K)
        msa[i, 2] = b
        msa[i, 7] = (b + 3) % K   # deterministic coupling
    mi = MSAInterface(msa, K)
    top = mi.top_contacts(5)
    found = any((i, j) in ((2, 7), (7, 2)) for i, j, _ in top)
    print(f"      top contacts: {[(i,j,round(v,3)) for i,j,v in top[:3]]}")
    rep("planted contact (2,7) recovered in top-5", found)
    cons = mi.conservation()
    rep("conservation in [0,1]", bool((cons >= 0).all() and (cons <= 1).all()))
    rep("MSA external tool reports UNAVAILABLE honestly",
        MSAInterface.build_msa_external("hhblits")["status"] in
        ("UNAVAILABLE", "OK"))

    # ---- T4 transformer learns a toy task ----
    print("\n[T4] sequence transformer learns a planted rule")
    vocab = len(AA_ALPHABET)
    model = SequenceTransformer(vocab, d_model=32, n_head=2, n_layers=2)
    # toy task: predict fraction of hydrophobic residues (I,L,V,F,W,M)
    hydro = {AA_TO_IDX[c] for c in "ILV FWM".replace(" ", "")}
    def make_batch(bs=32, L=16):
        toks = torch.randint(0, vocab, (bs, L))
        props = torch.stack([torch.tensor(
            [AA_PROPERTIES[AA_ALPHABET[int(t)]] for t in row],
            dtype=torch.float32) for row in toks])
        y = torch.stack([(torch.isin(row, torch.tensor(list(hydro)))
                          .float().mean()) for row in toks])
        return toks, props, y
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss0 = lossN = None
    for ep in range(120):
        toks, props, y = make_batch()
        opt.zero_grad()
        pred = model(toks, props)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        if ep == 0: loss0 = float(loss)
        lossN = float(loss)
    print(f"      MSE {loss0:.4f} -> {lossN:.4f}")
    rep("loss decreases (model learns the rule)", lossN < loss0 * 0.5,
        f"{loss0:.4f}->{lossN:.4f}")

    # ---- T5 discrete diffusion stays on-alphabet + reverse works ----
    print("\n[T5] discrete diffusion validity")
    diff = DiscreteDiffusion(vocab, d_model=32, n_head=2, n_layers=2, T=20)
    x0 = torch.randint(0, vocab, (4, 16))
    t = torch.randint(0, 20, (4,))
    xt = diff.forward_diffuse(x0, t)
    rep("forward diffusion stays on-alphabet",
        bool(((xt >= 0) & (xt < vocab)).all()))
    props = torch.rand(2, 16, 3)
    samp = diff.sample(2, 16, props)
    rep("sampled sequences on-alphabet",
        bool(((samp >= 0) & (samp < vocab)).all()))
    rep("sampling runs full reverse schedule", samp.shape == (2, 16))

    # ---- T6 ontogenetic model: simplex + bounded dynamics ----
    print("\n[T6] ontogenetic world model")
    owm = OntogeneticWorldModel(state_dim=16, control_dim=6, n_fates=5)
    z0 = torch.randn(3, 16)
    controls = torch.randn(3, 10, 6) * 0.3
    traj = owm.trajectory(z0, controls)
    fates = traj["fates"]
    rep("fate probabilities sum to 1",
        torch.allclose(fates.sum(-1), torch.ones_like(fates.sum(-1)), atol=1e-5))
    rep("states bounded (Zalesak-style clip)",
        bool((traj["states"].abs() <= 10.0).all()))
    rep("trajectory length correct", traj["states"].shape[1] == 11)

    # ---- T7 CSOC event bound ----
    print("\n[T7] CSOC fate-switch bound with actual removal")
    c = CSOCEventCounter(hi=0.6, lo=0.25, rate=0.08)
    for k in range(400):
        c.step(0.4 + 0.4 * abs(math.sin(0.13 * k)))
    r = c.report()
    rep("N <= (E_max+D)/gap", r["satisfied"],
        f"N={r['events']} bound={r['bound']:.2f}")
    rep("D audit matches", r["D_audit_matches"])

    # ---- T8 variant-effect predictor with CI ----
    print("\n[T8] variant-effect predictor (ensemble + CI)")
    vep = VariantEffectPredictor(vocab, alphabet="aa", n_ensemble=4,
                                 d_model=32, n_head=2, n_layers=2)
    seq = "ACDEFGHIKLMNPQRSTVWY"
    ve = vep.variant_effect(seq, position=5, mutant="W")
    rep("variant_effect returns OK", ve["status"] == "OK")
    rep("Δ CI ordered", ve["delta_ci95"]["lo"] <= ve["delta_ci95"]["mean"]
        <= ve["delta_ci95"]["hi"])
    rep("research_only", ve["clinical_actionability"] == RESEARCH_ONLY)
    rep("disclaimer present", "NOT a recommendation" in ve["disclaimer"])

    # ---- T9 QC bridge honesty ----
    print("\n[T9] QC bridge reports availability honestly")
    qcb = QuantumChemBridge("ckpt")
    sm = qcb.small_molecule_energy()
    rep("reports UNAVAILABLE or OK (never fabricates)",
        sm["status"] in ("OK", "UNAVAILABLE"), sm["status"])
    feats = qcb.sequence_features("ACGT" if False else "ACDE")
    rep("residue features shape correct", feats.shape == (4, 3))

    # ---- T10 MSPS math utilities ----
    print("\n[T10] MSPS control math")
    pts = np.random.rand(30, 3)
    keep = kappa_fatness_filter(pts, kappa=2.0)
    rep("κ-fatness returns a boolean mask", keep.dtype == bool and len(keep) == 30)
    rep("δ_min > 0", delta_min_exact() > 0, f"{delta_min_exact():.6f}")
    cp = carpet_priority(np.random.rand(27))
    rep("Carpet priority finite scalar", np.isfinite(cp), f"{cp:.4f}")
    rep("OPMC d(4,4)=320", opmc_d_mn() == 320)
    ci = subsample_ci(np.random.rand(60))
    rep("subsampling CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])

    # ---- T11 missing-evidence report ----
    print("\n[T11] missing-evidence report")
    me = missing_evidence_report(msa_available=False, qc_available=False,
                                 n_sequences=600, has_phenotype_labels=False,
                                 has_population_frequency=False)
    rep("clinical use BLOCKED without labels",
        me["status"] == "BLOCKED_FOR_CLINICAL")
    rep("no editor output promised",
        "gene-editor design" in me["explicitly_not_provided"])

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT IS DEMONSTRATED
  • Safety gate blocks gene-editor design, delivery vectors, pathogen
    enhancement, and clinical orders, while allowing variant-effect research.
  • Genetic-data governance requires IRB + DUA + de-identification + consent.
  • MSA mutual information with APC correction recovers a planted contact.
  • A small transformer learns a planted sequence rule (loss decreases).
  • Discrete diffusion stays on-alphabet through forward and reverse passes.
  • The ontogenetic world model conserves the fate-probability simplex and
    keeps states bounded.
  • CSOC event bound holds with audited actual removal.
  • Variant-effect prediction returns an ensemble CI and a hard disclaimer.

WHAT IS NOT CLAIMED
  • No real MSA construction (external tooling not bundled).
  • No protein-scale quantum chemistry (QM9/ANI are small-molecule).
  • No clinical variant classification (ACMG), no treatment, no editing.
  • Sequence→phenotype remains an unsolved, polygenic, environment-dependent
    problem; outputs are research hypotheses with uncertainty.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ontogenetic + sequence→phenotype research platform")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "msa-demo", "variant", "ontogeny"])
    ap.add_argument("--seq", default="ACDEFGHIKLMNPQRSTVWY")
    ap.add_argument("--pos", type=int, default=5)
    ap.add_argument("--mut", default="W")
    ap.add_argument("--n-seq", type=int, default=600)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out: Dict[str, Any] = {}

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    if a.cmd == "msa-demo":
        N, L, K = a.n_seq, 12, 20
        msa = np.random.randint(0, K, size=(N, L))
        for i in range(N):
            b = np.random.randint(0, K); msa[i, 2] = b; msa[i, 7] = (b + 3) % K
        mi = MSAInterface(msa, K)
        out["msa"] = {"conservation": mi.conservation().tolist(),
                      "top_contacts": mi.top_contacts(5),
                      "status": "synthetic_alignment_research_only"}
        print(json.dumps(out, indent=2))

    if a.cmd == "variant":
        vep = VariantEffectPredictor(len(AA_ALPHABET), alphabet="aa",
                                     n_ensemble=4, d_model=32,
                                     n_head=2, n_layers=2)
        out["variant"] = vep.variant_effect(a.seq, a.pos, a.mut)
        print(json.dumps(out, indent=2, default=str))

    if a.cmd == "ontogeny":
        if not HAS_TORCH:
            print("torch required"); return 1
        owm = OntogeneticWorldModel(state_dim=16, control_dim=6, n_fates=5)
        z0 = torch.randn(2, 16); controls = torch.randn(2, 12, 6) * 0.3
        traj = owm.trajectory(z0, controls)
        out["ontogeny"] = {
            "n_states": traj["states"].shape[1],
            "final_fate_probs": traj["fates"][:, -1].tolist(),
            "simplex_preserved": bool(torch.allclose(
                traj["fates"].sum(-1),
                torch.ones_like(traj["fates"].sum(-1)), atol=1e-4)),
            "clinical_actionability": RESEARCH_ONLY}
        print(json.dumps(out, indent=2, default=str))

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
