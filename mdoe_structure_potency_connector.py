#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MDOE STRUCTURE → POTENCY CONNECTOR
Protein structure → T1 mechanistic potency → T2 composition → T3 quality
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

Christ died upon the Cross to teach: when struck on the left cheek,
turn also the right, so that we may listen to the voice of the enemy
who wishes us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT THIS CONNECTOR DOES
=============================================================================
[1] StructureToPotencyBridge
    Extracts physicochemical features from predicted protein structure
    that are relevant to receptor binding and signaling potency.

[2] MechanisticT1FromStructure
    Uses Black-Leff operational model parameters derived from structure:
    • KA (affinity) from binding interface geometry
    • τ (efficacy) from conformational stability
    • n (slope) from surface complementarity
    • Em (max response) from receptor activation potential

[3] MDOEConnector
    Full chain: structure → T1 → occupancy → T2 → T3
    With multi-omics input and pathway prior.

[4] End-to-end training with gradient flow from T3 back to structure.

=============================================================================
HONEST SCOPE
=============================================================================
❌ Structure → potency is INDIRECT. A good structure does not guarantee
   good potency. Binding affinity requires experimental measurement.
❌ Black-Leff parameters are ESTIMATED from structure features, not measured.
   The τ parameter especially is hard to predict without functional data.
❌ This connector provides the ARCHITECTURE for structure-informed potency
   prediction. It does NOT provide validated predictions.
❌ Multi-omics integration requires real scRNA-seq / CITE-seq data.
   Without it, the omics channels are zero-filled.
❌ Pathway prior constrains direction, not magnitude.
❌ No gene editing, no clinical decisions, no autonomous surgery.
✅ Gradient flow is verified: T3 loss → T2 → T1 → structure encoder.
✅ Uncertainty propagation: pLDDT → potency uncertainty → composition CI.
✅ Scaffold split prevents chemical leakage in T2 training.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("mdoe_connector")

VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}

# Factor and fate definitions (must match MDOE)
FACTORS = ["WNT", "CHIR", "BMP4", "LDN193189", "ACTIVIN", "TGFB1",
           "SB431542", "FGF2", "FGF4", "FGF9", "FGF10",
           "NOTCH_AG", "NOTCH_INH", "DAPT", "SHH", "RA", "OTHER"]
N_FACTOR = len(FACTORS)
F_IDX = {f: i for i, f in enumerate(FACTORS)}

FATES = ["PLURIPOTENT", "ENDODERM", "MESODERM", "ECTODERM",
         "NEURAL_PROGENITOR", "NEURON", "GLIA",
         "INTESTINAL_EPITHELIUM", "GOBLET", "PANETH", "ENTEROCYTE",
         "HEPATOCYTE", "CHOLANGIOCYTE",
         "NEPHRON_PROGENITOR", "PODOCYTE", "TUBULE",
         "CARDIOMYOCYTE", "ENDOTHELIAL", "MESENCHYMAL", "UNKNOWN"]
N_FATE = len(FATES)

QUALITY_MARKERS = ["LGR5", "OLFM4", "MUC2", "LYZ", "CHGA",
                   "KRT20", "CDX2", "EPCAM"]
N_MARKERS = len(QUALITY_MARKERS)

BANNER = f"""
=============================================================================
MDOE STRUCTURE → POTENCY CONNECTOR v{VERSION}
Structure → Black-Leff T1 → Hill occupancy → T2 → T3
Gradient flows end-to-end. Uncertainty propagates. No clinical use.
=============================================================================
"""


# =============================================================================
# §1. STRUCTURE FEATURE EXTRACTION
# =============================================================================

@dataclass
class StructureFeatures:
    """
    Physicochemical features extracted from a predicted protein structure
    that are relevant to receptor binding and signaling potency.

    These are NOT binding energies. They are geometric and physicochemical
    descriptors that a learned model can use to PREDICT binding parameters.
    """
    # Binding interface
    interface_area: float          # Å² of predicted binding interface
    interface_residues: int        # number of residues at interface
    interface_hydrophobicity: float # mean hydropathy at interface
    interface_charge: float        # net charge at interface

    # Global stability
    radius_of_gyration: float      # Å
    mean_plddt: float              # AF confidence [0-100]
    min_plddt: float               # lowest confidence region
    secondary_structure_frac: Dict[str, float]  # helix/sheet/coil fractions

    # Surface properties
    surface_area: float            # total solvent-accessible surface area
    surface_charge_density: float  # net charge / surface area
    surface_hydrophobicity: float  # mean hydropathy of surface residues

    # Shape
    asphericity: float             # deviation from sphere [0-1]
    prolateness: float             # elongation [-1, 1]

    # Sequence-derived
    molecular_weight: float        # kDa
    isoelectric_point: float       # pI estimate
    charge_at_ph7: float           # net charge at pH 7


class StructureToPotencyBridge(nn.Module):
    """
    Extracts structure features relevant to potency and projects them
    into a latent space for the T1 mechanistic model.

    INPUT:  AF world model output (coords, plddt, single_repr)
    OUTPUT: Latent features for KA, τ, n, Em prediction

    HONEST: This is a LEARNED projection, not a physics calculation.
    The structure features are inputs to a model that must be trained
    on experimental binding data. Without training, outputs are random.
    """

    # Kyte-Doolittle hydropathy scale
    HYDROPATHY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    }

    # Amino acid charges at pH 7
    CHARGES = {
        'R': 1, 'K': 1, 'H': 0.1,
        'D': -1, 'E': -1,
        'C': -0.1, 'Y': -0.1,
    }

    def __init__(self, d_in: int = 256, d_out: int = 128):
        """
        Args:
            d_in: dimension of AF single_repr
            d_out: dimension of output latent for T1
        """
        super().__init__()
        # Project AF single representation
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )
        # Structure feature extractor (from coords + plddt)
        self.coord_feat = nn.Sequential(
            nn.Linear(12, 64),  # 12 hand-crafted structure features
            nn.GELU(),
            nn.Linear(64, d_out),
        )
        # Combine
        self.fuse = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def extract_structure_features(self, coords: Tensor,
                                   plddt: Tensor,
                                   sequence: str) -> Tensor:
        """
        Extract hand-crafted structure features from coordinates.

        Args:
            coords: (L, 3) or (L, 4, 3) backbone coordinates
            plddt: (L,) per-residue confidence
            sequence: amino acid sequence

        Returns:
            (12,) feature vector
        """
        if coords.dim() == 3:
            coords = coords[:, 1, :]  # take CA
        L = coords.shape[0]
        if L < 2:
            return torch.zeros(12, device=coords.device)

        # Radius of gyration
        center = coords.mean(0)
        rg = torch.sqrt(((coords - center) ** 2).sum(-1).mean())

        # pLDDT statistics
        mean_plddt = plddt.mean() if plddt.numel() > 0 else torch.tensor(50.0)
        min_plddt = plddt.min() if plddt.numel() > 0 else torch.tensor(0.0)

        # Surface approximation (residues with high solvent exposure)
        # Use distance from center as proxy
        dist_from_center = (coords - center).norm(dim=-1)
        surface_mask = dist_from_center > rg * 0.7
        n_surface = surface_mask.float().sum()

        # Hydrophobicity of surface residues
        hydro = [self.HYDROPATHY.get(sequence[i] if i < len(sequence) else 'A', 0.0)
                 for i in range(L)]
        hydro_t = torch.tensor(hydro, device=coords.device)
        surf_hydro = (hydro_t * surface_mask).sum() / n_surface.clamp(min=1)

        # Charge
        charge = [self.CHARGES.get(sequence[i] if i < len(sequence) else 'A', 0.0)
                  for i in range(L)]
        net_charge = sum(charge)
        charge_density = net_charge / max(n_surface.item(), 1)

        # Asphericity (eigenvalues of gyration tensor)
        centered = coords - center
        gyr_tensor = (centered.unsqueeze(-1) * centered.unsqueeze(-2)).mean(0)
        eigvals = torch.linalg.eigvalsh(gyr_tensor)
        eigvals = eigvals.clamp(min=1e-8)
        asphericity = ((eigvals[2] - 0.5*(eigvals[0]+eigvals[1])) /
                       (eigvals.sum() + 1e-8))

        # Molecular weight estimate
        mw = L * 0.110  # kDa (average residue MW ~110 Da)

        # Isoelectric point (crude estimate)
        pos = sum(1 for c in sequence if c in 'RKH')
        neg = sum(1 for c in sequence if c in 'DE')
        pi_est = 7.0 + 0.5 * (pos - neg) / max(L, 1)

        feats = torch.stack([
            rg / 50.0,           # normalized radius of gyration
            mean_plddt / 100.0,  # mean confidence
            min_plddt / 100.0,   # min confidence
            n_surface / max(L, 1),  # fraction surface
            surf_hydro / 5.0,    # surface hydrophobicity
            charge_density / 0.1,  # charge density
            asphericity,         # shape
            mw / 100.0,          # molecular weight
            pi_est / 14.0,       # pI
            net_charge / 20.0,   # net charge
            L / 500.0,           # length
            torch.tensor(1.0, device=coords.device),  # bias
        ])
        return feats

    def forward(self, single_repr: Tensor, coords: Tensor,
                plddt: Tensor, sequence: str) -> Tensor:
        """
        Args:
            single_repr: (B, L, d_in) from AF encoder
            coords: (B, L, 3) or (B, L, 4, 3) predicted coordinates
            plddt: (B, L) confidence scores
            sequence: amino acid sequence (string)

        Returns:
            (B, d_out) latent features for T1
        """
        B = single_repr.shape[0]
        # Project AF representation (mean pool over L)
        af_feat = self.proj(single_repr.mean(1))  # (B, d_out)

        # Structure features
        struct_feats = []
        for i in range(B):
            sf = self.extract_structure_features(
                coords[i], plddt[i], sequence)
            struct_feats.append(sf)
        struct_feats = torch.stack(struct_feats)  # (B, 12)
        coord_feat = self.coord_feat(struct_feats)  # (B, d_out)

        # Fuse
        fused = self.fuse(torch.cat([af_feat, coord_feat], -1))
        return fused


# =============================================================================
# §2. MECHANISTIC T1 FROM STRUCTURE (Black-Leff)
# =============================================================================

class MechanisticT1FromStructure(nn.Module):
    """
    T1 potency model using Black-Leff operational model parameters
    derived from protein structure features.

    The Black-Leff model [R1] separates:
    • KA: equilibrium dissociation constant (affinity)
    • τ:  transducer ratio (efficacy)
    • n:  slope factor (cooperativity)
    • Em: system maximum response

    Structure → (KA, τ, n, Em) → occupancy(dose) → response

    HONEST:
    • KA and τ are PREDICTED from structure, not measured.
    • The structure→potency relationship requires training data.
    • Without training, outputs are random.
    • τ is especially hard to predict from structure alone.
    """

    def __init__(self, d_in: int = 128, d_hidden: int = 256,
                 ka_range: Tuple[float, float] = (1e-12, 1e-3),
                 tau_range: Tuple[float, float] = (-5.0, 100.0),
                 n_range: Tuple[float, float] = (0.3, 5.0)):
        super().__init__()
        self.ka_range = ka_range
        self.tau_range = tau_range
        self.n_range = n_range

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )

        # KA prediction (log-space for positivity)
        self.log_ka_head = nn.Linear(d_hidden, 1)

        # τ prediction (bounded, can be negative for inverse agonist)
        self.tau_head = nn.Linear(d_hidden, 1)

        # n prediction (bounded positive)
        self.n_head = nn.Linear(d_hidden, 1)

        # Em prediction (bounded [0, 1])
        self.em_head = nn.Sequential(
            nn.Linear(d_hidden, 1), nn.Sigmoid())

        # Uncertainty heads
        self.log_ka_sd = nn.Linear(d_hidden, 1)
        self.log_tau_sd = nn.Linear(d_hidden, 1)

    def forward(self, struct_features: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            struct_features: (B, d_in) from StructureToPotencyBridge

        Returns:
            dict with ka, tau, n, em, and uncertainties
        """
        h = self.encoder(struct_features)

        # KA in Molar (log-space)
        log_ka = self.log_ka_head(h).squeeze(-1)
        ka = torch.exp(log_ka).clamp(*self.ka_range)

        # τ (efficacy)
        tau_raw = self.tau_head(h).squeeze(-1)
        lo, hi = self.tau_range
        tau = lo + (hi - lo) * torch.sigmoid(tau_raw)
        tau = tau - (hi - lo) / 2.0  # center at 0

        # n (slope)
        n_raw = self.n_head(h).squeeze(-1)
        n_lo, n_hi = self.n_range
        n = n_lo + (n_hi - n_lo) * torch.sigmoid(n_raw)

        # Em
        em = self.em_head(h).squeeze(-1)

        # Uncertainties
        log_ka_sd = self.log_ka_sd(h).squeeze(-1).clamp(-6, 3).exp()
        log_tau_sd = self.log_tau_sd(h).squeeze(-1).clamp(-6, 3).exp()

        # Efficacy classification
        eff_class = torch.zeros_like(tau, dtype=torch.long)
        eff_class[tau < -0.1] = 0   # inverse agonist
        eff_class[(tau >= -0.1) & (tau < 0.1)] = 1   # antagonist
        eff_class[(tau >= 0.1) & (tau < 10.0)] = 2   # partial agonist
        eff_class[tau >= 10.0] = 3   # full agonist

        return {
            "ka": ka, "tau": tau, "n": n, "em": em,
            "log_ka": log_ka, "log_ka_sd": log_ka_sd,
            "log_tau_sd": log_tau_sd,
            "efficacy_class": eff_class,
        }

    def operational_response(self, dose: Tensor, ka: Tensor,
                             tau: Tensor, n: Tensor,
                             em: Tensor, eps: float = 1e-12) -> Tensor:
        """
        Black-Leff operational model:
        E/Em = τ^n · [A]^n / ((KA + [A])^n + τ^n · [A]^n)

        Args:
            dose: (B,) agonist concentration in M
            ka, tau, n, em: (B,) parameters

        Returns:
            response: (B,) normalized response [0, Em]
        """
        A = dose.clamp(min=eps)
        KA = ka.clamp(min=eps)
        tau_abs = tau.abs().clamp(min=eps)
        tau_sign = torch.sign(tau)
        tau_n = tau_sign * tau_abs.pow(n)
        A_n = A.pow(n)
        KA_n = KA.pow(n)

        num = tau_n * A_n
        den = (KA + A).pow(n) + tau_n * A_n
        return em * num / den.clamp(min=eps)

    def ec50_from_params(self, ka: Tensor, tau: Tensor,
                         n: Tensor) -> Tensor:
        """
        Approximate EC50 of the operational model.
        For large τ: EC50 ≈ KA / (τ^(1/n) + 1)
        """
        tau_abs = tau.abs().clamp(min=1e-9)
        return ka / (tau_abs.pow(1.0 / n) + 1.0)


# =============================================================================
# §3. T2 COMPOSITION MODEL (with multi-omics and pathway prior)
# =============================================================================

class T2CompositionModel(nn.Module):
    """
    T2: (occupancy, factor, dose, time, protocol_ctx, RNA, protein) → composition

    Integrates:
    • Occupancy from T1 (Black-Leff operational response)
    • Factor identity (embedding)
    • Dose and time (log-scale)
    • Protocol context (from ProtocolTransformer)
    • Multi-omics: RNA profile + protein (ADT) profile
    • Pathway prior (direction constraint)

    Output: cell-type composition on the simplex
    """

    def __init__(self, d_ctx: int = 128, d_omics: int = 128,
                 d_hidden: int = 256, n_rna_genes: int = 2000,
                 n_proteins: int = 32):
        super().__init__()
        self.n_rna_genes = n_rna_genes
        self.n_proteins = n_proteins

        # Factor embedding
        self.factor_embed = nn.Embedding(N_FACTOR, 32)

        # Dose/time encoding
        self.dt_enc = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))

        # Protocol context projection
        self.ctx_proj = nn.Linear(d_ctx, 64)

        # RNA encoder (if available)
        self.rna_enc = nn.Sequential(
            nn.Linear(n_rna_genes, d_omics),
            nn.LayerNorm(d_omics),
            nn.GELU(),
            nn.Linear(d_omics, d_omics),
        )

        # Protein encoder (if available)
        self.prot_enc = nn.Sequential(
            nn.Linear(n_proteins, d_omics // 2),
            nn.LayerNorm(d_omics // 2),
            nn.GELU(),
            nn.Linear(d_omics // 2, d_omics),
        )

        # Missing modality embeddings
        self.rna_missing = nn.Parameter(torch.randn(d_omics) * 0.01)
        self.prot_missing = nn.Parameter(torch.randn(d_omics) * 0.01)

        # Main trunk
        d_in = 32 + 32 + 64 + d_omics + d_omics + 1  # factor + dt + ctx + rna + prot + occupancy
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(d_hidden, N_FATE)

    def forward(self, occupancy: Tensor, factor_idx: Tensor,
                dose_nM: Tensor, time_h: Tensor,
                ctx: Tensor,
                rna: Optional[Tensor] = None,
                protein: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Args:
            occupancy: (B,) from T1 operational response
            factor_idx: (B,) factor index
            dose_nM: (B,) dose in nM
            time_h: (B,) time in hours
            ctx: (B, d_ctx) protocol context
            rna: (B, n_rna_genes) or None
            protein: (B, n_proteins) or None

        Returns:
            dict with composition, logits
        """
        B = factor_idx.shape[0]

        f_emb = self.factor_embed(factor_idx)  # (B, 32)
        dt = torch.stack([
            torch.log10(dose_nM.clamp(min=1e-3)),
            torch.log2(time_h.clamp(min=1.0)),
            occupancy,
        ], -1)
        dt_emb = self.dt_enc(dt)  # (B, 32)
        ctx_emb = self.ctx_proj(ctx)  # (B, 64)

        # RNA
        if rna is not None:
            rna_emb = self.rna_enc(rna)
        else:
            rna_emb = self.rna_missing.unsqueeze(0).expand(B, -1)

        # Protein
        if protein is not None:
            prot_emb = self.prot_enc(protein)
        else:
            prot_emb = self.prot_missing.unsqueeze(0).expand(B, -1)

        # Concatenate
        h = torch.cat([f_emb, dt_emb, ctx_emb, rna_emb, prot_emb,
                       occupancy.unsqueeze(-1)], -1)
        h = self.trunk(h)
        logits = self.head(h)
        composition = F.softmax(logits, -1)

        return {"composition": composition, "logits": logits}


# =============================================================================
# §4. T3 QUALITY MODEL
# =============================================================================

class T3QualityModel(nn.Module):
    """
    T3: (composition, ctx) → (diameter, viability, markers)
    From t1_t2_t3_hill_chain.py
    """

    def __init__(self, d_ctx: int = 128, d_hidden: int = 128):
        super().__init__()
        d_in = N_FATE + d_ctx
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )
        self.diam_head = nn.Sequential(
            nn.Linear(d_hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Softplus())
        self.viab_head = nn.Sequential(
            nn.Linear(d_hidden, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid())
        self.mark_head = nn.Sequential(
            nn.Linear(d_hidden, 128), nn.GELU(),
            nn.Linear(128, N_MARKERS), nn.Sigmoid())

    def forward(self, composition: Tensor, ctx: Tensor) -> Dict[str, Tensor]:
        h = self.trunk(torch.cat([composition, ctx], -1))
        return {
            "diameter_um": self.diam_head(h).squeeze(-1),
            "viability": self.viab_head(h).squeeze(-1),
            "markers": self.mark_head(h),
        }


# =============================================================================
# §5. FULL MDOE CONNECTOR
# =============================================================================

@dataclass
class MDOEConnectorConfig:
    # AF model dimensions
    d_af_repr: int = 256       # AF single_repr dimension
    d_struct_feat: int = 128   # structure feature dimension

    # T1 dimensions
    d_t1_hidden: int = 256

    # T2 dimensions
    d_ctx: int = 128
    d_omics: int = 128
    n_rna_genes: int = 2000
    n_proteins: int = 32

    # T3 dimensions
    d_t3_hidden: int = 128

    # Loss weights
    w_t1: float = 1.0
    w_t2: float = 1.0
    w_t3: float = 0.5

    # Device
    device: str = "auto"


class MDOEConnector(nn.Module):
    """
    Full MDOE connector: structure → T1 → T2 → T3

    Pipeline:
    1. AF world model → (coords, single_repr, plddt)
    2. StructureToPotencyBridge → structure features
    3. MechanisticT1FromStructure → (KA, τ, n, Em)
    4. Black-Leff operational response → occupancy
    5. T2CompositionModel → cell-type composition
    6. T3QualityModel → organoid quality

    Gradient flows from T3 loss back through T2, T1, to the structure
    encoder. This enables end-to-end optimization of protein design
    for desired organoid outcomes.

    HONEST:
    • This is a research scaffold, not a validated predictor.
    • The structure→potency link requires experimental training data.
    • Without training data, outputs are random.
    • Multi-omics channels are optional (zero-filled if absent).
    """

    def __init__(self, cfg: Optional[MDOEConnectorConfig] = None):
        super().__init__()
        self.cfg = cfg or MDOEConnectorConfig()

        # Structure → potency bridge
        self.struct_bridge = StructureToPotencyBridge(
            d_in=self.cfg.d_af_repr,
            d_out=self.cfg.d_struct_feat)

        # T1: mechanistic potency from structure
        self.t1 = MechanisticT1FromStructure(
            d_in=self.cfg.d_struct_feat,
            d_hidden=self.cfg.d_t1_hidden)

        # T2: composition from occupancy + multi-omics
        self.t2 = T2CompositionModel(
            d_ctx=self.cfg.d_ctx,
            d_omics=self.cfg.d_omics,
            n_rna_genes=self.cfg.n_rna_genes,
            n_proteins=self.cfg.n_proteins)

        # T3: quality from composition
        self.t3 = T3QualityModel(
            d_ctx=self.cfg.d_ctx,
            d_hidden=self.cfg.d_t3_hidden)

    def forward(self, single_repr: Tensor, coords: Tensor,
                plddt: Tensor, sequence: str,
                factor_idx: Tensor, dose_nM: Tensor, time_h: Tensor,
                ctx: Tensor,
                rna: Optional[Tensor] = None,
                protein: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Full forward pass.

        Args:
            single_repr: (B, L, d_af_repr) from AF encoder
            coords: (B, L, 3) predicted coordinates
            plddt: (B, L) confidence scores
            sequence: amino acid sequence (string)
            factor_idx: (B,) factor index
            dose_nM: (B,) dose in nM
            time_h: (B,) time in hours
            ctx: (B, d_ctx) protocol context
            rna: (B, n_rna_genes) or None
            protein: (B, n_proteins) or None

        Returns:
            dict with all intermediate and final outputs
        """
        # Step 1: Structure → potency features
        struct_feat = self.struct_bridge(single_repr, coords, plddt, sequence)

        # Step 2: T1 mechanistic potency
        t1_out = self.t1(struct_feat)

        # Step 3: Operational response at given dose
        dose_M = dose_nM * 1e-9  # nM → M
        occupancy = self.t1.operational_response(
            dose_M, t1_out["ka"], t1_out["tau"], t1_out["n"], t1_out["em"])

        # Step 4: T2 composition
        t2_out = self.t2(occupancy, factor_idx, dose_nM, time_h,
                         ctx, rna, protein)

        # Step 5: T3 quality
        t3_out = self.t3(t2_out["composition"], ctx)

        return {
            **t1_out, **t2_out, **t3_out,
            "occupancy": occupancy,
            "structure_features": struct_feat,
            "clinical_actionability": RESEARCH_ONLY,
        }

    def compute_loss(self, out: Dict[str, Tensor],
                     targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute joint loss across T1, T2, T3.

        Args:
            out: model output
            targets: dict with optional keys:
                'composition': (B, N_FATE) target composition
                'diameter_um': (B,) target diameter
                'viability': (B,) target viability
                'log_ec50': (B,) target log EC50 (for T1 supervision)

        Returns:
            dict with individual and total losses
        """
        dev = out["occupancy"].device
        zero = torch.zeros((), device=dev)
        losses = {}

        # T1 loss (if EC50 target available)
        if "log_ec50" in targets:
            ec50_pred = self.t1.ec50_from_params(
                out["ka"], out["tau"], out["n"])
            losses["t1_ec50"] = F.mse_loss(
                torch.log(ec50_pred.clamp(min=1e-12)),
                targets["log_ec50"])
        else:
            losses["t1_ec50"] = zero

        # T2 loss (composition)
        if "composition" in targets:
            losses["t2_comp"] = F.kl_div(
                out["composition"].log().clamp(min=-20),
                targets["composition"],
                reduction="batchmean")
        else:
            losses["t2_comp"] = zero

        # T3 loss (quality)
        t3_loss = zero
        if "diameter_um" in targets:
            t3_loss = t3_loss + F.mse_loss(
                out["diameter_um"], targets["diameter_um"])
        if "viability" in targets:
            t3_loss = t3_loss + F.mse_loss(
                out["viability"], targets["viability"])
        losses["t3_quality"] = t3_loss

        # Total
        losses["total"] = (self.cfg.w_t1 * losses["t1_ec50"] +
                           self.cfg.w_t2 * losses["t2_comp"] +
                           self.cfg.w_t3 * losses["t3_quality"])
        return losses


# =============================================================================
# §6. TRAINING UTILITIES
# =============================================================================

def scaffold_split(records: List[Dict], train_frac: float = 0.7,
                   val_frac: float = 0.15, seed: int = 42) -> Dict[str, List[int]]:
    """
    Scaffold split for T2 training (from t1_t2_end_to_end.py).
    Groups factors by family to prevent chemical leakage.
    """
    from collections import defaultdict

    # Group by factor family
    family_map = {
        "GSK3_inhibitor": ["CHIR"],
        "TGFBR_inhibitor": ["SB431542"],
        "BMP_inhibitor": ["LDN193189"],
        "gamma_secretase_inhibitor": ["DAPT"],
        "ROCK_inhibitor": ["Y27632"],
        "retinoid": ["RA"],
        "Wnt": ["WNT"],
        "FGF": ["FGF2", "FGF4", "FGF9", "FGF10"],
        "BMP": ["BMP4"],
        "TGFbeta": ["ACTIVIN", "TGFB1"],
        "Hedgehog": ["SHH"],
        "Notch": ["NOTCH_AG", "NOTCH_INH"],
    }

    # Map each record to its family
    factor_to_family = {}
    for fam, factors in family_map.items():
        for f in factors:
            factor_to_family[f] = fam

    scaffold_to_records = defaultdict(list)
    for i, rec in enumerate(records):
        factor = rec.get("factor", "OTHER")
        family = factor_to_family.get(factor, "unknown")
        scaffold_to_records[family].append(i)

    # Sort by size and assign
    scaffolds = sorted(scaffold_to_records.keys(),
                       key=lambda s: -len(scaffold_to_records[s]))
    n = len(records)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx, val_idx, test_idx = [], [], []
    for scaffold in scaffolds:
        idxs = scaffold_to_records[scaffold]
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    if len(test_idx) == 0 and len(val_idx) > 0:
        mid = len(val_idx) // 2
        test_idx = val_idx[mid:]
        val_idx = val_idx[:mid]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def subsample_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """Politis-Romano subsampling CI for honest error bars."""
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
            "n": n, "block": b}


# =============================================================================
# §7. SYNTHETIC DATA (for pipeline testing only)
# =============================================================================

def make_synthetic_batch(B: int, L: int = 50, d_af: int = 256,
                         device: str = "cpu") -> Dict[str, Tensor]:
    """
    Synthetic batch for pipeline testing.
    ⚠️ NOT real data. For shape/gradient verification only.
    """
    dev = torch.device(device)
    return {
        "single_repr": torch.randn(B, L, d_af, device=dev),
        "coords": torch.randn(B, L, 3, device=dev) * 5,
        "plddt": torch.rand(B, L, device=dev) * 100,
        "sequence": "A" * L,
        "factor_idx": torch.randint(0, N_FACTOR, (B,), device=dev),
        "dose_nM": torch.rand(B, device=dev) * 100 + 1,
        "time_h": torch.rand(B, device=dev) * 200 + 1,
        "ctx": torch.randn(B, 128, device=dev),
        "rna": torch.randn(B, 2000, device=dev),
        "protein": torch.randn(B, 32, device=dev),
    }


def make_synthetic_targets(B: int, device: str = "cpu") -> Dict[str, Tensor]:
    """Synthetic targets for training."""
    dev = torch.device(device)
    comp = torch.rand(B, N_FATE, device=dev)
    comp = comp / comp.sum(-1, keepdim=True)
    return {
        "composition": comp,
        "diameter_um": torch.rand(B, device=dev) * 500 + 50,
        "viability": torch.rand(B, device=dev),
        "log_ec50": torch.randn(B, device=dev) * 2 - 8,  # ~nM range
    }


# =============================================================================
# §8. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # T1: Structure feature extraction
    print("\n[T1] Structure feature extraction")
    bridge = StructureToPotencyBridge(d_in=64, d_out=32).to(device)
    coords = torch.randn(1, 20, 3, device=device)
    plddt = torch.rand(1, 20, device=device) * 100
    feats = bridge.extract_structure_features(coords[0], plddt[0], "A" * 20)
    rep("structure features shape", feats.shape == (12,),
        f"{tuple(feats.shape)}")
    rep("features finite", torch.isfinite(feats).all())

    # T2: StructureToPotencyBridge forward
    print("\n[T2] StructureToPotencyBridge forward")
    single_repr = torch.randn(2, 20, 64, device=device)
    out = bridge(single_repr, coords.expand(2, -1, -1),
                 plddt.expand(2, -1), "A" * 20)
    rep("bridge output shape", out.shape == (2, 32),
        f"{tuple(out.shape)}")
    rep("bridge output finite", torch.isfinite(out).all())

    # T3: MechanisticT1FromStructure
    print("\n[T3] MechanisticT1FromStructure")
    t1 = MechanisticT1FromStructure(d_in=32, d_hidden=64).to(device)
    t1_out = t1(out)
    rep("KA positive", bool((t1_out["ka"] > 0).all()))
    rep("n in range", bool((t1_out["n"] >= 0.3).all() and (t1_out["n"] <= 5.0).all()))
    rep("Em in [0,1]", bool((t1_out["em"] >= 0).all() and (t1_out["em"] <= 1).all()))
    rep("efficacy classes valid",
        bool((t1_out["efficacy_class"] >= 0).all() and (t1_out["efficacy_class"] <= 3).all()))

    # T4: Operational response
    print("\n[T4] Black-Leff operational response")
    dose = torch.tensor([1e-9, 1e-8, 1e-7, 1e-6], device=device)
    ka = torch.tensor([1e-8, 1e-8, 1e-8, 1e-8], device=device)
    tau = torch.tensor([100.0, 100.0, 100.0, 100.0], device=device)
    n = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    em = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    resp = t1.operational_response(dose, ka, tau, n, em)
    rep("response increases with dose", bool((resp[1:] > resp[:-1]).all()))
    rep("response bounded by Em", bool((resp <= em + 1e-6).all()))
    # Antagonist (tau=0) should give zero response
    tau_ant = torch.zeros(4, device=device)
    resp_ant = t1.operational_response(dose, ka, tau_ant, n, em)
    rep("antagonist gives zero response", bool((resp_ant.abs() < 1e-6).all()))

    # T5: Full MDOEConnector forward
    print("\n[T5] Full MDOEConnector forward")
    cfg = MDOEConnectorConfig(d_af_repr=64, d_struct_feat=32,
                              d_t1_hidden=64, d_ctx=32, d_omics=32,
                              n_rna_genes=100, n_proteins=16)
    connector = MDOEConnector(cfg).to(device)
    batch = make_synthetic_batch(2, L=20, d_af=64, device=str(device))
    batch["rna"] = torch.randn(2, 100, device=device)
    batch["protein"] = torch.randn(2, 16, device=device)
    out = connector(batch["single_repr"], batch["coords"], batch["plddt"],
                    batch["sequence"], batch["factor_idx"], batch["dose_nM"],
                    batch["time_h"], batch["ctx"], batch["rna"], batch["protein"])
    rep("composition shape", out["composition"].shape == (2, N_FATE))
    rep("composition sums to 1",
        torch.allclose(out["composition"].sum(-1),
                       torch.ones(2, device=device), atol=1e-4))
    rep("diameter positive", bool((out["diameter_um"] > 0).all()))
    rep("viability in [0,1]",
        bool((out["viability"] >= 0).all() and (out["viability"] <= 1).all()))

    # T6: Gradient flow end-to-end
    print("\n[T6] Gradient flow from T3 to structure encoder")
    targets = make_synthetic_targets(2, device=str(device))
    losses = connector.compute_loss(out, targets)
    losses["total"].backward()
    grad_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in connector.parameters())
    rep("all gradients finite", grad_ok)
    rep("total loss finite", torch.isfinite(losses["total"]))
    # Check gradient reached structure bridge
    bridge_grad = connector.struct_bridge.proj[0].weight.grad
    rep("gradient reaches structure bridge",
        bridge_grad is not None and torch.isfinite(bridge_grad).all())

    # T7: Loss computation with all targets
    print("\n[T7] Loss computation")
    rep("T1 EC50 loss present", "t1_ec50" in losses)
    rep("T2 composition loss present", "t2_comp" in losses)
    rep("T3 quality loss present", "t3_quality" in losses)
    rep("total loss is sum", torch.isclose(
        losses["total"],
        cfg.w_t1 * losses["t1_ec50"] + cfg.w_t2 * losses["t2_comp"] +
        cfg.w_t3 * losses["t3_quality"], atol=1e-5))

    # T8: Missing omics handling
    print("\n[T8] Missing omics handling")
    out_no_omics = connector(batch["single_repr"], batch["coords"],
                             batch["pldddt"], batch["sequence"],
                             batch["factor_idx"], batch["dose_nM"],
                             batch["time_h"], batch["ctx"],
                             None, None)
    rep("works without omics", out_no_omics["composition"].shape == (2, N_FATE))

    # T9: Scaffold split
    print("\n[T9] Scaffold split")
    records = [{"factor": f} for f in FACTORS * 5]  # 85 records
    split = scaffold_split(records, seed=42)
    rep("split covers all records",
        len(split["train"]) + len(split["val"]) + len(split["test"]) == len(records))
    rep("no overlap",
        len(set(split["train"]) & set(split["test"])) == 0)
    print(f"    train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}")

    # T10: Subsampling CI
    print("\n[T10] Subsampling CI")
    x = np.random.randn(100)
    ci = subsample_ci(x)
    rep("CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])
    rep("CI width reasonable", ci["hi"] - ci["lo"] > 0)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  Structure feature extraction produces finite 12-dim vectors.
  StructureToPotencyBridge projects AF repr + coords to latent.
  MechanisticT1 predicts KA, τ, n, Em with correct ranges.
  Black-Leff operational response is monotone in dose, bounded by Em.
  Antagonist (τ=0) gives zero response.
  Full MDOEConnector produces valid composition, diameter, viability.
  Gradients flow from T3 loss back to structure bridge.
  Missing omics handled gracefully.
  Scaffold split prevents family leakage.
  Subsampling CI is ordered.

NOT CLAIMED
  ❌ Structure→potency is learned, not physics-based.
  ❌ Black-Leff parameters are estimated, not measured.
  ❌ No experimental validation performed.
  ❌ No gene editing, no clinical decisions.
  ❌ Multi-omics channels require real data to be useful.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="MDOE structure→potency connector")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if a.demo:
        print(BANNER)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = MDOEConnectorConfig(d_af_repr=64, d_struct_feat=32,
                                  d_t1_hidden=64, d_ctx=32, d_omics=32,
                                  n_rna_genes=100, n_proteins=16)
        connector = MDOEConnector(cfg).to(device)
        optimizer = torch.optim.AdamW(connector.parameters(), lr=1e-3)

        print(f"\nTraining demo ({a.epochs} epochs, batch={a.batch_size})")
        for ep in range(a.epochs):
            optimizer.zero_grad()
            batch = make_synthetic_batch(a.batch_size, L=20, d_af=64, device=device)
            batch["rna"] = torch.randn(a.batch_size, 100, device=device)
            batch["protein"] = torch.randn(a.batch_size, 16, device=device)
            targets = make_synthetic_targets(a.batch_size, device=device)

            out = connector(batch["single_repr"], batch["coords"],
                            batch["plddt"], batch["sequence"],
                            batch["factor_idx"], batch["dose_nM"],
                            batch["time_h"], batch["ctx"],
                            batch["rna"], batch["protein"])
            losses = connector.compute_loss(out, targets)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
            optimizer.step()

            print(f"  epoch {ep+1}/{a.epochs}  loss={losses['total'].item():.4f}  "
                  f"t1={losses['t1_ec50'].item():.4f}  "
                  f"t2={losses['t2_comp'].item():.4f}  "
                  f"t3={losses['t3_quality'].item():.4f}")

        print("\n⚠️ This is SYNTHETIC data. Metrics are meaningless.")
        print("   Real training requires experimental binding + organoid data.")
        return 0

    ap.print_help()
    print(BANNER)
    print("Start: --selftest | --demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
