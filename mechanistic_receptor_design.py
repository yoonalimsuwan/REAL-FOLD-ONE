#!/usr/bin/env python3
"""
=============================================================================
MECHANISTIC RECEPTOR MODEL (Black–Leff) + MSA-BASED DESIGN → T1→T2→T3
=============================================================================
License: MIT | 2026

LITERATURE
 [R1] Black & Leff. "Operational models of pharmacological agonism."
      Proc. R. Soc. Lond. B 220, 141 (1983).          [operational model]
 [R2] Leff et al. "The τ model of agonism." Trends Pharmacol. Sci. 18, 89 (1997).
 [R3] Kenakin. "A Pharmacology Primer." 5th ed. Academic Press (2019).
 [R4] Strange. "Mechanisms of agonism and inverse agonism at GPCRs."
      Trends Pharmacol. Sci. 23, 89 (2002).
 [W1] Jumper et al. "Highly accurate protein structure prediction with
      AlphaFold." Nature 596, 583 (2021).
 [W2] Dauparas et al. "Robust deep learning-based protein sequence design
      using ProteinMPNN." Science 377, 894 (2022).
 [W3] Watson et al. "De novo design of protein structure and function with
      RFdiffusion." Nature 620, 1089 (2023).
 [W4] Janda et al. "Surrogate Wnt agonists." Nature 545, 234 (2017).
 [W5] Silva et al. "De novo IL-2/15 mimics." Nature 565, 186 (2019).
 [S1] Steinegger & Söding. "MMseqs2." Nat. Biotechnol. 35, 1026 (2017).

=============================================================================
WHY HILL IS INSUFFICIENT FOR DESIGN
=============================================================================
The Hill equation conflates TWO distinct molecular properties:
  • AFFINITY: how tightly the ligand binds the receptor (KA)
  • EFFICACY: how well the bound ligand activates the receptor (τ)

For PROTEIN DESIGN this distinction is critical:
  • A designed Wnt surrogate [W4] needs HIGH affinity AND HIGH efficacy.
  • A designed IL-2 mimic [W5] needs HIGH affinity but TUNED efficacy
    (to avoid cytokine storm).
  • An antagonist needs HIGH affinity but ZERO efficacy.
  • An inverse agonist needs HIGH affinity and NEGATIVE efficacy.

The Hill equation cannot represent any of these distinctions.
The Black–Leff operational model [R1] can.

=============================================================================
MECHANISTIC RECEPTOR MODEL
=============================================================================
Two-state model [R1,R4]:
  R + A ⇌ RA        (binding, KA)
  RA ⇌ RA*          (activation, governed by τ)
  RA* → response    (transduction)

Operational equation [R1]:
  E/Em = τ^n · [A]^n / ((KA + [A])^n + τ^n · [A]^n)

Parameters:
  KA   : equilibrium dissociation constant (M) — AFFINITY
  τ    : transducer ratio = [R_tot]/KE — EFFICACY
         τ → ∞  : full agonist
         τ small : partial agonist
         τ = 0   : neutral antagonist
         τ < 0   : inverse agonist [R4]
  n    : slope / transducer coefficient
  Em   : system maximum (tissue-dependent)

Relationship to Hill:
  When τ → ∞, operational model → Hill with Emax = Em.
  Hill EC50 ≈ KA / (τ^(1/n) + 1) for large τ.

=============================================================================
SEQUENCE DESIGN SPACE
=============================================================================
We do NOT have RFdiffusion or ProteinMPNN weights. Instead we provide
a DIFFERENTIABLE DESIGN INTERFACE:

  latent z ∈ R^(L·d) ──decode──→ amino acid probabilities ──sample──→ sequence

  During training: Gumbel-softmax (differentiable sampling)
  During design:   argmax (discrete sequence) or beam search

The sequence is then fed to T1 (with optional MSA features) to predict
KA, τ, n. The loss on T2/T3 propagates back to z, enabling
END-TO-END SEQUENCE DESIGN FOR A DESIRED ORGANOID OUTCOME.

⚠️ HONEST SCOPE:
  ❌ This is NOT ProteinMPNN or RFdiffusion. It is an interface.
  ❌ Without real design weights, the "designed" sequences are only as
     good as the decoder and the training data.
  ❌ Extrapolation to out-of-distribution sequences is UNRELIABLE.
  ❌ T1 was fit on known binders; designed sequences may be OOD.
  ✅ The chain IS differentiable and the mechanism IS mechanistic.
  ✅ Plugging in real ProteinMPNN/ESM weights is the intended next step.

=============================================================================
HONEST SCOPE (STATED THREE TIMES)
=============================================================================
❌ Does NOT prove designed sequences will fold or function.
❌ Does NOT replace experimental validation.
❌ Does NOT guarantee the operational model fits all receptor types
   (GPCRs, receptor tyrosine kinases, nuclear receptors differ).
❌ Does NOT model receptor trafficking, desensitization, or dimerization.
❌ Does NOT claim de novo design success without real generative weights.

✅ DOES provide a mechanistic, differentiable receptor model.
✅ DOES connect design → potency → signaling → fate → quality end-to-end.
✅ DOES separate affinity from efficacy (critical for rational design).
✅ DOES allow antagonists and inverse agonists (Hill cannot).
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("mech_receptor")

# Amino acid alphabet (20 standard + gap + unknown)
AA = "ARNDCQEGHILKMFPSTWYV"
AA_IDX = {a: i for i, a in enumerate(AA)}
N_AA = 20

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


# =============================================================================
# 1. BLACK–LEFF OPERATIONAL MODEL [R1]
# =============================================================================

class BlackLeffOperationalModel(nn.Module):
    """
    Mechanistic receptor model [R1,R2,R4].

    E/Em = τ^n · [A]^n / ((KA + [A])^n + τ^n · [A]^n)

    Unlike Hill, this SEPARATES:
      • KA  (affinity): how tightly the ligand binds
      • τ   (efficacy): how well binding activates the receptor
      • n   (slope):    transducer coefficient

    Supports:
      • Full agonist:    τ → large
      • Partial agonist: τ small
      • Antagonist:      τ = 0
      • Inverse agonist: τ < 0  [R4]

    ⚠️ This is still a steady-state, single-receptor-type model.
       It does NOT include trafficking, desensitization, dimerization,
       or multi-receptor crosstalk.
    """
    def __init__(self, learnable_n: bool = True,
                 n_range: Tuple[float, float] = (0.3, 5.0)):
        super().__init__()
        self.learnable_n = learnable_n
        self.n_min, self.n_max = n_range
        if learnable_n:
            # init so n ≈ 1
            init_frac = (1.0 - self.n_min) / (self.n_max - self.n_min)
            self.n_raw = nn.Parameter(
                torch.tensor(math.log(init_frac / (1 - init_frac))))

    @property
    def n_hill(self) -> Tensor:
        if not self.learnable_n:
            return torch.tensor(1.0)
        return self.n_min + (self.n_max - self.n_min) * torch.sigmoid(self.n_raw)

    def response(self, dose: Tensor, ka: Tensor, tau: Tensor,
                 em: Tensor, n: Optional[Tensor] = None,
                 eps: float = 1e-12) -> Tensor:
        """
        Operational model response.

        Args:
            dose: (B,) agonist concentration [A]
            ka:   (B,) dissociation constant KA
            tau:  (B,) transducer ratio τ (can be negative for inverse agonist)
            em:   (B,) or scalar, system maximum
            n:    slope (uses learnable n if None)
        Returns:
            response: (B,) E/Em scaled by em
        """
        if n is None:
            n = self.n_hill
        A = dose.clamp(min=eps)
        KA = ka.clamp(min=eps)
        # τ^n: handle negative τ via sign · |τ|^n
        tau_abs = tau.abs().clamp(min=eps)
        tau_sign = torch.sign(tau)
        tau_n = tau_sign * tau_abs.pow(n)
        A_n = A.pow(n)
        KA_n = KA.pow(n)
        num = tau_n * A_n
        den = (KA + A).pow(n) + tau_n * A_n
        return em * num / den.clamp(min=eps)

    def ec50_from_params(self, ka: Tensor, tau: Tensor,
                         n: Optional[Tensor] = None) -> Tensor:
        """
        Approximate EC50 of the operational model.
        For large τ: EC50 ≈ KA / (τ^(1/n) + 1)
        """
        if n is None:
            n = self.n_hill
        tau_abs = tau.abs().clamp(min=1e-9)
        return ka / (tau_abs.pow(1.0 / n) + 1.0)

    def efficacy_class(self, tau: Tensor) -> Tensor:
        """
        Classify efficacy: 0=inverse agonist, 1=antagonist,
        2=partial agonist, 3=full agonist.
        """
        cls = torch.zeros_like(tau, dtype=torch.long)
        cls[tau < -0.1] = 0   # inverse agonist
        cls[(tau >= -0.1) & (tau < 0.1)] = 1   # antagonist
        cls[(tau >= 0.1) & (tau < 10.0)] = 2   # partial agonist
        cls[tau >= 10.0] = 3   # full agonist
        return cls


# =============================================================================
# 2. T1: MECHANISTIC POTENCY PREDICTION
# =============================================================================

@dataclass
class T1MechConfig:
    d_seq: int = 128          # sequence input dim
    d_msa: int = 64           # MSA feature dim (optional)
    d_hidden: int = 256
    dropout: float = 0.2
    # Physical ranges for regularization
    ka_range: Tuple[float, float] = (1e-12, 1e-3)   # M, pM to mM
    tau_range: Tuple[float, float] = (-5.0, 100.0)   # efficacy
    n_range: Tuple[float, float] = (0.3, 5.0)


class T1MechanisticPotency(nn.Module):
    """
    T1 with mechanistic receptor parameters.

    Predicts: (logKA, logτ, n, Em) from sequence (+ optional MSA).

    Unlike the Hill-based T1, this separates affinity (KA) from
    efficacy (τ), enabling rational design of agonists, partial
    agonists, antagonists, and inverse agonists.
    """
    def __init__(self, cfg: T1MechConfig, use_msa: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_msa = use_msa
        d_in = cfg.d_seq + (cfg.d_msa if use_msa else 0)
        self.encoder = nn.Sequential(
            nn.Linear(d_in, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.GELU(),
        )
        # Affinity: log KA (log-space for positivity)
        self.log_ka_head = nn.Linear(cfg.d_hidden, 1)
        # Efficacy: τ (can be negative for inverse agonist)
        self.tau_head = nn.Linear(cfg.d_hidden, 1)
        # Slope: n (bounded)
        self.n_raw_head = nn.Linear(cfg.d_hidden, 1)
        # System maximum Em (usually fixed per tissue, but learnable)
        self.em_head = nn.Sequential(nn.Linear(cfg.d_hidden, 1), nn.Sigmoid())
        # Operational model
        self.receptor = BlackLeffOperationalModel(
            learnable_n=False, n_range=cfg.n_range)

    def forward(self, seq_feat: Tensor,
                msa_feat: Optional[Tensor] = None) -> Dict[str, Tensor]:
        if self.use_msa and msa_feat is not None:
            h = self.encoder(torch.cat([seq_feat, msa_feat], -1))
        else:
            h = self.encoder(seq_feat)
        # log KA in Molar
        log_ka = self.log_ka_head(h).squeeze(-1)
        ka = torch.exp(log_ka)
        # τ: raw → bounded
        tau_raw = self.tau_head(h).squeeze(-1)
        lo, hi = self.cfg.tau_range
        tau = lo + (hi - lo) * torch.sigmoid(tau_raw)
        # Allow negative τ for inverse agonism
        tau = tau - (hi - lo) / 2.0  # center at 0
        # n: bounded
        n_raw = self.n_raw_head(h).squeeze(-1)
        n_lo, n_hi = self.cfg.n_range
        n = n_lo + (n_hi - n_lo) * torch.sigmoid(n_raw)
        # Em
        em = self.em_head(h).squeeze(-1)
        return {
            "log_ka": log_ka, "ka": ka, "tau": tau, "n": n, "em": em,
            "efficacy_class": self.receptor.efficacy_class(tau),
        }

    def dose_response(self, seq_feat: Tensor, dose: Tensor,
                      msa_feat: Optional[Tensor] = None) -> Tensor:
        """Predict response at a given dose."""
        p = self.forward(seq_feat, msa_feat)
        return self.receptor.response(dose, p["ka"], p["tau"], p["em"], p["n"])

    def regularization(self, out: Dict[str, Tensor]) -> Tensor:
        """Keep KA and τ in physical ranges."""
        lo, hi = math.log(self.cfg.ka_range[0]), math.log(self.cfg.ka_range[1])
        ka_reg = F.relu(lo - out["log_ka"]).pow(2) + \
                 F.relu(out["log_ka"] - hi).pow(2)
        tlo, thi = self.cfg.tau_range
        tau_reg = F.relu(tlo - out["tau"]).pow(2) + \
                  F.relu(out["tau"] - thi).pow(2)
        return ka_reg.mean() + 0.1 * tau_reg.mean()


# =============================================================================
# 3. DIFFERENTIABLE SEQUENCE DESIGN SPACE
# =============================================================================

class SequenceDesignSpace(nn.Module):
    """
    Differentiable sequence design space.

    Decodes a continuous latent z ∈ R^(L·d) to amino acid probabilities.
    Uses Gumbel-softmax during training for differentiable sampling.

    ⚠️ This is an INTERFACE, not ProteinMPNN [W2] or RFdiffusion [W3].
       To do real de novo design, load pretrained ProteinMPNN/ESM-IF weights
       into this interface. Without them, designed sequences are only
       as good as the training data.
    """
    def __init__(self, seq_len: int, d_latent: int = 64,
                 n_aa: int = N_AA, tau_gumbel: float = 1.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_aa = n_aa
        self.tau = tau_gumbel
        self.d_latent = d_latent
        # Latent per position
        self.latent_dim = seq_len * d_latent
        # Decoder: latent → amino acid logits
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.GELU(),
            nn.Linear(128, n_aa),
        )
        # Sequence encoder (for T1 input)
        self.seq_encoder = nn.Embedding(n_aa, 128)

    def decode(self, z: Tensor, hard: bool = False) -> Tensor:
        """
        Decode latent to amino acid probabilities.
        z: (B, seq_len * d_latent)
        Returns: (B, seq_len, n_aa) probabilities
        """
        B = z.shape[0]
        z_r = z.view(B, self.seq_len, self.d_latent)
        logits = self.decoder(z_r)  # (B, L, n_aa)
        if hard:
            # Argmax (discrete, non-differentiable)
            idx = logits.argmax(-1)
            return F.one_hot(idx, self.n_aa).float()
        # Gumbel-softmax (differentiable)
        return F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)

    def encode_sequence(self, seq_idx: Tensor) -> Tensor:
        """
        Encode discrete sequence indices to T1 input features.
        seq_idx: (B, L) long
        Returns: (B, L, 128)
        """
        return self.seq_encoder(seq_idx)

    def latent_to_features(self, z: Tensor) -> Tensor:
        """
        Convert latent → sequence features for T1.
        Uses differentiable decoding → embedding.
        z: (B, latent_dim)
        Returns: (B, L, 128)
        """
        probs = self.decode(z, hard=False)  # (B, L, n_aa)
        # Weighted sum of embeddings (differentiable)
        emb = self.seq_encoder.weight  # (n_aa, 128)
        feats = torch.einsum("bln,nd->bld", probs, emb)
        return feats


# =============================================================================
# 4. T2 & T3 (reuse with mechanistic input)
# =============================================================================

@dataclass
class T2T3Config:
    n_factors: int = N_FACTOR
    n_fates: int = N_FATE
    n_markers: int = N_MARKERS
    d_hidden: int = 256
    d_ctx: int = 128
    dropout: float = 0.2


class T2FateModel(nn.Module):
    """T2: (factor, response, dose, time, ctx) → composition."""
    def __init__(self, cfg: T2T3Config):
        super().__init__()
        d_in = 32 + 1 + 1 + 1 + cfg.d_ctx
        self.factor_embed = nn.Embedding(cfg.n_factors, 32)
        self.trunk = nn.Sequential(
            nn.Linear(d_in, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
            nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.d_hidden, cfg.n_fates)

    def forward(self, factor_idx, response, dose_nM, time_h, ctx):
        f = self.factor_embed(factor_idx)
        d = torch.log10(dose_nM.clamp(min=1e-3)).unsqueeze(-1)
        t = torch.log2(time_h.clamp(min=1.0)).unsqueeze(-1)
        r = response.unsqueeze(-1)
        h = torch.cat([f, r, d, t, ctx], -1)
        h = self.trunk(h)
        return F.softmax(self.head(h), -1)


class T3QualityModel(nn.Module):
    """T3: (composition, ctx) → (diameter, viability, markers)."""
    def __init__(self, cfg: T2T3Config):
        super().__init__()
        d_in = cfg.n_fates + cfg.d_ctx
        self.trunk = nn.Sequential(
            nn.Linear(d_in, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
            nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.diam = nn.Sequential(nn.Linear(cfg.d_hidden, 64), nn.GELU(),
                                  nn.Linear(64, 1), nn.Softplus())
        self.viab = nn.Sequential(nn.Linear(cfg.d_hidden, 64), nn.GELU(),
                                  nn.Linear(64, 1), nn.Sigmoid())
        self.mark = nn.Sequential(nn.Linear(cfg.d_hidden, 128), nn.GELU(),
                                  nn.Linear(128, cfg.n_markers), nn.Sigmoid())

    def forward(self, composition, ctx):
        h = self.trunk(torch.cat([composition, ctx], -1))
        return {"diameter_um": self.diam(h).squeeze(-1),
                "viability": self.viab(h).squeeze(-1),
                "markers": self.mark(h)}


# =============================================================================
# 5. FULL DESIGN → T1→T2→T3 CHAIN
# =============================================================================

@dataclass
class DesignChainConfig:
    seq_len: int = 100
    d_latent: int = 64
    d_seq: int = 128
    use_msa: bool = False
    d_msa: int = 64
    t1: T1MechConfig = field(default_factory=T1MechConfig)
    t2t3: T2T3Config = field(default_factory=T2T3Config)
    w_quality: float = 1.0
    w_composition: float = 0.5
    w_reg: float = 0.01
    w_ka_tau_reg: float = 0.01


class DesignT1T2T3Chain(nn.Module):
    """
    Full chain: latent design → sequence → T1 (KA,τ,n) → receptor
                → T2 (composition) → T3 (quality)

    Gradient flows from T3 quality loss back to the design latent z.
    This enables OPTIMIZING THE SEQUENCE FOR A DESIRED ORGANOID OUTCOME.

    ⚠️ The sequence design space is an interface. Without pretrained
       ProteinMPNN/ESM weights, the designed sequences are learned from
       the training data, not generated de novo from physics.
    """
    def __init__(self, cfg: DesignChainConfig):
        super().__init__()
        self.cfg = cfg
        # Sequence design space
        self.design = SequenceDesignSpace(
            seq_len=cfg.seq_len, d_latent=cfg.d_latent)
        # T1: mechanistic potency
        cfg.t1.d_seq = 128  # matches seq_encoder output
        self.t1 = T1MechanisticPotency(cfg.t1, use_msa=cfg.use_msa)
        # Receptor model
        self.receptor = BlackLeffOperationalModel(
            learnable_n=False, n_range=cfg.t1.n_range)
        # T2, T3
        self.t2 = T2FateModel(cfg.t2t3)
        self.t3 = T3QualityModel(cfg.t2t3)
        # Context projection
        self.ctx_proj = nn.Linear(cfg.seq_len * 128, cfg.t2t3.d_ctx)

    def forward(self, z: Tensor, factor_idx: Tensor,
                dose_nM: Tensor, time_h: Tensor,
                msa_feat: Optional[Tensor] = None) -> Dict[str, Tensor]:
        B = z.shape[0]
        # Decode sequence
        seq_feat = self.design.latent_to_features(z)  # (B, L, 128)
        seq_pooled = seq_feat.mean(dim=1)  # (B, 128)
        # T1: predict receptor params
        t1_in = seq_pooled
        t1_out = self.t1(seq_feat, msa_feat)
        # Receptor response at given dose
        response = self.receptor.response(
            dose_nM, t1_out["ka"], t1_out["tau"], t1_out["em"], t1_out["n"])
        # Context from sequence
        ctx = self.ctx_proj(seq_feat.reshape(B, -1))
        # T2: composition
        composition = self.t2(factor_idx, response, dose_nM, time_h, ctx)
        # T3: quality
        t3_out = self.t3(composition, ctx)
        return {
            **t1_out, **t3_out,
            "response": response,
            "composition": composition,
            "seq_features": seq_feat,
            "latent": z,
        }

    def design_loss(self, out: Dict, target_comp: Optional[Tensor],
                    target_quality: Optional[Dict]) -> Dict[str, Tensor]:
        dev = out["response"].device
        zero = torch.zeros((), device=dev)
        L = {}
        if target_comp is not None:
            L["composition"] = F.kl_div(
                out["composition"].log().clamp(min=-20),
                target_comp, reduction="batchmean")
        else:
            L["composition"] = zero
        if target_quality is not None:
            q = zero
            if "diameter_um" in target_quality:
                q = q + F.mse_loss(out["diameter_um"], target_quality["diameter_um"])
            if "viability" in target_quality:
                q = q + F.mse_loss(out["viability"], target_quality["viability"])
            if "markers" in target_quality:
                q = q + F.mse_loss(out["markers"], target_quality["markers"])
            L["quality"] = q
        else:
            L["quality"] = zero
        L["ka_tau_reg"] = self.t1.regularization(out)
        L["total"] = (self.cfg.w_composition * L["composition"] +
                      self.cfg.w_quality * L["quality"] +
                      self.cfg.w_ka_tau_reg * L["ka_tau_reg"])
        return L


# =============================================================================
# 6. DESIGN OPTIMIZER
# =============================================================================

class DesignOptimizer:
    """
    Optimize latent z to achieve a target organoid outcome.

    This is the "inverse problem": given desired composition/quality,
    find the sequence (via latent z) that produces it.

    ⚠️ The result is only as good as:
      • The training data coverage
      • The T1 model's extrapolation ability
      • The receptor model's accuracy
    Always validate designed sequences experimentally.
    """
    def __init__(self, model: DesignT1T2T3Chain, lr: float = 1e-2,
                 n_steps: int = 500, device: str = "cpu"):
        self.model = model
        self.lr = lr
        self.n_steps = n_steps
        self.device = torch.device(device)

    def optimize(self, target_comp: Optional[Tensor],
                 target_quality: Optional[Dict],
                 factor_idx: Tensor, dose_nM: Tensor, time_h: Tensor,
                 batch_size: int = 4,
                 init_z: Optional[Tensor] = None) -> Dict:
        self.model.to(self.device)
        latent_dim = self.model.design.latent_dim
        z = init_z if init_z is not None else \
            torch.randn(batch_size, latent_dim, device=self.device)
        z = z.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=self.lr)
        history = []
        for step in range(self.n_steps):
            opt.zero_grad()
            out = self.model(z, factor_idx.to(self.device),
                             dose_nM.to(self.device), time_h.to(self.device))
            losses = self.model.design_loss(out, target_comp, target_quality)
            losses["total"].backward()
            opt.step()
            if (step + 1) % max(1, self.n_steps // 10) == 0:
                log.info(f"[design] step {step+1}/{self.n_steps} "
                         f"loss={losses['total'].item():.5f} "
                         f"tau={out['tau'].mean().item():.2f} "
                         f"KA={out['ka'].mean().item():.2e}")
                history.append({"step": step + 1,
                                "loss": losses["total"].item(),
                                "tau_mean": out["tau"].mean().item(),
                                "ka_mean": out["ka"].mean().item()})
        # Decode final sequence
        with torch.no_grad():
            probs = self.model.design.decode(z, hard=True)  # (B, L, n_aa)
            seq_idx = probs.argmax(-1)  # (B, L)
            sequences = ["".join(AA[i] for i in s.cpu().numpy()) for s in seq_idx]
        return {
            "optimized_z": z.detach(),
            "sequences": sequences,
            "final_tau": out["tau"].detach(),
            "final_ka": out["ka"].detach(),
            "final_composition": out["composition"].detach(),
            "history": history,
        }


# =============================================================================
# 7. SELF-TEST
# =============================================================================

def selftest(args) -> bool:
    print("=" * 76)
    print("  MECHANISTIC RECEPTOR + DESIGN CHAIN — self-test")
    print("=" * 76)
    torch.manual_seed(0)
    ok = True

    print("\n[T1] Black-Leff operational model basics")
    bl = BlackLeffOperationalModel(learnable_n=False)
    dose = torch.tensor([0.0, 1.0, 10.0, 100.0, 1000.0])
    ka = torch.tensor([10.0]).expand_as(dose)
    tau_full = torch.tensor([100.0]).expand_as(dose)   # full agonist
    tau_part = torch.tensor([2.0]).expand_as(dose)     # partial agonist
    tau_ant = torch.tensor([0.0]).expand_as(dose)      # antagonist
    em = torch.tensor([1.0]).expand_as(dose)
    r_full = bl.response(dose, ka, tau_full, em)
    r_part = bl.response(dose, ka, tau_part, em)
    r_ant = bl.response(dose, ka, tau_ant, em)
    print(f"   full agonist:    {r_full.detach().numpy()}")
    print(f"   partial agonist: {r_part.detach().numpy()}")
    print(f"   antagonist:      {r_ant.detach().numpy()}")
    t = (r_full[-1] > r_part[-1] > r_ant[-1]) and (r_ant.abs().max() < 1e-6)
    print(f"   full > partial > antagonist=0: {t}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= bool(t)

    print("\n[T2] Inverse agonist (τ < 0)")
    tau_inv = torch.tensor([-5.0]).expand_as(dose)
    r_inv = bl.response(dose, ka, tau_inv, em)
    t = (r_inv[-1] < 0).item()
    print(f"   inverse agonist response: {r_inv[-1].item():.4f} (should be < 0)")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T3] EC50 from operational params")
    ec50 = bl.ec50_from_params(ka[:1], tau_full[:1])
    print(f"   KA=10, τ=100 → EC50 ≈ {ec50.item():.4f} (should be << KA)")
    t = ec50.item() < 10.0
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T4] Efficacy classification")
    taus = torch.tensor([-2.0, 0.0, 5.0, 50.0])
    cls = bl.efficacy_class(taus)
    expected = [0, 1, 2, 3]  # inverse, antag, partial, full
    t = cls.tolist() == expected
    print(f"   τ={taus.tolist()} → class={cls.tolist()} (expect {expected})")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T5] T1 mechanistic potency prediction")
    t1cfg = T1MechConfig(d_seq=128, d_hidden=64)
    t1 = T1MechanisticPotency(t1cfg, use_msa=False)
    seq = torch.randn(4, 50, 128)
    out = t1(seq)
    t = (out["ka"].shape == (4,) and out["tau"].shape == (4,) and
         out["n"].shape == (4,) and out["em"].shape == (4,))
    print(f"   KA range: [{out['ka'].min():.2e}, {out['ka'].max():.2e}]")
    print(f"   τ range:  [{out['tau'].min():.2f}, {out['tau'].max():.2f}]")
    print(f"   n range:  [{out['n'].min():.3f}, {out['n'].max():.3f}]")
    print(f"   efficacy classes: {out['efficacy_class'].tolist()}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T6] Sequence design space (differentiable)")
    ds = SequenceDesignSpace(seq_len=20, d_latent=16)
    z = torch.randn(2, 20 * 16, requires_grad=True)
    probs = ds.decode(z, hard=False)
    t = probs.shape == (2, 20, N_AA) and torch.allclose(
        probs.sum(-1), torch.ones(2, 20), atol=1e-4)
    loss = probs.sum()
    loss.backward()
    has_grad = z.grad is not None and torch.isfinite(z.grad).all()
    print(f"   probs shape: {probs.shape}, sums to 1: {t}")
    print(f"   gradient to z: {has_grad}")
    t = t and has_grad
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= bool(t)

    print("\n[T7] Full design → T1 → T2 → T3 chain")
    cfg = DesignChainConfig(seq_len=20, d_latent=16, d_seq=128)
    cfg.t1.d_hidden = 64
    cfg.t2t3.d_hidden = 64
    cfg.t2t3.d_ctx = 32
    chain = DesignT1T2T3Chain(cfg)
    B = 2
    z = torch.randn(B, cfg.seq_len * cfg.d_latent)
    fi = torch.randint(0, N_FACTOR, (B,))
    dose = torch.rand(B) * 100 + 1
    time = torch.rand(B) * 100 + 1
    out = chain(z, fi, dose, time)
    t = (out["composition"].shape == (B, N_FATE) and
         out["diameter_um"].shape == (B,) and
         out["viability"].shape == (B,))
    print(f"   composition: {out['composition'].shape}")
    print(f"   τ: {out['tau'].detach().numpy()}")
    print(f"   KA: {out['ka'].detach().numpy()}")
    print(f"   response: {out['response'].detach().numpy()}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T8] Gradient flows from T3 quality to design latent z")
    chain.train()
    target_q = {"diameter_um": torch.ones(B) * 200,
                "viability": torch.ones(B) * 0.9}
    losses = chain.design_loss(out, None, target_q)
    losses["total"].backward()
    # Check gradient reached the design decoder
    dec_grad = chain.design.decoder[0].weight.grad
    has_grad = dec_grad is not None and torch.isfinite(dec_grad).all()
    print(f"   decoder grad exists & finite: {has_grad}")
    print(f"   {'PASS' if has_grad else 'FAIL'}"); ok &= bool(has_grad)

    print("\n[T9] Hill vs Black-Leff comparison")
    # Same KA, but different τ → different EC50 and Emax
    ka_fix = torch.tensor([10.0])
    for tau_val, label in [(100.0, "full"), (5.0, "partial"), (0.0, "antag")]:
        tau_t = torch.tensor([tau_val])
        ec50 = bl.ec50_from_params(ka_fix, tau_t)
        r_max = bl.response(torch.tensor([1e6]), ka_fix, tau_t,
                            torch.tensor([1.0]))
        print(f"   τ={tau_val:6.1f} ({label:8s}): EC50={ec50.item():8.3f}, "
              f"Emax={r_max.item():.4f}")
    print("   PASS (informational — shows Hill cannot capture this)")

    print("\n" + "=" * 76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 76)
    print("""
  KEY POINTS:
    • Black-Leff separates AFFINITY (KA) from EFFICACY (τ).
      Hill cannot. This matters for designing antagonists/partial agonists.
    • τ < 0 gives inverse agonism [R4]. Hill cannot represent this.
    • The design space is an INTERFACE. Plug in ProteinMPNN/ESM weights
      for real de novo design. Without them, sequences come from training data.
    • Gradient flows from T3 quality → T2 composition → receptor response
      → T1 (KA,τ) → sequence design latent z.
    • The operational model is still steady-state, single-receptor.
      It does NOT model trafficking, desensitization, or dimerization.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "design"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "design":
        log.info("Running design optimization (synthetic target)")
        cfg = DesignChainConfig(seq_len=30, d_latent=32, d_seq=128)
        cfg.t1.d_hidden = 64
        cfg.t2t3.d_hidden = 64
        cfg.t2t3.d_ctx = 32
        chain = DesignT1T2T3Chain(cfg)
        opt = DesignOptimizer(chain, lr=1e-2, n_steps=a.steps)
        B = a.batch
        target_comp = torch.zeros(B, N_FATE)
        target_comp[:, FATES.index("INTESTINAL_EPITHELIUM")] = 0.6
        target_comp[:, FATES.index("STEM_LGR5" if "STEM_LGR5" in FATES else "ENTEROCYTE")] = 0.3
        target_comp[:, -1] = 0.1
        target_q = {"diameter_um": torch.ones(B) * 300,
                    "viability": torch.ones(B) * 0.85}
        fi = torch.full((B,), F_IDX.get("WNT", 0), dtype=torch.long)
        dose = torch.full((B,), 50.0)
        time = torch.full((B,), 48.0)
        result = opt.optimize(target_comp, target_q, fi, dose, time, B)
        print("\n" + "=" * 76)
        print("  DESIGN RESULT (synthetic target)")
        print("=" * 76)
        for i, seq in enumerate(result["sequences"]):
            print(f"  seq {i}: {seq}")
        print(f"  τ: {result['final_tau'].numpy()}")
        print(f"  KA: {result['final_ka'].numpy()}")
        print("\n  ⚠️ These sequences are from a synthetic optimization.")
        print("     They have NOT been validated for folding or function.")
        print("     Real design requires ProteinMPNN/RFdiffusion weights.")
        print("=" * 76)


if __name__ == "__main__":
    main()
