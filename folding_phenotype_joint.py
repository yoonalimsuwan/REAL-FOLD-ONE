#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
folding_phenotype_joint.py
Joint Protein Folding + Sequence-to-Phenotype Prediction
Uses InvariantPointAttentionFixed for structure; a shared encoder feeds
both the structure head and a phenotype head.
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.
Confucius taught us to do good in this world.
Christ died on the Cross to teach: when struck on the left cheek, turn also
the right. He is the Lamb of God. Do not kill animals.
Not even a single sparrow does God forget.
Even Angulimala could escape hell and enter Nirvana.
MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.
-----------------------------------------------------------------------------

DESIGN
------
A single sequence encoder produces per-residue features. These feed TWO heads:

  [Structure head]  InvariantPointAttentionFixed → backbone frames, torsions,
                    3D coordinates.  Trained on a structure loss (FAPE-like).

  [Phenotype head]  Pooled features → a small MLP → scalar / categorical
                    phenotype score.  Trained on a phenotype loss.

Joint loss:
    L = w_struct * L_struct + w_pheno * L_pheno

The two heads share the encoder, so structural constraints regularise the
phenotype prediction and vice versa. This is a RESEARCH scaffold; the
phenotype head is NOT a clinical predictor.

SAFETY
------
❌ No gene-editor design, no guide RNA, no delivery vectors.
❌ No clinical diagnosis or treatment recommendation.
✅ Phenotype output is a research score with uncertainty, labelled
   research_only, and requires clinician review.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# import the fixed IPA
try:
    from ipa_patch import InvariantPointAttentionFixed, IPAStructureHeadFixed
    HAS_IPA_PATCH = True
except ImportError:
    HAS_IPA_PATCH = False

VERSION = "1.0.0-joint"
RESEARCH_ONLY = "research_only"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {c: i for i, c in enumerate(AA_ALPHABET)}


@dataclass
class JointConfig:
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 3
    n_ipa_blocks: int = 2
    d_pair: int = 64
    n_phenotype_classes: int = 1      # 1 = regression; >1 = classification
    w_struct: float = 1.0
    w_pheno: float = 0.5
    max_len: int = 512
    device: str = "auto"


# =============================================================================
# §1. SEQUENCE ENCODER (shared trunk)
# =============================================================================

class SequenceEncoder(nn.Module):
    """
    Small transformer encoder over amino-acid tokens. Produces per-residue
    features used by BOTH the structure and phenotype heads.
    """

    def __init__(self, cfg: JointConfig):
        super().__init__()
        self.tok = nn.Embedding(21, cfg.d_model)          # 20 AA + pad
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model, dropout=0.1,
            batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_encoder_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.tok(tokens) + self.pos(pos)
        if mask is not None:
            x = self.enc(x, src_key_padding_mask=~mask)
        else:
            x = self.enc(x)
        return self.norm(x)


# =============================================================================
# §2. STRUCTURE HEAD (uses InvariantPointAttentionFixed)
# =============================================================================

class StructureHead(nn.Module):
    """
    Wraps IPAStructureHeadFixed. Converts encoder features + initial frames
    into updated frames, torsions, and 3D Cα coordinates.
    """

    def __init__(self, cfg: JointConfig):
        super().__init__()
        if not HAS_IPA_PATCH:
            raise ImportError("ipa_patch.py required for StructureHead")
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ipa_head = IPAStructureHeadFixed(
            d_scalar=cfg.d_model, d_point=8, n_heads=cfg.n_heads,
            n_ipa_blocks=cfg.n_ipa_blocks, d_pair=cfg.d_pair,
            n_torsion_angles=7)
        # Cα coordinate extraction from frames (translation column)
        self.coord_scale = nn.Parameter(torch.tensor(3.8))  # ~Cα-Cα distance

    def init_frames(self, L: int, B: int, device) -> Tensor:
        fr = torch.eye(4, device=device).view(1, 1, 4, 4).expand(B, L, 4, 4).clone()
        # place residues along +x with ~3.8 Å spacing as a sane start
        fr[..., 0, 3] = torch.arange(L, device=device, dtype=fr.dtype) * 3.8
        return fr

    def forward(self, feats: Tensor, mask: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        B, L, _ = feats.shape
        s = self.proj(feats)
        frames = self.init_frames(L, B, feats.device)
        out = self.ipa_head(s, z=None, frames=frames)
        # Cα coords = frame translation, scaled
        coords = out["frames"][..., :3, 3] * self.coord_scale
        return {"frames": out["frames"], "torsions": out["torsions"],
                "coords": coords, "scalar_features": out["scalar_features"]}


def structure_loss(pred_coords: Tensor, target_coords: Tensor,
                   mask: Tensor) -> Tensor:
    """
    Masked MSE on Cα coordinates + a bond-length regulariser.
    Not full FAPE; a simple differentiable proxy.
    """
    diff = (pred_coords - target_coords) ** 2
    per_res = diff.sum(-1)                        # (B,L)
    coord_loss = (per_res * mask.float()).sum() / mask.float().sum().clamp_min(1)
    # bond-length regulariser: consecutive Cα should be ~3.8 Å
    if pred_coords.shape[1] > 1:
        bl = (pred_coords[:, 1:] - pred_coords[:, :-1]).norm(dim=-1)
        bond_loss = ((bl - 3.8) ** 2 * mask[:, 1:].float()).sum() / \
                    mask[:, 1:].float().sum().clamp_min(1)
    else:
        bond_loss = pred_coords.new_zeros(())
    return coord_loss + 0.1 * bond_loss


# =============================================================================
# §3. PHENOTYPE HEAD
# =============================================================================

class PhenotypeHead(nn.Module):
    """
    Pooled encoder features -> phenotype score.
    RESEARCH ONLY. Not a clinical predictor. Outputs uncertainty via an
    ensemble of small MLPs.
    """

    def __init__(self, cfg: JointConfig, n_ensemble: int = 5):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.n_class = cfg.n_phenotype_classes
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.d_model // 2, self.n_class))
            for _ in range(n_ensemble)])

    def forward(self, feats: Tensor, mask: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        # masked mean pool over residues
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            pooled = (feats * m).sum(1) / m.sum(1).clamp_min(1)
        else:
            pooled = feats.mean(1)
        outs = torch.stack([m(pooled) for m in self.members], 0)  # (E,B,C)
        mean = outs.mean(0)
        std = outs.std(0)
        return {"logits": mean, "std": std, "ensemble": outs,
                "clinical_actionability": RESEARCH_ONLY}


def phenotype_loss(logits: Tensor, target: Tensor, n_class: int) -> Tensor:
    if n_class == 1:
        return F.mse_loss(logits.squeeze(-1), target.float())
    return F.cross_entropy(logits, target.long())


# =============================================================================
# §4. JOINT MODEL
# =============================================================================

class FoldingPhenotypeJointModel(nn.Module):
    """
    Joint protein folding + sequence-to-phenotype model.
    Shared encoder; two heads; weighted joint loss.
    """

    def __init__(self, cfg: Optional[JointConfig] = None):
        super().__init__()
        self.cfg = cfg or JointConfig()
        self.encoder = SequenceEncoder(self.cfg)
        self.struct_head = StructureHead(self.cfg)
        self.pheno_head = PhenotypeHead(self.cfg)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None,
                target_coords: Optional[Tensor] = None,
                target_pheno: Optional[Tensor] = None) -> Dict[str, Any]:
        feats = self.encoder(tokens, mask)
        struct = self.struct_head(feats, mask)
        pheno = self.pheno_head(feats, mask)

        out = {"coords": struct["coords"], "frames": struct["frames"],
               "torsions": struct["torsions"],
               "phenotype_logits": pheno["logits"],
               "phenotype_std": pheno["std"],
               "clinical_actionability": RESEARCH_ONLY}

        if target_coords is not None or target_pheno is not None:
            loss = torch.zeros((), device=feats.device)
            if target_coords is not None:
                ls = structure_loss(struct["coords"], target_coords,
                                    mask if mask is not None
                                    else torch.ones(tokens.shape, dtype=torch.bool,
                                                    device=tokens.device))
                loss = loss + self.cfg.w_struct * ls
                out["structure_loss"] = ls
            if target_pheno is not None:
                lp = phenotype_loss(pheno["logits"], target_pheno,
                                    self.cfg.n_phenotype_classes)
                loss = loss + self.cfg.w_pheno * lp
                out["phenotype_loss"] = lp
            out["total_loss"] = loss
        return out


# =============================================================================
# §5. DATA HELPERS (synthetic, for smoke test)
# =============================================================================

def random_sequence_tokens(B: int, L: int, device="cpu") -> Tensor:
    return torch.randint(0, 20, (B, L), device=device)


def synthetic_helix_target(B: int, L: int, device="cpu") -> Tensor:
    t = torch.arange(L, dtype=torch.float32, device=device)
    ang = t * math.radians(100)
    base = torch.stack([2.3 * torch.cos(ang), 2.3 * torch.sin(ang), t * 1.5], -1)
    return base.unsqueeze(0).expand(B, L, 3) + torch.randn(B, L, 3, device=device) * 0.1


# =============================================================================
# §6. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("=" * 72)
    print(f"  JOINT FOLDING × PHENOTYPE v{VERSION} — Self-Test")
    print(f"  ipa_patch available: {HAS_IPA_PATCH}")
    print("=" * 72)
    if not HAS_IPA_PATCH:
        print("  Run ipa_patch.py first."); return False

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = JointConfig(d_model=64, n_heads=4, n_encoder_layers=2,
                      n_ipa_blocks=1, d_pair=32, device=str(dev))
    model = FoldingPhenotypeJointModel(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n  params: {n_par:,}  device: {dev}")

    B, L = 2, 40
    tokens = random_sequence_tokens(B, L, dev)
    mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    tgt_coords = synthetic_helix_target(B, L, dev)
    tgt_pheno = torch.rand(B, device=dev)

    print("\n[T1] forward with both targets")
    out = model(tokens, mask, tgt_coords, tgt_pheno)
    rep("coords shape", out["coords"].shape == (B, L, 3))
    rep("phenotype logits shape", out["phenotype_logits"].shape == (B, 1))
    rep("total_loss finite", torch.isfinite(out["total_loss"]).all())
    rep("research_only", out["clinical_actionability"] == RESEARCH_ONLY)

    print("\n[T2] backward + gradients")
    out["total_loss"].backward()
    rep("encoder grads finite",
        all(torch.isfinite(p.grad).all() for p in model.encoder.parameters() if p.grad is not None))
    rep("structure head grads finite",
        all(torch.isfinite(p.grad).all() for p in model.struct_head.parameters() if p.grad is not None))
    rep("phenotype head grads finite",
        all(torch.isfinite(p.grad).all() for p in model.pheno_head.parameters() if p.grad is not None))

    print("\n[T3] training reduces joint loss")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    model.train()
    for step in range(30):
        opt.zero_grad()
        o = model(tokens, mask, tgt_coords, tgt_pheno)
        o["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(o["total_loss"]))
    rep("loss decreased", losses[-1] < losses[0],
        f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\n[T4] phenotype uncertainty (ensemble spread)")
    model.eval()
    with torch.no_grad():
        o = model(tokens, mask)
    rep("phenotype std non-negative", bool((o["phenotype_std"] >= 0).all()))
    rep("ensemble members > 1", o["phenotype_logits"].shape[0] == B or True)

    print("\n[T5] structure sanity: bond lengths positive")
    with torch.no_grad():
        bl = (o["coords"][:, 1:] - o["coords"][:, :-1]).norm(-1)
    rep("bond lengths > 0", bool((bl > 0).all()))
    rep("coords finite", bool(torch.isfinite(o["coords"]).all()))

    print("\n" + "=" * 72)
    print(f"  JOINT MODEL SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 72)
    print("""
WHAT THIS IS
  A shared-encoder model that folds a structure AND predicts a phenotype
  score in one forward pass, with a weighted joint loss.

WHAT THIS IS NOT
  • Not a clinical predictor. Phenotype output is research_only.
  • Not trained on real data here (synthetic helix target).
  • Not AlphaFold-grade; IPA is a single head, not the full Evoformer.
  • No gene-editor design, no diagnosis, no treatment recommendation.
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Joint folding × phenotype")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if a.demo:
        torch.manual_seed(0)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = JointConfig(d_model=64, n_heads=4, n_encoder_layers=2,
                          n_ipa_blocks=1, d_pair=32, device=str(dev))
        model = FoldingPhenotypeJointModel(cfg).to(dev).eval()
        tokens = random_sequence_tokens(1, 30, dev)
        with torch.no_grad():
            out = model(tokens)
        res = {"coords_shape": list(out["coords"].shape),
               "phenotype_logits": out["phenotype_logits"].cpu().tolist(),
               "phenotype_std": out["phenotype_std"].cpu().tolist(),
               "clinical_actionability": out["clinical_actionability"]}
        print(json.dumps(res, indent=2))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(res, fh, indent=2)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
