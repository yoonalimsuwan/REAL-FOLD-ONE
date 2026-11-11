#!/usr/bin/env python3
"""
=============================================================================
AF2 AUXILIARY LOSSES: DISTOGRAM + pLDDT + VIOLATION  (with FAPE)
=============================================================================
License: MIT | 2026

LITERATURE
 [A1] Jumper et al. "Highly accurate protein structure prediction with
      AlphaFold." Nature 596, 583 (2021).
      Supplementary: §1.9 losses, Alg. 23 FAPE, Alg. 27 lDDT-Cα,
      Alg. 28–30 violation losses.
 [A2] Mariani et al. "lDDT: a local superposition-free score for comparing
      protein structures and models using distance difference tests."
      Bioinformatics 29, 2722 (2013).
 [A3] Zhou et al. "On the continuity of rotation representations in neural
      networks." CVPR (2019).
 [A4] Engh & Huber. "Accurate bond and angle parameters for X-ray protein
      structure refinement." Acta Cryst. A47, 392 (1991).

=============================================================================
WHY FAPE ALONE IS NOT ENOUGH
=============================================================================
FAPE supervises frames + atom positions but gives weak signal on:
  • global topology at long range   → distogram loss
  • per-residue reliability         → pLDDT loss (calibrated confidence)
  • physical plausibility           → violation loss (bonds, angles, clashes)
AF2 uses all four jointly. Without them, training is unstable and produces
geometrically invalid structures even when FAPE looks acceptable.

HONEST SCOPE
  ✅ Distogram, lDDT-Cα, pLDDT binning and violation terms follow the
     published definitions; each is unit-tested below.
  ⚠️ Constants (bin edges, tolerances, ideal geometry) are the commonly
     cited values. VERIFY against the AF2 supplementary before publication.
  ❌ No experimentally-resolved loss, no per-atom sidechain torsion loss,
     no Amber relaxation, no template loss.
  ❌ These losses do not create accuracy. They need real training data,
     a real MSA, and compute at a scale this file does not supply.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("af2_losses")


@dataclass
class LossConfig:
    # distogram [A1]
    n_dist_bins: int = 64
    dist_min: float = 2.3125
    dist_max: float = 21.6875
    # lDDT / pLDDT [A1,A2]
    lddt_thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    lddt_inclusion_radius: float = 15.0
    n_plddt_bins: int = 50
    # violation [A1,A4]
    bond_cn: float = 1.329          # C–N peptide bond (Å)
    bond_n_ca: float = 1.459
    bond_ca_c: float = 1.525
    angle_n_ca_c: float = 111.0     # degrees
    angle_ca_c_n: float = 116.2
    angle_c_n_ca: float = 121.7
    bond_tolerance: float = 0.10    # Å, 4σ-style slack
    angle_tolerance: float = 6.0    # degrees
    clash_tolerance: float = 1.5    # Å below vdW sum
    # FAPE
    fape_clamp: float = 10.0
    fape_chunk: int = 64
    # weights (AF2-like defaults; tune)
    w_fape: float = 1.0
    w_distogram: float = 0.3
    w_plddt: float = 0.01
    w_violation: float = 0.03


VDW_R = {"C":1.70,"N":1.55,"O":1.52,"S":1.80}
BB = ["N","CA","C","O"]


# =============================================================================
# 1. FAPE  (Alg. 23)
# =============================================================================

def fape(pred_frames, pred_pos, true_frames, true_pos,
         clamp=10.0, frame_mask=None, atom_mask=None,
         chunk=64, eps=1e-8) -> Tensor:
    B, L = pred_frames.shape[:2]
    Rp, tp = pred_frames[..., :3, :3], pred_frames[..., :3, 3]
    Rt, tt = true_frames[..., :3, :3], true_frames[..., :3, 3]
    if frame_mask is None: frame_mask = pred_pos.new_ones(B, L)
    if atom_mask is None: atom_mask = pred_pos.new_ones(B, L, pred_pos.shape[2])
    w = frame_mask[:, :, None, None]*atom_mask[:, None, :, :]
    tot = pred_pos.new_zeros(B); cnt = pred_pos.new_zeros(B)
    for s in range(0, L, chunk):
        e = min(s+chunk, L)
        dp = pred_pos[:, None] - tp[:, s:e, None, None]
        dt = true_pos[:, None] - tt[:, s:e, None, None]
        lp = torch.einsum("bijk,bilaj->bilak", Rp[:, s:e], dp)
        lt = torch.einsum("bijk,bilaj->bilak", Rt[:, s:e], dt)
        d = torch.sqrt(((lp-lt)**2).sum(-1)+eps)
        if clamp: d = d.clamp(max=clamp)
        wc = w[:, s:e]
        tot = tot + (d*wc).sum((1,2,3)); cnt = cnt + wc.sum((1,2,3))
    return (tot/cnt.clamp(min=1))/(clamp or 1.0)


# =============================================================================
# 2. DISTOGRAM LOSS
# =============================================================================

def cb_or_ca(atoms: Tensor, atom_mask: Optional[Tensor] = None) -> Tensor:
    """
    Cβ for non-Gly; Cα fallback. Here we use Cα (index 1) as a documented
    simplification when Cβ is unavailable.
    atoms: (B,L,4,3) ordered N,CA,C,O
    """
    return atoms[..., 1, :]


def distogram_bins(cfg: LossConfig, device=None) -> Tensor:
    return torch.linspace(cfg.dist_min, cfg.dist_max, cfg.n_dist_bins,
                          device=device)


def distogram_loss(logits: Tensor, atoms: Tensor,
                   mask: Optional[Tensor], cfg: LossConfig) -> Dict[str, Tensor]:
    """
    logits: (B,L,L,n_bins) predicted distogram
    Cross-entropy against binned true Cβ/Cα distances.
    Last bin is an open "greater than" bin.
    """
    pts = cb_or_ca(atoms)
    d = torch.cdist(pts, pts)
    edges = distogram_bins(cfg, d.device)
    width = edges[1]-edges[0]
    idx = torch.floor((d - cfg.dist_min)/width).long()
    idx = idx.clamp(0, cfg.n_dist_bins-1)
    target = F.one_hot(idx, cfg.n_dist_bins).float()
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(target*logp).sum(-1)
    if mask is not None:
        pm = mask[:, :, None]*mask[:, None, :]
        ce = (ce*pm).sum()/pm.sum().clamp(min=1)
    else:
        ce = ce.mean()
    return {"distogram": ce, "distogram_bins": edges}


# =============================================================================
# 3. lDDT-Cα  (Alg. 27) + pLDDT LOSS
# =============================================================================

def lddt_ca(pred_ca: Tensor, true_ca: Tensor,
            mask: Optional[Tensor], cfg: LossConfig,
            per_residue: bool = True) -> Tensor:
    """
    For each residue i, over j≠i with true distance < inclusion radius:
    preserved if |d_pred − d_true| < threshold, for each threshold.
    Returns lDDT in [0,100], per residue (B,L) or mean (B,).
    """
    dp = torch.cdist(pred_ca, pred_ca)
    dt = torch.cdist(true_ca, true_ca)
    B, L = dt.shape
    if mask is None: mask = torch.ones(B, L, device=dt.device)
    pair = mask[:, :, None]*mask[:, None, :]
    eye = torch.eye(L, device=dt.device)[None]
    pair = pair*(1-eye)
    include = (dt < cfg.lddt_inclusion_radius).float()*pair
    diff = (dp-dt).abs()
    preserved = torch.stack(
        [(diff < t).float() for t in cfg.lddt_thresholds], dim=-1).mean(-1)
    num = (preserved*include).sum(-1)
    den = include.sum(-1).clamp(min=1)
    score = 100.0*num/den
    score = score*mask
    return score if per_residue else score.sum(-1)/mask.sum(-1).clamp(min=1)


def plddt_loss(logits: Tensor, pred_ca: Tensor, true_ca: Tensor,
               mask: Optional[Tensor], cfg: LossConfig) -> Dict[str, Tensor]:
    """
    logits: (B,L,n_plddt_bins). Target = binned TRUE lDDT (stop-gradient).
    """
    with torch.no_grad():
        true_lddt = lddt_ca(pred_ca.detach(), true_ca, mask, cfg)
        binw = 100.0/cfg.n_plddt_bins
        idx = torch.floor(true_lddt/binw).long().clamp(0, cfg.n_plddt_bins-1)
        target = F.one_hot(idx, cfg.n_plddt_bins).float()
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = -(target*logp).sum(-1)
    if mask is not None:
        ce = (ce*mask).sum()/mask.sum().clamp(min=1)
    else:
        ce = ce.mean()
    # expected pLDDT for reporting
    centers = (torch.arange(cfg.n_plddt_bins, device=logits.device).float()+0.5)*binw
    plddt_pred = (F.softmax(logits.float(), -1)*centers).sum(-1)
    return {"plddt": ce, "plddt_pred": plddt_pred,
            "lddt_true": true_lddt}


# =============================================================================
# 4. VIOLATION LOSSES  (Alg. 28–30)
# =============================================================================

def _angle(a: Tensor, b: Tensor, c: Tensor, eps: float = 1e-8) -> Tensor:
    v1 = a-b; v2 = c-b
    cos = (v1*v2).sum(-1)/(v1.norm(dim=-1)*v2.norm(dim=-1)+eps)
    return torch.rad2deg(torch.acos(cos.clamp(-1+1e-6, 1-1e-6)))


def between_residue_bond_loss(atoms: Tensor, mask: Optional[Tensor],
                              cfg: LossConfig) -> Dict[str, Tensor]:
    """Peptide-bond geometry across consecutive residues."""
    N, CA, C = atoms[..., 0, :], atoms[..., 1, :], atoms[..., 2, :]
    if mask is None: mask = torch.ones(atoms.shape[:2], device=atoms.device)
    pm = mask[:, :-1]*mask[:, 1:]
    d_cn = (C[:, :-1]-N[:, 1:]).norm(dim=-1)
    l_cn = F.relu((d_cn-cfg.bond_cn).abs()-cfg.bond_tolerance)
    a1 = _angle(CA[:, :-1], C[:, :-1], N[:, 1:])
    a2 = _angle(C[:, :-1], N[:, 1:], CA[:, 1:])
    l_a1 = F.relu((a1-cfg.angle_ca_c_n).abs()-cfg.angle_tolerance)
    l_a2 = F.relu((a2-cfg.angle_c_n_ca).abs()-cfg.angle_tolerance)
    den = pm.sum().clamp(min=1)
    return {"bond_violation": ((l_cn+l_a1+l_a2)*pm).sum()/den}


def within_residue_geometry_loss(atoms: Tensor, mask: Optional[Tensor],
                                 cfg: LossConfig) -> Dict[str, Tensor]:
    N, CA, C = atoms[..., 0, :], atoms[..., 1, :], atoms[..., 2, :]
    if mask is None: mask = torch.ones(atoms.shape[:2], device=atoms.device)
    d1 = (N-CA).norm(dim=-1); d2 = (CA-C).norm(dim=-1)
    l = (F.relu((d1-cfg.bond_n_ca).abs()-cfg.bond_tolerance) +
         F.relu((d2-cfg.bond_ca_c).abs()-cfg.bond_tolerance))
    ang = _angle(N, CA, C)
    l = l + F.relu((ang-cfg.angle_n_ca_c).abs()-cfg.angle_tolerance)
    return {"within_residue_violation":
            (l*mask).sum()/mask.sum().clamp(min=1)}


def clash_loss(atoms: Tensor, mask: Optional[Tensor], cfg: LossConfig,
               atom_types=("C","N","O","S")) -> Dict[str, Tensor]:
    """
    Non-bonded clash: distance below (vdW_i + vdW_j − tolerance).
    Excludes bonded pairs (same residue adjacent backbone atoms, and the
    peptide C–N bond) by a sequence/atom-index separation rule.
    """
    B, L, A, _ = atoms.shape
    pts = atoms.reshape(B, L*A, 3)
    radii = torch.tensor([VDW_R.get(atom_types[j % len(atom_types)], 1.7)
                          for j in range(L*A)], device=atoms.device)
    d = torch.cdist(pts, pts)
    lim = radii[:, None]+radii[None, :]-cfg.clash_tolerance
    idx = torch.arange(L*A, device=atoms.device)
    res_i = idx // A; res_j = idx // A
    # exclude same-residue bonded and adjacent-residue peptide bond
    sep = (res_i[:, None]-res_j[None, :]).abs()
    same = sep == 0
    adj = sep == 1
    exclude = same | adj | (d > lim)
    viol = F.relu(lim-d)*(~exclude).float()
    if mask is not None:
        am = mask.repeat_interleave(A, dim=1)
        pm = am[:, :, None]*am[:, None, :]
        viol = viol*pm
        den = pm.sum().clamp(min=1)
    else:
        den = torch.tensor(float(B*(L*A)**2), device=atoms.device)
    return {"clash": viol.sum()/den}


# =============================================================================
# 5. COMBINED LOSS
# =============================================================================

class AF2Loss(nn.Module):
    def __init__(self, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()

    def forward(self, pred: Dict[str, Tensor], true: Dict[str, Tensor],
                mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        c = self.cfg
        out = {}
        out["fape"] = fape(pred["frames"], pred["positions"],
                           true["frames"], true["atoms"],
                           clamp=c.fape_clamp, frame_mask=mask,
                           atom_mask=true.get("atom_mask"),
                           chunk=c.fape_chunk).mean()
        if "distogram_logits" in pred:
            out.update({k: v for k, v in distogram_loss(
                pred["distogram_logits"], true["atoms"], mask, c).items()
                if k == "distogram"})
        else:
            out["distogram"] = torch.zeros((), device=out["fape"].device)
        pca = pred["positions"][..., 1, :]
        tca = true["atoms"][..., 1, :]
        if "plddt_logits" in pred:
            pl = plddt_loss(pred["plddt_logits"], pca, tca, mask, c)
            out["plddt"] = pl["plddt"]
            out["plddt_pred"] = pl["plddt_pred"]
            out["lddt_true"] = pl["lddt_true"]
        else:
            out["plddt"] = torch.zeros((), device=out["fape"].device)
        out.update(between_residue_bond_loss(pred["positions"], mask, c))
        out.update(within_residue_geometry_loss(pred["positions"], mask, c))
        out.update(clash_loss(pred["positions"], mask, c))
        out["violation"] = (out["bond_violation"] +
                            out["within_residue_violation"] + out["clash"])
        out["total"] = (c.w_fape*out["fape"] + c.w_distogram*out["distogram"] +
                        c.w_plddt*out["plddt"] + c.w_violation*out["violation"])
        return out


# =============================================================================
# 6. SELF-TEST
# =============================================================================

def frames_from_atoms(atoms: Tensor, eps=1e-8) -> Tensor:
    N, CA, C = atoms[..., 0, :], atoms[..., 1, :], atoms[..., 2, :]
    v1, v2 = C-CA, N-CA
    e1 = v1/(v1.norm(dim=-1, keepdim=True)+eps)
    u2 = v2-(e1*v2).sum(-1, keepdim=True)*e1
    e2 = u2/(u2.norm(dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)
    B, L = atoms.shape[:2]
    T = torch.eye(4, device=atoms.device)[None, None].expand(B, L, 4, 4).clone()
    T[..., :3, :3] = R; T[..., :3, 3] = CA
    return T


def ideal_chain(L: int) -> Tensor:
    """Crude extended backbone for testing (not a real protein geometry)."""
    atoms = torch.zeros(1, L, 4, 3)
    for i in range(L):
        x = 3.8*i
        atoms[0, i, 0] = torch.tensor([x-1.459, 0.0, 0.0])   # N
        atoms[0, i, 1] = torch.tensor([x, 0.0, 0.0])         # CA
        atoms[0, i, 2] = torch.tensor([x+1.525, 0.5, 0.0])   # C
        atoms[0, i, 3] = torch.tensor([x+2.4, 1.3, 0.0])     # O
    return atoms


def selftest(args):
    print("="*76); print("  AF2 AUXILIARY LOSSES — self-test"); print("="*76)
    torch.manual_seed(0)
    cfg = LossConfig()
    ok = True
    B, L = 1, 24
    atoms = ideal_chain(L)
    T = frames_from_atoms(atoms)
    mask = torch.ones(B, L)

    print("\n[T1] FAPE zero on identity, invariant to rigid motion")
    f0 = fape(T, atoms, T, atoms).item()
    R0 = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.linalg.det(R0) < 0: R0[:, 0] *= -1
    t0 = torch.randn(3)*30
    T2 = T.clone()
    T2[..., :3, :3] = torch.einsum("ij,bljk->blik", R0, T[..., :3, :3])
    T2[..., :3, 3] = torch.einsum("ij,blj->bli", R0, T[..., :3, 3])+t0
    a2 = torch.einsum("ij,blaj->blai", R0, atoms)+t0
    f1 = fape(T2, a2, T, atoms).item()
    t = f0 < 1e-5 and f1 < 1e-4
    print(f"   identity={f0:.2e}  rigid={f1:.2e}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T2] lDDT = 100 for identical structures")
    ca = atoms[..., 1, :]
    s = lddt_ca(ca, ca, mask, cfg)
    t = float(s.mean()) > 99.9
    print(f"   lDDT={float(s.mean()):.3f}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T3] lDDT drops under perturbation")
    ca_p = ca + torch.randn_like(ca)*2.0
    s2 = lddt_ca(ca_p, ca, mask, cfg)
    t = float(s2.mean()) < float(s.mean())
    print(f"   perturbed lDDT={float(s2.mean()):.2f} < {float(s.mean()):.2f}"
          f"  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T4] distogram bin index correctness")
    edges = distogram_bins(cfg)
    width = edges[1]-edges[0]
    d = torch.cdist(ca, ca)
    idx = torch.floor((d-cfg.dist_min)/width).long().clamp(0, cfg.n_dist_bins-1)
    # a distance exactly at an edge should map to that bin
    probe = torch.tensor([[[cfg.dist_min+3*width]]])
    ip = int(torch.floor((probe-cfg.dist_min)/width).item())
    t = ip == 3 and idx.min() >= 0 and idx.max() < cfg.n_dist_bins
    print(f"   edge probe → bin {ip} (expect 3)  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T5] distogram loss is 0 for perfect one-hot logits")
    lg = F.one_hot(idx, cfg.n_dist_bins).float()*30.0
    dl = distogram_loss(lg, atoms, mask, cfg)["distogram"].item()
    t = dl < 1e-3
    print(f"   perfect-prediction CE={dl:.2e}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T6] pLDDT loss is 0 when logits match true lDDT bins")
    with torch.no_grad():
        tl = lddt_ca(ca, ca, mask, cfg)
        binw = 100.0/cfg.n_plddt_bins
        ti = torch.floor(tl/binw).long().clamp(0, cfg.n_plddt_bins-1)
    plg = F.one_hot(ti, cfg.n_plddt_bins).float()*30.0
    pl = plddt_loss(plg, ca, ca, mask, cfg)
    t = pl["plddt"].item() < 1e-3 and abs(pl["plddt_pred"].mean().item()-100) < 2.5
    print(f"   CE={pl['plddt'].item():.2e}  E[pLDDT]={pl['plddt_pred'].mean():.2f}")
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T7] violation loss detects a stretched bond")
    bad = atoms.clone(); bad[0, 5, 1, 0] += 3.0
    v_good = within_residue_geometry_loss(atoms, mask, cfg)["within_residue_violation"]
    v_bad = within_residue_geometry_loss(bad, mask, cfg)["within_residue_violation"]
    t = v_bad > v_good
    print(f"   good={v_good:.4f}  stretched={v_bad:.4f}  {'PASS' if t else 'FAIL'}")
    ok &= bool(t)

    print("\n[T8] clash loss detects an introduced overlap")
    cl_good = clash_loss(atoms, mask, cfg)["clash"].item()
    clash_atoms = atoms.clone()
    clash_atoms[0, 15, 1] = clash_atoms[0, 2, 1] + 0.3
    cl_bad = clash_loss(clash_atoms, mask, cfg)["clash"].item()
    t = cl_bad > cl_good
    print(f"   good={cl_good:.5f}  clash={cl_bad:.5f}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T9] combined loss backpropagates")
    pred = {"frames": T.clone().requires_grad_(True),
            "positions": atoms.clone().requires_grad_(True),
            "distogram_logits": torch.randn(B, L, L, cfg.n_dist_bins,
                                            requires_grad=True),
            "plddt_logits": torch.randn(B, L, cfg.n_plddt_bins,
                                        requires_grad=True)}
    true = {"frames": T, "atoms": atoms}
    losses = AF2Loss(cfg)(pred, true, mask)
    losses["total"].backward()
    g = [p.grad for p in pred.values() if p.grad is not None]
    t = len(g) == 4 and all(torch.isfinite(x).all() for x in g)
    print("   " + "  ".join(f"{k}={float(v):.4f}" for k, v in losses.items()
                            if v.ndim == 0))
    print(f"   finite grads on {len(g)}/4 tensors  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n" + "="*76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*76)
    print("""
  These losses make FAPE trainable in the AF2 sense. They do NOT create
  accuracy on their own: real MSA depth, real training data, and compute
  at scale are still required. Verify all constants against the AF2
  supplementary before quoting them.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest"])
    a = p.parse_args()
    raise SystemExit(0 if selftest(a) else 1)


if __name__ == "__main__":
    main()
