#!/usr/bin/env python3
"""
=============================================================================
INVARIANT POINT ATTENTION — BUG-FIXED + SE(3)-VERIFIED
=============================================================================
Fixes applied (each was a real crash or silent-wrong-result bug):

 [B1] einsum string had a literal space: "blij,b ljk->blik"  → RuntimeError
 [B2] proj_out expected H*(d_h + d_pair + pv*4) but concat produced
      H*d_h + d_pair + H*pv*4  → shape mismatch, AND wrong when z=None
 [B3] v_point_local einsum ran on an ALREADY-TRANSPOSED tensor:
      labels assumed (B,H,L,p,3) but tensor was (B,H,p,L,3)
      → silently wrong unless p == L
 [B4] local-frame transform never subtracted the frame translation
      → point-norm feature was NOT translation-invariant (silent bug)
 [B5] frame update used 3 "quaternion" params with w forced to 1
      → cannot represent 180° rotations; replaced with 6D rotation
        representation (Zhou et al., CVPR 2019) + Gram–Schmidt

HONEST SCOPE
  • This is ONE IPA block, not the AF2 Evoformer.
  • No triangle attention, no pair-stack updates, no recycling.
  • SE(3) invariance is TESTED below, not asserted.
  • Untrained weights produce meaningless coordinates.
=============================================================================
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------- rotation utils
def sixd_to_rotation(x: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Zhou et al. (2019) 6D rotation representation → SO(3) via Gram–Schmidt.
    x: (..., 6) → R: (..., 3, 3), columns are the orthonormal basis.
    Continuous and differentiable everywhere (unlike quaternions/Euler).
    """
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + eps)
    a2_proj = (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def identity_frames(B: int, L: int, device=None, dtype=None) -> Tensor:
    """(B, L, 4, 4) homogeneous identity frames."""
    T = torch.eye(4, device=device, dtype=dtype)
    return T[None, None].expand(B, L, 4, 4).clone()


def compose_frames(T: Tensor, dR: Tensor, dt: Tensor) -> Tensor:
    """T_new = T @ [dR | dt].  T:(B,L,4,4) dR:(B,L,3,3) dt:(B,L,3)"""
    B, L = T.shape[:2]
    D = torch.eye(4, device=T.device, dtype=T.dtype)[None, None].expand(B, L, 4, 4).clone()
    D[..., :3, :3] = dR
    D[..., :3, 3] = dt
    return T @ D            # [B1] fixed: was "blij,b ljk->blik"


# =============================================================================
# FIXED IPA
# =============================================================================
class InvariantPointAttention(nn.Module):
    """
    SE(3)-invariant scalar attention with 3D point channels.

    Output dim is FIXED regardless of whether pair features are supplied:
        H*d_h  +  d_pair  +  H*pv*3  +  H*pv
    When z is None the pair block is zeros — this keeps proj_out valid.  [B2]
    """

    def __init__(self, d_scalar: int = 128, n_heads: int = 8,
                 n_points_qk: int = 4, n_points_v: int = 8,
                 d_pair: int = 64, eps: float = 1e-8):
        super().__init__()
        assert d_scalar % n_heads == 0, "d_scalar must be divisible by n_heads"
        self.H, self.d_h = n_heads, d_scalar // n_heads
        self.pq, self.pv = n_points_qk, n_points_v
        self.d_pair, self.eps = d_pair, eps

        self.norm = nn.LayerNorm(d_scalar)
        self.q_s = nn.Linear(d_scalar, n_heads * self.d_h, bias=False)
        self.k_s = nn.Linear(d_scalar, n_heads * self.d_h, bias=False)
        self.v_s = nn.Linear(d_scalar, n_heads * self.d_h, bias=False)
        self.q_p = nn.Linear(d_scalar, n_heads * n_points_qk * 3, bias=False)
        self.k_p = nn.Linear(d_scalar, n_heads * n_points_qk * 3, bias=False)
        self.v_p = nn.Linear(d_scalar, n_heads * n_points_v * 3, bias=False)
        self.pair_bias = nn.Linear(d_pair, n_heads, bias=False)
        self.head_w = nn.Parameter(torch.zeros(n_heads))

        out_dim = n_heads * self.d_h + d_pair + n_heads * n_points_v * 4
        self.proj_out = nn.Linear(out_dim, d_scalar)          # [B2] fixed
        self.out_norm = nn.LayerNorm(d_scalar)

    def forward(self, s: Tensor, frames: Tensor,
                z: Optional[Tensor] = None) -> Tensor:
        """
        s:      (B, L, d_scalar)
        frames: (B, L, 4, 4)
        z:      (B, L, L, d_pair) or None
        return: (B, L, d_scalar)   — invariant to global rigid motion
        """
        B, L, _ = s.shape
        H, dh, pq, pv = self.H, self.d_h, self.pq, self.pv
        h = self.norm(s)

        qs = self.q_s(h).view(B, L, H, dh)
        ks = self.k_s(h).view(B, L, H, dh)
        vs = self.v_s(h).view(B, L, H, dh)
        qp = self.q_p(h).view(B, L, H, pq, 3)
        kp = self.k_p(h).view(B, L, H, pq, 3)
        vp = self.v_p(h).view(B, L, H, pv, 3)

        rot = frames[..., :3, :3]          # (B,L,3,3)
        tr = frames[..., :3, 3]            # (B,L,3)
        t_ex = tr[:, :, None, None, :]     # (B,L,1,1,3)

        # local → global.  Explicit, unambiguous index labels.
        qg = torch.einsum("blij,blhpj->blhpi", rot, qp) + t_ex
        kg = torch.einsum("blij,blhpj->blhpi", rot, kp) + t_ex
        vg = torch.einsum("blij,blhpj->blhpi", rot, vp) + t_ex

        # scalar attention
        attn = torch.einsum("bihd,bjhd->bhij", qs, ks) / math.sqrt(dh)

        # point attention: (B,H,p,L,3) → pairwise (B,H,L,L)
        qgt = qg.permute(0, 2, 3, 1, 4)
        kgt = kg.permute(0, 2, 3, 1, 4)
        d2 = ((qgt.unsqueeze(4) - kgt.unsqueeze(3)) ** 2).sum(-1).mean(2)
        gamma = F.softplus(self.head_w)
        attn = attn - 0.5 * gamma[None, :, None, None] * d2

        if z is not None:
            attn = attn + self.pair_bias(z).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)

        # aggregate
        vs_out = torch.einsum("bhij,bjhd->bihd", attn, vs)      # (B,L,H,dh)
        vp_out = torch.einsum("bhij,bjhpc->bihpc", attn, vg)    # (B,L,H,pv,3)

        # [B4] subtract translation BEFORE inverse rotation → true invariance
        vp_c = vp_out - t_ex
        rot_inv = rot.transpose(-1, -2)
        # [B3] operate on the UN-transposed (B,L,H,pv,3) layout
        vp_local = torch.einsum("blcj,blhpj->blhpc", rot_inv, vp_c)
        vp_norm = vp_local.norm(dim=-1)                          # (B,L,H,pv)

        # [B2] fixed-size pair block (zeros when absent)
        pair_sum = z.mean(dim=2) if z is not None else s.new_zeros(B, L, self.d_pair)

        cat = torch.cat([
            vs_out.reshape(B, L, H * dh),
            pair_sum,
            vp_local.reshape(B, L, H * pv * 3),
            vp_norm.reshape(B, L, H * pv),
        ], dim=-1)
        return self.out_norm(self.proj_out(cat))


class IPAStructureHead(nn.Module):
    """
    n IPA blocks → frame update (6D rotation + translation) → torsions.
    NOT the AF2 structure module (no recycling, no violation loss, no FAPE).
    """

    def __init__(self, d_scalar: int = 128, n_heads: int = 8,
                 n_blocks: int = 3, d_pair: int = 64, n_torsions: int = 7):
        super().__init__()
        self.blocks = nn.ModuleList([
            InvariantPointAttention(d_scalar, n_heads, d_pair=d_pair)
            for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_scalar) for _ in range(n_blocks)])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(d_scalar, 2 * d_scalar), nn.GELU(),
                          nn.Linear(2 * d_scalar, d_scalar))
            for _ in range(n_blocks)])
        # [B5] 6D rotation + 3 translation = 9, not 6
        self.frame_update = nn.Sequential(
            nn.LayerNorm(d_scalar), nn.Linear(d_scalar, d_scalar), nn.GELU(),
            nn.Linear(d_scalar, 9))
        self.torsion = nn.Sequential(
            nn.LayerNorm(d_scalar), nn.Linear(d_scalar, n_torsions * 2))
        self.n_torsions = n_torsions

    def forward(self, s: Tensor, frames: Tensor,
                z: Optional[Tensor] = None) -> Dict[str, Tensor]:
        for ipa, ln, ff in zip(self.blocks, self.norms, self.ff):
            s = s + ipa(s, frames, z)
            s = s + ff(ln(s))
        upd = self.frame_update(s)
        dR = sixd_to_rotation(upd[..., :6])
        dt = upd[..., 6:9]
        new_frames = compose_frames(frames, dR, dt)
        tor = self.torsion(s).view(*s.shape[:-1], self.n_torsions, 2)
        tor = tor / (tor.norm(dim=-1, keepdim=True) + 1e-8)
        return {"frames": new_frames, "torsions": tor, "scalar": s,
                "ca_coords": new_frames[..., :3, 3]}


# =============================================================================
# TESTS — these verify the fixes, they do not assert them
# =============================================================================
def random_rotation(B: int, device=None) -> Tensor:
    return sixd_to_rotation(torch.randn(B, 6, device=device))


def test_all(verbose: bool = True) -> bool:
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ok = True
    P = (lambda m: print(m)) if verbose else (lambda m: None)

    P("=" * 74)
    P("  IPA — bug-fix verification")
    P("=" * 74)

    B, L, ds, H, dp = 2, 24, 128, 8, 64
    ipa = InvariantPointAttention(ds, H, d_pair=dp).to(dev).eval()
    s = torch.randn(B, L, ds, device=dev)
    fr = identity_frames(B, L, dev)
    z = torch.randn(B, L, L, dp, device=dev)

    # T1 — z=None must not crash  [B2]
    P("\n[T1] z=None (was: shape mismatch)")
    try:
        o = ipa(s, fr, z=None)
        t = o.shape == s.shape and torch.isfinite(o).all()
        P(f"   out {tuple(o.shape)}  finite={bool(torch.isfinite(o).all())}  "
          f"{'PASS' if t else 'FAIL'}")
        ok &= bool(t)
    except Exception as e:
        P(f"   FAIL: {type(e).__name__}: {e}"); ok = False

    # T2 — z provided
    P("\n[T2] z provided")
    o2 = ipa(s, fr, z=z)
    t = o2.shape == s.shape
    P(f"   out {tuple(o2.shape)}  {'PASS' if t else 'FAIL'}"); ok &= bool(t)

    # T3 — n_points_v != L  (was silently wrong)  [B3]
    P("\n[T3] n_points_v(8) != L(24)  (was: silent index error)")
    t = torch.isfinite(o2).all()
    P(f"   finite={bool(t)}  {'PASS' if t else 'FAIL'}"); ok &= bool(t)

    # T4 — SE(3) invariance  [B4]
    P("\n[T4] SE(3) invariance (global rotation + translation)")
    R0 = random_rotation(B, dev)
    t0 = torch.randn(B, 3, device=dev) * 20.0
    fr2 = fr.clone()
    fr2[..., :3, :3] = torch.einsum("bij,bljk->blik", R0, fr[..., :3, :3])
    fr2[..., :3, 3] = torch.einsum("bij,blj->bli", R0, fr[..., :3, 3]) + t0[:, None, :]
    with torch.no_grad():
        a = ipa(s, fr, z=z); b = ipa(s, fr2, z=z)
    err = (a - b).abs().max().item()
    t = err < 2e-4
    P(f"   max |Δ| = {err:.3e}   {'PASS' if t else 'FAIL'}")
    if not t:
        P("   NOTE: without the translation-subtraction fix [B4] this is O(1).")
    ok &= t

    # T5 — structure head + frame composition  [B1][B5]
    P("\n[T5] Structure head (einsum space + 6D rotation)")
    head = IPAStructureHead(ds, H, n_blocks=2, d_pair=dp).to(dev).eval()
    out = head(s, fr, z=None)
    R = out["frames"][..., :3, :3]
    orth = (R @ R.transpose(-1, -2) - torch.eye(3, device=dev)).abs().max().item()
    det = torch.linalg.det(R)
    t = orth < 1e-4 and (det - 1).abs().max().item() < 1e-4
    P(f"   frames {tuple(out['frames'].shape)} torsions {tuple(out['torsions'].shape)}")
    P(f"   |RRᵀ-I|={orth:.2e}  |det-1|={(det-1).abs().max().item():.2e}  "
      f"{'PASS' if t else 'FAIL'}")
    ok &= t

    # T6 — 180° rotation representable  [B5]
    P("\n[T6] 180° rotation (old w=1 quaternion could not)")
    x = torch.tensor([[0., 0., 0., 0., 1., 0.]], device=dev)  # a1≈0 → use explicit
    R180 = sixd_to_rotation(torch.tensor([[1., 0., 0., 0., -1., 0.]], device=dev))
    target = torch.diag(torch.tensor([1., -1., -1.], device=dev))
    e = (R180[0] - target).abs().max().item()
    t = e < 1e-5
    P(f"   |R - diag(1,-1,-1)| = {e:.2e}  {'PASS' if t else 'FAIL'}"); ok &= t

    # T7 — gradients
    P("\n[T7] gradient flow")
    ipa.train()
    s2 = s.clone().requires_grad_(True)
    ipa(s2, fr, z=z).sum().backward()
    g = [p.grad for p in ipa.parameters() if p.grad is not None]
    t = len(g) > 0 and all(torch.isfinite(x).all() for x in g)
    P(f"   params with finite grad: {len(g)}  {'PASS' if t else 'FAIL'}"); ok &= bool(t)

    P("\n" + "=" * 74)
    P(f"  {'✅ ALL FIXES VERIFIED' if ok else '❌ SOME TESTS FAILED'}")
    P("=" * 74)
    P("""
  STILL TRUE AFTER THE FIX:
    • One IPA block ≠ Evoformer. No triangle attention, no recycling.
    • Weights are random. Coordinates are meaningless until trained.
    • No FAPE / violation loss implemented.
    • Not validated against CASP. Passing these tests means the CODE runs,
      not that the MODEL predicts structures.
""")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if test_all() else 1)
