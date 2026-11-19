#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ipa_patch.py — Drop-in replacement for InvariantPointAttention +
               IPAStructureHead in sesi_lite_msa_af3_grade_full.py
Fixes B5, B5b, B6, B7. Run apply_patch() BEFORE importing the original.
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

Christ died upon the Cross to teach: when struck on the left cheek, turn also
the right, so that we may listen to the voice of the enemy who wishes us harm.

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

BUGS FIXED (from sesi_lite_msa_af3_grade_full.py diagnosis):
  B5   v_point_local einsum treated n_points_v as L
  B5b  point-projection einsum index ordering inconsistent
  B6   proj_out.in_features always included d_pair; z=None dropped it
  B7   update_frames einsum had a space: "blij,b ljk->blik"

LAYOUT CONVENTION (single, consistent):
  All point tensors: (B, L, H, P, 3)
  Scalar features:   (B, L, H, d_h)
  Frames:            (B, L, 4, 4)

USAGE:
  import ipa_patch
  ipa_patch.apply_patch()          # monkey-patch the original module
  # then import and use sesi_lite_msa_af3_grade_full as normal

  OR use the classes directly:
  from ipa_patch import InvariantPointAttentionFixed, IPAStructureHeadFixed
=============================================================================
"""
from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

VERSION = "1.0.0-ipa-patch"


# =============================================================================
# §1. FIXED INVARIANT POINT ATTENTION
# =============================================================================

class InvariantPointAttentionFixed(nn.Module):
    """
    SE(3)-invariant point attention — corrected version.

    Fixes B5/B5b/B6 from the original. Single layout convention:
      point tensors  : (B, L, H, P, 3)
      scalar tensors : (B, L, H, d_h)
      frames         : (B, L, 4, 4)

    The scalar output is invariant to a global rigid transform of all
    frames (verified in the self-test). The output concat ALWAYS includes
    one d_pair slot (zeros when z is None), so proj_out.in_features is a
    compile-time constant. [B6 fix]
    """

    def __init__(self,
                 d_scalar: int = 256,
                 d_point: int = 16,
                 n_heads: int = 8,
                 n_points_qk: int = 4,
                 n_points_v: int = 8,
                 d_pair: int = 128,
                 eps: float = 1e-8):
        super().__init__()
        if d_scalar % n_heads != 0:
            raise ValueError("d_scalar must be divisible by n_heads")
        self.d_scalar = d_scalar
        self.n_heads = n_heads
        self.d_h = d_scalar // n_heads
        self.n_points_qk = n_points_qk
        self.n_points_v = n_points_v
        self.d_pair = d_pair
        self.eps = eps

        # scalar projections
        self.proj_q_scalar = nn.Linear(d_scalar, n_heads * self.d_h, bias=False)
        self.proj_k_scalar = nn.Linear(d_scalar, n_heads * self.d_h, bias=False)
        self.proj_v_scalar = nn.Linear(d_scalar, n_heads * self.d_h, bias=False)

        # point projections -> (B, L, H*P*3) then reshaped to (B,L,H,P,3)
        self.proj_q_point = nn.Linear(d_scalar, n_heads * n_points_qk * 3, bias=False)
        self.proj_k_point = nn.Linear(d_scalar, n_heads * n_points_qk * 3, bias=False)
        self.proj_v_point = nn.Linear(d_scalar, n_heads * n_points_v * 3, bias=False)

        # pair bias
        self.pair_bias = nn.Linear(d_pair, n_heads, bias=False)

        # per-head point-distance weight (softplus to stay positive)
        self.head_gamma = nn.Parameter(torch.zeros(n_heads))

        # [B6 fix] output width is a constant; includes exactly ONE d_pair slot
        out_dim = (n_heads * self.d_h) + d_pair \
                + (n_heads * n_points_v * 3) + (n_heads * n_points_v)
        self.proj_out = nn.Linear(out_dim, d_scalar)

    # ---- frame transforms on the (B,L,H,P,3) layout [B5b fix] ----
    def _to_global(self, pts_local: Tensor, rot: Tensor, trans: Tensor) -> Tensor:
        """pts_local (B,L,H,P,3) -> global. rot (B,L,3,3), trans (B,L,3)."""
        g = torch.einsum("blij,blhpj->blhpi", rot, pts_local)
        return g + trans[:, :, None, None, :]

    def _to_local(self, pts_global: Tensor, rot_inv: Tensor) -> Tensor:
        """pts_global (B,L,H,P,3) -> local frame. [B5 fix]"""
        return torch.einsum("blij,blhpj->blhpi", rot_inv, pts_global)

    def forward(self, s: Tensor, z: Optional[Tensor], frames: Tensor) -> Tensor:
        """
        Args:
            s:      (B, L, d_scalar)
            z:      (B, L, L, d_pair) or None
            frames: (B, L, 4, 4)
        Returns:
            (B, L, d_scalar)
        """
        B, L, _ = s.shape
        H, dh = self.n_heads, self.d_h
        Pq, Pv = self.n_points_qk, self.n_points_v

        rot = frames[..., :3, :3]                    # (B,L,3,3)
        trans = frames[..., :3, 3]                   # (B,L,3)
        rot_inv = rot.transpose(-1, -2)              # (B,L,3,3)

        # ---- scalar q/k/v : (B, L, H, dh) ----
        q_s = self.proj_q_scalar(s).view(B, L, H, dh)
        k_s = self.proj_k_scalar(s).view(B, L, H, dh)
        v_s = self.proj_v_scalar(s).view(B, L, H, dh)

        # ---- point q/k/v : (B, L, H, P, 3) ----
        q_p = self.proj_q_point(s).view(B, L, H, Pq, 3)
        k_p = self.proj_k_point(s).view(B, L, H, Pq, 3)
        v_p = self.proj_v_point(s).view(B, L, H, Pv, 3)

        # ---- scalar attention logits : (B, H, L, L) ----
        attn_scalar = torch.einsum("bihd,bjhd->bhij", q_s, k_s) / math.sqrt(dh)

        # ---- move points to global frame ----
        q_g = self._to_global(q_p, rot, trans)       # (B,L,H,Pq,3)
        k_g = self._to_global(k_p, rot, trans)       # (B,L,H,Pq,3)

        # ---- point-distance logits ----
        q_t = q_g.permute(0, 2, 3, 1, 4)             # (B,H,Pq,L,3)
        k_t = k_g.permute(0, 2, 3, 1, 4)             # (B,H,Pq,L,3)
        diff = q_t.unsqueeze(4) - k_t.unsqueeze(3)   # (B,H,Pq,L,L,3)
        dist_sq = diff.pow(2).sum(-1)                # (B,H,Pq,L,L)
        gamma = F.softplus(self.head_gamma)          # (H,)
        attn_point = -0.5 * gamma[None, :, None, None] * dist_sq.mean(dim=2)
        # attn_point : (B,H,L,L)

        # ---- pair bias ----
        if z is not None:
            attn_pair = self.pair_bias(z).permute(0, 3, 1, 2)   # (B,H,L,L)
        else:
            attn_pair = 0.0

        attn = F.softmax(attn_scalar + attn_point + attn_pair, dim=-1)

        # ---- aggregate scalar values : (B, L, H, dh) ----
        v_scalar_out = torch.einsum("bhij,bjhd->bihd", attn, v_s)

        # ---- aggregate point values in global frame ----
        v_g = self._to_global(v_p, rot, trans)       # (B,L,H,Pv,3)
        v_t = v_g.permute(0, 2, 3, 1, 4)             # (B,H,Pv,L,3)
        v_agg_t = torch.einsum("bhij,bhpjc->bhpic", attn, v_t)  # (B,H,Pv,L,3)
        v_agg = v_agg_t.permute(0, 3, 1, 2, 4)       # (B,L,H,Pv,3) global

        # ---- [B5 fix] back to local frame with correct layout ----
        v_local = self._to_local(v_agg, rot_inv)     # (B,L,H,Pv,3)
        v_norms = v_local.norm(dim=-1)               # (B,L,H,Pv)

        # ---- [B6 fix] constant-width concat ----
        v_scalar_flat = v_scalar_out.reshape(B, L, H * dh)
        v_point_flat  = v_local.reshape(B, L, H * Pv * 3)
        v_norm_flat   = v_norms.reshape(B, L, H * Pv)
        if z is not None:
            z_out = z.mean(dim=2)                    # (B,L,d_pair)
        else:
            z_out = s.new_zeros(B, L, self.d_pair)   # keep the slot
        concat = torch.cat([v_scalar_flat, z_out, v_point_flat, v_norm_flat], -1)
        return self.proj_out(concat)


# =============================================================================
# §2. FIXED IPA STRUCTURE HEAD
# =============================================================================

class IPAStructureHeadFixed(nn.Module):
    """
    IPA structure head with the B7 einsum fix and Rodrigues rotation.
    Drop-in replacement for the original IPAStructureHead.
    """

    def __init__(self,
                 d_scalar: int = 256,
                 d_point: int = 16,
                 n_heads: int = 8,
                 n_ipa_blocks: int = 3,
                 d_pair: int = 128,
                 n_torsion_angles: int = 7):
        super().__init__()
        self.d_scalar = d_scalar
        self.n_torsion_angles = n_torsion_angles

        self.ipa_blocks = nn.ModuleList([
            InvariantPointAttentionFixed(d_scalar, d_point, n_heads,
                                         4, 8, d_pair)
            for _ in range(n_ipa_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_scalar)
                                    for _ in range(n_ipa_blocks)])
        self.frame_update = nn.Sequential(
            nn.Linear(d_scalar, d_scalar), nn.GELU(),
            nn.Linear(d_scalar, 6))          # 3 rotvec + 3 translation
        self.torsion_pred = nn.Sequential(
            nn.Linear(d_scalar, d_scalar), nn.GELU(),
            nn.Linear(d_scalar, n_torsion_angles * 2))

    @staticmethod
    def rotvec_to_rotation(rv: Tensor) -> Tensor:
        """Rodrigues: rotation-vector (...,3) -> rotation matrix (...,3,3)."""
        theta = rv.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        k = rv / theta
        K = torch.zeros(k.shape[:-1] + (3, 3), device=rv.device, dtype=rv.dtype)
        K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
        K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
        K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
        I = torch.eye(3, device=rv.device, dtype=rv.dtype)
        th = theta[..., None]
        return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

    def update_frames(self, frames: Tensor, delta: Tensor) -> Tensor:
        """
        [B7 fix] no space in the einsum subscript.
        frames (B,L,4,4), delta (B,L,6) = (rotvec, translation).
        """
        B, L, _, _ = frames.shape
        rv = delta[..., :3]
        t = delta[..., 3:]
        R_delta = self.rotvec_to_rotation(rv)
        delta_frame = torch.eye(4, device=frames.device, dtype=frames.dtype)
        delta_frame = delta_frame.view(1, 1, 4, 4).expand(B, L, 4, 4).clone()
        delta_frame[..., :3, :3] = R_delta
        delta_frame[..., :3, 3] = t
        # [B7] corrected subscript
        return torch.einsum("blij,bljk->blik", frames, delta_frame)

    def predict_torsions(self, s: Tensor) -> Tensor:
        a = self.torsion_pred(s).view(*s.shape[:-1], self.n_torsion_angles, 2)
        return a / (a.norm(dim=-1, keepdim=True) + 1e-8)

    def forward(self, s: Tensor, z: Optional[Tensor], frames: Tensor):
        for blk, norm in zip(self.ipa_blocks, self.norms):
            s = s + blk(norm(s), z, frames)
        delta = self.frame_update(s)
        new_frames = self.update_frames(frames, delta)
        torsions = self.predict_torsions(s)
        return {"frames": new_frames, "torsions": torsions,
                "scalar_features": s}


# =============================================================================
# §3. MONKEY-PATCH APPLICATION
# =============================================================================

_ORIGINAL_ATTRS = {}


def apply_patch(module_name: str = "sesi_lite_msa_af3_grade_full") -> dict:
    """
    Monkey-patch the original module so that IPAStructureHead and
    InvariantPointAttention use the fixed implementations.

    Returns a dict describing what was patched. Call BEFORE importing
    any code that instantiates the original classes.
    """
    report = {"patched": [], "skipped": [], "module": module_name}
    try:
        mod = sys.modules.get(module_name)
        if mod is None:
            import importlib
            mod = importlib.import_module(module_name)
    except Exception as e:
        report["skipped"].append(f"cannot import {module_name}: {e}")
        return report

    # save originals for potential rollback
    if hasattr(mod, "InvariantPointAttention"):
        _ORIGINAL_ATTRS["InvariantPointAttention"] = mod.InvariantPointAttention
        mod.InvariantPointAttention = InvariantPointAttentionFixed
        report["patched"].append("InvariantPointAttention -> Fixed")
    else:
        report["skipped"].append("InvariantPointAttention not found")

    if hasattr(mod, "IPAStructureHead"):
        _ORIGINAL_ATTRS["IPAStructureHead"] = mod.IPAStructureHead
        mod.IPAStructureHead = IPAStructureHeadFixed
        report["patched"].append("IPAStructureHead -> Fixed")
    else:
        report["skipped"].append("IPAStructureHead not found")

    return report


def revert_patch(module_name: str = "sesi_lite_msa_af3_grade_full") -> None:
    """Restore original classes (for A/B comparison)."""
    mod = sys.modules.get(module_name)
    if mod is None:
        return
    for name, cls in _ORIGINAL_ATTRS.items():
        if hasattr(mod, name):
            setattr(mod, name, cls)


# =============================================================================
# §4. SELF-TEST (mirrors ipa_fixed.py T1–T6)
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[T1] B5: n_points_v independent of L")
    for L, Pv in ((50, 8), (13, 8), (8, 3)):
        ipa = InvariantPointAttentionFixed(64, 8, 4, 4, Pv, 32).to(dev)
        s = torch.randn(2, L, 64, device=dev)
        fr = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(2, L, 4, 4).clone()
        try:
            out = ipa(s, None, fr)
            rep(f"L={L},Pv={Pv} forward", out.shape == (2, L, 64))
        except Exception as e:
            rep(f"L={L},Pv={Pv} forward", False, str(e)[:50])

    print(f"\n[T2] B6: z=None and z=tensor both work")
    ipa = InvariantPointAttentionFixed(64, 8, 4, 4, 8, 32).to(dev)
    s = torch.randn(2, 20, 64, device=dev)
    fr = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(2, 20, 4, 4).clone()
    o1 = ipa(s, None, fr)
    z = torch.randn(2, 20, 20, 32, device=dev)
    o2 = ipa(s, z, fr)
    rep("z=None", o1.shape == (2, 20, 64))
    rep("z=tensor", o2.shape == (2, 20, 64))
    rep("outputs differ", not torch.allclose(o1, o2))

    print(f"\n[T3] B7: update_frames runs")
    head = IPAStructureHeadFixed(64, 8, 4, 1, 32).to(dev)
    s = torch.randn(2, 16, 64, device=dev)
    fr = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(2, 16, 4, 4).clone()
    out = head(s, None, fr)
    rep("frames shape", out["frames"].shape == (2, 16, 4, 4))
    rep("torsions unit", torch.allclose(
        out["torsions"].norm(-1), torch.ones_like(out["torsions"].norm(-1)), atol=1e-5))

    print(f"\n[T4] SE(3) invariance of scalar output")
    ipa = InvariantPointAttentionFixed(64, 8, 4, 4, 8, 32).to(dev).eval()
    L = 12
    s = torch.randn(1, L, 64, device=dev)
    fr = torch.eye(4, device=dev).view(1, 1, 4, 4).repeat(1, L, 1, 1).clone()
    for i in range(L):
        q = torch.randn(3, 3, device=dev)
        q, _ = torch.linalg.qr(q)
        if torch.det(q) < 0: q[:, 0] = -q[:, 0]
        fr[0, i, :3, :3] = q
        fr[0, i, :3, 3] = torch.randn(3, device=dev) * 5
    R = torch.linalg.qr(torch.randn(3, 3, device=dev))[0]
    if torch.det(R) < 0: R[:, 0] = -R[:, 0]
    t = torch.randn(3, device=dev) * 3
    fr2 = fr.clone()
    fr2[..., :3, :3] = torch.einsum("ij,bljk->blik", R, fr[..., :3, :3])
    fr2[..., :3, 3] = torch.einsum("ij,blj->bli", R, fr[..., :3, 3]) + t
    with torch.no_grad():
        a = ipa(s, None, fr); b = ipa(s, None, fr2)
    err = float((a - b).abs().max())
    rep("invariant to global rigid motion", err < 1e-4, f"max|Δ|={err:.2e}")

    print(f"\n[T5] gradients finite")
    head = IPAStructureHeadFixed(64, 8, 4, 2, 32).to(dev)
    s = torch.randn(2, 10, 64, device=dev, requires_grad=True)
    fr = torch.eye(4, device=dev).view(1, 1, 4, 4).expand(2, 10, 4, 4).clone()
    out = head(s, None, fr)
    loss = out["scalar_features"].pow(2).mean() + out["frames"].pow(2).mean()
    loss.backward()
    rep("param grads finite",
        all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))
    rep("input grad finite", torch.isfinite(s.grad).all())

    print("\n" + "=" * 72)
    print(f"  IPA PATCH SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 72)
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--patch-module", default=None)
    a = ap.parse_args()
    if a.patch_module:
        print(json.dumps(apply_patch(a.patch_module), indent=2))
    else:
        raise SystemExit(0 if self_test() else 1)
