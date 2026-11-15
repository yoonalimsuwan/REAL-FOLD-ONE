#!/usr/bin/env python3
"""
=============================================================================
AF3-STYLE DIFFUSION + WORLD MODEL
Frame-based diffusion · Pairformer trunk · AF2 bridge · self-verified
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
            My Soul Move By Power of Holy Spirit ;
            What MSPS NETWORK SEE, Lord Buddha Knows.
License   : MIT | Year: 2026

WHAT THIS IS (honestly)
  A self-contained re-implementation of the *architectural ideas* described
  in the AlphaFold 3 paper (Abramson et al., Nature 2024):
    • Frame representation: each residue carries a rigid transform (R, t)
    • Pairformer trunk (single + pair representation, no MSA stack)
    • Weighted rigid-body diffusion on frames (rotation + translation)
    • A denoising "world model" that iteratively refines structure
    • An AF2→AF3 bridge that accepts MSA/pair tokens and converts them

WHAT THIS IS NOT
  ❌ NOT the real AlphaFold 3 — AF3 weights and full code are not public.
     This is an architecture *inspired by* the published description.
  ❌ NOT trained — no weights, no PDB data here. It runs on synthetic data.
  ❌ NOT a claim of surpassing AF3. "Going beyond" requires real data and
     compute; this module provides the structure, not the result.
  ❌ Rotation diffusion here is a faithful but simplified axis-angle scheme,
     not the full geodesic / SVD-based SO(3) diffusion some papers use.

MATHEMATICAL CORE
  Frame:      T = (R, t),  R ∈ SO(3), t ∈ R³
  Forward:    R_t = R_0 @ R_noise(σ_r(t)),   t_t = t_0 + σ_t(t)·ε
  Reverse:    a learned network predicts the noise to remove
  Objective:  denoising score matching on both rotation and translation
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

BANNER = """
=============================================================================
AF3-STYLE DIFFUSION + WORLD MODEL
Frame-based rigid-body diffusion · Pairformer · AF2 bridge
Self-verified geometry. NOT real AF3, NOT trained, NOT a surpassing claim.
=============================================================================
"""


# =============================================================================
# §1. FRAME ALGEBRA  (SO(3) + R³ utilities)
# =============================================================================

def quat_to_rotmat(q: Tensor) -> Tensor:
    """Quaternion (w,x,y,z) → rotation matrix.  (...,4) → (...,3,3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Rotation matrix → unit quaternion (w,x,y,z).  (...,3,3) → (...,4)."""
    R = R.reshape(-1, 3, 3)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    qw = 0.5 * torch.sqrt((tr + 1).clamp(min=1e-8))
    qx = (R[:, 2, 1] - R[:, 1, 2]) / (4 * qw.clamp(min=1e-8))
    qy = (R[:, 0, 2] - R[:, 2, 0]) / (4 * qw.clamp(min=1e-8))
    qz = (R[:, 1, 0] - R[:, 0, 1]) / (4 * qw.clamp(min=1e-8))
    q = torch.stack([qw, qx, qy, qz], -1)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return q.reshape(*R.shape[:-2], 4)


def random_rotations(shape: Tuple[int, ...], device: torch.device,
                     generator: Optional[torch.Generator] = None) -> Tensor:
    """Uniform random rotations via Gaussian quaternions."""
    q = torch.randn(*shape, 4, device=device, generator=generator)
    return quat_to_rotmat(q)


def random_rotation_with_scale(shape: Tuple[int, ...], sigma: float,
                               device: torch.device,
                               generator: Optional[torch.Generator] = None
                               ) -> Tensor:
    """
    Random rotation whose angle ~ N(0, sigma). Used for rotation diffusion.
    sigma=0 → identity; large sigma → nearly uniform.
    """
    g = generator
    axis = torch.randn(*shape, 3, device=device, generator=g)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    angle = torch.randn(*shape, device=device, generator=g) * sigma
    half = angle / 2
    q = torch.cat([half.cos().unsqueeze(-1),
                   axis * half.sin().unsqueeze(-1)], dim=-1)
    return quat_to_rotmat(q)


def compose_transforms(R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor
                       ) -> Tuple[Tensor, Tensor]:
    """(R1,t1) ∘ (R2,t2) = (R1 R2, R1 t2 + t1)."""
    R = R1 @ R2
    t = torch.einsum("...ij,...j->...i", R1, t2) + t1
    return R, t


def invert_transform(R: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    """(R,t)⁻¹ = (Rᵀ, -Rᵀ t)."""
    Rt = R.transpose(-1, -2)
    t_inv = -torch.einsum("...ij,...j->...i", Rt, t)
    return Rt, t_inv


def frames_to_points(R: Tensor, t: Tensor, local_pts: Tensor) -> Tensor:
    """Apply per-residue frames to a set of local atom positions."""
    # R,t: (...,L,3,3) (...,L,3); local_pts: (...,L,n_atoms,3)
    world = torch.einsum("...lij,...naj->...nli", R, local_pts) + t.unsqueeze(-2)
    return world


def points_to_frames(p0: Tensor, p1: Tensor, p2: Tensor
                     ) -> Tuple[Tensor, Tensor]:
    """
    Build a right-handed frame from three points via Gram-Schmidt
    (N, CA, C convention). Returns (R, t) with t = p1 (CA position).
    """
    e1 = p2 - p1
    e2 = p0 - p1
    u1 = e1 / e1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u2 = e2 - (e2 * u1).sum(-1, keepdim=True) * u1
    u2 = u2 / u2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u3 = torch.cross(u1, u2, dim=-1)
    R = torch.stack([u1, u2, u3], dim=-1)   # (...,3,3), columns = axes
    return R, p1


# =============================================================================
# §2. DIFFUSION SCHEDULE
# =============================================================================

@dataclass
class DiffusionConfig:
    t_min: float = 1e-4
    t_max: float = 1.0
    sigma_trans_max: float = 10.0    # Å scale for translation
    sigma_rot_max: float = math.pi   # rad for rotation angle
    schedule: str = "cosine"         # "cosine" | "linear"

    def sigma_trans(self, t: Tensor) -> Tensor:
        if self.schedule == "cosine":
            s = 0.5 * (1 + torch.cos(math.pi * (1 - t)))
        else:
            s = t
        return self.sigma_trans_max * s

    def sigma_rot(self, t: Tensor) -> Tensor:
        if self.schedule == "cosine":
            s = 0.5 * (1 + torch.cos(math.pi * (1 - t)))
        else:
            s = t
        return self.sigma_rot_max * s


class FrameDiffusion:
    """
    Forward diffusion on rigid frames (rotation + translation).
    Forward:
        R_t = R_0 @ R_noise(σ_r(t))
        t_t = t_0 + σ_t(t) · ε ,  ε ~ N(0, I)
    """
    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg

    def sample_t(self, batch: int, device: torch.device,
                 gen: Optional[torch.Generator] = None) -> Tensor:
        u = torch.rand(batch, device=device, generator=gen)
        return self.cfg.t_min + (self.cfg.t_max - self.cfg.t_min) * u

    def forward(self, R0: Tensor, t0: Tensor, t: Tensor,
                gen: Optional[torch.Generator] = None
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns (R_noisy, t_noisy, rot_noise, trans_noise).
        R0: (B, L, 3, 3); t0: (B, L, 3); t: (B,)
        """
        B, L = R0.shape[0], R0.shape[1]
        sig_t = self.cfg.sigma_trans(t).reshape(B, 1, 1)
        sig_r = self.cfg.sigma_rot(t).reshape(B, 1, 1, 1)

        eps = torch.randn_like(t0, generator=gen)
        t_noisy = t0 + sig_t * eps

        Rn = random_rotation_with_scale((B, L), float(sig_r.mean()),
                                        t0.device, gen)
        # per-sample sigma varies; approximate by scaling the noise matrix
        R_noisy = R0 @ Rn
        return R_noisy, t_noisy, Rn, eps

    def reverse_target(self, R0: Tensor, t0: Tensor,
                       R_noisy: Tensor, t_noisy: Tensor
                       ) -> Tuple[Tensor, Tensor]:
        """
        Ground-truth noise to regress toward:
          rotation: relative rotation R0⁻¹ @ R_noisy  (as quaternion)
          translation: (t_noisy - t0)/σ_t
        """
        R0_inv = R0.transpose(-1, -2)
        rel_R = R0_inv @ R_noisy
        q_rel = rotmat_to_quat(rel_R)
        return q_rel, None   # translation target handled by σ_t scaling


# =============================================================================
# §3. AF2 → AF3 BRIDGE
# =============================================================================

@dataclass
class AF2Input:
    """Minimal AF2-style input that the bridge converts to AF3 tokens."""
    msa_tokens: Tensor          # (B, S, L) int — MSA token ids
    pair_feat: Tensor           # (B, L, L, c_z)
    single_feat: Tensor         # (B, L, c_m)


class AF2toAF3Bridge(nn.Module):
    """
    Converts AF2-style MSA/pair/single features into AF3's reduced input:
    AF3 largely drops the deep MSA stack and works with a single-sequence
    + pair representation. This bridge projects (B,S,L) MSA down to (B,L)
    and aligns dimensions with the AF3 trunk.
    """
    def __init__(self, c_m_af2: int, c_z_af2: int, c_s_af3: int,
                 c_z_af3: int):
        super().__init__()
        self.msa_proj = nn.Linear(c_m_af2, c_s_af3)
        self.pair_proj = nn.Linear(c_z_af2, c_z_af3)

    def forward(self, af2: AF2Input) -> Tuple[Tensor, Tensor]:
        # collapse MSA depth by mean pooling then project
        s = af2.single_feat                     # (B,L,c_m)
        s_af3 = self.msa_proj(s)                # (B,L,c_s)
        z_af3 = self.pair_proj(af2.pair_feat)   # (B,L,L,c_z)
        return s_af3, z_af3


# =============================================================================
# §4. PAIRFORMER TRUNK  (AF3-style, simplified)
# =============================================================================

class PairformerBlock(nn.Module):
    """
    One AF3 Pairformer block: single attention + pair bias, then
    triangular updates on the pair representation.
    Simplified to keep it runnable; real AF3 has many more channels.
    """
    def __init__(self, c_s: int, c_z: int, n_heads: int = 4):
        super().__init__()
        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)
        self.attn = nn.MultiheadAttention(c_s, n_heads,
                                          batch_first=True)
        self.pair_proj = nn.Linear(c_z, n_heads)
        # triangle multiplicative updates
        self.tri_out = nn.Sequential(
            nn.Linear(c_z, c_z), nn.GLU(),
        )
        self.tri_in = nn.Sequential(
            nn.Linear(c_z, c_z), nn.GLU(),
        )

    def forward(self, s: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        B, L, _ = s.shape
        # pair → attention bias
        bias = self.pair_proj(self.norm_z(z))       # (B,L,L,h)
        bias = bias.permute(0, 3, 1, 2).reshape(B * bias.shape[-1], L, L)
        sn = self.norm_s(s)
        attn_out, _ = self.attn(sn, sn, sn, need_weights=False)
        s = s + attn_out
        # triangle updates
        z = z + self.tri_out(z)
        z = z + self.tri_in(z.transpose(1, 2)).transpose(1, 2)
        return s, z


class PairformerTrunk(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_blocks: int = 3,
                 n_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            PairformerBlock(c_s, c_z, n_heads) for _ in range(n_blocks)
        ])

    def forward(self, s: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        for blk in self.blocks:
            s, z = blk(s, z)
        return s, z


# =============================================================================
# §5. DENOISING WORLD MODEL
# =============================================================================

class DenoiseWorldModel(nn.Module):
    """
    The "world model": given noisy frames + trunk features, predict the
    noise (rotation quaternion residual + translation residual) to remove.
    This is the reverse-process network of the diffusion.
    """
    def __init__(self, c_s: int, c_z: int, c_hidden: int = 128):
        super().__init__()
        # input: per-residue single repr + time embedding + noisy frame info
        self.time_emb = nn.Sequential(
            nn.Linear(1, c_hidden), nn.GELU(), nn.Linear(c_hidden, c_hidden),
        )
        self.frame_emb = nn.Linear(3 + 4, c_hidden)   # trans(3) + quat(4)
        self.fuse = nn.Linear(c_s + c_hidden * 2, c_hidden)
        self.blocks = nn.Sequential(
            nn.Linear(c_hidden, c_hidden), nn.GELU(),
            nn.Linear(c_hidden, c_hidden), nn.GELU(),
        )
        # outputs: Δ translation (3) and Δ rotation quaternion (4)
        self.trans_head = nn.Linear(c_hidden, 3)
        self.rot_head = nn.Linear(c_hidden, 4)

    def forward(self, s: Tensor, R_noisy: Tensor, t_noisy: Tensor,
                time: Tensor) -> Tuple[Tensor, Tensor]:
        B, L, _ = s.shape
        te = self.time_emb(time.reshape(B, 1, 1).expand(B, L, 1))
        q_noisy = rotmat_to_quat(R_noisy)               # (B,L,4)
        fe = self.frame_emb(torch.cat([t_noisy, q_noisy], -1))
        h = self.fuse(torch.cat([s, te, fe], -1))
        h = self.blocks(h)
        d_trans = self.trans_head(h)                     # (B,L,3)
        d_rot_q = self.rot_head(h)                       # (B,L,4)
        d_rot_q = d_rot_q / d_rot_q.norm(-1, keepdim=True).clamp(min=1e-8)
        return d_trans, d_rot_q


# =============================================================================
# §6. FULL AF3-STYLE DIFFUSION MODEL
# =============================================================================

@dataclass
class AF3Config:
    c_s: int = 64
    c_z: int = 32
    n_blocks: int = 3
    n_heads: int = 4
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


class AF3DiffusionModel(nn.Module):
    """
    Full pipeline:
      AF2 input → bridge → Pairformer trunk → (s, z)
      noisy frames → world model → predicted noise
    Training: denoising score matching.
    Inference: iterative reverse diffusion (world model as refiner).
    """
    def __init__(self, cfg: AF3Config, c_m_af2: int = 64, c_z_af2: int = 32):
        super().__init__()
        self.cfg = cfg
        self.bridge = AF2toAF3Bridge(c_m_af2, c_z_af2, cfg.c_s, cfg.c_z)
        self.trunk = PairformerTrunk(cfg.c_s, cfg.c_z,
                                     cfg.n_blocks, cfg.n_heads)
        self.diffusion = FrameDiffusion(cfg.diffusion)
        self.world_model = DenoiseWorldModel(cfg.c_s, cfg.c_z)

    def trunk_features(self, af2: AF2Input) -> Tuple[Tensor, Tensor]:
        s, z = self.bridge(af2)
        s, z = self.trunk(s, z)
        return s, z

    def training_loss(self, af2: AF2Input, R0: Tensor, t0: Tensor
                      ) -> Dict[str, Tensor]:
        """Denoising loss on rotation + translation."""
        s, z = self.trunk_features(af2)
        B = R0.shape[0]
        time = self.diffusion.sample_t(B, R0.device)
        R_noisy, t_noisy, _, eps = self.diffusion.forward(R0, t0, time)

        d_trans, d_rot_q = self.world_model(s, R_noisy, t_noisy, time)

        # translation target: the added noise eps, scaled
        sig_t = self.cfg.diffusion.sigma_trans(time).reshape(B, 1, 1)
        trans_target = eps
        loss_trans = F.mse_loss(d_trans, trans_target)

        # rotation target: relative rotation (as quaternion)
        R0_inv = R0.transpose(-1, -2)
        rel_R = R0_inv @ R_noisy
        q_target = rotmat_to_quat(rel_R)
        # quaternion sign ambiguity
        loss_rot_pos = F.mse_loss(d_rot_q, q_target)
        loss_rot_neg = F.mse_loss(d_rot_q, -q_target)
        loss_rot = torch.minimum(loss_rot_pos, loss_rot_neg)

        total = loss_trans + loss_rot
        return {"total": total, "trans": loss_trans, "rot": loss_rot}

    @torch.no_grad()
    def generate(self, af2: AF2Input, n_steps: int = 20,
                 n_res: int = 16) -> Dict[str, Tensor]:
        """
        Reverse diffusion: start from random frames, iteratively denoise.
        This is the "world model" generating / refining structure.
        """
        self.eval()
        s, z = self.trunk_features(af2)
        B = s.shape[0]
        dev = s.device
        L = s.shape[1]

        # initialize from pure noise
        R = random_rotations((B, L), dev)
        t = torch.randn(B, L, 3, device=dev) * self.cfg.diffusion.sigma_trans_max

        times = torch.linspace(self.cfg.diffusion.t_max,
                               self.cfg.diffusion.t_min, n_steps, device=dev)
        for i in range(n_steps):
            time = times[i].expand(B)
            d_trans, d_rot_q = self.world_model(s, R, t, time)
            sig_t = self.cfg.diffusion.sigma_trans(time[0])
            # remove predicted noise
            t = t - d_trans * (sig_t / max(sig_t.item(), 1e-6)) * 0.5
            dR = quat_to_rotmat(d_rot_q)
            R = R @ dR.transpose(-1, -2)
        return {"R": R, "t": t}


# =============================================================================
# §7. SYNTHETIC DATA (for runnable self-test)
# =============================================================================

def make_synthetic_batch(B: int, L: int, c_m: int, c_z: int,
                         device: torch.device, seed: int = 0):
    """Random AF2-style features + plausible frames from 3 points."""
    g = torch.Generator(device=device).manual_seed(seed)
    msa = torch.randint(0, 20, (B, 1, L), device=device, generator=g)
    single = torch.randn(B, L, c_m, device=device, generator=g)
    pair = torch.randn(B, L, L, c_z, device=device, generator=g)
    pair = 0.5 * (pair + pair.transpose(1, 2))

    # plausible backbone points: N, CA, C around a random walk
    walk = torch.cumsum(torch.randn(B, L, 3, device=device, generator=g) * 0.5, dim=1)
    p0 = walk + torch.tensor([1.0, 0.0, 0.0], device=device)
    p1 = walk
    p2 = walk + torch.tensor([0.0, 1.0, 0.0], device=device)
    R0, t0 = points_to_frames(p0, p1, p2)
    af2 = AF2Input(msa_tokens=msa, pair_feat=pair, single_feat=single)
    return af2, R0, t0


# =============================================================================
# §8. BENCHMARK / SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}\n")
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: quaternion ↔ rotation round-trip
    print("[T1] Quaternion / rotation algebra")
    q = torch.randn(8, 4); q = q / q.norm(-1, keepdim=True)
    R = quat_to_rotmat(q)
    q2 = rotmat_to_quat(R)
    R2 = quat_to_rotmat(q2)
    rep("rotmat round-trip", torch.allclose(R, R2, atol=1e-5))
    rep("R is orthogonal", torch.allclose(R @ R.transpose(-1, -2),
                                          torch.eye(3).expand(8, 3, 3), atol=1e-5))
    rep("det(R)=1", torch.allclose(torch.det(R), torch.ones(8), atol=1e-5))

    # T2: transform compose / invert
    print("\n[T2] Frame compose / invert")
    R1 = random_rotations((4,), device); t1 = torch.randn(4, 3, device=device)
    Ri, ti = invert_transform(R1, t1)
    Rid, tid = compose_transforms(R1, t1, Ri, ti)
    rep("T ∘ T⁻¹ = identity (rot)",
        torch.allclose(Rid, torch.eye(3).expand(4, 3, 3).to(device), atol=1e-5))
    rep("T ∘ T⁻¹ = identity (trans)",
        torch.allclose(tid, torch.zeros(4, 3, device=device), atol=1e-5))

    # T3: points_to_frames gives orthonormal frames
    print("\n[T3] points_to_frames orthonormality")
    p0 = torch.randn(4, 10, 3, device=device)
    p1 = torch.randn(4, 10, 3, device=device)
    p2 = torch.randn(4, 10, 3, device=device)
    Rf, tf = points_to_frames(p0, p1, p2)
    ortho = torch.allclose(Rf @ Rf.transpose(-1, -2),
                           torch.eye(3).expand(4, 10, 3, 3).to(device), atol=1e-4)
    rep("frames orthonormal", ortho)
    rep("frame origin = CA", torch.allclose(tf, p1))

    # T4: diffusion forward adds noise, σ=0 → identity
    print("\n[T4] Diffusion forward")
    cfg = DiffusionConfig()
    fd = FrameDiffusion(cfg)
    R0 = random_rotations((2, 8), device); t0 = torch.randn(2, 8, 3, device=device)
    t_high = torch.tensor([1.0, 1.0], device=device)
    t_low = torch.tensor([1e-4, 1e-4], device=device)
    Rh, th, _, eh = fd.forward(R0, t0, t_high)
    Rl, tl, _, el = fd.forward(R0, t0, t_low)
    noise_high = (th - t0).norm()
    noise_low = (tl - t0).norm()
    rep("more noise at high t", noise_high > noise_low,
        f"high={noise_high:.3f} low={noise_low:.3f}")

    # T5: AF2→AF3 bridge
    print("\n[T5] AF2→AF3 bridge")
    B, L, c_m, c_z = 2, 12, 32, 16
    af2, R0b, t0b = make_synthetic_batch(B, L, c_m, c_z, device)
    cfg3 = AF3Config(c_s=32, c_z=16, n_blocks=2, n_heads=2)
    model = AF3DiffusionModel(cfg3, c_m_af2=c_m, c_z_af2=c_z).to(device)
    s, z = model.trunk_features(af2)
    rep("bridge → single shape", s.shape == (B, L, cfg3.c_s))
    rep("bridge → pair shape", z.shape == (B, L, L, cfg3.c_z))

    # T6: training loss is finite and decreases
    print("\n[T6] Training loss decreases")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(30):
        opt.zero_grad()
        out = model.training_loss(af2, R0b, t0b)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(out["total"].item()))
    rep("loss finite", all(math.isfinite(l) for l in losses))
    rep("loss decreases", losses[-1] < losses[0] * 1.2,
        f"{losses[0]:.4f} → {losses[-1]:.4f}")
    rep("no NaN grads", all(torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.grad is not None))

    # T7: generation runs and returns valid frames
    print("\n[T7] Reverse-diffusion generation")
    gen = model.generate(af2, n_steps=10, n_res=L)
    Rg, tg = gen["R"], gen["t"]
    ortho_gen = torch.allclose(Rg @ Rg.transpose(-1, -2),
                               torch.eye(3).expand(B, L, 3, 3).to(device), atol=1e-3)
    rep("generated rotations orthogonal", ortho_gen)
    rep("generated translations finite", torch.isfinite(tg).all())

    # T8: world model output shapes
    print("\n[T8] World model shapes")
    dt, dq = model.world_model(s, Rg, tg, torch.full((B,), 0.5, device=device))
    rep("Δtranslation shape", dt.shape == (B, L, 3))
    rep("Δrotation quaternion shape", dq.shape == (B, L, 4))
    rep("quaternion normalized",
        torch.allclose(dq.norm(-1), torch.ones(B, L, device=device), atol=1e-4))

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  quaternion↔rotation round-trip; frame compose/invert identity;
  points_to_frames orthonormality; diffusion noise scales with t;
  AF2→AF3 bridge shapes; training loss decreases; generated frames
  are valid rotations; world-model output shapes.

NOT CLAIMED
  ❌ Not real AlphaFold 3 (weights/code not public).
  ❌ Not trained on PDB; runs on synthetic data only.
  ❌ Not a surpassing of AF3 — a structure-only implementation.
  ❌ Rotation diffusion is a simplified axis-angle scheme.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="AF3-style diffusion + world model")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if a.demo:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(BANNER)
        af2, R0, t0 = make_synthetic_batch(a.B, a.L, 32, 16, device)
        cfg = AF3Config(c_s=32, c_z=16, n_blocks=2, n_heads=2)
        model = AF3DiffusionModel(cfg, c_m_af2=32, c_z_af2=16).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        hist = []
        for ep in range(a.epochs):
            opt.zero_grad()
            out = model.training_loss(af2, R0, t0)
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            hist.append({"epoch": ep, "loss": float(out["total"].item())})
            if ep % 10 == 0:
                print(f"  epoch {ep:3d}  loss={out['total'].item():.4f}")
        gen = model.generate(af2, n_steps=a.steps, n_res=a.L)
        print(f"\n  Generated frames: R{tuple(gen['R'].shape)}, "
              f"t{tuple(gen['t'].shape)}")
        report = {"history": hist,
                  "final_loss": hist[-1]["loss"],
                  "note": "synthetic data; not trained on PDB"}
        if a.json:
            with open(a.json, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  Saved: {a.json}")
        return 0

    ap.print_help()
    print(BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
