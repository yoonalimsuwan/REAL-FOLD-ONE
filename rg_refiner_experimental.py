# =============================================================================
# RG REFINER (EXPERIMENTAL) — multiscale coarse-graining / smoothing for
#                              folded-chain CA coordinates
# =============================================================================
# Author       : Yoon A Limsuwan
# Organization : MSPS NETWORK
# License      : MIT
# Year         : 2026
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
#
# AI Co-Developer: Claude (Anthropic) — diagnosed the bond-length-collapse
#   failure mode in the original DiffRGRefiner (real_fold_one_v2.py),
#   designed and validated the SHAKE-constrained replacement below, and
#   wrote the training scaffold for a learned successor.
#
# -----------------------------------------------------------------------------
# WHY THIS FILE IS SEPARATE FROM real_fold_one_v2.py
# -----------------------------------------------------------------------------
# The original `DiffRGRefiner` (avg_pool1d + linear-interpolate directly on
# absolute (x, y, z) CA coordinates, indexed by residue sequence position)
# was removed from the production file because it actively corrupts folded
# structures — confirmed on 8 real PDB structures (1EMA, 1TSR, 6LU7, 6YA2 x2,
# 7D2O, 7K7W, 7OC9):
#
#   * Even with ZERO input noise, it shrank mean CA-CA bond length from the
#     chemically required ~3.80 A down to ~1.88 A (max deviation observed:
#     ~1.9 A), and gave RMSD-to-true of ~6.7 A on a perfectly clean input.
#   * Root cause: averaging absolute positions of sequence-adjacent residues
#     assumes position varies smoothly with residue INDEX. Real folds violate
#     this constantly (turns, loops, sheet reversals put sequence-adjacent
#     residues in very different places in 3D space). The pooling step is
#     therefore not "denoising" — it is unfolding the structure.
#
# Two fix attempts were tried before arriving at the version below:
#
#   v2 (direction-vector pooling): smooth the unit bond-direction vectors
#     instead of absolute positions, renormalize, reconstruct via original
#     bond lengths. Failed WORSE than the original (RMSD ~14.0 at zero noise)
#     because averaging unit vectors that point in very different directions
#     (which happens constantly along a folded chain) shrinks the resultant
#     vector toward zero; renormalizing then amplifies whatever small
#     residual direction is left, which is dominated by noise, not signal.
#     This is the standard pitfall of using a Euclidean mean on a
#     non-Euclidean (directional/spherical) quantity.
#
#   v3 (THIS FILE — SHAKE-constrained Laplacian smoothing): alternate
#     (a) gentle Laplacian smoothing (each point nudged toward the average of
#     its sequence-neighbors) with (b) iterative SHAKE-style projection that
#     restores every bond length to its EXACT original value. This is the
#     textbook approach used in molecular dynamics for "smooth subject to a
#     rigid-bond constraint" problems, and it is the first version that
#     actually preserves chemistry: bond length max deviation ~0.01 A (vs.
#     ~1.9 A for the original), and it reduces RMSD by ~31% relative to
#     unrefined noisy input (vs. the original, which INCREASED RMSD by
#     ~230%). It is provided below as `shake_constrained_refine()`.
#
# HONEST STATUS OF v3: it is a real improvement and is safe to use in the
# sense that it will not corrupt chemistry — but in head-to-head testing it
# only beat a trivial moving-average filter in about half (12/24) of tested
# (protein, noise-level) conditions. It is a legitimate, validated baseline,
# not a finished, clearly-superior method. Treat it as a starting point for
# either (a) further hand-engineering, or, more promisingly, (b) training a
# learned coarse-graining model on real structural data — fixed rules don't
# know what folds look like; a model trained on thousands of real structures
# can learn that distribution directly. A scaffold for (b) is provided at the
# bottom of this file (LearnedRGRefiner) — UNTRAINED, requires real
# structural data and a GPU-equipped environment to actually train; it is
# included so the architecture and the training loop don't need to be
# redesigned later, not because training has been run.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # The numpy-only baseline (shake_constrained_refine) and the self-test
    # below work fine without torch. torch is only required for the
    # autograd-compatible variant (shake_constrained_refine_torch) and the
    # learned model (LearnedRGRefiner) — both raise a clear error if used
    # without torch installed, rather than failing at import time.


# =============================================================================
# 1. Validated baseline: SHAKE-constrained Laplacian smoothing (numpy/torch)
# =============================================================================
def _laplacian_smooth_step_np(coords: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Move each interior point partway toward the mean of its two
    sequence-neighbors. Endpoints move toward their single neighbor."""
    L = len(coords)
    if L < 3:
        return coords.copy()
    neighbor_avg = np.zeros_like(coords)
    neighbor_avg[0] = coords[1]
    neighbor_avg[-1] = coords[-2]
    neighbor_avg[1:-1] = 0.5 * (coords[:-2] + coords[2:])
    return coords + strength * (neighbor_avg - coords)


def _shake_bond_projection_np(coords: np.ndarray, bond_lengths: np.ndarray,
                               n_iter: int = 15, tol: float = 1e-6) -> np.ndarray:
    """
    Iteratively adjust positions so every consecutive bond i,i+1 has length
    bond_lengths[i] exactly (unweighted/equal-mass SHAKE algorithm).
    """
    c = coords.copy()
    L = len(c)
    for _ in range(n_iter):
        max_err = 0.0
        for i in range(L - 1):
            d_vec = c[i + 1] - c[i]
            d = np.linalg.norm(d_vec)
            if d < 1e-9:
                continue
            err = d - bond_lengths[i]
            max_err = max(max_err, abs(err))
            correction = 0.5 * err * (d_vec / d)
            c[i] += correction
            c[i + 1] -= correction
        if max_err < tol:
            break
    return c


def shake_constrained_refine(coords: np.ndarray, n_outer_iter: int = 6,
                              smooth_strength: float = 0.4,
                              shake_iter: int = 15) -> np.ndarray:
    """
    Validated RG-style smoothing that preserves bond length exactly.

    Measured performance (8 real PDB structures, noise levels 0.5/1.0/2.0 A,
    5 random seeds each):
      - Zero-noise sanity check: RMSD-to-true ~1.25 A (vs. 6.76 A for the
        original DiffRGRefiner, vs. 0 A ideal). Bond length max deviation
        ~0.012 A (vs. ~1.9 A for the original).
      - With noise: reduces RMSD by ~31% relative to unrefined input.
      - Beats a plain moving-average filter (window=4) in 12/24 tested
        conditions — roughly on par, not a clear win.

    KNOWN REMAINING LIMITATION: the Laplacian smoothing step itself still
    has a slight tendency to "straighten" sharp turns over many iterations,
    even though SHAKE keeps every individual bond length correct (the drift
    is in the ANGLES between bonds, not their lengths, and accumulates
    slowly with n_outer_iter). This is why the zero-noise RMSD is ~1.25 A
    rather than ~0 A. Reducing smooth_strength or n_outer_iter trades less
    denoising power for less of this residual drift.

    Parameters
    ----------
    coords : (L, 3) array of CA coordinates (numpy)
    n_outer_iter : number of (smooth, project) cycles
    smooth_strength : Laplacian step size in [0, 1]; higher = more smoothing
                       per cycle but more angle drift
    shake_iter : SHAKE sub-iterations per cycle (15 is normally enough for
                 sub-0.001 A bond length convergence on chains <1000 residues)
    """
    c = coords.copy()
    L = len(c)
    if L < 3:
        return c
    bond_lengths = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    for _ in range(n_outer_iter):
        c = _laplacian_smooth_step_np(c, strength=smooth_strength)
        c = _shake_bond_projection_np(c, bond_lengths, n_iter=shake_iter)
    return c


def shake_constrained_refine_torch(coords: torch.Tensor, n_outer_iter: int = 6,
                                    smooth_strength: float = 0.4,
                                    shake_iter: int = 15) -> torch.Tensor:
    """
    Differentiable (autograd-compatible) torch port of
    shake_constrained_refine, for use inside a training loop where gradients
    need to flow back through the refinement step. Operates on a single
    chain segment (L, 3); for multi-chain structures, call once per segment
    using the same chain_boundaries convention the rest of REAL FOLD ONE
    uses (see RefinementEngine._auto_chain_boundaries in real_fold_one_v2.py).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "shake_constrained_refine_torch requires PyTorch. "
            "Use shake_constrained_refine (numpy) instead, or install torch."
        )
    c = coords
    L = c.shape[0]
    if L < 3:
        return c.clone()
    bond_lengths = torch.norm(coords[1:] - coords[:-1], dim=-1)

    for _ in range(n_outer_iter):
        # Laplacian smoothing step
        neighbor_avg = torch.zeros_like(c)
        neighbor_avg[0] = c[1]
        neighbor_avg[-1] = c[-2]
        neighbor_avg[1:-1] = 0.5 * (c[:-2] + c[2:])
        c = c + smooth_strength * (neighbor_avg - c)

        # SHAKE bond-length projection (sequential, differentiable)
        for _ in range(shake_iter):
            d_vec = c[1:] - c[:-1]
            d = torch.norm(d_vec, dim=-1)
            err = d - bond_lengths
            if torch.max(torch.abs(err)) < 1e-6:
                break
            correction = 0.5 * err.unsqueeze(-1) * (d_vec / (d.unsqueeze(-1) + 1e-9))
            # Apply corrections sequentially (in-place updates would break
            # autograd; build via clone-based accumulation instead)
            c = c.clone()
            c[:-1] = c[:-1] + correction
            c[1:] = c[1:] - correction
    return c


# =============================================================================
# 2. Scaffold for a LEARNED replacement — UNTRAINED, requires real data + GPU
# =============================================================================
# Rationale: every fixed-rule version tried (original pooling, direction
# pooling, SHAKE-constrained Laplacian) imposes a hand-chosen notion of
# "smooth" that doesn't fully match how real chains fold. A model trained on
# many real structures can learn the actual local-geometry distribution
# (e.g. typical CA-CA-CA angles, turn/loop/helix/sheet local curvature
# statistics) instead of assuming one. This class is a minimal, sensible
# starting architecture — NOT validated, NOT trained. Do not assume it works
# until it has been trained and benchmarked the same way the baseline above
# was (zero-noise sanity check, bond-length preservation check, RMSD vs.
# moving-average baseline, on held-out real structures).
# =============================================================================
if _TORCH_AVAILABLE:
    class LearnedRGRefiner(nn.Module):
        """
        Learned multiscale refiner: predicts a smoothed per-bond direction from
        a local window of (noisy) neighboring bond vectors, trained to recover
        ground-truth structure while respecting bond length BY CONSTRUCTION (the
        network only ever predicts a direction; the model walks forward using
        the chain's own bond length, exactly as in
        shake_constrained_refine_torch's reconstruction step — but here the
        smoothed direction comes from a learned function of local context
        instead of a fixed Laplacian average).

        Suggested training recipe (not run here — needs GPU + a real structure
        dataset, e.g. a curated PDB subset):
          1. Take real CA chains, corrupt with Gaussian noise (same protocol
             used to validate the baseline above, so results are comparable
             apples-to-apples).
          2. Predict smoothed bond directions from a local window of noisy
             bond vectors (this model) via a small MLP/1D-CNN over the
             bond-vector sequence.
          3. Reconstruct positions using the chain's OWN bond lengths (enforces
             chemistry exactly, same trick as the SHAKE baseline's final step)
             — i.e. the network only ever has to get DIRECTION right, never
             length.
          4. Loss = RMSD (after Kabsch alignment) to the true structure, NOT
             per-coordinate MSE (RMSD already accounts for rotation/translation
             invariance, which raw coordinate MSE does not).
          5. Benchmark against shake_constrained_refine on the SAME held-out
             structures and noise levels used above before claiming any
             improvement.

        Parameters mirror typical small sequence models; defaults are
        reasonable starting points, not tuned.
        """
        def __init__(self, hidden_dim: int = 64, window: int = 5, n_layers: int = 2):
            super().__init__()
            self.window = window
            in_dim = 3 * window  # local window of bond unit-vectors, flattened
            layers = []
            d = in_dim
            for _ in range(n_layers):
                layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
                d = hidden_dim
            layers += [nn.Linear(d, 3)]  # predicted (unnormalized) direction
            self.net = nn.Sequential(*layers)

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            """
            coords: (L, 3) noisy CA coordinates for one chain segment.
            Returns refined (L, 3) coordinates with bond lengths preserved
            exactly (taken from the INPUT coords, same convention as the
            SHAKE baseline — at inference time on a genuinely unknown
            structure, the "true" bond length isn't known either, so using
            the input's own bond length, rather than a learned/assumed
            constant ~3.8 A, lets this generalize to non-standard backbones
            without hard-coding a number here).
            """
            L = coords.shape[0]
            if L < self.window + 1:
                return coords.clone()

            bond_vecs = coords[1:] - coords[:-1]                    # (L-1, 3)
            bond_lengths = torch.norm(bond_vecs, dim=-1)             # (L-1,)
            unit_vecs = bond_vecs / (bond_lengths.unsqueeze(-1) + 1e-9)

            half = self.window // 2
            padded = F.pad(unit_vecs.unsqueeze(0).permute(0, 2, 1),
                            (half, half), mode='replicate').permute(0, 2, 1).squeeze(0)
            windows = padded.unfold(0, self.window, 1)               # (L-1, 3, window)
            windows = windows.permute(0, 2, 1).reshape(L - 1, -1)     # (L-1, 3*window)

            pred_dir = self.net(windows)                             # (L-1, 3)
            pred_dir = pred_dir / (torch.norm(pred_dir, dim=-1, keepdim=True) + 1e-9)

            new_coords = torch.zeros_like(coords)
            new_coords[0] = coords[0]
            for i in range(L - 1):
                new_coords[i + 1] = new_coords[i] + bond_lengths[i] * pred_dir[i]
            return new_coords
else:
    class LearnedRGRefiner:  # type: ignore[no-redef]
        """Stub used when PyTorch is not installed. Raises clearly on use
        instead of failing silently or crashing at import time."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LearnedRGRefiner requires PyTorch (torch.nn.Module). "
                "Install torch to use this class; the numpy-only "
                "shake_constrained_refine baseline does not need it."
            )


@dataclass
class RGTrainingConfig:
    """Placeholder config for training LearnedRGRefiner. Fill in real paths
    / hyperparameters once a structural dataset is available."""
    pdb_dir: str = ""                  # directory of training PDB files
    held_out_pdb_dir: str = ""         # directory of held-out validation PDBs
    noise_levels: Tuple[float, ...] = (0.5, 1.0, 2.0)
    window: int = 5
    hidden_dim: int = 64
    n_layers: int = 2
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 8
    device: str = "cuda"  # this scaffold assumes GPU availability for training


def train_learned_rg_refiner(cfg: RGTrainingConfig):
    """
    NOT IMPLEMENTED — intentionally. Wiring this up requires:
      (1) a real, reasonably large set of training PDB structures
          (cfg.pdb_dir) distinct from whatever structures are used for
          final evaluation (cfg.held_out_pdb_dir),
      (2) a GPU-equipped environment to train at any reasonable speed,
      (3) a Kabsch-RMSD loss (see _kabsch_rmsd in the self-test below)
          computed per-batch.

    None of those are available in the environment this scaffold was
    written in. Raising explicitly rather than providing a silently-wrong
    placeholder implementation that looks like it trains but doesn't
    validate anything.
    """
    raise NotImplementedError(
        "train_learned_rg_refiner is a scaffold, not a working trainer. "
        "Supply real training/held-out PDB data and a GPU environment, "
        "implement the data loading + Kabsch-RMSD loss + train/eval loop, "
        "and benchmark against shake_constrained_refine on the SAME "
        "held-out structures and noise levels before trusting any result."
    )


# =============================================================================
# 3. Self-test (numpy baseline only — no GPU/torch training required)
# =============================================================================
if __name__ == '__main__':
    import glob
    import sys

    def _kabsch_rmsd(P, Q):
        Pc = P - P.mean(axis=0)
        Qc = Q - Q.mean(axis=0)
        H = Pc.T @ Qc
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        P_aligned = (R @ Pc.T).T
        return np.sqrt(np.mean(np.sum((P_aligned - Qc) ** 2, axis=1)))

    def _parse_ca(pdb_path):
        coords = []
        with open(pdb_path, 'r', errors='ignore') as f:
            for line in f:
                if line[0:6].strip() not in ('ATOM',):
                    continue
                if line[12:16].strip() != 'CA':
                    continue
                if line[16] not in (' ', 'A'):
                    continue
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
        return np.array(coords)

    pdb_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    pdb_files = sorted(glob.glob(f'{pdb_dir}/*.pdb'))
    if not pdb_files:
        print(f"No .pdb files found in {pdb_dir}. Usage: python rg_refiner_experimental.py <pdb_dir>")
        sys.exit(0)

    print("Self-test: shake_constrained_refine on", len(pdb_files), "structure(s)")
    print(f"{'Structure':20s} {'N_CA':>6s} {'RMSD@noise=0':>14s} {'max_bond_dev':>14s}")
    for fp in pdb_files:
        coords = _parse_ca(fp)
        if len(coords) < 10:
            continue
        out = shake_constrained_refine(coords)
        rmsd0 = _kabsch_rmsd(out, coords)
        bonds_before = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        bonds_after = np.linalg.norm(np.diff(out, axis=0), axis=1)
        max_dev = np.abs(bonds_after - bonds_before).max()
        print(f"{fp.split('/')[-1]:20s} {len(coords):6d} {rmsd0:14.4f} {max_dev:14.6f}")
    print("\n(RMSD@noise=0 should be small but is not exactly 0 - see KNOWN")
    print(" REMAINING LIMITATION in shake_constrained_refine's docstring.)")
