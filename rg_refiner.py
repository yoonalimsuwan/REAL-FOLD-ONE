# =============================================================================
# RG REFINER — multiscale coarse-graining / smoothing for folded-chain
#              CA coordinates, with a working trainer for the learned model
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
#   implemented the real training loop for LearnedRGRefiner (this file was
#   previously rg_refiner_experimental.py, where that loop was a scaffold
#   that raised NotImplementedError).
#
# -----------------------------------------------------------------------------
# HISTORY (kept for context — do not delete without checking real_fold_one_v2.py
# doesn't still reference the removed original)
# -----------------------------------------------------------------------------
# The original `DiffRGRefiner` (avg_pool1d + linear-interpolate directly on
# absolute (x, y, z) CA coordinates, indexed by residue sequence position)
# was removed from production because it corrupts folded structures —
# confirmed on 8 real PDB structures (1EMA, 1TSR, 6LU7, 6YA2 x2, 7D2O, 7K7W,
# 7OC9): even with ZERO input noise it shrank mean CA-CA bond length from
# ~3.80 A to ~1.88 A and gave RMSD-to-true ~6.7 A on clean input. Root cause:
# averaging absolute positions of sequence-adjacent residues assumes position
# varies smoothly with residue INDEX, which real folds violate constantly.
#
# A direction-vector-pooling fix (v2) failed WORSE (RMSD ~14.0 at zero noise)
# because averaging unit vectors pointing in very different directions
# shrinks the resultant toward zero, and renormalizing amplifies noise.
#
# `shake_constrained_refine` (v3, below) alternates Laplacian smoothing with
# SHAKE-style bond-length projection. It preserves bond length almost exactly
# (max deviation ~0.01 A vs ~1.9 A for the original) and reduces RMSD ~31%
# relative to unrefined noisy input. It beat a plain moving-average filter in
# only 12/24 tested (protein, noise-level) conditions — a legitimate,
# validated baseline, not a clearly-superior method. It remains here as the
# benchmark that `LearnedRGRefiner` must beat before being trusted.
#
# CHANGE IN THIS FILE: `shake_constrained_refine_torch` (a differentiable
# torch port of the baseline) was removed. It was never on the path from
# noisy input to a trained model — `LearnedRGRefiner` predicts directions
# with its own network and reconstructs bond lengths from the INPUT chain
# directly, so a separately differentiable SHAKE projector added nothing to
# training and did not affect accuracy. If a future experiment wants to use
# SHAKE-projected coordinates as a network input feature or as a loss term,
# re-add it then, with a stated reason.
# =============================================================================

from __future__ import annotations

import glob
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # The numpy-only baseline (shake_constrained_refine) works fine without
    # torch. torch is required for LearnedRGRefiner and
    # train_learned_rg_refiner — both raise a clear error if used without
    # torch installed, rather than failing at import time.


# =============================================================================
# 1. Validated baseline: SHAKE-constrained Laplacian smoothing (numpy)
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
      - Zero-noise sanity check: RMSD-to-true ~1.25 A. Bond length max
        deviation ~0.012 A.
      - With noise: reduces RMSD by ~31% relative to unrefined input.
      - Beats a plain moving-average filter (window=4) in 12/24 tested
        conditions — roughly on par, not a clear win.

    This is the benchmark `LearnedRGRefiner` is trained and evaluated
    against in `train_learned_rg_refiner` below — a training run whose
    validation RMSD doesn't beat this baseline should not be reported as an
    improvement.

    Parameters
    ----------
    coords : (L, 3) array of CA coordinates (numpy)
    n_outer_iter : number of (smooth, project) cycles
    smooth_strength : Laplacian step size in [0, 1]
    shake_iter : SHAKE sub-iterations per cycle
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


# =============================================================================
# 2. Shared geometry / IO helpers (used by both the baseline eval and the
#    trainer, so there is exactly one implementation of each)
# =============================================================================
def _kabsch_rmsd_np(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between P and Q after optimal rigid alignment of P onto Q."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_aligned = (R @ Pc.T).T
    return float(np.sqrt(np.mean(np.sum((P_aligned - Qc) ** 2, axis=1))))


def _parse_ca(pdb_path: str) -> np.ndarray:
    """Parse CA atom coordinates (altloc '' or 'A' only) from a PDB file."""
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
    return np.array(coords, dtype=np.float64)


def _discover_pdb_files(pdb_dir: str) -> List[str]:
    return sorted(glob.glob(f'{pdb_dir}/*.pdb'))


def _load_chains(pdb_dir: str, min_len: int = 10) -> List[np.ndarray]:
    """Load all CA chains from every .pdb file in pdb_dir, dropping chains
    shorter than min_len (too short to define a meaningful local window)."""
    chains = []
    for fp in _discover_pdb_files(pdb_dir):
        coords = _parse_ca(fp)
        if len(coords) >= min_len:
            chains.append(coords.astype(np.float32))
    return chains


# =============================================================================
# 3. LearnedRGRefiner — trainable model
# =============================================================================
if _TORCH_AVAILABLE:
    class LearnedRGRefiner(nn.Module):
        """
        Learned multiscale refiner: predicts a smoothed per-bond direction
        from a local window of (noisy) neighboring bond vectors, trained to
        recover ground-truth structure while respecting bond length BY
        CONSTRUCTION — the network only ever predicts a direction; the chain
        is walked forward using its OWN input bond lengths, so bond length is
        exact at both train and inference time regardless of what the
        network learns.

        Training recipe (implemented in train_learned_rg_refiner below):
          1. Real CA chains, corrupted with Gaussian noise at the same levels
             used to validate shake_constrained_refine, for a fair
             comparison.
          2. Predict smoothed bond directions from a local window of noisy
             bond vectors via a small MLP.
          3. Reconstruct positions using the chain's own (noisy) bond
             lengths — the network only has to get direction right.
          4. Loss = RMSD after Kabsch alignment to the true structure (not
             raw coordinate MSE, which is not rotation/translation
             invariant).
          5. Every validation pass also runs shake_constrained_refine on the
             SAME held-out structures/noise levels, so the trained model's
             number is always reported next to the baseline it must beat.
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
            exactly (taken from the INPUT coords — at inference time on a
            genuinely unknown structure the "true" bond length isn't known
            either, so using the input's own bond length rather than a
            learned/assumed constant lets this generalize to non-standard
            backbones without hard-coding a number).
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

            # Build the walked chain without in-place ops on a tensor that
            # needs grad (in-place indexed writes on a leaf-derived tensor
            # break autograd); accumulate via torch.cat instead.
            steps = [coords[0].unsqueeze(0)]
            cur = coords[0]
            for i in range(L - 1):
                cur = cur + bond_lengths[i] * pred_dir[i]
                steps.append(cur.unsqueeze(0))
            return torch.cat(steps, dim=0)

    def _kabsch_rmsd_torch(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Differentiable RMSD between P and Q after optimal rigid alignment
        of P onto Q. Same convention as _kabsch_rmsd_np."""
        Pc = P - P.mean(dim=0, keepdim=True)
        Qc = Q - Q.mean(dim=0, keepdim=True)
        H = Pc.t() @ Qc
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(Vt.t() @ U.t()))
        D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
        R = Vt.t() @ D @ U.t()
        P_aligned = (R @ Pc.t()).t()
        return torch.sqrt(torch.mean(torch.sum((P_aligned - Qc) ** 2, dim=1)) + eps)

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


# =============================================================================
# 4. Real trainer
# =============================================================================
@dataclass
class RGTrainingConfig:
    """Config for training LearnedRGRefiner. pdb_dir and held_out_pdb_dir
    must point at DIFFERENT sets of real PDB structures — train_learned_rg_refiner
    raises if either is empty rather than silently training on nothing."""
    pdb_dir: str = ""                  # directory of training PDB files
    held_out_pdb_dir: str = ""         # directory of held-out validation PDBs
    noise_levels: Tuple[float, ...] = (0.5, 1.0, 2.0)
    min_chain_len: int = 10
    window: int = 5
    hidden_dim: int = 64
    n_layers: int = 2
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 8                # gradient-accumulation batch (chains
                                        # vary in length, so there is no
                                        # padded tensor batch — batch_size
                                        # structures are accumulated before
                                        # each optimizer.step())
    grad_clip: float = 1.0
    eval_interval: int = 5             # epochs between held-out evaluations
    save_path: str = ""                # if set, best checkpoint is saved here
    device: str = "cuda"               # falls back to cpu with a warning if
                                        # cuda isn't actually available
    seed: int = 0


def _evaluate(model, chains: List[np.ndarray], noise_levels: Tuple[float, ...],
              device, rng: np.random.Generator) -> Dict[str, float]:
    """Run model, the shake baseline, and raw noisy input over every
    (chain, noise level) pair in `chains`, returning mean RMSD-to-true for
    each — the three numbers a training run must be judged by together."""
    model_rmsds, base_rmsds, noisy_rmsds = [], [], []
    with torch.no_grad():
        for clean in chains:
            for sigma in noise_levels:
                noisy = clean + rng.normal(0, sigma, clean.shape).astype(np.float32)
                noisy_t = torch.tensor(noisy, dtype=torch.float32, device=device)
                refined = model(noisy_t).cpu().numpy()
                model_rmsds.append(_kabsch_rmsd_np(refined, clean))
                base_rmsds.append(_kabsch_rmsd_np(shake_constrained_refine(noisy), clean))
                noisy_rmsds.append(_kabsch_rmsd_np(noisy, clean))
    return {
        "model": float(np.mean(model_rmsds)),
        "shake_baseline": float(np.mean(base_rmsds)),
        "noisy_input": float(np.mean(noisy_rmsds)),
    }


def train_learned_rg_refiner(cfg: RGTrainingConfig):
    """
    Trains LearnedRGRefiner on real PDB structures and validates every
    `cfg.eval_interval` epochs against shake_constrained_refine on the same
    held-out structures and noise levels.

    Requires:
      - cfg.pdb_dir: a directory of training .pdb files (real structures)
      - cfg.held_out_pdb_dir: a DIFFERENT directory of .pdb files, not used
        for training, only for validation
      - PyTorch installed; a GPU is used automatically if cfg.device="cuda"
        and one is available, otherwise falls back to CPU with a printed
        warning (CPU training will be slow for anything beyond a handful of
        short chains — that's a performance issue, not a correctness one)

    Returns (model, history) where history is a dict of per-epoch train loss
    and per-eval-interval validation RMSDs for model/baseline/noisy-input, so
    a run can be plotted and the model/baseline comparison checked directly
    rather than trusted on the trainer's word.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "train_learned_rg_refiner requires PyTorch. Install torch to use it."
        )

    train_files = _discover_pdb_files(cfg.pdb_dir)
    held_out_files = _discover_pdb_files(cfg.held_out_pdb_dir)
    if not train_files:
        raise FileNotFoundError(
            f"No .pdb files found in cfg.pdb_dir={cfg.pdb_dir!r}. "
            "Point this at a directory of real training structures."
        )
    if not held_out_files:
        raise FileNotFoundError(
            f"No .pdb files found in cfg.held_out_pdb_dir={cfg.held_out_pdb_dir!r}. "
            "Point this at a directory of real structures DISJOINT from pdb_dir."
        )
    overlap = set(train_files) & set(held_out_files)
    if overlap:
        raise ValueError(
            f"pdb_dir and held_out_pdb_dir share {len(overlap)} file(s), e.g. "
            f"{sorted(overlap)[0]!r}. Held-out structures must not be used for "
            "training, or the validation RMSD is meaningless."
        )

    train_chains = _load_chains(cfg.pdb_dir, min_len=cfg.min_chain_len)
    held_out_chains = _load_chains(cfg.held_out_pdb_dir, min_len=cfg.min_chain_len)
    if not train_chains:
        raise ValueError(
            f"Found {len(train_files)} .pdb file(s) in {cfg.pdb_dir!r} but none "
            f"had a CA chain of length >= {cfg.min_chain_len}."
        )
    if not held_out_chains:
        raise ValueError(
            f"Found {len(held_out_files)} .pdb file(s) in {cfg.held_out_pdb_dir!r} "
            f"but none had a CA chain of length >= {cfg.min_chain_len}."
        )

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[train_learned_rg_refiner] cfg.device='cuda' but no CUDA device "
              f"is available — falling back to CPU. Training will be slow.")
        device = torch.device("cpu")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = LearnedRGRefiner(hidden_dim=cfg.hidden_dim, window=cfg.window,
                              n_layers=cfg.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: Dict[str, list] = {
        "train_loss": [], "eval_epoch": [],
        "val_model": [], "val_shake_baseline": [], "val_noisy_input": [],
    }
    best_val = float("inf")

    print(f"Training on {len(train_chains)} chain(s) from {cfg.pdb_dir!r}, "
          f"validating on {len(held_out_chains)} held-out chain(s) from "
          f"{cfg.held_out_pdb_dir!r}, device={device}.")

    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(train_chains))
        opt.zero_grad()
        running_loss = 0.0
        n_since_step = 0

        for pos, idx in enumerate(perm):
            clean = train_chains[idx]
            sigma = float(rng.choice(cfg.noise_levels))
            noisy = clean + rng.normal(0, sigma, clean.shape).astype(np.float32)

            noisy_t = torch.tensor(noisy, dtype=torch.float32, device=device)
            clean_t = torch.tensor(clean, dtype=torch.float32, device=device)

            refined = model(noisy_t)
            loss = _kabsch_rmsd_torch(refined, clean_t)
            (loss / cfg.batch_size).backward()

            running_loss += loss.item()
            n_since_step += 1
            is_last = (pos == len(perm) - 1)
            if n_since_step >= cfg.batch_size or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                opt.zero_grad()
                n_since_step = 0

        epoch_loss = running_loss / len(perm)
        history["train_loss"].append(epoch_loss)

        is_last_epoch = (epoch == cfg.epochs - 1)
        if (epoch + 1) % cfg.eval_interval == 0 or is_last_epoch:
            model.eval()
            val = _evaluate(model, held_out_chains, cfg.noise_levels, device, rng)
            history["eval_epoch"].append(epoch + 1)
            history["val_model"].append(val["model"])
            history["val_shake_baseline"].append(val["shake_baseline"])
            history["val_noisy_input"].append(val["noisy_input"])
            beats_baseline = val["model"] < val["shake_baseline"]
            print(f"epoch {epoch + 1:4d}/{cfg.epochs}  train_loss={epoch_loss:.4f}  "
                  f"val_model={val['model']:.4f}  val_shake_baseline={val['shake_baseline']:.4f}  "
                  f"val_noisy_input={val['noisy_input']:.4f}  "
                  f"{'(beats baseline)' if beats_baseline else '(NOT beating baseline yet)'}")

            if val["model"] < best_val:
                best_val = val["model"]
                if cfg.save_path:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg,
                        "epoch": epoch + 1,
                        "val_rmsd": val,
                    }, cfg.save_path)

    if best_val >= _evaluate(model, held_out_chains, cfg.noise_levels, device, rng)["shake_baseline"]:
        print("\nWARNING: final model did not beat the shake_constrained_refine "
              "baseline on held-out data. Do not report this run as an improvement; "
              "treat it as a training run to iterate on (more data, more epochs, "
              "different window/hidden_dim, etc.).")

    return model, history


# =============================================================================
# 5. CLI
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Train LearnedRGRefiner on real PDB structures, or, with "
                    "only --pdb_dir given, run the shake_constrained_refine "
                    "baseline self-test (no training, no torch required)."
    )
    parser.add_argument("--pdb_dir", type=str, default=".",
                         help="Training structures (or the only structures, "
                              "for baseline-only mode).")
    parser.add_argument("--held_out_pdb_dir", type=str, default="",
                         help="Held-out validation structures. If omitted, "
                              "runs baseline-only self-test instead of training.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="rg_refiner_best.pt")
    args = parser.parse_args()

    if args.held_out_pdb_dir:
        if not _TORCH_AVAILABLE:
            print("PyTorch is not installed — cannot train. Install torch, or "
                  "omit --held_out_pdb_dir to run the baseline-only self-test.")
            sys.exit(1)
        cfg = RGTrainingConfig(
            pdb_dir=args.pdb_dir,
            held_out_pdb_dir=args.held_out_pdb_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            window=args.window,
            n_layers=args.n_layers,
            device=args.device,
            save_path=args.save_path,
        )
        train_learned_rg_refiner(cfg)
    else:
        pdb_files = _discover_pdb_files(args.pdb_dir)
        if not pdb_files:
            print(f"No .pdb files found in {args.pdb_dir}. Usage: "
                  f"python rg_refiner.py --pdb_dir <dir> [--held_out_pdb_dir <dir> ...]")
            sys.exit(0)

        print("Baseline-only self-test: shake_constrained_refine on", len(pdb_files), "structure(s)")
        print(f"{'Structure':20s} {'N_CA':>6s} {'RMSD@noise=0':>14s} {'max_bond_dev':>14s}")
        for fp in pdb_files:
            coords = _parse_ca(fp)
            if len(coords) < 10:
                continue
            out = shake_constrained_refine(coords)
            rmsd0 = _kabsch_rmsd_np(out, coords)
            bonds_before = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            bonds_after = np.linalg.norm(np.diff(out, axis=0), axis=1)
            max_dev = np.abs(bonds_after - bonds_before).max()
            print(f"{fp.split('/')[-1]:20s} {len(coords):6d} {rmsd0:14.4f} {max_dev:14.6f}")
        print("\n(RMSD@noise=0 should be small but is not exactly 0 - the Laplacian")
        print(" step drifts bond ANGLES slightly even though SHAKE keeps every")
        print(" bond LENGTH exact. Pass --held_out_pdb_dir to train the learned")
        print(" model instead of just running this baseline.)")
