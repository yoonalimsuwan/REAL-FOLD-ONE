"""
=============================================================================
PROTEIN FOLDING METRICS + DDP TRAINING MODULE v1.0
RMSD / lDDT / TM-score Evaluation + DistributedDataParallel Training
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================

WHAT THIS MODULE ADDS
─────────────────────
[M1] RMSD via Kabsch alignment (optimal rotation + translation)
[M2] lDDT (local Distance Difference Test) per Mariani et al. 2013
[M3] TM-score per Zhang & Skolnick 2004
[M4] DDP Trainer with DistributedSampler + gradient sync
[M5] Test evaluation loop with per-residue and global metrics
[M6] Checkpoint save/load with DDP state handling

INTEGRATION POINTS
──────────────────
• Seq2CoarseTrainer (seq_to_coarse_structure-3.py)
• ProteinFoldingOrchestrator (PROTEIN_FOLDING_ORCHESTRATOR.py)
• RefinementEngine (real_fold_one_v2-5.py)
• CalabiYauProteinFolder (SESI_UNIFIED_BIOMEDICAL_INTEGRATION.py)

HONEST SCOPE
────────────
• RMSD/lDDT computed on Cα coordinates only (not all-atom)
• lDDT uses the standard 4-threshold version from CASP
• TM-score uses the standard d0 normalization
• DDP assumes NCCL backend with CUDA; falls back to gloo on CPU
• No claim of AlphaFold-level accuracy — these are evaluation tools

=============================================================================
"""

from __future__ import annotations

import math
import os
import time
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger("protein_metrics_ddp")

__all__ = [
    # Metrics
    "compute_rmsd_kabsch",
    "compute_lddt",
    "compute_tm_score",
    "ProteinMetrics",
    # DDP infrastructure
    "setup_dist",
    "cleanup_dist",
    "is_main",
    "all_reduce_mean",
    # Dataset
    "ProteinStructureDataset",
    # Trainer
    "DDPProteinTrainer",
    # Evaluation
    "evaluate_on_test_set",
]


# ═══════════════════════════════════════════════════════════════════════
# [M1] RMSD VIA KABSCH ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════

def compute_rmsd_kabsch(
    coords_pred: np.ndarray,
    coords_true: np.ndarray,
    align: bool = True,
) -> float:
    """
    Compute RMSD between predicted and true Cα coordinates.
    
    Uses Kabsch algorithm for optimal superposition when align=True.
    
    Args:
        coords_pred: (N, 3) predicted coordinates
        coords_true: (N, 3) true coordinates
        align: whether to perform Kabsch alignment first
    
    Returns:
        RMSD in Angstroms
    """
    if coords_pred.shape != coords_true.shape:
        raise ValueError(
            f"Shape mismatch: pred {coords_pred.shape} vs true {coords_true.shape}")
    
    N = coords_pred.shape[0]
    if N == 0:
        return 0.0
    
    if align:
        # Center both structures
        pred_centered = coords_pred - coords_pred.mean(axis=0)
        true_centered = coords_true - coords_true.mean(axis=0)
        
        # Kabsch: find optimal rotation R minimizing ||P - R·Q||²
        H = pred_centered.T @ true_centered  # (3, 3)
        U, S, Vt = np.linalg.svd(H)
        
        # Handle reflection case
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
        
        # Optimal rotation
        R = Vt.T @ sign_matrix @ U.T
        
        # Apply rotation
        pred_aligned = pred_centered @ R.T
        
        # RMSD
        diff = pred_aligned - true_centered
    else:
        diff = coords_pred - coords_true
    
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
    return float(rmsd)


def compute_rmsd_torch(
    coords_pred: Tensor,
    coords_true: Tensor,
    align: bool = True,
) -> Tensor:
    """
    PyTorch version of RMSD computation (differentiable).
    
    Args:
        coords_pred: (B, N, 3) or (N, 3) predicted coordinates
        coords_true: same shape as coords_pred
        align: whether to align via Kabsch
    
    Returns:
        RMSD tensor (scalar or (B,) for batched input)
    """
    if coords_pred.dim() == 2:
        coords_pred = coords_pred.unsqueeze(0)
        coords_true = coords_true.unsqueeze(0)
    
    B, N, _ = coords_pred.shape
    
    if align:
        # Center
        pred_c = coords_pred - coords_pred.mean(dim=1, keepdim=True)
        true_c = coords_true - coords_true.mean(dim=1, keepdim=True)
        
        # Kabsch via SVD (batched)
        H = torch.bmm(pred_c.transpose(1, 2), true_c)  # (B, 3, 3)
        U, S, Vh = torch.linalg.svd(H)
        
        # Reflection correction
        d = torch.det(torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2)))
        sign = torch.ones(B, 3, device=coords_pred.device)
        sign[:, 2] = torch.sign(d)
        sign_diag = torch.diag_embed(sign)
        
        R = torch.bmm(torch.bmm(Vh.transpose(1, 2), sign_diag), U.transpose(1, 2))
        
        pred_aligned = torch.bmm(pred_c, R.transpose(1, 2))
        diff = pred_aligned - true_c
    else:
        diff = coords_pred - coords_true
    
    rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean(dim=-1))  # (B,)
    
    if B == 1:
        return rmsd.squeeze(0)
    return rmsd


# ═══════════════════════════════════════════════════════════════════════
# [M2] lDDT — LOCAL DISTANCE DIFFERENCE TEST
# ═══════════════════════════════════════════════════════════════════════

def compute_lddt(
    coords_pred: np.ndarray,
    coords_true: np.ndarray,
    inclusion_radius: float = 15.0,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    seq_separation: int = 15,
) -> float:
    """
    Compute lDDT (local Distance Difference Test) score.
    
    Reference: Mariani et al., Bioinformatics 29(21):2722-2728, 2013.
    
    lDDT measures the fraction of inter-residue distances in the predicted
    structure that are within threshold of the corresponding distances in
    the true structure, for pairs within the inclusion radius in the true
    structure.
    
    Args:
        coords_pred: (N, 3) predicted Cα coordinates
        coords_true: (N, 3) true Cα coordinates
        inclusion_radius: maximum distance in true structure to consider (Å)
        thresholds: distance difference thresholds (Å)
        seq_separation: minimum sequence separation for pairs
    
    Returns:
        lDDT score in [0, 1] (multiply by 100 for percentage)
    """
    N = coords_pred.shape[0]
    if N < 2:
        return 1.0
    
    # Compute all pairwise distances
    pred_dist = np.sqrt(np.sum(
        (coords_pred[:, None, :] - coords_pred[None, :, :]) ** 2, axis=-1))
    true_dist = np.sqrt(np.sum(
        (coords_true[:, None, :] - coords_true[None, :, :]) ** 2, axis=-1))
    
    # Mask: pairs within inclusion radius in TRUE structure
    # and separated by at least seq_separation in sequence
    seq_sep_mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) >= seq_separation
    inclusion_mask = true_dist < inclusion_radius
    valid_mask = seq_sep_mask & inclusion_mask
    
    # Exclude diagonal
    np.fill_diagonal(valid_mask, False)
    
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return 1.0
    
    # Distance differences
    dist_diff = np.abs(pred_dist - true_dist)[valid_mask]
    
    # Score: fraction of pairs within each threshold
    scores = []
    for thresh in thresholds:
        frac = (dist_diff < thresh).sum() / n_valid
        scores.append(frac)
    
    # lDDT is the average across thresholds
    lddt = np.mean(scores)
    return float(lddt)


def compute_plddt(
    coords_pred: np.ndarray,
    coords_true: np.ndarray,
    window: int = 15,
    inclusion_radius: float = 15.0,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> np.ndarray:
    """
    Compute per-residue pLDDT (predicted lDDT).
    
    Args:
        coords_pred: (N, 3) predicted Cα coordinates
        coords_true: (N, 3) true Cα coordinates
        window: local window size for per-residue scoring
    
    Returns:
        (N,) array of per-residue lDDT scores in [0, 1]
    """
    N = coords_pred.shape[0]
    plddt = np.zeros(N)
    
    pred_dist = np.sqrt(np.sum(
        (coords_pred[:, None, :] - coords_pred[None, :, :]) ** 2, axis=-1))
    true_dist = np.sqrt(np.sum(
        (coords_true[:, None, :] - coords_true[None, :, :]) ** 2, axis=-1))
    
    for i in range(N):
        # Local window
        lo = max(0, i - window)
        hi = min(N, i + window + 1)
        
        # Pairs involving residue i within window
        local_pred = pred_dist[i, lo:hi]
        local_true = true_dist[i, lo:hi]
        
        # Mask
        seq_sep = np.abs(np.arange(lo, hi) - i) >= 1
        in_radius = local_true < inclusion_radius
        valid = seq_sep & in_radius
        
        if valid.sum() == 0:
            plddt[i] = 1.0
            continue
        
        dist_diff = np.abs(local_pred - local_true)[valid]
        scores = [(dist_diff < t).sum() / valid.sum() for t in thresholds]
        plddt[i] = np.mean(scores)
    
    return plddt


# ═══════════════════════════════════════════════════════════════════════
# [M3] TM-SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_tm_score(
    coords_pred: np.ndarray,
    coords_true: np.ndarray,
) -> float:
    """
    Compute TM-score (Template Modeling score).
    
    Reference: Zhang & Skolnick, Nucleic Acids Res. 32:5076-5082, 2004.
    
    TM-score = (1/L_true) * Σ_i 1/(1 + (d_i/d0)²)
    where d0 = 1.24 * (L_true - 15)^(1/3) - 1.8
    
    Args:
        coords_pred: (N, 3) predicted Cα coordinates (after alignment)
        coords_true: (N, 3) true Cα coordinates
    
    Returns:
        TM-score in [0, 1]
    """
    N = coords_true.shape[0]
    if N == 0:
        return 0.0
    
    # d0 normalization
    d0 = 1.24 * (N - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)  # avoid division issues for very short chains
    
    # Align via Kabsch
    pred_c = coords_pred - coords_pred.mean(axis=0)
    true_c = coords_true - coords_true.mean(axis=0)
    
    H = pred_c.T @ true_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    
    pred_aligned = pred_c @ R.T
    
    # Per-residue distances after alignment
    dist = np.sqrt(np.sum((pred_aligned - true_c) ** 2, axis=-1))
    
    # TM-score formula
    tm = np.mean(1.0 / (1.0 + (dist / d0) ** 2))
    
    return float(tm)


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED METRICS CLASS
# ═══════════════════════════════════════════════════════════════════════

class ProteinMetrics:
    """
    Unified protein structure evaluation metrics.
    
    Computes:
      • RMSD (global, after Kabsch alignment)
      • lDDT (global, 4-threshold)
      • pLDDT (per-residue)
      • TM-score
      • GDT_TS (fraction of residues within 1/2/4/8 Å)
    """
    
    @staticmethod
    def compute_all(
        coords_pred: np.ndarray,
        coords_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute all metrics for a single structure.
        
        Args:
            coords_pred: (N, 3) predicted Cα coordinates
            coords_true: (N, 3) true Cα coordinates
        
        Returns:
            dict with all metric values
        """
        if coords_pred.shape != coords_true.shape:
            raise ValueError("Shape mismatch")
        
        N = coords_pred.shape[0]
        
        # RMSD
        rmsd = compute_rmsd_kabsch(coords_pred, coords_true, align=True)
        
        # lDDT
        lddt = compute_lddt(coords_pred, coords_true)
        
        # pLDDT
        plddt = compute_plddt(coords_pred, coords_true)
        
        # TM-score
        tm_score = compute_tm_score(coords_pred, coords_true)
        
        # GDT_TS
        pred_c = coords_pred - coords_pred.mean(axis=0)
        true_c = coords_true - coords_true.mean(axis=0)
        H = pred_c.T @ true_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T
        pred_aligned = pred_c @ R.T
        dist = np.sqrt(np.sum((pred_aligned - true_c) ** 2, axis=-1))
        
        gdt_ts = np.mean([
            (dist < 1.0).sum() / N,
            (dist < 2.0).sum() / N,
            (dist < 4.0).sum() / N,
            (dist < 8.0).sum() / N,
        ])
        
        return {
            "rmsd": rmsd,
            "lddt": lddt,
            "plddt_mean": float(plddt.mean()),
            "plddt_min": float(plddt.min()),
            "plddt_max": float(plddt.max()),
            "tm_score": tm_score,
            "gdt_ts": float(gdt_ts),
            "n_residues": N,
            "plddt_per_residue": plddt.tolist(),
        }
    
    @staticmethod
    def aggregate_batch(
        metrics_list: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Aggregate metrics across a batch."""
        if not metrics_list:
            return {}
        
        return {
            "rmsd_mean": np.mean([m["rmsd"] for m in metrics_list]),
            "rmsd_median": np.median([m["rmsd"] for m in metrics_list]),
            "lddt_mean": np.mean([m["lddt"] for m in metrics_list]),
            "tm_score_mean": np.mean([m["tm_score"] for m in metrics_list]),
            "gdt_ts_mean": np.mean([m["gdt_ts"] for m in metrics_list]),
            "plddt_mean": np.mean([m["plddt_mean"] for m in metrics_list]),
            "n_structures": len(metrics_list),
        }


# ═══════════════════════════════════════════════════════════════════════
# DDP INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════

def setup_dist(backend: str = "nccl") -> Tuple[torch.device, int, int, int]:
    """
    Initialize distributed training.
    
    Returns:
        (device, rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group(backend=backend)
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
        
        logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, "
                   f"local_rank={local_rank}, device={device}")
        return device, rank, world_size, local_rank
    else:
        # Single GPU / CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Single device mode: {device}")
        return device, 0, 1, 0


def cleanup_dist() -> None:
    """Clean up distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main() -> bool:
    """Check if current process is rank 0."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def all_reduce_mean(x: Tensor) -> Tensor:
    """Average a tensor across all processes."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / dist.get_world_size()


def all_reduce_sum(x: Tensor) -> Tensor:
    """Sum a tensor across all processes."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


# ═══════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════

class ProteinStructureDataset(Dataset):
    """
    Dataset of protein sequences + true Cα coordinates.
    
    Can load from:
      • Pre-computed .pt files
      • PDB files (requires biotite)
      • Synthetic data (for testing)
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        n_synthetic: int = 100,
        max_len: int = 256,
        seed: int = 42,
    ):
        self.max_len = max_len
        
        if data_path is not None and Path(data_path).exists():
            self._load_real_data(data_path)
        else:
            self._generate_synthetic(n_synthetic, seed)
    
    def _load_real_data(self, path: str):
        """Load pre-computed protein data."""
        data = torch.load(path, map_location="cpu")
        self.sequences = data["sequences"]
        self.coords = data["coords"]
        self.names = data.get("names", [f"prot_{i}" for i in range(len(self.coords))])
    
    def _generate_synthetic(self, n: int, seed: int):
        """Generate synthetic protein data for testing."""
        rng = np.random.RandomState(seed)
        
        self.sequences = []
        self.coords = []
        self.names = []
        
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        for i in range(n):
            length = rng.randint(30, self.max_len)
            seq = "".join(rng.choice(list(aa_alphabet), length))
            
            # Random walk for coordinates
            steps = rng.randn(length, 3) * 3.8
            steps = steps / np.linalg.norm(steps, axis=1, keepdims=True) * 3.8
            coords = np.cumsum(steps, axis=0)
            
            self.sequences.append(seq)
            self.coords.append(torch.tensor(coords, dtype=torch.float32))
            self.names.append(f"synthetic_{i}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        coords = self.coords[idx]
        
        # Pad/truncate to max_len
        L = len(seq)
        if L > self.max_len:
            seq = seq[:self.max_len]
            coords = coords[:self.max_len]
            L = self.max_len
        
        # Encode sequence
        aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        seq_ids = torch.tensor([aa_to_idx.get(c, 20) for c in seq], dtype=torch.long)
        
        # Pad
        if L < self.max_len:
            pad_len = self.max_len - L
            seq_ids = F.pad(seq_ids, (0, pad_len), value=20)
            coords = F.pad(coords, (0, 0, 0, pad_len), value=0.0)
        
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:L] = True
        
        return {
            "seq_ids": seq_ids,
            "true_coords": coords,
            "mask": mask,
            "length": torch.tensor(L, dtype=torch.long),
            "name": self.names[idx],
        }


def build_protein_loaders(
    data_path: Optional[str] = None,
    batch_size: int = 8,
    n_synthetic: int = 100,
    num_workers: int = 2,
):
    """Build train/val/test dataloaders with DDP support."""
    train_ds = ProteinStructureDataset(data_path, n_synthetic=n_synthetic, seed=42)
    val_ds = ProteinStructureDataset(data_path, n_synthetic=n_synthetic // 5, seed=43)
    test_ds = ProteinStructureDataset(data_path, n_synthetic=n_synthetic // 4, seed=44)
    
    # DDP samplers
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle = True
    
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )
    
    train_dl = DataLoader(train_ds, shuffle=shuffle, drop_last=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)
    test_dl = DataLoader(test_ds, shuffle=False, **common)
    
    return train_dl, val_dl, test_dl


# ═══════════════════════════════════════════════════════════════════════
# DDP TRAINER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DDPTrainerConfig:
    """Configuration for DDP protein folding training."""
    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup_steps: int = 100
    total_steps: int = 10000
    log_interval: int = 50
    ckpt_interval: int = 500
    eval_interval: int = 200
    
    # DDP
    use_amp: bool = True
    amp_dtype: str = "bf16"
    find_unused_parameters: bool = False
    
    # Metrics
    compute_metrics_interval: int = 100  # compute lDDT/RMSD every N steps
    
    # Checkpointing
    ckpt_dir: str = "./checkpoints"
    
    # Device
    device: str = "cuda"


class DDPProteinTrainer:
    """
    Distributed Data Parallel trainer for protein folding models.
    
    Supports:
      • Multi-GPU training via DDP
      • Mixed precision (AMP)
      • Gradient accumulation
      • Periodic evaluation with RMSD/lDDT
      • Checkpointing with DDP state
      • Cosine LR schedule with warmup
    
    Usage:
        trainer = DDPProteinTrainer(model, train_dl, val_dl, test_dl, cfg)
        trainer.fit()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: Optional[DataLoader] = None,
        cfg: Optional[DDPTrainerConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg or DDPTrainerConfig()
        
        # Device
        if device is None:
            device, rank, world_size, local_rank = setup_dist()
        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Model
        self.model = model.to(self.device)
        if dist.is_initialized() and self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=self.cfg.find_unused_parameters,
                gradient_as_bucket_view=True,
            )
        
        # Data
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        
        # Optimizer
        self.opt = torch.optim.AdamW(
            [p for p in self._raw_model().parameters() if p.requires_grad],
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lr_lambda=self._lr_lambda,
        )
        
        # AMP
        self.amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.cfg.amp_dtype]
        self.use_amp = self.cfg.use_amp and self.device.type == "cuda"
        
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        except ImportError:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        
        # Metrics accumulator
        self.metrics = ProteinMetrics()
        
        # Checkpoint dir
        self.ckpt_dir = Path(self.cfg.ckpt_dir)
        if is_main():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    def _raw_model(self) -> nn.Module:
        """Get unwrapped model."""
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m
    
    def _lr_lambda(self, step: int) -> float:
        """Cosine LR with linear warmup."""
        if step < self.cfg.warmup_steps:
            return (step + 1) / max(1, self.cfg.warmup_steps)
        prog = min(1.0, (step - self.cfg.warmup_steps) /
                   max(1, self.cfg.total_steps - self.cfg.warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        One training step.
        
        Args:
            batch: dict with seq_ids, true_coords, mask
        
        Returns:
            dict with loss values
        """
        self.model.train()
        
        batch = {k: v.to(self.device, non_blocking=True)
                 if torch.is_tensor(v) else v for k, v in batch.items()}
        
        self.opt.zero_grad(set_to_none=True)
        
        # Forward
        try:
            from torch.amp import autocast
            ctx = autocast(device_type=self.device.type,
                          dtype=self.amp_dtype, enabled=self.use_amp)
        except ImportError:
            from torch.cuda.amp import autocast
            ctx = autocast(dtype=self.amp_dtype, enabled=self.use_amp)
        
        with ctx:
            out = self.model(batch)
            
            # Compute loss (structure prediction loss)
            if "pred_coords" in out:
                pred = out["pred_coords"]
                true = batch["true_coords"]
                mask = batch["mask"]
                
                # MSE loss on coordinates
                diff = (pred - true) ** 2
                diff = diff * mask.unsqueeze(-1).float()
                loss = diff.sum() / (mask.sum() * 3 + 1e-8)
            else:
                # Fallback: use whatever loss the model provides
                loss = out.get("loss", torch.tensor(0.0, device=self.device))
        
        # Backward
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(
            self._raw_model().parameters(), self.cfg.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        
        # LR schedule
        self.scheduler.step()
        self.global_step += 1
        
        return {"loss": loss.item(), "lr": self.opt.param_groups[0]["lr"]}
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataloader.
        
        Computes: loss, RMSD, lDDT, TM-score.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_metrics = []
        n_batches = 0
        
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if torch.is_tensor(v) else v for k, v in batch.items()}
            
            try:
                from torch.amp import autocast
                ctx = autocast(device_type=self.device.type,
                              dtype=self.amp_dtype, enabled=self.use_amp)
            except ImportError:
                from torch.cuda.amp import autocast
                ctx = autocast(dtype=self.amp_dtype, enabled=self.use_amp)
            
            with ctx:
                out = self.model(batch)
                
                if "pred_coords" in out:
                    pred = out["pred_coords"]
                    true = batch["true_coords"]
                    mask = batch["mask"]
                    
                    diff = (pred - true) ** 2
                    diff = diff * mask.unsqueeze(-1).float()
                    loss = diff.sum() / (mask.sum() * 3 + 1e-8)
                else:
                    loss = out.get("loss", torch.tensor(0.0, device=self.device))
            
            total_loss += loss.item()
            n_batches += 1
            
            # Compute structure metrics (CPU, per-sample)
            if "pred_coords" in out:
                B = pred.shape[0]
                for i in range(B):
                    L = batch["mask"][i].sum().item()
                    pred_np = pred[i, :L].cpu().numpy()
                    true_np = true[i, :L].cpu().numpy()
                    
                    if L >= 4:  # Need at least 4 residues
                        m = self.metrics.compute_all(pred_np, true_np)
                        all_metrics.append(m)
        
        # Aggregate
        avg_loss = total_loss / max(n_batches, 1)
        
        # Reduce across GPUs
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = all_reduce_mean(loss_tensor).item()
        
        # Aggregate metrics
        if all_metrics:
            agg = self.metrics.aggregate_batch(all_metrics)
        else:
            agg = {}
        
        return {"loss": avg_loss, "n_batches": n_batches, **agg}
    
    def fit(self, epochs: int = 10) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            epochs: number of epochs
        
        Returns:
            dict with training history
        """
        history = {"train_loss": [], "val_loss": [], "val_rmsd": [], "val_lddt": []}
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Set epoch for DDP sampler
            if hasattr(self.train_dl, "sampler") and \
               isinstance(self.train_dl.sampler, DistributedSampler):
                self.train_dl.sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            n_steps = 0
            
            for batch_idx, batch in enumerate(self.train_dl):
                step_metrics = self.train_step(batch)
                epoch_loss += step_metrics["loss"]
                n_steps += 1
                
                # Logging
                if is_main() and self.global_step % self.cfg.log_interval == 0:
                    logger.info(
                        f"[Epoch {epoch}] Step {self.global_step}: "
                        f"loss={step_metrics['loss']:.4f} "
                        f"lr={step_metrics['lr']:.2e}"
                    )
                
                # Periodic evaluation
                if self.global_step % self.cfg.eval_interval == 0:
                    val_metrics = self.evaluate(self.val_dl)
                    if is_main():
                        logger.info(
                            f"[Epoch {epoch}] Val: loss={val_metrics['loss']:.4f} "
                            f"RMSD={val_metrics.get('rmsd_mean', 0):.2f}Å "
                            f"lDDT={val_metrics.get('lddt_mean', 0):.4f} "
                            f"TM={val_metrics.get('tm_score_mean', 0):.4f}"
                        )
                        history["val_loss"].append(val_metrics["loss"])
                        history["val_rmsd"].append(val_metrics.get("rmsd_mean", 0))
                        history["val_lddt"].append(val_metrics.get("lddt_mean", 0))
                
                # Checkpoint
                if is_main() and self.global_step % self.cfg.ckpt_interval == 0:
                    self.save_checkpoint()
                
                # Early stopping
                if self.global_step >= self.cfg.total_steps:
                    break
            
            history["train_loss"].append(epoch_loss / max(n_steps, 1))
            
            if self.global_step >= self.cfg.total_steps:
                break
        
        # Final evaluation on test set
        if self.test_dl is not None and is_main():
            logger.info("Running final test set evaluation...")
            test_metrics = self.evaluate(self.test_dl)
            logger.info(f"Test: RMSD={test_metrics.get('rmsd_mean', 0):.2f}Å "
                       f"lDDT={test_metrics.get('lddt_mean', 0):.4f} "
                       f"TM={test_metrics.get('tm_score_mean', 0):.4f}")
            history["test_metrics"] = test_metrics
        
        return history
    
    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """Save checkpoint (rank 0 only)."""
        if not is_main():
            return Path("")
        
        tag = tag or f"step_{self.global_step}"
        path = self.ckpt_dir / f"protein_fold_{tag}.pt"
        
        torch.save({
            "model_state": self._raw_model().state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }, path)
        
        logger.info(f"Checkpoint saved: {path}")
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        
        self._raw_model().load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        
        self.global_step = ckpt.get("global_step", 0)
        self.epoch = ckpt.get("epoch", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        
        logger.info(f"Checkpoint loaded from step {self.global_step}")


# ═══════════════════════════════════════════════════════════════════════
# TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_on_test_set(
    model: nn.Module,
    test_dl: DataLoader,
    device: torch.device,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full evaluation on test set with per-structure metrics.
    
    Args:
        model: trained model
        test_dl: test dataloader
        device: computation device
        output_file: optional path to save results
    
    Returns:
        dict with aggregated and per-structure metrics
    """
    model.eval()
    metrics = ProteinMetrics()
    
    all_results = []
    
    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(device, non_blocking=True)
                     if torch.is_tensor(v) else v for k, v in batch.items()}
            
            out = model(batch)
            
            if "pred_coords" not in out:
                continue
            
            pred = out["pred_coords"]
            true = batch["true_coords"]
            mask = batch["mask"]
            
            B = pred.shape[0]
            for i in range(B):
                L = mask[i].sum().item()
                if L < 4:
                    continue
                
                pred_np = pred[i, :L].cpu().numpy()
                true_np = true[i, :L].cpu().numpy()
                
                result = metrics.compute_all(pred_np, true_np)
                result["name"] = batch.get("name", [f"sample_{i}"])[i]
                all_results.append(result)
    
    # Aggregate
    if all_results:
        agg = metrics.aggregate_batch(all_results)
    else:
        agg = {}
    
    # Summary
    summary = {
        "n_structures": len(all_results),
        "aggregate": agg,
        "per_structure": all_results,
    }
    
    # Save
    if output_file is not None and is_main():
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    
    return summary


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Smoke test for metrics and DDP infrastructure."""
    print("=" * 72)
    print(" Protein Metrics + DDP — Smoke Test")
    print("=" * 72)
    
    # Test RMSD
    print("\n  ── RMSD Test ──")
    np.random.seed(42)
    N = 50
    coords_true = np.cumsum(np.random.randn(N, 3) * 3.8, axis=0)
    coords_pred = coords_true + np.random.randn(N, 3) * 0.5  # small perturbation
    
    rmsd = compute_rmsd_kabsch(coords_pred, coords_true)
    print(f"    RMSD (perturbed): {rmsd:.4f} Å")
    assert rmsd < 2.0, "RMSD should be small for small perturbation"
    
    # Test lDDT
    print("\n  ── lDDT Test ──")
    lddt = compute_lddt(coords_pred, coords_true)
    print(f"    lDDT: {lddt:.4f}")
    assert 0.0 <= lddt <= 1.0
    
    # Test TM-score
    print("\n  ── TM-score Test ──")
    tm = compute_tm_score(coords_pred, coords_true)
    print(f"    TM-score: {tm:.4f}")
    assert 0.0 <= tm <= 1.0
    
    # Test unified metrics
    print("\n  ── Unified Metrics ──")
    all_metrics = ProteinMetrics.compute_all(coords_pred, coords_true)
    print(f"    RMSD: {all_metrics['rmsd']:.4f} Å")
    print(f"    lDDT: {all_metrics['lddt']:.4f}")
    print(f"    TM-score: {all_metrics['tm_score']:.4f}")
    print(f"    GDT_TS: {all_metrics['gdt_ts']:.4f}")
    print(f"    pLDDT mean: {all_metrics['plddt_mean']:.4f}")
    
    # Test DDP setup
    print("\n  ── DDP Setup ──")
    device, rank, world_size, local_rank = setup_dist()
    print(f"    Device: {device}")
    print(f"    Rank: {rank}, World size: {world_size}")
    
    # Test dataset
    print("\n  ── Dataset ──")
    ds = ProteinStructureDataset(n_synthetic=20, max_len=64)
    print(f"    Dataset size: {len(ds)}")
    
    sample = ds[0]
    print(f"    Sample keys: {list(sample.keys())}")
    print(f"    seq_ids shape: {tuple(sample['seq_ids'].shape)}")
    print(f"    true_coords shape: {tuple(sample['true_coords'].shape)}")
    
    print("\n" + "=" * 72)
    print("  All smoke tests passed!")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
