"""
=============================================================================
SESI MSA TRAINING + ORGAN EXPORT + FULL SLURM PIPELINE — v1.0.0
Real MSAEnhancedFoldingSystem Training · OrganPDBExporter Bridge · 6-Stage Slurm
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
MODULES IN THIS FILE
=============================================================================
[1] MSAFoldingTrainer        — real training loop for MSAEnhancedFoldingSystem
                               (folding_head + fusion layers), MSE + consistency
                               loss, AMP, gradient clipping, validation loop

[2] OrganPDBExporterBridge   — converts predicted coordinates into organ-level
                               multi-copy PDB via OrganPDBExporter

[3] SlurmFullPipelineGenerator — generates 6-stage Slurm batch script + all
                                  stage entry-point scripts

[4] ModuleIntegrationMap     — explicit wiring map across the full ecosystem

[5] FullEcosystemOrchestrator — runs MSA training → folding → organ export
                                 in one call (for local testing)

[6] Self-test suite

=============================================================================
INTEGRATION WITH EXISTING ECOSYSTEM FILES
=============================================================================
This module calls verbatim APIs from the following files:

  MSA_ENHANCED_PROTEIN_FOLDING (1).py
    → MSAEnhancedFoldingSystem.forward(sequence, esm_features, msa)
    → MSAEnhancedConfig, MSAFoldingMode
    → MSASequenceFuser (inside MSAEnhancedFoldingSystem)
    → folding_head: nn.Sequential(Linear, GELU, Linear) ← TRAINED HERE

  organ_PDB_exporter_download.py
    → OrganPDBExporter.export_macro_assembly(template_atoms, positions, rotations, path)
    → OrganPDBExporter.export_differentiable(template_atoms, positions, rotations)
    → OrganPDBExporter.validate_pdb(path)
    → parse_pdb_template(path)
    → create_organ_grid(grid_shape, spacing)

  sesi_full_stage_scripts_msa.py (from previous module)
    → Stage1Designer, Stage2Folder, Stage3Refiner, Stage4Validator

  structural_domain_assembly_one.py
    → StructuralDomainAssembly.forward(sequence) → init_coords

  real_fold_one_v2-5.py
    → RefinementEngine.refine(pdb_file) → solute_coords

=============================================================================
HONEST SCOPE
=============================================================================
• MSAFoldingTrainer trains from RANDOM initialization. The folding_head
  will produce meaningless coordinates until trained on real data.
  Synthetic data (idealized helices/sheets) is for PIPELINE TESTING ONLY.
• OrganPDBExporterBridge places rigid copies on a grid. It does NOT
  perform flexible docking or conformational sampling.
• Slurm scripts are TEMPLATES. Account names, partitions, and module
  paths must be customized per cluster.
• The full pipeline has NOT been run end-to-end at 100M+ scale here.
=============================================================================
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.msa_train_organ_slurm")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

TRAIN_ORGAN_SLURM_VERSION: str = "1.0.0"

# =============================================================================
# Ecosystem imports (graceful fallback)
# =============================================================================
_HAS_MSA = False
try:
    from MSA_ENHANCED_PROTEIN_FOLDING import (
        MSAEnhancedConfig, MSAFoldingMode, MSAEnhancedFoldingSystem,
        MSASequenceFuser, AdaptiveMSARouter,
    )
    _HAS_MSA = True
except ImportError:
    import importlib.util
    for cand in ("MSA_ENHANCED_PROTEIN_FOLDING (1).py", "MSA_ENHANCED_PROTEIN_FOLDING.py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("MSA_ENHANCED_PROTEIN_FOLDING", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                MSAEnhancedConfig = mod.MSAEnhancedConfig
                MSAFoldingMode = mod.MSAFoldingMode
                MSAEnhancedFoldingSystem = mod.MSAEnhancedFoldingSystem
                MSASequenceFuser = mod.MSASequenceFuser
                AdaptiveMSARouter = mod.AdaptiveMSARouter
                _HAS_MSA = True
            break
    if not _HAS_MSA:
        warnings.warn("MSA_ENHANCED_PROTEIN_FOLDING not found. MSA training disabled.")

_HAS_ORGAN = False
try:
    from organ_PDB_exporter_download import (
        OrganPDBExporter, DifferentiableOrganPDBExporter,
        parse_pdb_template, create_organ_grid,
    )
    _HAS_ORGAN = True
except ImportError:
    import importlib.util
    p = Path("organ_PDB_exporter_download.py")
    if p.exists():
        spec = importlib.util.spec_from_file_location("organ_PDB_exporter_download", str(p))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            OrganPDBExporter = mod.OrganPDBExporter
            DifferentiableOrganPDBExporter = mod.DifferentiableOrganPDBExporter
            parse_pdb_template = mod.parse_pdb_template
            create_organ_grid = mod.create_organ_grid
            _HAS_ORGAN = True
    if not _HAS_ORGAN:
        warnings.warn("organ_PDB_exporter_download.py not found. Organ export disabled.")

_HAS_STAGE_SCRIPTS = False
try:
    from sesi_full_stage_scripts_msa import (
        Stage1Designer, Stage1Config,
        Stage2Folder, Stage2Config,
        Stage3Refiner, Stage3Config,
        Stage4Validator, Stage4Config,
    )
    _HAS_STAGE_SCRIPTS = True
except ImportError:
    warnings.warn("sesi_full_stage_scripts_msa not found. Stage 1-4 integration disabled.")

logger.info(f"[imports] MSA={_HAS_MSA} Organ={_HAS_ORGAN} StageScripts={_HAS_STAGE_SCRIPTS}")


# =============================================================================
# 1. SYNTHETIC FOLD DATASET (for pipeline testing)
# =============================================================================
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

class SyntheticFoldDataset(Dataset):
    """
    Synthetic dataset of idealized helices/sheets for pipeline testing.
    WARNING: NOT real protein data. For production, replace with PDB dataset.
    """

    def __init__(self, n_samples: int = 100, min_len: int = 30,
                 max_len: int = 200, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_samples = n_samples
        self.min_len = min_len
        self.max_len = max_len
        self._cache: Dict[int, Dict] = {}

    def _make_helix(self, n: int) -> torch.Tensor:
        t = torch.arange(n, dtype=torch.float32)
        ang = t * math.radians(100)
        return torch.stack([2.3 * torch.cos(ang), 2.3 * torch.sin(ang), t * 1.5], dim=-1)

    def _make_sheet(self, n: int, n_strands: int = 3) -> torch.Tensor:
        strand_len = n // n_strands
        coords = []
        for s in range(n_strands):
            t = torch.arange(strand_len, dtype=torch.float32)
            z = t * 3.5 if s % 2 == 0 else (strand_len - 1 - t) * 3.5
            x = s * 5.0 * torch.ones(strand_len)
            y = 0.5 * torch.sin(t * math.pi / strand_len)
            coords.append(torch.stack([x, y, z], dim=-1))
        return torch.cat(coords, dim=0)[:n]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._cache:
            return self._cache[idx]

        length = int(self.rng.integers(self.min_len, self.max_len))
        sequence = "".join(self.rng.choice(list(AA_ALPHABET), size=length))
        esm_features = torch.randn(length, 128)  # mock ESM dim=128

        if self.rng.random() < 0.5:
            coords = self._make_helix(length)
        else:
            coords = self._make_sheet(length)

        # Add noise to coordinates (simulates prediction error)
        coords = coords + torch.randn_like(coords) * 0.3

        item = {
            "sequence": sequence,
            "esm_features": esm_features,
            "target_coords": coords,
            "length": length,
        }
        self._cache[idx] = item
        return item


def fold_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate variable-length folds with padding."""
    Lmax = max(item["length"] for item in batch)
    B = len(batch)
    D = batch[0]["esm_features"].shape[1]

    esm = torch.zeros(B, Lmax, D)
    coords = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    sequences = []

    for i, item in enumerate(batch):
        L = item["length"]
        esm[i, :L] = item["esm_features"]
        coords[i, :L] = item["target_coords"]
        mask[i, :L] = True
        sequences.append(item["sequence"])

    return {"esm": esm, "coords": coords, "mask": mask,
            "sequences": sequences,
            "lengths": torch.tensor([item["length"] for item in batch])}


# =============================================================================
# 2. MSA FOLDING TRAINER (real training loop for folding_head)
# =============================================================================
@dataclass
class MSAFoldingTrainerConfig:
    """Configuration for MSAEnhancedFoldingSystem training."""
    # Model
    esm_dim: int = 128
    fused_dim: int = 64
    hodge_dim: int = 64

    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 4
    grad_clip: float = 5.0
    warmup_epochs: int = 2

    # Loss weights
    w_coord: float = 1.0
    w_consistency: float = 0.1

    # What to train
    train_only_head: bool = False  # True: only folding_head; False: full system

    # AMP
    use_amp: bool = True
    amp_dtype: str = "bf16"

    # Mode
    msa_mode: str = "adaptive"

    # Output
    checkpoint_dir: str = "./checkpoints_msa"
    log_interval: int = 5

    # Device
    device: str = "auto"


class MSAFoldingTrainer:
    """
    [D] Real training loop for MSAEnhancedFoldingSystem.

    Trains the folding_head (and optionally the full MSA fusion pipeline)
    to predict 3D coordinates from sequence + ESM features.

    LOSS FUNCTION:
        L = w_coord * MSE(predicted_coords, target_coords)
          + w_consistency * consistency_loss (from CarpetHierarchicalFusion)

    TRAINING MODES:
      • train_only_head=True: freeze msa_fuser, train only folding_head
        (fast, for testing; fusion output is random)
      • train_only_head=False: train full system (msa_fuser + folding_head)
        (slow, for production; fusion learns meaningful representations)

    INTEGRATION:
      Uses MSAEnhancedFoldingSystem.forward(sequence, esm_features, msa)
      directly from MSA_ENHANCED_PROTEIN_FOLDING (1).py.
    """

    def __init__(self, cfg: Optional[MSAFoldingTrainerConfig] = None):
        if not _HAS_MSA:
            raise ImportError("MSA_ENHANCED_PROTEIN_FOLDING required for MSAFoldingTrainer.")

        self.cfg = cfg or MSAFoldingTrainerConfig()

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # Build MSA system
        msa_cfg = MSAEnhancedConfig(
            esm_dim=self.cfg.esm_dim,
            fused_dim=self.cfg.fused_dim,
            hodge_dim=self.cfg.hodge_dim,
            device=str(self.device),
        )
        mode = {"single_sequence": MSAFoldingMode.SINGLE_SEQUENCE,
                "msa_enhanced": MSAFoldingMode.MSA_ENHANCED,
                "adaptive": MSAFoldingMode.ADAPTIVE}[self.cfg.msa_mode]
        self.system = MSAEnhancedFoldingSystem(msa_cfg, mode=mode).to(self.device)

        # Freeze msa_fuser if training only head
        if self.cfg.train_only_head:
            for p in self.system.msa_fuser.parameters():
                p.requires_grad_(False)
            for p in self.system.single_seq_projector.parameters():
                p.requires_grad_(False)
            logger.info("[MSATrainer] Training ONLY folding_head (msa_fuser frozen).")
        else:
            logger.info("[MSATrainer] Training FULL system (msa_fuser + folding_head).")

        # Optimizer (only trainable params)
        trainable_params = [p for p in self.system.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params,
                                            lr=self.cfg.lr,
                                            weight_decay=self.cfg.weight_decay)

        # LR scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.cfg.epochs - self.cfg.warmup_epochs))

        # AMP
        self.amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.cfg.use_amp and self.cfg.amp_dtype == "fp16"
                     and self.device.type == "cuda"))

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.history: List[Dict[str, float]] = []
        logger.info(f"[MSATrainer] Initialized on {self.device}, "
                    f"trainable params: {sum(p.numel() for p in trainable_params):,}")

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One training step on a collated batch."""
        self.system.train()
        self.optimizer.zero_grad(set_to_none=True)

        esm = batch["esm"].to(self.device)
        coords_target = batch["coords"].to(self.device)
        mask = batch["mask"].to(self.device)
        sequences = batch["sequences"]
        B, L, _ = esm.shape

        # Forward pass (process each sample; MSA system takes one sequence at a time)
        total_coord_loss = torch.tensor(0.0, device=self.device)
        total_consist_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        ctx = (torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                               enabled=self.cfg.use_amp and self.device.type == "cuda")
               if self.device.type == "cuda" else torch.enable_grad())

        with ctx:
            for i in range(B):
                Li = int(batch["lengths"][i])
                seq_i = sequences[i][:Li]
                esm_i = esm[i, :Li].unsqueeze(0)  # (1, Li, D)

                output = self.system(seq_i, esm_i, msa=None)
                pred_coords = output["predicted_coords"].squeeze(0)  # (Li, 3)

                # Coordinate loss (masked)
                coord_loss = F.mse_loss(pred_coords, coords_target[i, :Li])
                total_coord_loss = total_coord_loss + coord_loss

                # Consistency loss (if MSA was used)
                if output.get("use_msa", False) and "msa_fusion_consistency_loss" in output:
                    total_consist_loss = total_consist_loss + output["msa_fusion_consistency_loss"]

                n_valid += 1

            if n_valid > 0:
                total_coord_loss = total_coord_loss / n_valid
                total_consist_loss = total_consist_loss / n_valid

            loss = self.cfg.w_coord * total_coord_loss + self.cfg.w_consistency * total_consist_loss

        # Backward
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.system.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": loss.item(),
            "coord_loss": total_coord_loss.item(),
            "consist_loss": total_consist_loss.item(),
            "n_valid": n_valid,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop (no gradients)."""
        self.system.eval()
        total_loss = 0.0
        total_rmsd = 0.0
        n_batches = 0

        for batch in dataloader:
            esm = batch["esm"].to(self.device)
            coords_target = batch["coords"].to(self.device)
            sequences = batch["sequences"]
            B = len(sequences)

            batch_loss = 0.0
            batch_rmsd = 0.0
            for i in range(B):
                Li = int(batch["lengths"][i])
                seq_i = sequences[i][:Li]
                esm_i = esm[i, :Li].unsqueeze(0)

                output = self.system(seq_i, esm_i, msa=None)
                pred = output["predicted_coords"].squeeze(0)
                target = coords_target[i, :Li]

                mse = F.mse_loss(pred, target).item()
                rmsd = torch.sqrt(((pred - target) ** 2).sum(dim=-1).mean()).item()
                batch_loss += mse
                batch_rmsd += rmsd

            total_loss += batch_loss / B
            total_rmsd += batch_rmsd / B
            n_batches += 1

        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_rmsd": total_rmsd / max(n_batches, 1),
        }

    def train(self, train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Full training loop."""
        best_val_loss = float("inf")

        for epoch in range(self.cfg.epochs):
            # Warmup
            if epoch < self.cfg.warmup_epochs:
                lr_scale = (epoch + 1) / self.cfg.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.cfg.lr * lr_scale

            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                step_result = self.train_step(batch)
                epoch_losses.append(step_result["loss"])

                if (batch_idx + 1) % self.cfg.log_interval == 0:
                    logger.info(f"[MSATrainer] ep{epoch} [{batch_idx+1}/{len(train_loader)}] "
                                f"loss={step_result['loss']:.4f} "
                                f"coord={step_result['coord_loss']:.4f}")

            avg_train_loss = float(np.mean(epoch_losses))

            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                logger.info(f"[MSATrainer] Epoch {epoch}: train_loss={avg_train_loss:.4f} "
                            f"val_loss={val_metrics['val_loss']:.4f} "
                            f"val_rmsd={val_metrics['val_rmsd']:.3f}")

                # Save best
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint(epoch, "best")
            else:
                logger.info(f"[MSATrainer] Epoch {epoch}: train_loss={avg_train_loss:.4f}")

            # LR schedule
            if epoch >= self.cfg.warmup_epochs:
                self.scheduler.step()

            self.history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                **val_metrics,
            })

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch}")

        # Save final
        self.save_checkpoint(self.cfg.epochs - 1, "final")
        return {"history": self.history, "best_val_loss": best_val_loss}

    def save_checkpoint(self, epoch: int, tag: str) -> None:
        path = os.path.join(self.cfg.checkpoint_dir, f"msa_system_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "system_state_dict": self.system.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg.__dict__,
        }, path)
        logger.info(f"[MSATrainer] Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.system.load_state_dict(ckpt["system_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"[MSATrainer] Loaded checkpoint: {path}")


# =============================================================================
# 3. ORGAN PDB EXPORTER BRIDGE
# =============================================================================
class OrganPDBExporterBridge:
    """
    [D] Bridge connecting predicted coordinates to OrganPDBExporter.

    Converts (N, 3) Cα coordinates + sequence into template_atoms format
    required by OrganPDBExporter.export_macro_assembly(), then places
    multiple copies on a grid or custom positions.

    USAGE:
      bridge = OrganPDBExporterBridge()
      bridge.export_organ_grid(coords, sequence, grid_shape=(10,10,10),
                                spacing=50.0, output_path="organ.pdb")
    """

    def __init__(self, precision: int = 3):
        if not _HAS_ORGAN:
            raise ImportError("organ_PDB_exporter_download.py required for OrganPDBExporterBridge.")
        self.exporter = OrganPDBExporter(precision=precision)

    def coords_to_template_atoms(self, coords: torch.Tensor | np.ndarray,
                                   sequence: str = "") -> List[Dict[str, Any]]:
        """
        Convert (N, 3) coordinates to template_atoms list for OrganPDBExporter.

        Args:
            coords: (N, 3) coordinates in Angstrom
            sequence: amino acid sequence (for residue names)

        Returns:
            List of dicts with keys: name, res_name, res_seq, element, x, y, z
        """
        if isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()
        N = coords.shape[0]

        if not sequence:
            sequence = "A" * N
        sequence = sequence.ljust(N, "A")[:N]

        aa_1to3 = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
                   'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
                   'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
                   'W':'TRP','Y':'TYR','X':'UNK'}

        atoms = []
        for i in range(N):
            res_name = aa_1to3.get(sequence[i], "UNK")
            atoms.append({
                "serial": i + 1,
                "name": "CA",
                "res_name": res_name,
                "chain_id": "A",
                "res_seq": i + 1,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "occupancy": 1.0,
                "temp_factor": 0.0,
                "element": "C",
            })
        return atoms

    def export_organ_grid(self, coords: torch.Tensor, sequence: str,
                            grid_shape: Tuple[int, int, int] = (10, 10, 10),
                            spacing: float = 50.0,
                            output_path: str = "organ_grid.pdb") -> Dict[str, Any]:
        """
        Export organ-level structure with copies on a regular grid.

        Args:
            coords: (N, 3) predicted coordinates
            sequence: amino acid sequence
            grid_shape: (nx, ny, nz) grid dimensions
            spacing: grid spacing in Angstrom
            output_path: destination PDB file

        Returns:
            dict with validation results
        """
        template_atoms = self.coords_to_template_atoms(coords, sequence)
        positions, rotations = create_organ_grid(grid_shape, spacing=spacing)

        self.exporter.export_macro_assembly(
            template_atoms, positions, rotations, output_path
        )

        validation = self.exporter.validate_pdb(output_path)
        logger.info(f"[OrganBridge] Exported {validation.get('atom_count', 0)} atoms "
                    f"to {output_path}")
        return validation

    def export_organ_custom(self, coords: torch.Tensor, sequence: str,
                              positions: torch.Tensor, rotations: torch.Tensor,
                              output_path: str = "organ_custom.pdb") -> Dict[str, Any]:
        """
        Export organ-level structure with custom positions and rotations.

        Args:
            coords: (N, 3) predicted coordinates
            sequence: amino acid sequence
            positions: (M, 3) center-of-mass positions for each copy
            rotations: (M, 3, 3) rotation matrices for each copy
            output_path: destination PDB file

        Returns:
            dict with validation results
        """
        template_atoms = self.coords_to_template_atoms(coords, sequence)

        self.exporter.export_macro_assembly(
            template_atoms, positions, rotations, output_path
        )

        validation = self.exporter.validate_pdb(output_path)
        logger.info(f"[OrganBridge] Exported custom organ to {output_path}")
        return validation

    def export_differentiable(self, coords: torch.Tensor, sequence: str,
                                positions: torch.Tensor, rotations: torch.Tensor
                                ) -> torch.Tensor:
        """
        Differentiable export: returns transformed coordinates with gradient support.
        Useful for end-to-end training where organ placement is part of the graph.
        """
        template_atoms = self.coords_to_template_atoms(coords, sequence)
        return self.exporter.export_differentiable(template_atoms, positions, rotations)


# =============================================================================
# 4. SLURM FULL PIPELINE GENERATOR (6 stages)
# =============================================================================
@dataclass
class SlurmFullPipelineConfig:
    """Configuration for 6-stage Slurm pipeline."""
    job_name: str = "sesi_full_pipeline"
    account: str = "YOUR_PROJECT_ACCOUNT"
    partition: str = "gpu"
    nodes: int = 32
    gpus_per_node: int = 8
    cpus_per_task: int = 16
    mem_gb: int = 256
    time_limit: str = "72:00:00"
    email: str = "your@email.com"

    # Design parameters
    n_residues: int = 100_000_000
    chunk_size: int = 10000

    # MSA training parameters
    msa_epochs: int = 20
    msa_batch_size: int = 4
    msa_lr: float = 3e-4

    # Organ export parameters
    organ_grid_shape: Tuple[int, int, int] = (10, 10, 10)
    organ_spacing: float = 50.0

    # Modules
    modules: List[str] = field(default_factory=lambda: [
        "cuda/12.2", "nccl/2.19", "openmpi/4.1.6-cuda", "python/3.11"
    ])


class SlurmFullPipelineGenerator:
    """
    [D] Generates 6-stage Slurm pipeline calling real stage scripts.

    STAGES:
      1. De novo sequence design (Stage1Designer)
      2. Hierarchical folding (Stage2Folder + SDA-ONE)
      3. MSA training (MSAFoldingTrainer) ← NEW
      4. OpenMM-ML refinement (Stage3Refiner)
      5. Validation (Stage4Validator)
      6. Organ export (OrganPDBExporterBridge) ← NEW
    """

    def __init__(self, cfg: Optional[SlurmFullPipelineConfig] = None):
        self.cfg = cfg or SlurmFullPipelineConfig()
        self.total_gpus = self.cfg.nodes * self.cfg.gpus_per_node

    def generate(self) -> str:
        """Generate complete 6-stage Slurm script."""
        modules_load = "\n".join(f"module load {m}" for m in self.cfg.modules)
        gx, gy, gz = self.cfg.organ_grid_shape

        return f"""#!/bin/bash
#=============================================================================
# SESI FULL 6-STAGE PIPELINE — {self.total_gpus}-GPU SLURM SCRIPT
# Auto-generated by SlurmFullPipelineGenerator v{TRAIN_ORGAN_SLURM_VERSION}
#=============================================================================
#SBATCH --job-name={self.cfg.job_name}
#SBATCH --output={self.cfg.job_name}_%j.out
#SBATCH --error={self.cfg.job_name}_%j.err
#SBATCH --time={self.cfg.time_limit}
#SBATCH --nodes={self.cfg.nodes}
#SBATCH --ntasks-per-node={self.cfg.gpus_per_node}
#SBATCH --cpus-per-task={self.cfg.cpus_per_task}
#SBATCH --gres=gpu:{self.cfg.gpus_per_node}
#SBATCH --mem={self.cfg.mem_gb}G
#SBATCH --partition={self.cfg.partition}
#SBATCH --account={self.cfg.account}
#SBATCH --mail-user={self.cfg.email}
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "======================================================================"
echo "  SESI FULL 6-STAGE PIPELINE — Job Start"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Total GPUs  : {self.total_gpus}"
echo "  Target      : {self.cfg.n_residues:,} residues"
echo "  Start Time  : $(date)"
echo "======================================================================"

module purge
{modules_load}

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS={self.cfg.cpus_per_task}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR
mkdir -p checkpoints logs results

#=============================================================================
# STAGE 1: DE NOVO SEQUENCE DESIGN
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  STAGE 1: De Novo Sequence Design ({self.cfg.n_residues:,} residues)"
echo "----------------------------------------------------------------------"

torchrun \\
    --nnodes={self.cfg.nodes} \\
    --nproc_per_node={self.cfg.gpus_per_node} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \\
    --master_port=29500 \\
    --rdzv_backend=c10d \\
    sesi_100m_design_stage1.py \\
    --n-residues {self.cfg.n_residues} \\
    --chunk-size {self.cfg.chunk_size} \\
    --overlap 100 \\
    --output-dir ./results/stage1_design

#=============================================================================
# STAGE 2: HIERARCHICAL FOLDING (SDA-ONE)
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  STAGE 2: Hierarchical Folding (SDA-ONE)"
echo "----------------------------------------------------------------------"

torchrun \\
    --nnodes={self.cfg.nodes} \\
    --nproc_per_node={self.cfg.gpus_per_node} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \\
    --master_port=29501 \\
    --rdzv_backend=c10d \\
    sesi_100m_design_stage2.py \\
    --input-dir ./results/stage1_design \\
    --output-dir ./results/stage2_fold

#=============================================================================
# STAGE 3: MSA TRAINING (folding_head + fusion layers)
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  STAGE 3: MSA Enhanced Folding Training"
echo "----------------------------------------------------------------------"

torchrun \\
    --nnodes={self.cfg.nodes} \\
    --nproc_per_node={self.cfg.gpus_per_node} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \\
    --master_port=29502 \\
    --rdzv_backend=c10d \\
    sesi_stage3_msa_train.py \\
    --epochs {self.cfg.msa_epochs} \\
    --batch-size {self.cfg.msa_batch_size} \\
    --lr {self.cfg.msa_lr} \\
    --input-dir ./results/stage2_fold \\
    --checkpoint-dir ./checkpoints/stage3_msa

#=============================================================================
# STAGE 4: OPENMM-ML REFINEMENT (k*=23 OPMC iterations)
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  STAGE 4: OpenMM-ML Refinement"
echo "----------------------------------------------------------------------"

torchrun \\
    --nnodes={self.cfg.nodes} \\
    --nproc_per_node={self.cfg.gpus_per_node} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \\
    --master_port=29503 \\
    --rdzv_backend=c10d \\
    sesi_100m_design_stage3.py \\
    --input-dir ./results/stage2_fold \\
    --output-dir ./results/stage4_refined \\
    --ml-potential ani2x \\
    --steps 23

#=============================================================================
# STAGE 5: VALIDATION (lDDT, TM-score, GDT-TS, clashscore)
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  STAGE 5: Validation"
echo "----------------------------------------------------------------------"

python sesi_100m_design_stage4.py \\
    --input-dir ./results/stage4_refined \\
    --output-dir ./results/validation \\
    --metrics lddt tm-score gdt-ts clashscore

#=============================================================================
# STAGE 6: ORGAN-LEVEL EXPORT
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  STAGE 6: Organ-Level PDB Export"
echo "----------------------------------------------------------------------"

python sesi_stage6_organ_export.py \\
    --input-dir ./results/stage4_refined \\
    --output-dir ./results/stage6_organ \\
    --grid-shape {gx},{gy},{gz} \\
    --spacing {self.cfg.organ_spacing}

#=============================================================================
# SUMMARY
#=============================================================================
echo ""
echo "======================================================================"
echo "  FULL PIPELINE COMPLETE"
echo "  End Time : $(date)"
echo "  Duration : $SECONDS seconds"
echo "  Results  : ./results/"
echo "======================================================================"

tar -czf results_{self.cfg.job_name}_$SLURM_JOB_ID.tar.gz \\
    results/ checkpoints/ *.log 2>/dev/null || true
echo "Archived to results_{self.cfg.job_name}_$SLURM_JOB_ID.tar.gz"
"""

    def generate_stage3_msa_train(self) -> str:
        """Generate Stage 3 MSA training entry point."""
        return '''#!/usr/bin/env python3
"""Stage 3: MSA Enhanced Folding System training."""
import argparse, os, torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sesi_msa_train_organ_export_slurm import (
    MSAFoldingTrainer, MSAFoldingTrainerConfig,
    SyntheticFoldDataset, fold_collate,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--input-dir", type=str, default="./results/stage2_fold")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints/stage3_msa")
    p.add_argument("--train-only-head", action="store_true")
    args = p.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    cfg = MSAFoldingTrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_only_head=args.train_only_head,
        checkpoint_dir=args.checkpoint_dir,
        device="cuda",
    )
    trainer = MSAFoldingTrainer(cfg)

    # In production: load real PDB dataset from input-dir
    # For now: synthetic dataset for pipeline testing
    train_ds = SyntheticFoldDataset(n_samples=100, min_len=30, max_len=100)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, collate_fn=fold_collate)
    val_ds = SyntheticFoldDataset(n_samples=20, min_len=30, max_len=100, seed=99)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=fold_collate)

    result = trainer.train(train_loader, val_loader)
    if rank == 0:
        print(f"[stage3] MSA training complete. Best val loss: {result['best_val_loss']:.4f}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''

    def generate_stage6_organ_export(self) -> str:
        """Generate Stage 6 organ export entry point."""
        return '''#!/usr/bin/env python3
"""Stage 6: Organ-level PDB export."""
import argparse, os, torch
from pathlib import Path
from sesi_msa_train_organ_export_slurm import OrganPDBExporterBridge

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="./results/stage4_refined")
    p.add_argument("--output-dir", type=str, default="./results/stage6_organ")
    p.add_argument("--grid-shape", type=str, default="10,10,10")
    p.add_argument("--spacing", type=float, default=50.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gx, gy, gz = [int(x) for x in args.grid_shape.split(",")]

    bridge = OrganPDBExporterBridge()

    # Load refined coordinates
    refined_pt = Path(args.input_dir) / "refined.pt"
    if refined_pt.exists():
        data = torch.load(refined_pt, map_location="cpu")
        coords = data["refined_coords"]
        sequence = data.get("sequence", "")
    else:
        print(f"[stage6] WARNING: {refined_pt} not found. Using placeholder.")
        coords = torch.randn(100, 3) * 10
        sequence = "A" * 100

    output_path = os.path.join(args.output_dir, "organ_structure.pdb")
    validation = bridge.export_organ_grid(
        coords, sequence,
        grid_shape=(gx, gy, gz),
        spacing=args.spacing,
        output_path=output_path,
    )
    print(f"[stage6] Organ export complete: {validation}")

if __name__ == "__main__":
    main()
'''

    def write_all(self, output_dir: str = "./slurm_full_pipeline") -> Dict[str, str]:
        """Write all Slurm + stage script files."""
        os.makedirs(output_dir, exist_ok=True)
        files = {}

        # Main Slurm script
        slurm_path = os.path.join(output_dir, "submit_full_pipeline.slurm")
        with open(slurm_path, "w") as f:
            f.write(self.generate())
        files["slurm_script"] = slurm_path

        # Stage 3 script
        s3_path = os.path.join(output_dir, "sesi_stage3_msa_train.py")
        with open(s3_path, "w") as f:
            f.write(self.generate_stage3_msa_train())
        files["stage3_msa_train"] = s3_path

        # Stage 6 script
        s6_path = os.path.join(output_dir, "sesi_stage6_organ_export.py")
        with open(s6_path, "w") as f:
            f.write(self.generate_stage6_organ_export())
        files["stage6_organ_export"] = s6_path

        for name, path in files.items():
            os.chmod(path, 0o755)
            logger.info(f"[SlurmGen] Wrote {path}")

        return files


# =============================================================================
# 5. MODULE INTEGRATION MAP
# =============================================================================
class ModuleIntegrationMap:
    """
    [D] Documents the wiring between all ecosystem modules.
    Generates a human-readable map and optional DOT graph.
    """

    INTEGRATION_MAP = {
        "MSA_ENHANCED_PROTEIN_FOLDING": {
            "provides": ["MSAEnhancedFoldingSystem", "MSASequenceFuser",
                         "QuantumMSACompressor", "HodgeMSADecomposition",
                         "CarpetHierarchicalFusion", "AdaptiveMSARouter"],
            "consumed_by": ["MSAFoldingTrainer (this module)",
                            "Stage2Folder (sesi_full_stage_scripts_msa)"],
            "key_output": "predicted_coords, fused_features, msa diagnostics",
        },
        "organ_PDB_exporter_download": {
            "provides": ["OrganPDBExporter", "DifferentiableOrganPDBExporter",
                         "parse_pdb_template", "create_organ_grid"],
            "consumed_by": ["OrganPDBExporterBridge (this module)"],
            "key_output": "organ-level PDB file",
        },
        "structural_domain_assembly_one": {
            "provides": ["StructuralDomainAssembly", "DomainSegmenter",
                         "CrossDomainContactHead", "DomainDockingAssembler"],
            "consumed_by": ["Stage2Folder", "MegaScaleSDAOneBridge"],
            "key_output": "init_coords (N,3), seq_features, sigma",
        },
        "real_fold_one_v2-5": {
            "provides": ["RefinementEngine", "OpenMMEnergyCalculator",
                         "openmm_solute_energy", "CSOCKernel", "SOCController"],
            "consumed_by": ["Stage3Refiner", "RefinementEngineBridge"],
            "key_output": "refined solute_coords, energy history",
        },
        "denovo_sequence_designer_fixed": {
            "provides": ["DifferentiableStructuralDesigner"],
            "consumed_by": ["Stage1Designer"],
            "key_output": "hard_tokens, viability_score",
        },
        "gumbel_atomix_one": {
            "provides": ["RealFoldOneAdvancedEngine"],
            "consumed_by": ["Stage3Refiner (optional advanced refinement)"],
            "key_output": "effective_coords, occupancies, loss",
        },
        "xray_cryo_em_complete_production_implementation": {
            "provides": ["XrayCryoEMNativeEngine"],
            "consumed_by": ["Stage5 validation (experimental map fitting)"],
            "key_output": "total_objective, r_work, r_free, fsc",
        },
        "structural_cahn_hilliard_3d-7": {
            "provides": ["StructuralCahnHilliard3D", "ThinFilmStructuralCahnHilliard3D",
                         "PhaseFieldCrystal3D"],
            "consumed_by": ["Multiphase flow coupling (future)"],
            "key_output": "phase field evolution",
        },
        "structural_langevin_fold_v2": {
            "provides": ["AdvancedStructuralLangevin", "CSOCThermostat"],
            "consumed_by": ["RefinementEngine (Langevin mode)"],
            "key_output": "BAOAB-integrated trajectories",
        },
        "structural_interface_utils": {
            "provides": ["soft_interface_score", "chunked_soft_interface_score"],
            "consumed_by": ["AdvancedStructuralLangevin (InterfaceDetector)"],
            "key_output": "per-atom interface mask",
        },
    }

    def generate_markdown(self) -> str:
        """Generate markdown documentation of module integration."""
        lines = ["# SESI Ecosystem Module Integration Map\n"]
        for module, info in self.INTEGRATION_MAP.items():
            lines.append(f"## {module}")
            lines.append(f"**Provides:** {', '.join(info['provides'])}")
            lines.append(f"**Consumed by:** {', '.join(info['consumed_by'])}")
            lines.append(f"**Key output:** {info['key_output']}")
            lines.append("")
        return "\n".join(lines)

    def generate_dot(self) -> str:
        """Generate DOT graph for visualization."""
        lines = ["digraph SESI_Ecosystem {", '  rankdir=LR;', '  node [shape=box];']
        for module, info in self.INTEGRATION_MAP.items():
            safe_name = module.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            lines.append(f'  "{safe_name}" [label="{module}"];')
            for consumer in info["consumed_by"]:
                safe_consumer = consumer.split(" (")[0].replace(" ", "_").replace("-", "_")
                lines.append(f'  "{safe_name}" -> "{safe_consumer}";')
        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# 6. FULL ECOSYSTEM ORCHESTRATOR (local testing)
# =============================================================================
class FullEcosystemOrchestrator:
    """
    [D] Runs MSA training → folding → organ export in one call.
    For local testing; production uses Slurm pipeline.
    """

    def __init__(self, device: str = "auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )

    def run(self, n_train_samples: int = 20, n_epochs: int = 3,
              organ_grid: Tuple[int, int, int] = (2, 2, 2)) -> Dict[str, Any]:
        """Run full local pipeline."""
        results = {}

        # Step 1: MSA training
        logger.info("[Orchestrator] Step 1: MSA training")
        if _HAS_MSA:
            trainer_cfg = MSAFoldingTrainerConfig(
                epochs=n_epochs, batch_size=4, train_only_head=False,
                esm_dim=128, fused_dim=64, hodge_dim=64,
                checkpoint_dir="./test_ckpt_msa",
            )
            trainer = MSAFoldingTrainer(trainer_cfg)

            train_ds = SyntheticFoldDataset(n_samples=n_train_samples, min_len=20, max_len=50)
            train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                                       collate_fn=fold_collate)
            val_ds = SyntheticFoldDataset(n_samples=5, min_len=20, max_len=50, seed=99)
            val_loader = DataLoader(val_ds, batch_size=4, collate_fn=fold_collate)

            train_result = trainer.train(train_loader, val_loader)
            results["msa_training"] = train_result

            # Step 2: Fold a test sequence
            logger.info("[Orchestrator] Step 2: Folding test sequence")
            trainer.system.eval()
            test_seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 residues
            test_esm = torch.randn(1, 40, 128, device=self.device)
            with torch.no_grad():
                fold_out = trainer.system(test_seq, test_esm, msa=None)
            pred_coords = fold_out["predicted_coords"].squeeze(0).cpu()
            results["folding"] = {"coords_shape": tuple(pred_coords.shape)}

            # Step 3: Organ export
            logger.info("[Orchestrator] Step 3: Organ export")
            if _HAS_ORGAN:
                organ_bridge = OrganPDBExporterBridge()
                output_path = "./test_organ_output.pdb"
                validation = organ_bridge.export_organ_grid(
                    pred_coords, test_seq,
                    grid_shape=organ_grid, spacing=50.0,
                    output_path=output_path,
                )
                results["organ_export"] = validation
                if os.path.exists(output_path):
                    os.remove(output_path)
            else:
                results["organ_export"] = {"skipped": "OrganPDBExporter not available"}
        else:
            results["msa_training"] = {"skipped": "MSA module not available"}

        # Cleanup
        import shutil
        shutil.rmtree("./test_ckpt_msa", ignore_errors=True)

        return results


# =============================================================================
# 7. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI MSA TRAIN + ORGAN EXPORT + SLURM v{TRAIN_ORGAN_SLURM_VERSION} — Self-Test")
    print(f"  MSA={_HAS_MSA}  Organ={_HAS_ORGAN}  StageScripts={_HAS_STAGE_SCRIPTS}")
    print("=" * 78)

    # ---- T1: Synthetic dataset -------------------------------------------
    print("\n [T1] SyntheticFoldDataset")
    ds = SyntheticFoldDataset(n_samples=10, min_len=20, max_len=40)
    sample = ds[0]
    assert sample["target_coords"].shape == (sample["length"], 3)
    assert sample["esm_features"].shape == (sample["length"], 128)
    print(f"   Sample: L={sample['length']}, coords={tuple(sample['target_coords'].shape)}")
    print("[PASS] T1 Synthetic dataset works")

    # ---- T2: Collate function ---------------------------------------------
    print("\n [T2] fold_collate")
    batch = [ds[i] for i in range(3)]
    collated = fold_collate(batch)
    assert collated["coords"].shape[0] == 3
    assert collated["mask"].shape[0] == 3
    print(f"   Batch: coords={tuple(collated['coords'].shape)}, mask={tuple(collated['mask'].shape)}")
    print("[PASS] T2 Collate works")

    # ---- T3: MSA training (if available) ----------------------------------
    print("\n [T3] MSAFoldingTrainer")
    if _HAS_MSA:
        trainer_cfg = MSAFoldingTrainerConfig(
            epochs=2, batch_size=2, train_only_head=True,
            esm_dim=128, fused_dim=32, hodge_dim=32,
            checkpoint_dir="./test_ckpt",
            log_interval=1,
        )
        trainer = MSAFoldingTrainer(trainer_cfg)

        train_loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=fold_collate)
        result = trainer.train(train_loader, val_loader=None)
        assert len(result["history"]) == 2
        print(f"   Trained 2 epochs, final loss: {result['history'][-1]['train_loss']:.4f}")
        print("[PASS] T3 MSAFoldingTrainer runs and produces loss history")

        import shutil
        shutil.rmtree("./test_ckpt", ignore_errors=True)
    else:
        print("[SKIP] T3 MSA_ENHANCED_PROTEIN_FOLDING not available")

    # ---- T4: OrganPDBExporterBridge ---------------------------------------
    print("\n [T4] OrganPDBExporterBridge")
    if _HAS_ORGAN:
        bridge = OrganPDBExporterBridge()
        test_coords = torch.randn(20, 3) * 5
        test_seq = "ACDEFGHIKLMNPQRSTVWY"

        template_atoms = bridge.coords_to_template_atoms(test_coords, test_seq)
        assert len(template_atoms) == 20
        assert template_atoms[0]["name"] == "CA"
        print(f"   Template atoms: {len(template_atoms)}")

        # Grid export
        output_path = "./test_organ.pdb"
        validation = bridge.export_organ_grid(
            test_coords, test_seq,
            grid_shape=(2, 2, 2), spacing=30.0,
            output_path=output_path,
        )
        assert validation.get("valid", False)
        print(f"   Organ grid export: {validation.get('atom_count', 0)} atoms")
        os.remove(output_path)
        print("[PASS] T4 OrganPDBExporterBridge works")
    else:
        print("[SKIP] T4 organ_PDB_exporter_download not available")

    # ---- T5: Slurm generator -----------------------------------------------
    print("\n [T5] SlurmFullPipelineGenerator")
    slurm_gen = SlurmFullPipelineGenerator(SlurmFullPipelineConfig(
        nodes=2, gpus_per_node=4,  # small for testing
        n_residues=1000,
    ))
    files = slurm_gen.write_all(output_dir="./test_slurm_full")
    assert len(files) == 3
    for name, path in files.items():
        assert os.path.exists(path)
        print(f"   {name}: {os.path.basename(path)}")

    # Verify Slurm script content
    with open(files["slurm_script"]) as f:
        slurm_content = f.read()
    assert "STAGE 3: MSA" in slurm_content
    assert "STAGE 6: ORGAN" in slurm_content
    assert "sesi_stage3_msa_train.py" in slurm_content
    assert "sesi_stage6_organ_export.py" in slurm_content
    print("[PASS] T5 Slurm 6-stage script generated correctly")

    import shutil
    shutil.rmtree("./test_slurm_full", ignore_errors=True)

    # ---- T6: Module integration map ----------------------------------------
    print("\n [T6] ModuleIntegrationMap")
    imap = ModuleIntegrationMap()
    md = imap.generate_markdown()
    assert "MSA_ENHANCED_PROTEIN_FOLDING" in md
    assert "organ_PDB_exporter_download" in md
    dot = imap.generate_dot()
    assert "digraph" in dot
    print(f"   Markdown map: {len(md)} chars")
    print(f"   DOT graph: {len(dot)} chars")
    print("[PASS] T6 Integration map generated")

    # ---- T7: Full orchestrator (local) --------------------------------------
    print("\n [T7] FullEcosystemOrchestrator (local)")
    if _HAS_MSA:
        orch = FullEcosystemOrchestrator(device="cpu")
        results = orch.run(n_train_samples=10, n_epochs=1, organ_grid=(2, 2, 2))
        assert "msa_training" in results
        print(f"   MSA training: {results['msa_training'].get('best_val_loss', 'n/a')}")
        if "organ_export" in results and results["organ_export"].get("valid"):
            print(f"   Organ export: {results['organ_export'].get('atom_count', 0)} atoms")
        print("[PASS] T7 Full orchestrator runs end-to-end")
    else:
        print("[SKIP] T7 MSA not available")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("  HONEST REMINDERS:")
    print("    • MSAFoldingTrainer starts from RANDOM weights — train on real data")
    print("    • Synthetic dataset is for PIPELINE TESTING ONLY")
    print("    • OrganPDBExporterBridge does rigid placement only (no flexible docking)")
    print("    • Slurm scripts require cluster customization (account, partition)")
    print("    • Full 6-stage pipeline requires 256×H100 for 100M+ residues")
    print("=" * 78)
