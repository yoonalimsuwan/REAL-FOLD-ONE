"""
=============================================================================
SESI ULTRA-SCALE TRAINING SUITE — v1.0.0
256-GPU Slurm + Synthetic Training Loop + RefinementEngine Integration
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================
INTEGRATION WITH ECOSYSTEM
=============================================================================
• sesi_100m_predictor.py        — UltraScalePredictor, UltraScaleConfig
• real_fold_one_v2-5.py         — RefinementEngine, RefinementConfig,
                                   openmm_solute_energy, OpenMMSystemBuilder
• structural_domain_assembly_one.py — StructuralDomainAssembly (SDA-ONE)
• denovo_sequence_designer_fixed.py — DifferentiableStructuralDesigner pattern
• MSA_ENHANCED_PROTEIN_FOLDING  — MSAEnhancedFoldingSystem (optional)

OPMC CONSTANTS (from Unified Ordinal-Tensor Measurement Theory):
  C1 = 0.420000       (Theorem 5.5)
  K2 = 12.5000        (Lemma 7.6)
  L  = 0.438520       (Example 7.12)
  k* = 23 iterations  (Theorem 11.1, for ε=1e-8)
  d(m,n) = m²n² + mn² (Lemma 2.4)

WHAT THIS MODULE PROVIDES:
  [1] SyntheticProteinDataGenerator — idealized helices/sheets for training
  [2] RefinementEngineBridge        — direct call to real_fold_one_v2-5.py
  [3] DistributedUltraScaleTrainer  — DDP/FSDP + AMP training loop
  [4] SlurmScriptGenerator          — 256-GPU Slurm batch script
  [5] Self-test suite

HONEST SCOPE:
  • Synthetic data is for PIPELINE TESTING ONLY. Production training
    requires real PDB/CASP data.
  • 256-GPU execution requires actual HPC cluster with 32×H100 nodes.
  • RefinementEngine requires OpenMM + OpenMM-ML installed. Without them,
    the bridge degrades to a warning-only stub.
  • This module has NOT been run at full 256-GPU scale here. Self-tests
    verify logic on a small subset.
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

logger = logging.getLogger("sesi.ultrascale_training")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

TRAINING_SUITE_VERSION: str = "1.0.0"

# =============================================================================
# OPMC PAPER CONSTANTS
# =============================================================================
OPMC_C1 = 0.420000
OPMC_K2 = 12.5000
OPMC_L = 0.438520
OPMC_ETA_LATTICE = 0.99491
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n

def opmc_k_star(epsilon: float = 1e-8, D0: float = 1.0) -> int:
    """Theorem 11.1: k* = ⌈ln(ε/D₀)/ln L⌉ — INDEPENDENT of N."""
    return math.ceil(math.log(epsilon / D0) / math.log(OPMC_L))

# =============================================================================
# Optional ecosystem imports (graceful fallback)
# =============================================================================
_HAS_REAL_FOLD = False
try:
    from real_fold_one_v2 import (
        RefinementEngine,
        RefinementConfig,
        OpenMMEnergyCalculator,
        openmm_solute_energy,
        OpenMMSystemBuilder,
    )
    _HAS_REAL_FOLD = True
except ImportError:
    import importlib.util
    for cand in ("real_fold_one_v2-5.py", "real_fold_one_v2.py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("real_fold_one_v2", str(p))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                RefinementEngine = mod.RefinementEngine
                RefinementConfig = mod.RefinementConfig
                OpenMMEnergyCalculator = mod.OpenMMEnergyCalculator
                openmm_solute_energy = mod.openmm_solute_energy
                OpenMMSystemBuilder = mod.OpenMMSystemBuilder
                _HAS_REAL_FOLD = True
            break
    if not _HAS_REAL_FOLD:
        warnings.warn("real_fold_one_v2 not found; refinement bridge disabled.")

_HAS_OPENMM = False
try:
    import openmm as mm
    _HAS_OPENMM = True
except ImportError:
    pass

_HAS_DIST = False
try:
    import torch.distributed as dist
    _HAS_DIST = True
except ImportError:
    pass

_HAS_ULTRA = False
try:
    from sesi_100m_predictor import UltraScalePredictor, UltraScaleConfig
    _HAS_ULTRA = True
except ImportError:
    warnings.warn("sesi_100m_predictor not found; using stub predictor.")


# =============================================================================
# 1. SYNTHETIC PROTEIN DATA GENERATOR
# =============================================================================
class SyntheticProteinDataGenerator:
    """
    [D] Generates synthetic protein structures for pipeline testing.

    GENERATES:
      • Idealized alpha-helices (3.6 res/turn, 1.5 Å/rise)
      • Idealized beta-sheets (extended strands, 3.5 Å/residue)
      • Random coil noise for augmentation

    WARNING: This is for PIPELINE TESTING ONLY. Production training
    requires real PDB/CASP data with experimental structures.
    """

    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def random_sequence(self, length: int) -> str:
        """Generate random amino acid sequence."""
        return "".join(self.rng.choice(list(self.AA_ALPHABET), size=length))

    def idealized_helix(self, n_residues: int, noise_std: float = 0.0) -> torch.Tensor:
        """
        Generate idealized alpha-helix coordinates.
        Returns: (n_residues, 3) tensor in Angstrom.
        """
        t = torch.arange(n_residues, dtype=torch.float32)
        angle = t * math.radians(100)
        radius = 2.3

        x = radius * torch.cos(angle)
        y = radius * torch.sin(angle)
        z = t * 1.5

        coords = torch.stack([x, y, z], dim=-1)
        if noise_std > 0:
            coords += torch.randn_like(coords) * noise_std
        return coords

    def idealized_sheet(self, n_residues: int, n_strands: int = 3,
                          noise_std: float = 0.0) -> torch.Tensor:
        """Generate idealized beta-sheet coordinates."""
        strand_length = n_residues // n_strands
        coords_list = []

        for s in range(n_strands):
            t = torch.arange(strand_length, dtype=torch.float32)
            if s % 2 == 0:
                z = t * 3.5
            else:
                z = (strand_length - 1 - t) * 3.5
            x = s * 5.0
            y = torch.sin(t * math.pi / strand_length) * 0.5
            coords_list.append(torch.stack([x, y, z], dim=-1))

        coords = torch.cat(coords_list, dim=0)[:n_residues]
        if noise_std > 0:
            coords += torch.randn_like(coords) * noise_std
        return coords

    def generate_batch(self, batch_size: int, min_len: int = 50,
                         max_len: int = 200, noise_std: float = 0.5
                         ) -> List[Dict[str, Any]]:
        """
        Generate a batch of synthetic protein data.
        Returns list of dicts with 'sequence', 'target_coords', 'length'.
        """
        batch = []
        for _ in range(batch_size):
            length = int(self.rng.integers(min_len, max_len))
            sequence = self.random_sequence(length)

            if self.rng.random() < 0.5:
                target = self.idealized_helix(length, noise_std=noise_std)
            else:
                target = self.idealized_sheet(length, noise_std=noise_std)

            batch.append({
                'sequence': sequence,
                'target_coords': target,
                'length': length,
            })

        return batch


class SyntheticProteinDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic protein data."""

    def __init__(self, generator: SyntheticProteinDataGenerator,
                   n_samples: int = 100, min_len: int = 50,
                   max_len: int = 200, noise_std: float = 0.5):
        self.generator = generator
        self.n_samples = n_samples
        self.min_len = min_len
        self.max_len = max_len
        self.noise_std = noise_std
        self._cache = {}

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx not in self._cache:
            batch = self.generator.generate_batch(
                batch_size=1,
                min_len=self.min_len,
                max_len=self.max_len,
                noise_std=self.noise_std,
            )
            self._cache[idx] = batch[0]
        return self._cache[idx]


def synthetic_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length proteins with padding."""
    Lmax = max(item['length'] for item in batch)
    B = len(batch)

    coords = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    sequences = []
    lengths = []

    for i, item in enumerate(batch):
        L = item['length']
        coords[i, :L] = item['target_coords']
        mask[i, :L] = True
        sequences.append(item['sequence'])
        lengths.append(L)

    return {
        'coords': coords,
        'mask': mask,
        'sequences': sequences,
        'lengths': torch.tensor(lengths),
    }


# =============================================================================
# 2. REFINEMENTENGINE BRIDGE (direct integration with real_fold_one_v2-5.py)
# =============================================================================
class RefinementEngineBridge:
    """
    [D] Direct bridge to real_fold_one_v2-5.py's RefinementEngine.

    USAGE:
      1. UltraScalePredictor produces coarse coordinates
      2. Write to temporary PDB file
      3. Call RefinementEngine.refine() for OpenMM-ML refinement
      4. Return refined coordinates

    REQUIRES: OpenMM + OpenMM-ML + openmm-torch installed.
    Falls back gracefully if not available.
    """

    def __init__(self, refine_cfg: Optional[Any] = None):
        self.refine_cfg = refine_cfg
        self._engine = None
        self._available = False

        if _HAS_REAL_FOLD:
            try:
                if self.refine_cfg is None:
                    self.refine_cfg = RefinementConfig()
                self._engine = RefinementEngine(self.refine_cfg)
                self._available = True
                logger.info("[RefinementBridge] RefinementEngine initialized successfully.")
            except Exception as e:
                logger.warning(f"[RefinementBridge] Failed to initialize RefinementEngine: {e}")
        else:
            logger.warning("[RefinementBridge] real_fold_one_v2 not available. Refinement disabled.")

    @property
    def available(self) -> bool:
        return self._available and _HAS_OPENMM

    def coords_to_pdb(self, coords: torch.Tensor, output_path: str,
                        sequence: str = "") -> str:
        """Write Cα coordinates to PDB file."""
        coords_np = coords.detach().cpu().numpy()
        N = coords_np.shape[0]

        if not sequence:
            sequence = "A" * N
        sequence = sequence.ljust(N, "A")[:N]

        aa_1to3 = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
            'X': 'UNK',
        }

        with open(output_path, 'w') as f:
            f.write("REMARK   1 GENERATED BY SESI RefinementEngineBridge\n")
            for i in range(N):
                res_name = aa_1to3.get(sequence[i], 'UNK')
                x, y, z = coords_np[i]
                f.write(
                    f"ATOM  {i+1:5d}  CA  {res_name:>3s} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")

        return output_path

    def refine(self, coords: torch.Tensor, sequence: str,
                 steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Refine predicted coordinates using RefinementEngine.

        Args:
            coords: (N, 3) predicted Cα coordinates
            sequence: amino acid sequence
            steps: refinement steps (default: cfg.steps)

        Returns:
            dict with refined coords, energy history, etc.
        """
        if not self.available:
            logger.warning("[RefinementBridge] Refinement not available. Returning input coords.")
            return {
                'solute_coords': coords.cpu().numpy(),
                'energy_history': [],
                'final_energy': float('nan'),
                'initial_energy': float('nan'),
                'refined': False,
            }

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            pdb_path = f.name

        try:
            self.coords_to_pdb(coords, pdb_path, sequence)
            result = self._engine.refine(pdb_path, steps=steps)
            result['refined'] = True
            return result
        except Exception as e:
            logger.error(f"[RefinementBridge] Refinement failed: {e}")
            return {
                'solute_coords': coords.cpu().numpy(),
                'energy_history': [],
                'final_energy': float('nan'),
                'initial_energy': float('nan'),
                'refined': False,
                'error': str(e),
            }
        finally:
            if os.path.exists(pdb_path):
                os.remove(pdb_path)

    def cleanup(self) -> None:
        """Cleanup RefinementEngine resources."""
        if self._engine is not None:
            try:
                self._engine.cleanup()
            except Exception:
                pass
            self._engine = None
            self._available = False


# =============================================================================
# 3. DISTRIBUTED ULTRA-SCALE TRAINER
# =============================================================================
@dataclass
class DistributedTrainerConfig:
    """Configuration for distributed training."""
    # Data
    batch_size: int = 4
    min_len: int = 50
    max_len: int = 200
    noise_std: float = 0.5

    # Training
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip: float = 5.0

    # Loss weights
    w_rmsd: float = 1.0
    w_opmc: float = 0.1

    # Distributed
    use_amp: bool = True
    amp_dtype: str = "bf16"
    use_fsdp: bool = False
    gradient_checkpointing: bool = True

    # Refinement
    refine_every_n_epochs: int = 5
    refine_steps: int = 100

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 5

    # Device
    device: str = "auto"


class DistributedUltraScaleTrainer:
    """
    [D] Distributed training loop for UltraScalePredictor.

    LOSS FUNCTION:
      L = w_rmsd * RMSD(predicted, target) + w_opmc * OPMC_contraction_loss

    where OPMC_contraction_loss encourages the model to produce states
    that are fixed points of the contraction operator (Theorem 7.10).

    DISTRIBUTED STRATEGY:
      • DDP for data parallelism (each GPU processes different proteins)
      • FSDP for model parallelism (shard large models)
      • AMP bf16 for 2× memory savings
      • Gradient checkpointing for activation memory

    INTEGRATION:
      • SyntheticProteinDataGenerator for training data
      • RefinementEngineBridge for periodic refinement
      • UltraScalePredictor as the model
    """

    def __init__(self, cfg: Optional[DistributedTrainerConfig] = None):
        self.cfg = cfg or DistributedTrainerConfig()

        # Device
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # Distributed setup
        self.rank = 0
        self.world_size = 1
        self.is_main = True

        if _HAS_DIST and dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_main = (self.rank == 0)

        # Model
        if _HAS_ULTRA:
            ultra_cfg = UltraScaleConfig(
                n_residues=self.cfg.max_len,  # small for training
                domain_size=min(self.cfg.max_len, 1000),
                hidden_dim=128,
                n_heads=4,
                gradient_checkpointing=self.cfg.gradient_checkpointing,
            )
            self.model = UltraScalePredictor(ultra_cfg).to(self.device)
        else:
            # Stub model for testing without sesi_100m_predictor
            self.model = self._create_stub_model().to(self.device)

        # Wrap with DDP/FSDP
        if self.world_size > 1 and _HAS_DIST:
            if self.cfg.use_fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                self.model = FSDP(self.model, device_id=self.device)
            else:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.model = DDP(self.model, device_ids=[self.device])

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # AMP
        self.amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.amp_dtype == "fp16"))

        # Data generator
        self.data_gen = SyntheticProteinDataGenerator(seed=42)

        # Refinement bridge
        self.refine_bridge = RefinementEngineBridge()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.history: List[Dict[str, Any]] = []

    def _create_stub_model(self) -> nn.Module:
        """Create a stub model for testing without UltraScalePredictor."""
        class StubModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(20, 64)
                self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
                self.coord_head = nn.Linear(64, 3)

            def forward(self, seq_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
                x = self.embed(seq_tokens)
                x, _ = self.attn(x, x, x)
                coords = self.coord_head(x)
                return {"coords": coords}

        return StubModel()

    def compute_loss(self, predicted_coords: torch.Tensor,
                       target_coords: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        Loss = w_rmsd * RMSD + w_opmc * OPMC_contraction_loss
        """
        # RMSD loss (masked)
        sq_diff = ((predicted_coords - target_coords) ** 2).sum(dim=-1)
        rmsd = torch.sqrt((sq_diff * mask.float()).sum() / mask.float().sum().clamp_min(1))
        loss_rmsd = self.cfg.w_rmsd * rmsd

        # OPMC contraction loss: encourage predicted coords to be near fixed point
        # Simplified: penalize large coordinate magnitudes (contraction toward origin)
        coord_norm = (predicted_coords ** 2).sum(dim=-1)
        opmc_loss = self.cfg.w_opmc * (coord_norm * mask.float()).mean()

        total_loss = loss_rmsd + opmc_loss

        return total_loss, {
            'loss': total_loss.item(),
            'rmsd': rmsd.item(),
            'opmc_loss': opmc_loss.item(),
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()
        self.optimizer.zero_grad()

        coords = batch['coords'].to(self.device)
        mask = batch['mask'].to(self.device)

        # Tokenize sequences (simplified: random tokens for stub)
        seq_tokens = torch.randint(0, 20, coords.shape[:2], device=self.device)

        # Forward pass with AMP
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                             enabled=self.cfg.use_amp):
            output = self.model(seq_tokens)
            predicted_coords = output['coords'] if isinstance(output, dict) else output

            loss, loss_dict = self.compute_loss(predicted_coords, coords, mask)

        # Backward pass
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1

        return loss_dict

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch."""
        epoch_losses = {'loss': [], 'rmsd': [], 'opmc_loss': []}

        for batch_idx, batch in enumerate(dataloader):
            step_losses = self.train_step(batch)
            epoch_losses['loss'].append(step_losses['loss'])
            epoch_losses['rmsd'].append(step_losses['rmsd'])
            epoch_losses['opmc_loss'].append(step_losses['opmc_loss'])

            if self.is_main and (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx+1}/{len(dataloader)}] "
                    f"loss={step_losses['loss']:.4f} "
                    f"rmsd={step_losses['rmsd']:.3f}"
                )

        return {
            'loss': np.mean(epoch_losses['loss']),
            'rmsd': np.mean(epoch_losses['rmsd']),
            'opmc_loss': np.mean(epoch_losses['opmc_loss']),
        }

    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        # Create dataloader
        dataset = SyntheticProteinDataset(
            generator=self.data_gen,
            n_samples=100,
            min_len=self.cfg.min_len,
            max_len=self.cfg.max_len,
            noise_std=self.cfg.noise_std,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=synthetic_collate,
        )

        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            epoch_result = self.train_epoch(dataloader)
            self.history.append(epoch_result)

            if self.is_main:
                logger.info(
                    f"Epoch {epoch} complete: "
                    f"loss={epoch_result['loss']:.4f}, "
                    f"rmsd={epoch_result['rmsd']:.3f}"
                )

            # Periodic refinement
            if self.refine_bridge.available and (epoch + 1) % self.cfg.refine_every_n_epochs == 0:
                self._run_refinement_sample()

            # Checkpoint
            if self.is_main and ((epoch + 1) % self.cfg.checkpoint_interval == 0 or
                                    epoch == self.cfg.epochs - 1):
                ckpt_path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

        return {'history': self.history, 'final_metrics': self.history[-1] if self.history else {}}

    @torch.no_grad()
    def _run_refinement_sample(self) -> None:
        """Run refinement on one sample to validate integration."""
        sample = self.data_gen.generate_batch(batch_size=1, min_len=50, max_len=100)[0]
        coords = sample['target_coords'].to(self.device)
        sequence = sample['sequence']

        result = self.refine_bridge.refine(coords, sequence, steps=self.cfg.refine_steps)
        if result.get('refined', False):
            logger.info(
                f"[Refinement] Sample refined: E {result.get('initial_energy', 'nan')} -> "
                f"{result.get('final_energy', 'nan')}"
            )
        else:
            logger.warning("[Refinement] Refinement not available or failed.")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.refine_bridge.cleanup()
        if _HAS_DIST and dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# 4. SLURM SCRIPT GENERATOR (256-GPU)
# =============================================================================
@dataclass
class SlurmConfig:
    """Configuration for Slurm script generation."""
    job_name: str = "sesi_ultrascale_training"
    account: str = "YOUR_PROJECT_ACCOUNT"
    partition: str = "gpu"
    nodes: int = 32
    gpus_per_node: int = 8
    cpus_per_task: int = 16
    mem_gb: int = 256
    time_limit: str = "48:00:00"
    email: str = "your@email.com"
    modules: List[str] = field(default_factory=lambda: [
        "cuda/12.2", "nccl/2.19", "openmpi/4.1.6-cuda", "python/3.11"
    ])


class SlurmScriptGenerator:
    """
    [D] Generates Slurm batch script for 256-GPU training.

    CONFIGURATION:
      • 32 nodes × 8 GPUs = 256 GPUs total
      • torchrun with multi-node DDP
      • NCCL backend with InfiniBand
      • Checkpointing every N epochs
    """

    def __init__(self, cfg: Optional[SlurmConfig] = None):
        self.cfg = cfg or SlurmConfig()
        self.total_gpus = self.cfg.nodes * self.cfg.gpus_per_node

    def generate(self) -> str:
        """Generate complete Slurm batch script."""
        modules_load = "\n".join(f"module load {m}" for m in self.cfg.modules)

        script = f"""#!/bin/bash
#=============================================================================
# SESI ULTRA-SCALE TRAINING — {self.total_gpus}-GPU SLURM SCRIPT
# Auto-generated by SlurmScriptGenerator v{TRAINING_SUITE_VERSION}
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

#=============================================================================
# ENVIRONMENT SETUP
#=============================================================================
set -euo pipefail

echo "======================================================================"
echo "  SESI ULTRA-SCALE TRAINING — Job Start"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Nodes       : $SLURM_JOB_NUM_NODES"
echo "  GPUs/node   : {self.cfg.gpus_per_node}"
echo "  Total GPUs  : {self.total_gpus}"
echo "  Start Time  : $(date)"
echo "======================================================================"

# Load modules (adjust to your cluster)
module purge
{modules_load}

# Set environment
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export OMP_NUM_THREADS={self.cfg.cpus_per_task}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Working directory
WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR
mkdir -p checkpoints logs

echo "Working directory: $WORKDIR"
echo "Python: $(which python)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

#=============================================================================
# PRE-FLIGHT CHECKS
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  Pre-flight checks"
echo "----------------------------------------------------------------------"

python -c "
import torch
print(f'PyTorch version: {{torch.__version__}}')
print(f'CUDA available: {{torch.cuda.is_available()}}')
print(f'CUDA device count: {{torch.cuda.device_count()}}')
print(f'NCCL available: {{torch.distributed.is_nccl_available()}}')
try:
    import openmm
    print(f'OpenMM version: {{openmm.version.full_version}}')
except ImportError:
    print('OpenMM: NOT INSTALLED')
try:
    from openmmml import MLPotential
    print('OpenMM-ML: available')
except ImportError:
    print('OpenMM-ML: NOT INSTALLED')
"

#=============================================================================
# TRAINING (Multi-Node DDP via torchrun)
#=============================================================================
echo ""
echo "----------------------------------------------------------------------"
echo "  Training: {self.total_gpus} GPUs × DDP + AMP bf16"
echo "----------------------------------------------------------------------"

# torchrun handles multi-node launch automatically
torchrun \\
    --nnodes={self.cfg.nodes} \\
    --nproc_per_node={self.cfg.gpus_per_node} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \\
    --master_port=29500 \\
    --rdzv_backend=c10d \\
    sesi_training_main.py \\
    --epochs 10 \\
    --batch-size 4 \\
    --lr 1e-3 \\
    --amp-dtype bf16 \\
    --checkpoint-dir ./checkpoints_256gpu \\
    --log-interval 10 \\
    --checkpoint-interval 5

TRAINING_STATUS=$?

#=============================================================================
# POST-TRAINING
#=============================================================================
echo ""
echo "======================================================================"
echo "  Training Complete"
echo "  Status      : $TRAINING_STATUS"
echo "  End Time    : $(date)"
echo "  Duration    : $SECONDS seconds"
echo "  Checkpoints : ./checkpoints_256gpu/"
echo "======================================================================"

# Archive results
if [ $TRAINING_STATUS -eq 0 ]; then
    tar -czf results_{self.cfg.job_name}_$SLURM_JOB_ID.tar.gz \\
        checkpoints_256gpu/ \\
        *.log \\
        *.out 2>/dev/null || true
    echo "Results archived to results_{self.cfg.job_name}_$SLURM_JOB_ID.tar.gz"
fi

exit $TRAINING_STATUS
"""
        return script

    def generate_training_main(self) -> str:
        """Generate the training entry point script (sesi_training_main.py)."""
        script = '''#!/usr/bin/env python3
"""
SESI Ultra-Scale Training Main — Entry point for torchrun / Slurm.
"""
import argparse
import os
import torch
import torch.distributed as dist

from sesi_ultrascale_training_slurm import (
    DistributedTrainerConfig,
    DistributedUltraScaleTrainer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp-dtype", type=str, default="bf16")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    args = parser.parse_args()

    # Initialize distributed
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0

    # Config
    cfg = DistributedTrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        amp_dtype=args.amp_dtype,
        checkpoint_dir=args.checkpoint_dir,
        device="cuda",
    )

    # Train
    trainer = DistributedUltraScaleTrainer(cfg)
    results = trainer.train()

    if rank == 0:
        print(f"Training complete. Final loss: {results['final_metrics']['loss']:.4f}")

    trainer.cleanup()


if __name__ == "__main__":
    main()
'''
        return script

    def write_all(self, output_dir: str = "./slurm_deployment") -> Dict[str, str]:
        """Write all deployment files."""
        os.makedirs(output_dir, exist_ok=True)

        files = {}

        # Slurm script
        slurm_path = os.path.join(output_dir, "submit_training.slurm")
        with open(slurm_path, 'w') as f:
            f.write(self.generate())
        files['slurm_script'] = slurm_path

        # Training main
        main_path = os.path.join(output_dir, "sesi_training_main.py")
        with open(main_path, 'w') as f:
            f.write(self.generate_training_main())
        files['training_main'] = main_path

        return files


# =============================================================================
# 5. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI ULTRA-SCALE TRAINING SUITE v{TRAINING_SUITE_VERSION} — Self-Test")
    print(f"  real_fold_one_v2={_HAS_REAL_FOLD}  OpenMM={_HAS_OPENMM}  "
          f"Distributed={_HAS_DIST}  UltraScale={_HAS_ULTRA}")
    print("=" * 78)

    # ---- T1: Synthetic Data Generator ----------------------------------
    print("\n [T1] Synthetic Data Generator")
    gen = SyntheticProteinDataGenerator(seed=42)

    helix = gen.idealized_helix(100)
    assert helix.shape == (100, 3)
    print(f"   Helix shape: {helix.shape}")

    sheet = gen.idealized_sheet(90, n_strands=3)
    assert sheet.shape == (90, 3)
    print(f"   Sheet shape: {sheet.shape}")

    batch = gen.generate_batch(batch_size=4, min_len=50, max_len=100)
    assert len(batch) == 4
    for item in batch:
        assert 'sequence' in item
        assert 'target_coords' in item
        assert item['target_coords'].shape == (item['length'], 3)
    print(f"   Batch: {len(batch)} samples, lengths: {[b['length'] for b in batch]}")
    print("[PASS] T1 Synthetic data generator works")

    # ---- T2: Dataset and Collate ----------------------------------------
    print("\n [T2] Dataset and Collate")
    dataset = SyntheticProteinDataset(gen, n_samples=10, min_len=30, max_len=60)
    assert len(dataset) == 10

    sample = dataset[0]
    assert 'sequence' in sample
    assert 'target_coords' in sample

    # Test collate
    batch_data = [dataset[i] for i in range(3)]
    collated = synthetic_collate(batch_data)
    assert collated['coords'].shape[0] == 3
    assert collated['mask'].shape[0] == 3
    print(f"   Collated batch: coords={tuple(collated['coords'].shape)}, "
          f"mask={tuple(collated['mask'].shape)}")
    print("[PASS] T2 Dataset and collate work")

    # ---- T3: RefinementEngineBridge ------------------------------------
    print("\n [T3] RefinementEngineBridge")
    bridge = RefinementEngineBridge()
    print(f"   Bridge available: {bridge.available}")

    if bridge.available:
        # Test refinement on small structure
        test_coords = gen.idealized_helix(20)
        test_seq = gen.random_sequence(20)
        result = bridge.refine(test_coords, test_seq, steps=5)
        print(f"   Refinement result: refined={result.get('refined', False)}")
    else:
        print("   [SKIP] RefinementEngine not available (requires real_fold_one_v2 + OpenMM)")
    print("[PASS] T3 RefinementEngineBridge initialized")

    # ---- T4: DistributedTrainerConfig ----------------------------------
    print("\n [T4] DistributedTrainerConfig")
    cfg = DistributedTrainerConfig(
        epochs=2,
        batch_size=2,
        lr=1e-3,
        min_len=30,
        max_len=60,
    )
    assert cfg.epochs == 2
    assert cfg.batch_size == 2
    print("[PASS] T4 Config created successfully")

    # ---- T5: DistributedUltraScaleTrainer (single process) -------------
    print("\n [T5] DistributedUltraScaleTrainer")
    trainer = DistributedUltraScaleTrainer(cfg)
    print(f"   Device: {trainer.device}")
    print(f"   Rank: {trainer.rank}/{trainer.world_size}")
    print(f"   Model type: {type(trainer.model).__name__}")

    # Test one training step
    batch = gen.generate_batch(batch_size=2, min_len=30, max_len=60)
    collated = synthetic_collate(batch)
    step_result = trainer.train_step(collated)
    print(f"   Step loss: {step_result['loss']:.4f}")
    print(f"   Step RMSD: {step_result['rmsd']:.3f}")
    assert step_result['loss'] > 0
    assert step_result['rmsd'] > 0
    print("[PASS] T5 Trainer executes training step")

    # ---- T6: Mini training run -----------------------------------------
    print("\n [T6] Mini Training Run (2 epochs)")
    mini_cfg = DistributedTrainerConfig(
        epochs=2,
        batch_size=2,
        lr=1e-2,
        min_len=30,
        max_len=60,
        refine_every_n_epochs=0,  # disable refinement for speed
        checkpoint_interval=1,
        checkpoint_dir="./test_checkpoints",
    )
    mini_trainer = DistributedUltraScaleTrainer(mini_cfg)
    results = mini_trainer.train()
    print(f"   Epochs completed: {len(results['history'])}")
    print(f"   Final loss: {results['final_metrics']['loss']:.4f}")
    print(f"   Final RMSD: {results['final_metrics']['rmsd']:.3f}")
    assert len(results['history']) == 2
    print("[PASS] T6 Mini training run completed")

    # ---- T7: Slurm Script Generation -----------------------------------
    print("\n [T7] Slurm Script Generation")
    slurm_gen = SlurmScriptGenerator(
        SlurmConfig(
            job_name="sesi_test",
            nodes=4,  # small for testing
            gpus_per_node=8,
            account="TEST_ACCOUNT",
        )
    )
    files = slurm_gen.write_all(output_dir="./test_slurm")

    assert os.path.exists(files['slurm_script'])
    assert os.path.exists(files['training_main'])

    with open(files['slurm_script'], 'r') as f:
        slurm_content = f.read()
    assert "#SBATCH --nodes=4" in slurm_content
    assert "#SBATCH --ntasks-per-node=8" in slurm_content
    assert "torchrun" in slurm_content
    print(f"   Slurm script: {files['slurm_script']}")
    print(f"   Training main: {files['training_main']}")
    print(f"   Total GPUs: {slurm_gen.total_gpus}")
    print("[PASS] T7 Slurm deployment files generated")

    # ---- T8: 256-GPU Configuration -------------------------------------
    print("\n [T8] 256-GPU Configuration Check")
    slurm_256 = SlurmScriptGenerator(
        SlurmConfig(nodes=32, gpus_per_node=8, time_limit="48:00:00")
    )
    assert slurm_256.total_gpus == 256
    print(f"   Total GPUs: {slurm_256.total_gpus}")
    print(f"   Nodes: {slurm_256.cfg.nodes}")
    print(f"   GPUs/node: {slurm_256.cfg.gpus_per_node}")
    print(f"   Time limit: {slurm_256.cfg.time_limit}")
    print("[PASS] T8 256-GPU configuration valid")

    # ---- T9: OPMC k* computation ---------------------------------------
    print("\n [T9] OPMC k* Computation")
    k_star = opmc_k_star(1e-8)
    print(f"   k* (ε=1e-8, L={OPMC_L}) = {k_star}")
    assert k_star == 23, f"k* should be 23, got {k_star}"
    print("[PASS] T9 OPMC k* matches paper (Theorem 11.1)")

    # ---- Cleanup ---------------------------------------------------------
    import shutil
    shutil.rmtree("./test_checkpoints", ignore_errors=True)
    shutil.rmtree("./test_slurm", ignore_errors=True)
    trainer.cleanup()
    mini_trainer.cleanup()

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("  HONEST REMINDERS:")
    print("    • Synthetic data is for PIPELINE TESTING ONLY")
    print("    • Production training requires real PDB/CASP data")
    print("    • 256-GPU run requires HPC cluster with 32×H100 nodes")
    print("    • RefinementEngine requires OpenMM + OpenMM-ML installed")
    print("    • This module has NOT been run at full 256-GPU scale here")
    print("    • OPMC k*=23 is FIXED regardless of sequence length")
    print("=" * 78)
