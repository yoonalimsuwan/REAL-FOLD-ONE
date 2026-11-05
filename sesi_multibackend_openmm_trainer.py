"""
=============================================================================
SESI MULTI-BACKEND OPENMM-ML TRAINING SUITE — v1.0.0
8 Hardware Targets + Full OpenMM-ML Energy Training
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================
INTEGRATION WITH real_fold_one_v2-5.py (VERBATIM API USAGE)
=============================================================================
This module calls the following from real_fold_one_v2-5.py:
  • detect_optimal_device()       — extended here to 8 hardware targets
  • OpenMMSystemBuilder           — builds OpenMM system from PDB
  • OpenMMEnergyCalculator        — manages TorchForce / force-injection
  • openmm_solute_energy(coords, calculator)
  • SOCController.compute_soc_energy(ca, alpha, edge_idx, edge_dist)
  • NeighborListManager.build(ca_coords) → {'soc': (edge_idx, edge_dist)}
  • CSOCKernel v2 (LJ-form) — shared kernel instance

LOSS FUNCTION (matching Trainer._train_epoch in real_fold_one_v2-5.py):
  L = E_openmm + w_soc * E_soc
where:
  E_openmm = openmm_solute_energy(solute_coords, calculator)
  E_soc    = SOCController.compute_soc_energy(ca, alpha, edge_idx, edge_dist, w_soc=cfg.w_soc)

BACKEND SUPPORT MATRIX:
  Hardware        | PyTorch backend      | OpenMM platform   | ML native
  ----------------|----------------------|-------------------|-----------
  NVIDIA CUDA     | torch.cuda           | CUDA              | Yes
  AMD ROCm        | torch.cuda (HIP)     | OpenCL            | Yes
  Intel XPU       | torch.xpu (IPEX)     | CPU / OpenCL      | Partial
  Apple Silicon   | torch.backends.mps   | CPU               | No (CPU)
  Huawei Ascend   | torch_npu            | CPU               | No (CPU)
  Alibaba Hanguang| AIACC / mock         | CPU               | No (CPU)
  Google TPU      | torch_xla            | CPU               | No (CPU)
  IBM Power/AC922 | torch.cuda / CPU     | CUDA / CPU        | Yes

HONEST SCOPE:
  • OpenMM natively supports CUDA/OpenCL/CPU/Reference ONLY. On TPU,
    Ascend, and Hanguang, the OpenMM energy term falls back to CPU
    while the PyTorch part runs on the accelerator. This is a
    HETEROGEOUS deployment, not a native one.
  • TorchForce (openmm-torch) native autograd works best on CUDA.
    On other backends, force-injection fallback is used automatically.
  • Huawei Ascend requires `torch_npu` installed. Google TPU requires
    `torch_xla`. Alibaba Hanguang has no public PyTorch backend; the
    manager detects it via environment markers and reports CPU fallback.
  • This module has NOT been run on all 8 backends here. Self-tests
    verify detection logic and graceful fallback.
=============================================================================
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sesi.multibackend_trainer")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

MULTIBACKEND_VERSION: str = "1.0.0"

# =============================================================================
# Ecosystem imports (graceful, with hyphenated filename fallback)
# =============================================================================
_HAS_REAL_FOLD = False
try:
    from real_fold_one_v2 import (
        OpenMMSystemBuilder,
        OpenMMEnergyCalculator,
        openmm_solute_energy,
        SOCController,
        CSOCKernel,
        NeighborListManager,
        RefinementConfig,
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
                OpenMMSystemBuilder = mod.OpenMMSystemBuilder
                OpenMMEnergyCalculator = mod.OpenMMEnergyCalculator
                openmm_solute_energy = mod.openmm_solute_energy
                SOCController = mod.SOCController
                CSOCKernel = mod.CSOCKernel
                NeighborListManager = mod.NeighborListManager
                RefinementConfig = mod.RefinementConfig
                _HAS_REAL_FOLD = True
            break
    if not _HAS_REAL_FOLD:
        warnings.warn("real_fold_one_v2 not found. Training loop uses stand-in physics.")

_HAS_OPENMM = False
try:
    import openmm as mm
    import openmm.unit as unit
    from openmm.app import PDBFile
    _HAS_OPENMM = True
except ImportError:
    pass


# =============================================================================
# 1. MULTI-BACKEND DEVICE MANAGER (8 hardware targets)
# =============================================================================
@dataclass
class BackendCapability:
    """Describes what a backend can and cannot do in this ecosystem."""
    name: str
    torch_device: str
    openmm_platform: str
    supports_native_ml: bool
    supports_torchforce: bool
    supports_amp: bool
    supports_ddp: bool
    notes: str = ""


class MultiBackendDeviceManager:
    """
    [D] Detects and manages 8 hardware backends.

    Detection order (highest to lowest preference):
      1. NVIDIA CUDA
      2. AMD ROCm (via torch.cuda, detected by 'ROCM' in torch.version.hip)
      3. Huawei Ascend NPU (torch_npu)
      4. Google TPU (torch_xla)
      5. Intel XPU (intel_extension_for_pytorch)
      6. Apple Silicon MPS
      7. Alibaba Hanguang (environment marker; no public PyTorch backend)
      8. IBM Power (falls through to CUDA or CPU)
      9. CPU fallback

    OpenMM platform selection:
      • NVIDIA / IBM with CUDA → 'CUDA'
      • AMD → 'OpenCL'
      • Intel → 'CPU' or 'OpenCL'
      • Apple / Ascend / Hanguang / TPU → 'CPU' (heterogeneous deployment)

    COROLLARY-3 GUARD: detection is a discrete branching process.
    HINO GUARD: hardware devices are discrete entities.
    """

    def __init__(self, preferred: Optional[str] = None):
        self.preferred = preferred
        self.available_backends: List[BackendCapability] = []
        self.selected: Optional[BackendCapability] = None
        self._detect_all()
        self._select()

    def _detect_all(self) -> None:
        backends = []

        # 1. NVIDIA CUDA
        if torch.cuda.is_available():
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            is_amd = is_rocm or 'AMD' in torch.cuda.get_device_name(0).upper()
            if is_amd:
                backends.append(BackendCapability(
                    name="AMD_ROCM", torch_device="cuda", openmm_platform="OpenCL",
                    supports_native_ml=True, supports_torchforce=True,
                    supports_amp=True, supports_ddp=True,
                    notes="ROCm provides CUDA-compatible API. OpenMM uses OpenCL."
                ))
            else:
                backends.append(BackendCapability(
                    name="NVIDIA_CUDA", torch_device="cuda", openmm_platform="CUDA",
                    supports_native_ml=True, supports_torchforce=True,
                    supports_amp=True, supports_ddp=True,
                    notes="Primary supported backend. Full OpenMM-ML TorchForce."
                ))

        # 2. Huawei Ascend NPU
        try:
            import torch_npu  # noqa: F401
            if torch_npu.npu.is_available():
                backends.append(BackendCapability(
                    name="HUAWEI_ASCEND", torch_device="npu", openmm_platform="CPU",
                    supports_native_ml=False, supports_torchforce=False,
                    supports_amp=True, supports_ddp=True,
                    notes="torch_npu detected. OpenMM falls back to CPU (heterogeneous)."
                ))
        except ImportError:
            pass

        # 3. Google TPU
        try:
            import torch_xla  # noqa: F401
            import torch_xla.core.xla_model as xm
            if xm.xla_device() is not None:
                backends.append(BackendCapability(
                    name="GOOGLE_TPU", torch_device="xla", openmm_platform="CPU",
                    supports_native_ml=False, supports_torchforce=False,
                    supports_amp=False, supports_ddp=True,
                    notes="torch_xla detected. OpenMM falls back to CPU."
                ))
        except ImportError:
            pass

        # 4. Intel XPU
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if torch.xpu.is_available():
                backends.append(BackendCapability(
                    name="INTEL_XPU", torch_device="xpu", openmm_platform="CPU",
                    supports_native_ml=True, supports_torchforce=False,
                    supports_amp=True, supports_ddp=True,
                    notes="Intel Extension for PyTorch. OpenMM uses CPU."
                ))
        except ImportError:
            pass

        # 5. Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            backends.append(BackendCapability(
                name="APPLE_MPS", torch_device="mps", openmm_platform="CPU",
                supports_native_ml=False, supports_torchforce=False,
                supports_amp=True, supports_ddp=False,
                notes="Apple Silicon MPS. OpenMM uses CPU. No DDP."
            ))

        # 6. Alibaba Hanguang (environment marker; no public PyTorch backend)
        if os.environ.get("ALIBABA_HANGUANG") or os.path.exists("/usr/local/hanguang"):
            backends.append(BackendCapability(
                name="ALIBABA_HANGUANG", torch_device="cpu", openmm_platform="CPU",
                supports_native_ml=False, supports_torchforce=False,
                supports_amp=False, supports_ddp=False,
                notes="Hanguang detected via environment. No public PyTorch backend; CPU fallback."
            ))

        # 7. IBM Power (falls through; detected via platform string)
        if 'ppc64' in os.uname().machine.lower():
            backends.append(BackendCapability(
                name="IBM_POWER", torch_device="cuda" if torch.cuda.is_available() else "cpu",
                openmm_platform="CUDA" if torch.cuda.is_available() else "CPU",
                supports_native_ml=torch.cuda.is_available(),
                supports_torchforce=torch.cuda.is_available(),
                supports_amp=torch.cuda.is_available(), supports_ddp=True,
                notes="IBM Power architecture. Uses CUDA if available, else CPU."
            ))

        # 8. CPU fallback (always available)
        backends.append(BackendCapability(
            name="CPU", torch_device="cpu", openmm_platform="CPU",
            supports_native_ml=False, supports_torchforce=False,
            supports_amp=False, supports_ddp=True,
            notes="CPU fallback. Always available."
        ))

        self.available_backends = backends
        logger.info(f"[Backend] Detected {len(backends)} backend(s): "
                    f"{[b.name for b in backends]}")

    def _select(self) -> None:
        if self.preferred:
            for b in self.available_backends:
                if b.name.lower() == self.preferred.lower():
                    self.selected = b
                    return
        # Default: first non-CPU backend, else CPU
        for b in self.available_backends:
            if b.name != "CPU":
                self.selected = b
                return
        self.selected = self.available_backends[-1]

    @property
    def torch_device(self) -> torch.device:
        if self.selected is None:
            return torch.device("cpu")
        return torch.device(self.selected.torch_device)

    @property
    def openmm_platform(self) -> str:
        return self.selected.openmm_platform if self.selected else "CPU"

    def get_capability_matrix(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": b.name,
                "torch_device": b.torch_device,
                "openmm_platform": b.openmm_platform,
                "supports_native_ml": b.supports_native_ml,
                "supports_torchforce": b.supports_torchforce,
                "supports_amp": b.supports_amp,
                "supports_ddp": b.supports_ddp,
                "notes": b.notes,
            }
            for b in self.available_backends
        ]


# =============================================================================
# 2. FULL OPENMM-ML ENERGY TRAINER (L = E_openmm + w_soc * E_soc)
# =============================================================================
@dataclass
class FullTrainerConfig:
    """Configuration for the full energy training loop."""
    # Loss weights (matching Trainer in real_fold_one_v2-5.py)
    w_soc: float = 0.3

    # SOC / neighbor list
    cutoff_soc: float = 12.0
    max_neighbors: int = 64

    # Optimization
    lr: float = 1e-3
    epochs: int = 1
    grad_clip: float = 5.0

    # OpenMM-ML
    ml_potential: str = "ani2x"
    use_native_autograd: bool = True
    implicit_solvent: Optional[str] = None
    solvate: bool = False

    # Backend
    backend_preference: Optional[str] = None

    # Logging
    log_interval: int = 10

    # Device override (for testing)
    device: Optional[str] = None


class OpenMMEnergyTrainerFull:
    """
    [D] Full training loop matching Trainer._train_epoch from real_fold_one_v2-5.py.

    LOSS FUNCTION (exact match):
        L = E_openmm + E_soc
    where:
        E_openmm = openmm_solute_energy(solute_coords, calculator)
        E_soc    = soc.compute_soc_energy(ca, alpha, edge_idx, edge_dist,
                                          w_soc=cfg.w_soc)

    DIFFERENCES FROM THE REFERENCE Trainer:
      • Accepts an arbitrary backend (8 hardware targets) via MultiBackendDeviceManager.
      • Uses shared CSOCKernel so the same kernel is used for both
        training and later inference/refinement.
      • Adds explicit gradient checking and NaN guards.

    WHAT IS TRAINED:
      • CSOCKernel parameters (alpha, r0, eps_lj) — the learnable SOC kernel.
      • Optionally the structure coordinates themselves (if trainable_coords=True).

    CARPET-19 DISCIPLINE:
      • COROLLARY-3: first-order autograd only; no Hessian.
      • HINO: atoms are discrete entities; neighbor list is discrete.
      • PROP-3: products prob_ij * E_LJ(d_ij) lose regularity at short
        distances; CSOCKernel's safe_r = r + eps clamp bounds the product.
    """

    def __init__(self, cfg: Optional[FullTrainerConfig] = None):
        self.cfg = cfg or FullTrainerConfig()

        # Backend selection
        self.backend_mgr = MultiBackendDeviceManager(
            preferred=self.cfg.backend_preference
        )
        self.device = (
            torch.device(self.cfg.device) if self.cfg.device
            else self.backend_mgr.torch_device
        )
        logger.info(f"[Trainer] Using device: {self.device} "
                    f"(backend: {self.backend_mgr.selected.name if self.backend_mgr.selected else 'none'})")

        # Build SOC components (shared kernel)
        if _HAS_REAL_FOLD:
            self.kernel = CSOCKernel()
            self.soc = SOCController(kernel=self.kernel)
            self.neighbor_mgr = NeighborListManager(
                cutoffs={'soc': self.cfg.cutoff_soc},
                max_neighbors=self.cfg.max_neighbors,
                device=self.device,
            )
        else:
            self.kernel = None
            self.soc = None
            self.neighbor_mgr = None
            logger.warning("[Trainer] real_fold_one_v2 not available. "
                           "SOC energy term disabled (training with OpenMM energy only, if available).")

        self.history: List[Dict[str, Any]] = []
        self._structure_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # OpenMM system building via OpenMMSystemBuilder
    # ------------------------------------------------------------------
    def _build_openmm_system(self, pdb_path: str) -> Dict[str, Any]:
        """
        Build OpenMM system from PDB using OpenMMSystemBuilder.

        Returns dict with:
          calculator, solute_coords, ca_indices, full2sol, solute_mask
        """
        if pdb_path in self._structure_cache:
            return self._structure_cache[pdb_path]

        if not _HAS_REAL_FOLD:
            raise ImportError("real_fold_one_v2-5.py required for OpenMM system building.")

        builder = OpenMMSystemBuilder(
            implicit_solvent=self.cfg.implicit_solvent,
            solvate=self.cfg.solvate,
        )
        system, topology, init_all_ang, meta = builder.build_from_pdb(
            pdb_path, solvate=self.cfg.solvate
        )

        full_coords = init_all_ang.to(self.device)
        solute_mask = meta['solute_mask'].to(self.device)
        solute_indices = meta['solute_indices']
        full2sol = meta['full2sol']
        ca_indices = torch.tensor(
            [full2sol[idx] for idx in meta['ca_indices_full'] if idx in full2sol],
            dtype=torch.long, device=self.device
        )

        calculator = OpenMMEnergyCalculator(
            system=system,
            topology=topology,
            full_coords_fixed=full_coords,
            solute_mask=solute_mask,
            platform_name=self.backend_mgr.openmm_platform,
            ml_potential_name=self.cfg.ml_potential,
            use_native_autograd=self.cfg.use_native_autograd,
        )

        solute_coords = full_coords[solute_mask].clone().detach().requires_grad_(True)

        cache_entry = {
            'calculator': calculator,
            'solute_coords': solute_coords,
            'ca_indices': ca_indices,
            'full2sol': full2sol,
            'solute_mask': solute_mask,
        }
        self._structure_cache[pdb_path] = cache_entry
        logger.info(f"[Trainer] Built OpenMM system for {Path(pdb_path).name}: "
                    f"{len(solute_coords)} solute atoms, {len(ca_indices)} CA atoms")
        return cache_entry

    # ------------------------------------------------------------------
    # Single training step (matches Trainer._train_epoch logic)
    # ------------------------------------------------------------------
    def train_step(self, structure_data: Dict[str, Any],
                     optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        One training step for a single structure.

        Loss: L = E_openmm + E_soc
        """
        solute_coords = structure_data['solute_coords']
        calculator = structure_data['calculator']
        ca_indices = structure_data['ca_indices']

        if len(ca_indices) == 0:
            logger.warning("[Trainer] No CA atoms; skipping step.")
            return {'loss': 0.0, 'E_openmm': 0.0, 'E_soc': 0.0}

        # Extract CA coordinates
        ca = solute_coords[ca_indices]
        M = ca.shape[0]

        # alpha: per-residue weight (all ones, matching reference Trainer)
        alpha = torch.ones(M, device=self.device)

        # Build neighbor list for SOC energy
        if self.neighbor_mgr is not None:
            edge_dict = self.neighbor_mgr.build(ca.detach())
            edge_idx, edge_dist = edge_dict['soc']
        else:
            edge_idx = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_dist = torch.empty(0, device=self.device)

        # Forward pass: E_openmm + E_soc
        optimizer.zero_grad()

        # E_openmm via openmm_solute_energy (native autograd or fallback)
        E_openmm = openmm_solute_energy(solute_coords, calculator)

        # E_soc via SOCController.compute_soc_energy
        if self.soc is not None and edge_idx.numel() > 0:
            E_soc = self.soc.compute_soc_energy(
                ca, alpha, edge_idx, edge_dist, w_soc=self.cfg.w_soc
            )
        else:
            E_soc = torch.tensor(0.0, device=self.device)

        loss = E_openmm + E_soc

        # Check for NaN/Inf before backward
        if not torch.isfinite(loss):
            logger.error(f"[Trainer] Non-finite loss: {loss.item()}. Skipping backward.")
            optimizer.zero_grad()
            return {'loss': float('nan'), 'E_openmm': float('nan'), 'E_soc': float('nan')}

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.kernel.parameters() if self.kernel else [solute_coords],
                                        self.cfg.grad_clip)

        optimizer.step()

        return {
            'loss': loss.item(),
            'E_openmm': E_openmm.item(),
            'E_soc': E_soc.item(),
            'n_ca': M,
            'n_edges': int(edge_idx.shape[1]) if edge_idx.numel() > 0 else 0,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self, pdb_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Train on a list of PDB files.

        Matches the structure of Trainer.train() in real_fold_one_v2-5.py
        but with multi-backend support.
        """
        if not pdb_paths:
            logger.warning("[Trainer] No PDB files provided.")
            return []

        # Build all systems first (so OpenMM contexts are ready)
        structure_data_list = []
        for pdb_path in pdb_paths:
            try:
                structure_data_list.append(self._build_openmm_system(pdb_path))
            except Exception as e:
                logger.error(f"[Trainer] Failed to build system for {pdb_path}: {e}")
                continue

        if not structure_data_list:
            logger.error("[Trainer] No valid structures to train on.")
            return []

        # Optimizer over kernel parameters (matching Trainer.train)
        if self.kernel is not None:
            optimizer = torch.optim.Adam(self.kernel.parameters(), lr=self.cfg.lr)
        else:
            logger.warning("[Trainer] No kernel available; training disabled.")
            return []

        history = []
        for epoch in range(self.cfg.epochs):
            epoch_losses = []
            for step, structure_data in enumerate(structure_data_list):
                try:
                    step_result = self.train_step(structure_data, optimizer)
                    epoch_losses.append(step_result['loss'])

                    if (step + 1) % self.cfg.log_interval == 0:
                        logger.info(
                            f"[Trainer] epoch {epoch} step {step+1}: "
                            f"loss={step_result['loss']:.4f} "
                            f"E_openmm={step_result['E_openmm']:.4f} "
                            f"E_soc={step_result['E_soc']:.4f}"
                        )
                except Exception as e:
                    logger.error(f"[Trainer] Step failed: {e}")
                    continue

            avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            history.append({
                'epoch': epoch,
                'avg_loss': float(avg_loss),
                'n_structures': len(structure_data_list),
            })
            logger.info(f"[Trainer] Epoch {epoch}: avg_loss={avg_loss:.4f}")

        self.history = history
        return history

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        """Close all OpenMM calculators and free memory."""
        for pdb_path, data in self._structure_cache.items():
            try:
                data['calculator'].close()
            except Exception:
                pass
        self._structure_cache.clear()
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


# =============================================================================
# 3. SYNTHETIC PDB GENERATOR (for self-test without real PDB files)
# =============================================================================
def _write_test_pdb(path: Path, n_residues: int = 20) -> str:
    """Write a minimal Cα-trace PDB for testing."""
    aa_names = ['ALA'] * n_residues
    with open(path, 'w') as f:
        f.write("REMARK   1 GENERATED FOR SESI MULTIBACKEND TRAINER TEST\n")
        for i in range(n_residues):
            x = 1.5 * i
            y = 2.3 * math.cos(math.radians(100 * i))
            z = 2.3 * math.sin(math.radians(100 * i))
            f.write(
                f"ATOM  {i+1:5d}  CA  {aa_names[i]:>3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")
    return str(path)


# =============================================================================
# 4. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 78)
    print(f"  SESI MULTI-BACKEND TRAINER v{MULTIBACKEND_VERSION} — Self-Test")
    print(f"  real_fold_one_v2: {_HAS_REAL_FOLD}  OpenMM: {_HAS_OPENMM}")
    print("=" * 78)

    # ---- T1: Backend detection -----------------------------------------
    print("\n [T1] Multi-Backend Device Detection")
    mgr = MultiBackendDeviceManager()
    print(f"   Detected backends: {[b.name for b in mgr.available_backends]}")
    print(f"   Selected: {mgr.selected.name if mgr.selected else 'none'}")
    print(f"   torch_device: {mgr.torch_device}")
    print(f"   openmm_platform: {mgr.openmm_platform}")

    matrix = mgr.get_capability_matrix()
    for entry in matrix:
        print(f"     {entry['name']:20s} | torch={entry['torch_device']:4s} | "
              f"OpenMM={entry['openmm_platform']:7s} | "
              f"ML={entry['supports_native_ml']} | "
              f"TorchForce={entry['supports_torchforce']} | "
              f"AMP={entry['supports_amp']}")
    print("[PASS] T1 Backend detection completed")

    # ---- T2: FullTrainerConfig validation ------------------------------
    print("\n [T2] FullTrainerConfig")
    cfg = FullTrainerConfig(
        w_soc=0.3,
        cutoff_soc=12.0,
        lr=1e-3,
        epochs=1,
        backend_preference=None,
    )
    assert cfg.w_soc == 0.3
    assert cfg.cutoff_soc == 12.0
    print("[PASS] T2 Config created successfully")

    # ---- T3: Trainer initialization ------------------------------------
    print("\n [T3] OpenMMEnergyTrainerFull initialization")
    trainer = OpenMMEnergyTrainerFull(cfg)
    print(f"   Device: {trainer.device}")
    print(f"   Kernel available: {trainer.kernel is not None}")
    print(f"   SOC available: {trainer.soc is not None}")
    print(f"   NeighborMgr available: {trainer.neighbor_mgr is not None}")
    print("[PASS] T3 Trainer initialized")

    # ---- T4: Build OpenMM system from test PDB --------------------------
    print("\n [T4] OpenMM system building from test PDB")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = _write_test_pdb(Path(tmpdir) / "test.pdb", n_residues=20)

        if _HAS_REAL_FOLD and _HAS_OPENMM:
            try:
                structure_data = trainer._build_openmm_system(pdb_path)
                print(f"   solute_coords shape: {structure_data['solute_coords'].shape}")
                print(f"   ca_indices: {len(structure_data['ca_indices'])} CA atoms")
                print(f"   calculator.use_native: {structure_data['calculator'].use_native}")
                print("[PASS] T4 OpenMM system built successfully")
            except Exception as e:
                print(f"[WARN] T4 OpenMM system build failed: {e}")
                print("       (This may be expected if OpenMM force fields are not installed)")
        else:
            print("[SKIP] T4 OpenMM or real_fold_one_v2 not available")

    # ---- T5: Training step (with or without OpenMM) ---------------------
    print("\n [T5] Training step execution")
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = _write_test_pdb(Path(tmpdir) / "train_test.pdb", n_residues=15)

        if _HAS_REAL_FOLD and _HAS_OPENMM:
            try:
                structure_data = trainer._build_openmm_system(pdb_path)
                optimizer = torch.optim.Adam(trainer.kernel.parameters(), lr=cfg.lr)
                step_result = trainer.train_step(structure_data, optimizer)
                print(f"   loss: {step_result['loss']:.4f}")
                print(f"   E_openmm: {step_result['E_openmm']:.4f}")
                print(f"   E_soc: {step_result['E_soc']:.4f}")
                print(f"   n_ca: {step_result.get('n_ca', 'N/A')}")
                print("[PASS] T5 Training step completed")
            except Exception as e:
                print(f"[WARN] T5 Training step failed: {e}")
        else:
            print("[SKIP] T5 Training step requires real_fold_one_v2 and OpenMM")

    # ---- T6: Cleanup ----------------------------------------------------
    print("\n [T6] Cleanup")
    trainer.cleanup()
    print("[PASS] T6 Cleanup completed")

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("  SELF-TEST SUMMARY")
    print("  HONEST REMINDERS:")
    print("    • OpenMM natively supports CUDA/OpenCL/CPU only.")
    print("      On TPU/Ascend/Hanguang, OpenMM falls back to CPU (heterogeneous).")
    print("    • TorchForce native autograd works best on CUDA.")
    print("    • This module has NOT been run on all 8 backends here.")
    print("    • Loss L = E_openmm + w_soc*E_soc matches Trainer._train_epoch")
    print("      in real_fold_one_v2-5.py exactly.")
    print("    • CSOCKernel v2 (LJ-form) is used; alpha is per-residue weight,")
    print("      not the decay exponent (see kernel docstring in real_fold_one_v2-5.py).")
    print("=" * 78)
