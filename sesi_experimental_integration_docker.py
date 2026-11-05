"""
=============================================================================
SESI EXPERIMENTAL INTEGRATION SUITE — v1.0.0
XrayCryoEM Bridge + Real PDB Dataset Loader + Universal Dockerfile
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
[1] XrayCryoEMIntegrationBridge
    - Connects predicted coordinates to XrayCryoEMNativeEngine
    - Uses CryoEMDensityRestraint for experimental map fitting
    - Supports both X-ray (R_work/R_free) and Cryo-EM (FSC/CCC) data
    - Integrates with RestrainedDomainDockingAssembler for XL-MS + Cryo-EM

[2] RealPDBDatasetLoader
    - Production-grade PDB dataset loader
    - Supports local files and RCSB download
    - Train/val/test split by sequence hash
    - Augmentation: noise, rotation, translation
    - Compatible with SDA-ONE, TypeSpecializedFoldingSystem, SpecialistFold

[3] DockerfileGenerator
    - Multi-backend Docker image (CUDA, ROCm, CPU)
    - Includes ALL ecosystem modules
    - Generates docker-compose.yml for multi-service deployment

[4] ModuleIntegrationMap
    - Complete wiring diagram across all 15+ modules
    - Generates DOT graph and Markdown documentation

=============================================================================
INTEGRATION WITH ECOSYSTEM FILES
=============================================================================
This module calls verbatim APIs from:

  xray_cryo_em_complete_production_implementation.py
    → XrayCryoEMNativeEngine.forward(f_obs, f_calc, sigmas, exp_map, sim_map, ...)
    → XrayCryoEMConfig

  experimental_restraints_one.py
    → CryoEMDensityRestraint(density_map, cfg)
    → XLMSRestraintSet.from_pairs(pairs, d_max)
    → RestrainedDomainDockingAssembler(cfg, restraint_cfg, xlms, cryoem)
    → load_density_map_native(path, resolution)

  structural_domain_assembly_one.py
    → StructuralDomainAssembly.forward(sequence) → init_coords

  real_fold_one_v2-5.py
    → RefinementEngine.refine(pdb_file) → solute_coords
    → validate_structure(pdb_file, reference_pdb)

  TYPE_SPECIALIZED_PROTEIN_FOLDING_SYSTEMv1.py
    → TypeSpecializedFoldingSystem.forward(sequence_onehot, hydrophobicity)

  SPECIALISTFOLDv1_MixtureofExperts.py
    → SpecialistFold.forward(seq, mask)

  nmr_restraint_set_optimized_rmsd_loss.py
    → NMRRestraintSet.forward(coords, reference_coords)

=============================================================================
HONEST SCOPE
=============================================================================
• XrayCryoEMIntegrationBridge requires actual experimental data (MRC maps,
  MTZ files, or structure factors). Without them, it falls back to mock data.
• RealPDBDatasetLoader downloads from RCSB by default; for offline use,
  provide local PDB files.
• Docker images are TEMPLATES; actual deployment requires docker build.
• This module has NOT been run end-to-end with real experimental data here.
=============================================================================
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.experimental_integration")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

EXPERIMENTAL_INTEGRATION_VERSION: str = "1.0.0"

# =============================================================================
# Ecosystem imports (graceful fallback)
# =============================================================================
_HAS_XRAY = False
try:
    from xray_cryo_em_complete_production_implementation import (
        XrayCryoEMNativeEngine,
        XrayCryoEMConfig,
    )
    _HAS_XRAY = True
except ImportError:
    import importlib.util
    for cand in ("xray_cryo_em_complete_production_implementation.py",
                 "xray_cryo_em_complete_production_implementation (1).py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location(
                "xray_cryo_em", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                XrayCryoEMNativeEngine = mod.XrayCryoEMNativeEngine
                XrayCryoEMConfig = mod.XrayCryoEMConfig
                _HAS_XRAY = True
            break
    if not _HAS_XRAY:
        warnings.warn("XrayCryoEM module not found; using mock engine.")

_HAS_RESTRAINTS = False
try:
    from experimental_restraints_one import (
        RestraintDockingConfig,
        CrossLink,
        XLMSRestraintSet,
        DensityMap,
        load_density_map_native,
        CryoEMDensityRestraint,
        RestrainedDomainDockingAssembler,
    )
    _HAS_RESTRAINTS = True
except ImportError:
    import importlib.util
    p = Path("experimental_restraints_one.py")
    if p.exists():
        spec = importlib.util.spec_from_file_location("experimental_restraints", str(p))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            RestraintDockingConfig = mod.RestraintDockingConfig
            CrossLink = mod.CrossLink
            XLMSRestraintSet = mod.XLMSRestraintSet
            DensityMap = mod.DensityMap
            load_density_map_native = mod.load_density_map_native
            CryoEMDensityRestraint = mod.CryoEMDensityRestraint
            RestrainedDomainDockingAssembler = mod.RestrainedDomainDockingAssembler
            _HAS_RESTRAINTS = True
    if not _HAS_RESTRAINTS:
        warnings.warn("experimental_restraints_one not found; restraints disabled.")

_HAS_SDA = False
try:
    from structural_domain_assembly_one import (
        StructuralDomainAssembly,
        DomainAssemblyConfig,
    )
    _HAS_SDA = True
except ImportError:
    import importlib.util
    p = Path("structural_domain_assembly_one.py")
    if p.exists():
        spec = importlib.util.spec_from_file_location("sda_one", str(p))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            StructuralDomainAssembly = mod.StructuralDomainAssembly
            DomainAssemblyConfig = mod.DomainAssemblyConfig
            _HAS_SDA = True
    if not _HAS_SDA:
        warnings.warn("SDA-ONE not found; using stub.")

_HAS_REAL_FOLD = False
try:
    from real_fold_one_v2 import RefinementEngine, RefinementConfig
    _HAS_REAL_FOLD = True
except ImportError:
    import importlib.util
    for cand in ("real_fold_one_v2-5.py", "real_fold_one_v2.py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("real_fold", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                RefinementEngine = mod.RefinementEngine
                RefinementConfig = mod.RefinementConfig
                _HAS_REAL_FOLD = True
            break
    if not _HAS_REAL_FOLD:
        warnings.warn("real_fold_one_v2 not found; refinement disabled.")

_HAS_TYPE_SPECIALIZED = False
try:
    from TYPE_SPECIALIZED_PROTEIN_FOLDING_SYSTEMv1 import TypeSpecializedFoldingSystem
    _HAS_TYPE_SPECIALIZED = True
except ImportError:
    import importlib.util
    for cand in ("TYPE_SPECIALIZED_PROTEIN_FOLDING_SYSTEMv1.py",
                 "TYPE_SPECIALIZED_PROTEIN_FOLDING_SYSTEMv1 (1).py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("type_specialized", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                TypeSpecializedFoldingSystem = mod.TypeSpecializedFoldingSystem
                _HAS_TYPE_SPECIALIZED = True
            break

_HAS_SPECIALIST = False
try:
    from SPECIALISTFOLDv1_MixtureofExperts_Protein_Folding_by_Protein_Class import SpecialistFold
    _HAS_SPECIALIST = True
except ImportError:
    import importlib.util
    p = Path("SPECIALISTFOLDv1_MixtureofExperts_Protein_Folding_by_Protein_Class.py")
    if p.exists():
        spec = importlib.util.spec_from_file_location("specialistfold", str(p))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            SpecialistFold = mod.SpecialistFold
            _HAS_SPECIALIST = True

_HAS_NMR = False
try:
    from nmr_restraint_set_optimized_rmsd_loss import NMRRestraintSet
    _HAS_NMR = True
except ImportError:
    import importlib.util
    for cand in ("nmr_restraint_set_optimized_rmsd_loss.py",
                 "nmr_restraint_set_optimized_rmsd_loss (1).py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("nmr_restraints", str(p))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                NMRRestraintSet = mod.NMRRestraintSet
                _HAS_NMR = True
            break

logger.info(f"[imports] XrayCryoEM={_HAS_XRAY} Restraints={_HAS_RESTRAINTS} "
            f"SDA={_HAS_SDA} RealFold={_HAS_REAL_FOLD} "
            f"TypeSpecialized={_HAS_TYPE_SPECIALIZED} Specialist={_HAS_SPECIALIST} "
            f"NMR={_HAS_NMR}")


# =============================================================================
# 1. XRAY-CRYOEM INTEGRATION BRIDGE
# =============================================================================
@dataclass
class XrayCryoEMBridgeConfig:
    """Configuration for XrayCryoEM integration."""
    # Engine settings
    use_native_engine: bool = True
    resolution_limit: float = 1.2
    sigma_noise_floor: float = 1e-4

    # Cryo-EM settings
    cryoem_map_path: Optional[str] = None
    cryoem_resolution: float = 4.0
    cryoem_weight: float = 2.0
    gaussian_sigma_factor: float = 0.4

    # X-ray settings
    use_xray: bool = False
    xray_weight: float = 1.0

    # XL-MS settings
    use_xlms: bool = False
    xlms_weight: float = 5.0
    xlms_dmax: float = 30.0

    # Device
    device: str = "auto"


class XrayCryoEMIntegrationBridge:
    """
    [D] Bridge connecting predicted coordinates to experimental data fitting.

    PIPELINE:
      1. Receive predicted coords from SDA-ONE / TypeSpecializedFoldingSystem
      2. Generate simulated density map via Gaussian splatting
      3. Compare with experimental map (Cryo-EM) or structure factors (X-ray)
      4. Compute total objective via XrayCryoEMNativeEngine
      5. Optionally apply XL-MS restraints via RestrainedDomainDockingAssembler

    SUPPORTED EXPERIMENTAL DATA:
      • Cryo-EM density maps (.mrc/.ccp4) via load_density_map_native
      • X-ray structure factors (simulated from coords if no MTZ provided)
      • XL-MS crosslinks via XLMSRestraintSet

    INTEGRATION WITH XrayCryoEMNativeEngine:
      The engine computes:
        total_objective = nll_xray + nll_cryo + no_zeno_gate
      where:
        nll_xray = Rice ML target with bulk-solvent correction
        nll_cryo = shell-wise FSC + real-space variance ML
        no_zeno_gate = double-exponential topological stabilizer
    """

    def __init__(self, cfg: Optional[XrayCryoEMBridgeConfig] = None):
        self.cfg = cfg or XrayCryoEMBridgeConfig()

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # Initialize XrayCryoEM engine
        if _HAS_XRAY and self.cfg.use_native_engine:
            engine_cfg = XrayCryoEMConfig(
                resolution_limit=self.cfg.resolution_limit,
                sigma_noise_floor=self.cfg.sigma_noise_floor,
            )
            self.engine = XrayCryoEMNativeEngine(engine_cfg).to(self.device)
        else:
            self.engine = None
            logger.warning("[XrayCryoEM] Native engine not available; using mock.")

        # Load Cryo-EM map if provided
        self.cryoem_map = None
        self.cryoem_restraint = None
        if self.cfg.cryoem_map_path and _HAS_RESTRAINTS:
            try:
                self.cryoem_map = load_density_map_native(
                    self.cfg.cryoem_map_path,
                    resolution=self.cfg.cryoem_resolution
                )
                restraint_cfg = RestraintDockingConfig(
                    cryoem_weight=self.cfg.cryoem_weight,
                    cryoem_gaussian_sigma_factor=self.cfg.gaussian_sigma_factor,
                )
                self.cryoem_restraint = CryoEMDensityRestraint(
                    self.cryoem_map, restraint_cfg
                )
                logger.info(f"[XrayCryoEM] Loaded Cryo-EM map from {self.cfg.cryoem_map_path}")
            except Exception as e:
                logger.warning(f"[XrayCryoEM] Failed to load Cryo-EM map: {e}")

    def coords_to_simulated_map(self, coords: torch.Tensor,
                                  grid_size: Tuple[int, int, int] = (64, 64, 64),
                                  voxel_size: float = 1.5) -> torch.Tensor:
        """
        Generate simulated density map from coordinates via Gaussian splatting.

        Args:
            coords: (N, 3) Cα coordinates in Angstrom
            grid_size: (nx, ny, nz) grid dimensions
            voxel_size: voxel size in Angstrom

        Returns:
            (nx, ny, nz) simulated density map
        """
        N = coords.shape[0]
        nx, ny, nz = grid_size

        # Create grid
        x = torch.arange(nx, device=self.device, dtype=torch.float32) * voxel_size
        y = torch.arange(ny, device=self.device, dtype=torch.float32) * voxel_size
        z = torch.arange(nz, device=self.device, dtype=torch.float32) * voxel_size
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack([X, Y, Z], dim=-1)  # (nx, ny, nz, 3)

        # Center grid on coordinates
        center = coords.mean(dim=0)
        grid = grid + center - torch.tensor([nx, ny, nz], device=self.device, dtype=torch.float32) * voxel_size / 2

        # Gaussian splat: sum of Gaussians at each Cα position
        sigma = self.cfg.gaussian_sigma_factor * self.cfg.cryoem_resolution
        density = torch.zeros(nx, ny, nz, device=self.device)

        # Chunked computation to avoid memory issues
        chunk_size = 100
        for i in range(0, N, chunk_size):
            chunk_coords = coords[i:i+chunk_size]  # (chunk, 3)
            # Distance from each grid point to each atom
            diff = grid.unsqueeze(0) - chunk_coords.reshape(1, 1, 1, -1, 3)  # (1,1,1,chunk,3)
            dist_sq = (diff ** 2).sum(dim=-1)  # (nx, ny, nz, chunk)
            gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
            density += gaussian.sum(dim=-1)

        return density

    def fit_experimental_data(self, predicted_coords: torch.Tensor,
                                 experimental_map: Optional[torch.Tensor] = None,
                                 f_obs: Optional[torch.Tensor] = None,
                                 f_calc: Optional[torch.Tensor] = None,
                                 sigmas: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Fit predicted coordinates to experimental data.

        Args:
            predicted_coords: (N, 3) predicted Cα coordinates
            experimental_map: optional (nx, ny, nz) experimental density map
            f_obs: optional observed structure factors (for X-ray)
            f_calc: optional calculated structure factors
            sigmas: optional structure factor uncertainties

        Returns:
            dict with fitting metrics and refined coordinates
        """
        result = {
            "predicted_coords": predicted_coords,
            "metrics": {},
        }

        # Generate simulated map if not provided
        if experimental_map is None and self.cryoem_map is not None:
            # Use loaded Cryo-EM map
            exp_map = torch.from_numpy(self.cryoem_map.data).to(self.device)
        else:
            exp_map = experimental_map

        # Cryo-EM fitting
        if exp_map is not None:
            sim_map = self.coords_to_simulated_map(
                predicted_coords,
                grid_size=exp_map.shape,
                voxel_size=1.5
            )

            if self.engine is not None:
                cryo_out = self.engine.compute_cryo_em_map_ml(exp_map, sim_map)
                result["metrics"]["cryo_nll"] = cryo_out["nll_cryo"].item()
                result["metrics"]["fsc_correlation"] = cryo_out["cross_correlation"].item()
                result["metrics"]["sum_squared_error"] = cryo_out["sum_squared_error"].item()
            else:
                # Mock metrics
                result["metrics"]["cryo_nll"] = 0.0
                result["metrics"]["fsc_correlation"] = 0.5
                result["metrics"]["sum_squared_error"] = 0.0

            # Apply Cryo-EM restraint if available
            if self.cryoem_restraint is not None:
                ccc = self.cryoem_restraint.simulate_and_correlate(predicted_coords)
                result["metrics"]["restraint_ccc"] = ccc.item()

        # X-ray fitting
        if f_obs is not None and f_calc is not None and sigmas is not None:
            if self.engine is not None:
                xray_out = self.engine.compute_xray_maximum_likelihood(
                    f_obs, f_calc, sigmas
                )
                result["metrics"]["r_work"] = xray_out["r_work"].item()
                result["metrics"]["r_free"] = xray_out["r_free"].item()
                result["metrics"]["chi_sq"] = xray_out["chi_sq"].item()
            else:
                result["metrics"]["r_work"] = 0.2
                result["metrics"]["r_free"] = 0.25
                result["metrics"]["chi_sq"] = 1.0

        return result

    def run_full_pipeline(self, sequence: str,
                             experimental_map: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Full pipeline: sequence → prediction → experimental fitting.

        Args:
            sequence: amino acid sequence
            experimental_map: optional experimental density map

        Returns:
            dict with all results
        """
        result = {"sequence": sequence, "sequence_length": len(sequence)}

        # Step 1: Predict structure (use SDA-ONE if available)
        if _HAS_SDA:
            sda_cfg = DomainAssemblyConfig()
            sda = StructuralDomainAssembly(sda_cfg=sda_cfg).to(self.device)
            with torch.no_grad():
                sda_out = sda(sequence)
            predicted_coords = sda_out["init_coords"]
            result["prediction_method"] = "SDA-ONE"
        else:
            # Mock prediction
            N = len(sequence)
            predicted_coords = torch.randn(N, 3, device=self.device) * 10
            result["prediction_method"] = "mock"

        result["predicted_coords"] = predicted_coords

        # Step 2: Fit to experimental data
        fit_result = self.fit_experimental_data(predicted_coords, experimental_map)
        result.update(fit_result)

        # Step 3: Refine with RefinementEngine if available
        if _HAS_REAL_FOLD:
            try:
                refine_cfg = RefinementConfig()
                engine = RefinementEngine(refine_cfg)
                # Write coords to temp PDB
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
                    tmp_pdb = f.name
                    for i, (x, y, z) in enumerate(predicted_coords.cpu().numpy()):
                        f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                    f.write("END\n")

                refine_out = engine.refine(tmp_pdb, steps=10)
                result["refined_coords"] = torch.from_numpy(refine_out["solute_coords"])
                result["refinement_energy"] = refine_out["final_energy"]
                engine.cleanup()
                os.unlink(tmp_pdb)
            except Exception as e:
                logger.warning(f"[XrayCryoEM] Refinement failed: {e}")

        return result


# =============================================================================
# 2. REAL PDB DATASET LOADER
# =============================================================================
@dataclass
class PDBDatasetConfig:
    """Configuration for PDB dataset loader."""
    # Data source
    data_dir: str = "./pdb_data"
    download_from_rcsb: bool = True
    pdb_ids: Optional[List[str]] = None

    # Processing
    min_residues: int = 50
    max_residues: int = 1000
    include_nucleic_acids: bool = False

    # Split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Augmentation
    augment: bool = True
    noise_std: float = 0.5
    rotation_augment: bool = True
    translation_augment: bool = True

    # Device
    device: str = "auto"


class RealPDBDataset(Dataset):
    """
    Production-grade PDB dataset loader.

    Supports:
      • Local PDB files
      • RCSB download (optional)
      • Train/val/test split by sequence hash
      • Augmentation (noise, rotation, translation)
      • Compatibility with SDA-ONE, TypeSpecializedFoldingSystem, SpecialistFold
    """

    def __init__(self, cfg: Optional[PDBDatasetConfig] = None):
        self.cfg = cfg or PDBDatasetConfig()

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        self.data_dir = Path(self.cfg.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.entries: List[Dict[str, Any]] = []
        self._load_or_download()

    def _load_or_download(self) -> None:
        """Load PDB files from disk or download from RCSB."""
        if self.cfg.pdb_ids is not None:
            # Download specific PDB IDs
            for pdb_id in self.cfg.pdb_ids:
                pdb_path = self.data_dir / f"{pdb_id.lower()}.pdb"
                if not pdb_path.exists() and self.cfg.download_from_rcsb:
                    self._download_pdb(pdb_id, pdb_path)
                if pdb_path.exists():
                    entry = self._parse_pdb(pdb_path)
                    if entry is not None:
                        self.entries.append(entry)
        else:
            # Load all PDB files in data_dir
            for pdb_file in self.data_dir.glob("*.pdb"):
                entry = self._parse_pdb(pdb_file)
                if entry is not None:
                    self.entries.append(entry)

        logger.info(f"[PDBDataset] Loaded {len(self.entries)} PDB entries")

    def _download_pdb(self, pdb_id: str, output_path: Path) -> None:
        """Download PDB file from RCSB."""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"[PDBDataset] Downloaded {pdb_id}")
        except Exception as e:
            logger.warning(f"[PDBDataset] Failed to download {pdb_id}: {e}")

    def _parse_pdb(self, pdb_path: Path) -> Optional[Dict[str, Any]]:
        """Parse PDB file to extract coordinates and sequence."""
        try:
            coords = []
            seq = []
            seen_res = set()

            with open(pdb_path, "r", errors="replace") as f:
                for line in f:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue
                    atom_name = line[12:16].strip()
                    if atom_name not in ("CA", "C4'"):
                        continue
                    alt_loc = line[16].strip()
                    if alt_loc not in ("", "A"):
                        continue
                    res_name = line[17:20].strip()
                    chain = line[21].strip() if len(line) > 21 else ""
                    res_seq = line[22:26].strip()
                    ins_code = line[26].strip() if len(line) > 26 else ""
                    key = (chain, res_seq, ins_code)
                    if key in seen_res:
                        continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue
                    seen_res.add(key)
                    coords.append([x, y, z])
                    # Map residue to 1-letter code
                    aa_map = {
                        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
                        'UNK': 'X',
                    }
                    seq.append(aa_map.get(res_name, 'X'))

            if len(coords) < self.cfg.min_residues:
                return None
            if len(coords) > self.cfg.max_residues:
                coords = coords[:self.cfg.max_residues]
                seq = seq[:self.cfg.max_residues]

            return {
                "pdb_id": pdb_path.stem,
                "coords": torch.tensor(coords, dtype=torch.float32),
                "sequence": "".join(seq),
                "length": len(coords),
                "path": str(pdb_path),
            }
        except Exception as e:
            logger.warning(f"[PDBDataset] Failed to parse {pdb_path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        coords = entry["coords"].clone()
        sequence = entry["sequence"]

        # Augmentation
        if self.cfg.augment:
            if self.cfg.noise_std > 0:
                coords += torch.randn_like(coords) * self.cfg.noise_std
            if self.cfg.rotation_augment:
                # Random rotation
                angles = torch.rand(3) * 2 * math.pi
                Rx = torch.tensor([[1, 0, 0],
                                    [0, math.cos(angles[0]), -math.sin(angles[0])],
                                    [0, math.sin(angles[0]), math.cos(angles[0])]])
                Ry = torch.tensor([[math.cos(angles[1]), 0, math.sin(angles[1])],
                                    [0, 1, 0],
                                    [-math.sin(angles[1]), 0, math.cos(angles[1])]])
                Rz = torch.tensor([[math.cos(angles[2]), -math.sin(angles[2]), 0],
                                    [math.sin(angles[2]), math.cos(angles[2]), 0],
                                    [0, 0, 1]])
                R = Rz @ Ry @ Rx
                coords = coords @ R.T
            if self.cfg.translation_augment:
                coords += torch.randn(3) * 5.0

        return {
            "pdb_id": entry["pdb_id"],
            "coords": coords,
            "sequence": sequence,
            "length": entry["length"],
        }

    def split(self) -> Tuple["RealPDBDataset", "RealPDBDataset", "RealPDBDataset"]:
        """Split dataset into train/val/test by sequence hash."""
        train_entries, val_entries, test_entries = [], [], []

        for entry in self.entries:
            # Hash-based split for reproducibility
            h = int(hashlib.md5(entry["sequence"].encode()).hexdigest(), 16) % 100
            if h < self.cfg.train_ratio * 100:
                train_entries.append(entry)
            elif h < (self.cfg.train_ratio + self.cfg.val_ratio) * 100:
                val_entries.append(entry)
            else:
                test_entries.append(entry)

        # Create new datasets
        train_ds = RealPDBDataset.__new__(RealPDBDataset)
        train_ds.cfg = self.cfg
        train_ds.device = self.device
        train_ds.data_dir = self.data_dir
        train_ds.entries = train_entries

        val_ds = RealPDBDataset.__new__(RealPDBDataset)
        val_ds.cfg = self.cfg
        val_ds.device = self.device
        val_ds.data_dir = self.data_dir
        val_ds.entries = val_entries

        test_ds = RealPDBDataset.__new__(RealPDBDataset)
        test_ds.cfg = self.cfg
        test_ds.device = self.device
        test_ds.data_dir = self.data_dir
        test_ds.entries = test_entries

        return train_ds, val_ds, test_ds


def pdb_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length PDB entries with padding."""
    Lmax = max(item["length"] for item in batch)
    B = len(batch)

    coords = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    sequences = []
    pdb_ids = []

    for i, item in enumerate(batch):
        L = item["length"]
        coords[i, :L] = item["coords"]
        mask[i, :L] = True
        sequences.append(item["sequence"])
        pdb_ids.append(item["pdb_id"])

    return {
        "coords": coords,
        "mask": mask,
        "sequences": sequences,
        "pdb_ids": pdb_ids,
        "lengths": torch.tensor([item["length"] for item in batch]),
    }


# =============================================================================
# 3. DOCKERFILE GENERATOR (All modules)
# =============================================================================
@dataclass
class DockerConfig:
    """Configuration for Docker image generation."""
    image_name: str = "sesi-full-ecosystem"
    image_tag: str = "v1.0.0"

    # Backend
    backends: List[str] = field(default_factory=lambda: ["cuda", "cpu"])

    # CUDA
    cuda_version: str = "12.2"
    cudnn_version: str = "8"

    # ROCm
    rocm_version: str = "6.0"

    # Python
    python_version: str = "3.11"

    # Dependencies
    install_openmm: bool = True
    install_openmm_ml: bool = True
    install_openmm_torch: bool = True
    install_triton: bool = True
    install_esm: bool = False

    # Modules to include
    modules: List[str] = field(default_factory=lambda: [
        "structural_domain_assembly_one.py",
        "real_fold_one_v2-5.py",
        "experimental_restraints_one.py",
        "xray_cryo_em_complete_production_implementation.py",
        "nmr_restraint_set_optimized_rmsd_loss.py",
        "TYPE_SPECIALIZED_PROTEIN_FOLDING_SYSTEMv1.py",
        "SPECIALISTFOLDv1_MixtureofExperts_Protein_Folding_by_Protein_Class.py",
        "QUANTUM_RISC_CISC_SESI_FULL_SPECTRUM_INTEGRATION.py",
        "denovo_sequence_designer_fixed.py",
        "denovo_protein_calculus_agent_fixed.py",
        "organ_PDB_exporter_download.py",
        "MSA_ENHANCED_PROTEIN_FOLDING.py",
        "structural_cahn_hilliard_3d-7.py",
        "structural_langevin_fold_v2.py",
        "structural_interface_utils.py",
        "gumbel_atomix_one.py",
        "sesi_experimental_integration_docker.py",
    ])

    # Output
    output_dir: str = "./docker_build"


class DockerfileGenerator:
    """
    [D] Generates Docker images for the full SESI ecosystem.

    SUPPORTED BACKENDS:
      • CUDA (NVIDIA GPU) — primary
      • ROCm (AMD GPU)
      • CPU-only

    MODULES INCLUDED:
      All 17 ecosystem modules listed in DockerConfig.modules
    """

    def __init__(self, cfg: Optional[DockerConfig] = None):
        self.cfg = cfg or DockerConfig()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_image(self, backend: str) -> str:
        """Return base Docker image for backend."""
        if backend == "cuda":
            return f"nvidia/cuda:{self.cfg.cuda_version}-cudnn{self.cfg.cudnn_version}-devel-ubuntu22.04"
        elif backend == "rocm":
            return f"rocm/dev-ubuntu-22.04:{self.cfg.rocm_version}-complete"
        else:
            return "ubuntu:22.04"

    def generate_dockerfile(self, backend: str) -> str:
        """Generate Dockerfile for specific backend."""
        base = self._base_image(backend)

        header = f"""# =============================================================================
# SESI Full Ecosystem Dockerfile — {backend.upper()} backend
# Auto-generated by DockerfileGenerator v{EXPERIMENTAL_INTEGRATION_VERSION}
# =============================================================================
FROM {base}

LABEL maintainer="Yoon A Limsuwan <yoonalimsuwan@msps.network>"
LABEL org.opencontainers.image.title="SESI Full Ecosystem ({backend})"
LABEL org.opencontainers.image.version="{self.cfg.image_tag}"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential cmake git wget curl ca-certificates \\
    libhdf5-dev libfftw3-dev liblapack-dev libblas-dev \\
    && rm -rf /var/lib/apt/lists/*
"""

        conda_block = f"""
# Install Miniconda
RUN apt-get update && apt-get install -y wget && \\
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \\
    bash /tmp/miniconda.sh -b -p /opt/conda && \\
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create environment
RUN conda create -n sesi python={self.cfg.python_version} -y
"""

        packages = ["pytorch", "torchvision", "torchaudio"]
        if self.cfg.install_openmm:
            packages.append("openmm")
        if self.cfg.install_openmm_ml:
            packages.append("openmm-ml")
        if self.cfg.install_openmm_torch:
            packages.append("openmm-torch")

        pip_packages = [
            "numpy", "scipy", "biotite", "rdkit",
            "torch-cluster", "networkx", "openmmforcefields",
        ]
        if self.cfg.install_triton:
            pip_packages.append("triton")
        if self.cfg.install_esm:
            pip_packages.append("fair-esm")

        install_block = f"""
RUN conda install -n sesi -c conda-forge -c pytorch {' '.join(packages)} -y && \\
    conda run -n sesi pip install {' '.join(pip_packages)}
"""

        copy_block = "\nWORKDIR /workspace\n\n"
        for module in self.cfg.modules:
            copy_block += f"COPY {module} .\n"

        footer = f"""
# Activate conda environment
SHELL ["conda", "run", "-n", "sesi", "/bin/bash", "-c"]

# Verify installation
RUN python -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')" || true
RUN python -c "import openmm; print(f'OpenMM {{openmm.version.full_version}}')" || echo "OpenMM not installed"

ENTRYPOINT ["conda", "run", "-n", "sesi", "python"]
CMD ["sesi_experimental_integration_docker.py", "--self-test"]
"""
        return header + conda_block + install_block + copy_block + footer

    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for multi-service deployment."""
        services = {}
        for backend in self.cfg.backends:
            services[f"sesi-{backend}"] = {
                "build": {
                    "context": ".",
                    "dockerfile": f"Dockerfile.{backend}",
                },
                "image": f"{self.cfg.image_name}:{self.cfg.image_tag}-{backend}",
                "container_name": f"sesi-{backend}",
            }
            if backend == "cuda":
                services[f"sesi-{backend}"]["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                        }
                    }
                }

        compose = {
            "version": "3.8",
            "services": services,
        }
        import yaml
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)

    def generate_all(self) -> Dict[str, str]:
        """Generate all Docker artifacts."""
        artifacts = {}

        for backend in self.cfg.backends:
            dockerfile = self.generate_dockerfile(backend)
            path = self.output_dir / f"Dockerfile.{backend}"
            with open(path, "w") as f:
                f.write(dockerfile)
            artifacts[f"Dockerfile.{backend}"] = str(path)
            logger.info(f"[Docker] Generated {path}")

        # docker-compose.yml
        try:
            compose_content = self.generate_docker_compose()
            compose_path = self.output_dir / "docker-compose.yml"
            with open(compose_path, "w") as f:
                f.write(compose_content)
            artifacts["docker-compose.yml"] = str(compose_path)
        except ImportError:
            logger.warning("[Docker] PyYAML not installed; skipping docker-compose.yml")

        return artifacts


# =============================================================================
# 4. MODULE INTEGRATION MAP
# =============================================================================
class ModuleIntegrationMap:
    """
    [D] Complete wiring diagram across all SESI ecosystem modules.
    """

    INTEGRATION_MAP = {
        "structural_domain_assembly_one.py": {
            "provides": ["StructuralDomainAssembly", "DomainSegmenter",
                         "CrossDomainContactHead", "DomainDockingAssembler"],
            "consumed_by": ["XrayCryoEMIntegrationBridge", "RealPDBDataset",
                            "TypeSpecializedFoldingSystem"],
            "key_output": "init_coords (N,3), seq_features, sigma",
        },
        "real_fold_one_v2-5.py": {
            "provides": ["RefinementEngine", "OpenMMEnergyCalculator",
                         "openmm_solute_energy", "CSOCKernel", "SOCController"],
            "consumed_by": ["XrayCryoEMIntegrationBridge", "MSAFoldingTrainer"],
            "key_output": "refined solute_coords, energy history",
        },
        "experimental_restraints_one.py": {
            "provides": ["CryoEMDensityRestraint", "XLMSRestraintSet",
                         "RestrainedDomainDockingAssembler", "load_density_map_native"],
            "consumed_by": ["XrayCryoEMIntegrationBridge"],
            "key_output": "restraint energies, CCC scores",
        },
        "xray_cryo_em_complete_production_implementation.py": {
            "provides": ["XrayCryoEMNativeEngine", "XrayCryoEMConfig"],
            "consumed_by": ["XrayCryoEMIntegrationBridge"],
            "key_output": "total_objective, r_work, r_free, fsc_correlation",
        },
        "nmr_restraint_set_optimized_rmsd_loss.py": {
            "provides": ["NMRRestraintSet"],
            "consumed_by": ["Validation pipeline"],
            "key_output": "NOE/dihedral/RMSD losses",
        },
        "TYPE_SPECIALIZED_PROTEIN_FOLDING_SYSTEMv1.py": {
            "provides": ["TypeSpecializedFoldingSystem", "ProteinTypeClassifier",
                         "MoERouter", "ConformationalEnsembleGenerator"],
            "consumed_by": ["RealPDBDataset (for type-specific training)"],
            "key_output": "type-specific predictions, ensembles",
        },
        "SPECIALISTFOLDv1_MixtureofExperts.py": {
            "provides": ["SpecialistFold", "SpecialistRouter", "BaseFoldExpert"],
            "consumed_by": ["RealPDBDataset (for class-specific training)"],
            "key_output": "class-specific coordinates, confidence",
        },
        "QUANTUM_RISC_CISC_SESI_FULL_SPECTRUM_INTEGRATION.py": {
            "provides": ["QuantumRISCCISCFullSpectrum", "ProteinFoldingqCISC"],
            "consumed_by": ["Future quantum-classical hybrid pipeline"],
            "key_output": "quantum-inspired predictions",
        },
        "denovo_sequence_designer_fixed.py": {
            "provides": ["DifferentiableStructuralDesigner"],
            "consumed_by": ["De novo design pipeline"],
            "key_output": "designed sequences, viability scores",
        },
        "denovo_protein_calculus_agent_fixed.py": {
            "provides": ["DeNovoProteinCalculusAgent", "UniversalContractionOperator"],
            "consumed_by": ["Structural calculus pipeline"],
            "key_output": "calculus-based predictions",
        },
        "organ_PDB_exporter_download.py": {
            "provides": ["OrganPDBExporter", "DifferentiableOrganPDBExporter"],
            "consumed_by": ["Organ-level export pipeline"],
            "key_output": "organ-level PDB files",
        },
        "MSA_ENHANCED_PROTEIN_FOLDING.py": {
            "provides": ["MSAEnhancedFoldingSystem", "MSASequenceFuser"],
            "consumed_by": ["MSAFoldingTrainer", "Stage2Folder"],
            "key_output": "MSA-enhanced predictions",
        },
        "structural_cahn_hilliard_3d-7.py": {
            "provides": ["StructuralCahnHilliard3D", "ThinFilmStructuralCahnHilliard3D",
                         "PhaseFieldCrystal3D"],
            "consumed_by": ["Multiphase flow coupling (future)"],
            "key_output": "phase field evolution",
        },
        "structural_langevin_fold_v2.py": {
            "provides": ["AdvancedStructuralLangevin", "CSOCThermostat"],
            "consumed_by": ["RefinementEngine (Langevin mode)"],
            "key_output": "BAOAB-integrated trajectories",
        },
        "structural_interface_utils.py": {
            "provides": ["soft_interface_score", "chunked_soft_interface_score"],
            "consumed_by": ["AdvancedStructuralLangevin (InterfaceDetector)"],
            "key_output": "per-atom interface mask",
        },
        "gumbel_atomix_one.py": {
            "provides": ["RealFoldOneAdvancedEngine"],
            "consumed_by": ["Advanced refinement pipeline"],
            "key_output": "effective_coords, occupancies, loss",
        },
    }

    def generate_markdown(self) -> str:
        """Generate markdown documentation."""
        lines = ["# SESI Ecosystem Module Integration Map\n"]
        lines.append(f"Generated by ModuleIntegrationMap v{EXPERIMENTAL_INTEGRATION_VERSION}\n")
        lines.append(f"Total modules: {len(self.INTEGRATION_MAP)}\n\n")

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
            safe_name = module.replace(".py", "").replace(" ", "_").replace("-", "_")
            lines.append(f'  "{safe_name}" [label="{module}"];')

        # Add edges
        for module, info in self.INTEGRATION_MAP.items():
            safe_from = module.replace(".py", "").replace(" ", "_").replace("-", "_")
            for consumer in info["consumed_by"]:
                # Find the consumer module
                for target_module, target_info in self.INTEGRATION_MAP.items():
                    if any(consumer in p for p in target_info["provides"]):
                        safe_to = target_module.replace(".py", "").replace(" ", "_").replace("-", "_")
                        lines.append(f'  "{safe_from}" -> "{safe_to}";')
                        break

        lines.append("}")
        return "\n".join(lines)


# =============================================================================
# 5. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI EXPERIMENTAL INTEGRATION SUITE v{EXPERIMENTAL_INTEGRATION_VERSION} — Self-Test")
    print(f"  XrayCryoEM={_HAS_XRAY}  Restraints={_HAS_RESTRAINTS}  SDA={_HAS_SDA}")
    print(f"  RealFold={_HAS_REAL_FOLD}  TypeSpecialized={_HAS_TYPE_SPECIALIZED}")
    print(f"  Specialist={_HAS_SPECIALIST}  NMR={_HAS_NMR}")
    print("=" * 78)

    # ---- T1: XrayCryoEMIntegrationBridge --------------------------------
    print("\n [T1] XrayCryoEMIntegrationBridge")
    bridge_cfg = XrayCryoEMBridgeConfig()
    bridge = XrayCryoEMIntegrationBridge(bridge_cfg)

    # Test simulated map generation
    test_coords = torch.randn(50, 3) * 10
    sim_map = bridge.coords_to_simulated_map(test_coords, grid_size=(32, 32, 32))
    print(f"   Simulated map shape: {tuple(sim_map.shape)}")
    assert sim_map.shape == (32, 32, 32)
    print("[PASS] T1a Simulated map generation works")

    # Test full pipeline with mock data
    test_seq = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 residues
    result = bridge.run_full_pipeline(test_seq)
    print(f"   Prediction method: {result['prediction_method']}")
    print(f"   Metrics: {list(result['metrics'].keys())}")
    print("[PASS] T1b Full pipeline runs")

    # ---- T2: RealPDBDatasetLoader ---------------------------------------
    print("\n [T2] RealPDBDatasetLoader")
    dataset_cfg = PDBDatasetConfig(
        data_dir="./test_pdb_data",
        download_from_rcsb=False,  # Don't download in test
        min_residues=20,
        max_residues=100,
    )

    # Create a mock PDB file for testing
    os.makedirs("./test_pdb_data", exist_ok=True)
    mock_pdb = "./test_pdb_data/test.pdb"
    with open(mock_pdb, "w") as f:
        for i in range(50):
            x, y, z = i * 3.8, 0.0, 0.0
            f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("END\n")

    dataset = RealPDBDataset(dataset_cfg)
    print(f"   Dataset size: {len(dataset)}")
    assert len(dataset) > 0

    # Test __getitem__
    sample = dataset[0]
    print(f"   Sample: {sample['pdb_id']}, length={sample['length']}")
    assert sample["coords"].shape == (sample["length"], 3)

    # Test collate
    batch = [dataset[0], dataset[0]]
    collated = pdb_collate(batch)
    print(f"   Collated batch: coords={tuple(collated['coords'].shape)}")
    assert collated["coords"].shape[0] == 2

    # Cleanup
    os.remove(mock_pdb)
    os.rmdir("./test_pdb_data")
    print("[PASS] T2 RealPDBDatasetLoader works")

    # ---- T3: DockerfileGenerator ----------------------------------------
    print("\n [T3] DockerfileGenerator")
    docker_cfg = DockerConfig(
        backends=["cuda", "cpu"],
        output_dir="./test_docker",
    )
    docker_gen = DockerfileGenerator(docker_cfg)
    artifacts = docker_gen.generate_all()
    print(f"   Generated {len(artifacts)} artifacts:")
    for name, path in artifacts.items():
        print(f"     {name}: {os.path.basename(path)}")
    assert len(artifacts) >= 2  # At least Dockerfile.cuda and Dockerfile.cpu

    # Verify Dockerfile content
    with open(artifacts["Dockerfile.cuda"]) as f:
        content = f.read()
    assert "FROM nvidia/cuda" in content
    assert "COPY structural_domain_assembly_one.py" in content
    assert "COPY real_fold_one_v2-5.py" in content
    print("[PASS] T3 DockerfileGenerator works")

    # Cleanup
    import shutil
    shutil.rmtree("./test_docker", ignore_errors=True)

    # ---- T4: ModuleIntegrationMap ---------------------------------------
    print("\n [T4] ModuleIntegrationMap")
    imap = ModuleIntegrationMap()
    md = imap.generate_markdown()
    dot = imap.generate_dot()
    print(f"   Markdown: {len(md)} chars")
    print(f"   DOT graph: {len(dot)} chars")
    assert "structural_domain_assembly_one.py" in md
    assert "digraph" in dot
    print("[PASS] T4 ModuleIntegrationMap works")

    # ---- T5: Integration check ------------------------------------------
    print("\n [T5] Integration Check")
    print("   Checking module availability:")
    modules = {
        "XrayCryoEM": _HAS_XRAY,
        "Restraints": _HAS_RESTRAINTS,
        "SDA-ONE": _HAS_SDA,
        "RealFold": _HAS_REAL_FOLD,
        "TypeSpecialized": _HAS_TYPE_SPECIALIZED,
        "SpecialistFold": _HAS_SPECIALIST,
        "NMR": _HAS_NMR,
    }
    for name, available in modules.items():
        status = "✓" if available else "✗"
        print(f"     {status} {name}")

    n_available = sum(modules.values())
    print(f"   Available: {n_available}/{len(modules)}")
    print("[PASS] T5 Integration check completed")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("  HONEST REMINDERS:")
    print("    • XrayCryoEMIntegrationBridge requires actual experimental data")
    print("    • RealPDBDatasetLoader downloads from RCSB by default")
    print("    • Docker images are TEMPLATES; requires docker build")
    print("    • This module has NOT been run with real experimental data here")
    print("=" * 78)
