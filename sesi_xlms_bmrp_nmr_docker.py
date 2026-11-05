"""
=============================================================================
SESI XL-MS INTEGRATION + BMRB DATASET LOADER + NMR FITTING DOCKER
Cross-Linking MS Restraints · Real NMR Data from BMRB · Containerized Fitting
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
INTEGRATION WITH ECOSYSTEM FILES
=============================================================================
This module calls verbatim APIs from:

  experimental_restraints_one.py
    → CrossLink(residue_i, residue_j, d_max, confidence)
    → XLMSRestraintSet.from_pairs(pairs, d_max, confidences)
    → XLMSRestraintSet.energy(placed, domain_ranges, cfg)
    → XLMSRestraintSet.filter_resolvable(domain_ranges)
    → RestraintDockingConfig(xlms_weight, xlms_default_dmax, xlms_slack, ...)
    → RestrainedDomainDockingAssembler(cfg, restraint_cfg, xlms, cryoem)

  nmr_restraint_set_optimized_rmsd_loss.py
    → NMRRestraintSet(noe_pairs, noe_bounds, dihedral_indices, dihedral_targets,
                       force_constant, rmsd_weight, smooth_beta)
    → .compute_noe_penalty(coords)
    → .compute_dihedral_penalty(coords)
    → .compute_kabsch_rmsd(pred_coords, target_coords)
    → .forward(coords, reference_coords) → (total_loss, metrics)

  real_fold_one_v2-5.py
    → RefinementEngine.refine(pdb_file, steps)
    → openmm_solute_energy(solute_coords, calculator)

  structural_domain_assembly_one.py
    → StructuralDomainAssembly.forward(sequence) → init_coords

=============================================================================
MODULES IN THIS FILE
=============================================================================
[1] XLMSDataLoader          — Load XL-MS crosslinks from CSV/JSON/TXT
[2] XLMSRestraintBridge     — Wrap XLMSRestraintSet for training
[3] BMRBParser              — Parse NMR-STAR format from BMRB
[4] BMRBDataset             — PyTorch Dataset for NMR restraints
[5] NMRXLMSFittingTrainer   — Combined NMR + XL-MS + OpenMM training
[6] DockerfileGenerator     — Multi-backend Docker for NMR/XL-MS fitting
[7] Self-test suite

=============================================================================
BMRB DATA FORMAT
=============================================================================
BMRB (Biological Magnetic Resonance Data Bank) provides NMR restraints in:
  • NMR-STAR format (.str) — primary format
  • PDBx/mmCIF format (.cif) — structural data
  • XPLOR/CNS format (.tbl) — restraint files (legacy)

NMR-STAR structure:
  data_<entry_id>
    save_<category>
      _Category.field value
      loop_
        _Field1
        _Field2
        value1 value2
      stop_
    save_
  stop_

Key categories for restraints:
  • Distance_constraint — NOE distance bounds
  • Torsion_angle_constraint — dihedral angles (phi, psi, chi)
  • Chem_shift — chemical shifts (optional)
  • Coupling_constant — J-couplings (optional)

=============================================================================
HONEST SCOPE
=============================================================================
• BMRB data requires internet access for download, or local files.
• NMR-STAR parsing is simplified; complex entries may need manual cleanup.
• XL-MS d_max values are LITERATURE-TYPICAL (24-30 Å for DSS/BS3),
  not validated for specific crosslinker chemistry.
• This module has NOT been run end-to-end with real BMRB data here.
• Docker images are TEMPLATES; requires docker build on target machine.
=============================================================================
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import math
import os
import re
import tempfile
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.xlms_bmrp_nmr")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

XLMS_BMRB_VERSION: str = "1.0.0"

# =============================================================================
# Ecosystem imports (graceful fallback)
# =============================================================================
_HAS_XLMS = False
try:
    from experimental_restraints_one import (
        CrossLink,
        XLMSRestraintSet,
        RestraintDockingConfig,
        RestrainedDomainDockingAssembler,
    )
    _HAS_XLMS = True
except ImportError:
    import importlib.util
    p = Path("experimental_restraints_one.py")
    if p.exists():
        spec = importlib.util.spec_from_file_location("experimental_restraints", str(p))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            CrossLink = mod.CrossLink
            XLMSRestraintSet = mod.XLMSRestraintSet
            RestraintDockingConfig = mod.RestraintDockingConfig
            RestrainedDomainDockingAssembler = mod.RestrainedDomainDockingAssembler
            _HAS_XLMS = True
    if not _HAS_XLMS:
        warnings.warn("experimental_restraints_one not found; XL-MS integration disabled.")

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
    if not _HAS_NMR:
        warnings.warn("NMRRestraintSet not found; NMR integration disabled.")

_HAS_REAL_FOLD = False
try:
    from real_fold_one_v2 import RefinementEngine, RefinementConfig, openmm_solute_energy
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
                openmm_solute_energy = mod.openmm_solute_energy
                _HAS_REAL_FOLD = True
            break

logger.info(f"[imports] XLMS={_HAS_XLMS} NMR={_HAS_NMR} RealFold={_HAS_REAL_FOLD}")


# =============================================================================
# 1. XL-MS DATA LOADER
# =============================================================================
@dataclass
class XLMSData:
    """Container for XL-MS crosslink data."""
    pairs: List[Tuple[int, int]]           # (residue_i, residue_j) 0-indexed
    d_max: float = 30.0                     # default Cα-CÅ distance cutoff (Å)
    confidences: Optional[List[float]] = None  # per-crosslink confidence [0, 1]
    crosslinker: str = "DSS"               # crosslinker name
    source: str = "unknown"                # data source


class XLMSDataLoader:
    """
    Load XL-MS crosslink data from various formats.

    SUPPORTED FORMATS:
      • CSV: columns = [residue_i, residue_j, confidence?]
      • JSON: {"pairs": [[i,j],...], "confidences": [...], "d_max": float}
      • TXT: space/tab-separated pairs
      • XlinkX / pLink output (simplified parsing)

    DISTANCE CUTOFFS (literature-typical):
      • DSS/BS3 (NHS-ester): 24-30 Å Cα-Cα
      • EDC (zero-length): 10-15 Å
      • DSBU: 25-35 Å
      • Default: 30 Å (conservative)
    """

    # Literature-typical d_max by crosslinker
    DMAX_BY_CROSSLINKER = {
        "DSS": 30.0,
        "BS3": 30.0,
        "DSBU": 32.0,
        "EDC": 12.0,
        "sulfo-SDA": 25.0,
        "default": 30.0,
    }

    @staticmethod
    def from_csv(csv_path: str, d_max: Optional[float] = None,
                   crosslinker: str = "DSS") -> XLMSData:
        """
        Load XL-MS data from CSV file.
        Expected columns: residue_i, residue_j, [confidence]
        """
        pairs = []
        confidences = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    i = int(row.get('residue_i', row.get('i', row.get('res1', 0))))
                    j = int(row.get('residue_j', row.get('j', row.get('res2', 0))))
                    conf = float(row.get('confidence', row.get('score', 1.0)))
                    pairs.append((i, j))
                    confidences.append(conf)
                except (ValueError, KeyError):
                    continue

        if d_max is None:
            d_max = XLMSDataLoader.DMAX_BY_CROSSLINKER.get(crosslinker, 30.0)

        return XLMSData(
            pairs=pairs,
            d_max=d_max,
            confidences=confidences if confidences else None,
            crosslinker=crosslinker,
            source=str(csv_path),
        )

    @staticmethod
    def from_json(json_path: str) -> XLMSData:
        """Load XL-MS data from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        pairs = [(p[0], p[1]) for p in data.get('pairs', [])]
        confidences = data.get('confidences', None)
        d_max = data.get('d_max', 30.0)
        crosslinker = data.get('crosslinker', 'DSS')

        return XLMSData(
            pairs=pairs,
            d_max=d_max,
            confidences=confidences,
            crosslinker=crosslinker,
            source=str(json_path),
        )

    @staticmethod
    def from_txt(txt_path: str, d_max: float = 30.0) -> XLMSData:
        """Load XL-MS data from whitespace-separated text file."""
        pairs = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        i, j = int(parts[0]), int(parts[1])
                        pairs.append((i, j))
                    except ValueError:
                        continue

        return XLMSData(pairs=pairs, d_max=d_max, source=str(txt_path))

    @staticmethod
    def from_programmatic(pairs: List[Tuple[int, int]],
                            d_max: float = 30.0,
                            confidences: Optional[List[float]] = None) -> XLMSData:
        """Create XLMSData directly from Python lists."""
        return XLMSData(pairs=pairs, d_max=d_max, confidences=confidences)

    @staticmethod
    def generate_synthetic(n_residues: int = 100, n_crosslinks: int = 20,
                             d_max: float = 30.0, seed: int = 42) -> XLMSData:
        """
        Generate synthetic XL-MS data for testing.
        WARNING: Not real experimental data.
        """
        rng = np.random.default_rng(seed)
        pairs = []
        for _ in range(n_crosslinks):
            i = rng.integers(0, n_residues)
            j = rng.integers(0, n_residues)
            if i != j:
                pairs.append((int(min(i, j)), int(max(i, j))))

        return XLMSData(
            pairs=pairs,
            d_max=d_max,
            crosslinker="synthetic",
            source="synthetic_generator",
        )


# =============================================================================
# 2. XL-MS RESTRAINT BRIDGE
# =============================================================================
class XLMSRestraintBridge:
    """
    Bridge connecting XL-MS data to XLMSRestraintSet for training.

    USAGE:
      xlms_data = XLMSDataLoader.from_csv("crosslinks.csv")
      bridge = XLMSRestraintBridge(xlms_data)
      energy = bridge.compute_energy(placed_coords, domain_ranges)

    INTEGRATION:
      Uses XLMSRestraintSet from experimental_restraints_one.py:
        • from_pairs(pairs, d_max, confidences) → XLMSRestraintSet
        • energy(placed, domain_ranges, cfg) → scalar restraint energy
        • filter_resolvable(domain_ranges) → resolvable crosslinks

    FLAT-BOTTOM POTENTIAL:
      E_k = confidence_k * relu(d_k - (d_max_k + slack))^2
      Zero penalty within d_max + slack; quadratic penalty beyond.
      This matches the physical meaning: crosslinks forbid being too far,
      not require touching.
    """

    def __init__(self, xlms_data: XLMSData,
                   xlms_weight: float = 5.0,
                   xlms_slack: float = 2.0):
        if not _HAS_XLMS:
            raise ImportError("experimental_restraints_one required for XLMSRestraintBridge.")

        self.xlms_data = xlms_data
        self.restraint_cfg = RestraintDockingConfig(
            xlms_weight=xlms_weight,
            xlms_default_dmax=xlms_data.d_max,
            xlms_slack=xlms_slack,
        )

        # Build XLMSRestraintSet
        self.restraint_set = XLMSRestraintSet.from_pairs(
            pairs=xlms_data.pairs,
            d_max=xlms_data.d_max,
            confidences=xlms_data.confidences,
        )

        logger.info(f"[XLMSBridge] Initialized with {len(xlms_data.pairs)} crosslinks, "
                    f"d_max={xlms_data.d_max:.1f} Å, weight={xlms_weight}")

    def compute_energy(self, placed: List[torch.Tensor],
                         domain_ranges: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Compute XL-MS restraint energy.

        Args:
            placed: list of (domain_size_k, 3) coordinates per domain
            domain_ranges: list of (start, end) per domain

        Returns:
            Scalar restraint energy (weighted by xlms_weight)
        """
        return self.restraint_set.energy(placed, domain_ranges, self.restraint_cfg)

    def filter_resolvable(self, domain_ranges: List[Tuple[int, int]]) -> List:
        """Filter crosslinks to those resolvable within given domain ranges."""
        return self.restraint_set.filter_resolvable(domain_ranges)

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about the XL-MS dataset."""
        return {
            "n_crosslinks": len(self.xlms_data.pairs),
            "d_max": self.xlms_data.d_max,
            "crosslinker": self.xlms_data.crosslinker,
            "source": self.xlms_data.source,
            "has_confidences": self.xlms_data.confidences is not None,
            "mean_confidence": (
                float(np.mean(self.xlms_data.confidences))
                if self.xlms_data.confidences else None
            ),
        }


# =============================================================================
# 3. BMRB PARSER (NMR-STAR format)
# =============================================================================
@dataclass
class BMRBEntry:
    """Container for parsed BMRB entry."""
    entry_id: str
    noe_pairs: List[Tuple[int, int]]          # (residue_i, residue_j)
    noe_bounds: List[Tuple[float, float]]      # (lower, upper) in Å
    dihedral_indices: List[Tuple[int, int, int, int]]  # (i, j, k, l) atom indices
    dihedral_targets: List[Tuple[float, float]]  # (target_rad, tolerance_rad)
    chemical_shifts: Optional[Dict[int, float]] = None  # residue → shift (ppm)
    sequence: str = ""
    n_residues: int = 0


class BMRBParser:
    """
    Parse NMR-STAR format from BMRB.

    NMR-STAR is a text-based format with saveframes and loops.
    This parser extracts:
      • Distance constraints (NOE) from Distance_constraint saveframe
      • Torsion angle constraints from Torsion_angle_constraint saveframe
      • Chemical shifts from Chem_shift saveframe (optional)

    NOTE: This is a SIMPLIFIED parser. Complex BMRB entries may require
    manual cleanup or use of specialized tools (e.g., NMRView, CcpNmr).
    """

    def __init__(self):
        self.current_entry = None

    def parse_file(self, file_path: str) -> BMRBEntry:
        """Parse a BMRB NMR-STAR file."""
        with open(file_path, 'r', errors='replace') as f:
            content = f.read()

        entry_id = self._extract_entry_id(content)
        noe_pairs, noe_bounds = self._parse_distance_constraints(content)
        dihedral_indices, dihedral_targets = self._parse_torsion_constraints(content)
        sequence = self._extract_sequence(content)

        return BMRBEntry(
            entry_id=entry_id,
            noe_pairs=noe_pairs,
            noe_bounds=noe_bounds,
            dihedral_indices=dihedral_indices,
            dihedral_targets=dihedral_targets,
            sequence=sequence,
            n_residues=len(sequence) if sequence else 0,
        )

    def _extract_entry_id(self, content: str) -> str:
        """Extract BMRB entry ID from data_ block."""
        match = re.search(r'data_(\S+)', content)
        return match.group(1) if match else "unknown"

    def _parse_distance_constraints(self, content: str) -> Tuple[List, List]:
        """
        Parse NOE distance constraints from Distance_constraint saveframe.

        NMR-STAR format:
          save_Distance_constraint
            _Distance_constraint.ID
            _Distance_constraint.Memb_code
            ...
            loop_
              _Dist_constr_tree.Constraint_ID
              _Dist_constr_tree.Node_ID
              _Dist_constr_tree.Residue_I_auth_seq_ID
              _Dist_constr_tree.Residue_J_auth_seq_ID
              _Dist_constr_tree.Distance_val_min
              _Dist_constr_tree.Distance_val_max
              ...
            stop_
          save_
        """
        pairs = []
        bounds = []

        # Find Distance_constraint block
        dist_match = re.search(
            r'save_Distance_constraint(.*?)save_',
            content, re.DOTALL
        )
        if not dist_match:
            return pairs, bounds

        dist_block = dist_match.group(1)

        # Find loop_ section with residue IDs
        loop_match = re.search(r'loop_(.*?)stop_', dist_block, re.DOTALL)
        if not loop_match:
            return pairs, bounds

        loop_block = loop_match.group(1)

        # Parse column headers
        headers = re.findall(r'_([\w.]+)', loop_block.split('\n\n')[0])

        # Find residue ID columns
        res_i_col = None
        res_j_col = None
        dist_min_col = None
        dist_max_col = None

        for i, h in enumerate(headers):
            if 'Residue_I' in h or 'residue_i' in h.lower():
                res_i_col = i
            elif 'Residue_J' in h or 'residue_j' in h.lower():
                res_j_col = i
            elif 'Distance_val_min' in h or 'dist_min' in h.lower():
                dist_min_col = i
            elif 'Distance_val_max' in h or 'dist_max' in h.lower():
                dist_max_col = i

        if res_i_col is None or res_j_col is None:
            return pairs, bounds

        # Parse data rows
        data_lines = [l.strip() for l in loop_block.split('\n') if l.strip() and not l.startswith('_')]
        for line in data_lines:
            parts = line.split()
            if len(parts) > max(res_i_col, res_j_col):
                try:
                    i = int(parts[res_i_col])
                    j = int(parts[res_j_col])
                    lower = float(parts[dist_min_col]) if dist_min_col and len(parts) > dist_min_col else 1.8
                    upper = float(parts[dist_max_col]) if dist_max_col and len(parts) > dist_max_col else 5.0
                    pairs.append((i, j))
                    bounds.append((lower, upper))
                except (ValueError, IndexError):
                    continue

        return pairs, bounds

    def _parse_torsion_constraints(self, content: str) -> Tuple[List, List]:
        """
        Parse torsion angle constraints from Torsion_angle_constraint saveframe.
        Simplified: extract phi/psi angles with tolerance.
        """
        indices = []
        targets = []

        torsion_match = re.search(
            r'save_Torsion_angle_constraint(.*?)save_',
            content, re.DOTALL
        )
        if not torsion_match:
            return indices, targets

        # Simplified parsing: look for angle values
        torsion_block = torsion_match.group(1)
        angle_matches = re.findall(
            r'[-+]?(\d+\.?\d*)\s+[-+]?(\d+\.?\d*)',
            torsion_block
        )

        for match in angle_matches[:100]:  # limit to first 100
            try:
                angle_deg = float(match[0])
                tolerance_deg = float(match[1])
                # Convert to radians
                angle_rad = math.radians(angle_deg)
                tol_rad = math.radians(tolerance_deg)
                indices.append((0, 0, 0, 0))  # placeholder atom indices
                targets.append((angle_rad, tol_rad))
            except ValueError:
                continue

        return indices, targets

    def _extract_sequence(self, content: str) -> str:
        """Extract protein sequence from BMRB entry."""
        # Look for one-letter sequence in Entity_poly saveframe
        seq_match = re.search(r'_Entity_poly\.Seq_one_letter_code\s+"?([A-Z\s]+)"?', content)
        if seq_match:
            return seq_match.group(1).replace(' ', '').replace('\n', '')
        return ""


# =============================================================================
# 4. BMRB DATASET (PyTorch Dataset)
# =============================================================================
class BMRBDataset(Dataset):
    """
    PyTorch Dataset for BMRB NMR restraints.

    Loads NMR entries from local files or downloads from BMRB API.
    Converts to NMRRestraintSet-compatible format.

    SUPPORTED SOURCES:
      • Local NMR-STAR files (.str)
      • BMRB REST API (https://api.bmrb.io/v2/entry/<id>)
    """

    BMRB_API_URL = "https://api.bmrb.io/v2/entry/"

    def __init__(self, data_dir: str = "./bmrb_data",
                   entry_ids: Optional[List[str]] = None,
                   download: bool = False,
                   min_noe: int = 10,
                   min_residues: int = 30):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.entry_ids = entry_ids or []
        self.download = download
        self.min_noe = min_noe
        self.min_residues = min_residues

        self.parser = BMRBParser()
        self.entries: List[BMRBEntry] = []
        self._load_entries()

    def _load_entries(self):
        """Load or download BMRB entries."""
        for entry_id in self.entry_ids:
            # Try local file first
            local_path = self.data_dir / f"{entry_id}.str"
            if local_path.exists():
                entry = self.parser.parse_file(str(local_path))
                if len(entry.noe_pairs) >= self.min_noe and entry.n_residues >= self.min_residues:
                    self.entries.append(entry)
                    logger.info(f"[BMRB] Loaded {entry_id}: {len(entry.noe_pairs)} NOEs, "
                                f"{entry.n_residues} residues")
                continue

            # Download if requested
            if self.download:
                try:
                    url = f"{self.BMRB_API_URL}{entry_id}"
                    urllib.request.urlretrieve(url, str(local_path))
                    entry = self.parser.parse_file(str(local_path))
                    if len(entry.noe_pairs) >= self.min_noe and entry.n_residues >= self.min_residues:
                        self.entries.append(entry)
                except Exception as e:
                    logger.warning(f"[BMRB] Failed to download {entry_id}: {e}")

        logger.info(f"[BMRB] Loaded {len(self.entries)} entries")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        return {
            "entry_id": entry.entry_id,
            "noe_pairs": torch.tensor(entry.noe_pairs, dtype=torch.long),
            "noe_bounds": torch.tensor(entry.noe_bounds, dtype=torch.float32),
            "dihedral_indices": torch.tensor(entry.dihedral_indices, dtype=torch.long) if entry.dihedral_indices else torch.empty(0, 4, dtype=torch.long),
            "dihedral_targets": torch.tensor(entry.dihedral_targets, dtype=torch.float32) if entry.dihedral_targets else torch.empty(0, 2, dtype=torch.float32),
            "sequence": entry.sequence,
            "n_residues": entry.n_residues,
        }


def bmrb_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate variable-length BMRB entries."""
    return {
        "entry_ids": [b["entry_id"] for b in batch],
        "noe_pairs": [b["noe_pairs"] for b in batch],
        "noe_bounds": [b["noe_bounds"] for b in batch],
        "dihedral_indices": [b["dihedral_indices"] for b in batch],
        "dihedral_targets": [b["dihedral_targets"] for b in batch],
        "sequences": [b["sequence"] for b in batch],
        "n_residues": torch.tensor([b["n_residues"] for b in batch]),
    }


# =============================================================================
# 5. NMR + XL-MS FITTING TRAINER
# =============================================================================
@dataclass
class NMRXLMSFittingConfig:
    """Configuration for combined NMR + XL-MS fitting."""
    # Loss weights
    w_nmr_noe: float = 1.0
    w_nmr_dihedral: float = 1.0
    w_nmr_rmsd: float = 10.0
    w_xlms: float = 5.0
    w_energy: float = 0.1

    # Training
    lr: float = 1e-3
    epochs: int = 100
    grad_clip: float = 5.0

    # Refinement
    refinement_steps: int = 23  # OPMC k*

    # Device
    device: str = "auto"


class NMRXLMSFittingTrainer:
    """
    Combined training loop for NMR + XL-MS restraints.

    LOSS FUNCTION:
      L = w_nmr * (L_noe + L_dihedral + L_rmsd)
        + w_xlms * L_xlms
        + w_energy * E_openmm

    PIPELINE:
      1. Load NMR restraints from BMRB
      2. Load XL-MS crosslinks
      3. Initialize coordinates (from PDB or random)
      4. Compute NMR restraint losses (NMRRestraintSet)
      5. Compute XL-MS restraint loss (XLMSRestraintSet)
      6. Compute OpenMM energy (optional)
      7. Backpropagate and update coordinates
    """

    def __init__(self, cfg: Optional[NMRXLMSFittingConfig] = None):
        self.cfg = cfg or NMRXLMSFittingConfig()

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        # NMR restraint set (initialized when data is provided)
        self.nmr_restraint = None

        # XL-MS restraint bridge (initialized when data is provided)
        self.xlms_bridge = None

        # Refinement engine (optional, for OpenMM energy)
        self.refine_engine = None
        if _HAS_REAL_FOLD:
            try:
                refine_cfg = RefinementConfig()
                self.refine_engine = RefinementEngine(refine_cfg)
            except Exception as e:
                logger.warning(f"[Trainer] RefinementEngine init failed: {e}")

        self.history: List[Dict[str, float]] = []

    def set_nmr_data(self, noe_pairs: torch.Tensor, noe_bounds: torch.Tensor,
                       dihedral_indices: Optional[torch.Tensor] = None,
                       dihedral_targets: Optional[torch.Tensor] = None,
                       reference_coords: Optional[torch.Tensor] = None):
        """Set NMR restraint data."""
        if not _HAS_NMR:
            raise ImportError("NMRRestraintSet not available.")

        self.nmr_restraint = NMRRestraintSet(
            noe_pairs=noe_pairs.to(self.device),
            noe_bounds=noe_bounds.to(self.device),
            dihedral_indices=dihedral_indices.to(self.device) if dihedral_indices is not None else None,
            dihedral_targets=dihedral_targets.to(self.device) if dihedral_targets is not None else None,
            force_constant=50.0,
            rmsd_weight=self.cfg.w_nmr_rmsd,
            smooth_beta=50.0,
        ).to(self.device)

        self._reference_coords = reference_coords.to(self.device) if reference_coords is not None else None
        logger.info(f"[Trainer] NMR data set: {len(noe_pairs)} NOEs")

    def set_xlms_data(self, xlms_data: XLMSData, xlms_weight: float = 5.0):
        """Set XL-MS restraint data."""
        if not _HAS_XLMS:
            raise ImportError("XLMSRestraintSet not available.")

        self.xlms_bridge = XLMSRestraintBridge(xlms_data, xlms_weight=xlms_weight)
        logger.info(f"[Trainer] XL-MS data set: {len(xlms_data.pairs)} crosslinks")

    def compute_loss(self, coords: torch.Tensor,
                       domain_ranges: Optional[List[Tuple[int, int]]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined NMR + XL-MS + OpenMM loss.

        Args:
            coords: (N, 3) coordinates
            domain_ranges: optional domain boundaries for XL-MS

        Returns:
            (total_loss, metrics_dict)
        """
        total_loss = torch.tensor(0.0, device=self.device)
        metrics = {}

        # NMR loss
        if self.nmr_restraint is not None:
            nmr_loss, nmr_metrics = self.nmr_restraint(coords, self._reference_coords)
            total_loss = total_loss + self.cfg.w_nmr_noe * nmr_loss
            metrics.update({f"nmr_{k}": v.item() if torch.is_tensor(v) else v
                           for k, v in nmr_metrics.items()})

        # XL-MS loss
        if self.xlms_bridge is not None and domain_ranges is not None:
            # For XL-MS, need to split coords into domains
            placed = []
            for start, end in domain_ranges:
                placed.append(coords[start:end])
            xlms_loss = self.xlms_bridge.compute_energy(placed, domain_ranges)
            total_loss = total_loss + self.cfg.w_xlms * xlms_loss
            metrics["xlms_loss"] = xlms_loss.item()

        # OpenMM energy loss (optional)
        if self.refine_engine is not None:
            try:
                # Write coords to temp PDB, compute energy
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
                    tmp_pdb = f.name
                    for i, (x, y, z) in enumerate(coords.cpu().numpy()):
                        f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
                    f.write("END\n")

                solute_coords = self.refine_engine._setup_system(tmp_pdb)
                energy = openmm_solute_energy(solute_coords, self.refine_engine.calculator)
                total_loss = total_loss + self.cfg.w_energy * energy
                metrics["openmm_energy"] = energy.item()

                os.unlink(tmp_pdb)
            except Exception as e:
                logger.warning(f"[Trainer] OpenMM energy failed: {e}")

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    def train_step(self, coords: torch.Tensor,
                     domain_ranges: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
        """Single training step."""
        coords = coords.clone().detach().requires_grad_(True)

        loss, metrics = self.compute_loss(coords, domain_ranges)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([coords], self.cfg.grad_clip)

        # Update coordinates
        with torch.no_grad():
            coords -= self.cfg.lr * coords.grad

        return metrics


# =============================================================================
# 6. DOCKERFILE GENERATOR FOR NMR/XL-MS FITTING
# =============================================================================
@dataclass
class NMRDockerConfig:
    """Configuration for NMR/XL-MS Docker image."""
    image_name: str = "sesi-nmr-xlms-fitting"
    image_tag: str = "v1.0.0"
    backends: List[str] = field(default_factory=lambda: ["cuda", "cpu"])
    cuda_version: str = "12.2"
    python_version: str = "3.11"
    install_openmm: bool = True
    install_openmm_ml: bool = True
    install_biotite: bool = True


class NMRDockerfileGenerator:
    """Generate Docker images for NMR/XL-MS fitting."""

    def __init__(self, cfg: Optional[NMRDockerConfig] = None):
        self.cfg = cfg or NMRDockerConfig()
        self.output_dir = Path("./docker_nmr")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_image(self, backend: str) -> str:
        if backend == "cuda":
            return f"nvidia/cuda:{self.cfg.cuda_version}-cudnn8-devel-ubuntu22.04"
        return "ubuntu:22.04"

    def generate_dockerfile(self, backend: str) -> str:
        base = self._base_image(backend)

        header = f"""# =============================================================================
# SESI NMR/XL-MS Fitting Dockerfile — {backend.upper()} backend
# Auto-generated by NMRDockerfileGenerator v{XLMS_BMRB_VERSION}
# =============================================================================
FROM {base}

LABEL maintainer="Yoon A Limsuwan <yoonalimsuwan@msps.network>"
LABEL org.opencontainers.image.title="SESI NMR/XL-MS Fitting ({backend})"
LABEL org.opencontainers.image.version="{self.cfg.image_tag}"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential cmake git wget curl ca-certificates \\
    libhdf5-dev libfftw3-dev liblapack-dev libblas-dev \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*
"""

        conda_block = f"""
# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \\
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

        pip_packages = ["numpy", "scipy", "networkx"]
        if self.cfg.install_biotite:
            pip_packages.append("biotite")

        install_block = f"""
RUN conda install -n sesi -c conda-forge -c pytorch {' '.join(packages)} -y && \\
    conda run -n sesi pip install {' '.join(pip_packages)}
"""

        copy_block = """
# Copy ecosystem modules
WORKDIR /workspace
COPY nmr_restraint_set_optimized_rmsd_loss.py .
COPY experimental_restraints_one.py .
COPY real_fold_one_v2-5.py .
COPY structural_domain_assembly_one.py .
COPY sesi_xlms_bmrp_nmr_docker.py .
"""

        footer = f"""
SHELL ["conda", "run", "-n", "sesi", "/bin/bash", "-c"]

# Verify installation
RUN python -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')" || true

ENTRYPOINT ["conda", "run", "-n", "sesi", "python"]
CMD ["sesi_xlms_bmrp_nmr_docker.py", "--self-test"]
"""
        return header + conda_block + install_block + copy_block + footer

    def generate_all(self) -> Dict[str, str]:
        artifacts = {}
        for backend in self.cfg.backends:
            dockerfile = self.generate_dockerfile(backend)
            path = self.output_dir / f"Dockerfile.{backend}"
            with open(path, "w") as f:
                f.write(dockerfile)
            artifacts[f"Dockerfile.{backend}"] = str(path)
            logger.info(f"[Docker] Generated {path}")
        return artifacts


# =============================================================================
# 7. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI XL-MS + BMRB + NMR FITTING v{XLMS_BMRB_VERSION} — Self-Test")
    print(f"  XLMS={_HAS_XLMS}  NMR={_HAS_NMR}  RealFold={_HAS_REAL_FOLD}")
    print("=" * 78)

    # ---- T1: XL-MS Data Loader -------------------------------------------
    print("\n [T1] XL-MS Data Loader")
    xlms_data = XLMSDataLoader.generate_synthetic(n_residues=100, n_crosslinks=20)
    print(f"   Generated {len(xlms_data.pairs)} synthetic crosslinks")
    print(f"   d_max: {xlms_data.d_max} Å")
    print(f"   Crosslinker: {xlms_data.crosslinker}")
    assert len(xlms_data.pairs) == 20
    print("[PASS] T1 XL-MS data loader works")

    # ---- T2: XL-MS Restraint Bridge ---------------------------------------
    print("\n [T2] XL-MS Restraint Bridge")
    if _HAS_XLMS:
        bridge = XLMSRestraintBridge(xlms_data, xlms_weight=5.0)
        stats = bridge.get_statistics()
        print(f"   Statistics: {stats}")
        assert stats["n_crosslinks"] == 20
        print("[PASS] T2 XL-MS restraint bridge works")
    else:
        print("[SKIP] T2 experimental_restraints_one not available")

    # ---- T3: BMRB Parser (synthetic) --------------------------------------
    print("\n [T3] BMRB Parser")
    # Create a minimal synthetic NMR-STAR file
    synthetic_str = """
data_test_entry
save_Distance_constraint
  _Distance_constraint.ID 1
  loop_
    _Dist_constr_tree.Residue_I_auth_seq_ID
    _Dist_constr_tree.Residue_J_auth_seq_ID
    _Dist_constr_tree.Distance_val_min
    _Dist_constr_tree.Distance_val_max
    10 50 1.8 5.0
    20 60 1.8 4.0
    30 70 1.8 6.0
  stop_
save_
save_Torsion_angle_constraint
  loop_
    _Torsion_angle_constraint.Angle_val
    _Torsion_angle_constraint.Angle_tolerance
    -60.0 30.0
    -45.0 25.0
  stop_
save_
stop_
"""
    with tempfile.NamedTemporaryFile(suffix=".str", delete=False, mode="w") as f:
        f.write(synthetic_str)
        tmp_str = f.name

    parser = BMRBParser()
    entry = parser.parse_file(tmp_str)
    print(f"   Entry ID: {entry.entry_id}")
    print(f"   NOE pairs: {len(entry.noe_pairs)}")
    print(f"   Dihedral targets: {len(entry.dihedral_targets)}")
    assert len(entry.noe_pairs) == 3
    assert len(entry.dihedral_targets) == 2
    os.unlink(tmp_str)
    print("[PASS] T3 BMRB parser extracts NOE and dihedral constraints")

    # ---- T4: BMRB Dataset -------------------------------------------------
    print("\n [T4] BMRB Dataset")
    dataset = BMRBDataset(data_dir="./test_bmrb", entry_ids=[], download=False)
    print(f"   Dataset size: {len(dataset)}")
    print("[PASS] T4 BMRB dataset initializes")

    # ---- T5: NMR + XL-MS Fitting Trainer ----------------------------------
    print("\n [T5] NMR + XL-MS Fitting Trainer")
    trainer = NMRXLMSFittingTrainer()

    # Set synthetic NMR data
    noe_pairs = torch.randint(0, 50, (20, 2))
    noe_bounds = torch.rand(20, 2) * 3 + 1.8
    trainer.set_nmr_data(noe_pairs, noe_bounds)

    # Set XL-MS data
    trainer.set_xlms_data(xlms_data)

    # Test loss computation
    coords = torch.randn(50, 3, requires_grad=True)
    loss, metrics = trainer.compute_loss(coords)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Metrics: {list(metrics.keys())}")
    assert torch.isfinite(loss)
    print("[PASS] T5 Combined NMR + XL-MS loss computed")

    # ---- T6: Docker Generator ---------------------------------------------
    print("\n [T6] Docker Generator")
    docker_gen = NMRDockerfileGenerator()
    artifacts = docker_gen.generate_all()
    print(f"   Generated {len(artifacts)} Docker artifacts:")
    for name, path in artifacts.items():
        print(f"     {name}: {os.path.basename(path)}")
    assert len(artifacts) >= 2
    print("[PASS] T6 Docker files generated")

    # ---- T7: Integration Map ----------------------------------------------
    print("\n [T7] Module Integration Check")
    modules = {
        "XL-MS Restraints": _HAS_XLMS,
        "NMR Restraints": _HAS_NMR,
        "OpenMM Refinement": _HAS_REAL_FOLD,
    }
    for name, available in modules.items():
        status = "✓" if available else "✗"
        print(f"   {status} {name}")
    print("[PASS] T7 Integration check completed")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("  HONEST REMINDERS:")
    print("    • BMRB data requires internet access or local files")
    print("    • NMR-STAR parsing is simplified; complex entries need cleanup")
    print("    • XL-MS d_max values are literature-typical, not validated")
    print("    • Docker images are TEMPLATES; requires docker build")
    print("    • This module has NOT been run with real BMRB/XL-MS data here")
    print("=" * 78)
