"""
=============================================================================
SESI MSA INTEGRATION + REAL PDB LOADER + TM-SCORE/GDT-TS VALIDATION
MSAEnhancedFoldingSystem Bridge + Multi-Format DataLoader + Standard Metrics
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
INTEGRATION WITH MSA_ENHANCED_PROTEIN_FOLDING (1).py
=============================================================================
This module calls the following APIs from MSA_ENHANCED_PROTEIN_FOLDING:
  • MSAEnhancedConfig       — configuration (hodge_dim, esm_dim, fused_dim, etc.)
  • MSAFoldingMode          — SINGLE_SEQUENCE | MSA_ENHANCED | ADAPTIVE
  • MSAFeatureExtractor     — conservation, co-evolution, gap patterns
  • QuantumMSACompressor    — weak-measurement MSA compression
  • HodgeMSADecomposition   — harmonic/exact/coexact decomposition
  • DeltaMinMSAFilter       — δ_min column filtering
  • CarpetHierarchicalFusion — 3-level MSA+ESM fusion
  • MSASequenceFuser        — full MSA processing pipeline
  • AdaptiveMSARouter       — automatic MSA/single-sequence selection
  • MSAEnhancedFoldingSystem — top-level system with 3 modes

DATA LOADING:
  • PDB: parse ATOM records → Cα coords + sequence (reuse parse_pdb_file)
  • MSA: parse .a3m / .sto / .fasta → (D, L) token tensor
  • ESM: load pre-computed embeddings from .pt files (or mock for testing)

VALIDATION METRICS:
  • TM-score [Zhang & Skolnick 2004]:
      TM = (1/L_target) * Σ_i 1/(1 + (d_i/d_0)^2)
      d_0 = 1.24 * (L_target - 15)^(1/3) - 1.8
      Range: [0, 1], > 0.5 = same fold, < 0.17 = random
  • GDT-TS [Zemla 2003]:
      GDT-TS = (GDT_1Å + GDT_2Å + GDT_4Å + GDT_8Å) / 4
      GDT_xÅ = % residues within x Å after optimal superposition
  • Kabsch alignment [Kabsch 1976]: SVD-based optimal rotation

HONEST SCOPE:
  • MSA mode requires actual MSA files (.a3m/.sto/.fasta). Without them,
    falls back to SINGLE_SEQUENCE mode automatically.
  • ESM features require pre-computed .pt files. If absent, uses random
    embeddings (clearly labeled as mock).
  • TM-score and GDT-TS are implemented from the published formulas,
    but may differ slightly from TM-align / GDT_TS binaries due to
    implementation details (especially the superposition optimization).
  • This module has NOT been validated against CASP benchmark data.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sesi.msa_pdb_validation")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

MSA_PDB_VAL_VERSION: str = "1.0.0"

# =============================================================================
# Optional ecosystem imports (graceful fallback)
# =============================================================================
_HAS_MSA = False
try:
    from MSA_ENHANCED_PROTEIN_FOLDING import (
        MSAEnhancedConfig,
        MSAFoldingMode,
        MSAEnhancedFoldingSystem,
        MSAFeatureExtractor,
        QuantumMSACompressor,
        HodgeMSADecomposition,
        DeltaMinMSAFilter,
        CarpetHierarchicalFusion,
        MSASequenceFuser,
        AdaptiveMSARouter,
    )
    _HAS_MSA = True
except ImportError:
    import importlib.util
    for cand in ("MSA_ENHANCED_PROTEIN_FOLDING (1).py", "MSA_ENHANCED_PROTEIN_FOLDING.py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("MSA_ENHANCED_PROTEIN_FOLDING", str(p))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                MSAEnhancedConfig = mod.MSAEnhancedConfig
                MSAFoldingMode = mod.MSAFoldingMode
                MSAEnhancedFoldingSystem = mod.MSAEnhancedFoldingSystem
                MSAFeatureExtractor = mod.MSAFeatureExtractor
                QuantumMSACompressor = mod.QuantumMSACompressor
                HodgeMSADecomposition = mod.HodgeMSADecomposition
                DeltaMinMSAFilter = mod.DeltaMinMSAFilter
                CarpetHierarchicalFusion = mod.CarpetHierarchicalFusion
                MSASequenceFuser = mod.MSASequenceFuser
                AdaptiveMSARouter = mod.AdaptiveMSARouter
                _HAS_MSA = True
            break
    if not _HAS_MSA:
        warnings.warn("MSA_ENHANCED_PROTEIN_FOLDING not found. MSA integration disabled.")

# Import PDB parser from previous module if available
_HAS_PDB_PARSER = False
try:
    from sesi_pdb_refine_csoc_training import parse_pdb_file, PDBStructure, AA_3_TO_1
    _HAS_PDB_PARSER = True
except ImportError:
    # Inline fallback parser
    AA_3_TO_1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
        'TRP': 'W', 'TYR': 'Y', 'UNK': 'X',
    }
    _RESOLUTION_RE = re.compile(r"RESOLUTION\.\s*([\d.]+)")

    @dataclass
    class PDBStructure:
        name: str
        sequence: str
        ca_coords: np.ndarray
        resolution: Optional[float]
        n_atoms_total: int
        path: str

    def parse_pdb_file(path) -> PDBStructure:
        """Fallback PDB parser (Cα trace extraction)."""
        path = Path(path)
        coords, resnames = [], []
        seen_res, n_atoms, resolution = set(), 0, None
        model_count = 0
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("MODEL"):
                    model_count += 1
                    if model_count > 1:
                        break
                if resolution is None and line.startswith("REMARK   2"):
                    m = _RESOLUTION_RE.search(line)
                    if m:
                        try:
                            resolution = float(m.group(1))
                        except ValueError:
                            pass
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                n_atoms += 1
                if len(line) < 54:
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
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                seen_res.add(key)
                coords.append([x, y, z])
                resnames.append(res_name)
        sequence = "".join(AA_3_TO_1.get(rn, "X") for rn in resnames)
        arr = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
        return PDBStructure(
            name=path.stem, sequence=sequence, ca_coords=arr,
            resolution=resolution, n_atoms_total=n_atoms, path=str(path),
        )
    _HAS_PDB_PARSER = True  # fallback is available


# =============================================================================
# 1. MSA PARSING (a3m / sto / fasta)
# =============================================================================
def parse_msa_file(path: os.PathLike | str, max_depth: int = 128) -> Optional[List[str]]:
    """
    Parse an MSA file (.a3m, .sto, .fasta) into a list of aligned sequences.

    Supports:
      • FASTA: lines starting with '>' are headers, rest is sequence
      • a3m: same as FASTA but lowercase = insertions (removed)
      • Stockholm (.sto): lines with '#' or '//' are metadata

    Returns list of sequences (strings), truncated to max_depth.
    """
    path = Path(path)
    if not path.exists():
        return None

    sequences: List[str] = []
    current_seq: List[str] = []
    is_a3m = path.suffix.lower() in (".a3m",)
    is_sto = path.suffix.lower() in (".sto", ".stockholm")

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue  # Stockholm metadata
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
                continue
            if is_sto and (line.startswith("#") or line.startswith("//")):
                continue
            # Sequence line: take second column if whitespace-separated
            parts = line.split()
            seq_part = parts[-1] if parts else ""
            if is_a3m:
                # a3m: lowercase = insertions relative to first sequence → remove
                seq_part = re.sub(r"[a-z]", "", seq_part)
            seq_part = seq_part.replace(".", "-").replace(" ", "")
            if seq_part:
                current_seq.append(seq_part)
    if current_seq:
        sequences.append("".join(current_seq))

    if not sequences:
        return None

    # Truncate to max_depth
    sequences = sequences[:max_depth]

    # Pad to same length if needed (some MSA formats have ragged ends)
    max_len = max(len(s) for s in sequences)
    sequences = [s.ljust(max_len, "-") for s in sequences]

    return sequences


def msa_to_tensor(sequences: List[str], vocab_size: int = 23) -> torch.Tensor:
    """
    Convert MSA sequences to integer tensor (D, L).

    Vocab: 20 AA + gap(-) + unknown(X) + mask = 23 tokens.
    """
    aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    aa_to_idx["-"] = 20  # gap
    aa_to_idx["."] = 20
    aa_to_idx["X"] = 21
    aa_to_idx["U"] = 21
    aa_to_idx["O"] = 21
    # mask token = 22

    D = len(sequences)
    L = len(sequences[0]) if D > 0 else 0
    tensor = torch.full((D, L), 21, dtype=torch.long)  # default: unknown

    for i, seq in enumerate(sequences):
        for j, c in enumerate(seq[:L]):
            tensor[i, j] = aa_to_idx.get(c, 21)

    return tensor


# =============================================================================
# 2. MSA ENHANCED INTEGRATION
# =============================================================================
@dataclass
class MSAIntegrationConfig:
    """Configuration for MSA-enhanced integration."""
    # MSA processing
    max_msa_depth: int = 128
    msa_vocab_size: int = 23

    # Feature dimensions (must match MSAEnhancedConfig)
    hodge_dim: int = 64
    esm_dim: int = 128
    fused_dim: int = 64

    # Mode
    mode: str = "adaptive"  # "single_sequence" | "msa_enhanced" | "adaptive"

    # Device
    device: str = "auto"

    def to_msa_enhanced_config(self) -> Any:
        """Convert to MSAEnhancedConfig from MSA_ENHANCED_PROTEIN_FOLDING."""
        if not _HAS_MSA:
            raise ImportError("MSA_ENHANCED_PROTEIN_FOLDING not available.")
        return MSAEnhancedConfig(
            max_msa_depth=self.max_msa_depth,
            msa_vocab_size=self.msa_vocab_size,
            hodge_dim=self.hodge_dim,
            esm_dim=self.esm_dim,
            fused_dim=self.fused_dim,
            device=self.device,
        )

    def to_msa_folding_mode(self) -> Any:
        """Convert mode string to MSAFoldingMode enum."""
        if not _HAS_MSA:
            raise ImportError("MSA_ENHANCED_PROTEIN_FOLDING not available.")
        mode_map = {
            "single_sequence": MSAFoldingMode.SINGLE_SEQUENCE,
            "msa_enhanced": MSAFoldingMode.MSA_ENHANCED,
            "adaptive": MSAFoldingMode.ADAPTIVE,
        }
        return mode_map.get(self.mode, MSAFoldingMode.ADAPTIVE)


class MSAEnhancedIntegration:
    """
    [D] Bridge connecting MSAEnhancedFoldingSystem to the SESI pipeline.

    USAGE:
      1. Load PDB structure (coords + sequence)
      2. Load MSA file (.a3m/.sto/.fasta)
      3. Load or mock ESM features
      4. Call MSAEnhancedFoldingSystem.forward(sequence, esm_features, msa)
      5. Get predicted coords + MSA diagnostics

    MODES (from MSAFoldingMode):
      • SINGLE_SEQUENCE: no MSA, uses ESM-2 only
      • MSA_ENHANCED: uses MSA + ESM-2 with quantum-inspired compression
      • ADAPTIVE: system decides based on sequence length, MSA depth, memory
    """

    def __init__(self, cfg: Optional[MSAIntegrationConfig] = None):
        self.cfg = cfg or MSAIntegrationConfig()

        if not _HAS_MSA:
            raise ImportError(
                "MSA_ENHANCED_PROTEIN_FOLDING (1).py not found. "
                "MSA integration requires this module."
            )

        # Build MSAEnhancedConfig
        msa_cfg = self.cfg.to_msa_enhanced_config()
        mode = self.cfg.to_msa_folding_mode()

        # Initialize the folding system
        self.folding_system = MSAEnhancedFoldingSystem(msa_cfg, mode=mode)

        # Move to device
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        self.folding_system = self.folding_system.to(self.device)

        logger.info(f"[MSAIntegration] Initialized in mode={self.cfg.mode}, "
                    f"hodge_dim={self.cfg.hodge_dim}, esm_dim={self.cfg.esm_dim}")

    def load_esm_features(self, esm_path: Optional[str],
                            sequence: str) -> torch.Tensor:
        """
        Load ESM-2 features from .pt file, or generate mock features.

        Args:
            esm_path: path to .pt file with ESM embeddings (L, esm_dim)
            sequence: amino acid sequence (for length)

        Returns:
            (1, L, esm_dim) tensor
        """
        L = len(sequence)
        if esm_path is not None and os.path.exists(esm_path):
            data = torch.load(esm_path, map_location="cpu")
            if isinstance(data, dict):
                # ESM output format: {'representations': {33: tensor}}
                if "representations" in data:
                    # Take the last layer
                    layers = sorted(data["representations"].keys())
                    esm = data["representations"][layers[-1]]
                else:
                    esm = data.get("embeddings", data.get("features", None))
            else:
                esm = data

            if esm is not None:
                # Remove BOS/EOS tokens if present (ESM adds them)
                if esm.shape[0] > L:
                    esm = esm[1:L+1]
                elif esm.shape[0] < L:
                    # Pad if needed
                    pad = torch.zeros(L - esm.shape[0], esm.shape[1])
                    esm = torch.cat([esm, pad], dim=0)
                logger.info(f"[MSAIntegration] Loaded ESM features from {esm_path}")
                return esm.unsqueeze(0).to(self.device)

        # Mock ESM features (random)
        logger.warning("[MSAIntegration] ESM features not found; using random mock embeddings.")
        return torch.randn(1, L, self.cfg.esm_dim, device=self.device)

    def run(self, sequence: str,
              msa_sequences: Optional[List[str]] = None,
              esm_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run MSA-enhanced folding pipeline.

        Args:
            sequence: amino acid sequence
            msa_sequences: optional list of aligned sequences (from parse_msa_file)
            esm_path: optional path to pre-computed ESM features

        Returns:
            dict with predicted_coords, use_msa, reason, and MSA diagnostics
        """
        L = len(sequence)

        # Load ESM features
        esm_features = self.load_esm_features(esm_path, sequence)

        # Prepare MSA tensor
        msa_tensor = None
        if msa_sequences is not None and len(msa_sequences) > 0:
            msa_tensor = msa_to_tensor(msa_sequences, self.cfg.msa_vocab_size)
            msa_tensor = msa_tensor.unsqueeze(0).to(self.device)  # (1, D, L)
            logger.info(f"[MSAIntegration] MSA loaded: {msa_tensor.shape[1]} sequences, "
                        f"{msa_tensor.shape[2]} positions")

        # Run folding system
        output = self.folding_system(sequence, esm_features, msa=msa_tensor)

        # Build result dict
        result = {
            "use_msa": bool(output.get("use_msa", False)),
            "msa_decision_reason": output.get("msa_decision_reason", ""),
            "predicted_coords": output.get("predicted_coords"),
            "sequence_length": L,
        }

        # Add MSA-specific diagnostics if available
        if result["use_msa"]:
            result["compression_ratio"] = float(output.get("msa_compress_compression_ratio", 0.0))
            result["final_confidence"] = float(output.get("msa_compress_final_confidence", torch.tensor(0.0)).mean().item())
            result["hodge_snr"] = float(output.get("msa_hodge_signal_to_noise_ratio", torch.tensor(0.0)).mean().item())
            result["columns_kept"] = int(output.get("msa_filter_n_kept", torch.tensor(0)).sum().item())
            result["consistency_loss"] = float(output.get("msa_fusion_consistency_loss", torch.tensor(0.0)).item())

        return result


# =============================================================================
# 3. REAL PDB + MSA DATA LOADER
# =============================================================================
@dataclass
class RealPDBMSAEntry:
    """One PDB entry with optional MSA and ESM features."""
    name: str
    sequence: str
    ca_coords: np.ndarray          # (L, 3)
    resolution: Optional[float]
    msa_sequences: Optional[List[str]]
    esm_features: Optional[torch.Tensor]  # (L, esm_dim) or None
    pdb_path: str
    msa_path: Optional[str]
    esm_path: Optional[str]


class RealPDBMSADataset(Dataset):
    """
    Dataset loading real PDB files + optional MSA + optional ESM features.

    Directory structure expected:
      root/
        1abc.pdb          # PDB file (required)
        1abc.a3m          # MSA file (optional, .a3m/.sto/.fasta)
        1abc_esm.pt       # ESM features (optional)
        2xyz.pdb
        ...

    Splits are deterministic by MD5 hash of filename (same as previous module).
    """

    def __init__(self, root: os.PathLike | str, split: str = "train",
                 train_pct: int = 80, val_pct: int = 10,
                 min_residues: int = 20, max_residues: Optional[int] = None,
                 max_msa_depth: int = 128,
                 esm_dim: int = 128):
        self.root = Path(root)
        self.split = split
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.min_residues = min_residues
        self.max_residues = max_residues
        self.max_msa_depth = max_msa_depth
        self.esm_dim = esm_dim
        self._cache: Dict[int, RealPDBMSAEntry] = {}

        # Find all PDB files
        all_files = sorted(list(self.root.glob("*.pdb")) + list(self.root.glob("*.ent")))
        self.files = [f for f in all_files if self._fold_of(f) == split]

        logger.info(f"[RealPDBMSADataset] root={self.root}, split={split}, "
                    f"{len(self.files)}/{len(all_files)} files")

    def _fold_of(self, path: Path) -> str:
        h = int(hashlib.md5(path.name.encode("utf-8")).hexdigest(), 16) % 100
        if h < self.train_pct:
            return "train"
        if h < self.train_pct + self.val_pct:
            return "val"
        return "test"

    def _find_msa(self, stem: str) -> Optional[Path]:
        """Find MSA file matching the PDB stem."""
        for ext in (".a3m", ".sto", ".fasta", ".fa"):
            p = self.root / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def _find_esm(self, stem: str) -> Optional[Path]:
        """Find ESM feature file matching the PDB stem."""
        for suffix in ("_esm.pt", "_esm2.pt", "_embeddings.pt"):
            p = self.root / f"{stem}{suffix}"
            if p.exists():
                return p
        return None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx in self._cache:
            entry = self._cache[idx]
        else:
            pdb_path = self.files[idx]
            struct = parse_pdb_file(pdb_path)

            # Filter by length
            L = struct.ca_coords.shape[0]
            if L < self.min_residues:
                # Return first valid entry (deterministic fallback)
                for j in range(len(self.files)):
                    entry = self.__getitem__(j)
                    if entry["length"] >= self.min_residues:
                        return entry
                raise RuntimeError("No valid PDB entry found")

            if self.max_residues is not None and L > self.max_residues:
                struct = PDBStructure(
                    name=struct.name,
                    sequence=struct.sequence[:self.max_residues],
                    ca_coords=struct.ca_coords[:self.max_residues],
                    resolution=struct.resolution,
                    n_atoms_total=struct.n_atoms_total,
                    path=struct.path,
                )

            # Load MSA
            msa_path = self._find_msa(pdb_path.stem)
            msa_sequences = None
            if msa_path is not None:
                msa_sequences = parse_msa_file(msa_path, max_depth=self.max_msa_depth)

            # Load ESM
            esm_path = self._find_esm(pdb_path.stem)
            esm_features = None
            if esm_path is not None:
                try:
                    data = torch.load(esm_path, map_location="cpu")
                    if isinstance(data, dict):
                        if "representations" in data:
                            layers = sorted(data["representations"].keys())
                            esm_features = data["representations"][layers[-1]]
                        else:
                            esm_features = data.get("embeddings", data.get("features"))
                    else:
                        esm_features = data
                    if esm_features is not None and esm_features.shape[0] > len(struct.sequence):
                        esm_features = esm_features[1:len(struct.sequence)+1]
                except Exception as e:
                    logger.warning(f"[RealPDBMSADataset] Failed to load ESM from {esm_path}: {e}")

            entry = RealPDBMSAEntry(
                name=struct.name,
                sequence=struct.sequence,
                ca_coords=struct.ca_coords,
                resolution=struct.resolution,
                msa_sequences=msa_sequences,
                esm_features=esm_features,
                pdb_path=str(pdb_path),
                msa_path=str(msa_path) if msa_path else None,
                esm_path=str(esm_path) if esm_path else None,
            )
            self._cache[idx] = entry

        # Convert to tensors
        coords = torch.from_numpy(entry.ca_coords).float()
        L = coords.shape[0]

        result = {
            "name": entry.name,
            "sequence": entry.sequence,
            "coords": coords,
            "mask": torch.ones(L, dtype=torch.bool),
            "length": L,
            "resolution": entry.resolution if entry.resolution is not None else -1.0,
            "has_msa": entry.msa_sequences is not None,
            "has_esm": entry.esm_features is not None,
        }

        # MSA tensor
        if entry.msa_sequences is not None:
            msa_tensor = msa_to_tensor(entry.msa_sequences)
            result["msa"] = msa_tensor  # (D, L)
        else:
            result["msa"] = None

        # ESM tensor
        if entry.esm_features is not None:
            result["esm"] = entry.esm_features  # (L, esm_dim)
        else:
            result["esm"] = None

        return result


def real_pdb_msa_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length PDB entries with padding."""
    Lmax = max(item["length"] for item in batch)
    B = len(batch)

    coords = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    sequences = []
    names = []
    resolutions = []
    has_msa = []
    has_esm = []

    # Collect MSA and ESM (variable depth/length)
    msa_list = []
    esm_list = []

    for i, item in enumerate(batch):
        L = item["length"]
        coords[i, :L] = item["coords"]
        mask[i, :L] = True
        sequences.append(item["sequence"])
        names.append(item["name"])
        resolutions.append(item["resolution"])
        has_msa.append(item["has_msa"])
        has_esm.append(item["has_esm"])
        msa_list.append(item["msa"])
        esm_list.append(item["esm"])

    # Stack MSA if all have same depth, otherwise keep as list
    msa_batch = None
    if all(m is not None for m in msa_list):
        Dmax = max(m.shape[0] for m in msa_list)
        # Pad MSA to same depth
        msa_batch = torch.zeros(B, Dmax, Lmax, dtype=torch.long)
        msa_mask = torch.zeros(B, Dmax, dtype=torch.bool)
        for i, m in enumerate(msa_list):
            D, L = m.shape
            msa_batch[i, :D, :L] = m
            msa_mask[i, :D] = True
        result_msa = (msa_batch, msa_mask)
    else:
        result_msa = None

    # Stack ESM if available
    esm_batch = None
    if all(e is not None for e in esm_list):
        Emax = max(e.shape[1] for e in esm_list)
        esm_batch = torch.zeros(B, Lmax, Emax)
        for i, e in enumerate(esm_list):
            L, E = e.shape
            esm_batch[i, :L, :E] = e
    else:
        esm_batch = None

    return {
        "coords": coords,
        "mask": mask,
        "sequences": sequences,
        "names": names,
        "resolutions": torch.tensor(resolutions),
        "has_msa": has_msa,
        "has_esm": has_esm,
        "msa": result_msa,       # (tensor, mask) or None
        "esm": esm_batch,        # (B, L, E) or None
        "lengths": torch.tensor([item["length"] for item in batch]),
    }


def build_real_pdb_msa_dataloaders(root: os.PathLike | str, batch_size: int = 4,
                                     num_workers: int = 0, **kwargs) -> Dict[str, DataLoader]:
    """Build train/val/test dataloaders for real PDB + MSA data."""
    out = {}
    for split in ("train", "val", "test"):
        ds = RealPDBMSADataset(root, split=split, **kwargs)
        if len(ds) == 0:
            out[split] = None
            continue
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                 num_workers=num_workers, collate_fn=real_pdb_msa_collate)
    return out


# =============================================================================
# 4. STRUCTURAL VALIDATION METRICS (TM-score, GDT-TS)
# =============================================================================
class StructuralValidationMetrics:
    """
    [C] Standard structural validation metrics.

    TM-SCORE [Zhang & Skolnick 2004]:
      TM = (1/L_target) * Σ_i 1/(1 + (d_i/d_0)^2)
      d_0 = 1.24 * (L_target - 15)^(1/3) - 1.8
      Range: [0, 1]
      Interpretation: > 0.5 = same fold, < 0.17 = random

    GDT-TS [Zemla 2003]:
      GDT-TS = (GDT_1Å + GDT_2Å + GDT_4Å + GDT_8Å) / 4
      GDT_xÅ = % residues within x Å after optimal superposition
      Range: [0, 100]
      Standard CASP metric

    KABSCH ALIGNMENT [Kabsch 1976]:
      Optimal rotation via SVD to minimize RMSD.
    """

    @staticmethod
    def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kabsch alignment: find optimal rotation R and translation t
        to superpose P onto Q (minimizing RMSD).

        Args:
            P: (N, 3) coordinates to be rotated/translated
            Q: (N, 3) reference coordinates

        Returns:
            P_aligned, R, t
        """
        assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"

        # Center both
        P_center = P.mean(axis=0)
        Q_center = Q.mean(axis=0)
        P_centered = P - P_center
        Q_centered = Q - Q_center

        # Covariance matrix
        H = P_centered.T @ Q_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Correct for reflection
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.eye(3)
        sign_matrix[2, 2] = np.sign(d)

        # Optimal rotation
        R = Vt.T @ sign_matrix @ U.T

        # Optimal translation
        t = Q_center - R @ P_center

        # Apply
        P_aligned = (R @ P.T).T + t

        return P_aligned, R, t

    @staticmethod
    def compute_rmsd(P: np.ndarray, Q: np.ndarray, aligned: bool = False) -> float:
        """Compute RMSD between two coordinate sets."""
        if not aligned:
            P, _, _ = StructuralValidationMetrics.kabsch_align(P, Q)
        diff = P - Q
        return float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    @staticmethod
    def compute_tm_score(P: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
        """
        Compute TM-score between predicted (P) and native (Q) structures.

        Args:
            P: (N, 3) predicted coordinates
            Q: (N, 3) native/reference coordinates

        Returns:
            dict with tm_score, rmsd, n_aligned, d0
        """
        assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"
        L_target = Q.shape[0]

        if L_target < 15:
            d0 = 0.5
        else:
            d0 = 1.24 * (L_target - 15) ** (1.0 / 3.0) - 1.8

        # Kabsch alignment
        P_aligned, _, _ = StructuralValidationMetrics.kabsch_align(P, Q)

        # Per-residue distances
        distances = np.linalg.norm(P_aligned - Q, axis=1)

        # TM-score formula
        tm_per_residue = 1.0 / (1.0 + (distances / d0) ** 2)
        tm_score = float(tm_per_residue.sum() / L_target)

        # RMSD
        rmsd = float(np.sqrt((distances ** 2).mean()))

        return {
            "tm_score": tm_score,
            "rmsd": rmsd,
            "n_aligned": L_target,
            "d0": d0,
            "tm_interpretation": StructuralValidationMetrics._interpret_tm(tm_score),
        }

    @staticmethod
    def _interpret_tm(tm_score: float) -> str:
        """Interpret TM-score value."""
        if tm_score > 0.5:
            return "same fold (likely correct topology)"
        elif tm_score > 0.3:
            return "similar fold (possible homology)"
        elif tm_score > 0.17:
            return "weak similarity (uncertain)"
        else:
            return "random (no structural similarity)"

    @staticmethod
    def compute_gdt_ts(P: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
        """
        Compute GDT-TS between predicted (P) and native (Q) structures.

        GDT-TS = (GDT_1Å + GDT_2Å + GDT_4Å + GDT_8Å) / 4
        where GDT_xÅ = % residues within x Å after superposition.

        Note: The official GDT-TS uses an iterative superposition that
        maximizes the number of residues under each threshold. This
        implementation uses a single Kabsch alignment for simplicity.

        Args:
            P: (N, 3) predicted coordinates
            Q: (N, 3) native/reference coordinates

        Returns:
            dict with gdt_ts, gdt_1, gdt_2, gdt_4, gdt_8, rmsd
        """
        assert P.shape == Q.shape, f"Shape mismatch: {P.shape} vs {Q.shape}"
        N = P.shape[0]

        # Kabsch alignment
        P_aligned, _, _ = StructuralValidationMetrics.kabsch_align(P, Q)

        # Per-residue distances
        distances = np.linalg.norm(P_aligned - Q, axis=1)

        # GDT at each threshold
        thresholds = [1.0, 2.0, 4.0, 8.0]
        gdt_scores = []
        for t in thresholds:
            n_within = (distances <= t).sum()
            gdt = 100.0 * n_within / N
            gdt_scores.append(gdt)

        gdt_ts = float(np.mean(gdt_scores))
        rmsd = float(np.sqrt((distances ** 2).mean()))

        return {
            "gdt_ts": gdt_ts,
            "gdt_1a": gdt_scores[0],
            "gdt_2a": gdt_scores[1],
            "gdt_4a": gdt_scores[2],
            "gdt_8a": gdt_scores[3],
            "rmsd": rmsd,
            "n_residues": N,
            "gdt_interpretation": StructuralValidationMetrics._interpret_gdt(gdt_ts),
        }

    @staticmethod
    def _interpret_gdt(gdt_ts: float) -> str:
        """Interpret GDT-TS value."""
        if gdt_ts > 90:
            return "excellent (near-experimental accuracy)"
        elif gdt_ts > 70:
            return "good (correct topology, minor errors)"
        elif gdt_ts > 50:
            return "fair (roughly correct fold)"
        elif gdt_ts > 30:
            return "poor (partial correctness)"
        else:
            return "very poor (likely incorrect)"

    @staticmethod
    def compute_all_metrics(P: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
        """
        Compute all validation metrics.

        Args:
            P: (N, 3) predicted coordinates
            Q: (N, 3) native/reference coordinates

        Returns:
            dict with tm_score, gdt_ts, rmsd, and interpretations
        """
        tm_result = StructuralValidationMetrics.compute_tm_score(P, Q)
        gdt_result = StructuralValidationMetrics.compute_gdt_ts(P, Q)

        return {
            "tm_score": tm_result["tm_score"],
            "tm_interpretation": tm_result["tm_interpretation"],
            "gdt_ts": gdt_result["gdt_ts"],
            "gdt_interpretation": gdt_result["gdt_interpretation"],
            "rmsd": tm_result["rmsd"],
            "n_residues": tm_result["n_aligned"],
            "d0": tm_result["d0"],
            "gdt_details": gdt_result,
        }


# =============================================================================
# 5. SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI MSA + PDB + VALIDATION SUITE v{MSA_PDB_VAL_VERSION} — Self-Test")
    print(f"  MSA_ENHANCED={_HAS_MSA}  PDB_PARSER={_HAS_PDB_PARSER}")
    print("=" * 78)

    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="sesi_msa_pdb_"))

    # ---- T1: PDB parser -------------------------------------------------
    print("\n [T1] PDB Parser")
    # Create a test PDB file (idealized helix)
    test_pdb = tmpdir / "test_helix.pdb"
    with open(test_pdb, "w") as f:
        f.write("REMARK   2 RESOLUTION.    2.00 ANGSTROMS.\n")
        for i in range(20):
            ang = math.radians(100 * i)
            x = 2.3 * math.cos(ang)
            y = 2.3 * math.sin(ang)
            z = 1.5 * i
            f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("END\n")

    struct = parse_pdb_file(test_pdb)
    assert struct.ca_coords.shape == (20, 3)
    assert struct.sequence == "A" * 20
    assert struct.resolution == 2.0
    print(f"   Parsed: {struct.name}, L={len(struct.sequence)}, resolution={struct.resolution}")
    print("[PASS] T1 PDB parser extracts coords/sequence/resolution")

    # ---- T2: MSA parser -------------------------------------------------
    print("\n [T2] MSA Parser")
    # Create a test MSA file (FASTA format)
    test_msa = tmpdir / "test_msa.fasta"
    with open(test_msa, "w") as f:
        f.write(">seq1\n")
        f.write("ACDEFGHIKLMNPQRSTVWY\n")
        f.write(">seq2\n")
        f.write("ACDEFGHIKLMNPQRSTVW-\n")
        f.write(">seq3\n")
        f.write("ACDEFGHIKLMNPQRSTVWY\n")

    msa_seqs = parse_msa_file(test_msa, max_depth=128)
    assert msa_seqs is not None
    assert len(msa_seqs) == 3
    assert all(len(s) == 20 for s in msa_seqs)
    print(f"   MSA parsed: {len(msa_seqs)} sequences, L={len(msa_seqs[0])}")

    # Convert to tensor
    msa_tensor = msa_to_tensor(msa_seqs)
    assert msa_tensor.shape == (3, 20)
    assert msa_tensor.dtype == torch.long
    print(f"   MSA tensor: {msa_tensor.shape}")
    print("[PASS] T2 MSA parser works for FASTA format")

    # ---- T3: Kabsch alignment -------------------------------------------
    print("\n [T3] Kabsch Alignment")
    # Create two identical helices, one rotated
    P = np.array([[2.3 * math.cos(math.radians(100 * i)),
                    2.3 * math.sin(math.radians(100 * i)),
                    1.5 * i] for i in range(20)], dtype=np.float64)
    # Apply a known rotation
    theta = math.radians(30)
    R_true = np.array([[math.cos(theta), -math.sin(theta), 0],
                        [math.sin(theta), math.cos(theta), 0],
                        [0, 0, 1]])
    Q = (R_true @ P.T).T + np.array([5, 3, 2])

    P_aligned, R_est, t_est = StructuralValidationMetrics.kabsch_align(P, Q)
    rmsd_after = np.sqrt(((P_aligned - Q) ** 2).sum(axis=1).mean())
    print(f"   RMSD after alignment: {rmsd_after:.6f} (should be ~0)")
    assert rmsd_after < 1e-6, f"Kabsch alignment failed: RMSD={rmsd_after}"
    print("[PASS] T3 Kabsch alignment recovers known rotation")

    # ---- T4: TM-score ---------------------------------------------------
    print("\n [T4] TM-score Computation")
    # Perfect match → TM-score should be 1.0
    tm_result_perfect = StructuralValidationMetrics.compute_tm_score(P, P)
    print(f"   TM-score (perfect): {tm_result_perfect['tm_score']:.4f}")
    assert abs(tm_result_perfect["tm_score"] - 1.0) < 1e-6

    # Add noise → TM-score should decrease
    P_noisy = P + np.random.randn(20, 3) * 2.0
    tm_result_noisy = StructuralValidationMetrics.compute_tm_score(P_noisy, P)
    print(f"   TM-score (noisy): {tm_result_noisy['tm_score']:.4f}")
    assert tm_result_noisy["tm_score"] < 1.0
    assert tm_result_noisy["tm_score"] > 0.0
    print(f"   Interpretation: {tm_result_noisy['tm_interpretation']}")
    print("[PASS] T4 TM-score computed correctly")

    # ---- T5: GDT-TS -----------------------------------------------------
    print("\n [T5] GDT-TS Computation")
    # Perfect match → GDT-TS should be 100
    gdt_perfect = StructuralValidationMetrics.compute_gdt_ts(P, P)
    print(f"   GDT-TS (perfect): {gdt_perfect['gdt_ts']:.2f}")
    assert abs(gdt_perfect["gdt_ts"] - 100.0) < 1e-6

    # Noisy → GDT-TS should decrease
    gdt_noisy = StructuralValidationMetrics.compute_gdt_ts(P_noisy, P)
    print(f"   GDT-TS (noisy): {gdt_noisy['gdt_ts']:.2f}")
    assert gdt_noisy["gdt_ts"] < 100.0
    assert gdt_noisy["gdt_ts"] > 0.0
    print(f"   GDT details: 1Å={gdt_noisy['gdt_1a']:.1f}%, 2Å={gdt_noisy['gdt_2a']:.1f}%, "
          f"4Å={gdt_noisy['gdt_4a']:.1f}%, 8Å={gdt_noisy['gdt_8a']:.1f}%")
    print(f"   Interpretation: {gdt_noisy['gdt_interpretation']}")
    print("[PASS] T5 GDT-TS computed correctly")

    # ---- T6: All metrics together ---------------------------------------
    print("\n [T6] Combined Metrics")
    all_metrics = StructuralValidationMetrics.compute_all_metrics(P_noisy, P)
    print(f"   TM-score: {all_metrics['tm_score']:.4f} ({all_metrics['tm_interpretation']})")
    print(f"   GDT-TS: {all_metrics['gdt_ts']:.2f} ({all_metrics['gdt_interpretation']})")
    print(f"   RMSD: {all_metrics['rmsd']:.3f} Å")
    print(f"   N residues: {all_metrics['n_residues']}")
    assert all_metrics["tm_score"] == tm_result_noisy["tm_score"]
    assert all_metrics["gdt_ts"] == gdt_noisy["gdt_ts"]
    print("[PASS] T6 Combined metrics consistent")

    # ---- T7: RealPDBMSADataset ------------------------------------------
    print("\n [T7] RealPDBMSADataset")
    # Create multiple test PDB files
    for i, n_res in enumerate([30, 40, 50, 60, 70]):
        pdb_path = tmpdir / f"prot_{i:02d}_{n_res}.pdb"
        with open(pdb_path, "w") as f:
            f.write(f"REMARK   2 RESOLUTION.    {2.0 + i * 0.5:.2f} ANGSTROMS.\n")
            for j in range(n_res):
                ang = math.radians(100 * j)
                x = 2.3 * math.cos(ang)
                y = 2.3 * math.sin(ang)
                z = 1.5 * j
                f.write(f"ATOM  {j+1:5d}  CA  ALA A{j+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("END\n")

    ds_train = RealPDBMSADataset(tmpdir, split="train", min_residues=25)
    ds_val = RealPDBMSADataset(tmpdir, split="val", min_residues=25)
    ds_test = RealPDBMSADataset(tmpdir, split="test", min_residues=25)
    print(f"   Split sizes: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

    if len(ds_train) > 0:
        sample = ds_train[0]
        print(f"   Sample: {sample['name']}, L={sample['length']}, "
              f"has_msa={sample['has_msa']}, has_esm={sample['has_esm']}")
        assert sample["coords"].shape == (sample["length"], 3)
        assert sample["mask"].shape == (sample["length"],)
        print("[PASS] T7 Dataset loads PDB files correctly")
    else:
        print("[SKIP] T7 No files in train split (all went to val/test by hash)")

    # ---- T8: Collate function -------------------------------------------
    print("\n [T8] Collate Function")
    if len(ds_train) >= 2:
        batch = [ds_train[i] for i in range(min(2, len(ds_train)))]
        collated = real_pdb_msa_collate(batch)
        print(f"   Batch coords: {tuple(collated['coords'].shape)}")
        print(f"   Batch mask: {tuple(collated['mask'].shape)}")
        print(f"   Sequences: {len(collated['sequences'])}")
        print(f"   MSA present: {collated['msa'] is not None}")
        print(f"   ESM present: {collated['esm'] is not None}")
        assert collated["coords"].shape[0] == len(batch)
        assert collated["mask"].shape[0] == len(batch)
        print("[PASS] T8 Collate pads variable-length structures")
    else:
        print("[SKIP] T8 Not enough samples in train split")

    # ---- T9: MSA Integration (if available) -----------------------------
    print("\n [T9] MSA Integration")
    if _HAS_MSA:
        msa_cfg = MSAIntegrationConfig(
            hodge_dim=64,
            esm_dim=128,
            fused_dim=64,
            mode="adaptive",
        )
        msa_integration = MSAEnhancedIntegration(msa_cfg)
        print(f"   MSA integration initialized in mode={msa_cfg.mode}")
        print(f"   Device: {msa_integration.device}")

        # Test with mock data
        test_seq = "ACDEFGHIKLMNPQRSTVWY"
        test_msa_seqs = [test_seq, test_seq[:-1] + "-", test_seq]
        result = msa_integration.run(test_seq, msa_sequences=test_msa_seqs)
        print(f"   use_msa: {result['use_msa']}")
        print(f"   reason: {result['msa_decision_reason']}")
        print(f"   predicted_coords shape: {result['predicted_coords'].shape}")
        print("[PASS] T9 MSA integration runs")
    else:
        print("[SKIP] T9 MSA_ENHANCED_PROTEIN_FOLDING not available")

    # ---- Cleanup ---------------------------------------------------------
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("  HONEST REMINDERS:")
    print("    • MSA mode requires actual MSA files (.a3m/.sto/.fasta).")
    print("    • ESM features require pre-computed .pt files; mock is random.")
    print("    • TM-score/GDT-TS are pure Python implementations, may differ")
    print("      slightly from TM-align / GDT_TS binaries.")
    print("    • This module has NOT been validated against CASP benchmark data.")
    print("    • MSA_ENHANCED_PROTEIN_FOLDING (1).py must be importable.")
    print("=" * 78)
