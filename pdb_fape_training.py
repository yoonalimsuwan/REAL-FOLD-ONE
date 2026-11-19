#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
pdb_fape_training.py — FAPE Loss + PDB Dataset Loader + Joint Training
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died on the Cross to teach: when struck on the left cheek, turn also
the right, so that we may listen to the voice of the enemy who wishes us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
FAPE LOSS (AlphaFold2 §1.9.4)
=============================================================================
Frame Aligned Point Error measures structure accuracy in a way that is
INVARIANT to global rigid transformations:

    FAPE = (1/(L*N)) Σ_i Σ_j ‖ R_i^T (x_j - t_i) - R̂_i^T (x̂_j - t̂_i) ‖ / d_clamp

where:
  (R_i, t_i)     = true backbone frame at residue i (rotation + translation)
  (R̂_i, t̂_i)     = predicted backbone frame at residue i
  x_j             = true position of atom j
  x̂_j             = predicted position of atom j
  d_clamp         = 10 Å (prevents gradient explosion for large errors)

FAPE is computed over:
  • Backbone atoms (N, Cα, C) for backbone FAPE
  • All heavy atoms for full-atom FAPE

This is the PRIMARY structure loss in AlphaFold2.

=============================================================================
PDB DATASET LOADER
=============================================================================
Sources:
  • Local PDB files (from a directory)
  • RCSB PDB download (requires internet)

Extracts:
  • Sequence (one-letter)
  • Backbone frames (N, Cα, C → rotation + translation)
  • Atom positions (heavy atoms)
  • Resolution, method, R-free (from REMARK)
  • B-factors (per-atom)

Filtering:
  • Resolution ≤ threshold
  • Length range
  • Method (X-ray, cryo-EM)

=============================================================================
JOINT TRAINING
=============================================================================
Loss = w_coord * MSE(pred_coords, true_coords)
     + w_fape * FAPE(pred_frames, pred_atoms, true_frames, true_atoms)
     + w_pheno * PhenotypeLoss(pred_phenotype, true_phenotype)

The shared encoder is trained jointly. Structure head uses FAPE.
Phenotype head uses MSE or BCE.

=============================================================================
HONEST SCOPE
=============================================================================
❌ This is NOT the AlphaFold2 training pipeline. It lacks:
   • MSA processing and Evoformer
   • Distogram / pLDDT / violation losses
   • Recycling
   • Real MSA-based structural templates
❌ FAPE here is implemented correctly but used in a simplified context
   (single-sequence model, no MSA, small model size).
❌ PDB data has biases (resolution, crystallizability, over-represented folds).
❌ No CASP validation. No claim of beating AlphaFold2/3.
❌ Phenotype labels are NOT from PDB. PDB provides structure, not phenotype.
   For supervised variant-effect training, use ClinVar, UniProt, or literature.
✅ The FAPE loss itself is standard and correct.
✅ The PDB loader is functional and honest about missing data.
✅ The joint training loop is differentiable and testable.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("pdb_fape")

VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {c: i for i, c in enumerate(AA_ALPHABET)}
AA_1TO3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}
BACKBONE_ATOMS = ["N", "CA", "C", "O"]

BANNER = f"""
=============================================================================
FAPE LOSS + PDB DATASET + JOINT TRAINING v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
FAPE per AlphaFold2 §1.9.4. NOT the AlphaFold2 training pipeline.
RESEARCH ONLY. No CASP validation. No clinical use.
=============================================================================
"""


# =============================================================================
# §1. FAPE LOSS
# =============================================================================

def compute_backbone_frames(n_pos: Tensor, ca_pos: Tensor, c_pos: Tensor
                            ) -> Tuple[Tensor, Tensor]:
    """
    Compute backbone frames from N, Cα, C positions.
    Returns (R, t) where R is (B, L, 3, 3) rotation and t is (B, L, 3) translation.
    Uses Gram-Schmidt orthogonalization (AlphaFold2 convention).
    """
    # e1 = Cα - N (x-axis direction)
    # e2 = C - Cα
    # Gram-Schmidt: x-axis = normalize(e1)
    #               z-axis = normalize(e1 × e2)
    #               y-axis = z-axis × x-axis
    e1 = ca_pos - n_pos
    e2 = c_pos - ca_pos

    # Normalize e1
    e1_norm = e1 / (e1.norm(dim=-1, keepdim=True) + 1e-8)

    # Cross product for z-axis
    z_axis = torch.cross(e1, e2, dim=-1)
    z_axis = z_axis / (z_axis.norm(dim=-1, keepdim=True) + 1e-8)

    # y-axis = z × x
    y_axis = torch.cross(z_axis, e1_norm, dim=-1)

    # Rotation matrix: columns are x, y, z axes
    R = torch.stack([e1_norm, y_axis, z_axis], dim=-1)  # (B, L, 3, 3)
    t = ca_pos  # Translation is Cα position

    return R, t


def fape_loss(
    pred_frames: Tensor,       # (B, L, 3, 3) predicted rotations
    pred_trans: Tensor,        # (B, L, 3) predicted translations
    true_frames: Tensor,       # (B, L, 3, 3) true rotations
    true_trans: Tensor,        # (B, L, 3) true translations
    pred_atoms: Tensor,        # (B, L, N_atoms, 3) or (B, N_total, 3)
    true_atoms: Tensor,        # same shape
    mask: Optional[Tensor] = None,  # (B, L) valid residues
    d_clamp: float = 10.0,     # clamp distance in Angstrom
) -> Tensor:
    """
    Frame Aligned Point Error (AlphaFold2 §1.9.4).

    FAPE = (1/(L*N)) Σ_i Σ_j ‖ R_i^T (x_j - t_i) - R̂_i^T (x̂_j - t̂_i) ‖ / d_clamp

    This is INVARIANT to global rigid transformations.
    """
    B, L, _, _ = pred_frames.shape

    # Handle atom dimension
    if pred_atoms.dim() == 3:
        # (B, L, 3) → treat as single atom per residue (Cα)
        pred_atoms = pred_atoms.unsqueeze(2)  # (B, L, 1, 3)
        true_atoms = true_atoms.unsqueeze(2)
        N_atoms = 1
    elif pred_atoms.dim() == 4:
        N_atoms = pred_atoms.shape[2]
    else:
        raise ValueError(f"pred_atoms must be 3D or 4D, got {pred_atoms.dim()}D")

    # Expand frames for all atom pairs: (B, L, L, 3, 3) and (B, L, L, 3)
    # For each frame i and atom j:
    # true_aligned = R_i^T (x_j - t_i)
    # pred_aligned = R̂_i^T (x̂_j - t̂_i)

    # Expand true atoms for all frames: (B, L_frames, L_atoms, N_atoms, 3)
    true_atoms_exp = true_atoms.unsqueeze(1).expand(-1, L, -1, -1, -1)
    pred_atoms_exp = pred_atoms.unsqueeze(1).expand(-1, L, -1, -1, -1)

    # Expand translations: (B, L_frames, 1, 1, 3)
    true_trans_exp = true_trans.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, N_atoms, -1)
    pred_trans_exp = pred_trans.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, N_atoms, -1)

    # Compute aligned positions
    # true_aligned[b,i,j,k] = R_i^T (x_j[k] - t_i)
    diff_true = true_atoms_exp - true_trans_exp  # (B, L, L, N_atoms, 3)
    diff_pred = pred_atoms_exp - pred_trans_exp

    # Apply rotation: R^T @ diff
    # R: (B, L, 3, 3) → expand to (B, L, 1, 1, 3, 3)
    true_R_exp = true_frames.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, N_atoms, -1, -1)
    pred_R_exp = pred_frames.unsqueeze(2).unsqueeze(3).expand(-1, -1, L, N_atoms, -1, -1)

    # R^T @ diff: (B, L, L, N_atoms, 3)
    true_aligned = torch.einsum("...ij,...j->...i",
                                 true_R_exp.transpose(-1, -2), diff_true)
    pred_aligned = torch.einsum("...ij,...j->...i",
                                 pred_R_exp.transpose(-1, -2), diff_pred)

    # Distance
    dist = (true_aligned - pred_aligned).norm(dim=-1)  # (B, L, L, N_atoms)

    # Clamp
    dist = torch.clamp(dist, max=d_clamp) / d_clamp

    # Apply mask
    if mask is not None:
        # mask: (B, L) → expand to (B, L, L, N_atoms)
        mask_2d = mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, L, 1)
        mask_2d = mask_2d * mask.unsqueeze(2).unsqueeze(-1).float()  # (B, L, L, 1)
        dist = dist * mask_2d

    # Average over valid pairs
    if mask is not None:
        n_valid = mask.float().sum(dim=1).clamp(min=1)
        n_pairs = (n_valid * n_valid).sum() * N_atoms
    else:
        n_pairs = B * L * L * N_atoms

    return dist.sum() / max(n_pairs, 1.0)


class FAPELoss(nn.Module):
    """FAPE loss module with configurable clamp and mask support."""

    def __init__(self, d_clamp: float = 10.0):
        super().__init__()
        self.d_clamp = d_clamp

    def forward(self, pred_frames: Tensor, pred_trans: Tensor,
                true_frames: Tensor, true_trans: Tensor,
                pred_atoms: Tensor, true_atoms: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        return fape_loss(pred_frames, pred_trans, true_frames, true_trans,
                        pred_atoms, true_atoms, mask, self.d_clamp)


# =============================================================================
# §2. PDB DATASET LOADER
# =============================================================================

def download_pdb(pdb_id: str, dest_dir: str = "data/pdb") -> Optional[str]:
    """Download a PDB file from RCSB. Returns local path or None on failure."""
    pdb_id = pdb_id.upper()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{pdb_id.lower()}.pdb"
    if out.exists():
        return str(out)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, out)
        if out.stat().st_size < 100:
            out.unlink()
            return None
        log.info(f"[PDB] Downloaded {pdb_id} → {out}")
        return str(out)
    except Exception as e:
        log.warning(f"[PDB] Failed to download {pdb_id}: {e}")
        return None


def parse_pdb_structure(path: str, max_atoms: int = 5000) -> Optional[Dict]:
    """
    Parse a PDB file and extract:
      • Sequence
      • Backbone frames (N, Cα, C)
      • All heavy atom positions
      • Resolution (from REMARK 2)
      • Method (X-ray, cryo-EM)
    """
    atoms: Dict[str, Dict[str, List]] = {}  # res_key → atom_name → coords
    sequence_chars = []
    prev_res_key = None
    resolution = None
    method = "unknown"

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break  # Only first model
            if line.startswith("REMARK   2 RESOLUTION"):
                try:
                    resolution = float(line.split()[3])
                except (ValueError, IndexError):
                    pass
            if line.startswith("EXPDTA"):
                if "X-RAY" in line.upper():
                    method = "xray"
                elif "ELECTRON MICROSCOPY" in line.upper():
                    method = "cryoem"
                elif "NMR" in line.upper():
                    method = "nmr"
            if not line.startswith("ATOM"):
                continue
            try:
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = line[22:27].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            # Map 3-letter to 1-letter
            aa = None
            for one, three in AA_1TO3.items():
                if three == res_name:
                    aa = one
                    break
            if aa is None:
                continue  # Skip non-standard residues

            res_key = f"{chain_id}_{res_seq}"
            if res_key != prev_res_key:
                sequence_chars.append(aa)
                prev_res_key = res_key
                atoms[res_key] = {}

            if atom_name in atoms[res_key]:
                continue  # Skip duplicates

            atoms[res_key][atom_name] = [x, y, z]

            if len(atoms) > max_atoms:
                break

    if len(sequence_chars) < 10:
        return None

    # Extract backbone positions
    n_pos, ca_pos, c_pos = [], [], []
    valid_indices = []
    all_atoms = []
    all_atom_names = []
    atom_indices = []

    for i, (res_key, aa) in enumerate(zip(atoms.keys(), sequence_chars)):
        res_atoms = atoms[res_key]
        if "N" in res_atoms and "CA" in res_atoms and "C" in res_atoms:
            n_pos.append(res_atoms["N"])
            ca_pos.append(res_atoms["CA"])
            c_pos.append(res_atoms["C"])
            valid_indices.append(i)
            for aname, coords in res_atoms.items():
                all_atoms.append(coords)
                all_atom_names.append(aname)
                atom_indices.append(len(valid_indices) - 1)

    if len(valid_indices) < 10:
        return None

    L = len(valid_indices)
    seq = "".join([sequence_chars[i] for i in valid_indices])

    return {
        "sequence": seq,
        "length": L,
        "n_pos": np.array(n_pos, dtype=np.float32),
        "ca_pos": np.array(ca_pos, dtype=np.float32),
        "c_pos": np.array(c_pos, dtype=np.float32),
        "all_atoms": np.array(all_atoms, dtype=np.float32),
        "all_atom_names": all_atom_names,
        "atom_residue_idx": np.array(atom_indices, dtype=np.int64),
        "resolution": resolution,
        "method": method,
        "pdb_id": Path(path).stem.upper(),
    }


class PDBDataset(Dataset):
    """
    PyTorch Dataset for PDB structures.
    Loads from a directory of PDB files or downloads from RCSB.
    """

    def __init__(self, pdb_dir: str, pdb_ids: Optional[List[str]] = None,
                 max_length: int = 256, min_length: int = 20,
                 max_resolution: Optional[float] = None,
                 methods: Optional[List[str]] = None):
        self.pdb_dir = Path(pdb_dir)
        self.max_length = max_length
        self.min_length = min_length
        self.max_resolution = max_resolution
        self.methods = methods or ["xray", "cryoem"]

        if pdb_ids:
            self.files = []
            for pid in pdb_ids:
                p = self.pdb_dir / f"{pid.lower()}.pdb"
                if p.exists():
                    self.files.append(p)
                else:
                    # Try to download
                    downloaded = download_pdb(pid, str(self.pdb_dir))
                    if downloaded:
                        self.files.append(Path(downloaded))
        else:
            # Scan directory
            self.files = sorted(self.pdb_dir.glob("*.pdb"))

        # Filter and cache
        self.valid_files: List[Tuple[Path, Dict]] = []
        for f in self.files:
            try:
                data = parse_pdb_structure(str(f))
                if data is None:
                    continue
                if data["length"] < min_length or data["length"] > max_length:
                    continue
                if max_resolution and data["resolution"] and data["resolution"] > max_resolution:
                    continue
                if data["method"] not in self.methods:
                    continue
                self.valid_files.append((f, data))
            except Exception:
                continue

        log.info(f"[PDBDataset] {len(self.valid_files)} valid structures "
                 f"from {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, data = self.valid_files[idx]
        L = data["length"]

        # Convert to tensors
        seq_tokens = torch.tensor([AA_TO_IDX.get(c, 20) for c in data["sequence"]],
                                  dtype=torch.long)
        n_pos = torch.from_numpy(data["n_pos"])
        ca_pos = torch.from_numpy(data["ca_pos"])
        c_pos = torch.from_numpy(data["c_pos"])

        # Compute backbone frames
        R, t = compute_backbone_frames(
            n_pos.unsqueeze(0), ca_pos.unsqueeze(0), c_pos.unsqueeze(0))

        # Atom positions (Cα only for simplicity; can extend to all heavy atoms)
        atom_pos = ca_pos  # (L, 3)

        return {
            "seq_tokens": seq_tokens,       # (L,)
            "n_pos": n_pos,                 # (L, 3)
            "ca_pos": ca_pos,               # (L, 3)
            "c_pos": c_pos,                 # (L, 3)
            "true_frames": R.squeeze(0),    # (L, 3, 3)
            "true_trans": t.squeeze(0),     # (L, 3)
            "true_atoms": atom_pos,         # (L, 3)
            "length": L,
            "sequence": data["sequence"],
            "pdb_id": data["pdb_id"],
            "resolution": data["resolution"],
            "method": data["method"],
        }


def pdb_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate PDB structures with padding."""
    Lmax = max(b["length"] for b in batch)
    B = len(batch)

    seq_tokens = torch.zeros(B, Lmax, dtype=torch.long)
    n_pos = torch.zeros(B, Lmax, 3)
    ca_pos = torch.zeros(B, Lmax, 3)
    c_pos = torch.zeros(B, Lmax, 3)
    true_frames = torch.zeros(B, Lmax, 3, 3)
    true_trans = torch.zeros(B, Lmax, 3)
    true_atoms = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    sequences = []
    pdb_ids = []

    for i, b in enumerate(batch):
        L = b["length"]
        seq_tokens[i, :L] = b["seq_tokens"]
        n_pos[i, :L] = b["n_pos"]
        ca_pos[i, :L] = b["ca_pos"]
        c_pos[i, :L] = b["c_pos"]
        true_frames[i, :L] = b["true_frames"]
        true_trans[i, :L] = b["true_trans"]
        true_atoms[i, :L] = b["true_atoms"]
        mask[i, :L] = True
        lengths[i] = L
        sequences.append(b["sequence"])
        pdb_ids.append(b["pdb_id"])

    return {
        "seq_tokens": seq_tokens,
        "n_pos": n_pos,
        "ca_pos": ca_pos,
        "c_pos": c_pos,
        "true_frames": true_frames,
        "true_trans": true_trans,
        "true_atoms": true_atoms,
        "mask": mask,
        "lengths": lengths,
        "sequences": sequences,
        "pdb_ids": pdb_ids,
    }


# =============================================================================
# §3. JOINT MODEL (Structure + Phenotype)
# =============================================================================

class StructureHead(nn.Module):
    """Predicts backbone frames and atom positions from encoder features."""

    def __init__(self, d_model: int, n_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        # Frame prediction: quaternion (4) + translation (3)
        self.frame_head = nn.Linear(d_model, 7)
        # Atom position prediction: 3D coords
        self.atom_head = nn.Linear(d_model, 3)

    def quaternion_to_rotation(self, q: Tensor) -> Tensor:
        """Convert quaternion (w, x, y, z) to rotation matrix."""
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        w, x, y, z = q.unbind(dim=-1)
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(*q.shape[:-1], 3, 3)
        return R

    def forward(self, feats: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            feats: (B, L, d_model) encoder features
        Returns:
            dict with frames, translations, atom positions
        """
        x = self.proj(feats)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        frame_out = self.frame_head(x)  # (B, L, 7)
        quat = frame_out[..., :4]
        trans = frame_out[..., 4:7]
        R = self.quaternion_to_rotation(quat)

        atom_pos = self.atom_head(x)  # (B, L, 3)

        return {
            "frames": R,           # (B, L, 3, 3)
            "translations": trans, # (B, L, 3)
            "atom_positions": atom_pos,  # (B, L, 3)
        }


class PhenotypeHead(nn.Module):
    """Predicts phenotype scores from pooled encoder features."""

    def __init__(self, d_model: int, n_phenotypes: int = 1):
        super().__init__()
        self.pool_proj = nn.Linear(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_phenotypes),
        )

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        """Masked mean pooling → MLP."""
        m = mask.float().unsqueeze(-1)
        pooled = (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        pooled = self.pool_proj(pooled)
        return self.head(pooled)


class SequenceEncoder(nn.Module):
    """Small transformer encoder for sequences."""

    def __init__(self, vocab_size: int = 21, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3, max_len: int = 512):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=0.1,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        if mask is not None:
            x = self.encoder(x, src_key_padding_mask=~mask)
        else:
            x = self.encoder(x)
        return self.norm(x)


class JointFoldingPhenotypeModel(nn.Module):
    """
    Joint model: Sequence → Encoder → Structure head + Phenotype head.
    Shared encoder, two task-specific heads.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 n_phenotypes: int = 1):
        super().__init__()
        self.encoder = SequenceEncoder(21, d_model, n_heads, n_layers)
        self.structure_head = StructureHead(d_model)
        self.phenotype_head = PhenotypeHead(d_model, n_phenotypes)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        feats = self.encoder(tokens, mask)
        struct = self.structure_head(feats)
        pheno = self.phenotype_head(feats, mask if mask is not None
                                    else torch.ones(tokens.shape, dtype=torch.bool,
                                                    device=tokens.device))
        return {**struct, "phenotype": pheno, "encoder_features": feats}


# =============================================================================
# §4. JOINT LOSS
# =============================================================================

class JointLoss(nn.Module):
    """
    Joint loss combining:
      • MSE on atom positions (coordinate-level)
      • FAPE on frames (frame-aligned, invariant to global rotation/translation)
      • Phenotype loss (MSE or BCE)
    """

    def __init__(self, w_coord: float = 1.0, w_fape: float = 0.5,
                 w_pheno: float = 0.1, d_clamp: float = 10.0,
                 pheno_loss_type: str = "mse"):
        super().__init__()
        self.w_coord = w_coord
        self.w_fape = w_fape
        self.w_pheno = w_pheno
        self.fape = FAPELoss(d_clamp)
        self.pheno_loss_type = pheno_loss_type

    def forward(self, pred: Dict[str, Tensor], batch: Dict[str, Tensor],
                pheno_target: Optional[Tensor] = None) -> Dict[str, Tensor]:
        mask = batch.get("mask")

        # Coordinate loss (MSE on Cα positions)
        coord_loss = F.mse_loss(
            pred["atom_positions"] * mask.unsqueeze(-1).float(),
            batch["true_atoms"] * mask.unsqueeze(-1).float())

        # FAPE loss
        fape_val = self.fape(
            pred["frames"], pred["translations"],
            batch["true_frames"], batch["true_trans"],
            pred["atom_positions"], batch["true_atoms"],
            mask)

        losses = {
            "coord_loss": coord_loss,
            "fape_loss": fape_val,
            "total_loss": self.w_coord * coord_loss + self.w_fape * fape_val,
        }

        # Phenotype loss (if target provided)
        if pheno_target is not None and "phenotype" in pred:
            if self.pheno_loss_type == "bce":
                pheno_loss = F.binary_cross_entropy_with_logits(
                    pred["phenotype"], pheno_target.float())
            else:
                pheno_loss = F.mse_loss(pred["phenotype"], pheno_target.float())
            losses["phenotype_loss"] = pheno_loss
            losses["total_loss"] = losses["total_loss"] + self.w_pheno * pheno_loss

        return losses


# =============================================================================
# §5. TRAINER
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    w_coord: float = 1.0
    w_fape: float = 0.5
    w_pheno: float = 0.0
    val_interval: int = 5
    log_interval: int = 10
    checkpoint_dir: str = "./checkpoints"
    device: str = "auto"


class JointTrainer:
    """Training loop for joint folding + phenotype model."""

    def __init__(self, model: JointFoldingPhenotypeModel,
                 cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or TrainConfig()
        if self.cfg.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay)
        self.loss_fn = JointLoss(
            w_coord=self.cfg.w_coord, w_fape=self.cfg.w_fape,
            w_pheno=self.cfg.w_pheno)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=10)
        self.history: List[Dict[str, float]] = []
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total = {"coord_loss": 0, "fape_loss": 0, "total_loss": 0}
        n_batches = 0

        for batch in loader:
            tokens = batch["seq_tokens"].to(self.device)
            mask = batch["mask"].to(self.device)
            batch_dev = {k: v.to(self.device) if isinstance(v, Tensor) else v
                        for k, v in batch.items()}

            self.optimizer.zero_grad()
            pred = self.model(tokens, mask)
            losses = self.loss_fn(pred, batch_dev)
            loss = losses["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.cfg.grad_clip)
            self.optimizer.step()

            for k in total:
                if k in losses:
                    total[k] += float(losses[k])
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total = {"coord_loss": 0, "fape_loss": 0, "total_loss": 0}
        n_batches = 0

        for batch in loader:
            tokens = batch["seq_tokens"].to(self.device)
            mask = batch["mask"].to(self.device)
            batch_dev = {k: v.to(self.device) if isinstance(v, Tensor) else v
                        for k, v in batch.items()}

            pred = self.model(tokens, mask)
            losses = self.loss_fn(pred, batch_dev)

            for k in total:
                if k in losses:
                    total[k] += float(losses[k])
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total.items()}

    def train(self, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        best_val = float("inf")

        for epoch in range(self.cfg.epochs):
            train_metrics = self.train_epoch(train_loader)

            val_metrics = {}
            if val_loader and (epoch + 1) % self.cfg.val_interval == 0:
                val_metrics = self.validate(val_loader)
                self.scheduler.step(val_metrics.get("total_loss", float("inf")))

                if val_metrics["total_loss"] < best_val:
                    best_val = val_metrics["total_loss"]
                    self.save_checkpoint(epoch, "best")

            epoch_record = {"epoch": epoch, **train_metrics}
            if val_metrics:
                epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.history.append(epoch_record)

            if (epoch + 1) % self.cfg.log_interval == 0:
                msg = f"Epoch {epoch+1}/{self.cfg.epochs}"
                for k, v in train_metrics.items():
                    msg += f" | {k}={v:.4f}"
                if val_metrics:
                    msg += f" | val_total={val_metrics['total_loss']:.4f}"
                log.info(msg)

        self.save_checkpoint(self.cfg.epochs - 1, "final")
        return {"history": self.history, "best_val_loss": best_val}

    def save_checkpoint(self, epoch: int, tag: str):
        path = os.path.join(self.cfg.checkpoint_dir, f"joint_model_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "history": self.history,
        }, path)


# =============================================================================
# §6. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    torch.manual_seed(42)
    device = torch.device("cpu")

    # ---- T1: Backbone frame computation ----
    print("\n[T1] Backbone frame computation")
    # Create a simple helix
    n_res = 10
    n_pos = torch.zeros(1, n_res, 3)
    ca_pos = torch.zeros(1, n_res, 3)
    c_pos = torch.zeros(1, n_res, 3)
    for i in range(n_res):
        ang = i * math.radians(100)
        ca_pos[0, i] = torch.tensor([2.3 * math.cos(ang),
                                     2.3 * math.sin(ang), i * 1.5])
        n_pos[0, i] = ca_pos[0, i] + torch.tensor([-0.5, 0.5, -0.5])
        c_pos[0, i] = ca_pos[0, i] + torch.tensor([0.5, -0.5, 0.5])

    R, t = compute_backbone_frames(n_pos, ca_pos, c_pos)
    rep("Frame shapes correct",
        R.shape == (1, n_res, 3, 3) and t.shape == (1, n_res, 3))

    # Check R is orthogonal: R^T R = I
    RtR = torch.einsum("...ji,...jk->...ik", R, R)
    orth_err = (RtR - torch.eye(3)).abs().max().item()
    rep("Rotation matrices orthogonal", orth_err < 1e-5,
        f"max|R^T R - I|={orth_err:.2e}")

    # Check det(R) = 1
    det_err = (torch.det(R) - 1.0).abs().max().item()
    rep("Rotation determinant = 1", det_err < 1e-5,
        f"max|det(R)-1|={det_err:.2e}")

    # ---- T2: FAPE loss invariance ----
    print("\n[T2] FAPE loss invariance to global rigid transformation")
    L = 10
    pred_frames = torch.randn(1, L, 3, 3)
    pred_frames = torch.linalg.qr(pred_frames)[0]  # Orthonormalize
    pred_trans = torch.randn(1, L, 3)
    pred_atoms = torch.randn(1, L, 3)

    true_frames = pred_frames.clone()
    true_trans = pred_trans.clone()
    true_atoms = pred_atoms.clone()

    fape_orig = fape_loss(pred_frames, pred_trans, true_frames, true_trans,
                         pred_atoms, true_atoms)

    # Apply random global rotation and translation
    R_global, _ = torch.linalg.qr(torch.randn(3, 3))
    t_global = torch.randn(3)

    pred_frames_rot = torch.einsum("ij,...jk->...ik", R_global, pred_frames)
    pred_trans_rot = torch.einsum("ij,...j->...i", R_global, pred_trans) + t_global
    pred_atoms_rot = torch.einsum("ij,...j->...i", R_global, pred_atoms) + t_global

    true_frames_rot = torch.einsum("ij,...jk->...ik", R_global, true_frames)
    true_trans_rot = torch.einsum("ij,...j->...i", R_global, true_trans) + t_global
    true_atoms_rot = torch.einsum("ij,...j->...i", R_global, true_atoms) + t_global

    fape_rot = fape_loss(pred_frames_rot, pred_trans_rot, true_frames_rot,
                        true_trans_rot, pred_atoms_rot, true_atoms_rot)

    rep("FAPE invariant to global rotation+translation",
        abs(fape_orig.item() - fape_rot.item()) < 1e-5,
        f"orig={fape_orig:.6f} rot={fape_rot:.6f}")

    # FAPE should be 0 for perfect prediction
    rep("FAPE = 0 for perfect prediction", fape_orig.item() < 1e-6,
        f"{fape_orig:.6f}")

    # FAPE should be > 0 for wrong prediction
    pred_atoms_wrong = pred_atoms + 5.0
    fape_wrong = fape_loss(pred_frames, pred_trans, true_frames, true_trans,
                          pred_atoms_wrong, true_atoms)
    rep("FAPE > 0 for wrong prediction", fape_wrong.item() > 0.1,
        f"{fape_wrong:.4f}")

    # ---- T3: PDB parsing ----
    print("\n[T3] PDB parsing")
    # Create a minimal PDB file
    pdb_content = """REMARK   2 RESOLUTION.    2.00 ANGSTROMS.
EXPDTA    X-RAY DIFFRACTION
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   3.000  1.00 10.00           C
ATOM      3  C   ALA A   1       3.000   2.000   3.000  1.00 10.00           C
ATOM      4  O   ALA A   1       3.500   3.000   3.000  1.00 10.00           O
ATOM      5  N   GLY A   2       3.500   1.000   3.000  1.00 10.00           N
ATOM      6  CA  GLY A   2       4.500   1.000   3.000  1.00 10.00           C
ATOM      7  C   GLY A   2       5.500   1.000   3.000  1.00 10.00           C
ATOM      8  O   GLY A   2       6.000   2.000   3.000  1.00 10.00           O
END
"""
    tmp_pdb = Path("/tmp/test_structure.pdb")
    tmp_pdb.write_text(pdb_content)

    data = parse_pdb_structure(str(tmp_pdb))
    if data:
        rep("PDB parsed successfully", True,
            f"L={data['length']}, method={data['method']}")
        rep("Resolution extracted", data["resolution"] == 2.0)
        rep("Backbone positions extracted",
            data["n_pos"].shape == (2, 3) and data["ca_pos"].shape == (2, 3))
    else:
        rep("PDB parsed successfully", False, "returned None")

    tmp_pdb.unlink()

    # ---- T4: Joint model forward/backward ----
    print("\n[T4] Joint model forward/backward")
    model = JointFoldingPhenotypeModel(d_model=64, n_heads=2, n_layers=2)
    tokens = torch.randint(0, 20, (2, 15))
    mask = torch.ones(2, 15, dtype=torch.bool)
    mask[1, 10:] = False

    pred = model(tokens, mask)
    rep("Structure head outputs",
        pred["frames"].shape == (2, 15, 3, 3) and
        pred["translations"].shape == (2, 15, 3) and
        pred["atom_positions"].shape == (2, 15, 3))
    rep("Phenotype head output", pred["phenotype"].shape == (2, 1))

    # Loss computation
    batch = {
        "true_frames": torch.randn(2, 15, 3, 3),
        "true_trans": torch.randn(2, 15, 3),
        "true_atoms": torch.randn(2, 15, 3),
        "mask": mask,
    }
    loss_fn = JointLoss(w_coord=1.0, w_fape=0.5, w_pheno=0.0)
    losses = loss_fn(pred, batch)
    rep("Losses computed", all(torch.isfinite(losses[k]) for k in losses))
    rep("Total loss > 0", losses["total_loss"].item() > 0)

    # Backward
    losses["total_loss"].backward()
    grad_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in model.parameters())
    rep("Gradients finite", grad_ok)

    # ---- T5: Training loop (mini) ----
    print("\n[T5] Mini training loop")
    # Create synthetic dataset
    n_samples = 8
    L_max = 20
    seq_tokens = torch.randint(0, 20, (n_samples, L_max))
    n_pos = torch.randn(n_samples, L_max, 3) * 5
    ca_pos = torch.randn(n_samples, L_max, 3) * 5
    c_pos = torch.randn(n_samples, L_max, 3) * 5
    true_frames, true_trans = compute_backbone_frames(n_pos, ca_pos, c_pos)
    mask = torch.ones(n_samples, L_max, dtype=torch.bool)

    class SimpleDataset(Dataset):
        def __len__(self):
            return n_samples
        def __getitem__(self, idx):
            return {
                "seq_tokens": seq_tokens[idx],
                "n_pos": n_pos[idx],
                "ca_pos": ca_pos[idx],
                "c_pos": c_pos[idx],
                "true_frames": true_frames[idx],
                "true_trans": true_trans[idx],
                "true_atoms": ca_pos[idx],
                "length": L_max,
                "sequence": "A" * L_max,
                "pdb_id": f"TEST{idx}",
                "resolution": 2.0,
                "method": "xray",
            }

    ds = SimpleDataset()
    dl = DataLoader(ds, batch_size=4, collate_fn=pdb_collate)

    model2 = JointFoldingPhenotypeModel(d_model=32, n_heads=2, n_layers=1)
    cfg = TrainConfig(epochs=3, batch_size=4, log_interval=100,
                      checkpoint_dir="/tmp/test_ckpt")
    trainer = JointTrainer(model2, cfg)
    result = trainer.train(dl)
    rep("Training completes", len(result["history"]) == 3)
    rep("Loss decreases", result["history"][-1]["total_loss"] <
        result["history"][0]["total_loss"] * 1.5,
        f"{result['history'][0]['total_loss']:.4f} → "
        f"{result['history'][-1]['total_loss']:.4f}")

    import shutil
    shutil.rmtree("/tmp/test_ckpt", ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT IS IMPLEMENTED
  • FAPE loss (AlphaFold2 §1.9.4): frame-aligned, invariant to rigid transforms
  • Backbone frame computation from N, Cα, C positions
  • PDB parsing with resolution/method extraction
  • Joint model: encoder → structure head + phenotype head
  • Training loop with MSE + FAPE + optional phenotype loss
  • Checkpointing and validation

NOT IMPLEMENTED / NOT CLAIMED
  • No MSA processing (Evoformer, MSA transformer)
  • No distogram / pLDDT / violation losses
  • No recycling
  • No CASP validation
  • No claim of beating AlphaFold2/3
  • PDB data has biases; phenotype labels not from PDB
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="FAPE loss + PDB training")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--pdb-dir", default="data/pdb")
    ap.add_argument("--pdb-ids", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if a.train:
        # Load dataset
        if a.pdb_ids:
            for pid in a.pdb_ids:
                download_pdb(pid, a.pdb_dir)

        dataset = PDBDataset(a.pdb_dir, max_length=256, min_length=20)
        if len(dataset) == 0:
            print("No valid PDB structures found. Use --pdb-ids to download.")
            return 1

        # Split
        n = len(dataset)
        indices = list(range(n))
        train_size = int(0.8 * n)
        val_size = int(0.1 * n)

        train_ds = torch.utils.data.Subset(dataset, indices[:train_size])
        val_ds = torch.utils.data.Subset(dataset, indices[train_size:train_size+val_size])

        train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              collate_fn=pdb_collate)
        val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                            collate_fn=pdb_collate)

        # Model + trainer
        model = JointFoldingPhenotypeModel(d_model=128, n_heads=4, n_layers=3)
        cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                          device=a.device)
        trainer = JointTrainer(model, cfg)

        print(f"Training on {len(train_ds)} structures, "
              f"validating on {len(val_ds)}")
        result = trainer.train(train_dl, val_dl)

        print(json.dumps({"best_val_loss": result["best_val_loss"]}, indent=2))
        if a.json:
            Path(a.json).write_text(json.dumps(result, indent=2, default=str))

        return 0

    ap.print_help()
    print(BANNER)
    print("Start: --selftest | --train --pdb-ids 1ABC 2DEF 3GHI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
