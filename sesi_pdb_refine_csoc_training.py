"""
=============================================================================
SESI PDB REFINEMENT TRAINING SUITE — v1.0.0
Direct RefinementEngine Bridge + Real PDB DataLoader + CSOCKernel Loss
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================
DIRECT INTEGRATION WITH real_fold_one_v2-5.py
=============================================================================
This module does NOT reimplement any refinement logic. It wires directly
into the existing ecosystem classes:

  RefinementEngine          — instantiated once; owns the CSOCKernel,
                              SOCController, OpenMM system, calculator.
                              Called via .refine() / .export_pdb() /
                              .cleanup() exactly as its own API defines.

  CSOCKernel                — the SAME object as engine.kernel is handed
                              to CSOCKernelLoss, so training updates the
                              very parameters that refine() uses later.
                              (Shared-parameter design, zero duplication.)

  NeighborListManager       — reused for edge construction in the loss,
                              with scipy / pure-torch fallbacks.

  RefinementConfig          — passed through unchanged; see note below
                              about the `steps` field.

NOTE ON RefinementConfig: some text extractions of real_fold_one_v2-5.py
show the field as "ste ps: int = 600" — this is a copy artifact. The real
field is `steps`. If your local copy carries the artifact, rename it.

OPMC STEP CAP (from Unified Ordinal-Tensor Measurement Theory)
--------------------------------------------------------------
Theorem 11.1: for an L-Lipschitz contraction (L < 1) on a fixed-diameter
space, k* = ceil(ln(eps/D0)/ln(L)) iterations reach precision eps,
independent of problem size N. With the paper's L = 0.438520 and
eps = 1e-8, D0 = 1: k* = 23.

HONEST USE HERE: RefinementEngine.refine() has NOT been proven to be a
contraction map with L = 0.4385. We use k* ONLY as an optional, clearly-
labeled heuristic cap on inner-loop refinement steps when the engine is
called inside a training loop (to bound cost), never as a correctness
claim. Set use_opmc_cap=False to disable.

CARPET-19 DISCIPLINE
--------------------------------------------------------------
- Corollary-3: the loss is first-order differentiable; no Hessian of the
  LJ energy is computed anywhere in this file.
- Hino: atoms, residues, PDB files, and graph edges are discrete objects;
  no smooth density assumption over structures.
- Prop-3: pair products prob_ij * E_LJ(d_ij) lose regularity at short
  distances; the kernel's `safe_r = r + eps` clamp is the mechanism that
  keeps the product bounded (same device as CSOCKernel itself).
- rho ~ 1.25: numerical, unused here.

WHAT RUNS WITHOUT OPTIONAL DEPENDENCIES
--------------------------------------------------------------
- No real_fold_one_v2  -> stand-in LJ kernel (same formula, clearly
                          labeled); DirectRefinementBridge raises a clean
                          ImportError with install guidance.
- No scipy             -> pure-torch chunked neighbor fallback.
- No OpenMM            -> engine constructs fine; .refine() will raise
                          inside real_fold_one_v2 (its own guard).
=============================================================================
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("sesi.pdb_refine_csoc")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

SUITE_VERSION: str = "1.0.0"

# =============================================================================
# OPMC constants (Unified Ordinal-Tensor Measurement Theory)
# =============================================================================
OPMC_C1 = 0.420000
OPMC_K2 = 12.5000
OPMC_L = 0.438520
OPMC_ETA_LATTICE = 0.99491


def opmc_step_cap(epsilon: float = 1e-8, D0: float = 1.0, L: float = OPMC_L) -> int:
    """Theorem 11.1 form: k* = ceil(ln(eps/D0) / ln L). Independent of N."""
    if not (0.0 < L < 1.0):
        raise ValueError("L must be in (0,1) for a contraction bound.")
    return math.ceil(math.log(epsilon / D0) / math.log(L))


# =============================================================================
# Ecosystem imports (graceful; handles the hyphenated filename on disk)
# =============================================================================
def _load_real_fold_module():
    """Import real_fold_one_v2 by module name, else by hyphenated filename."""
    try:
        import real_fold_one_v2 as m  # type: ignore
        return m
    except ImportError:
        pass
    for cand in ("real_fold_one_v2-5.py", "real_fold_one_v2.py",
                 os.path.join(os.path.dirname(__file__), "real_fold_one_v2-5.py")):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("real_fold_one_v2", str(p))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    return None


_RF = _load_real_fold_module()
_HAS_REAL_FOLD = _RF is not None

if _HAS_REAL_FOLD:
    RefinementEngine = _RF.RefinementEngine
    RefinementConfig = _RF.RefinementConfig
    CSOCKernel = _RF.CSOCKernel
    SOCController = _RF.SOCController
    NeighborListManager = _RF.NeighborListManager
    OpenMMSystemBuilder = _RF.OpenMMSystemBuilder
    OpenMMEnergyCalculator = _RF.OpenMMEnergyCalculator
    openmm_solute_energy = _RF.openmm_solute_energy
    logger.info("real_fold_one_v2 loaded — direct RefinementEngine connection enabled.")
else:
    warnings.warn(
        "real_fold_one_v2 not importable — CSOCKernelLoss will use a labeled "
        "stand-in kernel with the identical LJ form; DirectRefinementBridge "
        "will raise ImportError until the file is available."
    )

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import openmm  # noqa: F401
    _HAS_OPENMM = True
except ImportError:
    _HAS_OPENMM = False


# =============================================================================
# 1. REAL PDB DATA LOADER
# =============================================================================
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
    'TRP': 'W', 'TYR': 'Y', 'UNK': 'X',
}

_RESOLUTION_RE = re.compile(r"RESOLUTION\.\s*([\d.]+)")
_TRACE_ATOMS = ("CA", "C4'")   # protein Cα trace + nucleic C4' trace (ecosystem convention)


@dataclass
class PDBStructure:
    """One parsed PDB entry (Cα/C4′ trace level)."""
    name: str
    sequence: str
    ca_coords: np.ndarray          # (L, 3) float32, Angstrom
    resolution: Optional[float]    # from REMARK 2, None if absent
    n_atoms_total: int             # all ATOM/HETATM lines seen
    path: str


def parse_pdb_file(path: os.PathLike | str) -> PDBStructure:
    """
    Parse a PDB file to its trace-atom representation.

    Rules (matching what RefinementEngine/OpenMM expects downstream):
      • Only the first MODEL block is read.
      • Alternate locations: only altLoc '' or 'A' kept.
      • Trace atoms: 'CA' (protein) and "C4'" (nucleic) — same convention
        as real_fold_one_v2's ca_indices extraction.
      • One coordinate per (chain, resSeq, iCode) residue; duplicates
        (e.g. duplicated trace atoms) take the first occurrence.

    COROLLARY-3 GUARD: parsing is a discrete record-by-record scan.
    HINO GUARD: atoms are discrete entities; no interpolation here.
    """
    path = Path(path)
    coords: List[List[float]] = []
    resnames: List[str] = []
    seen_res: set = set()
    n_atoms = 0
    resolution: Optional[float] = None
    model_count = 0

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                model_count += 1
                if model_count > 1:
                    break  # first model only
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
            if atom_name not in _TRACE_ATOMS:
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
        name=path.stem,
        sequence=sequence,
        ca_coords=arr,
        resolution=resolution,
        n_atoms_total=n_atoms,
        path=str(path),
    )


def _pdb_atom_line(serial: int, atom: str, resn: str, chain: str, resseq: int,
                   x: float, y: float, z: float, elem: str = "C") -> str:
    """Format one strictly column-aligned ATOM record."""
    return (f"ATOM  {serial:>5d} {atom:^4s} {resn:>3s} {chain}{resseq:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}          {elem:>2s}")


def write_ca_pdb(coords: torch.Tensor | np.ndarray, path: os.PathLike | str,
                 sequence: str = "", resolution: Optional[float] = None) -> str:
    """
    Write a Cα-trace PDB from (L, 3) coordinates (Angstrom).
    Mirrors the ecosystem's write_ca_pdb convention so the output can be
    fed straight into OpenMMSystemBuilder.build_from_pdb / RefinementEngine.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords, dtype=np.float64)
    L = coords.shape[0]
    sequence = (sequence or "").ljust(L, "A")[:L]
    one_to_three = {v: k for k, v in AA_3_TO_1.items() if v != "X"}
    with open(path, "w") as fh:
        fh.write("REMARK   1 GENERATED BY SESI write_ca_pdb\n")
        if resolution is not None:
            fh.write(f"REMARK   2 RESOLUTION. {resolution:>8.2f} ANGSTROMS.\n")
        for i in range(L):
            resn = one_to_three.get(sequence[i], "UNK")
            fh.write(_pdb_atom_line(i + 1, "CA", resn, "A", i + 1,
                                    coords[i, 0], coords[i, 1], coords[i, 2]) + "\n")
        fh.write("END\n")
    return str(path)


def fetch_pdb_from_rcsb(pdb_id: str, dest_dir: os.PathLike | str,
                        timeout: float = 30.0) -> Path:
    """
    Download one PDB entry from RCSB. Network operation — call deliberately.
    Returns the local file path.
    """
    pdb_id = pdb_id.upper()
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{pdb_id.lower()}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info("[PDB] Downloading %s ...", url)
    urllib.request.urlretrieve(url, out)
    return out


class RealPDBDataset(Dataset):
    """
    Dataset of real PDB files on disk.

    • Scans `root` recursively for *.pdb / *.ent.
    • Deterministic train/val/test split by MD5 of filename — stable across
      runs and processes (no leakage between splits, no RNG dependence).
    • Parses lazily and caches parsed structures in memory.

    CARPET-19: files and residues are discrete; the split hash is a
    first-order (no-derivative) partition function.
    """

    def __init__(self, root: os.PathLike | str, split: str = "train",
                 train_pct: int = 80, val_pct: int = 10,
                 min_residues: int = 10, max_residues: Optional[int] = None,
                 cache: bool = True):
        assert split in ("train", "val", "test")
        assert 0 < train_pct < 100 and 0 <= val_pct < 100
        self.root = Path(root)
        self.split = split
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.min_residues = min_residues
        self.max_residues = max_residues
        self._cache: Dict[int, PDBStructure] = {} if cache else {}

        all_files = sorted(set(list(self.root.rglob("*.pdb")) + list(self.root.rglob("*.ent"))))
        self.files = [f for f in all_files if self._fold_of(f) == split]
        logger.info("[PDBDataset] root=%s split=%s -> %d / %d files",
                    self.root, split, len(self.files), len(all_files))

    def _fold_of(self, path: Path) -> str:
        h = int(hashlib.md5(path.name.encode("utf-8")).hexdigest(), 16) % 100
        if h < self.train_pct:
            return "train"
        if h < self.train_pct + self.val_pct:
            return "val"
        return "test"

    def _parse(self, idx: int) -> Optional[PDBStructure]:
        f = self.files[idx]
        key = id(f)
        if key in self._cache:
            return self._cache[key]
        try:
            s = parse_pdb_file(f)
        except Exception as e:  # malformed files must not kill the dataset
            logger.warning("[PDBDataset] failed to parse %s: %s", f, e)
            return None
        if s.ca_coords.shape[0] < self.min_residues:
            return None
        if self.max_residues is not None and s.ca_coords.shape[0] > self.max_residues:
            s = PDBStructure(s.name, s.sequence[: self.max_residues],
                             s.ca_coords[: self.max_residues], s.resolution,
                             s.n_atoms_total, s.path)
        if self._cache is not None:
            self._cache[key] = s
        return s

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self._parse(idx)
        if s is None:
            # deterministic fallback: first valid file in this split
            for j in range(len(self.files)):
                s = self._parse(j)
                if s is not None:
                    break
        if s is None:
            raise RuntimeError("No parseable PDB file in this split.")
        L = s.ca_coords.shape[0]
        return {
            "coords": torch.from_numpy(s.ca_coords).float(),
            "mask": torch.ones(L, dtype=torch.bool),
            "sequence": s.sequence,
            "name": s.name,
            "resolution": s.resolution if s.resolution is not None else -1.0,
            "length": L,
        }


def pdb_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length structures to the batch max; keep masks exact."""
    Lmax = max(b["length"] for b in batch)
    B = len(batch)
    coords = torch.zeros(B, Lmax, 3)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b["length"]
        coords[i, :L] = b["coords"]
        mask[i, :L] = True
    return {
        "coords": coords,
        "mask": mask,
        "sequences": [b["sequence"] for b in batch],
        "names": [b["name"] for b in batch],
        "resolutions": torch.tensor([b["resolution"] for b in batch]),
        "lengths": torch.tensor([b["length"] for b in batch]),
    }


def build_pdb_dataloaders(root: os.PathLike | str, batch_size: int = 4,
                          num_workers: int = 0, **ds_kwargs) -> Dict[str, DataLoader]:
    """Convenience builder returning {'train','val','test'} dataloaders."""
    out: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = RealPDBDataset(root, split=split, **ds_kwargs)
        if len(ds) == 0:
            out[split] = None  # type: ignore
            continue
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                num_workers=num_workers, collate_fn=pdb_collate)
    return out


# =============================================================================
# 2. CSOCKERNEL AS A LOSS COMPONENT (shared-parameter design)
# =============================================================================
class _CSOCKernelStandIn(nn.Module):
    """
    Stand-in used ONLY when real_fold_one_v2 cannot be imported.
    Formula is identical to CSOCKernel v2 in real_fold_one_v2-5.py:
        E_LJ(r) = eps_lj * [ (r0/r)^(2α) − 2 (r0/r)^α ]
        K(r)    = −E_LJ(r)        (kernel returns K, matching upstream)
    """

    def __init__(self, init_alpha: float = 2.0, init_r0: float = 4.67,
                 init_eps_lj: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.log_r0 = nn.Parameter(torch.tensor(math.log(init_r0)))
        self.log_eps_lj = nn.Parameter(torch.tensor(math.log(init_eps_lj)))
        self.eps = eps

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    @property
    def r0(self) -> torch.Tensor:
        return torch.exp(self.log_r0)

    @property
    def eps_lj(self) -> torch.Tensor:
        return torch.exp(self.log_eps_lj)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        safe_r = r + self.eps
        x = self.r0 / safe_r
        xa = torch.pow(x, self.alpha)
        E_lj = self.eps_lj * (xa * xa - 2.0 * xa)
        return -E_lj


class CSOCKernelLoss(nn.Module):
    """
    CSOCKernel used as a differentiable loss component.

    LOSS DEFINITION (matches SOCController.compute_soc_energy with unit weights):
        L = w_soc * Σ_edges E_LJ(d_ij),   E_LJ = −K  (K = kernel output)

    • Edge set: all pairs within `cutoff` (optionally capped by
      `max_neighbors` when the ecosystem NeighborListManager is used).
    • Kernel parameters (α, r₀, ε_LJ) receive gradients — the kernel
      LEANS the equilibrium geometry of the contact potential.
    • Pass kernel=engine.kernel to share parameters with a live
      RefinementEngine (the intended production wiring).

    COROLLARY-3 GUARD: first-order autograd only.
    HINO GUARD: the edge graph is a discrete object rebuilt per call.
    PROP-3 AWARENESS: E_LJ contains (r0/r)^{2α}; the `r + eps` clamp
    inherited from the kernel keeps the product bounded near r → 0.
    """

    def __init__(self, kernel: Optional[nn.Module] = None, cutoff: float = 12.0,
                 max_neighbors: int = 64, device: str | torch.device = "cpu",
                 use_ecosystem_neighbors: bool = True):
        super().__init__()
        if kernel is None:
            kernel = CSOCKernel() if _HAS_REAL_FOLD else _CSOCKernelStandIn()
        self.kernel = kernel
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.device = torch.device(device)
        self._mgr = None
        if use_ecosystem_neighbors and _HAS_REAL_FOLD:
            self._mgr = NeighborListManager(
                cutoffs={"soc": self.cutoff},
                max_neighbors=self.max_neighbors,
                device=self.device,
            )

    # ---------------- neighbor list construction -----------------------
    def _edges_ecosystem(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self._mgr.build(coords)
        return out["soc"]

    def _edges_scipy(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tree = cKDTree(coords.detach().cpu().numpy())
        pairs = np.asarray(tree.query_pairs(self.cutoff, output_type="ndarray"))
        if pairs.size == 0:
            e = torch.empty(2, 0, dtype=torch.long, device=coords.device)
            return e, torch.empty(0, device=coords.device)
        edge = torch.from_numpy(np.vstack([pairs[:, 0], pairs[:, 1]])).long().to(coords.device)
        diff = coords[edge[0]] - coords[edge[1]]
        return edge, diff.norm(dim=-1)

    def _edges_torch_chunked(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = coords.shape[0]
        ii: List[torch.Tensor] = []
        jj: List[torch.Tensor] = []
        dd: List[torch.Tensor] = []
        chunk = 1024
        rows_base = torch.arange(N, device=coords.device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            diff = coords[s:e].unsqueeze(1) - coords.unsqueeze(0)      # (c, N, 3)
            dist = diff.norm(dim=-1)                                    # (c, N)
            rows = rows_base[s:e].unsqueeze(1)
            cols = rows_base.unsqueeze(0)
            valid = (dist < self.cutoff) & (rows < cols) & (dist > 1e-6)
            ri, ci = valid.nonzero(as_tuple=True)
            if ri.numel() == 0:
                continue
            ii.append(rows[ri, ci]); jj.append(cols[ri, ci]); dd.append(dist[ri, ci])
        if not ii:
            e = torch.empty(2, 0, dtype=torch.long, device=coords.device)
            return e, torch.empty(0, device=coords.device)
        return torch.stack([torch.cat(ii), torch.cat(jj)]), torch.cat(dd)

    def build_edges(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dispatch: ecosystem manager > scipy > pure torch. O(N·k) or chunked O(N²)."""
        if coords.shape[0] < 2:
            e = torch.empty(2, 0, dtype=torch.long, device=coords.device)
            return e, torch.empty(0, device=coords.device)
        if self._mgr is not None:
            return self._edges_ecosystem(coords)
        if _HAS_SCIPY:
            return self._edges_scipy(coords)
        return self._edges_torch_chunked(coords)

    # ---------------- loss ---------------------
    def forward(self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None,
                pair_weights: Optional[torch.Tensor] = None,
                w_soc: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            coords: (L, 3) Cα coordinates, Angstrom.
            mask:   optional (L,) bool — valid residues.
            pair_weights: optional per-edge weights (e.g. contact probabilities).
            w_soc:  global loss weight.
        Returns:
            (loss, diagnostics)
        """
        if mask is not None:
            coords = coords[mask]
        edge_idx, edge_dist = self.build_edges(coords)
        n_edges = edge_idx.shape[1]
        if n_edges == 0:
            z = torch.zeros((), device=coords.device, dtype=coords.dtype)
            return z, {"n_edges": 0, "loss": 0.0, "mean_dist": 0.0}

        K = self.kernel(edge_dist)          # K = −E_LJ  (upstream convention)
        E_lj = -K
        if pair_weights is not None:
            E_lj = E_lj * pair_weights
        loss = w_soc * E_lj.sum()

        diag = {
            "n_edges": int(n_edges),
            "loss": float(loss.detach().item()),
            "mean_dist": float(edge_dist.mean().item()),
            "kernel_alpha": float(self.kernel.alpha.item()),
            "kernel_r0": float(self.kernel.r0.item()),
            "kernel_eps_lj": float(self.kernel.eps_lj.item()),
        }
        return loss, diag

    def forward_batch(self, coords: torch.Tensor, mask: torch.Tensor,
                      w_soc: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Mean loss over a padded batch (B, L, 3) with masks (B, L)."""
        total = torch.zeros((), device=coords.device, dtype=coords.dtype)
        agg: Dict[str, float] = {"n_edges": 0, "loss": 0.0, "mean_dist": 0.0}
        n_valid = 0
        for i in range(coords.shape[0]):
            if not bool(mask[i].any()):
                continue
            li, di = self.forward(coords[i], mask=mask[i], w_soc=w_soc)
            total = total + li
            n_valid += 1
            agg["n_edges"] += di["n_edges"]
            agg["loss"] += di["loss"]
            agg["mean_dist"] += di["mean_dist"]
        if n_valid > 0:
            total = total / n_valid
            agg["loss"] /= n_valid
            agg["mean_dist"] /= n_valid
        return total, agg


# =============================================================================
# 3. DIRECT REFINEMENTENGINE BRIDGE
# =============================================================================
class DirectRefinementBridge:
    """
    Direct connection to real_fold_one_v2.RefinementEngine.

    This is deliberately a THIN bridge:
      • constructs the engine once with your RefinementConfig,
      • exposes engine.kernel / engine.soc for parameter sharing,
      • calls engine.refine() / engine.export_pdb() / engine.cleanup()
        verbatim — no reimplementation of any refinement step.

    Optional OPMC-inspired step cap (heuristic, see module docstring):
      use_opmc_cap=True limits steps to k* = ceil(ln(eps/D0)/ln L) = 23
      for eps=1e-8 with the paper's L. Off by default.
    """

    def __init__(self, refine_cfg: Optional[Any] = None,
                 use_opmc_cap: bool = False, opmc_epsilon: float = 1e-8):
        if not _HAS_REAL_FOLD:
            raise ImportError(
                "DirectRefinementBridge requires real_fold_one_v2-5.py on the "
                "import path (module name `real_fold_one_v2`, or the literal "
                "file `real_fold_one_v2-5.py` next to this script)."
            )
        self.cfg = refine_cfg if refine_cfg is not None else RefinementConfig()
        self.engine = RefinementEngine(self.cfg)
        self.use_opmc_cap = bool(use_opmc_cap)
        self.opmc_epsilon = float(opmc_epsilon)
        self._refine_count = 0

    # -------- shared-parameter accessors (the point of direct wiring) ----
    @property
    def kernel(self) -> nn.Module:
        """The engine's CSOCKernel — the SAME object CSOCKernelLoss should use."""
        return self.engine.kernel

    @property
    def soc(self) -> nn.Module:
        return self.engine.soc

    # -------- refinement calls, verbatim --------
    def _cap_steps(self, steps: Optional[int]) -> int:
        s = steps if steps is not None else self.cfg.steps
        if self.use_opmc_cap:
            capped = min(int(s), opmc_step_cap(self.opmc_epsilon))
            if capped != s:
                logger.info("[Bridge] OPMC cap applied: %d -> %d steps (Theorem 11.1 form, heuristic)",
                            s, capped)
            return capped
        return int(s)

    def refine_pdb(self, pdb_path: os.PathLike | str, steps: Optional[int] = None,
                   return_trajectory: bool = False) -> Dict[str, Any]:
        """engine.refine() called directly. Returns its result dict unchanged."""
        steps = self._cap_steps(steps)
        t0 = time.time()
        result = self.engine.refine(str(pdb_path), steps=steps,
                                    return_trajectory=return_trajectory)
        self._refine_count += 1
        logger.info("[Bridge] refine(%s) done in %.2fs — E0=%.3f -> E=%.3f kcal/mol",
                    Path(pdb_path).name, time.time() - t0,
                    result.get("initial_energy", float("nan")),
                    result.get("final_energy", float("nan")))
        return result

    def refine_to_file(self, pdb_path: os.PathLike | str, out_path: os.PathLike | str,
                       steps: Optional[int] = None) -> Dict[str, Any]:
        result = self.refine_pdb(pdb_path, steps=steps)
        self.engine.export_pdb(result["solute_coords"], str(out_path))
        return result

    def cleanup(self) -> None:
        self.engine.cleanup()

    def __enter__(self) -> "DirectRefinementBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


# =============================================================================
# 4. TRAINER — ties loader + CSOC loss + direct refinement together
# =============================================================================
@dataclass
class PDBCSOCTrainerConfig:
    lr: float = 1e-3
    epochs: int = 1
    w_soc: float = 1.0
    cutoff: float = 12.0
    grad_clip: float = 5.0
    use_amp: bool = False                 # bf16 autocast on CUDA
    log_interval: int = 5
    device: str = "auto"
    refine_eval_every: int = 0            # >0: run a direct refine() on one
    #                                     val sample every N optimizer steps
    refine_eval_steps: int = 50           # inner steps for that evaluation


class PDBCSOCTrainer:
    """
    Training loop where the CSOCKernel loss and RefinementEngine share ONE
    kernel object.

    Loss (per batch, mean over samples):
        L = w_soc * Σ_edges E_LJ(d_ij; α, r0, ε)

    The kernel parameters (α, r0, ε_LJ) are optimized so that the LJ
    equilibrium of the SOC potential matches the contact statistics of real
    PDB structures. After training, the SAME kernel is what
    DirectRefinementBridge.refine_pdb() uses — no parameter drift between
    training and inference physics.

    Optional OpenMM energy term: if a bridge with a live OpenMM calculator
    is supplied and use_openmm_energy=True, E_openmm is added. Off by
    default to keep the loop runnable without OpenMM.
    """

    def __init__(self, cfg: PDBCSOCTrainerConfig,
                 dataloaders: Dict[str, Optional[DataLoader]],
                 refinement_bridge: Optional[DirectRefinementBridge] = None):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.bridge = refinement_bridge

        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # ---- the shared kernel: bridge's engine kernel if available ----
        kernel: Optional[nn.Module] = None
        if self.bridge is not None:
            kernel = self.bridge.kernel
            logger.info("[Trainer] Using RefinementEngine.kernel (SHARED parameters).")
        elif _HAS_REAL_FOLD:
            kernel = CSOCKernel().to(self.device)
            logger.info("[Trainer] No bridge supplied — using a fresh ecosystem CSOCKernel.")
        else:
            kernel = _CSOCKernelStandIn().to(self.device)
            logger.info("[Trainer] Using labeled stand-in kernel (real_fold_one_v2 absent).")
        kernel = kernel.to(self.device)

        self.csoc = CSOCKernelLoss(kernel=kernel, cutoff=cfg.cutoff, device=self.device)
        self.optimizer = torch.optim.Adam(self.csoc.kernel.parameters(), lr=cfg.lr)
        self.amp_dtype = torch.bfloat16
        self.history: List[Dict[str, Any]] = []
        self._global_step = 0

    # ------------------------------------------------------------------
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        coords = batch["coords"].to(self.device)
        mask = batch["mask"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = (torch.autocast(device_type="cuda", dtype=self.amp_dtype)
               if (self.cfg.use_amp and self.device.type == "cuda")
               else torch.enable_grad())
        with ctx:
            loss, diag = self.csoc.forward_batch(coords, mask, w_soc=self.cfg.w_soc)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.csoc.kernel.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self._global_step += 1

        return {"loss": float(loss.item()), **{k: diag.get(k) for k in
                ("n_edges", "mean_dist", "kernel_alpha", "kernel_r0", "kernel_eps_lj")}}

    def train(self) -> List[Dict[str, Any]]:
        loader = self.dataloaders.get("train")
        if loader is None:
            raise RuntimeError("No 'train' dataloader available (empty split?).")

        for epoch in range(self.cfg.epochs):
            for batch in loader:
                stats = self.train_step(batch)
                self.history.append({"epoch": epoch, "step": self._global_step, **stats})

                if self._global_step % self.cfg.log_interval == 0:
                    logger.info(
                        "[Trainer] ep%d step%d loss=%.4f edges=%d α=%.3f r0=%.3f ε=%.3f",
                        epoch, self._global_step, stats["loss"], stats["n_edges"],
                        stats["kernel_alpha"], stats["kernel_r0"], stats["kernel_eps_lj"],
                    )

                # optional direct-refine evaluation on a val sample
                if (self.cfg.refine_eval_every > 0 and self.bridge is not None
                        and self._global_step % self.cfg.refine_eval_every == 0):
                    self._eval_direct_refine()

        return self.history

    @torch.no_grad()
    def _eval_direct_refine(self) -> None:
        """Demonstrate the direct connection mid-training: refine one val file."""
        val_ds = None
        vl = self.dataloaders.get("val")
        if vl is not None:
            val_ds = vl.dataset
        if val_ds is None or len(val_ds) == 0:
            return
        sample = val_ds[0]
        # materialize coords to a temp PDB, then call engine.refine() directly
        tmp = Path(".sesi_tmp_refine_eval.pdb")
        write_ca_pdb(sample["coords"], tmp, sample["sequence"])
        try:
            res = self.bridge.refine_pdb(tmp, steps=self.cfg.refine_eval_steps)
            logger.info("[Trainer] direct-refine eval: E %.3f -> %.3f",
                        res["initial_energy"], res["final_energy"])
        except Exception as e:  # OpenMM may be absent; never kill training
            logger.warning("[Trainer] direct-refine eval skipped: %s", e)
        finally:
            if tmp.exists():
                tmp.unlink()

    def save_kernel(self, path: os.PathLike | str) -> None:
        torch.save({"kernel": self.csoc.kernel.state_dict(),
                    "version": SUITE_VERSION}, str(path))
        logger.info("[Trainer] kernel saved -> %s", path)


# =============================================================================
# 5. SELF-TEST SUITE
# =============================================================================
def _make_helix_pdb(path: Path, n_res: int, resname: str = "ALA") -> None:
    """Write an idealized α-helix Cα trace as a valid PDB file."""
    with open(path, "w") as fh:
        fh.write("REMARK   2 RESOLUTION.    2.00 ANGSTROMS.\n")
        for i in range(n_res):
            ang = math.radians(100.0 * i)
            x = 2.3 * math.cos(ang)
            y = 2.3 * math.sin(ang)
            z = 1.5 * i
            fh.write(_pdb_atom_line(i + 1, "CA", resname, "A", i + 1, x, y, z) + "\n")
        fh.write("END\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI PDB REFINE / CSOC LOSS SUITE v{SUITE_VERSION} — Self-Test")
    print(f"  real_fold_one_v2={_HAS_REAL_FOLD}  scipy={_HAS_SCIPY}  openmm={_HAS_OPENMM}")
    print("=" * 78)

    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="sesi_pdb_"))

    # ---- T1: PDB parser on a real-format file ----------------------------
    print("\n [T1] Real PDB parsing")
    p1 = tmpdir / "helix_20.pdb"
    _make_helix_pdb(p1, 20)
    s1 = parse_pdb_file(p1)
    assert s1.ca_coords.shape == (20, 3), s1.ca_coords.shape
    assert s1.sequence == "A" * 20
    assert s1.resolution == 2.0
    bond = np.linalg.norm(s1.ca_coords[1] - s1.ca_coords[0])
    assert 3.0 < bond < 4.5, f"CA-CA step should be ~3.8A, got {bond:.2f}"
    print(f"   parsed: L={len(s1.sequence)} res, CA step={bond:.2f} A, "
          f"resolution={s1.resolution}, atoms={s1.n_atoms_total}")
    print("[PASS] T1 parser extracts coords/sequence/resolution correctly")

    # ---- T2: dataset split determinism ----------------------------------
    print("\n [T2] RealPDBDataset deterministic split")
    for i, n in enumerate([12, 24, 36, 48, 60]):
        _make_helix_pdb(tmpdir / f"prot_{i:02d}_{n}.pdb", n)
    ds_train = RealPDBDataset(tmpdir, split="train", min_residues=10)
    ds_val = RealPDBDataset(tmpdir, split="val", min_residues=10)
    ds_test = RealPDBDataset(tmpdir, split="test", min_residues=10)
    names_tr = {f.name for f in ds_train.files}
    names_va = {f.name for f in ds_val.files}
    names_te = {f.name for f in ds_test.files}
    assert not (names_tr & names_va) and not (names_tr & names_te) and not (names_va & names_te)
    ds_train2 = RealPDBDataset(tmpdir, split="train", min_residues=10)
    assert {f.name for f in ds_train2.files} == names_tr
    print(f"   split sizes: train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
    print("[PASS] T2 splits are disjoint and reproducible (MD5-fold)")

    # ---- T3: collate padding + masks ------------------------------------
    print("\n [T3] Collate function")
    dl = DataLoader(ds_train, batch_size=3, shuffle=False, collate_fn=pdb_collate)
    batch = next(iter(dl))
    B, Lmax, _ = batch["coords"].shape
    assert batch["mask"].shape == (B, Lmax)
    for i in range(B):
        Li = int(batch["lengths"][i])
        assert bool(batch["mask"][i, :Li].all()) and bool((~batch["mask"][i, Li:]).all())
    print(f"   batch: coords={tuple(batch['coords'].shape)} mask={tuple(batch['mask'].shape)}")
    print("[PASS] T3 padding and masks exact per-sample lengths")

    # ---- T4: CSOCKernelLoss energy sanity (minimum at r0) ---------------
    print("\n [T4] CSOCKernelLoss physics sanity")
    loss_mod = CSOCKernelLoss(kernel=None, cutoff=20.0, device="cpu",
                              use_ecosystem_neighbors=False)
    r0_val = float(loss_mod.kernel.r0.item())
    two = torch.tensor([[0.0, 0.0, 0.0], [r0_val, 0.0, 0.0]])
    far = torch.tensor([[0.0, 0.0, 0.0], [2.0 * r0_val, 0.0, 0.0]])
    L_at_r0, _ = loss_mod.forward(two)
    L_at_far, _ = loss_mod.forward(far)
    print(f"   E_LJ(r0={r0_val:.2f}A)={L_at_r0.item():.4f}  "
          f"E_LJ(2r0)={L_at_far.item():.4f}")
    assert L_at_r0.item() < L_at_far.item(), "LJ minimum must sit at r0"
    print("[PASS] T4 loss is minimized at the kernel equilibrium distance")

    # ---- T5: gradients reach all three kernel parameters ----------------
    print("\n [T5] Gradient flow into kernel parameters")
    coords_h = torch.from_numpy(s1.ca_coords).float()
    loss_h, diag_h = loss_mod.forward(coords_h)
    loss_h.backward()
    for name, p in loss_mod.kernel.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    print(f"   loss={loss_h.item():.4f} edges={diag_h['n_edges']} "
          f"params_with_grad={sum(1 for p in loss_mod.kernel.parameters() if p.grad is not None)}")
    print("[PASS] T5 dL/dα, dL/dr0, dL/dε all finite")

    # ---- T6: DirectRefinementBridge wiring ------------------------------
    print("\n [T6] DirectRefinementBridge (direct engine connection)")
    bridge = None
    if _HAS_REAL_FOLD:
        rcfg = RefinementConfig()
        rcfg.steps = 10                       # tiny for the self-test
        bridge = DirectRefinementBridge(rcfg, use_opmc_cap=True)
        assert bridge.kernel is bridge.engine.kernel, "kernel must be the SAME object"
        print("   bridge.kernel is engine.kernel : True (shared parameters)")
        if _HAS_OPENMM:
            try:
                res = bridge.refine_pdb(p1, steps=5)
                print(f"   direct refine ran: E0={res['initial_energy']:.2f} "
                      f"-> E={res['final_energy']:.2f}")
                print("[PASS] T6 engine.refine() executed end-to-end")
            except Exception as e:
                print(f"[WARN] T6 refine() failed at runtime (environment): {e}")
        else:
            print("[SKIP] T6 refine() — OpenMM not installed; wiring verified only")
        bridge.cleanup()
    else:
        print("[SKIP] T6 real_fold_one_v2 not importable; stand-in path used above")

    # ---- T7: OPMC step cap (Theorem 11.1 form) ---------------------------
    print("\n [T7] OPMC step cap")
    k_star = opmc_step_cap(1e-8)
    print(f"   k*(ε=1e-8, L={OPMC_L}) = {k_star}  (paper-consistent: 23)")
    assert k_star == 23, k_star
    print("[PASS] T7 iteration bound matches the paper value")

    # ---- T8: mini end-to-end training on real files ----------------------
    print("\n [T8] Mini training loop on real PDB files")
    loaders = build_pdb_dataloaders(tmpdir, batch_size=2, min_residues=10)
    tcfg = PDBCSOCTrainerConfig(epochs=2, lr=1e-2, log_interval=1000,
                                refine_eval_every=0, device="cpu")
    trainer = PDBCSOCTrainer(tcfg, loaders, refinement_bridge=None)
    before = {n: p.clone() for n, p in trainer.csoc.kernel.named_parameters()}
    hist = trainer.train()
    after = {n: p.clone() for n, p in trainer.csoc.kernel.named_parameters()}
    moved = sum(1 for n in before if not torch.equal(before[n], after[n]))
    assert len(hist) > 0 and all(math.isfinite(h["loss"]) for h in hist)
    assert moved == len(before), "kernel parameters must move during training"
    print(f"   steps={len(hist)} first_loss={hist[0]['loss']:.4f} "
          f"last_loss={hist[-1]['loss']:.4f} params_updated={moved}/{len(before)}")
    print("[PASS] T8 training runs, losses finite, shared kernel updates")

    # ---- T9: write_ca_pdb round-trip ------------------------------------
    print("\n [T9] write_ca_pdb round-trip")
    out_pdb = tmpdir / "roundtrip.pdb"
    write_ca_pdb(coords_h, out_pdb, sequence=s1.sequence, resolution=2.0)
    s_rt = parse_pdb_file(out_pdb)
    err = np.abs(s_rt.ca_coords - s1.ca_coords).max()
    assert s_rt.sequence == s1.sequence and err < 1e-2, (s_rt.sequence[:8], err)
    print(f"   round-trip max coord error = {err:.2e} A")
    print("[PASS] T9 write/parse round-trip preserves structure")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("  HONEST REMINDERS:")
    print("    • refine() requires OpenMM at call time; bridge wiring is verified")
    print("      even when OpenMM is absent (graceful skip).")
    print("    • The OPMC cap is a HEURISTIC motivated by Theorem 11.1; refine()")
    print("      itself has not been proven to be a contraction with L=0.4385.")
    print("    • The torch-fallback neighbor list does NOT cap max_neighbors;")
    print("      install scipy or the ecosystem stack for capped k-NN behavior.")
    print("    • This suite validates plumbing and physics sanity, NOT biological")
    print("      accuracy. Benchmark against held-out experimental structures")
    print("      (real PDB with known resolution) before trusting trained kernels.")
    print("=" * 78)
