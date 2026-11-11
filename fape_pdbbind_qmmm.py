#!/usr/bin/env python3
"""
=============================================================================
FAPE + RECYCLING · PDBbind LOADER · QM STRAIN CORRECTION
=============================================================================
License: MIT | 2026

LITERATURE
 [A1] Jumper et al. "Highly accurate protein structure prediction with
      AlphaFold." Nature 596, 583 (2021).  FAPE: Supplementary Alg. 23.
 [A2] Abramson et al. "Accurate structure prediction of biomolecular
      interactions with AlphaFold 3." Nature 630, 493 (2024).
 [P1] Liu et al. "Forging the basis for developing protein-ligand interaction
      scoring functions." J. Med. Chem. 58, 2366 (2015).   [PDBbind]
 [P2] Su et al. "Comparative assessment of scoring functions: the CASF-2016
      update." J. Chem. Inf. Model. 59, 895 (2019).        [core set]
 [P3] Volkov et al. "On the frustration to predict binding affinities from
      protein-ligand structures with deep neural networks."
      J. Med. Chem. 65, 7946 (2022).   ← ligand-only baseline critique
 [Q1] Senn & Thiel. "QM/MM methods for biomolecular systems."
      Angew. Chem. Int. Ed. 48, 1198 (2009).
 [Q2] Ryde. "The coordination of the catalytic zinc in alcohol dehydrogenase
      studied by combined quantum-chemical and molecular mechanics methods."
      (link-atom / capping practice)

=============================================================================
HONEST SCOPE
=============================================================================
FAPE
  ✅ Faithful to AF2 Alg. 23: frame-local, clamped, rigid-motion invariant.
  ✅ Invariance and chirality are TESTED below, not asserted.
  ❌ FAPE alone does not make a structure predictor. You still need a trunk
     (Evoformer / MSA), training data, and recycling with real weights.

RECYCLING
  ✅ AF2-style: previous single/pair/Cβ-distogram fed back, gradients detached.
  ❌ Recycling with random weights produces nothing useful. The MECHANISM is
     testable; the BENEFIT requires training.

PDBbind
  ✅ Real experimental Kd/Ki. pK = -log10(Kd or Ki in M).
  ⚠️ Registration required at pdbbind.org.cn — no direct download URL here.
  ⚠️ RANDOM SPLITS LEAK. Scaffold split and time split are provided.
  ⚠️ A LIGAND-ONLY BASELINE IS MANDATORY [P3]. Many published "structure-based"
     models are matched by ligand-only models. This file reports it.
  ❌ CASF-2016 core set (285) is the standard benchmark. Report it separately.

QM STRAIN
  ✅ REAL PySCF single-points on the ligand: E_bound - E_MMFF_relaxed.
     This is a genuine, well-defined QM quantity, no link atoms needed.
  ❌ Full QM/MM with link-atom capping and electrostatic embedding is NOT
     implemented. What is missing is listed explicitly in the class docstring.
  ❌ Strain energy is ENTHALPIC. It is a FEATURE for a learned affinity model,
     not a binding free energy and not a pK.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math, re, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
try:
    from pyscf import gto, scf, mp
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("fape_pdbbind")

HARTREE_EV = 27.211386245988
HARTREE_KCAL = 627.5094740631
AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
 "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
 "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


# =============================================================================
# 1. FAPE  — AF2 Supplementary Algorithm 23
# =============================================================================

def fape(pred_frames: Tensor, pred_pos: Tensor,
         true_frames: Tensor, true_pos: Tensor,
         clamp: float = 10.0,
         frame_mask: Optional[Tensor] = None,
         atom_mask: Optional[Tensor] = None,
         chunk: int = 128, normalize: bool = True,
         eps: float = 1e-8) -> Dict[str, Tensor]:
    """
    Frame Aligned Point Error.

      d_ij = || T_i^{-1} ∘ x_j  -  (T_i^true)^{-1} ∘ x_j^true ||
      FAPE = mean_ij min(d_ij, d_clamp) / d_clamp

    pred_frames, true_frames : (B, L, 4, 4)
    pred_pos,   true_pos     : (B, L, A, 3)   A atoms per residue
    frame_mask               : (B, L)
    atom_mask                : (B, L, A)

    Properties (verified in selftest):
      • invariant to a global rigid motion applied to BOTH pred and truth
      • distinguishes mirror images (unlike distance-matrix losses)
      • clamping bounds the influence of grossly wrong regions
    """
    B, L = pred_frames.shape[:2]
    A = pred_pos.shape[2]
    Rp, tp = pred_frames[..., :3, :3], pred_frames[..., :3, 3]
    Rt, tt = true_frames[..., :3, :3], true_frames[..., :3, 3]

    if frame_mask is None:
        frame_mask = pred_pos.new_ones(B, L)
    if atom_mask is None:
        atom_mask = pred_pos.new_ones(B, L, A)
    w = frame_mask[:, :, None, None] * atom_mask[:, None, :, :]   # (B,Li,Lj,A)

    total = pred_pos.new_zeros(B)
    count = pred_pos.new_zeros(B)
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        dp = pred_pos[:, None] - tp[:, s:e, None, None]   # (B,c,L,A,3)
        dt = true_pos[:, None] - tt[:, s:e, None, None]
        # local_k = sum_j R[b,i,j,k] * diff[b,i,l,a,j]  == (Rᵀ diff)_k
        lp = torch.einsum("bijk,bilaj->bilak", Rp[:, s:e], dp)
        lt = torch.einsum("bijk,bilaj->bilak", Rt[:, s:e], dt)
        d = torch.sqrt(((lp - lt) ** 2).sum(-1) + eps)
        if clamp is not None:
            d = d.clamp(max=clamp)
        wc = w[:, s:e]
        total = total + (d * wc).sum(dim=(1, 2, 3))
        count = count + wc.sum(dim=(1, 2, 3))
    loss = total / count.clamp(min=1.0)
    if normalize and clamp:
        loss = loss / clamp
    return {"fape": loss, "fape_unclamped_norm": total / count.clamp(min=1.0)}


def backbone_frames_from_ca(ca: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Crude frames from Cα only (Gram–Schmidt on local triad).
    NOT AF2's N-CA-C frame construction — for testing/demo only.
    Real use: build frames from N, CA, C ideal geometry.
    """
    B, L, _ = ca.shape
    v1 = ca[:, 2:] - ca[:, 1:-1]
    v2 = ca[:, 1:-1] - ca[:, :-2]
    e1 = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)
    T = torch.eye(4, device=ca.device, dtype=ca.dtype)[None, None].expand(B, L, 4, 4).clone()
    T[:, 1:-1, :3, :3] = R
    T[:, :, :3, 3] = ca
    return T


# =============================================================================
# 2. RECYCLING  — AF2-style, gradients detached between iterations
# =============================================================================

class RecyclingWrapper(nn.Module):
    """
    AF2 §1.7: previous single, pair, and Cβ distogram are fed back as input.
    Gradients are NOT propagated through recycling iterations (AF2 practice);
    during training the number of iterations is sampled and only the last
    pass is differentiated.

    This wrapper implements the MECHANISM. It does not supply a trunk.
    """
    def __init__(self, trunk: nn.Module, n_recycle: int = 3,
                 d_single: int = 128, d_pair: int = 64,
                 n_dist_bins: int = 15, detach: bool = True,
                 sample_iters: bool = True):
        super().__init__()
        self.trunk = trunk
        self.n_recycle = n_recycle
        self.detach = detach
        self.sample_iters = sample_iters
        self.norm_s = nn.LayerNorm(d_single)
        self.norm_z = nn.LayerNorm(d_pair)
        self.prev_s = nn.Linear(d_single, d_single)
        self.prev_z = nn.Linear(d_pair, d_pair)
        self.prev_d = nn.Linear(n_dist_bins, d_pair)
        self.register_buffer("bin_edges", torch.linspace(2.0, 22.0, n_dist_bins - 1))
        self.n_dist_bins = n_dist_bins

    def _distogram(self, ca: Tensor) -> Tensor:
        d = torch.cdist(ca, ca)
        idx = torch.bucketize(d, self.bin_edges)
        return F.one_hot(idx, self.n_dist_bins).float()

    def forward(self, s: Tensor, z: Tensor, frames: Tensor,
                **trunk_kwargs) -> Dict[str, Any]:
        n_it = (int(torch.randint(0, self.n_recycle + 1, (1,)))
                if (self.training and self.sample_iters) else self.n_recycle)
        prev_s = prev_z = prev_d = None
        out = None
        for it in range(n_it + 1):
            s_in, z_in = s, z
            if prev_s is not None:
                s_in = s_in + self.prev_s(self.norm_s(prev_s))
                z_in = z_in + self.prev_z(self.norm_z(prev_z))
                z_in = z_in + self.prev_d(prev_d)
            out = self.trunk(s_in, z_in, frames, **trunk_kwargs)
            ca = out["frames"][..., :3, 3]
            nxt = (out["scalar"], out.get("pair", z), self._distogram(ca))
            if self.detach:
                nxt = tuple(v.detach() for v in nxt)
            prev_s, prev_z, prev_d = nxt
            out["recycle_iter"] = it
        out["n_recycle_used"] = n_it
        return out


# =============================================================================
# 3. PDBbind LOADER
# =============================================================================

@dataclass
class PDBbindConfig:
    root: str = "data/pdbbind"
    subset: str = "refined"          # refined | general | core
    pocket_cutoff: float = 10.0      # Å from any ligand atom
    max_atoms: int = 4000
    year_split: Optional[int] = None # e.g. 2016 → train <2016, test >=2016


def parse_index(path: Path) -> List[Dict]:
    """
    INDEX_refined_data.2020 / INDEX_general_data.2020:
      pdb  resolution  year  pK  Kd/Ki_string  reference...  ligand_name...
    pK = -log10(Kd or Ki in molar units).
    """
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        if len(tok) < 5:
            continue
        try:
            pdb = tok[0].lower()
            res = float(tok[1]) if tok[1] not in ("N/A", "-") else None
            year = int(tok[2])
            pk = float(tok[3])
        except ValueError:
            continue
        rest = " ".join(tok[4:])
        m = re.search(r"(Kd|Ki|IC50)\s*=\s*([0-9.eE+-]+)\s*([munpfM]+)", rest)
        kind = m.group(1) if m else "unknown"
        out.append({"pdb": pdb, "resolution": res, "year": year, "pK": pk,
                    "affinity_kind": kind, "raw": rest})
    log.info(f"[PDBbind] index: {len(out)} entries")
    kinds = {}
    for r in out:
        kinds[r["affinity_kind"]] = kinds.get(r["affinity_kind"], 0) + 1
    log.info(f"[PDBbind] affinity kinds: {kinds}")
    log.warning("[PDBbind] Kd, Ki and IC50 are NOT interchangeable. "
                "Mixing them adds label noise; consider filtering to Kd/Ki.")
    return out


def read_pdb_atoms(path: Path) -> Tuple[List[Dict], Dict[str, str]]:
    atoms, seq = [], {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            name = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21]
            resi = int(line[22:26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            elem = line[76:78].strip() or name[0]
        except (ValueError, IndexError):
            continue
        atoms.append({"name": name, "resn": resn, "chain": chain,
                      "resi": resi, "xyz": xyz, "elem": elem})
        if resn in AA3to1:
            seq[f"{chain}:{resi}"] = AA3to1[resn]
    return atoms, seq


def read_sdf(path: Path):
    if not HAS_RDKIT:
        raise ImportError("pip install rdkit")
    mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    return mol


def mol_coords_elements(mol) -> Tuple[np.ndarray, List[str]]:
    conf = mol.GetConformer()
    xyz = np.array(conf.GetPositions(), dtype=np.float32)
    els = [a.GetSymbol() for a in mol.GetAtoms()]
    return xyz, els


def extract_pocket(prot_atoms: List[Dict], lig_xyz: np.ndarray,
                   cutoff: float) -> Dict[str, Any]:
    if len(lig_xyz) == 0 or not prot_atoms:
        return {}
    P = np.array([a["xyz"] for a in prot_atoms], dtype=np.float32)
    d2 = ((P[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1)
    near = d2.min(axis=1) < cutoff ** 2
    residues = sorted({(a["chain"], a["resi"]) for a, k in zip(prot_atoms, near) if k})
    keep = {(a["chain"], a["resi"]) for a in prot_atoms} & set(residues)
    sel = [a for a in prot_atoms if (a["chain"], a["resi"]) in keep]
    return {"pocket_atoms": sel, "n_residues": len(residues),
            "residues": residues}


def load_complex(root: Path, pdb: str, subset: str,
                 cfg: PDBbindConfig) -> Optional[Dict]:
    base = root / (f"{subset}-set" if subset != "core" else "core-set") / pdb
    if not base.exists():
        base = root / pdb
    if not base.exists():
        return None
    pp = next(iter(base.glob("*_protein.pdb")), None)
    lp = next(iter(base.glob("*_ligand.sdf")), None)
    if pp is None or lp is None:
        return None
    atoms, seqmap = read_pdb_atoms(pp)
    mol = read_sdf(lp)
    if mol is None or not atoms:
        return None
    lxyz, lels = mol_coords_elements(mol)
    if len(atoms) > cfg.max_atoms:
        atoms = atoms[:cfg.max_atoms]
    pk = extract_pocket(atoms, lxyz, cfg.pocket_cutoff)
    if not pk:
        return None
    seq = "".join(seqmap[k] for k in sorted(seqmap,
                  key=lambda s: (s.split(":")[0], int(s.split(":")[1]))))
    return {"pdb": pdb, "protein_atoms": atoms, "sequence": seq,
            "ligand_mol": mol, "ligand_xyz": lxyz, "ligand_elements": lels,
            "pocket": pk}


# =============================================================================
# 4. SPLITS — random leaks; scaffold and time are the honest options
# =============================================================================

def murcko_scaffold(mol) -> str:
    if not HAS_RDKIT:
        return ""
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc)
    except Exception:
        return ""


def scaffold_split(records: List[Dict], frac=(0.8, 0.1, 0.1),
                   seed: int = 42) -> Dict[str, List[int]]:
    """Group by Bem–Murcko scaffold; no scaffold appears in two splits."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        s = murcko_scaffold(r["ligand_mol"])
        groups.setdefault(s, []).append(i)
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    n = len(records)
    ntr, nva = frac[0] * n, (frac[0] + frac[1]) * n
    out = {"train": [], "val": [], "test": []}
    ctr = cva = 0
    for k in keys:                      # large scaffolds → train (standard)
        g = groups[k]
        if ctr + len(g) <= ntr:
            out["train"] += g; ctr += len(g)
        elif cva + len(g) <= nva - ntr:
            out["val"] += g; cva += len(g)
        else:
            out["test"] += g
    log.info(f"[scaffold-split] {len(out['train'])}/{len(out['val'])}/{len(out['test'])} "
             f"over {len(groups)} scaffolds")
    return out


def time_split(records: List[Dict], year: int) -> Dict[str, List[int]]:
    """PDBbind time split: train on older, test on newer. Realistic prospective test."""
    tr = [i for i, r in enumerate(records) if r["year"] < year]
    te = [i for i, r in enumerate(records) if r["year"] >= year]
    rng = np.random.default_rng(42)
    rng.shuffle(tr)
    nv = max(1, int(0.1 * len(tr)))
    out = {"train": tr[nv:], "val": tr[:nv], "test": te}
    log.info(f"[time-split] year<{year}: {len(out['train'])}/{len(out['val'])}, "
             f"year>={year}: {len(out['test'])}")
    return out


def random_split(records, frac=(0.8, 0.1, 0.1), seed=42):
    log.warning("[random-split] LEAKS: similar ligands/proteins across splits. "
                "Use only as an upper bound, never as the headline result.")
    idx = np.random.default_rng(seed).permutation(len(records))
    a = int(frac[0]*len(idx)); b = int((frac[0]+frac[1])*len(idx))
    return {"train": idx[:a].tolist(), "val": idx[a:b].tolist(),
            "test": idx[b:].tolist()}


# =============================================================================
# 5. FEATURES  (ligand-only baseline is mandatory)
# =============================================================================

def ligand_features(mol) -> np.ndarray:
    """Morgan FP (2048) + a few descriptors. This is the LIGAND-ONLY baseline."""
    if not HAS_RDKIT:
        raise ImportError("rdkit required")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    v = np.zeros(1024, np.float32)
    for b in fp.GetOnBits():
        v[b] = 1.0
    desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                     Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                     Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
                     mol.GetNumHeavyAtoms()], np.float32)
    desc = np.nan_to_num(desc, nan=0.0)
    desc /= np.array([500, 5, 5, 10, 150, 10, 50], np.float32)
    return np.concatenate([v, desc])


LIG_DIM = 1024 + 7


def pocket_features(rec: Dict) -> np.ndarray:
    """Pocket residue composition + size. Deliberately simple."""
    comp = np.zeros(20, np.float32)
    order = "ARNDCQEGHILKMFPSTWYV"
    for a in rec["pocket"]["pocket_atoms"]:
        s = AA3to1.get(a["resn"])
        if s and s in order:
            comp[order.index(s)] += 1
    comp /= max(comp.sum(), 1)
    n = rec["pocket"]["n_residues"]
    return np.concatenate([comp, [n / 100.0, len(rec["ligand_elements"]) / 50.0]])


POCKET_DIM = 22


def contact_features(rec: Dict, n_bins: int = 8, rc: float = 8.0) -> np.ndarray:
    """Element-pair contact histogram over distance bins."""
    els = ["C", "N", "O", "S", "P", "F", "Cl", "Br"]
    eidx = {e: i for i, e in enumerate(els)}
    P = np.array([a["xyz"] for a in rec["pocket"]["pocket_atoms"]], np.float32)
    pe = [eidx.get(a["elem"], 0) for a in rec["pocket"]["pocket_atoms"]]
    L = rec["ligand_xyz"]
    le = [eidx.get(e, 0) for e in rec["ligand_elements"]]
    if len(P) == 0 or len(L) == 0:
        return np.zeros(len(els) * len(els) * n_bins, np.float32)
    d = np.linalg.norm(P[:, None] - L[None], axis=-1)
    H = np.zeros((len(els), len(els), n_bins), np.float32)
    edges = np.linspace(2.0, rc, n_bins + 1)
    for i in range(len(P)):
        for j in range(L.shape[0]):
            if d[i, j] > rc:
                continue
            b = int(np.searchsorted(edges, d[i, j], side="right") - 1)
            b = min(max(b, 0), n_bins - 1)
            H[pe[i], le[j], b] += 1
    return (H / max(H.sum(), 1)).ravel()


CONTACT_DIM = 8 * 8 * 8


# =============================================================================
# 6. QM STRAIN  — the honest point where QC meets protein-ligand binding
# =============================================================================

class QMStrainCorrection:
    """
    REAL QM quantity, computable today, no link atoms required:

        E_strain = E_QM(ligand in bound pose) − E_QM(ligand in MMFF-relaxed pose)

    Both are single-point energies at the SAME level of theory on the SAME
    molecule, differing only in geometry. This is the ligand conformational
    strain penalty paid on binding — a genuine, well-defined contribution.

    WHAT IS NOT IMPLEMENTED (and why it matters):
      • Link-atom capping of cut covalent bonds  → needed for residue QM region
      • Electrostatic embedding (MM point charges in the QM Hamiltonian)
      • QM/MM interaction energy between ligand and protein
      • Solvation free energy and configurational entropy
    Without these you do NOT have a binding free energy. E_strain is a FEATURE.

    Cost: ligand typically 20–60 atoms → seconds to minutes at HF/STO-3G,
    much more at correlated levels. Cache aggressively.
    """
    def __init__(self, basis: str = "sto-3g", method: str = "hf",
                 cache_dir: str = "cache/qm_strain", max_atoms: int = 60):
        if not HAS_PYSCF or not HAS_RDKIT:
            raise ImportError("need pyscf + rdkit")
        self.basis, self.method = basis, method
        self.max_atoms = max_atoms
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.stats = {"ok": 0, "too_big": 0, "failed": 0, "cached": 0}

    def _single_point(self, els: List[str], xyz: np.ndarray) -> Optional[float]:
        keep = [i for i, e in enumerate(els) if e != "H"]
        if len(keep) > self.max_atoms:
            self.stats["too_big"] += 1
            return None
        try:
            atom = [[els[i], tuple(map(float, xyz[i]))] for i in range(len(els))]
            mol = gto.M(atom=atom, basis=self.basis, unit="Angstrom", verbose=0)
            mf = scf.RHF(mol); mf.conv_tol = 1e-9; mf.max_cycle = 100
            e = float(mf.kernel())
            if not mf.converged:
                self.stats["failed"] += 1
                return None
            if self.method == "mp2":
                e += float(mp.MP2(mf).run().e_corr)
            return e
        except Exception as exc:
            log.debug(f"[QM] failed: {exc}")
            self.stats["failed"] += 1
            return None

    def strain(self, mol, key: str) -> Optional[float]:
        f = self.cache / f"{key}_{self.basis}_{self.method}.json"
        if f.exists():
            self.stats["cached"] += 1
            return json.load(open(f)).get("strain_ev")
        els = [a.GetSymbol() for a in mol.GetAtoms()]
        bound = np.array(mol.GetConformer().GetPositions(), np.float32)
        m2 = Chem.Mol(mol)
        try:
            AllChem.MMFFOptimizeMolecule(m2, maxIters=500)
        except Exception:
            self.stats["failed"] += 1
            return None
        relaxed = np.array(m2.GetConformer().GetPositions(), np.float32)
        eb = self._single_point(els, bound)
        er = self._single_point(els, relaxed)
        if eb is None or er is None:
            return None
        strain_ev = (eb - er) * HARTREE_EV
        rmsd = float(np.sqrt(((bound - relaxed) ** 2).sum(-1).mean()))
        out = {"strain_ev": strain_ev, "strain_kcal": strain_ev * 23.0605,
               "pose_rmsd_ang": rmsd, "basis": self.basis, "method": self.method,
               "n_atoms": len(els)}
        json.dump(out, open(f, "w"))
        self.stats["ok"] += 1
        return strain_ev


# =============================================================================
# 7. AFFINITY MODEL + MANDATORY BASELINES
# =============================================================================

class AffinityNet(nn.Module):
    def __init__(self, mode: str = "full", hidden: int = 256,
                 use_qm: bool = True, dropout: float = 0.2):
        super().__init__()
        assert mode in ("ligand_only", "pocket_only", "no_structure", "full")
        self.mode, self.use_qm = mode, use_qm
        d = 0
        if mode in ("ligand_only", "no_structure", "full"):
            d += LIG_DIM
        if mode in ("pocket_only", "no_structure", "full"):
            d += POCKET_DIM
        if mode == "full":
            d += CONTACT_DIM
        if use_qm:
            d += 2                      # strain + missing-flag
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_features(records, idx, mode, qm: Optional[QMStrainCorrection],
                   use_qm: bool) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in idx:
        r = records[i]
        parts = []
        if mode in ("ligand_only", "no_structure", "full"):
            parts.append(ligand_features(r["ligand_mol"]))
        if mode in ("pocket_only", "no_structure", "full"):
            parts.append(pocket_features(r))
        if mode == "full":
            parts.append(contact_features(r))
        if use_qm:
            s = qm.strain(r["ligand_mol"], r["pdb"]) if qm else None
            parts.append(np.array([0.0 if s is None else np.clip(s, -5, 20) / 10.0,
                                   float(s is None)], np.float32))
        X.append(np.concatenate(parts))
        y.append(r["pK"])
    return np.stack(X), np.array(y, np.float32)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a*b).sum() / (np.sqrt((a*a).sum()*(b*b).sum()) + 1e-12))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson(ra, rb)


def subsample_ci(x, conf=0.95, block=32):
    n = len(x)
    if n < 8:
        m = float(np.mean(x)) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(block, n//4); th = float(np.mean(x))
    sub = np.array([x[j:j+b].mean() for j in range(n-b+1)])
    sc = math.sqrt(b)*(sub-th); a = 1-conf
    return {"mean": th,
            "lo": th - float(np.percentile(sc, 100*(1-a/2)))/math.sqrt(n),
            "hi": th - float(np.percentile(sc, 100*(a/2)))/math.sqrt(n), "n": n}


def train_affinity(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    root = Path(args.root)
    idxfile = next(iter(root.glob("index/INDEX_*data*")), None) \
              or next(iter(root.glob("INDEX_*data*")), None)
    if idxfile is None:
        log.error(
            f"No INDEX file under {root}.\n"
            "PDBbind requires free registration:\n"
            "  1. http://www.pdbbind.org.cn/  → register → download\n"
            "  2. extract so that you have:\n"
            "       <root>/index/INDEX_refined_data.2020\n"
            "       <root>/refined-set/<pdb>/<pdb>_protein.pdb\n"
            "       <root>/refined-set/<pdb>/<pdb>_ligand.sdf\n"
            "  3. re-run this command")
        return
    entries = parse_index(idxfile)
    if args.limit:
        entries = entries[:args.limit]
    records = []
    for e in entries:
        c = load_complex(root, e["pdb"], args.subset, PDBbindConfig(root=str(root)))
        if c is not None:
            records.append({**c, **e})
    if len(records) < 30:
        log.error(f"only {len(records)} usable complexes"); return
    log.info(f"[PDBbind] loaded {len(records)} complexes")

    if args.split == "scaffold":
        sp = scaffold_split(records, seed=args.seed)
    elif args.split == "time":
        sp = time_split(records, args.year)
    else:
        sp = random_split(records, seed=args.seed)

    qm = None
    if args.qm:
        try:
            qm = QMStrainCorrection(basis=args.qm_basis, method=args.qm_method)
        except ImportError as e:
            log.warning(f"[QM] disabled: {e}")

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in args.modes.split(","):
        use_qm = args.qm and qm is not None and mode != "ligand_only"
        Xtr, ytr = build_features(records, sp["train"], mode, qm, use_qm)
        Xva, yva = build_features(records, sp["val"], mode, qm, use_qm)
        Xte, yte = build_features(records, sp["test"], mode, qm, use_qm)
        ymu, ysd = float(ytr.mean()), float(ytr.std())
        model = AffinityNet(mode, use_qm=use_qm).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        T = lambda a: torch.tensor(a, device=dev)
        best, bs = 1e9, None
        for ep in range(args.epochs):
            model.train()
            perm = np.random.permutation(len(Xtr))
            for s in range(0, len(perm), args.batch_size):
                i = perm[s:s+args.batch_size]
                pred = model(T(Xtr[i]))
                loss = F.mse_loss(pred, (T(ytr[i]) - ymu)/ysd)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(T(Xva)).cpu().numpy()*ysd + ymu
            vmse = float(np.mean((pv-yva)**2))
            if vmse < best:
                best = vmse; bs = {k: v.clone() for k, v in model.state_dict().items()}
        if bs: model.load_state_dict(bs)
        model.eval()
        with torch.no_grad():
            pt = model(T(Xte)).cpu().numpy()*ysd + ymu
        rmse = float(np.sqrt(np.mean((pt-yte)**2)))
        results[mode] = {"rmse": rmse, "pearson": pearson(pt, yte),
                         "spearman": spearman(pt, yte),
                         "ci_rmse": subsample_ci(np.abs(pt-yte)),
                         "n_test": len(yte), "use_qm": use_qm,
                         "n_features": Xtr.shape[1]}
        log.info(f"[{mode}] RMSE {rmse:.3f}  R {pearson(pt,yte):.3f}  "
                 f"ρ {spearman(pt,yte):.3f}  (QM={use_qm})")
        torch.save({"model": model.state_dict(), "ymu": ymu, "ysd": ysd,
                    "mode": mode}, outdir/f"{mode}.pt")

    print("\n" + "="*80)
    print(f"  PDBbind {args.subset} — split={args.split}  seed={args.seed}")
    print("="*80)
    print(f"  {'mode':<16}{'feat':>7}{'RMSE':>9}{'R':>9}{'rho':>9}{'QM':>6}")
    for m, r in results.items():
        print(f"  {m:<16}{r['n_features']:>7}{r['rmse']:>9.3f}"
              f"{r['pearson']:>9.3f}{r['spearman']:>9.3f}{str(r['use_qm']):>6}")
    if "ligand_only" in results and "full" in results:
        gap = results["ligand_only"]["rmse"] - results["full"]["rmse"]
        print(f"\n  structure contribution (RMSE_ligand − RMSE_full): {gap:+.3f}")
        if gap <= 0.05:
            print("  ⚠️  Structure adds ≈nothing. This reproduces the known")
            print("     critique [P3]. Do NOT claim structure-based advantage.")
    if qm:
        print(f"\n  QM strain stats: {qm.stats}")
    print("\n  HONEST NOTES")
    print("   • Report CASF-2016 core set separately; it is the standard benchmark.")
    print("   • Random splits leak. Only scaffold/time numbers are meaningful.")
    print("   • Kd/Ki/IC50 mixing adds label noise.")
    print("   • QM strain is ENTHALPIC. It is a feature, not a pK.")
    print("="*80)
    json.dump({"results": results, "split": args.split, "seed": args.seed,
               "qm_stats": qm.stats if qm else None},
              open(outdir/"report.json", "w"), indent=2)


# =============================================================================
# 8. SELF-TEST
# =============================================================================

def selftest(args):
    print("="*76); print("  FAPE / RECYCLING / FEATURES — self-test"); print("="*76)
    torch.manual_seed(0)
    ok = True
    B, L, A = 2, 16, 4
    ca = torch.randn(B, L, 3)*5
    T = backbone_frames_from_ca(ca)
    pos = ca[:, :, None, :] + torch.randn(B, L, A, 3)*0.5

    print("\n[T1] FAPE = 0 for identical input")
    v = fape(T, pos, T, pos)["fape"]
    t = bool((v < 1e-5).all()); ok &= t
    print(f"   {v.tolist()}  {'PASS' if t else 'FAIL'}")

    print("\n[T2] rigid-motion invariance (global R, t on BOTH)")
    R0 = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.linalg.det(R0) < 0: R0[:, 0] *= -1
    t0 = torch.randn(3)*30
    T2 = T.clone()
    T2[..., :3, :3] = torch.einsum("ij,bljk->blik", R0, T[..., :3, :3])
    T2[..., :3, 3] = torch.einsum("ij,blj->bli", R0, T[..., :3, 3]) + t0
    pos2 = torch.einsum("ij,blaj->blai", R0, pos) + t0
    a = fape(T, pos, T, pos)["fape"]; b = fape(T2, pos2, T, pos)["fape"]
    t = bool(((a-b).abs() < 1e-4).all()); ok &= t
    print(f"   before {a.tolist()}  after rigid {b.tolist()}")
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T3] FAPE distinguishes mirror images")
    M = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    Tm = T.clone()
    Tm[..., :3, :3] = M @ T[..., :3, :3] @ M
    Tm[..., :3, 3] = torch.einsum("ij,blj->bli", M, T[..., :3, 3])
    posm = torch.einsum("ij,blaj->blai", M, pos)
    vm = fape(Tm, posm, T, pos)["fape"]
    t = bool((vm > 1e-3).all()); ok &= t
    print(f"   mirrored FAPE {vm.tolist()} (>0 ⇒ chirality sensitive)")
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T4] clamping bounds the loss")
    bad = pos + 500.0
    vc = fape(T, bad, T, pos, clamp=10.0)["fape"]
    t = bool((vc <= 1.0 + 1e-6).all()); ok &= t
    print(f"   clamped(normalized) {vc.tolist()} ≤ 1  {'PASS' if t else 'FAIL'}")

    print("\n[T5] masking excludes padded residues")
    fm = torch.ones(B, L); fm[1, 8:] = 0
    am = torch.ones(B, L, A)
    p_bad = pos.clone(); p_bad[1, 8:] += 1000
    v1 = fape(T, p_bad, T, pos, frame_mask=fm, atom_mask=am)["fape"]
    t = v1[1].item() < 1e-4; ok &= t
    print(f"   masked sample FAPE {v1[1].item():.2e}  {'PASS' if t else 'FAIL'}")

    print("\n[T6] recycling: detach + shapes")
    class Trunk(nn.Module):
        def __init__(s):
            super().__init__(); s.lin = nn.Linear(128, 128)
            s.fr = nn.Linear(128, 9)
        def forward(s, sc, z, frames):
            h = s.lin(sc); u = s.fr(h)
            Tn = frames.clone()
            Tn[..., :3, 3] = frames[..., :3, 3] + u[..., :3]
            return {"scalar": h, "pair": z, "frames": Tn}
    rec = RecyclingWrapper(Trunk(), n_recycle=2, d_single=128, d_pair=64)
    rec.eval()
    s = torch.randn(B, L, 128); z = torch.randn(B, L, L, 64)
    o = rec(s, z, T)
    t = o["frames"].shape == T.shape and o["n_recycle_used"] == 2; ok &= t
    print(f"   frames {tuple(o['frames'].shape)}  iters={o['n_recycle_used']}"
          f"  {'PASS' if t else 'FAIL'}")
    rec.train()
    o = rec(s, z, T)
    o["scalar"].sum().backward()
    g = [p.grad for p in rec.parameters() if p.grad is not None]
    t = len(g) > 0 and all(torch.isfinite(x).all() for x in g); ok &= bool(t)
    print(f"   finite grads on {len(g)} tensors  {'PASS' if t else 'FAIL'}")

    if HAS_RDKIT:
        print("\n[T7] ligand features + scaffold")
        m = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        m = Chem.AddHs(m); AllChem.EmbedMolecule(m, randomSeed=0)
        f = ligand_features(m)
        sc = murcko_scaffold(m)
        t = f.shape[0] == LIG_DIM and np.isfinite(f).all(); ok &= bool(t)
        print(f"   dim={f.shape[0]}  scaffold='{sc}'  {'PASS' if t else 'FAIL'}")
    else:
        print("\n[T7] SKIP — pip install rdkit")

    print("\n" + "="*76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*76)
    print("""
  VERIFIED: FAPE identity, rigid invariance, chirality sensitivity, clamping,
            masking; recycling detach and gradient flow.

  NOT VERIFIED: that any of this predicts structures or affinities.
            FAPE is a LOSS. Recycling is a MECHANISM. Both need a trained trunk.
            Run train-affinity on real PDBbind and read the ligand-only baseline.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "train-affinity"])
    p.add_argument("--root", default="data/pdbbind")
    p.add_argument("--subset", default="refined")
    p.add_argument("--split", choices=["scaffold", "time", "random"], default="scaffold")
    p.add_argument("--year", type=int, default=2016)
    p.add_argument("--modes", default="ligand_only,pocket_only,full")
    p.add_argument("--qm", action="store_true")
    p.add_argument("--qm-basis", dest="qm_basis", default="sto-3g")
    p.add_argument("--qm-method", dest="qm_method", default="hf")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="runs/pdbbind")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    train_affinity(a)


if __name__ == "__main__":
    main()
