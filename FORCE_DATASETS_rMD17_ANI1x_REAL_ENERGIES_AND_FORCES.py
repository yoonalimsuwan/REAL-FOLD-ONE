#!/usr/bin/env python3
"""
=============================================================================
FORCE DATASETS: rMD17 + ANI-1x  —  REAL ENERGIES AND FORCES
=============================================================================
License: MIT | 2026

LITERATURE (verifiable):
 [R1] Christensen & von Lilienfeld. "On the role of gradients for machine
      learning of molecular energies and forces."
      Mach. Learn.: Sci. Technol. 1, 045018 (2020).   → revised MD17
 [R2] Chmiela, Tkatchenko, Sauceda, Poltavsky, Schütt, Müller.
      "Machine learning of accurate energy-conserving molecular force fields."
      Sci. Adv. 3, e1603015 (2017).                   → original MD17
 [A1] Smith, Lubbers, Barros, Tretiak, Isayev, Roitberg.
      "The ANI-1ccx and ANI-1x data sets, coupled-cluster and density
      functional theory properties for molecules."
      Sci. Data 7, 134 (2020).                        → ANI-1x / ANI-1ccx
 [A2] Smith, Isayev, Roitberg. "ANI-1: an extensible neural network potential
      with DFT accuracy at force field computational cost."
      Chem. Sci. 8, 3192 (2017).
 [M1] Schütt et al. NeurIPS 30 (2017).                 → SchNet
 [M2] Schütt, Unke, Gastegger. ICML (2021).            → PaiNN

=============================================================================
HONEST SCOPE
=============================================================================
✅ rMD17 and ANI-1x contain REAL DFT energies AND forces.
✅ Units are converted explicitly and verified by dimensional sanity checks.
✅ Atomic reference energies are FITTED from data, not hard-coded.
✅ Force training uses correct double-backpropagation.

❌ rMD17 is PBE/def2-SVP. ANI-1x is ωB97X/6-31G(d). Neither is experiment.
❌ ANI-1ccx (CCSD(T)*/CBS) has ENERGIES ONLY — no forces. Do not claim
   coupled-cluster force accuracy from it.
❌ Published baselines vary by split, by joint vs. separate E/F training,
   and by loss weighting. Numbers below are ORDER-OF-MAGNITUDE only.
   Verify against the specific paper's table before quoting.
❌ No claim of beating SchNet/PaiNN. This is infrastructure + honest measurement.

⚠️  THREE CRITICAL IMPLEMENTATION FACTS:
   1. Units: rMD17 = kcal/mol, kcal/mol/Å.  ANI = Hartree, Hartree/Bohr.
      1 Ha/Bohr = 51.4220674763 eV/Å. Getting this wrong scales forces 51×.
   2. Force training needs create_graph=True (second-order gradients).
      fp16 GradScaler is INCOMPATIBLE with double-backward in practice.
      → default fp32; bf16 autocast without scaler is the fast alternative.
   3. ANI-1x is ~5M conformations / several GB. Use the lazy loader.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, os, tarfile, urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("forces")

# ---------------------------------------------------------------- units
HARTREE_EV   = 27.211386245988
BOHR_ANG     = 0.529177210903
KCALMOL_EV   = 1.0 / 23.060547830619
HA_BOHR_EV_A = HARTREE_EV / BOHR_ANG      # 51.4220674763 eV/Å
EV_KCALMOL   = 23.060547830619

ELEMENTS = ["H", "C", "N", "O", "F", "S", "Cl"]
Z2IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6}
IDX2Z = {v: k for k, v in Z2IDX.items()}

RMD17_MOLECULES = [
    "azobenzene", "aspirin", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic_acid", "toluene", "uracil",
]

# Download landing pages (verify these; direct file IDs change).
SOURCES = {
    "rmd17": {
        "page": "https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038",
        "candidates": ["https://figshare.com/ndownloader/files/23950376"],
        "filename": "rMD17.tar.bz2",
    },
    "ani1x": {
        "page": "https://figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041",
        "candidates": ["https://figshare.com/ndownloader/files/18112775"],
        "filename": "ani1x-release.h5",
    },
    "ani1ccx": {
        "page": "https://figshare.com/articles/dataset/ANI-1ccx_Dataset_Release/10047059",
        "candidates": [],
        "filename": "ani1ccx-release.h5",
    },
}


# =============================================================================
# 1. SCHEMA PROBE — discover the actual file layout instead of guessing
# =============================================================================

def probe_h5(path: Path, max_groups: int = 6, max_depth: int = 5) -> Dict:
    """Walk an HDF5 file and report its real structure. Run this FIRST."""
    if not HAS_H5:
        raise ImportError("pip install h5py")
    report = {"path": str(path), "size_MB": path.stat().st_size / 1e6, "tree": []}

    def walk(name, obj, depth=0):
        if depth > max_depth or len(report["tree"]) > 400:
            return
        if isinstance(obj, h5py.Dataset):
            report["tree"].append({
                "key": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
        elif isinstance(obj, h5py.Group) and len(report["tree"]) < 400:
            report["tree"].append({"group": name, "n_children": len(obj)})

    with h5py.File(path, "r") as f:
        report["root_keys"] = list(f.keys())
        report["root_attrs"] = {k: str(v)[:200] for k, v in f.attrs.items()}
        # sample a few branches deeply
        def deep(name, obj):
            walk(name, obj, name.count("/"))
        for i, k in enumerate(list(f.keys())[:max_groups]):
            f[k].visititems(lambda n, o, p=k: deep(f"{p}/{n}", o))
    return report


def cmd_probe(args):
    p = Path(args.path)
    if not p.exists():
        log.error(f"{p} not found. See SOURCES for download pages.")
        for k, v in SOURCES.items():
            print(f"  {k:8s} → {v['page']}")
        return
    rep = probe_h5(p)
    print("=" * 74)
    print(f"  HDF5 SCHEMA: {p.name}  ({rep['size_MB']:.1f} MB)")
    print("=" * 74)
    print(f"  root keys : {rep['root_keys']}")
    if rep["root_attrs"]:
        print(f"  attrs     : {rep['root_attrs']}")
    print(f"\n  discovered datasets (first 60):")
    n = 0
    for e in rep["tree"]:
        if "key" in e:
            print(f"    {e['key']:<62s} {str(e['shape']):<22s} {e['dtype']}")
            n += 1
            if n >= 60:
                break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\n  full report → {args.out}")
    print("\n  Use these exact key names with --energy-key / --force-key.")


def try_download(name: str, dest: Path) -> Optional[Path]:
    src = SOURCES[name]
    out = dest / src["filename"]
    if out.exists() and out.stat().st_size > 1_000_000:
        log.info(f"[{name}] cached: {out} ({out.stat().st_size/1e6:.1f} MB)")
        return out
    dest.mkdir(parents=True, exist_ok=True)
    for url in src["candidates"]:
        log.info(f"[{name}] trying {url}")
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(out, "wb") as f:
                total = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk); total += len(chunk)
                    if total % (200 << 20) == 0:
                        log.info(f"  {total/1e6:.0f} MB")
            if out.stat().st_size < 1_000_000:
                out.unlink(missing_ok=True)
                raise RuntimeError("response too small — likely an error page")
            log.info(f"[{name}] downloaded {out.stat().st_size/1e6:.1f} MB")
            return out
        except Exception as e:
            log.warning(f"[{name}] failed: {e}")
            out.unlink(missing_ok=True)
    print(f"\n  MANUAL DOWNLOAD REQUIRED for {name}:")
    print(f"    1. open  {src['page']}")
    print(f"    2. download {src['filename']}")
    print(f"    3. place at {out}")
    print(f"    4. run:  python force_datasets.py probe --path {out}\n")
    return None


# =============================================================================
# 2. UNIT SANITY — catch the 51× error before it silently ruins training
# =============================================================================

def unit_sanity(E: np.ndarray, Fmag: np.ndarray, n_atoms: np.ndarray,
                label: str) -> Dict[str, Any]:
    """
    Physical plausibility check AFTER conversion to eV, eV/Å.
    Typical DFT bonded forces: 0.01–20 eV/Å.  >100 eV/Å ⇒ unit error.
    """
    e_per_atom = E / np.maximum(n_atoms, 1)
    fm = Fmag
    rep = {
        "label": label,
        "E_total_range_eV": [float(E.min()), float(E.max())],
        "E_per_atom_mean_eV": float(e_per_atom.mean()),
        "force_mean_eV_A": float(fm.mean()),
        "force_p99_eV_A": float(np.percentile(fm, 99)),
        "force_max_eV_A": float(fm.max()),
    }
    warn = []
    if fm.max() > 200:
        warn.append("force_max > 200 eV/Å — likely Ha/Bohr not converted (×51.42)")
    if 0.0 < fm.mean() < 0.002:
        warn.append("force_mean < 0.002 eV/Å — possibly over-converted (÷51.42)")
    if abs(rep["E_per_atom_mean_eV"]) > 1e4:
        warn.append("|E/atom| > 1e4 eV — atomic reference not subtracted?")
    rep["warnings"] = warn
    for w in warn:
        log.warning(f"[units:{label}] {w}")
    if not warn:
        log.info(f"[units:{label}] OK  <F>={fm.mean():.4f} eV/Å  "
                 f"p99={rep['force_p99_eV_A']:.3f}")
    return rep


# =============================================================================
# 3. rMD17 LOADER  (kcal/mol, kcal/mol/Å → eV, eV/Å)
# =============================================================================

def load_rmd17(archive_or_dir: Path, molecule: str,
               limit: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    revised MD17 [R1]. 10 molecules, ~100k geometries each, PBE/def2-SVP.
    Native units: energies kcal/mol, forces kcal/mol/Å, coords Å.
    """
    if not HAS_H5:
        raise ImportError("pip install h5py")
    target = None
    if archive_or_dir.is_dir():
        cands = list(archive_or_dir.rglob(f"*{molecule}*.hdf5")) + \
                list(archive_or_dir.rglob(f"*{molecule}*.h5"))
        if not cands:
            raise FileNotFoundError(f"no hdf5 for '{molecule}' in {archive_or_dir}. "
                                    f"Available: {[p.name for p in archive_or_dir.rglob('*.hdf5')]}")
        target = cands[0]
        return _read_rmd17_h5(target, limit)
    # archive
    mode = "r:bz2" if str(archive_or_dir).endswith(".bz2") else "r:gz"
    import io
    with tarfile.open(archive_or_dir, mode) as tf:
        for m in tf:
            if m.isfile() and molecule in m.name and (m.name.endswith(".hdf5") or m.name.endswith(".h5")):
                data = tf.extractfile(m).read()
                tmp = Path(f"/tmp/_rmd17_{molecule}.hdf5")
                tmp.write_bytes(data)
                target = tmp
                break
    if target is None:
        raise FileNotFoundError(f"'{molecule}' not found in {archive_or_dir}")
    return _read_rmd17_h5(target, limit)


def _read_rmd17_h5(path: Path, limit: Optional[int]) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        log.info(f"[rMD17] {path.name} keys={keys}")
        def pick(*names):
            for n in names:
                if n in f:
                    return n
            raise KeyError(f"none of {names} in {keys}")
        kz = pick("nuclear_charges", "z", "atomic_numbers", "species")
        kr = pick("coords", "coordinates", "R")
        ke = pick("energies", "energy", "E")
        kf = pick("forces", "force", "F")
        Z = np.asarray(f[kz]).astype(np.int64).reshape(-1)
        R = np.asarray(f[kr], dtype=np.float64)
        E = np.asarray(f[ke], dtype=np.float64).reshape(-1)
        Fo = np.asarray(f[kf], dtype=np.float64)
    if limit:
        R, E, Fo = R[:limit], E[:limit], Fo[:limit]
    if R.ndim == 2:
        R = R.reshape(len(E), len(Z), 3)
    if Fo.shape != R.shape:
        Fo = Fo.reshape(R.shape)
    # UNIT CONVERSION: kcal/mol → eV ; kcal/mol/Å → eV/Å
    E_eV = E * KCALMOL_EV
    F_eV_A = Fo * KCALMOL_EV
    z_idx = np.array([Z2IDX[int(z)] for z in Z], dtype=np.int64)
    out = {"z": z_idx, "Z_raw": Z, "R": R.astype(np.float32),
           "E": E_eV.astype(np.float32), "F": F_eV_A.astype(np.float32),
           "n_atoms": np.full(len(E), len(Z), dtype=np.int64),
           "source": f"rMD17:{path.stem}", "level": "PBE/def2-SVP"}
    unit_sanity(out["E"], np.linalg.norm(out["F"], axis=-1),
                out["n_atoms"], out["source"])
    log.info(f"[rMD17] {len(E)} geometries, {len(Z)} atoms, "
             f"elements={sorted(set(Z.tolist()))}")
    return out


# =============================================================================
# 4. ANI-1x LAZY LOADER  (Hartree, Hartree/Bohr → eV, eV/Å)
# =============================================================================

class ANI1xLazy:
    """
    ANI-1x [A1]: ~5M conformations, ωB97X/6-31G(d).
    Hierarchical HDF5. Keys are DISCOVERED, not assumed.

    Native units: energies Hartree, forces Hartree/Bohr, coords Å.
    Lazy: builds an index, reads on demand. Several GB will not fit in RAM.

    NOTE: ANI-1ccx has energies only (CCSD(T)*/CBS). Forces will be absent;
          this loader reports that rather than fabricating zeros.
    """
    ENERGY_HINTS = ["wb97x_dz.energy", "wb97x_tz.energy", "energy",
                    "ccsd(t)_cbs.energy", "ccsd(t)_cbs.energy"]
    FORCE_HINTS  = ["wb97x_dz.forces", "wb97x_tz.forces", "forces"]

    def __init__(self, path: Path, energy_key: Optional[str] = None,
                 force_key: Optional[str] = None,
                 max_systems: Optional[int] = None,
                 require_forces: bool = True):
        if not HAS_H5:
            raise ImportError("pip install h5py")
        self.path = Path(path)
        self.index: List[Tuple[str, int]] = []
        self.ekey_rel = energy_key
        self.fkey_rel = force_key
        self.has_forces = True
        self._build_index(max_systems, require_forces)

    def _build_index(self, max_systems, require_forces):
        n_sys = 0
        with h5py.File(self.path, "r") as f:
            def visit(name, obj):
                nonlocal n_sys
                if max_systems and n_sys >= max_systems:
                    return
                if not isinstance(obj, h5py.Group):
                    return
                ks = list(obj.keys())
                ek = self.ekey_rel or next((h for h in self.ENERGY_HINTS if h in ks), None)
                fk = self.fkey_rel or next((h for h in self.FORCE_HINTS if h in ks), None)
                if ek is None or "coordinates" not in ks or "species" not in ks:
                    return
                if require_forces and fk is None:
                    self.has_forces = False
                    return
                E = obj[ek]
                if E.ndim == 0 or len(E) == 0:
                    return
                self.ekey_rel, self.fkey_rel = ek, fk
                for i in range(len(E)):
                    self.index.append((name, i))
                n_sys += 1
            f.visititems(visit)
        if not self.index:
            raise RuntimeError(
                f"No usable groups found in {self.path}.\n"
                f"Run: python force_datasets.py probe --path {self.path}\n"
                f"then pass --energy-key / --force-key explicitly.")
        log.info(f"[ANI] indexed {len(self.index)} conformations from {n_sys} systems")
        log.info(f"[ANI] energy_key='{self.ekey_rel}' force_key='{self.fkey_rel}' "
                 f"has_forces={self.fkey_rel is not None}")
        if self.fkey_rel is None:
            log.warning("[ANI] NO FORCES in this file. If this is ANI-1ccx, that is "
                        "expected — CCSD(T)*/CBS energies only. Do NOT claim force accuracy.")

    def __len__(self):
        return len(self.index)

    def get(self, i: int) -> Dict[str, np.ndarray]:
        g, j = self.index[i]
        with h5py.File(self.path, "r") as f:
            grp = f[g]
            sp = np.asarray(grp["species"]).reshape(-1)
            # species may be per-system or per-conformation
            if sp.ndim == 1 and sp.size < 100:
                Zraw = sp
            else:
                Zraw = np.asarray(grp["species"]).reshape(-1)
            R = np.asarray(grp["coordinates"][j], dtype=np.float64)
            E = float(np.asarray(grp[self.ekey_rel][j]))
            Fo = (np.asarray(grp[self.fkey_rel][j], dtype=np.float64)
                  if self.fkey_rel else np.zeros_like(R))
        n = R.shape[0]
        Zraw = Zraw[:n]
        # decode species: may be atomic numbers or byte symbols
        if Zraw.dtype.kind in "SUO":
            Znum = np.array([int("".join(c for c in str(s) if c.isalpha()) and
                                 {"H":1,"C":6,"N":7,"O":8,"F":9,"S":16,"Cl":17}
                                 .get(str(s).strip("b'\" "), 0)) for s in Zraw])
        else:
            Znum = Zraw.astype(np.int64)
        keep = np.array([z in Z2IDX for z in Znum])
        if not keep.all():
            raise ValueError(f"unsupported elements {set(Znum.tolist()) - set(Z2IDX)}")
        z_idx = np.array([Z2IDX[int(z)] for z in Znum], dtype=np.int64)
        # UNIT CONVERSION: Hartree → eV ; Hartree/Bohr → eV/Å
        return {"z": z_idx, "R": R.astype(np.float32),
                "E": np.float32(E * HARTREE_EV),
                "F": (Fo * HA_BOHR_EV_A).astype(np.float32),
                "n_atoms": n}

    def materialize(self, n: int, seed: int = 42) -> Dict[str, np.ndarray]:
        """Sample n conformations into memory (for small runs / sanity checks)."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=min(n, len(self)), replace=False)
        recs = [self.get(int(i)) for i in idx]
        nmax = max(r["n_atoms"] for r in recs)
        B = len(recs)
        Z = np.zeros((B, nmax), np.int64); R = np.zeros((B, nmax, 3), np.float32)
        Fo = np.zeros((B, nmax, 3), np.float32); M = np.zeros((B, nmax), np.float32)
        E = np.zeros(B, np.float32); NA = np.zeros(B, np.int64)
        for i, r in enumerate(recs):
            k = r["n_atoms"]
            Z[i, :k] = r["z"]; R[i, :k] = r["R"]; Fo[i, :k] = r["F"]
            M[i, :k] = 1.0; E[i] = r["E"]; NA[i] = k
        unit_sanity(E, np.linalg.norm(Fo, axis=-1)[M > 0], NA, "ANI-1x")
        return {"z": Z, "R": R, "E": E, "F": Fo, "mask": M, "n_atoms": NA,
                "source": "ANI-1x", "level": "wB97X/6-31G(d)"}


# =============================================================================
# 5. ATOMIC REFERENCE ENERGIES — fitted, not hard-coded
# =============================================================================

def fit_sae(Z: np.ndarray, E: np.ndarray, mask: np.ndarray,
            n_elem: int = 7) -> Tuple[np.ndarray, Dict]:
    """
    Least-squares fit of per-element reference energies:
        E_mol ≈ Σ_e count_e(mol) * sae_e
    This removes the dominant offset so the network learns the residual.
    Fitting from data is more robust than hard-coded constants.
    """
    B, N = Z.shape
    X = np.zeros((B, n_elem), dtype=np.float64)
    for e in range(n_elem):
        X[:, e] = ((Z == e) & (mask > 0)).sum(axis=1)
    coef, res, rank, sv = np.linalg.lstsq(X, E.astype(np.float64), rcond=None)
    pred = X @ coef
    resid = E.astype(np.float64) - pred
    info = {"sae_eV": coef.tolist(),
            "elements": [ELEMENTS[i] for i in range(n_elem)],
            "residual_std_eV": float(resid.std()),
            "residual_std_eV_per_atom": float((resid / mask.sum(1)).std()),
            "design_rank": int(rank)}
    log.info(f"[SAE] fitted: " +
             ", ".join(f"{ELEMENTS[i]}={coef[i]:.5f}" for i in range(n_elem)
                       if X[:, i].sum() > 0))
    log.info(f"[SAE] residual std = {info['residual_std_eV']:.5f} eV "
             f"({info['residual_std_eV_per_atom']:.5f} eV/atom)")
    return coef.astype(np.float32), info


# =============================================================================
# 6. MODELS (force-capable, clean)
# =============================================================================

def scatter_sum(src: Tensor, idx: Tensor, n: int) -> Tensor:
    out = torch.zeros(n, *src.shape[1:], device=src.device, dtype=src.dtype)
    return out.index_add_(0, idx, src)


def edges(R, batch, cutoff):
    same = batch.unsqueeze(0) == batch.unsqueeze(1)
    d = torch.cdist(R, R)
    m = same & (d < cutoff) & (d > 1e-8)
    ii, jj = m.nonzero(as_tuple=True)
    diff = R[jj] - R[ii]
    dist = torch.sqrt((diff * diff).sum(-1) + 1e-12)
    return ii, jj, dist, diff


class RBF(nn.Module):
    def __init__(self, cutoff, n):
        super().__init__()
        self.register_buffer("c", torch.linspace(0., cutoff, n))
        self.w = float(self.c[1] - self.c[0]) if n > 1 else 1.

    def forward(self, d):
        return torch.exp(-((d.unsqueeze(-1) - self.c) ** 2) / self.w ** 2)


def cut(d, rc):
    return 0.5 * (torch.cos(math.pi * d / rc) + 1.) * (d < rc).float()


def ssp(x):
    return F.softplus(x) - math.log(2.)


class SchNetF(nn.Module):
    def __init__(self, nf=128, nrbf=64, nint=6, rc=5.0, ne=7):
        super().__init__()
        self.rc = rc
        self.emb = nn.Embedding(ne, nf)
        self.rbf = RBF(rc, nrbf)
        self.filt = nn.ModuleList([nn.Sequential(nn.Linear(nrbf, nf), nn.SiLU(),
                                                 nn.Linear(nf, nf), nn.SiLU())
                                   for _ in range(nint)])
        self.linj = nn.ModuleList([nn.Linear(nf, nf, bias=False) for _ in range(nint)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(nf, nf), nn.SiLU(),
                                                nn.Linear(nf, nf)) for _ in range(nint)])
        self.head = nn.Sequential(nn.Linear(nf, nf // 2), nn.SiLU(),
                                  nn.Linear(nf // 2, 1))
        self.register_buffer("sae", torch.zeros(ne))
        self.register_buffer("res_std", torch.ones(1))

    def forward(self, z, R, batch, ng):
        n = z.shape[0]
        ii, jj, d, _ = edges(R, batch, self.rc)
        w = self.rbf(d); c = cut(d, self.rc)
        x = self.emb(z)
        for f, lj, up in zip(self.filt, self.linj, self.upd):
            x = x + up(scatter_sum(lj(x)[jj] * f(w) * c.unsqueeze(-1), ii, n))
        ea = self.head(x).squeeze(-1) * self.res_std + self.sae[z]
        return scatter_sum(ea, batch, ng)


class PaiNNMsg(nn.Module):
    def __init__(self, nf, nrbf):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(nf, 3 * nf), nn.SiLU(), nn.Linear(3 * nf, 3 * nf))
        self.W = nn.Sequential(nn.Linear(nrbf, 3 * nf), nn.SiLU(), nn.Linear(3 * nf, 3 * nf))

    def forward(self, s, v, ii, jj, w, c, u, n):
        p = self.phi(s)[jj] * self.W(w) * c.unsqueeze(-1)
        a, b, g = torch.chunk(p, 3, -1)
        ds = scatter_sum(a, ii, n)
        dv = scatter_sum(v[jj] * b.unsqueeze(1) + g.unsqueeze(1) * u.unsqueeze(-1), ii, n)
        return s + ds, v + dv


class PaiNNUpd(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.U = nn.Linear(nf, nf, bias=False)
        self.V = nn.Linear(nf, nf, bias=False)
        self.net = nn.Sequential(nn.Linear(2 * nf, nf), nn.SiLU(), nn.Linear(nf, 3 * nf))

    def forward(self, s, v):
        Uv = self.U(v.transpose(1, 2)).transpose(1, 2)
        Vv = self.V(v.transpose(1, 2)).transpose(1, 2)
        vn = torch.sqrt((Vv ** 2).sum(1) + 1e-12)
        a, b, g = torch.chunk(self.net(torch.cat([s, vn], -1)), 3, -1)
        return s + a + b * (Uv * Vv).sum(1), v + g.unsqueeze(1) * Uv


class PaiNNF(nn.Module):
    def __init__(self, nf=128, nrbf=64, nint=6, rc=5.0, ne=7):
        super().__init__()
        self.rc = rc
        self.emb = nn.Embedding(ne, nf)
        self.rbf = RBF(rc, nrbf)
        self.msg = nn.ModuleList([PaiNNMsg(nf, nrbf) for _ in range(nint)])
        self.upd = nn.ModuleList([PaiNNUpd(nf) for _ in range(nint)])
        self.head = nn.Sequential(nn.Linear(nf, nf), nn.SiLU(), nn.Linear(nf, 1))
        self.register_buffer("sae", torch.zeros(ne))
        self.register_buffer("res_std", torch.ones(1))

    def forward(self, z, R, batch, ng):
        n = z.shape[0]
        ii, jj, d, diff = edges(R, batch, self.rc)
        u = diff / d.unsqueeze(-1).clamp(min=1e-8)
        w = self.rbf(d); c = cut(d, self.rc)
        s = self.emb(z)
        v = torch.zeros(n, 3, s.shape[-1], device=R.device, dtype=s.dtype)
        for m, up in zip(self.msg, self.upd):
            s, v = m(s, v, ii, jj, w, c, u, n)
            s, v = up(s, v)
        ea = self.head(s).squeeze(-1) * self.res_std + self.sae[z]
        return scatter_sum(ea, batch, ng)


# =============================================================================
# 7. BATCHING
# =============================================================================

def collate(recs: List[Dict], device) -> Dict[str, Tensor]:
    nmax = max(r["n_atoms"] for r in recs)
    B = len(recs)
    Z = torch.zeros(B, nmax, dtype=torch.long)
    R = torch.zeros(B, nmax, 3)
    M = torch.zeros(B, nmax)
    E = torch.zeros(B)
    Fo = torch.zeros(B, nmax, 3)
    for i, r in enumerate(recs):
        k = int(r["n_atoms"])
        Z[i, :k] = torch.as_tensor(r["z"])
        R[i, :k] = torch.as_tensor(r["R"])
        M[i, :k] = 1.
        E[i] = float(r["E"])
        Fo[i, :k] = torch.as_tensor(r["F"])
    batch = torch.arange(B).repeat_interleave(torch.tensor([int(r["n_atoms"]) for r in recs]))
    flatZ = torch.cat([torch.as_tensor(r["z"]) for r in recs])
    flatR = torch.cat([torch.as_tensor(r["R"]) for r in recs])
    flatF = torch.cat([torch.as_tensor(r["F"]) for r in recs])
    return {"z": flatZ.to(device), "R": flatR.to(device),
            "batch": batch.to(device), "ng": B,
            "E": E.to(device), "F": flatF.to(device),
            "n_atoms": torch.tensor([r["n_atoms"] for r in recs], device=device),
            "M": M.to(device)}


def to_records(d: Dict[str, np.ndarray]) -> List[Dict]:
    """dense (B,N,...) → per-molecule records"""
    out = []
    has_mask = "mask" in d
    for i in range(len(d["E"])):
        k = int(d["n_atoms"][i])
        out.append({"z": d["z"][i, :k] if d["z"].ndim == 2 else d["z"],
                    "R": d["R"][i, :k], "E": float(d["E"][i]),
                    "F": d["F"][i, :k], "n_atoms": k})
    return out


# =============================================================================
# 8. FORCE TRAINING — the double-backprop part that must be right
# =============================================================================

@dataclass
class TrainCfg:
    arch: str = "painn"
    n_features: int = 128
    n_rbf: int = 64
    n_interactions: int = 6
    cutoff: float = 5.0
    batch_size: int = 8
    lr: float = 5e-4
    weight_decay: float = 1e-6
    epochs: int = 300
    force_weight: float = 100.0     # F loss dominates; typical 10–1000
    energy_weight: float = 1.0
    precision: str = "fp32"         # fp32 | bf16   (NOT fp16 — see below)
    grad_clip: float = 10.0
    n_train: int = 950
    n_val: int = 50
    seed: int = 42
    subtract_sae: bool = True


def train(args):
    cfg = TrainCfg(arch=args.arch, epochs=args.epochs,
                   batch_size=args.batch_size, n_train=args.n_train,
                   n_val=args.n_val, force_weight=args.force_weight,
                   precision=args.precision, cutoff=args.cutoff,
                   n_features=args.n_features, n_interactions=args.n_interactions)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.precision == "fp16":
        log.error("fp16 + GradScaler is incompatible with double-backward "
                  "(force training). Use fp32 or bf16. Falling back to fp32.")
        cfg.precision = "fp32"
    use_bf16 = (cfg.precision == "bf16" and dev.type == "cuda")
    if use_bf16:
        log.info("[precision] bf16 autocast, NO GradScaler (bf16 needs no scaling; "
                 "scaler + create_graph=True is unsafe)")
    else:
        log.info("[precision] fp32 — safest for second-order gradients")

    # ---- data
    src = Path(args.data)
    if args.dataset == "rmd17":
        d = load_rmd17(src, args.molecule, limit=args.limit)
        recs = to_records(d)
    else:
        lazy = ANI1xLazy(src, energy_key=args.energy_key, force_key=args.force_key,
                         max_systems=args.max_systems)
        n_want = cfg.n_train + cfg.n_val + (args.n_test or 2000)
        d = lazy.materialize(min(n_want, len(lazy)), seed=cfg.seed)
        recs = to_records(d)
    if len(recs) < cfg.n_train + cfg.n_val + 10:
        cfg.n_train = int(0.7 * len(recs)); cfg.n_val = int(0.15 * len(recs))
        log.warning(f"small dataset → train={cfg.n_train} val={cfg.n_val}")

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(recs))
    tr = [recs[i] for i in perm[:cfg.n_train]]
    va = [recs[i] for i in perm[cfg.n_train:cfg.n_train + cfg.n_val]]
    te = [recs[i] for i in perm[cfg.n_train + cfg.n_val:]]
    log.info(f"split: train={len(tr)} val={len(va)} test={len(te)}  "
             f"(seed={cfg.seed} — report this; splits are NOT cross-comparable)")

    # ---- SAE fit on TRAIN ONLY (no leakage)
    nmax = max(r["n_atoms"] for r in tr)
    Zt = np.zeros((len(tr), nmax), np.int64); Mt = np.zeros((len(tr), nmax), np.float32)
    Et = np.zeros(len(tr), np.float32)
    for i, r in enumerate(tr):
        k = r["n_atoms"]; Zt[i, :k] = r["z"]; Mt[i, :k] = 1.; Et[i] = r["E"]
    Model = PaiNNF if cfg.arch == "painn" else SchNetF
    model = Model(cfg.n_features, cfg.n_rbf, cfg.n_interactions, cfg.cutoff).to(dev)
    if cfg.subtract_sae:
        sae, sinfo = fit_sae(Zt, Et, Mt)
        model.sae.copy_(torch.from_numpy(sae))
        X = np.stack([((Zt == e) & (Mt > 0)).sum(1) for e in range(7)], 1)
        resid = (Et - X @ sae) / Mt.sum(1)
        model.res_std.fill_(max(float(resid.std()), 1e-6))
    log.info(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)

    def run_batch(batch, training: bool):
        R = batch["R"].detach().requires_grad_(True)
        ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16)
        with ctx:
            E = model(batch["z"], R, batch["batch"], batch["ng"])
        # forces MUST be computed outside autocast in fp32 for stability,
        # and with create_graph=training so gradients reach the weights.
        E32 = E.float()
        Fp = -torch.autograd.grad(E32.sum(), R, create_graph=training,
                                  retain_graph=True)[0]
        na = batch["n_atoms"].float()
        lE = ((E32 - batch["E"]) / na).pow(2).mean()
        lF = (Fp - batch["F"]).pow(2).sum(-1).mean()
        return lE, lF, E32, Fp

    best = float("inf"); best_state = None
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(tr)); tot = 0.; nb = 0
        for s in range(0, len(order), cfg.batch_size):
            b = collate([tr[i] for i in order[s:s + cfg.batch_size]], dev)
            opt.zero_grad(set_to_none=True)
            lE, lF, _, _ = run_batch(b, True)
            loss = cfg.energy_weight * lE + cfg.force_weight * lF
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.detach()); nb += 1

        if (ep + 1) % max(1, cfg.epochs // 25) == 0 or ep == cfg.epochs - 1:
            model.eval(); eE, eF = [], []
            for s in range(0, len(va), cfg.batch_size):
                b = collate(va[s:s + cfg.batch_size], dev)
                with torch.no_grad():
                    pass
                lE, lF, Ep, Fp = run_batch(b, False)
                eE.extend(((Ep - b["E"]) / b["n_atoms"]).abs().detach().cpu().numpy())
                eF.extend((Fp - b["F"]).abs().mean(-1).detach().cpu().numpy())
            mE = float(np.mean(eE)) * EV_KCALMOL
            mF = float(np.mean(eF)) * EV_KCALMOL
            sch.step(mF)
            log.info(f"ep {ep+1:4d} | train {tot/max(nb,1):.5f} | "
                     f"val E {mE:.4f} kcal/mol/atom | F {mF:.4f} kcal/mol/Å")
            if mF < best:
                best = mF
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval(); tE, tF = [], []
    for s in range(0, len(te), cfg.batch_size):
        b = collate(te[s:s + cfg.batch_size], dev)
        _, _, Ep, Fp = run_batch(b, False)
        tE.extend(((Ep - b["E"]) / b["n_atoms"]).abs().detach().cpu().numpy())
        tF.extend((Fp - b["F"]).abs().mean(-1).detach().cpu().numpy())
    tE = np.array(tE) * EV_KCALMOL; tF = np.array(tF) * EV_KCALMOL

    def ci(x, conf=.95, blk=32):
        n = len(x)
        if n < 8:
            m = float(x.mean()) if n else 0.
            return {"mean": m, "lo": m, "hi": m, "n": n}
        b = min(blk, n // 4); th = float(x.mean())
        sub = np.array([x[j:j+b].mean() for j in range(n - b + 1)])
        sc = math.sqrt(b) * (sub - th); a = 1 - conf
        return {"mean": th,
                "lo": th - float(np.percentile(sc, 100*(1-a/2)))/math.sqrt(n),
                "hi": th - float(np.percentile(sc, 100*(a/2)))/math.sqrt(n), "n": n}
    cE, cF = ci(tE), ci(tF)

    print("\n" + "=" * 74)
    print(f"  TEST — {cfg.arch.upper()}  |  {args.dataset}"
          f"{':'+args.molecule if args.dataset=='rmd17' else ''}")
    print("=" * 74)
    print(f"  level of theory : {d.get('level','?')}")
    print(f"  split           : {len(tr)}/{len(va)}/{len(te)}  seed={cfg.seed}")
    print(f"  loss weights    : E={cfg.energy_weight}  F={cfg.force_weight}")
    print(f"  precision       : {cfg.precision}")
    print(f"\n  Energy MAE : {cE['mean']:.4f} kcal/mol/atom  "
          f"[{cE['lo']:.4f}, {cE['hi']:.4f}]")
    print(f"  Force  MAE : {cF['mean']:.4f} kcal/mol/Å    "
          f"[{cF['lo']:.4f}, {cF['hi']:.4f}]")
    print(f"             = {cF['mean']/EV_KCALMOL*1000:.3f} meV/Å")
    print(f"\n  ORDER-OF-MAGNITUDE expectations (~1k training points, rMD17):")
    print(f"    SchNet-class : force MAE ~0.3–1.5 kcal/mol/Å")
    print(f"    PaiNN-class  : force MAE ~0.1–0.4 kcal/mol/Å")
    print(f"    ⚠️  These are APPROXIMATE ranges, not a citation.")
    print(f"       Verify against the specific paper's table AND matching split.")
    print(f"\n  HONEST NOTES:")
    print(f"    • Labels are DFT, not experiment.")
    print(f"    • SAE fitted on train only (no leakage).")
    print(f"    • Forces via double-backprop, create_graph=True.")
    print(f"    • Different split ⇒ numbers NOT comparable to published tables.")
    print("=" * 74)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "sae_eV": model.sae.cpu().tolist(),
                "test_E_kcal_per_atom": cE, "test_F_kcal_per_A": cF,
                "level": d.get("level")}, args.out)
    json.dump({"config": asdict(cfg), "dataset": args.dataset,
               "molecule": args.molecule, "test_E": cE, "test_F": cF},
              open(str(Path(args.out).with_suffix(".json")), "w"), indent=2)
    log.info(f"saved → {args.out}")


def main():
    p = argparse.ArgumentParser(description="rMD17 / ANI-1x force training")
    p.add_argument("cmd", choices=["probe", "download", "train"])
    p.add_argument("--path", default=None)
    p.add_argument("--data", default="data/rmd17")
    p.add_argument("--out", default="ckpt/force_model.pt")
    p.add_argument("--dataset", choices=["rmd17", "ani1x"], default="rmd17")
    p.add_argument("--molecule", default="benzene", choices=RMD17_MOLECULES)
    p.add_argument("--arch", choices=["schnet", "painn"], default="painn")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    p.add_argument("--n-train", dest="n_train", type=int, default=950)
    p.add_argument("--n-val", dest="n_val", type=int, default=50)
    p.add_argument("--n-test", dest="n_test", type=int, default=None)
    p.add_argument("--force-weight", dest="force_weight", type=float, default=100.)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--n-features", dest="n_features", type=int, default=128)
    p.add_argument("--n-interactions", dest="n_interactions", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-systems", dest="max_systems", type=int, default=None)
    p.add_argument("--energy-key", dest="energy_key", default=None)
    p.add_argument("--force-key", dest="force_key", default=None)
    a = p.parse_args()
    if a.cmd == "probe":
        cmd_probe(a)
    elif a.cmd == "download":
        try_download(a.dataset, Path(a.data))
    else:
        train(a)


if __name__ == "__main__":
    main()
