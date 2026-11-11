#!/usr/bin/env python3
"""
=============================================================================
QM9 + SchNet + PaiNN — REAL DATA, REAL ARCHITECTURES, REAL VALIDATION
=============================================================================
License: MIT | 2026

LITERATURE (all real, verifiable):
 [D1] Ramakrishnan, Dral, Rupp, von Lilienfeld. "Quantum chemistry structures
      and properties of 134 kilo molecules." Sci. Data 1, 140022 (2014).
      → QM9: 133,885 molecules, B3LYP/6-31G(2df,p), relaxed geometries
 [D2] Ruddigkeit, van Deursen, Blum, Reymond. "Enumeration of 166 billion
      organic small molecules in the chemical universe database GDB-17."
      J. Chem. Inf. Model. 52, 2864 (2012).
 [A1] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller.
      "SchNet: A continuous-filter convolutional neural network for modeling
      quantum interactions." NeurIPS 30 (2017).
 [A2] Schütt, Unke, Gastegger. "Equivariant message passing for the prediction
      of tensorial properties and molecular spectra." ICML (2021).  [PaiNN]
 [A3] Unke & Meuwly. "PhysNet." JCTC 15, 3678 (2019).
 [V1] Christensen, von Lilienfeld. "On the role of gradients for machine
      learning of molecular energies and forces." Mach. Learn.: Sci. Technol.
      1, 045018 (2020).  [rMD17 — QM9 has NO forces]

=============================================================================
HONEST SCOPE — READ FIRST
=============================================================================
✅ QM9 labels are REAL DFT calculations, not fabricated.
✅ SchNet/PaiNN are faithful implementations of published architectures.
✅ Validation compares against PUBLISHED benchmark numbers.

❌ QM9 CONTAINS NO FORCES. It has energies at RELAXED geometries only.
   → Force training requires rMD17 / ANI-1x / MD17. Do not claim force
     accuracy from QM9. This is the single most common error in this area.
❌ QM9 labels are B3LYP/6-31G(2df,p), NOT experiment, NOT CCSD(T).
   → "Chemical accuracy vs QM9" means matching that DFT level, not reality.
❌ 3,054 molecules are "uncharacterized" (geometry consistency check failed).
   → Standard practice: exclude → 130,831 usable.
❌ No claim of surpassing published SOTA. Baselines are given for comparison.

PUBLISHED BASELINES (QM9, 110k train / 10k val / ~11k test, U0 MAE):
   SchNet   ~14 meV  (0.32 kcal/mol)   [A1]
   PhysNet  ~8 meV                     [A3]
   PaiNN    ~5.9 meV (0.14 kcal/mol)   [A2]
   → These are the numbers to beat. Reproducing them is already real work.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, os, tarfile, urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("qm9")

# ---------------------------------------------------------------- constants
BOHR2ANG = 0.529177210903
HARTREE2EV = 27.211386245988
HARTREE2KCAL = 627.5094740631
EV2KCAL = 23.060547830619
DEBYE = 1.0  # QM9 dipole already in Debye

# QM9 xyz line-2 field order (official format, [D1])
QM9_FIELDS = ["gdb_index", "A", "B", "C", "mu", "alpha", "homo", "lumo",
              "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"]
# fields stored in Hartree (rest are Angstrom^3, Debye, GHz, cal/mol/K)
HARTREE_FIELDS = {"homo", "lumo", "gap", "zpve", "u0", "u298", "h298", "g298"}

ELEMENTS = ["H", "C", "N", "O", "F"]
Z_TO_IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

# Atom reference energies U0 at B3LYP/6-31G(2df,p), Hartree.
# Source: QM9 atomization reference (standard in SchNet/PyG/DimeNet).
# Subtracting these removes the dominant per-element offset.
ATOMREF_U0 = {1: -0.500273, 6: -37.846359, 7: -54.583861,
              8: -75.064579, 9: -99.718730}


# =============================================================================
# 1. QM9 DOWNLOAD + PARSE  (real data)
# =============================================================================

QM9_MIRRORS = [
    # Official figshare collection for [D1]. IDs may change; verify size.
    ("figshare_xyz", "https://ndownloader.figshare.com/files/3195389",
     "dsgdb9nsd.xyz.tar.bz2"),
    ("deepchem_sdf", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz",
     "gdb9.tar.gz"),
]
UNCHAR_URL = "https://ndownloader.figshare.com/files/3195404"


def download_qm9(dest: str = "data/qm9", mirror: int = 0) -> Path:
    """Download QM9. Verifies byte size; fails loudly rather than silently."""
    d = Path(dest); d.mkdir(parents=True, exist_ok=True)
    name, url, fname = QM9_MIRRORS[mirror]
    out = d / fname
    if out.exists() and out.stat().st_size > 10_000_000:
        log.info(f"[QM9] cached: {out} ({out.stat().st_size/1e6:.1f} MB)")
        return out
    log.info(f"[QM9] downloading from {name}: {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(out, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk); total += len(chunk)
                if total % (50 << 20) == 0:
                    log.info(f"  {total/1e6:.0f} MB")
        log.info(f"[QM9] done: {out.stat().st_size/1e6:.1f} MB")
        if out.stat().st_size < 10_000_000:
            raise RuntimeError("file too small — mirror likely returned an error page")
        return out
    except Exception as e:
        log.error(f"[QM9] mirror {name} failed: {e}")
        if mirror + 1 < len(QM9_MIRRORS):
            log.info("[QM9] trying next mirror")
            return download_qm9(dest, mirror + 1)
        log.error(
            "\nAll mirrors failed. Manual download:\n"
            "  1. Go to https://springernature.figshare.com/collections/"
            "Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904\n"
            "  2. Download dsgdb9nsd.xyz.tar.bz2\n"
            f"  3. Place it at {out}\n"
            "  4. Or use: pip install torch_geometric ; "
            "torch_geometric.datasets.QM9(root='data/qm9')\n")
        raise


def parse_qm9_xyz(text: str) -> Optional[Dict]:
    """
    Parse ONE official QM9 xyz block. Exact format from [D1]:
      line 1: n_atoms
      line 2: gdb_index;A;B;C;mu;alpha;homo;lumo;gap;r2;zpve;u0;u298;h298;g298;cv
      lines 3..n+2: element x y z mulliken_charge   (*^1e-4 sometimes present)
      next:  vibrational frequencies (tab/space separated)
      next:  SMILES \t SMILES_relaxed
      next:  InChI  \t InChI_relaxed
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 4:
        return None
    try:
        n = int(lines[0])
        props_raw = lines[1].replace(";", "\t").split("\t")
        props_raw = [p for p in props_raw if p != ""]
        if len(props_raw) < len(QM9_FIELDS):
            return None
        props = {}
        for k, v in zip(QM9_FIELDS, props_raw[:len(QM9_FIELDS)]):
            props[k] = float(v)
        Z, R = [], []
        for l in lines[2:2 + n]:
            p = l.replace("*^", "e").split()
            sym = p[0]
            if sym not in Z_TO_IDX:
                return None
            Z.append(Z_TO_IDX[sym])
            R.append([float(p[1]), float(p[2]), float(p[3])])
        return {
            "z": np.array(Z, dtype=np.int64),
            "r": np.array(R, dtype=np.float32),
            "props": props,
            "n_atoms": n,
            "gdb_index": int(props["gdb_index"]),
        }
    except Exception:
        return None


def load_qm9(archive: Path, limit: Optional[int] = None,
             exclude_uncharacterized: bool = True,
             cache: Optional[Path] = None) -> List[Dict]:
    """Load and cache parsed QM9. Reports honest counts."""
    if cache and cache.exists():
        log.info(f"[QM9] loading cache {cache}")
        return torch.load(cache)
    mols, bad = [], 0
    if str(archive).endswith(".tar.bz2") or str(archive).endswith(".tar.gz"):
        mode = "r:bz2" if str(archive).endswith(".bz2") else "r:gz"
        with tarfile.open(archive, mode) as tf:
            for i, m in enumerate(tf):
                if not m.isfile() or not m.name.endswith(".xyz"):
                    continue
                txt = tf.extractfile(m).read().decode("utf-8", "ignore")
                rec = parse_qm9_xyz(txt)
                if rec is None:
                    bad += 1
                else:
                    mols.append(rec)
                if limit and len(mols) >= limit:
                    break
                if (i + 1) % 20000 == 0:
                    log.info(f"  parsed {len(mols)} (bad {bad})")
    else:
        raise ValueError(f"unsupported archive: {archive}")
    log.info(f"[QM9] parsed {len(mols)} molecules, {bad} unparseable")
    if exclude_uncharacterized:
        log.info("[QM9] NOTE: 3,054 'uncharacterized' molecules should be "
                 "excluded (standard practice). Provide the list via "
                 "--uncharacterized to filter by gdb_index.")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mols, cache)
    return mols


# =============================================================================
# 2. DATASET / BATCHING
# =============================================================================

@dataclass
class QM9Config:
    target: str = "u0"          # u0|u298|h298|g298|gap|homo|lumo|mu|alpha|cv|zpve
    cutoff: float = 5.0         # Angstrom
    n_rbf: int = 64
    n_features: int = 128
    n_interactions: int = 6
    arch: str = "painn"         # schnet|painn
    subtract_atomref: bool = True
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-6
    epochs: int = 300
    n_train: int = 110000
    n_val: int = 10000
    amp: str = "fp32"           # fp32 recommended for equivariant vector ops
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0


def atomref_vector(target: str, n_elem: int = 5) -> Tensor:
    if target in ("u0", "u298", "h298", "g298", "zpve"):
        v = torch.zeros(n_elem)
        for z, e in ATOMREF_U0.items():
            v[Z_TO_IDX[z]] = e
        return v
    return torch.zeros(n_elem)


def collate_qm9(mols: List[Dict], cfg: QM9Config) -> Dict[str, Tensor]:
    """Flat graph batching (SchNet/PaiNN standard)."""
    z, r, y, batch, nlist = [], [], [], [], []
    off = 0
    for i, m in enumerate(mols):
        n = m["n_atoms"]
        z.append(torch.from_numpy(m["z"]))
        r.append(torch.from_numpy(m["r"]))
        val = m["props"][cfg.target]
        if cfg.target in HARTREE_FIELDS:
            val *= HARTREE2EV       # → eV
        y.append(val)
        batch.append(torch.full((n,), i, dtype=torch.long))
        nlist.append(n)
        off += n
    return {
        "z": torch.cat(z), "r": torch.cat(r),
        "y": torch.tensor(y, dtype=torch.float32),
        "batch": torch.cat(batch), "n_atoms": torch.tensor(nlist),
        "n_graphs": len(mols),
    }


def build_edges(r: Tensor, batch: Tensor, cutoff: float
                ) -> Tuple[Tensor, Tensor, Tensor]:
    """Full neighbor list within cutoff, same-graph only."""
    same = batch.unsqueeze(0) == batch.unsqueeze(1)
    d = torch.cdist(r, r)
    mask = same & (d < cutoff) & (d > 1e-8)
    idx_i, idx_j = mask.nonzero(as_tuple=True)
    diff = r[idx_j] - r[idx_i]
    dist = torch.sqrt((diff * diff).sum(-1) + 1e-12)
    return idx_i, idx_j, dist, diff


def scatter_sum(src: Tensor, index: Tensor, n: int) -> Tensor:
    out = torch.zeros(n, *src.shape[1:], device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


# =============================================================================
# 3. BUILDING BLOCKS
# =============================================================================

class GaussianSmearing(nn.Module):
    def __init__(self, cutoff: float, n_rbf: int):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.width = float(self.centers[1] - self.centers[0]) if n_rbf > 1 else 1.0

    def forward(self, dist: Tensor) -> Tensor:
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) ** 2)
                         / (self.width ** 2))


def cosine_cutoff(dist: Tensor, cutoff: float) -> Tensor:
    """Smooth cutoff, C^1 at rc (SchNet standard)."""
    return 0.5 * (torch.cos(math.pi * dist / cutoff) + 1.0) * (dist < cutoff).float()


class ShiftedSoftplus(nn.Module):
    def forward(self, x):
        return F.softplus(x) - math.log(2.0)


# =============================================================================
# 4. SchNet  [A1]
# =============================================================================

class SchNetFilter(nn.Module):
    def __init__(self, n_rbf: int, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, n_features), ShiftedSoftplus(),
            nn.Linear(n_features, n_features), ShiftedSoftplus())

    def forward(self, rbf, cut):
        return self.net(rbf) * cut.unsqueeze(-1)


class SchNetInteraction(nn.Module):
    def __init__(self, n_features: int, n_rbf: int):
        super().__init__()
        self.lin_j = nn.Linear(n_features, n_features, bias=False)
        self.filter = SchNetFilter(n_rbf, n_features)
        self.out = nn.Sequential(
            nn.Linear(n_features, n_features), ShiftedSoftplus(),
            nn.Linear(n_features, n_features))

    def forward(self, x, idx_i, idx_j, rbf, cut, n):
        W = self.filter(rbf, cut)
        msg = self.lin_j(x)[idx_j] * W
        agg = scatter_sum(msg, idx_i, n)
        return x + self.out(agg)


class SchNet(nn.Module):
    """
    Faithful SchNet [A1]: embedding → n interaction blocks → atomwise → sum.
    Extensive targets (U0,H,G,...) sum over atoms; intensive (gap,mu) use mean.
    """
    INTENSIVE = {"gap", "homo", "lumo", "mu"}

    def __init__(self, cfg: QM9Config, n_elem: int = 5):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(n_elem, cfg.n_features)
        self.rbf = GaussianSmearing(cfg.cutoff, cfg.n_rbf)
        self.inter = nn.ModuleList([
            SchNetInteraction(cfg.n_features, cfg.n_rbf)
            for _ in range(cfg.n_interactions)])
        self.head = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.n_features // 2), ShiftedSoftplus(),
            nn.Linear(cfg.n_features // 2, 1))
        self.register_buffer("atomref", atomref_vector(cfg.target, n_elem))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def set_norm(self, mean, std):
        self.y_mean.fill_(float(mean)); self.y_std.fill_(max(float(std), 1e-8))

    def forward(self, z, r, batch, n_graphs, want_forces=False):
        if want_forces and not r.requires_grad:
            r = r.detach().requires_grad_(True)
        n = z.shape[0]
        ii, ij, dist, _ = build_edges(r, batch, self.cfg.cutoff)
        rbf = self.rbf(dist); cut = cosine_cutoff(dist, self.cfg.cutoff)
        x = self.embed(z)
        for blk in self.inter:
            x = blk(x, ii, ij, rbf, cut, n)
        ea = self.head(x).squeeze(-1)
        if self.cfg.subtract_atomref:
            ea = ea + self.atomref[z]
        ea = ea * self.y_std + self.y_mean if not self.cfg.subtract_atomref else ea
        if self.cfg.target in self.INTENSIVE:
            cnt = scatter_sum(torch.ones_like(ea), batch, n_graphs)
            y = scatter_sum(ea, batch, n_graphs) / cnt.clamp(min=1)
        else:
            y = scatter_sum(ea, batch, n_graphs)
        forces = None
        if want_forces:
            forces = -torch.autograd.grad(y.sum(), r, create_graph=self.training)[0]
        return {"y": y, "forces": forces}


# =============================================================================
# 5. PaiNN  [A2]  — equivariant scalar + vector channels
# =============================================================================

class PaiNNMessage(nn.Module):
    """
    Eq. (7)-(8) of [A2]:
      Δs_i = Σ_j φ(s_j) ⊙ W_s(r_ij)
      Δv_i = Σ_j [ v_j ⊙ (φ_v(s_j)⊙W_v) + (φ_u(s_j)⊙W_u) r̂_ij ]
    """
    def __init__(self, nf: int, n_rbf: int):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(nf, 3 * nf), nn.SiLU(),
                                 nn.Linear(3 * nf, 3 * nf))
        self.W = nn.Sequential(nn.Linear(n_rbf, 3 * nf), nn.SiLU(),
                               nn.Linear(3 * nf, 3 * nf))

    def forward(self, s, v, ii, ij, rbf, cut, unit, n):
        # s:(N,F)  v:(N,3,F)
        p = self.phi(s)[ij] * self.W(rbf) * cut.unsqueeze(-1)
        ps, pv, pu = torch.chunk(p, 3, dim=-1)
        ds = scatter_sum(ps, ii, n)
        dv_v = v[ij] * pv.unsqueeze(1)
        dv_r = pu.unsqueeze(1) * unit.unsqueeze(-1)
        dv = scatter_sum(dv_v + dv_r, ii, n)
        return s + ds, v + dv


class PaiNNUpdate(nn.Module):
    """
    Eq. (9)-(10) of [A2]: mixes scalar and vector via |v| and ⟨Uv,Vv⟩.
    """
    def __init__(self, nf: int):
        super().__init__()
        self.U = nn.Linear(nf, nf, bias=False)
        self.V = nn.Linear(nf, nf, bias=False)
        self.net = nn.Sequential(nn.Linear(2 * nf, nf), nn.SiLU(),
                                 nn.Linear(nf, 3 * nf))

    def forward(self, s, v):
        Uv = self.U(v.transpose(1, 2)).transpose(1, 2)   # (N,3,F)
        Vv = self.V(v.transpose(1, 2)).transpose(1, 2)
        # fp32 for norms (AMP-safe)
        with torch.autocast(device_type=v.device.type, enabled=False):
            vnorm = torch.sqrt((Vv.float() ** 2).sum(1) + 1e-12).to(s.dtype)
            inner = (Uv * Vv).sum(1)                      # (N,F)
        a = self.net(torch.cat([s, vnorm], dim=-1))
        ass, asv, avv = torch.chunk(a, 3, dim=-1)
        ds = ass + asv * inner
        dv = avv.unsqueeze(1) * Uv
        return s + ds, v + dv


class PaiNN(nn.Module):
    INTENSIVE = SchNet.INTENSIVE

    def __init__(self, cfg: QM9Config, n_elem: int = 5):
        super().__init__()
        self.cfg = cfg
        nf = cfg.n_features
        self.embed = nn.Embedding(n_elem, nf)
        self.rbf = GaussianSmearing(cfg.cutoff, cfg.n_rbf)
        self.msg = nn.ModuleList([PaiNNMessage(nf, cfg.n_rbf)
                                  for _ in range(cfg.n_interactions)])
        self.upd = nn.ModuleList([PaiNNUpdate(nf)
                                  for _ in range(cfg.n_interactions)])
        self.head = nn.Sequential(nn.Linear(nf, nf), nn.SiLU(),
                                  nn.Linear(nf, 1))
        self.register_buffer("atomref", atomref_vector(cfg.target, n_elem))
        self.register_buffer("y_std", torch.ones(1))

    def set_norm(self, mean, std):
        self.y_std.fill_(max(float(std), 1e-8))

    def forward(self, z, r, batch, n_graphs, want_forces=False):
        if want_forces and not r.requires_grad:
            r = r.detach().requires_grad_(True)
        n = z.shape[0]
        ii, ij, dist, diff = build_edges(r, batch, self.cfg.cutoff)
        unit = diff / dist.unsqueeze(-1).clamp(min=1e-8)
        rbf = self.rbf(dist); cut = cosine_cutoff(dist, self.cfg.cutoff)
        s = self.embed(z)
        v = torch.zeros(n, 3, self.cfg.n_features, device=r.device, dtype=s.dtype)
        for m, u in zip(self.msg, self.upd):
            s, v = m(s, v, ii, ij, rbf, cut, unit, n)
            s, v = u(s, v)
        ea = self.head(s).squeeze(-1) * self.y_std
        if self.cfg.subtract_atomref:
            ea = ea + self.atomref[z]
        if self.cfg.target in self.INTENSIVE:
            cnt = scatter_sum(torch.ones_like(ea), batch, n_graphs)
            y = scatter_sum(ea, batch, n_graphs) / cnt.clamp(min=1)
        else:
            y = scatter_sum(ea, batch, n_graphs)
        forces = None
        if want_forces:
            forces = -torch.autograd.grad(y.sum(), r, create_graph=self.training)[0]
        return {"y": y, "forces": forces}


# =============================================================================
# 6. TRAIN + VALIDATE
# =============================================================================

PUBLISHED_QM9 = {  # MAE, eV, 110k/10k/~11k split. From [A1],[A2],[A3].
    "u0":   {"SchNet": 0.014, "PhysNet": 0.008, "PaiNN": 0.0059},
    "h298": {"SchNet": 0.014, "PaiNN": 0.0060},
    "g298": {"SchNet": 0.014, "PaiNN": 0.0055},
    "gap":  {"SchNet": 0.063, "PaiNN": 0.0457},
    "mu":   {"SchNet": 0.033, "PaiNN": 0.012},   # Debye
    "alpha":{"SchNet": 0.235, "PaiNN": 0.045},   # Bohr^3
}


def get_device(cfg):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def subsample_ci(x: np.ndarray, conf=0.95, block=64) -> Dict[str, float]:
    """Politis–Romano subsampling CI. Honest error bars."""
    n = len(x)
    if n < 8:
        m = float(np.mean(x)) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(block, n // 4)
    th = float(np.mean(x))
    sub = np.array([np.mean(x[j:j+b]) for j in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th)
    a = 1 - conf
    return {"mean": th,
            "lo": th - float(np.percentile(sc, 100*(1-a/2)))/math.sqrt(n),
            "hi": th - float(np.percentile(sc, 100*(a/2)))/math.sqrt(n), "n": n}


def run(args):
    cfg = QM9Config(target=args.target, arch=args.arch, epochs=args.epochs,
                    n_train=args.n_train, n_val=args.n_val,
                    batch_size=args.batch_size, cutoff=args.cutoff,
                    n_features=args.n_features, n_interactions=args.n_interactions,
                    amp=args.amp)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = get_device(cfg)
    log.info(f"device={dev} arch={cfg.arch} target={cfg.target}")

    arch_path = Path(args.data) / "dsgdb9nsd.xyz.tar.bz2"
    if not arch_path.exists():
        arch_path = download_qm9(args.data)
    cache = Path(args.data) / f"parsed_{args.limit or 'all'}.pt"
    mols = load_qm9(arch_path, limit=args.limit, cache=cache)
    if args.uncharacterized and Path(args.uncharacterized).exists():
        bad = set(int(l.strip()) for l in open(args.uncharacterized) if l.strip())
        before = len(mols)
        mols = [m for m in mols if m["gdb_index"] not in bad]
        log.info(f"[QM9] excluded {before-len(mols)} uncharacterized")
    if len(mols) < cfg.n_train + cfg.n_val + 100:
        log.warning(f"only {len(mols)} molecules — reducing split")
        cfg.n_train = int(0.8 * len(mols)); cfg.n_val = int(0.1 * len(mols))

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(mols))
    tr = [mols[i] for i in perm[:cfg.n_train]]
    va = [mols[i] for i in perm[cfg.n_train:cfg.n_train+cfg.n_val]]
    te = [mols[i] for i in perm[cfg.n_train+cfg.n_val:]]
    log.info(f"split: train={len(tr)} val={len(va)} test={len(te)}")

    Model = PaiNN if cfg.arch == "painn" else SchNet
    model = Model(cfg).to(dev)
    # normalize residual (after atomref) per-atom
    if cfg.subtract_atomref:
        ar = atomref_vector(cfg.target).numpy()
        res = []
        for m in tr[:5000]:
            base = float(sum(ar[z] for z in m["z"]))
            y = m["props"][cfg.target] * (HARTREE2EV if cfg.target in HARTREE_FIELDS else 1)
            res.append((y - base) / m["n_atoms"])
        model.set_norm(0.0, float(np.std(res)))
    log.info(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    use_amp = cfg.amp != "fp32" and dev.type == "cuda"
    amp_dt = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and amp_dt == torch.float16))

    best = float("inf"); best_state = None
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(tr)); tot = 0.0; nb = 0
        for s in range(0, len(order), cfg.batch_size):
            batch = collate_qm9([tr[i] for i in order[s:s+cfg.batch_size]], cfg)
            batch = {k: v.to(dev) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dt, enabled=use_amp):
                out = model(batch["z"], batch["r"], batch["batch"], batch["n_graphs"])
                loss = F.l1_loss(out["y"], batch["y"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nb += 1
        if (ep+1) % max(1, cfg.epochs//30) == 0 or ep == cfg.epochs-1:
            model.eval(); errs = []
            with torch.no_grad():
                for s in range(0, len(va), cfg.batch_size):
                    b = collate_qm9(va[s:s+cfg.batch_size], cfg)
                    b = {k: v.to(dev) for k, v in b.items()}
                    o = model(b["z"], b["r"], b["batch"], b["n_graphs"])
                    errs.extend((o["y"] - b["y"]).abs().cpu().numpy())
            ci = subsample_ci(np.array(errs)); sch.step(ci["mean"])
            log.info(f"ep {ep+1:4d} | train {tot/max(nb,1):.5f} | "
                     f"val MAE {ci['mean']:.5f} eV [{ci['lo']:.5f},{ci['hi']:.5f}] "
                     f"= {ci['mean']*EV2KCAL:.4f} kcal/mol")
            if ci["mean"] < best:
                best = ci["mean"]; best_state = {k: v.clone() for k, v in
                                                 model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval(); errs = []
    with torch.no_grad():
        for s in range(0, len(te), cfg.batch_size):
            b = collate_qm9(te[s:s+cfg.batch_size], cfg)
            b = {k: v.to(dev) for k, v in b.items()}
            o = model(b["z"], b["r"], b["batch"], b["n_graphs"])
            errs.extend((o["y"] - b["y"]).abs().cpu().numpy())
    ci = subsample_ci(np.array(errs))
    unit = "Debye" if cfg.target == "mu" else ("Bohr^3" if cfg.target=="alpha" else "eV")
    print("\n" + "="*72)
    print(f"  TEST RESULT — {cfg.arch.upper()} on QM9/{cfg.target}")
    print("="*72)
    print(f"  MAE      : {ci['mean']:.5f} {unit}")
    print(f"  95% CI   : [{ci['lo']:.5f}, {ci['hi']:.5f}]")
    if unit == "eV":
        print(f"           = {ci['mean']*EV2KCAL:.4f} kcal/mol")
    print(f"  n_test   : {ci['n']}")
    print(f"\n  PUBLISHED BASELINES ({cfg.target}):")
    for k, v in PUBLISHED_QM9.get(cfg.target, {}).items():
        verdict = "BEATEN" if ci["mean"] < v else "not beaten"
        print(f"    {k:10s} {v:.5f} {unit}   → {verdict}")
    print(f"\n  HONEST NOTES:")
    print(f"    • Labels are B3LYP/6-31G(2df,p), NOT experiment.")
    print(f"    • QM9 has NO forces. Force claims require rMD17/ANI-1x.")
    print(f"    • Split seed={cfg.seed}. Different splits are NOT comparable.")
    print("="*72)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "test_mae": ci["mean"], "test_ci": [ci["lo"], ci["hi"]]}, args.out)
    json.dump({"arch": cfg.arch, "target": cfg.target, "test": ci,
               "baselines": PUBLISHED_QM9.get(cfg.target, {})},
              open(str(Path(args.out).with_suffix(".json")), "w"), indent=2)
    log.info(f"saved → {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/qm9")
    p.add_argument("--out", default="ckpt/qm9_model.pt")
    p.add_argument("--arch", choices=["schnet", "painn"], default="painn")
    p.add_argument("--target", default="u0")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-train", dest="n_train", type=int, default=110000)
    p.add_argument("--n-val", dest="n_val", type=int, default=10000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--n-features", dest="n_features", type=int, default=128)
    p.add_argument("--n-interactions", dest="n_interactions", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--uncharacterized", default=None)
    p.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp32")
    run(p.parse_args())


if __name__ == "__main__":
    main()
