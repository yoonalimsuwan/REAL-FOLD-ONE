#!/usr/bin/env python3
"""
=============================================================================
REAL BACKEND BRIDGE — dict fixes, real QC/LGT, clinical gates, real training
=============================================================================

WHAT THIS FILE DOES
  1. Fixes trailing-space dict keys (KeyError on every organ / protein type)
  2. Replaces fake QC (nn.Parameter randn) with PySCF-computed integrals
  3. Replaces random SU(3) links with Monte-Carlo configurations
  4. REMOVES surgical-planning outputs (random-weight success probability
     for brain surgery is not a bug to fix — it is a hazard to delete)
  5. GATES brain decoding behind trained weights + calibration + closed set
  6. Provides a trainer that writes logs, checkpoints, and a held-out test score

DEPENDENCIES (from the earlier files in this project)
  real_qc_lg.py              → PySCF reference, ACSF, NeuralPotential
  lattice_heatbath_rk3.py    → SU3Lattice, LatticeConfig, MC generation
  force_datasets.py          → rMD17 loader, PaiNNF, collate
  qm9_schnet_painn.py        → QM9 loader, PaiNN

HONEST SCOPE
  ❌ Fixing syntax does not create capability.
  ❌ No surgical / diagnostic output is produced anywhere in this file.
  ❌ Untrained surrogates raise, they do not return plausible-looking numbers.
  ✅ Real integrals, real gauge configurations, real logs, real test split.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("real_bridge")


# =============================================================================
# 1. DICT KEY FIX  (crash #1)
# =============================================================================

def normalize_keys(d: Dict) -> Dict:
    """Strip whitespace from string keys AND string values, recursively."""
    out = {}
    for k, v in d.items():
        kk = k.strip() if isinstance(k, str) else k
        if isinstance(v, dict):
            vv = normalize_keys(v)
        elif isinstance(v, str):
            vv = v.strip()
        else:
            vv = v
        out[kk] = vv
    return out


class SafeMeta(dict):
    """Dict that tolerates legacy trailing-space keys instead of raising."""
    def __missing__(self, key):
        if isinstance(key, str):
            for k in self.keys():
                if isinstance(k, str) and k.strip() == key.strip():
                    return self[k]
        raise KeyError(key)


# --- FIXED: iPSC_ORGAN_REGENERATION ---
ORGAN_META: Dict[str, Dict[str, Any]] = normalize_keys({
    "KIDNEY":          {"output_dim": 8,  "complexity": "medium",    "surgical_difficulty": 3},
    "LIVER":           {"output_dim": 10, "complexity": "high",      "surgical_difficulty": 4},
    "HEART":           {"output_dim": 12, "complexity": "very_high", "surgical_difficulty": 5},
    "LARGE_INTESTINE": {"output_dim": 8,  "complexity": "medium",    "surgical_difficulty": 3},
    "SMALL_INTESTINE": {"output_dim": 8,  "complexity": "medium",    "surgical_difficulty": 3},
    "STOMACH":         {"output_dim": 8,  "complexity": "medium",    "surgical_difficulty": 3},
    "BRAIN":           {"output_dim": 16, "complexity": "extreme",   "surgical_difficulty": 5},
    "EYE":             {"output_dim": 10, "complexity": "high",      "surgical_difficulty": 4},
    "BONE":            {"output_dim": 6,  "complexity": "low",       "surgical_difficulty": 2},
    "ARM":             {"output_dim": 8,  "complexity": "high",      "surgical_difficulty": 4},
    "LEG":             {"output_dim": 8,  "complexity": "high",      "surgical_difficulty": 4},
    "UTERUS":          {"output_dim": 8,  "complexity": "medium",    "surgical_difficulty": 3},
    "TESTIS":          {"output_dim": 6,  "complexity": "medium",    "surgical_difficulty": 3},
    "RED_BLOOD_CELLS": {"output_dim": 4,  "complexity": "low",       "surgical_difficulty": 1},
})
ORGAN_META = SafeMeta(ORGAN_META)


# --- FIXED: sesi_lite PROTEIN_TYPE_PARAMS ---
@dataclass
class ProteinTypeParams:
    name: str
    friction: float
    temperature: float
    noise_amplitude: float
    bond_stiffness: float
    angle_stiffness: float


_PROTEIN_RAW = {
    "globular_alpha":      ("Globular α",        1.0, 300., 0.10, 100., 50.),
    "globular_beta":       ("Globular β",        1.2, 300., 0.10, 120., 60.),
    "globular_alpha_beta": ("Globular α/β",      1.0, 300., 0.10, 100., 50.),
    "membrane_alpha":      ("Membrane α-helix",  2.0, 310., 0.05, 150., 80.),
    "membrane_beta":       ("Membrane β-barrel", 2.5, 310., 0.05, 180., 100.),
    "fibrous_collagen":    ("Collagen",          0.5, 298., 0.20, 200., 120.),
    "fibrous_keratin":     ("Keratin",           0.5, 298., 0.20, 200., 120.),
    "idp":                 ("Disordered",        0.1, 300., 0.50,  20., 10.),
    "metalloprotein":      ("Metalloprotein",    1.5, 300., 0.08, 150., 80.),
    "antibody":            ("Antibody",          1.2, 300., 0.10, 120., 60.),
    "enzyme":              ("Enzyme",            1.0, 310., 0.10, 100., 50.),
    "default":             ("Default",           1.0, 300., 0.10, 100., 50.),
}
PROTEIN_TYPE_PARAMS: Dict[str, ProteinTypeParams] = SafeMeta({
    k.strip(): ProteinTypeParams(n, g, T, s, kb, ka)
    for k, (n, g, T, s, kb, ka) in _PROTEIN_RAW.items()
})


def resolve_protein_type(name: str) -> ProteinTypeParams:
    """Never raises KeyError for a whitespace variant."""
    key = (name or "default").strip()
    if key not in PROTEIN_TYPE_PARAMS:
        log.warning(f"[protein-type] '{name}' unknown → 'default'")
        key = "default"
    return PROTEIN_TYPE_PARAMS[key]


# =============================================================================
# 2. CLINICAL GATE — the honest fix for random-weight medical outputs
# =============================================================================

class ClinicalGateError(RuntimeError):
    pass


class ClinicalGate:
    """
    Blocks any output that could be read as a clinical prediction unless
    ALL of the following are supplied:
      • a trained checkpoint that actually exists on disk
      • a calibration report (ECE / coverage) from a held-out set
      • an explicit research-only acknowledgment
      • an ethics/IRB reference string

    This is deliberately inconvenient. A random-weight `success_probability`
    for brain surgery is not a bug to repair — it is a hazard to remove.
    """
    def __init__(self, ckpt: Optional[str] = None,
                 calibration: Optional[str] = None,
                 research_only: bool = False,
                 ethics_ref: Optional[str] = None):
        self.ckpt, self.calibration = ckpt, calibration
        self.research_only, self.ethics_ref = research_only, ethics_ref

    def check(self, what: str) -> None:
        missing = []
        if not self.ckpt or not Path(self.ckpt).exists():
            missing.append("trained checkpoint (--ckpt) that exists")
        if not self.calibration or not Path(self.calibration).exists():
            missing.append("held-out calibration report (--calibration)")
        if not self.research_only:
            missing.append("--research-only acknowledgment")
        if not self.ethics_ref:
            missing.append("--ethics-ref (IRB / approval reference)")
        if missing:
            raise ClinicalGateError(
                f"'{what}' is BLOCKED. Missing:\n  - " + "\n  - ".join(missing) +
                "\n\nThis head previously emitted numbers from randomly initialised "
                "weights. Those numbers were not predictions. They are removed, "
                "not repaired.\n"
                "Required before any clinical-adjacent use: real data, prospective "
                "validation, regulatory clearance, and clinical oversight.")


class RemovedSurgicalPlanningHead(nn.Module):
    """
    REPLACEMENT for SurgicalPlanningEngine.
    Emits nothing. Raises on every call. There is no 'untrained mode'.
    """
    def forward(self, *a, **k):
        raise ClinicalGateError(
            "SurgicalPlanningEngine has been REMOVED.\n"
            "It produced incision_path, risk_assessment and success_probability "
            "from untrained weights. For brain/heart surgery those values are "
            "dangerous regardless of how they are labelled.\n"
            "No replacement is provided. Surgical planning is a regulated "
            "clinical activity, not a model head.")


class GatedBrainDecodeHead(nn.Module):
    """
    REPLACEMENT for the 30,000-word open-vocabulary decoder.

    Changes:
      • open 30k vocab → small CLOSED label set (default 10)
      • requires ClinicalGate to pass before returning anything
      • returns logits + explicit untrained flag, never a confident 'thought'
    """
    def __init__(self, d_in: int, n_labels: int = 10,
                 gate: Optional[ClinicalGate] = None):
        super().__init__()
        assert n_labels <= 64, "closed-set only; open-vocab EEG decoding is not supported"
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 128),
                                 nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(128, n_labels))
        self.gate = gate or ClinicalGate()
        self.n_labels = n_labels

    def forward(self, x: Tensor) -> Dict[str, Any]:
        self.gate.check("brain decoding")
        logits = self.net(x)
        return {"logits": logits,
                "n_labels": self.n_labels,
                "NOTICE": ("Closed-set statistical inference from noisy neural "
                           "signals. Not telepathy. Requires independent "
                           "validation on the target subject and session.")}


# =============================================================================
# 3. REAL QUANTUM CHEMISTRY  (replaces nn.Parameter(torch.randn) basis)
# =============================================================================

class RealQCService:
    """
    PySCF-computed integrals. NOT differentiable — this is an OFFLINE labeler.
    The differentiable object is the surrogate trained on these labels.
    """
    def __init__(self, basis: str = "sto-3g", method: str = "mp2",
                 cache_dir: str = "cache/qc"):
        try:
            from real_qc_lg import QCConfig, pyscf_reference, HAS_PYSCF
        except ImportError as e:
            raise ImportError("real_qc_lg.py must be importable") from e
        if not HAS_PYSCF:
            raise ImportError("pip install pyscf")
        self._ref = pyscf_reference
        self.cfg = QCConfig(basis=basis, method=method, compute_forces=True)
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)

    def label(self, symbols: List[str], coords: np.ndarray) -> Optional[Dict]:
        import hashlib
        key = hashlib.sha256(json.dumps(
            [symbols, np.round(coords, 6).tolist(),
             self.cfg.basis, self.cfg.method], sort_keys=True).encode()).hexdigest()[:16]
        f = self.cache / f"{key}.json"
        if f.exists():
            return json.load(open(f))
        r = self._ref(symbols, np.asarray(coords), self.cfg)
        if r is None:
            return None
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        out.update(symbols=symbols, coords=np.asarray(coords).tolist())
        json.dump(out, open(f, "w"))
        return out


class QCSurrogate(nn.Module):
    """
    Drop-in for the old QuantumChemistryEngine.

    ⚠️ REQUIRES NUCLEAR GEOMETRY. A latent embedding does not specify a
       molecule: it carries no nuclear charges and no coordinates.
       The old module accepted an embedding and returned a number. That
       number was not an energy.
    """
    def __init__(self, elements: List[str], ckpt: Optional[str] = None,
                 basis: str = "sto-3g"):
        super().__init__()
        try:
            from real_qc_lg import QCConfig, NeuralPotential
        except ImportError as e:
            raise ImportError("real_qc_lg.py must be importable") from e
        self.elements = sorted(set(elements))
        self.eidx = {e: i for i, e in enumerate(self.elements)}
        cfg = QCConfig(basis=basis)
        self.model = NeuralPotential(cfg, self.elements)
        self.trained = False
        if ckpt and Path(ckpt).exists():
            blob = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict(blob["model"])
            self.trained = True
            log.info(f"[QC] loaded surrogate: val MAE "
                     f"{blob.get('val_mae_kcal_per_atom', float('nan')):.4f} kcal/mol/atom")
        else:
            log.warning("[QC] NO trained weights — forward() will raise, not guess")

    def forward(self, symbols: List[List[str]], coords: Tensor,
                mask: Optional[Tensor] = None,
                embedding: Optional[Tensor] = None, **_) -> Dict[str, Tensor]:
        if embedding is not None and (symbols is None or coords is None):
            raise ValueError(
                "Real QC needs (symbols, coords). An embedding does not "
                "determine nuclear charges or geometry.")
        if not self.trained:
            raise ClinicalGateError(
                "QC surrogate is untrained. Run: real_qc_lg.py qc-label → qc-train.")
        dev = coords.device
        B, N, _ = coords.shape
        Z = torch.tensor([[self.eidx[s] for s in row] for row in symbols],
                         dtype=torch.long, device=dev)
        M = mask if mask is not None else torch.ones(B, N, device=dev)
        out = self.model(Z, coords, M, want_forces=True)
        return {"energy_hartree": out["energy"], "forces": out["forces"],
                "_surrogate": True, "_trained": True}


# =============================================================================
# 4. REAL LATTICE GAUGE  (replaces random links)
# =============================================================================

class RealLatticeService:
    """Loads Monte-Carlo SU(3) configurations. Not a random-parameter module."""
    def __init__(self, config_dir: str):
        self.dir = Path(config_dir)
        self.files = sorted(self.dir.glob("cfg_*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No MC configs in {self.dir}. Generate them:\n"
                f"  python lattice_heatbath_rk3.py validate\n"
                f"  python lattice_heatbath_rk3.py generate --L 8 --beta 6.0")
        log.info(f"[LGT] {len(self.files)} MC configurations found")

    def load(self, i: int) -> Dict[str, Any]:
        d = np.load(self.files[i])
        return {k: d[k] for k in d.files}

    def as_tensor(self, i: int, device=None) -> Tensor:
        d = self.load(i)
        U = torch.from_numpy(d["links"].astype(np.complex64))
        D = U.shape[0]; N = U.shape[1]
        L = int(round(N ** (1 / D)))
        assert L ** D == N, f"lattice size mismatch: N={N}, D={D}"
        return U.reshape(1, D, *([L] * D), 3, 3).to(device)


def wilson_action_exact(links: Tensor, beta: float) -> Dict[str, Tensor]:
    """
    EXACT, differentiable Wilson action from supplied links.
    No training, no surrogate — this is a closed-form formula.
    links: (B, D, L,...,L, 3, 3) complex
    """
    B, D = links.shape[0], links.shape[1]
    L = links.shape[2]

    def shift(x, mu, n=1):
        return torch.roll(x, shifts=-n, dims=2 + mu)

    def dag(x):
        return x.conj().transpose(-1, -2)

    total = torch.zeros(B, device=links.device, dtype=torch.float32)
    plaq = []
    for mu in range(D):
        for nu in range(mu + 1, D):
            Um, Un = links[:, mu], links[:, nu]
            P = Um @ shift(Un, mu) @ dag(shift(Um, nu)) @ dag(Un)
            tr = torch.diagonal(P, dim1=-2, dim2=-1).real.sum(-1) / 3.0
            m = tr.mean(dim=tuple(range(1, tr.ndim)))
            plaq.append(m)
            total = total + (1.0 - m) * (L ** D)
    return {"wilson_action": beta * total,
            "plaquette_mean": torch.stack(plaq, -1).mean(-1),
            "_exact": True}


# =============================================================================
# 5. REAL TRAINER — logs, checkpoints, held-out test
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    lr: float = 5e-4
    weight_decay: float = 1e-6
    force_weight: float = 100.0
    n_train: int = 950
    n_val: int = 50
    grad_clip: float = 10.0
    seed: int = 42
    precision: str = "fp32"     # fp16 + double-backward is unsafe


class RealTrainer:
    """
    Writes:
      runs/<name>/log.jsonl     — every epoch, machine-readable
      runs/<name>/best.pt       — selected on VALIDATION only
      runs/<name>/test.json     — evaluated ONCE, after selection
    """
    def __init__(self, name: str, cfg: TrainConfig, out_root: str = "runs"):
        self.cfg = cfg
        self.dir = Path(out_root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.dir / "log.jsonl", "a")
        json.dump(asdict(cfg), open(self.dir / "config.json", "w"), indent=2)
        self.best = float("inf")
        self.best_state = None
        self.t0 = time.time()

    def log(self, rec: Dict):
        rec["t"] = round(time.time() - self.t0, 2)
        self.logf.write(json.dumps(rec) + "\n"); self.logf.flush()

    def save_best(self, model: nn.Module, val: float, epoch: int):
        if val < self.best:
            self.best = val
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            torch.save({"model": self.best_state, "epoch": epoch,
                        "val_metric": val, "config": asdict(self.cfg)},
                       self.dir / "best.pt")
            return True
        return False

    def finalize(self, model: nn.Module, test_metrics: Dict):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        payload = {"best_val": self.best, "test": test_metrics,
                   "config": asdict(self.cfg),
                   "note": "test evaluated once, after validation-based selection"}
        json.dump(payload, open(self.dir / "test.json", "w"), indent=2)
        self.logf.close()
        log.info(f"[train] artifacts → {self.dir}")
        return payload


def train_rmd17(args):
    """Real training on real DFT energies + forces, with real artifacts."""
    try:
        from force_datasets import (load_rmd17, to_records, collate,
                                    PaiNNF, SchNetF, fit_sae)
    except ImportError as e:
        log.error("force_datasets.py must be importable"); raise
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      n_train=args.n_train, n_val=args.n_val,
                      precision=args.precision)
    if cfg.precision == "fp16":
        log.warning("fp16 + create_graph=True is unsafe → fp32")
        cfg.precision = "fp32"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    d = load_rmd17(Path(args.data), args.molecule, limit=args.limit)
    recs = to_records(d)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(recs))
    ntr = min(cfg.n_train, int(0.8 * len(recs)))
    nva = min(cfg.n_val, int(0.1 * len(recs)))
    tr = [recs[i] for i in perm[:ntr]]
    va = [recs[i] for i in perm[ntr:ntr + nva]]
    te = [recs[i] for i in perm[ntr + nva:]]
    log.info(f"split train/val/test = {len(tr)}/{len(va)}/{len(te)}  seed={cfg.seed}")

    nmax = max(r["n_atoms"] for r in tr)
    Zt = np.zeros((len(tr), nmax), np.int64)
    Mt = np.zeros((len(tr), nmax), np.float32)
    Et = np.array([r["E"] for r in tr], np.float32)
    for i, r in enumerate(tr):
        k = r["n_atoms"]; Zt[i, :k] = r["z"]; Mt[i, :k] = 1.
    Model = PaiNNF if args.arch == "painn" else SchNetF
    model = Model(128, 64, 6, 5.0).to(dev)
    sae, _ = fit_sae(Zt, Et, Mt)                    # train only, no leakage
    model.sae.copy_(torch.from_numpy(sae))
    X = np.stack([((Zt == e) & (Mt > 0)).sum(1) for e in range(7)], 1)
    model.res_std.fill_(max(float(((Et - X @ sae) / Mt.sum(1)).std()), 1e-6))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    run = RealTrainer(f"rmd17_{args.molecule}_{args.arch}", cfg)
    use_bf16 = cfg.precision == "bf16" and dev.type == "cuda"

    def step(batch, training):
        R = batch["R"].detach().requires_grad_(True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=use_bf16):
            E = model(batch["z"], R, batch["batch"], batch["ng"])
        E = E.float()
        Fp = -torch.autograd.grad(E.sum(), R, create_graph=training,
                                  retain_graph=True)[0]
        lE = ((E - batch["E"]) / batch["n_atoms"]).pow(2).mean()
        lF = (Fp - batch["F"]).pow(2).sum(-1).mean()
        return lE, lF, E, Fp

    EV2KCAL = 23.060547830619
    for ep in range(cfg.epochs):
        model.train(); order = rng.permutation(len(tr)); tot = 0.; nb = 0
        for s in range(0, len(order), cfg.batch_size):
            b = collate([tr[i] for i in order[s:s + cfg.batch_size]], dev)
            lE, lF, _, _ = step(b, True)
            loss = lE + cfg.force_weight * lF
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        model.eval(); eE, eF = [], []
        for s in range(0, len(va), cfg.batch_size):
            b = collate(va[s:s + cfg.batch_size], dev)
            _, _, Ep, Fp = step(b, False)
            eE.extend(((Ep - b["E"]) / b["n_atoms"]).abs().detach().cpu().numpy())
            eF.extend((Fp - b["F"]).abs().mean(-1).detach().cpu().numpy())
        vE = float(np.mean(eE)) * EV2KCAL
        vF = float(np.mean(eF)) * EV2KCAL
        improved = run.save_best(model, vF, ep)
        run.log({"epoch": ep + 1, "train_loss": tot / max(nb, 1),
                 "val_E_kcal_per_atom": vE, "val_F_kcal_per_A": vF,
                 "lr": sch.get_last_lr()[0], "best": improved})
        if (ep + 1) % max(1, cfg.epochs // 10) == 0:
            log.info(f"ep {ep+1:4d} | val E {vE:.4f} kcal/mol/atom | "
                     f"F {vF:.4f} kcal/mol/Å {'*' if improved else ''}")

    model.eval(); tE, tF = [], []
    for s in range(0, len(te), cfg.batch_size):
        b = collate(te[s:s + cfg.batch_size], dev)
        _, _, Ep, Fp = step(b, False)
        tE.extend(((Ep - b["E"]) / b["n_atoms"]).abs().detach().cpu().numpy())
        tF.extend((Fp - b["F"]).abs().mean(-1).detach().cpu().numpy())
    tm = {"E_MAE_kcal_per_atom": float(np.mean(tE)) * EV2KCAL,
          "F_MAE_kcal_per_A": float(np.mean(tF)) * EV2KCAL,
          "n_test": len(te), "level_of_theory": d.get("level", "PBE/def2-SVP")}
    out = run.finalize(model, tm)
    print("\n" + "=" * 72)
    print(f"  TEST — rMD17/{args.molecule}  {args.arch}")
    print("=" * 72)
    print(f"  E MAE : {tm['E_MAE_kcal_per_atom']:.4f} kcal/mol/atom")
    print(f"  F MAE : {tm['F_MAE_kcal_per_A']:.4f} kcal/mol/Å")
    print(f"  n_test: {tm['n_test']}   split seed {cfg.seed}")
    print(f"  artifacts: {run.dir}/ (log.jsonl, best.pt, test.json)")
    print("\n  These are real numbers on a held-out split of real DFT data.")
    print("  They are NOT a claim of beating published baselines — compare")
    print("  against the specific paper's table AND matching split first.")
    print("=" * 72)
    return out


# =============================================================================
# 6. SELF-TEST
# =============================================================================
def selftest(args) -> bool:
    print("=" * 74); print("  REAL BRIDGE — self-test"); print("=" * 74)
    ok = True

    print("\n[T1] ORGAN_META keys (was: KeyError on all 14 organs)")
    try:
        for name in ["KIDNEY", "BRAIN", "HEART", "RED_BLOOD_CELLS"]:
            m = ORGAN_META[name]
            _ = m["output_dim"], m["surgical_difficulty"]
        legacy = SafeMeta(normalize_keys({"output_dim ": 8}))
        _ = legacy["output_dim"]
        print(f"   {len(ORGAN_META)} organs, legacy-space key tolerated  PASS")
    except Exception as e:
        print(f"   FAIL: {e}"); ok = False

    print("\n[T2] PROTEIN_TYPE_PARAMS (was: KeyError on 'default')")
    try:
        for n in ["default", "default ", "idp", "membrane_alpha", "nonexistent"]:
            p = resolve_protein_type(n)
        print(f"   {len(PROTEIN_TYPE_PARAMS)} types, unknown→default  PASS")
    except Exception as e:
        print(f"   FAIL: {e}"); ok = False

    print("\n[T3] Surgical head is REMOVED, not repaired")
    try:
        RemovedSurgicalPlanningHead()(torch.randn(2, 8))
        print("   FAIL — should have raised"); ok = False
    except ClinicalGateError as e:
        print(f"   correctly refused: {str(e).splitlines()[0]}  PASS")

    print("\n[T4] Brain decoder gated + closed-set")
    head = GatedBrainDecodeHead(64, n_labels=10)
    try:
        head(torch.randn(2, 64))
        print("   FAIL — should have raised"); ok = False
    except ClinicalGateError:
        print("   untrained → blocked  PASS")
    try:
        GatedBrainDecodeHead(64, n_labels=30000)
        print("   FAIL — open vocab should be rejected"); ok = False
    except AssertionError:
        print("   30k open vocab → rejected  PASS")

    print("\n[T5] QC refuses bare embedding")
    try:
        from real_qc_lg import HAS_PYSCF
        qc = QCSurrogate(["H", "O"])
        qc(None, None, embedding=torch.randn(1, 64))
        print("   FAIL — should have raised"); ok = False
    except ImportError:
        print("   SKIP (real_qc_lg.py not importable)")
    except (ValueError, ClinicalGateError) as e:
        print(f"   correctly refused: {str(e).splitlines()[0]}  PASS")

    print("\n[T6] Exact Wilson action on identity links → plaquette = 1")
    D, L = 4, 4
    U = torch.eye(3, dtype=torch.complex64)[None, None, None].expand(
        1, D, *([L] * D), 3, 3).clone()
    r = wilson_action_exact(U, beta=6.0)
    p = r["plaquette_mean"].item()
    t = abs(p - 1.0) < 1e-5
    print(f"   <plaq> = {p:.8f}  (expect 1.0)  {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T7] Trainer writes log / checkpoint / test")
    run = RealTrainer("_selftest", TrainConfig(epochs=1))
    lin = nn.Linear(4, 1)
    run.log({"epoch": 1, "val": 0.5})
    run.save_best(lin, 0.5, 1)
    run.finalize(lin, {"dummy": 1.0})
    files = {f.name for f in run.dir.iterdir()}
    t = {"log.jsonl", "best.pt", "test.json", "config.json"} <= files
    print(f"   {sorted(files)}  {'PASS' if t else 'FAIL'}"); ok &= t
    import shutil; shutil.rmtree(run.dir, ignore_errors=True)

    print("\n" + "=" * 74)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME CHECKS FAILED'}")
    print("=" * 74)
    print("""
  WHAT IS NOW TRUE
    • dict-key crashes fixed, with defensive fallback
    • surgical planning removed (not "made to work")
    • brain decoding gated + closed-set only
    • QC requires geometry and trained weights
    • LGT uses real MC configs; Wilson action is exact + differentiable
    • training produces log.jsonl, best.pt, test.json

  WHAT IS STILL TRUE
    • Fixing code ≠ creating capability.
    • No clinical or diagnostic output exists in this file.
    • Untrained models raise instead of emitting plausible numbers.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "train-rmd17", "lgt-check"])
    p.add_argument("--data", default="data/rmd17")
    p.add_argument("--molecule", default="benzene")
    p.add_argument("--arch", choices=["painn", "schnet"], default="painn")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    p.add_argument("--n-train", dest="n_train", type=int, default=950)
    p.add_argument("--n-val", dest="n_val", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--configs", default="data/lg")
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    elif a.cmd == "train-rmd17":
        train_rmd17(a)
    else:
        svc = RealLatticeService(a.configs)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        U = svc.as_tensor(0, dev)
        r = wilson_action_exact(U, beta=6.0)
        print(f"MC config 0: <plaq>={r['plaquette_mean'].item():.6f} "
              f"S_W={r['wilson_action'].item():.3f}")


if __name__ == "__main__":
    main()
