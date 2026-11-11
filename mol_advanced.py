#!/usr/bin/env python3
"""
=============================================================================
Δ-LEARNING (ANI-1ccx → ANI-1x) · ENSEMBLE UQ · CONSERVATIVE-FORCE VALIDATION
=============================================================================
License: MIT | 2026
Requires: force_datasets.py (from previous turn) for PaiNNF/SchNetF/loaders

LITERATURE
 [D1] Smith et al. "The ANI-1ccx and ANI-1x data sets..." Sci.Data 7,134 (2020)
 [D2] Ramakrishnan et al. "Quantum chemistry structures and properties of
      134 kilo molecules." Sci.Data 1,140022 (2014)          [QM9]
 [Δ1] Ramakrishnan, Dral, Rupp, von Lilienfeld. "Big data meets quantum
      chemistry approximations: the Δ-machine learning approach."
      JCTC 11, 2087 (2015).
 [Δ2] Smith et al. "Approaching coupled cluster accuracy with a general-purpose
      neural network potential through transfer learning." Nat.Commun.10,2903(2019)
 [U1] Lakshminarayanan, Pritzel, Blundell. "Simple and scalable predictive
      uncertainty estimation using deep ensembles." NeurIPS 30 (2017).
 [U2] Hirschfeld et al. "Uncertainty quantification using neural networks for
      molecular property prediction." JCIM 60, 3770 (2020).
 [C1] Christensen & von Lilienfeld. MLST 1, 045018 (2020).   [rMD17]
 [C2] Chmiela et al. "Towards exact molecular dynamics simulations with
      machine-learned force fields." Nat.Commun. 9, 3887 (2018).  [sGDML]

=============================================================================
HONEST SCOPE — THREE THINGS THAT ARE OFTEN GOTTEN WRONG
=============================================================================
1. ANI-1ccx HAS ENERGIES ONLY (CCSD(T)*/CBS). NO FORCES.
   → Δ-learning corrects ENERGIES. Forces remain at the ωB97X/6-31G(d) level.
   → You may NOT claim "CCSD(T) forces" from this pipeline.
   → The correct statement: "CCSD(T)-quality energies with DFT-quality forces."

2. ENSEMBLE SPREAD IS NOT AN ERROR BAR UNTIL CALIBRATED.
   → We report z-score coverage, ECE, and Spearman(σ, |error|).
   → If Spearman ≈ 0, the ensemble is confident but useless as an uncertainty.
   → Ensembles capture EPISTEMIC spread only, not label noise (aleatoric).

3. CURL-FREE IS A CONSTRUCTION, NOT A DISCOVERY.
   → If F = -autograd(E, R), then ∂F_i/∂x_j = -∂²E/∂x_i∂x_j is symmetric,
     so curl ≡ 0 to machine precision. The test verifies the IMPLEMENTATION.
   → A separate force head is generally NOT conservative. We measure both.
   → Reporting "our forces are conservative" without this distinction is
     misleading; reporting curl≈0 for autograd forces is a unit test, not
     a scientific result.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import h5py; HAS_H5 = True
except ImportError:
    HAS_H5 = False

# reuse from previous turn
try:
    from force_datasets import (
        PaiNNF, SchNetF, ANI1xLazy, load_rmd17, to_records, collate,
        fit_sae, unit_sanity, HARTREE_EV, HA_BOHR_EV_A, EV_KCALMOL, ELEMENTS,
    )
    HAS_BASE = True
except ImportError:
    HAS_BASE = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("mol_adv")


# =============================================================================
# 1. Δ-LEARNING:  CCSD(T)*/CBS  −  ωB97X/6-31G(d)
# =============================================================================

CCX_KEYS = ["ccsd(t)_cbs.energy", "ccsd(t)_cbs.energy", "ccsd(t).energy"]
DFT_KEYS = ["wb97x_dz.energy", "wb97x_tz.energy"]


class DeltaPairs:
    """
    Extract MATCHED (E_low, E_high) pairs from ANI-1ccx.
    Both levels live on the SAME geometry — this is what makes Δ-learning
    well-posed [Δ1,Δ2]. No geometry alignment needed.
    """
    def __init__(self, path: Path, max_systems: Optional[int] = None):
        if not HAS_H5:
            raise ImportError("pip install h5py")
        self.path = Path(path)
        self.index: List[Tuple[str, int]] = []
        self.hi_key = self.lo_key = None
        self._build(max_systems)

    def _build(self, max_systems):
        nsys = 0
        with h5py.File(self.path, "r") as f:
            def visit(name, obj):
                nonlocal nsys
                if max_systems and nsys >= max_systems: return
                if not isinstance(obj, h5py.Group): return
                ks = list(obj.keys())
                hi = next((k for k in CCX_KEYS if k in ks), None)
                lo = next((k for k in DFT_KEYS if k in ks), None)
                if hi is None or lo is None: return
                if "coordinates" not in ks or "species" not in ks: return
                Eh, El = obj[hi], obj[lo]
                if Eh.ndim == 0 or len(Eh) == 0: return
                if len(Eh) != len(El):
                    log.warning(f"{name}: length mismatch, skipping"); return
                self.hi_key, self.lo_key = hi, lo
                for i in range(len(Eh)):
                    self.index.append((name, i))
                nsys += 1
            f.visititems(visit)
        if not self.index:
            raise RuntimeError(
                f"No matched (CCSD(T), DFT) pairs in {self.path}.\n"
                f"Run: python force_datasets.py probe --path {self.path}\n"
                f"Expected keys: {CCX_KEYS} and {DFT_KEYS}")
        log.info(f"[Δ] {len(self.index)} matched pairs from {nsys} systems")
        log.info(f"[Δ] high='{self.hi_key}'  low='{self.lo_key}'")

    def __len__(self): return len(self.index)

    def get(self, i: int) -> Dict:
        g, j = self.index[i]
        with h5py.File(self.path, "r") as f:
            grp = f[g]
            sp = np.asarray(grp["species"]).reshape(-1)
            R = np.asarray(grp["coordinates"][j], dtype=np.float64)
            Eh = float(np.asarray(grp[self.hi_key][j]))
            El = float(np.asarray(grp[self.lo_key][j]))
        n = R.shape[0]
        sp = sp[:n]
        Z2I = {1:0, 6:1, 7:2, 8:3, 9:4, 16:5, 17:6}
        if sp.dtype.kind in "SUO":
            sym = {"H":1,"C":6,"N":7,"O":8,"F":9,"S":16,"Cl":17}
            Zs = [sym.get(str(s).strip("b'\" "), 0) for s in sp]
        else:
            Zs = sp.astype(int).tolist()
        if any(z not in Z2I for z in Zs):
            raise ValueError(f"unsupported elements {Zs}")
        return {
            "z": np.array([Z2I[z] for z in Zs], dtype=np.int64),
            "R": R.astype(np.float32),
            "E_low": np.float32(El * HARTREE_EV),      # eV
            "E_high": np.float32(Eh * HARTREE_EV),     # eV
            "delta": np.float32((Eh - El) * HARTREE_EV),
            "n_atoms": n,
            "F": np.zeros_like(R, dtype=np.float32),   # ccx has NO forces
            "E": np.float32(El * HARTREE_EV),
        }

    def materialize(self, n: int, seed: int = 42) -> List[Dict]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=min(n, len(self)), replace=False)
        out = []
        for i in idx:
            try: out.append(self.get(int(i)))
            except Exception as e: log.warning(f"skip {i}: {e}")
        return out


class DeltaModel(nn.Module):
    """
    E_pred = E_base(ωB97X)  +  Δ_θ(geometry)
    Forces = -∇E_base   (DFT-level; Δ head is NOT differentiated for forces)

    If you want Δ-corrected forces you must either
      (a) differentiate the Δ head too — but then you have NO force labels
          to validate against, or
      (b) obtain CCSD(T) forces elsewhere (not in public ANI-1ccx).
    We expose `delta_forces=False` by default and warn if enabled.
    """
    def __init__(self, base: nn.Module, nf=128, nrbf=64, rc=5.0, ne=7,
                 delta_forces: bool = False):
        super().__init__()
        self.base = base
        self.delta_forces = delta_forces
        # separate small equivariant trunk for Δ
        self.delta_trunk = PaiNNF(nf//2, nrbf//2, 3, rc, ne) if HAS_BASE else None
        if self.delta_trunk is None:
            raise ImportError("force_datasets.py required")
        self.delta_scale = nn.Parameter(torch.tensor(1.0))
        if delta_forces:
            log.warning("[Δ] delta_forces=True: forces will include the Δ head, "
                        "but there are NO CCSD(T) force labels to validate them. "
                        "Do not claim coupled-cluster forces.")

    def forward(self, z, R, batch, ng, want_forces=True):
        if want_forces and not R.requires_grad:
            R = R.detach().requires_grad_(True)
        E_base = self.base(z, R, batch, ng)
        d = self.delta_trunk(z, R, batch, ng) * self.delta_scale
        E = E_base + d
        forces = None
        if want_forces:
            src = E if self.delta_forces else E_base
            forces = -torch.autograd.grad(src.sum(), R,
                                          create_graph=self.training,
                                          retain_graph=True)[0]
        return {"E": E, "E_base": E_base, "delta": d, "forces": forces}


def train_delta(args):
    if not HAS_BASE:
        log.error("force_datasets.py must be importable"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    pairs = DeltaPairs(Path(args.ccx), max_systems=args.max_systems)
    recs = pairs.materialize(args.n_samples, seed=args.seed)
    if len(recs) < 20: log.error("too few pairs"); return

    d = np.array([r["delta"] for r in recs])
    na = np.array([r["n_atoms"] for r in recs])
    log.info(f"[Δ] target statistics (eV): mean={d.mean():.5f} std={d.std():.5f}")
    log.info(f"[Δ] per-atom (eV/atom): mean={(d/na).mean():.6f} "
             f"std={(d/na).std():.6f}  = {(d/na).std()*EV_KCALMOL:.4f} kcal/mol/atom")
    log.info(f"[Δ] |Δ| range: {np.abs(d).min():.5f} – {np.abs(d).max():.5f} eV")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(recs))
    ntr = int(0.8*len(recs)); nva = int(0.1*len(recs))
    tr = [recs[i] for i in perm[:ntr]]
    va = [recs[i] for i in perm[ntr:ntr+nva]]
    te = [recs[i] for i in perm[ntr+nva:]]
    log.info(f"[Δ] split {len(tr)}/{len(va)}/{len(te)} seed={args.seed}")

    base = PaiNNF(128, 64, 6, 5.0).to(dev)
    # base SAE from low-level energies (train only)
    nmax = max(r["n_atoms"] for r in tr)
    Zt = np.zeros((len(tr), nmax), np.int64); Mt = np.zeros((len(tr), nmax), np.float32)
    Et = np.array([r["E_low"] for r in tr], np.float32)
    for i, r in enumerate(tr):
        k = r["n_atoms"]; Zt[i,:k] = r["z"]; Mt[i,:k] = 1.
    sae, _ = fit_sae(Zt, Et, Mt)
    base.sae.copy_(torch.from_numpy(sae))
    X = np.stack([((Zt==e)&(Mt>0)).sum(1) for e in range(7)],1)
    base.res_std.fill_(max(float(((Et - X@sae)/Mt.sum(1)).std()), 1e-6))

    model = DeltaModel(base, delta_forces=args.delta_forces).to(dev)
    model.delta_trunk.sae.zero_()
    model.delta_trunk.res_std.fill_(max(float((d/na).std()), 1e-6))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr)); tot=0.; nb=0
        for s in range(0, len(order), args.batch_size):
            b = collate([tr[i] for i in order[s:s+args.batch_size]], dev)
            tgt = torch.tensor([tr[i]["delta"] for i in order[s:s+args.batch_size]],
                               device=dev)
            opt.zero_grad(set_to_none=True)
            out = model(b["z"], b["R"], b["batch"], b["ng"], want_forces=False)
            loss = F.mse_loss(out["delta"], tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.detach()); nb += 1
        if (ep+1) % max(1,args.epochs//10) == 0:
            model.eval(); errs=[]; base_errs=[]
            with torch.no_grad():
                for s in range(0,len(va),args.batch_size):
                    b = collate(va[s:s+args.batch_size], dev)
                    o = model(b["z"], b["R"], b["batch"], b["ng"], want_forces=False)
                    th = torch.tensor([r["E_high"] for r in va[s:s+args.batch_size]],
                                      device=dev)
                    errs.extend(((o["E"]-th)/b["n_atoms"]).abs().cpu().numpy())
                    base_errs.extend(((o["E_base"]-th)/b["n_atoms"]).abs().cpu().numpy())
            m = np.mean(errs)*EV_KCALMOL; bm = np.mean(base_errs)*EV_KCALMOL
            log.info(f"ep {ep+1:4d} | loss {tot/max(nb,1):.6f} | "
                     f"val Δ-corrected {m:.4f} vs base {bm:.4f} kcal/mol/atom "
                     f"({100*(1-m/max(bm,1e-9)):+.1f}%)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    log.info(f"[Δ] saved → {args.out}")
    log.info("[Δ] REMINDER: energies are CCSD(T)-corrected; FORCES ARE NOT.")


# =============================================================================
# 2. ENSEMBLE + CALIBRATED UNCERTAINTY
# =============================================================================

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra**2).sum() * (rb**2).sum())) + 1e-12
    return float((ra*rb).sum()/den)


def calibration_report(sigma: np.ndarray, err: np.ndarray) -> Dict:
    """
    [U1,U2] An ensemble is only useful if σ tracks |error|.
      z = err/σ  should be ~N(0,1) if calibrated
      coverage(|z|<1) ≈ 0.6827, (|z|<2) ≈ 0.9545
      Spearman(σ,|err|) ≈ 0 → σ carries NO information about error
    """
    s = np.maximum(sigma, 1e-9)
    z = err / s
    absz = np.abs(z)
    # expected calibration error over nominal coverage levels
    levels = np.linspace(0.05, 0.95, 19)
    ece = 0.0
    curve = []
    for p in levels:
        from math import erf, sqrt
        zc = sqrt(2)*_erfinv(p)
        obs = float((absz < zc).mean())
        ece += abs(obs - p)
        curve.append((float(p), obs))
    ece /= len(levels)
    return {
        "n": int(len(err)),
        "mean_sigma": float(s.mean()),
        "mean_abs_err": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "z_mean": float(z.mean()), "z_std": float(z.std()),
        "coverage_1sigma": float((absz<1).mean()),
        "coverage_2sigma": float((absz<2).mean()),
        "nominal_1sigma": 0.6827, "nominal_2sigma": 0.9545,
        "ECE": float(ece),
        "spearman_sigma_vs_abserr": spearman(s, np.abs(err)),
        "calibration_curve": curve,
        "verdict": (
            "USEFUL" if spearman(s, np.abs(err)) > 0.3 else
            "WEAK — σ barely tracks error; do not use as an error bar"
        ),
    }


def _erfinv(x: float) -> float:
    # Winitzki approximation, adequate for calibration curve
    a = 0.147
    ln = math.log(1 - x*x + 1e-12)
    t1 = 2/(math.pi*a) + ln/2
    return math.copysign(math.sqrt(max(math.sqrt(t1*t1 - ln/a) - t1, 0)), x)


def train_ensemble(args):
    if not HAS_BASE: log.error("force_datasets.py required"); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "rmd17":
        d = load_rmd17(Path(args.data), args.molecule, limit=args.limit)
        recs = to_records(d)
    else:
        lazy = ANI1xLazy(Path(args.data), max_systems=args.max_systems)
        recs = to_records(lazy.materialize(args.n_samples, seed=args.seed))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(recs))
    ntr = min(args.n_train, int(0.8*len(recs))); nva = int(0.1*len(recs))
    tr=[recs[i] for i in perm[:ntr]]; va=[recs[i] for i in perm[ntr:ntr+nva]]
    te=[recs[i] for i in perm[ntr+nva:]]
    log.info(f"[ens] {len(tr)}/{len(va)}/{len(te)}  M={args.n_models}")

    models = []
    for m in range(args.n_models):
        seed = args.seed + 1000*m
        torch.manual_seed(seed)
        net = PaiNNF(128, 64, 6, 5.0).to(dev)
        # SAE from this member's bootstrap sample
        idx = (rng.choice(len(tr), len(tr), replace=True)
               if args.bootstrap else np.arange(len(tr)))
        sub = [tr[i] for i in idx]
        nmax = max(r["n_atoms"] for r in sub)
        Zt=np.zeros((len(sub),nmax),np.int64); Mt=np.zeros((len(sub),nmax),np.float32)
        Et=np.array([r["E"] for r in sub],np.float32)
        for i,r in enumerate(sub):
            k=r["n_atoms"]; Zt[i,:k]=r["z"]; Mt[i,:k]=1.
        sae,_ = fit_sae(Zt,Et,Mt); net.sae.copy_(torch.from_numpy(sae))
        X=np.stack([((Zt==e)&(Mt>0)).sum(1) for e in range(7)],1)
        net.res_std.fill_(max(float(((Et-X@sae)/Mt.sum(1)).std()),1e-6))
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
        for ep in range(args.epochs):
            net.train(); order = rng.permutation(len(sub))
            for s in range(0,len(order),args.batch_size):
                b = collate([sub[i] for i in order[s:s+args.batch_size]], dev)
                R = b["R"].detach().requires_grad_(True)
                E = net(b["z"], R, b["batch"], b["ng"])
                Fp = -torch.autograd.grad(E.sum(), R, create_graph=True)[0]
                lE = ((E-b["E"])/b["n_atoms"]).pow(2).mean()
                lF = (Fp-b["F"]).pow(2).sum(-1).mean()
                loss = lE + args.force_weight*lF
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        log.info(f"[ens] member {m+1}/{args.n_models} done (seed={seed})")
        models.append(net.eval())

    # ensemble prediction on test
    Es, Fs = [], []
    for m in models:
        e_, f_ = [], []
        for s in range(0,len(te),args.batch_size):
            b = collate(te[s:s+args.batch_size], dev)
            R = b["R"].detach().requires_grad_(True)
            E = m(b["z"], R, b["batch"], b["ng"])
            Fp = -torch.autograd.grad(E.sum(), R, create_graph=False)[0]
            e_.append((E/b["n_atoms"]).detach().cpu().numpy())
            f_.append(Fp.detach().cpu().numpy())
        Es.append(np.concatenate(e_)); Fs.append(np.concatenate(f_))
    Es = np.stack(Es); Fs = np.stack(Fs)
    E_mu, E_sd = Es.mean(0), Es.std(0)
    F_mu, F_sd = Fs.mean(0), Fs.std(0).mean(-1)
    E_true = np.array([r["E"]/r["n_atoms"] for r in te])
    F_true = np.concatenate([r["F"] for r in te])

    rep_E = calibration_report(E_sd*EV_KCALMOL, (E_mu-E_true)*EV_KCALMOL)
    rep_F = calibration_report(F_sd*EV_KCALMOL,
                               (F_mu-F_true).mean(-1)*EV_KCALMOL)
    print("\n"+"="*74)
    print(f"  ENSEMBLE UQ — M={args.n_models}  bootstrap={args.bootstrap}")
    print("="*74)
    print(f"  ENERGY  RMSE {rep_E['rmse']:.4f} kcal/mol/atom   "
          f"mean σ {rep_E['mean_sigma']:.4f}")
    print(f"    coverage 1σ {rep_E['coverage_1sigma']:.3f} (nominal 0.683) | "
          f"2σ {rep_E['coverage_2sigma']:.3f} (0.954)")
    print(f"    ECE {rep_E['ECE']:.4f} | Spearman(σ,|err|) "
          f"{rep_E['spearman_sigma_vs_abserr']:.3f} → {rep_E['verdict']}")
    print(f"  FORCE   RMSE {rep_F['rmse']:.4f} kcal/mol/Å   "
          f"mean σ {rep_F['mean_sigma']:.4f}")
    print(f"    ECE {rep_F['ECE']:.4f} | Spearman "
          f"{rep_F['spearman_sigma_vs_abserr']:.3f} → {rep_F['verdict']}")
    print("\n  HONEST: ensemble spread captures EPISTEMIC uncertainty only.")
    print("  It does NOT capture DFT label error (aleatoric). A calibrated")
    print("  ensemble still cannot tell you the functional is wrong.")
    print("="*74)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"energy": rep_E, "force": rep_F, "args": vars(args)},
              open(args.out,"w"), indent=2)
    for i,m in enumerate(models):
        torch.save(m.state_dict(), str(Path(args.out).with_suffix(f".m{i}.pt")))


# =============================================================================
# 3. CONSERVATIVE-FORCE VALIDATION  (curl / net force / torque)
# =============================================================================

class NonConservativeHead(nn.Module):
    """Deliberately non-conservative baseline: predicts F directly."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.fhead = nn.Sequential(nn.Linear(base.head[0].in_features, 64),
                                   nn.SiLU(), nn.Linear(64, 3))
    def forward(self, z, R, batch, ng):
        n = z.shape[0]
        ii, jj, dist, _ = _edges(R, batch, 5.0)
        # recompute features (mirror base trunk)
        from force_datasets import RBF, cut
        rbf = self.base.rbf(dist); c = cut(dist, 5.0)
        u = (R[jj]-R[ii]) / dist.unsqueeze(-1).clamp(min=1e-8)
        s = self.base.emb(z)
        v = torch.zeros(n,3,s.shape[-1], device=R.device, dtype=s.dtype)
        for m,up in zip(self.base.msg, self.base.upd):
            s,v = m(s,v,ii,jj,rbf,c,u,n); s,v = up(s,v)
        E = self.base.head(s).squeeze(-1)
        F = self.fhead(s)          # NOT a gradient of anything
        return E.sum(), F


def _edges(R, batch, rc):
    same = batch.unsqueeze(0)==batch.unsqueeze(1)
    d = torch.cdist(R,R)
    m = same & (d<rc) & (d>1e-8)
    ii,jj = m.nonzero(as_tuple=True)
    diff = R[jj]-R[ii]
    return ii,jj,torch.sqrt((diff*diff).sum(-1)+1e-12),diff


def curl_check(F: Tensor, R: Tensor) -> Dict[str, float]:
    """
    J_ij = ∂F_i/∂x_j  (3N×3N).  Conservative ⟺ J = Jᵀ.
    curl measure = ||J - Jᵀ||_F / (||J||_F + eps)   ∈ [0, √2]
    """
    Rf = R.reshape(-1)
    Ff = F.reshape(-1)
    n = Rf.numel()
    J = torch.stack([torch.autograd.grad(Ff[i], R, retain_graph=True,
                                         allow_unused=True)[0].reshape(-1)
                     for i in range(n)])
    A = 0.5*(J - J.T)
    rel = float(A.norm()/(J.norm()+1e-12))
    return {"curl_relative": rel,
            "curl_abs": float(A.norm()),
            "jacobian_norm": float(J.norm()),
            "max_asymmetry": float(A.abs().max())}


def fd_curl_check(model, z, R, batch, ng, h=1e-3, n_pairs=24) -> Dict:
    """Independent finite-difference check (does not trust autograd)."""
    Rf = R.reshape(-1).clone()
    n = Rf.numel()
    rng = np.random.default_rng(0)
    pairs = rng.choice(n, size=(min(n_pairs, n*(n-1)//2), 2), replace=False)
    vals = []
    for i, j in pairs:
        if i == j: continue
        def Fi_at(k, sign):
            Rp = Rf.clone(); Rp[k] += sign*h
            Rp = Rp.reshape(R.shape).detach().requires_grad_(True)
            E = model(z, Rp, batch, ng)
            F = -torch.autograd.grad(E.sum(), Rp)[0].reshape(-1)
            return float(F[i])
        dFij = (Fi_at(j,+1)-Fi_at(j,-1))/(2*h)
        dFji = (Fi_at(i,+1)-Fi_at(i,-1))/(2*h)
        vals.append(abs(dFij-dFji))
    return {"fd_max_asymmetry": float(max(vals)) if vals else 0.0,
            "fd_mean_asymmetry": float(np.mean(vals)) if vals else 0.0,
            "n_pairs": len(vals), "h": h}


def validate_conservative(args):
    if not HAS_BASE: log.error("force_datasets.py required"); return
    dev = torch.device("cpu")   # small system, exact checks
    torch.manual_seed(0)
    d = load_rmd17(Path(args.data), args.molecule, limit=8)
    rec = to_records(d)[0]
    b = collate([rec], dev)
    print("\n"+"="*74)
    print("  CONSERVATIVE-FORCE VALIDATION")
    print("="*74)
    print(f"  system: {args.molecule}, {rec['n_atoms']} atoms "
          f"({3*rec['n_atoms']} DOF)")

    # --- A: autograd forces (conservative BY CONSTRUCTION) ---
    net = PaiNNF(64, 32, 3, 5.0).to(dev).eval()
    R = b["R"].detach().requires_grad_(True)
    E = net(b["z"], R, b["batch"], b["ng"])
    Fa = -torch.autograd.grad(E.sum(), R, create_graph=True)[0]
    ca = curl_check(Fa, R)
    fd = fd_curl_check(net, b["z"], R, b["batch"], b["ng"])
    print(f"\n  [A] F = -autograd(E, R)")
    print(f"      curl_relative   = {ca['curl_relative']:.3e}   (expect ~1e-7)")
    print(f"      max asymmetry   = {ca['max_asymmetry']:.3e}")
    print(f"      FD check        = {fd['fd_max_asymmetry']:.3e} "
          f"({fd['n_pairs']} pairs, h={fd['h']})")
    print(f"      → conservative BY CONSTRUCTION. This is a UNIT TEST.")

    # --- B: direct force head (generally NOT conservative) ---
    nc = NonConservativeHead(net).to(dev).eval()
    R2 = b["R"].detach().requires_grad_(True)
    E2, Fd = nc(b["z"], R2, b["batch"], b["ng"])
    cb = curl_check(Fd, R2)
    print(f"\n  [B] direct force head (no energy gradient)")
    print(f"      curl_relative   = {cb['curl_relative']:.3e}   (expect O(1))")
    print(f"      max asymmetry   = {cb['max_asymmetry']:.3e}")
    print(f"      → NOT conservative. Energy will drift in MD.")

    # --- C: net force & torque (translational / rotational invariance) ---
    coords = R.detach().reshape(-1,3)
    netF = Fa.detach().reshape(-1,3).sum(0)
    torque = torch.cross(coords, Fa.detach().reshape(-1,3), dim=-1).sum(0)
    print(f"\n  [C] invariances (autograd model, distance-only features)")
    print(f"      |Σ F|   = {netF.norm():.3e}   (translation → should be ~0)")
    print(f"      |Σ r×F| = {torque.norm():.3e}   (rotation → should be ~0)")

    # --- D: energy drift along a short trajectory ---
    print(f"\n  [D] energy drift, velocity-Verlet, 200 steps, dt=0.5 fs")
    for tag, get_F in (("autograd", "A"), ("direct-head", "B")):
        Rr = b["R"].detach().clone().reshape(-1,3)
        v = torch.randn_like(Rr)*0.01
        mass = torch.tensor([12.0,1.0,14.0,16.0,19.0,32.0,35.5])[b["z"]].reshape(-1,1)
        def force(Rx):
            Rx = Rx.reshape(1,-1,3).detach().requires_grad_(True)
            if get_F=="A":
                ee = net(b["z"], Rx, b["batch"], 1)
                return -torch.autograd.grad(ee.sum(), Rx)[0].reshape(-1,3), float(ee)
            ee, ff = nc(b["z"], Rx, b["batch"], 1)
            return ff.reshape(-1,3), float(ee)
        f0, E0 = force(Rr); Etot0 = E0 + 0.5*float((mass*v*v).sum())
        for _ in range(200):
            v = v + 0.5*0.5*f0/mass
            Rr = Rr + 0.5*v
            f1, E1 = force(Rr)
            v = v + 0.5*0.5*f1/mass
            f0 = f1
        Etot1 = E1 + 0.5*float((mass*v*v).sum())
        print(f"      {tag:12s} ΔE_total = {Etot1-Etot0:+.6f} eV")
    print("\n  HONEST: [A] verifies implementation, not physics.")
    print("  [B] shows why a separate force head breaks energy conservation.")
    print("  Only [A]-type models are safe for microcanonical MD.")
    print("="*74)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"autograd": ca, "fd": fd, "direct_head": cb,
               "net_force": float(netF.norm()),
               "torque": float(torque.norm())}, open(args.out,"w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["delta","ensemble","curl"])
    p.add_argument("--ccx", default="data/ani1ccx/ani1ccx-release.h5")
    p.add_argument("--data", default="data/rmd17")
    p.add_argument("--dataset", choices=["rmd17","ani1x"], default="rmd17")
    p.add_argument("--molecule", default="benzene")
    p.add_argument("--out", default="ckpt/adv.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    p.add_argument("--n-models", dest="n_models", type=int, default=5)
    p.add_argument("--n-samples", dest="n_samples", type=int, default=3000)
    p.add_argument("--n-train", dest="n_train", type=int, default=950)
    p.add_argument("--max-systems", dest="max_systems", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--force-weight", dest="force_weight", type=float, default=100.)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--delta-forces", dest="delta_forces", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    {"delta": train_delta, "ensemble": train_ensemble,
     "curl": validate_conservative}[a.cmd](a)


if __name__ == "__main__":
    main()
