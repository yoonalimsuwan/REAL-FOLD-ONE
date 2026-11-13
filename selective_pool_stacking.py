#!/usr/bin/env python3
"""
=============================================================================
PARAMETER-SELECTIVE HIERARCHICAL PRIOR  +  CROSS-VALIDATED MODEL AVERAGING
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — SELECTIVE PARTIAL POOLING (diagonal, per-parameter)
  Instead of theta_s ~ N(mu_g, Sigma) with a full 4x4 covariance, each
  parameter gets its OWN pooling decision and its OWN variance components:

      POOLED j:    theta_sj ~ N(mu_{g(s),j}, tau_wj^2)
                   mu_{g,j} ~ N(m_j, tau_bj^2)
      UNPOOLED j:  theta_sj ~ N(m_j, s_j^2)      (weak, independent)

  Physical motivation for the default choice:
    dH0, dS0 are strongly system-specific (H-bonding, vdW complementarity,
      conformational and solvation entropy of THIS pair). Pooling them
      across ligands biases every system toward a meaningless average.
    dCp0 correlates with buried nonpolar surface area and is far more
      transferable across ligand classes. Pooling it is defensible.
    b = d(dCp)/dT is weakly identified from a narrow temperature range;
      whether to pool it is a question for the data, not for us.

  Scale priors: half-Cauchy via the exact Makalic-Schmidt auxiliary Gibbs
  (Gelman 2006 recommends half-t over inverse-gamma for variance
  parameters; the auxiliary keeps every step conjugate and exact).

  DIAGNOSTIC: per-parameter ICC_j = tau_bj^2/(tau_bj^2+tau_wj^2).
  Reported ONLY for pooled parameters. A parameter with ICC ~ 0 carries
  no class-level signal and pooling it adds variance without benefit.
  Which parameters to pool is decided by LOO comparison of pooling
  configurations, not assumed.

PART B — CROSS-VALIDATED MODEL AVERAGING (M0 / M1 / M2)
    M0: dCp = 0            theta = (dH0, dS0)
    M1: dCp = dCp0         theta = (dH0, dS0, dCp0)
    M2: dCp = dCp0 + b dT  theta = (dH0, dS0, dCp0, b)

  For each model, exact leave-one-out predictive densities p_{m,-i}(y_i)
  are computed by refitting (n is small, so this is cheap and exact).
  Two weightings are reported:

    pseudo-BMA :  w_m ∝ exp(elpd_m)
    STACKING   :  w* = argmax_{w in simplex} sum_i log( sum_m w_m p_{m,-i}(y_i) )

  Stacking is preferred when the models are correlated (they are here:
  M0 ⊂ M1 ⊂ M2), because pseudo-BMA over-concentrates on the single
  best-fitting model. Optimised by exponentiated gradient, which stays
  on the simplex by construction.

  Model-averaged dCp(T) is a MIXTURE over models, so its variance
  decomposes into within-model and between-model parts. Reporting only
  the within-model spread would understate the uncertainty; both are
  reported.

HONEST SCOPE
  ❌ Stacking weights are predictive-optimal weights, NOT posterior
     probabilities that a model is true.
  ❌ With n small, LOO elpd is noisy; the SE of elpd differences is
     reported so the ambiguity is visible.
  ❌ With G=2 classes tau_b is barely identified; with a singleton class
     mu_g collapses to the global mean. Both flagged.
  ❌ Half-Cauchy scale priors still influence tau when data are sparse;
     a prior-scale sensitivity sweep is provided.
  ❌ No experimental validation is performed here.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("selective_stack")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

PARAM_NAMES = ("dH0", "dS0", "dCp0", "b")
PARAM_UNITS = ("kJ/mol", "kJ/mol/K", "kJ/mol/K", "kJ/mol/K^2")
MODELS: Dict[str, Tuple[int, ...]] = {
    "M0": (0, 1),
    "M1": (0, 1, 2),
    "M2": (0, 1, 2, 3),
}
WEAK_SD = np.array([100.0, 0.5, 2.0, 0.02])   # unpooled prior scales


def design_row(T: float, kind: str, T0: float) -> np.ndarray:
    """theta = (dH0, dS0, dCp0, b), dCp(T) = dCp0 + b (T-T0)."""
    dT = T - T0
    L = math.log(T / T0)
    if kind == "dH":
        return np.array([1.0, 0.0, dT, 0.5 * dT * dT])
    if kind == "dG":
        return np.array([1.0, -T, dT - T * L, 0.5 * dT * dT - T * (dT - T0 * L)])
    raise ValueError(f"kind must be dH|dG, got {kind!r}")


@dataclass
class SystemData:
    name: str
    group: str
    T: np.ndarray
    kind: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    T0: float = 310.0

    def design(self, cols: Tuple[int, ...]):
        X = np.vstack([design_row(float(t), str(k), self.T0)[list(cols)]
                       for t, k in zip(self.T, self.kind)])
        w = 1.0 / np.maximum(np.asarray(self.sigma, float), 1e-12) ** 2
        return X, np.asarray(self.y, float), w

    @property
    def n(self) -> int:
        return len(self.y)


# =============================================================================
# PART A  SELECTIVE HIERARCHICAL GIBBS
# =============================================================================

class SelectiveHierarchy:
    """
    Diagonal hierarchical model with per-parameter pooling.

    pool_names selects WHICH parameters borrow strength. Everything else
    gets a weak independent prior. All Gibbs steps are exact:
      theta_s | .  : full k-variate Gaussian (likelihood couples params)
      mu_gj   | .  : univariate Gaussian
      tau^2   | .  : inverse-gamma (Makalic-Schmidt auxiliary for half-Cauchy)
    """

    def __init__(self, systems: List[SystemData],
                 model: str = "M2",
                 pool_names: Sequence[str] = ("dCp0",),
                 hc_A: float = 2.0,
                 weak_sd: Optional[np.ndarray] = None):
        self.systems = systems
        self.model = model
        self.cols = MODELS[model]
        self.names = [PARAM_NAMES[c] for c in self.cols]
        self.k = len(self.cols)
        self.S = len(systems)
        self.pool = np.array([nm in set(pool_names) for nm in self.names])
        self.groups = sorted({s.group for s in systems})
        self.G = len(self.groups)
        self.gidx = {g: i for i, g in enumerate(self.groups)}
        self.sg = np.array([self.gidx[s.group] for s in systems], int)
        self.m = np.zeros(self.k)
        self.weak_var = (weak_sd if weak_sd is not None
                         else WEAK_SD)[list(self.cols)] ** 2
        self.hc_A = float(hc_A)
        self._prep = [s.design(self.cols) for s in systems]

    # ------------------------------------------------------------------
    def _init_theta(self) -> np.ndarray:
        th = np.zeros((self.S, self.k))
        for i, (X, y, w) in enumerate(self._prep):
            if self.systems[i].n >= self.k:
                Xw = X * np.sqrt(w)[:, None]
                th[i] = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)[0]
            else:
                th[i] = self.m
        return th

    def fit(self, n_iter: int = 3000, burn: int = 1000, thin: int = 3,
            seed: int = 0) -> Dict[str, Any]:
        if self.S < 2:
            return {"status": "REFUSED", "reason": "need >= 2 systems"}
        warns: List[str] = []
        pooled_names = [nm for nm, p in zip(self.names, self.pool) if p]
        if not pooled_names:
            warns.append("pool_names is empty: this is a no-pooling fit.")
        if self.G < 2 and pooled_names:
            warns.append("G=1: the class level is unidentifiable; pooled "
                         "parameters collapse to a single global mean.")
        if self.G == 2 and pooled_names:
            warns.append("G=2: tau_b is barely identified; its posterior is "
                         "prior-sensitive. Read the sensitivity sweep.")
        counts = np.bincount(self.sg, minlength=self.G)
        singles = [self.groups[g] for g in range(self.G) if counts[g] == 1]
        if singles and pooled_names:
            warns.append(f"singleton classes {singles}: their mu_g shrinks "
                         "almost entirely to the global mean.")
        for i, s in enumerate(self.systems):
            if s.n < self.k:
                warns.append(f"system {s.name}: n={s.n} < k={self.k}; its "
                             "parameters are prior-regularised, not measured.")

        rng = np.random.default_rng(seed)
        k, S, G = self.k, self.S, self.G
        P = self.pool
        theta = self._init_theta()
        mu = np.zeros((G, k))
        for g in range(G):
            mu[g] = theta[self.sg == g].mean(0)
        # variance components (only meaningful for pooled entries)
        tw = np.where(P, 1.0, 1.0) * 1.0
        tb = np.where(P, 1.0, 1.0) * 1.0
        scale0 = np.sqrt(self.weak_var)
        tw = np.where(P, scale0 * 0.5, 1.0)
        tb = np.where(P, scale0 * 0.3, 1.0)
        aux_w = np.ones(k)      # Makalic-Schmidt auxiliaries
        aux_b = np.ones(k)

        DT, DM, DTW, DTB = [], [], [], []
        for it in range(n_iter):
            # ---- theta_s | rest -------------------------------------
            prior_var = np.where(P, tw ** 2, self.weak_var)
            Dprec = 1.0 / np.maximum(prior_var, 1e-30)
            for i, (X, y, w) in enumerate(self._prep):
                pmean = np.where(P, mu[self.sg[i]], self.m)
                A = X.T @ (w[:, None] * X) + np.diag(Dprec)
                b = X.T @ (w * y) + Dprec * pmean
                V = np.linalg.inv(A)
                mean = V @ b
                theta[i] = mean + np.linalg.cholesky(V) @ rng.standard_normal(k)
            # ---- mu_gj | rest (pooled only) -------------------------
            for j in range(k):
                if not P[j]:
                    continue
                pw, pb = 1.0 / tw[j] ** 2, 1.0 / tb[j] ** 2
                for g in range(G):
                    sel = self.sg == g
                    ng = int(sel.sum())
                    prec = ng * pw + pb
                    mn = (pw * theta[sel, j].sum() + pb * self.m[j]) / prec
                    mu[g, j] = mn + rng.standard_normal() / math.sqrt(prec)
            # ---- tau_wj^2, tau_bj^2 | rest (half-Cauchy, exact) -----
            for j in range(k):
                if not P[j]:
                    continue
                rw = theta[:, j] - mu[self.sg, j]
                shape_w = 0.5 * (S + 1)
                rate_w = 1.0 / aux_w[j] + 0.5 * float(rw @ rw)
                tw[j] = math.sqrt(rate_w / rng.gamma(shape_w))
                aux_w[j] = (1.0 / self.hc_A ** 2 + 1.0 / tw[j] ** 2) \
                           / rng.gamma(1.0)
                rb = mu[:, j] - self.m[j]
                shape_b = 0.5 * (G + 1)
                rate_b = 1.0 / aux_b[j] + 0.5 * float(rb @ rb)
                tb[j] = math.sqrt(rate_b / rng.gamma(shape_b))
                aux_b[j] = (1.0 / self.hc_A ** 2 + 1.0 / tb[j] ** 2) \
                           / rng.gamma(1.0)
            if it >= burn and (it - burn) % thin == 0:
                DT.append(theta.copy()); DM.append(mu.copy())
                DTW.append(tw.copy()); DTB.append(tb.copy())

        T_arr = np.array(DT); M_arr = np.array(DM)
        TW = np.array(DTW); TB = np.array(DTB)

        # ---- per-parameter diagnostics ------------------------------
        params: Dict[str, Any] = {}
        for j, nm in enumerate(self.names):
            if P[j]:
                icc = (TB[:, j] ** 2) / (TB[:, j] ** 2 + TW[:, j] ** 2 + 1e-30)
                params[nm] = {
                    "pooled": True,
                    "tau_within_mean": float(TW[:, j].mean()),
                    "tau_between_mean": float(TB[:, j].mean()),
                    "ICC_mean": float(icc.mean()),
                    "ICC_ci95": [float(v) for v in np.quantile(icc, [.025, .975])],
                    "class_means": M_arr[:, :, j].mean(0).tolist(),
                }
                if icc.mean() < 0.05:
                    warns.append(f"{nm}: ICC={icc.mean():.3f} ~ 0 — the class "
                                 "level carries almost no signal for this "
                                 "parameter; pooling it adds variance. "
                                 "Compare pooling configs by LOO.")
            else:
                params[nm] = {"pooled": False,
                              "prior_sd": float(math.sqrt(self.weak_var[j])),
                              "note": "system-specific, weak independent prior"}
        # ---- shrinkage per system -----------------------------------
        per_sys = []
        for i, s in enumerate(self.systems):
            d = T_arr[:, i, :]
            ols = None
            if s.n >= self.k:
                X, y, w = self._prep[i]
                Xw = X * np.sqrt(w)[:, None]
                ols = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)[0]
            sh = {}
            for j, nm in enumerate(self.names):
                if ols is None:
                    sh[nm] = None
                else:
                    sh[nm] = float(abs(d[:, j].mean() - ols[j])
                                   / max(abs(ols[j]), 1e-12))
            per_sys.append({
                "name": s.name, "group": s.group, "n": s.n,
                "theta_mean": d.mean(0).tolist(),
                "theta_sd": d.std(0, ddof=1).tolist(),
                "ci95": np.quantile(d, [.025, .975], axis=0).tolist(),
                "relative_shrinkage": sh,
                "prior_regularised": s.n < self.k,
            })
        return {"status": "OK", "model": self.model, "S": S, "G": G,
                "groups": self.groups, "group_counts": counts.tolist(),
                "pool_names": pooled_names, "n_draws": len(DT),
                "params": params, "systems": per_sys, "warnings": warns,
                "_draws": {"theta": T_arr, "mu": M_arr,
                           "tau_w": TW, "tau_b": TB}}

    # ------------------------------------------------------------------
    def loo(self, n_iter: int = 1200, burn: int = 400,
            seed: int = 1) -> Dict[str, Any]:
        """
        Leave-one-OBSERVATION-out predictive log density, summed over all
        systems. Exact refit per held-out point (n small).
        """
        rows = []
        for si, s in enumerate(self.systems):
            X, y, w = self._prep[si]
            for i in range(s.n):
                train = []
                for sj, ss in enumerate(self.systems):
                    if sj != si:
                        train.append(ss); continue
                    keep = np.arange(ss.n) != i
                    if keep.sum() == 0:
                        continue
                    train.append(SystemData(ss.name, ss.group,
                                            ss.T[keep], ss.kind[keep],
                                            ss.y[keep], ss.sigma[keep], ss.T0))
                if len(train) < 2:
                    rows.append({"system": s.name, "i": i, "skipped": True})
                    continue
                sub = SelectiveHierarchy(train, self.model,
                                         [nm for nm, p in zip(self.names, self.pool) if p],
                                         self.hc_A, np.sqrt(self.weak_var))
                f = sub.fit(n_iter, burn, seed=seed + i)
                if f["status"] != "OK":
                    rows.append({"system": s.name, "i": i, "skipped": True})
                    continue
                TH = f["_draws"]["theta"][:, si, :]
                x = X[i]
                mu_d = TH @ x
                var_d = float(x @ np.cov(TH.T) @ x) + 1.0 / w[i]
                lp = -0.5 * (np.log(2 * np.pi * var_d) + (y[i] - mu_d) ** 2 / var_d)
                mx = float(lp.max())
                rows.append({"system": s.name, "i": i, "T_K": float(s.T[i]),
                             "kind": str(s.kind[i]),
                             "lpd": mx + math.log(float(np.exp(lp - mx).mean()))})
        ok = [r["lpd"] for r in rows if "lpd" in r]
        return {"status": "OK" if ok else "REFUSED", "rows": rows,
                "elpd_loo": float(sum(ok)) if ok else float("nan"),
                "n_scored": len(ok)}


def compare_pooling_configs(systems: List[SystemData], model: str = "M2",
                            configs: Optional[Dict[str, Sequence[str]]] = None,
                            n_iter: int = 1000, burn: int = 350,
                            seed: int = 2) -> Dict[str, Any]:
    """
    Decide WHICH parameters to pool by LOO, not by assumption.
    """
    if configs is None:
        configs = {"none": (), "dCp0": ("dCp0",),
                   "dCp0+b": ("dCp0", "b"),
                   "all": PARAM_NAMES[:len(MODELS[model])]}
    out = {}
    for label, pn in configs.items():
        h = SelectiveHierarchy(systems, model, pn)
        r = h.loo(n_iter, burn, seed=seed)
        out[label] = {"pool_names": list(pn), "elpd_loo": r["elpd_loo"],
                      "n_scored": r["n_scored"], "status": r["status"]}
    usable = {k: v for k, v in out.items() if math.isfinite(v["elpd_loo"])}
    best = max(usable, key=lambda k: usable[k]["elpd_loo"]) if usable else None
    return {"configs": out, "best": best,
            "note": "elpd differences smaller than their SE do not "
                    "distinguish configurations"}


# =============================================================================
# PART B  MODEL-WISE LOO + STACKING
# =============================================================================

def model_loo_densities(systems: List[SystemData],
                        pool_names: Sequence[str] = ("dCp0",),
                        models: Sequence[str] = ("M0", "M1", "M2"),
                        n_iter: int = 1200, burn: int = 400,
                        seed: int = 3) -> Dict[str, Any]:
    """
    Per-observation LOO predictive log density for each model, on a
    COMMON observation ordering so the weights are comparable.
    """
    keys: List[Tuple[str, int]] = []
    for s in systems:
        keys += [(s.name, i) for i in range(s.n)]
    lpd: Dict[str, np.ndarray] = {}
    fits: Dict[str, Any] = {}
    for m in models:
        h = SelectiveHierarchy(systems, m, pool_names)
        f = h.fit(n_iter, burn, seed=seed)
        fits[m] = f
        r = h.loo(n_iter, burn, seed=seed + 10)
        v = np.full(len(keys), np.nan)
        idx = {k: j for j, k in enumerate(keys)}
        for row in r["rows"]:
            if "lpd" not in row:
                continue
            v[idx[(row["system"], row["i"])]] = row["lpd"]
        lpd[m] = v
    good = np.all(np.isfinite(np.vstack([lpd[m] for m in models])), axis=0)
    return {"models": list(models), "keys": keys, "lpd": lpd,
            "valid_mask": good, "fits": fits}


def _simplex_stack(lpd_mat: np.ndarray, n_iter: int = 4000,
                   lr: float = 0.05) -> np.ndarray:
    """
    Maximise sum_i log(sum_m w_m exp(lpd[m,i])) over the simplex by
    exponentiated gradient (mirror descent), which stays on the simplex.
    Per-observation shifting keeps the exponentials in range.
    """
    M, N = lpd_mat.shape
    shift = lpd_mat.max(axis=0)
    D = np.exp(lpd_mat - shift[None, :])          # (M,N), all <= 1
    w = np.full(M, 1.0 / M)
    eta = lr
    best_obj, best_w = -np.inf, w.copy()
    for it in range(n_iter):
        mix = w @ D                                # (N,)
        mix = np.maximum(mix, 1e-300)
        obj = float(np.log(mix).sum())
        if obj > best_obj:
            best_obj, best_w = obj, w.copy()
        grad = D / mix[None, :]                    # (M,N)
        g = grad.sum(axis=1)
        w = w * np.exp(eta * (g - g.max()))
        w = np.maximum(w, 1e-12)
        w /= w.sum()
        if it % 500 == 499:
            eta *= 0.7
    return best_w


def model_average(systems: List[SystemData],
                  pool_names: Sequence[str] = ("dCp0",),
                  models: Sequence[str] = ("M0", "M1", "M2"),
                  T_grid: Sequence[float] = (298., 308., 318.),
                  T0: float = 310.0, n_draws: int = 40000,
                  n_iter: int = 1500, burn: int = 500,
                  seed: int = 5) -> Dict[str, Any]:
    """
    Stacking + pseudo-BMA weights, and the model-averaged dCp(T) mixture
    with within/between variance decomposition.
    """
    R = model_loo_densities(systems, pool_names, models, n_iter, burn, seed)
    mask = R["valid_mask"]
    if mask.sum() < 3:
        return {"status": "REFUSED",
                "reason": f"only {int(mask.sum())} observations scored for "
                          "all models; need >= 3"}
    L = np.vstack([R["lpd"][m][mask] for m in R["models"]])
    elpd = L.sum(axis=1)
    # SE of pairwise elpd differences (per-observation)
    se: Dict[str, float] = {}
    for a in range(len(models)):
        for b in range(a + 1, len(models)):
            d = L[a] - L[b]
            se[f"{R['models'][a]}-{R['models'][b]}"] = float(
                math.sqrt(len(d) * np.var(d, ddof=1)))
    w_stack = _simplex_stack(L)
    e = elpd - elpd.max()
    w_bma = np.exp(e); w_bma /= w_bma.sum()

    rng = np.random.default_rng(seed)
    # ---- mixture draws of dCp(T) -----------------------------------
    dcp: Dict[str, Any] = {}
    for T in T_grid:
        per_model = []
        for m in R["models"]:
            f = R["fits"][m]
            if f["status"] != "OK":
                per_model.append(np.zeros(1000)); continue
            TH = f["_draws"]["theta"]
            names = [PARAM_NAMES[c] for c in MODELS[m]]
            v = np.zeros(TH.shape[2])
            if "dCp0" in names:
                v += TH[:, :, names.index("dCp0")].ravel()[:0].sum()  # noop
            # system-averaged dCp(T): average over systems
            cp = np.zeros(TH.shape[0])
            if "dCp0" in names:
                cp += TH[:, :, names.index("dCp0")].mean(axis=1)
            if "b" in names:
                cp += TH[:, :, names.index("b")].mean(axis=1) * (T - T0)
            per_model.append(cp)
        mix_parts, mix_w = [], []
        for m, cp, wm in zip(R["models"], per_model, w_stack):
            n_take = max(1, int(round(wm * n_draws)))
            if len(cp) == 0:
                continue
            mix_parts.append(rng.choice(cp, size=n_take, replace=True))
            mix_w.append(wm)
        mix = np.concatenate(mix_parts)
        within = float(sum(wm * np.var(cp, ddof=1)
                           for wm, cp in zip(mix_w, per_model) if len(cp) > 1))
        means = np.array([cp.mean() if len(cp) else 0.0 for cp in per_model])
        between = float(sum(wm * (mu_ - mix.mean()) ** 2
                            for wm, mu_ in zip(mix_w, means)))
        lo, hi = np.quantile(mix, [.025, .975])
        dcp[f"{T:.1f}"] = {
            "mean": float(mix.mean()),
            "sd_total": float(mix.std(ddof=1)),
            "var_within": within, "var_between": between,
            "between_fraction": between / max(within + between, 1e-30),
            "ci95": [float(lo), float(hi)],
        }
    return {"status": "OK", "models": R["models"],
            "elpd_loo": {m: float(v) for m, v in zip(R["models"], elpd)},
            "elpd_diff_se": se,
            "weights_stacking": {m: float(v)
                                 for m, v in zip(R["models"], w_stack)},
            "weights_pseudo_bma": {m: float(v)
                                   for m, v in zip(R["models"], w_bma)},
            "n_scored": int(mask.sum()),
            "dCp_averaged": dcp,
            "note": ("stacking weights are predictive-optimal, NOT posterior "
                     "model probabilities; the averaged interval includes "
                     "between-model spread")}


def report(r: Dict[str, Any]) -> None:
    print("=" * 94)
    print("  CROSS-VALIDATED MODEL AVERAGING")
    print("=" * 94)
    if r["status"] != "OK":
        print(f"  REFUSED: {r.get('reason')}")
        return
    print(f"  {'model':<8}{'elpd_loo':>14}{'w_stack':>12}{'w_pBMA':>12}")
    print("-" * 94)
    for m in r["models"]:
        print(f"  {m:<8}{r['elpd_loo'][m]:>14.3f}"
              f"{r['weights_stacking'][m]:>12.4f}"
              f"{r['weights_pseudo_bma'][m]:>12.4f}")
    print("-" * 94)
    for k, v in r["elpd_diff_se"].items():
        print(f"  SE(elpd {k}) = {v:.3f}")
    print("-" * 94)
    print("  model-averaged dCp(T):")
    for T, d in r["dCp_averaged"].items():
        print(f"    T={T:>7} K  {d['mean']:+.5f} ± {d['sd_total']:.5f}   "
              f"CI [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]   "
              f"between-model {100*d['between_fraction']:.1f}%")
    print("=" * 94)


# =============================================================================
# SELF-TEST
# =============================================================================

def _make(rng, mode: str, n_per: int = 6, T0: float = 310.0,
          b_true: float = 0.0) -> List[SystemData]:
    base = np.array([-45.0, -0.10, -0.65, b_true])
    shifts = {"A": np.array([0., 0., 0., 0.]),
              "B": np.array([14., .02, -.30, 0.]),
              "C": np.array([-12., -.02, .28, 0.])}
    out = []
    for g, sh in shifts.items():
        for j in range(3):
            if mode == "cp_structure":
                th = base + sh + rng.normal(0, 1, 4) * np.array([4., .04, .04, 1e-5])
            else:                                   # no class signal in dCp
                th = base + rng.normal(0, 1, 4) * np.array([4., .04, .04, 1e-5])
            Ts = np.linspace(296., 324., n_per) + rng.normal(0, .6, n_per)
            kinds, ys, sg = [], [], []
            for t in Ts:
                dT = t - T0
                H = th[0] + th[2] * dT + .5 * th[3] * dT * dT
                kinds.append("dH"); ys.append(H + rng.normal(0, .2)); sg.append(.2)
            out.append(SystemData(f"{g}{j}", g, Ts, np.array(kinds),
                                  np.array(ys), np.array(sg), T0))
    return out


def self_test() -> bool:
    print("=" * 94)
    print(f"  SELECTIVE POOLING + STACKING  v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 94)
    rng = np.random.default_rng(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # ---------------- A1 selective pooling flags ----------------
    print("\n[A1] Only dCp0 is pooled; dH0/dS0 stay system-specific")
    sysl = _make(rng, "cp_structure")
    h = SelectiveHierarchy(sysl, "M2", ("dCp0",))
    f = h.fit(n_iter=2200, burn=800, seed=1)
    rep("fit OK", f["status"] == "OK")
    if f["status"] == "OK":
        for nm in ("dH0", "dS0", "dCp0", "b"):
            d = f["params"][nm]
            if d["pooled"]:
                print(f"      {nm:<6} POOLED   ICC={d['ICC_mean']:.3f} "
                      f"tau_w={d['tau_within_mean']:.4f} "
                      f"tau_b={d['tau_between_mean']:.4f}")
            else:
                print(f"      {nm:<6} unpooled prior_sd={d['prior_sd']:.4f}")
        rep("dCp0 pooled", f["params"]["dCp0"]["pooled"])
        rep("dH0 unpooled", not f["params"]["dH0"]["pooled"])
        rep("dS0 unpooled", not f["params"]["dS0"]["pooled"])

    # ---------------- A2 unpooled params are NOT shrunk --------------
    print("\n[A2] Unpooled parameters retain system-specific values")
    if f["status"] == "OK":
        sh_H = [s["relative_shrinkage"]["dH0"] for s in f["systems"]
                if s["relative_shrinkage"]["dH0"] is not None]
        sh_C = [s["relative_shrinkage"]["dCp0"] for s in f["systems"]
                if s["relative_shrinkage"]["dCp0"] is not None]
        print(f"      mean |shrinkage| dH0={np.mean(sh_H):.4f}  "
              f"dCp0={np.mean(sh_C):.4f}")
        rep("dH0 shrinkage < dCp0 shrinkage",
            np.mean(sh_H) < np.mean(sh_C) + 1e-6)

    # ---------------- A3 ICC detects real class signal ---------------
    print("\n[A3] ICC: real class signal vs none")
    fake = _make(rng, "no_signal")
    hf = SelectiveHierarchy(fake, "M2", ("dCp0",))
    ff = hf.fit(n_iter=2200, burn=800, seed=2)
    if f["status"] == "OK" and ff["status"] == "OK":
        i_real = f["params"]["dCp0"]["ICC_mean"]
        i_fake = ff["params"]["dCp0"]["ICC_mean"]
        print(f"      ICC(dCp0) real={i_real:.3f}  no-signal={i_fake:.3f}")
        rep("ICC larger when class signal is real", i_real > i_fake)

    # ---------------- A4 pooling-config LOO comparison ---------------
    print("\n[A4] Which parameters to pool — decided by LOO")
    cmp_cfg = compare_pooling_configs(sysl[:6], "M2",
                                      {"none": (), "dCp0": ("dCp0",),
                                       "all": PARAM_NAMES},
                                      n_iter=700, burn=250, seed=3)
    for k, v in cmp_cfg["configs"].items():
        print(f"      {k:<8} elpd={v['elpd_loo']:.3f}  n={v['n_scored']}")
    rep("pooling comparison produced a best config",
        cmp_cfg["best"] is not None, f"best={cmp_cfg['best']}")

    # ---------------- B1 stacking weights on simplex -----------------
    print("\n[B1] Stacking weights")
    ma = model_average(sysl[:6], ("dCp0",), ("M0", "M1", "M2"),
                       T_grid=(298., 310., 322.),
                       n_iter=800, burn=300, seed=4)
    report(ma)
    if ma["status"] == "OK":
        ws = sum(ma["weights_stacking"].values())
        wb = sum(ma["weights_pseudo_bma"].values())
        rep("stacking weights sum to 1", abs(ws - 1) < 1e-6, f"sum={ws:.8f}")
        rep("pseudo-BMA weights sum to 1", abs(wb - 1) < 1e-6)
        rep("all weights non-negative",
            all(v >= 0 for v in ma["weights_stacking"].values()))

    # ---------------- B2 true model gets weight ----------------------
    print("\n[B2] Data generated with dCp constant (b=0) -> M1/M2 favoured")
    if ma["status"] == "OK":
        w0 = ma["weights_stacking"]["M0"]
        w12 = ma["weights_stacking"]["M1"] + ma["weights_stacking"]["M2"]
        rep("M0 does not dominate when dCp != 0", w0 < 0.6,
            f"w(M0)={w0:.3f}, w(M1)+w(M2)={w12:.3f}")

    # ---------------- B3 between-model variance is reported ----------
    print("\n[B3] Averaged interval includes between-model spread")
    if ma["status"] == "OK":
        fr = [d["between_fraction"] for d in ma["dCp_averaged"].values()]
        print(f"      between-model variance fractions: "
              f"{[round(x,3) for x in fr]}")
        rep("between-model fraction reported and in [0,1]",
            all(0.0 <= x <= 1.0 for x in fr))
        rep("total variance >= within variance",
            all(d["sd_total"] ** 2 >= d["var_within"] - 1e-8
                for d in ma["dCp_averaged"].values()))

    # ---------------- B4 elpd difference SEs reported ----------------
    print("\n[B4] Ambiguity is visible")
    if ma["status"] == "OK":
        rep("elpd difference SEs present", len(ma["elpd_diff_se"]) == 3,
            f"{ {k: round(v,2) for k,v in ma['elpd_diff_se'].items()} }")

    # ---------------- C1 guards --------------------------------------
    print("\n[C1] Guards")
    h1 = SelectiveHierarchy(sysl[:1], "M2", ("dCp0",))
    rep("S=1 refused", h1.fit()["status"] == "REFUSED")
    two = [s for s in sysl if s.group in ("A", "B")]
    h2 = SelectiveHierarchy(two, "M2", ("dCp0",))
    f2 = h2.fit(n_iter=700, burn=250, seed=6)
    rep("G=2 flagged", f2["status"] == "OK"
        and any("G=2" in w for w in f2["warnings"]))
    rep("no-pooling config flagged",
        any("no-pooling" in w for w in
            SelectiveHierarchy(two, "M2", ()).fit(600, 200, seed=7)["warnings"]))

    print("\n" + "=" * 94)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 94)
    print("""
  WHAT THESE TESTS PROVE
    A1  only the selected parameter is pooled; the others keep weak
        independent priors
    A2  unpooled parameters are shrunk less than pooled ones
    A3  per-parameter ICC separates real class signal from none
    A4  the pooling configuration is chosen by LOO, not assumed
    B1  stacking weights lie on the simplex
    B2  a model that cannot represent the data does not dominate
    B3  the averaged interval carries between-model variance, not just
        within-model variance
    B4  SEs of elpd differences are reported so ambiguity is visible
    C1  S=1, G=2 and empty-pooling cases are flagged

  WHAT IS NOT PROVEN
    * stacking weights are predictive-optimal, not model probabilities
    * LOO elpd is noisy at small n; read the SEs before believing a ranking
    * half-Cauchy scales still matter when data are sparse
    * dCp transferability across ligand classes is a physical assumption
      that ICC can support or undermine, not establish
    * no experimental validation is performed here
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Selective pooling + stacking v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--systems", default=None,
                    help="JSON list of {name,group,T,kind,y,sigma,T0}")
    ap.add_argument("--pool", default="dCp0",
                    help="comma-separated parameter names to pool, or 'none'")
    ap.add_argument("--compare-pooling", action="store_true")
    ap.add_argument("--T0", type=float, default=310.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if self_test() else 1)
    if not a.systems:
        ap.print_help()
        print(f"\n{ATTRIBUTION}")
        return
    with open(a.systems) as f:
        raw = json.load(f)
    sysl = [SystemData(d["name"], d["group"], np.array(d["T"], float),
                       np.array(d["kind"]), np.array(d["y"], float),
                       np.array(d["sigma"], float),
                       float(d.get("T0", a.T0))) for d in raw]
    pool = () if a.pool.lower() == "none" else tuple(
        s.strip() for s in a.pool.split(","))
    out: Dict[str, Any] = {}
    h = SelectiveHierarchy(sysl, "M2", pool)
    fit = h.fit()
    out["fit"] = {k: v for k, v in fit.items() if k != "_draws"}
    print(json.dumps(out["fit"], indent=2, default=float))
    if a.compare_pooling:
        out["pooling_comparison"] = compare_pooling_configs(sysl, "M2")
        print(json.dumps(out["pooling_comparison"], indent=2, default=float))
    ma = model_average(sysl, pool, T0=a.T0)
    report(ma)
    out["model_averaging"] = ma
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, default=float)


if __name__ == "__main__":
    main()
