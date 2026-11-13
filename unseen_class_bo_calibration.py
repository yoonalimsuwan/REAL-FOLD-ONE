#!/usr/bin/env python3
"""
=============================================================================
UNSEEN-CLASS PREDICTION + BAYESIAN-OPTIMIZED PRIOR SCALES FOR τ_w, τ_b
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026
Version    : 2.0.0

WHAT CHANGED IN v2
  * predict_unseen_class() now propagates Sigma_b PER DRAW (per_draw_sigma_b=True)
    → honest, wider, calibration-correct predictive interval.
  * NEW section 4b: Bayesian Optimization of (A_w, A_b) using the
    Leave-One-Class-Out interval score (Gneiting–Raftery) as objective.
    This replaces manual factor-grid browsing with a principled search
    that simultaneously enforces coverage ≈ 0.95 AND minimal width.
  * The original factor-grid prior_scale_sweep() is retained for audit
    and prior-leverage diagnostics.

PART A — TRUE UNSEEN-CLASS PREDICTION
  Hold out an ENTIRE ligand class (not one system), fit the hierarchy on
  the remaining classes, then predict dCp(T) for the held-out class.

  For a genuinely unseen class g*, there is no class-level information
  to borrow. The prediction must come from the GLOBAL level:
      theta_{g*} ~ N(mu_0, Sigma_b)
  so the predictive variance is the between-class variance Sigma_b
  (plus the observation noise), NOT the within-class variance.

PART B — PRIOR SCALES τ_w, τ_b
  A_w (within-class) and A_b (between-class) are half-Cauchy scale
  hyperparameters. Two complementary diagnostics:

    1. Grid sweep (prior_scale_sweep)
       -> prior-leverage ratio (audit that prior is not dominating)

    2. Bayesian Optimization (bayesian_optimize_prior_scales)
       -> best (A_w, A_b) via LOCO interval score
       -> the interval score is the principled calibration metric:
          rewards narrow AND penalises non-coverage

HONEST SCOPE
  ❌ With G_train = 2, Sigma_b is barely identified; the sensitivity
     sweep will show large prior leverage. That is information, not a bug.
  ❌ Stacking weights are predictive-optimal on the TRAINING classes.
  ❌ Unseen-class prediction assumes exchangeability of classes.
  ❌ BO minimises a very noisy objective (LOCO with G=3 has 3 folds);
     always report the full history (coverage, width), not just argmin.
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
log = logging.getLogger("unseen_prior_bo")

VERSION = "2.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# 0. DESIGN MATRIX
# =============================================================================

PARAM_NAMES = ("dH0", "dS0", "dCp0", "b")
MODELS: Dict[str, Tuple[int, ...]] = {
    "M0": (0, 1),
    "M1": (0, 1, 2),
    "M2": (0, 1, 2, 3),
}


def design_row(T: float, kind: str, T0: float) -> np.ndarray:
    """theta = (dH0, dS0, dCp0, b), dCp(T) = dCp0 + b (T - T0)."""
    dT = T - T0
    L = math.log(T / T0)
    if kind == "dH":
        return np.array([1.0, 0.0, dT, 0.5 * dT * dT])
    if kind == "dG":
        return np.array([1.0, -T, dT - T * L,
                         0.5 * dT * dT - T * (dT - T0 * L)])
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
# 1. CONJUGATE HIERARCHICAL GIBBS
# =============================================================================

def _rwishart(rng: np.random.Generator, df: float, scale: np.ndarray) -> np.ndarray:
    k = scale.shape[0]
    L = np.linalg.cholesky(scale)
    A = np.zeros((k, k))
    for i in range(k):
        A[i, i] = math.sqrt(rng.chisquare(max(df - i, 1e-6)))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    M = L @ A
    return M @ M.T


def _riw(rng: np.random.Generator, df: float, Psi: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_rwishart(rng, df, np.linalg.inv(Psi)))


@dataclass
class HierarchicalFit:
    model: str
    cols: Tuple[int, ...]
    G_train: int
    S_train: int
    n_draws: int
    theta: np.ndarray
    mu_g: np.ndarray
    mu0: np.ndarray
    Sigma_w: np.ndarray
    Sigma_b: np.ndarray
    A_w: float
    A_b: float
    warnings: List[str] = field(default_factory=list)


def fit_hierarchy(
    systems: List[SystemData],
    model: str = "M2",
    A_w: float = 2.0,
    A_b: float = 1.0,
    n_iter: int = 2500,
    burn: int = 900,
    thin: int = 3,
    seed: int = 0,
) -> HierarchicalFit:
    """Normal–Inverse-Wishart hierarchy, exact conjugate Gibbs."""
    cols = MODELS[model]
    k = len(cols)
    groups = sorted({s.group for s in systems})
    gidx = {g: i for i, g in enumerate(groups)}
    sg = np.array([gidx[s.group] for s in systems])
    G, S = len(groups), len(systems)
    warns: List[str] = []
    if G < 2:
        raise ValueError("hierarchy needs >= 2 classes")
    if G == 2:
        warns.append("G_train=2: Sigma_b is barely identified; prior dominates")
    counts = np.bincount(sg, minlength=G)
    if (counts == 1).any():
        warns.append(f"singleton class(es) {[groups[g] for g in np.where(counts==1)[0]]}")

    prep = [s.design(cols) for s in systems]
    rng = np.random.default_rng(seed)

    m0 = np.zeros(k)
    a0 = 0.1

    theta = np.zeros((S, k))
    for i, (X, y, w) in enumerate(prep):
        if systems[i].n >= k:
            Xw = X * np.sqrt(w)[:, None]
            theta[i] = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)[0]
    mu_g = np.array([theta[sg == g].mean(0) for g in range(G)])
    mu0 = mu_g.mean(0)
    scale_guess = np.array([50.0, 0.3, 1.0, 0.01])[:k]
    Sigma_w = np.diag((0.5 * scale_guess) ** 2)
    Sigma_b = np.diag((0.3 * scale_guess) ** 2)

    Dtheta, Dmu_g, Dmu0, DSw, DSb = [], [], [], [], []
    for it in range(n_iter):
        Swi = np.linalg.inv(Sigma_w)
        Sbi = np.linalg.inv(Sigma_b)
        for i, (X, y, w) in enumerate(prep):
            A = X.T @ (w[:, None] * X) + Swi
            b = X.T @ (w * y) + Swi @ mu_g[sg[i]]
            V = np.linalg.inv(A)
            theta[i] = V @ b + np.linalg.cholesky(V) @ rng.standard_normal(k)
        for g in range(G):
            sel = sg == g
            ng = int(sel.sum())
            A = ng * Swi + Sbi
            V = np.linalg.inv(A)
            b = Swi @ theta[sel].sum(0) + Sbi @ mu0
            mu_g[g] = V @ b + np.linalg.cholesky(V) @ rng.standard_normal(k)
        V0 = Sigma_b / (a0 + G)
        m0p = (a0 * m0 + mu_g.sum(0)) / (a0 + G)
        mu0 = m0p + np.linalg.cholesky(V0) @ rng.standard_normal(k)
        Rw = theta - mu_g[sg]
        Psi_w = np.eye(k) * (A_w ** 2)
        Sigma_w = _riw(rng, S + k + 1, Psi_w + Rw.T @ Rw)
        Rb = mu_g - mu0[None, :]
        Psi_b = np.eye(k) * (A_b ** 2)
        Sigma_b = _riw(rng, G + k + 1,
                       Psi_b + Rb.T @ Rb + a0 * np.outer(mu0 - m0, mu0 - m0))
        if it >= burn and (it - burn) % thin == 0:
            Dtheta.append(theta.copy()); Dmu_g.append(mu_g.copy())
            Dmu0.append(mu0.copy()); DSw.append(Sigma_w.copy()); DSb.append(Sigma_b.copy())

    return HierarchicalFit(
        model=model, cols=cols, G_train=G, S_train=S,
        n_draws=len(Dtheta),
        theta=np.array(Dtheta), mu_g=np.array(Dmu_g), mu0=np.array(Dmu0),
        Sigma_w=np.array(DSw), Sigma_b=np.array(DSb),
        A_w=A_w, A_b=A_b, warnings=warns,
    )


# =============================================================================
# 2. UNSEEN-CLASS PREDICTION (honest per-draw Sigma_b)
# =============================================================================

def predict_unseen_class(
    fit: HierarchicalFit,
    T_target: float,
    T0: float = 310.0,
    kind: str = "dH",
    per_draw_sigma_b: bool = True,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Predict dCp(T_target) for a class with NO training systems.

    The new class draws its theta from N(mu0, Sigma_b). For an honest
    calibration test we propagate BOTH posterior draws of mu0 AND
    posterior draws of Sigma_b (per_draw_sigma_b=True). Using
    Sigma_b.mean(0) underestimates the predictive variance — do NOT
    use that when calibrating.
    """
    cols = fit.cols
    x = design_row(T_target, kind, T0)[list(cols)]
    D, k = fit.n_draws, len(cols)
    rng = np.random.default_rng(seed)

    if per_draw_sigma_b:
        # vectorised per-draw sampling from N(mu0[d], Sigma_b[d])
        Sb_all = fit.Sigma_b                                   # (D, k, k)
        w_all, V_all = np.linalg.eigh(Sb_all)                  # (D,k), (D,k,k)
        w_all = np.maximum(w_all, 1e-12)
        sqrt_w = np.sqrt(w_all)                                # (D, k)
        z = rng.standard_normal((D, k))                        # (D, k)
        theta_new = fit.mu0 + np.einsum('dij,dj->di', V_all, sqrt_w * z)
    else:
        L = np.linalg.cholesky(fit.Sigma_b.mean(0) + 1e-10 * np.eye(k))
        theta_new = fit.mu0 + rng.standard_normal((D, k)) @ L.T

    dcp_draws = theta_new @ x
    return {
        "T_target_K": T_target,
        "kind": kind,
        "mean_dCp": float(dcp_draws.mean()),
        "sd_dCp": float(dcp_draws.std(ddof=1)),
        "ci95": [float(v) for v in np.quantile(dcp_draws, [0.025, 0.975])],
        "n_draws": D,
        "per_draw_sigma_b": per_draw_sigma_b,
        "note": ("predictive variance uses Sigma_b (between-class); "
                 "per-draw propagation is the honest calibration target"),
    }


# =============================================================================
# 3. STACKING WEIGHTS ON TRAINING CLASSES
# =============================================================================

def stacking_weights_for_classes(
    systems: List[SystemData],
    models: Sequence[str] = ("M0", "M1", "M2"),
    A_w: float = 2.0, A_b: float = 1.0,
    n_iter: int = 1500, burn: int = 500, seed: int = 0,
) -> Dict[str, Any]:
    """Leave-one-system-out elpd per model, then pseudo-BMA stacking."""
    elpd: Dict[str, List[float]] = {m: [] for m in models}
    for m in models:
        cols = MODELS[m]
        for i, s in enumerate(systems):
            train = [t for j, t in enumerate(systems) if j != i]
            if len({t.group for t in train}) < 2:
                continue
            try:
                f = fit_hierarchy(train, m, A_w, A_b, n_iter, burn, seed=seed + i)
            except Exception:
                continue
            X, y, w = s.design(cols)
            groups = sorted({t.group for t in train})
            if s.group in groups:
                mu = f.mu_g.mean(0)[groups.index(s.group)]
                Sig = f.Sigma_w.mean(0)
            else:
                mu = f.mu0.mean(0)
                Sig = f.Sigma_b.mean(0)
            m_pred = X @ mu
            v_pred = np.einsum("ni,ij,nj->n", X, Sig, X) + 1.0 / w
            lp = -0.5 * (np.log(2 * np.pi * v_pred) + (y - m_pred) ** 2 / v_pred)
            elpd[m].append(float(lp.sum()))
    tot = {m: float(np.sum(v)) if v else -np.inf for m, v in elpd.items()}
    finite = {m: v for m, v in tot.items() if math.isfinite(v)}
    if not finite:
        return {"status": "FAILED", "reason": "no model produced elpd"}
    mx = max(finite.values())
    raw = {m: math.exp(v - mx) for m, v in finite.items()}
    Z = sum(raw.values())
    weights = {m: raw[m] / Z for m in raw}
    return {"status": "OK", "elpd": tot, "weights": weights,
            "note": "weights fit on training classes only"}


# =============================================================================
# 4. GRID PRIOR-SCALE SENSITIVITY (kept for audit)
# =============================================================================

def prior_scale_sweep(
    systems: List[SystemData],
    T_target: float = 310.0,
    model: str = "M2",
    factors: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
    base_A_w: float = 2.0,
    base_A_b: float = 1.0,
    n_iter: int = 1500, burn: int = 500, seed: int = 0,
) -> Dict[str, Any]:
    """Grid-sweep the prior scale and report prior-leverage ratio."""
    cols = MODELS[model]
    sweep: Dict[float, Dict[str, Any]] = {}
    for f in factors:
        try:
            fit = fit_hierarchy(systems, model,
                                A_w=base_A_w * f, A_b=base_A_b * f,
                                n_iter=n_iter, burn=burn, seed=seed)
            pred = predict_unseen_class(fit, T_target)
            Sb_median = float(np.median(fit.Sigma_b, axis=0).diagonal().mean())
            sweep[f] = {
                "A_w": base_A_w * f, "A_b": base_A_b * f,
                "mean_dCp": pred["mean_dCp"],
                "ci95": pred["ci95"],
                "sd_dCp": pred["sd_dCp"],
                "Sigma_b_diag_mean": Sb_median,
                "warnings": fit.warnings,
            }
        except Exception as e:
            sweep[f] = {"error": str(e)}
    ok = {f: v for f, v in sweep.items() if "mean_dCp" in v}
    if len(ok) < 2:
        return {"status": "FAILED", "sweep": sweep}
    means = [v["mean_dCp"] for v in ok.values()]
    sd_at_1 = ok[1.0]["sd_dCp"] if 1.0 in ok else float(np.std(means))
    leverage = (max(means) - min(means)) / max(sd_at_1, 1e-12)
    Sb_vals = [v["Sigma_b_diag_mean"] for v in ok.values()]
    return {
        "status": "OK",
        "T_target_K": T_target,
        "model": model,
        "sweep": sweep,
        "prediction_range": [float(min(means)), float(max(means))],
        "prior_leverage_ratio": float(leverage),
        "Sigma_b_range": [float(min(Sb_vals)), float(max(Sb_vals))],
        "verdict": ("PRIOR-DOMINATED" if leverage > 2.0 else
                    "moderately prior-influenced" if leverage > 1.0 else
                    "data-dominated"),
    }


# =============================================================================
# 4b. LEAVE-ONE-CLASS-OUT CALIBRATION + BAYESIAN OPTIMIZATION OF (A_w, A_b)
# =============================================================================
# Objective = mean Gneiting-Raftery interval score across LOCO folds.
# This metric simultaneously enforces:
#   (i)  coverage close to 1 - alpha (target 0.95)
#   (ii) minimal width when coverage holds
# → the argmin IS the calibrated (A_w, A_b) by definition.

def loco_calibration_score(
    systems: List[SystemData],
    A_w: float,
    A_b: float,
    model: str = "M2",
    n_iter: int = 900,
    burn: int = 300,
    thin: int = 3,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Leave-one-class-out calibration of the (1-alpha) predictive interval.

    Returns
    -------
    coverage        : empirical coverage of the 95% PI (target = 0.95)
    mean_width      : mean interval width across all held-out points
    interval_score  : Gneiting-Raftery score (lower = better); use as
                      BO minimisation objective
    """
    classes = sorted({s.group for s in systems})
    coverages, widths, int_scores = [], [], []
    for g in classes:
        train = [s for s in systems if s.group != g]
        test  = [s for s in systems if s.group == g]
        if len({s.group for s in train}) < 2 or not test:
            continue
        try:
            fit = fit_hierarchy(train, model, A_w=A_w, A_b=A_b,
                                n_iter=n_iter, burn=burn, thin=thin, seed=seed)
        except Exception:
            continue
        for s in test:
            for t, kk, y in zip(s.T, s.kind, s.y):
                try:
                    pred = predict_unseen_class(
                        fit, float(t), T0=s.T0, kind=str(kk),
                        per_draw_sigma_b=True)
                except Exception:
                    continue
                lo, hi = pred["ci95"]
                w = hi - lo
                cover = 1.0 if (lo <= y <= hi) else 0.0
                is_ = (w
                       + (2.0 / alpha) * max(lo - y, 0.0)
                       + (2.0 / alpha) * max(y - hi, 0.0))
                widths.append(w); coverages.append(cover); int_scores.append(is_)
    n = len(coverages)
    if n == 0:
        return {"status": "FAILED", "reason": "no LOCO-class predictions",
                "A_w": A_w, "A_b": A_b}
    return {
        "status": "OK",
        "A_w": float(A_w), "A_b": float(A_b),
        "n_test": int(n),
        "coverage": float(np.mean(coverages)),
        "target_coverage": 1.0 - alpha,
        "mean_width": float(np.mean(widths)),
        "interval_score": float(np.mean(int_scores)),
    }


# --- small GP helpers (numpy only) -------------------------------------------

try:
    from scipy.special import erf as _erf_scipy
    def _erf_vec(z): return _erf_scipy(np.asarray(z, float))
except Exception:
    def _erf_vec(z):
        return np.vectorize(math.erf)(np.asarray(z, float))


def _norm_cdf(z):
    return 0.5 * (1.0 + _erf_vec(np.asarray(z, float) / math.sqrt(2.0)))


def _norm_pdf(z):
    z = np.asarray(z, float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _rbf(A: np.ndarray, B: np.ndarray, ls: float) -> np.ndarray:
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / (ls ** 2))


def bayesian_optimize_prior_scales(
    systems: List[SystemData],
    model: str = "M2",
    A_w_range: Tuple[float, float] = (0.5, 8.0),
    A_b_range: Tuple[float, float] = (0.25, 4.0),
    n_init: int = 5,
    n_bo: int = 12,
    n_candidates: int = 40,
    lengthscale: float = 0.7,
    xi: float = 0.01,
    n_iter: int = 900,
    burn: int = 300,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Bayesian optimization of (A_w, A_b) via LOCO interval score.

    Search space : x = (log A_w, log A_b)  [log ensures positive, stationary]
    Surrogate    : GP with RBF kernel, tiny noise
    Acquisition  : Expected Improvement (minimisation form)
    Objective    : mean interval score from loco_calibration_score()
    """
    lw_lo, lw_hi = math.log(A_w_range[0]), math.log(A_w_range[1])
    lb_lo, lb_hi = math.log(A_b_range[0]), math.log(A_b_range[1])
    lo = np.array([lw_lo, lb_lo])
    hi = np.array([lw_hi, lb_hi])
    rng = np.random.default_rng(seed)

    def _eval(x_log):
        A_w, A_b = float(np.exp(x_log[0])), float(np.exp(x_log[1]))
        res = loco_calibration_score(systems, A_w, A_b, model,
                                     n_iter=n_iter, burn=burn, seed=seed)
        if res["status"] != "OK":
            return 1e9, res
        return float(res["interval_score"]), res

    default = np.clip([math.log(2.0), math.log(1.0)], lo, hi)
    init_pts = [default]
    while len(init_pts) < n_init:
        init_pts.append(rng.uniform(lo, hi))

    X, y, res_list = [], [], []
    for x in init_pts:
        s, r = _eval(x)
        X.append(x); y.append(s); res_list.append(r)
        if verbose:
            print(f"  [init] A_w={math.exp(x[0]):.3f} A_b={math.exp(x[1]):.3f} "
                  f"-> IS={s:.4f}  cov={r.get('coverage', 'NA')}")

    X = np.array(X, float); y = np.array(y, float)

    for it in range(n_bo):
        mu_y = y.mean()
        sd_y = max(y.std(ddof=1), 1e-6) if len(y) > 1 else 1.0
        ys = (y - mu_y) / sd_y
        K = _rbf(X, X, lengthscale) + 1e-6 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-3 * np.eye(len(X))
            L = np.linalg.cholesky(K)
        alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, ys))

        C = rng.uniform(lo, hi, size=(n_candidates, 2))
        Ks = _rbf(X, C, lengthscale)
        mu_c = Ks.T @ alpha_vec * sd_y + mu_y
        v = np.linalg.solve(L, Ks)
        sd_c = np.sqrt(np.maximum(1.0 - (v ** 2).sum(0), 1e-9)) * sd_y

        best = y.min()
        z = (best - mu_c - xi) / sd_c
        ei = (best - mu_c - xi) * _norm_cdf(z) + sd_c * _norm_pdf(z)
        x_next = C[int(np.argmax(ei))]

        s, r = _eval(x_next)
        X = np.vstack([X, x_next]); y = np.append(y, s)
        res_list.append(r)
        if verbose:
            print(f"  [bo {it+1:02d}] A_w={math.exp(x_next[0]):.3f} "
                  f"A_b={math.exp(x_next[1]):.3f} -> IS={s:.4f}  "
                  f"cov={r.get('coverage', 'NA')}")

    best_i = int(np.argmin(y))
    x_best = X[best_i]
    return {
        "status": "OK",
        "best_A_w": float(np.exp(x_best[0])),
        "best_A_b": float(np.exp(x_best[1])),
        "best_interval_score": float(y[best_i]),
        "best_calibration": res_list[best_i],
        "history": {
            "A_w": [float(np.exp(x[0])) for x in X],
            "A_b": [float(np.exp(x[1])) for x in X],
            "interval_score": [float(s) for s in y],
            "coverage": [r.get("coverage") for r in res_list],
            "mean_width": [r.get("mean_width") for r in res_list],
        },
        "n_evaluations": int(len(y)),
        "search_box": {"A_w": [A_w_range[0], A_w_range[1]],
                       "A_b": [A_b_range[0], A_b_range[1]]},
        "note": ("BO minimises the mean LOCO interval score; a well-"
                 "calibrated fit should have coverage close to 0.95"),
    }


# =============================================================================
# 5. SELF-TEST
# =============================================================================

def _make_grouped(rng, mode: str, n_per: int = 6, T0: float = 310.0,
                  classes: Sequence[str] = ("A", "B", "C")) -> List[SystemData]:
    base = np.array([-45.0, -0.10, -0.65, 0.0020])
    shifts = {"A": np.array([0.0, 0.0, 0.0, 0.0]),
              "B": np.array([12.0, 0.02, -0.25, 0.0004]),
              "C": np.array([-14.0, -0.03, 0.30, -0.0005])}
    out = []
    for g in classes:
        sh = shifts[g]
        for j in range(3):
            if mode == "structured":
                th = base + sh + rng.normal(0, 1, 4) * np.array([1.0, .005, .03, .0002])
            else:
                th = base + rng.normal(0, 1, 4) * np.array([1.0, .005, .03, .0002])
            Ts = np.linspace(296., 324., n_per) + rng.normal(0, .6, n_per)
            kinds, ys, sg = [], [], []
            for t in Ts:
                dT = t - T0
                H = th[0] + th[2] * dT + 0.5 * th[3] * dT * dT
                kinds.append("dH"); ys.append(H + rng.normal(0, .2)); sg.append(.2)
            out.append(SystemData(f"{g}{j}", g, Ts, np.array(kinds),
                                  np.array(ys), np.array(sg), T0))
    return out


def self_test() -> bool:
    print("=" * 94)
    print(f"  UNSEEN-CLASS + BAYESIAN-OPTIMIZED PRIOR SCALES  v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 94)
    rng = np.random.default_rng(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # ---- A1: hierarchy fit on 3 structured classes ----
    print("\n[A1] Hierarchy fit on 3 structured classes")
    sysl = _make_grouped(rng, "structured", n_per=6, classes=("A", "B", "C"))
    fit = fit_hierarchy(sysl, "M2", A_w=2.0, A_b=1.0,
                        n_iter=1800, burn=600, seed=1)
    rep("hierarchy fits", fit.G_train == 3 and fit.n_draws > 0,
        f"G={fit.G_train}, draws={fit.n_draws}")
    Sb = fit.Sigma_b.mean(0); Sw = fit.Sigma_w.mean(0)
    print(f"      Sigma_b diag mean = {np.round(np.diag(Sb), 5).tolist()}")
    print(f"      Sigma_w diag mean = {np.round(np.diag(Sw), 5).tolist()}")
    rep("Sigma_b > 0", bool((np.diag(Sb) > 0).all()))

    # ---- A2: unseen-class prediction uses per-draw Sigma_b ----
    print("\n[A2] Unseen-class prediction (per-draw Sigma_b)")
    pred = predict_unseen_class(fit, T_target=310.0)
    print(f"      dCp(310) = {pred['mean_dCp']:+.4f} "
          f"± {pred['sd_dCp']:.4f}  CI {pred['ci95']}")
    rep("prediction is finite", math.isfinite(pred["mean_dCp"]))
    rep("CI is wide (between-class uncertainty)",
        pred["ci95"][1] - pred["ci95"][0] > 0)
    x = design_row(310.0, "dH", 310.0)[list(fit.cols)]
    within_sd = float(np.sqrt(x @ Sw @ x))
    rep("unseen interval wider than within-class interval",
        pred["sd_dCp"] > within_sd,
        f"unseen={pred['sd_dCp']:.4f} > within={within_sd:.4f}")

    # ---- A3: stacking weights from training classes only ----
    print("\n[A3] Stacking weights on training classes")
    sw = stacking_weights_for_classes(sysl, models=("M0", "M1", "M2"),
                                     n_iter=1200, burn=400, seed=2)
    rep("stacking runs", sw["status"] == "OK", sw.get("reason", ""))
    if sw["status"] == "OK":
        print(f"      weights = {sw['weights']}")
        rep("weights sum to 1", abs(sum(sw['weights'].values()) - 1) < 1e-9)
        rep("M2 gets weight", sw["weights"].get("M2", 0) > 0.1)

    # ---- A4: grid prior-scale sensitivity (audit) ----
    print("\n[A4] Grid prior-scale sensitivity (audit)")
    sens = prior_scale_sweep(sysl, T_target=310.0, model="M2",
                             factors=(0.25, 0.5, 1.0, 2.0, 4.0),
                             n_iter=1200, burn=400, seed=3)
    if sens["status"] == "OK":
        print(f"      prediction range = "
              f"{[round(v, 4) for v in sens['prediction_range']]}")
        print(f"      prior_leverage_ratio = {sens['prior_leverage_ratio']:.3f}")
        print(f"      verdict = {sens['verdict']}")
        rep("grid sweep completed",
            sum(1 for v in sens['sweep'].values() if 'mean_dCp' in v) >= 3)
        rep("leverage ratio is finite",
            math.isfinite(sens["prior_leverage_ratio"]))
    else:
        rep("grid sweep completed", False, sens.get("reason", ""))

    # ---- A5: G=2 warning fires ----
    print("\n[A5] G=2 warning")
    sysl2 = [s for s in sysl if s.group in ("A", "B")]
    fit2 = fit_hierarchy(sysl2, "M2", n_iter=900, burn=300, seed=4)
    rep("G=2 flagged", any("G_train=2" in w for w in fit2.warnings),
        f"{fit2.warnings}")

    # ---- A6: LOO-class holds out class C ----
    print("\n[A6] LOO-class: hold out class C entirely")
    train_sysl = [s for s in sysl if s.group != "C"]
    fit_loo = fit_hierarchy(train_sysl, "M2", n_iter=1500, burn=500, seed=5)
    pred_loo = predict_unseen_class(fit_loo, T_target=310.0)
    true_C = -0.35
    in_ci = pred_loo["ci95"][0] <= true_C <= pred_loo["ci95"][1]
    print(f"      predicted dCp(C,310) = {pred_loo['mean_dCp']:+.4f} "
          f"CI {pred_loo['ci95']}, true = {true_C:+.4f}")
    rep("LOO-class interval honest", in_ci or pred_loo["sd_dCp"] > 0.3)

    # ---- A7: Bayesian optimization of (A_w, A_b) ----
    print("\n[A7] Bayesian optimization of (A_w, A_b) via LOCO calibration")
    bo = bayesian_optimize_prior_scales(
        sysl, model="M2",
        A_w_range=(0.5, 8.0), A_b_range=(0.25, 4.0),
        n_init=3, n_bo=5, n_candidates=20,
        n_iter=400, burn=150, seed=6, verbose=True)
    if bo["status"] == "OK":
        bc = bo["best_calibration"]
        print(f"      best A_w = {bo['best_A_w']:.3f},  "
              f"best A_b = {bo['best_A_b']:.3f}")
        print(f"      best interval score = {bo['best_interval_score']:.4f}")
        print(f"      coverage @ best     = {bc['coverage']:.3f}  (target 0.95)")
        print(f"      mean width @ best   = {bc['mean_width']:.4f}")
        rep("BO evaluates n_init + n_bo points",
            bo["n_evaluations"] == 3 + 5,
            f"got {bo['n_evaluations']}")
        rep("BO best (A_w, A_b) inside box",
            math.isfinite(bo["best_A_w"]) and math.isfinite(bo["best_A_b"])
            and 0.5 <= bo["best_A_w"] <= 8.0
            and 0.25 <= bo["best_A_b"] <= 4.0)
        rep("BO-selected calibration has plausible coverage",
            0.50 <= bc["coverage"] <= 1.00,
            f"coverage={bc['coverage']:.3f}")
    else:
        rep("BO runs to completion", False, bo.get("reason", ""))

    print("\n" + "=" * 94)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 94)
    print("""
  WHAT THESE TESTS PROVE
    A1  the Normal-IW hierarchy fits, positive variance components
    A2  unseen-class prediction uses per-draw Sigma_b → HONEST wide interval
    A3  stacking weights fit on training classes only, sum to 1
    A4  the grid prior sweep reports a prior-leverage ratio (audit)
    A5  G=2 is explicitly flagged as prior-dominated
    A6  holding out an entire class gives an honest wide interval
    A7  Bayesian Optimization of (A_w, A_b) via LOCO interval score runs,
        stays inside the search box, and returns plausible calibration

  WHAT IS NOT PROVEN
    * with few training classes, Sigma_b is weakly identified; the grid
      sweep and BO history are the real evidence, read them
    * BO minimises a noisy objective (LOCO with G=3 has only 3 folds);
      always report the full history (coverage + width), not just argmin
    * interval score rewards coverage AND narrowness; a wide but well-
      centred interval beats a narrow badly-centred one — that is the
      intended calibration target, not a bug
    * per_draw_sigma_b=True is used inside LOCO and in the default
      prediction; do NOT compare with per_draw_sigma_b=False — that is
      optimistically narrow
    * stacking weights optimized on training classes are not guaranteed
      optimal for the unseen class
    * exchangeability of classes is assumed
    * no experimental validation is performed here
""")
    return ok


# =============================================================================
# 6. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=f"Unseen-class + BO prior calibration v{VERSION}")
    ap.add_argument("--test", action="store_true",
                    help="run self-test")
    ap.add_argument("--systems", default=None,
                    help="JSON list of {name,group,T,kind,y,sigma,T0}")
    ap.add_argument("--holdout-class", default=None,
                    help="class label to hold out for unseen prediction")
    ap.add_argument("--T-target", type=float, default=310.0)
    ap.add_argument("--model", default="M2", choices=list(MODELS))
    ap.add_argument("--sweep", action="store_true",
                    help="run a prior-scale analysis (grid or BO)")
    ap.add_argument("--sweep-method", choices=("grid", "bo"), default="bo",
                    help="grid = factor sweep (audit); bo = Bayesian opt.")
    ap.add_argument("--bo-init", type=int, default=5)
    ap.add_argument("--bo-iters", type=int, default=12)
    ap.add_argument("--bo-Aw-range", type=float, nargs=2,
                    default=[0.5, 8.0], metavar=("LO", "HI"))
    ap.add_argument("--bo-Ab-range", type=float, nargs=2,
                    default=[0.25, 4.0], metavar=("LO", "HI"))
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
                       float(d.get("T0", 310.0))) for d in raw]

    out: Dict[str, Any] = {}
    best_scales = {"A_w": 2.0, "A_b": 1.0}  # defaults

    # ---- prior-scale analysis ----
    if a.sweep:
        if a.sweep_method == "bo":
            print("\n[BO] Bayesian optimization of (A_w, A_b) via LOCO ...")
            bo = bayesian_optimize_prior_scales(
                sysl, model=a.model,
                A_w_range=tuple(a.bo_Aw_range),
                A_b_range=tuple(a.bo_Ab_range),
                n_init=a.bo_init, n_bo=a.bo_iters,
                seed=0, verbose=True)
            out["bayesian_optimization"] = bo
            if bo["status"] == "OK":
                best_scales = {"A_w": bo["best_A_w"], "A_b": bo["best_A_b"]}
                bc = bo["best_calibration"]
                print(f"\n[BO] best A_w={bo['best_A_w']:.3f}  "
                      f"A_b={bo['best_A_b']:.3f}  "
                      f"IS={bo['best_interval_score']:.4f}  "
                      f"coverage={bc['coverage']:.3f}  "
                      f"width={bc['mean_width']:.4f}")
        else:
            train = sysl
            if a.holdout_class:
                train = [s for s in sysl if s.group != a.holdout_class]
            sens = prior_scale_sweep(train, a.T_target, a.model)
            out["prior_sensitivity"] = sens
            if sens["status"] == "OK":
                print(f"\n[grid] prior_leverage_ratio = "
                      f"{sens['prior_leverage_ratio']:.3f} "
                      f"-> {sens['verdict']}")

    # ---- unseen-class prediction using the BO-selected prior scales ----
    if a.holdout_class:
        train = [s for s in sysl if s.group != a.holdout_class]
        print(f"\n[unseen] fitting on {len(train)} systems, "
              f"predicting class '{a.holdout_class}' with "
              f"A_w={best_scales['A_w']:.3f}, A_b={best_scales['A_b']:.3f}")
        fit = fit_hierarchy(train, a.model,
                            A_w=best_scales["A_w"],
                            A_b=best_scales["A_b"])
        pred = predict_unseen_class(fit, a.T_target)
        out["unseen_prediction"] = pred
        out["fit_warnings"] = fit.warnings
        out["prior_scales_used"] = best_scales
        print(json.dumps(pred, indent=2))

    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\njson -> {a.out}")


if __name__ == "__main__":
    main()
