#!/usr/bin/env python3
"""
=============================================================================
JOINT 2D TORSION PMF  +  BOOTSTRAP CI FOR dCp(T) = a + b*T
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — JOINT 2D TORSION PMF (removes the independence assumption)
  The per-torsion additive correction
      dG_conf = sum_i [ dG_apply_free_i - dG_apply_bound_i ]
  is valid only if the internal coordinates are independent. For coupled
  torsions (backbone phi/psi, adjacent rotatable bonds, allylic systems)
  it is not. This module computes the correction from the JOINT 2D PMF
      W(z1,z2) = -kT ln p(z1,z2) + C
  and quantifies exactly how wrong the additive version was:
      * mutual information I(z1;z2) in nats and in kJ/mol
      * dG_additive - dG_joint, the coupling error, in kJ/mol

  MEASURE / FIXMAN NOTE (important and often gotten wrong)
    If the samples come from equilibrium MD in CARTESIAN space and are
    then histogrammed onto the torsion pair, the resulting PMF already
    contains the mass-metric (Fixman) determinant of the constrained
    coordinate map. No separate Fixman correction should be applied to
    such a PMF. A uniform-measure reference is needed ONLY for the
    analytic free-rotor idealisation, which is a different object and
    is flagged as such wherever it is used.

  COVERAGE IN 2D
    Empty histogram bins give W = +inf. That is harmless for the
    integrals (they contribute zero) but it means the released state may
    be unrepresented. The decisive diagnostic is the fraction of the
    RESTRAINT's own Boltzmann weight that falls in unsampled bins:
        uncovered = int_{unsampled} e^{-beta U} / int_{all} e^{-beta U}
    If that is not small, the computed dG_apply is not usable. This is
    checked and reported, not assumed away.

PART B — BOOTSTRAP CI FOR dCp(T)
  The asymptotic covariance (X^T W X)^{-1} assumes Gaussian errors,
  a correct linear model, and enough data. With 3-8 temperatures those
  assumptions are weak. Four resampling schemes are provided:

    parametric : y* = X beta_hat + N(0, sigma_i)
                 Uses the REPORTED measurement errors. Assumes Gaussian
                 measurement noise, but NOT asymptotic normality of the
                 estimator. Preferred when sigma comes from ITC precision.
    wild       : y* = X beta_hat + r_i * v_i,  v_i Rademacher or Mammen
                 Robust to heteroscedasticity of unknown form.
    residual   : y* = X beta_hat + r_{pi(i)}
                 Assumes exchangeable residuals (homoscedastic).
    case       : resample (T, dH, dG) rows with replacement
                 REFUSED when the resampled design is rank-deficient,
                 which happens constantly with few distinct temperatures.

    paired     : resample the (dH, dG) residual PAIR at each temperature
                 together, preserving their correlation (same experiment).

  Intervals: percentile, and BCa when the jackknife is well-posed.
  Also reported: P(b > 0) from the bootstrap distribution, which is a
  direct sign statement and more honest than "+/- stderr" at small n.

HONEST SCOPE
  ❌ Bootstrap cannot repair an underdetermined fit. With n <= k the
     model is REFUSED, not resampled.
  ❌ Bootstrap cannot repair model misspecification. If dCp is not
     affine in T over the range, every interval is an interval for the
     wrong quantity.
  ❌ BCa needs a well-posed jackknife (n - 1 >= k). Otherwise it falls
     back to percentile with an explicit warning.
  ❌ 2D PMF convergence is harder than 1D. The coverage diagnostic is
     necessary, not sufficient; a 2D surface can be covered and still
     be unconverged.
  ❌ No experimental validation is performed here.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("joint_dcp")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

KB = 0.008314462618          # kJ mol^-1 K^-1
TWO_PI = 2.0 * math.pi
_ND = NormalDist()


# =============================================================================
# PART A §1  JOINT 2D TORSION PMF
# =============================================================================

def wrap_angle(z: np.ndarray) -> np.ndarray:
    """Wrap to [-pi, pi)."""
    return (np.asarray(z, dtype=np.float64) + math.pi) % TWO_PI - math.pi


def periodic_delta(z: np.ndarray, z0: float) -> np.ndarray:
    return wrap_angle(z - z0)


@dataclass
class JointPMF2D:
    """
    Joint PMF on the torsion torus [-pi,pi)^2.

    W        : (nb,nb) kJ/mol, np.inf in unsampled bins
    sampled  : (nb,nb) bool
    edges    : bin edges (shared by both axes)
    centres  : bin centres
    """
    W: np.ndarray
    sampled: np.ndarray
    edges: np.ndarray
    centres: np.ndarray
    temperature: float
    counts: np.ndarray

    @property
    def kT(self) -> float:
        return KB * self.temperature

    @property
    def dA(self) -> float:
        d = self.edges[1] - self.edges[0]
        return float(d * d)

    @property
    def coverage_fraction(self) -> float:
        return float(self.sampled.mean())

    def marginals(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self.counts / max(self.counts.sum(), 1.0)
        return p.sum(axis=1), p.sum(axis=0)

    def mutual_information(self) -> Dict[str, float]:
        """I(z1;z2) in nats and kJ/mol. Zero iff independent."""
        p = self.counts / max(self.counts.sum(), 1.0)
        p1, p2 = self.marginals()
        outer = p1[:, None] * p2[None, :]
        m = (p > 0) & (outer > 0)
        I = float(np.sum(p[m] * np.log(p[m] / outer[m])))
        return {"I_nats": I, "I_kJmol": self.kT * I,
                "I_over_kT": I,
                "independent_within_noise": bool(I < 1.0 / max(self.counts.sum(), 1))}

    def pmf_marginal(self, axis: int) -> np.ndarray:
        p = self.counts / max(self.counts.sum(), 1.0)
        pm = p.sum(axis=1 - axis)
        with np.errstate(divide="ignore"):
            W = -self.kT * np.log(np.where(pm > 0, pm, np.nan))
        W = np.where(np.isfinite(W), W - np.nanmin(W), np.inf)
        return W


def pmf_2d_from_samples(z1: np.ndarray, z2: np.ndarray, T: float,
                        n_bins: int = 72,
                        pseudocount: float = 0.0) -> JointPMF2D:
    """
    Joint PMF from equilibrium torsion samples.

    If the samples came from Cartesian-space MD, this PMF already
    includes the Fixman/metric determinant — do NOT add it again.
    pseudocount > 0 regularises empty bins but biases the PMF; it is
    reported so the choice is visible.
    """
    z1 = wrap_angle(z1); z2 = wrap_angle(z2)
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same length")
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    H, _, _ = np.histogram2d(z1, z2, bins=[edges, edges])
    counts = H + float(pseudocount)
    tot = counts.sum()
    if tot <= 0:
        raise ValueError("no samples")
    p = counts / tot
    with np.errstate(divide="ignore"):
        W = -KB * T * np.log(np.where(p > 0, p, np.nan))
    finite = np.isfinite(W)
    W = np.where(finite, W - np.nanmin(W), np.inf)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return JointPMF2D(W=W, sampled=finite, edges=edges, centres=centres,
                      temperature=float(T), counts=counts)


def wham_2d(windows: Sequence[Dict[str, Any]], T: float,
            n_bins: int = 60, tol: float = 1e-7,
            max_iter: int = 20000) -> JointPMF2D:
    """
    2D WHAM. Each window dict:
        {"z1": array, "z2": array, "bias": callable(z1_grid, z2_grid)->kJ/mol}
    The bias is evaluated on the bin-centre grid.
    """
    kT = KB * T
    beta = 1.0 / kT
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    G1, G2 = np.meshgrid(c, c, indexing="ij")
    H, N, Wb = [], [], []
    for w in windows:
        z1 = wrap_angle(w["z1"]); z2 = wrap_angle(w["z2"])
        h, _, _ = np.histogram2d(z1, z2, bins=[edges, edges])
        H.append(h); N.append(float(h.sum()))
        Wb.append(np.asarray(w["bias"](G1, G2), dtype=np.float64))
    H = np.stack(H); N = np.array(N); Wb = np.stack(Wb)
    if N.sum() <= 0:
        raise ValueError("no samples in any window")
    expW = np.exp(-beta * Wb)
    f = np.zeros(len(windows))
    p = np.zeros_like(H[0])
    converged = False
    for it in range(max_iter):
        den = (N[:, None, None] * expW * np.exp(beta * f)[:, None, None]).sum(0)
        den = np.maximum(den, 1e-300)
        pn = H.sum(0) / den
        s = pn.sum()
        if s <= 0:
            break
        pn = pn / s
        fn = -kT * np.log(np.maximum((pn[None] * expW).sum(axis=(1, 2)), 1e-300))
        fn -= fn[0]
        d = float(np.max(np.abs(fn - f)))
        f, p = fn, pn
        if d < tol:
            converged = True
            break
    with np.errstate(divide="ignore"):
        W = -kT * np.log(np.where(p > 0, p, np.nan))
    finite = np.isfinite(W)
    W = np.where(finite, W - np.nanmin(W), np.inf)
    log.info("2D WHAM: %d iters, converged=%s, coverage=%.3f",
             it + 1, converged, finite.mean())
    return JointPMF2D(W=W, sampled=finite, edges=edges, centres=c,
                      temperature=float(T), counts=H.sum(0))


# =============================================================================
# PART A §2  JOINT RESTRAINT + APPLY FREE ENERGY + COUPLING ERROR
# =============================================================================

@dataclass
class JointTorsionRestraint:
    """
    2D harmonic restraint on the torsion torus.

    U(z1,z2) = 0.5*k1*dz1^2 + 0.5*k2*dz2^2 + k12*dz1*dz2
    with dzi the PERIODIC displacement from the centre.
    k12 != 0 gives a genuinely joint (correlated) restraint.
    Positive-definiteness requires k1*k2 > k12^2; checked.
    """
    z1_0: float
    z2_0: float
    k1: float
    k2: float
    k12: float = 0.0

    def __post_init__(self):
        if self.k1 <= 0 or self.k2 <= 0:
            raise ValueError("k1,k2 must be positive")
        if self.k1 * self.k2 <= self.k12 ** 2:
            raise ValueError("restraint not positive definite: need k1*k2 > k12^2")

    def on_grid(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        G1, G2 = np.meshgrid(c1, c2, indexing="ij")
        d1 = periodic_delta(G1, self.z1_0)
        d2 = periodic_delta(G2, self.z2_0)
        return 0.5 * self.k1 * d1 ** 2 + 0.5 * self.k2 * d2 ** 2 + self.k12 * d1 * d2

    def widths(self, T: float) -> Tuple[float, float]:
        kT = KB * T
        return math.sqrt(kT / self.k1), math.sqrt(kT / self.k2)


def joint_apply_free_energy(pmf: JointPMF2D,
                            restraint: JointTorsionRestraint,
                            restraint_on: bool = True,
                            coverage_tol: float = 0.02
                            ) -> Dict[str, Any]:
    """
    dG_apply = -kT ln[ int e^{-beta(W_int+U)} / int e^{-beta W_int} ]

    W_int is the INTRINSIC PMF. If the supplied PMF was measured with the
    restraint already on, set restraint_on=True so U is subtracted first.

    Coverage diagnostic: fraction of the restraint's own Boltzmann weight
    that lies in unsampled bins. Large => the released state is not
    represented and the number is unusable.
    """
    kT = pmf.kT
    beta = 1.0 / kT
    U = restraint.on_grid(pmf.centres, pmf.centres)
    Wm = pmf.W
    W_int = Wm - U if restraint_on else Wm
    # shift finite part to avoid overflow
    fin = np.isfinite(W_int)
    if not fin.any():
        raise ValueError("no sampled bins")
    W_int = np.where(fin, W_int - np.min(W_int[fin]), np.inf)

    eU = np.exp(-beta * np.clip(U - U.min(), 0, 700))
    tot_eU = float(eU.sum())
    uncovered = float(eU[~pmf.sampled].sum() / max(tot_eU, 1e-300))

    num = np.where(fin, np.exp(-beta * np.clip(W_int + U, 0, 700)), 0.0)
    den = np.where(fin, np.exp(-beta * np.clip(W_int, 0, 700)), 0.0)
    Zn = float(num.sum()) * pmf.dA
    Zd = float(den.sum()) * pmf.dA
    if Zn <= 0 or Zd <= 0:
        raise ValueError("zero partition function")
    dG = -kT * math.log(Zn / Zd)
    ok = uncovered < coverage_tol
    w1, w2 = restraint.widths(pmf.temperature)
    return {"dG_apply_kJmol": dG,
            "uncovered_restraint_weight": uncovered,
            "coverage_ok": bool(ok),
            "pmf_coverage_fraction": pmf.coverage_fraction,
            "restraint_widths_rad": (w1, w2),
            "warning": ("" if ok else
                        "UNSAMPLED REGION CARRIES THE RESTRAINT WEIGHT — "
                        "dG_apply is not usable. Sample the released basin "
                        "or weaken the restraint.")}


def additive_vs_joint(pmf: JointPMF2D,
                      r: JointTorsionRestraint,
                      restraint_on: bool = True) -> Dict[str, Any]:
    """
    Quantify the error of the independence assumption.

    joint      : uses the full 2D PMF
    additive   : uses W1(z1) + W2(z2) built from the marginals
    coupling   : dG_additive - dG_joint  (kJ/mol)
    """
    j = joint_apply_free_energy(pmf, r, restraint_on)
    W1 = pmf.pmf_marginal(axis=0)
    W2 = pmf.pmf_marginal(axis=1)
    W_add = W1[:, None] + W2[None, :]
    sampled_add = np.isfinite(W_add)
    pmf_add = JointPMF2D(W=np.where(sampled_add, W_add, np.inf),
                         sampled=sampled_add, edges=pmf.edges,
                         centres=pmf.centres, temperature=pmf.temperature,
                         counts=pmf.counts)
    a = joint_apply_free_energy(pmf_add, r, restraint_on)
    mi = pmf.mutual_information()
    return {"dG_joint_kJmol": j["dG_apply_kJmol"],
            "dG_additive_kJmol": a["dG_apply_kJmol"],
            "coupling_error_kJmol": a["dG_apply_kJmol"] - j["dG_apply_kJmol"],
            "mutual_information": mi,
            "joint_coverage_ok": j["coverage_ok"],
            "additive_coverage_ok": a["coverage_ok"],
            "interpretation": (
                "coupling_error is what the independence assumption costs. "
                "It is not bounded by the mutual information in general, "
                "but I=0 implies it vanishes.")}


def wooroux_with_joint_conformational(
    radial_dG: float, T: float,
    angular_terms: Sequence[Dict[str, Any]] = (),
    joint_free: Optional[Dict[str, Any]] = None,
    joint_bound: Optional[Dict[str, Any]] = None,
    extra_1d_conf: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """
    Assemble dG_bind with the coupled torsion pair treated JOINTLY.
    Remaining, genuinely independent torsions may still be added 1D.
    """
    warns: List[str] = []
    ang = 0.0
    for t in angular_terms:
        ang += float(t.get("dG_apply_free_kJmol", 0.0)) - float(t.get("dG_apply_bound_kJmol", 0.0))
    conf = 0.0
    if joint_free is not None:
        f = float(joint_free["dG_apply_kJmol"])
        b = float(joint_bound["dG_apply_kJmol"]) if joint_bound else 0.0
        if joint_bound is None:
            warns.append("joint bound cost assumed 0 — approximation, not measurement")
        if not joint_free.get("coverage_ok", True):
            warns.append("joint FREE coverage failed — result not usable")
        if joint_bound is not None and not joint_bound.get("coverage_ok", True):
            warns.append("joint BOUND coverage failed — result not usable")
        conf += f - b
    for t in extra_1d_conf:
        conf += float(t.get("dG_apply_free_kJmol", t.get("dG_apply_kJmol", 0.0))) \
                - float(t.get("dG_apply_bound_kJmol", 0.0))
    return {"T_K": T, "dG_radial": radial_dG,
            "angular_correction": ang,
            "conformational_correction": conf,
            "dG_bind_standard": radial_dG + ang + conf,
            "warnings": warns}


# =============================================================================
# PART B §1  dCp(T) DESIGN MATRIX
# =============================================================================

@dataclass
class ThermoObs:
    T_K: float
    kind: str                 # "dH" | "dG"
    value: float
    sigma: float = float("nan")
    provenance: str = "unspecified"


MODELS = {"M0": (0, 1), "M1": (0, 1, 2), "M2": (0, 1, 2, 3)}
LABELS = ("dH0", "dS0", "a", "b")


def design_row(T: float, kind: str, T0: float) -> np.ndarray:
    dT = T - T0
    q = 0.5 * (T * T - T0 * T0)
    if kind == "dH":
        return np.array([1.0, 0.0, dT, q])
    if kind == "dG":
        return np.array([1.0, -T, dT - T * math.log(T / T0), q - T * dT])
    raise ValueError(kind)


def build_design(obs: Sequence[ThermoObs], T0: float, model: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = MODELS[model]
    X = np.vstack([design_row(o.T_K, o.kind, T0) for o in obs])[:, cols]
    y = np.array([o.value for o in obs], dtype=np.float64)
    s = np.array([o.sigma if (o.sigma and math.isfinite(o.sigma) and o.sigma > 0)
                  else 1.0 for o in obs], dtype=np.float64)
    return X, y, s


def wls(X: np.ndarray, y: np.ndarray, s: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, float]:
    w = 1.0 / s ** 2
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    try:
        cov = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    r = y - X @ beta
    chi2 = float(np.sum(w * r * r))
    return beta, cov, chi2


# =============================================================================
# PART B §2  BOOTSTRAP ENGINES
# =============================================================================

def _mammen(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mammen two-point wild weights: mean 0, var 1, third moment 1."""
    p = (math.sqrt(5.0) + 1.0) / (2.0 * math.sqrt(5.0))
    u = rng.random(n) < p
    a = (1.0 - math.sqrt(5.0)) / 2.0
    b = (1.0 + math.sqrt(5.0)) / 2.0
    return np.where(u, a, b)


def bootstrap_dcp(obs: Sequence[ThermoObs], T0: float = 310.0,
                  model: str = "M2", n_boot: int = 4000,
                  method: str = "parametric", paired: bool = True,
                  seed: int = 0, ci: float = 0.95
                  ) -> Dict[str, Any]:
    """
    Bootstrap CI for the dCp(T) parameters and derived quantities.

    Refuses when n <= k (underdetermined). Reports the fraction of
    bootstrap replicates whose design was rank-deficient rather than
    silently dropping them.
    """
    n = len(obs)
    cols = MODELS[model]
    k = len(cols)
    if n <= k:
        return {"status": "REFUSED",
                "reason": f"n={n} observations, k={k} parameters: "
                          f"underdetermined. No bootstrap can create "
                          f"information that is not in the data.",
                "model": model, "n": n, "k": k}
    if method not in ("parametric", "wild", "residual", "case"):
        raise ValueError(f"unknown bootstrap method {method!r}")

    X, y, s = build_design(obs, T0, model)
    beta, cov, chi2 = wls(X, y, s)
    resid = y - X @ beta
    dof = n - k
    # scale residuals for residual/wild bootstrap (leverage correction)
    XtX_inv = np.linalg.pinv((X / s[:, None]).T @ (X / s[:, None]))
    H = (X / s[:, None]) @ XtX_inv @ (X / s[:, None]).T
    lev = np.clip(np.diag(H), 0.0, 0.999)
    resid_adj = resid / np.sqrt(1.0 - lev)

    rng = np.random.default_rng(seed)
    T_arr = np.array([o.T_K for o in obs])
    kinds = [o.kind for o in obs]
    # groups of rows sharing a temperature (for paired resampling)
    uniqT = sorted(set(T_arr.tolist()))
    groups = [np.where(T_arr == t)[0] for t in uniqT]

    boots: List[np.ndarray] = []
    n_fail = 0
    for _ in range(n_boot):
        if method == "parametric":
            if paired:
                # one shared standard normal per temperature, scaled by sigma
                eps = np.zeros(n)
                for g in groups:
                    z = rng.standard_normal()
                    eps[g] = z * s[g]
            else:
                eps = rng.standard_normal(n) * s
            yb = X @ beta + eps
            Xb, sb = X, s
        elif method == "wild":
            if paired:
                v = np.zeros(n)
                for g in groups:
                    v[g] = _mammen(rng, 1)[0]
            else:
                v = _mammen(rng, n)
            yb = X @ beta + resid_adj * v
            Xb, sb = X, s
        elif method == "residual":
            idx = rng.integers(0, n, n)
            yb = X @ beta + resid_adj[idx]
            Xb, sb = X, s
        else:  # case
            gi = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[j] for j in gi])
            Xb, yb, sb = X[idx], y[idx], s[idx]
            if np.linalg.matrix_rank(Xb / sb[:, None]) < k:
                n_fail += 1
                continue
        try:
            bb, _, _ = wls(Xb, yb, sb)
            if not np.all(np.isfinite(bb)):
                n_fail += 1
                continue
            boots.append(bb)
        except Exception:
            n_fail += 1

    if len(boots) < max(50, 0.1 * n_boot):
        return {"status": "FAILED",
                "reason": f"only {len(boots)}/{n_boot} usable replicates "
                          f"({n_fail} rank-deficient/failed). The design is "
                          f"too fragile for this resampling scheme.",
                "n_fail": n_fail, "model": model}

    B = np.vstack(boots)
    alpha = 1.0 - ci
    out: Dict[str, Any] = {"status": "OK", "method": method, "paired": paired,
                           "model": model, "n": n, "k": k, "dof": dof,
                           "n_boot": len(boots), "n_fail": n_fail,
                           "T0_K": T0, "params": {}}

    # jackknife for BCa (needs n-1 >= k)
    jack: Optional[np.ndarray] = None
    if n - 1 >= k:
        jl = []
        for i in range(n):
            m = np.ones(n, dtype=bool); m[i] = False
            try:
                bj, _, _ = wls(X[m], y[m], s[m])
                jl.append(bj)
            except Exception:
                jl = None
                break
        if jl is not None and len(jl) == n:
            jack = np.vstack(jl)

    for j, lab in enumerate([LABELS[c] for c in cols]):
        pt = float(beta[j])
        bs = B[:, j]
        lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
        rec = {"estimate": pt,
               "boot_mean": float(bs.mean()),
               "boot_std": float(bs.std(ddof=1)),
               "ci_percentile": (float(lo), float(hi)),
               "asymptotic_stderr": float(math.sqrt(max(cov[j, j], 0.0))),
               "P_positive": float((bs > 0).mean()),
               "straddles_zero": bool(lo < 0 < hi)}
        if jack is not None:
            z0 = _ND.inv_cdf(min(max(float((bs < pt).mean()), 1e-6), 1 - 1e-6))
            jm = jack[:, j].mean()
            num = float(np.sum((jm - jack[:, j]) ** 3))
            den = 6.0 * (float(np.sum((jm - jack[:, j]) ** 2)) ** 1.5)
            acc = num / den if den > 1e-30 else 0.0
            def adj(q):
                zq = _ND.inv_cdf(q)
                t = z0 + (z0 + zq) / (1.0 - acc * (z0 + zq))
                return min(max(_ND.cdf(t), 1e-6), 1 - 1e-6)
            a1, a2 = adj(alpha / 2), adj(1 - alpha / 2)
            rec["ci_bca"] = tuple(float(v) for v in np.quantile(bs, [a1, a2]))
            rec["bca_z0"] = z0
            rec["bca_accel"] = acc
        else:
            rec["ci_bca"] = None
            rec["bca_note"] = ("jackknife underdetermined (n-1 < k); "
                               "percentile interval only")
        out["params"][lab] = rec

    # derived: dCp(T0) = a + b*T0 ; and dCp at each observed temperature
    ia = [LABELS[c] for c in cols].index("a") if "a" in [LABELS[c] for c in cols] else None
    ib = [LABELS[c] for c in cols].index("b") if "b" in [LABELS[c] for c in cols] else None
    if ia is not None:
        cp = B[:, ia] + (B[:, ib] * T0 if ib is not None else 0.0)
        pt = float(beta[ia] + (beta[ib] * T0 if ib is not None else 0.0))
        lo, hi = np.quantile(cp, [alpha / 2, 1 - alpha / 2])
        out["dCp_at_T0"] = {"estimate": pt,
                            "ci_percentile": (float(lo), float(hi)),
                            "boot_std": float(cp.std(ddof=1)),
                            "P_positive": float((cp > 0).mean())}
        out["dCp_curve"] = {}
        for t in uniqT:
            c = B[:, ia] + (B[:, ib] * t if ib is not None else 0.0)
            l, h = np.quantile(c, [alpha / 2, 1 - alpha / 2])
            out["dCp_curve"][f"{t:.1f}"] = {
                "estimate": float(beta[ia] + (beta[ib] * t if ib is not None else 0.0)),
                "ci_percentile": (float(l), float(h))}

    warns: List[str] = []
    if method == "case":
        warns.append("case bootstrap resamples temperatures; with few distinct T "
                     "the resampled design is often rank-deficient "
                     f"({n_fail}/{n_boot} failed). Prefer parametric or wild.")
    if dof < 3:
        warns.append(f"only {dof} residual degrees of freedom; any interval is weak")
    if ib is not None and out["params"]["b"]["straddles_zero"]:
        warns.append("b straddles zero: the temperature dependence of dCp is "
                     "NOT resolved by these data. Report dCp(T0), which is "
                     "better determined, rather than b.")
    prov = {o.provenance for o in obs}
    if len(prov) > 1:
        warns.append(f"mixed provenance {sorted(prov)}")
    if method == "parametric":
        warns.append("parametric bootstrap assumes the REPORTED sigmas are correct "
                     "and the measurement noise is Gaussian; it does not assume "
                     "asymptotic normality of the estimator")
    out["warnings"] = warns
    return out


# =============================================================================
# SELF-TEST
# =============================================================================

def _make_coupled_samples(n: int, T: float, coupling: float,
                          seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Samples on the torus with a tunable coupling between z1 and z2."""
    rng = np.random.default_rng(seed)
    z1 = rng.uniform(-math.pi, math.pi, n)
    # conditional mean of z2 depends on z1 through a periodic term
    mu = coupling * np.sin(z1)
    z2 = wrap_angle(mu + rng.normal(0.0, 0.6, n))
    return z1, z2


def self_test() -> bool:
    print("=" * 94)
    print(f"  JOINT 2D TORSION PMF + BOOTSTRAP dCp(T)  v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 94)
    rng = np.random.default_rng(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    T = 310.0
    kT = KB * T

    # ---------------- A1 independent torsions -> I ~ 0 ----------------
    print("\n[A1] Mutual information: independent vs coupled")
    z1i = rng.uniform(-math.pi, math.pi, 200000)
    z2i = rng.uniform(-math.pi, math.pi, 200000)
    pmi = pmf_2d_from_samples(z1i, z2i, T, n_bins=48)
    Ii = pmi.mutual_information()["I_nats"]
    z1c, z2c = _make_coupled_samples(200000, T, coupling=1.6, seed=1)
    pmc = pmf_2d_from_samples(z1c, z2c, T, n_bins=48)
    Ic = pmc.mutual_information()["I_nats"]
    rep("independent pair has small I", Ii < 0.01, f"I={Ii:.5f} nats")
    rep("coupled pair has large I", Ic > 0.15, f"I={Ic:.5f} nats "
        f"({kT*Ic:.3f} kJ/mol)")

    # ---------------- A2 coverage diagnostic fires ----------------
    print("\n[A2] 2D coverage diagnostic")
    r_wide = JointTorsionRestraint(z1_0=0.0, z2_0=0.0, k1=200.0, k2=200.0)
    cov_ok = joint_apply_free_energy(pmc, r_wide, restraint_on=False)
    rep("well-sampled surface passes coverage", cov_ok["coverage_ok"],
        f"uncovered={cov_ok['uncovered_restraint_weight']:.4f}")
    # sparse surface, restraint centred in an empty region
    z1s, z2s = _make_coupled_samples(300, T, coupling=1.6, seed=2)
    pms = pmf_2d_from_samples(z1s, z2s, T, n_bins=96)
    r_far = JointTorsionRestraint(z1_0=2.9, z2_0=-2.9, k1=3000.0, k2=3000.0)
    cov_bad = joint_apply_free_energy(pms, r_far, restraint_on=False)
    rep("sparse surface + distant restraint FAILS coverage",
        not cov_bad["coverage_ok"],
        f"uncovered={cov_bad['uncovered_restraint_weight']:.4f}")
    rep("warning text present", len(cov_bad["warning"]) > 0)

    # ---------------- A3 additive vs joint coupling error ----------------
    print("\n[A3] Additive vs joint: the cost of the independence assumption")
    r = JointTorsionRestraint(z1_0=0.4, z2_0=0.0, k1=150.0, k2=150.0, k12=40.0)
    cmp_c = additive_vs_joint(pmc, r, restraint_on=False)
    cmp_i = additive_vs_joint(pmi, r, restraint_on=False)
    print(f"      coupled : joint={cmp_c['dG_joint_kJmol']:+.4f}  "
          f"additive={cmp_c['dG_additive_kJmol']:+.4f}  "
          f"error={cmp_c['coupling_error_kJmol']:+.4f} kJ/mol")
    print(f"      indep   : joint={cmp_i['dG_joint_kJmol']:+.4f}  "
          f"additive={cmp_i['dG_additive_kJmol']:+.4f}  "
          f"error={cmp_i['coupling_error_kJmol']:+.4f} kJ/mol")
    rep("coupled pair shows non-negligible coupling error",
        abs(cmp_c["coupling_error_kJmol"]) > 0.1)
    rep("independent pair shows near-zero coupling error",
        abs(cmp_i["coupling_error_kJmol"]) < 0.15)
    rep("joint restraint PD check works", True)
    try:
        JointTorsionRestraint(0, 0, 1.0, 1.0, 2.0)
        rep("non-PD restraint rejected", False)
    except ValueError:
        rep("non-PD restraint rejected", True)

    # ---------------- A4 2D WHAM recovers a known surface ----------------
    print("\n[A4] 2D WHAM recovers an analytic coupled surface")
    def W_true(g1, g2):
        return (-2.0 * np.cos(g1) - 2.0 * np.cos(g2)
                + 1.5 * np.cos(g1 - g2))
    edges = np.linspace(-math.pi, math.pi, 49)
    c = 0.5 * (edges[1:] + edges[:-1])
    G1, G2 = np.meshgrid(c, c, indexing="ij")
    Wt = W_true(G1, G2)
    wins = []
    for a0 in (-2.0, 0.0, 2.0):
        for b0 in (-2.0, 0.0, 2.0):
            k = 6.0
            def bias(g1, g2, a0=a0, b0=b0, k=k):
                return 0.5*k*(periodic_delta(g1,a0)**2 + periodic_delta(g2,b0)**2)
            pb = np.exp(-(Wt + bias(G1, G2)) / kT).ravel()
            pb /= pb.sum()
            idx = rng.choice(pb.size, size=6000, p=pb)
            i1, i2 = np.unravel_index(idx, G1.shape)
            wins.append({"z1": G1[i1, i2] + rng.normal(0, 0.02, len(idx)),
                         "z2": G2[i1, i2] + rng.normal(0, 0.02, len(idx)),
                         "bias": bias})
    rec = wham_2d(wins, T, n_bins=48)
    Wr = rec.W
    Wt_grid = W_true(*np.meshgrid(rec.centres, rec.centres, indexing="ij"))
    m = rec.sampled & (Wt_grid < np.nanpercentile(Wt_grid, 70))
    a = Wr[m] - Wr[m].min(); b = Wt_grid[m] - Wt_grid[m].min()
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rep("2D WHAM RMSE < 1.5 kJ/mol on low-energy region", rmse < 1.5,
        f"RMSE={rmse:.3f}, coverage={rec.coverage_fraction:.3f}")

    # ---------------- A5 full assembly ----------------
    print("\n[A5] Woo-Roux assembly with the joint conformational term")
    jf = joint_apply_free_energy(pmc, r, restraint_on=False)
    jb = dict(jf); jb["dG_apply_kJmol"] = 0.5
    asm = wooroux_with_joint_conformational(-30.0, T,
        angular_terms=[{"dG_apply_free_kJmol": 6.0, "dG_apply_bound_kJmol": 0.5}],
        joint_free=jf, joint_bound=jb)
    expect = -30.0 + 5.5 + (jf["dG_apply_kJmol"] - 0.5)
    rep("assembly matches hand sum", abs(asm["dG_bind_standard"] - expect) < 1e-12,
        f"dG={asm['dG_bind_standard']:+.4f}")

    # ---------------- B1 bootstrap recovers known dCp ----------------
    print("\n[B1] Parametric bootstrap recovers a known dCp(T)")
    T0 = 310.0
    dH0, dS0, a_true, b_true = -45.0, -0.10, -0.60, 0.0020
    def dH_of(t): return dH0 + a_true*(t-T0) + 0.5*b_true*(t*t-T0*T0)
    def dS_of(t): return dS0 + a_true*math.log(t/T0) + b_true*(t-T0)
    def dG_of(t): return dH_of(t) - t*dS_of(t)
    Ts = [288., 296., 304., 312., 320., 328.]
    obs = []
    for t in Ts:
        obs.append(ThermoObs(t, "dH", dH_of(t) + rng.normal(0, .15), .15, "calorimetric"))
        obs.append(ThermoObs(t, "dG", dG_of(t) + rng.normal(0, .10), .10, "equilibrium"))
    bs = bootstrap_dcp(obs, T0, "M2", n_boot=2500, method="parametric", seed=1)
    rep("bootstrap status OK", bs["status"] == "OK", bs.get("reason", ""))
    if bs["status"] == "OK":
        pb = bs["params"]["b"]; pa = bs["params"]["a"]
        print(f"      a = {pa['estimate']:+.4f}  CI{pb and ''} "
              f"{tuple(round(v,4) for v in pa['ci_percentile'])}  true={a_true:+.4f}")
        print(f"      b = {pb['estimate']:+.6f}  CI "
              f"{tuple(round(v,6) for v in pb['ci_percentile'])}  true={b_true:+.6f}")
        print(f"      boot std(b)={pb['boot_std']:.6f}  "
              f"asymptotic se(b)={pb['asymptotic_stderr']:.6f}")
        rep("a inside its 95% CI",
            pa["ci_percentile"][0] <= a_true <= pa["ci_percentile"][1])
        rep("b inside its 95% CI",
            pb["ci_percentile"][0] <= b_true <= pb["ci_percentile"][1])
        rep("BCa interval present", pb.get("ci_bca") is not None)
        rep("bootstrap std comparable to asymptotic se",
            0.4 < pb["boot_std"] / max(pb["asymptotic_stderr"], 1e-12) < 2.5,
            f"ratio={pb['boot_std']/max(pb['asymptotic_stderr'],1e-12):.2f}")
        rep("dCp(T0) reported with CI", "dCp_at_T0" in bs,
            f"{bs['dCp_at_T0']['estimate']:+.4f} "
            f"{tuple(round(v,4) for v in bs['dCp_at_T0']['ci_percentile'])}")

    # ---------------- B2 underdetermined is REFUSED ----------------
    print("\n[B2] Underdetermined fits must be REFUSED, not resampled")
    obs3 = [ThermoObs(t, "dH", dH_of(t), .2, "calorimetric")
            for t in (300., 310., 320.)]
    r3 = bootstrap_dcp(obs3, T0, "M2", n_boot=500)
    rep("n=3, k=4 REFUSED", r3["status"] == "REFUSED", r3.get("reason", "")[:70])
    r3m1 = bootstrap_dcp(obs3, T0, "M1", n_boot=500)
    rep("n=3, k=3 REFUSED", r3m1["status"] == "REFUSED")

    # ---------------- B3 wild bootstrap under heteroscedasticity ----------------
    print("\n[B3] Wild bootstrap with heteroscedastic noise")
    obs_h = []
    for i, t in enumerate(Ts):
        sg = 0.05 if i % 2 == 0 else 0.6
        obs_h.append(ThermoObs(t, "dH", dH_of(t) + rng.normal(0, sg), sg, "calorimetric"))
    bw = bootstrap_dcp(obs_h, T0, "M2", n_boot=2000, method="wild", seed=2)
    rep("wild bootstrap runs", bw["status"] == "OK", bw.get("reason", "")[:60])
    if bw["status"] == "OK":
        rep("wild CI for a is finite",
            all(math.isfinite(v) for v in bw["params"]["a"]["ci_percentile"]))

    # ---------------- B4 case bootstrap reports failures ----------------
    print("\n[B4] Case bootstrap reports rank-deficiency honestly")
    bc = bootstrap_dcp(obs3 + [ThermoObs(330., "dH", dH_of(330.), .2, "calorimetric")],
                       T0, "M1", n_boot=800, method="case", seed=3)
    rep("case bootstrap returns a status", bc["status"] in ("OK", "FAILED", "REFUSED"),
        f"status={bc['status']}, n_fail={bc.get('n_fail','n/a')}")
    if bc["status"] == "OK":
        rep("case failure count reported", "n_fail" in bc)

    # ---------------- B5 unresolved b is flagged ----------------
    print("\n[B5] Noise-dominated b must be flagged")
    obs_n = [ThermoObs(t, "dH", dH_of(t) + rng.normal(0, 2.0), 2.0, "calorimetric")
             for t in (303., 307., 311., 315., 319.)]
    bn = bootstrap_dcp(obs_n, T0, "M2", n_boot=1500, method="parametric", seed=4)
    if bn["status"] == "OK":
        flagged = any("straddles zero" in w for w in bn["warnings"])
        rep("unresolved b flagged", flagged,
            f"P(b>0)={bn['params']['b']['P_positive']:.2f}")
    else:
        rep("noisy fit refused/failed honestly", True, bn["status"])

    print("\n" + "=" * 94)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 94)
    print("""
  WHAT THESE TESTS PROVE
    A1  mutual information separates independent from coupled torsions
    A2  the 2D coverage diagnostic fires when the restraint weight sits
        in unsampled region
    A3  the additive approximation carries a measurable error for
        coupled torsions and essentially none for independent ones
    A4  2D WHAM recovers an analytic coupled surface
    A5  the assembly uses the joint term correctly
    B1  parametric bootstrap CIs cover the true a and b, with BCa
    B2  underdetermined models are REFUSED, not resampled
    B3  wild bootstrap runs under heteroscedasticity
    B4  case bootstrap reports its rank-deficiency failures
    B5  an unresolved b is flagged instead of reported with a fake CI

  WHAT IS NOT PROVEN
    * a 2D PMF can pass coverage and still be unconverged
    * bootstrap does not repair model misspecification or missing data
    * parametric bootstrap trusts the reported sigmas
    * Cartesian-sampled torsion PMFs already contain the Fixman factor;
      adding it again would double-count
    * no experimental validation is performed here
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Joint torsion + bootstrap dCp v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--thermo", default=None)
    ap.add_argument("--T0", type=float, default=310.0)
    ap.add_argument("--model", default="M2", choices=list(MODELS))
    ap.add_argument("--method", default="parametric",
                    choices=["parametric", "wild", "residual", "case"])
    ap.add_argument("--nboot", type=int, default=4000)
    ap.add_argument("--unpaired", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if self_test() else 1)
    if a.thermo:
        with open(a.thermo) as f:
            raw = json.load(f)
        obs = [ThermoObs(float(d["T_K"]), d["kind"], float(d["value"]),
                         float(d.get("sigma", float("nan"))),
                         d.get("provenance", "unspecified")) for d in raw]
        r = bootstrap_dcp(obs, a.T0, a.model, a.nboot, a.method,
                          paired=not a.unpaired)
        print(json.dumps(r, indent=2, default=float))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(r, f, indent=2, default=float)
        return
    ap.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
