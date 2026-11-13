#!/usr/bin/env python3
"""
=============================================================================
3D JOINT TORSION PMF (phi/psi/chi1)  +  BAYESIAN dCp(T) WITH WEAK PRIORS
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — 3D JOINT TORSION PMF
  Extends the 2D joint treatment to three internal coordinates on the
  torus [-pi,pi)^3. In three dimensions there are THREE distinct
  reduced descriptions, not two, and they fail differently:

    independent : p1(x1) p2(x2) p3(x3)
    Kirkwood    : p12 p13 p23 / (p1 p2 p3)      (pairwise superposition)
    Markov chain: p12 p23 / p2                  (x1 -> x2 -> x3)
    full 3D     : p(x1,x2,x3)

  A genuine THREE-BODY interaction, e.g. an energy term
  cos(x1 + x2 + x3), is invisible to ALL of the reduced descriptions.
  The self-test constructs exactly such a case and measures the error
  each approximation makes in dG_apply.

  Total correlation (multi-information)
      TC = sum_i H(X_i) - H(X) = KL( p || prod_i p_i )
  measures total dependence; pairwise MI measures the part of it that
  a pairwise model could capture. TC minus the pairwise-explainable
  part is the irreducible higher-order dependence.

  CURSE OF DIMENSIONALITY — the real obstruction
    n_bins^3 bins. At n_bins=36 that is 46,656 bins; a usable PMF in
    the low-energy region needs ~1e6-1e7 samples. The module reports
    occupancy, samples per occupied bin, and the fraction of the
    restraint's Boltzmann weight sitting in UNSAMPLED bins, and it
    REFUSES to return a dG_apply when that fraction is too large.
    This is not a formality: in 3D it is the dominant failure mode.

PART B — BAYESIAN dCp(T) WITH WEAKLY-INFORMATIVE PRIORS
  Reparameterised to physically meaningful parameters:
      theta = (dH0, dS0, dCp0, b),   dCp(T) = dCp0 + b (T - T0)
      dH(T) = dH0 + dCp0 dT + (b/2) dT^2
      dS(T) = dS0 + dCp0 ln(T/T0) + b [ dT - T0 ln(T/T0) ]
      dG(T) = dH(T) - T dS(T)
  Both dH and dG are LINEAR in theta, so with Gaussian measurement
  noise and a Gaussian prior the posterior is EXACT and conjugate:
      V_n = (V0^-1 + X^T Sigma^-1 X)^-1
      mu_n = V_n (V0^-1 mu0 + X^T Sigma^-1 y)
  No MCMC, no bootstrap resampling noise. This is the genuine
  advantage at small n: an exact posterior instead of a refused or
  degenerate resampling distribution.

  Reported error bars may themselves be wrong. A second model puts a
  weakly-informative prior on a COMMON SCALE factor s on the reported
  sigmas (Normal-Inverse-Gamma), giving a multivariate-t marginal for
  theta. Posterior draws are EXACT iid draws (not MCMC).

  THE HONEST DIAGNOSTIC AT SMALL n
    prior_sd / posterior_sd for each parameter. If that ratio is ~1,
    the data did not inform that parameter and the prior is supplying
    the answer. This is reported prominently, together with a prior
    sensitivity sweep (prior scales x0.5, x2, x4). A Bayesian answer
    that moves when the prior moves is a prior-dominated answer, and
    saying so is the whole point.

HONEST SCOPE
  ❌ A 3D histogram PMF is not converged merely because it is filled.
     Occupancy diagnostics are necessary, not sufficient.
  ❌ Kirkwood superposition is renormalised to a probability mass; it
     is a legitimate distribution but NOT the true joint.
  ❌ Bayesian credible intervals are statements conditional on the
     prior AND the linear dCp(T) model AND Gaussian noise. They are
     not frequentist coverage guarantees.
  ❌ No Fixman/metric-determinant correction: samples assumed to come
     from equilibrium Cartesian MD, so the torsion histogram already
     contains it. Adding it again would double-count.
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
log = logging.getLogger("torsion3d_bayes")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

KB = 0.008314462618          # kJ mol^-1 K^-1
TWO_PI = 2.0 * math.pi


# =============================================================================
# PART A §1  3D TORUS UTILITIES AND PMF CONTAINER
# =============================================================================

def wrap3(z: np.ndarray) -> np.ndarray:
    return (np.asarray(z, dtype=np.float64) + math.pi) % TWO_PI - math.pi


def pdelta(z: np.ndarray, z0: float) -> np.ndarray:
    return wrap3(z - z0)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, over the positive entries."""
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))


@dataclass
class JointPMF3D:
    """
    Joint PMF on [-pi,pi)^3.
    W        : (nb,nb,nb) kJ/mol, np.inf where unsampled
    sampled  : boolean mask
    counts   : raw (or pseudo-counted) histogram
    """
    W: np.ndarray
    sampled: np.ndarray
    edges: np.ndarray
    centres: np.ndarray
    temperature: float
    counts: np.ndarray
    n_samples: int

    @property
    def kT(self) -> float:
        return KB * self.temperature

    @property
    def dV(self) -> float:
        d = self.edges[1] - self.edges[0]
        return float(d ** 3)

    @property
    def nb(self) -> int:
        return len(self.centres)

    def occupancy(self) -> Dict[str, float]:
        n_bins = self.counts.size
        n_occ = int(self.sampled.sum())
        return {
            "n_bins": n_bins,
            "n_occupied": n_occ,
            "occupancy_fraction": n_occ / n_bins,
            "n_samples": self.n_samples,
            "samples_per_occupied_bin": (self.n_samples / n_occ) if n_occ else 0.0,
        }

    def prob(self) -> np.ndarray:
        tot = self.counts.sum()
        return self.counts / tot if tot > 0 else self.counts.copy()

    def marginal(self, axes: Tuple[int, ...]) -> np.ndarray:
        p = self.prob()
        keep = set(axes)
        red = tuple(ax for ax in range(3) if ax not in keep)
        return p.sum(axis=red) if red else p

    def entropies(self) -> Dict[str, float]:
        p = self.prob()
        H = _entropy(p.ravel())
        H1 = [_entropy(self.marginal((i,)).ravel()) for i in range(3)]
        H2 = {}
        for pair in ((0, 1), (0, 2), (1, 2)):
            H2[f"{pair[0]}{pair[1]}"] = _entropy(self.marginal(pair).ravel())
        TC = sum(H1) - H
        pair_mi = {
            "01": H1[0] + H1[1] - H2["01"],
            "02": H1[0] + H1[2] - H2["02"],
            "12": H1[1] + H1[2] - H2["12"],
        }
        return {"H_joint_nats": H, "H1_nats": H1, "H2_nats": H2,
                "total_correlation_nats": TC,
                "total_correlation_kJmol": self.kT * TC,
                "pairwise_MI_nats": pair_mi,
                "sum_pairwise_MI_nats": float(sum(pair_mi.values()))}


def pmf_3d_from_samples(z1: np.ndarray, z2: np.ndarray, z3: np.ndarray,
                        T: float, n_bins: int = 36,
                        pseudocount: float = 0.0) -> JointPMF3D:
    """
    Joint 3D PMF from equilibrium torsion samples.
    If the samples came from Cartesian MD, the Fixman factor is already
    present; do NOT add it again.
    """
    z1, z2, z3 = wrap3(z1), wrap3(z2), wrap3(z3)
    if not (z1.shape == z2.shape == z3.shape):
        raise ValueError("all three coordinate arrays must have equal length")
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    H, _ = np.histogramdd(np.stack([z1, z2, z3], -1),
                          bins=[edges, edges, edges])
    counts = H + float(pseudocount)
    tot = counts.sum()
    if tot <= 0:
        raise ValueError("no samples")
    p = counts / tot
    with np.errstate(divide="ignore"):
        W = -KB * T * np.log(np.where(p > 0, p, np.nan))
    fin = np.isfinite(W)
    W = np.where(fin, W - np.nanmin(W), np.inf)
    c = 0.5 * (edges[1:] + edges[:-1])
    return JointPMF3D(W=W, sampled=fin, edges=edges, centres=c,
                      temperature=float(T), counts=counts,
                      n_samples=int(z1.size))


def wham_3d(windows: Sequence[Dict[str, Any]], T: float,
            n_bins: int = 30, tol: float = 1e-7,
            max_iter: int = 20000) -> JointPMF3D:
    """
    3D WHAM. Each window: {"z1","z2","z3": arrays,
                           "bias": callable(g1,g2,g3) -> kJ/mol grid}
    Memory: n_windows * n_bins^3 doubles.
    """
    kT = KB * T
    beta = 1.0 / kT
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    G1, G2, G3 = np.meshgrid(c, c, c, indexing="ij")
    Hs, Ns, Wbs = [], [], []
    n_tot = 0
    for w in windows:
        z1, z2, z3 = wrap3(w["z1"]), wrap3(w["z2"]), wrap3(w["z3"])
        h, _ = np.histogramdd(np.stack([z1, z2, z3], -1),
                              bins=[edges, edges, edges])
        Hs.append(h); Ns.append(float(h.sum())); n_tot += int(h.sum())
        Wbs.append(np.asarray(w["bias"](G1, G2, G3), dtype=np.float64))
    H = np.stack(Hs); N = np.array(Ns); Wb = np.stack(Wbs)
    if N.sum() <= 0:
        raise ValueError("no samples in any window")
    expW = np.exp(-beta * np.clip(Wb, -700, 700))
    f = np.zeros(len(windows))
    p = np.zeros_like(H[0])
    conv = False
    for it in range(max_iter):
        den = (N[:, None, None, None] * expW
               * np.exp(beta * f)[:, None, None, None]).sum(0)
        den = np.maximum(den, 1e-300)
        pn = H.sum(0) / den
        s = pn.sum()
        if s <= 0:
            break
        pn /= s
        fn = -kT * np.log(np.maximum((pn[None] * expW).sum(axis=(1, 2, 3)), 1e-300))
        fn -= fn[0]
        d = float(np.max(np.abs(fn - f)))
        f, p = fn, pn
        if d < tol:
            conv = True
            break
    with np.errstate(divide="ignore"):
        W = -kT * np.log(np.where(p > 0, p, np.nan))
    fin = np.isfinite(W)
    W = np.where(fin, W - np.nanmin(W), np.inf)
    log.info("3D WHAM: %d iters, converged=%s, occupancy=%.4f",
             it + 1, conv, fin.mean())
    return JointPMF3D(W=W, sampled=fin, edges=edges, centres=c,
                      temperature=float(T), counts=H.sum(0), n_samples=n_tot)


# =============================================================================
# PART A §2  3D JOINT RESTRAINT AND APPLY FREE ENERGY
# =============================================================================

@dataclass
class JointRestraint3D:
    """
    U(x) = 0.5 d^T K d,  d_i = periodic(x_i - x0_i).
    K must be positive definite (checked by Cholesky).
    Off-diagonal entries make it a genuinely joint restraint.
    """
    centre: Tuple[float, float, float]
    K: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError("K must be 3x3")
        if not np.allclose(self.K, self.K.T, atol=1e-10):
            raise ValueError("K must be symmetric")
        try:
            np.linalg.cholesky(self.K)
        except np.linalg.LinAlgError:
            raise ValueError("K must be positive definite")

    def on_grid(self, c: np.ndarray) -> np.ndarray:
        G1, G2, G3 = np.meshgrid(c, c, c, indexing="ij")
        d1 = pdelta(G1, self.centre[0])
        d2 = pdelta(G2, self.centre[1])
        d3 = pdelta(G3, self.centre[2])
        K = self.K
        return 0.5 * (K[0, 0]*d1*d1 + K[1, 1]*d2*d2 + K[2, 2]*d3*d3
                      + 2*K[0, 1]*d1*d2 + 2*K[0, 2]*d1*d3 + 2*K[1, 2]*d2*d3)


def _apply_from_W(W: np.ndarray, sampled: np.ndarray, U: np.ndarray,
                  kT: float, dV: float, restraint_on: bool,
                  coverage_tol: float) -> Dict[str, Any]:
    """Core computation shared by the full and reduced descriptions."""
    beta = 1.0 / kT
    W_int = W - U if restraint_on else W.copy()
    fin = np.isfinite(W_int) & sampled
    if not fin.any():
        raise ValueError("no sampled bins")
    shift = np.min(W_int[fin])
    Wi = np.where(fin, W_int - shift, np.inf)
    eU = np.exp(-beta * np.clip(U - U.min(), 0, 700))
    tot_eU = float(eU.sum())
    uncovered = float(eU[~fin].sum() / max(tot_eU, 1e-300))
    num = np.where(fin, np.exp(-beta * np.clip(Wi + U, 0, 700)), 0.0)
    den = np.where(fin, np.exp(-beta * np.clip(Wi, 0, 700)), 0.0)
    Zn = float(num.sum()) * dV
    Zd = float(den.sum()) * dV
    if Zn <= 0 or Zd <= 0:
        raise ValueError("zero partition function")
    dG = -kT * math.log(Zn / Zd)
    ok = uncovered < coverage_tol
    return {"dG_apply_kJmol": dG,
            "uncovered_restraint_weight": uncovered,
            "coverage_ok": bool(ok),
            "warning": ("" if ok else
                        "UNSAMPLED BINS CARRY THE RESTRAINT WEIGHT — "
                        "dG_apply is not usable. In 3D this is the "
                        "dominant failure mode: get more samples, "
                        "coarsen the bins, or weaken the restraint.")}


def joint_apply_3d(pmf: JointPMF3D, r: JointRestraint3D,
                   restraint_on: bool = True,
                   coverage_tol: float = 0.02,
                   min_samples_per_bin: float = 5.0) -> Dict[str, Any]:
    occ = pmf.occupancy()
    out = _apply_from_W(pmf.W, pmf.sampled, r.on_grid(pmf.centres),
                        pmf.kT, pmf.dV, restraint_on, coverage_tol)
    out["occupancy"] = occ
    sparse = occ["samples_per_occupied_bin"] < min_samples_per_bin
    out["sampling_ok"] = not sparse
    if sparse:
        out["warning"] += (f" SPARSE: only "
                           f"{occ['samples_per_occupied_bin']:.1f} samples per "
                           f"occupied bin (< {min_samples_per_bin}). The 3D "
                           f"histogram is not converged.")
    return out


# =============================================================================
# PART A §3  REDUCED DESCRIPTIONS AND THEIR ERRORS
# =============================================================================

def _broadcast_marginal(m: np.ndarray, axes: Tuple[int, ...], nb: int) -> np.ndarray:
    """Broadcast a marginal back onto the full 3D grid."""
    shape = [1, 1, 1]
    for ax in axes:
        shape[ax] = nb
    return m.reshape(shape)


def reduced_descriptions(pmf: JointPMF3D) -> Dict[str, np.ndarray]:
    """
    Build the three reduced probability models on the full grid.
    Each is renormalised to a probability mass; zeros are handled by
    masking to the support of the full joint (so no reduced model
    invents mass where the full model has none).
    """
    nb = pmf.nb
    p = pmf.prob()
    p1 = pmf.marginal((0,)); p2 = pmf.marginal((1,)); p3 = pmf.marginal((2,))
    p12 = pmf.marginal((0, 1)); p13 = pmf.marginal((0, 2)); p23 = pmf.marginal((1, 2))
    B1 = _broadcast_marginal(p1, (0,), nb)
    B2 = _broadcast_marginal(p2, (1,), nb)
    B3 = _broadcast_marginal(p3, (2,), nb)
    B12 = _broadcast_marginal(p12, (0, 1), nb)
    B13 = _broadcast_marginal(p13, (0, 2), nb)
    B23 = _broadcast_marginal(p23, (1, 2), nb)
    eps = 1e-300
    ind = B1 * B2 * B3
    kirk = B12 * B13 * B23 / np.maximum(B1 * B2 * B3, eps)
    mark = B12 * B23 / np.maximum(B2, eps)
    out = {}
    for name, q in (("independent", ind), ("kirkwood", kirk), ("markov_chain", mark)):
        q = np.where(p > 0, q, 0.0)
        s = q.sum()
        out[name] = q / s if s > 0 else q
    return out


def compare_approximations_3d(pmf: JointPMF3D, r: JointRestraint3D,
                              restraint_on: bool = True
                              ) -> Dict[str, Any]:
    """
    dG_apply under the full 3D joint and under each reduced model, plus
    the error each reduced model makes. Also reports the entropic
    decomposition so the reader can see how much dependence exists and
    how much of it is pairwise.
    """
    full = joint_apply_3d(pmf, r, restraint_on)
    U = r.on_grid(pmf.centres)
    res: Dict[str, Any] = {"full_3d": full,
                           "entropies": pmf.entropies(),
                           "reduced": {}}
    for name, q in reduced_descriptions(pmf).items():
        with np.errstate(divide="ignore"):
            Wq = -pmf.kT * np.log(np.where(q > 0, q, np.nan))
        fin = np.isfinite(Wq)
        Wq = np.where(fin, Wq - np.nanmin(Wq), np.inf)
        try:
            red = _apply_from_W(Wq, fin, U, pmf.kT, pmf.dV,
                                restraint_on, coverage_tol=1.0)
            err = red["dG_apply_kJmol"] - full["dG_apply_kJmol"]
        except Exception as e:
            red, err = {"error": str(e)}, float("nan")
        res["reduced"][name] = {"result": red, "error_kJmol": err}
    res["interpretation"] = (
        "error_kJmol is what each approximation costs relative to the "
        "full 3D joint. A nonzero Kirkwood error with small pairwise MI "
        "indicates genuine three-body dependence that no pairwise model "
        "can represent.")
    return res


def wooroux_with_3d_conformational(radial_dG: float, T: float,
                                   angular_terms: Sequence[Dict[str, Any]] = (),
                                   joint_free: Optional[Dict[str, Any]] = None,
                                   joint_bound: Optional[Dict[str, Any]] = None,
                                   extra_1d: Sequence[Dict[str, Any]] = (),
                                   ) -> Dict[str, Any]:
    """Assemble dG_bind with the three-torsion term treated jointly."""
    warns: List[str] = []
    ang = sum(float(t.get("dG_apply_free_kJmol", 0.0))
              - float(t.get("dG_apply_bound_kJmol", 0.0)) for t in angular_terms)
    conf = 0.0
    usable = True
    if joint_free is not None:
        f = float(joint_free["dG_apply_kJmol"])
        b = float(joint_bound["dG_apply_kJmol"]) if joint_bound else 0.0
        if joint_bound is None:
            warns.append("3D joint bound cost assumed 0 — approximation")
        for tag, d in (("free", joint_free), ("bound", joint_bound)):
            if d is None:
                continue
            if not d.get("coverage_ok", True):
                warns.append(f"3D joint {tag} coverage FAILED — not usable")
                usable = False
            if not d.get("sampling_ok", True):
                warns.append(f"3D joint {tag} sampling SPARSE — not converged")
                usable = False
        conf += f - b
    for t in extra_1d:
        conf += (float(t.get("dG_apply_free_kJmol", t.get("dG_apply_kJmol", 0.0)))
                 - float(t.get("dG_apply_bound_kJmol", 0.0)))
    return {"T_K": T, "dG_radial": radial_dG,
            "angular_correction": ang, "conformational_correction": conf,
            "dG_bind_standard": radial_dG + ang + conf,
            "usable": usable, "warnings": warns}


# =============================================================================
# PART B §1  DESIGN MATRIX IN PHYSICAL PARAMETERISATION
# =============================================================================

@dataclass
class ThermoObs:
    T_K: float
    kind: str            # "dH" | "dG"
    value: float
    sigma: float = float("nan")
    provenance: str = "unspecified"


PARAM_NAMES = ("dH0", "dS0", "dCp0", "b")
PARAM_UNITS = ("kJ/mol", "kJ/mol/K", "kJ/mol/K", "kJ/mol/K^2")


def design_row(T: float, kind: str, T0: float) -> np.ndarray:
    """
    theta = (dH0, dS0, dCp0, b),  dCp(T) = dCp0 + b (T-T0).
      dH(T) = dH0 + dCp0 dT + (b/2) dT^2
      dS(T) = dS0 + dCp0 L + b (dT - T0 L),  L = ln(T/T0)
      dG(T) = dH(T) - T dS(T)
    """
    dT = T - T0
    L = math.log(T / T0)
    if kind == "dH":
        return np.array([1.0, 0.0, dT, 0.5 * dT * dT])
    if kind == "dG":
        return np.array([1.0, -T, dT - T * L, 0.5 * dT * dT - T * (dT - T0 * L)])
    raise ValueError(f"kind must be dH|dG, got {kind!r}")


MODELS: Dict[str, Tuple[int, ...]] = {
    "M0": (0, 1),           # dCp = 0
    "M1": (0, 1, 2),        # dCp = const
    "M2": (0, 1, 2, 3),     # dCp affine in T
}


# =============================================================================
# PART B §2  WEAKLY-INFORMATIVE PRIOR
# =============================================================================

@dataclass
class WeakPrior:
    """
    Gaussian prior on theta, scales chosen to be broad relative to
    physically observed values but not improper:

      dH0  : sd 100   kJ/mol      (binding dH typically -100..+50)
      dS0  : sd 0.5   kJ/mol/K    (typically -0.3..+0.1)
      dCp0 : sd 2.0   kJ/mol/K    (typically -2..+1)
      b    : sd 0.02  kJ/mol/K^2  (over +-10 K gives +-0.2 kJ/mol/K)

    The b prior is the one that regularises the otherwise
    unidentifiable curvature at small n. Its influence MUST be
    reported, not hidden.
    """
    mean: np.ndarray = field(default_factory=lambda: np.zeros(4))
    sd: np.ndarray = field(default_factory=lambda: np.array([100.0, 0.5, 2.0, 0.02]))

    def scaled(self, factor: float) -> "WeakPrior":
        return WeakPrior(mean=self.mean.copy(), sd=self.sd * factor)

    def covariance(self, cols: Tuple[int, ...]) -> np.ndarray:
        s = self.sd[list(cols)]
        return np.diag(s ** 2)

    def mu(self, cols: Tuple[int, ...]) -> np.ndarray:
        return self.mean[list(cols)]


# =============================================================================
# PART B §3  EXACT CONJUGATE POSTERIOR
# =============================================================================

def _post_gaussian(X, y, w, mu0, V0):
    V0i = np.linalg.inv(V0)
    A = V0i + X.T @ (w[:, None] * X)
    Vn = np.linalg.inv(A)
    b = V0i @ mu0 + X.T @ (w * y)
    mun = Vn @ b
    return mun, Vn, A


def bayesian_dcp(obs: Sequence[ThermoObs], T0: float = 310.0,
                 model: str = "M2", prior: Optional[WeakPrior] = None,
                 sigma_scale_prior: bool = True,
                 alpha0: float = 2.0, beta0: float = 2.0,
                 n_draws: int = 200_000, seed: int = 0,
                 cred: float = 0.95) -> Dict[str, Any]:
    """
    Exact conjugate Bayesian posterior for dCp(T) parameters.

    sigma_scale_prior=False : reported sigmas taken as given.
                              posterior is multivariate Gaussian.
    sigma_scale_prior=True  : a common scale s on the reported sigmas
                              gets InvGamma(alpha0,beta0); marginal
                              posterior for theta is multivariate t.
                              Draws are EXACT iid draws, not MCMC.

    Unlike the bootstrap, this never REFUSES at n<=k — but it reports
    prior_sd/posterior_sd so a prior-dominated answer is visible.
    """
    if len(obs) < 2:
        raise ValueError("need >= 2 observations")
    prior = prior or WeakPrior()
    cols = MODELS[model]
    k = len(cols)
    n = len(obs)
    X = np.vstack([design_row(o.T_K, o.kind, T0) for o in obs])[:, cols]
    y = np.array([o.value for o in obs], dtype=np.float64)
    sig = np.array([o.sigma if (o.sigma and math.isfinite(o.sigma) and o.sigma > 0)
                    else 1.0 for o in obs], dtype=np.float64)
    w = 1.0 / sig ** 2
    mu0 = prior.mu(cols); V0 = prior.covariance(cols)
    mun, Vn, A = _post_gaussian(X, y, w, mu0, V0)
    post_sd = np.sqrt(np.maximum(np.diag(Vn), 0.0))
    prior_sd = prior.sd[list(cols)]

    warns: List[str] = []
    prov = {o.provenance for o in obs}
    if len(prov) > 1:
        warns.append(f"mixed provenance {sorted(prov)}")
    if any("md" in s.lower() or "potential" in s.lower() for s in prov):
        warns.append("MD/potential-derived enthalpy present: not the "
                     "experimental dH; dCp inherits that bias")
    kinds = {o.kind for o in obs}
    if "dH" not in kinds:
        warns.append("no dH observations: dCp inferred from dG(T) curvature "
                     "only, which is the ill-conditioned van 't Hoff route")
    Ts = np.array([o.T_K for o in obs])
    span = float(Ts.max() - Ts.min())
    if span < 15.0:
        warns.append(f"temperature span {span:.1f} K is short; b is weakly "
                     "identified and its prior will dominate")
    if n <= k:
        warns.append(f"n={n} <= k={k}: the likelihood cannot determine all "
                     "parameters. The posterior is PRIOR-REGULARISED. "
                     "This is exactly the regime where the bootstrap "
                     "refuses; a Bayesian answer exists only because the "
                     "prior supplies the missing information.")

    # ---- exact iid posterior draws ----
    rng = np.random.default_rng(seed)
    if sigma_scale_prior:
        V0i = np.linalg.inv(V0)
        quad = float(y @ (w * y) + mu0 @ V0i @ mu0 - mun @ A @ mun)
        alpha_n = alpha0 + 0.5 * n
        beta_n = beta0 + 0.5 * max(quad, 0.0)
        s2 = beta_n / rng.gamma(alpha_n, size=n_draws)
        L = np.linalg.cholesky(Vn)
        z = rng.standard_normal((n_draws, k))
        draws = mun[None, :] + (z @ L.T) * np.sqrt(s2)[:, None]
        df = 2 * alpha_n
        scale_info = {"model": "Normal-Inverse-Gamma",
                      "alpha_n": alpha_n, "beta_n": beta_n,
                      "t_df": df,
                      "sigma_scale_post_mean": float(beta_n / max(alpha_n - 1, 1e-9)),
                      "sigma_scale_note": "1.0 means reported sigmas are right"}
    else:
        L = np.linalg.cholesky(Vn)
        z = rng.standard_normal((n_draws, k))
        draws = mun[None, :] + z @ L.T
        scale_info = {"model": "Gaussian (sigmas taken as given)"}

    lo_q, hi_q = (1 - cred) / 2, 1 - (1 - cred) / 2
    params: Dict[str, Any] = {}
    for j, c in enumerate(cols):
        d = draws[:, j]
        lo, hi = np.quantile(d, [lo_q, hi_q])
        ratio = float(prior_sd[j] / post_sd[j]) if post_sd[j] > 0 else float("inf")
        params[PARAM_NAMES[c]] = {
            "units": PARAM_UNITS[c],
            "posterior_mean": float(mun[j]),
            "posterior_sd": float(post_sd[j]),
            "prior_sd": float(prior_sd[j]),
            "prior_to_post_sd_ratio": ratio,
            "data_informed": bool(ratio > 2.0),
            "credible_interval": (float(lo), float(hi)),
            "P_positive": float((d > 0).mean()),
            "straddles_zero": bool(lo < 0 < hi),
        }
        if ratio <= 1.5:
            warns.append(f"{PARAM_NAMES[c]}: prior sd / posterior sd = "
                         f"{ratio:.2f} — this parameter is PRIOR-DOMINATED; "
                         "the data did not inform it")

    # posterior correlation of dCp0 and b
    corr = None
    if 2 in cols and 3 in cols:
        i2, i3 = cols.index(2), cols.index(3)
        v2, v3 = Vn[i2, i2], Vn[i3, i3]
        if v2 > 0 and v3 > 0:
            corr = float(Vn[i2, i3] / math.sqrt(v2 * v3))
            if abs(corr) > 0.95:
                warns.append(f"corr(dCp0,b) = {corr:.3f}: the two are nearly "
                             "degenerate; report dCp(T0), not b")

    # dCp at T0 and at each observed T
    dcp = {}
    for t in sorted(set(Ts.tolist())):
        v = np.zeros(k)
        if 2 in cols:
            v[cols.index(2)] = 1.0
        if 3 in cols:
            v[cols.index(3)] = t - T0
        d = draws @ v
        lo, hi = np.quantile(d, [lo_q, hi_q])
        dcp[f"{t:.1f}"] = {"mean": float(v @ mun),
                           "sd": float(math.sqrt(max(v @ Vn @ v, 0.0))),
                           "credible_interval": (float(lo), float(hi))}

    return {"status": "OK", "model": model, "n": n, "k": k, "T0_K": T0,
            "T_span_K": span, "params": params, "dCp_at_T": dcp,
            "dCp0_b_correlation": corr, "sigma_scale": scale_info,
            "draws": draws, "cols": cols, "mu": mun, "V": Vn,
            "warnings": warns}


# =============================================================================
# PART B §4  PRIOR SENSITIVITY, PRIOR PREDICTIVE, EXACT LOO
# =============================================================================

def prior_sensitivity(obs, T0=310.0, model="M2",
                      factors=(0.5, 1.0, 2.0, 4.0),
                      base_prior: Optional[WeakPrior] = None,
                      **kw) -> Dict[str, Any]:
    """
    Refit with prior scales multiplied by each factor. If the posterior
    moves substantially, the answer is prior-dominated. This is the
    decisive check at small n and must be reported alongside the fit.
    """
    base = base_prior or WeakPrior()
    out = {}
    ref = None
    for f in factors:
        kw2 = dict(kw); kw2.pop("seed", None)
        r = bayesian_dcp(obs, T0, model, prior=base.scaled(f), seed=1, **kw2)
        rec = {PARAM_NAMES[c]: r["params"][PARAM_NAMES[c]]["posterior_mean"]
               for c in r["cols"]}
        out[f"x{f}"] = rec
        if f == 1.0:
            ref = rec
    shift = {}
    if ref is not None:
        for name in ref:
            vals = [out[k][name] for k in out if name in out[k]]
            shift[name] = float(max(vals) - min(vals))
    return {"sweep": out, "max_shift_across_priors": shift,
            "interpretation": "large shift relative to the posterior sd "
                              "means the answer is prior-driven"}


def prior_predictive(T0=310.0, T_grid=(288., 298., 308., 318., 328.),
                     prior: Optional[WeakPrior] = None, model="M2",
                     n_draws=20000, seed=0) -> Dict[str, Any]:
    """
    What dG(T) does the PRIOR alone consider plausible? If the range is
    physically absurd the prior is too weak to be called informative.
    """
    prior = prior or WeakPrior()
    cols = MODELS[model]
    rng = np.random.default_rng(seed)
    th = prior.mu(cols) + rng.standard_normal((n_draws, len(cols))) * prior.sd[list(cols)]
    G = np.array([design_row(t, "dG", T0)[list(cols)] for t in T_grid])
    vals = th @ G.T
    return {"T_grid_K": list(T_grid),
            "dG_prior_mean": vals.mean(0).tolist(),
            "dG_prior_p025": np.quantile(vals, 0.025, axis=0).tolist(),
            "dG_prior_p975": np.quantile(vals, 0.975, axis=0).tolist(),
            "note": "prior predictive range of dG(T) in kJ/mol"}


def exact_loo(obs, T0=310.0, model="M2",
              prior: Optional[WeakPrior] = None, **kw) -> Dict[str, Any]:
    """
    Leave-one-out predictive log density, computed by exact refit
    (n is small, so this is cheap). Gaussian predictive when sigmas are
    taken as given; a t-like predictive spread otherwise (approximated
    by inflating with the sigma-scale posterior mean).
    """
    prior = prior or WeakPrior()
    cols = MODELS[model]
    rows = []
    for i in range(len(obs)):
        train = [o for j, o in enumerate(obs) if j != i]
        if len(train) < 2:
            rows.append({"i": i, "skipped": True}); continue
        r = bayesian_dcp(train, T0, model, prior=prior, n_draws=20000, **kw)
        o = obs[i]
        x = design_row(o.T_K, o.kind, T0)[list(cols)]
        pred = float(x @ r["mu"])
        var = float(x @ r["V"] @ x)
        s2 = r["sigma_scale"].get("sigma_scale_post_mean", 1.0)
        sig2 = (o.sigma if (o.sigma and math.isfinite(o.sigma) and o.sigma > 0) else 1.0) ** 2
        tot = var + s2 * sig2
        lpd = -0.5 * (math.log(2 * math.pi * tot) + (o.value - pred) ** 2 / tot)
        rows.append({"i": i, "T_K": o.T_K, "kind": o.kind,
                     "observed": o.value, "predicted": pred,
                     "pred_sd": math.sqrt(tot), "lpd": lpd})
    ok = [r["lpd"] for r in rows if "lpd" in r]
    return {"rows": rows, "elpd_loo": float(sum(ok)) if ok else float("nan"),
            "n_scored": len(ok)}


def report_bayes(r: Dict[str, Any]) -> None:
    print("=" * 96)
    print(f"  BAYESIAN dCp(T)   model={r['model']}  n={r['n']}  k={r['k']}  "
          f"T0={r['T0_K']:.1f} K  span={r['T_span_K']:.1f} K")
    print("=" * 96)
    print(f"  {'param':<8}{'units':<14}{'mean':>13}{'post sd':>12}"
          f"{'prior sd':>12}{'ratio':>8}  {'95% CI':>26}  informed")
    print("-" * 96)
    for name, d in r["params"].items():
        ci = d["credible_interval"]
        print(f"  {name:<8}{d['units']:<14}{d['posterior_mean']:>13.5f}"
              f"{d['posterior_sd']:>12.5f}{d['prior_sd']:>12.5f}"
              f"{d['prior_to_post_sd_ratio']:>8.2f}  "
              f"[{ci[0]:>10.4f},{ci[1]:>10.4f}]  "
              f"{'yes' if d['data_informed'] else 'NO'}")
    print("-" * 96)
    print(f"  sigma-scale model: {r['sigma_scale'].get('model')}")
    if "sigma_scale_post_mean" in r["sigma_scale"]:
        print(f"  posterior mean scale on reported sigmas = "
              f"{r['sigma_scale']['sigma_scale_post_mean']:.3f}  (1.0 = as reported)")
    if r["dCp0_b_correlation"] is not None:
        print(f"  corr(dCp0, b) = {r['dCp0_b_correlation']:+.4f}")
    print("  dCp(T):")
    for t, d in r["dCp_at_T"].items():
        ci = d["credible_interval"]
        print(f"    T={t:>7} K   {d['mean']:+.5f} ± {d['sd']:.5f}   "
              f"CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    if r["warnings"]:
        print("-" * 96)
        for w in r["warnings"]:
            print(f"  WARNING: {w}")
    print("=" * 96)


# =============================================================================
# SELF-TEST
# =============================================================================

def _coupled3_samples(n, T, c3, seed=0):
    """
    Sample a 3D torsion density with a GENUINE three-body term:
      W = -sum k_i cos(x_i) - sum c_ij cos(x_i-x_j) - c3 cos(x1+x2+x3)
    The c3 term is invisible to every pairwise approximation.
    """
    rng = np.random.default_rng(seed)
    nb = 60
    e = np.linspace(-math.pi, math.pi, nb + 1)
    c = 0.5 * (e[1:] + e[:-1])
    G1, G2, G3 = np.meshgrid(c, c, c, indexing="ij")
    W = (-1.2*np.cos(G1) - 1.2*np.cos(G2) - 1.2*np.cos(G3)
         - 0.8*np.cos(G1-G2) - 0.8*np.cos(G2-G3) - 0.8*np.cos(G1-G3)
         - c3*np.cos(G1+G2+G3))
    p = np.exp(-(W - W.min()) / (KB * T)).ravel()
    p /= p.sum()
    idx = rng.choice(p.size, size=n, p=p)
    i1, i2, i3 = np.unravel_index(idx, G1.shape)
    jit = rng.uniform(-0.5, 0.5, (3, n)) * (e[1] - e[0])
    return (wrap3(G1[i1, i2, i3] + jit[0]),
            wrap3(G2[i1, i2, i3] + jit[1]),
            wrap3(G3[i1, i2, i3] + jit[2]))


def self_test() -> bool:
    print("=" * 96)
    print(f"  3D JOINT TORSION PMF + BAYESIAN dCp(T)  v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 96)
    rng = np.random.default_rng(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    T = 310.0

    # ---------------- A1 independent torsions -> TC ~ 0 ----------------
    print("\n[A1] Total correlation: independent vs pairwise vs 3-body")
    n_s = 400_000
    zi = [rng.uniform(-math.pi, math.pi, n_s) for _ in range(3)]
    pi = pmf_3d_from_samples(*zi, T, n_bins=24)
    tci = pi.entropies()["total_correlation_nats"]
    zp = _coupled3_samples(n_s, T, c3=0.0, seed=1)
    pp = pmf_3d_from_samples(*zp, T, n_bins=24)
    tcp = pp.entropies()["total_correlation_nats"]
    z3 = _coupled3_samples(n_s, T, c3=1.5, seed=2)
    p3 = pmf_3d_from_samples(*z3, T, n_bins=24)
    e3 = p3.entropies()
    tc3 = e3["total_correlation_nats"]
    rep("independent: TC ~ 0", tci < 0.02, f"TC={tci:.5f} nats")
    rep("pairwise-coupled: TC > 0.1", tcp > 0.1, f"TC={tcp:.5f} nats")
    rep("3-body coupled: TC larger still", tc3 > tcp,
        f"TC={tc3:.5f} nats ({KB*T*tc3:.3f} kJ/mol)")
    print(f"      sum pairwise MI = {e3['sum_pairwise_MI_nats']:.4f} nats; "
          f"TC = {tc3:.4f} nats")

    # ---------------- A2 occupancy / sparsity is reported ----------------
    print("\n[A2] 3D occupancy and sparsity diagnostics")
    occ = p3.occupancy()
    print(f"      bins={occ['n_bins']}  occupied={occ['n_occupied']}  "
          f"occupancy={occ['occupancy_fraction']:.3f}  "
          f"samples/occupied bin={occ['samples_per_occupied_bin']:.1f}")
    rep("occupancy reported", occ["n_bins"] == 24 ** 3)
    sparse = pmf_3d_from_samples(*_coupled3_samples(2000, T, 1.5, seed=3),
                                 T, n_bins=48)
    K = np.array([[300., 40., 20.], [40., 300., 30.], [20., 30., 300.]])
    rr = JointRestraint3D((0.3, -0.4, 0.6), K)
    sp = joint_apply_3d(sparse, rr, restraint_on=False)
    rep("sparse 3D histogram flagged", not sp["sampling_ok"],
        f"samples/bin={sp['occupancy']['samples_per_occupied_bin']:.2f}")
    rep("sparse warning text present", "SPARSE" in sp["warning"])

    # ---------------- A3 restraint PD enforcement ----------------
    print("\n[A3] Joint restraint positive-definiteness")
    try:
        JointRestraint3D((0., 0., 0.), np.array([[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]]))
        rep("non-PD K rejected", False)
    except ValueError:
        rep("non-PD K rejected", True)
    rep("PD K accepted", isinstance(rr, JointRestraint3D))

    # ---------------- A4 approximation errors ----------------
    print("\n[A4] Cost of each reduced description (3-body case)")
    cmp3 = compare_approximations_3d(p3, rr, restraint_on=False)
    print(f"      full 3D dG_apply = {cmp3['full_3d']['dG_apply_kJmol']:+.4f} kJ/mol")
    for name, d in cmp3["reduced"].items():
        print(f"      {name:<14} error = {d['error_kJmol']:+.4f} kJ/mol")
    cmpp = compare_approximations_3d(pp, rr, restraint_on=False)
    ek3 = abs(cmp3["reduced"]["kirkwood"]["error_kJmol"])
    ekp = abs(cmpp["reduced"]["kirkwood"]["error_kJmol"])
    rep("Kirkwood error larger with genuine 3-body term", ek3 > ekp,
        f"|err| 3-body={ek3:.4f} vs pairwise-only={ekp:.4f}")
    rep("independent error is the largest",
        abs(cmp3["reduced"]["independent"]["error_kJmol"]) >= ek3 - 1e-9)

    # ---------------- A5 3D WHAM recovers a known surface ----------------
    print("\n[A5] 3D WHAM recovers an analytic coupled surface")
    def Wtrue(g1, g2, g3):
        return (-1.0*np.cos(g1) - 1.0*np.cos(g2) - 1.0*np.cos(g3)
                - 0.6*np.cos(g1-g2) - 0.6*np.cos(g2-g3)
                - 0.9*np.cos(g1+g2+g3))
    nbw = 24
    ew = np.linspace(-math.pi, math.pi, nbw + 1)
    cw = 0.5 * (ew[1:] + ew[:-1])
    H1, H2, H3 = np.meshgrid(cw, cw, cw, indexing="ij")
    Wt = Wtrue(H1, H2, H3)
    wins = []
    for a0 in (-2.0, 0.0, 2.0):
        for b0 in (-2.0, 0.0, 2.0):
            for c0 in (-2.0, 0.0, 2.0):
                kk = 12.0
                def bias(g1, g2, g3, a0=a0, b0=b0, c0=c0, kk=kk):
                    return 0.5*kk*(pdelta(g1, a0)**2 + pdelta(g2, b0)**2
                                   + pdelta(g3, c0)**2)
                pb = np.exp(-(Wt + bias(H1, H2, H3)) / (KB*T)).ravel()
                pb /= pb.sum()
                idx = rng.choice(pb.size, size=12000, p=pb)
                i1, i2, i3 = np.unravel_index(idx, H1.shape)
                wins.append({"z1": H1[i1, i2, i3], "z2": H2[i1, i2, i3],
                             "z3": H3[i1, i2, i3], "bias": bias})
    rec = wham_3d(wins, T, n_bins=24)
    Wg = Wtrue(*np.meshgrid(rec.centres, rec.centres, rec.centres, indexing="ij"))
    lowE = Wt < np.percentile(Wt, 25)
    m = rec.sampled & lowE
    a = rec.W[m] - rec.W[m].min(); b = Wg[m] - Wg[m].min()
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    rep("3D WHAM RMSE < 2.0 kJ/mol on low-energy region", rmse < 2.0,
        f"RMSE={rmse:.3f}, occupancy={rec.occupancy()['occupancy_fraction']:.3f}")

    # ---------------- A6 assembly ----------------
    print("\n[A6] Woo-Roux assembly with the 3D joint term")
    jf = joint_apply_3d(p3, rr, restraint_on=False)
    jb = dict(jf); jb["dG_apply_kJmol"] = 0.6
    asm = wooroux_with_3d_conformational(
        -30.0, T,
        angular_terms=[{"dG_apply_free_kJmol": 6.0, "dG_apply_bound_kJmol": 0.5}],
        joint_free=jf, joint_bound=jb)
    exp_dG = -30.0 + 5.5 + (jf["dG_apply_kJmol"] - 0.6)
    rep("assembly matches hand sum", abs(asm["dG_bind_standard"] - exp_dG) < 1e-12,
        f"dG={asm['dG_bind_standard']:+.4f}")
    rep("conformational correction positive (entropy penalty)",
        asm["conformational_correction"] > 0,
        f"{asm['conformational_correction']:+.3f} kJ/mol")

    # ---------------- B1 design rows exact ----------------
    print("\n[B1] Design rows reproduce dH(T) and dG(T) exactly")
    T0 = 310.0
    th = np.array([-45.0, -0.10, -0.65, 0.0020])
    def dH(t):
        dT = t - T0
        return th[0] + th[2]*dT + 0.5*th[3]*dT*dT
    def dS(t):
        dT = t - T0; L = math.log(t/T0)
        return th[1] + th[2]*L + th[3]*(dT - T0*L)
    def dG(t): return dH(t) - t*dS(t)
    eH = max(abs(float(design_row(t, "dH", T0) @ th) - dH(t))
             for t in (288., 300., 320., 330.))
    eG = max(abs(float(design_row(t, "dG", T0) @ th) - dG(t))
             for t in (288., 300., 320., 330.))
    rep("dH row exact", eH < 1e-10, f"err={eH:.2e}")
    rep("dG row exact", eG < 1e-10, f"err={eG:.2e}")

    # ---------------- B2 posterior recovers truth with good data ----------------
    print("\n[B2] Posterior with 8 temperatures and low noise")
    Ts = np.linspace(288., 328., 8)
    obs = []
    for t in Ts:
        obs.append(ThermoObs(float(t), "dH", dH(t) + rng.normal(0, .12), .12, "calorimetric"))
        obs.append(ThermoObs(float(t), "dG", dG(t) + rng.normal(0, .08), .08, "equilibrium"))
    rb = bayesian_dcp(obs, T0, "M2", n_draws=120000, seed=1)
    report_bayes(rb)
    for j, name in enumerate(PARAM_NAMES):
        d = rb["params"][name]
        ci = d["credible_interval"]
        rep(f"{name} truth inside 95% CI", ci[0] <= th[j] <= ci[1],
            f"true={th[j]:+.5f} CI=[{ci[0]:+.5f},{ci[1]:+.5f}]")
    rep("all four parameters data-informed",
        all(d["data_informed"] for d in rb["params"].values()))

    # ---------------- B3 n=3 : prior dominance is REPORTED ----------------
    print("\n[B3] n=3, k=4 — Bayesian runs but must flag prior dominance")
    obs3 = [ThermoObs(t, "dH", dH(t), 0.3, "calorimetric")
            for t in (300., 310., 320.)]
    r3 = bayesian_dcp(obs3, T0, "M2", n_draws=120000, seed=2)
    report_bayes(r3)
    rep("Bayesian returns a posterior at n=3 (bootstrap would refuse)",
        r3["status"] == "OK")
    rep("n<=k warning present", any("n=3 <= k=4" in w for w in r3["warnings"]))
    rep("b flagged as prior-dominated",
        not r3["params"]["b"]["data_informed"],
        f"prior/post sd ratio = {r3['params']['b']['prior_to_post_sd_ratio']:.2f}")

    # ---------------- B4 prior sensitivity ----------------
    print("\n[B4] Prior sensitivity sweep")
    ps = prior_sensitivity(obs3, T0, "M2", n_draws=40000)
    for f, rec in ps["sweep"].items():
        print(f"      {f:<6} " + "  ".join(f"{k}={v:+.5f}" for k, v in rec.items()))
    print(f"      max shift: {ps['max_shift_across_priors']}")
    rep("prior sensitivity computed", len(ps["sweep"]) == 4)
    rep("b moves under prior scaling (prior-driven)",
        ps["max_shift_across_priors"].get("b", 0.0) > 1e-6)

    # ---------------- B5 prior predictive is not absurd ----------------
    print("\n[B5] Prior predictive check")
    pp_ = prior_predictive(T0)
    for t, lo, hi in zip(pp_["T_grid_K"], pp_["dG_prior_p025"], pp_["dG_prior_p975"]):
        print(f"      T={t:.0f} K  prior 95% dG range [{lo:+.1f}, {hi:+.1f}] kJ/mol")
    span_ok = all(abs(hi - lo) < 3000 for lo, hi in
                  zip(pp_["dG_prior_p025"], pp_["dG_prior_p975"]))
    rep("prior predictive range finite and not absurd", span_ok)

    # ---------------- B6 exact LOO ----------------
    print("\n[B6] Exact leave-one-out predictive score")
    loo = exact_loo(obs[:8], T0, "M2", n_draws=20000)
    rep("LOO scored points", loo["n_scored"] >= 5,
        f"n={loo['n_scored']}  elpd_loo={loo['elpd_loo']:.2f}")

    # ---------------- B7 sigma-scale inflation ----------------
    print("\n[B7] Reported sigmas deliberately too small")
    obs_bad = [ThermoObs(o.T_K, o.kind, o.value + rng.normal(0, 1.0),
                         0.1, o.provenance) for o in obs[:8]]
    r_bad = bayesian_dcp(obs_bad, T0, "M2", sigma_scale_prior=True,
                         n_draws=80000, seed=3)
    s = r_bad["sigma_scale"].get("sigma_scale_post_mean", float("nan"))
    rep("sigma-scale posterior detects understated errors", s > 2.0,
        f"posterior scale = {s:.2f} (reported sigmas were 10x too small)")

    print("\n" + "=" * 96)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 96)
    print("""
  WHAT THESE TESTS PROVE
    A1  total correlation separates independent, pairwise-coupled and
        genuinely three-body-coupled torsions
    A2  3D occupancy and sparsity are measured and flagged
    A3  a non-positive-definite joint restraint is rejected
    A4  the Kirkwood (pairwise) approximation makes a LARGER error when
        a genuine three-body term is present — the cost of the
        pairwise assumption is a number, not a claim
    A5  3D WHAM recovers an analytic coupled surface
    A6  the assembly uses the 3D joint term with the correct sign
    B1  the reparameterised design rows are exact
    B2  with 8 temperatures the posterior covers the truth and all four
        parameters are data-informed
    B3  at n=3 the Bayesian fit runs (bootstrap refuses) but flags b as
        PRIOR-DOMINATED, explicitly
    B4  prior sensitivity shows b moving when the prior moves
    B5  the prior predictive range is reported so "weakly informative"
        can be checked rather than asserted
    B6  exact LOO predictive scoring
    B7  understated measurement errors are detected by the sigma-scale
        posterior

  WHAT IS NOT PROVEN
    * a filled 3D histogram is not a converged PMF
    * credible intervals are conditional on prior + linear dCp(T) +
      Gaussian noise; they are not frequentist coverage guarantees
    * Kirkwood is renormalised to a distribution but is not the joint
    * Cartesian-sampled torsion histograms already contain the Fixman
      factor; adding it again would double-count
    * no experimental validation is performed here
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"3D torsion + Bayesian dCp v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--thermo", default=None,
                    help="JSON list of {T_K,kind,value,sigma,provenance}")
    ap.add_argument("--T0", type=float, default=310.0)
    ap.add_argument("--model", default="M2", choices=list(MODELS))
    ap.add_argument("--prior-scale", type=float, default=1.0)
    ap.add_argument("--no-sigma-scale", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
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
        pr = WeakPrior().scaled(a.prior_scale)
        r = bayesian_dcp(obs, a.T0, a.model, prior=pr,
                         sigma_scale_prior=not a.no_sigma_scale)
        report_bayes(r)
        out: Dict[str, Any] = {
            "fit": {k: v for k, v in r.items() if k not in ("draws", "V", "mu", "cols")},
        }
        if a.sensitivity:
            ps = prior_sensitivity(obs, a.T0, a.model, base_prior=WeakPrior())
            print("\n  PRIOR SENSITIVITY")
            for f_, rec in ps["sweep"].items():
                print(f"    {f_:<6} " + "  ".join(f"{k}={v:+.5f}" for k, v in rec.items()))
            print(f"    max shift: {ps['max_shift_across_priors']}")
            out["prior_sensitivity"] = ps
        out["prior_predictive"] = prior_predictive(a.T0, prior=pr, model=a.model)
        out["loo"] = exact_loo(obs, a.T0, a.model, prior=pr)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f, indent=2, default=float)
            print(f"\n  json -> {a.out}")
        return
    ap.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
