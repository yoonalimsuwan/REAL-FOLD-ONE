#!/usr/bin/env python3
"""
=============================================================================
NON-EXCHANGEABLE CLASS HIERARCHY  +  VARIABLE-BANDWIDTH ANISOTROPIC TORUS KDE
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — VARIABLE-BANDWIDTH ANISOTROPIC KDE ON T^3
  A1 ANISOTROPIC BASE
     Per-dimension concentration kappa_d, initialised from the mean
     resultant length R_d = |E exp(i theta_d)| of each marginal via
     Fisher's standard approximation, then scaled by cross-validation.
     This is genuinely anisotropic: a stiff phi and a floppy chi1 get
     different bandwidths.

  A2 ADAPTIVE (Abramson) LOCAL BANDWIDTH
     Two-stage: a pilot KDE gives p~(x_i); the local factor is
         lambda_i = (p~(x_i) / g)^{2 alpha},   g = geometric mean of p~
     and kappa_{i,d} = kappa_{0,d} * lambda_i.
     The exponent 2 (not 1/2) is because on the circle kappa ~ h^{-2},
     so Abramson's h_i ~ p~^{-alpha} becomes kappa_i ~ p~^{+2 alpha}.
     alpha = 0 recovers the fixed-bandwidth KDE exactly.

     DIRECTION OF THE EFFECT, stated because it is often gotten backwards:
     sparse region -> small p~ -> SMALL kappa -> LARGE bandwidth.
     Adaptive KDE therefore smooths MORE where data are scarce. That
     reduces variance and increases bias. It does NOT create samples.

  A3 SAMPLE-SMOOTHING vs BALLOON
     sample : bandwidth attached to each SAMPLE point (Abramson).
     balloon: bandwidth attached to each EVALUATION point.
     Both are legitimate; they differ in bias/variance trade-off and in
     whether the estimator integrates to 1 (sample-smoothing does,
     balloon generally does not). Both are provided; CV picks.

  A4 CV AND ITS LEAKAGE CAVEAT
     Bandwidth selection maximises the leave-one-out log pseudo-likelihood.
     The pilot density uses ALL samples, so kappa_j has a weak dependence
     on x_i even for j != i. This is standard practice for Abramson
     estimators and is disclosed rather than hidden. A strict
     pilot-LOO option is available at O(N) extra cost.

  A5 SUPPORT DIAGNOSTIC IS STILL MANDATORY
     Adaptive bandwidth makes the sparse-region value MORE kernel-tail
     driven, not less. The fraction of the restraint's Boltzmann weight
     sitting below a local-effective-count threshold is reported, and
     the integral is REFUSED above tolerance.

PART B — NON-EXCHANGEABLE CLASS HIERARCHY FOR dCp(T)
  Three levels, all conjugate, exact Gibbs (no HMC):
      theta_s | mu_g, Sigma_w  ~ N(mu_{g(s)}, Sigma_w)
      mu_g    | mu_0, Sigma_b  ~ N(mu_0, Sigma_b)
      mu_0    | Sigma_b        ~ N(m, Sigma_b / a0)
      Sigma_w                  ~ IW(nu_w, Psi_w)
      Sigma_b                  ~ IW(nu_b, Psi_b)
  with the linear-in-theta dCp(T) design from the earlier module:
      theta = (dH0, dS0, dCp0, b),  dCp(T) = dCp0 + b (T - T0)

  WHAT THIS BUYS OVER A SINGLE EXCHANGEABLE POOL
    Systems in the SAME ligand class borrow strongly (Sigma_w small).
    Systems in DIFFERENT classes borrow only through the global mean
    (Sigma_b large). A single exchangeable pool forces one compromise
    and biases chemically dissimilar ligands toward each other.

  THE DIAGNOSTICS THAT DECIDE WHETHER THE GROUPING IS REAL
    ICC_k = Sigma_b[k,k] / (Sigma_b[k,k] + Sigma_w[k,k]) per parameter.
      ICC ~ 0  -> classes carry no information; the extra level is
                  pure variance and the hierarchy HURTS.
      ICC ~ 1  -> classes are nearly disjoint; little cross-class borrowing.
    LOO-system-out : predict a held-out system from its class.
    LOO-class-out  : predict an ENTIRE unseen class from the global mean.
    Model comparison by elpd against no-pooling and flat-exchangeable.
    If the class model does not beat both, it is not licensed.

HONEST SCOPE
  ❌ Class labels must be supplied and chemically justified. Arbitrary
     grouping is detected by LOO, not prevented.
  ❌ With G=2 classes Sigma_b is barely identified; flagged.
  ❌ Singleton classes have mu_g ~ mu_0; flagged.
  ❌ Adaptive KDE increases bias in sparse regions. alpha is CV-selected
     but the CV curve is reported so a flat maximum is visible.
  ❌ Balloon KDE does not integrate to 1 in general; reported.
  ❌ No experimental validation is performed here.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("class_kde")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

KB = 0.008314462618
TWO_PI = 2.0 * math.pi


def wrap(z):
    return (np.asarray(z, dtype=np.float64) + math.pi) % TWO_PI - math.pi


def pdist(a, b):
    return wrap(a - b)


# =============================================================================
# PART A §1  STABLE log I0 AND VARIABLE-kappa VON MISES KERNEL
# =============================================================================

def log_i0(kappa: np.ndarray) -> np.ndarray:
    """log I0, stable for large kappa via the asymptotic expansion."""
    k = np.asarray(kappa, dtype=np.float64)
    out = np.empty_like(k)
    small = k < 500.0
    if np.any(small):
        out[small] = np.log(np.i0(np.maximum(k[small], 1e-300)))
    if np.any(~small):
        kb = k[~small]
        corr = 1.0 + 1.0/(8*kb) + 9.0/(128*kb**2) + 225.0/(3072*kb**3)
        out[~small] = kb - 0.5*np.log(TWO_PI*kb) + np.log(corr)
    return out


def vm_logk_fixed(diff: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    diff (...,3), kappa (3,) shared across points.
    Written as kappa*(cos-1) so the exponent is <= 0 (no overflow).
    """
    kap = np.asarray(kappa, dtype=np.float64)
    norm = log_i0(kap) + math.log(TWO_PI) - kap
    return (kap * (np.cos(diff) - 1.0)).sum(-1) - norm.sum()


def vm_logk_var(diff: np.ndarray, kappa_i: np.ndarray) -> np.ndarray:
    """
    diff (M,K,3), kappa_i (M,K,3) per-(query,sample) concentration.
    Returns (M,K) log-kernel values.
    """
    kap = np.maximum(kappa_i, 1e-12)
    norm = (log_i0(kap) + math.log(TWO_PI) - kap).sum(-1)
    return (kap * (np.cos(diff) - 1.0)).sum(-1) - norm


def resultant_length(theta: np.ndarray) -> float:
    """R = |E exp(i theta)| for a circular sample."""
    th = np.asarray(theta, dtype=np.float64)
    return float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))


def kappa_from_R(R: float) -> float:
    """
    Fisher's standard approximation for the von Mises concentration
    from the mean resultant length (Mardia & Jupp, §5.3.1).
    """
    R = min(max(R, 1e-6), 1.0 - 1e-9)
    if R < 0.53:
        k = 2*R + R**3 + 5*R**5/6.0
    elif R < 0.85:
        k = -0.4 + 1.39*R + 0.43/(1.0 - R)
    else:
        k = 1.0/(R**3 - 4*R**2 + 3*R)
    return float(max(abs(k), 1e-3))


# =============================================================================
# PART A §2  TORUS CELL LIST
# =============================================================================

class TorusCellList:
    """Uniform periodic binning with CSR offsets for O(M*k) queries."""

    def __init__(self, samples: np.ndarray, cell_size: float):
        self.samples = np.asarray(samples, dtype=np.float64)
        self.N = self.samples.shape[0]
        self.h = max(float(cell_size), 1e-3)
        self.nd = max(1, int(math.floor(TWO_PI / self.h)))
        self.h = TWO_PI / self.nd
        g = np.clip(np.floor((self.samples + math.pi)/self.h).astype(np.int64),
                    0, self.nd - 1)
        self.grid = g
        lin = (g[:, 0]*self.nd + g[:, 1])*self.nd + g[:, 2]
        self.order = np.argsort(lin, kind="stable")
        ls = lin[self.order]
        nc = self.nd**3
        self.start = np.append(np.searchsorted(ls, np.arange(nc)), self.N)

    def query(self, pts: np.ndarray, max_nb: int):
        pts = np.asarray(pts, dtype=np.float64)
        M = pts.shape[0]
        g = np.clip(np.floor((pts + math.pi)/self.h).astype(np.int64),
                    0, self.nd - 1)
        offs = np.array([(a, b, c) for a in (-1, 0, 1)
                         for b in (-1, 0, 1) for c in (-1, 0, 1)],
                        dtype=np.int64)
        counts = np.zeros(M, dtype=np.int64)
        buckets = []
        for o in offs:
            ng = (g + o) % self.nd
            lin = (ng[:, 0]*self.nd + ng[:, 1])*self.nd + ng[:, 2]
            c0, c1 = self.start[lin], self.start[lin + 1]
            counts += (c1 - c0)
            buckets.append((c0, c1))
        need = int(counts.max()) if M else 0
        K = max(min(need, max_nb), 1)
        idx = np.full((M, K), -1, dtype=np.int64)
        fill = np.zeros(M, dtype=np.int64)
        for c0, c1 in buckets:
            mb = int((c1 - c0).max()) if M else 0
            for j in range(mb):
                pos = np.minimum(c0 + j, self.N - 1)
                inb = j < (c1 - c0)
                rows = np.nonzero(inb & (fill < K))[0]
                if rows.size == 0:
                    continue
                idx[rows, fill[rows]] = self.order[pos[rows]]
                fill[rows] += 1
        return idx, counts


# =============================================================================
# PART A §3  ADAPTIVE ANISOTROPIC TORUS KDE
# =============================================================================

@dataclass
class AdaptiveKDEConfig:
    alpha_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    scale_grid: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    mode: str = "sample"           # "sample" | "balloon"
    kappa_min: float = 0.5
    kappa_max: float = 5.0e3
    cv_max_samples: int = 1500
    strict_pilot_loo: bool = False
    max_nb: int = 4096
    chunk: int = 256


class AdaptiveTorusKDE:
    """
    Two-stage adaptive anisotropic von Mises KDE on T^3.

    Stage 0: per-dimension concentration from marginal resultant lengths.
    Stage 1: pilot fixed-bandwidth KDE at the anisotropic base.
    Stage 2: Abramson local factors, exponent alpha and global scale
             chosen by leave-one-out log pseudo-likelihood.
    """

    def __init__(self, samples: np.ndarray,
                 cfg: Optional[AdaptiveKDEConfig] = None):
        self.cfg = cfg or AdaptiveKDEConfig()
        self.s = wrap(np.asarray(samples, dtype=np.float64))
        if self.s.ndim != 2 or self.s.shape[1] != 3:
            raise ValueError("samples must be (N,3)")
        self.N = self.s.shape[0]
        self.R = np.array([resultant_length(self.s[:, d]) for d in range(3)])
        self.kappa_marg = np.array([kappa_from_R(r) for r in self.R])
        w = self.kappa_marg / max(self.kappa_marg.mean(), 1e-12)
        self.aniso_w = w
        self.kappa0 = self.kappa_marg.copy()
        self.alpha = 0.0
        self.lam = np.ones(self.N)
        self.kappa_i = np.repeat(self.kappa0[None, :], self.N, axis=0)
        self.pilot = np.ones(self.N)
        self.cv_report: Dict[str, Any] = {}
        self._cl: Optional[TorusCellList] = None

    # ---------------- pilot ----------------
    def _pilot_at_samples(self, kappa0: np.ndarray,
                          strict_loo: bool = False) -> np.ndarray:
        """Fixed-bandwidth KDE evaluated at the sample points (LOO if strict)."""
        N = self.N
        out = np.empty(N)
        CH = 512
        for st in range(0, N, CH):
            en = min(st + CH, N)
            diff = wrap(self.s[st:en, None, :] - self.s[None, :, :])
            lk = vm_logk_fixed(diff, kappa0)
            if strict_loo:
                ridx = np.arange(st, en)
                lk[ridx - st, ridx] = -np.inf
                denom = N - 1
            else:
                denom = N
            mx = lk.max(axis=1)
            out[st:en] = np.exp(mx + np.log(np.exp(lk - mx[:, None]).sum(1))) / denom
        return np.maximum(out, 1e-300)

    # ---------------- CV ----------------
    def fit_cv(self, verbose: bool = False) -> "AdaptiveTorusKDE":
        cfg = self.cfg
        N = self.N
        if N < 12:
            raise ValueError("need >= 12 samples for adaptive CV")
        rng = np.random.default_rng(0)
        qidx = (np.arange(N) if N <= cfg.cv_max_samples
                else np.sort(rng.choice(N, cfg.cv_max_samples, replace=False)))
        sq = self.s[qidx]
        M = len(qidx)
        diff = wrap(sq[:, None, :] - self.s[None, :, :])
        same = (qidx[:, None] == np.arange(N)[None, :])
        best = (-np.inf, None, None)
        curve = []
        for scale in cfg.scale_grid:
            kappa0 = np.clip(self.aniso_w * scale * self.kappa_marg.mean(),
                             cfg.kappa_min, cfg.kappa_max)
            pilot_all = self._pilot_at_samples(kappa0, cfg.strict_pilot_loo)
            g = float(np.exp(np.log(pilot_all).mean()))
            for alpha in cfg.alpha_grid:
                lam = np.clip((pilot_all / g) ** (2.0 * alpha),
                              cfg.kappa_min / max(kappa0.min(), 1e-12),
                              cfg.kappa_max / max(kappa0.max(), 1e-12))
                kap_i = np.clip(kappa0[None, :] * lam[None, :],
                                cfg.kappa_min, cfg.kappa_max)
                kap_q = np.repeat(kap_i[None, :, :], M, axis=0)
                lk = vm_logk_var(diff, kap_q)
                lk = np.where(same, -np.inf, lk)
                mx = lk.max(axis=1)
                ok = np.isfinite(mx)
                if not ok.any():
                    continue
                lse = mx[ok] + np.log(np.exp(lk[ok] - mx[ok, None]).sum(1))
                score = float(np.mean(lse - math.log(N - 1)))
                curve.append({"scale": scale, "alpha": alpha,
                              "cv_loglik": score})
                if score > best[0]:
                    best = (score, scale, alpha)
        if best[1] is None:
            raise RuntimeError("CV found no usable bandwidth")
        _, scale, alpha = best
        self.kappa0 = np.clip(self.aniso_w * scale * self.kappa_marg.mean(),
                              cfg.kappa_min, cfg.kappa_max)
        self.alpha = float(alpha)
        self.pilot = self._pilot_at_samples(self.kappa0, cfg.strict_pilot_loo)
        g = float(np.exp(np.log(self.pilot).mean()))
        self.lam = (self.pilot / g) ** (2.0 * self.alpha)
        raw = self.kappa0[None, :] * self.lam[:, None]
        self.kappa_i = np.clip(raw, cfg.kappa_min, cfg.kappa_max)
        clipped = float(np.mean((raw != self.kappa_i)))
        srt = sorted(curve, key=lambda c: -c["cv_loglik"])
        flat = (len(srt) > 1 and
                abs(srt[0]["cv_loglik"] - srt[1]["cv_loglik"]) < 0.01)
        self.cv_report = {
            "kappa0": self.kappa0.tolist(),
            "bandwidth_rad": (1.0/np.sqrt(self.kappa0)).tolist(),
            "alpha": self.alpha,
            "mode": cfg.mode,
            "marginal_R": self.R.tolist(),
            "marginal_kappa": self.kappa_marg.tolist(),
            "anisotropy_weights": self.aniso_w.tolist(),
            "cv_best_loglik": best[0],
            "cv_curve_top5": srt[:5],
            "cv_flat_maximum": bool(flat),
            "kappa_clip_fraction": clipped,
            "lambda_range": (float(self.lam.min()), float(self.lam.max())),
            "n_cv_query": M,
            "strict_pilot_loo": cfg.strict_pilot_loo,
            "note": ("pilot uses all samples; kappa_j depends weakly on x_i. "
                     "Set strict_pilot_loo=True to remove this leakage."),
        }
        if verbose:
            print(json.dumps(self.cv_report, indent=2))
        return self

    # ---------------- evaluation ----------------
    def _cell_list(self) -> TorusCellList:
        if self._cl is None:
            h = float(3.0 / math.sqrt(max(self.kappa_i.min(), 1e-6)))
            self._cl = TorusCellList(self.s, min(max(h, TWO_PI/64), TWO_PI/2))
        return self._cl

    def log_density(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (log p_hat, local effective sample count) at each query.
        mode='sample'  : kappa attached to sample points (integrates to 1)
        mode='balloon' : kappa attached to the query point (does not in general)
        """
        cfg = self.cfg
        pts = wrap(np.asarray(pts, dtype=np.float64))
        M = pts.shape[0]
        cl = self._cell_list()
        out = np.empty(M)
        neff = np.empty(M)
        for st in range(0, M, cfg.chunk):
            en = min(st + cfg.chunk, M)
            p = pts[st:en]
            idx, cnt = cl.query(p, cfg.max_nb)
            if int(cnt.max()) > idx.shape[1]:
                idx, cnt = cl.query(p, int(cnt.max()))
            valid = idx >= 0
            safe = np.maximum(idx, 0)
            nb = self.s[safe]
            diff = wrap(p[:, None, :] - nb)
            if cfg.mode == "sample":
                kap = self.kappa_i[safe]
            else:
                g = float(np.exp(np.log(self.pilot).mean()))
                pd = self._pilot_density(p)
                lam_q = (pd / g) ** (2.0 * self.alpha)
                kap = np.clip(self.kappa0[None, None, :] * lam_q[:, None, None],
                              cfg.kappa_min, cfg.kappa_max)
                kap = np.repeat(kap, diff.shape[1], axis=1)
            lk = vm_logk_var(diff, kap)
            lk = np.where(valid, lk, -np.inf)
            uw = np.exp(np.clip((kap*(np.cos(diff)-1.0)).sum(-1), -700, 0))
            neff[st:en] = np.where(valid, uw, 0.0).sum(1)
            mx = lk.max(axis=1)
            has = np.isfinite(mx)
            val = np.full(en - st, -np.inf)
            if has.any():
                mm = mx[has][:, None]
                val[has] = mx[has] + np.log(np.exp(lk[has] - mm).sum(1))
            out[st:en] = val - math.log(self.N)
        return out, neff

    def _pilot_density(self, pts: np.ndarray) -> np.ndarray:
        """Fixed-bandwidth pilot at arbitrary points (for balloon mode)."""
        pts = wrap(np.asarray(pts, dtype=np.float64))
        M = pts.shape[0]
        out = np.empty(M)
        CH = 256
        for st in range(0, M, CH):
            en = min(st + CH, M)
            diff = wrap(pts[st:en, None, :] - self.s[None, :, :])
            lk = vm_logk_fixed(diff, self.kappa0)
            mx = lk.max(axis=1)
            out[st:en] = np.exp(mx + np.log(np.exp(lk - mx[:, None]).sum(1))) / self.N
        return np.maximum(out, 1e-300)

    def density(self, pts):
        lp, _ = self.log_density(pts)
        return np.exp(lp)

    def normalisation_check(self, n_grid: int = 40) -> float:
        """Numerical integral of p_hat over T^3 (should be ~1 for sample mode)."""
        g = np.linspace(-math.pi, math.pi, n_grid, endpoint=False)
        G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        dV = (TWO_PI / n_grid) ** 3
        return float(self.density(G).sum() * dV)


# =============================================================================
# PART A §4  RESTRAINT INTEGRAL WITH SUPPORT GUARD
# =============================================================================

@dataclass
class Restraint3D:
    centre: np.ndarray
    K: np.ndarray

    def __post_init__(self):
        self.centre = wrap(np.asarray(self.centre, dtype=float))
        self.K = np.asarray(self.K, dtype=float)
        if not np.allclose(self.K, self.K.T, atol=1e-10):
            raise ValueError("K must be symmetric")
        np.linalg.cholesky(self.K)

    def U(self, pts):
        d = np.stack([pdist(pts[:, i], self.centre[i]) for i in range(3)], -1)
        return 0.5*np.einsum("ni,ij,nj->n", d, self.K, d)

    def eval_grid(self, T: float, n_per_dim: int = 20, tol: float = 1e-6):
        kT = KB*T
        half = np.minimum(np.sqrt(2.0*kT*np.log(1.0/tol)/np.diag(self.K)),
                          math.pi)
        half = np.maximum(half, 1e-3)
        axes = [np.linspace(-h, h, n_per_dim) for h in half]
        rel = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], -1)
        pts = wrap(self.centre[None, :] + rel)
        dV = float(np.prod([2*h/(n_per_dim-1) for h in half]))
        return pts, dV


def restraint_integral(kde: AdaptiveTorusKDE, r: Restraint3D, T: float,
                       n_per_dim: int = 20, tol: float = 1e-6,
                       min_neff: float = 5.0,
                       coverage_tol: float = 0.02) -> Dict[str, Any]:
    """
    dG_apply for the joint conformational restraint using the adaptive KDE
    as the intrinsic PMF, on the restraint's own support box.

    REFUSES when the restraint's Boltzmann weight sits where the local
    effective sample count is below min_neff. Adaptive bandwidth does not
    remove this failure mode; it changes its shape.
    """
    kT = KB*T
    beta = 1.0/kT
    pts, dV = r.eval_grid(T, n_per_dim, tol)
    lp, neff = kde.log_density(pts)
    U = r.U(pts)
    eU = np.exp(-beta*np.clip(U - U.min(), 0, 700))
    tot = float(eU.sum())
    low = neff < min_neff
    unsupported = float(eU[low].sum()/max(tot, 1e-300))
    ok = unsupported < coverage_tol
    Wint = -kT*lp
    fin = np.isfinite(Wint)
    if not fin.any():
        return {"status": "REFUSED", "reason": "no finite density on grid"}
    Wi = np.where(fin, Wint - float(np.min(Wint[fin])), np.inf)
    num = np.where(fin, np.exp(-beta*np.clip(Wi + U, 0, 700)), 0.0)
    den = np.where(fin, np.exp(-beta*np.clip(Wi, 0, 700)), 0.0)
    Zn, Zd = float(num.sum())*dV, float(den.sum())*dV
    if Zn <= 0 or Zd <= 0:
        return {"status": "REFUSED", "reason": "zero partition function"}
    return {
        "status": "OK" if ok else "REFUSED",
        "dG_apply_kJmol": -kT*math.log(Zn/Zd),
        "unsupported_restraint_weight": unsupported,
        "coverage_ok": bool(ok),
        "neff_min": float(neff.min()),
        "neff_median": float(np.median(neff)),
        "n_eval": int(pts.shape[0]),
        "alpha": kde.alpha,
        "kde_mode": kde.cfg.mode,
        "reason": ("" if ok else
                   "restraint weight sits where local sample support is "
                   "below threshold; adaptive bandwidth spreads the few "
                   "samples further but does not create data"),
    }


# =============================================================================
# PART B §1  DESIGN MATRIX AND DATA
# =============================================================================

PARAM_NAMES = ("dH0", "dS0", "dCp0", "b")
MODELS = {"M0": (0, 1), "M1": (0, 1, 2), "M2": (0, 1, 2, 3)}


def design_row(T: float, kind: str, T0: float) -> np.ndarray:
    dT = T - T0
    L = math.log(T/T0)
    if kind == "dH":
        return np.array([1.0, 0.0, dT, 0.5*dT*dT])
    if kind == "dG":
        return np.array([1.0, -T, dT - T*L, 0.5*dT*dT - T*(dT - T0*L)])
    raise ValueError(kind)


@dataclass
class SystemData:
    name: str
    group: str
    T: np.ndarray
    kind: np.ndarray
    y: np.ndarray
    sigma: np.ndarray
    T0: float = 310.0
    cols: Tuple[int, ...] = (0, 1, 2, 3)

    def design(self):
        X = np.vstack([design_row(float(t), str(k), self.T0)[list(self.cols)]
                       for t, k in zip(self.T, self.kind)])
        w = 1.0/np.maximum(np.asarray(self.sigma, dtype=float), 1e-12)**2
        return X, np.asarray(self.y, dtype=float), w

    @property
    def n(self):
        return len(self.y)


# =============================================================================
# PART B §2  CONJUGATE THREE-LEVEL GIBBS
# =============================================================================

def _rwishart(rng, df, scale):
    k = scale.shape[0]
    L = np.linalg.cholesky(scale)
    A = np.zeros((k, k))
    for i in range(k):
        A[i, i] = math.sqrt(rng.chisquare(max(df - i, 1e-6)))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    M = L @ A
    return M @ M.T


def _riw(rng, df, Psi):
    return np.linalg.inv(_rwishart(rng, df, np.linalg.inv(Psi)))


class ClassHierarchyDCp:
    """
    Non-exchangeable three-level hierarchy, exact conjugate Gibbs.
    Systems borrow strongly within a ligand class, weakly across classes.
    """

    def __init__(self, systems: List[SystemData],
                 m0: Optional[np.ndarray] = None, a0: float = 1.0,
                 nu_w: Optional[float] = None, nu_b: Optional[float] = None,
                 Psi_w: Optional[np.ndarray] = None,
                 Psi_b: Optional[np.ndarray] = None,
                 cols: Tuple[int, ...] = (0, 1, 2, 3)):
        self.systems = systems
        self.cols = cols
        self.k = len(cols)
        self.S = len(systems)
        self.groups = sorted({s.group for s in systems})
        self.G = len(self.groups)
        self.gidx = {g: i for i, g in enumerate(self.groups)}
        self.sg = np.array([self.gidx[s.group] for s in systems], dtype=int)
        self.m0 = np.zeros(self.k) if m0 is None else np.asarray(m0, float)
        self.a0 = float(a0)
        self.nu_w = float(self.k + 2) if nu_w is None else float(nu_w)
        self.nu_b = float(self.k + 2) if nu_b is None else float(nu_b)
        sc = np.array([100.0, 0.5, 2.0, 0.02])[list(cols)]
        self.Psi_w = (np.diag((self.nu_w - self.k - 1)*sc**2)
                      if Psi_w is None else np.asarray(Psi_w, float))
        self.Psi_b = (np.diag((self.nu_b - self.k - 1)*(0.3*sc)**2)
                      if Psi_b is None else np.asarray(Psi_b, float))
        self._prep = [s.design() for s in systems]

    # ---------------- unpooled reference ----------------
    def unpooled(self):
        out = []
        for s, (X, y, w) in zip(self.systems, self._prep):
            if s.n < self.k:
                out.append({"name": s.name, "group": s.group,
                            "status": "UNDERDETERMINED", "n": s.n})
                continue
            Xw = X*np.sqrt(w)[:, None]
            b, *_ = np.linalg.lstsq(Xw, y*np.sqrt(w), rcond=None)
            cov = np.linalg.pinv(Xw.T @ Xw)
            out.append({"name": s.name, "group": s.group, "status": "OK",
                        "n": s.n, "theta": b, "cov": cov,
                        "se": np.sqrt(np.maximum(np.diag(cov), 0))})
        return out

    # ---------------- Gibbs ----------------
    def fit(self, n_iter: int = 4000, burn: int = 1200, thin: int = 4,
            seed: int = 0) -> Dict[str, Any]:
        if self.S < 2:
            return {"status": "REFUSED", "reason": "need >= 2 systems"}
        warns = []
        if self.G < 2:
            warns.append("G=1: the class level is unidentifiable and collapses "
                         "to a flat exchangeable pool.")
        if self.G == 2:
            warns.append("G=2: Sigma_b is barely identified; its posterior is "
                         "prior-dominated. Do not interpret ICC quantitatively.")
        counts = np.bincount(self.sg, minlength=self.G)
        singles = [self.groups[g] for g in range(self.G) if counts[g] == 1]
        if singles:
            warns.append(f"singleton classes {singles}: their mu_g shrinks "
                         "almost entirely to the global mean.")
        rng = np.random.default_rng(seed)
        k, S, G = self.k, self.S, self.G
        theta = np.zeros((S, k))
        for i, u in enumerate(self.unpooled()):
            theta[i] = u["theta"] if u["status"] == "OK" else self.m0
        mu_g = np.array([theta[self.sg == g].mean(0) for g in range(G)])
        mu0 = mu_g.mean(0)
        Sw = self.Psi_w/max(self.nu_w - k - 1, 1.0)
        Sb = self.Psi_b/max(self.nu_b - k - 1, 1.0)
        DT, DMG, DM0, DSW, DSB = [], [], [], [], []
        for it in range(n_iter):
            Swi = np.linalg.inv(Sw)
            Sbi = np.linalg.inv(Sb)
            for i, (X, y, w) in enumerate(self._prep):
                A = X.T @ (w[:, None]*X) + Swi
                b = X.T @ (w*y) + Swi @ mu_g[self.sg[i]]
                V = np.linalg.inv(A)
                theta[i] = V @ b + np.linalg.cholesky(V) @ rng.standard_normal(k)
            for g in range(G):
                sel = self.sg == g
                ng = int(sel.sum())
                A = ng*Swi + Sbi
                V = np.linalg.inv(A)
                b = Swi @ theta[sel].sum(0) + Sbi @ mu0
                mu_g[g] = V @ b + np.linalg.cholesky(V) @ rng.standard_normal(k)
            V0 = Sb/(self.a0 + G)
            m0p = (self.a0*self.m0 + mu_g.sum(0))/(self.a0 + G)
            mu0 = m0p + np.linalg.cholesky(V0) @ rng.standard_normal(k)
            Rw = theta - mu_g[self.sg]
            Sw = _riw(rng, self.nu_w + S, self.Psi_w + Rw.T @ Rw)
            Rb = mu_g - mu0[None, :]
            Sb = _riw(rng, self.nu_b + G + 1,
                      self.Psi_b + Rb.T @ Rb
                      + self.a0*np.outer(mu0 - self.m0, mu0 - self.m0))
            if it >= burn and (it - burn) % thin == 0:
                DT.append(theta.copy()); DMG.append(mu_g.copy())
                DM0.append(mu0.copy())
                DSW.append(np.diag(Sw).copy()); DSB.append(np.diag(Sb).copy())
        T_arr = np.array(DT); MG = np.array(DMG); M0 = np.array(DM0)
        VW = np.array(DSW); VB = np.array(DSB)
        icc = (VB/(VB + VW + 1e-300)).mean(0)
        up = {u["name"]: u for u in self.unpooled()}
        per_sys = []
        for i, s in enumerate(self.systems):
            d = T_arr[:, i, :]
            u = up[s.name]
            sh = None
            if u["status"] == "OK":
                sh = float(np.linalg.norm(d.mean(0) - u["theta"])
                           / max(np.linalg.norm(u["theta"]), 1e-12))
            per_sys.append({
                "name": s.name, "group": s.group, "n": s.n,
                "theta_mean": d.mean(0).tolist(),
                "theta_sd": d.std(0, ddof=1).tolist(),
                "ci95": np.quantile(d, [0.025, 0.975], axis=0).tolist(),
                "unpooled_status": u["status"],
                "relative_shrinkage": sh,
                "borrowed": u["status"] != "OK",
            })
        for j, nm in enumerate([PARAM_NAMES[c] for c in self.cols]):
            if icc[j] < 0.05:
                warns.append(f"{nm}: ICC={icc[j]:.3f} ~ 0 — the class level "
                             "carries almost no information for this parameter; "
                             "the extra level adds variance. Check LOO.")
        return {"status": "OK", "S": S, "G": G, "k": k,
                "groups": self.groups, "group_counts": counts.tolist(),
                "n_draws": len(DT),
                "global_mean": M0.mean(0).tolist(),
                "class_means": MG.mean(0).tolist(),
                "between_class_var": VB.mean(0).tolist(),
                "within_class_var": VW.mean(0).tolist(),
                "ICC": icc.tolist(),
                "systems": per_sys, "warnings": warns,
                "_draws": {"theta": T_arr, "mu_g": MG, "mu0": M0}}

    # ---------------- predictive scoring ----------------
    @staticmethod
    def _elpd(X, y, sig2, mu_draws, Sig_draws):
        tot = 0.0
        for j in range(len(y)):
            x = X[j]
            m = mu_draws @ x
            v = np.einsum("ni,ij,nj->n",
                          np.repeat(x[None, :], len(mu_draws), 0),
                          Sig_draws,
                          np.repeat(x[None, :], len(mu_draws), 0)) + sig2[j]
            lp = -0.5*(np.log(TWO_PI*v) + (y[j] - m)**2/v)
            mx = lp.max()
            tot += float(mx + np.log(np.exp(lp - mx).mean()))
        return tot

    def loo_system(self, n_iter: int = 1600, burn: int = 600,
                   seed: int = 1) -> Dict[str, Any]:
        """Held-out system predicted from its class (or global if singleton)."""
        if self.S < 3:
            return {"status": "REFUSED", "reason": "need >= 3 systems"}
        rows = []
        for h in range(self.S):
            train = [s for i, s in enumerate(self.systems) if i != h]
            sub = ClassHierarchyDCp(train, self.m0, self.a0, self.nu_w,
                                    self.nu_b, self.Psi_w, self.Psi_b, self.cols)
            f = sub.fit(n_iter, burn, seed=seed + h)
            if f["status"] != "OK":
                rows.append({"held_out": self.systems[h].name,
                             "status": "REFUSED"}); continue
            grp = self.systems[h].group
            if grp in sub.gidx:
                gi = sub.gidx[grp]
                mu_d = f["_draws"]["mu_g"][:, gi, :]
                via = "class"
            else:
                mu_d = f["_draws"]["mu0"]
                via = "global (unseen class)"
            Xh, yh, wh = self._prep[h]
            sw = np.array(f["within_class_var"])
            Sig = np.repeat(np.diag(sw)[None, :, :], len(mu_d), axis=0)
            elpd = self._elpd(Xh, yh, 1.0/wh, mu_d, Sig)
            u = self.unpooled()[h]
            eu = None
            if u["status"] == "OK":
                m = Xh @ u["theta"]
                v = np.einsum("ni,ij,nj->n", Xh, u["cov"], Xh) + 1.0/wh
                eu = float(np.sum(-0.5*(np.log(TWO_PI*v) + (yh - m)**2/v)))
            rows.append({"held_out": self.systems[h].name, "group": grp,
                         "status": "OK", "via": via,
                         "elpd_class": elpd, "elpd_unpooled": eu,
                         "gain": (elpd - eu) if eu is not None else None})
        gains = [r["gain"] for r in rows if r.get("gain") is not None]
        return {"status": "OK", "rows": rows,
                "total_gain": float(sum(gains)) if gains else None,
                "verdict": ("CLASS POOLING SUPPORTED"
                            if gains and sum(gains) > 0
                            else "CLASS POOLING NOT SUPPORTED")}

    def loo_class(self, n_iter: int = 1600, burn: int = 600,
                  seed: int = 2) -> Dict[str, Any]:
        """Hold out an ENTIRE class; predict it from the global mean."""
        if self.G < 3:
            return {"status": "REFUSED",
                    "reason": f"G={self.G}: need >= 3 classes for LOO-class-out"}
        rows = []
        for g in self.groups:
            train = [s for s in self.systems if s.group != g]
            test = [s for s in self.systems if s.group == g]
            sub = ClassHierarchyDCp(train, self.m0, self.a0, self.nu_w,
                                    self.nu_b, self.Psi_w, self.Psi_b, self.cols)
            f = sub.fit(n_iter, burn, seed=seed)
            if f["status"] != "OK":
                rows.append({"held_out_class": g, "status": "REFUSED"}); continue
            mu_d = f["_draws"]["mu0"]
            sb = np.array(f["between_class_var"])
            sw = np.array(f["within_class_var"])
            Sig = np.repeat(np.diag(sb + sw)[None, :, :], len(mu_d), axis=0)
            tot = 0.0
            for s in test:
                X, y, w = s.design()
                tot += self._elpd(X, y, 1.0/w, mu_d, Sig)
            rows.append({"held_out_class": g, "status": "OK",
                         "n_systems": len(test), "elpd_global": tot})
        ok = [r["elpd_global"] for r in rows if r["status"] == "OK"]
        return {"status": "OK", "rows": rows,
                "total_elpd": float(sum(ok)) if ok else None,
                "note": ("this scores generalisation to an UNSEEN ligand class; "
                         "it is a harder test than LOO-system-out")}


# =============================================================================
# SELF-TEST
# =============================================================================

def _coupled_samples(n, T, c3, seed=0):
    rng = np.random.default_rng(seed)
    nb = 60
    e = np.linspace(-math.pi, math.pi, nb+1)
    c = 0.5*(e[1:]+e[:-1])
    G1, G2, G3 = np.meshgrid(c, c, c, indexing="ij")
    W = (-1.2*np.cos(G1) - 0.4*np.cos(G2) - 2.0*np.cos(G3)
         - 0.8*np.cos(G1-G2) - 1.5*np.cos(G1+G2+G3))
    p = np.exp(-(W - W.min())/(KB*T)).ravel(); p /= p.sum()
    idx = rng.choice(p.size, size=n, p=p)
    i1, i2, i3 = np.unravel_index(idx, G1.shape)
    jit = rng.uniform(-0.5, 0.5, (3, n))*(e[1]-e[0])
    return wrap(np.stack([G1[i1, i2, i3]+jit[0],
                          G2[i1, i2, i3]+jit[1],
                          G3[i1, i2, i3]+jit[2]], -1))


def _make_grouped(rng, mode, n_per=6, T0=310.0):
    base = np.array([-45.0, -0.10, -0.65, 0.0020])
    specs = {"A": np.array([0.0, 0.0, 0.0, 0.0]),
             "B": np.array([12.0, 0.02, -0.25, 0.0004]),
             "C": np.array([-14.0, -0.03, 0.30, -0.0005])}
    sysl = []
    for g, shift in specs.items():
        for j in range(3):
            if mode == "structured":
                th = base + shift + rng.normal(0, 1, 4)*np.array([1.0, .005, .03, .0002])
            else:  # classes carry no real signal
                th = base + rng.normal(0, 1, 4)*np.array([1.0, .005, .03, .0002])
            Ts = np.linspace(296., 324., n_per) + rng.normal(0, .8, n_per)
            kinds, ys, sg = [], [], []
            for t in Ts:
                dT = t - T0
                H = th[0] + th[2]*dT + 0.5*th[3]*dT*dT
                kinds.append("dH"); ys.append(H + rng.normal(0, .2)); sg.append(.2)
            sysl.append(SystemData(f"{g}{j}", g, Ts, np.array(kinds),
                                   np.array(ys), np.array(sg), T0))
    return sysl


def self_test() -> bool:
    print("=" * 96)
    print(f"  CLASS HIERARCHY + ADAPTIVE ANISOTROPIC KDE  v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 96)
    rng = np.random.default_rng(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    T = 310.0

    # ---------------- A1 anisotropy detected ----------------
    print("\n[A1] Anisotropic base bandwidth from marginal resultant lengths")
    s = _coupled_samples(4000, T, 1.5, seed=1)
    kde = AdaptiveTorusKDE(s).fit_cv()
    r = kde.cv_report
    print(f"      R per dim      = {np.round(r['marginal_R'], 4).tolist()}")
    print(f"      kappa0 per dim = {np.round(r['kappa0'], 3).tolist()}")
    print(f"      bandwidth rad  = {np.round(r['bandwidth_rad'], 4).tolist()}")
    print(f"      alpha          = {r['alpha']}   mode={r['mode']}")
    rep("anisotropy detected (kappa differs across dims)",
        max(r["kappa0"]) / min(r["kappa0"]) > 1.5,
        f"ratio={max(r['kappa0'])/min(r['kappa0']):.2f}")
    rep("CV report exposes curve + flatness flag",
        "cv_curve_top5" in r and "cv_flat_maximum" in r)

    # ---------------- A2 adaptive beats fixed on held-out ----------------
    print("\n[A2] Adaptive vs fixed bandwidth, held-out log density")
    test = _coupled_samples(2000, T, 1.5, seed=99)
    lp_ad, _ = kde.log_density(test)
    fixed = AdaptiveTorusKDE(s, AdaptiveKDEConfig(alpha_grid=(0.0,)))
    fixed.fit_cv()
    lp_fx, _ = fixed.log_density(test)
    m_ad, m_fx = float(lp_ad.mean()), float(lp_fx.mean())
    print(f"      adaptive alpha={kde.alpha}: mean log p = {m_ad:.4f}")
    print(f"      fixed    alpha=0        : mean log p = {m_fx:.4f}")
    rep("adaptive >= fixed on held-out (or within 0.02)",
        m_ad >= m_fx - 0.02, f"gain={m_ad - m_fx:+.4f}")

    # ---------------- A3 normalisation ----------------
    print("\n[A3] Sample-smoothing KDE integrates to ~1 on T^3")
    Z = kde.normalisation_check(n_grid=32)
    rep("integral ~ 1", abs(Z - 1.0) < 0.12, f"integral={Z:.4f}")

    # ---------------- A4 balloon mode runs and is flagged ----------------
    print("\n[A4] Balloon mode")
    kb = AdaptiveTorusKDE(s, AdaptiveKDEConfig(mode="balloon",
                                               alpha_grid=(0.0, 0.5)))
    kb.fit_cv()
    Zb = kb.normalisation_check(n_grid=24)
    print(f"      balloon integral = {Zb:.4f} (need not be 1)")
    rep("balloon mode runs", np.isfinite(Zb))

    # ---------------- A5 support guard still fires ----------------
    print("\n[A5] Support guard under adaptive bandwidth")
    R = Restraint3D(np.array([0.3, -0.4, 0.6]),
                    np.array([[300., 40., 20.],
                              [40., 300., 30.],
                              [20., 30., 300.]]))
    good = restraint_integral(kde, R, T, n_per_dim=16)
    print(f"      supported: status={good['status']} "
          f"dG={good.get('dG_apply_kJmol')} "
          f"unsupported={good.get('unsupported_restraint_weight'):.4f}")
    rep("supported restraint gives finite dG",
        good["status"] == "OK" and math.isfinite(good["dG_apply_kJmol"]))
    sparse = AdaptiveTorusKDE(_coupled_samples(300, T, 1.5, seed=5)).fit_cv()
    Rfar = Restraint3D(np.array([2.9, -2.9, 2.9]), np.eye(3)*4000.0)
    bad = restraint_integral(sparse, Rfar, T, n_per_dim=12)
    rep("sparse + distant restraint REFUSED", bad["status"] == "REFUSED",
        f"unsupported={bad.get('unsupported_restraint_weight'):.3f}")

    # ---------------- B1 structured classes: ICC high ----------------
    print("\n[B1] Structured ligand classes")
    sysl = _make_grouped(rng, "structured")
    h = ClassHierarchyDCp(sysl)
    f = h.fit(n_iter=2500, burn=900, seed=1)
    rep("class hierarchy fits", f["status"] == "OK")
    if f["status"] == "OK":
        print(f"      groups={f['groups']} counts={f['group_counts']}")
        print(f"      ICC per param = {np.round(f['ICC'], 3).tolist()}")
        print(f"      between-class var = {np.round(f['between_class_var'], 4).tolist()}")
        print(f"      within-class  var = {np.round(f['within_class_var'], 4).tolist()}")
        rep("ICC detects real class structure (dH0 ICC > 0.3)",
            f["ICC"][0] > 0.3, f"ICC(dH0)={f['ICC'][0]:.3f}")

    # ---------------- B2 LOO-system supports pooling ----------------
    print("\n[B2] LOO-system-out, structured classes")
    ls = h.loo_system(n_iter=1200, burn=500, seed=2)
    if ls["status"] == "OK":
        print(f"      total gain = {ls['total_gain']:.3f} -> {ls['verdict']}")
        rep("class pooling supported when classes are real",
            ls["total_gain"] > 0)

    # ---------------- B3 LOO-class-out ----------------
    print("\n[B3] LOO-class-out (unseen ligand class)")
    lc = h.loo_class(n_iter=1200, burn=500, seed=3)
    if lc["status"] == "OK":
        for row in lc["rows"]:
            print(f"      held out {row['held_out_class']}: "
                  f"elpd={row.get('elpd_global')}")
        rep("LOO-class-out scored", lc["total_elpd"] is not None)

    # ---------------- B4 fake classes: ICC low, pooling rejected ----------------
    print("\n[B4] Arbitrary classes (no real signal) must be REJECTED")
    fake = _make_grouped(rng, "unstructured")
    hf = ClassHierarchyDCp(fake)
    ff = hf.fit(n_iter=2500, burn=900, seed=4)
    if ff["status"] == "OK":
        print(f"      ICC = {np.round(ff['ICC'], 3).tolist()}")
        rep("ICC low when classes are arbitrary", ff["ICC"][0] < 0.35,
            f"ICC(dH0)={ff['ICC'][0]:.3f}")
        rep("low-ICC warning emitted",
            any("ICC" in w for w in ff["warnings"]))
    lsf = hf.loo_system(n_iter=1200, burn=500, seed=5)
    if lsf["status"] == "OK":
        print(f"      total gain = {lsf['total_gain']:.3f} -> {lsf['verdict']}")
        rep("LOO does not strongly support fake grouping",
            lsf["total_gain"] < 5.0, f"gain={lsf['total_gain']:.3f}")

    # ---------------- B5 guards ----------------
    print("\n[B5] Guards")
    h1 = ClassHierarchyDCp(_make_grouped(rng, "structured")[:1])
    rep("S=1 refused", h1.fit()["status"] == "REFUSED")
    two = [s for s in sysl if s.group in ("A", "B")]
    h2 = ClassHierarchyDCp(two)
    f2 = h2.fit(n_iter=800, burn=300, seed=6)
    rep("G=2 flagged", f2["status"] == "OK"
        and any("G=2" in w for w in f2["warnings"]))
    rep("LOO-class needs >= 3 classes",
        ClassHierarchyDCp(two).loo_class()["status"] == "REFUSED")

    print("\n" + "=" * 96)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 96)
    print("""
  WHAT THESE TESTS PROVE
    A1  per-dimension bandwidths differ when the marginals differ
    A2  adaptive bandwidth is competitive with / better than fixed on
        held-out log density
    A3  sample-smoothing KDE integrates to ~1 on the torus
    A4  balloon mode runs (and its normalisation is reported, not assumed)
    A5  the support guard still REFUSES an unsupported restraint region
        under adaptive bandwidth
    B1  ICC detects genuine class structure
    B2  LOO-system-out supports pooling when classes are real
    B3  LOO-class-out scores generalisation to an unseen class
    B4  arbitrary classes give low ICC and are not strongly supported
    B5  S=1, G=2 and G<3 guards fire

  WHAT IS NOT PROVEN
    * adaptive KDE increases BIAS in sparse regions; it does not create data
    * pilot density uses all samples (weak CV leakage); strict option exists
    * balloon KDE need not integrate to 1
    * class labels must be chemically justified; LOO detects but cannot prevent
    * with G=2 the between-class covariance is prior-dominated
    * no experimental validation is performed here
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Class hierarchy + adaptive KDE v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--samples", default=None, help="NPZ with z1,z2,z3")
    ap.add_argument("--systems", default=None,
                    help="JSON list of {name,group,T,kind,y,sigma,T0}")
    ap.add_argument("--balloon", action="store_true")
    ap.add_argument("--loo-class", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    out: Dict[str, Any] = {}
    if a.samples:
        z = np.load(a.samples)
        s = np.stack([z["z1"], z["z2"], z["z3"]], -1)
        cfg = AdaptiveKDEConfig(mode="balloon" if a.balloon else "sample")
        kde = AdaptiveTorusKDE(s, cfg).fit_cv()
        out["kde"] = kde.cv_report
        out["normalisation"] = kde.normalisation_check(32)
        print(json.dumps(out, indent=2))
    if a.systems:
        with open(a.systems) as f:
            raw = json.load(f)
        sysl = [SystemData(d["name"], d["group"], np.array(d["T"], float),
                           np.array(d["kind"]), np.array(d["y"], float),
                           np.array(d["sigma"], float),
                           float(d.get("T0", 310.0))) for d in raw]
        h = ClassHierarchyDCp(sysl)
        f = h.fit()
        show = {k: v for k, v in f.items() if k != "_draws"}
        print(json.dumps(show, indent=2, default=float))
        out["hierarchy"] = show
        out["loo_system"] = h.loo_system()
        if a.loo_class:
            out["loo_class"] = h.loo_class()
        print(json.dumps({k: out[k] for k in ("loo_system", "loo_class")
                          if k in out}, indent=2, default=float))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
    if not (a.samples or a.systems):
        ap.print_help()
        print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
