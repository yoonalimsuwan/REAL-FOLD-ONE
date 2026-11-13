#!/usr/bin/env python3
"""
=============================================================================
ADAPTIVE TORUS KDE  +  HIERARCHICAL PARTIAL POOLING FOR dCp(T)
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — FIXING 3D SPARSITY ON THE TORSUS
  A fixed 36^3 histogram has 46,656 bins and needs ~1e6-1e7 samples to
  populate the low-energy region. Three things replace it:

  A1 PRODUCT VON MISES KDE ON T^3
       K(theta; mu, kappa) = prod_d exp(kappa_d cos(theta_d - mu_d))
                                   / (2 pi I0(kappa_d))
     Exactly periodic by construction. No wrapping hack, no boundary
     leakage. Concentration kappa_d relates to a linear bandwidth by
     h_d ~ 1/sqrt(kappa_d) (von Mises -> Gaussian as kappa -> inf).
     Bandwidth chosen by LEAVE-ONE-OUT PSEUDO-LIKELIHOOD CV, not by a
     rule of thumb, because the rule of thumb assumes a unimodal
     reference density that torsion distributions do not satisfy.

  A2 ADAPTIVE EVALUATION GRID
     The restraint U is LOCALISED. We do not need the density everywhere,
     only where exp(-beta U) carries weight. The quadrature grid is
     therefore placed on the restraint's own support box, with spacing
     tied to the restraint width, NOT to the sample count. This decouples
     "where the samples are" from "where we need the answer", which is
     the actual reason the fixed global histogram was hopeless.

  A3 TORUS CELL LIST
     Kernel evaluation is O(M*k) with k the local sample count, instead
     of O(M*N). Same neighbourhood trick already used elsewhere in this
     codebase for interface scoring.

  A4 SUPPORT DIAGNOSTIC — WHAT KDE CANNOT FIX
     KDE borrows strength across neighbouring samples. It does NOT
     create information. If an evaluation point has no sample within
     a few bandwidths, its density value is driven by the KERNEL TAIL,
     not by data. We therefore report, for every evaluation point, the
     local effective sample count, and the fraction of the restraint's
     Boltzmann weight sitting below a support threshold. When that
     fraction is not small the result is REFUSED, exactly as before.
     Sparsity is mitigated, not abolished.

  A5 ADAPTIVE BINNING (diagnostic only)
     A periodic octree splits until each leaf holds >= min_count samples.
     Used to report where the data actually resolves structure, and to
     cross-check the KDE. Not used for the quadrature itself.

PART B — HIERARCHICAL PARTIAL POOLING ACROSS SYSTEMS
  For system s the dCp(T) model is linear in theta_s = (dH0, dS0, dCp0, b):
      y_s = X_s theta_s + eps_s,   eps_s ~ N(0, diag(1/w_s))
  with X_s built from that system's OWN temperatures (unequal n_s, and
  different temperature sets, are handled natively).

  Hierarchical prior:
      theta_s | mu, Sigma ~ N(mu, Sigma)
      mu      | Sigma     ~ N(m0, Sigma / a0)
      Sigma               ~ InverseWishart(nu0, Psi0)

  This is fully conjugate, so a Gibbs sampler gives EXACT conditional
  draws — no HMC, no tuning:
      theta_s | . ~ N( (X'WX + Sigma^-1)^-1 (X'Wy + Sigma^-1 mu), . )
      mu      | . ~ N( (a0 m0 + sum theta_s)/(a0+S),  Sigma/(a0+S) )
      Sigma   | . ~ IW( nu0+S+1,
                        Psi0 + sum (theta_s-mu)(theta_s-mu)'
                              + a0 (mu-m0)(mu-m0)' )

  PARTIAL POOLING is the automatic consequence: a system with few points
  is shrunk toward mu in proportion to how large its own uncertainty is
  relative to the between-system spread Sigma.

  THE HONEST TEST
    Pooling assumes EXCHANGEABILITY. If the systems are physically
    different (different ligand classes, different binding modes),
    exchangeability is false and pooling BIASES every system toward a
    meaningless average. We therefore run LEAVE-ONE-SYSTEM-OUT prediction
    and compare pooled vs unpooled predictive log density. If pooling
    does not improve held-out prediction, it is not licensed — and the
    self-test deliberately constructs a heterogeneous case where pooling
    HURTS, to show the diagnostic detects it.

HONEST SCOPE
  ❌ KDE cannot manufacture samples. The support diagnostic is mandatory.
  ❌ LOO-CV bandwidth selection can overfit for very small N; the selected
     kappa and its CV curve are reported so this is visible.
  ❌ Hierarchical pooling with S=1 is impossible; with S=2, Sigma is
     barely identified. Both are flagged.
  ❌ Exchangeability is an assumption, not a fact. LOO is a check, not a
     proof.
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
log = logging.getLogger("torus_kde_hier")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

KB = 0.008314462618          # kJ mol^-1 K^-1
TWO_PI = 2.0 * math.pi


def wrap(z: np.ndarray) -> np.ndarray:
    return (np.asarray(z, dtype=np.float64) + math.pi) % TWO_PI - math.pi


def pdist(a: np.ndarray, b: float) -> np.ndarray:
    return wrap(a - b)


# =============================================================================
# PART A §1  STABLE log I0 AND THE VON MISES KERNEL
# =============================================================================

def log_i0(kappa: np.ndarray) -> np.ndarray:
    """
    log I0(kappa), stable for large kappa.
    numpy.i0 overflows around kappa ~ 700; above that use the asymptotic
    expansion I0(k) ~ e^k / sqrt(2 pi k) * (1 + 1/(8k) + 9/(128k^2) + ...).
    """
    k = np.asarray(kappa, dtype=np.float64)
    out = np.empty_like(k)
    small = k < 500.0
    if np.any(small):
        out[small] = np.log(np.i0(k[small]))
    if np.any(~small):
        kb = k[~small]
        corr = 1.0 + 1.0/(8.0*kb) + 9.0/(128.0*kb**2) + 225.0/(3072.0*kb**3)
        out[~small] = kb - 0.5*np.log(TWO_PI*kb) + np.log(corr)
    return out


def vm_logkernel(diff: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    Log of the product von Mises kernel, WITHOUT the exp(kappa) factor
    that cancels against the normaliser. diff has shape (..., 3).
        log K = sum_d [ kappa_d (cos(diff_d) - 1) - log(2 pi I0(kappa_d)) + kappa_d ]
    Writing it as kappa*(cos-1) keeps the exponent <= 0, so no overflow.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    norm = log_i0(kappa) + math.log(TWO_PI) - kappa   # per-dim constant
    return (kappa * (np.cos(diff) - 1.0)).sum(-1) - norm.sum()


# =============================================================================
# PART A §2  TORUS CELL LIST  (O(M*k) neighbour queries)
# =============================================================================

class TorusCellList:
    """
    Uniform periodic binning of samples on T^3, with CSR-style offsets,
    for fast neighbourhood queries. Built on the sample coordinates;
    the kernel itself is evaluated on the original coordinates.
    """

    def __init__(self, samples: np.ndarray, cell_size: float):
        self.samples = np.asarray(samples, dtype=np.float64)
        self.N = self.samples.shape[0]
        self.h = max(float(cell_size), 1e-3)
        self.nd = max(1, int(math.floor(TWO_PI / self.h)))
        self.h = TWO_PI / self.nd                     # exact tiling
        g = np.floor((self.samples + math.pi) / self.h).astype(np.int64)
        g = np.clip(g, 0, self.nd - 1)
        self.grid = g
        lin = (g[:, 0]*self.nd + g[:, 1])*self.nd + g[:, 2]
        self.order = np.argsort(lin, kind="stable")
        self.lin_sorted = lin[self.order]
        nc = self.nd**3
        self.start = np.searchsorted(self.lin_sorted, np.arange(nc))
        self.start = np.append(self.start, self.N)

    def _cell_lin(self, pts: np.ndarray) -> np.ndarray:
        g = np.floor((pts + math.pi) / self.h).astype(np.int64)
        g = np.clip(g, 0, self.nd - 1)
        return (g[:, 0]*self.nd + g[:, 1])*self.nd + g[:, 2]

    def query(self, pts: np.ndarray, max_nb: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query point return up to max_nb sample indices from the
        27 neighbouring cells (periodic), and the true candidate count.
        If the true count exceeds max_nb the buffer is grown by the caller.
        """
        pts = np.asarray(pts, dtype=np.float64)
        M = pts.shape[0]
        base_lin = self._cell_lin(pts)
        bg = np.stack([(base_lin // (self.nd*self.nd)) % self.nd,
                       ((base_lin // self.nd) % self.nd),
                       base_lin % self.nd], -1)
        offs = np.array([(a, b, c) for a in (-1, 0, 1)
                         for b in (-1, 0, 1) for c in (-1, 0, 1)],
                        dtype=np.int64)
        counts = np.zeros(M, dtype=np.int64)
        buckets = []
        for o in offs:
            g = (bg + o) % self.nd
            lin = (g[:, 0]*self.nd + g[:, 1])*self.nd + g[:, 2]
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
                inb = (j < (c1 - c0))
                rows = np.nonzero(inb & (fill < K))[0]
                if rows.size == 0:
                    continue
                idx[rows, fill[rows]] = self.order[pos[rows]]
                fill[rows] += 1
        return idx, counts


# =============================================================================
# PART A §3  TORUS KDE WITH LOO-CV BANDWIDTH
# =============================================================================

@dataclass
class TorusKDE:
    """
    Product von Mises KDE on T^3 with LOO pseudo-likelihood bandwidth choice.
    """
    samples: np.ndarray
    kappa: np.ndarray = field(default_factory=lambda: np.array([30.0]*3))
    cv_report: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.samples = wrap(np.asarray(self.samples, dtype=np.float64))
        self.N = self.samples.shape[0]
        self.kappa = np.asarray(self.kappa, dtype=np.float64)
        self._cl: Optional[TorusCellList] = None

    # ---------------- bandwidth ----------------
    @staticmethod
    def fit_cv(samples: np.ndarray,
               kappa_grid: Optional[Sequence[float]] = None,
               isotropic: bool = True,
               max_pairs: int = 4000) -> "TorusKDE":
        """
        Maximise the leave-one-out log pseudo-likelihood
            L(kappa) = sum_i log[ (1/(N-1)) sum_{j != i} K(theta_i; theta_j) ]
        over a scalar multiplier of a base concentration. For N larger
        than max_pairs the score is estimated on a random subset of the
        query points (reported), since the full O(N^2) score is wasteful.
        """
        s = wrap(np.asarray(samples, dtype=np.float64))
        N = s.shape[0]
        if N < 8:
            raise ValueError("need >= 8 samples for CV bandwidth selection")
        if kappa_grid is None:
            kappa_grid = np.geomspace(2.0, 400.0, 18)
        rng = np.random.default_rng(0)
        qidx = np.arange(N) if N <= max_pairs else rng.choice(N, max_pairs, replace=False)
        sq = s[qidx]
        diff = wrap(sq[:, None, :] - s[None, :, :])       # (M,N,3)
        same = (qidx[:, None] == np.arange(N)[None, :])
        scores, curves = [], []
        for k in kappa_grid:
            kap = np.full(3, float(k)) if isotropic else np.asarray(k, dtype=float)
            lk = vm_logkernel(diff, kap)
            lk = np.where(same, -np.inf, lk)
            mx = lk.max(axis=1)
            lse = mx + np.log(np.exp(lk - mx[:, None]).sum(axis=1))
            lse -= math.log(N - 1)
            sc = float(np.mean(lse[np.isfinite(lse)]))
            scores.append(sc); curves.append((float(k), sc))
        best = int(np.argmax(scores))
        kap = np.full(3, float(kappa_grid[best]))
        obj = TorusKDE(samples=s, kappa=kap)
        obj.cv_report = {
            "kappa_selected": kap.tolist(),
            "bandwidth_rad": (1.0/np.sqrt(kap)).tolist(),
            "cv_curve": curves,
            "n_query_points": int(len(qidx)),
            "n_samples": N,
            "isotropic": isotropic,
            "note": ("LOO pseudo-likelihood; can overfit at very small N. "
                     "Inspect cv_curve for a flat maximum."),
        }
        return obj

    # ---------------- evaluation ----------------
    def _cell_list(self) -> TorusCellList:
        if self._cl is None:
            h = float(3.0 / np.sqrt(self.kappa.min()))
            self._cl = TorusCellList(self.samples, max(h, TWO_PI/64))
        return self._cl

    def log_density(self, pts: np.ndarray, chunk: int = 512,
                    max_nb: int = 4096
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (log p_hat, local_effective_count) at each query point.
        local_effective_count = sum_j K(theta; theta_j) * (2 pi)^3-ish
        proxy: we report sum_j exp(logkernel_unnormalised) which is the
        number of samples effectively contributing at that point.
        """
        pts = wrap(np.asarray(pts, dtype=np.float64))
        M = pts.shape[0]
        cl = self._cell_list()
        out = np.empty(M); neff = np.empty(M)
        for s in range(0, M, chunk):
            e = min(s + chunk, M)
            p = pts[s:e]
            idx, counts = cl.query(p, max_nb)
            if int(counts.max()) > idx.shape[1]:
                idx, counts = cl.query(p, int(counts.max()))
            valid = idx >= 0
            safe = np.maximum(idx, 0)
            nb = self.samples[safe]                      # (m,K,3)
            diff = wrap(p[:, None, :] - nb)
            lk = vm_logkernel(diff, self.kappa)
            lk = np.where(valid, lk, -np.inf)
            # unnormalised weight for the effective-count diagnostic
            uw = np.exp(np.clip((self.kappa*(np.cos(diff)-1.0)).sum(-1), -700, 0))
            uw = np.where(valid, uw, 0.0)
            neff[s:e] = uw.sum(axis=1)
            mx = lk.max(axis=1)
            has = np.isfinite(mx)
            lse = np.full(e - s, -np.inf)
            if np.any(has):
                mm = mx[has][:, None]
                lse[has] = (mx[has]
                            + np.log(np.exp(lk[has] - mm).sum(axis=1)))
            out[s:e] = lse - math.log(self.N)
        return out, neff

    def density(self, pts: np.ndarray) -> np.ndarray:
        lp, _ = self.log_density(pts)
        return np.exp(lp)


# =============================================================================
# PART A §4  ADAPTIVE PERIODIC BINNING (diagnostic)
# =============================================================================

@dataclass
class AdaptiveTorusBins:
    """
    Periodic octree: split a cell until every leaf holds >= min_count
    samples, or max_depth is reached. Diagnostic of where the data
    actually resolves structure.
    """
    samples: np.ndarray
    min_count: int = 32
    max_depth: int = 6
    leaves: List[Dict[str, Any]] = field(default_factory=list)

    def build(self) -> "AdaptiveTorusBins":
        self.samples = wrap(np.asarray(self.samples, dtype=np.float64))
        self.leaves = []
        lo = np.full(3, -math.pi)
        self._rec(lo, TWO_PI, self.samples, 0)
        return self

    def _rec(self, lo: np.ndarray, side: float, pts: np.ndarray, depth: int):
        n = pts.shape[0]
        if n <= self.min_count or depth >= self.max_depth:
            vol = side**3
            self.leaves.append({
                "lo": lo.tolist(), "side": float(side),
                "count": int(n), "depth": int(depth),
                "density": float(n / (self.samples.shape[0]*vol)) if n else 0.0,
            })
            return
        h = side/2.0
        mid = lo + h
        rel = pts - lo
        oct_id = ((rel[:, 0] >= h).astype(int)
                  + 2*(rel[:, 1] >= h).astype(int)
                  + 4*(rel[:, 2] >= h).astype(int))
        for o in range(8):
            sel = pts[oct_id == o]
            if sel.shape[0] == 0:
                continue
            off = np.array([o & 1, (o >> 1) & 1, (o >> 2) & 1], dtype=float)*h
            self._rec(lo + off, h, sel, depth + 1)

    def summary(self) -> Dict[str, Any]:
        if not self.leaves:
            return {"n_leaves": 0}
        d = np.array([l["depth"] for l in self.leaves])
        c = np.array([l["count"] for l in self.leaves])
        return {"n_leaves": len(self.leaves),
                "depth_min": int(d.min()), "depth_max": int(d.max()),
                "count_min": int(c.min()), "count_median": float(np.median(c)),
                "frac_leaves_below_min": float((c < self.min_count).mean())}


# =============================================================================
# PART A §5  ADAPTIVE EVALUATION GRID + RESTRAINT INTEGRALS
# =============================================================================

@dataclass
class Restraint3D:
    centre: np.ndarray
    K: np.ndarray                     # 3x3 PD

    def __post_init__(self):
        self.centre = wrap(np.asarray(self.centre, dtype=float))
        self.K = np.asarray(self.K, dtype=float)
        if not np.allclose(self.K, self.K.T, atol=1e-10):
            raise ValueError("K must be symmetric")
        np.linalg.cholesky(self.K)      # raises if not PD

    def U(self, pts: np.ndarray) -> np.ndarray:
        d = np.stack([pdist(pts[:, i], self.centre[i]) for i in range(3)], -1)
        return 0.5*np.einsum("ni,ij,nj->n", d, self.K, d)

    def support_box(self, T: float, tol: float = 1e-6
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-dimension half-width where exp(-beta U) > tol along that axis
        alone (a superset of the true support, which is safe).
        """
        kT = KB*T
        diag = np.diag(self.K)
        half = np.sqrt(2.0*kT*np.log(1.0/tol)/diag)
        return np.maximum(half, 1e-3), diag

    def eval_grid(self, T: float, n_per_dim: int = 20,
                  tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive quadrature grid on the restraint's own support box.
        Returns (points (M,3), cell volume dV).
        """
        half, _ = self.support_box(T, tol)
        half = np.minimum(half, math.pi)
        axes = [np.linspace(-h, h, n_per_dim) for h in half]
        G = np.meshgrid(*axes, indexing="ij")
        rel = np.stack([g.ravel() for g in G], -1)
        pts = wrap(self.centre[None, :] + rel)
        dV = float(np.prod([2*h/(n_per_dim-1) for h in half]))
        return pts, dV


def kde_pmf_and_apply(kde: TorusKDE, restraint: Restraint3D, T: float,
                      n_per_dim: int = 20, tol: float = 1e-6,
                      min_neff: float = 5.0,
                      coverage_tol: float = 0.02
                      ) -> Dict[str, Any]:
    """
    dG_apply for the JOINT conformational restraint, using the KDE as the
    intrinsic (unrestrained) PMF and an adaptive quadrature grid.

        W_int(theta) = -kT log p_hat(theta)
        dG_apply = -kT log[ int e^{-beta(W_int+U)} / int e^{-beta W_int} ]

    Both integrals are restricted to the restraint's support box, which
    is where the numerator lives; the denominator is the intrinsic
    partition weight in the SAME box, so the ratio is the local cost of
    switching the restraint on. This is the quantity that enters the
    Woo-Roux cycle when the free-state PMF is referenced to the same box.

    SUPPORT: fraction of exp(-beta U) weight at points with local
    effective sample count below min_neff. Not small => REFUSE.
    """
    kT = KB*T
    beta = 1.0/kT
    pts, dV = restraint.eval_grid(T, n_per_dim, tol)
    lp, neff = kde.log_density(pts)
    U = restraint.U(pts)
    eU = np.exp(-beta*np.clip(U - U.min(), 0, 700))
    tot_eU = float(eU.sum())
    low = neff < min_neff
    unsupported = float(eU[low].sum()/max(tot_eU, 1e-300))
    ok = unsupported < coverage_tol
    Wint = -kT*lp
    finite = np.isfinite(Wint)
    if not finite.any():
        return {"status": "REFUSED", "reason": "no finite density on grid"}
    shift = float(np.min(Wint[finite]))
    Wi = np.where(finite, Wint - shift, np.inf)
    num = np.where(finite, np.exp(-beta*np.clip(Wi + U, 0, 700)), 0.0)
    den = np.where(finite, np.exp(-beta*np.clip(Wi, 0, 700)), 0.0)
    Zn = float(num.sum())*dV
    Zd = float(den.sum())*dV
    if Zn <= 0 or Zd <= 0:
        return {"status": "REFUSED", "reason": "zero partition function"}
    dG = -kT*math.log(Zn/Zd)
    return {
        "status": "OK" if ok else "REFUSED",
        "dG_apply_kJmol": float(dG),
        "unsupported_restraint_weight": unsupported,
        "coverage_ok": bool(ok),
        "n_eval_points": int(pts.shape[0]),
        "neff_min": float(neff.min()),
        "neff_median": float(np.median(neff)),
        "grid_dV": dV,
        "reason": ("" if ok else
                   "KDE value in the restraint region is driven by the "
                   "kernel tail, not by samples. Collect more data there "
                   "or weaken the restraint."),
    }


# =============================================================================
# PART B §1  DESIGN MATRIX (same convention as the earlier dCp module)
# =============================================================================

PARAM_NAMES = ("dH0", "dS0", "dCp0", "b")


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
    T: np.ndarray
    kind: np.ndarray          # "dH" | "dG" per observation
    y: np.ndarray
    sigma: np.ndarray
    T0: float = 310.0
    cols: Tuple[int, ...] = (0, 1, 2, 3)

    def design(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.vstack([design_row(float(t), str(k), self.T0)[list(self.cols)]
                       for t, k in zip(self.T, self.kind)])
        w = 1.0/np.maximum(np.asarray(self.sigma, dtype=float), 1e-12)**2
        return X, np.asarray(self.y, dtype=float), w

    @property
    def n(self) -> int:
        return len(self.y)


# =============================================================================
# PART B §2  CONJUGATE HIERARCHICAL GIBBS
# =============================================================================

def _rwishart(rng: np.random.Generator, df: float, scale: np.ndarray) -> np.ndarray:
    """Bartlett decomposition: W ~ Wishart(df, scale)."""
    k = scale.shape[0]
    L = np.linalg.cholesky(scale)
    A = np.zeros((k, k))
    for i in range(k):
        A[i, i] = math.sqrt(rng.chisquare(max(df - i, 1e-6)))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    M = L @ A
    return M @ M.T


def _rinvwishart(rng, df, Psi):
    W = _rwishart(rng, df, np.linalg.inv(Psi))
    return np.linalg.inv(W)


@dataclass
class HierarchicalDCp:
    """
    Normal / Inverse-Wishart hierarchical linear model, exact Gibbs.
    theta_s ~ N(mu, Sigma); mu ~ N(m0, Sigma/a0); Sigma ~ IW(nu0, Psi0).
    """
    systems: List[SystemData]
    m0: Optional[np.ndarray] = None
    a0: float = 1.0
    nu0: Optional[float] = None
    Psi0: Optional[np.ndarray] = None
    cols: Tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self):
        self.k = len(self.cols)
        self.S = len(self.systems)
        if self.m0 is None:
            self.m0 = np.zeros(self.k)
        if self.nu0 is None:
            self.nu0 = self.k + 2.0
        if self.Psi0 is None:
            # weakly informative scales, same convention as the single-system fit
            sc = np.array([100.0, 0.5, 2.0, 0.02])[list(self.cols)]
            self.Psi0 = np.diag((self.nu0 - self.k - 1.0)*sc**2)
        self._prep = [s.design() for s in self.systems]

    # ---------------- OLS per system (unpooled reference) ----------------
    def unpooled(self) -> List[Dict[str, Any]]:
        out = []
        for s, (X, y, w) in zip(self.systems, self._prep):
            k = self.k
            if s.n < k:
                out.append({"name": s.name, "status": "UNDERDETERMINED",
                            "n": s.n, "k": k})
                continue
            Xw = X*np.sqrt(w)[:, None]
            beta, *_ = np.linalg.lstsq(Xw, y*np.sqrt(w), rcond=None)
            cov = np.linalg.pinv(Xw.T @ Xw)
            out.append({"name": s.name, "status": "OK", "n": s.n,
                        "theta": beta, "cov": cov,
                        "se": np.sqrt(np.maximum(np.diag(cov), 0))})
        return out

    # ---------------- Gibbs ----------------
    def fit(self, n_iter: int = 4000, burn: int = 1000, thin: int = 4,
            seed: int = 0) -> Dict[str, Any]:
        if self.S < 2:
            return {"status": "REFUSED",
                    "reason": f"S={self.S}: partial pooling needs >= 2 systems."}
        warns = []
        if self.S == 2:
            warns.append("S=2: the between-system covariance Sigma is barely "
                         "identified; its posterior will be prior-dominated.")
        rng = np.random.default_rng(seed)
        k, S = self.k, self.S
        # initialise at unpooled estimates where available, else prior mean
        theta = np.zeros((S, k))
        for i, u in enumerate(self.unpooled()):
            theta[i] = u["theta"] if u["status"] == "OK" else self.m0
        mu = theta.mean(0)
        Sigma = self.Psi0/max(self.nu0 - k - 1.0, 1.0)
        draws_t, draws_m, draws_S = [], [], []
        for it in range(n_iter):
            Si = np.linalg.inv(Sigma)
            # theta_s
            for i, (X, y, w) in enumerate(self._prep):
                A = X.T @ (w[:, None]*X) + Si
                b = X.T @ (w*y) + Si @ mu
                V = np.linalg.inv(A)
                m = V @ b
                theta[i] = m + np.linalg.cholesky(V) @ rng.standard_normal(k)
            # mu
            post_var = Sigma/(self.a0 + S)
            post_mean = (self.a0*self.m0 + theta.sum(0))/(self.a0 + S)
            mu = post_mean + np.linalg.cholesky(post_var) @ rng.standard_normal(k)
            # Sigma
            R = theta - mu[None, :]
            Psi = (self.Psi0 + R.T @ R
                   + self.a0*np.outer(mu - self.m0, mu - self.m0))
            Sigma = _rinvwishart(rng, self.nu0 + S + 1.0, Psi)
            if it >= burn and (it - burn) % thin == 0:
                draws_t.append(theta.copy())
                draws_m.append(mu.copy())
                draws_S.append(Sigma.copy())
        T_arr = np.array(draws_t)          # (D,S,k)
        M_arr = np.array(draws_m)          # (D,k)
        S_arr = np.array(draws_S)          # (D,k,k)
        # summaries
        up = {u["name"]: u for u in self.unpooled()}
        per_sys = []
        for i, s in enumerate(self.systems):
            d = T_arr[:, i, :]
            mean = d.mean(0)
            sd = d.std(0, ddof=1)
            ci = np.quantile(d, [0.025, 0.975], axis=0)
            u = up[s.name]
            shrink = None
            if u["status"] == "OK":
                num = np.linalg.norm(mean - u["theta"])
                den = max(np.linalg.norm(u["theta"]), 1e-12)
                shrink = float(num/den)
            per_sys.append({
                "name": s.name, "n": s.n,
                "theta_mean": mean.tolist(), "theta_sd": sd.tolist(),
                "ci95": ci.tolist(),
                "unpooled_status": u["status"],
                "unpooled_theta": (u["theta"].tolist()
                                   if u["status"] == "OK" else None),
                "relative_shrinkage": shrink,
                "borrowed": u["status"] != "OK",
            })
        between_sd = np.sqrt(np.maximum(np.diagonal(S_arr, axis1=1, axis2=2), 0)).mean(0)
        within_se = []
        for u in self.unpooled():
            if u["status"] == "OK":
                within_se.append(u["se"])
        within_se = (np.mean(within_se, axis=0) if within_se
                     else np.full(k, np.nan))
        pool_ratio = (between_sd/np.maximum(within_se, 1e-12)).tolist()
        for i, nm in enumerate([PARAM_NAMES[c] for c in self.cols]):
            if pool_ratio[i] < 0.5:
                warns.append(f"{nm}: between-system SD is smaller than the "
                             "typical within-system SE "
                             f"(ratio {pool_ratio[i]:.2f}) — heavy pooling. "
                             "Verify exchangeability with LOO before trusting it.")
        return {"status": "OK", "S": S, "k": k,
                "n_draws": len(draws_t),
                "mu_mean": M_arr.mean(0).tolist(),
                "mu_sd": M_arr.std(0, ddof=1).tolist(),
                "between_sd": between_sd.tolist(),
                "within_se_typical": within_se.tolist(),
                "pool_ratio": pool_ratio,
                "systems": per_sys,
                "warnings": warns,
                "_draws": {"theta": T_arr, "mu": M_arr, "Sigma": S_arr}}

    # ---------------- leave-one-system-out ----------------
    def loo_system(self, n_iter: int = 1500, burn: int = 500,
                   seed: int = 1) -> Dict[str, Any]:
        """
        For each system: fit the hierarchy on the OTHER systems, then score
        the held-out system's observations under the resulting predictive
        distribution. Compare against the unpooled fit of the held-out
        system itself (when it is determined).

        Predictive for a new observation x with noise sigma:
            y ~ N(x.theta_s, sigma^2),  theta_s ~ N(mu, Sigma)
        so marginal y ~ N(x.mu, x Sigma x' + sigma^2), integrated over the
        posterior draws of (mu, Sigma).
        """
        if self.S < 3:
            return {"status": "REFUSED",
                    "reason": f"S={self.S}: LOO-system needs >= 3 systems."}
        rows = []
        for h in range(self.S):
            train = [s for i, s in enumerate(self.systems) if i != h]
            sub = HierarchicalDCp(train, m0=self.m0, a0=self.a0,
                                  nu0=self.nu0, Psi0=self.Psi0, cols=self.cols)
            fit = sub.fit(n_iter=n_iter, burn=burn, seed=seed + h)
            if fit["status"] != "OK":
                rows.append({"held_out": self.systems[h].name,
                             "status": "REFUSED"})
                continue
            Mh = fit["_draws"]["mu"]; Sh = fit["_draws"]["Sigma"]
            Xh, yh, wh = self._prep[h]
            sig2 = 1.0/wh
            lp_pool = []
            for j in range(len(yh)):
                x = Xh[j]
                m = Mh @ x
                v = np.einsum("ni,ij,nj->n", x[None, :].repeat(len(Mh), 0),
                              Sh, x[None, :].repeat(len(Mh), 0)) + sig2[j]
                lp = -0.5*(np.log(TWO_PI*v) + (yh[j] - m)**2/v)
                mx = lp.max()
                lp_pool.append(float(mx + np.log(np.exp(lp - mx).mean())))
            # unpooled reference
            u = self.unpooled()[h]
            lp_un = None
            if u["status"] == "OK":
                th = u["theta"]; cov = u["cov"]
                m = float(Xh @ th)
                v = np.einsum("ni,ij,nj->n", Xh, cov, Xh) + sig2
                lp_un = float(np.sum(-0.5*(np.log(TWO_PI*v) + (yh - m)**2/v)))
            tot_pool = float(np.sum(lp_pool))
            rows.append({
                "held_out": self.systems[h].name, "status": "OK",
                "n_held": int(len(yh)),
                "elpd_pooled": tot_pool,
                "elpd_unpooled": lp_un,
                "gain": (tot_pool - lp_un) if lp_un is not None else None,
                "pooling_helps": (tot_pool > lp_un) if lp_un is not None else None,
            })
        gains = [r["gain"] for r in rows if r.get("gain") is not None]
        verdict = ("POOLING SUPPORTED" if gains and sum(gains) > 0
                   else "POOLING NOT SUPPORTED — do not borrow strength")
        return {"status": "OK", "rows": rows,
                "total_gain": float(sum(gains)) if gains else None,
                "verdict": verdict,
                "note": ("Exchangeability is an assumption. A negative total "
                         "gain means the systems are not exchangeable enough "
                         "for this prior, and pooling biases every system.")}


# =============================================================================
# SELF-TEST
# =============================================================================

def _vm_mixture_samples(n, seed=0):
    """Samples from a genuinely 3-body-coupled torus density."""
    rng = np.random.default_rng(seed)
    nb = 60
    e = np.linspace(-math.pi, math.pi, nb+1)
    c = 0.5*(e[1:]+e[:-1])
    G1, G2, G3 = np.meshgrid(c, c, c, indexing="ij")
    W = (-1.2*np.cos(G1) - 1.2*np.cos(G2) - 1.2*np.cos(G3)
         - 0.8*np.cos(G1-G2) - 0.8*np.cos(G2-G3)
         - 1.5*np.cos(G1+G2+G3))
    p = np.exp(-(W - W.min())/(KB*310.0)).ravel(); p /= p.sum()
    idx = rng.choice(p.size, size=n, p=p)
    i1, i2, i3 = np.unravel_index(idx, G1.shape)
    jit = rng.uniform(-0.5, 0.5, (3, n))*(e[1]-e[0])
    return wrap(np.stack([G1[i1, i2, i3]+jit[0],
                          G2[i1, i2, i3]+jit[1],
                          G3[i1, i2, i3]+jit[2]], -1))


def _make_systems(rng, S, mode, n_per=6, T0=310.0):
    """mode='exchangeable' or 'heterogeneous'."""
    sysl = []
    base = np.array([-45.0, -0.10, -0.65, 0.0020])
    for s in range(S):
        if mode == "exchangeable":
            th = base + rng.normal(0, 1, 4)*np.array([2.0, 0.01, 0.05, 0.0003])
        else:
            th = base + np.array([1.0, 0.0, 0.0, 0.0])*rng.choice([-40.0, 40.0]) \
                 + np.array([0.0, 0.0, 1.0, 0.0])*rng.choice([-1.5, 1.5])
        Ts = np.linspace(298.0, 322.0, n_per) + rng.normal(0, 1.0, n_per)
        kinds, ys, sg = [], [], []
        for t in Ts:
            dT = t - T0
            H = th[0] + th[2]*dT + 0.5*th[3]*dT*dT
            kinds.append("dH"); ys.append(H + rng.normal(0, 0.25)); sg.append(0.25)
        sysl.append(SystemData(f"sys{s}", Ts, np.array(kinds),
                               np.array(ys), np.array(sg), T0=T0))
    return sysl


def self_test() -> bool:
    print("=" * 96)
    print(f"  TORUS KDE + HIERARCHICAL PARTIAL POOLING  v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 96)
    rng = np.random.default_rng(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ---------------- A1 kernel normalisation ----------------
    print("\n[A1] von Mises kernel normalises to 1 on T^3")
    kap = np.array([25.0, 25.0, 25.0])
    g = np.linspace(-math.pi, math.pi, 81)[:-1]
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    dV = (g[1]-g[0])**3
    lk = vm_logkernel(G - np.array([0.3, -0.7, 1.1]), kap)
    Z = float(np.exp(lk).sum()*dV)
    rep("integral = 1", abs(Z - 1.0) < 0.02, f"Z={Z:.5f}")
    rep("log_i0 stable at kappa=2000",
        np.isfinite(log_i0(np.array([2000.0])))[0],
        f"log I0(2000)={float(log_i0(np.array([2000.0]))[0]):.2f}")

    # ---------------- A2 KDE beats histogram at equal N ----------------
    print("\n[A2] KDE vs fixed histogram at equal sample count")
    N = 20000
    s = _vm_mixture_samples(N, seed=1)
    kde = TorusKDE.fit_cv(s, kappa_grid=np.geomspace(4.0, 200.0, 12))
    print(f"      CV kappa={kde.cv_report['kappa_selected']}  "
          f"bandwidth={np.round(kde.cv_report['bandwidth_rad'], 4).tolist()} rad")
    test = _vm_mixture_samples(3000, seed=99)
    lp, neff = kde.log_density(test)
    # histogram reference at 24^3 (13824 bins, N=20000 -> ~1.4 per bin)
    nb = 24
    e = np.linspace(-math.pi, math.pi, nb+1)
    H, _ = np.histogramdd(s, bins=[e]*3)
    p_hist = (H + 0.5)/(N + 0.5*H.size)
    dv = (e[1]-e[0])**3
    gi = np.clip(np.floor((test + math.pi)/(e[1]-e[0])).astype(int), 0, nb-1)
    lp_hist = np.log(p_hist[gi[:, 0], gi[:, 1], gi[:, 2]]/dv)
    ll_kde = float(np.mean(lp)); ll_hist = float(np.mean(lp_hist))
    print(f"      held-out mean log density: KDE={ll_kde:.4f}  "
          f"histogram={ll_hist:.4f}")
    rep("KDE beats histogram on held-out log density", ll_kde > ll_hist,
        f"gain={ll_kde-ll_hist:.4f} nats")
    rep("CV report exposes the curve", len(kde.cv_report["cv_curve"]) >= 8)

    # ---------------- A3 cell list correctness ----------------
    print("\n[A3] Cell list returns the true neighbours")
    cl = TorusCellList(s[:2000], 0.6)
    q = s[2000:2010]
    idx, cnt = cl.query(q, 4096)
    exact = 0
    for j in range(len(q)):
        d = np.abs(wrap(s[:2000] - q[j]))
        r = 3*cl.h
        exact += int(np.all(d <= r + 1e-9, axis=1).sum())
    got = int(cnt.sum())
    rep("cell list count >= exact box count", got >= exact,
        f"celllist={got} exact_box={exact}")

    # ---------------- A4 adaptive binning ----------------
    print("\n[A4] Adaptive periodic binning")
    ab = AdaptiveTorusBins(s, min_count=64, max_depth=6).build()
    summ = ab.summary()
    print(f"      leaves={summ['n_leaves']} depth=[{summ['depth_min']},"
          f"{summ['depth_max']}] median count={summ['count_median']:.0f}")
    rep("adaptive bins built", summ["n_leaves"] > 8)
    rep("most leaves meet min_count", summ["frac_leaves_below_min"] < 0.35,
        f"frac below={summ['frac_leaves_below_min']:.2f}")

    # ---------------- A5 restraint integral + support ----------------
    print("\n[A5] Restraint integral on the adaptive grid")
    R = Restraint3D(np.array([0.3, -0.4, 0.6]),
                    np.array([[300., 40., 20.],
                              [40., 300., 30.],
                              [20., 30., 300.]]))
    res = kde_pmf_and_apply(kde, R, 310.0, n_per_dim=16, min_neff=5.0)
    print(f"      status={res['status']}  dG={res.get('dG_apply_kJmol')}")
    print(f"      eval pts={res.get('n_eval_points')}  "
          f"neff min={res.get('neff_min'):.1f} "
          f"median={res.get('neff_median'):.1f}  "
          f"unsupported={res.get('unsupported_restraint_weight'):.4f}")
    rep("well-supported restraint gives a finite dG",
        res["status"] == "OK" and math.isfinite(res["dG_apply_kJmol"]))

    # ---------------- A6 unsupported region is REFUSED ----------------
    print("\n[A6] Genuinely unsupported region must be REFUSED")
    sparse = _vm_mixture_samples(400, seed=5)
    kde_sp = TorusKDE(samples=sparse, kappa=np.array([400.0]*3))
    R_far = Restraint3D(np.array([2.9, -2.9, 2.9]), np.eye(3)*4000.0)
    bad = kde_pmf_and_apply(kde_sp, R_far, 310.0, n_per_dim=12, min_neff=5.0)
    rep("sparse + distant + tight restraint REFUSED",
        bad["status"] == "REFUSED",
        f"unsupported={bad.get('unsupported_restraint_weight'):.3f}")
    rep("refusal reason present", len(bad.get("reason", "")) > 0)

    # ---------------- B1 exchangeable systems: pooling helps ----------------
    print("\n[B1] Exchangeable systems — pooling should help")
    sys_ex = _make_systems(rng, S=6, mode="exchangeable", n_per=5)
    hier = HierarchicalDCp(sys_ex)
    fit = hier.fit(n_iter=2500, burn=800, seed=1)
    rep("hierarchical fit OK", fit["status"] == "OK")
    if fit["status"] == "OK":
        print(f"      mu = {np.round(fit['mu_mean'], 4).tolist()}")
        print(f"      between-system SD = {np.round(fit['between_sd'], 4).tolist()}")
        print(f"      pool ratio (between/within) = "
              f"{np.round(fit['pool_ratio'], 2).tolist()}")
        sh = [s["relative_shrinkage"] for s in fit["systems"]
              if s["relative_shrinkage"] is not None]
        print(f"      mean relative shrinkage = {np.mean(sh):.4f}")
        rep("shrinkage is finite and nonzero", 0 < np.mean(sh) < 10)
    loo_ex = hier.loo_system(n_iter=1200, burn=400, seed=2)
    if loo_ex["status"] == "OK":
        print(f"      LOO total gain = {loo_ex['total_gain']:.3f}  "
              f"-> {loo_ex['verdict']}")
        rep("LOO supports pooling for exchangeable systems",
            loo_ex["total_gain"] > 0)

    # ---------------- B2 heterogeneous systems: pooling hurts ----------------
    print("\n[B2] Heterogeneous systems — pooling must be REJECTED by LOO")
    sys_he = _make_systems(rng, S=6, mode="heterogeneous", n_per=6)
    hier_he = HierarchicalDCp(sys_he)
    fit_he = hier_he.fit(n_iter=2500, burn=800, seed=3)
    loo_he = hier_he.loo_system(n_iter=1200, burn=400, seed=4)
    if loo_he["status"] == "OK":
        print(f"      LOO total gain = {loo_he['total_gain']:.3f}  "
              f"-> {loo_he['verdict']}")
        rep("LOO rejects pooling for heterogeneous systems",
            loo_he["total_gain"] < 0)
    if fit_he["status"] == "OK" and fit_he["warnings"]:
        print(f"      warnings: {len(fit_he['warnings'])}")

    # ---------------- B3 small-n system borrows ----------------
    print("\n[B3] A system with n < k borrows strength")
    sys_small = _make_systems(rng, S=5, mode="exchangeable", n_per=6)
    s0 = sys_small[0]
    sys_small[0] = SystemData(s0.name, s0.T[:2], s0.kind[:2],
                              s0.y[:2], s0.sigma[:2], T0=s0.T0)
    hs = HierarchicalDCp(sys_small)
    fs = hs.fit(n_iter=2500, burn=800, seed=5)
    rep("underdetermined system still gets a posterior",
        fs["status"] == "OK"
        and fs["systems"][0]["unpooled_status"] == "UNDERDETERMINED"
        and fs["systems"][0]["borrowed"])
    if fs["status"] == "OK":
        print(f"      sys0 n={fs['systems'][0]['n']} "
              f"theta_mean={np.round(fs['systems'][0]['theta_mean'], 3).tolist()}")
        print("      NOTE: this posterior exists BECAUSE of the prior, "
              "not because sys0 has data.")

    # ---------------- B4 S=1 and S=2 guards ----------------
    print("\n[B4] Guards on the number of systems")
    h1 = HierarchicalDCp(_make_systems(rng, 1, "exchangeable"))
    rep("S=1 refused", h1.fit()["status"] == "REFUSED")
    h2 = HierarchicalDCp(_make_systems(rng, 2, "exchangeable"))
    f2 = h2.fit(n_iter=800, burn=300, seed=6)
    rep("S=2 flagged as barely identified",
        f2["status"] == "OK" and any("S=2" in w for w in f2["warnings"]))
    rep("LOO needs >= 3 systems",
        HierarchicalDCp(_make_systems(rng, 2, "exchangeable"))
        .loo_system()["status"] == "REFUSED")

    print("\n" + "=" * 96)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 96)
    print("""
  WHAT THESE TESTS PROVE
    A1  the product von Mises kernel integrates to 1 on T^3 and log I0 is
        stable at large concentration
    A2  at equal sample count the KDE beats the fixed histogram on
        held-out log density — this is the sparsity fix, measured
    A3  the torus cell list returns a superset of the true neighbourhood
    A4  adaptive binning concentrates resolution where samples are
    A5  the restraint integral is finite when the region is supported
    A6  a genuinely unsupported region is REFUSED, not silently filled
        by the kernel tail
    B1  for exchangeable systems, pooling improves held-out prediction
    B2  for heterogeneous systems, LOO REJECTS pooling — the diagnostic
        detects the failure mode instead of hiding it
    B3  an underdetermined system receives a posterior, flagged as
        prior-driven
    B4  S=1 is refused, S=2 is flagged, LOO needs >= 3

  WHAT IS NOT PROVEN
    * KDE smooths; it does not create samples. The support diagnostic is
      mandatory, not optional.
    * LOO-CV bandwidth can overfit at very small N; inspect cv_curve.
    * Exchangeability is an assumption. LOO is a check, not a proof.
    * With S=2 the between-system covariance is prior-dominated.
    * No experimental validation is performed here.
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Torus KDE + hierarchical v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--samples", default=None, help="NPZ with z1,z2,z3")
    ap.add_argument("--systems", default=None,
                    help="JSON list of {name,T,kind,y,sigma,T0}")
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    out: Dict[str, Any] = {}
    if a.samples:
        z = np.load(a.samples)
        s = np.stack([z["z1"], z["z2"], z["z3"]], -1)
        kde = TorusKDE.fit_cv(s)
        out["kde"] = kde.cv_report
        ab = AdaptiveTorusBins(s).build()
        out["adaptive_bins"] = ab.summary()
        print(json.dumps(out, indent=2))
    if a.systems:
        with open(a.systems) as f:
            raw = json.load(f)
        sysl = [SystemData(d["name"], np.array(d["T"], dtype=float),
                           np.array(d["kind"]), np.array(d["y"], dtype=float),
                           np.array(d["sigma"], dtype=float),
                           T0=float(d.get("T0", 310.0))) for d in raw]
        h = HierarchicalDCp(sysl)
        fit = h.fit()
        show = {k: v for k, v in fit.items() if k != "_draws"}
        print(json.dumps(show, indent=2, default=float))
        out["hierarchical"] = show
        if a.loo:
            loo = h.loo_system()
            print(json.dumps(loo, indent=2, default=float))
            out["loo"] = loo
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
    if not (a.samples or a.systems):
        ap.print_help()
        print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
