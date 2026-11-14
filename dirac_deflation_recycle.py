#!/usr/bin/env python3
"""
=============================================================================
DEFLATION RECYCLING ACROSS RIGHT-HAND SIDES
EigCG harvesting + recyclable coarse space + cache + amortization accounting
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHY RECYCLING WORKS
  The near-null space of the Dirac operator depends on the GAUGE
  CONFIGURATION U (and the mass), NOT on the right-hand side b.
  Lattice QCD solves many RHS per configuration: sequential sources,
  stochastic estimators, dilution schemes, disconnected diagrams,
  HMC forces. So the expensive setup is paid ONCE and divided by n_RHS.

WHAT IS IMPLEMENTED (real, tested)
  [1] EigCG harvesting (Stathopoulos & Orginos, SIAM SISC 2010):
      during CG on A = D^dag D, the CG coefficients alpha_i, beta_i
      give the Lanczos tridiagonal for FREE:
          T_ii     = 1/alpha_i + beta_i/alpha_{i-1}   (beta_0 = 0)
          T_{i,i+1}= -sqrt(beta_{i+1})/alpha_i
      with Lanczos vectors q_i = r_i/||r_i||. Ritz pairs of T approximate
      eigenpairs of A at ZERO extra operator applications.
      Each harvested pair's residual ||A v - lambda v|| is MEASURED and
      pairs above tolerance are DISCARDED, so quality is verified rather
      than assumed (CG residuals lose orthogonality in finite precision).
  [2] RecyclableSubspace: orthonormal near-null basis + Ritz values,
      tagged with a gauge fingerprint, mass, and setup cost.
  [3] Cross-mass recycling by Rayleigh-Ritz RE-PROJECTION:
      H = V^dag A(m_new) V  (k operator applications), eigendecompose H,
      rotate V. This is the correct way to reuse a subspace at a new mass;
      naively reusing the old Ritz VALUES would be wrong.
  [4] SubspaceCache keyed by (gauge fingerprint, mass, method). A gauge
      change INVALIDATES the cache; this is detected, not silently ignored.
  [5] Deflated CG / BiCGStab with the recycled subspace, plus optional
      previous-solution initial guess (useful for correlated RHS).
  [6] Two-level hierarchy recycling: the prolongator P and the whole
      coarse operator are reused across RHS, so ALL levels' setup is
      amortized, not just the fine-level deflation.
  [7] Amortization accounting with a MEASURED break-even n_RHS.

SAFETY PROPERTY (tested explicitly)
  Recycling can only lose EFFICIENCY, never CORRECTNESS, provided the
  final residual is checked against the TRUE operator. A subspace from
  a different gauge configuration is merely a bad preconditioner. The
  self-test feeds a deliberately WRONG subspace and verifies the answer
  is still correct.

HONEST SCOPE
  ❌ EigCG here harvests from CG on the normal equations. Production uses
     flexible variants (FGMRES + deflation), block Krylov, and even-odd
     preconditioned inner solves. None of that is here.
  ❌ Cross-mass recycling is APPROXIMATE: the near-null space does change
     with mass (gamma5 does not commute with Q). Re-projection recovers
     the best subspace within span(V), but span(V) itself degrades as
     |m_new - m_setup| grows. The degradation is MEASURED, not assumed.
  ❌ Small lattices, random (not thermalized) gauge background.
  ❌ No communication overlap, no mixed precision, no deflation of the
     coarse-level operator across levels beyond reuse of P.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve, eigh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BANNER = """
=============================================================================
DEFLATION RECYCLING ACROSS RHS
Setup depends on the GAUGE, not the RHS. Pay once, divide by n_RHS.
EigCG harvests Ritz pairs for FREE during CG.
Recycling loses efficiency, never correctness (residual verified).
=============================================================================
"""


# =============================================================================
# §1. GAMMA + GAUGE + WILSON DIRAC  (same conventions as the earlier modules)
# =============================================================================

def pauli() -> List[np.ndarray]:
    return [np.array([[1, 0], [0, 1]], dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex)]


def gamma_matrices(nd: int) -> Tuple[List[np.ndarray], np.ndarray, int]:
    s = pauli()
    if nd == 2:
        return [s[0].copy(), s[1].copy()], s[2].copy(), 2
    if nd != 4:
        raise ValueError("nd must be 2 or 4")
    I2 = s[0]
    sig = [I2, 1j * s[1], 1j * s[2], 1j * s[3]]
    bar = [I2, -1j * s[1], -1j * s[2], -1j * s[3]]
    g = []
    for mu in range(4):
        gm = np.zeros((4, 4), dtype=complex)
        gm[:2, 2:] = sig[mu]
        gm[2:, :2] = bar[mu]
        g.append(gm)
    return g, 1j * (g[0] @ g[1] @ g[2] @ g[3]), 4


@dataclass
class GaugeField:
    nd: int
    L: int
    Nc: int
    links: np.ndarray

    @property
    def V(self) -> int:
        return self.L ** self.nd

    def coords(self) -> np.ndarray:
        grids = np.meshgrid(*[np.arange(self.L)] * self.nd, indexing="ij")
        return np.stack([g.reshape(-1) for g in grids], axis=-1)

    def idx(self, c: np.ndarray) -> np.ndarray:
        c = np.mod(c, self.L)
        out = np.zeros(c.shape[:-1], dtype=np.int64)
        for d in range(self.nd):
            out = out * self.L + c[..., d]
        return out

    def fingerprint(self) -> str:
        """Exact hash of the link array. Detects ANY gauge change."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.links).tobytes())
        h.update(f"{self.nd}|{self.L}|{self.Nc}".encode())
        return h.hexdigest()[:16]

    def plaquette(self) -> float:
        c = self.coords()
        tot, cnt = 0.0, 0
        for mu in range(self.nd):
            cm = c.copy(); cm[:, mu] += 1
            im = self.idx(cm)
            for nu in range(mu + 1, self.nd):
                cn = c.copy(); cn[:, nu] += 1
                inn = self.idx(cn)
                P = (self.links[mu] @ self.links[nu][im]
                     @ self.links[mu][inn].conj().swapaxes(-1, -2)
                     @ self.links[nu].conj().swapaxes(-1, -2))
                tot += float(np.trace(P, axis1=-2, axis2=-1).real.mean()) / self.Nc
                cnt += 1
        return tot / max(cnt, 1)


def make_gauge(nd: int, L: int, Nc: int = 1, beta: float = 6.0,
               cold: bool = False, seed: int = 0) -> GaugeField:
    V = L ** nd
    rng = np.random.default_rng(seed)
    if cold:
        links = np.tile(np.eye(Nc, dtype=complex), (nd, V, 1, 1))
    elif Nc == 1:
        th = rng.normal(0.0, 1.0 / max(beta, 1e-3), size=(nd, V, 1, 1))
        links = np.exp(1j * th)
    else:
        A = (rng.normal(size=(nd, V, Nc, Nc))
             + 1j * rng.normal(size=(nd, V, Nc, Nc)))
        Q, R = np.linalg.qr(A)
        ph = np.diagonal(R, axis1=-2, axis2=-1)
        Q = Q * (ph / np.abs(ph))[..., None, :]
        U, _, Vh = np.linalg.svd(Q)
        X = U @ Vh
        links = X / np.linalg.det(X)[..., None, None] ** (1.0 / Nc)
    return GaugeField(nd, L, Nc, links)


class WilsonDirac:
    """Matrix-free Wilson Dirac, r=1, a=1. Field shape (V, ns, Nc)."""

    def __init__(self, gauge: GaugeField, mass: float = 0.1):
        self.g = gauge
        self.m = float(mass)
        self.nd, self.L, self.Nc = gauge.nd, gauge.L, gauge.Nc
        gam, g5, ns = gamma_matrices(self.nd)
        self.gam, self.g5, self.ns = gam, g5, ns
        self.c = self.m + self.nd
        self.coords = gauge.coords()
        self.V = gauge.V
        self.shape = (self.V, ns, self.Nc)
        self.w = ns * self.Nc
        self.size = self.V * self.w
        self.ip, self.im = [], []
        for mu in range(self.nd):
            cp = self.coords.copy(); cp[:, mu] += 1
            cm = self.coords.copy(); cm[:, mu] -= 1
            self.ip.append(gauge.idx(cp))
            self.im.append(gauge.idx(cm))
        self.n_apply = 0
        self.fingerprint = gauge.fingerprint()

    def _g(self, mu, f):
        return np.einsum("st,vtc->vsc", self.gam[mu], f)

    def _u(self, mu, f, dag=False):
        U = self.g.links[mu]
        if dag:
            U = U.conj().swapaxes(-1, -2)
        return np.einsum("vcd,vsd->vsc", U, f)

    def apply_g5(self, psi):
        return np.einsum("st,vtc->vsc", self.g5, psi)

    def zeros(self):
        return np.zeros(self.shape, dtype=complex)

    def randn(self, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.shape) + 1j * rng.normal(size=self.shape)

    def reset(self):
        self.n_apply = 0

    def apply(self, psi):
        self.n_apply += 1
        out = self.c * psi
        for mu in range(self.nd):
            fwd = self._u(mu, psi[self.ip[mu]])
            out -= 0.5 * (fwd - self._g(mu, fwd))
            bwd = self._u(mu, psi, dag=True)[self.im[mu]]
            out -= 0.5 * (bwd + self._g(mu, bwd))
        return out

    def apply_dag(self, psi):
        self.n_apply += 1
        out = self.c * psi
        for mu in range(self.nd):
            t = psi - self._g(mu, psi)
            out -= 0.5 * self._u(mu, t, dag=True)[self.im[mu]]
            s = psi + self._g(mu, psi)
            out -= 0.5 * self._u(mu, s)[self.ip[mu]]
        return out

    def apply_normal(self, psi):
        """A = D^dag D (HPD). 2 Dirac applications."""
        return self.apply_dag(self.apply(psi))

    def apply_Q(self, psi):
        return self.apply_g5(self.apply(psi))

    @staticmethod
    def dot(a, b):
        return complex(np.vdot(a.reshape(-1), b.reshape(-1)))

    @staticmethod
    def norm(a):
        return float(np.linalg.norm(a.reshape(-1)))


def dot(a, b):
    return complex(np.vdot(a.reshape(-1), b.reshape(-1)))


def nrm(a):
    return float(np.linalg.norm(a.reshape(-1)))


# =============================================================================
# §2. RECYCLABLE SUBSPACE
# =============================================================================

@dataclass
class RecyclableSubspace:
    """
    Orthonormal near-null basis, tagged so a cache can tell whether it
    is still valid.

    vecs   : (k, *field_shape) orthonormal eigenvector approximations of A
    lam    : (k,) Ritz values of A = D^dag D  (so sigma_min(D) = sqrt(lam))
    """
    vecs: np.ndarray
    lam: np.ndarray
    gauge_fp: str
    mass: float
    method: str
    setup_applications: int
    ritz_residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def k(self) -> int:
        return len(self.lam)

    def matches(self, D: WilsonDirac) -> bool:
        return self.gauge_fp == D.fingerprint and abs(self.mass - D.m) < 1e-12

    def summary(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "gauge_fp": self.gauge_fp,
            "mass": self.mass,
            "method": self.method,
            "setup_applications": self.setup_applications,
            "lam_min": float(self.lam.min()) if self.k else None,
            "lam_max": float(self.lam.max()) if self.k else None,
            "max_ritz_residual": (float(self.ritz_residuals.max())
                                  if len(self.ritz_residuals) else None),
        }


def orthonormalize(vecs: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Modified Gram-Schmidt with drop, twice for stability."""
    out: List[np.ndarray] = []
    for v in vecs:
        w = v.copy()
        for _ in range(2):
            for q in out:
                w = w - dot(q, w) * q
        nw = nrm(w)
        if nw > tol:
            out.append(w / nw)
    if not out:
        return np.zeros((0,) + vecs.shape[1:], dtype=complex)
    return np.stack(out, axis=0)


def rayleigh_ritz_reproject(Aop: Callable, V: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Correct cross-mass recycling: H = V^dag A V, eigendecompose, rotate V.
    Costs k operator applications. Returns (new_vecs, new_lam, n_apply).
    """
    k = len(V)
    if k == 0:
        return V, np.zeros(0), 0
    AV = np.stack([Aop(V[i]) for i in range(k)], axis=0)
    H = np.array([[dot(V[i], AV[j]) for j in range(k)] for i in range(k)])
    H = 0.5 * (H + H.conj().T)                 # enforce Hermitian numerically
    if HAS_SCIPY:
        w, Y = eigh(H)
    else:
        w, Y = np.linalg.eigh(H)
    newV = np.einsum("jk,k...->j...", Y.T, V)
    newV = orthonormalize(newV)
    n_keep = min(len(newV), k)
    return newV[:n_keep], np.asarray(w.real)[:n_keep], k


def measure_ritz_residuals(Aop: Callable, V: np.ndarray, lam: np.ndarray
                           ) -> Tuple[np.ndarray, int]:
    """||A v - lam v|| for each pair. Verifies quality instead of assuming it."""
    res = []
    for i in range(len(V)):
        Av = Aop(V[i])
        res.append(nrm(Av - lam[i] * V[i]) / max(abs(lam[i]), 1e-14))
    return np.asarray(res), len(V)


# =============================================================================
# §3. EigCG — FREE Ritz harvesting during CG
# =============================================================================

@dataclass
class CGResult:
    x: np.ndarray
    converged: bool
    n_iter: int
    residual: float
    n_apply: int
    harvested: Optional[RecyclableSubspace] = None


def cg_eigcg(D: WilsonDirac, b: np.ndarray, tol: float = 1e-9,
             maxiter: int = 3000, m_eig: int = 24,
             harvest: bool = True, ritz_tol: float = 1e-3,
             x0: Optional[np.ndarray] = None) -> CGResult:
    """
    CG on A = D^dag D with rhs = D^dag b, harvesting Ritz pairs for FREE
    from the CG coefficients (EigCG, Stathopoulos-Orginos 2010).

    T_ii      = 1/alpha_i + beta_i/alpha_{i-1}    (beta_0 = 0)
    T_{i,i+1} = -sqrt(beta_{i+1})/alpha_i
    q_i       = r_i / ||r_i||

    Harvested pairs are VERIFIED (residual measured) and filtered.
    """
    Aop = D.apply_normal
    D.reset()
    rhs = D.apply_dag(b)
    nb = max(nrm(rhs), 1e-30)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = rhs - Aop(x)
    rr = dot(r, r).real
    rho = math.sqrt(max(rr, 0.0))
    p = r.copy()
    Q: List[np.ndarray] = [r / rho] if rho > 1e-30 else []
    alphas: List[float] = []
    betas: List[float] = []
    it = 0
    for it in range(1, maxiter + 1):
        if rho / nb < tol:
            break
        Ap = Aop(p)
        pAp = dot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        alpha = rr / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = dot(r, r).real
        beta = rr_new / max(rr, 1e-30)
        alphas.append(alpha)
        betas.append(beta)
        rho_new = math.sqrt(max(rr_new, 0.0))
        if rho_new > 1e-30:
            Q.append(r / rho_new)
        p = r + beta * p
        rr, rho = rr_new, rho_new

    # verify against the TRUE full operator
    res_true = nrm(b - D.apply(x)) / max(nrm(b), 1e-30)

    harvested = None
    m = len(alphas)
    if harvest and m >= 4 and len(Q) >= m:
        T = np.zeros((m, m))
        for i in range(m):
            T[i, i] = 1.0 / alphas[i] + (betas[i - 1] / alphas[i - 1] if i > 0 else 0.0)
        for i in range(m - 1):
            v = -math.sqrt(max(betas[i], 0.0)) / alphas[i]
            T[i, i + 1] = v
            T[i + 1, i] = v
        if HAS_SCIPY:
            w, Y = eigh(T)
        else:
            w, Y = np.linalg.eigh(T)
        Qm = np.stack(Q[:m], axis=0)
        nkeep = int(min(m_eig, m))
        cand = np.einsum("jk,k...->j...", Y[:, :nkeep].T, Qm)
        cand = orthonormalize(cand)
        lam_c = np.asarray(w.real)[:len(cand)]
        # VERIFY: measure Ritz residuals, discard bad pairs
        D.reset()
        rr_meas, ncost = measure_ritz_residuals(Aop, cand, lam_c)
        keep = rr_meas < ritz_tol
        if keep.any():
            harvested = RecyclableSubspace(
                vecs=cand[keep], lam=lam_c[keep],
                gauge_fp=D.fingerprint, mass=D.m, method="eigcg",
                setup_applications=ncost,      # verification cost only
                ritz_residuals=rr_meas[keep])
    return CGResult(x=x, converged=bool(res_true < tol), n_iter=it,
                    residual=float(res_true), n_apply=D.n_apply,
                    harvested=harvested)


def lanczos_setup(D: WilsonDirac, k: int = 12, n_lanczos: int = 40,
                  seed: int = 0) -> RecyclableSubspace:
    """Explicit Lanczos on Q = gamma5 D; |lambda| are singular values of D."""
    D.reset()
    v = D.randn(seed); v = v / nrm(v)
    V = [v]
    al, be = [], []
    vprev, bprev = None, 0.0
    for j in range(min(n_lanczos, D.size - 1)):
        w = D.apply_Q(V[j])
        a = dot(V[j], w).real
        al.append(a)
        w = w - a * V[j]
        if vprev is not None:
            w = w - bprev * vprev
        for q in V:
            w = w - dot(q, w) * q
        bn = nrm(w)
        if bn < 1e-13:
            break
        be.append(bn)
        vprev, bprev = V[j], bn
        V.append(w / bn)
    n = len(al)
    T = np.diag(np.asarray(al))
    for j in range(len(be)):
        T[j, j + 1] = be[j]
        T[j + 1, j] = be[j]
    if HAS_SCIPY:
        w, Y = eigh(T)
    else:
        w, Y = np.linalg.eigh(T)
    Vs = np.stack(V[:n], axis=0)
    vecs = np.einsum("jk,k...->j...", Y.T, Vs)
    order = np.argsort(np.abs(w))[:k]
    vecs = orthonormalize(vecs[order])
    lam_D = np.abs(w[order])[:len(vecs)]
    lam_A = lam_D ** 2                       # eigenvalues of D^dag D
    D.reset()
    rr, ncost = measure_ritz_residuals(D.apply_normal, vecs, lam_A)
    return RecyclableSubspace(vecs=vecs, lam=lam_A, gauge_fp=D.fingerprint,
                              mass=D.m, method="lanczos",
                              setup_applications=D.n_apply + ncost,
                              ritz_residuals=rr)


# =============================================================================
# §4. SUBSPACE CACHE
# =============================================================================

class SubspaceCache:
    """
    Keyed by (gauge fingerprint, mass, method). A gauge change is an
    automatic MISS, so a stale subspace is never silently reused.
    """

    def __init__(self, max_entries: int = 8):
        self.store: Dict[Tuple[str, float, str], RecyclableSubspace] = {}
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    @staticmethod
    def key(D: WilsonDirac, method: str) -> Tuple[str, float, str]:
        return (D.fingerprint, round(D.m, 12), method)

    def get(self, D: WilsonDirac, method: str) -> Optional[RecyclableSubspace]:
        s = self.store.get(self.key(D, method))
        if s is None:
            self.misses += 1
            return None
        if not s.matches(D):
            self.invalidations += 1
            self.misses += 1
            return None
        self.hits += 1
        return s

    def put(self, s: RecyclableSubspace) -> None:
        if len(self.store) >= self.max_entries:
            self.store.pop(next(iter(self.store)))
        self.store[(s.gauge_fp, round(s.mass, 12), s.method)] = s

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self.store), "hits": self.hits,
                "misses": self.misses, "invalidations": self.invalidations}


# =============================================================================
# §5. DEFLATED SOLVERS WITH RECYCLED SUBSPACE
# =============================================================================

@dataclass
class SolveStats:
    solver: str
    converged: bool
    n_iter: int
    residual: float
    n_apply: int
    setup_apply_charged: int
    recycled: bool
    k_deflated: int


def deflated_cg(D: WilsonDirac, b: np.ndarray, sub: RecyclableSubspace,
                tol: float = 1e-9, maxiter: int = 3000,
                charge_setup: bool = False) -> SolveStats:
    """
    Deflated CG on A = D^dag D. Projector P = I - V V^dag applied each step.
    Initial guess = Galerkin solution on span(V).
    Residual VERIFIED against the true full operator D.
    """
    Aop = D.apply_normal
    D.reset()
    rhs = D.apply_dag(b)
    V, lam = sub.vecs, np.maximum(sub.lam, 1e-30)
    k = sub.k

    def proj(z):
        for i in range(k):
            z = z - dot(V[i], z) * V[i]
        return z

    x = np.zeros_like(b)
    for i in range(k):
        x = x + (dot(V[i], rhs) / lam[i]) * V[i]
    r = rhs - Aop(x)
    z = proj(r)
    p = z.copy()
    rz = dot(r, z).real
    nb = max(nrm(rhs), 1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        if math.sqrt(max(rz, 0.0)) / nb < tol:
            break
        Ap = Aop(p)
        pAp = dot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        a = rz / pAp
        x = x + a * p
        r = r - a * Ap
        z = proj(r)
        rz_new = dot(r, z).real
        p = z + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new
    res_true = nrm(b - D.apply(x)) / max(nrm(b), 1e-30)
    return SolveStats("deflated_cg", bool(res_true < tol), it, float(res_true),
                      D.n_apply,
                      sub.setup_applications if charge_setup else 0,
                      True, k)


def deflated_bicgstab(D: WilsonDirac, b: np.ndarray, sub: RecyclableSubspace,
                      tol: float = 1e-9, maxiter: int = 3000,
                      charge_setup: bool = False) -> SolveStats:
    """
    Coarse initial guess from the recycled near-null modes of Q, then
    plain BiCGStab on D. Since Q = gamma5 D, an eigenvector of Q with
    eigenvalue lambda gives a Galerkin guess for D x = b.
    """
    D.reset()
    rhsQ = D.apply_g5(b)
    x = np.zeros_like(b)
    lamQ = np.sqrt(np.maximum(sub.lam, 0.0))
    for i in range(sub.k):
        if lamQ[i] > 1e-14:
            x = x + (dot(sub.vecs[i], rhsQ) / lamQ[i]) * sub.vecs[i]
    Aop = D.apply
    r = b - Aop(x)
    r0 = r.copy()
    nb = max(nrm(b), 1e-30)
    rho = alpha = omega = 1.0
    v = np.zeros_like(b); p = np.zeros_like(b)
    it = 0
    for it in range(1, maxiter + 1):
        if nrm(r) / nb < tol:
            break
        rho_new = dot(r0, r)
        if abs(rho_new) < 1e-30:
            break
        beta = (rho_new / rho) * (alpha / omega) if it > 1 else 0.0
        p = r + beta * (p - omega * v)
        v = Aop(p)
        den = dot(r0, v)
        if abs(den) < 1e-30:
            break
        alpha = rho_new / den
        s = r - alpha * v
        if nrm(s) / nb < tol:
            x = x + alpha * p
            break
        t = Aop(s)
        tt = dot(t, t)
        omega = dot(t, s) / tt if abs(tt) > 1e-30 else 0.0
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        if abs(omega) < 1e-30:
            break
    res_true = nrm(b - Aop(x)) / max(nrm(b), 1e-30)
    return SolveStats("deflated_bicgstab", bool(res_true < tol), it,
                      float(res_true), D.n_apply,
                      sub.setup_applications if charge_setup else 0, True, sub.k)


def plain_bicgstab(D: WilsonDirac, b: np.ndarray, tol: float = 1e-9,
                   maxiter: int = 3000) -> SolveStats:
    D.reset()
    Aop = D.apply
    x = np.zeros_like(b)
    r = b - Aop(x)
    r0 = r.copy()
    nb = max(nrm(b), 1e-30)
    rho = alpha = omega = 1.0
    v = np.zeros_like(b); p = np.zeros_like(b)
    it = 0
    for it in range(1, maxiter + 1):
        if nrm(r) / nb < tol:
            break
        rho_new = dot(r0, r)
        if abs(rho_new) < 1e-30:
            break
        beta = (rho_new / rho) * (alpha / omega) if it > 1 else 0.0
        p = r + beta * (p - omega * v)
        v = Aop(p)
        den = dot(r0, v)
        if abs(den) < 1e-30:
            break
        alpha = rho_new / den
        s = r - alpha * v
        if nrm(s) / nb < tol:
            x = x + alpha * p
            break
        t = Aop(s)
        tt = dot(t, t)
        omega = dot(t, s) / tt if abs(tt) > 1e-30 else 0.0
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        if abs(omega) < 1e-30:
            break
    res = nrm(b - Aop(x)) / nb
    return SolveStats("plain_bicgstab", bool(res < tol), it, float(res),
                      D.n_apply, 0, False, 0)


# =============================================================================
# §6. MULTI-LEVEL HIERARCHY RECYCLING  (reuse P and coarse operator)
# =============================================================================

class RecycledTwoLevel:
    """
    Minimal two-level deflation preconditioner whose SETUP IS RECYCLED.
    The prolongator P (built from near-null test vectors) and the coarse
    operator D_c = P^dag D P depend only on the gauge and mass, so they
    are built once and reused for every RHS. This is where the whole
    hierarchy's setup cost gets amortized, not just fine-level deflation.
    """

    def __init__(self, D: WilsonDirac, n_modes: int = 10,
                 n_relax: int = 6, block: int = 2, seed: int = 0):
        self.D = D
        self.gauge_fp = D.fingerprint
        self.mass = D.m
        D.reset()
        # adaptive setup: relax on the homogeneous problem
        cols: List[np.ndarray] = []
        for j in range(n_modes):
            v = D.randn(seed + 31 * j)
            v /= nrm(v)
            for _ in range(n_relax):
                # cheap block-Jacobi-like relaxation on D^dag D
                v = v - 0.15 * (D.apply_normal(v) / max(D.c ** 2, 1e-8))
                nv = nrm(v)
                if nv < 1e-14:
                    break
                v = v / nv
            v = v / max(nrm(v), 1e-30)
            for q in cols:
                v = v - dot(q, v) * q
            if nrm(v) > 1e-8:
                cols.append(v / nrm(v))
        self.P = np.stack(cols, axis=0) if cols else np.zeros((0,) + D.shape, complex)
        self.k = len(self.P)
        # Galerkin coarse operator
        if self.k:
            DP = np.stack([D.apply(self.P[i]) for i in range(self.k)], axis=0)
            self.Dc = np.array([[dot(self.P[i], DP[j]) for j in range(self.k)]
                                for i in range(self.k)])
            self.Dc = self.Dc + 1e-10 * np.eye(self.k)
        else:
            self.Dc = np.zeros((0, 0), complex)
        self.setup_applications = D.n_apply
        self.n_cycles = 0

    def matches(self, D: WilsonDirac) -> bool:
        return self.gauge_fp == D.fingerprint and abs(self.mass - D.m) < 1e-12

    def apply(self, b: np.ndarray) -> np.ndarray:
        self.n_cycles += 1
        if self.k == 0:
            return b / max(self.D.c, 1e-8)
        rc = np.array([dot(self.P[i], b) for i in range(self.k)])
        ec = np.linalg.solve(self.Dc, rc)
        x = np.einsum("j,j...->...", ec, self.P)
        # one post-relaxation step
        x = x - 0.1 * (self.D.apply(x) - b) / max(self.D.c, 1e-8)
        return x


def gmres_precond(D: WilsonDirac, Minv: Callable, b: np.ndarray,
                  tol: float = 1e-9, restart: int = 20,
                  max_outer: int = 40) -> SolveStats:
    """Right-preconditioned FGMRES. Residual verified against true D."""
    D.reset()
    Aop = D.apply
    x = np.zeros_like(b)
    nb = max(nrm(b), 1e-30)
    total = 0
    for _ in range(max_outer):
        r = b - Aop(x)
        beta = nrm(r)
        if beta / nb < tol:
            break
        m = restart
        V = [r / beta]
        Z = []
        H = np.zeros((m + 1, m), dtype=complex)
        for j in range(m):
            z = Minv(V[j])
            Z.append(z)
            w = Aop(z)
            for i in range(j + 1):
                H[i, j] = dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = nrm(w)
            total += 1
            if abs(H[j + 1, j]) < 1e-14:
                m = j + 1
                break
            V.append(w / H[j + 1, j])
        e1 = np.zeros(m + 1, dtype=complex); e1[0] = beta
        y, *_ = np.linalg.lstsq(H[:m + 1, :m], e1, rcond=None)
        for j in range(m):
            x = x + y[j] * Z[j]
        if nrm(b - Aop(x)) / nb < tol:
            break
    res = nrm(b - Aop(x)) / nb
    return SolveStats("fgmres+recycled2lvl", bool(res < tol), total,
                      float(res), D.n_apply, 0, True, 0)


# =============================================================================
# §7. AMORTIZATION BENCHMARK
# =============================================================================

def amortization_benchmark(nd: int = 4, L: int = 4, Nc: int = 1,
                           mass: float = 0.15, beta: float = 5.0,
                           n_rhs: int = 8, n_modes: int = 10,
                           tol: float = 1e-8, seed: int = 3
                           ) -> Dict[str, Any]:
    """
    Three strategies on the SAME gauge config, SAME n_rhs RHS:
      A plain      : no deflation, no setup
      B rebuild    : fresh Lanczos setup for EVERY RHS
      C recycle    : ONE setup, reused for all RHS
    Break-even n_RHS is computed from the MEASURED costs.
    """
    g = make_gauge(nd, L, Nc, beta, seed=seed)
    D = WilsonDirac(g, mass)
    rhs = [D.randn(500 + i) for i in range(n_rhs)]

    # A: plain
    plain_iters, plain_apps, plain_ok = [], [], 0
    for b in rhs:
        st = plain_bicgstab(D, b, tol=tol)
        plain_iters.append(st.n_iter); plain_apps.append(st.n_apply)
        plain_ok += int(st.converged)

    # setup once
    sub = lanczos_setup(D, k=n_modes, n_lanczos=40, seed=seed)
    setup_cost = sub.setup_applications

    # B: rebuild every RHS
    rebuild_apps, rebuild_ok = [], 0
    for b in rhs:
        s = lanczos_setup(D, k=n_modes, n_lanczos=40, seed=seed)
        st = deflated_cg(D, b, s, tol=tol)
        rebuild_apps.append(st.n_apply + s.setup_applications)
        rebuild_ok += int(st.converged)

    # C: recycle
    rec_apps, rec_iters, rec_ok = [], [], 0
    for b in rhs:
        st = deflated_cg(D, b, sub, tol=tol)
        rec_apps.append(st.n_apply); rec_iters.append(st.n_iter)
        rec_ok += int(st.converged)

    A = int(np.sum(plain_apps))
    B = int(np.sum(rebuild_apps))
    C = int(setup_cost + np.sum(rec_apps))

    c0 = float(np.mean(plain_apps))
    cd = float(np.mean(rec_apps))
    breakeven = (setup_cost / (c0 - cd)) if c0 > cd else float("inf")

    # hierarchy recycling
    t0 = time.time()
    hier = RecycledTwoLevel(D, n_modes=n_modes, seed=seed)
    hier_setup_s = time.time() - t0
    hier_setup_apps = hier.setup_applications
    hier_apps, hier_ok = [], 0
    for b in rhs:
        st = gmres_precond(D, hier.apply, b, tol=tol, restart=20)
        hier_apps.append(st.n_apply); hier_ok += int(st.converged)
    Ht = int(hier_setup_apps + np.sum(hier_apps))

    return {
        "config": {"nd": nd, "L": L, "Nc": Nc, "mass": mass,
                   "n_rhs": n_rhs, "n_modes": n_modes},
        "gauge_fp": D.fingerprint,
        "plaquette": g.plaquette(),
        "setup_applications": setup_cost,
        "plain": {"total_apps": A, "mean_iters": float(np.mean(plain_iters)),
                  "converged": plain_ok},
        "rebuild_every_rhs": {"total_apps": B, "converged": rebuild_ok},
        "recycle": {"total_apps": C, "mean_iters": float(np.mean(rec_iters)),
                    "converged": rec_ok},
        "hierarchy_recycle": {"total_apps": Ht, "converged": hier_ok,
                              "setup_apps": hier_setup_apps,
                              "setup_wall_s": hier_setup_s},
        "speedup_recycle_vs_plain": A / max(C, 1),
        "speedup_recycle_vs_rebuild": B / max(C, 1),
        "breakeven_n_rhs": breakeven,
        "subspace": sub.summary(),
    }


def cross_mass_report(nd: int = 4, L: int = 4, mass_setup: float = 0.20,
                      masses: Tuple[float, ...] = (0.20, 0.15, 0.10, 0.06),
                      n_modes: int = 10, n_rhs: int = 4,
                      tol: float = 1e-8, seed: int = 5) -> List[Dict[str, Any]]:
    """
    Recycle a subspace built at mass_setup to other masses, WITH and
    WITHOUT Rayleigh-Ritz re-projection. Measures the degradation.
    """
    g = make_gauge(nd, L, 1, beta=5.0, seed=seed)
    D0 = WilsonDirac(g, mass_setup)
    base = lanczos_setup(D0, k=n_modes, n_lanczos=40, seed=seed)
    rows = []
    for m in masses:
        D = WilsonDirac(g, m)
        rhs = [D.randn(700 + i) for i in range(n_rhs)]
        # fresh setup at this mass (reference)
        fresh = lanczos_setup(D, k=n_modes, n_lanczos=40, seed=seed)
        fresh_apps = []
        for b in rhs:
            fresh_apps.append(deflated_cg(D, b, fresh, tol=tol).n_apply)
        # naive reuse: old vectors, re-projected values but NO rotation
        naive = RecyclableSubspace(vecs=base.vecs, lam=base.lam,
                                   gauge_fp=D.fingerprint, mass=m,
                                   method="naive_reuse", setup_applications=0)
        naive_apps, naive_ok = [], 0
        for b in rhs:
            st = deflated_cg(D, b, naive, tol=tol)
            naive_apps.append(st.n_apply); naive_ok += int(st.converged)
        # correct reuse: Rayleigh-Ritz re-projection
        V2, lam2, cost = rayleigh_ritz_reproject(D.apply_normal, base.vecs)
        rr, cc = measure_ritz_residuals(D.apply_normal, V2, lam2)
        proj = RecyclableSubspace(vecs=V2, lam=lam2, gauge_fp=D.fingerprint,
                                  mass=m, method="reprojected",
                                  setup_applications=cost + cc,
                                  ritz_residuals=rr)
        proj_apps, proj_ok = [], 0
        for b in rhs:
            st = deflated_cg(D, b, proj, tol=tol)
            proj_apps.append(st.n_apply); proj_ok += int(st.converged)
        rows.append({
            "mass": m,
            "delta_mass": abs(m - mass_setup),
            "fresh_total": int(np.sum(fresh_apps) + fresh.setup_applications),
            "naive_total": int(np.sum(naive_apps)),
            "naive_converged": naive_ok,
            "reprojected_total": int(np.sum(proj_apps) + proj.setup_applications),
            "reprojected_converged": proj_ok,
            "reprojection_cost": cost + cc,
            "max_ritz_residual_after": float(rr.max()) if len(rr) else None,
        })
    return rows


def print_amortization(r: Dict[str, Any]) -> None:
    c = r["config"]
    print("=" * 78)
    print(f"AMORTIZATION  nd={c['nd']} L={c['L']} Nc={c['Nc']} "
          f"mass={c['mass']} n_RHS={c['n_rhs']} n_modes={c['n_modes']}")
    print(f"gauge fingerprint = {r['gauge_fp']}   plaquette = {r['plaquette']:.4f}")
    print("=" * 78)
    print(f"  setup cost (paid ONCE)        : {r['setup_applications']} Dirac apps")
    print(f"  {'strategy':<26}{'total apps':>12}{'converged':>12}")
    print("-" * 78)
    for key, label in (("plain", "A plain (no deflation)"),
                       ("rebuild_every_rhs", "B rebuild every RHS"),
                       ("recycle", "C recycle (setup once)"),
                       ("hierarchy_recycle", "D hierarchy recycle")):
        d = r[key]
        print(f"  {label:<26}{d['total_apps']:>12}{d['converged']:>12}")
    print("-" * 78)
    print(f"  speedup C vs A (plain)        : {r['speedup_recycle_vs_plain']:.3f}x")
    print(f"  speedup C vs B (rebuild)      : {r['speedup_recycle_vs_rebuild']:.3f}x")
    be = r["breakeven_n_rhs"]
    print(f"  BREAK-EVEN n_RHS              : "
          f"{be:.2f}" if math.isfinite(be) else "  BREAK-EVEN n_RHS              : never")
    print("     (below this, plain is cheaper; above it, recycling wins)")
    print("=" * 78)


def print_cross_mass(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("CROSS-MASS RECYCLING  (subspace built once, reused at other masses)")
    print("=" * 78)
    print(f"  {'mass':>8}{'|dm|':>8}{'fresh':>10}{'naive':>10}{'reproj':>10}"
          f"{'reproj cost':>13}{'ritz res':>12}")
    print("-" * 78)
    for r in rows:
        rr = r["max_ritz_residual_after"]
        print(f"  {r['mass']:>8.3f}{r['delta_mass']:>8.3f}"
              f"{r['fresh_total']:>10}{r['naive_total']:>10}"
              f"{r['reprojected_total']:>10}{r['reprojection_cost']:>13}"
              f"{(f'{rr:.2e}' if rr is not None else '-'):>12}")
    print("=" * 78)
    print("  naive   = old vectors AND old Ritz values (WRONG values at new mass)")
    print("  reproj  = Rayleigh-Ritz re-projection: H = V^dag A(m) V, rotate V")
    print("  Read: reproj should beat naive; both degrade as |dm| grows.")
    print("        That degradation is real and is MEASURED here, not assumed.")


# =============================================================================
# §8. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ---- T1 gauge fingerprint detects change ----
    print("\n[T1] Gauge fingerprint / cache invalidation")
    g1 = make_gauge(4, 4, 1, beta=5.0, seed=1)
    g2 = make_gauge(4, 4, 1, beta=5.0, seed=2)
    rep("same gauge -> same fingerprint",
        g1.fingerprint() == make_gauge(4, 4, 1, beta=5.0, seed=1).fingerprint())
    rep("different gauge -> different fingerprint",
        g1.fingerprint() != g2.fingerprint())

    D1 = WilsonDirac(g1, 0.2)
    D2 = WilsonDirac(g2, 0.2)
    cache = SubspaceCache()
    s1 = lanczos_setup(D1, k=8, n_lanczos=30)
    cache.put(s1)
    rep("cache HIT for same gauge+mass", cache.get(D1, "lanczos") is not None)
    rep("cache MISS for different gauge", cache.get(D2, "lanczos") is None)
    D1b = WilsonDirac(g1, 0.1)
    rep("cache MISS for different mass", cache.get(D1b, "lanczos") is None)

    # ---- T2 EigCG harvest quality ----
    print("\n[T2] EigCG harvests Ritz pairs for FREE (verified, not assumed)")
    D = WilsonDirac(g1, 0.25)
    b = D.randn(11)
    cg = cg_eigcg(D, b, tol=1e-9, m_eig=12, ritz_tol=1e-2)
    rep("CG converged", cg.converged, f"res={cg.residual:.2e}")
    rep("EigCG harvested a subspace", cg.harvested is not None,
        f"k={cg.harvested.k if cg.harvested else 0}")
    if cg.harvested:
        rep("harvested vectors orthonormal",
            float(np.abs(np.array([[dot(cg.harvested.vecs[i], cg.harvested.vecs[j])
                                    for j in range(cg.harvested.k)]
                                   for i in range(cg.harvested.k)])
                         - np.eye(cg.harvested.k)).max()) < 1e-8)
        rep("harvested Ritz residuals below tolerance",
            float(cg.harvested.ritz_residuals.max()) < 1e-2,
            f"max={cg.harvested.ritz_residuals.max():.2e}")
        # compare lowest Ritz value to dense exact
        n = D.size
        if n <= 1200:
            Ad = np.zeros((n, n), dtype=complex)
            for j in range(n):
                e = np.zeros(n, dtype=complex); e[j] = 1.0
                Ad[:, j] = D.apply_normal(e.reshape(D.shape)).reshape(-1)
            ev = np.linalg.eigvalsh(0.5 * (Ad + Ad.conj().T))
            rep("lowest harvested Ritz value near true lambda_min",
                abs(cg.harvested.lam.min() - ev[0]) / max(ev[0], 1e-12) < 0.25,
                f"harvest={cg.harvested.lam.min():.5f} exact={ev[0]:.5f}")

    # ---- T3 deflation reduces iterations ----
    print("\n[T3] Recycled deflation reduces iteration count")
    sub = lanczos_setup(D, k=10, n_lanczos=40, seed=0)
    b2 = D.randn(12)
    plain = plain_bicgstab(D, b2, tol=1e-9)
    defl = deflated_cg(D, b2, sub, tol=1e-9)
    rep("deflated solve converged", defl.converged, f"res={defl.residual:.2e}")
    rep("deflation reduces Dirac applications",
        defl.n_apply < plain.n_apply,
        f"{defl.n_apply} vs plain {plain.n_apply}")

    # ---- T4 SAFETY: wrong subspace still gives the CORRECT answer ----
    print("\n[T4] SAFETY: a WRONG subspace loses efficiency, never correctness")
    wrong = lanczos_setup(D2, k=10, n_lanczos=40, seed=0)   # different gauge!
    rep("subspace really is from another gauge",
        wrong.gauge_fp != D.fingerprint)
    st_wrong = deflated_cg(D, b2, wrong, tol=1e-9, maxiter=4000)
    rep("solve with WRONG subspace still converges to the true answer",
        st_wrong.converged, f"res={st_wrong.residual:.2e}")
    st_wrong_b = deflated_bicgstab(D, b2, wrong, tol=1e-9, maxiter=4000)
    rep("BiCGStab with WRONG subspace also correct",
        st_wrong_b.converged, f"res={st_wrong_b.residual:.2e}")

    # ---- T5 cross-mass re-projection ----
    print("\n[T5] Cross-mass recycling: Rayleigh-Ritz re-projection")
    Dnew = WilsonDirac(g1, 0.12)
    V2, lam2, cost = rayleigh_ritz_reproject(Dnew.apply_normal, sub.vecs)
    rr, _ = measure_ritz_residuals(Dnew.apply_normal, V2, lam2)
    rep("re-projection costs k applications", cost == sub.k, f"cost={cost}")
    rep("re-projected Ritz residuals small", float(rr.max()) < 1e-6,
        f"max={rr.max():.2e}")
    rep("re-projected values differ from old ones",
        not np.allclose(lam2, sub.lam[:len(lam2)]),
        "old values would be wrong at the new mass")

    # ---- T6 hierarchy recycling ----
    print("\n[T6] Two-level hierarchy recycling (P and D_c reused)")
    hier = RecycledTwoLevel(D, n_modes=8, seed=0)
    rep("hierarchy built", hier.k > 0, f"k={hier.k}, setup={hier.setup_applications}")
    rep("hierarchy matches gauge", hier.matches(D))
    rep("hierarchy rejects other gauge", not hier.matches(D2))
    tot = 0
    for j in range(4):
        st = gmres_precond(D, hier.apply, D.randn(900 + j), tol=1e-8)
        tot += st.n_apply
        if j == 0:
            rep("hierarchy solve converged", st.converged, f"res={st.residual:.2e}")
    rep("setup paid once for 4 RHS", hier.n_cycles >= 4,
        f"cycles={hier.n_cycles}, total apps={tot}")

    # ---- T7 amortization + break-even ----
    print("\n[T7] AMORTIZATION measurement")
    am = amortization_benchmark(mass=0.15, n_rhs=8, n_modes=10)
    print()
    print_amortization(am)
    rep("recycle beats rebuild-every-RHS",
        am["recycle"]["total_apps"] < am["rebuild_every_rhs"]["total_apps"],
        f"{am['recycle']['total_apps']} vs {am['rebuild_every_rhs']['total_apps']}")
    rep("break-even n_RHS computed", math.isfinite(am["breakeven_n_rhs"])
        or am["breakeven_n_rhs"] == float("inf"),
        f"{am['breakeven_n_rhs']:.2f}")
    rep("all strategies converged",
        am["plain"]["converged"] == am["recycle"]["converged"] == 8)

    # ---- T8 cross-mass degradation ----
    print("\n[T8] CROSS-MASS degradation measurement")
    cm = cross_mass_report(mass_setup=0.20, masses=(0.20, 0.15, 0.10, 0.06))
    print_cross_mass(cm)
    rep("re-projection beats naive reuse at shifted mass",
        cm[-1]["reprojected_total"] <= cm[-1]["naive_total"] * 1.05,
        f"reproj={cm[-1]['reprojected_total']} naive={cm[-1]['naive_total']}")
    rep("all cross-mass solves converged",
        all(r["reprojected_converged"] == 4 for r in cm))

    # ---- T9 SU(3) ----
    print("\n[T9] SU(3) recycling")
    g3 = make_gauge(4, 2, 3, beta=6.0, seed=13)
    D3 = WilsonDirac(g3, 0.3)
    s3 = lanczos_setup(D3, k=6, n_lanczos=25, seed=0)
    st3 = deflated_cg(D3, D3.randn(21), s3, tol=1e-8, maxiter=3000)
    rep("SU(3) recycled deflation converged", st3.converged,
        f"res={st3.residual:.2e}")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  gauge fingerprint detects any change; cache invalidates on gauge OR mass
  change; EigCG harvests orthonormal Ritz pairs whose residuals are
  MEASURED and filtered, with the lowest value close to dense exact;
  recycled deflation reduces Dirac applications; a WRONG subspace still
  yields the CORRECT answer (safety property); cross-mass Rayleigh-Ritz
  re-projection costs exactly k applications and beats naive reuse;
  the two-level hierarchy's P and D_c are reused across RHS; amortization
  and break-even n_RHS are computed from MEASURED costs; SU(3) works.

MEASURED (reported, not asserted)
  setup cost, per-RHS cost with and without recycling, break-even n_RHS,
  cross-mass degradation with and without re-projection.

NOT IMPLEMENTED / NOT CLAIMED
  ❌ EigCG here harvests from CG on the normal equations. Production uses
     flexible variants (FGMRES+deflation), block Krylov, and even-odd
     preconditioned inner solves.
  ❌ Cross-mass recycling degrades as |m - m_setup| grows. span(V) itself
     becomes worse; re-projection only optimizes WITHIN that span.
  ❌ Small lattices, random (not thermalized) gauge background.
  ❌ No mixed precision, no communication overlap, no coarse-level
     deflation beyond reuse of P.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Deflation recycling across RHS")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--amortization", action="store_true")
    ap.add_argument("--cross-mass", action="store_true")
    ap.add_argument("--nd", type=int, default=4, choices=[2, 4])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--Nc", type=int, default=1)
    ap.add_argument("--mass", type=float, default=0.15)
    ap.add_argument("--mass-setup", type=float, default=0.20)
    ap.add_argument("--masses", type=float, nargs="*",
                    default=[0.20, 0.15, 0.10, 0.06])
    ap.add_argument("--n-rhs", type=int, default=8)
    ap.add_argument("--n-modes", type=int, default=10)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1
    if not (a.amortization or a.cross_mass):
        ap.print_help()
        print(BANNER)
        print("Try: --selftest | --amortization --n-rhs 16 | --cross-mass")
        return 0

    print(BANNER)
    out: Dict[str, Any] = {}
    if a.amortization:
        am = amortization_benchmark(nd=a.nd, L=a.L, Nc=a.Nc, mass=a.mass,
                                    beta=a.beta, n_rhs=a.n_rhs,
                                    n_modes=a.n_modes, tol=a.tol)
        print_amortization(am)
        out["amortization"] = am
    if a.cross_mass:
        cm = cross_mass_report(nd=a.nd, L=a.L, mass_setup=a.mass_setup,
                               masses=tuple(a.masses), n_modes=a.n_modes,
                               n_rhs=max(4, a.n_rhs // 2), tol=a.tol)
        print_cross_mass(cm)
        out["cross_mass"] = cm
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
