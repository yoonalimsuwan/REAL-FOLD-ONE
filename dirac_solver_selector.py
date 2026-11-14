#!/usr/bin/env python3
"""
=============================================================================
ADAPTIVE SOLVER SELECTOR FOR LATTICE DIRAC INVERSION
Wilson Dirac (U(1) / SU(Nc)) + gamma5-hermiticity + cost-model selector
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

PHYSICS — WHAT IS ACTUALLY IMPLEMENTED
  * Wilson Dirac operator, matrix-free, d in {2,4}, Nc >= 1:
        D psi(x) = (m + d) psi(x)
                   - 1/2 sum_mu [ (1-gamma_mu) U_mu(x) psi(x+mu)
                                + (1+gamma_mu) U_mu^dag(x-mu) psi(x-mu) ]
  * Exact gamma5-hermiticity:  D^dag = gamma5 D gamma5   (TESTED, not assumed)
  * Q = gamma5 D is Hermitian INDEFINITE. Therefore:
        - CG is NOT applicable to D or to Q directly.
        - CG applies to D^dag D (HPD), but kappa(D^dag D) = kappa(Q)^2.
        - MINRES applies to Q (Hermitian indefinite), 1 Dirac apply/iter.
        - BiCGStab applies to D directly (non-Hermitian), 2 applies/iter.
  * Lanczos on Q gives condition-number features and near-null vectors.
  * Deflated CG / deflated BiCGStab use those near-null vectors as a coarse
    space. This is the coarse-space ingredient of adaptive algebraic
    multigrid, NOT a full DD-alphaAMG implementation.

HONEST SCOPE
  ❌ "multigrid" here = two-level DEFLATION, not full adaptive AMG.
     A production DD-alphaAMG needs: Schwarz alternating smoother,
     block aggregation, adaptive setup with test-vector relaxation,
     multiple levels, and a flexible outer Krylov (FGMRES/GCR).
     None of that is implemented. The deflation option captures the
     coarse-space benefit only.
  ❌ MINRES here is Lanczos + least-squares with FULL reorthogonalization.
     Mathematically equivalent to MINRES, but memory O(k*V). Production
     uses the short-recurrence Paige-Saunders form.
  ❌ No even-odd (red-black) preconditioning, no SAP, no mixed precision.
     These are the largest real-world speedups and are absent here.
  ❌ Cost-model exponents are CALIBRATED FROM MEASUREMENTS, not asserted.
     Uncalibrated defaults are labelled as such.
  ❌ Small lattices only in the self-test. No continuum limit, no
     dynamical fermions (quenched gauge background).
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


BANNER = """
=============================================================================
ADAPTIVE DIRAC SOLVER SELECTOR
CG applies to D^dag D only. MINRES to gamma5 D. BiCGStab to D.
"multigrid" here = two-level deflation, NOT full DD-alphaAMG.
Cost is counted in DIRAC APPLICATIONS, not iterations.
=============================================================================
"""


# =============================================================================
# §1. GAMMA MATRICES  (Euclidean, chiral basis)
# =============================================================================

def pauli() -> List[np.ndarray]:
    return [
        np.array([[1, 0], [0, 1]], dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]


def gamma_matrices(nd: int) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    Returns (gammas, gamma5, n_spin).
    nd=2: gamma0=sigma1, gamma1=sigma2, gamma5=sigma3   (n_spin=2)
    nd=4: chiral basis gamma_mu = [[0, sigma_mu],[barsigma_mu, 0]]  (n_spin=4)
    All gammas Hermitian; Clifford algebra {g_mu,g_nu}=2 delta_mu nu.
    """
    s = pauli()
    if nd == 2:
        g = [s[0].copy(), s[1].copy()]
        g5 = s[2].copy()
        return g, g5, 2
    if nd != 4:
        raise ValueError("nd must be 2 or 4")
    I2 = s[0]
    sig = [I2, 1j * s[1], 1j * s[2], 1j * s[3]]        # sigma_mu = (I, i*sigma_k)
    barsig = [I2, -1j * s[1], -1j * s[2], -1j * s[3]]  # barsigma_mu
    g = []
    for mu in range(4):
        gm = np.zeros((4, 4), dtype=complex)
        gm[:2, 2:] = sig[mu]
        gm[2:, :2] = barsig[mu]
        g.append(gm)
    g5 = 1j * (g[0] @ g[1] @ g[2] @ g[3])
    return g, g5, 4


def check_clifford(nd: int) -> Dict[str, float]:
    g, g5, ns = gamma_matrices(nd)
    I = np.eye(ns, dtype=complex)
    herm = max(float(np.abs(x - x.conj().T).max()) for x in g)
    cliff = 0.0
    for mu in range(nd):
        for nu in range(nd):
            target = 2.0 * I if mu == nu else np.zeros_like(I)
            cliff = max(cliff, float(np.abs(g[mu] @ g[nu] + g[nu] @ g[mu] - target).max()))
    g5sq = float(np.abs(g5 @ g5 - I).max())
    g5herm = float(np.abs(g5 - g5.conj().T).max())
    anti = 0.0
    for mu in range(nd):
        anti = max(anti, float(np.abs(g5 @ g[mu] + g[mu] @ g5).max()))
    return {"hermiticity": herm, "clifford": cliff, "g5_squared": g5sq,
            "g5_hermitian": g5herm, "g5_anticommute": anti}


# =============================================================================
# §2. GAUGE FIELD  (U(1) if Nc=1, SU(Nc) otherwise)
# =============================================================================

@dataclass
class GaugeField:
    nd: int
    L: int
    Nc: int
    links: np.ndarray          # (nd, V, Nc, Nc) complex, V = L**nd
    beta: float = 6.0

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

    def plaquette(self) -> float:
        """Average real trace of the elementary plaquette / Nc."""
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


def _project_su(M: np.ndarray) -> np.ndarray:
    """Project to SU(Nc) via SVD + determinant phase."""
    U, _, Vh = np.linalg.svd(M)
    X = U @ Vh
    det = np.linalg.det(X)
    return X / det[..., None, None] ** (1.0 / M.shape[-1])


def make_gauge(nd: int, L: int, Nc: int = 1, beta: float = 6.0,
               cold: bool = False, seed: int = 0) -> GaugeField:
    """
    cold=True  -> all links identity (free field; used for exact tests)
    cold=False -> random gauge (U(1) phases, or SU(Nc) via QR of Ginibre)
    NOTE: random is NOT a thermalized ensemble. For a real background,
    load links from lattice_heatbath_rk3.py (see load_su3_npz).
    """
    V = L ** nd
    if cold:
        links = np.tile(np.eye(Nc, dtype=complex), (nd, V, 1, 1))
    elif Nc == 1:
        rng = np.random.default_rng(seed)
        # beta controls roughness: theta ~ N(0, 1/beta)
        th = rng.normal(0.0, 1.0 / max(beta, 1e-3), size=(nd, V, 1, 1))
        links = np.exp(1j * th)
    else:
        rng = np.random.default_rng(seed)
        A = (rng.normal(size=(nd, V, Nc, Nc))
             + 1j * rng.normal(size=(nd, V, Nc, Nc)))
        Q, R = np.linalg.qr(A)
        ph = np.diagonal(R, axis1=-2, axis2=-1)
        Q = Q * (ph / np.abs(ph))[..., None, :]
        links = _project_su(Q)
    return GaugeField(nd=nd, L=L, Nc=Nc, links=links, beta=beta)


def load_su3_npz(path: str | Path, nd: int = 4) -> GaugeField:
    """
    Load links written by lattice_heatbath_rk3.py:
        np.savez_compressed(..., links=U, ...)  with U shape (D, N, 3, 3)
    """
    z = np.load(path)
    U = np.asarray(z["links"], dtype=complex)
    if U.ndim != 4 or U.shape[-1] != 3:
        raise ValueError(f"expected (D,N,3,3), got {U.shape}")
    D, N = U.shape[0], U.shape[1]
    L = int(round(N ** (1.0 / D)))
    if L ** D != N:
        raise ValueError(f"N={N} is not L**{D}")
    beta = float(z["beta"]) if "beta" in z else 6.0
    return GaugeField(nd=D, L=L, Nc=3, links=U, beta=beta)


# =============================================================================
# §3. WILSON DIRAC OPERATOR  (matrix-free, with application counter)
# =============================================================================

class WilsonDirac:
    """
    Matrix-free Wilson Dirac operator with r=1, a=1.

      D psi(x)      = (m+d) psi(x)
                      - 1/2 sum_mu [ (1-g_mu) U_mu(x) psi(x+mu)
                                   + (1+g_mu) U_mu^dag(x-mu) psi(x-mu) ]
      D^dag psi(x)  = (m+d) psi(x)
                      - 1/2 sum_mu [ U_mu^dag(x-mu)(1-g_mu) psi(x-mu)
                                   + U_mu(x)(1+g_mu) psi(x+mu) ]

    gamma acts on SPIN, U acts on COLOR -> they commute. This is exactly
    why gamma5 D gamma5 = D^dag holds. Verified numerically in self-test.

    Field layout: psi has shape (V, n_spin, Nc).
    """

    def __init__(self, gauge: GaugeField, mass: float = 0.1):
        self.g = gauge
        self.m = float(mass)
        self.nd, self.L, self.Nc = gauge.nd, gauge.L, gauge.Nc
        gam, g5, ns = gamma_matrices(self.nd)
        self.gam, self.g5, self.ns = gam, g5, ns
        self.c = gauge.coords()
        self.V = gauge.V
        self.shape = (self.V, ns, self.Nc)
        self.size = self.V * ns * self.Nc
        # neighbour indices
        self.ip = []
        self.im = []
        for mu in range(self.nd):
            cp = self.c.copy(); cp[:, mu] += 1
            cm = self.c.copy(); cm[:, mu] -= 1
            self.ip.append(gauge.idx(cp))
            self.im.append(gauge.idx(cm))
        self.n_applications = 0
        self.n_dag_applications = 0

    # ---- helpers ----
    def _gapply(self, mu: int, f: np.ndarray) -> np.ndarray:
        return np.einsum("st,vtc->vsc", self.gam[mu], f)

    def _uapply(self, mu: int, f: np.ndarray, dag: bool = False) -> np.ndarray:
        U = self.g.links[mu]
        if dag:
            U = U.conj().swapaxes(-1, -2)
        return np.einsum("vcd,vsd->vsc", U, f)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=complex)

    def randn(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.normal(size=self.shape) + 1j * rng.normal(size=self.shape))

    def reset_counters(self) -> None:
        self.n_applications = 0
        self.n_dag_applications = 0

    @property
    def total_applications(self) -> int:
        return self.n_applications + self.n_dag_applications

    # ---- operators ----
    def apply(self, psi: np.ndarray) -> np.ndarray:
        self.n_applications += 1
        out = (self.m + self.nd) * psi
        for mu in range(self.nd):
            fwd = psi[self.ip[mu]]
            fwd = self._uapply(mu, fwd)
            out = out - 0.5 * (fwd - self._gapply(mu, fwd))
            bwd = self._uapply(mu, psi, dag=True)[self.im[mu]]
            out = out - 0.5 * (bwd + self._gapply(mu, bwd))
        return out

    def apply_dag(self, psi: np.ndarray) -> np.ndarray:
        self.n_dag_applications += 1
        out = (self.m + self.nd) * psi
        for mu in range(self.nd):
            t = psi[self.im[mu]]
            t = t - self._gapply(mu, t)
            out = out - 0.5 * self._uapply(mu, t, dag=True)[self.im[mu]] \
                if False else out
            # correct ordering: U^dag(x-mu) (1-g) psi(x-mu)
            t2 = psi - self._gapply(mu, psi)
            t2 = self._uapply(mu, t2, dag=True)[self.im[mu]]
            out = out - 0.5 * t2
            s2 = psi + self._gapply(mu, psi)
            s2 = self._uapply(mu, s2)[self.ip[mu]]
            out = out - 0.5 * s2
        return out

    def apply_g5(self, psi: np.ndarray) -> np.ndarray:
        return np.einsum("st,vtc->vsc", self.g5, psi)

    def apply_Q(self, psi: np.ndarray) -> np.ndarray:
        """Q = gamma5 D  (Hermitian, indefinite). 1 D-application."""
        return self.apply_g5(self.apply(psi))

    def apply_normal(self, psi: np.ndarray) -> np.ndarray:
        """D^dag D (HPD). 2 D-applications."""
        return self.apply_dag(self.apply(psi))

    # ---- inner products ----
    @staticmethod
    def dot(a: np.ndarray, b: np.ndarray) -> complex:
        return complex(np.vdot(a.reshape(-1), b.reshape(-1)))

    @staticmethod
    def norm(a: np.ndarray) -> float:
        return float(np.linalg.norm(a.reshape(-1)))


# =============================================================================
# §4. LANCZOS ON Q  ->  condition-number features + near-null vectors
# =============================================================================

@dataclass
class SpectralInfo:
    ritz: np.ndarray              # Ritz values of Q (real, ascending by |.|)
    ritz_vecs: np.ndarray         # (k, V, ns, Nc) corresponding vectors
    kappa_Q: float
    sigma_min: float
    sigma_max: float
    n_lanczos: int
    n_applications: int
    gap_ratios: List[float] = field(default_factory=list)


def lanczos_Q(D: WilsonDirac, k: int = 30, seed: int = 0,
              reorth: bool = True) -> SpectralInfo:
    """
    Lanczos on the Hermitian operator Q = gamma5 D, with full
    reorthogonalization (small problems). Ritz values approximate
    eigenvalues of Q; |lambda| are singular values of D.

      kappa(Q)      = max|lambda| / min|lambda|
      kappa(D^dag D)= kappa(Q)^2
    """
    k = int(min(k, D.size - 1))
    v = D.randn(seed)
    v /= D.norm(v)
    V = [v]
    alphas, betas = [], []
    v_prev = None
    beta = 0.0
    D.reset_counters()
    for j in range(k):
        w = D.apply_Q(V[j])
        a = float(D.dot(V[j], w).real)
        alphas.append(a)
        w = w - a * V[j]
        if v_prev is not None:
            w = w - beta * v_prev
        if reorth:
            for vv in V:
                w = w - D.dot(vv, w) * vv
        b = D.norm(w)
        if b < 1e-13:
            break
        betas.append(b)
        v_prev = V[j]
        V.append(w / b)
        beta = b
    n = len(alphas)
    T = np.diag(np.array(alphas))
    for j in range(len(betas)):
        T[j, j + 1] = betas[j]
        T[j + 1, j] = betas[j]
    ev, evec = np.linalg.eigh(T)
    Vs = np.stack(V[:n], axis=0)
    rvecs = np.einsum("jk,k...->j...", evec.T, Vs)
    order = np.argsort(np.abs(ev))
    ev, rvecs = ev[order], rvecs[order]
    smin = float(np.abs(ev[0]))
    smax = float(np.abs(ev[-1]))
    kappa = smax / max(smin, 1e-30)
    gaps = [float(abs(ev[i]) / max(smax, 1e-30)) for i in range(min(8, len(ev)))]
    return SpectralInfo(ritz=ev, ritz_vecs=rvecs, kappa_Q=kappa,
                        sigma_min=smin, sigma_max=smax, n_lanczos=n,
                        n_applications=D.total_applications,
                        gap_ratios=gaps)


# =============================================================================
# §5. SOLVERS  (all count Dirac applications via the operator)
# =============================================================================

@dataclass
class SolveResult:
    solver: str
    converged: bool
    n_iter: int
    residual: float
    n_applications: int
    wall_s: float
    setup_applications: int = 0
    x: Optional[np.ndarray] = None


def _finish(D, name, x, b, t0, n_iter, tol, setup=0, keep=False) -> SolveResult:
    r = D.norm(b - D.apply(x)) / max(D.norm(b), 1e-30)
    return SolveResult(solver=name, converged=bool(r < tol), n_iter=n_iter,
                       residual=float(r),
                       n_applications=D.total_applications,
                       wall_s=time.time() - t0,
                       setup_applications=setup,
                       x=x if keep else None)


def cg_normal(D: WilsonDirac, b: np.ndarray, tol: float = 1e-8,
              maxiter: int = 3000, x0: Optional[np.ndarray] = None
              ) -> SolveResult:
    """CG on A = D^dag D, rhs = D^dag b  (HPD). 2 applies/iter."""
    t0 = time.time(); D.reset_counters()
    rhs = D.apply_dag(b)
    x = D.zeros() if x0 is None else x0.copy()
    r = rhs - D.apply_normal(x)
    p = r.copy()
    rz = D.dot(r, r).real
    nb = max(D.norm(rhs), 1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        if math.sqrt(max(rz, 0)) / nb < tol:
            break
        Ap = D.apply_normal(p)
        pAp = D.dot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rz_new = D.dot(r, r).real
        p = r + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new
    return _finish(D, "cg_normal", x, b, t0, it, tol)


def bicgstab(D: WilsonDirac, b: np.ndarray, tol: float = 1e-8,
             maxiter: int = 3000, x0: Optional[np.ndarray] = None
             ) -> SolveResult:
    """BiCGStab on D directly (non-Hermitian). 2 applies/iter."""
    t0 = time.time(); D.reset_counters()
    x = D.zeros() if x0 is None else x0.copy()
    r = b - D.apply(x)
    r0 = r.copy()
    nb = max(D.norm(b), 1e-30)
    rho = alpha = omega = 1.0
    v = D.zeros(); p = D.zeros()
    it = 0
    for it in range(1, maxiter + 1):
        if D.norm(r) / nb < tol:
            break
        rho_new = D.dot(r0, r)
        if abs(rho_new) < 1e-30:
            break
        beta = (rho_new / rho) * (alpha / omega) if it > 1 else 0.0
        p = r + beta * (p - omega * v)
        v = D.apply(p)
        denom = D.dot(r0, v)
        if abs(denom) < 1e-30:
            break
        alpha = rho_new / denom
        s = r - alpha * v
        if D.norm(s) / nb < tol:
            x = x + alpha * p
            r = s
            break
        t = D.apply(s)
        tt = D.dot(t, t)
        omega = D.dot(t, s) / tt if abs(tt) > 1e-30 else 0.0
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        if abs(omega) < 1e-30:
            break
    return _finish(D, "bicgstab", x, b, t0, it, tol)


def minres_Q(D: WilsonDirac, b: np.ndarray, tol: float = 1e-8,
             maxiter: int = 600, restart_tol: float = 1e-30
             ) -> SolveResult:
    """
    MINRES on Q = gamma5 D (Hermitian indefinite), rhs = gamma5 b.
    Implemented as Lanczos + least-squares with FULL reorthogonalization.
    Mathematically equivalent to MINRES; memory O(k*V). Production uses
    the short-recurrence Paige-Saunders form.
    1 D-application per Lanczos step.
    """
    t0 = time.time(); D.reset_counters()
    rhs = D.apply_g5(b)
    nb = max(D.norm(rhs), 1e-30)
    x = D.zeros()
    total_it = 0
    for _ in range(6):                       # restarts
        r = rhs - D.apply_Q(x)
        beta = D.norm(r)
        if beta / nb < tol:
            break
        V = [r / beta]
        alphas, betas = [], []
        v_prev, bprev = None, 0.0
        k = int(min(maxiter, D.size - 1))
        for j in range(k):
            w = D.apply_Q(V[j])
            a = float(D.dot(V[j], w).real)
            alphas.append(a)
            w = w - a * V[j]
            if v_prev is not None:
                w = w - bprev * v_prev
            for vv in V:
                w = w - D.dot(vv, w) * vv
            bn = D.norm(w)
            total_it += 1
            if bn < restart_tol:
                break
            betas.append(bn)
            v_prev, bprev = V[j], bn
            V.append(w / bn)
        n = len(alphas)
        Tbar = np.zeros((n + 1, n))
        for j in range(n):
            Tbar[j, j] = alphas[j]
        for j in range(len(betas)):
            Tbar[j + 1, j] = betas[j]
        e1 = np.zeros(n + 1); e1[0] = beta
        y, *_ = np.linalg.lstsq(Tbar, e1, rcond=None)
        Vs = np.stack(V[:n], axis=0)
        x = x + np.einsum("j,j...->...", y, Vs)
        if D.norm(rhs - D.apply_Q(x)) / nb < tol:
            break
    return _finish(D, "minres_Q", x, b, t0, total_it, tol)


# ---- two-level (deflation) -------------------------------------------------

@dataclass
class CoarseSpace:
    """Near-null eigenvectors of Q -> low modes of D^dag D (lambda^2)."""
    vecs: np.ndarray            # (k, V, ns, Nc)
    lam_Q: np.ndarray           # (k,) real eigenvalues of Q
    setup_applications: int
    n_modes: int

    @property
    def lam2(self) -> np.ndarray:
        return self.lam_Q ** 2


def build_coarse_space(D: WilsonDirac, n_modes: int = 8,
                       n_lanczos: int = 40, seed: int = 0) -> CoarseSpace:
    info = lanczos_Q(D, k=n_lanczos, seed=seed)
    k = int(min(n_modes, len(info.ritz)))
    return CoarseSpace(vecs=info.ritz_vecs[:k].copy(),
                       lam_Q=info.ritz[:k].copy(),
                       setup_applications=info.n_applications,
                       n_modes=k)


def deflated_cg(D: WilsonDirac, b: np.ndarray, cs: CoarseSpace,
                tol: float = 1e-8, maxiter: int = 3000) -> SolveResult:
    """
    Deflated CG on A = D^dag D with coarse space V (low modes).
    Projector P = I - V V^dag applied to the residual each step.
    This is the coarse-space ingredient of AMG, NOT full DD-alphaAMG.
    """
    t0 = time.time(); D.reset_counters()
    V, lam2 = cs.vecs, np.maximum(cs.lam2, 1e-30)
    rhs = D.apply_dag(b)

    def proj(z):
        for i in range(cs.n_modes):
            z = z - D.dot(V[i], z) * V[i]
        return z

    # coarse initial guess
    x = D.zeros()
    for i in range(cs.n_modes):
        x = x + (D.dot(V[i], rhs) / lam2[i]) * V[i]
    r = rhs - D.apply_normal(x)
    z = proj(r)
    p = z.copy()
    rz = D.dot(r, z).real
    nb = max(D.norm(rhs), 1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        if math.sqrt(max(rz, 0)) / nb < tol:
            break
        Ap = D.apply_normal(p)
        pAp = D.dot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = proj(r)
        rz_new = D.dot(r, z).real
        p = z + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new
    res = _finish(D, "deflated_cg", x, b, t0, it, tol,
                  setup=cs.setup_applications)
    return res


def deflated_bicgstab(D: WilsonDirac, b: np.ndarray, cs: CoarseSpace,
                      tol: float = 1e-8, maxiter: int = 3000) -> SolveResult:
    """Coarse initial guess from near-null modes, then plain BiCGStab."""
    t0 = time.time(); D.reset_counters()
    V, lam = cs.vecs, cs.lam_Q
    rhsQ = D.apply_g5(b)
    x = D.zeros()
    for i in range(cs.n_modes):
        if abs(lam[i]) > 1e-30:
            x = x + (D.dot(V[i], rhsQ) / lam[i]) * V[i]
    setup = cs.setup_applications
    sub = bicgstab(D, b, tol=tol, maxiter=maxiter, x0=x)
    sub.solver = "deflated_bicgstab"
    sub.setup_applications = setup
    sub.wall_s = time.time() - t0
    return sub


SOLVERS = ("cg_normal", "bicgstab", "minres_Q",
           "deflated_cg", "deflated_bicgstab")


def run_solver(name: str, D: WilsonDirac, b: np.ndarray,
               cs: Optional[CoarseSpace] = None,
               tol: float = 1e-8, maxiter: int = 3000) -> SolveResult:
    if name == "cg_normal":
        return cg_normal(D, b, tol, maxiter)
    if name == "bicgstab":
        return bicgstab(D, b, tol, maxiter)
    if name == "minres_Q":
        return minres_Q(D, b, tol, max(200, maxiter // 4))
    if cs is None:
        cs = build_coarse_space(D)
    if name == "deflated_cg":
        return deflated_cg(D, b, cs, tol, maxiter)
    if name == "deflated_bicgstab":
        return deflated_bicgstab(D, b, cs, tol, maxiter)
    raise ValueError(name)


# =============================================================================
# §6. FEATURES
# =============================================================================

FEATURE_NAMES = [
    "log_kappa_Q", "log_volume", "mass", "nd", "Nc",
    "plaquette", "log_n_rhs", "has_setup", "log_gap_ratio",
]
FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(D: WilsonDirac, spec: SpectralInfo,
                     n_rhs: int = 1, has_setup: bool = False,
                     n_modes: int = 8) -> np.ndarray:
    gap = spec.gap_ratios[min(n_modes, len(spec.gap_ratios) - 1)] \
        if spec.gap_ratios else 1.0
    return np.array([
        math.log(max(spec.kappa_Q, 1.0)),
        math.log(float(D.V)),
        D.m,
        float(D.nd),
        float(D.Nc),
        D.g.plaquette(),
        math.log(float(max(n_rhs, 1))),
        1.0 if has_setup else 0.0,
        math.log(max(gap, 1e-12)),
    ], dtype=np.float64)


# =============================================================================
# §7. ANALYTIC COST MODEL + SELECTOR
# =============================================================================

@dataclass
class CostModel:
    """
    Predicted Dirac applications to reach tolerance.

      n_iter  = c * kappa_eff ** p
      applies = n_iter * applies_per_iter
      kappa_eff = kappa_Q            (no deflation)
                = 1/gap_ratio(k)     (with k-mode deflation)

    Applies/iter are STRUCTURAL facts (2 for CG-on-normal-equations and
    BiCGStab, 1 for MINRES). The coefficients c and exponents p are
    CALIBRATED from measured solves; defaults are labelled uncalibrated.
    """
    coeff: Dict[str, float] = field(default_factory=lambda: {
        "cg_normal": 2.0, "bicgstab": 2.0, "minres_Q": 2.0,
        "deflated_cg": 2.0, "deflated_bicgstab": 2.0})
    expo: Dict[str, float] = field(default_factory=lambda: {
        "cg_normal": 1.0, "bicgstab": 0.7, "minres_Q": 1.0,
        "deflated_cg": 1.0, "deflated_bicgstab": 0.7})
    applies_per_iter: Dict[str, int] = field(default_factory=lambda: {
        "cg_normal": 2, "bicgstab": 2, "minres_Q": 1,
        "deflated_cg": 2, "deflated_bicgstab": 2})
    calibrated: bool = False
    n_calibration_samples: int = 0

    def predict(self, solver: str, kappa_Q: float, gap_ratio: float,
                n_rhs: int, setup_applications: int,
                has_setup: bool) -> float:
        deflated = solver.startswith("deflated")
        keff = max(1.0 / max(gap_ratio, 1e-8), 1.0) if deflated else max(kappa_Q, 1.0)
        n_iter = self.coeff[solver] * keff ** self.expo[solver]
        cost = n_iter * self.applies_per_iter[solver] * max(n_rhs, 1)
        if deflated and not has_setup:
            cost += setup_applications
        return float(cost)

    def predict_all(self, kappa_Q: float, gap_ratio: float, n_rhs: int,
                    setup_applications: int, has_setup: bool
                    ) -> Dict[str, float]:
        return {s: self.predict(s, kappa_Q, gap_ratio, n_rhs,
                                setup_applications, has_setup)
                for s in SOLVERS}


def calibrate_cost_model(records: List[Dict[str, Any]]) -> CostModel:
    """
    Fit log(n_iter) = log(c) + p*log(kappa_eff) per solver, by least squares
    on MEASURED solves. Requires >= 3 converged samples per solver.
    """
    cm = CostModel()
    n_used = 0
    for s in SOLVERS:
        rows = [r for r in records
                if r["solver"] == s and r["converged"] and r["n_iter"] > 0]
        if len(rows) < 3:
            continue
        deflated = s.startswith("deflated")
        xs = [math.log(max(1.0 / max(r["gap_ratio"], 1e-8), 1.0)
                       if deflated else max(r["kappa_Q"], 1.0)) for r in rows]
        ys = [math.log(float(r["n_iter"])) for r in rows]
        if np.std(xs) < 1e-8:
            cm.coeff[s] = float(math.exp(np.mean(ys)))
            n_used += len(rows)
            continue
        p, lc = np.polyfit(xs, ys, 1)
        cm.expo[s] = float(np.clip(p, 0.1, 2.0))
        cm.coeff[s] = float(math.exp(lc))
        n_used += len(rows)
    cm.calibrated = n_used > 0
    cm.n_calibration_samples = n_used
    return cm


class SolverSelector:
    """
    Analytic cost model + optional learned residual correction.
    The learned part is a small ridge on log(measured/predicted); with
    few samples it does nothing, which the ablation reports honestly.
    """

    def __init__(self, model: Optional[CostModel] = None,
                 use_learned: bool = False, ridge: float = 1.0):
        self.model = model or CostModel()
        self.use_learned = use_learned
        self.ridge = ridge
        self.W: Optional[np.ndarray] = None

    def fit_learned(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows = [r for r in records if r["converged"] and r["n_applications"] > 0]
        if len(rows) < FEATURE_DIM + 3:
            return {"status": "insufficient_data", "n": len(rows)}
        onehot = np.zeros((len(rows), len(SOLVERS)))
        for i, r in enumerate(rows):
            onehot[i, SOLVERS.index(r["solver"])] = 1.0
        X = np.hstack([np.stack([r["features"] for r in rows]), onehot])
        X = np.hstack([np.ones((len(X), 1)), X])
        y = np.array([math.log(max(r["n_applications"], 1)
                               / max(r["predicted"], 1.0)) for r in rows])
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        A[0, 0] -= self.ridge
        self.W = np.linalg.solve(A, X.T @ y)
        pred = X @ self.W
        return {"status": "ok", "n": len(rows),
                "train_rmse_log": float(np.sqrt(np.mean((pred - y) ** 2)))}

    def _correction(self, solver: str, feats: np.ndarray) -> float:
        if not self.use_learned or self.W is None:
            return 1.0
        oh = np.zeros(len(SOLVERS)); oh[SOLVERS.index(solver)] = 1.0
        x = np.concatenate([[1.0], feats, oh])
        return float(math.exp(np.clip(x @ self.W, -3.0, 3.0)))

    def select(self, feats: np.ndarray, spec: SpectralInfo,
               n_modes: int, n_rhs: int, has_setup: bool
               ) -> Dict[str, Any]:
        gap = spec.gap_ratios[min(n_modes, len(spec.gap_ratios) - 1)] \
            if spec.gap_ratios else 1.0
        base = self.model.predict_all(spec.kappa_Q, gap, n_rhs,
                                      spec.n_applications, has_setup)
        adj = {s: base[s] * self._correction(s, feats) for s in SOLVERS}
        best = min(adj, key=adj.get)
        return {"selected": best, "predicted_applications": adj,
                "uncalibrated": not self.model.calibrated,
                "learned_active": bool(self.use_learned and self.W is not None)}


# =============================================================================
# §8. BENCHMARK HARNESS
# =============================================================================

def measure_all(nd: int, L: int, Nc: int, mass: float, beta: float,
                n_rhs: int = 1, n_modes: int = 8, n_lanczos: int = 30,
                tol: float = 1e-7, maxiter: int = 1500,
                seed: int = 0, cold: bool = False) -> Dict[str, Any]:
    """Run every solver on the SAME (gauge, mass, rhs). Measure real cost."""
    g = make_gauge(nd, L, Nc, beta, cold=cold, seed=seed)
    D = WilsonDirac(g, mass)
    spec = lanczos_Q(D, k=n_lanczos, seed=seed)
    cs = build_coarse_space(D, n_modes=n_modes, n_lanczos=n_lanczos, seed=seed)
    gap = spec.gap_ratios[min(n_modes, len(spec.gap_ratios) - 1)]
    rhs_list = [D.randn(1000 + i) for i in range(n_rhs)]

    results: Dict[str, SolveResult] = {}
    for s in SOLVERS:
        tot_app, tot_it, tot_wall, ok = 0, 0, 0.0, True
        last = None
        for b in rhs_list:
            r = run_solver(s, D, b, cs=cs, tol=tol, maxiter=maxiter)
            tot_app += r.n_applications
            tot_it += r.n_iter
            tot_wall += r.wall_s
            ok &= r.converged
            last = r
        # amortize setup across rhs for deflated solvers
        if s.startswith("deflated"):
            tot_app += cs.setup_applications
        results[s] = SolveResult(
            solver=s, converged=ok, n_iter=tot_it // max(n_rhs, 1),
            residual=last.residual if last else float("nan"),
            n_applications=tot_app, wall_s=tot_wall,
            setup_applications=cs.setup_applications if s.startswith("deflated") else 0)

    feats = extract_features(D, spec, n_rhs=n_rhs, has_setup=False, n_modes=n_modes)
    return {"config": {"nd": nd, "L": L, "Nc": Nc, "mass": mass,
                       "beta": beta, "n_rhs": n_rhs, "n_modes": n_modes},
            "kappa_Q": spec.kappa_Q, "sigma_min": spec.sigma_min,
            "sigma_max": spec.sigma_max, "gap_ratio": gap,
            "plaquette": g.plaquette(), "features": feats,
            "results": results}


def build_records(measurements: List[Dict[str, Any]],
                  model: CostModel) -> List[Dict[str, Any]]:
    recs = []
    for m in measurements:
        for s, r in m["results"].items():
            recs.append({
                "solver": s, "converged": r.converged, "n_iter": r.n_iter,
                "n_applications": r.n_applications,
                "kappa_Q": m["kappa_Q"], "gap_ratio": m["gap_ratio"],
                "features": m["features"],
                "predicted": model.predict(
                    s, m["kappa_Q"], m["gap_ratio"],
                    m["config"]["n_rhs"],
                    lanczos_cost_estimate(m), False),
            })
    return recs


def lanczos_cost_estimate(m: Dict[str, Any]) -> int:
    return int(m["config"]["n_modes"] * 4 + 20)


def benchmark_selector(measurements: List[Dict[str, Any]],
                       model: CostModel,
                       selector: SolverSelector) -> Dict[str, Any]:
    """Compare selector against every fixed-solver baseline."""
    n = len(measurements)
    if n == 0:
        return {"status": "no_data"}
    totals = {s: 0 for s in SOLVERS}
    sel_total, sel_ok = 0, 0
    picks: Dict[str, int] = {}
    for m in measurements:
        cfg = m["config"]
        g = make_gauge(cfg["nd"], cfg["L"], cfg["Nc"], cfg["beta"], seed=0)
        D = WilsonDirac(g, cfg["mass"])
        spec = lanczos_Q(D, k=30, seed=0)
        feats = m["features"]
        dec = selector.select(feats, spec, cfg["n_modes"], cfg["n_rhs"], False)
        s = dec["selected"]
        picks[s] = picks.get(s, 0) + 1
        r = m["results"][s]
        sel_total += r.n_applications
        sel_ok += int(r.converged)
        for k in SOLVERS:
            totals[k] += m["results"][k].n_applications
    best_fixed = min(totals, key=totals.get)
    return {
        "status": "ok", "n_configs": n,
        "selector_total_applications": sel_total,
        "selector_converged": sel_ok,
        "fixed_totals": totals,
        "best_fixed_baseline": best_fixed,
        "speedup_vs_best_fixed": totals[best_fixed] / max(sel_total, 1),
        "picks": picks,
        "verdict": ("SELECTOR WINS" if sel_total < 0.97 * totals[best_fixed]
                    else "NO GAIN over best fixed solver"),
        "honest_note": (
            "Speedup is measured in Dirac applications on these configs only. "
            "It does not transfer automatically to other volumes, masses, "
            "or gauge ensembles. Retrain/calibrate on your own workload."),
    }


def print_table(measurements: List[Dict[str, Any]]) -> None:
    hdr = (f"{'nd':>3}{'L':>4}{'Nc':>3}{'mass':>8}{'kappa_Q':>11}"
           + "".join(f"{s[:11]:>13}" for s in SOLVERS) + f"{'best':>13}")
    print("=" * len(hdr))
    print("MEASURED DIRAC APPLICATIONS (lower is better; * = did not converge)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for m in measurements:
        c = m["config"]
        row = f"{c['nd']:>3}{c['L']:>4}{c['Nc']:>3}{c['mass']:>8.3f}{m['kappa_Q']:>11.1f}"
        best, bv = None, float("inf")
        for s in SOLVERS:
            r = m["results"][s]
            mark = "" if r.converged else "*"
            row += f"{str(r.n_applications)+mark:>13}"
            if r.converged and r.n_applications < bv:
                bv, best = r.n_applications, s
        row += f"{(best or '-')[:13]:>13}"
        print(row)
    print("=" * len(hdr))


# =============================================================================
# §9. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ---- T1 gamma algebra ----
    print("\n[T1] Gamma matrix algebra")
    for nd in (2, 4):
        c = check_clifford(nd)
        rep(f"nd={nd} Hermitian + Clifford + g5",
            max(c.values()) < 1e-12,
            f"max_err={max(c.values()):.2e}")

    # ---- T2 adjoint identity ----
    print("\n[T2] D^dag is the true adjoint: <phi, D psi> == <D^dag phi, psi>")
    g = make_gauge(4, 4, Nc=1, beta=4.0, seed=1)
    D = WilsonDirac(g, mass=0.3)
    psi, phi = D.randn(1), D.randn(2)
    lhs = D.dot(phi, D.apply(psi))
    rhs = D.dot(D.apply_dag(phi), psi)
    rel = abs(lhs - rhs) / max(abs(lhs), 1e-30)
    rep("adjoint identity", rel < 1e-11, f"rel_err={rel:.2e}")

    # ---- T3 gamma5-hermiticity ----
    print("\n[T3] gamma5-hermiticity: gamma5 D gamma5 == D^dag")
    a = D.apply_g5(D.apply(D.apply_g5(psi)))
    b = D.apply_dag(psi)
    err = D.norm(a - b) / max(D.norm(b), 1e-30)
    rep("gamma5 D gamma5 == D^dag", err < 1e-12, f"rel_err={err:.2e}")

    # ---- T4 Q Hermitian ----
    print("\n[T4] Q = gamma5 D is Hermitian")
    q1 = D.dot(phi, D.apply_Q(psi))
    q2 = D.dot(D.apply_Q(phi), psi)
    rep("Q Hermitian", abs(q1 - q2) / max(abs(q1), 1e-30) < 1e-11,
        f"rel_err={abs(q1-q2)/max(abs(q1),1e-30):.2e}")

    # ---- T5 free-field sigma_min == mass ----
    print("\n[T5] Free field (U=I): sigma_min(D) == mass")
    for nd, L, m in ((2, 8, 0.5), (4, 4, 0.4)):
        gf = make_gauge(nd, L, Nc=1, cold=True)
        Df = WilsonDirac(gf, mass=m)
        sp = lanczos_Q(Df, k=min(60, Df.size - 1), seed=3)
        rel = abs(sp.sigma_min - m) / m
        rep(f"nd={nd} L={L} m={m}", rel < 0.05,
            f"sigma_min={sp.sigma_min:.6f} vs m={m}")

    # ---- T6 all solvers converge ----
    print("\n[T6] Every solver reaches the tolerance")
    g = make_gauge(4, 4, Nc=1, beta=5.0, seed=7)
    D = WilsonDirac(g, mass=0.25)
    bb = D.randn(11)
    cs = build_coarse_space(D, n_modes=6, n_lanczos=30, seed=7)
    conv = {}
    for s in SOLVERS:
        r = run_solver(s, D, bb, cs=cs, tol=1e-8, maxiter=2000)
        conv[s] = r
        rep(f"{s} converged", r.converged,
            f"res={r.residual:.2e} iters={r.n_iter} applies={r.n_applications}")

    # ---- T7 MINRES cheaper per iteration ----
    print("\n[T7] MINRES uses fewer Dirac applications than CG-on-normal-eq")
    rep("minres_Q applies < cg_normal applies",
        conv["minres_Q"].n_applications < conv["cg_normal"].n_applications,
        f"{conv['minres_Q'].n_applications} vs {conv['cg_normal'].n_applications}")

    # ---- T8 deflation reduces iterations ----
    print("\n[T8] Deflation reduces iteration count")
    rep("deflated_cg iters <= cg_normal iters",
        conv["deflated_cg"].n_iter <= conv["cg_normal"].n_iter,
        f"{conv['deflated_cg'].n_iter} vs {conv['cg_normal'].n_iter}")

    # ---- T9 Lanczos kappa vs exact (tiny lattice) ----
    print("\n[T9] Lanczos kappa_Q vs dense exact (2D L=4)")
    gt = make_gauge(2, 4, Nc=1, beta=5.0, seed=5)
    Dt = WilsonDirac(gt, mass=0.4)
    n = Dt.size
    Qm = np.zeros((n, n), dtype=complex)
    for j in range(n):
        e = np.zeros(n, dtype=complex); e[j] = 1.0
        Qm[:, j] = Dt.apply_Q(e.reshape(Dt.shape)).reshape(-1)
    ev = np.linalg.eigvalsh(Qm)
    kap_exact = float(np.abs(ev).max() / np.abs(ev).min())
    sp = lanczos_Q(Dt, k=min(n - 1, 60), seed=5)
    rel = abs(sp.kappa_Q - kap_exact) / kap_exact
    rep("kappa_Q within 5% of exact", rel < 0.05,
        f"lanczos={sp.kappa_Q:.3f} exact={kap_exact:.3f}")

    # ---- T10 mass dependence of kappa ----
    print("\n[T10] kappa_Q grows as mass decreases (critical slowing down)")
    ks = []
    for m in (0.8, 0.4, 0.15):
        Dm = WilsonDirac(make_gauge(4, 4, Nc=1, beta=5.0, seed=9), mass=m)
        ks.append(lanczos_Q(Dm, k=40, seed=9).kappa_Q)
    rep("kappa monotone increasing as m decreases",
        ks[0] < ks[1] < ks[2], f"{ks[0]:.1f} < {ks[1]:.1f} < {ks[2]:.1f}")

    # ---- T11 SU(3) operator works ----
    print("\n[T11] SU(3) gauge background (Nc=3)")
    g3 = make_gauge(4, 2, Nc=3, beta=6.0, seed=13)
    D3 = WilsonDirac(g3, mass=0.3)
    p3, f3 = D3.randn(1), D3.randn(2)
    e3 = D.norm(D.apply_g5(D3.apply(D3.apply_g5(p3))) - D3.apply_dag(p3)) \
        / max(D3.norm(D3.apply_dag(p3)), 1e-30)
    rep("SU(3) gamma5-hermiticity", e3 < 1e-11, f"rel_err={e3:.2e}")
    r3 = bicgstab(D3, D3.randn(3), tol=1e-8, maxiter=2000)
    rep("SU(3) BiCGStab converges", r3.converged, f"res={r3.residual:.2e}")
    rep("plaquette in (0,1]", 0.0 < g3.plaquette() <= 1.0,
        f"plaq={g3.plaquette():.4f}")

    # ---- T12 selector + benchmark ----
    print("\n[T12] Cost-model calibration + selector benchmark")
    meas = []
    for m in (0.6, 0.3, 0.15):
        meas.append(measure_all(4, 4, 1, m, beta=5.0, n_rhs=1,
                                n_modes=6, n_lanczos=30, tol=1e-7,
                                maxiter=1200, seed=17))
    cm0 = CostModel()
    recs = build_records(meas, cm0)
    cm = calibrate_cost_model(recs)
    rep("cost model calibrated", cm.calibrated,
        f"n_samples={cm.n_calibration_samples}")
    sel = SolverSelector(cm, use_learned=False)
    bm = benchmark_selector(meas, cm, sel)
    rep("benchmark runs", bm["status"] == "ok", bm.get("verdict", ""))
    rep("speedup reported", "speedup_vs_best_fixed" in bm,
        f"{bm.get('speedup_vs_best_fixed', 0):.3f}x vs {bm.get('best_fixed_baseline')}")
    print("\n  Calibrated exponents (measured, not asserted):")
    for s in SOLVERS:
        print(f"    {s:<20} c={cm.coeff[s]:8.3f}  p={cm.expo[s]:6.3f}  "
              f"applies/iter={cm.applies_per_iter[s]}")

    # ---- T13 learned ablation ----
    print("\n[T13] Learned-correction ablation")
    sel_l = SolverSelector(cm, use_learned=True)
    fit = sel_l.fit_learned(recs)
    rep("learned fit attempted", fit["status"] in ("ok", "insufficient_data"),
        fit["status"])
    bm_l = benchmark_selector(meas, cm, sel_l)
    rep("learned benchmark runs", bm_l["status"] == "ok",
        f"analytic={bm['speedup_vs_best_fixed']:.3f}x  "
        f"learned={bm_l['speedup_vs_best_fixed']:.3f}x")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT IS VERIFIED
  gamma algebra, exact adjoint, exact gamma5-hermiticity, Q Hermitian,
  free-field sigma_min == mass, all solvers converge, Lanczos kappa vs
  dense exact, kappa grows toward the chiral limit, SU(3) works,
  cost-model exponents CALIBRATED from measurements.

WHAT IS NOT
  ❌ "multigrid" = two-level deflation, NOT DD-alphaAMG.
  ❌ No even-odd preconditioning, no SAP, no mixed precision — these are
     the biggest real-world speedups and are absent.
  ❌ MINRES uses full reorthogonalization: memory O(k*V).
  ❌ Gauge backgrounds here are random, not thermalized. Load real links
     with load_su3_npz() from lattice_heatbath_rk3.py.
  ❌ Speedups measured on tiny lattices do not extrapolate.
""")
    return ok


# =============================================================================
# §10. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptive Dirac solver selector")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--nd", type=int, default=4, choices=[2, 4])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--Nc", type=int, default=1)
    ap.add_argument("--masses", type=float, nargs="*",
                    default=[0.6, 0.3, 0.15, 0.08])
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--n-rhs", type=int, default=1)
    ap.add_argument("--n-modes", type=int, default=8)
    ap.add_argument("--lanczos", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--su3-npz", type=str, default=None,
                    help="links npz from lattice_heatbath_rk3.py")
    ap.add_argument("--learned", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if not a.benchmark:
        ap.print_help()
        print(BANNER)
        print("Try: --selftest   or   --benchmark --masses 0.5 0.2 0.1")
        return 0

    print(BANNER)
    meas = []
    if a.su3_npz:
        g = load_su3_npz(a.su3_npz, nd=a.nd)
        print(f"[load] SU(3) links from {a.su3_npz}: L={g.L}, plaq={g.plaquette():.4f}")
        for m in a.masses:
            D = WilsonDirac(g, m)
            spec = lanczos_Q(D, k=a.lanczos)
            cs = build_coarse_space(D, a.n_modes, a.lanczos)
            gap = spec.gap_ratios[min(a.n_modes, len(spec.gap_ratios) - 1)]
            rhs = [D.randn(1000 + i) for i in range(a.n_rhs)]
            res = {}
            for s in SOLVERS:
                tot, it, wall, okc = 0, 0, 0.0, True
                for b in rhs:
                    r = run_solver(s, D, b, cs, a.tol, a.maxiter)
                    tot += r.n_applications; it += r.n_iter
                    wall += r.wall_s; okc &= r.converged
                if s.startswith("deflated"):
                    tot += cs.setup_applications
                res[s] = SolveResult(s, okc, it // max(a.n_rhs, 1), r.residual,
                                     tot, wall)
            meas.append({"config": {"nd": g.nd, "L": g.L, "Nc": 3, "mass": m,
                                    "beta": g.beta, "n_rhs": a.n_rhs,
                                    "n_modes": a.n_modes},
                         "kappa_Q": spec.kappa_Q, "sigma_min": spec.sigma_min,
                         "sigma_max": spec.sigma_max, "gap_ratio": gap,
                         "plaquette": g.plaquette(),
                         "features": extract_features(D, spec, a.n_rhs, False,
                                                      a.n_modes),
                         "results": res})
    else:
        for m in a.masses:
            meas.append(measure_all(a.nd, a.L, a.Nc, m, a.beta, a.n_rhs,
                                    a.n_modes, a.lanczos, a.tol, a.maxiter))

    print_table(meas)
    cm0 = CostModel()
    recs = build_records(meas, cm0)
    cm = calibrate_cost_model(recs)
    sel = SolverSelector(cm, use_learned=a.learned)
    if a.learned:
        print("\n[learned]", sel.fit_learned(recs))
    bm = benchmark_selector(meas, cm, sel)
    print("\nSELECTOR BENCHMARK")
    print(json.dumps(bm, indent=2, default=float))
    print("\nCALIBRATED COST MODEL (from measurements)")
    print(json.dumps({"coeff": cm.coeff, "expo": cm.expo,
                      "applies_per_iter": cm.applies_per_iter,
                      "calibrated": cm.calibrated}, indent=2))

    if a.json:
        out = {"measurements": [
            {k: (v if k != "results"
                 else {s: asdict(r) for s, r in v.items()})
             for k, v in m.items() if k != "features"} for m in meas],
            "cost_model": {"coeff": cm.coeff, "expo": cm.expo},
            "benchmark": bm}
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
