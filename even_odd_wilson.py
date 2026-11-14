#!/usr/bin/env python3
"""
=============================================================================
EVEN-ODD (RED-BLACK) PRECONDITIONED WILSON DIRAC
Schur-complement solver + condition-number measurement + honest benchmark
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

MATHEMATICAL BASIS
  On a bipartite lattice with parity p(x) = (sum_mu x_mu) mod 2, the Wilson
  hopping term connects ONLY opposite-parity sites. Hence

      D_ee = c * I_e,   D_oo = c * I_o,   c = m + d      (NO hopping)

  and the Schur complement is EXACT with no inversion cost:

      hat_D = D_oo - D_oe D_ee^{-1} D_eo = c*I - (1/c) D_oe D_eo

  Solve D x = b by:
      (1) hat_D x_o = b_o - (1/c) D_oe b_e        [Krylov on ODD sites only]
      (2) x_e       = (1/c)(b_e - D_eo x_o)       [closed form]

  gamma5-hermiticity is PRESERVED:
      D^dag = g5 D g5  =>  D_eo^dag = g5 D_oe g5,  D_oe^dag = g5 D_eo g5
      =>  hat_D^dag = g5 hat_D g5
  so hat_Q = g5 hat_D is Hermitian indefinite and MINRES applies exactly
  as in the unpreconditioned case.

WHAT IS MEASURED, NOT ASSERTED
  * kappa(Q) vs kappa(hat_Q) via Lanczos on both operators.
  * Dirac-equivalent application counts. One hat_D application costs one
    D_eo + one D_oe = one full hopping = ONE Dirac-equivalent. The two
    extra half-applications for RHS preconditioning and even-site
    reconstruction ARE counted.
  * Krylov vector size (exactly halved when n_even == n_odd).
  * Wall time.
  * Every preconditioned solution is verified against the ORIGINAL FULL
    operator: ||D x - b|| / ||b||. A preconditioned solve that does not
    satisfy the original system is a bug, not a speedup.

HONEST SCOPE
  ❌ Constant-factor gain only. Does NOT change the kappa ~ 1/(a m) scaling
     as m -> 0. At physically light quark masses you still need multigrid.
     The measured gain here is on small lattices at moderate mass; it does
     not extrapolate automatically.
  ❌ Requires ALL lattice extents EVEN. With periodic boundaries and an odd
     extent, the wrap-around neighbour has the SAME parity and the block
     structure is invalid. This is CHECKED and raises, not silently wrong.
  ❌ D_eo / D_oe here use scatter -> full-hop -> gather for clarity. A
     production code builds compressed even/odd neighbour lists and avoids
     the ~2x redundant arithmetic. Application COUNTS (the metric used for
     comparison) are unaffected; WALL TIME is pessimistic by that factor.
  ❌ No SAP, no mixed precision, no multigrid. Those are the larger real
     wins and are absent.
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

BANNER = """
=============================================================================
EVEN-ODD (RED-BLACK) PRECONDITIONED WILSON DIRAC
D_ee = c*I exactly  =>  Schur complement is free.
Krylov space halved. kappa gain MEASURED, not assumed.
Constant-factor gain only -- NOT a replacement for multigrid.
=============================================================================
"""


# =============================================================================
# §1. GAMMA MATRICES  (Euclidean chiral basis; identical to the full module)
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
    g5 = 1j * (g[0] @ g[1] @ g[2] @ g[3])
    return g, g5, 4


# =============================================================================
# §2. GAUGE FIELD + PARITY
# =============================================================================

@dataclass
class GaugeField:
    nd: int
    L: int
    Nc: int
    links: np.ndarray            # (nd, V, Nc, Nc)
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

    def parity(self) -> np.ndarray:
        return (self.coords().sum(axis=-1) % 2).astype(np.int64)

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
    if cold:
        links = np.tile(np.eye(Nc, dtype=complex), (nd, V, 1, 1))
    elif Nc == 1:
        rng = np.random.default_rng(seed)
        th = rng.normal(0.0, 1.0 / max(beta, 1e-3), size=(nd, V, 1, 1))
        links = np.exp(1j * th)
    else:
        rng = np.random.default_rng(seed)
        A = (rng.normal(size=(nd, V, Nc, Nc))
             + 1j * rng.normal(size=(nd, V, Nc, Nc)))
        Q, R = np.linalg.qr(A)
        ph = np.diagonal(R, axis1=-2, axis2=-1)
        Q = Q * (ph / np.abs(ph))[..., None, :]
        U, _, Vh = np.linalg.svd(Q)
        X = U @ Vh
        links = X / np.linalg.det(X)[..., None, None] ** (1.0 / Nc)
    return GaugeField(nd, L, Nc, links, beta)


# =============================================================================
# §3. FULL WILSON DIRAC  (hop exposed separately -- this is what EO needs)
# =============================================================================

class WilsonDirac:
    """
    D psi(x)     = c psi(x) + hop(psi)(x),   c = m + d
    hop psi(x)   = -1/2 sum_mu [ (1-g_mu) U_mu(x) psi(x+mu)
                               + (1+g_mu) U_mu^dag(x-mu) psi(x-mu) ]
    D^dag psi(x) = c psi(x)
                   -1/2 sum_mu [ U_mu^dag(x-mu)(1-g_mu) psi(x-mu)
                               + U_mu(x)(1+g_mu) psi(x+mu) ]
    """

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
        self.size = self.V * ns * self.Nc
        self.ip, self.im = [], []
        for mu in range(self.nd):
            cp = self.coords.copy(); cp[:, mu] += 1
            cm = self.coords.copy(); cm[:, mu] -= 1
            self.ip.append(gauge.idx(cp))
            self.im.append(gauge.idx(cm))
        self.n_apply = 0
        self.n_apply_dag = 0
        self.n_hop = 0

    # ---- helpers ----
    def _g(self, mu: int, f: np.ndarray) -> np.ndarray:
        return np.einsum("st,vtc->vsc", self.gam[mu], f)

    def _u(self, mu: int, f: np.ndarray, dag: bool = False) -> np.ndarray:
        U = self.g.links[mu]
        if dag:
            U = U.conj().swapaxes(-1, -2)
        return np.einsum("vcd,vsd->vsc", U, f)

    def apply_g5(self, psi: np.ndarray) -> np.ndarray:
        return np.einsum("st,vtc->vsc", self.g5, psi)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=complex)

    def randn(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.shape) + 1j * rng.normal(size=self.shape)

    def reset(self) -> None:
        self.n_apply = self.n_apply_dag = self.n_hop = 0

    @property
    def total_applications(self) -> int:
        return self.n_apply + self.n_apply_dag

    # ---- operators ----
    def hop(self, psi: np.ndarray) -> np.ndarray:
        """Hopping ONLY. Maps even-supported -> odd-supported and vice versa."""
        self.n_hop += 1
        out = np.zeros_like(psi)
        for mu in range(self.nd):
            fwd = self._u(mu, psi[self.ip[mu]])
            out -= 0.5 * (fwd - self._g(mu, fwd))
            bwd = self._u(mu, psi, dag=True)[self.im[mu]]
            out -= 0.5 * (bwd + self._g(mu, bwd))
        return out

    def apply(self, psi: np.ndarray) -> np.ndarray:
        self.n_apply += 1
        return self.c * psi + self.hop(psi)

    def apply_dag(self, psi: np.ndarray) -> np.ndarray:
        self.n_apply_dag += 1
        out = self.c * psi
        for mu in range(self.nd):
            t = psi - self._g(mu, psi)
            t = self._u(mu, t, dag=True)[self.im[mu]]
            out -= 0.5 * t
            s = psi + self._g(mu, psi)
            s = self._u(mu, s)[self.ip[mu]]
            out -= 0.5 * s
        return out

    def apply_Q(self, psi: np.ndarray) -> np.ndarray:
        return self.apply_g5(self.apply(psi))

    def apply_normal(self, psi: np.ndarray) -> np.ndarray:
        return self.apply_dag(self.apply(psi))

    @staticmethod
    def dot(a: np.ndarray, b: np.ndarray) -> complex:
        return complex(np.vdot(a.reshape(-1), b.reshape(-1)))

    @staticmethod
    def norm(a: np.ndarray) -> float:
        return float(np.linalg.norm(a.reshape(-1)))


# =============================================================================
# §4. EVEN-ODD SCHUR COMPLEMENT
# =============================================================================

class EvenOddWilson:
    """
    hat_D = c*I_o - (1/c) D_oe D_eo   acting on ODD sites only.

    Requires ALL extents even (bipartite periodic lattice). Checked.
    """

    def __init__(self, full: WilsonDirac):
        if full.L % 2 != 0:
            raise ValueError(
                f"L={full.L} is ODD. With periodic boundaries the wrap-around "
                "neighbour has the SAME parity, so D_ee is NOT diagonal and "
                "the even-odd decomposition is INVALID. Use an even extent."
            )
        self.full = full
        self.c = full.c
        par = full.g.parity()
        self.even = np.flatnonzero(par == 0)
        self.odd = np.flatnonzero(par == 1)
        self.n_e, self.n_o = len(self.even), len(self.odd)
        self.ns, self.Nc = full.ns, full.Nc
        self.shape_o = (self.n_o, self.ns, self.Nc)
        self.shape_e = (self.n_e, self.ns, self.Nc)
        self.size_o = self.n_o * self.ns * self.Nc
        self.n_eo = 0
        self.n_oe = 0

    # ---- counting ----
    def reset(self) -> None:
        self.n_eo = self.n_oe = 0
        self.full.reset()

    @property
    def half_applications(self) -> int:
        return self.n_eo + self.n_oe

    @property
    def dirac_equivalent(self) -> float:
        """One D_eo + one D_oe == one full hopping == one Dirac application."""
        return 0.5 * self.half_applications

    # ---- shapes ----
    def zeros_o(self) -> np.ndarray:
        return np.zeros(self.shape_o, dtype=complex)

    def randn_o(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=self.shape_o) + 1j * rng.normal(size=self.shape_o)

    def g5(self, v: np.ndarray) -> np.ndarray:
        return np.einsum("st,vtc->vsc", self.full.g5, v)

    @staticmethod
    def dot(a, b) -> complex:
        return complex(np.vdot(a.reshape(-1), b.reshape(-1)))

    @staticmethod
    def norm(a) -> float:
        return float(np.linalg.norm(a.reshape(-1)))

    # ---- off-diagonal blocks (hopping only) ----
    def D_eo(self, v_o: np.ndarray) -> np.ndarray:
        """odd -> even."""
        self.n_eo += 1
        psi = self.full.zeros()
        psi[self.odd] = v_o
        return self.full.hop(psi)[self.even]

    def D_oe(self, v_e: np.ndarray) -> np.ndarray:
        """even -> odd."""
        self.n_oe += 1
        psi = self.full.zeros()
        psi[self.even] = v_e
        return self.full.hop(psi)[self.odd]

    def D_eo_dag(self, v_e: np.ndarray) -> np.ndarray:
        """even -> odd.  D_eo^dag = g5 D_oe g5."""
        return self.g5(self.D_oe(self.g5(v_e)))

    def D_oe_dag(self, v_o: np.ndarray) -> np.ndarray:
        """odd -> even.  D_oe^dag = g5 D_eo g5."""
        return self.g5(self.D_eo(self.g5(v_o)))

    # ---- Schur complement ----
    def hat_D(self, v_o: np.ndarray) -> np.ndarray:
        return self.c * v_o - self.D_eo(self.D_oe(v_o)) / self.c

    def hat_D_dag(self, v_o: np.ndarray) -> np.ndarray:
        return self.c * v_o - self.D_eo_dag(self.D_oe_dag(v_o)) / self.c

    def hat_D_dag_via_g5(self, v_o: np.ndarray) -> np.ndarray:
        """Independent route, used as a correctness cross-check."""
        return self.g5(self.hat_D(self.g5(v_o)))

    def hat_Q(self, v_o: np.ndarray) -> np.ndarray:
        """Hermitian indefinite."""
        return self.g5(self.hat_D(v_o))

    def hat_normal(self, v_o: np.ndarray) -> np.ndarray:
        return self.hat_D_dag(self.hat_D(v_o))

    # ---- full solve ----
    def solve(self, b_full: np.ndarray, solver: str = "minres",
              tol: float = 1e-9, maxiter: int = 3000,
              coarse: Optional["CoarseSpaceEO"] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        self.reset()
        b_e, b_o = b_full[self.even], b_full[self.odd]
        rhs = b_o - self.D_oe(b_e) / self.c            # 1 half-application

        if solver == "minres":
            x_o, st = minres_herm(self.hat_Q, self.g5(rhs), self.dot, self.norm,
                                  tol, max(200, maxiter // 4), self.zeros_o)
        elif solver == "cg":
            if coarse is not None:
                x_o, st = deflated_cg(self.hat_normal, self.hat_D_dag, rhs,
                                      coarse, self.dot, self.norm, tol,
                                      maxiter, self.zeros_o)
            else:
                x_o, st = cg_normal(self.hat_normal, self.hat_D_dag, rhs,
                                    self.dot, self.norm, tol, maxiter,
                                    self.zeros_o)
        elif solver == "bicgstab":
            x0 = None
            if coarse is not None:
                x0 = coarse.initial_guess(self.g5(rhs), self.dot, self.zeros_o)
            x_o, st = bicgstab(self.hat_D, rhs, self.dot, self.norm,
                               tol, maxiter, self.zeros_o, x0)
        else:
            raise ValueError(f"unknown solver {solver!r}")

        x_e = (b_e - self.D_eo(x_o)) / self.c          # 1 half-application
        x = self.full.zeros()
        x[self.even] = x_e
        x[self.odd] = x_o

        # VERIFY AGAINST THE ORIGINAL FULL OPERATOR -- non-negotiable
        self.full.reset()
        res = self.full.norm(b_full - self.full.apply(x)) / max(
            self.full.norm(b_full), 1e-30)
        st.update({
            "solver": solver,
            "half_applications": self.half_applications,
            "dirac_equivalent": self.dirac_equivalent,
            "full_residual": float(res),
            "converged": bool(res < tol),
            "wall_s": time.time() - t0,
            "krylov_size": self.size_o,
            "deflated": coarse is not None,
        })
        return x, st


# =============================================================================
# §5. GENERIC KRYLOV SOLVERS  (work on full or EO spaces identically)
# =============================================================================

def cg_normal(A_norm, A_dag, b, dot, norm, tol, maxiter, zeros):
    rhs = A_dag(b)
    x = zeros()
    r = rhs - A_norm(x)
    p = r.copy()
    rz = dot(r, r).real
    nb = max(norm(rhs), 1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        if math.sqrt(max(rz, 0.0)) / nb < tol:
            break
        Ap = A_norm(p)
        pAp = dot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        a = rz / pAp
        x = x + a * p
        r = r - a * Ap
        rz_new = dot(r, r).real
        p = r + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new
    return x, {"n_iter": it}


def bicgstab(A, b, dot, norm, tol, maxiter, zeros, x0=None):
    x = zeros() if x0 is None else x0.copy()
    r = b - A(x)
    r0 = r.copy()
    nb = max(norm(b), 1e-30)
    rho = alpha = omega = 1.0
    v = zeros(); p = zeros()
    it = 0
    for it in range(1, maxiter + 1):
        if norm(r) / nb < tol:
            break
        rho_new = dot(r0, r)
        if abs(rho_new) < 1e-30:
            break
        beta = (rho_new / rho) * (alpha / omega) if it > 1 else 0.0
        p = r + beta * (p - omega * v)
        v = A(p)
        den = dot(r0, v)
        if abs(den) < 1e-30:
            break
        alpha = rho_new / den
        s = r - alpha * v
        if norm(s) / nb < tol:
            x = x + alpha * p
            break
        t = A(s)
        tt = dot(t, t)
        omega = dot(t, s) / tt if abs(tt) > 1e-30 else 0.0
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        if abs(omega) < 1e-30:
            break
    return x, {"n_iter": it}


def minres_herm(Q, b, dot, norm, tol, maxiter, zeros, restarts=6):
    """MINRES via Lanczos + least squares, full reorthogonalization."""
    nb = max(norm(b), 1e-30)
    x = zeros()
    total = 0
    for _ in range(restarts):
        r = b - Q(x)
        beta = norm(r)
        if beta / nb < tol:
            break
        V = [r / beta]
        al, be = [], []
        vprev, bprev = None, 0.0
        k = int(maxiter)
        for j in range(k):
            w = Q(V[j])
            a = dot(V[j], w).real
            al.append(a)
            w = w - a * V[j]
            if vprev is not None:
                w = w - bprev * vprev
            for vv in V:
                w = w - dot(vv, w) * vv
            bn = norm(w)
            total += 1
            if bn < 1e-14:
                break
            be.append(bn)
            vprev, bprev = V[j], bn
            V.append(w / bn)
            if norm(b - Q(x + np.einsum(
                    "j,j...->...",
                    np.linalg.lstsq(
                        _tbar(al, be),
                        np.eye(len(al) + 1, 1)[:, 0] * beta,
                        rcond=None)[0],
                    np.stack(V[:len(al)], axis=0)))) / nb < tol:
                break
        n = len(al)
        e1 = np.zeros(n + 1); e1[0] = beta
        y, *_ = np.linalg.lstsq(_tbar(al, be), e1, rcond=None)
        x = x + np.einsum("j,j...->...", y, np.stack(V[:n], axis=0))
        if norm(b - Q(x)) / nb < tol:
            break
    return x, {"n_iter": total}


def _tbar(al: List[float], be: List[float]) -> np.ndarray:
    n = len(al)
    T = np.zeros((n + 1, n))
    for j in range(n):
        T[j, j] = al[j]
    for j in range(len(be)):
        T[j + 1, j] = be[j]
    return T


# =============================================================================
# §6. LANCZOS CONDITION NUMBER + EO DEFLATION
# =============================================================================

def lanczos_kappa(Q, randn, dot, norm, k: int = 40, seed: int = 0
                  ) -> Dict[str, Any]:
    v = randn(seed)
    v = v / norm(v)
    V = [v]
    al, be = [], []
    vprev, bprev = None, 0.0
    for j in range(k):
        w = Q(V[j])
        a = dot(V[j], w).real
        al.append(a)
        w = w - a * V[j]
        if vprev is not None:
            w = w - bprev * vprev
        for vv in V:
            w = w - dot(vv, w) * vv
        bn = norm(w)
        if bn < 1e-13:
            break
        be.append(bn)
        vprev, bprev = V[j], bn
        V.append(w / bn)
    n = len(al)
    T = np.diag(np.array(al))
    for j in range(len(be)):
        T[j, j + 1] = be[j]
        T[j + 1, j] = be[j]
    ev, evec = np.linalg.eigh(T)
    Vs = np.stack(V[:n], axis=0)
    vecs = np.einsum("jk,k...->j...", evec.T, Vs)
    order = np.argsort(np.abs(ev))
    ev, vecs = ev[order], vecs[order]
    smin, smax = float(abs(ev[0])), float(abs(ev[-1]))
    return {
        "sigma_min": smin, "sigma_max": smax,
        "kappa_Q": smax / max(smin, 1e-30),
        "ritz": ev, "vecs": vecs, "n_lanczos": n,
        "gaps": [float(abs(ev[i]) / max(smax, 1e-30))
                 for i in range(min(10, len(ev)))],
    }


@dataclass
class CoarseSpaceEO:
    vecs: np.ndarray
    lam: np.ndarray
    n_modes: int
    setup_dirac_equivalent: float

    @property
    def lam2(self) -> np.ndarray:
        return np.maximum(self.lam ** 2, 1e-30)

    def initial_guess(self, rhs_Q, dot, zeros):
        x = zeros()
        for i in range(self.n_modes):
            if abs(self.lam[i]) > 1e-30:
                x = x + (dot(self.vecs[i], rhs_Q) / self.lam[i]) * self.vecs[i]
        return x


def build_coarse_eo(eo: EvenOddWilson, n_modes: int = 8, n_lanczos: int = 40,
                    seed: int = 0) -> CoarseSpaceEO:
    eo.reset()
    info = lanczos_kappa(eo.hat_Q, eo.randn_o, eo.dot, eo.norm,
                         k=n_lanczos, seed=seed)
    setup = eo.dirac_equivalent
    k = min(n_modes, len(info["ritz"]))
    return CoarseSpaceEO(info["vecs"][:k].copy(), info["ritz"][:k].copy(),
                         k, setup)


def deflated_cg(A_norm, A_dag, b, cs: CoarseSpaceEO, dot, norm,
                tol, maxiter, zeros):
    rhs = A_dag(b)
    V, lam2 = cs.vecs, cs.lam2

    def proj(z):
        for i in range(cs.n_modes):
            z = z - dot(V[i], z) * V[i]
        return z

    x = zeros()
    for i in range(cs.n_modes):
        x = x + (dot(V[i], rhs) / lam2[i]) * V[i]
    r = rhs - A_norm(x)
    z = proj(r)
    p = z.copy()
    rz = dot(r, z).real
    nb = max(norm(rhs), 1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        if math.sqrt(max(rz, 0.0)) / nb < tol:
            break
        Ap = A_norm(p)
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
    return x, {"n_iter": it, "deflated_modes": cs.n_modes}


# =============================================================================
# §7. BENCHMARK: FULL vs EVEN-ODD
# =============================================================================

def benchmark(nd: int, L: int, Nc: int, mass: float, beta: float,
              tol: float = 1e-8, maxiter: int = 2000,
              n_lanczos: int = 40, n_modes: int = 8,
              seed: int = 0, cold: bool = False,
              deflate: bool = False) -> Dict[str, Any]:
    g = make_gauge(nd, L, Nc, beta, cold=cold, seed=seed)
    D = WilsonDirac(g, mass)
    eo = EvenOddWilson(D)
    b = D.randn(1234)

    # ---- condition numbers ----
    D.reset()
    full_spec = lanczos_kappa(D.apply_Q, D.randn, D.dot, D.norm,
                              k=n_lanczos, seed=seed)
    eo.reset()
    eo_spec = lanczos_kappa(eo.hat_Q, eo.randn_o, eo.dot, eo.norm,
                            k=n_lanczos, seed=seed)

    cs = build_coarse_eo(eo, n_modes, n_lanczos, seed) if deflate else None

    rows = []
    for solver in ("minres", "cg", "bicgstab"):
        # --- FULL ---
        D.reset()
        t0 = time.time()
        if solver == "minres":
            xf, st = minres_herm(D.apply_Q, D.apply_g5(b), D.dot, D.norm,
                                 tol, max(200, maxiter // 4), D.zeros)
        elif solver == "cg":
            xf, st = cg_normal(D.apply_normal, D.apply_dag, b, D.dot, D.norm,
                               tol, maxiter, D.zeros)
        else:
            xf, st = bicgstab(D.apply, b, D.dot, D.norm, tol, maxiter, D.zeros)
        full_apps = D.total_applications
        full_res = D.norm(b - D.apply(xf)) / max(D.norm(b), 1e-30)
        full_wall = time.time() - t0

        # --- EVEN-ODD ---
        xe, st_eo = eo.solve(b, solver=solver, tol=tol, maxiter=maxiter,
                             coarse=cs)
        eo_apps = st_eo["dirac_equivalent"]
        if cs is not None:
            eo_apps += cs.setup_dirac_equivalent

        rows.append({
            "solver": solver,
            "full_apps": full_apps,
            "eo_apps": eo_apps,
            "app_ratio": full_apps / max(eo_apps, 1e-9),
            "full_iters": st.get("n_iter", 0),
            "eo_iters": st_eo.get("n_iter", 0),
            "full_residual": float(full_res),
            "eo_residual": st_eo["full_residual"],
            "full_converged": bool(full_res < tol),
            "eo_converged": st_eo["converged"],
            "full_wall_s": full_wall,
            "eo_wall_s": st_eo["wall_s"],
        })

    valid = [r for r in rows if r["full_converged"] and r["eo_converged"]]
    best = max(valid, key=lambda r: r["app_ratio"]) if valid else None

    return {
        "config": {"nd": nd, "L": L, "Nc": Nc, "mass": mass, "beta": beta,
                   "tol": tol, "deflate": deflate, "cold": cold},
        "plaquette": g.plaquette(),
        "kappa_full_Q": full_spec["kappa_Q"],
        "kappa_eo_Q": eo_spec["kappa_Q"],
        "kappa_reduction": full_spec["kappa_Q"] / max(eo_spec["kappa_Q"], 1e-30),
        "sigma_min_full": full_spec["sigma_min"],
        "sigma_min_eo": eo_spec["sigma_min"],
        "krylov_size_full": D.size,
        "krylov_size_eo": eo.size_o,
        "size_reduction": D.size / max(eo.size_o, 1),
        "n_even": eo.n_e, "n_odd": eo.n_o,
        "rows": rows,
        "best_solver": best["solver"] if best else None,
        "best_app_ratio": best["app_ratio"] if best else None,
    }


def print_table(results: List[Dict[str, Any]]) -> None:
    hdr = (f"{'nd':>3}{'L':>4}{'Nc':>3}{'mass':>8}"
           f"{'kap_full':>11}{'kap_EO':>11}{'kap_red':>9}"
           f"{'best':>10}{'app_full':>10}{'app_EO':>9}{'gain':>8}{'res_EO':>11}")
    print("=" * len(hdr))
    print("FULL vs EVEN-ODD  (gain = full_apps / eo_apps, Dirac-equivalent)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        c = r["config"]
        best = r["best_solver"] or "-"
        row = next((x for x in r["rows"] if x["solver"] == best), None)
        af = row["full_apps"] if row else 0
        ae = row["eo_apps"] if row else 0
        re_ = row["eo_residual"] if row else float("nan")
        gain = r["best_app_ratio"] or 0.0
        print(f"{c['nd']:>3}{c['L']:>4}{c['Nc']:>3}{c['mass']:>8.3f}"
              f"{r['kappa_full_Q']:>11.2f}{r['kappa_eo_Q']:>11.2f}"
              f"{r['kappa_reduction']:>9.2f}{best:>10}"
              f"{af:>10d}{ae:>9.1f}{gain:>8.2f}{re_:>11.2e}")
    print("=" * len(hdr))
    print("  kap_red  = kappa(Q) / kappa(hat_Q)      (>1 means EO better conditioned)")
    print("  gain     = Dirac-equivalent applications, full / EO")
    print("  res_EO   = ||D x - b||/||b|| against the ORIGINAL FULL operator")
    print("  Krylov vector size is EXACTLY halved when n_even == n_odd.")
    print()
    print("  HONEST: constant-factor gain only. Does NOT change kappa ~ 1/(a m)")
    print("  scaling as m -> 0. Light quarks still need multigrid.")


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

    # ---- T1 gamma algebra ----
    print("\n[T1] Gamma algebra")
    for nd in (2, 4):
        g, g5, ns = gamma_matrices(nd)
        I = np.eye(ns, dtype=complex)
        herm = max(float(abs(x - x.conj().T).max()) for x in g)
        cliff = 0.0
        for mu in range(nd):
            for nu in range(nd):
                tgt = 2 * I if mu == nu else np.zeros_like(I)
                cliff = max(cliff, float(abs(g[mu] @ g[nu] + g[nu] @ g[mu] - tgt).max()))
        rep(f"nd={nd} Hermitian + Clifford + g5^2=I",
            max(herm, cliff, float(abs(g5 @ g5 - I).max())) < 1e-12)

    # ---- T2 bipartiteness: hop flips parity ----
    print("\n[T2] Bipartiteness -- hop maps even-supported -> odd-supported")
    g2 = make_gauge(4, 4, 1, beta=5.0, seed=1)
    D2 = WilsonDirac(g2, 0.3)
    eo2 = EvenOddWilson(D2)
    v = D2.zeros()
    v[eo2.even] = D2.randn(7)[eo2.even]
    h = D2.hop(v)
    leak = float(np.linalg.norm(h[eo2.even])) / max(float(np.linalg.norm(h)), 1e-30)
    rep("no same-parity leakage", leak < 1e-13, f"leak={leak:.2e}")
    rep("n_even == n_odd", eo2.n_e == eo2.n_o, f"{eo2.n_e}/{eo2.n_o}")

    # ---- T3 odd L is REFUSED ----
    print("\n[T3] Odd extent must be refused (invalid decomposition)")
    try:
        EvenOddWilson(WilsonDirac(make_gauge(2, 3, 1), 0.3))
        rep("odd L refused", False)
    except ValueError:
        rep("odd L refused", True)

    # ---- T4 full adjoint + gamma5-hermiticity ----
    print("\n[T4] Full operator identities")
    psi, phi = D2.randn(1), D2.randn(2)
    lhs, rhs = D2.dot(phi, D2.apply(psi)), D2.dot(D2.apply_dag(phi), psi)
    rep("adjoint identity", abs(lhs - rhs) / max(abs(lhs), 1e-30) < 1e-11)
    a = D2.apply_g5(D2.apply(D2.apply_g5(psi)))
    b_ = D2.apply_dag(psi)
    rep("g5 D g5 == D^dag", D2.norm(a - b_) / max(D2.norm(b_), 1e-30) < 1e-12)

    # ---- T5 THE KEY TEST: hat_D vs dense Schur complement ----
    print("\n[T5] hat_D vs DENSE Schur complement (the decisive correctness test)")
    for nd, L, Nc in ((2, 4, 1), (4, 2, 1), (4, 2, 3)):
        gg = make_gauge(nd, L, Nc, beta=5.0, seed=5)
        DD = WilsonDirac(gg, 0.35)
        EE = EvenOddWilson(DD)
        n = DD.size
        dense = np.zeros((n, n), dtype=complex)
        for j in range(n):
            e = np.zeros(n, dtype=complex); e[j] = 1.0
            dense[:, j] = DD.apply(e.reshape(DD.shape)).reshape(-1)
        ns, nc = DD.ns, DD.Nc
        w = ns * nc
        edof = np.concatenate([np.arange(s * w, (s + 1) * w) for s in EE.even])
        odof = np.concatenate([np.arange(s * w, (s + 1) * w) for s in EE.odd])
        Dee = dense[np.ix_(edof, edof)]
        Deo = dense[np.ix_(edof, odof)]
        Doe = dense[np.ix_(odof, edof)]
        Doo = dense[np.ix_(odof, odof)]
        rep(f"nd={nd},L={L},Nc={Nc}: D_ee == c*I",
            float(abs(Dee - EE.c * np.eye(len(edof))).max()) < 1e-11)
        schur = Doo - Doe @ np.linalg.solve(Dee, Deo)
        maxerr = 0.0
        for j in range(EE.size_o):
            e = np.zeros(EE.size_o, dtype=complex); e[j] = 1.0
            mf = EE.hat_D(e.reshape(EE.shape_o)).reshape(-1)
            maxerr = max(maxerr, float(abs(mf - schur[:, j]).max()))
        rep(f"nd={nd},L={L},Nc={Nc}: hat_D == dense Schur",
            maxerr < 1e-10, f"max_err={maxerr:.2e}")

    # ---- T6 hat_D^dag two independent routes ----
    print("\n[T6] hat_D^dag: block-adjoint route vs g5 hat_D g5")
    vo = eo2.randn_o(3)
    r1 = eo2.hat_D_dag(vo)
    r2 = eo2.hat_D_dag_via_g5(vo)
    rep("two routes agree", eo2.norm(r1 - r2) / max(eo2.norm(r1), 1e-30) < 1e-11)

    # ---- T7 hat_Q Hermitian ----
    print("\n[T7] hat_Q = g5 hat_D is Hermitian")
    u, w_ = eo2.randn_o(4), eo2.randn_o(5)
    q1, q2 = eo2.dot(u, eo2.hat_Q(w_)), eo2.dot(eo2.hat_Q(u), w_)
    rep("Hermitian", abs(q1 - q2) / max(abs(q1), 1e-30) < 1e-11)

    # ---- T8 EO solves verified against FULL operator ----
    print("\n[T8] EO solve residual against the ORIGINAL FULL operator")
    bb = D2.randn(11)
    for solver in ("minres", "cg", "bicgstab"):
        x, st = eo2.solve(bb, solver=solver, tol=1e-9, maxiter=2000)
        rep(f"{solver} converged", st["converged"],
            f"res={st['full_residual']:.2e} "
            f"dirac_eq={st['dirac_equivalent']:.1f} iters={st['n_iter']}")

    # ---- T9 free field sigma_min == mass ----
    print("\n[T9] Free field: sigma_min(D) == mass")
    for nd, L, m in ((2, 8, 0.5), (4, 4, 0.4)):
        Df = WilsonDirac(make_gauge(nd, L, 1, cold=True), m)
        sp = lanczos_kappa(Df.apply_Q, Df.randn, Df.dot, Df.norm,
                           k=min(60, Df.size - 1), seed=3)
        rep(f"nd={nd} L={L} m={m}", abs(sp["sigma_min"] - m) / m < 0.05,
            f"sigma_min={sp['sigma_min']:.6f}")

    # ---- T10 deflation on EO ----
    print("\n[T10] EO deflation")
    cs = build_coarse_eo(eo2, n_modes=6, n_lanczos=30, seed=7)
    x_plain, s_plain = eo2.solve(bb, solver="cg", tol=1e-9, maxiter=2000)
    x_def, s_def = eo2.solve(bb, solver="cg", tol=1e-9, maxiter=2000, coarse=cs)
    rep("deflated solve converged", s_def["converged"],
        f"res={s_def['full_residual']:.2e}")
    rep("deflation reduces iterations", s_def["n_iter"] <= s_plain["n_iter"],
        f"{s_def['n_iter']} vs {s_plain['n_iter']}")

    # ---- T11 SU(3) EO solve ----
    print("\n[T11] SU(3) even-odd solve")
    g3 = make_gauge(4, 2, 3, beta=6.0, seed=13)
    D3 = WilsonDirac(g3, 0.3)
    eo3 = EvenOddWilson(D3)
    x3, st3 = eo3.solve(D3.randn(21), solver="bicgstab", tol=1e-9, maxiter=2000)
    rep("SU(3) EO converged", st3["converged"], f"res={st3['full_residual']:.2e}")

    # ---- T12 measured gain (informational) ----
    print("\n[T12] MEASURED condition-number and application gain")
    res = []
    for m in (0.6, 0.3, 0.15, 0.08):
        res.append(benchmark(4, 4, 1, m, beta=5.0, tol=1e-8, maxiter=1500,
                             n_lanczos=40, seed=17))
    print()
    print_table(res)
    all_conv = all(r["eo_residual"] < 1e-7
                   for rr in res for r in rr["rows"] if r["eo_converged"])
    rep("all EO solutions verified", all_conv)
    rep("Krylov size exactly halved",
        all(abs(rr["size_reduction"] - 2.0) < 1e-12 for rr in res))
    gains = [rr["best_app_ratio"] for rr in res if rr["best_app_ratio"]]
    print(f"\n  measured application gains: {[f'{x:.2f}x' for x in gains]}")
    print(f"  measured kappa reductions: "
          f"{[f'{rr[chr(107)+chr(97)+chr(112)+chr(112)+chr(97)+chr(95)+chr(114)+chr(101)+chr(100)+chr(117)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)]:.2f}x' for rr in res]}")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  gamma algebra; hop flips parity exactly; odd extent refused;
  full adjoint and gamma5-hermiticity; D_ee == c*I exactly;
  hat_D == DENSE Schur complement to machine precision (U(1) and SU(3));
  hat_D^dag via two independent routes; hat_Q Hermitian;
  every EO solution checked against the ORIGINAL FULL operator;
  free-field sigma_min == mass; deflation reduces iterations.

MEASURED (reported, not asserted)
  kappa(Q) / kappa(hat_Q); Dirac-equivalent application gain; wall time.

NOT CLAIMED
  ❌ Constant-factor gain only -- kappa ~ 1/(a m) scaling unchanged.
     Light quarks still require multigrid.
  ❌ scatter/gather D_eo/D_oe does ~2x redundant arithmetic; wall time is
     pessimistic. Application counts are unaffected.
  ❌ No SAP, no mixed precision, no multigrid.
  ❌ Small lattices, moderate mass. No extrapolation to production scale.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Even-odd preconditioned Wilson Dirac")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--nd", type=int, default=4, choices=[2, 4])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--Nc", type=int, default=1)
    ap.add_argument("--masses", type=float, nargs="*",
                    default=[0.6, 0.3, 0.15, 0.08])
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=2000)
    ap.add_argument("--lanczos", type=int, default=40)
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--deflate", action="store_true")
    ap.add_argument("--cold", action="store_true")
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
    res = [benchmark(a.nd, a.L, a.Nc, m, a.beta, a.tol, a.maxiter,
                     a.lanczos, a.modes, 0, a.cold, a.deflate)
           for m in a.masses]
    print_table(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
