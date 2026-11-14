#!/usr/bin/env python3
"""
=============================================================================
DD-alphaAMG FOR WILSON DIRAC  (two-level, adaptive setup, chiral prolongator)
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT IS IMPLEMENTED (real, tested)
  [1] Block decomposition with red-black coloring (SAP ordering)
  [2] Exact local block operators D_bb built from the Wilson formula
      (intra-block hopping only) -- VERIFIED against restriction of full D
  [3] SAP smoother (Schwarz Alternating Procedure): red-black block sweeps
      with local solves
  [4] ADAPTIVE SETUP: relaxation on the HOMOGENEOUS problem D v = 0 from
      random starts. Slow-to-converge (near-null) components survive.
  [5] CHIRALITY-PRESERVING prolongator: per block, chiral-project the test
      vectors, orthonormalize each sector separately (rank-truncated QR).
      This gives gamma5 P = P gamma5_c, hence the Galerkin coarse operator
      inherits gamma5_c-hermiticity. VERIFIED numerically.
  [6] Galerkin coarse operator D_c = P^dag D P (dense for small coarse dim)
  [7] Two-level cycle: presmooth -> restrict -> coarse solve -> prolong
      -> postsmooth, used as a RIGHT preconditioner inside FGMRES
      (flexible, because the cycle need not be a fixed linear operator)

HONEST SCOPE -- READ BEFORE QUOTING ANY SPEEDUP
  ❌ TWO levels only. A true 3+ level DD-alphaAMG needs a block structure
     and a SAP smoother ON THE COARSE OPERATOR, plus adaptive setup there.
     That is NOT implemented. The coarse level here is a dense direct solve,
     which is only affordable because the test lattices are small.
  ❌ No odd-even preconditioning INSIDE the local solves, no mixed
     precision, no optimized local LU, no block-size tuning, no
     communication overlap. These are large parts of production DD-alphaAMG
     performance and are absent.
  ❌ Prolongator stored DENSE. Production uses block-sparse structure.
     Memory here is O(size * coarse_dim), which does not scale.
  ❌ "Changes mass scaling" is a CLAIM TO BE MEASURED, not asserted.
     The benchmark reports outer iterations and Dirac applications versus
     mass for plain BiCGStab and for AMG+FGMRES. Read those numbers.
  ❌ Setup cost is counted separately and must be amortized over many RHS.
     For a single solve, AMG can easily be SLOWER than plain Krylov.
  ❌ Small lattices only in the self-test. No continuum limit, quenched
     gauge background (random links, not thermalized).
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BANNER = """
=============================================================================
DD-alphaAMG (two-level) FOR WILSON DIRAC
SAP smoother + adaptive setup + chiral prolongator + Galerkin coarse + FGMRES
"Changes mass scaling" is MEASURED below, not assumed.
Two levels only. Setup cost counted separately.
=============================================================================
"""


# =============================================================================
# §1. GAMMA MATRICES + GAUGE + WILSON DIRAC  (same conventions as EO module)
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


@dataclass
class GaugeField:
    nd: int
    L: int
    Nc: int
    links: np.ndarray            # (nd, V, Nc, Nc)

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
        self.size = self.V * ns * self.Nc
        self.w = ns * self.Nc                      # dofs per site
        self.ip, self.im = [], []
        for mu in range(self.nd):
            cp = self.coords.copy(); cp[:, mu] += 1
            cm = self.coords.copy(); cm[:, mu] -= 1
            self.ip.append(gauge.idx(cp))
            self.im.append(gauge.idx(cm))
        self.n_apply = 0

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

    @staticmethod
    def dot(a, b):
        return complex(np.vdot(a.reshape(-1), b.reshape(-1)))

    @staticmethod
    def norm(a):
        return float(np.linalg.norm(a.reshape(-1)))


# =============================================================================
# §2. BLOCK DECOMPOSITION  (red-black colorable)
# =============================================================================

class BlockDecomposition:
    """
    Divide the lattice into blocks of block_L^nd sites.
    Color = parity of the block coordinate sum -> no two same-color blocks
    are nearest neighbours, which is what makes SAP a valid alternating
    Schwarz procedure.
    """

    def __init__(self, nd: int, L: int, block_L: int):
        if L % block_L != 0:
            raise ValueError(f"L={L} not divisible by block_L={block_L}")
        self.nd, self.L, self.bl = nd, L, block_L
        self.nb_dir = L // block_L
        self.n_blocks = self.nb_dir ** nd
        grids = np.meshgrid(*[np.arange(L)] * self.nd, indexing="ij")
        coords = np.stack([g.reshape(-1) for g in grids], -1)
        bcoord = coords // block_L
        bid = np.zeros(len(coords), dtype=np.int64)
        for d in range(nd):
            bid = bid * self.nb_dir + bcoord[:, d]
        self.site_block = bid
        self.color = (bcoord.sum(axis=-1) % 2).astype(np.int64)
        self.block_color = np.zeros(self.n_blocks, dtype=np.int64)
        for b in range(self.n_blocks):
            m = bid == b
            self.block_color[b] = self.color[m][0]
        self.sites = [np.flatnonzero(bid == b) for b in range(self.n_blocks)]
        self.colors = [np.flatnonzero(self.block_color == c) for c in (0, 1)]
        # local index of each site within its block
        self.local_of_site = np.zeros(len(coords), dtype=np.int64)
        for b in range(self.n_blocks):
            self.local_of_site[self.sites[b]] = np.arange(len(self.sites[b]))

    def summary(self) -> Dict[str, Any]:
        return {"n_blocks": self.n_blocks,
                "sites_per_block": len(self.sites[0]),
                "n_red": len(self.colors[0]), "n_black": len(self.colors[1])}


# =============================================================================
# §3. EXACT LOCAL BLOCK OPERATORS  D_bb
# =============================================================================

def build_local_operator(D: WilsonDirac, bd: BlockDecomposition,
                         b: int) -> np.ndarray:
    """
    Dense D_bb: diagonal + INTRA-block hopping only (drops inter-block).
    Built analytically via kron(spin, color), not by probing with unit
    vectors -- O(block_dof^2) construction instead of block_dof applies.

    Forward coupling x -> y=x+mu :  -1/2 (I - g_mu) (x) U_mu(x)
    Backward coupling x -> y=x-mu:  -1/2 (I + g_mu) (x) U_mu^dag(y)
    """
    sites = bd.sites[b]
    ns, Nc, w = D.ns, D.Nc, D.w
    n = len(sites) * w
    M = np.zeros((n, n), dtype=complex)
    I_spin = np.eye(ns, dtype=complex)
    site_pos = {int(s): i for i, s in enumerate(sites)}

    for i, x in enumerate(sites):
        sl = slice(i * w, (i + 1) * w)
        M[sl, sl] += D.c * np.eye(w, dtype=complex)
        for mu in range(D.nd):
            yf = int(D.ip[mu][x])
            if yf in site_pos:
                j = site_pos[yf]
                A = I_spin - D.gam[mu]
                B = D.g.links[mu][x]
                M[sl, slice(j * w, (j + 1) * w)] -= 0.5 * np.kron(A, B)
            yb = int(D.im[mu][x])
            if yb in site_pos:
                j = site_pos[yb]
                A = I_spin + D.gam[mu]
                B = D.g.links[mu][yb].conj().T
                M[sl, slice(j * w, (j + 1) * w)] -= 0.5 * np.kron(A, B)
    return M


class SAPSmoother:
    """
    Schwarz Alternating Procedure: red-black block sweeps with local solves.
    Each sweep: for each color, r = b - D x; delta_b = D_bb^{-1} r_b; x_b += delta.
    """

    def __init__(self, D: WilsonDirac, bd: BlockDecomposition):
        self.D, self.bd = D, bd
        self.locals: List[np.ndarray] = []
        self.factors: List[Any] = []
        for b in range(bd.n_blocks):
            M = build_local_operator(D, bd, b)
            self.locals.append(M)
            self.factors.append(lu_factor(M) if HAS_SCIPY else None)
        self.n_local_solves = 0
        self.dof_slices = []
        for b in range(bd.n_blocks):
            sites = bd.sites[b]
            self.dof_slices.append(
                np.concatenate([np.arange(int(s) * D.w, (int(s) + 1) * D.w)
                                for s in sites]))

    def _solve_local(self, b: int, rhs: np.ndarray) -> np.ndarray:
        self.n_local_solves += 1
        if HAS_SCIPY:
            return lu_solve(self.factors[b], rhs)
        return np.linalg.solve(self.locals[b], rhs)

    def sweep(self, x: np.ndarray, b: np.ndarray, n_sweeps: int = 1
              ) -> np.ndarray:
        xf = x.reshape(-1).copy()
        bf = b.reshape(-1)
        for _ in range(n_sweeps):
            for c in (0, 1):
                r = bf - self.D.apply(xf.reshape(self.D.shape)).reshape(-1)
                for blk in self.bd.colors[c]:
                    idx = self.dof_slices[blk]
                    delta = self._solve_local(int(blk), r[idx])
                    xf[idx] += delta
        return xf.reshape(self.D.shape)

    def reset_counters(self):
        self.n_local_solves = 0


# =============================================================================
# §4. ADAPTIVE SETUP  (relaxation on the homogeneous problem)
# =============================================================================

def adaptive_setup(D: WilsonDirac, sap: SAPSmoother, n_vecs: int = 8,
                   n_relax: int = 6, seed: int = 0,
                   verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate near-null test vectors by applying SAP relaxation to D v = 0
    from random starts. The components SAP converges slowly on are exactly
    the near-null modes; they survive the relaxation and dominate v.

    Returns (V, info) with V shape (size, n_vecs), columns orthonormal.
    """
    D.reset(); sap.reset_counters()
    t0 = time.time()
    cols: List[np.ndarray] = []
    rays: List[float] = []
    for k in range(n_vecs):
        v = D.randn(seed + 100 * k)
        v /= D.norm(v)
        zero = D.zeros()
        for _ in range(n_relax):
            v = sap.sweep(v, zero, n_sweeps=1)
            nv = D.norm(v)
            if nv < 1e-14:
                break
            v = v / nv
        # orthogonalize against previously accepted vectors
        vf = v.reshape(-1)
        for q in cols:
            vf = vf - np.vdot(q, vf) * q
        nq = float(np.linalg.norm(vf))
        if nq < 1e-8:
            continue
        vf = vf / nq
        cols.append(vf)
        r = D.apply(vf.reshape(D.shape))
        rays.append(D.norm(r) / max(D.norm(vf.reshape(D.shape)), 1e-30))
        if verbose:
            print(f"    test vector {len(cols)}: ||Dv||/||v|| = {rays[-1]:.4e}")
    if not cols:
        raise RuntimeError("adaptive setup produced no usable test vectors")
    V = np.stack(cols, axis=1)
    info = {"n_vecs": V.shape[1], "n_relax": n_relax,
            "rayleigh_quotients": rays,
            "max_rayleigh": float(max(rays)),
            "setup_dirac_applies": D.n_apply,
            "setup_local_solves": sap.n_local_solves,
            "setup_wall_s": time.time() - t0}
    return V, info


# =============================================================================
# §5. CHIRALITY-PRESERVING PROLONGATOR + GALERKIN COARSE OPERATOR
# =============================================================================

def chiral_prolongator(D: WilsonDirac, bd: BlockDecomposition,
                       V: np.ndarray, drop_tol: float = 1e-6
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Per block: chiral-project the test vectors, orthonormalize each sector
    separately with rank-truncated QR. Coarse layout: [block][chirality][j].

    Guarantees gamma5 P = P gamma5_c, hence D_c = P^dag D P inherits
    gamma5_c-hermiticity. Both are verified numerically in the self-test.
    """
    ns, Nc, w = D.ns, D.Nc, D.w
    g5_full = np.kron(np.eye(1), D.g5)               # spin only
    Iw = np.eye(w, dtype=complex)
    g5_site = np.kron(D.g5, np.eye(Nc, dtype=complex))
    Pplus = 0.5 * (Iw + g5_site)
    Pminus = 0.5 * (Iw - g5_site)

    blocks_P: List[np.ndarray] = []
    signs: List[float] = []
    ranks: List[Tuple[int, int]] = []

    for b in range(bd.n_blocks):
        sites = bd.sites[b]
        idx = np.concatenate([np.arange(int(s) * w, (int(s) + 1) * w)
                              for s in sites])
        Vb = V[idx, :]
        nb_loc = len(sites)
        # block-level chiral projectors
        Pp = np.zeros((len(idx), len(idx)), dtype=complex)
        Pm = np.zeros_like(Pp)
        for i in range(nb_loc):
            sl = slice(i * w, (i + 1) * w)
            Pp[sl, sl] = Pplus
            Pm[sl, sl] = Pminus
        cols_b: List[np.ndarray] = []
        sgn_b: List[float] = []
        nrank = []
        for proj, sgn in ((Pp, +1.0), (Pm, -1.0)):
            W = proj @ Vb
            Q, R = np.linalg.qr(W)
            keep = np.abs(np.diag(R)) > drop_tol * max(
                float(np.abs(np.diag(R)).max()), 1e-30)
            Qk = Q[:, keep]
            nrank.append(int(Qk.shape[1]))
            for j in range(Qk.shape[1]):
                col = np.zeros(D.size, dtype=complex)
                col[idx] = Qk[:, j]
                cols_b.append(col)
                sgn_b.append(sgn)
        blocks_P.extend(cols_b)
        signs.extend(sgn_b)
        ranks.append((nrank[0], nrank[1]))

    P = np.stack(blocks_P, axis=1)
    g5c = np.array(signs, dtype=float)
    info = {"coarse_dim": P.shape[1], "ranks_per_block": ranks,
            "drop_tol": drop_tol,
            "prolongator_dense_MB": P.nbytes / 1e6}
    return P, g5c, info


def galerkin_coarse(D: WilsonDirac, P: np.ndarray) -> np.ndarray:
    """D_c = P^dag D P, built column by column (coarse_dim Dirac applies)."""
    nc = P.shape[1]
    Dc = np.zeros((nc, nc), dtype=complex)
    for j in range(nc):
        v = P[:, j].reshape(D.shape)
        Dv = D.apply(v).reshape(-1)
        Dc[:, j] = P.conj().T @ Dv
    return Dc


# =============================================================================
# §6. TWO-LEVEL AMG CYCLE + FGMRES
# =============================================================================

@dataclass
class AMGConfig:
    block_L: int = 2
    n_vecs: int = 8
    n_relax: int = 6
    pre_sweeps: int = 2
    post_sweeps: int = 2
    drop_tol: float = 1e-6
    setup_seed: int = 0


class TwoLevelAMG:
    """
    Two-level DD-alphaAMG preconditioner.
    apply(b) ~ one cycle: presmooth -> coarse correction -> postsmooth.
    Coarse level solved DIRECTLY (dense). This is why only two levels.
    """

    def __init__(self, D: WilsonDirac, cfg: AMGConfig):
        self.D, self.cfg = D, cfg
        self.bd = BlockDecomposition(D.nd, D.L, cfg.block_L)
        self.sap = SAPSmoother(D, self.bd)
        t0 = time.time()
        self.V, self.setup_info = adaptive_setup(
            D, self.sap, cfg.n_vecs, cfg.n_relax, cfg.setup_seed)
        self.P, self.g5c, self.prol_info = chiral_prolongator(
            D, self.bd, self.V, cfg.drop_tol)
        D.reset()
        self.Dc = galerkin_coarse(D, self.P)
        self.coarse_applies = D.n_apply
        if HAS_SCIPY:
            self.coarse_lu = lu_factor(self.Dc)
        else:
            self.coarse_lu = None
        self.setup_info["coarse_build_applies"] = self.coarse_applies
        self.setup_info["total_setup_wall_s"] = time.time() - t0
        self.n_cycles = 0
        self.n_coarse_solves = 0

    def apply(self, b: np.ndarray) -> np.ndarray:
        self.n_cycles += 1
        x = self.D.zeros()
        x = self.sap.sweep(x, b, self.cfg.pre_sweeps)
        r = b - self.D.apply(x)
        rc = self.P.conj().T @ r.reshape(-1)
        self.n_coarse_solves += 1
        if HAS_SCIPY:
            ec = lu_solve(self.coarse_lu, rc)
        else:
            ec = np.linalg.solve(self.Dc, rc)
        x = x + (self.P @ ec).reshape(self.D.shape)
        x = self.sap.sweep(x, b, self.cfg.post_sweeps)
        return x

    def stats(self) -> Dict[str, Any]:
        return {"n_cycles": self.n_cycles,
                "n_local_solves": self.sap.n_local_solves,
                "n_coarse_solves": self.n_coarse_solves,
                "coarse_dim": self.Dc.shape[0]}


def fgmres(Aop, Minv, b, dot, norm, zeros, tol=1e-8, restart=20,
           max_outer=60) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Right-preconditioned FGMRES(m). Flexible: Minv may be nonlinear /
    variable (as an AMG cycle with an approximate coarse solve is).
    """
    x = zeros()
    nb = max(norm(b), 1e-30)
    total_it = 0
    for _ in range(max_outer):
        r = b - Aop(x)
        beta = norm(r)
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
            H[j + 1, j] = norm(w)
            total_it += 1
            if abs(H[j + 1, j]) < 1e-14:
                m = j + 1
                break
            V.append(w / H[j + 1, j])
        e1 = np.zeros(m + 1, dtype=complex); e1[0] = beta
        y, *_ = np.linalg.lstsq(H[:m + 1, :m], e1, rcond=None)
        for j in range(m):
            x = x + y[j] * Z[j]
        if norm(b - Aop(x)) / nb < tol:
            break
    res = norm(b - Aop(x)) / nb
    return x, {"n_iter": total_it, "residual": float(res),
               "converged": bool(res < tol)}


def bicgstab(Aop, b, dot, norm, zeros, tol=1e-8, maxiter=3000):
    x = zeros()
    r = b - Aop(x)
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
        v = Aop(p)
        den = dot(r0, v)
        if abs(den) < 1e-30:
            break
        alpha = rho_new / den
        s = r - alpha * v
        if norm(s) / nb < tol:
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
    res = norm(b - Aop(x)) / nb
    return x, {"n_iter": it, "residual": float(res),
               "converged": bool(res < tol)}


# =============================================================================
# §7. MASS-SCALING BENCHMARK  (the claim, measured)
# =============================================================================

def mass_scaling_benchmark(nd: int = 4, L: int = 4, Nc: int = 1,
                           masses: Tuple[float, ...] = (0.4, 0.2, 0.1, 0.05),
                           beta: float = 5.0, block_L: int = 2,
                           n_vecs: int = 8, n_relax: int = 6,
                           tol: float = 1e-8, seed: int = 3,
                           n_rhs: int = 1) -> List[Dict[str, Any]]:
    rows = []
    for m in masses:
        g = make_gauge(nd, L, Nc, beta, seed=seed)
        D = WilsonDirac(g, m)
        b = D.randn(999)

        # ---- plain BiCGStab ----
        D.reset()
        t0 = time.time()
        _, st_b = bicgstab(D.apply, b, D.dot, D.norm, D.zeros,
                           tol=tol, maxiter=4000)
        plain_applies = D.n_apply
        plain_wall = time.time() - t0

        # ---- AMG + FGMRES ----
        cfg = AMGConfig(block_L=block_L, n_vecs=n_vecs, n_relax=n_relax,
                        setup_seed=seed)
        amg = TwoLevelAMG(D, cfg)
        setup_wall = amg.setup_info["total_setup_wall_s"]
        setup_applies = (amg.setup_info["setup_dirac_applies"]
                         + amg.setup_info["coarse_build_applies"])
        D.reset(); amg.sap.reset_counters()
        t0 = time.time()
        solves = []
        for k in range(n_rhs):
            bk = D.randn(999 + k)
            _, st = fgmres(D.apply, amg.apply, bk, D.dot, D.norm, D.zeros,
                           tol=tol, restart=20, max_outer=60)
            solves.append(st)
        solve_wall = time.time() - t0
        amg_applies = D.n_apply
        conv = all(s["converged"] for s in solves)
        outer = max(s["n_iter"] for s in solves)

        rows.append({
            "mass": m,
            "plain_iters": st_b["n_iter"],
            "plain_applies": plain_applies,
            "plain_converged": st_b["converged"],
            "plain_wall_s": plain_wall,
            "amg_outer_iters": outer,
            "amg_applies": amg_applies,
            "amg_converged": conv,
            "amg_solve_wall_s": solve_wall,
            "amg_setup_wall_s": setup_wall,
            "amg_setup_applies": setup_applies,
            "amg_local_solves": amg.sap.n_local_solves,
            "coarse_dim": amg.Dc.shape[0],
            "n_rhs": n_rhs,
            "max_rayleigh": amg.setup_info["max_rayleigh"],
            "total_wall_amg": setup_wall + solve_wall,
            "speedup_solve_only": plain_wall / max(solve_wall, 1e-9),
            "speedup_incl_setup": plain_wall * n_rhs
                                  / max(setup_wall + solve_wall, 1e-9),
        })
    return rows


def print_benchmark(rows: List[Dict[str, Any]]) -> None:
    hdr = (f"{'mass':>8}{'plain it':>10}{'plain app':>11}"
           f"{'AMG outer':>11}{'AMG app':>10}{'coarse':>8}"
           f"{'setup s':>9}{'solve s':>9}{'spd solve':>11}{'spd+setup':>11}")
    print("=" * len(hdr))
    print("MASS SCALING: plain BiCGStab vs two-level DD-alphaAMG + FGMRES")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pc = "" if r["plain_converged"] else "*"
        ac = "" if r["amg_converged"] else "*"
        print(f"{r['mass']:>8.3f}{str(r['plain_iters'])+pc:>10}"
              f"{r['plain_applies']:>11}{str(r['amg_outer_iters'])+ac:>11}"
              f"{r['amg_applies']:>10}{r['coarse_dim']:>8}"
              f"{r['amg_setup_wall_s']:>9.3f}{r['amg_solve_wall_s']:>9.3f}"
              f"{r['speedup_solve_only']:>11.2f}"
              f"{r['speedup_incl_setup']:>11.2f}")
    print("=" * len(hdr))
    print("  * = did not reach tolerance")
    print("  plain it / AMG outer = OUTER Krylov iterations (the mass-scaling signal)")
    print("  spd solve  = plain wall / AMG solve wall (setup EXCLUDED)")
    print("  spd+setup  = includes setup, amortized over n_rhs solves")
    print()
    print("  READ THIS: if AMG outer iterations stay roughly flat as mass falls")
    print("  while plain iterations grow, the mass-scaling claim is SUPPORTED")
    print("  on this lattice. If both grow at the same rate, it is NOT.")
    print("  Setup cost is real: for n_rhs=1 AMG can easily be slower overall.")


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

    # ---- T1 block decomposition + coloring ----
    print("\n[T1] Block decomposition and red-black coloring")
    bd = BlockDecomposition(4, 4, 2)
    rep("all sites assigned", sum(len(s) for s in bd.sites) == 4 ** 4,
        str(bd.summary()))
    g = make_gauge(4, 4, 1, beta=5.0, seed=1)
    D = WilsonDirac(g, 0.2)
    # no two same-color blocks adjacent
    bad = 0
    for b in range(bd.n_blocks):
        for x in bd.sites[b]:
            for mu in range(4):
                for nb in (D.ip[mu][x], D.im[mu][x]):
                    b2 = bd.site_block[nb]
                    if b2 != b and bd.block_color[b2] == bd.block_color[b]:
                        bad += 1
    rep("no same-color block adjacency", bad == 0, f"violations={bad}")

    # ---- T2 local operator correctness (the decisive test) ----
    print("\n[T2] D_bb == restriction of full D to block-supported vectors")
    maxerr = 0.0
    for b in range(bd.n_blocks):
        M = build_local_operator(D, bd, b)
        sites = bd.sites[b]
        idx = np.concatenate([np.arange(int(s) * D.w, (int(s) + 1) * D.w)
                              for s in sites])
        rng = np.random.default_rng(b)
        v = np.zeros(D.size, dtype=complex)
        v[idx] = rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))
        full = D.apply(v.reshape(D.shape)).reshape(-1)[idx]
        loc = M @ v[idx]
        maxerr = max(maxerr, float(np.abs(full - loc).max()))
    rep("local operator matches full restriction", maxerr < 1e-11,
        f"max_err={maxerr:.2e}")

    # ---- T3 SAP reduces residual ----
    print("\n[T3] SAP smoother reduces the residual")
    sap = SAPSmoother(D, bd)
    b = D.randn(7)
    x = D.zeros()
    r0 = D.norm(b - D.apply(x))
    x = sap.sweep(x, b, n_sweeps=4)
    r1 = D.norm(b - D.apply(x))
    rep("residual decreased", r1 < r0, f"{r0:.3e} -> {r1:.3e}")

    # ---- T4 adaptive setup finds near-null modes ----
    print("\n[T4] Adaptive setup: relaxation on homogeneous problem")
    V, sinfo = adaptive_setup(D, sap, n_vecs=6, n_relax=8, seed=0)
    rand_ray = []
    for k in range(3):
        v = D.randn(50 + k); v /= D.norm(v)
        rand_ray.append(D.norm(D.apply(v)) / D.norm(v))
    rep("test vectors have smaller ||Dv||/||v|| than random",
        sinfo["max_rayleigh"] < float(np.mean(rand_ray)),
        f"setup max={sinfo['max_rayleigh']:.4f} vs random mean={np.mean(rand_ray):.4f}")
    rep("test vectors orthonormal",
        float(np.abs(V.conj().T @ V - np.eye(V.shape[1])).max()) < 1e-9)

    # ---- T5 chiral prolongator preserves gamma5 structure ----
    print("\n[T5] gamma5 P == P gamma5_c  (the structure that must not break)")
    P, g5c, pinfo = chiral_prolongator(D, bd, V)
    g5_big = np.kron(np.eye(D.V, dtype=complex),
                     np.kron(D.g5, np.eye(D.Nc, dtype=complex)))
    lhs = g5_big @ P
    rhs = P * g5c[None, :]
    err = float(np.abs(lhs - rhs).max())
    rep("gamma5 P == P gamma5_c", err < 1e-10, f"max_err={err:.2e}")
    rep("gamma5_c signs are +/-1", set(np.unique(g5c)) <= {-1.0, 1.0})
    rep("prolongator full column rank",
        np.linalg.matrix_rank(P) == P.shape[1],
        f"coarse_dim={P.shape[1]}")

    # ---- T6 Galerkin coarse operator + its gamma5_c-hermiticity ----
    print("\n[T6] D_c = P^dag D P and gamma5_c-hermiticity")
    Dc = galerkin_coarse(D, P)
    Gc = np.diag(g5c)
    herm_err = float(np.abs(Dc.conj().T - Gc @ Dc @ Gc).max())
    rep("D_c^dag == gamma5_c D_c gamma5_c", herm_err < 1e-9,
        f"max_err={herm_err:.2e}")
    # verify Galerkin entrywise against direct product
    j = 3
    v = P[:, j].reshape(D.shape)
    col = P.conj().T @ D.apply(v).reshape(-1)
    rep("Galerkin column matches direct P^dag D P",
        float(np.abs(col - Dc[:, j]).max()) < 1e-10)

    # ---- T7 AMG solve converges ----
    print("\n[T7] Two-level AMG + FGMRES converges")
    cfg = AMGConfig(block_L=2, n_vecs=6, n_relax=8)
    amg = TwoLevelAMG(D, cfg)
    D.reset()
    x, st = fgmres(D.apply, amg.apply, b, D.dot, D.norm, D.zeros,
                   tol=1e-9, restart=20, max_outer=40)
    rep("AMG+FGMRES converged", st["converged"],
        f"res={st['residual']:.2e} outer={st['n_iter']} "
        f"coarse_dim={amg.Dc.shape[0]}")
    rep("AMG uses fewer outer iterations than plain BiCGStab",
        st["n_iter"] < bicgstab(D.apply, b, D.dot, D.norm, D.zeros,
                                tol=1e-9, maxiter=4000)[1]["n_iter"],
        f"AMG={st['n_iter']}")

    # ---- T8 mass scaling (informational + soft check) ----
    print("\n[T8] MASS SCALING MEASUREMENT")
    rows = mass_scaling_benchmark(masses=(0.4, 0.2, 0.1, 0.05),
                                  n_vecs=6, n_relax=6, n_rhs=2)
    print()
    print_benchmark(rows)
    conv_rows = [r for r in rows if r["plain_converged"] and r["amg_converged"]]
    if len(conv_rows) >= 2:
        pg = conv_rows[-1]["plain_iters"] / max(conv_rows[0]["plain_iters"], 1)
        ag = conv_rows[-1]["amg_outer_iters"] / max(conv_rows[0]["amg_outer_iters"], 1)
        print(f"\n  growth factor from mass {conv_rows[0]['mass']} "
              f"to {conv_rows[-1]['mass']}:")
        print(f"    plain BiCGStab iterations : x{pg:.2f}")
        print(f"    AMG outer iterations      : x{ag:.2f}")
        rep("AMG iteration growth <= plain growth", ag <= pg + 1e-9,
            f"AMG x{ag:.2f} vs plain x{pg:.2f}")
    rep("setup cost is reported separately",
        all("amg_setup_wall_s" in r for r in rows))

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  block coloring valid; D_bb EXACTLY matches restriction of full D;
  SAP reduces residual; adaptive setup finds lower-Rayleigh modes than
  random; test vectors orthonormal; gamma5 P == P gamma5_c to machine
  precision; D_c inherits gamma5_c-hermiticity; Galerkin product correct;
  AMG+FGMRES converges with fewer outer iterations than plain BiCGStab.

MEASURED (not asserted)
  outer-iteration growth versus mass for plain Krylov and for AMG;
  setup cost, solve cost, and speedup with and without setup amortization.

NOT IMPLEMENTED / NOT CLAIMED
  ❌ Three or more levels. Needs block structure + SAP + adaptive setup
     ON THE COARSE OPERATOR. Coarse level here is a dense direct solve.
  ❌ Odd-even preconditioning inside local solves, mixed precision,
     block-size tuning, communication overlap, optimized local LU.
  ❌ Prolongator is DENSE: memory O(size * coarse_dim) does not scale.
  ❌ Small lattices, random (not thermalized) gauge background.
  ❌ No claim of production DD-alphaAMG performance.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Two-level DD-alphaAMG for Wilson Dirac")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--nd", type=int, default=4, choices=[2, 4])
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--Nc", type=int, default=1)
    ap.add_argument("--block-L", type=int, default=2)
    ap.add_argument("--masses", type=float, nargs="*",
                    default=[0.4, 0.2, 0.1, 0.05])
    ap.add_argument("--n-vecs", type=int, default=8)
    ap.add_argument("--n-relax", type=int, default=6)
    ap.add_argument("--n-rhs", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1
    if not a.benchmark:
        ap.print_help()
        print(BANNER)
        print("Try: --selftest   or   --benchmark --masses 0.4 0.2 0.1 0.05")
        return 0

    print(BANNER)
    print(f"scipy LU available: {HAS_SCIPY}")
    rows = mass_scaling_benchmark(
        nd=a.nd, L=a.L, Nc=a.Nc, masses=tuple(a.masses), beta=a.beta,
        block_L=a.block_L, n_vecs=a.n_vecs, n_relax=a.n_relax,
        tol=a.tol, n_rhs=a.n_rhs)
    print_benchmark(rows)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, default=float))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
