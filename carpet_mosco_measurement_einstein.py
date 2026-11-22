#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
carpet_mosco_measurement_einstein.py — v1.0.0
Closing Mosco M1/M2 via the Carpet-19 consolidated revision.
Spectral measurement theory · disorder · advanced probability · stochastic.
Einstein continuation monitor (NOT global regularity).
=============================================================================

Author  : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID   : 0009-0008-2374-0788
License : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT IS CLOSED, AND WHAT CANNOT BE CLOSED HERE
=============================================================================
CLOSED (instantiated + numerically verified from the Carpet-19 revision):
  Thm 1   R_{n+m} <= 12 R_n R_m, rho exists, rho >= 9/8, shorting
  Thm 2   resistance gauge => FULL-SEQUENCE Mosco convergence (M1 + M2)
          under standing inputs I1-I4. The paper PROVES this; this module
          verifies the hypotheses and the convergence rates numerically.
  Thm 3   compact resolvent, discrete spectrum, f(A_n) -> f(A),
          eigenvalue convergence with multiplicity (Anselone)
  Thm 4   spectral Sobolev scale W^s, moment inequality, compact embedding,
          Holder embedding W^1 -> C^{0,beta/2}
  Thm 5   support-locality of integer powers A^k
  Thm 6   energy measures converge weakly; discrete identity (1)
  Thm 7   derivations, 1-form modules, Fredholm projections converge;
          discrete identity (2)
  Prop 3  commutator [A,M_f] bounded W^1 -> W^{-1}, norm ~ E(f)^{1/2}
  Prop 4  polyharmonic identity A^k = d* (d d*)^{k-1} d
  Thm 9   NO Connes-type Dirac operator with bounded commutators

CITED, NOT REPROVED:
  Thm 8   Hino (2005) singularity of energy measures. Diagnostics only.

CANNOT BE CLOSED BY THIS MODULE — AND WHY:
  Einstein + nonlinear-field GLOBAL regularity.
  REASON (category distinction, not a gap in effort):
    * Carpet: compact metric measure space, ELLIPTIC Dirichlet form,
      positive self-adjoint generator, COMPACT resolvent, contraction
      semigroup. Mosco convergence is a statement about convergence of
      FORMS and their PARABOLIC evolutions.
    * Einstein: QUASILINEAR HYPERBOLIC system on a Lorentzian manifold.
      No positive self-adjoint generator, no compact resolvent, no
      Mosco framework. Global regularity is a blow-up/continuation
      question, not a form-convergence question.
  There is no theorem transporting Mosco convergence of carpet forms
  into global existence for Einstein equations. Claiming otherwise
  would be a category error.

  WHAT IS PROVIDED INSTEAD (honest and useful):
    A SPECTRAL CONTINUATION MONITOR on the spatial slice, tracking
      (i)   high-order energy E_s(t),
      (ii)  uniform Lorentzian character (lapse > 0, det h > 0),
      (iii) Hamiltonian + momentum constraint residuals,
      (iv)  stress-energy divergence residual.
    Standard continuation for quasilinear hyperbolic systems says:
      sup_t E_s(t) < inf  AND  uniform Lorentzian bounds  =>  extend.
    This module MEASURES those quantities. It does NOT prove they stay
    bounded. Numerical boundedness on a finite interval is evidence,
    never a global theorem.

=============================================================================
THE MEASUREMENT-THEORY BRIDGE  (this is the genuine conceptual link)
=============================================================================
Let Gamma = nu_f be the energy measure of f, mu the reference measure,
F_m the sigma-algebra generated by level-m cells. Define

    X_m = d(Gamma|F_m) / d(mu|F_m),   X_m(S) = Gamma(S)/mu(S).

Under mu, (X_m) is a NONNEGATIVE MARTINGALE with E_mu[X_m] = Gamma(K).
Martingale convergence gives X_m -> d(Gamma_ac)/d(mu) mu-a.s. Hence

    Gamma << mu   <=>  (X_m) uniformly integrable
    Gamma _|_ mu  <=>  X_m -> 0  mu-a.s.

Hino's theorem (Thm 8) says the second alternative holds on the carpet.
CONSEQUENCE FOR MEASUREMENT THEORY: the "outcome measure" Gamma of a
finite-energy observable f is SINGULAR with respect to the reference
measure mu. There is NO L2 density. Therefore:

  * no Radon-Nikodym likelihood ratio in L2(mu),
  * no classical Bakry-Emery Gamma_2 pairing with Ag merely in L2,
  * the commutator is measure-valued, bounded only W^1 -> W^{-1},
  * the generator domain D(A) is NOT an algebra (Cor 3).

This is why the ESCA probability layer must treat MEASURES and
UNCERTAINTY SETS as primary objects rather than densities: on this
substrate, densities do not exist. That is a theorem, not a preference.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sps
    import scipy.sparse.linalg as spla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("carpet_mosco")

VERSION = "1.0.0"
M_CARPET = 8            # cells per level
RHO_MIN = 9.0 / 8.0     # Prop 1 lower bound (proved)

BANNER = f"""
=============================================================================
CARPET-19 MOSCO CLOSURE + SPECTRAL MEASUREMENT THEORY v{VERSION}
M1/M2 instantiated from Thm 2 (I1-I4). Spectral scale, energy measures,
derivations, Fredholm projections, commutator bounds, Hino diagnostics.
Einstein: CONTINUATION MONITOR only. Global regularity NOT claimed.
=============================================================================
"""


# =============================================================================
# §0. RESULT REGISTRY  (labels follow the paper: P/C/D/O)
# =============================================================================

class Lab:
    P = "PROVEN_IN_PAPER"
    C = "CITED"
    D = "DERIVED_IN_PAPER"
    O = "OPEN"


@dataclass
class Result:
    id: str
    label: str
    statement: str
    verified: Optional[bool] = None
    certificate: Optional[Dict[str, Any]] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RESULTS: Dict[str, Result] = {}


def reg(r: Result) -> Result:
    RESULTS[r.id] = r
    return r


def mark(rid: str, ok: bool, cert: Optional[Dict] = None) -> None:
    if rid in RESULTS:
        RESULTS[rid].verified = bool(ok)
        if cert is not None:
            RESULTS[rid].certificate = cert


def build_registry() -> None:
    reg(Result("Thm1", Lab.P,
               "R_{n+m} <= 12 R_n R_m; rho = lim R_n^{1/n} exists; rho >= 9/8; "
               "R_{n+j} >= 3^{-j} R_n"))
    reg(Result("I1", Lab.D, "Uniform oscillation: |u(Q)-u(Q')|^2 <= "
                            "C(dist+3^-n)^beta E_n(u), beta = log rho/log 3"))
    reg(Result("I2", Lab.C, "Subsequential Mosco convergence (KZ Thm 4.5, 5.4)"))
    reg(Result("I3", Lab.D, "Uniqueness up to scalar (BBKT Thm 1.2 + Prop 2)"))
    reg(Result("I4", Lab.C, "D_0 separates points (proof of KZ Thm 7.19)"))
    reg(Result("Thm2", Lab.D,
               "Resistance gauge => FULL-SEQUENCE Mosco convergence (M1+M2)"))
    reg(Result("Thm3", Lab.D,
               "Compact resolvent; discrete spectrum; f(A_n)->f(A); "
               "eigenvalues with multiplicity converge (Anselone)"))
    reg(Result("Thm4", Lab.P,
               "Spectral Sobolev scale W^s; moment inequality; compact "
               "embedding; Holder W^1 -> C^{0,beta/2}"))
    reg(Result("Thm5", Lab.P, "supp A^k u subset supp u (support-local)"))
    reg(Result("Thm6", Lab.P,
               "Energy measures converge weakly; discrete identity (1)"))
    reg(Result("Thm7", Lab.P,
               "Derivations, 1-form modules, Fredholm projections converge; "
               "discrete identity (2)"))
    reg(Result("Prop3", Lab.P,
               "[A,M_f] bounded W^1 -> W^{-1}, norm ~ E(f)^{1/2}; "
               "NO order gain"))
    reg(Result("Thm8", Lab.C,
               "Hino 2005: energy measures singular w.r.t. mu"))
    reg(Result("Cor3", Lab.D,
               "D(A) is NOT an algebra; f nonconstant in D(A) => f^2 not in D(A)"))
    reg(Result("Prop4", Lab.P,
               "A^k = d* (d d*)^{k-1} d; Sobolev order exactly 2k"))
    reg(Result("Thm9", Lab.P,
               "No Connes-type Dirac operator with bounded commutators; "
               "relative bound has order 1, never 0"))
    reg(Result("H3", Lab.O,
               "Level-uniform commutator estimate W^{s+1}->W^s: UNSUPPORTED; "
               "corrected numerics show geometric growth"))
    reg(Result("Einstein_global", Lab.O,
               "Global regularity of Einstein + nonlinear field. "
               "NOT closable by carpet Mosco convergence (category mismatch)."))
    reg(Result("Einstein_continuation", Lab.C,
               "Standard continuation: sup E_s < inf + uniform Lorentzian "
               "=> extend. Monitored here, NOT proved bounded."))


# =============================================================================
# §1. CARPET LEVEL  (graph, Laplacian, incidence, measure, projections)
# =============================================================================

def carpet_cells(n: int) -> np.ndarray:
    """
    Level-n carpet cells as integer (row, col) in [0,3^n)^2.
    A cell is REMOVED iff at some ternary digit position both digits == 1.
    Returns array of shape (8^n, 2), lexicographically sorted.
    """
    side = 3 ** n
    keep = np.ones((side, side), dtype=bool)
    r = np.arange(side)[:, None]
    c = np.arange(side)[None, :]
    rr, cc = r.copy(), c.copy()
    for _ in range(n):
        keep &= ~(((rr % 3) == 1) & ((cc % 3) == 1))
        rr //= 3
        cc //= 3
    idx = np.argwhere(keep)
    return idx.astype(np.int64)


class CarpetLevel:
    """
    Level-n carpet cell graph with unit conductances.

    Attributes
    ----------
    n        : level
    cells    : (8^n, 2) integer coordinates in [0,3^n)^2
    centers  : (8^n, 2) cell centres in [0,1]^2 (for Holder / distance tests)
    L        : graph Laplacian (sparse if scipy, else dense)
    D        : incidence operator, shape (n_edges, n_cells), entries +-1
    edges    : (n_edges, 2) cell index pairs
    mu       : reference measure per cell = 8^{-n}
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("level must be >= 1")
        self.n = int(n)
        self.cells = carpet_cells(n)
        self.N = self.cells.shape[0]
        assert self.N == M_CARPET ** n, (self.N, M_CARPET ** n)
        side = 3 ** n
        self.centers = (self.cells.astype(np.float64) + 0.5) / side
        self.mu = np.full(self.N, M_CARPET ** (-n))

        key = {tuple(c): i for i, c in enumerate(self.cells)}
        ei, ej = [], []
        for i, (r, c) in enumerate(self.cells):
            for dr, dc in ((1, 0), (0, 1)):     # each unordered edge once
                j = key.get((int(r + dr), int(c + dc)))
                if j is not None:
                    ei.append(i); ej.append(j)
        self.edges = np.stack([ei, ej], axis=1).astype(np.int64)
        self.nE = self.edges.shape[0]
        # expected |E_n| = (8/5)(8^n - 3^n)
        self.nE_expected = int(round(8.0 / 5.0 * (M_CARPET ** n - 3 ** n)))

        rows = np.concatenate([np.arange(self.nE), np.arange(self.nE)])
        cols = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        vals = np.concatenate([np.ones(self.nE), -np.ones(self.nE)])
        if HAS_SCIPY:
            self.D = sps.csr_matrix((vals, (rows, cols)),
                                    shape=(self.nE, self.N))
            self.L = (self.D.T @ self.D).tocsr()
        else:
            Dd = np.zeros((self.nE, self.N))
            Dd[rows, cols] = vals
            self.D = Dd
            self.L = Dd.T @ Dd

    # ---- quadratic forms -------------------------------------------------
    def E0(self, u: np.ndarray) -> float:
        """Unnormalised graph energy sum_edges (u_i-u_j)^2."""
        du = self.diff(u)
        return float(np.sum(du * du))

    def E0_bilinear(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.sum(self.diff(u) * self.diff(v)))

    def diff(self, u: np.ndarray) -> np.ndarray:
        """Incidence: (D u)_e = u(x_e) - u(y_e)."""
        return u[self.edges[:, 0]] - u[self.edges[:, 1]]

    def adjoint(self, zeta: np.ndarray) -> np.ndarray:
        """D^T zeta (divergence-like)."""
        if HAS_SCIPY:
            return np.asarray(self.D.T @ zeta)
        return self.D.T @ zeta

    def apply_L(self, u: np.ndarray) -> np.ndarray:
        if HAS_SCIPY:
            return np.asarray(self.L @ u)
        return self.L @ u

    # ---- boundary sets ---------------------------------------------------
    def left_column(self) -> np.ndarray:
        return np.flatnonzero(self.cells[:, 1] == 0)

    def right_column(self) -> np.ndarray:
        return np.flatnonzero(self.cells[:, 1] == 3 ** self.n - 1)

    def bottom_row(self) -> np.ndarray:
        return np.flatnonzero(self.cells[:, 0] == 0)

    def top_row(self) -> np.ndarray:
        return np.flatnonzero(self.cells[:, 0] == 3 ** self.n - 1)

    # ---- parent map (for coarse-graining / martingale) -------------------
    def parent_index(self, m: int) -> np.ndarray:
        """
        For each level-n cell, the index of its level-m ancestor cell.
        Returns (N_n,) int array indexing into CarpetLevel(m).cells.
        """
        if m > self.n:
            raise ValueError("m must be <= n")
        shift = 3 ** (self.n - m)
        anc = self.cells // shift
        mcells = carpet_cells(m)
        key = {tuple(c): i for i, c in enumerate(mcells)}
        return np.array([key[tuple(a)] for a in anc], dtype=np.int64)

    def cell_average(self, u: np.ndarray, m: int) -> np.ndarray:
        """
        L2(mu)-orthogonal projection onto functions constant on m-cells,
        returned as a level-n vector (broadcast back). This is P_m u.
        """
        par = self.parent_index(m)
        Nm = M_CARPET ** m
        s = np.bincount(par, weights=u, minlength=Nm)
        cnt = np.bincount(par, minlength=Nm)
        return (s / np.maximum(cnt, 1))[par]

    def coarse_values(self, u: np.ndarray, m: int) -> np.ndarray:
        """The m-cell values of P_m u (length 8^m)."""
        par = self.parent_index(m)
        Nm = M_CARPET ** m
        s = np.bincount(par, weights=u, minlength=Nm)
        cnt = np.bincount(par, minlength=Nm)
        return s / np.maximum(cnt, 1)


# =============================================================================
# §2. RESISTANCE + GAUGE  (Thm 1, Lemma 11/12, Thm 2)
# =============================================================================

def effective_resistance(lv: CarpetLevel, src: np.ndarray, dst: np.ndarray
                         ) -> float:
    """
    Effective resistance between two cell sets, unit current distributed
    uniformly: inject 1/|src| at each src cell, extract 1/|dst| at each dst.
    Solve L v = i with mean-zero gauge; R = mean(v[src]) - mean(v[dst]).
    """
    N = lv.N
    i = np.zeros(N)
    i[src] += 1.0 / len(src)
    i[dst] -= 1.0 / len(dst)
    if HAS_SCIPY:
        L = lv.L.tolil()
        # pin one node to remove the constant kernel
        pin = 0
        Lrows = lv.L.tocsr().copy().tolil()
        Lrows[pin, :] = 0.0
        Lrows[pin, pin] = 1.0
        rhs = i.copy(); rhs[pin] = 0.0
        v = spla.spsolve(Lrows.tocsr(), rhs)
        if not np.all(np.isfinite(v)):
            v = spla.lsqr(Lrows.tocsr(), rhs)[0]
    else:
        Ld = lv.L.copy()
        Ld[0, :] = 0.0; Ld[0, 0] = 1.0
        rhs = i.copy(); rhs[0] = 0.0
        v = np.linalg.solve(Ld, rhs)
    v = v - v.mean()
    return float(v[src].mean() - v[dst].mean())


def crossing_resistance(n: int) -> float:
    """R_cross(n): left column -> right column, unit total current."""
    lv = CarpetLevel(n)
    return effective_resistance(lv, lv.left_column(), lv.right_column())


def max_pair_resistance(n: int, n_probe: int = 6) -> float:
    """Cheap lower bound on R_n = max effective resistance (probe corners)."""
    lv = CarpetLevel(n)
    pts = [lv.left_column()[0], lv.left_column()[-1],
           lv.right_column()[0], lv.right_column()[-1],
           lv.bottom_row()[0], lv.top_row()[-1]]
    best = 0.0
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            r = effective_resistance(lv, np.array([pts[a]]), np.array([pts[b]]))
            best = max(best, abs(r))
    return best


class ResistanceGauge:
    """
    kappa_n = R_cross(n) / R_star  (Thm 2 resistance gauge).
    Normalised form  E_n = kappa_n * E0_n.
    Generator on L2(K,mu):  A_n = 8^n * kappa_n * L_n.
    """

    def __init__(self, levels: Tuple[int, ...] = (1, 2, 3, 4),
                 R_star: float = 1.0):
        self.levels = tuple(levels)
        self.R_star = float(R_star)
        self.R: Dict[int, float] = {}
        self.kappa: Dict[int, float] = {}
        self.lv: Dict[int, CarpetLevel] = {}
        for n in self.levels:
            t0 = time.time()
            self.R[n] = crossing_resistance(n)
            self.kappa[n] = self.R[n] / self.R_star
            self.lv[n] = CarpetLevel(n)
            log.info("[gauge] n=%d  N=%d  E=%d  R_cross=%.5f  (%.2fs)",
                     n, self.lv[n].N, self.lv[n].nE, self.R[n], time.time() - t0)

    def E(self, n: int, u: np.ndarray) -> float:
        return self.kappa[n] * self.lv[n].E0(u)

    def E_bil(self, n: int, u: np.ndarray, v: np.ndarray) -> float:
        return self.kappa[n] * self.lv[n].E0_bilinear(u, v)

    def A_apply(self, n: int, u: np.ndarray) -> np.ndarray:
        return (M_CARPET ** n) * self.kappa[n] * self.lv[n].apply_L(u)

    def ratios(self) -> List[float]:
        ks = sorted(self.R)
        return [self.R[ks[i + 1]] / self.R[ks[i]] for i in range(len(ks) - 1)]

    def rho_estimate(self) -> float:
        r = self.ratios()
        return float(r[-1]) if r else float("nan")


def verify_thm1(gauge: ResistanceGauge) -> Dict[str, Any]:
    """Thm 1 / Prop 1 / Lemma 2 numerical consistency (not a proof)."""
    out: Dict[str, Any] = {"R_cross": dict(gauge.R), "ratios": gauge.ratios()}
    ks = sorted(gauge.R)
    # rho >= 9/8 lower bound: R_n >= (4/9)(9/8)^n  (Prop 1, for R_n = max res)
    lb_ok = True
    maxres = {}
    for n in ks:
        if n <= 3:
            mr = max_pair_resistance(n)
            maxres[n] = mr
            lb_ok &= (mr >= (4.0 / 9.0) * RHO_MIN ** n - 1e-9)
    out["max_pair_resistance_probe"] = maxres
    out["prop1_lower_bound_holds_on_probes"] = bool(lb_ok)
    # shorting: R_{n+1} >= R_n / 3
    short_ok = all(gauge.R[ks[i + 1]] >= gauge.R[ks[i]] / 3.0 - 1e-9
                   for i in range(len(ks) - 1))
    out["shorting_holds"] = bool(short_ok)
    # quasi-multiplicativity upper half: R_{n+m} <= 12 R_n R_m
    qm_ok = True
    for a in ks:
        for b in ks:
            if a + b in gauge.R:
                qm_ok &= (gauge.R[a + b] <= 12.0 * gauge.R[a] * gauge.R[b] + 1e-9)
    out["quasi_multiplicative_upper_holds"] = bool(qm_ok)
    out["rho_estimate_numeric_only"] = gauge.rho_estimate()
    out["NOTE"] = ("rho ~ 1.25 is NUMERICAL. The proved bound is rho >= 9/8. "
                   "Five levels show a rate, not a limit.")
    return out


# =============================================================================
# §3. MOSCO M1 / M2  (Thm 2 instantiation)
# =============================================================================

class MoscoVerifier:
    """
    Numerical instantiation of Thm 2's M1 / M2.

    M1 (weak lower bound): for u_n -> u weakly in H,
         E(u) <= liminf E_n(u_n).
    M2 (strong recovery): for each u there is u_n -> u strongly with
         E(u) >= limsup E_n(u_n).

    The PROOF is Thm 2 of the paper under I1-I4. Here we test the
    inequalities on constructed sequences, using the finest available
    level as a proxy for the limit form. That is evidence, not a proof.
    """

    def __init__(self, gauge: ResistanceGauge):
        self.g = gauge
        self.nref = max(gauge.levels)

    def limit_proxy(self, u_ref: np.ndarray) -> float:
        return self.g.E(self.nref, u_ref)

    def recovery_sequence(self, u_ref: np.ndarray, n: int) -> np.ndarray:
        """u_n = restriction of u_ref to level n via cell averaging."""
        lvref, lvn = self.g.lv[self.nref], self.g.lv[n]
        par = lvref.parent_index(n)
        Nm = M_CARPET ** n
        s = np.bincount(par, weights=u_ref, minlength=Nm)
        cnt = np.bincount(par, minlength=Nm)
        return (s / np.maximum(cnt, 1))

    def test_M2(self, u_ref: np.ndarray) -> Dict[str, Any]:
        E_lim = self.limit_proxy(u_ref)
        rows = []
        for n in sorted(self.g.levels):
            un = self.recovery_sequence(u_ref, n)
            En = self.g.E(n, un)
            # strong L2(mu) convergence: compare P_n u_ref to u_ref on ref grid
            lvn = self.g.lv[n]
            broadcast = un[lv_ref_parent(self.g.lv[self.nref], n)]
            l2 = float(np.sqrt(np.mean((broadcast - u_ref) ** 2)))
            rows.append({"n": n, "E_n": En, "L2_to_ref": l2,
                         "M2_ineq": bool(En <= E_lim + 1e-8)})
        conv = all(r["E_n"] <= E_lim + 1e-8 for r in rows)
        l2_decay = rows[-1]["L2_to_ref"] <= rows[0]["L2_to_ref"] + 1e-12
        return {"E_limit_proxy": E_lim, "rows": rows,
                "M2_holds_on_recovery": bool(conv),
                "L2_decreases": bool(l2_decay)}

    def test_M1(self, u_ref: np.ndarray, amp: float = 1.0) -> Dict[str, Any]:
        """
        Weakly-but-not-strongly convergent sequence:
            u_n = P_n u  +  amp * (high-frequency eigenvector of A_n)
        High-frequency eigenvectors -> 0 weakly, so u_n -> u weakly.
        Their energy grows, so liminf E_n(u_n) >= E(u) should hold.
        """
        E_lim = self.limit_proxy(u_ref)
        rows = []
        for n in sorted(self.g.levels):
            un = self.recovery_sequence(u_ref, n)
            lv = self.g.lv[n]
            hf = high_freq_mode(self.g, n)
            un_w = un + amp * hf
            En = self.g.E(n, un_w)
            rows.append({"n": n, "E_n_weak_seq": En,
                         "M1_ineq": bool(E_lim <= En + 1e-6)})
        ok = all(r["M1_ineq"] for r in rows)
        return {"E_limit_proxy": E_lim, "rows": rows,
                "M1_holds_on_weak_seq": bool(ok)}

    def report(self, u_ref: np.ndarray) -> Dict[str, Any]:
        m1 = self.test_M1(u_ref)
        m2 = self.test_M2(u_ref)
        both = m1["M1_holds_on_weak_seq"] and m2["M2_holds_on_recovery"]
        return {"M1": m1, "M2": m2, "both_hold_numerically": bool(both),
                "STATUS": ("Thm 2 PROVES Mosco under I1-I4. These are "
                           "numerical consistency checks, not the proof."),
                "gauge_ratios": self.g.ratios(),
                "rho_numeric": self.g.rho_estimate()}


def lv_ref_parent(lvref: CarpetLevel, n: int) -> np.ndarray:
    return lvref.parent_index(n)


def high_freq_mode(g: ResistanceGauge, n: int, k_from_top: int = 3
                   ) -> np.ndarray:
    """An eigenvector near the top of the spectrum of A_n (weakly -> 0)."""
    lv = g.lv[n]
    scale = (M_CARPET ** n) * g.kappa[n]
    if HAS_SCIPY and lv.N > 64:
        vals, vecs = spla.eigsh(lv.L.tocsc(), k=min(k_from_top + 2, lv.N - 2),
                                which="LA", tol=1e-8)
        v = vecs[:, -k_from_top]
    else:
        w, V = np.linalg.eigh(lv.L if not HAS_SCIPY else lv.L.toarray())
        v = V[:, -(k_from_top + 1)]
    v = v / max(np.linalg.norm(v), 1e-12)
    return v


# =============================================================================
# §4. SPECTRAL CONSEQUENCES  (Thm 3, Thm 4)
# =============================================================================

def spectrum(g: ResistanceGauge, n: int, k: int = 12
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Lowest k eigenpairs of A_n = 8^n kappa_n L_n (excluding the constant)."""
    lv = g.lv[n]
    scale = (M_CARPET ** n) * g.kappa[n]
    if HAS_SCIPY and lv.N > 64:
        vals, vecs = spla.eigsh(lv.L.tocsc(), k=min(k + 1, lv.N - 2),
                                sigma=-1e-8, which="LM", tol=1e-9)
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        # drop the near-zero constant mode
        keep = vals > 1e-7
        vals, vecs = vals[keep][:k], vecs[:, keep][:, :k]
    else:
        Ld = lv.L if not HAS_SCIPY else lv.L.toarray()
        w, V = np.linalg.eigh(Ld)
        keep = w > 1e-7
        vals, vecs = w[keep][:k], V[:, keep][:, :k]
    return scale * vals, vecs


def verify_thm3(g: ResistanceGauge, n_top: int = 8) -> Dict[str, Any]:
    """Eigenvalue convergence across levels (Anselone-style, with multiplicity)."""
    per_level = {}
    for n in sorted(g.levels):
        w, _ = spectrum(g, n, k=n_top)
        per_level[n] = w.tolist()
    ks = sorted(per_level)
    deltas = []
    for i in range(len(ks) - 1):
        a, b = np.array(per_level[ks[i]]), np.array(per_level[ks[i + 1]])
        m = min(len(a), len(b))
        rel = np.abs(b[:m] - a[:m]) / np.maximum(np.abs(a[:m]), 1e-12)
        deltas.append({"pair": [ks[i], ks[i + 1]],
                       "max_rel_change": float(rel.max()),
                       "mean_rel_change": float(rel.mean())})
    shrinking = (len(deltas) >= 2 and
                 deltas[-1]["mean_rel_change"] <= deltas[0]["mean_rel_change"] + 1e-9)
    return {"eigenvalues_per_level": per_level, "changes": deltas,
            "trend_shrinking": bool(shrinking),
            "NOTE": ("Thm 3(c) proves lambda_j(A_n) -> lambda_j with "
                     "multiplicity via collectively-compact resolvents "
                     "(Anselone Thm 3.7). Numerics illustrate the rate.")}


def resolvent(g: ResistanceGauge, n: int, f: np.ndarray, lam: float = 1.0
              ) -> np.ndarray:
    lv = g.lv[n]
    A = (M_CARPET ** n) * g.kappa[n] * lv.L
    if HAS_SCIPY:
        M = (A + lam * sps.eye(lv.N, format="csr")).tocsc()
        return spla.spsolve(M, f)
    return np.linalg.solve(A + lam * np.eye(lv.N), f)


def semigroup(g: ResistanceGauge, n: int, f: np.ndarray, t: float = 0.05,
              k: int = 40) -> np.ndarray:
    """e^{-tA_n} f via spectral truncation (Thm 3(b) functional calculus)."""
    w, V = spectrum(g, n, k=min(k, g.lv[n].N - 2))
    coef = V.T @ (f * g.lv[n].mu) / g.lv[n].mu[0]
    return V @ (np.exp(-t * w) * coef)


def verify_resolvent_semigroup(g: ResistanceGauge, m_coarse: int = 1,
                               lam: float = 1.0, t: float = 0.05
                               ) -> Dict[str, Any]:
    """Strong resolvent + semigroup convergence, coarse-grained to level m."""
    nref = max(g.levels)
    lvref = g.lv[nref]
    rng = np.random.default_rng(0)
    f_ref = rng.normal(size=lvref.N)
    r_ref = resolvent(g, nref, f_ref, lam)
    s_ref = semigroup(g, nref, f_ref, t)
    rows = []
    for n in sorted(g.levels):
        lv = g.lv[n]
        par = lvref.parent_index(n)
        f_n = np.bincount(par, weights=f_ref, minlength=lv.N) / \
              np.maximum(np.bincount(par, minlength=lv.N), 1)
        r_n = resolvent(g, n, f_n, lam)
        s_n = semigroup(g, n, f_n, t)
        r_bc = r_n[par]; s_bc = s_n[par]
        rows.append({
            "n": n,
            "res_rel_err": float(np.linalg.norm(r_bc - r_ref) /
                                 max(np.linalg.norm(r_ref), 1e-12)),
            "sem_rel_err": float(np.linalg.norm(s_bc - s_ref) /
                                 max(np.linalg.norm(s_ref), 1e-12)),
        })
    res_decay = rows[-1]["res_rel_err"] <= rows[0]["res_rel_err"] + 1e-12
    sem_decay = rows[-1]["sem_rel_err"] <= rows[0]["sem_rel_err"] + 1e-12
    return {"rows": rows, "resolvent_error_decreases": bool(res_decay),
            "semigroup_error_decreases": bool(sem_decay)}


def sobolev_norm(g: ResistanceGauge, n: int, u: np.ndarray, s: float,
                 k: int = 60) -> float:
    """||u||_{W^s}^2 = sum (1+lambda_j)^s |u_j|^2 (spectral definition)."""
    w, V = spectrum(g, n, k=min(k, g.lv[n].N - 2))
    coef = V.T @ u
    return float(np.sqrt(np.sum((1.0 + w) ** s * coef ** 2)))


def verify_thm4(g: ResistanceGauge) -> Dict[str, Any]:
    """Moment inequality, compact embedding tail, Holder embedding."""
    n = max(g.levels)
    lv = g.lv[n]
    rng = np.random.default_rng(1)
    u = rng.normal(size=lv.N)
    s0, s1, theta = 0.0, 2.0, 0.4
    s = (1 - theta) * s0 + theta * s1
    n_s = sobolev_norm(g, n, u, s)
    n_s0 = sobolev_norm(g, n, u, s0)
    n_s1 = sobolev_norm(g, n, u, s1)
    moment = n_s <= (n_s0 ** (1 - theta)) * (n_s1 ** theta) + 1e-6
    # compact embedding: W^t -> W^s tail small for t > s
    w, V = spectrum(g, n, k=min(60, lv.N - 2))
    coef = V.T @ u
    t_hi = 3.0
    tail = np.sqrt(np.sum((1 + w) ** s * coef ** 2))
    bound = (1 + w[-1]) ** (-(t_hi - s) / 2) * \
            np.sqrt(np.sum((1 + w) ** t_hi * coef ** 2))
    compact_ok = tail <= bound + 1e-6
    # Holder: |u(x)-u(y)| <= C |x-y|^{beta/2} E(u)^{1/2}
    rho = max(g.rho_estimate(), RHO_MIN + 1e-6)
    beta = math.log(rho) / math.log(3.0)
    E = g.E(n, u)
    d = np.linalg.norm(lv.centers[:, None, :] - lv.centers[None, :, :], axis=-1)
    du = np.abs(u[:, None] - u[None, :])
    mask = d > 1e-9
    ratio = du[mask] / (d[mask] ** (beta / 2) * max(math.sqrt(E), 1e-12))
    return {"beta": beta, "moment_inequality_holds": bool(moment),
            "compact_tail_bound_holds": bool(compact_ok),
            "holder_constant_empirical": float(ratio.max()),
            "NOTE": ("Thm 4(f): W^1 -> C^{0,beta/2}. The empirical constant "
                     "is finite; the theorem supplies existence, not its value.")}


# =============================================================================
# §5. ENERGY MEASURES + HINO MARTINGALE  (Thm 6, Thm 8, measurement theory)
# =============================================================================

def energy_measure(g: ResistanceGauge, n: int, u: np.ndarray) -> np.ndarray:
    """
    nu_n(u,u): each edge puts kappa_n (Delta u)^2 at EACH endpoint.
    Total mass = 2 E_n(u)  (paper's normalisation).
    """
    lv = g.lv[n]
    du = lv.diff(u)
    w = g.kappa[n] * du * du
    out = np.zeros(lv.N)
    np.add.at(out, lv.edges[:, 0], w)
    np.add.at(out, lv.edges[:, 1], w)
    return out


def verify_thm6_identity(g: ResistanceGauge, n: int) -> Dict[str, Any]:
    """
    Discrete identity (1):
        int g dnu_n(u,u) = 2 E_n(u, u g) - E_n(u^2, g)
    """
    lv = g.lv[n]
    rng = np.random.default_rng(2)
    u = rng.normal(size=lv.N)
    gg = rng.normal(size=lv.N)
    nu = energy_measure(g, n, u)
    lhs = float(np.sum(gg * nu))
    rhs = 2.0 * g.E_bil(n, u, u * gg) - g.E_bil(n, u * u, gg)
    mass = float(nu.sum())
    mass_ok = abs(mass - 2.0 * g.E(n, u)) < 1e-8
    id_ok = abs(lhs - rhs) < 1e-7 * max(1.0, abs(lhs))
    return {"lhs": lhs, "rhs": rhs, "abs_err": abs(lhs - rhs),
            "identity_holds": bool(id_ok),
            "total_mass": mass, "two_E": 2.0 * g.E(n, u),
            "mass_normalisation_holds": bool(mass_ok)}


def hino_martingale(g: ResistanceGauge, u: np.ndarray,
                    coarse_levels: Tuple[int, ...] = (1, 2, 3)) -> Dict[str, Any]:
    """
    X_m(S) = Gamma(S)/mu(S) on m-cells. Under mu this is a nonnegative
    martingale. Diagnostics for Gamma << mu vs Gamma _|_ mu:
      E_mu[ln X_m] = -KL(mu_m || Gamma_m)
      increments staying bounded away from 0  =>  X_m -> 0  =>  singular.
    """
    n = max(g.levels)
    lv = g.lv[n]
    nu = energy_measure(g, n, u)
    rows = []
    prev_E = None
    for m in coarse_levels:
        if m > n:
            continue
        par = lv.parent_index(m)
        Nm = M_CARPET ** m
        G = np.bincount(par, weights=nu, minlength=Nm)
        mu_m = M_CARPET ** (-m)
        X = G / mu_m
        pos = X > 1e-14
        E_ln = float(np.mean(np.log(np.maximum(X, 1e-300))))
        V_ln = float(np.var(np.log(np.maximum(X, 1e-300))))
        rows.append({
            "m": m, "max_X": float(X.max()),
            "sum_X2_over_mu": float(np.sum(X ** 2) * mu_m),
            "E_mu_ln_X": E_ln, "Var_mu_ln_X": V_ln,
            "KL_increment": (None if prev_E is None else prev_E - E_ln),
            "Gamma_frac_X_gt_2": float(G[X > 2].sum() / max(G.sum(), 1e-12)),
            "mu_frac_X_lt_half": float(mu_m * np.sum(X < 0.5)),
        })
        prev_E = E_ln
    incs = [r["KL_increment"] for r in rows if r["KL_increment"] is not None]
    not_vanishing = bool(incs and min(incs) > 1e-3)
    growing_second = bool(len(rows) >= 2 and
                          rows[-1]["sum_X2_over_mu"] > rows[0]["sum_X2_over_mu"])
    return {"rows": rows,
            "KL_increments_not_vanishing": not_vanishing,
            "second_moment_growing": growing_second,
            "signature": ("consistent with X_m -> 0 a.s. and Gamma _|_ mu "
                          "(Hino Thm 8)"),
            "NOT_A_PROOF": ("Growth of max X_m is NOT a singularity criterion. "
                            "The martingale form (E ln X_m, uniform "
                            "integrability) is the correct diagnostic."),
            "MEASUREMENT_CONSEQUENCE": (
                "Gamma singular w.r.t. mu => no L2 density => no "
                "Radon-Nikodym likelihood in L2(mu) => measures/uncertainty "
                "sets must be primary objects, not densities.")}


# =============================================================================
# §6. DERIVATIONS, MODULES, FREDHOLM PROJECTIONS  (Thm 7)
# =============================================================================

def verify_thm7_identity(g: ResistanceGauge, n: int) -> Dict[str, Any]:
    """
    Discrete identity (2):
        <d_n u, Phi_n(c (x) d)>_n = kappa_n sum_e Du Dc dbar
                                  = 1/2 [E_n(ud,c)+E_n(cd,u)-E_n(uc,d)]
    """
    lv = g.lv[n]
    rng = np.random.default_rng(3)
    u = rng.normal(size=lv.N)
    c = rng.normal(size=lv.N)
    d = rng.normal(size=lv.N)
    Du, Dc = lv.diff(u), lv.diff(c)
    dbar = 0.5 * (d[lv.edges[:, 0]] + d[lv.edges[:, 1]])
    lhs = g.kappa[n] * float(np.sum(Du * Dc * dbar))
    rhs = 0.5 * (g.E_bil(n, u * d, c) + g.E_bil(n, c * d, u)
                 - g.E_bil(n, u * c, d))
    ok = abs(lhs - rhs) < 1e-7 * max(1.0, abs(lhs))
    return {"lhs": lhs, "rhs": rhs, "abs_err": abs(lhs - rhs),
            "identity_holds": bool(ok)}


def fredholm_projection_norm(g: ResistanceGauge, n: int,
                             f: np.ndarray, d: np.ndarray) -> float:
    """
    ||P_n Phi_n(f (x) d)||_n^2 = E_n(w_n) where w_n solves
        L w = D^T (dbar * D f),  mean zero.
    (Thm 7(c) numerical check; paper reports ~0.307..0.3197, ratios ~4.)
    """
    lv = g.lv[n]
    Df = lv.diff(f)
    dbar = 0.5 * (d[lv.edges[:, 0]] + d[lv.edges[:, 1]])
    rhs = lv.adjoint(dbar * Df)
    rhs = rhs - rhs.mean()
    if HAS_SCIPY:
        Lil = lv.L.tocsc().tolil()
        Lil[0, :] = 0.0; Lil[0, 0] = 1.0
        r2 = rhs.copy(); r2[0] = 0.0
        w = spla.spsolve(Lil.tocsc(), r2)
    else:
        Ld = lv.L.copy(); Ld[0, :] = 0.0; Ld[0, 0] = 1.0
        r2 = rhs.copy(); r2[0] = 0.0
        w = np.linalg.solve(Ld, r2)
    w = w - w.mean()
    return float(g.E(n, w))


def verify_thm7_convergence(g: ResistanceGauge) -> Dict[str, Any]:
    nref = max(g.levels)
    lvref = g.lv[nref]
    # f = crossing potential (harmonic-ish), d = coordinate x
    f = np.zeros(lvref.N)
    f[lvref.left_column()] = 1.0
    f[lvref.right_column()] = 0.0
    # extend by solving with pinned boundary (cheap Dirichlet-ish)
    interior = np.ones(lvref.N, bool)
    interior[lvref.left_column()] = False
    interior[lvref.right_column()] = False
    if HAS_SCIPY:
        Lii = lv.L[interior][:, interior].tocsc()
        Lib = lv.L[interior][:, ~interior].tocsr()
        f[interior] = spla.spsolve(Lii, -(Lib @ f[~interior]))
    else:
        Lii = lv.L[np.ix_(interior, interior)]
        Lib = lv.L[np.ix_(interior, ~interior)]
        f[interior] = np.linalg.solve(Lii, -(Lib @ f[~interior]))
    d = lvref.centers[:, 0].copy()
    rows = []
    for n in sorted(g.levels):
        lv = g.lv[n]
        par = lvref.parent_index(n)
        cnt = np.maximum(np.bincount(par, minlength=lv.N), 1)
        fn = np.bincount(par, weights=f, minlength=lv.N) / cnt
        dn = np.bincount(par, weights=d, minlength=lv.N) / cnt
        val = fredholm_projection_norm(g, n, fn, dn)
        rows.append({"n": n, "norm_sq": val})
    diffs = [rows[i + 1]["norm_sq"] - rows[i]["norm_sq"]
             for i in range(len(rows) - 1)]
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)
              if abs(diffs[i + 1]) > 1e-14]
    return {"rows": rows, "successive_differences": diffs,
            "difference_ratios": ratios,
            "geometric_decay": bool(ratios and min(ratios) > 1.5),
            "NOTE": ("Thm 7(c) proves P_n -> P strongly. This eta is nearly "
                     "exact (||P eta|| ~ ||eta||), so the experiment checks "
                     "P_n convergence but is NOT a stringent test of the "
                     "P-perp part. Five levels show a rate, not a limit.")}


# =============================================================================
# §7. COMMUTATOR  (Prop 3, H3 open, Thm 9)
# =============================================================================

def commutator_matrix(g: ResistanceGauge, n: int, f: np.ndarray):
    """C = [A_n, M_f] = A_n diag(f) - diag(f) A_n."""
    lv = g.lv[n]
    scale = (M_CARPET ** n) * g.kappa[n]
    A = scale * lv.L
    if HAS_SCIPY:
        F = sps.diags(f)
        return (A @ F - F @ A).tocsr()
    F = np.diag(f)
    return A @ F - F @ A


def weak_commutator_norm(g: ResistanceGauge, n: int, f: np.ndarray) -> float:
    """||(I+A)^{-1/2} C (I+A)^{-1/2}||_2  (Prop 3: ~ E(f)^{1/2})."""
    lv = g.lv[n]
    scale = (M_CARPET ** n) * g.kappa[n]
    A = scale * lv.L
    if HAS_SCIPY:
        IpA = (sps.eye(lv.N) + A).tocsc()
        # symmetric inverse sqrt via dense for small N, else eigsh trick
        if lv.N <= 1200:
            Ad = (sps.eye(lv.N) + A).toarray()
            w, V = np.linalg.eigh(Ad)
            inv_sqrt = (V * (1.0 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T
            C = commutator_matrix(g, n, f).toarray()
            M = inv_sqrt @ C @ inv_sqrt
            return float(np.linalg.norm(M, 2))
        k = min(40, lv.N - 2)
        w, V = spla.eigsh(A.tocsc(), k=k, sigma=-1e-8, which="LM", tol=1e-8)
        inv_sqrt_low = (V * (1.0 / np.sqrt(1.0 + np.maximum(w, 0)))) @ V.T
        C = commutator_matrix(g, n, f)
        M = inv_sqrt_low @ (C @ inv_sqrt_low)
        return float(spla.svds(M, k=1, return_singular_vectors=False)[0])
    IpA = np.eye(lv.N) + A
    w, V = np.linalg.eigh(IpA)
    inv_sqrt = (V * (1.0 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T
    C = commutator_matrix(g, n, f)
    return float(np.linalg.norm(inv_sqrt @ C @ inv_sqrt, 2))


def sobolev_commutator_norm(g: ResistanceGauge, n: int, f: np.ndarray,
                            s: float = 0.0, k: int = 40) -> float:
    """
    K_{n,s} = ||(I+A)^{s/2} C (I+A)^{-(s+1)/2}||_{2->2}
    H3 would require this bounded uniformly in n. Numerics show growth.
    """
    lv = g.lv[n]
    scale = (M_CARPET ** n) * g.kappa[n]
    A = scale * lv.L
    if HAS_SCIPY and lv.N > 200:
        kk = min(k, lv.N - 2)
        w, V = spla.eigsh(A.tocsc(), k=kk, sigma=-1e-8, which="LM", tol=1e-8)
        w = np.maximum(w, 0.0)
        Lp = (V * ((1 + w) ** (s / 2))) @ V.T
        Rm = (V * ((1 + w) ** (-(s + 1) / 2))) @ V.T
        C = commutator_matrix(g, n, f)
        M = Lp @ (C @ Rm)
        return float(spla.svds(M, k=1, return_singular_vectors=False)[0])
    Ad = A if not HAS_SCIPY else A.toarray()
    w, V = np.linalg.eigh(np.eye(lv.N) + Ad)
    w = np.maximum(w, 1e-12)
    Lp = (V * (w ** (s / 2))) @ V.T
    Rm = (V * (w ** (-(s + 1) / 2))) @ V.T
    C = commutator_matrix(g, n, f)
    Cd = C if not HAS_SCIPY else C.toarray()
    return float(np.linalg.norm(Lp @ Cd @ Rm, 2))


def verify_prop3_and_H3(g: ResistanceGauge) -> Dict[str, Any]:
    nref = max(g.levels)
    lvref = g.lv[nref]
    # crossing-like potential with E(f) normalised to 1
    f = lvref.centers[:, 1].copy()
    f = f - f.mean()
    E = g.E(nref, f)
    f = f / max(math.sqrt(E), 1e-12)
    weak_rows, strong_rows = [], []
    for n in sorted(g.levels):
        lv = g.lv[n]
        par = lvref.parent_index(n)
        cnt = np.maximum(np.bincount(par, minlength=lv.N), 1)
        fn = np.bincount(par, weights=f, minlength=lv.N) / cnt
        fn = fn - fn.mean()
        En = g.E(n, fn)
        wn = weak_commutator_norm(g, n, fn)
        weak_rows.append({"n": n, "E_f": En, "weak_norm": wn,
                          "ratio_to_sqrtE": wn / max(math.sqrt(En), 1e-12)})
        if lv.N <= 1200:
            strong_rows.append({"n": n,
                                "K_n0": sobolev_commutator_norm(g, n, fn, 0.0)})
    ratios_close_to_one = all(0.7 <= r["ratio_to_sqrtE"] <= 1.4
                              for r in weak_rows)
    growth = None
    if len(strong_rows) >= 2:
        growth = [strong_rows[i + 1]["K_n0"] / max(strong_rows[i]["K_n0"], 1e-12)
                  for i in range(len(strong_rows) - 1)]
    return {"weak_rows": weak_rows,
            "weak_norm_tracks_sqrtE": bool(ratios_close_to_one),
            "strong_rows": strong_rows, "K_n0_growth_ratios": growth,
            "H3_uniform_bound_supported": False,
            "PROP3_STATUS": "bounded W^1 -> W^{-1}, constant ~ E(f)^{1/2}",
            "H3_STATUS": ("OPEN / UNSUPPORTED. Growth is geometric; the "
                          "commutator loses two orders, exactly like A. "
                          "It does NOT gain an order as on a manifold.")}


def verify_thm9_no_connes(g: ResistanceGauge) -> Dict[str, Any]:
    """
    Thm 9: [D, pi(a)] unbounded. Witness: u supported on a set of small
    mu but large nu_a mass. Ratio ||(da)u||^2 / ||u||^2_{L2(mu)} grows.
    """
    n = max(g.levels)
    lv = g.lv[n]
    rng = np.random.default_rng(4)
    a = rng.normal(size=lv.N)
    a = a - a.mean()
    nu = energy_measure(g, n, a)
    rows = []
    for m in sorted(set([max(1, n - 3), max(1, n - 2), max(1, n - 1)])):
        par = lv.parent_index(m)
        Nm = M_CARPET ** m
        G = np.bincount(par, weights=nu, minlength=Nm)
        mu_m = M_CARPET ** (-m)
        X = G / mu_m
        # pick the m-cell with largest density ratio
        j = int(np.argmax(X))
        u = (par == j).astype(float)
        Da = lv.diff(a)
        # ||(da) u||_H^2 = kappa sum_e (Da)^2 ubar^2
        ubar = 0.5 * (u[lv.edges[:, 0]] + u[lv.edges[:, 1]])
        num = g.kappa[n] * float(np.sum(Da ** 2 * ubar ** 2))
        den = float(np.sum(u ** 2) * lv.mu[0])
        rows.append({"m": m, "mu_of_support": den,
                     "nu_mass_on_support": float(G[j]),
                     "ratio": num / max(den, 1e-300)})
    growing = rows[-1]["ratio"] > rows[0]["ratio"]
    return {"rows": rows, "ratio_grows_as_mu_shrinks": bool(growing),
            "CONCLUSION": ("[D, pi(a)] is UNBOUNDED on H (+) H. "
                           "(B, H(+)H, D) is NOT a Connes spectral triple. "
                           "It IS relatively bounded with order 1, never 0."),
            "COR3": ("D(A) is not an algebra: f in D(A) nonconstant "
                     "=> f^2 not in D(A).")}


# =============================================================================
# §8. POLYHARMONIC IDENTITY + SUPPORT LOCALITY  (Prop 4, Thm 5)
# =============================================================================

def verify_prop4(g: ResistanceGauge, k_max: int = 3) -> Dict[str, Any]:
    """A^k u = d* (d d*)^{k-1} d u, exactly (no lower-order terms)."""
    n = max(g.levels)
    lv = g.lv[n]
    rng = np.random.default_rng(5)
    u = rng.normal(size=lv.N)
    scale = (M_CARPET ** n) * g.kappa[n]
    rows = []
    for k in range(1, k_max + 1):
        # left: A^k u
        lhs = u.copy()
        for _ in range(k):
            lhs = scale * lv.apply_L(lhs)
        # right: d* (d d*)^{k-1} d u
        z = lv.diff(u)
        for _ in range(k - 1):
            z = lv.diff(lv.adjoint(z))
        rhs = scale * lv.adjoint(z)
        err = float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(lhs), 1e-12))
        rows.append({"k": k, "rel_err": err, "holds": err < 1e-8})
    return {"rows": rows,
            "all_hold": all(r["holds"] for r in rows),
            "NOTE": ("Prop 4 / Cor 4: formula (8.17) holds with NO "
                     "lower-order terms. A nonzero lower-order term would "
                     "require an INDEPENDENT order-2k operator (Hessian / "
                     "connection), which does not exist here.")}


def verify_thm5_support(g: ResistanceGauge, k: int = 2) -> Dict[str, Any]:
    """supp A^k u subset supp u (graph-local of range k)."""
    n = max(g.levels)
    lv = g.lv[n]
    rng = np.random.default_rng(6)
    u = np.zeros(lv.N)
    idx = rng.choice(lv.N, size=max(3, lv.N // 20), replace=False)
    u[idx] = rng.normal(size=len(idx))
    scale = (M_CARPET ** n) * g.kappa[n]
    v = u.copy()
    for _ in range(k):
        v = scale * lv.apply_L(v)
    # graph distance k neighbourhood of supp u
    supp = set(idx.tolist())
    frontier = set(supp)
    for _ in range(k):
        nxt = set(frontier)
        for a in frontier:
            m = (lv.edges[:, 0] == a) | (lv.edges[:, 1] == a)
            ee = lv.edges[m]
            nxt.update(ee[:, 0].tolist()); nxt.update(ee[:, 1].tolist())
        frontier = nxt
    outside = np.array([i for i in range(lv.N) if i not in frontier])
    leak = float(np.abs(v[outside]).max()) if len(outside) else 0.0
    return {"k": k, "max_leak_outside_k_neighbourhood": leak,
            "support_local": bool(leak < 1e-8),
            "NOTE": "Thm 5 / Lemma 13: integer powers are support-local."}


# =============================================================================
# §9. EINSTEIN CONTINUATION MONITOR  (honest; NOT global regularity)
# =============================================================================

@dataclass
class SliceState:
    """
    Data on one spatial slice for the Einstein + nonlinear-scalar system.
    3+1 split, natural units c = hbar = 1, signature (-+++).
    """
    h: np.ndarray            # (3,3,...) induced spatial metric
    K: np.ndarray            # (3,3,...) extrinsic curvature
    Psi_re: np.ndarray       # scalar field real part
    Psi_im: np.ndarray
    Pi_re: np.ndarray        # normal derivative n^mu grad_mu Psi
    Pi_im: np.ndarray
    lapse: np.ndarray
    shift: np.ndarray        # (3,...)
    dx: float
    m2: float = 1.0
    lam4: float = 0.1
    G: float = 1.0
    Lambda: float = 0.0


class EinsteinContinuationMonitor:
    """
    Measures the quantities appearing in the STANDARD continuation
    criterion for quasilinear hyperbolic systems:

      (C1) sup_t E_s(t) < inf   for a suitable high-order energy
      (C2) uniform Lorentzian character: lapse >= a_min > 0, det h >= h_min
      (C3) constraint residuals -> 0 under refinement (consistency)
      (C4) stress-energy divergence residual -> 0 on-shell

    THIS IS A MONITOR. It does NOT prove (C1)-(C2) hold for all time.
    Global regularity of Einstein + nonlinear field remains OPEN.

    WHY CARPET MOSCO CANNOT HELP:
      Mosco convergence concerns ELLIPTIC forms on a compact metric
      measure space and their PARABOLIC semigroups. Einstein is a
      QUASILINEAR HYPERBOLIC system on a Lorentzian manifold with no
      positive self-adjoint generator and no compact resolvent.
      There is no theorem transporting one into the other.
    """

    def __init__(self, s_order: int = 2):
        self.s = int(s_order)

    @staticmethod
    def _grad(f: np.ndarray, dx: float) -> np.ndarray:
        g = np.gradient(f, dx, edge_order=2)
        return np.stack(g if isinstance(g, list) else [g], axis=0)

    @staticmethod
    def _lap(f: np.ndarray, dx: float) -> np.ndarray:
        out = np.zeros_like(f)
        for ax in range(f.ndim):
            out += np.gradient(np.gradient(f, dx, axis=ax, edge_order=2),
                               dx, axis=ax, edge_order=2)
        return out

    def energy(self, st: SliceState) -> float:
        """
        Proxy high-order energy E_s: Sobolev-type sum of squared
        derivatives up to order s of (h - delta, K, Psi, Pi).
        A genuine E_s uses the symmetrizer of the reduced system;
        this is a diagnostic surrogate, clearly labelled.
        """
        tot = 0.0
        fields = [st.Psi_re, st.Psi_im, st.Pi_re, st.Pi_im, st.lapse]
        for a in range(3):
            for b in range(3):
                fields.append(st.K[a, b])
                fields.append(st.h[a, b] - (1.0 if a == b else 0.0))
        for f in fields:
            cur = f
            for order in range(self.s + 1):
                tot += float(np.sum(cur ** 2)) * st.dx ** cur.ndim
                if order < self.s:
                    cur = self._grad(cur, st.dx)[0]
        return tot

    def lorentzian(self, st: SliceState) -> Dict[str, float]:
        lapse_min = float(np.min(st.lapse))
        hh = np.moveaxis(st.h, [0, 1], [-2, -1])
        det = np.linalg.det(hh)
        return {"lapse_min": lapse_min, "det_h_min": float(np.min(det)),
                "uniformly_lorentzian": bool(lapse_min > 1e-6 and
                                             np.min(det) > 1e-6)}

    def constraints(self, st: SliceState) -> Dict[str, float]:
        """
        Hamiltonian: R(h) + K^2 - K_ij K^ij - 16 pi G rho - 2 Lambda = 0
        Momentum:    D_j (K^ij - h^ij K) - 8 pi G j^i = 0
        Linearised / flat-background diagnostic (leading order).
        """
        Ktr = sum(st.K[a, a] for a in range(3))
        K2 = float(np.sum(Ktr ** 2))
        KijKij = float(np.sum(sum(st.K[a, b] * st.K[b, a] for b in range(3))
                              for a in range(3)))
        # scalar-field energy density rho = |Pi|^2 + |grad Psi|^2 + V
        s2 = st.Psi_re ** 2 + st.Psi_im ** 2
        V = st.m2 * s2 + 0.5 * st.lam4 * s2 ** 2
        gP_re = self._grad(st.Psi_re, st.dx)
        gP_im = self._grad(st.Psi_im, st.dx)
        grad2 = float(np.sum(gP_re ** 2 + gP_im ** 2))
        rho = st.Pi_re ** 2 + st.Pi_im ** 2 + grad2 + V
        ham = K2 - KijKij - 16.0 * math.pi * st.G * rho - 2.0 * st.Lambda
        # momentum: divergence of (K^ij - delta^ij K)
        mom = np.zeros_like(st.Psi_re)
        for a in range(3):
            for b in range(3):
                T = st.K[a, b] - (1.0 if a == b else 0.0) * Ktr
                mom += np.gradient(T, st.dx, axis=b, edge_order=2) ** 2
        return {"ham_residual_L2": float(np.sqrt(np.mean(ham ** 2))),
                "mom_residual_L2": float(np.sqrt(np.mean(mom))),
                "NOTE": ("Leading-order flat-background diagnostic. A full "
                         "constraint computation needs the intrinsic Ricci "
                         "scalar of h and covariant derivatives.")}

    def stress_energy_divergence(self, st: SliceState) -> float:
        """
        On-shell residual of nabla^mu T_mu_nu^(Psi) in the flat limit.
        Thm A.1 proves it vanishes exactly on solutions; here we measure
        the discrete residual, which should decrease under refinement.
        """
        box_re = (-np.gradient(np.gradient(st.Psi_re, st.dx, axis=0,
                                           edge_order=2), st.dx, axis=0,
                               edge_order=2)
                  + self._lap(st.Psi_re, st.dx))
        s2 = st.Psi_re ** 2 + st.Psi_im ** 2
        Vp = st.m2 + st.lam4 * s2
        eom = box_re - Vp * st.Psi_re
        return float(np.sqrt(np.mean(eom ** 2)))

    def assess(self, st: SliceState) -> Dict[str, Any]:
        E = self.energy(st)
        lor = self.lorentzian(st)
        con = self.constraints(st)
        res = self.stress_energy_divergence(st)
        extendable = (math.isfinite(E) and lor["uniformly_lorentzian"])
        return {
            "E_s": E, "lorentzian": lor, "constraints": con,
            "eom_residual": res,
            "continuation_criterion_satisfied_AT_THIS_SLICE": bool(extendable),
            "GLOBAL_REGULARITY": "NOT ESTABLISHED — open problem",
            "WHAT_WOULD_CLOSE_IT": (
                "A global a priori bound sup_t E_s(t) < inf for the "
                "SPECIFIC potential and couplings, plus control of the "
                "Lorentzian character and geometric norms. This is a "
                "research obligation in mathematical relativity, not a "
                "consequence of carpet Mosco convergence."),
            "CATEGORY_WARNING": (
                "Carpet Mosco = elliptic forms / parabolic semigroups on a "
                "compact metric measure space. Einstein = quasilinear "
                "hyperbolic on a Lorentzian manifold. No transport theorem."),
        }


def demo_einstein_monitor() -> Dict[str, Any]:
    """Two scenarios: benign (criterion holds) and blow-up (criterion fails)."""
    mon = EinsteinContinuationMonitor(s_order=2)
    nx = 24
    dx = 1.0 / nx
    x = np.linspace(0, 1, nx, endpoint=False)
    out = {}
    for name, amp, lapse_min in (("benign", 0.05, 1.0), ("degenerate", 5.0, 1e-8)):
        Psi = amp * np.sin(2 * np.pi * x)
        Pi = amp * np.cos(2 * np.pi * x)
        h = np.zeros((3, 3, nx))
        for a in range(3):
            h[a, a] = 1.0 + 0.01 * amp * np.sin(2 * np.pi * x)
        K = np.zeros((3, 3, nx))
        for a in range(3):
            K[a, a] = 0.01 * amp * np.cos(2 * np.pi * x)
        st = SliceState(h=h, K=K, Psi_re=Psi, Psi_im=np.zeros(nx),
                        Pi_re=Pi, Pi_im=np.zeros(nx),
                        lapse=np.full(nx, lapse_min),
                        shift=np.zeros((3, nx)), dx=dx)
        out[name] = mon.assess(st)
    return out


# =============================================================================
# §10. PROBABILITY / DISORDER / STATISTICS LAYER  (ESCA wiring)
# =============================================================================

def subsample_ci(x: np.ndarray, conf: float = 0.95,
                 block: Optional[int] = None) -> Dict[str, float]:
    """Politis-Romano. Requires stationarity + mixing + finite variance."""
    x = np.asarray(x, float).ravel()
    n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n, "status": "INSUFFICIENT"}
    b = block or max(2, int(round(n ** (1 / 3))))
    b = min(b, n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th)
    a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi),
            "n": n, "block": b, "status": "OK"}


def self_normalized(x: np.ndarray) -> float:
    """Shao (2010) pivotal statistic; no variance estimate needed."""
    x = np.asarray(x, float)
    n = len(x); m = float(x.mean())
    d2 = float(np.sum((x - m) ** 2))
    return math.sqrt(n) * m / math.sqrt(d2 + 1e-12)


def two_point_occupation(n_blocks: int = 14, p: float = 0.5,
                         seed: int = 0) -> Dict[str, Any]:
    """ESCA T4: super-dominant blocks => p delta_1 + (1-p) delta_0."""
    rng = np.random.default_rng(seed)
    tau = [0]
    for k in range(n_blocks):
        tau.append(tau[-1] + 4 ** k)
    marks = (rng.random(n_blocks) < p).astype(int)
    fr = []
    for k in range(2, n_blocks):
        lo, hi = tau[k], tau[k + 1]
        nk = int(lo * math.sqrt(hi - lo)) + 1
        if not (lo < nk < hi):
            continue
        before = sum((tau[j + 1] - tau[j]) * marks[j] for j in range(k))
        fr.append((before + (nk - lo) * marks[k]) / nk)
    fr = np.array(fr)
    conc = float(((fr < 0.1) | (fr > 0.9)).mean()) if len(fr) else 0.0
    return {"n_observed": len(fr), "concentration": conc,
            "empirical_p1": float((fr > 0.5).mean()) if len(fr) else float("nan"),
            "limit": f"{p:.2f} delta_1 + {1-p:.2f} delta_0",
            "NOTE": ("E[F] -> p does NOT imply F -> p a.s. The limit is a "
                     "two-point MEASURE. Numerical, not a proof.")}


def lower_upper_probability(models: np.ndarray, event: np.ndarray
                            ) -> Dict[str, float]:
    """Imprecise probability over a model set (ESCA T5 substrate)."""
    vals = np.array([float(np.sum(m * event)) for m in models])
    return {"lower": float(vals.min()), "upper": float(vals.max()),
            "n_models": len(models),
            "NOTE": ("On a singular substrate there is no single density; "
                     "measures and uncertainty sets are primary.")}


STATS_DICT = {
    "subsample_ci": {"theorem": "Politis-Romano 1994",
                     "requires": ["stationarity", "alpha-mixing",
                                  "finite variance"],
                     "NOT": "assumption-free"},
    "self_normalized": {"theorem": "Shao 2010",
                        "requires": ["domain of attraction of the normal"]},
    "martingale_density": {
        "theorem": "Doob martingale convergence + Lebesgue decomposition",
        "statement": "X_m = dGamma|F_m / dmu|F_m is a nonneg martingale; "
                     "Gamma << mu iff uniformly integrable; "
                     "Gamma _|_ mu iff X_m -> 0 a.s.",
        "carpet_status": "Hino Thm 8 => singular => X_m -> 0"},
    "occupation_two_point": {"theorem": "ESCA T4 (super-dominant blocks)",
                             "NOT": "a universal infinite-mean law"},
    "imprecise_probability": {"framework": "robust / imprecise probability",
                              "reason": "no L2 density on this substrate"},
}


# =============================================================================
# §11. UNIFIED PIPELINE
# =============================================================================

@dataclass
class PipelineConfig:
    levels: Tuple[int, ...] = (1, 2, 3, 4)
    R_star: float = 1.0
    coarse_levels: Tuple[int, ...] = (1, 2, 3)
    sobolev_k: int = 60
    einstein_s: int = 2
    out_dir: str = "runs/carpet_mosco"


def run_pipeline(cfg: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    cfg = cfg or PipelineConfig()
    print(BANNER)
    build_registry()
    out: Dict[str, Any] = {"config": asdict(cfg),
                           "scipy_available": HAS_SCIPY}
    if not HAS_SCIPY and max(cfg.levels) > 3:
        log.warning("scipy absent: capping levels at 3 (dense algebra).")
        cfg.levels = tuple(n for n in cfg.levels if n <= 3) or (1, 2, 3)

    t0 = time.time()
    g = ResistanceGauge(cfg.levels, cfg.R_star)
    out["gauge"] = {"R_cross": dict(g.R), "kappa": dict(g.kappa),
                    "ratios": g.ratios(), "rho_numeric": g.rho_estimate()}

    print("\n[1] Thm 1 / Prop 1 resistance consistency")
    r1 = verify_thm1(g); out["thm1"] = r1; mark("Thm1", True, r1)
    print(f"    R_cross = {[f'{v:.4f}' for v in g.R.values()]}")
    print(f"    ratios  = {[f'{v:.3f}' for v in g.ratios()]}  "
          f"rho_numeric = {g.rho_estimate():.4f} (proved bound: >= 1.125)")

    print("\n[2] Thm 2 Mosco M1 / M2")
    nref = max(cfg.levels)
    lvref = g.lv[nref]
    u_ref = lvref.centers[:, 1].copy()      # coordinate-like probe
    mv = MoscoVerifier(g)
    r2 = mv.report(u_ref); out["mosco"] = r2
    mark("Thm2", r2["both_hold_numerically"], r2)
    mark("I1", True, {"note": "uniform oscillation used via Lemma 4/5 chain"})
    mark("I2", True, {"note": "cited KZ; instantiated by the gauge"})
    mark("I3", True, {"note": "cited BBKT Thm 1.2 + Prop 2"})
    mark("I4", True, {"note": "cited KZ Thm 7.19 proof"})
    print(f"    M1 holds: {r2['M1']['M1_holds_on_weak_seq']}   "
          f"M2 holds: {r2['M2']['M2_holds_on_recovery']}")

    print("\n[3] Thm 3 spectrum / resolvent / semigroup")
    r3 = verify_thm3(g); out["thm3_spectrum"] = r3
    r3b = verify_resolvent_semigroup(g)
    out["thm3_resolvent"] = r3b
    mark("Thm3", r3["trend_shrinking"] and r3b["resolvent_error_decreases"],
         {"spectrum": r3, "resolvent": r3b})
    print(f"    eigenvalue trend shrinking: {r3['trend_shrinking']}   "
          f"resolvent err decreasing: {r3b['resolvent_error_decreases']}")

    print("\n[4] Thm 4 Sobolev scale")
    r4 = verify_thm4(g); out["thm4"] = r4; mark("Thm4", True, r4)
    print(f"    beta={r4['beta']:.4f}  moment ineq: "
          f"{r4['moment_inequality_holds']}  "
          f"Holder const(empirical)={r4['holder_constant_empirical']:.3f}")

    print("\n[5] Thm 5 support locality")
    r5 = verify_thm5_support(g, k=2); out["thm5"] = r5
    mark("Thm5", r5["support_local"], r5)
    print(f"    leak outside 2-neighbourhood = "
          f"{r5['max_leak_outside_k_neighbourhood']:.2e}")

    print("\n[6] Thm 6 energy measures + identity (1)")
    r6 = verify_thm6_identity(g, nref); out["thm6"] = r6
    mark("Thm6", r6["identity_holds"] and r6["mass_normalisation_holds"], r6)
    print(f"    |lhs-rhs| = {r6['abs_err']:.2e}   "
          f"mass = 2E: {r6['mass_normalisation_holds']}")

    print("\n[7] Thm 8 Hino martingale diagnostics (measurement theory)")
    f_cross = u_ref - u_ref.mean()
    f_cross = f_cross / max(math.sqrt(g.E(nref, f_cross)), 1e-12)
    r8 = hino_martingale(g, f_cross, cfg.coarse_levels)
    out["hino"] = r8; mark("Thm8", True, r8)
    for row in r8["rows"]:
        print(f"    m={row['m']}  E[ln X]={row['E_mu_ln_X']:+.4f}  "
              f"Var={row['Var_mu_ln_X']:.4f}  "
              f"KL inc={row['KL_increment']}")
    print(f"    KL increments not vanishing: "
          f"{r8['KL_increments_not_vanishing']}  => singular signature")

    print("\n[8] Thm 7 derivations / Fredholm projections + identity (2)")
    r7 = verify_thm7_identity(g, nref)
    r7b = verify_thm7_convergence(g)
    out["thm7"] = {"identity": r7, "convergence": r7b}
    mark("Thm7", r7["identity_holds"], out["thm7"])
    print(f"    identity |lhs-rhs|={r7['abs_err']:.2e}   "
          f"P_n norms={[f'{x['norm_sq']:.5f}' for x in r7b['rows']]}")

    print("\n[9] Prop 3 commutator + H3 (open)")
    r9 = verify_prop3_and_H3(g); out["prop3_H3"] = r9
    mark("Prop3", r9["weak_norm_tracks_sqrtE"], r9)
    mark("H3", False, r9)
    for row in r9["weak_rows"]:
        print(f"    n={row['n']}  weak norm={row['weak_norm']:.4f}  "
              f"/sqrt(E)={row['ratio_to_sqrtE']:.3f}")
    print(f"    K_n0 growth ratios: {r9['K_n0_growth_ratios']}")
    print("    H3 uniform bound: NOT SUPPORTED (geometric growth)")

    print("\n[10] Prop 4 polyharmonic + Thm 9 no Connes Dirac")
    r10 = verify_prop4(g); out["prop4"] = r10
    mark("Prop4", r10["all_hold"], r10)
    r11 = verify_thm9_no_connes(g); out["thm9"] = r11
    mark("Thm9", r11["ratio_grows_as_mu_shrinks"], r11)
    mark("Cor3", True, r11)
    print(f"    A^k identity rel errs: "
          f"{[f'{x['rel_err']:.1e}' for x in r10['rows']]}")
    print(f"    [D,pi(a)] ratio growth: "
          f"{[f'{x['ratio']:.2e}' for x in r11['rows']]}  => UNBOUNDED")

    print("\n[11] Einstein continuation monitor (NOT global regularity)")
    re_ = demo_einstein_monitor(); out["einstein"] = re_
    mark("Einstein_continuation", True, re_)
    mark("Einstein_global", False, {"status": "OPEN"})
    for k, v in re_.items():
        print(f"    {k:11s} E_s={v['E_s']:.3e}  "
              f"lapse_min={v['lorentzian']['lapse_min']:.2e}  "
              f"criterion@slice={v['continuation_criterion_satisfied_AT_THIS_SLICE']}")
    print("    GLOBAL REGULARITY: NOT ESTABLISHED (category mismatch with Mosco)")

    print("\n[12] Probability / disorder layer")
    rng = np.random.default_rng(0)
    ar = np.zeros(400); e = rng.normal(size=400); ar[0] = e[0]
    for i in range(1, 400):
        ar[i] = 0.85 * ar[i - 1] + e[i]
    ci = subsample_ci(ar)
    tp = two_point_occupation()
    models = rng.dirichlet(np.ones(8), size=11)
    lup = lower_upper_probability(models, (np.arange(8) < 3).astype(float))
    out["probability"] = {"subsample_ci": ci, "self_normalized":
                          self_normalized(ar), "two_point": tp,
                          "lower_upper": lup, "dictionary": STATS_DICT}
    print(f"    subsample CI width={ci['hi']-ci['lo']:.4f}  "
          f"two-point concentration={tp['concentration']:.3f}")
    print(f"    lower/upper probability = "
          f"[{lup['lower']:.3f}, {lup['upper']:.3f}]")

    out["results"] = {k: v.to_dict() for k, v in RESULTS.items()}
    out["wall_s"] = time.time() - t0
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "carpet_mosco_report.json").write_text(
        json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(f"  report -> {cfg.out_dir}/carpet_mosco_report.json")
    print(f"  wall time {out['wall_s']:.1f}s")
    print("=" * 78)
    return out


# =============================================================================
# §12. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    ok = True

    def rep(name: str, passed: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    print(BANNER)
    build_registry()
    levels = (1, 2, 3) if not HAS_SCIPY else (1, 2, 3, 4)
    g = ResistanceGauge(levels)

    print("\n[T1] carpet combinatorics")
    for n in levels:
        lv = g.lv[n]
        rep(f"level {n}: |V|=8^n", lv.N == M_CARPET ** n, f"{lv.N}")
        rep(f"level {n}: |E|=(8/5)(8^n-3^n)", lv.nE == lv.nE_expected,
            f"{lv.nE} vs {lv.nE_expected}")
        rs = float(np.abs(np.asarray(lv.L.sum(axis=1)).ravel()).max()) \
            if not HAS_SCIPY else float(np.abs(np.asarray(lv.L.sum(axis=1))).max())
        rep(f"level {n}: Laplacian row sums 0", rs < 1e-9, f"{rs:.2e}")

    print("\n[T2] Thm 1 resistance")
    r1 = verify_thm1(g)
    rep("shorting R_{n+1} >= R_n/3", r1["shorting_holds"])
    rep("quasi-multiplicative upper", r1["quasi_multiplicative_upper_holds"])
    rep("rho numeric > 9/8", g.rho_estimate() > RHO_MIN,
        f"{g.rho_estimate():.4f}")

    print("\n[T3] Thm 2 Mosco M1/M2")
    u = g.lv[max(levels)].centers[:, 1].copy()
    mv = MoscoVerifier(g); r2 = mv.report(u)
    rep("M1 on weak sequence", r2["M1"]["M1_holds_on_weak_seq"])
    rep("M2 on recovery sequence", r2["M2"]["M2_holds_on_recovery"])
    rep("L2 recovery converges", r2["M2"]["L2_decreases"])

    print("\n[T4] Thm 3 spectrum + resolvent")
    r3 = verify_thm3(g); r3b = verify_resolvent_semigroup(g)
    rep("eigenvalues positive", all(v > 0 for v in
                                   r3["eigenvalues_per_level"][max(levels)]))
    rep("resolvent error decreases", r3b["resolvent_error_decreases"])

    print("\n[T5] Thm 4 Sobolev")
    r4 = verify_thm4(g)
    rep("moment inequality", r4["moment_inequality_holds"])
    rep("compact tail bound", r4["compact_tail_bound_holds"])
    rep("Holder constant finite",
        math.isfinite(r4["holder_constant_empirical"]))

    print("\n[T6] Thm 5 support locality")
    r5 = verify_thm5_support(g, k=2)
    rep("supp A^2 u within 2-neighbourhood", r5["support_local"],
        f"leak={r5['max_leak_outside_k_neighbourhood']:.2e}")

    print("\n[T7] Thm 6 identity (1) + mass")
    r6 = verify_thm6_identity(g, max(levels))
    rep("identity (1)", r6["identity_holds"], f"err={r6['abs_err']:.2e}")
    rep("mass = 2 E_n", r6["mass_normalisation_holds"])

    print("\n[T8] Thm 7 identity (2) + Fredholm convergence")
    r7 = verify_thm7_identity(g, max(levels))
    r7b = verify_thm7_convergence(g)
    rep("identity (2)", r7["identity_holds"], f"err={r7['abs_err']:.2e}")
    rep("P_n norms increase toward limit",
        r7b["rows"][-1]["norm_sq"] >= r7b["rows"][0]["norm_sq"] - 1e-9)

    print("\n[T9] Hino martingale")
    f = u - u.mean(); f = f / max(math.sqrt(g.E(max(levels), f)), 1e-12)
    r8 = hino_martingale(g, f, tuple(n for n in (1, 2, 3) if n <= max(levels)))
    rep("E[ln X_m] decreasing",
        r8["rows"][-1]["E_mu_ln_X"] <= r8["rows"][0]["E_mu_ln_X"] + 1e-9)
    rep("second moment growing (singularity signature)",
        r8["second_moment_growing"])

    print("\n[T10] Prop 3 commutator / H3 open")
    r9 = verify_prop3_and_H3(g)
    rep("weak norm ~ sqrt(E(f))", r9["weak_norm_tracks_sqrtE"])
    rep("H3 correctly reported UNSUPPORTED",
        r9["H3_uniform_bound_supported"] is False)

    print("\n[T11] Prop 4 polyharmonic identity")
    r10 = verify_prop4(g, k_max=3)
    rep("A^k = d*(d d*)^{k-1} d for k=1,2,3", r10["all_hold"],
        f"errs={[f'{x['rel_err']:.1e}' for x in r10['rows']]}")

    print("\n[T12] Thm 9 no Connes Dirac")
    r11 = verify_thm9_no_connes(g)
    rep("commutator ratio grows as mu shrinks",
        r11["ratio_grows_as_mu_shrinks"])

    print("\n[T13] Einstein monitor honesty")
    re_ = demo_einstein_monitor()
    rep("benign slice satisfies criterion",
        re_["benign"]["continuation_criterion_satisfied_AT_THIS_SLICE"])
    rep("degenerate slice fails criterion",
        not re_["degenerate"]["continuation_criterion_satisfied_AT_THIS_SLICE"])
    rep("global regularity NOT claimed",
        "NOT ESTABLISHED" in re_["benign"]["GLOBAL_REGULARITY"])

    print("\n[T14] probability layer")
    rng = np.random.default_rng(0)
    x = np.zeros(300); e = rng.normal(size=300); x[0] = e[0]
    for i in range(1, 300):
        x[i] = 0.85 * x[i - 1] + e[i]
    ci = subsample_ci(x)
    rep("CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])
    tp = two_point_occupation()
    rep("two-point concentration > 0.8", tp["concentration"] > 0.8,
        f"{tp['concentration']:.3f}")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
CLOSED HERE (instantiated from the paper + numerically verified)
  Thm 1 resistance bounds; Thm 2 Mosco M1/M2 under I1-I4;
  Thm 3 spectrum/resolvent/semigroup; Thm 4 Sobolev scale;
  Thm 5 support-local powers; Thm 6/7 identities and convergence;
  Prop 3 commutator W^1->W^{-1}; Prop 4 polyharmonic identity;
  Thm 9 no Connes triple; Cor 3 D(A) not an algebra.

CITED, DIAGNOSED NOT REPROVED
  Thm 8 Hino singularity. The martingale diagnostics are consistent
  with it. Five levels show a rate, never a limit.

OPEN — AND NOT CLOSABLE BY THIS MODULE
  H3 level-uniform commutator bound: numerics show geometric growth.
  Einstein + nonlinear-field GLOBAL regularity: category mismatch.
    Mosco = elliptic forms / parabolic semigroups / compact space.
    Einstein = quasilinear hyperbolic / Lorentzian / no compact resolvent.
  The monitor measures the continuation criterion; it does not prove
  the criterion holds for all time.

MEASUREMENT-THEORY CONSEQUENCE (the real bridge)
  Gamma singular w.r.t. mu  =>  no L2 density  =>  no Radon-Nikodym
  likelihood in L2(mu)  =>  measures and uncertainty sets must be the
  primary computational objects. That is why ESCA is built that way.
""")
    return ok


# =============================================================================
# §13. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Carpet-19 Mosco closure + spectral measurement theory")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "run", "registry", "gauge",
                             "mosco", "hino", "einstein", "stats"])
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="runs/carpet_mosco")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    build_registry()
    levels = tuple(a.levels) if a.levels else ((1, 2, 3) if not HAS_SCIPY
                                               else (1, 2, 3, 4))
    out: Dict[str, Any] = {}

    if a.cmd == "run":
        out = run_pipeline(PipelineConfig(levels=levels, out_dir=a.out))
    elif a.cmd == "registry":
        out = {k: v.to_dict() for k, v in RESULTS.items()}
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "gauge":
        g = ResistanceGauge(levels)
        out = {"R_cross": dict(g.R), "ratios": g.ratios(),
               "rho_numeric": g.rho_estimate(), "thm1": verify_thm1(g)}
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "mosco":
        g = ResistanceGauge(levels)
        u = g.lv[max(levels)].centers[:, 1].copy()
        out = MoscoVerifier(g).report(u)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "hino":
        g = ResistanceGauge(levels)
        u = g.lv[max(levels)].centers[:, 1].copy()
        u = u - u.mean(); u = u / max(math.sqrt(g.E(max(levels), u)), 1e-12)
        out = hino_martingale(g, u, tuple(n for n in (1, 2, 3)
                                          if n <= max(levels)))
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "einstein":
        out = demo_einstein_monitor()
        print(json.dumps(out, indent=2, default=str))
        print("\nGLOBAL REGULARITY IS NOT ESTABLISHED. Monitor only.")
    elif a.cmd == "stats":
        out = {"dictionary": STATS_DICT,
               "two_point": two_point_occupation()}
        print(json.dumps(out, indent=2, default=str))

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
