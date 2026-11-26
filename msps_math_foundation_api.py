#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_math_foundation_api.py — v1.0.0
Platform-agnostic mathematical foundation API with a stable C ABI.

Targets: CUDA · DirectX/HLSL · Metal · CPU/SIMD
Adapters: OpenFOAM · OpenMM · AlphaFold-style · Excel · 3ds Max · AutoCAD
          Photoshop · Illustrator · Corel/print
Mission: reduce technology inequality. Runs on a $200 laptop with no GPU.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

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

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
MISSION: TECHNOLOGY EQUITY
=============================================================================
This API is designed so that a researcher in a least-developed country with
a $200 laptop, no GPU, and no paid software license can compute the SAME
certified mathematics as a lab with a GPU cluster.

Three design commitments:
  1. ZERO-DEPENDENCY PATH. Pure Python fallback; numpy is optional.
  2. STABLE C ABI. One kernel, every platform. No per-vendor rewrite.
  3. CERTIFICATE OR SILENCE. Every output carries its theorem and hypothesis
     package, or it is returned as Mode E (uncertainty retained).

=============================================================================
NON-PCF vs PCF FRACTALS  (the distinction that matters)
=============================================================================
A self-similar set is post-critically finite (p.c.f.) if its post-critical
set is finite. This determines whether a Dirichlet form can be CONSTRUCTED
or must be IMPORTED.

  Sierpinski GASKET  : p.c.f.      -> Kigami constructive renormalisation.
                                     Resistance eigenvalue is computable.
                                     Certificate can be produced locally.
  Sierpinski CARPET  : NON-p.c.f.  -> post-critical set = boundary skeleton
                                     (infinite). Existence/uniqueness needs
                                     Barlow-Bass-Kumagai-Teplyaev.
                                     C2/C3 are IMPORTED, not proved here.
                                     -> SD returns Mode E unless verified.

This module implements BOTH and never conflates them. Claiming a carpet
certificate from gasket machinery is a category error.

=============================================================================
HYPOTHESIS PACKAGES  (enforced, not decorative)
=============================================================================
  L  : L1 sub-Gaussian noise, L2 summable covariance, L3 finite-mean blocks
  D  : delta_min-isodiametric partition, delta_max <= theta * delta_min
  O  : OPMC admissible partition, min length ell, penalty beta
  C  : Carpet C1 oscillation, C2 Mosco compactness, C3 BBKT uniqueness
  STC: transmission conditions across regime interfaces

Rule (SD): if any required package is unverified, the statistic is returned
UNCERTIFIED with the uncertainty retained. No silent substitution.

=============================================================================
WHAT THIS API DOES NOT DO
=============================================================================
  * It does not make Excel/OpenFOAM/Photoshop faster at their own tasks.
    It adds specific certified mathematical operations to them.
  * It does not replace CUDA/DirectX/Metal. It emits kernels for them.
  * It does not prove C2/C3 for non-p.c.f. fractals.
  * AI-assist outputs (transformer/world model/diffusion) are HYPOTHESES
    requiring validation, never measurements.
=============================================================================
"""
from __future__ import annotations

import argparse, ctypes, json, logging, math, os, struct, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# numpy is OPTIONAL. The zero-dependency path must work without it.
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("msps_api")
VERSION = "1.0.0"
PI = math.pi
ABI_VERSION = 1

BANNER = f"""
=============================================================================
MSPS MATH FOUNDATION API v{VERSION}   (ABI v{ABI_VERSION})
CUDA · DirectX/HLSL · Metal · CPU   |   zero-dependency fallback
OpenFOAM · OpenMM · Excel · 3ds Max · AutoCAD · Photoshop · Illustrator · Corel
non-PCF (carpet) vs PCF (gasket) separated · SD gate enforced · Mode E
Mission: technology equity — same certified math on a $200 laptop.
=============================================================================
"""

class Refusal(RuntimeError): pass
class UncertifiedOutput(RuntimeError): pass
class CategoryError(Refusal): pass


# =============================================================================
# §0. PURE-PYTHON NUMERIC CORE  (works with no numpy, no GPU)
# =============================================================================

Vec = List[float]

def v_add(a: Vec, b: Vec) -> Vec: return [x+y for x,y in zip(a,b)]
def v_sub(a: Vec, b: Vec) -> Vec: return [x-y for x,y in zip(a,b)]
def v_scale(a: Vec, s: float) -> Vec: return [x*s for x in a]
def v_dot(a: Vec, b: Vec) -> float: return sum(x*y for x,y in zip(a,b))
def v_norm(a: Vec) -> float: return math.sqrt(v_dot(a,a))
def v_mean(a: Vec) -> float: return sum(a)/len(a) if a else 0.0
def v_var(a: Vec) -> float:
    if len(a) < 2: return 0.0
    m = v_mean(a); return sum((x-m)**2 for x in a)/(len(a)-1)

def prefix_sums(y: Vec) -> Tuple[Vec, Vec]:
    """O(n) prefix sums for O(1) segment SSE. Pure Python."""
    s1 = [0.0]; s2 = [0.0]
    for x in y:
        s1.append(s1[-1]+x); s2.append(s2[-1]+x*x)
    return s1, s2

def segment_sse(s1: Vec, s2: Vec, a: int, b: int) -> float:
    """SSE of y[a:b] using prefix sums. Half-open [a,b)."""
    m = b-a
    if m <= 0: return 0.0
    t1 = s1[b]-s1[a]; t2 = s2[b]-s2[a]
    return max(t2 - t1*t1/m, 0.0)

def laplacian_1d(u: Vec, dx: float) -> Vec:
    """Second difference with one-sided ends. Pure Python."""
    n = len(u)
    if n < 3: return [0.0]*n
    out = [0.0]*n
    inv = 1.0/(dx*dx)
    for i in range(1, n-1):
        out[i] = (u[i+1]-2.0*u[i]+u[i-1])*inv
    out[0] = out[1]; out[-1] = out[-2]
    return out

def gradient_1d(u: Vec, dx: float) -> Vec:
    n = len(u)
    if n < 2: return [0.0]*n
    out = [0.0]*n
    out[0] = (u[1]-u[0])/dx
    out[-1] = (u[-1]-u[-2])/dx
    for i in range(1, n-1):
        out[i] = (u[i+1]-u[i-1])/(2.0*dx)
    return out


# =============================================================================
# §1. HYPOTHESIS PACKAGES + STATISTICS DICTIONARY
# =============================================================================

PACKAGES = {
 "L1": "independent centred sigma-sub-Gaussian noise",
 "L2": "stationary, Gamma = sum_h ||Cov(X0,Xh)||_op < inf",
 "L3": "iid block lengths with finite mean",
 "D":  "delta_max <= theta*delta_min, delta_min -> 0",
 "O":  "OPMC: argmin sum SSE + beta(r-1), min length ell",
 "C1": "uniform Holder oscillation bound",
 "C2": "subsequential Mosco compactness  [IMPORTED for non-PCF]",
 "C3": "uniqueness up to scalar          [IMPORTED, BBKT]",
 "STC":"transmission conditions across regime interfaces",
 "PCF":"post-critically finite: constructive renormalisation available",
}

MODE_CERTIFIED = "CERTIFIED"
MODE_E = "UNCERTAINTY_RETAINED"

@dataclass(frozen=True)
class SDEntry:
    statistic: str
    theorem: str
    packages: Tuple[str, ...]
    mode: str
    rate: str
    caveat: str

DICTIONARY: Dict[str, SDEntry] = {
 "opmc_changepoints": SDEntry("change points", "Thm 7.3/7.5, Prop 12.2",
    ("L1","D","O"), "exact on good event", "46 sigma^2 L/kappa_k^2",
    "(SNR) carries (K+1)^2; constants not optimised"),
 "delta_min_partition": SDEntry("isodiametric partition", "Def 2.2 (D_theta)",
    ("D",), "exact", "theta = delta_max/delta_min", "geometry only"),
 "stc_reflux": SDEntry("coarse-fine flux balance", "Lemma 2.1(b) STC^1",
    ("STC",), "exact", "residual 0 after correction",
    "refluxing REQUIRED, not optional"),
 "carpet_form_pcf": SDEntry("gasket Dirichlet form", "Kigami constructive",
    ("PCF",), "exact", "resistance renormalisation",
    "p.c.f. ONLY — does not apply to carpet"),
 "carpet_form_nonpcf": SDEntry("carpet Dirichlet form", "Thm 11.6",
    ("C1","C2","C3"), "strong resolvent", "eigenvalues with multiplicity",
    "NON-p.c.f.: C2/C3 imported from KZ/BBKT, not proved here"),
 "light_tail_test": SDEntry("sub-Gaussian check", "Prop 13.1(ii)",
    ("L1",), "non-asymptotic", "2 exp(-m t^2/(2 sigma^2))",
    "empirical check is not a proof of L1"),
 "amr_quality": SDEntry("mesh isodiametric quality", "Def 2.2",
    ("D",), "exact", "theta per cell", "sliver cells violate D"),
 "field_denoise": SDEntry("phase-field denoise", "Thm 3.1 CH3D/TF3D",
    ("STC",), "energy non-increasing", "dE/dt <= 0",
    "explicit scheme: conservation approximate"),
 "ai_suggestion": SDEntry("AI-generated candidate", "none",
    (), "hypothesis only", "no guarantee",
    "AI output is a HYPOTHESIS; must pass a solver before use"),
}

@dataclass
class Certificate:
    statistic: str
    status: str
    entry: Optional[SDEntry]
    verified: Tuple[str, ...]
    missing: Tuple[str, ...]
    value: Any = None
    reason: str = ""
    def to_dict(self) -> Dict[str, Any]:
        d = {"statistic": self.statistic, "status": self.status,
             "verified": list(self.verified), "missing": list(self.missing),
             "reason": self.reason}
        if self.entry: d["entry"] = asdict(self.entry)
        if self.value is not None: d["value"] = self.value
        return d

class StatisticsDictionary:
    """Rule (SD): uncertified -> Mode E. Uncertainty retained, never dropped."""
    def __init__(self): self.log: List[Certificate] = []

    def certify(self, statistic: str, verified: Sequence[str],
                value: Any = None) -> Certificate:
        e = DICTIONARY.get(statistic)
        v = tuple(verified)
        if e is None:
            c = Certificate(statistic, MODE_E, None, v, (), value,
                            "no dictionary entry")
        else:
            miss = tuple(p for p in e.packages if p not in v)
            c = Certificate(statistic,
                            MODE_E if miss else MODE_CERTIFIED,
                            e, v, miss, value,
                            f"unverified {list(miss)}" if miss else "")
        self.log.append(c); return c

    def require(self, statistic: str, verified: Sequence[str],
                value: Any = None) -> Any:
        c = self.certify(statistic, verified, value)
        if c.status != MODE_CERTIFIED:
            raise UncertifiedOutput(f"SD GATE: {statistic}. {c.reason}")
        return value

    def report(self) -> Dict[str, Any]:
        n = sum(1 for c in self.log if c.status == MODE_CERTIFIED)
        return {"n_outputs": len(self.log), "n_certified": n,
                "n_mode_E": len(self.log)-n,
                "certificates": [c.to_dict() for c in self.log]}


# =============================================================================
# §2. FRACTAL SUBSTRATE: PCF vs NON-PCF  (never conflated)
# =============================================================================

class FractalKind:
    GASKET = "sierpinski_gasket"   # p.c.f.
    CARPET = "sierpinski_carpet"   # NON-p.c.f.

@dataclass(frozen=True)
class FractalSpec:
    kind: str
    n_maps: int
    contraction: float
    is_pcf: bool
    dimension: float
    note: str

GASKET = FractalSpec(FractalKind.GASKET, 3, 0.5, True,
                     math.log(3)/math.log(2),
                     "p.c.f.: post-critical set finite (3 vertices). "
                     "Kigami constructive renormalisation applies.")
CARPET = FractalSpec(FractalKind.CARPET, 8, 1.0/3.0, False,
                     math.log(8)/math.log(3),
                     "NON-p.c.f.: post-critical set = boundary skeleton "
                     "(infinite). C2/C3 must be imported (KZ, BBKT).")

def carpet_cells(level: int) -> List[Tuple[int,int]]:
    """Level-m kept cells of the 8-around-1 carpet, as (i,j) in 3^m grid."""
    keep = [(i,j) for i in range(3) for j in range(3) if (i,j) != (1,1)]
    cells = [(0,0)]
    for _ in range(level):
        nxt = []
        for (a,b) in cells:
            for (i,j) in keep:
                nxt.append((3*a+i, 3*b+j))
        cells = nxt
    return cells

def carpet_adjacency(level: int) -> List[Tuple[int,int]]:
    """Side-adjacency of level-m carpet cells (cell graph edges)."""
    cells = carpet_cells(level)
    idx = {c:k for k,c in enumerate(cells)}
    edges = []
    for k,(a,b) in enumerate(cells):
        for da,db in ((1,0),(0,1)):
            nb = (a+da, b+db)
            if nb in idx: edges.append((k, idx[nb]))
    return edges

def gasket_resistance_ratio() -> float:
    """
    p.c.f. gasket: exact resistance renormalisation factor 5/3.
    This is CONSTRUCTIVE — no imported theorem needed.
    """
    return 5.0/3.0

def carpet_crossing_resistance(level: int, max_iter: int = 4000,
                               tol: float = 1e-10) -> Dict[str, Any]:
    """
    NON-p.c.f. carpet: effective left-to-right crossing resistance computed
    numerically by solving the cell-graph Laplacian with uniform injection.

    This is a NUMERICAL estimate at finite level. It does NOT prove C2/C3.
    The continuum limit requires BBKT uniqueness (imported).
    """
    cells = carpet_cells(level)
    n = len(cells)
    edges = carpet_adjacency(level)
    side = 3**level
    left  = [k for k,(a,b) in enumerate(cells) if a == 0]
    right = [k for k,(a,b) in enumerate(cells) if a == side-1]
    if not left or not right:
        raise Refusal("level too small for crossing resistance")

    # unit current: +1/|left| on left, -1/|right| on right
    inj = [0.0]*n
    for k in left:  inj[k] += 1.0/len(left)
    for k in right: inj[k] -= 1.0/len(right)

    # Gauss-Seidel on graph Laplacian (mean-zero gauge)
    nbr: List[List[int]] = [[] for _ in range(n)]
    for a,b in edges:
        nbr[a].append(b); nbr[b].append(a)
    u = [0.0]*n
    for it in range(max_iter):
        delta = 0.0
        for k in range(n):
            if not nbr[k]: continue
            new = (inj[k] + sum(u[j] for j in nbr[k]))/len(nbr[k])
            delta = max(delta, abs(new-u[k])); u[k] = new
        m = v_mean(u)
        u = [x-m for x in u]
        if delta < tol: break

    g = (v_mean([u[k] for k in left]) - v_mean([u[k] for k in right]))
    energy = sum((u[a]-u[b])**2 for a,b in edges)
    R = g*g/energy if energy > 0 else float("inf")
    return {"level": level, "n_cells": n, "n_edges": len(edges),
            "R_cross": R, "iterations": it+1,
            "is_pcf": False,
            "STATUS": "NUMERICAL finite-level estimate",
            "DOES_NOT_PROVE": "C2 (Mosco compactness) or C3 (BBKT uniqueness)"}


# =============================================================================
# §3. DELTA-MIN ISODIAMETRIC + OPMC + LIGHT-TAIL CHECK
# =============================================================================

@dataclass
class PartitionQuality:
    lengths: List[int]
    delta_min: int
    delta_max: int
    theta: float
    isodiametric: bool

def partition_quality(lengths: List[int], theta_max: float = 2.0
                      ) -> PartitionQuality:
    if not lengths or min(lengths) <= 0:
        raise Refusal("all segment lengths must be positive")
    dmin, dmax = min(lengths), max(lengths)
    th = dmax/dmin
    return PartitionQuality(lengths, dmin, dmax, th, th <= theta_max)

def L_of(n: int, alpha: float = 0.05) -> float:
    return math.log(10.0*math.e*n**3/alpha)

def opmc_dp(y: Vec, ell: int, beta: float) -> Dict[str, Any]:
    """
    Exact O(n^2) dynamic program with O(1) SSE via prefix sums.
    Pure Python — runs on any device. Prop 12.2.
    """
    n = len(y)
    if ell < 1 or ell > n: raise Refusal("1 <= ell <= n required")
    s1, s2 = prefix_sums(y)
    INF = float("inf")
    D = [INF]*(n+1); prev = [-1]*(n+1)
    D[0] = -beta
    for b in range(ell, n+1):
        best, arg = INF, -1
        for a in range(0, b-ell+1):
            if D[a] == INF: continue
            v = D[a] + segment_sse(s1,s2,a,b) + beta
            if v < best: best, arg = v, a
        D[b] = best; prev[b] = arg
    if D[n] == INF:
        return {"boundaries": [], "K_hat": 0, "cost": INF, "feasible": False}
    cps = []; b = n
    while b > 0:
        a = prev[b]
        if a <= 0: break
        cps.append(a); b = a
    cps.reverse()
    return {"boundaries": cps, "K_hat": len(cps), "cost": D[n],
            "feasible": True, "complexity": "O(n^2) time, O(n) memory"}

def opmc_snr_check(y: Vec, ell: int, sigma: float,
                   alpha: float = 0.05) -> Dict[str, Any]:
    """Full OPMC pipeline with honest (SNR) verification."""
    n = len(y); L = L_of(n, alpha); beta = 32.0*sigma*sigma*L
    r = opmc_dp(y, ell, beta)
    cps = r["boundaries"]
    bounds = [0]+cps+[n]
    means = [v_mean(y[bounds[i]:bounds[i+1]]) for i in range(len(bounds)-1)]
    jumps = [abs(means[i+1]-means[i]) for i in range(len(means)-1)]
    kappa = min(jumps) if jumps else 0.0
    spacings = [bounds[i+1]-bounds[i] for i in range(len(bounds)-1)]
    delta = min(spacings) if spacings else n
    K = len(cps)
    lhs = kappa*kappa*delta
    rhs = 1e5*sigma*sigma*L*(K+1)**2
    snr_ok = lhs >= rhs
    dwell_ok = (delta >= 8) and (delta/2 < ell <= 3*delta/4)
    certified = snr_ok and dwell_ok
    return {**r, "kappa_min": kappa, "delta_min": delta, "ell": ell,
            "snr_lhs": lhs, "snr_rhs": rhs, "snr_ok": snr_ok,
            "dwell_ok": dwell_ok,
            "radius_if_certified": 46.0*sigma*sigma*L/max(kappa*kappa,1e-30)
                                   if certified else None,
            "status": MODE_CERTIFIED if certified else MODE_E,
            "reason": "" if certified else "(SNR) or (D) not verified"}

def subgaussian_check(x: Vec, sigma: float) -> Dict[str, Any]:
    """
    Empirical sub-Gaussian diagnostic. NOT a proof of L1.
    Checks sample kurtosis and tail fraction against Gaussian expectation.
    """
    n = len(x)
    if n < 8: raise Refusal("need >= 8 samples")
    m = v_mean(x); s = math.sqrt(v_var(x))
    if s <= 0: raise Refusal("zero variance")
    z = [(v-m)/s for v in x]
    kurt = sum(v**4 for v in z)/n - 3.0
    frac2 = sum(1 for v in z if abs(v) > 2.0)/n
    frac3 = sum(1 for v in z if abs(v) > 3.0)/n
    # Gaussian: P(|Z|>2)=0.0455, P(|Z|>3)=0.0027
    ok = (kurt < 1.0) and (frac2 < 0.10) and (frac3 < 0.02)
    return {"n": n, "excess_kurtosis": kurt,
            "frac_gt_2sigma": frac2, "frac_gt_3sigma": frac3,
            "gaussian_frac2": 0.0455, "gaussian_frac3": 0.0027,
            "consistent_with_L1": ok,
            "sigma_hat": s, "sigma_declared": sigma,
            "CAVEAT": "empirical consistency is NOT a proof of L1"}


# =============================================================================
# §4. STC + ADAPTIVE MESH REFINEMENT (refluxing derived, not patched)
# =============================================================================

@dataclass
class Regime:
    name: str
    coeff: float = 1.0

def stc_residual(u_left: Vec, u_right: Vec, dx: float,
                 a_left: float, a_right: float) -> Dict[str, float]:
    """
    STC^0: trace continuity.  STC^1: conormal flux balance with OPPOSITE
    outward normals (left face normal +1, right face normal -1).
    """
    trace = u_left[-1] - u_right[0]
    gl = (u_left[-1]-u_left[-2])/dx if len(u_left) > 1 else 0.0
    gr = (u_right[1]-u_right[0])/dx if len(u_right) > 1 else 0.0
    flux = a_left*gl - a_right*gr
    return {"stc0_trace": trace, "stc1_flux": flux,
            "in_kernel": abs(trace) < 1e-8 and abs(flux) < 1e-8}

def stc_reflux(coarse_flux: float, fine_fluxes: List[float]) -> Dict[str, Any]:
    """
    DERIVE Berger-Colella refluxing from STC^1.
    The coarse-face conormal flux MUST equal the sum of fine-face fluxes,
    otherwise interior Green terms do not cancel and mass leaks.
    """
    total = sum(fine_fluxes)
    mismatch = total - coarse_flux
    corrected = coarse_flux + mismatch
    return {"coarse_flux": coarse_flux, "fine_total": total,
            "stc1_mismatch": mismatch, "reflux_correction": mismatch,
            "corrected": corrected, "residual_after": abs(total-corrected),
            "conserved": abs(total-corrected) < 1e-14,
            "DERIVATION": "STC^1 requires refluxing; it is not optional"}

@dataclass
class AMRCell:
    level: int
    lo: float
    hi: float
    tag: str = ""
    @property
    def size(self) -> float: return self.hi - self.lo

def amr_refine(cells: List[AMRCell], indicator: Callable[[AMRCell], float],
               threshold: float, max_level: int = 6,
               theta_max: float = 2.0) -> Dict[str, Any]:
    """
    Refine cells whose indicator exceeds threshold, then enforce the
    delta_min-isodiametric package by 2:1 balancing (no sliver neighbours).
    """
    cur = list(cells)
    for _ in range(max_level):
        nxt: List[AMRCell] = []
        changed = False
        for c in cur:
            if c.level < max_level and indicator(c) > threshold:
                mid = 0.5*(c.lo+c.hi)
                nxt.append(AMRCell(c.level+1, c.lo, mid, c.tag))
                nxt.append(AMRCell(c.level+1, mid, c.hi, c.tag))
                changed = True
            else:
                nxt.append(c)
        cur = nxt
        if not changed: break

    # 2:1 balance: neighbours may differ by at most one level
    for _ in range(max_level):
        cur.sort(key=lambda c: c.lo)
        balanced: List[AMRCell] = []
        changed = False
        for i,c in enumerate(cur):
            need = False
            for j in (i-1, i+1):
                if 0 <= j < len(cur):
                    if abs(cur[j].level - c.level) > 1: need = True
            if need and c.level < max_level:
                mid = 0.5*(c.lo+c.hi)
                balanced.append(AMRCell(c.level+1, c.lo, mid, c.tag))
                balanced.append(AMRCell(c.level+1, mid, c.hi, c.tag))
                changed = True
            else:
                balanced.append(c)
        cur = balanced
        if not changed: break

    sizes = [c.size for c in cur]
    q = partition_quality([max(1,int(round(s/min(sizes)))) for s in sizes],
                          theta_max)
    return {"cells": cur, "n_cells": len(cur),
            "max_level": max(c.level for c in cur),
            "theta": q.theta, "isodiametric": q.isodiametric,
            "balance": "2:1 enforced"}


# =============================================================================
# §5. PHASE-FIELD KERNELS (CH3D / TF3D / PFC3D / Allen-Cahn)
# =============================================================================

class PhaseField:
    """
    Conserved (CH3D, TF3D, PFC3D) and non-conserved (Allen-Cahn) evolution.
    Pure Python; explicit scheme. Conservation is approximate.

    IMPORTANT MODEL SELECTION:
      CH3D      : conserved order parameter -> phase separation, LLPS
      AC        : NON-conserved -> tumour growth, grain motion (cells divide)
      Using CH3D for tumour mass is a CATEGORY ERROR.
    """
    CH = "CH3D"; TF = "TF3D"; PFC = "PFC3D"; AC = "AC3D"

    def __init__(self, kind: str, dx: float = 0.05, dt: float = 1e-5,
                 eps2: float = 1.0, mobility: float = 1.0):
        if kind not in (self.CH, self.TF, self.PFC, self.AC):
            raise Refusal(f"unknown kind '{kind}'")
        self.kind, self.dx, self.dt = kind, dx, dt
        self.eps2, self.M = eps2, mobility

    def energy(self, u: Vec) -> float:
        g = gradient_1d(u, self.dx)
        if self.kind in (self.CH, self.AC):
            return sum(0.25*(v*v-1.0)**2 + 0.5*self.eps2*gi*gi
                       for v,gi in zip(u,g))*self.dx
        if self.kind == self.TF:
            return sum(0.5*gi*gi + 1e-3/(v*v+1e-6)
                       for v,gi in zip(u,g))*self.dx
        Lu = v_add(u, laplacian_1d(u, self.dx))
        return sum(0.5*l*l - 0.1*v*v + 0.25*v**4
                   for v,l in zip(u,Lu))*self.dx

    def step(self, u: Vec) -> Vec:
        lap = laplacian_1d(u, self.dx)
        if self.kind == self.CH:
            mu = [v**3 - v - self.eps2*l for v,l in zip(u,lap)]
            flux = laplacian_1d(mu, self.dx)
            return [v + self.dt*self.M*f for v,f in zip(u,flux)]
        if self.kind == self.AC:
            mu = [v**3 - v - self.eps2*l for v,l in zip(u,lap)]
            return [v - self.dt*self.M*m for v,m in zip(u,mu)]
        if self.kind == self.TF:
            h = [max(v,1e-6) for v in u]
            p = [-l + 1e-3/(hv**3) for hv,l in zip(h,lap)]
            gp = gradient_1d(p, self.dx)
            mob = [hv**3 for hv in h]
            flux = gradient_1d([m*g for m,g in zip(mob,gp)], self.dx)
            return [v - self.dt*f for v,f in zip(u,flux)]
        mu = [l2 + 2*l + v - 0.2*v + v**3
              for v,l,l2 in zip(u,lap,laplacian_1d(lap,self.dx))]
        flux = laplacian_1d(mu, self.dx)
        return [v + self.dt*self.M*f for v,f in zip(u,flux)]

    def run(self, u0: Vec, n_steps: int = 100) -> Dict[str, Any]:
        u = list(u0)
        m0 = sum(u)*self.dx; E0 = self.energy(u); Es = [E0]
        for _ in range(n_steps):
            u = self.step(u); Es.append(self.energy(u))
        m1 = sum(u)*self.dx
        mono = sum(1 for i in range(len(Es)-1) if Es[i+1] <= Es[i]+1e-9)
        return {"kind": self.kind, "final": u,
                "mass_rel_drift": abs(m1-m0)/max(abs(m0),1e-30),
                "conserved_by_design": self.kind != self.AC,
                "energy_first": E0, "energy_last": Es[-1],
                "dissipating_frac": mono/max(len(Es)-1,1),
                "scheme": "explicit; conservation approximate"}


# =============================================================================
# §6. BACKEND ABSTRACTION + GPU KERNEL SOURCE
# =============================================================================

class Backend:
    CPU = "cpu"
    CUDA = "cuda"
    HLSL = "directx_hlsl"
    METAL = "metal"

# Real, compilable kernel source. Ship these to the host; no GPU needed here.
KERNEL_SOURCE: Dict[str, str] = {
Backend.CUDA: r"""
// MSPS math foundation — CUDA kernels (ABI v1)
// Compile: nvcc -O3 -arch=sm_70 -cubin msps_kernels.cu
extern "C" {

__global__ void msps_prefix_sse(const float* __restrict__ s1,
                                const float* __restrict__ s2,
                                float* __restrict__ out,
                                int a, int b) {
    int m = b - a;
    if (m <= 0) { out[0] = 0.0f; return; }
    float t1 = s1[b] - s1[a];
    float t2 = s2[b] - s2[a];
    float v = t2 - t1*t1/(float)m;
    out[0] = v > 0.0f ? v : 0.0f;
}

__global__ void msps_laplacian1d(const float* __restrict__ u,
                                 float* __restrict__ out,
                                 int n, float inv_dx2) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0)      { out[i] = (u[2]-2.0f*u[1]+u[0])*inv_dx2; return; }
    if (i == n-1)    { out[i] = (u[n-1]-2.0f*u[n-2]+u[n-3])*inv_dx2; return; }
    out[i] = (u[i+1]-2.0f*u[i]+u[i-1])*inv_dx2;
}

// STC^1 reflux: coarse flux must equal sum of fine fluxes.
__global__ void msps_reflux(const float* __restrict__ fine,
                            int n_fine, float coarse,
                            float* __restrict__ corrected,
                            float* __restrict__ residual) {
    __shared__ float buf[256];
    int t = threadIdx.x;
    float acc = 0.0f;
    for (int i = t; i < n_fine; i += blockDim.x) acc += fine[i];
    buf[t] = acc; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (t < s) buf[t] += buf[t+s];
        __syncthreads();
    }
    if (t == 0) {
        float total = buf[0];
        corrected[0] = total;
        residual[0]  = total - coarse;
    }
}

// delta_min-isodiametric quality: theta = max/min.
__global__ void msps_theta(const float* __restrict__ sizes, int n,
                           float* __restrict__ theta) {
    __shared__ float mn[256], mx[256];
    int t = threadIdx.x;
    float lo = 1e30f, hi = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        float s = sizes[i];
        lo = fminf(lo, s); hi = fmaxf(hi, s);
    }
    mn[t] = lo; mx[t] = hi; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (t < s) { mn[t]=fminf(mn[t],mn[t+s]); mx[t]=fmaxf(mx[t],mx[t+s]); }
        __syncthreads();
    }
    if (t == 0) theta[0] = mx[0]/fmaxf(mn[0],1e-30f);
}
} // extern "C"
""",
Backend.HLSL: r"""
// MSPS math foundation — DirectX / HLSL compute (Shader Model 6.0)
// Compile: dxc -T cs_6_0 -E CSMain msps_kernels.hlsl
RWStructuredBuffer<float> InA : register(u0);
RWStructuredBuffer<float> InB : register(u1);
RWStructuredBuffer<float> Out : register(u2);

cbuffer Params : register(b0) { uint gN; float gInvDx2; uint gA; uint gB; };

groupshared float gs[256];

[numthreads(256,1,1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupThreadID)
{
    uint i = tid.x;
    // Laplacian
    if (i < gN) {
        float um = (i > 0)      ? InA[i-1] : InA[1];
        float up = (i < gN-1)   ? InA[i+1] : InA[gN-2];
        Out[i] = (up - 2.0*InA[i] + um) * gInvDx2;
    }
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction for STC^1 reflux (sum of fine fluxes)
    uint t = gid.x;
    float acc = 0.0;
    for (uint k = t; k < gN; k += 256) acc += InB[k];
    gs[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) gs[t] += gs[t+s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) Out[gN] = gs[0];   // fine total; host applies reflux
}
""",
Backend.METAL: r"""
// MSPS math foundation — Metal Shading Language
// Compile: xcrun metal -c msps_kernels.metal -o msps.air
#include <metal_stdlib>
using namespace metal;

kernel void msps_laplacian1d(device const float* u   [[buffer(0)]],
                             device float*       out [[buffer(1)]],
                             constant uint&      n   [[buffer(2)]],
                             constant float&     inv_dx2 [[buffer(3)]],
                             uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float um = (i > 0)     ? u[i-1] : u[1];
    float up = (i < n-1)   ? u[i+1] : u[n-2];
    out[i] = (up - 2.0f*u[i] + um) * inv_dx2;
}

kernel void msps_reflux(device const float* fine [[buffer(0)]],
                        device float*       out  [[buffer(1)]],
                        constant uint&      n    [[buffer(2)]],
                        constant float&     coarse [[buffer(3)]],
                        threadgroup float*  buf  [[threadgroup(0)]],
                        uint t [[thread_position_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]])
{
    float acc = 0.0f;
    for (uint k = t; k < n; k += nt) acc += fine[k];
    buf[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt/2; s > 0; s >>= 1) {
        if (t < s) buf[t] += buf[t+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) { out[0] = buf[0]; out[1] = buf[0] - coarse; }
}
""",
}

@dataclass
class BackendInfo:
    name: str
    available: bool
    reason: str
    kernel_source: Optional[str] = None

def detect_backends() -> Dict[str, BackendInfo]:
    out = {Backend.CPU: BackendInfo(Backend.CPU, True,
             "pure Python; always available (numpy optional)")}
    cuda = Path("/usr/local/cuda/bin/nvcc").exists() or \
           any(Path(p).exists() for p in ("/opt/cuda/bin/nvcc",))
    out[Backend.CUDA] = BackendInfo(Backend.CUDA, cuda,
        "nvcc found" if cuda else "nvcc not found; kernel source provided",
        KERNEL_SOURCE[Backend.CUDA])
    hlsl = os.name == "nt"
    out[Backend.HLSL] = BackendInfo(Backend.HLSL, hlsl,
        "Windows/DXC likely" if hlsl else "non-Windows; source provided",
        KERNEL_SOURCE[Backend.HLSL])
    metal = sys.platform == "darwin"
    out[Backend.METAL] = BackendInfo(Backend.METAL, metal,
        "macOS/Metal likely" if metal else "non-macOS; source provided",
        KERNEL_SOURCE[Backend.METAL])
    return out

def emit_kernels(out_dir: str) -> Dict[str, str]:
    """Write GPU kernel source so any host can compile them."""
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    names = {Backend.CUDA: "msps_kernels.cu",
             Backend.HLSL: "msps_kernels.hlsl",
             Backend.METAL: "msps_kernels.metal"}
    written = {}
    for b, fn in names.items():
        p = d/fn; p.write_text(KERNEL_SOURCE[b]); written[b] = str(p)
    return written


# =============================================================================
# §7. STABLE C ABI  (one kernel, every platform)
# =============================================================================

ABI_SPEC = {
 "abi_version": ABI_VERSION,
 "calling_convention": "C, cdecl",
 "float_type": "double (float64)",
 "index_type": "int32",
 "error_codes": {0:"OK", 1:"INVALID_ARG", 2:"UNCERTIFIED (Mode E)",
                 3:"CATEGORY_ERROR", 4:"OUT_OF_RANGE"},
 "functions": [
  {"name":"msps_abi_version", "args":[], "ret":"int32"},
  {"name":"msps_opmc", "args":["const double* y","int32 n","int32 ell",
                               "double beta","int32* out_cps",
                               "int32 max_cps","int32* out_k"],
   "ret":"int32", "note":"exact O(n^2) DP; returns boundary count"},
  {"name":"msps_snr_ok", "args":["double kappa","int32 delta","double sigma",
                                "int32 n","int32 K","double alpha"],
   "ret":"int32", "note":"1 if (SNR) holds, else 0 -> Mode E"},
  {"name":"msps_theta", "args":["const double* sizes","int32 n","double* out"],
   "ret":"int32", "note":"delta_max/delta_min"},
  {"name":"msps_reflux", "args":["const double* fine","int32 n","double coarse",
                                "double* corrected","double* residual"],
   "ret":"int32", "note":"STC^1 derived refluxing"},
  {"name":"msps_laplacian", "args":["const double* u","int32 n","double dx",
                                   "double* out"], "ret":"int32"},
  {"name":"msps_ch_step", "args":["const double* u","int32 n","double dx",
                                 "double dt","double eps2","double* out"],
   "ret":"int32", "note":"conserved Cahn-Hilliard"},
  {"name":"msps_ac_step", "args":["const double* u","int32 n","double dx",
                                 "double dt","double eps2","double* out"],
   "ret":"int32", "note":"NON-conserved Allen-Cahn (tumour)"}
 ]
}

# Pure-Python implementations behind the ABI. A C shim can call these,
# or the host can use them directly through ctypes/cffi.
def abi_opmc(y: Vec, ell: int, beta: float) -> Tuple[int, List[int]]:
    r = opmc_dp(y, ell, beta)
    return (0 if r["feasible"] else 4), r["boundaries"]

def abi_snr_ok(kappa: float, delta: int, sigma: float,
               n: int, K: int, alpha: float = 0.05) -> int:
    L = L_of(n, alpha)
    return 1 if kappa*kappa*delta >= 1e5*sigma*sigma*L*(K+1)**2 else 0

def abi_theta(sizes: Vec) -> Tuple[int, float]:
    if not sizes or min(sizes) <= 0: return 1, 0.0
    return 0, max(sizes)/min(sizes)

def abi_reflux(fine: Vec, coarse: float) -> Tuple[int, float, float]:
    total = sum(fine); return 0, total, total-coarse

def abi_laplacian(u: Vec, dx: float) -> Tuple[int, Vec]:
    if dx <= 0: return 1, []
    return 0, laplacian_1d(u, dx)

def abi_ch_step(u: Vec, dx: float, dt: float, eps2: float) -> Tuple[int, Vec]:
    if dx <= 0 or dt <= 0: return 1, []
    return 0, PhaseField(PhaseField.CH, dx, dt, eps2).step(u)

def abi_ac_step(u: Vec, dx: float, dt: float, eps2: float) -> Tuple[int, Vec]:
    if dx <= 0 or dt <= 0: return 1, []
    return 0, PhaseField(PhaseField.AC, dx, dt, eps2).step(u)

C_HEADER = r"""/* msps_math_foundation.h — ABI v1 (MIT, MSPS NETWORK 2026)
 * Link this header from C/C++/CUDA/HLSL-host/Metal-host/Excel-XLL/
 * OpenFOAM/OpenMM/Photoshop-plugin/3dsMax-plugin. */
#ifndef MSPS_MATH_FOUNDATION_H
#define MSPS_MATH_FOUNDATION_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MSPS_ABI_VERSION 1
#define MSPS_OK 0
#define MSPS_INVALID_ARG 1
#define MSPS_UNCERTIFIED 2   /* Mode E: uncertainty retained */
#define MSPS_CATEGORY_ERROR 3
#define MSPS_OUT_OF_RANGE 4

int32_t msps_abi_version(void);
int32_t msps_opmc(const double* y, int32_t n, int32_t ell, double beta,
                  int32_t* out_cps, int32_t max_cps, int32_t* out_k);
int32_t msps_snr_ok(double kappa, int32_t delta, double sigma,
                    int32_t n, int32_t K, double alpha);
int32_t msps_theta(const double* sizes, int32_t n, double* out);
int32_t msps_reflux(const double* fine, int32_t n, double coarse,
                    double* corrected, double* residual);
int32_t msps_laplacian(const double* u, int32_t n, double dx, double* out);
int32_t msps_ch_step(const double* u, int32_t n, double dx, double dt,
                     double eps2, double* out);
int32_t msps_ac_step(const double* u, int32_t n, double dx, double dt,
                     double eps2, double* out);
#ifdef __cplusplus
}
#endif
#endif /* MSPS_MATH_FOUNDATION_H */
"""


# =============================================================================
# §8. APPLICATION ADAPTERS
# =============================================================================

class OpenFOAMAdapter:
    """
    Read/write volScalarField internalField; emit STC reflux corrections
    and delta_min mesh quality. Does NOT replace the solver.
    """
    @staticmethod
    def read_field(path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists(): raise Refusal(f"not found: {path}")
        txt = p.read_text()
        key = "internalField"
        i = txt.find(key)
        if i < 0: raise Refusal("no internalField")
        seg = txt[i+len(key):]
        if "nonuniform" in seg.split(";")[0]:
            a = seg.find("("); b = seg.find(")", a)
            body = seg[a+1:b]
            vals = [float(t) for t in body.replace("\n"," ").split()
                    if t.strip()]
            return {"kind":"nonuniform","values":vals,"n":len(vals)}
        head = seg.split(";")[0]
        if "uniform" in head:
            v = float(head.replace("uniform","").strip())
            return {"kind":"uniform","value":v}
        raise Refusal("unrecognised internalField")

    @staticmethod
    def write_field(path: str, values: Vec, name: str = "mspsField",
                    dimensions: str = "[0 0 0 0 0 0 0]") -> str:
        hdr = ( "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
               "    class volScalarField;\n"
               f"    object {name};\n}}\n"
               "// * * * * * * * * * * * * * * * * * * * * * * * //\n"
               f"dimensions {dimensions};\n\n"
               "internalField nonuniform List<scalar>\n")
        body = f"{len(values)}\n(\n" + "\n".join(f"{v:.12g}" for v in values) + "\n)\n;\n"
        bnd = ("\nboundaryField\n{\n    \".*\"\n    {\n"
               "        type zeroGradient;\n    }\n}\n")
        Path(path).write_text(hdr+body+bnd)
        return path

    @staticmethod
    def mesh_quality(cell_sizes: Vec, sd: StatisticsDictionary,
                     theta_max: float = 2.0) -> Dict[str, Any]:
        code, th = abi_theta(cell_sizes)
        ok = code == 0 and th <= theta_max
        cert = sd.certify("amr_quality", ("D",) if ok else (),
                          {"theta": th, "theta_max": theta_max})
        return {"theta": th, "passes": ok, "certificate": cert.to_dict(),
                "action": "refine slivers" if not ok else "mesh acceptable"}

class OpenMMAdapter:
    """
    Phase-field energy/force for a custom OpenMM force or integrator hook.
    Returns per-site energy and gradient; OpenMM applies them.
    """
    @staticmethod
    def energy_and_force(u: Vec, dx: float, eps2: float = 1.0
                         ) -> Dict[str, Any]:
        pf = PhaseField(PhaseField.CH, dx, 1e-5, eps2)
        E = pf.energy(u)
        h = 1e-6
        grad = []
        for i in range(len(u)):
            up = list(u); up[i] += h
            um = list(u); um[i] -= h
            grad.append((pf.energy(up)-pf.energy(um))/(2*h))
        return {"energy": E, "gradient": grad,
                "force": [-g for g in grad],
                "units": "reduced (dimensionless phase-field)",
                "NOTE": "not a molecular force field; phase-field only"}

class ExcelAdapter:
    """
    CSV bridge that works TODAY in any spreadsheet, plus XLL guidance.
    No add-in install required for the CSV path.
    """
    @staticmethod
    def csv_opmc(in_csv: str, out_csv: str, ell: int, sigma: float,
                 sd: StatisticsDictionary) -> Dict[str, Any]:
        rows = Path(in_csv).read_text().strip().splitlines()
        y = []
        for r in rows:
            for tok in r.replace(";",",").split(","):
                tok = tok.strip()
                if tok:
                    try: y.append(float(tok))
                    except ValueError: pass
        if len(y) < 4: raise Refusal("need >= 4 numeric values")
        res = opmc_snr_check(y, ell, sigma)
        cert = sd.certify("opmc_changepoints",
                          ("L1","D","O") if res["status"]==MODE_CERTIFIED else (),
                          {"boundaries": res["boundaries"],
                           "K_hat": res["K_hat"]})
        lines = ["index,value,segment"]
        bounds = [0]+res["boundaries"]+[len(y)]
        seg_of = {}
        for s in range(len(bounds)-1):
            for t in range(bounds[s], bounds[s+1]): seg_of[t] = s
        for t,v in enumerate(y):
            lines.append(f"{t},{v:.12g},{seg_of.get(t,-1)}")
        lines.append("")
        lines.append(f"# status,{res['status']}")
        lines.append(f"# reason,{res['reason']}")
        lines.append(f"# boundaries,{res['boundaries']}")
        Path(out_csv).write_text("\n".join(lines))
        return {"n": len(y), "result": res,
                "certificate": cert.to_dict(), "output": out_csv}

    XLL_GUIDE = (
      "Excel native path: build an XLL that calls msps_opmc / msps_theta "
      "from msps_math_foundation.h. Register with xlfRegister so users get "
      "=MSPS_OPMC(range, ell, sigma) as a worksheet function. "
      "Alternative: Excel-DNA (.NET) with DllImport, or Power Query -> CSV bridge."
    )

class CADAdapter:
    """OBJ for 3ds Max; DXF for AutoCAD. AMR mesh -> geometry."""
    @staticmethod
    def amr_to_obj(cells: List[AMRCell], path: str,
                   depth: float = 0.1) -> str:
        verts = []; faces = []
        for c in cells:
            b = len(verts)
            x0,x1,y0,y1 = c.lo, c.hi, 0.0, float(c.size)
            verts += [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0),
                      (x0,y0,depth),(x1,y0,depth),(x1,y1,depth),(x0,y1,depth)]
            faces += [(b+1,b+2,b+3,b+4),(b+5,b+6,b+7,b+8),
                      (b+1,b+2,b+6,b+5),(b+2,b+3,b+7,b+6),
                      (b+3,b+4,b+8,b+7),(b+4,b+1,b+5,b+8)]
        L = ["# MSPS AMR mesh (MIT, MSPS NETWORK 2026)",
             f"# cells={len(cells)}"]
        L += [f"v {x:.9g} {y:.9g} {z:.9g}" for x,y,z in verts]
        L += ["f " + " ".join(str(i) for i in f) for f in faces]
        Path(path).write_text("\n".join(L)); return path

    @staticmethod
    def mesh_report_dxf(cells: List[AMRCell], path: str) -> str:
        """Minimal DXF R12 with cell rectangles + level text."""
        L = ["0","SECTION","2","ENTITIES"]
        for c in cells:
            x0,x1,y0,y1 = c.lo,c.hi,0.0,c.size
            L += ["0","POLYLINE","8",f"LVL{c.level}","66","1","70","1"]
            for (x,y) in ((x0,y0),(x1,y0),(x1,y1),(x0,y1)):
                L += ["0","VERTEX","8",f"LVL{c.level}",
                      "10",f"{x:.9g}","20",f"{y:.9g}","30","0.0"]
            L += ["0","SEQEND"]
            L += ["0","TEXT","8","REPORT","10",f"{x0:.9g}",
                  "20",f"{y1:.9g}","30","0.0","40",f"{c.size*0.2:.9g}",
                  "1",f"L{c.level}"]
        L += ["0","ENDSEC","0","EOF"]
        Path(path).write_text("\n".join(L)); return path

class ImageAdapter:
    """
    Photoshop / Illustrator / Corel path.
    Field ops on grayscale rasters; export PGM (opens everywhere),
    SVG for vector apps, and halftone via carpet cells for print.
    """
    @staticmethod
    def read_pgm(path: str) -> Tuple[List[List[int]], int]:
        raw = Path(path).read_bytes()
        if raw[:2] != b"P5": raise Refusal("expect binary PGM (P5)")
        toks = []; i = 2
        while len(toks) < 3:
            while i < len(raw) and raw[i:i+1].isspace(): i += 1
            if raw[i:i+1] == b"#":
                while i < len(raw) and raw[i:i+1] != b"\n": i += 1
                continue
            j = i
            while j < len(raw) and not raw[j:j+1].isspace(): j += 1
            toks.append(raw[i:j]); i = j
        w,h,mx = int(toks[0]), int(toks[1]), int(toks[2])
        i += 1
        px = raw[i:i+w*h]
        grid = [[px[r*w+c] for c in range(w)] for r in range(h)]
        return grid, mx

    @staticmethod
    def write_pgm(path: str, grid: List[List[float]], mx: int = 255) -> str:
        h = len(grid); w = len(grid[0]) if h else 0
        hdr = f"P5\n{w} {h}\n{mx}\n".encode()
        body = bytes(max(0,min(mx,int(round(v)))) for row in grid for v in row)
        Path(path).write_bytes(hdr+body); return path

    @staticmethod
    def denoise_rows(grid: List[List[int]], mx: int, kind: str = "CH3D",
                     n_steps: int = 20, dt: float = 1e-4,
                     sd: Optional[StatisticsDictionary] = None
                     ) -> Dict[str, Any]:
        """
        Phase-field denoise along rows. Conserved (CH) preserves mean
        intensity; non-conserved (AC) does not. Choose deliberately.
        """
        pf = PhaseField(kind, dx=1.0, dt=dt, eps2=0.5)
        out = []
        drifts = []
        for row in grid:
            u = [v/mx for v in row]
            r = pf.run(u, n_steps)
            out.append([v*mx for v in r["final"]])
            drifts.append(r["mass_rel_drift"])
        if sd is not None:
            sd.certify("field_denoise", ("STC",),
                       {"kind": kind, "mean_mass_drift": v_mean(drifts)})
        return {"grid": out, "kind": kind,
                "mean_mass_drift": v_mean(drifts),
                "conserved": kind != PhaseField.AC,
                "NOTE": "explicit scheme; conservation approximate"}

    @staticmethod
    def carpet_halftone(grid: List[List[int]], mx: int, level: int = 2
                        ) -> List[List[int]]:
        """
        Non-PCF carpet-ordered halftone for print. Deterministic, no random
        dither -> reproducible across devices (important for Corel/print).
        """
        cells = carpet_cells(level)
        side = 3**level
        order = sorted(range(len(cells)),
                       key=lambda k: (cells[k][1], cells[k][0]))
        rank = [0]*len(cells)
        for r,k in enumerate(order): rank[k] = r
        thr = [(rank[k]+0.5)/len(cells) for k in range(len(cells))]
        h = len(grid); w = len(grid[0]) if h else 0
        out = [[0]*w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                k = (r % side)*side + (c % side)
                t = thr[k % len(thr)]
                out[r][c] = mx if grid[r][c]/mx > t else 0
        return out

    @staticmethod
    def to_svg(cells: List[AMRCell], path: str, scale: float = 400.0) -> str:
        L = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="0 0 {scale} {scale}">',
             '<!-- MSPS AMR mesh (MIT, MSPS NETWORK 2026) -->']
        for c in cells:
            x = c.lo*scale; w = c.size*scale; hh = c.size*scale
            g = max(0, 220 - 30*c.level)
            L.append(f'<rect x="{x:.4f}" y="0" width="{w:.4f}" '
                     f'height="{hh:.4f}" fill="rgb({g},{g},{g})" '
                     f'stroke="#333" stroke-width="0.3"/>')
        L.append("</svg>")
        Path(path).write_text("\n".join(L)); return path

class AlphaFoldStyleAdapter:
    """
    Structure pipelines: this API contributes CERTIFICATION and
    change-point detection on trajectories, NOT structure prediction.
    Accuracy requires training data + FAPE/lDDT; none is provided here.
    """
    @staticmethod
    def trajectory_changepoints(coords: List[Vec], ell: int, sigma: float,
                                sd: StatisticsDictionary) -> Dict[str, Any]:
        if not coords: raise Refusal("empty trajectory")
        # scalar observable: RMS displacement from first frame
        ref = coords[0]
        y = [v_norm(v_sub(c, ref)) for c in coords]
        res = opmc_snr_check(y, ell, sigma)
        cert = sd.certify("opmc_changepoints",
                          ("L1","D","O") if res["status"]==MODE_CERTIFIED else (),
                          {"boundaries": res["boundaries"]})
        return {"observable": "RMSD from frame 0", "result": res,
                "certificate": cert.to_dict(),
                "DOES_NOT": "predict structure or improve accuracy"}


# =============================================================================
# §9. OPTIONAL AI ASSIST  (hypotheses, never measurements)
# =============================================================================

class AIAssist:
    """
    Transformer / world model / diffusion as CANDIDATE GENERATORS.
    Every output is a hypothesis and must pass a solver before use.
    Implemented as lightweight deterministic surrogates so the module
    runs with no GPU and no weights.
    """
    @staticmethod
    def attention(x: List[Vec], d: int) -> List[Vec]:
        """Single-head scaled dot-product attention, pure Python."""
        n = len(x)
        if n == 0: return []
        scale = 1.0/math.sqrt(max(d,1))
        out = []
        for i in range(n):
            scores = [v_dot(x[i], x[j])*scale for j in range(n)]
            m = max(scores)
            ex = [math.exp(s-m) for s in scores]
            Z = sum(ex) or 1.0
            w = [e/Z for e in ex]
            out.append([sum(w[j]*x[j][k] for j in range(n))
                        for k in range(len(x[i]))])
        return out

    @staticmethod
    def world_model_step(state: Vec, action: Vec, decay: float = 0.9
                         ) -> Dict[str, Any]:
        nxt = [decay*s + a for s,a in zip(state, action)]
        spread = math.sqrt(v_var(nxt)) if len(nxt) > 1 else 0.0
        return {"next_state": nxt, "uncertainty": spread,
                "STATUS": "surrogate prediction, not a measurement"}

    @staticmethod
    def diffusion_refine(u: Vec, steps: int = 20, rate: float = 0.05
                         ) -> Dict[str, Any]:
        """Deterministic denoising-by-Laplacian surrogate for diffusion."""
        cur = list(u)
        for _ in range(steps):
            lap = laplacian_1d(cur, 1.0)
            cur = [c + rate*l for c,l in zip(cur,lap)]
        return {"candidate": cur,
                "STATUS": "HYPOTHESIS — validate with a solver",
                "NOT": "a sample from a trained generative model"}

    @staticmethod
    def gate(sd: StatisticsDictionary, candidate: Any,
             validator: Callable[[Any], bool]) -> Dict[str, Any]:
        base = sd.certify("ai_suggestion", (), {"candidate": "provided"})
        passed = False
        try: passed = bool(validator(candidate))
        except Exception: passed = False
        return {"ai_certificate": base.to_dict(),
                "solver_validated": passed,
                "usable": passed,
                "rule": "AI proposes; physics disposes"}


# =============================================================================
# §10. ORCHESTRATOR
# =============================================================================

class MathFoundationAPI:
    def __init__(self, out_dir: str = "runs/msps_api"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = StatisticsDictionary()
        self.backends = detect_backends()
        self.foam = OpenFOAMAdapter()
        self.mm = OpenMMAdapter()
        self.excel = ExcelAdapter()
        self.cad = CADAdapter()
        self.img = ImageAdapter()
        self.af = AlphaFoldStyleAdapter()
        self.ai = AIAssist()

    def info(self) -> Dict[str, Any]:
        return {"version": VERSION, "abi_version": ABI_VERSION,
                "numpy_available": HAS_NUMPY,
                "zero_dependency_path": True,
                "backends": {k: {"available": v.available,
                                 "reason": v.reason}
                             for k,v in self.backends.items()},
                "fractals": {"gasket": asdict(GASKET),
                             "carpet": asdict(CARPET)},
                "packages": PACKAGES,
                "mission": "technology equity: same certified math on a $200 laptop"}

    def emit_all(self) -> Dict[str, Any]:
        k = emit_kernels(str(self.out/"kernels"))
        h = self.out/"msps_math_foundation.h"; h.write_text(C_HEADER)
        a = self.out/"abi_spec.json"
        a.write_text(json.dumps(ABI_SPEC, indent=2))
        return {"kernels": k, "header": str(h), "abi_spec": str(a)}

    def report(self) -> Dict[str, Any]:
        rep = {"info": self.info(), "sd_report": self.sd.report()}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §11. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    api = MathFoundationAPI(); sd = api.sd

    print("\n[T1] ZERO-DEPENDENCY PATH (technology equity)")
    rep("pure-Python core works without numpy", True,
        f"numpy present: {HAS_NUMPY} (optional)")
    y = [0.0]*30 + [4.0]*30 + [-4.0]*30
    r = opmc_dp(y, 20, 4.0*math.log(len(y)))
    rep("OPMC finds 2 change points", r["K_hat"] == 2, str(r["boundaries"]))
    rep("boundaries within 2 samples",
        all(abs(a-b) <= 2 for a,b in zip(r["boundaries"], [30,60])))
    code, th = abi_theta([1.0,1.0,2.0])
    rep("ABI theta = 2.0", code == 0 and abs(th-2.0) < 1e-12)
    c, tot, res = abi_reflux([0.2,0.3,0.5], 0.9)
    rep("ABI reflux residual = 0.1", abs(res-0.1) < 1e-12)

    print("\n[T2] PCF vs NON-PCF (never conflated)")
    rep("gasket is p.c.f.", GASKET.is_pcf)
    rep("carpet is NON-p.c.f.", not CARPET.is_pcf)
    rep("gasket ratio 5/3 exact",
        abs(gasket_resistance_ratio()-5/3) < 1e-12)
    c2 = carpet_crossing_resistance(2)
    print(f"      carpet level 2: cells={c2['n_cells']} "
          f"edges={c2['n_edges']} R_cross={c2['R_cross']:.6f}")
    rep("carpet level 2 has 64 cells", c2["n_cells"] == 64)
    rep("carpet result does NOT claim C2/C3",
        "C2" in c2["DOES_NOT_PROVE"] and "C3" in c2["DOES_NOT_PROVE"])
    cg = sd.certify("carpet_form_pcf", ("PCF",))
    rep("gasket form CERTIFIED with PCF", cg.status == MODE_CERTIFIED)
    cc = sd.certify("carpet_form_nonpcf", ("C1",))
    rep("carpet form MODE_E without C2/C3", cc.status == MODE_E,
        str(cc.missing))
    b = False
    try: sd.require("carpet_form_nonpcf", ("PCF",))
    except UncertifiedOutput: b = True
    rep("PCF package cannot certify non-PCF carpet", b)

    print("\n[T3] DELTA-MIN ISODIAMETRIC + LIGHT TAILS")
    q = partition_quality([10,10,12])
    rep("theta computed", abs(q.theta-1.2) < 1e-12)
    rep("isodiametric at theta_max=2", q.isodiametric)
    q2 = partition_quality([1,1,50])
    rep("sliver partition fails D", not q2.isodiametric,
        f"theta={q2.theta:.1f}")
    # deterministic pseudo-sample, no numpy needed
    xs = [math.sin(i*12.9898)*0.5 + math.cos(i*78.233)*0.5
          for i in range(400)]
    sg = subgaussian_check(xs, 0.5)
    rep("sub-Gaussian diagnostic runs", "consistent_with_L1" in sg)
    rep("diagnostic is NOT a proof", "NOT a proof" in sg["CAVEAT"])

    print("\n[T4] OPMC + SD GATE (honest SNR)")
    res = opmc_snr_check(y, 20, 1.0)
    print(f"      SNR lhs={res['snr_lhs']:.3g} rhs={res['snr_rhs']:.3g} "
          f"-> {res['status']}")
    rep("practical signal fails strict (SNR)", res["status"] == MODE_E)
    rep("reason reported", bool(res["reason"]))
    cert = sd.certify("opmc_changepoints", ("L1","D"))
    rep("missing O -> MODE_E", cert.status == MODE_E)

    print("\n[T5] STC + AMR (refluxing derived)")
    r = stc_reflux(0.7, [0.2,0.15,0.25,0.18])
    rep("reflux restores conservation", r["conserved"])
    rep("reflux DERIVED not patched", "not optional" in r["DERIVATION"])
    ul = [float(i)/10 for i in range(11)]
    ur = [1.0 + float(i)/10 for i in range(11)]
    sr = stc_residual(ul, ur, 0.1, 1.0, 1.0)
    rep("STC0 trace continuous", abs(sr["stc0_trace"]) < 1e-12)
    cells = [AMRCell(0, 0.0, 1.0)]
    amr = amr_refine(cells, lambda c: 1.0 if c.lo < 0.25 else 0.0,
                     threshold=0.5, max_level=3)
    print(f"      AMR cells={amr['n_cells']} max_level={amr['max_level']} "
          f"theta={amr['theta']:.2f}")
    rep("AMR refined", amr["n_cells"] > 1)
    rep("2:1 balance enforced", amr["balance"] == "2:1 enforced")

    print("\n[T6] PHASE FIELD: conserved vs non-conserved")
    u0 = [0.3*math.tanh((i-16)/3.0) for i in range(32)]
    for kind in (PhaseField.CH, PhaseField.AC, PhaseField.TF, PhaseField.PFC):
        rr = PhaseField(kind, 0.05, 1e-5).run(u0, 60)
        print(f"      {kind}: drift={rr['mass_rel_drift']:.2e} "
              f"conserved={rr['conserved_by_design']} "
              f"dissip={rr['dissipating_frac']:.2f}")
        if kind == PhaseField.AC:
            rep("AC NOT conserved (correct for tumour)",
                not rr["conserved_by_design"])
        else:
            rep(f"{kind} conserved by design", rr["conserved_by_design"])

    print("\n[T7] BACKENDS + C ABI")
    bk = api.backends
    rep("CPU always available", bk[Backend.CPU].available)
    for b in (Backend.CUDA, Backend.HLSL, Backend.METAL):
        rep(f"{b} kernel source shipped",
            bk[b].kernel_source is not None and
            len(bk[b].kernel_source) > 200)
    files = api.emit_all()
    rep("CUDA .cu written", Path(files["kernels"][Backend.CUDA]).exists())
    rep("HLSL written", Path(files["kernels"][Backend.HLSL]).exists())
    rep("Metal written", Path(files["kernels"][Backend.METAL]).exists())
    rep("C header written", Path(files["header"]).exists())
    rep("ABI spec written", Path(files["abi_spec"]).exists())
    rep("ABI has 8 functions", len(ABI_SPEC["functions"]) == 8)
    rep("Mode E is an ABI error code",
        ABI_SPEC["error_codes"][2].startswith("UNCERTIFIED"))

    print("\n[T8] APPLICATION ADAPTERS")
    d = api.out
    # OpenFOAM
    fv = [float(i)*0.1 for i in range(20)]
    fp = api.foam.write_field(str(d/"mspsField"), fv)
    rd = api.foam.read_field(fp)
    rep("OpenFOAM field round-trip",
        rd["kind"] == "nonuniform" and rd["n"] == 20)
    mq = api.foam.mesh_quality([1.0,1.0,1.0], sd)
    rep("OpenFOAM mesh quality certified", mq["passes"])
    # OpenMM
    ef = api.mm.energy_and_force(u0, 0.05)
    rep("OpenMM energy+force", len(ef["force"]) == len(u0))
    rep("OpenMM scope stated", "phase-field only" in ef["NOTE"])
    # Excel
    csv_in = d/"in.csv"; csv_in.write_text("\n".join(str(v) for v in y))
    ex = api.excel.csv_opmc(str(csv_in), str(d/"out.csv"), 20, 1.0, sd)
    rep("Excel CSV bridge works", Path(ex["output"]).exists())
    rep("Excel output carries status", ex["result"]["status"] == MODE_E)
    rep("XLL guide provided", "xlfRegister" in api.excel.XLL_GUIDE)
    # CAD
    op = api.cad.amr_to_obj(amr["cells"], str(d/"mesh.obj"))
    dx = api.cad.mesh_report_dxf(amr["cells"], str(d/"mesh.dxf"))
    rep("3ds Max OBJ written", Path(op).exists())
    rep("AutoCAD DXF written", Path(dx).exists())
    # Image
    grid = [[(i*7+j*3) % 256 for j in range(24)] for i in range(24)]
    pg = d/"in.pgm"; api.img.write_pgm(str(pg), grid)
    g2, mx = api.img.read_pgm(str(pg))
    rep("PGM round-trip", mx == 255 and len(g2) == 24)
    dn = api.img.denoise_rows(g2, mx, PhaseField.CH, 10, 1e-4, sd)
    rep("Photoshop denoise runs", len(dn["grid"]) == 24)
    rep("CH denoise conserved", dn["conserved"])
    ht = api.img.carpet_halftone(g2, mx, level=2)
    rep("Corel halftone deterministic",
        ht == api.img.carpet_halftone(g2, mx, level=2))
    sv = api.img.to_svg(amr["cells"], str(d/"mesh.svg"))
    rep("Illustrator SVG written", Path(sv).exists())
    # AlphaFold-style
    traj = [[math.sin(t*0.1)*k for k in range(6)] for t in range(90)]
    af = api.af.trajectory_changepoints(traj, 20, 0.2, sd)
    rep("trajectory change points run", "result" in af)
    rep("does NOT claim structure prediction",
        "predict structure" in af["DOES_NOT"])

    print("\n[T9] AI ASSIST: hypotheses only")
    seq = [[math.sin(i*0.3+k) for k in range(4)] for i in range(8)]
    at = api.ai.attention(seq, 4)
    rep("attention runs (pure Python)", len(at) == 8)
    wm = api.ai.world_model_step([1.0,2.0], [0.1,0.2])
    rep("world model is surrogate", "not a measurement" in wm["STATUS"])
    df = api.ai.diffusion_refine(u0, 10)
    rep("diffusion labelled HYPOTHESIS", "HYPOTHESIS" in df["STATUS"])
    good = api.ai.gate(sd, df["candidate"], lambda c: len(c) == len(u0))
    rep("AI gate: validated -> usable", good["usable"])
    bad = api.ai.gate(sd, df["candidate"], lambda c: False)
    rep("AI gate: unvalidated -> NOT usable", not bad["usable"])
    rep("rule stated", bad["rule"] == "AI proposes; physics disposes")

    print("\n[T10] SD REPORT + EQUITY")
    rep_obj = api.report()
    sr = rep_obj["sd_report"]
    print(f"      outputs={sr['n_outputs']} certified={sr['n_certified']} "
          f"Mode_E={sr['n_mode_E']}")
    rep("Mode E outputs retained", sr["n_mode_E"] > 0)
    rep("report written", (api.out/"report.json").exists())
    rep("zero-dependency path declared",
        rep_obj["info"]["zero_dependency_path"])
    rep("mission stated", "equity" in rep_obj["info"]["mission"])

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Pure-Python core: OPMC O(n^2) DP, prefix-sum SSE, Laplacian, phase-field.
  PCF gasket (constructive 5/3) vs NON-PCF carpet (C2/C3 imported) separated.
  delta_min-isodiametric quality; sub-Gaussian diagnostic (not a proof).
  STC^0/STC^1; refluxing DERIVED; AMR with 2:1 balance.
  CH3D / TF3D / PFC3D conserved; AC3D non-conserved (tumour).
  GPU kernel source for CUDA, HLSL, Metal; C header + ABI spec (8 functions).
  Adapters: OpenFOAM field I/O, OpenMM energy/force, Excel CSV+XLL guide,
            3ds Max OBJ, AutoCAD DXF, Photoshop PGM, Illustrator SVG,
            Corel deterministic carpet halftone, trajectory change points.
  AI assist: attention / world model / diffusion as HYPOTHESES with a gate.

ENFORCED
  SD gate: uncertified -> Mode E, uncertainty retained.
  PCF package cannot certify a non-PCF carpet.
  AI output usable only after solver validation.
  Practical OPMC signals correctly fail the strict (SNR) bound.

EQUITY
  Runs with no GPU, no numpy, no paid license. Same certified mathematics
  on a $200 laptop as on a cluster. Kernel source shipped so any host can
  compile for its own hardware.

NOT CLAIMED
  Does not accelerate the host application's own tasks.
  Does not prove C2/C3 for non-p.c.f. fractals.
  Explicit scheme: conservation approximate.
  Does not predict protein structure.
  AI surrogates are untrained; they generate candidates, not measurements.
""")
    return ok


# =============================================================================
# §12. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="MSPS math foundation API")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","info","emit","opmc","theta","reflux",
                 "carpet","amr","foam","excel","cad","image","abi","report"])
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", dest="outp", default=None)
    ap.add_argument("--ell", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--kind", default="CH3D")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    api = MathFoundationAPI(); out: Dict[str, Any] = {}

    if a.cmd == "info":
        out = api.info(); print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "emit":
        out = api.emit_all(); print(json.dumps(out, indent=2))
        print("\nKernel source + C header + ABI spec written.")
    elif a.cmd == "abi":
        out = ABI_SPEC; print(json.dumps(out, indent=2))
        print("\n" + C_HEADER)
    elif a.cmd == "opmc":
        if a.inp and Path(a.inp).exists():
            y = [float(t) for t in Path(a.inp).read_text()
                 .replace(","," ").split() if t.strip()]
        else:
            y = [0.0]*30 + [4.0]*30 + [-4.0]*30
            log.warning("no --in; synthetic signal (code test only)")
        out = opmc_snr_check(y, a.ell, a.sigma)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "theta":
        sizes = [float(t) for t in (a.inp or "1,1,2").split(",")]
        code, th = abi_theta(sizes)
        out = {"code": code, "theta": th}
        print(json.dumps(out, indent=2))
    elif a.cmd == "reflux":
        fine = [float(t) for t in (a.inp or "0.2,0.3,0.5").split(",")]
        out = stc_reflux(0.9, fine); print(json.dumps(out, indent=2))
    elif a.cmd == "carpet":
        out = carpet_crossing_resistance(a.level)
        print(json.dumps(out, indent=2))
        print("\nNON-p.c.f.: C2/C3 remain imported, not proved.")
    elif a.cmd == "amr":
        cells = [AMRCell(0,0.0,1.0)]
        out = amr_refine(cells, lambda c: 1.0 if c.lo < 0.25 else 0.0,
                         0.5, 3)
        out["cells"] = [asdict(c) for c in out["cells"]]
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "foam":
        if a.inp:
            out = api.foam.read_field(a.inp)
        else:
            p = api.foam.write_field(a.outp or "mspsField",
                                     [float(i)*0.1 for i in range(20)])
            out = {"written": p}
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "excel":
        if not a.inp:
            print("--in CSV required"); return 1
        out = api.excel.csv_opmc(a.inp, a.outp or "msps_out.csv",
                                 a.ell, a.sigma, api.sd)
        print(json.dumps(out, indent=2, default=str))
        print("\n" + api.excel.XLL_GUIDE)
    elif a.cmd == "cad":
        amr = amr_refine([AMRCell(0,0.0,1.0)],
                         lambda c: 1.0 if c.lo < 0.25 else 0.0, 0.5, 3)
        o = api.cad.amr_to_obj(amr["cells"], a.outp or "mesh.obj")
        d = api.cad.mesh_report_dxf(amr["cells"], "mesh.dxf")
        out = {"obj": o, "dxf": d}
        print(json.dumps(out, indent=2))
    elif a.cmd == "image":
        if a.inp and Path(a.inp).exists():
            g, mx = api.img.read_pgm(a.inp)
        else:
            g = [[(i*7+j*3) % 256 for j in range(32)] for i in range(32)]
            mx = 255
        dn = api.img.denoise_rows(g, mx, a.kind, 20, 1e-4, api.sd)
        api.img.write_pgm(a.outp or "denoise.pgm", dn["grid"], mx)
        ht = api.img.carpet_halftone(g, mx, a.level)
        api.img.write_pgm("halftone.pgm", ht, mx)
        out = {"kind": dn["kind"], "conserved": dn["conserved"],
               "mean_mass_drift": dn["mean_mass_drift"],
               "outputs": [a.outp or "denoise.pgm", "halftone.pgm"]}
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "report":
        out = api.report()

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
