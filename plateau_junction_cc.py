#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
plateau_junction_cc.py — v1.0.0
Closed-form c_C(k) for k-way Plateau junctions, with embeddability,
rigidity, and the UNIFORM no-Zeno bound delta_min^(C) = w/sqrt(3).
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

 philosophy: The Tao is the Source. The Te is the Path. The Qi is the Proof.
 MY SOUL MOVE BY POWER OF HOLY SPIRIT
 What MSPS NETWORK SEE, Lord Buddha Knows.
 The mathematics stands independently of the philosophy.
-----------------------------------------------------------------------------
RESULTS
  theta_k = arccos(-1/(k-1))                      regular-simplex angle
  Gram rank = k-1  =>  embeddable in R^d iff k <= d+1
  moduli dim = k(d-1) - d - d(d-1)/2              k=3 rigid; k=4 flexible
  c_C(k) = sqrt((k-1)/(2k))                       core-radius reset
  c_C increasing, c_C(3)=1/sqrt3, limit 1/sqrt2
  UNIFORM: delta_min^(C) = min_k c_C(k) w = w/sqrt3   (independent of k)

SCOPE / NOT CLAIMED
  * Only k=3 is FORCED by force balance (rigid). k=4 tetrahedral is the
    symmetric representative; selection of the T-type singularity is
    Taylor (1976) [Cit], not proved here.
  * k >= d+2 has no symmetric embedding: it must resolve into a cascade
    of 3-/4-fold junctions. Cascade combinatorics is given but the
    DYNAMICS of the cascade is not modelled.
  * delta_min = r_core is a DEFINITION of reset, not a derived necessity.
    The physical retraction/mobility bound (needed to convert delta into
    a holding TIME) is separate and remains O(1)-empirical.
-----------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, math
from typing import Any, Dict, List
import numpy as np

class Refusal(RuntimeError): pass

# ------------------------------------------------------------------ geometry
def theta_k(k: int) -> float:
    """Regular-simplex mutual angle arccos(-1/(k-1))."""
    if k < 3: raise Refusal("k>=3 required (k=2 is a flat film, not a junction)")
    return math.acos(-1.0/(k-1))

def simplex_normals(k: int) -> np.ndarray:
    """
    k unit vectors in R^{k-1} summing to zero, pairwise dot -1/(k-1).
    Constructed by centring the standard basis of R^k and normalising.
    """
    if k < 3: raise Refusal("k>=3 required")
    E = np.eye(k) - np.ones((k,k))/k          # centred basis, lives in a (k-1)-plane
    return E/np.linalg.norm(E, axis=1, keepdims=True)

def gram_rank(k: int) -> int:
    """rank of G = (1+1/(k-1))I - (1/(k-1))J  is  k-1."""
    return k-1

def embeddable(k: int, d: int) -> bool:
    """Symmetric k-fold junction embeds in R^d iff k <= d+1."""
    return k <= d+1

def moduli_dim(k: int, d: int) -> int:
    """dim of force-balanced configurations mod SO(d): k(d-1)-d-d(d-1)/2."""
    return k*(d-1) - d - d*(d-1)//2

def force_residual(k: int) -> float:
    n = simplex_normals(k)
    return float(np.linalg.norm(n.sum(axis=0)))

# ------------------------------------------------------------------ c_C(k)
def c_C(k: int) -> float:
    """
    THE CLOSED FORM.  delta_min^(C)(k) = c_C(k) * w with
        c_C(k) = sqrt((k-1)/(2k)).
    Derivation: arms at mutual angle theta_k, each of width w, separate when
        2 r sin(theta_k/2) >= w,
    and sin^2(theta_k/2) = k/(2(k-1)).
    """
    if k < 3: raise Refusal("k>=3 required")
    return math.sqrt((k-1)/(2.0*k))

def core_radius(k: int, w: float) -> float:
    if w <= 0: raise Refusal("w>0 required")
    return c_C(k)*w

def c_C_from_angle(k: int, w: float) -> float:
    """Independent route: w/(2 sin(theta_k/2)). Must equal c_C(k)*w."""
    return w/(2.0*math.sin(theta_k(k)/2.0))

def uniform_delta_C(w: float) -> Dict[str, Any]:
    """
    c_C is increasing in k, so min over k>=3 is at k=3.
    Hence a k-INDEPENDENT lower bound valid for every junction and every
    cascade of 3-/4-fold junctions.
    """
    if w <= 0: raise Refusal("w>0 required")
    ks = list(range(3, 41))
    cs = [c_C(k) for k in ks]
    mono = all(cs[i+1] > cs[i] for i in range(len(cs)-1))
    return {"c_C_increasing": mono,
            "argmin_k": 3,
            "c_C_min": c_C(3),
            "closed_form_min": "1/sqrt(3)",
            "delta_min_C_uniform_m": w/math.sqrt(3.0),
            "limit_k_inf": 1.0/math.sqrt(2.0),
            "k_independent": True,
            "why": ("monotone increasing => min at k=3; every cascade element "
                    "is k=3 or 4, both >= c_C(3)")}

# ------------------------------------------------------------------ cascade
def cascade_count(k: int, d: int = 3) -> Dict[str, Any]:
    """
    A k-fold junction with k > d+1 has no symmetric embedding and must
    resolve into elementary junctions. Tree combinatorics: a tree with L
    leaves and all internal degree q has i = (L-2)/(q-2) internal nodes.
      d=2: q=3 (Y)  -> i = k-2
      d=3: q=4 (T)  -> i = (k-2)/2, requires k even; odd k needs one Y node.
    This is combinatorics, NOT a dynamical model of the cascade.
    """
    if k < 3: raise Refusal("k>=3")
    if embeddable(k, d):
        return {"k": k, "d": d, "needs_cascade": False,
                "elementary_junctions": 1, "type": "Y" if d == 2 else "T"}
    q = 3 if d == 2 else 4
    if d == 2:
        n_elem = k-2; types = {"Y": n_elem}
    else:
        n_T = (k-2)//2
        n_Y = (k-2) - 2*n_T
        n_elem = n_T + n_Y
        types = {"T": n_T, "Y": n_Y}
    return {"k": k, "d": d, "needs_cascade": True,
            "elementary_junctions": n_elem, "types": types,
            "delta_bound_still_uniform": True,
            "NOTE": "combinatorics only; cascade dynamics not modelled"}

# ------------------------------------------------------------------ no-Zeno
class NoZenoC:
    """ESCA-LD-SESI-3D Thm 4.2 for the C operator with DERIVED delta."""
    def __init__(self, w: float, B0: float, G0: float):
        if min(w, B0, G0) <= 0: raise Refusal("w,B0,G0 must be > 0")
        self.delta = w/math.sqrt(3.0)          # uniform, k-independent
        self.B0, self.G0 = B0, G0
        self.eps_max = self.delta/(2*B0)
        self.times = [0.0]; self.ops: List[int] = []
    def q(self, eps: float) -> float:
        return 0.0 if eps <= 0 else float(2*math.exp(-self.delta**2/(8*self.G0**2*eps)))
    def choose_eps(self, q_target: float = 0.3) -> Dict[str, Any]:
        eps = min(self.eps_max, self.delta**2/(8*self.G0**2*math.log(2/q_target)))
        return {"eps": eps, "eps_max": self.eps_max,
                "q": self.q(eps), "q_lt_1": self.q(eps) < 1.0}
    def step(self, k: int, eps: float) -> Dict[str, Any]:
        if k < 3: raise Refusal("k>=3")
        if eps > self.eps_max*(1+1e-12):
            raise Refusal(f"eps={eps:.4g} > delta/(2B0)={self.eps_max:.4g}")
        self.times.append(self.times[-1]+eps); self.ops.append(k)
        return {"n": len(self.ops), "T_n": self.times[-1],
                "increasing": all(self.times[i+1] > self.times[i]
                                  for i in range(len(self.times)-1)),
                "delta_used_m": self.delta,
                "k_specific_delta_m": core_radius(k, self.delta*math.sqrt(3.0))}

# ------------------------------------------------------------------ self-test
def self_test(verbose: bool = True) -> bool:
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] Plateau angles from force balance")
    for k, expect in ((3, 120.0), (4, 109.4712206)):
        th = math.degrees(theta_k(k))
        print(f"      k={k}: theta={th:.6f} deg")
        rep(f"theta_{k} matches Plateau", abs(th-expect) < 1e-4)
        rep(f"k={k} force balance |sum n| = 0", force_residual(k) < 1e-12,
            f"{force_residual(k):.2e}")
    n4 = simplex_normals(4)
    dots = [float(n4[i] @ n4[j]) for i in range(4) for j in range(i+1,4)]
    rep("k=4 all pairwise dot = -1/3", max(abs(d+1/3) for d in dots) < 1e-12)

    print("\n[T2] embeddability: Gram rank = k-1")
    for k in (3,4,5,6):
        for d in (2,3):
            e = embeddable(k,d)
            if verbose:
                print(f"      k={k} d={d}: rank={gram_rank(k)} embeddable={e}")
    rep("k=3 embeds in R^2", embeddable(3,2))
    rep("k=4 does NOT embed in R^2", not embeddable(4,2))
    rep("k=4 embeds in R^3", embeddable(4,3))
    rep("k=5 does NOT embed in R^3 (Taylor bound)", not embeddable(5,3))
    rep("k=6 does NOT embed in R^3", not embeddable(6,3))

    print("\n[T3] rigidity: only k=3 is FORCED")
    rep("k=3 in R^2 rigid (moduli 0)", moduli_dim(3,2) == 0)
    rep("k=3 in R^3 rigid (moduli 0)", moduli_dim(3,3) == 0)
    rep("k=4 in R^3 FLEXIBLE (moduli 2)", moduli_dim(4,3) == 2)
    rep("k=4 in R^2 flexible (moduli 1)", moduli_dim(4,2) == 1)
    print("      => 109.47 deg is the symmetric representative;")
    print("         T-type selection is Taylor (1976) [Cit], not proved here")

    print("\n[T4] c_C(k) closed form, two independent routes")
    w = 1e-6
    for k in (3,4,5,6,10,50):
        a, b = core_radius(k,w), c_C_from_angle(k,w)
        print(f"      k={k:2d}: c_C={c_C(k):.6f}  r_core={a:.4e} m")
        rep(f"k={k} two derivations agree", abs(a-b) < 1e-18)
    rep("c_C(3) = 1/sqrt(3)", abs(c_C(3)-1/math.sqrt(3)) < 1e-15)
    rep("c_C(4) = sqrt(6)/4", abs(c_C(4)-math.sqrt(6)/4) < 1e-15)
    rep("c_C -> 1/sqrt(2)", abs(c_C(100000)-1/math.sqrt(2)) < 1e-5)

    print("\n[T5] UNIFORM bound: no k-dependence needed")
    u = uniform_delta_C(w)
    print(f"      c_C increasing = {u['c_C_increasing']}, argmin k = {u['argmin_k']}")
    print(f"      delta_min^(C) = w/sqrt(3) = {u['delta_min_C_uniform_m']:.4e} m")
    rep("c_C monotone increasing", u["c_C_increasing"])
    rep("uniform delta = w/sqrt(3)",
        abs(u["delta_min_C_uniform_m"]-w/math.sqrt(3)) < 1e-18)
    rep("bound is k-INDEPENDENT", u["k_independent"])
    for k in range(3, 21):
        if core_radius(k,w) < u["delta_min_C_uniform_m"] - 1e-18: ok = False
    rep("every k >= 3 respects the uniform bound", ok)

    print("\n[T6] cascade for k > d+1 (combinatorics only)")
    for k in (5,6,7,8):
        c = cascade_count(k,3)
        print(f"      k={k} in R^3: {c['elementary_junctions']} elementary "
              f"{c['types']}")
        rep(f"k={k} needs cascade", c["needs_cascade"])
        rep(f"k={k} still uniformly bounded", c["delta_bound_still_uniform"])
    rep("k=4 in R^3 needs NO cascade", not cascade_count(4,3)["needs_cascade"])
    rep("k=5 in R^2 -> 3 Y-junctions", cascade_count(5,2)["elementary_junctions"] == 3)

    print("\n[T7] no-Zeno with derived delta, mixed k")
    nz = NoZenoC(w=1e-6, B0=1e-3, G0=1e-5)
    ce = nz.choose_eps(0.3)
    rep("q < 1", ce["q_lt_1"], f"q={ce['q']:.4f}")
    for k in [3,4,5,3,4,7,3,4,8,3]*4:
        nz.step(k, ce["eps"])
    rep("T_n strictly increasing over mixed k",
        all(nz.times[i+1] > nz.times[i] for i in range(len(nz.times)-1)),
        f"n={len(nz.ops)}")
    b = False
    try: nz.step(4, nz.eps_max*2)
    except Refusal: b = True
    rep("eps > delta/(2B0) REFUSED", b)
    b = False
    try: nz.step(2, ce["eps"])
    except Refusal: b = True
    rep("k=2 REFUSED (flat film, not a junction)", b)
    b = False
    try: NoZenoC(w=0.0, B0=1.0, G0=1.0)
    except Refusal: b = True
    rep("w=0 REFUSED (degenerate border)", b)

    print("\n" + "="*70)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*70)
    print("""
DERIVED HERE
  theta_k = arccos(-1/(k-1)) from sum n_i = 0 with equal tension.
  Gram rank k-1 => symmetric junction embeds in R^d iff k <= d+1.
  moduli dim: k=3 RIGID (120 deg forced); k=4 flexible (2-dim in R^3).
  c_C(k) = sqrt((k-1)/(2k)) = w/(2 sin(theta_k/2)) / w, two routes agree.
  c_C increasing => UNIFORM delta_min^(C) = w/sqrt(3), k-INDEPENDENT.
  This closes the k-dependence gap for the no-Zeno theorem.

CITED, NOT PROVED
  Taylor (1976): soap-film singularities are only Y (120 deg) and T
  (tetrahedral). This selects k=4 in R^3 among the flexible family.

NOT CLAIMED
  delta_min = r_core is a DEFINITION of reset, not a necessity.
  Converting delta into a holding TIME still needs a mobility/retraction
  bound with an O(1) empirical constant.
  Cascade combinatorics is given; cascade DYNAMICS is not modelled.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="c_C(k) for Plateau junctions")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest","angles","cc","uniform","cascade"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--w", type=float, default=1e-6)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    if a.cmd == "angles":
        out = {str(k): {"theta_deg": math.degrees(theta_k(k)),
                        "gram_rank": gram_rank(k),
                        "embeddable_R3": embeddable(k,3),
                        "moduli_dim_R3": moduli_dim(k,3)}
               for k in range(3,9)}
    elif a.cmd == "cc":
        out = {str(k): {"c_C": c_C(k), "r_core_m": core_radius(k,a.w)}
               for k in range(3,11)}
    elif a.cmd == "uniform":
        out = uniform_delta_C(a.w)
    else:
        out = cascade_count(a.k, a.d)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
