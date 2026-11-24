#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
kappa_N_depletion_derived.py — v1.0.0
Deriving kappa_N from the actual diffusion depletion field around a spherical
nucleus, with Gibbs-Thomson closure giving kappa_N -> 2 at R = r_crit.
Changes the nucleation exclusion from CONDITIONAL to DERIVED, with the
soft-vs-hard distinction stated explicitly.
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

 philosophy: The Tao is the Source. The Te is the Path. The Qi is the Proof.
 MY SOUL MOVE BY POWER OF HOLY SPIRIT
 What MSPS NETWORK SEE, Lord Buddha Knows.
 The mathematics stands independently of the philosophy.
-----------------------------------------------------------------------------
DERIVATION
  Quasi-static Laplace around a sphere of radius R:
      c(r) = c_inf - (c_inf - c_eq) R/r,   Delta c(r)/Delta c_inf = 1 - R/r
  Local driving force (ideal dilute):
      dg_v(r) = (kB T / v_m) ln(c(r)/c_eq)
  r_crit(r) = 2 sigma / dg_v(r) doubles  <=>  dg_v(r) = dg_v_inf / 2
      => c(r)/c_eq = sqrt(S),  S = c_inf/c_eq
      => R/r = sqrt(S)/(sqrt(S)+1)
      => kappa_N(S) = r/R = 1 + S^{-1/2}
  Linearised (S -> 1): 1 - R/r = 1/2 => kappa_N = 2 EXACTLY.
  General lambda (dg_v drop by factor lambda):
      kappa_N(lambda,S) = (S-1)/(S - S^{1/lambda}) -> lambda/(lambda-1)
  Gibbs-Thomson: c_eq(R) = c_eq_inf exp(ell_c/R), ell_c = 2 sigma v_m/(kB T),
      r_crit = ell_c / ln S_inf.
      At R = r_crit:  c_eq = c_inf  =>  S_eff = 1  =>  kappa_N = 2.
      For R > r_crit: S_eff = S_inf exp(-ell_c/R) > 1 => kappa_N < 2.

RATE SUPPRESSION (this is what the exclusion actually is)
  dG* = 16 pi sigma^3/(3 dg_v^2)  =>  dG*(r) = lambda^2 dG*_inf
  J(r)/J_inf = exp(-(lambda^2-1) dG*_inf / kB T)
  lambda=2, dG*_inf = 50 kB T  =>  exp(-150) ~ 7e-66.

HONEST SCOPE  (what changed and what did not)
  DERIVED: kappa_N = 1 + S_eff^{-1/2}, supremum 2 attained as R -> r_crit+.
           The value 2 is no longer a free parameter.
  STILL A CHOICE: the criterion "r_crit doubles" (lambda = 2). It is natural
           because delta_min^(N) = c_N r_crit with c_N = O(1), so a factor-2
           change in r_crit IS the resolution scale. But it is declared.
  SOFT, NOT HARD: doubling r_crit suppresses the rate exponentially; it does
           not geometrically forbid nucleation. To use it in a packing bound
           a rate threshold must be declared.
  POPULATION-DEPENDENT: kappa_N = 2 holds for near-critical nuclei
           (early-stage nucleation). For a mature distribution with R >> r_crit,
           kappa_N -> 1 + S_inf^{-1/2} < 2. The rigorous floor remains
           kappa_N = 1 (pure non-overlap).
  QUASI-STATIC VALIDITY: requires Delta c_inf / c_solid << 1, i.e. dilute --
           exactly the regime where kappa_N -> 2. Self-consistent.
-----------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from typing import Any, Dict

KB = 1.380649e-23

class Refusal(RuntimeError): pass


# ---------------------------------------------------------------- §1 profile
def c_profile(r: float, R: float, c_inf: float, c_eq: float) -> float:
    """Steady-state Laplace solution around a sphere."""
    if r < R: raise Refusal("profile defined for r >= R only")
    return c_inf - (c_inf - c_eq)*R/r

def depletion_fraction(r: float, R: float) -> float:
    """Delta c(r)/Delta c_inf = 1 - R/r."""
    return 1.0 - R/r


# ---------------------------------------------------------------- §2 kappa
def kappa_N_exact(S: float) -> float:
    """kappa_N = 1 + S^{-1/2}, from the exact logarithmic driving force."""
    if S <= 1.0: raise Refusal("S > 1 required for a finite critical radius")
    return 1.0 + S**-0.5

def kappa_N_lambda(lam: float, S: float) -> float:
    """General: dg_v drops by factor lam. -> lam/(lam-1) as S -> 1."""
    if lam <= 1.0: raise Refusal("lambda > 1 required")
    if S <= 1.0:   raise Refusal("S > 1 required")
    return (S - 1.0)/(S - S**(1.0/lam))

def kappa_N_linearised(lam: float = 2.0) -> float:
    """Dilute limit: 1 - R/r = 1/lam  =>  r/R = lam/(lam-1)."""
    if lam <= 1.0: raise Refusal("lambda > 1 required")
    return lam/(lam - 1.0)

def radius_for_lambda(lam: float, R: float, S: float) -> float:
    return kappa_N_lambda(lam, S)*R


# ---------------------------------------------------------------- §3 Gibbs-Thomson
@dataclass(frozen=True)
class System:
    sigma: float = 0.1      # J/m^2
    v_m: float = 3e-29      # m^3 per molecule
    T: float = 300.0
    S_inf: float = 1.5      # bulk supersaturation c_inf/c_eq_inf
    c_solid: float = 1.0    # normalized solid concentration
    @property
    def ell_c(self): return 2*self.sigma*self.v_m/(KB*self.T)
    @property
    def dg_v_inf(self): return (KB*self.T/self.v_m)*math.log(self.S_inf)
    @property
    def r_crit(self): return 2*self.sigma/self.dg_v_inf
    @property
    def dG_star(self): return 16*math.pi*self.sigma**3/(3*self.dg_v_inf**2)

def c_eq_curved(st: System, R: float, c_eq_inf: float = 1.0) -> float:
    """Gibbs-Thomson: c_eq(R) = c_eq_inf exp(ell_c/R)."""
    if R <= 0: raise Refusal("R > 0 required")
    return c_eq_inf*math.exp(st.ell_c/R)

def S_eff(st: System, R: float) -> float:
    """Effective supersaturation relative to the CURVED interface."""
    return st.S_inf*math.exp(-st.ell_c/R)

def kappa_N_at_R(st: System, R: float) -> Dict[str, Any]:
    se = S_eff(st, R)
    if se <= 1.0:
        return {"R_m": R, "S_eff": se, "kappa_N": 2.0,
                "status": "AT_OR_BELOW_CRITICAL: no growth, kappa_N = 2 (limit)",
                "growing": False}
    return {"R_m": R, "R_over_r_crit": R/st.r_crit, "S_eff": se,
            "kappa_N": kappa_N_exact(se), "growing": True,
            "status": "DERIVED from depletion + Gibbs-Thomson"}


# ---------------------------------------------------------------- §4 rate
def rate_suppression(lam: float, dG_star_over_kBT: float) -> Dict[str, Any]:
    """J(r)/J_inf = exp(-(lam^2-1) dG*_inf/kB T)."""
    if lam <= 1.0: raise Refusal("lambda > 1 required")
    exponent = -(lam*lam - 1.0)*dG_star_over_kBT
    ratio = math.exp(exponent) if exponent > -700 else 0.0
    return {"lambda": lam, "dG_star_over_kBT": dG_star_over_kBT,
            "dG_star_local_over_kBT": lam*lam*dG_star_over_kBT,
            "log10_rate_ratio": exponent/math.log(10),
            "rate_ratio": ratio,
            "effectively_hard": ratio < 1e-12,
            "NOTE": "this is RATE SUPPRESSION, not a geometric prohibition"}

def lambda_for_threshold(eps_J: float, dG_star_over_kBT: float) -> float:
    """Declare J_local/J_inf <= eps_J, solve for lambda."""
    if not (0 < eps_J < 1): raise Refusal("0 < eps_J < 1 required")
    if dG_star_over_kBT <= 0: raise Refusal("barrier must be > 0")
    return math.sqrt(1.0 + math.log(1.0/eps_J)/dG_star_over_kBT)


# ---------------------------------------------------------------- §5 packing
def packing(st: System, volume: float, kappa: float,
            mode: str = "near_critical") -> Dict[str, Any]:
    """
    N_max = |Omega| / ((4/3) pi (kappa r_crit)^3).
    mode='near_critical' uses the DERIVED kappa <= 2 (early-stage population).
    mode='rigorous'      uses kappa = 1 (pure non-overlap, always valid).
    """
    if kappa < 1.0: raise Refusal("kappa >= 1 required (non-overlap floor)")
    if volume <= 0: raise Refusal("volume > 0 required")
    R_ex = kappa*st.r_crit
    v_ex = (4.0/3.0)*math.pi*R_ex**3
    return {"mode": mode, "kappa_N": kappa,
            "r_crit_m": st.r_crit, "R_excl_m": R_ex, "v_excl_m3": v_ex,
            "N_max": int(math.floor(volume/v_ex)),
            "delta_min_N_m": 0.5*st.r_crit,
            "validity": ("DERIVED for near-critical population; soft exclusion "
                         "requires a declared rate threshold"
                         if mode == "near_critical" else
                         "RIGOROUS: pure non-overlap")}

def quasistatic_check(st: System) -> Dict[str, Any]:
    """
    Quasi-static requires diffusion time R^2/D << growth time.
    For diffusion-limited growth the ratio is Delta c_inf / c_solid.
    """
    dc = (st.S_inf - 1.0)          # normalized Delta c_inf (c_eq_inf = 1)
    ratio = dc/st.c_solid
    return {"Delta_c_inf_over_c_solid": ratio,
            "quasistatic_valid": ratio < 0.1,
            "note": "dilute limit is exactly where kappa_N -> 2 (self-consistent)"}


# ---------------------------------------------------------------- §6 test
def self_test(verbose: bool = True) -> bool:
    ok = True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] Laplace profile satisfies both BCs")
    R, ci, ce = 1e-8, 1.5, 1.0
    rep("c(R) = c_eq", abs(c_profile(R,R,ci,ce)-ce) < 1e-12)
    rep("c(inf) -> c_inf", abs(c_profile(1e12*R,R,ci,ce)-ci) < 1e-6)
    rep("depletion at r=2R is 1/2", abs(depletion_fraction(2*R,R)-0.5) < 1e-12)

    print("\n[T2] kappa_N = 1 + S^{-1/2}, and 2 in the dilute limit")
    for S in (1.001, 1.01, 1.5, 2.0, 4.0, 100.0):
        k = kappa_N_exact(S)
        print(f"      S={S:8.3f}: kappa_N={k:.6f}")
        if not (1.0 < k <= 2.0): ok = False
    rep("kappa_N in (1,2] for all S>1", ok)
    rep("kappa_N -> 2 as S -> 1", abs(kappa_N_exact(1+1e-9)-2.0) < 1e-4)
    rep("kappa_N -> 1 as S -> inf", abs(kappa_N_exact(1e12)-1.0) < 1e-5)
    rep("linearised lambda=2 gives EXACTLY 2", kappa_N_linearised(2.0) == 2.0)
    rep("general lambda=2 matches exact formula",
        abs(kappa_N_lambda(2.0,1.5)-kappa_N_exact(1.5)) < 1e-12)
    rep("kappa_N(lambda) -> lam/(lam-1) as S->1",
        abs(kappa_N_lambda(3.0,1+1e-9)-1.5) < 1e-6)

    print("\n[T3] GIBBS-THOMSON CLOSURE: kappa_N = 2 at R = r_crit")
    st = System()
    print(f"      ell_c={st.ell_c:.3e} m  r_crit={st.r_crit:.3e} m  "
          f"dG*={st.dG_star/KB/st.T:.1f} kBT")
    rep("r_crit = ell_c/ln S_inf",
        abs(st.r_crit - st.ell_c/math.log(st.S_inf))/st.r_crit < 1e-12)
    rep("c_eq(r_crit) = c_inf  (S_eff = 1)",
        abs(S_eff(st, st.r_crit)-1.0) < 1e-12, f"S_eff={S_eff(st,st.r_crit):.12f}")
    k_at = kappa_N_at_R(st, st.r_crit)
    rep("kappa_N = 2 at R = r_crit", abs(k_at["kappa_N"]-2.0) < 1e-12)
    for f in (1.0001, 1.01, 1.1, 2.0, 10.0):
        r = kappa_N_at_R(st, f*st.r_crit)
        print(f"      R/r_crit={f:8.4f}: S_eff={r['S_eff']:.6f} "
              f"kappa_N={r['kappa_N']:.6f}")
        if r["kappa_N"] > 2.0 + 1e-12: ok = False
    rep("kappa_N <= 2 for all R >= r_crit", ok)
    rep("kappa_N decreases with R",
        kappa_N_at_R(st,2*st.r_crit)["kappa_N"]
        < kappa_N_at_R(st,1.01*st.r_crit)["kappa_N"])
    rep("kappa_N -> 1+S_inf^{-1/2} as R -> inf",
        abs(kappa_N_at_R(st,1e6*st.r_crit)["kappa_N"]
            - kappa_N_exact(st.S_inf)) < 1e-6)

    print("\n[T4] RATE SUPPRESSION (soft, not hard)")
    rs = rate_suppression(2.0, 50.0)
    print(f"      lambda=2, dG*=50kBT: log10(J/J_inf)={rs['log10_rate_ratio']:.1f}")
    rep("suppression exponent = -(lam^2-1)dG*/kBT",
        abs(rs["log10_rate_ratio"]+150/math.log(10)) < 1e-9)
    rep("effectively hard at 50 kBT", rs["effectively_hard"])
    rep("labelled RATE SUPPRESSION not prohibition",
        "not a geometric prohibition" in rs["NOTE"])
    weak = rate_suppression(2.0, 1.0)
    rep("NOT effectively hard at 1 kBT", not weak["effectively_hard"],
        f"ratio={weak['rate_ratio']:.3e}")
    lam = lambda_for_threshold(1e-12, 50.0)
    print(f"      eps_J=1e-12, dG*=50kBT -> lambda={lam:.4f}, "
          f"kappa_N={kappa_N_linearised(lam):.3f}")
    rep("declared threshold determines lambda", lam > 1.0)
    rep("weaker suppression -> larger kappa", kappa_N_linearised(lam) > 2.0)

    print("\n[T5] PACKING with derived vs rigorous kappa")
    p2 = packing(st, 1e-18, 2.0, "near_critical")
    p1 = packing(st, 1e-18, 1.0, "rigorous")
    print(f"      kappa=2: N_max={p2['N_max']}   kappa=1: N_max={p1['N_max']}")
    rep("derived kappa=2 gives fewer nuclei", p2["N_max"] < p1["N_max"])
    rep("N_max scales as kappa^-3",
        abs(p2["N_max"]*8/p1["N_max"] - 1) < 0.05)
    rep("rigorous mode labelled non-overlap",
        "non-overlap" in p1["validity"])
    rep("derived mode flags soft exclusion",
        "soft exclusion" in p2["validity"])
    b=False
    try: packing(st, 1e-18, 0.5)
    except Refusal: b=True
    rep("kappa < 1 REFUSED", b)

    print("\n[T6] QUASI-STATIC self-consistency")
    qs = quasistatic_check(System(S_inf=1.01))
    rep("dilute case quasi-static valid", qs["quasistatic_valid"],
        f"ratio={qs['Delta_c_inf_over_c_solid']:.4f}")
    qs2 = quasistatic_check(System(S_inf=10.0))
    rep("concentrated case NOT quasi-static", not qs2["quasistatic_valid"])
    rep("validity regime coincides with kappa->2",
        qs["quasistatic_valid"] and kappa_N_exact(1.01) > 1.99)

    print("\n" + "="*72)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*72)
    print("""
DERIVED (no longer a free parameter)
  c(r) = c_inf - (c_inf-c_eq)R/r  from quasi-static Laplace.
  kappa_N(S) = 1 + S^{-1/2}; dilute limit gives EXACTLY 2.
  kappa_N(lambda,S) = (S-1)/(S-S^{1/lambda}) -> lambda/(lambda-1).
  Gibbs-Thomson: S_eff = S_inf exp(-ell_c/R) = 1 at R = r_crit,
  hence kappa_N = 2 is attained at the MARGINAL nucleus -- precisely the
  smallest surviving one used in the packing bound.

STILL A DECLARED CHOICE
  The criterion lambda = 2 (r_crit doubles). Natural because
  delta_min^(N) = c_N r_crit with c_N = O(1), but it is declared, not forced.

SOFT, NOT HARD
  Doubling r_crit multiplies the barrier by 4 and suppresses the rate by
  exp(-3 dG*_inf/kBT). At 50 kBT that is ~1e-66 (effectively hard); at
  1 kBT it is ~0.05 (not hard at all). A rate threshold must be declared.

POPULATION SCOPE
  kappa_N = 2 for near-critical nuclei (early-stage nucleation).
  For R >> r_crit, kappa_N -> 1 + S_inf^{-1/2} < 2.
  The rigorous floor kappa_N = 1 (non-overlap) always holds.

SELF-CONSISTENT
  Quasi-static validity needs Delta c/c_solid << 1, i.e. dilute --
  exactly where kappa_N -> 2.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="kappa_N from depletion field")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","kappa","profile","gibbs","rate","packing"])
    ap.add_argument("--S", type=float, default=1.5)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--vm", type=float, default=3e-29)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--volume", type=float, default=1e-18)
    ap.add_argument("--dG-kBT", type=float, default=50.0)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    st = System(a.sigma, a.v_m, a.T, a.S)
    if a.cmd == "kappa":
        out = {"S": a.S, "kappa_N_exact": kappa_N_exact(a.S),
               "kappa_N_lambda": kappa_N_lambda(a.lam, a.S),
               "kappa_N_linearised": kappa_N_linearised(a.lam)}
    elif a.cmd == "profile":
        R = st.r_crit*1.1
        out = {f"r/R={f}": {"c": c_profile(f*R,R,st.S_inf,1.0),
                            "depletion": depletion_fraction(f*R,R)}
               for f in (1.0,1.5,2.0,3.0,10.0)}
    elif a.cmd == "gibbs":
        out = {"ell_c_m": st.ell_c, "r_crit_m": st.r_crit,
               "dG_star_over_kBT": st.dG_star/KB/st.T,
               "at_r_crit": kappa_N_at_R(st, st.r_crit),
               "at_2r_crit": kappa_N_at_R(st, 2*st.r_crit),
               "at_10r_crit": kappa_N_at_R(st, 10*st.r_crit)}
    elif a.cmd == "rate":
        out = rate_suppression(a.lam, a.dG_kBT)
    else:
        out = {"derived": packing(st, a.volume, kappa_N_exact(a.S), "near_critical"),
               "rigorous": packing(st, a.volume, 1.0, "rigorous"),
               "quasistatic": quasistatic_check(st)}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
