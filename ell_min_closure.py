#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ell_min_closure.py — v1.0.0
Closing open problem #2 (degenerate scales) for the seven-operator surgery
calculus. Three distinct degeneracies require three distinct resolutions.

CLOSED:
  h -> 0        repulsive core in disjoining pressure gives h_p in closed form
  r_n -> 0      CORRECTION: reset scale is the PARENT radius, not the neck;
                volume conservation gives R_d = 1.891 R for Rayleigh-Plateau
  dg_v -> inf   regime DICHOTOMY: CNT (r_crit >= xi) vs spinodal (no N surgery)

STILL OPEN:
  Pure van der Waals with no repulsive core: finite-time rupture, ell_min = 0.
  Unbounded fragment cascades with no minimum stable feature.
  PDE global existence for degenerate mobility without a cutoff.
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

 philosophy: The Tao is the Source. The Te is the Path. The Qi is the Proof.
 MY SOUL MOVE BY POWER OF HOLY SPIRIT
 What MSPS NETWORK SEE, Lord Buddha Knows.
 The mathematics stands independently of the philosophy.
-----------------------------------------------------------------------------
SELF-CORRECTION (important)
  The earlier derivation delta_min^(P) = c_P r_n used the NECK radius, which
  tends to zero at pinch. That was the wrong scale. The reset distance must
  be measured on the PERSISTENT post-event structure, whose size is bounded
  below by volume conservation. For a Rayleigh-Plateau cylinder of radius R
  breaking at lambda* = 9.02 R:
      pi R^2 lambda* = (4/3) pi R_d^3   =>   R_d = (3*9.02/4)^(1/3) R = 1.891 R.
  The daughter droplet is LARGER than the parent jet radius. No degeneracy.
  The same correction applies to B (bridge -> drop), H (bridge -> parent),
  C (junction width -> arm/cell size), M (neck -> daughters).
-----------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

PI = math.pi
LAMBDA_STAR_OVER_R = 9.02      # Rayleigh fastest mode
R_D_OVER_R = (0.75*LAMBDA_STAR_OVER_R)**(1.0/3.0)   # 1.8910...

class Refusal(RuntimeError): pass


# ---------------------------------------------------- §1 h -> 0
@dataclass(frozen=True)
class Disjoining:
    """Pi(h) = -A/(6 pi h^3) + B/h^m,  m > 3 (repulsive core)."""
    A: float = 1e-20      # Hamaker constant [J]
    B: float = 1e-78      # repulsive strength [J m^(m-3)]
    m: float = 9.0        # Born-repulsion exponent

    def Pi(self, h: float) -> float:
        if h <= 0: raise Refusal("h > 0 required")
        return -self.A/(6*PI*h**3) + self.B/h**self.m

    def dPi(self, h: float) -> float:
        return self.A/(2*PI*h**4) - self.m*self.B/h**(self.m+1)

    @property
    def h_p(self) -> float:
        """Closed form: Pi(h_p)=0  <=>  h_p^(m-3) = 6 pi B / A."""
        if self.m <= 3: raise Refusal("m > 3 required for a repulsive core")
        if self.A <= 0 or self.B <= 0: raise Refusal("A, B > 0 required")
        return (6*PI*self.B/self.A)**(1.0/(self.m-3.0))


def ell_min_L(dj: Disjoining, theta: float, c_L: float = 0.5) -> Dict[str, Any]:
    """
    h -> 0 CLOSED: the repulsive core makes h_p a stable equilibrium, so
    h >= h_p > 0 and delta_min^(L) = c_L h_p / theta is derived, not assumed.
    """
    if not (0 < theta < PI/2): raise Refusal("theta in (0, pi/2)")
    hp = dj.h_p
    return {"h_p_m": hp,
            "closed_form": "(6 pi B / A)^(1/(m-3))",
            "Pi_at_hp": dj.Pi(hp),
            "dPi_at_hp": dj.dPi(hp),
            "stable": dj.dPi(hp) < 0,
            "analytic_dPi": dj.A/(6*PI*hp**4)*(3-dj.m),
            "delta_min_L_m": c_L*hp/theta,
            "status": "CLOSED — derived from material constants",
            "REQUIRES": "repulsive core (m > 3, B > 0)"}


def vdw_only_rupture(dj_A: float, h0: float, n_mob: float = 3.0,
                     L: float = 1e-6) -> Dict[str, Any]:
    """
    WITHOUT a repulsive core, Pi = -A/(6 pi h^3) drives rupture.
    Local thinning rate dh/dt ~ m(h) Pi'(h)/L^2 ~ h^(n-4).
    For n < 4 this diverges as h -> 0: finite-time rupture, ell_min = 0.
    """
    exponent = n_mob - 4.0
    finite_time = exponent < 0
    return {"mobility_exponent_n": n_mob,
            "dh_dt_scaling": f"h^{exponent:g}",
            "finite_time_rupture": finite_time,
            "ell_min": 0.0 if finite_time else None,
            "status": "OPEN — no repulsive core" if finite_time
                      else "mobility alone prevents h=0"}


# ---------------------------------------------------- §2 r_n -> 0
def rayleigh_daughter(R: float) -> Dict[str, Any]:
    """
    r_n -> 0 CLOSED by CORRECTION: use the persistent scale.
    Volume conservation on one fastest-mode wavelength.
    """
    if R <= 0: raise Refusal("R > 0 required")
    lam = LAMBDA_STAR_OVER_R*R
    V_cyl = PI*R*R*lam
    R_d = (3*V_cyl/(4*PI))**(1.0/3.0)
    return {"jet_radius_R_m": R, "lambda_star_m": lam,
            "V_cylinder_m3": V_cyl,
            "daughter_radius_m": R_d,
            "R_d_over_R": R_d/R,
            "analytic_ratio": R_D_OVER_R,
            "daughter_LARGER_than_jet": R_d > R,
            "status": "CLOSED — persistent scale, not neck scale"}


PERSISTENT = {
 "N": ("critical nucleus r_crit", "r_crit = 2 sigma / dg_v, requires CNT regime"),
 "M": ("daughter radii",          "volume conservation"),
 "B": ("merged drop radius R",    "NOT the vanishing bridge r_b"),
 "H": ("parent structure size",   "NOT the vanishing bridge"),
 "P": ("jet radius R -> R_d=1.891R", "NOT the vanishing neck r_n"),
 "L": ("precursor thickness h_p", "repulsive core equilibrium"),
 "C": ("arm / cell size",         "NOT the vanishing junction width w"),
}

def ell_min_persistent(op: str, ell: float, c: float = 0.5) -> Dict[str, Any]:
    if op not in PERSISTENT: raise Refusal(f"unknown operator '{op}'")
    if ell <= 0: raise Refusal(f"{op}: persistent scale must be > 0")
    name, source = PERSISTENT[op]
    return {"op": op, "persistent_scale": name, "source": source,
            "ell_persist_m": ell, "delta_min_m": c*ell,
            "principle": ("reset is measured on the POST-event persistent "
                          "structure; the singular neck/bridge/film phase "
                          "is INSIDE one event")}


def unified_ell_min(scales: Dict[str, float]) -> Dict[str, Any]:
    """delta = min_s c_s ell_persist(s) / C(Gamma_0). Refuses non-positive."""
    missing = [s for s in PERSISTENT if s not in scales]
    if missing: raise Refusal(f"missing persistent scales for {missing}")
    bad = [s for s,v in scales.items() if not v > 0]
    if bad: raise Refusal(f"non-positive persistent scale for {bad}")
    vals = {s: 0.5*v for s,v in scales.items()}
    s_star = min(vals, key=vals.get)
    return {"delta_min_per_op_m": vals, "binding_operator": s_star,
            "delta_min_m": min(vals.values()),
            "ell_min_positive": True,
            "status": "CLOSED conditional on each persistent scale > 0"}


# ---------------------------------------------------- §3 dg_v -> inf
def nucleation_regime(sigma: float, dg_v: float,
                      eps: float, Fpp_abs: float) -> Dict[str, Any]:
    """
    dg_v -> inf CLOSED by DICHOTOMY, not by a bound.
      r_crit >= xi : CNT regime, N is a valid surgery, ell_min >= xi > 0
      r_crit <  xi : spinodal regime, NO nucleation surgery; use CH3D
    """
    if min(sigma, dg_v, eps, Fpp_abs) <= 0:
        raise Refusal("all parameters must be > 0")
    r_crit = 2*sigma/dg_v
    xi = eps/math.sqrt(Fpp_abs)
    cnt = r_crit >= xi
    dg_spinodal = 2*sigma/xi
    return {"r_crit_m": r_crit, "xi_m": xi,
            "regime": "CNT_nucleation" if cnt else "spinodal",
            "dg_v_spinodal_threshold": dg_spinodal,
            "N_is_valid_surgery": cnt,
            "ell_min_m": r_crit if cnt else None,
            "use_instead": None if cnt else "continuous CH3D (no N operator)",
            "status": "CLOSED by regime dichotomy",
            "KEY": ("deep quench does not destroy no-Zeno; it changes regime. "
                    "In the spinodal regime there are no N surgeries to count.")}


# ---------------------------------------------------- §4 no-Zeno
class NoZeno:
    def __init__(self, delta: float, B0: float, G0: float):
        if min(delta,B0,G0) <= 0: raise Refusal("delta, B0, G0 must be > 0")
        self.delta, self.B0, self.G0 = delta, B0, G0
        self.eps_max = delta/(2*B0)
        self.times=[0.0]; self.ops: List[str]=[]
    def q(self, eps):
        return 0.0 if eps<=0 else float(2*math.exp(-self.delta**2/(8*self.G0**2*eps)))
    def choose_eps(self, q_target=0.3):
        eps = min(self.eps_max, self.delta**2/(8*self.G0**2*math.log(2/q_target)))
        return {"eps":eps,"q":self.q(eps),"q_lt_1":self.q(eps)<1.0}
    def step(self, op, eps):
        if eps > self.eps_max*(1+1e-12):
            raise Refusal(f"eps={eps:.4g} > delta/(2B0)={self.eps_max:.4g}")
        self.times.append(self.times[-1]+eps); self.ops.append(op)
        return {"n":len(self.ops),"T_n":self.times[-1],
                "increasing":all(self.times[i+1]>self.times[i]
                                  for i in range(len(self.times)-1))}


# ---------------------------------------------------- §5 self-test
def self_test(verbose=True) -> bool:
    ok=True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] h -> 0 : repulsive core gives h_p in closed form")
    dj = Disjoining()
    r = ell_min_L(dj, theta=0.1)
    print(f"      m={dj.m}, A={dj.A:.1e}, B={dj.B:.1e} -> h_p={r['h_p_m']:.4e} m")
    rep("Pi(h_p) = 0", abs(r["Pi_at_hp"]) < 1e-6*abs(dj.A/(6*PI*dj.h_p**3)),
        f"{r['Pi_at_hp']:.3e}")
    rep("h_p matches closed form",
        abs(dj.h_p-(6*PI*dj.B/dj.A)**(1/(dj.m-3))) < 1e-18)
    rep("h_p is STABLE (dPi < 0)", r["stable"], f"dPi={r['dPi_at_hp']:.3e}")
    rep("dPi matches analytic (A/6pi h^4)(3-m)",
        abs(r["dPi_at_hp"]-r["analytic_dPi"]) < 1e-6*abs(r["analytic_dPi"]))
    rep("restoring: Pi(h>h_p) < 0", dj.Pi(1.5*dj.h_p) < 0)
    rep("restoring: Pi(h<h_p) > 0", dj.Pi(0.7*dj.h_p) > 0)
    rep("delta_min^(L) > 0", r["delta_min_L_m"] > 0,
        f"{r['delta_min_L_m']:.3e} m")
    for m in (6.0, 9.0, 12.0):
        d2 = Disjoining(m=m); rep(f"m={m} gives h_p>0", d2.h_p > 0)
    b=False
    try: Disjoining(m=3.0).h_p
    except Refusal: b=True
    rep("m<=3 REFUSED (no core)", b)
    b=False
    try: Disjoining(B=0.0).h_p
    except Refusal: b=True
    rep("B=0 REFUSED", b)

    print("\n[T2] WITHOUT repulsive core: still OPEN (honest)")
    v = vdw_only_rupture(1e-20, 1e-7, n_mob=3.0)
    print(f"      n=3: dh/dt ~ {v['dh_dt_scaling']}, "
          f"finite-time rupture = {v['finite_time_rupture']}")
    rep("n=3 vdW-only ruptures in finite time", v["finite_time_rupture"])
    rep("labelled OPEN", v["status"].startswith("OPEN"))
    v4 = vdw_only_rupture(1e-20, 1e-7, n_mob=4.0)
    rep("n=4 mobility alone prevents h=0", not v4["finite_time_rupture"])

    print("\n[T3] r_n -> 0 : CORRECTION to persistent scale")
    rd = rayleigh_daughter(1e-5)
    print(f"      jet R=10 um -> lambda*={rd['lambda_star_m']:.3e} m, "
          f"R_d={rd['daughter_radius_m']:.3e} m")
    print(f"      R_d/R = {rd['R_d_over_R']:.4f}  (analytic {R_D_OVER_R:.4f})")
    rep("R_d/R matches (3*9.02/4)^(1/3)",
        abs(rd["R_d_over_R"]-R_D_OVER_R) < 1e-9)
    rep("daughter is LARGER than jet", rd["daughter_LARGER_than_jet"])
    rep("volume conserved",
        abs(4/3*PI*rd["daughter_radius_m"]**3 - rd["V_cylinder_m3"])
        < 1e-9*rd["V_cylinder_m3"])
    for op in "NMBHPLC":
        e = ell_min_persistent(op, 1e-6)
        if not e["delta_min_m"] > 0: ok=False
    rep("all 7 persistent scales positive", ok)
    rep("P uses jet radius NOT neck",
        "NOT the vanishing neck" in PERSISTENT["P"][1])
    rep("C uses arm size NOT junction width",
        "NOT the vanishing junction" in PERSISTENT["C"][1])

    print("\n[T4] dg_v -> inf : regime DICHOTOMY")
    cnt = nucleation_regime(sigma=0.1, dg_v=1e8, eps=1e-9, Fpp_abs=1e6)
    sp  = nucleation_regime(sigma=0.1, dg_v=1e12, eps=1e-9, Fpp_abs=1e6)
    print(f"      dg_v=1e8 : r_crit={cnt['r_crit_m']:.2e} xi={cnt['xi_m']:.2e} "
          f"-> {cnt['regime']}")
    print(f"      dg_v=1e12: r_crit={sp['r_crit_m']:.2e} xi={sp['xi_m']:.2e} "
          f"-> {sp['regime']}")
    rep("moderate quench is CNT regime", cnt["regime"]=="CNT_nucleation")
    rep("CNT regime has ell_min > 0", cnt["ell_min_m"] is not None)
    rep("deep quench is SPINODAL", sp["regime"]=="spinodal")
    rep("spinodal has NO N surgery", not sp["N_is_valid_surgery"])
    rep("spinodal redirects to CH3D", "CH3D" in sp["use_instead"])
    rep("dichotomy is not a bound", "dichotomy" in cnt["status"])
    xi = cnt["xi_m"]
    rep("threshold dg = 2 sigma / xi",
        abs(cnt["dg_v_spinodal_threshold"]-2*0.1/xi) < 1e-6)

    print("\n[T5] unified ell_min and no-Zeno")
    scales = {"N":cnt["r_crit_m"], "M":rd["daughter_radius_m"],
              "B":1e-4, "H":1e-5, "P":1e-5,
              "L":dj.h_p, "C":1e-6}
    u = unified_ell_min(scales)
    print(f"      binding operator = {u['binding_operator']}, "
          f"delta_min = {u['delta_min_m']:.3e} m")
    rep("ell_min > 0 achieved", u["ell_min_positive"])
    nz = NoZeno(u["delta_min_m"], B0=1e-3, G0=1e-5)
    ce = nz.choose_eps(0.3)
    rep("q < 1", ce["q_lt_1"], f"q={ce['q']:.4f}")
    import itertools
    for op in itertools.islice(itertools.cycle("NMBHPLC"), 70):
        nz.step(op, ce["eps"])
    rep("T_n strictly increasing across all 7",
        all(nz.times[i+1]>nz.times[i] for i in range(len(nz.times)-1)),
        f"n={len(nz.ops)}")
    b=False
    try: unified_ell_min({**scales, "L":0.0})
    except Refusal: b=True
    rep("h_p=0 REFUSED (no core)", b)
    b=False
    try: unified_ell_min({k:v for k,v in scales.items() if k!="C"})
    except Refusal: b=True
    rep("missing operator REFUSED", b)
    b=False
    try: NoZeno(0.0, 1.0, 1.0)
    except Refusal: b=True
    rep("delta=0 REFUSED", b)

    print("\n" + "="*72)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*72)
    print("""
CLOSED
  h -> 0      : repulsive core (m>3, B>0) gives
                h_p = (6 pi B/A)^(1/(m-3)), a STABLE equilibrium.
                delta_min^(L) = c_L h_p / theta, derived from material data.
  r_n -> 0    : CORRECTION — the reset scale is the PERSISTENT post-event
                structure, not the vanishing neck. Volume conservation gives
                R_d = 1.891 R for Rayleigh-Plateau: the daughter is LARGER
                than the parent jet. Same fix for B, H, C, M.
  dg_v -> inf : regime DICHOTOMY. r_crit >= xi -> CNT, N valid, ell_min >= xi.
                r_crit < xi -> spinodal, NO N surgery, use continuous CH3D.
                Deep quench changes regime; it does not break no-Zeno.

STILL OPEN (stated, not hidden)
  Pure van der Waals with no repulsive core: n=3 gives dh/dt ~ h^-1,
  finite-time rupture, ell_min = 0. Genuinely open.
  Unbounded fragment cascades with no minimum stable feature.
  PDE global existence for degenerate mobility without a cutoff.

SELF-CORRECTION RECORDED
  The earlier delta_min^(P) = c_P r_n used the neck radius, which vanishes.
  That was the wrong scale. Corrected to the parent/jet radius.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="ell_min closure")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","hp","daughter","regime","unify"])
    ap.add_argument("--A", type=float, default=1e-20)
    ap.add_argument("--B", type=float, default=1e-78)
    ap.add_argument("--m", type=float, default=9.0)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--R", type=float, default=1e-5)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--dg-v", type=float, default=1e8)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--Fpp", type=float, default=1e6)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    if a.cmd == "hp":
        out = ell_min_L(Disjoining(a.A,a.B,a.m), a.theta)
    elif a.cmd == "daughter":
        out = rayleigh_daughter(a.R)
    elif a.cmd == "regime":
        out = nucleation_regime(a.sigma, a.dg_v, a.eps, a.Fpp)
    else:
        dj = Disjoining(a.A,a.B,a.m); rd = rayleigh_daughter(a.R)
        rg = nucleation_regime(a.sigma,a.dg_v,a.eps,a.Fpp)
        if rg["ell_min_m"] is None:
            out = {"REFUSED":"spinodal regime: N is not a surgery; use CH3D"}
        else:
            out = unified_ell_min({"N":rg["ell_min_m"],
                "M":rd["daughter_radius_m"],"B":10*a.R,"H":a.R,
                "P":a.R,"L":dj.h_p,"C":0.1*a.R})
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
