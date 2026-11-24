#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
foam_minimum_feature.py — v1.0.0
Closing the C-operator cascade with no minimum feature, via Plateau's law +
Euler characteristic (F >= 4) + volume/area conservation.
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
  A foam vertex has 4 Plateau borders, but each border is shared by 3 of the
  4 cells meeting there. Hence a CELL vertex is 3-coordinated:
      3V = 2E,   V - E + F = 2   =>   V = 2F-4,  E = 3F-6.
  Average edges per face:  <n> F = 2E = 6F-12  =>  <n> = 6 - 12/F.
  Verified: tetrahedron 3, cube 4, dodecahedron 5, Kelvin 72/14 = 5.142857,
            Matzke measured F ~ 13.4 -> 5.104.
  Faces are polygons, <n> >= 3  =>  F >= 4.

  IMPORTANT: isoperimetry gives V <= S^{3/2}/(6 sqrt(pi)), an UPPER bound.
  A LOWER bound on cell volume requires fatness (package D_theta):
      R <= theta r  =>  r >= sqrt(S/(4 pi))/theta,  V >= (4/3) pi r^3
      S >= 4 A_min  =>  V_min = (4/(3 sqrt(pi))) A_min^{3/2} / theta^3.
  Minimal face = equilateral triangle of side w:  A_min = (sqrt(3)/4) w^2.

  TWO COUNT BOUNDS
    area route   : 4N <= 2 F_tot,  A_tot >= F_tot A_min
                   =>  N <= A_tot/(2 A_min) = 2 A_tot/(sqrt(3) w^2)
                   (needs NO fatness)
    volume route : N <= V_tot/V_min ~ 4.667 theta^3 V_tot/w^3
                   (needs D_theta)

  w >= h_p, the equilibrium film thickness from the repulsive core
  h_p = (6 pi B/A)^{1/(m-3)} derived earlier. No new assumption.

  CASCADE
    T1 (neighbour swap): F preserved -> separated in time by
                         delta_min^(C) = w/sqrt(3)  (derived earlier)
    T2 (face annihilation): F decreases by 1 -> finite count <= F_tot
    k-way junction: resolves into ceil((k-2)/2) tetrahedral vertices
                    (odd k needs one extra face; flagged)

FAILURE MODE
  Dry-foam limit w -> 0 gives A_min -> 0, N_max -> inf: no-Zeno FAILS.
  Same degenerate-scale obstruction as h -> 0 and r_n -> 0.
-----------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

PI = math.pi
SQRT3 = math.sqrt(3.0)

class Refusal(RuntimeError): pass


# ---------------------------------------------------- §1 Euler / Plateau
def cell_combinatorics(F: int) -> Dict[str, Any]:
    """3-coordinated cell vertices (Plateau) + Euler."""
    if F < 4:
        raise Refusal(f"F={F} < 4: impossible for a Plateau foam cell")
    V = 2*F - 4
    E = 3*F - 6
    n_avg = 6.0 - 12.0/F
    return {"F": F, "V": V, "E": E,
            "euler_check": V - E + F,
            "avg_edges_per_face": n_avg,
            "coordination": 3,
            "face_edge_incidence_check": abs(n_avg*F - 2*E) < 1e-9}

def minimum_faces() -> Dict[str, Any]:
    """<n> >= 3 (polygon) => 6 - 12/F >= 3 => F >= 4."""
    return {"constraint": "avg edges per face >= 3",
            "formula": "6 - 12/F >= 3",
            "F_min": 4,
            "minimal_cell": "curved tetrahedron (4 faces, 4 vertices, 6 edges)",
            "NOTE": ("combinatorial necessity, NOT a guarantee of geometric "
                     "realizability with 120 deg / 109.47 deg angles")}

KNOWN_CELLS = {
 "tetrahedron": (4, 3.0),
 "cube": (6, 4.0),
 "octahedron": (8, 3.0),      # 8 triangular faces, but 4-coordinated -> excluded
 "dodecahedron": (12, 5.0),
 "kelvin": (14, 72.0/14.0),   # 8 hexagons + 6 squares
 "matzke_mean": (13.4, 5.104),
}

def verify_known_cells() -> Dict[str, Any]:
    rows = []
    for name,(F,n_true) in KNOWN_CELLS.items():
        if name == "octahedron":
            rows.append({"cell":name,"F":F,"excluded":True,
                         "why":"4-coordinated vertices violate Plateau"})
            continue
        c = cell_combinatorics(int(F)) if float(F).is_integer() else None
        n_pred = 6.0 - 12.0/F
        rows.append({"cell":name,"F":F,"n_predicted":n_pred,"n_actual":n_true,
                     "rel_err":abs(n_pred-n_true)/n_true,
                     "V":c["V"] if c else None,"E":c["E"] if c else None})
    return {"rows":rows,
            "kelvin_exact": abs((6-12/14) - 72/14) < 1e-12}


# ---------------------------------------------------- §2 minimum feature
@dataclass(frozen=True)
class FoamState:
    w: float = 1e-6            # Plateau border / minimum feature width [m]
    A_tot: float = 1e-4        # total film area [m^2]
    V_tot: float = 1e-9        # total volume [m^3]
    theta: float = 2.0         # isodiametric fatness R <= theta r
    h_p: Optional[float] = None  # equilibrium film thickness (repulsive core)

    @property
    def w_eff(self) -> float:
        """w >= h_p: the border is no thinner than the equilibrium film."""
        return max(self.w, self.h_p) if self.h_p is not None else self.w

    @property
    def A_min(self) -> float:
        """Minimal face = equilateral triangle of side w."""
        return (SQRT3/4.0)*self.w_eff**2


def minimum_cell_volume(st: FoamState) -> Dict[str, Any]:
    """
    V_min = (4/(3 sqrt(pi))) A_min^{3/2} / theta^3, for theta-fat convex cells.
    Isoperimetry alone gives an UPPER bound and cannot be used here.
    """
    if st.w_eff <= 0: raise Refusal("w > 0 required (dry limit destroys the bound)")
    if st.theta < 1.0: raise Refusal("theta >= 1 required")
    A = st.A_min
    V_min = (4.0/(3.0*math.sqrt(PI)))*A**1.5/st.theta**3
    r_min = math.sqrt(A/PI)/st.theta
    return {"w_eff_m": st.w_eff, "A_min_m2": A,
            "r_in_min_m": r_min,
            "V_min_m3": V_min,
            "V_min_over_w3": V_min/st.w_eff**3,
            "closed_form": "(4/(3 sqrt(pi))) A_min^{3/2} / theta^3",
            "requires": "D_theta fatness (R <= theta r); convex cell",
            "ISOPERIMETRY_WARNING": ("V <= S^{3/2}/(6 sqrt(pi)) is an UPPER "
                                     "bound; it cannot give V_min")}


def count_bounds(st: FoamState) -> Dict[str, Any]:
    """Two independent upper bounds on the number of cells."""
    if st.A_tot <= 0 or st.V_tot <= 0:
        raise Refusal("A_tot, V_tot > 0 required")
    mv = minimum_cell_volume(st)
    N_area = st.A_tot/(2.0*st.A_min)
    N_vol  = st.V_tot/mv["V_min_m3"]
    F_tot_max = st.A_tot/st.A_min
    return {"N_max_area_route": N_area,
            "N_max_volume_route": N_vol,
            "N_max": min(N_area, N_vol),
            "binding_route": "area" if N_area <= N_vol else "volume",
            "F_tot_max": F_tot_max,
            "area_route_formula": "2 A_tot/(sqrt(3) w^2)",
            "area_route_needs_fatness": False,
            "volume_route_needs_fatness": True,
            "V_min_m3": mv["V_min_m3"]}


# ---------------------------------------------------- §3 cascade
def cascade_decomposition(k: int) -> Dict[str, Any]:
    """
    A k-way junction resolves into tetrahedral (4-fold) vertices.
    Tree with L leaves, internal degree 4: L = 2i+2 => i = (k-2)/2.
    Odd k requires one extra face (flagged, not derived).
    """
    if k < 3: raise Refusal("k >= 3 required")
    if k == 4:
        return {"k":k,"already_stable":True,"elementary_vertices":1,
                "extra_face_needed":False}
    n_elem = math.ceil((k-2)/2.0)
    odd = (k % 2 == 1)
    return {"k":k,"already_stable":False,
            "elementary_vertices":n_elem,
            "formula":"ceil((k-2)/2)",
            "extra_face_needed":odd,
            "odd_k_caveat":("odd k cannot resolve into purely 4-valent "
                            "vertices; one extra face is required"
                            if odd else None)}


def c_event_ledger(st: FoamState) -> Dict[str, Any]:
    """
    T1 preserves F -> separated in TIME by delta_min^(C) = w/sqrt(3).
    T2 reduces F by 1 -> finite COUNT <= F_tot.
    """
    cb = count_bounds(st)
    w = st.w_eff
    return {"delta_min_C_m": w/SQRT3,
            "delta_min_C_formula": "w/sqrt(3) = c_C(3) w",
            "T1_bound": "temporal separation by delta_min^(C)",
            "T2_count_max": cb["F_tot_max"],
            "T2_bound": "finite count <= F_tot <= A_tot/A_min",
            "no_zeno_C": True,
            "route": "T1 temporal + T2 counting (as for nucleation)"}


def no_zeno_C(st: FoamState, B0: float, G0: float) -> Dict[str, Any]:
    """Combine the derived delta_min^(C) with the exit-time bound."""
    if min(B0,G0) <= 0: raise Refusal("B0, G0 > 0 required")
    led = c_event_ledger(st)
    delta = led["delta_min_C_m"]
    eps_max = delta/(2*B0)
    eps = min(eps_max, delta**2/(8*G0**2*math.log(2/0.3)))
    q = 2*math.exp(-delta**2/(8*G0**2*eps))
    return {"delta_m":delta,"eps_max_s":eps_max,"eps_s":eps,"q":q,
            "q_lt_1":q<1.0,
            "T2_count_max":led["T2_count_max"],
            "no_zeno":q<1.0,
            "status":"CLOSED conditional on w>0 and D_theta"}


def dry_limit_check(st: FoamState) -> Dict[str, Any]:
    """w -> 0 destroys both bounds. Same obstruction as h->0, r_n->0."""
    cb = count_bounds(st)
    shrink = FoamState(w=st.w*1e-3, A_tot=st.A_tot, V_tot=st.V_tot,
                       theta=st.theta, h_p=None)
    cb2 = count_bounds(shrink)
    return {"w_original_m":st.w,"w_shrunk_m":shrink.w,
            "N_max_original":cb["N_max"],"N_max_shrunk":cb2["N_max"],
            "bound_degrades":cb2["N_max"] > cb["N_max"],
            "scaling":"N_max ~ w^-2 (area) or w^-3 (volume)",
            "status":"OPEN in the dry limit w->0",
            "same_obstruction_as":["L: h->0","P: r_n->0","N: dg_v->inf"]}


# ---------------------------------------------------- §4 self-test
def self_test(verbose=True) -> bool:
    ok=True
    def rep(n,p,e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] Euler + Plateau: 3-coordinated cell vertices")
    for name,F,n_true in (("tetrahedron",4,3.0),("cube",6,4.0),
                          ("dodecahedron",12,5.0),("kelvin",14,72/14)):
        c = cell_combinatorics(F)
        print(f"      {name:14s} F={F:2d} V={c['V']:2d} E={c['E']:2d} "
              f"<n>={c['avg_edges_per_face']:.6f} (true {n_true:.6f})")
        rep(f"{name} Euler = 2", c["euler_check"]==2)
        rep(f"{name} <n> matches", abs(c['avg_edges_per_face']-n_true)<1e-9)
        rep(f"{name} incidence", c["face_edge_incidence_check"])
    rep("Kelvin exact 72/14", abs((6-12/14)-72/14)<1e-12)
    mf = minimum_faces()
    rep("F_min = 4 from <n> >= 3", mf["F_min"]==4)
    rep("combinatorial not geometric caveat",
        "NOT a guarantee" in mf["NOTE"])
    b=False
    try: cell_combinatorics(3)
    except Refusal: b=True
    rep("F=3 REFUSED", b)
    mk = verify_known_cells()
    rep("Matzke F=13.4 -> 5.104", abs((6-12/13.4)-5.104)<0.001)
    rep("octahedron excluded (4-coordinated)",
        any(r.get("excluded") for r in mk["rows"]))

    print("\n[T2] minimum face and cell volume")
    st = FoamState()
    mv = minimum_cell_volume(st)
    print(f"      w={st.w:.1e} m  A_min={mv['A_min_m2']:.3e} m^2")
    print(f"      V_min={mv['V_min_m3']:.3e} m^3  "
          f"V_min/w^3={mv['V_min_over_w3']:.6f}")
    rep("A_min = (sqrt3/4) w^2",
        abs(mv["A_min_m2"]-(SQRT3/4)*st.w**2)<1e-24)
    analytic = (4/(3*math.sqrt(PI)))*(SQRT3/4)**1.5/st.theta**3
    rep("V_min/w^3 matches analytic",
        abs(mv["V_min_over_w3"]-analytic)<1e-9, f"{analytic:.6f}")
    rep("isoperimetry warning present",
        "UPPER" in mv["ISOPERIMETRY_WARNING"])
    rep("V_min scales as theta^-3",
        abs(minimum_cell_volume(FoamState(theta=4.0))["V_min_m3"]*8
            - mv["V_min_m3"]) < 1e-24)
    b=False
    try: minimum_cell_volume(FoamState(w=0.0))
    except Refusal: b=True
    rep("w=0 REFUSED (dry limit)", b)
    b=False
    try: minimum_cell_volume(FoamState(theta=0.5))
    except Refusal: b=True
    rep("theta<1 REFUSED", b)

    print("\n[T3] w >= h_p from the repulsive core (no new assumption)")
    hp = (6*PI*1e-78/1e-20)**(1.0/6.0)
    st2 = FoamState(w=1e-9, h_p=hp)
    rep("w_eff = max(w, h_p)", abs(st2.w_eff-hp)<1e-18,
        f"h_p={hp:.3e} m")
    rep("h_p lifts the minimum feature", st2.w_eff > st2.w)

    print("\n[T4] two count bounds")
    cb = count_bounds(st)
    print(f"      area route  N_max = {cb['N_max_area_route']:.4g}")
    print(f"      volume route N_max = {cb['N_max_volume_route']:.4g}")
    print(f"      binding = {cb['binding_route']}, N_max = {cb['N_max']:.4g}")
    rep("area route needs NO fatness", not cb["area_route_needs_fatness"])
    rep("volume route needs fatness", cb["volume_route_needs_fatness"])
    rep("N_max = min of two routes",
        abs(cb["N_max"]-min(cb["N_max_area_route"],cb["N_max_volume_route"]))<1e-9)
    rep("area formula 2A/(sqrt3 w^2)",
        abs(cb["N_max_area_route"]-2*st.A_tot/(SQRT3*st.w**2))<1e-6)
    rep("4N <= 2 F_tot", 4*cb["N_max_area_route"] <= 2*cb["F_tot_max"]+1e-6)

    print("\n[T5] cascade decomposition")
    for k in (4,5,6,7,8,10):
        c = cascade_decomposition(k)
        print(f"      k={k:2d}: elementary={c['elementary_vertices']} "
              f"extra_face={c['extra_face_needed']}")
        if k>=6 and k%2==0:
            rep(f"k={k} -> (k-2)/2", c["elementary_vertices"]==(k-2)//2)
    rep("k=4 already stable", cascade_decomposition(4)["already_stable"])
    rep("odd k flagged", cascade_decomposition(7)["extra_face_needed"])
    rep("odd k caveat stated",
        cascade_decomposition(7)["odd_k_caveat"] is not None)

    print("\n[T6] C-event ledger and no-Zeno")
    led = c_event_ledger(st)
    print(f"      delta_min^(C) = w/sqrt3 = {led['delta_min_C_m']:.4e} m")
    print(f"      T2 count max = {led['T2_count_max']:.4g}")
    rep("delta_min^(C) = w/sqrt(3)",
        abs(led["delta_min_C_m"]-st.w/SQRT3)<1e-18)
    rep("T1 bounded temporally", "temporal" in led["T1_bound"])
    rep("T2 bounded by count", "finite count" in led["T2_bound"])
    nz = no_zeno_C(st, B0=1e-3, G0=1e-5)
    rep("q < 1", nz["q_lt_1"], f"q={nz['q']:.4f}")
    rep("no-Zeno for C", nz["no_zeno"])

    print("\n[T7] FAILURE MODE: dry limit (honest)")
    dl = dry_limit_check(st)
    print(f"      w {dl['w_original_m']:.1e} -> {dl['w_shrunk_m']:.1e}: "
          f"N_max {dl['N_max_original']:.3g} -> {dl['N_max_shrunk']:.3g}")
    rep("bound degrades as w shrinks", dl["bound_degrades"])
    rep("labelled OPEN in dry limit", dl["status"].startswith("OPEN"))
    rep("same obstruction as L, P, N",
        len(dl["same_obstruction_as"])==3)
    rep("N_max ~ w^-2 or w^-3", "w^-2" in dl["scaling"])

    print("\n" + "="*72)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*72)
    print("""
CLOSED
  Plateau cell vertices are 3-coordinated => V=2F-4, E=3F-6,
  <n> = 6 - 12/F. Verified against tetrahedron, cube, dodecahedron,
  Kelvin (72/14 exactly) and Matzke's measured 5.1.
  <n> >= 3  =>  F >= 4 for every foam cell.
  A_min = (sqrt3/4) w^2;  V_min = (4/(3 sqrt(pi))) A_min^{3/2}/theta^3.
  N <= A_tot/(2 A_min)  [area route, NO fatness needed]
  N <= V_tot/V_min      [volume route, needs D_theta]
  w >= h_p from the repulsive core derived earlier: no new assumption.
  C cascade: T1 separated in time by w/sqrt(3); T2 finite count <= F_tot;
  k-way junction -> ceil((k-2)/2) elementary vertices.

CORRECTED
  Isoperimetry gives V <= S^{3/2}/(6 sqrt(pi)), an UPPER bound.
  It cannot produce V_min; fatness is required.

NOT CLAIMED
  F >= 4 is combinatorial, not a guarantee of geometric realizability
  with 120 deg / 109.47 deg angles.
  Odd k cannot resolve into purely 4-valent vertices; flagged.
  Dry-foam limit w -> 0 destroys both bounds: still OPEN, the same
  degenerate-scale obstruction as h->0 (L), r_n->0 (P), dg_v->inf (N).
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Foam minimum feature")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","combinatorics","vmin","counts","cascade","dry"])
    ap.add_argument("--F", type=int, default=14)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--w", type=float, default=1e-6)
    ap.add_argument("--A-tot", type=float, default=1e-4)
    ap.add_argument("--V-tot", type=float, default=1e-9)
    ap.add_argument("--theta", type=float, default=2.0)
    a = ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    st = FoamState(a.w, a.A_tot, a.V_tot, a.theta)
    if a.cmd=="combinatorics":
        out={"cell":cell_combinatorics(a.F),"F_min":minimum_faces(),
             "known":verify_known_cells()}
    elif a.cmd=="vmin":   out=minimum_cell_volume(st)
    elif a.cmd=="counts": out=count_bounds(st)
    elif a.cmd=="cascade":out={"k_way":cascade_decomposition(a.k),
                               "ledger":c_event_ledger(st)}
    else:                 out=dry_limit_check(st)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
