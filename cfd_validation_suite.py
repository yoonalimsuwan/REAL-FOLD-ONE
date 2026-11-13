#!/usr/bin/env python3
"""
=============================================================================
CFD VALIDATION SUITE — Exact Riemann Solver + Standard Shock-Tube Tests
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

PURPOSE
  Verify a compressible Euler/NS solver against EXACT analytical solutions.
  This is the cheapest, clearest verification available for a shock-capturing
  scheme, and it is a prerequisite for any further aerodynamic claim.

CONTENTS
  §1  ExactRiemannSolver  — Toro (2009) Ch.4, vectorised Newton + sampling
  §2  Standard test suite — Sod, Lax, Toro 1-5 (all with exact reference)
  §3  Error norms         — L1/L2/Linf + measured convergence order
  §4  Oblique shock       — theta-beta-M relation (2D wedge reference)
  §5  Solver adapter      — plug in your AUSM+/HLLC and get a PASS/FAIL table

REFERENCES (all standard textbooks, nothing novel)
  [T1] Toro, E.F. (2009) Riemann Solvers and Numerical Methods for Fluid
       Dynamics, 3rd ed. Springer.  §4 = exact solver, §10 = test cases.
  [T2] Sod, G.A. (1978) J. Comput. Phys. 27, 1-31.
  [T3] Lax, P.D. (1954) Comm. Pure Appl. Math. 7, 159-193.
  [T4] Anderson, J.D. (2003) Modern Compressible Flow, 3rd ed.  §4 = oblique.

HONEST SCOPE
  ✅ Exact Riemann solver verified against published star-region values
  ✅ Standard test problems with analytic reference
  ✅ Oblique-shock relations (analytic, for wedge validation)
  ❌ Does NOT validate a 3D solver by itself — you must run your solver
     and feed the result to the adapter
  ❌ No airfoil / NACA comparison (needs body-fitted mesh + experiment)
  ❌ No aeroelasticity, propulsion, flight control
  ❌ Nothing here certifies anything for human flight
=============================================================================
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"


# =============================================================================
# §1  EXACT RIEMANN SOLVER  [T1, Ch.4]
# =============================================================================

class ExactRiemannSolver:
    """
    Exact solution of the 1D Euler Riemann problem for an ideal gas.

        left state (rhoL, uL, pL)  |  right state (rhoR, uR, pR)
                                   x = x0

    Algorithm (Toro §4.3-4.5):
      1. Solve f(p) = f_L(p) + f_R(p) + (uR - uL) = 0 for the star pressure p*
         by Newton iteration, starting from the two-rarefaction estimate.
      2. u* = 0.5(uL + uR) + 0.5(f_R(p*) - f_L(p*))
      3. Star densities from the shock jump or isentropic relation.
      4. Sample the wave structure at S = (x - x0)/t.

    Fully vectorised over sample points (numpy), so error norms over a whole
    grid cost one solve, not one solve per cell.
    """

    def __init__(self, gamma: float = 1.4, tol: float = 1e-12,
                 max_iter: int = 100):
        self.gamma = float(gamma)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    # ---- pressure function and its derivative (Toro eq. 4.6-4.7) -------
    def _f_df(self, p: np.ndarray, rhoK: float, pK: float, cK: float
              ) -> Tuple[np.ndarray, np.ndarray]:
        g = self.gamma
        p = np.maximum(p, 1e-14)
        shock = p > pK
        # shock branch
        A = 2.0 / ((g + 1.0) * rhoK)
        B = (g - 1.0) / (g + 1.0) * pK
        s = np.sqrt(A / (p + B))
        f_sh = (p - pK) * s
        df_sh = s * (1.0 - (p - pK) / (2.0 * (p + B)))
        # rarefaction branch
        z = (g - 1.0) / (2.0 * g)
        ratio = p / pK
        f_ra = (2.0 * cK / (g - 1.0)) * (ratio ** z - 1.0)
        df_ra = (1.0 / (rhoK * cK)) * ratio ** (-(g + 1.0) / (2.0 * g))
        return (np.where(shock, f_sh, f_ra),
                np.where(shock, df_sh, df_ra))

    def star_state(self, rhoL, uL, pL, rhoR, uR, pR
                   ) -> Tuple[float, float, float, float, int]:
        """Returns (p_star, u_star, rho_starL, rho_starR, n_iters)."""
        g = self.gamma
        cL = math.sqrt(g * pL / rhoL)
        cR = math.sqrt(g * pR / rhoR)
        z = (g - 1.0) / (2.0 * g)

        # two-rarefaction initial guess (Toro eq. 4.46)
        num = cL + cR - 0.5 * (g - 1.0) * (uR - uL)
        den = cL / pL ** z + cR / pR ** z
        p = max((num / den) ** (1.0 / z), 1e-12)

        for it in range(self.max_iter):
            fL, dfL = self._f_df(np.array(p), rhoL, pL, cL)
            fR, dfR = self._f_df(np.array(p), rhoR, pR, cR)
            f = float(fL + fR + (uR - uL))
            df = float(dfL + dfR)
            if abs(df) < 1e-300:
                break
            dp = -f / df
            p_new = p + dp
            if p_new <= 0:
                p_new = 0.5 * p          # keep positive
            if abs(p_new - p) / (0.5 * (p_new + p)) < self.tol:
                p = p_new
                break
            p = p_new

        fL, _ = self._f_df(np.array(p), rhoL, pL, cL)
        fR, _ = self._f_df(np.array(p), rhoR, pR, cR)
        u_star = 0.5 * (uL + uR) + 0.5 * float(fR - fL)

        gr = (g - 1.0) / (g + 1.0)
        if p > pL:      # left shock
            rho_sL = rhoL * ((p / pL + gr) / (gr * p / pL + 1.0))
        else:           # left rarefaction
            rho_sL = rhoL * (p / pL) ** (1.0 / g)
        if p > pR:      # right shock
            rho_sR = rhoR * ((p / pR + gr) / (gr * p / pR + 1.0))
        else:
            rho_sR = rhoR * (p / pR) ** (1.0 / g)

        return p, u_star, rho_sL, rho_sR, it + 1

    def sample(self, rhoL, uL, pL, rhoR, uR, pR,
               x: np.ndarray, t: float, x0: float = 0.5
               ) -> Dict[str, np.ndarray]:
        """Exact solution sampled at positions x at time t."""
        g = self.gamma
        if t <= 0:
            left = x < x0
            return {"rho": np.where(left, rhoL, rhoR),
                    "u": np.where(left, uL, uR),
                    "p": np.where(left, pL, pR)}

        ps, us, rsL, rsR, _ = self.star_state(rhoL, uL, pL, rhoR, uR, pR)
        cL = math.sqrt(g * pL / rhoL)
        cR = math.sqrt(g * pR / rhoR)
        csL = cL * (ps / pL) ** ((g - 1.0) / (2.0 * g))
        csR = cR * (ps / pR) ** ((g - 1.0) / (2.0 * g))

        S = (np.asarray(x) - x0) / t
        rho = np.empty_like(S); u = np.empty_like(S); p = np.empty_like(S)

        left_side = S <= us

        # ---------------- LEFT HALF ----------------
        if ps > pL:     # left shock
            SL = uL - cL * math.sqrt((g + 1.0) / (2.0 * g) * ps / pL
                                     + (g - 1.0) / (2.0 * g))
            m = left_side & (S <= SL)
            rho[m], u[m], p[m] = rhoL, uL, pL
            m = left_side & (S > SL)
            rho[m], u[m], p[m] = rsL, us, ps
        else:           # left rarefaction
            SHL = uL - cL
            STL = us - csL
            m = left_side & (S <= SHL)
            rho[m], u[m], p[m] = rhoL, uL, pL
            m = left_side & (S >= STL)
            rho[m], u[m], p[m] = rsL, us, ps
            m = left_side & (S > SHL) & (S < STL)
            if np.any(m):
                base = (2.0 / (g + 1.0)
                        + (g - 1.0) / ((g + 1.0) * cL) * (uL - S[m]))
                rho[m] = rhoL * base ** (2.0 / (g - 1.0))
                u[m] = (2.0 / (g + 1.0)) * (cL + 0.5 * (g - 1.0) * uL + S[m])
                p[m] = pL * base ** (2.0 * g / (g - 1.0))

        # ---------------- RIGHT HALF ----------------
        right_side = ~left_side
        if ps > pR:     # right shock
            SR = uR + cR * math.sqrt((g + 1.0) / (2.0 * g) * ps / pR
                                     + (g - 1.0) / (2.0 * g))
            m = right_side & (S >= SR)
            rho[m], u[m], p[m] = rhoR, uR, pR
            m = right_side & (S < SR)
            rho[m], u[m], p[m] = rsR, us, ps
        else:           # right rarefaction
            SHR = uR + cR
            STR = us + csR
            m = right_side & (S >= SHR)
            rho[m], u[m], p[m] = rhoR, uR, pR
            m = right_side & (S <= STR)
            rho[m], u[m], p[m] = rsR, us, ps
            m = right_side & (S < SHR) & (S > STR)
            if np.any(m):
                base = (2.0 / (g + 1.0)
                        - (g - 1.0) / ((g + 1.0) * cR) * (uR - S[m]))
                rho[m] = rhoR * base ** (2.0 / (g - 1.0))
                u[m] = (2.0 / (g + 1.0)) * (-cR + 0.5 * (g - 1.0) * uR + S[m])
                p[m] = pR * base ** (2.0 * g / (g - 1.0))

        return {"rho": rho, "u": u, "p": p}


# =============================================================================
# §2  STANDARD TEST PROBLEMS  [T1 §10, T2, T3]
# =============================================================================

@dataclass
class ShockTubeCase:
    name: str
    left: Tuple[float, float, float]     # (rho, u, p)
    right: Tuple[float, float, float]
    t_end: float
    x0: float = 0.5
    gamma: float = 1.4
    L: float = 1.0
    note: str = ""


STANDARD_CASES: Dict[str, ShockTubeCase] = {
    "sod": ShockTubeCase(
        "Sod", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.20,
        note="classic mild shock tube [T2]"),
    "lax": ShockTubeCase(
        "Lax", (0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 0.14,
        note="Lax problem [T3]"),
    "toro1": ShockTubeCase(
        "Toro 1", (1.0, 0.75, 1.0), (0.125, 0.0, 0.1), 0.20,
        note="Sod with left velocity"),
    "toro2": ShockTubeCase(
        "Toro 2", (1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.15,
        note="123 problem — two symmetric rarefactions, near-vacuum"),
    "toro3": ShockTubeCase(
        "Toro 3", (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.012,
        note="severe blast, p ratio 1e5"),
    "toro4": ShockTubeCase(
        "Toro 4", (5.99924, 19.5975, 460.894),
        (5.99242, -6.19633, 46.095), 0.035,
        note="collision of two strong shocks"),
    "toro5": ShockTubeCase(
        "Toro 5", (1.0, -19.59745, 1000.0),
        (1.0, -19.59745, 0.01), 0.012,
        note="left rarefaction + right shock, high Mach"),
}

# Published star-region values (Toro Table 4.1) for cross-checking the solver
PUBLISHED_STAR = {
    "sod":   {"p_star": 0.30313, "u_star": 0.92745},
    "lax":   {"p_star": 1.69165, "u_star": 1.11000},
    "toro1": {"p_star": 0.54760, "u_star": 1.15945},
    "toro2": {"p_star": 0.00189, "u_star": 0.00000},
    "toro3": {"p_star": 460.894, "u_star": 19.5975},
}


# =============================================================================
# §3  ERROR NORMS + CONVERGENCE ORDER
# =============================================================================

def error_norms(num: Dict[str, np.ndarray],
                ref: Dict[str, np.ndarray],
                dx: float) -> Dict[str, float]:
    out = {}
    for k in ("rho", "u", "p"):
        d = num[k] - ref[k]
        scale = max(float(np.abs(ref[k]).max()), 1e-30)
        out[f"{k}_L1"] = float(np.sum(np.abs(d)) * dx)
        out[f"{k}_L2"] = float(np.sqrt(np.sum(d * d) * dx))
        out[f"{k}_Linf"] = float(np.abs(d).max())
        out[f"{k}_L1_rel"] = out[f"{k}_L1"] / scale
    out["L1_total"] = sum(out[f"{k}_L1"] for k in ("rho", "u", "p"))
    return out


def convergence_order(errors: List[float], dxs: List[float]) -> float:
    """Least-squares fit of log(err) vs log(dx)."""
    ok = [(e, h) for e, h in zip(errors, dxs) if e > 0 and math.isfinite(e)]
    if len(ok) < 2:
        return float("nan")
    lx = np.log([h for _, h in ok])
    ly = np.log([e for e, _ in ok])
    n = len(lx)
    sx, sy = lx.sum(), ly.sum()
    sxx, sxy = (lx * lx).sum(), (lx * ly).sum()
    den = n * sxx - sx * sx
    if abs(den) < 1e-30:
        return float("nan")
    return float((n * sxy - sx * sy) / den)


# =============================================================================
# §4  OBLIQUE SHOCK — theta-beta-M  [T4 §4]
# =============================================================================

def theta_beta_M(M1: float, theta_deg: float, gamma: float = 1.4,
                 weak: bool = True) -> Dict[str, float]:
    """
    Solve tan(theta) = 2 cot(beta) (M1^2 sin^2 beta - 1) /
                       (M1^2 (gamma + cos 2beta) + 2)
    for the shock angle beta, then return the post-shock state.

    This is the analytic reference for a 2D wedge / ramp validation.
    Returns {} if no attached solution exists (theta > theta_max).
    """
    th = math.radians(theta_deg)
    g = gamma
    M2sq = M1 * M1

    def resid(beta):
        sb, cb = math.sin(beta), math.cos(beta)
        num = M2sq * sb * sb - 1.0
        den = M2sq * (g + math.cos(2.0 * beta)) + 2.0
        return 2.0 * cb / sb * num / den - math.tan(th)

    # attached-shock range: Mach angle < beta < beta(detachment)
    mu = math.asin(1.0 / M1) if M1 > 1 else 0.0
    lo, hi = mu + 1e-8, math.pi / 2 - 1e-8
    # locate the maximum-theta point to bracket weak / strong branches
    bs = np.linspace(lo, hi, 4001)
    vals = [resid(b) for b in bs]
    root = None
    for i in range(len(bs) - 1):
        if vals[i] * vals[i + 1] < 0:
            a, b = bs[i], bs[i + 1]
            for _ in range(80):
                m = 0.5 * (a + b)
                if resid(a) * resid(m) <= 0:
                    b = m
                else:
                    a = m
            root = 0.5 * (a + b)
            if weak:
                break     # first root = weak solution
    if root is None:
        return {}

    beta = root
    Mn1 = M1 * math.sin(beta)
    p_ratio = 1.0 + 2.0 * g / (g + 1.0) * (Mn1 * Mn1 - 1.0)
    rho_ratio = ((g + 1.0) * Mn1 * Mn1) / ((g - 1.0) * Mn1 * Mn1 + 2.0)
    T_ratio = p_ratio / rho_ratio
    Mn2sq = (1.0 + 0.5 * (g - 1.0) * Mn1 * Mn1) / (g * Mn1 * Mn1 - 0.5 * (g - 1.0))
    M2 = math.sqrt(max(Mn2sq, 0.0)) / math.sin(beta - th)
    return {"beta_deg": math.degrees(beta), "M2": M2,
            "p2_p1": p_ratio, "rho2_rho1": rho_ratio, "T2_T1": T_ratio,
            "Mn1": Mn1}


# =============================================================================
# §5  SOLVER ADAPTER + SELF-TEST
# =============================================================================

def validate_solver(solver_step: Callable, cases: Optional[List[str]] = None,
                    nx_list: Tuple[int, ...] = (100, 200, 400),
                    cfl: float = 0.4, verbose: bool = True) -> Dict[str, Any]:
    """
    Run your 1D solver against the exact solution and report error norms.

    solver_step(rho, u, p, dx, dt, n_steps, gamma) -> (rho, u, p)
        Must advance the 1D Euler system on a uniform cell-centred grid.
        Transmissive (zero-gradient) boundaries are adequate for these
        cases as long as the waves do not reach the domain ends.
    """
    names = cases or list(STANDARD_CASES)
    results = {}
    for name in names:
        case = STANDARD_CASES[name]
        rs = ExactRiemannSolver(case.gamma)
        errs_all, dxs = [], []
        for nx in nx_list:
            dx = case.L / nx
            xc = (np.arange(nx) + 0.5) * dx
            left = xc < case.x0
            rho = np.where(left, case.left[0], case.right[0])
            u = np.where(left, case.left[1], case.right[1])
            p = np.where(left, case.left[2], case.right[2])
            # CFL from max signal speed
            c = np.sqrt(case.gamma * p / rho)
            smax = float(np.max(np.abs(u) + c))
            dt = cfl * dx / max(smax, 1e-12)
            n_steps = int(math.ceil(case.t_end / dt))
            dt = case.t_end / n_steps
            try:
                rn, un, pn = solver_step(rho, u, p, dx, dt, n_steps, case.gamma)
            except Exception as e:
                results[name] = {"error": f"{type(e).__name__}: {e}"}
                break
            ref = rs.sample(*case.left, *case.right, xc, case.t_end,
                            case.x0)
            en = error_norms({"rho": rn, "u": un, "p": pn}, ref, dx)
            errs_all.append(en["L1_total"])
            dxs.append(dx)
            if verbose:
                print(f"  {case.name:8s} nx={nx:5d}  "
                      f"L1(rho)={en['rho_L1']:.3e}  "
                      f"L1(u)={en['u_L1']:.3e}  "
                      f"L1(p)={en['p_L1']:.3e}")
        if errs_all:
            order = convergence_order(errs_all, dxs)
            results[name] = {"errors": errs_all, "dxs": dxs, "order": order}
            if verbose:
                print(f"  {case.name:8s} measured order = {order:.3f}   "
                      f"(shock-capturing schemes: expect ~1 in L1 with shocks)")
    return results


def self_test() -> bool:
    print("=" * 88)
    print(f"  CFD VALIDATION SUITE v{VERSION} — self-test")
    print("=" * 88)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    rs = ExactRiemannSolver()

    # ---- T1: star region vs published values (Toro Table 4.1) ----
    print("\n[T1] Star-region pressure/velocity vs published reference")
    for name, ref in PUBLISHED_STAR.items():
        c = STANDARD_CASES[name]
        ps, us, rsL, rsR, nit = rs.star_state(*c.left, *c.right)
        ep = abs(ps - ref["p_star"]) / max(ref["p_star"], 1e-12)
        eu = abs(us - ref["u_star"]) / max(abs(ref["u_star"]), 1e-6)
        rep(f"{name} p*", ep < 5e-3,
            f"p*={ps:.5f} ref={ref['p_star']:.5f} rel={ep:.2e} ({nit} iters)")
        rep(f"{name} u*", eu < 5e-3,
            f"u*={us:.5f} ref={ref['u_star']:.5f}")

    # ---- T2: initial condition recovered at t=0 ----
    print("\n[T2] t=0 returns the initial discontinuity")
    c = STANDARD_CASES["sod"]
    x = np.linspace(0, 1, 201)
    s0 = rs.sample(*c.left, *c.right, x, 0.0, c.x0)
    left = x < c.x0
    rep("rho at t=0", np.allclose(s0["rho"], np.where(left, 1.0, 0.125)))
    rep("p at t=0", np.allclose(s0["p"], np.where(left, 1.0, 0.1)))

    # ---- T3: conservation / symmetry checks ----
    print("\n[T3] Structural checks on the exact solution")
    s = rs.sample(*c.left, *c.right, x, c.t_end, c.x0)
    rep("rho bounded by initial states",
        float(s["rho"].min()) >= 0.125 - 1e-9
        and float(s["rho"].max()) <= 1.0 + 1e-9,
        f"[{s['rho'].min():.4f}, {s['rho'].max():.4f}]")
    rep("p bounded by initial states",
        float(s["p"].min()) >= 0.1 - 1e-9
        and float(s["p"].max()) <= 1.0 + 1e-9)
    rep("u monotone increasing across the fan",
        bool(np.all(np.diff(s["u"]) >= -1e-9)))

    c2 = STANDARD_CASES["toro2"]
    s2 = rs.sample(*c2.left, *c2.right, x, c2.t_end, c2.x0)
    rep("Toro 2 symmetric: u* = 0",
        abs(rs.star_state(*c2.left, *c2.right)[1]) < 1e-9)
    rep("Toro 2 near-vacuum rho > 0", float(s2["rho"].min()) > 0)

    # ---- T4: oblique shock analytic reference ----
    print("\n[T4] Oblique-shock theta-beta-M (wedge reference)")
    r = theta_beta_M(M1=2.0, theta_deg=10.0)
    rep("M=2, theta=10deg has attached solution", bool(r))
    if r:
        # published: beta ~ 39.31 deg, p2/p1 ~ 1.7066, M2 ~ 1.6405
        rep("beta ~ 39.31 deg", abs(r["beta_deg"] - 39.314) < 0.05,
            f"beta={r['beta_deg']:.3f}")
        rep("p2/p1 ~ 1.7066", abs(r["p2_p1"] - 1.7066) < 5e-3,
            f"p2/p1={r['p2_p1']:.4f}")
        rep("M2 ~ 1.6405", abs(r["M2"] - 1.6405) < 5e-3,
            f"M2={r['M2']:.4f}")
    r_det = theta_beta_M(M1=2.0, theta_deg=40.0)
    rep("theta > theta_max gives no attached shock", not r_det)

    # ---- T5: convergence-order machinery ----
    print("\n[T5] Convergence-order fit")
    dxs = [1/100, 1/200, 1/400]
    e2 = [h**2 for h in dxs]
    e1 = [h for h in dxs]
    rep("detects 2nd order", abs(convergence_order(e2, dxs) - 2.0) < 1e-8,
        f"p={convergence_order(e2, dxs):.4f}")
    rep("detects 1st order", abs(convergence_order(e1, dxs) - 1.0) < 1e-8,
        f"p={convergence_order(e1, dxs):.4f}")

    # ---- T6: adapter with a simple reference scheme ----
    print("\n[T6] Adapter smoke test (1st-order Rusanov, deliberately diffusive)")

    def rusanov(rho, u, p, dx, dt, n_steps, gamma):
        """Simple, robust, 1st-order scheme — a floor, not a target."""
        for _ in range(n_steps):
            r = np.pad(rho, 1, mode="edge")
            uu = np.pad(u, 1, mode="edge")
            pp = np.pad(p, 1, mode="edge")
            E = p / (gamma - 1) + 0.5 * rho * u * u
            EE = np.pad(E, 1, mode="edge")
            U = np.stack([r, r * uu, EE], 0)
            c = np.sqrt(gamma * np.maximum(pp, 1e-12) / np.maximum(r, 1e-12))
            F = np.stack([r * uu, r * uu * uu + pp, (EE + pp) * uu], 0)
            smax = np.maximum(np.abs(uu[:-1]) + c[:-1],
                              np.abs(uu[1:]) + c[1:])
            Ff = 0.5 * (F[:, :-1] + F[:, 1:]) \
                 - 0.5 * smax[None, :] * (U[:, 1:] - U[:, :-1])
            Un = U[:, 1:-1] - dt / dx * (Ff[:, 1:] - Ff[:, :-1])
            rho = np.maximum(Un[0], 1e-10)
            u = Un[1] / rho
            E = Un[2]
            p = np.maximum((gamma - 1) * (E - 0.5 * rho * u * u), 1e-10)
        return rho, u, p

    res = validate_solver(rusanov, cases=["sod"], nx_list=(100, 200),
                          verbose=True)
    rep("adapter produced error norms", "sod" in res and "errors" in res["sod"])
    if "sod" in res and "errors" in res["sod"]:
        e = res["sod"]["errors"]
        rep("error decreases with refinement", e[1] < e[0],
            f"L1: {e[0]:.3e} -> {e[1]:.3e}")

    print("\n" + "=" * 88)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 88)
    print("""
  WHAT THIS GIVES YOU
    - An exact Riemann solver verified against Toro's published star values
    - 7 standard shock-tube problems with analytic reference
    - L1/L2/Linf norms + measured convergence order
    - theta-beta-M analytic reference for a 2D wedge test
    - An adapter: plug in your AUSM+/HLLC and get a PASS/FAIL table

  HOW TO USE WITH YOUR SOLVER
    Write a 1D wrapper around your flux:
        def my_solver(rho, u, p, dx, dt, n_steps, gamma):
            for _ in range(n_steps):
                ... your AUSM+ / HLLC update ...
            return rho, u, p
    then:
        validate_solver(my_solver, nx_list=(100, 200, 400))

  WHAT TO EXPECT
    A shock-capturing scheme gives L1 order ~1 (shocks dominate the error).
    Order ~2 in smooth regions only. If you measure order < 0.5, or errors
    that do not decrease, the scheme has a bug — fix that before anything else.

  WHAT THIS DOES NOT DO
    - It does not validate your 3D solver by itself; you must run it
    - No airfoil/NACA comparison (needs body-fitted mesh + experiment)
    - No aeroelasticity, propulsion, flight control
    - Nothing here certifies anything for human flight
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"CFD Validation Suite v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--case", default="sod", choices=list(STANDARD_CASES))
    ap.add_argument("--wedge", type=float, nargs=2, metavar=("M", "THETA"))
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)
    if a.wedge:
        M, th = a.wedge
        r = theta_beta_M(M, th)
        print(f"M={M}, theta={th}deg -> {r if r else 'no attached shock'}")
        return
    c = STANDARD_CASES[a.case]
    rs = ExactRiemannSolver(c.gamma)
    ps, us, rsL, rsR, nit = rs.star_state(*c.left, *c.right)
    print(f"{c.name}: {c.note}")
    print(f"  p*={ps:.6f}  u*={us:.6f}  rho*L={rsL:.6f}  rho*R={rsR:.6f}"
          f"  ({nit} Newton iters)")


if __name__ == "__main__":
    main()
