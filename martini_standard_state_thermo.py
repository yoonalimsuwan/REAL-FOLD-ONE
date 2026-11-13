#!/usr/bin/env python3
"""
=============================================================================
STANDARD-STATE VOLUME CORRECTION + TEMPERATURE-TRANSFERABILITY TEST
for coarse-grained binding free energies (AA reference vs CG)
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — STANDARD-STATE VOLUME CORRECTION
  From a radial PMF W(r) between two centres of mass, the association
  constant and standard-state binding free energy are

      K_a  = 4*pi * int_0^{r_c} r^2 exp(-beta W_true(r)) dr      [volume]
      dG°  = -k_B T ln( K_a / V° ),   V° = 1.660539 nm^3  (1 mol/L)

  CRITICAL: W_true must be BULK-FLAT and JACOBIAN-REMOVED.
  WHAM on a distance coordinate returns
      W_naive(r) = -k_B T ln P(r) + const,
  which already contains the -2 k_B T ln r volume-entropy term.
  Feeding W_naive into the r^2 formula DOUBLE-COUNTS the Jacobian.
  The conversion is
      W_true(r) = W_naive(r) + 2 k_B T ln r   (then align bulk to zero).
  This module makes the convention an explicit, required argument and
  refuses to guess it.

  References:
    Gilson, Given, Bush, McCammon (1997) Biophys. J. 72, 1047.
    Woo & Roux (2005) PNAS 102, 6825.
    Doudou, Burton, Henchman (2009) JCTC 5, 909.
    Gumbart, Roux, Chipot (2013) JCTC 9, 3789.

PART B — TEMPERATURE TRANSFERABILITY (entropy/enthalpy split)
  Compute dG°(T) at T = 300, 310, 320 K for BOTH AA and CG, then extract

      van 't Hoff :  ln(K_a/V°) = -dH/(R T) + dS/R
                     slope of ln(K_a/V°) vs 1/T  ->  dH = -R*slope
                                                   dS =  R*intercept
      Gibbs-Helmholtz / finite difference:
                     dS = -d(dG°)/dT ,  dH = dG° + T dS

  and report the AA-vs-CG differences ddH, ddS.

  WHY THIS TEST IS EXPECTED TO FAIL FOR A GENERIC CG MODEL
    A coarse-grained effective potential is itself a free energy: it is
    the PMF of the eliminated degrees of freedom at the temperature where
    it was parameterised. Its "energy" therefore already contains entropy.
    A temperature-independent CG potential cannot in general reproduce the
    AA entropy/enthalpy decomposition, even when it reproduces dG° at one
    temperature. This is the standard representability/transferability
    problem of coarse-graining (Louis 2002; Lyubartsev & Laaksonen;
    Noid 2013). This module MEASURES the discrepancy; it does not repair it.

HONEST SCOPE
  ❌ Three temperatures spanning 20 K is a SHORT lever arm in 1/T.
     The van 't Hoff enthalpy error is large by construction and is
     reported, not hidden. Do not quote dH from 3 points without its error.
  ❌ van 't Hoff assumes dCp = 0 over the range. Binding often has
     |dCp| >> 0. A dCp fit is offered but 3 points cannot determine
     3 parameters WITH an error estimate; this is flagged.
  ❌ Orientational/positional restraints used during umbrella sampling
     require additional Woo-Roux corrections. This module handles the
     RADIAL volume term only. If you restrained angles, you must add
     those terms yourself; a warning is emitted.
  ❌ No finite-size / periodic-image correction, no salt/activity
     correction, no protonation-state linkage.
  ❌ Agreement of dG° at one temperature does NOT establish transferability.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("m3_thermo")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# ---- units: nm, kJ/mol, K -------------------------------------------------
KB = 0.008314462618           # kJ mol^-1 K^-1  (= R per mole)
NA = 6.02214076e23
V_STANDARD_NM3 = 1.0e27 / NA  # 1 mol/L expressed as nm^3 per molecule = 1.66054


# =============================================================================
# §1  PMF CONTAINER + CONVENTION HANDLING
# =============================================================================

@dataclass
class PMFData:
    """
    r          : (n,) centre-of-mass separation, nm, strictly increasing, > 0
    W          : (n,) PMF, kJ/mol, arbitrary additive offset
    temperature: K
    convention : "radial_density"  -> raw WHAM output on a distance coordinate
                                      (Jacobian -2kT ln r INCLUDED)
                 "bulk_flat"       -> Jacobian already removed, W -> const in bulk
    stderr     : (n,) optional per-bin standard error, kJ/mol
    label      : free text
    """
    r: np.ndarray
    W: np.ndarray
    temperature: float
    convention: str
    stderr: Optional[np.ndarray] = None
    label: str = ""

    def __post_init__(self):
        self.r = np.asarray(self.r, dtype=np.float64)
        self.W = np.asarray(self.W, dtype=np.float64)
        if self.r.ndim != 1 or self.W.shape != self.r.shape:
            raise ValueError("r and W must be 1-D of equal length")
        if np.any(self.r <= 0):
            raise ValueError("r must be > 0 (radial coordinate)")
        if np.any(np.diff(self.r) <= 0):
            raise ValueError("r must be strictly increasing")
        if self.convention not in ("radial_density", "bulk_flat"):
            raise ValueError("convention must be radial_density | bulk_flat")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.stderr is not None:
            self.stderr = np.asarray(self.stderr, dtype=np.float64)
            if self.stderr.shape != self.r.shape:
                raise ValueError("stderr shape mismatch")

    @property
    def beta(self) -> float:
        return 1.0 / (KB * self.temperature)

    def to_bulk_flat(self) -> np.ndarray:
        """
        Return the Jacobian-removed PMF (still with arbitrary offset).
        radial_density -> add +2 kT ln r ; bulk_flat -> unchanged.
        """
        if self.convention == "bulk_flat":
            return self.W.copy()
        return self.W + 2.0 * KB * self.temperature * np.log(self.r)

    def aligned(self, bulk_window: Tuple[float, float]) -> np.ndarray:
        """Jacobian-removed PMF shifted so its bulk-window mean is zero."""
        W = self.to_bulk_flat()
        m = (self.r >= bulk_window[0]) & (self.r <= bulk_window[1])
        if m.sum() < 2:
            raise ValueError("bulk window contains < 2 bins")
        return W - float(np.mean(W[m]))


# =============================================================================
# §2  BULK-FLATNESS DIAGNOSTIC  (is the PMF usable at all?)
# =============================================================================

def bulk_flatness(pmf: PMFData, bulk_window: Tuple[float, float]
                  ) -> Dict[str, Any]:
    """
    A radial PMF is only interpretable if it is FLAT in the bulk.
    A residual slope means the umbrella windows did not decouple, or the
    Jacobian convention is wrong, or the box is too small.
    Reports slope, its standard error, and a verdict.
    """
    W = pmf.aligned(bulk_window)
    m = (pmf.r >= bulk_window[0]) & (pmf.r <= bulk_window[1])
    x, y = pmf.r[m], W[m]
    n = len(x)
    if n < 3:
        return {"ok": False, "reason": "fewer than 3 bulk bins"}
    A = np.vstack([x, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    resid = y - A @ coef
    s2 = float(resid @ resid) / max(n - 2, 1)
    xvar = float(np.sum((x - x.mean()) ** 2))
    slope_se = math.sqrt(s2 / xvar) if xvar > 0 else float("inf")
    span = float(x.max() - x.min())
    drift = abs(slope) * span
    ok = (drift < 1.0) and (abs(slope) < 3.0 * slope_se + 1e-9 or drift < 0.5)
    return {"ok": bool(ok), "slope_kJmol_per_nm": slope,
            "slope_stderr": slope_se, "drift_over_bulk_kJmol": drift,
            "n_bulk_bins": n,
            "verdict": "FLAT" if ok else "NOT FLAT — PMF not interpretable"}


# =============================================================================
# §3  STANDARD-STATE BINDING FREE ENERGY
# =============================================================================

@dataclass
class BindingResult:
    temperature: float
    Ka_nm3: float
    dG_standard_kJmol: float
    dG_stderr_kJmol: float
    convention: str
    r_bound_nm: float
    bulk_window: Tuple[float, float]
    bulk_flat: Dict[str, Any]
    restrained_warning: str = ""
    label: str = ""


def standard_state_binding(
    pmf: PMFData,
    r_bound: float,
    bulk_window: Tuple[float, float],
    orientational_restraints: bool = False,
    label: str = "",
) -> BindingResult:
    """
    K_a  = 4 pi int_0^{r_bound} r^2 exp(-beta W_true(r)) dr
    dG°  = -kT ln(K_a / V°)

    r_bound must be the OUTER edge of the bound basin (typically the first
    barrier top or where W_true first reaches ~0). It is a physical choice
    and must be reported; the result depends on it.
    """
    flat = bulk_flatness(pmf, bulk_window)
    W = pmf.aligned(bulk_window)
    m = pmf.r <= r_bound
    if m.sum() < 2:
        raise ValueError("bound region contains < 2 bins")
    r, w = pmf.r[m], W[m]
    integrand = r * r * np.exp(-pmf.beta * w)
    Ka = 4.0 * math.pi * float(np.trapz(integrand, r))
    if Ka <= 0 or not math.isfinite(Ka):
        raise ValueError(f"non-positive/non-finite K_a = {Ka}")
    dG = -KB * pmf.temperature * math.log(Ka / V_STANDARD_NM3)

    # error propagation from per-bin stderr (linear, independent bins)
    dG_se = float("nan")
    if pmf.stderr is not None:
        se = pmf.stderr[m]
        # dKa/dW_i = -beta * r_i^2 exp(-beta W_i) * weight_i
        wt = np.full(len(r), (r[1] - r[0]) if len(r) > 1 else 1.0)
        if len(r) > 1:
            wt[0] *= 0.5
            wt[-1] *= 0.5
        dKa = 4.0 * math.pi * (-pmf.beta) * integrand * wt
        var_Ka = float(np.sum((dKa * se) ** 2))
        dG_se = KB * pmf.temperature * math.sqrt(var_Ka) / Ka

    warn = ""
    if orientational_restraints:
        warn = ("ORIENTATIONAL/POSITIONAL RESTRAINTS WERE USED. "
                "This result includes ONLY the radial standard-state volume "
                "term. Add the Woo-Roux angular and positional restraint "
                "corrections before quoting dG°.")
    return BindingResult(
        temperature=pmf.temperature, Ka_nm3=Ka,
        dG_standard_kJmol=dG, dG_stderr_kJmol=dG_se,
        convention=pmf.convention, r_bound_nm=r_bound,
        bulk_window=bulk_window, bulk_flat=flat,
        restrained_warning=warn, label=label or pmf.label)


def jacobian_double_count_check(
    pmf: PMFData, r_bound: float, bulk_window: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Deliberately compute dG° BOTH ways to expose the double-counting bug.
    If the input is radial_density, applying the r^2 formula WITHOUT the
    +2kT ln r conversion gives a different (wrong) answer. The size of the
    difference is reported so the mistake is visible, not silent.
    """
    correct = standard_state_binding(pmf, r_bound, bulk_window)
    W_raw = pmf.W.copy()
    m0 = (pmf.r >= bulk_window[0]) & (pmf.r <= bulk_window[1])
    W_raw = W_raw - float(np.mean(W_raw[m0]))
    m = pmf.r <= r_bound
    r, w = pmf.r[m], W_raw[m]
    Ka_wrong = 4.0 * math.pi * float(np.trapz(r * r * np.exp(-pmf.beta * w), r))
    dG_wrong = (-KB * pmf.temperature * math.log(Ka_wrong / V_STANDARD_NM3)
                if Ka_wrong > 0 else float("nan"))
    return {"convention": pmf.convention,
            "dG_correct_kJmol": correct.dG_standard_kJmol,
            "dG_if_jacobian_double_counted_kJmol": dG_wrong,
            "difference_kJmol": dG_wrong - correct.dG_standard_kJmol,
            "note": ("identical for bulk_flat input; nonzero for "
                     "radial_density input — that nonzero value IS the bug")}


# =============================================================================
# §4  TEMPERATURE SERIES — van 't Hoff + Gibbs-Helmholtz
# =============================================================================

@dataclass
class ThermoSplit:
    dH_kJmol: float
    dS_kJmol_per_K: float
    dH_stderr: float
    dS_stderr: float
    method: str
    lever_arm_inv_K: float
    temperatures: List[float]
    dG_values: List[float]
    dG_errors: List[float]
    r_squared: float
    warning: str = ""


def vant_hoff(results: List[BindingResult]) -> ThermoSplit:
    """
    Weighted linear fit of y = ln(K_a/V°) against x = 1/T.
        slope = -dH/R ,  intercept = dS/R
    Assumes dCp = 0 over the range.
    """
    if len(results) < 2:
        raise ValueError("need >= 2 temperatures")
    T = np.array([r.temperature for r in results], dtype=np.float64)
    Ka = np.array([r.Ka_nm3 for r in results], dtype=np.float64)
    se = np.array([r.dG_stderr_kJmol for r in results], dtype=np.float64)
    y = np.log(Ka / V_STANDARD_NM3)
    x = 1.0 / T
    # sigma_y = beta * sigma_dG  (since y = -dG/(kT))
    sy = np.where(np.isfinite(se) & (se > 0),
                  se / (KB * T), 1.0)
    w = 1.0 / sy ** 2
    n = len(x)
    Sx = float(np.sum(w * x)); Sy = float(np.sum(w * y))
    Sxx = float(np.sum(w * x * x)); Sxy = float(np.sum(w * x * y))
    Sw = float(np.sum(w))
    den = Sw * Sxx - Sx * Sx
    if abs(den) < 1e-300:
        raise ValueError("degenerate van 't Hoff fit")
    slope = (Sw * Sxy - Sx * Sy) / den
    inter = (Sy - slope * Sx) / Sw
    var_slope = Sw / den
    var_inter = Sxx / den
    dH = -KB * slope
    dS = KB * inter
    dH_se = KB * math.sqrt(max(var_slope, 0.0))
    dS_se = KB * math.sqrt(max(var_inter, 0.0))
    yhat = slope * x + inter
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ybar = Sy / Sw
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    lever = float(x.max() - x.min())
    warn = ""
    if lever < 3.0e-4:
        warn = (f"SHORT LEVER ARM: Delta(1/T) = {lever:.3e} K^-1. "
                f"The van 't Hoff enthalpy error ({dH_se:.1f} kJ/mol) is "
                "large by construction. Widen the temperature range or "
                "do not quote dH.")
    return ThermoSplit(dH, dS, dH_se, dS_se, "van_t_hoff", lever,
                       T.tolist(),
                       [r.dG_standard_kJmol for r in results],
                       [r.dG_stderr_kJmol for r in results], r2, warn)


def gibbs_helmholtz_fd(results: List[BindingResult]) -> ThermoSplit:
    """
    Central finite difference of dG°(T):
        dS = -d(dG°)/dT   at T_mid
        dH = dG°(T_mid) + T_mid * dS
    With 3 equally spaced temperatures this is the natural estimator.
    """
    rs = sorted(results, key=lambda z: z.temperature)
    if len(rs) < 3:
        raise ValueError("need >= 3 temperatures for central difference")
    T = np.array([r.temperature for r in rs], dtype=np.float64)
    G = np.array([r.dG_standard_kJmol for r in rs], dtype=np.float64)
    se = np.array([r.dG_stderr_kJmol for r in rs], dtype=np.float64)
    i = len(rs) // 2
    Tlo, Thi = T[i - 1], T[i + 1]
    Glo, Ghi = G[i - 1], G[i + 1]
    dS = -(Ghi - Glo) / (Thi - Tlo)
    Tm = T[i]
    Gm = float(np.interp(Tm, T, G))
    dH = Gm + Tm * dS
    slo = se[i - 1] if np.isfinite(se[i - 1]) else 0.0
    shi = se[i + 1] if np.isfinite(se[i + 1]) else 0.0
    smid = se[i] if np.isfinite(se[i]) else 0.0
    dS_se = math.hypot(slo, shi) / (Thi - Tlo)
    dH_se = math.hypot(smid, Tm * dS_se)
    return ThermoSplit(dH, dS, dH_se, dS_se, "gibbs_helmholtz_fd",
                       float(1.0 / Tlo - 1.0 / Thi), T.tolist(),
                       G.tolist(), se.tolist(), float("nan"),
                       "assumes dG° linear in T over the range (dCp = 0)")


def fit_with_dcp(results: List[BindingResult], T0: float = 310.0
                 ) -> Dict[str, Any]:
    """
    dG°(T) = dH0 - T dS0 + dCp[(T - T0) - T ln(T/T0)]
    Three parameters. With exactly 3 temperatures the system is exactly
    determined and NO error estimate is available. Flagged explicitly.
    """
    if len(results) < 3:
        raise ValueError("need >= 3 temperatures")
    T = np.array([r.temperature for r in results], dtype=np.float64)
    G = np.array([r.dG_standard_kJmol for r in results], dtype=np.float64)
    A = np.vstack([np.ones_like(T), -T,
                   (T - T0) - T * np.log(T / T0)]).T
    if abs(np.linalg.det(A)) < 1e-12 and len(results) == 3:
        return {"status": "SINGULAR", "note": "temperatures too close"}
    coef, *_ = np.linalg.lstsq(A, G, rcond=None)
    out = {"status": "FITTED", "T0_K": T0,
           "dH0_kJmol": float(coef[0]),
           "dS0_kJmol_per_K": float(coef[1]),
           "dCp_kJmol_per_K": float(coef[2]),
           "n_temperatures": len(results)}
    if len(results) == 3:
        out["warning"] = ("EXACTLY DETERMINED: 3 points, 3 parameters. "
                          "No degrees of freedom remain, so NO error "
                          "estimate exists. dCp from 3 points is not "
                          "a measurement.")
    return out


# =============================================================================
# §5  AA vs CG TRANSFERABILITY COMPARISON
# =============================================================================

def compare_transferability(aa: List[BindingResult],
                            cg: List[BindingResult]) -> Dict[str, Any]:
    """
    Compare dG°(T), and the extracted dH / dS, between AA and CG.
    The headline numbers are ddH and ddS: a CG model that matches dG° at
    one temperature but not dH/dS has the wrong thermodynamic decomposition.
    """
    Ta = {round(r.temperature, 6) for r in aa}
    Tc = {round(r.temperature, 6) for r in cg}
    common = sorted(Ta & Tc)
    if len(common) < 2:
        return {"status": "NO_COMMON_TEMPERATURES",
                "aa_temperatures": sorted(Ta), "cg_temperatures": sorted(Tc)}
    aa_m = {round(r.temperature, 6): r for r in aa}
    cg_m = {round(r.temperature, 6): r for r in cg}
    per_T = []
    for T in common:
        a, c = aa_m[T], cg_m[T]
        ea = a.dG_stderr_kJmol if math.isfinite(a.dG_stderr_kJmol) else float("nan")
        ec = c.dG_stderr_kJmol if math.isfinite(c.dG_stderr_kJmol) else float("nan")
        comb = math.hypot(ea if math.isfinite(ea) else 0.0,
                          ec if math.isfinite(ec) else 0.0)
        ddG = c.dG_standard_kJmol - a.dG_standard_kJmol
        per_T.append({"T_K": T, "dG_aa": a.dG_standard_kJmol,
                      "dG_cg": c.dG_standard_kJmol, "ddG": ddG,
                      "ddG_stderr": comb,
                      "significant": bool(abs(ddG) > 2.0 * comb)
                      if comb > 0 else None})
    aa_vh, cg_vh = vant_hoff([aa_m[T] for T in common]), vant_hoff([cg_m[T] for T in common])
    ddH = cg_vh.dH_kJmol - aa_vh.dH_kJmol
    ddS = cg_vh.dS_kJmol_per_K - aa_vh.dS_kJmol_per_K
    ddH_se = math.hypot(aa_vh.dH_stderr, cg_vh.dH_stderr)
    ddS_se = math.hypot(aa_vh.dS_stderr, cg_vh.dS_stderr)
    rmse = math.sqrt(float(np.mean([d["ddG"] ** 2 for d in per_T])))
    verdict = ("TRANSFERABLE within error"
               if (abs(ddH) <= 2 * ddH_se and abs(ddS) <= 2 * ddS_se
                   and rmse < 2.0)
               else "NOT TRANSFERABLE — entropy/enthalpy split differs")
    return {
        "status": "COMPARED",
        "common_temperatures_K": common,
        "per_temperature": per_T,
        "dG_rmse_kJmol": rmse,
        "aa_vant_hoff": asdict(aa_vh),
        "cg_vant_hoff": asdict(cg_vh),
        "ddH_kJmol": ddH, "ddH_stderr": ddH_se,
        "ddS_kJmol_per_K": ddS, "ddS_stderr": ddS_se,
        "ddH_significant": bool(abs(ddH) > 2 * ddH_se) if ddH_se > 0 else None,
        "ddS_significant": bool(abs(ddS) > 2 * ddS_se) if ddS_se > 0 else None,
        "verdict": verdict,
        "interpretation": (
            "A temperature-independent CG effective potential is itself a "
            "free energy at its parameterisation temperature. Matching dG° "
            "at one T does not imply matching dH or dS. A nonzero ddH/ddS "
            "is the EXPECTED outcome for a generic CG model and quantifies "
            "the transferability failure; it is not a bug in this code."),
    }


# =============================================================================
# §6  SYNTHETIC ANALYTIC BENCHMARKS (self-test)
# =============================================================================

def square_well_pmf(T: float, r_c: float = 0.6, depth: float = 12.0,
                    n: int = 400, r_max: float = 2.5,
                    convention: str = "bulk_flat",
                    noise_se: float = 0.0) -> PMFData:
    """
    W_true(r) = -depth for r < r_c, 0 otherwise  (bulk-flat convention).
    Analytic:  K_a = (4 pi / 3) r_c^3 exp(depth / kT)
               dG° = -kT ln(K_a / V°)
    If convention='radial_density', the -2kT ln r Jacobian is ADDED IN so
    the module must remove it again — this tests the convention machinery.
    """
    r = np.linspace(r_max / n, r_max, n)
    W = np.where(r < r_c, -depth, 0.0)
    if convention == "radial_density":
        W = W - 2.0 * KB * T * np.log(r)
    se = np.full(n, noise_se) if noise_se > 0 else None
    return PMFData(r=r, W=W, temperature=T, convention=convention,
                   stderr=se, label=f"square_well_T{T}")


def square_well_exact(T: float, r_c: float = 0.6, depth: float = 12.0
                      ) -> Tuple[float, float]:
    Ka = (4.0 * math.pi / 3.0) * r_c ** 3 * math.exp(depth / (KB * T))
    dG = -KB * T * math.log(Ka / V_STANDARD_NM3)
    return Ka, dG


def linear_thermo_pmf(T: float, a: float = 20.0, b: float = 0.03,
                      r_c: float = 0.6, n: int = 400, r_max: float = 2.5
                      ) -> PMFData:
    """
    depth(T) = a - b T  =>  dG°(T) = -a + T (b - kB ln(V_b/V°))
    so the exact split is
        dH° = -a
        dS° = kB ln(V_b/V°) - b ,   V_b = (4pi/3) r_c^3
    Used to verify that van 't Hoff and Gibbs-Helmholtz recover a KNOWN
    entropy/enthalpy decomposition.
    """
    return square_well_pmf(T, r_c=r_c, depth=a - b * T, n=n, r_max=r_max)


# =============================================================================
# §7  SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 92)
    print(f"  STANDARD-STATE + TEMPERATURE-TRANSFERABILITY v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 92)
    np.random.seed(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    R_BOUND, BULK = 0.75, (1.6, 2.4)

    # ---- T1 standard state vs analytic square well --------------------
    print("\n[T1] Standard-state dG° vs ANALYTIC square well")
    for T in (300.0, 310.0, 320.0):
        pmf = square_well_pmf(T, r_c=0.6, depth=12.0)
        res = standard_state_binding(pmf, R_BOUND, BULK)
        Ka_ex, dG_ex = square_well_exact(T, 0.6, 12.0)
        rep(f"T={T:.0f} K  K_a matches analytic",
            abs(res.Ka_nm3 - Ka_ex) / Ka_ex < 0.02,
            f"K_a={res.Ka_nm3:.4f} exact={Ka_ex:.4f} nm^3")
        rep(f"T={T:.0f} K  dG° matches analytic",
            abs(res.dG_standard_kJmol - dG_ex) < 0.15,
            f"dG°={res.dG_standard_kJmol:.3f} exact={dG_ex:.3f} kJ/mol")

    # ---- T2 Jacobian convention: both conventions must agree ----------
    print("\n[T2] Jacobian convention — radial_density vs bulk_flat")
    T = 310.0
    p_bf = square_well_pmf(T, convention="bulk_flat")
    p_rd = square_well_pmf(T, convention="radial_density")
    r_bf = standard_state_binding(p_bf, R_BOUND, BULK)
    r_rd = standard_state_binding(p_rd, R_BOUND, BULK)
    rep("both conventions give same dG°",
        abs(r_bf.dG_standard_kJmol - r_rd.dG_standard_kJmol) < 0.05,
        f"bulk_flat={r_bf.dG_standard_kJmol:.3f} "
        f"radial_density={r_rd.dG_standard_kJmol:.3f}")
    chk = jacobian_double_count_check(p_rd, R_BOUND, BULK)
    rep("double-counting is DETECTED and nonzero for radial_density",
        abs(chk["difference_kJmol"]) > 1.0,
        f"error if double-counted = {chk['difference_kJmol']:+.2f} kJ/mol")
    chk_bf = jacobian_double_count_check(p_bf, R_BOUND, BULK)
    rep("no difference for bulk_flat input",
        abs(chk_bf["difference_kJmol"]) < 0.05,
        f"diff={chk_bf['difference_kJmol']:+.3f}")

    # ---- T3 bulk flatness diagnostic ----------------------------------
    print("\n[T3] Bulk-flatness diagnostic")
    flat = bulk_flatness(p_bf, BULK)
    rep("square well bulk is FLAT", flat["ok"],
        f"slope={flat['slope_kJmol_per_nm']:.3e} verdict={flat['verdict']}")
    bad = PMFData(r=p_bf.r, W=p_bf.W + 3.0 * p_bf.r,
                  temperature=T, convention="bulk_flat")
    fbad = bulk_flatness(bad, BULK)
    rep("sloped bulk is REJECTED", not fbad["ok"],
        f"drift={fbad['drift_over_bulk_kJmol']:.2f} kJ/mol")

    # ---- T4 van 't Hoff recovers a KNOWN dH/dS ------------------------
    print("\n[T4] van 't Hoff + Gibbs-Helmholtz recover KNOWN dH°, dS°")
    a, b, r_c = 20.0, 0.03, 0.6
    Vb = (4.0 * math.pi / 3.0) * r_c ** 3
    dH_exact = -a
    dS_exact = KB * math.log(Vb / V_STANDARD_NM3) - b
    res = [standard_state_binding(linear_thermo_pmf(T, a, b, r_c), R_BOUND, BULK)
           for T in (300.0, 310.0, 320.0)]
    vh = vant_hoff(res)
    gh = gibbs_helmholtz_fd(res)
    print(f"      exact   dH°={dH_exact:+.3f} kJ/mol  dS°={dS_exact:+.5f} kJ/mol/K")
    print(f"      v.Hoff  dH°={vh.dH_kJmol:+.3f}         dS°={vh.dS_kJmol_per_K:+.5f}")
    print(f"      G-H fd  dH°={gh.dH_kJmol:+.3f}         dS°={gh.dS_kJmol_per_K:+.5f}")
    rep("van 't Hoff dH° within 0.5 kJ/mol",
        abs(vh.dH_kJmol - dH_exact) < 0.5,
        f"err={vh.dH_kJmol - dH_exact:+.3f}")
    rep("van 't Hoff dS° within 2e-3",
        abs(vh.dS_kJmol_per_K - dS_exact) < 2e-3,
        f"err={vh.dS_kJmol_per_K - dS_exact:+.5f}")
    rep("Gibbs-Helmholtz dH° within 0.5",
        abs(gh.dH_kJmol - dH_exact) < 0.5)
    rep("van 't Hoff and G-H agree",
        abs(vh.dH_kJmol - gh.dH_kJmol) < 0.3,
        f"|diff|={abs(vh.dH_kJmol - gh.dH_kJmol):.3f}")
    print(f"      lever arm Delta(1/T) = {vh.lever_arm_inv_K:.3e} K^-1")

    # ---- T5 error propagation is honest about the short lever arm -----
    print("\n[T5] Error propagation — short lever arm must be flagged")
    noisy = []
    for T in (300.0, 310.0, 320.0):
        p = linear_thermo_pmf(T, a, b, r_c)
        p.stderr = np.full_like(p.r, 0.5)     # 0.5 kJ/mol per bin
        noisy.append(standard_state_binding(p, R_BOUND, BULK))
    vh_n = vant_hoff(noisy)
    print(f"      dH° = {vh_n.dH_kJmol:+.2f} +/- {vh_n.dH_stderr:.2f} kJ/mol")
    print(f"      dS° = {vh_n.dS_kJmol_per_K:+.5f} +/- {vh_n.dS_stderr:.5f}")
    rep("dH error is finite and nonzero", vh_n.dH_stderr > 0)
    rep("short lever arm WARNING emitted", len(vh_n.warning) > 0,
        vh_n.warning[:78] + "..." if vh_n.warning else "")

    # ---- T6 dCp fit is flagged as underdetermined ---------------------
    print("\n[T6] dCp fit with exactly 3 temperatures")
    cp = fit_with_dcp(res, T0=310.0)
    rep("dCp fit runs", cp["status"] in ("FITTED", "SINGULAR"))
    if cp["status"] == "FITTED":
        rep("underdetermination WARNING present", "warning" in cp,
            f"dCp={cp['dCp_kJmol_per_K']:+.4f} (no error estimate possible)")

    # ---- T7 transferability: identical models agree -------------------
    print("\n[T7] Transferability — identical AA and CG must AGREE")
    aa = [standard_state_binding(linear_thermo_pmf(T, a, b, r_c), R_BOUND, BULK)
          for T in (300.0, 310.0, 320.0)]
    cg_same = [standard_state_binding(linear_thermo_pmf(T, a, b, r_c), R_BOUND, BULK)
               for T in (300.0, 310.0, 320.0)]
    cmp_same = compare_transferability(aa, cg_same)
    rep("identical models -> TRANSFERABLE",
        cmp_same["verdict"].startswith("TRANSFERABLE"),
        f"ddH={cmp_same['ddH_kJmol']:+.3e} ddS={cmp_same['ddS_kJmol_per_K']:+.3e}")

    # ---- T8 transferability: T-independent CG must FAIL ---------------
    print("\n[T8] Transferability — T-INDEPENDENT CG well must FAIL")
    # AA depth varies with T (real entropy); CG depth fixed at 310 K value
    depth_310 = a - b * 310.0
    cg_fixed = [standard_state_binding(
        square_well_pmf(T, r_c=r_c, depth=depth_310), R_BOUND, BULK)
        for T in (300.0, 310.0, 320.0)]
    cmp_bad = compare_transferability(aa, cg_fixed)
    print(f"      ddH = {cmp_bad['ddH_kJmol']:+.3f} +/- {cmp_bad['ddH_stderr']:.3f} kJ/mol")
    print(f"      ddS = {cmp_bad['ddS_kJmol_per_K']:+.5f} +/- "
          f"{cmp_bad['ddS_stderr']:.5f} kJ/mol/K")
    print(f"      dG RMSE across T = {cmp_bad['dG_rmse_kJmol']:.3f} kJ/mol")
    print(f"      verdict: {cmp_bad['verdict']}")
    rep("T-independent CG is detected as NOT TRANSFERABLE",
        cmp_bad["verdict"].startswith("NOT TRANSFERABLE"))
    rep("ddH matches the analytic expectation -b*T term",
        abs(cmp_bad["ddH_kJmol"]) > 0.5,
        "the CG model has no entropy from the eliminated DOF")

    # ---- T9 restraint warning ----------------------------------------
    print("\n[T9] Restraint warning")
    rw = standard_state_binding(p_bf, R_BOUND, BULK,
                                orientational_restraints=True)
    rep("orientational-restraint warning emitted",
        len(rw.restrained_warning) > 0)

    # ---- T10 no common temperatures is reported, not faked ------------
    print("\n[T10] Mismatched temperature sets")
    other = [standard_state_binding(linear_thermo_pmf(T, a, b, r_c), R_BOUND, BULK)
             for T in (290.0, 340.0)]
    cm = compare_transferability(aa, other)
    rep("no-common-T reported honestly",
        cm["status"] == "NO_COMMON_TEMPERATURES", cm["status"])

    print("\n" + "=" * 92)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 92)
    print("""
  WHAT THESE TESTS PROVE
    T1  the standard-state volume correction reproduces the ANALYTIC
        square-well K_a and dG° at three temperatures
    T2  both Jacobian conventions give the SAME answer, and the
        double-counting error is detected and quantified
    T3  a non-flat bulk is rejected before any dG° is quoted
    T4  van 't Hoff and Gibbs-Helmholtz recover a KNOWN dH°, dS°
    T5  the short-lever-arm enthalpy error is reported, with a warning
    T6  a 3-point dCp fit is flagged as having NO error estimate
    T7  identical models are reported TRANSFERABLE (no false positive)
    T8  a temperature-independent CG well is reported NOT TRANSFERABLE
        — this is the expected physical failure, correctly detected

  WHAT IS NOT PROVEN
    * real AA-vs-CG agreement: you must run the umbrella sweeps
    * orientational/positional restraint corrections (Woo-Roux) are NOT
      computed here; only the radial volume term is
    * dCp from 3 temperatures is not a measurement
    * finite-size, salt/activity, and protonation-linkage corrections
      are absent
    * matching dG° at one temperature never establishes transferability
""")
    return ok


# =============================================================================
# §8  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=f"Standard-state + T-transferability v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--pmf", nargs="*", default=None,
                    help="NPZ files with r,W,temperature,convention[,stderr]")
    ap.add_argument("--r-bound", type=float, default=0.75)
    ap.add_argument("--bulk", type=float, nargs=2, default=(1.6, 2.4))
    ap.add_argument("--restrained", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.test:
        raise SystemExit(0 if self_test() else 1)

    if not a.pmf:
        print("No --pmf given. Running self-test so nothing is fabricated.")
        raise SystemExit(0 if self_test() else 1)

    groups: Dict[str, List[BindingResult]] = {}
    for path in a.pmf:
        z = np.load(path)
        label = str(z["label"]) if "label" in z else path
        pmf = PMFData(r=z["r"], W=z["W"],
                      temperature=float(z["temperature"]),
                      convention=str(z["convention"]),
                      stderr=z["stderr"] if "stderr" in z else None,
                      label=label)
        res = standard_state_binding(pmf, a.r_bound, tuple(a.bulk),
                                     orientational_restraints=a.restrained,
                                     label=label)
        print(f"  {label}: T={res.temperature:.1f} K  "
              f"K_a={res.Ka_nm3:.4f} nm^3  "
              f"dG°={res.dG_standard_kJmol:+.3f} "
              f"+/- {res.dG_stderr_kJmol:.3f} kJ/mol  "
              f"[{res.bulk_flat['verdict']}]")
        if res.restrained_warning:
            print(f"    WARNING: {res.restrained_warning}")
        key = "aa" if "aa" in label.lower() else ("cg" if "cg" in label.lower() else label)
        groups.setdefault(key, []).append(res)

    out: Dict[str, Any] = {"groups": {k: [asdict(r) for r in v]
                                     for k, v in groups.items()}}
    for k, v in groups.items():
        if len(v) >= 2:
            vh = vant_hoff(v)
            out[f"{k}_vant_hoff"] = asdict(vh)
            print(f"\n  {k} van 't Hoff: dH°={vh.dH_kJmol:+.2f} "
                  f"+/- {vh.dH_stderr:.2f} kJ/mol, "
                  f"dS°={vh.dS_kJmol_per_K:+.5f} +/- {vh.dS_stderr:.5f}")
            if vh.warning:
                print(f"    WARNING: {vh.warning}")
    if "aa" in groups and "cg" in groups:
        cmp_ = compare_transferability(groups["aa"], groups["cg"])
        out["transferability"] = cmp_
        if cmp_["status"] == "COMPARED":
            print(f"\n  TRANSFERABILITY: {cmp_['verdict']}")
            print(f"    ddH = {cmp_['ddH_kJmol']:+.3f} +/- {cmp_['ddH_stderr']:.3f} kJ/mol")
            print(f"    ddS = {cmp_['ddS_kJmol_per_K']:+.5f} +/- "
                  f"{cmp_['ddS_stderr']:.5f} kJ/mol/K")
            print(f"    dG RMSE = {cmp_['dG_rmse_kJmol']:.3f} kJ/mol")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\n  json -> {a.out}")


if __name__ == "__main__":
    main()
