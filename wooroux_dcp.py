#!/usr/bin/env python3
"""
=============================================================================
WOO–ROUX ANGULAR RESTRAINT CORRECTION  +  DIRECT ΔCp FROM ENTHALPY
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — WOO–ROUX ANGULAR RESTRAINT CORRECTION
  Absolute binding free energies from a restrained radial PMF require the
  free-energy cost of the orientational/positional-angular restraints to be
  removed. The thermodynamic cycle is

      free(no restraint) --apply--> free(restrained)
                                        |
                                     bind (restrained PMF)
                                        v
      bound(no restraint) <--release-- bound(restrained)

  giving
      dG_bind = dG_r^(R) + sum_i [ dG_apply_free_i - dG_apply_bound_i ]

  * dG_apply_free_i is an ANALYTIC quadrature of the restraint potential
    against the ideal (uniform) angular measure. No simulation needed.
        dG_apply_free = -kT ln[ (1/mu_tot) int mu(xi) exp(-beta U(xi)) dxi ]
  * dG_apply_bound_i is obtained from a bound-state PMF along xi measured
    WITH the restraint on:
        dG_apply_bound = -kT ln[ int mu e^{-beta W_meas}
                                 / int mu e^{-beta (W_meas - U)} ]
    i.e. the restraint is removed from the measured PMF before integrating.

  MEASURE BOOKKEEPING (the classic double-counting trap)
    position angular : polar (sin th dth, tot 2) x azimuthal (dphi, tot 2pi)
                       -> 4pi
    orientation      : euler_polar (sin Th, tot 2) x euler_azimuthal (2pi)
                       x euler_twist (2pi)  -> 8 pi^2
    If the positional angular coordinates are RESTRAINED, the radial term
    must NOT carry the 4pi factor; those coordinates then appear in the
    correction sum instead. This is an explicit, required flag.

  References
    Woo & Roux (2005) PNAS 102, 6825.
    Deng & Roux (2009) J. Phys. Chem. B 113, 2234.
    Gumbart, Roux & Chipot (2013) JCTC 9, 3789.
    Gilson, Given, Bush & McCammon (1997) Biophys. J. 72, 1047.

PART B — ΔCp FROM DIRECT ENTHALPY (avoids the van 't Hoff lever arm)
  van 't Hoff extracts dH from the slope of ln(K) vs 1/T. Over 300-320 K,
  Delta(1/T) ~ 2.2e-4 K^-1, so the enthalpy error is amplified enormously.
  If dH(T) is available DIRECTLY (ITC; or a properly constructed energy
  difference), then
        dCp = d(dH)/dT
  is a plain slope against T, with a lever arm of 20 K rather than 2.2e-4.
  With dCp fixed, the full temperature dependence is
        dH(T) = dH0 + dCp (T - T0)
        dS(T) = dS0 + dCp ln(T/T0)
        dG(T) = dH0 + dCp (T - T0) - T [ dS0 + dCp ln(T/T0) ]
  and the measured dG(T) becomes an INDEPENDENT consistency check rather
  than the source of dH.

HONEST SCOPE
  ❌ dG_apply_bound requires the bound PMF to cover the region where the
     UNRESTRAINED integrand has weight. A coverage diagnostic is computed
     and a warning is emitted when it fails; the number is then not usable.
  ❌ The separable (radial x angular) factorisation is an approximation.
     Strong radial-angular coupling in the bound state breaks it.
  ❌ A "direct enthalpy" from MD potential-energy differences is NOT the
     experimental dH: it omits pV, solvent reorganisation, and (for CG
     models) the entropy of the eliminated degrees of freedom. The module
     accepts whatever dH you supply and flags its provenance.
  ❌ Symmetry-number / equivalent-orientation factors are NOT automatic.
     A symmetric ligand has a smaller effective 8pi^2; supply the factor.
  ❌ No claim of state of the art.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("wooroux_dcp")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

KB = 0.008314462618            # kJ mol^-1 K^-1
NA = 6.02214076e23
V_STD_NM3 = 1.0e27 / NA        # 1 mol/L -> 1.660539 nm^3
FOUR_PI = 4.0 * math.pi
EIGHT_PI2 = 8.0 * math.pi ** 2


# =============================================================================
# §1  ANGULAR COORDINATES AND THEIR MEASURES
# =============================================================================

@dataclass
class AngularCoord:
    """
    kind determines the integration range and the invariant measure.

      polar            theta in [0,pi],   measure sin(theta),  total 2
      azimuthal        phi   in [0,2pi),  measure 1,           total 2pi
      euler_polar      Theta in [0,pi],   measure sin(Theta),  total 2
      euler_azimuthal  Phi   in [0,2pi),  measure 1,           total 2pi
      euler_twist      Psi   in [0,2pi),  measure 1,           total 2pi
      cartesian_1d     x in [0,L],        measure 1,           total L
                       (a positional restraint; NOT an angle)

    `positional_angular` marks the coordinates whose free measure supplies
    the 4pi in the radial term, so the bookkeeping can be checked.
    """
    name: str
    kind: str
    k: float                      # kJ/mol/rad^2 (or /nm^2 for cartesian)
    center: float                 # rad (or nm)
    length: float = 1.0           # only for cartesian_1d
    symmetry_number: int = 1      # equivalent orientations; divides measure

    @property
    def positional_angular(self) -> bool:
        return self.kind in ("polar", "azimuthal")

    @property
    def orientational(self) -> bool:
        return self.kind in ("euler_polar", "euler_azimuthal", "euler_twist")

    def grid(self, n: int = 4001) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (xi, measure_density, total_measure)."""
        if self.kind in ("polar", "euler_polar"):
            xi = np.linspace(0.0, math.pi, n)
            mu = np.sin(xi)
            tot = 2.0
        elif self.kind in ("azimuthal", "euler_azimuthal", "euler_twist"):
            xi = np.linspace(0.0, 2.0 * math.pi, n)
            mu = np.ones_like(xi)
            tot = 2.0 * math.pi
        elif self.kind == "cartesian_1d":
            xi = np.linspace(0.0, self.length, n)
            mu = np.ones_like(xi)
            tot = self.length
        else:
            raise ValueError(f"unknown kind {self.kind!r}")
        return xi, mu, tot / max(self.symmetry_number, 1)

    def potential(self, xi: np.ndarray) -> np.ndarray:
        """Harmonic restraint, angle-wrapped where periodic."""
        d = xi - self.center
        if self.kind in ("azimuthal", "euler_azimuthal", "euler_twist"):
            d = (d + math.pi) % (2.0 * math.pi) - math.pi
        return 0.5 * self.k * d ** 2


def free_apply_free_energy(coord: AngularCoord, T: float,
                           n_grid: int = 4001) -> Dict[str, Any]:
    """
    ANALYTIC (quadrature) free-energy cost of applying the restraint to an
    otherwise free coordinate distributed according to its invariant measure.

        dG_apply_free = -kT ln[ (1/mu_tot) int mu(xi) e^{-beta U(xi)} dxi ]

    Exactly zero when k = 0. Positive for k > 0.
    """
    xi, mu, tot = coord.grid(n_grid)
    U = coord.potential(xi)
    beta = 1.0 / (KB * T)
    w = mu * np.exp(-beta * U)
    Z = float(np.trapz(w, xi))
    Z0 = float(np.trapz(mu, xi))
    if Z <= 0 or Z0 <= 0:
        raise ValueError("zero measure; check restraint center/range")
    frac = Z / Z0
    dG = -KB * T * math.log(frac)
    # narrow-restraint analytic cross-check (Gaussian around the center)
    approx = None
    if coord.k > 0:
        sig = math.sqrt(KB * T / coord.k)
        if coord.kind in ("polar", "euler_polar"):
            m0 = math.sin(coord.center)
            if m0 > 1e-6 and sig < 0.2:
                approx = -KB * T * math.log(
                    m0 * sig * math.sqrt(2.0 * math.pi) / Z0)
        elif coord.kind in ("azimuthal", "euler_azimuthal", "euler_twist"):
            if sig < 0.5:
                approx = -KB * T * math.log(
                    sig * math.sqrt(2.0 * math.pi) / Z0)
    return {"name": coord.name, "kind": coord.kind, "T_K": T,
            "dG_apply_free_kJmol": dG, "accessible_fraction": frac,
            "narrow_restraint_approx_kJmol": approx,
            "measure_total": tot, "symmetry_number": coord.symmetry_number}


def bound_apply_free_energy(xi: np.ndarray, W_measured: np.ndarray,
                            coord: AngularCoord, T: float,
                            coverage_tol: float = 0.02) -> Dict[str, Any]:
    """
    Free-energy cost of applying the restraint IN THE BOUND STATE, from a
    bound PMF measured WITH the restraint already on.

        numerator   = int mu e^{-beta W_meas}
        denominator = int mu e^{-beta (W_meas - U)}      (restraint removed)
        dG_apply_bound = -kT ln(numerator / denominator)

    COVERAGE DIAGNOSTIC: if the unrestrained integrand carries more than
    `coverage_tol` of its weight in the outer 5% of the sampled range, the
    bound PMF does not cover the released state and the number is unusable.
    """
    xi = np.asarray(xi, dtype=np.float64)
    W = np.asarray(W_measured, dtype=np.float64)
    if xi.shape != W.shape or xi.ndim != 1:
        raise ValueError("xi and W_measured must be 1-D of equal length")
    _, mu_full, _ = coord.grid(8)
    # interpolate the measure density onto the supplied grid
    if coord.kind in ("polar", "euler_polar"):
        mu = np.sin(xi)
    else:
        mu = np.ones_like(xi)
    U = coord.potential(xi)
    beta = 1.0 / (KB * T)
    W0 = W - U                                   # intrinsic bound PMF
    W0 = W0 - np.min(W0)
    Wm = W - np.min(W)
    num = mu * np.exp(-beta * Wm)
    den = mu * np.exp(-beta * W0)
    Zn = float(np.trapz(num, xi))
    Zd = float(np.trapz(den, xi))
    if Zn <= 0 or Zd <= 0:
        raise ValueError("zero partition function in bound state")
    dG = -KB * T * math.log(Zn / Zd)
    # coverage: weight of the released integrand near the range edges
    n = len(xi)
    edge = max(1, int(0.05 * n))
    tail = float(np.trapz(den[:edge], xi[:edge])
                 + np.trapz(den[-edge:], xi[-edge:]))
    tail_frac = tail / Zd
    ok = tail_frac < coverage_tol
    return {"name": coord.name, "T_K": T,
            "dG_apply_bound_kJmol": dG,
            "released_tail_fraction": tail_frac,
            "coverage_ok": bool(ok),
            "warning": ("" if ok else
                        "BOUND PMF DOES NOT COVER THE RELEASED STATE — "
                        "dG_apply_bound is not usable. Extend the sampling "
                        "window or weaken the restraint.")}


# =============================================================================
# §2  RADIAL TERM WITH EXPLICIT 4pi BOOKKEEPING
# =============================================================================

def radial_binding_term(r: np.ndarray, W: np.ndarray, T: float,
                        r_bound: float,
                        bulk_window: Tuple[float, float],
                        positional_angular_restrained: bool,
                        jacobian: str = "radial_density") -> Dict[str, Any]:
    """
    dG_r^(R) = -kT ln[ (C/V0) int_0^{r_bound} r^2 e^{-beta W_true} dr ]

      C = 4pi  if the positional angular coordinates are FREE
      C = 1    if they are RESTRAINED (their measure is then paid for by
               the corresponding angular correction terms)

    `jacobian` follows the convention already used in the PMF pipeline:
      radial_density -> raw WHAM output on a distance coordinate; the
                        +2kT ln r conversion is applied here
      bulk_flat      -> Jacobian already removed
    """
    r = np.asarray(r, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if np.any(r <= 0):
        raise ValueError("r must be > 0")
    beta = 1.0 / (KB * T)
    Wt = W + 2.0 * KB * T * np.log(r) if jacobian == "radial_density" else W.copy()
    mb = (r >= bulk_window[0]) & (r <= bulk_window[1])
    if mb.sum() < 2:
        raise ValueError("bulk window has < 2 points")
    Wt = Wt - float(np.mean(Wt[mb]))
    m = r <= r_bound
    if m.sum() < 2:
        raise ValueError("bound region has < 2 points")
    integ = r[m] ** 2 * np.exp(-beta * Wt[m])
    I = float(np.trapz(integ, r[m]))
    C = 1.0 if positional_angular_restrained else FOUR_PI
    if I <= 0:
        raise ValueError("non-positive radial integral")
    dG = -KB * T * math.log(C * I / V_STD_NM3)
    # bulk flatness check
    x, y = r[mb], Wt[mb]
    A = np.vstack([x, np.ones(len(x))]).T
    slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    drift = abs(slope) * float(x.max() - x.min())
    return {"dG_radial_restrained_kJmol": dG,
            "angular_prefactor": C,
            "positional_angular_restrained": positional_angular_restrained,
            "radial_integral_nm3": I,
            "bulk_slope_kJmol_per_nm": slope,
            "bulk_drift_kJmol": drift,
            "bulk_flat": bool(drift < 1.0)}


# =============================================================================
# §3  FULL WOO–ROUX CORRECTED BINDING FREE ENERGY
# =============================================================================

@dataclass
class WooRouxResult:
    T_K: float
    dG_radial_restrained: float
    corrections: List[Dict[str, Any]]
    dG_correction_total: float
    dG_bind_standard: float
    orientation_measure_used: float
    all_coverage_ok: bool
    warnings: List[str] = field(default_factory=list)


def wooroux_binding_free_energy(
    r: np.ndarray, W_r: np.ndarray, T: float,
    r_bound: float, bulk_window: Tuple[float, float],
    coords: List[AngularCoord],
    bound_pmfs: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    jacobian: str = "radial_density",
    symmetry_number_total: int = 1,
) -> WooRouxResult:
    """
    Assemble the corrected absolute binding free energy.

    coords      : every restrained coordinate (positional-angular AND
                  orientational). Coordinates NOT listed are assumed free
                  in BOTH states and contribute nothing.
    bound_pmfs  : {coord.name: (xi, W_measured)} for the bound-state PMF
                  along that coordinate, measured with its restraint on.
                  If a coordinate has no entry, dG_apply_bound is taken
                  as 0 (the bound state is assumed to already confine it
                  fully) and a WARNING is recorded — this is an
                  approximation, not a measurement.
    """
    pos_restrained = any(c.positional_angular for c in coords)
    rad = radial_binding_term(r, W_r, T, r_bound, bulk_window,
                              pos_restrained, jacobian)
    warnings: List[str] = []
    if not rad["bulk_flat"]:
        warnings.append("Radial PMF bulk region is NOT flat "
                        f"(drift {rad['bulk_drift_kJmol']:.2f} kJ/mol).")
    # bookkeeping check: if positional angular free, 4pi must be present
    if not pos_restrained and abs(rad["angular_prefactor"] - FOUR_PI) > 1e-9:
        raise RuntimeError("bookkeeping error: free positional angular "
                           "requires the 4pi prefactor")
    corr: List[Dict[str, Any]] = []
    total = 0.0
    all_ok = True
    for c in coords:
        fa = free_apply_free_energy(c, T)
        if bound_pmfs is not None and c.name in bound_pmfs:
            xi, Wm = bound_pmfs[c.name]
            ba = bound_apply_free_energy(xi, Wm, c, T)
            all_ok &= ba["coverage_ok"]
            if ba["warning"]:
                warnings.append(f"[{c.name}] {ba['warning']}")
            dgb = ba["dG_apply_bound_kJmol"]
            src = "bound PMF"
            extra = {"released_tail_fraction": ba["released_tail_fraction"],
                     "coverage_ok": ba["coverage_ok"]}
        else:
            dgb = 0.0
            src = "ASSUMED ZERO (no bound PMF supplied)"
            warnings.append(f"[{c.name}] dG_apply_bound assumed 0 — the bound "
                            "state is assumed to confine this coordinate "
                            "completely. This is an approximation.")
            extra = {}
        d = fa["dG_apply_free_kJmol"] - dgb
        total += d
        corr.append({"name": c.name, "kind": c.kind,
                     "dG_apply_free_kJmol": fa["dG_apply_free_kJmol"],
                     "dG_apply_bound_kJmol": dgb,
                     "bound_source": src,
                     "correction_kJmol": d, **extra})
    # orientation symmetry: reduces the free orientational measure
    orient = [c for c in coords if c.orientational]
    om = EIGHT_PI2
    if symmetry_number_total > 1:
        om /= symmetry_number_total
        total += -KB * T * math.log(1.0 / symmetry_number_total)
        warnings.append(f"Symmetry number {symmetry_number_total} applied to "
                        "the free orientational measure. Verify it is the "
                        "correct rotational symmetry of the ligand.")
    dG = rad["dG_radial_restrained_kJmol"] + total
    return WooRouxResult(
        T_K=T, dG_radial_restrained=rad["dG_radial_restrained_kJmol"],
        corrections=corr, dG_correction_total=total,
        dG_bind_standard=dG, orientation_measure_used=om,
        all_coverage_ok=all_ok, warnings=warnings)


# =============================================================================
# §4  DIRECT ΔCp FROM ENTHALPY
# =============================================================================

@dataclass
class ThermoPoint:
    T_K: float
    dH_kJmol: float
    dH_stderr: float = float("nan")
    dG_kJmol: float = float("nan")
    dG_stderr: float = float("nan")
    provenance: str = "unspecified"


@dataclass
class DirectCpResult:
    dCp_kJmol_per_K: float
    dCp_stderr: float
    dH0_kJmol: float
    dH0_stderr: float
    T0_K: float
    lever_arm_K: float
    r_squared: float
    n_points: int
    dS0_kJmol_per_K: Optional[float]
    dG_reconstructed: List[Dict[str, float]]
    consistency_ok: bool
    warnings: List[str] = field(default_factory=list)


def _wls(x: np.ndarray, y: np.ndarray, sy: np.ndarray
         ) -> Tuple[float, float, float, float, float]:
    """Weighted linear fit y = a + b x. Returns a,b,se_a,se_b,R2."""
    w = 1.0 / np.where(np.isfinite(sy) & (sy > 0), sy, 1.0) ** 2
    Sw = w.sum(); Sx = (w * x).sum(); Sy = (w * y).sum()
    Sxx = (w * x * x).sum(); Sxy = (w * x * y).sum()
    den = Sw * Sxx - Sx * Sx
    if abs(den) < 1e-300:
        raise ValueError("degenerate fit")
    b = (Sw * Sxy - Sx * Sy) / den
    a = (Sy - b * Sx) / Sw
    se_b = math.sqrt(Sw / den)
    se_a = math.sqrt(Sxx / den)
    yhat = a + b * x
    ybar = Sy / Sw
    ss_res = float((w * (y - yhat) ** 2).sum())
    ss_tot = float((w * (y - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(se_a), float(se_b), r2


def direct_cp_from_enthalpy(points: List[ThermoPoint], T0: float = 310.0
                            ) -> DirectCpResult:
    """
    dCp = d(dH)/dT by weighted linear regression of dH against T.

    The lever arm is Delta(T) in KELVIN, not Delta(1/T). This is the whole
    point: it removes the van 't Hoff error amplification.

    If dG is also supplied at each T, dS0 is obtained from
        dS(T) = (dH(T) - dG(T)) / T
    and dG(T) is RECONSTRUCTED from (dH0, dS0, dCp) as an independent
    consistency check.
    """
    if len(points) < 3:
        raise ValueError("need >= 3 temperatures for a dCp with an error bar")
    T = np.array([p.T_K for p in points], dtype=np.float64)
    H = np.array([p.dH_kJmol for p in points], dtype=np.float64)
    sH = np.array([p.dH_stderr for p in points], dtype=np.float64)
    warnings: List[str] = []
    prov = {p.provenance for p in points}
    if any("md" in s.lower() or "potential" in s.lower() for s in prov):
        warnings.append(
            "At least one dH is flagged as MD/potential-energy derived. "
            "That is NOT the experimental enthalpy: pV, solvent "
            "reorganisation, and (for CG) eliminated-DOF entropy are "
            "missing. dCp inherits that bias.")
    if len(prov) > 1:
        warnings.append(f"Mixed enthalpy provenance {sorted(prov)} — "
                        "do not mix calorimetric and model-derived dH.")
    x = T - T0
    a, b, se_a, se_b, r2 = _wls(x, H, sH)
    lever = float(T.max() - T.min())
    if lever < 10.0:
        warnings.append(f"Temperature span {lever:.1f} K is short even for "
                        "a direct dH slope; dCp error will be large.")
    # dS0 and reconstruction
    dS0 = None
    recon: List[Dict[str, float]] = []
    consistent = True
    G = np.array([p.dG_kJmol for p in points], dtype=np.float64)
    sG = np.array([p.dG_stderr for p in points], dtype=np.float64)
    if np.all(np.isfinite(G)):
        S = (H - G) / T
        sS = np.sqrt(np.where(np.isfinite(sH), sH, 0.0) ** 2
                     + np.where(np.isfinite(sG), sG, 0.0) ** 2) / T
        # dS(T) = dS0 + dCp ln(T/T0); subtract the known dCp term and average
        S0_samples = S - b * np.log(T / T0)
        wS = 1.0 / np.where(np.isfinite(sS) & (sS > 0), sS, 1.0) ** 2
        dS0 = float((wS * S0_samples).sum() / wS.sum())
        for i, p in enumerate(points):
            Ti = p.T_K
            Hr = a + b * (Ti - T0)
            Sr = dS0 + b * math.log(Ti / T0)
            Gr = Hr - Ti * Sr
            se = sG[i] if np.isfinite(sG[i]) and sG[i] > 0 else float("nan")
            dev = Gr - p.dG_kJmol
            ok = (abs(dev) <= 2.0 * se) if np.isfinite(se) else True
            consistent &= bool(ok)
            recon.append({"T_K": Ti, "dG_measured": p.dG_kJmol,
                          "dG_reconstructed": Gr, "deviation": dev,
                          "within_2sigma": bool(ok)})
        if not consistent:
            warnings.append("Reconstructed dG(T) disagrees with the measured "
                            "dG(T) beyond 2 sigma at one or more temperatures. "
                            "Either dH and dG are mutually inconsistent, or "
                            "dCp is not constant over the range.")
    else:
        warnings.append("dG not supplied at every T; dS0 and the independent "
                        "consistency check are unavailable.")
    return DirectCpResult(
        dCp_kJmol_per_K=b, dCp_stderr=se_b, dH0_kJmol=a, dH0_stderr=se_a,
        T0_K=T0, lever_arm_K=lever, r_squared=r2, n_points=len(points),
        dS0_kJmol_per_K=dS0, dG_reconstructed=recon,
        consistency_ok=bool(consistent), warnings=warnings)


def vant_hoff_from_dG(points: List[ThermoPoint]) -> Dict[str, float]:
    """
    For COMPARISON ONLY: dH from the slope of ln(K) vs 1/T, where
    ln(K/V0) = -dG/(kT). Reproduces the lever-arm amplification so the
    advantage of the direct route is measurable, not asserted.
    """
    T = np.array([p.T_K for p in points], dtype=np.float64)
    G = np.array([p.dG_kJmol for p in points], dtype=np.float64)
    sG = np.array([p.dG_stderr for p in points], dtype=np.float64)
    if not np.all(np.isfinite(G)):
        raise ValueError("dG required at every T")
    y = -G / (KB * T)
    sy = np.where(np.isfinite(sG) & (sG > 0), sG, 1.0) / (KB * T)
    x = 1.0 / T
    _, b, _, se_b, r2 = _wls(x, y, sy)
    return {"dH_vant_hoff_kJmol": -KB * b,
            "dH_vant_hoff_stderr": KB * se_b,
            "lever_arm_inv_K": float(x.max() - x.min()),
            "r_squared": r2}


# =============================================================================
# §5  SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 92)
    print(f"  WOO–ROUX + DIRECT ΔCp v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 92)
    np.random.seed(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    T = 310.0
    kT = KB * T

    # ---- T1 zero restraint costs nothing -------------------------------
    print("\n[T1] Zero restraint -> dG_apply_free = 0 exactly")
    for kind in ("polar", "azimuthal", "euler_polar", "euler_twist"):
        c = AngularCoord(name=f"free_{kind}", kind=kind, k=0.0,
                         center=math.pi / 2 if "polar" in kind else 0.0)
        r = free_apply_free_energy(c, T)
        rep(f"{kind} k=0", abs(r["dG_apply_free_kJmol"]) < 1e-10,
            f"dG={r['dG_apply_free_kJmol']:.2e}")

    # ---- T2 restraint costs free energy, monotone in k ------------------
    print("\n[T2] dG_apply_free > 0 and monotone increasing in k")
    ks = [1.0, 10.0, 100.0, 1000.0]
    vals = []
    for k in ks:
        c = AngularCoord("th", "polar", k, math.pi / 2)
        vals.append(free_apply_free_energy(c, T)["dG_apply_free_kJmol"])
    rep("all positive", all(v > 0 for v in vals),
        " ".join(f"{v:.2f}" for v in vals))
    rep("monotone in k", all(b > a for a, b in zip(vals, vals[1:])))

    # ---- T3 narrow-restraint analytic cross-check ----------------------
    print("\n[T3] Quadrature vs narrow-restraint Gaussian formula")
    for kind, ctr in (("polar", math.pi / 2), ("azimuthal", 1.0)):
        c = AngularCoord("x", kind, 2000.0, ctr)
        r = free_apply_free_energy(c, T)
        ap = r["narrow_restraint_approx_kJmol"]
        rep(f"{kind} narrow limit", ap is not None
            and abs(r["dG_apply_free_kJmol"] - ap) < 0.05,
            f"quad={r['dG_apply_free_kJmol']:.4f} approx={ap:.4f}")

    # ---- T4 measure totals: 4pi and 8pi^2 ------------------------------
    print("\n[T4] Free measures assemble to 4pi and 8pi^2")
    pa = AngularCoord("th", "polar", 0.0, 0.0)
    az = AngularCoord("ph", "azimuthal", 0.0, 0.0)
    _, _, t1 = pa.grid(); _, _, t2 = az.grid()
    rep("positional angular = 4pi", abs(t1 * t2 - FOUR_PI) < 1e-9,
        f"{t1*t2:.6f} vs {FOUR_PI:.6f}")
    ep = AngularCoord("Th", "euler_polar", 0.0, 0.0)
    ea = AngularCoord("Ph", "euler_azimuthal", 0.0, 0.0)
    et = AngularCoord("Ps", "euler_twist", 0.0, 0.0)
    _, _, u1 = ep.grid(); _, _, u2 = ea.grid(); _, _, u3 = et.grid()
    rep("orientation = 8pi^2", abs(u1 * u2 * u3 - EIGHT_PI2) < 1e-8,
        f"{u1*u2*u3:.6f} vs {EIGHT_PI2:.6f}")

    # ---- T5 analytic cycle: harmonic bound + harmonic restraint --------
    print("\n[T5] Full cycle on an ANALYTICALLY solvable angular model")
    # intrinsic bound PMF W0(x) = 0.5*k0*(x-x0)^2 on x in [0,pi], measure sin x
    k0, x0 = 300.0, 1.2
    kr, xr = 800.0, 1.25
    c = AngularCoord("th", "polar", kr, xr)
    xi, mu, tot = c.grid(20001)
    U = c.potential(xi)
    W0 = 0.5 * k0 * (xi - x0) ** 2
    beta = 1.0 / kT
    Zb = float(np.trapz(mu * np.exp(-beta * W0), xi))
    ZbR = float(np.trapz(mu * np.exp(-beta * (W0 + U)), xi))
    dG_bound_exact = -kT * math.log(ZbR / Zb)
    Wmeas = W0 + U
    got = bound_apply_free_energy(xi, Wmeas, c, T)
    rep("dG_apply_bound matches analytic",
        abs(got["dG_apply_bound_kJmol"] - dG_bound_exact) < 0.02,
        f"got={got['dG_apply_bound_kJmol']:.4f} exact={dG_bound_exact:.4f}")
    rep("coverage OK for a well-confined bound state", got["coverage_ok"],
        f"tail={got['released_tail_fraction']:.4f}")
    # cycle: dG_bind = dG_bind^R + (apply_free - apply_bound)
    fa = free_apply_free_energy(c, T)["dG_apply_free_kJmol"]
    Zf = float(np.trapz(mu, xi))
    ZfR = float(np.trapz(mu * np.exp(-beta * U), xi))
    dG_bind_R = -kT * math.log(ZbR / ZfR)
    dG_bind_true = -kT * math.log(Zb / Zf)
    dG_cycle = dG_bind_R + (fa - got["dG_apply_bound_kJmol"])
    rep("cycle recovers the true binding free energy",
        abs(dG_cycle - dG_bind_true) < 0.03,
        f"cycle={dG_cycle:.4f} true={dG_bind_true:.4f}")

    # ---- T6 coverage failure is DETECTED -------------------------------
    print("\n[T6] Coverage failure must be flagged, not silently used")
    xi2 = np.linspace(1.10, 1.30, 401)      # narrow window
    c2 = AngularCoord("th", "polar", 5.0, 1.2)   # very weak restraint
    Wm2 = 0.5 * 300.0 * (xi2 - 1.2) ** 2 + c2.potential(xi2)
    bad = bound_apply_free_energy(xi2, Wm2, c2, T)
    rep("weak restraint + narrow window flagged",
        not bad["coverage_ok"],
        f"tail={bad['released_tail_fraction']:.3f}")
    rep("warning text present", len(bad["warning"]) > 0)

    # ---- T7 4pi bookkeeping -------------------------------------------
    print("\n[T7] 4pi double-counting bookkeeping")
    r = np.linspace(0.05, 2.5, 300)
    Wr = np.where(r < 0.6, -12.0, 0.0)
    a_free = radial_binding_term(r, Wr, T, 0.75, (1.6, 2.4), False,
                                 "bulk_flat")
    a_rest = radial_binding_term(r, Wr, T, 0.75, (1.6, 2.4), True,
                                 "bulk_flat")
    rep("free-angular carries 4pi", abs(a_free["angular_prefactor"] - FOUR_PI) < 1e-12)
    rep("restrained-angular carries 1", abs(a_rest["angular_prefactor"] - 1.0) < 1e-12)
    d = a_rest["dG_radial_restrained_kJmol"] - a_free["dG_radial_restrained_kJmol"]
    rep("difference == kT ln(4pi)", abs(d - kT * math.log(FOUR_PI)) < 1e-8,
        f"diff={d:.6f} kT ln4pi={kT*math.log(FOUR_PI):.6f}")

    # ---- T8 direct dCp recovers a known slope --------------------------
    print("\n[T8] Direct dCp from dH(T) recovers a KNOWN slope")
    dCp_true, dH0_true, dS0_true, T0 = -0.65, -45.0, -0.10, 310.0
    Ts = [300.0, 305.0, 310.0, 315.0, 320.0]
    rng = np.random.default_rng(0)
    pts = []
    for Ti in Ts:
        H = dH0_true + dCp_true * (Ti - T0)
        S = dS0_true + dCp_true * math.log(Ti / T0)
        G = H - Ti * S
        Hn = H + rng.normal(0, 0.4)
        Gn = G + rng.normal(0, 0.3)
        pts.append(ThermoPoint(Ti, Hn, 0.4, Gn, 0.3, "calorimetric"))
    dc = direct_cp_from_enthalpy(pts, T0)
    rep("dCp within 3 stderr of truth",
        abs(dc.dCp_kJmol_per_K - dCp_true) < 3 * dc.dCp_stderr,
        f"dCp={dc.dCp_kJmol_per_K:+.4f}±{dc.dCp_stderr:.4f} "
        f"true={dCp_true:+.4f}")
    rep("dH0 within 3 stderr", abs(dc.dH0_kJmol - dH0_true) < 3 * dc.dH0_stderr,
        f"dH0={dc.dH0_kJmol:+.3f}±{dc.dH0_stderr:.3f}")
    rep("dG(T) reconstruction consistent", dc.consistency_ok,
        f"max dev={max(abs(x['deviation']) for x in dc.dG_reconstructed):.3f}")
    print(f"      lever arm = {dc.lever_arm_K:.1f} K   R2={dc.r_squared:.4f}")

    # ---- T9 direct beats van 't Hoff on the SAME data ------------------
    print("\n[T9] Direct dH vs van 't Hoff dH on IDENTICAL data")
    vh = vant_hoff_from_dG(pts)
    H_true_310 = dH0_true
    err_direct = abs(dc.dH0_kJmol - H_true_310)
    err_vh = abs(vh["dH_vant_hoff_kJmol"] - H_true_310)
    print(f"      direct      dH = {dc.dH0_kJmol:+.3f} ± {dc.dH0_stderr:.3f} "
          f"kJ/mol   (lever {dc.lever_arm_K:.1f} K)")
    print(f"      van 't Hoff dH = {vh['dH_vant_hoff_kJmol']:+.3f} ± "
          f"{vh['dH_vant_hoff_stderr']:.3f} kJ/mol   "
          f"(lever {vh['lever_arm_inv_K']:.2e} 1/K)")
    print(f"      truth          = {H_true_310:+.3f} kJ/mol")
    rep("direct dH error smaller than van 't Hoff", err_direct < err_vh,
        f"|err| direct={err_direct:.3f} vs vh={err_vh:.3f}")
    rep("direct stderr smaller than van 't Hoff",
        dc.dH0_stderr < vh["dH_vant_hoff_stderr"],
        f"{dc.dH0_stderr:.3f} vs {vh['dH_vant_hoff_stderr']:.3f}")

    # ---- T10 provenance and mixed-source warnings ----------------------
    print("\n[T10] Provenance warnings")
    md_pts = [ThermoPoint(p.T_K, p.dH_kJmol, p.dH_stderr, p.dG_kJmol,
                          p.dG_stderr, "MD potential energy") for p in pts]
    dc_md = direct_cp_from_enthalpy(md_pts, T0)
    rep("MD-derived dH flagged", any("MD" in w or "potential" in w
                                    for w in dc_md.warnings))
    mixed = pts[:3] + md_pts[3:]
    dc_mx = direct_cp_from_enthalpy(mixed, T0)
    rep("mixed provenance flagged", any("Mixed" in w for w in dc_mx.warnings))

    # ---- T11 too-few-points and short-span guards ----------------------
    print("\n[T11] Guards")
    try:
        direct_cp_from_enthalpy(pts[:2], T0)
        rep("< 3 points rejected", False)
    except ValueError:
        rep("< 3 points rejected", True)
    short = [ThermoPoint(310.0 + 0.5 * i, -45.0 - 0.3 * i, 0.4)
             for i in range(3)]
    ds = direct_cp_from_enthalpy(short, 310.0)
    rep("short span flagged", any("short" in w for w in ds.warnings))

    print("\n" + "=" * 92)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 92)
    print("""
  WHAT THESE TESTS PROVE
    T1  a zero restraint costs exactly zero free energy
    T2  the restraint cost is positive and monotone in the force constant
    T3  the quadrature agrees with the narrow-restraint Gaussian formula
    T4  the free measures assemble to exactly 4pi and 8pi^2
    T5  the FULL cycle recovers an analytically known binding free energy
    T6  insufficient bound-PMF coverage is DETECTED and flagged
    T7  the 4pi bookkeeping difference is exactly kT ln(4pi) — no double count
    T8  direct dCp recovers a known slope, and dG(T) reconstructs consistently
    T9  on identical data, the direct dH has a smaller error AND a smaller
        error bar than van 't Hoff — measured, not asserted
    T10 MD-derived and mixed-provenance enthalpies are flagged
    T11 too-few-point and short-span inputs are guarded

  WHAT IS NOT PROVEN
    * the separable radial x angular factorisation is an approximation
    * symmetry numbers must be supplied and verified by the user
    * an MD potential-energy difference is not the experimental enthalpy
    * dCp is assumed constant over the fitted range
    * no comparison against a published absolute-binding benchmark
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Woo–Roux + direct ΔCp v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--thermo", default=None,
                    help="JSON list of {T_K,dH,dH_se,dG,dG_se,provenance}")
    ap.add_argument("--T0", type=float, default=310.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if self_test() else 1)
    if a.thermo:
        with open(a.thermo) as f:
            raw = json.load(f)
        pts = [ThermoPoint(float(d["T_K"]), float(d["dH"]),
                           float(d.get("dH_se", float("nan"))),
                           float(d.get("dG", float("nan"))),
                           float(d.get("dG_se", float("nan"))),
                           d.get("provenance", "unspecified")) for d in raw]
        dc = direct_cp_from_enthalpy(pts, a.T0)
        vh = vant_hoff_from_dG(pts) if all(
            math.isfinite(p.dG_kJmol) for p in pts) else None
        print(f"  direct  dCp = {dc.dCp_kJmol_per_K:+.4f} ± {dc.dCp_stderr:.4f} kJ/mol/K")
        print(f"  direct  dH0 = {dc.dH0_kJmol:+.3f} ± {dc.dH0_stderr:.3f} kJ/mol")
        if dc.dS0_kJmol_per_K is not None:
            print(f"  direct  dS0 = {dc.dS0_kJmol_per_K:+.5f} kJ/mol/K")
        print(f"  lever arm   = {dc.lever_arm_K:.1f} K   R2={dc.r_squared:.4f}")
        print(f"  dG(T) consistency: {dc.consistency_ok}")
        for w in dc.warnings:
            print(f"  WARNING: {w}")
        if vh:
            print(f"  van 't Hoff dH = {vh['dH_vant_hoff_kJmol']:+.3f} ± "
                  f"{vh['dH_vant_hoff_stderr']:.3f} kJ/mol "
                  f"(lever {vh['lever_arm_inv_K']:.2e} 1/K) — for comparison")
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"direct": asdict(dc), "vant_hoff": vh}, f, indent=2)
        return
    ap.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
