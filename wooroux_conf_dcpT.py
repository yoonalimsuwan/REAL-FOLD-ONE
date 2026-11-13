#!/usr/bin/env python3
"""
=============================================================================
WOO–ROUX STEP 6: CONFORMATIONAL RESTRAINT FOR A FLEXIBLE LIGAND
+ TEMPERATURE-DEPENDENT HEAT CAPACITY  dCp(T) = a + b*T  WITH MODEL SELECTION
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

PART A — CONFORMATIONAL RESTRAINT (Woo–Roux step 6)
  For a FLEXIBLE ligand the absolute binding free energy needs one more
  term beyond the radial and angular corrections: the free-energy cost of
  restricting the ligand's internal conformational degrees of freedom.

  Full cycle, with ALL restraints (radial + angular + conformational):

      A (free, unrestrained, standard state)
        --apply all restraints-->  A_R
        --bind (restrained PMF)--> B_R
        --release all restraints--> B (bound, unrestrained)

      dG_bind = dG_r^(R)
                + sum_angular [ dG_apply_free - dG_apply_bound ]
                + [ dG_conf_apply_free - dG_conf_apply_bound ]

  IMPORTANT: internal coordinates carry NO standard-state volume factor.
  V° = 1.660539 nm^3 applies only to the translational part. The
  conformational term is a pure ratio of conformational partition
  functions. Adding a V° here would be a dimensional error.

  SIGN / PHYSICS
    A flexible ligand in solution samples many conformers. Restraining it
    to the bound conformation costs free energy (dG_conf_apply_free > 0,
    often several kT). In the bound state the ligand is already confined,
    so dG_conf_apply_bound is small. The net correction is therefore
    POSITIVE and makes binding WEAKER than a naive restrained estimate.
    This is the conformational-entropy penalty of binding, not a
    favourable term.

  WHAT MUST BE SUPPLIED vs WHAT CAN BE ASSUMED
    * dG_conf_apply_bound  : from a bound-state PMF along the internal
                             coordinate, measured WITH the restraint on.
    * dG_conf_apply_free   : from a free-state (solution) PMF along the
                             SAME coordinate. An analytic "free rotor"
                             reference is provided but is an IDEALISATION
                             (uniform torsional measure, no intrinsic
                             barrier). Real ligands have torsional
                             barriers; using the ideal reference then
                             OVERESTIMATES the penalty. The module flags
                             which reference was used.
    * Coupled torsions     : the sum-over-coordinates form assumes
                             approximate independence. A joint PMF is
                             required otherwise. Flagged.

  MEASURE CAVEAT
    For a single torsion in a rigid-rotor-like approximation the
    configurational measure is uniform in the dihedral. For coupled
    internal coordinates the full configurational integral carries a
    nontrivial metric determinant (Fixman-type correction). This module
    does NOT compute that determinant; it is a named, unaddressed
    approximation, not a silent one.

PART B — dCp(T) = a + b*T WITH MODEL SELECTION
  Integrating dCp(T) = a + b T from T0:

      dH(T) = dH0 + a (T - T0) + (b/2)(T^2 - T0^2)
      dS(T) = dS0 + a ln(T/T0) + b (T - T0)
      dG(T) = dH(T) - T dS(T)

  Both dH and dG are LINEAR in the parameter vector (dH0, dS0, a, b),
  so a joint weighted linear least-squares fit is exact — no nonlinear
  optimiser is needed. Design rows:

      dH at T : [ 1,  0,  (T-T0),                 (T^2-T0^2)/2                ]
      dG at T : [ 1, -T,  (T-T0) - T ln(T/T0),    (T^2-T0^2)/2 - T (T-T0)     ]

  Nested model hierarchy:
      M0 : dCp = 0        params (dH0, dS0)              k=2
      M1 : dCp = a        params (dH0, dS0, a)           k=3
      M2 : dCp = a + bT   params (dH0, dS0, a, b)        k=4

  Selection tools, all reported together because no single one is
  decisive on a handful of points:
      AIC / AICc / BIC   (AICc mandatory at small n)
      Akaike weights
      nested F-test      (M0 vs M1, M1 vs M2)
      leave-one-out CV   (predictive, not in-sample)
      identifiability    (design condition number, a-b correlation)

HONEST SCOPE
  ❌ Three temperatures cannot determine four parameters. M2 is REFUSED
     when degrees of freedom <= 0, not fitted and reported.
  ❌ Model selection on 3-6 points is weak evidence. Akaike weights are
     reported so the reader sees the ambiguity instead of a verdict.
  ❌ b (the temperature slope of dCp) is a CURVATURE in dH(T). It needs a
     wide temperature range and precise enthalpies. When it is not
     identifiable, that is flagged, not hidden behind a fitted number.
  ❌ Calorimetric dH and MD potential-energy differences are not the same
     quantity; mixing them is flagged.
  ❌ No Fixman/metric-determinant correction for coupled internal coordinates.
  ❌ No experimental validation is performed here.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("wooroux_conf")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

KB = 0.008314462618            # kJ mol^-1 K^-1
NA = 6.02214076e23
V_STD_NM3 = 1.0e27 / NA        # 1.660539 nm^3, TRANSLATIONAL only


# =============================================================================
# PART A §1  CONFORMATIONAL COORDINATES AND RESTRAINTS
# =============================================================================

@dataclass
class ConfCoord:
    """
    An internal (conformational) coordinate.

    kind:
      "torsion"  zeta in [-pi, pi), uniform measure, total 2*pi
      "angle"    zeta in [0, pi],   measure sin(zeta), total 2
      "distance" zeta in [lo, hi],  uniform measure (internal distance;
                 NOT a COM separation, so NO V° factor)
      "rmsd"     zeta in [0, hi],   measure NOT uniform in general.
                 The ideal reference is INVALID for rmsd; a sampled
                 free-state PMF is REQUIRED. Enforced below.
    """
    name: str
    kind: str
    k: float                       # kJ/mol/unit^2
    center: float
    lo: float = -math.pi
    hi: float = math.pi
    periodic: bool = False

    def __post_init__(self):
        if self.kind == "torsion":
            self.lo, self.hi, self.periodic = -math.pi, math.pi, True
        elif self.kind == "angle":
            self.lo, self.hi, self.periodic = 0.0, math.pi, False
        elif self.kind in ("distance", "rmsd"):
            if not (self.hi > self.lo):
                raise ValueError(f"{self.kind}: need hi > lo")
            self.periodic = False
        else:
            raise ValueError(f"unknown kind {self.kind!r}")

    def grid(self, n: int = 4001) -> Tuple[np.ndarray, np.ndarray, float]:
        z = np.linspace(self.lo, self.hi, n)
        if self.kind == "angle":
            mu = np.sin(z)
            tot = 2.0
        else:
            mu = np.ones_like(z)
            tot = self.hi - self.lo
        return z, mu, tot

    def potential(self, z: np.ndarray) -> np.ndarray:
        d = z - self.center
        if self.periodic:
            d = (d + math.pi) % (2.0 * math.pi) - math.pi
        return 0.5 * self.k * d ** 2

    @property
    def ideal_reference_valid(self) -> bool:
        """rmsd has no uniform-measure ideal reference."""
        return self.kind in ("torsion", "angle", "distance")


def conf_apply_free_ideal(coord: ConfCoord, T: float,
                          n_grid: int = 4001) -> Dict[str, Any]:
    """
    IDEALISED free-state cost of applying the conformational restraint,
    assuming the unrestrained coordinate is distributed according to its
    bare measure (free rotor / free angle / uniform internal distance).

        dG = -kT ln[ (1/mu_tot) int mu(z) exp(-beta U(z)) dz ]

    This OVERESTIMATES the penalty for a real ligand with intrinsic
    torsional barriers, because the true free-state distribution is
    already partly confined. Prefer conf_apply_from_pmf with a sampled
    solution PMF.
    """
    if not coord.ideal_reference_valid:
        raise ValueError(f"{coord.kind}: no valid ideal reference; "
                         "supply a sampled free-state PMF instead")
    z, mu, tot = coord.grid(n_grid)
    U = coord.potential(z)
    beta = 1.0 / (KB * T)
    Zr = float(np.trapz(mu * np.exp(-beta * U), z))
    Z0 = float(np.trapz(mu, z))
    if Zr <= 0 or Z0 <= 0:
        raise ValueError("zero measure")
    dG = -KB * T * math.log(Zr / Z0)
    sig = math.sqrt(KB * T / coord.k) if coord.k > 0 else float("inf")
    approx = None
    if coord.kind == "torsion" and coord.k > 0 and sig < 0.4:
        approx = -KB * T * math.log(sig / math.sqrt(2.0 * math.pi))
    return {"name": coord.name, "kind": coord.kind, "T_K": T,
            "dG_apply_free_kJmol": dG,
            "reference": "IDEAL free rotor / uniform measure",
            "accessible_fraction": Zr / Z0,
            "restraint_width": sig,
            "narrow_limit_approx_kJmol": approx,
            "warning": ("ideal reference: real torsional barriers make the "
                        "true penalty SMALLER than this value")}


def conf_apply_from_pmf(z: np.ndarray, W: np.ndarray, coord: ConfCoord,
                        T: float, restraint_on: bool = True,
                        coverage_tol: float = 0.02) -> Dict[str, Any]:
    """
    Cost of applying the restraint given an intrinsic (unrestrained) PMF
    W(z) along the coordinate, from simulation.

    If restraint_on=True, W is the MEASURED PMF with the restraint already
    applied; the intrinsic PMF is recovered as W - U before integrating.

        dG_apply = -kT ln[ int mu e^{-beta(W_int+U)} / int mu e^{-beta W_int} ]

    Coverage: if the UNRESTRAINED integrand carries more than coverage_tol
    of its weight in the outer 5% of the sampled range, the PMF does not
    cover the released state and the number is not usable.
    """
    z = np.asarray(z, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if z.shape != W.shape or z.ndim != 1:
        raise ValueError("z and W must be 1-D of equal length")
    mu = np.sin(z) if coord.kind == "angle" else np.ones_like(z)
    U = coord.potential(z)
    W_int = W - U if restraint_on else W.copy()
    beta = 1.0 / (KB * T)
    W_int = W_int - np.nanmin(W_int)
    num = mu * np.exp(-beta * (W_int + U))
    den = mu * np.exp(-beta * W_int)
    Zn = float(np.trapz(num, z))
    Zd = float(np.trapz(den, z))
    if Zn <= 0 or Zd <= 0:
        raise ValueError("zero partition function")
    dG = -KB * T * math.log(Zn / Zd)
    n = len(z)
    e = max(1, int(0.05 * n))
    tail = float(np.trapz(den[:e], z[:e]) + np.trapz(den[-e:], z[-e:]))
    frac = tail / Zd
    ok = frac < coverage_tol
    return {"name": coord.name, "kind": coord.kind, "T_K": T,
            "dG_apply_kJmol": dG,
            "released_tail_fraction": frac,
            "coverage_ok": bool(ok),
            "source": "sampled PMF",
            "warning": ("" if ok else
                        "PMF does not cover the released state — extend "
                        "the sampling window or weaken the restraint")}


def rotamer_penalty(n_rotamers_free: int, n_rotamers_bound: int = 1,
                    T: float = 310.0) -> Dict[str, Any]:
    """
    Crude counting estimate of the conformational penalty:
        dG ~ kT ln(N_free / N_bound)
    Assumes equally populated, non-interacting rotamers and a single bound
    rotamer. This is an ORDER-OF-MAGNITUDE estimate, not a measurement.
    """
    if n_rotamers_free < 1 or n_rotamers_bound < 1:
        raise ValueError("rotamer counts must be >= 1")
    dG = KB * T * math.log(n_rotamers_free / n_rotamers_bound)
    return {"T_K": T, "N_free": n_rotamers_free, "N_bound": n_rotamers_bound,
            "dG_conf_estimate_kJmol": dG,
            "method": "rotamer counting (equal populations assumed)",
            "warning": "estimate only; not a substitute for a PMF"}


# =============================================================================
# PART A §2  FULL WOO–ROUX ASSEMBLY WITH THE CONFORMATIONAL TERM
# =============================================================================

@dataclass
class WooRouxFullResult:
    T_K: float
    dG_radial_restrained: float
    angular_correction: float
    conformational_correction: float
    dG_bind_standard: float
    terms: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    all_coverage_ok: bool = True


def wooroux_full(radial_dG: float, T: float,
                 angular_terms: Sequence[Dict[str, Any]] = (),
                 conf_free: Sequence[Dict[str, Any]] = (),
                 conf_bound: Sequence[Dict[str, Any]] = (),
                 independent_torsions: bool = True) -> WooRouxFullResult:
    """
    Assemble the absolute binding free energy including step 6.

    radial_dG      : dG_r^(R) from the restrained radial PMF, already
                     carrying the correct translational V° factor and the
                     correct 4pi bookkeeping (see wooroux_dcp.py).
    angular_terms  : list of dicts with 'dG_apply_free_kJmol' and
                     'dG_apply_bound_kJmol' (and optionally coverage_ok).
    conf_free      : list of dicts with 'dG_apply_free_kJmol' or
                     'dG_apply_kJmol' for each internal coordinate,
                     evaluated in the FREE state.
    conf_bound     : matching list for the BOUND state. If a coordinate is
                     absent, its bound cost is taken as 0 and flagged.
    """
    warns: List[str] = []
    terms: List[Dict[str, Any]] = []
    ok = True

    ang = 0.0
    for t in angular_terms:
        f = float(t.get("dG_apply_free_kJmol", 0.0))
        b = float(t.get("dG_apply_bound_kJmol", 0.0))
        d = f - b
        ang += d
        ok &= bool(t.get("coverage_ok", True))
        terms.append({"stage": "angular", "name": t.get("name", "?"),
                      "free": f, "bound": b, "correction": d})

    if conf_free and not independent_torsions:
        warns.append("Coupled internal coordinates: summing per-coordinate "
                     "terms assumes independence. A JOINT PMF is required "
                     "for a controlled result. No Fixman/metric-determinant "
                     "correction is applied.")

    bmap = {t.get("name"): t for t in conf_bound}
    conf = 0.0
    for t in conf_free:
        nm = t.get("name", "?")
        f = float(t.get("dG_apply_free_kJmol",
                        t.get("dG_apply_kJmol", 0.0)))
        if nm in bmap:
            tb = bmap[nm]
            b = float(tb.get("dG_apply_bound_kJmol",
                             tb.get("dG_apply_kJmol", 0.0)))
            ok &= bool(tb.get("coverage_ok", True))
            src = "bound PMF"
        else:
            b = 0.0
            src = "ASSUMED ZERO (no bound PMF supplied)"
            warns.append(f"[conf:{nm}] bound cost assumed 0 — the bound "
                         "state is assumed to confine this coordinate "
                         "completely. Approximation, not measurement.")
        ok &= bool(t.get("coverage_ok", True))
        if "IDEAL" in str(t.get("reference", "")):
            warns.append(f"[conf:{nm}] free-state cost uses the IDEAL "
                         "uniform-measure reference; the true penalty is "
                         "smaller if the ligand has torsional barriers.")
        d = f - b
        conf += d
        terms.append({"stage": "conformational", "name": nm,
                      "free": f, "bound": b, "bound_source": src,
                      "correction": d})

    dG = radial_dG + ang + conf
    return WooRouxFullResult(
        T_K=T, dG_radial_restrained=radial_dG,
        angular_correction=ang, conformational_correction=conf,
        dG_bind_standard=dG, terms=terms, warnings=warns,
        all_coverage_ok=bool(ok))


# =============================================================================
# PART B §1  dCp(T) MODELS AND JOINT DESIGN MATRIX
# =============================================================================

@dataclass
class ThermoObs:
    T_K: float
    kind: str                  # "dH" | "dG"
    value: float               # kJ/mol
    sigma: float = float("nan")
    provenance: str = "unspecified"


MODELS: Dict[str, Dict[str, Any]] = {
    "M0": {"cols": (0, 1), "label": "dCp = 0", "k": 2},
    "M1": {"cols": (0, 1, 2), "label": "dCp = a", "k": 3},
    "M2": {"cols": (0, 1, 2, 3), "label": "dCp = a + b*T", "k": 4},
}


def design_row(T: float, kind: str, T0: float) -> np.ndarray:
    """Row of the joint design matrix for parameters (dH0, dS0, a, b)."""
    dT = T - T0
    q = 0.5 * (T * T - T0 * T0)
    if kind == "dH":
        return np.array([1.0, 0.0, dT, q], dtype=np.float64)
    if kind == "dG":
        return np.array([1.0, -T, dT - T * math.log(T / T0), q - T * dT],
                        dtype=np.float64)
    raise ValueError(f"kind must be dH|dG, got {kind!r}")


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray
         ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Weighted LS. Returns beta, cov, chi2, cond."""
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    XtX = Xw.T @ Xw
    cond = float(np.linalg.cond(Xw))
    try:
        cov = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    r = y - X @ beta
    chi2 = float(np.sum(w * r * r))
    return beta, cov, chi2, cond


def fit_dcp_models(obs: Sequence[ThermoObs], T0: float = 310.0,
                   sigmas_known: bool = True,
                   min_dof: int = 1) -> Dict[str, Any]:
    """
    Fit M0/M1/M2 jointly to dH(T) and dG(T), then compare.

    sigmas_known=True  -> AIC = chi2 + 2k  (Gaussian, known variances)
    sigmas_known=False -> AIC = n ln(RSS/n) + 2k  (variance estimated)
    AICc is used for ranking in both cases; it is mandatory at small n.
    """
    if len(obs) < 3:
        raise ValueError("need >= 3 observations")
    prov = {o.provenance for o in obs}
    warns: List[str] = []
    if len(prov) > 1:
        warns.append(f"Mixed provenance {sorted(prov)} — do not mix "
                     "calorimetric and model-derived enthalpies.")
    if any("md" in p.lower() or "potential" in p.lower() for p in prov):
        warns.append("At least one value is MD/potential-energy derived; "
                     "that is not the experimental enthalpy.")
    kinds = {o.kind for o in obs}
    if "dH" not in kinds:
        warns.append("No dH observations: dCp is being inferred from dG(T) "
                     "curvature only, which reintroduces the van 't Hoff "
                     "lever-arm problem.")
    Ts = np.array([o.T_K for o in obs])
    span = float(Ts.max() - Ts.min())
    if span < 15.0:
        warns.append(f"Temperature span {span:.1f} K is short; the dCp "
                     "slope b will be poorly determined.")

    X_full = np.vstack([design_row(o.T_K, o.kind, T0) for o in obs])
    y = np.array([o.value for o in obs], dtype=np.float64)
    sig = np.array([o.sigma if (o.sigma and math.isfinite(o.sigma) and o.sigma > 0)
                    else 1.0 for o in obs], dtype=np.float64)
    w = 1.0 / sig ** 2
    n = len(obs)

    fits: Dict[str, Any] = {}
    for name, spec in MODELS.items():
        cols = spec["cols"]
        k = spec["k"]
        dof = n - k
        X = X_full[:, cols]
        if dof < min_dof:
            fits[name] = {"label": spec["label"], "k": k, "dof": dof,
                          "status": "REFUSED",
                          "reason": f"n={n} observations, k={k} parameters "
                                    f"-> dof={dof} < {min_dof}"}
            continue
        beta, cov, chi2, cond = _wls(X, y, w)
        rss = float(np.sum((y - X @ beta) ** 2))
        if sigmas_known:
            aic = chi2 + 2.0 * k
            bic = chi2 + k * math.log(n)
        else:
            aic = n * math.log(max(rss / n, 1e-300)) + 2.0 * k
            bic = n * math.log(max(rss / n, 1e-300)) + k * math.log(n)
        aicc = (aic + 2.0 * k * (k + 1) / (n - k - 1)
                if n - k - 1 > 0 else float("inf"))
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        # a-b identifiability
        ab_corr = None
        if name == "M2" and cov.shape == (4, 4):
            va, vb = cov[2, 2], cov[3, 3]
            if va > 0 and vb > 0:
                ab_corr = float(cov[2, 3] / math.sqrt(va * vb))
        params = dict(zip(("dH0", "dS0", "a", "b"), (None,) * 4))
        errs = dict(params)
        labels = ("dH0", "dS0", "a", "b")
        for j, c in enumerate(cols):
            params[labels[c]] = float(beta[j])
            errs[labels[c]] = float(se[j])
        dcp_T0 = None
        dcp_T0_se = None
        if params["a"] is not None:
            dcp_T0 = params["a"] + (params["b"] or 0.0) * T0
            va = errs["a"] ** 2
            vb = (errs["b"] ** 2) if errs["b"] is not None else 0.0
            cab = cov[2, 3] if (name == "M2" and cov.shape == (4, 4)) else 0.0
            dcp_T0_se = math.sqrt(max(va + T0 * T0 * vb + 2 * T0 * cab, 0.0))
        fits[name] = {"label": spec["label"], "k": k, "dof": dof,
                      "status": "FITTED", "params": params, "stderr": errs,
                      "chi2": chi2, "rss": rss, "reduced_chi2": chi2 / dof,
                      "AIC": aic, "AICc": aicc, "BIC": bic,
                      "design_cond": cond, "ab_correlation": ab_corr,
                      "dCp_at_T0": dcp_T0, "dCp_at_T0_stderr": dcp_T0_se}

    # ---- ranking on AICc ----
    usable = {k: v for k, v in fits.items() if v["status"] == "FITTED"
              and math.isfinite(v["AICc"])}
    ranking: List[Dict[str, Any]] = []
    if usable:
        best = min(v["AICc"] for v in usable.values())
        raw = {k: math.exp(-0.5 * (v["AICc"] - best)) for k, v in usable.items()}
        Z = sum(raw.values())
        for k, v in sorted(usable.items(), key=lambda kv: kv[1]["AICc"]):
            ranking.append({"model": k, "label": v["label"],
                            "AICc": v["AICc"],
                            "delta_AICc": v["AICc"] - best,
                            "akaike_weight": raw[k] / Z})
        top = ranking[0]
        if len(ranking) > 1 and ranking[1]["delta_AICc"] < 2.0:
            warns.append(f"Model selection is AMBIGUOUS: {top['model']} vs "
                         f"{ranking[1]['model']} differ by "
                         f"{ranking[1]['delta_AICc']:.2f} AICc units (<2). "
                         "Do not report a single model as decided.")
    for k, v in fits.items():
        if v["status"] == "FITTED" and v.get("ab_correlation") is not None:
            if abs(v["ab_correlation"]) > 0.99:
                warns.append(f"{k}: a and b are almost perfectly correlated "
                             f"(r={v['ab_correlation']:.4f}); b is NOT "
                             "identifiable from these data.")
        if v["status"] == "FITTED" and v["design_cond"] > 1e6:
            warns.append(f"{k}: design condition number "
                         f"{v['design_cond']:.2e} — ill-conditioned fit.")

    # ---- nested F-tests ----
    ftests: List[Dict[str, Any]] = []
    for lo, hi in (("M0", "M1"), ("M1", "M2")):
        a, b = fits.get(lo), fits.get(hi)
        if not (a and b and a["status"] == b["status"] == "FITTED"):
            continue
        dk = b["k"] - a["k"]
        dof2 = b["dof"]
        if dof2 <= 0 or b["chi2"] <= 0:
            continue
        F = ((a["chi2"] - b["chi2"]) / dk) / (b["chi2"] / dof2)
        ftests.append({"nested": f"{lo}->{hi}", "F": float(F),
                       "df1": dk, "df2": dof2,
                       "note": "compare against F(df1,df2); large F favours "
                               "the richer model"})

    # ---- leave-one-out CV ----
    loo: Dict[str, Any] = {}
    for name, spec in MODELS.items():
        if fits.get(name, {}).get("status") != "FITTED":
            continue
        cols = spec["cols"]
        X = X_full[:, cols]
        err = 0.0
        for i in range(n):
            m = np.ones(n, dtype=bool); m[i] = False
            try:
                bi, *_ = np.linalg.lstsq(X[m] * np.sqrt(w[m])[:, None],
                                         y[m] * np.sqrt(w[m]), rcond=None)
            except np.linalg.LinAlgError:
                err = float("inf"); break
            pred = float(X[i] @ bi)
            err += ((y[i] - pred) / sig[i]) ** 2
        loo[name] = {"loo_chi2": float(err),
                     "loo_rmse_standardised": math.sqrt(err / n)
                     if math.isfinite(err) else float("inf")}

    return {"T0_K": T0, "n_obs": n, "T_span_K": span,
            "sigmas_known": sigmas_known, "fits": fits,
            "ranking": ranking, "nested_F": ftests, "loo_cv": loo,
            "warnings": warns}


def report(res: Dict[str, Any]) -> None:
    print("=" * 96)
    print(f"  dCp(T) MODEL SELECTION   T0 = {res['T0_K']:.1f} K   "
          f"n = {res['n_obs']}   span = {res['T_span_K']:.1f} K")
    print("=" * 96)
    print(f"  {'model':<6}{'label':<18}{'k':>3}{'dof':>5}{'chi2':>12}"
          f"{'AICc':>12}{'dAICc':>9}{'w':>8}  status")
    print("-" * 96)
    wmap = {r["model"]: r for r in res["ranking"]}
    for name, f in res["fits"].items():
        if f["status"] != "FITTED":
            print(f"  {name:<6}{f['label']:<18}{f['k']:>3}{f['dof']:>5}"
                  f"{'—':>12}{'—':>12}{'—':>9}{'—':>8}  REFUSED: {f['reason']}")
            continue
        r = wmap.get(name, {})
        print(f"  {name:<6}{f['label']:<18}{f['k']:>3}{f['dof']:>5}"
              f"{f['chi2']:>12.3f}{f['AICc']:>12.3f}"
              f"{r.get('delta_AICc', float('nan')):>9.2f}"
              f"{r.get('akaike_weight', float('nan')):>8.3f}  FITTED")
    print("-" * 96)
    for name, f in res["fits"].items():
        if f["status"] != "FITTED":
            continue
        p, e = f["params"], f["stderr"]
        print(f"  {name}: dH0={p['dH0']:+.3f}±{e['dH0']:.3f}  "
              f"dS0={p['dS0']:+.5f}±{e['dS0']:.5f}")
        if p["a"] is not None:
            print(f"        a={p['a']:+.5f}±{e['a']:.5f}"
                  + (f"   b={p['b']:+.6f}±{e['b']:.6f}"
                     if p["b"] is not None else ""))
            print(f"        dCp(T0)={f['dCp_at_T0']:+.5f}"
                  f"±{f['dCp_at_T0_stderr']:.5f} kJ/mol/K")
        if f.get("ab_correlation") is not None:
            print(f"        corr(a,b)={f['ab_correlation']:+.4f}  "
                  f"cond={f['design_cond']:.2e}")
    if res["nested_F"]:
        print("-" * 96)
        for t in res["nested_F"]:
            print(f"  F-test {t['nested']}: F={t['F']:.3f} "
                  f"(df1={t['df1']}, df2={t['df2']})")
    if res["loo_cv"]:
        print("-" * 96)
        for k, v in res["loo_cv"].items():
            print(f"  LOO-CV {k}: chi2={v['loo_chi2']:.3f}  "
                  f"std-RMSE={v['loo_rmse_standardised']:.3f}")
    if res["warnings"]:
        print("-" * 96)
        for w in res["warnings"]:
            print(f"  WARNING: {w}")
    print("=" * 96)


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 96)
    print(f"  WOO–ROUX STEP 6 + dCp(T) MODEL SELECTION v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 96)
    rng = np.random.default_rng(0)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    T = 310.0
    kT = KB * T

    # ---- A1 ideal torsion restraint vs narrow-limit analytic ----------
    print("\n[A1] Ideal conformational restraint vs narrow-limit analytic")
    c = ConfCoord("phi1", "torsion", k=2000.0, center=1.0)
    r = conf_apply_free_ideal(c, T)
    ap = r["narrow_limit_approx_kJmol"]
    rep("ideal torsion dG > 0", r["dG_apply_free_kJmol"] > 0,
        f"dG={r['dG_apply_free_kJmol']:.4f} kJ/mol")
    rep("quadrature matches narrow limit", ap is not None
        and abs(r["dG_apply_free_kJmol"] - ap) < 0.02,
        f"quad={r['dG_apply_free_kJmol']:.4f} approx={ap:.4f}")
    rep("ideal reference flagged", "warning" in r and len(r["warning"]) > 0)

    # ---- A2 zero restraint costs nothing -------------------------------
    print("\n[A2] Zero conformational restraint costs nothing")
    c0 = ConfCoord("free", "torsion", k=0.0, center=0.0)
    r0 = conf_apply_free_ideal(c0, T)
    rep("k=0 -> dG=0", abs(r0["dG_apply_free_kJmol"]) < 1e-10,
        f"dG={r0['dG_apply_free_kJmol']:.2e}")

    # ---- A3 rmsd has no ideal reference --------------------------------
    print("\n[A3] rmsd must refuse the ideal reference")
    cr = ConfCoord("rmsd", "rmsd", k=100.0, center=0.2, lo=0.0, hi=1.0)
    try:
        conf_apply_free_ideal(cr, T)
        rep("rmsd ideal refused", False)
    except ValueError:
        rep("rmsd ideal refused", True)

    # ---- A4 bound cost from PMF vs analytic Gaussian -------------------
    print("\n[A4] Bound conformational cost from PMF vs analytic")
    kb_, kr_, z0 = 400.0, 1600.0, 0.8
    cb = ConfCoord("phi_b", "torsion", k=kr_, center=z0)
    z = np.linspace(-math.pi, math.pi, 20001)
    W_int = 0.5 * kb_ * (z - z0) ** 2
    W_meas = W_int + cb.potential(z)
    got = conf_apply_from_pmf(z, W_meas, cb, T, restraint_on=True)
    exact = 0.5 * kT * math.log((kb_ + kr_) / kb_)
    rep("bound cost matches analytic Gaussian",
        abs(got["dG_apply_kJmol"] - exact) < 0.02,
        f"got={got['dG_apply_kJmol']:.4f} exact={exact:.4f}")
    rep("bound coverage OK", got["coverage_ok"],
        f"tail={got['released_tail_fraction']:.4f}")

    # ---- A5 coverage failure detected ----------------------------------
    print("\n[A5] Coverage failure must be flagged")
    zn = np.linspace(0.75, 0.85, 401)
    cn = ConfCoord("narrow", "torsion", k=5.0, center=0.8)
    Wn = 0.5 * 400.0 * (zn - 0.8) ** 2 + cn.potential(zn)
    bad = conf_apply_from_pmf(zn, Wn, cn, T)
    rep("narrow window flagged", not bad["coverage_ok"],
        f"tail={bad['released_tail_fraction']:.3f}")

    # ---- A6 rotamer counting -------------------------------------------
    print("\n[A6] Rotamer counting estimate")
    rr = rotamer_penalty(10, 1, T)
    rep("N=10 -> kT ln 10", abs(rr["dG_conf_estimate_kJmol"]
                               - kT * math.log(10)) < 1e-10,
        f"dG={rr['dG_conf_estimate_kJmol']:.3f} kJ/mol")

    # ---- A7 full assembly recovers a known total -----------------------
    print("\n[A7] Full Woo–Roux assembly with step 6")
    radial = -30.0
    ang = [{"name": "theta", "dG_apply_free_kJmol": 6.0,
            "dG_apply_bound_kJmol": 0.5, "coverage_ok": True}]
    cf = [{"name": "phi1", "dG_apply_free_kJmol": 8.0,
           "reference": "sampled PMF", "coverage_ok": True}]
    cbd = [{"name": "phi1", "dG_apply_kJmol": 0.7, "coverage_ok": True}]
    full = wooroux_full(radial, T, ang, cf, cbd)
    expect = radial + (6.0 - 0.5) + (8.0 - 0.7)
    rep("total matches hand sum", abs(full.dG_bind_standard - expect) < 1e-12,
        f"dG={full.dG_bind_standard:.4f} expect={expect:.4f}")
    rep("conformational correction is POSITIVE (entropy penalty)",
        full.conformational_correction > 0,
        f"conf={full.conformational_correction:+.3f} kJ/mol")
    rep("no V° factor applied to internal coordinate",
        abs(full.conformational_correction - (8.0 - 0.7)) < 1e-12)

    # ---- A8 missing bound PMF and coupling are flagged -----------------
    print("\n[A8] Missing bound PMF / coupled torsions flagged")
    f2 = wooroux_full(radial, T, ang, cf, [])
    rep("missing bound PMF flagged",
        any("assumed 0" in w for w in f2.warnings))
    f3 = wooroux_full(radial, T, ang, cf, cbd, independent_torsions=False)
    rep("coupled torsions flagged",
        any("Coupled" in w for w in f3.warnings))

    # ---- B1 dCp(T) design rows are exact -------------------------------
    print("\n[B1] Design rows reproduce dH(T) and dG(T) exactly")
    T0 = 310.0
    dH0, dS0, a_, b_ = -45.0, -0.10, -0.60, 0.0020
    def dH_of(T): return dH0 + a_ * (T - T0) + 0.5 * b_ * (T * T - T0 * T0)
    def dS_of(T): return dS0 + a_ * math.log(T / T0) + b_ * (T - T0)
    def dG_of(T): return dH_of(T) - T * dS_of(T)
    ptrue = np.array([dH0, dS0, a_, b_])
    eH = max(abs(float(design_row(t, "dH", T0) @ ptrue) - dH_of(t))
             for t in (295., 305., 315., 325.))
    eG = max(abs(float(design_row(t, "dG", T0) @ ptrue) - dG_of(t))
             for t in (295., 305., 315., 325.))
    rep("dH row exact", eH < 1e-10, f"err={eH:.2e}")
    rep("dG row exact", eG < 1e-10, f"err={eG:.2e}")

    # ---- B2 M2 recovery with enough data -------------------------------
    print("\n[B2] M2 recovery with 8 temperatures, low noise")
    Ts = np.linspace(288.0, 328.0, 8)
    obs = []
    for t in Ts:
        obs.append(ThermoObs(float(t), "dH", dH_of(t) + rng.normal(0, 0.10),
                             0.10, "calorimetric"))
        obs.append(ThermoObs(float(t), "dG", dG_of(t) + rng.normal(0, 0.06),
                             0.06, "calorimetric"))
    res = fit_dcp_models(obs, T0)
    m2 = res["fits"]["M2"]
    rep("M2 fitted", m2["status"] == "FITTED")
    if m2["status"] == "FITTED":
        rep("a recovered within 3 sigma",
            abs(m2["params"]["a"] - a_) < 3 * m2["stderr"]["a"],
            f"a={m2['params']['a']:+.4f}±{m2['stderr']['a']:.4f} true={a_:+.4f}")
        rep("b recovered within 3 sigma",
            abs(m2["params"]["b"] - b_) < 3 * m2["stderr"]["b"],
            f"b={m2['params']['b']:+.6f}±{m2['stderr']['b']:.6f} true={b_:+.6f}")
    rep("M2 ranked first when signal is real",
        res["ranking"] and res["ranking"][0]["model"] == "M2",
        f"top={res['ranking'][0]['model'] if res['ranking'] else None}")
    report(res)

    # ---- B3 three temperatures refuse M2 -------------------------------
    print("\n[B3] Three temperatures must REFUSE the 4-parameter model")
    obs3 = []
    for t in (300.0, 310.0, 320.0):
        obs3.append(ThermoObs(t, "dH", dH_of(t), 0.2, "calorimetric"))
    r3 = fit_dcp_models(obs3, T0)
    rep("M2 refused at n=3", r3["fits"]["M2"]["status"] == "REFUSED",
        r3["fits"]["M2"].get("reason", ""))
    rep("M1 also refused at n=3 (dof=0)",
        r3["fits"]["M1"]["status"] == "REFUSED")
    rep("M0 fitted at n=3", r3["fits"]["M0"]["status"] == "FITTED")

    # ---- B4 Akaike weights normalise -----------------------------------
    print("\n[B4] Akaike weights sum to 1")
    wsum = sum(r["akaike_weight"] for r in res["ranking"])
    rep("weights normalised", abs(wsum - 1.0) < 1e-12, f"sum={wsum:.12f}")

    # ---- B5 noise-dominated b is flagged -------------------------------
    print("\n[B5] Noise-dominated b must be flagged, not reported silently")
    obsn = []
    for t in np.linspace(303.0, 317.0, 6):
        obsn.append(ThermoObs(float(t), "dH",
                              dH_of(t) + rng.normal(0, 2.0), 2.0,
                              "calorimetric"))
    rn = fit_dcp_models(obsn, T0)
    flagged = any(("not identifiable" in w.lower())
                  or ("ill-conditioned" in w.lower())
                  or ("AMBIGUOUS" in w) for w in rn["warnings"])
    rep("weak-data warning emitted", flagged,
        f"{len(rn['warnings'])} warnings")

    # ---- B6 nested F and LOO present -----------------------------------
    print("\n[B6] Nested F-test and LOO-CV reported")
    rep("nested F tests present", len(res["nested_F"]) >= 1,
        f"{[t['nested'] for t in res['nested_F']]}")
    rep("LOO-CV present for fitted models",
        set(res["loo_cv"]) >= {"M0", "M1", "M2"},
        f"{sorted(res['loo_cv'])}")

    # ---- B7 mixed provenance flagged -----------------------------------
    print("\n[B7] Mixed provenance flagged")
    obsm = [ThermoObs(300.0, "dH", -44.0, 0.2, "calorimetric"),
            ThermoObs(310.0, "dH", -45.0, 0.2, "MD potential energy"),
            ThermoObs(320.0, "dH", -46.0, 0.2, "calorimetric"),
            ThermoObs(330.0, "dH", -47.0, 0.2, "calorimetric")]
    rm = fit_dcp_models(obsm, T0)
    rep("mixed provenance flagged",
        any("Mixed provenance" in w for w in rm["warnings"]))
    rep("MD-derived flagged",
        any("MD/potential" in w for w in rm["warnings"]))

    print("\n" + "=" * 96)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 96)
    print("""
  WHAT THESE TESTS PROVE
    A1-A2  the ideal conformational restraint quadrature matches its
           narrow-limit analytic form, and costs nothing at k=0
    A3     rmsd correctly REFUSES the uniform-measure ideal reference
    A4     the bound conformational cost from a PMF matches the analytic
           Gaussian result
    A5     insufficient PMF coverage is detected
    A6     rotamer counting reproduces kT ln N
    A7     the full assembly adds the conformational term with the correct
           sign and NO spurious V° factor
    A8     missing bound PMFs and coupled torsions are flagged
    B1     the dCp(T) design rows reproduce dH(T) and dG(T) exactly
    B2     with 8 temperatures and low noise, a and b are recovered and M2
           is ranked first
    B3     with 3 temperatures the 4-parameter model is REFUSED
    B4     Akaike weights normalise
    B5     noise-dominated b produces an explicit warning
    B6-B7  nested F, LOO-CV, and provenance warnings are reported

  WHAT IS NOT PROVEN
    * the ideal free-rotor reference OVERESTIMATES the real penalty
    * no Fixman/metric-determinant correction for coupled coordinates
    * model selection on few points is weak evidence, not a verdict
    * b requires curvature in dH(T): wide T range and precise data
    * no experimental validation is performed here
""")
    return ok


def main():
    ap = argparse.ArgumentParser(description=f"Woo-Roux step6 + dCp(T) v{VERSION}")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--thermo", default=None,
                    help="JSON list of {T_K,kind,value,sigma,provenance}")
    ap.add_argument("--T0", type=float, default=310.0)
    ap.add_argument("--sigmas-unknown", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if self_test() else 1)
    if a.thermo:
        with open(a.thermo) as f:
            raw = json.load(f)
        obs = [ThermoObs(float(d["T_K"]), d["kind"], float(d["value"]),
                         float(d.get("sigma", float("nan"))),
                         d.get("provenance", "unspecified")) for d in raw]
        res = fit_dcp_models(obs, a.T0, sigmas_known=not a.sigmas_unknown)
        report(res)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=2, default=float)
        return
    ap.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
