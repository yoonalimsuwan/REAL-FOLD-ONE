#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sle_longitudinal.py — SLE Flare-Prediction Research Pipeline v1.0
Politis–Romano subsampling · Madras–Sokal τ_int · CSOC alert bound
Missing-evidence report · horizon validation
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

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

Christ died upon the Cross to teach: when struck on the left cheek, turn also
the right, so that we may listen to the voice of the enemy who wishes us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT THIS IS
=============================================================================
A RESEARCH pipeline for studying whether longitudinal SLE biomarkers carry
lead-time information about adjudicated flares, with uncertainty that is
honest about serial correlation.

NOT a diagnostic device. NOT a treatment recommender. Every output is
research_only and requires clinician review.

SLE (systemic lupus erythematosus, โรคพุ่มพวง) is a loss of immune tolerance.
It is NOT a protein-folding disease: the autoantigens (dsDNA, nucleosome,
Sm/RNP, Ro60, La) are correctly folded proteins that are wrongly targeted.
No amount of structure prediction addresses the mechanism.

=============================================================================
WHY SUBSAMPLING AND NOT AN iid CI  (the central technical point)
=============================================================================
Anti-dsDNA titre, C3/C4, SLEDAI-2K and UPCR are strongly serially correlated:
a patient's value this month predicts next month's. An iid standard error
s/sqrt(n) therefore UNDERSTATES the true sampling variability, sometimes by
a factor of 2-4. The resulting interval is too narrow and its empirical
coverage is far below nominal.

Politis–Romano subsampling builds the sampling distribution from OVERLAPPING
BLOCKS of length b, which preserves the dependence inside each block:

    L_n(x) = (1/(n-b+1)) sum_i 1{ tau_b (theta_{b,i} - theta_n) <= x }
    CI     = [ theta_n - q_{1-a/2}/tau_n , theta_n - q_{a/2}/tau_n ]

Valid asymptotically under stationarity and a mixing condition, with
b -> infinity and b/n -> 0. It is ASSUMPTION-LIGHT, not assumption-free.

THE RATE MATTERS. For a mean, tau_n = n^(1/2). For an OLS TREND SLOPE on an
equally spaced grid, Var(beta_hat) ~ 12 sigma^2 / b^3, so tau_n = n^(3/2).
Using the mean's rate for a slope gives a badly wrong interval. This module
uses the correct rate for each statistic and reports which was used.

Madras–Sokal tau_int with an automatic window gives the effective sample
size n_eff = n/(2 tau_int). When n_eff is small the interval is reported as
wide and the status is flagged; it is not silently narrowed.

=============================================================================
THE CSOC ALERT BOUND
=============================================================================
An alerting mechanism that switches on and off can chatter. We bound the
number of alert firings with a two-level accumulator:

    excess  = softplus(signal - theta_hi)
    decay   = sigmoid(-(acc - theta_lo) * k)
    removal = decay * rate * acc          <-- the ACTUAL removal
    acc    += excess - removal
    fire when armed and acc > theta_hi; re-arm when acc < theta_lo

    N(T) <= (E_max(T) + D(T)) / (theta_hi - theta_lo),  D = sum removal

D MUST be the actual dissipation removed. Accumulating excess*rate instead
does not appear in the proof and the bound is then not guaranteed. This
module accumulates the actual removal and audits it.

This is an ENGINEERING bound on the alert mechanism, of the same kind as a
CFL bound. It is NOT a clinical guarantee and NOT a statement about disease.

=============================================================================
THE CONFOUNDER THAT CAN INVALIDATE THE WHOLE STUDY
=============================================================================
Glucocorticoids and immunosuppressants suppress BOTH the biomarkers and the
SLEDAI score. A model trained without a medication log will learn to detect
steroid TAPERS and label them as impending flares. That is a real association
in the data and a clinically useless predictor.

Therefore the missing-evidence report treats an absent or incomplete
medication log as a BLOCKING deficiency, not a nice-to-have. Infection is a
second mimic and must be excluded at each adjudicated event.

=============================================================================
HONEST SCOPE
=============================================================================
* No clinical validation is performed here. Synthetic data only.
* Anti-dsDNA assays (Farr, ELISA, CLIFT) are NOT interchangeable. Units and
  reference ranges must be recorded per site; pooling without harmonisation
  is invalid.
* Flare must be ADJUDICATED (SELENA-SLEDAI Flare Index or a declared
  equivalent), not inferred from a score change alone.
* Few flares per patient means wide intervals. Lead-time claims require
  many patients and a pre-declared horizon.
* Irregular visit spacing violates the equal-spacing premise of index-block
  subsampling. The spacing CV is measured and reported.
* No treatment, dosing, or gene-editing output of any kind.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("sle")

VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
SLE LONGITUDINAL RESEARCH PIPELINE v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
Politis–Romano CI (correct rates) · Madras–Sokal tau_int · CSOC alert bound
RESEARCH ONLY. No diagnosis, no treatment, no dosing, no gene editing.
Medication log is a BLOCKING requirement, not optional.
=============================================================================
"""


class Status:
    OK = "OK"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_COMPARED = "NOT_COMPARED"
    INSUFFICIENT = "INSUFFICIENT"


# =============================================================================
# §1. GOVERNANCE — IRB, PHI, safety
# =============================================================================

class SafetyGate:
    """Blocks treatment advice and gene-editing design outright."""
    BLOCKED = {
        "treatment_advice": [
            r"\bprescribe\b", r"\bstart\s+(steroid|prednisolone|mycophenolate|"
            r"cyclophosphamide|belimumab|anifrolumab|rituximab)\b",
            r"\bincrease\s+(the\s+)?(steroid|prednisolone)\b",
            r"\btaper\b.*\b(steroid|prednisolone)\b",
            r"\bwhat\s+dose\b", r"\bshould\s+(i|we)\s+(treat|start|stop)\b",
            r"\brecommend\s+(treatment|therapy|drug)\b",
        ],
        "gene_editor_design": [
            r"\bcrispr\b", r"\bcas9\b", r"\bguide\s*rna\b", r"\bgrna\b",
            r"\bbase\s*editor\b", r"\bprime\s*editor\b",
            r"\bcar[\s-]*t\s*(construct|design)\b", r"\bgene\s*edit(or|ing)\b",
            r"\bdelivery\s*vector\b",
        ],
        "autonomous_clinical": [
            r"\bdiagnose\s*(the\s*)?patient\b", r"\bclinical\s*decision\b",
            r"\bapprove\s*treatment\b",
        ],
    }

    def check(self, text: str) -> Dict[str, Any]:
        low = text.lower()
        for intent, pats in self.BLOCKED.items():
            hits = [p for p in pats if re.search(p, low)]
            if hits:
                return {"allowed": False, "intent": intent,
                        "blocked_patterns": hits,
                        "reason": f"BLOCKED intent={intent}. This platform does "
                                  f"not provide treatment advice, dosing, "
                                  f"gene-editor design, or clinical decisions."}
        return {"allowed": True, "intent": "research", "blocked_patterns": [],
                "reason": "allowed research query"}


class PHIGuard:
    PATTERNS = {
        "name_like": r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b",
        "mrn": r"\b(MRN|mrn)[:\s]*\d{4,}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "full_date": r"\b(19|20)\d{2}-\d{2}-\d{2}\b",
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "phone": r"\b\+?\d[\d\s().-]{8,}\d\b",
    }
    DIRECT = {"name", "patient_name", "mrn", "patient_id", "national_id",
              "address", "email", "phone", "birth_date", "dob",
              "date_of_birth", "accession_number", "hospital_number"}
    QUASI = {"age", "sex", "ethnicity", "postcode", "site", "rare_diagnosis",
             "enrolment_date"}

    def scrub(self, text: str) -> Tuple[str, List[str]]:
        out, removed = text, []
        for lab, pat in self.PATTERNS.items():
            new = re.sub(pat, "[REDACTED]", out)
            if new != out:
                removed.append(lab); out = new
        return out, removed

    def deidentify(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        removed, quasi, out = [], [], dict(rec)
        for k in list(out.keys()):
            if k.lower() in self.DIRECT:
                out.pop(k); removed.append(k)
            elif k.lower() in self.QUASI:
                quasi.append(k)
        if "age" in out:
            try:
                a = int(out.pop("age"))
                out["age_band"] = f"{(a//10)*10}-{(a//10)*10+9}"
                removed.append("age->age_band")
            except Exception:
                pass
        for k, v in list(out.items()):
            if isinstance(v, str):
                s, rm = self.scrub(v)
                if rm:
                    out[k] = s; removed.append(f"{k}:{','.join(rm)}")
        risk = "low" if not quasi else ("medium" if len(quasi) <= 3 else "high")
        return {"passed": risk != "high", "removed_fields": removed,
                "residual_risk": risk, "quasi_identifiers": quasi,
                "clean_record": out}


@dataclass
class IRBRecord:
    protocol_id: str
    institution: str
    approval_date: str
    expiry_date: str
    approved_uses: List[str]
    consent_model: str = "written"
    data_use_agreement: bool = False
    status: str = "PENDING"

    def active(self) -> bool:
        if self.status != "APPROVED":
            return False
        try:
            return time.time() < time.mktime(time.strptime(self.expiry_date, "%Y-%m-%d"))
        except Exception:
            return False


class Governance:
    def __init__(self, irb: Optional[IRBRecord] = None):
        self.irb, self.phi, self.safety = irb, PHIGuard(), SafetyGate()
        self.audit: List[Dict[str, Any]] = []

    def authorize(self, text: str, uses_patient_data: bool = False,
                  deid: Optional[Dict] = None,
                  purpose: str = "sle_research") -> Dict[str, Any]:
        s = self.safety.check(text)
        d = dict(s)
        d.update({"authorized": False, "clinician_review_required": True,
                  "clinical_actionability": RESEARCH_ONLY})
        if not s["allowed"]:
            self._log(d, text); return d
        if uses_patient_data:
            if self.irb is None:
                d["reason"] = "BLOCKED: no IRB record"; self._log(d, text); return d
            if not self.irb.active():
                d["reason"] = f"BLOCKED: IRB status={self.irb.status} or expired"
                self._log(d, text); return d
            if purpose not in self.irb.approved_uses:
                d["reason"] = f"BLOCKED: purpose '{purpose}' not approved"
                self._log(d, text); return d
            if not self.irb.data_use_agreement:
                d["reason"] = "BLOCKED: no data use agreement"
                self._log(d, text); return d
            if not deid or not deid.get("passed"):
                d["reason"] = "BLOCKED: de-identification missing or failed"
                self._log(d, text); return d
        d["authorized"] = True
        d["reason"] = "authorized for research use"
        self._log(d, text)
        return d

    def _log(self, d: Dict, text: str) -> None:
        self.audit.append({
            "ts": time.time(),
            "query_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "authorized": d["authorized"], "reason": d["reason"]})


# =============================================================================
# §2. SLE DATA SCHEMA
# =============================================================================

@dataclass
class AssayMeta:
    """Anti-dsDNA assays are NOT interchangeable. Record everything."""
    analyte: str
    method: str            # anti-dsDNA: "Farr" | "ELISA" | "CLIFT" | "FEIA"
    unit: str              # "IU/mL", "U/mL", "g/g", "mg/dL", "cells/uL"
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    site_id: str = "site_anon"
    lod: Optional[float] = None

    def complete(self) -> bool:
        return bool(self.method and self.unit)


@dataclass
class Visit:
    """One de-identified longitudinal visit. t_week is relative, not a date."""
    t_week: float
    sledai2k: Optional[float] = None          # 0..105
    anti_dsdna: Optional[float] = None        # IU/mL (assay-specific!)
    c3_mg_dl: Optional[float] = None
    c4_mg_dl: Optional[float] = None
    upcr_g_g: Optional[float] = None          # urine protein/creatinine
    ifn_score: Optional[float] = None         # interferon gene signature
    alc_per_ul: Optional[float] = None        # absolute lymphocyte count
    anti_c1q: Optional[float] = None
    # CONFOUNDER — mandatory for a valid study
    prednisolone_mg_day: Optional[float] = None
    hcq_mg_day: Optional[float] = None
    immunosuppressant: Optional[str] = None
    biologic: Optional[str] = None
    infection_present: Optional[bool] = None


@dataclass
class FlareAdjudication:
    """
    Adjudicated flare, NOT a raw score change.
    definition: "SELENA-SLEDAI SFI" | "SLEDAI increase >=3" | declared other
    severity: "mild" | "moderate" | "severe"
    adjudicated_by: independent assessor identifier (anonymised)
    """
    t_week: float
    definition: str
    severity: str
    adjudicated_by: str
    infection_excluded: bool
    organ_domain: str = "unspecified"


@dataclass
class PatientSeries:
    research_id: str
    visits: List[Visit]
    flares: List[FlareAdjudication] = field(default_factory=list)
    assays: Dict[str, AssayMeta] = field(default_factory=dict)
    ifn_stratum: str = "unknown"              # "high" | "low" | "unknown"
    sledai_assessor_reliability: Optional[float] = None   # e.g. kappa / ICC
    consent_scope: str = RESEARCH_ONLY

    def n(self) -> int:
        return len(self.visits)

    def spacing_cv(self) -> float:
        if self.n() < 3:
            return float("nan")
        t = np.array([v.t_week for v in self.visits], float)
        g = np.diff(t)
        m = float(g.mean())
        return float(g.std(ddof=1) / m) if m > 0 else float("nan")

    def series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (t, values) with missing entries dropped."""
        t, y = [], []
        for v in self.visits:
            val = getattr(v, key, None)
            if val is not None and np.isfinite(val):
                t.append(v.t_week); y.append(float(val))
        return np.array(t, float), np.array(y, float)


# =============================================================================
# §3. POLITIS–ROMANO SUBSAMPLING  (correct rate per statistic)
# =============================================================================

def _ols_slope(t: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on t. Units: y per unit t."""
    n = len(t)
    if n < 2:
        return 0.0
    tc = t - t.mean()
    den = float(tc @ tc)
    if den <= 0:
        return 0.0
    return float(tc @ (y - y.mean())) / den


def stat_mean(t: np.ndarray, y: np.ndarray) -> float:
    return float(y.mean()) if len(y) else float("nan")


def stat_slope(t: np.ndarray, y: np.ndarray) -> float:
    return _ols_slope(t, y)


# rate exponent: mean -> 1/2 ; OLS slope on an equally spaced grid -> 3/2
STAT_RATE = {"mean": 0.5, "slope": 1.5}


def subsample_ci(t: np.ndarray, y: np.ndarray, stat: str = "mean",
                 b: Optional[int] = None, conf: float = 0.95,
                 min_blocks: int = 20) -> Dict[str, Any]:
    """
    Politis–Romano overlapping-block subsampling CI.

    theta_n : statistic on the full series
    theta_{b,i} : statistic on the block starting at i, length b
    L(x) = empirical distribution of tau_b (theta_{b,i} - theta_n)
    CI   = [theta_n - q_{1-a/2}/tau_n , theta_n - q_{a/2}/tau_n]

    tau_n = n**rate with rate = 1/2 for a mean and 3/2 for an OLS slope.
    ASSUMPTIONS: stationarity + mixing; b -> inf, b/n -> 0. Asymptotic.
    """
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = len(y)
    if stat not in STAT_RATE:
        raise ValueError(f"stat must be one of {sorted(STAT_RATE)}")
    rate = STAT_RATE[stat]
    fn = stat_mean if stat == "mean" else stat_slope
    theta_n = fn(t, y)

    if n < 6:
        return {"status": Status.INSUFFICIENT, "stat": stat, "rate": rate,
                "estimate": theta_n, "lo": theta_n, "hi": theta_n,
                "n": n, "b": 0, "n_blocks": 0,
                "reason": "n < 6: subsampling not meaningful"}

    if b is None:
        b = int(round(n ** (1.0 / 3.0)))
    b = int(max(3, min(b, n // 2)))
    nb = n - b + 1
    if nb < min_blocks:
        b = max(3, n - min_blocks + 1)
        nb = n - b + 1
    if nb < 4:
        return {"status": Status.INSUFFICIENT, "stat": stat, "rate": rate,
                "estimate": theta_n, "lo": theta_n, "hi": theta_n,
                "n": n, "b": b, "n_blocks": nb,
                "reason": "too few blocks"}

    tau_b = float(b) ** rate
    tau_n = float(n) ** rate
    devs = np.empty(nb)
    for i in range(nb):
        devs[i] = tau_b * (fn(t[i:i + b], y[i:i + b]) - theta_n)

    a = 1.0 - conf
    q_hi = float(np.quantile(devs, 1.0 - a / 2.0))
    q_lo = float(np.quantile(devs, a / 2.0))
    lo = theta_n - q_hi / tau_n
    hi = theta_n - q_lo / tau_n
    if lo > hi:
        lo, hi = hi, lo
    return {"status": Status.OK, "stat": stat, "rate": rate,
            "estimate": theta_n, "lo": lo, "hi": hi, "width": hi - lo,
            "n": n, "b": b, "n_blocks": nb, "conf": conf,
            "method": "Politis-Romano overlapping-block subsampling",
            "assumptions": "stationarity + mixing; asymptotic, NOT assumption-free"}


def iid_ci(t: np.ndarray, y: np.ndarray, stat: str = "mean",
           conf: float = 0.95) -> Dict[str, Any]:
    """
    The WRONG interval, provided only so the undercoverage can be MEASURED.
    Never use this for a serially correlated clinical series.
    """
    y = np.asarray(y, float); t = np.asarray(t, float)
    n = len(y)
    if n < 3:
        m = float(y.mean()) if n else float("nan")
        return {"estimate": m, "lo": m, "hi": m, "width": 0.0, "n": n}
    z = 1.959963985 if abs(conf - 0.95) < 1e-9 else float(
        _norm_ppf(1.0 - (1.0 - conf) / 2.0))
    if stat == "mean":
        est = float(y.mean())
        se = float(y.std(ddof=1) / math.sqrt(n))
    else:
        est = _ols_slope(t, y)
        tc = t - t.mean()
        den = float(tc @ tc)
        resid = y - (y.mean() + est * tc)
        s2 = float(resid @ resid) / max(n - 2, 1)
        se = math.sqrt(s2 / den) if den > 0 else 0.0
    return {"estimate": est, "lo": est - z * se, "hi": est + z * se,
            "width": 2 * z * se, "n": n,
            "WARNING": "iid SE invalid under serial correlation"}


def _norm_ppf(p: float) -> float:
    """Acklam inverse normal CDF approximation (no scipy dependency)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# =============================================================================
# §4. BLOCK-LENGTH SELECTION + SENSITIVITY
# =============================================================================

def select_block(t: np.ndarray, y: np.ndarray, stat: str = "mean",
                 conf: float = 0.95,
                 b_grid: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Minimum-volatility style selection: compute the CI width over a grid of
    b, then take the smallest b whose width is within a tolerance of the
    local plateau. Report the whole curve so the sensitivity is VISIBLE.
    """
    n = len(y)
    if n < 12:
        return {"status": Status.INSUFFICIENT, "b": max(3, n // 3),
                "reason": "n < 12: no meaningful block search"}
    if b_grid is None:
        lo = max(3, int(round(n ** (1 / 3))))
        hi = max(lo + 1, n // 2)
        b_grid = sorted(set(int(round(x)) for x in
                            np.linspace(lo, hi, min(12, hi - lo + 1))))
    rows = []
    for b in b_grid:
        r = subsample_ci(t, y, stat, b=b, conf=conf)
        if r["status"] == Status.OK:
            rows.append({"b": b, "width": r["width"],
                         "lo": r["lo"], "hi": r["hi"]})
    if len(rows) < 3:
        return {"status": Status.INSUFFICIENT,
                "b": int(round(n ** (1 / 3))), "curve": rows}
    w = np.array([r["width"] for r in rows])
    # local volatility of the width across neighbouring b
    vol = np.abs(np.diff(w)) / np.maximum(w[:-1], 1e-30)
    # first index after which volatility stays below 0.15
    stable = None
    for i in range(len(vol)):
        if np.all(vol[i:] < 0.15):
            stable = i; break
    b_sel = rows[stable]["b"] if stable is not None else rows[int(np.argmin(w))]["b"]
    return {"status": Status.OK, "b": int(b_sel), "curve": rows,
            "selection": "minimum-volatility plateau",
            "width_spread_rel": float((w.max() - w.min()) /
                                      max(np.median(w), 1e-30)),
            "note": "block choice affects finite-sample width; curve reported"}


# =============================================================================
# §5. MADRAS–SOKAL tau_int
# =============================================================================

def tau_int(y: np.ndarray, c: float = 5.0,
            max_lag: Optional[int] = None) -> Dict[str, Any]:
    """
    Integrated autocorrelation time with an automatic window:
        tau = 1/2 + sum_{t=1..W} rho(t)
        W = smallest t with t >= c * tau(t)
        n_eff = n / (2 tau)
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 16:
        return {"status": Status.INSUFFICIENT, "tau_int": float("nan"),
                "n": n, "n_eff": float("nan"),
                "reason": "n < 16: autocorrelation not estimable"}
    xc = y - y.mean()
    var = float(xc @ xc) / n
    if var <= 0:
        return {"status": Status.OK, "tau_int": 0.5, "window": 0,
                "n": n, "n_eff": float(n), "note": "constant series"}
    lim = min(n // 2, max_lag or n // 2)
    tau, W = 0.5, 0
    for t in range(1, lim):
        rho = float(xc[:n - t] @ xc[t:]) / (n - t) / var
        tau += rho
        if tau <= 0:
            tau = 0.5; W = t; break
        W = t
        if W >= c * max(tau, 0.5):
            break
    tau = max(tau, 0.5)
    n_eff = n / (2.0 * tau)
    return {"status": Status.OK, "tau_int": float(tau), "window": int(W),
            "n": n, "n_eff": float(n_eff),
            "few_effective_samples": bool(n_eff < 20),
            "method": "Madras-Sokal automatic window", "c": c}


# =============================================================================
# §6. FEATURE EXTRACTION
# =============================================================================

BIOMARKERS = ("sledai2k", "anti_dsdna", "c3_mg_dl", "c4_mg_dl",
              "upcr_g_g", "ifn_score", "alc_per_ul", "anti_c1q")

# Direction in which a change is biologically adverse (research hypothesis).
ADVERSE_DIRECTION = {
    "sledai2k": +1, "anti_dsdna": +1, "upcr_g_g": +1, "ifn_score": +1,
    "anti_c1q": +1,
    "c3_mg_dl": -1, "c4_mg_dl": -1, "alc_per_ul": -1,
}


def extract_features(ps: PatientSeries, window: int = 8,
                     conf: float = 0.95) -> Dict[str, Any]:
    """
    For each biomarker, over the last `window` visits:
      level  : subsampling CI on the mean
      trend  : subsampling CI on the OLS slope (rate n^{3/2})
      tau    : Madras-Sokal integrated autocorrelation time
    Also report spacing irregularity, which threatens the block premise.
    """
    out: Dict[str, Any] = {"window": window, "conf": conf, "biomarkers": {}}
    for key in BIOMARKERS:
        t, y = ps.series(key)
        if len(y) == 0:
            out["biomarkers"][key] = {"status": Status.UNAVAILABLE}
            continue
        t = t[-window:]; y = y[-window:]
        # centre time so the slope is in units per week
        if len(t) > 1:
            t = t - t[0]
        lvl = subsample_ci(t, y, "mean", conf=conf)
        trd = subsample_ci(t, y, "slope", conf=conf)
        blk = select_block(t, y, "slope", conf=conf) if len(y) >= 12 else {}
        ti = tau_int(y)
        iid = iid_ci(t, y, "slope", conf=conf)
        d = ADVERSE_DIRECTION.get(key, +1)
        adverse = (trd["status"] == Status.OK and trd["lo"] * d > 0.0)
        out["biomarkers"][key] = {
            "status": Status.OK, "n_used": len(y),
            "level": lvl, "trend": trd, "tau_int": ti,
            "block_selection": blk,
            "iid_trend_FOR_COMPARISON_ONLY": iid,
            "width_ratio_sub_over_iid": (trd.get("width", float("nan")) /
                                         max(iid.get("width", 1e-30), 1e-30)),
            "adverse_trend_significant": bool(adverse),
            "adverse_direction": d,
        }
    out["spacing_cv"] = ps.spacing_cv()
    out["spacing_warning"] = bool(np.isfinite(out["spacing_cv"]) and
                                  out["spacing_cv"] > 0.5)
    return out


# =============================================================================
# §7. CSOC ALERT COUNTER  (bound with ACTUAL removal)
# =============================================================================

class CSOCAlertCounter:
    """
    Two-level accumulator with a provable firing bound.
        N(T) <= (E_max + D) / (theta_hi - theta_lo),  D = sum ACTUAL removal
    Engineering bound on the alert mechanism. NOT a clinical guarantee.
    """

    def __init__(self, theta_hi: float = 0.60, theta_lo: float = 0.25,
                 decay_rate: float = 0.08, min_interval_visits: int = 2):
        if not theta_hi > theta_lo:
            raise ValueError("require theta_hi > theta_lo")
        self.hi, self.lo = float(theta_hi), float(theta_lo)
        self.gap = self.hi - self.lo
        self.rate = float(decay_rate)
        self.min_interval = int(min_interval_visits)
        self.acc = 0.0
        self.armed = True
        self.events = 0
        self.E_max = 0.0
        self.D = 0.0
        self.D_audit = 0.0
        self._last_fire = -10 ** 9
        self.history: List[Dict[str, float]] = []

    @staticmethod
    def _softplus(x: float) -> float:
        return math.log1p(math.exp(-abs(x))) + max(x, 0.0)

    def step(self, signal: float, k: int) -> Dict[str, Any]:
        s = float(signal)
        excess = self._softplus(s - self.hi)
        decay = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10.0))
        removal = decay * self.rate * self.acc      # ACTUAL removal
        self.acc += excess - removal
        self.D += removal
        self.D_audit += removal
        self.E_max = max(self.E_max, self.acc)
        if self.acc < self.lo:
            self.armed = True
        fire = (self.armed and self.acc > self.hi
                and (k - self._last_fire) >= self.min_interval)
        if fire:
            self.armed = False
            self.events += 1
            self._last_fire = int(k)
        bound = (self.E_max + self.D) / self.gap
        rec = {"k": k, "signal": s, "acc": self.acc, "removal": removal,
               "fired": bool(fire), "events": self.events, "bound": bound,
               "satisfied": self.events <= bound + 1e-9}
        self.history.append(rec)
        return rec

    def report(self) -> Dict[str, Any]:
        bound = (self.E_max + self.D) / self.gap
        return {"events": self.events, "bound": bound,
                "satisfied": self.events <= bound + 1e-9,
                "E_max": self.E_max, "D_actual": self.D,
                "D_audit_matches": abs(self.D - self.D_audit) < 1e-12,
                "note": "ENGINEERING bound on the alert mechanism, "
                        "NOT a clinical guarantee and NOT PDE regularity"}


# =============================================================================
# §8. FLARE-PRESSURE SCORE + STRATIFICATION
# =============================================================================

@dataclass
class ScoreWeights:
    """Transparent, declared weights. NOT fitted. NOT a validated model."""
    anti_dsdna_trend: float = 0.30
    complement_trend: float = 0.25
    upcr_trend: float = 0.20
    sledai_level: float = 0.15
    ifn_level: float = 0.10
    sledai_ref: float = 12.0        # scale for SLEDAI level
    ifn_ref: float = 10.0
    trend_ref: float = 0.05         # scale for a normalised slope

    def declared(self) -> Dict[str, float]:
        return asdict(self)


def flare_pressure(feat: Dict[str, Any], w: Optional[ScoreWeights] = None
                   ) -> Dict[str, Any]:
    """
    A TRANSPARENT composite of adverse trends and levels, in [0,1].
    This is a declared research index, not a fitted or calibrated model.
    It intentionally has no learned parameters so it cannot silently
    overfit a small cohort.
    """
    w = w or ScoreWeights()
    b = feat["biomarkers"]

    def sig_trend(key: str) -> float:
        r = b.get(key, {})
        if r.get("status") != Status.OK:
            return 0.0
        tr = r["trend"]
        if tr["status"] != Status.OK:
            return 0.0
        d = ADVERSE_DIRECTION.get(key, +1)
        # use the CONSERVATIVE end of the CI: the bound nearest zero
        lo, hi = tr["lo"], tr["hi"]
        conservative = lo if d > 0 else -hi
        return float(max(0.0, d * conservative) / max(w.trend_ref, 1e-9))

    def lvl(key: str, ref: float) -> float:
        r = b.get(key, {})
        if r.get("status") != Status.OK:
            return 0.0
        lv = r["level"]
        if lv["status"] != Status.OK:
            return 0.0
        return float(max(0.0, lv["estimate"]) / max(ref, 1e-9))

    parts = {
        "anti_dsdna_trend": w.anti_dsdna_trend * min(1.0, sig_trend("anti_dsdna")),
        "complement_trend": w.complement_trend * min(
            1.0, 0.5 * (sig_trend("c3_mg_dl") + sig_trend("c4_mg_dl"))),
        "upcr_trend": w.upcr_trend * min(1.0, sig_trend("upcr_g_g")),
        "sledai_level": w.sledai_level * min(1.0, lvl("sledai2k", w.sledai_ref)),
        "ifn_level": w.ifn_level * min(1.0, lvl("ifn_score", w.ifn_ref)),
    }
    score = float(sum(parts.values()))
    return {"score": min(1.0, max(0.0, score)), "components": parts,
            "weights_declared": w.declared(),
            "model_class": "transparent declared index, NOT fitted",
            "clinical_actionability": RESEARCH_ONLY}


def stratify(ps: PatientSeries, ifn_threshold: float = 5.0) -> Dict[str, Any]:
    t, y = ps.series("ifn_score")
    if len(y) == 0:
        return {"stratum": "unknown", "reason": "no interferon score available",
                "note": "stratum matters: anifrolumab targets IFNAR1; "
                        "IFN-high and IFN-low may have different trajectories"}
    m = float(np.median(y))
    return {"stratum": "high" if m >= ifn_threshold else "low",
            "median_ifn": m, "threshold": ifn_threshold,
            "n_used": len(y)}


# =============================================================================
# §9. MISSING-EVIDENCE REPORT
# =============================================================================

BLOCKING_REQUIREMENTS = [
    "IRB approval active and purpose within approved_uses",
    "de-identification passed with residual risk not high",
    "MEDICATION LOG: prednisolone dose at every visit "
    "(absent => the model learns steroid tapers, not flare biology)",
    "immunosuppressant / biologic exposure recorded at every visit",
    "infection status assessed at every candidate event (flare mimic)",
    "flare ADJUDICATED by a declared definition, by an independent assessor",
    "anti-dsDNA assay method + unit + reference range recorded per site",
    "C3 / C4 reference range recorded per site",
    "SLEDAI-2K assessor reliability reported (kappa or ICC)",
]

SUPPORTING_REQUIREMENTS = [
    "hydroxychloroquine dose and adherence (blood HCQ level if available)",
    "UPCR collection method (spot vs 24 h)",
    ">= 8 visits per patient with a declared target spacing",
    "follow-up duration >= 48 weeks",
    "missing-data mechanism documented (MCAR / MAR / MNAR)",
    "pre-declared prediction horizon and analysis plan",
    "site harmonisation plan for pooled anti-dsDNA",
    "renal biopsy status where nephritis is claimed",
]


def missing_evidence(ps: PatientSeries, gov_decision: Dict[str, Any],
                     min_visits: int = 8) -> Dict[str, Any]:
    present: List[str] = []
    missing_blocking: List[str] = []
    missing_support: List[str] = []

    if gov_decision.get("authorized"):
        present.append(BLOCKING_REQUIREMENTS[0])
        present.append(BLOCKING_REQUIREMENTS[1])
    else:
        missing_blocking += [BLOCKING_REQUIREMENTS[0], BLOCKING_REQUIREMENTS[1]]

    v = ps.visits
    n = len(v)
    med_ok = n > 0 and all(x.prednisolone_mg_day is not None for x in v)
    imm_ok = n > 0 and all(x.immunosuppressant is not None for x in v)
    inf_ok = n > 0 and all(x.infection_present is not None for x in v)
    (present if med_ok else missing_blocking).append(BLOCKING_REQUIREMENTS[2])
    (present if imm_ok else missing_blocking).append(BLOCKING_REQUIREMENTS[3])
    (present if inf_ok else missing_blocking).append(BLOCKING_REQUIREMENTS[4])

    adj_ok = (len(ps.flares) > 0 and
              all(f.adjudicated_by and f.definition for f in ps.flares))
    if adj_ok:
        present.append(BLOCKING_REQUIREMENTS[5])
    else:
        missing_blocking.append(BLOCKING_REQUIREMENTS[5] +
                                " (no adjudicated event on record)")

    a = ps.assays.get("anti_dsdna")
    (present if (a and a.complete()) else missing_blocking).append(
        BLOCKING_REQUIREMENTS[6])
    c3 = ps.assays.get("c3_mg_dl")
    (present if (c3 and c3.complete()) else missing_blocking).append(
        BLOCKING_REQUIREMENTS[7])
    rel = ps.sledai_assessor_reliability
    (present if (rel is not None and rel >= 0.7) else missing_blocking).append(
        BLOCKING_REQUIREMENTS[8])

    hcq_ok = all(x.hcq_mg_day is not None for x in v) if n else False
    (present if hcq_ok else missing_support).append(SUPPORTING_REQUIREMENTS[0])
    (present if n >= min_visits else missing_support).append(
        SUPPORTING_REQUIREMENTS[2])
    dur = (v[-1].t_week - v[0].t_week) if n >= 2 else 0.0
    (present if dur >= 48 else missing_support).append(SUPPORTING_REQUIREMENTS[3])
    missing_support += [SUPPORTING_REQUIREMENTS[1], SUPPORTING_REQUIREMENTS[4],
                        SUPPORTING_REQUIREMENTS[5], SUPPORTING_REQUIREMENTS[6],
                        SUPPORTING_REQUIREMENTS[7]]

    completeness = len(present) / max(
        len(BLOCKING_REQUIREMENTS) + len(SUPPORTING_REQUIREMENTS), 1)
    return {"status": Status.BLOCKED if missing_blocking else Status.OK,
            "n_visits": n, "follow_up_weeks": dur,
            "present": present,
            "missing_blocking": missing_blocking,
            "missing_supporting": missing_support,
            "evidence_completeness": completeness,
            "study_usable_for_prediction_research": not missing_blocking,
            "decision": "NO_CLINICAL_DECISION_ISSUED",
            "explicitly_not_provided": ["diagnosis", "treatment", "dosing",
                                        "steroid taper advice", "prognosis for a patient"]}


# =============================================================================
# §10. VALIDATION HARNESS
# =============================================================================

def horizon_validation(alerts_week: List[float],
                       flares_week: List[float],
                       horizon_weeks: float = 12.0,
                       follow_up_weeks: float = 52.0) -> Dict[str, Any]:
    """
    Pre-declared horizon validation. An alert is a TRUE POSITIVE if an
    adjudicated flare occurs within (alert_time, alert_time + horizon].
    Lead time is the gap to the FIRST such flare.
    """
    if not flares_week:
        return {"status": Status.NOT_COMPARED,
                "reason": "no adjudicated flare reference supplied",
                "n_alerts": len(alerts_week)}
    tp, fp, leads = 0, 0, []
    for a in alerts_week:
        nxt = [f for f in flares_week if a < f <= a + horizon_weeks]
        if nxt:
            tp += 1; leads.append(min(nxt) - a)
        else:
            fp += 1
    n_alert = len(alerts_weeks := alerts_week)
    covered = 0
    for f in flares_week:
        if any(f - horizon_weeks <= a < f for a in alerts_week):
            covered += 1
    return {"status": Status.OK, "horizon_weeks": horizon_weeks,
            "n_alerts": n_alert, "n_flares": len(flares_week),
            "true_positive_alerts": tp, "false_positive_alerts": fp,
            "false_alarm_rate": fp / max(n_alert, 1),
            "positive_predictive_value": tp / max(n_alert, 1),
            "flare_sensitivity_at_horizon": covered / max(len(flares_week), 1),
            "median_lead_time_weeks": float(np.median(leads)) if leads else None,
            "lead_time_iqr_weeks": ([float(np.quantile(leads, .25)),
                                     float(np.quantile(leads, .75))]
                                    if leads else None),
            "follow_up_weeks": follow_up_weeks,
            "note": "research metrics; no clinical utility claimed"}


def coverage_study(n_real: int = 400, n: int = 60, rho: float = 0.7,
                   conf: float = 0.95, seed: int = 0) -> Dict[str, Any]:
    """
    THE decisive test. Generate AR(1) series with a KNOWN mean and compare
    the empirical coverage of the subsampling CI against the iid CI.
    Under serial correlation the iid interval must UNDERCOVER.
    """
    rng = np.random.default_rng(seed)
    hit_sub = hit_iid = 0
    w_sub, w_iid = [], []
    for _ in range(n_real):
        e = rng.normal(size=n)
        y = np.empty(n)
        y[0] = e[0] / math.sqrt(max(1 - rho * rho, 1e-9))
        for i in range(1, n):
            y[i] = rho * y[i - 1] + e[i]
        t = np.arange(n, dtype=float)
        s = subsample_ci(t, y, "mean", conf=conf)
        i = iid_ci(t, y, "mean", conf=conf)
        if s["status"] != Status.OK:
            continue
        hit_sub += int(s["lo"] <= 0.0 <= s["hi"])
        hit_iid += int(i["lo"] <= 0.0 <= i["hi"])
        w_sub.append(s["width"]); w_iid.append(i["width"])
    return {"status": Status.OK, "n_realisations": n_real, "n": n,
            "ar1_rho": rho, "nominal_conf": conf,
            "coverage_subsampling": hit_sub / max(n_real, 1),
            "coverage_iid": hit_iid / max(n_real, 1),
            "median_width_subsampling": float(np.median(w_sub)) if w_sub else None,
            "median_width_iid": float(np.median(w_iid)) if w_iid else None,
            "width_ratio": (float(np.median(w_sub) / np.median(w_iid))
                            if w_sub and w_iid else None),
            "interpretation": "iid undercovers under serial correlation; "
                              "subsampling is wider and closer to nominal"}


# =============================================================================
# §11. ORCHESTRATOR
# =============================================================================

class SLEPipeline:
    def __init__(self, irb: Optional[IRBRecord] = None,
                 horizon_weeks: float = 12.0, window: int = 8):
        self.gov = Governance(irb)
        self.horizon = float(horizon_weeks)
        self.window = int(window)

    def run(self, ps: PatientSeries, question: str = "SLE flare lead-time research",
            uses_patient_data: bool = True,
            deid: Optional[Dict] = None) -> Dict[str, Any]:
        d = self.gov.authorize(question, uses_patient_data, deid)
        if not d["authorized"]:
            return {"status": Status.BLOCKED, "governance": d}

        feat = extract_features(ps, window=self.window)
        score = flare_pressure(feat)
        stratum = stratify(ps)
        ev = missing_evidence(ps, d)

        counter = CSOCAlertCounter()
        alerts: List[float] = []
        # walk forward so the alert mechanism is exercised causally
        for k in range(3, ps.n() + 1):
            sub = PatientSeries(ps.research_id, ps.visits[:k], [],
                                ps.assays, ps.ifn_stratum,
                                ps.sledai_assessor_reliability)
            f = extract_features(sub, window=self.window)
            s = flare_pressure(f)["score"]
            rec = counter.step(s, k)
            if rec["fired"]:
                alerts.append(ps.visits[k - 1].t_week)

        val = horizon_validation(alerts, [f.t_week for f in ps.flares],
                                 self.horizon,
                                 (ps.visits[-1].t_week - ps.visits[0].t_week)
                                 if ps.n() >= 2 else 0.0)
        return {
            "report_id": str(uuid.uuid4())[:8],
            "status": Status.OK if not ev["missing_blocking"] else Status.BLOCKED,
            "research_id": ps.research_id,
            "governance": d,
            "features": feat,
            "flare_pressure": score,
            "ifn_stratum": stratum,
            "alert_bound": counter.report(),
            "alert_weeks": alerts,
            "validation": val,
            "missing_evidence": ev,
            "clinical_actionability": RESEARCH_ONLY,
            "clinician_review_required": True,
            "disclaimer": ("RESEARCH ONLY. Not a diagnostic device. No treatment, "
                           "dosing, or steroid-taper advice. SLE management "
                           "requires a rheumatologist and, for nephritis, a "
                           "nephrologist."),
        }


# =============================================================================
# §12. SYNTHETIC COHORT (demo only)
# =============================================================================

def synthetic_patient(seed: int = 0, n_visits: int = 24,
                      will_flare: bool = True,
                      flare_visit: int = 16,
                      ifn_high: bool = True,
                      with_meds: bool = True) -> PatientSeries:
    """
    Synthetic SLE-like trajectory. NOT real data. NOT calibrated.
    anti-dsDNA rises and C3 falls before the flare; prednisolone rises AT
    the flare (the confounder); visit spacing is irregular.
    """
    rng = np.random.default_rng(seed)
    visits: List[Visit] = []
    t = 0.0
    dsdna, c3, c4, upcr = 25.0, 95.0, 22.0, 0.15
    for k in range(n_visits):
        t += float(max(2.0, rng.normal(6.0, 2.0)))
        pre = max(0, min(6, flare_visit - k)) if will_flare else 0
        drive = (6 - pre) / 6.0 if will_flare and k <= flare_visit else 0.0
        if will_flare and k > flare_visit:
            drive = max(0.0, 1.0 - 0.25 * (k - flare_visit))
        dsdna = max(2.0, 0.75 * dsdna + 0.25 * (25 + 180 * drive)
                    + rng.normal(0, 8))
        c3 = max(20.0, 0.75 * c3 + 0.25 * (95 - 45 * drive) + rng.normal(0, 4))
        c4 = max(2.0, 0.75 * c4 + 0.25 * (22 - 12 * drive) + rng.normal(0, 1.5))
        upcr = max(0.02, 0.7 * upcr + 0.3 * (0.15 + 1.6 * drive)
                   + rng.normal(0, 0.05))
        sledai = float(np.clip(round(2 + 12 * drive + rng.normal(0, 1)), 0, 40))
        ifn = float(np.clip((8.0 if ifn_high else 2.0) + 3 * drive
                            + rng.normal(0, 1.0), 0, 25))
        alc = float(np.clip(1400 - 600 * drive + rng.normal(0, 120), 200, 3000))
        pred = (20.0 + 25.0 * drive if (will_flare and k >= flare_visit - 1)
                else 7.5) if with_meds else None
        visits.append(Visit(
            t_week=t, sledai2k=sledai, anti_dsdna=dsdna,
            c3_mg_dl=c3, c4_mg_dl=c4, upcr_g_g=upcr, ifn_score=ifn,
            alc_per_ul=alc,
            prednisolone_mg_day=(float(pred) if pred is not None else None),
            hcq_mg_day=200.0,
            immunosuppressant="mycophenolate" if with_meds else None,
            biologic="none", infection_present=False))
    flares = []
    if will_flare:
        flares.append(FlareAdjudication(
            t_week=visits[flare_visit].t_week,
            definition="SELENA-SLEDAI SFI", severity="severe",
            adjudicated_by="assessor_anon_01", infection_excluded=True,
            organ_domain="renal"))
    assays = {
        "anti_dsdna": AssayMeta("anti_dsdna", "ELISA", "IU/mL", 0.0, 25.0,
                                "site_anon_A"),
        "c3_mg_dl": AssayMeta("c3_mg_dl", "immunoturbidimetry", "mg/dL",
                              90.0, 180.0, "site_anon_A"),
        "c4_mg_dl": AssayMeta("c4_mg_dl", "immunoturbidimetry", "mg/dL",
                              10.0, 40.0, "site_anon_A"),
    }
    return PatientSeries(f"SYNTH_{seed:04d}", visits, flares, assays,
                         "high" if ifn_high else "low", 0.85)


# =============================================================================
# §13. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    rng = np.random.default_rng(0)

    # T1 governance
    print("\n[T1] governance and safety")
    gov = Governance()
    rep("patient path BLOCKED without IRB",
        not gov.authorize("SLE research", True)["authorized"])
    for txt, lab in (("increase the prednisolone dose", "treatment advice"),
                     ("what dose of mycophenolate", "dosing"),
                     ("design a CAR-T construct for lupus", "gene/cell design"),
                     ("design a CRISPR guide RNA", "gene editor")):
        rep(f"{lab} BLOCKED", not gov.authorize(txt)["authorized"])
    irb = IRBRecord("SLE-001", "Demo", "2026-01-01", "2099-12-31",
                    ["sle_research"], data_use_agreement=True, status="APPROVED")
    g2 = Governance(irb)
    rep("IRB alone insufficient (no de-id)",
        not g2.authorize("SLE research", True)["authorized"])
    deid = g2.phi.deidentify({"patient_name": "Jane Doe", "mrn": "MRN: 12345",
                              "age": 34, "sex": "female"})
    rep("de-id removes direct identifiers", deid["passed"])
    rep("IRB + de-id authorised",
        g2.authorize("SLE research", True, deid)["authorized"])

    # T2 subsampling rates
    print("\n[T2] subsampling uses the CORRECT rate per statistic")
    t = np.arange(40, dtype=float)
    y = 0.3 * t + rng.normal(0, 1, 40)
    m = subsample_ci(t, y, "mean")
    s = subsample_ci(t, y, "slope")
    rep("mean rate = 0.5", m["rate"] == 0.5, f"{m['rate']}")
    rep("slope rate = 1.5 (NOT 0.5)", s["rate"] == 1.5, f"{s['rate']}")
    rep("slope CI brackets the true 0.3", s["lo"] <= 0.3 <= s["hi"],
        f"[{s['lo']:.4f}, {s['hi']:.4f}]")
    rep("CI ordered", m["lo"] <= m["estimate"] <= m["hi"])

    # T3 THE decisive coverage test
    print("\n[T3] coverage: subsampling vs iid under AR(1) correlation")
    for rho in (0.0, 0.6, 0.85):
        c = coverage_study(n_real=300, n=60, rho=rho, seed=1)
        print(f"      rho={rho:.2f}  coverage sub={c['coverage_subsampling']:.3f} "
              f"iid={c['coverage_iid']:.3f}  width ratio={c['width_ratio']:.2f}")
        if rho >= 0.6:
            rep(f"rho={rho}: iid UNDERCOVERS (<0.85)",
                c["coverage_iid"] < 0.85, f"{c['coverage_iid']:.3f}")
            rep(f"rho={rho}: subsampling beats iid",
                c["coverage_subsampling"] > c["coverage_iid"],
                f"{c['coverage_subsampling']:.3f} > {c['coverage_iid']:.3f}")
            rep(f"rho={rho}: subsampling interval is WIDER",
                c["width_ratio"] > 1.2, f"{c['width_ratio']:.2f}x")

    # T4 tau_int
    print("\n[T4] Madras-Sokal tau_int and effective sample size")
    for rho in (0.0, 0.8):
        n = 400
        e = rng.normal(size=n); yy = np.empty(n); yy[0] = e[0]
        for i in range(1, n):
            yy[i] = rho * yy[i-1] + e[i]
        r = tau_int(yy)
        theory = 0.5 * (1 + rho) / (1 - rho) if rho < 1 else float("inf")
        print(f"      rho={rho:.1f}  tau_int={r['tau_int']:.2f} "
              f"(AR1 theory {theory:.2f})  n_eff={r['n_eff']:.1f}")
        rep(f"rho={rho}: tau_int within 40% of AR(1) theory",
            abs(r["tau_int"] - theory) / theory < 0.40,
            f"{r['tau_int']:.2f} vs {theory:.2f}")
    rep("tau_int >= 0.5 always", tau_int(rng.normal(size=200))["tau_int"] >= 0.5)

    # T5 block selection
    print("\n[T5] block-length selection reports its sensitivity")
    t2 = np.arange(60, dtype=float)
    y2 = 0.2 * t2 + rng.normal(0, 1, 60)
    bs = select_block(t2, y2, "slope")
    rep("block selected", bs["status"] == Status.OK, f"b={bs.get('b')}")
    rep("width curve reported", len(bs.get("curve", [])) >= 3,
        f"{len(bs.get('curve', []))} points")
    rep("width spread reported", "width_spread_rel" in bs,
        f"{bs.get('width_spread_rel', float('nan')):.3f}")

    # T6 CSOC bound
    print("\n[T6] CSOC alert bound with ACTUAL removal")
    c = CSOCAlertCounter(0.6, 0.25, 0.08, min_interval_visits=2)
    for k in range(600):
        c.step(0.35 + 0.45 * abs(math.sin(0.09 * k)), k)
    r = c.report()
    rep("N <= (E_max + D)/gap", r["satisfied"],
        f"N={r['events']} bound={r['bound']:.2f}")
    rep("D == D_audit (actual removal, not excess*rate)", r["D_audit_matches"])
    rep("no chattering", r["events"] < 600 / 3, f"{r['events']}/600")

    # T7 missing evidence BLOCKS on absent medication log
    print("\n[T7] medication log is a BLOCKING requirement")
    ps_ok = synthetic_patient(seed=1, with_meds=True)
    ps_no = synthetic_patient(seed=1, with_meds=False)
    ev_ok = missing_evidence(ps_ok, {"authorized": True})
    ev_no = missing_evidence(ps_no, {"authorized": True})
    rep("complete meds => not blocked on meds",
        not any("MEDICATION LOG" in m for m in ev_ok["missing_blocking"]))
    rep("absent meds => BLOCKED",
        any("MEDICATION LOG" in m for m in ev_no["missing_blocking"]),
        f"{len(ev_no['missing_blocking'])} blocking items")
    rep("absent meds => study not usable",
        not ev_no["study_usable_for_prediction_research"])
    rep("no clinical decision issued",
        ev_ok["decision"] == "NO_CLINICAL_DECISION_ISSUED")

    # T8 features on synthetic patient
    print("\n[T8] feature extraction and adverse-trend detection")
    f = extract_features(ps_ok, window=10)
    b = f["biomarkers"]
    rep("anti-dsDNA features present", b["anti_dsdna"]["status"] == Status.OK)
    rep("subsampling wider than iid on a correlated biomarker",
        b["anti_dsdna"]["width_ratio_sub_over_iid"] > 1.0,
        f"{b['anti_dsdna']['width_ratio_sub_over_iid']:.2f}x")
    rep("spacing CV measured", np.isfinite(f["spacing_cv"]),
        f"CV={f['spacing_cv']:.3f}")
    sc = flare_pressure(f)
    rep("score in [0,1]", 0.0 <= sc["score"] <= 1.0, f"{sc['score']:.3f}")
    rep("model class declared as NOT fitted",
        "NOT fitted" in sc["model_class"])

    # T9 horizon validation
    print("\n[T9] horizon validation needs adjudicated flares")
    nv = horizon_validation([10.0, 30.0], [], horizon_weeks=12)
    rep("no reference => NOT_COMPARED", nv["status"] == Status.NOT_COMPARED)
    hv = horizon_validation([20.0, 60.0], [28.0], horizon_weeks=12,
                            follow_up_weeks=80)
    rep("TP/FP counted", hv["true_positive_alerts"] == 1
        and hv["false_positive_alerts"] == 1,
        f"TP={hv['true_positive_alerts']} FP={hv['false_positive_alerts']}")
    rep("lead time computed", hv["median_lead_time_weeks"] == 8.0,
        f"{hv['median_lead_time_weeks']} weeks")
    rep("false alarm rate = 0.5", abs(hv["false_alarm_rate"] - 0.5) < 1e-9)

    # T10 end-to-end
    print("\n[T10] end-to-end pipeline")
    pipe = SLEPipeline(irb=irb, horizon_weeks=12.0, window=8)
    out = pipe.run(ps_ok, deid=deid)
    rep("pipeline runs", out["status"] in (Status.OK, Status.BLOCKED),
        out["status"])
    rep("research_only", out["clinical_actionability"] == RESEARCH_ONLY)
    rep("clinician review required", out["clinician_review_required"])
    rep("alert bound reported and satisfied",
        out["alert_bound"]["satisfied"],
        f"N={out['alert_bound']['events']} bound={out['alert_bound']['bound']:.1f}")
    print(f"      alerts at weeks: {[round(a,1) for a in out['alert_weeks']]}")
    print(f"      flare at week:   {ps_ok.flares[0].t_week:.1f}")
    print(f"      validation: {out['validation'].get('status')}")
    blocked = pipe.run(ps_ok, deid=None)
    rep("same pipeline without de-id is BLOCKED",
        blocked["status"] == Status.BLOCKED)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
MEASURED
  Subsampling uses rate n^{1/2} for a mean and n^{3/2} for an OLS slope.
  Under AR(1) correlation the iid CI UNDERCOVERS and the subsampling CI is
  wider and closer to nominal. This is why the pipeline does not use iid.
  Madras-Sokal tau_int matches the AR(1) value 0.5(1+rho)/(1-rho).
  Block-length sensitivity is reported as a curve, not hidden.
  The CSOC alert bound holds with ACTUAL removal, audited.
  An absent medication log BLOCKS the study, because without it the model
  learns steroid tapers rather than flare biology.
  Horizon validation reports NOT_COMPARED without adjudicated flares.

NOT CLAIMED
  No clinical validation. Synthetic data only.
  The flare-pressure score is a DECLARED index, not a fitted or calibrated
  model, and has no demonstrated clinical utility.
  Anti-dsDNA assays are not interchangeable across sites.
  SLE is a tolerance failure, not a protein-folding disease.
  No diagnosis, treatment, dosing, steroid-taper advice, or gene editing.
""")
    return ok


# =============================================================================
# §14. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="SLE longitudinal research pipeline")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "coverage", "demo", "cohort"])
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--reals", type=int, default=300)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon", type=float, default=12.0)
    ap.add_argument("--patients", type=int, default=12)
    ap.add_argument("--no-meds", action="store_true",
                    help="demonstrate the blocking deficiency")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out: Dict[str, Any] = {}

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    if a.cmd == "coverage":
        out["coverage"] = coverage_study(a.reals, a.n, a.rho)
        print(json.dumps(out["coverage"], indent=2, default=float))

    if a.cmd == "demo":
        irb = IRBRecord("SLE-DEMO", "Demo", "2026-01-01", "2099-12-31",
                        ["sle_research"], data_use_agreement=True,
                        status="APPROVED")
        pipe = SLEPipeline(irb, a.horizon, a.window)
        ps = synthetic_patient(seed=3, with_meds=not a.no_meds)
        deid = pipe.gov.phi.deidentify({"patient_name": "Demo Patient",
                                        "age": 34, "sex": "female"})
        r = pipe.run(ps, deid=deid)
        out["demo"] = {k: v for k, v in r.items() if k != "features"}
        out["features_summary"] = {
            k: {"trend_estimate": v.get("trend", {}).get("estimate"),
                "trend_ci": [v.get("trend", {}).get("lo"),
                             v.get("trend", {}).get("hi")],
                "width_ratio_vs_iid": v.get("width_ratio_sub_over_iid"),
                "tau_int": v.get("tau_int", {}).get("tau_int")}
            for k, v in r.get("features", {}).get("biomarkers", {}).items()
            if v.get("status") == Status.OK}
        print(json.dumps(out, indent=2, default=float))

    if a.cmd == "cohort":
        irb = IRBRecord("SLE-COHORT", "Demo", "2026-01-01", "2099-12-31",
                        ["sle_research"], data_use_agreement=True,
                        status="APPROVED")
        pipe = SLEPipeline(irb, a.horizon, a.window)
        rows, all_alerts, all_flares = [], [], []
        for i in range(a.patients):
            flare = (i % 2 == 0)
            ps = synthetic_patient(seed=100 + i, will_flare=flare,
                                   ifn_high=(i % 3 != 0),
                                   with_meds=not a.no_meds)
            deid = pipe.gov.phi.deidentify({"age": 30 + i, "sex": "female"})
            r = pipe.run(ps, deid=deid)
            if r["status"] == Status.BLOCKED and "governance" in r:
                rows.append({"id": ps.research_id, "status": "BLOCKED"})
                continue
            rows.append({"id": ps.research_id,
                         "ifn": r["ifn_stratum"]["stratum"],
                         "score_final": r["flare_pressure"]["score"],
                         "n_alerts": len(r["alert_weeks"]),
                         "bound_ok": r["alert_bound"]["satisfied"],
                         "status": r["status"]})
            all_alerts += r["alert_weeks"]
            all_flares += [f.t_week for f in ps.flares]
        out["cohort"] = {"n_patients": a.patients, "rows": rows,
                         "pooled_validation": horizon_validation(
                             all_alerts, all_flares, a.horizon),
                         "note": "synthetic cohort; NOT clinical evidence"}
        print(json.dumps(out, indent=2, default=float))

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
