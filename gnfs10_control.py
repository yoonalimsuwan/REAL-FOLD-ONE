#!/usr/bin/env python3
"""
=============================================================================
GNFS-10 RESEARCH CONTROL LAYER
Murphy-E polynomial scoring + real lattice-sieving log parser
+ adaptive world-model controller
+ diagnostics from five papers:
  - Delta_Min_Isodiametric_Closure
  - Six_Problems_Closed_As_Far_As_They_Go
  - Revision22_Coercivity_and_Fractafold_Transmission
  - Stochastic_NoZeno_SESI_Repair
  - Paper9_Statistical_Dictionary_FiniteSample
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
            My Soul Move By Power of Holy Spirit ;
            What MSPS NETWORK SEE, Lord Buddha Knows.
License   : MIT | Year: 2026

HONEST SCOPE
------------
This file is a RESEARCH CONTROL AND DIAGNOSTIC scaffold.

It provides:
  1) Murphy-E-style polynomial scoring proxy
  2) lattice-sieve log parsing
  3) assumption-light statistics: Hill tail, subsampling CI, CUSUM
  4) isodiametric kappa-fatness filter from Delta_Min
  5) No-Zeno event/dwell guard from Stochastic_NoZeno
  6) fractafold/Kirchhoff pipeline flux balancer from Revision 22
  7) GRU world model for yield prediction
  8) adaptive controller decisions

It does NOT provide:
  ❌ a production GNFS implementation
  ❌ a full number-field sieve factorization pipeline
  ❌ any method claimed to factor 1000-4000 digit semiprimes
  ❌ any cryptographic attack capability
  ❌ any proof that Hilbert-Pólya or Hodge theory factors integers

The five papers are used as DIAGNOSTIC/CONTROL principles, not as magic
factorization theorems.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

VERSION = "1.0.0"

ATTRIBUTION = (
    "PAI AND Yoon A Limsuwan / MSPS NETWORK — "
    "My Soul Move By Power of Holy Spirit ; "
    "What MSPS NETWORK SEE, Lord Buddha Knows."
)

BANNER = f"""
=============================================================================
GNFS-10 RESEARCH CONTROL LAYER v{VERSION}
{ATTRIBUTION}

This is NOT a factorization tool.
It is a research/diagnostics/control scaffold.
=============================================================================
"""


# =============================================================================
# 0. Small numeric utilities
# =============================================================================

def _quantile(sorted_vals: List[float], q: float) -> float:
    """Quantile for already-sorted list. q in [0,1]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _linreg(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Simple linear regression slope, intercept."""
    n = len(x)
    if n < 2:
        return 0.0, (y[0] if y else 0.0)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _ema(values: List[float], alpha: float = 0.2) -> List[float]:
    """Exponential moving average."""
    out: List[float] = []
    if not values:
        return out
    s = float(values[0])
    out.append(s)
    for v in values[1:]:
        s = alpha * float(v) + (1.0 - alpha) * s
        out.append(s)
    return out


def _safe_log1p(x: float) -> float:
    try:
        return math.log1p(max(float(x), 0.0))
    except Exception:
        return 0.0


# =============================================================================
# 1. Data structures
# =============================================================================

@dataclass
class PolynomialSpec:
    """
    Minimal polynomial-pair representation.

    For real GNFS one usually has algebraic polynomial f(x) and linear
    rational polynomial g(x) = x - m, but this scaffold accepts general
    coefficient lists for research diagnostics.
    """
    name: str = "unknown"
    f_coeffs: List[float] = field(default_factory=lambda: [1.0, 0.0, 1.0])
    g_coeffs: List[float] = field(default_factory=lambda: [1.0, 0.0])
    m: int = 0
    degree: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        self.degree = max(len(self.f_coeffs) - 1, 0)


@dataclass
class LatticeBasis2D:
    """
    2D lattice basis used for special-q sieving diagnostics.
    Real GNFS lattice basis construction is more complex; this is only
    for kappa-fatness / isodiametric diagnostics.
    """
    b1: Tuple[float, float] = (1.0, 0.0)
    b2: Tuple[float, float] = (0.0, 1.0)

    def as_dict(self) -> Dict[str, Any]:
        return {"b1": list(self.b1), "b2": list(self.b2)}


@dataclass
class LogRecord:
    line_no: int
    raw: str
    q: Optional[int] = None
    rels: float = 0.0
    special_q: Optional[int] = None
    time_s: float = 0.0
    skew: float = 1.0
    fb_size: float = 0.0
    lattice_basis: Optional[LatticeBasis2D] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.lattice_basis is not None:
            d["lattice_basis"] = self.lattice_basis.as_dict()
        return d


@dataclass
class ControllerDecision:
    action: str
    reason: str
    confidence: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 2. Murphy-E style polynomial scoring
# =============================================================================

class MurphyEScorer:
    """
    Murphy-E-style scoring PROXY.

    Real Murphy E is a sophisticated smoothness-probability estimate over
    polynomial values. This class implements a transparent proxy:

      root_score = sum_p (roots_f(p) + roots_g(p)) / p
      size_penalty = log coefficient size + skew proxy
      score_proxy = root_score / (1 + size_penalty)

    It is NOT the exact Murphy E from CADO-NFS/GGNFS.
    """

    def __init__(self, prime_bound: int = 200):
        self.prime_bound = max(10, int(prime_bound))
        self.primes = self._sieve_primes(self.prime_bound)

    @staticmethod
    def _sieve_primes(n: int) -> List[int]:
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                step = i
                start = i * i
                sieve[start:n + 1:step] = [False] * (((n - start) // step) + 1)
        return [i for i, ok in enumerate(sieve) if ok]

    @staticmethod
    def _eval_poly_mod(coeffs: List[float], x: int, p: int) -> int:
        acc = 0
        for c in reversed(coeffs):
            acc = (acc * x + int(round(c))) % p
        return acc

    def _count_roots_mod(self, coeffs: List[float], p: int) -> int:
        if not coeffs:
            return 0
        cnt = 0
        for x in range(p):
            if self._eval_poly_mod(coeffs, x, p) == 0:
                cnt += 1
        return cnt

    @staticmethod
    def _coeff_size_score(coeffs: List[float]) -> float:
        if not coeffs:
            return 0.0
        vals = [abs(float(c)) for c in coeffs]
        return math.log1p(max(vals)) + 0.1 * math.log1p(sum(vals))

    def score_polynomial(self, poly: PolynomialSpec) -> Dict[str, Any]:
        root_score = 0.0
        f_roots_total = 0
        g_roots_total = 0
        p_used = 0

        for p in self.primes:
            rf = self._count_roots_mod(poly.f_coeffs, p)
            rg = self._count_roots_mod(poly.g_coeffs, p)
            root_score += (rf + rg) / float(p)
            f_roots_total += rf
            g_roots_total += rg
            p_used += 1

        f_size = self._coeff_size_score(poly.f_coeffs)
        g_size = self._coeff_size_score(poly.g_coeffs)
        size_penalty = f_size + 0.25 * g_size

        # crude skew proxy: leading/trailing coefficient ratio
        try:
            lead = abs(float(poly.f_coeffs[-1]))
            const = abs(float(poly.f_coeffs[0]))
            skew_proxy = math.log1p(lead / (const + 1e-12))
        except Exception:
            skew_proxy = 0.0

        score = root_score / (1.0 + size_penalty + 0.1 * skew_proxy)

        return {
            "name": poly.name,
            "degree": poly.degree,
            "prime_bound": self.prime_bound,
            "primes_used": p_used,
            "f_roots_total": f_roots_total,
            "g_roots_total": g_roots_total,
            "root_score": root_score,
            "size_penalty": size_penalty,
            "skew_proxy": skew_proxy,
            "murphy_e_proxy": score,
            "note": "Proxy score only. Not the exact Murphy E.",
        }


# =============================================================================
# 3. Real lattice-sieve log parser
# =============================================================================

class LatticeSieveLogParser:
    """
    Parser for key=value style lattice-sieve logs.

    Supports common fields:
      q, special_q, specialq, rels, relations, found, time, sec,
      skew, fb, factor_base, b1, b2 basis fields

    Example lines:
      sieving q=104729 rels=1342 time=3.41 skew=1.21 fb=7000
      special_q=104729 relations_found=1342 elapsed_s=3.41
      lattice_b1=104729,1 lattice_b2=731,104729

    This parser is deliberately permissive: unknown numeric fields are
    stored in metrics.
    """

    KEY_RE = re.compile(
        r"([A-Za-z_][A-Za-z0-9_\.]*)\s*=\s*"
        r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|"
        r"[-+]?[0-9]+(?:/[0-9]+)?|"
        r"[A-Za-z0-9_\-]+)"
    )

    PAIR_RE = re.compile(
        r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*[,;]\s*"
        r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    )

    Q_KEYS = {"q", "special_q", "specialq", "q0", "special_prime"}
    REL_KEYS = {"rels", "relations", "relations_found", "found", "rel_count"}
    TIME_KEYS = {"time", "sec", "seconds", "elapsed", "elapsed_s", "wall_s"}
    SKEW_KEYS = {"skew", "poly_skew"}
    FB_KEYS = {"fb", "factor_base", "fb_size", "factor_base_size"}

    def parse_text(self, text: str) -> List[LogRecord]:
        records: List[LogRecord] = []
        basis_b1: Optional[Tuple[float, float]] = None
        basis_b2: Optional[Tuple[float, float]] = None

        for line_no, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue

            rec = LogRecord(line_no=line_no, raw=raw.strip())

            # detect basis pairs if present
            if "lattice_b1" in raw or "basis_b1" in raw or "b1=" in raw:
                m = self.PAIR_RE.search(raw)
                if m:
                    basis_b1 = (float(m.group(1)), float(m.group(2)))
            if "lattice_b2" in raw or "basis_b2" in raw or "b2=" in raw:
                m = self.PAIR_RE.search(raw)
                if m:
                    basis_b2 = (float(m.group(1)), float(m.group(2)))

            for key, val in self.KEY_RE.findall(raw):
                k = key.lower()
                num = self._to_float(val)
                if num is None:
                    continue

                if k in self.Q_KEYS:
                    rec.q = int(num)
                    rec.special_q = int(num)
                elif k in self.REL_KEYS:
                    rec.rels = float(num)
                elif k in self.TIME_KEYS:
                    rec.time_s = float(num)
                elif k in self.SKEW_KEYS:
                    rec.skew = float(num)
                elif k in self.FB_KEYS:
                    rec.fb_size = float(num)
                else:
                    rec.metrics[k] = float(num)

            if basis_b1 is not None and basis_b2 is not None:
                rec.lattice_basis = LatticeBasis2D(b1=basis_b1, b2=basis_b2)

            # keep the record only if it carries useful numeric payload
            if rec.q is not None or rec.rels != 0.0 or rec.metrics:
                records.append(rec)

        return records

    def parse_file(self, path: str) -> List[LogRecord]:
        p = Path(path)
        if not p.exists():
            return []
        return self.parse_text(p.read_text(encoding="utf-8", errors="ignore"))

    @staticmethod
    def _to_float(val: str) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            try:
                if "/" in val:
                    a, b = val.split("/", 1)
                    return float(a) / float(b)
            except Exception:
                return None
        return None


# =============================================================================
# 4. Paper 9: assumption-light statistics
# =============================================================================

class AssumptionLightStatistics:
    """
    Finite-sample diagnostics inspired by Paper 9:
      - Hill tail-index estimator
      - subsampling mean confidence interval
      - simple CUSUM change-point detector

    These are assumption-light but NOT assumption-free.
    """

    @staticmethod
    def hill_tail_index(values: List[float], k: Optional[int] = None) -> Dict[str, Any]:
        vals = [float(v) for v in values if v is not None and float(v) > 0.0]
        if len(vals) < 10:
            return {"theta": None, "k": 0, "n": len(vals), "note": "insufficient data"}

        s = sorted(vals, reverse=True)
        if k is None:
            k = max(5, len(s) // 10)
        k = min(k, len(s) - 1)

        threshold = s[k] if k < len(s) else max(s[-1] * 0.5, 1e-12)
        if threshold <= 0:
            threshold = 1e-12

        logs = []
        for x in s[:k]:
            if x > threshold:
                logs.append(math.log(x / threshold))

        if not logs:
            return {"theta": None, "k": k, "n": len(vals), "note": "degenerate top order stats"}

        mean_log = sum(logs) / len(logs)
        theta = 1.0 / mean_log if mean_log > 1e-12 else float("inf")

        return {
            "theta": theta,
            "k": k,
            "n": len(vals),
            "threshold": threshold,
            "interpretation": (
                "theta > 1 suggests finite mean heavy-tail; "
                "theta <= 1 suggests very heavy tail; "
                "theta near inf suggests light tail"
            ),
        }

    @staticmethod
    def subsample_mean_ci(values: List[float], block: Optional[int] = None,
                          confidence: float = 0.95) -> Dict[str, Any]:
        vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        n = len(vals)
        if n < 4:
            m = statistics.fmean(vals) if vals else 0.0
            return {"mean": m, "lo": m, "hi": m, "n": n, "note": "too few samples"}

        if block is None:
            block = max(2, int(math.sqrt(n)))
        block = min(block, n - 1)

        overall = statistics.fmean(vals)
        subs = []
        for i in range(0, n - block + 1):
            subs.append(statistics.fmean(vals[i:i + block]))

        centered = [math.sqrt(block) * (x - overall) for x in subs]
        centered.sort()
        alpha = 1.0 - confidence
        q_lo = _quantile(centered, alpha / 2.0)
        q_hi = _quantile(centered, 1.0 - alpha / 2.0)

        lo = overall - q_hi / math.sqrt(n)
        hi = overall - q_lo / math.sqrt(n)

        return {
            "mean": overall,
            "lo": min(lo, hi),
            "hi": max(lo, hi),
            "n": n,
            "block": block,
            "confidence": confidence,
        }

    @staticmethod
    def cusum_change_points(values: List[float], threshold: float = 3.0,
                            drift: float = 0.25) -> Dict[str, Any]:
        vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if len(vals) < 8:
            return {"change_points": [], "n": len(vals), "note": "too few samples"}

        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals)
        if std < 1e-12:
            std = 1.0

        pos = 0.0
        neg = 0.0
        cps: List[int] = []

        for i, x in enumerate(vals):
            z = (x - mean) / std
            pos = max(0.0, pos + z - drift)
            neg = max(0.0, neg - z - drift)
            if pos > threshold or neg > threshold:
                cps.append(i)
                pos = 0.0
                neg = 0.0

        return {
            "change_points": cps,
            "n": len(vals),
            "threshold": threshold,
            "drift": drift,
        }


# =============================================================================
# 5. Delta_Min: isodiametric kappa-fatness filter
# =============================================================================

class IsodiametricLatticeFilter:
    """
    Inspired by Delta_Min_Isodiametric_Closure.

    This is used as a GEOMETRIC GUARD: if a special-q lattice basis is too
    thin/slivery, expected sieving yield is unstable. We reject or flag it.

    kappa-fatness is approximated by eccentricity of the 2D parallelogram.
    """

    def __init__(self, kappa_max: float = 12.0, a_min: float = 1e-3):
        self.kappa_max = float(kappa_max)
        self.a_min = float(a_min)
        self.omega_1 = 2.0  # unit 1-sphere measure

    def basis_fatness(self, basis: LatticeBasis2D) -> Dict[str, Any]:
        b1 = basis.b1
        b2 = basis.b2
        area = abs(b1[0] * b2[1] - b1[1] * b2[0])
        n1 = math.hypot(b1[0], b1[1])
        n2 = math.hypot(b2[0], b2[1])

        if area <= 1e-12 or n1 <= 1e-12 or n2 <= 1e-12:
            return {
                "ok": False,
                "kappa": float("inf"),
                "area": area,
                "width": 0.0,
                "note": "degenerate lattice basis",
            }

        # width relative to longest vector
        width = area / max(n1, n2)
        min_side = min(n1, n2)
        kappa = max(min_side / max(width, 1e-12), 1.0)

        return {
            "ok": kappa <= self.kappa_max,
            "kappa": kappa,
            "area": area,
            "width": width,
            "note": "kappa-fatness approximation for 2D basis",
        }

    def delta_min(self, kappa: float) -> float:
        """
        δ_min = (a_min / omega_{n-1})^{1/(n-1)} / kappa, with n=2 here.
        """
        if kappa <= 0 or not math.isfinite(kappa):
            return 0.0
        return (self.a_min / self.omega_1) / kappa

    def check_record(self, rec: LogRecord) -> Dict[str, Any]:
        if rec.lattice_basis is None:
            return {
                "ok": True,
                "kappa": 1.0,
                "delta_min": self.delta_min(1.0),
                "note": "no lattice basis in log; filter bypassed",
            }
        fat = self.basis_fatness(rec.lattice_basis)
        fat["delta_min"] = self.delta_min(fat["kappa"])
        return fat


# =============================================================================
# 6. Stochastic No-Zeno guard
# =============================================================================

class NoZenoGuard:
    """
    Inspired by Stochastic_NoZeno_SESI_Repair.

    Prevents controller/changepoint resets from firing in a Zeno-like burst.
    This is an engineering dwell-time guard, not a proof about PDE regularity.
    """

    def __init__(self, min_dwell_s: float = 0.05, max_events: int = 5,
                 window_s: float = 1.0):
        self.min_dwell_s = float(min_dwell_s)
        self.max_events = int(max_events)
        self.window_s = float(window_s)
        self.events: List[float] = []
        self.denied: List[Dict[str, Any]] = []

    def allow_event(self, now: Optional[float] = None) -> Tuple[bool, str]:
        t = float(time.monotonic() if now is None else now)

        # remove old events outside window
        self.events = [x for x in self.events if t - x <= self.window_s]

        if self.events:
            last = self.events[-1]
            if t - last < self.min_dwell_s:
                reason = f"min dwell violated: {t-last:.6f} < {self.min_dwell_s}"
                self.denied.append({"t": t, "reason": reason})
                return False, reason

            if len(self.events) >= self.max_events:
                reason = f"too many events: {len(self.events)} >= {self.max_events}"
                self.denied.append({"t": t, "reason": reason})
                return False, reason

        self.events.append(t)
        return True, "allowed"

    def summary(self) -> Dict[str, Any]:
        return {
            "events": len(self.events),
            "denied": len(self.denied),
            "min_dwell_s": self.min_dwell_s,
            "max_events": self.max_events,
            "window_s": self.window_s,
        }


# =============================================================================
# 7. Six Problems: change-point tower analyzer
# =============================================================================

class ChangePointTowerAnalyzer:
    """
    Inspired by Six_Problems_Closed_As_Far_As_They_Go.

    Given change points, estimate block lengths and classify growth:
      - regular / finite-mean
      - regularly varying
      - double-exponential-like

    This is a diagnostic classification, not a theorem prover.
    """

    def analyze(self, change_points: List[int], n_total: int) -> Dict[str, Any]:
        if len(change_points) < 3:
            return {
                "status": "insufficient_change_points",
                "n_cp": len(change_points),
            }

        pts = sorted([int(x) for x in change_points if 0 <= x <= n_total])
        if len(pts) < 3:
            return {
                "status": "insufficient_change_points",
                "n_cp": len(pts),
            }

        lengths = []
        prev = 0
        for p in pts:
            if p > prev:
                lengths.append(p - prev)
            prev = p
        if n_total > prev:
            lengths.append(n_total - prev)

        lengths = [max(1, int(x)) for x in lengths]
        if len(lengths) < 3:
            return {"status": "too_few_blocks", "n_blocks": len(lengths)}

        log_len = [math.log(x) for x in lengths]
        idx = list(range(1, len(log_len) + 1))
        log_idx = [math.log(x) for x in idx]

        slope_regular, _ = _linreg(log_idx, log_len)
        loglog_len = [math.log(max(x, 1.0) + 1.0) for x in lengths]
        slope_dexp, _ = _linreg([float(x) for x in idx], loglog_len)

        if slope_dexp > 0.25 and slope_regular > 1.5:
            cls = "double_exponential_like"
        elif slope_regular > 1.0:
            cls = "regularly_varying_like"
        elif slope_regular <= 0.25:
            cls = "finite_mean_like"
        else:
            cls = "intermediate"

        return {
            "status": "ok",
            "n_change_points": len(pts),
            "n_blocks": len(lengths),
            "block_lengths": lengths[:50],
            "slope_regular": slope_regular,
            "slope_double_exp": slope_dexp,
            "classification": cls,
        }


# =============================================================================
# 8. Revision22: fractafold pipeline balancer
# =============================================================================

class FractafoldPipelineBalancer:
    """
    Inspired by Revision22_Coercivity_and_Fractafold_Transmission.

    We model the GNFS pipeline as nodes with fluxes:
      poly_select -> sieving -> filtering -> linear_algebra

    Kirchhoff-style balance: accumulation at a node indicates pressure /
    bottleneck. This is a control heuristic, not a proof of coercivity.
    """

    NODES = ("poly_select", "sieving", "filtering", "linear_algebra")

    def __init__(self, capacities: Optional[Dict[str, float]] = None):
        self.capacities = capacities or {
            "poly_select": 1.0,
            "sieving": 10.0,
            "filtering": 8.0,
            "linear_algebra": 4.0,
        }

    def check_flux(self, flux: Dict[str, float]) -> Dict[str, Any]:
        pressure: Dict[str, float] = {}
        bottlenecks: List[str] = []

        for node in self.NODES:
            inflow = float(flux.get(f"{node}_in", 0.0))
            outflow = float(flux.get(f"{node}_out", 0.0))
            cap = max(float(self.capacities.get(node, 1.0)), 1e-12)
            accum = inflow - outflow
            p = accum / cap
            pressure[node] = p
            if p > 0.75:
                bottlenecks.append(node)

        return {
            "pressure": pressure,
            "bottlenecks": bottlenecks,
            "balanced": len(bottlenecks) == 0,
            "note": "Kirchhoff-style flux balance heuristic",
        }

    def flux_from_records(self, records: List[LogRecord],
                          window: int = 10) -> Dict[str, float]:
        if not records:
            return {}
        tail = records[-window:]
        rels = [max(float(r.rels), 0.0) for r in tail]
        times = [max(float(r.time_s), 1e-6) for r in tail]
        sieve_rate = sum(rels) / sum(times) if times else 0.0

        return {
            "poly_select_in": 1.0,
            "poly_select_out": 1.0,
            "sieving_in": sieve_rate,
            "sieving_out": sieve_rate * 0.95,
            "filtering_in": sieve_rate * 0.95,
            "filtering_out": sieve_rate * 0.80,
            "linear_algebra_in": sieve_rate * 0.80,
            "linear_algebra_out": sieve_rate * 0.55,
        }


# =============================================================================
# 9. World model: GRU predictor
# =============================================================================

FEATURE_NAMES = [
    "log_q",
    "log_rels",
    "smooth_rels",
    "hill_theta",
    "cusum_flag",
    "fatness_kappa",
    "delta_min",
    "pipeline_pressure",
]

FEATURE_DIM = len(FEATURE_NAMES)


if HAS_TORCH:

    class YieldGRU(nn.Module):
        def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 32):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])


class WorldModelWrapper:
    """
    Small sequence model for predicting the next feature vector.
    Falls back to persistence if Torch is unavailable or data is too small.
    """

    def __init__(self, seq_len: int = 4, hidden_dim: int = 32, epochs: int = 30,
                 lr: float = 1e-2):
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.model = None
        self.mean: Optional[List[float]] = None
        self.std: Optional[List[float]] = None
        self.mode = "fallback"

    def fit(self, features: List[List[float]]) -> Dict[str, Any]:
        if not HAS_TORCH:
            self.mode = "fallback_no_torch"
            return {"mode": self.mode, "status": "no torch"}

        if len(features) < self.seq_len + 6:
            self.mode = "fallback_small_data"
            return {"mode": self.mode, "status": "too few samples"}

        X = torch.tensor(features, dtype=torch.float32)
        self.mean = X.mean(dim=0).tolist()
        std = X.std(dim=0).tolist()
        self.std = [s if s > 1e-8 else 1.0 for s in std]

        mean_t = torch.tensor(self.mean, dtype=torch.float32)
        std_t = torch.tensor(self.std, dtype=torch.float32)
        Xn = (X - mean_t) / std_t

        seqs = []
        targets = []
        for i in range(len(Xn) - self.seq_len):
            seqs.append(Xn[i:i + self.seq_len])
            targets.append(Xn[i + self.seq_len])

        if len(seqs) < 4:
            self.mode = "fallback_small_data"
            return {"mode": self.mode, "status": "too few sequences"}

        xb = torch.stack(seqs)
        yb = torch.stack(targets)

        self.model = YieldGRU(input_dim=FEATURE_DIM, hidden_dim=self.hidden_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        losses = []
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        self.mode = "torch_gru"
        return {
            "mode": self.mode,
            "status": "trained",
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "n_sequences": len(seqs),
        }

    def predict_next(self, tail: List[List[float]]) -> List[float]:
        if not tail:
            return [0.0] * FEATURE_DIM

        if self.mode != "torch_gru" or self.model is None or self.mean is None or self.std is None:
            return list(tail[-1])

        seq = tail[-self.seq_len:]
        if len(seq) < self.seq_len:
            pad = [seq[0]] * (self.seq_len - len(seq))
            seq = pad + seq

        mean_t = torch.tensor(self.mean, dtype=torch.float32)
        std_t = torch.tensor(self.std, dtype=torch.float32)
        x = torch.tensor(seq, dtype=torch.float32)
        xn = (x - mean_t) / std_t

        self.model.eval()
        with torch.no_grad():
            yn = self.model(xn.unsqueeze(0))[0]
        y = yn * std_t + mean_t
        return y.tolist()


# =============================================================================
# 10. Adaptive controller
# =============================================================================

class AdaptiveWorldModelController:
    """
    Combines diagnostics and world-model prediction into control actions.

    Actions are advisory only:
      hold
      continue
      reduce_special_q
      increase_factor_base
      slow_sieving_rebalance
      consider_new_polynomial
      pause_for_human_review
    """

    def __init__(self, no_zeno: Optional[NoZenoGuard] = None):
        self.no_zeno = no_zeno or NoZenoGuard()
        self.last_action = "hold"

    def decide(
        self,
        records: List[LogRecord],
        features: List[List[float]],
        world: WorldModelWrapper,
        poly_score: Optional[Dict[str, Any]] = None,
        iso_report: Optional[Dict[str, Any]] = None,
        tower_report: Optional[Dict[str, Any]] = None,
        flux_report: Optional[Dict[str, Any]] = None,
    ) -> ControllerDecision:

        warnings: List[str] = []

        if not records:
            return ControllerDecision(
                action="pause_for_human_review",
                reason="no lattice-sieve log records parsed",
                confidence=0.0,
                metrics={},
                warnings=["empty log"],
            )

        tail = features[-world.seq_len:] if features else []
        pred = world.predict_next(tail)
        current = features[-1] if features else [0.0] * FEATURE_DIM

        cur_rels = current[1] if len(current) > 1 else 0.0
        pred_rels = pred[1] if len(pred) > 1 else 0.0

        # de-normalize approximate if world model normalized internally
        ratio = 1.0
        if abs(cur_rels) > 1e-8:
            ratio = pred_rels / cur_rels
        else:
            ratio = 1.0 if pred_rels >= 0 else 0.0

        pressure = current[7] if len(current) > 7 else 0.0
        kappa = current[5] if len(current) > 5 else 1.0
        theta = current[3] if len(current) > 3 else 1.0

        if iso_report and not iso_report.get("ok", True):
            warnings.append("isodiametric kappa-fatness check failed")

        if flux_report and flux_report.get("bottlenecks"):
            warnings.append("pipeline bottleneck: " + ", ".join(flux_report["bottlenecks"]))

        if tower_report and tower_report.get("classification") == "double_exponential_like":
            warnings.append("change-point tower shows double-exponential-like blocks")

        action = "continue"
        reason = "predicted yield stable"

        if ratio < 0.35:
            action = "reduce_special_q"
            reason = "world model predicts sharp yield drop"
            confidence = min(0.9, max(0.1, 1.0 - ratio))
        elif ratio < 0.75:
            action = "increase_factor_base"
            reason = "predicted yield decline, moderate"
            confidence = min(0.8, max(0.1, 1.0 - ratio))
        elif pressure > 0.75:
            action = "slow_sieving_rebalance"
            reason = "pipeline pressure high"
            confidence = 0.6
        elif poly_score is not None and float(poly_score.get("murphy_e_proxy", 1.0)) < 0.05:
            action = "consider_new_polynomial"
            reason = "Murphy-E proxy low"
            confidence = 0.5
        elif kappa > 20.0:
            action = "reduce_special_q"
            reason = "lattice basis too thin"
            confidence = 0.6
        elif theta is not None and math.isfinite(theta) and theta < 0.8:
            action = "pause_for_human_review"
            reason = "heavy-tail yield suggests unstable sieving regime"
            confidence = 0.4
            warnings.append("Hill theta below 0.8")
        elif ratio > 1.5:
            action = "continue"
            reason = "predicted yield improvement"
            confidence = 0.6
        else:
            action = "continue"
            reason = "no strong degradation signal"
            confidence = 0.5

        # No-Zeno guard prevents action chattering
        if action != "continue":
            allowed, why = self.no_zeno.allow_event()
            if not allowed:
                warnings.append(f"no-zeno guard denied action: {why}")
                action = "hold"
                reason = "action suppressed by no-zeno dwell guard"
                confidence = min(confidence, 0.3)

        self.last_action = action

        return ControllerDecision(
            action=action,
            reason=reason,
            confidence=confidence,
            metrics={
                "current_log_rels": cur_rels,
                "predicted_log_rels": pred_rels,
                "pred_ratio": ratio,
                "pipeline_pressure": pressure,
                "kappa": kappa,
                "hill_theta": theta,
                "world_mode": world.mode,
                "n_records": len(records),
            },
            warnings=warnings,
        )


# =============================================================================
# 11. Full GNFS-10 research pipeline
# =============================================================================

class GNFS10Pipeline:
    """
    Full research pipeline:
      parse logs -> diagnostics -> features -> world model -> controller
    """

    def __init__(self):
        self.parser = LatticeSieveLogParser()
        self.stats = AssumptionLightStatistics()
        self.iso = IsodiametricLatticeFilter()
        self.no_zeno = NoZenoGuard()
        self.tower = ChangePointTowerAnalyzer()
        self.balancer = FractafoldPipelineBalancer()
        self.world = WorldModelWrapper(seq_len=4, hidden_dim=32, epochs=25)
        self.controller = AdaptiveWorldModelController(no_zeno=self.no_zeno)
        self.poly_scorer = MurphyEScorer(prime_bound=120)

    def compute_features(
        self,
        records: List[LogRecord],
        hill_theta: Optional[float],
        cusum_cps: List[int],
        iso_reports: List[Dict[str, Any]],
        flux_report: Dict[str, Any],
    ) -> List[List[float]]:
        rels = [max(float(r.rels), 0.0) for r in records]
        smooth = _ema(rels, alpha=0.2)

        pressures = []
        maxp = 0.0
        if flux_report and "pressure" in flux_report:
            maxp = max([abs(float(x)) for x in flux_report["pressure"].values()] + [0.0])

        cpset = set(cusum_cps)
        features: List[List[float]] = []

        for i, rec in enumerate(records):
            q = float(rec.q or rec.special_q or 0.0)
            fat = iso_reports[i] if i < len(iso_reports) else {"kappa": 1.0, "delta_min": 0.0}
            kappa = float(fat.get("kappa", 1.0)) if math.isfinite(float(fat.get("kappa", 1.0))) else 1.0
            delta_min = float(fat.get("delta_min", 0.0)) if math.isfinite(float(fat.get("delta_min", 0.0))) else 0.0

            feat = [
                _safe_log1p(q),
                _safe_log1p(rels[i]),
                _safe_log1p(smooth[i]),
                float(hill_theta) if hill_theta is not None and math.isfinite(hill_theta) else 1.0,
                1.0 if i in cpset else 0.0,
                kappa,
                delta_min,
                maxp,
            ]
            features.append(feat)

        return features

    def run_text(self, text: str, poly: Optional[PolynomialSpec] = None) -> Dict[str, Any]:
        records = self.parser.parse_text(text)
        return self.run_records(records, poly=poly)

    def run_file(self, path: str, poly: Optional[PolynomialSpec] = None) -> Dict[str, Any]:
        records = self.parser.parse_file(path)
        return self.run_records(records, poly=poly)

    def run_records(self, records: List[LogRecord],
                    poly: Optional[PolynomialSpec] = None) -> Dict[str, Any]:

        report: Dict[str, Any] = {
            "version": VERSION,
            "attribution": ATTRIBUTION,
            "disclaimer": "RESEARCH SCAFFOLD — NOT A FACTORIZATION TOOL",
            "n_records": len(records),
        }

        if not records:
            report["status"] = "no_records"
            report["decision"] = ControllerDecision(
                action="pause_for_human_review",
                reason="no parseable records",
                confidence=0.0,
            ).to_dict()
            return report

        rels = [max(float(r.rels), 0.0) for r in records]

        hill = self.stats.hill_tail_index(rels)
        ci = self.stats.subsample_mean_ci(rels)
        cusum = self.stats.cusum_change_points(rels)
        tower_report = self.tower.analyze(cusum["change_points"], len(records))
        flux = self.balancer.flux_from_records(records)
        flux_report = self.balancer.check_flux(flux)

        iso_reports = [self.iso.check_record(r) for r in records]

        features = self.compute_features(
            records=records,
            hill_theta=hill.get("theta"),
            cusum_cps=cusum["change_points"],
            iso_reports=iso_reports,
            flux_report=flux_report,
        )

        world_fit = self.world.fit(features)

        poly_score = None
        if poly is not None:
            poly_score = self.poly_scorer.score_polynomial(poly)

        decision = self.controller.decide(
            records=records,
            features=features,
            world=self.world,
            poly_score=poly_score,
            iso_report=iso_reports[-1] if iso_reports else None,
            tower_report=tower_report,
            flux_report=flux_report,
        )

        report.update({
            "status": "ok",
            "hill_tail": hill,
            "mean_ci": ci,
            "cusum": cusum,
            "tower": tower_report,
            "flux": flux,
            "flux_report": flux_report,
            "iso_last": iso_reports[-1] if iso_reports else {},
            "no_zeno": self.no_zeno.summary(),
            "world_model": world_fit,
            "poly_score": poly_score,
            "decision": decision.to_dict(),
        })

        return report


# =============================================================================
# 12. Synthetic log generator for self-test
# =============================================================================

def make_synthetic_log(n: int = 80, seed: int = 1234) -> str:
    """
    Generate synthetic lattice-sieve log with a yield drop after n//2.
    """
    import random
    rng = random.Random(seed)
    lines = []
    q = 100000
    for i in range(n):
        q += rng.randint(500, 2500)
        if i < n // 2:
            rels = rng.uniform(900, 1500)
        else:
            rels = rng.uniform(80, 350)
        t = rng.uniform(0.5, 3.0)
        skew = rng.uniform(0.8, 2.2)
        fb = rng.randint(5000, 20000)
        b1x = q
        b1y = 1
        b2x = rng.randint(100, 5000)
        b2y = q
        lines.append(
            f"sieving q={q} rels={rels:.3f} time={t:.3f} skew={skew:.3f} fb={fb}"
        )
        if i % 5 == 0:
            lines.append(f"lattice_basis q={q} b1={b1x},{b1y} b2={b2x},{b2y}")
    return "\n".join(lines)


# =============================================================================
# 13. Self-test
# =============================================================================

def self_test() -> int:
    print(BANNER)
    print("Running self-test...")
    ok = True

    def rep(name: str, passed: bool, extra: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {name} {extra}")

    # 1. Parser
    text = make_synthetic_log(80)
    parser = LatticeSieveLogParser()
    records = parser.parse_text(text)
    rep("parser produced records", len(records) > 30, f"n={len(records)}")
    rep("records have q", all(r.q is not None or r.special_q is not None for r in records[:10]))
    rep("records have rels", any(r.rels > 0 for r in records[:10]))

    # 2. Murphy-E scorer
    poly = PolynomialSpec(name="test_poly", f_coeffs=[1, 0, 0, 0, 1], g_coeffs=[1, -2])
    scorer = MurphyEScorer(prime_bound=80)
    ps = scorer.score_polynomial(poly)
    rep("murphy proxy finite", math.isfinite(ps["murphy_e_proxy"]), f"{ps['murphy_e_proxy']:.6f}")

    # 3. Stats
    stats = AssumptionLightStatistics()
    rels = [max(float(r.rels), 1e-6) for r in records]
    hill = stats.hill_tail_index(rels)
    rep("hill estimator returns theta", hill.get("theta") is None or math.isfinite(hill["theta"]),
        f"theta={hill.get('theta')}")
    ci = stats.subsample_mean_ci(rels)
    rep("subsample CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"],
        f"[{ci['lo']:.3f}, {ci['mean']:.3f}, {ci['hi']:.3f}]")
    cusum = stats.cusum_change_points(rels)
    rep("cusum returns list", isinstance(cusum["change_points"], list),
        f"n_cp={len(cusum['change_points'])}")

    # 4. Delta_Min filter
    iso = IsodiametricLatticeFilter()
    good_basis = LatticeBasis2D(b1=(100.0, 0.0), b2=(0.0, 100.0))
    thin_basis = LatticeBasis2D(b1=(1000.0, 0.0), b2=(0.0, 0.01))
    good = iso.basis_fatness(good_basis)
    thin = iso.basis_fatness(thin_basis)
    rep("fat basis accepted", good["ok"] is True, f"kappa={good['kappa']:.3f}")
    rep("thin basis rejected", thin["ok"] is False, f"kappa={thin['kappa']:.3f}")
    rep("delta_min positive", iso.delta_min(2.0) > 0)

    # 5. No-Zeno guard
    guard = NoZenoGuard(min_dwell_s=0.02, max_events=3, window_s=1.0)
    a1, _ = guard.allow_event(0.0)
    a2, _ = guard.allow_event(0.001)
    a3, _ = guard.allow_event(0.1)
    a4, _ = guard.allow_event(0.11)
    a5, _ = guard.allow_event(0.12)
    rep("no-zeno allows first", a1)
    rep("no-zeno denies dwell violation", not a2)
    rep("no-zeno allows later", a3)
    rep("no-zeno denies burst eventually", not (a4 and a5), f"a4={a4}, a5={a5}")

    # 6. Tower analyzer
    tower = ChangePointTowerAnalyzer()
    tr = tower.analyze(cusum["change_points"], len(rels))
    rep("tower returns status", "status" in tr, tr.get("status", ""))

    # 7. Fractafold balancer
    bal = FractafoldPipelineBalancer()
    flux = bal.flux_from_records(records)
    fx = bal.check_flux(flux)
    rep("flux report has pressure", "pressure" in fx)
    rep("flux report has bottlenecks", "bottlenecks" in fx)

    # 8. World model + controller
    pipeline = GNFS10Pipeline()
    report = pipeline.run_text(text, poly=poly)
    rep("pipeline status ok", report.get("status") == "ok")
    rep("pipeline has decision", "decision" in report)
    rep("decision action is string", isinstance(report["decision"]["action"], str),
        report["decision"]["action"])
    rep("decision has reason", len(report["decision"]["reason"]) > 0)
    rep("world model mode reported", "world_model" in report)

    # 9. No overclaim guard
    rep("report includes disclaimer", "RESEARCH SCAFFOLD" in report["disclaimer"])
    rep("no factorization promised", "NOT A FACTORIZATION TOOL" in report["disclaimer"])

    print("\n" + "=" * 78)
    print("SELF-TEST RESULT:", "ALL PASS" if ok else "SOME FAIL")
    print("=" * 78)

    return 0 if ok else 1


# =============================================================================
# 14. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="GNFS-10 research control layer")
    ap.add_argument("--selftest", action="store_true", help="Run self-test")
    ap.add_argument("--log", type=str, default=None, help="Path to lattice-sieve log")
    ap.add_argument("--text", type=str, default=None, help="Literal log text")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic log")
    ap.add_argument("--poly-f", type=str, default=None, help="Comma-separated f coeffs")
    ap.add_argument("--poly-g", type=str, default=None, help="Comma-separated g coeffs")
    ap.add_argument("--out", type=str, default=None, help="Write JSON report")
    args = ap.parse_args()

    if args.selftest:
        return self_test()

    print(BANNER)

    poly = None
    if args.poly_f or args.poly_g:
        f_coeffs = [float(x) for x in args.poly_f.split(",")] if args.poly_f else [1.0, 0.0, 1.0]
        g_coeffs = [float(x) for x in args.poly_g.split(",")] if args.poly_g else [1.0, 0.0]
        poly = PolynomialSpec(name="cli_poly", f_coeffs=f_coeffs, g_coeffs=g_coeffs)

    pipeline = GNFS10Pipeline()

    if args.synthetic:
        text = make_synthetic_log()
        report = pipeline.run_text(text, poly=poly)
    elif args.log:
        report = pipeline.run_file(args.log, poly=poly)
    elif args.text:
        report = pipeline.run_text(args.text, poly=poly)
    else:
        ap.print_help()
        print("\nNo input provided. Try --selftest or --synthetic.")
        return 0

    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main()
)
