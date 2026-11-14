#!/usr/bin/env python3
"""
=============================================================================
CADO-NFS TELEMETRY INGESTION + PARAMETER ADVISOR
Measures real speedup instead of estimating it.
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

THE CENTRAL DISTINCTION
-----------------------
OFFLINE  : analyse one existing log, fit yield = f(params, q), predict what
           a different setting would give. This is a PREDICTION from
           observational data. Parameter settings in a single run are
           usually correlated with the special-q range, so the estimate is
           CONFOUNDED. Reported with a confounding diagnostic, never as a
           measurement.

ONLINE   : run CADO twice on matched special-q ranges with two settings,
           ingest both logs, compute a paired bootstrap speedup. This is a
           MEASUREMENT. It is the only number in this file that may be
           quoted as "we measured X% faster".

WHAT IS AND IS NOT CLAIMED
  ✅ Parses CADO-NFS / msieve / GGNFS style logs, reports parse coverage.
  ✅ Fits a yield model with cross-validation, reports held-out error.
  ✅ Designs a paired A/B experiment and emits CADO parameter overrides.
  ✅ Computes paired-bootstrap speedup with confidence intervals.
  ✅ Projects sieving-only gain to total gain via Amdahl, using YOUR
     measured stage fractions.
  ❌ Does not factor anything.
  ❌ Does not claim a speedup until an A/B run has been ingested.
  ❌ Cannot extrapolate reliably to parameter settings never observed.
  ❌ relations/sec is not the whole objective: higher lpb yields more
     relations but a larger matrix. Sieving-only gain is capped by
     Amdahl at 1/(1 - f_sieve).

RELIABLE INGESTION PATH
  Log regexes are best-effort and version-dependent. The RELIABLE path is
  a CSV you export yourself (see --schema). Use it when possible.
=============================================================================
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("cado_advisor")

VERSION = "1.0.0"

BANNER = """
=============================================================================
CADO-NFS TELEMETRY + PARAMETER ADVISOR
OFFLINE numbers are PREDICTIONS (confounded).
ONLINE A/B numbers are MEASUREMENTS.
Every table below is labelled with which one it is.
=============================================================================
"""

CSV_SCHEMA = """
CSV SCHEMA (the reliable ingestion path)
----------------------------------------
Required columns:
  q_lo        lower bound of the special-q range for this workunit
  q_hi        upper bound
  rels        relations produced by this workunit
  seconds     wall-clock seconds for this workunit

Optional but strongly recommended:
  unique_rels relations surviving duplicate removal (if known)
  arm         A/B arm label, e.g. "baseline" / "candidate"
  host        machine identifier (to control for hardware)

Parameter columns (any subset; names follow CADO tasks.sieve.*):
  lpb0 lpb1 mfb0 mfb1 lim0 lim1 I lambda0 lambda1 nq qside

Example:
  q_lo,q_hi,rels,seconds,lpb0,lpb1,mfb0,mfb1,lim0,lim1,I,arm
  1000000,1010000,4123,52.1,29,29,58,58,4000000,4000000,13,baseline
  1000000,1010000,4890,55.9,30,30,60,60,4000000,4000000,13,candidate

Notes:
  * One row per workunit, not per relation.
  * For A/B, both arms must cover the SAME q ranges (that is what makes
    the comparison paired rather than confounded).
  * If you cannot export unique_rels, the advisor optimises raw relations
    and you must check matrix size separately.
"""

PARAM_KEYS = ("lpb0", "lpb1", "mfb0", "mfb1", "lim0", "lim1",
              "I", "lambda0", "lambda1", "nq", "qside")


# =============================================================================
# §1. WORKUNIT
# =============================================================================

@dataclass
class Workunit:
    source: str
    line_no: int
    q_lo: Optional[float] = None
    q_hi: Optional[float] = None
    rels: Optional[float] = None
    seconds: Optional[float] = None
    unique_rels: Optional[float] = None
    arm: str = ""
    host: str = ""
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def q_mid(self) -> Optional[float]:
        if self.q_lo is None:
            return None
        hi = self.q_hi if self.q_hi is not None else self.q_lo
        return 0.5 * (self.q_lo + hi)

    @property
    def usable(self) -> bool:
        return (self.rels is not None and self.seconds is not None
                and self.seconds > 0 and self.q_mid is not None)

    @property
    def yield_rate(self) -> float:
        if not self.usable:
            return float("nan")
        return float(self.rels) / float(self.seconds)

    @property
    def unique_rate(self) -> Optional[float]:
        if self.unique_rels is None or not self.usable:
            return None
        return float(self.unique_rels) / float(self.seconds)

    @property
    def config_key(self) -> str:
        if not self.params:
            return "default"
        return ",".join(f"{k}={self.params[k]:g}"
                        for k in sorted(self.params))


# =============================================================================
# §2. PARSERS
# =============================================================================

_KV = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)
_RANGE = re.compile(r"[\[\(]\s*(\d+)\s*[,\-]\s*(\d+)\s*[\]\)]")
_TS = re.compile(r"^\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?\s*")

_ALIAS: Dict[str, Tuple[str, ...]] = {
    "q_lo": ("q_lo", "qmin", "q0", "specialq_min", "sq_min"),
    "q_hi": ("q_hi", "qmax", "q1", "specialq_max", "sq_max"),
    "rels": ("rels", "nrels", "relations", "relations_found", "rel_count",
             "found_rels", "produced"),
    "seconds": ("seconds", "sec", "s", "time", "cpu", "cpu_time",
                "elapsed", "wall", "wall_s", "elapsed_s", "t"),
    "unique_rels": ("unique_rels", "unique", "nunique", "rels_unique"),
}
_ALIAS_REV = {a: canon for canon, alist in _ALIAS.items() for a in alist}

# Best-effort natural-language patterns seen in CADO / msieve style output.
_NL_PATTERNS = [
    # "... 4123 relations produced in 52.1s ..."
    re.compile(r"(?P<rels>\d+)\s+relations?\s+(?:produced\s+)?in\s+"
               r"(?P<sec>[0-9.]+)\s*s", re.I),
    # "... found 4123 relations ... 52.1 s ..."
    re.compile(r"found\s+(?P<rels>\d+)\s+relations?.*?"
               r"(?P<sec>[0-9.]+)\s*s(?:ec)?\b", re.I),
    # "special-q 1000003 ... rels 4123 ... 52.1s"
    re.compile(r"special[-_ ]?q\s*[:=]?\s*(?P<q>\d+)", re.I),
]


def _stage_of(line: str) -> str:
    low = line.lower()
    if any(k in low for k in ("polyselect", "murphy", "polynomial")):
        return "polyselect"
    if any(k in low for k in ("las", "sieve", "sieving", "lattice", "special-q",
                              "specialq")):
        return "sieve"
    if any(k in low for k in ("dup", "purge", "filter", "merge", "replay")):
        return "filter"
    if any(k in low for k in ("linalg", "bwc", "krylov", "wiedemann",
                              "lanczos", "matrix")):
        return "linalg"
    if "sqrt" in low:
        return "sqrt"
    return "unknown"


class LogParser:
    """Best-effort parser. Always reports coverage so you know what it missed."""

    def __init__(self) -> None:
        self.n_lines = 0
        self.n_kept = 0
        self.n_with_params = 0

    def parse_line(self, line: str, source: str, line_no: int) -> Optional[Workunit]:
        s = line.strip()
        if not s or s.startswith("#!"):
            return None
        w = Workunit(source=source, line_no=line_no)

        # 1. key=value pairs (most reliable)
        for key, val in _KV.findall(s):
            kl = key.lower()
            try:
                v = float(val)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            if kl in _ALIAS_REV:
                setattr(w, _ALIAS_REV[kl], v)
            elif kl in PARAM_KEYS:
                w.params[kl] = v
            elif kl in ("arm", "host"):
                pass  # strings handled below
            elif kl == "q":
                if w.q_lo is None:
                    w.q_lo = v

        # 2. string fields
        for key in ("arm", "host"):
            m = re.search(rf"\b{key}\s*=\s*([A-Za-z0-9_.\-]+)", s, re.I)
            if m:
                setattr(w, key, m.group(1))

        # 3. bracketed range [q_lo,q_hi)
        if w.q_lo is None or w.q_hi is None:
            m = _RANGE.search(s)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                if b > a >= 0:
                    w.q_lo, w.q_hi = a, b

        # 4. natural-language fallbacks
        for pat in _NL_PATTERNS:
            m = pat.search(s)
            if not m:
                continue
            gd = m.groupdict()
            if "rels" in gd and w.rels is None:
                w.rels = float(gd["rels"])
            if "sec" in gd and w.seconds is None:
                w.seconds = float(gd["sec"])
            if "q" in gd and w.q_lo is None:
                w.q_lo = float(gd["q"])

        if not w.usable:
            return None
        if w.q_hi is None:
            w.q_hi = w.q_lo
        return w

    def parse_text(self, text: str, source: str = "text") -> List[Workunit]:
        out: List[Workunit] = []
        for i, line in enumerate(text.splitlines(), 1):
            self.n_lines += 1
            w = self.parse_line(line, source, i)
            if w is not None:
                self.n_kept += 1
                if w.params:
                    self.n_with_params += 1
                out.append(w)
        return out

    def parse_file(self, path: str | Path) -> List[Workunit]:
        p = Path(path)
        if not p.exists():
            log.warning(f"missing file: {p}")
            return []
        return self.parse_text(p.read_text(encoding="utf-8", errors="ignore"),
                               source=p.name)

    def coverage(self) -> Dict[str, Any]:
        return {
            "lines_seen": self.n_lines,
            "workunits_parsed": self.n_kept,
            "workunits_with_params": self.n_with_params,
            "coverage_frac": (self.n_kept / self.n_lines
                              if self.n_lines else 0.0),
            "note": ("Low coverage is normal for verbose logs. If "
                     "workunits_with_params is 0, the advisor cannot "
                     "compare configurations — export a CSV instead."),
        }


def load_csv(path: str | Path) -> List[Workunit]:
    """The reliable ingestion path."""
    p = Path(path)
    out: List[Workunit] = []
    with open(p, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        for req in ("q_lo", "rels", "seconds"):
            if req not in cols:
                raise ValueError(f"CSV missing required column '{req}'. "
                                 f"See --schema.")
        for i, row in enumerate(rd, 1):
            def num(k: str) -> Optional[float]:
                c = cols.get(k)
                if c is None or row[c] in (None, ""):
                    return None
                try:
                    v = float(row[c])
                    return v if math.isfinite(v) else None
                except ValueError:
                    return None

            w = Workunit(source=p.name, line_no=i,
                         q_lo=num("q_lo"), q_hi=num("q_hi"),
                         rels=num("rels"), seconds=num("seconds"),
                         unique_rels=num("unique_rels"),
                         arm=(row.get(cols.get("arm", ""), "") or ""),
                         host=(row.get(cols.get("host", ""), "") or ""))
            for k in PARAM_KEYS:
                v = num(k)
                if v is not None:
                    w.params[k] = v
            if w.q_hi is None:
                w.q_hi = w.q_lo
            if w.usable:
                out.append(w)
    return out


# =============================================================================
# §3. FEATURES + YIELD MODEL
# =============================================================================

def feature_vector(w: Workunit, param_names: List[str]) -> np.ndarray:
    q = w.q_mid or 1.0
    base = [math.log(max(q, 2.0)),
            math.log(max((w.q_hi or q) - (w.q_lo or q), 1.0))]
    vals = [w.params.get(k, 0.0) for k in param_names]
    return np.array(base + vals, dtype=np.float64)


class RidgeYieldModel:
    """
    Ridge regression on log(rels/sec). Deliberately simple: with a few
    hundred workunits a transformer overfits. Cross-validated.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.ymu = 0.0
        self.ysd = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeYieldModel":
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd
        Z = np.hstack([np.ones((len(Z), 1)), Z])
        self.ymu = float(y.mean())
        self.ysd = float(y.std()) or 1.0
        yn = (y - self.ymu) / self.ysd
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        A[0, 0] -= self.alpha           # do not penalise intercept
        self.beta = np.linalg.solve(A, Z.T @ yn)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("model not fitted")
        Z = (X - self.mu) / self.sd
        Z = np.hstack([np.ones((len(Z), 1)), Z])
        return Z @ self.beta * self.ysd + self.ymu


def kfold_report(X: np.ndarray, y: np.ndarray, k: int = 5,
                 alpha: float = 1.0, seed: int = 0) -> Dict[str, Any]:
    n = len(y)
    if n < 10:
        return {"status": "insufficient_data", "n": n}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    pred = np.zeros(n)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        m = RidgeYieldModel(alpha).fit(X[tr], y[tr])
        pred[f] = m.predict(X[f])
    mae = float(np.mean(np.abs(pred - y)))
    base = float(np.mean(np.abs(y.mean() - y)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {
        "status": "ok", "n": n, "k": k,
        "mae_log_yield": mae,
        "mae_baseline_mean": base,
        "improvement_vs_mean_pct": 100.0 * (base - mae) / max(base, 1e-9),
        "r2": 1.0 - ss_res / ss_tot,
        "note": ("Target is log(rels/sec). r2 <= 0 means the model is no "
                 "better than predicting the mean — do not trust the "
                 "advisor's recommendations in that case."),
    }


# =============================================================================
# §4. OFFLINE COUNTERFACTUAL (PREDICTION, NOT MEASUREMENT)
# =============================================================================

def confounding_report(units: List[Workunit],
                       param_names: List[str]) -> Dict[str, Any]:
    """
    In a single run, parameters are usually tied to the q schedule.
    If so, an offline 'what if' is confounded and must not be quoted
    as a measured speedup.
    """
    qs = np.array([math.log(max(u.q_mid or 2.0, 2.0)) for u in units])
    out: Dict[str, Any] = {"correlations": {}, "confounded_params": []}
    for k in param_names:
        v = np.array([u.params.get(k, np.nan) for u in units])
        ok = np.isfinite(v)
        if ok.sum() < 8 or np.std(v[ok]) < 1e-9:
            continue
        c = float(np.corrcoef(qs[ok], v[ok])[0, 1])
        out["correlations"][k] = c
        if abs(c) > 0.5:
            out["confounded_params"].append(k)
    out["n_distinct_configs"] = len({u.config_key for u in units})
    out["verdict"] = (
        "SEVERELY CONFOUNDED — offline gain estimates are unreliable; "
        "run an A/B experiment"
        if out["confounded_params"] else
        "mild confounding — offline estimates still only predictions"
    )
    return out


def offline_counterfactual(units: List[Workunit],
                           param_names: List[str],
                           alpha: float = 1.0) -> Dict[str, Any]:
    usable = [u for u in units if u.usable]
    if len(usable) < 20:
        return {"status": "insufficient_data", "n": len(usable)}

    X = np.stack([feature_vector(u, param_names) for u in usable])
    y = np.array([math.log(max(u.yield_rate, 1e-9)) for u in usable])
    cv = kfold_report(X, y, alpha=alpha)
    if cv.get("status") != "ok" or cv["r2"] <= 0:
        return {"status": "model_not_predictive", "cv": cv}

    model = RidgeYieldModel(alpha).fit(X, y)
    configs = sorted({u.config_key for u in usable})
    reps: Dict[str, Workunit] = {}
    for u in usable:
        reps.setdefault(u.config_key, u)

    # Predict every observed config at every observed q, then compare
    # config means at matched q. This removes the q confound *within the
    # model*, but cannot remove unmodelled confounding.
    per_cfg: Dict[str, List[float]] = {c: [] for c in configs}
    for u in usable:
        for c in configs:
            v = Workunit(source="", line_no=0, q_lo=u.q_lo, q_hi=u.q_hi,
                         rels=1.0, seconds=1.0,
                         params=dict(reps[c].params))
            xv = feature_vector(v, param_names)[None, :]
            per_cfg[c].append(float(model.predict(xv)[0]))
    means = {c: float(np.mean(v)) for c, v in per_cfg.items()}
    best = max(means, key=means.get)
    logged = float(np.mean(y))
    gain = math.exp(means[best] - logged) - 1.0

    return {
        "status": "ok",
        "kind": "PREDICTION (offline, observational)",
        "cv": cv,
        "best_config": best,
        "predicted_gain_pct": 100.0 * gain,
        "config_mean_log_yield": means,
        "confounding": confounding_report(usable, param_names),
        "warning": ("This is NOT a measured speedup. It is a model's "
                    "counterfactual on observational data. Confirm with "
                    "--design-ab and then --ab."),
    }


# =============================================================================
# §5. A/B DESIGN
# =============================================================================

def design_ab(units: List[Workunit], param_names: List[str],
              n_buckets: int = 8, candidate: Optional[Dict[str, float]] = None
              ) -> Dict[str, Any]:
    """
    Pick matched q buckets and two configs. Emits CADO parameter overrides.
    """
    usable = [u for u in units if u.usable and u.q_mid is not None]
    if len(usable) < 8:
        return {"status": "insufficient_data"}
    usable.sort(key=lambda u: u.q_mid)
    buckets = np.array_split(np.array(usable, dtype=object), n_buckets)
    spans = [(float(b[0].q_lo), float(b[-1].q_hi or b[-1].q_lo))
             for b in buckets if len(b)]

    counts: Dict[str, int] = {}
    for u in usable:
        counts[u.config_key] = counts.get(u.config_key, 0) + 1
    baseline_key = max(counts, key=counts.get)
    baseline = next(u.params for u in usable if u.config_key == baseline_key)

    if candidate is None:
        # propose a modest, standard step: raise lpb by 1, mfb by 2
        candidate = dict(baseline)
        for k in ("lpb0", "lpb1"):
            if k in candidate:
                candidate[k] = candidate[k] + 1
        for k in ("mfb0", "mfb1"):
            if k in candidate:
                candidate[k] = candidate[k] + 2

    def overrides(p: Dict[str, float]) -> str:
        return " ".join(f"tasks.sieve.{k}={p[k]:g}"
                        for k in sorted(p) if k in PARAM_KEYS)

    runs = []
    for i, (lo, hi) in enumerate(spans):
        for arm, p in (("baseline", baseline), ("candidate", candidate)):
            runs.append({
                "bucket": i, "arm": arm,
                "qmin": int(lo), "qmax": int(hi),
                "overrides": overrides(p),
            })

    return {
        "status": "ok",
        "n_buckets": len(spans),
        "baseline_config": baseline_key,
        "candidate_params": candidate,
        "runs": runs,
        "instructions": [
            "1. Verify parameter names against YOUR CADO version's "
            "parameters/factor/params.c* file. Names differ across versions.",
            "2. Run each (bucket, arm) on the SAME host, same thread count, "
            "same binary. Randomise the order to avoid thermal/cache drift.",
            "3. Record q_lo,q_hi,rels,seconds,arm plus the parameter columns "
            "into one CSV (see --schema).",
            "4. Run: python cado_advisor.py --ab results.csv",
            "5. Only the number from step 4 may be quoted as measured.",
        ],
        "caveat": ("Higher lpb produces more relations but a larger matrix. "
                   "Also record filtering output and matrix dimension so "
                   "the total-cost projection is meaningful."),
    }


# =============================================================================
# §6. A/B ANALYSIS — PAIRED BOOTSTRAP (MEASUREMENT)
# =============================================================================

def _bucket_key(u: Workunit, width: float) -> int:
    return int(math.floor((u.q_mid or 0.0) / width))


def ab_analysis(units: List[Workunit], n_boot: int = 10000,
                seed: int = 0, metric: str = "raw") -> Dict[str, Any]:
    """
    Paired-by-q-bucket log-ratio bootstrap. This is a MEASUREMENT.
    """
    usable = [u for u in units if u.usable and u.arm]
    arms = sorted({u.arm for u in usable})
    if len(arms) != 2:
        return {"status": "need_exactly_two_arms", "arms": arms}

    if metric == "unique":
        if any(u.unique_rate is None for u in usable):
            return {"status": "unique_rels_missing",
                    "note": "Re-export CSV with unique_rels to use this metric."}
        rate = lambda u: u.unique_rate
    else:
        rate = lambda u: u.yield_rate

    qs = [u.q_mid for u in usable]
    span = max(qs) - min(qs)
    width = max(span / 20.0, 1.0)

    a, b = arms
    grouped: Dict[int, Dict[str, List[float]]] = {}
    for u in usable:
        k = _bucket_key(u, width)
        grouped.setdefault(k, {}).setdefault(u.arm, []).append(
            math.log(max(rate(u), 1e-9)))

    pairs: List[Tuple[int, float]] = []
    for k, d in grouped.items():
        if a in d and b in d:
            pairs.append((k, float(np.mean(d[b]) - np.mean(d[a]))))
    if len(pairs) < 3:
        return {"status": "too_few_matched_buckets",
                "n_matched": len(pairs),
                "note": "Both arms must cover the same special-q ranges."}

    diffs = np.array([d for _, d in pairs])
    point = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boots = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    speedup = math.exp(point)
    ci = (math.exp(lo), math.exp(hi))
    hosts = {u.host for u in usable if u.host}

    return {
        "status": "ok",
        "kind": "MEASUREMENT (paired A/B, bootstrap CI)",
        "baseline_arm": a, "candidate_arm": b,
        "metric": metric,
        "n_matched_buckets": len(pairs),
        "speedup_x": speedup,
        "speedup_ci95": [float(ci[0]), float(ci[1])],
        "gain_pct": 100.0 * (speedup - 1.0),
        "gain_ci95_pct": [100.0 * (ci[0] - 1.0), 100.0 * (ci[1] - 1.0)],
        "significant": bool(ci[0] > 1.0 or ci[1] < 1.0),
        "n_hosts": len(hosts) or 1,
        "warnings": ([ "Multiple hosts detected — hardware variation may "
                       "inflate the interval." ] if len(hosts) > 1 else [])
                    + ([ "Fewer than 8 matched buckets — interval is wide "
                         "and fragile." ] if len(pairs) < 8 else []),
    }


# =============================================================================
# §7. AMDAHL PROJECTION
# =============================================================================

def amdahl_projection(sieving_speedup: float,
                      f_sieve: float = 0.70) -> Dict[str, Any]:
    f = min(max(f_sieve, 0.0), 1.0)
    total = 1.0 / ((1.0 - f) + f / max(sieving_speedup, 1e-9))
    return {
        "f_sieve": f,
        "sieving_speedup_x": sieving_speedup,
        "projected_total_speedup_x": total,
        "projected_total_gain_pct": 100.0 * (total - 1.0),
        "ceiling_total_speedup_x": (1.0 / (1.0 - f)) if f < 1 else float("inf"),
        "note": ("f_sieve should be MEASURED from your own run's stage "
                 "timings, not assumed. The default 0.70 is a literature-"
                 "order placeholder. Larger lpb also enlarges the matrix, "
                 "which this projection does NOT account for."),
    }


# =============================================================================
# §8. ADVISOR + ABLATION
# =============================================================================

def advisor_ablation(units: List[Workunit], param_names: List[str],
                     alpha: float = 1.0, seed: int = 0) -> Dict[str, Any]:
    """
    Compare prediction strategies on held-out workunits. This tests
    whether the model adds anything over trivial heuristics.
    """
    usable = [u for u in units if u.usable]
    if len(usable) < 20:
        return {"status": "insufficient_data", "n": len(usable)}
    X = np.stack([feature_vector(u, param_names) for u in usable])
    y = np.array([math.log(max(u.yield_rate, 1e-9)) for u in usable])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(usable))
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    counts: Dict[str, List[float]] = {}
    for i in tr:
        counts.setdefault(usable[i].config_key, []).append(y[i])
    cfg_mean = {c: float(np.mean(v)) for c, v in counts.items()}
    most_freq = max(counts, key=lambda c: len(counts[c]))
    best_mean = max(cfg_mean, key=cfg_mean.get)

    model = RidgeYieldModel(alpha).fit(X[tr], y[tr])
    pred_model = model.predict(X[te])
    yte = y[te]

    strategies = {
        "global_mean": np.full(len(te), float(y[tr].mean())),
        "most_frequent_config": np.full(len(te), cfg_mean[most_freq]),
        "best_historical_config": np.full(len(te), cfg_mean[best_mean]),
        "ridge_advisor": pred_model,
    }
    out = {}
    for name, p in strategies.items():
        mae = float(np.mean(np.abs(p - yte)))
        out[name] = {"mae_log_yield": mae}
    base = out["best_historical_config"]["mae_log_yield"]
    adv = out["ridge_advisor"]["mae_log_yield"]
    return {
        "status": "ok",
        "n_test": len(te),
        "strategies": out,
        "advisor_vs_best_heuristic_pct": 100.0 * (base - adv) / max(base, 1e-9),
        "verdict": ("ADVISOR ADDS VALUE" if adv < 0.95 * base
                    else "NO ADDED VALUE — use the simple heuristic"),
        "note": ("This measures prediction accuracy, not counterfactual "
                 "correctness. Only an A/B run validates a recommendation."),
    }


# =============================================================================
# §9. REPORT
# =============================================================================

def print_report(rep: Dict[str, Any]) -> None:
    print(BANNER)
    if "coverage" in rep:
        c = rep["coverage"]
        print("PARSE COVERAGE")
        print(f"  lines seen           : {c['lines_seen']}")
        print(f"  workunits parsed     : {c['workunits_parsed']}")
        print(f"  with parameters      : {c['workunits_with_params']}")
        print(f"  {c['note']}\n")

    if "offline" in rep and rep["offline"].get("status") == "ok":
        o = rep["offline"]
        print("OFFLINE COUNTERFACTUAL  —  *** PREDICTION, NOT MEASUREMENT ***")
        print(f"  best config          : {o['best_config']}")
        print(f"  predicted gain       : {o['predicted_gain_pct']:+.2f}%")
        print(f"  model r2 (held out)  : {o['cv']['r2']:.3f}")
        print(f"  confounding verdict  : {o['confounding']['verdict']}")
        if o["confounding"]["correlations"]:
            print("  corr(log q, param):")
            for k, v in o["confounding"]["correlations"].items():
                print(f"    {k:<10} {v:+.3f}")
        print(f"  ⚠️  {o['warning']}\n")

    if "ablation" in rep and rep["ablation"].get("status") == "ok":
        a = rep["ablation"]
        print("ADVISOR ABLATION (prediction accuracy on held-out workunits)")
        print(f"  {'strategy':<26} {'MAE(log yield)':>16}")
        for k, v in a["strategies"].items():
            print(f"  {k:<26} {v['mae_log_yield']:>16.4f}")
        print(f"  verdict: {a['verdict']}\n")

    if "ab" in rep and rep["ab"].get("status") == "ok":
        ab = rep["ab"]
        print("A/B RESULT  —  *** MEASUREMENT ***")
        print(f"  arms                 : {ab['baseline_arm']} → {ab['candidate_arm']}")
        print(f"  metric               : {ab['metric']}")
        print(f"  matched q buckets    : {ab['n_matched_buckets']}")
        print(f"  speedup              : {ab['speedup_x']:.4f}x")
        print(f"  95% CI               : [{ab['speedup_ci95'][0]:.4f}, "
              f"{ab['speedup_ci95'][1]:.4f}]")
        print(f"  gain                 : {ab['gain_pct']:+.2f}% "
              f"[{ab['gain_ci95_pct'][0]:+.2f}%, {ab['gain_ci95_pct'][1]:+.2f}%]")
        print(f"  significant          : {ab['significant']}")
        for w in ab["warnings"]:
            print(f"  ⚠️  {w}")
        if "amdahl" in rep:
            am = rep["amdahl"]
            print(f"\n  AMDAHL PROJECTION (f_sieve={am['f_sieve']:.2f})")
            print(f"    projected total    : {am['projected_total_speedup_x']:.4f}x "
                  f"({am['projected_total_gain_pct']:+.2f}%)")
            print(f"    ceiling if sieve→∞ : {am['ceiling_total_speedup_x']:.2f}x")
            print(f"    {am['note']}")
        print()

    if "design" in rep and rep["design"].get("status") == "ok":
        d = rep["design"]
        print("A/B EXPERIMENT DESIGN")
        print(f"  buckets              : {d['n_buckets']}")
        print(f"  baseline             : {d['baseline_config']}")
        print(f"  candidate            : {d['candidate_params']}")
        print(f"  runs to execute      : {len(d['runs'])}")
        print("\n  First 4 runs:")
        for r in d["runs"][:4]:
            print(f"    bucket={r['bucket']} arm={r['arm']:<10} "
                  f"q=[{r['qmin']},{r['qmax']})")
            print(f"      {r['overrides']}")
        print("\n  INSTRUCTIONS")
        for s in d["instructions"]:
            print(f"    {s}")
        print(f"\n  ⚠️  {d['caveat']}\n")

    print("=" * 77)
    print("Only the A/B section may be quoted as a measured speedup.")
    print("=" * 77)


# =============================================================================
# §10. DEMO DATA + SELF-TEST
# =============================================================================

def make_demo_csv(path: str | Path, n: int = 240, seed: int = 0,
                  ab: bool = False) -> str:
    rng = np.random.default_rng(seed)
    rows = []
    q = 1_000_000.0
    for i in range(n):
        q += rng.uniform(4000, 9000)
        base_lpb = 29 if (i % 2 == 0 or not ab) else 30
        arm = "baseline" if base_lpb == 29 else "candidate"
        # ground truth: lpb30 gives ~15% more relations, ~6% more time
        rel_mult = 1.15 if base_lpb == 30 else 1.0
        t_mult = 1.06 if base_lpb == 30 else 1.0
        decay = math.exp(-0.10 * (q - 1_000_000) / 1_000_000)
        rels = 4000 * decay * rel_mult * rng.lognormal(0, 0.12)
        secs = 50.0 * t_mult * rng.lognormal(0, 0.08)
        rows.append({
            "q_lo": int(q), "q_hi": int(q + 8000),
            "rels": round(rels, 2), "seconds": round(secs, 3),
            "unique_rels": round(rels * 0.82, 2),
            "lpb0": base_lpb, "lpb1": base_lpb,
            "mfb0": 2 * base_lpb, "mfb1": 2 * base_lpb,
            "lim0": 4_000_000, "lim1": 4_000_000, "I": 13,
            "arm": arm if ab else "", "host": "node01",
        })
    p = Path(path)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return str(p)


def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    import tempfile
    tmp = Path(tempfile.mkdtemp())

    # T1 CSV round trip
    p = make_demo_csv(tmp / "obs.csv", n=120)
    units = load_csv(p)
    rep("T1 CSV ingestion", len(units) == 120, f"n={len(units)}")
    rep("T1 params parsed", all(u.params for u in units))

    # T2 log parser
    txt = "\n".join([
        "Info:las: [1000000,1010000) 4123 relations produced in 52.1s lpb0=29 I=13",
        "special-q=1020000 rels=3900 seconds=49.0 lpb0=30",
        "noise line without any telemetry",
    ])
    lp = LogParser()
    got = lp.parse_text(txt, "test")
    rep("T2 log parser extracts workunits", len(got) == 2, f"n={len(got)}")
    rep("T2 coverage reported", lp.coverage()["lines_seen"] == 3)

    # T3 confounding detection (obs.csv has lpb tied to parity, q increasing)
    names = sorted({k for u in units for k in u.params})
    conf = confounding_report(units, names)
    rep("T3 confounding computed", "verdict" in conf, conf["verdict"][:40])

    # T4 offline model
    off = offline_counterfactual(units, names)
    rep("T4 offline runs", off.get("status") in ("ok", "model_not_predictive"),
        off.get("status"))
    if off.get("status") == "ok":
        rep("T4 labelled PREDICTION", "PREDICTION" in off["kind"])

    # T5 ablation
    abl = advisor_ablation(units, names)
    rep("T5 ablation runs", abl.get("status") == "ok", abl.get("verdict", ""))

    # T6 A/B design
    des = design_ab(units, names, n_buckets=6)
    rep("T6 design produced runs", des.get("status") == "ok"
        and len(des["runs"]) == 12, f"runs={len(des.get('runs', []))}")

    # T7 A/B measurement recovers injected ~9% gain (1.15/1.06 = 1.085)
    pa = make_demo_csv(tmp / "ab.csv", n=400, seed=1, ab=True)
    au = load_csv(pa)
    ab = ab_analysis(au, n_boot=3000, seed=0)
    rep("T7 A/B is MEASUREMENT", ab.get("kind", "").startswith("MEASUREMENT"))
    if ab.get("status") == "ok":
        g = ab["gain_pct"]
        rep("T7 recovers injected gain (~8.5%)", 3.0 < g < 15.0, f"{g:+.2f}%")
        rep("T7 CI brackets truth",
            ab["gain_ci95_pct"][0] < 8.5 < ab["gain_ci95_pct"][1],
            f"[{ab['gain_ci95_pct'][0]:+.2f}, {ab['gain_ci95_pct'][1]:+.2f}]")

    # T8 unique metric
    abu = ab_analysis(au, n_boot=1500, metric="unique")
    rep("T8 unique_rels metric works", abu.get("status") == "ok")

    # T9 Amdahl
    am = amdahl_projection(1.085, f_sieve=0.70)
    rep("T9 Amdahl total < sieving gain",
        am["projected_total_speedup_x"] < 1.085,
        f"{am['projected_total_speedup_x']:.4f}x")
    rep("T9 ceiling ~3.33x", abs(am["ceiling_total_speedup_x"] - 3.333) < 0.01)

    # T10 unmatched arms rejected
    for u in au:
        if u.arm == "candidate":
            u.q_lo += 1e9
            u.q_hi += 1e9
    bad = ab_analysis(au, n_boot=200)
    rep("T10 unmatched q rejected", bad.get("status") != "ok", bad.get("status"))

    print("\n" + "=" * 77)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 77)
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CADO-NFS telemetry + advisor")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--csv", type=str, help="workunit CSV (reliable path)")
    ap.add_argument("--log", type=str, help="raw CADO/msieve log (best effort)")
    ap.add_argument("--ab", type=str, help="A/B results CSV -> MEASUREMENT")
    ap.add_argument("--design-ab", action="store_true")
    ap.add_argument("--metric", choices=["raw", "unique"], default="raw")
    ap.add_argument("--f-sieve", type=float, default=0.70)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args(argv)

    if selftest_needs := a.selftest:
        return 0 if self_test() else 1
    if a.schema:
        print(CSV_SCHEMA)
        return 0

    rep: Dict[str, Any] = {"version": VERSION}
    units: List[Workunit] = []

    if a.demo:
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        p = make_demo_csv(tmp / "obs.csv")
        units = load_csv(p)
        pa = make_demo_csv(tmp / "ab.csv", n=400, seed=1, ab=True)
        rep["ab"] = ab_analysis(load_csv(pa), n_boot=a.nboot, metric=a.metric)
        rep["amdahl"] = amdahl_projection(
            rep["ab"].get("speedup_x", 1.0), a.f_sieve)
    elif a.csv:
        units = load_csv(a.csv)
    elif a.log:
        lp = LogParser()
        units = lp.parse_file(a.log)
        rep["coverage"] = lp.coverage()
    elif a.ab:
        rep["ab"] = ab_analysis(load_csv(a.ab), n_boot=a.nboot, metric=a.metric)
        rep["amdahl"] = amdahl_projection(
            rep["ab"].get("speedup_x", 1.0), a.f_sieve)
    else:
        ap.print_help()
        print("\nStart with: --schema, --selftest, or --demo")
        return 0

    if units:
        names = sorted({k for u in units for k in u.params})
        if names:
            rep["offline"] = offline_counterfactual(units, names, a.alpha)
            rep["ablation"] = advisor_ablation(units, names, a.alpha)
            if a.design_ab:
                rep["design"] = design_ab(units, names)
        else:
            rep["offline"] = {"status": "no_parameters_in_data",
                              "note": "Export a CSV with parameter columns."}

    if a.ab and "ab" not in rep:
        rep["ab"] = ab_analysis(load_csv(a.ab), n_boot=a.nboot, metric=a.metric)
        rep["amdahl"] = amdahl_projection(
            rep["ab"].get("speedup_x", 1.0), a.f_sieve)

    print_report(rep)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2, default=float),
                                encoding="utf-8")
        print(f"\nJSON -> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
