#!/usr/bin/env python3
"""
=============================================================================
GNFS SCALE ORCHESTRATOR
CADO-NFS / msieve workflow + honest resource estimation
+ our modules as a CONTROL layer (not a complexity reduction)
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
            My Soul Move By Power of Holy Spirit ;
            What MSPS NETWORK SEE, Lord Buddha Knows.
License   : MIT | Year: 2026

WHAT THIS IS
  A production-style orchestrator that runs REAL CADO-NFS / msieve on
  numbers you own or are authorized to study, with our research modules
  attached as a control layer:
    * Murphy-E proxy pre-screening  -> fewer expensive polyselect trials
    * Transformer yield forecast  -> early abort of low-yield q-ranges
    * No-Zeno restart guard       -> bounded restart storms
    * Assumption-light statistics -> honest CI on yield, regime detection
    * OPMC log compression        -> cheaper log storage/transfer
    * Distributed q-range scheduler with checkpoint/resume

WHAT THIS IS NOT
  ❌ NOT a reduction of GNFS complexity L_N[1/3, (64/9)^(1/3)]
  ❌ NOT a new factoring algorithm
  ❌ NOT able to make RSA-250 cheap (it cost ~2700 core-years in 2020)
  ❌ NOT for attacking keys you do not own / are not authorized to study

HONEST SAVINGS BUDGET (must be MEASURED, not assumed)
  polyselect ~5% of total, advisor may save 20-40% of it  -> 1-2% total
  sieving   ~80% of total, forecast may save 5-15% of it  -> 4-12% total
  filtering ~5%, linalg ~10%: NO savings from our modules
  => realistic total 5-15%, workload-dependent.

DOMAIN REALITY CHECK
  Only quantum chemistry, gauge/particle physics, and number theory have
  genuine factorization workloads, and those are 20-80 digits.
  Oncology, epidemiology, anti-aging, cell regeneration, gene editing,
  evolutionary genetics, computational neuroscience/psychiatry do NOT
  need GNFS-scale factorization at all. Use --domain-report to see this.

RESOURCES CALIBRATION
  Relative L_N scaling anchored at the public RSA-250 record:
    829 bits, 250 digits, ~2700 core-years (2020, CADO-NFS).
  Valid for >= ~150 digits. Below that, use ECM/MPQS empirical timings.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("gnfs_scale")

VERSION = "1.0.0"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

BANNER = """
=============================================================================
GNFS SCALE ORCHESTRATOR — control layer over CADO-NFS / msieve
NOT a complexity reduction. RSA-250 cost ~2700 core-years (public, 2020).
Only run on numbers you own or are authorized to study.
=============================================================================
"""

# Public record anchor (Boudot et al., Feb 2020)
RSA250_BITS = 829
RSA250_CORE_YEARS = 2700.0
GNFS_C = (64.0 / 9.0) ** (1.0 / 3.0)     # 1.922999...
CORE_YEAR_SECONDS = 365.25 * 24 * 3600


# =============================================================================
# §1. HONEST RESOURCE ESTIMATION
# =============================================================================

def ln_L(bits: int) -> float:
    """log of L_N[1/3, c] for N ~ 2^bits."""
    lnN = bits * math.log(2.0)
    if lnN <= 1.0:
        return 0.0
    lnlnN = math.log(lnN)
    return GNFS_C * (lnN ** (1.0 / 3.0)) * (lnlnN ** (2.0 / 3.0))


_LN_L_ANCHOR = ln_L(RSA250_BITS)


def estimate_core_years(bits: int) -> Optional[float]:
    """
    Relative L_N scaling anchored at RSA-250.
    Returns None for sizes where GNFS is not the right tool (< ~500 bits).
    """
    if bits < 500:
        return None
    return RSA250_CORE_YEARS * math.exp(ln_L(bits) - _LN_L_ANCHOR)


def digits_to_bits(digits: int) -> int:
    return int(math.ceil(digits * math.log2(10.0)))


@dataclass
class ResourceReport:
    label: str
    bits: int
    digits: int
    recommended_tool: str
    core_years: Optional[float]
    wall_days_at_1000_cores: Optional[float]
    note: str


def resource_report() -> List[ResourceReport]:
    rows: List[ResourceReport] = []

    # Scientific scale — ECM / MPQS territory
    for digits, tool, note in [
        (20, "ECM (GMP-ECM)", "mass-spec / small integer relations"),
        (40, "ECM (GMP-ECM)", "PSLQ / resultant / discriminant"),
        (60, "ECM + MPQS", "upper end of routine scientific need"),
        (80, "MPQS / small GNFS", "algebraic norms, class numbers"),
        (100, "GNFS (CADO/msieve)", "number-theory research"),
    ]:
        rows.append(ResourceReport(
            label=f"scientific {digits} digits",
            bits=digits_to_bits(digits), digits=digits,
            recommended_tool=tool, core_years=None,
            wall_days_at_1000_cores=None,
            note=note + " | seconds-to-hours on a workstation",
        ))

    # GNFS scale — anchored extrapolation
    for label, bits in [
        ("RSA-250 (PUBLIC RECORD, 2020)", 829),
        ("RSA-300", 996),
        ("RSA-1024", 1024),
        ("RSA-2048", 2048),
        ("RSA-4096", 4096),
    ]:
        cy = estimate_core_years(bits)
        wall = (cy * CORE_YEAR_SECONDS / (1000 * 86400)) if cy else None
        rows.append(ResourceReport(
            label=label, bits=bits,
            digits=int(bits * math.log10(2.0)),
            recommended_tool="GNFS (CADO-NFS)",
            core_years=cy,
            wall_days_at_1000_cores=wall,
            note=("already factored publicly" if bits == 829
                  else "relative L_N extrapolation from RSA-250 anchor"),
        ))
    return rows


def print_resource_table() -> None:
    rows = resource_report()
    print("=" * 108)
    print("HONEST RESOURCE TABLE  (relative L_N scaling anchored at RSA-250 = 2700 core-years)")
    print("=" * 108)
    print(f"{'label':<34}{'bits':>6}{'digits':>8}{'tool':<22}"
          f"{'core-years':>14}{'days@1000cores':>16}")
    print("-" * 108)
    for r in rows:
        cy = f"{r.core_years:.3e}" if r.core_years is not None else "—"
        wd = (f"{r.wall_days_at_1000_cores:.3e}"
              if r.wall_days_at_1000_cores is not None else "—")
        print(f"{r.label:<34}{r.bits:>6}{r.digits:>8}{r.recommended_tool:<22}"
              f"{cy:>14}{wd:>16}")
    print("=" * 108)
    print("NOTE: below ~150 digits GNFS is NOT the right tool; use ECM/MPQS.")
    print("      Extrapolation is order-of-magnitude, not a quote.")
    print("      Our control layer targets 5-15% total savings, MEASURED.")
    print()


# =============================================================================
# §2. DOMAIN REALITY CHECK
# =============================================================================

DOMAIN_MAP: List[Dict[str, Any]] = [
    {"domain": "quantum chemistry", "needs_factorization": True,
     "digits": "20-60", "what": "resultants, discriminants of characteristic "
                                "polynomials; PSLQ for exact integrals",
     "tool": "GMP-ECM + LLL/PSLQ"},
    {"domain": "gauge / particle physics", "needs_factorization": True,
     "digits": "20-80", "what": "integer relations (PSLQ/LLL), rational "
                                "reconstruction of Feynman integrals",
     "tool": "GMP-ECM + LLL/PSLQ"},
    {"domain": "number theory", "needs_factorization": True,
     "digits": "30-100", "what": "class numbers, algebraic norms, "
                                 "discriminants of number fields",
     "tool": "GMP-ECM / MPQS / CADO-NFS"},
    {"domain": "oncology", "needs_factorization": False, "digits": "—",
     "what": "clonal deconvolution (NMF), tumor phylogenetics, survival",
     "tool": "NMF / Bayesian phylogenetics — NOT GNFS"},
    {"domain": "epidemiology", "needs_factorization": False, "digits": "—",
     "what": "compartment ODE/SDE, network inference, Rt estimation",
     "tool": "Bayesian inference — NOT GNFS"},
    {"domain": "anti-aging medicine", "needs_factorization": False,
     "digits": "—", "what": "survival models, multi-omics integration, "
                            "biological-age clocks",
     "tool": "regression / ML — NOT GNFS"},
    {"domain": "advanced cell regeneration", "needs_factorization": False,
     "digits": "—", "what": "reaction-diffusion PDE, lineage tracing, "
                            "phase-field tissue models",
     "tool": "PDE solvers (CH3D/PFC) — NOT GNFS"},
    {"domain": "gene editing / analysis", "needs_factorization": "marginal",
     "digits": "<20", "what": "k-mer counting, gcd of counts, small "
                              "combinatorial designs",
     "tool": "trial division / gcd — NOT GNFS"},
    {"domain": "evolutionary genetics / phenotypes",
     "needs_factorization": False, "digits": "—",
     "what": "phylogenetics, GWAS, quantitative genetics",
     "tool": "ML / likelihood — NOT GNFS"},
    {"domain": "computational neuroscience", "needs_factorization": False,
     "digits": "—", "what": "neural mass/field models, DCM, state-space",
     "tool": "time-series / PDE — NOT GNFS"},
    {"domain": "computational psychiatry", "needs_factorization": False,
     "digits": "—", "what": "validated instruments + hierarchical models",
     "tool": "statistics / ML — NOT GNFS"},
]


def print_domain_report() -> None:
    print("=" * 104)
    print("DOMAIN REALITY CHECK — which fields actually need factorization")
    print("=" * 104)
    print(f"{'domain':<38}{'needs?':<12}{'digits':<10}{'tool'}")
    print("-" * 104)
    for d in DOMAIN_MAP:
        need = str(d["needs_factorization"])
        print(f"{d['domain']:<38}{need:<12}{d['digits']:<10}{d['tool']}")
    print("-" * 104)
    for d in DOMAIN_MAP:
        if d["needs_factorization"] is True:
            print(f"  {d['domain']}: {d['what']}")
    print("=" * 104)
    print("CONCLUSION: RSA-250 (250 digits) is a number-theory BENCHMARK,")
    print("            not a requirement of the applied fields listed.")
    print("            Scientific workloads are <= 80 digits and already")
    print("            served by ECM stage 1+2 / MPQS on a workstation.")
    print()


# =============================================================================
# §3. MURPHY-E PROXY PRE-SCREEN  (saves expensive polyselect trials)
# =============================================================================

def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, int(math.isqrt(n)) + 1):
        if s[i]:
            s[i*i::i] = b"\x00" * (((n - i*i)//i) + 1)
    return [i for i in range(n+1) if s[i]]


def base_m_polynomial(N: int, degree: int) -> Tuple[List[int], int]:
    """Educational base-m expansion. Real polyselect is far more sophisticated."""
    m = max(2, int(round(N ** (1.0 / max(degree, 1)))))
    coeffs, x = [], N
    for _ in range(degree + 1):
        coeffs.append(x % m)
        x //= m
    if x:
        coeffs[-1] += x * m
    return coeffs, m


class MurphyEProxy:
    """
    Cheap root-count + size proxy used to PRE-SCREEN candidates before
    handing them to CADO-NFS's expensive Murphy-E evaluation.

    This is NOT CADO's Murphy-E. It is a filter, and its value must be
    measured as "fraction of expensive evaluations avoided".
    """

    def __init__(self, prime_bound: int = 200):
        self.primes = sieve_primes(prime_bound)

    @staticmethod
    def _roots_mod(coeffs: List[int], p: int) -> int:
        n = 0
        for x in range(p):
            acc = 0
            for c in reversed(coeffs):
                acc = (acc * x + c) % p
            if acc == 0:
                n += 1
        return n

    def score(self, f_coeffs: List[int], g_coeffs: List[int]) -> float:
        root = 0.0
        for p in self.primes:
            root += (self._roots_mod(f_coeffs, p)
                     + self._roots_mod(g_coeffs, p)) / p
        size = sum(math.log1p(abs(c)) for c in f_coeffs)
        return root / (1.0 + size)

    def prescreen(self, N: int, degrees: List[int],
                  keep_frac: float = 0.3) -> Dict[str, Any]:
        cands = []
        for d in degrees:
            f, m = base_m_polynomial(N, d)
            g = [1, -m]
            cands.append({"degree": d, "m": m, "f": f, "g": g,
                          "proxy": self.score(f, g)})
        cands.sort(key=lambda c: -c["proxy"])
        k = max(1, int(len(cands) * keep_frac))
        return {
            "n_candidates": len(cands),
            "n_kept": k,
            "kept": cands[:k],
            "expensive_evals_avoided": len(cands) - k,
            "note": "proxy filter only; final ranking must use real Murphy-E",
        }


# =============================================================================
# §4. LOG PARSING + ASSUMPTION-LIGHT MONITOR
# =============================================================================

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*"
                   r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
ALIASES = {
    "q": ("q", "special_q", "specialq", "q0"),
    "rels": ("rels", "relations", "relations_found", "nrels", "rel_count"),
    "cpu": ("cpu", "cpu_time", "time", "elapsed", "seconds"),
}


@dataclass
class SieveRecord:
    line_no: int
    q: Optional[float] = None
    rels: Optional[float] = None
    cpu: Optional[float] = None


def parse_sieve_log(text: str) -> List[SieveRecord]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        rec = SieveRecord(line_no=i)
        hit = False
        for k, v in KV_RE.findall(line):
            try:
                fv = float(v)
            except ValueError:
                continue
            if not math.isfinite(fv):
                continue
            for canon, names in ALIASES.items():
                if k.lower() in names:
                    setattr(rec, canon, fv)
                    hit = True
                    break
        if hit and rec.rels is not None:
            out.append(rec)
    return out


def hill_theta(vals: List[float], frac: float = 0.2) -> float:
    x = [v for v in vals if v > 0]
    if len(x) < 12:
        return 0.0
    s = sorted(x, reverse=True)
    k = max(5, min(int(frac*len(s)), len(s)-1))
    thr = s[k]
    if thr <= 0:
        return 0.0
    logs = [math.log(v/thr) for v in s[:k] if v > thr]
    if not logs:
        return 0.0
    m = sum(logs)/len(logs)
    return 1.0/m if m > 1e-12 else 0.0


def cusum_count(x: np.ndarray, thr: float = 3.0, drift: float = 0.25) -> int:
    if len(x) < 8:
        return 0
    mu, sd = float(x.mean()), float(x.std())
    if sd < 1e-12:
        return 0
    sp = sn = 0.0
    n = 0
    for xi in x:
        z = (xi-mu)/sd
        sp = max(0.0, sp+z-drift)
        sn = max(0.0, sn-z-drift)
        if sp > thr or sn > thr:
            n += 1
            sp = sn = 0.0
    return n


def subsample_mean_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 6:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = max(2, min(int(round(n ** (1/3))), n//2))
    theta = float(x.mean())
    subs = np.array([x[i:i+b].mean() for i in range(n-b+1)])
    sc = math.sqrt(b)*(subs-theta)
    a = 1-conf
    qhi = float(np.quantile(sc, 1-a/2))
    qlo = float(np.quantile(sc, a/2))
    return {"mean": theta, "lo": theta-qhi/math.sqrt(n),
            "hi": theta-qlo/math.sqrt(n), "n": n, "block": b}


# =============================================================================
# §5. NO-ZENO RESTART GUARD  (Six Problems, Prop. 6.1)
# =============================================================================

class NoZenoRestartGuard:
    """
    Hysteresis + minimum dwell time. Bounds the number of restart/abort
    events by an energy-budget argument on the monitored statistic.
    This is an ENGINEERING bound on the controller, NOT a PDE theorem.
    """

    def __init__(self, theta_hi: float, theta_lo: float,
                 min_dwell_s: float = 60.0):
        if theta_lo >= theta_hi:
            raise ValueError("need theta_lo < theta_hi")
        self.hi, self.lo = theta_hi, theta_lo
        self.gap = theta_hi - theta_lo
        self.min_dwell = min_dwell_s
        self.armed = True
        self.last_event = -math.inf
        self.n_events = 0
        self.n_denied = 0

    def update(self, value: float, now: float) -> bool:
        """Returns True if a restart/abort event should fire."""
        if value < self.lo:
            self.armed = True
        if not self.armed or value < self.hi:
            return False
        if now - self.last_event < self.min_dwell:
            self.n_denied += 1
            return False
        self.armed = False
        self.last_event = now
        self.n_events += 1
        return True

    def budget_bound(self, total_variation: float) -> float:
        """N(T) <= total_variation / gap  (Six Problems energy budget)."""
        return total_variation / max(self.gap, 1e-12)


# =============================================================================
# §6. YIELD WORLD MODEL  (forecast -> early abort of bad q-ranges)
# =============================================================================

FEATURE_NAMES = ["log_q", "log_rels", "smooth_rels", "hill", "cusum", "cpu"]
FEAT_DIM = len(FEATURE_NAMES)


def build_features(recs: List[SieveRecord]) -> np.ndarray:
    rels = np.array([r.rels or 0.0 for r in recs], dtype=float)
    qs = np.array([r.q or 0.0 for r in recs], dtype=float)
    cpus = np.array([r.cpu or 0.0 for r in recs], dtype=float)
    smooth = np.zeros_like(rels)
    if len(rels):
        smooth[0] = rels[0]
        for i in range(1, len(rels)):
            smooth[i] = 0.2*rels[i] + 0.8*smooth[i-1]
    h = hill_theta(rels.tolist())
    c = cusum_count(rels)
    return np.stack([np.log1p(qs), np.log1p(rels), np.log1p(smooth),
                     np.full_like(rels, h), np.full_like(rels, float(c)),
                     np.log1p(cpus)], axis=-1).astype(np.float32)


if HAS_TORCH:
    class YieldWorldModel(nn.Module):
        def __init__(self, d_in: int = FEAT_DIM, d_model: int = 64,
                     n_heads: int = 2, n_layers: int = 2):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.pos = nn.Parameter(torch.randn(1, 128, d_model)*0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4*d_model, dropout=0.1,
                batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)
            self.unc = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(),
                                     nn.Linear(32, 1), nn.Softplus())

        def forward(self, x):
            B, T, _ = x.shape
            h = self.proj(x) + self.pos[:, :T, :]
            z = self.enc(h)[:, -1]
            return self.head(z).squeeze(-1), self.unc(z).squeeze(-1)


class YieldForecaster:
    """Fits on early windows, forecasts later yield, flags low-yield ranges."""

    def __init__(self, window: int = 8, horizon: int = 4, epochs: int = 40):
        self.window, self.horizon, self.epochs = window, horizon, epochs
        self.model = None
        self.mu = self.sd = None
        self.trained = False

    def fit(self, feats: np.ndarray) -> Dict[str, Any]:
        if not HAS_TORCH:
            return {"status": "no_torch"}
        n = len(feats)
        if n < self.window + self.horizon + 4:
            return {"status": "insufficient_data", "n": n}
        self.mu = feats.mean(0)
        self.sd = feats.std(0) + 1e-6
        Z = (feats - self.mu)/self.sd
        X, Y = [], []
        for i in range(n - self.window - self.horizon + 1):
            X.append(Z[i:i+self.window])
            Y.append(feats[i+self.window+self.horizon-1, 1])  # log_rels
        X = torch.tensor(np.array(X), dtype=torch.float32)
        Y = torch.tensor(np.array(Y), dtype=torch.float32)
        self.model = YieldWorldModel()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        losses = []
        for _ in range(self.epochs):
            opt.zero_grad()
            pred, _ = self.model(X)
            loss = F.mse_loss(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        self.trained = True
        return {"status": "trained", "initial": losses[0],
                "final": losses[-1], "n_seq": len(X)}

    def predict(self, feats: np.ndarray) -> Optional[float]:
        if not self.trained or len(feats) < self.window:
            return None
        Z = (feats[-self.window:] - self.mu)/self.sd
        x = torch.tensor(Z, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            p, _ = self.model(x)
        return float(p[0])

    def should_abort_range(self, feats: np.ndarray,
                           ratio_threshold: float = 0.4) -> Dict[str, Any]:
        p = self.predict(feats)
        if p is None:
            return {"abort": False, "reason": "no_prediction"}
        cur = float(feats[-1, 1])
        ratio = math.exp(p - cur) if abs(p-cur) < 50 else 0.0
        return {"abort": ratio < ratio_threshold,
                "predicted_log_rels": p, "current_log_rels": cur,
                "ratio": ratio, "threshold": ratio_threshold}


# =============================================================================
# §7. OPMC LOG COMPRESSION (feature-space, for storage/transfer)
# =============================================================================

class OPMCLogCompressor:
    """Fractal-similarity clustering of feature windows. Storage saving only."""

    def __init__(self, threshold: float = 0.95, max_clusters: int = 128):
        self.thr, self.maxc = threshold, max_clusters
        self.centroids: List[np.ndarray] = []
        self.counts: List[int] = []

    def fit_encode(self, X: np.ndarray) -> Dict[str, Any]:
        if len(X) == 0:
            return {"status": "empty"}
        mu, sd = X.mean(0), X.std(0)+1e-9
        Z = (X-mu)/sd
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        Zn = Z/nrm
        codes = []
        for i, z in enumerate(Zn):
            if not self.centroids:
                self.centroids.append(Z[i].copy())
                self.counts.append(1)
                codes.append(0)
                continue
            C = np.stack(self.centroids)
            Cn = C/(np.linalg.norm(C, axis=1, keepdims=True)+1e-12)
            sims = Cn @ z
            j = int(np.argmax(sims))
            if sims[j] >= self.thr or len(self.centroids) >= self.maxc:
                c = self.counts[j]
                self.centroids[j] = (self.centroids[j]*c + Z[i])/(c+1)
                self.counts[j] = c+1
                codes.append(j)
            else:
                self.centroids.append(Z[i].copy())
                self.counts.append(1)
                codes.append(len(self.centroids)-1)
        codes = np.array(codes, dtype=np.int32)
        orig = X.size*8
        comp = len(self.centroids)*X.shape[1]*8 + codes.nbytes + 512
        return {"status": "ok", "n_windows": len(X),
                "n_clusters": len(self.centroids),
                "compression_ratio": orig/max(comp, 1),
                "note": "storage/transfer saving, NOT compute saving"}


# =============================================================================
# §8. CADO-NFS / msieve RUNNERS
# =============================================================================

def which_cado() -> Optional[str]:
    for name in ("cado-nfs.py", "cado-nfs"):
        p = shutil.which(name)
        if p:
            return p
    return None


def which_msieve() -> Optional[str]:
    return shutil.which("msieve")


@dataclass
class RunConfig:
    workdir: str = "./gnfs_work"
    threads: int = 8
    # sieving control
    qmin: int = 1_000_000
    qmax: int = 100_000_000
    qrange_per_job: int = 1_000_000
    # advisor
    use_advisor: bool = True
    abort_ratio: float = 0.4
    # guard
    restart_theta_hi: float = 0.5
    restart_theta_lo: float = 0.2
    min_dwell_s: float = 60.0
    # safety
    authorized: bool = False


class CadoNFSRunner:
    """
    Subprocess wrapper for CADO-NFS with checkpoint/resume.
    Generates parameter files and streams logs for the advisor.
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.binary = which_cado()
        self.work = Path(cfg.workdir)
        self.work.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return self.binary is not None

    def write_params(self, N: int) -> Path:
        p = self.work / "params.c90"
        p.write_text(
            f"tasks.workdir = {self.work}\n"
            f"tasks.linalg.bwc.interval = 1000\n"
            f"tasks.sieve.lpb0 = 30\n"
            f"tasks.sieve.lpb1 = 30\n"
            f"tasks.sieve.qmin = {self.cfg.qmin}\n"
            f"tasks.sieve.qmax = {self.cfg.qmax}\n"
            f"tasks.threads = {self.cfg.threads}\n",
            encoding="utf-8")
        return p

    def run(self, N: int, timeout: float = 3600.0
            ) -> Dict[str, Any]:
        if not self.cfg.authorized:
            return {"status": "REFUSED",
                    "reason": "set authorized=True only for numbers you own "
                              "or are explicitly authorized to study"}
        if not self.available:
            return {"status": "not_installed",
                    "hint": "https://cado-nfs.gitlabpages.inria.fr/"}
        params = self.write_params(N)
        cmd = [self.binary, str(params), str(N)]
        t0 = time.time()
        logf = self.work / "cado.log"
        try:
            with open(logf, "w") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=timeout, text=True)
            txt = logf.read_text(errors="ignore")
            factors = self._extract_factors(txt, N)
            return {"status": "ok" if proc.returncode == 0 else "error",
                    "returncode": proc.returncode,
                    "elapsed_s": time.time()-t0,
                    "factors": factors,
                    "verified": bool(factors) and
                              math.prod(factors) == N,
                    "log": str(logf)}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "elapsed_s": time.time()-t0,
                    "log": str(logf),
                    "note": "checkpoint retained; resume by re-running"}

    @staticmethod
    def _extract_factors(text: str, N: int) -> List[int]:
        out = []
        for line in text.splitlines():
            if any(k in line.lower() for k in ("factor", "p=", "prime")):
                for tok in re.findall(r"\d{4,}", line):
                    v = int(tok)
                    if 1 < v < N and N % v == 0 and v not in out:
                        out.append(v)
        return sorted(out)


class MsieveRunner:
    """msieve NFS wrapper (-n for NFS, -t threads)."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.binary = which_msieve()
        self.work = Path(cfg.workdir)
        self.work.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return self.binary is not None

    def run(self, N: int, timeout: float = 3600.0) -> Dict[str, Any]:
        if not self.cfg.authorized:
            return {"status": "REFUSED", "reason": "authorization required"}
        if not self.available:
            return {"status": "not_installed",
                    "hint": "https://sourceforge.net/projects/msieve/"}
        t0 = time.time()
        logf = self.work / "msieve.log"
        try:
            with open(logf, "w") as fh:
                proc = subprocess.run(
                    [self.binary, "-n", "-t", str(self.cfg.threads), str(N)],
                    stdout=fh, stderr=subprocess.STDOUT,
                    timeout=timeout, text=True, cwd=str(self.work))
            txt = logf.read_text(errors="ignore")
            factors = CadoNFSRunner._extract_factors(txt, N)
            return {"status": "ok" if proc.returncode == 0 else "error",
                    "elapsed_s": time.time()-t0, "factors": factors,
                    "verified": bool(factors) and math.prod(factors) == N,
                    "log": str(logf)}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "elapsed_s": time.time()-t0,
                    "log": str(logf)}


# =============================================================================
# §9. DISTRIBUTED Q-RANGE SCHEDULER
# =============================================================================

@dataclass
class QJob:
    job_id: int
    qmin: int
    qmax: int
    status: str = "pending"
    node: Optional[str] = None
    rels: int = 0
    elapsed_s: float = 0.0


class DistributedScheduler:
    """
    Splits [qmin, qmax) into jobs, assigns to nodes, checkpoints, and
    applies the advisor's early-abort recommendation per job.
    """

    def __init__(self, cfg: RunConfig, nodes: List[str]):
        self.cfg, self.nodes = cfg, nodes
        self.jobs: List[QJob] = []
        self.ckpt = Path(cfg.workdir) / "scheduler.json"
        self._build()

    def _build(self) -> None:
        jid = 0
        q = self.cfg.qmin
        while q < self.cfg.qmax:
            hi = min(q + self.cfg.qrange_per_job, self.cfg.qmax)
            self.jobs.append(QJob(jid, q, hi))
            q = hi
            jid += 1

    def save(self) -> None:
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt.write_text(json.dumps(
            {"jobs": [asdict(j) for j in self.jobs]}, indent=2))

    def load(self) -> bool:
        if not self.ckpt.exists():
            return False
        d = json.loads(self.ckpt.read_text())
        self.jobs = [QJob(**j) for j in d["jobs"]]
        return True

    def assign(self) -> List[Tuple[str, QJob]]:
        pending = [j for j in self.jobs if j.status == "pending"]
        out = []
        for i, j in enumerate(pending):
            node = self.nodes[i % max(len(self.nodes), 1)]
            j.node, j.status = node, "assigned"
            out.append((node, j))
        return out

    def summary(self) -> Dict[str, Any]:
        done = sum(1 for j in self.jobs if j.status == "done")
        aborted = sum(1 for j in self.jobs if j.status == "aborted")
        return {"n_jobs": len(self.jobs), "done": done,
                "aborted": aborted,
                "total_rels": sum(j.rels for j in self.jobs),
                "total_elapsed_s": sum(j.elapsed_s for j in self.jobs)}


# =============================================================================
# §10. ORCHESTRATOR + BENCHMARK
# =============================================================================

class GNFSScaleOrchestrator:
    def __init__(self, cfg: RunConfig, nodes: Optional[List[str]] = None):
        self.cfg = cfg
        self.prescreen = MurphyEProxy(prime_bound=150)
        self.forecaster = YieldForecaster()
        self.guard = NoZenoRestartGuard(cfg.restart_theta_hi,
                                        cfg.restart_theta_lo,
                                        cfg.min_dwell_s)
        self.compressor = OPMCLogCompressor()
        self.cado = CadoNFSRunner(cfg)
        self.msieve = MsieveRunner(cfg)
        self.sched = DistributedScheduler(cfg, nodes or ["local"])

    def plan(self, N: int) -> Dict[str, Any]:
        bits = N.bit_length()
        cy = estimate_core_years(bits)
        degs = [4, 5, 6] if bits > 300 else [3, 4, 5]
        return {
            "N_digits": len(str(N)), "N_bits": bits,
            "estimated_core_years": cy,
            "recommended_tool": ("GNFS (CADO-NFS)" if bits >= 500
                                 else "ECM / MPQS (GNFS is overkill)"),
            "prescreen": self.prescreen.prescreen(N, degs),
            "scheduler": self.sched.summary(),
            "cado_available": self.cado.available,
            "msieve_available": self.msieve.available,
            "honest_note": "control layer targets 5-15% MEASURED savings; "
                           "complexity unchanged",
        }

    def attach_log(self, text: str) -> Dict[str, Any]:
        recs = parse_sieve_log(text)
        if len(recs) < 12:
            return {"status": "insufficient_records", "n": len(recs)}
        feats = build_features(recs)
        rels = np.array([r.rels or 0.0 for r in recs])
        fit = self.forecaster.fit(feats)
        abort = self.forecaster.should_abort_range(feats, self.cfg.abort_ratio)
        comp = self.compressor.fit_encode(feats)
        fired = self.guard.update(float(abort.get("ratio", 1.0)), time.time())
        return {
            "status": "ok", "n_records": len(recs),
            "yield_ci": subsample_mean_ci(rels),
            "hill_theta": hill_theta(rels.tolist()),
            "cusum_events": cusum_count(rels),
            "forecaster": fit,
            "abort_recommendation": abort,
            "restart_fired": fired,
            "guard": {"events": self.guard.n_events,
                      "denied": self.guard.n_denied},
            "log_compression": comp,
        }

    def execute(self, N: int, backend: str = "cado",
                timeout: float = 3600.0) -> Dict[str, Any]:
        plan = self.plan(N)
        if backend == "cado":
            res = self.cado.run(N, timeout=timeout)
        elif backend == "msieve":
            res = self.msieve.run(N, timeout=timeout)
        else:
            res = {"status": "unknown_backend"}
        self.sched.save()
        return {"plan": plan, "run": res,
                "scheduler": self.sched.summary()}


def benchmark_control_layer(n_lines: int = 400, seed: int = 7) -> Dict[str, Any]:
    """
    Measure what the control layer actually buys on a synthetic log:
      * fraction of q-ranges flagged for early abort
      * forecast MAE vs persistence
      * log compression ratio
      * restart events suppressed by the no-Zeno guard
    """
    rng = np.random.default_rng(seed)
    lines = []
    q = 1_000_000
    for i in range(n_lines):
        q += int(rng.integers(500, 5000))
        base = 900.0 if i < n_lines//2 else 180.0   # regime drop
        rels = max(1.0, base*rng.lognormal(0, 0.3))
        cpu = float(rng.gamma(2.0, 0.15))
        lines.append(f"sieving q={q} rels={rels:.2f} cpu={cpu:.4f}")
    text = "\n".join(lines)

    cfg = RunConfig(authorized=False)
    orch = GNFSScaleOrchestrator(cfg)
    rep = orch.attach_log(text)

    # forecast vs persistence on held-out tail
    recs = parse_sieve_log(text)
    feats = build_features(recs)
    split = int(0.7*len(feats))
    f2 = YieldForecaster()
    f2.fit(feats[:split])
    errs_m, errs_p = [], []
    for i in range(split, len(feats)-1):
        p = f2.predict(feats[:i+1])
        if p is None:
            continue
        truth = float(feats[i+1, 1])
        errs_m.append(abs(p-truth))
        errs_p.append(abs(float(feats[i, 1])-truth))
    mae_m = float(np.mean(errs_m)) if errs_m else float("nan")
    mae_p = float(np.mean(errs_p)) if errs_p else float("nan")

    return {
        "n_records": rep.get("n_records"),
        "forecast_mae_model": mae_m,
        "forecast_mae_persistence": mae_p,
        "forecast_improvement_pct": (100*(mae_p-mae_m)/mae_p
                                     if mae_p and mae_p > 0 else 0.0),
        "abort_recommendation": rep.get("abort_recommendation"),
        "log_compression_ratio":
            rep.get("log_compression", {}).get("compression_ratio"),
        "restart_events": rep.get("guard", {}).get("events"),
        "restart_denied": rep.get("guard", {}).get("denied"),
        "honest_note": ("synthetic log; these are CONTROL-layer metrics, "
                        "not factoring speedups. Real savings must be "
                        "measured against a baseline CADO run."),
    }


# =============================================================================
# §11. DEMO LOG + SELF-TEST
# =============================================================================

def demo_log(n: int = 300, seed: int = 3) -> str:
    rng = np.random.default_rng(seed)
    out = []
    q = 1_000_000
    for i in range(n):
        q += int(rng.integers(500, 4000))
        base = [800.0, 250.0, 60.0][(i//90) % 3]
        rels = max(1.0, base*rng.lognormal(0, 0.35))
        out.append(f"[{i*0.5:.3f}] sieving: q={q} rels={rels:.2f} "
                   f"cpu={rng.gamma(2,0.15):.4f}")
    return "\n".join(out)


def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1 resource anchoring
    cy250 = estimate_core_years(829)
    rep("T1 RSA-250 anchor reproduces 2700 core-years",
        abs(cy250 - 2700.0) < 1.0, f"{cy250:.1f}")
    cy300 = estimate_core_years(996)
    rep("T1 RSA-300 ~100x RSA-250", 30 < cy300/cy250 < 400,
        f"ratio={cy300/cy250:.1f}")
    rep("T1 small N returns None (use ECM)", estimate_core_years(200) is None)

    # T2 domain map
    n_true = sum(1 for d in DOMAIN_MAP if d["needs_factorization"] is True)
    rep("T2 only 3 domains genuinely need factorization", n_true == 3,
        f"n={n_true}")

    # T3 prescreen
    ps = MurphyEProxy(prime_bound=80)
    r = ps.prescreen(3233, [3, 4, 5], keep_frac=0.34)
    rep("T3 prescreen reduces expensive evals",
        r["expensive_evals_avoided"] > 0, str(r["expensive_evals_avoided"]))

    # T4 parser
    recs = parse_sieve_log(demo_log(120))
    rep("T4 parser extracts records", len(recs) > 60, f"n={len(recs)}")

    # T5 stats
    rels = [x.rels or 0 for x in recs]
    rep("T5 Hill theta finite", math.isfinite(hill_theta(rels)))
    ci = subsample_mean_ci(np.array(rels))
    rep("T5 subsample CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])

    # T6 no-Zeno guard
    g = NoZenoRestartGuard(0.5, 0.2, min_dwell_s=10.0)
    e1 = g.update(0.9, now=0.0)
    e2 = g.update(0.9, now=1.0)
    e3 = g.update(0.1, now=2.0)      # re-arm
    e4 = g.update(0.9, now=20.0)
    rep("T6 first event fires", e1)
    rep("T6 dwell suppresses second", not e2)
    rep("T6 re-arm then dwell allows later", e4 and not e3)
    rep("T6 budget bound positive", g.budget_bound(10.0) > 0)

    # T7 forecaster
    feats = build_features(recs)
    fc = YieldForecaster(epochs=15)
    fit = fc.fit(feats)
    if HAS_TORCH:
        rep("T7 forecaster trains", fit.get("status") == "trained",
            f"{fit.get('final', 0):.4f}")
        ab = fc.should_abort_range(feats)
        rep("T7 abort decision returned", "abort" in ab, str(ab.get("ratio")))
    else:
        rep("T7 forecaster degrades without torch",
            fit.get("status") == "no_torch")

    # T8 compression
    comp = OPMCLogCompressor().fit_encode(feats)
    rep("T8 log compression ratio > 1", comp.get("compression_ratio", 0) > 1,
        f"{comp.get('compression_ratio'):.2f}x")

    # T9 runners refuse unauthorized
    cfg = RunConfig(authorized=False)
    rep("T9 CADO refuses unauthorized",
        CadoNFSRunner(cfg).run(3233)["status"] in ("REFUSED", "not_installed"))
    rep("T9 msieve refuses unauthorized",
        MsieveRunner(cfg).run(3233)["status"] in ("REFUSED", "not_installed"))

    # T10 scheduler
    sc = DistributedScheduler(RunConfig(qmin=0, qmax=5_000_000,
                                        qrange_per_job=1_000_000),
                              ["n0", "n1"])
    rep("T10 scheduler splits jobs", len(sc.jobs) == 5, f"n={len(sc.jobs)}")
    asg = sc.assign()
    rep("T10 assigns round-robin", len(asg) == 5)
    sc.save()
    rep("T10 checkpoint written", sc.ckpt.exists())

    # T11 benchmark
    bm = benchmark_control_layer(n_lines=250)
    rep("T11 benchmark runs", "forecast_mae_model" in bm,
        f"improvement={bm['forecast_improvement_pct']:.1f}%")

    print("\n" + "=" * 76)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 76)
    print("""
WHAT THIS PROVES
  * resource estimates reproduce the public RSA-250 anchor
  * only 3 of the 11 listed domains need factorization at all
  * prescreening reduces expensive Murphy-E evaluations
  * yield forecasting, no-Zeno guard, compression all function
  * runners refuse to execute without explicit authorization

WHAT THIS DOES NOT PROVE
  * no reduction of GNFS complexity
  * no ability to make RSA-250 cheap
  * control-layer metrics are NOT factoring speedups
  * real savings require an A/B run against baseline CADO-NFS
""")
    return ok


# =============================================================================
# §12. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="GNFS scale orchestrator")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--estimate", action="store_true",
                    help="print honest resource table")
    ap.add_argument("--domain-report", action="store_true")
    ap.add_argument("--plan", type=str, default=None, help="integer N")
    ap.add_argument("--attach-log", type=str, default=None)
    ap.add_argument("--demo-log", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--run", type=str, default=None, help="integer N to run")
    ap.add_argument("--backend", choices=["cado", "msieve"], default="cado")
    ap.add_argument("--authorized", action="store_true",
                    help="assert you own / are authorized for this N")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workdir", default="./gnfs_work")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1
    if a.estimate:
        print_resource_table()
        return 0
    if a.domain_report:
        print_domain_report()
        return 0

    cfg = RunConfig(workdir=a.workdir, threads=a.threads,
                    authorized=a.authorized)
    orch = GNFSScaleOrchestrator(cfg)
    out: Dict[str, Any] = {}

    if a.plan:
        out["plan"] = orch.plan(int(a.plan))
    if a.attach_log:
        out["log_analysis"] = orch.attach_log(
            Path(a.attach_log).read_text(errors="ignore"))
    if a.demo_log:
        out["log_analysis"] = orch.attach_log(demo_log())
    if a.benchmark:
        out["benchmark"] = benchmark_control_layer()
    if a.run:
        out["execute"] = orch.execute(int(a.run), backend=a.backend,
                                      timeout=a.timeout)

    if out:
        print(json.dumps(out, indent=2, default=str))
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=2, default=str))
            print(f"\nsaved -> {a.json}")
        return 0

    ap.print_help()
    print(f"\n{ATTRIBUTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
