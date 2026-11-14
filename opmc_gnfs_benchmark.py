#!/usr/bin/env python3
"""
=============================================================================
OPMC-GNFS BENCHMARK HARNESS
Real-log parser + OPMC fractal-similarity compression + Paper diagnostic suite
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

PURPOSE
  Benchmark harness for GNFS sieving/filtering logs (CADO-NFS / GGNFS style).
  Uses an OPMC-style structural compressor under fractal similarity, and
  reports results as tables.

USES THE FIVE PAPERS AS DIAGNOSTIC/CONTROL LAYERS
  [P7] External noise diagnostic protocol
       -> classifies external log noise: stationarity, AMS-like Cesaro,
          log-scale stability, residual/change-point behavior.
  [P8] Internal diagnostic battery
       -> each convergence statistic is used as its own test:
          scaling/self-similarity, tail behavior, change points.
  [R21] OPMC under fractal interfaces / double-exponential timing
       -> structural weak value is computed on a compact feature space;
          real-time averages are not assumed to converge.
  [R22] Exact finite-time predictability under known generating law
       -> finite-window prediction benchmark; synthetic known-law mode
          can verify exact predictability.
  [UOT] Unified Ordinal-Tensor measurement theory
       -> feature vectors are treated as structural observables; OPMC
          compression clusters them by fractal similarity.

WHAT IS MEASURED
  1. Parse real CADO/GGNFS-like logs into relation-yield time series.
  2. Extract windows and structural features.
  3. Run Paper7/Paper8 style diagnostics.
  4. Compress feature trajectory using OPMC fractal-similarity clustering.
  5. Reconstruct and predict relation yield.
  6. Report compression ratio, reconstruction error, prediction MAE,
     tail index, Hurst/scaling exponent, change-point class, etc.

HONEST SCOPE
  ❌ No real CADO-NFS/GGNFS log is bundled.
  ❌ This does not factor integers.
  ❌ Compression is feature-space/semantic compression, not proof that
     GNFS complexity is reduced.
  ❌ If parser does not match your site-specific log format, extend the
     alias table or provide normalized key=value logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"

# =============================================================================
# 1. LOG RECORD AND PARSER
# =============================================================================

@dataclass
class LogRecord:
    source: str
    line_no: int
    raw: str
    timestamp: Optional[float] = None
    stage: str = "unknown"
    q: Optional[float] = None
    rels: Optional[float] = None
    ideals: Optional[float] = None
    cpu: Optional[float] = None
    norm: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def primary_yield(self) -> Optional[float]:
        if self.rels is not None:
            return float(self.rels)
        if self.ideals is not None:
            return float(self.ideals)
        return None


KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "q": ("q", "special_q", "specialq", "q0", "prime", "prime_q", "special_prime"),
    "rels": ("rels", "relations", "relations_found", "rel_count", "nrels",
             "relations_total", "rel"),
    "ideals": ("ideals", "prime_ideals", "ideals_found", "ideals_total",
               "primeideals"),
    "cpu": ("cpu", "cpu_time", "time", "elapsed", "seconds", "sec", "t",
            "wall", "wall_time"),
    "norm": ("norm", "normq", "qnorm", "norm_q"),
}

STAGE_KEYWORDS = {
    "polysel": ("polyselect", "polynomial", "polysel", "murphy"),
    "sieve": ("sieve", "sieving", "lattice", "specialq", "q="),
    "filter": ("filter", "filtering", "dedup", "merge", "forfilter"),
    "linalg": ("linalg", "linear algebra", "matrix", "nullspace", "bwc"),
}

TS_RE = re.compile(r"^\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?\s*")
KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*"
    r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)


def parse_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


class CadoGgnfsLogParser:
    """
    Flexible key=value parser for CADO-NFS/GGNFS-ish logs.

    Works best when lines contain explicit fields such as:
      q=123456 rels=789 cpu=1.23
      special_q=123456 relations_found=789 time=1.23

    If your log format is different, normalize it to key=value lines or
    extend KEY_ALIASES.
    """

    def __init__(self, require_yield: bool = True):
        self.require_yield = require_yield

    def detect_stage(self, line: str) -> str:
        low = line.lower()
        for stage, words in STAGE_KEYWORDS.items():
            for w in words:
                if w in low:
                    return stage
        return "unknown"

    def parse_line(self, line: str, source: str, line_no: int) -> Optional[LogRecord]:
        line = line.strip()
        if not line:
            return None

        ts = None
        m = TS_RE.match(line)
        if m:
            ts = parse_float(m.group(1))

        rec = LogRecord(
            source=source,
            line_no=line_no,
            raw=line[:500],
            timestamp=ts,
            stage=self.detect_stage(line),
        )

        found_any = False
        for key, val_s in KV_RE.findall(line):
            k = key.lower()
            val = parse_float(val_s)
            if val is None:
                continue
            found_any = True

            matched = False
            for canonical, aliases in KEY_ALIASES.items():
                if k in aliases:
                    setattr(rec, canonical, val)
                    matched = True
                    break
            if not matched:
                rec.extra[k] = val

        # If q appears as bare token "123456q" etc., ignore for now.
        if self.require_yield and rec.primary_yield() is None:
            return None

        if not found_any:
            return None

        return rec

    def parse_text(self, text: str, source: str = "text") -> List[LogRecord]:
        out: List[LogRecord] = []
        for i, line in enumerate(text.splitlines(), 1):
            rec = self.parse_line(line, source, i)
            if rec is not None:
                out.append(rec)
        return out

    def parse_file(self, path: str | Path) -> List[LogRecord]:
        p = Path(path)
        if not p.exists():
            return []
        text = p.read_text(encoding="utf-8", errors="ignore")
        return self.parse_text(text, source=p.name)


# =============================================================================
# 2. SYNTHETIC DEMO LOG GENERATOR
# =============================================================================

def generate_demo_logs(outdir: str | Path, n_lines: int = 1200, seed: int = 42) -> Dict[str, str]:
    """
    Generate demo logs in CADO-like and GGNFS-like styles.

    These are synthetic, but structured to contain:
      - regime changes
      - heavy-tailed relation bursts
      - CPU time variations
      - q progression
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    cado_path = outdir / "demo_cado.log"
    ggnfs_path = outdir / "demo_ggnfs.log"

    cado_lines: List[str] = []
    ggnfs_lines: List[str] = []

    t = 0.0
    q = 100_000.0
    regime = 0

    for i in range(n_lines):
        # Regime lengths roughly geometric, with occasional heavy-tail burst
        if i % 173 == 0:
            regime = (regime + 1) % 3

        base_rate = [80.0, 220.0, 45.0][regime]
        q += rng.uniform(500.0, 5000.0)

        # Heavy-tailed multiplicative noise
        burst = 1.0
        if rng.random() < 0.04:
            burst = rng.pareto(1.3) + 1.0

        rels = max(0.0, base_rate * burst * rng.lognormal(0.0, 0.35))
        ideals = rels * rng.uniform(0.75, 0.98)
        cpu = max(1e-3, rng.gamma(2.0, 0.15) + 0.001 * math.log1p(q))
        norm = q * rng.uniform(0.9, 1.1)
        t += cpu

        cado_lines.append(
            f"[{t:.6f}] sieving: q={q:.0f} rels={rels:.3f} ideals={ideals:.3f} "
            f"cpu={cpu:.6f} norm={norm:.3f}"
        )

        ggnfs_lines.append(
            f"q={q:.0f} relations_found={rels:.3f} time={cpu:.6f} norm={norm:.3f}"
        )

        # Occasional filter/linalg style lines
        if i % 97 == 0:
            cado_lines.append(
                f"[{t:.6f}] filter: rels={rels*0.42:.3f} cpu={cpu*0.11:.6f}"
            )
        if i % 151 == 0:
            ggnfs_lines.append(
                f"lattice_sieve q={q:.0f} rels={rels*0.87:.3f} cpu={cpu*0.43:.6f}"
            )

    cado_path.write_text("\n".join(cado_lines) + "\n", encoding="utf-8")
    ggnfs_path.write_text("\n".join(ggnfs_lines) + "\n", encoding="utf-8")

    return {"cado": str(cado_path), "ggnfs": str(ggnfs_path)}


# =============================================================================
# 3. TIME SERIES UTILITIES
# =============================================================================

def records_to_series(records: List[LogRecord], field: str = "rels") -> np.ndarray:
    vals: List[float] = []
    for r in records:
        if field == "rels":
            v = r.rels
        elif field == "ideals":
            v = r.ideals
        elif field == "cpu":
            v = r.cpu
        elif field == "q":
            v = r.q
        elif field == "norm":
            v = r.norm
        else:
            v = r.extra.get(field)

        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            vals.append(v)

    return np.asarray(vals, dtype=np.float64)


def safe_finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def hill_theta(pos_values: np.ndarray, frac: float = 0.2) -> float:
    """
    Hill tail-index estimate.
    Returns theta; larger theta means lighter tail.
    If unavailable, return 0.
    """
    x = pos_values[pos_values > 0]
    if len(x) < 12:
        return 0.0

    k = max(5, int(frac * len(x)))
    k = min(k, len(x) - 1)

    xs = np.sort(x)[::-1]
    x_k = xs[k]
    if x_k <= 0:
        return 0.0

    logs = np.log(xs[:k] / x_k)
    mean_log = float(np.mean(logs))
    if mean_log <= 1e-12:
        return 0.0

    return float(1.0 / mean_log)


def cusum_change_points(x: np.ndarray, threshold: float = 4.0, drift: float = 0.3) -> List[int]:
    """
    Simple CUSUM change-point detector.
    """
    if len(x) < 8:
        return []

    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-12:
        return []

    z = (x - mu) / sd
    s_pos = 0.0
    s_neg = 0.0
    cps: List[int] = []

    for i, zi in enumerate(z):
        s_pos = max(0.0, s_pos + zi - drift)
        s_neg = max(0.0, s_neg - zi - drift)
        if s_pos > threshold or s_neg > threshold:
            cps.append(i)
            s_pos = 0.0
            s_neg = 0.0

    return cps


def hurst_like_exponent(x: np.ndarray) -> float:
    """
    Rough Hurst-like scaling exponent via structure function.
    Returns H, with H=0.5 as neutral fallback.
    """
    if len(x) < 16:
        return 0.5

    lags = []
    sfs = []

    max_lag = min(64, len(x) // 4)
    lag = 1
    while lag <= max_lag:
        d = x[lag:] - x[:-lag]
        if len(d) < 4:
            break
        sf = float(np.mean(d * d))
        if sf > 1e-18:
            lags.append(lag)
            sfs.append(sf)
        lag *= 2

    if len(lags) < 3:
        return 0.5

    lx = np.log(np.asarray(lags, dtype=float))
    ly = np.log(np.asarray(sfs, dtype=float))
    try:
        slope = float(np.polyfit(lx, ly, 1)[0])
        H = 0.5 * slope
        if not math.isfinite(H):
            return 0.5
        return float(np.clip(H, 0.0, 1.5))
    except Exception:
        return 0.5


def multiscale_stability(x: np.ndarray) -> float:
    """
    Multiscale block-mean stability feature.
    Returns value in [0,1]; higher means more stable across scales.
    """
    if len(x) < 16:
        return 0.5

    means = []
    for block in (2, 4, 8):
        if len(x) < block * 4:
            continue
        n = len(x) // block
        if n < 2:
            continue
        xb = x[:n * block].reshape(n, block).mean(axis=1)
        means.append(float(np.mean(xb)))

    if len(means) < 2:
        return 0.5

    m = float(np.mean(means))
    s = float(np.std(means))
    return float(1.0 / (1.0 + s / (abs(m) + 1e-12)))


def classify_change_point_tower(cps: List[int], n: int) -> Dict[str, Any]:
    """
    Paper6/R21-inspired block-length classification.
    """
    if len(cps) < 4 or n <= 0:
        return {
            "status": "insufficient",
            "n_change_points": len(cps),
            "classification": "unknown",
        }

    pts = sorted(set([0] + cps + [n]))
    lens = np.diff(pts).astype(float)
    lens = lens[lens > 0]

    if len(lens) < 4:
        return {
            "status": "insufficient",
            "n_change_points": len(cps),
            "classification": "unknown",
        }

    idx = np.arange(1, len(lens) + 1, dtype=float)

    try:
        slope_regular = float(np.polyfit(np.log(idx), np.log(lens), 1)[0])
    except Exception:
        slope_regular = float("nan")

    try:
        slope_dexp = float(np.polyfit(idx, np.log(np.log(lens + 1.0) + 1.0), 1)[0])
    except Exception:
        slope_dexp = float("nan")

    if math.isfinite(slope_dexp) and math.isfinite(slope_regular):
        if slope_dexp > 0.15 and slope_regular > 1.3:
            cls = "double_exponential_like"
        elif slope_regular > 0.9:
            cls = "regularly_varying_like"
        elif slope_regular <= 0.25:
            cls = "finite_mean_like"
        else:
            cls = "intermediate"
    else:
        cls = "unknown"

    return {
        "status": "ok",
        "n_change_points": len(cps),
        "classification": cls,
        "slope_regular": slope_regular,
        "slope_double_exp": slope_dexp,
        "block_lengths_head": lens[:10].tolist(),
    }


def diagnostic_battery(series: np.ndarray) -> Dict[str, Any]:
    """
    Paper7/Paper8 style diagnostic battery for external log series.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 8:
        return {
            "status": "insufficient_data",
            "n": int(len(x)),
        }

    pos = x[x > 0]
    logx = np.log1p(np.maximum(x, 0.0))

    theta = hill_theta(pos)
    H = hurst_like_exponent(logx)
    cps = cusum_change_points(x)
    tower = classify_change_point_tower(cps, len(x))
    ms_stable = multiscale_stability(x)

    # Cesaro/AMS-like stability: compare late running mean to earlier mean
    half = len(x) // 2
    mean1 = float(np.mean(x[:half])) if half > 0 else 0.0
    mean2 = float(np.mean(x[half:])) if half > 0 else 0.0
    denom = max(abs(mean1), abs(mean2), 1e-12)
    cesaro_drift = abs(mean2 - mean1) / denom

    # Log-scale stationarity proxy: compare log means of halves
    log1 = float(np.mean(logx[:half])) if half > 0 else 0.0
    log2 = float(np.mean(logx[half:])) if half > 0 else 0.0
    log_drift = abs(log2 - log1) / max(1e-6, abs(log1) + abs(log2) + 1e-12)

    return {
        "status": "ok",
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "positive_count": int(len(pos)),
        "hill_theta": theta,
        "hurst_like": H,
        "multiscale_stability": ms_stable,
        "n_change_points": len(cps),
        "change_points_head": cps[:25],
        "tower": tower,
        "cesaro_drift": cesaro_drift,
        "log_scale_drift": log_drift,
        "heuristic_stationary": bool(cesaro_drift < 0.35),
        "heuristic_logscale_stationary": bool(log_drift < 0.35),
    }


# =============================================================================
# 4. WINDOW FEATURES
# =============================================================================

def make_windows(x: np.ndarray, window: int = 16, step: int = 8) -> List[Tuple[int, float, np.ndarray]]:
    """
    Returns list of (start_index, target_mean, feature_vector).
    Target is mean raw yield in window.
    """
    out: List[Tuple[int, float, np.ndarray]] = []
    if len(x) < window:
        return out

    for start in range(0, len(x) - window + 1, step):
        w = x[start:start + window]
        target = float(np.mean(w))
        feat = window_features(w)
        out.append((start, target, feat))

    return out


def window_features(w: np.ndarray) -> np.ndarray:
    """
    Structural feature vector for one window.

    Features:
      0  log_mean
      1  raw_mean
      2  std
      3  cv
      4  hill_theta
      5  hurst_like
      6  multiscale_stability
      7  n_change_points
      8  skewness-like
      9  p25
      10 p75
      11 range
    """
    w = np.asarray(w, dtype=np.float64)
    pos = w[w > 0]

    mean = float(np.mean(w))
    log_mean = float(np.mean(np.log1p(np.maximum(w, 0.0))))
    std = float(np.std(w))
    cv = std / (abs(mean) + 1e-12)

    theta = hill_theta(pos)
    H = hurst_like_exponent(np.log1p(np.maximum(w, 0.0)))
    ms = multiscale_stability(w)
    cps = cusum_change_points(w)

    if std > 1e-12:
        skew = float(np.mean(((w - mean) / std) ** 3))
    else:
        skew = 0.0

    p25 = float(np.percentile(w, 25)) if len(w) else 0.0
    p75 = float(np.percentile(w, 75)) if len(w) else 0.0
    rng = float(np.max(w) - np.min(w)) if len(w) else 0.0

    feat = np.array([
        log_mean,
        mean,
        std,
        cv,
        theta,
        H,
        ms,
        float(len(cps)),
        skew,
        p25,
        p75,
        rng,
    ], dtype=np.float64)

    feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
    return feat


# =============================================================================
# 5. OPMC FRACTAL-SIMILARITY COMPRESSOR
# =============================================================================

class OPMCFractalCompressor:
    """
    OPMC-style compressor under fractal similarity.

    This is a structural/semantic compressor:
      - windows are mapped to structural feature vectors
      - similar windows are clustered by cosine similarity in normalized
        feature space
      - each cluster stores a structural weak value (centroid + target mean)
      - original trajectory is encoded as cluster IDs

    It is NOT a formal proof of GNFS complexity reduction.
    It is a measurable benchmark of redundancy in log dynamics.
    """

    def __init__(self, similarity_threshold: float = 0.94, max_clusters: int = 128):
        self.similarity_threshold = float(similarity_threshold)
        self.max_clusters = int(max_clusters)

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

        self.centroids_raw_: List[np.ndarray] = []
        self.centroids_norm_: List[np.ndarray] = []
        self.counts_: List[int] = []
        self.target_means_: List[float] = []

        self.last_codes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------ helpers
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Compressor not fitted")
        return (X - self.mean_) / self.std_

    @staticmethod
    def _row_norm(Z: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        return Z / n

    def _add_cluster(self, z_raw: np.ndarray, target: float) -> int:
        idx = len(self.centroids_raw_)
        self.centroids_raw_.append(z_raw.copy())
        nz = np.linalg.norm(z_raw)
        self.centroids_norm_.append(z_raw / nz if nz > 1e-12 else z_raw.copy())
        self.counts_.append(1)
        self.target_means_.append(float(target))
        return idx

    def _update_cluster(self, idx: int, z_raw: np.ndarray, target: float) -> None:
        c = self.counts_[idx]
        new_raw = (self.centroids_raw_[idx] * c + z_raw) / (c + 1)
        new_target = (self.target_means_[idx] * c + float(target)) / (c + 1)

        self.centroids_raw_[idx] = new_raw
        self.counts_[idx] = c + 1
        self.target_means_[idx] = new_target

        nz = np.linalg.norm(new_raw)
        self.centroids_norm_[idx] = new_raw / nz if nz > 1e-12 else new_raw.copy()

    # ------------------------------------------------------------ fit/encode
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OPMCFractalCompressor":
        """
        Fit compressor on feature matrix X.
        y = target yields per window (optional but recommended).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y is None:
            y = np.zeros(len(X), dtype=np.float64)
        else:
            y = np.asarray(y, dtype=np.float64)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9

        Z = self._standardize(X)
        ZN = self._row_norm(Z)

        self.centroids_raw_.clear()
        self.centroids_norm_.clear()
        self.counts_.clear()
        self.target_means_.clear()

        for i in range(len(ZN)):
            z = ZN[i]
            target = float(y[i])

            if not self.centroids_norm_:
                self._add_cluster(Z[i], target)
                continue

            C = np.stack(self.centroids_norm_, axis=0)
            sims = C @ z
            best = int(np.argmax(sims))
            best_sim = float(sims[best])

            if best_sim >= self.similarity_threshold:
                self._update_cluster(best, Z[i], target)
            else:
                if len(self.centroids_raw_) < self.max_clusters:
                    self._add_cluster(Z[i], target)
                else:
                    # fallback: assign to nearest cluster
                    self._update_cluster(best, Z[i], target)

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode windows to cluster IDs.
        """
        if self.mean_ is None or not self.centroids_norm_:
            raise RuntimeError("Compressor not fitted")

        X = np.asarray(X, dtype=np.float64)
        Z = self._standardize(X)
        ZN = self._row_norm(Z)

        C = np.stack(self.centroids_norm_, axis=0)
        sims = ZN @ C.T
        codes = np.argmax(sims, axis=1).astype(np.int32)
        self.last_codes_ = codes
        return codes

    def decode_features(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct standardized feature vectors from codes.
        """
        if not self.centroids_raw_:
            raise RuntimeError("Compressor not fitted")

        C = np.stack(self.centroids_raw_, axis=0)
        Zrec = C[np.asarray(codes, dtype=np.int64)]
        Xrec = Zrec * self.std_ + self.mean_
        return Xrec

    def compression_stats(self, X: np.ndarray, codes: np.ndarray) -> Dict[str, Any]:
        """
        Feature-space and byte-space compression statistics.
        """
        X = np.asarray(X, dtype=np.float64)

        original_bytes = X.size * 8
        centroid_bytes = len(self.centroids_raw_) * X.shape[1] * 8
        code_bytes = codes.nbytes
        overhead_bytes = 1024  # conservative metadata overhead

        compressed_bytes = centroid_bytes + code_bytes + overhead_bytes
        ratio = original_bytes / max(1, compressed_bytes)

        payload = {
            "codes": codes.tolist(),
            "centroids": [c.tolist() for c in self.centroids_raw_],
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "std": self.std_.tolist() if self.std_ is not None else [],
            "target_means": self.target_means_,
        }
        zb = zlib.compress(json.dumps(payload).encode("utf-8"), level=9)
        zlib_ratio = original_bytes / max(1, len(zb))

        return {
            "n_windows": int(len(codes)),
            "n_clusters": int(len(self.centroids_raw_)),
            "feature_dim": int(X.shape[1]),
            "original_bytes": int(original_bytes),
            "compressed_bytes": int(compressed_bytes),
            "feature_compression_ratio": float(ratio),
            "zlib_bytes": int(len(zb)),
            "zlib_compression_ratio": float(zlib_ratio),
        }

    def structural_weak_value(self, codes: np.ndarray) -> float:
        """
        OPMC/R21-style structural weak value:
        weighted average of cluster target means.

        This is computed on the compact structural space (clusters),
        not on raw time order. It intentionally avoids assuming that
        the raw time-average converges under pathological timing.
        """
        if not self.target_means_:
            return 0.0

        codes = np.asarray(codes, dtype=np.int64)
        total = 0.0
        wsum = 0.0

        for idx, cnt in enumerate(self.counts_):
            if idx < len(self.target_means_):
                total += float(cnt) * float(self.target_means_[idx])
                wsum += float(cnt)

        if wsum <= 0:
            return 0.0

        return float(total / wsum)


# =============================================================================
# 6. BENCHMARK RUNNER
# =============================================================================

@dataclass
class BenchmarkRow:
    source: str
    n_records: int
    n_series: int
    n_windows: int
    n_clusters: int
    feature_compression_ratio: float
    zlib_compression_ratio: float
    reconstruction_rmse_log_mean: float
    baseline_mae_next: float
    opmc_mae_next: float
    hill_theta: float
    hurst_like: float
    n_change_points: int
    tower_class: str
    structural_weak_value: float
    status: str = "ok"


def run_benchmark_on_series(
    source_name: str,
    series: np.ndarray,
    window: int = 16,
    step: int = 8,
    train_frac: float = 0.7,
    similarity_threshold: float = 0.94,
    max_clusters: int = 128,
) -> Optional[BenchmarkRow]:
    series = np.asarray(series, dtype=np.float64)
    series = series[np.isfinite(series)]

    if len(series) < max(32, window * 3):
        return BenchmarkRow(
            source=source_name,
            n_records=0,
            n_series=len(series),
            n_windows=0,
            n_clusters=0,
            feature_compression_ratio=0.0,
            zlib_compression_ratio=0.0,
            reconstruction_rmse_log_mean=0.0,
            baseline_mae_next=0.0,
            opmc_mae_next=0.0,
            hill_theta=0.0,
            hurst_like=0.0,
            n_change_points=0,
            tower_class="insufficient_data",
            structural_weak_value=0.0,
            status="insufficient_data",
        )

    diag = diagnostic_battery(series)
    windows = make_windows(series, window=window, step=step)

    if len(windows) < 8:
        return BenchmarkRow(
            source=source_name,
            n_records=0,
            n_series=len(series),
            n_windows=len(windows),
            n_clusters=0,
            feature_compression_ratio=0.0,
            zlib_compression_ratio=0.0,
            reconstruction_rmse_log_mean=0.0,
            baseline_mae_next=0.0,
            opmc_mae_next=0.0,
            hill_theta=safe_finite(diag.get("hill_theta")),
            hurst_like=safe_finite(diag.get("hurst_like")),
            n_change_points=int(diag.get("n_change_points", 0)),
            tower_class=str(diag.get("tower", {}).get("classification", "unknown")),
            structural_weak_value=0.0,
            status="too_few_windows",
        )

    starts = np.array([w[0] for w in windows], dtype=np.int64)
    targets = np.array([w[1] for w in windows], dtype=np.float64)
    X = np.stack([w[2] for w in windows], axis=0)

    n = len(X)
    split = max(4, int(train_frac * n))
    split = min(split, n - 4)

    X_train = X[:split]
    y_train = targets[:split]
    X_test = X[split:]
    y_test = targets[split:]

    comp = OPMCFractalCompressor(
        similarity_threshold=similarity_threshold,
        max_clusters=max_clusters,
    )
    comp.fit(X_train, y_train)

    codes_all = comp.encode(X)
    Xrec = comp.decode_features(codes_all)
    comp_stats = comp.compression_stats(X, codes_all)
    weak_value = comp.structural_weak_value(codes_all)

    # Reconstruction RMSE on log_mean feature (index 0)
    rmse_log_mean = float(np.sqrt(np.mean((Xrec[:, 0] - X[:, 0]) ** 2)))

    # Next-window prediction benchmark
    test_codes = comp.encode(X_test)
    preds_opmc = np.array(
        [comp.target_means_[c] if c < len(comp.target_means_) else 0.0
         for c in test_codes],
        dtype=np.float64,
    )

    # Predict next target from current window
    if len(y_test) >= 2:
        y_true_next = y_test[1:]
        y_pred_base = y_test[:-1]

        y_pred_opmc = preds_opmc[:-1]
        # OPMC prediction is cluster target of current window;
        # baseline is persistence.
        mae_base = float(np.mean(np.abs(y_pred_base - y_true_next)))
        mae_opmc = float(np.mean(np.abs(y_pred_opmc - y_true_next)))
    else:
        mae_base = 0.0
        mae_opmc = 0.0

    return BenchmarkRow(
        source=source_name,
        n_records=0,
        n_series=len(series),
        n_windows=n,
        n_clusters=comp_stats["n_clusters"],
        feature_compression_ratio=comp_stats["feature_compression_ratio"],
        zlib_compression_ratio=comp_stats["zlib_compression_ratio"],
        reconstruction_rmse_log_mean=rmse_log_mean,
        baseline_mae_next=mae_base,
        opmc_mae_next=mae_opmc,
        hill_theta=safe_finite(diag.get("hill_theta")),
        hurst_like=safe_finite(diag.get("hurst_like")),
        n_change_points=int(diag.get("n_change_points", 0)),
        tower_class=str(diag.get("tower", {}).get("classification", "unknown")),
        structural_weak_value=weak_value,
        status="ok",
    )


def run_benchmark_on_file(
    path: str | Path,
    window: int = 16,
    step: int = 8,
    train_frac: float = 0.7,
    similarity_threshold: float = 0.94,
    max_clusters: int = 128,
) -> BenchmarkRow:
    parser = CadoGgnfsLogParser(require_yield=True)
    records = parser.parse_file(path)

    rel_series = records_to_series(records, field="rels")
    if len(rel_series) < 8:
        rel_series = records_to_series(records, field="ideals")

    row = run_benchmark_on_series(
        source_name=Path(path).name,
        series=rel_series,
        window=window,
        step=step,
        train_frac=train_frac,
        similarity_threshold=similarity_threshold,
        max_clusters=max_clusters,
    )

    if row is None:
        return BenchmarkRow(
            source=Path(path).name,
            n_records=len(records),
            n_series=0,
            n_windows=0,
            n_clusters=0,
            feature_compression_ratio=0.0,
            zlib_compression_ratio=0.0,
            reconstruction_rmse_log_mean=0.0,
            baseline_mae_next=0.0,
            opmc_mae_next=0.0,
            hill_theta=0.0,
            hurst_like=0.0,
            n_change_points=0,
            tower_class="unknown",
            structural_weak_value=0.0,
            status="failed",
        )

    row.n_records = len(records)
    return row


# =============================================================================
# 7. REPORTING
# =============================================================================

def print_table(rows: List[BenchmarkRow]) -> None:
    if not rows:
        print("No benchmark rows.")
        return

    header = (
        f"{'source':24s} {'recs':>6s} {'pts':>6s} {'wins':>5s} "
        f"{'clus':>5s} {'ratioF':>7s} {'ratioZ':>7s} "
        f"{'RMSElog':>8s} {'MAEbase':>8s} {'MAEopmc':>8s} "
        f"{'theta':>6s} {'H':>5s} {'cp':>4s} {'tower':24s} "
        f"{'OPMC weak value':>16s}"
    )

    print("=" * len(header))
    print("OPMC-GNFS BENCHMARK REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in rows:
        line = (
            f"{r.source[:24]:24s} "
            f"{r.n_records:6d} "
            f"{r.n_series:6d} "
            f"{r.n_windows:5d} "
            f"{r.n_clusters:5d} "
            f"{r.feature_compression_ratio:7.2f} "
            f"{r.zlib_compression_ratio:7.2f} "
            f"{r.reconstruction_rmse_log_mean:8.4f} "
            f"{r.baseline_mae_next:8.4f} "
            f"{r.opmc_mae_next:8.4f} "
            f"{r.hill_theta:6.3f} "
            f"{r.hurst_like:5.3f} "
            f"{r.n_change_points:4d} "
            f"{r.tower_class[:24]:24s} "
            f"{r.structural_weak_value:16.4f}"
        )
        print(line)

    print("=" * len(header))
    print("Legend:")
    print("  recs      = parsed log records with usable yield")
    print("  pts       = raw series length")
    print("  wins      = number of analysis windows")
    print("  clus      = OPMC fractal clusters")
    print("  ratioF    = feature-space compression ratio")
    print("  ratioZ    = serialized+zlib compression ratio")
    print("  RMSElog   = reconstruction RMSE of log-mean feature")
    print("  MAEbase   = persistence next-window MAE")
    print("  MAEopmc   = OPMC centroid next-window MAE")
    print("  theta     = Hill tail index estimate")
    print("  H         = Hurst-like scaling exponent")
    print("  cp        = number of CUSUM change points")
    print("  tower     = change-point block growth classification")
    print("  weak value= OPMC structural weak value (cluster-weighted)")
    print()
    print("HONEST NOTES")
    print("  - This benchmark measures log redundancy and predictability.")
    print("  - It does NOT prove faster integer factorization.")
    print("  - OPMC weak value is computed on compressed structural space,")
    print("    not by assuming raw time averages converge.")
    print("  - Real CADO/GGNFS logs may need site-specific parser tuning.")


def rows_to_json(rows: List[BenchmarkRow]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in rows]


# =============================================================================
# 8. CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="OPMC-GNFS benchmark harness for CADO/GGNFS-like logs"
    )
    ap.add_argument("--logs", nargs="*", default=None,
                    help="Paths to real CADO/GGNFS log files")
    ap.add_argument("--demo", action="store_true",
                    help="Generate synthetic demo logs and benchmark them")
    ap.add_argument("--outdir", default="opmc_gnfs_demo_logs",
                    help="Directory for demo logs")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--similarity", type=float, default=0.94)
    ap.add_argument("--max-clusters", type=int, default=128)
    ap.add_argument("--json", default=None, help="Write JSON report to path")

    args = ap.parse_args(argv)

    log_paths: List[str] = []

    if args.demo:
        print("Generating synthetic CADO/GGNFS-like demo logs...")
        files = generate_demo_logs(args.outdir)
        log_paths.extend(files.values())

    if args.logs:
        log_paths.extend(args.logs)

    if not log_paths:
        print("No input logs provided. Generating demo logs automatically.\n")
        files = generate_demo_logs(args.outdir)
        log_paths.extend(files.values())

    rows: List[BenchmarkRow] = []

    for p in log_paths:
        if not Path(p).exists():
            print(f"[WARN] missing log file: {p}")
            continue

        row = run_benchmark_on_file(
            p,
            window=args.window,
            step=args.step,
            train_frac=args.train_frac,
            similarity_threshold=args.similarity,
            max_clusters=args.max_clusters,
        )
        rows.append(row)

    print_table(rows)

    if args.json:
        Path(args.json).write_text(
            json.dumps(rows_to_json(rows), indent=2),
            encoding="utf-8",
        )
        print(f"JSON report written to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
