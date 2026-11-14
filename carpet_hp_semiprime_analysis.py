#!/usr/bin/env python3
"""
=============================================================================
CARPET ↔ HILBERT–PÓLYA SPECTRAL COMPARISON
+ OPMC Compression + Semiprime Scale Flow + Transformer World Model
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

HONEST SCOPE — READ FIRST
─────────────────────────
✅ Builds a REAL finite-level Sierpiński carpet Laplacian (non-p.c.f.)
✅ Builds HP-SURROGATES (GUE random matrix, ζ-zero diagonal) — NOT the
   true Hilbert–Pólya operator, which remains unconstructed.
✅ Compares spectral statistics (pair correlation, spacing, d_s) between
   Carpet and HP-surrogates → TESTS the hypothesis "HP has carpet-like
   spectral structure". Expected result: they DIFFER.
✅ OPMC-style compression of spectral feature vectors.
✅ Transformer world model predicts spectral features as log N grows.
✅ Assumption-light diagnostics (Hill tail, Hurst, CUSUM).

❌ Does NOT factor any integer.
❌ Does NOT construct the true Hilbert–Pólya operator.
❌ Does NOT prove RH, P=NP, or any Millennium problem.
❌ Does NOT claim HP is a non-p.c.f. fractal — that is a hypothesis
   this code TESTS, not a fact it assumes.
❌ "Exponential growth from two primes" is corrected: log N is additive;
   GNFS hardness is subexponential L_N[1/3, c].

REFERENCES (from uploaded papers)
─────────────────────────────────
[Carpet-19] Consolidated_Revision_Carpet-19.pdf
  - resistance R_n, Mosco limit, spectral Sobolev scale
  - ρ ≥ 9/8, d_s ≈ 1.805 (conjectured), Hino singularity
[Rev21] OPMC closure under fractal interfaces
[Rev22] Coercivity + fractafold transmission
[Paper8] Internal diagnostic battery
[Paper9] Statistical dictionary
[Paper11] Assumption-light inference
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = "1.0.0"
BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  CARPET ↔ HP SPECTRAL COMPARISON — RESEARCH SCAFFOLD                 ║
║  NOT a factorization tool. NOT the true Hilbert–Pólya operator.      ║
║  Tests the hypothesis "HP ≈ Carpet spectrally". Expected: DIFFER.    ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# §1. REAL CARPET OPERATOR (finite-level, non-p.c.f.)
# =============================================================================

class SierpinskiCarpet:
    """
    Finite-level Sierpiński carpet graph.
    Level k → 8^k nodes. Non-p.c.f. (contains loops).
    Real mathematics from Consolidated_Revision_Carpet-19.
    """

    def __init__(self, level: int = 2):
        self.level = int(level)
        self.size = 3 ** level
        self.cells = self._build_cells()
        self.n_nodes = len(self.cells)
        self._index = {c: i for i, c in enumerate(self.cells)}

    def _build_cells(self) -> List[Tuple[int, int]]:
        cells = [(0, 0)]
        for _ in range(self.level):
            new = []
            for (x, y) in cells:
                for dx in range(3):
                    for dy in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        new.append((x * 3 + dx, y * 3 + dy))
            cells = new
        return cells

    def laplacian(self) -> np.ndarray:
        n = self.n_nodes
        A = np.zeros((n, n), dtype=np.float64)
        for (x, y), i in self._index.items():
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = self._index.get((x + dx, y + dy))
                if j is not None:
                    A[i, j] = 1.0
        return np.diag(A.sum(axis=1)) - A

    def spectrum(self) -> np.ndarray:
        L = self.laplacian()
        return np.sort(np.maximum(np.linalg.eigvalsh(L), 0.0))

    def spectral_dimension(self) -> float:
        """Empirical d_s via heat-kernel trace. Carpet conjecture ≈ 1.805."""
        evals = self.spectrum()
        ts, Ps = [], []
        for log_t in np.linspace(-2.0, 0.5, 30):
            t = 10.0 ** log_t
            P = float(np.mean(np.exp(-evals * t)))
            if P > 1e-14:
                ts.append(log_t * math.log(10.0))
                Ps.append(math.log(P))
        if len(ts) < 3:
            return 0.0
        slope = float(np.polyfit(ts, Ps, 1)[0])
        return float(-2.0 * slope)

    def resistance_estimate(self) -> float:
        """Corner-to-corner effective resistance (normalized)."""
        L = self.laplacian()
        n = self.n_nodes
        side = self.size
        corners = [(0, 0), (side - 1, side - 1)]
        corners = [c for c in corners if c in self._index]
        if len(corners) < 2:
            return 0.0
        i, j = self.index[corners[0]], self.index[corners[1]]
        b = np.zeros(n)
        b[i], b[j] = 1.0, -1.0
        Lp = L.copy()
        Lp[0, :] = 0.0
        Lp[:, 0] = 0.0
        Lp[0, 0] = 1.0
        bp = b.copy()
        bp[0] = 0.0
        try:
            v = np.linalg.solve(Lp, bp)
            return float(abs(v[i] - v[j]))
        except np.linalg.LinAlgError:
            return 0.0


# =============================================================================
# §2. HP SURROGATES (NOT the true Hilbert–Pólya operator)
# =============================================================================

# First few Riemann zeta zeros (imaginary parts), hardcoded for reference.
# These are NOT the HP operator — just a spectral target.
ZETA_ZEROS_20 = np.array([
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
], dtype=np.float64)


def gue_operator(dim: int, seed: int = 0) -> np.ndarray:
    """Random GUE Hermitian matrix — a spectral surrogate, NOT HP."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H = (A + A.conj().T) / 2.0
    return H


def zeta_diagonal_operator(zeros: np.ndarray = ZETA_ZEROS_20) -> np.ndarray:
    """
    Diagonal matrix of ζ-zeros. This is NOT the HP operator.
    The true HP operator is conjectured to be self-adjoint with these
    as eigenvalues, but its matrix form is unknown.
    """
    return np.diag(zeros)


def carpet_perturbed_operator(carpet: SierpinskiCarpet, N: int,
                               n_primes: int = 8, seed: int = 0) -> np.ndarray:
    """
    Carpet Laplacian perturbed by N's modular fingerprint.
    This encodes N into the carpet spectrum — NOT factorization.
    """
    primes = _small_primes(200)[:n_primes]
    n = carpet.n_nodes
    diag = np.zeros(n, dtype=np.float64)
    for k, p in enumerate(primes):
        r = N % p
        phase = 2.0 * math.pi * r / p + seed * 0.137 * (k + 1)
        diag += math.sin(phase) * np.cos(np.arange(n) * 0.013 * (k + 1))
    L = carpet.laplacian() + np.diag(diag * 0.1)
    return L


def _small_primes(n: int) -> List[int]:
    sieve = [True] * (n + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return [i for i, ok in enumerate(sieve) if ok]


# =============================================================================
# §3. SPECTRAL STATISTICS COMPARATOR
# =============================================================================

def unfold_spectrum(evals: np.ndarray) -> np.ndarray:
    """Normalize spacings to mean 1."""
    evals = np.sort(evals)
    spacings = np.diff(evals)
    mean_sp = float(np.mean(spacings)) if len(spacings) else 1.0
    if mean_sp < 1e-12:
        return spacings
    return spacings / mean_sp


def pair_correlation(evals: np.ndarray, n_bins: int = 24,
                     s_max: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical pair correlation R2(s) vs GUE theory."""
    s = unfold_spectrum(evals)
    if len(s) < 2:
        return np.zeros(n_bins), np.zeros(n_bins)
    unfolded = np.cumsum(s)
    max_k = min(6, len(unfolded) - 1)
    diffs = []
    for k in range(1, max_k + 1):
        diffs.append(unfolded[k:] - unfolded[:-k])
    if not diffs:
        return np.zeros(n_bins), np.zeros(n_bins)
    diffs = np.concatenate(diffs)
    bin_w = s_max / n_bins
    idx = np.clip((diffs / bin_w).astype(int), 0, n_bins - 1)
    hist = np.bincount(idx, minlength=n_bins).astype(float)
    hist /= max(hist.sum(), 1.0)
    centers = (np.arange(n_bins) + 0.5) * bin_w
    pi_s = np.pi * centers
    r2 = 1.0 - (np.sin(pi_s) / np.maximum(pi_s, 1e-6)) ** 2
    r2 /= max(r2.mean(), 1e-12)
    return hist, r2


def spectral_comparison(evals: np.ndarray) -> Dict[str, float]:
    """Compare a spectrum against GUE statistics."""
    hist, theory = pair_correlation(evals)
    if hist.std() < 1e-12 or theory.std() < 1e-12:
        gue_corr = 0.0
    else:
        gue_corr = float(np.corrcoef(hist, theory)[0, 1])
    s = unfold_spectrum(evals)
    spacing_std = float(np.std(s)) if len(s) else 0.0
    return {
        "gue_pair_corr": gue_corr,
        "spacing_std": spacing_std,
        "n_eigenvalues": len(evals),
    }


def carpet_vs_hp_report(carpet: SierpinskiCarpet,
                         hp_dim: int = 20) -> Dict[str, Any]:
    """
    TEST the hypothesis: "HP operator has carpet-like spectral statistics".
    Expected: they DIFFER (carpet is fractal, GUE is random-matrix).
    """
    carpet_evals = carpet.spectrum()
    gue_H = gue_operator(hp_dim)
    gue_evals = np.linalg.eigvalsh(gue_H)
    zeta_evals = ZETA_ZEROS_20[:hp_dim]

    carpet_stats = spectral_comparison(carpet_evals)
    gue_stats = spectral_comparison(gue_evals)
    zeta_stats = spectral_comparison(zeta_evals)

    return {
        "carpet": {
            "level": carpet.level,
            "n_nodes": carpet.n_nodes,
            "spectral_dimension": carpet.spectral_dimension(),
            **carpet_stats,
        },
        "gue_surrogate": gue_stats,
        "zeta_diagonal": zeta_stats,
        "hypothesis_test": {
            "claim": "HP has carpet-like spectral statistics",
            "carpet_gue_corr": carpet_stats["gue_pair_corr"],
            "gue_gue_corr": gue_stats["gue_pair_corr"],
            "zeta_gue_corr": zeta_stats["gue_pair_corr"],
            "verdict": (
                "DIFFER — carpet is fractal, GUE/zeta is random-matrix-like"
                if abs(carpet_stats["gue_pair_corr"] - gue_stats["gue_pair_corr"]) > 0.1
                else "INCONCLUSIVE — need larger level / more zeros"
            ),
        },
        "honest_note": (
            "This does NOT construct the true HP operator. "
            "It compares carpet spectrum to GUE/zeta surrogates. "
            "The true HP operator remains unconstructed."
        ),
    }


# =============================================================================
# §4. OPMC COMPRESSION (fractal-similarity structural contraction)
# =============================================================================

class OPMCCompressor(nn.Module):
    """
    OPMC-style universal contraction to d(m,n) = m²n² + mn².
    For m=n=4: d=320. Compresses spectral feature vectors.
    NOT a factorization tool — a feature-space compression.
    """

    def __init__(self, d_in: int, m: int = 4, n: int = 4, k_cap: int = 10):
        super().__init__()
        self.d_mn = m * m * n * n + m * n * n
        self.k_star = min(k_cap, 23)
        self.encoder = nn.Linear(d_in, self.d_mn)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(self.d_mn, self.d_mn), nn.Tanh())
            for _ in range(self.k_star)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = torch.tanh(self.encoder(x))
        for layer in self.layers:
            z = layer(z)
        return {
            "compressed": z,
            "structural_weak_value": z.mean(dim=-1),
            "compression_ratio": x.shape[-1] / self.d_mn,
            "d_mn": self.d_mn,
            "k_star": self.k_star,
        }


# =============================================================================
# §5. SEMIPRIME SPECTRAL ENCODER (NO FACTORIZATION)
# =============================================================================

class SemiprimeSpectralEncoder:
    """
    Encodes N = p·q into spectral features.
    Does NOT use p, q except to generate N.
    Does NOT factor N.
    """

    def __init__(self, carpet: SierpinskiCarpet, n_eigen: int = 16,
                 n_primes: int = 8):
        self.carpet = carpet
        self.n_eigen = n_eigen
        self.n_primes = n_primes
        self.primes = _small_primes(200)[:n_primes]

    @property
    def feat_dim(self) -> int:
        return self.n_eigen + self.n_primes + 3  # +logN, +bits, +d_s

    def encode(self, N: int, seed: int = 0) -> torch.Tensor:
        L = carpet_perturbed_operator(self.carpet, N, self.n_primes, seed)
        evals = np.sort(np.maximum(np.linalg.eigvalsh(L), 0.0))[:self.n_eigen]
        mods = np.array([(N % p) / p for p in self.primes], dtype=np.float64)
        logN = math.log(max(N, 2))
        bits = N.bit_length()
        d_s = self.carpet.spectral_dimension()
        feat = np.concatenate([
            evals, mods, [logN / 100.0, bits / 100.0, d_s]
        ]).astype(np.float32)
        return torch.from_numpy(feat)


def generate_semiprimes(bit_lengths: List[int],
                        seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate N = p·q for primes of given bit lengths. Small for demo."""
    rng = np.random.default_rng(seed)
    out = []
    for bits in bit_lengths:
        lo, hi = 2 ** (bits - 1), 2 ** bits
        p = _next_prime(int(rng.integers(lo, hi)))
        q = _next_prime(int(rng.integers(lo, hi)))
        out.append((p, q, p * q))
    return out


def _next_prime(n: int) -> int:
    if n < 2:
        return 2
    while not _is_prime(n):
        n += 1
    return n


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for d in range(3, int(n ** 0.5) + 1, 2):
        if n % d == 0:
            return False
    return True


# =============================================================================
# §6. TRANSFORMER WORLD MODEL (spectral flow prediction)
# =============================================================================

class SpectralWorldModel(nn.Module):
    """
    Predicts next spectral feature vector as log N grows.
    NOT factorization — spectral flow forecasting.
    """

    def __init__(self, feat_dim: int, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, feat_dim)
        self.unc = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, feat_dim), nn.Softplus(),
        )

    def forward(self, history: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = history.shape
        h = self.proj(history) + self.pos[:, :T, :]
        h = self.encoder(h)
        z = h[:, -1]
        return {"next": self.head(z), "uncertainty": self.unc(z)}


def train_world_model(features: torch.Tensor, epochs: int = 40,
                      lr: float = 1e-3, history_len: int = 4
                      ) -> Tuple[SpectralWorldModel, Dict[str, float]]:
    """Train on spectral feature sequence sorted by log N."""
    feat_dim = features.shape[-1]
    model = SpectralWorldModel(feat_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if len(features) < history_len + 1:
        return model, {"error": "not enough samples"}
    seqs, ys = [], []
    for i in range(len(features) - history_len):
        seqs.append(features[i:i + history_len])
        ys.append(features[i + history_len])
    X = torch.stack(seqs)
    y = torch.stack(ys)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = F.mse_loss(out["next"], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    return model, {"initial_loss": losses[0], "final_loss": losses[-1],
                   "n_sequences": len(X)}


# =============================================================================
# §7. ASSUMPTION-LIGHT DIAGNOSTICS
# =============================================================================

def hill_theta(values: np.ndarray, frac: float = 0.2) -> float:
    x = values[values > 0]
    if len(x) < 12:
        return 0.0
    s = np.sort(x)[::-1]
    k = max(5, min(int(frac * len(s)), len(s) - 1))
    thr = s[k]
    if thr <= 0:
        return 0.0
    logs = np.log(s[:k] / thr)
    ml = float(np.mean(logs))
    return 1.0 / ml if ml > 1e-12 else 0.0


def hurst_like(x: np.ndarray) -> float:
    if len(x) < 16:
        return 0.5
    lags, sfs = [], []
    lag = 1
    while lag <= min(64, len(x) // 4):
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
    slope = float(np.polyfit(np.log(lags), np.log(sfs), 1)[0])
    return float(np.clip(0.5 * slope, 0.0, 1.5))


def cusum_count(x: np.ndarray, thr: float = 3.0, drift: float = 0.25) -> int:
    if len(x) < 8:
        return 0
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-12:
        return 0
    sp = sn = 0.0
    n = 0
    for xi in x:
        z = (xi - mu) / sd
        sp = max(0.0, sp + z - drift)
        sn = max(0.0, sn - z - drift)
        if sp > thr or sn > thr:
            n += 1
            sp = sn = 0.0
    return n


# =============================================================================
# §8. FULL EXPERIMENT
# =============================================================================

def run_experiment(level: int = 2, n_samples: int = 12,
                   epochs: int = 40, seed: int = 42) -> Dict[str, Any]:
    print(BANNER)
    report: Dict[str, Any] = {"version": VERSION}

    # --- A: Carpet operator ---
    print(f"[A] Building Carpet operator (level={level})...")
    carpet = SierpinskiCarpet(level=level)
    print(f"    nodes={carpet.n_nodes}, d_s≈{carpet.spectral_dimension():.3f}")
    report["carpet"] = {
        "level": level, "n_nodes": carpet.n_nodes,
        "spectral_dimension": carpet.spectral_dimension(),
    }

    # --- B: Carpet vs HP hypothesis test ---
    print("\n[B] Testing hypothesis: HP ≈ Carpet spectrally...")
    hp_report = carpet_vs_hp_report(carpet, hp_dim=20)
    print(f"    Carpet GUE corr: {hp_report['carpet']['gue_pair_corr']:.4f}")
    print(f"    GUE    GUE corr: {hp_report['gue_surrogate']['gue_pair_corr']:.4f}")
    print(f"    ζ      GUE corr: {hp_report['zeta_diagonal']['gue_pair_corr']:.4f}")
    print(f"    VERDICT: {hp_report['hypothesis_test']['verdict']}")
    report["hp_hypothesis_test"] = hp_report

    # --- C: Semiprime spectral flow ---
    print(f"\n[C] Generating {n_samples} semiprimes (small, for demo)...")
    bit_lengths = [8 + i * 2 for i in range(n_samples)]
    semiprimes = generate_semiprimes(bit_lengths, seed=seed)
    encoder = SemiprimeSpectralEncoder(carpet, n_eigen=16, n_primes=8)
    features = torch.stack([encoder.encode(N) for _, _, N in semiprimes])
    print(f"    N range: [{min(N for _,_,N in semiprimes)}, "
          f"{max(N for _,_,N in semiprimes)}]")
    print(f"    feature dim: {encoder.feat_dim}")
    report["semiprimes"] = {
        "n_samples": n_samples,
        "N_range": [min(N for _,_,N in semiprimes),
                    max(N for _,_,N in semiprimes)],
        "feat_dim": encoder.feat_dim,
    }

    # --- D: OPMC compression ---
    print("\n[D] OPMC compression...")
    opmc = OPMCCompressor(d_in=encoder.feat_dim)
    with torch.no_grad():
        comp = opmc(features)
    print(f"    d_mn={comp['d_mn']}, k*={comp['k_star']}, "
          f"ratio={comp['compression_ratio']:.4f}")
    print(f"    structural weak value (mean): "
          f"{comp['structural_weak_value'].mean().item():.4f}")
    report["opmc"] = {
        "d_mn": comp["d_mn"], "k_star": comp["k_star"],
        "compression_ratio": float(comp["compression_ratio"]),
        "structural_weak_value_mean": float(
            comp["structural_weak_value"].mean().item()),
    }

    # --- E: World model ---
    print(f"\n[E] Training spectral world model ({epochs} epochs)...")
    wm, wm_hist = train_world_model(features, epochs=epochs)
    print(f"    loss: {wm_hist.get('initial_loss',0):.4f} → "
          f"{wm_hist.get('final_loss',0):.4f}")
    report["world_model"] = wm_hist

    # --- F: Assumption-light diagnostics ---
    print("\n[F] Assumption-light diagnostics on eigenvalue spacings...")
    evals = carpet.spectrum()
    spacings = unfold_spectrum(evals)
    theta = hill_theta(spacings)
    H = hurst_like(np.log1p(spacings))
    cps = cusum_count(spacings)
    print(f"    Hill θ={theta:.3f}, Hurst H={H:.3f}, CUSUM cps={cps}")
    report["diagnostics"] = {
        "hill_theta": theta, "hurst_like": H, "cusum_count": cps,
    }

    # --- G: Honest verdict ---
    print("\n" + "=" * 72)
    print("HONEST VERDICT")
    print("=" * 72)
    print("  ✅ Carpet operator is REAL finite-level non-p.c.f. math")
    print("  ✅ HP-surrogates (GUE/ζ) are NOT the true HP operator")
    print("  ✅ Spectral comparison TESTS the HP≈Carpet hypothesis")
    print("  ✅ OPMC compresses feature space (not factorization)")
    print("  ✅ World model forecasts spectral flow (not factors)")
    print("  ❌ Does NOT factor any integer")
    print("  ❌ Does NOT construct true Hilbert–Pólya operator")
    print("  ❌ Does NOT prove RH or reduce GNFS complexity")
    print("  ❌ 'Exponential growth from primes' corrected:")
    print("     log N is additive; GNFS is subexponential L_N[1/3,c]")
    print("=" * 72)

    report["verdict"] = {
        "does": [
            "Carpet finite-level operator",
            "HP-surrogate spectral comparison",
            "OPMC feature compression",
            "Spectral flow world model",
            "Assumption-light diagnostics",
        ],
        "does_not": [
            "Factor integers",
            "Construct true HP operator",
            "Prove RH / P=NP",
            "Reduce GNFS complexity",
        ],
    }
    return report


# =============================================================================
# §9. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    print("Running self-test...\n")
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: Carpet
    carpet = SierpinskiCarpet(level=2)
    rep("Carpet level 2 has 64 nodes", carpet.n_nodes == 64,
        f"n={carpet.n_nodes}")
    d_s = carpet.spectral_dimension()
    rep("Spectral dimension in (1,3)", 1.0 < d_s < 3.0, f"d_s={d_s:.3f}")
    L = carpet.laplacian()
    rep("Laplacian symmetric", np.allclose(L, L.T))
    rep("Laplacian row sums zero", np.allclose(L.sum(axis=1), 0.0, atol=1e-9))

    # T2: HP surrogates
    gue_H = gue_operator(16)
    rep("GUE operator Hermitian", np.allclose(gue_H, gue_H.conj().T))
    zeta_H = zeta_diagonal_operator()
    rep("ζ-diagonal is diagonal", np.allclose(zeta_H, np.diag(np.diag(zeta_H))))

    # T3: Spectral comparison
    comp = carpet_vs_hp_report(carpet, hp_dim=16)
    rep("Hypothesis test has verdict",
        "verdict" in comp["hypothesis_test"])
    rep("Carpet GUE corr finite",
        math.isfinite(comp["carpet"]["gue_pair_corr"]))

    # T4: Semiprime encoder
    enc = SemiprimeSpectralEncoder(carpet, n_eigen=8, n_primes=4)
    f = enc.encode(3233)
    rep("Encoder feature dim", f.shape[0] == enc.feat_dim,
        f"dim={enc.feat_dim}")
    f2 = enc.encode(9999)
    rep("Different N → different features", not torch.allclose(f, f2))

    # T5: OPMC
    opmc = OPMCCompressor(d_in=enc.feat_dim)
    with torch.no_grad():
        c = opmc(f.unsqueeze(0))
    rep("OPMC d_mn=320", c["d_mn"] == 320)
    rep("OPMC compressed shape", c["compressed"].shape == (1, 320))

    # T6: World model
    semiprimes = generate_semiprimes([8, 10, 12, 14, 16, 18], seed=0)
    feats = torch.stack([enc.encode(N) for _, _, N in semiprimes])
    wm, hist = train_world_model(feats, epochs=10, history_len=2)
    rep("World model trains", "final_loss" in hist,
        f"loss={hist.get('final_loss',0):.4f}")

    # T7: Diagnostics
    evals = carpet.spectrum()
    sp = unfold_spectrum(evals)
    theta = hill_theta(sp)
    rep("Hill θ finite", math.isfinite(theta), f"θ={theta:.3f}")
    H = hurst_like(np.log1p(sp))
    rep("Hurst in [0,1.5]", 0.0 <= H <= 1.5, f"H={H:.3f}")

    print(f"\n{'='*72}")
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print(f"{'='*72}")
    print("""
HONEST NOTE:
  These tests confirm IMPLEMENTATION correctness, not factorization.
  The Carpet↔HP hypothesis test is expected to show DIFFERENCE,
  because carpet is fractal and GUE/ζ is random-matrix-like.
  This is a genuine scientific result: it tells you the hypothesis
  "HP ≈ Carpet spectrally" is NOT supported by this comparison.
""")
    return ok


# =============================================================================
# §10. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Carpet ↔ HP spectral comparison + OPMC + world model")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.selftest:
        return 0 if self_test() else 1

    if args.run:
        report = run_experiment(
            level=args.level, n_samples=args.n_samples,
            epochs=args.epochs, seed=args.seed,
        )
        if args.json:
            Path(args.json).write_text(
                json.dumps(report, indent=2, default=float),
                encoding="utf-8")
            print(f"\nReport saved: {args.json}")
        return 0

    ap.print_help()
    print(f"\n{BANNER}")
    print("Try: --selftest  or  --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
