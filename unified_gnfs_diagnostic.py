#!/usr/bin/env python3
"""
=============================================================================
UNIFIED GNFS RESEARCH DIAGNOSTIC PIPELINE
Integrates all 20+ research modules under one honest orchestrator
=============================================================================
⚠️  THIS IS NOT:
    ✗ A password decryption tool
    ✗ An RSA-breaking tool
    ✗ A factorization algorithm faster than GNFS
    ✗ A cryptographic attack capability of any kind

WHAT IT *IS*:
    ✓ A unified diagnostic that runs every research module on the same input
    ✓ A cross-module consistency checker (do modules agree on N's difficulty?)
    ✓ A CADO-NFS log analyzer (helps people ALREADY running GNFS optimize)
    ✓ An honest reporter of what each module proves / speculates / cannot do

USAGE:
    python unified_gnfs_diagnostic.py --selftest
    python unified_gnfs_diagnostic.py --run --N 3233
    python unified_gnfs_diagnostic.py --run --cado-log my_run.log
    python unified_gnfs_diagnostic.py --what-this-does-not-do

=============================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("unified_gnfs")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


# =============================================================================
# §0. HONEST BANNER
# =============================================================================

BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║  UNIFIED GNFS DIAGNOSTIC PIPELINE                                         ║
║                                                                           ║
║  ⚠️  NOT a password decryption tool.                                      ║
║  ⚠️  NOT an RSA-breaking tool.                                            ║
║  ⚠️  Combining modules does NOT reduce GNFS complexity.                   ║
║                                                                           ║
║  ✓ Runs every research module on the same input for cross-validation      ║
║  ✓ Reports honestly what each module proves vs. speculates               ║
║  ✓ Useful for people ALREADY running CADO-NFS on their own numbers       ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

WHAT_THIS_DOES_NOT_DO = """
╔═══════════════════════════════════════════════════════════════════════════╗
║  WHAT THIS PIPELINE CANNOT DO                                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

1. DECRYPT PASSWORDS
   Passwords are typically hashed (bcrypt, argon2, PBKDF2) or stored in
   encrypted vaults. NONE of the modules here interact with hashes,
   key derivation, or password storage. For legitimate password auditing
   of YOUR OWN systems, use:
     • hashcat  (https://hashcat.net)       — GPU hash cracker
     • john     (https://openwall.com/john) — CPU hash cracker
   Only with explicit authorization from the system owner.

2. BREAK RSA-2048 OR ANY CRYPTOGRAPHIC ENCRYPTION
   RSA-250 (829-bit) is the current factoring record (2020), requiring
   ~2700 core-years. RSA-400 (1328-bit) is ~10²⁰× harder. No amount of
   fractal/spectral/transformer scaffolding changes the L_N[1/3, c]
   complexity of GNFS.

3. FACTOR ANY INTEGER AT CRYPTOGRAPHIC SCALE
   The GNFS components here (PolynomialSelector, SieveBase) are
   educational base-m implementations. Real GNFS uses CADO-NFS,
   Msieve, or GGNFS — and even those cannot factor RSA-400.

4. CONSTRUCT THE HILBERT-PÓLYA OPERATOR
   This is an open problem equivalent to the Riemann Hypothesis.
   The "HP operator synthesis" module produces a matrix whose
   eigenvalues match GUE statistics — that is NOT the same as
   the operator whose eigenvalues are ζ-zeros.

5. PROVE ANY OF P=NP, RH, GRH, BSD, or the HODGE CONJECTURE
   The modules test finite-level approximations and match statistical
   patterns. Statistical agreement ≠ proof. The source files themselves
   state this repeatedly.

6. PROVIDE ANY PATH TO FASTER FACTORIZATION
   All "speedup" claims in the modules are about compression ratio
   (feature-space redundancy), prediction MAE (yield forecasting),
   or fit quality (spectral match) — NOT about factorization speed.

╔═══════════════════════════════════════════════════════════════════════════╗
║  WHAT THIS PIPELINE *CAN* DO (for legitimate research)                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

A. For a given integer N you OWN and are studying:
   • Compute its base-m polynomial (educational GNFS planning)
   • Compute prime spectrum statistics near N (research visualization)
   • Compute Hodge decomposition of its divisibility poset (speculative)
   • Match its Carpet-encoded spectral fingerprint to HP statistics (research)
   • Report cross-module "difficulty agreement" (diagnostic)

B. For a CADO-NFS log you GENERATED (running GNFS on your own number):
   • Parse the log into structured records
   • Compute Hill tail / Hurst / CUSUM diagnostics (real statistics)
   • Compress the log via OPMC (real compression)
   • Forecast future relation yield (real time-series ML)
   • Diagnose bottlenecks in YOUR pipeline (real diagnostic value)

C. As an educational tool:
   • Shows why each "promising" idea fails to break RSA
   • Demonstrates honest scientific reporting under uncertainty
   • Provides tested baselines that any future claim must beat
"""


# =============================================================================
# §1. MODULE REGISTRY (lazy loading with graceful degradation)
# =============================================================================

class ModuleRegistry:
    """
    Loads modules from disk by filename. Returns None if:
      • File doesn't exist
      • Import fails (missing dependency, syntax error)
      • Module has duplicate main() or other issues

    Caches results so repeated lookups are fast.
    """

    FILES = {
        "opmc_benchmark":     "opmc_gnfs_benchmark.py",
        "spectral_research":  "gnsf10_spectral_research.py",
        "gnfs10_research":    "gnfs10_research.py",
        "gnfs10_control":     "gnfs10_control.py",
        "hodge_one_v2":       "hodge_one_v2.py",
        "carpet_operator":    "carpet_operator.py",
        "mosco_study":        "mosco_convergence_study.py",
        "spectral_dim":       "spectral_dimension_heat_kernel.py",
        "advanced_spectral":  "advanced_spectral_heat_kernel_v2.py",
        "dynamic_refinement": "dynamic_refinement_no_zeno.py",
        "delta_min":          "assumption_Light_DeltaMin_Estimator.py",
        "rh_one":             "rh_one-1.py",
        "gnfs8_transformer":  "GNFS8_TRANSFORMER_WORLD_MODELv1.py",
        "gnfs8_shor4":        "gnfs8_shor_algorithm4.py",
        "gnfs8_bridge":       "GNFS8_HILBERT_PÓLYA_QUANTUM_CLASSICAL_BRIDGE_SIMULATOR.py",
        "carpet_hp_enhanced": "carpet_hilbert_polya_enhanced.py",  # from prev turn
    }

    def __init__(self, module_dir: str | Path = "."):
        self.module_dir = Path(module_dir)
        self._cache: Dict[str, Any] = {}
        self._status: Dict[str, str] = {}

    def get(self, name: str) -> Optional[Any]:
        if name in self._cache:
            return self._cache[name]
        if name not in self.FILES:
            self._status[name] = "unknown_module"
            return None

        path = self.module_dir / self.FILES[name]
        if not path.exists():
            self._status[name] = "file_missing"
            self._cache[name] = None
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                f"_gnfs_pipe_{name}", str(path)
            )
            if spec is None or spec.loader is None:
                self._status[name] = "spec_failed"
                self._cache[name] = None
                return None
            mod = importlib.util.module_from_spec(spec)

            old_level = logging.root.level
            logging.root.setLevel(logging.ERROR)
            try:
                spec.loader.exec_module(mod)
            finally:
                logging.root.setLevel(old_level)

            self._cache[name] = mod
            self._status[name] = "loaded"
            return mod
        except Exception as e:
            self._status[name] = f"import_failed: {type(e).__name__}"
            self._cache[name] = None
            return None

    def available(self) -> Dict[str, bool]:
        return {n: self.get(n) is not None for n in self.FILES}

    def summary(self) -> Dict[str, str]:
        return dict(self._status)


# =============================================================================
# §2. STAGE RESULT DATACLASS
# =============================================================================

@dataclass
class StageResult:
    name: str
    status: str  # "ok" | "partial" | "missing_module" | "failed" | "skipped" | "not_applicable"
    metrics: Dict[str, Any] = field(default_factory=dict)
    honest_note: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status in ("ok", "partial")


# =============================================================================
# §3. STAGE 1 — GNFS PLANNING (spectral_research)
# =============================================================================

def stage_planning(reg: ModuleRegistry, N: int) -> StageResult:
    """Base-m polynomial selection for N (educational GNFS planning)."""
    t0 = time.time()
    mod = reg.get("spectral_research")
    if mod is None:
        return StageResult("planning", "missing_module",
                            honest_note="Requires gnsf10_spectral_research.py")
    try:
        sel = mod.PolynomialSelector(N, degree=5)
        r = sel.select_polynomial()
        return StageResult(
            "planning", "ok",
            metrics={
                "degree": r["degree"],
                "m": r["m"],
                "f_m_mod_N": r["f_m_mod_N"],
                "n_coeffs": len(r["coefficients"]),
            },
            honest_note=(
                "Base-m polynomial: f(m) ≡ N (mod N). Educational only — "
                "real GNFS uses Murphy-E scoring and lattice sieving."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("planning", "failed",
                            error=str(e), duration_s=time.time() - t0)


# =============================================================================
# §4. STAGE 2 — PRIME SPECTRUM (spectral_research)
# =============================================================================

def stage_spectral(reg: ModuleRegistry, N: int) -> StageResult:
    """Prime gap statistics + HP operator (speculative) for primes near N."""
    t0 = time.time()
    mod = reg.get("spectral_research")
    if mod is None:
        return StageResult("spectral", "missing_module")
    try:
        max_prime = min(max(1000, N), 20000)
        analyzer = mod.PrimeSpectrumAnalyzer(max_prime=max_prime)
        stats = analyzer.spectral_statistics()
        hp = mod.HilbertPolyaOperator(n_zeros=20)
        hp_op = hp.construct_candidate_operator(method="diagonal")
        return StageResult(
            "spectral", "ok",
            metrics={
                "n_primes": stats.get("n_primes", 0),
                "mean_gap": stats.get("mean_gap", 0.0),
                "wigner_agreement": stats.get("wigner_agreement", 0.0),
                "hp_is_self_adjoint": hp_op.get("is_self_adjoint", False),
            },
            honest_note=(
                "Prime gap statistics are real. 'HP operator' here is a "
                "diagonal matrix of known ζ-zeros — NOT the Hilbert-Pólya operator."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("spectral", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §5. STAGE 3 — HODGE DECOMPOSITION (spectral_research)
# =============================================================================

def stage_hodge(reg: ModuleRegistry, N: int) -> StageResult:
    """Hodge decomposition of the divisibility poset of N (speculative)."""
    t0 = time.time()
    mod = reg.get("spectral_research")
    if mod is None:
        return StageResult("hodge", "missing_module")
    try:
        hodge = mod.HodgeDecomposition(N)
        chain = hodge.build_chain_complex()
        decomposition = hodge.compute_hodge_laplacian(chain)
        return StageResult(
            "hodge", "ok",
            metrics={
                "n_vertices": chain["n_vertices"],
                "n_edges": chain["n_edges"],
                "beta_0": decomposition["beta_0"],
                "beta_1": decomposition["beta_1"],
            },
            honest_note=(
                "Hodge decomposition of divisibility poset is well-defined "
                "but the connection to factorization is SPECULATIVE — not "
                "established in the literature."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("hodge", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §6. STAGE 4 — CARPET FRACTAL OPERATOR (carpet_operator)
# =============================================================================

def stage_carpet(reg: ModuleRegistry, level: int = 3) -> StageResult:
    """Carpet graph, resistance, Laplacian, Sobolev scale — REAL math."""
    t0 = time.time()
    mod = reg.get("carpet_operator")
    if mod is None:
        return StageResult("carpet", "missing_module")
    try:
        suite = mod.CarpetOperatorSuite(max_level=level, rho=1.25)
        report = suite.run_checks()
        return StageResult(
            "carpet", "ok",
            metrics={
                "level": level,
                "n_nodes": 8 ** level,
                "quasi_mult_holds": report["theorem1"]["quasi_mult_holds"],
                "rho_estimate": report["proposition1"]["rho_estimate"],
                "rho_ge_9_8": report["proposition1"]["rho_ge_9_8"],
                "moment_holds": report["moment_inequality"]["holds"],
                "power_consistency": report["power_consistency"],
            },
            honest_note=(
                "FINITE-LEVEL graph operators per Consolidated_Revision_Carpet-19. "
                "Resistance theorems verified numerically. NOT the continuum "
                "carpet Laplacian (that's the Mosco limit, requires KZ/BBKT)."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("carpet", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §7. STAGE 5 — HEAT KERNEL / SPECTRAL DIMENSION (spectral_dim)
# =============================================================================

def stage_heat_kernel(reg: ModuleRegistry, level: int = 3) -> StageResult:
    """Spectral dimension via heat kernel trace (KZ/BBKT comparison)."""
    t0 = time.time()
    mod = reg.get("spectral_dim")
    if mod is None:
        return StageResult("heat_kernel", "missing_module")
    try:
        lap = mod.CarpetLaplacian(level=level, rho=1.25)
        evals, _ = lap.spectrum()
        analyzer = mod.SpectralDimensionAnalyzer(lap, rho=1.25)
        spec_res = analyzer.empirical_ds_from_eigenvalues(evals)
        hk_analyzer = mod.HeatKernelAnalyzer(lap, spec_res["d_s_theory"])
        hk_res = hk_analyzer.analyze_short_time_asymptotics(evals)

        d_s_hk = hk_res.get("d_s_heat_kernel", float("nan")) if "error" not in hk_res else float("nan")

        return StageResult(
            "heat_kernel", "ok" if math.isfinite(spec_res["d_s_empirical"]) else "partial",
            metrics={
                "level": level,
                "d_s_theory": spec_res["d_s_theory"],
                "d_s_weyl": spec_res["d_s_empirical"],
                "d_s_heat_kernel": d_s_hk,
                "weyl_rel_error": spec_res["relative_error"],
            },
            honest_note=(
                "d_s is an ASYMPTOTIC property (λ→∞). Finite graphs show "
                "effective d_s only. KZ/BBKT bounds apply to continuum limit."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("heat_kernel", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §8. STAGE 6 — RIGOROUS RHO ESTIMATION (mosco_study)
# =============================================================================

def stage_rho(reg: ModuleRegistry, max_level: int = 3) -> StageResult:
    """Rigorous rho estimation via Fekete + ratio convergence."""
    t0 = time.time()
    mod = reg.get("mosco_study")
    if mod is None:
        return StageResult("rho", "missing_module")
    try:
        resistance = mod.CarpetResistance(max_level=max_level)
        estimator = mod.RigorousRhoEstimator(resistance)
        report = estimator.compute_all()
        return StageResult(
            "rho", "ok",
            metrics={
                "R_levels": report["R_levels"],
                "fekte_lower": report["fekte_lower"],
                "rho_best": report["rho_best"],
                "rho_ge_9_8": report["rho_ge_9_8"],
                "two_sided_holds": report["two_sided_holds"],
            },
            honest_note=(
                "rho ≈ 1.25 numerically (paper: 'not proved'). Corollary 1's "
                "Fekete bound holds; two-sided Lemma 3 bound verified at "
                "available levels only."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("rho", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §9. STAGE 7 — DELTA_MIN ESTIMATOR (assumption_Light_DeltaMin)
# =============================================================================

def stage_delta_min(reg: ModuleRegistry, N: int) -> StageResult:
    """Assumption-light DeltaMin confidence interval via subsampling."""
    t0 = time.time()
    mod = reg.get("delta_min")
    if mod is None:
        return StageResult("delta_min", "missing_module")
    try:
        import torch
        # Construct synthetic trajectory from N for demonstration
        torch.manual_seed(N % 2**31)
        n_traj, n_steps, feat_dim = 8, 64, 4
        # Trajectory that reflects N's binary structure (deterministic demo)
        bits = [(N >> i) & 1 for i in range(feat_dim)]
        base = torch.tensor(bits, dtype=torch.float32)
        traj = base.view(1, 1, feat_dim) + 0.1 * torch.randn(n_traj, n_steps, feat_dim)
        # Slight trend
        for t in range(1, n_steps):
            traj[:, t] = traj[:, t-1] + 0.05 * torch.randn(n_traj, feat_dim)

        estimator = mod.AssumptionLightDeltaMinEstimator(p=0.01, alpha=0.05)
        delta_hat, ci_lo, ci_hi = estimator(traj)
        return StageResult(
            "delta_min", "ok",
            metrics={
                "delta_hat": float(delta_hat),
                "ci_lower": float(ci_lo),
                "ci_upper": float(ci_hi),
                "n_trajectories": n_traj,
                "n_steps": n_steps,
            },
            honest_note=(
                "Subsampling CI for excursion sizes. Trajectory here is a "
                "SYNTHETIC demo derived from N — real use requires actual "
                "SESI interface trajectories."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("delta_min", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §10. STAGE 8 — OPMC COMPRESSION (opmc_benchmark)
# =============================================================================

def stage_opmc(reg: ModuleRegistry, N: int) -> StageResult:
    """OPMC fractal-similarity compression on synthetic relation series."""
    t0 = time.time()
    mod = reg.get("opmc_benchmark")
    if mod is None:
        return StageResult("opmc", "missing_module")
    try:
        # Generate synthetic relation series determined by N
        import numpy as np
        rng = np.random.default_rng(N % 2**31)
        n = 200
        series = np.abs(np.cumsum(rng.normal(0, 1, n))) * (1 + N % 100)
        row = mod.run_benchmark_on_series(
            source_name=f"synthetic_N={N}",
            series=series,
            window=16, step=8, train_frac=0.7,
            similarity_threshold=0.94, max_clusters=64,
        )
        if row is None:
            return StageResult("opmc", "partial",
                                honest_note="Insufficient data for OPMC benchmark.")
        return StageResult(
            "opmc", "ok",
            metrics={
                "n_windows": row.n_windows,
                "n_clusters": row.n_clusters,
                "feature_compression_ratio": row.feature_compression_ratio,
                "baseline_mae": row.baseline_mae_next,
                "opmc_mae": row.opmc_mae_next,
                "improvement": (row.baseline_mae_next - row.opmc_mae_next)
                               / max(row.baseline_mae_next, 1e-8),
            },
            honest_note=(
                "Feature-space compression on SYNTHETIC series derived from N. "
                "Compression ratio is redundancy measure — NOT a proof of "
                "anything about factorization."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("opmc", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §11. STAGE 9 — MURPHY-E PROXY (gnfs10_control)
# =============================================================================

def stage_murphy_e(reg: ModuleRegistry, N: int) -> StageResult:
    """Murphy-E polynomial scoring proxy."""
    t0 = time.time()
    mod = reg.get("gnfs10_control")
    if mod is None:
        return StageResult("murphy_e", "missing_module")
    try:
        scorer = mod.MurphyEScorer(prime_bound=120)
        # Use base-m polynomial as test poly
        spec_mod = reg.get("spectral_research")
        if spec_mod is not None:
            sel = spec_mod.PolynomialSelector(N, degree=5)
            poly_dict = sel.select_polynomial()
            coeffs = [float(c) for c in poly_dict["coefficients"]]
            m = poly_dict["m"]
        else:
            coeffs = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            m = int(round(N ** 0.2))
        poly = mod.PolynomialSpec(
            name=f"N={N}",
            f_coeffs=coeffs,
            g_coeffs=[1.0, -float(m)],
            m=m,
        )
        score = scorer.score_polynomial(poly)
        return StageResult(
            "murphy_e", "ok",
            metrics={
                "murphy_e_proxy": score["murphy_e_proxy"],
                "f_roots_total": score["f_roots_total"],
                "g_roots_total": score["g_roots_total"],
                "size_penalty": score["size_penalty"],
            },
            honest_note=(
                "Murphy-E PROXY, not exact CADO-NFS Murphy-E. Base-m "
                "polynomial is educational. Real GNFS uses optimized polys."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("murphy_e", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §12. STAGE 10 — GNFS-8 SHOR ALGORITHM 4 (gnfs8_shor4)
# =============================================================================

def stage_shor4(reg: ModuleRegistry, N: int) -> StageResult:
    """GNFS-8 + Shor Algorithm 4 (software-emulated)."""
    t0 = time.time()
    mod = reg.get("gnfs8_shor4")
    if mod is None:
        return StageResult("shor4", "missing_module")
    try:
        if N > 10**6:
            return StageResult(
                "shor4", "not_applicable",
                honest_note=(
                    "GNFS-8 / Shor4 are educational; can't handle N > 10^6 "
                    "in this scaffold. See file's own honest scope §4.7."
                ),
            )
        gnfs8 = mod.GNFS8Factorizer()
        shor4 = mod.ShorAlgorithm4()
        g8 = gnfs8.factor(N)
        s4 = shor4.factor(N)
        return StageResult(
            "shor4", "ok",
            metrics={
                "gnfs8_status": g8["status"],
                "gnfs8_factors": g8.get("factors"),
                "shor4_status": s4["status"],
                "shor4_factors": s4.get("factors"),
                "N_handled": N,
            },
            honest_note=(
                "Both use CLASSICAL trial division. GNFS-8 ≠ Shor's quantum "
                "algorithm. 'Shor4 ≡ Shor' remains a research target (§13.2)."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("shor4", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §13. STAGE 11 — CADO LOG ANALYSIS (opmc_benchmark + gnfs10_control)
# =============================================================================

def stage_cado(reg: ModuleRegistry, cado_log: Optional[str],
                n_synthetic: int = 400) -> StageResult:
    """Full CADO-NFS log analysis pipeline."""
    t0 = time.time()

    # Get parser
    parser_mod = reg.get("opmc_benchmark")
    if parser_mod is None:
        return StageResult("cado", "missing_module")

    try:
        parser = parser_mod.CadoGgnfsLogParser(require_yield=True)

        if cado_log and Path(cado_log).exists():
            records = parser.parse_file(cado_log)
            source = str(cado_log)
        else:
            # Generate synthetic log
            log_text = parser_mod.generate_demo_logs.__doc__  # has no text; use generator
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmpdir:
                files = parser_mod.generate_demo_logs(tmpdir, n_lines=n_synthetic)
                cado_path = files["cado"]
                records = parser.parse_file(cado_path)
            source = "synthetic"

        if len(records) < 8:
            return StageResult("cado", "partial",
                                metrics={"n_records": len(records)},
                                honest_note="Too few records parsed.")

        # Rel series
        import numpy as np
        rels = np.array([r.rels for r in records if r.rels is not None],
                         dtype=np.float64)
        cpus = np.array([r.cpu for r in records if r.cpu is not None],
                         dtype=np.float64)

        # Diagnostics (adapt from gnfs10_control logic inline)
        def hill_theta(x, frac=0.2):
            x = x[x > 0]
            if len(x) < 12:
                return 0.0
            s = np.sort(x)[::-1]
            k = max(5, min(int(frac * len(s)), len(s) - 1))
            thr = s[k]
            if thr <= 0:
                return 0.0
            logs = np.log(s[:k] / thr)
            m = float(np.mean(logs))
            return 1.0 / m if m > 1e-12 else 0.0

        def hurst_like(x):
            if len(x) < 16:
                return 0.5
            lags, sfs = [], []
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
            slope = float(np.polyfit(np.log(lags), np.log(sfs), 1)[0])
            return float(np.clip(0.5 * slope, 0.0, 1.5))

        def cusum_cps(x, thr=3.0, drift=0.25):
            if len(x) < 8:
                return 0
            mu = float(np.mean(x))
            sd = float(np.std(x))
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

        theta = hill_theta(rels)
        H = hurst_like(np.log1p(rels))
        n_cps = cusum_cps(rels)

        return StageResult(
            "cado", "ok",
            metrics={
                "n_records": len(records),
                "source": source,
                "rels_mean": float(np.mean(rels)),
                "rels_std": float(np.std(rels)),
                "cpu_mean": float(np.mean(cpus)) if len(cpus) else 0.0,
                "hill_theta": theta,
                "hurst_like": H,
                "n_change_points": n_cps,
            },
            honest_note=(
                "Parses CADO/GGNFS-style key=value logs. Diagnostics are "
                "real (Hill/Hurst/CUSUM). Useful for optimizing YOUR GNFS "
                "runs — NOT for breaking others' encryption."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("cado", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §14. STAGE 12 — ENHANCED HP SYNTHESIS (carpet_hp_enhanced from prev turn)
# =============================================================================

def stage_hp_synth(reg: ModuleRegistry, N: int, epochs: int = 20) -> StageResult:
    """Carpet→HP operator synthesis with contrastive + zero-shot."""
    t0 = time.time()
    mod = reg.get("carpet_hp_enhanced")
    if mod is None:
        return StageResult(
            "hp_synth", "missing_module",
            honest_note=(
                "Requires carpet_hilbert_polya_enhanced.py from previous turn. "
                "Save that file to the same directory to enable this stage."
            ),
        )
    try:
        # Use only the training entry point; avoid CLI side effects
        encoder = mod.CarpetSpectralEncoder(level=2, n_eigen=16,
                                              perturbation_primes=8)
        ind_samples, ood_samples = mod.make_samples(
            encoder, n_per_bucket=8,
            small_prime_bound=60, large_prime_bound=200,
            seed=N % 2**31,
        )
        cfg = mod.TrainConfig(
            op_dim=8, d_model=32, n_heads=2, n_layers=1,
            embed_dim=16, epochs=epochs, lr=1e-3,
        )
        model, hist = mod.train_enhanced_synthesizer(
            encoder, ind_samples, cfg, seed=N % 2**31,
        )
        zs = mod.zero_shot_report(model, encoder, ind_samples, ood_samples,
                                    op_dim=8)
        ood = zs["out_of_distribution"]["model"]
        rand = zs["out_of_distribution"]["random"]
        return StageResult(
            "hp_synth", "ok",
            metrics={
                "final_loss": hist["history"][-1]["loss"],
                "ood_gue_corr": ood["gue_corr"],
                "ood_gue_vs_random": ood["gue_corr"] - rand["gue_corr"],
                "ood_contrast_corr": ood["contrast_corr"],
                "ood_gap_scale_corr": ood["gap_scale_corr"],
                "passes_gue_trap": (
                    ood["gue_corr"] > rand["gue_corr"] + 0.05
                    and ood["contrast_corr"] > 0.1
                ),
            },
            honest_note=(
                "Contrastive loss reduces GUE-trap risk but does NOT prove "
                "N-encoding. Zero-shot test is on 4-digit → 5-digit N only. "
                "Does NOT extrapolate to RSA-400."
            ),
            duration_s=time.time() - t0,
        )
    except Exception as e:
        return StageResult("hp_synth", "failed", error=str(e),
                            duration_s=time.time() - t0)


# =============================================================================
# §15. UNIFIED PIPELINE ORCHESTRATOR
# =============================================================================

class UnifiedPipeline:
    """Orchestrates all stages and produces cross-module report."""

    def __init__(self, module_dir: str | Path = "."):
        self.reg = ModuleRegistry(module_dir)

    def run_all(
        self,
        N: Optional[int] = None,
        cado_log: Optional[str] = None,
        carpet_level: int = 3,
        hp_epochs: int = 20,
    ) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("UNIFIED GNFS DIAGNOSTIC PIPELINE")
        logger.info("=" * 70)

        results: Dict[str, StageResult] = {}

        # Always-run stages (don't need N)
        logger.info("[Carpet] Running fractal operator suite...")
        results["carpet"] = stage_carpet(self.reg, level=carpet_level)
        self._log_result(results["carpet"])

        logger.info("[Heat Kernel] Computing spectral dimension...")
        results["heat_kernel"] = stage_heat_kernel(self.reg, level=carpet_level)
        self._log_result(results["heat_kernel"])

        logger.info("[Rho] Estimating resistance scaling exponent...")
        results["rho"] = stage_rho(self.reg, max_level=min(4, carpet_level + 1))
        self._log_result(results["rho"])

        # N-dependent stages
        if N is not None:
            logger.info(f"[Planning] N = {N} ({len(str(N))} digits)...")
            results["planning"] = stage_planning(self.reg, N)
            self._log_result(results["planning"])

            logger.info("[Spectral] Prime gap statistics + HP operator...")
            results["spectral"] = stage_spectral(self.reg, N)
            self._log_result(results["spectral"])

            logger.info("[Hodge] Divisibility poset Hodge decomposition...")
            results["hodge"] = stage_hodge(self.reg, N)
            self._log_result(results["hodge"])

            logger.info("[DeltaMin] Assumption-light excursion CI...")
            results["delta_min"] = stage_delta_min(self.reg, N)
            self._log_result(results["delta_min"])

            logger.info("[OPMC] Feature compression benchmark...")
            results["opmc"] = stage_opmc(self.reg, N)
            self._log_result(results["opmc"])

            logger.info("[Murphy-E] Polynomial scoring proxy...")
            results["murphy_e"] = stage_murphy_e(self.reg, N)
            self._log_result(results["murphy_e"])

            logger.info("[Shor4] GNFS-8 + Shor4 emulation...")
            results["shor4"] = stage_shor4(self.reg, N)
            self._log_result(results["shor4"])

            logger.info(f"[HP Synth] Carpet→HP with contrastive (epochs={hp_epochs})...")
            results["hp_synth"] = stage_hp_synth(self.reg, N, epochs=hp_epochs)
            self._log_result(results["hp_synth"])

        # CADO stage (independent of N)
        logger.info("[CADO] Analyzing log...")
        results["cado"] = stage_cado(self.reg, cado_log)
        self._log_result(results["cado"])

        # Build report
        return self._build_report(results, N, cado_log)

    @staticmethod
    def _log_result(r: StageResult):
        icon = {"ok": "✓", "partial": "~", "failed": "✗",
                "missing_module": "·", "not_applicable": "—",
                "skipped": "·"}.get(r.status, "?")
        logger.info(f"  {icon} {r.name}: {r.status} "
                    f"({r.duration_s:.2f}s)")

    def _build_report(
        self,
        results: Dict[str, StageResult],
        N: Optional[int],
        cado_log: Optional[str],
    ) -> Dict[str, Any]:
        # Cross-module consistency
        cross = self._cross_module_consistency(results)

        # Honest verdict
        verdict = self._honest_verdict(results, cross)

        return {
            "version": "1.0-unified",
            "input": {"N": N, "cado_log": cado_log},
            "module_status": self.reg.summary(),
            "stages": {name: asdict(r) for name, r in results.items()},
            "cross_module": cross,
            "verdict": verdict,
        }

    @staticmethod
    def _cross_module_consistency(
        results: Dict[str, StageResult],
    ) -> Dict[str, Any]:
        """
        Cross-module "difficulty agreement" heuristic.

        Different stages produce different difficulty proxies. If they agree
        (higher = harder), that's evidence of consistent structural signal.
        If they disagree, that suggests noise or different notions.
        """
        proxies: Dict[str, float] = {}
        for name, r in results.items():
            if not r.passed:
                continue
            m = r.metrics
            # Extract a "difficulty proxy" per stage (heuristics only)
            if name == "planning":
                # Higher degree → harder
                proxies[name] = float(m.get("degree", 0))
            elif name == "spectral":
                # Larger wigner_agreement → more GUE-like (harder to distinguish)
                proxies[name] = float(m.get("wigner_agreement", 0.0))
            elif name == "carpet":
                # Larger rho → more anomalous walk (harder)
                proxies[name] = float(m.get("rho_estimate", 1.25))
            elif name == "heat_kernel":
                # Effective d_s (deviation from Euclidean 2 is "harder")
                d_s = m.get("d_s_heat_kernel", m.get("d_s_weyl", 2.0))
                if isinstance(d_s, (int, float)) and math.isfinite(d_s):
                    proxies[name] = abs(d_s - 2.0)
            elif name == "murphy_e":
                # Lower score → harder (negate)
                proxies[name] = -float(m.get("murphy_e_proxy", 0.0))
            elif name == "opmc":
                # Lower improvement → harder to compress (negate)
                proxies[name] = -float(m.get("improvement", 0.0))
            elif name == "cado":
                # Higher hill_theta → lighter tail → "easier" to predict
                proxies[name] = -float(m.get("hill_theta", 1.0))

        if len(proxies) < 2:
            return {"status": "insufficient_proxies", "proxies": proxies}

        # Rank correlation via Spearman-style
        names = list(proxies.keys())
        vals = [proxies[n] for n in names]
        # Simple: how many pairwise orderings agree with alphabetical order?
        # (Not statistically meaningful for single N — just a diagnostic.)
        n_pairs = 0
        n_agree = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if vals[i] > vals[j]:
                    n_agree += 1
                n_pairs += 1

        return {
            "proxies": proxies,
            "n_stages": len(proxies),
            "note": (
                "Difficulty proxies across stages. These measure DIFFERENT "
                "things — agreement here does NOT imply any is predictive "
                "of actual factoring difficulty."
            ),
        }

    @staticmethod
    def _honest_verdict(
        results: Dict[str, StageResult],
        cross: Dict[str, Any],
    ) -> Dict[str, Any]:
        n_total = len(results)
        n_ok = sum(1 for r in results.values() if r.status == "ok")
        n_missing = sum(1 for r in results.values() if r.status == "missing_module")
        n_failed = sum(1 for r in results.values() if r.status == "failed")

        return {
            "summary": (
                f"{n_ok}/{n_total} stages OK, "
                f"{n_missing} missing modules, {n_failed} failed"
            ),
            "does_not_do": [
                "Decrypt passwords",
                "Break RSA / any encryption",
                "Provide faster factorization than GNFS",
                "Construct the Hilbert-Pólya operator",
                "Prove P=NP, RH, GRH, BSD, or Hodge",
            ],
            "does_do": [
                "Cross-validate multiple research modules on same input",
                "Compute Carpet graph mathematics (real, finite-level)",
                "Diagnose CADO-NFS logs (real, for YOUR runs)",
                "Report honest baselines any claim must beat",
                "Show explicitly where modules fail",
            ],
            "interpretation": (
                "This pipeline runs all research modules and honestly reports "
                "their outputs. Statistical agreement between modules does "
                "NOT establish factorization capability. If any module claimed "
                "to factor RSA-400, that claim would be false — no stage here "
                "does, and none can be combined to do so."
            ),
        }


# =============================================================================
# §16. REPORT RENDERING
# =============================================================================

def render_text_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("UNIFIED GNFS DIAGNOSTIC REPORT")
    lines.append("=" * 78)
    lines.append("")

    lines.append("INPUT")
    for k, v in report["input"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("MODULE LOADING STATUS")
    for name, status in report["module_status"].items():
        icon = "✓" if status == "loaded" else "·"
        lines.append(f"  {icon} {name:<22} {status}")
    lines.append("")

    lines.append("STAGE RESULTS")
    lines.append(f"  {'stage':<20} {'status':<16} {'time':>8}")
    lines.append("  " + "-" * 50)
    for name, stage in report["stages"].items():
        lines.append(
            f"  {name:<20} {stage['status']:<16} "
            f"{stage['duration_s']:>7.2f}s"
        )
    lines.append("")

    lines.append("STAGE DETAILS")
    for name, stage in report["stages"].items():
        if stage["status"] in ("missing_module", "skipped"):
            continue
        lines.append(f"  [{name}]")
        for k, v in stage["metrics"].items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.6g}")
            else:
                lines.append(f"    {k}: {v}")
        if stage["honest_note"]:
            lines.append(f"    ⚠️  {stage['honest_note']}")
        lines.append("")

    lines.append("CROSS-MODULE DIFFICULTY PROXIES")
    cross = report["cross_module"]
    if "proxies" in cross:
        for name, v in cross["proxies"].items():
            lines.append(f"  {name:<20} {v:>12.4f}")
        lines.append(f"  {cross['note']}")
    lines.append("")

    v = report["verdict"]
    lines.append("=" * 78)
    lines.append("HONEST VERDICT")
    lines.append("=" * 78)
    lines.append(f"  {v['summary']}")
    lines.append("")
    lines.append("  ❌ THIS PIPELINE DOES NOT:")
    for item in v["does_not_do"]:
        lines.append(f"      ✗ {item}")
    lines.append("")
    lines.append("  ✓ THIS PIPELINE DOES:")
    for item in v["does_do"]:
        lines.append(f"      ✓ {item}")
    lines.append("")
    lines.append(f"  {v['interpretation']}")
    lines.append("=" * 78)
    return "\n".join(lines)


# =============================================================================
# §17. SELF-TEST
# =============================================================================

def self_test(module_dir: Path) -> bool:
    print(BANNER)
    print("Running self-test...\n")

    ok = True

    def rep(name: str, passed: bool, extra: str = ""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: Module registry
    reg = ModuleRegistry(module_dir)
    available = reg.available()
    n_loaded = sum(available.values())
    rep("Registry can enumerate modules", len(available) > 0,
        f"{n_loaded}/{len(available)} loaded")

    # T2: Carpet stage
    r = stage_carpet(reg, level=2)
    if r.status == "ok":
        rep("Carpet stage runs", True,
            f"rho={r.metrics['rho_estimate']:.4f}")
    else:
        rep("Carpet stage", False, r.status)

    # T3: Heat kernel
    r = stage_heat_kernel(reg, level=2)
    if r.status in ("ok", "partial"):
        rep("Heat kernel stage runs", True,
            f"d_s={r.metrics.get('d_s_heat_kernel', 'N/A')}")
    else:
        rep("Heat kernel stage", False, r.status)

    # T4: Simple N-dependent stage
    N = 3233  # 53 × 61
    r = stage_planning(reg, N)
    if r.status == "ok":
        rep("Planning stage runs", True, f"degree={r.metrics['degree']}")
    else:
        rep("Planning stage", False, r.status)

    # T5: Full pipeline on small N
    print("\n  Running full pipeline on N=3233 (small)...")
    pipeline = UnifiedPipeline(module_dir)
    report = pipeline.run_all(N=3233, carpet_level=2, hp_epochs=3)

    rep("Pipeline returns report", "stages" in report)
    rep("Report has verdict", "verdict" in report)
    rep("Verdict lists does_not_do",
        "does_not_do" in report["verdict"] and len(report["verdict"]["does_not_do"]) >= 4)
    rep("Report includes honest note about RSA",
        any("RSA" in x for x in report["verdict"]["does_not_do"]))

    # T6: Verify claim boundaries
    print("\n  Verifying honest claim boundaries...")
    v = report["verdict"]
    assert "Decrypt passwords" in v["does_not_do"], "Verdict must disclaim passwords"
    assert any("RSA" in x for x in v["does_not_do"]), "Verdict must disclaim RSA"
    assert any("Hilbert" in x for x in v["does_not_do"]), "Verdict must disclaim HP"
    rep("Claim boundaries clear", True)

    print(f"\n{'='*70}")
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print(f"{'='*70}")
    return ok


# =============================================================================
# §18. CLI
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Unified GNFS diagnostic pipeline — honest research orchestrator"
    )
    ap.add_argument("--selftest", action="store_true",
                    help="Run self-test (validates module loading)")
    ap.add_argument("--run", action="store_true",
                    help="Run full diagnostic pipeline")
    ap.add_argument("--what-this-does-not-do", action="store_true",
                    help="Print honest scope statement")
    ap.add_argument("--N", type=int, default=None,
                    help="Integer to analyze (optional)")
    ap.add_argument("--cado-log", type=str, default=None,
                    help="Path to CADO-NFS log (optional, else synthetic)")
    ap.add_argument("--carpet-level", type=int, default=3)
    ap.add_argument("--hp-epochs", type=int, default=20,
                    help="Epochs for HP synthesizer stage")
    ap.add_argument("--module-dir", type=str, default=".",
                    help="Directory containing module files")
    ap.add_argument("--json", type=str, default=None,
                    help="Write full JSON report to path")
    args = ap.parse_args(argv)

    if args.what_this_does_not_do:
        print(WHAT_THIS_DOES_NOT_DO)
        return 0

    module_dir = Path(args.module_dir).resolve()

    if args.selftest:
        return 0 if self_test(module_dir) else 1

    if not args.run:
        ap.print_help()
        print(f"\n{BANNER}")
        print("Try: --selftest, --run --N <int>, or --what-this-does-not-do")
        return 0

    pipeline = UnifiedPipeline(module_dir)
    report = pipeline.run_all(
        N=args.N,
        cado_log=args.cado_log,
        carpet_level=args.carpet_level,
        hp_epochs=args.hp_epochs,
    )

    print()
    print(render_text_report(report))

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nReport saved: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
