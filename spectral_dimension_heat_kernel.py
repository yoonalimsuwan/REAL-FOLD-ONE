"""
=============================================================================
SPECTRAL DIMENSION & HEAT KERNEL ANALYSIS (KZ/BBKT COMPARISON)
Per Consolidated_Revision_Carpet-19: Section 9, Kusuoka-Zhou (1992),
Barlow-Bass (1999), BBKT (2010).
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS FILE COMPUTES
-----------------------
[M3] Spectral Dimension (d_s) Estimation:
      • Theoretical d_s from the resistance scaling factor rho:
        d_s = 2 * ln(8) / ln(8 * rho)
      • Empirical d_s from Fractal Weyl's Law: lambda_j ~ j^{2/d_s}
      • Empirical d_s from Heat Kernel Trace: Z(t) = sum exp(-t lambda_j) ~ t^{-d_s/2}

[M4] Comparison with KZ/BBKT Heat Kernel Bounds:
      • Verifies if the finite-level graph exhibits the crossover regime
        where the trace scales as t^{-d_s/2}, consistent with the
        continuum bounds proved by Barlow-Bass and Kusuoka-Zhou.

HONEST SCOPE (Crucial)
----------------------
• FINITE-SIZE LIMITATION: d_s is an asymptotic property (lambda -> infinity, t -> 0).
  On finite pre-carpet graphs, the spectrum is bounded, so the true asymptotic
  regime is unreachable. We observe an "effective" dimension in the intermediate
  crossover regime.
• RHO IS NUMERICAL: The theoretical d_s depends on rho, which is only known
  numerically (rho ~ 1.25). Thus, the comparison is between two numerical
  estimates, not a proven exact constant.
• CONTINUUM VS DISCRETE: The KZ/BBKT bounds apply to the continuum Mosco limit
  operator. The finite-level graph only approximates this in a limited window.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Tuple

import torch
import numpy as np
from scipy import stats

logger = logging.getLogger("sesi.spectral_dim")

__all__ = [
    "CarpetGraph",
    "CarpetLaplacian",
    "SpectralDimensionAnalyzer",
    "HeatKernelAnalyzer",
]


# ═══════════════════════════════════════════════════════════════════════
# CARPET GRAPH & LAPLACIAN (Inlined for standalone execution)
# ═══════════════════════════════════════════════════════════════════════

class CarpetGraph:
    def __init__(self, level: int):
        self.level = level
        self.cells = self._build_cells(level)
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.n_vertices = len(self.cells)
        self.edges = self._build_edges()

    def _build_cells(self, level: int):
        side = 3 ** level
        kept = []
        for x in range(side):
            for y in range(side):
                if self._survives(x, y, level):
                    kept.append((x, y))
        return sorted(kept)

    @staticmethod
    def _survives(x: int, y: int, level: int) -> bool:
        for _ in range(level):
            if (x % 3 == 1) and (y % 3 == 1):
                return False
            x //= 3
            y //= 3
        return True

    def _build_edges(self):
        vset = set(self.cells)
        edges = []
        for (x, y) in self.cells:
            for dx, dy in ((1, 0), (0, 1)):
                nb = (x + dx, y + dy)
                if nb in vset:
                    edges.append((self.index[(x, y)], self.index[nb]))
        return edges

    def graph_laplacian(self) -> torch.Tensor:
        A = torch.zeros(self.n_vertices, self.n_vertices)
        for i, j in self.edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
        deg = A.sum(dim=1)
        return torch.diag(deg) - A


class CarpetLaplacian:
    def __init__(self, level: int, rho: float = 1.25):
        self.level = level
        self.rho = rho
        self.graph = CarpetGraph(level)
        self.L = self.graph.graph_laplacian()
        self.gauge = rho ** level
        self.A = self.gauge * self.L
        self._evals = None
        self._evecs = None

    def spectrum(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._evals is None:
            evals, evecs = torch.linalg.eigh(self.A)
            self._evals = evals
            self._evecs = evecs
        return self._evals, self._evecs


# ═══════════════════════════════════════════════════════════════════════
# [M3] SPECTRAL DIMENSION ANALYZER (Fractal Weyl's Law)
# ═══════════════════════════════════════════════════════════════════════

class SpectralDimensionAnalyzer:
    """
    Estimates the spectral dimension d_s from the eigenvalue growth.
    
    Theory (KZ/BBKT):
      d_f = ln(8) / ln(3)
      d_w = ln(8 * rho) / ln(3)
      d_s = 2 * d_f / d_w = 2 * ln(8) / ln(8 * rho)
      
    Weyl's Law on Fractals:
      N(lambda) ~ C * lambda^{d_s / 2}  =>  lambda_j ~ C' * j^{2 / d_s}
    """

    def __init__(self, laplacian: CarpetLaplacian, rho: float):
        self.laplacian = laplacian
        self.rho = rho
        self.d_f = math.log(8) / math.log(3)
        self.d_w = math.log(8 * rho) / math.log(3)
        self.d_s_theory = 2 * self.d_f / self.d_w

    def empirical_ds_from_eigenvalues(self, evals: torch.Tensor) -> Dict[str, float]:
        """
        Fit log(lambda_j) = (2/d_s) * log(j) + const
        """
        # Filter out the zero eigenvalue and numerical noise
        evals_np = evals[evals > 1e-8].numpy()
        n = len(evals_np)
        
        if n < 10:
            return {"error": "Not enough eigenvalues for fit"}

        # Use the bulk of the spectrum (avoiding the very first few and the very last)
        start = max(1, n // 10)
        end = min(n, 9 * n // 10)
        
        evals_subset = evals_np[start:end]
        j = np.arange(start, end)
        
        log_j = np.log(j)
        log_evals = np.log(evals_subset)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_j, log_evals)
        
        # slope = 2 / d_s  =>  d_s = 2 / slope
        d_s_empirical = 2.0 / slope if slope > 0 else float('inf')
        
        return {
            "d_s_empirical": d_s_empirical,
            "d_s_theory": self.d_s_theory,
            "slope": slope,
            "r_squared": r_value**2,
            "relative_error": abs(d_s_empirical - self.d_s_theory) / self.d_s_theory,
            "n_modes_used": end - start
        }


# ═══════════════════════════════════════════════════════════════════════
# [M4] HEAT KERNEL ANALYZER (KZ/BBKT Bounds Comparison)
# ═══════════════════════════════════════════════════════════════════════

class HeatKernelAnalyzer:
    """
    Computes the heat kernel trace Z(t) = sum exp(-t lambda_j) and
    extracts the short-time asymptotic exponent to compare with KZ/BBKT.
    
    KZ/BBKT Theorem:
      For the continuum carpet, p_t(x,x) ~ t^{-d_s/2}.
      Therefore, the trace Z(t) = int p_t(x,x) dmu ~ t^{-d_s/2}.
    """

    def __init__(self, laplacian: CarpetLaplacian, d_s_theory: float):
        self.laplacian = laplacian
        self.d_s_theory = d_s_theory

    def compute_trace(self, evals: torch.Tensor, t_values: np.ndarray) -> np.ndarray:
        """Z(t) = sum_j exp(-t * lambda_j)"""
        evals_np = evals.numpy()
        Z = np.zeros_like(t_values, dtype=float)
        for i, t in enumerate(t_values):
            Z[i] = np.sum(np.exp(-t * evals_np))
        return Z

    def analyze_short_time_asymptotics(self, evals: torch.Tensor) -> Dict[str, object]:
        """
        Fit log Z(t) = - (d_s / 2) * log t + const for small t.
        """
        lambda_max = evals.max().item()
        N = len(evals)
        
        # t range: from somewhat below 1 to well above 1/lambda_max
        t_min = 10.0 / lambda_max
        t_max = 1.0
        t_values = np.logspace(np.log10(t_min), np.log10(t_max), 50)
        
        Z = self.compute_trace(evals, t_values)
        
        # Filter out t where Z(t) is too close to N (finite size saturation)
        # We want the regime where Z(t) is decaying, not saturated at N.
        valid_mask = Z < 0.5 * N
        t_valid = t_values[valid_mask]
        Z_valid = Z[valid_mask]
        
        if len(t_valid) < 3:
            return {"error": "Not enough valid t values for asymptotic fit (finite-size saturation)"}
            
        log_t = np.log(t_valid)
        log_Z = np.log(Z_valid)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_Z)
        
        # slope = - d_s / 2  =>  d_s = -2 * slope
        d_s_heat_kernel = -2.0 * slope
        
        return {
            "d_s_heat_kernel": d_s_heat_kernel,
            "d_s_theory": self.d_s_theory,
            "slope": slope,
            "r_squared": r_value**2,
            "relative_error": abs(d_s_heat_kernel - self.d_s_theory) / self.d_s_theory,
            "t_range": (float(t_valid.min()), float(t_valid.max())),
            "Z_range": (float(Z_valid.min()), float(Z_valid.max()))
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Spectral Dimension & Heat Kernel Analysis (KZ/BBKT)")
    print("=" * 72)
    torch.manual_seed(0)

    # Use level 3 (512 vertices) or level 4 (4096 vertices) if memory permits
    level = 3
    rho = 1.25  # Numerical estimate for the standard carpet
    
    print(f"\n  Building level-{level} carpet graph...")
    lap = CarpetLaplacian(level=level, rho=rho)
    evals, _ = lap.spectrum()
    print(f"  Graph vertices: {lap.graph.n_vertices}")
    print(f"  Eigenvalues computed: {len(evals)}")

    # ── [M3] Spectral Dimension from Weyl's Law ─────────────────────
    print(f"\n  ── [M3] Spectral Dimension (Fractal Weyl's Law) ──")
    spec_analyzer = SpectralDimensionAnalyzer(lap, rho)
    spec_res = spec_analyzer.empirical_ds_from_eigenvalues(evals)
    
    print(f"    Theoretical d_s (from rho={rho}): {spec_res['d_s_theory']:.4f}")
    print(f"    Empirical d_s (from lambda_j)   : {spec_res['d_s_empirical']:.4f}")
    print(f"    Relative error                  : {spec_res['relative_error']*100:.2f}%")
    print(f"    R^2 of log-log fit              : {spec_res['r_squared']:.4f}")
    print(f"    Modes used                      : {spec_res['n_modes_used']}")

    # ── [M4] Heat Kernel Trace & KZ/BBKT Bounds ─────────────────────
    print(f"\n  ── [M4] Heat Kernel Trace (KZ/BBKT Bounds) ──")
    hk_analyzer = HeatKernelAnalyzer(lap, spec_res['d_s_theory'])
    hk_res = hk_analyzer.analyze_short_time_asymptotics(evals)
    
    if "error" in hk_res:
        print(f"    [WARN] {hk_res['error']}")
    else:
        print(f"    Theoretical d_s                 : {hk_res['d_s_theory']:.4f}")
        print(f"    Empirical d_s (from Z(t))       : {hk_res['d_s_heat_kernel']:.4f}")
        print(f"    Relative error                  : {hk_res['relative_error']*100:.2f}%")
        print(f"    R^2 of log-log fit              : {hk_res['r_squared']:.4f}")
        print(f"    Valid t range                   : [{hk_res['t_range'][0]:.4e}, {hk_res['t_range'][1]:.4e}]")
        print(f"    Valid Z(t) range                : [{hk_res['Z_range'][0]:.2f}, {hk_res['Z_range'][1]:.2f}]")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE & LIMITATIONS")
    print("  • FINITE-SIZE EFFECTS: d_s is an asymptotic property (lambda -> infinity).")
    print("    On finite graphs, the spectrum is bounded, so the true t->0 regime")
    print("    is unreachable. We observe an 'effective' dimension in the crossover.")
    print("  • RHO IS NUMERICAL: The theoretical d_s depends on rho ~ 1.25, which")
    print("    is a numerical estimate, not a proven exact constant.")
    print("  • CONTINUUM VS DISCRETE: KZ/BBKT bounds apply to the Mosco limit.")
    print("    The finite graph only approximates this in a limited window.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
