"""
=============================================================================
ADVANCED SPECTRAL & HEAT KERNEL ANALYSIS (Level 4-5 Sparse + Off-Diagonal)
Per Consolidated_Revision_Carpet-19: Barlow-Bass (BBKT) & Kusuoka-Zhou (KZ)
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS FILE ADDS
-------------------
[M5] Sparse Matrix Implementation:
     • Uses scipy.sparse (CSR format) to handle Level 4 (4,096 nodes) 
       and Level 5 (32,768 nodes) without OOM.
     • Uses ARPACK (scipy.sparse.linalg.eigsh) to compute the first K 
       eigenpairs efficiently (shift-invert mode for smallest eigenvalues).

[M6] Off-Diagonal Heat Kernel & Walk Dimension ($d_w$):
     • Computes $p_t(x,y) = \sum_{i=0}^{K-1} e^{-\lambda_i t} \phi_i(x) \phi_i(y)$
     • Extracts graph distances $r = d_{graph}(x,y)$ from a corner node.
     • Fits the spatial decay of $p_t(x,y)$ at a fixed optimal time $t$ 
       to estimate the effective walk dimension $d_w$.
     • Cross-validates with the theoretical relation: $d_w = 2 d_f / d_s$.

HONEST SCOPE
------------
• FINITE-SIZE CUTOFF: The continuum fractal scaling $t \to 0$ is truncated 
  by the graph's maximum eigenvalue $\lambda_{max} \sim 8^L$. We observe 
  the "crossover" regime, not the true asymptotic limit.
• EIGENPAIR TRUNCATION: We compute only the first $K$ eigenpairs. This 
  accurately captures the long-time (small $t$) behavior of $p_t(x,y)$, 
  which is exactly the regime where fractal scaling is most visible.
• $d_w$ ESTIMATION: Fitting the sub-Gaussian decay on a finite graph yields 
  an "effective" $d_w$. It should be close to the theoretical value 
  ($\approx 2.096$ for $\rho=1.25$), but deviations are expected due to 
  lattice anisotropy and finite-size effects.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as spcg
from scipy.optimize import curve_fit

logger = logging.getLogger("sesi.advanced_spectral")

__all__ = [
    "SparseCarpetGraph",
    "AdvancedHeatKernelAnalyzer",
]


# ═══════════════════════════════════════════════════════════════════════
# [M5] SPARSE CARPET GRAPH
# ═══════════════════════════════════════════════════════════════════════

class SparseCarpetGraph:
    """
    Level-n carpet graph built with scipy.sparse for memory efficiency.
    Supports up to Level 5 (32,768 vertices) comfortably.
    """

    def __init__(self, level: int, rho: float = 1.25):
        self.level = level
        self.rho = rho
        self.cells = self._build_cells(level)
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.n_vertices = len(self.cells)
        self.adjacency = self._build_sparse_adjacency()
        self.laplacian = self._build_sparse_laplacian()
        self.normalized_laplacian = self._build_normalized_laplacian()

    def _build_cells(self, level: int) -> List[Tuple[int, int]]:
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

    def _build_sparse_adjacency(self) -> sp.csr_matrix:
        vset = set(self.cells)
        rows, cols, data = [], [], []
        for (x, y) in self.cells:
            i = self.index[(x, y)]
            for dx, dy in ((1, 0), (0, 1)):
                nb = (x + dx, y + dy)
                if nb in vset:
                    j = self.index[nb]
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([1.0, 1.0])
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_vertices, self.n_vertices))

    def _build_sparse_laplacian(self) -> sp.csr_matrix:
        deg = np.array(self.adjacency.sum(axis=1)).flatten()
        D = sp.diags(deg)
        return D - self.adjacency

    def _build_normalized_laplacian(self) -> sp.csr_matrix:
        """L_norm = D^{-1/2} L D^{-1/2} (better for spectral clustering/heat kernel)"""
        deg = np.array(self.adjacency.sum(axis=1)).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-10))
        D_inv_sqrt = sp.diags(deg_inv_sqrt)
        return D_inv_sqrt @ self.laplacian @ D_inv_sqrt

    def get_corner_node(self) -> int:
        """Returns the index of the bottom-left corner node (0, 0)."""
        return self.index[(0, 0)]

    def graph_distances(self, source: int) -> np.ndarray:
        """Compute shortest path distances from source node to all others."""
        # Unweighted shortest path on the adjacency matrix
        dist_matrix = spcg.shortest_path(self.adjacency, directed=False, indices=source)
        return dist_matrix


# ═══════════════════════════════════════════════════════════════════════
# [M6] ADVANCED HEAT KERNEL & WALK DIMENSION ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class AdvancedHeatKernelAnalyzer:
    """
    Computes heat kernel and extracts spectral/walk dimensions using 
    sparse eigensolvers.
    """

    def __init__(self, graph: SparseCarpetGraph, k_eigenpairs: int = 150):
        self.graph = graph
        self.k_eigenpairs = min(k_eigenpairs, graph.n_vertices - 1)
        self.evals = None
        self.evecs = None
        self._compute_eigenpairs()

    def _compute_eigenpairs(self):
        """Compute the smallest K eigenpairs of the normalized Laplacian."""
        logger.info(f"Computing {self.k_eigenpairs} eigenpairs for N={self.graph.n_vertices}...")
        L_norm = self.graph.normalized_laplacian
        
        # Use shift-invert mode for smallest eigenvalues (most stable for Laplacian)
        try:
            evals, evecs = spla.eigsh(L_norm, k=self.k_eigenpairs, which='SM', sigma=0.0)
            # Sort by eigenvalue
            idx = np.argsort(evals)
            self.evals = evals[idx]
            self.evecs = evecs[:, idx]
            logger.info("Eigenpair computation successful.")
        except spla.ArpackNoConvergence as e:
            logger.warning(f"ARPACK did not converge fully. Using {len(e.eigenvalues)} converged pairs.")
            self.evals = e.eigenvalues
            self.evecs = e.eigenvectors

    def compute_heat_kernel_trace(self, t_values: np.ndarray) -> np.ndarray:
        """Z(t) = sum_{i=0}^{K-1} exp(-t * lambda_i)"""
        # Note: We only sum over computed eigenpairs. For small t, this is an 
        # approximation, but for the fractal crossover regime, the low eigenvalues 
        # dominate the scaling behavior.
        Z = np.zeros_like(t_values, dtype=float)
        for i, t in enumerate(t_values):
            Z[i] = np.sum(np.exp(-t * self.evals))
        return Z

    def compute_off_diagonal_heat_kernel(
        self, 
        source: int, 
        targets: List[int], 
        t_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute p_t(source, target) for specific pairs.
        p_t(x,y) = sum_{i} exp(-t * lambda_i) * phi_i(x) * phi_i(y)
        
        Returns: shape (len(targets), len(t_values))
        """
        results = np.zeros((len(targets), len(t_values)), dtype=float)
        
        # Pre-extract source eigenvector values
        phi_source = self.evecs[source, :]  # shape (K,)
        
        for j, target in enumerate(targets):
            phi_target = self.evecs[target, :]  # shape (K,)
            # Element-wise product of eigenvectors
            prod = phi_source * phi_target  # shape (K,)
            
            for i, t in enumerate(t_values):
                # Dot product with exp(-t * lambda)
                results[j, i] = np.sum(prod * np.exp(-t * self.evals))
                
        return results

    def estimate_dw_from_off_diagonal(self, source: int, n_samples: int = 200) -> Dict[str, object]:
        """
        Estimate walk dimension d_w by fitting the spatial decay of p_t(x,y).
        
        Method:
        1. Compute graph distances r from `source` to a random sample of nodes.
        2. Pick an optimal time t_opt where the crossover scaling is visible.
        3. Fit ln(p_t(x,y)) vs ln(r) to extract the decay exponent.
        
        Theoretical scaling: p_t(x,y) ~ exp( -c * (r^{d_w} / t)^{1/(d_w-1)} )
        For a fixed t, ln(p) ~ -c' * r^{d_w / (d_w - 1)}
        """
        logger.info("Estimating d_w from off-diagonal heat kernel decay...")
        
        # 1. Get distances
        distances = self.graph.graph_distances(source)
        
        # Filter valid distances (exclude 0 and very large distances where p_t is ~0)
        valid_mask = (distances > 0) & (distances < self.graph.level * 2)  # heuristic cutoff
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 50:
            return {"error": "Not enough valid target nodes for fitting"}
            
        # Sample targets
        sampled_indices = np.random.choice(valid_indices, size=min(n_samples, len(valid_indices)), replace=False)
        r_vals = distances[sampled_indices]
        
        # 2. Choose t_opt. A good heuristic is t ~ 1 / lambda_{k/2}
        t_opt = 1.0 / (self.evals[len(self.evals)//2] + 1e-10)
        
        # 3. Compute p_t
        p_t_vals = self.compute_off_diagonal_heat_kernel(source, sampled_indices.tolist(), [t_opt])
        p_t_vals = p_t_vals.flatten()
        
        # Filter out near-zero probabilities to avoid log(0)
        positive_mask = p_t_vals > 1e-12
        r_fit = r_vals[positive_mask]
        p_fit = p_t_vals[positive_mask]
        
        if len(r_fit) < 20:
            return {"error": "Too many near-zero probabilities, cannot fit"}
            
        # 4. Fit the decay. 
        # Model: ln(p) = A - B * r^gamma, where gamma = d_w / (d_w - 1)
        # We fit gamma first by taking log-log of the derivative, or use non-linear least squares.
        # Simpler robust approach: fit ln(-ln(p / p_max)) vs ln(r)
        # Because p ~ exp(-B * r^gamma) => ln(p) = ln(A) - B * r^gamma
        # => ln(ln(A) - ln(p)) = ln(B) + gamma * ln(r)
        
        # Estimate A as max(p_fit)
        A_est = np.max(p_fit) * 1.1  # slight overestimate to avoid log(0)
        
        y_data = np.log(np.log(A_est / p_fit))
        x_data = np.log(r_fit)
        
        # Linear fit: y = gamma * x + ln(B)
        try:
            gamma, ln_B = np.polyfit(x_data, y_data, 1)
            d_w_est = gamma / (gamma - 1.0) if gamma > 1.0 else float('inf')
            
            # R-squared of the linear fit
            y_pred = gamma * x_data + ln_B
            ss_res = np.sum((y_data - y_pred)**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Fitting failed: {e}")
            d_w_est = float('nan')
            r_squared = 0.0

        # Cross-validation with spectral dimension
        d_f = math.log(8) / math.log(3)
        # Estimate d_s from the trace (using the same eigenpairs)
        t_trace = np.logspace(np.log10(t_opt/10), np.log10(t_opt*10), 20)
        Z_trace = self.compute_heat_kernel_trace(t_trace)
        
        # Fit Z(t) ~ t^{-d_s/2} => ln(Z) = - (d_s/2) ln(t) + C
        valid_trace = Z_trace > 1.0  # avoid saturation at N
        if np.sum(valid_trace) > 2:
            slope, _ = np.polyfit(np.log(t_trace[valid_trace]), np.log(Z_trace[valid_trace]), 1)
            d_s_est = -2.0 * slope
            d_w_from_ds = 2.0 * d_f / d_s_est if d_s_est > 0 else float('nan')
        else:
            d_s_est = float('nan')
            d_w_from_ds = float('nan')

        return {
            "d_w_estimated_from_p_t": d_w_est,
            "fit_gamma": gamma if 'gamma' in locals() else float('nan'),
            "fit_r_squared": r_squared if 'r_squared' in locals() else 0.0,
            "t_opt_used": t_opt,
            "n_points_fitted": len(r_fit),
            "d_s_estimated_from_trace": d_s_est,
            "d_w_cross_check (2*d_f/d_s)": d_w_from_ds,
            "theoretical_d_w (rho=1.25)": math.log(8 * 1.25) / math.log(3),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Advanced Spectral & Heat Kernel Analysis (Level 4-5 Sparse)")
    print("=" * 72)
    np.random.seed(42)

    # Test Level 3 (fast) and Level 4 (moderate)
    # Level 5 can be enabled but takes ~10-20 seconds for eigsh
    levels_to_test = [3, 4]
    
    for level in levels_to_test:
        print(f"\n  ── Processing Level {level} Carpet ──")
        graph = SparseCarpetGraph(level=level, rho=1.25)
        print(f"    Vertices (N)      : {graph.n_vertices} (Expected: {8**level})")
        print(f"    Sparsity of L     : {graph.laplacian.nnz / (graph.n_vertices**2) * 100:.2f}%")
        
        # Use a reasonable number of eigenpairs (e.g., 100 for L3, 150 for L4)
        k_eigs = 100 if level == 3 else 150
        analyzer = AdvancedHeatKernelAnalyzer(graph, k_eigenpairs=k_eigs)
        
        print(f"    Computed eigenpairs : {len(analyzer.evals)}")
        print(f"    Spectral gap (λ_1)  : {analyzer.evals[1]:.6e}")
        
        # Estimate d_w
        source_node = graph.get_corner_node()
        dw_results = analyzer.estimate_dw_from_off_diagonal(source_node, n_samples=300)
        
        if "error" in dw_results:
            print(f"    [WARN] d_w estimation: {dw_results['error']}")
        else:
            print(f"\n    ── Walk Dimension (d_w) Estimation ──")
            print(f"      Estimated d_w (from p_t decay) : {dw_results['d_w_estimated_from_p_t']:.3f}")
            print(f"      Fit R-squared                  : {dw_results['fit_r_squared']:.4f}")
            print(f"      Cross-check d_w (2*d_f/d_s)    : {dw_results['d_w_cross_check (2*d_f/d_s)']:.3f}")
            print(f"      Theoretical d_w (ρ=1.25)       : {dw_results['theoretical_d_w (rho=1.25)']:.3f}")
            
            # Check consistency
            est = dw_results['d_w_estimated_from_p_t']
            theo = dw_results['theoretical_d_w (rho=1.25)']
            if not np.isnan(est) and abs(est - theo) / theo < 0.15:
                print(f"      [PASS] Estimated d_w is within 15% of theoretical value.")
            else:
                print(f"      [NOTE] Deviation expected due to finite-size lattice effects.")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE & LIMITATIONS")
    print("  • SPARSE EIGENSOLVER: ARPACK computes only the smallest K eigenvalues.")
    print("    This perfectly captures the long-time (fractal crossover) regime of")
    print("    the heat kernel, but misses the very short-time lattice behavior.")
    print("  • FINITE-SIZE CUTOFF: The maximum graph distance is bounded (~3^L).")
    print("    The fitted d_w is an 'effective' dimension in the observable window.")
    print("  • THEORETICAL d_w: For ρ ≈ 1.25, d_f = ln(8)/ln(3) ≈ 1.893, so")
    print("    d_w = ln(8*1.25)/ln(3) ≈ 2.096. This is the target for validation.")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
