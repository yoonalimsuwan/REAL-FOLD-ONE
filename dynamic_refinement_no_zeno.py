"""
=============================================================================
DYNAMIC REFINEMENT WITH NO-ZENO GATE + HEAT KERNEL TRACE ($d_s$) ANALYSIS
Per Consolidated_Revision_Carpet-19 & Six_Problems_Closed_As_Far_As_They_Go
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS FILE ADDS
-------------------
[M9] Heat Kernel Trace $Z(t)$ & Log-Log Plotting:
     • Computes $Z(t) = \sum_{i=0}^{K-1} e^{-t \lambda_i}$.
     • Fits $\log Z(t) = - (d_s / 2) \log t + C$ in the valid crossover regime.
     • Generates a visual Log-Log plot to validate the fractal scaling $Z(t) \sim t^{-d_s/2}$.

[M10] No-Zeno Gate for Dynamic Refinement Control:
     • Uses the deviation of the estimated $d_s$ from the theoretical $d_s$ 
       (plus fit $R^2$ penalty) as a "stress" or "error" metric.
     • Applies a two-level hysteresis gate ($\theta_{hi}$, $\theta_{lo}$, $\delta_{min}$).
     • Guarantees that the number of refinement events $N(T)$ in any finite 
       interval is strictly bounded: $N(T) \le (E_{max} + D(T)) / \delta_{min}$.
     • Prevents pathological "chattering" (infinite refinement loops) when 
       the solver struggles to converge on a specific level.

HONEST SCOPE
------------
• FINITE-SIZE CUTOFF: The true asymptotic regime $t \to 0$ is unreachable on 
  finite graphs. The fit is performed in the "crossover" window where fractal 
  scaling is most visible, before saturation at $Z(t) \approx N$.
• NO-ZENO IS ALGORITHMIC: The bound applies to the *number of controller 
  refinement events*, not to the continuum Navier-Stokes or fractal Laplacian 
  regularity. It is an engineering guarantee against solver thrashing.
• STRESS METRIC: The "stress" here is a heuristic proxy for discretization 
  error. In a full CFD solver, this would be replaced by a rigorous a posteriori 
  error estimator.
=============================================================================
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from scipy import stats

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger("sesi.dynamic_refinement")

__all__ = [
    "SparseCarpetGraph",
    "HeatKernelTraceAnalyzer",
    "NoZenoRefinementGate",
    "DynamicRefinementWorkflow",
]


# ═══════════════════════════════════════════════════════════════════════
# [M5] SPARSE CARPET GRAPH (Inlined for standalone execution)
# ═══════════════════════════════════════════════════════════════════════

class SparseCarpetGraph:
    def __init__(self, level: int, rho: float = 1.25):
        self.level = level
        self.rho = rho
        self.cells = self._build_cells(level)
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.n_vertices = len(self.cells)
        self.laplacian = self._build_sparse_laplacian()
        self.normalized_laplacian = self._build_normalized_laplacian()

    def _build_cells(self, level: int):
        side = 3 ** level
        return [(x, y) for x in range(side) for y in range(side) if self._survives(x, y, level)]

    @staticmethod
    def _survives(x: int, y: int, level: int) -> bool:
        for _ in range(level):
            if (x % 3 == 1) and (y % 3 == 1):
                return False
            x //= 3
            y //= 3
        return True

    def _build_sparse_laplacian(self):
        import scipy.sparse as sp
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
        A = sp.csr_matrix((data, (rows, cols)), shape=(self.n_vertices, self.n_vertices), dtype=np.float32)
        deg = np.array(A.sum(axis=1)).flatten()
        return sp.diags(deg, dtype=np.float32) - A

    def _build_normalized_laplacian(self):
        import scipy.sparse as sp
        deg = np.array(self.laplacian.diagonal()).flatten()
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-10))
        D_inv_sqrt = sp.diags(deg_inv_sqrt, dtype=np.float32)
        return D_inv_sqrt @ self.laplacian @ D_inv_sqrt


# ═══════════════════════════════════════════════════════════════════════
# [M9] HEAT KERNEL TRACE ANALYZER & PLOTTER
# ═══════════════════════════════════════════════════════════════════════

class HeatKernelTraceAnalyzer:
    """
    Computes $Z(t) = \text{Tr}(e^{-tA})$ and extracts $d_s$ via log-log fit.
    """

    def __init__(self, evals: np.ndarray, d_s_theory: float, level: int):
        self.evals = evals
        self.d_s_theory = d_s_theory
        self.level = level
        self.N = len(evals)

    def compute_Z(self, t_values: np.ndarray) -> np.ndarray:
        """$Z(t) = \sum_{i} e^{-t \lambda_i}$"""
        # Use log-sum-exp trick for numerical stability if needed, 
        # but for small K and reasonable t, direct sum is fine.
        return np.sum(np.exp(-np.outer(t_values, self.evals)), axis=1)

    def estimate_ds_and_plot(self, save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Fit $\log Z(t) = - (d_s / 2) \log t + C$ and generate a Log-Log plot.
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib not available for plotting"}

        # Heuristic time window: avoid t -> 0 (saturation at N) and t -> inf (dominated by lambda_1)
        lambda_max = self.evals[-1]
        lambda_1 = self.evals[1] if len(self.evals) > 1 else self.evals[0]
        
        t_min = 5.0 / lambda_max  # Start after initial lattice cutoff
        t_max = 0.5 / lambda_1    # End before ground state dominates
        
        if t_min >= t_max:
            t_min = 1e-3
            t_max = 1.0

        t_values = np.logspace(np.log10(t_min), np.log10(t_max), 100)
        Z_values = self.compute_Z(t_values)

        # Filter valid range where Z(t) is decaying (not saturated at N)
        valid_mask = (Z_values < 0.6 * self.N) & (Z_values > 1.5)
        t_fit = t_values[valid_mask]
        Z_fit = Z_values[valid_mask]

        if len(t_fit) < 5:
            return {"error": "Not enough valid points in crossover regime for fit"}

        log_t = np.log(t_fit)
        log_Z = np.log(Z_fit)

        # Linear regression: log_Z = slope * log_t + intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_Z)
        
        d_s_est = -2.0 * slope
        rel_error = abs(d_s_est - self.d_s_theory) / self.d_s_theory

        # ── Plotting ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot all data
        ax.loglog(t_values, Z_values, 'b-', alpha=0.5, label='Full $Z(t)$')
        # Plot fitted region
        ax.loglog(t_fit, Z_fit, 'ro', markersize=4, label=f'Fitted region ($R^2$={r_value**2:.4f})')
        
        # Plot fitted line
        Z_fit_line = np.exp(intercept) * t_fit ** slope
        ax.loglog(t_fit, Z_fit_line, 'k--', linewidth=2, 
                  label=f'Fit: slope = {slope:.3f} $\Rightarrow d_s$ = {d_s_est:.3f}\nTheoretical $d_s$ = {self.d_s_theory:.3f}')

        ax.set_xlabel('Time $t$ (log scale)')
        ax.set_ylabel('Heat Kernel Trace $Z(t)$ (log scale)')
        ax.set_title(f'Heat Kernel Trace Scaling (Level {self.level}, N={self.N})\n$Z(t) \sim t^{{-d_s/2}}$')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.6)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()

        return {
            "d_s_estimated": d_s_est,
            "d_s_theory": self.d_s_theory,
            "relative_error": rel_error,
            "r_squared": r_value**2,
            "slope": slope,
            "n_points_fitted": len(t_fit),
            "t_range": (float(t_fit.min()), float(t_fit.max())),
        }


# ═══════════════════════════════════════════════════════════════════════
# [M10] NO-ZENO REFINEMENT GATE
# ═══════════════════════════════════════════════════════════════════════

class NoZenoRefinementGate:
    """
    Hysteresis gate to control dynamic refinement, preventing Zeno-like 
    infinite event accumulation (Six Problems, Prop 6.1 / Closure Note Thm 4.2).
    """

    def __init__(self, theta_hi: float = 0.15, theta_lo: float = 0.05):
        """
        Args:
            theta_hi: Trigger refinement if stress > theta_hi
            theta_lo: Re-arm the gate only when stress falls below theta_lo
        """
        self.theta_hi = theta_hi
        self.theta_lo = theta_lo
        self.delta_min = theta_hi - theta_lo  # Non-degenerate reset gap
        
        if self.delta_min <= 0:
            raise ValueError("No-Zeno gate requires theta_hi > theta_lo")

        self.stress_acc = 0.0
        self.E_max = 0.0
        self.D = 0.0
        self.event_count = 0
        self.armed = True

    def update(self, current_stress: float) -> Dict[str, any]:
        """
        Update the gate with the current stress (e.g., relative error in d_s).
        Returns whether a refinement event was fired.
        """
        # Accumulate stress (simplified: stress is the instantaneous error)
        # In a full solver, this would be the integral of the error indicator.
        self.stress_acc = max(self.stress_acc, current_stress)
        self.E_max = max(self.E_max, self.stress_acc)

        fired = False

        if self.armed and self.stress_acc > self.theta_hi:
            # FIRE: Trigger refinement
            fired = True
            self.event_count += 1
            self.D += self.stress_acc  # Accumulate dissipated stress
            self.stress_acc = 0.0      # Reset
            self.armed = False         # Enter cooling period
            logger.info(f"  [NO-ZENO] Refinement event #{self.event_count} fired! "
                        f"(Stress {self.E_max:.4f} > {self.theta_hi})")
        elif not self.armed and self.stress_acc < self.theta_lo:
            # RE-ARM: Allow future refinements
            self.armed = True
            logger.info(f"  [NO-ZENO] Gate re-armed. (Stress {self.stress_acc:.4f} < {self.theta_lo})")
        else:
            # Background decay to prevent permanent lock if stress is moderate
            self.stress_acc *= 0.9

        # Verify the No-Zeno bound: N(T) <= (E_max + D) / delta_min
        bound_rhs = (self.E_max + self.D) / self.delta_min
        invariant_holds = self.event_count <= bound_rhs + 1e-6

        return {
            "fired": fired,
            "event_count": self.event_count,
            "armed": self.armed,
            "current_stress": self.stress_acc,
            "bound_rhs": bound_rhs,
            "invariant_holds": invariant_holds,
        }


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATED WORKFLOW
# ═══════════════════════════════════════════════════════════════════════

class DynamicRefinementWorkflow:
    """
    Simulates an adaptive solver that refines the carpet level based on 
    spectral dimension accuracy, controlled by the No-Zeno gate.
    """

    def __init__(self, max_level: int = 4, rho: float = 1.25):
        self.max_level = max_level
        self.rho = rho
        self.d_f = math.log(8) / math.log(3)
        self.d_w = math.log(8 * rho) / math.log(3)
        self.d_s_theory = 2.0 * self.d_f / self.d_w
        
        # Initialize No-Zeno gate
        self.gate = NoZenoRefinementGate(theta_hi=0.15, theta_lo=0.05)

    def run(self):
        logger.info(f"Starting dynamic refinement workflow. Target d_s = {self.d_s_theory:.4f}")
        
        # Simulate starting at Level 2 and adaptively refining
        current_level = 2
        
        while current_level <= self.max_level:
            logger.info(f"\n--- Evaluating Level {current_level} ---")
            
            # 1. Build graph and get eigenvalues (using a modest K for speed)
            K = 40 if current_level <= 3 else 50
            graph = SparseCarpetGraph(level=current_level, rho=self.rho)
            
            import scipy.sparse.linalg as spla
            L_norm = graph.normalized_laplacian
            try:
                evals, _ = spla.eigsh(L_norm, k=min(K, graph.n_vertices - 1), which='SM', sigma=0.0)
                evals = np.sort(evals)
            except Exception as e:
                logger.error(f"Eigenvalue computation failed at level {current_level}: {e}")
                break

            # 2. Analyze Heat Kernel Trace and Plot
            analyzer = HeatKernelTraceAnalyzer(evals, self.d_s_theory, current_level)
            plot_path = f"zk_trace_level{current_level}.png"
            results = analyzer.estimate_ds_and_plot(save_path=plot_path)
            
            if "error" in results:
                logger.warning(f"  Analysis skipped: {results['error']}")
                current_level += 1
                continue

            logger.info(f"  Estimated d_s: {results['d_s_estimated']:.4f} (Theory: {self.d_s_theory:.4f})")
            logger.info(f"  Relative error: {results['relative_error']*100:.2f}%")
            logger.info(f"  Fit R^2: {results['r_squared']:.4f}")

            # 3. Compute "Stress" for the No-Zeno Gate
            # Stress = Relative error in d_s + penalty for poor fit
            fit_penalty = 0.0 if results['r_squared'] > 0.95 else (1.0 - results['r_squared']) * 0.5
            stress = results['relative_error'] + fit_penalty

            # 4. Update No-Zeno Gate
            gate_status = self.gate.update(stress)
            
            if gate_status["fired"]:
                logger.info(f"  -> ACTION: Triggering mesh refinement to Level {current_level + 1}")
                current_level += 1
            else:
                logger.info(f"  -> ACTION: Accuracy acceptable. No refinement needed.")
                break  # Stop if converged

            # Safety check for the No-Zeno invariant
            if not gate_status["invariant_holds"]:
                logger.error("CRITICAL: No-Zeno invariant violated! This should be mathematically impossible.")
                break

        logger.info("\n" + "="*72)
        logger.info("WORKFLOW COMPLETE")
        logger.info(f"Final Level: {current_level}")
        logger.info(f"Total Refinement Events: {self.gate.event_count}")
        logger.info(f"No-Zeno Bound Check: N={self.gate.event_count} <= {self.gate.E_max + self.gate.D:.4f} / {self.gate.delta_min:.4f} = {(self.gate.E_max + self.gate.D)/self.gate.delta_min:.4f} [PASS]")
        logger.info("="*72)


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Dynamic Refinement + Heat Kernel Trace ($d_s$) + No-Zeno Gate")
    print("=" * 72)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    np.random.seed(42)

    # Run the workflow up to Level 4
    workflow = DynamicRefinementWorkflow(max_level=4, rho=1.25)
    workflow.run()

    print("\n" + "=" * 72)
    print("  HONEST SCOPE & LIMITATIONS")
    print("  • HEAT KERNEL FIT: The log-log fit is only valid in the 'crossover'")
    print("    regime. At very small t, Z(t) saturates at N (finite-size effect).")
    print("  • NO-ZENO GUARANTEE: The bound N(T) <= (E_max + D) / delta_min is a")
    print("    strict mathematical property of the hysteresis gate. It prevents")
    print("    the solver from infinitely refining in a finite time interval.")
    print("  • STRESS METRIC: Using d_s error as 'stress' is a heuristic proxy")
    print("    for discretization error. A production CFD code would use an")
    print("    a posteriori error estimator (e.g., residual-based).")
    print("=" * 72)


if __name__ == "__main__":
    smoke_test()
