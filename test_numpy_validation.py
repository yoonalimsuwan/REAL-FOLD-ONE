"""
Torch-free validation. Pulls pure-NumPy functions out of the source files with
`ast` (so torch is not needed) and checks them against known ground truth, plus
a NumPy mirror of the DeltaMin subsampling algorithm to measure CI coverage.
Run:  python -m unittest tests_math.test_numpy_validation -v
"""
import __future__, ast, math, os, unittest
import numpy as np

SRC = os.environ.get("NSEM_SRC", "/mnt/user-data/outputs/fixed")


def extract(path, func):
    src = open(os.path.join(SRC, path)).read()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == func)
    ns = {"np": np}
    code = compile(ast.Module([node], []), path, "exec",
                   flags=__future__.annotations.compiler_flag)   # lazy annotations
    exec(code, ns)
    return ns[func]


raw_process_subsampling = extract("consolidated_implementation_library.py",
                                  "raw_process_subsampling")


def delta_min_ci(traj, p=0.01, alpha=0.05):
    """NumPy mirror of AssumptionLightDeltaMinEstimator._estimate ([N,T,F])."""
    exc = np.linalg.norm(traj[:, 1:] - traj[:, :-1], axis=-1)          # [N,T-1]
    N, T1 = exc.shape
    d_hat = np.quantile(exc.reshape(-1), p)
    b = max(2, min(int(round(T1 ** (1 / 3))), T1))
    nb = T1 - b + 1
    blocks = np.lib.stride_tricks.sliding_window_view(exc, b, axis=1)   # [N,nb,b]
    sub = np.quantile(blocks.transpose(1, 0, 2).reshape(nb, -1), p, axis=-1)
    stat = math.sqrt(b) * (sub - d_hat)
    lo_c, hi_c = np.quantile(stat, [alpha / 2, 1 - alpha / 2])
    return d_hat, d_hat - hi_c / math.sqrt(b), d_hat - lo_c / math.sqrt(b)


class TestHill(unittest.TestCase):
    def test_hill_recovers_pareto_tail_index(self):
        rng = np.random.default_rng(0)
        alpha = 2.0                                        # theta = 1/alpha = 0.5
        Y = rng.pareto(alpha, 6000) + 1.0
        out = raw_process_subsampling(Y, b_n=1500, k_frac=0.1)
        self.assertAlmostEqual(float(np.median(out["theta"])), 1 / alpha, delta=0.08)

    def test_hill_has_no_input_guard_for_nonpositive_data(self):
        """Hill needs positive data. Mostly-negative input silently yields NaN,
        and mixed-sign input silently yields a meaningless number (no error)."""
        Y = np.random.default_rng(1).normal(size=400) - 3.0      # top order stats < 0
        out = raw_process_subsampling(Y, b_n=100)
        self.assertTrue(np.isnan(out["theta"]).any())

    def test_cusum_flags_mean_shift(self):
        rng = np.random.default_rng(2)
        Y = np.concatenate([rng.pareto(3, 500) + 1, rng.pareto(3, 500) + 6])
        out = raw_process_subsampling(Y, b_n=200, cusum_threshold=3.0)
        self.assertTrue(out["flagged"].any())
        self.assertFalse(out["flagged"][:100].all())


class TestDeltaMinCoverage(unittest.TestCase):
    def test_empirical_coverage_reported(self):
        """True delta_min = p-quantile of ||step|| for iid N(0,I_F) steps."""
        rng = np.random.default_rng(3)
        F, p, N, T, reps = 3, 0.05, 6, 500, 200
        truth = np.quantile(np.linalg.norm(rng.normal(size=(2_000_000, F)), axis=1), p)
        hit = 0
        for _ in range(reps):
            steps = rng.normal(size=(N, T, F))
            traj = np.cumsum(steps, axis=1)
            _, lo, hi = delta_min_ci(traj, p=p, alpha=0.05)
            hit += lo <= truth <= hi
        cov = hit / reps
        print(f"\n[DeltaMin] nominal 95% CI empirical coverage = {cov:.2f}")
        # Measured: 1.00 at nominal 0.95 -> valid but conservative (CI ~45-80% of truth).
        self.assertGreaterEqual(cov, 0.90)

    def test_ci_ordering_and_width_shrinks_with_T(self):
        rng = np.random.default_rng(4)
        w = []
        for T in (200, 1600):
            traj = np.cumsum(rng.normal(size=(6, T, 3)), axis=1)
            d, lo, hi = delta_min_ci(traj, p=0.05)
            self.assertLessEqual(lo, hi)
            w.append(hi - lo)
        self.assertLess(w[1], w[0])


class TestCoercivityClaim(unittest.TestCase):
    """margin = |L^4 u|^2 - |L u|^2 - |L^3 u|^2 is NOT sign-definite."""

    def test_margin_sign_depends_on_spectrum(self):
        n = 40
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        L = np.diag(A.sum(1)) - A
        w, V = np.linalg.eigh(L)
        def margin(u):
            d1 = L @ u; d3 = L @ (L @ d1); d4 = L @ d3
            return d4 @ d4 - d1 @ d1 - d3 @ d3
        low, high = V[:, 1], V[:, -1]            # smallest nonzero / largest eigenvalue
        self.assertLess(margin(low), 0)           # low-frequency mode: NOT coercive
        self.assertGreater(margin(high), 0)       # high-frequency mode: coercive


if __name__ == "__main__":
    unittest.main()
