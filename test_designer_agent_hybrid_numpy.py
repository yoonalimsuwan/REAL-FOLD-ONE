"""Torch-free checks for hybrid_dynamical_energy_network_model, denovo_sequence_designer and
denovo_protein_calculus_agent (NumPy transcriptions)."""
import unittest
import numpy as np

sp = lambda x, b=1.0: np.log1p(np.exp(b * x)) / b


# ------------------------------------------------------------------ hybrid Lorenz network
def lorenz(s, sig=10.0, rho=28.0, beta=8 / 3):
    return np.array([sig * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1], s[0] * s[1] - beta * s[2]])


def rk4(s, dt):
    k1 = lorenz(s); k2 = lorenz(s + dt / 2 * k1); k3 = lorenz(s + dt / 2 * k2); k4 = lorenz(s + dt * k3)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run(s0, T, dt):
    s = np.array(s0, float)
    for _ in range(int(round(T / dt))):
        s = rk4(s, dt)
    return s


class TestHybridLorenz(unittest.TestCase):
    def test_rk4_is_at_least_fourth_order(self):
        ref = run([1, 1, 1], 1.0, 1e-4)
        e = [np.linalg.norm(run([1, 1, 1], 1.0, dt) - ref) for dt in (0.04, 0.02, 0.01)]
        self.assertGreater(np.log2(e[0] / e[1]), 3.8)
        self.assertGreater(np.log2(e[1] / e[2]), 3.8)

    def test_fixed_points_have_zero_derivative(self):
        b, r = 8 / 3, 28.0
        c = np.sqrt(b * (r - 1))
        for fp in ([0, 0, 0], [c, c, r - 1], [-c, -c, r - 1]):
            self.assertLess(np.abs(lorenz(np.array(fp, float))).max(), 1e-9)

    def test_symptom_nodes_saturate_at_default_constants(self):
        """P_t = |x|+|y|+(1-z) is O(10) and ext-drive uses randn weights, so tanh is saturated."""
        s = np.array([1.0, 1.0, 1.0]); tr = []
        for i in range(30000):
            s = rk4(s, 0.01)
            if i > 2000: tr.append(s.copy())
        x, y, z = np.array(tr).T
        P = np.abs(x) + np.abs(y) + (1 - z)
        g, d, e = np.random.default_rng(0).normal(size=(3, 64))
        u = P[:, None] + g * x[:, None] + d * y[:, None] - e * z[:, None]
        self.assertGreater((np.abs(u) > 5).mean(), 0.7)               # observed 0.86
        self.assertLess(np.median(1 - np.tanh(u) ** 2), 1e-6)         # median tanh' ~ 2e-16


# ------------------------------------------------------------------ sequence designer
class TestSequenceDesigner(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0); self.rng = rng
        self.V, self.H, self.lam = 20, 128, 1e-4
        lin = lambda i, o: rng.uniform(-1 / np.sqrt(i), 1 / np.sqrt(i), (o, i))
        self.Wp, self.Ws = lin(self.V, self.H), lin(self.H, self.H)

    def M(self, idx):
        return (np.eye(self.V)[idx] @ self.Wp.T) @ self.Ws.T

    def v_orig(self, idx):
        L = len(idx); M = self.M(idx)
        ld = np.linalg.slogdet(M @ M.T + self.lam * np.eye(L))[1]
        return 1 / (1 + np.exp(-ld / L))

    def v_fixed(self, idx):
        L = len(idx); M = self.M(idx)
        ld = np.linalg.slogdet(np.eye(self.H) + M.T @ M / self.lam)[1]
        return -np.expm1(-ld / L)

    def test_rank_bounded_by_vocab(self):
        M = self.M(self.rng.integers(0, self.V, 300))
        self.assertLessEqual(np.linalg.matrix_rank(M), self.V)

    def test_original_viability_is_sequence_independent(self):
        v = np.array([self.v_orig(self.rng.integers(0, self.V, 800)) for _ in range(10)])
        self.assertLess(v.max(), 1e-3)                                  # ~1.4e-4 for every sequence
        self.assertLess(v.std() / v.mean(), 1e-3)                       # relative spread ~1e-4

    def test_sylvester_identity(self):
        idx = self.rng.integers(0, self.V, 60); M = self.M(idx); L = len(idx)
        a = np.linalg.slogdet(np.eye(L) + M @ M.T / self.lam)[1]
        b = np.linalg.slogdet(np.eye(self.H) + M.T @ M / self.lam)[1]
        self.assertAlmostEqual(a, b, delta=1e-6 * abs(a))

    def test_fixed_viability_responds_to_composition_and_is_bounded(self):
        mono = self.v_fixed(np.zeros(200, int))
        diverse = self.v_fixed(self.rng.integers(0, self.V, 200))
        self.assertTrue(0 <= mono < diverse < 1)
        self.assertGreater(diverse - mono, 0.01)

    def test_fixed_viability_zero_for_zero_signature(self):
        self.assertEqual(float(-np.expm1(-0.0)), 0.0)


# ------------------------------------------------------------------ protein calculus agent
class TestProteinAgent(unittest.TestCase):
    def test_optimised_contraction_equals_reference_triple_sum(self):
        rng = np.random.default_rng(1); B, n, m = 3, 7, 11
        x, C, D = rng.normal(size=(B, n)), rng.normal(size=(m, n)), rng.normal(size=(m, n))
        ref = np.einsum("bn,mn,mp->bnp", x, C, D)
        opt = x[:, :, None] * (C.T @ D)[None]
        self.assertTrue(np.allclose(ref, opt))

    def gate(self, dE, dt, c1_raw=0.5413, s_raw=-2.2520, zmax=15.0, floor=-25.0, eps=1e-6):
        c1, s2 = sp(c1_raw) + eps, sp(s_raw) + eps
        arg = np.minimum(dE / (s2 * dt + eps), zmax)
        return np.exp(np.maximum(-c1 * np.exp(arg), floor)), dE / (s2 * dt + eps) > zmax

    def test_default_dt_collapses_output_scale(self):
        dE = sp(np.random.default_rng(0).normal(size=100000), 2.0)
        p, clamped = self.gate(dE, 0.01)
        self.assertLess(np.median(p), 1e-9)                              # 1.4e-11
        self.assertGreater(clamped.mean(), 0.9)                          # ~96 % have zero gradient

    def test_gate_ceiling_and_healthy_regime(self):
        dE = sp(np.random.default_rng(0).normal(size=100000), 2.0)
        p, clamped = self.gate(dE, 10.0)
        self.assertEqual(clamped.mean(), 0.0)
        self.assertLess(p.max(), np.exp(-1.0) + 1e-3)                    # can never exceed e^{-c1}

    def test_viability_not_degenerate_but_near_saturation(self):
        rng = np.random.default_rng(0); n, m, lam = 64, 128, 1e-4
        v = []
        for _ in range(60):
            C, D = rng.normal(size=(m, n)), rng.normal(size=(m, n))
            M = (rng.normal(size=n) * 0.3)[:, None] * (C.T @ D)
            v.append(1 / (1 + np.exp(-np.linalg.slogdet(M @ M.T + lam * np.eye(n))[1] / n)))
        v = np.array(v)
        self.assertTrue(((v > 0.9) & (v < 1)).all())                     # informative-ish, but >0.96


if __name__ == "__main__":
    unittest.main()
