"""Torch-free checks: (1) CRISPR-evolution No-Zeno gate at default constants,
(2) inverse source localization: ill-posedness, classical baseline, activation asymmetry."""
import unittest
import numpy as np

sp = lambda x: np.log1p(np.exp(x))


def gate(var, dt, c1_init=1.2, de_init=2.5, inner_max=15.0, floor=-25.0, eps=1e-6):
    c1, de = sp(c1_init) + eps, sp(de_init) + eps
    s2 = sp(var) + eps
    raw = de / (s2 * dt)
    inner = np.minimum(raw, inner_max)
    return np.exp(np.maximum(-c1 * np.exp(inner), floor)), raw > inner_max


class TestEvolutionGate(unittest.TestCase):
    def test_default_dt_freezes_evolution(self):
        for var in (0.0, 0.5, 1.0):                         # variance = 1 - viability in [0, 1]
            p, clamped = gate(var, 0.01)
            self.assertLess(p, 1e-9)
            self.assertTrue(clamped)                        # gradient is exactly zero here

    def test_gate_has_a_ceiling_below_one(self):
        p, _ = gate(1.0, 1e9)
        self.assertAlmostEqual(float(p), np.exp(-(sp(1.2) + 1e-6)), places=4)
        self.assertLess(float(p), 0.24)                     # can never mutate more than ~23 %

    def test_monotone_in_variance_and_dt_outside_clamp(self):
        v = np.linspace(0, 1, 6)
        p, c = gate(v, 3.0)
        self.assertFalse(c.any())
        self.assertTrue((np.diff(p) > 0).all())
        d = np.array([2.0, 3.0, 5.0, 10.0])
        self.assertTrue((np.diff(gate(0.5, d)[0]) > 0).all())

    def test_mutation_active_only_for_dt_roughly_two_or_more(self):
        self.assertLess(gate(1.0, 0.5)[0], 1e-6)
        self.assertGreater(gate(1.0, 5.0)[0], 0.05)


class TestInverseLocalization(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.S, self.V, self.T = 64, 512, 20
        self.A = rng.uniform(-1, 1, (self.S, self.V)) / np.sqrt(self.V)    # random stand-in lead field
        x = np.zeros((self.V, self.T))
        self.active = [30, 200, 411]
        t = np.linspace(0, 1, self.T)
        for k, i in enumerate(self.active):
            x[i] = np.sin(2 * np.pi * (k + 1) * t)
        self.x = x

    def test_forward_is_linear(self):
        a, b = self.x, np.roll(self.x, 3, axis=0)
        self.assertTrue(np.allclose(self.A @ (2 * a + 3 * b), 2 * self.A @ a + 3 * self.A @ b))

    def test_problem_is_underdetermined(self):
        self.assertEqual(np.linalg.matrix_rank(self.A), self.S)
        null = np.linalg.svd(self.A)[2][self.S:].T                 # V-S null-space directions
        x2 = self.x + null[:, [0]] * 5.0
        self.assertTrue(np.allclose(self.A @ x2, self.A @ self.x, atol=1e-9))   # identical sensors
        self.assertFalse(np.allclose(x2, self.x))                                # different sources

    def test_minimum_norm_baseline_fits_sensors_but_blurs_sources(self):
        y = self.A @ self.x
        lam = 1e-6 * np.trace(self.A @ self.A.T) / self.S
        xh = self.A.T @ np.linalg.solve(self.A @ self.A.T + lam * np.eye(self.S), y)
        sensor_rel = np.linalg.norm(self.A @ xh - y) / np.linalg.norm(y)
        self.assertLess(sensor_rel, 1e-3)                                       # data are explained
        energy = (xh ** 2).sum(1)
        peak = np.argsort(energy)[-3:]
        hits = len(set(peak) & set(self.active))
        print(f"\n[localization] MNE baseline: sensor misfit {sensor_rel:.1e}, "
              f"true sources in top-3 voxels: {hits}/3, source rel. error "
              f"{np.linalg.norm(xh - self.x) / np.linalg.norm(self.x):.2f}")
        self.assertGreater(np.linalg.norm(xh - self.x) / np.linalg.norm(self.x), 0.5)   # blurred

    def test_refinement_activation_is_not_sign_symmetric(self):
        """raw * sigmoid(raw * g) with g=0.1 distorts negative amplitudes vs positive ones."""
        f = lambda r, g=0.1: r / (1 + np.exp(-r * g))
        self.assertGreater(abs(f(-1.0)) / abs(f(1.0)), 0.85)       # ~symmetric near zero
        for r in (5.0, 20.0):                                       # strongly asymmetric at large amplitude
            self.assertLess(abs(f(-r)) / abs(f(r)), 0.65)
        self.assertAlmostEqual(f(5.0), 3.1, delta=0.05)
        self.assertAlmostEqual(f(-20.0), -2.38, delta=0.05)


if __name__ == "__main__":
    unittest.main()
