"""Torch-free checks of the exact first-order updates used in the patched kinetics modules,
and a reproduction (with numbers) of the original softplus artefacts."""
import unittest
import numpy as np


def softplus(x, beta):
    return np.log1p(np.exp(beta * x)) / beta


def old_crispr_edits(target, beta=20.0, eps=1e-8):          # zero RNP -> nothing should be edited
    return target - (softplus(target - eps, beta) + eps)


def old_payload_decay(wanted, enc, beta=50.0):
    return enc - softplus(enc - wanted, beta) if False else wanted_smooth_min(wanted, enc, beta)


def wanted_smooth_min(a, b, beta):
    return b - softplus(b - a, beta)


def new_first_order(x, k, dt):
    return x * (-np.expm1(-k * dt))


class TestOldArtefacts(unittest.TestCase):
    def test_crispr_negative_edits_with_zero_rnp(self):
        self.assertLess(old_crispr_edits(0.01), -0.02)       # observed: -0.0299

    def test_payload_negative_release_at_low_concentration(self):
        enc, wanted = 0.05, 1e-5                               # tiny release request
        self.assertLess(wanted_smooth_min(wanted, enc, 50.0), 0.0)   # cargo flows backwards


class TestFixedFormulas(unittest.TestCase):
    def test_bounded_in_zero_and_x(self):
        rng = np.random.default_rng(0)
        x = 10 ** rng.uniform(-8, 2, 5000)
        k = 10 ** rng.uniform(-6, 3, 5000)
        d = new_first_order(x, k, 1e-3)
        self.assertTrue((d >= 0).all() and (d <= x + 1e-15).all())

    def test_zero_rate_gives_exactly_zero(self):
        self.assertEqual(new_first_order(np.array([0.01, 1.0]), 0.0, 1e-3).max(), 0.0)

    def test_matches_analytic_exponential_over_many_steps(self):
        e, k, dt, n = 1.0, 0.05, 1e-2, 5000
        x = e
        for _ in range(n):
            x -= new_first_order(x, k, dt)
        self.assertAlmostEqual(x, e * np.exp(-k * dt * n), places=10)   # exact, no dt error

    def test_total_cargo_conserved(self):
        enc, rel = np.array([0.3, 0.02, 1.0]), np.array([0.0, 0.5, 0.1])
        tot = enc + rel
        for _ in range(1000):
            d = new_first_order(enc, 0.04, 0.1)
            enc, rel = enc - d, rel + d
        self.assertTrue(np.allclose(enc + rel, tot, atol=1e-14))


if __name__ == "__main__":
    unittest.main()
