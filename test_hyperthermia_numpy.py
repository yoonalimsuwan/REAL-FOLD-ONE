"""Torch-free check of the *fixed* hyperthermia physics (NumPy mirror of one explicit step)."""
import unittest
import numpy as np

rho, c, k, w = 1050.0, 3600.0, 0.51, 0.005
rho_b, c_b = 1060.0, 3617.0
rho_cp = rho * c
alpha = k / rho_cp
Tc = 310.15


def step(T, q, dx, dt, boundary="neumann"):
    P = np.pad(T, 1, mode="edge") if boundary == "neumann" else np.pad(T, 1, constant_values=Tc)
    lap = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] + P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1]
           + P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] - 6 * T) / dx ** 2
    rate = w * rho_b * c_b / rho_cp
    return T + dt * (alpha * lap + q / rho_cp - rate * (T - Tc))


class TestHyperthermiaFixed(unittest.TestCase):
    def test_uniform_body_temperature_is_stationary(self):
        T = np.full((6, 6, 6), Tc)
        for b in ("neumann", "core"):
            self.assertLess(np.abs(step(T, 0.0, 1e-4, 1e-3, b) - T).max(), 1e-9, b)

    def test_perfusion_matches_analytic_exponential(self):
        T = np.full((4, 4, 4), Tc + 5.0)
        dt, n = 1e-2, 3000                                    # 30 s
        for _ in range(n):
            T = step(T, 0.0, 1e-3, dt)
        rate = w * rho_b * c_b / rho_cp
        self.assertAlmostEqual(T[2, 2, 2] - Tc, 5.0 * np.exp(-rate * dt * n), delta=1e-3)

    def test_steady_state_with_heating(self):
        q = 2e5
        T = np.full((4, 4, 4), Tc)
        rate = w * rho_b * c_b / rho_cp
        for _ in range(int(2400 / 0.05)):                     # 40 min ~ 12 time constants (1/rate ~ 197 s)
            T = step(T, q, 1e-3, 0.05)
        self.assertAlmostEqual(T[2, 2, 2] - Tc, q / (rho_cp * rate), delta=0.02)

    def test_gaussian_diffusion_variance(self):
        """Pure conduction (perfusion off by zeroing deviation): var grows as 2*alpha*t."""
        n, dx, dt = 41, 1e-4, 1e-3
        x = (np.arange(n) - n // 2) * dx
        s0 = 3 * dx
        g = np.exp(-(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2) / (2 * s0 ** 2))
        T = Tc + g
        steps = 300
        for _ in range(steps):
            P = np.pad(T, 1, mode="edge")
            lap = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] + P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1]
                   + P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] - 6 * T) / dx ** 2
            T = T + dt * alpha * lap
        prof = (T - Tc)[n // 2, n // 2, :]
        var = (prof * x ** 2).sum() / prof.sum()
        self.assertAlmostEqual(var, s0 ** 2 + 2 * alpha * steps * dt, delta=0.05 * s0 ** 2)

    def test_cfl_value_at_defaults(self):
        self.assertLess(alpha * 1e-3 / 1e-4 ** 2, 1 / 6)


if __name__ == "__main__":
    unittest.main()
