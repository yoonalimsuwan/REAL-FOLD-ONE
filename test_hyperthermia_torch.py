"""Torch tests for fixed/nanobot_hyperthermia_fixed.py (not executed in authoring sandbox)."""
import os, sys, unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed")
sys.path.insert(0, os.environ.get("NSEM_SRC", "/mnt/user-data/outputs/fixed"))
from nanobot_hyperthermia_fixed import NanobotHyperthermiaAblationModule as M, TissueProperties

Tc = TissueProperties().core_body_temp


def fields(B=1, n=8, dtype=torch.float32):
    T = torch.full((B, 1, n, n, n), Tc, dtype=dtype)
    return T, torch.zeros_like(T), T.clone()


class TestFixed(unittest.TestCase):
    def test_uniform_stays_uniform_both_boundaries(self):
        for b in ("neumann", "core"):
            m = M(1e-4, 1e-3, device="cpu", boundary=b)
            T, rho, g = fields()
            out, *_ = m(T, rho, g, 0.0, 0.0)
            self.assertLess((out - T).abs().max().item(), 1e-3, b)   # fp32 on 310 K

    def test_perfusion_decay_rate(self):
        t = TissueProperties()
        rate = t.blood_perfusion_rate * t.blood_density * t.blood_specific_heat / (t.density * t.specific_heat)
        m = M(1e-3, 1e-2, device="cpu")
        T, rho, g = fields(n=4, dtype=torch.float64)
        T = T + 5.0
        m = m.double()
        for _ in range(1000):
            T, _, g = m(T, rho.double(), g, 0.0, 0.0)
        self.assertAlmostEqual((T - Tc).mean().item(), 5.0 * float(torch.exp(torch.tensor(-rate * 10.0))), delta=1e-3)

    def test_cfl_guard(self):
        with self.assertRaises(ValueError):
            M(1e-4, 1.0, device="cpu")                    # alpha*dt/dx^2 >> 1/6

    def test_checkpointing_runs(self):
        m = M(1e-4, 1e-3, device="cpu", gradient_checkpointing=True).train()
        T, rho, g = fields()
        T.requires_grad_()
        out, dmg, _ = m(T, rho + 1e-2, g, 0.05, 1e5)
        (out.sum() + dmg.sum()).backward()
        self.assertTrue(torch.isfinite(T.grad).all())

    def test_gradients_wrt_field_amplitude(self):
        m = M(1e-4, 1e-3, device="cpu")
        T, rho, g = fields()
        B0 = torch.tensor(0.05, requires_grad=True)
        out, *_ = m(T, rho + 1e-2, g, B0, 1e5)
        out.sum().backward()
        self.assertGreater(B0.grad.abs().item(), 0.0)

    def test_baseline_damage_at_body_temperature_is_tiny(self):
        m = M(1e-4, 1e-3, device="cpu")
        T, rho, g = fields()
        _, dmg, _ = m(T, rho, g, 0.0, 0.0)
        self.assertLess(dmg.max().item(), 1e-4)        # softplus tail: nonzero, documented


if __name__ == "__main__":
    unittest.main()
