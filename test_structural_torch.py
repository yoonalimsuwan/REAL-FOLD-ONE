"""Torch tests for the patched structural / evolution / localization modules.
NOT executed in the authoring sandbox. Run: python -m unittest tests_math.test_structural_torch -v"""
import os, sys, unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed")
sys.path.insert(0, os.environ.get("NSEM_SRC", "/mnt/user-data/outputs/fixed"))
sys.path.insert(0, "/mnt/user-data/uploads")

from nmr_restraint_fixed import NMRRestraintSet
from xray_cryo_em_fixed import XrayCryoEMNativeEngine
from rice_ml import rice_acentric_nll


def rot(axis, th):
    a = torch.tensor(axis, dtype=torch.float64); a = a / a.norm()
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + torch.sin(torch.tensor(th)) * K + (1 - torch.cos(torch.tensor(th))) * K @ K


class TestNMRFixed(unittest.TestCase):
    def mod(self):
        return NMRRestraintSet(torch.tensor([[0, 5]]), torch.tensor([[1.8, 5.0]]))

    def test_kabsch_zero_on_rigid_copy(self):
        torch.manual_seed(0)
        P = torch.randn(40, 3, dtype=torch.float64)
        T = P @ rot([0.3, 1.0, 0.2], 1.1).T + 5.0
        self.assertLess(self.mod().compute_kabsch_rmsd(P, T).item(), 1e-3)   # was ~2.4

    def test_kabsch_batched_and_gradient(self):
        P = torch.randn(3, 20, 3, requires_grad=True)
        T = torch.randn(3, 20, 3)
        r = self.mod().compute_kabsch_rmsd(P, T)
        self.assertEqual(r.shape, (3,))
        r.sum().backward()
        self.assertTrue(torch.isfinite(P.grad).all())

    def test_noe_zero_inside_bounds(self):
        c = torch.zeros(6, 3); c[5, 0] = 3.0
        self.assertEqual(self.mod().compute_noe_penalty(c).item(), 0.0)
        c[5, 0] = 6.0
        self.assertAlmostEqual(self.mod().compute_noe_penalty(c).item(), 50.0, places=4)


class TestCryoFixed(unittest.TestCase):
    def maps(self, n=24):
        torch.manual_seed(1)
        fr = torch.fft.fftfreq(n)
        r2 = fr.view(-1, 1, 1) ** 2 + fr.view(1, -1, 1) ** 2 + fr.view(1, 1, -1) ** 2
        g = torch.fft.ifftn(torch.fft.fftn(torch.randn(n, n, n)) * torch.exp(-r2 / 0.5)).real
        return g

    def test_identical_maps_fsc_one(self):
        e = XrayCryoEMNativeEngine(n_fsc_shells=20); g = self.maps()
        self.assertAlmostEqual(e.compute_cryo_em_map_ml(g, g)["cross_correlation"].item(), 1.0, places=3)

    def test_h_axis_shift_is_penalised_like_3d_fsc(self):
        e = XrayCryoEMNativeEngine(n_fsc_shells=20); g = self.maps()
        v = e.compute_cryo_em_map_ml(g, torch.roll(g, 2, dims=1))["cross_correlation"].item()
        self.assertLess(v, 0.5)                                  # 3-D FSC ~0.28; skipping H gave ~-0.06/0.5

    def test_gradient_flows(self):
        e = XrayCryoEMNativeEngine(n_fsc_shells=20); g = self.maps()
        s = (g + 0.3 * torch.randn_like(g)).requires_grad_()
        e.compute_cryo_em_map_ml(g, s)["nll_cryo"].backward()
        self.assertTrue(torch.isfinite(s.grad).all())


class TestRice(unittest.TestCase):
    def test_matches_numpy_density(self):
        import numpy as np
        fo = torch.linspace(0.05, 20, 4000, dtype=torch.float64)
        nll = rice_acentric_nll(fo, torch.tensor(3.0, dtype=torch.float64),
                                torch.tensor(0.9, dtype=torch.float64), torch.tensor(2.0, dtype=torch.float64))
        pdf = torch.exp(-nll).numpy()
        self.assertAlmostEqual(float(np.trapezoid(pdf, fo.numpy())), 1.0, places=3)


class TestEvolutionGate(unittest.TestCase):
    def test_default_dt_clamps_gate(self):
        from super_dns_crispr_evolution_fixed import DoubleExponentialNoZenoGate
        g = DoubleExponentialNoZenoGate()
        p = g(torch.tensor([0.0, 0.5, 1.0]), 0.01)
        self.assertTrue((p < 1e-9).all())
        self.assertEqual(g.last_clamped_fraction.item(), 1.0)

    def test_gate_active_at_large_dt_with_gradient(self):
        from super_dns_crispr_evolution_fixed import DoubleExponentialNoZenoGate
        g = DoubleExponentialNoZenoGate()
        v = torch.tensor([0.5], requires_grad=True)
        p = g(v, 5.0); p.sum().backward()
        self.assertGreater(p.item(), 0.05)
        self.assertGreater(v.grad.abs().item(), 0.0)


class TestLocalization(unittest.TestCase):
    def test_forward_matches_matrix_product_and_linear(self):
        from remote_neural_monitoring_differentiable_inverse_source_localization__1_ import DifferentiableForwardPhysics
        m = DifferentiableForwardPhysics(8, 20)
        a, b = torch.randn(2, 20, 5), torch.randn(2, 20, 5)
        self.assertTrue(torch.allclose(m(2 * a + b), 2 * m(a) + m(b), atol=1e-5))
        self.assertTrue(torch.allclose(m(a), m.lead_field @ a, atol=1e-6))

    def test_solver_shapes_and_gradient(self):
        from remote_neural_monitoring_differentiable_inverse_source_localization__1_ import EfficientInverseSolver
        s = EfficientInverseSolver(16, 40, hidden_dim=16, conv_groups=4)
        y = torch.randn(2, 16, 10, requires_grad=True)
        out = s(y); self.assertEqual(out.shape, (2, 40, 10))
        out.sum().backward(); self.assertTrue(torch.isfinite(y.grad).all())


if __name__ == "__main__":
    unittest.main()
