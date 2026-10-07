"""Torch tests for the fixed files. Skipped automatically if torch is missing.
NOT executed in the authoring sandbox. Run: python -m unittest tests_math.test_torch_math -v"""
import os, sys, unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed")

sys.path.insert(0, os.environ.get("NSEM_SRC", "/mnt/user-data/outputs/fixed"))
sys.path.insert(0, "/mnt/user-data/uploads")
from consolidated_implementation_library import FractalGraphNormCoercivity, HysteresisGateSTE
from assumption_Light_DeltaMin_Estimator__1_ import AssumptionLightDeltaMinEstimator


class TestDeltaMinTorch(unittest.TestCase):
    def test_shapes_batch_and_unbatched(self):
        est = AssumptionLightDeltaMinEstimator(p=0.05, alpha=0.05)
        x = torch.cumsum(torch.randn(4, 6, 300, 3), dim=2)
        d, lo, hi = est(x)
        self.assertEqual(d.shape, (4,))
        self.assertTrue((lo <= hi).all())
        d1, lo1, hi1 = est(x[0])
        self.assertTrue(torch.allclose(d1.reshape(()), d[0], atol=1e-5))

    def test_matches_numpy_mirror(self):
        from tests_math.test_numpy_validation import delta_min_ci
        est = AssumptionLightDeltaMinEstimator(p=0.05, alpha=0.05)
        x = torch.cumsum(torch.randn(6, 300, 3, dtype=torch.float64), dim=1)
        d, lo, hi = est(x)
        nd, nlo, nhi = delta_min_ci(x.numpy(), p=0.05, alpha=0.05)
        for a, b in ((d, nd), (lo, nlo), (hi, nhi)):
            self.assertAlmostEqual(float(a.reshape(())), float(b), places=3)

    def test_gradient_flows(self):
        est = AssumptionLightDeltaMinEstimator()
        x = torch.cumsum(torch.randn(5, 200, 3), 1).requires_grad_()
        d, lo, hi = est(x)
        (hi - lo).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class TestCoercivityTorch(unittest.TestCase):
    def _lap(self, n=40):
        A = torch.zeros(n, n, dtype=torch.float64)
        i = torch.arange(n - 1)
        A[i, i + 1] = 1.0; A[i + 1, i] = 1.0
        return torch.diag(A.sum(1)) - A

    def test_margin_sign_follows_spectrum(self):
        L = self._lap()
        w, V = torch.linalg.eigh(L)
        m = FractalGraphNormCoercivity(L)
        self.assertFalse(bool(m(V[:, 1])["is_coercive"]))      # low mode
        self.assertTrue(bool(m(V[:, -1])["is_coercive"]))      # high mode

    def test_margin_is_differentiable(self):
        m = FractalGraphNormCoercivity(self._lap())
        u = torch.randn(40, dtype=torch.float64, requires_grad=True)
        m(u)["margin"].backward()
        self.assertTrue(torch.isfinite(u.grad).all())


if __name__ == "__main__":
    unittest.main()
