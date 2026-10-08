"""Torch tests for the patched designer / agent (not executed in the authoring sandbox)."""
import os, sys, unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed")
sys.path.insert(0, os.environ.get("NSEM_SRC", "/mnt/user-data/outputs/fixed"))
from denovo_sequence_designer_fixed import DifferentiableStructuralDesigner
from denovo_protein_calculus_agent_fixed import NoZenoTopologicalGating


class TestDesigner(unittest.TestCase):
    def test_viability_in_unit_interval_and_matches_sylvester(self):
        d = DifferentiableStructuralDesigner(seq_length=40, vocab_size=20, hidden_dim=32).eval()
        M = torch.randn(3, 40, 32)
        v = d._compute_viability(M)
        self.assertTrue(((v >= 0) & (v < 1)).all())
        lam = float(d._lambda)
        ref = torch.linalg.slogdet(torch.eye(40) + M @ M.transpose(-2, -1) / lam)[1]
        got = -torch.log1p(-v.squeeze(-1)) * float(d._viability_scale)
        self.assertTrue(torch.allclose(got, ref, rtol=1e-3))

    def test_viability_varies_with_sequence(self):
        d = DifferentiableStructuralDesigner(seq_length=60, vocab_size=20, hidden_dim=32).eval()
        a = d.signature_matrix(d.phi_u_tensor(torch.zeros(1, 60, 20).index_fill_(-1, torch.tensor([0]), 1.0)))
        idx = torch.randint(0, 20, (1, 60))
        b = d.signature_matrix(d.phi_u_tensor(torch.nn.functional.one_hot(idx, 20).float()))
        self.assertGreater((d._compute_viability(b) - d._compute_viability(a)).item(), 0.0)

    def test_gradient_reaches_logits(self):
        d = DifferentiableStructuralDesigner(seq_length=30, vocab_size=20, hidden_dim=16).train()
        out = d(4) if callable(d) else None
        loss = out["viability_score"].sum() if isinstance(out, dict) else out[1].sum()
        loss.backward()
        self.assertGreater(d.seq_logits.grad.abs().sum().item(), 0.0)


class TestGate(unittest.TestCase):
    def test_default_dt_is_clamped_and_reported(self):
        g = NoZenoTopologicalGating()
        dE = torch.nn.functional.softplus(torch.randn(10000), beta=2.0)
        p = g(dE, 0.01)
        self.assertGreater(g.last_clamped_fraction.item(), 0.9)
        self.assertLess(p.median().item(), 1e-9)

    def test_healthy_regime_has_gradient(self):
        g = NoZenoTopologicalGating()
        dE = torch.full((8,), 0.3, requires_grad=True)
        g(dE, 10.0).sum().backward()
        self.assertGreater(dE.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
