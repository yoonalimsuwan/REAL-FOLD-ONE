"""Torch tests for fixed/nanobot_crispr_fixed.py and fixed/nanobot_payload_fixed.py
(not executed in the authoring sandbox)."""
import os, sys, unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch not installed")
sys.path.insert(0, os.environ.get("NSEM_SRC", "/mnt/user-data/outputs/fixed"))
from nanobot_crispr_fixed import CRISPRConfig, NanobotCRISPRCasEditingModule
from nanobot_payload_fixed import NanobotPayloadDeliveryModule


def crispr(**kw):
    return NanobotCRISPRCasEditingModule(CRISPRConfig(device="cpu", learnable_kinetics=True, **kw))


def vox(v, n=4):
    return torch.full((1, 1, n, n, n), float(v))


class TestCRISPR(unittest.TestCase):
    def run_mod(self, m, rnp, dna):
        return m(vox(rnp), vox(dna), vox(310.15), vox(2.0))

    def test_edits_bounded_and_nonnegative(self):
        m = crispr()
        for dna in (0.005, 0.05, 0.5, 5.0):
            for rnp in (0.0, 1e-4, 1.0):
                rem, edits, _ = self.run_mod(m, rnp, dna)
                self.assertTrue((edits >= 0).all(), (dna, rnp))
                self.assertTrue((edits <= dna + 1e-7).all(), (dna, rnp))

    def test_zero_rnp_zero_edits(self):
        _, edits, _ = self.run_mod(crispr(), 0.0, 0.01)
        self.assertLess(edits.abs().max().item(), 1e-9)       # was -0.03 before the fix

    def test_conservation(self):
        rem, edits, _ = self.run_mod(crispr(), 1.0, 0.7)
        self.assertTrue(torch.allclose(rem + edits, vox(0.7), atol=1e-6))

    def test_more_rnp_more_edits(self):
        m = crispr()
        lo = self.run_mod(m, 1e-4, 1.0)[1].mean()
        hi = self.run_mod(m, 1e-2, 1.0)[1].mean()
        self.assertGreater(hi.item(), lo.item())

    def test_gradients_reach_kinetic_parameters(self):
        m = crispr()
        _, edits, met = self.run_mod(m, 1e-3, 1.0)
        (edits.sum() + met["off_target_risk_map"].sum()).backward()
        self.assertGreater(m.v_max_cleavage.grad.abs().item(), 0.0)

    @unittest.expectedFailure
    def test_higher_import_rate_should_not_reduce_nuclear_rnp(self):
        """KNOWN QUIRK (flagged, not changed): nuclear_rnp = active - k*active*gate*dt, i.e. the
        NON-imported remainder, so raising k_nuc_import lowers it. Author must decide intent."""
        a, b = crispr(), crispr()
        with torch.no_grad():
            b.k_nuc_import.fill_(15.0)
        na = self.run_mod(a, 1.0, 1.0)[2]["nuclear_rnp"].mean()
        nb = self.run_mod(b, 1.0, 1.0)[2]["nuclear_rnp"].mean()
        self.assertGreaterEqual(nb.item(), na.item())


class TestPayload(unittest.TestCase):
    def mk(self):
        return NanobotPayloadDeliveryModule(dt=0.1, device="cpu").eval()

    def inputs(self, enc, rel=0.0, ph=7.4, mmp=0.0, topo=0.0, n=6):
        p = torch.cat([vox(enc, n), vox(rel, n)], dim=1)
        return p, vox(ph, n), vox(mmp, n), vox(topo, n)

    def test_mass_conserved_without_topology_blend(self):
        m = self.mk()
        p, ph, mmp, tw = self.inputs(0.3, 0.1, ph=6.0, mmp=2.0)
        out, _ = m(p, ph, mmp, tw)
        self.assertTrue(torch.allclose(out.sum(1), p.sum(1), atol=1e-6))

    def test_no_negative_concentrations_at_low_density(self):
        m = self.mk()
        for enc in (0.0, 0.01, 0.05, 0.5):
            out, _ = m(*self.inputs(enc))
            self.assertGreaterEqual(out.min().item(), -1e-9, enc)   # was -0.0016 .. -0.0139

    def test_matches_analytic_decay(self):
        m = self.mk()
        p, ph, mmp, tw = self.inputs(1.0, ph=7.4, mmp=-50.0)
        k = m.kinetics.k_baseline                       # triggers ~off (pH 7.4, MMP very low)
        for _ in range(200):
            p, _ = m(p, ph, mmp, tw)
        self.assertAlmostEqual(p[:, 0].mean().item(), float(torch.exp(torch.tensor(-k * 0.1 * 200))), delta=1e-4)

    def test_topology_blend_changes_mass_only_via_pooling(self):
        """INFORMATIONAL: with topo weight 1, avg_pool3d zero-padding is not mass-conserving."""
        m = self.mk()
        p, ph, mmp, tw = self.inputs(0.0, rel=1.0, topo=1.0)
        out, _ = m(p, ph, mmp, tw)
        print(f"\n[payload] total mass after topo blend: {out.sum().item():.3f} vs {p.sum().item():.3f}")
        self.assertTrue(torch.isfinite(out).all())

    def test_gradients(self):
        m = self.mk()
        p, ph, mmp, tw = self.inputs(0.5, ph=6.8)
        ph.requires_grad_()
        out, met = m(p, ph, mmp, tw)
        out[:, 1].sum().backward()
        self.assertGreater(ph.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
