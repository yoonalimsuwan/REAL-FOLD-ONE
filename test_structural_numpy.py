"""Torch-free ground-truth checks for the structural-biology files (NumPy transcriptions of the
original and fixed formulas). Run: python -m unittest tests_math.test_structural_numpy -v"""
import unittest
import numpy as np


def rot(axis, th):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def kabsch(p, t, fixed=True):
    """Transcription of compute_kabsch_rmsd; fixed=False reproduces the original `p @ R`."""
    p = p - p.mean(0); t = t - t.mean(0)
    U, S, Vh = np.linalg.svd(p.T @ t); V = Vh.T
    d = np.linalg.det(V @ U.T)
    R = V @ np.diag([1, 1, 1.0 if d >= 0 else -1.0]) @ U.T
    prot = p @ (R.T if fixed else R)
    return np.sqrt(((prot - t) ** 2).sum(-1).mean() + 1e-12), R


class TestKabsch(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.P = self.rng.normal(size=(40, 3))

    def test_original_is_wrong_on_rigid_copy(self):
        T = self.P @ rot([0.3, 1, 0.2], 1.1).T + 5.0
        self.assertGreater(kabsch(self.P, T, fixed=False)[0], 1.0)      # observed ~2.4 A

    def test_fixed_zero_rmsd_for_any_rigid_copy(self):
        for th in (0.05, 0.3, 1.1, 2.5, 3.0):
            T = self.P @ rot(self.rng.normal(size=3), th).T + self.rng.normal(size=3)
            self.assertLess(kabsch(self.P, T)[0], 1e-5, th)

    def test_proper_rotation_even_for_mirror_image(self):
        T = self.P * np.array([1, 1, -1.0])                              # chirality-flipped
        rmsd, R = kabsch(self.P, T)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)          # never a reflection
        self.assertGreater(rmsd, 0.1)                                    # mirror image is NOT a match

    def test_translation_and_symmetry(self):
        T = self.P + np.array([10.0, -3.0, 7.0])
        self.assertLess(kabsch(self.P, T)[0], 1e-5)
        U = self.P + 0.2 * self.rng.normal(size=self.P.shape)
        self.assertAlmostEqual(kabsch(self.P, U)[0], kabsch(U, self.P)[0], places=6)


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    return np.arctan2((n1 * b2).sum() * np.linalg.norm(b1), (n1 * n2).sum())


class TestRestraints(unittest.TestCase):
    def test_dihedral_matches_prescribed_angle(self):
        for deg in (0, 60, 90, 180, -60, -120):
            phi = np.deg2rad(deg)
            p1, p2 = np.zeros(3), np.array([0, 0, 1.5])
            p0, p3 = p1 + [1.0, 0, -0.5], p2 + [np.cos(phi), np.sin(phi), 0.5]
            d = dihedral(p0, p1, p2, p3)
            self.assertAlmostEqual(np.rad2deg(np.arctan2(np.sin(d - phi), np.cos(d - phi))), 0, places=6)

    def test_periodic_difference_wraps(self):
        wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
        self.assertAlmostEqual(np.rad2deg(wrap(np.deg2rad(179 - -179))), -2.0, places=9)

    def test_noe_flat_bottom(self):
        f, lo, hi = 50.0, 1.8, 5.0
        pen = lambda d: f * (max(lo - d, 0) + max(d - hi, 0)) ** 2
        self.assertEqual(pen(3.0), 0.0)
        self.assertAlmostEqual(pen(6.0), 50.0, places=9)
        self.assertAlmostEqual(pen(1.0), f * 0.64, places=9)


def fsc_mean(a, b, axes, n_shells=20):
    D, H, W = a.shape
    Fa, Fb = np.fft.fftn(a, axes=axes), np.fft.fftn(b, axes=axes)
    f = [np.fft.fftfreq(n) for n in (D, H, W)]
    r = np.sqrt(f[0][:, None, None] ** 2 + f[1][None, :, None] ** 2 + f[2][None, None, :] ** 2).ravel()
    edges = np.linspace(0, 0.5 * np.sqrt(3), n_shells + 1)
    idx = np.clip(np.searchsorted(edges, r, side="left") - 1, 0, n_shells - 1)
    fa, fb = Fa.ravel(), Fb.ravel()
    num = np.bincount(idx, (fa * np.conj(fb)).real, n_shells)
    da, db = np.bincount(idx, np.abs(fa) ** 2, n_shells), np.bincount(idx, np.abs(fb) ** 2, n_shells)
    return (num / np.sqrt(da * db + 1e-8)).mean()


class TestFSC(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1); n = 32; fr = np.fft.fftfreq(n)
        r2 = fr[:, None, None] ** 2 + fr[None, :, None] ** 2 + fr[None, None, :] ** 2
        self.g = np.fft.ifftn(np.fft.fftn(rng.normal(size=(n, n, n))) * np.exp(-r2 / 0.5)).real

    def test_identical_maps_give_one(self):
        self.assertAlmostEqual(fsc_mean(self.g, self.g, (0, 1, 2)), 1.0, places=6)

    def test_skipping_H_axis_misreports_H_shift(self):
        shifted = np.roll(self.g, 2, axis=1)
        true3d = fsc_mean(self.g, shifted, (0, 1, 2))
        buggy = fsc_mean(self.g, shifted, (0, 2))                         # original dim=(-3,-1)
        self.assertAlmostEqual(true3d, 0.28, delta=0.05)
        self.assertGreater(abs(true3d - buggy), 0.2)

    def test_fsc_decreases_with_noise(self):
        rng = np.random.default_rng(2)
        vals = [fsc_mean(self.g, self.g + s * self.g.std() * rng.normal(size=self.g.shape), (0, 1, 2))
                for s in (0.1, 0.5, 2.0)]
        self.assertTrue(vals[0] > vals[1] > vals[2])


class TestRice(unittest.TestCase):
    @staticmethod
    def pdf(fo, fc, d, s):
        x = 2 * fo * d * fc / s
        return (2 * fo / s) * np.exp(-(fo ** 2 + (d * fc) ** 2) / s) * np.i0(x)

    def test_normalised_for_several_models(self):
        fo = np.linspace(0, 40, 400001)
        for fc, d, s in ((0.0, 0.8, 4.0), (3.0, 0.9, 2.0), (6.0, 0.7, 5.0)):
            self.assertAlmostEqual(np.trapezoid(self.pdf(fo, fc, d, s), fo), 1.0, places=5)

    def test_reduces_to_rayleigh_when_model_empty(self):
        s = 4.0; fo = np.linspace(0, 40, 400001)
        mean = np.trapezoid(fo * self.pdf(fo, 0.0, 0.8, s), fo)
        self.assertAlmostEqual(mean, np.sqrt(np.pi * s) / 2, places=4)

    def test_gaussian_least_squares_is_not_rice(self):
        """0.5*((Fo-Fc)/sigma)^2 has its minimum at Fc=Fo; Rice (D<1) does not."""
        fo, d, s = 5.0, 0.8, 2.0
        fc = np.linspace(0.1, 15, 20000)
        nll = -np.log(self.pdf(fo, fc, d, s))
        self.assertGreater(abs(fc[np.argmin(nll)] - fo), 0.3)


if __name__ == "__main__":
    unittest.main()
