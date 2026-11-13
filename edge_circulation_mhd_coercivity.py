#!/usr/bin/env python3
"""
=============================================================================
EDGE-CIRCULATION MHD INDUCTION SOLVER
+ FULL GRAPH-NORM COERCIVITY VERIFIER FOR ORDER 6 AND ORDER 8
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

PART A — EDGE-CIRCULATION MHD
  State variable is the edge circulation of the magnetic vector potential:
      A_e = ∫_e A · dl
  Magnetic flux through each face is the discrete curl:
      B_f = (dA)_f = Σ_{e ∈ ∂f} orientation(e,f) A_e
  Therefore:
      div B = d B = d² A = 0
  exactly, by the discrete identity d² = 0.

  Faraday / induction update:
      A_e^{n+1} = A_e^n - dt E_e
  where E_e is the edge circulation of the electric field.
  MHD electric field:
      E = -u × B + η J,   J = ∇ × B

PART B — FULL GRAPH-NORM COERCIVITY, ORDER 6 AND 8
  Let A_op = -Δ_clamped be positive self-adjoint with compact resolvent and
  trivial kernel. Let λ₁ > 0 be its first eigenvalue.

  For m = 3 (sixth order, Δ³) and m = 4 (eighth order, Δ⁴):
      ||A_op^j u||² <= λ₁^{-2(m-j)} ||A_op^m u||²,  j = 0,...,m

  Hence:
      ||A_op^m u||² >= α_m Σ_{j=0}^m ||A_op^j u||²
      α_m = (Σ_{r=0}^m λ₁^{-2r})^{-1}

  This is a spectral proof of full graph-norm coercivity, conditional on
  positivity / trivial kernel of the clamped realization.

HONEST LIMITS
  ❌ The MHD solver is explicit, periodic, first-order in time, and does not
     include harmonic magnetic flux cohomology on the torus.
  ❌ The discrete MHD operators are not a fully mimetic Hodge-star scheme;
     energy monotonicity is not guaranteed by construction.
  ❌ The coercivity verifier tests discrete Euclidean surrogate operators.
     For the Sierpinski gasket, the analytic step still requires the
     appropriate clamped domain and kernel triviality.
=============================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

VERSION = "1.0.0"


# =============================================================================
# PART A — EDGE-CIRCULATION MHD SOLVER
# =============================================================================

@dataclass
class EdgeMHDCfg:
    nx: int = 32
    ny: int = 32
    nz: int = 32
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    eta: float = 0.0          # resistivity
    device: str = "cpu"
    dtype: str = "float64"

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


class EdgeCirculationMHDSolver:
    """
    3D periodic MHD induction solver with edge-circulation state A_e.

    Variables
      Ax, Ay, Az : edge circulations of vector potential A
      Bx, By, Bz : magnetic fluxes on faces, B = dA

    Discrete identities
      B  = dA
      dB = d²A = 0   -> magnetic divergence is identically zero
    """

    def __init__(self, cfg: Optional[EdgeMHDCfg] = None):
        self.cfg = cfg or EdgeMHDCfg()
        self.dev = self.cfg.torch_device
        self.dt_ = self.cfg.torch_dtype

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def zeros_A(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cfg
        sh = (c.nx, c.ny, c.nz)
        return (torch.zeros(sh, device=self.dev, dtype=self.dt_),
                torch.zeros(sh, device=self.dev, dtype=self.dt_),
                torch.zeros(sh, device=self.dev, dtype=self.dt_))

    def smooth_initial_A(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Smooth periodic vector potential mode.
        Gives a nontrivial divergence-free magnetic field.
        """
        c = self.cfg
        x = torch.arange(c.nx, device=self.dev, dtype=self.dt_).view(-1, 1, 1) * c.dx
        y = torch.arange(c.ny, device=self.dev, dtype=self.dt_).view(1, -1, 1) * c.dy
        z = torch.arange(c.nz, device=self.dev, dtype=self.dt_).view(1, 1, -1) * c.dz

        Lx = c.nx * c.dx
        Ly = c.ny * c.dy

        Ax = torch.zeros(c.nx, c.ny, c.nz, device=self.dev, dtype=self.dt_)
        Ay = torch.zeros(c.nx, c.ny, c.nz, device=self.dev, dtype=self.dt_)

        Az = (torch.sin(2.0 * math.pi * x / Lx)
              * torch.sin(2.0 * math.pi * y / Ly))
        Az = Az.expand(c.nx, c.ny, c.nz).clone()
        return Ax, Ay, Az

    # ------------------------------------------------------------------
    # discrete exterior calculus operators
    # ------------------------------------------------------------------
    def curl_A_to_B(self, A: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        B = dA.

        Face orientations:
          Bx = Ay(i,j,k) + Az(i,j+1,k) - Ay(i,j,k+1) - Az(i,j,k)
          By = Az(i,j,k) + Ax(i,j,k+1) - Az(i+1,j,k) - Ax(i,j,k)
          Bz = Ax(i,j,k) + Ay(i+1,j,k) - Ax(i,j+1,k) - Ay(i,j,k)
        """
        Ax, Ay, Az = A

        Bx = (Ay
              + torch.roll(Az, shifts=-1, dims=1)
              - torch.roll(Ay, shifts=-1, dims=2)
              - Az)

        By = (Az
              + torch.roll(Ax, shifts=-1, dims=2)
              - torch.roll(Az, shifts=-1, dims=0)
              - Ax)

        Bz = (Ax
              + torch.roll(Ay, shifts=-1, dims=0)
              - torch.roll(Ax, shifts=-1, dims=1)
              - Ay)

        return Bx, By, Bz

    def divergence_B(self, B: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                     ) -> torch.Tensor:
        """
        dB on cells. For B = dA this must vanish identically.
        """
        Bx, By, Bz = B
        return (torch.roll(Bx, shifts=-1, dims=0) - Bx
                + torch.roll(By, shifts=-1, dims=1) - By
                + torch.roll(Bz, shifts=-1, dims=2) - Bz)

    def B_cell_center(self, B: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Average face fluxes to cell centers."""
        Bx, By, Bz = B
        Bxc = 0.5 * (Bx + torch.roll(Bx, shifts=-1, dims=0))
        Byc = 0.5 * (By + torch.roll(By, shifts=-1, dims=1))
        Bzc = 0.5 * (Bz + torch.roll(Bz, shifts=-1, dims=2))
        return Bxc, Byc, Bzc

    def curl_B_cell_center(self, Bc: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cell-centered J = curl B using central differences.
        """
        Bx, By, Bz = Bc
        dx, dy, dz = self.cfg.dx, self.cfg.dy, self.cfg.dz

        def ddx(f):
            return (torch.roll(f, -1, 0) - torch.roll(f, 1, 0)) / (2.0 * dx)

        def ddy(f):
            return (torch.roll(f, -1, 1) - torch.roll(f, 1, 1)) / (2.0 * dy)

        def ddz(f):
            return (torch.roll(f, -1, 2) - torch.roll(f, 1, 2)) / (2.0 * dz)

        Jx = ddy(Bz) - ddz(By)
        Jy = ddz(Bx) - ddx(Bz)
        Jz = ddx(By) - ddy(Bx)
        return Jx, Jy, Jz

    # ------------------------------------------------------------------
    # electric field on edges
    # ------------------------------------------------------------------
    def electric_field_edges(
        self,
        A: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        u: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute edge circulations of E = -u × B + η J.
        """
        B = self.curl_A_to_B(A)
        Bxc, Byc, Bzc = self.B_cell_center(B)
        Jx, Jy, Jz = self.curl_B_cell_center((Bxc, Byc, Bzc))

        ux, uy, uz = u
        eta = self.cfg.eta

        # E = -u × B + η J
        Ex = -uy * Bzc + uz * Byc + eta * Jx
        Ey = -uz * Bxc + ux * Bzc + eta * Jy
        Ez = -ux * Byc + uy * Bxc + eta * Jz

        # Interpolate cell-centered E to primal edges.
        # Ax-edge at (i+1/2, j, k): average over four cells around y,z.
        Ex_e = 0.25 * (
            Ex
            + torch.roll(Ex, shifts=1, dims=1)
            + torch.roll(Ex, shifts=1, dims=2)
            + torch.roll(Ex, shifts=(1, 1), dims=(1, 2))
        )

        # Ay-edge at (i, j+1/2, k): average over four cells around x,z.
        Ey_e = 0.25 * (
            Ey
            + torch.roll(Ey, shifts=1, dims=0)
            + torch.roll(Ey, shifts=1, dims=2)
            + torch.roll(Ey, shifts=(1, 1), dims=(0, 2))
        )

        # Az-edge at (i, j, k+1/2): average over four cells around x,y.
        Ez_e = 0.25 * (
            Ez
            + torch.roll(Ez, shifts=1, dims=0)
            + torch.roll(Ez, shifts=1, dims=1)
            + torch.roll(Ez, shifts=(1, 1), dims=(0, 1))
        )

        return Ex_e, Ey_e, Ez_e

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def step(
        self,
        A: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        u: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        A^{n+1} = A^n - dt E^n.
        B remains divergence-free because B = dA at every step.
        """
        Ex, Ey, Ez = self.electric_field_edges(A, u)
        Ax, Ay, Az = A
        return (Ax - dt * Ex, Ay - dt * Ey, Az - dt * Ez)

    def magnetic_energy(self, A: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                        ) -> float:
        B = self.curl_A_to_B(A)
        Bxc, Byc, Bzc = self.B_cell_center(B)
        vol = self.cfg.dx * self.cfg.dy * self.cfg.dz
        e = 0.5 * vol * (
            torch.sum(Bxc * Bxc) + torch.sum(Byc * Byc) + torch.sum(Bzc * Bzc)
        )
        return float(e)

    def diagnostics(self, A: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    u: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                    ) -> Dict[str, float]:
        B = self.curl_A_to_B(A)
        divB = self.divergence_B(B)
        return {
            "magnetic_energy": self.magnetic_energy(A),
            "max_div_B": float(divB.abs().max()),
            "mean_abs_Bx": float(B[0].abs().mean()),
            "mean_abs_By": float(B[1].abs().mean()),
            "mean_abs_Bz": float(B[2].abs().mean()),
        }


# =============================================================================
# PART B — FULL GRAPH-NORM COERCIVITY VERIFIER
# =============================================================================

def dirichlet_laplacian_1d(n: int, h: float = 1.0) -> np.ndarray:
    """
    Positive 1D Dirichlet Laplacian A = -d²/dx².
    This is a surrogate clamped positive operator for testing the spectral
    coercivity argument.
    """
    main = 2.0 / (h * h) * np.eye(n, dtype=np.float64)
    off = -1.0 / (h * h) * (np.eye(n, k=1, dtype=np.float64)
                            + np.eye(n, k=-1, dtype=np.float64))
    return main + off


def dirichlet_laplacian_2d(n: int, h: float = 1.0) -> np.ndarray:
    """
    Positive 2D Dirichlet Laplacian via Kronecker sum.
    """
    L1 = dirichlet_laplacian_1d(n, h)
    I = np.eye(n, dtype=np.float64)
    return np.kron(L1, I) + np.kron(I, L1)


class FullGraphCoercivityVerifier:
    """
    Verifies:
        ||A^m u||² >= α_m Σ_{j=0}^m ||A^j u||²

    for positive self-adjoint A.

    Order convention:
        m = 3  -> sixth order  (A³ ~ (-Δ)³)
        m = 4  -> eighth order (A⁴ ~ (-Δ)⁴)
    """

    def __init__(self, A: np.ndarray):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if not np.allclose(A, A.T, atol=1e-12):
            raise ValueError("A must be symmetric")
        self.A = A.astype(np.float64)
        self.eigvals = np.linalg.eigvalsh(self.A)
        if self.eigvals[0] <= 0.0:
            raise ValueError("A must be positive definite for this coercivity form")

    def alpha_theory(self, m: int) -> float:
        """
        α_m = (Σ_{r=0}^m λ₁^{-2r})^{-1}
            = λ₁^{2m} / Σ_{j=0}^m λ₁^{2j}
        """
        lam1 = float(self.eigvals[0])
        s = sum(lam1 ** (-2 * r) for r in range(m + 1))
        return 1.0 / s

    def alpha_spectrum_exact(self, m: int) -> float:
        """
        Exact minimum of the Rayleigh quotient over eigenvectors:
            min_λ λ^{2m} / Σ_{j=0}^m λ^{2j}
        """
        alphas = []
        for lam in self.eigvals:
            lam = float(lam)
            den = sum(lam ** (2 * j) for j in range(m + 1))
            alphas.append(lam ** (2 * m) / den)
        return float(min(alphas))

    def empirical_min_ratio(self, m: int, trials: int = 32,
                            seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        n = self.A.shape[0]
        best = math.inf
        for _ in range(trials):
            u = rng.standard_normal(n)
            w = u.copy()
            graph = float(np.dot(w, w))
            top = 0.0
            for j in range(1, m + 1):
                w = self.A @ w
                n2 = float(np.dot(w, w))
                graph += n2
                if j == m:
                    top = n2
            if graph > 0:
                best = min(best, top / graph)
        return float(best)

    def verify(self, m: int, trials: int = 32) -> Dict[str, float]:
        a_theory = self.alpha_theory(m)
        a_exact = self.alpha_spectrum_exact(m)
        a_emp = self.empirical_min_ratio(m, trials=trials)
        return {
            "m": m,
            "order": 2 * m,
            "lambda_1": float(self.eigvals[0]),
            "alpha_theory": a_theory,
            "alpha_spectrum_exact": a_exact,
            "alpha_empirical_lower_bound": a_emp,
            "consistent": bool(a_emp + 1e-10 >= min(a_theory, a_exact)),
        }


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 84)
    print(f"  EDGE-CIRCULATION MHD + FULL GRAPH-NORM COERCIVITY v{VERSION}")
    print("=" * 84)
    ok = True

    def rep(name: str, passed: bool, extra: str = ""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ------------------------------------------------------------------
    # MHD tests
    # ------------------------------------------------------------------
    print("\n[A] Edge-circulation MHD")
    cfg = EdgeMHDCfg(nx=16, ny=16, nz=16, eta=0.0, dtype="float64")
    solver = EdgeCirculationMHDSolver(cfg)
    A = solver.smooth_initial_A()
    zeros_u = solver.zeros_A()

    B = solver.curl_A_to_B(A)
    divB = solver.divergence_B(B)
    rep("B = dA shapes",
        all(b.shape == (16, 16, 16) for b in B))
    rep("div B = d²A = 0 (exact structure)",
        float(divB.abs().max()) < 1e-12,
        f"max|divB|={float(divB.abs().max()):.2e}")

    # ideal, no velocity: A unchanged
    A2 = solver.step(A, zeros_u, dt=1e-3)
    same = all(torch.allclose(a, b, atol=1e-14) for a, b in zip(A, A2))
    rep("ideal E=0 leaves A unchanged", same)

    # resistive step stays finite and div-free
    cfg_r = EdgeMHDCfg(nx=16, ny=16, nz=16, eta=1e-2, dtype="float64")
    solver_r = EdgeCirculationMHDSolver(cfg_r)
    A_r = solver_r.smooth_initial_A()
    e0 = solver_r.magnetic_energy(A_r)
    for _ in range(5):
        A_r = solver_r.step(A_r, zeros_u, dt=1e-4)
    e1 = solver_r.magnetic_energy(A_r)
    B_r = solver_r.curl_A_to_B(A_r)
    div_r = solver_r.divergence_B(B_r)
    rep("resistive step finite",
        math.isfinite(e0) and math.isfinite(e1))
    rep("resistive step remains div-free",
        float(div_r.abs().max()) < 1e-12,
        f"max|divB|={float(div_r.abs().max()):.2e}")
    print(f"      energy: {e0:.6e} -> {e1:.6e} "
          f"({'decay' if e1 < e0 else 'no guaranteed monotone decay'})")

    # ------------------------------------------------------------------
    # Coercivity tests
    # ------------------------------------------------------------------
    print("\n[B] Full graph-norm coercivity")

    # 1D surrogate clamped positive operator
    n = 32
    h = 1.0 / (n + 1)
    A1 = dirichlet_laplacian_1d(n, h=h)
    ver1 = FullGraphCoercivityVerifier(A1)

    for m in (3, 4):
        r = ver1.verify(m, trials=32)
        print(f"      1D order {r['order']}: λ1={r['lambda_1']:.6e}, "
              f"α_theory={r['alpha_theory']:.6e}, "
              f"α_exact={r['alpha_spectrum_exact']:.6e}, "
              f"α_emp≥={r['alpha_empirical_lower_bound']:.6e}")
        rep(f"1D order {r['order']} full-graph coercivity",
            r["consistent"] and r["alpha_theory"] > 0.0)

    # 2D surrogate clamped positive operator
    n2 = 8
    h2 = 1.0 / (n2 + 1)
    A2 = dirichlet_laplacian_2d(n2, h=h2)
    ver2 = FullGraphCoercivityVerifier(A2)

    for m in (3, 4):
        r = ver2.verify(m, trials=16)
        print(f"      2D order {r['order']}: λ1={r['lambda_1']:.6e}, "
              f"α_theory={r['alpha_theory']:.6e}, "
              f"α_exact={r['alpha_spectrum_exact']:.6e}, "
              f"α_emp≥={r['alpha_empirical_lower_bound']:.6e}")
        rep(f"2D order {r['order']} full-graph coercivity",
            r["consistent"] and r["alpha_theory"] > 0.0)

    # ------------------------------------------------------------------
    # Theoretical statement
    # ------------------------------------------------------------------
    print("\n[C] Theorem encoded by the verifier")
    print("""
      Let A = -Δ_clamped be positive self-adjoint with compact resolvent and
      ker(A) = {0}. Let λ1 > 0 be the first eigenvalue. For m = 3 or 4,

          ||A^m u||² >= α_m Σ_{j=0}^m ||A^j u||²,

      where

          α_m = (Σ_{r=0}^m λ1^{-2r})^{-1}
              = λ1^{2m} / Σ_{j=0}^m λ1^{2j}.

      This controls the FULL graph norm up to order 2m, including odd powers
      A¹ and A³, not only the even-energy norm.

      For SG / fractafold use, the remaining analytic obligation is to show
      that the relevant clamped realization is positive with trivial kernel.
      Once that is established, the spectral argument above closes order 6
      and order 8 coercivity.
""")

    print("=" * 84)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 84)
    return ok


def main():
    raise SystemExit(0 if self_test() else 1)


if __name__ == "__main__":
    main()
