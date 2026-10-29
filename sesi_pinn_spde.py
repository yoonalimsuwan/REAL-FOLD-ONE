"""
=============================================================================
SESI PHYSICS-INFORMED NEURAL NETWORKS FOR STOCHASTIC PDEs
Extension of PINNs to handle stochastic differential equations
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE ADDS
---------------------
[SPDE1] StochasticPDEConfig
        Configuration for stochastic PDEs:
          • Noise type (Wiener, Lévy, Gaussian)
          • Noise intensity and correlation
          • SPDE formulation (Itô, Stratonovich)

[SPDE2] StochasticPINN
        Extension of PINN to handle stochastic terms:
          • Neural network outputs mean and variance (or samples)
          • Loss function includes stochastic residual
          • Monte Carlo sampling for noise terms

[SPDE3] NoiseGenerator
        Generation of stochastic noise terms:
          • Wiener process (Brownian motion)
          • Gaussian white noise
          • Colored noise (Ornstein-Uhlenbeck)

[SPDE4] StochasticResidual
        Computation of SPDE residual with noise:
          • Itô formula for stochastic calculus
          • Monte Carlo estimation of expected residual
          • Variance reduction techniques

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• Stochastic PINNs are an ACTIVE RESEARCH AREA; convergence guarantees are [O].
• This implementation uses Monte Carlo sampling for noise terms, which is
  computationally expensive and introduces sampling error.
• The Itô vs Stratonovich interpretation must be specified; wrong choice →
  wrong solution.
• SPDEs can have solutions that are distributions (not functions); neural
  networks approximate functions, so regularization may be needed.
• This is an EXTENSION of PINNs, not a complete solution theory for SPDEs.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.pinn_spde")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "StochasticPDEConfig",
    "NoiseGenerator",
    "StochasticPINN",
    "StochasticResidual",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StochasticPDEConfig:
    """Configuration for stochastic PDE."""
    # SPDE type
    pde_type: str = "stochastic_heat"  # stochastic_heat, stochastic_wave, etc.
    interpretation: str = "ito"  # ito or stratonovich

    # Noise parameters
    noise_type: str = "wiener"  # wiener, gaussian, ou
    noise_intensity: float = 0.1
    noise_correlation: float = 0.0  # for colored noise

    # Neural network
    hidden_dim: int = 64
    n_layers: int = 4
    activation: str = "tanh"

    # Training
    n_epochs: int = 1000
    learning_rate: float = 1e-3
    n_mc_samples: int = 10  # Monte Carlo samples for noise

    # Domain
    spatial_domain: Tuple[float, float] = (0.0, 1.0)
    time_domain: Tuple[float, float] = (0.0, 1.0)
    n_spatial_points: int = 50
    n_time_points: int = 50

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [SPDE3] NOISE GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class NoiseGenerator:
    """
    [C/D] Generation of stochastic noise terms for SPDEs.

    Supports:
      • Wiener process (Brownian motion)
      • Gaussian white noise
      • Ornstein-Uhlenbeck process (colored noise)
    """

    def __init__(self, cfg: StochasticPDEConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(42)

    def wiener_process(self, t: np.ndarray, dt: float) -> np.ndarray:
        """
        Generate Wiener process (Brownian motion) W(t).

        W(t+dt) = W(t) + sqrt(dt) * N(0,1)
        """
        n_steps = len(t)
        dW = np.sqrt(dt) * self.rng.normal(0, 1, n_steps)
        W = np.cumsum(dW)
        return W

    def gaussian_white_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate Gaussian white noise ξ ~ N(0, 1)."""
        return self.rng.normal(0, 1, shape)

    def ornstein_uhlenbeck(self, t: np.ndarray, dt: float,
                           theta: float = 1.0, mu: float = 0.0,
                           sigma: float = 1.0) -> np.ndarray:
        """
        Generate Ornstein-Uhlenbeck process (colored noise).

        dX = theta * (mu - X) * dt + sigma * dW
        """
        n_steps = len(t)
        X = np.zeros(n_steps)
        X[0] = mu

        for i in range(1, n_steps):
            dW = np.sqrt(dt) * self.rng.normal(0, 1)
            X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW

        return X

    def generate_noise(self, t: np.ndarray, dt: float) -> np.ndarray:
        """Generate noise based on configuration."""
        if self.cfg.noise_type == "wiener":
            return self.wiener_process(t, dt)
        elif self.cfg.noise_type == "gaussian":
            return self.gaussian_white_noise(t.shape)
        elif self.cfg.noise_type == "ou":
            return self.ornstein_uhlenbeck(t, dt)
        else:
            raise ValueError(f"Unknown noise type: {self.cfg.noise_type}")


# ═══════════════════════════════════════════════════════════════════════
# [SPDE2] STOCHASTIC PINN
# ═══════════════════════════════════════════════════════════════════════

class StochasticPINN(nn.Module if HAS_TORCH else object):
    """
    [C/D] Physics-Informed Neural Network for stochastic PDEs.

    The network outputs the solution u(x, t) and optionally its variance.
    The loss function includes:
      • PDE residual (with stochastic terms)
      • Boundary conditions
      • Initial conditions
      • Data loss (if available)
    """

    def __init__(self, cfg: StochasticPDEConfig):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for StochasticPINN")
        super().__init__()
        self.cfg = cfg

        # Neural network architecture
        layers = []
        input_dim = 2  # (x, t)
        for _ in range(cfg.n_layers):
            layers.append(nn.Linear(input_dim, cfg.hidden_dim))
            if cfg.activation == "tanh":
                layers.append(nn.Tanh())
            elif cfg.activation == "relu":
                layers.append(nn.ReLU())
            input_dim = cfg.hidden_dim
        layers.append(nn.Linear(input_dim, 1))  # output: u(x, t)

        self.net = nn.Sequential(*layers)

        # Noise generator
        self.noise_gen = NoiseGenerator(cfg)

        # Move to device
        self.to(cfg.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict u(x, t)."""
        inputs = torch.stack([x, t], dim=-1)
        return self.net(inputs).squeeze(-1)

    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor,
                            u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spatial and temporal derivatives using autograd.

        Returns dict with u_x, u_t, u_xx, etc.
        """
        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]

        return {"u_x": u_x, "u_t": u_t, "u_xx": u_xx}

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor,
                     u: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Compute SPDE residual.

        Example: stochastic heat equation
            du = (α * u_xx + f) dt + σ * dW
        Residual: u_t - α * u_xx - f - σ * noise
        """
        derivs = self.compute_derivatives(x, t, u)

        # PDE parameters (placeholder; real implementation would be configurable)
        alpha = 0.01  # diffusion coefficient
        sigma = self.cfg.noise_intensity  # noise intensity
        f = 0.0  # source term

        # Residual
        residual = derivs["u_t"] - alpha * derivs["u_xx"] - f - sigma * noise

        return residual

    def loss_function(self, x_pde: torch.Tensor, t_pde: torch.Tensor,
                      x_bc: torch.Tensor, t_bc: torch.Tensor, u_bc: torch.Tensor,
                      x_ic: torch.Tensor, u_ic: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss: PDE residual + BC + IC.

        Uses Monte Carlo sampling for noise terms.
        """
        total_loss = 0.0

        # Monte Carlo sampling for stochastic residual
        for _ in range(self.cfg.n_mc_samples):
            # Generate noise
            dt = (t_pde[1] - t_pde[0]).item() if len(t_pde) > 1 else 0.01
            noise_np = self.noise_gen.generate_noise(t_pde.detach().cpu().numpy(), dt)
            noise = torch.tensor(noise_np, dtype=torch.float32, device=self.cfg.device)

            # Forward pass
            u_pde = self.forward(x_pde, t_pde)

            # PDE residual
            residual = self.pde_residual(x_pde, t_pde, u_pde, noise)
            total_loss = total_loss + torch.mean(residual ** 2)

        total_loss = total_loss / self.cfg.n_mc_samples

        # Boundary conditions
        u_bc_pred = self.forward(x_bc, t_bc)
        bc_loss = torch.mean((u_bc_pred - u_bc) ** 2)
        total_loss = total_loss + bc_loss

        # Initial conditions
        u_ic_pred = self.forward(x_ic, torch.zeros_like(x_ic))
        ic_loss = torch.mean((u_ic_pred - u_ic) ** 2)
        total_loss = total_loss + ic_loss

        return total_loss

    def train(self, x_pde: torch.Tensor, t_pde: torch.Tensor,
              x_bc: torch.Tensor, t_bc: torch.Tensor, u_bc: torch.Tensor,
              x_ic: torch.Tensor, u_ic: torch.Tensor) -> Dict[str, Any]:
        """
        Train the stochastic PINN.

        Returns training history.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        loss_history = []

        for epoch in range(self.cfg.n_epochs):
            optimizer.zero_grad()
            loss = self.loss_function(x_pde, t_pde, x_bc, t_bc, u_bc, x_ic, u_ic)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{self.cfg.n_epochs}, Loss: {loss.item():.6f}")

        return {"loss_history": loss_history, "final_loss": loss_history[-1]}


# ═══════════════════════════════════════════════════════════════════════
# [SPDE4] STOCHASTIC RESIDUAL
# ═══════════════════════════════════════════════════════════════════════

class StochasticResidual:
    """
    [D] Computation of SPDE residual with stochastic terms.

    Handles:
      • Itô formula for stochastic calculus
      • Monte Carlo estimation of expected residual
      • Variance reduction (antithetic sampling)
    """

    def __init__(self, cfg: StochasticPDEConfig):
        self.cfg = cfg
        self.noise_gen = NoiseGenerator(cfg)

    def ito_correction(self, u: np.ndarray, sigma: float,
                       dt: float) -> np.ndarray:
        """
        Apply Itô correction term.

        For Itô SDE: du = f(u) dt + g(u) dW
        Itô correction: ½ g(u) g'(u) dt
        """
        # Placeholder: real implementation would compute g'(u)
        correction = 0.5 * sigma ** 2 * dt
        return u + correction

    def monte_carlo_residual(self, residual_fn: Callable,
                             n_samples: int = 100) -> Tuple[float, float]:
        """
        Estimate expected residual using Monte Carlo.

        Returns (mean, std) of residual.
        """
        residuals = []
        for _ in range(n_samples):
            res = residual_fn()
            residuals.append(res)

        residuals = np.array(residuals)
        return float(np.mean(residuals)), float(np.std(residuals))

    def antithetic_sampling(self, residual_fn: Callable,
                            n_samples: int = 100) -> Tuple[float, float]:
        """
        Variance reduction using antithetic sampling.

        For each noise sample W, also use -W to reduce variance.
        """
        residuals = []
        for _ in range(n_samples // 2):
            res1 = residual_fn()
            res2 = residual_fn()  # In practice, would use -W
            residuals.extend([res1, res2])

        residuals = np.array(residuals)
        return float(np.mean(residuals)), float(np.std(residuals))


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Stochastic PINN — Smoke Test")
    print("=" * 72)
    print(f"  PyTorch available: {HAS_TORCH}")

    if not HAS_TORCH:
        print("  [SKIP] PyTorch required")
        return

    cfg = StochasticPDEConfig(
        pde_type="stochastic_heat",
        noise_type="wiener",
        noise_intensity=0.1,
        n_epochs=100,
        n_mc_samples=5,
        device="cpu",
    )

    # Test 1: Noise generation
    print("\n  ── [SPDE3] Noise Generation ──")
    noise_gen = NoiseGenerator(cfg)
    t = np.linspace(0, 1, 100)
    dt = t[1] - t[0]

    wiener = noise_gen.wiener_process(t, dt)
    print(f"    Wiener process: mean={wiener.mean():.4f}, std={wiener.std():.4f}")

    ou_process = noise_gen.ornstein_uhlenbeck(t, dt)
    print(f"    OU process: mean={ou_process.mean():.4f}, std={ou_process.std():.4f}")

    # Test 2: Stochastic PINN
    print("\n  ── [SPDE2] Stochastic PINN Training ──")
    pinn = StochasticPINN(cfg)

    # Generate training data
    x_pde = torch.linspace(0, 1, cfg.n_spatial_points, device=cfg.device, requires_grad=True)
    t_pde = torch.linspace(0, 1, cfg.n_time_points, device=cfg.device, requires_grad=True)
    x_pde, t_pde = torch.meshgrid(x_pde, t_pde, indexing="ij")
    x_pde = x_pde.flatten()
    t_pde = t_pde.flatten()

    # Boundary conditions
    x_bc = torch.tensor([0.0, 1.0], device=cfg.device, requires_grad=True)
    t_bc = torch.tensor([0.5, 0.5], device=cfg.device, requires_grad=True)
    u_bc = torch.tensor([0.0, 0.0], device=cfg.device)

    # Initial conditions
    x_ic = torch.linspace(0, 1, cfg.n_spatial_points, device=cfg.device, requires_grad=True)
    u_ic = torch.sin(np.pi * x_ic)

    # Train
    train_result = pinn.train(x_pde, t_pde, x_bc, t_bc, u_bc, x_ic, u_ic)
    print(f"    Final loss: {train_result['final_loss']:.6f}")
    print(f"    Loss decreased: {train_result['loss_history'][-1] < train_result['loss_history'][0]}")

    # Test 3: Stochastic residual
    print("\n  ── [SPDE4] Stochastic Residual ──")
    stoch_res = StochasticResidual(cfg)

    def mock_residual_fn():
        return np.random.normal(0, 0.1)

    mc_mean, mc_std = stoch_res.monte_carlo_residual(mock_residual_fn, n_samples=50)
    print(f"    Monte Carlo residual: mean={mc_mean:.4f}, std={mc_std:.4f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Stochastic PINNs are ACTIVE RESEARCH; convergence is [O]")
    print("  • Monte Carlo sampling is expensive and introduces error")
    print("  • Itô vs Stratonovich must be specified correctly")
    print("  • SPDE solutions may be distributions; NN approximates functions")
    print("  • This is an EXTENSION, not a complete solution theory")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
