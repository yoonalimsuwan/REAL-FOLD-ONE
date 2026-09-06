# =============================================================================
# Unified Structural CSOC-Lorenz Dynamics
# =============================================================================
#
# Developer  : PAI AND Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788
# GitHub     : https://github.com/yoonalimsuwan
# Email      : msps4u@gmail.com
#
# =============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedCSOCLorenzDynamics(nn.Module):
    """
    Production-grade, fully differentiable PyTorch module for Unified Structural 
    CSOC-Lorenz Dynamics, featuring 8th-order polyharmonic regularization and SSC control[span_3](start_span)[span_3](end_span).
    """
    def __init__(self, grid_size: int = 64, dim: int = 2):
        super().__init__()
        self.grid_size = grid_size
        self.dim = dim

        # Learnable parameters for CSOC and Kernel tuning
        self.alpha = nn.Parameter(torch.tensor(2.8)) # Universal tuning parameter alpha ~ 2.8[span_4](start_span)[span_4](end_span)
        self.length_scale = nn.Parameter(torch.tensor(1.0)) # Interaction length scale l
        self.epsilon = 1e-4

        # SSC Control Coefficients (initialized near target fixed-point values)[span_5](start_span)[span_5](end_span)
        self.a1 = nn.Parameter(torch.tensor(3.0e-4))
        self.a2 = nn.Parameter(torch.tensor(1.0e-3))
        self.a3 = nn.Parameter(torch.tensor(1.0e-3))
        self.a4 = nn.Parameter(torch.tensor(3.0e-4))

        # Lorenz system coupling parameters
        self.sigma = nn.Parameter(torch.tensor(10.0))
        self.rho = nn.Parameter(torch.tensor(28.0))
        self.beta_lorenz = nn.Parameter(torch.tensor(8.0 / 3.0))

    def compute_polyharmonic_8th(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Computes the 8th-order polyharmonic operator delta^4(Y)[span_6](start_span)[span_6](end_span) efficiently 
        in the Fourier domain for periodic/bounded topologies.
        """
        # Transform to spectral domain
        Y_fft = torch.fft.fftn(Y, dim=(-2, -1))
        
        # Construct frequency coordinates
        device = Y.device
        freqs = [torch.fft.fftfreq(s, d=1.0, device=device) for s in Y.shape[-2:]]
        grid_freqs = torch.meshgrid(*freqs, indexing='ij')
        
        # Calculate squared wave number magnitude |xi|^2
        xi_sq = sum(f**2 for f in grid_freqs)
        
        # 8th-order operator symbol in Fourier space: (-|xi|^2)^4 = |xi|^8
        symbol = (xi_sq ** 4)
        
        # Apply operator and invert FFT (fully differentiable)
        Y_poly_fft = Y_fft * symbol
        return torch.fft.ifftn(Y_poly_fft, dim=(-2, -1)).real

    def compute_ssc_control(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the Semantic-State Contraction (SSC) control law[span_7](start_span)[span_7](end_span):
        SSC(X) = a1 * <X> + a2 * Delta X + a3 * |nabla X|^2 - a4 * X
        """
        # Mean configuration <X>
        mean_X = torch.mean(X, dim=(-2, -1), keepdim=True)

        # Spatial gradients using central differences
        grad_y, grad_x = torch.gradient(X, dim=(-2, -1))
        grad_sq_mag = grad_x**2 + grad_y**2

        # Laplacian Delta X via divergence of gradients
        _, lap_x = torch.gradient(grad_x, dim=(-1,))
        lap_y, _ = torch.gradient(grad_y, dim=(-2,))
        delta_X = lap_x + lap_y

        # SSC Control formulation[span_8](start_span)[span_8](end_span)
        ssc_term = (
            self.a1 * mean_X +
            self.a2 * delta_X +
            self.a3 * grad_sq_mag -
            self.a4 * X
        )
        return ssc_term

    def forward(self, state: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Performs a single continuous-time step update of the coupled (X, Y, Z) system[span_9](start_span)[span_9](end_span).
        Input state tensor shape: (Batch, 3, Height, Width) where channels represent X, Y, Z.
        """
        X = state[:, 0:1, :, :]
        Y = state[:, 1:2, :, :]
        Z = state[:, 2:3, :, :]

        # 1. Avalanche State Evolution (X)[span_10](start_span)[span_10](end_span)
        dX_dt = self.sigma * (Y - X)

        # 2. Fractal Interface Strain (Y) regularized by 8th-order polyharmonic operator[span_11](start_span)[span_11](end_span)
        poly_Y = self.compute_polyharmonic_8th(Y)
        dY_dt = X * (self.rho - Z) - Y - poly_Y

        # 3. Learnable CSOC Parameter Control (Z) with SSC[span_12](start_span)[span_12](end_span)
        ssc_control = self.compute_ssc_control(X)
        dZ_dt = X * Y - self.beta_lorenz * Z + ssc_control

        # Pack derivatives
        dstate_dt = torch.cat([dX_dt, dY_dt, dZ_dt], dim=1)

        # Explicit Euler update step (fully differentiable graph retention)
        next_state = state + dstate_dt * dt
        return next_state
