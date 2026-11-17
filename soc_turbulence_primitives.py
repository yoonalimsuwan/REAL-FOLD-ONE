"""C-infinity primitives and the isodiametric delta_min bound. See THEOREMS.md T2."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor

CARPET_RHO = 9.0 / 8.0      # Carpet-19 Prop 3.1 lower bound
KAPPA_FATNESS = 2.0         # design parameter (NOT a theorem constant)
A_MIN = 0.01                # design parameter


def soft_abs(x: Tensor, eps: float = 1e-10) -> Tensor:
    """C-infinity |x| = sqrt(x^2 + eps^2). Finite gradient at 0."""
    return torch.sqrt(x * x + eps * eps)


def soft_pos(x: Tensor, beta: float = 40.0) -> Tensor:
    """C-infinity max(x, 0)."""
    return F.softplus(x, beta=beta)


def soft_gate(x: Tensor, tau: float = 1e-2) -> Tensor:
    """Smooth gate in (0,1); -> Heaviside as tau -> 0."""
    return torch.sigmoid(x / tau)


def soft_clamp(x: Tensor, lo: float, hi: float) -> Tensor:
    """C-infinity clamp via tanh."""
    mid = 0.5 * (hi + lo)
    half = max(0.5 * (hi - lo), 1e-9)
    return mid + half * torch.tanh((x - mid) / half)


def soft_min(a: Tensor, b: Tensor, tau: float = 1e-3) -> Tensor:
    """C-infinity min via -tau*logsumexp(-x/tau). logsumexp is stable."""
    return -tau * torch.logsumexp(torch.stack([-a / tau, -b / tau]), dim=0)


def soft_max(a: Tensor, b: Tensor, tau: float = 1e-3) -> Tensor:
    """C-infinity max via tau*logsumexp(x/tau)."""
    return tau * torch.logsumexp(torch.stack([a / tau, b / tau]), dim=0)


def omega_ball(dim: int) -> float:
    """Volume of the unit ball in R^dim."""
    return math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0)


def delta_min_exact(n: int = 3, kappa: float = KAPPA_FATNESS,
                    a_min: float = A_MIN) -> float:
    """
    delta_min = (1/kappa) * (a_min / omega_{n-1})^{1/(n-1)}.

    THEOREMS.md T2. Conditional on kappa-fatness. Uses the NON-SHARP
    isodiametric form H^{n-1}(P) <= omega_{n-1} diam(P)^{n-1}; the sharp
    form carries a factor 2 in the diameter and would double delta_min.
    This bound is therefore conservative.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    d = n - 1
    return (1.0 / kappa) * (a_min / omega_ball(d)) ** (1.0 / d)
