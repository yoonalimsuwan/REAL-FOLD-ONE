"""
OPMC design parameterisation. theta in R^{d(m,n)} -> decoder -> control field.

This is the ONLY legitimate place for OPMC on the gradient path: it
parameterises the CONTROL, so gradients w.r.t. theta are EXACT.
Compressing the adjoint tape would corrupt gradients — not done here.
See THEOREMS.md T4.
"""
from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


def opmc_d_mn(m: int = 4, n: int = 4) -> int:
    """d(m,n) = m^2 n^2 + m n^2.  m=n=4 -> 320."""
    return m * m * n * n + m * n * n


def opmc_k_star(eps: float = 1e-8, D0: float = 1.0, L: float = 0.438520) -> int:
    """
    k* = ceil(ln(eps/D0)/ln L). Bounds ITERATION COUNT of a contraction on
    a space of FIXED dimension. Does NOT bound total cost. THEOREMS.md T4.
    """
    if not (0 < L < 1):
        raise ValueError("contraction requires 0 < L < 1")
    return int(math.ceil(math.log(eps / D0) / math.log(L)))


class OPMCDesign(nn.Module):
    """theta (d_mn,) -> per-cell gain field. Gradients exact via the decoder."""

    def __init__(self, grid_shape: Tuple[int, int, int], m: int = 4, n: int = 4,
                 seed: int = 0):
        super().__init__()
        self.d_mn = opmc_d_mn(m, n)
        self.shape = tuple(grid_shape)
        n_out = self.shape[0] * self.shape[1] * self.shape[2]
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(self.d_mn, n_out, generator=gen,
                                              dtype=torch.float64)
                             / math.sqrt(self.d_mn))
        self.theta = nn.Parameter(torch.zeros(self.d_mn, dtype=torch.float64))
        self.k_star = opmc_k_star()

    def forward(self) -> Tensor:
        return (self.theta @ self.W.to(self.theta.dtype)).reshape(self.shape)
