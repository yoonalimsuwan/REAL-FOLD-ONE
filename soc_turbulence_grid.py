"""Periodic cell-centred grid and strain-rate invariant."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import Tensor


@dataclass
class Grid:
    nx: int = 32
    ny: int = 32
    nz: int = 32
    Lx: float = 2 * math.pi
    Ly: float = 2 * math.pi
    Lz: float = 2 * math.pi
    device: str = "cpu"
    dtype: str = "float64"

    @property
    def h(self) -> Tuple[float, float, float]:
        return (self.Lx / self.nx, self.Ly / self.ny, self.Lz / self.nz)

    @property
    def delta(self) -> float:
        """Filter width Delta = (dx dy dz)^(1/3)."""
        dx, dy, dz = self.h
        return (dx * dy * dz) ** (1.0 / 3.0)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32

    def coords(self, device=None):
        dev = device or self.device
        ax = [(torch.arange(n, device=dev, dtype=self.torch_dtype) + 0.5) * hh
              for n, hh in zip(self.shape, self.h)]
        return torch.meshgrid(*ax, indexing="ij")


def grad_cc(f: Tensor, g: Grid) -> Tensor:
    """Cell-centred gradient, periodic. (nx,ny,nz) -> (3,nx,ny,nz)."""
    return torch.stack([(torch.roll(f, -1, ax) - torch.roll(f, 1, ax))
                        / (2.0 * g.h[ax]) for ax in range(3)], 0)


def strain_rate(u: Tensor, g: Grid) -> Tuple[Tensor, Tensor]:
    """
    u : (3, nx, ny, nz) cell-centred velocity.
    Returns (|S|, S) with S_ij = 0.5(du_i/dx_j + du_j/dx_i) and
    |S| = sqrt(2 S:S).  Therefore  S:S = 0.5 |S|^2  (used by WALE).
    """
    if u.dim() != 4 or u.shape[0] != 3:
        raise ValueError("u must be (3, nx, ny, nz)")
    gu = torch.stack([grad_cc(u[c], g) for c in range(3)], 0)  # gu[i,j]=du_i/dx_j
    S = 0.5 * (gu + gu.transpose(0, 1))
    mag = torch.sqrt(2.0 * (S * S).sum(dim=(0, 1)) + 1e-24)
    return mag, S
