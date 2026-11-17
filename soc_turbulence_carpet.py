"""
Carpet-like graph Laplacian, multi-scale energy PRIORITY, kappa-fatness gate.

HONEST: the graph is carpet-LIKE (centre cells removed from a finite grid).
This is a multi-scale priority weight, NOT a resistance form on the
Sierpinski carpet and NOT a Mosco-convergence proof. See THEOREMS.md T3.
"""
from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .primitives import CARPET_RHO, KAPPA_FATNESS, A_MIN, soft_gate, delta_min_exact


def build_carpet_laplacian(side: int = 9) -> Tensor:
    """Graph Laplacian on a carpet-LIKE grid: centre cell of each 3x3 removed."""
    side = max(3, int(side))
    n = side * side
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < side and 0 <= jj < side):
                    continue
                if ii % 3 == 1 and jj % 3 == 1:
                    continue
                A[i * side + j, ii * side + jj] = 1.0
    return torch.diag(A.sum(1)) - A


class CarpetAMRIndicator(nn.Module):
    """
    Refinement score combines THREE signals:
      (a) DNS necessity : Delta > c_eta*eta        (primary, physical)
      (b) Carpet energy : rho^m graph energy       (priority, OURS)
      (c) kappa-fatness : reject thin slivers      (guard, OURS)

    [B2 FIX] proj is Linear(n_nodes, n_nodes) acting on the pooled node
    VECTOR, so multiscale_energy returns a SCALAR (the previous
    Linear(1, n_nodes) produced a matrix and silently broke the sum).
    """

    def __init__(self, side: int = 9, n_levels: int = 3,
                 kappa: float = KAPPA_FATNESS, spatial_dim: int = 3):
        super().__init__()
        L = build_carpet_laplacian(side)
        self.register_buffer("L", L)
        self.n_nodes = L.shape[0]
        self.n_levels = int(n_levels)
        self.kappa = float(kappa)
        self.delta_min = delta_min_exact(spatial_dim, kappa, A_MIN)
        self.log_rho_raw = nn.Parameter(torch.tensor(math.log(1.25 - CARPET_RHO)))
        # [B2] vector -> vector, NOT scalar -> vector
        self.proj = nn.ModuleList([nn.Linear(self.n_nodes, self.n_nodes)
                                   for _ in range(self.n_levels)])
        self.log_thresh = nn.Parameter(torch.tensor(math.log(0.1)))

    @property
    def rho(self) -> Tensor:
        return CARPET_RHO + F.softplus(self.log_rho_raw)

    def _pool_to_nodes(self, field: Tensor) -> Tensor:
        """Block-average a 3-D field onto the carpet node VECTOR (n_nodes,)."""
        flat = field.reshape(-1)
        n = flat.numel()
        grp = (torch.arange(n, device=flat.device) * self.n_nodes // n)
        num = torch.zeros(self.n_nodes, dtype=flat.dtype, device=flat.device)
        den = torch.zeros(self.n_nodes, dtype=flat.dtype, device=flat.device)
        num = num.index_add(0, grp, flat)
        den = den.index_add(0, grp, torch.ones_like(flat))
        return num / den.clamp_min(1.0)

    def carpet_energy(self, field: Tensor) -> Tensor:
        s = self._pool_to_nodes(field)
        return s @ self.L.to(s.dtype) @ s

    def multiscale_energy(self, field: Tensor) -> Tensor:
        """[B2] Returns a SCALAR: sum_m rho^m * v_m^T L v_m."""
        s = self._pool_to_nodes(field)
        L = self.L.to(s.dtype)
        tot = torch.zeros((), dtype=field.dtype, device=field.device)
        for m, p in enumerate(self.proj):
            v = p(s.to(p.weight.dtype)).to(field.dtype)   # (n_nodes,)
            tot = tot + (self.rho ** m) * (v @ L @ v)
        return tot

    def kappa_fatness(self, field: Tensor) -> Tensor:
        """Local mean / local value >= 1/kappa. Rejects degenerate slivers."""
        f = field.unsqueeze(0).unsqueeze(0)
        avg = F.avg_pool3d(f, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
        ratio = avg / (field + 1e-12)
        return soft_gate(ratio - 1.0 / self.kappa, tau=0.05)

    def forward(self, delta_over_eta: Tensor, energy_field: Tensor,
                c_eta: float = 2.0) -> Dict[str, Tensor]:
        need_dns = soft_gate(delta_over_eta - c_eta, tau=0.2)
        fat = self.kappa_fatness(energy_field)
        priority = soft_gate(energy_field - torch.exp(self.log_thresh), tau=0.02)
        refine = need_dns * fat * (0.5 + 0.5 * priority)
        return {
            "refine_score": refine, "need_dns": need_dns,
            "kappa_valid": fat, "carpet_priority": priority,
            "carpet_energy": self.carpet_energy(energy_field),
            "multiscale_energy": self.multiscale_energy(energy_field),
            "rho": self.rho,
            "delta_min": torch.tensor(self.delta_min, device=refine.device),
            "n_refine_suggested": (refine > 0.5).sum(),
        }
