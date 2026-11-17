"""
Hodge-INSPIRED split (NOT Hodge theory) and assumption-light CI.
See THEOREMS.md T5 and T6.
"""
from __future__ import annotations
import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class HodgeInspiredSplit(nn.Module):
    """
    3-way split with a symmetric learned Laplacian kernel penalty.

    HONEST: NOT Hodge theory. No exterior derivative d, no codifferential,
    no graded complex, no d^2 = 0, no orthogonality guarantee. It is a
    regularised representation split INSPIRED by the decomposition.
    """

    def __init__(self, d: int = 16):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, 3 * d)
        self.L_raw = nn.Parameter(torch.randn(d, d, dtype=torch.float64) * 0.02)

    @property
    def laplacian(self) -> Tensor:
        return 0.5 * (self.L_raw + self.L_raw.T)

    def forward(self, nu_t: Tensor) -> Dict[str, Tensor]:
        v = nu_t.reshape(-1).mean().reshape(1, 1).to(self.proj.weight.dtype)
        h, e, c = self.proj(v).reshape(3, self.d).to(nu_t.dtype)
        lap = h @ self.laplacian.to(h.dtype)
        return {"harmonic_norm": h.norm(), "exact_norm": e.norm(),
                "coexact_norm": c.norm(),
                "kernel_penalty": (lap * lap).sum(),
                "snr": (h @ h + e @ e) / (c @ c + 1e-12)}


def subsample_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """
    Politis-Romano subsampling CI. Asymptotically valid under stationarity
    + some mixing + finite variance. "Assumption-light" is NOT
    "assumption-free". THEOREMS.md T6.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 6:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n, "block": 0}
    b = min(max(2, int(n ** (1 / 3))), n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th)
    a = 1.0 - conf
    lo = th - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi), "n": n, "block": b}
