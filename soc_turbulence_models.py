"""
Standard turbulence closures, wrapped UNCHANGED. All cited, none novel.

[B1 FIX] TurbBase now defines forward() delegating to nut(), so BOTH
         model.nut(...) and model(...) work. Previously model(...) raised
         NotImplementedError because nn.Module.forward was never defined.
[B3 FIX] WALE uses S:S = 0.5|S|^2 (standard), not |S|^2.
[B4 FIX] dead loops / branches removed.
"""
from __future__ import annotations
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .primitives import soft_pos, soft_max
from .grid import Grid, grad_cc

TURB: Dict[str, type] = {}


def register(name: str):
    def deco(cls):
        TURB[name] = cls
        cls.of_name = name
        return cls
    return deco


class TurbBase(nn.Module):
    """
    Interface. Subclasses implement nut(); forward() delegates to it so
    both call styles work. [B1]

    nut(state, S, delta, nu, grid) -> nu_t   (nx,ny,nz), kinematic.
    `state` must contain every name in `needs`.
    """
    of_name = "base"
    needs: Tuple[str, ...] = ()

    def nut(self, state: Dict[str, Tensor], S: Tensor, delta: float,
            nu: float, grid: Grid) -> Tensor:
        raise NotImplementedError

    def forward(self, state: Dict[str, Tensor], S: Tensor, delta: float,
                nu: float, grid: Grid) -> Tensor:
        return self.nut(state, S, delta, nu, grid)

    def missing(self, state: Dict[str, Tensor]) -> Tuple[str, ...]:
        return tuple(k for k in self.needs if k not in state)


@register("zero_dns")
class ZeroModel(TurbBase):
    """nu_t = 0. Correct ONLY where the grid resolves eta (Delta <= c_eta*eta)."""
    def nut(self, state, S, delta, nu, grid):
        return torch.zeros_like(S)


@register("smagorinsky")
class Smagorinsky(TurbBase):
    """Smagorinsky (1963): nu_t = (Cs Delta)^2 |S|."""
    def __init__(self, Cs: float = 0.17, learn: bool = True):
        super().__init__()
        v = torch.tensor(math.log(Cs))
        if learn:
            self.log_Cs = nn.Parameter(v)
        else:
            self.register_buffer("log_Cs", v)

    def nut(self, state, S, delta, nu, grid):
        return (torch.exp(self.log_Cs) * delta) ** 2 * S


@register("wale")
class WALE(TurbBase):
    """
    Nicoud & Ducros (1999).
      g2_ij = sum_k (du_i/dx_k)(du_k/dx_j)
      Sd    = sym(g2) - (1/3) tr(g2) I
      nu_t  = (Cw Delta)^2 (Sd:Sd)^{3/2} / [(S:S)^{5/2} + (Sd:Sd)^{5/4}]
    [B3] S:S = 0.5 |S|^2, NOT |S|^2.
    """
    def __init__(self, Cw: float = 0.5):
        super().__init__()
        self.log_Cw = nn.Parameter(torch.tensor(math.log(Cw)))

    def nut(self, state, S, delta, nu, grid):
        u = state["u"]
        gu = torch.stack([grad_cc(u[c], grid) for c in range(3)], 0)
        g2 = torch.einsum("ik...,kj...->ij...", gu, gu)
        tr = torch.diagonal(g2, dim1=0, dim2=1).sum(0) / 3.0
        Sd = 0.5 * (g2 + g2.transpose(0, 1))
        for i in range(3):
            Sd[i, i] = Sd[i, i] - tr
        sd2 = (Sd * Sd).sum(dim=(0, 1)).clamp_min(1e-30)
        s2 = (0.5 * S * S).clamp_min(1e-30)          # [B3] S:S
        num = sd2 ** 1.5
        den = s2 ** 2.5 + sd2 ** 1.25
        return (torch.exp(self.log_Cw) * delta) ** 2 * num / den.clamp_min(1e-30)


@register("vreman")
class Vreman(TurbBase):
    """
    Vreman (2004).
      alpha = a_ij a_ij,  beta_ij = sum_m a_mi a_mj
      B_beta = alpha^2 - sum_ij beta_ij^2
      nu_t = Cv Delta^2 sqrt(B_beta / alpha),  Cv ~ 2.5 Cs^2 ~ 0.07
    [B4] dead accumulation loops removed.
    """
    def __init__(self, Cv: float = 0.07):
        super().__init__()
        self.log_Cv = nn.Parameter(torch.tensor(math.log(Cv)))

    def nut(self, state, S, delta, nu, grid):
        u = state["u"]
        a = torch.stack([grad_cc(u[c], grid) for c in range(3)], 0)
        alpha = (a * a).sum(dim=(0, 1)).clamp_min(1e-30)
        inner = torch.zeros_like(S)
        for i in range(3):
            for j in range(3):
                b = sum(a[m, i] * a[m, j] for m in range(3))
                inner = inner + b * b
        Bb = (alpha * alpha - inner).clamp_min(0.0)
        return torch.exp(self.log_Cv) * delta ** 2 * torch.sqrt(Bb / alpha + 1e-30)


@register("k_epsilon")
class KEpsilon(TurbBase):
    """Launder & Spalding (1974): nu_t = Cmu k^2 / eps. Needs TRANSPORTED k, eps."""
    needs = ("k", "epsilon")

    def __init__(self, Cmu: float = 0.09):
        super().__init__()
        self.log_Cmu = nn.Parameter(torch.tensor(math.log(Cmu)))

    def nut(self, state, S, delta, nu, grid):
        k = soft_pos(state["k"])
        eps = soft_pos(state["epsilon"], beta=200.0) + 1e-12
        return torch.exp(self.log_Cmu) * k * k / eps


@register("k_omega_sst")
class KOmegaSST(TurbBase):
    """
    Menter (1994) SST eddy viscosity: nu_t = a1 k / max(a1 w, b1 F2 |S|).

    [B5] WITHOUT `d_wall` in state, F2 = 1 (the FAR-FIELD limit). This is
    NOT valid for wall-bounded flow. Supply d_wall to enable F2.
    Only the eddy-viscosity relation is implemented; the full SST
    transport equations with F1 blending and cross-diffusion are the
    host solver's responsibility.
    """
    needs = ("k", "omega")

    def __init__(self, a1: float = 0.31, b1: float = 1.0, beta_star: float = 0.09):
        super().__init__()
        self.a1, self.b1, self.bs = a1, b1, beta_star

    def nut(self, state, S, delta, nu, grid):
        k = soft_pos(state["k"])
        om = soft_pos(state["omega"], beta=200.0) + 1e-12
        d = state.get("d_wall")
        if d is None:
            F2 = torch.ones_like(S)
        else:
            arg = soft_max(2 * torch.sqrt(k) / (self.bs * om * d + 1e-12),
                           500 * nu / (d * d * om + 1e-12))
            F2 = torch.tanh(arg ** 2)
        den = soft_max(torch.full_like(S, self.a1) * om, self.b1 * F2 * S)
        return self.a1 * k / den.clamp_min(1e-12)

    def wall_distance_supplied(self, state) -> bool:
        return "d_wall" in state


@register("spalart_allmaras")
class SpalartAllmaras(TurbBase):
    """Spalart & Allmaras (1992): nu_t = nuTilda * fv1. Needs TRANSPORTED nuTilda."""
    needs = ("nuTilda",)

    def __init__(self, Cv1: float = 7.1):
        super().__init__()
        self.Cv1 = Cv1

    def nut(self, state, S, delta, nu, grid):
        nt = soft_pos(state["nuTilda"])
        chi = nt / max(nu, 1e-30)
        fv1 = chi ** 3 / (chi ** 3 + self.Cv1 ** 3 + 1e-14)
        return nt * fv1


def make_model(name: str, **kw) -> TurbBase:
    if name not in TURB:
        raise KeyError(f"unknown model '{name}'. Registered: {sorted(TURB)}")
    return TURB[name](**kw)


def local_equilibrium_state(u: Tensor, S: Tensor, delta: float, nu: float,
                            d_wall: Tensor | None = None) -> Dict[str, Tensor]:
    """
    DEMO ONLY. Mixing-length-style estimates of k, eps, omega, nuTilda.
    In production these MUST be transported by the host solver.
    Labelled `_source` so downstream code can detect the fallback.
    """
    ell = max(delta, 1e-8)
    k = 0.5 * (0.1 * S * ell) ** 2 + 1e-8
    eps = 0.09 ** 0.75 * k ** 1.5 / ell + 1e-12
    st = {"u": u, "k": k, "epsilon": eps,
          "omega": eps / (0.09 * k + 1e-12) + 1e-12,
          "nuTilda": 0.09 * k * k / eps,
          "_source": "LOCAL_EQUILIBRIUM_DEMO"}
    if d_wall is not None:
        st["d_wall"] = d_wall
    return st
