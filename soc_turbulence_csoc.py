"""
CSOC controller with a RIGOROUS event-count bound.

THEOREMS.md T1:  N(T) <= (E_max(T) + D(T)) / delta_E_min
D MUST be the ACTUAL dissipation removed, not excess*rate.
All state buffers are DETACHED: no autograd-graph accumulation.
"""
from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CSOCController(nn.Module):
    """
    excess  = softplus(s - theta_hi)
    decay   = sigmoid(-(acc - theta_lo)*10)
    removal = decay * decay_rate * acc        <-- ACTUAL dissipation
    acc'    = acc + excess - removal

    Fire when armed and acc > theta_hi; re-arm only when acc < theta_lo.
    Each fire consumes >= delta_E_min of gain, total gain <= E_max + D.

    ENGINEERING bound on this mechanism. NOT a PDE regularity result.
    """

    def __init__(self, theta_hi: float = 0.15, theta_lo: float = 0.05,
                 nu_collapse: float = 1e-3, decay_rate: float = 0.05,
                 learn_thresholds: bool = False):
        super().__init__()
        if not theta_hi > theta_lo:
            raise ValueError("require theta_hi > theta_lo")
        self.decay_rate = float(decay_rate)
        if learn_thresholds:
            self.hi_raw = nn.Parameter(torch.tensor(float(theta_hi)))
            self.gap_raw = nn.Parameter(torch.tensor(math.log(theta_hi - theta_lo)))
        else:
            self.register_buffer("hi_raw", torch.tensor(float(theta_hi)))
            self.register_buffer("gap_raw", torch.tensor(math.log(theta_hi - theta_lo)))
        self.log_nu_collapse = nn.Parameter(torch.tensor(math.log(nu_collapse)))
        self.gate_net = nn.Sequential(nn.Linear(1, 16), nn.GELU(),
                                      nn.Linear(16, 1), nn.Sigmoid())
        for n in ("acc", "events", "E_max", "D_total", "n_steps"):
            self.register_buffer(n, torch.zeros((), dtype=torch.float64))
        self.register_buffer("in_event", torch.zeros((), dtype=torch.bool))

    @property
    def theta_hi(self) -> Tensor:
        return F.softplus(self.hi_raw) + 1e-6

    @property
    def delta_e_min(self) -> Tensor:
        return F.softplus(self.gap_raw) + 1e-6

    @property
    def theta_lo(self) -> Tensor:
        return self.theta_hi - self.delta_e_min

    @property
    def nu_collapse(self) -> Tensor:
        return torch.exp(self.log_nu_collapse).clamp(1e-12, 1e3)

    def reset(self) -> None:
        for n in ("acc", "events", "E_max", "D_total", "n_steps"):
            getattr(self, n).zero_()
        self.in_event.zero_()

    def forward(self, stress: Tensor) -> Dict[str, Tensor]:
        s = stress.reshape(()).to(torch.float64)
        hi, lo = self.theta_hi.to(s.dtype), self.theta_lo.to(s.dtype)
        excess = F.softplus(s - hi)
        acc = self.acc.detach()
        decay = torch.sigmoid(-(acc - lo) * 10.0)
        removal = decay * self.decay_rate * acc          # ACTUAL D
        acc_new = acc + excess.detach() - removal

        fire = (acc_new > hi.detach()) & (~self.in_event)
        clear = (acc_new < lo.detach()) & self.in_event
        new_event = (self.in_event | fire) & (~clear)

        # detached, out-of-place: graph does NOT accumulate across steps
        self.acc = acc_new.detach()
        self.in_event = new_event.detach()
        self.events = (self.events + fire.to(self.events.dtype)).detach()
        self.E_max = torch.maximum(self.E_max, acc_new).detach()
        self.D_total = (self.D_total + removal.detach()).detach()
        self.n_steps = (self.n_steps + 1.0).detach()

        gate = self.gate_net(
            acc_new.reshape(1, 1).to(self.gate_net[0].weight.dtype)
        ).reshape(()).to(s.dtype)
        nu_soc = self.nu_collapse.to(s.dtype) * gate * excess.to(s.dtype)
        bound = (self.E_max + self.D_total) / self.delta_e_min.detach().clamp_min(1e-12)
        return {
            "gate": gate, "nu_soc": nu_soc, "acc": acc_new,
            "excess": excess, "removal": removal,
            "event_count": self.events, "n_bound": bound,
            "bound_satisfied": self.events <= bound + 1e-9,
            "delta_e_min": self.delta_e_min,
            "E_max": self.E_max, "D_total": self.D_total,
        }
