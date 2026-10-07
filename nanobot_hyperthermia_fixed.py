# =============================================================================
# Nanobot Hyperthermia Ablation Module (SESI)
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================
"""
Production-grade, fully differentiable 3D Pennes bio-heat solver with
nanobot-mediated SAR heating and soft (C^∞) ALE chart re-centering.

Design goals
------------
1. Full differentiability — no `.detach()`, no hard `torch.where` gates.
2. Minimal wall-clock cost — Conv3d Laplacian, `addcmul`/`lerp` fused ops,
   no intermediate clones, no CPU↔GPU scalar transfers inside `forward`.
3. DDP-safe — stateless buffers, zero rank-local state, deterministic graph.
4. `torch.compile`-friendly — no Python-scalar graph breaks, no data-dependent
   control flow inside the compute path.
5. AMP-compatible — caller-controlled `torch.amp.autocast`.

Multi-GPU (DDP) usage
---------------------
    module = NanobotHyperthermiaAblationModule(dx=1e-4, dt=1e-3).to(rank)
    module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[rank])
    # Recommended: torch.backends.cudnn.benchmark = True
    # Optional:    module = torch.compile(module, mode="max-autotune")

Because the module holds **no trainable parameters** (only non-persistent
buffers), DDP's gradient all-reduce is a no-op on this module — it exists to
make the surrounding differentiable therapy loop DDP-clean. All reduction
happens on the outer loss / parameters that consume this module's outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (torch.utils.checkpoint is not imported by `import torch`)

__all__ = ["TissueProperties", "NanobotHyperthermiaAblationModule"]


# -----------------------------------------------------------------------------
# Tissue / physics configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TissueProperties:
    """Physical constants for the Pennes bio-heat model (SI units)."""
    density: float = 1050.0               # kg·m⁻³
    specific_heat: float = 3600.0         # J·kg⁻¹·K⁻¹
    thermal_conductivity: float = 0.51    # W·m⁻¹·K⁻¹
    blood_perfusion_rate: float = 0.005   # s⁻¹  volumetric perfusion omega_b
    blood_density: float = 1060.0         # kg·m⁻³
    blood_specific_heat: float = 3617.0   # J·kg⁻¹·K⁻¹
    core_body_temp: float = 310.15        # K  (37 °C)
    damage_threshold: float = 315.15      # K  (42 °C, softplus knee)
    ablation_threshold: float = 323.15    # K  (50 °C, re-centering gate)


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------
class NanobotHyperthermiaAblationModule(nn.Module):
    """
    One explicit forward-Euler step of the Pennes bio-heat equation with
    nanobot-mediated specific absorption rate (SAR) and differentiable
    ALE reference-chart re-centering.

    Parameters
    ----------
    dx, dt : float
        Spatial grid spacing [m] and time-step [s]. Must be positive.
    tissue : TissueProperties, optional
        Physical constants. Defaults to soft-tissue values.
    device : str | torch.device
        Hint only; buffers follow input dtype/device at runtime.
    sar_loss_factor : float
        Magnetic-loss coupling constant of the nanobot swarm.
    soft_mask_temperature : float
        Sigmoid sharpness (K) for the ablation re-centering gate.
        Larger → smoother / more diffuse; smaller → closer to hard step.
    damage_beta : float
        Sharpness of the differentiable thermal-damage accumulator.
    gradient_checkpointing : bool
        Recompute the step in backward at the cost of ~2× compute to save
        activation memory during long unrolled therapy loops.
    validate_inputs : bool
        Enable cheap shape/dtype guards (disable under `torch.compile`
        if recompilation from shape asserts becomes a bottleneck).
    """

    def __init__(
        self,
        dx: float,
        dt: float,
        tissue: Optional[TissueProperties] = None,
        device: str | torch.device = "cuda",
        *,
        sar_loss_factor: float = 2.0e-8,
        soft_mask_temperature: float = 0.5,
        damage_beta: float = 2.0,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
        boundary: str = "neumann",
    ) -> None:
        super().__init__()
        if not (math.isfinite(dx) and dx > 0.0):
            raise ValueError("dx must be a positive finite scalar.")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be a positive finite scalar.")
        if not (math.isfinite(soft_mask_temperature) and soft_mask_temperature > 0.0):
            raise ValueError("soft_mask_temperature must be > 0.")

        if boundary not in ("neumann", "core"):
            raise ValueError("boundary must be 'neumann' (insulated) or 'core' (T=core at faces).")
        self.boundary = boundary
        self.dx = float(dx)
        self.dt = float(dt)
        self.device_hint = torch.device(device)
        self.tissue = tissue or TissueProperties()
        self.sar_loss_factor = float(sar_loss_factor)
        self.soft_mask_temperature = float(soft_mask_temperature)
        self.damage_beta = float(damage_beta)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

        # ---- Pre-computed scalar coefficients (non-persistent buffers) ------
        rho_cp = self.tissue.density * self.tissue.specific_heat
        alpha = self.tissue.thermal_conductivity / rho_cp
        # FIX: Pennes sink is  omega_b*rho_b*c_b*(T-T_a)/(rho*c)  [1/s]. The original used
        # c_b/(rho*c) (missing rho_b) and then divided by rho*c AGAIN in the RHS,
        # making perfusion ~4e9x too weak. perf_coef is now the finished rate in 1/s.
        perf_coef = (self.tissue.blood_perfusion_rate * self.tissue.blood_density
                     * self.tissue.blood_specific_heat / rho_cp)
        if alpha * dt / (dx * dx) > 1.0 / 6.0:
            raise ValueError(
                f"explicit 3-D scheme unstable: alpha*dt/dx^2 = {alpha*dt/(dx*dx):.3g} > 1/6; reduce dt or increase dx.")

        self.register_buffer("_alpha",     torch.tensor(alpha,      dtype=torch.float32), persistent=False)
        self.register_buffer("_perf_coef", torch.tensor(perf_coef,  dtype=torch.float32), persistent=False)
        self.register_buffer("_inv_dx2",   torch.tensor(1.0 / (dx * dx), dtype=torch.float32), persistent=False)
        self.register_buffer("_dt",        torch.tensor(self.dt,    dtype=torch.float32), persistent=False)
        self.register_buffer("_inv_rho_cp",torch.tensor(1.0 / rho_cp, dtype=torch.float32), persistent=False)

        # ---- 7-point Laplacian kernel for a single fused F.conv3d call ----
        k = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        k[0, 0, 1, 1, 1] = -6.0
        k[0, 0, 1, 1, 0] = k[0, 0, 1, 1, 2] = 1.0
        k[0, 0, 1, 0, 1] = k[0, 0, 1, 2, 1] = 1.0
        k[0, 0, 0, 1, 1] = k[0, 0, 2, 1, 1] = 1.0
        self.register_buffer("_lap_kernel", k, persistent=False)

    # ------------------------------------------------------------------ #
    # Core differentiable step                                           #
    # ------------------------------------------------------------------ #
    def _step(
        self,
        temperature_field: torch.Tensor,     # [B, 1, Z, Y, X]  K
        nanobot_density: torch.Tensor,       # [B, 1, Z, Y, X]  a.u.
        reference_chart: torch.Tensor,       # [B, 1, Z, Y, X]  ALE Γ₀
        ac_magnetic_field_amp: torch.Tensor, # scalar tensor (Tesla)
        ac_frequency: torch.Tensor,          # scalar tensor (Hz)
        damage_accumulator: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        t = self.tissue
        dtype = temperature_field.dtype
        device = temperature_field.device

        # Promote scalar buffers to input dtype/device once per step.
        alpha     = self._alpha.to(dtype=dtype, device=device)
        perf      = self._perf_coef.to(dtype=dtype, device=device)
        inv_dx2   = self._inv_dx2.to(dtype=dtype, device=device)
        dt        = self._dt.to(dtype=dtype, device=device)
        inv_rho_cp = self._inv_rho_cp.to(dtype=dtype, device=device)
        lap_k     = self._lap_kernel.to(dtype=dtype, device=device)

        # ---- 1. Laplacian: single fused Conv3d (≈6× cheaper than 6× roll) ----
        # FIX: zero padding put the boundary neighbours at 0 K (about -4 K per step at faces).
        if self.boundary == "neumann":
            padded = F.pad(temperature_field, (1, 1, 1, 1, 1, 1), mode="replicate")
        else:
            padded = F.pad(temperature_field, (1, 1, 1, 1, 1, 1), mode="constant", value=t.core_body_temp)
        lap_T = F.conv3d(padded, lap_k) * inv_dx2

        # ---- 2. SAR: nanobot-mediated magnetic-loss heating (fully smooth) --
        #   scalar tensors * tensor field — no Python-float graph breaks.
        sar = nanobot_density * (self.sar_loss_factor * (ac_magnetic_field_amp ** 2) * (ac_frequency ** 2))

        # ---- 3. Pennes perfusion sink (linear in T; differentiable everywhere) --
        delta_T   = temperature_field - t.core_body_temp
        perfusion = -perf * delta_T

        # ---- 4. Bio-heat RHS + forward-Euler integration (fused) ------------
        dT_dt = alpha * lap_T + sar * inv_rho_cp + perfusion
        updated_temperature = torch.addcmul(temperature_field, dT_dt, dt)

        # ---- 5. Differentiable thermal-damage accumulator -------------------
        #   softplus(knee) replaces ReLU → C^∞, no zero-gradient dead zone.
        damage_increment = F.softplus(
            updated_temperature - t.damage_threshold, beta=self.damage_beta
        ) * dt
        thermal_damage_metric = (
            damage_increment if damage_accumulator is None
            else damage_accumulator + damage_increment
        )

        # ---- 6. Soft ALE re-centering (differentiable, no detach) -----------
        #   sigmoid gate smoothly interpolates Γ₀ ← T as T crosses the
        #   ablation threshold. Gradient flows through both branches.
        recenter_weight = torch.sigmoid(
            (updated_temperature - t.ablation_threshold) / self.soft_mask_temperature
        )
        updated_reference_chart = torch.lerp(reference_chart, updated_temperature, recenter_weight)

        return updated_temperature, thermal_damage_metric, updated_reference_chart, recenter_weight

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        temperature_field: torch.Tensor,          # [B, 1, Z, Y, X] K
        nanobot_density: torch.Tensor,            # [B, 1, Z, Y, X] a.u.
        reference_chart: torch.Tensor,            # [B, 1, Z, Y, X] ALE Γ₀
        ac_magnetic_field_amp: torch.Tensor | float,
        ac_frequency: torch.Tensor | float,
        damage_accumulator: Optional[torch.Tensor] = None,
        *,
        return_recenter_weight: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        updated_temperature    : [B, 1, Z, Y, X]
        thermal_damage_metric  : [B, 1, Z, Y, X]  (accumulated if provided)
        updated_reference_chart: [B, 1, Z, Y, X]
        recenter_weight (opt.) : [B, 1, Z, Y, X]  in (0, 1)
        """
        if self.validate_inputs:
            if temperature_field.dim() != 5:
                raise ValueError("Expected [B, 1, Z, Y, X] tensors.")
            if not (temperature_field.shape == nanobot_density.shape == reference_chart.shape):
                raise ValueError("temperature_field, nanobot_density, reference_chart must share shape.")
            if damage_accumulator is not None and damage_accumulator.shape != temperature_field.shape:
                raise ValueError("damage_accumulator must match temperature_field shape.")

        # Promote scalar physics inputs to tensors on the correct device so
        # no CPU→GPU sync occurs inside the graph (torch.compile-friendly).
        dev = temperature_field.device
        if not isinstance(ac_magnetic_field_amp, torch.Tensor):
            ac_magnetic_field_amp = torch.as_tensor(ac_magnetic_field_amp, dtype=temperature_field.dtype, device=dev)
        if not isinstance(ac_frequency, torch.Tensor):
            ac_frequency = torch.as_tensor(ac_frequency, dtype=temperature_field.dtype, device=dev)

        if self.gradient_checkpointing and self.training:
            step = torch.utils.checkpoint.checkpoint(
                self._step,
                temperature_field, nanobot_density, reference_chart,
                ac_magnetic_field_amp, ac_frequency, damage_accumulator,
                use_reentrant=False,
            )
        else:
            step = self._step(
                temperature_field, nanobot_density, reference_chart,
                ac_magnetic_field_amp, ac_frequency, damage_accumulator,
            )

        updated_temperature, thermal_damage_metric, updated_reference_chart, w = step

        if return_recenter_weight:
            return updated_temperature, thermal_damage_metric, updated_reference_chart, w
        return updated_temperature, thermal_damage_metric, updated_reference_chart


# -----------------------------------------------------------------------------
# Reference DDP driver (illustrative — not executed at import time)
# -----------------------------------------------------------------------------
# import os, torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
#
# def main(rank: int, world_size: int) -> None:
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cuda.matmul.allow_tf32 = True
#
#     model = NanobotHyperthermiaAblationModule(dx=1e-4, dt=1e-3).to(rank)
#     model = DDP(model, device_ids=[rank])
#     model = torch.compile(model, mode="max-autotune")  # optional
#
#     B, Z, Y, X = 4, 128, 128, 128
#     T0 = torch.full((B, 1, Z, Y, X), 310.15, device=rank)
#     rho = torch.rand_like(T0) * 1e-2
#     g0 = T0.clone()
#
#     with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
#         for _ in range(100):
#             T0, dmg, g0 = model(T0, rho, g0, 0.05, 1.0e5)
#
#     (T0.mean() + dmg.mean()).backward()
#     dist.destroy_process_group()
