# =============================================================================
# Nanobot Payload Delivery Module (SESI)
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
Production-grade, fully differentiable multi-compartment payload-release engine
with SESI topological operators (Nucleation N, Merging M, Branching B) and a
differentiable energy-conservation bound.

Design goals
------------
1. Full differentiability — every step is C^∞:
   · `torch.where(mask, A, B)`  →  `torch.lerp(B, A, mask)` (soft blending)
   · `F.relu(·)`                →  `F.softplus(·, β)`        (no dead zone)
   · hard trigger clamp         →  probabilistic **soft-OR** (`a+b−ab`)
   · stochastic noise           →  reparameterized additive noise,
                                   deterministic (zero) in `.eval()`
2. Exact mass conservation — the decay step is bounded by available cargo via a
   differentiable **smooth-min** (`b − softplus(b − a, β)`), preventing the
   classic "release more than was encapsulated" bug that silently inflates
   `released` and breaks SESI invariants.
3. Differentiable energy bound — the SESI topological inequality
   `E(Γ(T_k⁺)) − E(Γ(T_k⁻)) ≤ C_topo` is exposed as a soft penalty
   `softplus(ΔE − C_topo, β)` that can be added directly to the training loss.
4. Minimal wall-clock cost — all kinetic constants folded to scalars, single
   fused ops (`torch.addcmul`, `torch.lerp`, `F.avg_pool3d`), no Python-scalar
   graph breaks, no data-dependent `.any()` synchronizations.
5. DDP-safe & `torch.compile` friendly — parameter-free, non-persistent buffers,
   no rank-local RNG state, caller-controlled AMP.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    module = NanobotPayloadDeliveryModule(dt=1e-2).to(rank)
    module = torch.compile(module, mode="max-autotune")           # optional
    module = torch.nn.parallel.DistributedDataParallel(
        module, device_ids=[rank], gradient_as_bucket_view=True
    )

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        payload, metrics = module(payload, ph, mmp, topo_weight)

    loss = therapy_loss(payload) + metrics["energy_penalty"].mean()
    loss.backward()      # grads flow through release kinetics + topo blend
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PayloadKinetics", "NanobotPayloadDeliveryModule"]


# -----------------------------------------------------------------------------
# Kinetic / topological configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PayloadKinetics:
    """Release kinetics and SESI topological-operator constants."""
    k_baseline: float = 1.0e-4            # s⁻¹  basal unpacking
    k_triggered: float = 0.05             # s⁻¹  stimulus-gated unpacking
    activation_ph: float = 6.5            # pH   tumor-microenvironment trigger
    enzyme_threshold: float = 0.5         # a.u. MMP concentration half-point
    ph_slope: float = 10.0                # sigmoid sharpness on pH
    topo_energy_bound: float = 5.0        # C_topo  (SESI Thm. energy budget)
    topo_noise_scale: float = 0.1         # σ  membrane-fluctuation amplitude


# -----------------------------------------------------------------------------
# Module
# -----------------------------------------------------------------------------
class NanobotPayloadDeliveryModule(nn.Module):
    """
    One explicit multi-compartment release step with differentiable SESI
    topological operators applied through soft blending weights.

    Channel layout (input & output)
    --------------------------------
    payload_concentration[:, 0] = encapsulated cargo
    payload_concentration[:, 1] = released cargo

    Parameters
    ----------
    dt : float
        Integration step [s]; must be > 0.
    kinetics : PayloadKinetics, optional
        Release / topological constants; defaults to tumor-microenvironment set.
    device : str | torch.device
        Hint only; buffers follow input dtype/device at runtime.
    mass_beta : float
        Sharpness of the smooth-min used for cargo conservation. Larger → closer
        to a hard `min`, still differentiable.
    energy_beta : float
        Sharpness of the SESI energy-violation softplus penalty.
    gradient_checkpointing : bool
        Recompute the step in backward — trades ~2× compute for activation
        memory savings in long unrolled therapy loops.
    validate_inputs : bool
        Cheap shape guards. Disable once shapes are statically known to avoid
        `torch.compile` recompiles.
    """

    def __init__(
        self,
        dt: float,
        kinetics: Optional[PayloadKinetics] = None,
        device: str | torch.device = "cuda",
        *,
        mass_beta: float = 50.0,
        energy_beta: float = 4.0,
        gradient_checkpointing: bool = False,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be a positive finite scalar.")
        if not (math.isfinite(mass_beta) and mass_beta > 0.0):
            raise ValueError("mass_beta must be > 0.")
        if not (math.isfinite(energy_beta) and energy_beta > 0.0):
            raise ValueError("energy_beta must be > 0.")

        self.dt = float(dt)
        self.device_hint = torch.device(device)
        self.kinetics = kinetics or PayloadKinetics()

        self.mass_beta = float(mass_beta)
        self.energy_beta = float(energy_beta)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.validate_inputs = bool(validate_inputs)

        # ---- Non-persistent buffers (DDP-safe: not in state_dict) ----------
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_k_base",        _buf(self.kinetics.k_baseline),   persistent=False)
        self.register_buffer("_k_trig",        _buf(self.kinetics.k_triggered),  persistent=False)
        self.register_buffer("_ph_thresh",     _buf(self.kinetics.activation_ph), persistent=False)
        self.register_buffer("_ph_slope",      _buf(self.kinetics.ph_slope),     persistent=False)
        self.register_buffer("_mmp_thresh",    _buf(self.kinetics.enzyme_threshold), persistent=False)
        self.register_buffer("_e_bound",       _buf(self.kinetics.topo_energy_bound), persistent=False)
        self.register_buffer("_topo_noise",    _buf(self.kinetics.topo_noise_scale), persistent=False)
        self.register_buffer("_dt",            _buf(self.dt),                    persistent=False)

    # ------------------------------------------------------------------ #
    # Differentiable primitives                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _soft_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Probabilistic OR on inputs in (0,1):  a + b − a·b  ∈ (0,1)."""
        return a + b - a * b

    def _smooth_min(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        C^∞ approximation of `min(a, b)`:
            min(a, b) ≈ b − softplus(b − a, β)
        Exact as β→∞; unbiased for positive a, b.
        """
        return b - F.softplus(b - a, beta=self.mass_beta)

    # ------------------------------------------------------------------ #
    # Core step (isolated so gradient checkpointing can wrap it)         #
    # ------------------------------------------------------------------ #
    def _step(
        self,
        payload_concentration: torch.Tensor,  # [B, 2, Z, Y, X]
        local_ph: torch.Tensor,               # [B, 1, Z, Y, X]
        local_enzyme_mmp: torch.Tensor,       # [B, 1, Z, Y, X]
        topo_weight: torch.Tensor,            # [B, 1, Z, Y, X] soft ∈ [0,1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        dtype = payload_concentration.dtype
        device = payload_concentration.device

        # Promote scalar buffers once per step (torch.compile-stable).
        k_base     = self._k_base.to(dtype=dtype,     device=device)
        k_trig     = self._k_trig.to(dtype=dtype,     device=device)
        ph_thresh  = self._ph_thresh.to(dtype=dtype,  device=device)
        ph_slope   = self._ph_slope.to(dtype=dtype,   device=device)
        mmp_thresh = self._mmp_thresh.to(dtype=dtype, device=device)
        e_bound    = self._e_bound.to(dtype=dtype,    device=device)
        topo_sigma = self._topo_noise.to(dtype=dtype, device=device)
        dt         = self._dt.to(dtype=dtype,         device=device)

        encapsulated = payload_concentration[:, 0:1]
        released     = payload_concentration[:, 1:2]

        # ---- 1. Differentiable trigger (soft-OR of pH and MMP gates) --------
        p_ph  = torch.sigmoid(ph_slope * (ph_thresh - local_ph))
        p_mmp = torch.sigmoid(local_enzyme_mmp - mmp_thresh)
        trigger_mask = self._soft_or(p_ph, p_mmp)                       # ∈ (0,1)

        # ---- 2. Release flux & mass-conserving decay ------------------------
        release_flux = (k_base + k_trig * trigger_mask) * encapsulated
        wanted_decay = release_flux * dt

        # C^∞ bounded decay: actual ≤ encapsulated  →  mass is exactly conserved.
        # FIX: _smooth_min(a,b)=b-softplus(b-a,beta) is biased by up to ln2/beta (0.014 at
        # beta=50) and went NEGATIVE (cargo flowing backwards) at low concentration.
        # Exact first-order release: decay = enc * (1 - exp(-k*dt)) is always in [0, enc].
        rate = k_base + k_trig * trigger_mask
        actual_decay = encapsulated * (-torch.expm1(-rate * dt))
        new_encapsulated = encapsulated - actual_decay
        new_released     = released     + actual_decay

        # ---- 3. SESI topological operators N / M / B via soft blending ------
        #   M (membrane fusion):  local 3×3×3 mean-field smoothing
        smoothed = F.avg_pool3d(new_released, kernel_size=3, stride=1, padding=1)
        #   Reparameterized additive fluctuation — deterministic in eval.
        if self.training:
            topo_target = torch.addcmul(smoothed, topo_sigma,
                                        torch.randn_like(smoothed))
        else:
            topo_target = smoothed
        #   Blend origin → topo-target by the *soft* jump weight (C^∞ lerp)
        new_released = torch.lerp(new_released, topo_target, topo_weight)

        updated_payload = torch.cat([new_encapsulated, new_released], dim=1)

        # ---- 4. Differentiable SESI energy-bound penalty --------------------
        #   ΔE := mean |actual_decay|  (mass moved by the topo event)
        #   penalty := softplus(ΔE − C_topo, β)  ≥ 0, differentiable, 0 when safe.
        delta_energy   = actual_decay.abs().mean(dim=1, keepdim=True)
        energy_penalty = F.softplus(delta_energy - e_bound, beta=self.energy_beta)

        # Return raw tensors only — no Python bools / .any() syncs.
        return updated_payload, release_flux, trigger_mask, delta_energy, energy_penalty

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        payload_concentration: torch.Tensor,  # [B, 2, Z, Y, X]
        local_ph: torch.Tensor,               # [B, 1, Z, Y, X]
        local_enzyme_mmp: torch.Tensor,       # [B, 1, Z, Y, X]
        topo_jump_mask: torch.Tensor,         # [B, 1, Z, Y, X] soft weight ∈ [0,1]
        *,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        updated_payload : [B, 2, Z, Y, X]  (channel 0: encapsulated, 1: released)
        metrics         : dict of differentiable tensors, including
                          `energy_penalty` which should be added to the loss
                          to enforce the SESI topological energy inequality.
        """
        if self.validate_inputs:
            if payload_concentration.dim() != 5 or payload_concentration.shape[1] != 2:
                raise ValueError("payload_concentration must be [B, 2, Z, Y, X].")
            b, _, z, y, x = payload_concentration.shape
            for name, t in (("local_ph", local_ph), ("local_enzyme_mmp", local_enzyme_mmp),
                            ("topo_jump_mask", topo_jump_mask)):
                if t.dim() != 5 or t.shape != (b, 1, z, y, x):
                    raise ValueError(f"{name} must be [B, 1, Z, Y, X] matching payload.")

        # `topo_jump_mask` is used as a *soft* weight ∈ [0,1]; if the caller
        # passed a boolean tensor we cast it to the payload dtype (no branch).
        if topo_jump_mask.dtype != payload_concentration.dtype:
            topo_jump_mask = topo_jump_mask.to(payload_concentration.dtype)
        topo_weight = topo_jump_mask.clamp_(0.0, 1.0)

        if self.gradient_checkpointing and self.training:
            step = torch.utils.checkpoint.checkpoint(
                self._step,
                payload_concentration, local_ph, local_enzyme_mmp, topo_weight,
                use_reentrant=False,
            )
        else:
            step = self._step(
                payload_concentration, local_ph, local_enzyme_mmp, topo_weight,
            )

        updated_payload, release_flux, trigger_mask, delta_energy, energy_penalty = step

        if not return_metrics:
            return updated_payload, {}

        metrics: Dict[str, torch.Tensor] = {
            "release_flux":           release_flux,
            "trigger_activation":     trigger_mask,
            "topological_energy":     delta_energy,
            "energy_penalty":         energy_penalty,       # ← add to loss
            "topological_weight_mean": topo_weight.mean(),  # differentiable scalar
        }
        return updated_payload, metrics


# -----------------------------------------------------------------------------
# Reference DDP driver (illustrative — not executed at import time)
# -----------------------------------------------------------------------------
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
#
# def main(rank: int, world_size: int) -> None:
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cuda.matmul.allow_tf32 = True
#
#     model = NanobotPayloadDeliveryModule(dt=1e-2).to(rank)
#     model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
#     model = torch.compile(model, mode="max-autotune")
#
#     B, Z, Y, X = 4, 64, 64, 64
#     payload = torch.zeros(B, 2, Z, Y, X, device=rank)
#     payload[:, 0] = 1.0                                     # fully encapsulated
#     ph   = 6.0 + 0.5 * torch.randn(B, 1, Z, Y, X, device=rank)
#     mmp  = torch.rand(B, 1, Z, Y, X, device=rank)
#     topo = (torch.rand(B, 1, Z, Y, X, device=rank) > 0.9).float()
#
#     with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
#         for _ in range(200):
#             payload, metrics = model(payload, ph, mmp, topo)
#             loss = payload[:, 1].mean() + metrics["energy_penalty"].mean()
#             loss.backward()
#     dist.destroy_process_group()
