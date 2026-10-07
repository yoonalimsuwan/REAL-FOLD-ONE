# =============================================================================
# MorphogeneticSESIInterface
# =============================================================================
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================




from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:                                   # base class lives in the NoZeno optogenetics module
    from sesi_ontogenetic_optical_neuro_modulation_module import NoZenoOptogeneticInterface
except ImportError:                    # uploaded copy carries a browser "__1_" suffix
    from sesi_ontogenetic_optical_neuro_modulation_module__1_ import NoZenoOptogeneticInterface


__all__ = ["MorphogeneticSESIInterface"]


class MorphogeneticSESIInterface(NoZenoOptogeneticInterface):
    """
    Morphogenetic SESI interface with spatial bioelectric patterning.

    Extends :class:`NoZenoOptogeneticInterface` with gap-junction mediated
    ionic diffusion across a tissue topology, enabling fully differentiable
    bioelectric patterning during training.

    The membrane voltage update is

        V_{t+1} = f_isolated(V_t, I_light, h_t) + dt * G_gap * (L @ V_t)

    where ``L = A - diag(deg(A))`` is the signed graph Laplacian of the
    tissue adjacency ``A`` and ``G_gap`` is a learnable scalar gap-junction
    conductance.

    Design guarantees
    -----------------
    * **Fully differentiable.** ``gap_junction_weight`` is a single scalar
      ``Parameter``; DDP performs exactly one all-reduce per optimizer step.
    * **AMP-safe.** The parameter is kept in ``fp32`` for stability and cast
      once at point-of-use; the Laplacian product runs in the autocast dtype.
    * **Layout-agnostic.** Accepts adjacency as ``[N, N]`` (shared topology)
      or ``[B, N, N]`` (per-sample topology).
    * **Memory-efficient.** The Laplacian is never materialized — the product
      is computed as ``A @ V - deg * V`` (a matmul + a broadcast mul).
    * **Compile-friendly.** No ``.item()``, no host syncs, no Python branches
      on tensor values; ``torch.compile(model, mode="max-autotune")`` works
      out of the box.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)

        # Learnable gap-junction conductance.  Scalar shape keeps DDP cheap
        # and avoids any gradient-sync overhead.
        self.gap_junction_weight = nn.Parameter(
            torch.tensor(0.05, dtype=torch.float32)
        )

        # Lazy cache for the integration step as a tensor on the right
        # dtype/device.  Not a buffer → does not break DDP state_dict sync.
        self._dt_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _dt_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        """Return ``self._dt`` as a tensor matching ``ref``'s dtype/device."""
        dt = getattr(self, "_dt", None)
        if dt is None:
            raise AttributeError(
                "Parent class did not expose `self._dt`; cannot integrate diffusion."
            )

        if torch.is_tensor(dt):
            if dt.dtype != ref.dtype or dt.device != ref.device:
                dt = dt.to(dtype=ref.dtype, device=ref.device)
            return dt

        cached = self._dt_cache
        if (
            cached is None
            or cached.dtype != ref.dtype
            or cached.device != ref.device
        ):
            cached = torch.as_tensor(dt, dtype=ref.dtype, device=ref.device)
            self._dt_cache = cached
        return cached

    @staticmethod
    def _laplacian_product(
        adjacency: torch.Tensor,
        voltage: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ``L @ V`` with ``L = A - diag(deg(A))`` **without** ever
        forming ``L``.

        Uses the identity ``L @ V = A @ V - deg(A) ⊙ V``, which is a single
        matmul plus a broadcast multiplication — meaningfully cheaper than
        building the Laplacian and doing a second matmul.

        Parameters
        ----------
        adjacency : Tensor
            ``[N, N]`` (shared) or ``[B, N, N]`` (per-sample), same device
            and dtype-compatible with ``voltage``.
        voltage : Tensor
            ``[B, N]`` membrane potentials.

        Returns
        -------
        Tensor
            ``[B, N]`` Laplacian-vector product.
        """
        if adjacency.dim() == 2:
            # Shared topology.  For symmetric A:  A @ Vᵀ == (V @ A)ᵀ,
            # so ``voltage @ adjacency`` directly gives [B, N].
            degree = adjacency.sum(dim=-1)                      # [N]
            axv = voltage @ adjacency                           # [B, N]
            return axv - degree.unsqueeze(0) * voltage

        if adjacency.dim() == 3:
            # Per-sample topology.
            degree = adjacency.sum(dim=-1)                      # [B, N]
            axv = torch.bmm(adjacency, voltage.unsqueeze(-1)).squeeze(-1)
            return axv - degree * voltage

        raise ValueError(
            "adjacency must be 2-D [N, N] or 3-D [B, N, N]; "
            f"got shape {tuple(adjacency.shape)}"
        )

    # ------------------------------------------------------------------ #
    # Single-step update with spatial coupling
    # ------------------------------------------------------------------ #
    def _step_with_spatial_coupling(
        self,
        membrane_voltage: torch.Tensor,
        light_stimulus: torch.Tensor,
        interface_height: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Isolated-cell dynamics (parent).
        updated_voltage, h_recentered, prob_bound, trigger = super()._step(
            membrane_voltage, light_stimulus, interface_height
        )

        # 2. Gap-junction diffusion.
        #    Param stays fp32 for stability; cast once into the autocast dtype.
        g_gap = self.gap_junction_weight.to(dtype=updated_voltage.dtype)

        lap_v = self._laplacian_product(adjacency_matrix, membrane_voltage)
        dt = self._dt_tensor(updated_voltage)

        # Fused single-expression update; no temporaries beyond ``lap_v``.
        final_voltage = updated_voltage + (g_gap * dt) * lap_v

        return final_voltage, h_recentered, prob_bound, trigger

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        membrane_voltage: torch.Tensor,
        light_stimulus: torch.Tensor,
        interface_height: torch.Tensor,
        adjacency_matrix: Optional[torch.Tensor] = None,
        *,
        return_metrics: bool = True,
    ):
        """
        Parameters
        ----------
        membrane_voltage : Tensor ``[B, N]``
        light_stimulus   : Tensor ``[B, N]``
        interface_height : Tensor ``[B, N]``
        adjacency_matrix : Tensor, optional
            ``[N, N]`` or ``[B, N, N]``.  If ``None``, falls back to the
            parent's isolated-cell behavior (no bioelectric patterning).
        return_metrics : bool
            If ``False``, returns an empty metrics dict (fast path).

        Returns
        -------
        updated_voltage : Tensor ``[B, N]``
        h_recentered    : Tensor
        metrics         : dict[str, Tensor]
        """
        # Fast-path: no topology → delegate to parent.
        if adjacency_matrix is None:
            return super().forward(
                membrane_voltage,
                light_stimulus,
                interface_height,
                return_metrics=return_metrics,
            )

        updated_voltage, h_recentered, prob_bound, trigger = (
            self._step_with_spatial_coupling(
                membrane_voltage,
                light_stimulus,
                interface_height,
                adjacency_matrix,
            )
        )

        if not return_metrics:
            return updated_voltage, h_recentered, {}

        node_dims = tuple(range(1, trigger.dim()))
        metrics: Dict[str, torch.Tensor] = {
            "transition_prob_bound": prob_bound,
            "trigger_field": trigger,
            "trigger_activations_per_sample": trigger.mean(dim=node_dims),
            "mean_membrane_voltage_per_sample": updated_voltage.mean(dim=node_dims),
            # Detach so logging never drags a graph through `.item()`.
            "gap_junction_conductance": self.gap_junction_weight.detach(),
        }
        return updated_voltage, h_recentered, metrics
