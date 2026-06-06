# =============================================================================
# ONE CORE FOLD — Shared Foundation for the REAL FOLD ONE Ecosystem
# =============================================================================
# Developer : Yoon A Limsuwan / MSPS NETWORK
# License   : MIT
# Year      : 2026
# ORCID     : 0009-0008-2374-0788
# GitHub    : yoonalimsuwan
#
# Single source of truth for components shared across:
#   real_fold_one.py          — full-atom differentiable refinement engine
#   real_fold_one_ht.py       — high-throughput mutation / epistasis scanner
#   structural_langevin.py    — BAOAB Langevin MD integrator
#
# This module is intentionally separate from one_core.py (the DNS/CFD
# ecosystem) because the two ecosystems operate at different physical scales:
#
#   REAL FOLD ONE ecosystem   → molecular / residue scale (Å, kcal/mol)
#   SUPER DNS ONE ecosystem   → continuum / CFD scale     (m, Pa, m²/s)
#
# Shared components (this file)
# ─────────────────────────────
#   SemanticStateContraction  — SSC EMA filter             (Paper 4)
#   CSOCBase                  — abstract CSOC base class    (Paper 4)
#   InterfaceDetectorBase     — abstract interface detector
#   StructuralItoBase         — abstract Itô correction     (Papers 2 & 3)
#   LangevinBridge            — connects RefinementEngine ↔ AdvancedStructuralLangevin
#   get_device                — unified hardware-backend selector
#   FOLD_VERSION              — ecosystem-wide version string
# =============================================================================

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

FOLD_VERSION: str = "1.0.0"


# =============================================================================
# 0. Hardware-backend selector
# =============================================================================

def get_device(preferred: str = "cuda") -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.

    Args:
        preferred : ``"cuda"``, ``"mps"``, ``"ascend"``, or ``"cpu"``.
    Returns:
        :class:`torch.device`
    """
    p = preferred.lower()
    if p == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if p == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if p == "ascend":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. Semantic State Contraction (SSC) — Paper 4
# =============================================================================

class SemanticStateContraction(nn.Module):
    """
    SSC EMA low-pass filter for structural stress σ  (Paper 4).

    **Canonical implementation** — used by all three files in the REAL FOLD
    ONE ecosystem.  Do not redefine locally in individual solver files.

    The filter tracks structural stress via a first-order EMA:

        σ[t] = σ[t-1] + ε · (σ_raw[t] − σ[t-1])

    Fixes over the original ``real_fold_one.py`` version
    ─────────────────────────────────────────────────────
    •  Uses a boolean ``_initialized`` buffer rather than ``prev == 0.0``
       (the old check breaks when the true first stress is exactly zero).
    •  ``reset()`` clears both buffer and flag — safe to call between
       independent protein refinement runs or MD trajectories.
    •  Buffer auto-migrates to the device of the incoming tensor, so
       CPU checkpoints loaded onto GPU work without manual ``.to(device)``.

    Args:
        epsilon_fp    : EMA blending factor ∈ (0, 1).
        sigma_target  : reference stress (stored for downstream use,
                        not used inside forward()).
    """

    def __init__(
        self,
        epsilon_fp:   float = 0.0028,
        sigma_target: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 < epsilon_fp < 1.0):
            raise ValueError(
                f"epsilon_fp must be in (0, 1); got {epsilon_fp!r}.")
        self.eps    = epsilon_fp
        self.target = sigma_target
        self.register_buffer("prev_sigma",   torch.tensor(0.0))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset(self) -> None:
        """
        Reset EMA state.  Call between independent trajectories or
        refinement runs so that stale stress history does not bleed in.
        """
        self.prev_sigma.zero_()
        self._initialized.fill_(False)

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_sigma : scalar stress tensor (differentiable).
        Returns:
            Filtered stress scalar (same device / dtype as ``raw_sigma``).
        """
        # Migrate buffers if needed (e.g. after .to(device) call)
        if self.prev_sigma.device != raw_sigma.device:
            self.prev_sigma   = self.prev_sigma.to(raw_sigma.device)
            self._initialized = self._initialized.to(raw_sigma.device)

        if not self._initialized.item():
            self.prev_sigma.data = raw_sigma.detach()
            self._initialized.fill_(True)
            return raw_sigma

        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


# =============================================================================
# 2. CSOC Base — Paper 4
# =============================================================================

class CSOCBase(nn.Module, ABC):
    """
    Abstract base class for CSOC adaptive-parameter modules  (Paper 4).

    Provides the shared SSC filter, ``reset()``, and two helper methods
    (``_normalised_deviation`` and ``_smooth_boost``) so that subclasses
    — :class:`CSOCThermostat` in ``structural_langevin.py`` and
    :class:`SOCController` in ``real_fold_one.py`` — share consistent logic.

    Args:
        sigma_target : reference structural stress.
        epsilon_fp   : SSC EMA blending factor.
        boost_factor : maximum parameter multiplier at high stress.
    """

    def __init__(
        self,
        sigma_target: float = 1.0,
        epsilon_fp:   float = 0.0028,
        boost_factor: float = 3.0,
    ) -> None:
        super().__init__()
        if sigma_target <= 0:
            raise ValueError(f"sigma_target must be positive; got {sigma_target!r}.")
        if boost_factor < 1.0:
            raise ValueError(f"boost_factor must be ≥ 1; got {boost_factor!r}.")
        self.sigma_target = sigma_target
        self.boost_factor = boost_factor
        self.ssc = SemanticStateContraction(epsilon_fp, sigma_target)

    def reset(self) -> None:
        """Reset SSC EMA state (call between independent runs)."""
        self.ssc.reset()

    def _normalised_deviation(self, sigma: torch.Tensor) -> torch.Tensor:
        """(σ − σ_target) / σ_target  — scalar deviation from criticality."""
        return (sigma - self.sigma_target) / max(self.sigma_target, 1e-12)

    def _smooth_boost(self, dev: torch.Tensor) -> torch.Tensor:
        """Sigmoid boost ∈ (0, 1) for smooth parameter interpolation."""
        return torch.sigmoid(dev)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute adaptive parameters from current structural state."""


# =============================================================================
# 3. Interface Detector Base
# =============================================================================

class InterfaceDetectorBase(nn.Module, ABC):
    """
    Abstract base for differentiable interface / sharp-gradient detectors.

    Subclasses must return a tensor ∈ [0, 1] that is fully differentiable
    w.r.t. the inputs.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Returns a mask tensor ∈ [0, 1]."""


# =============================================================================
# 4. Structural Itô Base — Papers 2 & 3
# =============================================================================

class StructuralItoBase(nn.Module, ABC):
    """
    Abstract base class for Structural Itô drift-correction modules.

    Both the Langevin integrator (per-atom, shape N×3) and the continuum
    FH solver (per-cell) implement the same ½ G(x) ∇_x G(x) correction;
    only the dimensionality and interface detector differ.

    Args:
        interface_amplification : G-field amplitude boost at interfaces.
    """

    def __init__(self, interface_amplification: float = 2.0) -> None:
        super().__init__()
        if interface_amplification < 0:
            raise ValueError(
                f"interface_amplification must be ≥ 0; got {interface_amplification!r}.")
        self.amp = interface_amplification

    def get_g_field(self, interface_mask: torch.Tensor) -> torch.Tensor:
        """G(x) = 1 + amp · mask(x).  Identical formula in all domains."""
        return 1.0 + self.amp * interface_mask

    @abstractmethod
    def compute_ito_correction(
        self,
        field: torch.Tensor,
        interface_detector: InterfaceDetectorBase,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute ½ G(x) ∇_x G(x).

        Returns:
            Itô drift tensor, same shape as ``field``, **detached**.
        """


# =============================================================================
# 5. Langevin Bridge
# =============================================================================

class LangevinBridge:
    """
    Bridge between :class:`real_fold_one.RefinementEngine` and
    :class:`structural_langevin.AdvancedStructuralLangevin`.

    Problem this solves
    ───────────────────
    ``RefinementEngine`` has a built-in Langevin step (``use_langevin=True``)
    that uses a simple Euler-Maruyama discretisation with isotropic Gaussian
    noise — it does *not* use the BAOAB splitting or the Structural Itô /
    CSOC extensions from Paper 3/4.

    ``AdvancedStructuralLangevin`` implements the full BAOAB integrator with
    multiplicative noise and Itô correction, but knows nothing about OpenMM
    energies or protein-specific force fields.

    This bridge lets ``RefinementEngine`` call ``AdvancedStructuralLangevin``
    for its stochastic integration step while still using OpenMM / SOC for
    the energy / force computation.

    Usage::

        from one_core_fold import LangevinBridge
        from real_fold_one import RefinementEngine, RefinementConfig
        from structural_langevin import AdvancedStructuralLangevin

        engine     = RefinementEngine(RefinementConfig(use_langevin=False))
        integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=300.0)
        bridge     = LangevinBridge(engine, integrator)

        # Inside your refinement loop:
        coords, velocities = bridge.step(coords, velocities, jumps=None)

    Args:
        refinement_engine : a ``RefinementEngine`` instance (provides force_fn).
        langevin           : an ``AdvancedStructuralLangevin`` instance.
        kb                 : Boltzmann constant in the engine's energy units.
                             Default: 0.001987 kcal mol⁻¹ K⁻¹.
    """

    def __init__(
        self,
        refinement_engine,         # RefinementEngine — not type-annotated to
        langevin,                  # avoid circular imports at module level
        kb: float = 0.001987,
    ) -> None:
        self.engine    = refinement_engine
        self.langevin  = langevin
        self.kb        = kb
        self._velocities: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset Langevin velocities and integrator state."""
        self._velocities = None
        self.langevin.reset()

    # ------------------------------------------------------------------

    def _force_fn(
        self,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wraps the RefinementEngine's energy computation into the
        ``(energy, force)`` callable expected by
        ``AdvancedStructuralLangevin.full_step()``.

        The force is the negative gradient of the total energy
        (OpenMM + SOC + alpha regularisation).

        Args:
            coords : (N_solute, 3) float32 tensor, requires_grad=True.
        Returns:
            energy : scalar tensor.
            force  : (N_solute, 3) tensor  [= −∂E/∂coords].
        """
        from real_fold_one import openmm_solute_energy  # local import to avoid circular

        coords = coords.requires_grad_(True)

        # OpenMM energy (differentiable via TorchForce or fallback)
        E_openmm = openmm_solute_energy(coords, self.engine.calculator)

        # SOC energy (requires neighbour list built on detached coords)
        ca = self.engine._get_ca_coords(coords)
        ca_det = ca.detach()
        edge_dict = self.engine.neighbor_mgr.build(ca_det)
        alpha = self.engine._alpha if hasattr(self.engine, "_alpha") else \
                torch.ones(ca.shape[0], device=coords.device, dtype=coords.dtype)
        E_soc = self.engine.soc.compute_soc_energy(
            ca, alpha,
            edge_dict["soc"][0], edge_dict["soc"][1],
            w_soc=self.engine.cfg.w_soc,
        )

        E_total = E_openmm + E_soc
        force   = -torch.autograd.grad(E_total, coords, create_graph=False)[0]
        return E_total.detach(), force.detach()

    # ------------------------------------------------------------------

    def step(
        self,
        coords:     torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        jumps:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one full BAOAB Langevin step using
        ``AdvancedStructuralLangevin.full_step()``.

        Args:
            coords     : (N, 3) current atomic positions (Å).
            velocities : (N, 3) current velocities, or None (→ zeros on first call).
            jumps      : (N, 3) BV jump vectors, or None (→ zeros).

        Returns:
            new_coords     : (N, 3) updated positions.
            new_velocities : (N, 3) updated velocities.
        """
        N      = coords.shape[0]
        device = coords.device
        dtype  = coords.dtype

        if velocities is None:
            if self._velocities is None or self._velocities.shape[0] != N:
                self._velocities = torch.zeros(N, 3, device=device, dtype=dtype)
            velocities = self._velocities

        new_coords, new_velocities, T_sc, sigma_sc = self.langevin.full_step(
            coords,
            velocities,
            self._force_fn,
            jumps=jumps,
        )

        self._velocities = new_velocities.detach()
        logger.debug(
            "LangevinBridge step: T=%.2f K  σ=%.4f Å",
            T_sc, sigma_sc,
        )
        return new_coords.detach(), new_velocities.detach()

    # ------------------------------------------------------------------

    def run(
        self,
        coords:     torch.Tensor,
        n_steps:    int,
        jumps:      Optional[torch.Tensor] = None,
        log_every:  int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run ``n_steps`` of Structural Langevin MD via the bridge.

        Args:
            coords    : (N, 3) initial positions (Å).
            n_steps   : number of integration steps.
            jumps     : (N, 3) BV jump vectors, constant across steps.
            log_every : log diagnostics every this many steps.

        Returns:
            final_coords     : (N, 3)
            final_velocities : (N, 3)
        """
        velocities = torch.zeros_like(coords)
        for step in range(n_steps):
            coords, velocities = self.step(coords, velocities, jumps=jumps)
            if log_every > 0 and step % log_every == 0:
                logger.info(
                    "Bridge MD step %d / %d  |x|_mean=%.4f Å",
                    step, n_steps,
                    coords.norm(dim=-1).mean().item(),
                )
        return coords, velocities


# =============================================================================
# Module banner
# =============================================================================

logger.debug("ONE Core Fold v%s loaded.", FOLD_VERSION)
