# =============================================================================
# ONE CORE FOLD — Shared Foundation for the REAL FOLD ONE Ecosystem
# =============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
# License      : MIT
# Year         : 2026
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
#
# AI Co-Developers (mathematical derivation, architecture design, audits):
#   - Claude   (Anthropic)  — register_buffer patterns, full differentiability
#                             audit across all REAL FOLD ONE files, LangevinBridge
#                             design, FoldCahnHilliardBridge architecture,
#                             CahnHilliardSSCAdapter, cross-ecosystem integration
#   - GPT      (OpenAI)     — literature cross-check, energy unit conventions,
#                             OpenMM-ML API guidance
#   - Gemini   (Google)     — structural operator scaffolding, BAOAB splitting
#                             verification, abstract base class design
#   - DeepSeek              — alternative CSOC formulation review, SSC EMA
#                             convergence analysis
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
#   FoldCahnHilliardBridge    — cross-ecosystem bridge REAL FOLD ONE ↔ CH3D
#   CahnHilliardSSCAdapter    — wraps SemanticStateContraction for CH3D attach_ssc()
#   get_device                — unified hardware-backend selector
#   FOLD_VERSION              — ecosystem-wide version string
# =============================================================================

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

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
        self._alpha_cache = None
        self.langevin.reset()

    def set_alpha(self, alpha: torch.Tensor) -> None:
        """
        Cache the current alpha tensor so that _force_fn can use it for
        SOC energy evaluation.  Call this each time alpha is updated in
        the outer refine() loop before calling bridge.step().

        Args:
            alpha : (N_ca,) learnable criticality weights (detached copy).
        """
        self._alpha_cache = alpha.detach().clone()

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
        # Bug 3 fix: RefinementEngine does not store alpha as an attribute —
        # alpha is a local variable inside refine().  Use bridge._alpha_cache
        # if set externally (bridge.set_alpha(alpha)), otherwise fall back to
        # uniform ones so SOC energy is evaluated (not silently skipped).
        alpha = getattr(self, "_alpha_cache", None)
        if alpha is None or alpha.shape[0] != ca.shape[0]:
            alpha = torch.ones(ca.shape[0], device=coords.device, dtype=coords.dtype)
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
# 6. CahnHilliardSSCAdapter  — wraps SSC for CH3D attach_ssc()
# =============================================================================

class CahnHilliardSSCAdapter(nn.Module):
    """
    Thin adapter that wraps :class:`SemanticStateContraction` for direct use
    with :class:`StructuralCahnHilliard3D.attach_ssc()`.

    The Cahn–Hilliard solver's ``_resolve_sigma()`` checks for a ``sigma``
    *attribute* on the attached SSC object.  However, ``SemanticStateContraction``
    stores the running estimate as ``prev_sigma``, not ``sigma``.  This adapter
    bridges that naming gap and keeps the mapping native and differentiable.

    Usage::

        ssc     = SemanticStateContraction(epsilon_fp=0.003)
        adapter = CahnHilliardSSCAdapter(ssc)
        ch3d.attach_ssc(adapter)

        # Inside your time loop (update before each CH step):
        adapter.update(raw_stress_scalar)   # e.g. u.std() or energy norm

    Args:
        ssc            : :class:`SemanticStateContraction` instance.
        broadcast_shape: Optional (Nx,Ny,Nz) tuple.  When set, ``sigma``
                         is broadcast to a full 3-D tensor so that
                         ``_resolve_sigma`` receives the correct shape.
                         If None, ``sigma`` is a scalar tensor.
    """

    def __init__(
        self,
        ssc: "SemanticStateContraction",
        broadcast_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.ssc             = ssc
        self.broadcast_shape = broadcast_shape
        # Expose ``sigma`` as a buffer so .to(device) migrates it automatically
        self.register_buffer("sigma", torch.tensor(1.0))

    def update(self, raw_stress: torch.Tensor) -> torch.Tensor:
        """
        Feed a new raw stress measurement through the SSC filter and
        update the public ``sigma`` attribute.

        Args:
            raw_stress : scalar (or reducible) differentiable tensor.
        Returns:
            filtered sigma scalar tensor (differentiable).
        """
        filtered = self.ssc(raw_stress.mean() if raw_stress.dim() > 0 else raw_stress)
        # Store for _resolve_sigma access — detach so CH doesn't back-prop through SSC
        self.sigma = filtered.detach()
        if self.broadcast_shape is not None:
            self.sigma = self.sigma.expand(self.broadcast_shape).clone()
        return filtered   # differentiable return for external use

    def forward(self, raw_stress: torch.Tensor) -> torch.Tensor:
        """Callable alias for update() — makes adapter usable as nn.Module."""
        return self.update(raw_stress)

    def reset(self) -> None:
        """Reset SSC state and cached sigma."""
        self.ssc.reset()
        self.sigma = torch.tensor(1.0)


# =============================================================================
# 7. FoldCahnHilliardBridge  — REAL FOLD ONE ↔ Structural CH3D
# =============================================================================

class FoldCahnHilliardBridge(nn.Module):
    """
    Cross-ecosystem bridge connecting the **REAL FOLD ONE** molecular
    refinement engine to the **Structural Cahn–Hilliard 3D** continuum
    phase-field solver.

    Physical rationale
    ------------------
    Protein folding / structure refinement at the residue scale (Å) is
    described by REAL FOLD ONE's BAOAB Langevin dynamics.  The same
    system can exhibit mesoscale phase separation (e.g. protein condensate
    formation, lipid raft ordering, amphiphilic assembly) that is captured
    by a Cahn–Hilliard order parameter on a 3-D voxel grid.

    This bridge provides a **fully differentiable** two-way coupling:

    Fold → CH  (density projection)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Atomic coordinates (N_atoms, 3) → voxel density field (Nx, Ny, Nz)
    via a differentiable Gaussian kernel projection:

        rho(r) = sum_i  exp(-|r - r_i|^2 / (2*sigma_vox^2))

    normalised to [-1, 1] for the CH order parameter u.

    CH → Fold  (sigma-field back-coupling)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The CH structural sigma-field sigma(r) ∈ [sigma_min, ∞) is sampled
    at each atom position via trilinear interpolation and returned as a
    per-atom modulation of the Langevin noise amplitude:

        G_i = 1 + amp * interp(sigma, r_i)

    This back-coupling is **fully differentiable** w.r.t. both coords and
    the sigma field via PyTorch's grid_sample (differentiable trilinear
    interpolation).

    Usage::

        bridge  = FoldCahnHilliardBridge(grid_shape=(32,32,32),
                                          box_size=50.0)
        # Forward: coords → CH order parameter field
        u       = bridge.coords_to_field(coords)           # (Nx,Ny,Nz)

        # Backward: sigma field → per-atom noise scale
        g_atoms = bridge.sigma_to_atom_scale(sigma, coords) # (N_atoms,)

        # Compute CH sigma from u (optional SSC update)
        ch_sigma = bridge.field_to_sigma(u, ssc_adapter)    # (Nx,Ny,Nz)

    Args:
        grid_shape     : (Nx, Ny, Nz) voxel grid dimensions.
        box_size       : Simulation box side length in Å (isotropic cube
                         assumed; for non-cubic boxes pass a 3-tuple).
        sigma_vox      : Gaussian kernel width for density projection (Å).
        amp            : noise amplification factor at high-sigma voxels.
        sigma_min      : minimum sigma floor (matches CH config sigma_min).
        device         : target compute device.
    """

    def __init__(
        self,
        grid_shape:   Tuple[int, int, int] = (32, 32, 32),
        box_size:     Union[float, Tuple[float, float, float]] = 50.0,
        sigma_vox:    float = 3.0,
        amp:          float = 2.0,
        sigma_min:    float = 1e-3,
        device:       Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        try:
            from typing import Union  # noqa: F401
        except ImportError:
            pass
        self.grid_shape = grid_shape
        if isinstance(box_size, (int, float)):
            self.box_size: Tuple[float, float, float] = (
                float(box_size), float(box_size), float(box_size))
        else:
            self.box_size = tuple(box_size)  # type: ignore[assignment]
        self.sigma_vox = sigma_vox
        self.amp       = amp
        self.sigma_min = sigma_min
        self._device   = device or get_device()

        Nx, Ny, Nz = grid_shape
        Lx, Ly, Lz = self.box_size
        # Voxel-centre coordinates — registered as buffers (auto .to(device))
        xs = torch.linspace(0.0, Lx, Nx)
        ys = torch.linspace(0.0, Ly, Ny)
        zs = torch.linspace(0.0, Lz, Nz)
        GX, GY, GZ = torch.meshgrid(xs, ys, zs, indexing="ij")  # each (Nx,Ny,Nz)
        self.register_buffer("grid_x", GX)
        self.register_buffer("grid_y", GY)
        self.register_buffer("grid_z", GZ)

    # ------------------------------------------------------------------
    # Fold → CH  (differentiable density projection)
    # ------------------------------------------------------------------

    def coords_to_field(
        self,
        coords:        torch.Tensor,           # (N_atoms, 3)  Å
        weights:       Optional[torch.Tensor] = None,  # (N_atoms,)  optional
    ) -> torch.Tensor:
        """
        Project atomic coordinates onto a 3-D voxel grid as a soft
        density field via differentiable Gaussian kernels.

        Result is normalised to [-1, 1] (CH order-parameter convention):
            u = 2 * (rho / rho_max) - 1,  clipped to [-1, 1]

        Args:
            coords  : (N_atoms, 3) float32 tensor in Å, may have
                      ``requires_grad=True``.
            weights : (N_atoms,) optional per-atom weights.  Default: ones.
        Returns:
            u : (Nx, Ny, Nz) float32 tensor, fully differentiable w.r.t.
                ``coords`` and ``weights``.
        """
        if coords.dim() != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3); got {tuple(coords.shape)}")
        N      = coords.shape[0]
        device = coords.device
        dtype  = coords.dtype

        if weights is None:
            weights = torch.ones(N, device=device, dtype=dtype)

        # Grid centres: (Nx,Ny,Nz) → (Nx*Ny*Nz, 3) for batched distance
        gx = self.grid_x.to(device=device, dtype=dtype).reshape(-1)
        gy = self.grid_y.to(device=device, dtype=dtype).reshape(-1)
        gz = self.grid_z.to(device=device, dtype=dtype).reshape(-1)

        Nx, Ny, Nz = self.grid_shape
        Ng = Nx * Ny * Nz   # total voxels

        # (Ng, 3) voxel centres
        grid_pts = torch.stack([gx, gy, gz], dim=-1)  # (Ng, 3)

        # (N_atoms, 1, 3) - (1, Ng, 3) → (N_atoms, Ng, 3) → (N_atoms, Ng)
        diff    = coords.unsqueeze(1) - grid_pts.unsqueeze(0)  # (N, Ng, 3)
        dist2   = (diff ** 2).sum(dim=-1)                       # (N, Ng)
        kernel  = torch.exp(-dist2 / (2.0 * self.sigma_vox**2)) # (N, Ng)

        # Weighted sum over atoms → (Ng,) density
        rho = (weights.unsqueeze(1) * kernel).sum(dim=0)        # (Ng,)
        rho = rho.reshape(Nx, Ny, Nz)                           # (Nx,Ny,Nz)

        # Normalise to [-1, 1]
        rho_max = rho.max().clamp(min=1e-12)
        u       = 2.0 * (rho / rho_max) - 1.0
        return u

    # ------------------------------------------------------------------
    # CH → Fold  (differentiable trilinear interpolation)
    # ------------------------------------------------------------------

    def sigma_to_atom_scale(
        self,
        sigma:  torch.Tensor,   # (Nx, Ny, Nz)
        coords: torch.Tensor,   # (N_atoms, 3)  Å
    ) -> torch.Tensor:
        """
        Sample the CH structural sigma field at each atom's position via
        differentiable trilinear interpolation (torch.nn.functional.grid_sample).

        Returns G_i = 1 + amp * sigma_interp_i, shape (N_atoms,).

        Gradient flows back through both ``sigma`` and ``coords``.

        Args:
            sigma  : (Nx, Ny, Nz) structural regime field.
            coords : (N_atoms, 3) atomic positions in Å.
        Returns:
            g_atoms : (N_atoms,) noise amplification per atom.
        """
        Nx, Ny, Nz   = self.grid_shape
        Lx, Ly, Lz   = self.box_size
        device, dtype = sigma.device, sigma.dtype

        # Normalise atom coordinates to [-1, 1] for grid_sample
        cx = coords[:, 0] / Lx * 2.0 - 1.0  # (N,)
        cy = coords[:, 1] / Ly * 2.0 - 1.0
        cz = coords[:, 2] / Lz * 2.0 - 1.0

        # grid_sample expects (N, H, W, D, 3) in 3-D with align_corners=True
        # We treat each atom as a single-voxel "image" query
        grid = torch.stack([cz, cy, cx], dim=-1)           # (N, 3)  xyz→zyx
        grid = grid.view(1, N, 1, 1, 3)                     # (1, N, 1, 1, 3)

        # sigma: (Nx,Ny,Nz) → (1, 1, Nx, Ny, Nz) for grid_sample
        sigma_5d = sigma.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

        interp = torch.nn.functional.grid_sample(
            sigma_5d, grid,
            mode="bilinear", padding_mode="border", align_corners=True
        )                                                    # (1, 1, N, 1, 1)
        sigma_at_atoms = interp.squeeze()                    # (N,) or scalar

        g_atoms = 1.0 + self.amp * sigma_at_atoms
        return g_atoms   # (N_atoms,)

    # ------------------------------------------------------------------
    # CH sigma field from u via SSC
    # ------------------------------------------------------------------

    def field_to_sigma(
        self,
        u:           torch.Tensor,
        ssc_adapter: Optional["CahnHilliardSSCAdapter"] = None,
    ) -> torch.Tensor:
        """
        Derive a structural sigma field from the CH order parameter.

        If an ``ssc_adapter`` is provided, the interface stress (|grad u|)
        is passed through the SSC filter to produce a spatially uniform
        sigma scalar broadcast to the grid shape.

        Without an adapter, sigma = 1 + amp * |u| (differentiable proxy).

        Args:
            u           : (Nx, Ny, Nz) CH order parameter.
            ssc_adapter : optional :class:`CahnHilliardSSCAdapter`.
        Returns:
            sigma : (Nx, Ny, Nz) structural field.
        """
        if ssc_adapter is not None:
            dx = self.box_size[0] / self.grid_shape[0]
            gx = (torch.roll(u, -1, 0) - torch.roll(u, +1, 0)) / (2 * dx)
            gy = (torch.roll(u, -1, 1) - torch.roll(u, +1, 1)) / (2 * dx)
            gz = (torch.roll(u, -1, 2) - torch.roll(u, +1, 2)) / (2 * dx)
            grad_norm = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-12).mean()
            ssc_adapter.update(grad_norm)
            sigma = ssc_adapter.sigma.to(u.device)
            if sigma.shape != u.shape:
                sigma = sigma.expand(u.shape).clone()
            return sigma.clamp(min=self.sigma_min)
        else:
            return (1.0 + self.amp * u.abs()).clamp(min=self.sigma_min)

    # ------------------------------------------------------------------
    # Full coupled step
    # ------------------------------------------------------------------

    def coupled_step(
        self,
        coords:      torch.Tensor,
        ch_solver:   "nn.Module",
        ssc_adapter: Optional["CahnHilliardSSCAdapter"] = None,
        weights:     Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method: one full Fold→CH projection + CH time step.

        Steps:
          1. Project coords → u (density field)
          2. Derive sigma from u (via ssc_adapter or proxy)
          3. Advance CH solver by one dt: u_new = ch_solver.step(u, sigma)
          4. Compute per-atom noise scale from new sigma

        Args:
            coords      : (N_atoms, 3) atomic positions in Å.
            ch_solver   : :class:`StructuralCahnHilliard3D` (or subclass).
            ssc_adapter : optional SSC adapter for sigma derivation.
            weights     : optional per-atom projection weights.
        Returns:
            u_new      : (Nx, Ny, Nz) updated order parameter.
            sigma      : (Nx, Ny, Nz) structural sigma field.
            g_atoms    : (N_atoms,) per-atom noise amplification.
        """
        u     = self.coords_to_field(coords, weights)
        sigma = self.field_to_sigma(u, ssc_adapter)
        u_new = ch_solver.step(u, sigma)
        g_atoms = self.sigma_to_atom_scale(sigma, coords)
        return u_new, sigma, g_atoms


# =============================================================================
# Module banner
# =============================================================================

logger.debug("ONE Core Fold v%s loaded.", FOLD_VERSION)
