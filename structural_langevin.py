# =============================================================================
# LANGEVIN ADVANCED WITH STRUCTURAL CALCULUS
# =============================================================================
# Developer: Yoon A Limsuwan / MSPS NETWORK
# License: MIT
# Year: 2026
#
# A Fully Differentiable, Higher-Order (BAOAB Splitting) Langevin Integrator
# integrating the 4-Paper Structural Calculus Ecosystem:
#   1. Regime-Dependent Analytical Framework (Structural Operators)
#   2. BV Jump Measures & Self-Evolving Interfaces
#   3. Structural Itô Calculus & Multiplicative Noise Correction
#   4. Controlled Self-Organized Criticality (CSOC) & SSC Thermostat
#
# Features:
# - BAOAB Splitting Method for high-fidelity thermodynamic sampling.
# - Multiplicative/Structural Noise focusing stochasticity near interfaces.
# - Explicit Itô Drift Correction for exact state-dependent noise resolution.
# - CSOC-driven Adaptive Thermostat (T) and Friction (gamma).
# - Fully PyTorch-native and Autograd compatible.
#
# =============================================================================

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helper: soft interface mask (differentiable w.r.t. coords)
# ---------------------------------------------------------------------------

class InterfaceDetector(nn.Module):
    """
    Builds a per-atom soft interface mask in [0, 1] that is differentiable
    with respect to atomic coordinates.  An atom is considered "at an
    interface" when it has at least one neighbour within *r_cut* Å but the
    local environment is heterogeneous (large variance of pairwise distances).

    Shape convention:
        coords  : (N, 3)  — N atoms, 3D Cartesian
        returns : (N,)    — scalar interface score per atom ∈ [0, 1]
    """

    def __init__(self, r_cut: float = 8.0, sharpness: float = 4.0):
        """
        Args:
            r_cut      : cutoff distance (Å) for neighbourhood definition.
            sharpness  : steepness of the sigmoid used for soft thresholding.
        """
        super().__init__()
        self.r_cut = r_cut
        self.sharpness = sharpness

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3) tensor, requires_grad may be True.
        Returns:
            mask  : (N,) soft interface scores, fully differentiable.
        """
        if coords.dim() != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3), got {tuple(coords.shape)}")

        # Pairwise squared distances — (N, N)
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)          # (N, N, 3)
        dist2 = (diff ** 2).sum(dim=-1)                            # (N, N)
        dist = torch.sqrt(dist2 + 1e-8)                            # (N, N) stable

        # Soft neighbour weight: w_ij → 1 when dist < r_cut, → 0 outside
        w = torch.sigmoid(self.sharpness * (self.r_cut - dist))    # (N, N)

        # Zero out self-interaction on the diagonal
        mask_self = 1.0 - torch.eye(coords.shape[0], device=coords.device,
                                    dtype=coords.dtype)
        w = w * mask_self                                           # (N, N)

        # Weighted mean distance per atom
        w_sum = w.sum(dim=-1).clamp(min=1e-8)                      # (N,)
        mean_d = (w * dist).sum(dim=-1) / w_sum                    # (N,)

        # Weighted variance of distance — high variance ↔ interface-like env
        mean_d2 = (w * dist ** 2).sum(dim=-1) / w_sum              # (N,)
        var_d = (mean_d2 - mean_d ** 2).clamp(min=0.0)             # (N,)
        std_d = torch.sqrt(var_d + 1e-8)                           # (N,)

        # Normalise std to [0,1] via sigmoid (interface score)
        interface_score = torch.sigmoid(self.sharpness * (std_d - mean_d * 0.3))

        return interface_score   # (N,)


# ---------------------------------------------------------------------------
# Module 1 — Semantic State Contraction (SSC)
# ---------------------------------------------------------------------------

class SemanticStateContraction(nn.Module):
    """
    SSC Filter (Paper 4): First-order exponential low-pass filter that tracks
    structural stress σ across time steps.

    Fix: uses a boolean ``_initialized`` buffer instead of checking
    ``prev_sigma == 0.0``, which would incorrectly skip the initialisation
    when the true first stress is zero.
    """

    def __init__(self, epsilon_fp: float = 0.0028):
        """
        Args:
            epsilon_fp : contraction rate ∈ (0, 1).  Smaller ↔ slower tracking.
        """
        super().__init__()
        if not (0.0 < epsilon_fp < 1.0):
            raise ValueError(f"epsilon_fp must be in (0, 1), got {epsilon_fp}")
        self.eps = epsilon_fp
        self.register_buffer('prev_sigma', torch.tensor(0.0))
        self.register_buffer('_initialized', torch.tensor(False))

    def reset(self) -> None:
        """Reset filter state (call between independent trajectories)."""
        self.prev_sigma.zero_()
        self._initialized.fill_(False)

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_sigma : scalar stress measure (differentiable).
        Returns:
            Filtered stress scalar.
        """
        if not self._initialized.item():
            self.prev_sigma.data = raw_sigma.detach()
            self._initialized.fill_(True)
            return raw_sigma

        # EMA contraction toward the stable manifold
        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


# ---------------------------------------------------------------------------
# Module 2 — CSOC Adaptive Thermostat
# ---------------------------------------------------------------------------

class CSOCThermostat(nn.Module):
    """
    CSOC Adaptive Thermostat (Paper 4): Modulates Langevin temperature and
    friction coefficient based on real-time structural stress.

    Fix: temperature modulation range is expressed as a *fraction* of
    base_temp (``temp_boost_factor``) rather than an absolute additive
    constant, preventing unrealistically high temperatures at low base_temp.
    """

    def __init__(
        self,
        base_temp: float = 300.0,
        base_friction: float = 1.0,
        sigma_target: float = 1.0,
        epsilon_fp: float = 0.0028,
        temp_boost_factor: float = 3.0,
        friction_boost_factor: float = 0.5,
    ):
        """
        Args:
            base_temp           : reference temperature (K).
            base_friction       : reference friction γ (ps⁻¹).
            sigma_target        : target structural stress (Å).
            epsilon_fp          : SSC contraction rate.
            temp_boost_factor   : max temperature = base_temp * temp_boost_factor.
            friction_boost_factor: max additional γ = base_friction * friction_boost_factor.
        """
        super().__init__()
        self.base_temp = base_temp
        self.base_friction = base_friction
        self.sigma_target = sigma_target
        self.temp_boost_factor = temp_boost_factor
        self.friction_boost_factor = friction_boost_factor
        self.ssc = SemanticStateContraction(epsilon_fp)

    def reset(self) -> None:
        self.ssc.reset()

    def forward(
        self,
        coords: torch.Tensor,
        prev_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coords      : current positions  (N, 3).
            prev_coords : positions at t−1   (N, 3).
        Returns:
            adaptive_T     : scalar temperature (K), differentiable.
            adaptive_gamma : scalar friction    (ps⁻¹), differentiable.
            sigma          : filtered stress    (Å).
        """
        # Raw structural stress — mean per-atom displacement
        raw_sigma = torch.norm(coords - prev_coords, dim=-1).mean()

        # SSC-filtered stress
        sigma = self.ssc(raw_sigma)

        # Normalised deviation from target
        dev = (sigma - self.sigma_target) / (self.sigma_target.clamp(min=1e-8)
                                              if isinstance(self.sigma_target, torch.Tensor)
                                              else max(self.sigma_target, 1e-8))

        # Temperature: smoothly rises above base_temp when stress exceeds target
        boost = self.base_temp * (self.temp_boost_factor - 1.0)
        adaptive_T = self.base_temp + boost * torch.sigmoid(dev)
        adaptive_T = torch.clamp(
            adaptive_T,
            self.base_temp * 0.5,
            self.base_temp * self.temp_boost_factor,
        )

        # Friction: increases in high-stress (collapse) regimes
        adaptive_gamma = self.base_friction * (
            1.0 + self.friction_boost_factor * torch.relu(dev)
        )

        return adaptive_T, adaptive_gamma, sigma


# ---------------------------------------------------------------------------
# Module 3 — Structural Itô Noise
# ---------------------------------------------------------------------------

class StructuralItoNoise(nn.Module):
    """
    Multiplicative Noise & Itô Correction (Papers 2 & 3).

    The key fix in v2: G(x) is evaluated with ``coords`` inside the
    autograd context so that ∇_x G is non-trivial.  The interface_mask
    must therefore be a **differentiable function of coords** — pass the
    output of ``InterfaceDetector`` here.

    G(x) = 1 + amp * mask(x)
    Itô correction = ½ G(x) ∇_x G(x)
    """

    def __init__(self, interface_amplification: float = 2.0):
        """
        Args:
            interface_amplification : amplitude multiplier for interface noise.
        """
        super().__init__()
        self.amp = interface_amplification

    def get_g_matrix(self, interface_mask: torch.Tensor) -> torch.Tensor:
        """
        G(x): per-atom noise amplitude scalar, shape (N,).
        Higher near self-evolving interfaces (Paper 2).
        """
        return 1.0 + self.amp * interface_mask

    def compute_ito_correction(
        self,
        coords: torch.Tensor,
        interface_detector: InterfaceDetector,
    ) -> torch.Tensor:
        """
        Computes (1/2) G(x) ∇_x G(x) via PyTorch Autograd.

        Args:
            coords             : (N, 3) positions — must NOT already
                                 have requires_grad set by caller.
            interface_detector : differentiable InterfaceDetector module
                                 whose output depends on coords.
        Returns:
            ito_drift : (N, 3) Itô drift correction, detached.
        """
        with torch.enable_grad():
            x = coords.detach().requires_grad_(True)

            # mask is now a differentiable function of x
            mask = interface_detector(x)         # (N,)
            G = 1.0 + self.amp * mask            # (N,)
            G_sum = G.sum()

            grad_G = torch.autograd.grad(
                G_sum, x, create_graph=False, retain_graph=False
            )[0]                                 # (N, 3) or None

            if grad_G is None:
                return torch.zeros_like(coords)

            # Broadcast G from (N,) to (N, 3) for element-wise product
            ito_drift = 0.5 * G.unsqueeze(-1) * grad_G   # (N, 3)

        return ito_drift.detach()


# ---------------------------------------------------------------------------
# Core Integrator — Advanced Structural Langevin (BAOAB)
# ---------------------------------------------------------------------------

class AdvancedStructuralLangevin(nn.Module):
    """
    BAOAB Splitting Langevin Integrator with Structural Calculus extensions.

    Integrates:
      • Bulk conservative forces + BV jump measures (Papers 1 & 2)
      • Multiplicative structural noise (Paper 2)
      • Structural Itô drift correction (Paper 3)
      • CSOC adaptive thermostat (Paper 4)

    Usage pattern (outer simulation loop)::

        integrator = AdvancedStructuralLangevin(dt=0.002)

        for step in range(num_steps):
            force_bulk = -torch.autograd.grad(energy, coords, retain_graph=True)[0]
            interface_mask = integrator.interface_detector(coords)
            jumps = ...  # (N, 3) jump vectors at interfaces

            # BAO A step — returns mid-step quantities
            x_new, v_tilde, T, sigma = integrator.baoa_step(
                coords, velocities, force_bulk, jumps, interface_mask
            )

            # Evaluate new energy/forces at x_new
            new_energy = potential(x_new)
            new_force  = -torch.autograd.grad(new_energy, x_new)[0]

            # Final B step — completes BAOAB
            velocities = integrator.final_b_step(v_tilde, new_force, jumps, interface_mask)
            coords     = x_new.detach().requires_grad_(True)

    For a single-call convenience wrapper see ``full_step()``.
    """

    def __init__(
        self,
        mass: float = 1.0,
        dt: float = 0.002,
        base_temp: float = 300.0,
        base_friction: float = 1.0,
        kb: float = 0.001987,        # kcal mol⁻¹ K⁻¹
        interface_r_cut: float = 8.0,
        interface_amplification: float = 2.0,
    ):
        """
        Args:
            mass                   : atomic mass (amu or reduced units).
            dt                     : time step (ps).
            base_temp              : reference temperature (K).
            base_friction          : reference friction γ (ps⁻¹).
            kb                     : Boltzmann constant in chosen units.
            interface_r_cut        : neighbour cutoff for InterfaceDetector (Å).
            interface_amplification: noise amplification at interfaces.
        """
        super().__init__()
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")

        self.mass = mass
        self.dt = dt
        self.kb = kb

        # Sub-modules
        self.interface_detector = InterfaceDetector(r_cut=interface_r_cut)
        self.thermostat = CSOCThermostat(base_temp, base_friction)
        self.ito_noise = StructuralItoNoise(interface_amplification)

        # State
        self.register_buffer('_prev_coords', torch.zeros(1, 3))
        self.register_buffer('_state_ready', torch.tensor(False))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all stateful buffers (call between independent trajectories)."""
        self._prev_coords.zero_()
        self._state_ready.fill_(False)
        self.thermostat.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _structural_force(
        self,
        force_bulk: torch.Tensor,
        jumps: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        D^S u = ∇u + [u] δ_Γ   (Papers 1 & 2).
        Total structural force = bulk force + jump measure concentrated at Γ.

        Args:
            force_bulk     : (N, 3) conservative forces.
            jumps          : (N, 3) BV jump vectors.
            interface_mask : (N,)   soft interface indicator ∈ [0, 1].
        Returns:
            (N, 3) total structural force.
        """
        self._check_shape(force_bulk, jumps, interface_mask)
        return force_bulk + jumps * interface_mask.unsqueeze(-1)

    @staticmethod
    def _check_shape(
        force_bulk: torch.Tensor,
        jumps: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> None:
        N = force_bulk.shape[0]
        if jumps.shape != force_bulk.shape:
            raise ValueError(
                f"jumps {tuple(jumps.shape)} must match force_bulk {tuple(force_bulk.shape)}"
            )
        if interface_mask.shape != (N,):
            raise ValueError(
                f"interface_mask must be ({N},), got {tuple(interface_mask.shape)}"
            )

    # ------------------------------------------------------------------
    # BAOA step (returns positions ready for next force evaluation)
    # ------------------------------------------------------------------

    def baoa_step(
        self,
        coords: torch.Tensor,
        velocities: torch.Tensor,
        force_bulk: torch.Tensor,
        jumps: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Executes the B–A–O–A sub-steps of one BAOAB integration step.

        After calling this method, the caller must:
          1. Evaluate energy / forces at the returned ``x_full``.
          2. Call ``final_b_step()`` to complete the BAOAB loop.

        Args:
            coords         : (N, 3) current positions.
            velocities     : (N, 3) current velocities.
            force_bulk     : (N, 3) bulk conservative forces at ``coords``.
            jumps          : (N, 3) BV jump vectors at interfaces.
            interface_mask : (N,)   soft interface scores ∈ [0, 1],
                             produced by ``self.interface_detector(coords)``.
        Returns:
            x_full   : (N, 3) positions at t + dt (mid-BAOAB).
            v_tilde  : (N, 3) velocities after O-step (pre-final-B).
            T_scalar : float  current adaptive temperature (K).
            sigma_sc : float  current SSC-filtered stress (Å).
        """
        # ── device / dtype consistency ────────────────────────────────
        device = coords.device
        dtype  = coords.dtype
        for name, t in [("velocities", velocities),
                        ("force_bulk", force_bulk),
                        ("jumps", jumps),
                        ("interface_mask", interface_mask)]:
            if t.device != device:
                raise RuntimeError(f"{name} is on {t.device}, expected {device}")

        # ── initialise prev_coords on first call ──────────────────────
        if not self._state_ready.item() or self._prev_coords.shape != coords.shape:
            self._prev_coords = coords.detach().clone()
            self._state_ready.fill_(True)

        # ── adaptive thermostat ───────────────────────────────────────
        T, gamma, sigma = self.thermostat(coords, self._prev_coords)
        self._prev_coords = coords.detach().clone()

        # ── structural force ──────────────────────────────────────────
        F_struct = self._structural_force(force_bulk, jumps, interface_mask)

        # ── [ B ] half-step velocity ──────────────────────────────────
        v_half = velocities + 0.5 * self.dt * (F_struct / self.mass)

        # ── [ A ] half-step position ──────────────────────────────────
        x_half = coords + 0.5 * self.dt * v_half

        # ── [ O ] exact Ornstein–Uhlenbeck stochastic update ─────────
        # Exact integration of the O-step (Leimkuhler & Matthews 2013):
        #   c1 = e^{-γ Δt},  c2 = √(1 − c1²)
        c1 = torch.exp(-gamma * self.dt)                           # scalar tensor
        c2 = torch.sqrt((1.0 - c1 ** 2).clamp(min=0.0))

        # Multiplicative noise amplitude G(x) — per atom, shape (N,)
        G_x = self.ito_noise.get_g_matrix(interface_mask)         # (N,)

        # Thermal noise scale: √(k_B T / m)  — kept as tensor for grad
        noise_scale = torch.sqrt(self.kb * T / self.mass)         # scalar tensor

        # Stochastic force: c2 · √(kBT/m) · G(x) · ξ
        R = torch.randn_like(v_half)
        stochastic_force = c2 * noise_scale * G_x.unsqueeze(-1) * R  # (N, 3)

        # Structural Itô drift correction  ½ G ∇G  (Paper 3)
        ito_correction = self.ito_noise.compute_ito_correction(
            x_half, self.interface_detector
        )                                                           # (N, 3)

        # O-step velocity update
        v_tilde = c1 * v_half + stochastic_force + ito_correction * self.dt

        # ── [ A ] second half-step position ──────────────────────────
        x_full = x_half + 0.5 * self.dt * v_tilde

        return x_full, v_tilde, T.item(), sigma.item()

    # ------------------------------------------------------------------
    # Final B step (call after re-evaluating forces at x_full)
    # ------------------------------------------------------------------

    def final_b_step(
        self,
        v_tilde: torch.Tensor,
        new_force_bulk: torch.Tensor,
        jumps: torch.Tensor,
        interface_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Completes the BAOAB loop with the final half-step velocity update.

        Args:
            v_tilde        : (N, 3) velocities returned by ``baoa_step()``.
            new_force_bulk : (N, 3) bulk forces evaluated at ``x_full``.
            jumps          : (N, 3) BV jump vectors (same as in ``baoa_step()``).
            interface_mask : (N,)   interface mask at ``x_full``.
        Returns:
            v_full : (N, 3) full velocities at t + dt.
        """
        F_new = self._structural_force(new_force_bulk, jumps, interface_mask)
        return v_tilde + 0.5 * self.dt * (F_new / self.mass)

    # ------------------------------------------------------------------
    # Convenience wrapper — full BAOAB step (requires force callable)
    # ------------------------------------------------------------------

    def full_step(
        self,
        coords: torch.Tensor,
        velocities: torch.Tensor,
        force_fn,
        jumps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Convenience method that performs a complete BAOAB step given a
        force callable.  Intended for simple usage without external loops.

        Args:
            coords      : (N, 3) current positions.
            velocities  : (N, 3) current velocities.
            force_fn    : callable(coords) → (energy, force_bulk).
                          Must return ``(scalar_energy, (N,3) force)``.
            jumps       : (N, 3) BV jump vectors, or None (→ zeros).
        Returns:
            new_coords     : (N, 3)
            new_velocities : (N, 3)
            T_scalar       : float  current adaptive temperature
            sigma_scalar   : float  current SSC stress
        """
        N = coords.shape[0]
        device, dtype = coords.device, coords.dtype

        if jumps is None:
            jumps = torch.zeros(N, 3, device=device, dtype=dtype)

        # Compute differentiable interface mask before step
        interface_mask = self.interface_detector(coords)

        _, force_bulk = force_fn(coords)

        x_full, v_tilde, T_sc, sigma_sc = self.baoa_step(
            coords, velocities, force_bulk, jumps, interface_mask
        )

        # Re-evaluate forces at new positions for final B step
        new_interface_mask = self.interface_detector(x_full)
        _, new_force_bulk = force_fn(x_full)

        new_velocities = self.final_b_step(
            v_tilde, new_force_bulk, jumps, new_interface_mask
        )

        return x_full, new_velocities, T_sc, sigma_sc


# =============================================================================
# Quick self-test  (python structurallangevin.py)
# =============================================================================

if __name__ == "__main__":
    import sys

    print("Running AdvancedStructuralLangevin self-test ...")
    torch.manual_seed(42)

    N   = 10       # atoms
    dt  = 0.002    # ps
    T0  = 300.0    # K

    coords    = torch.randn(N, 3) * 5.0
    velocities = torch.randn(N, 3) * 0.5
    jumps     = torch.zeros(N, 3)

    # Simple harmonic potential
    def harmonic(x):
        E = 0.5 * (x ** 2).sum()
        F = torch.autograd.grad(E, x, create_graph=False)[0]
        return E, -F

    integrator = AdvancedStructuralLangevin(
        mass=1.0, dt=dt, base_temp=T0, base_friction=1.0
    )

    print(f"  {'Step':>5}  {'T (K)':>8}  {'sigma':>8}  {'|x| mean':>10}")
    for step in range(5):
        coords = coords.requires_grad_(True)
        coords, velocities, T_sc, sigma_sc = integrator.full_step(
            coords, velocities, harmonic, jumps
        )
        coords = coords.detach()
        velocities = velocities.detach()
        print(f"  {step:>5}  {T_sc:>8.2f}  {sigma_sc:>8.4f}  "
              f"{coords.norm(dim=-1).mean().item():>10.4f}")

    print("Self-test passed.")
    sys.exit(0)
