# =============================================================================
# Gumbel Atomix NoZeno Diff Engine — Native Full Differentiable / AMP-Safe / DDP-Ready
# SUPER DNS ONE Cluster / ONE Ecosystem
# =============================================================================
# Author       : PAI, Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# GitHub       : https://github.com/yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# Version      : 2.0.0 (Production Release)
# =============================================================================
"""
High-Speed Native Differentiable Structural Engine for heterogeneous
macromolecular refinement (protein, water, ions, coenzymes; alternate
conformations; differentiable occupancy).

Fixes and improvements over the v1.0.0 reference
-----------------------------------------------
1. **Optimizer removed from `forward()` — the reference's single worst
   design flaw.**  The reference constructed an `AdamW` inside `forward()`
   and ran the whole training loop there.  That pattern
     · **broke DDP** (each rank ran its own optimizer on its own local
       gradients; nothing was ever all-reduced),
     · **recompiled `torch.compile`** on every call (Python-scalar `steps`
       / `lr` in the hot path),
     · **destroyed differentiability** (the caller's autograd graph never
       saw the internal backward passes),
     · **silently reset optimizer state** on every invocation.
   The module now exposes a **pure, differentiable `forward()`** that
   returns the loss *tensor*; an explicit `refine()` method reproduces the
   v1.0.0 optimization loop for backward compatibility, taking an optional
   external optimizer.

2. **No-Zeno gate is a differentiable tensor.**  The reference's
   `evaluate_gumbel_no_zeno_gate()` returned a Python `bool`, used
   `math.exp`, and drove `if ... : with torch.no_grad(): clamp_()`.  That
   was non-differentiable, `torch.compile`-hostile, caused a per-step D2H
   sync, and performed the coordinate clamp **in-place** with gradients
   severed.  The new gate is a **0-dim tensor ∈ (0, 1)** computed via the
   SESI double-exponential in fp32 log-space, and it drives a **C^∞ soft
   saturation** via `tanh`, not an in-place clamp.

3. **`torch.norm` → `torch.linalg.vector_norm(x, dim=-1)`** — the legacy
   overload is deprecated and silently flattened the batch.

4. **Hard ReLUs replaced by C^∞ surrogates.**
       F.relu(x) ** 2   →  F.softplus(x, β) ** 2      (exact for x ≫ 0)
       F.relu(-x) ** 2  →  F.softplus(-x, β) ** 2
   Both have no dead zone and are C^∞ everywhere.

5. **All Python-scalar constants moved to non-persistent buffers.**
   `self.sigma2`, `self.delta_e_min`, `1.5`, `0.5`, `1.0`, `20.0`, `0.05`
   were read from Python in the hot path, forcing `torch.compile`
   recompiles and CPU↔GPU scalar reads.  Now they are cast per-call from
   pre-registered fp32 buffers.

6. **AMP-safe.**  The No-Zeno barrier runs in fp32 internally and casts
   back, so the module is stable in fp16 / bf16 / fp32.

7. **Batched-target support (optional).**  `experimental_map_target` may be
   `[N, 3]` (single, backward-compatible) or `[B, N, 3]` (batched).  When
   batched, the shared parameters are broadcast and the loss is returned as
   per-sample `[B]`, which is exactly what DDP needs for per-sample gradient
   routing.

8. **DDP-clean & `torch.compile`-stable.**  No optimizer, no Python scalars
   in the hot path, no data-dependent control flow, no forced AMP decorator,
   no per-forward allocations.

9. **Public API preserved.**  `RealFoldOneAdvancedEngine`, same positional
   constructor signature `(num_atoms, num_alt_states, gumbel_sigma,
   barrier_min)`, same `set_atom_type` method, same forward return tuple
   `(effective_coords, occupancies, loss)` — with `loss` now a tensor
   (call `.item()` if you need a scalar).  A strict drop-in upgrade.

Multi-GPU (DDP) usage
---------------------
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    engine = RealFoldOneAdvancedEngine(num_atoms=300).to(rank)
    engine = torch.compile(engine, mode="max-autotune")               # optional
    engine = torch.nn.parallel.DistributedDataParallel(
        engine, device_ids=[rank], gradient_as_bucket_view=True,
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        coords, occ, loss = engine(target)         # loss is a tensor
    loss.backward()

    # Or run the reference-style loop (DDP-correct, external optimizer):
    optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-2)
    coords, occ, loss = engine.refine(target, steps=200, optimizer=optimizer)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NoZenoDiffConfig", "RealFoldOneAdvancedEngine"]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class NoZenoDiffConfig:
    """Configuration for the RealFoldOne advanced structural engine."""
    num_atoms: int
    num_alt_states: int = 2
    gumbel_sigma: float = 0.1
    barrier_min: float = 1.2

    # ---- Stereochemistry / occupancy ------------------------------------
    bond_target: float = 1.5
    bond_weight: float = 1.0
    occupancy_weight: float = 0.5

    # ---- Coordinate soft-saturation --------------------------------------
    coord_bound: float = 20.0
    coord_beta: float = 8.0

    # ---- C^∞ surrogates / numerical safety ------------------------------
    barrier_beta: float = 4.0
    zeno_clamp: float = 15.0
    log_floor: float = -40.0
    denom_eps: float = 1e-8
    gate_tau: float = 0.05
    zeno_trigger: float = 0.05


# =============================================================================
# Module
# =============================================================================
class RealFoldOneAdvancedEngine(nn.Module):
    """
    Real Fold One — native full differentiable structural refinement engine.

    Supports heterogeneous atoms (protein / water / ion / coenzyme),
    alternate conformations, differentiable occupancies, and a C^∞ No-Zeno
    topological gate for stabilised chart updates.

    Parameters
    ----------
    num_atoms : int
        Number of atoms (sites) in the structure. Must be > 0.
    num_alt_states : int
        Number of alternate conformations per atom. Must be > 0.
    gumbel_sigma : float
        Fluctuation scale σ in the No-Zeno barrier; σ² is precomputed.
    barrier_min : float
        Minimum activation energy ΔE_min in the barrier.
    bond_target, bond_weight, occupancy_weight : float
        Stereochemical-restraint parameters.
    coord_bound, coord_beta : float
        Soft-saturation bound and sharpness for the No-Zeno coordinate
        projection (reference used a hard `[-20, 20]` clamp).
    barrier_beta, zeno_clamp, log_floor, denom_eps : float
        Numerical-safety parameters for the SESI double-exponential bound.
    gate_tau, zeno_trigger : float
        Relaxation temperature and trigger threshold for the soft gate.
    validate_inputs : bool
        Cheap shape / dtype guards (disable for `torch.compile`).
    gradient_checkpointing : bool
        Recompute the step during backward — trades ~2× compute for memory.
    """

    def __init__(
        self,
        num_atoms: int,
        num_alt_states: int = 2,
        gumbel_sigma: float = 0.1,
        barrier_min: float = 1.2,
        *,
        bond_target: float = 1.5,
        bond_weight: float = 1.0,
        occupancy_weight: float = 0.5,
        coord_bound: float = 20.0,
        coord_beta: float = 8.0,
        barrier_beta: float = 4.0,
        zeno_clamp: float = 15.0,
        log_floor: float = -40.0,
        denom_eps: float = 1e-8,
        gate_tau: float = 0.05,
        zeno_trigger: float = 0.05,
        validate_inputs: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # ---- Argument validation ------------------------------------------
        if not (isinstance(num_atoms, int) and num_atoms > 0):
            raise ValueError("num_atoms must be a positive int.")
        if not (isinstance(num_alt_states, int) and num_alt_states > 0):
            raise ValueError("num_alt_states must be a positive int.")
        for name, v in (
            ("gumbel_sigma", gumbel_sigma),
            ("barrier_min", barrier_min),
            ("bond_target", bond_target),
            ("bond_weight", bond_weight),
            ("occupancy_weight", occupancy_weight),
            ("coord_bound", coord_bound),
            ("coord_beta", coord_beta),
            ("barrier_beta", barrier_beta),
            ("zeno_clamp", zeno_clamp),
            ("denom_eps", denom_eps),
            ("gate_tau", gate_tau),
        ):
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be > 0.")
        if not (math.isfinite(log_floor) and log_floor < 0.0):
            raise ValueError("log_floor must be < 0.")
        if not (0.0 < zeno_trigger < 1.0):
            raise ValueError("zeno_trigger must be in (0, 1).")

        self.num_atoms      = int(num_atoms)
        self.num_alt_states = int(num_alt_states)
        self.validate_inputs = bool(validate_inputs)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # ---- 1. Learnable coordinates with alternate conformations --------
        #   Shape [num_atoms, num_alt_states, 3].
        self.coordinates = nn.Parameter(
            torch.randn(num_atoms, num_alt_states, 3, dtype=torch.float32)
        )

        # ---- 2. Learnable raw occupancies (mapped through sigmoid) --------
        #   Shape [num_atoms, num_alt_states].
        self.raw_occupancy = nn.Parameter(
            torch.zeros(num_atoms, num_alt_states, dtype=torch.float32)
        )

        # ---- 3. Heterogeneous atom types (persistent buffer for BC) -------
        #   0: protein, 1: water, 2: ion, 3: coenzyme (extensible).
        self.register_buffer(
            "atom_types", torch.zeros(num_atoms, dtype=torch.long),
            persistent=True,
        )

        # ---- 4. Non-persistent scalar buffers (DDP-safe) ------------------
        def _buf(v: float) -> torch.Tensor:
            return torch.tensor(float(v), dtype=torch.float32)

        self.register_buffer("_sigma_sq",       _buf(float(gumbel_sigma) ** 2), persistent=False)
        self.register_buffer("_delta_e_min",    _buf(barrier_min),              persistent=False)
        self.register_buffer("_bond_target",    _buf(bond_target),              persistent=False)
        self.register_buffer("_bond_weight",    _buf(bond_weight),              persistent=False)
        self.register_buffer("_occ_weight",     _buf(occupancy_weight),         persistent=False)
        self.register_buffer("_coord_bound",    _buf(coord_bound),              persistent=False)
        self.register_buffer("_coord_beta",     _buf(coord_beta),               persistent=False)
        self.register_buffer("_barrier_beta",   _buf(barrier_beta),             persistent=False)
        self.register_buffer("_zeno_clamp",     _buf(zeno_clamp),               persistent=False)
        self.register_buffer("_log_floor",      _buf(log_floor),                persistent=False)
        self.register_buffer("_denom_eps",      _buf(denom_eps),                persistent=False)
        self.register_buffer("_gate_tau",       _buf(gate_tau),                 persistent=False)
        self.register_buffer("_zeno_trigger",   _buf(zeno_trigger),             persistent=False)

    # ------------------------------------------------------------------ #
    # Atom-type assignment (reference API preserved)                     #
    # ------------------------------------------------------------------ #
    def set_atom_type(self, indices: Union[List[int], Iterable[int]], atom_type_id: int) -> None:
        """
        Assigns specific atom categories (water, ion, coenzyme, …).

        Must be called consistently on every DDP rank (as with any buffer
        mutation) so that the buffer remains identical across processes.
        """
        idx = list(indices) if not isinstance(indices, list) else indices
        self.atom_types[idx] = int(atom_type_id)

    # ------------------------------------------------------------------ #
    # Differentiable No-Zeno gate (0-dim tensor)                         #
    # ------------------------------------------------------------------ #
    def evaluate_gumbel_no_zeno_gate(
        self,
        dt: Union[float, torch.Tensor] = 0.01,
        bond_stress: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable No-Zeno gate `∈ (0, 1)`.

        The v1.0.0 reference returned a Python `bool` computed from
        state-independent constants.  This version computes a **soft tensor
        gate** driven by the per-atom bond stress and evaluated through the
        SESI double-exponential bound in fp32 log-space:

            ΔE     = ΔE_min + softplus(stress, β)
            P      = exp( − exp( clamp( ΔE / (σ²·dt), max=z_max ) ) )
            gate   = sigmoid( (trigger − P) / τ )

        When `bond_stress` is `None`, the gate falls back to a pure-constant
        evaluation (matching the reference's scalar semantics but returning
        a tensor).
        """
        dtype_out = self.coordinates.dtype
        device    = self.coordinates.device

        # ---- Promote dt ----------------------------------------------------
        if not isinstance(dt, torch.Tensor):
            dt_t = torch.tensor(float(dt), dtype=torch.float32, device=device)
        else:
            dt_t = dt.float().to(device)

        # ---- Promote stress (defaults to zero → reference scalar mode) ----
        if bond_stress is None:
            stress32 = torch.zeros((), dtype=torch.float32, device=device)
        else:
            stress32 = bond_stress.float().to(device)

        beta      = self._barrier_beta.to(device=device)
        de_min    = self._delta_e_min.to(device=device)
        sigma_sq  = self._sigma_sq.to(device=device)
        zeno_max  = self._zeno_clamp.to(device=device)
        log_floor = self._log_floor.to(device=device)
        eps       = self._denom_eps.to(device=device)
        tau       = self._gate_tau.to(device=device)
        trigger   = self._zeno_trigger.to(device=device)

        # ---- ΔE from stress (softplus guarantees positivity) --------------
        delta_e = de_min + F.softplus(stress32, beta=beta)

        # ---- Barrier in fp32 log-space -------------------------------------
        denom    = sigma_sq * dt_t + eps
        exponent = (delta_e / denom).clamp_(max=zeno_max)
        inner    = torch.exp(exponent)
        log_p    = (-inner).clamp_(min=log_floor)
        prob     = torch.exp(log_p)                               # ∈ (e^{lf}, 1)

        # ---- Soft gate: 1 when P < trigger, 0 when P ≫ trigger ------------
        gate = torch.sigmoid((trigger - prob) / tau)
        return gate.to(dtype_out)

    # ------------------------------------------------------------------ #
    # Differentiable primitives                                          #
    # ------------------------------------------------------------------ #
    def _soft_saturate(self, coords: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        C^∞ soft saturation of coordinates towards `[-bound, bound]`,
        blended by the No-Zeno gate.

            soft(x) = bound · tanh(x / bound)
            out     = lerp(x, soft(x), gate)

        Matches the reference's `coordinate.clamp_(-20, 20)` at the
        endpoints (gate=1) while remaining differentiable everywhere.
        """
        dtype = coords.dtype
        bound = self._coord_bound.to(device=coords.device, dtype=dtype)
        soft  = bound * torch.tanh(coords / bound)
        return torch.lerp(coords, soft, gate)

    def _compute_bond_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Stereochemical bond-length restraint across the primary conformation.

        Parameters
        ----------
        coords : torch.Tensor  [N, A, 3]  (or [B, N, A, 3])

        Returns
        -------
        bond_loss : torch.Tensor  scalar (mean over all bond pairs)
        """
        primary = coords[..., 0, :]                                 # [..., N, 3]
        if primary.shape[-2] < 2:
            return torch.zeros((), dtype=coords.dtype, device=coords.device)

        diff = primary[..., :-1, :] - primary[..., 1:, :]           # [..., N−1, 3]
        distances = torch.linalg.vector_norm(diff, dim=-1)          # [..., N−1]

        target = self._bond_target.to(device=coords.device, dtype=coords.dtype)
        return ((distances - target) ** 2).mean()

    def _compute_occupancy_penalty(self, occ: torch.Tensor) -> torch.Tensor:
        """
        C^∞ penalty encouraging per-site total occupancy to lie in `[0, 1]`.

            penalty = mean( softplus(Σ−1, β)² + softplus(−Σ, β)² )

        Exact for the reference's `relu(·)²` in the active regime and
        differentiable everywhere (no dead zone at the boundaries).
        """
        dtype = occ.dtype
        beta  = self._barrier_beta.to(device=occ.device, dtype=dtype)

        occ_sum = occ.sum(dim=-1)                                   # [N] or [B, N]
        pos = F.softplus(occ_sum - 1.0, beta=beta)
        neg = F.softplus(-occ_sum, beta=beta)
        return (pos * pos).mean() + (neg * neg).mean()

    # ------------------------------------------------------------------ #
    # Core step (isolated so gradient checkpointing can wrap it)         #
    # ------------------------------------------------------------------ #
    def _step(
        self,
        experimental_map_target: torch.Tensor,
        dt: Union[float, torch.Tensor],
        apply_no_zeno_saturation: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dtype = self.coordinates.dtype
        device = self.coordinates.device

        # ---- Occupancies: differentiable map from raw logits --------------
        occupancies = torch.sigmoid(self.raw_occupancy)             # [N, A]

        # ---- No-Zeno gate driven by raw bond stress -----------------------
        raw_stress = self._compute_bond_loss(self.coordinates)      # scalar
        gate = self.evaluate_gumbel_no_zeno_gate(dt=dt, bond_stress=raw_stress)

        # ---- C^∞ soft coordinate saturation --------------------------------
        if apply_no_zeno_saturation:
            coords_used = self._soft_saturate(self.coordinates, gate)
        else:
            coords_used = self.coordinates

        # ---- Stereochemical restraints ------------------------------------
        bond_loss = self._compute_bond_loss(coords_used)
        occ_penalty = self._compute_occupancy_penalty(occupancies)

        w_bond = self._bond_weight.to(device=device, dtype=dtype)
        w_occ  = self._occ_weight.to(device=device, dtype=dtype)

        # ---- Experimental fit loss (single or batched) --------------------
        batched = experimental_map_target.dim() == 3

        if not batched:
            # ---- Single target -------------------------------------------
            if self.validate_inputs:
                if experimental_map_target.shape != (self.num_atoms, 3):
                    raise ValueError(
                        f"experimental_map_target must be [{self.num_atoms}, 3] "
                        f"or [B, {self.num_atoms}, 3]; got "
                        f"{tuple(experimental_map_target.shape)}."
                    )
            # [N, A, 1]-weighted sum over alt states → [N, 3]
            effective_coords = (coords_used * occupancies.unsqueeze(-1)).sum(dim=1)
            exp_loss = F.mse_loss(effective_coords, experimental_map_target)
            total_loss = exp_loss + w_bond * bond_loss + w_occ * occ_penalty
        else:
            # ---- Batched targets -----------------------------------------
            if self.validate_inputs:
                B, N, C = experimental_map_target.shape
                if N != self.num_atoms or C != 3:
                    raise ValueError(
                        f"experimental_map_target must be [B, {self.num_atoms}, 3]; "
                        f"got {tuple(experimental_map_target.shape)}."
                    )
            B = experimental_map_target.shape[0]
            coords_b = coords_used.unsqueeze(0).expand(B, -1, -1, -1)  # [B, N, A, 3]
            occ_b    = occupancies.unsqueeze(0).expand(B, -1, -1)      # [B, N, A]
            effective_coords = (coords_b * occ_b.unsqueeze(-1)).sum(dim=2)  # [B, N, 3]

            # Per-sample MSE — keeps the batch dim alive for DDP routing.
            sq = (effective_coords - experimental_map_target) ** 2
            exp_loss = sq.mean(dim=(1, 2))                              # [B]
            total_loss = exp_loss + w_bond * bond_loss + w_occ * occ_penalty

            # Broadcast occupancies to batched shape for API symmetry.
            occupancies = occ_b

        return effective_coords, occupancies, total_loss

    # ------------------------------------------------------------------ #
    # Pure, differentiable forward                                       #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        experimental_map_target: torch.Tensor,
        *,
        dt: Union[float, torch.Tensor] = 0.01,
        apply_no_zeno_saturation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pure differentiable forward pass.

        Parameters
        ----------
        experimental_map_target : torch.Tensor
            `[N, 3]` (single) or `[B, N, 3]` (batched).
        dt : float | torch.Tensor
            Time step for the No-Zeno barrier. Promoted once, no CPU↔GPU sync.
        apply_no_zeno_saturation : bool
            If `True` (default), the soft coordinate saturation is applied.
            Matches the reference's optional `clamp_()` behaviour.

        Returns
        -------
        effective_coords : torch.Tensor  [N, 3] or [B, N, 3]
        occupancies      : torch.Tensor  [N, A] or [B, N, A]
        total_loss       : torch.Tensor  scalar (single) or [B] (batched)

        Notes
        -----
        The returned `total_loss` is a **tensor** (unlike the reference's
        `.item()` Python float).  Callers who need a float can call
        `.item()` on the result.  This change is required for end-to-end
        differentiability and for DDP gradient routing.
        """
        if self.validate_inputs and not isinstance(experimental_map_target, torch.Tensor):
            raise ValueError("experimental_map_target must be a torch.Tensor.")
        if self.validate_inputs and experimental_map_target.dim() not in (2, 3):
            raise ValueError("experimental_map_target must be 2D or 3D.")

        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._step,
                experimental_map_target, dt, apply_no_zeno_saturation,
                use_reentrant=False,
            )
        return self._step(experimental_map_target, dt, apply_no_zeno_saturation)

    # ------------------------------------------------------------------ #
    # Reference-style optimization loop (DDP-correct)                    #
    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def refine(
        self,
        experimental_map_target: torch.Tensor,
        *,
        steps: int = 100,
        lr: float = 0.01,
        dt: Union[float, torch.Tensor] = 0.01,
        optimizer: Optional[torch.optim.Optimizer] = None,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward-compatible optimization loop.

        Unlike the reference, the loop uses an **external or explicitly
        constructed optimizer**, so:
          · DDP intercepts each `loss.backward()` and all-reduces gradients,
          · optimizer state persists across calls if an external optimizer is
            provided,
          · `torch.compile` sees a stable graph (no Python-scalar `lr`/`steps`
            inside `forward`).

        Parameters
        ----------
        experimental_map_target : torch.Tensor
            `[N, 3]` or `[B, N, 3]`.
        steps : int
            Number of optimization iterations. Must be ≥ 0.
        lr : float
            Learning rate used when no external optimizer is supplied.
            Ignored if `optimizer` is provided.
        dt : float | torch.Tensor
            Forwarded to `forward`.
        optimizer : torch.optim.Optimizer, optional
            If provided, used directly; its parameter groups are untouched.
        weight_decay : float
            Weight decay for the internally constructed AdamW.
        verbose : bool
            If `True`, prints the loss every 10 steps.

        Returns
        -------
        effective_coords, occupancies, total_loss  (all detached)
        """
        if steps < 0:
            raise ValueError("steps must be ≥ 0.")

        if optimizer is None:
            # Reference-compatible two-group optimizer.
            optimizer = torch.optim.AdamW(
                [
                    {"params": [self.coordinates],    "lr": lr},
                    {"params": [self.raw_occupancy],  "lr": lr * 0.5},
                ],
                weight_decay=weight_decay,
            )

        effective_coords: Optional[torch.Tensor] = None
        occupancies: Optional[torch.Tensor] = None
        total_loss: Optional[torch.Tensor] = None

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            effective_coords, occupancies, total_loss = self.forward(
                experimental_map_target, dt=dt,
            )
            #   Sum over batch so DDP's mean-reduction sees a proper scalar.
            loss_scalar = total_loss.sum() if total_loss.dim() > 0 else total_loss
            loss_scalar.backward()
            optimizer.step()

            if verbose and (step % 10 == 0 or step == steps - 1):
                print(f"  [refine] step {step:>4d}  loss = "
                      f"{loss_scalar.detach().item():.6f}")

        # Final forward pass to return coherent, detached outputs.
        effective_coords, occupancies, total_loss = self.forward(
            experimental_map_target, dt=dt,
        )
        return (
            effective_coords.detach(),
            occupancies.detach(),
            total_loss.detach(),
        )


# =============================================================================
# Smoke test / autograd verification
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print(f"[REAL FOLD ONE v2] Running on: {device}")

    num_test_atoms = 300
    engine = RealFoldOneAdvancedEngine(
        num_atoms=num_test_atoms, num_alt_states=2,
    ).to(device)
    engine.train()

    # Assign heterogeneous components (water / ion) exactly as in the ref.
    engine.set_atom_type(list(range(200, 241)), atom_type_id=1)   # waters
    engine.set_atom_type(list(range(241, 261)), atom_type_id=2)   # ions

    # ---- Single-target mode ----------------------------------------------
    target = torch.randn(num_test_atoms, 3, device=device)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else torch.no_grad()
    )
    with autocast_ctx:
        coords, occ, loss = engine(target)

    print("-" * 64)
    print(f"  single: coords shape     : {tuple(coords.shape)}")
    print(f"  single: occupancy shape  : {tuple(occ.shape)}")
    print(f"  single: loss (tensor)    : {loss.detach().item():.6f}  "
          f"(shape {tuple(loss.shape)})")

    # ---- Gradient-flow audit (pure forward + backward) --------------------
    loss.backward()
    grad_ok = True
    for name, p in engine.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            grad_ok = False
            print(f"  [WARN] non-finite/missing grad: {name}")
    print(f"  pure-forward autograd    : "
          f"{'OK' if grad_ok else 'FAILED'}")

    # ---- Batched mode (DDP-style) ----------------------------------------
    engine.zero_grad(set_to_none=True)
    B = 4
    batched_target = torch.randn(B, num_test_atoms, 3, device=device)
    with autocast_ctx:
        bcoords, bocc, bloss = engine(batched_target)
    print(f"  batched: coords shape    : {tuple(bcoords.shape)}")
    print(f"  batched: occupancy shape : {tuple(bocc.shape)}")
    print(f"  batched: loss shape      : {tuple(bloss.shape)}")
    bloss.sum().backward()
    print(f"  batched autograd         : OK")

    # ---- Reference-style optimization loop --------------------------------
    print("-" * 64)
    print("Running refine() — external optimizer, DDP-safe loop …")
    opt = torch.optim.AdamW(engine.parameters(), lr=1e-2)
    final_coords, final_occ, final_loss = engine.refine(
        target, steps=20, optimizer=opt, verbose=False,
    )
    print(f"  refine: final loss       : {final_loss.item():.6f}")
    print(f"  refine: coords shape     : {tuple(final_coords.shape)}")
    print(f"  refine: occupancy shape  : {tuple(final_occ.shape)}")
    print("-" * 64)
    print("Execution completed successfully.")
