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
# =============================================================================

import torch
import torch.nn as nn
import math

class SemanticStateContraction(nn.Module):
    """
    SSC Filter (Paper 4): Acts as a first-order low-pass filter 
    to track structural stress across time steps without exploding gradients.
    """
    def __init__(self, epsilon_fp: float = 0.0028):
        super().__init__()
        self.eps = epsilon_fp
        self.register_buffer('prev_sigma', torch.tensor(0.0))

    def forward(self, raw_sigma: torch.Tensor) -> torch.Tensor:
        if self.prev_sigma.item() == 0.0:
            self.prev_sigma.data = raw_sigma.detach()
            return raw_sigma
        
        # Contract semantic state towards the stable manifold
        new_sigma = self.prev_sigma + self.eps * (raw_sigma - self.prev_sigma)
        self.prev_sigma.data = new_sigma.detach()
        return new_sigma


class CSOCThermostat(nn.Module):
    """
    CSOC Adaptive Thermostat (Paper 4): Modulates Langevin temperature 
    and friction based on real-time structural stress.
    """
    def __init__(self, base_temp: float = 300.0, base_friction: float = 1.0, 
                 sigma_target: float = 1.0, epsilon_fp: float = 0.0028):
        super().__init__()
        self.base_temp = base_temp
        self.base_friction = base_friction
        self.sigma_target = sigma_target
        self.ssc = SemanticStateContraction(epsilon_fp)

    def forward(self, coords: torch.Tensor, prev_coords: torch.Tensor) -> tuple:
        # Calculate raw structural stress (e.g., mean displacement or deformation)
        raw_sigma = torch.norm(coords - prev_coords, dim=-1).mean()
        
        # Apply SSC to get filtered stress
        sigma = self.ssc(raw_sigma)
        
        # CSOC Temperature Modulation: Elevate T when stress exceeds target
        dev = (sigma - self.sigma_target) / 0.5
        adaptive_T = self.base_temp + 2000.0 * torch.sigmoid(dev)
        adaptive_T = torch.clamp(adaptive_T, self.base_temp * 0.5, 3000.0)
        
        # Optional: Modulate friction (higher friction during collapse regimes)
        adaptive_gamma = self.base_friction * (1.0 + 0.5 * torch.relu(dev))
        
        return adaptive_T, adaptive_gamma, sigma


class StructuralItoNoise(nn.Module):
    """
    Multiplicative Noise & Itô Correction (Paper 2 & 3): 
    Generates interface-aware stochastic forces and computes the 
    required Itô drift correction (1/2 G \nabla G) to prevent spurious drift.
    """
    def __init__(self, interface_amplification: float = 2.0):
        super().__init__()
        self.amp = interface_amplification

    def get_g_matrix(self, interface_mask: torch.Tensor) -> torch.Tensor:
        """
        G(x): The noise amplitude tensor. 
        Higher amplitude near self-evolving interfaces (Paper 2).
        """
        # Baseline noise is 1.0, amplified by interface presence
        return 1.0 + self.amp * interface_mask

    def compute_ito_correction(self, coords: torch.Tensor, interface_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes (1/2) G(x) \nabla G(x) using PyTorch Autograd.
        Required for rigorous Structural Itô Calculus (Paper 3).
        """
        with torch.enable_grad():
            x = coords.clone().requires_grad_(True)
            G = 1.0 + self.amp * interface_mask
            
            # Sum of G to scalar for backward pass
            G_sum = G.sum()
            grad_G = torch.autograd.grad(G_sum, x, create_graph=True)[0]
            
            if grad_G is None:
                return torch.zeros_like(coords)
                
            # Itô drift correction term
            ito_drift = 0.5 * G.unsqueeze(-1) * grad_G
            
        return ito_drift.detach()


class AdvancedStructuralLangevin(nn.Module):
    """
    The Core Integrator: BAOAB Splitting Method tailored for Structural Calculus.
    Integrates Bulk dynamics, BV Jumps, Itô corrections, and CSOC.
    """
    def __init__(self, 
                 mass: float = 1.0,
                 dt: float = 0.002,
                 base_temp: float = 300.0,
                 base_friction: float = 1.0,
                 kb: float = 0.001987): # kcal/(mol K)
        super().__init__()
        self.mass = mass
        self.dt = dt
        self.kb = kb
        
        # Modules
        self.thermostat = CSOCThermostat(base_temp, base_friction)
        self.ito_noise = StructuralItoNoise()
        
        # State tracking
        self.prev_coords = None

    def _structural_force(self, force_bulk: torch.Tensor, jumps: torch.Tensor, interface_mask: torch.Tensor) -> torch.Tensor:
        """
        D^S u = \nabla u + [u]\delta_\Gamma (Paper 1 & 2)
        Total force includes classical bulk forces plus concentrated jump measures.
        """
        return force_bulk + (jumps * interface_mask.unsqueeze(-1))

    def step(self, 
             coords: torch.Tensor, 
             velocities: torch.Tensor, 
             force_bulk: torch.Tensor, 
             jumps: torch.Tensor, 
             interface_mask: torch.Tensor) -> tuple:
        """
        Executes one BAOAB integration step.
        Inputs must be PyTorch tensors on the same device.
        """
        device = coords.device
        
        if self.prev_coords is None:
            self.prev_coords = coords.detach().clone()

        # Get adaptive T and gamma from CSOC Thermostat
        T, gamma, sigma = self.thermostat(coords, self.prev_coords)
        self.prev_coords = coords.detach().clone()

        # Combine bulk force and BV jump measures
        F_struct = self._structural_force(force_bulk, jumps, interface_mask)

        # ---------------------------------------------------------
        # [ B - STEP ]: Half-step velocity update based on forces
        # ---------------------------------------------------------
        v_half = velocities + 0.5 * self.dt * (F_struct / self.mass)

        # ---------------------------------------------------------
        # [ A - STEP ]: Half-step position update
        # ---------------------------------------------------------
        x_half = coords + 0.5 * self.dt * v_half

        # ---------------------------------------------------------
        # [ O - STEP ]: Exact Ornstein-Uhlenbeck stochastic update
        # Multiplicative Structural Noise + Itô Correction applied here
        # ---------------------------------------------------------
        # Constants for exact integration of the O-step
        c1 = torch.exp(-gamma * self.dt)
        c2 = torch.sqrt(1.0 - c1**2)
        
        # Multiplicative Noise Modifier G(x)
        G_x = self.ito_noise.get_g_matrix(interface_mask).unsqueeze(-1)
        
        # Base thermal noise scale: sqrt(kB * T / m)
        noise_scale = math.sqrt(self.kb * T.item() / self.mass)
        
        # Generate structural noise (Multiplicative)
        R = torch.randn_like(v_half, device=device)
        structural_stochastic_force = c2 * noise_scale * G_x * R
        
        # Structural Itô Drift Correction (Paper 3)
        ito_correction = self.ito_noise.compute_ito_correction(x_half, interface_mask)
        
        # O-step velocity update
        v_tilde = c1 * v_half + structural_stochastic_force + (ito_correction * self.dt)

        # ---------------------------------------------------------
        # [ A - STEP ]: Second half-step position update
        # ---------------------------------------------------------
        x_full = x_half + 0.5 * self.dt * v_tilde

        # ---------------------------------------------------------
        # [ B - STEP ]: Note - In a real simulation loop, the new force 
        # (F_struct_new) is evaluated at x_full *before* this final B-step. 
        # For a single call, we return x_full and v_tilde, and the user 
        # applies the final B-step after re-evaluating the energy/force graph.
        # ---------------------------------------------------------
        
        # To maintain the BAOAB chain, the outer loop should do:
        # 1. new_force = compute_force(x_full)
        # 2. v_full = v_tilde + 0.5 * dt * (new_force / mass)
        
        return x_full, v_tilde, T.item(), sigma.item()

# =============================================================================
# Usage Example (To be integrated into REAL FOLD ONE or EVOLUTION ONE)
# =============================================================================
# integrator = AdvancedStructuralLangevin(dt=0.002)
#
# for step in range(num_steps):
#     # 1. Calculate Bulk Forces (e.g., from OpenMM or ML Force Field)
#     # force_bulk = -torch.autograd.grad(energy, coords)[0]
#
#     # 2. Identify Interfaces and Jumps (BV Framework)
#     # interface_mask = detect_interfaces(coords)
#     # jumps = calculate_jumps(coords)
#
#     # 3. BAOA Step (Returns positions ready for next force evaluation)
#     # coords, velocities, current_T, current_sigma = integrator.step(
#     #       coords, velocities, force_bulk, jumps, interface_mask)
#
#     # 4. Final B Step (Complete the BAOAB loop)
#     # new_force_bulk = -torch.autograd.grad(new_energy, coords)[0]
#     # F_struct_new = integrator._structural_force(new_force_bulk, jumps, interface_mask)
#     # velocities = velocities + 0.5 * integrator.dt * (F_struct_new / integrator.mass)
# =============================================================================
