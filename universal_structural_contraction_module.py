# =============================================================================
# Universal Structural Contraction (USC) Module : New Class of Neural Network Layer
# =============================================================================
#
# Developer  : PAI AND Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788
# GitHub     : https://github.com/yoonalimsuwan
# Email      : msps4u@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelNoZenoActivation(nn.Module):
    """
    Implements the double-exponential extreme-value activation to enforce 
    the No-Zeno condition and bound topological transitions.
    """
    def __init__(self, sigma: float = 1.0, c1: float = 1.0):
        super().__init__()
        # Learnable variance and geometric constants for the disordered medium
        self.sigma_sq = nn.Parameter(torch.tensor(sigma ** 2))
        self.c1 = nn.Parameter(torch.tensor(c1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # P(dT < dt) <= exp[-C1 * exp(Delta E_min / (sigma^2 dt))]
        # We translate this into a differentiable gating mechanism.
        barrier_scaled = x / (self.sigma_sq.abs() + 1e-6)
        return torch.exp(-self.c1.abs() * torch.exp(-barrier_scaled))

class UniversalContractionModule(nn.Module):
    """
    One-Shot Structural Calculus Layer.
    Complexity: O(N * D) instead of O(N^2 * D) of standard attention.
    """
    def __init__(self, d_model: int, num_structural_classes: int = 64):
        super().__init__()
        self.d_model = d_model
        self.k = num_structural_classes # Simulates the polynomial bound P(n, m)
        
        # Semantic-State Contraction Vectors (Delta_i) and Constraint Hyperplanes (C_i)
        self.W_constraint = nn.Linear(d_model, self.k)
        self.W_contraction = nn.Linear(d_model, self.k)
        
        # Outward projection after quotient mapping
        self.W_out = nn.Linear(self.k, d_model)
        
        # No-Zeno gating mechanism
        self.no_zeno_gate = GumbelNoZenoActivation()
        
        # Layer Normalization for Asymptotically Mean Stationary (AMS) stabilization
        self.ams_norm = nn.LayerNorm(self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (Batch, Sequence_Length, D_model)
        """
        B, L, D = x.shape
        
        # 1. Generate constraint hyperplanes (C_i)
        # Shape: (B, L, K)
        constraints = self.W_constraint(x)
        
        # 2. Generate Semantic-State Contraction vectors (\Delta_i)
        # Shape: (B, L, K)
        contractions = self.W_contraction(x)
        
        # 3. Topological Signature Extraction (Branch Elimination)
        # We form the signature matrix without exponential enumeration by taking
        # the Hadamard product, representing the intersection of semantic states.
        # Shape: (B, L, K)
        structural_state = constraints * contractions
        
        # Apply No-Zeno condition to filter out chaotic micro-state fluctuations
        # Shape: (B, L, K)
        topological_active_state = self.no_zeno_gate(structural_state)
        
        # 4. Change-Point Induced Homogenization (Kakutani Averaging)
        # Collapse the sequence length deterministically along the structural class dimension.
        # This acts as the quotient mapping \Omega / \equiv_\Phi.
        # Shape: (B, K)
        quotient_space = topological_active_state.sum(dim=1) 
        
        # Stabilize via AMS principles (mean-reverting stabilization)
        homogenized_state = self.ams_norm(quotient_space)
        
        # 5. Broadcast back to continuous macroscopic geometry
        # Shape: (B, L, D_model)
        # We expand the bounded structural classes back into the original dimensional space.
        macroscopic_geometry = self.W_out(homogenized_state).unsqueeze(1).expand(B, L, D)
        
        # Residual connection combining microscopic flow with macroscopic geometry
        return x + macroscopic_geometry

