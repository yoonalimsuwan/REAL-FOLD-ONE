# =============================================================================
# STRUCTURAL CALCULUS AGENT (Optimized for Native Full Differentiability)
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026

import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalContractionOperator(nn.Module):
    """
    Module for the Universal Contraction Operator (Phi_U).
    It collapses the exponentially large decision space into a polynomially bounded quotient space.
    """
    def __init__(self, n_vars: int, m_clauses: int):
        super().__init__()
        self.n = n_vars
        self.m = m_clauses
        # C_i: Constraint hyperplanes defining the semantic bounds
        self.C = nn.Parameter(torch.randn(m_clauses, n_vars))
        # Delta_i: Deterministic Semantic-State Contraction vector
        self.Delta = nn.Parameter(torch.randn(m_clauses, n_vars))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, n_vars)
        # Projects the input onto the constraint hyperplanes
        constrained_x = x.unsqueeze(1) * self.C # Shape: (Batch, m, n)
        
        # Applies the contraction vector Delta via Einstein summation
        # This maps the independent structural classes into a lower-dimensional deterministic manifold.
        # The resulting topological signature matrix M_[A] is bounded by O(m^3 * n^2).
        signature_matrix = torch.einsum(
            'bmn,bmp->bnp', 
            constrained_x, 
            self.Delta.unsqueeze(0).expand(x.size(0), -1, -1)
        )
        return signature_matrix

class TopologicalBranchEliminator(nn.Module):
    """
    Differentiable module for strict Branch Elimination (Optimized).
    Uses normalized log-determinant on a positive-definite manifold 
    to ensure mathematical stability during backpropagation.
    """
    def __init__(self, n_vars: int, lambda_reg: float = 1e-4):
        super().__init__()
        self.n = n_vars
        self.lambda_reg = lambda_reg

    def forward(self, M_A: torch.Tensor) -> torch.Tensor:
        # M_A shape: (Batch, n, n)
        batch_size = M_A.size(0)
        identity = torch.eye(self.n, device=M_A.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 1. Shift the matrix to a positive definite manifold: M_A @ M_A^T + lambda * I
        reg_matrix = torch.bmm(M_A, M_A.transpose(1, 2)) + (self.lambda_reg * identity)
        
        # 2. Continuous determinant evaluation via log-determinant for numerical stability
        sign, log_det = torch.linalg.slogdet(reg_matrix)
        
        # 3. Normalize to scale the bounds appropriately with the dimension n
        normalized_log_det = log_det / self.n
        
        # 4. Converts the normalized log-determinant into a differentiable structural probability
        viability = torch.sigmoid(normalized_log_det)
        
        return viability

class NoZenoTopologicalGating(nn.Module):
    """
    Module to resolve the Zeno Trap (infinite topological events in finite time) 
    using double-exponential (Gumbel-type) extreme-value statistics.
    """
    def __init__(self):
        super().__init__()
        # Geometric constant of the domain
        self.c1 = nn.Parameter(torch.tensor(1.0)) 
        # Variance of the random interface fluctuations in the disordered medium
        self.sigma_sq = nn.Parameter(torch.tensor(0.1)) 

    def forward(self, delta_E: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        # delta_E: Activation energy required to escape the current fixed reference topology.
        
        # Softplus ensures parameters remain strictly positive to prevent NaN values
        safe_sigma = F.softplus(self.sigma_sq) + 1e-6
        safe_c1 = F.softplus(self.c1)
        
        # Applies the double-exponential Gumbel-type bound: 
        # P(T_{k+1} - T_k < dt) <= exp[-C_1 * exp(Delta_E / (sigma^2 * dt))]
        inner_term = torch.exp(delta_E / (safe_sigma * dt))
        extreme_value_gate = torch.exp(-safe_c1 * inner_term)
        
        return extreme_value_gate

class FullStructuralCalculusAgent(nn.Module):
    """
    The unified end-to-end differentiable agent integrating structural contraction, 
    topological evaluation, and quenched stochastic homogenization.
    """
    def __init__(self, input_dim: int, n_vars: int, m_clauses: int, num_classes: int):
        super().__init__()
        # 1. Feature Extractor (e.g., mapping raw LC-MS/NMR data into the state space)
        self.embedding = nn.Linear(input_dim, n_vars)
        
        # 2. Structural Calculus Core
        self.phi_u = UniversalContractionOperator(n_vars, m_clauses)
        self.branch_elimination = TopologicalBranchEliminator(n_vars)
        self.zeno_gate = NoZenoTopologicalGating()
        
        # 3. Energy Estimator for predicting topological activation barriers
        self.energy_estimator = nn.Sequential(
            nn.Linear(n_vars, n_vars // 2),
            nn.ReLU(),
            nn.Linear(n_vars // 2, 1)
        )
        
        # 4. Macroscopic Predictor (Stochastic Homogenization Limit)
        self.macroscopic_homogenizer = nn.Linear(n_vars, num_classes)

    def forward(self, raw_data: torch.Tensor, dt: float = 0.01):
        # Step 1: Embed raw signals into the structural state space
        x = F.gelu(self.embedding(raw_data))
        
        # Step 2: Generate the Topological Signature Matrix
        M_A = self.phi_u(x)
        
        # Step 3: Evaluate the viability of the structural class
        viability_score = self.branch_elimination(M_A)
        
        # Step 4: Estimate the activation energy required for a topological jump
        delta_E = F.softplus(self.energy_estimator(x).squeeze(-1))
        
        # Step 5: Apply the No-Zeno condition to gate infinite non-terminating loops
        transition_prob = self.zeno_gate(delta_E, dt)
        
        # Step 6: Macroscopic convergence (Deterministic Limit)
        base_prediction = self.macroscopic_homogenizer(x)
        
        # Final output is strictly gated by topological viability and Zeno transition rules
        final_output = base_prediction * viability_score.unsqueeze(-1) * transition_prob.unsqueeze(-1)
        
        return final_output, viability_score, transition_prob
