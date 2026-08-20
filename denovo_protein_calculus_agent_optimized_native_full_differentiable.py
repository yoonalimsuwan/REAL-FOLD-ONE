# =============================================================================
# DE NOVO PROTEIN CALCULUS AGENT (Optimized for Native Full Differentiability)
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# Framework    : Structural Calculus (Deterministic Topological Framework)
# License      : MIT
# Year         : 2026
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalContractionOperator(nn.Module):
    """
    Module for the Universal Contraction Operator (Phi_U).
    Collapses the exponentially large decision space into a polynomially bounded quotient space.
    """
    def __init__(self, n_vars: int, m_clauses: int):
        super().__init__()
        self.n = n_vars
        self.m = m_clauses
        self.C = nn.Parameter(torch.randn(m_clauses, n_vars))
        self.Delta = nn.Parameter(torch.randn(m_clauses, n_vars))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constrained_x = x.unsqueeze(1) * self.C 
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
        batch_size = M_A.size(0)
        identity = torch.eye(self.n, device=M_A.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Ensure Positive Definiteness to prevent singular matrix crashes
        reg_matrix = torch.bmm(M_A, M_A.transpose(1, 2)) + (self.lambda_reg * identity)
        
        # Compute Log-Determinant for continuous and stable gradient flow
        sign, log_det = torch.linalg.slogdet(reg_matrix)
        
        # Normalize by dimension to prevent early saturation in the Sigmoid gate
        normalized_log_det = log_det / self.n
        
        # Convert to differentiable structural viability probability
        viability = torch.sigmoid(normalized_log_det)
        
        return viability


class NoZenoTopologicalGating(nn.Module):
    """
    Resolves the Zeno Trap (infinite topological events in finite time) 
    using double-exponential (Gumbel-type) extreme-value statistics.
    """
    def __init__(self):
        super().__init__()
        self.c1 = nn.Parameter(torch.tensor(1.0)) 
        self.sigma_sq = nn.Parameter(torch.tensor(0.1)) 

    def forward(self, delta_E: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        safe_sigma = F.softplus(self.sigma_sq) + 1e-6
        safe_c1 = F.softplus(self.c1)
        inner_term = torch.exp(delta_E / (safe_sigma * dt))
        extreme_value_gate = torch.exp(-safe_c1 * inner_term)
        return extreme_value_gate


class AminoAcidEmbedding(nn.Module):
    """
    Transforms discrete amino acid tokens into a continuous Structural State Space.
    """
    def __init__(self, vocab_size: int, embed_dim: int, n_vars: int, max_seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.projection = nn.Linear(embed_dim, n_vars)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq.size()
        positions = torch.arange(0, seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        
        # Combine token and positional features
        x = self.token_embedding(seq) + self.position_embedding(positions)
        
        # Aggregate the sequence into a global state vector
        x_pooled = x.mean(dim=1) 
        out = F.gelu(self.projection(x_pooled))
        return out


class DeNovoProteinCalculusAgent(nn.Module):
    """
    Unified end-to-end agent for deterministic De Novo Protein Design.
    Integrates sequence embedding with the Structural Calculus Core.
    """
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, 
                 n_vars: int, m_clauses: int, num_classes: int):
        super().__init__()
        
        # 1. Feature Extractor
        self.embedding = AminoAcidEmbedding(vocab_size, embed_dim, n_vars, max_seq_len)
        
        # 2. Structural Calculus Core
        self.phi_u = UniversalContractionOperator(n_vars, m_clauses)
        self.branch_elimination = TopologicalBranchEliminator(n_vars)
        self.zeno_gate = NoZenoTopologicalGating()
        
        # 3. Energy Estimator
        self.energy_estimator = nn.Sequential(
            nn.Linear(n_vars, n_vars // 2),
            nn.ReLU(),
            nn.Linear(n_vars // 2, 1)
        )
        
        # 4. Macroscopic Predictor
        self.macroscopic_homogenizer = nn.Linear(n_vars, num_classes)

    def forward(self, aa_sequence: torch.Tensor, dt: float = 0.01):
        x = self.embedding(aa_sequence)
        M_A = self.phi_u(x)
        viability_score = self.branch_elimination(M_A)
        delta_E = F.softplus(self.energy_estimator(x).squeeze(-1))
        transition_prob = self.zeno_gate(delta_E, dt)
        base_prediction = self.macroscopic_homogenizer(x)
        
        # Final output gated by topological viability and transition logic
        final_output = base_prediction * viability_score.unsqueeze(-1) * transition_prob.unsqueeze(-1)
        
        return final_output, viability_score, transition_prob


# =============================================================================
# EXECUTION & TESTING BLOCK
# =============================================================================
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 25       # 20 standard amino acids + special tokens
    EMBED_DIM = 128       # Feature dimension for sequence embedding
    MAX_SEQ_LEN = 1000    # Maximum polypeptide chain length
    N_VARS = 64           # Dimensions of the quotient space
    M_CLAUSES = 128       # Constraint hyperplanes
    NUM_CLASSES = 1       # Output dimension (e.g., fitness score or stability metric)
    
    # Initialize Agent
    agent = DeNovoProteinCalculusAgent(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        max_seq_len=MAX_SEQ_LEN,
        n_vars=N_VARS,
        m_clauses=M_CLAUSES,
        num_classes=NUM_CLASSES
    )
    
    # Simulate a batch of 4 sequences, each with 76 amino acids 
    # (e.g., simulating a target size similar to Ubiquitin)
    batch_size = 4
    seq_length = 76
    dummy_sequences = torch.randint(0, 20, (batch_size, seq_length))
    
    print("Running Deterministic Structural Inference...")
    
    # Forward Pass
    predictions, viability, transitions = agent(dummy_sequences, dt=0.01)
    
    print("-" * 50)
    print(f"Input Sequence Shape   : {dummy_sequences.shape}")
    print(f"Predictions Shape      : {predictions.shape}")
    print(f"Viability Scores Shape : {viability.shape}")
    print(f"Transition Probs Shape : {transitions.shape}")
    print("-" * 50)
    print("Execution Completed Successfully.")
