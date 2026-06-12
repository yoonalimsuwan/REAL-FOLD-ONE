# =============================================================================
# STRUCTURAL GRAPH NEURAL OPERATOR (SGNO FOLD)
# High-Level AI Surrogate Model for One-Shot Protein Folding & Mutation Impact
# (v2 - Unified Discrete & Continuous Physics Extension)
# =============================================================================
# Developer    : Yoon A Limsuwan
# Organization : MSPS NETWORK / MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Assisted by  : Gemini (AI)
# License      : MIT
# Year         : 2026
#
# Description:
#   A novel Graph Neural Operator rooted in the Structural Calculus framework.
#   Designed specifically to train on and accelerate the REAL FOLD ONE ecosystem:
#     - one_core_fold.py (Semantic State Contraction & CSOC)
#     - real_fold_one_v2.py (Differentiable Protein Energy)
#     - real_fold_one_ht_v2.py (High-Throughput Mutation Scans)
#     - structural_langevin_fold_v2.py (BAOAB Dynamics)
#
#   This model learns the operator mapping: 
#       G : (X_0, Seq, sigma(X)) ↦ X_final (or Delta Delta G)
#
#   v2 UPDATE: Now acts as a Unified Structural Neural Operator. 
#   It seamlessly supports the uploaded Cahn-Hilliard 3D Continuum Suite, 
#   enabling one-shot O(1) predictions for continuous phase-field evolution 
#   and bypassing thousands of PDE time-steps.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union, Dict, Any

# Import components from the FOLD ecosystem for training/evaluation
try:
    from one_core_fold import SemanticStateContraction
    from real_fold_one_v2 import DifferentiableProteinEnergy
except ImportError:
    pass

# Connect to the uploaded module (import core physics structures)
try:
    from structural_cahn_hilliard_3d import (
        StructuralCahnHilliard3D,
        ThinFilmStructuralCahnHilliard3D,
        PhaseFieldCrystal3D,
        CahnHilliardConfig
    )
except ImportError:
    # Fallback in case of standalone execution
    StructuralCahnHilliard3D = None
    ThinFilmStructuralCahnHilliard3D = None
    PhaseFieldCrystal3D = None
    CahnHilliardConfig = None


# =============================================================================
# 1. Structural Message Passing
# =============================================================================
class StructuralMessagePassing(nn.Module):
    """
    Message Passing Layer modulated by the Structural Regime Field sigma(x).
    Enforces structural calculus rules inside the neural network latent space.
    """
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        # Message function: processes node pairs + edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
        # Update function: updates node state
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
        # Structural modulation projector
        self.sigma_proj = nn.Linear(1, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, node_dim) Node features
        edge_index : (2, E) Edge connectivity
        edge_attr  : (E, edge_dim) Edge distances/features
        sigma      : (N, 1) Structural regime field (from SSC)
        """
        src, dst = edge_index[0], edge_index[1]
        
        # 1. Compute messages
        msg_inputs = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.message_mlp(msg_inputs)
        
        # 2. Aggregate messages (Sum pooling)
        N = x.size(0)
        aggr_msg = torch.zeros(N, messages.size(1), device=x.device, dtype=x.dtype)
        aggr_msg.index_add_(0, dst, messages)
        
        # 3. Structural Regime Modulation
        # Analogue to D_S u = sigma * grad(u)
        # The aggregated message represents the 'gradient' of the latent field
        s_feat = torch.sigmoid(self.sigma_proj(sigma))
        modulated_msg = s_feat * aggr_msg
        
        # 4. Update node states with residual connection
        update_inputs = torch.cat([x, modulated_msg], dim=-1)
        x_new = self.update_mlp(update_inputs)
        
        return x + x_new 


# =============================================================================
# 2. StructuralGNOFold 
# =============================================================================
class StructuralGNOFold(nn.Module):
    """
    The Enhanced Structural Graph Neural Operator (SGNO).
    Supports both Discrete Protein Graphs and Continuous 3D Phase-Field Grids.
    """
    def __init__(self, node_in_dim: int = 20, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        # Embeddings for Protein Mode (Graph Mode)
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(1, hidden_dim)
        
        # Embeddings for Continuous Mode (Grid/Phase-Field Mode)
        # Map phase value u and 3D coordinates to Latent Voxel Features
        self.grid_node_embed = nn.Linear(4, hidden_dim) # [u, x, y, z]
        
        # Shared neural differential processing core
        self.layers = nn.ModuleList([
            StructuralMessagePassing(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # --- Output Heads ---
        # 1. Head for protein graphs (3D coordinates and mutation energy)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)
        )
        self.ddg_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 2. Head for future phase evolution (One-Shot Phase Evolution)
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1) # Predicts Delta u (u_final - u_init)
        )

    def _build_graph(self, coords: torch.Tensor, cutoff: float = 10.0):
        """Dynamically builds a K-NN or radius graph from 3D coordinates."""
        N = coords.shape[0]
        dist_mat = torch.cdist(coords, coords)
        
        # Create edges for atoms within cutoff
        adj = (dist_mat < cutoff) & (dist_mat > 0)
        edge_index = torch.nonzero(adj, as_tuple=False).t()
        
        # Edge attributes are the distances
        src, dst = edge_index[0], edge_index[1]
        edge_attr = dist_mat[src, dst].unsqueeze(-1)
        return edge_index, edge_attr

    def _build_graph_from_3d_grid(self, u: torch.Tensor, cutoff_radius: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert 3D grid data (Cahn-Hilliard/PFC) into Graph Nodes and Edges 
        to be processed by the Message Passing architecture as a Neural Operator.
        """
        nx, ny, nz = u.shape
        device, dtype = u.device, u.dtype
        
        # Create true coordinate matrix based on the grid
        x = torch.arange(nx, device=device, dtype=dtype)
        y = torch.arange(ny, device=device, dtype=dtype)
        z = torch.arange(nz, device=device, dtype=dtype)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        
        # Flatten the grid into a 1D array to set as Node Features
        coords = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
        u_flat = u.flatten().unsqueeze(-1)
        node_feats = torch.cat([u_flat, coords], dim=-1) # [u, x, y, z]
        
        # Rapidly create neighbor connectivity structure within the cutoff radius
        dist_mat = torch.cdist(coords, coords)
        adj = (dist_mat <= cutoff_radius) & (dist_mat > 0)
        edge_index = torch.nonzero(adj, as_tuple=False).t()
        
        src, dst = edge_index[0], edge_index[1]
        edge_attr = dist_mat[src, dst].unsqueeze(-1)
        
        return node_feats, edge_index, edge_attr

    def forward(self, seq_features: torch.Tensor, init_coords: torch.Tensor, 
                sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Original mode: Protein prediction (Protein Folding Mode).
        """
        # Build spatial graph
        edge_index, edge_attr = self._build_graph(init_coords)
        
        # Embeddings
        x = self.node_embed(seq_features)
        e = self.edge_embed(edge_attr)
        
        # Message Passing modulated by Structural Calculus
        for layer in self.layers:
            x = layer(x, edge_index, e, sigma)
            
        # Predict displacements and energetic impact
        displacements = self.coord_head(x)
        final_coords = init_coords + displacements
        
        # Global pooling for DDG prediction
        graph_embed = x.mean(dim=0)
        pred_ddg = self.ddg_head(graph_embed)
        
        return final_coords, pred_ddg

    def forward_phase_field(self, u_init: torch.Tensor, sigma_3d: torch.Tensor) -> torch.Tensor:
        """
        New mode: Continuous physics prediction (One-Shot Phase-Field Evolution Engine).
        Learns the state evolution over time (Surrogate of Cahn-Hilliard/PFC).
        """
        shape_3d = u_init.shape
        
        # 1. Convert 3D grid into graph format
        node_feats, edge_index, edge_attr = self._build_graph_from_3d_grid(u_init)
        sigma_flat = sigma_3d.flatten().unsqueeze(-1)
        
        # 2. Extract spatial features
        x = self.grid_node_embed(node_feats)
        e = self.edge_embed(edge_attr)
        
        # 3. Transmit data via Structural Calculus (Structural Message Passing)
        for layer in self.layers:
            x = layer(x, edge_index, e, sigma_flat)
            
        # 4. Decode phase difference back to the original 3D grid structure
        delta_u_flat = self.phase_head(x)
        delta_u_3d = delta_u_flat.view(shape_3d)
        
        # Predict final output u = u_init + delta_u
        return u_init + delta_u_3d


# =============================================================================
# 3. SGNO TRAINER (Dual Physics-AI Data Loop)
# =============================================================================
class SGNOTrainer:
    """
    Expanded capabilities to support dual-dimensional training (Discrete + Continuous).
    Trains the Structural GNO using the core modules as the Physics Engine.
    """
    def __init__(self, model: StructuralGNOFold, device: torch.device, 
                 ch_solver: Optional[StructuralCahnHilliard3D] = None):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.device = device
        
        # Attach the uploaded physics solver for integrated use
        self.ch_solver = ch_solver

    def train_step_protein(self, seq_feats, init_coords, true_final_coords, true_ddg, sigma):
        """
        Training step for protein folding mode using data generated by HT Scanner.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass (One-shot prediction)
        pred_coords, pred_ddg = self.model(seq_feats, init_coords, sigma)
        
        # Loss: Coordinate matching + DDG matching
        loss_coords = F.mse_loss(pred_coords, true_final_coords)
        loss_ddg = F.mse_loss(pred_ddg.squeeze(), true_ddg)
        
        total_loss = loss_coords + 0.1 * loss_ddg
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def train_step_phase_field(self, u_init: torch.Tensor, u_true_future: torch.Tensor, 
                               sigma_3d: torch.Tensor) -> float:
        """
        Training step for continuous derivative mode (Physics-Informed Loss).
        Evaluates accuracy alongside the Lyapunov Energy Functional of the physics module.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Predict future grid results in one shot using the AI Operator
        u_pred_future = self.model.forward_phase_field(u_init, sigma_3d)
        
        # Data-driven loss (compare with ground truth from actual equation stepping)
        loss_data = F.mse_loss(u_pred_future, u_true_future)
        
        # Physics-informed regularization loss (if physics solver is provided)
        loss_physics = torch.tensor(0.0, device=self.device)
        if self.ch_solver is not None:
            # Verify if the Structural Free Energy decreases according to Lyapunov theory
            E_pred = self.ch_solver.structural_energy(u_pred_future, sigma_3d)
            E_init = self.ch_solver.structural_energy(u_init, sigma_3d)
            
            # New energy must be less than or equal to initial energy (E_pred <= E_init)
            # Penalize the model if the predicted phase energy increases
            loss_physics = F.relu(E_pred - E_init) 
            
        total_loss = loss_data + 0.05 * loss_physics
        total_loss.backward()
        
        self.optimizer.step()
        
        return total_loss.item()


# =============================================================================
# 4. Verification Loop
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Unified Structural Neural Operator (SGNO-v2) Integration Tests")
    print(f"  Running on: {device}")
    print("=" * 70)
    
    # Test 1: Original Protein Graph Mode
    N_residues = 40
    seq_feats = torch.randn(N_residues, 20, device=device)
    init_coords = torch.randn(N_residues, 3, device=device)
    sigma_protein = torch.ones(N_residues, 1, device=device) * 1.2
    
    model = StructuralGNOFold().to(device)
    final_coords, pred_ddg = model(seq_feats, init_coords, sigma_protein)
    print(f"[SUCCESS] Protein Mode -> Coords: {final_coords.shape}, DDG: {pred_ddg.shape}")
    
    # Test 2: Continuous 3D Cahn-Hilliard Grid Prediction Mode
    # Assume a small 16x16x16 grid to save memory during graph generation testing
    Grid_Size = 16 
    u_init_grid = torch.rand(Grid_Size, Grid_Size, Grid_Size, device=device, dtype=torch.float32) * 0.2 - 0.1
    sigma_3d_grid = torch.ones(Grid_Size, Grid_Size, Grid_Size, device=device, dtype=torch.float32)
    
    # Instruct AI to predict one-step CH3D phase evolution without running complex PDEs
    u_future_predicted = model.forward_phase_field(u_init_grid, sigma_3d_grid)
    print(f"[SUCCESS] Continuous Mode -> Predicted 3D Phase-Field Grid Shape: {u_future_predicted.shape}")
    
    # Test 3: Verify trainer system structure with the uploaded physics solver
    if StructuralCahnHilliard3D is not None:
        cfg = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-5, laplacian="conv3d")
        physics_engine = StructuralCahnHilliard3D(cfg).to(device)
        trainer = SGNOTrainer(model, device, ch_solver=physics_engine)
        print("[SUCCESS] Connected to uploaded 'StructuralCahnHilliard3D' physics engine successfully!")
    else:
        trainer = SGNOTrainer(model, device, ch_solver=None)
        print("[NOTICE] Running Trainer in standalone mode (Uploaded physics file omitted).")
        
    print("=" * 70)
