import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class NMRRestraintSet(nn.Module):
    """
    High-performance, memory-optimized module for handling NMR experimental restraints 
    (NOE distance bounds and Dihedral angles) and driving structural RMSD minimization 
    with O(N) or O(N log N) computational complexity.
    """
    def __init__(self, 
                 noe_pairs: torch.Tensor, 
                 noe_bounds: torch.Tensor, 
                 dihedral_indices: Optional[torch.Tensor] = None,
                 dihedral_targets: Optional[torch.Tensor] = None,
                 force_constant: float = 50.0):
        super().__init__()
        # Register tensors to device automatically, using half-precision or contiguous layout for speed
        self.register_buffer('noe_pairs', noe_pairs.long())          # Shape: [M, 2]
        self.register_buffer('noe_bounds', noe_bounds.float())        # Shape: [M, 2] (lower, upper)
        
        has_dihedral = dihedral_indices is not None and dihedral_targets is not None
        if has_dihedral:
            self.register_buffer('dihedral_indices', dihedral_indices.long()) # Shape: [K, 4]
            self.register_buffer('dihedral_targets', dihedral_targets.float()) # Shape: [K, 2] (target, tolerance)
        else:
            self.register_buffer('dihedral_indices', torch.empty((0, 4), dtype=torch.long))
            self.register_buffer('dihedral_targets', torch.empty((0, 2), dtype=torch.float))
            
        self.force_constant = force_constant

    @torch.jit.export
    def compute_noe_penalty(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Vectorized and memory-efficient NOE distance violation penalty calculation.
        Complexity: O(M) where M is the number of NOE restraints.
        """
        if self.noe_pairs.numel() == 0:
            return torch.tensor(0.0, device=coords.device)
            
        # Gather coordinates for restraint pairs: Shape [M, 3]
        p1 = torch.index_select(coords, 0, self.noe_pairs[:, 0])
        p2 = torch.index_select(coords, 0, self.noe_pairs[:, 1])
        
        # Calculate Euclidean distances vector-wise without explicit Python loops
        distances = torch.norm(p1 - p2, dim=-1)
        
        lower_bound = self.noe_bounds[:, 0]
        upper_bound = self.noe_bounds[:, 1]
        
        # Vectorized soft-square violation mask
        lower_violations = torch.clamp(lower_bound - distances, min=0.0)
        upper_violations = torch.clamp(distances - upper_bound, min=0.0)
        total_violations = lower_violations + upper_violations
        
        return self.force_constant * torch.sum(total_violations ** 2)

    @torch.jit.export
    def compute_dihedral_penalty(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Computes dihedral angle penalties efficiently using vectorized cross products.
        """
        if self.dihedral_indices.numel() == 0:
            return torch.tensor(0.0, device=coords.device)
            
        i0 = self.dihedral_indices[:, 0]
        i1 = self.dihedral_indices[:, 1]
        i2 = self.dihedral_indices[:, 2]
        i3 = self.dihedral_indices[:, 3]
        
        b0 = coords[i1] - coords[i0]
        b1 = coords[i2] - coords[i1]
        b2 = coords[i3] - coords[i2]
        
        # Normal vectors
        n1 = torch.cross(b0, b1, dim=-1)
        n2 = torch.cross(b1, b2, dim=-1)
        
        # Vectorized angle computation via atan2
        b1_norm = torch.norm(b1, dim=-1, keepdim=True)
        y = torch.sum(torch.mul(n1, b2), dim=-1, keepdim=True) * b1_norm.squeeze(-1)
        x = torch.sum(torch.mul(n1, n2), dim=-1, keepdim=True)
        
        angles = torch.atan2(y.squeeze(-1), x.squeeze(-1))
        
        # Target differences with periodic boundary handling
        diff = torch.remainder(angles - self.dihedral_targets[:, 0] + np.pi, 2 * np.pi) - np.pi
        violations = torch.clamp(torch.abs(diff) - self.dihedral_targets[:, 1], min=0.0)
        
        return self.force_constant * 0.5 * torch.sum(violations ** 2)

    @torch.jit.export
    def compute_kabsch_rmsd(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast differentiable Kabsch RMSD calculation optimized for batch-free 
        production pipelines with O(N) matrix operations.
        """
        # Center coordinates to origin (Zero-mean)
        p_centered = pred_coords - torch.mean(pred_coords, dim=0, keepdim=True)
        t_centered = target_coords - torch.mean(target_coords, dim=0, keepdim=True)
        
        # Covariance matrix
        H = torch.matmul(p_centered.t(), t_centered)
        
        # SVD decomposition
        U, S, V = torch.svd(H)
        
        # Correct reflection if necessary
        d = torch.det(torch.matmul(V, U.t()))
        ident = torch.eye(3, device=pred_coords.device)
        if d < 0:
            ident[2, 2] = -1.0
            
        R = torch.matmul(torch.matmul(V, ident), U.t())
        
        # Rotated coordinates
        p_rotated = torch.matmul(p_centered, R)
        
        # Root Mean Square Deviation
        msd = torch.mean(torch.sum((p_rotated - t_centered) ** 2, dim=-1))
        return torch.sqrt(msd + 1e-8)

    def forward(self, coords: torch.Tensor, reference_coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass executing production-grade penalty calculation with minimal overhead.
        """
        noe_loss = self.compute_noe_penalty(coords)
        dihedral_loss = self.compute_dihedral_penalty(coords)
        
        total_loss = noe_loss + dihedral_loss
        metrics = {'noe_loss': noe_loss, 'dihedral_loss': dihedral_loss}
        
        if reference_coords is not None:
            rmsd = self.compute_kabsch_rmsd(coords, reference_coords)
            # Add harmonic RMSD pulling term for fast structural convergence
            rmsd_loss = 100.0 * rmsd
            total_loss = total_loss + rmsd_loss
            metrics['rmsd'] = rmsd
            metrics['rmsd_loss'] = rmsd_loss
            
        return total_loss, metrics
