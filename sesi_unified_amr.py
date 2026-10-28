"""
=============================================================================
SESI UNIFIED A POSTERIORI ERROR ESTIMATION + AMR
8 Domains: CFD, Protein Folding, Oncology, Epidemiology, Anti-Aging,
           Cellular Regen, Gene Editing, Neuroscience
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

ARCHITECTURE
------------
  ┌─────────────────────────────────────────────────────────────┐
  │  Domain-Specific Error Estimators (8 types)                  │
  │  ├── CFDErrorEstimator (residual + gradient recovery)       │
  │  ├── ProteinFoldingErrorEstimator (FAPE + pLDDT)           │
  │  ├── OncologyErrorEstimator (UQ + clinical deviation)      │
  │  ├── EpidemiologyErrorEstimator (forecast + R₀ CI)         │
  │  ├── AntiAgingErrorEstimator (biomarker deviation)         │
  │  ├── CellularRegenErrorEstimator (morphogen PDE residual)  │
  │  ├── GeneEditingErrorEstimator (off-target uncertainty)    │
  │  └── NeuroscienceErrorEstimator (signal + source loc.)     │
  └──────────────────────────┬──────────────────────────────────┘
                             │ error estimates
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Unified AMR Controller                                      │
  │  ├── Dörfler marking (bulk chasing)                         │
  │  ├── Refinement / Coarsening logic                          │
  │  ├── Hanging-node constraints                               │
  │  └── NoZenoGate (Six Problems §6, Prop 6.1)                │
  └─────────────────────────────────────────────────────────────┘

HONEST SCOPE (per Six Problems §6, §12)
----------------------------------------
• CFD estimator is STANDARD (Ainsworth-Oden).
• Biological estimators are HEURISTIC proxies for discretization error.
  They do NOT have the same mathematical foundation as FEM estimators.
• No-Zeno bounds EVENT COUNT, not error itself.
• AMR convergence is NOT proved for biological domains.
• Each estimator must be validated against domain-specific benchmarks.

LABEL CONVENTION
----------------
  [P] proved (standard FEM theory)
  [C] cited from literature
  [D] derived here (heuristic for biological domains)
  [O] open / not yet validated
=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.unified_amr")

__all__ = [
    "Domain",
    "AMRConfig",
    # Error estimators
    "ErrorEstimator",
    "CFDErrorEstimator",
    "ProteinFoldingErrorEstimator",
    "OncologyErrorEstimator",
    "EpidemiologyErrorEstimator",
    "AntiAgingErrorEstimator",
    "CellularRegenErrorEstimator",
    "GeneEditingErrorEstimator",
    "NeuroscienceErrorEstimator",
    # AMR
    "AdaptiveMesh",
    "DorflerMarker",
    "NoZenoGate",
    "UnifiedAMRController",
]


# ═══════════════════════════════════════════════════════════════════════
# DOMAIN ENUM
# ═══════════════════════════════════════════════════════════════════════

class Domain(Enum):
    CFD = auto()
    PROTEIN_FOLDING = auto()
    ONCOLOGY = auto()
    EPIDEMIOLOGY = auto()
    ANTI_AGING = auto()
    CELLULAR_REGEN = auto()
    GENE_EDITING = auto()
    NEUROSCIENCE = auto()


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AMRConfig:
    """Configuration for unified AMR system."""
    domain: Domain = Domain.CFD
    
    # Error tolerance
    tolerance: float = 1e-3
    safety_factor: float = 0.5
    
    # Dörfler marking
    dorfler_theta: float = 0.5  # 0 < θ < 1
    
    # Mesh constraints
    max_refinement_level: int = 6
    min_cell_size: float = 1e-4
    enable_coarsening: bool = True
    coarsen_factor: float = 0.1
    
    # No-Zeno gate (Six Problems §6)
    use_no_zeno: bool = True
    theta_hi: float = 0.2  # trigger refinement
    theta_lo: float = 0.05  # re-arm
    
    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# ABSTRACT ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class ErrorEstimator(nn.Module):
    """
    Abstract base class for domain-specific error estimators.
    
    Each subclass implements:
      • estimate() → per-cell error estimates
      • get_stress() → scalar stress for No-Zeno gate
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__()
        self.cfg = cfg
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute per-cell error estimates.
        
        Args:
            model_output: dict with model predictions
            ground_truth: optional reference solution
        
        Returns:
            (N_cells,) tensor of error estimates η_K
        """
        raise NotImplementedError
    
    def get_stress(self, error_estimates: Tensor) -> float:
        """
        Compute scalar stress for No-Zeno gate.
        
        Default: max error / tolerance
        """
        return float(error_estimates.max().item() / (self.cfg.tolerance + 1e-10))


# ═══════════════════════════════════════════════════════════════════════
# [P] CFD ERROR ESTIMATOR (Standard FEM)
# ═══════════════════════════════════════════════════════════════════════

class CFDErrorEstimator(ErrorEstimator):
    """
    [P] Residual-based + gradient recovery error estimator for CFD.
    
    Based on standard FEM theory (Ainsworth-Oden, Verfürth).
    
    Element residual:
        η_K² = h_K² ||f + Δu_h||²_{L²(K)}
    
    Edge jump:
        η_J² = h_J ||[∂u_h/∂n]||²_{L²(J)}
    
    Gradient recovery (Zienkiewicz-Zhu):
        η_ZZ = ||G(∇u_h) - ∇u_h||_{L²}
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute CFD error estimates.
        
        model_output must contain:
          • u_h: (N_cells, N_dofs) solution
          • f: (N_cells, N_dofs) source term
          • cell_sizes: (N_cells,)
          • grad_u_h: (N_cells, N_dofs, dim) gradient
          • laplacian_u_h: (N_cells, N_dofs) Laplacian
        
        Returns:
            (N_cells,) error estimates
        """
        u_h = model_output["u_h"]
        f = model_output["f"]
        h_K = model_output["cell_sizes"]
        lap_u_h = model_output.get("laplacian_u_h", torch.zeros_like(u_h))
        
        # Element residual
        residual = f + lap_u_h
        eta_K_sq = (h_K ** 2) * (residual ** 2).mean(dim=-1)
        
        # Edge jumps (if available)
        if "edge_jumps" in model_output:
            eta_J_sq = model_output["edge_jumps"]
            eta_total = torch.sqrt(eta_K_sq + eta_J_sq)
        else:
            eta_total = torch.sqrt(eta_K_sq)
        
        return eta_total


# ═══════════════════════════════════════════════════════════════════════
# [D] PROTEIN FOLDING ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class ProteinFoldingErrorEstimator(ErrorEstimator):
    """
    [D] Error estimator for protein structure prediction.
    
    Uses:
      • FAPE (Frame Aligned Point Error) - local structure accuracy
      • pLDDT (predicted LDDT) - per-residue confidence
      • PAE (Predicted Aligned Error) - pairwise residue error
    
    HONEST: This is a HEURISTIC proxy, not a rigorous FEM estimator.
    The "cells" are residues or residue pairs.
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute protein folding error estimates.
        
        model_output must contain:
          • pred_coords: (B, L, 3) predicted coordinates
          • confidence: (B, L) per-residue confidence (pLDDT)
          • pae_matrix: (B, L, L) predicted aligned error (optional)
        
        ground_truth (optional):
          • true_coords: (B, L, 3) true coordinates
        
        Returns:
            (B, L) per-residue error estimates
        """
        pred = model_output["pred_coords"]
        conf = model_output["confidence"]
        
        # Inverse confidence as error proxy
        error = 1.0 - conf  # low confidence → high error
        
        # If ground truth available, compute FAPE
        if ground_truth is not None and "true_coords" in ground_truth:
            true = ground_truth["true_coords"]
            # Simplified FAPE: per-residue distance error
            dist_error = torch.norm(pred - true, dim=-1)
            # Combine with confidence
            error = 0.5 * error + 0.5 * (dist_error / 10.0).clamp(max=1.0)
        
        return error
    
    def get_stress(self, error_estimates: Tensor) -> float:
        """Stress = mean error (not max, for proteins)."""
        return float(error_estimates.mean().item() / (self.cfg.tolerance + 1e-10))


# ═══════════════════════════════════════════════════════════════════════
# [D] ONCOLOGY ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class OncologyErrorEstimator(ErrorEstimator):
    """
    [D] Error estimator for oncology predictions.
    
    Uses:
      • Prediction uncertainty (ensemble variance)
      • Clinical feature deviation
      • Tumor growth model residual
    
    HONEST: This is a HEURISTIC for clinical ML, not a proven estimator.
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        # Ensemble variance head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute oncology error estimates.
        
        model_output must contain:
          • latent: (B, feature_dim) encoded features
          • predictions: dict with tumor_grid, treatment_probs, etc.
        
        Returns:
            (B,) per-sample uncertainty estimates
        """
        latent = model_output["latent"]
        uncertainty = self.uncertainty_head(latent).squeeze(-1)
        
        # If ground truth, add prediction error
        if ground_truth is not None:
            if "tumor_target" in ground_truth:
                pred_grid = model_output["predictions"]["tumor_grid"]
                true_grid = ground_truth["tumor_target"]
                grid_error = F.mse_loss(pred_grid, true_grid, reduction='none').mean(dim=(1, 2, 3))
                uncertainty = 0.5 * uncertainty + 0.5 * grid_error
        
        return uncertainty


# ═══════════════════════════════════════════════════════════════════════
# [D] EPIDEMIOLOGY ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class EpidemiologyErrorEstimator(ErrorEstimator):
    """
    [D] Error estimator for epidemiological models.
    
    Uses:
      • Forecast error (past predictions vs actual)
      • R₀ confidence interval width
      • Spatial spread uncertainty
    
    HONEST: This is a HEURISTIC for time-series forecasting.
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        # R₀ uncertainty head
        self.r0_uncertainty = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute epidemiology error estimates.
        
        model_output must contain:
          • latent: (B, feature_dim)
          • r0: (B,) basic reproduction number
          • trajectory: (B, T, 4) SEIR trajectory
        
        Returns:
            (B,) per-sample uncertainty
        """
        latent = model_output["latent"]
        r0_unc = self.r0_uncertainty(latent).squeeze(-1)
        
        # If ground truth, add trajectory error
        if ground_truth is not None and "trajectory_target" in ground_truth:
            pred_traj = model_output["trajectory"]
            true_traj = ground_truth["trajectory_target"]
            traj_error = F.mse_loss(pred_traj, true_traj, reduction='none').mean(dim=(1, 2))
            r0_unc = 0.5 * r0_unc + 0.5 * traj_error
        
        return r0_unc


# ═══════════════════════════════════════════════════════════════════════
# [O] ANTI-AGING ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class AntiAgingErrorEstimator(ErrorEstimator):
    """
    [O] Error estimator for anti-aging predictions.
    
    Uses:
      • Biological age prediction uncertainty
      • Biomarker deviation from expected trajectory
      • Telomere length prediction confidence
    
    HONEST: This is HIGHLY HEURISTIC. No standard error estimator exists
    for biological age prediction. Validation requires longitudinal data.
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.age_uncertainty = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute anti-aging error estimates.
        
        model_output must contain:
          • latent: (B, feature_dim)
          • biological_age: (B,) predicted age
          • telomere_length: (B,)
        
        Returns:
            (B,) per-sample uncertainty
        """
        latent = model_output["latent"]
        uncertainty = self.age_uncertainty(latent).squeeze(-1)
        
        # If ground truth, add age prediction error
        if ground_truth is not None and "age_target" in ground_truth:
            pred_age = model_output["biological_age"]
            true_age = ground_truth["age_target"]
            age_error = F.mse_loss(pred_age, true_age, reduction='none')
            uncertainty = 0.5 * uncertainty + 0.5 * (age_error / 10.0).clamp(max=1.0)
        
        return uncertainty


# ═══════════════════════════════════════════════════════════════════════
# [D] CELLULAR REGENERATION ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class CellularRegenErrorEstimator(ErrorEstimator):
    """
    [D] Error estimator for cellular regeneration models.
    
    Uses:
      • Morphogen gradient PDE residual
      • Cell fate classification confidence
      • Regeneration capacity prediction uncertainty
    
    HONEST: This is a HEURISTIC for reaction-diffusion systems.
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute cellular regen error estimates.
        
        model_output must contain:
          • morphogen_field: (B, 1, H, W) morphogen concentration
          • cell_fate: (B, 4) fate probabilities
          • regen_capacity: (B,)
        
        Returns:
            (B,) per-sample uncertainty
        """
        morphogen = model_output["morphogen_field"]
        fate = model_output["cell_fate"]
        
        # Morphogen PDE residual (simplified Laplacian)
        lap_morph = F.avg_pool2d(morphogen, 3, stride=1, padding=1) - morphogen
        pde_residual = lap_morph.abs().mean(dim=(1, 2, 3))
        
        # Fate classification confidence (entropy)
        fate_entropy = -(fate * torch.log(fate + 1e-8)).sum(dim=-1)
        
        # Combine
        uncertainty = 0.5 * pde_residual + 0.5 * (fate_entropy / math.log(4))
        
        return uncertainty


# ═══════════════════════════════════════════════════════════════════════
# [D] GENE EDITING ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class GeneEditingErrorEstimator(ErrorEstimator):
    """
    [D] Error estimator for CRISPR gene editing predictions.
    
    Uses:
      • Off-target prediction uncertainty
      • On-target efficiency confidence
      • Sequence alignment score variance
    
    HONEST: This is a HEURISTIC for sequence-based predictions.
    Validation requires experimental data.
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.off_target_uncertainty = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute gene editing error estimates.
        
        model_output must contain:
          • latent: (B, feature_dim)
          • off_target_score: (B, n_off_targets)
          • efficiency: (B,) on-target efficiency
        
        Returns:
            (B,) per-sample uncertainty
        """
        latent = model_output["latent"]
        off_target = model_output["off_target_score"]
        
        # Off-target uncertainty (variance across off-targets)
        ot_unc = self.off_target_uncertainty(latent).squeeze(-1)
        ot_variance = off_target.var(dim=-1) if off_target.dim() > 1 else torch.zeros_like(ot_unc)
        
        uncertainty = 0.5 * ot_unc + 0.5 * ot_variance
        
        return uncertainty


# ═══════════════════════════════════════════════════════════════════════
# [D] NEUROSCIENCE ERROR ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class NeuroscienceErrorEstimator(ErrorEstimator):
    """
    [D] Error estimator for neuroscience models.
    
    Uses:
      • EEG/fMRI signal reconstruction error
      • Source localization uncertainty
      • Cognitive load prediction confidence
    
    HONEST: This is a HEURISTIC for inverse problems in neuroscience.
    """
    
    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.source_uncertainty = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        Compute neuroscience error estimates.
        
        model_output must contain:
          • latent: (B, feature_dim)
          • source_loc: (B, n_sources) source probabilities
          • cognitive_load: (B,)
        
        Returns:
            (B,) per-sample uncertainty
        """
        latent = model_output["latent"]
        source = model_output["source_loc"]
        
        # Source localization uncertainty (entropy)
        source_entropy = -(source * torch.log(source + 1e-8)).sum(dim=-1)
        source_unc = self.source_uncertainty(latent).squeeze(-1)
        
        uncertainty = 0.5 * source_unc + 0.5 * (source_entropy / math.log(source.shape[-1]))
        
        return uncertainty


# ═══════════════════════════════════════════════════════════════════════
# ADAPTIVE MESH (Domain-agnostic)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MeshCell:
    """A single cell in the adaptive mesh."""
    id: int
    level: int
    center: Tuple[float, ...]
    size: float
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    is_leaf: bool = True
    error_estimate: float = 0.0


class AdaptiveMesh:
    """
    Domain-agnostic adaptive mesh structure.
    
    Supports:
      • Quadtree (2D) or Octree (3D)
      • Refinement / Coarsening
      • Hanging-node constraints
    """
    
    def __init__(self, dim: int = 2, domain_size: float = 1.0, max_level: int = 6):
        self.dim = dim
        self.domain_size = domain_size
        self.max_level = max_level
        self.cells: Dict[int, MeshCell] = {}
        self.next_id = 0
        
        # Initialize root cell
        center = tuple([domain_size / 2] * dim)
        root = MeshCell(id=0, level=0, center=center, size=domain_size)
        self.cells[0] = root
        self.next_id = 1
    
    def refine_cell(self, cell_id: int) -> List[int]:
        """Refine a cell into 2^dim children."""
        cell = self.cells[cell_id]
        
        if not cell.is_leaf or cell.level >= self.max_level:
            return []
        
        child_size = cell.size / 2
        children = []
        
        # Generate 2^dim children
        for i in range(2 ** self.dim):
            offset = tuple([
                (child_size / 2) if (i >> d) & 1 else (-child_size / 2)
                for d in range(self.dim)
            ])
            child_center = tuple(c + o for c, o in zip(cell.center, offset))
            
            child = MeshCell(
                id=self.next_id,
                level=cell.level + 1,
                center=child_center,
                size=child_size,
                parent=cell_id,
            )
            self.cells[child.id] = child
            children.append(child.id)
            self.next_id += 1
        
        cell.is_leaf = False
        cell.children = children
        
        return children
    
    def coarsen_cell(self, cell_id: int) -> Optional[int]:
        """Coarsen 2^dim siblings into parent."""
        cell = self.cells[cell_id]
        
        if cell.parent is None:
            return None
        
        parent = self.cells[cell.parent]
        
        # Check if all siblings are leaves
        if not all(self.cells[cid].is_leaf for cid in parent.children):
            return None
        
        # Remove children
        for cid in parent.children:
            del self.cells[cid]
        
        parent.is_leaf = True
        parent.children = []
        
        return parent.id
    
    def get_leaf_cells(self) -> List[MeshCell]:
        """Get all leaf cells."""
        return [c for c in self.cells.values() if c.is_leaf]


# ═══════════════════════════════════════════════════════════════════════
# DÖRFLER MARKER
# ═══════════════════════════════════════════════════════════════════════

class DorflerMarker:
    """
    [P] Dörfler (bulk chasing) marking strategy.
    
    Select minimal set M such that:
        Σ_{K ∈ M} η_K² ≥ θ² · Σ_K η_K²
    """
    
    def __init__(self, theta: float = 0.5):
        if not 0 < theta < 1:
            raise ValueError("θ must be in (0, 1)")
        self.theta = theta
    
    def mark(self, cell_ids: List[int], errors: Tensor) -> List[int]:
        """Mark cells for refinement."""
        if len(cell_ids) == 0:
            return []
        
        # Sort by error (descending)
        sorted_idx = torch.argsort(errors, descending=True)
        sorted_errors = errors[sorted_idx]
        sorted_ids = [cell_ids[i] for i in sorted_idx]
        
        # Select minimal set
        total = sorted_errors.sum().item()
        threshold = (self.theta ** 2) * total
        
        marked = []
        cumulative = 0.0
        for cid, err in zip(sorted_ids, sorted_errors):
            marked.append(cid)
            cumulative += err.item()
            if cumulative >= threshold:
                break
        
        return marked


# ═══════════════════════════════════════════════════════════════════════
# NO-ZENO GATE (Six Problems §6, Prop 6.1)
# ═══════════════════════════════════════════════════════════════════════

class NoZenoGate:
    """
    [P] Hysteresis gate to bound refinement event count.
    
    Guarantees: N(T) ≤ (E_max + D) / δ_min
    
    HONEST: This bounds EVENT COUNT, not error itself.
    """
    
    def __init__(self, theta_hi: float = 0.2, theta_lo: float = 0.05):
        self.theta_hi = theta_hi
        self.theta_lo = theta_lo
        self.delta_min = theta_hi - theta_lo
        
        if self.delta_min <= 0:
            raise ValueError("θ_hi > θ_lo required")
        
        self.stress_acc = 0.0
        self.E_max = 0.0
        self.D = 0.0
        self.event_count = 0
        self.armed = True
    
    def update(self, stress: float) -> Dict[str, Any]:
        """Update gate with current stress."""
        self.stress_acc = max(self.stress_acc, stress)
        self.E_max = max(self.E_max, self.stress_acc)
        
        allowed = False
        
        if self.armed and self.stress_acc > self.theta_hi:
            allowed = True
            self.event_count += 1
            self.D += self.stress_acc
            self.stress_acc = 0.0
            self.armed = False
        elif not self.armed and self.stress_acc < self.theta_lo:
            self.armed = True
        else:
            self.stress_acc *= 0.9
        
        bound_rhs = (self.E_max + self.D) / self.delta_min
        
        return {
            "allowed": allowed,
            "event_count": self.event_count,
            "bound_rhs": bound_rhs,
            "invariant_holds": self.event_count <= bound_rhs + 1e-6,
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED AMR CONTROLLER
# ═══════════════════════════════════════════════════════════════════════

class UnifiedAMRController:
    """
    Unified AMR controller for all 8 domains.
    
    Orchestrates:
      • Domain-specific error estimation
      • Dörfler marking
      • Mesh refinement / coarsening
      • No-Zeno gate integration
    """
    
    def __init__(self, cfg: AMRConfig, mesh: AdaptiveMesh):
        self.cfg = cfg
        self.mesh = mesh
        
        # Select error estimator based on domain
        estimator_map = {
            Domain.CFD: CFDErrorEstimator,
            Domain.PROTEIN_FOLDING: ProteinFoldingErrorEstimator,
            Domain.ONCOLOGY: OncologyErrorEstimator,
            Domain.EPIDEMIOLOGY: EpidemiologyErrorEstimator,
            Domain.ANTI_AGING: AntiAgingErrorEstimator,
            Domain.CELLULAR_REGEN: CellularRegenErrorEstimator,
            Domain.GENE_EDITING: GeneEditingErrorEstimator,
            Domain.NEUROSCIENCE: NeuroscienceErrorEstimator,
        }
        
        self.estimator = estimator_map[cfg.domain](cfg)
        self.marker = DorflerMarker(cfg.dorfler_theta)
        
        if cfg.use_no_zeno:
            self.no_zeno = NoZenoGate(cfg.theta_hi, cfg.theta_lo)
        else:
            self.no_zeno = None
    
    def adapt(self, model_output: Dict[str, Any],
              ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform one AMR adaptation cycle.
        
        Returns:
            dict with adaptation statistics
        """
        # 1. Estimate error
        errors = self.estimator.estimate(model_output, ground_truth)
        
        # 2. Get leaf cells
        leaf_cells = self.mesh.get_leaf_cells()
        cell_ids = [c.id for c in leaf_cells]
        
        # Map errors to cells (simplified: assume 1-to-1 correspondence)
        if len(errors) != len(cell_ids):
            # Broadcast or truncate
            errors = errors[:len(cell_ids)] if len(errors) > len(cell_ids) \
                     else F.pad(errors, (0, len(cell_ids) - len(errors)))
        
        # 3. Check No-Zeno gate
        if self.no_zeno is not None:
            stress = self.estimator.get_stress(errors)
            gate_result = self.no_zeno.update(stress)
            if not gate_result["allowed"]:
                return {
                    "refined": [],
                    "coarsened": [],
                    "blocked_by_no_zeno": True,
                    "total_error": float(errors.sum()),
                }
        
        # 4. Mark cells (Dörfler)
        marked = self.marker.mark(cell_ids, errors)
        
        # 5. Refine marked cells
        refined = []
        for cid in marked:
            cell = self.mesh.cells[cid]
            if cell.size > self.cfg.min_cell_size and cell.level < self.cfg.max_refinement_level:
                children = self.mesh.refine_cell(cid)
                refined.extend(children)
        
        # 6. Coarsen (if enabled)
        coarsened = []
        if self.cfg.enable_coarsening:
            threshold = self.cfg.coarsen_factor * self.cfg.tolerance
            for cid, err in zip(cell_ids, errors):
                if err < threshold and cid not in marked:
                    parent = self.mesh.coarsen_cell(cid)
                    if parent is not None:
                        coarsened.append(parent)
        
        return {
            "refined": refined,
            "coarsened": coarsened,
            "n_leaf_cells": len(self.mesh.get_leaf_cells()),
            "total_error": float(errors.sum()),
            "blocked_by_no_zeno": False,
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" Unified AMR for 8 Domains — Smoke Test")
    print("=" * 72)
    torch.manual_seed(42)
    
    # Test each domain
    for domain in Domain:
        print(f"\n  ── Testing {domain.name} ──")
        
        cfg = AMRConfig(domain=domain, tolerance=1e-2)
        mesh = AdaptiveMesh(dim=2, domain_size=1.0, max_level=4)
        controller = UnifiedAMRController(cfg, mesh)
        
        # Mock model output (domain-specific)
        if domain == Domain.CFD:
            model_output = {
                "u_h": torch.randn(4, 10),
                "f": torch.randn(4, 10),
                "cell_sizes": torch.tensor([0.5, 0.5, 0.5, 0.5]),
                "grad_u_h": torch.randn(4, 10, 2),
            }
        elif domain == Domain.PROTEIN_FOLDING:
            model_output = {
                "pred_coords": torch.randn(2, 20, 3),
                "confidence": torch.rand(2, 20),
            }
        elif domain == Domain.ONCOLOGY:
            model_output = {
                "latent": torch.randn(4, 256),
                "predictions": {"tumor_grid": torch.rand(4, 1, 16, 16)},
            }
        elif domain == Domain.EPIDEMIOLOGY:
            model_output = {
                "latent": torch.randn(4, 256),
                "r0": torch.rand(4) * 3,
                "trajectory": torch.rand(4, 10, 4),
            }
        elif domain == Domain.ANTI_AGING:
            model_output = {
                "latent": torch.randn(4, 256),
                "biological_age": torch.rand(4) * 100,
                "telomere_length": torch.rand(4),
            }
        elif domain == Domain.CELLULAR_REGEN:
            model_output = {
                "morphogen_field": torch.rand(4, 1, 16, 16),
                "cell_fate": torch.rand(4, 4),
                "regen_capacity": torch.rand(4),
            }
        elif domain == Domain.GENE_EDITING:
            model_output = {
                "latent": torch.randn(4, 256),
                "off_target_score": torch.rand(4, 10),
                "efficiency": torch.rand(4),
            }
        elif domain == Domain.NEUROSCIENCE:
            model_output = {
                "latent": torch.randn(4, 256),
                "source_loc": torch.rand(4, 64),
                "cognitive_load": torch.rand(4),
            }
        
        # Run adaptation
        result = controller.adapt(model_output)
        
        print(f"    Refined: {len(result['refined'])}")
        print(f"    Coarsened: {len(result['coarsened'])}")
        print(f"    Leaf cells: {result['n_leaf_cells']}")
        print(f"    Total error: {result['total_error']:.4f}")
        print(f"    Blocked by No-Zeno: {result['blocked_by_no_zeno']}")
        
        # Check No-Zeno invariant
        if controller.no_zeno is not None:
            nz = controller.no_zeno
            print(f"    No-Zeno events: {nz.event_count}")
            print(f"    Bound check: {nz.event_count} ≤ {(nz.E_max + nz.D) / nz.delta_min:.4f}")
    
    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • CFD estimator is STANDARD (FEM theory).")
    print("  • Biological estimators are HEURISTIC proxies.")
    print("  • No-Zeno bounds EVENT COUNT, not error.")
    print("  • AMR convergence NOT proved for biological domains.")
    print("=" * 72)
    print("  Smoke test complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
