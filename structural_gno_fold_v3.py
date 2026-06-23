# =============================================================================
# STRUCTURAL GRAPH NEURAL OPERATOR (SGNO FOLD) — v3 Production
# Unified Discrete & Continuous Physics Surrogate for REAL FOLD ONE Ecosystem
# =============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026
#
# AI Co-Developers (architecture, numerical methods, production hardening):
#   - Claude   (Anthropic)  — production refactor, EMA checkpointing,
#                             multi-loss weighting, physics-informed losses,
#                             LR scheduling, gradient monitoring, full docstrings
#   - GPT      (OpenAI)     — early architecture exploration, message-passing
#                             design, phase-field surrogate concept
#   - Gemini   (Google)     — v2 unified discrete/continuous extension,
#                             one-shot phase evolution framing
#
# Description:
#   Production-grade Structural Graph Neural Operator (SGNO) — the AI surrogate
#   and training engine for the REAL FOLD ONE ecosystem.
#
#   The model learns two coupled operator mappings:
#
#     Protein mode  : G_p : (X_0, Seq, σ(X)) ↦ (X_final, ΔΔG)
#     Phase-field   : G_c : (u_0, σ_3D)       ↦ u_T  (one-shot CH/PFC evolution)
#
#   Ecosystem integration:
#     • one_core_fold.py             — SemanticStateContraction, CSOCBase
#     • real_fold_one_v2.py          — SOCController, RefinementEngine
#     • real_fold_one_ht_v2.py       — HighThroughputScanner (data generator)
#     • structural_langevin_fold_v2.py — AdvancedStructuralLangevin (BAOAB)
#     • structural_cahn_hilliard_3d.py — StructuralCahnHilliard3D (CH/PFC)
#
#   v3 Production additions:
#     • SGNOConfig       — centralised, validated hyperparameter dataclass
#     • FiLM-based sigma modulation (replaces additive gate)
#     • Attention-augmented pooling for DDG head
#     • Separate LR groups (backbone / output heads)
#     • OneCycleLR scheduler with warmup
#     • EMA weight averaging for inference stability
#     • Checkpoint save/load with metadata
#     • GradMonitor — per-layer gradient norm logging
#     • SGNODataset  — PyTorch Dataset wrapping RefinementEngine outputs
#     • SGNOEvaluator — RMSD, Pearson-r DDG, CH energy monotonicity metrics
#     • Physics-informed mass-conservation loss for CH mode
#     • Full type annotations throughout
# =============================================================================

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

# =============================================================================
# Optional ecosystem imports (graceful fallback for standalone execution)
# =============================================================================
try:
    from one_core_fold import (
        SemanticStateContraction,
        CSOCBase,
        FOLD_VERSION,
    )
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False
    SemanticStateContraction = None  # type: ignore[assignment]
    CSOCBase = nn.Module            # type: ignore[assignment]
    FOLD_VERSION = "unknown"
    warnings.warn("one_core_fold not found — running in standalone mode.")

try:
    from real_fold_one_v2 import SOCController, RefinementEngine, RefinementConfig
    _HAS_FOLD = True
except ImportError:
    _HAS_FOLD = False
    SOCController = None        # type: ignore[assignment]
    RefinementEngine = None     # type: ignore[assignment]
    RefinementConfig = None     # type: ignore[assignment]

try:
    from structural_langevin_fold_v2 import AdvancedStructuralLangevin
    _HAS_LANGEVIN = True
except ImportError:
    _HAS_LANGEVIN = False
    AdvancedStructuralLangevin = None  # type: ignore[assignment]

try:
    from structural_cahn_hilliard_3d import (
        StructuralCahnHilliard3D,
        ThinFilmStructuralCahnHilliard3D,
        PhaseFieldCrystal3D,
        CahnHilliardConfig,
    )
    _HAS_CH = True
except ImportError:
    _HAS_CH = False
    StructuralCahnHilliard3D      = None  # type: ignore[assignment]
    ThinFilmStructuralCahnHilliard3D = None  # type: ignore[assignment]
    PhaseFieldCrystal3D           = None  # type: ignore[assignment]
    CahnHilliardConfig            = None  # type: ignore[assignment]

SGNO_VERSION: str = "3.0.0"


# =============================================================================
# 1.  Configuration Dataclass
# =============================================================================

@dataclass
class SGNOConfig:
    """
    Centralised, validated hyperparameter store for StructuralGNOFold.

    Architecture
    ------------
    node_in_dim   : Input amino-acid / voxel feature dimension.
    hidden_dim    : Latent node embedding dimension.
    num_layers    : Number of StructuralMessagePassing layers.
    dropout       : Dropout probability applied after each MP layer.

    Graph construction
    ------------------
    cutoff_protein : Å radius for protein residue graph.
    cutoff_grid    : Grid-unit radius for phase-field voxel graph.

    Training
    --------
    lr_backbone    : Learning rate for message-passing backbone.
    lr_heads       : Learning rate for output heads (coord / ddg / phase).
    weight_decay   : AdamW weight-decay coefficient.
    max_epochs     : Total training epochs.
    warmup_epochs  : Linear LR warmup epochs (OneCycleLR).
    grad_clip      : Max gradient norm for clipping.
    ema_decay      : Exponential moving average decay for inference weights.
    lambda_ddg     : Loss weight for ΔΔG term (protein mode).
    lambda_physics : Loss weight for physics-informed term (phase-field mode).
    lambda_mass    : Loss weight for CH mass-conservation term.

    Checkpoint
    ----------
    checkpoint_dir : Directory for checkpoint files.
    save_every     : Save checkpoint every N epochs.
    """

    # Architecture
    node_in_dim:    int   = 20
    hidden_dim:     int   = 128
    num_layers:     int   = 6
    dropout:        float = 0.1

    # Graph construction
    cutoff_protein: float = 10.0
    cutoff_grid:    float = 1.5

    # Training
    lr_backbone:    float = 3e-4
    lr_heads:       float = 1e-3
    weight_decay:   float = 1e-4
    max_epochs:     int   = 200
    warmup_epochs:  int   = 10
    grad_clip:      float = 1.0
    ema_decay:      float = 0.999
    lambda_ddg:     float = 0.1
    lambda_physics: float = 0.05
    lambda_mass:    float = 0.01

    # Checkpointing
    checkpoint_dir: str   = "./sgno_checkpoints"
    save_every:     int   = 10

    def __post_init__(self) -> None:
        assert self.hidden_dim > 0
        assert self.num_layers >= 1
        assert 0.0 <= self.dropout < 1.0
        assert self.cutoff_protein > 0.0
        assert self.grad_clip > 0.0
        assert 0.0 < self.ema_decay < 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SGNOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# 2.  FiLM-Modulated Structural Message Passing
# =============================================================================

class StructuralMessagePassing(nn.Module):
    """
    Message-passing layer with **FiLM-based** structural regime modulation.

    Standard additive gates are replaced by Feature-wise Linear Modulation
    (Perez et al., 2018):

        modulated = gamma(σ) * aggregated_msg + beta(σ)

    This allows σ to both scale and shift the aggregated latent signal,
    giving strictly more expressive structural coupling than a sigmoid gate.

    Args:
        node_dim  : Dimension of input node features.
        edge_dim  : Dimension of edge features.
        out_dim   : Output node feature dimension.
        dropout   : Dropout applied after the update MLP.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim:  int,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()

        # Message network: processes source, dest, and edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

        # Update network: combines node state with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )

        # FiLM modulator: σ → (γ, β) pair
        self.film_gamma = nn.Linear(1, out_dim)
        self.film_beta  = nn.Linear(1, out_dim)

        # Layer norm for training stability
        self.norm = nn.LayerNorm(out_dim)

        # Residual projection when node_dim ≠ out_dim
        self.res_proj = (
            nn.Linear(node_dim, out_dim, bias=False)
            if node_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x:          torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, edge_dim)
        sigma:      torch.Tensor,   # (N, 1)  structural regime
    ) -> torch.Tensor:
        """
        Returns updated node features of shape (N, out_dim).
        Gradient flows through all inputs including sigma.
        """
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        # --- Messages ---
        msg_in  = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.message_mlp(msg_in)                       # (E, out_dim)

        # --- Aggregation (sum) ---
        aggr = x.new_zeros(N, messages.size(1))
        aggr.index_add_(0, dst, messages)                         # (N, out_dim)

        # --- FiLM modulation (replaces sigmoid gate) ---
        gamma = self.film_gamma(sigma)                            # (N, out_dim)
        beta  = self.film_beta(sigma)                             # (N, out_dim)
        modulated = gamma * aggr + beta                           # (N, out_dim)

        # --- Update + residual ---
        upd_in  = torch.cat([x, modulated], dim=-1)
        x_new   = self.update_mlp(upd_in)                        # (N, out_dim)
        out     = self.norm(self.res_proj(x) + x_new)
        return out


# =============================================================================
# 3.  Attention Pooling
# =============================================================================

class AttentionPooling(nn.Module):
    """
    Soft attention-based global graph pooling.

    Computes a weighted mean over node embeddings:
        e_i = MLP(h_i) ∈ R  (scalar attention score)
        a_i = softmax(e_i)
        z   = Σ_i a_i * h_i

    Substantially more expressive than mean pooling for graph-level
    predictions such as ΔΔG, where a small number of residues dominates
    the energetic signal.

    Args:
        hidden_dim : Node feature dimension.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, hidden_dim) node embeddings.
        Returns:
            z : (hidden_dim,) graph-level embedding.
        """
        scores = self.score(x)                   # (N, 1)
        weights = torch.softmax(scores, dim=0)   # (N, 1)
        return (weights * x).sum(dim=0)          # (hidden_dim,)


# =============================================================================
# 4.  StructuralGNOFold  (main model)
# =============================================================================

class StructuralGNOFold(nn.Module):
    """
    Structural Graph Neural Operator — production v3.

    Supports two forward modes:

    **Protein mode** (``forward``):
        Input  : (seq_features, init_coords, sigma)
        Output : (final_coords, pred_ddg)

    **Phase-field mode** (``forward_phase_field``):
        Input  : (u_init, sigma_3d)
        Output : u_pred_future  (one-shot surrogate of CH/PFC time-stepping)

    Both modes share the same StructuralMessagePassing backbone weighted by
    the structural regime field σ, ensuring physical consistency throughout.

    Args:
        cfg : SGNOConfig instance (created with defaults if None).
    """

    def __init__(self, cfg: Optional[SGNOConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or SGNOConfig()
        d = self.cfg.hidden_dim

        # --- Protein mode embeddings ---
        self.node_embed = nn.Sequential(
            nn.Linear(self.cfg.node_in_dim, d),
            nn.LayerNorm(d),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(1, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # --- Phase-field mode embeddings ---
        # Input: [u, x, y, z] → 4 features per voxel
        self.grid_node_embed = nn.Sequential(
            nn.Linear(4, d),
            nn.LayerNorm(d),
        )

        # --- Shared FiLM-modulated backbone ---
        self.layers = nn.ModuleList([
            StructuralMessagePassing(d, d, d, dropout=self.cfg.dropout)
            for _ in range(self.cfg.num_layers)
        ])

        # --- Output heads: protein mode ---
        self.coord_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(d // 2, 3),
        )
        self.attn_pool = AttentionPooling(d)
        self.ddg_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(d // 2, 1),
        )

        # --- Output head: phase-field mode ---
        self.phase_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(d // 2, 1),
        )

        self._init_weights()
        logger.info(
            "StructuralGNOFold v%s | params=%s | hidden=%d | layers=%d",
            SGNO_VERSION,
            f"{sum(p.numel() for p in self.parameters()):,}",
            d,
            self.cfg.num_layers,
        )

    def _init_weights(self) -> None:
        """Xavier uniform for Linear, zeros for bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _build_protein_graph(
        self,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build radius graph from Cα coordinates.

        Args:
            coords : (N, 3) Å
        Returns:
            edge_index : (2, E)
            edge_attr  : (E, 1) Euclidean distances
        """
        dist_mat = torch.cdist(coords, coords)   # (N, N)
        adj      = (dist_mat < self.cfg.cutoff_protein) & (dist_mat > 1e-6)
        edge_idx = torch.nonzero(adj, as_tuple=False).t().contiguous()
        src, dst = edge_idx[0], edge_idx[1]
        edge_attr = dist_mat[src, dst].unsqueeze(-1)
        return edge_idx, edge_attr

    def _build_grid_graph(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert 3-D phase-field grid into a voxel graph.

        Uses integer grid coordinates so that edges only connect
        adjacent/near-adjacent voxels within ``cutoff_grid`` grid units.

        Args:
            u : (Nx, Ny, Nz) order parameter
        Returns:
            node_feats : (Nx*Ny*Nz, 4)  [u, x, y, z]
            edge_index : (2, E)
            edge_attr  : (E, 1)
        """
        nx, ny, nz    = u.shape
        device, dtype = u.device, u.dtype

        gx = torch.arange(nx, device=device, dtype=dtype)
        gy = torch.arange(ny, device=device, dtype=dtype)
        gz = torch.arange(nz, device=device, dtype=dtype)
        GX, GY, GZ = torch.meshgrid(gx, gy, gz, indexing="ij")

        coords     = torch.stack([GX.flatten(), GY.flatten(), GZ.flatten()], dim=-1)  # (Ng, 3)
        u_flat     = u.flatten().unsqueeze(-1)                                         # (Ng, 1)
        node_feats = torch.cat([u_flat, coords], dim=-1)                               # (Ng, 4)

        dist_mat  = torch.cdist(coords, coords)
        adj       = (dist_mat <= self.cfg.cutoff_grid) & (dist_mat > 1e-6)
        edge_idx  = torch.nonzero(adj, as_tuple=False).t().contiguous()
        src, dst  = edge_idx[0], edge_idx[1]
        edge_attr = dist_mat[src, dst].unsqueeze(-1)

        return node_feats, edge_idx, edge_attr

    # ------------------------------------------------------------------
    # Forward: protein mode
    # ------------------------------------------------------------------

    def forward(
        self,
        seq_features: torch.Tensor,   # (N, node_in_dim)
        init_coords:  torch.Tensor,   # (N, 3)  Å
        sigma:        torch.Tensor,   # (N, 1)  structural regime
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Protein folding / mutation-impact prediction.

        Returns:
            final_coords : (N, 3) — refined Cα positions (Å)
            pred_ddg     : (1,)   — predicted ΔΔG (kcal/mol)
        """
        edge_index, edge_attr = self._build_protein_graph(init_coords)

        x = self.node_embed(seq_features)      # (N, d)
        e = self.edge_embed(edge_attr)         # (E, d)

        for layer in self.layers:
            x = layer(x, edge_index, e, sigma)

        # Coordinate refinement (residual displacement)
        displacements = self.coord_head(x)     # (N, 3)
        final_coords  = init_coords + displacements

        # Attention-pooled DDG prediction
        graph_embed = self.attn_pool(x)        # (d,)
        pred_ddg    = self.ddg_head(graph_embed)  # (1,)

        return final_coords, pred_ddg

    # ------------------------------------------------------------------
    # Forward: phase-field mode
    # ------------------------------------------------------------------

    def forward_phase_field(
        self,
        u_init:   torch.Tensor,   # (Nx, Ny, Nz)
        sigma_3d: torch.Tensor,   # (Nx, Ny, Nz)
    ) -> torch.Tensor:
        """
        One-shot surrogate prediction of Cahn-Hilliard / PFC time evolution.

        Learns the operator G_c : (u_0, σ) → u_T, bypassing potentially
        thousands of explicit PDE time-steps.

        Returns:
            u_pred : (Nx, Ny, Nz)
        """
        shape_3d = u_init.shape

        node_feats, edge_index, edge_attr = self._build_grid_graph(u_init)
        sigma_flat = sigma_3d.flatten().unsqueeze(-1)   # (Ng, 1)

        x = self.grid_node_embed(node_feats)            # (Ng, d)
        e = self.edge_embed(edge_attr)                  # (E,  d)

        for layer in self.layers:
            x = layer(x, edge_index, e, sigma_flat)

        delta_u = self.phase_head(x).view(shape_3d)    # (Nx, Ny, Nz)
        return u_init + delta_u


# =============================================================================
# 5.  EMA (Exponential Moving Average) wrapper
# =============================================================================

class EMAWrapper:
    """
    Maintains an exponential moving average of model parameters
    for more stable inference.

    Usage::

        ema = EMAWrapper(model, decay=0.999)
        for batch in loader:
            loss = train_step(model, batch)
            ema.update()
        with ema.average_parameters():
            metrics = evaluate(model, val_loader)

    Args:
        model : The model whose parameters will be tracked.
        decay : EMA decay coefficient (e.g. 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model  = model
        self.decay  = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update EMA shadow weights after each optimiser step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self) -> None:
        """Replace model weights with EMA shadow for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original (training) weights."""
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    class _Context:
        def __init__(self, ema: "EMAWrapper") -> None:
            self._ema = ema
        def __enter__(self) -> None:
            self._ema.apply_shadow()
        def __exit__(self, *_: Any) -> None:
            self._ema.restore()

    def average_parameters(self) -> "_Context":
        """Context manager: temporarily swap in EMA weights."""
        return self._Context(self)


# =============================================================================
# 6.  Gradient Monitor
# =============================================================================

class GradMonitor:
    """
    Per-layer gradient norm logger.

    Attaches backward hooks to all ``nn.Linear`` layers in a model and
    accumulates per-step gradient norms.  Call ``report()`` at the end
    of each epoch to log summary statistics and reset counters.

    Args:
        model  : The model to monitor.
        log_fn : Callable that receives a log string (default: logger.debug).
    """

    def __init__(
        self,
        model: nn.Module,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._log = log_fn or logger.debug
        self._norms: Dict[str, List[float]] = {}
        self._handles = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._norms[name] = []
                handle = module.register_full_backward_hook(
                    self._make_hook(name)
                )
                self._handles.append(handle)

    def _make_hook(self, name: str) -> Callable:
        def hook(
            module: nn.Module,
            grad_input: Tuple,
            grad_output: Tuple,
        ) -> None:
            if grad_output[0] is not None:
                norm = grad_output[0].detach().norm().item()
                self._norms[name].append(norm)
        return hook

    def report(self, epoch: int) -> Dict[str, float]:
        """
        Log and return mean gradient norm per layer.
        Resets counters after reporting.
        """
        summary: Dict[str, float] = {}
        for name, norms in self._norms.items():
            if norms:
                mean_norm = sum(norms) / len(norms)
                summary[name] = mean_norm
                self._log(f"Epoch {epoch:03d} | grad_norm | {name}: {mean_norm:.4e}")
            self._norms[name] = []
        return summary

    def remove_hooks(self) -> None:
        """Detach all backward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# =============================================================================
# 7.  Dataset
# =============================================================================

class SGNODataset(Dataset):
    """
    PyTorch Dataset that wraps REAL FOLD ONE outputs.

    Two modes:

    ``mode="protein"``
        Each sample is a dict with keys:
            seq_feats        : (N, node_in_dim)
            init_coords      : (N, 3)
            final_coords     : (N, 3)
            ddg              : scalar
            sigma            : (N, 1)

    ``mode="phase_field"``
        Each sample is a dict with keys:
            u_init   : (Nx, Ny, Nz)
            u_future : (Nx, Ny, Nz)
            sigma_3d : (Nx, Ny, Nz)

    Args:
        samples : List of dicts as described above.
        mode    : ``"protein"`` or ``"phase_field"``.
    """

    def __init__(
        self,
        samples: List[Dict[str, torch.Tensor]],
        mode: str = "protein",
    ) -> None:
        assert mode in {"protein", "phase_field"}, \
            f"mode must be 'protein' or 'phase_field'; got {mode!r}"
        self.samples = samples
        self.mode    = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

    @staticmethod
    def collate_protein(
        batch: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Return a list (not a stacked tensor) because protein graphs
        have variable node counts — stacking would require padding.
        """
        return batch

    @staticmethod
    def collate_phase_field(
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Stack fixed-size phase-field grids into batch tensors."""
        return {
            "u_init":   torch.stack([s["u_init"]   for s in batch]),
            "u_future": torch.stack([s["u_future"] for s in batch]),
            "sigma_3d": torch.stack([s["sigma_3d"] for s in batch]),
        }


# =============================================================================
# 8.  SGNOTrainer  (production)
# =============================================================================

class SGNOTrainer:
    """
    Production training loop for StructuralGNOFold.

    Features:
        • Separate LR groups for backbone and output heads
        • OneCycleLR scheduler with linear warmup
        • EMA weight averaging (``ema_decay`` from config)
        • Gradient norm clipping and per-layer monitoring
        • Physics-informed Lyapunov loss (phase-field mode)
        • CH mass-conservation penalty
        • Checkpoint save / resume

    Args:
        model      : StructuralGNOFold instance.
        cfg        : SGNOConfig (uses model.cfg if None).
        device     : Compute device.
        ch_solver  : StructuralCahnHilliard3D for physics-informed loss
                     (optional; phase-field mode only).
        steps_per_epoch : Used to configure OneCycleLR.
    """

    def __init__(
        self,
        model:             StructuralGNOFold,
        cfg:               Optional[SGNOConfig] = None,
        device:            Optional[torch.device] = None,
        ch_solver:         Optional[Any] = None,
        steps_per_epoch:   int = 100,
    ) -> None:
        self.cfg     = cfg or model.cfg
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = model.to(self.device)
        self.ch_solver = ch_solver

        # Separate LR groups: backbone layers vs output heads
        backbone_params = list(self.model.layers.parameters()) + \
                          list(self.model.node_embed.parameters()) + \
                          list(self.model.edge_embed.parameters()) + \
                          list(self.model.grid_node_embed.parameters())
        head_params = (
            list(self.model.coord_head.parameters()) +
            list(self.model.ddg_head.parameters()) +
            list(self.model.attn_pool.parameters()) +
            list(self.model.phase_head.parameters())
        )

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.cfg.lr_backbone},
                {"params": head_params,     "lr": self.cfg.lr_heads},
            ],
            weight_decay=self.cfg.weight_decay,
        )

        # OneCycleLR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[self.cfg.lr_backbone * 10, self.cfg.lr_heads * 10],
            total_steps=self.cfg.max_epochs * steps_per_epoch,
            pct_start=self.cfg.warmup_epochs / max(self.cfg.max_epochs, 1),
            anneal_strategy="cos",
        )

        self.ema     = EMAWrapper(self.model, decay=self.cfg.ema_decay)
        self.monitor = GradMonitor(self.model)
        self._epoch  = 0

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        logger.info(
            "SGNOTrainer | device=%s | backbone_lr=%.2e | head_lr=%.2e | EMA=%.4f",
            self.device, self.cfg.lr_backbone, self.cfg.lr_heads, self.cfg.ema_decay,
        )

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def train_step_protein(
        self,
        seq_feats:         torch.Tensor,   # (N, node_in_dim)
        init_coords:       torch.Tensor,   # (N, 3)
        true_final_coords: torch.Tensor,   # (N, 3)
        true_ddg:          torch.Tensor,   # scalar
        sigma:             torch.Tensor,   # (N, 1)
    ) -> Dict[str, float]:
        """
        One training step in protein mode.

        Loss:
            L = MSE(coords) + λ_ddg * MSE(ddg)

        Returns:
            dict with keys ``total``, ``coords``, ``ddg``.
        """
        self.model.train()
        self.optimizer.zero_grad()

        seq_feats         = seq_feats.to(self.device)
        init_coords       = init_coords.to(self.device)
        true_final_coords = true_final_coords.to(self.device)
        true_ddg          = true_ddg.to(self.device)
        sigma             = sigma.to(self.device)

        pred_coords, pred_ddg = self.model(seq_feats, init_coords, sigma)

        loss_coords = F.mse_loss(pred_coords, true_final_coords)
        loss_ddg    = F.mse_loss(pred_ddg.squeeze(), true_ddg.squeeze())
        total_loss  = loss_coords + self.cfg.lambda_ddg * loss_ddg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()

        return {
            "total":  total_loss.item(),
            "coords": loss_coords.item(),
            "ddg":    loss_ddg.item(),
        }

    def train_step_phase_field(
        self,
        u_init:       torch.Tensor,   # (Nx, Ny, Nz)
        u_true_future: torch.Tensor,  # (Nx, Ny, Nz)
        sigma_3d:     torch.Tensor,   # (Nx, Ny, Nz)
    ) -> Dict[str, float]:
        """
        One training step in phase-field mode.

        Loss:
            L = MSE(u_pred, u_true)
              + λ_physics * ReLU(E_pred - E_init)    [Lyapunov penalty]
              + λ_mass    * |mass(u_pred) - mass(u_init)|  [conservation]

        Returns:
            dict with keys ``total``, ``data``, ``physics``, ``mass``.
        """
        self.model.train()
        self.optimizer.zero_grad()

        u_init        = u_init.to(self.device)
        u_true_future = u_true_future.to(self.device)
        sigma_3d      = sigma_3d.to(self.device)

        u_pred = self.model.forward_phase_field(u_init, sigma_3d)

        loss_data = F.mse_loss(u_pred, u_true_future)

        loss_physics = torch.zeros(1, device=self.device)
        loss_mass    = torch.zeros(1, device=self.device)

        if self.ch_solver is not None:
            try:
                E_pred = self.ch_solver.structural_energy(u_pred,   sigma_3d)
                E_init = self.ch_solver.structural_energy(u_init,   sigma_3d)
                loss_physics = F.relu(E_pred - E_init)

                mass_pred = u_pred.sum()
                mass_init = u_init.sum()
                loss_mass = (mass_pred - mass_init).abs()
            except Exception as exc:
                logger.warning("Physics loss failed: %s", exc)

        total_loss = (
            loss_data
            + self.cfg.lambda_physics * loss_physics
            + self.cfg.lambda_mass    * loss_mass
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()

        return {
            "total":   total_loss.item(),
            "data":    loss_data.item(),
            "physics": loss_physics.item(),
            "mass":    loss_mass.item(),
        }

    # ------------------------------------------------------------------
    # Epoch-level loop helpers
    # ------------------------------------------------------------------

    def run_epoch_protein(
        self,
        loader: DataLoader,
        epoch:  int,
    ) -> Dict[str, float]:
        """
        Run one full epoch over a protein DataLoader.

        Args:
            loader : DataLoader yielding lists of per-sample dicts
                     (use ``SGNODataset.collate_protein``).
            epoch  : Current epoch index (for logging).
        Returns:
            Mean loss dict over all batches.
        """
        totals: Dict[str, float] = {"total": 0.0, "coords": 0.0, "ddg": 0.0}
        n = 0
        for batch in loader:
            for sample in batch:
                metrics = self.train_step_protein(
                    sample["seq_feats"],
                    sample["init_coords"],
                    sample["final_coords"],
                    sample["ddg"],
                    sample["sigma"],
                )
                for k in totals:
                    totals[k] += metrics[k]
                n += 1
        means = {k: v / max(n, 1) for k, v in totals.items()}
        self.monitor.report(epoch)
        logger.info("Epoch %03d [protein] %s", epoch, means)
        return means

    def run_epoch_phase_field(
        self,
        loader: DataLoader,
        epoch:  int,
    ) -> Dict[str, float]:
        """
        Run one full epoch over a phase-field DataLoader.

        Args:
            loader : DataLoader yielding batched grids
                     (use ``SGNODataset.collate_phase_field``).
            epoch  : Current epoch index.
        Returns:
            Mean loss dict over all batches.
        """
        totals: Dict[str, float] = {
            "total": 0.0, "data": 0.0, "physics": 0.0, "mass": 0.0
        }
        n = 0
        for batch in loader:
            B = batch["u_init"].shape[0]
            for i in range(B):
                metrics = self.train_step_phase_field(
                    batch["u_init"][i],
                    batch["u_future"][i],
                    batch["sigma_3d"][i],
                )
                for k in totals:
                    totals[k] += metrics[k]
                n += 1
        means = {k: v / max(n, 1) for k, v in totals.items()}
        self.monitor.report(epoch)
        logger.info("Epoch %03d [phase_field] %s", epoch, means)
        return means

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """
        Save model weights, EMA shadow, optimiser, scheduler, and config.

        Args:
            path : Output file path.  If None, auto-named by epoch.
        Returns:
            Absolute path of saved checkpoint.
        """
        if path is None:
            path = os.path.join(
                self.cfg.checkpoint_dir,
                f"sgno_epoch_{self._epoch:04d}.pt",
            )
        state = {
            "epoch":          self._epoch,
            "sgno_version":   SGNO_VERSION,
            "fold_version":   FOLD_VERSION,
            "model_state":    self.model.state_dict(),
            "ema_shadow":     self.ema.shadow,
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
            "config":         self.cfg.to_dict(),
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        torch.save(state, path)
        logger.info("Checkpoint saved → %s", os.path.abspath(path))
        return os.path.abspath(path)

    def load_checkpoint(self, path: str) -> int:
        """
        Resume training from a checkpoint.

        Args:
            path : Path to a ``.pt`` checkpoint saved by ``save_checkpoint``.
        Returns:
            Epoch number stored in the checkpoint.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.ema.shadow = {
            k: v.to(self.device) for k, v in state["ema_shadow"].items()
        }
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self._epoch = state["epoch"]
        logger.info(
            "Checkpoint loaded ← %s (epoch %d, sgno_v%s, fold_v%s)",
            path, self._epoch, state.get("sgno_version"), state.get("fold_version"),
        )
        return self._epoch


# =============================================================================
# 9.  SGNOEvaluator
# =============================================================================

class SGNOEvaluator:
    """
    Evaluation metrics for StructuralGNOFold.

    Protein mode metrics:
        • RMSD (Å) between predicted and true Cα positions.
        • Pearson-r between predicted and true ΔΔG values.

    Phase-field mode metrics:
        • MSE of u_pred vs u_true.
        • CH energy monotonicity: fraction of predictions where
          E(u_pred) ≤ E(u_init)  (should approach 1.0 after training).
        • Mass conservation error: |mass_pred − mass_init| / |mass_init|.

    Args:
        model      : StructuralGNOFold (evaluated in EMA mode if ema given).
        ema        : EMAWrapper instance (optional).
        device     : Compute device.
        ch_solver  : StructuralCahnHilliard3D for energy metrics.
    """

    def __init__(
        self,
        model:     StructuralGNOFold,
        ema:       Optional[EMAWrapper] = None,
        device:    Optional[torch.device] = None,
        ch_solver: Optional[Any] = None,
    ) -> None:
        self.model     = model
        self.ema       = ema
        self.device    = device or torch.device("cpu")
        self.ch_solver = ch_solver

    @torch.no_grad()
    def evaluate_protein(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Evaluate protein mode over a list of samples.

        Returns:
            dict with keys ``rmsd_mean``, ``rmsd_std``, ``pearson_r_ddg``.
        """
        ctx = self.ema.average_parameters() if self.ema else _null_context()
        self.model.eval()
        rmsds, pred_ddgs, true_ddgs = [], [], []

        with ctx:
            for s in samples:
                seq_feats   = s["seq_feats"].to(self.device)
                init_coords = s["init_coords"].to(self.device)
                true_coords = s["final_coords"].to(self.device)
                sigma       = s["sigma"].to(self.device)
                ddg_true    = s["ddg"].item()

                pred_coords, pred_ddg = self.model(seq_feats, init_coords, sigma)

                diff  = pred_coords - true_coords
                rmsd  = diff.pow(2).sum(-1).mean().sqrt().item()
                rmsds.append(rmsd)
                pred_ddgs.append(pred_ddg.item())
                true_ddgs.append(ddg_true)

        rmsd_t = torch.tensor(rmsds)
        pr     = _pearson_r(
            torch.tensor(pred_ddgs, dtype=torch.float32),
            torch.tensor(true_ddgs, dtype=torch.float32),
        )
        return {
            "rmsd_mean":    rmsd_t.mean().item(),
            "rmsd_std":     rmsd_t.std().item(),
            "pearson_r_ddg": pr,
        }

    @torch.no_grad()
    def evaluate_phase_field(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Evaluate phase-field mode over a list of samples.

        Returns:
            dict with keys ``mse``, ``energy_monotone_frac``,
            ``mass_rel_error_mean``.
        """
        ctx = self.ema.average_parameters() if self.ema else _null_context()
        self.model.eval()
        mses, mono, mass_errs = [], [], []

        with ctx:
            for s in samples:
                u_init   = s["u_init"].to(self.device)
                u_future = s["u_future"].to(self.device)
                sigma_3d = s["sigma_3d"].to(self.device)

                u_pred = self.model.forward_phase_field(u_init, sigma_3d)
                mses.append(F.mse_loss(u_pred, u_future).item())

                if self.ch_solver is not None:
                    try:
                        E_pred = self.ch_solver.structural_energy(u_pred, sigma_3d).item()
                        E_init = self.ch_solver.structural_energy(u_init, sigma_3d).item()
                        mono.append(float(E_pred <= E_init))
                        m_pred = u_pred.sum().item()
                        m_init = u_init.sum().item()
                        mass_errs.append(
                            abs(m_pred - m_init) / (abs(m_init) + 1e-12)
                        )
                    except Exception:
                        pass

        result: Dict[str, float] = {"mse": float(sum(mses) / max(len(mses), 1))}
        if mono:
            result["energy_monotone_frac"] = sum(mono) / len(mono)
            result["mass_rel_error_mean"]  = sum(mass_errs) / len(mass_errs)
        return result


# =============================================================================
# 10.  Utilities
# =============================================================================

def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient between two 1-D tensors."""
    if x.numel() < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    r  = (xm * ym).sum() / (xm.norm() * ym.norm() + 1e-12)
    return r.item()


class _null_context:
    """No-op context manager (replaces EMA context when EMA is absent)."""
    def __enter__(self) -> None: ...
    def __exit__(self, *_: Any) -> None: ...


def build_trainer_from_ecosystem(
    cfg:            Optional[SGNOConfig] = None,
    device_str:     str = "auto",
    ch_cfg:         Optional[Any] = None,
    steps_per_epoch: int = 100,
) -> SGNOTrainer:
    """
    Convenience factory: build a ready-to-use SGNOTrainer with a
    StructuralCahnHilliard3D physics solver attached (if available).

    Args:
        cfg            : SGNOConfig (defaults used if None).
        device_str     : ``"auto"``, ``"cuda"``, or ``"cpu"``.
        ch_cfg         : CahnHilliardConfig (defaults used if None and CH available).
        steps_per_epoch: Passed to SGNOTrainer for scheduler setup.
    Returns:
        A configured SGNOTrainer ready for ``.train_step_*`` calls.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model     = StructuralGNOFold(cfg)
    ch_solver = None

    if _HAS_CH:
        ch_cfg    = ch_cfg or CahnHilliardConfig(
            dx=1.0, epsilon=1.5, dt=1e-5, laplacian="conv3d"
        )
        ch_solver = StructuralCahnHilliard3D(ch_cfg).to(device)
        logger.info("Physics engine: StructuralCahnHilliard3D attached.")
    else:
        logger.info("Physics engine: not available (standalone mode).")

    return SGNOTrainer(
        model=model,
        cfg=cfg,
        device=device,
        ch_solver=ch_solver,
        steps_per_epoch=steps_per_epoch,
    )


# =============================================================================
# 11.  Self-test
# =============================================================================

if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Structural GNO Fold v3 — Production Integration Tests")
    print(f"  FOLD_VERSION = {FOLD_VERSION} | SGNO_VERSION = {SGNO_VERSION}")
    print(f"  Device: {_device}")
    print("=" * 70)

    _cfg   = SGNOConfig(hidden_dim=64, num_layers=3)
    _model = StructuralGNOFold(_cfg).to(_device)
    _ema   = EMAWrapper(_model, decay=_cfg.ema_decay)

    # ── Test 1: Protein mode ────────────────────────────────────────────
    N = 40
    seq_f  = torch.randn(N, 20,  device=_device)
    coords = torch.randn(N, 3,   device=_device)
    sigma  = torch.ones(N,  1,   device=_device) * 1.2

    pred_c, pred_ddg = _model(seq_f, coords, sigma)
    assert pred_c.shape  == (N, 3), f"Expected ({N},3) got {pred_c.shape}"
    assert pred_ddg.shape == (1,),  f"Expected (1,) got {pred_ddg.shape}"
    print(f"[PASS] Protein mode  → coords {pred_c.shape}, ddg {pred_ddg.shape}")

    # ── Test 2: Phase-field mode ────────────────────────────────────────
    G = 12
    u_init   = torch.rand(G, G, G, device=_device) * 0.2 - 0.1
    sigma_3d = torch.ones(G, G, G, device=_device)

    u_pred = _model.forward_phase_field(u_init, sigma_3d)
    assert u_pred.shape == (G, G, G), f"Expected ({G},{G},{G}) got {u_pred.shape}"
    print(f"[PASS] Phase-field mode → u {u_pred.shape}")

    # ── Test 3: EMA context manager ─────────────────────────────────────
    _ema.update()
    with _ema.average_parameters():
        u_ema = _model.forward_phase_field(u_init, sigma_3d)
    assert u_ema.shape == (G, G, G)
    print(f"[PASS] EMA inference mode → u {u_ema.shape}")

    # ── Test 4: Trainer + checkpointing ────────────────────────────────
    _trainer = build_trainer_from_ecosystem(cfg=_cfg, steps_per_epoch=10)
    m = _trainer.train_step_protein(seq_f, coords, coords, torch.tensor(0.5, device=_device), sigma)
    assert "total" in m
    print(f"[PASS] Trainer protein step → losses {m}")

    m2 = _trainer.train_step_phase_field(u_init, u_init, sigma_3d)
    assert "total" in m2
    print(f"[PASS] Trainer phase-field step → losses {m2}")

    _ckpt = _trainer.save_checkpoint()
    _trainer.load_checkpoint(_ckpt)
    print(f"[PASS] Checkpoint save/load → {_ckpt}")

    # ── Test 5: Evaluator ───────────────────────────────────────────────
    _sample_p = {
        "seq_feats":   seq_f,
        "init_coords": coords,
        "final_coords": coords + 0.01,
        "ddg":         torch.tensor(-0.3, device=_device),
        "sigma":       sigma,
    }
    _eval = SGNOEvaluator(_model, ema=_ema, device=_device)
    res   = _eval.evaluate_protein([_sample_p])
    print(f"[PASS] Evaluator protein → {res}")

    if _HAS_CH:
        _ch_cfg    = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-5, laplacian="conv3d")
        _ch_solver = StructuralCahnHilliard3D(_ch_cfg).to(_device)
        _eval_cf   = SGNOEvaluator(_model, ema=_ema, device=_device, ch_solver=_ch_solver)
        _sample_cf = {"u_init": u_init, "u_future": u_init * 0.95, "sigma_3d": sigma_3d}
        res_cf     = _eval_cf.evaluate_phase_field([_sample_cf])
        print(f"[PASS] Evaluator phase-field → {res_cf}")
    else:
        print("[SKIP] Phase-field evaluator (CH3D not available)")

    print("=" * 70)
    print("  All tests passed.")
    print("=" * 70)
