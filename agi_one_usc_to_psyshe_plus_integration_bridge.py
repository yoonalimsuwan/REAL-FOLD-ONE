# =============================================================================
# USC to AGI ONE Psyche Plus Integration Bridge
# Multi-Framework Orchestrator for Structural Calculus Layer
# =============================================================================
#
# Developer  : PAI AND Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788
# GitHub     : https://github.com/yoonalimsuwan
# Email      : msps4u@gmail.com
#
# =============================================================================
# Description:
# This module provides a deterministic bridge between the macroscopic geometry 
# outputs of the Universal Structural Contraction (USC) layer across various 
# frameworks (PyTorch, JAX, MLX, PaddlePaddle, MindSpore) and the central AGI ONE 
# PyTorch ecosystem. It ensures seamless tensor conversion and strictly routes 
# the topological signature into the Psyche Plus speculative-axiom evolution 
# layer and the Psyche Executive layer.
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Any, Tuple, Union, Dict

# Set up dedicated logger for the USC Bridge
logging.basicConfig(level=logging.INFO, format="%(asctime)s  [USC_BRIDGE]  %(levelname)s  %(message)s")
logger = logging.getLogger("USC_AGI_BRIDGE")

class USCTensorConverter:
    """
    Safely and deterministically converts multi-framework macroscopic geometry 
    tensors (PyTorch, JAX, MLX, PaddlePaddle, MindSpore) into PyTorch tensors.
    """
    @staticmethod
    def to_pytorch(usc_tensor: Any, device: torch.device) -> torch.Tensor:
        # 1. Handle PyTorch Dictionary/ModelOutput returns
        if isinstance(usc_tensor, dict):
            for key in ['macroscopic_geometry', 'last_hidden_state', 'logits', 'output', 'out']:
                if key in usc_tensor:
                    usc_tensor = usc_tensor[key]
                    break
            else:
                usc_tensor = next(iter(usc_tensor.values()))

        # 2. Handle Tuple or List outputs from PyTorch modules
        if isinstance(usc_tensor, (tuple, list)):
            usc_tensor = usc_tensor[0]

        # 3. Native PyTorch Tensor
        if isinstance(usc_tensor, torch.Tensor):
            return usc_tensor.to(device)

        tensor_type = str(type(usc_tensor)).lower()
        
        try:
            # JAX / Flax Array
            if 'jax' in tensor_type:
                import jax
                np_array = np.asarray(usc_tensor)
                return torch.from_numpy(np_array).to(device)
            
            # Apple MLX Array
            elif 'mlx' in tensor_type:
                np_array = np.array(usc_tensor)
                return torch.from_numpy(np_array).to(device)
            
            # PaddlePaddle Tensor
            elif 'paddle' in tensor_type:
                np_array = usc_tensor.numpy()
                return torch.from_numpy(np_array).to(device)
            
            # MindSpore Tensor
            elif 'mindspore' in tensor_type:
                np_array = usc_tensor.asnumpy()
                return torch.from_numpy(np_array).to(device)
            
            # Standard NumPy or string match fallback
            elif 'numpy' in tensor_type:
                return torch.from_numpy(usc_tensor).to(device)
            elif 'torch' in tensor_type:
                return usc_tensor.to(device)
            
            else:
                logger.warning(f"Unrecognized tensor type '{tensor_type}'. Attempting default conversion.")
                if hasattr(usc_tensor, '__array__'):
                    return torch.from_numpy(np.asarray(usc_tensor)).to(device)
                return torch.tensor(usc_tensor, device=device)
                
        except Exception as e:
            logger.error(f"Failed to convert USC tensor to PyTorch: {e}")
            raise RuntimeError(f"Tensor conversion failed for deterministic projection: {e}")


class USCPsycheIntegrationModule(nn.Module):
    """
    Production-grade projection and routing module. Aligns the external/native USC 
    structural calculus dimensions with the internal AGI ONE latent dimensions, 
    then processes the unified state through Psyche Plus and Psyche Executive.
    """
    def __init__(
        self, 
        usc_d_model: int, 
        agi_latent_dim: int, 
        quality_score_override: float = 1.0
    ):
        super().__init__()
        self.usc_d_model = usc_d_model
        self.agi_latent_dim = agi_latent_dim
        self.quality_score_override = quality_score_override
        
        # Deterministic macroscopic projection layer
        self.structural_projection = nn.Sequential(
            nn.Linear(usc_d_model, agi_latent_dim),
            nn.LayerNorm(agi_latent_dim),
            nn.Tanh() # Bounding chaotic fluctuations during modality merge
        )
        
        logger.info(f"Initialized USC-Psyche Bridge: mapping d_model({usc_d_model}) -> agi_latent({agi_latent_dim})")

    def forward(
        self, 
        raw_usc_output: Any, 
        agi_workspace_latent: torch.Tensor, 
        agi_core_instance: Any
    ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            raw_usc_output: Output from PyTorch, JAX, MLX, Paddle, or MindSpore USC module.
            agi_workspace_latent: The current state of the AGI ONE workspace.
            agi_core_instance: Reference to the central AGI ONE class (AGI_ONE_v2) 
                               to access Psyche layers.
                               
        Returns:
            Tuple containing the safe executive action and the updated workspace latent.
        """
        device = agi_workspace_latent.device
        
        # 1. Cross-Platform & Native Tensor Normalization
        pt_usc_tensor = USCTensorConverter.to_pytorch(raw_usc_output, device)
        
        # Optional: Collapse sequence length if USC returned (B, L, D) and AGI expects (B, D)
        if pt_usc_tensor.dim() == 3 and agi_workspace_latent.dim() == 2:
            # Deterministic mean-pooling to preserve macroscopic geometry
            pt_usc_tensor = pt_usc_tensor.mean(dim=1) 
            
        # 2. Project into AGI ONE Latent Space
        aligned_usc_geometry = self.structural_projection(pt_usc_tensor)
        
        # 3. Fuse structural state into the global workspace
        fused_workspace = agi_workspace_latent + aligned_usc_geometry
        
        # 4. Route through Psyche Plus (Speculative Axiom Evolution)
        if hasattr(agi_core_instance, "config") and getattr(agi_core_instance.config, "use_psyche_plus", False):
            if hasattr(agi_core_instance, "psyche_plus"):
                psyche_plus_out = agi_core_instance.psyche_plus(
                    fused_workspace, 
                    quality_score=self.quality_score_override
                )
                fused_workspace = fused_workspace + psyche_plus_out
            else:
                logger.warning("Config specified use_psyche_plus=True, but PsychePlus module is missing.")
                
        # 5. Route through Psyche Executive Layer (Id, Ego, Superego Safety Gating)
        executive_action = None
        if hasattr(agi_core_instance, "psyche_executive"):
            executive_action, safe_state = agi_core_instance.psyche_executive(fused_workspace)
            fused_workspace = safe_state
            
        return executive_action, fused_workspace
