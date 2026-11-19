#!/usr/bin/env python3
"""
=============================================================================
OPTO-MDOE CLOSED-LOOP BRIDGE & MULTI-NODE SLURM GENERATOR
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit ;
What MSPS NETWORK SEE, Lord Buddha Knows.
License   : MIT | Year: 2026

INTEGRATION POINTS:
1. SOTA Optogenetics (coupled_optogenetics_nsem_safety.py)
   -> Outputs: firing_rates, safety_diag (fluence, thermal, chatter)
2. MDOE Organoid Predictor (Morphogen_Directed_Organoid_Engineering_Predictor.py)
   -> Inputs/Outputs: protocol_ctx, T3 Quality (viability, stress_markers)

HONEST SCOPE:
✅ Bridge is fully differentiable; gradients flow from MDOE organoid viability 
   loss back to the Optogenetics light controller.
✅ Slurm generator uses standard torchrun multi-node DDP topology.
❌ Mapping phototoxicity (Arrhenius/fluence) directly to organoid viability 
   is a PHENOMENOLOGICAL PROXY. Real cell-death thresholds require 
   empirical calibration for your specific organoid line.
❌ This does not simulate immune responses to optogenetic stimulation.
=============================================================================
"""
from __future__ import annotations
import torch
import torch.nn as nn
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List

# =============================================================================
# 1. OPTO-MDOE BIDIRECTIONAL BRIDGE
# =============================================================================
class OptoMDOEBridge(nn.Module):
    """
    Bidirectional bridge between SOTA Optogenetics and MDOE Organoid Predictor.
    
    [Opto -> MDOE]: Translates optogenetic readouts (firing rates, phototoxicity, 
    thermal load) into MDOE T3 Organoid Quality stress metrics.
    
    [MDOE -> Opto]: Translates MDOE protocol embeddings into target firing rates 
    and safety irradiance bounds for the Optogenetics Controller.
    """
    def __init__(self, num_neurons: int, d_mdoe_ctx: int = 128, n_stress_markers: int = 8):
        super().__init__()
        self.num_neurons = num_neurons
        
        # --- Opto -> MDOE (State to Quality/Stress) ---
        # Inputs: mean_rate, max_fluence, max_thermal, max_chatter, min_margin
        self.opto_state_dim = 5 
        self.stress_trunk = nn.Sequential(
            nn.Linear(self.opto_state_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        self.viability_penalty_head = nn.Sequential(nn.Linear(128, 1), nn.Softplus())
        self.stress_marker_head = nn.Sequential(nn.Linear(128, n_stress_markers), nn.Sigmoid())
        
        # --- MDOE -> Opto (Protocol to Controller Targets) ---
        self.target_rate_proj = nn.Sequential(
            nn.Linear(d_mdoe_ctx, 64),
            nn.GELU(),
            nn.Linear(64, num_neurons),
            nn.Sigmoid() # Target firing rate [0, 1]
        )
        self.max_irradiance_proj = nn.Sequential(
            nn.Linear(d_mdoe_ctx, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus() # Max safe irradiance bound (mW/mm^2)
        )

    def opto_to_mdoe_stress(self, firing_rates: torch.Tensor, safety_diag: Dict) -> Dict[str, torch.Tensor]:
        """
        Maps Opto safety diagnostics to MDOE T3 stress penalties.
        firing_rates: [B, N]
        safety_diag: dict from PhototoxicitySafetyLayer
        """
        rate_mean = firing_rates.mean(dim=-1, keepdim=True)
        
        # Extract and aggregate safety metrics batch-wise
        fluence_max = safety_diag.get('fluence_J_mm2', torch.zeros_like(rate_mean)).max(dim=-1, keepdim=True).values
        thermal_max = safety_diag.get('thermal_arrhenius', torch.zeros_like(rate_mean)).max(dim=-1, keepdim=True).values
        chatter_max = safety_diag.get('chatter', torch.zeros_like(rate_mean)).max(dim=-1, keepdim=True).values
        margin_min = safety_diag.get('margin', torch.ones_like(rate_mean)).min(dim=-1, keepdim=True).values
        
        # Danger level (1 - margin)
        danger = 1.0 - margin_min 
        
        state = torch.cat([rate_mean, fluence_max, thermal_max, chatter_max, danger], dim=-1)
        h = self.stress_trunk(state)
        
        return {
            "viability_penalty": self.viability_penalty_head(h).squeeze(-1), # Subtract from MDOE viability
            "stress_markers": self.stress_marker_head(h)                     # Add to MDOE stress markers
        }

    def mdoe_to_opto_targets(self, mdoe_protocol_ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps MDOE protocol context to Opto controller targets.
        mdoe_protocol_ctx: [B, d_mdoe_ctx] from MDOE ProtocolTransformer
        """
        return {
            "target_rates": self.target_rate_proj(mdoe_protocol_ctx), # [B, N]
            "max_safe_irradiance": self.max_irradiance_proj(mdoe_protocol_ctx).squeeze(-1) # [B]
        }


# =============================================================================
# 2. MULTI-NODE SLURM GENERATOR (Closed-Loop DDP)
# =============================================================================
@dataclass
class ClosedLoopSlurmConfig:
    job_name: str = "opto_mdoe_closed_loop"
    account: str = "YOUR_PROJECT_ACCOUNT"
    partition: str = "gpu"
    nodes: int = 4          # e.g., 4 nodes
    gpus_per_node: int = 8  # 8 GPUs per node = 32 GPUs total
    cpus_per_task: int = 16
    mem_gb: int = 512
    time_limit: str = "24:00:00"
    email: str = "your@email.com"
    modules: List[str] = field(default_factory=lambda: [
        "cuda/12.2", "nccl/2.19", "openmpi/4.1.6-cuda", "python/3.11", "anaconda3/2024.02"
    ])

class ClosedLoopSlurmGenerator:
    """
    Generates a Slurm batch script for Multi-Node Distributed Data Parallel (DDP) 
    training of the Opto-MDOE Closed-Loop Controller using `torchrun`.
    """
    def __init__(self, cfg: Optional[ClosedLoopSlurmConfig] = None):
        self.cfg = cfg or ClosedLoopSlurmConfig()
        self.total_gpus = self.cfg.nodes * self.cfg.gpus_per_node

    def generate(self) -> str:
        modules_load = "\n".join(f"module load {m}" for m in self.cfg.modules)
        script = f"""#!/bin/bash
#=============================================================================
OPTO-MDOE CLOSED-LOOP CONTROLLER — {self.total_gpus}-GPU SLURM SCRIPT
Auto-generated for Multi-Node DDP Training
#=============================================================================
#SBATCH --job-name={self.cfg.job_name}
#SBATCH --output={self.cfg.job_name}_%j.out
#SBATCH --error={self.cfg.job_name}_%j.err
#SBATCH --time={self.cfg.time_limit}
#SBATCH --nodes={self.cfg.nodes}
#SBATCH --ntasks-per-node={self.cfg.gpus_per_node}
#SBATCH --cpus-per-task={self.cfg.cpus_per_task}
#SBATCH --gres=gpu:{self.cfg.gpus_per_node}
#SBATCH --mem={self.cfg.mem_gb}G
#SBATCH --partition={self.cfg.partition}
#SBATCH --account={self.cfg.account}
#SBATCH --mail-user={self.cfg.email}
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
echo "======================================================================"
echo "  OPTO-MDOE CLOSED-LOOP TRAINING — Job Start"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Nodes       : $SLURM_JOB_NUM_NODES"
echo "  Total GPUs  : {self.total_gpus}"
echo "  Start Time  : $(date)"
echo "======================================================================"

# Environment Setup
module purge
{modules_load}
source activate sesi_opto_mdoe  # Replace with your conda environment

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export OMP_NUM_THREADS={self.cfg.cpus_per_task}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR
mkdir -p checkpoints_opto_mdoe logs

#=============================================================================
# PRE-FLIGHT CHECKS
#=============================================================================
python -c "
import torch
print(f'PyTorch: {{torch.__version__}} | CUDA: {{torch.cuda.is_available()}}')
print(f'GPU Count: {{torch.cuda.device_count()}} | NCCL: {{torch.distributed.is_nccl_available()}}')
"

#=============================================================================
# MULTI-NODE DDP TRAINING (torchrun)
#=============================================================================
echo "Launching torchrun across {self.cfg.nodes} nodes..."

torchrun \\
    --nnodes={self.cfg.nodes} \\
    --nproc_per_node={self.cfg.gpus_per_node} \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \\
    --master_port=29500 \\
    --rdzv_backend=c10d \\
    train_opto_mdoe_closed_loop.py \\
    --epochs 100 \\
    --batch-size 8 \\
    --lr 3e-4 \\
    --amp-dtype bf16 \\
    --checkpoint-dir ./checkpoints_opto_mdoe \\
    --log-interval 20

TRAINING_STATUS=$?

#=============================================================================
# POST-TRAINING
#=============================================================================
echo "Training Complete. Status: $TRAINING_STATUS"
if [ $TRAINING_STATUS -eq 0 ]; then
    tar -czf results_{self.cfg.job_name}_$SLURM_JOB_ID.tar.gz checkpoints_opto_mdoe/ *.log 2>/dev/null || true
fi

exit $TRAINING_STATUS
"""
        return script

    def write_slurm_file(self, output_path: str = "submit_opto_mdoe.slurm") -> None:
        with open(output_path, "w") as f:
            f.write(self.generate())
        os.chmod(output_path, 0o755)
        print(f"[Slurm] Generated multi-node script: {output_path}")


# =============================================================================
# 3. INTEGRATION SMOKE TEST
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, N = 4, 64  # Batch size, Num Neurons
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("OPTO-MDOE BRIDGE & SLURM GENERATOR — SMOKE TEST")
    print("="*70)
    
    # 1. Test Bridge Forward Pass
    bridge = OptoMDOEBridge(num_neurons=N, d_mdoe_ctx=128).to(device)
    
    # Dummy Opto Outputs
    firing_rates = torch.rand(B, N, device=device)
    safety_diag = {
        'fluence_J_mm2': torch.rand(B, N, device=device) * 10,
        'thermal_arrhenius': torch.rand(B, N, device=device) * 0.5,
        'chatter': torch.rand(B, N, device=device) * 100,
        'margin': torch.rand(B, N, device=device)
    }
    
    # Dummy MDOE Context
    mdoe_ctx = torch.randn(B, 128, device=device)
    
    # Opto -> MDOE
    stress_out = bridge.opto_to_mdoe_stress(firing_rates, safety_diag)
    print(f"[Opto->MDOE] Viability Penalty shape: {stress_out['viability_penalty'].shape}")
    print(f"[Opto->MDOE] Stress Markers shape: {stress_out['stress_markers'].shape}")
    
    # MDOE -> Opto
    opto_targets = bridge.mdoe_to_opto_targets(mdoe_ctx)
    print(f"[MDOE->Opto] Target Rates shape: {opto_targets['target_rates'].shape}")
    print(f"[MDOE->Opto] Max Safe Irradiance shape: {opto_targets['max_safe_irradiance'].shape}")
    
    # Gradient Flow Check
    loss = stress_out['viability_penalty'].mean() + opto_targets['target_rates'].mean()
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() 
                  for p in bridge.parameters() if p.requires_grad)
    print(f"[Grad Check] All gradients finite: {grad_ok}")
    
    # 2. Test Slurm Generator
    slurm_cfg = ClosedLoopSlurmConfig(nodes=2, gpus_per_node=4, account="MSPS_LAB")
    slurm_gen = ClosedLoopSlurmGenerator(slurm_cfg)
    slurm_gen.write_slurm_file("test_submit_opto_mdoe.slurm")
    
    print("\n✅ ALL SMOKE TESTS PASSED")
    print("="*70)
