"""
HIV LTR Bistable Dynamics Engine — NATIVE FULL DIFFERENTIABLE
Based on: HybridPsychopathologyEngine (SESI Stack)
Purpose: Simulate 'Shock and Kill' Latency Reversal Agents (LRAs)
Developer: PAI AND Yoon A Limsuwan / MSPS NETWORK
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HIV_LTR_BistableEngine(nn.Module):
    """
    จำลองพลวัตของ HIV LTR Promoter (Tat feedback + Epigenetic silencing)
    Continuous core: Bistable ODE (Tat, HDAC, NF-kB) integrated with RK4 (fp32-internal).
    Discrete network: Cellular states (CD4 activation, Apoptosis, Cytokine release).
    """
    def __init__(self, num_cell_states: int = 16, dt: float = 0.01):
        super().__init__()
        self.num_nodes = num_cell_states
        self.dt = dt
        
        # Continuous Core Parameters (Learnable biological rates)
        self.k_tat_prod = nn.Parameter(torch.tensor(0.5))   # Tat production rate
        self.k_tat_deg = nn.Parameter(torch.tensor(0.1))    # Tat degradation
        self.k_hdac_prod = nn.Parameter(torch.tensor(0.2))  # Epigenetic silencing (HDAC)
        self.k_nfkb_act = nn.Parameter(torch.tensor(1.0))   # NF-kB activation (Shock)
        
        # C^∞ Surrogate parameters (AMP-Safe)
        self.register_buffer("_dt", torch.tensor(dt, dtype=torch.float32), persistent=False)
        self.register_buffer("_dt_half", torch.tensor(dt * 0.5, dtype=torch.float32), persistent=False)
        self.register_buffer("_dt_sixth", torch.tensor(dt / 6.0, dtype=torch.float32), persistent=False)
        self.register_buffer("_eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False)

    def _safe_abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x * x + self._eps)

    def _ltr_derivatives(self, tat, hdac, nfkb, lra_shock_agent):
        """HIV LTR ODE: Tat positive feedback inhibited by HDAC, activated by NF-kB"""
        # Tat activates its own promoter (Hill function approximated via C^∞ softplus)
        tat_feedback = F.softplus(tat - hdac, beta=10.0) 
        
        # Shock agent (e.g., PKC agonist, HDAC inhibitor) forces NF-kB
        shock_effect = lra_shock_agent * self.k_nfkb_act
        
        d_tat = self.k_tat_prod * tat_feedback * (1.0 + shock_effect) - self.k_tat_deg * tat
        d_hdac = self.k_hdac_prod * (1.0 - F.sigmoid(tat)) - 0.05 * hdac # Tat represses HDAC
        d_nfkb = 0.1 * shock_effect - 0.1 * nfkb
        
        return d_tat, d_hdac, d_nfkb

    def _rk4_step(self, tat, hdac, nfkb, lra_shock_agent):
        """fp32-internal RK4 for chaotic/bistable stability"""
        dtype_in = tat.dtype
        t32, h32, n32 = tat.float(), hdac.float(), nfkb.float()
        lra32 = lra_shock_agent.float()
        
        k1t, k1h, k1n = self._ltr_derivatives(t32, h32, n32, lra32)
        k2t, k2h, k2n = self._ltr_derivatives(t32 + self._dt_half*k1t, h32 + self._dt_half*k1h, n32 + self._dt_half*k1n, lra32)
        k3t, k3h, k3n = self._ltr_derivatives(t32 + self._dt_half*k2t, h32 + self._dt_half*k2h, n32 + self._dt_half*k2n, lra32)
        k4t, k4h, k4n = self._ltr_derivatives(t32 + self._dt*k3t, h32 + self._dt*k3h, n32 + self._dt*k3n, lra32)
        
        tat_next = t32 + self._dt_sixth * (k1t + 2*k2t + 2*k3t + k4t)
        hdac_next = h32 + self._dt_sixth * (k1h + 2*k2h + 2*k3h + k4h)
        nfkb_next = n32 + self._dt_sixth * (k1n + 2*k2n + 2*k3n + k4n)
        
        return tat_next.to(dtype_in), hdac_next.to(dtype_in), nfkb_next.to(dtype_in)

    def forward(self, tat0, hdac0, nfkb0, lra_sequence, steps=100):
        """
        lra_sequence: (B, steps) ความเข้มข้นของยา Latency Reversal Agent ตามเวลา
        Returns: Tat trajectory (ถ้า Tat พุ่งสูง = Shock successful, ไวรัสตื่น)
        """
        tat, hdac, nfkb = tat0, hdac0, nfkb0
        tat_trajectory = []
        
        for t in range(steps):
            lra_t = lra_sequence[:, t].unsqueeze(-1)
            tat, hdac, nfkb = self._rk4_step(tat, hdac, nfkb, lra_t)
            tat_trajectory.append(tat)
            
        return torch.stack(tat_trajectory, dim=1) # (B, steps, 1)
