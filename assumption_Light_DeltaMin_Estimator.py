# =============================================================================
# Assumption Light DeltaMin Estimator
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# Framework    : Structural Calculus (Deterministic Topological Framework)
# License      : MIT
# Year         : 2026
# =============================================================================


import torch
import torch.nn as nn
import math

class AssumptionLightDeltaMinEstimator(nn.Module):
    """
    Native Differentiable Module for estimating \delta_{min} 
    and constructing an Assumption-light Confidence Interval via Subsampling
    according to Theorem 4.4 and Definition 4.3 (Paper 11).
    """
    def __init__(self, p=0.01, alpha=0.05, auto_block_length=True):
        super().__init__()
        self.p = p # Probability (p) for the Empirical (1-p)-quantile (POT approach)
        self.alpha = alpha # Significance level for the confidence interval (e.g., 0.05 for 95% CI)
        self.auto_block_length = auto_block_length

    def _compute_excursions(self, trajectories):
        """
        Calculates the excursion sizes between successive states.
        Serves as a surrogate for Hausdorff-distance (d_H).
        """
        # trajectories shape: [N_trajectories, T_steps, feature_dim]
        # Calculate L2 Norm (or adapt to Hausdorff if the data structure is a set of points)
        deltas = torch.norm(trajectories[:, 1:, :] - trajectories[:, :-1, :], dim=-1)
        return deltas # Shape: [N, T-1]

    def _select_block_length(self, T):
        """
        Automatic block-length selection (b_N) 
        following the principles of Politis & White (2004) for an a-mixing process.
        """
        # In a full-scale deployment, an AR(1) fitting can be embedded here.
        # For this module, we use the heuristic b_N ~ T^(1/3), which is theoretically safe for mixing processes.
        b_n = max(2, int(math.pow(T, 1.0/3.0)))
        return b_n

    def forward(self, trajectories):
        """
        trajectories: Tensor of SESI interface trajectories
        """
        N, T, _ = trajectories.shape
        
        # 1. Extract excursion sizes l_j^{(i)}
        excursions = self._compute_excursions(trajectories)
        pooled_excursions = excursions.reshape(-1)
        
        # 2. Estimate the Empirical Block-Minimum \hat{\delta}_N
        # Using torch.quantile which supports Autograd (Subgradient routing)
        delta_hat = torch.quantile(pooled_excursions, self.p)
        
        # 3. Construct the Subsampling Confidence Interval
        b_N = self._select_block_length(T - 1) if self.auto_block_length else int(math.sqrt(T-1))
        num_blocks = (T - 1) - b_N + 1
        
        subsample_deltas = []
        for start in range(num_blocks):
            # Extract subsample block of length b_N
            block = excursions[:, start:start+b_N].reshape(-1)
            sub_delta = torch.quantile(block, self.p)
            subsample_deltas.append(sub_delta)
            
        subsample_deltas = torch.stack(subsample_deltas)
        
        # 4. Compute the Empirical Distribution of \sqrt{b_N}(\delta_{N,b} - \hat{\delta}_N)
        scaled_stat = math.sqrt(b_N) * (subsample_deltas - delta_hat)
        
        # 5. Extract bounds c_{1-\alpha/2} and c_{\alpha/2}
        c_lower = torch.quantile(scaled_stat, self.alpha / 2)
        c_upper = torch.quantile(scaled_stat, 1 - self.alpha / 2)
        
        # 6. Construct the Assumption-light CI for \delta_{min}
        ci_lower = delta_hat - (c_upper / math.sqrt(b_N))
        ci_upper = delta_hat - (c_lower / math.sqrt(b_N))
        
        # Returns \hat{\delta}_N and the confidence bounds
        return delta_hat, ci_lower, ci_upper
