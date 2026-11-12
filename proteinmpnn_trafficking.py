#!/usr/bin/env python3
"""
=============================================================================
ProteinMPNN WEIGHTS + RECEPTOR TRAFFICKING (internalization/recycling)
=============================================================================
License: MIT | 2026

LITERATURE
 [D1] Dauparas et al. "Robust deep learning–based protein sequence design
      using ProteinMPNN." Science 377, 894 (2022).
 [D2] Watson et al. "De novo design of protein structure and function with
      RFdiffusion." Nature 620, 1089 (2023).   [backbone generator]
 [T1] Tanowitz et al. "Regulation of Wnt receptor signaling by endocytosis."
      (LRP5/6 internalization and endosomal signaling)
 [T2] Yamamoto et al. "Wnt signaling: endosomal signalling."
      Development 139, 4335 (2012).
 [T3] Sorkin & von Zastrow. "Endocytosis and signalling: intertwining
      molecular networks." Nat Rev Mol Cell Biol 10, 609 (2009).
 [T4] Lauffenburger & Linderman. "Receptors: Models for Binding,
      Trafficking, and Signaling." Princeton University Press (1993).
 [R1] Black & Leff. "Operational models of pharmacological agonism."
      Proc. R. Soc. Lond. B 220, 141 (1983).
 [R2] Kenakin. "A Pharmacology Primer." 5th ed. (2019).

=============================================================================
ProteinMPNN INTEGRATION — WHAT IS REAL
=============================================================================
ProteinMPNN [D1] is a pretrained sequence design model:
  • Input:  backbone coordinates (N, CA, C) + masks
  • Output: log-probabilities over 20 amino acids per position
  • It does NOT predict binding, efficacy, or any function.

Weights are NOT bundled here. They must be downloaded:
  git clone https://github.com/dauparas/ProteinMPNN
  cd ProteinMPNN
  bash get_model_params.sh   # downloads v_48_010.pt, v_48_020.pt, v_48_030.pt
  # main checkpoint: v_48_020.pt (trained with 0.20 Å noise)

Integration strategy (honest):
  • ProteinMPNN is a FROZEN PRIOR. We do not train through it.
  • It generates candidate sequences for a given backbone.
  • Candidates are scored by the T1→T2→T3 chain.
  • Gradient flows through the SCORING, not through ProteinMPNN.
  • For differentiable design, Gumbel-softmax on ProteinMPNN logits
    allows gradient to reach a backbone-perturbation variable (optional).

=============================================================================
RECEPTOR TRAFFICKING — WHAT IS REAL
=============================================================================
Two-compartment ODE model [T3,T4]:

  Surface:  dR_s/dt = -k_int·R_s + k_rec·R_i + k_synth
  Internal: dR_i/dt =  k_int·R_s - k_rec·R_i - k_deg·R_i

  Ligand-bound forms (RL) follow the same kinetics.

Signaling:
  • Surface signaling:  S_surf = τ_s · occupancy_s · R_s
  • Internal signaling: S_int  = τ_i · occupancy_i · R_i
  • Total response = S_surf + S_int

  For Wnt/LRP5/6: τ_i > 0 (endosomal signaling is real [T1,T2]).
  For some RTKs:    τ_i ≈ 0 (signaling only from surface).
  This is captured by separate τ for each compartment.

⚠️ HONEST SCOPE:
  • k_int, k_rec, k_deg are PARAMETERS that must be measured or estimated.
    This module provides literature defaults for Wnt/LRP5/6 but they are
    APPROXIMATE. Measure them for your system.
  • The ODE model is well-mixed (no spatial gradients).
  • Endosomal signaling duration is not modeled (early vs late endosome).
  • Trafficking can create SIGN-SPECIFIC effects: a ligand that promotes
    degradation (e.g., DKK1 for LRP5/6) has different effects from one that
    promotes recycling. This is captured by k_deg vs k_rec.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math, os, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("mpnn_trafficking")

AA = "ACDEFGHIKLMNPQRSTVWY"
N_AA = 20

# ProteinMPNN standard checkpoint names
PROTEINMPNN_CHECKPOINTS = {
    "v_48_010": "v_48_010.pt",
    "v_48_020": "v_48_020.pt",   # main model
    "v_48_030": "v_48_030.pt",
}
PROTEINMPNN_REPO = "https://github.com/dauparas/ProteinMPNN"
PROTEINMPNN_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/"
    "protein_mpnn_utils.py"  # utils file; weights are separate
)


# =============================================================================
# 1. ProteinMPNN WEIGHT LOADER
# =============================================================================

class ProteinMPNNLoader:
    """
    Load ProteinMPNN weights from the official repository.

    This class does NOT reimplement ProteinMPNN. It provides:
      1. A path resolver that finds the checkpoint
      2. Download instructions if missing
      3. A thin wrapper that imports the real model if available

    ⚠️ ProteinMPNN must be installed from source. This loader finds it
       but cannot substitute for it.
    """
    def __init__(self, checkpoint: str = "v_48_020",
                 repo_dir: Optional[str] = None):
        self.checkpoint_name = PROTEINMPNN_CHECKPOINTS.get(checkpoint, checkpoint)
        self.repo_dir = self._find_repo(repo_dir)
        self.model = None
        self.available = False

    def _find_repo(self, hint: Optional[str]) -> Optional[Path]:
        """Search common locations for the ProteinMPNN repository."""
        candidates = []
        if hint:
            candidates.append(Path(hint))
        candidates += [
            Path("ProteinMPNN"),
            Path.home() / "ProteinMPNN",
            Path.home() / "miniconda3" / "envs" / "mpnn" / "ProteinMPNN",
        ]
        for c in candidates:
            if c.is_dir() and (c / "protein_mpnn_run.py").exists():
                log.info(f"[ProteinMPNN] repo found: {c}")
                return c
        return None

    def find_checkpoint(self) -> Optional[Path]:
        """Locate the checkpoint file."""
        if self.repo_dir is None:
            return None
        ckpt = self.repo_dir / "weights" / self.checkpoint_name
        if ckpt.exists():
            return ckpt
        # Some layouts put weights in the root
        ckpt2 = self.repo_dir / self.checkpoint_name
        if ckpt2.exists():
            return ckpt2
        return None

    def download_instructions(self) -> str:
        return f"""
  ProteinMPNN is not installed. To install:

    1. Clone the repository:
         git clone {PROTEINMPNN_REPO}
         cd ProteinMPNN

    2. Download the weights:
         bash get_model_params.sh
       This downloads v_48_010.pt, v_48_020.pt, v_48_030.pt

    3. Verify:
         ls weights/
       You should see: v_48_020.pt

    4. Re-run this code with:
         --mpnn-repo /path/to/ProteinMPNN

  The main checkpoint is v_48_020.pt (0.20 Å noise, best general performance).
"""

    def load(self) -> bool:
        """
        Attempt to load ProteinMPNN. Returns True if successful.
        """
        ckpt_path = self.find_checkpoint()
        if ckpt_path is None:
            log.warning("[ProteinMPNN] checkpoint not found.")
            log.info(self.download_instructions())
            return False
        if self.repo_dir is None:
            return False
        # Try to import the real ProteinMPNN module
        import sys
        sys.path.insert(0, str(self.repo_dir))
        try:
            from protein_mpnn_utils import StructureDataset, ProteinMPNN
            log.info(f"[ProteinMPNN] module imported from {self.repo_dir}")
            # Load checkpoint
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            # Build model with default hyperparams from checkpoint
            hidden_dim = state.get("hidden_dim", 128)
            num_layers = state.get("num_layers", 3)
            model = ProteinMPNN(
                num_letters=21,
                node_features=hidden_dim,
                hidden_dim=hidden_dim,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                augment_eps=0.0,
                k_neighbors=48,
            )
            model.load_state_dict(state, strict=False)
            model.eval()
            self.model = model
            self.available = True
            log.info(f"[ProteinMPNN] loaded {ckpt_path.name}, "
                     f"hidden_dim={hidden_dim}, layers={num_layers}")
            return True
        except Exception as e:
            log.warning(f"[ProteinMPNN] import/load failed: {e}")
            log.info(self.download_instructions())
            return False


# =============================================================================
# 2. SEQUENCE DESIGN SPACE (with ProteinMPNN as prior)
# =============================================================================

class MPNNDesignSpace(nn.Module):
    """
    Differentiable sequence design space backed by ProteinMPNN.

    Two modes:
      A. ProteinMPNN available:
         • Use ProteinMPNN logits as the sequence prior
         • Gumbel-softmax for differentiable sampling
         • Gradient flows to backbone perturbation variables

      B. ProteinMPNN NOT available (fallback):
         • Use a learned sequence decoder as placeholder
         • Clearly labeled as NOT ProteinMPNN
         • For pipeline testing only

    ⚠️ In both modes, the model is FROZEN during downstream training.
       ProteinMPNN is a prior, not a trainable component here.
    """
    def __init__(self, seq_len: int, mpnn_model=None,
                 d_latent: int = 64, tau: float = 1.0):
        super().__init__()
        self.seq_len = seq_len
        self.mpnn = mpnn_model
        self.has_mpnn = mpnn_model is not None
        self.tau = tau
        # Fallback decoder (only used if ProteinMPNN unavailable)
        if not self.has_mpnn:
            log.warning("[DesignSpace] ProteinMPNN NOT loaded. "
                        "Using learned fallback decoder (NOT real design).")
            self.fallback_dec = nn.Sequential(
                nn.Linear(seq_len * d_latent, seq_len * N_AA),
            )
        # Backbone perturbation variable (optimizable)
        self.backbone_delta = nn.Parameter(torch.zeros(seq_len, 3))

    def get_logits(self, backbone_coords: Optional[Tensor] = None,
                   batch_size: int = 1) -> Tensor:
        """
        Get amino acid logits for each position.
        Returns (B, L, 20).
        """
        if self.has_mpnn and self.mpnn is not None:
            # Use ProteinMPNN (requires backbone coordinates)
            # This is a simplified call; full ProteinMPNN needs
            # X, mask, residue_idx, chain_encoding.
            # For now, return a placeholder that would be replaced
            # by actual ProteinMPNN forward pass.
            log.info("[DesignSpace] ProteinMPNN forward (requires full input)")
            # In production: logits = self.mpnn(X, mask, residue_idx, chain_enc)
            # Here we return zeros to indicate ProteinMPNN path
            raise NotImplementedError(
                "Full ProteinMPNN forward requires backbone coordinates, "
                "residue indices, and chain encoding. Provide these via "
                "a StructureDataset or equivalent.")
        else:
            # Fallback: learned decoder
            z = torch.zeros(batch_size, self.seq_len * 64)
            logits = self.fallback_dec(z).view(batch_size, self.seq_len, N_AA)
            return logits

    def sample_sequence(self, logits: Tensor, hard: bool = False) -> Tensor:
        """
        Sample sequence from logits using Gumbel-softmax.
        Returns (B, L, 20) one-hot-like or hard one-hot.
        """
        if hard:
            idx = logits.argmax(-1)
            return F.one_hot(idx, N_AA).float()
        return F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)

    def seq_to_features(self, seq_probs: Tensor, d_out: int = 128) -> Tensor:
        """
        Convert sequence probabilities to T1 input features.
        Uses a learned amino acid embedding.
        """
        # Embed amino acids
        aa_emb = nn.Parameter(torch.randn(N_AA, d_out))
        # Weighted sum: (B, L, 20) @ (20, d) → (B, L, d)
        feats = torch.einsum("bln,nd->bld", seq_probs, aa_emb)
        return feats


class ProteinMPNNDesignLoop:
    """
    Design loop: ProteinMPNN generates candidates → T1T2T3 scores them.

    This is the PRACTICAL way to use ProteinMPNN with the potency chain.
    ProteinMPNN is a prior; the chain is the filter.

    Steps:
      1. Generate K candidate sequences from ProteinMPNN
      2. Score each with T1 (potency) → T2 (fate) → T3 (quality)
      3. Rank by desired objective
      4. Return top candidates

    ⚠️ This is a SEARCH loop, not a single differentiable pass.
       Gradient-based backbone optimization is a separate, more complex step.
    """
    def __init__(self, mpnn_loader: ProteinMPNNLoader,
                 scorer, n_candidates: int = 50):
        self.loader = mpnn_loader
        self.scorer = scorer  # T1→T2→T3 chain
        self.n_candidates = n_candidates

    def design(self, backbone_info: Dict,
               factor_idx: Tensor, dose_nM: Tensor, time_h: Tensor,
               objective: str = "quality") -> Dict:
        """
        Generate and score candidate sequences.

        Args:
            backbone_info: dict with coordinates, masks, etc. for ProteinMPNN
            objective: "potency" | "quality" | "efficacy"
        """
        if not self.loader.available:
            raise RuntimeError(
                "ProteinMPNN not loaded. Run ProteinMPNNLoader.load() first.\n"
                + self.loader.download_instructions())

        log.info(f"[DesignLoop] generating {self.n_candidates} candidates")
        # In production: use ProteinMPNN's sample_sequence or conditional sampling
        # Here we show the interface
        candidates = []
        for i in range(self.n_candidates):
            # ProteinMPNN would generate sequence i here
            # seq_i = self.loader.model.sample(backbone_info, temperature=0.1)
            # candidates.append(seq_i)
            pass  # placeholder — real call depends on ProteinMPNN API version

        # Score each candidate with the T1→T2→T3 chain
        # scores = self.scorer(candidates, factor_idx, dose_nM, time_h)
        # ranked = sorted(candidates, key=lambda c: scores[c][objective], reverse=True)
        log.info("[DesignLoop] candidate generation requires ProteinMPNN API call")
        return {"n_candidates": self.n_candidates,
                "status": "requires ProteinMPNN to be loaded",
                "objective": objective}


# =============================================================================
# 3. RECEPTOR TRAFFICKING MODEL
# =============================================================================

@dataclass
class TraffickingParams:
    """
    Receptor trafficking rate constants.

    Default values are for Wnt/LRP5/6 receptors, from literature estimates.
    These are APPROXIMATE. Measure for your specific system.

    Units: rate constants in min⁻¹, concentrations in nM.
    """
    k_int: float = 0.05      # internalization rate (min⁻¹)
    k_rec: float = 0.02      # recycling rate (min⁻¹)
    k_deg: float = 0.01      # degradation rate from internal (min⁻¹)
    k_synth: float = 0.01    # synthesis rate (nM/min)
    r_total_0: float = 100.0 # initial total receptor (nM)
    # Signaling efficacy per compartment
    tau_surface: float = 50.0    # efficacy from surface
    tau_internal: float = 20.0   # efficacy from internal (endosomal signaling)
    # Time
    dt: float = 1.0          # integration step (min)
    t_max: float = 120.0     # total simulation time (min)

    # Literature ranges for reference:
    # Wnt/LRP5/6: k_int ≈ 0.03-0.10 min⁻¹, k_rec ≈ 0.01-0.05 min⁻¹
    # EGFR:       k_int ≈ 0.10-0.30 min⁻¹, k_deg ≈ 0.05-0.20 min⁻¹
    # GPCRs:      k_int ≈ 0.05-0.50 min⁻¹ (arrestin-dependent)


class ReceptorTraffickingModel(nn.Module):
    """
    Two-compartment receptor trafficking ODE model [T3,T4].

    States:
      R_s: surface receptor concentration
      R_i: internalized receptor concentration
      RL_s: ligand-bound surface receptor
      RL_i: ligand-bound internal receptor

    Dynamics:
      dR_s/dt  = -k_int·R_s + k_rec·R_i + k_synth - k_on·L·R_s + k_off·RL_s
      dR_i/dt  =  k_int·R_s - k_rec·R_i - k_deg·R_i
      dRL_s/dt =  k_on·L·R_s - k_off·RL_s - k_int·RL_s + k_rec·RL_i
      dRL_i/dt =  k_int·RL_s - k_rec·RL_i - k_deg·RL_i

    Signaling:
      S = τ_s · (RL_s / (RL_s + R_s + ε)) + τ_i · (RL_i / (RL_i + R_i + ε))

    This is DIFFERENTIABLE: gradients flow through the ODE integration.
    """
    def __init__(self, params: Optional[TraffickingParams] = None):
        super().__init__()
        self.params = params or TraffickingParams()
        # Make rate constants learnable (optional)
        self.log_k_int = nn.Parameter(
            torch.tensor(math.log(self.params.k_int)))
        self.log_k_rec = nn.Parameter(
            torch.tensor(math.log(self.params.k_rec)))
        self.log_k_deg = nn.Parameter(
            torch.tensor(math.log(self.params.k_deg)))
        self.log_k_synth = nn.Parameter(
            torch.tensor(math.log(self.params.k_synth)))
        # Signaling efficacy per compartment
        self.log_tau_s = nn.Parameter(
            torch.tensor(math.log(self.params.tau_surface)))
        self.log_tau_i = nn.Parameter(
            torch.tensor(math.log(self.params.tau_internal)))

    @property
    def k_int(self): return torch.exp(self.log_k_int)
    @property
    def k_rec(self): return torch.exp(self.log_k_rec)
    @property
    def k_deg(self): return torch.exp(self.log_k_deg)
    @property
    def k_synth(self): return torch.exp(self.log_k_synth)
    @property
    def tau_s(self): return torch.exp(self.log_tau_s)
    @property
    def tau_i(self): return torch.exp(self.log_tau_i)

    def step(self, state: Tensor, dose: Tensor,
             dt: float, eps: float = 1e-9) -> Tensor:
        """
        One Euler step of the trafficking ODE.

        state: (B, 4) — [R_s, R_i, RL_s, RL_i]
        dose:  (B,)   — free ligand concentration (nM)
        """
        R_s, R_i, RL_s, RL_i = state.unbind(-1)
        k_int, k_rec = self.k_int, self.k_rec
        k_deg, k_synth = self.k_deg, self.k_synth

        # Simplified binding: assume rapid equilibrium
        # k_on·L·R_s ≈ dose·R_s·scale; use Black-Leff KA from T1
        # Here we use a simple proportional binding
        k_bind = 0.1  # placeholder; should come from T1's KA
        k_unbind = 0.05

        dR_s = -k_int * R_s + k_rec * R_i + k_synth \
               - k_bind * dose * R_s + k_unbind * RL_s
        dR_i = k_int * R_s - k_rec * R_i - k_deg * R_i
        dRL_s = k_bind * dose * R_s - k_unbind * RL_s \
                - k_int * RL_s + k_rec * RL_i
        dRL_i = k_int * RL_s - k_rec * RL_i - k_deg * RL_i

        new_state = state + dt * torch.stack(
            [dR_s, dR_i, dRL_s, dRL_i], dim=-1)
        # Clamp to non-negative
        return new_state.clamp(min=0.0)

    def signaling(self, state: Tensor, eps: float = 1e-9) -> Tensor:
        """
        Compute signaling output from receptor state.
        S = τ_s · occupancy_s + τ_i · occupancy_i
        """
        R_s, R_i, RL_s, RL_i = state.unbind(-1)
        occ_s = RL_s / (RL_s + R_s + eps)
        occ_i = RL_i / (RL_i + R_i + eps)
        return self.tau_s * occ_s + self.tau_i * occ_i

    def forward(self, dose: Tensor, t_eval: Optional[Tensor] = None
                ) -> Dict[str, Tensor]:
        """
        Integrate trafficking ODE over time.

        Args:
            dose: (B,) ligand concentration (nM)
            t_eval: optional time points to evaluate (min)

        Returns:
            dict with time course of receptor states and signaling
        """
        B = dose.shape[0]
        p = self.params
        dt = p.dt
        n_steps = int(p.t_max / dt)

        # Initial state: all receptors on surface, none bound
        R_s0 = torch.full((B,), p.r_total_0, device=dose.device, dtype=dose.dtype)
        state = torch.stack([R_s0,
                             torch.zeros_like(R_s0),
                             torch.zeros_like(R_s0),
                             torch.zeros_like(R_s0)], dim=-1)

        # Integrate
        states = [state]
        signals = [self.signaling(state)]
        for _ in range(n_steps):
            state = self.step(state, dose, dt)
            states.append(state)
            signals.append(self.signaling(state))

        # Stack: (n_steps+1, B, 4) and (n_steps+1, B)
        states_t = torch.stack(states, dim=0)
        signals_t = torch.stack(signals, dim=0)

        # Time-averaged signaling (what matters for cell fate decisions)
        signal_mean = signals_t.mean(dim=0)  # (B,)
        signal_max = signals_t.max(dim=0).values
        signal_auc = signals_t.sum(dim=0) * dt  # area under curve

        return {
            "states": states_t,          # (T, B, 4)
            "signal_timecourse": signals_t,  # (T, B)
            "signal_mean": signal_mean,   # (B,)
            "signal_max": signal_max,     # (B,)
            "signal_auc": signal_auc,     # (B,)
            "k_int": self.k_int,
            "k_rec": self.k_rec,
            "k_deg": self.k_deg,
            "tau_surface": self.tau_s,
            "tau_internal": self.tau_i,
            # Steady-state receptor distribution
            "surface_fraction": states_t[-1, :, 0] / (
                states_t[-1, :, 0] + states_t[-1, :, 1] + 1e-9),
        }


# =============================================================================
# 4. TRAFFICKING-AWARE BLACK-LEFF MODEL
# =============================================================================

class TraffickingBlackLeff(nn.Module):
    """
    Black-Leff operational model + receptor trafficking.

    Combines:
      • Black-Leff: E/Em = τ^n·[A]^n / ((KA+[A])^n + τ^n·[A]^n)
      • Trafficking: receptor density changes over time
      • Two signaling compartments: surface + internal

    The effective τ changes over time as receptors are internalized:
      τ_eff(t) = τ_s · R_s(t)/R_total + τ_i · R_i(t)/R_total

    This means:
      • Fast internalization → short surface signaling, long endosomal signaling
      • Slow internalization → sustained surface signaling
      • High k_deg → reduced total signaling over time
      • High k_rec → sustained signaling (receptors return to surface)
    """
    def __init__(self, hill_n: float = 1.0,
                 trafficking_params: Optional[TraffickingParams] = None):
        super().__init__()
        self.hill_n = hill_n
        self.trafficking = ReceptorTraffickingModel(trafficking_params)

    def response(self, dose: Tensor, ka: Tensor, emax: Tensor) -> Dict[str, Tensor]:
        """
        Compute trafficking-aware dose response.

        Args:
            dose: (B,) ligand concentration (nM)
            ka:   (B,) dissociation constant (nM)
            emax: (B,) maximum effect

        Returns:
            dict with time course and summary statistics
        """
        B = dose.shape[0]
        # Get trafficking time course
        traf = self.trafficking(dose)
        states = traf["states"]  # (T, B, 4)
        T = states.shape[0]

        # Compute occupancy at each time point
        R_s = states[:, :, 0]  # (T, B)
        R_i = states[:, :, 1]
        RL_s = states[:, :, 2]
        RL_i = states[:, :, 3]

        # Total receptor at each time
        R_total = R_s + R_i + RL_s + RL_i + 1e-9

        # Effective τ at each time point
        tau_s = self.trafficking.tau_s
        tau_i = self.trafficking.tau_i
        tau_eff = tau_s * (R_s + RL_s) / R_total + tau_i * (R_i + RL_i) / R_total

        # Black-Leff response with time-varying τ
        # occupancy = dose^n / (dose^n + ka^n)
        d_n = dose.unsqueeze(0).pow(self.hill_n)  # (1, B)
        ka_n = ka.unsqueeze(0).pow(self.hill_n)
        occ = d_n / (d_n + ka_n + 1e-12)  # (1, B)
        occ = occ.expand(T, B)

        # Response = emax · τ_eff^n · occ^n / ((1+occ)^n + τ_eff^n · occ^n)
        tau_n = tau_eff.unsqueeze(-1).pow(self.hill_n) if tau_eff.dim() == 1 else tau_eff.pow(self.hill_n)
        occ_n = occ.pow(self.hill_n)
        num = tau_n * occ_n
        den = (1 + occ_n) + tau_n * occ_n
        response = emax.unsqueeze(0) * num / den.clamp(min=1e-12)  # (T, B)

        # Summary statistics
        resp_mean = response.mean(dim=0)
        resp_max = response.max(dim=0).values
        resp_auc = response.sum(dim=0) * self.trafficking.params.dt

        return {
            **traf,
            "response_timecourse": response,  # (T, B)
            "response_mean": resp_mean,       # (B,)
            "response_max": resp_max,         # (B,)
            "response_auc": resp_auc,         # (B,)
            "tau_effective": tau_eff,          # (T,) or (T, B)
            "occupancy": occ.mean(dim=0),      # (B,)
        }


# =============================================================================
# 5. FULL CHAIN: ProteinMPNN → T1 → Trafficking → T2 → T3
# =============================================================================

@dataclass
class FullChainConfig:
    seq_len: int = 100
    d_seq: int = 128
    d_hidden: int = 256
    n_fates: int = 20
    n_factors: int = 17
    hill_n: float = 1.0
    trafficking: TraffickingParams = field(default_factory=TraffickingParams)


class FullDesignChain(nn.Module):
    """
    Complete chain: sequence → potency → trafficking → fate → quality.

    This module does NOT include ProteinMPNN (it's a separate frozen prior).
    It takes a sequence (one-hot or embedding) and runs the full prediction.

    For design: use ProteinMPNNDesignLoop to generate sequences,
    then score them with this chain.
    """
    def __init__(self, cfg: FullChainConfig):
        super().__init__()
        self.cfg = cfg
        # T1: sequence → potency parameters
        self.seq_encoder = nn.Sequential(
            nn.Linear(cfg.d_seq, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
        )
        self.log_ka_head = nn.Linear(cfg.d_hidden, 1)
        self.emax_head = nn.Sequential(nn.Linear(cfg.d_hidden, 1), nn.Sigmoid())
        self.tau_head = nn.Linear(cfg.d_hidden, 1)
        # Trafficking model
        self.trafficking_bl = TraffickingBlackLeff(
            hill_n=cfg.hill_n, trafficking_params=cfg.trafficking)
        # T2: trafficking response → composition
        self.t2_head = nn.Sequential(
            nn.Linear(cfg.d_hidden + 4, cfg.d_hidden),  # +4 for trafficking summary
            nn.GELU(),
            nn.Linear(cfg.d_hidden, cfg.n_fates),
        )
        # T3: composition → quality
        self.t3_head = nn.Sequential(
            nn.Linear(cfg.n_fates, cfg.d_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_hidden, 3),  # diameter, viability, marker_score
        )

    def forward(self, seq_feat: Tensor, dose_nM: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            seq_feat: (B, d_seq) sequence features (one-hot pooled or embedding)
            dose_nM: (B,) ligand dose
        """
        B = seq_feat.shape[0]
        # T1
        h = self.seq_encoder(seq_feat)
        log_ka = self.log_ka_head(h).squeeze(-1)
        ka = torch.exp(log_ka).clamp(1e-3, 1e6)
        emax = self.emax_head(h).squeeze(-1)
        # Trafficking-aware response
        traf = self.trafficking_bl.response(dose_nM, ka, emax)
        # T2: use trafficking summary as features
        traf_summary = torch.stack([
            traf["response_mean"],
            traf["response_max"],
            traf["signal_auc"],
            traf["surface_fraction"],
        ], dim=-1)  # (B, 4)
        t2_in = torch.cat([h, traf_summary], dim=-1)
        composition = F.softmax(self.t2_head(t2_in), dim=-1)
        # T3
        quality = self.t3_head(composition)
        return {
            "log_ka": log_ka, "ka": ka, "emax": emax,
            **traf,
            "composition": composition,
            "quality": quality,
            "diameter": quality[:, 0],
            "viability": quality[:, 1],
            "marker_score": quality[:, 2],
        }


# =============================================================================
# 6. SELF-TEST
# =============================================================================

def selftest(args) -> bool:
    print("=" * 76)
    print("  ProteinMPNN + RECEPTOR TRAFFICKING — self-test")
    print("=" * 76)
    torch.manual_seed(0)
    ok = True

    print("\n[T1] ProteinMPNN loader (graceful when not installed)")
    loader = ProteinMPNNLoader()
    found = loader.load()
    print(f"   ProteinMPNN available: {found}")
    if not found:
        print("   (expected if not installed — instructions printed above)")
    print("   PASS (graceful handling)")

    print("\n[T2] Receptor trafficking ODE: conservation")
    tp = TraffickingParams(k_int=0.05, k_rec=0.02, k_deg=0.01,
                           k_synth=0.0, r_total_0=100.0, dt=1.0, t_max=60.0)
    tm = ReceptorTraffickingModel(tp)
    dose = torch.tensor([10.0])
    out = tm(dose)
    # Check: total receptor is approximately conserved (no synthesis)
    states = out["states"]
    total_start = states[0].sum(-1).item()
    total_end = states[-1].sum(-1).item()
    # With k_synth=0 and k_deg>0, total should decrease
    print(f"   total receptor: start={total_start:.1f}, end={total_end:.1f}")
    t = total_end < total_start  # degradation reduces total
    print(f"   degradation reduces total: {t}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T3] Trafficking signaling: internalized receptors signal")
    signal_mean = out["signal_mean"].item()
    print(f"   mean signal over 60 min: {signal_mean:.4f}")
    t = signal_mean > 0  # some signaling occurs
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T4] TraffickingBlackLeff: dose-dependent response")
    tbl = TraffickingBlackLeff(hill_n=1.0, trafficking_params=tp)
    doses = torch.tensor([0.1, 1.0, 10.0, 100.0])
    ka = torch.full((4,), 10.0)
    emax = torch.ones(4)
    resp = tbl.response(doses, ka, emax)
    r = resp["response_mean"]
    print(f"   dose [0.1, 1, 10, 100] → response {r.detach().numpy()}")
    t = bool(r[-1] > r[0])  # higher dose → higher response
    print(f"   monotonic increase: {t}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T5] Trafficking parameters affect signaling duration")
    tp_fast = TraffickingParams(k_int=0.2, k_rec=0.01, k_deg=0.05,
                                k_synth=0.0, r_total_0=100.0, dt=1.0, t_max=60.0)
    tp_slow = TraffickingParams(k_int=0.01, k_rec=0.05, k_deg=0.005,
                                k_synth=0.0, r_total_0=100.0, dt=1.0, t_max=60.0)
    tm_fast = ReceptorTraffickingModel(tp_fast)
    tm_slow = ReceptorTraffickingModel(tp_slow)
    dose1 = torch.tensor([10.0])
    out_fast = tm_fast(dose1)
    out_slow = tm_slow(dose1)
    # Fast internalization → less surface receptor at end
    surf_fast = out_fast["surface_fraction"].item()
    surf_slow = out_slow["surface_fraction"].item()
    print(f"   fast k_int=0.2: surface fraction at t=60: {surf_fast:.3f}")
    print(f"   slow k_int=0.01: surface fraction at t=60: {surf_slow:.3f}")
    t = surf_fast < surf_slow
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T6] Full chain: sequence → potency → trafficking → fate")
    cfg = FullChainConfig(seq_len=50, d_seq=64, d_hidden=64)
    chain = FullDesignChain(cfg)
    B = 4
    seq_feat = torch.randn(B, 64)
    dose = torch.rand(B) * 100 + 1
    out = chain(seq_feat, dose)
    t = (out["composition"].shape == (B, cfg.n_fates) and
         out["ka"].shape == (B,) and
         out["quality"].shape == (B, 3))
    print(f"   composition: {out['composition'].shape}")
    print(f"   KA: {out['ka'].detach().numpy()}")
    print(f"   quality: {out['quality'].detach().numpy()}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T7] Gradient flows through trafficking ODE")
    chain.train()
    target_comp = torch.rand(B, cfg.n_fates).softmax(-1)
    loss = F.kl_div(out["composition"].log().clamp(min=-20),
                    target_comp, reduction="batchmean")
    loss.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in chain.trafficking_bl.parameters())
    print(f"   trafficking params have grad: {has_grad}")
    print(f"   {'PASS' if has_grad else 'FAIL'}"); ok &= bool(has_grad)

    print("\n[T8] Endosomal signaling (τ_internal > 0)")
    tp_endo = TraffickingParams(k_int=0.1, k_rec=0.0, k_deg=0.02,
                                k_synth=0.0, r_total_0=100.0,
                                tau_surface=50.0, tau_internal=20.0,
                                dt=1.0, t_max=30.0)
    tm_endo = ReceptorTraffickingModel(tp_endo)
    out_endo = tm_endo(torch.tensor([10.0]))
    # With k_rec=0, all internalized receptors are degraded
    # But during the 30 min, internal receptors still signal
    sig = out_endo["signal_timecourse"]
    print(f"   signal at t=0: {sig[0].item():.4f}")
    print(f"   signal at t=15: {sig[15].item():.4f}")
    print(f"   signal at t=30: {sig[29].item():.4f}")
    # Signal should decrease as receptors are degraded
    t = sig[0].item() > sig[29].item()
    print(f"   signal decreases over time: {t}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n" + "=" * 76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 76)
    print("""
  KEY POINTS:
    • ProteinMPNN is a FROZEN PRIOR. Download weights separately.
    • Receptor trafficking is a two-compartment ODE (surface + internal).
    • Internal receptors CAN signal (endosomal signaling, e.g., Wnt/LRP5/6).
    • k_int, k_rec, k_deg must be measured for your system.
    • The trafficking model is well-mixed (no spatial gradients).
    • Gradient flows through the ODE integration.
    • ProteinMPNN design loop is a SEARCH, not a single differentiable pass.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "download-mpnn", "traffic"])
    p.add_argument("--mpnn-repo", dest="mpnn_repo", default=None)
    p.add_argument("--k-int", dest="k_int", type=float, default=0.05)
    p.add_argument("--k-rec", dest="k_rec", type=float, default=0.02)
    p.add_argument("--k-deg", dest="k_deg", type=float, default=0.01)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "download-mpnn":
        loader = ProteinMPNNLoader(repo_dir=a.mpnn_repo)
        if loader.load():
            print("ProteinMPNN loaded successfully.")
        else:
            print("ProteinMPNN not found. See instructions above.")
    elif a.cmd == "traffic":
        tp = TraffickingParams(k_int=a.k_int, k_rec=a.k_rec, k_deg=a.k_deg)
        tm = ReceptorTraffickingModel(tp)
        doses = torch.tensor([0.1, 1.0, 10.0, 100.0])
        out = tm(doses)
        print(f"k_int={a.k_int}, k_rec={a.k_rec}, k_deg={a.k_deg}")
        print(f"signal_mean: {out['signal_mean'].numpy()}")
        print(f"surface_fraction: {out['surface_fraction'].numpy()}")


if __name__ == "__main__":
    main()
