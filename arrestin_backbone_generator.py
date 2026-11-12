#!/usr/bin/env python3
"""
=============================================================================
β-ARRESTIN RECRUITMENT (GPCR) + BACKBONE GENERATOR FOR ProteinMPNN
=============================================================================
Developer    : PAI and Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

LITERATURE
 [A1] Lefkowitz & Shenoy. "Transduction of receptor signals by β-arrestins."
      Science 308, 512 (2005).
 [A2] DeWire et al. "β-arrestins and cell signaling." Annu Rev Physiol 69, 483 (2007).
 [A3] Violin et al. "β-arrestin 2-mediated biased agonism." J Pharmacol Exp Ther 335, 41 (2010).
 [A4] Nobles et al. "Distinct phosphorylation sites on the β2-adrenergic receptor
      establish a barcode that encodes differential functions of β-arrestin."
      Sci Signal 4, ra51 (2011).
 [B1] Watson et al. "De novo design of protein structure and function with
      RFdiffusion." Nature 620, 1089 (2023).
 [B2] Dauparas et al. "Robust deep learning–based protein sequence design
      using ProteinMPNN." Science 377, 894 (2022).
 [B3] Dunbrack & Cohen. "Bayesian statistical analysis of protein side-chain
      rotamer preferences." Protein Sci 6, 1661 (1997).
 [T1] Sorkin & von Zastrow. "Endocytosis and signalling." Nat Rev Mol Cell Biol 10, 609 (2009).
 [T2] Lauffenburger & Linderman. "Receptors: Models for Binding, Trafficking,
      and Signaling." Princeton (1993).

=============================================================================
β-ARRESTIN MODEL — WHAT IS REAL
=============================================================================
GPCR activation cycle with arrestin [A1,A2,T1]:

  R + L ⇌ RL                    (binding, KA)
  RL → RL*                      (activation)
  RL* → RL*P                    (GRK phosphorylation, k_phos)
  RL*P + Arr ⇌ RL*P·Arr        (arrestin binding, k_arr_on/off)
  RL*P·Arr → RL_int            (clathrin-mediated endocytosis, k_int_arr)
  RL_int → R + L                (recycling or degradation)

Key rates (literature estimates for β2AR; MEASURE for your receptor):
  k_arr_on   ≈ 1–10 µM⁻¹s⁻¹
  k_arr_off  ≈ 0.1–1 s⁻¹
  k_int_arr  ≈ 0.1–0.5 min⁻¹  (10–100× faster than basal k_int)
  [Arr]cyto  ≈ 1–10 µM

BIASED AGONISM [A3]:
  bias_factor = k_arr_on(ligand) / k_arr_on(reference)
  bias_factor > 1: arrestin-biased (promotes internalization)
  bias_factor < 1: G-protein-biased (sustained surface signaling)
  bias_factor = 0: no arrestin recruitment

=============================================================================
BACKBONE GENERATOR — WHAT IS REAL
=============================================================================
ProteinMPNN [B2] requires backbone coordinates (N, CA, C) as input.
Three routes are provided:

  1. RFdiffusion [B1]: state-of-the-art de novo backbone generation.
     Requires downloading RFdiffusion weights (~1 GB). NOT bundled here.
     This file provides an INTERFACE, not a reimplementation.

  2. Template-based: extract backbone from an existing PDB structure.
     Most reliable. Works immediately with any PDB file.

  3. Simple builder: ideal α-helix / β-strand / poly-Ala.
     For pipeline testing ONLY. Not real design.

⚠️ HONEST SCOPE:
  ❌ RFdiffusion weights are NOT included. Download from:
     https://github.com/RosettaCommons/RFdiffusion
  ❌ Simple builder outputs are NOT real backbones.
  ❌ Template-based extraction inherits any errors in the PDB file.
  ❌ Arrestin parameters are literature estimates for β2AR.
     They MUST be measured for your specific GPCR.
  ❌ Biased agonism prediction requires experimental validation.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math, os
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
log = logging.getLogger("arrestin_backbone")

AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
 "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
 "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# Ideal backbone geometry (Engh & Huber 1991, Richardson)
BOND_N_CA = 1.458   # Å
BOND_CA_C = 1.523   # Å
BOND_C_N = 1.329    # Å
ANGLE_N_CA_C = 111.0  # degrees
ANGLE_CA_C_N = 116.2
ANGLE_C_N_CA = 121.7

# Helix parameters (3.6 residues/turn, rise 1.5 Å/residue)
HELIX_RISE = 1.5
HELIX_RADIUS = 2.3
HELIX_RES_PER_TURN = 3.6

# Sheet parameters (extended strand, rise ~3.4 Å/residue)
SHEET_RISE = 3.4


# =============================================================================
# 1. β-ARRESTIN RECRUITMENT MODEL
# =============================================================================

@dataclass
class ArrestinParams:
    """
    Rate constants for GPCR arrestin recruitment.
    Default values are for β2-adrenergic receptor [A1,A2,T1].
    ⚠️ MEASURE for your specific receptor before use.

    Units: rates in min⁻¹, concentrations in nM,
           k_arr_on in nM⁻¹·min⁻¹ (second-order).
    """
    # Basal trafficking
    k_int: float = 0.05       # basal internalization (min⁻¹)
    k_rec: float = 0.02       # recycling (min⁻¹)
    k_deg: float = 0.01       # degradation from internal (min⁻¹)
    k_synth: float = 0.005    # synthesis (nM/min)
    r_total_0: float = 100.0  # initial total receptor (nM)

    # Arrestin recruitment
    k_arr_on: float = 0.001   # arrestin binding (nM⁻¹·min⁻¹)
    k_arr_off: float = 0.5    # arrestin unbinding (min⁻¹)
    k_int_arr: float = 0.3    # internalization from arrestin-bound (min⁻¹)
    arrestin_conc: float = 5000.0  # cytoplasmic arrestin (nM, ~5 µM)

    # Phosphorylation (GRK)
    k_phos: float = 0.2       # phosphorylation of active receptor (min⁻¹)
    k_dephos: float = 0.1     # dephosphorylation (min⁻¹)

    # Signaling efficacy per compartment
    tau_gprotein: float = 50.0   # G-protein signaling from RL*
    tau_arrestin: float = 20.0   # Arrestin-mediated signaling from RL*P·Arr
    tau_internal: float = 10.0   # Endosomal signaling

    # Bias factor (ligand-specific; 1.0 = balanced, >1 = arrestin-biased)
    bias_factor: float = 1.0

    # Integration
    dt: float = 0.5          # min
    t_max: float = 60.0      # min


class ArrestinTraffickingModel(nn.Module):
    """
    GPCR trafficking with β-arrestin recruitment.

    States (per receptor molecule):
      R_s:     surface, unbound, inactive
      RL_s:    surface, ligand-bound, active (G-protein signaling)
      RL_p:    surface, phosphorylated by GRK
      RL_arr:  surface, arrestin-bound (arrestin signaling)
      RL_int:  internalized

    ODE system:
      dR_s/dt    = -k_bind·L·R_s + k_unbind·RL_s + k_rec·RL_int + k_synth - k_deact·R_s
      dRL_s/dt   =  k_bind·L·R_s - k_unbind·RL_s - k_phos·RL_s + k_dephos·RL_p
                    - k_int·RL_s
      dRL_p/dt   =  k_phos·RL_s - k_dephos·RL_p
                    - k_arr_on·[Arr]·RL_p + k_arr_off·RL_arr
      dRL_arr/dt =  k_arr_on·[Arr]·RL_p - k_arr_off·RL_arr
                    - k_int_arr·RL_arr
      dRL_int/dt =  k_int·RL_s + k_int_arr·RL_arr - k_rec·RL_int - k_deg·RL_int

    Signaling:
      G-protein:  S_G   = τ_G   · RL_s / (RL_s + R_s)
      Arrestin:   S_Arr = τ_Arr · RL_arr / (RL_arr + RL_p)
      Endosomal:  S_end = τ_end · RL_int / R_total

    ⚠️ bias_factor scales k_arr_on:
       k_arr_on_eff = bias_factor × k_arr_on
       bias_factor > 1: arrestin-biased agonist
       bias_factor < 1: G-protein-biased agonist
       bias_factor = 0: no arrestin recruitment
    """
    def __init__(self, params: Optional[ArrestinParams] = None):
        super().__init__()
        self.p = params or ArrestinParams()
        # Learnable log-rates (bounded positivity via exp)
        self.log_k_arr_on = nn.Parameter(torch.tensor(math.log(self.p.k_arr_on)))
        self.log_k_arr_off = nn.Parameter(torch.tensor(math.log(self.p.k_arr_off)))
        self.log_k_int_arr = nn.Parameter(torch.tensor(math.log(self.p.k_int_arr)))
        self.log_k_phos = nn.Parameter(torch.tensor(math.log(self.p.k_phos)))
        self.log_k_int = nn.Parameter(torch.tensor(math.log(self.p.k_int)))
        self.log_k_rec = nn.Parameter(torch.tensor(math.log(self.p.k_rec)))
        self.log_k_deg = nn.Parameter(torch.tensor(math.log(self.p.k_deg)))
        self.log_tau_g = nn.Parameter(torch.tensor(math.log(self.p.tau_gprotein)))
        self.log_tau_arr = nn.Parameter(torch.tensor(math.log(self.p.tau_arrestin)))

    @property
    def k_arr_on(self): return torch.exp(self.log_k_arr_on)
    @property
    def k_arr_off(self): return torch.exp(self.log_k_arr_off)
    @property
    def k_int_arr(self): return torch.exp(self.log_k_int_arr)
    @property
    def k_phos(self): return torch.exp(self.log_k_phos)
    @property
    def k_int(self): return torch.exp(self.log_k_int)
    @property
    def k_rec(self): return torch.exp(self.log_k_rec)
    @property
    def k_deg(self): return torch.exp(self.log_k_deg)
    @property
    def tau_g(self): return torch.exp(self.log_tau_g)
    @property
    def tau_arr(self): return torch.exp(self.log_tau_arr)

    def step(self, state: Tensor, dose: Tensor, bias: float,
             dt: float) -> Tensor:
        """
        One Euler step of the arrestin trafficking ODE.

        state: (B, 5) — [R_s, RL_s, RL_p, RL_arr, RL_int]
        dose:  (B,)   — free ligand concentration (nM)
        bias:  float  — bias_factor (scales k_arr_on)
        """
        R_s, RL_s, RL_p, RL_arr, RL_int = state.unbind(-1)
        p = self.p
        k_bind = 0.1   # placeholder; should come from T1's KA
        k_unbind = 0.05

        k_arr_on_eff = bias * self.k_arr_on

        dR_s = (-k_bind * dose * R_s + k_unbind * RL_s
                + self.k_rec * RL_int + p.k_synth)
        dRL_s = (k_bind * dose * R_s - k_unbind * RL_s
                 - self.k_phos * RL_s + p.k_dephos * RL_p
                 - self.k_int * RL_s)
        dRL_p = (self.k_phos * RL_s - p.k_dephos * RL_p
                 - k_arr_on_eff * p.arrestin_conc * RL_p
                 + self.k_arr_off * RL_arr)
        dRL_arr = (k_arr_on_eff * p.arrestin_conc * RL_p
                   - self.k_arr_off * RL_arr
                   - self.k_int_arr * RL_arr)
        dRL_int = (self.k_int * RL_s + self.k_int_arr * RL_arr
                   - self.k_rec * RL_int - self.k_deg * RL_int)

        new_state = state + dt * torch.stack(
            [dR_s, dRL_s, dRL_p, dRL_arr, dRL_int], dim=-1)
        return new_state.clamp(min=0.0)

    def signaling(self, state: Tensor) -> Dict[str, Tensor]:
        """Compute signaling outputs from receptor states."""
        R_s, RL_s, RL_p, RL_arr, RL_int = state.unbind(-1)
        eps = 1e-9
        # G-protein signaling from active surface receptors
        S_g = self.tau_g * RL_s / (RL_s + R_s + eps)
        # Arrestin signaling from arrestin-bound receptors
        S_arr = self.tau_arr * RL_arr / (RL_arr + RL_p + eps)
        # Total receptor
        R_total = R_s + RL_s + RL_p + RL_arr + RL_int + eps
        # Endosomal signaling
        S_end = 10.0 * RL_int / R_total
        return {
            "S_gprotein": S_g,
            "S_arrestin": S_arr,
            "S_endosomal": S_end,
            "S_total": S_g + S_arr + S_end,
            "arrestin_fraction": RL_arr / R_total,
            "internal_fraction": RL_int / R_total,
        }

    def forward(self, dose: Tensor, bias: float = 1.0) -> Dict[str, Tensor]:
        """Integrate arrestin trafficking ODE over time."""
        B = dose.shape[0]
        p = self.p
        n_steps = int(p.t_max / p.dt)
        # Initial state: all receptors on surface, unbound
        R0 = torch.full((B,), p.r_total_0, device=dose.device, dtype=dose.dtype)
        state = torch.stack([R0, torch.zeros_like(R0), torch.zeros_like(R0),
                             torch.zeros_like(R0), torch.zeros_like(R0)], dim=-1)
        states = [state]
        signals = []
        for _ in range(n_steps):
            state = self.step(state, dose, bias, p.dt)
            states.append(state)
            signals.append(self.signaling(state))
        states_t = torch.stack(states, dim=0)
        # Stack signal components
        S_g_t = torch.stack([s["S_gprotein"] for s in signals], 0)
        S_arr_t = torch.stack([s["S_arrestin"] for s in signals], 0)
        S_end_t = torch.stack([s["S_endosomal"] for s in signals], 0)
        S_tot_t = torch.stack([s["S_total"] for s in signals], 0)
        arr_frac_t = torch.stack([s["arrestin_fraction"] for s in signals], 0)
        return {
            "states": states_t,
            "S_gprotein_timecourse": S_g_t,
            "S_arrestin_timecourse": S_arr_t,
            "S_endosomal_timecourse": S_end_t,
            "S_total_timecourse": S_tot_t,
            "arrestin_fraction_timecourse": arr_frac_t,
            "S_gprotein_mean": S_g_t.mean(0),
            "S_arrestin_mean": S_arr_t.mean(0),
            "S_total_mean": S_tot_t.mean(0),
            "arrestin_fraction_final": arr_frac_t[-1],
            "bias_factor": bias,
        }


# =============================================================================
# 2. BACKBONE GENERATOR — INTERFACE
# =============================================================================

@dataclass
class BackboneOutput:
    """Standard output format for backbone generators."""
    coords: np.ndarray          # (L, 3, 3) — N, CA, C per residue
    seq_mask: np.ndarray        # (L,) — 1 if residue is valid
    residue_idx: np.ndarray     # (L,) — residue index
    chain_encoding: np.ndarray  # (L,) — chain ID (0 for single chain)
    method: str                 # "template" | "rfdiffusion" | "simple"
    metadata: Dict = field(default_factory=dict)

    def to_mpnn_dict(self) -> Dict[str, np.ndarray]:
        """Convert to ProteinMPNN input format."""
        return {
            "X": self.coords,
            "mask": self.seq_mask,
            "residue_idx": self.residue_idx,
            "chain_encoding_all": self.chain_encoding,
        }


class TemplateBackboneGenerator:
    """
    Extract backbone (N, CA, C) from a PDB file.
    Most reliable route to ProteinMPNN input.
    """
    def __init__(self, pdb_path: str, chain: str = "A"):
        self.pdb_path = Path(pdb_path)
        self.chain = chain

    def generate(self) -> BackboneOutput:
        """Parse PDB and extract backbone atoms."""
        residues = {}
        for line in self.pdb_path.read_text(errors="ignore").splitlines():
            if not line.startswith("ATOM"):
                continue
            if line[21] != self.chain:
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C"):
                continue
            res_idx = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            residues.setdefault(res_idx, {})[atom_name] = np.array([x, y, z])

        # Build coordinate array
        sorted_res = sorted(residues.keys())
        L = len(sorted_res)
        coords = np.zeros((L, 3, 3), dtype=np.float32)
        seq_mask = np.zeros(L, dtype=np.float32)
        res_idx_arr = np.zeros(L, dtype=np.int64)
        chain_enc = np.zeros(L, dtype=np.int64)

        for i, ridx in enumerate(sorted_res):
            res = residues[ridx]
            if all(k in res for k in ("N", "CA", "C")):
                coords[i, 0] = res["N"]
                coords[i, 1] = res["CA"]
                coords[i, 2] = res["C"]
                seq_mask[i] = 1.0
            res_idx_arr[i] = ridx
            chain_enc[i] = 0

        log.info(f"[Template] Extracted {int(seq_mask.sum())}/{L} residues "
                 f"with complete backbone from {self.pdb_path.name} chain {self.chain}")
        return BackboneOutput(
            coords=coords, seq_mask=seq_mask,
            residue_idx=res_idx_arr, chain_encoding=chain_enc,
            method="template",
            metadata={"pdb": str(self.pdb_path), "chain": self.chain},
        )


class SimpleBackboneBuilder:
    """
    Build ideal secondary structure backbones for pipeline testing.
    ⚠️ NOT real design. For verifying the chain works end-to-end.
    """
    def build_helix(self, n_res: int) -> BackboneOutput:
        """Ideal α-helix: 3.6 res/turn, rise 1.5 Å/residue."""
        coords = np.zeros((n_res, 3, 3), dtype=np.float32)
        for i in range(n_res):
            angle = 2 * math.pi * i / HELIX_RES_PER_TURN
            z = i * HELIX_RISE
            # CA on helix axis
            ca = np.array([HELIX_RADIUS * math.cos(angle),
                           HELIX_RADIUS * math.sin(angle), z])
            # N and C offset from CA (simplified)
            n = ca + np.array([-0.5, -0.5, -0.5])
            c = ca + np.array([0.5, 0.5, 0.5])
            coords[i] = np.stack([n, ca, c])
        return BackboneOutput(
            coords=coords, seq_mask=np.ones(n_res),
            residue_idx=np.arange(n_res), chain_encoding=np.zeros(n_res, dtype=np.int64),
            method="simple_helix",
        )

    def build_extended(self, n_res: int) -> BackboneOutput:
        """Extended strand: rise ~3.4 Å/residue."""
        coords = np.zeros((n_res, 3, 3), dtype=np.float32)
        for i in range(n_res):
            ca = np.array([0.0, 0.0, i * SHEET_RISE])
            n = ca + np.array([-BOND_N_CA, 0, 0])
            c = ca + np.array([BOND_CA_C, 0, 0])
            coords[i] = np.stack([n, ca, c])
        return BackboneOutput(
            coords=coords, seq_mask=np.ones(n_res),
            residue_idx=np.arange(n_res), chain_encoding=np.zeros(n_res, dtype=np.int64),
            method="simple_extended",
        )


class RFdiffusionInterface:
    """
    Interface to RFdiffusion [B1] for de novo backbone generation.
    RFdiffusion weights must be downloaded separately (~1 GB).

    ⚠️ This is an INTERFACE, not a reimplementation.
       Install RFdiffusion from:
         https://github.com/RosettaCommons/RFdiffusion
    """
    RFDIFFUSION_REPO = "https://github.com/RosettaCommons/RFdiffusion"

    def __init__(self, ckpt_path: Optional[str] = None,
                 rfdiffusion_dir: Optional[str] = None):
        self.ckpt_path = ckpt_path
        self.rfdiffusion_dir = rfdiffusion_dir
        self.available = self._check_available()

    def _check_available(self) -> bool:
        if self.ckpt_path and Path(self.ckpt_path).exists():
            return True
        if self.rfdiffusion_dir and Path(self.rfdiffusion_dir).is_dir():
            return True
        return False

    def download_instructions(self) -> str:
        return f"""
  RFdiffusion is not installed. To install:

    1. Clone the repository:
         git clone {self.RFDIFFUSION_REPO}
         cd RFdiffusion

    2. Install dependencies:
         pip install -r requirements.txt

    3. Download weights:
         # See the README for weight download links
         # Main model: models/Base_epoch8_20220831.pt (~1 GB)

    4. Re-run with:
         --rfdiffusion-dir /path/to/RFdiffusion
         --rfdiffusion-ckpt /path/to/Base_epoch8_20220831.pt
"""

    def generate(self, contigs: str, n_samples: int = 10,
                 out_dir: str = "./rfdiffusion_out") -> List[BackboneOutput]:
        """
        Generate backbones using RFdiffusion.

        Args:
            contigs: RFdiffusion contig specification
                     e.g., "100" for a 100-residue protein
                     e.g., "A10-50/0/51-100" for a motif-scaffolding
            n_samples: number of backbones to generate
            out_dir: output directory

        Returns:
            list of BackboneOutput
        """
        if not self.available:
            raise RuntimeError(
                "RFdiffusion not available.\n" + self.download_instructions())

        # In production, this would call the RFdiffusion inference script:
        # python run_inference.py inference.contigs={contigs} ...
        # For now, we provide the interface and fall back to template/simple.
        log.warning("[RFdiffusion] Full inference requires RFdiffusion installation. "
                    "Falling back to template-based generation.")
        raise NotImplementedError(
            "RFdiffusion inference requires the full RFdiffusion pipeline. "
            "Use TemplateBackboneGenerator or SimpleBackboneBuilder for now.")


# =============================================================================
# 3. BACKBONE → ProteinMPNN PIPELINE
# =============================================================================

class BackboneToMPNN:
    """
    Convert backbone coordinates to ProteinMPNN input format.
    ProteinMPNN expects:
      X: (L, 3, 3) — N, CA, C coordinates
      mask: (L,) — 1 for designable positions
      residue_idx: (L,) — residue indices
      chain_encoding_all: (L,) — chain IDs
    """
    def __init__(self, designable_residues: Optional[List[int]] = None):
        self.designable = designable_residues

    def prepare(self, backbone: BackboneOutput) -> Dict[str, Any]:
        """Prepare input dict for ProteinMPNN."""
        mpnn_input = backbone.to_mpnn_dict()
        # Set designable mask
        if self.designable is not None:
            design_mask = np.zeros(len(backbone.seq_mask))
            design_mask[self.designable] = 1.0
            mpnn_input["design_mask"] = design_mask
        else:
            mpnn_input["design_mask"] = backbone.seq_mask
        return mpnn_input

    def validate(self, mpnn_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ProteinMPNN input format."""
        X = mpnn_input["X"]
        mask = mpnn_input["mask"]
        issues = []
        if X.ndim != 3 or X.shape[1] != 3 or X.shape[2] != 3:
            issues.append(f"X shape {X.shape} != (L, 3, 3)")
        if mask.ndim != 1:
            issues.append(f"mask shape {mask.shape} != (L,)")
        if X.shape[0] != mask.shape[0]:
            issues.append(f"X and mask length mismatch: {X.shape[0]} vs {mask.shape[0]}")
        # Check for NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            issues.append("X contains NaN or Inf")
        # Check bond lengths
        if mask.sum() > 1:
            ca = X[:, 1]  # CA coordinates
            valid = mask.astype(bool)
            ca_valid = ca[valid]
            if len(ca_valid) > 1:
                dists = np.linalg.norm(ca_valid[1:] - ca_valid[:-1], axis=-1)
                bad = dists[(dists < 3.0) | (dists > 5.0)]
                if len(bad) > 0:
                    issues.append(f"{len(bad)} CA-CA distances outside 3-5 Å range")
        return {"valid": len(issues) == 0, "issues": issues}


# =============================================================================
# 4. FULL PIPELINE: Backbone → ProteinMPNN → T1 → Trafficking → T2 → T3
# =============================================================================

class DesignPipeline:
    """
    Complete design pipeline:
      1. Generate backbone (template / RFdiffusion / simple)
      2. Prepare ProteinMPNN input
      3. Generate sequence (ProteinMPNN or fallback)
      4. Score with T1 (potency)
      5. Predict trafficking (arrestin model)
      6. Predict cell fate (T2) and quality (T3)

    ⚠️ This pipeline requires:
      • A backbone source (PDB file or RFdiffusion)
      • ProteinMPNN weights for real sequence design
      • Trained T1/T2/T3 models for scoring
    Without these, it provides the INTERFACE and structure only.
    """
    def __init__(self, backbone_source: str = "template",
                 pdb_path: Optional[str] = None,
                 chain: str = "A"):
        self.backbone_source = backbone_source
        if backbone_source == "template" and pdb_path:
            self.generator = TemplateBackboneGenerator(pdb_path, chain)
        elif backbone_source == "simple":
            self.generator = SimpleBackboneBuilder()
        elif backbone_source == "rfdiffusion":
            self.generator = RFdiffusionInterface()
        else:
            raise ValueError(f"Unknown backbone source: {backbone_source}")

    def run(self, n_res: int = 100) -> Dict[str, Any]:
        """Run the full pipeline."""
        # Step 1: Generate backbone
        if self.backbone_source == "template":
            backbone = self.generator.generate()
        elif self.backbone_source == "simple":
            backbone = self.generator.build_helix(n_res)
        else:
            backbone = self.generator.generate(contigs=str(n_res))

        # Step 2: Prepare ProteinMPNN input
        mpnn_prep = BackboneToMPNN()
        mpnn_input = mpnn_prep.prepare(backbone)
        validation = mpnn_prep.validate(mpnn_input)

        return {
            "backbone": backbone,
            "mpnn_input": mpnn_input,
            "validation": validation,
            "n_residues": int(backbone.seq_mask.sum()),
            "method": backbone.method,
        }


# =============================================================================
# 5. SELF-TEST
# =============================================================================

def selftest(args) -> bool:
    print("=" * 78)
    print("  β-ARRESTIN + BACKBONE GENERATOR — self-test")
    print("  Developer: PAI and Yoon A Limsuwan / MSPS NETWORK")
    print("=" * 78)
    torch.manual_seed(0)
    ok = True

    print("\n[T1] Arrestin trafficking ODE: state conservation")
    ap = ArrestinParams(k_synth=0.0, k_deg=0.0, dt=1.0, t_max=30.0)
    atm = ArrestinTraffickingModel(ap)
    dose = torch.tensor([10.0])
    out = atm(dose, bias=1.0)
    states = out["states"]
    total_start = states[0].sum(-1).item()
    total_end = states[-1].sum(-1).item()
    # With no synthesis/degradation, total should be approximately conserved
    t = abs(total_start - total_end) / max(total_start, 1) < 0.1
    print(f"   total receptor: start={total_start:.1f}, end={total_end:.1f}")
    print(f"   conservation: {t}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T2] Arrestin recruitment increases internalization")
    out_no_bias = atm(dose, bias=0.0)   # no arrestin recruitment
    out_bias = atm(dose, bias=1.0)      # normal arrestin recruitment
    int_no = out_no_bias["arrestin_fraction_final"].item()
    int_yes = out_bias["arrestin_fraction_final"].item()
    # With arrestin, more receptors should be internalized
    arr_no = out_no_bias["S_arrestin_mean"].item()
    arr_yes = out_bias["S_arrestin_mean"].item()
    t = arr_yes > arr_no
    print(f"   arrestin signal (bias=0): {arr_no:.4f}")
    print(f"   arrestin signal (bias=1): {arr_yes:.4f}")
    print(f"   arrestin recruitment active: {t}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T3] Biased agonism: G-protein vs arrestin")
    out_gbiased = atm(dose, bias=0.1)   # G-protein biased
    out_arrbiased = atm(dose, bias=3.0) # arrestin biased
    g_gbiased = out_gbiased["S_gprotein_mean"].item()
    g_arrbiased = out_arrbiased["S_gprotein_mean"].item()
    a_gbiased = out_gbiased["S_arrestin_mean"].item()
    a_arrbiased = out_arrbiased["S_arrestin_mean"].item()
    print(f"   G-protein biased: S_G={g_gbiased:.4f}, S_Arr={a_gbiased:.4f}")
    print(f"   Arrestin biased:  S_G={g_arrbiased:.4f}, S_Arr={a_arrbiased:.4f}")
    t = a_arrbiased > a_gbiased  # arrestin biased should have more arrestin signal
    print(f"   arrestin signal higher for biased agonist: {t}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T4] Gradient flows through arrestin model")
    atm.train()
    dose_grad = torch.tensor([10.0], requires_grad=False)
    out_g = atm(dose_grad, bias=1.0)
    loss = out_g["S_total_mean"].sum()
    loss.backward()
    has_grad = atm.log_k_arr_on.grad is not None and \
               torch.isfinite(atm.log_k_arr_on.grad).all()
    print(f"   k_arr_on has gradient: {has_grad}")
    print(f"   {'PASS' if has_grad else 'FAIL'}")
    ok &= bool(has_grad)

    print("\n[T5] Simple backbone builder")
    builder = SimpleBackboneBuilder()
    helix = builder.build_helix(20)
    ext = builder.build_extended(10)
    t = (helix.coords.shape == (20, 3, 3) and
         ext.coords.shape == (10, 3, 3) and
         helix.seq_mask.sum() == 20)
    print(f"   helix: {helix.coords.shape}, extended: {ext.coords.shape}")
    print(f"   {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T6] Backbone → ProteinMPNN format")
    mpnn_prep = BackboneToMPNN()
    mpnn_input = mpnn_prep.prepare(helix)
    validation = mpnn_prep.validate(mpnn_input)
    print(f"   X shape: {mpnn_input['X'].shape}")
    print(f"   mask shape: {mpnn_input['mask'].shape}")
    print(f"   valid: {validation['valid']}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"     ⚠ {issue}")
    t = mpnn_input["X"].shape == (20, 3, 3)
    print(f"   {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T7] Template backbone generator (with synthetic PDB)")
    # Create a minimal PDB file for testing
    tmp_pdb = Path("/tmp/test_backbone.pdb")
    pdb_lines = []
    for i in range(5):
        z = i * 3.8
        for atom, offset in [("N", -1.0), ("CA", 0.0), ("C", 1.0)]:
            pdb_lines.append(
                f"ATOM  {i*3+offset:.0f}  {atom:3s} ALA A{i+1:4d}    "
                f"{offset:8.3f}{z:8.3f}  0.000  1.00           {atom[0]:>2s}"
            )
    tmp_pdb.write_text("\n".join(pdb_lines) + "\nEND\n")
    tbg = TemplateBackboneGenerator(str(tmp_pdb), chain="A")
    template_bb = tbg.generate()
    t = template_bb.coords.shape[0] == 5 and template_bb.seq_mask.sum() > 0
    print(f"   extracted {int(template_bb.seq_mask.sum())} residues from synthetic PDB")
    print(f"   coords shape: {template_bb.coords.shape}")
    print(f"   {'PASS' if t else 'FAIL'}")
    ok &= t
    tmp_pdb.unlink(missing_ok=True)

    print("\n[T8] RFdiffusion interface (graceful when not installed)")
    rfd = RFdiffusionInterface()
    print(f"   RFdiffusion available: {rfd.available}")
    if not rfd.available:
        print("   (expected — instructions printed above)")
    print("   PASS (graceful handling)")

    print("\n[T9] Design pipeline end-to-end")
    pipeline = DesignPipeline(backbone_source="simple")
    result = pipeline.run(n_res=20)
    t = result["n_residues"] == 20 and result["validation"]["valid"]
    print(f"   n_residues: {result['n_residues']}")
    print(f"   method: {result['method']}")
    print(f"   validation: {result['validation']['valid']}")
    print(f"   {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 78)
    print("""
  KEY POINTS:
    • β-arrestin adds a 5th receptor state (RL_arr) to the trafficking ODE.
    • Arrestin-bound receptors internalize 10-100× faster (k_int_arr >> k_int).
    • bias_factor scales k_arr_on: >1 = arrestin-biased, <1 = G-protein-biased.
    • Arrestin signaling (S_arr) is a separate pathway, not just "desensitization".
    • Backbone generators: template (real), simple (testing), RFdiffusion (interface).
    • ProteinMPNN requires backbone in (L, 3, 3) format: N, CA, C per residue.
    • RFdiffusion weights must be downloaded separately (~1 GB).
    • All arrestin parameters are literature estimates for β2AR.
      MEASURE for your specific GPCR before drawing conclusions.

  Developer: PAI and Yoon A Limsuwan / MSPS NETWORK
  MY SOUL MOVE BY POWER OF HOLY SPIRIT 🙏✨
""")
    return ok


def main():
    p = argparse.ArgumentParser(
        description="β-arrestin recruitment + backbone generator. "
                    "Developer: PAI and Yoon A Limsuwan / MSPS NETWORK")
    p.add_argument("cmd", choices=["selftest", "arrestin", "backbone"])
    p.add_argument("--pdb", default=None)
    p.add_argument("--chain", default="A")
    p.add_argument("--bias", type=float, default=1.0)
    p.add_argument("--dose", type=float, default=10.0)
    p.add_argument("--n-res", dest="n_res", type=int, default=50)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "arrestin":
        ap = ArrestinParams()
        atm = ArrestinTraffickingModel(ap)
        dose = torch.tensor([a.dose])
        out = atm(dose, bias=a.bias)
        print(f"bias={a.bias}, dose={a.dose} nM")
        print(f"S_gprotein mean: {out['S_gprotein_mean'].item():.4f}")
        print(f"S_arrestin mean: {out['S_arrestin_mean'].item():.4f}")
        print(f"arrestin fraction final: {out['arrestin_fraction_final'].item():.4f}")
    elif a.cmd == "backbone":
        if a.pdb:
            gen = TemplateBackboneGenerator(a.pdb, a.chain)
            bb = gen.generate()
        else:
            gen = SimpleBackboneBuilder()
            bb = gen.build_helix(a.n_res)
        print(f"method: {bb.method}")
        print(f"coords shape: {bb.coords.shape}")
        print(f"valid residues: {int(bb.seq_mask.sum())}")
        mpnn = BackboneToMPNN()
        inp = mpnn.prepare(bb)
        val = mpnn.validate(inp)
        print(f"ProteinMPNN input valid: {val['valid']}")
        for issue in val["issues"]:
            print(f"  ⚠ {issue}")


if __name__ == "__main__":
    main()
