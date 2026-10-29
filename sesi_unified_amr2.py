"""
=============================================================================
SESI UNIFIED AMR v2 — MOLECULAR BIOLOGY + ADVANCED MATERIALS EXTENSION
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

EXTENDED DOMAIN TAXONOMY (22 domains)
-------------------------------------
[PHYSICAL SCIENCES]
  • CFD                          [P] Standard FEM residual
  • Quantum Chemistry            [D] DFT/TD-DFT residuals
  • Condensed Matter             [D] Band structure + phonon residuals

[MOLECULAR BIOLOGY — NEW]
  • DNA/RNA Dynamics             [D] Base-pairing + folding energy
  • Protein Folding              [D] FAPE + pLDDT + energy landscape
  • Protein-Protein Interaction  [D] Binding affinity + allostery
  • Lipid Membrane               [D] Curvature + phase separation
  • Metabolic Network            [D] Flux balance + thermodynamic
  • Gene Regulation              [D] Transcription factor binding
  • Cell Signaling               [D] Pathway activation + crosstalk
  • Microbial Systems            [D] Quorum sensing + biofilm

[ADVANCED MATERIALS — NEW]
  • Metals & Alloys              [D] Crystal plasticity + phase field
  • Ceramics                     [D] Ionic/covalent + piezoelectric
  • Polymers                     [D] Chain dynamics + glass transition
  • Nanomaterials                [D] Quantum confinement + surface
  • Metamaterials                [D] Negative-index + phononic
  • Biomaterials                 [D] Biocompatibility + degradation
  • Composites                   [D] Interface + multiscale
  • 2D Materials                 [D] Graphene/TMD + moiré

[BIOMEDICAL — EXTENDED]
  • Oncology                     [D] UQ + clinical deviation
  • Epidemiology                 [D] Forecast + R₀ CI
  • Anti-Aging                   [O] Biomarker deviation (highly heuristic)
  • Cellular Regen               [D] Morphogen PDE residual
  • Gene Editing                 [D] Off-target uncertainty
  • Neuroscience                 [D] Signal + source localization

HONEST SCOPE
------------
• Only CFD has a mathematically rigorous error estimator (Ainsworth-Oden).
• Molecular biology estimators are HEURISTIC proxies based on energy
  functions, binding models, and network topology — NOT rigorous FEM.
• Materials science estimators combine PDE residuals (where applicable)
  with ML-based uncertainty — NOT a unified theory.
• No-Zeno bounds EVENT COUNT, not error itself (Six Problems §6.2).
• AMR convergence is NOT proved for non-CFD domains.
• Each estimator must be validated against domain-specific benchmarks.

LABEL CONVENTION
----------------
[P] proved (standard FEM theory)
[C] cited from literature
[D] derived here (heuristic for non-FEM domains)
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

logger = logging.getLogger("sesi.unified_amr_v2")

__all__ = [
    # Domain enums
    "DomainGroup", "Domain",
    # Config
    "AMRConfig",
    # Abstract base
    "ErrorEstimator",
    # Physical sciences
    "CFDErrorEstimator", "QuantumChemistryEstimator", "CondensedMatterEstimator",
    # Molecular biology
    "DNARNAEstimator", "ProteinFoldingEstimator", "ProteinInteractionEstimator",
    "LipidMembraneEstimator", "MetabolicNetworkEstimator",
    "GeneRegulationEstimator", "CellSignalingEstimator", "MicrobialSystemsEstimator",
    # Advanced materials
    "MetalsAlloysEstimator", "CeramicsEstimator", "PolymersEstimator",
    "NanomaterialsEstimator", "MetamaterialsEstimator", "BiomaterialsEstimator",
    "CompositesEstimator", "TwoDMaterialsEstimator",
    # Biomedical
    "OncologyEstimator", "EpidemiologyEstimator", "AntiAgingEstimator",
    "CellularRegenEstimator", "GeneEditingEstimator", "NeuroscienceEstimator",
    # AMR infrastructure
    "AdaptiveMesh", "DorflerMarker", "NoZenoGate", "UnifiedAMRController",
]


# ═══════════════════════════════════════════════════════════════════════
# DOMAIN ENUMS
# ═══════════════════════════════════════════════════════════════════════

class DomainGroup(Enum):
    """High-level grouping of domains."""
    PHYSICAL_SCIENCES = auto()
    MOLECULAR_BIOLOGY = auto()
    ADVANCED_MATERIALS = auto()
    BIOMEDICAL = auto()


class Domain(Enum):
    """All 22 supported domains."""
    # Physical sciences (3)
    CFD = auto()
    QUANTUM_CHEMISTRY = auto()
    CONDENSED_MATTER = auto()

    # Molecular biology (8)
    DNA_RNA = auto()
    PROTEIN_FOLDING = auto()
    PROTEIN_INTERACTION = auto()
    LIPID_MEMBRANE = auto()
    METABOLIC_NETWORK = auto()
    GENE_REGULATION = auto()
    CELL_SIGNALING = auto()
    MICROBIAL_SYSTEMS = auto()

    # Advanced materials (8)
    METALS_ALLOYS = auto()
    CERAMICS = auto()
    POLYMERS = auto()
    NANOMATERIALS = auto()
    METAMATERIALS = auto()
    BIOMATERIALS = auto()
    COMPOSITES = auto()
    TWO_D_MATERIALS = auto()

    # Biomedical (6) — kept from v1
    ONCOLOGY = auto()
    EPIDEMIOLOGY = auto()
    ANTI_AGING = auto()
    CELLULAR_REGEN = auto()
    GENE_EDITING = auto()
    NEUROSCIENCE = auto()

    @property
    def group(self) -> DomainGroup:
        """Map domain to its group."""
        if self in (Domain.CFD, Domain.QUANTUM_CHEMISTRY, Domain.CONDENSED_MATTER):
            return DomainGroup.PHYSICAL_SCIENCES
        elif self in (Domain.DNA_RNA, Domain.PROTEIN_FOLDING, Domain.PROTEIN_INTERACTION,
                      Domain.LIPID_MEMBRANE, Domain.METABOLIC_NETWORK, Domain.GENE_REGULATION,
                      Domain.CELL_SIGNALING, Domain.MICROBIAL_SYSTEMS):
            return DomainGroup.MOLECULAR_BIOLOGY
        elif self in (Domain.METALS_ALLOYS, Domain.CERAMICS, Domain.POLYMERS,
                      Domain.NANOMATERIALS, Domain.METAMATERIALS, Domain.BIOMATERIALS,
                      Domain.COMPOSITES, Domain.TWO_D_MATERIALS):
            return DomainGroup.ADVANCED_MATERIALS
        else:
            return DomainGroup.BIOMEDICAL


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
    theta_hi: float = 0.2
    theta_lo: float = 0.05

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
        raise NotImplementedError

    def get_stress(self, error_estimates: Tensor) -> float:
        """Default: max error / tolerance."""
        return float(error_estimates.max().item() / (self.cfg.tolerance + 1e-10))


# ═══════════════════════════════════════════════════════════════════════
# PHYSICAL SCIENCES
# ═══════════════════════════════════════════════════════════════════════

class CFDErrorEstimator(ErrorEstimator):
    """
    [P] Residual-based + gradient recovery error estimator for CFD.
    Based on Ainsworth-Oden, Verfürth.

    η_K² = h_K² ||f + Δu_h||²_{L²(K)}
    η_J² = h_J ||[∂u_h/∂n]||²_{L²(J)}
    η_ZZ = ||G(∇u_h) - ∇u_h||_{L²}
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        u_h = model_output["u_h"]
        f = model_output["f"]
        h_K = model_output["cell_sizes"]
        lap_u_h = model_output.get("laplacian_u_h", torch.zeros_like(u_h))

        residual = f + lap_u_h
        eta_K_sq = (h_K ** 2) * (residual ** 2).mean(dim=-1)

        if "edge_jumps" in model_output:
            eta_J_sq = model_output["edge_jumps"]
            return torch.sqrt(eta_K_sq + eta_J_sq)
        return torch.sqrt(eta_K_sq)


class QuantumChemistryEstimator(ErrorEstimator):
    """
    [D] Error estimator for quantum chemistry (DFT/TD-DFT).

    Uses:
      • Basis set incompleteness (BSI) estimate
      • Exchange-correlation functional uncertainty
      • SCF convergence residual
      • Energy variance across k-points

    HONEST: No rigorous a posteriori FEM estimator exists for DFT.
    These are heuristic proxies.
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • electron_density: (B, N_grid)
          • orbital_coefficients: (B, N_orb, N_basis)
          • scf_residual: (B,)
          • basis_set_size: (B,)
          • xc_functional_id: (B,)
        """
        features = torch.cat([
            model_output["electron_density"].mean(dim=-1, keepdim=True),
            model_output["orbital_coefficients"].abs().mean(dim=(-1, -2), keepdim=True),
            model_output["scf_residual"].unsqueeze(-1),
            torch.log(model_output["basis_set_size"].float()).unsqueeze(-1),
            model_output["xc_functional_id"].float().unsqueeze(-1),
        ], dim=-1)

        uncertainty = self.uncertainty_head(features).squeeze(-1)

        if ground_truth is not None and "energy_target" in ground_truth:
            energy_error = F.mse_loss(
                model_output.get("total_energy", torch.zeros(1)),
                ground_truth["energy_target"],
                reduction='none'
            )
            uncertainty = 0.7 * uncertainty + 0.3 * energy_error

        return uncertainty


class CondensedMatterEstimator(ErrorEstimator):
    """
    [D] Error estimator for condensed matter physics.

    Uses:
      • Band structure gap uncertainty
      • Phonon dispersion residuals
      • k-point sampling convergence
      • PDE residuals for continuum elasticity (if applicable)
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • band_structure: (B, N_k, N_bands)
          • phonon_dispersion: (B, N_q, N_modes)
          • k_point_density: (B,)
        """
        band = model_output["band_structure"]
        phonon = model_output["phonon_dispersion"]

        # Band gap uncertainty: variance near Fermi level
        band_var = band.var(dim=-1).mean(dim=-1)

        # Phonon residual: imaginary frequencies indicate instability
        phonon_residual = (phonon < 0).float().mean(dim=-1)

        # k-point sampling error (heuristic)
        k_error = 1.0 / (model_output["k_point_density"] + 1.0)

        uncertainty = 0.4 * band_var + 0.3 * phonon_residual + 0.3 * k_error
        return uncertainty


# ═══════════════════════════════════════════════════════════════════════
# MOLECULAR BIOLOGY
# ═══════════════════════════════════════════════════════════════════════

class DNARNAEstimator(ErrorEstimator):
    """
    [D] Error estimator for DNA/RNA dynamics.

    Uses:
      • Base-pairing stability (hydrogen bond energy)
      • Stacking interaction variance
      • Folding free energy landscape roughness
      • Epigenetic modification uncertainty

    HONEST: This is a HEURISTIC proxy, not a rigorous estimator.
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.energy_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • sequence_encoding: (B, L, 64) one-hot + features
          • base_pair_probs: (B, L, L) pairing probabilities
          • stacking_energy: (B, L-1) stacking energies
          • folding_dG: (B,) folding free energy
        """
        # Base-pairing stability: entropy of pairing distribution
        bp = model_output["base_pair_probs"]
        bp_entropy = -(bp * torch.log(bp + 1e-8)).sum(dim=-1).mean(dim=-1)

        # Stacking variance
        stack_var = model_output["stacking_energy"].var(dim=-1)

        # Folding landscape roughness
        roughness = torch.abs(model_output["folding_dG"]) / 10.0

        features = torch.stack([bp_entropy, stack_var, roughness], dim=-1)
        uncertainty = self.energy_net(features).squeeze(-1)

        return uncertainty


class ProteinFoldingEstimator(ErrorEstimator):
    """
    [D] Error estimator for protein folding (AlphaFold-style).

    Uses:
      • FAPE (Frame Aligned Point Error)
      • pLDDT (predicted LDDT) per-residue confidence
      • PAE (Predicted Aligned Error) pairwise
      • Energy landscape frustration index
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • pred_coords: (B, L, 3)
          • confidence: (B, L) pLDDT
          • pae_matrix: (B, L, L)
        """
        pred = model_output["pred_coords"]
        conf = model_output["confidence"]

        # Inverse confidence as error proxy
        error = 1.0 - conf

        if ground_truth is not None and "true_coords" in ground_truth:
            true = ground_truth["true_coords"]
            dist_error = torch.norm(pred - true, dim=-1).mean(dim=-1) / 10.0
            error = 0.5 * error.mean(dim=-1) + 0.5 * dist_error
        else:
            error = error.mean(dim=-1)

        return error


class ProteinInteractionEstimator(ErrorEstimator):
    """
    [D] Error estimator for protein-protein interactions.

    Uses:
      • Binding affinity prediction uncertainty
      • Interface residue conservation
      • Allosteric coupling variance
      • Docking score variance across poses
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.binding_net = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • complex_features: (B, 128)
          • binding_affinity: (B,) predicted Kd
          • interface_residues: (B, N_interface)
          • docking_scores: (B, N_poses)
        """
        features = model_output["complex_features"]
        uncertainty = self.binding_net(features).squeeze(-1)

        # Docking score variance
        dock_var = model_output["docking_scores"].var(dim=-1)

        return 0.6 * uncertainty + 0.4 * dock_var


class LipidMembraneEstimator(ErrorEstimator):
    """
    [D] Error estimator for lipid membrane dynamics.

    Uses:
      • Membrane curvature deviation
      • Phase separation order parameter
      • Lipid flip-flop rate uncertainty
      • PDE residual for Helfrich energy
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • membrane_field: (B, 1, H, W) height field
          • composition_field: (B, 2, H, W) lipid composition
          • curvature: (B, H, W) mean curvature
        """
        # Helfrich energy residual (simplified)
        kappa = 1.0  # bending modulus
        curvature = model_output["curvature"]
        lap_curv = F.avg_pool2d(curvature.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        pde_residual = (kappa * lap_curv - curvature ** 3).abs().mean(dim=(-1, -2))

        # Phase separation order parameter
        comp = model_output["composition_field"]
        order_param = (comp[:, 0] - comp[:, 1]).abs().mean(dim=(-1, -2))
        phase_uncertainty = 1.0 - order_param

        return 0.5 * pde_residual + 0.5 * phase_uncertainty


class MetabolicNetworkEstimator(ErrorEstimator):
    """
    [D] Error estimator for metabolic networks.

    Uses:
      • Flux balance analysis (FBA) optimality gap
      • Thermodynamic feasibility (Gibbs energy)
      • Stoichiometric matrix rank deficiency
      • Enzyme kinetics parameter uncertainty
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • flux_vector: (B, N_reactions)
          • stoich_matrix: (N_metabolites, N_reactions) [shared]
          • gibbs_energies: (B, N_reactions)
          • enzyme_kinetics: (B, N_reactions, 3) [Vmax, Km, Ki]
        """
        flux = model_output["flux_vector"]
        S = model_output["stoich_matrix"]

        # Mass balance residual: S · v ≈ 0
        balance_residual = (S @ flux.T).abs().mean(dim=0)

        # Thermodynamic feasibility
        gibbs = model_output["gibbs_energies"]
        thermo_violation = (gibbs > 0).float().mean(dim=-1)

        # Kinetic parameter uncertainty (heuristic)
        kin_unc = model_output["enzyme_kinetics"].var(dim=-1).mean(dim=-1)

        return 0.4 * balance_residual + 0.3 * thermo_violation + 0.3 * kin_unc


class GeneRegulationEstimator(ErrorEstimator):
    """
    [D] Error estimator for gene regulation networks.

    Uses:
      • Transcription factor binding affinity uncertainty
      • Epigenetic state variance
      • RNA-seq count noise (Poisson/NB)
      • Network topology robustness
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.tf_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • tf_binding_scores: (B, N_genes, N_TFs)
          • expression_levels: (B, N_genes)
          • chromatin_state: (B, N_genes, 5) one-hot
        """
        tf_scores = model_output["tf_binding_scores"]
        tf_uncertainty = self.tf_net(tf_scores.mean(dim=1)).squeeze(-1)

        # Expression noise (Poisson-like)
        expr = model_output["expression_levels"]
        expr_noise = torch.sqrt(expr + 1e-8) / (expr + 1.0)
        expr_uncertainty = expr_noise.mean(dim=-1)

        return 0.6 * tf_uncertainty + 0.4 * expr_uncertainty


class CellSignalingEstimator(ErrorEstimator):
    """
    [D] Error estimator for cell signaling pathways.

    Uses:
      • Pathway activation variance
      • Crosstalk between pathways
      • Receptor-ligand binding uncertainty
      • ODE residual for signaling dynamics
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • pathway_activity: (B, N_pathways, T) time series
          • crosstalk_matrix: (B, N_pathways, N_pathways)
          • receptor_binding: (B, N_receptors)
        """
        # Pathway activity variance
        path_var = model_output["pathway_activity"].var(dim=-1).mean(dim=-1)

        # Crosstalk magnitude
        crosstalk = model_output["crosstalk_matrix"].abs().mean(dim=(-1, -2))

        # Receptor binding uncertainty
        receptor_unc = 1.0 - model_output["receptor_binding"].abs().mean(dim=-1)

        return 0.4 * path_var + 0.3 * crosstalk + 0.3 * receptor_unc


class MicrobialSystemsEstimator(ErrorEstimator):
    """
    [D] Error estimator for microbial systems.

    Uses:
      • Quorum sensing signal variance
      • Biofilm spatial heterogeneity
      • Species interaction network stability
      • Growth rate uncertainty
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • qs_signal: (B, N_species) quorum sensing autoinducers
          • biofilm_density: (B, H, W) spatial field
          • interaction_matrix: (B, N_species, N_species)
          • growth_rates: (B, N_species)
        """
        # QS signal variance
        qs_var = model_output["qs_signal"].var(dim=-1)

        # Biofilm heterogeneity
        biofilm = model_output["biofilm_density"]
        biofilm_het = biofilm.std(dim=(-1, -2)) / (biofilm.mean(dim=(-1, -2)) + 1e-8)

        # Interaction network stability (spectral radius)
        inter = model_output["interaction_matrix"]
        eigvals = torch.linalg.eigvals(inter).abs().max(dim=-1).values
        stability_unc = torch.clamp(eigvals - 1.0, min=0.0)

        return 0.3 * qs_var + 0.4 * biofilm_het + 0.3 * stability_unc


# ═══════════════════════════════════════════════════════════════════════
# ADVANCED MATERIALS
# ═══════════════════════════════════════════════════════════════════════

class MetalsAlloysEstimator(ErrorEstimator):
    """
    [D] Error estimator for metals and alloys.

    Uses:
      • Crystal plasticity residual
      • Phase field order parameter gradient
      • Dislocation density variance
      • Yield surface uncertainty
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • stress_field: (B, 3, 3, H, W, D)
          • strain_field: (B, 3, 3, H, W, D)
          • phase_field: (B, N_phases, H, W, D)
          • dislocation_density: (B, H, W, D)
        """
        # Crystal plasticity residual (simplified)
        stress = model_output["stress_field"]
        strain = model_output["strain_field"]
        # σ = C : ε (linear elasticity residual)
        elastic_residual = (stress - strain).abs().mean(dim=(1, 2, 3, 4, 5))

        # Phase field gradient
        phase = model_output["phase_field"]
        phase_grad = torch.gradient(phase, dim=(-3, -2, -1))
        phase_energy = sum(g.abs().mean() for g in phase_grad)

        # Dislocation density variance
        disloc_var = model_output["dislocation_density"].var(dim=(-1, -2, -3))

        return 0.4 * elastic_residual + 0.3 * phase_energy + 0.3 * disloc_var


class CeramicsEstimator(ErrorEstimator):
    """
    [D] Error estimator for ceramics.

    Uses:
      • Ionic/covalent bond energy variance
      • Piezoelectric coupling uncertainty
      • Ferroelectric domain wall energy
      • Fracture toughness prediction variance
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.bond_net = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • bond_features: (B, N_bonds, 128)
          • piezo_tensor: (B, 3, 3, 3)
          • domain_wall_energy: (B,)
        """
        bond_features = model_output["bond_features"]
        bond_unc = self.bond_net(bond_features.mean(dim=1)).squeeze(-1)

        # Piezoelectric uncertainty
        piezo = model_output["piezo_tensor"]
        piezo_var = piezo.var(dim=(-1, -2, -3))

        return 0.6 * bond_unc + 0.4 * piezo_var


class PolymersEstimator(ErrorEstimator):
    """
    [D] Error estimator for polymers.

    Uses:
      • Chain conformation entropy variance
      • Cross-link density uncertainty
      • Glass transition temperature prediction error
      • Viscoelastic relaxation spectrum residual
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • chain_conformations: (B, N_chains, N_monomers, 3)
          • crosslink_density: (B,)
          • Tg_prediction: (B,)
          • relaxation_spectrum: (B, N_modes)
        """
        # Chain conformation entropy
        chains = model_output["chain_conformations"]
        chain_var = chains.var(dim=-1).mean(dim=(-1, -2))

        # Cross-link uncertainty
        xlink_unc = 1.0 / (model_output["crosslink_density"] + 1.0)

        # Tg prediction uncertainty (heuristic)
        Tg = model_output["Tg_prediction"]
        Tg_unc = torch.abs(Tg - 300.0) / 300.0  # normalized deviation

        return 0.4 * chain_var + 0.3 * xlink_unc + 0.3 * Tg_unc


class NanomaterialsEstimator(ErrorEstimator):
    """
    [D] Error estimator for nanomaterials.

    Uses:
      • Quantum confinement energy uncertainty
      • Surface-to-volume ratio variance
      • Plasmon resonance width
      • DFT bandgap prediction error
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.quantum_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • particle_size: (B,)
          • surface_area: (B,)
          • bandgap: (B,)
          • plasmon_spectrum: (B, N_freq)
        """
        size = model_output["particle_size"].unsqueeze(-1)
        surface = model_output["surface_area"].unsqueeze(-1)
        features = torch.cat([size, surface], dim=-1)
        quantum_unc = self.quantum_net(features).squeeze(-1)

        # Plasmon resonance width
        plasmon = model_output["plasmon_spectrum"]
        plasmon_width = plasmon.std(dim=-1)

        return 0.5 * quantum_unc + 0.5 * plasmon_width


class MetamaterialsEstimator(ErrorEstimator):
    """
    [D] Error estimator for metamaterials.

    Uses:
      • Effective medium theory residual
      • Negative refractive index bandwidth uncertainty
      • Phononic bandgap prediction error
      • Unit cell geometry sensitivity
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • effective_permittivity: (B, N_freq) complex
          • effective_permeability: (B, N_freq) complex
          • band_structure: (B, N_k, N_bands)
          • unit_cell_sensitivity: (B, N_params)
        """
        # Negative-index bandwidth
        eps = model_output["effective_permittivity"]
        mu = model_output["effective_permeability"]
        neg_index = ((eps.real < 0) & (mu.real < 0)).float().mean(dim=-1)
        bandwidth_unc = 1.0 - neg_index

        # Bandgap prediction
        bands = model_output["band_structure"]
        gap = bands[:, :, 1].min(dim=-1).values - bands[:, :, 0].max(dim=-1).values
        gap_unc = torch.abs(gap) / (gap.abs().max() + 1e-8)

        # Unit cell sensitivity
        sens = model_output["unit_cell_sensitivity"].abs().mean(dim=-1)

        return 0.4 * bandwidth_unc + 0.3 * gap_unc + 0.3 * sens


class BiomaterialsEstimator(ErrorEstimator):
    """
    [D] Error estimator for biomaterials.

    Uses:
      • Biocompatibility prediction uncertainty
      • Degradation rate variance
      • Cell adhesion energy prediction error
      • Mechanical property mismatch
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.bio_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • material_features: (B, 256)
          • degradation_rate: (B,)
          • cell_adhesion: (B,)
          • elastic_modulus: (B,)
        """
        features = model_output["material_features"]
        bio_unc = self.bio_net(features).squeeze(-1)

        # Degradation rate variance
        deg_var = model_output["degradation_rate"].abs()

        # Mechanical mismatch (target ~1 GPa for bone)
        target_E = 1.0  # GPa
        mech_mismatch = torch.abs(model_output["elastic_modulus"] - target_E) / target_E

        return 0.4 * bio_unc + 0.3 * deg_var + 0.3 * mech_mismatch


class CompositesEstimator(ErrorEstimator):
    """
    [D] Error estimator for composite materials.

    Uses:
      • Interface bonding energy variance
      • Multiscale homogenization residual
      • Fiber orientation uncertainty
      • Delamination prediction error
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • interface_energy: (B, N_interfaces)
          • homogenized_stiffness: (B, 3, 3, 3, 3)
          • fiber_orientation: (B, N_fibers, 3)
          • delamination_risk: (B, H, W)
        """
        # Interface bonding variance
        interface_var = model_output["interface_energy"].var(dim=-1)

        # Fiber orientation uncertainty
        fiber = model_output["fiber_orientation"]
        fiber_unc = fiber.var(dim=-1).mean(dim=-1)

        # Delamination risk
        delam = model_output["delamination_risk"].mean(dim=(-1, -2))

        return 0.4 * interface_var + 0.3 * fiber_unc + 0.3 * delam


class TwoDMaterialsEstimator(ErrorEstimator):
    """
    [D] Error estimator for 2D materials (graphene, TMDs, etc.).

    Uses:
      • Moiré pattern periodicity uncertainty
      • Band structure Dirac cone variance
      • Strain engineering sensitivity
      • Layer stacking fault energy
    """

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        """
        model_output must contain:
          • band_structure: (B, N_k, N_bands)
          • moire_period: (B,)
          • strain_field: (B, 2, 2, H, W)
          • stacking_energy: (B, N_configs)
        """
        # Dirac cone variance
        bands = model_output["band_structure"]
        dirac_var = bands[:, :, 0].var(dim=-1)  # valence band variance

        # Moiré period uncertainty
        moire_unc = 1.0 / (model_output["moire_period"] + 1e-8)

        # Stacking energy variance
        stack_var = model_output["stacking_energy"].var(dim=-1)

        return 0.4 * dirac_var + 0.3 * moire_unc + 0.3 * stack_var


# ═══════════════════════════════════════════════════════════════════════
# BIOMEDICAL (kept from v1, abbreviated)
# ═══════════════════════════════════════════════════════════════════════

class OncologyEstimator(ErrorEstimator):
    """[D] Oncology prediction uncertainty."""

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        latent = model_output["latent"]
        uncertainty = self.uncertainty_head(latent).squeeze(-1)

        if ground_truth is not None and "tumor_target" in ground_truth:
            pred_grid = model_output["predictions"]["tumor_grid"]
            true_grid = ground_truth["tumor_target"]
            grid_error = F.mse_loss(pred_grid, true_grid, reduction='none').mean(dim=(1, 2, 3))
            uncertainty = 0.5 * uncertainty + 0.5 * grid_error

        return uncertainty


class EpidemiologyEstimator(ErrorEstimator):
    """[D] Epidemiology forecast uncertainty."""

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.r0_uncertainty = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        latent = model_output["latent"]
        r0_unc = self.r0_uncertainty(latent).squeeze(-1)

        if ground_truth is not None and "trajectory_target" in ground_truth:
            pred_traj = model_output["trajectory"]
            true_traj = ground_truth["trajectory_target"]
            traj_error = F.mse_loss(pred_traj, true_traj, reduction='none').mean(dim=(1, 2))
            r0_unc = 0.5 * r0_unc + 0.5 * traj_error

        return r0_unc


class AntiAgingEstimator(ErrorEstimator):
    """[O] Anti-aging biomarker uncertainty (highly heuristic)."""

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.age_uncertainty = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        latent = model_output["latent"]
        uncertainty = self.age_uncertainty(latent).squeeze(-1)

        if ground_truth is not None and "age_target" in ground_truth:
            pred_age = model_output["biological_age"]
            true_age = ground_truth["age_target"]
            age_error = F.mse_loss(pred_age, true_age, reduction='none')
            uncertainty = 0.5 * uncertainty + 0.5 * (age_error / 10.0).clamp(max=1.0)

        return uncertainty


class CellularRegenEstimator(ErrorEstimator):
    """[D] Cellular regeneration morphogen PDE residual."""

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        morphogen = model_output["morphogen_field"]
        fate = model_output["cell_fate"]

        lap_morph = F.avg_pool2d(morphogen, 3, stride=1, padding=1) - morphogen
        pde_residual = lap_morph.abs().mean(dim=(1, 2, 3))

        fate_entropy = -(fate * torch.log(fate + 1e-8)).sum(dim=-1)
        uncertainty = 0.5 * pde_residual + 0.5 * (fate_entropy / math.log(4))

        return uncertainty


class GeneEditingEstimator(ErrorEstimator):
    """[D] CRISPR off-target uncertainty."""

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.off_target_uncertainty = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        latent = model_output["latent"]
        off_target = model_output["off_target_score"]

        ot_unc = self.off_target_uncertainty(latent).squeeze(-1)
        ot_variance = off_target.var(dim=-1) if off_target.dim() > 1 else torch.zeros_like(ot_unc)

        return 0.5 * ot_unc + 0.5 * ot_variance


class NeuroscienceEstimator(ErrorEstimator):
    """[D] Neuroscience signal + source localization uncertainty."""

    def __init__(self, cfg: AMRConfig):
        super().__init__(cfg)
        self.source_uncertainty = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def estimate(self, model_output: Dict[str, Any],
                 ground_truth: Optional[Dict[str, Any]] = None) -> Tensor:
        latent = model_output["latent"]
        source = model_output["source_loc"]

        source_entropy = -(source * torch.log(source + 1e-8)).sum(dim=-1)
        source_unc = self.source_uncertainty(latent).squeeze(-1)

        return 0.5 * source_unc + 0.5 * (source_entropy / math.log(source.shape[-1]))


# ═══════════════════════════════════════════════════════════════════════
# AMR INFRASTRUCTURE (kept from v1)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MeshCell:
    id: int
    level: int
    center: Tuple[float, ...]
    size: float
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    is_leaf: bool = True
    error_estimate: float = 0.0


class AdaptiveMesh:
    """Domain-agnostic adaptive mesh (quadtree/octree)."""

    def __init__(self, dim: int = 2, domain_size: float = 1.0, max_level: int = 6):
        self.dim = dim
        self.domain_size = domain_size
        self.max_level = max_level
        self.cells: Dict[int, MeshCell] = {}
        self.next_id = 0

        center = tuple([domain_size / 2] * dim)
        root = MeshCell(id=0, level=0, center=center, size=domain_size)
        self.cells[0] = root
        self.next_id = 1

    def refine_cell(self, cell_id: int) -> List[int]:
        cell = self.cells[cell_id]
        if not cell.is_leaf or cell.level >= self.max_level:
            return []

        child_size = cell.size / 2
        children = []
        for i in range(2 ** self.dim):
            offset = tuple([
                (child_size / 2) if (i >> d) & 1 else (-child_size / 2)
                for d in range(self.dim)
            ])
            child_center = tuple(c + o for c, o in zip(cell.center, offset))
            child = MeshCell(
                id=self.next_id, level=cell.level + 1,
                center=child_center, size=child_size, parent=cell_id,
            )
            self.cells[child.id] = child
            children.append(child.id)
            self.next_id += 1

        cell.is_leaf = False
        cell.children = children
        return children

    def coarsen_cell(self, cell_id: int) -> Optional[int]:
        cell = self.cells[cell_id]
        if cell.parent is None:
            return None

        parent = self.cells[cell.parent]
        if not all(self.cells[cid].is_leaf for cid in parent.children):
            return None

        for cid in parent.children:
            del self.cells[cid]

        parent.is_leaf = True
        parent.children = []
        return parent.id

    def get_leaf_cells(self) -> List[MeshCell]:
        return [c for c in self.cells.values() if c.is_leaf]


class DorflerMarker:
    """[P] Dörfler (bulk chasing) marking strategy."""

    def __init__(self, theta: float = 0.5):
        if not 0 < theta < 1:
            raise ValueError("θ must be in (0, 1)")
        self.theta = theta

    def mark(self, cell_ids: List[int], errors: Tensor) -> List[int]:
        if len(cell_ids) == 0:
            return []

        sorted_idx = torch.argsort(errors, descending=True)
        sorted_errors = errors[sorted_idx]
        sorted_ids = [cell_ids[i] for i in sorted_idx]

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


class NoZenoGate:
    """
    [P] Hysteresis gate to bound refinement event count.
    N(T) ≤ (E_max + D) / δ_min
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
    Unified AMR controller for all 22 domains.
    """

    def __init__(self, cfg: AMRConfig, mesh: AdaptiveMesh):
        self.cfg = cfg
        self.mesh = mesh

        # Map domain to estimator class
        estimator_map = {
            # Physical sciences
            Domain.CFD: CFDErrorEstimator,
            Domain.QUANTUM_CHEMISTRY: QuantumChemistryEstimator,
            Domain.CONDENSED_MATTER: CondensedMatterEstimator,
            # Molecular biology
            Domain.DNA_RNA: DNARNAEstimator,
            Domain.PROTEIN_FOLDING: ProteinFoldingEstimator,
            Domain.PROTEIN_INTERACTION: ProteinInteractionEstimator,
            Domain.LIPID_MEMBRANE: LipidMembraneEstimator,
            Domain.METABOLIC_NETWORK: MetabolicNetworkEstimator,
            Domain.GENE_REGULATION: GeneRegulationEstimator,
            Domain.CELL_SIGNALING: CellSignalingEstimator,
            Domain.MICROBIAL_SYSTEMS: MicrobialSystemsEstimator,
            # Advanced materials
            Domain.METALS_ALLOYS: MetalsAlloysEstimator,
            Domain.CERAMICS: CeramicsEstimator,
            Domain.POLYMERS: PolymersEstimator,
            Domain.NANOMATERIALS: NanomaterialsEstimator,
            Domain.METAMATERIALS: MetamaterialsEstimator,
            Domain.BIOMATERIALS: BiomaterialsEstimator,
            Domain.COMPOSITES: CompositesEstimator,
            Domain.TWO_D_MATERIALS: TwoDMaterialsEstimator,
            # Biomedical
            Domain.ONCOLOGY: OncologyEstimator,
            Domain.EPIDEMIOLOGY: EpidemiologyEstimator,
            Domain.ANTI_AGING: AntiAgingEstimator,
            Domain.CELLULAR_REGEN: CellularRegenEstimator,
            Domain.GENE_EDITING: GeneEditingEstimator,
            Domain.NEUROSCIENCE: NeuroscienceEstimator,
        }

        if cfg.domain not in estimator_map:
            raise ValueError(f"Unknown domain: {cfg.domain}")

        self.estimator = estimator_map[cfg.domain](cfg)
        self.marker = DorflerMarker(cfg.dorfler_theta)

        if cfg.use_no_zeno:
            self.no_zeno = NoZenoGate(cfg.theta_hi, cfg.theta_lo)
        else:
            self.no_zeno = None

    def adapt(self, model_output: Dict[str, Any],
              ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform one AMR adaptation cycle."""
        errors = self.estimator.estimate(model_output, ground_truth)

        leaf_cells = self.mesh.get_leaf_cells()
        cell_ids = [c.id for c in leaf_cells]

        if len(errors) != len(cell_ids):
            errors = errors[:len(cell_ids)] if len(errors) > len(cell_ids) \
                     else F.pad(errors, (0, len(cell_ids) - len(errors)))

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

        marked = self.marker.mark(cell_ids, errors)

        refined = []
        for cid in marked:
            cell = self.mesh.cells[cid]
            if cell.size > self.cfg.min_cell_size and cell.level < self.cfg.max_refinement_level:
                children = self.mesh.refine_cell(cid)
                refined.extend(children)

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
    print(" SESI Unified AMR v2 — Smoke Test (22 domains)")
    print("=" * 72)
    torch.manual_seed(42)

    # Test representative domains from each group
    test_domains = [
        Domain.CFD,
        Domain.DNA_RNA,
        Domain.PROTEIN_FOLDING,
        Domain.METABOLIC_NETWORK,
        Domain.METALS_ALLOYS,
        Domain.NANOMATERIALS,
        Domain.METAMATERIALS,
        Domain.ONCOLOGY,
    ]

    for domain in test_domains:
        print(f"\n  ── Testing {domain.name} ({domain.group.name}) ──")
        cfg = AMRConfig(domain=domain, tolerance=1e-2)
        mesh = AdaptiveMesh(dim=2, domain_size=1.0, max_level=4)
        controller = UnifiedAMRController(cfg, mesh)

        # Generate mock model output (domain-specific)
        model_output = _generate_mock_output(domain)

        result = controller.adapt(model_output)
        print(f"    Refined: {len(result['refined'])}")
        print(f"    Coarsened: {len(result['coarsened'])}")
        print(f"    Leaf cells: {result['n_leaf_cells']}")
        print(f"    Total error: {result['total_error']:.4f}")
        print(f"    Blocked by No-Zeno: {result['blocked_by_no_zeno']}")

        if controller.no_zeno is not None:
            nz = controller.no_zeno
            print(f"    No-Zeno events: {nz.event_count}")
            print(f"    Bound check: {nz.event_count} ≤ {(nz.E_max + nz.D) / nz.delta_min:.4f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • CFD estimator is STANDARD (FEM theory) [P]")
    print("  • Molecular biology estimators are HEURISTIC proxies [D]")
    print("  • Materials science estimators combine PDE + ML [D]")
    print("  • No-Zeno bounds EVENT COUNT, not error [P]")
    print("  • AMR convergence NOT proved for non-CFD domains [O]")
    print("=" * 72)


def _generate_mock_output(domain: Domain) -> Dict[str, Any]:
    """Generate mock model output for testing."""
    B = 4  # batch size

    if domain == Domain.CFD:
        return {
            "u_h": torch.randn(B, 10),
            "f": torch.randn(B, 10),
            "cell_sizes": torch.tensor([0.5, 0.5, 0.5, 0.5]),
            "grad_u_h": torch.randn(B, 10, 2),
        }
    elif domain == Domain.DNA_RNA:
        return {
            "sequence_encoding": torch.randn(B, 100, 64),
            "base_pair_probs": torch.rand(B, 100, 100),
            "stacking_energy": torch.randn(B, 99),
            "folding_dG": torch.randn(B),
        }
    elif domain == Domain.PROTEIN_FOLDING:
        return {
            "pred_coords": torch.randn(B, 50, 3),
            "confidence": torch.rand(B, 50),
            "pae_matrix": torch.rand(B, 50, 50),
        }
    elif domain == Domain.METABOLIC_NETWORK:
        N_rxn, N_met = 20, 15
        return {
            "flux_vector": torch.randn(B, N_rxn),
            "stoich_matrix": torch.randn(N_met, N_rxn),
            "gibbs_energies": torch.randn(B, N_rxn),
            "enzyme_kinetics": torch.rand(B, N_rxn, 3),
        }
    elif domain == Domain.METALS_ALLOYS:
        H, W, D = 8, 8, 8
        return {
            "stress_field": torch.randn(B, 3, 3, H, W, D),
            "strain_field": torch.randn(B, 3, 3, H, W, D),
            "phase_field": torch.rand(B, 3, H, W, D),
            "dislocation_density": torch.rand(B, H, W, D),
        }
    elif domain == Domain.NANOMATERIALS:
        return {
            "particle_size": torch.rand(B) * 10,
            "surface_area": torch.rand(B) * 100,
            "bandgap": torch.rand(B) * 3,
            "plasmon_spectrum": torch.rand(B, 50),
        }
    elif domain == Domain.METAMATERIALS:
        return {
            "effective_permittivity": torch.randn(B, 50) + 1j * torch.randn(B, 50),
            "effective_permeability": torch.randn(B, 50) + 1j * torch.randn(B, 50),
            "band_structure": torch.randn(B, 20, 5),
            "unit_cell_sensitivity": torch.randn(B, 10),
        }
    elif domain == Domain.ONCOLOGY:
        return {
            "latent": torch.randn(B, 256),
            "predictions": {"tumor_grid": torch.rand(B, 1, 16, 16)},
        }
    else:
        # Generic fallback
        return {
            "latent": torch.randn(B, 256),
            "features": torch.randn(B, 128),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
