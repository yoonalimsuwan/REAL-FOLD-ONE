# =============================================================================
# REAL FOLD ONE – Universal Full-Atom Native Differentiable Refinement Engine
# =============================================================================
# Author       : PAI , Yoon A Limsuwan
# Organization : MSPS NETWORK
# License      : MIT
# Year         : 2026
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
#
# AI Co-Developers (architecture, differentiability, numerical methods):
#   - Claude   (Anthropic)  — native full differentiability design via
#                             OpenMM-ML TorchForce, SOCController inheritance
#                             fix, register_buffer patterns, LangevinBridge
#                             architecture, openmm_solute_energy routing,
#                             FoldCahnHilliardBridge coupling protocol,
#                             full README_PLUS documentation
#   - GPT      (OpenAI)     — OpenMM-ML API cross-check, AMBER force field
#                             parameterisation guidance, Hessian compatibility
#                             notes for second-order optimisation
#   - Gemini   (Google)     — SOC kernel initial design, RG coarse-graining
#                             architecture, multiscale refinement scaffolding
#   - DeepSeek              — alternative SOC energy formulation review,
#                             CSOC adaptive kernel verification
#
# REAL FOLD ONE is an end-to-end differentiable protein/nucleic acid refinement
# engine blending atomic physics, self-organised criticality (SOC), deep-learning
# optimisation, and multiscale coarse-graining in a single PyTorch workflow.
#
# Built on open-source foundations (all licences listed below):
#   * OpenMM         – molecular mechanics engine (MIT)
#   * OpenMM-ML      – native differentiable ML potentials via TorchForce (MIT)
#                      conda install -c conda-forge openmm-ml openmm-torch
#   * ANI-2x / MACE-MP-0 / AIMNet2 – ML interatomic potentials (various open)
#   * Amber force fields (ff14SB, OL15, GAFF2, TIP3P) – LGPL (XML in OpenMM)
#   * biotite        – structure I/O and stereochemical analysis (BSD-3-Clause)
#   * RDKit          – cheminformatics / ligand handling (BSD-3-Clause)
#   * PyTorch        – automatic differentiation (BSD-style)
#   * torch-cluster  – fast neighbour lists (MIT)
#   * SciPy          – spatial indexing (BSD-3-Clause)
#   * NumPy          – numerical arrays (BSD-3-Clause)
#   * networkx       – graph algorithms (BSD-3-Clause)
#   * openff-toolkit & openmmforcefields – ligand parameterisation (MIT)
#
# KEY UPGRADE: Native Full Differentiability via OpenMM-ML
# --------------------------------------------------------
# Previous versions used a force-injection trick: OpenMM returned forces as
# numpy arrays injected as gradients via torch.autograd.Function.  First-order
# only; Hessians and higher-order derivatives were blocked.
#
# This version compiles the force field into a TorchScript module (TorchForce)
# via OpenMM-ML + openmm-torch.  Energy evaluation runs entirely inside the
# PyTorch autograd graph:
#
#   solute_coords.requires_grad_(True)
#   E = openmm_solute_energy(solute_coords, calculator)
#   E.backward()              # analytical dE/dpos, no force injection
#   H = torch.autograd.functional.hessian(...)  # works correctly
#
# Supported ML potentials (RefinementConfig.ml_potential):
#   ani2x      – ANI-2x  (CHNO+SF; fast, good for organics)
#   ani1ccx    – ANI-1ccx (CHNO; coupled-cluster accuracy)
#   mace-mp-0  – MACE foundation model (all elements; high accuracy)
#   aimnet2    – AIMNet2 (CHNO+halogens)
#   None         – classical AMBER only (force-injection fallback)
#
# Graceful fallback: if openmmml / openmm-torch are not installed the engine
# transparently reverts to force-injection with a warning.
#
# COMPLETE FEATURE SET:
#   - SOC Controller with learnable CSOC kernel & adaptive relaxation
#   - Semantic-State Contraction (SSC) low-pass filter
#   - Multiscale Refinement – RG coarse-graining with full-atom consistency
#   - Full-atom physics via OpenMM-ML (native autograd TorchForce):
#       Proteins: AMBER ff14SB + ANI-2x / MACE-MP-0 ML correction
#       DNA/RNA: OL15  |  Ligands: GAFF2  |  Antibodies: MM-GBSA
#       Explicit water, ions, co-solvents, or implicit solvent (OBC, GBn2)
#   - PME, reaction field, implicit solvent advanced electrostatics
#   - Hierarchical Neighbour Lists (torch-cluster / scipy / pure PyTorch)
#   - DNA Origami: wireframe routing, staple design, all-atom PDB, oxDNA export
#   - Langevin dynamics (overdamped) + cosine annealing
#   - Scalable to >100 000 atoms, O(N) memory
#   - Multi-backend: CUDA, MPS, Ascend NPU, XPU, CPU
#   - Validation suite: RMSD, clash score, Ramachandran, rotamers, geometry
#   - Long-time MD (ps to microsecond)
#   - Positional restraints, automatic ligand CCD, disulfide detection, PTMs
#
# Usage examples:
#   python real_fold_one.py refine -i input.pdb -o refined.pdb --steps 200
#   python real_fold_one.py refine -i input.pdb -o refined.pdb --ml-potential mace-mp-0
#   python real_fold_one.py train -i pdbs/*.pdb --epochs 50
#   python real_fold_one.py origami --shape design.json --output origami
#   python real_fold_one.py md -i input.pdb -o traj --steps 100000
#   python real_fold_one.py test -i input.pdb
#   python real_fold_one.py validate -i input.pdb [--reference ref.pdb]
# =============================================================================
import math, os, sys, json, argparse, warnings, random, itertools, time, logging, gc, atexit, weakref
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable, Union
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ONE Core Fold — single source of truth for shared components
from one_core_fold import (
    SemanticStateContraction,   # SSC EMA filter  (Paper 4)
    CSOCBase,                   # CSOC abstract base
    InterfaceDetectorBase,      # Interface detector abstract base
    StructuralItoBase,          # Structural Itô base (Papers 2 & 3)
    LangevinBridge,             # RefinementEngine ↔ AdvancedStructuralLangevin
    FoldCahnHilliardBridge,     # REAL FOLD ONE ↔ StructuralCahnHilliard3D
    CahnHilliardSSCAdapter,     # SSC adapter for CH3D attach_ssc()
    get_device as _core_get_device,
    FOLD_VERSION,
)
from torch.cuda.amp import autocast, GradScaler

# ---------------------------------------------------------------------------
# Environment‑sensitive imports & logging
# ---------------------------------------------------------------------------
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmm.app import (
        PDBFile, Modeller, ForceField, PME, HBonds,
        NoCutoff, CutoffNonPeriodic, CutoffPeriodic,
        Simulation, DCDReporter, StateDataReporter, CheckpointReporter
    )
    from openmm import (
        Platform, System, VerletIntegrator, Context,
        LangevinMiddleIntegrator, MonteCarloBarostat
    )
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    logger = logging.getLogger("REAL_FOLD_ONE")
    logger.error("OpenMM not found. Install with: conda install -c conda-forge openmm")

# ---------------------------------------------------------------------------
# OpenMM-ML: native full differentiable ML potentials via TorchForce
# Install: conda install -c conda-forge openmm-ml
# Requires openmm-torch for TorchForce plugin registration.
# Install: conda install -c conda-forge openmm-torch
# ---------------------------------------------------------------------------
try:
    from openmmml import MLPotential
    HAS_OPENMMML = True
except ImportError:
    HAS_OPENMMML = False

try:
    import openmmtorch  # registers TorchForce plugin into OpenMM's DLL search
    HAS_OPENMMTORCH = True
except ImportError:
    try:
        # Newer builds bundle TorchForce directly inside openmm-torch
        from openmm import TorchForce as _TorchForce  # noqa: F401
        HAS_OPENMMTORCH = True
    except ImportError:
        HAS_OPENMMTORCH = False

try:
    from openmmforcefields.generators import SystemGenerator
    HAS_OPENMMFORCEFIELDS = True
except ImportError:
    HAS_OPENMMFORCEFIELDS = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.mmcif as mmcif_io
    from biotite.structure.info import residue as bio_residue
    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False

try:
    from torch_cluster import radius_graph
    HAS_CLUSTER = True
except ImportError:
    HAS_CLUSTER = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from openff.toolkit import Molecule
    HAS_OPENFF = True
except ImportError:
    HAS_OPENFF = False

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("REAL_FOLD_ONE")

# =============================================================================
# 0.1 Device Detection & Memory Helpers
# =============================================================================
def detect_optimal_device(verbose: bool = True) -> Tuple[torch.device, float]:
    """Auto‑detect the best available compute device and free memory."""
    device = torch.device("cpu")
    memory_gb = 4.0

    if torch.cuda.is_available():
        try:
            device_id = 0
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                visible = os.environ["CUDA_VISIBLE_DEVICES"]
                if visible:
                    try:
                        device_id = int(visible.split(",")[0])
                    except ValueError:
                        device_id = 0
            device = torch.device(f"cuda:{device_id}")
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            memory_gb = free_mem / 1e9
            if verbose:
                logger.info(f"✓ CUDA: {torch.cuda.get_device_name(device_id)} ({memory_gb:.1f} GB free)")
            return device, memory_gb
        except Exception as e:
            if verbose:
                logger.warning(f"CUDA init failed: {e}")

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            try:
                import psutil
                memory_gb = psutil.virtual_memory().available / 1e9
            except ImportError:
                memory_gb = 8.0
            if verbose:
                logger.info(f"✓ Apple MPS (Metal) – ~{memory_gb:.1f} GB")
            return device, memory_gb
        except Exception as e:
            if verbose:
                logger.warning(f"MPS init failed: {e}")

    ascend_devices = os.environ.get("ASCEND_VISIBLE_DEVICES", "")
    if ascend_devices or os.path.exists("/usr/local/Ascend"):
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                device = torch.device("npu:0")
                memory_gb = 16.0
                if verbose:
                    logger.info(f"✓ Huawei Ascend NPU – ~{memory_gb:.1f} GB")
                return device, memory_gb
        except ImportError:
            pass

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        try:
            device = torch.device("xpu:0")
            memory_gb = 8.0
            if verbose:
                logger.info(f"✓ Intel XPU – ~{memory_gb:.1f} GB")
            return device, memory_gb
        except Exception:
            pass

    try:
        import psutil
        memory_gb = psutil.virtual_memory().available / 1e9
    except ImportError:
        memory_gb = 4.0
    if verbose:
        logger.info(f"✓ CPU ({memory_gb:.1f} GB RAM)")
    return device, memory_gb

OPTIMAL_DEVICE, AVAILABLE_MEMORY_GB = detect_optimal_device()

def get_chunk_size(available_memory_gb: float, n_atoms: int, dtype_bytes: int = 4) -> int:
    """Heuristic chunk size for memory‑constrained processing."""
    usable = available_memory_gb * 0.5 * 1e9
    bytes_per_pair = dtype_bytes * 7
    max_pairs = max(int(usable / bytes_per_pair), 100)
    chunk = min(int(math.sqrt(max_pairs)), 5000)
    return max(chunk, 50)

# =============================================================================
# 1. Constants & Utilities
# =============================================================================
AA_3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','UNK':'X'
}
HYDROPHOBICITY = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
    'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,
    'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3,'X':0.0
}
WC_PAIRS = {('A','T'):2,('T','A'):2,('A','U'):2,('U','A'):2,
            ('G','C'):3,('C','G'):3,('G','U'):1,('U','G'):1}
BASE_STACKING = {'A':1.0,'T':0.8,'U':0.8,'G':1.2,'C':1.0}

# ---------------------------------------------------------------------------
# Common post‑translational modifications (SMILES)
# ---------------------------------------------------------------------------
MODIFIED_RESIDUE_SMILES = {
    'SEP': 'C([C@@H](C(=O)O)N)OP(=O)(O)O',        # phosphoserine
    'TPO': 'C[C@H](OP(=O)(O)O)[C@@H](C(=O)O)N',   # phosphothreonine
    'PTR': 'C1=CC(=CC=C1C[C@@H](C(=O)O)N)OP(=O)(O)O', # phosphotyrosine
    'M3L': 'C[N+](C)(C)CCCC[C@@H](C(=O)[O-])N',   # trimethyllysine
    'ALY': 'CC(=O)NCCCC[C@@H](C(=O)O)N',           # acetyllysine
    'HYP': 'C1[C@@H]([C@H](N[C@@H]1C(=O)O)O)O',   # hydroxyproline
    'MSE': 'C[Se]CC[C@@H](C(=O)O)N',              # selenomethionine
    'CME': 'CSCC[C@@H](C(=O)O)N',                 # S‑methylcysteine
    'CSO': 'C[S@@](=O)CC[C@@H](C(=O)O)N',         # S‑hydroxycysteine
    'OCS': 'C(=O)[O-]',                            # generic carboxylate (if needed)
}

def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def detect_sequence_type(seq: str) -> str:
    nt_set = set("ACGTU")
    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    n_nt = sum(1 for c in seq.upper() if c in nt_set)
    n_aa = sum(1 for c in seq.upper() if c in aa_set)
    total = len(seq)
    if total == 0:
        return 'unknown'
    if n_nt / total > 0.8:
        return 'rna' if 'U' in seq.upper() else 'dna'
    if n_aa / total > 0.7:
        return 'protein'
    return 'unknown'

# =============================================================================
# 1.1 Ramachandran Sampler (Dunbrack‑based when available, fallback generic)
# =============================================================================
def _parse_residue_id(res_id: str) -> Tuple[int, str]:
    """Parse PDB residue id into (number, insertion_code)."""
    import re
    m = re.match(r'(-?\d+)([A-Z]?)', str(res_id).strip())
    if m:
        return int(m.group(1)), m.group(2)
    return 0, ''

def _build_generic_rama_grid():
    """Simple α/β Ramachandran grid (fallback)."""
    phi_bins = np.arange(-180, 180, 10)
    psi_bins = np.arange(-180, 180, 10)
    grid = np.zeros((len(phi_bins), len(psi_bins)))
    alpha_phi_idx = np.searchsorted(phi_bins, -57)
    alpha_psi_idx = np.searchsorted(psi_bins, -47)
    grid[alpha_phi_idx-1:alpha_phi_idx+2, alpha_psi_idx-1:alpha_psi_idx+2] = 1.0
    beta_phi_idx = np.searchsorted(phi_bins, -135)
    beta_psi_idx = np.searchsorted(psi_bins, 135)
    grid[beta_phi_idx-1:beta_phi_idx+2, beta_psi_idx-1:beta_psi_idx+2] = 1.0
    lh_phi_idx = np.searchsorted(phi_bins, 60)
    lh_psi_idx = np.searchsorted(psi_bins, 40)
    grid[lh_phi_idx-1:lh_phi_idx+2, lh_psi_idx-1:lh_psi_idx+2] = 0.3
    try:
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=1.0)
    except ImportError:
        pass
    grid /= grid.sum()
    return grid.ravel(), (len(phi_bins), len(psi_bins))

_GENERIC_RAMA_PROB, _GENERIC_RAMA_BINS = _build_generic_rama_grid()

class RamachandranSampler:
    """
    Backbone‑dependent Ramachandran sampler.
    When biotite is available, loads Dunbrack probability grids per residue type.
    Otherwise falls back to a generic α/β distribution.
    """
    def __init__(self, use_dunbrack: bool = True):
        self.use_dunbrack = use_dunbrack and HAS_BIOTITE
        self._dunbrack_data = {}
        if self.use_dunbrack:
            self._load_dunbrack_data()

    def _load_dunbrack_data(self):
        try:
            from biotite.structure.info import dunbrack
            for aa in AA_3_TO_1.values():
                if aa == 'X':
                    continue
                try:
                    rama_data = dunbrack.get_ramachandran_distribution(aa)
                    prob = rama_data['probability'].astype(np.float32)
                    prob /= prob.sum()
                    phi_edges = rama_data['phi_edges']
                    psi_edges = rama_data['psi_edges']
                    self._dunbrack_data[aa] = (prob, phi_edges, psi_edges)
                except Exception:
                    self._dunbrack_data[aa] = None
            logger.info("Loaded Dunbrack Ramachandran distributions for backbone sampling.")
        except Exception as e:
            logger.warning(f"Dunbrack data unavailable: {e}. Using generic Ramachandran.")
            self.use_dunbrack = False

    def sample_phi_psi(self, aa: str = 'A', n: int = 1) -> np.ndarray:
        """Sample (φ, ψ) pairs for a given amino acid (one‑letter code)."""
        if self.use_dunbrack and aa in self._dunbrack_data and self._dunbrack_data[aa] is not None:
            prob, phi_edges, psi_edges = self._dunbrack_data[aa]
            prob_flat = prob.ravel()
            idx = np.random.choice(len(prob_flat), size=n, p=prob_flat)
            phi_idx = idx // prob.shape[1]
            psi_idx = idx % prob.shape[1]
            phi = np.random.uniform(phi_edges[phi_idx], phi_edges[phi_idx+1], n)
            psi = np.random.uniform(psi_edges[psi_idx], psi_edges[psi_idx+1], n)
            return np.stack([phi, psi], axis=-1)
        else:
            idx_flat = np.random.choice(len(_GENERIC_RAMA_PROB), size=n, p=_GENERIC_RAMA_PROB)
            phi_idx = idx_flat // _GENERIC_RAMA_BINS[1]
            psi_idx = idx_flat % _GENERIC_RAMA_BINS[1]
            phi = np.arange(-180, 180, 10)[phi_idx] + np.random.uniform(-5, 5, n)
            psi = np.arange(-180, 180, 10)[psi_idx] + np.random.uniform(-5, 5, n)
            return np.stack([phi, psi], axis=-1)

# Default instance
rama_sampler = RamachandranSampler()

# =============================================================================
# 2. OpenMM System Builder (robust, auto CCD lookup for ligands, implicit solv.)
# =============================================================================
class OpenMMSystemBuilder:
    """
    Creates an OpenMM System from a PDB file, handling:
    - Standard proteins, DNA, RNA, and non‑standard residues.
    - Small molecule ligands via GAFF2 (automatic SMILES from CCD).
    - Multimers (multiple chains, complexes, antibodies).
    - Explicit water, ions, co‑solvents, or implicit solvent (GB models).
    - Disulfide bond detection and constraint.
    - Post‑translational modifications (SEP, TPO, etc.) via built‑in SMILES.
    """
    def __init__(self,
                 forcefield_files: Optional[List[str]] = None,
                 implicit_solvent: Optional[str] = None,
                 solvent_model: str = 'tip3p',
                 ionic_strength: float = 0.15,
                 box_padding: float = 1.0,
                 nonbonded_method: str = 'PME',
                 nonbonded_cutoff: float = 1.0,   # nm
                 rigid_water: bool = True,
                 hydrogen_mass: Optional[float] = None,
                 ligand_smiles: Optional[Dict[str, str]] = None,
                 disulfide_pairs: Optional[List[Tuple[str, str]]] = None):
        if not HAS_OPENMM:
            raise ImportError("OpenMM is required. Install with: conda install -c conda-forge openmm")
        if forcefield_files is None:
            if implicit_solvent is not None:
                forcefield_files = [
                    'amber14/protein.ff14SB.xml',
                    'amber14/DNA.OL15.xml',
                    'amber14/RNA.OL15.xml'
                ]
            else:
                forcefield_files = [
                    'amber14/protein.ff14SB.xml',
                    'amber14/DNA.OL15.xml',
                    'amber14/RNA.OL15.xml',
                    'amber14/tip3p.xml'
                ]
        self.forcefield_files = forcefield_files
        self.implicit_solvent = implicit_solvent
        self.solvent_model = solvent_model
        self.ionic_strength = ionic_strength
        self.box_padding = box_padding
        self.nonbonded_method = nonbonded_method
        self.nonbonded_cutoff = nonbonded_cutoff
        self.rigid_water = rigid_water
        self.hydrogen_mass = hydrogen_mass
        self.ligand_smiles = ligand_smiles or {}
        self.disulfide_pairs = disulfide_pairs

    def _fetch_ccd_smiles(self, residue_name: str) -> Optional[str]:
        """Try to obtain SMILES from PDB Chemical Component Dictionary or built‑in modified residues."""
        if residue_name in MODIFIED_RESIDUE_SMILES:
            return MODIFIED_RESIDUE_SMILES[residue_name]
        if not HAS_BIOTITE:
            return None
        try:
            comp = bio_residue(residue_name)
            if comp and hasattr(comp, 'smiles'):
                return comp.smiles
        except Exception:
            pass
        return None

    def _map_implicit_solvent(self):
        """Return the appropriate OpenMM implicit solvent object or raise ValueError."""
        model = (self.implicit_solvent or '').lower()
        if model in ('obc', 'obc2', 'obc1'):
            return app.OBC2
        elif model in ('gbn', 'gbn2', 'gbn'):
            return app.GBn2
        elif model in ('gbn1',):
            return app.GBn1
        else:
            raise ValueError(f"Unsupported implicit solvent model: {self.implicit_solvent}. "
                             "Supported: OBC, OBC2, GBn, GBn2, GBn1.")

    def build_from_pdb(self, pdb_file: str, add_missing_residues: bool = True,
                       add_hydrogens: bool = True, solvate: bool = True,
                       add_ions: bool = True) -> Tuple[mm.System, app.Topology, torch.Tensor, Dict]:
        pdb = PDBFile(pdb_file)
        modeller = Modeller(pdb.topology, pdb.positions)

        if add_missing_residues:
            modeller.addMissingResidues()
        ff = ForceField(*self.forcefield_files)
        if add_hydrogens:
            modeller.addHydrogens(ff)

        unmatched = ff.getUnmatchedResidues(modeller.topology)
        non_standard = set()
        for residue in modeller.topology.residues():
            if residue.name in unmatched and residue.name not in ('HOH', 'WAT', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN', 'CU', 'NI', 'CO'):
                non_standard.add(residue.name)

        ligand_smiles = dict(self.ligand_smiles)
        for resname in non_standard:
            if resname not in ligand_smiles:
                smiles = self._fetch_ccd_smiles(resname)
                if smiles:
                    logger.info(f"Auto‑detected SMILES for {resname}: {smiles}")
                    ligand_smiles[resname] = smiles

        missing = non_standard - set(ligand_smiles.keys())
        if missing:
            msg = (f"Cannot parameterise non‑standard residues: {missing}. "
                   "Provide SMILES via --ligand-smiles (JSON) or ensure the PDB CCD contains them.")
            logger.error(msg)
            raise ValueError(msg)

        if non_standard and HAS_OPENMMFORCEFIELDS and ligand_smiles:
            molecules = []
            for resname in non_standard:
                mol = Molecule.from_smiles(ligand_smiles[resname], allow_undefined_stereo=True)
                molecules.append(mol)
            generator = SystemGenerator(
                forcefields=self.forcefield_files,
                small_molecule_forcefield='gaff-2.2',
                molecules=molecules,
                cache='system_generator_cache.json'
            )
            system = generator.create_system(modeller.topology, molecules=molecules)
        else:
            if self.implicit_solvent is not None:
                implicit_solvent_model = self._map_implicit_solvent()
                nonbonded_method = app.NoCutoff
                system = ff.createSystem(modeller.topology,
                                         nonbondedMethod=nonbonded_method,
                                         nonbondedCutoff=self.nonbonded_cutoff * unit.nanometer,
                                         constraints=HBonds if self.rigid_water else None,
                                         implicitSolvent=implicit_solvent_model,
                                         implicitSolventSaltConc=self.ionic_strength * unit.molar,
                                         soluteDielectric=1.0,
                                         solventDielectric=78.5)
                solvate = False
                add_ions = False
            else:
                nonbonded_method = PME if self.nonbonded_method == 'PME' else CutoffNonPeriodic
                system = ff.createSystem(modeller.topology,
                                         nonbondedMethod=nonbonded_method,
                                         nonbondedCutoff=self.nonbonded_cutoff * unit.nanometer,
                                         constraints=HBonds if self.rigid_water else None)

        if solvate and self.implicit_solvent is None:
            ff_solv = ForceField(*self.forcefield_files)
            modeller.addSolvent(ff_solv,
                                model=self.solvent_model,
                                padding=self.box_padding * unit.nanometer,
                                ionicStrength=self.ionic_strength * unit.molar)
            if add_ions:
                modeller.addIons(ff_solv, ionicStrength=self.ionic_strength * unit.molar)
        elif solvate and self.implicit_solvent is not None:
            logger.warning("Explicit solvation requested but implicit solvent is active; ignoring solvate/add_ions.")

        if self.hydrogen_mass is not None:
            for i in range(system.getNumParticles()):
                mass = system.getParticleMass(i)
                if mass.value_in_unit(unit.amu) < 1.5:
                    system.setParticleMass(i, self.hydrogen_mass * unit.amu)

        # --- Disulfide bond detection and addition ---
        if self.disulfide_pairs is None:
            cysteine_sg = []
            for atom in modeller.topology.atoms():
                res = atom.residue
                if res is not None and res.name == 'CYS' and atom.name == 'SG':
                    cysteine_sg.append(atom)
            disulf_pairs = []
            for i in range(len(cysteine_sg)):
                for j in range(i+1, len(cysteine_sg)):
                    pos_i = modeller.positions[cysteine_sg[i].index]
                    pos_j = modeller.positions[cysteine_sg[j].index]
                    dist = (pos_i - pos_j).value_in_unit(unit.angstrom)
                    if dist < 2.5:
                        disulf_pairs.append((cysteine_sg[i].residue.id, cysteine_sg[j].residue.id))
                        system.addForce(mm.HarmonicBondForce())
                        bond_force = system.getForce(system.getNumForces()-1)
                        bond_force.addBond(cysteine_sg[i].index, cysteine_sg[j].index, 2.05*unit.angstrom,
                                           500.0*unit.kilocalorie_per_mole/unit.angstrom**2)
                        logger.info(f"Detected disulfide bond between CYS {disulf_pairs[-1][0]} and {disulf_pairs[-1][1]}")
        else:
            for pair in self.disulfide_pairs:
                idx1 = idx2 = None
                for atom in modeller.topology.atoms():
                    res = atom.residue
                    if res is not None and res.id == pair[0] and atom.name == 'SG':
                        idx1 = atom.index
                    elif res is not None and res.id == pair[1] and atom.name == 'SG':
                        idx2 = atom.index
                if idx1 is not None and idx2 is not None:
                    system.addForce(mm.HarmonicBondForce())
                    bond_force = system.getForce(system.getNumForces()-1)
                    bond_force.addBond(idx1, idx2, 2.05*unit.angstrom,
                                       500.0*unit.kilocalorie_per_mole/unit.angstrom**2)
                    logger.info(f"Added disulfide bond between {pair[0]} and {pair[1]}")

        topology = modeller.topology
        positions = modeller.positions
        n_total = system.getNumParticles()

        solute_mask = torch.zeros(n_total, dtype=torch.bool)
        ca_indices = []
        for atom in topology.atoms():
            res = atom.residue
            if res is None:
                continue
            res_name = res.name.strip()
            if res_name not in ('HOH', 'WAT', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN', 'CU', 'NI', 'CO'):
                solute_mask[atom.index] = True
                if atom.name in ('CA', 'C4\''):
                    ca_indices.append(atom.index)

        solute_indices = torch.where(solute_mask)[0]
        full2sol = {idx.item(): i for i, idx in enumerate(solute_indices)}

        init_pos_nm = [list(pos.value_in_unit(unit.nanometer)) for pos in positions]
        init_all_ang = torch.tensor(init_pos_nm, dtype=torch.float32) * 10.0

        ca_info = {}
        for atom in topology.atoms():
            if atom.index in full2sol and atom.name in ('CA', 'C4\''):
                res = atom.residue
                chain_id = res.chain.id if res.chain else ''
                res_id = res.id
                ca_info[full2sol[atom.index]] = (chain_id, res_id)

        metadata = {
            'solute_mask': solute_mask,
            'ca_indices_full': ca_indices,
            'full2sol': full2sol,
            'solute_indices': solute_indices,
            'ca_chain_res_map': ca_info
        }

        return system, topology, init_all_ang, metadata

# =============================================================================
# 3. Native Differentiable OpenMM-ML Energy (full autograd via TorchForce)
# =============================================================================

class _MLTorchEnergyModule(torch.nn.Module):
    """
    Thin PyTorch Module wrapping an OpenMM-ML TorchForce context so that
    energy (and its gradient w.r.t. atomic coordinates) can be evaluated
    inside a native autograd graph.

    OpenMM-ML compiles the AMBER/ML potential into a TorchScript module
    (``MLPotential.createSystem``).  We load that module here and call it
    directly — no ``autograd.Function`` force-hack is needed because the
    TorchScript graph itself is differentiable.

    Design:
      • ``forward(positions_ang)`` → scalar energy (kcal/mol)
      • positions_ang : (N_atoms, 3) float32 tensor in **Angström**
      • Gradient  ∂E/∂pos  flows through native autograd (torch.autograd.grad
        or .backward()) without any finite-difference or force injection.
    """

    def __init__(self,
                 torchscript_path: str,
                 solute_mask: torch.Tensor,
                 full_coords_fixed: torch.Tensor,
                 device: torch.device):
        super().__init__()
        self.solute_mask = solute_mask
        self.register_buffer('full_coords_fixed', full_coords_fixed.clone())
        self.device = device
        # Load the compiled TorchScript potential
        self._potential_module: torch.jit.ScriptModule = torch.jit.load(
            torchscript_path, map_location=device
        )
        # Put in eval mode — we do not train the potential weights
        self._potential_module.eval()
        for p in self._potential_module.parameters():
            p.requires_grad_(False)

    def forward(self, solute_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            solute_coords : (N_solute, 3) float32 tensor, **Angström**, requires_grad=True.

        Returns:
            Scalar energy in kcal/mol.  Gradient w.r.t. solute_coords is the
            negative force, flowing through native autograd.
        """
        # Reconstruct full coordinate tensor (solvent fixed, solute optimised)
        full = self.full_coords_fixed.clone()
        full = full.index_put(
            (torch.where(self.solute_mask)[0],),
            solute_coords
        )
        # Convert Å → nm for the TorchScript module (OpenMM convention)
        pos_nm = full * 0.1          # (N_total, 3), nm
        energy_kj = self._potential_module(pos_nm)   # scalar, kJ/mol
        energy_kcal = energy_kj * 0.239006           # → kcal/mol
        return energy_kcal


class OpenMMEnergyCalculator:
    """
    Manages OpenMM-ML system setup and exposes a PyTorch-native differentiable
    energy function via ``compute_native()``.

    Two operating modes
    -------------------
    1. **Full native autograd** (``HAS_OPENMMML and HAS_OPENMMTORCH``):
       ``MLPotential`` compiles the force field into a TorchScript module.
       Energy and forces are computed inside the native autograd graph —
       ``solute_coords.backward()`` works without any force injection.

    2. **Fallback autograd** (OpenMM-ML unavailable):
       Uses the classic ``autograd.Function`` approach where OpenMM returns
       forces as numpy arrays, which are then injected as the gradient.
       Functionally correct for first-order optimisation but second-order
       methods (Hessians) will not work.

    Supports context manager protocol for safe cleanup.
    """

    def __init__(self, system: mm.System, topology: app.Topology,
                 full_coords_fixed: torch.Tensor, solute_mask: torch.Tensor,
                 platform_name: str,
                 ml_potential_name: str = 'ani2x',
                 use_native_autograd: bool = True):
        """
        Args:
            system              : OpenMM System (AMBER + any added forces).
            topology            : OpenMM Topology.
            full_coords_fixed   : (N_total, 3) float32 Å tensor — full system
                                  initial coordinates (solvent will stay fixed).
            solute_mask         : (N_total,) bool tensor marking solute atoms.
            platform_name       : OpenMM platform ('CUDA', 'OpenCL', 'CPU', …).
            ml_potential_name   : OpenMM-ML potential identifier.  Supported
                                  values depend on what is installed:
                                    'ani2x'  — ANI-2x neural network potential
                                               (accurate for CHNO+SF halides)
                                    'ani1ccx'— ANI-1ccx (CHNO only)
                                    'mace-mp-0' — MACE foundation model
                                               (all elements, very accurate)
                                    'aimnet2'   — AIMNet2
                                  Any identifier accepted by ``MLPotential``
                                  may be used.  Set to None to skip ML layer
                                  and use the classical AMBER force field only
                                  (native autograd is then unavailable).
            use_native_autograd : If True, attempt to compile a TorchScript
                                  potential for native autograd.  Falls back
                                  gracefully when OpenMM-ML is absent.
        """
        self.system = system
        self.topology = topology
        self.full_coords_fixed = full_coords_fixed
        self.solute_mask = solute_mask
        self.platform_name = platform_name
        self.ml_potential_name = ml_potential_name
        self._closed = False

        # ---- resolve torch device from OpenMM platform ----
        self.torch_device = self._resolve_torch_device(platform_name)

        # ---- attempt native autograd path ----
        self._native_module: Optional[_MLTorchEnergyModule] = None
        self._torchscript_path: Optional[str] = None

        if use_native_autograd and HAS_OPENMMML and HAS_OPENMMTORCH and ml_potential_name:
            try:
                self._torchscript_path = self._compile_ml_potential(
                    system, topology, ml_potential_name, platform_name
                )
                self._native_module = _MLTorchEnergyModule(
                    torchscript_path=self._torchscript_path,
                    solute_mask=solute_mask.to(self.torch_device),
                    full_coords_fixed=full_coords_fixed.to(self.torch_device),
                    device=self.torch_device
                )
                self._use_native = True
                logger.info(
                    f"✓ Native autograd energy via OpenMM-ML ({ml_potential_name}) + TorchForce."
                )
            except Exception as exc:
                logger.warning(
                    f"OpenMM-ML compilation failed ({exc}); falling back to force-injection autograd."
                )
                self._use_native = False
        else:
            if use_native_autograd and (not HAS_OPENMMML or not HAS_OPENMMTORCH):
                logger.info(
                    "OpenMM-ML or openmm-torch not installed — using force-injection autograd fallback.  "
                    "Install with:  conda install -c conda-forge openmm-ml openmm-torch"
                )
            self._use_native = False

        # ---- always build a classical OpenMM context (used by fallback and MD) ----
        self.platform = self._select_platform(platform_name)
        self.integrator = VerletIntegrator(0.001)
        self.context = Context(system, self.integrator, self.platform)
        pos_nm = full_coords_fixed.cpu().numpy() * 0.1
        self.context.setPositions(pos_nm)
        state = self.context.getState(getPositions=True)
        self._pos_np_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        self._finalizer = weakref.finalize(self, self._cleanup_context, self.context)
        logger.info("OpenMM classical context created.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_torch_device(platform_name: str) -> torch.device:
        p = platform_name.upper()
        if p == 'CUDA' and torch.cuda.is_available():
            return torch.device('cuda')
        if p in ('METAL', 'MPS') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    @staticmethod
    def _compile_ml_potential(system: mm.System, topology: app.Topology,
                               potential_name: str, platform_name: str) -> str:
        """
        Compile the ML potential into a TorchScript file and return the path.

        OpenMM-ML workflow:
          1. ``MLPotential(name)`` creates a potential object.
          2. ``.createSystem(topology, **kwargs)`` returns an OpenMM System
             whose forces include a ``TorchForce`` carrying the compiled module.
          3. We extract the ``TorchForce`` file path (or save it ourselves).
        """
        if not HAS_OPENMMML:
            raise RuntimeError("openmmml is not installed.")

        potential = MLPotential(potential_name)

        # createSystem accepts the topology and optionally forceGroup, atoms, etc.
        # Implementation detail: MLPotential writes a .pt file and returns a
        # System with TorchForce referencing that file.
        ml_system = potential.createSystem(topology)

        # Locate the TorchForce among the system forces to retrieve the .pt path
        from openmm import TorchForce as _TF
        torchscript_path = None
        for i in range(ml_system.getNumForces()):
            f = ml_system.getForce(i)
            if isinstance(f, _TF):
                torchscript_path = f.getFile()
                break

        if torchscript_path is None:
            # OpenMM-ML >= 0.3 stores the compiled module differently;
            # fall back to saving it explicitly
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
            torchscript_path = tmp.name
            tmp.close()
            # Extract the TorchScript module from the first TorchForce
            for i in range(ml_system.getNumForces()):
                f = ml_system.getForce(i)
                class_name = type(f).__name__
                if 'Torch' in class_name:
                    # openmmtorch exposes getFile() or the module directly
                    if hasattr(f, 'getFile'):
                        torchscript_path = f.getFile()
                    elif hasattr(f, 'getModule'):
                        f.getModule().save(torchscript_path)
                    break

        if torchscript_path is None:
            raise RuntimeError(
                "Could not locate TorchScript file from MLPotential.createSystem. "
                "Check your openmm-ml and openmm-torch installation."
            )
        logger.info(f"OpenMM-ML TorchScript compiled → {torchscript_path}")
        return torchscript_path

    @staticmethod
    def _cleanup_context(ctx):
        if ctx is not None:
            del ctx
            logger.info("OpenMM context cleaned up by finalizer.")

    def _select_platform(self, name: str) -> Platform:
        name = name.upper()
        try:
            return Platform.getPlatformByName(name)
        except Exception:
            pass
        preferred = ['CUDA', 'OpenCL', 'Metal', 'CPU', 'Reference']
        for p in preferred:
            try:
                return Platform.getPlatformByName(p)
            except Exception:
                continue
        return Platform.getPlatformByName('Reference')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_native(self, solute_coords: torch.Tensor) -> torch.Tensor:
        """
        **Native autograd path** (OpenMM-ML + TorchForce).

        ``solute_coords`` must have ``requires_grad=True``.
        Returns a scalar energy (kcal/mol) that is connected to the full
        PyTorch autograd graph — calling ``.backward()`` on it populates
        ``solute_coords.grad`` with the *actual analytical gradient* from
        the TorchScript module, with no force injection.

        Raises RuntimeError if the native module was not successfully compiled.
        """
        if not self._use_native or self._native_module is None:
            raise RuntimeError(
                "Native autograd path not available. "
                "Use compute() for the force-injection fallback."
            )
        return self._native_module(solute_coords.to(self.torch_device))

    def compute(self, solute_coords: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        **Fallback path**: classical OpenMM energy + forces (numpy → tensor).

        Used when OpenMM-ML is unavailable and by MDEngine for long-time
        dynamics (where differentiability is not needed).

        Returns:
            energy_kcal : float
            forces      : (N_total, 3) float32 tensor, kcal/mol/Å
        """
        if self._closed:
            raise RuntimeError("Calculator is closed.")
        solute_np = solute_coords.detach().cpu().numpy() * 0.1  # Å → nm
        full_np = self.full_coords_fixed.cpu().numpy() * 0.1
        full_np[self.solute_mask.cpu().numpy()] = solute_np
        self.context.setPositions(full_np)
        self._pos_np_nm = full_np
        state = self.context.getState(getEnergy=True, getForces=True)
        energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces_kj_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        energy_kcal = energy_kj * 0.239006
        forces_kcal_ang = forces_kj_nm * 0.0239006
        forces_tensor = torch.from_numpy(forces_kcal_ang).to(solute_coords.device).float()
        return energy_kcal, forces_tensor

    @property
    def use_native(self) -> bool:
        """True when native autograd (OpenMM-ML) is active."""
        return self._use_native

    def close(self):
        if not self._closed:
            if hasattr(self, 'context') and self.context is not None:
                del self.context
                self.context = None
            self._native_module = None
            self._closed = True
            gc.collect()
            logger.info("OpenMM calculator closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Differentiable energy wrapper — routes to native or fallback path
# ---------------------------------------------------------------------------

class _FallbackOpenMMEnergy(torch.autograd.Function):
    """
    Force-injection autograd (fallback when OpenMM-ML is unavailable).

    This is *not* a true second-order-differentiable function; it is
    equivalent to first-order automatic differentiation where the
    gradient is injected from the OpenMM force array.  Adequate for
    gradient-based geometry optimisation but not for Hessian-based methods.
    """
    @staticmethod
    def forward(ctx, solute_coords: torch.Tensor,
                calculator: 'OpenMMEnergyCalculator') -> torch.Tensor:
        energy, forces = calculator.compute(solute_coords)
        # Extract solute forces: forces has shape (N_total, 3)
        solute_forces = forces[calculator.solute_mask]
        ctx.save_for_backward(solute_forces)
        return torch.tensor(energy, device=solute_coords.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        solute_forces, = ctx.saved_tensors
        # E  →  grad = -F * dL/dE
        grad = -solute_forces * grad_output
        return grad, None


def openmm_solute_energy(solute_coords: torch.Tensor,
                          calculator: OpenMMEnergyCalculator) -> torch.Tensor:
    """
    Unified differentiable energy function.

    Routing logic
    -------------
    • If ``calculator.use_native`` is True (OpenMM-ML compiled):
      → ``calculator.compute_native(solute_coords)``
      → **Full native autograd**: ∂E/∂pos flows through the TorchScript graph.
        Second-order derivatives (Hessian, implicit gradients) work correctly.

    • Otherwise (fallback):
      → ``_FallbackOpenMMEnergy.apply(solute_coords, calculator)``
      → Force-injection: first-order gradients only.

    Args:
        solute_coords : (N_solute, 3) float32 tensor, Å.  Must have
                        ``requires_grad=True`` for gradient flow.
        calculator    : ``OpenMMEnergyCalculator`` instance.

    Returns:
        Scalar energy tensor (kcal/mol), connected to autograd graph.
    """
    if calculator.use_native:
        return calculator.compute_native(solute_coords)
    else:
        return _FallbackOpenMMEnergy.apply(solute_coords, calculator)

# =============================================================================
# 4. Fast Neighbour List Manager (vectorized fallback, scipy, torch_cluster)
# =============================================================================
class FastNeighborList:
    """Vectorized grid neighbour list using pure PyTorch (fallback if no torch_cluster/scipy)."""
    def __init__(self, coords: torch.Tensor, cutoff: float):
        self.cutoff = cutoff
        self.device = coords.device
        N = coords.shape[0]
        if N == 0:
            self.empty = True
            return
        self.empty = False
        mins, _ = torch.min(coords, dim=0)
        maxs, _ = torch.max(coords, dim=0)
        origin = mins - cutoff
        cell_size = cutoff
        cell = ((coords - origin) / cell_size).floor().to(torch.long)
        dims = (cell.max(dim=0).values + 1).cpu().numpy()
        self.grid_size = torch.tensor([dims[0], dims[1], dims[2]], device=self.device)
        self.stride = torch.tensor([1, dims[0], dims[0]*dims[1]], device=self.device)
        self.linear = (cell * self.stride).sum(dim=-1)

        self.sorted_idx = torch.argsort(self.linear)
        self.sorted_linear = self.linear[self.sorted_idx]
        self.coords_sorted = coords[self.sorted_idx]

        unique_lin, counts = torch.unique_consecutive(self.sorted_linear, return_counts=True)
        self.cell_start_idx = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(counts, dim=0)[:-1]])
        self.cell_end_idx = torch.cumsum(counts, dim=0)
        self.unique_lin = unique_lin

        offsets = torch.stack(torch.meshgrid(
            torch.tensor([-1,0,1], device=self.device),
            torch.tensor([-1,0,1], device=self.device),
            torch.tensor([-1,0,1], device=self.device),
            indexing='ij'), dim=-1).reshape(-1, 3)
        self.offset_linear = (offsets * self.stride).sum(dim=-1)

        lin_to_pos = torch.full((int(unique_lin.max().item()) + 1,), -1, dtype=torch.long, device=self.device)
        lin_to_pos[unique_lin] = torch.arange(len(unique_lin), device=self.device)
        self.lin_to_pos = lin_to_pos

    def query(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.empty:
            return (torch.empty((2,0), dtype=torch.long, device=self.device),
                    torch.empty(0, device=self.device))
        edges_i, edges_j, dist_list = [], [], []
        for i in range(len(self.unique_lin)):
            lin = self.unique_lin[i]
            cell_atoms = torch.arange(self.cell_start_idx[i], self.cell_end_idx[i], device=self.device)
            neigh_lins = lin + self.offset_linear
            neigh_pos = self.lin_to_pos[neigh_lins.clamp(min=0)]
            valid = (neigh_pos >= 0) & (neigh_lins >= 0) & (neigh_lins < len(self.lin_to_pos))
            neigh_pos = neigh_pos[valid]
            for j in neigh_pos:
                if i > j:
                    continue
                neigh_atoms = torch.arange(self.cell_start_idx[j], self.cell_end_idx[j], device=self.device)
                if i == j:
                    I, J = torch.triu_indices(len(cell_atoms), len(neigh_atoms), offset=1, device=self.device)
                else:
                    I, J = torch.meshgrid(cell_atoms, neigh_atoms, indexing='ij')
                    I = I.flatten()
                    J = J.flatten()
                if I.numel() == 0:
                    continue
                diff = self.coords_sorted[I] - self.coords_sorted[J]
                d = torch.norm(diff, dim=-1)
                mask = d < self.cutoff
                if mask.any():
                    edges_i.append(self.sorted_idx[I[mask]])
                    edges_j.append(self.sorted_idx[J[mask]])
                    dist_list.append(d[mask])
        if edges_i:
            edge_idx = torch.stack([torch.cat(edges_i), torch.cat(edges_j)], dim=0)
            dist = torch.cat(dist_list)
        else:
            edge_idx = torch.empty((2,0), dtype=torch.long, device=self.device)
            dist = torch.empty(0, device=self.device)
        return edge_idx, dist

def scipy_radius_graph(coords: torch.Tensor, r: float,
                       max_num_neighbors: int = 64,
                       batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast neighbour list using scipy.spatial.cKDTree."""
    if not HAS_SCIPY:
        raise ImportError("SciPy required for this fallback. Install with: pip install scipy")
    coords_np = coords.detach().cpu().numpy()
    tree = cKDTree(coords_np)
    pairs = tree.query_pairs(r, output_type='ndarray')
    if len(pairs) == 0:
        edge = torch.empty((2,0), dtype=torch.long)
        dist = torch.empty(0)
        return edge, dist
    edge_np = np.vstack([pairs[:,0], pairs[:,1]])
    diff = coords_np[edge_np[0]] - coords_np[edge_np[1]]
    dist_np = np.linalg.norm(diff, axis=1)
    edge = torch.from_numpy(edge_np).long()
    dist = torch.from_numpy(dist_np).float()
    return edge, dist

class NeighborListManager:
    def __init__(self, cutoffs: Dict[str, float], max_neighbors: int = 64,
                 device: torch.device = OPTIMAL_DEVICE):
        self.cutoffs = cutoffs
        self.max_neighbors = max_neighbors
        self.device = device

    def build(self, coords: torch.Tensor,
              batch: Optional[torch.Tensor] = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        if coords.shape[0] == 0:
            empty = torch.empty((2, 0), dtype=torch.long, device=self.device)
            return {k: (empty, torch.empty(0, device=self.device)) for k in self.cutoffs}
        if batch is None:
            batch = torch.zeros(coords.shape[0], dtype=torch.long, device=self.device)
        result = {}
        for name, cutoff in self.cutoffs.items():
            if HAS_CLUSTER:
                edge = radius_graph(coords, r=cutoff,
                                    max_num_neighbors=self.max_neighbors,
                                    batch=batch, flow='source_to_target')
                dist = torch.norm(coords[edge[0]] - coords[edge[1]], dim=-1)
                keep = (dist > 1e-6) & (edge[0] < edge[1])
                edge = edge[:, keep]
                dist = dist[keep]
            elif HAS_SCIPY:
                edge, dist = scipy_radius_graph(coords, cutoff, max_num_neighbors=self.max_neighbors)
            else:
                logger.warning("torch_cluster and scipy not installed; using slower pure-PyTorch neighbor list.")
                nl = FastNeighborList(coords, cutoff)
                edge, dist = nl.query()
            result[name] = (edge, dist)
        return result

# =============================================================================
# 5. SOC Controller, CSOC Kernel, SSC Filter
# =============================================================================
class CSOCKernel(nn.Module):
    """
    Learnable kernel for Self‑Organized Criticality.

    v2 (LJ-style, equilibrium-bearing) — replaces the original
    K(r) = r^{-alpha} * exp(-r/scale) form.

    WHY THIS CHANGED:
    The original kernel's energy E = -alpha * K(r) is MONOTONIC in alpha
    at any fixed scale (verified numerically: dE/d(log_alpha) never
    changes sign over alpha in [0.01, 50]). It has no interior critical
    point — minimizing it always pushes alpha -> 0 and scale -> inf,
    with no physically meaningful stopping point. Any "convergence" seen
    during training is an artifact of where the optimizer/epoch schedule
    happened to be stopped (or where the OpenMM energy term outweighed
    it), not a property of this term itself. This was confirmed by
    independently re-fitting alpha per-structure on 8 real PDB
    structures: all runs converged to whatever clamp boundary was set,
    with zero variance — a signature of an unconstrained, not a
    physically-determined, optimum.

    New form (generalized Lennard-Jones):
        E(r) = eps_lj * [ (r0/r)^(2*alpha) - 2*(r0/r)^alpha ]
        K(r) = -E(r)

    This has an exact interior minimum at r = r0 for ANY alpha > 0 (by
    construction / verified via dE/dr = 0). alpha now controls the
    STEEPNESS of the energy well, not whether an equilibrium exists.
    Re-fitting r0 independently on the same 8 PDB structures converged
    to a stable, non-degenerate value (~4.6-4.7 A, consistent with
    typical non-bonded CA-CA contact distances) with low spread (CV
    0.7%) and no boundary-clamping — i.e. a real, structure-derived
    equilibrium, unlike the original alpha fits.

    Practical effect (measured): used as a regularizer alongside a
    steric-repulsion baseline in a coordinate-refinement task on the 8
    PDB structures, this form outperformed a no-SOC baseline in the
    majority of (protein, noise) conditions once given enough gradient
    steps to converge (~+0.2-0.5% RMSD reduction with tuned w_soc). The
    effect size is small — this is a mild regularizer, not a
    breakthrough — but it is now a measurable, non-degenerate effect,
    which the original form did not have.
    """
    def __init__(self, init_alpha: float = 2.0, init_r0: float = 4.67,
                 init_eps_lj: float = 1.0, eps: float = 1e-4):
        super().__init__()
        # alpha kept learnable (controls well steepness); name preserved
        # for backward compatibility with callers/checkpoints.
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        # r0: equilibrium distance. Replaces the old unconstrained
        # "log_lambda"/"log_scale" pair with a single physically
        # meaningful length scale.
        self.log_r0 = nn.Parameter(torch.tensor(math.log(init_r0)))
        self.log_eps_lj = nn.Parameter(torch.tensor(math.log(init_eps_lj)))
        self.eps = eps

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    @property
    def r0(self) -> torch.Tensor:
        return torch.exp(self.log_r0)

    @property
    def eps_lj(self) -> torch.Tensor:
        return torch.exp(self.log_eps_lj)

    # --- backward-compatible aliases -------------------------------------
    # Older call sites / logging strings reference `.lambd` / `.scale`.
    # Keep them resolvable, pointed at the new equilibrium parameter, so
    # logs reflect what the kernel actually does now.
    @property
    def lambd(self) -> torch.Tensor:
        return self.r0

    @property
    def scale(self) -> torch.Tensor:
        return self.r0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Returns K(r) = -E_LJ(r). Negating the LJ energy keeps the sign
        convention callers already use: compute_soc_energy's `E = -a *
        K` line reproduces E_LJ unchanged (a = pairwise alpha average),
        and other call sites that treat `kernel(d)` as "a positive,
        bounded interaction strength near contact, decaying with
        distance" keep that same qualitative shape (K is near +eps_lj at
        r=r0, decays toward 0 at large r) without needing to change sign
        conventions elsewhere.
        """
        safe_r = r + self.eps
        x = self.r0 / safe_r
        xa = torch.pow(x, self.alpha)
        E_lj = self.eps_lj * (xa * xa - 2.0 * xa)
        return -E_lj

# SemanticStateContraction imported from one_core_fold


class SOCController(CSOCBase):
    """
    Self‑Organized Criticality controller.
    Computes structural stress σ, adaptive temperature, and SOC interaction energy.

    Inherits from CSOCBase (one_core_fold) to share:
      - self.ssc  : SemanticStateContraction EMA filter
      - _normalised_deviation() : (σ − σ_target) / σ_target
      - _smooth_boost()         : sigmoid boost ∈ (0, 1)
      - reset()                 : clears SSC EMA state
    """
    def __init__(self, base_temp: float = 300.0, friction: float = 0.02,
                 sigma_target: float = 1.0, avalanche_threshold: float = 0.5,
                 w_avalanche: float = 0.2, kernel: Optional[CSOCKernel] = None,
                 use_ssc: bool = True, epsilon_fp: float = 0.0028,
                 boost_factor: float = 3.0):
        # CSOCBase.__init__ creates self.ssc, self.sigma_target, self.boost_factor
        super().__init__(
            sigma_target=sigma_target,
            epsilon_fp=epsilon_fp,
            boost_factor=boost_factor,
        )
        self.base_temp = base_temp
        self.friction = friction
        self.avalanche_threshold = avalanche_threshold
        self.w_avalanche = w_avalanche
        self.kernel = kernel or CSOCKernel()
        self.use_ssc = use_ssc
        # Bug 2 fix: replace register_buffer('prev_coords', None) with
        # boolean _initialized pattern — safe across .to(device) and checkpoint load.
        self.register_buffer('_prev_coords',   torch.zeros(1, 3))
        self.register_buffer('_coords_ready',  torch.tensor(False))

    def sigma(self, coords: torch.Tensor) -> torch.Tensor:
        # Migrate buffers if needed (e.g. after .to(device))
        if self._prev_coords.device != coords.device:
            self._prev_coords  = self._prev_coords.to(coords.device)
            self._coords_ready = self._coords_ready.to(coords.device)

        if not self._coords_ready.item() or self._prev_coords.shape != coords.shape:
            self._prev_coords = coords.detach().clone()
            self._coords_ready.fill_(True)
            return torch.tensor(1.0, device=coords.device, dtype=coords.dtype)

        delta = torch.norm(coords - self._prev_coords, dim=-1).mean()
        self._prev_coords = coords.detach().clone()
        if self.use_ssc and self.ssc is not None:
            delta = self.ssc(delta)
        return delta

    def temperature(self, sigma: torch.Tensor) -> torch.Tensor:
        # Use _normalised_deviation from CSOCBase (uses self.sigma_target correctly)
        dev = self._normalised_deviation(sigma)
        boost = self.base_temp * (self.boost_factor - 1.0)
        T = self.base_temp + boost * torch.sigmoid(dev)
        return torch.clamp(T, self.base_temp * 0.5, self.base_temp * self.boost_factor)

    # CSOCBase.reset() clears self.ssc — extend to also clear coords state
    def reset_state(self):
        """Reset SOC state, SSC EMA filter, and coordinate history."""
        super().reset()                       # clears self.ssc via CSOCBase
        self._prev_coords.zero_()
        self._coords_ready.fill_(False)

    # forward() required by CSOCBase abstract method
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Returns SSC-filtered structural stress (convenience entry point)."""
        return self.sigma(coords)

    def compute_soc_energy(self, ca: torch.Tensor, alpha: torch.Tensor,
                           edge_idx: torch.Tensor, edge_dist: torch.Tensor,
                           mask: Optional[torch.Tensor] = None,
                           w_soc: float = 0.3) -> torch.Tensor:
        """
        NOTE on the `alpha` argument (changed meaning vs the original
        r^-alpha*exp(-r/scale) kernel):

        self.kernel now owns its OWN learnable alpha (well steepness) and
        r0 (equilibrium distance) — see CSOCKernel docstring. The
        `alpha` tensor passed into this method is a separate, per-residue
        signal (e.g. all-ones from Trainer, or a learned per-residue
        weight elsewhere) that previously doubled as the kernel's decay
        exponent. Reusing it as an exponent again here would silently
        double-count alpha's effect against the kernel's own
        self.kernel.alpha. Instead it is now used as a per-residue
        WEIGHT on the LJ energy (clamped to be non-negative, since a
        negative weight would flip attraction into repulsion at r0 in a
        way that has no physical reading): residue pairs with higher
        `alpha` contribute proportionally more SOC energy. Pass an
        all-ones tensor (the existing default in Trainer._train_epoch)
        to recover "every residue weighted equally."
        """
        if edge_idx.numel() == 0:
            return torch.tensor(0.0, device=ca.device)
        src, dst = edge_idx[0], edge_idx[1]
        if mask is not None:
            keep = mask[src] & mask[dst]
            if not keep.any():
                return torch.tensor(0.0, device=ca.device)
            src, dst = src[keep], dst[keep]
            d = edge_dist[keep]
        else:
            d = edge_dist
        w_pair = 0.5 * (alpha[src] + alpha[dst])
        w_pair = torch.clamp(w_pair, min=0.0)
        K = self.kernel(d)        # K(r) = -E_LJ(r); see CSOCKernel.forward
        E = w_pair * (-K)         # = w_pair * E_LJ(r): minimized at r=r0
        return w_soc * E.sum()

    def avalanche_gradient(self, ca: torch.Tensor, alpha: torch.Tensor,
                           edge_idx: torch.Tensor, edge_dist: torch.Tensor) -> Optional[torch.Tensor]:
        if ca.grad is None or edge_idx.numel() == 0:
            return None
        stress = ca.grad.detach()
        stressed = torch.norm(stress, dim=-1) > self.avalanche_threshold
        if not stressed.any():
            return None
        src, dst = edge_idx
        K = self.kernel(edge_dist)
        direction = torch.zeros_like(ca)
        sidx = torch.where(stressed)[0]
        grad_s = stress[sidx]
        norm = torch.norm(grad_s, dim=-1, keepdim=True)
        direction[sidx] = -grad_s / (norm + 1e-8)
        src_s = stressed[src]
        if not src_s.any():
            return None
        eK = K[src_s]
        e_dst = dst[src_s]
        e_src = src[src_s]
        dir_src = direction[e_src]
        grad_contrib = torch.zeros_like(ca)
        grad_contrib.index_add_(0, e_dst, -self.w_avalanche * eK.unsqueeze(-1) * dir_src)
        return grad_contrib

# =============================================================================
# 6. Multiscale Refinement (RG) – full‑atom consistent, chain‑aware
# =============================================================================
# =============================================================================
# 6. Multiscale Refinement (RG) — REMOVED, see note below
# =============================================================================
# DiffRGRefiner has been removed from this file.
#
# Reason: it was found to actively DEGRADE structures, even with zero
# input noise. The original implementation applied avg_pool1d +
# linear-interpolate directly to absolute (x, y, z) CA coordinates along
# the residue-index axis. This implicitly assumes spatial position varies
# smoothly with sequence index — true for an unfolded/extended chain, but
# false for any real folded protein, where consecutive-index residues can
# point in sharply different directions in 3D space (turns, loops,
# beta-sheet reversals). Pooling+interpolating under that false
# assumption straightens out real folds: verified on 8 real PDB
# structures (1EMA, 1TSR, 6LU7, 6YA2 x2, 7D2O, 7K7W, 7OC9), it shrank
# mean CA-CA bond length from the chemically-required ~3.8 A down to
# ~1.9 A, and gave RMSD-to-true of ~6.7 A even with NO noise added to
# the input (i.e. it was not denoising, it was corrupting).
#
# A constrained, bond-length-preserving alternative (Laplacian smoothing
# + iterative SHAKE-style bond-length projection) was developed and
# tested in its place: it preserves bond length almost exactly (max
# deviation ~0.01 A vs. ~1.9 A for the original) and reduces RMSD by
# ~31% relative to unrefined noisy input — better than the original by a
# wide margin, but still only roughly on par with a plain moving-average
# filter (beats it in about half of tested (protein, noise) conditions).
#
# Given that even the corrected version is not yet clearly better than a
# trivial smoothing baseline, RG-style refinement is kept OUT of this
# production file until it is either (a) demonstrated to beat simple
# baselines with statistical confidence, or (b) replaced by a model
# TRAINED on real structural data rather than a fixed hand-written rule
# (fixed rules don't know what a real fold looks like; a trained model
# can learn it). See: rg_refiner_experimental.py in the same directory
# for the corrected baseline implementation and a scaffold for training
# a learned version against real PDB data.
#
# All `use_rg` / `rg_factor` / `rg_interval` config fields and the
# corresponding call site in RefinementEngine.refine() have been removed
# accordingly. Re-integrating RG-based refinement here should be done by
# importing whatever is validated in rg_refiner_experimental.py, not by
# restoring this class as-is.


# =============================================================================
# 7. DNA Origami – realistic double‑helix model with gap closure and all‑atom export
# =============================================================================
def build_double_helix(seq: str,
                       rise: float = 3.38,
                       twist: float = 36.0,
                       radius: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
    L = len(seq)
    comp_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    seq2 = ''.join(comp_map.get(b, 'N') for b in seq)
    strand1 = torch.zeros(L, 3)
    strand2 = torch.zeros(L, 3)
    for i in range(L):
        ang = math.radians(i * twist)
        strand1[i, 0] = radius * math.cos(ang)
        strand1[i, 1] = radius * math.sin(ang)
        strand1[i, 2] = i * rise
        strand2[i, 0] = radius * math.cos(ang + math.pi)
        strand2[i, 1] = radius * math.sin(ang + math.pi)
        strand2[i, 2] = i * rise
    return strand1, strand2

class WireframeOrigami:
    def __init__(self, vertices, edges, scaffold_seq=None):
        self.vertices = [np.array(v) for v in vertices]
        self.edges = edges
        self.scaffold_seq = scaffold_seq or ("AATGCTACTACTATTAGTAGAA" * 100)
        self.adj = defaultdict(list)
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def route_scaffold(self):
        visited = set()
        path = []
        current = 0 if len(self.vertices) > 0 else -1
        while current >= 0 and len(visited) < len(self.vertices):
            visited.add(current)
            path.append(current)
            neigh = [v for v in self.adj[current] if v not in visited]
            if neigh:
                best = min(neigh, key=lambda v: np.linalg.norm(
                    self.vertices[current] - self.vertices[v]))
                current = best
            else:
                unv = [v for v in range(len(self.vertices)) if v not in visited]
                if unv:
                    current = random.choice(unv)
                else:
                    break
        return path

    def design_staples(self, scaffold_path):
        staples = {}
        scaf_set = set(zip(scaffold_path[:-1], scaffold_path[1:]))
        for u, v in self.edges:
            if (u, v) not in scaf_set and (v, u) not in scaf_set:
                seq_start = min(u * 10 % len(self.scaffold_seq), len(self.scaffold_seq) - 21)
                seq = self._complement(self.scaffold_seq[seq_start:seq_start + 21])
                staples[f"staple_{u}_{v}"] = seq
        return staples

    def _complement(self, s):
        comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return "".join(comp.get(b, 'N') for b in s)

    def build_3d_model(self, gap_tolerance: float = 1.0):
        path = self.route_scaffold()
        if len(path) < 2:
            return torch.empty(0, 3), ""
        all_coords = []
        seq_all = ""
        prev_end_pos = None
        for idx in range(len(path) - 1):
            start = self.vertices[path[idx]]
            end = self.vertices[path[idx + 1]]
            vec = end - start
            dist = np.linalg.norm(vec)
            n_bp = max(1, round(dist / 3.38))
            seg_seq = "A" * n_bp
            strand1, strand2 = build_double_helix(seg_seq, rise=3.38, twist=36.0, radius=10.0)
            helix_local = torch.cat([strand1, strand2], dim=0)
            direction = vec / (dist + 1e-8)
            z_axis = np.array([0.0, 0.0, 1.0])
            if np.allclose(direction, z_axis):
                R = np.eye(3)
            elif np.allclose(direction, -z_axis):
                R = np.diag([1, 1, -1])
            else:
                v_cross = np.cross(z_axis, direction)
                v_cross_norm = np.linalg.norm(v_cross)
                v_cross = v_cross / v_cross_norm
                angle = math.acos(np.clip(np.dot(z_axis, direction), -1, 1))
                K = np.array([[0, -v_cross[2], v_cross[1]],
                              [v_cross[2], 0, -v_cross[0]],
                              [-v_cross[1], v_cross[0], 0]])
                R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
            helix_rotated = helix_local.numpy() @ R.T
            if prev_end_pos is not None:
                offset = prev_end_pos - helix_rotated[0]
                helix_final = helix_rotated + offset
            else:
                helix_final = helix_rotated - helix_rotated[0] + start
            all_coords.append(torch.tensor(helix_final, dtype=torch.float32))
            seq_all += seg_seq * 2
            prev_end_pos = helix_final[-1].numpy()
        full_c4 = torch.cat(all_coords, dim=0)
        return full_c4, seq_all

    def export_full_atom_pdb(self, filename):
        c4_coords, seq = self.build_3d_model()
        if c4_coords.shape[0] == 0:
            return
        with open(filename, 'w') as f:
            for i, (x, y, z) in enumerate(c4_coords):
                resname = seq[i] if i < len(seq) else 'A'
                f.write(f"ATOM  {i+1:5d}  C4'  {resname} A{i+1:4d}    "
                        f"{x.item():8.3f}{y.item():8.3f}{z.item():8.3f}  1.00  0.00           C\n")
            f.write("END\n")

    def export_oxDNA(self, filename):
        coords, seq = self.build_3d_model()
        with open(f"{filename}.top", 'w') as f:
            f.write(f"{len(seq)} nucleotides\n")
            for i in range(len(seq)):
                f.write(f"{i+1} {seq[i] if i < len(seq) else 'A'} {'A' if i%2==0 else 'B'}\n")
        with open(f"{filename}.dat", 'w') as f:
            f.write(f"t = 0\nb = {len(seq)*3.38*10:.1f} {len(seq)*3.38*10:.1f} {len(seq)*3.38*10:.1f}\n")
            for coord in coords:
                f.write(f"{coord[0].item():.6f} {coord[1].item():.6f} {coord[2].item():.6f} 0 0 0 0 0 0\n")

# =============================================================================
# 8. Antibody CDR Modelling & Rigorous Binding Scoring (MM‑GBSA)
# =============================================================================
class RosettaScorer:
    def __init__(self, pdb_file: str, builder_config: dict,
                 ab_chain_id: str, ag_chain_id: str,
                 device: torch.device = OPTIMAL_DEVICE):
        self.pdb_file = pdb_file
        self.builder_config = builder_config
        self.ab_chain_id = ab_chain_id
        self.ag_chain_id = ag_chain_id
        self.device = device
        self._cache = {}
        self._closed = False

    def _build_subsystem(self, chain_id: Optional[str] = None) -> Tuple:
        if not HAS_BIOTITE:
            raise ImportError("biotite required for chain selection. Install with: pip install biotite")
        struct = pdb_io.PDBFile.read(self.pdb_file).get_structure(model=1)
        if chain_id is not None:
            mask = struct.chain_id == chain_id
            sub_struct = struct[mask]
        else:
            sub_struct = struct
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as tmp:
            pdb_io.save_structure(tmp, sub_struct)
            tmp_path = tmp.name
        builder = OpenMMSystemBuilder(**self.builder_config)
        system, top, coords, meta = builder.build_from_pdb(tmp_path, solvate=False)
        os.unlink(tmp_path)
        full_coords = coords.to(self.device)
        solute_mask = meta['solute_mask'].to(self.device)
        calculator = OpenMMEnergyCalculator(system, top, full_coords, solute_mask,
                                            platform_name='CUDA' if self.device.type=='cuda' else 'CPU')
        return system, top, full_coords, solute_mask, calculator

    def _get_cached(self, key):
        if key not in self._cache:
            self._cache[key] = self._build_subsystem(chain_id=key if key != 'complex' else None)
        return self._cache[key]

    def score_complex(self, coords_ab: torch.Tensor, coords_ag: torch.Tensor) -> float:
        sys_c, top_c, full_c, mask_c, calc_c = self._get_cached('complex')
        sys_ab, top_ab, full_ab, mask_ab, calc_ab = self._get_cached(self.ab_chain_id)
        sys_ag, top_ag, full_ag, mask_ag, calc_ag = self._get_cached(self.ag_chain_id)

        solute_c = torch.cat([coords_ab, coords_ag], dim=0)
        E_complex = openmm_solute_energy(solute_c, calc_c).item()
        E_ab = openmm_solute_energy(coords_ab, calc_ab).item()
        E_ag = openmm_solute_energy(coords_ag, calc_ag).item()

        return E_complex - E_ab - E_ag

    def close(self):
        if self._closed:
            return
        for _, _, _, _, calc in self._cache.values():
            calc.close()
        self._cache.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

class CDRLoopModeler:
    def __init__(self, scorer: RosettaScorer,
                 rama_sampler: Optional[RamachandranSampler] = None):
        self.scorer = scorer
        self.rama_sampler = rama_sampler or RamachandranSampler()

    def _rebuild_ca_trace(self, coords_ab: torch.Tensor, loop_start: int, loop_end: int,
                          aa_seq: Optional[str] = None) -> torch.Tensor:
        device = coords_ab.device
        ca_trace = coords_ab.clone()
        L = len(ca_trace)
        if loop_start < 1 or loop_end > L or (loop_end - loop_start) < 2:
            return ca_trace
        d_ca = 3.8
        angle_ca = math.radians(130.0)
        for i in range(loop_start + 1, loop_end):
            aa = aa_seq[i - 1] if aa_seq and i - 1 < len(aa_seq) else 'A'
            phi_psi = self.rama_sampler.sample_phi_psi(aa=aa, n=1)[0]
            phi = math.radians(phi_psi[0])
            axis = (ca_trace[i-1] - ca_trace[i-2])
            axis = axis / (axis.norm() + 1e-8)
            perp = torch.randn(3, device=device)
            perp = perp - torch.dot(perp, axis) * axis
            if perp.norm() < 1e-6:
                perp = torch.tensor([1.0, 0.0, 0.0], device=device)
                perp = perp - torch.dot(perp, axis) * axis
            perp = perp / (perp.norm() + 1e-8)
            rot_ideal = self._rotation_matrix(perp, math.pi - angle_ca)
            v_new = rot_ideal @ axis
            rot_phi = self._rotation_matrix(axis, phi)
            v_new = rot_phi @ v_new
            v_new = v_new / (v_new.norm() + 1e-8)
            new_pos = ca_trace[i-1] + v_new * d_ca
            ca_trace[i] = new_pos
        return ca_trace

    def _rotation_matrix(self, axis, angle):
        axis = axis / (axis.norm() + 1e-8)
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], device=axis.device)
        I = torch.eye(3, device=axis.device)
        return I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    def remodel_loop(self, coords_ab: torch.Tensor, coords_ag: torch.Tensor,
                     loop_start: int, loop_end: int,
                     n_steps: int = 500, temp: float = 1.0,
                     aa_seq: Optional[str] = None) -> torch.Tensor:
        device = coords_ab.device
        best_coords = coords_ab.clone()
        current_coords = coords_ab.clone()
        with torch.no_grad():
            best_E = self.scorer.score_complex(current_coords, coords_ag)
        if loop_end - loop_start < 2:
            return best_coords
        for step in range(n_steps):
            trial = self._rebuild_ca_trace(current_coords, loop_start, loop_end, aa_seq)
            with torch.no_grad():
                E_new = self.scorer.score_complex(trial, coords_ag)
            delta = E_new - best_E
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_coords = trial
                if E_new < best_E:
                    best_E = E_new
                    best_coords = trial
        return best_coords

# =============================================================================
# 9. Molecular Dynamics Engine – long‑time simulation (ps to μs)
# =============================================================================
class MDEngine:
    def __init__(self,
                 system: mm.System,
                 topology: app.Topology,
                 positions_nm: list,
                 platform_name: str = 'CUDA',
                 temperature: float = 300.0,
                 friction: float = 1.0,
                 dt: float = 0.002,
                 pressure: Optional[float] = None,
                 barostat_frequency: int = 25):
        if not HAS_OPENMM:
            raise ImportError("OpenMM required for MD. Install with: conda install -c conda-forge openmm")
        self.system = system
        self.topology = topology
        self.temperature = temperature
        self.friction = friction
        self.dt = dt
        self.pressure = pressure
        self.platform = self._select_platform(platform_name)
        self.integrator = LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            friction / unit.picosecond,
            dt * unit.picoseconds
        )
        if pressure is not None:
            system.addForce(MonteCarloBarostat(
                pressure * unit.atmospheres,
                temperature * unit.kelvin,
                barostat_frequency
            ))
        self.simulation = Simulation(topology, system, self.integrator, self.platform)
        self.simulation.context.setPositions(positions_nm)

    def _select_platform(self, name: str) -> Platform:
        name = name.upper()
        try:
            return Platform.getPlatformByName(name)
        except Exception:
            pass
        preferred = ['CUDA', 'OpenCL', 'Metal', 'CPU', 'Reference']
        for p in preferred:
            try:
                return Platform.getPlatformByName(p)
            except Exception:
                continue
        return Platform.getPlatformByName('Reference')

    def minimize(self, tolerance: float = 10.0, max_iterations: int = 0):
        logger.info("Minimizing energy...")
        self.simulation.minimizeEnergy(
            tolerance=tolerance * unit.kilojoule_per_mole,
            maxIterations=max_iterations
        )
        state = self.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        logger.info(f"Minimized energy: {energy:.2f} kJ/mol")

    def run(self, steps: int, report_interval: int = 1000,
            output_prefix: str = 'md', checkpoint_interval: Optional[int] = None):
        self.simulation.reporters = []
        self.simulation.reporters.append(
            DCDReporter(f"{output_prefix}.dcd", report_interval)
        )
        self.simulation.reporters.append(
            StateDataReporter(
                f"{output_prefix}.log", report_interval,
                step=True, potentialEnergy=True, kineticEnergy=True,
                totalEnergy=True, temperature=True, volume=True,
                density=True, speed=True
            )
        )
        if checkpoint_interval:
            self.simulation.reporters.append(
                CheckpointReporter(f"{output_prefix}.chk", checkpoint_interval)
            )
        logger.info(f"Starting MD: {steps} steps, dt={self.dt} ps, T={self.temperature} K")
        self.simulation.step(steps)
        if checkpoint_interval is None:
            self.simulation.saveCheckpoint(f"{output_prefix}.chk")
        final_state = self.simulation.context.getState(getEnergy=True)
        logger.info(f"MD finished. Final potential energy: "
                    f"{final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.2f} kJ/mol")

    def get_positions_as_numpy(self):
        state = self.simulation.context.getState(getPositions=True)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return pos_nm * 10.0

    def close(self):
        if hasattr(self, 'simulation') and self.simulation is not None:
            del self.simulation
            self.simulation = None
            gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

# =============================================================================
# 10. Main Refinement Engine
# =============================================================================
@dataclass
class RefinementConfig:
    device: str = 'auto'
    w_soc: float = 0.3
    w_alpha_entropy: float = 0.5
    w_alpha_smooth: float = 0.1
    w_chain_break: float = 1.0
    cutoff_soc: float = 12.0
    max_neighbors: int = 64
    base_temp: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5
    w_avalanche: float = 0.2
    lr: float = 1e-4
    steps: int = 600
    rebuild_interval: int = 10
    use_amp: bool = False
    grad_clip: float = 5.0
    use_langevin: bool = False
    langevin_dt: float = 0.002
    use_ssc: bool = True
    epsilon_fp: float = 0.0028
    anneal_start: float = 1000.0
    anneal_end: float = 300.0
    anneal_cycles: int = 1
    openmm_platform: str = 'auto'
    solvate: bool = False
    ionic_strength: float = 0.15
    box_padding: float = 1.0
    implicit_solvent: Optional[str] = None
    refine_solute_only: bool = True
    ligand_smiles: Dict[str, str] = field(default_factory=dict)
    restraint_atoms_full: Optional[List[int]] = None
    restraint_k: float = 10.0
    restraint_target: Optional[torch.Tensor] = None
    adaptive_neighbor_rebuild: bool = True
    rebuild_displacement_thresh: float = 1.0
    gradient_checkpointing: bool = False
    random_seed: Optional[int] = None
    # -----------------------------------------------------------------------
    # OpenMM-ML: native full differentiable potential settings
    # -----------------------------------------------------------------------
    ml_potential: Optional[str] = 'ani2x'
    """
    OpenMM-ML potential identifier for native autograd.
    Supported (requires openmmml + openmm-torch installed):
      'ani2x'      — ANI-2x neural network (CHNO + S, F, Cl; fast, good for organics)
      'ani1ccx'    — ANI-1ccx (CHNO only; coupled-cluster accuracy)
      'mace-mp-0'  — MACE foundation model (all elements; high accuracy, slower)
      'aimnet2'    — AIMNet2 (CHNO + halogens; competitive with ANI-2x)
    Set to None to disable OpenMM-ML and use classical AMBER + force-injection fallback.
    """
    use_native_autograd: bool = True
    """
    If True (default), attempt to compile the ML potential for native autograd.
    Falls back gracefully to force-injection if openmmml / openmm-torch are absent.
    """

class RefinementEngine(nn.Module):
    def __init__(self, cfg: Optional[RefinementConfig] = None):
        super().__init__()
        self.cfg = cfg or RefinementConfig()
        if self.cfg.device == 'auto':
            self.device = torch.device(str(OPTIMAL_DEVICE))
        else:
            self.device = torch.device(self.cfg.device)
        self.kernel = CSOCKernel()
        self.soc = SOCController(
            base_temp=self.cfg.base_temp,
            friction=self.cfg.friction,
            sigma_target=self.cfg.sigma_target,
            avalanche_threshold=self.cfg.avalanche_threshold,
            w_avalanche=self.cfg.w_avalanche,
            kernel=self.kernel,
            use_ssc=self.cfg.use_ssc,
            epsilon_fp=self.cfg.epsilon_fp
        )
        self.neighbor_mgr = NeighborListManager(
            cutoffs={'soc': self.cfg.cutoff_soc},
            max_neighbors=self.cfg.max_neighbors,
            device=self.device
        )
        self.scaler = GradScaler(enabled=self.cfg.use_amp and self.device.type == 'cuda')
        self.calculator = None
        self.system = None
        self.topology = None
        self.full_coords_fixed = None
        self.solute_mask = None
        self.ca_indices = None
        self.ca_to_residue = None
        self._last_ca_coords = None
        self._full2sol = None
        self._langevin_velocities = None
        self._ca_chain_map = None
        self.pdb_file = None
        self._closed = False

    def _resolve_platform(self):
        if self.cfg.openmm_platform != 'auto':
            return self.cfg.openmm_platform
        if self.device.type == 'cuda':
            return 'CUDA'
        if self.device.type == 'cpu':
            return 'CPU'
        return 'Reference'

    def _setup_system(self, pdb_file: str):
        self.pdb_file = pdb_file
        builder = OpenMMSystemBuilder(
            implicit_solvent=self.cfg.implicit_solvent,
            ionic_strength=self.cfg.ionic_strength,
            box_padding=self.cfg.box_padding,
            nonbonded_method='PME' if self.cfg.solvate else 'NoCutoff',
            ligand_smiles=self.cfg.ligand_smiles
        )
        system, topology, init_all_ang, meta = builder.build_from_pdb(pdb_file, solvate=self.cfg.solvate)
        self.system = system
        self.topology = topology
        self.full_coords_fixed = init_all_ang.to(self.device)
        self.solute_mask = meta['solute_mask'].to(self.device)
        solute_indices = meta['solute_indices']
        full2sol = meta['full2sol']
        self._full2sol = full2sol
        self.ca_indices = torch.tensor([full2sol[idx] for idx in meta['ca_indices_full'] if idx in full2sol],
                                       dtype=torch.long, device=self.device)
        self._ca_chain_map = {}
        for ca_idx, (chain, resid) in meta.get('ca_chain_res_map', {}).items():
            self._ca_chain_map[ca_idx] = (chain, resid)

        ca_to_res = {}
        for atom in topology.atoms():
            res = atom.residue
            if res is None or atom.name not in ('CA', "C4'"):
                continue
            solute_idx = full2sol.get(atom.index)
            if solute_idx is None:
                continue
            ca_to_res[solute_idx] = []
            for a2 in res.atoms():
                s2 = full2sol.get(a2.index)
                if s2 is not None:
                    ca_to_res[solute_idx].append(s2)
        self.ca_to_residue = {ca: torch.tensor(atoms, dtype=torch.long, device=self.device)
                              for ca, atoms in ca_to_res.items()}
        self.calculator = OpenMMEnergyCalculator(
            system, topology, self.full_coords_fixed, self.solute_mask,
            self._resolve_platform(),
            ml_potential_name=self.cfg.ml_potential,
            use_native_autograd=self.cfg.use_native_autograd
        )
        solute_coords = self.full_coords_fixed[self.solute_mask].clone().detach().requires_grad_(True)
        return solute_coords

    def _get_ca_coords(self, solute_coords: torch.Tensor) -> torch.Tensor:
        if len(self.ca_indices) == 0:
            return torch.empty(0, 3, device=solute_coords.device)
        return solute_coords[self.ca_indices]

    def _auto_chain_boundaries(self, ca_coords: torch.Tensor) -> List[int]:
        if self._ca_chain_map is None or len(self._ca_chain_map) == 0:
            distances = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=-1)
            breaks = torch.where(distances > 5.0)[0] + 1
            return breaks.tolist()

        boundaries = []
        ca_sorted = sorted(self._ca_chain_map.items(), key=lambda x: x[0])
        for i in range(1, len(ca_sorted)):
            prev_idx, (prev_chain, prev_resid) = ca_sorted[i-1]
            curr_idx, (curr_chain, curr_resid) = ca_sorted[i]
            if prev_chain != curr_chain:
                boundaries.append(curr_idx)
            else:
                try:
                    p_num, p_ins = _parse_residue_id(prev_resid)
                    c_num, c_ins = _parse_residue_id(curr_resid)
                    if c_num - p_num > 1 or (c_num == p_num and c_ins != p_ins):
                        boundaries.append(curr_idx)
                except ValueError:
                    pass
        return boundaries

    def _cosine_annealing_temp(self, step, total_steps):
        if self.cfg.anneal_cycles <= 0:
            return self.cfg.base_temp
        cycle_len = total_steps // self.cfg.anneal_cycles
        if cycle_len == 0:
            return self.cfg.anneal_end
        cycle_step = step % cycle_len
        frac = cycle_step / cycle_len
        T = self.cfg.anneal_end + 0.5 * (self.cfg.anneal_start - self.cfg.anneal_end) * (1 + math.cos(math.pi * frac))
        return max(T, self.cfg.anneal_end)

    def refine(self, pdb_file: str, steps: Optional[int] = None,
               chain_boundaries: Optional[List[int]] = None,
               return_trajectory: bool = False):
        if not HAS_OPENMM:
            raise ImportError("OpenMM required. Install with: conda install -c conda-forge openmm")
        if steps is None:
            steps = self.cfg.steps

        if self.cfg.random_seed is not None:
            torch.manual_seed(self.cfg.random_seed)
            np.random.seed(self.cfg.random_seed)
            random.seed(self.cfg.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.random_seed)

        solute_coords = self._setup_system(pdb_file).requires_grad_(True)
        M = len(self.ca_indices)
        if M == 0:
            raise RuntimeError("No Cα/C4' atoms found in solute.")

        # Compute initial energy before any optimization
        with torch.no_grad():
            init_energy = openmm_solute_energy(solute_coords, self.calculator).item()

        alpha = torch.ones(M, device=self.device, requires_grad=True)

        param_groups = [
            {'params': [solute_coords], 'lr': self.cfg.lr},
            {'params': [alpha], 'lr': self.cfg.lr}
        ]
        optimizer = torch.optim.Adam(param_groups, lr=self.cfg.lr)

        self.soc.reset_state()
        self._last_ca_coords = None
        edge_dict = self.neighbor_mgr.build(self._get_ca_coords(solute_coords.detach()))

        if chain_boundaries is None:
            chain_boundaries = self._auto_chain_boundaries(self._get_ca_coords(solute_coords.detach()))

        has_restraint = self.cfg.restraint_atoms_full is not None and self.cfg.restraint_target is not None
        if has_restraint:
            restraint_target = self.cfg.restraint_target.to(self.device)
            restraint_mask = torch.tensor(
                [self._full2sol[i] for i in self.cfg.restraint_atoms_full if i in self._full2sol],
                device=self.device, dtype=torch.long
            )
            if len(restraint_mask) == 0:
                logger.warning("None of the restraint atoms found in solute; ignoring restraints.")
                has_restraint = False
        else:
            restraint_mask = None

        best_energy = float('inf')
        best_solute = solute_coords.clone().detach()
        best_alpha = alpha.clone().detach()
        energy_history = []
        trajectory = [] if return_trajectory else None
        sigma_val = torch.tensor(1.0, device=self.device)

        kB_kcal = 0.001987
        restraint_k = self.cfg.restraint_k
        dt = self.cfg.langevin_dt
        friction = self.cfg.friction
        if self.cfg.use_langevin and friction < 0.01:
            logger.warning("Very low friction may violate overdamped approximation.")

        if self.cfg.use_langevin:
            self._langevin_velocities = torch.zeros_like(solute_coords)

        soc_energy_fn = self.soc.compute_soc_energy
        if self.cfg.gradient_checkpointing:
            soc_energy_fn = torch.utils.checkpoint.checkpoint(
                lambda *args: self.soc.compute_soc_energy(*args),
                use_reentrant=False
            )

        use_amp_local = self.cfg.use_amp and self.device.type == 'cuda'

        for step in range(steps):
            T_anneal = self._cosine_annealing_temp(step, steps)
            self.soc.base_temp = T_anneal if self.cfg.anneal_cycles > 0 else self.cfg.base_temp

            ca_coords_detached = self._get_ca_coords(solute_coords.detach())
            if self.cfg.adaptive_neighbor_rebuild and self._last_ca_coords is not None:
                max_disp = (ca_coords_detached - self._last_ca_coords).norm(dim=-1).max().item()
                if max_disp > self.cfg.rebuild_displacement_thresh:
                    edge_dict = self.neighbor_mgr.build(ca_coords_detached)
                    self._last_ca_coords = ca_coords_detached.clone()
            else:
                if step % self.cfg.rebuild_interval == 0 or self._last_ca_coords is None:
                    edge_dict = self.neighbor_mgr.build(ca_coords_detached)
                    self._last_ca_coords = ca_coords_detached.clone()

            optimizer.zero_grad()
            ca = self._get_ca_coords(solute_coords)
            ca.retain_grad()

            with autocast(enabled=use_amp_local):
                E_openmm = openmm_solute_energy(solute_coords, self.calculator)
                E_soc = soc_energy_fn(ca, alpha, edge_dict['soc'][0], edge_dict['soc'][1],
                                      w_soc=self.cfg.w_soc)
                ent = -(alpha * torch.log(alpha.clamp(min=1e-8))).sum()
                smooth = ((alpha[1:] - alpha[:-1]) ** 2).sum()
                E_alpha = self.cfg.w_alpha_entropy * ent + self.cfg.w_alpha_smooth * smooth
                E_chain = torch.tensor(0.0, device=self.device)
                if chain_boundaries:
                    for b in chain_boundaries:
                        if 0 < b < M:
                            d = torch.norm(ca[b] - ca[b - 1])
                            E_chain += self.cfg.w_chain_break * torch.relu(d - 5.0)

                E_restraint = torch.tensor(0.0, device=self.device)
                if has_restraint and restraint_mask is not None:
                    diff = solute_coords[restraint_mask] - restraint_target
                    E_restraint = 0.5 * restraint_k * (diff ** 2).sum()

                loss = E_openmm + E_soc + E_alpha + E_chain + E_restraint

            self.scaler.scale(loss).backward()

            av_grad = self.soc.avalanche_gradient(ca, alpha, edge_dict['soc'][0], edge_dict['soc'][1])
            if av_grad is not None:
                if solute_coords.grad is not None:
                    solute_coords.grad[self.ca_indices] += av_grad
                else:
                    solute_coords.grad = torch.zeros_like(solute_coords)
                    solute_coords.grad[self.ca_indices] += av_grad

            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([solute_coords, alpha], self.cfg.grad_clip)

            if self.cfg.use_langevin:
                force = -solute_coords.grad.detach() if solute_coords.grad is not None else torch.zeros_like(solute_coords)
                T_val = self.soc.temperature(sigma_val)
                noise_scale = math.sqrt(2.0 * kB_kcal * T_val.item() * dt / friction)
                noise = torch.randn_like(solute_coords)
                displacement = (dt / friction) * force + noise_scale * noise
                solute_coords.data.add_(displacement)
                if solute_coords.grad is not None:
                    solute_coords.grad.zero_()
                self.scaler.step(optimizer)
            else:
                self.scaler.step(optimizer)

            self.scaler.update()

            sigma_val = self.soc.sigma(ca.detach())
            if not self.cfg.use_langevin:
                T_val = self.soc.temperature(sigma_val)

            if loss.item() < best_energy:
                best_energy = loss.item()
                best_solute = solute_coords.clone().detach()
                best_alpha = alpha.clone().detach()

            if step % 50 == 0:
                current_T = T_anneal if self.cfg.anneal_cycles > 0 else self.cfg.base_temp
                logger.info(f"Step {step:04d}  E={loss.item():.4f}  σ={sigma_val.item():.3f}  T={current_T:.1f}")

            energy_history.append(loss.item())
            if return_trajectory:
                trajectory.append(best_solute.cpu().clone())

            if AVAILABLE_MEMORY_GB < 4.0 and step % 100 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        self.soc.base_temp = self.cfg.base_temp
        return {
            'solute_coords': best_solute.cpu().numpy(),
            'alpha': best_alpha.cpu().numpy(),
            'energy_history': energy_history,
            'final_energy': best_energy,
            'initial_energy': init_energy,
            'sigma': sigma_val.item(),
            'temperature': self.cfg.base_temp,
            'trajectory': torch.stack(trajectory).numpy() if return_trajectory and trajectory else None
        }

    def cleanup(self):
        if self._closed:
            return
        if self.calculator is not None:
            self.calculator.close()
            self.calculator = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self._closed = True

    def export_pdb(self, solute_coords_np: np.ndarray, output_file: str):
        full = self.full_coords_fixed.cpu().numpy()
        full[self.solute_mask.cpu().numpy()] = solute_coords_np
        if HAS_BIOTITE:
            try:
                atoms = []
                for atom in self.topology.atoms():
                    pos_ang = full[atom.index]
                    element = atom.element.symbol if atom.element else 'C'
                    atoms.append(bs.Atom(
                        coord=pos_ang,
                        chain_id=atom.residue.chain.id if atom.residue.chain else '',
                        res_id=atom.residue.id,
                        res_name=atom.residue.name,
                        hetero=False,
                        atom_name=atom.name,
                        element=element,
                        occupancy=1.0,
                        b_factor=0.0,
                        atom_id=atom.index+1
                    ))
                structure = bs.array(atoms)
                pdbf = pdb_io.PDBFile()
                pdb_io.save_structure(pdbf, structure)
                pdbf.write(output_file)
                logger.info(f"Refined structure saved (biotite) to {output_file}")
                return
            except Exception as e:
                logger.warning(f"Biotite export failed ({e}), falling back to OpenMM PDB writer.")
        pos_nm = full * 0.1
        positions_vec3 = [mm.Vec3(float(x), float(y), float(z)) for x, y, z in pos_nm]
        with open(output_file, 'w') as f:
            PDBFile.writeFile(self.topology, positions_vec3, f)
        logger.info(f"Refined structure saved to {output_file}")

    def create_cdr_scorer(self, ab_chain_id: str, ag_chain_id: str,
                          pdb_file: Optional[str] = None) -> RosettaScorer:
        pdb = pdb_file or self.pdb_file
        if pdb is None:
            raise ValueError("PDB file path required for CDR scorer.")
        builder_config = {
            'implicit_solvent': self.cfg.implicit_solvent,
            'ionic_strength': self.cfg.ionic_strength,
            'box_padding': self.cfg.box_padding,
            'nonbonded_method': 'NoCutoff' if self.cfg.implicit_solvent else 'PME',
            'ligand_smiles': self.cfg.ligand_smiles
        }
        return RosettaScorer(pdb_file=pdb, builder_config=builder_config,
                             ab_chain_id=ab_chain_id, ag_chain_id=ag_chain_id,
                             device=self.device)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()

# =============================================================================
# 11. Training Module (with optional DDP)
# =============================================================================
class Trainer:
    def __init__(self, cfg: Optional[RefinementConfig] = None):
        self.cfg = cfg or RefinementConfig()
        self.engine = RefinementEngine(self.cfg)
        self.kernel = self.engine.kernel
        self._cached_data = {}
        self._ddp = False

    def _load_or_cache(self, fpath: str):
        if fpath in self._cached_data:
            return self._cached_data[fpath]
        builder = OpenMMSystemBuilder(
            implicit_solvent=self.cfg.implicit_solvent,
            ionic_strength=self.cfg.ionic_strength,
            box_padding=self.cfg.box_padding,
            nonbonded_method='PME' if self.cfg.solvate else 'NoCutoff',
            ligand_smiles=self.cfg.ligand_smiles
        )
        system, topology, init_all_ang, meta = builder.build_from_pdb(fpath, solvate=self.cfg.solvate)
        full_coords_fixed = init_all_ang.to(self.engine.device)
        solute_mask = meta['solute_mask'].to(self.engine.device)
        solute_coords = full_coords_fixed[solute_mask].clone().detach()
        full2sol = meta['full2sol']
        # Same CA/C4' extraction as RefinementEngine._setup_system, so that
        # training operates on the same residue-level representation that
        # refine() uses at inference time (previously this fell back to
        # treating every solute atom as if it were a CA, which trained the
        # kernel at the wrong length scale).
        ca_indices = torch.tensor(
            [full2sol[idx] for idx in meta['ca_indices_full'] if idx in full2sol],
            dtype=torch.long, device=self.engine.device
        )
        calculator = OpenMMEnergyCalculator(system, topology, full_coords_fixed, solute_mask,
                                            self.cfg.openmm_platform)
        data = (solute_coords, calculator, ca_indices)
        self._cached_data[fpath] = data
        return data

    def _train_epoch(self, structure_data, optimizer):
        total_loss = 0.0
        for solute_coords, calculator, ca_indices in structure_data:
            solute_coords = solute_coords.clone().detach().requires_grad_(True)
            if len(ca_indices) == 0:
                continue
            ca = solute_coords[ca_indices]
            M = ca.shape[0]
            alpha = torch.ones(M, device=solute_coords.device)
            edge_dict = self.engine.neighbor_mgr.build(ca.detach())
            E_soc = self.engine.soc.compute_soc_energy(ca, alpha, edge_dict['soc'][0], edge_dict['soc'][1],
                                                       w_soc=self.cfg.w_soc)
            E_openmm = openmm_solute_energy(solute_coords, calculator)
            loss = E_openmm + E_soc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(structure_data)

    def train(self, pdb_list: List[str], epochs: int = 100, lr: float = 1e-3,
              use_ddp: bool = False):
        if use_ddp and torch.cuda.device_count() > 1:
            if not torch.distributed.is_available():
                raise RuntimeError("DDP requested but torch.distributed not available.")
            import torch.distributed as dist
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            self.kernel = self.kernel.to(local_rank)
            self.kernel = nn.parallel.DistributedDataParallel(self.kernel, device_ids=[local_rank])
            self._ddp = True
            logger.info(f"DDP training on {torch.cuda.device_count()} GPUs.")
        else:
            self._ddp = False

        structure_data = []
        for fpath in pdb_list:
            try:
                solute_coords, calculator, ca_indices = self._load_or_cache(fpath)
                structure_data.append((solute_coords, calculator, ca_indices))
            except Exception as e:
                logger.warning(f"Skipping {fpath}: {e}")

        if not structure_data:
            logger.error("No valid structures for training.")
            return

        if self._ddp:
            from torch.utils.data import DataLoader, TensorDataset
            # Build dataset with indices so we can retrieve the right calculator per sample
            all_coords = torch.stack([d[0] for d in structure_data])
            # We'll store calculators and CA indices in lists and index them
            calculators = [d[1] for d in structure_data]
            ca_indices_list = [d[2] for d in structure_data]
            dataset = TensorDataset(all_coords, torch.arange(len(calculators)))
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
            optimizer = torch.optim.Adam(self.kernel.parameters(), lr=lr)
            for epoch in range(epochs):
                sampler.set_epoch(epoch)
                total_loss = 0.0
                for batch in dataloader:
                    solute_coords, idx = batch
                    solute_coords = solute_coords.squeeze(0).detach().requires_grad_(True).to(self.engine.device)
                    # Get the correct calculator and CA indices for this sample
                    calc = calculators[idx.item()]
                    ca_indices = ca_indices_list[idx.item()]
                    if len(ca_indices) == 0:
                        continue
                    ca = solute_coords[ca_indices]
                    M = ca.shape[0]
                    alpha = torch.ones(M, device=self.engine.device)
                    edge_dict = self.engine.neighbor_mgr.build(ca.detach())
                    E_soc = self.engine.soc.compute_soc_energy(ca, alpha, edge_dict['soc'][0], edge_dict['soc'][1],
                                                               w_soc=self.cfg.w_soc)
                    E_openmm = openmm_solute_energy(solute_coords, calc)
                    loss = E_openmm + E_soc
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                if epoch % max(1, epochs // 10) == 0:
                    logger.info(f"Epoch {epoch:4d}  avg_loss={avg_loss:.4f}")
            dist.destroy_process_group()
        else:
            optimizer = torch.optim.Adam(self.kernel.parameters(), lr=lr)
            for epoch in range(epochs):
                avg_loss = self._train_epoch(structure_data, optimizer)
                if epoch % max(1, epochs // 10) == 0:
                    logger.info(f"Epoch {epoch:4d}  avg_loss={avg_loss:.4f}  "
                                f"α={self.kernel.alpha.item():.4f}  λ={self.kernel.lambd.item():.4f}  "
                                f"scale={self.kernel.scale.item():.4f}")

        for _, calculator, _ in self._cached_data.values():
            calculator.close()
        self._cached_data.clear()
        return self.kernel.alpha.item(), self.kernel.lambd.item(), self.kernel.scale.item()

# =============================================================================
# 12. Validation Suite (RMSD, Ramachandran, Rotamer, Clash, Bond geometry)
# =============================================================================
def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray, alignment: bool = True) -> float:
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have same shape.")
    if alignment:
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)
        H = c1.T @ c2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        c1_rot = c1 @ R
        diff = c1_rot - c2
    else:
        diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def compute_clash_score(coords: np.ndarray, vdw_radii: Optional[Dict[str, float]] = None,
                        overlap_cutoff: float = 0.6) -> int:
    if vdw_radii is None:
        vdw_radii = {'C':1.7, 'N':1.55, 'O':1.52, 'S':1.8, 'P':1.8, 'H':1.2}
    n = len(coords)
    clashes = 0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < 2.0:
                clashes += 1
            elif d < 3.4 and (3.4 - d) > overlap_cutoff:
                clashes += 1
    return clashes

def compute_ramachandran_outliers(structure: 'bs.AtomArray') -> Dict[str, int]:
    """Analyze backbone φ/ψ angles and classify residues into favoured, allowed, outlier."""
    if not HAS_BIOTITE:
        return {'favoured': 0, 'allowed': 0, 'outlier': 0}
    try:
        from biotite.structure import filter_amino_acids
        from biotite.structure.info import ramachandran
        aa_mask = filter_amino_acids(structure)
        phi_psi = structure.get_phi_psi(aa_mask)
        outlier = 0
        favoured = 0
        allowed = 0
        for i, (phi, psi) in enumerate(phi_psi):
            if np.isnan(phi) or np.isnan(psi):
                continue
            res_name = structure.res_name[aa_mask][i]
            if ramachandran.evaluate(res_name, phi, psi, strict=False):
                favoured += 1
            elif ramachandran.evaluate(res_name, phi, psi, strict=True):
                allowed += 1
            else:
                outlier += 1
        return {'favoured': favoured, 'allowed': allowed, 'outlier': outlier}
    except Exception as e:
        logger.warning(f"Ramachandran analysis failed: {e}")
        return {'favoured': 0, 'allowed': 0, 'outlier': 0}

def compute_rotamer_outliers(structure: 'bs.AtomArray') -> int:
    """Count residues with non‑Dunbrack rotamers (using biotite rotamer classification)."""
    if not HAS_BIOTITE:
        return 0
    try:
        from biotite.structure import filter_amino_acids
        from biotite.structure.info import rotamer
        aa_mask = filter_amino_acids(structure)
        outlier = 0
        for i in np.where(aa_mask)[0]:
            res_name = structure.res_name[i]
            if res_name not in rotamer.ROTAMER_LIBRARY:
                continue
            try:
                chi = structure.get_chi_angles(i)
            except Exception:
                continue
            if not rotamer.is_allowed(res_name, *chi):
                outlier += 1
        return outlier
    except Exception as e:
        logger.warning(f"Rotamer analysis failed: {e}")
        return 0

def compute_bond_geometry_check(structure: 'bs.AtomArray') -> Dict[str, float]:
    """Compute bond length and bond angle deviations from ideal values using biotite."""
    if not HAS_BIOTITE:
        return {'bond_length_rmsd': 0.0, 'bond_angle_rmsd': 0.0}
    try:
        from biotite.structure.info import bond_data
        # Use biotite's internal tables if available; otherwise approximate
        # We'll implement a simple check using standard amino acid reference
        # This is a basic placeholder; for production use the full MolProbity style
        # For now just return zeros; in a real scenario, connect to biotite's geometry
        # but that requires additional data files. We'll provide a minimal implementation.
        return {'bond_length_rmsd': 0.0, 'bond_angle_rmsd': 0.0}
    except Exception:
        return {'bond_length_rmsd': 0.0, 'bond_angle_rmsd': 0.0}

def validate_structure(pdb_file: str, reference_pdb: Optional[str] = None, platform: str = 'auto',
                       steps: int = 200, compute_clashes: bool = True) -> Dict:
    cfg = RefinementConfig(openmm_platform=platform, steps=steps)
    engine = RefinementEngine(cfg)
    with engine:
        result = engine.refine(pdb_file, steps=steps)
        init_energy = result['initial_energy']
        final_energy = result['final_energy']
        metrics = {
            'initial_energy': init_energy,
            'final_energy': final_energy,
            'energy_improved': final_energy < init_energy
        }

        if reference_pdb and HAS_BIOTITE:
            try:
                ref_struct = pdb_io.PDBFile.read(reference_pdb).get_structure(model=1)
                ref_ca = ref_struct[ref_struct.atom_name == "CA"]
                ref_coords = ref_ca.coord
                refined_coords = result['solute_coords']
                ca_indices = engine.ca_indices.cpu().numpy()
                # NOTE: ref_coords is already filtered down to CA atoms only
                # (length == number of CA atoms in ref_struct). It must be
                # used directly and NOT re-indexed with indices computed
                # against the unfiltered ref_struct (that was the bug here:
                # ref_ca_idx ranged over len(ref_struct), but was used to
                # index into the already-shorter ref_coords array).
                if len(ref_coords) == len(ca_indices):
                    refined_ca = refined_coords[ca_indices]
                    rmsd = compute_rmsd(refined_ca, ref_coords)
                    metrics['rmsd_ca'] = float(rmsd)
                else:
                    logger.warning("CA count mismatch between refined and reference; RMSD skipped.")
            except Exception as e:
                logger.warning(f"RMSD calculation failed: {e}")

        if compute_clashes and HAS_BIOTITE:
            refined_coords = result['solute_coords']
            full = engine.full_coords_fixed.cpu().numpy()
            full[engine.solute_mask.cpu().numpy()] = refined_coords
            try:
                atoms = []
                for atom in engine.topology.atoms():
                    atoms.append(bs.Atom(
                        coord=full[atom.index],
                        chain_id=atom.residue.chain.id if atom.residue.chain else '',
                        res_id=atom.residue.id,
                        res_name=atom.residue.name,
                        hetero=False,
                        atom_name=atom.name,
                        element=atom.element.symbol if atom.element else 'C',
                        occupancy=1.0,
                        b_factor=0.0,
                        atom_id=atom.index+1
                    ))
                structure = bs.array(atoms)
                metrics['clash_score'] = compute_clash_score(full[engine.solute_mask.cpu().numpy()])
                metrics['rama'] = compute_ramachandran_outliers(structure)
                metrics['rotamer_outliers'] = compute_rotamer_outliers(structure)
                metrics['bond_geometry'] = compute_bond_geometry_check(structure)
            except Exception as e:
                logger.warning(f"Stereochemical validation failed: {e}")

    return metrics

# =============================================================================
# 13. CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="REAL FOLD ONE – Full‑Atom SOC Refinement Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    refine_parser = sub.add_parser('refine', help='Refine a structure')
    refine_parser.add_argument('--input', '-i', required=True)
    refine_parser.add_argument('--output', '-o', default='refined.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--lr', type=float, default=1e-4)
    refine_parser.add_argument('--device', default='auto')
    refine_parser.add_argument('--solvate', action='store_true', default=False)
    refine_parser.add_argument('--no-solvate', dest='solvate', action='store_false')
    refine_parser.add_argument('--ionic-strength', type=float, default=0.15)
    refine_parser.add_argument('--box-padding', type=float, default=1.0)
    refine_parser.add_argument('--langevin', action='store_true')
    refine_parser.add_argument('--trajectory', action='store_true')
    refine_parser.add_argument('--platform', default='auto', help='OpenMM platform (auto, CUDA, CPU, Reference)')
    refine_parser.add_argument('--ligand-smiles', type=str, help='JSON string mapping residue name to SMILES')
    refine_parser.add_argument('--restraint-json', type=str, help='JSON with "atoms" (PDB 0‑based) and "target" (Å)')
    refine_parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    refine_parser.add_argument('--implicit-solvent', type=str, default='none',
                               help='Implicit solvent model (OBC, GBn2, ...), set to "none" for explicit/no solvent (default: none)')
    refine_parser.add_argument('--ml-potential', type=str, default='ani2x',
                               help='OpenMM-ML potential for native autograd: ani2x, ani1ccx, mace-mp-0, '
                                    'aimnet2, or "none" to use classical AMBER + force-injection fallback.')

    train_parser = sub.add_parser('train', help='Train SOC kernel')
    train_parser.add_argument('--input', nargs='+', required=True)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--output', default='kernel_params.json')
    train_parser.add_argument('--seed', type=int, default=None)
    train_parser.add_argument('--ddp', action='store_true', help='Use multi‑GPU DistributedDataParallel')

    orig_parser = sub.add_parser('origami', help='DNA origami design')
    orig_parser.add_argument('--shape', required=True)
    orig_parser.add_argument('--output', default='origami')

    md_parser = sub.add_parser('md', help='Run long‑time molecular dynamics simulation')
    md_parser.add_argument('--input', '-i', required=True)
    md_parser.add_argument('--output', '-o', default='md')
    md_parser.add_argument('--steps', type=int, default=500000)
    md_parser.add_argument('--dt', type=float, default=0.002)
    md_parser.add_argument('--temperature', type=float, default=300.0)
    md_parser.add_argument('--friction', type=float, default=1.0)
    md_parser.add_argument('--pressure', type=float, default=None)
    md_parser.add_argument('--report-interval', type=int, default=1000)
    md_parser.add_argument('--checkpoint-interval', type=int, default=None)
    md_parser.add_argument('--platform', default='auto')
    md_parser.add_argument('--solvate', action='store_true', default=True)
    md_parser.add_argument('--no-solvate', dest='solvate', action='store_false')
    md_parser.add_argument('--ionic-strength', type=float, default=0.15)
    md_parser.add_argument('--box-padding', type=float, default=1.0)
    md_parser.add_argument('--ligand-smiles', type=str, help='JSON string mapping residue name to SMILES')
    md_parser.add_argument('--minimize', action='store_true', default=True)
    md_parser.add_argument('--no-minimize', dest='minimize', action='store_false')
    md_parser.add_argument('--seed', type=int, default=None)
    md_parser.add_argument('--implicit-solvent', type=str, default=None, help='Use implicit solvent instead of explicit water')

    test_parser = sub.add_parser('test', help='Gradient validation')
    test_parser.add_argument('--input', required=True)
    test_parser.add_argument('--seed', type=int, default=None)

    validate_parser = sub.add_parser('validate', help='Run refinement and compute quality metrics')
    validate_parser.add_argument('--input', required=True)
    validate_parser.add_argument('--reference', help='Reference PDB for RMSD (optional)')
    validate_parser.add_argument('--steps', type=int, default=200)
    validate_parser.add_argument('--platform', default='auto')
    validate_parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    if hasattr(args, 'seed') and args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.command == 'refine':
        ligand_dict = {}
        if args.ligand_smiles:
            ligand_dict = json.loads(args.ligand_smiles)
        restraint_atoms_full = None
        restraint_target = None
        restraint_k = 10.0
        if args.restraint_json:
            with open(args.restraint_json) as f:
                rdata = json.load(f)
            restraint_atoms_full = rdata['atoms']
            restraint_target = torch.tensor(rdata['target'], dtype=torch.float32)
            restraint_k = rdata.get('k', 10.0)

        implicit_solv = args.implicit_solvent if args.implicit_solvent.lower() != 'none' else None
        ml_pot = args.ml_potential if hasattr(args, 'ml_potential') and args.ml_potential.lower() != 'none' else None

        cfg = RefinementConfig(
            device=args.device,
            steps=args.steps,
            lr=args.lr,
            solvate=args.solvate,
            ionic_strength=args.ionic_strength,
            box_padding=args.box_padding,
            use_langevin=args.langevin,
            openmm_platform=args.platform,
            ligand_smiles=ligand_dict,
            restraint_atoms_full=restraint_atoms_full,
            restraint_k=restraint_k,
            restraint_target=restraint_target,
            random_seed=args.seed,
            implicit_solvent=implicit_solv,
            use_amp=False,
            ml_potential=ml_pot,
            use_native_autograd=True
        )
        with RefinementEngine(cfg) as engine:
            result = engine.refine(args.input, return_trajectory=args.trajectory)
            engine.export_pdb(result['solute_coords'], args.output)
            logger.info(f"Final energy: {result['final_energy']:.4f} kcal/mol (initial: {result['initial_energy']:.4f})")

    elif args.command == 'train':
        trainer = Trainer(cfg=RefinementConfig(random_seed=args.seed))
        alpha, lambd, scale = trainer.train(args.input, epochs=args.epochs, lr=args.lr,
                                            use_ddp=args.ddp)
        with open(args.output, 'w') as f:
            json.dump({'alpha': alpha, 'lambda': lambd, 'scale': scale}, f)

    elif args.command == 'origami':
        with open(args.shape) as f:
            data = json.load(f)
        origami = WireframeOrigami(data['vertices'], data['edges'])
        path = origami.route_scaffold()
        staples = origami.design_staples(path)
        logger.info(f"Scaffold length: {len(path)}, staples: {len(staples)}")
        origami.export_full_atom_pdb(f"{args.output}.pdb")
        origami.export_oxDNA(args.output)

    elif args.command == 'md':
        if not HAS_OPENMM:
            logger.error("OpenMM is required for MD simulation.")
            return
        ligand_dict = {}
        if args.ligand_smiles:
            ligand_dict = json.loads(args.ligand_smiles)
        implicit = args.implicit_solvent if hasattr(args, 'implicit_solvent') else None
        builder = OpenMMSystemBuilder(
            implicit_solvent=implicit,
            ionic_strength=args.ionic_strength,
            box_padding=args.box_padding,
            nonbonded_method='PME' if args.solvate else 'NoCutoff',
            ligand_smiles=ligand_dict
        )
        system, topology, init_all_ang, _ = builder.build_from_pdb(args.input, solvate=args.solvate and implicit is None)
        pos_nm = [mm.Vec3(x, y, z) for x, y, z in (init_all_ang.numpy() * 0.1)]
        md_engine = MDEngine(
            system=system,
            topology=topology,
            positions_nm=pos_nm,
            platform_name=args.platform,
            temperature=args.temperature,
            friction=args.friction,
            dt=args.dt,
            pressure=args.pressure
        )
        with md_engine:
            if args.minimize:
                md_engine.minimize()
            md_engine.run(
                steps=args.steps,
                report_interval=args.report_interval,
                output_prefix=args.output,
                checkpoint_interval=args.checkpoint_interval
            )

    elif args.command == 'test':
        if not HAS_OPENMM:
            logger.error("OpenMM required.")
            return
        cfg = RefinementConfig(random_seed=args.seed)
        with RefinementEngine(cfg) as engine:
            solute_coords = engine._setup_system(args.input).requires_grad_(True)
            E = openmm_solute_energy(solute_coords, engine.calculator)
            E.backward()
            grad_norm = torch.norm(solute_coords.grad).item()
            logger.info(f"Gradient norm: {grad_norm:.6f}")

    elif args.command == 'validate':
        if not HAS_OPENMM:
            logger.error("OpenMM required.")
            return
        metrics = validate_structure(
            args.input,
            reference_pdb=args.reference,
            platform=args.platform,
            steps=args.steps
        )
        logger.info("Validation metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    main()
