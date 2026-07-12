# =============================================================================
# MATERIALS ONE v1.3 — Crystal / Solid-State Material Discovery Engine
# (Superconducting-Qubit Candidate Screening + EDA/QEDA Pipeline Bridge)
# =============================================================================
# Author       : PAI , Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License      : MIT
# Year         : 2026
# ORCID        : 0009-0008-2374-0788
# GitHub       : https://github.com/yoonalimsuwan
#
# AI Co-Developer: Claude (Anthropic) — v1.0 production-hardening pass,
#                  v1.1 KD-tree neighbor search, v1.2 GPU radius-graph
#                  backend, v1.3 superconducting-qubit screening heads +
#                  bridge into eda_qeda_adapter_layer.py.
#
# CHANGELOG v1.2 -> v1.3
# ------------------------
# [NEW] Two additional, OPT-IN prediction heads on DFTSurrogateGNN:
#         - tc_head: predicted superconducting critical temperature (K)
#         - tls_loss_proxy_head: a dimensionless, material-specific
#           scale factor on two-level-system (TLS) defect loss, used to
#           override the *constant* 1e-6 baseline loss-tangent assumption
#           that eda_qeda_adapter_layer.py's map_superconducting_qubit()
#           previously applied uniformly to every material.
#       Both heads are OFF by default (`MaterialsConfig.
#       predict_superconductor_properties=False`) — most candidate
#       materials are not superconductor candidates at all, and turning
#       this on for every structure would imply a relevance that isn't
#       there. Enable explicitly when screening specifically for qubit
#       substrates/traces.
# [NEW] `QubitCandidateReport` — combines tc_K, tls_loss_proxy, band_gap,
#       formation_energy into one screening record, plus a
#       `dilution_fridge_viable` boolean heuristic (Tc comfortably above
#       typical dilution-refrigerator base temperature, ~10-20 mK, with a
#       configurable safety margin — NOT a substitute for a real
#       cooldown/measurement campaign).
# [NEW] `to_eda_material_parameters()` / `screen_candidate_for_qubit_device()`
#       — bridge a screened candidate into
#       `StructuralFieldToDeviceMapper.map_superconducting_qubit()`.
#       `kinetic_inductance_per_square_h` flows through MaterialParameters
#       exactly as that function already uses it (genuinely unmodified
#       upstream physics). `tls_loss_proxy` has NO corresponding field in
#       MaterialParameters as written today — map_superconducting_qubit()
#       hardcodes its 1e-6 loss-tangent baseline in the function body, not
#       as a constructor parameter — so this module does NOT silently
#       claim to inject it there. Instead it applies tls_loss_proxy as an
#       explicit, clearly-labeled post-hoc rescaling of the RETURNED
#       loss_tangent_grid/Q_grid. This is a documented workaround chosen
#       specifically to avoid modifying eda_qeda_adapter_layer.py, not a
#       claim that the upstream function was changed.
# [NEW] eda_qeda_adapter_layer import is OPTIONAL and lazy: importing
#       materials_one.py never requires the EDA file to be present;
#       only calling the qubit-pipeline bridge functions does, and they
#       raise a clear BackendUnavailableError (not an ImportError at
#       module load) if it's missing.
#
# HONESTY NOTE — WHAT THE QUBIT-SCREENING PATH DOES AND DOES NOT PROVE
# ------------------------------------------------------------------------
# 1. tc_head and tls_loss_proxy_head are UNTRAINED, exactly like the
#    formation-energy and band-gap heads always have been. There is no
#    bundled superconductor dataset (e.g. SuperCon) in this repository —
#    training data for Tc prediction is materially different from the
#    Materials Project formation-energy/band-gap data this module's
#    other heads target, and has NOT been sourced or validated here.
# 2. The mapping from `tls_loss_proxy` (a learned, dimensionless GNN
#    output) to a rescaling of map_superconducting_qubit()'s loss-tangent
#    output is a LINEAR SCALING CHOICE made in
#    `screen_candidate_for_qubit_device()` (applied AFTER the EDA layer's
#    own, unmodified computation — see that function's docstring for why),
#    not a physically derived relationship. Two-level-system loss in real
#    superconducting qubits depends on surface oxide chemistry, fabrication
#    process, and substrate interface physics that this module's bulk-
#    crystal GNN representation does not capture at all.
# 3. `dilution_fridge_viable` is a coarse Tc-margin heuristic for ranking
#    candidates against each other, not a prediction of actual measured
#    qubit coherence time, which depends on far more than Tc.
# 4. The EDA bridge correctly reproduces map_superconducting_qubit()'s
#    existing (u, sigma) -> device-grid physics unchanged; what's new is
#    only *which* MaterialParameters feed into it. If eda_qeda_adapter_
#    layer.py is not present alongside this file, the qubit-device bridge
#    functions raise BackendUnavailableError with a clear message —
#    material-only screening (Tc, loss proxy, ranking) still works without it.
#
# CHANGELOG v1.1 -> v1.2
# ------------------------
# [NEW] periodic_neighbor_search() gains a third backend, TORCH_CLUSTER,
#       using torch_cluster.radius() — the same grid-hashed, GPU-capable
#       radius-graph primitive REAL FOLD ONE's structural_gno_fold_v4.py
#       already uses to fix its analogous O(N^2) neighbor-search problem
#       at large residue counts. The pattern is intentionally identical
#       across the two clusters: query points (home-cell atoms) against a
#       larger point cloud (periodic-image replicas), grid-hashed, capped
#       at `max_num_neighbors` per query point, batched, GPU-resident
#       when CUDA is available.
# [NEW] AUTO method selection is now three-tier:
#         TORCH_CLUSTER (if torch_cluster installed) for atom counts above
#         `kdtree_to_gpu_threshold` (default 5,000) -> KDTREE (if scipy
#         installed, for small/medium counts where its lower constant
#         factor wins) -> BRUTE_FORCE (zero-dependency fallback).
#       Below the threshold, KDTREE stays the default even when
#       torch_cluster is present, because torch_cluster's CUDA kernel
#       launch overhead is not worth it for a few hundred atoms.
# [NEW] `max_num_neighbors` safety cap (default 64) on the TORCH_CLUSTER
#       path: dense/pathological structures cannot silently explode GPU
#       memory by returning unbounded neighbor counts per atom — a
#       warning is logged if any atom hits the cap, since that atom's
#       neighbor list was truncated and downstream physics for it should
#       be treated as approximate.
# [NEW] Device-aware execution: `periodic_neighbor_search(..., device=...)`
#       moves the replica search onto CUDA when requested/available and
#       moves results back to CPU, mirroring how REAL FOLD ONE and AGI ONE
#       both already parameterise device placement.
# [NEW] Regression tests: TORCH_CLUSTER vs KDTREE equivalence on a small
#       structure (exact same pairs, allowing for max_num_neighbors
#       truncation being a non-issue at that size), plus a moderate-N
#       (5,000-atom) timed smoke test exercising the actual code path
#       this backend targets. True >100k-atom validation is explicitly
#       flagged as something to run on real hardware (see honesty note
#       below) — not something a CPU-only CI-style sandbox can credibly
#       claim to have verified.
#
# HONESTY NOTE ON THE ">100k atoms" CLAIM
# ------------------------------------------
# This module now CONTAINS a backend designed for >100k-atom supercells,
# using the same algorithmic approach (grid-hashed GPU radius graph) that
# REAL FOLD ONE already relies on at comparable scale. That is a real,
# verifiable engineering claim about the algorithm's asymptotic behavior
# and about torch_cluster's published design.
#
# It is NOT the same as "this sandbox has benchmarked a 100k-atom
# supercell and confirmed it runs in acceptable time/memory." This
# environment has no GPU, no torch, and no network access to install
# torch_cluster — every test below that exercises TORCH_CLUSTER is
# necessarily skipped here and has only been verified by static code
# review, not execution. Before relying on this for a real >100k-atom
# screening run, run the self-test suite (Section 9) on real hardware
# with torch_cluster installed, and separately profile memory/time at the
# actual target atom count — replica enumeration alone (N * n_images
# points materialized before the GPU call) can itself become the memory
# bottleneck at very large N if the cutoff or cell shape forces a large
# n_images; see `_build_replicas` complexity note.
#
# CHANGELOG v1.0 -> v1.1
# ------------------------
# [NEW] periodic_neighbor_search() now dispatches to a KD-tree-based
#       implementation (scipy.spatial.cKDTree) by default. Replaces the
#       O(N^2 x images) dense pairwise-distance matrix with:
#         (a) a small, cutoff-bounded set of periodic image replicas
#             (unchanged from v1.0 — still required for correctness, not
#             the bottleneck), and
#         (b) cKDTree.query_ball_tree, which finds all pairs within the
#             cutoff in roughly O(M log M) instead of O(M^2), where
#             M = N * n_images.
#       This is the standard linked-cell-equivalent approach used by
#       ASE/pymatgen-style neighbor finders, not a novel algorithm —
#       correctness matters more than originality here.
# [NEW] `NeighborSearchMethod` enum (`AUTO`, `BRUTE_FORCE`, `KDTREE`) and a
#       `method` parameter on `periodic_neighbor_search()`. AUTO picks
#       KD-tree when scipy is available and falls back to the v1.0
#       brute-force path (still correct, just slower) when it is not —
#       no hard dependency added, graceful degradation preserved.
# [NEW] Regression test asserting BRUTE_FORCE and KDTREE return identical
#       neighbor sets (same pair count, matching sorted distances) on the
#       same structure — the optimization must not change correctness.
# [NEW] `CrystalStructure.warn_if_expensive()` threshold raised and
#       reframed: it now only warns when BRUTE_FORCE is explicitly forced
#       on a large structure, since AUTO/KDTREE no longer has the same
#       pathological scaling.
# [CHANGED] Complexity guidance in docstrings updated: KD-tree path is
#       roughly O(M log M); brute-force path (still available via
#       `method=NeighborSearchMethod.BRUTE_FORCE` for testing/parity) is
#       O(M^2), where M = N * n_images.
#
# CHANGELOG v0.1 -> v1.0
# ------------------------
# [FIX] Periodic neighbor search was restricted to the home unit cell only
#       (no periodic images). For real crystals this silently miscounted
#       neighbors near cell boundaries — a correctness bug, not a cosmetic
#       one. Replaced with a minimum-image / supercell-replica neighbor
#       search that actually respects periodic boundary conditions.
# [NEW] Exception hierarchy (MaterialsONEError and subclasses), matching
#       the convention already used in eda_qeda_adapter_layer.py
#       (QEDAAdapterError / FieldValidationError / ...).
# [NEW] Structure validation: rejects singular/degenerate lattices, NaN/Inf
#       coordinates, empty structures, and atom counts that would make the
#       O(N^2 x images) neighbor search pathologically expensive without
#       an explicit override.
# [NEW] Effective-mass lookup table (per element, rough literature values)
#       instead of a single hardcoded 0.2 default — still flagged as an
#       approximation, but no longer a silent placeholder constant.
# [NEW] A real (if minimal) supervised training loop for DFTSurrogateGNN
#       against a user-supplied dataset, with train/val split, early
#       stopping, and checkpoint saving — so the architecture is no longer
#       inference-only scaffolding.
# [NEW] Assert-based self-test suite in __main__ (PASS/FAIL per test,
#       matching the ONE Ecosystem convention used elsewhere), replacing
#       the print-only smoke test.
# [NEW] Graceful handling of isolated atoms / disconnected graphs (no
#       neighbors within cutoff) instead of relying on a shape mismatch
#       crash deep inside the GNN.
#
# HONESTY NOTE — WHAT "PRODUCTION" DOES AND DOES NOT MEAN HERE
# ----------------------------------------------------------------
# "Production-grade" in this pass means: the code is correct, validated,
# and robust to the input space it claims to handle, with real error
# handling and a real (if simple) training loop — the same bar applied to
# eda_qeda_adapter_layer.py's production-hardening pass.
#
# It does NOT mean:
#   - DFTSurrogateGNN is trained. It still is not. Training requires a
#     real dataset (Materials Project / OQMD) which is not bundled here.
#     The training loop is real; the weights it would produce are not
#     included, because no training has actually been run.
#   - The MACE-MP-0 path has been benchmarked against experimental data
#     by this codebase. It is a real, pretrained, third-party potential —
#     trustworthy in its own right within its published accuracy range —
#     but this file has not independently validated it.
#   - The WKB tunneling proxy is a substitute for NEGF-level quantum
#     transport simulation. It remains a coarse, explicitly-labelled
#     ranking heuristic.
# Running and reading this file's test suite proves the engineering is
# sound. It does not prove the physics predictions are accurate — that
# still requires training + experimental validation, which is future work.
# =============================================================================

from __future__ import annotations

import math
import logging
import itertools
import enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("materials_one")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    from mace.calculators import mace_mp  # type: ignore
    HAS_MACE = True
except ImportError:
    HAS_MACE = False
    logger.warning(
        "mace-torch not installed — MATERIALS ONE will use a placeholder "
        "pairwise potential (NOT DFT-accurate). Install with: "
        "pip install mace-torch --break-system-packages"
    )

try:
    from ase import Atoms  # type: ignore
    from ase.data import atomic_numbers  # type: ignore
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    logger.warning(
        "ase not installed — using the built-in 118-element symbol table "
        "instead of ASE's database."
    )

try:
    from scipy.spatial import cKDTree  # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning(
        "scipy not installed — periodic_neighbor_search() will fall back to "
        "the O(N^2 x images) brute-force path regardless of structure size. "
        "Install scipy for KD-tree-accelerated neighbor search on large "
        "supercells: pip install scipy --break-system-packages"
    )

try:
    from torch_cluster import radius  # type: ignore
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False
    logger.warning(
        "torch_cluster not installed — periodic_neighbor_search() cannot use "
        "the GPU-capable grid-hashed radius-graph backend, so structures "
        "above ~5,000 atoms (and especially >100k atoms) will fall back to "
        "the slower KD-tree/brute-force CPU paths. Install with: "
        "pip install torch-cluster --break-system-packages "
        "(requires a matching torch build; see torch_cluster's install docs)."
    )


# =============================================================================
# SECTION 0 — Exceptions (matches eda_qeda_adapter_layer.py convention)
# =============================================================================

class MaterialsONEError(Exception):
    """Base error for the MATERIALS ONE cluster."""


class StructureValidationError(MaterialsONEError):
    """Raised when a CrystalStructure fails geometric or compositional validation."""


class NeighborSearchError(MaterialsONEError):
    """Raised when periodic neighbor search cannot proceed safely."""


class BackendUnavailableError(MaterialsONEError):
    """Raised when a requested optional backend (mace-torch, ase) is missing."""


class TrainingDataError(MaterialsONEError):
    """Raised when a training dataset fails validation before a training run."""


# =============================================================================
# SECTION 1 — Periodic Table
# =============================================================================

_ELEMENT_SYMBOLS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
    "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga",
    "Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd",
    "Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm",
    "Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os",
    "Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa",
    "U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg",
    "Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]
ELEMENT_TO_Z: Dict[str, int] = {sym: i + 1 for i, sym in enumerate(_ELEMENT_SYMBOLS)}
Z_TO_ELEMENT: Dict[int, str] = {i + 1: sym for i, sym in enumerate(_ELEMENT_SYMBOLS)}
NUM_ELEMENTS = len(_ELEMENT_SYMBOLS)

# Rough literature-order-of-magnitude electron effective-mass ratios (m*/m_e)
# at the conduction-band minimum, for a handful of common semiconductor /
# candidate-channel elements. THESE ARE APPROXIMATE, COMPOUND-INDEPENDENT
# VALUES FOR RANKING ONLY — a real screening pipeline must pull effective
# mass from the band-structure calculation for the actual compound, not
# from this per-element table. Unlisted elements fall back to a generic
# default and are logged as such.
_EFFECTIVE_MASS_TABLE: Dict[str, float] = {
    "Si": 0.26, "Ge": 0.12, "C": 0.20,       # group IV
    "Ga": 0.063, "As": 0.063, "In": 0.026, "P": 0.082, "Sb": 0.014,  # III-V-ish
    "Zn": 0.28, "O": 0.28, "S": 0.27, "Se": 0.21, "Te": 0.11,        # II-VI-ish
    "Mo": 0.5, "W": 0.5,                      # transition-metal dichalcogenide hosts
}
_DEFAULT_EFFECTIVE_MASS = 0.2


def symbol_to_z(symbol: str) -> int:
    if HAS_ASE:
        try:
            return atomic_numbers[symbol]
        except KeyError:
            pass
    if symbol not in ELEMENT_TO_Z:
        raise StructureValidationError(f"Unknown element symbol: {symbol!r}")
    return ELEMENT_TO_Z[symbol]


def lookup_effective_mass(symbol: str) -> float:
    if symbol in _EFFECTIVE_MASS_TABLE:
        return _EFFECTIVE_MASS_TABLE[symbol]
    logger.warning(
        "No effective-mass entry for element %r — using generic default %.3f. "
        "This is NOT compound-specific; verify against a real band-structure "
        "calculation before trusting tunneling-risk rankings for this material.",
        symbol, _DEFAULT_EFFECTIVE_MASS,
    )
    return _DEFAULT_EFFECTIVE_MASS


# =============================================================================
# SECTION 2 — Crystal Structure Representation (validated)
# =============================================================================

# Above this atom count, the brute-force O(N^2 x n_images) neighbor search
# below becomes expensive enough that callers should be warned explicitly
# rather than silently eating a multi-second (or worse) stall.
_DEFAULT_MAX_ATOMS_BRUTE_FORCE = 2000


@dataclass
class CrystalStructure:
    lattice: torch.Tensor
    frac_coords: torch.Tensor
    species: List[str]
    name: str = "unnamed_crystal"

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if self.lattice.shape != (3, 3):
            raise StructureValidationError(
                f"lattice must be shape (3,3), got {tuple(self.lattice.shape)}"
            )
        if not torch.isfinite(self.lattice).all():
            raise StructureValidationError("lattice contains NaN/Inf entries")
        det = torch.det(self.lattice.float())
        if abs(float(det)) < 1e-6:
            raise StructureValidationError(
                f"lattice is singular or near-degenerate (det={float(det):.3e}); "
                "unit cell has zero or near-zero volume"
            )

        n = self.frac_coords.shape[0]
        if n == 0:
            raise StructureValidationError("CrystalStructure must contain at least one atom")
        if self.frac_coords.shape != (n, 3):
            raise StructureValidationError(
                f"frac_coords must be shape (N,3), got {tuple(self.frac_coords.shape)}"
            )
        if not torch.isfinite(self.frac_coords).all():
            raise StructureValidationError("frac_coords contains NaN/Inf entries")
        if len(self.species) != n:
            raise StructureValidationError(
                f"species length ({len(self.species)}) must match frac_coords rows ({n})"
            )
        for s in self.species:
            symbol_to_z(s)  # raises StructureValidationError if unknown

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    def cart_coords(self) -> torch.Tensor:
        return self.frac_coords @ self.lattice

    def atomic_numbers(self) -> torch.Tensor:
        return torch.tensor([symbol_to_z(s) for s in self.species], dtype=torch.long)

    def cell_lengths(self) -> Tuple[float, float, float]:
        return tuple(float(torch.norm(self.lattice[i])) for i in range(3))

    def to_ase(self) -> "Atoms":
        if not HAS_ASE:
            raise BackendUnavailableError("ASE not installed; cannot export to ase.Atoms")
        return Atoms(
            symbols=self.species,
            scaled_positions=self.frac_coords.detach().cpu().numpy(),
            cell=self.lattice.detach().cpu().numpy(),
            pbc=True,
        )

    def warn_if_expensive(self, max_atoms: int = _DEFAULT_MAX_ATOMS_BRUTE_FORCE) -> None:
        if self.n_atoms > max_atoms:
            logger.warning(
                "Structure %r has %d atoms; the brute-force periodic neighbor "
                "search in this module is O(N^2 x images) and may be slow or "
                "memory-heavy beyond ~%d atoms. Consider a cell-list / KD-tree "
                "neighbor search for large supercells.",
                self.name, self.n_atoms, max_atoms,
            )


# =============================================================================
# SECTION 3 — Periodic Neighbor Search (FIXED: real minimum-image search)
# =============================================================================

def _replica_shift_range(lattice: torch.Tensor, cutoff: float) -> Tuple[int, int, int]:
    """
    Conservative number of periodic-image shifts needed along each lattice
    vector so that no neighbor within `cutoff` is missed. Uses the
    perpendicular (inter-planar) distance for each lattice direction —
    the standard, correct way to bound replica range for a possibly
    non-orthogonal cell (as opposed to naively using cell-vector length,
    which under-counts for skewed cells).
    """
    a, b, c = lattice[0], lattice[1], lattice[2]
    volume = torch.abs(torch.dot(a, torch.cross(b, c)))
    if float(volume) < 1e-9:
        raise NeighborSearchError("Cannot compute replica range: near-zero cell volume")
    # perpendicular distance between opposite faces of the parallelepiped
    d_a = volume / torch.norm(torch.cross(b, c))
    d_b = volume / torch.norm(torch.cross(a, c))
    d_c = volume / torch.norm(torch.cross(a, b))
    n_a = max(1, math.ceil(cutoff / float(d_a)))
    n_b = max(1, math.ceil(cutoff / float(d_b)))
    n_c = max(1, math.ceil(cutoff / float(d_c)))
    return n_a, n_b, n_c


class NeighborSearchMethod(str, enum.Enum):
    AUTO = "auto"                  # tiered: torch_cluster (large N) -> kdtree -> brute_force
    BRUTE_FORCE = "brute_force"    # O(M^2), M = N * n_images — v1.0 path, kept for parity testing
    KDTREE = "kdtree"              # O(M log M), requires scipy — v1.1 path
    TORCH_CLUSTER = "torch_cluster"  # grid-hashed, GPU-capable — v1.2 path for >100k atoms


# Atom count above which AUTO prefers TORCH_CLUSTER over KDTREE (when
# torch_cluster is installed). Below this, KDTREE's lower constant factor
# and lack of CUDA kernel-launch overhead typically wins. This is a
# heuristic default, not a measured crossover point in this environment —
# tune it after profiling on real hardware if it matters for your workload.
_DEFAULT_KDTREE_TO_GPU_THRESHOLD = 5_000

# Per-atom neighbor cap for the TORCH_CLUSTER backend. Prevents a
# pathological (e.g. near-overlapping atoms, wrong units) structure from
# returning an unbounded neighbor count and exploding GPU memory. If any
# atom hits this cap, its neighbor list was truncated and a warning is
# logged — treat that atom's local physics as approximate.
_DEFAULT_MAX_NUM_NEIGHBORS = 64


def periodic_neighbor_search(
    structure: CrystalStructure,
    cutoff: float,
    self_interaction: bool = False,
    method: NeighborSearchMethod = NeighborSearchMethod.AUTO,
    device: str = "cpu",
    max_num_neighbors: int = _DEFAULT_MAX_NUM_NEIGHBORS,
    kdtree_to_gpu_threshold: int = _DEFAULT_KDTREE_TO_GPU_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Correct periodic neighbor search via explicit replica enumeration
    (minimum-image-respecting, not limited to the home cell).

    Three backends, picked automatically by AUTO unless overridden:
      - TORCH_CLUSTER: grid-hashed radius graph via torch_cluster.radius(),
        GPU-resident when `device="cuda"` and a GPU is available. This is
        the backend that makes >100k-atom supercells tractable — it's the
        same algorithmic primitive (and the same library) REAL FOLD ONE's
        structural_gno_fold_v4.py already uses to fix its own O(N^2)
        neighbor-search bottleneck at large residue counts.
      - KDTREE: scipy.spatial.cKDTree, O(M log M), CPU-only. Lower
        overhead than TORCH_CLUSTER for small/medium structures.
      - BRUTE_FORCE: dense O(M^2) pairwise distance matrix. Zero
        dependencies; kept as the correctness reference the other two
        backends are regression-tested against.
    where M = N * n_images (N = atom count, n_images = periodic replicas
    needed for the given cutoff and cell shape).

    Returns:
        edge_index : (2, E) long tensor of (src, dst) HOME-CELL atom indices, on CPU
        edge_dist  : (E,) distances in Angstrom, on CPU
        edge_vec   : (E, 3) displacement vectors (Angstrom), on CPU
    """
    if cutoff <= 0:
        raise NeighborSearchError(f"cutoff must be positive, got {cutoff}")

    resolved_method = method
    if resolved_method == NeighborSearchMethod.AUTO:
        if HAS_TORCH_CLUSTER and structure.n_atoms >= kdtree_to_gpu_threshold:
            resolved_method = NeighborSearchMethod.TORCH_CLUSTER
        elif HAS_SCIPY:
            resolved_method = NeighborSearchMethod.KDTREE
        else:
            resolved_method = NeighborSearchMethod.BRUTE_FORCE

    if resolved_method == NeighborSearchMethod.TORCH_CLUSTER and not HAS_TORCH_CLUSTER:
        raise BackendUnavailableError(
            "method=TORCH_CLUSTER requested but torch_cluster is not installed. "
            "Use method=KDTREE/BRUTE_FORCE or install torch_cluster."
        )
    if resolved_method == NeighborSearchMethod.KDTREE and not HAS_SCIPY:
        raise BackendUnavailableError(
            "method=KDTREE requested but scipy is not installed. "
            "Use method=BRUTE_FORCE/TORCH_CLUSTER or install scipy."
        )

    if resolved_method == NeighborSearchMethod.TORCH_CLUSTER:
        return _periodic_neighbor_search_torch_cluster(
            structure, cutoff, self_interaction, device=device, max_num_neighbors=max_num_neighbors
        )
    if resolved_method == NeighborSearchMethod.BRUTE_FORCE:
        structure.warn_if_expensive()
        return _periodic_neighbor_search_brute_force(structure, cutoff, self_interaction)
    return _periodic_neighbor_search_kdtree(structure, cutoff, self_interaction)


def _build_replicas(
    structure: CrystalStructure, cutoff: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared replica-construction logic for both backends. Returns:
        replica_coords : (M, 3) Cartesian coords of every atom in every
                          periodic image within the cutoff-bounded shift range
        replica_home_idx : (M,) which home-cell atom index each replica came from
        replica_is_home : (M,) bool, True for the unshifted (ia=ib=ic=0) image
    """
    coords = structure.cart_coords()
    n = coords.shape[0]
    n_a, n_b, n_c = _replica_shift_range(structure.lattice, cutoff)

    shifts = [
        (ia, ib, ic)
        for ia, ib, ic in itertools.product(
            range(-n_a, n_a + 1), range(-n_b, n_b + 1), range(-n_c, n_c + 1)
        )
    ]
    shift_tensor = torch.tensor(shifts, dtype=coords.dtype)            # (S, 3)
    shift_cart = shift_tensor @ structure.lattice                      # (S, 3)

    n_shifts = shift_cart.shape[0]
    replica_coords = (coords.unsqueeze(0) + shift_cart.unsqueeze(1)).reshape(n_shifts * n, 3)
    replica_home_idx = torch.arange(n).repeat(n_shifts)
    is_home_shift = torch.all(shift_tensor == 0, dim=1)                # (S,)
    replica_is_home = is_home_shift.repeat_interleave(n)
    return replica_coords, replica_home_idx, replica_is_home


def _periodic_neighbor_search_kdtree(
    structure: CrystalStructure, cutoff: float, self_interaction: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = structure.cart_coords()
    n = coords.shape[0]
    replica_coords, replica_home_idx, replica_is_home = _build_replicas(structure, cutoff)

    # Query tree: home-cell atoms only. Search tree: all replicas. This
    # avoids building a tree over the (often much larger) full replica
    # set on both sides — query_ball_tree still needs two trees, but the
    # query side stays at N points instead of N * n_images.
    home_coords_np = coords.detach().cpu().numpy()
    replica_coords_np = replica_coords.detach().cpu().numpy()

    query_tree = cKDTree(home_coords_np)
    search_tree = cKDTree(replica_coords_np)
    # neighbors_per_query[i] = list of replica indices within cutoff of home atom i
    neighbors_per_query = query_tree.query_ball_tree(search_tree, r=cutoff)

    src_chunks, dst_chunks, dist_chunks, vec_chunks = [], [], [], []
    for home_i, replica_js in enumerate(neighbors_per_query):
        if not replica_js:
            continue
        replica_js_t = torch.tensor(replica_js, dtype=torch.long)
        dst_home = replica_home_idx[replica_js_t]                      # (k,) home-cell index of each match
        is_home_img = replica_is_home[replica_js_t]
        diff = coords[home_i].unsqueeze(0) - replica_coords[replica_js_t]  # (k, 3)
        dist = torch.norm(diff, dim=-1)
        keep = torch.ones(dist.shape[0], dtype=torch.bool)
        if not self_interaction:
            # exclude the atom's own home-cell self-pair (i == dst and same image == home)
            keep &= ~((dst_home == home_i) & is_home_img)
        if not keep.any():
            continue
        src_chunks.append(torch.full((int(keep.sum()),), home_i, dtype=torch.long))
        dst_chunks.append(dst_home[keep])
        dist_chunks.append(dist[keep])
        vec_chunks.append(diff[keep])

    if not src_chunks:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=coords.dtype),
            torch.zeros((0, 3), dtype=coords.dtype),
        )
    edge_index = torch.stack([torch.cat(src_chunks), torch.cat(dst_chunks)], dim=0)
    edge_dist = torch.cat(dist_chunks)
    edge_vec = torch.cat(vec_chunks, dim=0)
    return edge_index, edge_dist, edge_vec


def _periodic_neighbor_search_torch_cluster(
    structure: CrystalStructure,
    cutoff: float,
    self_interaction: bool,
    device: str = "cpu",
    max_num_neighbors: int = _DEFAULT_MAX_NUM_NEIGHBORS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Grid-hashed radius-graph backend via torch_cluster.radius(), GPU-
    resident when `device="cuda"`. Same algorithmic family torch_cluster's
    own consumers (e.g. SchNet/DimeNet-style models) and REAL FOLD ONE's
    structural_gno_fold_v4.py use to keep neighbor search sub-quadratic at
    large N: the search cloud (periodic replicas) is grid-hashed once, and
    each query point only scans its own and adjacent grid cells instead of
    every other point.

    The replica construction (`_build_replicas`) is unchanged in
    *algorithm* from the KD-tree path — still O(N * n_images) points
    materialized up front, which is correctness-necessary, not the part
    being optimized here. What changes is how those M points get searched:
    grid-hashed radius query instead of either a KD-tree or a dense matrix.
    """
    coords = structure.cart_coords()
    n = coords.shape[0]
    replica_coords, replica_home_idx, replica_is_home = _build_replicas(structure, cutoff)

    coords_dev = coords.to(device)
    replica_coords_dev = replica_coords.to(device)

    # torch_cluster.radius(x, y, r, ...): for each point in y (query),
    # finds all points in x (search cloud) within radius r. row indexes
    # y (home-cell queries), col indexes x (the replica cloud).
    row, col = radius(
        replica_coords_dev, coords_dev, r=cutoff, max_num_neighbors=max_num_neighbors
    )
    row = row.detach().cpu()
    col = col.detach().cpu()

    if row.numel() == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=coords.dtype),
            torch.zeros((0, 3), dtype=coords.dtype),
        )

    # Truncation guard: if any home atom hit the max_num_neighbors cap,
    # its neighbor list is incomplete — warn loudly rather than silently
    # returning partial physics for that atom.
    counts = torch.bincount(row, minlength=n)
    n_truncated = int((counts >= max_num_neighbors).sum())
    if n_truncated > 0:
        logger.warning(
            "TORCH_CLUSTER neighbor search hit max_num_neighbors=%d for %d/%d "
            "atom(s) in structure %r — their neighbor lists are truncated. "
            "Increase max_num_neighbors if this structure is unusually dense.",
            max_num_neighbors, n_truncated, n, structure.name,
        )

    dst_home = replica_home_idx[col]
    is_home_img = replica_is_home[col]
    diff = coords[row] - replica_coords[col]
    dist = torch.norm(diff, dim=-1)

    keep = torch.ones(dist.shape[0], dtype=torch.bool)
    if not self_interaction:
        keep &= ~((dst_home == row) & is_home_img)
    if not keep.any():
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=coords.dtype),
            torch.zeros((0, 3), dtype=coords.dtype),
        )

    edge_index = torch.stack([row[keep], dst_home[keep]], dim=0)
    edge_dist = dist[keep]
    edge_vec = diff[keep]
    return edge_index, edge_dist, edge_vec


def _periodic_neighbor_search_brute_force(
    structure: CrystalStructure,
    cutoff: float,
    self_interaction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """v1.0 dense pairwise-distance-matrix implementation. O(N^2 x images).
    Kept as a correctness reference for the KD-tree path and as a
    zero-dependency fallback."""
    coords = structure.cart_coords()
    n = coords.shape[0]
    n_a, n_b, n_c = _replica_shift_range(structure.lattice, cutoff)

    shifts = [
        (ia, ib, ic)
        for ia, ib, ic in itertools.product(
            range(-n_a, n_a + 1), range(-n_b, n_b + 1), range(-n_c, n_c + 1)
        )
    ]
    shift_tensor = torch.tensor(shifts, dtype=coords.dtype)
    shift_cart = shift_tensor @ structure.lattice

    src_list, dst_list, dist_list, vec_list = [], [], [], []
    for s_idx in range(shift_cart.shape[0]):
        offset = shift_cart[s_idx]
        is_home_cell = bool(torch.all(shift_tensor[s_idx] == 0))
        diff = coords.unsqueeze(1) - (coords.unsqueeze(0) + offset)
        dist = torch.norm(diff, dim=-1)
        mask = dist < cutoff
        if not self_interaction and is_home_cell:
            mask = mask & (~torch.eye(n, dtype=torch.bool))
        src, dst = mask.nonzero(as_tuple=True)
        if src.numel() == 0:
            continue
        src_list.append(src)
        dst_list.append(dst)
        dist_list.append(dist[src, dst])
        vec_list.append(diff[src, dst])

    if not src_list:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=coords.dtype),
            torch.zeros((0, 3), dtype=coords.dtype),
        )
    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
    edge_dist = torch.cat(dist_list)
    edge_vec = torch.cat(vec_list, dim=0)
    return edge_index, edge_dist, edge_vec


# =============================================================================
# SECTION 4 — Interatomic Potential Backend
# =============================================================================

class InteratomicPotential:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.is_real_physics = HAS_MACE and HAS_ASE
        if self.is_real_physics:
            self._calc = mace_mp(model="medium", device=device, default_dtype="float32")
            logger.info("MACE-MP-0 foundation potential loaded — real ML-DFT physics active.")
        else:
            self._calc = None
            logger.warning(
                "Using PLACEHOLDER Lennard-Jones-style potential. Energies/forces "
                "from this path are NOT physically meaningful beyond toy testing."
            )

    def energy_forces(self, structure: CrystalStructure) -> Tuple[float, torch.Tensor]:
        if self.is_real_physics:
            atoms = structure.to_ase()
            atoms.calc = self._calc
            energy = float(atoms.get_potential_energy())
            forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
            if not math.isfinite(energy):
                raise MaterialsONEError(f"MACE-MP-0 returned non-finite energy: {energy}")
            return energy, forces
        return self._placeholder_energy_forces(structure)

    @staticmethod
    def _placeholder_energy_forces(structure: CrystalStructure) -> Tuple[float, torch.Tensor]:
        edge_index, edge_dist, _ = periodic_neighbor_search(structure, cutoff=8.0)
        coords = structure.cart_coords().clone().requires_grad_(True)
        n = coords.shape[0]
        if edge_index.shape[1] == 0:
            return 0.0, torch.zeros((n, 3))
        # Recompute distances from `coords` (not the detached search output)
        # so autograd can flow through them.
        src, dst = edge_index[0], edge_index[1]
        diff = coords[src] - coords[dst]
        dist = torch.norm(diff + 1e-9, dim=-1)
        sigma, epsilon = 2.5, 0.05
        r6 = (sigma / (dist + 1e-9)) ** 6
        r12 = r6 ** 2
        lj = 4 * epsilon * (r12 - r6)
        energy = lj.sum() / 2.0  # each pair counted from both directions
        forces = -torch.autograd.grad(energy, coords, create_graph=False)[0]
        return float(energy.detach()), forces.detach()


# =============================================================================
# SECTION 5 — DFT-Surrogate GNN
# =============================================================================

@dataclass
class MaterialsConfig:
    elem_embed_dim: int = 32
    hidden_dim: int = 64
    n_message_passes: int = 3
    cutoff_radius: float = 5.0
    latent_dim: int = 128
    # [v1.3 NEW] opt-in: most candidate materials are not superconductor
    # candidates, so these heads are off by default. Enable explicitly
    # for qubit-substrate/trace screening.
    predict_superconductor_properties: bool = False


class _MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            # No edges (isolated atom / cutoff too small for this cell):
            # skip message passing this layer rather than crashing on an
            # empty index_add_.
            return node_feat
        src, dst = edge_index[0], edge_index[1]
        edge_w = self.edge_mlp(edge_dist.unsqueeze(-1))
        messages = node_feat[src] * edge_w
        agg = torch.zeros_like(node_feat)
        agg.index_add_(0, dst, messages)
        return node_feat + self.update_mlp(torch.cat([node_feat, agg], dim=-1))


class DFTSurrogateGNN(nn.Module):
    """
    Still an architecture that ships UNTRAINED. v1.0 adds a real training
    loop (Section 7) so it is no longer inference-only scaffolding, but no
    pretrained checkpoint is bundled — training on real data is a
    separate, required step before any prediction here should be trusted.
    """

    def __init__(self, cfg: MaterialsConfig):
        super().__init__()
        self.cfg = cfg
        self.elem_embed = nn.Embedding(NUM_ELEMENTS + 1, cfg.elem_embed_dim)
        self.input_proj = nn.Linear(cfg.elem_embed_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList(
            [_MessagePassingLayer(cfg.hidden_dim) for _ in range(cfg.n_message_passes)]
        )
        self.readout = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.formation_energy_head = nn.Linear(cfg.latent_dim, 1)
        self.band_gap_head = nn.Sequential(nn.Linear(cfg.latent_dim, 1), nn.Softplus())

        # [v1.3 NEW] opt-in superconductor-screening heads. Constructed
        # only when requested so a plain formation-energy/band-gap model
        # doesn't carry unused parameters (and so it's visually obvious in
        # a checkpoint/state_dict whether a given model was ever intended
        # for qubit-material screening).
        if cfg.predict_superconductor_properties:
            self.tc_head = nn.Sequential(nn.Linear(cfg.latent_dim, 1), nn.Softplus())              # Kelvin, >= 0
            self.tls_loss_proxy_head = nn.Sequential(nn.Linear(cfg.latent_dim, 1), nn.Softplus())   # dimensionless, >= 0
        else:
            self.tc_head = None
            self.tls_loss_proxy_head = None

    def forward(self, structure: CrystalStructure) -> Dict[str, torch.Tensor]:
        z = structure.atomic_numbers()
        node_feat = self.input_proj(self.elem_embed(z))
        edge_index, edge_dist, _ = periodic_neighbor_search(structure, self.cfg.cutoff_radius)
        for layer in self.layers:
            node_feat = layer(node_feat, edge_index, edge_dist)
        latent = self.readout(node_feat).mean(dim=0, keepdim=True)
        formation_energy = self.formation_energy_head(latent).squeeze(-1)
        band_gap = self.band_gap_head(latent).squeeze(-1)
        out: Dict[str, torch.Tensor] = {
            "latent": latent,
            "formation_energy_ev_per_atom": formation_energy,
            "band_gap_ev": band_gap,
        }
        if self.cfg.predict_superconductor_properties:
            out["tc_kelvin"] = self.tc_head(latent).squeeze(-1)
            out["tls_loss_proxy"] = self.tls_loss_proxy_head(latent).squeeze(-1)
        return out


# =============================================================================
# SECTION 6 — Quantum-Tunneling Risk Proxy
# =============================================================================

_HBAR_EVS = 6.582119569e-16
_ELECTRON_MASS_KG = 9.1093837015e-31
_EV_TO_JOULE = 1.602176634e-19
_ANGSTROM_TO_M = 1e-10


def wkb_tunneling_probability(
    barrier_height_ev: float,
    barrier_width_angstrom: float,
    effective_mass_ratio: float = _DEFAULT_EFFECTIVE_MASS,
) -> float:
    if barrier_height_ev < 0:
        raise MaterialsONEError(f"barrier_height_ev must be >= 0, got {barrier_height_ev}")
    if barrier_width_angstrom <= 0:
        raise MaterialsONEError(f"barrier_width_angstrom must be > 0, got {barrier_width_angstrom}")
    if barrier_height_ev == 0:
        return 1.0
    m_star = effective_mass_ratio * _ELECTRON_MASS_KG
    V_joule = barrier_height_ev * _EV_TO_JOULE
    hbar_js = _HBAR_EVS * _EV_TO_JOULE
    kappa = math.sqrt(2 * m_star * V_joule) / hbar_js
    L_m = barrier_width_angstrom * _ANGSTROM_TO_M
    return float(math.exp(-2.0 * kappa * L_m))


@dataclass
class TunnelingRiskReport:
    material_name: str
    band_gap_ev: float
    gate_length_angstrom: float
    effective_mass_ratio: float
    transmission_probability: float
    risk_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material": self.material_name,
            "band_gap_ev": round(self.band_gap_ev, 4),
            "gate_length_angstrom": self.gate_length_angstrom,
            "effective_mass_ratio": self.effective_mass_ratio,
            "transmission_probability": self.transmission_probability,
            "risk_level": self.risk_level,
        }


def assess_tunneling_risk(
    material_name: str,
    band_gap_ev: float,
    gate_length_angstrom: float,
    dominant_element: Optional[str] = None,
    effective_mass_ratio: Optional[float] = None,
) -> TunnelingRiskReport:
    m_star = (
        effective_mass_ratio if effective_mass_ratio is not None
        else (lookup_effective_mass(dominant_element) if dominant_element else _DEFAULT_EFFECTIVE_MASS)
    )
    T = wkb_tunneling_probability(band_gap_ev, gate_length_angstrom, m_star)
    risk = "low" if T < 1e-6 else ("moderate" if T < 1e-2 else "high")
    return TunnelingRiskReport(
        material_name=material_name,
        band_gap_ev=band_gap_ev,
        gate_length_angstrom=gate_length_angstrom,
        effective_mass_ratio=m_star,
        transmission_probability=T,
        risk_level=risk,
    )


# =============================================================================
# SECTION 6B — Superconducting-Qubit Candidate Screening (NEW in v1.3)
# =============================================================================
# Bridges DFTSurrogateGNN's opt-in Tc/TLS-loss heads to
# eda_qeda_adapter_layer.py's MaterialParameters / StructuralFieldToDevice
# Mapper.map_superconducting_qubit(), so a screened candidate material can
# flow into the same device-parameter-grid pipeline every other EDA
# platform already uses. See the v1.3 honesty note at the top of this
# file before trusting any number this section produces.
# =============================================================================

# Typical dilution-refrigerator base temperature range, millikelvin.
# Real systems vary (commercial dilution fridges commonly reach 10-20 mK);
# this is a coarse reference band for the viability heuristic below, not a
# claim about any specific fridge.
_TYPICAL_DILUTION_FRIDGE_BASE_TEMP_K = 0.015  # 15 mK, mid-range reference


@dataclass
class QubitCandidateReport:
    material_name: str
    tc_kelvin: float
    tls_loss_proxy: float
    band_gap_ev: float
    formation_energy_ev_per_atom: float
    dilution_fridge_viable: bool
    tc_margin_ratio: float  # tc_kelvin / fridge_base_temp_k — higher is more comfortably superconducting

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material": self.material_name,
            "tc_kelvin": round(self.tc_kelvin, 5),
            "tls_loss_proxy": round(self.tls_loss_proxy, 5),
            "band_gap_ev": round(self.band_gap_ev, 4),
            "formation_energy_ev_per_atom": round(self.formation_energy_ev_per_atom, 4),
            "dilution_fridge_viable": self.dilution_fridge_viable,
            "tc_margin_ratio": round(self.tc_margin_ratio, 2),
        }


def screen_qubit_candidate(
    structure: CrystalStructure,
    model: DFTSurrogateGNN,
    fridge_base_temp_k: float = _TYPICAL_DILUTION_FRIDGE_BASE_TEMP_K,
    min_margin_ratio: float = 10.0,
) -> QubitCandidateReport:
    """
    Run the GNN's (untrained, unless you've trained it yourself —
    see Section 7) superconductor heads on a candidate structure and
    produce a screening record.

    `min_margin_ratio`: require Tc to exceed the fridge base temperature
    by at least this factor before calling a candidate "viable". 10x is a
    conservative, configurable rule-of-thumb default (real qubit design
    typically wants Tc well above the operating point so the
    superconducting gap stays robust against thermal quasiparticles) —
    not a value derived from this module's own physics.
    """
    if not model.cfg.predict_superconductor_properties:
        raise MaterialsONEError(
            "model was constructed with predict_superconductor_properties=False; "
            "rebuild with MaterialsConfig(predict_superconductor_properties=True) "
            "before calling screen_qubit_candidate()."
        )
    out = model(structure)
    tc = float(out["tc_kelvin"].item())
    tls = float(out["tls_loss_proxy"].item())
    margin = tc / fridge_base_temp_k if fridge_base_temp_k > 0 else float("inf")
    return QubitCandidateReport(
        material_name=structure.name,
        tc_kelvin=tc,
        tls_loss_proxy=tls,
        band_gap_ev=float(out["band_gap_ev"].item()),
        formation_energy_ev_per_atom=float(out["formation_energy_ev_per_atom"].item()),
        dilution_fridge_viable=(margin >= min_margin_ratio),
        tc_margin_ratio=margin,
    )


def _import_eda_qeda_adapter_layer():
    """
    Lazy, optional import of eda_qeda_adapter_layer.py. Importing
    materials_one.py itself never requires this file to be present —
    only the qubit-device bridge functions below do, and they fail with a
    clear, actionable error rather than an import-time crash.
    """
    try:
        import eda_qeda_adapter_layer as eda  # type: ignore
        return eda
    except ImportError as e:
        raise BackendUnavailableError(
            "eda_qeda_adapter_layer.py is not importable (expected on the "
            "same PYTHONPATH as materials_one.py). The qubit-device bridge "
            "(to_eda_material_parameters / screen_candidate_for_qubit_device) "
            "requires it; material-only screening (screen_qubit_candidate) "
            f"does not. Original error: {e}"
        ) from e


# Baseline loss-tangent constant currently hardcoded inside
# eda_qeda_adapter_layer.py's map_superconducting_qubit() (1e-6, scaled
# only by the spatial sigma field). This module's tls_loss_proxy head, once
# trained, is meant to replace that one-size-fits-all constant with a
# material-specific value — see the linear-scaling honesty note above.
_EDA_BASELINE_LOSS_TANGENT = 1e-6


def to_eda_material_parameters(
    base_kinetic_inductance_per_square_h: Optional[float] = None,
):
    """
    Builds an eda_qeda_adapter_layer.MaterialParameters instance for a
    qubit-candidate run.

    IMPORTANT SCOPE LIMITATION (please read): MaterialParameters does
    carry `kinetic_inductance_per_square_h`, which this function does
    pass through when supplied. It does NOT carry a loss-tangent baseline
    field — eda_qeda_adapter_layer.py's map_superconducting_qubit()
    currently hardcodes that constant (1e-6) directly in the function
    body rather than reading it from MaterialParameters. This module does
    NOT modify eda_qeda_adapter_layer.py to add that field (out of scope
    for this change), so a candidate's predicted `tls_loss_proxy` CANNOT
    be injected through MaterialParameters today. See
    `screen_candidate_for_qubit_device()` below for how this module
    instead applies tls_loss_proxy as an explicit, clearly-labeled
    post-hoc rescaling of the returned loss_tangent_grid/Q_grid — a
    documented workaround, not a silent one.

    Raises BackendUnavailableError if eda_qeda_adapter_layer.py is not
    importable.
    """
    eda = _import_eda_qeda_adapter_layer()
    kwargs: Dict[str, Any] = {}
    if base_kinetic_inductance_per_square_h is not None:
        kwargs["kinetic_inductance_per_square_h"] = base_kinetic_inductance_per_square_h
    return eda.MaterialParameters(**kwargs)


def screen_candidate_for_qubit_device(
    structure: CrystalStructure,
    model: DFTSurrogateGNN,
    u_field: torch.Tensor,
    sigma_field: torch.Tensor,
    fridge_base_temp_k: float = _TYPICAL_DILUTION_FRIDGE_BASE_TEMP_K,
    min_margin_ratio: float = 10.0,
    base_kinetic_inductance_per_square_h: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full bridge: CrystalStructure -> material screening (Tc, TLS-loss
    proxy) -> eda_qeda_adapter_layer.MaterialParameters -> device-
    parameter grids via StructuralFieldToDeviceMapper.map_superconducting_
    qubit(u_field, sigma_field).

    `u_field`/`sigma_field` are NOT produced by this module — they are
    the same structural fields every other EDA adapter call site expects,
    coming from upstream Structural Cahn-Hilliard / GNO-fold simulation.

    `kinetic_inductance_per_square_h` flows through MaterialParameters
    exactly as map_superconducting_qubit() already uses it — unmodified
    upstream physics. `tls_loss_proxy`, by contrast, has no field to flow
    through (see to_eda_material_parameters' docstring), so this function
    applies it explicitly, AFTER the EDA layer's own computation, as a
    documented multiplicative rescaling of loss_tangent_grid (and the
    inverse rescaling of Q_grid, since Q = 1/loss_tangent in that layer).
    This keeps the EDA file completely unmodified while still letting a
    screened candidate's predicted loss-tangent proxy actually affect the
    returned device parameters, rather than being computed and silently
    discarded.

    Raises BackendUnavailableError if eda_qeda_adapter_layer.py is not
    importable. Raises MaterialsONEError if `model` was not built with
    predict_superconductor_properties=True.
    """
    eda = _import_eda_qeda_adapter_layer()
    material_report = screen_qubit_candidate(
        structure, model, fridge_base_temp_k=fridge_base_temp_k, min_margin_ratio=min_margin_ratio
    )
    eda_params = to_eda_material_parameters(base_kinetic_inductance_per_square_h)
    mapper = eda.StructuralFieldToDeviceMapper(material=eda_params)
    device_grids = dict(mapper.map_superconducting_qubit(u_field, sigma_field))

    # Documented post-hoc rescaling — see docstring above for why this
    # happens here instead of inside MaterialParameters. tls_loss_proxy is
    # dimensionless and centered around 1.0 only by construction of the
    # (untrained) Softplus head's typical output range, NOT by any
    # physical normalization — re-check this once the head is actually
    # trained, since a trained model's output distribution may differ.
    loss_scale = material_report.tls_loss_proxy
    if "loss_tangent_grid" in device_grids:
        device_grids["loss_tangent_grid"] = device_grids["loss_tangent_grid"] * loss_scale
        device_grids["Q_grid"] = device_grids["Q_grid"] / max(loss_scale, 1e-12)

    if not material_report.dilution_fridge_viable:
        logger.warning(
            "Candidate %r has tc_margin_ratio=%.2f (< min_margin_ratio=%.1f) — "
            "predicted Tc is not comfortably above the dilution-fridge base "
            "temperature. Proceeding with device-grid generation anyway "
            "(caller's choice), but treat this candidate as high-risk.",
            structure.name, material_report.tc_margin_ratio, min_margin_ratio,
        )

    return {
        "material_screening": material_report.to_dict(),
        "device_grids": device_grids,
        "loss_tangent_rescaled_by_tls_loss_proxy": loss_scale,
    }


# =============================================================================
# SECTION 7 — Training Loop (NEW in v1.0)
# =============================================================================

@dataclass
class MaterialsDatasetEntry:
    structure: CrystalStructure
    formation_energy_ev_per_atom: float
    band_gap_ev: float


def validate_dataset(dataset: Sequence[MaterialsDatasetEntry]) -> None:
    if len(dataset) == 0:
        raise TrainingDataError("Training dataset is empty")
    for i, entry in enumerate(dataset):
        if not math.isfinite(entry.formation_energy_ev_per_atom):
            raise TrainingDataError(f"dataset[{i}]: non-finite formation_energy_ev_per_atom")
        if not math.isfinite(entry.band_gap_ev) or entry.band_gap_ev < 0:
            raise TrainingDataError(f"dataset[{i}]: invalid band_gap_ev={entry.band_gap_ev}")


def train_dft_surrogate(
    model: DFTSurrogateGNN,
    train_set: Sequence[MaterialsDatasetEntry],
    val_set: Sequence[MaterialsDatasetEntry],
    n_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Minimal real supervised training loop: MSE on formation energy + band
    gap, Adam optimizer, early stopping on validation loss, optional
    checkpointing. This is intentionally simple (no batching across
    structures of different sizes, no LR schedule) — adequate to actually
    train on a small/medium dataset, not a claim of matching a
    production materials-informatics training pipeline (e.g. M3GNet,
    MEGNet's full training recipe).
    """
    validate_dataset(train_set)
    validate_dataset(val_set)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for entry in train_set:
            optimizer.zero_grad()
            out = model(entry.structure)
            loss = (
                F.mse_loss(out["formation_energy_ev_per_atom"],
                           torch.tensor([entry.formation_energy_ev_per_atom]))
                + F.mse_loss(out["band_gap_ev"], torch.tensor([entry.band_gap_ev]))
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for entry in val_set:
                out = model(entry.structure)
                loss = (
                    F.mse_loss(out["formation_energy_ev_per_atom"],
                               torch.tensor([entry.formation_energy_ev_per_atom]))
                    + F.mse_loss(out["band_gap_ev"], torch.tensor([entry.band_gap_ev]))
                )
                val_losses.append(float(loss))

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info("epoch %d/%d  train_loss=%.4f  val_loss=%.4f", epoch + 1, n_epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (no val improvement for %d epochs)", epoch + 1, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path and best_state is not None:
        torch.save(best_state, checkpoint_path)
        logger.info("Saved best checkpoint to %s", checkpoint_path)

    return history


# =============================================================================
# SECTION 8 — EcosystemOrchestrator Adapter
# =============================================================================

class MaterialsONEAdapter(nn.Module):
    def __init__(self, agi_latent_dim: int, cfg: Optional[MaterialsConfig] = None, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg or MaterialsConfig(latent_dim=agi_latent_dim)
        self.model = DFTSurrogateGNN(self.cfg)
        self.potential = InteratomicPotential(device=device)
        self._last_quality: float = 0.5
        self._projection = (
            nn.Linear(self.cfg.latent_dim, agi_latent_dim)
            if self.cfg.latent_dim != agi_latent_dim else nn.Identity()
        )

    def encode(self, structure: CrystalStructure) -> torch.Tensor:
        out = self.model(structure)
        latent = self._projection(out["latent"])
        fe = float(out["formation_energy_ev_per_atom"].detach())
        self._last_quality = float(max(0.0, min(1.0, math.exp(-abs(fe)))))
        return latent

    def get_quality_score(self) -> float:
        return self._last_quality


# =============================================================================
# SECTION 9 — Self-Test Suite (PASS/FAIL, ONE Ecosystem convention)
# =============================================================================

def _make_toy_structure(name: str = "toy_SiC") -> CrystalStructure:
    lattice = torch.eye(3) * 5.43
    frac_coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
        [0.99, 0.99, 0.99],   # deliberately near the cell boundary —
                                # exercises the periodic-image fix directly.
    ])
    species = ["Si", "Si", "C", "C", "Si"]
    return CrystalStructure(lattice=lattice, frac_coords=frac_coords, species=species, name=name)


def run_self_tests() -> bool:
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            results.append((name, True, ""))
        except Exception as e:  # noqa: BLE001 — self-test harness intentionally broad
            results.append((name, False, f"{type(e).__name__}: {e}"))

    def t_structure_validation_rejects_singular_lattice():
        bad_lattice = torch.zeros(3, 3)
        try:
            CrystalStructure(lattice=bad_lattice, frac_coords=torch.zeros(1, 3), species=["Si"])
            raise AssertionError("expected StructureValidationError, none raised")
        except StructureValidationError:
            pass

    def t_structure_validation_rejects_unknown_element():
        try:
            CrystalStructure(
                lattice=torch.eye(3) * 5.0,
                frac_coords=torch.zeros(1, 3),
                species=["Xx"],
            )
            raise AssertionError("expected StructureValidationError, none raised")
        except StructureValidationError:
            pass

    def t_periodic_neighbor_search_finds_boundary_neighbors():
        # Two atoms placed on opposite sides of a periodic boundary should
        # be found as neighbors when the periodic image distance is small,
        # even though their in-cell (non-periodic) distance is large. This
        # is the direct regression test for the v0.1 -> v1.0 bug fix.
        lattice = torch.eye(3) * 10.0
        frac_coords = torch.tensor([[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]])  # 0.2 A apart across boundary
        structure = CrystalStructure(lattice=lattice, frac_coords=frac_coords, species=["Si", "Si"])
        edge_index, edge_dist, _ = periodic_neighbor_search(structure, cutoff=1.0)
        assert edge_index.shape[1] > 0, "expected at least one periodic-image neighbor pair"
        assert float(edge_dist.min()) < 0.5, f"expected near-0.2A neighbor, got min dist {float(edge_dist.min())}"

    def t_neighbor_search_empty_graph_handled():
        lattice = torch.eye(3) * 100.0  # huge cell, single atom -> no neighbors
        structure = CrystalStructure(lattice=lattice, frac_coords=torch.zeros(1, 3), species=["Si"])
        edge_index, edge_dist, edge_vec = periodic_neighbor_search(structure, cutoff=2.0)
        assert edge_index.shape == (2, 0)
        assert edge_dist.shape == (0,)

    def t_kdtree_matches_brute_force_small_structure():
        # The optimization must not change correctness: KDTREE and
        # BRUTE_FORCE must agree exactly (same pair count, same sorted
        # distances) on the same input, including the periodic-boundary
        # case that v1.0 originally fixed.
        if not HAS_SCIPY:
            return  # nothing to compare against; AUTO already falls back correctly
        structure = _make_toy_structure()
        e_bf, d_bf, _ = periodic_neighbor_search(structure, cutoff=4.0, method=NeighborSearchMethod.BRUTE_FORCE)
        e_kd, d_kd, _ = periodic_neighbor_search(structure, cutoff=4.0, method=NeighborSearchMethod.KDTREE)
        assert e_bf.shape[1] == e_kd.shape[1], (
            f"pair count mismatch: brute_force={e_bf.shape[1]} kdtree={e_kd.shape[1]}"
        )
        sorted_bf = torch.sort(d_bf).values
        sorted_kd = torch.sort(d_kd).values
        assert torch.allclose(sorted_bf, sorted_kd, atol=1e-4), (
            "KD-tree and brute-force neighbor distances disagree beyond tolerance"
        )

    def t_kdtree_handles_moderate_supercell():
        # Not a true "large supercell" stress test (that needs a real GPU
        # box and minutes, not a unit test), but verifies the KD-tree path
        # runs correctly and agrees with brute-force at a size (200 atoms,
        # ~10^4-10^5 candidate pairs after replication) where the O(M^2)
        # path is already clearly the slower of the two — i.e. it
        # exercises the actual code path this optimization targets, not
        # just the N=5 toy structure.
        if not HAS_SCIPY:
            return
        torch.manual_seed(0)
        n = 200
        lattice = torch.eye(3) * 25.0
        frac_coords = torch.rand(n, 3)
        species = ["Si"] * n
        structure = CrystalStructure(lattice=lattice, frac_coords=frac_coords, species=species, name="random_200")
        edge_index, edge_dist, _ = periodic_neighbor_search(structure, cutoff=3.0, method=NeighborSearchMethod.KDTREE)
        assert edge_index.shape[1] > 0, "expected some neighbor pairs in a dense 200-atom random cell"
        assert torch.isfinite(edge_dist).all()
        assert float(edge_dist.max()) <= 3.0 + 1e-4, "found a pair beyond the requested cutoff"

    def t_brute_force_method_still_works_explicitly():
        structure = _make_toy_structure()
        edge_index, edge_dist, _ = periodic_neighbor_search(
            structure, cutoff=4.0, method=NeighborSearchMethod.BRUTE_FORCE
        )
        assert edge_index.shape[1] > 0

    def t_kdtree_requested_without_scipy_raises_clear_error():
        if HAS_SCIPY:
            return  # can't simulate "scipy missing" without uninstalling it; skip when present
        structure = _make_toy_structure()
        try:
            periodic_neighbor_search(structure, cutoff=4.0, method=NeighborSearchMethod.KDTREE)
            raise AssertionError("expected BackendUnavailableError when scipy is missing")
        except BackendUnavailableError:
            pass

    def t_torch_cluster_matches_kdtree_small_structure():
        if not (HAS_TORCH_CLUSTER and HAS_SCIPY):
            return  # need both backends present to compare
        structure = _make_toy_structure()
        e_kd, d_kd, _ = periodic_neighbor_search(structure, cutoff=4.0, method=NeighborSearchMethod.KDTREE)
        e_tc, d_tc, _ = periodic_neighbor_search(structure, cutoff=4.0, method=NeighborSearchMethod.TORCH_CLUSTER)
        assert e_kd.shape[1] == e_tc.shape[1], (
            f"pair count mismatch: kdtree={e_kd.shape[1]} torch_cluster={e_tc.shape[1]}"
        )
        assert torch.allclose(torch.sort(d_kd).values, torch.sort(d_tc).values, atol=1e-4), (
            "KD-tree and torch_cluster neighbor distances disagree beyond tolerance"
        )

    def t_torch_cluster_handles_moderate_large_supercell():
        # Practical ceiling for a unit test: 5,000 atoms, run on CPU in
        # this environment (no GPU available here). This exercises the
        # exact code path used at >100k atoms — same algorithm, same
        # backend, just a smaller N so the test finishes quickly. True
        # 100k+-atom timing/memory characteristics must be profiled on
        # real (ideally GPU) hardware; see the v1.2 honesty note at the
        # top of this file.
        if not HAS_TORCH_CLUSTER:
            return
        torch.manual_seed(1)
        n = 5_000
        lattice = torch.eye(3) * 60.0
        frac_coords = torch.rand(n, 3)
        species = ["Si"] * n
        structure = CrystalStructure(lattice=lattice, frac_coords=frac_coords, species=species, name="random_5000")
        edge_index, edge_dist, _ = periodic_neighbor_search(
            structure, cutoff=3.0, method=NeighborSearchMethod.TORCH_CLUSTER
        )
        assert edge_index.shape[1] > 0, "expected neighbor pairs in a 5,000-atom random cell"
        assert torch.isfinite(edge_dist).all()
        assert float(edge_dist.max()) <= 3.0 + 1e-4, "found a pair beyond the requested cutoff"

    def t_torch_cluster_max_neighbors_cap_does_not_crash():
        # Deliberately dense cluster (many atoms packed close together) to
        # force the max_num_neighbors truncation path and confirm it warns
        # rather than crashing or silently returning nonsense shapes.
        if not HAS_TORCH_CLUSTER:
            return
        torch.manual_seed(2)
        n = 100
        lattice = torch.eye(3) * 5.0  # small cell -> very dense packing
        frac_coords = torch.rand(n, 3)
        structure = CrystalStructure(lattice=lattice, frac_coords=frac_coords, species=["Si"] * n, name="dense_100")
        edge_index, edge_dist, _ = periodic_neighbor_search(
            structure, cutoff=3.0, method=NeighborSearchMethod.TORCH_CLUSTER, max_num_neighbors=4
        )
        assert edge_index.shape[1] > 0
        # per-atom neighbor count must respect the cap
        counts = torch.bincount(edge_index[0], minlength=n)
        assert int(counts.max()) <= 4, f"max_num_neighbors cap violated: max count {int(counts.max())}"

    def t_torch_cluster_requested_without_package_raises_clear_error():
        if HAS_TORCH_CLUSTER:
            return  # can't simulate "missing" without uninstalling it; skip when present
        structure = _make_toy_structure()
        try:
            periodic_neighbor_search(structure, cutoff=4.0, method=NeighborSearchMethod.TORCH_CLUSTER)
            raise AssertionError("expected BackendUnavailableError when torch_cluster is missing")
        except BackendUnavailableError:
            pass

    def t_auto_prefers_torch_cluster_above_threshold():
        # Verifies the AUTO tiering logic itself (not just that each
        # backend works in isolation): above kdtree_to_gpu_threshold with
        # torch_cluster installed, AUTO must resolve to TORCH_CLUSTER.
        if not HAS_TORCH_CLUSTER:
            return
        torch.manual_seed(3)
        n = 50
        lattice = torch.eye(3) * 20.0
        frac_coords = torch.rand(n, 3)
        structure = CrystalStructure(lattice=lattice, frac_coords=frac_coords, species=["Si"] * n)
        # Force a tiny threshold so this small structure should route to TORCH_CLUSTER under AUTO.
        e_auto, d_auto, _ = periodic_neighbor_search(
            structure, cutoff=3.0, method=NeighborSearchMethod.AUTO, kdtree_to_gpu_threshold=1
        )
        e_tc, d_tc, _ = periodic_neighbor_search(structure, cutoff=3.0, method=NeighborSearchMethod.TORCH_CLUSTER)
        assert e_auto.shape[1] == e_tc.shape[1], "AUTO did not route to TORCH_CLUSTER above the threshold"

    def t_gnn_forward_pass_isolated_atom_no_crash():
        cfg = MaterialsConfig(latent_dim=16, hidden_dim=8, n_message_passes=2)
        model = DFTSurrogateGNN(cfg)
        lattice = torch.eye(3) * 100.0
        structure = CrystalStructure(lattice=lattice, frac_coords=torch.zeros(1, 3), species=["Si"])
        out = model(structure)
        assert out["latent"].shape == (1, 16)
        assert torch.isfinite(out["formation_energy_ev_per_atom"]).all()

    def t_gnn_forward_pass_toy_structure():
        cfg = MaterialsConfig(latent_dim=32, hidden_dim=16, n_message_passes=2)
        model = DFTSurrogateGNN(cfg)
        structure = _make_toy_structure()
        out = model(structure)
        assert out["latent"].shape == (1, 32)
        assert out["band_gap_ev"].item() >= 0.0, "band gap head must be non-negative (Softplus)"

    def t_tunneling_risk_monotonic_in_gate_length():
        # Longer gate length -> lower (or equal) transmission probability,
        # for a fixed barrier height. This is a basic physical sanity
        # check the WKB formula must satisfy.
        r_short = assess_tunneling_risk("test", band_gap_ev=1.0, gate_length_angstrom=5.0)
        r_long = assess_tunneling_risk("test", band_gap_ev=1.0, gate_length_angstrom=20.0)
        assert r_long.transmission_probability <= r_short.transmission_probability

    def t_tunneling_risk_rejects_invalid_inputs():
        try:
            wkb_tunneling_probability(barrier_height_ev=-1.0, barrier_width_angstrom=5.0)
            raise AssertionError("expected MaterialsONEError for negative barrier height")
        except MaterialsONEError:
            pass
        try:
            wkb_tunneling_probability(barrier_height_ev=1.0, barrier_width_angstrom=0.0)
            raise AssertionError("expected MaterialsONEError for zero barrier width")
        except MaterialsONEError:
            pass

    def t_adapter_encode_matches_agi_latent_dim():
        adapter = MaterialsONEAdapter(agi_latent_dim=64, cfg=MaterialsConfig(latent_dim=32, hidden_dim=16))
        structure = _make_toy_structure()
        latent = adapter.encode(structure)
        assert latent.shape == (1, 64), f"expected (1,64), got {tuple(latent.shape)}"
        q = adapter.get_quality_score()
        assert 0.0 <= q <= 1.0

    def t_training_loop_reduces_loss_on_toy_overfit_set():
        # Not a claim of real-world accuracy — just verifies the training
        # loop is wired correctly by overfitting a single toy example and
        # checking the loss actually goes down.
        cfg = MaterialsConfig(latent_dim=16, hidden_dim=8, n_message_passes=1)
        model = DFTSurrogateGNN(cfg)
        structure = _make_toy_structure()
        entry = MaterialsDatasetEntry(structure=structure, formation_energy_ev_per_atom=-0.5, band_gap_ev=1.1)
        history = train_dft_surrogate(model, [entry], [entry], n_epochs=20, lr=5e-2, patience=20)
        assert history["train_loss"][-1] < history["train_loss"][0], (
            f"expected loss to decrease, got {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}"
        )

    def t_training_data_validation_rejects_bad_entries():
        bad_entry = MaterialsDatasetEntry(
            structure=_make_toy_structure(), formation_energy_ev_per_atom=float("nan"), band_gap_ev=1.0
        )
        try:
            validate_dataset([bad_entry])
            raise AssertionError("expected TrainingDataError for NaN formation energy")
        except TrainingDataError:
            pass

    def t_qubit_screening_requires_opted_in_model():
        cfg = MaterialsConfig(latent_dim=16, hidden_dim=8, n_message_passes=1)  # default: opt-in heads OFF
        model = DFTSurrogateGNN(cfg)
        assert model.tc_head is None and model.tls_loss_proxy_head is None
        structure = _make_toy_structure()
        try:
            screen_qubit_candidate(structure, model)
            raise AssertionError("expected MaterialsONEError when superconductor heads are disabled")
        except MaterialsONEError:
            pass

    def t_qubit_screening_runs_when_opted_in():
        cfg = MaterialsConfig(latent_dim=16, hidden_dim=8, n_message_passes=1, predict_superconductor_properties=True)
        model = DFTSurrogateGNN(cfg)
        assert model.tc_head is not None and model.tls_loss_proxy_head is not None
        structure = _make_toy_structure()
        report = screen_qubit_candidate(structure, model)
        assert report.tc_kelvin >= 0.0, "Tc head uses Softplus; must be non-negative"
        assert report.tls_loss_proxy >= 0.0, "TLS-loss-proxy head uses Softplus; must be non-negative"
        assert isinstance(report.dilution_fridge_viable, bool)
        d = report.to_dict()
        assert "tc_kelvin" in d and "tls_loss_proxy" in d

    def t_eda_bridge_raises_clear_error_when_eda_file_absent():
        # In this development sandbox eda_qeda_adapter_layer.py is a
        # sibling upload, not necessarily importable from this script's
        # working directory — verify the failure mode is the documented
        # BackendUnavailableError, not a raw ImportError/traceback, in
        # whichever state actually holds when this test runs.
        try:
            _import_eda_qeda_adapter_layer()
            eda_available = True
        except BackendUnavailableError:
            eda_available = False
        if eda_available:
            return  # nothing to test here in this environment; bridge tests below cover the happy path
        try:
            to_eda_material_parameters()
            raise AssertionError("expected BackendUnavailableError")
        except BackendUnavailableError:
            pass

    def t_eda_bridge_full_pipeline_when_available():
        # Only meaningfully runs in an environment where
        # eda_qeda_adapter_layer.py is on the import path; skips cleanly
        # otherwise (see t_eda_bridge_raises_clear_error_when_eda_file_absent
        # for the complementary "absent" case).
        try:
            eda = _import_eda_qeda_adapter_layer()
        except BackendUnavailableError:
            return
        cfg = MaterialsConfig(latent_dim=16, hidden_dim=8, n_message_passes=1, predict_superconductor_properties=True)
        model = DFTSurrogateGNN(cfg)
        structure = _make_toy_structure()
        u_field = torch.rand(8, 8)
        sigma_field = torch.rand(8, 8) + 0.1  # keep strictly positive; map_superconducting_qubit expects sigma > 0
        result = screen_candidate_for_qubit_device(structure, model, u_field, sigma_field)
        assert "material_screening" in result and "device_grids" in result
        assert "loss_tangent_grid" in result["device_grids"]
        assert "Q_grid" in result["device_grids"]

    check("structure_validation_rejects_singular_lattice", t_structure_validation_rejects_singular_lattice)
    check("structure_validation_rejects_unknown_element", t_structure_validation_rejects_unknown_element)
    check("periodic_neighbor_search_finds_boundary_neighbors", t_periodic_neighbor_search_finds_boundary_neighbors)
    check("neighbor_search_empty_graph_handled", t_neighbor_search_empty_graph_handled)
    check("kdtree_matches_brute_force_small_structure", t_kdtree_matches_brute_force_small_structure)
    check("kdtree_handles_moderate_supercell", t_kdtree_handles_moderate_supercell)
    check("brute_force_method_still_works_explicitly", t_brute_force_method_still_works_explicitly)
    check("kdtree_requested_without_scipy_raises_clear_error", t_kdtree_requested_without_scipy_raises_clear_error)
    check("torch_cluster_matches_kdtree_small_structure", t_torch_cluster_matches_kdtree_small_structure)
    check("torch_cluster_handles_moderate_large_supercell", t_torch_cluster_handles_moderate_large_supercell)
    check("torch_cluster_max_neighbors_cap_does_not_crash", t_torch_cluster_max_neighbors_cap_does_not_crash)
    check("torch_cluster_requested_without_package_raises_clear_error", t_torch_cluster_requested_without_package_raises_clear_error)
    check("auto_prefers_torch_cluster_above_threshold", t_auto_prefers_torch_cluster_above_threshold)
    check("gnn_forward_pass_isolated_atom_no_crash", t_gnn_forward_pass_isolated_atom_no_crash)
    check("gnn_forward_pass_toy_structure", t_gnn_forward_pass_toy_structure)
    check("tunneling_risk_monotonic_in_gate_length", t_tunneling_risk_monotonic_in_gate_length)
    check("tunneling_risk_rejects_invalid_inputs", t_tunneling_risk_rejects_invalid_inputs)
    check("adapter_encode_matches_agi_latent_dim", t_adapter_encode_matches_agi_latent_dim)
    check("training_loop_reduces_loss_on_toy_overfit_set", t_training_loop_reduces_loss_on_toy_overfit_set)
    check("training_data_validation_rejects_bad_entries", t_training_data_validation_rejects_bad_entries)
    check("qubit_screening_requires_opted_in_model", t_qubit_screening_requires_opted_in_model)
    check("qubit_screening_runs_when_opted_in", t_qubit_screening_runs_when_opted_in)
    check("eda_bridge_raises_clear_error_when_eda_file_absent", t_eda_bridge_raises_clear_error_when_eda_file_absent)
    check("eda_bridge_full_pipeline_when_available", t_eda_bridge_full_pipeline_when_available)

    print("=" * 65)
    print("  MATERIALS ONE v1.3 — Self-Test Suite")
    print("=" * 65)
    all_pass = True
    for name, passed, msg in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}" + (f"  ({msg})" if msg else ""))
        all_pass = all_pass and passed
    print("=" * 65)
    print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 65)
    return all_pass


# =============================================================================
# SECTION 10 — MAIN
# =============================================================================

if __name__ == "__main__":
    ok = run_self_tests()

    print(f"\nReal ML-DFT physics available (mace-torch+ase): {HAS_MACE and HAS_ASE}")
    structure = _make_toy_structure()
    potential = InteratomicPotential()
    energy, forces = potential.energy_forces(structure)
    print(f"Potential energy: {energy:.4f} eV  (real physics: {potential.is_real_physics})")

    report = assess_tunneling_risk(
        "toy_SiC", band_gap_ev=1.1, gate_length_angstrom=20.0, dominant_element="Si"
    )
    print(f"Tunneling risk @ 2nm gate length: {report.to_dict()}")

    print(
        "\nREMINDER: DFTSurrogateGNN ships untrained. The self-test suite "
        "verifies the engineering (validation, periodic neighbor search, "
        "training loop, adapter contract) is correct — it does NOT verify "
        "materials-science accuracy. Train on Materials Project / OQMD data "
        "via train_dft_surrogate() before using this for real screening."
    )

    if not ok:
        raise SystemExit(1)
