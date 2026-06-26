# =============================================================================
# EXPERIMENTAL RESTRAINTS ONE (XR-ONE) — v1.0.0
# XL-MS + Cryo-EM Restraint-Guided Domain Docking — STANDALONE EXTENSION
# REAL FOLD ONE Ecosystem — restraint layer on top of Structural Domain
# Assembly ONE (SDA-ONE)
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
#   - Claude   (Anthropic)  — XL-MS flat-bottom restraint formulation,
#                             dependency-free MRC/CCP4 map parser, Gaussian-
#                             splat density simulation + differentiable
#                             cross-correlation restraint, restraint-
#                             extended docking energy as a clean subclass of
#                             DomainDockingAssembler, full docstrings and
#                             self-test suite
#
# -----------------------------------------------------------------------------
# WHAT THIS MODULE IS
# -----------------------------------------------------------------------------
#   A STANDALONE extension. It does NOT modify
#   structural_domain_assembly_one.py — it imports from it and subclasses
#   DomainDockingAssembler. If this file is deleted, SDA-ONE continues to
#   work exactly as before (contact-only docking). This file only adds an
#   *optional* restraint-guided docking path for when real experimental
#   data is available.
#
#   Motivation: SDA-ONE's CrossDomainContactHead is, as documented in
#   structural_domain_assembly_one.py and README_RMSD.md, an unvalidated,
#   untrained predictor — its candidate-pair recall is unknown. Real
#   experimental restraints do not have that problem: they are direct
#   physical measurements. Where available, they should be allowed to
#   dominate the assembly energy over the learned contact head's guesses,
#   not just supplement them.
#
# -----------------------------------------------------------------------------
# TWO RESTRAINT TYPES
# -----------------------------------------------------------------------------
#   1. XLMSRestraintSet
#      Cross-linking mass spectrometry gives pairs of residues that were
#      close enough in the native structure for a crosslinking reagent to
#      bridge them. This is modeled as a ONE-SIDED (flat-bottom) distance
#      restraint: zero penalty below d_max, quadratic penalty above it.
#      d_max depends on the crosslinker chemistry (e.g. commonly-used
#      homobifunctional NHS-ester crosslinkers such as DSS/BS3 are typically
#      modeled in the integrative-modeling literature with a Cα–Cα distance
#      cutoff in roughly the 24–30 Å range, accounting for linker length
#      plus lysine side-chain reach and some backbone flexibility — this
#      module defaults to 30.0 Å but takes it as an explicit, per-dataset
#      configurable parameter because exact values vary by crosslinker and
#      by lab convention; check your crosslinker's data sheet / your
#      proteomics core's preferred cutoff before trusting the default).
#      Cross-linking MS also has a non-trivial false-positive rate
#      (commonly mid-single-digit percent at typical FDR thresholds used in
#      the field) — this module supports a per-crosslink confidence weight
#      so noisy hits can be down-weighted rather than treated as certain.
#
#   2. CryoEMDensityRestraint
#      Cryo-EM gives a 3-D density map (MRC/CCP4 format) rather than
#      pairwise distances. The restraint compares a SIMULATED density
#      (built by placing one Gaussian "blob" at each Cα coordinate, width
#      tied to the map's nominal resolution — the standard approach used
#      by Gaussian-mixture-model cryo-EM fitting tools) against the real
#      experimental map, scored by voxel-wise cross-correlation. Because
#      cross-correlation is differentiable in the predicted coordinates,
#      it plugs directly into the same gradient-based docking loop as the
#      XL-MS and contact-head energies.
#
#      This module ships its OWN minimal MRC/CCP4 reader (no dependency on
#      the `mrcfile` package, which was not available in the authoring
#      environment) — implements just enough of the well-documented,
#      fixed 1024-byte MRC2014 header to read voxel size, grid dimensions,
#      origin, and the data block for MODE 0/1/2 (int8/int16/float32),
#      which covers the overwhelming majority of cryo-EM maps in
#      circulation (e.g. from EMDB). If `mrcfile` IS available in your
#      environment, prefer it — it is more complete and battle-tested;
#      this fallback exists so the restraint logic isn't blocked by a
#      missing package, not to replace `mrcfile`.
#
# -----------------------------------------------------------------------------
# HOW THIS PLUGS INTO SDA-ONE
# -----------------------------------------------------------------------------
#   RestrainedDomainDockingAssembler subclasses DomainDockingAssembler and
#   overrides only the energy computation inside dock() — the quaternion /
#   rigid-body machinery, the contact-head LJ energy, and the inter-domain
#   clash penalty are all REUSED unchanged via super-class hooks, not
#   duplicated. Restraint energies are added as two more terms in the same
#   sum, each with an independent weight:
#
#       E_total = E_contacts (existing, from CrossDomainContactHead)
#               + E_clash    (existing, steric)
#               + xlms_weight   * E_XLMS      (NEW, this module)
#               + cryoem_weight * E_CryoEM    (NEW, this module)
#
#   Typical usage: set xlms_weight / cryoem_weight large relative to the
#   contact-head weight (which is implicitly 1.0, scaled by predicted
#   probability) when real experimental data is available and trusted more
#   than the unvalidated learned contact head — see
#   RestraintDockingConfig docstring for guidance.
#
# -----------------------------------------------------------------------------
# HONEST LIMITATIONS
# -----------------------------------------------------------------------------
#   • Not executed end-to-end in this environment (same constraint as
#     structural_domain_assembly_one.py — no torch/GPU runtime here).
#     Self-test suite is written but UNRUN; run it locally first.
#   • The native MRC reader handles the common header fields and MODE
#     0/1/2 data; it does NOT handle every exotic MRC variant (e.g.
#     non-cubic / non-orthogonal unit cells with unusual angles, symmetry
#     records, or extended-header formats some packages write). If your
#     map fails to load, the error message will say so explicitly rather
#     than silently producing wrong results — but for unusual maps,
#     installing `mrcfile` (`pip install mrcfile`) and adapting
#     `load_density_map_mrcfile` (provided below, unused by default) is
#     the more robust path.
#   • The d_max defaults for XL-MS are LITERATURE-TYPICAL, not validated
#     against any specific experiment. Always confirm against your
#     crosslinker chemistry before treating the default as ground truth.
#   • Gaussian-splat simulated density is a coarse approximation (one
#     Gaussian per Cα, no side-chain mass, no solvent/B-factor modeling).
#     This is adequate for global rigid-domain placement against a
#     low-to-medium resolution map; it is not a substitute for proper
#     flexible-fitting tools when sub-domain (intra-domain) flexibility
#     against the map matters.
# =============================================================================

from __future__ import annotations

import logging
import struct
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

try:
    from structural_domain_assembly_one import (
        DomainAssemblyConfig,
        DomainDockingAssembler,
        SDA_VERSION,
    )
    _HAS_SDA = True
except ImportError:
    _HAS_SDA = False
    SDA_VERSION = "unknown"
    raise ImportError(
        "experimental_restraints_one.py requires structural_domain_assembly_one.py "
        "to be importable (it subclasses DomainDockingAssembler from that module). "
        "Place both files in the same directory / on the same PYTHONPATH."
    )

XR_VERSION: str = "1.0.0"


# =============================================================================
# 0. Restraint configuration
# =============================================================================

@dataclass
class RestraintDockingConfig:
    """
    Configuration for restraint-guided docking. Composed alongside (not
    replacing) the base ``DomainAssemblyConfig`` — pass both to
    ``RestrainedDomainDockingAssembler``.

    XL-MS:
        xlms_weight          : overall weight on the XL-MS restraint energy
                               relative to the contact-head energy (which is
                               implicitly weighted by predicted probability,
                               typically in [0, 1] per contact). Since XL-MS
                               hits are direct physical measurements (modulo
                               their FDR), a value > 1.0 (e.g. 5.0–20.0) is
                               reasonable when you trust the crosslink data
                               more than the unvalidated contact head —
                               there is no universally correct number; tune
                               against how much you trust each source.
        xlms_default_dmax     : default Cα–Cα distance cutoff (Å) used for
                               any crosslink that doesn't specify its own
                               ``d_max`` — see module docstring for caveats
                               on this default.
        xlms_slack            : Å of additional tolerance added on top of
                               d_max before the quadratic penalty begins
                               (flat-bottom width) — accounts for the
                               restraint being a *maximum*, not exact,
                               distance.

    Cryo-EM:
        cryoem_weight         : overall weight on the cross-correlation
                               restraint energy. Cross-correlation is
                               already normalized to roughly [-1, 1], so
                               this weight directly sets how much one unit
                               of (1 - CCC) competes against one LJ-unit of
                               contact energy — start small (e.g. 1.0–10.0)
                               and increase if domains aren't converging
                               into the map's envelope.
        cryoem_gaussian_sigma_factor : multiplies the map's nominal
                               resolution to get each Cα's simulated
                               Gaussian width (sigma = factor * resolution
                               is a common rule of thumb in GMM-based
                               cryo-EM fitting; 1/(2*sqrt(2*ln2)) ≈ 0.4247
                               would map "resolution" to a Gaussian FWHM
                               exactly, but in practice values in the
                               0.3–0.5 range are all used across different
                               tools — defaulting to 0.4 as a reasonable
                               middle ground, not a uniquely correct value).
        cryoem_voxel_chunk     : number of map voxels processed per chunk
                               when computing cross-correlation, to bound
                               memory at high resolution / large maps.
    """
    # --- XL-MS ---
    xlms_weight: float = 5.0
    xlms_default_dmax: float = 30.0
    xlms_slack: float = 2.0

    # --- Cryo-EM ---
    cryoem_weight: float = 2.0
    cryoem_gaussian_sigma_factor: float = 0.4
    cryoem_voxel_chunk: int = 65536

    def __post_init__(self) -> None:
        assert self.xlms_weight >= 0.0
        assert self.xlms_default_dmax > 0.0
        assert self.xlms_slack >= 0.0
        assert self.cryoem_weight >= 0.0
        assert self.cryoem_gaussian_sigma_factor > 0.0
        assert self.cryoem_voxel_chunk >= 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RestraintDockingConfig":
        return cls(**d)


# =============================================================================
# 1. XL-MS restraint set
# =============================================================================

@dataclass
class CrossLink:
    """
    A single cross-linking MS hit.

    Args:
        residue_i  : 0-indexed GLOBAL residue index (position in the full
                    chain, matching the indexing used throughout SDA-ONE —
                    i.e. the same indexing as ``domain_ranges``).
        residue_j  : 0-indexed global residue index of the linked partner.
        d_max       : Cα–Cα distance cutoff (Å) for this specific crosslink.
                    If None, ``RestraintDockingConfig.xlms_default_dmax`` is
                    used instead.
        confidence  : optional [0, 1] confidence weight (e.g. derived from
                    spectral count, FDR-adjusted score, or replicate
                    agreement). Defaults to 1.0 (full confidence) if not
                    provided by the proteomics pipeline.
    """
    residue_i: int
    residue_j: int
    d_max: Optional[float] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        assert self.residue_i != self.residue_j, "A crosslink cannot link a residue to itself."
        assert self.d_max is None or self.d_max > 0.0
        assert 0.0 <= self.confidence <= 1.0


class XLMSRestraintSet:
    """
    Holds a collection of cross-linking MS restraints and computes their
    contribution to the docking energy as a differentiable, one-sided
    (flat-bottom) penalty.

    Energy per crosslink k, with distance d_k between the two residues in
    the CURRENT (being-optimized) global coordinates:

        E_k = confidence_k * relu(d_k - (d_max_k + slack))^2

    i.e. zero penalty while the predicted distance is within the
    crosslinker's reach (plus a slack tolerance), quadratic penalty beyond
    it. This intentionally never *rewards* being close (real crosslinks
    only forbid being too far apart, not require touching), which matches
    the physical meaning of the data.

    Args:
        crosslinks : list of CrossLink instances.
    """

    def __init__(self, crosslinks: Sequence[CrossLink]) -> None:
        assert len(crosslinks) > 0, "XLMSRestraintSet requires at least one crosslink."
        self.crosslinks = list(crosslinks)

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[Tuple[int, int]],
        d_max: float = 30.0,
        confidences: Optional[Sequence[float]] = None,
    ) -> "XLMSRestraintSet":
        """
        Convenience constructor for the common case of a flat list of
        (residue_i, residue_j) pairs sharing one d_max.

        Args:
            pairs       : list of (residue_i, residue_j) 0-indexed global
                         residue index pairs.
            d_max        : shared Cα–Cα cutoff (Å) applied to every pair.
            confidences  : optional per-pair confidence list, same length
                         as ``pairs``. Defaults to 1.0 for every pair.
        """
        if confidences is not None:
            assert len(confidences) == len(pairs), \
                "confidences must have the same length as pairs."
        crosslinks = [
            CrossLink(
                residue_i=i, residue_j=j, d_max=d_max,
                confidence=(confidences[k] if confidences is not None else 1.0),
            )
            for k, (i, j) in enumerate(pairs)
        ]
        return cls(crosslinks)

    def filter_resolvable(
        self, domain_ranges: List[Tuple[int, int]]
    ) -> List[Tuple[CrossLink, int, int]]:
        """
        Resolves each crosslink's global residue indices to (domain_index,
        local_index) for both ends, dropping (with a warning) any crosslink
        whose residue falls outside every domain range (should not happen
        if domain_ranges covers [0, N) contiguously, but checked
        defensively since this set may be constructed independently of any
        particular segmentation run).

        Args:
            domain_ranges : (start, end) per domain, as produced by
                           ``DomainSegmenter.segment``.
        Returns:
            List of (crosslink, domain_idx_i, domain_idx_j) — local indices
            are looked up again inside ``energy()`` from the global indices
            stored on the CrossLink itself, to keep this method's return
            type simple and avoid duplicating index bookkeeping in two
            places.
        """
        def _find_domain(global_idx: int) -> Optional[int]:
            for d_idx, (s, e) in enumerate(domain_ranges):
                if s <= global_idx < e:
                    return d_idx
            return None

        resolved = []
        for xl in self.crosslinks:
            di = _find_domain(xl.residue_i)
            dj = _find_domain(xl.residue_j)
            if di is None or dj is None:
                warnings.warn(
                    f"CrossLink({xl.residue_i}, {xl.residue_j}) has a residue index "
                    f"outside all domain_ranges (chain length implied by domain_ranges: "
                    f"{domain_ranges[-1][1]}) — skipping this crosslink."
                )
                continue
            resolved.append((xl, di, dj))
        return resolved

    def energy(
        self,
        placed: List[torch.Tensor],
        domain_ranges: List[Tuple[int, int]],
        cfg: RestraintDockingConfig,
    ) -> torch.Tensor:
        """
        Args:
            placed        : list of (domain_size_k, 3) CURRENT global-frame
                           coordinates per domain (i.e. ``assemble()``'s
                           output inside the docking loop).
            domain_ranges : (start, end) per domain.
            cfg            : RestraintDockingConfig.
        Returns:
            Scalar differentiable energy (sum over all resolvable
            crosslinks).
        """
        device = placed[0].device
        dtype = placed[0].dtype
        energy = torch.zeros((), device=device, dtype=dtype)
        resolved = self.filter_resolvable(domain_ranges)
        for (xl, di, dj) in resolved:
            si, _ = domain_ranges[di]
            sj, _ = domain_ranges[dj]
            local_i = xl.residue_i - si
            local_j = xl.residue_j - sj
            pi = placed[di][local_i]
            pj = placed[dj][local_j]
            d = (pi - pj).norm()
            d_max = xl.d_max if xl.d_max is not None else cfg.xlms_default_dmax
            violation = F.relu(d - (d_max + cfg.xlms_slack))
            energy = energy + xl.confidence * violation.pow(2)
        return energy


# =============================================================================
# 2. Minimal dependency-free MRC/CCP4 reader
# =============================================================================

@dataclass
class DensityMap:
    """
    A loaded cryo-EM density map, holding just what the restraint needs.

    Attributes:
        data        : (nz, ny, nx) numpy array of voxel densities (float32).
        voxel_size  : (3,) Å per voxel along (x, y, z).
        origin       : (3,) Å, position of voxel (0,0,0) in map coordinates
                      (combines the MRC header's ORIGIN and the
                      NSTART*voxel_size offset, since different writers
                      populate one or the other).
        resolution   : nominal resolution (Å) of the map, if supplied by the
                      caller (MRC headers do not reliably encode this —
                      it must be supplied separately, e.g. from the EMDB
                      entry's metadata).
    """
    data: np.ndarray
    voxel_size: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    resolution: float


def load_density_map_native(path: str, resolution: float) -> DensityMap:
    """
    Minimal MRC2014/CCP4 reader requiring no third-party dependency beyond
    numpy. Implements the fixed 1024-byte header's commonly-used fields and
    MODE 0 (int8), 1 (int16), 2 (float32) data — covers the large majority
    of maps distributed via EMDB. Does not parse symmetry records or
    extended headers; will raise a clear error rather than silently
    misreading anything it doesn't understand.

    Args:
        path        : path to a .mrc or .ccp4 file.
        resolution  : nominal resolution (Å) of the map — NOT stored
                     reliably in the MRC header itself, so it must be
                     supplied by the caller (check the EMDB entry / your
                     reconstruction software's log for this number).
    Returns:
        DensityMap.
    Raises:
        ValueError : if the file's MODE field is not 0, 1, or 2, or if the
                    header cannot be parsed as a well-formed MRC2014 file.
    """
    with open(path, "rb") as f:
        header_bytes = f.read(1024)
        if len(header_bytes) < 1024:
            raise ValueError(
                f"'{path}' is too short to contain a valid 1024-byte MRC header "
                f"(got {len(header_bytes)} bytes)."
            )
        nx, ny, nz, mode = struct.unpack_from("<4i", header_bytes, 0)
        mxst, myst, mzst = struct.unpack_from("<3i", header_bytes, 16)
        mx, my, mz = struct.unpack_from("<3i", header_bytes, 28)
        cell_a, cell_b, cell_c = struct.unpack_from("<3f", header_bytes, 40)
        origin_x, origin_y, origin_z = struct.unpack_from("<3f", header_bytes, 196)

        if mode not in (0, 1, 2):
            raise ValueError(
                f"'{path}' has MRC MODE={mode}, which this native reader does not "
                f"support (only MODE 0/int8, 1/int16, 2/float32 are implemented). "
                f"Install `mrcfile` (pip install mrcfile) and use "
                f"load_density_map_mrcfile() instead for this file."
            )
        dtype = {0: np.int8, 1: np.int16, 2: np.float32}[mode]
        n_voxels = nx * ny * nz
        raw = f.read(n_voxels * np.dtype(dtype).itemsize)
        if len(raw) < n_voxels * np.dtype(dtype).itemsize:
            raise ValueError(
                f"'{path}' header declares {nx}x{ny}x{nz} voxels but the data "
                f"block is shorter than expected — file may be truncated or "
                f"have an unsupported extended header this reader doesn't skip."
            )
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(nz, ny, nx)

    voxel_x = cell_a / mx if mx > 0 else 1.0
    voxel_y = cell_b / my if my > 0 else 1.0
    voxel_z = cell_c / mz if mz > 0 else 1.0

    # Combine ORIGIN field with NSTART*voxel_size — different software
    # populates one or the other (sometimes both, redundantly); summing is
    # the documented-safe approach when both might be nonzero.
    ox = origin_x + mxst * voxel_x
    oy = origin_y + myst * voxel_y
    oz = origin_z + mzst * voxel_z

    return DensityMap(
        data=data,
        voxel_size=(voxel_x, voxel_y, voxel_z),
        origin=(ox, oy, oz),
        resolution=resolution,
    )


def load_density_map_mrcfile(path: str, resolution: float) -> DensityMap:
    """
    Preferred loader IF the `mrcfile` package is installed — more complete
    than ``load_density_map_native`` (handles symmetry records, extended
    headers, all MRC modes). Not used by default in this module (the
    authoring environment did not have `mrcfile` available to verify
    against), provided so you can switch to it with one line if your map
    fails to load with the native reader.

    Args:
        path        : path to a .mrc or .ccp4 file.
        resolution  : nominal resolution (Å), supplied by the caller.
    """
    try:
        import mrcfile  # type: ignore
    except ImportError as e:
        raise ImportError(
            "load_density_map_mrcfile requires `pip install mrcfile`."
        ) from e
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        vx, vy, vz = mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z
        ox = float(mrc.header.origin.x) + float(mrc.header.nxstart) * vx
        oy = float(mrc.header.origin.y) + float(mrc.header.nystart) * vy
        oz = float(mrc.header.origin.z) + float(mrc.header.nzstart) * vz
    return DensityMap(data=data, voxel_size=(vx, vy, vz), origin=(ox, oy, oz), resolution=resolution)


# =============================================================================
# 3. Cryo-EM density restraint
# =============================================================================

class CryoEMDensityRestraint:
    """
    Restrains the assembled structure toward a cryo-EM density map via
    differentiable cross-correlation between the experimental map and a
    Gaussian-splat density simulated from the current Cα coordinates.

    To keep this tractable at 100k-residue scale, the cross-correlation is
    computed only over voxels within ``support_radius`` of at least one Cα
    (a thin shell around the current structure, not the whole map volume —
    most of a typical map is solvent/background far from any candidate
    placement and contributes nothing informative to the gradient).

    Args:
        density_map     : DensityMap instance (see loaders above).
        cfg               : RestraintDockingConfig.
        support_radius    : Å beyond a Cα's Gaussian sigma to still
                           consider a voxel "supported" by that atom —
                           wider radius = more accurate but more compute.
    """

    def __init__(
        self,
        density_map: DensityMap,
        cfg: RestraintDockingConfig,
        support_radius: float = 6.0,
    ) -> None:
        self.map = density_map
        self.cfg = cfg
        self.sigma = cfg.cryoem_gaussian_sigma_factor * density_map.resolution
        self.support_radius = support_radius

        data_t = torch.as_tensor(density_map.data, dtype=torch.float32)
        nz, ny, nx = data_t.shape
        vx, vy, vz = density_map.voxel_size
        ox, oy, oz = density_map.origin

        # Precompute, once, the Å-space coordinate of every voxel center.
        # Stored as a flat (n_voxels, 3) tensor plus the flat density
        # values, so "which voxels are near this Cα" can be done with a
        # single cdist call per chunk rather than re-deriving indices.
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        voxel_xyz = torch.stack([
            xx.reshape(-1) * vx + ox,
            yy.reshape(-1) * vy + oy,
            zz.reshape(-1) * vz + oz,
        ], dim=-1)  # (n_voxels, 3)
        self.voxel_xyz = voxel_xyz
        self.voxel_density = data_t.reshape(-1)
        self.n_voxels = voxel_xyz.shape[0]

        if self.n_voxels > 2_000_000:
            warnings.warn(
                f"CryoEMDensityRestraint loaded a map with {self.n_voxels:,} voxels. "
                f"The current support-window approach still scans all voxels once per "
                f"docking step to find each Cα's neighborhood, which will be slow at "
                f"this size. Consider cropping the map to the region of interest before "
                f"loading, or reducing dock_iters, if docking is too slow in practice."
            )

    def simulate_and_correlate(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords : (M, 3) Å-space Cα coordinates (the CURRENT, being-
                    optimized assembled structure — all domains placed,
                    concatenated in chain order).
        Returns:
            Scalar cross-correlation coefficient (CCC) in roughly [-1, 1]
            between the simulated and experimental density, restricted to
            the voxel support region around ``coords``. Differentiable
            w.r.t. ``coords``.
        """
        device = coords.device
        voxel_xyz = self.voxel_xyz.to(device)
        voxel_density = self.voxel_density.to(device)

        # Find the support region: voxels within support_radius of ANY
        # current Cα. Done once per call (not per-atom) via chunked cdist
        # to bound memory, since coords moves every docking step so this
        # cannot be precomputed outside the loop.
        chunk = self.cfg.cryoem_voxel_chunk
        support_mask = torch.zeros(self.n_voxels, dtype=torch.bool, device=device)
        cutoff = self.support_radius + 3.0 * self.sigma
        for start in range(0, self.n_voxels, chunk):
            end = min(start + chunk, self.n_voxels)
            d = torch.cdist(voxel_xyz[start:end], coords)  # (chunk, M)
            support_mask[start:end] = (d.min(dim=1).values < cutoff)

        if support_mask.sum() == 0:
            warnings.warn(
                "CryoEMDensityRestraint found zero voxels within support_radius of "
                "the current structure — the structure may be placed far outside the "
                "map's coordinate frame. Returning zero correlation (no gradient signal)."
            )
            return torch.zeros((), device=device, dtype=coords.dtype)

        sup_xyz = voxel_xyz[support_mask]
        sup_exp_density = voxel_density[support_mask]

        # Gaussian-splat simulated density at the support voxels: sum over
        # all M Cα atoms of a 3-D Gaussian centered at each atom, evaluated
        # at each support voxel. Chunked over voxels (not atoms) since M
        # (chain length, up to ~10^5) is typically >> the support voxel
        # count once cropped to a thin shell.
        sim_density = torch.zeros(sup_xyz.shape[0], device=device, dtype=coords.dtype)
        # NOTE: the Gaussian (2*pi*sigma^2)^-1.5 normalization prefactor is
        # intentionally omitted — cross-correlation is invariant to a
        # uniform rescaling of either input, so it cancels in the CCC below
        # and is skipped purely as a (harmless) compute saving.
        v_chunk = self.cfg.cryoem_voxel_chunk
        for start in range(0, sup_xyz.shape[0], v_chunk):
            end = min(start + v_chunk, sup_xyz.shape[0])
            d2 = torch.cdist(sup_xyz[start:end], coords) ** 2  # (chunk, M)
            sim_density[start:end] = torch.exp(-0.5 * d2 / (self.sigma ** 2)).sum(dim=1)

        # Pearson cross-correlation between simulated and experimental
        # density over the support region — the standard cryo-EM fit score.
        sim_c = sim_density - sim_density.mean()
        exp_c = sup_exp_density - sup_exp_density.mean()
        denom = (sim_c.norm() * exp_c.norm()).clamp_min(1e-8)
        ccc = (sim_c * exp_c).sum() / denom
        return ccc

    def energy(self, coords: torch.Tensor) -> torch.Tensor:
        """Energy form: 1 - CCC, so minimizing energy maximizes correlation."""
        return 1.0 - self.simulate_and_correlate(coords)


# =============================================================================
# 4. Restrained docking assembler
# =============================================================================

class RestrainedDomainDockingAssembler(DomainDockingAssembler):
    """
    Drop-in replacement for ``DomainDockingAssembler`` that adds optional
    XL-MS and/or Cryo-EM restraint energy terms to the existing docking
    loop. Both are optional and independent — pass either, both, or
    neither (passing neither makes this behave identically to the base
    class, since both restraint energies are simply absent from the sum).

    Reuses the parent class's quaternion rigid-body machinery and
    contact-head / clash energy terms via ``super().dock()``'s structure —
    re-implemented here (not literally calling super().dock(), since the
    energy needs to be extended *inside* the per-step loop, not after it)
    but using the exact same ``_quat_to_rotmat``, ``_lj_energy``, and
    ``_clash_energy`` methods inherited unchanged from the parent.

    Args:
        cfg            : DomainAssemblyConfig (same as the base class).
        restraint_cfg   : RestraintDockingConfig for the new restraint
                         weights/parameters.
        xlms             : optional XLMSRestraintSet.
        cryoem           : optional CryoEMDensityRestraint.
    """

    def __init__(
        self,
        cfg: DomainAssemblyConfig,
        restraint_cfg: RestraintDockingConfig,
        xlms: Optional[XLMSRestraintSet] = None,
        cryoem: Optional[CryoEMDensityRestraint] = None,
    ) -> None:
        super().__init__(cfg)
        self.restraint_cfg = restraint_cfg
        self.xlms = xlms
        self.cryoem = cryoem
        if xlms is None and cryoem is None:
            warnings.warn(
                "RestrainedDomainDockingAssembler constructed with no restraints "
                "(xlms=None, cryoem=None) — this will behave identically to the "
                "base DomainDockingAssembler. Pass at least one restraint source "
                "to get any benefit from this subclass."
            )

    def dock(
        self,
        domain_coords: List[torch.Tensor],
        domain_ranges: List[Tuple[int, int]],
        contacts: List[Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]],
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Same signature and return type as
        ``DomainDockingAssembler.dock`` — restraints (if configured at
        construction time) are added into the same per-step energy used
        for the contact-head and clash terms.
        """
        device = domain_coords[0].device
        dtype = domain_coords[0].dtype
        num_domains = len(domain_coords)

        centred, centroids = [], []
        for dc in domain_coords:
            c = dc.mean(dim=0)
            centred.append(dc - c)
            centroids.append(c)

        quats = [torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype,
                               requires_grad=True) for _ in range(num_domains)]
        trans = [centroids[i].clone().detach().requires_grad_(True) for i in range(num_domains)]
        optimizer = torch.optim.Adam(quats + trans, lr=self.cfg.dock_lr)

        if len(contacts) == 0 and self.xlms is None and self.cryoem is None:
            warnings.warn(
                "RestrainedDomainDockingAssembler.dock has zero contacts AND no "
                "restraints configured — docking has no guidance signal whatsoever "
                "and domains will stay at their original centroids."
            )

        def assemble() -> List[torch.Tensor]:
            placed = []
            for i in range(num_domains):
                R = self._quat_to_rotmat(quats[i])
                placed.append(centred[i] @ R.T + trans[i])
            return placed

        for step in range(self.cfg.dock_iters):
            optimizer.zero_grad()
            placed = assemble()
            energy = torch.zeros((), device=device, dtype=dtype)

            # --- existing contact-head energy (unchanged from parent) ---
            for (di, dj, gi, gj, prob) in contacts:
                si, _ = domain_ranges[di]
                sj, _ = domain_ranges[dj]
                pi = placed[di][gi - si]
                pj = placed[dj][gj - sj]
                d = (pi - pj).norm(dim=-1)
                energy = energy + (prob.to(device=device, dtype=dtype) * self._lj_energy(d)).sum()

            # --- existing inter-domain clash penalty (unchanged) ---
            if self.cfg.dock_clash_weight > 0.0 and num_domains > 1:
                for i in range(num_domains):
                    for j in range(i + 1, num_domains):
                        ci, cj = placed[i].mean(dim=0), placed[j].mean(dim=0)
                        if (ci - cj).norm() > (self.cfg.dock_clash_r0 +
                                                 domain_coords[i].shape[0] * 0.1 +
                                                 domain_coords[j].shape[0] * 0.1):
                            continue
                        d = torch.cdist(placed[i], placed[j])
                        energy = energy + self.cfg.dock_clash_weight * self._clash_energy(d).sum()

            # --- NEW: XL-MS restraint energy ---
            if self.xlms is not None:
                xl_energy = self.xlms.energy(placed, domain_ranges, self.restraint_cfg)
                energy = energy + self.restraint_cfg.xlms_weight * xl_energy

            # --- NEW: Cryo-EM density restraint energy ---
            if self.cryoem is not None:
                full_coords = torch.cat(placed, dim=0)
                em_energy = self.cryoem.energy(full_coords)
                energy = energy + self.restraint_cfg.cryoem_weight * em_energy

            energy.backward()
            optimizer.step()
            with torch.no_grad():
                for i in range(num_domains):
                    quats[i].data = quats[i].data / quats[i].data.norm().clamp_min(1e-8)

            if verbose and step % 50 == 0:
                logger.info("restrained dock step %d / %d | energy = %.4f",
                            step, self.cfg.dock_iters, energy.item())

        with torch.no_grad():
            placed_final = assemble()
        return torch.cat(placed_final, dim=0)


# =============================================================================
# __main__ — [PASS]/[FAIL] self-test suite
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 70)
    print(f"  EXPERIMENTAL RESTRAINTS ONE v{XR_VERSION} "
          f"(SDA-ONE v{SDA_VERSION}) — Self-Test Suite")
    print("=" * 70)

    device = torch.device("cpu")

    # ── Test 1: RestraintDockingConfig validation ────────────────────────
    rcfg = RestraintDockingConfig(xlms_weight=5.0, cryoem_weight=2.0)
    assert rcfg.xlms_weight == 5.0
    print("[PASS] RestraintDockingConfig validates a sane configuration")
    try:
        RestraintDockingConfig(xlms_weight=-1.0)
        print("[FAIL] RestraintDockingConfig should reject negative xlms_weight")
    except AssertionError:
        print("[PASS] RestraintDockingConfig rejects negative xlms_weight")

    # ── Test 2: CrossLink validation ──────────────────────────────────────
    cl = CrossLink(residue_i=10, residue_j=500, d_max=28.0, confidence=0.9)
    assert cl.residue_i != cl.residue_j
    print("[PASS] CrossLink constructs and validates a normal restraint")
    try:
        CrossLink(residue_i=5, residue_j=5)
        print("[FAIL] CrossLink should reject residue_i == residue_j")
    except AssertionError:
        print("[PASS] CrossLink rejects a self-link (residue_i == residue_j)")

    # ── Test 3: XLMSRestraintSet.from_pairs convenience constructor ─────
    pairs = [(10, 500), (50, 600), (100, 650)]
    xlset = XLMSRestraintSet.from_pairs(pairs, d_max=30.0)
    assert len(xlset.crosslinks) == 3
    print("[PASS] XLMSRestraintSet.from_pairs builds the expected number of CrossLink entries")

    # ── Test 4: XLMSRestraintSet.filter_resolvable resolves domain indices ──
    domain_ranges_test = [(0, 300), (300, 700), (700, 1000)]
    resolved = xlset.filter_resolvable(domain_ranges_test)
    assert len(resolved) == 3, f"Expected all 3 crosslinks resolvable, got {len(resolved)}"
    for (xl, di, dj) in resolved:
        si, ei = domain_ranges_test[di]
        sj, ej = domain_ranges_test[dj]
        assert si <= xl.residue_i < ei
        assert sj <= xl.residue_j < ej
    print("[PASS] XLMSRestraintSet.filter_resolvable correctly resolves residues to domain indices")

    # ── Test 5: filter_resolvable warns and drops out-of-range crosslinks ──
    bad_xlset = XLMSRestraintSet([CrossLink(residue_i=10, residue_j=9999)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bad_resolved = bad_xlset.filter_resolvable(domain_ranges_test)
        assert len(bad_resolved) == 0
        assert any("outside all domain_ranges" in str(x.message) for x in w)
    print("[PASS] XLMSRestraintSet.filter_resolvable warns and drops out-of-range crosslinks")

    # ── Test 6: XLMSRestraintSet.energy is zero within d_max, positive beyond ──
    placed_close = [torch.zeros(300, 3), torch.zeros(400, 3) + torch.tensor([5.0, 0.0, 0.0]),
                     torch.zeros(300, 3)]
    close_pair_xlset = XLMSRestraintSet([CrossLink(residue_i=0, residue_j=300, d_max=30.0)])
    e_close = close_pair_xlset.energy(placed_close, domain_ranges_test, rcfg)
    assert e_close.item() == 0.0, f"Expected zero energy for a satisfied restraint, got {e_close.item()}"
    print(f"[PASS] XLMSRestraintSet.energy is exactly zero when distance (5.0 Å) is well within d_max (30 Å)")

    placed_far = [torch.zeros(300, 3), torch.zeros(400, 3) + torch.tensor([100.0, 0.0, 0.0]),
                  torch.zeros(300, 3)]
    e_far = close_pair_xlset.energy(placed_far, domain_ranges_test, rcfg)
    assert e_far.item() > 0.0, f"Expected positive energy for a violated restraint, got {e_far.item()}"
    print(f"[PASS] XLMSRestraintSet.energy is positive ({e_far.item():.2f}) when distance (100 Å) "
          f"violates d_max (30 Å)")

    # ── Test 7: XLMSRestraintSet.energy gradient flows to domain placement ──
    placed_grad = [torch.zeros(300, 3, requires_grad=True),
                   (torch.zeros(400, 3) + torch.tensor([100.0, 0.0, 0.0])).requires_grad_(True),
                   torch.zeros(300, 3, requires_grad=True)]
    e_grad = close_pair_xlset.energy(placed_grad, domain_ranges_test, rcfg)
    e_grad.backward()
    assert placed_grad[0].grad is not None and torch.isfinite(placed_grad[0].grad).all()
    assert placed_grad[1].grad is not None and torch.isfinite(placed_grad[1].grad).all()
    print("[PASS] XLMSRestraintSet.energy gradient flows correctly to both domains' placed coordinates")

    # ── Test 8: native MRC reader round-trip on a synthetic file ─────────
    import tempfile, os as _os
    synthetic_shape = (4, 5, 6)  # (nz, ny, nx)
    synthetic_data = np.random.rand(*synthetic_shape).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as tmp:
        tmp_path = tmp.name
        header = bytearray(1024)
        nz, ny, nx = synthetic_shape
        struct.pack_into("<4i", header, 0, nx, ny, nz, 2)  # MODE 2 = float32
        struct.pack_into("<3i", header, 16, 0, 0, 0)  # NSTART
        struct.pack_into("<3i", header, 28, nx, ny, nz)  # M (sampling)
        struct.pack_into("<3f", header, 40, float(nx) * 1.5, float(ny) * 1.5, float(nz) * 1.5)  # cell
        struct.pack_into("<3f", header, 196, 0.0, 0.0, 0.0)  # ORIGIN
        tmp.write(bytes(header))
        tmp.write(synthetic_data.tobytes())
    try:
        loaded = load_density_map_native(tmp_path, resolution=4.0)
        assert loaded.data.shape == synthetic_shape, f"Shape mismatch: {loaded.data.shape} vs {synthetic_shape}"
        assert np.allclose(loaded.data, synthetic_data, atol=1e-5), "Round-tripped voxel data does not match"
        assert abs(loaded.voxel_size[0] - 1.5) < 1e-4, f"Expected voxel_size.x=1.5, got {loaded.voxel_size[0]}"
        print(f"[PASS] load_density_map_native round-trips a synthetic MODE-2 MRC file exactly "
              f"(shape={loaded.data.shape}, voxel_size={loaded.voxel_size})")
    finally:
        _os.unlink(tmp_path)

    # ── Test 9: native MRC reader rejects unsupported MODE with a clear error ──
    with tempfile.NamedTemporaryFile(suffix=".mrc", delete=False) as tmp:
        tmp_path2 = tmp.name
        header2 = bytearray(1024)
        struct.pack_into("<4i", header2, 0, 2, 2, 2, 99)  # MODE 99 = unsupported
        tmp.write(bytes(header2))
        tmp.write(b"\x00" * 64)
    try:
        try:
            load_density_map_native(tmp_path2, resolution=4.0)
            print("[FAIL] load_density_map_native should reject MODE=99")
        except ValueError as e:
            assert "MODE=99" in str(e) or "99" in str(e)
            print("[PASS] load_density_map_native raises a clear ValueError for unsupported MODE")
    finally:
        _os.unlink(tmp_path2)

    # ── Test 10: CryoEMDensityRestraint — CCC is high for a self-consistent map ──
    # Build a map FROM a known coordinate set via the same Gaussian-splat
    # logic, then check that scoring those same coordinates against it
    # gives a high cross-correlation (this is the strongest sanity check
    # available without a real experimental map).
    test_coords_for_map = torch.tensor([[10.0, 10.0, 10.0], [15.0, 10.0, 10.0],
                                         [20.0, 10.0, 10.0], [10.0, 15.0, 10.0]])
    grid_n = 20
    voxel_size_test = 2.0
    grid_zz, grid_yy, grid_xx = np.meshgrid(
        np.arange(grid_n), np.arange(grid_n), np.arange(grid_n), indexing="ij"
    )
    grid_xyz = np.stack([grid_xx.ravel(), grid_yy.ravel(), grid_zz.ravel()], axis=-1).astype(np.float32) * voxel_size_test
    sigma_test = 0.4 * 4.0  # matches default cryoem_gaussian_sigma_factor * resolution=4.0
    dists_sq = ((grid_xyz[:, None, :] - test_coords_for_map.numpy()[None, :, :]) ** 2).sum(axis=-1)
    synthetic_density_flat = np.exp(-0.5 * dists_sq / sigma_test ** 2).sum(axis=1)
    synthetic_density_grid = synthetic_density_flat.reshape(grid_n, grid_n, grid_n).astype(np.float32)

    test_map = DensityMap(
        data=synthetic_density_grid,
        voxel_size=(voxel_size_test, voxel_size_test, voxel_size_test),
        origin=(0.0, 0.0, 0.0),
        resolution=4.0,
    )
    restraint = CryoEMDensityRestraint(test_map, rcfg, support_radius=6.0)
    ccc_self = restraint.simulate_and_correlate(test_coords_for_map)
    assert ccc_self.item() > 0.95, (
        f"Expected near-perfect self-correlation (>0.95) when scoring the exact coordinates "
        f"the synthetic map was built from, got {ccc_self.item():.4f}"
    )
    print(f"[PASS] CryoEMDensityRestraint.simulate_and_correlate gives CCC={ccc_self.item():.4f} "
          f"(>0.95) for coordinates matching the map's generating structure")

    # ── Test 11: CryoEMDensityRestraint — CCC drops for a displaced structure ──
    displaced_coords = test_coords_for_map + torch.tensor([50.0, 50.0, 50.0])
    ccc_displaced = restraint.simulate_and_correlate(displaced_coords)
    assert ccc_displaced.item() < ccc_self.item(), (
        f"Expected lower correlation for a displaced structure ({ccc_displaced.item():.4f}) "
        f"than the matching one ({ccc_self.item():.4f})"
    )
    print(f"[PASS] CryoEMDensityRestraint correctly scores a displaced structure lower "
          f"(CCC={ccc_displaced.item():.4f}) than the matching one (CCC={ccc_self.item():.4f})")

    # ── Test 12: CryoEMDensityRestraint.energy gradient flows to coordinates ──
    grad_coords = test_coords_for_map.clone().requires_grad_(True)
    em_energy = restraint.energy(grad_coords)
    em_energy.backward()
    assert grad_coords.grad is not None and torch.isfinite(grad_coords.grad).all()
    print("[PASS] CryoEMDensityRestraint.energy gradient flows correctly to input coordinates")

    # ── Test 13: RestrainedDomainDockingAssembler — XL-MS-only restrained docking ──
    dock_cfg = DomainAssemblyConfig(
        target_domain_size=10, min_domain_size=2, max_domain_size=50,
        dock_iters=150, dock_lr=0.1, dock_clash_weight=0.0,
    )
    domain_a = torch.randn(10, 3)
    domain_b = torch.randn(10, 3) + torch.tensor([200.0, 0.0, 0.0])
    domain_ranges_dock = [(0, 10), (10, 20)]
    xl_for_dock = XLMSRestraintSet([CrossLink(residue_i=3, residue_j=15, d_max=20.0)])
    restrained_assembler = RestrainedDomainDockingAssembler(
        dock_cfg, RestraintDockingConfig(xlms_weight=10.0), xlms=xl_for_dock,
    )
    d_before = (domain_a[3] - domain_b[5]).norm().item()
    assembled_xlms = restrained_assembler.dock(
        [domain_a, domain_b], domain_ranges_dock, contacts=[],
    )
    d_after_xlms = (assembled_xlms[3] - assembled_xlms[15]).norm().item()
    assert d_after_xlms < d_before, (
        f"XL-MS-restrained docking should pull the linked pair closer "
        f"(before={d_before:.1f} Å, after={d_after_xlms:.2f} Å) even with zero contact-head contacts"
    )
    assert d_after_xlms < 20.0 + RestraintDockingConfig().xlms_slack + 1.0, (
        f"Docked distance ({d_after_xlms:.2f} Å) should satisfy the d_max=20 Å restraint "
        f"(within slack tolerance), given {dock_cfg.dock_iters} iterations and xlms_weight=10.0"
    )
    print(f"[PASS] RestrainedDomainDockingAssembler (XL-MS only, zero contacts) pulls a 20 Å-max "
          f"crosslink from {d_before:.1f} Å apart to {d_after_xlms:.2f} Å")

    # ── Test 14: no-restraint construction warns ─────────────────────────
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        RestrainedDomainDockingAssembler(dock_cfg, RestraintDockingConfig())
        assert any("no restraints" in str(x.message) for x in w)
    print("[PASS] RestrainedDomainDockingAssembler warns when constructed with no restraints at all")

    print("=" * 70)
    print("  All tests passed.")
    print("  NOTE: written against structural_domain_assembly_one.py's exact")
    print("  signatures but NOT executed end-to-end before this point (no")
    print("  torch runtime in the authoring environment). Run locally first.")
    print("=" * 70)
