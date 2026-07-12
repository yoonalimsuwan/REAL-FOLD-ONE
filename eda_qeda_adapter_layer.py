# =============================================================================
# EDA / QEDA ADAPTER LAYER — Structural Calculus to Chip Fabrication Bridge
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
# License      : MIT
# Year         : 2026
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
#
# AI Co-Developers (architecture, adapter design, data translation):
#   - Gemini   (Google)     — Adapter architecture design, phase-field to
#                             GDSII topology extraction, SPICE parameter
#                             mapping, QEDA netlist generation structuring.
#   - Claude   (Anthropic)  — Production hardening: real marching-squares /
#                             marching-cubes geometry extraction, spec-
#                             compliant binary GDSII stream writer, per-chip-
#                             type physical parameter models (CMOS digital,
#                             RF/analog, MEMS, photonic, superconducting
#                             qubit), SPICE deck generation, validation,
#                             error handling, and self-test suite.
#
# Description
# -----------
# Bridges continuous physical fields (u-field / sigma-field produced by
# Structural Cahn-Hilliard 3D and the Structural GNO fold surrogates) into
# the discrete artefacts an EDA/QEDA toolchain consumes:
#
#   1. GDSII stream files    — real polygon boundaries (not placeholder
#                               counts), written in spec-compliant binary
#                               GDSII record format. Uses gdspy if it is
#                               installed in the target environment, and
#                               otherwise falls back to a self-contained
#                               minimal GDSII writer (no external dependency
#                               required to ship a real .gds file).
#   2. SPICE netlists (.cir) — classical R/L/C lumped-element decks for
#                               CMOS, RF/analog, and MEMS targets.
#   3. QEDA JSON netlists     — quantum-device descriptions (S-Qubits,
#                               SSC Photonic waveguides) carrying physically
#                               distinct quantities (kinetic inductance,
#                               participation ratio, waveguide effective
#                               index) that a classical SPICE deck cannot
#                               represent.
#
# Supported chip platforms (ChipPlatform enum)
# ---------------------------------------------
#   CMOS_DIGITAL          — standard-cell digital logic
#   RF_ANALOG             — RF/analog/mixed-signal
#   MEMS                  — micro-electromechanical structures
#   PHOTONIC              — silicon-photonic waveguides (SSC Photonic)
#   SUPERCONDUCTING_QUBIT — S-Qubit transmon-class quantum devices
#   CARBON_NANOTUBE       — carbon-based chips (CNT-FET / graphene-class);
#                           quantum-limited contact resistance + series
#                           quantum/oxide gate capacitance, not bulk
#                           resistivity-driven RC like CMOS_DIGITAL
#   IN_MEMORY_COMPUTE     — ReRAM/memristor crossbar compute-in-memory;
#                           conductance-state device with an I-V
#                           nonlinearity exponent for sneak-path
#                           suppression, exported as a behavioral SPICE
#                           source since no passive RLC equivalent exists
#   OVER_THE_AIR_COMPUTE  — AirComp RF front-end (antenna + PA feeding a
#                           shared multiple-access channel); radiation
#                           impedance plus the phase-jitter/PA-distortion
#                           figures that govern analog superposition
#                           fidelity, which a generic RF_ANALOG model does
#                           not capture
#
# Each platform has its own physically-motivated mapping from the
# (u, sigma) structural fields to device parameters; a single linear
# R ~ rho/u heuristic is not adequate across these regimes and was the
# main limitation of the previous draft.
#
# Upstream field contract (unchanged from structural_cahn_hilliard_3d.py /
# structural_gno_fold_v3.py):
#   u_field     : torch.Tensor, shape (Nx, Ny) or (Nx, Ny, Nz), phase order
#                 parameter in [-1, 1] or [0, 1] depending on upstream
#                 convention; this adapter auto-detects range and normalises.
#   sigma_field : torch.Tensor, same shape as u_field (or scalar-broadcast),
#                 structural-stiffness / disorder field, sigma > 0.
# =============================================================================

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import gdspy  # type: ignore
    _HAS_GDSPY = True
except ImportError:
    _HAS_GDSPY = False

try:
    from skimage import measure as _skimage_measure
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# =============================================================================
# 0.  Logging
# =============================================================================

logger = logging.getLogger("QEDA_Adapter")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_ch)
    logger.propagate = False


# =============================================================================
# 1.  Exceptions
# =============================================================================

class QEDAAdapterError(Exception):
    """Base error for the EDA/QEDA adapter layer."""


class FieldValidationError(QEDAAdapterError):
    """Raised when an input field fails shape, dtype, or range validation."""


class GeometryExtractionError(QEDAAdapterError):
    """Raised when contour/iso-surface extraction yields no usable geometry."""


class BackendUnavailableError(QEDAAdapterError):
    """Raised when a requested optional backend (gdspy, skimage) is missing."""


# =============================================================================
# 2.  Chip platform definitions
# =============================================================================

class ChipPlatform(str, enum.Enum):
    """
    Target fabrication platform. Each platform routes the (u, sigma) fields
    through a distinct physical parameter model, because the dominant
    device physics differs qualitatively between them (e.g. kinetic
    inductance dominates superconducting qubits; classical RC parasitics
    dominate CMOS digital; modal effective index dominates photonics).
    """
    CMOS_DIGITAL = "cmos_digital"
    RF_ANALOG = "rf_analog"
    MEMS = "mems"
    PHOTONIC = "photonic"
    SUPERCONDUCTING_QUBIT = "superconducting_qubit"
    CARBON_NANOTUBE = "carbon_nanotube"
    IN_MEMORY_COMPUTE = "in_memory_compute"
    OVER_THE_AIR_COMPUTE = "over_the_air_compute"


@dataclasses.dataclass(frozen=True)
class MaterialParameters:
    """
    Physical constants for a fabrication process node / material stack.
    Defaults correspond to a generic Cu/SiO2 back-end-of-line stack at
    room temperature; override per real PDK (process design kit) values
    when available.
    """
    resistivity_ohm_m: float = 1.68e-8       # Cu bulk resistivity
    relative_permittivity: float = 3.9       # SiO2
    sheet_thickness_m: float = 1e-7          # 100 nm metal thickness
    kinetic_inductance_per_square_h: float = 2.0e-10
    # ^ typical thin-film Al kinetic inductance L_k per square, order of
    #   magnitude for transmon-class superconducting qubit traces.
    waveguide_core_index: float = 3.45       # silicon core, ~1550 nm
    waveguide_clad_index: float = 1.44       # SiO2 cladding

    # Carbon nanotube / graphene FET parameters
    cnt_diffusive_resistance_per_sigma_ohm: float = 5.0e3
    # ^ representative order-of-magnitude diffusive (defect/phonon
    #   scattering) channel resistance contributed per unit of structural
    #   disorder sigma, for short-channel CNT-FETs where total device
    #   resistance is contact-resistance-dominated (kOhm range) rather
    #   than length-scaling bulk resistivity as in classical CMOS.
    cnt_quantum_capacitance_f: float = 2.0e-15
    # ^ representative per-device quantum capacitance (order of a few
    #   aF-fF for a single CNT channel); combines in series with the
    #   electrostatic oxide capacitance, the dominant gate-capacitance
    #   physics distinguishing 1D/2D carbon channels from bulk silicon.
    gate_oxide_thickness_m: float = 2.0e-9   # 2 nm thin gate oxide

    # ReRAM / memristor crossbar (in-memory compute) parameters
    reram_g_off_s: float = 1.0e-6            # HRS conductance, ~1 MOhm
    reram_g_on_s: float = 1.0e-4             # LRS conductance, ~10 kOhm

    # Over-the-air computation (AirComp RF front-end) parameters
    ota_reference_impedance_ohm: float = 50.0  # standard RF system impedance


_EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m


# =============================================================================
# 3.  Field validation utilities
# =============================================================================

def _validate_field(
    field: torch.Tensor,
    name: str,
    expect_positive: bool = False,
    allow_scalar_broadcast: bool = False,
) -> torch.Tensor:
    """
    Validate a structural field tensor and return a detached float64 CPU
    copy ready for numpy interop. Raises FieldValidationError on failure
    rather than silently propagating NaNs/shape mismatches into geometry
    or netlist generation.
    """
    if not isinstance(field, torch.Tensor):
        raise FieldValidationError(f"{name} must be a torch.Tensor, got {type(field)!r}")
    if field.numel() == 0:
        raise FieldValidationError(f"{name} is empty (numel=0)")
    if field.dim() not in (2, 3) and not (allow_scalar_broadcast and field.numel() == 1):
        raise FieldValidationError(
            f"{name} must be 2D (Nx,Ny) or 3D (Nx,Ny,Nz); got shape {tuple(field.shape)}"
        )
    f = field.detach().to(dtype=torch.float64, device="cpu")
    if torch.isnan(f).any():
        raise FieldValidationError(f"{name} contains NaN values")
    if torch.isinf(f).any():
        raise FieldValidationError(f"{name} contains Inf values")
    if expect_positive and (f <= 0).any():
        n_bad = int((f <= 0).sum().item())
        raise FieldValidationError(
            f"{name} expected strictly positive values; {n_bad} non-positive "
            f"entries found (min={f.min().item():.6g})"
        )
    return f


def _broadcast_sigma(u_field: torch.Tensor, sigma_field: torch.Tensor, name: str = "sigma_field") -> torch.Tensor:
    """Broadcast a scalar or under-ranked sigma_field to u_field's shape."""
    if sigma_field.shape == u_field.shape:
        return sigma_field
    try:
        return sigma_field.expand_as(u_field).clone()
    except (RuntimeError, ValueError) as exc:
        raise FieldValidationError(
            f"{name} with shape {tuple(sigma_field.shape)} cannot be broadcast "
            f"to u_field shape {tuple(u_field.shape)}"
        ) from exc


def _normalise_u(u: np.ndarray) -> np.ndarray:
    """
    Map u into [0, 1] regardless of whether the upstream convention used
    [-1, 1] (standard Cahn-Hilliard double-well) or [0, 1]. Detection is by
    observed range, not a hardcoded assumption, since different upstream
    modules (structural_cahn_hilliard_3d.py vs structural_gno_fold_v3.py)
    have used both conventions historically.
    """
    u_min, u_max = float(u.min()), float(u.max())
    if u_max - u_min < 1e-12:
        # Degenerate constant field; cannot infer convention, treat as already
        # in [0, 1] but warn since downstream thresholding will be meaningless.
        logger.warning(
            "u_field is numerically constant (min=max=%.6g); geometry "
            "extraction will yield no interfaces.", u_min
        )
        return np.clip(u, 0.0, 1.0)
    if u_min < -1e-6:
        # Looks like the [-1, 1] double-well convention.
        return (u - u_min) / (u_max - u_min)
    return np.clip(u, 0.0, 1.0)


# =============================================================================
# 4.  Geometry extraction (real marching squares / marching cubes)
# =============================================================================

@dataclasses.dataclass
class ExtractedGeometry2D:
    polygons: List[np.ndarray]   # each (K, 2) array of (row, col) pixel coords
    resolution_nm: float
    threshold: float


@dataclasses.dataclass
class ExtractedGeometry3D:
    vertices: np.ndarray          # (V, 3)
    faces: np.ndarray             # (F, 3) triangle indices
    resolution_nm: float
    threshold: float
    layer_slices: Dict[int, List[np.ndarray]]  # z-index -> polygons at that slice


def extract_contours_2d(
    u_field: torch.Tensor,
    threshold: float = 0.5,
    resolution_nm: float = 1.0,
    min_polygon_area_px: float = 4.0,
) -> ExtractedGeometry2D:
    """
    Extract real iso-valued contours from a 2D phase field using marching
    squares (skimage.measure.find_contours). Replaces the previous mock
    implementation, which only counted thresholded pixels and never
    produced actual polygon boundaries.

    Parameters
    ----------
    u_field : 2D torch.Tensor, phase-field order parameter.
    threshold : iso-value defining the material/void boundary.
    resolution_nm : physical size of one grid cell, in nanometres.
    min_polygon_area_px : polygons whose extent metric (closed-loop area, or
        bounding-box area / path length for open boundary-clipped contours,
        see _shoelace_area) falls below this value are discarded as
        numerical noise (e.g. single-pixel speckle from a diffuse interface).

    Returns
    -------
    ExtractedGeometry2D with one polygon per closed (or open, boundary-
    clipped) contour found at the given threshold.
    """
    if not _HAS_SKIMAGE:
        raise BackendUnavailableError(
            "scikit-image is required for 2D contour extraction "
            "(`pip install scikit-image`)."
        )
    if u_field.dim() != 2:
        raise FieldValidationError(
            f"extract_contours_2d requires a 2D field, got shape {tuple(u_field.shape)}"
        )
    u_np = _validate_field(u_field, "u_field").numpy()
    u_norm = _normalise_u(u_np)

    raw_contours = _skimage_measure.find_contours(u_norm, level=threshold)
    polygons: List[np.ndarray] = []
    for c in raw_contours:
        area = _shoelace_area(c)
        if area >= min_polygon_area_px:
            polygons.append(c)

    if not polygons:
        raise GeometryExtractionError(
            f"No polygons with area >= {min_polygon_area_px} px^2 found at "
            f"threshold={threshold} (field range after normalisation: "
            f"[{u_norm.min():.4f}, {u_norm.max():.4f}]). Try a different "
            f"threshold or check that the field actually contains an interface."
        )

    logger.info(
        "Extracted %d real polygon(s) via marching squares at threshold=%.4f "
        "(%d raw contours, %d discarded as sub-resolution noise).",
        len(polygons), threshold, len(raw_contours), len(raw_contours) - len(polygons),
    )
    return ExtractedGeometry2D(polygons=polygons, resolution_nm=resolution_nm, threshold=threshold)


def extract_isosurface_3d(
    u_field: torch.Tensor,
    threshold: float = 0.5,
    resolution_nm: float = 1.0,
    slice_for_layers: bool = True,
) -> ExtractedGeometry3D:
    """
    Extract a real triangulated iso-surface from a 3D phase field using
    marching cubes (skimage.measure.marching_cubes), and optionally also
    slice the volume into per-z-layer 2D polygons for lithographic mask
    generation (GDSII is fundamentally a layered 2D format, so a 3D
    structure must be decomposed into layers before GDSII export).
    """
    if not _HAS_SKIMAGE:
        raise BackendUnavailableError(
            "scikit-image is required for 3D iso-surface extraction "
            "(`pip install scikit-image`)."
        )
    if u_field.dim() != 3:
        raise FieldValidationError(
            f"extract_isosurface_3d requires a 3D field, got shape {tuple(u_field.shape)}"
        )
    u_np = _validate_field(u_field, "u_field").numpy().astype(np.float32)
    u_norm = _normalise_u(u_np)

    if not (u_norm.min() < threshold < u_norm.max()):
        raise GeometryExtractionError(
            f"threshold={threshold} is outside the field range "
            f"[{u_norm.min():.4f}, {u_norm.max():.4f}]; marching cubes "
            f"requires the iso-value to lie strictly inside the data range."
        )

    verts, faces, _normals, _values = _skimage_measure.marching_cubes(
        u_norm, level=threshold, spacing=(1.0, 1.0, 1.0)
    )

    layer_slices: Dict[int, List[np.ndarray]] = {}
    if slice_for_layers:
        nz = u_norm.shape[2]
        for z in range(nz):
            try:
                contours = _skimage_measure.find_contours(u_norm[:, :, z], level=threshold)
            except Exception:
                contours = []
            kept = [c for c in contours if _shoelace_area(c) >= 4.0]
            if kept:
                layer_slices[z] = kept

    logger.info(
        "Extracted real 3D iso-surface via marching cubes: %d vertices, "
        "%d triangular faces, %d non-empty z-layers for lithographic slicing.",
        len(verts), len(faces), len(layer_slices),
    )
    return ExtractedGeometry3D(
        vertices=verts, faces=faces, resolution_nm=resolution_nm,
        threshold=threshold, layer_slices=layer_slices,
    )


def _shoelace_area(polygon: np.ndarray) -> float:
    """
    Shoelace formula for polygon area. For a closed loop this is the
    enclosed area. For an open contour that is clipped by the field
    boundary (e.g. a straight interface running edge-to-edge, which is
    common and physically valid — the boundary itself closes the region)
    the raw shoelace sum on the open path is near zero by construction,
    so this falls back to bounding-box area as a non-degeneracy proxy:
    it is nonzero whenever the contour spans more than a single point,
    which is all this filter needs to distinguish real geometry from
    single-pixel numerical speckle.
    """
    if polygon.shape[0] < 2:
        return 0.0
    x = polygon[:, 1]
    y = polygon[:, 0]
    if polygon.shape[0] >= 3:
        closed_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if closed_area > 1e-9:
            return closed_area
    bbox_area = (x.max() - x.min()) * (y.max() - y.min())
    if bbox_area > 1e-9:
        return bbox_area
    # Degenerate to path length for perfectly axis-aligned straight lines,
    # where the bounding box collapses to zero width in one dimension.
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


# =============================================================================
# 5.  GDSII export — real binary stream writer (gdspy if available, else
#     a self-contained minimal-but-spec-compliant fallback writer)
# =============================================================================

# GDSII record type codes (subset needed for header + boundary + structure).
_GDS_HEADER = 0x0002
_GDS_BGNLIB = 0x0102
_GDS_LIBNAME = 0x0206
_GDS_UNITS = 0x0305
_GDS_ENDLIB = 0x0400
_GDS_BGNSTR = 0x0502
_GDS_STRNAME = 0x0606
_GDS_ENDSTR = 0x0700
_GDS_BOUNDARY = 0x0800
_GDS_LAYER = 0x0D02
_GDS_DATATYPE = 0x0E02
_GDS_XY = 0x1003
_GDS_ENDEL = 0x1100


def _gds_pack_record(rec_type: int, payload: bytes) -> bytes:
    length = len(payload) + 4
    if length % 2 != 0:
        payload += b"\x00"
        length += 1
    return struct.pack(">HH", length, rec_type) + payload


def _gds_now() -> Tuple[int, int, int, int, int, int]:
    t = time.localtime()
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def _gds_real8(value: float) -> bytes:
    """
    Encode a float as GDSII 8-byte 'real' (excess-64 hex floating point),
    required for the UNITS record. Implements the format directly since
    struct/numpy have no native excess-64 float type.
    """
    if value == 0.0:
        return b"\x00" * 8
    sign = 0
    if value < 0:
        sign = 0x80
        value = -value
    exponent = 64
    while value >= 1.0:
        value /= 16.0
        exponent += 1
    while value < 1.0 / 16.0:
        value *= 16.0
        exponent -= 1
    mantissa = int(value * (1 << 56))
    byte0 = sign | (exponent & 0x7F)
    out = bytearray(8)
    out[0] = byte0
    for i in range(7):
        out[7 - i] = mantissa & 0xFF
        mantissa >>= 8
    return bytes(out)


def _write_gdsii_minimal(
    layer_polygons: Dict[int, List[np.ndarray]],
    filename: Union[str, Path],
    resolution_nm: float,
    structure_name: str = "TOPCELL",
    datatype: int = 0,
) -> int:
    """
    Self-contained minimal GDSII stream writer used when gdspy is not
    installed. Produces a real, spec-compliant .gds binary (readable by
    KLayout / Cadence / any GDSII-II reader) containing one BOUNDARY
    element per extracted polygon, placed on its corresponding layer
    number. This is not a mock: the output is a valid GDSII Stream
    Format file per the Calma GDSII specification.

    Returns the total number of BOUNDARY elements written.
    """
    user_unit_m = resolution_nm * 1e-9   # 1 database unit = 1 pixel = resolution_nm
    db_unit_in_user_unit = 1.0           # 1 DB unit == 1 user unit (no sub-grid)
    db_unit_in_m = user_unit_m

    out = bytearray()
    out += _gds_pack_record(_GDS_HEADER, struct.pack(">H", 600))
    y, mo, d, h, mi, s = _gds_now()
    out += _gds_pack_record(_GDS_BGNLIB, struct.pack(">12H", y, mo, d, h, mi, s, y, mo, d, h, mi, s))
    libname = b"QEDA_ADAPTER_LIB\x00"
    out += _gds_pack_record(_GDS_LIBNAME, libname)
    out += _gds_pack_record(
        _GDS_UNITS,
        _gds_real8(db_unit_in_user_unit) + _gds_real8(db_unit_in_m),
    )

    n_elements = 0
    out += _gds_pack_record(_GDS_BGNSTR, struct.pack(">12H", y, mo, d, h, mi, s, y, mo, d, h, mi, s))
    sname = structure_name.encode("ascii")[:32]
    if len(sname) % 2 != 0:
        sname += b"\x00"
    out += _gds_pack_record(_GDS_STRNAME, sname)

    for layer_num, polygons in sorted(layer_polygons.items()):
        for poly in polygons:
            if poly.shape[0] < 3:
                continue
            pts = np.round(poly).astype(np.int32)
            if not np.array_equal(pts[0], pts[-1]):
                pts = np.vstack([pts, pts[0]])
            xy_payload = bytearray()
            for row, col in pts:
                # GDSII XY is (x, y); polygon rows/cols come from
                # (row=y-index, col=x-index) per skimage convention.
                xy_payload += struct.pack(">ii", int(col), int(row))
            out += _gds_pack_record(_GDS_BOUNDARY, b"")
            out += _gds_pack_record(_GDS_LAYER, struct.pack(">H", layer_num))
            out += _gds_pack_record(_GDS_DATATYPE, struct.pack(">H", datatype))
            out += _gds_pack_record(_GDS_XY, bytes(xy_payload))
            out += _gds_pack_record(_GDS_ENDEL, b"")
            n_elements += 1

    out += _gds_pack_record(_GDS_ENDSTR, b"")
    out += _gds_pack_record(_GDS_ENDLIB, b"")

    Path(filename).write_bytes(bytes(out))
    return n_elements


def _write_gdsii_gdspy(
    layer_polygons: Dict[int, List[np.ndarray]],
    filename: Union[str, Path],
    resolution_nm: float,
    structure_name: str = "TOPCELL",
    datatype: int = 0,
) -> int:
    """GDSII export path using gdspy when it is available in the environment."""
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(structure_name)
    n_elements = 0
    scale_um = resolution_nm * 1e-3  # nm -> um, since gdspy works in 'unit' (um here)
    for layer_num, polygons in sorted(layer_polygons.items()):
        for poly in polygons:
            if poly.shape[0] < 3:
                continue
            pts = [(float(c) * scale_um, float(r) * scale_um) for r, c in poly]
            cell.add(gdspy.Polygon(pts, layer=layer_num, datatype=datatype))
            n_elements += 1
    lib.write_gds(str(filename))
    return n_elements


# =============================================================================
# 6.  Per-platform physical parameter mapping
# =============================================================================

class StructuralFieldToDeviceMapper:
    """
    Maps (u, sigma) structural fields into platform-specific device
    parameters. Each platform branch encodes the dominant physical
    mechanism for that regime rather than reusing a single classical-RC
    heuristic everywhere, since e.g. superconducting qubits are dominated
    by kinetic inductance and photonic waveguides by modal effective
    index, not by Ohmic resistance.
    """

    def __init__(self, material: Optional[MaterialParameters] = None):
        self.mat = material or MaterialParameters()

    # -- shared helper -----------------------------------------------------
    def _u_sigma_numpy(
        self, u_field: torch.Tensor, sigma_field: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        u_v = _validate_field(u_field, "u_field")
        sigma_v = _validate_field(
            _broadcast_sigma(u_field, sigma_field), "sigma_field", expect_positive=True
        )
        u_np = _normalise_u(u_v.numpy())
        sigma_np = sigma_v.numpy()
        return u_np, sigma_np

    # -- CMOS digital --------------------------------------------------------
    def map_cmos_digital(self, u_field: torch.Tensor, sigma_field: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Classical lumped RC interconnect model.
        R ~ rho * L / (A * conductive_fraction)   (sheet-resistance form, per unit cell)
        C ~ eps0 * eps_r * A / d                   (parallel-plate parasitic, per unit cell)
        u (normalised, high = conductive metal phase), sigma modulates grain/defect scattering.
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        u_safe = np.clip(u_np, 1e-4, 1.0)
        sheet_resistance = (self.mat.resistivity_ohm_m / self.mat.sheet_thickness_m) / u_safe
        resistance_grid = sheet_resistance * sigma_np  # sigma = scattering / defect penalty
        capacitance_grid = (
            _EPS0 * self.mat.relative_permittivity * u_np / np.clip(sigma_np, 1e-6, None)
        )
        inductance_grid = 5e-10 * resistance_grid / np.clip(resistance_grid.mean(), 1e-12, None)
        return {"R_grid": resistance_grid, "C_grid": capacitance_grid, "L_grid": inductance_grid}

    # -- RF / analog -----------------------------------------------------
    def map_rf_analog(self, u_field: torch.Tensor, sigma_field: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Same classical R/C backbone as CMOS digital but retains a
        frequency-relevant series inductance term sized from trace
        geometry (sigma here interpreted as a normalised trace-width
        proxy) instead of being scaled off resistance.
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        u_safe = np.clip(u_np, 1e-4, 1.0)
        resistance_grid = (self.mat.resistivity_ohm_m / self.mat.sheet_thickness_m) / u_safe * sigma_np
        capacitance_grid = _EPS0 * self.mat.relative_permittivity * u_np / np.clip(sigma_np, 1e-6, None)
        mu0 = 4e-7 * np.pi
        trace_width_proxy = np.clip(sigma_np, 1e-3, None)
        inductance_grid = (mu0 / (2.0 * np.pi)) * np.log1p(1.0 / trace_width_proxy)
        return {"R_grid": resistance_grid, "C_grid": capacitance_grid, "L_grid": inductance_grid}

    # -- MEMS -----------------------------------------------------------
    def map_mems(self, u_field: torch.Tensor, sigma_field: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        MEMS structures are mechanical first; sigma is reinterpreted as a
        local stiffness proxy (Pa-scale) rather than an electrical
        scattering term, and an effective spring constant is reported
        alongside parasitic electrical parameters for actuation
        electrodes (capacitive actuation is the dominant electrical
        coupling mechanism in MEMS, not resistive transport).
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        u_safe = np.clip(u_np, 1e-4, 1.0)
        stiffness_grid = sigma_np * 1e3   # Pa-scale effective stiffness proxy
        gap_proxy = np.clip(1.0 - u_safe, 1e-3, None) * self.mat.sheet_thickness_m * 10.0
        capacitance_grid = _EPS0 * u_safe / gap_proxy  # parallel-plate actuation capacitance
        resistance_grid = (self.mat.resistivity_ohm_m / self.mat.sheet_thickness_m) / u_safe
        return {
            "R_grid": resistance_grid,
            "C_grid": capacitance_grid,
            "K_grid": stiffness_grid,
            "L_grid": resistance_grid * 1e-12,
        }

    # -- Photonic ----------------------------------------------------
    def map_photonic(self, u_field: torch.Tensor, sigma_field: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Photonic waveguides are governed by modal effective index and
        propagation loss, not by R/L/C lumped elements. u interpolates
        between cladding and core index; sigma is interpreted as a
        scattering-loss proxy (disorder -> radiative loss).
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        n_eff_grid = (
            self.mat.waveguide_clad_index
            + u_np * (self.mat.waveguide_core_index - self.mat.waveguide_clad_index)
        )
        # Propagation loss (dB/cm) grows with structural disorder (sigma).
        loss_db_per_cm_grid = 0.1 + 2.0 * np.clip(sigma_np - 1.0, 0.0, None)
        return {"n_eff_grid": n_eff_grid, "loss_db_per_cm_grid": loss_db_per_cm_grid}

    # -- Superconducting qubit (S-Qubit) --------------------------------
    def map_superconducting_qubit(
        self, u_field: torch.Tensor, sigma_field: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Transmon-class superconducting qubits are dominated by kinetic
        inductance (L_k) and shunt capacitance, not Ohmic resistance
        (the device is operated at mK temperatures where DC resistance of
        the superconducting film is ~0). sigma is interpreted as a
        normalised quasiparticle-density / two-level-system (TLS) defect
        proxy, since structural disorder is the principal microscopic
        loss channel in superconducting qubits and directly sets the
        participation-ratio-weighted loss tangent.
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        u_safe = np.clip(u_np, 1e-4, 1.0)
        kinetic_inductance_grid = self.mat.kinetic_inductance_per_square_h / u_safe
        capacitance_grid = _EPS0 * self.mat.relative_permittivity * u_safe
        # Loss tangent grows with disorder (sigma); this sets the qubit's
        # intrinsic quality factor Q = 1 / loss_tangent.
        loss_tangent_grid = 1e-6 * np.clip(sigma_np, 1e-6, None)
        quality_factor_grid = 1.0 / np.clip(loss_tangent_grid, 1e-12, None)
        return {
            "Lk_grid": kinetic_inductance_grid,
            "C_grid": capacitance_grid,
            "loss_tangent_grid": loss_tangent_grid,
            "Q_grid": quality_factor_grid,
        }

    # -- Carbon nanotube / graphene FET (carbon-based chips) -------------
    def map_carbon_nanotube(
        self, u_field: torch.Tensor, sigma_field: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Carbon-based channels (CNT-FET, graphene-FET-class devices) are
        dominated by two mechanisms that have no equivalent in bulk-
        resistivity CMOS:

          1. Quantum-limited contact resistance. A single ballistic 1D
             conduction channel has a minimum resistance of
             R_q = h / (2 e^2) ~= 12.9 kOhm regardless of channel length;
             real CNT-FET contact resistance sits at this quantum limit
             times a degradation factor when the metal-CNT (Schottky
             barrier) contact is imperfect or the nanotube network is not
             fully percolated. u (normalised network coverage / contact
             quality, high = good contact) sets that degradation factor.
          2. Series quantum + electrostatic gate capacitance. Because the
             channel's density of states is finite (1D/2D, not a 3D bulk
             reservoir), the quantum capacitance C_q is *in series* with
             the classical oxide capacitance C_ox: C_total =
             (1/C_q + 1/C_ox)^-1. In bulk silicon C_q >> C_ox so this
             reduces to C_ox alone; in carbon channels the two are
             comparable and C_q becomes the limiting term, which a model
             that only computes C_ox (as CMOS_DIGITAL does) would miss.

        sigma is interpreted as a diffusive scattering (defect/phonon)
        density that adds a length-independent-in-this-model channel
        resistance on top of the quantum-limited contact resistance.
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        u_safe = np.clip(u_np, 1e-4, 1.0)

        h = 6.62607015e-34
        e = 1.602176634e-19
        r_quantum = 1.0 / (2.0 * e ** 2 / h)  # ~12906.4 Ohm per channel

        r_contact_grid = r_quantum / u_safe
        r_channel_grid = self.mat.cnt_diffusive_resistance_per_sigma_ohm * sigma_np
        resistance_grid = r_contact_grid + r_channel_grid

        c_quantum_grid = self.mat.cnt_quantum_capacitance_f * u_safe
        c_ox = _EPS0 * self.mat.relative_permittivity / self.mat.gate_oxide_thickness_m
        c_ox_grid = np.full_like(u_safe, c_ox)
        capacitance_grid = 1.0 / (1.0 / np.clip(c_quantum_grid, 1e-30, None) + 1.0 / c_ox_grid)

        return {
            "R_grid": resistance_grid,
            "C_grid": capacitance_grid,
            "Rcontact_grid": r_contact_grid,
            "Rchannel_grid": r_channel_grid,
            "Cq_grid": c_quantum_grid,
            "Cox_grid": c_ox_grid,
        }

    # -- ReRAM / memristor crossbar (in-memory compute) -------------------
    def map_in_memory_compute(
        self, u_field: torch.Tensor, sigma_field: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Compute-in-memory devices (ReRAM / memristor crossbars) store and
        compute with a conductance state directly — there is no separate
        passive R/L/C equivalent the way there is for an interconnect.
        u is interpreted as the filament/conductive-bridge formation
        state, interpolating linearly between the high-resistance state
        (HRS, u=0) and low-resistance state (LRS, u=1) device
        conductances. sigma sets the I-V nonlinearity exponent: real
        crossbar cells are deliberately operated with a nonlinear,
        self-rectifying I-V (I ~ V^k, k>1) to suppress sneak-path
        current through unselected cells in the array, and that
        nonlinearity strengthens with structural disorder in this model.
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        u_safe = np.clip(u_np, 0.0, 1.0)
        conductance_grid = self.mat.reram_g_off_s + u_safe * (
            self.mat.reram_g_on_s - self.mat.reram_g_off_s
        )
        nonlinearity_grid = 1.0 + sigma_np  # k=1 would be ohmic; sigma>0 always (validated)
        on_off_ratio_grid = conductance_grid / self.mat.reram_g_off_s
        return {
            "G_grid": conductance_grid,
            "nonlinearity_grid": nonlinearity_grid,
            "on_off_ratio_grid": on_off_ratio_grid,
        }

    # -- Over-the-air computation (AirComp RF front-end) ------------------
    def map_over_the_air_compute(
        self, u_field: torch.Tensor, sigma_field: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Over-the-air computation exploits the analog superposition of a
        shared wireless multiple-access channel, so the relevant device
        is the RF front-end (antenna + matching + power amplifier) of
        each transmitting node, not a digital logic block. Two distinct
        physical concerns matter here, neither of which RF_ANALOG's
        generic interconnect-parasitic model captures:

          1. Radiation impedance. u is interpreted as a normalised
             electrical-length proxy (0 = electrically short, 1 =
             half-wave resonant). Radiation resistance grows
             quadratically with electrical length in the short-dipole
             limit and reaches the classical half-wave dipole value
             (~73 Ohm) at resonance; this is a simplified but correctly-
             directed proxy, not a full method-of-moments solve.
             Reactance is capacitive (negative) below resonance and
             vanishes at resonance in this model.
          2. Aggregation fidelity. AirComp's computation accuracy depends
             on how coherently the distributed analog waveforms combine
             at the receiver, which is corrupted by (a) phase/timing
             misalignment across nodes and (b) power-amplifier
             nonlinearity distorting the analog values before they are
             summed by the channel. sigma is interpreted as a structural-
             disorder proxy for both impairments.

        Mismatch loss against the standard 50 Ohm reference impedance is
        reported via the standard reflection-coefficient formula, since
        impedance mismatch directly reduces radiated (useful) power.
        """
        u_np, sigma_np = self._u_sigma_numpy(u_field, sigma_field)
        u_safe = np.clip(u_np, 1e-4, 1.0)

        r_rad_grid = 73.0 * u_safe ** 2
        x_rad_grid = -200.0 * (1.0 - u_safe)

        z0 = self.mat.ota_reference_impedance_ohm
        z = r_rad_grid + 1j * x_rad_grid
        gamma = (z - z0) / (z + z0)
        gamma_mag_sq = np.clip(np.abs(gamma) ** 2, 0.0, 1.0 - 1e-12)
        mismatch_loss_db_grid = -10.0 * np.log10(1.0 - gamma_mag_sq)

        # AirComp-specific aggregation impairments, driven by disorder.
        phase_jitter_grid = 0.05 * sigma_np  # radians, synchronisation error proxy
        pa_distortion_grid = 1.0 - np.exp(-sigma_np)  # saturating distortion index in [0,1)

        return {
            "R_grid": r_rad_grid,
            "X_grid": x_rad_grid,
            "mismatch_loss_db_grid": mismatch_loss_db_grid,
            "phase_jitter_grid": phase_jitter_grid,
            "pa_distortion_grid": pa_distortion_grid,
        }

    def map(
        self, platform: ChipPlatform, u_field: torch.Tensor, sigma_field: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        dispatch = {
            ChipPlatform.CMOS_DIGITAL: self.map_cmos_digital,
            ChipPlatform.RF_ANALOG: self.map_rf_analog,
            ChipPlatform.MEMS: self.map_mems,
            ChipPlatform.PHOTONIC: self.map_photonic,
            ChipPlatform.SUPERCONDUCTING_QUBIT: self.map_superconducting_qubit,
            ChipPlatform.CARBON_NANOTUBE: self.map_carbon_nanotube,
            ChipPlatform.IN_MEMORY_COMPUTE: self.map_in_memory_compute,
            ChipPlatform.OVER_THE_AIR_COMPUTE: self.map_over_the_air_compute,
        }
        if platform not in dispatch:
            raise QEDAAdapterError(f"Unsupported platform: {platform!r}")
        logger.info("Mapping structural fields -> device parameters for platform=%s", platform.value)
        return dispatch[platform](u_field, sigma_field)


# =============================================================================
# 7.  Layout export orchestration (GDSII)
# =============================================================================

class QEDALayoutExporter:
    """
    Converts extracted real geometry (from skimage marching squares /
    marching cubes) into a GDSII layout file, and produces SPICE / QEDA
    netlists from per-platform device parameters.
    """

    def __init__(self, resolution_nm: float = 1.0, default_layer: int = 1):
        if resolution_nm <= 0:
            raise ValueError(f"resolution_nm must be positive, got {resolution_nm}")
        self.resolution_nm = resolution_nm
        self.default_layer = default_layer

    def export_gdsii_2d(
        self,
        u_field: torch.Tensor,
        threshold: float,
        filename: Union[str, Path],
        layer: Optional[int] = None,
        structure_name: str = "TOPCELL",
    ) -> int:
        """
        Extract real polygons from a 2D field and write a spec-compliant
        binary GDSII file. Returns the number of BOUNDARY elements written.
        """
        geom = extract_contours_2d(u_field, threshold=threshold, resolution_nm=self.resolution_nm)
        layer_num = layer if layer is not None else self.default_layer
        layer_polygons = {layer_num: geom.polygons}
        return self._write(layer_polygons, filename, structure_name)

    def export_gdsii_3d_layered(
        self,
        u_field: torch.Tensor,
        threshold: float,
        filename: Union[str, Path],
        structure_name: str = "TOPCELL",
        z_to_layer_offset: int = 1,
    ) -> int:
        """
        Extract a real 3D iso-surface, decompose it into per-z lithographic
        mask layers, and write a multi-layer GDSII file (one GDSII layer
        number per z-slice, offset by z_to_layer_offset so layer numbering
        starts at a non-zero value).
        """
        geom = extract_isosurface_3d(u_field, threshold=threshold, resolution_nm=self.resolution_nm)
        if not geom.layer_slices:
            raise GeometryExtractionError(
                "3D iso-surface extraction produced no per-layer 2D slices "
                "suitable for GDSII export; the volume may be uniform along z."
            )
        layer_polygons = {
            z + z_to_layer_offset: polys for z, polys in geom.layer_slices.items()
        }
        return self._write(layer_polygons, filename, structure_name)

    def _write(
        self, layer_polygons: Dict[int, List[np.ndarray]], filename: Union[str, Path], structure_name: str
    ) -> int:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_GDSPY:
            logger.info("Writing GDSII via gdspy backend -> %s", filename)
            n = _write_gdsii_gdspy(layer_polygons, filename, self.resolution_nm, structure_name)
        else:
            logger.info(
                "gdspy not installed; writing GDSII via built-in minimal "
                "spec-compliant writer -> %s", filename
            )
            n = _write_gdsii_minimal(layer_polygons, filename, self.resolution_nm, structure_name)
        logger.info("[GDSII] Wrote %d BOUNDARY element(s) across %d layer(s) to %s",
                    n, len(layer_polygons), filename)
        return n


# =============================================================================
# 8.  Netlist export (SPICE for classical platforms, QEDA JSON for quantum)
# =============================================================================

class NetlistExporter:
    """Generates SPICE decks and QEDA JSON netlists from device parameters."""

    @staticmethod
    def export_spice(
        device_params: Dict[str, np.ndarray],
        filename: Union[str, Path],
        platform: ChipPlatform,
        title: str = "QEDA Adapter Generated Netlist",
        freq_hz: float = 2.4e9,
    ) -> Path:
        """
        Write a real SPICE deck (.cir) for platforms that have a valid
        lumped-element or behavioral-source SPICE representation
        (CMOS_DIGITAL, RF_ANALOG, MEMS, CARBON_NANOTUBE,
        IN_MEMORY_COMPUTE, OVER_THE_AIR_COMPUTE). Aggregates the per-cell
        parameter grids into a single representative section per node,
        which is the standard reduction used when handing a continuum
        field off to a netlist-based circuit simulator.

        freq_hz : reference frequency used only to convert a reported
            reactance (X_grid, Ohm) into an equivalent inductance or
            capacitance for the deck (L = X/omega if X>0 inductive,
            C = -1/(omega*X) if X<0 capacitive); irrelevant for platforms
            that don't report X_grid. Default 2.4 GHz (ISM band, a
            common AirComp/IoT operating frequency).
        """
        if platform == ChipPlatform.PHOTONIC:
            raise QEDAAdapterError(
                "PHOTONIC platform parameters (n_eff, loss) are not SPICE "
                "lumped elements; use export_qeda_json instead."
            )
        if platform == ChipPlatform.SUPERCONDUCTING_QUBIT:
            raise QEDAAdapterError(
                "SUPERCONDUCTING_QUBIT platform parameters (kinetic "
                "inductance, loss tangent) require a quantum netlist; "
                "use export_qeda_json instead."
            )

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"* {title}", f"* Platform: {platform.value}", "* Auto-generated by QEDA Adapter Layer", ""]
        node = 1
        if "R_grid" in device_params:
            r_val = float(np.mean(device_params["R_grid"]))
            lines.append(f"R1 n{node} 0 {r_val:.6e}")
        if "C_grid" in device_params:
            c_val = float(np.mean(device_params["C_grid"]))
            lines.append(f"C1 n{node} 0 {c_val:.6e}")
        if "L_grid" in device_params:
            l_val = float(np.mean(device_params["L_grid"]))
            lines.append(f"L1 n{node} 0 {l_val:.6e}")
        if "K_grid" in device_params:
            k_val = float(np.mean(device_params["K_grid"]))
            lines.append(f"* Effective mechanical stiffness K = {k_val:.6e} N/m (no SPICE primitive; informational)")

        if "X_grid" in device_params:
            x_val = float(np.mean(device_params["X_grid"]))
            omega = 2.0 * np.pi * freq_hz
            if abs(x_val) < 1e-9:
                lines.append(f"* Reactance X ~= 0 Ohm at {freq_hz:.4g} Hz (resonant); no reactive element added")
            elif x_val > 0:
                l_eq = x_val / omega
                lines.append(f"L2 n{node} 0 {l_eq:.6e}  ; equivalent inductance for X={x_val:.4f} Ohm @ {freq_hz:.4g} Hz")
            else:
                c_eq = -1.0 / (omega * x_val)
                lines.append(f"C2 n{node} 0 {c_eq:.6e}  ; equivalent capacitance for X={x_val:.4f} Ohm @ {freq_hz:.4g} Hz")

        if "G_grid" in device_params:
            g_val = float(np.mean(device_params["G_grid"]))
            nlf = float(np.mean(device_params["nonlinearity_grid"])) if "nonlinearity_grid" in device_params else 1.0
            if abs(nlf - 1.0) < 1e-9:
                # Purely ohmic conductance: a linear VCCS is exact and simulator-portable.
                lines.append(f"G1 n{node} 0 n{node} 0 {g_val:.6e}  ; linear memristive conductance (S)")
            else:
                # Self-rectifying nonlinear I-V (sneak-path suppression), expressed as an
                # ngspice/SPICE3-style behavioral current source: I = G0*V*(1+|V|^(nlf-1)).
                lines.append(
                    f"B1 n{node} 0 I=V(n{node},0)*{g_val:.6e}*(1+abs(V(n{node},0))^{nlf - 1.0:.6f})"
                    f"  ; nonlinear memristive I-V, k={nlf:.4f}"
                )
        if "on_off_ratio_grid" in device_params:
            ratio = float(np.mean(device_params["on_off_ratio_grid"]))
            lines.append(f"* Mean ON/OFF conductance ratio = {ratio:.4f} (informational; crossbar sneak-path margin)")

        if "mismatch_loss_db_grid" in device_params:
            ml_val = float(np.mean(device_params["mismatch_loss_db_grid"]))
            lines.append(f"* Mean impedance mismatch loss = {ml_val:.4f} dB vs reference system impedance (informational)")
        if "phase_jitter_grid" in device_params:
            pj_val = float(np.mean(device_params["phase_jitter_grid"]))
            lines.append(f"* Mean inter-node phase jitter = {pj_val:.6f} rad (AirComp aggregation fidelity; informational)")
        if "pa_distortion_grid" in device_params:
            pa_val = float(np.mean(device_params["pa_distortion_grid"]))
            lines.append(f"* Mean PA distortion index = {pa_val:.6f} (0=linear, informational)")

        lines.append("")
        lines.append(".OP")
        lines.append(".END")

        filename.write_text("\n".join(lines), encoding="ascii")
        logger.info("[SPICE] Netlist written: %s", filename)
        return filename


    @staticmethod
    def export_qeda_json(
        device_params: Dict[str, np.ndarray],
        filename: Union[str, Path],
        platform: ChipPlatform,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write a QEDA JSON netlist for quantum platforms (or any platform,
        as a uniform machine-readable summary). Reports mean and standard
        deviation per parameter grid so downstream tooling can assess
        spatial uniformity, not just a single mean value.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        components = []
        for key, grid in device_params.items():
            arr = np.asarray(grid)
            components.append({
                "parameter": key,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "shape": list(arr.shape),
            })

        qeda_structure = {
            "metadata": {
                "generator": "QEDA_Adapter_Layer",
                "framework": "Structural Calculus Ecosystem",
                "platform": platform.value,
                "version": "2.0",
                **(extra_metadata or {}),
            },
            "components": components,
        }

        with open(filename, "w", encoding="ascii") as f:
            json.dump(qeda_structure, f, indent=4)

        logger.info("[QEDA] Netlist saved: %s", filename)
        return filename


# =============================================================================
# 9.  Main orchestration bridge
# =============================================================================

class StructuralToQEDABridge:
    """
    Main entry point: orchestrates the full pipeline from Structural
    Calculus field outputs (Cahn-Hilliard / SGNO fold) to fabrication-
    ready EDA/QEDA artefacts for any supported chip platform.
    """

    def __init__(
        self,
        resolution_nm: float = 1.0,
        material: Optional[MaterialParameters] = None,
        default_layer: int = 1,
    ):
        self.mapper = StructuralFieldToDeviceMapper(material=material)
        self.exporter = QEDALayoutExporter(resolution_nm=resolution_nm, default_layer=default_layer)
        self.netlist = NetlistExporter()

    def process_simulation_result(
        self,
        u_final: torch.Tensor,
        sigma_final: torch.Tensor,
        output_prefix: Union[str, Path],
        platform: ChipPlatform = ChipPlatform.CMOS_DIGITAL,
        gdsii_threshold: float = 0.5,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Full synthesis pipeline: validate fields -> extract real geometry
        -> write GDSII -> map device physics for the chosen platform ->
        write SPICE and/or QEDA JSON netlists.

        Returns a manifest dict with all output file paths and summary
        statistics, so calling code can verify the run programmatically
        instead of only reading log output.
        """
        t0 = time.time()
        logger.info(
            "Starting Structural-to-QEDA synthesis pipeline (platform=%s)...", platform.value
        )

        sigma_final = _broadcast_sigma(u_final, sigma_final)
        out_dir = Path(output_dir) if output_dir is not None else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = out_dir / Path(output_prefix).name

        manifest: Dict[str, Any] = {
            "platform": platform.value,
            "input_shape": list(u_final.shape),
            "gdsii_threshold": gdsii_threshold,
        }

        # 1. Geometry / layout
        gds_path = f"{prefix}_layout.gds"
        if u_final.dim() == 2:
            n_elements = self.exporter.export_gdsii_2d(u_final, gdsii_threshold, gds_path)
        elif u_final.dim() == 3:
            n_elements = self.exporter.export_gdsii_3d_layered(u_final, gdsii_threshold, gds_path)
        else:
            raise FieldValidationError(
                f"u_final must be 2D or 3D for GDSII export, got dim={u_final.dim()}"
            )
        manifest["gdsii_file"] = str(gds_path)
        manifest["gdsii_elements"] = n_elements

        # 2. Device physics mapping
        device_params = self.mapper.map(platform, u_final, sigma_final)
        manifest["device_parameters"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in device_params.items()
        }

        # 3. Netlist export — SPICE for classical platforms, QEDA JSON for
        #    quantum/photonic platforms whose parameters have no SPICE
        #    lumped-element equivalent.
        if platform in (
            ChipPlatform.CMOS_DIGITAL, ChipPlatform.RF_ANALOG, ChipPlatform.MEMS,
            ChipPlatform.CARBON_NANOTUBE, ChipPlatform.IN_MEMORY_COMPUTE,
            ChipPlatform.OVER_THE_AIR_COMPUTE,
        ):
            spice_path = self.netlist.export_spice(device_params, f"{prefix}_netlist.cir", platform)
            manifest["spice_file"] = str(spice_path)
        qeda_path = self.netlist.export_qeda_json(
            device_params, f"{prefix}_netlist.json", platform,
            extra_metadata={"gdsii_elements": n_elements},
        )
        manifest["qeda_json_file"] = str(qeda_path)

        manifest["elapsed_seconds"] = round(time.time() - t0, 4)
        logger.info(
            "Pipeline complete in %.3fs. GDSII elements=%d, outputs=%s",
            manifest["elapsed_seconds"], n_elements,
            [v for k, v in manifest.items() if k.endswith("_file")],
        )
        return manifest


# =============================================================================
# 10.  Self-test suite (verifies real outputs, not just log lines)
# =============================================================================

def _run_self_tests(tmp_dir: Union[str, Path] = "./_qeda_adapter_selftest") -> None:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    passed, failed = 0, 0

    def ok(name: str, extra: str = "") -> None:
        nonlocal passed
        passed += 1
        print(f"  [PASS] {name}{(' — ' + extra) if extra else ''}")

    def fail(name: str, msg: str = "") -> None:
        nonlocal failed
        failed += 1
        print(f"  [FAIL] {name}{(' — ' + msg) if msg else ''}")

    print("=" * 70)
    print("QEDA Adapter Layer — Self-Test Suite")
    print("=" * 70)

    # --- Test 1: field validation rejects NaN ---
    try:
        bad = torch.full((8, 8), float("nan"), dtype=torch.float64)
        _validate_field(bad, "u_field")
        fail("validation rejects NaN", "did not raise")
    except FieldValidationError:
        ok("validation rejects NaN")

    # --- Test 2: 2D contour extraction on a synthetic double-well field ---
    try:
        n = 64
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2)
        disk = torch.tensor(((r < n / 4).astype(np.float64)) * 2.0 - 1.0)
        geom = extract_contours_2d(disk, threshold=0.0, resolution_nm=2.0)
        if len(geom.polygons) > 0:
            ok("2D contour extraction", f"{len(geom.polygons)} polygon(s)")
        else:
            fail("2D contour extraction", "no polygons returned")
    except Exception as e:
        fail("2D contour extraction", str(e))

    # --- Test 3: 3D iso-surface extraction on a synthetic sphere field ---
    try:
        n = 32
        zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2)
        sphere = torch.tensor((r < n / 4).astype(np.float64))
        geom3d = extract_isosurface_3d(sphere, threshold=0.5, resolution_nm=1.0)
        if geom3d.vertices.shape[0] > 0 and geom3d.faces.shape[0] > 0 and len(geom3d.layer_slices) > 0:
            ok("3D iso-surface extraction",
               f"{geom3d.vertices.shape[0]} verts, {geom3d.faces.shape[0]} faces, "
               f"{len(geom3d.layer_slices)} layer slices")
        else:
            fail("3D iso-surface extraction", "empty geometry")
    except Exception as e:
        fail("3D iso-surface extraction", str(e))

    # --- Test 4: GDSII binary file is real and well-formed ---
    try:
        n = 48
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2)
        disk = torch.tensor(((r < n / 4).astype(np.float64)) * 2.0 - 1.0)
        exporter = QEDALayoutExporter(resolution_nm=5.0)
        gds_file = tmp_dir / "test_layout.gds"
        n_el = exporter.export_gdsii_2d(disk, 0.0, gds_file)
        raw = gds_file.read_bytes()
        header_rec_type = struct.unpack(">H", raw[2:4])[0]
        if n_el > 0 and len(raw) > 0 and header_rec_type == _GDS_HEADER:
            ok("GDSII binary export", f"{n_el} element(s), {len(raw)} bytes, valid HEADER record")
        else:
            fail("GDSII binary export", "missing elements or malformed header")
    except Exception as e:
        fail("GDSII binary export", str(e))

    # --- Test 5: per-platform device mapping for all 5 platforms ---
    mapper = StructuralFieldToDeviceMapper()
    u_test = torch.rand((16, 16), dtype=torch.float64) * 2 - 1
    sigma_test = torch.ones((16, 16), dtype=torch.float64) * 1.5
    for platform in ChipPlatform:
        try:
            params = mapper.map(platform, u_test, sigma_test)
            all_finite = all(np.isfinite(v).all() for v in params.values())
            if params and all_finite:
                ok(f"device mapping [{platform.value}]", f"{len(params)} parameter grid(s)")
            else:
                fail(f"device mapping [{platform.value}]", "non-finite or empty output")
        except Exception as e:
            fail(f"device mapping [{platform.value}]", str(e))

    # --- Test 6: SPICE export rejects quantum/photonic platforms ---
    try:
        qubit_params = mapper.map(ChipPlatform.SUPERCONDUCTING_QUBIT, u_test, sigma_test)
        NetlistExporter.export_spice(qubit_params, tmp_dir / "should_fail.cir", ChipPlatform.SUPERCONDUCTING_QUBIT)
        fail("SPICE rejects quantum platform", "did not raise")
    except QEDAAdapterError:
        ok("SPICE rejects quantum platform")

    # --- Test 7: full pipeline, CMOS digital ---
    try:
        bridge = StructuralToQEDABridge(resolution_nm=2.0)
        manifest = bridge.process_simulation_result(
            u_test, sigma_test, "cmos_chip_v1",
            platform=ChipPlatform.CMOS_DIGITAL, output_dir=tmp_dir,
        )
        required = {"gdsii_file", "spice_file", "qeda_json_file"}
        if required.issubset(manifest) and Path(manifest["gdsii_file"]).exists() \
           and Path(manifest["spice_file"]).exists() and Path(manifest["qeda_json_file"]).exists():
            ok("full pipeline [cmos_digital]", f"{manifest['gdsii_elements']} GDSII elements")
        else:
            fail("full pipeline [cmos_digital]", "missing output files")
    except Exception as e:
        fail("full pipeline [cmos_digital]", str(e))

    # --- Test 8: full pipeline, superconducting qubit (no SPICE, only QEDA JSON) ---
    try:
        bridge = StructuralToQEDABridge(resolution_nm=2.0)
        manifest = bridge.process_simulation_result(
            u_test, sigma_test, "squbit_chip_v1",
            platform=ChipPlatform.SUPERCONDUCTING_QUBIT, output_dir=tmp_dir,
        )
        if "qeda_json_file" in manifest and "spice_file" not in manifest \
           and Path(manifest["qeda_json_file"]).exists():
            ok("full pipeline [superconducting_qubit]", "QEDA JSON only, no invalid SPICE deck")
        else:
            fail("full pipeline [superconducting_qubit]", "unexpected output set")
    except Exception as e:
        fail("full pipeline [superconducting_qubit]", str(e))

    # --- Test 9: 3D pipeline end-to-end (multi-layer GDSII) ---
    try:
        n = 24
        zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2)
        sphere = torch.tensor((r < n / 3).astype(np.float64))
        sigma3d = torch.ones((n, n, n), dtype=torch.float64)
        bridge = StructuralToQEDABridge(resolution_nm=1.0)
        manifest = bridge.process_simulation_result(
            sphere, sigma3d, "photonic_chip_3d",
            platform=ChipPlatform.PHOTONIC, output_dir=tmp_dir,
        )
        if manifest.get("gdsii_elements", 0) > 0 and Path(manifest["gdsii_file"]).exists():
            ok("3D pipeline [photonic]", f"{manifest['gdsii_elements']} multi-layer GDSII elements")
        else:
            fail("3D pipeline [photonic]", "no geometry produced")
    except Exception as e:
        fail("3D pipeline [photonic]", str(e))

    # --- Test 10: carbon nanotube full pipeline + SPICE (R = Rcontact+Rchannel, series C) ---
    try:
        bridge = StructuralToQEDABridge(resolution_nm=2.0)
        manifest = bridge.process_simulation_result(
            u_test, sigma_test, "cnt_chip_v1",
            platform=ChipPlatform.CARBON_NANOTUBE, output_dir=tmp_dir,
        )
        spice_text = Path(manifest["spice_file"]).read_text()
        params = mapper.map(ChipPlatform.CARBON_NANOTUBE, u_test, sigma_test)
        contact_dominates = np.mean(params["Rcontact_grid"]) > np.mean(params["Rchannel_grid"])
        c_total_lt_components = np.all(params["C_grid"] <= params["Cq_grid"] * 1.0001) and \
            np.all(params["C_grid"] <= params["Cox_grid"] * 1.0001)
        if "R1" in spice_text and "C1" in spice_text and contact_dominates and c_total_lt_components:
            ok("carbon nanotube pipeline", "quantum-limited contact dominates; series C <= each component")
        else:
            fail("carbon nanotube pipeline", "unexpected resistance/capacitance physics")
    except Exception as e:
        fail("carbon nanotube pipeline", str(e))

    # --- Test 11: in-memory compute (ReRAM) — behavioral nonlinear SPICE B-source ---
    try:
        bridge = StructuralToQEDABridge(resolution_nm=2.0)
        manifest = bridge.process_simulation_result(
            u_test, sigma_test, "reram_chip_v1",
            platform=ChipPlatform.IN_MEMORY_COMPUTE, output_dir=tmp_dir,
        )
        spice_text = Path(manifest["spice_file"]).read_text()
        params = mapper.map(ChipPlatform.IN_MEMORY_COMPUTE, u_test, sigma_test)
        g_in_range = np.all(params["G_grid"] >= mapper.mat.reram_g_off_s - 1e-12) and \
            np.all(params["G_grid"] <= mapper.mat.reram_g_on_s + 1e-12)
        nlf_gt_one = np.all(params["nonlinearity_grid"] > 1.0)
        has_b_source = "B1" in spice_text and "nonlinear memristive" in spice_text
        if g_in_range and nlf_gt_one and has_b_source:
            ok("in-memory compute pipeline", "G within [G_off,G_on], nlf>1, behavioral B-source emitted")
        else:
            fail("in-memory compute pipeline",
                 f"g_in_range={g_in_range} nlf_gt_one={nlf_gt_one} has_b_source={has_b_source}")
    except Exception as e:
        fail("in-memory compute pipeline", str(e))

    # --- Test 12: over-the-air compute — radiation impedance + reactance-to-L/C conversion ---
    try:
        bridge = StructuralToQEDABridge(resolution_nm=2.0)
        manifest = bridge.process_simulation_result(
            u_test, sigma_test, "aircomp_chip_v1",
            platform=ChipPlatform.OVER_THE_AIR_COMPUTE, output_dir=tmp_dir,
        )
        spice_text = Path(manifest["spice_file"]).read_text()
        params = mapper.map(ChipPlatform.OVER_THE_AIR_COMPUTE, u_test, sigma_test)
        r_rad_bounded = np.all(params["R_grid"] >= 0.0) and np.all(params["R_grid"] <= 73.0 + 1e-9)
        has_reactive_element = ("L2" in spice_text) or ("C2" in spice_text) or ("resonant" in spice_text)
        has_aircomp_metrics = "phase jitter" in spice_text and "PA distortion" in spice_text
        if r_rad_bounded and has_reactive_element and has_aircomp_metrics:
            ok("over-the-air compute pipeline", "R_rad in [0,73] Ohm; reactance converted; AirComp metrics reported")
        else:
            fail("over-the-air compute pipeline",
                 f"r_rad_bounded={r_rad_bounded} has_reactive_element={has_reactive_element} "
                 f"has_aircomp_metrics={has_aircomp_metrics}")
    except Exception as e:
        fail("over-the-air compute pipeline", str(e))

    # --- Test 13: linear (ohmic) memristor falls back to a plain G-element, not a B-source ---
    try:
        params_lin = {"G_grid": np.full((4, 4), 5e-5), "nonlinearity_grid": np.full((4, 4), 1.0)}
        spice_path = tmp_dir / "linear_memristor_test.cir"
        NetlistExporter.export_spice(params_lin, spice_path, ChipPlatform.IN_MEMORY_COMPUTE)
        text = spice_path.read_text()
        if "G1" in text and "linear memristive" in text and "B1" not in text:
            ok("linear memristor uses G-element", "no unnecessary behavioral B-source for nlf=1")
        else:
            fail("linear memristor uses G-element", "expected plain G-element for nlf=1")
    except Exception as e:
        fail("linear memristor uses G-element", str(e))

    print("=" * 70)
    print(f"Self-test results: {passed} passed, {failed} failed (output dir: {tmp_dir})")
    print(f"GDSII backend: {'gdspy' if _HAS_GDSPY else 'built-in minimal writer'}")
    print("=" * 70)
    if failed > 0:
        raise SystemExit(1)


# =============================================================================
# Self-Test Example
# =============================================================================
if __name__ == "__main__":
    _run_self_tests()
