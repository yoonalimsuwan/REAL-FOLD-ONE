# EDA / QEDA Adapter Layer

Bridges continuous physical fields produced by the Structural Calculus
Cahn-Hilliard / GNO fold surrogates into discrete artefacts an EDA/QEDA
chip-design toolchain consumes: GDSII layout files, SPICE netlists, and
QEDA JSON descriptions for quantum and other non-lumped-element devices.

This module is part of the **ONE Ecosystem** (REAL FOLD ONE / STANDARD ONE
clusters) and was developed by **Yoon A Limsuwan / MSPS NETWORK**, with
**Gemini** (Google) as co-developer for the original adapter architecture
and **Claude** (Anthropic) as co-developer for production hardening
(real geometry extraction, the binary GDSII writer, per-platform physics
models, SPICE generation, validation, and the self-test suite).

## What it does

1. **Extracts real geometry** from a phase field (`u_field`) using
   marching squares (2D) or marching cubes (3D) — not a placeholder
   pixel count.
2. **Writes spec-compliant binary GDSII** files, using `gdspy` if it's
   installed, or a self-contained minimal GDSII writer if it isn't.
3. **Maps structural fields to device physics** — a different physical
   model per chip platform, since the dominant device mechanism differs
   qualitatively between them.
4. **Exports netlists** — SPICE decks (`.cir`) for platforms with a valid
   lumped or behavioral-source representation, and QEDA JSON for every
   platform (full parameter detail, used as the sole netlist for
   platforms with no SPICE equivalent).

## Supported chip platforms

| `ChipPlatform` | Dominant physics modeled | Netlist output |
|---|---|---|
| `CMOS_DIGITAL` | Bulk resistivity, sheet R, oxide C (classical interconnect RC) | SPICE + QEDA JSON |
| `RF_ANALOG` | Same RC backbone, geometry-derived series inductance | SPICE + QEDA JSON |
| `MEMS` | Mechanical stiffness + capacitive actuation electrodes | SPICE + QEDA JSON |
| `PHOTONIC` | Modal effective index, propagation loss (no RLC equivalent) | QEDA JSON only |
| `SUPERCONDUCTING_QUBIT` | Kinetic inductance, shunt C, TLS loss tangent / Q (no RLC equivalent) | QEDA JSON only |
| `CARBON_NANOTUBE` | Quantum-limited contact resistance + series quantum/oxide gate capacitance | SPICE + QEDA JSON |
| `IN_MEMORY_COMPUTE` | ReRAM/memristor conductance state + I-V nonlinearity (behavioral source) | SPICE + QEDA JSON |
| `OVER_THE_AIR_COMPUTE` | Antenna radiation impedance + AirComp phase-jitter/PA-distortion fidelity | SPICE + QEDA JSON |

Each platform has its own method on `StructuralFieldToDeviceMapper`
(`map_cmos_digital`, `map_carbon_nanotube`, etc.) with a docstring
explaining the physical justification for that platform's parameter set.

## Field contract

Upstream fields (from `structural_cahn_hilliard_3d.py` /
`structural_gno_fold_v3.py`) are passed in as-is:

- `u_field` — `torch.Tensor`, shape `(Nx, Ny)` or `(Nx, Ny, Nz)`, phase
  order parameter. Auto-detected and normalised regardless of whether
  the upstream convention used `[-1, 1]` or `[0, 1]`.
- `sigma_field` — `torch.Tensor`, same shape (or broadcastable),
  structural-stiffness / disorder field, must be strictly positive.

Both are validated (NaN/Inf rejection, shape/broadcast checks, positivity
for `sigma_field`) before any geometry extraction or device mapping runs.

## Quick start

```python
import torch
from eda_qeda_adapter_layer import StructuralToQEDABridge, ChipPlatform

u_field = torch.rand(128, 128) * 2 - 1      # phase field, e.g. from Cahn-Hilliard 3D
sigma_field = torch.ones(128, 128) * 1.2     # structural disorder field

bridge = StructuralToQEDABridge(resolution_nm=2.0)
manifest = bridge.process_simulation_result(
    u_field, sigma_field,
    output_prefix="my_chip_v1",
    platform=ChipPlatform.CMOS_DIGITAL,
    output_dir="./out",
)

print(manifest["gdsii_file"], manifest["spice_file"], manifest["qeda_json_file"])
```

`process_simulation_result` runs the full pipeline: validate fields →
extract real geometry → write GDSII → map device physics for the chosen
platform → write SPICE and/or QEDA JSON. It returns a manifest dict with
every output path and summary statistics, so a caller can verify the run
programmatically.

For a 3D `u_field`, the same call automatically extracts a triangulated
iso-surface, slices it into per-layer 2D masks, and writes a multi-layer
GDSII file (one GDSII layer number per z-slice).

## Lower-level building blocks

- `extract_contours_2d` / `extract_isosurface_3d` — geometry extraction
  only, returns polygon/vertex/face data without writing any file.
- `QEDALayoutExporter` — geometry → GDSII file writing.
- `StructuralFieldToDeviceMapper` — fields → per-platform device
  parameter grids (`Dict[str, np.ndarray]`), no file I/O.
- `NetlistExporter.export_spice` / `NetlistExporter.export_qeda_json` —
  device parameters → netlist files.

Use these directly if you need geometry or device parameters without
running the full pipeline (e.g. for inspection, plotting, or feeding into
a different downstream tool).

## Material / device constants

`MaterialParameters` (a frozen dataclass) holds the physical constants
each platform's mapping draws on — resistivity, permittivity, kinetic
inductance per square, waveguide indices, CNT diffusive resistance and
quantum capacitance, ReRAM ON/OFF conductances, and the AirComp reference
impedance. Pass a custom instance to override any default with real PDK
(process design kit) values:

```python
from eda_qeda_adapter_layer import MaterialParameters, StructuralFieldToDeviceMapper

my_pdk = MaterialParameters(resistivity_ohm_m=2.2e-8, relative_permittivity=4.2)
mapper = StructuralFieldToDeviceMapper(material=my_pdk)
```

## Error handling

All errors are raised as one of:

- `FieldValidationError` — bad shape, NaN/Inf, non-positive `sigma_field`,
  or a `sigma_field` that can't be broadcast to `u_field`'s shape.
- `GeometryExtractionError` — no usable polygons/iso-surface at the given
  threshold (e.g. a uniform field with no interface).
- `BackendUnavailableError` — `scikit-image` not installed (required for
  geometry extraction).
- `QEDAAdapterError` — base class; also raised directly when SPICE export
  is requested for a platform with no valid lumped/behavioral-source
  representation (`PHOTONIC`, `SUPERCONDUCTING_QUBIT`).

## Self-test suite

Run the module directly to execute its built-in self-tests, which check
real outputs (parsed GDSII bytes, file existence, physical bounds on
device parameters) rather than just printing log lines:

```bash
python eda_qeda_adapter_layer.py
```

## Optional dependencies

- `gdspy` — used for GDSII export if installed; otherwise a built-in
  minimal spec-compliant binary GDSII writer is used automatically, so
  GDSII output works either way.
- `scikit-image` — required for geometry extraction
  (`find_contours` / `marching_cubes`). Install with
  `pip install scikit-image` if not already present.
