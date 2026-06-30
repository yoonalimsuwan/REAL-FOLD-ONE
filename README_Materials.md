# MATERIALS ONE & TLS SURFACE ONE

**Crystal/Solid-State Material Discovery and Superconducting-Qubit TLS Loss Infrastructure** — two companion modules within the ONE Ecosystem, designed to extend the ecosystem's scientific reach beyond biomolecules and particle physics into solid-state materials science and quantum chip engineering.

| File | Role | Status |
|---|---|---|
| `materials_one_v1_3.py` | Crystal structure GNN, formation energy/band gap prediction, semiconductor candidate screening, quantum-tunneling risk proxy, EDA bridge, AGI ONE adapter | v1.3 — production-engineering scaffold; architecture and neighbor search are production-grade, GNN heads are untrained |
| `tls_surface_one.py` | Surface/interface physics for superconducting qubit TLS loss — literature lookup, participation ratio estimation, future-model contract | v0.1 — real physics components usable today; ML surrogate interface is a forward declaration pending training data |

---

## Developer

**Yoon A Limsuwan** / MSPS NETWORK
ORCID: `0009-0008-2374-0788` · GitHub: [yoonalimsuwan](https://github.com/yoonalimsuwan) · Email: msps4u@gmail.com
License: MIT

**AI development assistant:** Claude (Anthropic) — architecture co-design, neighbor-search correctness review, surface-physics scoping, honesty pass on all claims.

---

## Why Two Files, Not One

`materials_one_v1_3.py` uses `CrystalStructure`: a periodic lattice with atomic positions, which is the correct representation for bulk material properties (formation energy, band gap, quantum-tunneling risk from effective mass + band gap). This is also the input representation compatible with MACE-MP-0, Materials Project, and OQMD.

TLS loss in superconducting qubits does **not** arise from bulk crystal properties. It arises from amorphous native-oxide layers at metal surfaces, metal–substrate interface defects, and the geometric overlap of the qubit's electric-field mode with those lossy regions — none of which are representable as a periodic crystal lattice. Using a bulk-crystal GNN to predict TLS loss would be "garbage in, garbage out," so that problem is given its own module with its own representation (`InterfaceStack`, `DielectricLayer`) and its own physics (`ParticipationRatioEstimator`).

---

## 1. `materials_one_v1_3.py` — MATERIALS ONE

### What it is

A production-engineering scaffold for crystal/solid-state material discovery, with the following capabilities across three use cases:

**Semiconductor material screening (2nm node and beyond):** Formation energy and band gap prediction via a message-passing GNN (`DFTSurrogateGNN`), combined with a WKB-based quantum-tunneling risk proxy (`assess_tunneling_risk`) that estimates tunneling transmission probability at a given gate length using the material's band gap as a barrier-height proxy and a per-element effective-mass lookup table. This is the piece most directly relevant to finding channel materials that survive at sub-2nm gate lengths better than silicon.

**Superconducting qubit candidate pre-screening:** An opt-in pair of GNN prediction heads (`tc_head`, `tls_loss_proxy_head`) plus `QubitCandidateReport` and `screen_candidate_for_qubit_device()` — a bridge into `eda_qeda_adapter_layer.py`'s `map_superconducting_qubit()`. This path gives a first-pass ranking of candidate materials; serious TLS loss analysis should use `tls_surface_one.py` instead (see below).

**AGI ONE integration:** `MaterialsONEAdapter` mirrors the `SurrogateAdapter` contract used throughout `agi_one_v3_8.py` (`.encode()` and `quality_fn` hook), so a screened material's latent representation can flow into `EcosystemOrchestrator`'s cross-domain alignment without modifying any existing code.

### Architecture (10 sections)

The file is organized into numbered sections matching the ONE Ecosystem convention:

- **Section 0** — Exception hierarchy (`MaterialsONEError`, `StructureValidationError`, `NeighborSearchError`, `BackendUnavailableError`, `TrainingDataError`), matching `eda_qeda_adapter_layer.py`'s convention.
- **Section 1** — Periodic table (all 118 elements), per-element effective-mass lookup table, element validation.
- **Section 2** — `CrystalStructure`: validated periodic crystal representation (lattice + fractional coordinates + species). Rejects singular/degenerate lattices, NaN/Inf coordinates, empty structures, and unknown elements at construction time.
- **Section 3** — `periodic_neighbor_search()` with three backends, dispatched by `NeighborSearchMethod`:
  - `TORCH_CLUSTER` — grid-hashed GPU radius graph via `torch_cluster.radius()`. Selected by `AUTO` above 5,000 atoms when `torch_cluster` is installed. Scales to >100k-atom supercells; same primitive REAL FOLD ONE's `structural_gno_fold_v4.py` uses. Configurable `max_num_neighbors` cap prevents GPU memory explosion on pathologically dense structures; truncated neighbor lists are logged as warnings.
  - `KDTREE` — `scipy.spatial.cKDTree.query_ball_tree`, roughly O(M log M). Selected by `AUTO` for smaller structures when scipy is available.
  - `BRUTE_FORCE` — dense O(M²) pairwise distance matrix. Zero dependencies; kept as the correctness reference the other two backends are regression-tested against.

  All three backends share the same periodic-image replica construction (correctness-necessary: not limited to the home unit cell) and have been regression-tested for neighbor-set equivalence.

- **Section 4** — `InteratomicPotential`: wraps MACE-MP-0 (real ML-DFT physics, Materials Project–trained) when `mace-torch` and `ase` are installed; degrades to a clearly labeled Lennard-Jones placeholder otherwise. `.is_real_physics` flag makes the distinction explicit.
- **Section 5** — `DFTSurrogateGNN`: SchNet/CGCNN-style message-passing GNN with `formation_energy_head`, `band_gap_head`, and (opt-in via `MaterialsConfig.predict_superconductor_properties=True`) `tc_head` and `tls_loss_proxy_head`. **UNTRAINED as shipped** — architecture only.
- **Section 6** — WKB quantum-tunneling risk proxy: `wkb_tunneling_probability()` and `assess_tunneling_risk()`. Uses per-element effective-mass lookup (not a single hardcoded default); labeled explicitly as a ranking heuristic, not a full quantum-transport simulation (NEGF).
- **Section 6B** — Superconducting qubit candidate screening: `QubitCandidateReport`, `screen_qubit_candidate()`, `to_eda_material_parameters()`, `screen_candidate_for_qubit_device()`. The EDA bridge applies `tls_loss_proxy` as an explicit post-hoc rescaling of `loss_tangent_grid`/`Q_grid` rather than injecting it through `MaterialParameters` (which doesn't carry that field), so `eda_qeda_adapter_layer.py` is never modified.
- **Section 7** — `train_dft_surrogate()`: a real supervised training loop (Adam, MSE on formation energy + band gap, early stopping, optional checkpoint save). Not just inference scaffolding — the architecture can actually be trained once a dataset is supplied.
- **Section 8** — `MaterialsONEAdapter`: AGI ONE `EcosystemOrchestrator` adapter.
- **Section 9** — 22-test assert-based self-test suite (PASS/FAIL per test), covering structure validation, all three neighbor-search backends and their cross-equivalence, isolated-atom edge case, tunneling risk physics, qubit screening error handling, and the EDA bridge.

### Installation

```bash
pip install torch scipy --break-system-packages
# For real ML-DFT physics (optional but strongly recommended):
pip install mace-torch ase --break-system-packages
# For GPU-scale neighbor search (optional):
pip install torch-cluster --break-system-packages
```

### Quick start

```python
import torch
from materials_one_v1_3 import (
    CrystalStructure, MaterialsConfig, DFTSurrogateGNN,
    assess_tunneling_risk, MaterialsONEAdapter
)

# Build a crystal structure
lattice = torch.eye(3) * 5.43
frac_coords = torch.tensor([[0.0,0.0,0.0],[0.25,0.25,0.25],
                             [0.5,0.5,0.0],[0.75,0.75,0.25]])
structure = CrystalStructure(lattice=lattice, frac_coords=frac_coords,
                             species=["Ta","Ta","C","C"], name="toy_TaC")

# Semiconductor screening: quantum-tunneling risk at 2nm gate length
report = assess_tunneling_risk("TaC_candidate", band_gap_ev=1.5,
                               gate_length_angstrom=20.0, dominant_element="Ta")
print(report.to_dict())

# GNN forward pass (untrained — shapes only until trained)
cfg = MaterialsConfig(latent_dim=128)
model = DFTSurrogateGNN(cfg)
out = model(structure)
print(out["band_gap_ev"].item())   # placeholder until trained

# Qubit candidate screening (opt-in heads)
from materials_one_v1_3 import screen_qubit_candidate
cfg_q = MaterialsConfig(latent_dim=128, predict_superconductor_properties=True)
model_q = DFTSurrogateGNN(cfg_q)
qubit_report = screen_qubit_candidate(structure, model_q)
print(qubit_report.to_dict())

# AGI ONE adapter
adapter = MaterialsONEAdapter(agi_latent_dim=128)
latent = adapter.encode(structure)        # (1, 128) tensor
print(adapter.get_quality_score())        # quality_fn hook for EcosystemOrchestrator
```

### What is and is not production-ready

Production-ready: `CrystalStructure` validation, all three neighbor-search backends (including the periodic-boundary bug fix from v1.0), the exception hierarchy, `InteratomicPotential` wrapping MACE-MP-0, `assess_tunneling_risk()` with per-element effective mass, the training loop, the AGI ONE adapter contract, and all 22 self-tests.

Not yet production-ready: `DFTSurrogateGNN` is an untrained architecture. Formation energy, band gap, Tc, and tls_loss_proxy outputs are not physically meaningful until the model is trained against real data (Materials Project / OQMD for the first two; SuperCon database for Tc; no adequate public dataset exists for tls_loss_proxy — see `tls_surface_one.py` for why). The WKB tunneling risk proxy is explicitly a ranking tool, not a device-yield simulation.

### Version history

| Version | Key change |
|---|---|
| v0.1 | Initial scaffold: CrystalStructure, placeholder GNN, brute-force O(N²) neighbor search, smoke test |
| v1.0 | Production-hardening: exception hierarchy, structure validation, periodic-boundary neighbor search bug fix, real training loop, assert-based self-test suite |
| v1.1 | KD-tree neighbor search (scipy.spatial.cKDTree), O(M log M); brute-force kept as correctness reference |
| v1.2 | GPU radius-graph neighbor search (torch_cluster.radius()); three-tier AUTO dispatch; max_num_neighbors cap with truncation warning |
| v1.3 | Opt-in Tc/TLS-loss-proxy GNN heads; `QubitCandidateReport`; EDA bridge (`screen_candidate_for_qubit_device()`); `MaterialsONEAdapter`; 7 new qubit-screening self-tests |

---

## 2. `tls_surface_one.py` — TLS SURFACE ONE

### Why the problem requires a different module

TLS (two-level system) loss is the dominant decoherence mechanism limiting superconducting qubit coherence times today. Its physical origin is amorphous defects at metal–oxide and oxide–substrate interfaces — not bulk crystal properties. The key difference:

- A bulk-crystal GNN sees a perfect periodic lattice and produces bulk electronic structure. It has no information about surface oxide chemistry, fabrication-process-induced defects, or how the qubit's electric-field mode geometrically overlaps the lossy region.
- TLS loss depends on all three of those things. Changing only the bulk material (e.g. from Nb to Ta) helps primarily because it changes the native oxide chemistry and thickness, not because the bulk crystal has a lower loss tangent.

This module therefore uses `InterfaceStack`/`DielectricLayer` as its fundamental representation — a physically correct layered surface stack, not a periodic crystal — and provides three things: real physics usable today, an abstract contract for a future trained model, and a data schema for when the training data becomes available.

### Architecture (10 sections)

- **Section 0** — Exception hierarchy: `TLSSurfaceONEError`, `InterfaceDefinitionError`, `MaterialNotInLookupError`, `SurrogateNotTrainedError`.
- **Section 1** — `DielectricLayer` and `InterfaceStack`: the correct surface representation. Each layer carries thickness, permittivity, loss tangent, and flags for `is_amorphous` (native oxide layers cannot be represented as a periodic crystal) and `is_superconductor` (the qubit metal film). An empty stack or a layer with negative thickness/permittivity raises `InterfaceDefinitionError` at construction time.
- **Section 2** — Literature-grounded TLS material lookup table (`TLSMaterialEntry`, `lookup_tls_material()`). Every entry cites a published measurement with a DOI. Loss tangent values are representative of the single-photon, millikelvin regime relevant to qubit operation. Current entries: Nb, Al, Ta (alpha-phase), TiN, Si substrate, sapphire substrate. New entries require a published measurement and reference — the table does not accept unverified values.
- **Section 3** — `ParticipationRatioEstimator`: **real physics, usable today, no training required.** Implements the standard PR formalism from Wenner et al. (2011) and Wang et al. (2015): the participation ratio p_i of each lossy layer is the ratio of that layer's capacitance to the vacuum gap capacitance (parallel-plate approximation for a CPW/transmon geometry). The total quality factor estimate is 1/(Σ p_i δ_i). This is the correct first-principles reason why thinner oxides matter, why high-permittivity substrates can hurt coherence, and why Ta/sapphire outperforms Nb/Si even before accounting for bulk material differences. The estimator is analytic and configurable (qubit gap width, metal thickness).
- **Section 4** — `TLSSurrogateInterface`: an abstract base class defining the contract a future trained model must implement — a single `predict(stack, fabrication_metadata) -> TLSPrediction` method plus an `is_trained` property. The abstract design is intentional: no adequate training dataset currently exists for a model that generalizes across material systems. `FabricationMetadata` defines the process-context inputs such a model would need (deposition method, base pressure, anneal temperature, surface clean chemistry, substrate orientation).
- **Section 5** — `TLSDataEntry`: the schema for a future training dataset entry. Each entry corresponds to one experimental qubit cooldown and includes the measured T1, the interface stack, fabrication metadata, surface characterization data (XPS oxide thickness/composition, TEM interface roughness), and a `derived_loss_tangent()` method. No training data ships with this file.
- **Section 6** — `LiteratureLookupTLSSurrogate`: a concrete `TLSSurrogateInterface` implementation backed by the lookup table from Section 2. Returns the published loss tangent, corrected by the participation ratio from Section 3. `.is_trained` returns `False` — this is honest about being a lookup, not a model. Raises `MaterialNotInLookupError` for unknown metals rather than returning a fabricated value.
- **Section 7** — EDA bridge: `compute_tls_corrected_qubit_device_parameters()` wires a literature/PR-corrected loss tangent into `eda_qeda_adapter_layer.map_superconducting_qubit()` via the same post-hoc rescaling approach as `materials_one_v1_3.py`, without modifying `eda_qeda_adapter_layer.py`. The difference from `materials_one`'s bridge: the rescaling factor here comes from real published physics (PR × δ_TLS from literature) rather than an untrained GNN output.
- **Section 8** — Pre-built reference stacks: `make_nb_on_si_stack()`, `make_al_on_si_stack()`, `make_ta_on_sapphire_stack()`, using dielectric constants and oxide thicknesses from the literature. These are the three most common qubit material systems as of 2026.
- **Section 9** — 13-test assert-based self-test suite, including `t_ranking_ta_beats_al_beats_nb` (physics sanity check: the known ordering of TLS loss in the literature must be reproduced) and `t_pr_estimator_thinner_oxide_reduces_loss` (physics sanity check: the participation ratio formalism must correctly predict that thinning the oxide reduces loss).

### No training required — what works today

```python
from tls_surface_one import (
    make_ta_on_sapphire_stack, make_al_on_si_stack, make_nb_on_si_stack,
    LiteratureLookupTLSSurrogate, ParticipationRatioEstimator
)

# Compare three material stacks
surrogate = LiteratureLookupTLSSurrogate()
estimator = ParticipationRatioEstimator(qubit_gap_um=10.0)

for fn, label in [
    (make_nb_on_si_stack, "Nb/Nb2O5/Si"),
    (make_al_on_si_stack, "Al/Al2O3/Si"),
    (make_ta_on_sapphire_stack, "Ta/Ta2O5/Sapphire"),
]:
    stack = fn()
    pred = surrogate.predict(stack)
    pr = estimator.estimate(stack)
    print(f"{label}: tan_delta={pred.loss_tangent:.2e}, Q_est={pr.estimated_quality_factor:.2e}")
    print(f"  dominant loss layer: {pr.dominant_loss_layer}")
    print(f"  source: {pred.source_description}")
```

Expected output ordering (matching published experimental results): Ta/sapphire < Al/Si < Nb/Si.

### Defining a custom material stack

```python
from tls_surface_one import InterfaceStack, DielectricLayer

# Hypothetical NbTiN on sapphire (MKID / KID geometry)
stack = InterfaceStack(
    name="NbTiN_on_sapphire",
    layers=[
        DielectricLayer("NbTiN_film", thickness_nm=0, relative_permittivity=1.0,
                        loss_tangent_intrinsic=0.0, is_superconductor=True,
                        notes="NbTiN, Tc ~ 14K; used in KIDs"),
        DielectricLayer("native_oxide", thickness_nm=2.0, relative_permittivity=25.0,
                        loss_tangent_intrinsic=5e-4, is_amorphous=True,
                        notes="estimated; characterize via XPS"),
        DielectricLayer("sapphire_sub", thickness_nm=0, relative_permittivity=9.3,
                        loss_tangent_intrinsic=4e-6),
    ],
)
```

### How to plug in a future trained model

When a trained atomistic TLS model becomes available (e.g. from SQMS Center surface-chemistry datasets, or from non-periodic amorphous ML potentials):

```python
from tls_surface_one import TLSSurrogateInterface, TLSPrediction

class MyTrainedTLSSurrogate(TLSSurrogateInterface):

    def __init__(self, checkpoint_path: str):
        # load your trained model here
        self._model = ...

    @property
    def is_trained(self) -> bool:
        return True

    def predict(self, stack, fabrication_metadata=None) -> TLSPrediction:
        # run your model on stack + fabrication_metadata
        loss_tangent = self._model(stack, fabrication_metadata)
        return TLSPrediction(
            loss_tangent=loss_tangent,
            tls_density_relative=1.0,
            confidence=0.85,
            is_from_trained_model=True,
            source_description="MyTrainedTLSSurrogate checkpoint v2.0",
        )
```

Pass an instance of this class to `compute_tls_corrected_qubit_device_parameters(..., surrogate=MyTrainedTLSSurrogate("ckpt.pt"))` — no other changes needed.

### What data a future model would need

The `TLSDataEntry` schema shows the required fields: measured T1, measurement temperature, interface stack definition, fabrication metadata (deposition method, base pressure, surface clean), and ideally surface characterization from XPS (oxide thickness and composition) and TEM (interface roughness). The `derived_loss_tangent()` method converts T1 and qubit frequency into an effective loss tangent for training. This schema is ready to receive data; no data ships with this module.

---

## Relationship to the rest of the ONE Ecosystem

Both modules slot into the ONE Ecosystem via the `MaterialsONEAdapter` (in `materials_one_v1_3.py`) which follows the same `.encode()` / `quality_fn` adapter contract used throughout `agi_one_v3_8.py`. Once registered with `EcosystemOrchestrator`, materials latents participate in the same triadic-coherence alignment and curriculum training machinery as REAL FOLD ONE, EVOLUTION ONE, and MENTAL ONE — without modifying any existing cluster.

`tls_surface_one.py` connects to the rest of the ecosystem via `eda_qeda_adapter_layer.py`'s `map_superconducting_qubit()` function, bridging material-physics predictions into the same GDSII layout pipeline that all other chip platforms (CMOS, RF, photonic, carbon nanotube, in-memory compute, over-the-air compute) use.

```
CrystalStructure → materials_one_v1_3.py → MaterialsONEAdapter → EcosystemOrchestrator (AGI ONE)
                                          ↓
InterfaceStack → tls_surface_one.py → compute_tls_corrected_qubit_device_parameters()
                                          ↓
                          eda_qeda_adapter_layer.map_superconducting_qubit()
                                          ↓
                                   GDSII layout / netlist
```

---

## Notes on scope

These two files cover solid-state materials and quantum chip engineering. They deliberately do not cover molecular dynamics of proteins/DNA/RNA (REAL FOLD ONE), cancer genomics (EVOLUTION ONE), computational psychiatry (MENTAL ONE), fluid dynamics (SUPER DNS ONE), particle physics (STANDARD ONE), or mathematical conjectures (RH/HODGE/BSD/GRH ONE) — those are documented within their own clusters. This README covers only MATERIALS ONE and TLS SURFACE ONE.
