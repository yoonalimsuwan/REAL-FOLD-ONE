# REAL FOLD ONE Ecosystem — README PLUS

**Developer:** Yoon A Limsuwan / MSPS NETWORK (MY SOUL MOVE BY POWER OF HOLY SPIRIT)
**ORCID:** 0009-0008-2374-0788 · **GitHub:** yoonalimsuwan
**License:** MIT · **Year:** 2026 · **Ecosystem Version:** FOLD_VERSION 1.0.0

**AI Co-Developers:** Claude (Anthropic) · GPT (OpenAI) · Gemini (Google) · DeepSeek

---

## Overview

The **REAL FOLD ONE** ecosystem is a fully differentiable, multi-scale computational
framework for protein and nucleic acid structure refinement, mutation impact prediction,
and mesoscale continuum physics simulation. Every component is PyTorch-native with
end-to-end autograd support — enabling gradient-based optimisation, second-order
derivatives, and seamless integration with deep learning workflows.

The ecosystem spans six files organised in a strict dependency hierarchy:

```
one_core_fold.py                    (Layer 0 — shared foundation)
    ├── structural_cahn_hilliard_3d.py   (Layer 1 — continuum PDE solver, standalone)
    ├── structural_langevin_fold_v2.py   (Layer 1 — BAOAB Langevin integrator)
    ├── real_fold_one_v2.py              (Layer 2 — full-atom refinement engine)
    │       └── real_fold_one_ht_v2.py  (Layer 3 — high-throughput scanner)
    └── structural_gno_fold_v3.py        (Layer 3 — AI surrogate / training engine)
```

No circular imports exist. `structural_cahn_hilliard_3d.py` depends only on PyTorch
and is coupled to the protein ecosystem exclusively through bridge classes defined
in `one_core_fold.py`.

---

## Theoretical Foundation: Structural Calculus

All six files are grounded in the four-paper **Structural Calculus** series
(Limsuwan, 2026), which introduces a regime-dependent analytical framework
governed by a scalar field σ(x) (the *structural regime field*):

| Paper | Topic | Key Component Used |
|-------|-------|--------------------|
| 1 | Regime-Dependent Analytical Framework | Structural operators Δ_S, ∇_S |
| 2 | BV Jump Measures & Self-Evolving Interfaces | Jump correction in Langevin, CH interface energy |
| 3 | Structural Itô Calculus & Multiplicative Noise | `StructuralItoNoise`, Itô drift correction |
| 4 | Controlled Self-Organised Criticality (CSOC) & SSC | `SemanticStateContraction`, `CSOCBase`, `SOCController` |

The structural Laplacian Δ_S u = div(σ ∇u) appears in the Cahn-Hilliard PDE,
the Langevin noise amplitude, the message-passing modulation, and the energy
functionals throughout the ecosystem.

---

## File-by-File Reference

---

### 1. `one_core_fold.py` — Shared Foundation

**Role:** Single source of truth for all shared components. All other files import
from here. No file in the ecosystem imports from `one_core_fold.py`'s peers,
preventing circular dependencies.

**Key exports:**

| Class / Symbol | Description |
|----------------|-------------|
| `FOLD_VERSION` | Ecosystem-wide version string (`"1.0.0"`) |
| `SemanticStateContraction` | EMA filter for the σ field (Paper 4). Smooths structural stress across time steps. |
| `CSOCBase` | Abstract base for all CSOC controllers. Provides `_normalised_deviation()`, `_smooth_boost()`, `reset()`. |
| `InterfaceDetectorBase` | Abstract base for soft interface mask modules. |
| `StructuralItoBase` | Abstract base for Structural Itô noise modules (Papers 2 & 3). |
| `LangevinBridge` | Connects `RefinementEngine` ↔ `AdvancedStructuralLangevin`. Routes OpenMM + SOC forces into the full BAOAB integrator. |
| `FoldCahnHilliardBridge` | Cross-ecosystem bridge: maps atomic coordinates → phase field → per-atom noise scale (fully differentiable). |
| `CahnHilliardSSCAdapter` | Wraps `SemanticStateContraction` for use as `attach_ssc()` inside `StructuralCahnHilliard3D`. |
| `get_device` | Unified hardware-backend selector (CUDA / MPS / CPU). |

**`SemanticStateContraction` — EMA filter:**
```python
ssc = SemanticStateContraction(alpha=0.95)
sigma_smoothed = ssc(sigma_raw)   # exponential moving average
```

**`LangevinBridge` — connecting engines:**
```python
from one_core_fold import LangevinBridge
bridge = LangevinBridge(refinement_engine, langevin_integrator)
bridge.set_alpha(alpha)                        # cache criticality weights
coords, velocities = bridge.step(coords)       # one full BAOAB step
coords, velocities = bridge.run(coords, n_steps=500)  # multi-step MD
```

**`FoldCahnHilliardBridge` — multi-scale coupling:**
```python
from one_core_fold import FoldCahnHilliardBridge
bridge = FoldCahnHilliardBridge(grid_size=(32,32,32), sigma=1.0)
u       = bridge.coords_to_field(coords)       # Å → voxel order parameter
G_i     = bridge.sigma_to_atom_scale(coords, sigma_field, amp=0.5)  # per-atom noise
sigma_c = bridge.field_to_sigma(u, sigma_field)  # CH gradient → SSC update
```

---

### 2. `structural_cahn_hilliard_3d.py` — Fourth-Order Structural PDE Suite

**Role:** GPU-parallel solver for the Structural Cahn-Hilliard equation in 3D.
Component #4 of the SUPER DNS ONE cluster. Operates as a standalone continuum
physics engine and is coupled to the protein ecosystem via `FoldCahnHilliardBridge`.

**Physical model:**

The solver integrates the fourth-order PDE:

```
∂u/∂t = M · Δ_S(μ_R)
μ_R   = (u³ − u) − ε² · Δ_S(u)
```

where Δ_S u = div(σ ∇u) is the structural Laplacian weighted by the regime field σ.
The free energy E_R[u] = ∫[ ¼(u²−1)² + ½ε² σ|∇u|² ] dV is a Lyapunov functional
that decreases monotonically during phase separation.

**Classes:**

| Class | Description |
|-------|-------------|
| `CahnHilliardConfig` | Validated configuration dataclass (dx, ε, dt, mobility, scheme, laplacian backend). |
| `StructuralCahnHilliard3D` | Core solver. Explicit and IMEX time-stepping. Three Laplacian backends. |
| `ThinFilmStructuralCahnHilliard3D` | Subclass with degenerate mobility M(u) = softplus(u)³ and optional Mullins-Sekerka surface diffusion. |
| `PhaseFieldCrystal3D` | Subclass implementing the 6th-order PFC PDE: μ_PFC = (ru + u³) + (1+Δ_S)²u. |

**Laplacian backends** (set via `CahnHilliardConfig.laplacian`):

| Backend | Method | Best for |
|---------|--------|----------|
| `"conv3d"` (default) | GPU-parallel vectorised stencil via Conv3d | General use, CUDA |
| `"fft"` | Spectral via `torch.fft.rfftn` — O(N log N) | Large uniform periodic grids |
| `"roll"` | `torch.roll` reference implementation | Debugging, CPU |

**Quick start:**
```python
from structural_cahn_hilliard_3d import StructuralCahnHilliard3D, CahnHilliardConfig

cfg    = CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-5, laplacian="conv3d")
solver = StructuralCahnHilliard3D(cfg).to(device)

u      = torch.rand(64, 64, 64, device=device) * 0.2 - 0.1   # initial condition
u_next = solver.step(u)                                        # one time step

# Run 500 steps with energy / mass logging
u_final, history = solver.evolve(u, n_steps=500, log_interval=50)

# Lyapunov energy
E = solver.structural_energy(u, sigma=None)

# Attach SSC adapter
from one_core_fold import CahnHilliardSSCAdapter
solver.attach_ssc(CahnHilliardSSCAdapter())
```

**Thin-film and PFC variants:**
```python
from structural_cahn_hilliard_3d import (
    ThinFilmStructuralCahnHilliard3D,
    PhaseFieldCrystal3D,
    CahnHilliardConfig,
)

# Thin-film with surface diffusion
tf_cfg = CahnHilliardConfig(thin_film=True, surface_diffusion=True, kappa_s=0.01)
tf_solver = ThinFilmStructuralCahnHilliard3D(tf_cfg).to(device)

# Phase-field crystal (r < 0 → crystalline phase)
pfc_cfg = CahnHilliardConfig(pfc_r=-0.5, ssc_stabilise=True)
pfc_solver = PhaseFieldCrystal3D(pfc_cfg).to(device)
```

**Role in REAL FOLD ONE:** Provides mesoscale spatial context via
`FoldCahnHilliardBridge`. The CH phase field encodes protein condensate /
liquid-liquid phase separation (LLPS) behaviour — phenomena invisible to
single-scale atomic MD. The σ field from the CH solver modulates
per-atom Langevin noise in `RefinementEngine`, closing a fully differentiable
multi-scale loop.

---

### 3. `structural_langevin_fold_v2.py` — BAOAB Structural Langevin Integrator

**Role:** Fully differentiable Langevin molecular dynamics integrator implementing
the BAOAB splitting scheme with all four Structural Calculus extensions.

**Integration scheme (BAOAB):**

```
B  : v ← v + (dt/2m) · F_bulk
A  : x ← x + (dt/2) · v
O  : v ← e^{-γΔt} v + sqrt(1−e^{-2γΔt}) · ξ_structural
A  : x ← x + (dt/2) · v
B  : v ← v + (dt/2m) · (F_bulk + F_jump)
```

where ξ_structural is the Structural Itô noise with FiLM-modulated amplitude at
interface regions, and γ is the CSOC adaptive friction.

**Classes:**

| Class | Base | Description |
|-------|------|-------------|
| `InterfaceDetector` | `InterfaceDetectorBase` | Soft differentiable interface mask from pairwise distances. |
| `CSOCThermostat` | `CSOCBase` | CSOC adaptive temperature T(σ) and friction γ(σ). |
| `StructuralItoNoise` | `StructuralItoBase` | Multiplicative noise with Itô drift correction. Amplifies stochasticity near interfaces. |
| `AdvancedStructuralLangevin` | `nn.Module` | Full BAOAB integrator composing the three modules above. |

**Quick start — full simulation loop:**
```python
from structural_langevin_fold_v2 import AdvancedStructuralLangevin

integrator = AdvancedStructuralLangevin(
    mass=12.0, dt=0.002, base_temp=300.0, base_friction=1.0
)

# Detailed BAOAB loop (explicit force re-evaluation at mid-step)
for step in range(n_steps):
    force_bulk     = -torch.autograd.grad(energy, coords)[0]
    interface_mask = integrator.interface_detector(coords)

    x_new, v_tilde, T, sigma = integrator.baoa_step(
        coords, velocities, force_bulk, jumps, interface_mask
    )
    new_energy = potential(x_new)
    new_force  = -torch.autograd.grad(new_energy, x_new)[0]
    velocities = integrator.final_b_step(v_tilde, new_force, jumps, interface_mask)
    coords     = x_new.detach().requires_grad_(True)

# Convenience single-call wrapper
coords, velocities, T, sigma = integrator.full_step(coords, velocities, force_fn)
```

**Via LangevinBridge (recommended with RefinementEngine):**
```python
from one_core_fold import LangevinBridge

bridge = LangevinBridge(engine, integrator)
bridge.set_alpha(alpha)
coords, velocities = bridge.run(coords, n_steps=1000, log_every=100)
```

---

### 4. `real_fold_one_v2.py` — Full-Atom Differentiable Refinement Engine

**Role:** End-to-end differentiable protein and nucleic acid structure refinement.
Combines OpenMM-ML (TorchForce native autograd), self-organised criticality (SOC),
multiscale coarse-graining, and multi-scale CH coupling into a single PyTorch workflow.

**Native differentiability via OpenMM-ML:**

Previous versions injected OpenMM forces as numpy arrays (first-order only).
This version compiles the force field into a TorchScript `TorchForce` module:

```python
coords.requires_grad_(True)
E = openmm_solute_energy(coords, calculator)   # entirely within autograd graph
E.backward()                                   # analytical ∂E/∂pos
H = torch.autograd.functional.hessian(...)     # Hessians work correctly
```

**Supported ML potentials** (`RefinementConfig.ml_potential`):

| Value | Model | Elements |
|-------|-------|----------|
| `"ani2x"` (default) | ANI-2x neural network | C, H, N, O, S, F, Cl |
| `"ani1ccx"` | ANI-1ccx (coupled-cluster accuracy) | C, H, N, O |
| `"mace-mp-0"` | MACE foundation model | All elements |
| `"aimnet2"` | AIMNet2 | C, H, N, O + halogens |
| `None` | Classical AMBER (fallback) | Standard biopolymers |

**Key classes:**

| Class | Description |
|-------|-------------|
| `RefinementConfig` | Dataclass: device, force-field, SOC weights, Langevin settings, OpenMM-ML options. |
| `OpenMMSystemBuilder` | Builds OpenMM `System` + `Topology` from PDB with AMBER ff14SB / OL15 / GAFF2. Supports implicit/explicit solvent and ligands. |
| `OpenMMEnergyCalculator` | Wraps OpenMM simulation; exposes `TorchForce`-compiled energy for autograd. |
| `SOCController` | Inherits `CSOCBase`. Computes structural stress σ, adaptive temperature, SOC interaction energy, and avalanche gradient. |
| `DiffRGRefiner` | Differentiable renormalisation-group coarse-graining (configurable factor, levels). |
| `MDEngine` | Autonomous Langevin or gradient-descent MD loop with trajectory logging. |
| `RefinementEngine` | Main engine: OpenMM energy + SOC + RG coarse-graining + LangevinBridge + CH coupling. |
| `Trainer` | Multi-structure training loop with DDP support. |

**Quick start:**
```python
from real_fold_one_v2 import RefinementConfig, RefinementEngine

cfg = RefinementConfig(
    ml_potential="ani2x",
    use_native_autograd=True,
    base_temp=300.0,
    steps=600,
    use_langevin=True,
    use_rg=True,
)
engine = RefinementEngine(cfg)

# Refine a structure from PDB
result = engine.refine("my_protein.pdb")
# result contains: coords, energy_trajectory, sigma_trajectory, ddg

# Trainer for multi-structure training
from real_fold_one_v2 import Trainer
trainer = Trainer(cfg)
trainer.train(["1abc.pdb", "2xyz.pdb"], epochs=100, use_ddp=False)
```

**RefinementConfig key fields:**

```python
RefinementConfig(
    device              = "auto",          # "cuda" / "cpu" / "auto"
    ml_potential        = "ani2x",         # ML potential (see table above)
    use_native_autograd = True,            # TorchForce native autograd
    base_temp           = 300.0,           # K
    steps               = 600,            # optimisation steps
    use_langevin        = False,           # enable Euler-Maruyama (simple)
    use_rg              = False,           # renormalisation-group coarse-graining
    rg_factor           = 4,
    use_ssc             = True,            # SSC EMA filter
    w_soc               = 0.3,            # SOC energy weight
    implicit_solvent    = "OBC",           # None / "OBC" / "GBn2"
    gradient_checkpointing = False,
)
```

---

### 5. `real_fold_one_ht_v2.py` — High-Throughput Mutation & Epistasis Scanner

**Role:** Ultra-fast scanning of single and double mutations across proteins, DNA,
RNA, and multimers. Extends `RefinementEngine` with coarse-grained ΔΔG estimation
and batch-parallelised evaluation.

**Key classes:**

| Class | Description |
|-------|-------------|
| `HTRefinementEngine` | Subclass of `RefinementEngine`. Adds `relax_local()` (differentiable local window relaxation), `compute_ddg()`, `scan_single()`, `scan_double()`. |
| `HTConfig` | Dataclass for scanning parameters (PDB, output dir, GPU count, scan type, etc.). |
| `HighThroughputScanner` | Orchestrator: loads structure, dispatches scans, collects results, exports CSV/JSON, generates plots. |

**Supported scan modes:**

| Mode | CLI flag | Description |
|------|----------|-------------|
| Full single scan | `--scan` | All positions × all allowed monomers. |
| Targeted single | `--mutlist mutations.json` | User-supplied list of `[chain, pos, residue]`. |
| Epistasis (double) | `--epistasis` | Random or user-supplied residue pairs. |
| Single point | `--single 0:5:ALA` | One specified mutation. |

**Command-line usage:**
```bash
# Full single-mutation scan
python real_fold_one_ht_v2.py --pdb 1abc.pdb --scan --output ht_results

# Targeted mutation list (JSON)
python real_fold_one_ht_v2.py --pdb 1abc.pdb --mutlist mutations.json

# Double-mutation epistasis (up to 2000 pairs)
python real_fold_one_ht_v2.py --pdb 1abc.pdb --epistasis --max_epi 2000

# Single point mutation
python real_fold_one_ht_v2.py --pdb 1abc.pdb --single 0:5:ALA

# Resume interrupted scan
python real_fold_one_ht_v2.py --pdb 1abc.pdb --scan --resume
```

**Python API:**
```python
from real_fold_one_ht_v2 import HTConfig, HighThroughputScanner

cfg     = HTConfig(pdb_file="1abc.pdb", scan_full=True, output_dir="./scan_out",
                   num_gpus=2, relaxation_steps=30, relaxation_window=3)
scanner = HighThroughputScanner(cfg)
results = scanner.run()        # returns DataFrame: position, mutation, ddg
scanner.export_csv()
scanner.plot_landscape()       # publication-quality ΔΔG landscape
scanner.plot_epistasis()       # additivity scatter and epistasis distribution
```

**Outputs per run:**
- `results.csv` — position, chain, mutation, ΔΔG
- `landscape.png` — positional ΔΔG heatmap
- `tolerance.png` — per-position stability profile
- `epistasis.png` — double-mutation additivity scatter
- `checkpoint.json` — resumable scan state

---

### 6. `structural_gno_fold_v3.py` — AI Surrogate & Training Engine

**Role:** Production-grade Graph Neural Operator (GNO) that learns to replace
expensive physics simulation with O(1) inference. Serves as both the AI surrogate
for the REAL FOLD ONE ecosystem and the training engine that generates and consumes
data from all five physics modules.

**Two operator mappings:**

| Mode | Input | Output |
|------|-------|--------|
| Protein | (seq_features, init_coords, σ) | (final_coords, ΔΔG) |
| Phase-field | (u_init, σ_3D) | u_pred (one-shot CH/PFC evolution) |

Both modes share the same FiLM-modulated `StructuralMessagePassing` backbone,
ensuring σ-consistent structural calculus throughout.

**Key classes:**

| Class | Description |
|-------|-------------|
| `SGNOConfig` | Validated hyperparameter dataclass: architecture, graph cutoffs, LR, EMA, checkpointing. |
| `StructuralMessagePassing` | FiLM-modulated message-passing layer: γ(σ)·aggregated + β(σ). Replaces the sigmoid gate of v2. |
| `AttentionPooling` | Soft-attention graph pooling for ΔΔG head. More expressive than mean pooling for energetically localised mutations. |
| `StructuralGNOFold` | Main model: two forward modes, FiLM backbone, coord/ddg/phase heads. |
| `EMAWrapper` | Exponential moving average of model parameters for inference stability. |
| `GradMonitor` | Per-layer gradient norm logging via backward hooks. |
| `SGNODataset` | PyTorch `Dataset` wrapping `RefinementEngine` or `HighThroughputScanner` outputs. |
| `SGNOTrainer` | Production training loop: separate LR groups, OneCycleLR, EMA, physics-informed losses, checkpoint save/load. |
| `SGNOEvaluator` | Evaluation: RMSD, Pearson-r ΔΔG, CH energy monotonicity fraction, mass conservation error. |

**Quick start:**
```python
from structural_gno_fold_v3 import build_trainer_from_ecosystem, SGNOConfig

cfg     = SGNOConfig(hidden_dim=128, num_layers=6, max_epochs=200)
trainer = build_trainer_from_ecosystem(cfg=cfg)   # attaches CH3D solver automatically

# Protein training step
metrics = trainer.train_step_protein(
    seq_feats, init_coords, final_coords, ddg_tensor, sigma
)
# metrics = {"total": ..., "coords": ..., "ddg": ...}

# Phase-field training step
metrics = trainer.train_step_phase_field(u_init, u_future, sigma_3d)
# metrics = {"total": ..., "data": ..., "physics": ..., "mass": ...}

# Save / resume
trainer.save_checkpoint("./checkpoints/epoch_010.pt")
trainer.load_checkpoint("./checkpoints/epoch_010.pt")
```

**Physics-informed losses:**

| Loss term | Formula | Weight |
|-----------|---------|--------|
| Coordinate MSE | MSE(pred_coords, true_coords) | 1.0 |
| ΔΔG MSE | MSE(pred_ddg, true_ddg) | `lambda_ddg` (0.1) |
| Lyapunov penalty | ReLU(E_pred − E_init) | `lambda_physics` (0.05) |
| Mass conservation | |mass_pred − mass_init| | `lambda_mass` (0.01) |

**Evaluator usage:**
```python
from structural_gno_fold_v3 import SGNOEvaluator

evaluator = SGNOEvaluator(model, ema=trainer.ema, device=device, ch_solver=ch_solver)

# Protein mode
results = evaluator.evaluate_protein(val_samples)
# {"rmsd_mean": ..., "rmsd_std": ..., "pearson_r_ddg": ...}

# Phase-field mode
results = evaluator.evaluate_phase_field(pf_samples)
# {"mse": ..., "energy_monotone_frac": ..., "mass_rel_error_mean": ...}
```

---

## Cross-Ecosystem Integration Patterns

### Pattern 1 — Full multi-scale refinement with BAOAB + CH coupling

```python
from one_core_fold import LangevinBridge, FoldCahnHilliardBridge
from real_fold_one_v2 import RefinementEngine, RefinementConfig
from structural_langevin_fold_v2 import AdvancedStructuralLangevin
from structural_cahn_hilliard_3d import StructuralCahnHilliard3D, CahnHilliardConfig

# Build components
engine     = RefinementEngine(RefinementConfig(use_langevin=False))
integrator = AdvancedStructuralLangevin(dt=0.002, base_temp=300.0)
bridge     = LangevinBridge(engine, integrator)

ch_solver  = StructuralCahnHilliard3D(
    CahnHilliardConfig(dx=1.0, epsilon=1.5, dt=1e-5, laplacian="fft")
).to(device)
ch_bridge  = FoldCahnHilliardBridge(grid_size=(32, 32, 32)).to(device)

# Multi-scale loop
for step in range(n_steps):
    # 1. Molecular scale: BAOAB step
    bridge.set_alpha(alpha)
    coords, velocities = bridge.step(coords, velocities)

    # 2. Mesoscale: update phase field from atomic positions
    u       = ch_bridge.coords_to_field(coords)
    u       = ch_solver.step(u)

    # 3. Back-couple: CH σ modulates per-atom noise in next Langevin step
    G_scale = ch_bridge.sigma_to_atom_scale(coords, sigma_field=u, amp=0.5)
```

### Pattern 2 — HT scan feeding GNO training data

```python
from real_fold_one_ht_v2 import HTConfig, HighThroughputScanner
from structural_gno_fold_v3 import SGNODataset, SGNOTrainer, build_trainer_from_ecosystem

# Generate training data from HT scanner
scanner = HighThroughputScanner(HTConfig(pdb_file="1abc.pdb", scan_full=True))
df      = scanner.run()           # ΔΔG table

# Build dataset and trainer
samples = scanner.to_sgno_samples()   # converts to SGNODataset format
dataset = SGNODataset(samples, mode="protein")
trainer = build_trainer_from_ecosystem()

# Train surrogate
for epoch in range(cfg.max_epochs):
    trainer.run_epoch_protein(DataLoader(dataset), epoch)
    if epoch % cfg.save_every == 0:
        trainer.save_checkpoint()
```

### Pattern 3 — Standalone CH3D as PDE solver (no protein dependency)

```python
from structural_cahn_hilliard_3d import PhaseFieldCrystal3D, CahnHilliardConfig

cfg    = CahnHilliardConfig(pfc_r=-0.5, laplacian="fft", ssc_stabilise=True)
solver = PhaseFieldCrystal3D(cfg).to(device)

u       = 0.1 * torch.randn(64, 64, 64, device=device)
u_final, history = solver.evolve(u, n_steps=2000, log_interval=100)
```

---

## Installation & Requirements

### Core dependencies

```bash
pip install torch torchvision          # PyTorch (>= 2.0 recommended)
pip install numpy scipy networkx
pip install biotite rdkit-pypi
pip install pandas matplotlib
```

### OpenMM-ML (for native autograd in RefinementEngine)

```bash
conda install -c conda-forge openmm openmm-ml openmm-torch
# Then install ML potentials as needed:
pip install torchani            # ANI-2x, ANI-1ccx
pip install mace-torch          # MACE-MP-0
pip install aimnet2             # AIMNet2
```

### Optional (HT scanner)

```bash
pip install torch-cluster       # fast neighbour lists
pip install openff-toolkit openmmforcefields  # ligand parameterisation
```

All modules degrade gracefully when optional dependencies are absent —
`try/except ImportError` blocks guard every cross-ecosystem import.

---

## Dependency Graph (import flow)

```
torch  ──────────────────────────────────────────────────────────────────┐
                                                                          │
one_core_fold.py                                                          │
  (SemanticStateContraction, CSOCBase, LangevinBridge,                   │
   FoldCahnHilliardBridge, CahnHilliardSSCAdapter)                       │
         │                                                                │
         ├──→ structural_cahn_hilliard_3d.py  (torch only, standalone)   │
         │                                                                │
         ├──→ structural_langevin_fold_v2.py                              │
         │         (InterfaceDetector, CSOCThermostat,                    │
         │          StructuralItoNoise, AdvancedStructuralLangevin)       │
         │                                                                │
         └──→ real_fold_one_v2.py                                         │
                  (OpenMMSystemBuilder, SOCController,                    │
                   RefinementEngine, Trainer)                             │
                         │                                                │
                         ├──→ real_fold_one_ht_v2.py                      │
                         │         (HTRefinementEngine,                   │
                         │          HighThroughputScanner)                │
                         │                                                │
                         └──→ structural_gno_fold_v3.py  ←───────────────┘
                                   (StructuralGNOFold,
                                    SGNOTrainer, SGNOEvaluator)
```

No circular imports. `structural_cahn_hilliard_3d.py` imports only from
`torch`; all cross-ecosystem coupling is mediated by bridge classes in
`one_core_fold.py`.

---

## Quick Reference — All Public APIs

### one_core_fold

```python
FOLD_VERSION                          # str: "1.0.0"
SemanticStateContraction(alpha)       # EMA σ filter
CSOCBase(sigma_target, epsilon_fp, boost_factor)  # abstract CSOC base
LangevinBridge(engine, langevin)      # .step() .run() .set_alpha()
FoldCahnHilliardBridge(grid_size)     # .coords_to_field() .field_to_sigma() .sigma_to_atom_scale()
CahnHilliardSSCAdapter()              # attach to CH3D via solver.attach_ssc()
get_device(preference)                # returns torch.device
```

### structural_cahn_hilliard_3d

```python
CahnHilliardConfig(dx, epsilon, dt, laplacian, ...)
StructuralCahnHilliard3D(cfg)         # .step() .evolve() .structural_energy() .total_mass()
ThinFilmStructuralCahnHilliard3D(cfg) # + .get_thin_film_mobility() .thin_film_energy()
PhaseFieldCrystal3D(cfg)              # + .compute_pfc_chemical_potential() .pfc_energy()
structural_biharmonic_n(field, sigma, n, laplacian_fn)  # Δ_S^n u utility
```

### structural_langevin_fold_v2

```python
AdvancedStructuralLangevin(mass, dt, base_temp, base_friction, ...)
  .baoa_step(coords, velocities, force, jumps, interface_mask)
  .final_b_step(v_tilde, new_force, jumps, interface_mask)
  .full_step(coords, velocities, force_fn, jumps)
  .reset()
```

### real_fold_one_v2

```python
RefinementConfig(...)                 # dataclass — all hyperparameters
RefinementEngine(cfg)                 # .refine(pdb_path) .forward()
Trainer(cfg)                          # .train(pdb_list, epochs, use_ddp)
MDEngine(cfg)                         # .run(coords, n_steps)
openmm_solute_energy(coords, calc)    # differentiable energy scalar
```

### real_fold_one_ht_v2

```python
HTConfig(pdb_file, scan_full, output_dir, num_gpus, ...)
HighThroughputScanner(cfg)
  .run()                              # → pandas DataFrame
  .export_csv()
  .plot_landscape()
  .plot_epistasis()
HTRefinementEngine(cfg)               # .relax_local() .compute_ddg() .scan_single()
```

### structural_gno_fold_v3

```python
SGNOConfig(hidden_dim, num_layers, lr_backbone, lr_heads, ...)
StructuralGNOFold(cfg)
  .forward(seq_features, init_coords, sigma)          # → (coords, ddg)
  .forward_phase_field(u_init, sigma_3d)              # → u_pred
EMAWrapper(model, decay)              # .update() .average_parameters()
GradMonitor(model)                    # .report(epoch)
SGNODataset(samples, mode)            # PyTorch Dataset
SGNOTrainer(model, cfg, device, ch_solver)
  .train_step_protein(...)
  .train_step_phase_field(...)
  .run_epoch_protein(loader, epoch)
  .run_epoch_phase_field(loader, epoch)
  .save_checkpoint(path)
  .load_checkpoint(path)
SGNOEvaluator(model, ema, device, ch_solver)
  .evaluate_protein(samples)          # → {"rmsd_mean", "rmsd_std", "pearson_r_ddg"}
  .evaluate_phase_field(samples)      # → {"mse", "energy_monotone_frac", "mass_rel_error_mean"}
build_trainer_from_ecosystem(cfg, device_str, ch_cfg)  # factory
```

---

## License

MIT License — Yoon A Limsuwan / MSPS NETWORK, 2026.

All six files are released under the MIT License. Downstream dependencies
(OpenMM, biotite, RDKit, PyTorch, etc.) carry their own open-source licences
as noted in the individual file headers.

---

*README_PLUS.md — REAL FOLD ONE Ecosystem v1.0.0*
*Prepared with AI assistance from Claude (Anthropic), GPT (OpenAI), and Gemini (Google).*
