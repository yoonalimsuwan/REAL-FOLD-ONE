`
# REAL FOLD ONE

**SOC‑Controlled Universal Refinement & High‑Throughput Mutation Scanning Suite**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A unified physics‑based framework for macromolecular refinement and mutational scanning.
Built around a novel **Self‑Organised Criticality (SOC) controller**, it refines proteins, DNA, RNA,
and their complexes using a fully differentiable energy function, then scales to thousands of
*in silico* mutations across multiple GPUs — all without writing a single line of CUDA C++.

---

## Table of Contents

- [Architectural Philosophy](#architectural-philosophy)
- [Overview](#overview)
- [Key Features](#key-features)
- [Integration with Structure Predictors](#integration-with-structure-predictors)
  - [Natural Proteins](#natural-proteins)
  - [*De Novo* Designed Proteins](#de-novo-designed-proteins)
  - [Synthetic Proteins & Ligand Complexes](#synthetic-proteins--ligand-complexes)
  - [Typical Refinement Pipeline (AF3 → REAL FOLD ONE → OpenMM)](#typical-refinement-pipeline-af3--real-fold-one--openmm)
- [Installation](#installation)
- [Quick Start – Refinement Engine](#quick-start--refinement-engine)
- [High‑Throughput Mutation Scanning (HT Module)](#high-throughput-mutation-scanning-ht-module)
  - [Targeted vs Full Scan](#targeted-vs-full-scan)
  - [Epistasis Analysis](#epistasis-analysis)
  - [Workflow: Targeted → Global Optimization](#workflow-targeted--global-optimization)
- [Training the SOC Kernel](#training-the-soc-kernel)
- [Antibody CDR Modelling](#antibody-cdr-modelling)
- [DNA Origami Design](#dna-origami-design)
- [Validation & Testing](#validation--testing)
- [Performance Tips](#performance-tips)
- [Citing REAL FOLD ONE](#citing-real-fold-one)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Architectural Philosophy

REAL FOLD ONE is designed with **strategic autonomy** and **hardware democratisation** at its core.
Rather than hand‑crafting vendor‑specific CUDA kernels, the entire engine is written in
**PyTorch’s high‑level tensor primitives** — matrix operations, `torch.cdist`, `torch.fft`,
automatic differentiation. This architectural choice delivers three decisive advantages:

### 1. True Vendor Neutrality

- No `nvcc` dependency, no embedded CUDA C++.  
- PyTorch’s own runtime backends — **CUDA**, **MPS** (Apple Silicon), **`torch_npu`** (Huawei Ascend),
  **CPU** — are the only hardware abstraction layer.  
- When a chip vendor improves its PyTorch backend, **REAL FOLD ONE instantly benefits without
  a single line of code being changed**.

### 2. Automatic Kernel Fusion

- PyTorch’s graph compilers (`torch.compile`, TorchDynamo, Inductor) aggressively fuse chains of
  small operations — distance calculations, masking, scatter‑adds — into single on‑chip kernels.  
- This drastically reduces memory bandwidth pressure, the dominant bottleneck in GPU‑accelerated
  energy functions.  
- Consequently, performance on non‑NVIDIA hardware (Ascend NPU, Apple MPS) approaches or matches
  hand‑tuned CUDA implementations, while remaining 100 % portable.

### 3. Democratisation of Science

- The engine runs **unchanged** on a 3 GB‑RAM CPU, an Apple M1 laptop, a Colab T4 GPU, a
  multi‑GPU Ascend cluster, or a high‑end DGX workstation.  
- Switching hardware is a single command‑line flag: `--device cpu`, `--device cuda`, `--device mps`,
  `--device npu`.  
- This eliminates the traditional “rich lab / poor lab” divide, making cutting‑edge physics‑based
  refinement accessible to researchers everywhere, regardless of their hardware supply chain.

In a world of shifting semiconductor alliances, REAL FOLD ONE represents a **new generation of
biomolecular software** — one that refuses to be locked into a single vendor’s ecosystem.

---

## Overview

REAL FOLD ONE provides two tightly integrated modules:

| Module | File | Purpose |
|--------|------|---------|
| **Core Refinement Engine** | `real_fold_one.py` | Full‑atom refinement, training, antibody, origami |
| **HT Mutation Scanner** | `real_fold_one_ht.py` | High‑throughput ΔΔG and epistasis scanning (recommended) |

A legacy scanner (`real_fold_one_hts.py`) is also included for reference; however, **HT** is the
recommended, production‑ready version with checkpointing, resume, and safe multi‑GPU support.
Both modules share the same physics backend and can be run on CPU, single GPU, or multiple GPUs.

---

## Key Features

### Refinement Engine (`real_fold_one.py`)

- **SOC Controller** – learnable CSOC kernel and Semantic‑State Contraction (SSC) low‑pass filter
  adaptively tune temperature and friction during Langevin dynamics.
- **Multiscale Refinement** – RG coarse‑graining periodically removes high‑frequency noise.
- **Full‑Atom Physics**
  - Proteins: AMBER ff14SB‑like Lennard‑Jones parameters and partial charges for all side‑chain atoms.
  - DNA/RNA: OL15‑like parameters for all nucleotides (C4′, phosphate, base).
  - Ligands: GAFF2 force field with automatic atom‑type assignment and topology generation.
  - Antibodies: specialised Rosetta‑like scoring for antigen‑antibody interfaces.
- **Advanced Electrostatics** – Sparse PME, geometric multigrid Poisson solver, block‑wise
  multipole long‑range correction.
- **Hierarchical Neighbor Lists** – separate cutoffs for clash (2.5 Å), LJ (6 Å), and electrostatics
  (12 Å) for efficiency.
- **DNA Origami** – wireframe routing, staple design, full‑atom PDB export, and oxDNA format.
- **Itô SDE** – Milstein scheme for the Langevin equation, with Malliavin sensitivity for
  parameter Greeks.
- **Simulated Annealing** – optional temperature schedule from 1000 K to 300 K.
- **Scalable** – chunked O(N) graph building supports >100 000 residues.
- **Environment‑Adaptive** – works on CPU (3 GB RAM), Colab T4, single GPU, or multi‑GPU (DDP).

### High‑Throughput Mutation Scanner (`real_fold_one_ht.py`)

- **Full Single‑Mutation Scan** – every residue → every allowed monomer.
- **Targeted Mutation Lists** – provide a JSON file of specific mutations to evaluate.
- **Double‑Mutant Epistasis** – random sampling or user‑supplied pairs.
- **Local Relaxation Window** – fast ΔΔG estimation by relaxing only the mutation site.
- **Multi‑GPU Parallelism** – `torch.multiprocessing` pool with automatic checkpointing and resume.
- **Publication‑Ready Plots** – ΔΔG distribution, mutational landscape heatmap, position‑tolerance
  profile, epistasis distribution, additivity scatter.

### Training & Validation

- **Train SOC kernel** on native structures to learn optimal interaction parameters.
- **Gradient validation** (finite‑difference check).
- **RMSD** calculation with Kabsch superposition.

---

## Integration with Structure Predictors

REAL FOLD ONE is designed as a **post‑processing refinement engine** for any structure predictor
(AlphaFold 3, ESMFold, Rosetta, etc.). It does **not** predict structures from sequence; instead,
it takes initial Cα coordinates and uses its physics‑based energy function to:

- Correct local strain and steric clashes.
- Optimise hydrogen‑bond networks and electrostatics.
- Rebuild full side‑chain and nucleic acid conformations.

### Natural Proteins

```text
Sequence → Predictor (e.g. AlphaFold 3) → Cα model → REAL FOLD ONE refine → Full‑atom refined structure
```

After running a prediction (e.g., AlphaFold‑Multimer), simply pass the resulting PDB to
real_fold_one.py refine.  The engine extracts Cα atoms, reconstructs all backbone and
side‑chain atoms, and minimises the SOC‑weighted energy.  This often improves clash scores,
MolProbity statistics, and ligand‑binding pocket geometries.

De Novo Designed Proteins

For proteins designed de novo (e.g., RFdiffusion, ProteinMPNN), the design pipeline
provides a backbone model. REAL FOLD ONE can:

· Accept an arbitrary Cα trace (even idealised secondary‑structure fragments).
· Build full atoms de novo using only the Cα positions.
· Refine the structure to relax backbone strain while preserving the intended fold.

This enables designers to evaluate the “foldability” of a novel backbone and to inspect
packing quality before experimental characterisation.

Synthetic Proteins & Ligand Complexes

When working with non‑canonical amino acids, post‑translational modifications, or
protein‑ligand complexes, REAL FOLD ONE supports:

· Loading ligand files (SDF, MOL2, PDB) via --ligand.
· Automatically assigning GAFF2 types and generating bonded topology.
· Computing protein‑ligand interaction energy during refinement.

Thus, synthetic biology designs can be screened for stability and binding affinity
directly within the refinement workflow.

Typical Refinement Pipeline (AF3 → REAL FOLD ONE → OpenMM)

For production‑grade Molecular Dynamics (MD), REAL FOLD ONE acts as the essential
pre‑conditioner that bridges the gap between a predicted structure and a long‑term
explicit‑solvent simulation.

```
┌─────────────┐      ┌──────────────────┐      ┌────────────┐
│ AlphaFold 3 │ ──► │  REAL FOLD ONE   │ ──► │  OpenMM    │
│  (or any    │      │   refine /       │      │  (explicit │
│  predictor) │      │   full_atom      │      │   solvent  │
│             │      │   export)        │      │   MD)      │
└─────────────┘      └──────────────────┘      └────────────┘
```

Why this step is necessary:

· AF3 models are often near‑native but may still contain minor steric clashes or
  sub‑optimal side‑chain rotamers that would cause the MD integrator to “blow up”
  in the first few femtoseconds.
· REAL FOLD ONE’s SOC‑driven avalanche relaxation and full‑atom energy
  minimisation gently resolve these hot‑spots without distorting the overall
  fold, producing a clean, physics‑ready starting structure.
· The resulting PDB (full‑atom) can be directly loaded into OpenMM (or any MD engine)
  for solvation, equilibration, and production runs.

Complete command‑line example:

```bash
# Step 1: Obtain predicted structure from AlphaFold 3 (e.g., folded_model.pdb)

# Step 2: Refine and rebuild full atoms with REAL FOLD ONE
python real_fold_one.py refine \
    --input folded_model.pdb \
    --output refined_full.pdb \
    --steps 500 \
    --pme \
    --full_atom \
    --gpu

# Step 3: (Optional) Validate the refined structure
python real_fold_one.py test

# Step 4: Prepare OpenMM simulation using the refined structure as input
# (example using OpenMM’s Python API)
python -c "
import openmm.app as app
import openmm as mm
from openmm import unit

pdb = app.PDBFile('refined_full.pdb')
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=1.0*unit.nanometers,
                                 constraints=app.HBonds)
integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()  # optional, usually not needed after REAL FOLD ONE
simulation.reporters.append(app.PDBReporter('output.pdb', 1000))
simulation.step(100000)  # 200 ps
"
```

The refinement step ensures that the MD run starts from an energetically relaxed state,
allowing larger integration time‑steps and avoiding early crashes.

---

Installation

```bash
git clone https://github.com/yoonalimsuwan/REAL-FOLD-ONE.git
cd real-fold-one

# Create a fresh environment (optional)
conda create -n realfold python=3.10 -y
conda activate realfold

# Install core dependencies
pip install torch numpy pandas tqdm

# Optional: faster neighbor lists
pip install torch-cluster

# Optional: structure I/O and plotting
pip install biotite matplotlib seaborn networkx
```

---

Quick Start – Refinement Engine

```bash
# Basic refinement of a protein from PDB
python real_fold_one.py refine --input 1abc.pdb --output refined.pdb --steps 300

# Full refinement with PME, GPU, and full‑atom export
python real_fold_one.py refine -i 1abc.pdb -o refined_full.pdb --steps 500 --gpu --pme --full_atom

# Refine a DNA duplex (from PDB or ideal helix)
python real_fold_one.py refine -i dna.pdb -o dna_refined.pdb --steps 200

# Include a ligand file
python real_fold_one.py refine -i protein.pdb --ligand ligand.sdf ligand2.mol2 -o complex.pdb --full_atom

# Run a quick gradient validation test
python real_fold_one.py test
```

Detailed refine Options

Flag Description
--input, -i Input PDB or mmCIF file (Cα atoms extracted)
--chain Specific chain ID to refine
--output, -o Output PDB file (CA‑only unless --full_atom)
--steps Number of refinement steps (default: 600)
--lr Learning rate for Adam (default: 1e-4)
--pme Use Particle‑Mesh Ewald
--multigrid Use geometric multigrid Poisson solver
--block_lr Add block‑wise multipole long‑range correction
--ligand One or more ligand files (SDF, MOL2, PDB)
--full_atom Export full‑atom PDB (sidechains reconstructed)
--trajectory Save all‑step Cα trajectory as .npy
--device cpu, cuda, or auto (default)
--milstein Use Milstein scheme for the Langevin SDE
--no_rg Disable RG coarse‑graining
--no_ssc Disable Semantic‑State Contraction

---

High‑Throughput Mutation Scanning (HT Module)

The real_fold_one_ht.py module is the recommended tool for all in silico mutagenesis work.
It supports both targeted scanning (a user‑supplied list of mutations) and full‑scan
(all possible single mutations), as well as double‑mutant epistasis analysis.

Targeted vs Full Scan

· Full scan (--scan): evaluates every position × every allowed monomer automatically.
· Targeted list (--mutlist my_muts.json): evaluates only the mutations specified in a
  JSON file, saving time and resources when you know which residues are of interest.

Epistasis Analysis

Use --epistasis to perform double‑mutant scanning. Pairs can be supplied explicitly
(--epipairs pairs.json) or sampled randomly (--max_epi). The module first looks for
pre‑computed single‑mutation results (via --resume) to calculate additive ΔΔG, then
computes only the double‑mutant energy, giving you the epistasis value:

```
ε = ΔΔG_double – (ΔΔG_single1 + ΔΔG_single2)
```

Workflow: Targeted → Global Optimization

A particularly powerful strategy is to use HT in two sequential steps — first targeted,
then global — to perform lead optimisation of a protein variant:

```bash
# Step 1: Targeted scan on residues of known importance
python real_fold_one_ht.py --pdb wildtype.pdb --mutlist my_targets.json --output step1_out

# Step 2: Take the best variant from Step 1 and run a full scan
#         to discover additional stabilising mutations on the new energy landscape
python real_fold_one_ht.py --pdb step1_out/best_variant.pdb --scan --output step2_out
```

This approach efficiently searches for synergistic (epistatic) interactions without
exhaustively enumerating all possible double mutants.

Full command‑line options for HT:

```bash
# Full single‑mutation scan
python real_fold_one_ht.py --pdb 1abc.pdb --scan --output ht_out

# Scan a DNA sequence (ideal helix generated automatically)
python real_fold_one_ht.py --seq "ATGCGTACGTAG" --scan --output dna_scan

# Multi‑GPU with resume
python real_fold_one_ht.py --pdb 1abc.pdb --scan --gpu --num_gpus 2 --resume

# Targeted list of mutations (JSON file)
python real_fold_one_ht.py --pdb 1abc.pdb --mutlist my_muts.json --output targeted

# Quick single‑mutation ΔΔG
python real_fold_one_ht.py --pdb 1abc.pdb --single "0:5:A"

# Epistasis scan (random pairs, max 1000)
python real_fold_one_ht.py --pdb 1abc.pdb --epistasis --max_epi 500

# Epistasis from predefined list
python real_fold_one_ht.py --pdb 1abc.pdb --epistasis --epipairs my_pairs.json
```

Input JSON Formats

Mutation list (--mutlist): [[chain_index, pos_in_chain, new_monomer], ...]
Epistasis pairs (--epipairs): [[chain1, pos1, chain2, pos2], ...]

Output Files (HT)

File Content
single_mutations.csv ΔΔG, energies, mutation type
epistasis.csv Additive vs double‑mutant ΔΔG, epistasis
summary.json WT energy, mutation counts
ddg_distribution.png Histogram of ΔΔG values
mutational_landscape.png Position × mutant heatmap
position_profile.png Per‑residue mean ΔΔG ± std
epistasis_distribution.png Histogram of epistasis
additivity_scatter.png Additive vs double ΔΔG scatter

---

Training the SOC Kernel

```bash
# Train kernel on a set of native PDBs
python real_fold_one.py train --input native1.pdb native2.pdb --epochs 100 --output kernel_params.json
```

This minimises the energy of the native structures with respect to the kernel parameters
(alpha, lambda).  The optimised values can then be set in RefinementConfig.

---

Antibody CDR Modelling

```bash
python real_fold_one.py antibody --antigen antigen.pdb --cdr_start 95 --cdr_end 102 --output antibody.pdb
```

Performs fragment‑based loop remodelling of the CDR region and evaluates the interface
energy with the antigen.

---

DNA Origami Design

```bash
# shape.json must contain "vertices" and "edges"
python real_fold_one.py origami --shape shape.json --output my_origami
```

Outputs:

· my_origami.pdb – full‑atom DNA model.
· my_origami.top / .dat – files for oxDNA simulation.
· Optional Batalin–Vilkovisky topological consistency check with --bv_check.

---

Validation & Testing

· Gradient check: python real_fold_one.py test verifies analytical gradients against
  finite differences.
· RMSD: the function compute_rmsd() performs Kabsch alignment and returns the RMSD.
· BV check: for origami, --bv_check verifies the classical master equation.

---

Performance Tips

· Install torch-cluster for GPU‑accelerated neighbour lists.
· For large systems (>10 000 residues), increase --rebuild_interval to 200–400 and keep
  RG enabled.
· Use --full_atom only for final export; during scanning, side‑chains are rebuilt on‑the‑fly.
· For multi‑GPU scanning, set --num_gpus to the number of available devices and ensure
  enough VRAM per GPU (≈3 GB for a 500‑residue protein).
· On CPU, set OMP_NUM_THREADS to the desired number of cores.

---

Citing REAL FOLD ONE

If you use this software, please cite:

```
Yoon A Limsuwan. "REAL FOLD ONE: SOC‑Controlled Universal Refinement Engine."
Zenodo, 2026. DOI: 10.5281/zenodo.XXXXXXX
```

---

License

This project is licensed under the MIT License – see LICENSE for details.

---

Contributing

Contributions are welcome! Please open an issue to discuss proposed changes or submit
a pull request. For major features, we recommend contacting the author first.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/your-username/real-fold-one

```
