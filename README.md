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
- [High‑Throughput Mutation Scanning (HT)](#high-throughput-mutation-scanning-ht)
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
The entire engine is written in **PyTorch’s high‑level tensor primitives** — matrix operations,
`torch.cdist`, `torch.fft`, automatic differentiation. This architectural choice delivers three
decisive advantages:

### 1. True Vendor Neutrality
- No `nvcc` dependency, no embedded CUDA C++.
- PyTorch’s runtime backends — **CUDA**, **MPS** (Apple Silicon), **`torch_npu`** (Huawei Ascend),
  **CPU** — are the only hardware abstraction layer.
- When a chip vendor improves its PyTorch backend, **REAL FOLD ONE instantly benefits without
  changing a single line of code**.

### 2. Automatic Kernel Fusion
- PyTorch’s graph compilers (`torch.compile`, TorchDynamo, Inductor) aggressively fuse chains of
  small operations — distance calculations, masking, scatter‑adds — into single on‑chip kernels.
- This drastically reduces memory bandwidth pressure, the dominant bottleneck in GPU‑accelerated
  energy functions.
- Performance on non‑NVIDIA hardware (Ascend NPU, Apple MPS) approaches or matches hand‑tuned CUDA,
  while remaining 100 % portable.

### 3. Democratisation of Science
- The engine runs **unchanged** on a 3 GB‑RAM CPU, an Apple M1 laptop, a Colab T4 GPU, a
  multi‑GPU Ascend cluster, or a high‑end DGX workstation.
- Switching hardware is a single flag: `--device cpu`, `--device cuda`, `--device mps`, `--device npu`.
- This eliminates the traditional “rich lab / poor lab” divide, making cutting‑edge physics‑based
  refinement accessible to researchers everywhere.

---

## Overview

REAL FOLD ONE consists of two tightly integrated modules:

| Module | File | Purpose |
|--------|------|---------|
| **Core Refinement Engine** | `real_fold_one.py` | Full‑atom refinement, training, antibody, origami |
| **HT Mutation Scanner** | `real_fold_one_ht.py` | High‑throughput ΔΔG and epistasis scanning |

Both modules share the same physics backend and run on CPU, single GPU, or multi‑GPU via
`torch.multiprocessing`.

---

## Key Features

### Refinement Engine (`real_fold_one.py`)

- **SOC Controller** – learnable CSOC kernel and Semantic‑State Contraction (SSC) low‑pass filter
  adaptively tune temperature and friction during Langevin dynamics.
- **Multiscale Refinement** – RG coarse‑graining periodically removes high‑frequency noise.
- **Full‑Atom Physics**
  - Proteins: AMBER ff14SB‑like parameters for all side‑chain atoms.
  - DNA/RNA: OL15‑like parameters for all nucleotides (C4′, phosphate, base).
  - Ligands: GAFF2 force field with automatic atom‑type assignment and topology generation.
  - Antibodies: specialised Rosetta‑like scoring for antigen‑antibody interfaces.
- **Advanced Electrostatics** – Sparse PME, geometric multigrid Poisson solver, block‑wise
  multipole long‑range correction.
- **Hierarchical Neighbor Lists** – separate cutoffs for clash, LJ, and electrostatics.
- **DNA Origami** – wireframe routing, staple design, full‑atom PDB export, and oxDNA format.
- **Itô SDE** – Milstein scheme for the Langevin equation, with Malliavin sensitivity.
- **Simulated Annealing** – optional temperature schedule from 1000 K to 300 K.
- **Scalable** – chunked O(N) graph building supports >100 000 residues.
- **Environment‑Adaptive** – works on CPU (3 GB RAM), Colab T4, single GPU, or multi‑GPU (DDP).

### HT Mutation Scanner (`real_fold_one_ht.py`)

- **Full Single‑Mutation Scan** – every residue → every allowed monomer.
- **Targeted Mutation Lists** – provide a JSON file of specific mutations.
- **Double‑Mutant Epistasis** – random sampling or user‑supplied pairs.
- **Local Relaxation Window** – fast ΔΔG estimation by relaxing only the mutation site.
- **Multi‑GPU Parallelism** – `torch.multiprocessing` pool with checkpoint/resume.
- **Publication‑Ready Plots** – ΔΔG distribution, mutational landscape heatmap, position‑tolerance
  profile, epistasis distribution, additivity scatter.

### Training & Validation

- **Train SOC kernel** on native structures.
- **Gradient validation** (finite‑difference check).
- **RMSD** calculation with Kabsch superposition.

---

## Integration with Structure Predictors

REAL FOLD ONE is a **post‑processing refinement engine** for any structure predictor
(AlphaFold 3, ESMFold, Rosetta, etc.). It takes initial Cα coordinates and:

- Corrects local strain and steric clashes.
- Optimises hydrogen‑bond networks and electrostatics.
- Rebuilds full side‑chain and nucleic acid conformations.

### Natural Proteins

```text
Sequence → Predictor (e.g. AlphaFold 3) → Cα model → REAL FOLD ONE refine → Full‑atom refined structure
```

De Novo Designed Proteins

Accepts an arbitrary Cα trace (even idealised fragments), builds all atoms de novo, and
refines the structure to relax backbone strain while preserving the intended fold.

Synthetic Proteins & Ligand Complexes

Supports non‑canonical amino acids, PTMs, and protein‑ligand complexes via --ligand (SDF,
MOL2, PDB) with automatic GAFF2 typing and topology.

Typical Refinement Pipeline (AF3 → REAL FOLD ONE → OpenMM)

```
┌─────────────┐      ┌──────────────────┐      ┌────────────┐
│ AlphaFold 3 │ ──► │  REAL FOLD ONE   │ ──► │  OpenMM    │
│  (or any    │      │   refine /       │      │  (explicit │
│  predictor) │      │   full_atom      │      │   solvent  │
│             │      │   export)        │      │   MD)      │
└─────────────┘      └──────────────────┘      └────────────┘
```

REAL FOLD ONE’s SOC‑driven avalanche relaxation and full‑atom minimisation gently resolve
steric clashes and sub‑optimal rotamers, producing a clean starting structure for MD and
preventing early “blow‑up”.

Complete example:

```bash
# Obtain predicted structure from AlphaFold 3 (e.g., folded_model.pdb)

# Refine and rebuild full atoms with REAL FOLD ONE
python real_fold_one.py refine \
    --input folded_model.pdb \
    --output refined_full.pdb \
    --steps 500 \
    --pme \
    --full_atom \
    --gpu

# (Optional) Validate
python real_fold_one.py test

# Use the refined structure directly in OpenMM
```

---

Installation

```bash
git clone https://github.com/yoonalimsuwan/REAL-FOLD-ONE.git
cd real-fold-one

conda create -n realfold python=3.10 -y
conda activate realfold

pip install torch numpy pandas tqdm
pip install torch-cluster          # optional, faster neighbour lists
pip install biotite matplotlib seaborn networkx  # optional, I/O and plots
```

---

Quick Start – Refinement Engine

```bash
# Basic refinement
python real_fold_one.py refine --input 1abc.pdb --output refined.pdb --steps 300

# Full refinement with PME, GPU, and full‑atom export
python real_fold_one.py refine -i 1abc.pdb -o refined_full.pdb --steps 500 --gpu --pme --full_atom

# Include ligands
python real_fold_one.py refine -i protein.pdb --ligand ligand.sdf -o complex.pdb --full_atom

# Gradient validation test
python real_fold_one.py test
```

---

High‑Throughput Mutation Scanning (HT)

The real_fold_one_ht.py module handles all in silico mutagenesis — from single‑mutation scans
to double‑mutant epistasis — with checkpointing and multi‑GPU support.

Targeted vs Full Scan

· Full scan (--scan): evaluates every position × every allowed monomer.
· Targeted list (--mutlist my_muts.json): evaluates only the specified mutations.

Epistasis Analysis

--epistasis computes double‑mutant energies. Pairs can be supplied (--epipairs) or randomly
sampled (--max_epi). The engine reads pre‑computed single‑mutation results (via --resume) to
calculate additive ΔΔG, then evaluates only the double mutant to obtain epistasis:

```
ε = ΔΔG_double – (ΔΔG_single1 + ΔΔG_single2)
```

Workflow: Targeted → Global Optimization

A powerful strategy is to first scan a targeted list, then perform a full scan on the best
variant to discover epistatic stabilisations:

```bash
# Step 1: Targeted scan on residues of interest
python real_fold_one_ht.py --pdb wildtype.pdb --mutlist my_targets.json --output step1_out

# Step 2: Full scan on the best variant
python real_fold_one_ht.py --pdb step1_out/best_variant.pdb --scan --output step2_out
```

Full command examples:

```bash
# Full single‑mutation scan
python real_fold_one_ht.py --pdb 1abc.pdb --scan --output ht_out

# DNA scan (ideal helix built automatically)
python real_fold_one_ht.py --seq "ATGCGTACGTAG" --scan --output dna_scan

# Multi‑GPU with resume
python real_fold_one_ht.py --pdb 1abc.pdb --scan --gpu --num_gpus 2 --resume

# Targeted list
python real_fold_one_ht.py --pdb 1abc.pdb --mutlist my_muts.json --output targeted

# Quick single mutation
python real_fold_one_ht.py --pdb 1abc.pdb --single "0:5:A"

# Epistasis (random pairs)
python real_fold_one_ht.py --pdb 1abc.pdb --epistasis --max_epi 500

# Epistasis (predefined pairs)
python real_fold_one_ht.py --pdb 1abc.pdb --epistasis --epipairs my_pairs.json
```

Input JSON Formats

· Mutation list (--mutlist): [[chain_index, pos_in_chain, new_monomer], ...]
· Epistasis pairs (--epipairs): [[chain1, pos1, chain2, pos2], ...]

Output Files

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
python real_fold_one.py train --input native1.pdb native2.pdb --epochs 100 --output kernel_params.json
```

---

Antibody CDR Modelling

```bash
python real_fold_one.py antibody --antigen antigen.pdb --cdr_start 95 --cdr_end 102 --output antibody.pdb
```

---

DNA Origami Design

```bash
# shape.json contains "vertices" and "edges"
python real_fold_one.py origami --shape shape.json --output my_origami
```

Outputs: full‑atom PDB, oxDNA top/dat files, optional BV topological check (--bv_check).

---

Validation & Testing

· Gradient check: python real_fold_one.py test
· RMSD: compute_rmsd() with Kabsch alignment.
· BV check: for origami, --bv_check verifies the classical master equation.

---

Performance Tips

· Install torch-cluster for GPU‑accelerated neighbour lists.
· For large systems (>10 000 residues), increase --rebuild_interval and keep RG enabled.
· Use --full_atom only for final export; side‑chains are rebuilt on‑the‑fly during scanning.
· For multi‑GPU, set --num_gpus to available devices; ≈3 GB VRAM per 500‑residue protein.
· On CPU, set OMP_NUM_THREADS to control parallelism.

---

Citing REAL FOLD ONE

```
Yoon A Limsuwan. "REAL FOLD ONE: SOC‑Controlled Universal Refinement Engine."
Zenodo, 2026.  DOI: 10.5281/zenodo.20264580

```

---

License

This project is licensed under the MIT License – see LICENSE for details.

---

Contributing

Contributions are welcome! Please open an issue to discuss proposed changes or submit
a pull request. For major features, contact the author first.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/yoonalimsuwan/REAL-FOLD-ONE

```
