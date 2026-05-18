`
# REAL FOLD ONE

**SOC‑Controlled Universal Refinement & High‑Throughput Mutation Scanning Suite**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A unified physics‑based framework that refines and scans macromolecular structures—
from natural proteins and nucleic acids to *de novo* designs and synthetic complexes.
Built around a novel Self‑Organised Criticality (SOC) controller, it leverages a full‑atom
differentiable energy function (AMBER/OL15/GAFF2) and advanced electrostatics to
polish predicted models, evaluate mutational landscapes, and design DNA origami.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Integration with Structure Predictors](#integration-with-structure-predictors)
- [Installation](#installation)
- [Quick Start – Refinement Engine](#quick-start--refinement-engine)
- [High‑Throughput Mutation Scanning](#high-throughput-mutation-scanning)
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

## Overview

REAL FOLD ONE provides two tightly integrated modules:

| Module | File | Purpose |
|--------|------|---------|
| **Core Refinement Engine** | `real_fold_one.py` | Full‑atom refinement, training, antibody, origami |
| **HT Mutation Scanner** | `real_fold_one_ht.py` | High‑throughput ΔΔG and epistasis scanning |

Both modules share the same physics backend and can be run on CPU, single GPU, or multiple GPUs.

---

## Key Features

### Refinement Engine

- **SOC Controller** – learnable CSOC kernel and Semantic‑State Contraction (SSC) low‑pass filter adaptively tune temperature and friction during Langevin dynamics.
- **Multiscale Refinement** – RG coarse‑graining periodically removes high‑frequency noise.
- **Full‑Atom Physics**
  - Proteins: AMBER ff14SB‑like Lennard‑Jones parameters and partial charges for all side‑chain atoms.
  - DNA/RNA: OL15‑like parameters for all nucleotides (C4′, phosphate, base).
  - Ligands: GAFF2 force field with automatic atom‑type assignment and topology generation.
  - Antibodies: specialised Rosetta‑like scoring for antigen‑antibody interfaces.
- **Advanced Electrostatics**
  - Sparse **PME** (Particle Mesh Ewald) for long‑range electrostatics.
  - Geometric **multigrid Poisson solver** (V‑cycle) for direct potential calculation.
  - **Block‑wise long‑range correction** via multipole expansion.
- **Hierarchical Neighbor Lists** – separate cutoffs for clash (2.5 Å), LJ (6 Å), and electrostatics (12 Å) for efficiency.
- **DNA Origami** – wireframe routing, staple design, full‑atom PDB export, and oxDNA format.
- **Itô SDE** – Milstein scheme for the Langevin equation, with Malliavin sensitivity for parameter Greeks.
- **Simulated Annealing** – optional temperature schedule from 1000 K to 300 K.
- **Scalable** – chunked O(N) graph building supports >100 000 residues.
- **Environment‑Adaptive** – works on CPU (3 GB RAM), Colab T4, single GPU, or multi‑GPU (DDP).

### High‑Throughput Mutation Scanner

- **Full Single‑Mutation Scan** – every residue → every allowed monomer.
- **Targeted Mutation Lists** – provide a JSON list of specific mutations.
- **Double‑Mutant Epistasis** – random sampling or user‑supplied pairs.
- **Local Relaxation Window** – fast ΔΔG estimation by relaxing only the mutation site.
- **Multi‑GPU Parallelism** – `torch.multiprocessing` pool with checkpoint/resume.
- **Publication‑Ready Plots** – ΔΔG distribution, mutational landscape heatmap, position‑tolerance profile, epistasis distribution, additivity scatter.

### Training & Validation

- **Train SOC kernel** on native structures to learn optimal interaction parameters.
- **Gradient validation** (finite‑difference check).
- **RMSD** calculation with Kabsch superposition.

---

## Integration with Structure Predictors

REAL FOLD ONE is designed as a **post‑processing refinement engine** for any structure predictor
(AlphaFold, ESMFold, Rosetta, etc.). It does **not** predict structures from sequence; instead,
it takes initial Cα coordinates and uses its physics‑based energy function to:

- Correct local strain and steric clashes.
- Optimise hydrogen‑bond networks and electrostatics.
- Rebuild full side‑chain and nucleic acid conformations.

### Natural Proteins

```text
Sequence → Predictor (e.g. AlphaFold) → Cα model → REAL FOLD ONE refine → Full‑atom refined structure
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

High‑Throughput Mutation Scanning

The real_fold_one_ht.py module performs massive in silico mutagenesis.

```bash
# Full single‑mutation scan on a protein
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

· Gradient check: python real_fold_one.py test verifies analytical gradients against finite differences.
· RMSD: the function compute_rmsd() performs Kabsch alignment and returns the RMSD.
· BV check: for origami, --bv_check verifies the classical master equation.

---

Performance Tips

· Install torch-cluster for GPU‑accelerated neighbour lists.
· For large systems (>10 000 residues), increase --rebuild_interval to 200–400 and keep RG enabled.
· Use --full_atom only for final export; during scanning, side‑chains are rebuilt on‑the‑fly.
· For multi‑GPU scanning, set --num_gpus to the number of available devices and ensure enough VRAM per GPU (≈3 GB for a 500‑residue protein).
· On CPU, set OMP_NUM_THREADS to the desired number of cores.

---

Citing REAL FOLD ONE

If you use this software, please cite:

```
Yoon A Limsuwan. "REAL FOLD ONE: SOC‑Controlled Universal Refinement Engine."
Zenodo, 2026. DOI: https://doi.org/10.5281/zenodo.20257600
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
Project link: https://github.com/yoonalimsuwan/REAL-FOLD-ONE
