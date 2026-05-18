``
# REAL FOLD ONE

**SOC‑Controlled Universal Refinement & High‑Throughput Mutation Scanning Suite**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A unified physics‑based framework for macromolecular modelling and mutational scanning.
Built around a novel Self‑Organised Criticality (SOC) controller, it refines proteins, DNA, RNA, and their complexes
using a differentiable energy function, then scales to thousands of *in silico* mutations across multiple GPUs.

---

## Overview

**REAL FOLD ONE** is composed of three main components:

| Component | Description |
|-----------|-------------|
| **REAL FOLD ONE** | Core refinement engine with full‑atom force field, PME, multigrid, and SOC‑driven Langevin dynamics. |
| **REAL FOLD ONE HT** | High‑throughput mutation scanning and epistasis analysis (latest version). |
| **REAL FOLD ONE HTS** | Earlier prototype of the scanner; **HT** is the recommended, production‑ready upgrade. |

All tools share the same physics backend and can be run on CPU, a single GPU, or multiple GPUs via `torch.multiprocessing`.

---

## Installation

```bash
git clone https://github.com/your-username/real-fold-one.git
cd real-fold-one

# Create environment (optional)
conda create -n realfold python=3.10 -y
conda activate realfold

# Install dependencies
pip install torch numpy pandas tqdm
pip install biotite seaborn matplotlib networkx  # optional, for I/O and plots
pip install torch-cluster                          # optional, for faster neighbor lists
```

---

REAL FOLD ONE — Refinement Engine

Key Features

· SOC Controller – learnable CSOC kernel & Semantic‑State Contraction adaptively tune temperature and friction during refinement.
· Full‑Atom Physics – AMBER ff14SB‑like parameters for proteins, OL15‑like for DNA/RNA, GAFF2 for ligands.
· Advanced Electrostatics – Sparse PME, geometric multigrid Poisson solver, block‑wise multipole long‑range correction.
· Multiscale Refinement – RG coarse‑graining periodically smooths the trajectory.
· Simulated Annealing – optional temperature schedule from 1000 K down to 300 K.
· Training Module – fine‑tune the SOC kernel on native structures.
· DNA Origami – wireframe routing and full‑atom PDB export.

Quick Start

```bash
# Refine a protein from PDB (CA atoms are extracted)
python real_fold_one.py refine --input 1abc.pdb --output refined.pdb --steps 300

# Use GPU, enable PME, and export full-atom structure
python real_fold_one.py refine --input 1abc.pdb --output refined_full.pdb --steps 500 --gpu --pme --full_atom

# Run a gradient validation test
python real_fold_one.py test
```

Command Line Options (refine)

Flag Description
--input, -i Input PDB or mmCIF file
--chain Chain ID to extract
--output, -o Output PDB filename (CA‑only or full‑atom)
--steps Number of refinement steps (default: 600)
--lr Learning rate for Adam (default: 1e-4)
--pme Use Particle‑Mesh Ewald
--multigrid Use geometric multigrid Poisson solver
--block_lr Add block‑wise long‑range correction
--ligand One or more ligand files (SDF, MOL2, PDB)
--full_atom Export full‑atom PDB (sidechains reconstructed)
--trajectory Save trajectory as .npy file
--device cpu, cuda, or auto (default)
--milstein Use Milstein scheme for the Langevin SDE
--no_rg Disable RG refinement
--no_ssc Disable Semantic‑State Contraction

Antibody CDR Modelling

```bash
python real_fold_one.py antibody --antigen antigen.pdb --cdr_start 95 --cdr_end 102 --output antibody.pdb
```

DNA Origami Design

```bash
# shape.json contains "vertices" and "edges"
python real_fold_one.py origami --shape shape.json --output my_origami
```

This produces my_origami.pdb (full‑atom DNA model) and my_origami.top/.dat (oxDNA format).

---

REAL FOLD ONE HT — High‑Throughput Mutation Scanner

Note: real_fold_one_ht.py is the recommended scanner.
The earlier real_fold_one_hts.py is kept for reference but HT is more robust, supports resume/checkpointing, and handles multi‑GPU parallelism safely.

Features

· Full Single‑Mutation Scan – every residue → every allowed monomer.
· Targeted Mutation Lists – provide a JSON file of specific mutations to evaluate.
· Double‑Mutant Epistasis – random sampling or user‑supplied pairs.
· Local Relaxation – relaxes a small window around the mutation site for fast ΔΔG estimation.
· Multi‑GPU Parallelism – worker pool distributes mutations across available GPUs.
· Checkpoint & Resume – auto‑saves intermediate results every 100 mutations; use --resume to continue.
· Publication‑Ready Plots – ΔΔG distribution, mutational landscape heatmap, position‑tolerance profile, epistasis distribution, additivity scatter.

Quick Start

```bash
# Full scan on a protein (requires PDB)
python real_fold_one_ht.py --pdb 1abc.pdb --scan --output ht_output

# Scan DNA from sequence (ideal helix is built)
python real_fold_one_ht.py --seq "ATGCGTACGTAG" --scan --output dna_scan

# Scan with GPU and resume support
python real_fold_one_ht.py --pdb 1abc.pdb --scan --gpu --num_gpus 2 --resume

# Targeted mutations from a JSON file
python real_fold_one_ht.py --pdb 1abc.pdb --mutlist mutations.json --output targeted

# Quick single mutation evaluation
python real_fold_one_ht.py --pdb 1abc.pdb --single "0:5:A"
```

Epistasis Scanning

```bash
# Random epistasis pairs (max 1000 by default)
python real_fold_one_ht.py --pdb 1abc.pdb --epistasis --max_epi 500

# From a predefined list
python real_fold_one_ht.py --pdb 1abc.pdb --epistasis --epipairs pairs.json
```

Input File Formats

Mutation list JSON (for --mutlist):

```json
[[0, 5, "A"], [0, 10, "G"], [1, 23, "T"]]
```

Each entry: [chain_index, position_in_chain, new_monomer].

Epistasis pair list JSON (for --epipairs):

```json
[[0, 5, 0, 10], [0, 5, 1, 23]]
```

Each entry: [chain1, pos1, chain2, pos2].

Output Files

File Description
single_mutations.csv All single‑mutation results (ΔΔG, energies, type)
epistasis.csv Epistasis pairs with additive vs double‑mutant ΔΔG
summary.json Wild‑type energy and mutation counts
ddg_distribution.png Histogram of ΔΔG values
mutational_landscape.png Heatmap of mutations (position × mutant)
position_profile.png Mean ΔΔG per residue with standard deviation
epistasis_distribution.png Histogram of epistasis (ε)
additivity_scatter.png Additive vs double‑mutant ΔΔG scatter plot

---

REAL FOLD ONE HTS (Legacy)

The original high‑throughput script (real_fold_one_hts.py) works but lacks the refined multi‑GPU handling and checkpointing present in HT. It is kept for reproducibility; new projects should use real_fold_one_ht.py.

---

Training the SOC Kernel

```bash
# Train kernel on a set of native PDB files
python real_fold_one.py train --input native1.pdb native2.pdb --epochs 100 --output kernel_params.json
```

The trained alpha and lambda can then be loaded into the refinement engine (set init_alpha and init_lambda in CSOCKernel).

---

Validation

· Gradient check: python real_fold_one.py test verifies analytical gradients against finite differences.
· RMSD computation: compute_rmsd(coords1, coords2) is provided for comparing structures.
· BV topological check: For DNA origami, --bv_check verifies the classical master equation of the BV formalism.

---

Performance Tips

· Use torch-cluster for faster neighbour list construction.
· For large systems (>10 000 residues), enable --use_rg (default) and increase --rebuild_interval.
· Set OMP_NUM_THREADS to control CPU parallelism when running on CPU.
· For multi‑GPU scanning, ensure each GPU has enough memory for the full system (peak memory ~2–3 GB for a 500‑residue protein with sidechains).

---

Citing REAL FOLD ONE

If you use this software in your research, please cite:

```
Yoon A Limsuwan. "REAL FOLD ONE: SOC‑Controlled Universal Refinement Engine."
Zenodo, 2026. DOI: 10.5281/zenodo.XXXXXXX
```

---

License

This project is licensed under the MIT License – see the LICENSE file for details.

---

Contributing

Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request. For major features, consider contacting the author first.

---

Contact

Yoon A Limsuwan – GitHub
Project link: https://github.com/your-username/real-fold-one

```
