``
# REAL FOLD ONE

**SOC‑Controlled Universal Refinement & High‑Throughput Mutation Scanning Suite**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19814975-blue)](https://doi.org/10.5281/zenodo.19814975)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20821722-blue)](https://doi.org/10.5281/zenodo.20821722)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20379088-blue)](https://doi.org/10.5281/zenodo.20379088)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20633681-blue)](https://doi.org/10.5281/zenodo.20633681)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20623622-blue)](https://doi.org/10.5281/zenodo.20623622)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20755856-blue)](https://doi.org/10.5281/zenodo.20755856)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20755892-blue)](https://doi.org/10.5281/zenodo.20755892)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20730429-blue)](https://doi.org/10.5281/zenodo.20730429)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17777316-blue)](https://doi.org/10.5281/zenodo.17777316)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20824246-blue)](https://doi.org/10.5281/zenodo.20824246)


A unified physics‑based framework for macromolecular refinement and mutational scanning.
Built around a novel **Self‑Organised Criticality (SOC) controller**, it refines proteins, DNA, RNA,
and their complexes using a fully differentiable energy function powered by **OpenMM**, then scales to
thousands of *in silico* mutations across multiple GPUs — all without writing a single line of CUDA C++.

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
| **Core Refinement Engine** | `real_fold_one_v2.py` | Full‑atom refinement, training, antibody, origami, MD, validation |
| **HT Mutation Scanner** | `real_fold_one_ht_v2.py` | High‑throughput ΔΔG and epistasis scanning |

Both modules share the same physics backend (SOC kernel, neighbour lists) and run on CPU, single GPU,
or multi‑GPU via `torch.multiprocessing`. The HT scanner extends the core engine with a fast
coarse‑grained energy model for scanning thousands of mutations in minutes.

---

## Key Features

### Refinement Engine (`real_fold_one_v2.py`)

- **SOC Controller** – learnable CSOC kernel and Semantic‑State Contraction (SSC) low‑pass filter
  adaptively tune temperature and friction during Langevin dynamics.
- **Multiscale Refinement** – RG coarse‑graining periodically removes high‑frequency noise while
  respecting chain boundaries.
- **Full‑Atom Physics** – all forces computed via **OpenMM** with molecular mechanics force fields:
  - Proteins: AMBER ff14SB
  - DNA/RNA: OL15
  - Ligands: GAFF2 (automatic SMILES lookup from PDB Chemical Component Dictionary)
  - Post‑translational modifications (phosphoserine, methyllysine, etc.)
  - Disulfide bond detection and constraint
- **Implicit & Explicit Solvent** – support for GB models (OBC, GBn2) or explicit TIP3P water
  with ions and co‑solvents.
- **Advanced Electrostatics** – PME, reaction field, or implicit solvent; handled transparently
  by OpenMM.
- **Hierarchical Neighbor Lists** – fast GPU (`torch-cluster`), SciPy `cKDTree`, or pure PyTorch
  fallback; supports multiple cutoffs.
- **DNA Origami** – wireframe routing, staple design, full‑atom PDB export, and oxDNA format.
- **Langevin Dynamics** – overdamped Langevin with adaptive friction, temperature, and noise
  controlled by the SOC stress metric.
- **Simulated Annealing** – cosine annealing with warm restarts.
- **Scalable** – O(N) memory neighbour lists, >100 000 atoms.
- **Environment‑Adaptive** – works on CPU (3 GB RAM), Colab T4, single GPU, or multi‑GPU (DDP).
- **Training Module** – train the SOC kernel on native structures.
- **Validation Suite** – Kabsch RMSD, clash score, Ramachandran outliers, rotamer analysis,
  bond geometry checks.
- **Molecular Dynamics** – long‑time MD simulations (ps to μs) with explicit solvent, NPT/NVT,
  checkpointing, **all inside REAL FOLD ONE** — no external engine needed.
- **Antibody Modelling** – rigorous binding free energy via MM‑GBSA, CDR loop remodeling.
- **Restraints** – positional restraints for partial refinement (PDB‑index friendly).

### HT Mutation Scanner (`real_fold_one_ht_v2.py`)

- **Coarse‑grained Energy** – uses SOC kernel + residue‑type pairwise potentials:
  - Hydrophobicity‑based contact potential for proteins.
  - Base‑stacking and Watson‑Crick pairing pseudo‑energy for DNA/RNA.
- **Local Relaxation** – fast ΔΔG estimation by gradient‑descending only residues near the mutation site.
- **Full Single‑Mutation Scan** – every position × every allowed monomer.
- **Targeted Mutation Lists** – evaluate only user‑specified mutations.
- **Double‑Mutant Epistasis** – random or user‑supplied pairs; additive ΔΔG and epistasis ε reported.
- **Multi‑GPU Parallelism** – `torch.multiprocessing` with checkpoint/resume.
- **Publication‑Ready Plots** – ΔΔG distribution, mutational landscape, position tolerance profile,
  epistasis distribution, additivity scatter.

---

## Integration with Structure Predictors

REAL FOLD ONE is a **complete post‑prediction pipeline**. It takes an initial Cα model from
AlphaFold 3, ESMFold, Rosetta, or any predictor, and carries it all the way to a fully
solvated, equilibrated MD trajectory — without ever leaving the REAL FOLD ONE environment.

```

┌─────────────┐      ┌─────────────────────────────────────────────────┐
│ AlphaFold 3 │ ──► │              REAL FOLD ONE                        │
│  (or any    │      │  refine → validate → md (explicit solvent,       │
│  predictor) │      │  NPT/NVT, checkpointing, full analysis)          │
└─────────────┘      └─────────────────────────────────────────────────┘

```

- **Refine**: SOC‑guided energy minimisation, clash removal, side‑chain optimisation.
- **Validate**: RMSD, Ramachandran, rotamer, clash score — all built in.
- **MD**: Launch production MD with a single command. The system is solvated, ions added,
  and simulated with OpenMM under the hood, but you only ever interact with REAL FOLD ONE.

Complete example:

```bash
# Predict structure with AlphaFold 3 → folded_model.pdb

# Step 1: Refine
python real_fold_one_v2.py refine -i folded_model.pdb -o refined.pdb --steps 500 --gpu

# Step 2: Validate
python real_fold_one_v2.py validate --input refined.pdb --reference native.pdb

# Step 3: Run MD (explicit solvent, NPT, 100 ns)
python real_fold_one_v2.py md -i refined.pdb -o traj --steps 50000000 --temperature 310 --gpu
```

All steps share the same molecular topology and force field; there is no format conversion
or data loss between stages.

---

Installation

```bash
git clone https://github.com/yoonalimsuwan/REAL-FOLD-ONE.git
cd real-fold-one

conda create -n realfold python=3.10 -y
conda activate realfold

# Core dependencies (OpenMM is required)
conda install -c conda-forge openmm -y
pip install torch numpy pandas tqdm

# Optional but recommended
pip install torch-cluster          # faster GPU neighbour lists
pip install biotite matplotlib seaborn networkx rdkit openff-toolkit openmmforcefields
```

---

Quick Start – Refinement Engine

```bash
# Basic refinement with implicit solvent (default OBC)
python real_fold_one_v2.py refine --input 1abc.pdb --output refined.pdb --steps 300

# Explicit solvent, PME, GPU, and Langevin dynamics
python real_fold_one_v2.py refine -i 1abc.pdb -o refined.pdb --steps 500 --gpu --solvate --langevin

# Include ligand SMILES
python real_fold_one_v2.py refine -i protein.pdb --ligand-smiles '{"LIG":"c1ccccc1"}' -o complex.pdb

# Positional restraints (JSON with "atoms" and "target")
python real_fold_one_v2.py refine -i input.pdb --restraint-json restraints.json

# Gradient validation test
python real_fold_one_v2.py test --input 1abc.pdb

# Run explicit-solvent MD for 100 ns after refinement
python real_fold_one_v2.py md -i refined.pdb -o prod --steps 50000000 --gpu --temperature 310
```

---

High‑Throughput Mutation Scanning (HT)

The real_fold_one_ht_v2.py module handles all in silico mutagenesis — from single‑mutation scans
to double‑mutant epistasis — with checkpointing and multi‑GPU support.

Energy Model
The HT scanner uses a fast coarse‑grained potential derived from the core SOC kernel and
sequence‑dependent terms:

· Protein: hydrophobicity‑based contact energy (Miyazawa‑Jernigan style).
· DNA/RNA: base‑stacking and Watson‑Crick pairing pseudo‑energies.
· Local relaxation: only residues within ±window of the mutation are free to move.

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

```bash
# Step 1: Targeted scan on residues of interest
python real_fold_one_ht_v2.py --pdb wildtype.pdb --mutlist my_targets.json --output step1_out

# Step 2: Full scan on the best variant
python real_fold_one_ht_v2.py --pdb step1_out/best_variant.pdb --scan --output step2_out
```

Full command examples:

```bash
# Full single‑mutation scan
python real_fold_one_ht_v2.py --pdb 1abc.pdb --scan --output ht_out

# DNA scan (ideal helix built automatically)
python real_fold_one_ht_v2.py --seq "ATGCGTACGTAG" --scan --output dna_scan

# Multi‑GPU with resume
python real_fold_one_ht_v2.py --pdb 1abc.pdb --scan --gpu --num_gpus 2 --resume

# Targeted list
python real_fold_one_ht_v2.py --pdb 1abc.pdb --mutlist my_muts.json --output targeted

# Quick single mutation
python real_fold_one_ht_v2.py --pdb 1abc.pdb --single "0:5:A"

# Epistasis (random pairs)
python real_fold_one_ht_v2.py --pdb 1abc.pdb --epistasis --max_epi 500
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
python real_fold_one_v2.py train --input native1.pdb native2.pdb --epochs 100 --output kernel_params.json
```

---

Antibody CDR Modelling

```bash
python real_fold_one_v2.py antibody --antigen antigen.pdb --cdr_start 95 --cdr_end 102 --output antibody.pdb
```

---

DNA Origami Design

```bash
# shape.json contains "vertices" and "edges"
python real_fold_one_v2.py origami --shape shape.json --output my_origami
```

Outputs: full‑atom PDB, oxDNA top/dat files.

---

Validation & Testing

· Gradient check: python real_fold_one_v2.py test
· Structure validation: python real_fold_one_v2.py validate --input refined.pdb --reference native.pdb
  Reports initial/final energy, RMSD, clash score, Ramachandran outliers, rotamer outliers.

---

Performance Tips

· Install torch-cluster for GPU‑accelerated neighbour lists.
· For large systems (>10 000 residues), increase --rebuild_interval and enable RG.
· Use --full_atom only for final export; side‑chains are rebuilt on‑the‑fly during scanning.
· For multi‑GPU, set --num_gpus to available devices; ≈3 GB VRAM per 500‑residue protein.
· On CPU, set OMP_NUM_THREADS to control parallelism.

---

Future: AI‑Driven Refinement and the Path to O(1) Complexity

REAL FOLD ONE is built from the ground up as a differentiable physics engine.
Every component—the SOC controller, CSOC kernel, energy terms, and even the implicit solvent
approximation—runs inside PyTorch’s autograd graph. This architectural choice means that
REAL FOLD ONE is not merely a refinement tool; it is also a native AI platform.

Because the entire pipeline is differentiable, it can be directly embedded as a layer in a
larger neural network, or used to generate physically rigorous training data for deep learning
models. This opens three concrete paths toward near‑O(1) complexity in macromolecular
refinement and simulation:

1. Learned Refinement Surrogates
      A neural network (e.g., an SE(3)‑equivariant GNN) can be trained on pairs of
   (initial coarse structure, SOC‑refined full‑atom structure). Once trained, the network
   predicts the refined structure in a single forward pass—completely bypassing the iterative
   energy minimisation. The computational cost becomes independent of protein size, yielding
   de facto O(1) behaviour.
2. Adaptive Simulation Control
      The SOC controller outputs a real‑time stress metric (σ) that quantifies local strain.
   An AI agent can read this signal to dynamically decide:
   · when to hand a structure back to REAL FOLD ONE for further refinement,
   · when to change the integration time‑step in a downstream MD engine (GPUMD, OpenMM),
   · or when to terminate a simulation because the system has reached equilibrium.
     This removes the manual tuning that dominates large‑scale MD, making the entire pipeline
     more efficient and less dependent on human expertise.
3. Learned Force Fields and Differentiable MD
      The same differentiable backbone allows REAL FOLD ONE to adopt machine‑learning force
   fields (e.g., MACE‑MP, Allegro, NEP). In this scenario, the energy function itself is a
   neural network, and refinement becomes gradient descent through a learned potential.
   Coupled with a differentiable MD engine, the whole simulation loop could be optimised
   end‑to‑end—for instance, to minimise the SOC stress of the final equilibrated ensemble.

Vision
REAL FOLD ONE sits at the centre of a future stack:

```
Sequence → AF3 → REAL FOLD ONE → AI‑Guided MD (GPUMD / MindSPONGE)
                         ↕
              Learned Surrogate Models
```

In this ecosystem, REAL FOLD ONE serves as both a physics‑based teacher and a
differentiable evaluation module. The result is a self‑improving loop that continuously
shrinks the gap between prediction, refinement, and simulation—ultimately delivering
refined, production‑ready structures with a computational cost that approaches constant
time for the end user.

This is not a distant dream; the differentiable architecture of REAL FOLD ONE already
provides all the primitives necessary to build these AI‑driven capabilities.

---

Reducing the Need for Quantum Computing through Differentiable Physics and AI

Quantum computing has long been viewed as the ultimate solution for tackling the
exponential complexity of molecular simulation. However, REAL FOLD ONE demonstrates
an alternative path—one that achieves near-constant-time refinement by fusing
differentiable physics with modern deep learning.

1. Bypassing Computational Complexity
      Traditional MD and quantum chemistry scale as O(N²) or O(N³). By embedding a
   fully differentiable SOC‑based physics engine inside an autograd framework,
   REAL FOLD ONE enables the training of AI surrogate models. Once trained, these
   models predict the refined structure in a single forward pass, offering de facto
   O(1) complexity on commodity hardware.
2. A Physics‑Based Teacher for AI
      Pure deep learning often violates physical constraints. REAL FOLD ONE acts as a
   rigorous, differentiable teacher that supplies thermodynamic gradients back to the
   AI model. This creates a self‑improving loop where the AI learns to respect
   energy landscapes, torsional preferences, and steric constraints, closing the gap
   between data‑driven prediction and first‑principle physics.
3. Hardware Democratisation
      The engine is written entirely in PyTorch primitives without a single line of
   CUDA C++. This vendor‑neutral design runs unchanged on NVIDIA GPUs, Huawei
   Ascend NPUs, Apple MPS, or any future accelerator that supports PyTorch.
   It proves that strategic algorithmic design can overcome the need for specialised
   quantum hardware, making cutting‑edge biomolecular simulation accessible to every
   laboratory worldwide.

REAL FOLD ONE thus repositions the frontier: instead of waiting for fault‑tolerant
quantum computers, we can harness the synergy of differentiable physics and AI to
solve macromolecular problems at constant cost today.

---

REAL FOLD ONE vs. Phenix: Complementary Roles in Structural Refinement

Phenix is the gold standard for refining structures against experimental data
(Cryo‑EM density maps, X‑ray diffraction). It optimises atomic coordinates to
maximise agreement with the observed data, using a likelihood‑based target
function. REAL FOLD ONE, by contrast, is a physics‑ and AI‑driven refinement
engine—it does not require experimental data and instead minimises a fully
differentiable energy function under SOC control.

When REAL FOLD ONE can replace Phenix (or where it excels):

· Post‑AlphaFold 3 refinement – AF3 models often contain minor steric clashes
  or strained geometry. REAL FOLD ONE relaxes these purely through physics,
  without any experimental map, often outperforming Phenix’s geometry
  regularisation for this task.
· High‑throughput mutation scanning – Phenix is not designed for thousands
  of in silico mutations. REAL FOLD ONE’s HT module can evaluate ΔΔG values
  across massive mutational landscapes on multi‑GPU systems.
· Differentiable integration with AI – REAL FOLD ONE is fully differentiable
  (PyTorch autograd). It can backpropagate gradients through the entire
  refinement process, enabling AI‑driven surrogate models and closed‑loop
  training. Phenix has no comparable capability.

When Phenix remains essential:

· Experimental data fitting – Phenix refines against real electron density
  maps, optimising R‑work/R‑free. REAL FOLD ONE does not use experimental
  data and cannot replace this step.

Ideal hybrid workflow:

1. Build an initial model with Phenix (using Cryo‑EM/X‑ray data).
2. Pass the model to REAL FOLD ONE for final physics‑based cleanup,
   resolving clashes and optimising electrostatics.
3. For purely computational pipelines (AlphaFold → refinement → MD), REAL FOLD ONE
   can be used without Phenix.

Thus, REAL FOLD ONE and Phenix are not competitors but complementary tools that
together span the full spectrum from experiment‑driven to physics‑driven
refinement.

---

Connecting Genomic Data to Structural Impact: Cancer Mutations and Duons

REAL FOLD ONE is uniquely positioned to bridge clinical genomics and
structural biology. By ingesting mutation data from cancer institutes
(e.g., COSMIC, ICGC, ClinVar, and the Duon database), the engine can map
thousands of patient‑derived variants directly onto protein structures and
compute their effect on folding stability and electrostatic integrity.

· Physics‑based interpretation of clinical variants – For every observed
  missense mutation, REAL FOLD ONE evaluates ΔΔG and the local stress
  redistribution (σ) through its SOC controller. This allows researchers to
  distinguish between passenger mutations and driver mutations that
  destabilise key domains or binding interfaces.
· Duon analysis – Duons are codons that simultaneously encode an amino
  acid and a splicing or regulatory signal. Mutations at these positions
  can alter both protein sequence and gene regulation. REAL FOLD ONE can
  combine structural stability scores with regulatory impact scores,
  providing a unified picture of how duon mutations affect cellular function.
· Population‑scale scanning – The high‑throughput mutation pipeline
  (real_fold_one_ht_v2.py) can scan thousands of cancer‑associated variants
  in hours on a multi‑GPU cluster, generating structural scores that can
  be correlated with patient survival data, drug response, or evolutionary
  conservation.

When coupled with the AI‑driven refinement architecture, this creates a
feedback loop where clinical data inform structural models, and physics‑based
models improve the interpretation of future cancer genomes—moving closer to
truly personalised structural oncology.

---

REAL FOLD ONE as a Data Engine for Predictor AI

REAL FOLD ONE is not only a refinement engine—it is also a high‑fidelity
data generator for training and improving structure predictors.

1. Physics‑based ground‑truth structures
   REAL FOLD ONE takes an initial Cα trace (from any predictor) and produces
   a full‑atom, SOC‑optimised, clash‑free structure. These refined structures
   serve as superior training targets for models like AlphaFold, ESMFold, or
   RoseTTAFold, especially in loop regions and side‑chain packing where
   predictors often struggle.
2. ΔΔG training sets
   The high‑throughput mutation scanner (real_fold_one_ht_v2.py) can evaluate
   thousands of single mutations with full‑atom relaxation, producing large‑scale
   datasets of (mutation, ΔΔG) pairs. These can be used to train AI‑based
   stability predictors that run in constant time (O(1)) instead of requiring
   repeated physics simulations.
3. Differentiable feedback loop
   Because REAL FOLD ONE is written entirely in PyTorch, it can backpropagate
   energy gradients directly into a predictor network. This allows the predictor
   to be fine‑tuned with a physics‑informed loss, improving accuracy without
   requiring additional experimental data.
4. Surrogate model training
   Pairs of (initial coarse structure, SOC‑refined full‑atom structure) can be
   used to train an SE(3)‑equivariant GNN that predicts refined structures in a
   single forward pass, effectively replacing iterative refinement with an O(1)
   neural network.

In summary, REAL FOLD ONE serves as both a teacher (providing accurate
training targets) and a differentiable loss function (supplying energy
gradients) for the next generation of AI‑based structure predictors.

---

Citing REAL FOLD ONE

```
Yoon A Limsuwan. "REAL FOLD ONE: SOC‑Controlled Universal Refinement Engine."
Zenodo, 2026.
https://doi.org/10.5281/zenodo.20821722
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
