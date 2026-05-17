`
# REAL FOLD ONE — Physics‑Based Refinement Engine

**REAL FOLD ONE** is a SOC‑driven universal refinement engine for proteins, DNA/RNA, ligands, and multimers. It uses full all‑atom physics (bond, angle, torsion, Ramachandran, LJ, Coulomb, H‑bonds, solvation) with a novel Self‑Organised Criticality (SOC) controller that adaptively tunes temperature, friction, and avalanche propagation during optimisation. No retraining is needed – it works directly on any input structure.

**REAL FOLD ONE HTS** extends the engine to high‑throughput mutation scanning and epistasis analysis, enabling systematic ΔΔG evaluation for protein engineering and drug design.

---

## Why Refinement?

### For De Novo & Engineered Proteins (small to medium, N < 500)
Predictors/detectors (AlphaFold3, ESMFold, RFdiffusion, ProteinMPNN) produce plausible backbones but often have:

- Sidechain clashes or wrong rotamers
- Distorted bond lengths/angles
- Unphysical Ramachandran outliers

**REAL FOLD ONE** fixes all‑atom geometry, repacks sidechains, and resolves steric clashes using full physics (LJ, Coulomb, torsion, H‑bond, Ramachandran) – without retraining.  
*Perfect as a physics‑based polish after any de novo design pipeline.*

### For Large Natural Proteins / Complexes (N > 5,000 up to 100k+)
Predictors have severe size limits – AlphaFold2 (~2,500 residues), ESMFold (~4k). They cannot handle viral capsids, ribosomes, or large multimeric assemblies.

Cryo‑EM / tomography often produce medium‑resolution density maps → initial atomic models need physical relaxation. Homology models of large domains may contain steric clashes, distorted geometry, and incorrect sidechain packing.

**REAL FOLD ONE** runs on any length (tested up to 100k+ residues) using sparse graphs and O(N) memory, without retraining. It refines structure while preserving global topology.

---

## Key Features

- **SOC Controller** – learnable kernel $K_\alpha(r) = (r+\epsilon)^{-\alpha} e^{-r/\lambda}$, avalanche‑driven stress relaxation, adaptive temperature & friction.
- **Full‑atom protein sidechains** – all 20 amino acids with correct topology (no duplicate CB).
- **Full‑atom DNA/RNA** – nucleotides with base pairing, stacking, and backbone.
- **Ligand support** – SDF, MOL2, PDB ligands with GAFF‑style force field.
- **Multimer handling** – chain boundaries, cross‑chain interactions, chain‑break penalties.
- **Advanced electrostatics** – Sparse PME, geometric multigrid Poisson solver, block‑wise multipole correction.
- **Multiscale refinement** – RG coarse‑graining for fast convergence.
- **Itô SDE** – Milstein‑scheme Langevin dynamics for stochastic exploration.
- **Malliavin sensitivity** – compute Greeks for temperature or other parameters.
- **Antibody design** – CDR H3 loop modelling with Rosetta‑style scoring.
- **DNA origami** – scaffold routing, staple design, oxDNA export, BV topology check.
- **High‑throughput scanning** (HTS module) – single‑mutation ΔΔG scan, epistasis analysis, publication‑ready plots.
- **Scalability** – O(N) graphs, chunked edge building, CPU‑only mode for low‑RAM environments (3 GB), multi‑GPU parallelism.

---

## Installation

```bash
git clone https://github.com/your-org/real-fold-one.git
cd real-fold-one
pip install -r requirements.txt
```

Requirements

· Python ≥ 3.9
· PyTorch ≥ 2.0
· numpy, pandas, tqdm
· Optional: torch_cluster (for fast neighbour graphs), biotite (for PDB/mmCIF I/O), matplotlib+seaborn (for HTS plots), networkx (for DNA origami BV)

---

Quick Start

Refinement

```bash
python real_fold_one.py refine -i input.pdb -o refined.pdb --steps 300 --device cuda
```

For advanced electrostatics:

```bash
python real_fold_one.py refine -i input.pdb --pme --block_lr --device cuda
```

Energy evaluation (single point)

```bash
python real_fold_one.py refine -i input.pdb --steps 0
```

Antibody design

```bash
python real_fold_one.py antibody --antigen antigen.pdb --output antibody.pdb
```

DNA origami

```bash
python real_fold_one.py origami --shape shape.json --output my_origami --bv_check
```

Mutation scanning (HTS)

```bash
python real_fold_one_hts.py --pdb input.pdb --scan --gpu
```

Epistasis scan

```bash
python real_fold_one_hts.py --pdb input.pdb --epistasis --max_epi_pairs 1000 --gpu
```

---

How It Works

1. Load structure – CA coordinates and sequence (from PDB or DNA/RNA helix).
2. Build hierarchical neighbour lists – clash, LJ, and electrostatic cutoffs.
3. Reconstruct backbone – N, C, O positions.
4. Build full‑atom sidechains – using internal coordinates and χ torsion angles.
5. Define physics energy – bond, angle, Ramachandran, clash, H‑bond, electrostatics (with optional PME/Multigrid), solvation, SOC.
6. Refine – Adam optimizer with SOC‑controlled Langevin noise, avalanche gradients, and RG coarse‑graining.
7. Output – refined CA coordinates (and optional trajectory).

---

SOC Engine

The Self‑Organised Criticality controller brings three unique mechanisms:

· Avalanche propagation – when a region accumulates stress, the gradient is redistributed to neighbours via the learnable kernel $K_\alpha$.
· Adaptive temperature – computed from the system’s structural displacement (sigma), low‑pass filtered through SSC.
· Learnable α – the power‑law exponent is optimised during refinement to match the system’s correlation length.

These properties help escape local minima, accelerate convergence, and improve physical realism, especially in large, flexible assemblies.

---

HTS Module

The real_fold_one_hts.py script enables:

· Full single‑mutation scan – for every position, all possible monomer substitutions.
· Epistasis scan – double‑mutant ΔΔG, additivity, and epistasis (ε) computation.
· Local relaxation – only a window around the mutation is relaxed for speed.
· Outputs – CSV tables, ΔΔG distributions, mutational landscape heatmaps, position profiles, epistasis histograms, additivity scatter plots.

Ideal for deep mutational scanning (DMS), protein stability engineering, and predicting resistance mutations.

---

Performance

· CPU (Intel i9, 3 GB RAM): ~1–2 residues/second for small proteins.
· GPU (NVIDIA T4): ~30–100 residues/second.
· Multi‑GPU: linear speed‑up with number of GPUs (HTS).
· Large systems (50k residues): refinement runs in hours on a single GPU.

---

Citing

If you use REAL FOLD ONE in your research, please cite:

Limsuwan, Y. (2026). REAL FOLD ONE: SOC‑Controlled Universal Refinement Engine. 

---

License

MIT License. See LICENSE file for details.

---

Contact

Yoon A Limsuwan – [msps4u@gmail.com]

Project link: https://github.com/your-org/real-fold-one

```
