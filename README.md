`
# REAL FOLD ONE – SOC‑Controlled Universal Refinement Engine

**REAL FOLD ONE** is a training‑free, all‑atom refinement engine for protein and nucleic acid structures, driven by **SOC (Self‑Organized Criticality)** control. It refines initial structures from any source (experimental maps, predictors, de novo design, homology models) using physics‑based energy functions, adaptive temperature, and avalanche gradients. **Scales to N > 100,000 residues** with sparse graphs and chunked operations.

## Why Refinement?

### For De Novo & Engineered Proteins (small to medium, N < 500)

- **Predictors/detectors** (AlphaFold3, ESMFold, RFdiffusion, ProteinMPNN) produce plausible backbones but often have:
  - Sidechain clashes or wrong rotamers
  - Distorted bond lengths/angles
  - Unphysical Ramachandran outliers
- **REAL FOLD ONE** fixes all‑atom geometry, repacks sidechains, and resolves steric clashes using full physics (LJ, Coulomb, torsion, H‑bond, Ramachandran) – without retraining.
- Perfect as a **physics‑based polish** after any de novo design pipeline.

### For Large Natural Proteins / Complexes (N > 5,000 up to 100k+)

- **Predictors have severe size limits** – AlphaFold2 (~2,500 residues), ESMFold (~4k). They cannot handle viral capsids, ribosomes, or large multimeric assemblies.
- **Cryo‑EM / tomography** often produce medium‑resolution density maps → initial atomic models need physical relaxation.
- **Homology models** of large domains may contain steric clashes, distorted geometry, and incorrect sidechain packing.
- **REAL FOLD ONE** runs on any length (tested up to 100k+ residues) using sparse graphs and O(N) memory, **without retraining**. It refines structure while preserving global topology.

## ✨ Key Features

- **SOC Controller** – Learnable CSOC kernel `K_α(r) = (r+ε)^(−α)·exp(−r/λ)`, avalanche gradient, adaptive temperature.
- **Semantic‑State Contraction (SSC v6)** – Deterministic fixed‑point operator (`ε_FP = 0.0028`, `σ → 1`).
- **DiffRGRefiner** – Renormalization group coarse‑graining for large systems.
- **Full‑atom protein** – Sidechains, LJ, Coulomb, torsion (χ), Ramachandran, H‑bond, solvation, clash.
- **Full‑atom DNA/RNA** – Nucleotide topology, base pairing, stacking, backbone, LJ, Coulomb.
- **Full‑atom ligand** – SDF/MOL2/PDB support, protein‑ligand interaction (LJ + Coulomb).
- **Multimer & chain boundaries** – Cross‑chain interactions via sparse graphs.
- **Antibody design** – CDR H3 loop remodeling, Rosetta scoring, affinity prediction (GNN).
- **DNA origami** – Scaffold routing, staple design, oxDNA export, BV topological validation.
- **Itô calculus & Malliavin sensitivity** – Milstein scheme, tangent process, Greek estimation.
- **HTS support** – `compute_energy()`, `relax_local()` for mutation scanning.
- **CLI** – `refine`, `antibody`, `origami` commands.
- **Scalable** – O(N) memory, sparse graphs, chunked, mixed precision, GPU/CPU.

## 🔧 Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- NumPy
- (Optional) `biotite` – for PDB/mmCIF reading
- (Optional) `torch_cluster` – fast neighbor search
- (Optional) `networkx` – DNA origami cycle detection

Install dependencies:
```bash
pip install torch numpy biotite
```

🚀 Quick Start

Refine a de novo designed protein (from RFdiffusion/ProteinMPNN output)

```bash
python real_fold_one.py refine -i designed.pdb -o refined.pdb --steps 300 --gpu
```

Refine a large natural complex (e.g., ribosome from cryo‑EM model)

```bash
python real_fold_one.py refine -i large_complex.pdb -o refined.pdb --steps 600 --gpu
```

Refine with ligands

```bash
python real_fold_one.py refine -i complex.pdb --ligand ligand.sdf ligand.mol2 -o refined.pdb
```

Antibody design (CDR H3 remodeling)

```bash
python real_fold_one.py antibody --antigen antigen.pdb --cdr_start 95 --cdr_end 102 --output antibody.pdb
```

DNA origami design from JSON (vertices/edges)

```bash
python real_fold_one.py origami --shape design.json --output my_origami --bv_check
```

📖 Python API

```python
from real_fold_one import RefinementEngine, RefinementConfig

cfg = RefinementConfig(device='cuda', steps=600, lr=1e-4)
engine = RefinementEngine(cfg)

result = engine.refine(coords, sequence, ligand_files=['drug.sdf'])
refined_coords = result['coords']
final_energy = result['final_energy']
sigma = result['sigma']
temperature = result['temperature']
```

Compute energy of a structure

```python
energy = engine.compute_energy(coords, sequence)
```

Local relaxation around specific residues

```python
refined_coords, final_energy = engine.relax_local(coords, sequence, positions=[42, 55], steps=30)
```

🧬 Command Line Options

refine subcommand

Argument Description
--input, -i Input PDB/mmCIF file
--chain Chain ID (default: auto)
--output, -o Output PDB file
--steps Number of refinement steps (default: 600)
--lr Learning rate (default: 1e-4)
--ligand One or more ligand files (SDF/MOL2/PDB)
--boundaries Chain boundaries (indices)
--device cpu, cuda, or auto
--no_rg Disable DiffRGRefiner
--no_ssc Disable SSC fixed‑point
--milstein Use Milstein scheme for Langevin
--trajectory Save full trajectory to .npy

antibody subcommand

Argument Description
--antigen Antigen PDB file
--cdr_start Start residue index of CDR H3 (default: 95)
--cdr_end End residue index of CDR H3 (default: 102)
--output Output PDB for designed antibody

origami subcommand

Argument Description
--shape JSON file with vertices and edges
--output Base name for oxDNA .top and .dat files
--bv_check Verify topological consistency via BV formalism

📁 Output Files

· refined.pdb – Refined CA‑only coordinates (backbone trace)
· refined_traj.npy – Trajectory of best coordinates (if --trajectory)
· oxDNA files: <output>.top, <output>.dat

⚙️ Configuration

Modify RefinementConfig parameters to fine‑tune physics weights:

```python
cfg = RefinementConfig(
    w_bond=30.0,      # bond stretching
    w_angle=15.0,     # angle bending
    w_rama=8.0,       # Ramachandran
    w_clash=80.0,     # clash penalty
    w_soc=0.3,        # SOC contact energy
    base_temp=300.0,  # initial temperature (K)
    friction=0.02,    # Langevin friction
    sigma_target=1.0, # target structural avalanche
    ...
)
```

📚 References

· SOC theory: Bak, Tang & Wiesenfeld (1987)
· CSOC kernel: adaptive criticality for molecular systems
· BV formalism: Henneaux & Teitelboim (1992)

📄 License

MIT License – free for academic and commercial use.

👤 Author

Yoon A Limsuwan

---

REAL FOLD HTS – High‑Throughput Mutation Scanning & Epistasis Engine

REAL FOLD HTS performs comprehensive mutational scanning for proteins, DNA, RNA, and mixed multimers using the REAL FOLD ONE engine as backend. It computes ΔΔG for all possible single mutations and double‑mutation epistasis, with local relaxation and multi‑GPU support.

✨ Key Features

· Full single‑mutation scan – All positions × all monomers (20 AAs, 4 nucleotides)
· Epistasis scanning – Double mutations, additive vs double‑mutant ΔΔG, epistasis (ε)
· Auto‑detection – Sequence type (protein, DNA, RNA)
· Local relaxation – Only residues near mutation are optimized (window size adjustable)
· Parallel evaluation – Multi‑GPU / multi‑CPU via torch.multiprocessing
· Rich output – CSV, JSON, heatmaps, ΔΔG distributions, position profiles, epistasis plots

🔧 Requirements

· Same as REAL FOLD ONE (PyTorch, NumPy, biotite optional)
· Additional: pandas, matplotlib, seaborn, tqdm

```bash
pip install pandas matplotlib seaborn tqdm
```

🚀 Quick Start

Single‑mutation scan on a de novo designed protein (from PDB)

```bash
python real_fold_hts.py --pdb designed.pdb --scan --gpu
```

Scan natural protein for stability‑enhancing mutations

```bash
python real_fold_hts.py --pdb natural.pdb --scan --ddg_threshold 0.5 --gpu
```

Epistasis scan (double mutations)

```bash
python real_fold_hts.py --pdb multimer.pdb --epistasis --max_epi_pairs 500 --gpu
```

📖 Output Files (in --output directory)

File Description
single_mutations.csv ΔΔG for every single mutation
epistasis.csv Double‑mutation results, epistasis values
ddg_distribution.png Histogram of ΔΔG
mutational_landscape.png Heatmap (position vs mutant)
position_profile.png Mean ΔΔG ± std per position
epistasis_distribution.png Histogram of epistasis ε
additivity_scatter.png Additive vs double‑mutant ΔΔG scatter
summary.json Summary statistics

🧬 Command Line Options

Argument Description
--pdb Input PDB/mmCIF file
--seq Sequence (auto‑detect type, DNA/RNA only)
--chain Chain ID (default: auto)
--output, -o Output directory (default: ./hts_output)
--scan Perform full single‑mutation scan
--epistasis Perform epistasis scan (double mutations)
--relax_steps Local relaxation steps per mutant (default: 30)
--window Relaxation window (± residues, default: 3)
--ddg_threshold Significance threshold for ΔΔG / epistasis (default: 0.5 kcal/mol)
--gpu Use GPU (default: CPU)
--max_epi_pairs Maximum number of position pairs for epistasis (default: 500)

📊 Example Output (CSV)

single_mutations.csv:

```
chain,pos_in_chain,global_pos,wt,mut,ddg,type,e_wt,e_mut
0,12,12,A,G,0.45,transition,-123.4,-123.0
0,12,12,A,C,0.78,transversion,-123.4,-122.6
...
```

epistasis.csv:

```
chain1,pos1,mut1,wt1,chain2,pos2,mut2,wt2,ddg1,ddg2,ddg_double,ddg_additive,epistasis,significant
0,23,A,G,0,45,V,A,-0.3,0.5,-0.2,0.2,-0.4,False
...
```

⚙️ Performance Tips

· Use --gpu for large proteins (>500 residues)
· For epistasis, use --max_epi_pairs to limit combinatorial explosion
· Increase --window if mutations are in flexible regions (e.g., loops)
· For DNA/RNA, providing a PDB is better than de novo helix (more accurate)

🔗 Integration with REAL FOLD ONE

REAL FOLD HTS calls RefinementEngine.compute_energy() and RefinementEngine.relax_local() from REAL FOLD ONE. Ensure real_fold_one.py is in the same directory or Python path.

📄 License

MIT License

👤 Author

Yoon A Limsuwan

---

For academic and industrial protein/DNA/RNA design, mutation screening, and rational engineering.

```
