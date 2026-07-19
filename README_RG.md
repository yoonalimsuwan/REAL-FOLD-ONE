# RG Refiner

Multiscale coarse-graining / smoothing for folded-chain CA coordinates, part
of the REAL FOLD ONE cluster.

**Author:** Yoon A Limsuwan — MSPS NETWORK
**AI Co-Developer:** Claude (Anthropic) — diagnosed the bond-length-collapse
failure mode in the original `DiffRGRefiner`, designed and validated the
SHAKE-constrained replacement, and implemented the real training loop for
`LearnedRGRefiner`.
**License:** MIT

---

## What's in this file

| Component | Status | Requires |
|---|---|---|
| `shake_constrained_refine` | Validated baseline | numpy only |
| `LearnedRGRefiner` | Trainable model, **not yet trained** | PyTorch |
| `train_learned_rg_refiner` | Working trainer | PyTorch (GPU recommended) |

## Why this exists

An earlier refiner (`DiffRGRefiner`, removed from `real_fold_one_v2.py`)
pooled/interpolated absolute CA coordinates by sequence index. This
corrupted real folds — confirmed on 8 PDB structures — because it assumes
position varies smoothly with residue *index*, which turns, loops, and sheet
reversals violate constantly. A direction-vector-pooling fix was tried next
and failed worse (unit-vector averaging on a chain that changes direction
often shrinks toward zero, then renormalizing amplifies noise).

## 1. `shake_constrained_refine` — validated baseline

Alternates Laplacian smoothing (nudge each point toward its sequence
neighbors' mean) with SHAKE-style bond-length projection (restore every
bond to its exact original length). Numpy only, no training required.

**Measured performance** (8 real PDB structures, noise 0.5/1.0/2.0 Å, 5 seeds):
- Zero-noise sanity check: RMSD-to-true ≈ 1.25 Å, bond length max deviation ≈ 0.012 Å
- With noise: reduces RMSD ≈ 31% vs. unrefined input
- Beats a plain moving-average filter (window=4) in 12/24 tested conditions —
  a legitimate baseline, not a clear win

**Known limitation:** the Laplacian step still drifts bond *angles* slightly
over many outer iterations even though SHAKE keeps every bond *length*
exact — that's why zero-noise RMSD is ~1.25 Å instead of ~0.

```python
from rg_refiner import shake_constrained_refine
refined = shake_constrained_refine(ca_coords)  # ca_coords: (L, 3) numpy array
```

## 2. `LearnedRGRefiner` — trainable model (untrained by default)

Predicts a smoothed bond *direction* from a local window of noisy bond
vectors; reconstructs positions using the chain's own input bond lengths, so
bond length is exact by construction regardless of what the network learns.
This is a fresh `nn.Module` until you train it — instantiating it does not
give you a working refiner.

```python
from rg_refiner import LearnedRGRefiner
model = LearnedRGRefiner(hidden_dim=64, window=5, n_layers=2)
```

## 3. `train_learned_rg_refiner` — real trainer

```python
from rg_refiner import RGTrainingConfig, train_learned_rg_refiner

cfg = RGTrainingConfig(
    pdb_dir="data/train_pdbs",           # real training structures
    held_out_pdb_dir="data/val_pdbs",    # DISJOINT set, validation only
    epochs=200,
    batch_size=8,
    device="cuda",
    save_path="rg_refiner_best.pt",
)
model, history = train_learned_rg_refiner(cfg)
```

Or from the command line:

```bash
python rg_refiner.py --pdb_dir data/train_pdbs --held_out_pdb_dir data/val_pdbs \
    --epochs 200 --device cuda --save_path rg_refiner_best.pt
```

What it does each run:
1. Loads every `.pdb` in `pdb_dir` / `held_out_pdb_dir`, parses CA coordinates,
   drops chains shorter than `min_chain_len`. **Raises immediately** if either
   directory has no usable structures, or if the two directories overlap —
   it will not silently train on nothing or validate on training data.
2. Each epoch: for every training chain, adds Gaussian noise at a randomly
   chosen level from `noise_levels`, runs the model, computes Kabsch-RMSD
   loss to the clean structure, backprops (gradient-accumulated over
   `batch_size` chains — chains vary in length, so there's no padded tensor
   batch).
3. Every `eval_interval` epochs: evaluates on held-out data at every noise
   level, reporting three numbers side by side — **model RMSD**, **shake
   baseline RMSD**, **raw noisy-input RMSD** — and saves a checkpoint only
   when the model's held-out RMSD improves.
4. At the end, if the best model still doesn't beat the shake baseline, it
   prints a warning rather than letting the run pass as a success.

**Not yet run in production**: this trainer has been smoke-tested on the
non-training code paths (parsing, baseline eval) in an environment without
GPU/network access to install PyTorch, so the actual gradient/training loop
has not been executed end-to-end here. Run a short sanity pass (a handful of
epochs on a small structure set) before trusting a full training run.

## Command-line self-test (no training, no torch required)

```bash
python rg_refiner.py --pdb_dir some_pdb_dir
```

Runs `shake_constrained_refine` only and prints RMSD-at-zero-noise and max
bond-length deviation per structure — useful as a quick sanity check that
the numpy baseline still behaves as documented.

## Design conventions followed

- `soft` numerical guards (`+ 1e-9`) instead of hard clamps
- Every accuracy claim in this file is a number from an actual measured run,
  not an aspiration
- Explicit `FileNotFoundError` / `ValueError` on bad training config rather
  than a placeholder that looks like it works but doesn't validate anything
- The learned model is only ever claimed to be an improvement if it beats
  `shake_constrained_refine` on the same held-out structures and noise levels
