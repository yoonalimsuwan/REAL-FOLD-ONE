# README_RMSD.md
## Structural Domain Assembly ONE (SDA-ONE) + Experimental Restraints ONE (XR-ONE) — RMSD Status

This document explains **why** `structural_domain_assembly_one.py` and
`experimental_restraints_one.py` exist, **what** they are expected to fix,
and — most importantly — **what is and is not currently known about the
RMSD numbers they produce.**

---

## 1. The problem this module addresses

Across the existing REAL FOLD ONE pipeline —

```
seq_to_coarse_structure.py  →  structural_gno_fold_v3.py (StructuralGNOFold)
    →  structural_langevin_fold_v2.py (BAOAB integration)
    →  real_fold_one_v2.py (RefinementEngine / OpenMM-ML)
```

— the best observed RMSD at the ~100,000-residue scale has been around
**4 Å**, even after all four stages run to completion.

The root cause is **not** in the later refinement stages. It is in the
first stage. In `seq_to_coarse_structure.py`, once sequence length exceeds
`Seq2CoarseConfig.auto_window_attn_threshold` (default 8,000 residues),
three behaviors switch on at once:

| Component | Below threshold | Above threshold |
|---|---|---|
| `SequenceTransformerEncoder` | full bidirectional attention | sliding-window attention (±256 residues) |
| `SequenceEmbedder` | ESM-2 (evolutionary prior) | learned fallback embedding (no evolutionary prior) |
| `DifferentiableMDS` init | full pairwise MDS | Landmark MDS (approximate) |

Each of these independently discards **long-range / global** structural
information. The downstream stages — message passing, Langevin
integration, OpenMM-ML refinement — are all *local* refiners: they can
make a given topology's geometry more physically realistic, but they
cannot recover a topology that was never represented in the first place.

**This is why the 4 Å figure behaves like a ceiling rather than a noise
floor.** It is a wrong-topology error, not a noisy-topology error, and
no amount of additional local refinement removes it.

---

## 2. What SDA-ONE does about it

`structural_domain_assembly_one.py` (v1.0.0) never asks any single
attention or MDS call to reason about the whole 100k-residue chain at
once. Instead:

1. **`DomainSegmenter`** splits the chain into domains of roughly
   500–2,000 residues, using the existing `SigmaHead` field (regime
   changes in σ(x) tend to fall at domain/linker boundaries) plus a
   contact-density signal.
2. Each domain is folded **independently** by the existing
   `SeqToCoarseStructure`, small enough that none of the three
   auto-switches above ever fire — full attention, full ESM-2 embedding,
   full MDS, all preserved. Each domain is also **recycled** (folded
   `num_recycles` times, feeding coordinates back in as a warm start)
   using the `init_coords` argument `SeqToCoarseStructure.forward`
   already exposes — no change to that module was needed.
3. **`CrossDomainContactHead`** predicts a sparse set of inter-domain
   Cα–Cα contacts (never materializing a full N×N matrix).
4. **`DomainDockingAssembler`** rigidly places each domain (rotation +
   translation only — internal domain geometry is never distorted) into
   a single global frame, driven by those contacts via an LJ-style
   potential reused directly from `real_fold_one_v2.py`'s `CSOCKernel`.

The output dict (`init_coords`, `seq_features`, `sigma`) matches
`SeqToCoarseStructure.forward` exactly, so it is a drop-in replacement
feeding into the same `write_ca_pdb` → `RefinementEngine.refine` chain
already in use.

---

## 3. Optional layer: real experimental restraints (XR-ONE)

`experimental_restraints_one.py` is a **standalone** extension (it imports
from `structural_domain_assembly_one.py` and subclasses
`DomainDockingAssembler`; it does not modify that file, and deleting it
leaves SDA-ONE working exactly as before).

It exists because §6 below flags `CrossDomainContactHead`'s contact
predictions as unvalidated. Real experimental data does not have that
problem — it is a direct physical measurement, with its own (different,
better-characterized) error sources. Where available, it can anchor the
docking step instead of relying on the learned contact head alone:

| Restraint source | What it provides | How it's used |
|---|---|---|
| Cross-linking mass spectrometry (XL-MS) | Pairs of residues close enough for a crosslinking reagent to bridge them, i.e. an upper-bound Cα–Cα distance per pair | One-sided (flat-bottom) penalty: zero cost while within the crosslinker's reach, quadratic cost beyond it |
| Cryo-EM density map (MRC/CCP4) | A 3-D density volume the assembled structure should fit inside | Gaussian-splat the current coordinates into a simulated density, score against the real map via cross-correlation |

Both restraint types plug into the *same* docking energy that
`DomainDockingAssembler` already minimizes — they are added as extra terms
alongside the existing contact-head and steric-clash energies, each with
an independently tunable weight, so they can be used together, separately,
or not at all (the base SDA-ONE behavior).

`experimental_restraints_one.py` also ships its own minimal MRC/CCP4 file
reader (no dependency on the `mrcfile` package, which wasn't available in
the authoring environment) — it covers the common MRC2014 header fields
and the three most common data modes (int8/int16/float32), which is the
overwhelming majority of maps distributed via EMDB.

---

## 4. RMSD: what is actually known right now

**There is no measured RMSD number for SDA-ONE or XR-ONE yet, in either
direction.**

This is not a hedge — it is the accurate status, for four concrete
reasons:

1. **No trained weights.** `CrossDomainContactHead` is a newly written
   architecture. Its parameters are at random initialization. Any RMSD
   number quoted today would describe untrained noise, not the method.
2. **No execution of the autograd / docking path.** Neither module's
   GPU/PyTorch code path has been run end-to-end — the development
   environment used to write both files has no PyTorch runtime and no
   network access. What *has* been verified directly (see §5) is the
   pure Python/NumPy math underneath the restraint logic — MRC file
   parsing and the Gaussian-splat cross-correlation formula both ran and
   produced correct results outside of PyTorch. The torch-based pieces
   (autograd through the rigid-body docking loop, the quaternion
   parameterization, the full multi-domain assembly) remain unrun.
3. **No real experimental data plugged in yet.** XR-ONE's restraint
   logic has only been exercised on synthetic data (a hand-built
   crosslink pair, a density map generated from known coordinates). It
   has not yet been run against an actual XL-MS dataset or an actual
   EMDB map for a real protein.
4. **No benchmark.** Measuring "RMSD before vs. after" requires
   ground-truth structures at the relevant scale (large, genuinely
   multi-domain proteins with known coordinates). No such benchmark set
   has been assembled or run against yet.

Any of the following claims would currently be **unsupported** and
should not be repeated as if measured:

- A specific new RMSD value (e.g. "RMSD drops to 2 Å")
- A percentage improvement (e.g. "40% better than baseline")
- A guarantee that SDA-ONE outperforms the 4-stage baseline on every
  protein — it is expected to help most on proteins with genuine,
  sequence-contiguous multi-domain architecture, and is expected to help
  little or not at all on proteins that don't fit that shape (see §6)

- A guarantee that adding XL-MS or Cryo-EM restraints automatically
  improves RMSD over SDA-ONE alone — restraint-guided docking is only as
  good as the restraints given to it (garbage-in-garbage-out), and this
  has not been measured either

---

## 5. What would have to happen before a real number exists

| Step | Status |
|---|---|
| SDA-ONE syntax / structural validation (`py_compile`) | ✅ Done |
| SDA-ONE self-test suite written (12 cases, `[PASS]`/`[FAIL]` format) | ✅ Done |
| XR-ONE syntax / structural validation (`py_compile`) | ✅ Done |
| XR-ONE self-test suite written (14 cases) | ✅ Done |
| XR-ONE pure-NumPy verification of MRC reader + cross-correlation math (outside PyTorch, run and confirmed correct) | ✅ Done |
| Either self-test suite **executed end-to-end with PyTorch** (autograd, docking loop) | ❌ Not done |
| `CrossDomainContactHead` trained on real contact data | ❌ Not done |
| XR-ONE run against a real XL-MS dataset or EMDB map | ❌ Not done |
| Benchmark set of large, multi-domain ground-truth structures assembled | ❌ Not done |
| Baseline (4-stage pipeline alone) RMSD measured on that same benchmark set | ❌ Not done |
| SDA-ONE (no restraints) RMSD measured on that same benchmark set | ❌ Not done |
| SDA-ONE + XR-ONE (with restraints) RMSD measured on that same benchmark set | ❌ Not done |
| Comparison reported | ❌ Not done |

Until the bottom rows are complete, "what is the best RMSD" does not have
a factual answer — only a directional expectation (lower than the ~4 Å
baseline on genuinely multi-domain, contiguous-domain-architecture
proteins, by an amount that depends on contact-head recall,
domain-boundary accuracy, and — if used — how much real restraint data is
available and how reliable it is; none of these are currently measured).

---

## 6. Known limitations that will cap how much improvement is possible

Carried over from the module's own docstring, because they directly
bound what RMSD improvement is even possible:

- **`DomainSegmenter` is a heuristic**, not a learned domain predictor.
  It will misplace boundaries on interleaved or repeat-domain
  architectures where "domain" isn't a contiguous sequence span.
- **`CrossDomainContactHead`'s candidate-pair recall is unverified.**
  If true inter-domain contacts are pruned away by the k-NN candidate
  search before scoring, no amount of docking can recover them.
- **Docking is only as good as its contacts.** Garbage-in-garbage-out
  applies directly — and docking assumes each domain is near-rigid
  internally, which is a poor fit for long flexible linkers regardless
  of contact quality.
- **XL-MS default distance cutoff (30 Å) is literature-typical, not
  validated against any specific experiment.** The correct cutoff
  depends on the actual crosslinker chemistry used; always confirm
  before trusting the default.
- **XL-MS has a real false-positive rate** (the per-crosslink
  `confidence` weight exists specifically to let noisy hits be
  down-weighted, but supplying that confidence correctly is the user's
  responsibility — the module cannot infer it from the data itself).
- **The Cryo-EM restraint is a coarse, one-Gaussian-per-Cα approximation**
  with no side-chain mass or B-factor modeling. It is reasonable for
  global rigid-domain placement against a map, but is not a substitute
  for dedicated flexible-fitting tools when sub-domain flexibility
  against the map matters.
- **The native MRC reader does not handle every MRC variant** (e.g.
  unusual unit-cell geometry, symmetry records, extended headers some
  software writes) — it covers MODE 0/1/2, which is the common case, and
  raises a clear error rather than silently misreading anything else.

---

## 7. Bottom line

> **Before:** RMSD ≈ 4 Å at ~100k residues (measured, existing 4-stage
> pipeline).
> **After SDA-ONE alone:** Not yet known. Directionally expected to be
> lower on multi-domain, contiguous-architecture proteins.
> **After SDA-ONE + XR-ONE restraints:** Not yet known, and depends
> entirely on the quality/quantity of real experimental data supplied —
> potentially the largest source of improvement of everything described
> here, since it replaces an unvalidated learned guess with a direct
> physical measurement, but unmeasured until real data is run through it.
>
> No number — better or worse, for either module — should be quoted
> until the self-tests run successfully with PyTorch and a real benchmark
> comparison exists.

The honest next steps, in order: run both self-test suites locally with
PyTorch installed; if real XL-MS or Cryo-EM data is available, try XR-ONE
on it directly; then build the smallest possible benchmark (one or two
large, well-characterized multi-domain PDB structures with known
coordinates) to get the first real RMSD data point.
