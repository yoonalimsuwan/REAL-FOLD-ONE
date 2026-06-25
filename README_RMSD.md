# README_RMSD.md
## Structural Domain Assembly ONE (SDA-ONE) — RMSD Status

This document explains **why** `structural_domain_assembly_one.py` exists,
**what** it is expected to fix, and — most importantly — **what is and is
not currently known about the RMSD numbers it produces.**

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

## 3. RMSD: what is actually known right now

**There is no measured RMSD number for SDA-ONE yet, in either direction.**

This is not a hedge — it is the accurate status, for three concrete
reasons:

1. **No trained weights.** `CrossDomainContactHead` is a newly written
   architecture. Its parameters are at random initialization. Any RMSD
   number quoted today would describe untrained noise, not the method.
2. **No execution.** The development environment used to write this
   module has no GPU/PyTorch runtime and no network access, so the
   module has been verified for syntax and logical structure
   (`python3 -m py_compile`, plus a 12-case self-test suite in
   `__main__`) but **has not been run end-to-end**, including the
   self-tests themselves.
3. **No benchmark.** Measuring "RMSD before vs. after" requires
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
  little or not at all on proteins that don't fit that shape (see §4).

---

## 4. What would have to happen before a real number exists

| Step | Status |
|---|---|
| Syntax / structural validation (`py_compile`) | ✅ Done |
| Self-test suite written (12 cases, `[PASS]`/`[FAIL]` format) | ✅ Done |
| Self-test suite **executed** on a machine with PyTorch | ❌ Not done |
| `CrossDomainContactHead` trained on real contact data | ❌ Not done |
| Benchmark set of large, multi-domain ground-truth structures assembled | ❌ Not done |
| Baseline (4-stage pipeline alone) RMSD measured on that same benchmark set | ❌ Not done |
| SDA-ONE RMSD measured on that same benchmark set | ❌ Not done |
| Comparison reported | ❌ Not done |

Until the last three rows are complete, "what is the best RMSD" does not
have a factual answer — only a directional expectation (lower than the
~4 Å baseline on genuinely multi-domain, contiguous-domain-architecture
proteins, by an amount that depends on contact-head recall and
domain-boundary accuracy, both currently unmeasured).

---

## 5. Known limitations that will cap how much SDA-ONE can help

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

---

## 6. Bottom line

> **Before:** RMSD ≈ 4 Å at ~100k residues (measured, existing 4-stage
> pipeline).
> **After SDA-ONE:** Not yet known. Directionally expected to be lower
> on multi-domain, contiguous-architecture proteins. No number — better
> or worse — should be quoted until the self-tests run successfully and
> a real benchmark comparison exists.

The honest next step is running the self-test suite locally, then
building the smallest possible benchmark (one or two large, well-
characterized multi-domain PDB structures) to get the first real data
point.
