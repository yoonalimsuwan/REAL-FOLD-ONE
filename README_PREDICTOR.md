# REAL FOLD ONE — MSA-Free Structure Predictor

**Modules covered:** `seq_to_coarse_structure.py` (predictor) · `run_msa_free_pipeline.py` (end-to-end orchestrator)
**Pipeline version:** see `PIPELINE_VERSION` in `run_msa_free_pipeline.py` · **Predictor version:** see `SEQ2COARSE_VERSION` in `seq_to_coarse_structure.py`

A fully differentiable, **single-sequence, MSA-free** protein structure predictor for the REAL FOLD ONE ecosystem. No alignment search, no co-evolutionary profile, no template lookup, at any stage — only the raw amino-acid string goes in.

---

## 1. What this is

```
sequence (str)
    │
    ▼
SeqToCoarseStructure        (seq_to_coarse_structure.py)
    │  init_coords, seq_features, sigma
    ▼
StructuralGNOFold            (structural_gno_fold_v3.py — external)
    │  final_coords, pred_ddg
    ▼
Cα-only PDB
    │  (optional, Tier 2)
    ▼
RefinementEngine.refine(...)  (real_fold_one_v2.py — external)
    │
    ▼
All-atom refined structure + energy trace
```

`run_msa_free_pipeline.py` wires all of this into one call: `pipeline.predict(sequence)`. `seq_to_coarse_structure.py` is the first stage — the part that replaces what an MSA/Evoformer pipeline would otherwise need a sequence database and hours of alignment search to produce.

### Why MSA-free matters here

Standard structure predictors (AlphaFold2-style) need a Multiple Sequence Alignment: a search across genomic databases to find evolutionarily related sequences, from which co-evolutionary statistics are computed. That search can take minutes to days, and fails outright for sequences with no good homologs (orphan proteins, heavily engineered designs, fast-mutating viral proteins).

This predictor replaces that entire step with:

- A **pretrained protein language model (ESM-2)**, used as a frozen feature extractor — its evolutionary prior is already baked into pretrained weights, so no per-query search is needed.
- A **bidirectional transformer encoder** over the single sequence, supplying long-range context directly instead of via a co-evolution axis.
- A **differentiable MDS (SMACOF) solver** that turns a predicted distance distribution into 3-D coordinates, fully end-to-end trainable.

The trade-off is explicit: you give up the evolutionary signal an MSA provides, in exchange for being able to run on any sequence, instantly, including ones with no usable homologs.

---

## 2. Two-tier pipeline

### Tier 1 — AI surrogate (always available)

Pure PyTorch. No external dependency beyond the ecosystem's own files.

```
sequence → SeqToCoarseStructure → StructuralGNOFold → Cα coordinates (+ pred_ddg)
```

This is a complete, fully differentiable, MSA-free prediction on its own. If `structural_gno_fold_v3.py` is unavailable, `seq_to_coarse_structure.py` can still be used standalone — its own output (`init_coords`, `seq_features`, `sigma`) is exactly the triple `StructuralGNOFold.forward` expects, so it's a drop-in first stage for any compatible refinement module.

### Tier 2 — physics-based all-atom refinement (optional, best-effort)

```
Cα-only PDB → side-chain reconstruction (PDBFixer) → RefinementEngine.refine(...)
```

Requires `openmm` + `pdbfixer` + `real_fold_one_v2.RefinementEngine`. If any of these are missing, `run_msa_free_pipeline.py` logs a warning and silently falls back to Tier-1-only output — it never hard-fails because Tier 2 wasn't set up.

**Known gap, flagged honestly in the pipeline script itself:** `RefinementEngine._setup_system()` only fills in missing residues and missing hydrogens — it does not reconstruct missing side-chain heavy atoms. Since Tier 1 only ever produces a Cα-only PDB, `run_msa_free_pipeline.py` uses **PDBFixer** to build side chains via rotamer-library placement before handing the structure to `RefinementEngine`. PDBFixer's placement is based on backbone geometry alone — it has no knowledge of the network's actual predicted side-chain packing — so Tier 2 output should be read as *an approximate physical relaxation of the Tier-1 backbone*, not a from-scratch all-atom prediction. If your environment has the project-specific `build_sidechain_atoms` function (referenced by `evolution_one_v4.py` from a `one_core_evolution` module not included in this integration), swap it into `MSAFreePipeline._add_sidechains()` instead.

---

## 3. Quick start

```bash
# Tier 1 only (AI surrogate, no extra dependencies)
python run_msa_free_pipeline.py MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK --name my_protein

# Tier 1 + Tier 2 (physics-based refinement)
pip install openmm pdbfixer
python run_msa_free_pipeline.py <SEQUENCE> --name my_protein --tier2 --tier2-steps 600
```

```python
from run_msa_free_pipeline import PipelineConfig, MSAFreePipeline

cfg = PipelineConfig(run_tier2=False, output_dir="./outputs")
pipeline = MSAFreePipeline(cfg)
result = pipeline.predict("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK", name="my_protein")

print(result["tier1"]["pred_ddg"])
print(result["tier1"]["refined_pdb"])     # Path to the SGNO-refined Cα PDB
```

Using `SeqToCoarseStructure` on its own, without the rest of the pipeline:

```python
from seq_to_coarse_structure import Seq2CoarseConfig, SeqToCoarseStructure, write_ca_pdb

cfg = Seq2CoarseConfig(embed_backend="learned")   # or "esm2" — see §5
model = SeqToCoarseStructure(cfg)
out = model.predict("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK")

write_ca_pdb("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK",
             out["init_coords"], "my_protein_coarse.pdb")
```

### CLI flags (`run_msa_free_pipeline.py`)

| Flag | Default | Meaning |
|---|---|---|
| `sequence` | *(required)* | Single-letter amino-acid string. |
| `--name` | `"query"` | Output filename prefix. |
| `--output-dir` | `./msa_free_pipeline_outputs` | Where PDBs and intermediates go. |
| `--device` | `auto` | `auto` \| `cuda` \| `mps` \| `cpu`. |
| `--tier2` | off | Attempt physics-based all-atom refinement. |
| `--tier2-steps` | `600` | Optimisation steps for `RefinementEngine.refine`. |
| `--seq2coarse-checkpoint` | `None` | Path to a pretrained `SeqToCoarseStructure` checkpoint. |
| `--sgno-checkpoint` | `None` | Path to a pretrained `StructuralGNOFold` checkpoint. |

Without a checkpoint, both models run with randomly initialised weights — fine for verifying shapes/wiring end-to-end, not for real structure prediction.

---

## 4. Wiring contract between the two files

`PipelineConfig.__post_init__` enforces one rule automatically: `sgno_cfg.node_in_dim` must equal `seq2coarse_cfg.hidden_dim`, because `StructuralGNOFold.forward` consumes the *contextualised transformer latent* `seq_features` (width `hidden_dim`) as its node features — not the architecture's own default 20-dim one-hot encoding. If you build both configs yourself and the widths don't match, `PipelineConfig` raises a `ValueError` immediately rather than failing deep inside a forward pass. Leave `sgno_cfg=None` in `PipelineConfig` and this is handled for you.

The same contract is available directly from `seq_to_coarse_structure.py` via:

```python
from seq_to_coarse_structure import build_sgno_compatible_inputs
seq_features, init_coords, sigma = build_sgno_compatible_inputs(seq2coarse_output)
final_coords, pred_ddg = sgno_model(seq_features, init_coords, sigma)
```

---

## 5. Sequence-length scaling — what to set, and when

`seq_to_coarse_structure.py` defaults reproduce the original unchunked, full-precision, full-attention behaviour exactly. Every optimisation below is opt-in (manual config) or auto-switched only past a length threshold — short sequences see no behaviour change.

### 5.1 Memory-only optimisations (exact, no approximation)

These reduce *how much memory the forward pass needs*, not *what is computed* — verified numerically identical to the unchunked path in the predictor's own `[PASS]/[FAIL]` suite.

| Config field | Default | Effect |
|---|---|---|
| `pair_proj_factorized` | `True` | Splits the distogram head's first linear layer into two halves applied to `h_i`/`h_j` separately — avoids ever materialising the `(N, N, 2·hidden_dim)` concatenated pair tensor. Exact algebraic identity. |
| `pair_chunk_size` | `None` | Row-chunks the `(N, N, ·)` distogram computation. `None` = unchunked (original behaviour). |
| `pair_proj_dtype` | `None` | Runs the pairwise stage in reduced precision (e.g. `torch.bfloat16`); cast back before returning. |
| `mds_row_chunk_size` | `None` | Row-chunks each SMACOF Guttman-transform update instead of materialising full `(N, N)` intermediates. |
| `mds_use_landmarks` | `False` | Manually force Landmark MDS initialisation (O(N·L) instead of O(N³)) regardless of N. |

For the largest sequences, `DistogramHead.forward_expected_distance(h, use_2d_tiling=True)` (combined with `pair_chunk_size`) further tiles the pairwise MLP computation into `(block, block)` blocks rather than `(block, N)` strips — trading some wall-clock time for a genuine reduction in peak activation memory. Use `SeqToCoarseStructure.forward(..., return_distogram=False, use_2d_tiling=True)` to route through this fused path at inference time.

### 5.2 Why the `(N, N)` `expected_dist` matrix itself is *not* a bottleneck worth fusing away

It's tempting to think the `(N, N)` expected-distance matrix — handed from `DistogramHead` to `DifferentiableMDS` — needs to be streamed/fused to scale further. In practice it doesn't:

- It's a **2-D scalar matrix** (one float per residue pair), not a `(N, N, hidden_dim)` or `(N, N, bins)` tensor — those *are* eliminated by §5.1's chunking, and they were the actual O(N²·d) cost driver.
- At `N = 34,000` (Titin, the longest known single polypeptide chain), `expected_dist` in `float32` is **~4.6 GB** — comfortably within a single 24–48 GB GPU.
- At `N = 100,000`, it's **~40 GB** — feasible on a single 80 GB GPU (A100/H100), without fusing anything.

Fusing `DistogramHead` directly into the SMACOF loop (re-evaluating the distogram MLP on-the-fly every iteration instead of computing it once and caching it) would trade this manageable memory cost for a *much* larger compute-time cost — SMACOF runs ~200 iterations by default, so the MLP would need re-evaluating 200×. **This is deliberately not implemented** — the current architecture is the better trade-off for production use. Treat the gap as closed by design, not as missing work.

### 5.3 Scaling past ~8,000 residues: structural changes (approximate, opt-in / auto-switched)

These genuinely change *what* is computed (not just how memory is laid out), trading some model capacity / initialisation exactness for tractability at extreme N. All default to the original behaviour and only activate past a threshold, or when set explicitly.

| Config field | Default | Auto-switch threshold field | What happens past the threshold |
|---|---|---|---|
| `attn_window_size` | `None` (full O(N²) attention) | `auto_window_attn_threshold` (default `8000`) | Switches `SequenceTransformerEncoder` to banded sliding-window attention (`attn_window_size_default`, default `256`) — O(N·w) instead of O(N²). Residues farther apart than the window no longer attend directly in a single layer; long-range signal still mixes indirectly across stacked layers. |
| `mds_use_landmarks` | `False` (full classical MDS) | `auto_landmark_threshold` (default `8000`) | Switches MDS initialisation to Landmark MDS (`mds_num_landmarks` landmarks) — O(L³ + N·L·mds_dim) instead of O(N³). Full classical MDS's eigendecomposition becomes both infeasible and numerically unreliable well before this point regardless of available memory. |
| `embed_backend` | `"esm2"` | `auto_learned_embed_threshold` (default `8000`) | Routes that call through the `"learned"` embedding fallback instead of ESM-2. ESM-2 was never trained on contexts this long; its attention becomes unreliable and OOM-prone there. Only affects calls past the threshold — `cfg.embed_backend` itself is left unchanged, so shorter sequences in the same model instance keep using ESM-2. |

All three thresholds can be set to a very large number to disable auto-switching and keep full manual control (e.g. if you want full attention regardless of N, or want to force Landmark MDS at a different N than the default).

`max_seq_len` defaults to `120,000`. The `"learned"` backend's sinusoidal positional-encoding table is built **lazily** — it grows on demand to the longest sequence actually seen in the process, rather than being precomputed to `max_seq_len` rows at model-construction time.

### 5.4 Practical recipe for N ≳ 50,000

```python
cfg = Seq2CoarseConfig(
    pair_chunk_size=512,
    mds_row_chunk_size=512,
    # attn_window_size, mds_use_landmarks, embed_backend: leave as default
    # and let the auto-switch thresholds handle it at this scale.
)
model = SeqToCoarseStructure(cfg)
out = model.predict(long_sequence, return_distogram=False, use_2d_tiling=True)
```

Expect **wall-clock time**, not memory, to be the dominant constraint at this scale — see §6.

---

## 6. What "scaling up" actually buys you here

MSA-based pipelines hit a hard **memory wall** well before N reaches even 10,000: their core tensor scales as `(M, N, channels)` (M = alignment depth), and attention over it costs `O(M·N²)` or `O(M²·N)`. No amount of patience fixes that — the job simply doesn't fit in any GPU.

This architecture eliminates the `M` axis entirely (MSA-free by construction), which converts the problem from **memory-bound to compute-bound**: it doesn't run out of VRAM at large N, it just takes longer. At `N ≈ 100,000` on a single top-end GPU, that "longer" can mean hours to days for one sequence's SMACOF solve — but the job *completes*, where an MSA-based model would have failed outright with `CUDA out of memory` long before reaching that length.

The honest framing: the right comparison isn't "is this faster than an MSA model at small N" — at small N it generally isn't, since it forgoes the evolutionary signal an MSA provides. The comparison that matters is "does this complete at all at N this large" — and for single polypeptide chains beyond a few thousand residues, or composite use cases (multi-chain macromolecular complexes, synthetic polymers, large nucleic-acid structures), this is frequently the only architecture in the room that finishes.

---

## 7. Numerical-equivalence checks

Both files ship a `[PASS]/[FAIL]` verification suite, run directly via:

```bash
python seq_to_coarse_structure.py
python run_msa_free_pipeline.py
```

(no arguments — passing a sequence as the first CLI argument runs the real pipeline instead of the test suite). Key things each suite checks:

- **`seq_to_coarse_structure.py`**: factorized vs. unfactorized pairwise projection, chunked vs. unchunked distogram, fused vs. unfused expected-distance path, 2-D tiled vs. row-chunked path (incl. gradient flow), row-chunked vs. unchunked SMACOF (incl. gradient flow), Landmark MDS init, sliding-window attention locality (a residue outside the window must not influence another residue's output; one inside it must), and all three auto-switch thresholds (window attention, Landmark MDS, learned-embedding fallback) — plus the lazy positional-encoding cache growing monotonically and never shrinking.
- **`run_msa_free_pipeline.py`**: end-to-end shape consistency from raw sequence through to `final_coords`/`pred_ddg`, PDB files actually written to disk, and that `predict()` correctly reports `tier2: None` when Tier 2 is disabled.

If you change either file, re-run both suites before trusting new output — most of the optimisations above are explicitly "trades memory/speed, not correctness," and the tests are what back that claim.

---

## 8. External dependencies by tier

| Dependency | Required for | Install |
|---|---|---|
| `torch` (≥ 2.0 recommended) | Everything. PyTorch ≥ 2.0 lets `F.scaled_dot_product_attention` dispatch to FlashAttention automatically where supported. | `pip install torch` |
| `fair-esm` or `transformers` | `embed_backend="esm2"` (optional — falls back to `"learned"` if neither is installed). | `pip install fair-esm` or `pip install transformers` |
| `biopython` | Nicer 1-letter→3-letter residue-name mapping in `write_ca_pdb` (optional — has a built-in fallback table). | `pip install biopython` |
| `structural_gno_fold_v3.py` | Tier 1's refinement stage (`StructuralGNOFold`). Required by `run_msa_free_pipeline.py`; not bundled here. | *(ecosystem file)* |
| `openmm` + `real_fold_one_v2.py` | Tier 2 (`RefinementEngine`). | `pip install openmm` |
| `pdbfixer` | Tier 2 side-chain reconstruction. | `pip install pdbfixer` |

`seq_to_coarse_structure.py` runs standalone with only `torch` installed (everything else gracefully degrades to a fallback or is skipped).
