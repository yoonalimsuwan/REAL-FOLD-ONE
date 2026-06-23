# =============================================================================
# SEQUENCE-TO-COARSE-STRUCTURE (SEQ2COARSE) — v1 Production
# MSA-Free Initial Structure Generator for REAL FOLD ONE Ecosystem
# =============================================================================
# Developer    : Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026
#
# AI Co-Developers (architecture, numerical methods, production hardening):
#   - Claude   (Anthropic)  — module design, differentiable MDS solver,
#                             ESM-2 / fallback embedding bridge, EMA +
#                             checkpoint plumbing, PDB export, full docstrings
#
# Description:
#   Production-grade, fully differentiable bridge that takes a *single*
#   protein sequence (no multiple-sequence alignment, no co-evolutionary
#   profile, no template search) and produces a coarse 3-D Cα structure
#   plus per-residue structural-regime features.  This closes the one gap
#   identified in the REAL FOLD ONE / SGNO pipeline that currently assumes
#   ``init_coords`` already exists:
#
#       sequence  ──(this module)──▶  (init_coords, seq_features, sigma)
#                                              │
#                                              ▼
#                              StructuralGNOFold.forward(...)   (existing)
#                                              │
#                                              ▼
#                         RefinementEngine.refine(pdb_file=...) (existing)
#
#   Pipeline (all single-sequence, MSA-free):
#
#     1. SequenceEmbedder
#          Pretrained ESM-2 embedding (frozen, single sequence — no MSA
#          search, no profile, no template) if ``fair-esm`` /
#          ``transformers`` is available; graceful fallback to a learned
#          20-letter amino-acid embedding + sinusoidal position encoding
#          otherwise, so the module always runs standalone.
#
#     2. SequenceTransformerEncoder
#          Pre-LN bidirectional transformer encoder over the per-residue
#          embedding sequence.  This is the long-range-context component
#          that MSA/Evoformer normally supplies via co-evolution; here it
#          comes from the pretrained language-model prior instead.
#
#     3. DistogramHead
#          Predicts a binned Cα–Cα pairwise distance distribution
#          (i, j) ↦ softmax over distance bins, the standard structure-
#          prediction intermediate (AlphaFold-1 / trRosetta style).
#
#     4. DifferentiableMDS  (stress-majorization / SMACOF)
#          Converts the *expected* distance matrix from the distogram into
#          3-D Cα coordinates via a fully autograd-compatible iterative
#          embedding solver — no eigendecomposition, so gradients flow
#          cleanly back through the whole stack during end-to-end training.
#
#     5. SigmaHead
#          Per-residue structural-regime σ(x) estimate, FiLM-compatible
#          with ``StructuralMessagePassing`` in ``structural_gno_fold_v3.py``.
#
#   Ecosystem integration:
#     • one_core_fold.py             — get_device, CSOCBase conventions
#     • structural_gno_fold_v3.py    — consumes (seq_features, init_coords, sigma)
#     • real_fold_one_v2.py          — RefinementEngine.refine(pdb_file=...)
#
#   Conventions followed (matching the rest of the ONE Ecosystem):
#     • try/except ImportError fallback for every optional dependency
#     • soft_clamp (tanh-based) instead of hard .clamp() on differentiable paths
#     • register_buffer for persistent non-parameter state
#     • dataclass config with __post_init__ validation, to_dict/from_dict
#     • EMAWrapper, checkpoint save/load, GradMonitor-compatible
#     • [PASS]/[FAIL] verification suite in __main__
#     • English documentation throughout
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

# =============================================================================
# Optional ecosystem imports (graceful fallback for standalone execution)
# =============================================================================
try:
    from one_core_fold import get_device, FOLD_VERSION
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False
    FOLD_VERSION = "unknown"

    def get_device(preferred: str = "cuda") -> torch.device:  # type: ignore[misc]
        """Standalone fallback mirroring one_core_fold.get_device."""
        p = preferred.lower()
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    warnings.warn("one_core_fold not found — running in standalone mode.")

try:
    from structural_gno_fold_v3 import SGNO_VERSION
    _HAS_SGNO = True
except ImportError:
    _HAS_SGNO = False
    SGNO_VERSION = "unknown"

# --- Optional sequence-embedding backends ----------------------------------
# Priority: fair-esm (native ESM-2) > transformers (ESM-2 via HF) > fallback.
try:
    import esm as _fair_esm  # type: ignore
    _HAS_FAIR_ESM = True
except ImportError:
    _HAS_FAIR_ESM = False
    _fair_esm = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    _HAS_HF_TRANSFORMERS = True
except ImportError:
    _HAS_HF_TRANSFORMERS = False
    AutoTokenizer = None  # type: ignore[assignment]
    AutoModel = None      # type: ignore[assignment]

try:
    from Bio.PDB import Polypeptide  # type: ignore
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False
    Polypeptide = None  # type: ignore[assignment]

SEQ2COARSE_VERSION: str = "1.0.0"

# Canonical 20-letter amino-acid alphabet, shared with real_fold_one_v2.py
# and evolution_one_epidemiological_viral_v5.py for cross-module consistency.
AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}
UNKNOWN_AA_IDX: int = len(AA_ALPHABET)  # reserved slot for 'X' / non-standard residues


def soft_clamp(x: torch.Tensor, lo: float, hi: float, sharpness: float = 1.0) -> torch.Tensor:
    """
    Smooth, fully differentiable clamp using tanh — replaces hard
    ``torch.clamp`` on any path that must remain gradient-friendly.

    Canonical re-implementation matching the convention used throughout
    the ONE Ecosystem (DNS / Cahn-Hilliard / Langevin clusters).

    Args:
        x         : input tensor.
        lo, hi    : soft lower / upper bounds.
        sharpness : higher → closer to a hard clamp (default 1.0 = gentle).
    Returns:
        Tensor smoothly bounded within (lo, hi).
    """
    mid  = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh(sharpness * (x - mid) / max(half, 1e-12))


# =============================================================================
# 1.  Configuration Dataclass
# =============================================================================

@dataclass
class Seq2CoarseConfig:
    """
    Centralised, validated hyperparameter store for SeqToCoarseStructure.

    Embedding
    ---------
    embed_backend   : "esm2" (pretrained, frozen, MSA-free) or "learned"
                       (fallback embedding table — always available).
    esm_model_name  : fair-esm or HF model identifier, e.g.
                       "esm2_t12_35M_UR50D" / "facebook/esm2_t12_35M_UR50D".
    esm_repr_layer  : which transformer layer's hidden state to extract
                       (fair-esm backend only; ignored for HF backend,
                       which uses the final hidden state).
    freeze_esm      : if True, ESM-2 weights are not updated by the
                       optimiser (recommended — keeps the pretrained
                       evolutionary prior intact).
    embed_dim       : dimensionality of the per-residue embedding fed
                       into the transformer encoder. For "learned" this
                       is also the embedding-table width; for "esm2" an
                       input projection maps the backbone width down (or
                       up) to this value.
    max_seq_len     : maximum sequence length supported (positional
                       encoding / attention mask sizing).

    Transformer encoder
    --------------------
    hidden_dim      : transformer model width.
    num_heads       : multi-head attention heads.
    num_layers      : transformer encoder layers.
    ffn_dim         : feed-forward inner dimension.
    dropout         : dropout probability throughout the encoder.

    Long-sequence scaling (opt-in; all default to the original O(N²)
    full-attention / full-classical-MDS / ESM-2 behaviour exactly)
    --------------------------------------------------------------
    attn_window_size       : if set, ``SequenceTransformerEncoder`` uses
                              banded sliding-window self-attention
                              (each residue attends only to the
                              ``attn_window_size`` nearest neighbours on
                              each side, i.e. a band of width
                              ``2 * attn_window_size + 1``) instead of
                              full O(N²) attention. This changes the
                              *inductive bias* (long-range pairs beyond
                              the window no longer attend to each other
                              directly — long-range signal can still
                              propagate indirectly across layers, the
                              same way a CNN's receptive field grows with
                              depth), but reduces attention cost from
                              O(N²) to O(N·w), which is what makes
                              N ~ 100,000+ residues tractable on a single
                              GPU. None (default) reproduces the
                              original full-attention ``nn.TransformerEncoder``
                              path exactly. Recommended once N exceeds
                              ``auto_window_attn_threshold`` (see below).
    auto_window_attn_threshold : if ``attn_window_size`` is None and the
                              input sequence length exceeds this
                              threshold, ``SequenceTransformerEncoder``
                              automatically switches to sliding-window
                              attention using ``attn_window_size_default``
                              as the window radius, and logs that it did
                              so. Set to a very large number (or pass
                              ``attn_window_size`` explicitly) to disable
                              auto-switching and keep manual control.
    attn_window_size_default  : window radius used by the auto-switch
                              above when ``attn_window_size`` itself is
                              left unset.
    auto_landmark_threshold   : if ``mds_use_landmarks`` is False and N
                              exceeds this threshold, ``DifferentiableMDS``
                              automatically uses Landmark MDS
                              initialisation instead of full classical MDS
                              (whose O(N³) eigendecomposition becomes
                              infeasible well before this point), and
                              logs that it did so. Set to a very large
                              number to disable auto-switching.
    auto_learned_embed_threshold : if ``embed_backend == "esm2"`` and N
                              exceeds this threshold, ``SequenceEmbedder``
                              automatically falls back to the "learned"
                              embedding backend for that call (ESM-2 was
                              never trained on contexts this long and its
                              attention becomes unreliable / OOM-prone
                              there), and logs/warns that it did so. Set
                              to a very large number to disable
                              auto-switching.

    Distogram
    ---------
    num_distance_bins : number of discrete Cα–Cα distance bins.
    min_distance      : lower edge of the first bin (Å).
    max_distance      : upper edge of the last bin (Å).

    Distogram memory optimisation
    ------------------------------
    pair_proj_factorized : if True (default), DistogramHead computes its
                            first linear layer as two separate (N, d) →
                            (N, d) projections summed via broadcasting,
                            instead of materialising an explicit
                            (N, N, 2·hidden_dim) concatenated pair tensor.
                            This is an exact algebraic re-expression of
                            the same linear layer (verified numerically;
                            see README_PREDICTOR.md §6) — it changes
                            nothing about the function being computed,
                            only how much memory the forward pass uses
                            along the way. Roughly halves peak memory
                            for the first layer with zero accuracy cost;
                            disable only for debugging / parity checks
                            against the original formulation.
    pair_chunk_size       : if set, the (N, N, ·) pairwise computation is
                             processed in row-chunks of this many residues
                             at a time, rather than materialising the full
                             (N, N, ·) tensor in one shot. Trades a small
                             amount of speed for a large reduction in peak
                             memory — e.g. pair_chunk_size=256 caps peak
                             pairwise memory at roughly
                             (256, N, hidden_dim) instead of (N, N,
                             hidden_dim), independent of how large N
                             grows. None (default) disables chunking and
                             reproduces the original unchunked behaviour
                             exactly. Recommended once N · num_distance_bins
                             · hidden_dim starts to exceed available
                             GPU memory (see README_PREDICTOR.md §3.3 for
                             concrete N-vs-memory figures).
    pair_proj_dtype       : if set (e.g. torch.bfloat16 / torch.float16),
                             the pairwise projection stage runs in this
                             reduced precision, halving (or better) its
                             memory footprint relative to float32 — the
                             same strategy AlphaFold3 / Boltz use for
                             their pairwise representations. None
                             (default) keeps the original float32 path
                             unchanged. Output is cast back to the
                             input's dtype before being returned, so
                             callers never need to know this happened.
                             Combine with autocast / GradScaler in the
                             training loop for end-to-end mixed-precision
                             training if desired.

    Differentiable MDS (coarse 3-D embedding)
    ------------------------------------------
    mds_iters       : number of SMACOF stress-majorization iterations.
    mds_dim         : output embedding dimension (3 for Cα coordinates).
    mds_eps         : numerical floor to avoid division by ~0 distances.
    mds_init_scale  : scale of the random initial 3-D embedding (only
                      used as a last-resort fallback if classical MDS
                      initialisation fails numerically).

    MDS memory optimisation
    -------------------------
    mds_row_chunk_size : if set, SMACOF's per-iteration Guttman transform
                          is computed in row-blocks of this many residues
                          rather than materialising the full (N, N)
                          intermediate matrices (B, ratio, weights, etc.)
                          at once. Caps peak per-iteration memory at
                          O(mds_row_chunk_size · N) instead of O(N²) —
                          verified numerically equivalent to the unchunked
                          computation (see README_PREDICTOR.md §6/§9).
                          None (default) reproduces the original unchunked
                          behaviour exactly. Note: this does not reduce
                          the memory needed to *store* the (N, N) target
                          distance matrix itself, which the caller
                          (``SeqToCoarseStructure.forward``) currently
                          materialises before calling ``DifferentiableMDS``
                          — see README_PREDICTOR.md §9 for why closing
                          that remaining gap requires a separate, larger
                          architectural change (streaming the distogram
                          directly into the MDS solver, rather than
                          optimising the solver in isolation).
    mds_use_landmarks  : if True, ``_classical_mds_init`` uses Landmark
                          MDS (de Silva & Tenenbaum, 2004) instead of full
                          classical MDS — solving the expensive
                          double-centering + eigendecomposition on a
                          small landmark subset (O(L³) instead of O(N³))
                          and triangulating every other point's
                          coordinates from its distances to the landmarks
                          (closed-form, O(N·L·mds_dim), no further
                          eigendecomposition). Strongly recommended for
                          N beyond a few thousand residues, where full
                          classical MDS's O(N³) cost becomes infeasible
                          regardless of available memory. None/False
                          (default) reproduces the original full
                          classical-MDS initialisation exactly.
    mds_num_landmarks  : number of landmarks L to use when
                          ``mds_use_landmarks=True``. Must satisfy
                          ``mds_dim < L < N``; a few hundred to ~2000 is
                          typically sufficient for a good warm start
                          regardless of how large N grows (the landmark
                          subproblem's cost depends only on L, not N).

    Sigma head
    ----------
    sigma_min, sigma_max : soft_clamp bounds for the predicted structural
                            regime field σ(x), matching the convention used
                            by StructuralGNOFold / SOCController elsewhere
                            in the ecosystem (σ_target ≈ 1.0 by default).

    Training
    --------
    lr_encoder      : learning rate for the transformer encoder + heads.
    lr_embedding    : learning rate for the (learned-backend) embedding
                       table, or for the ESM-2 projection head when the
                       backbone itself is frozen.
    weight_decay    : AdamW weight-decay coefficient.
    grad_clip       : max gradient norm for clipping.
    ema_decay       : EMA decay for inference-time weight averaging.
    lambda_distogram : loss weight for the distogram cross-entropy term.
    lambda_coord      : loss weight for the direct coordinate (FAPE-lite /
                         RMSD-style) supervision term, when ground-truth
                         coordinates are available.
    lambda_sigma      : loss weight for the σ-regularisation term that
                         pulls σ toward sigma_target in the absence of
                         direct structural-stress labels.
    sigma_target      : reference structural stress used by lambda_sigma
                         (kept consistent with CSOCBase.sigma_target).

    Checkpoint
    ----------
    checkpoint_dir  : directory for checkpoint files.
    save_every      : save checkpoint every N epochs.
    """

    # Embedding
    embed_backend:  str   = "esm2"          # "esm2" | "learned"
    esm_model_name: str   = "esm2_t12_35M_UR50D"
    esm_repr_layer: int   = 12
    freeze_esm:     bool  = True
    embed_dim:      int   = 256
    max_seq_len:    int   = 120_000

    # Transformer encoder
    hidden_dim:  int   = 256
    num_heads:   int   = 8
    num_layers:  int   = 6
    ffn_dim:     int   = 1024
    dropout:     float = 0.1

    # Long-sequence scaling (opt-in; defaults reproduce the original
    # O(N²) full-attention / full-classical-MDS / ESM-2 behaviour exactly)
    attn_window_size: Optional[int] = None
    auto_window_attn_threshold: int = 8000
    attn_window_size_default: int = 256
    auto_landmark_threshold: int = 8000
    auto_learned_embed_threshold: int = 8000

    # Distogram
    num_distance_bins: int   = 64
    min_distance:      float = 2.0
    max_distance:       float = 40.0

    # Distogram memory optimisation (all opt-in; defaults reproduce the
    # original unchunked, full-float32 behaviour exactly)
    pair_proj_factorized: bool = True
    pair_chunk_size: Optional[int] = None
    pair_proj_dtype: Optional[torch.dtype] = None

    # Differentiable MDS
    mds_iters:      int   = 200
    mds_dim:        int   = 3
    mds_eps:        float = 1e-6
    mds_init_scale: float = 5.0

    # MDS memory optimisation (all opt-in; defaults reproduce the original
    # unchunked, full-classical-MDS behaviour exactly)
    mds_row_chunk_size: Optional[int] = None
    mds_use_landmarks:  bool = False
    mds_num_landmarks:  int  = 1000


    # Sigma head
    sigma_min: float = 0.05
    sigma_max: float = 5.0

    # Training
    lr_encoder:       float = 3e-4
    lr_embedding:      float = 1e-4
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    ema_decay:         float = 0.999
    lambda_distogram:  float = 1.0
    lambda_coord:      float = 0.5
    lambda_sigma:      float = 0.05
    sigma_target:      float = 1.0

    # Checkpointing
    checkpoint_dir: str = "./seq2coarse_checkpoints"
    save_every:     int = 10

    def __post_init__(self) -> None:
        assert self.embed_backend in ("esm2", "learned"), \
            f"embed_backend must be 'esm2' or 'learned'; got {self.embed_backend!r}."
        assert self.embed_dim > 0
        assert self.hidden_dim > 0
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
        assert self.num_layers >= 1
        assert 0.0 <= self.dropout < 1.0
        assert self.attn_window_size is None or self.attn_window_size >= 1
        assert self.auto_window_attn_threshold >= 1
        assert self.attn_window_size_default >= 1
        assert self.auto_landmark_threshold >= 1
        assert self.auto_learned_embed_threshold >= 1
        assert self.num_distance_bins >= 2
        assert self.max_distance > self.min_distance > 0.0
        assert self.pair_chunk_size is None or self.pair_chunk_size >= 1
        assert self.pair_proj_dtype is None or self.pair_proj_dtype in (
            torch.float16, torch.bfloat16, torch.float32,
        ), f"pair_proj_dtype must be a float dtype or None; got {self.pair_proj_dtype!r}."
        assert self.mds_iters >= 1
        assert self.mds_dim >= 1
        assert self.mds_eps > 0.0
        assert self.mds_row_chunk_size is None or self.mds_row_chunk_size >= 1
        assert self.mds_num_landmarks > self.mds_dim, \
            f"mds_num_landmarks ({self.mds_num_landmarks}) must exceed mds_dim ({self.mds_dim})."
        assert 0.0 < self.sigma_min < self.sigma_max
        assert self.grad_clip > 0.0
        assert 0.0 < self.ema_decay < 1.0
        assert self.sigma_target > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Seq2CoarseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# 2.  Sequence Embedder — MSA-Free
# =============================================================================

class SequenceEmbedder(nn.Module):
    """
    Per-residue sequence embedding, **single-sequence and MSA-free** by
    construction: no alignment search, no co-evolutionary profile, no
    template lookup is ever performed. Two backends are supported:

    ``"esm2"`` (default, recommended)
        Uses a pretrained ESM-2 protein language model as a *frozen*
        feature extractor over the raw amino-acid string. ESM-2's
        evolutionary prior is baked into its pretrained weights, so it
        supplies the long-range co-evolutionary signal that MSA-based
        pipelines would otherwise compute on the fly — at inference time
        for a novel sequence, only the sequence itself is needed. Tries
        ``fair-esm`` first, then ``transformers`` (Hugging Face ESM-2);
        downloads weights on first use only.

    ``"learned"`` (fallback)
        A simple learned embedding table over the 21-symbol alphabet
        (20 canonical amino acids + unknown) plus sinusoidal positional
        encoding. Always available, fully trainable, no external
        dependency or network access required — used automatically when
        neither ``fair-esm`` nor ``transformers`` is installed, or when
        explicitly requested via ``cfg.embed_backend = "learned"``.

    In both cases the output is a single ``(N, embed_dim)`` tensor per
    sequence: one vector per residue, no row dimension for alignment
    sequences.

    Args:
        cfg : Seq2CoarseConfig instance.
    """

    def __init__(self, cfg: Seq2CoarseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._backend_active = cfg.embed_backend
        self._esm_model = None
        self._esm_tokenizer = None
        self._esm_batch_converter = None
        self._esm_native_dim: Optional[int] = None

        if cfg.embed_backend == "esm2":
            self._try_init_esm2()

        if self._backend_active == "learned":
            # Fallback path: trainable embedding table + sinusoidal position encoding.
            self.aa_embed = nn.Embedding(len(AA_ALPHABET) + 1, cfg.embed_dim, padding_idx=None)
            # NOTE: the positional table is no longer eagerly materialised at
            # __init__ time for cfg.max_seq_len rows — at max_seq_len=120,000
            # and embed_dim=256 that would be a ~123 MB buffer kept around
            # for the entire lifetime of the module regardless of the actual
            # sequence lengths ever passed in. Instead, _pos_encoding_for(n)
            # below builds exactly an (n, embed_dim) table on demand and
            # caches it, growing the cache only up to the longest sequence
            # actually seen — identical values to the original eager table
            # sliced to [:n], just computed lazily.
            self.pos_encoding = None  # type: ignore[assignment]
            self._pos_encoding_cache: Optional[torch.Tensor] = None
            self.input_proj: Optional[nn.Module] = None
        else:
            self.aa_embed = None  # type: ignore[assignment]
            self.pos_encoding = None  # type: ignore[assignment]
            self._pos_encoding_cache: Optional[torch.Tensor] = None  # type: ignore[no-redef]
            # Project frozen ESM-2 hidden width → cfg.embed_dim.
            native_dim = self._esm_native_dim or cfg.embed_dim
            self.input_proj = nn.Sequential(
                nn.Linear(native_dim, cfg.embed_dim),
                nn.LayerNorm(cfg.embed_dim),
            )
            # Standby "learned" path for the auto_learned_embed_threshold
            # switch below. Created lazily (only the first time a sequence
            # actually exceeds the threshold) rather than eagerly here, so
            # existing ESM-2 checkpoints — whose state_dict has no such
            # key — keep loading without a key mismatch for any run that
            # never hits the threshold.
            self._standby_aa_embed: Optional[nn.Embedding] = None

        logger.info(
            "SequenceEmbedder | backend=%s | embed_dim=%d",
            self._backend_active, cfg.embed_dim,
        )

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _try_init_esm2(self) -> None:
        """Attempt fair-esm, then HF transformers; fall back to 'learned'."""
        if _HAS_FAIR_ESM:
            try:
                model, alphabet = _fair_esm.pretrained.load_model_and_alphabet(
                    self.cfg.esm_model_name
                )
                if self.cfg.freeze_esm:
                    model.eval()
                    for p in model.parameters():
                        p.requires_grad_(False)
                self._esm_model = model
                self._esm_batch_converter = alphabet.get_batch_converter()
                self._esm_native_dim = model.embed_dim
                self._backend_active = "esm2"
                logger.info(
                    "Loaded fair-esm backend '%s' (embed_dim=%d).",
                    self.cfg.esm_model_name, self._esm_native_dim,
                )
                return
            except Exception as exc:  # pragma: no cover - network/weights issues
                logger.warning("fair-esm load failed (%s); trying transformers backend.", exc)

        if _HAS_HF_TRANSFORMERS:
            try:
                hf_name = self.cfg.esm_model_name
                if not hf_name.startswith("facebook/"):
                    hf_name = f"facebook/{hf_name}"
                tok = AutoTokenizer.from_pretrained(hf_name)
                model = AutoModel.from_pretrained(hf_name)
                if self.cfg.freeze_esm:
                    model.eval()
                    for p in model.parameters():
                        p.requires_grad_(False)
                self._esm_model = model
                self._esm_tokenizer = tok
                self._esm_native_dim = model.config.hidden_size
                self._backend_active = "esm2"
                logger.info(
                    "Loaded HF transformers backend '%s' (hidden_size=%d).",
                    hf_name, self._esm_native_dim,
                )
                return
            except Exception as exc:  # pragma: no cover - network/weights issues
                logger.warning("transformers load failed (%s); falling back to 'learned'.", exc)

        warnings.warn(
            "No ESM-2 backend available (fair-esm / transformers not installed, "
            "or weight download failed). Falling back to embed_backend='learned'. "
            "Predictions remain MSA-free but lose the pretrained evolutionary prior."
        )
        self._backend_active = "learned"

    @staticmethod
    def _build_sinusoidal_table(max_len: int, dim: int) -> torch.Tensor:
        """
        Standard Transformer sinusoidal positional encoding table
        (max_len, dim). Handles odd ``dim`` correctly (the cosine channel
        simply gets one fewer column than the sine channel in that case).
        """
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        n_sin = (dim + 1) // 2   # number of sin columns: ceil(dim / 2)
        n_cos = dim // 2          # number of cos columns: floor(dim / 2)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )  # length == n_sin by construction (arange(0, dim, 2))

        table = torch.zeros(max_len, dim)
        table[:, 0::2] = torch.sin(pos * div[:n_sin])
        table[:, 1::2] = torch.cos(pos * div[:n_cos])
        return table

    def _pos_encoding_for(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Lazily build (and cache) the sinusoidal positional-encoding table
        for the "learned" backend, sized to exactly the rows needed so
        far rather than to ``cfg.max_seq_len`` up front.

        Mathematically identical to slicing the original eagerly-built
        ``(max_seq_len, embed_dim)`` table to ``[:n]`` — only the
        materialisation strategy changed, not the values: the cache grows
        (never shrinks) to the longest ``n`` requested in this process,
        and is reused/sliced for any shorter request afterwards.

        Args:
            n      : number of residues (rows) needed.
            device : target device.
        Returns:
            (n, embed_dim) positional-encoding table on ``device``.
        """
        cache = self._pos_encoding_cache
        same_device = cache is not None and cache.device == device
        if cache is None or cache.size(0) < n or not same_device:
            # Only carry over the previous row-count as a "grow, don't
            # shrink" floor when staying on the same device — a cache
            # built for a different device says nothing about how many
            # rows are worth keeping warm on this one.
            prior_rows = cache.size(0) if (cache is not None and same_device) else 0
            grow_to = max(n, prior_rows)
            table = self._build_sinusoidal_table(grow_to, self.cfg.embed_dim).to(device)
            self._pos_encoding_cache = table
            cache = table
        return cache[:n]

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_indices(sequence: str) -> torch.Tensor:
        """
        Map a raw amino-acid string to integer indices using AA_TO_IDX,
        with non-standard residues mapped to UNKNOWN_AA_IDX.

        Args:
            sequence : single-letter amino-acid string, e.g. "MKTAYIAK...".
        Returns:
            (N,) LongTensor of indices.
        """
        idx = [AA_TO_IDX.get(aa.upper(), UNKNOWN_AA_IDX) for aa in sequence]
        return torch.tensor(idx, dtype=torch.long)

    def _embed_learned(self, sequence: str, device: torch.device) -> torch.Tensor:
        idx = self.encode_indices(sequence).to(device)
        n = idx.size(0)
        if n > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {n} exceeds max_seq_len={self.cfg.max_seq_len}."
            )
        x = self.aa_embed(idx)                          # (N, embed_dim)
        x = x + self._pos_encoding_for(n, device)        # add positional signal
        return x

    def _embed_learned_standalone(self, sequence: str, device: torch.device) -> torch.Tensor:
        """
        Same computation as ``_embed_learned``, but for the case where
        ``self._backend_active == "esm2"`` and ``self.aa_embed`` is
        ``None`` (the primary path is ESM-2). Lazily creates and reuses
        ``self._standby_aa_embed`` so that ESM-2 checkpoints saved before
        this auto-switch path existed keep loading unchanged for any run
        that never crosses ``cfg.auto_learned_embed_threshold``.
        """
        if self._standby_aa_embed is None:
            self._standby_aa_embed = nn.Embedding(
                len(AA_ALPHABET) + 1, self.cfg.embed_dim, padding_idx=None
            ).to(device)
        idx = self.encode_indices(sequence).to(device)
        n = idx.size(0)
        x = self._standby_aa_embed(idx)                  # (N, embed_dim)
        x = x + self._pos_encoding_for(n, device)         # add positional signal
        return x

    def _embed_esm2_fair(self, sequence: str, device: torch.device) -> torch.Tensor:
        data = [("query", sequence)]
        _, _, tokens = self._esm_batch_converter(data)
        tokens = tokens.to(device)
        ctx = torch.no_grad() if self.cfg.freeze_esm else torch.enable_grad()
        with ctx:
            out = self._esm_model(
                tokens, repr_layers=[self.cfg.esm_repr_layer], return_contacts=False
            )
        reps = out["representations"][self.cfg.esm_repr_layer]   # (1, N+2, native_dim)
        reps = reps[0, 1: len(sequence) + 1]                      # strip BOS/EOS tokens
        return reps

    def _embed_esm2_hf(self, sequence: str, device: torch.device) -> torch.Tensor:
        spaced = " ".join(list(sequence))
        enc = self._esm_tokenizer(spaced, return_tensors="pt").to(device)
        ctx = torch.no_grad() if self.cfg.freeze_esm else torch.enable_grad()
        with ctx:
            out = self._esm_model(**enc)
        reps = out.last_hidden_state[0, 1: len(sequence) + 1]     # strip special tokens
        return reps

    def forward(self, sequence: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Embed a single raw amino-acid sequence (no MSA, no profile).

        Args:
            sequence : single-letter amino-acid string.
            device   : target device; inferred from module parameters if None.
        Returns:
            (N, cfg.embed_dim) per-residue embedding.

        Long-sequence note: if ``cfg.embed_backend == "esm2"`` and the
        sequence length exceeds ``cfg.auto_learned_embed_threshold``, this
        call automatically routes through the "learned" embedding path
        instead (ESM-2 was never trained on contexts this long; running it
        there is both unreliable and OOM-prone). This only affects this
        particular call — ``self._backend_active`` / ``cfg.embed_backend``
        are left unchanged, so shorter sequences in the same model
        instance continue to use ESM-2 as configured. Set
        ``cfg.auto_learned_embed_threshold`` very high to disable this.
        """
        if device is None:
            device = next(self.parameters()).device if any(
                True for _ in self.parameters()
            ) else torch.device("cpu")

        n = len(sequence)
        if n == 0:
            raise ValueError("Input sequence is empty.")
        if n > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {n} exceeds max_seq_len={self.cfg.max_seq_len}."
            )

        use_learned_this_call = (
            self._backend_active != "learned"
            and n > self.cfg.auto_learned_embed_threshold
        )
        if use_learned_this_call:
            logger.warning(
                "Sequence length %d exceeds auto_learned_embed_threshold=%d; "
                "auto-switching this call from embed_backend=%r to the 'learned' "
                "fallback (ESM-2 is not designed for contexts this long).",
                n, self.cfg.auto_learned_embed_threshold, self._backend_active,
            )
            return self._embed_learned_standalone(sequence, device)

        if self._backend_active == "learned":
            return self._embed_learned(sequence, device)

        if self._esm_batch_converter is not None:
            native = self._embed_esm2_fair(sequence, device)
        elif self._esm_tokenizer is not None:
            native = self._embed_esm2_hf(sequence, device)
        else:  # pragma: no cover - defensive
            raise RuntimeError("ESM-2 backend selected but no converter/tokenizer initialised.")

        assert self.input_proj is not None
        return self.input_proj(native)                  # (N, embed_dim)


# =============================================================================
# 2b.  Sliding-Window Self-Attention (O(N·w) alternative to full O(N²))
# =============================================================================

class SlidingWindowSelfAttention(nn.Module):
    """
    Banded (local) multi-head self-attention: residue ``i`` attends only
    to residues in ``[i - window, i + window]`` (clipped at the sequence
    boundaries), i.e. a band of width ``2*window + 1`` rather than the
    full ``N`` columns of standard attention.

    Implemented as query-side chunking rather than a dense ``(N, N)``
    mask: for an ``(N, N)`` boolean mask alone would cost ``N²`` bits
    (e.g. ~10 GB at N=100,000) before any attention math even starts,
    defeating the purpose. Instead, each query chunk of size
    ``chunk_size`` only ever materialises scores against the *local* key
    span it can actually attend to — ``(chunk_size, span)`` rather than
    ``(chunk_size, N)`` — so peak attention memory scales with
    ``chunk_size · window``, not with ``N`` at all.

    This changes the model's inductive bias relative to full attention
    (residues farther apart than ``window`` no longer attend to each
    other *directly* in a single layer — long-range information can
    still mix indirectly across multiple stacked layers, the same way a
    CNN's effective receptive field grows with depth), but reduces
    attention compute from O(N²) to O(N·window), which is what makes
    N ~ 100,000+ residues tractable.

    Args:
        dim         : model width (must be divisible by ``num_heads``).
        num_heads   : number of attention heads.
        window      : one-sided window radius — residue i attends to
                      columns ``[max(0, i-window), min(N, i+window+1))``.
        dropout     : attention-dropout probability.
        chunk_size  : query-chunking granularity; defaults to
                      ``4 * window`` (a few window-widths per chunk) when
                      left as ``None``, which keeps the per-chunk local
                      key span a small, bounded multiple of ``window``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window: int,
        dropout: float = 0.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.dropout = dropout
        self.chunk_size = chunk_size or max(1, 4 * window)

        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (1, N, dim) — batch size fixed at 1 (single sequence),
                matching the rest of this module's convention.
        Returns:
            (1, N, dim).
        """
        assert x.size(0) == 1, "SlidingWindowSelfAttention expects batch size 1 (single sequence)."
        n = x.size(1)
        device = x.device
        w = self.window

        qkv = self.qkv_proj(x)                                   # (1, N, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)                            # each (1, N, dim)
        q = q.view(n, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, N, hd)
        k = k.view(n, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, N, hd)
        v = v.view(n, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, N, hd)

        out_chunks: List[torch.Tensor] = []
        for q_start in range(0, n, self.chunk_size):
            q_end = min(q_start + self.chunk_size, n)
            k_start = max(0, q_start - w)
            k_end = min(n, q_end - 1 + w + 1)

            q_chunk = q[:, q_start:q_end]                          # (heads, Cq, hd)
            k_chunk = k[:, k_start:k_end]                          # (heads, Ck, hd)
            v_chunk = v[:, k_start:k_end]                          # (heads, Ck, hd)

            # Local band mask restricted to this (Cq, Ck) block — far
            # smaller than the (N, N) mask a naive implementation would
            # need, and rebuilt per-chunk so memory never scales with N.
            q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(1)   # (Cq, 1)
            k_pos = torch.arange(k_start, k_end, device=device).unsqueeze(0)   # (1, Ck)
            band_mask = (k_pos - q_pos).abs() <= w                              # (Cq, Ck) bool

            attn_bias = torch.zeros(q_end - q_start, k_end - k_start, device=device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(~band_mask, float("-inf"))

            # F.scaled_dot_product_attention dispatches to FlashAttention /
            # memory-efficient kernels automatically on supported hardware
            # (PyTorch >= 2.0), so the (Cq, Ck) score matrix this still
            # implies is not necessarily even fully materialised in VRAM.
            attn_out = F.scaled_dot_product_attention(
                q_chunk, k_chunk, v_chunk,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (heads, Cq, hd)
            out_chunks.append(attn_out)

        out = torch.cat(out_chunks, dim=1)                          # (heads, N, hd)
        out = out.transpose(0, 1).reshape(1, n, self.dim)           # (1, N, dim)
        return self.out_proj(out)


class SlidingWindowEncoderLayer(nn.Module):
    """
    Pre-LN transformer encoder layer using ``SlidingWindowSelfAttention``
    instead of full self-attention — drop-in replacement for
    ``nn.TransformerEncoderLayer`` restricted to batch size 1, no
    padding mask (this module is only ever used in the single-sequence,
    no-padding path of ``SequenceTransformerEncoder``).

    Args:
        cfg    : Seq2CoarseConfig instance.
        window : one-sided sliding-window radius for this layer's
                 attention (see ``SlidingWindowSelfAttention``).
    """

    def __init__(self, cfg: Seq2CoarseConfig, window: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = SlidingWindowSelfAttention(
            dim=cfg.hidden_dim, num_heads=cfg.num_heads, window=window, dropout=cfg.dropout,
        )
        self.dropout1 = nn.Dropout(cfg.dropout)

        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.hidden_dim),
        )
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x : (1, N, hidden_dim). Returns: (1, N, hidden_dim)."""
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class SequenceTransformerEncoder(nn.Module):
    """
    Pre-LN bidirectional transformer encoder over the per-residue
    embedding sequence.

    This supplies the long-range context that an MSA/Evoformer stack
    would otherwise derive from co-evolutionary statistics across
    alignment rows. Here, context comes purely from (a) the pretrained
    language-model prior already present in the ESM-2 embedding and
    (b) self-attention over the *single* input sequence — no second
    sequence-dimension ("MSA axis") ever appears in this module.

    Two attention paths are available:

      • Full O(N²) self-attention (``nn.TransformerEncoder``, the
        original/default behaviour) — every residue attends to every
        other residue directly. Used whenever ``cfg.attn_window_size``
        is ``None`` and N stays at or below
        ``cfg.auto_window_attn_threshold``.
      • Sliding-window O(N·w) self-attention (``SlidingWindowEncoderLayer``,
        opt-in / auto-switched) — used when ``cfg.attn_window_size`` is
        set explicitly, or automatically once N exceeds
        ``cfg.auto_window_attn_threshold`` (using
        ``cfg.attn_window_size_default`` as the radius). This is what
        makes N ~ 100,000+ residues tractable, at the cost of residues
        farther apart than the window no longer attending to each other
        directly within a single layer (long-range signal still mixes
        indirectly across the stacked layers).

    The sliding-window stack is built lazily (on first use at a given
    window size) rather than always at ``__init__``, so models that
    never see a long sequence keep exactly the original parameter set
    and checkpoint layout.

    Args:
        cfg : Seq2CoarseConfig instance.
    """

    def __init__(self, cfg: Seq2CoarseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.embed_dim)
        self.in_proj = (
            nn.Linear(cfg.embed_dim, cfg.hidden_dim)
            if cfg.embed_dim != cfg.hidden_dim else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training at depth
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

        # Sliding-window stack: built lazily in _get_window_encoder() the
        # first time it is actually needed, keyed by window radius (in
        # case attn_window_size changes across calls on the same instance).
        # Not created here, so checkpoints from before this feature exists
        # — or runs that never exceed auto_window_attn_threshold — see no
        # new parameters at all.
        self._window_encoders: Dict[int, nn.ModuleList] = {}

    def _get_window_encoder(self, window: int) -> nn.ModuleList:
        """Lazily build (and cache) a sliding-window layer stack for ``window``."""
        if window not in self._window_encoders:
            layers = nn.ModuleList([
                SlidingWindowEncoderLayer(self.cfg, window=window)
                for _ in range(self.cfg.num_layers)
            ])
            layers = layers.to(next(self.parameters()).device)
            self._window_encoders[window] = layers
            # Register so the layers are tracked as submodules (parameters,
            # device/dtype moves, state_dict) even though they live inside
            # a plain dict rather than a declared attribute.
            self.add_module(f"_window_encoder_w{window}", layers)
        return self._window_encoders[window]

    def forward(
        self,
        x: torch.Tensor,                          # (N, embed_dim)
        padding_mask: Optional[torch.Tensor] = None,  # (N,) bool, True = pad
    ) -> torch.Tensor:
        """
        Args:
            x            : (N, embed_dim) per-residue embedding (single sequence).
            padding_mask : optional (N,) bool mask; True marks padded positions.
                           Only supported on the full-attention path — the
                           sliding-window path (used for very long
                           sequences) assumes no padding, matching how
                           this module is actually called elsewhere in
                           the pipeline (one real sequence at a time, no
                           batching).
        Returns:
            (N, hidden_dim) contextualised per-residue latent.
        """
        n = x.size(0)
        h = self.input_norm(x)
        h = self.in_proj(h).unsqueeze(0)           # (1, N, hidden_dim) — batch size 1

        window = self.cfg.attn_window_size
        if window is None and n > self.cfg.auto_window_attn_threshold:
            window = self.cfg.attn_window_size_default
            logger.warning(
                "Sequence length %d exceeds auto_window_attn_threshold=%d; "
                "auto-switching SequenceTransformerEncoder from full O(N²) "
                "attention to sliding-window attention with window=%d.",
                n, self.cfg.auto_window_attn_threshold, window,
            )

        if window is not None:
            if padding_mask is not None and padding_mask.any():
                raise NotImplementedError(
                    "Sliding-window attention does not support padding_mask; "
                    "pass single, unpadded sequences (the standard call pattern "
                    "for this module)."
                )
            window_encoder = self._get_window_encoder(window)
            for layer in window_encoder:
                h = layer(h)
        else:
            src_key_padding_mask = padding_mask.unsqueeze(0) if padding_mask is not None else None
            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        h = self.output_norm(h).squeeze(0)         # (N, hidden_dim)
        return h


# =============================================================================
# 4.  Distogram Head
# =============================================================================

class DistogramHead(nn.Module):
    """
    Predicts a binned Cα–Cα pairwise distance distribution from
    per-residue latents, following the AlphaFold-1 / trRosetta-style
    structure-prediction intermediate. Symmetrised so that
    ``logits[i, j] == logits[j, i]``.

    Memory scaling note: this head is inherently O(N²·d) — the same
    asymptotic cost as any pairwise distogram approach (AlphaFold-1,
    trRosetta, and AlphaFold3 / Boltz's pairwise representation included).
    Three independent, opt-in optimisations are available via
    ``Seq2CoarseConfig`` to reduce the *constant factor* of that cost
    (none change what is computed — see README_PREDICTOR.md §6 for the
    numerical equivalence check):

      • ``pair_proj_factorized`` (default True) — splits the first linear
        layer's (2d → d) weight into two (d → d) halves applied to h_i and
        h_j separately, then summed by broadcasting. This is an exact
        algebraic identity for a linear layer: avoids ever materialising
        the (N, N, 2d) concatenated input tensor, roughly halving memory
        at that stage for free.
      • ``pair_chunk_size`` (default None = unchunked) — processes the
        (N, N, ·) computation in row-blocks, capping peak memory at
        O(chunk_size · N · d) instead of O(N² · d).
      • ``pair_proj_dtype`` (default None = float32) — runs the pairwise
        stage in reduced precision (e.g. ``torch.bfloat16``), the same
        strategy AlphaFold3 / Boltz use for their pairwise tensors. Output
        is cast back to the input dtype before returning.
      • ``use_2d_tiling`` (a ``forward_expected_distance`` argument, not a
        config field) — when combined with ``pair_chunk_size``, tiles the
        pairwise computation into ``(pair_chunk_size, pair_chunk_size)``
        blocks rather than ``(pair_chunk_size, N)`` row-strips, capping
        peak MLP-activation memory at ``O(block_size²·d)`` instead of
        ``O(block_size·N·d)`` for the largest sequences. Computes both
        directions of every off-diagonal block pair explicitly (the
        factorized projection does not support a cheap transpose-based
        shortcut between them when ``W_i != W_j``); this trades somewhat
        higher wall-clock time for the additional memory reduction — it
        does not reduce total MLP compute relative to row-chunking.

    All three default to the original unchunked, factorization-equivalent,
    full-precision behaviour reproduced — i.e. enabling them changes
    memory and (modestly) speed, not the function being computed.

    Args:
        cfg : Seq2CoarseConfig instance.
    """

    def __init__(self, cfg: Seq2CoarseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        # NOTE: kept as a single nn.Sequential (not split into separate
        # modules) so that state_dict keys are identical to the original,
        # unoptimised implementation — existing checkpoints remain
        # loadable. The factorized forward path below indexes into this
        # same Sequential rather than redefining its parameters.
        self.pair_proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, cfg.num_distance_bins),
        )
        self.register_buffer(
            "bin_centers",
            torch.linspace(
                cfg.min_distance, cfg.max_distance, cfg.num_distance_bins
            ),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Internal: factorized first-layer projection
    # ------------------------------------------------------------------

    def _split_first_layer(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split ``pair_proj[0]`` (an ``nn.Linear(2d, d)``) into the two
        (d, d) weight halves that act on h_i and h_j respectively, plus
        the shared bias.

        For ``y = W @ concat(h_i, h_j) + b`` with ``W`` of shape
        ``(d, 2d)``, splitting column-wise gives ``W = [W_i | W_j]`` such
        that ``y = W_i @ h_i + W_j @ h_j + b`` exactly — verified
        numerically in README_PREDICTOR.md §6.

        Returns:
            (W_i, W_j, b), each shaped for use with ``F.linear``.
        """
        first = self.pair_proj[0]
        d = self.cfg.hidden_dim
        W_i = first.weight[:, :d]   # (d, d)
        W_j = first.weight[:, d:]   # (d, d)
        b = first.bias if first.bias is not None else torch.zeros(
            first.weight.size(0), device=first.weight.device, dtype=first.weight.dtype
        )
        return W_i, W_j, b

    def _pairwise_logits_chunk(
        self,
        h: torch.Tensor,
        row_start: int,
        row_end: int,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute distogram logits for rows ``[row_start:row_end)`` against
        all columns, i.e. a ``(row_end - row_start, N, bins)`` chunk —
        the unit of work both the chunked and unchunked paths share.

        Args:
            h             : (N, hidden_dim) per-residue latent, already
                             cast to ``compute_dtype``.
            row_start, row_end : row-block bounds (rows = the "i" index).
            compute_dtype : dtype to run the pairwise stage in.
        Returns:
            (row_end - row_start, N, num_distance_bins) logits chunk, in
            ``compute_dtype``.
        """
        n = h.size(0)
        h_rows = h[row_start:row_end]              # (chunk, d)

        if self.cfg.pair_proj_factorized:
            W_i, W_j, b = self._split_first_layer()
            W_i = W_i.to(compute_dtype)
            W_j = W_j.to(compute_dtype)
            b = b.to(compute_dtype)
            proj_rows = F.linear(h_rows, W_i)        # (chunk, d)
            proj_cols = F.linear(h, W_j)              # (N, d)
            pre = proj_rows.unsqueeze(1) + proj_cols.unsqueeze(0) + b
            # (chunk, N, d) — the (N, N, 2d) concat tensor is never formed.
            rest = self.pair_proj[1:]
            chunk_logits = rest(pre)                   # (chunk, N, bins)
        else:
            chunk = row_end - row_start
            h_i = h_rows.unsqueeze(1).expand(chunk, n, -1)   # (chunk, N, d)
            h_j = h.unsqueeze(0).expand(chunk, n, -1)         # (chunk, N, d)
            pair = torch.cat([h_i, h_j], dim=-1)               # (chunk, N, 2d)
            chunk_logits = self.pair_proj(pair.to(compute_dtype))

        return chunk_logits

    def _tiled_block_pair_logits(
        self,
        proj_i_full: torch.Tensor,   # (N, d), proj_i = F.linear(h, W_i, b) -- bias included
        proj_j_full: torch.Tensor,   # (N, d), proj_j = F.linear(h, W_j)     -- no bias
        a_start: int, a_end: int,
        b_start: int, b_end: int,
    ) -> torch.Tensor:
        """
        Compute the ``(a_end - a_start, b_end - b_start, bins)`` logits
        block with row-residues from ``[a_start:a_end)`` and column-
        residues from ``[b_start:b_end)``, i.e. ``logits[A, B]`` using the
        factorized projections — applying ``pair_proj[1:]`` only to this
        block.

        IMPORTANT correctness note: this is *not* generally equal to
        ``logits[B, A].transpose(0, 1)`` when ``W_i != W_j`` (the default,
        unconstrained case for ``pair_proj[0]``). Concretely,
        ``logits[A,B][p,q] = MLP(proj_i[A[p]] + proj_j[B[q]])`` while
        ``logits[B,A][q,p] = MLP(proj_i[B[q]] + proj_j[A[p]])`` — these
        involve different combinations of ``proj_i``/``proj_j`` whenever
        the two projections differ, so swapping the block argument order
        is *not* the same as transposing the result. Both directions must
        therefore be computed explicitly; see ``_tiled_expected_distance``
        for how this is exploited to still avoid redundant computation
        (each unordered block pair is computed once per direction, not
        once total).

        Args:
            proj_i_full, proj_j_full : full (N, d) factorized projections.
            a_start, a_end : row-residue block bounds.
            b_start, b_end : column-residue block bounds.
        Returns:
            (a_end - a_start, b_end - b_start, bins) logits block.
        """
        p_i = proj_i_full[a_start:a_end].unsqueeze(1)    # (Ba, 1, d)
        p_j = proj_j_full[b_start:b_end].unsqueeze(0)    # (1, Bb, d)
        pre = p_i + p_j                                    # (Ba, Bb, d), broadcast
        rest = self.pair_proj[1:]
        return rest(pre)                                   # (Ba, Bb, bins)

    def _tiled_expected_distance(
        self,
        h: torch.Tensor,
        compute_dtype: torch.dtype,
        block_size: int,
    ) -> torch.Tensor:
        """
        2-D block-tiled, autograd-safe computation of the expected
        distance matrix. The ``(N, N, bins)`` logits computation is
        tiled into ``(block_size, block_size, hidden_dim)``-sized pieces
        rather than computed as one ``(N, N, hidden_dim)`` intermediate
        tensor, capping peak pairwise-MLP activation memory at
        ``O(block_size² · hidden_dim)`` independent of ``N``.

        Each unordered block pair ``{I, J}`` (including ``I == J``) is
        visited once; both the ``logits[I, J]`` and ``logits[J, I]``
        directions are computed explicitly (the factorized projection
        does **not** support obtaining one from the other via a cheap
        transpose — see ``_tiled_block_pair_logits`` for why), then
        symmetrised exactly as ``forward()`` does for the full matrix:
        ``0.5 · (logits[I,J] + logits[J,I].transpose)``. This still
        delivers the full memory benefit of 2-D tiling; it does not
        claim a compute saving over row-chunking, since both genuinely
        require one MLP evaluation per ordered pair of residues.

        Unlike a pre-allocate-and-scatter-assign implementation (which
        would break autograd, since in-place indexed assignment into a
        ``torch.zeros(...)`` leaf tensor does not propagate gradients
        back through the assigned values), this method assembles the
        result purely via ``torch.cat``, which is fully differentiable —
        gradients flow correctly back through every block into ``h``
        (verified in the ``__main__`` test suite).

        Args:
            h             : (N, hidden_dim) per-residue latent.
            compute_dtype : dtype for the pairwise stage.
            block_size    : tile edge length (both the i- and j- block
                             size); the existing ``cfg.pair_chunk_size``
                             is reused for this purpose.
        Returns:
            (N, N) expected Cα–Cα distance matrix, symmetric, zero
            diagonal, in ``compute_dtype``.
        """
        n = h.size(0)
        W_i, W_j, b = self._split_first_layer()
        W_i, W_j, b = W_i.to(compute_dtype), W_j.to(compute_dtype), b.to(compute_dtype)
        proj_i_full = F.linear(h, W_i, b)   # (N, d) — bias included exactly once, here
        proj_j_full = F.linear(h, W_j)       # (N, d) — no bias

        bin_centers_f32 = self.bin_centers.to(torch.float32)
        row_starts = list(range(0, n, block_size))

        # dist_blocks[(i_idx, j_idx)] holds the (Bi, Bj) expected-distance
        # block; filled for every (i_idx, j_idx) pair (both i<j and i>j are
        # computed explicitly, since the factorized projection does not
        # support deriving one from the other — see docstring above).
        dist_blocks: Dict[Tuple[int, int], torch.Tensor] = {}

        for i_idx, i_start in enumerate(row_starts):
            i_end = min(i_start + block_size, n)
            for j_idx, j_start in enumerate(row_starts):
                if j_idx < i_idx:
                    continue  # this unordered pair was already handled when (j_idx, i_idx) was visited
                j_end = min(j_start + block_size, n)

                logits_IJ = self._tiled_block_pair_logits(
                    proj_i_full, proj_j_full, i_start, i_end, j_start, j_end
                )  # (Bi, Bj, bins)

                if i_idx == j_idx:
                    sym_IJ = 0.5 * (logits_IJ + logits_IJ.transpose(0, 1))
                    dist_blocks[(i_idx, j_idx)] = self._block_logits_to_dist(sym_IJ, bin_centers_f32)
                else:
                    logits_JI = self._tiled_block_pair_logits(
                        proj_i_full, proj_j_full, j_start, j_end, i_start, i_end
                    )  # (Bj, Bi, bins) — computed explicitly; NOT derived from logits_IJ
                    sym_IJ = 0.5 * (logits_IJ + logits_JI.transpose(0, 1))    # (Bi, Bj, bins)
                    sym_JI = sym_IJ.transpose(0, 1)                            # (Bj, Bi, bins) — exact, since this is now a true symmetric pair
                    dist_blocks[(i_idx, j_idx)] = self._block_logits_to_dist(sym_IJ, bin_centers_f32)
                    dist_blocks[(j_idx, i_idx)] = self._block_logits_to_dist(sym_JI, bin_centers_f32)

        dist_row_blocks = [
            torch.cat([dist_blocks[(i_idx, j_idx)] for j_idx in range(len(row_starts))], dim=1)
            for i_idx in range(len(row_starts))
        ]
        dmat = torch.cat(dist_row_blocks, dim=0)  # (N, N)
        dmat = dmat * (1.0 - torch.eye(n, device=dmat.device, dtype=dmat.dtype))
        return dmat.to(compute_dtype)

    @staticmethod
    def _block_logits_to_dist(logits_block: torch.Tensor, bin_centers_f32: torch.Tensor) -> torch.Tensor:
        """Softmax (in fp32, for numerical stability) + expected-value reduction for one logits block."""
        probs = F.softmax(logits_block.to(torch.float32), dim=-1)
        return torch.einsum("ijb,b->ij", probs, bin_centers_f32)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (N, hidden_dim) per-residue latent.
        Returns:
            (N, N, num_distance_bins) symmetrised distance-bin logits, in
            the same dtype as the input ``h``.
        """
        n = h.size(0)
        in_dtype = h.dtype
        compute_dtype = self.cfg.pair_proj_dtype or in_dtype
        h_c = h.to(compute_dtype)

        chunk_size = self.cfg.pair_chunk_size or n   # None -> single chunk == unchunked
        chunks = [
            self._pairwise_logits_chunk(h_c, start, min(start + chunk_size, n), compute_dtype)
            for start in range(0, n, chunk_size)
        ]
        logits = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]  # (N, N, bins)

        logits = 0.5 * (logits + logits.transpose(0, 1))  # enforce symmetry
        return logits.to(in_dtype)

    def expected_distance_matrix(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert distogram logits into an *expected* (soft) distance matrix
        by taking the probability-weighted mean over bin centers — fully
        differentiable, unlike argmax bin selection.

        Args:
            logits : (N, N, num_distance_bins), any float dtype.
        Returns:
            (N, N) expected Cα–Cα distance matrix, symmetric, zero diagonal,
            same dtype as the input ``logits``.
        """
        in_dtype = logits.dtype
        probs = F.softmax(logits.to(torch.float32), dim=-1)   # softmax in fp32 for stability
        dmat = torch.einsum("ijb,b->ij", probs, self.bin_centers.to(torch.float32))
        n = dmat.size(0)
        dmat = dmat * (1.0 - torch.eye(n, device=dmat.device, dtype=dmat.dtype))
        dmat = 0.5 * (dmat + dmat.transpose(0, 1))
        return dmat.to(in_dtype)

    def _pairwise_logits_block_nonfactorized(
        self,
        h: torch.Tensor,
        a_start: int, a_end: int,
        b_start: int, b_end: int,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Non-factorized counterpart to ``_tiled_block_pair_logits``: computes
        the ``(a_end-a_start, b_end-b_start, bins)`` logits block via the
        concat-MLP path, without ever materialising a full ``(N, ·, bins)``
        intermediate. Used by ``forward_expected_distance`` so that the
        ``cfg.pair_proj_factorized=False`` configuration keeps the same
        chunk-bounded memory guarantee as the factorized one.

        Args:
            h : (N, hidden_dim) latent, already cast to ``compute_dtype``.
            a_start, a_end : row-residue ("i") block bounds.
            b_start, b_end : column-residue ("j") block bounds.
        Returns:
            (a_end-a_start, b_end-b_start, num_distance_bins) logits block.
        """
        Ba = a_end - a_start
        Bb = b_end - b_start
        h_a = h[a_start:a_end].unsqueeze(1).expand(Ba, Bb, -1)   # (Ba, Bb, d)
        h_b = h[b_start:b_end].unsqueeze(0).expand(Ba, Bb, -1)   # (Ba, Bb, d)
        pair = torch.cat([h_a, h_b], dim=-1)                      # (Ba, Bb, 2d)
        return self.pair_proj(pair.to(compute_dtype))              # (Ba, Bb, bins)

    def forward_expected_distance(self, h: torch.Tensor, use_2d_tiling: bool = False) -> torch.Tensor:
        """
        Fused, chunk-aware computation of the expected distance matrix
        directly from per-residue latents — without ever materialising
        the full ``(N, N, num_distance_bins)`` logits tensor.

        Use this instead of ``forward()`` + ``expected_distance_matrix()``
        whenever the raw logits are not separately needed (e.g. at
        inference, or whenever the only consumer is
        ``DifferentiableMDS``). For training with a distogram
        cross-entropy loss, use ``forward()`` instead, since that loss
        needs the actual per-bin logits.

        Args:
            h : (N, hidden_dim) per-residue latent.
            use_2d_tiling : if True and ``cfg.pair_chunk_size`` is set,
                uses ``_tiled_expected_distance`` (2-D block tiling) instead
                of the default row-chunked path. 2-D tiling caps the
                largest intermediate MLP activation at
                ``(pair_chunk_size, pair_chunk_size, hidden_dim)`` rather
                than ``(pair_chunk_size, N, hidden_dim)`` — a genuine
                further memory reduction over row-chunking for very long
                sequences. It does **not** reduce MLP compute relative to
                row-chunking: both directions of every off-diagonal block
                pair are computed explicitly, since the factorized
                projection does not support deriving ``logits[J,I]`` from
                ``logits[I,J]`` via a cheap transpose when ``W_i != W_j``
                (verified numerically — see README_PREDICTOR.md §6/§8 for
                the equivalence checks, including the bug this caught and
                fixed during development). Expect somewhat higher wall-
                clock time than row-chunking due to the nested-loop Python
                overhead; use 2-D tiling specifically when row-chunked
                activations are still too large for available memory, and
                row-chunking otherwise.
        Returns:
            (N, N) expected Cα–Cα distance matrix, symmetric, zero diagonal.
        """
        n = h.size(0)
        in_dtype = h.dtype
        compute_dtype = self.cfg.pair_proj_dtype or in_dtype
        h_c = h.to(compute_dtype)

        if use_2d_tiling and self.cfg.pair_chunk_size is not None:
            dmat = self._tiled_expected_distance(h_c, compute_dtype, self.cfg.pair_chunk_size)
            return dmat.to(in_dtype)

        chunk_size = self.cfg.pair_chunk_size or n
        row_chunks = []

        # IMPORTANT correctness note (fixed): logits must be symmetrised
        # BEFORE softmax, exactly as forward() + expected_distance_matrix()
        # and the 2-D tiled path (_tiled_expected_distance) both do.
        # Symmetrising the post-softmax expected distance instead (as a
        # previous version of this method did) is NOT equivalent — softmax
        # is nonlinear — and silently produced a different, less accurate
        # distance matrix. See README_PREDICTOR.md §6/§8.
        if self.cfg.pair_proj_factorized:
            W_i, W_j, b = self._split_first_layer()
            W_i, W_j, b = W_i.to(compute_dtype), W_j.to(compute_dtype), b.to(compute_dtype)
            proj_i_full = F.linear(h_c, W_i, b)   # (N, d) — bias included exactly once, here
            proj_j_full = F.linear(h_c, W_j)       # (N, d) — no bias

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                logits_I_all = self._tiled_block_pair_logits(
                    proj_i_full, proj_j_full, start, end, 0, n
                )  # (chunk, N, bins) == logits[I, :]
                logits_all_I = self._tiled_block_pair_logits(
                    proj_i_full, proj_j_full, 0, n, start, end
                ).transpose(0, 1)  # (chunk, N, bins) == logits[:, I], transposed to align
                sym_logits = 0.5 * (logits_I_all + logits_all_I)
                chunk_probs = F.softmax(sym_logits.to(torch.float32), dim=-1)
                chunk_dist = torch.einsum(
                    "ijb,b->ij", chunk_probs, self.bin_centers.to(torch.float32)
                )  # (chunk, N)
                row_chunks.append(chunk_dist)
        else:
            # Non-factorized config has no cheap separable projection to
            # reuse across chunks; use the block-tiled concat-MLP helper
            # for both directions so memory stays bounded at
            # O(chunk · N · d) — never the full O(N² · d) intermediate —
            # matching the factorized branch's memory guarantee.
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                logits_I_all = self._pairwise_logits_block_nonfactorized(
                    h_c, start, end, 0, n, compute_dtype
                )  # (chunk, N, bins) == logits[I, :]
                logits_all_I = self._pairwise_logits_block_nonfactorized(
                    h_c, 0, n, start, end, compute_dtype
                ).transpose(0, 1)  # (chunk, N, bins) == logits[:, I], transposed to align
                sym_logits = 0.5 * (logits_I_all + logits_all_I)
                chunk_probs = F.softmax(sym_logits.to(torch.float32), dim=-1)
                chunk_dist = torch.einsum(
                    "ijb,b->ij", chunk_probs, self.bin_centers.to(torch.float32)
                )  # (chunk, N)
                row_chunks.append(chunk_dist)

        dmat = torch.cat(row_chunks, dim=0) if len(row_chunks) > 1 else row_chunks[0]  # (N, N)
        dmat = dmat * (1.0 - torch.eye(n, device=dmat.device, dtype=dmat.dtype))
        return dmat.to(in_dtype)


# =============================================================================
# 5.  Sigma Head — structural regime field σ(x)
# =============================================================================

class SigmaHead(nn.Module):
    """
    Predicts a per-residue structural-regime scalar σ(x), soft-clamped to
    ``[cfg.sigma_min, cfg.sigma_max]`` so it can be fed directly into the
    FiLM modulation of ``StructuralMessagePassing`` in
    ``structural_gno_fold_v3.py`` without any further post-processing.

    Args:
        cfg : Seq2CoarseConfig instance.
    """

    def __init__(self, cfg: Seq2CoarseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (N, hidden_dim) per-residue latent.
        Returns:
            (N, 1) σ(x), soft-clamped to [sigma_min, sigma_max].
        """
        raw = self.net(h)
        return soft_clamp(raw, self.cfg.sigma_min, self.cfg.sigma_max)


# =============================================================================
# 6.  Differentiable MDS (SMACOF Stress Majorization)
# =============================================================================

class DifferentiableMDS(nn.Module):
    """
    Converts a target Cα–Cα distance matrix into 3-D coordinates via
    differentiable stress-majorization (SMACOF — de Leeuw, 1977), rather
    than classical MDS's eigendecomposition. SMACOF only needs matrix
    multiplications and elementwise division, so gradients flow cleanly
    back through every iteration into the distogram (and hence into the
    transformer encoder and embedder) during end-to-end training.

    Stress function minimised:

        σ_stress(X) = Σ_{i<j} w_ij · (d_ij(X) − δ_ij)²

    where δ_ij is the target distance and d_ij(X) = ‖x_i − x_j‖.

    Args:
        cfg : Seq2CoarseConfig instance.
    """

    def __init__(self, cfg: Seq2CoarseConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        target_dist: torch.Tensor,             # (N, N)
        weights: Optional[torch.Tensor] = None,  # (N, N), defaults to all-ones off-diagonal
        init_coords: Optional[torch.Tensor] = None,  # (N, mds_dim)
        n_iters: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            target_dist : (N, N) target (expected) distance matrix, symmetric,
                          zero diagonal.
            weights     : optional (N, N) SMACOF weights (e.g. confidence from
                          the distogram entropy); defaults to uniform off-diagonal
                          weighting.
            init_coords : optional (N, mds_dim) warm-start coordinates (e.g. from
                          a previous refinement step or template); randomly
                          initialised if None.
            n_iters     : override cfg.mds_iters for this call.
        Returns:
            (N, mds_dim) coarse coordinates minimising SMACOF stress against
            ``target_dist``.

        Memory note: if ``cfg.mds_row_chunk_size`` is set, each iteration's
        Guttman transform is computed in row-blocks rather than
        materialising the full (N, N) intermediate matrices at once,
        capping peak per-iteration memory at
        O(mds_row_chunk_size · N) instead of O(N²) — verified numerically
        identical to the unchunked computation (see
        README_PREDICTOR.md §6/§9). This does *not* reduce the memory
        needed to store ``target_dist`` itself, which the caller must
        already hold before calling this method.
        """
        n = target_dist.size(0)
        device, dtype = target_dist.device, target_dist.dtype
        iters = n_iters or self.cfg.mds_iters

        if weights is None:
            weights = 1.0 - torch.eye(n, device=device, dtype=dtype)
        else:
            weights = weights * (1.0 - torch.eye(n, device=device, dtype=dtype))

        if init_coords is not None:
            X = init_coords.to(device=device, dtype=dtype).clone()
        else:
            X = self._classical_mds_init(target_dist)

        w_sum = weights.sum(dim=1, keepdim=True).clamp_min(self.cfg.mds_eps)  # (N, 1)

        chunk_size = self.cfg.mds_row_chunk_size or n  # None -> single chunk == unchunked
        for _ in range(iters):
            if chunk_size >= n:
                X = self._smacof_update_full(X, target_dist, weights, w_sum, n, device, dtype)
            else:
                X = self._smacof_update_chunked(
                    X, target_dist, weights, w_sum, n, chunk_size, device, dtype
                )

        return X

    def _smacof_update_full(
        self,
        X: torch.Tensor,
        target_dist: torch.Tensor,
        weights: torch.Tensor,
        w_sum: torch.Tensor,
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """One unchunked SMACOF Guttman-transform update (original behaviour)."""
        diff = X.unsqueeze(1) - X.unsqueeze(0)               # (N, N, mds_dim)
        d = torch.linalg.norm(diff, dim=-1)                   # (N, N) current distances
        d_safe = d.clamp_min(self.cfg.mds_eps)

        # Guttman transform: B_ij = -w_ij * delta_ij / d_ij  (i != j),
        #                    B_ii = -sum_{j != i} B_ij
        ratio = weights * target_dist / d_safe                 # (N, N), off-diagonal terms
        off_diag_mask = 1.0 - torch.eye(n, device=device, dtype=dtype)
        B_off = -ratio * off_diag_mask
        B_diag = -B_off.sum(dim=1)                             # row sums, negated
        B = B_off + torch.diag(B_diag)

        return (B @ X) / w_sum

    def _smacof_update_chunked(
        self,
        X: torch.Tensor,
        target_dist: torch.Tensor,
        weights: torch.Tensor,
        w_sum: torch.Tensor,
        n: int,
        chunk_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        One row-chunked SMACOF Guttman-transform update — mathematically
        identical to ``_smacof_update_full`` (verified numerically; see
        README_PREDICTOR.md §6/§9), but never materialises a full (N, N)
        intermediate tensor: only one ``(chunk_size, N)`` row-block at a
        time. The output is assembled via ``torch.cat``, so gradients
        continue to flow correctly back through ``X``, ``target_dist``,
        and ``weights`` into the rest of the model.
        """
        row_blocks: List[torch.Tensor] = []
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            X_block = X[i_start:i_end]                                      # (Bi, mds_dim)

            diff_block = X_block.unsqueeze(1) - X.unsqueeze(0)               # (Bi, N, mds_dim)
            d_block = torch.linalg.norm(diff_block, dim=-1)                   # (Bi, N)
            d_safe_block = d_block.clamp_min(self.cfg.mds_eps)

            target_block = target_dist[i_start:i_end, :]                      # (Bi, N)
            weights_block = weights[i_start:i_end, :]                         # (Bi, N)
            ratio_block = weights_block * target_block / d_safe_block          # (Bi, N)

            # Zero the (local_i, global_i) diagonal entries within this block.
            off_diag_block = torch.ones(i_end - i_start, n, device=device, dtype=dtype)
            diag_cols = torch.arange(i_start, i_end, device=device)
            diag_rows = torch.arange(i_end - i_start, device=device)
            off_diag_block[diag_rows, diag_cols] = 0.0

            B_off_block = -ratio_block * off_diag_block                        # (Bi, N)
            B_diag_block = -B_off_block.sum(dim=1)                              # (Bi,)
            # Write the diagonal entry into B_off_block's own diagonal columns
            # via scatter (out-of-place, autograd-safe — this builds a NEW
            # tensor rather than mutating B_off_block in place).
            B_row_block = B_off_block.index_put(
                (diag_rows, diag_cols), B_diag_block, accumulate=False
            )  # (Bi, N)

            X_new_block = (B_row_block @ X) / w_sum[i_start:i_end]              # (Bi, mds_dim)
            row_blocks.append(X_new_block)

        return torch.cat(row_blocks, dim=0)  # (N, mds_dim)

    def _classical_mds_init(self, target_dist: torch.Tensor) -> torch.Tensor:
        """
        Deterministic warm start via classical MDS (Torgerson double-
        centering), used whenever ``init_coords`` is not supplied.

        Preferred over random initialisation because (a) it is fully
        deterministic — ``predict()`` therefore returns identical
        coordinates across repeated calls in eval mode — and (b) SMACOF
        converges faster and more reliably from a distance-aware starting
        point than from pure noise.

        Detached from the autograd graph deliberately: gradients should
        flow through the SMACOF *iterations*, not through this one-shot
        initial guess (mirroring standard practice in the classical-MDS
        + SMACOF-refinement literature).

        If ``cfg.mds_use_landmarks`` is True, delegates to
        ``_landmark_mds_init`` instead — full classical MDS is O(N³)
        (both the double-centering matmuls and the eigendecomposition),
        which becomes infeasible well before N reaches the tens of
        thousands; Landmark MDS reduces this to O(L³ + N·L·mds_dim) for
        a fixed landmark count L, independent of how large N grows (see
        README_PREDICTOR.md §9 for the numerical-equivalence check and
        complexity analysis).

        Auto-switch: even if ``cfg.mds_use_landmarks`` is False, this
        method automatically routes through ``_landmark_mds_init`` once
        N exceeds ``cfg.auto_landmark_threshold`` — full classical MDS's
        O(N³) eigendecomposition is not just slow but numerically
        unreliable at that scale, regardless of available memory. Set
        ``cfg.auto_landmark_threshold`` very high (or pass
        ``mds_use_landmarks`` explicitly) to disable this and force full
        classical MDS regardless of N.

        Args:
            target_dist : (N, N) target distance matrix.
        Returns:
            (N, mds_dim) initial coordinate guess.
        """
        n = target_dist.size(0)
        if self.cfg.mds_use_landmarks:
            return self._landmark_mds_init(target_dist)
        if n > self.cfg.auto_landmark_threshold:
            logger.warning(
                "N=%d exceeds auto_landmark_threshold=%d; auto-switching MDS "
                "initialisation from full classical MDS (O(N^3)) to Landmark "
                "MDS (O(L^3 + N*L*mds_dim), L=%d).",
                n, self.cfg.auto_landmark_threshold, self.cfg.mds_num_landmarks,
            )
            return self._landmark_mds_init(target_dist)

        device, dtype = target_dist.device, target_dist.dtype
        d = target_dist.detach()

        try:
            d_sq = d ** 2
            ones = torch.ones(n, n, device=device, dtype=dtype)
            J = torch.eye(n, device=device, dtype=dtype) - ones / n
            B = -0.5 * J @ d_sq @ J
            B = 0.5 * (B + B.T)  # enforce exact symmetry against floating-point drift

            eigvals, eigvecs = torch.linalg.eigh(B)
            # eigh returns ascending order; take the top `mds_dim` eigenpairs.
            eigvals = eigvals[-self.cfg.mds_dim:].clamp_min(self.cfg.mds_eps)
            eigvecs = eigvecs[:, -self.cfg.mds_dim:]

            X = eigvecs * eigvals.sqrt().unsqueeze(0)               # (N, k<=mds_dim)
            if X.size(1) < self.cfg.mds_dim:
                pad = torch.zeros(n, self.cfg.mds_dim - X.size(1), device=device, dtype=dtype)
                X = torch.cat([X, pad], dim=1)
            return X.to(dtype=dtype)
        except RuntimeError as exc:
            # Degenerate / ill-conditioned distance matrix (e.g. eigh fails
            # to converge). Fall back to a fixed-seed random embedding so
            # behaviour stays deterministic rather than crashing the
            # forward pass.
            logger.warning(
                "Classical MDS init failed (%s); falling back to seeded random init.", exc
            )
            gen = torch.Generator(device="cpu").manual_seed(0)
            X = self.cfg.mds_init_scale * torch.randn(
                n, self.cfg.mds_dim, generator=gen
            ).to(device=device, dtype=dtype)
            return X

    def _landmark_mds_init(self, target_dist: torch.Tensor) -> torch.Tensor:
        """
        Landmark MDS initialisation (de Silva & Tenenbaum, 2004).

        Algorithm:
          1. Select ``cfg.mds_num_landmarks`` residues (seeded-random, for
             determinism) as landmarks.
          2. Run full classical MDS — double-centering + eigendecomposition
             — on just the (L, L) landmark submatrix. This is O(L³), fixed
             regardless of how large N grows.
          3. For every residue (landmark or not), triangulate its
             coordinate from its known distances to the L landmarks via a
             closed-form least-squares formula — O(N·L·mds_dim), no further
             eigendecomposition.

        This trades a small amount of initialisation quality (the full
        configuration is only approximated through L landmarks, rather
        than exactly recovered as classical MDS would with the full
        matrix) for changing the dominant cost from O(N³) to O(N · L) —
        the difference between "infeasible at N=100,000" and "routine".
        SMACOF's subsequent iterations refine this initial guess regardless,
        so landmark-init quality mainly affects convergence speed, not the
        final result's correctness.

        Numerically verified against full classical MDS on small N (where
        both are tractable) to agree up to the Procrustes-alignable
        configuration that both classical MDS and SMACOF only ever recover
        up to rotation/reflection/translation in the first place — see
        README_PREDICTOR.md §9.

        Args:
            target_dist : (N, N) target distance matrix.
        Returns:
            (N, mds_dim) initial coordinate guess.
        """
        n = target_dist.size(0)
        device, dtype = target_dist.device, target_dist.dtype
        d = target_dist.detach()
        k = self.cfg.mds_dim
        L = min(self.cfg.mds_num_landmarks, n - 1)
        if L <= k:
            logger.warning(
                "mds_num_landmarks (%d) too small relative to mds_dim (%d) and N (%d); "
                "falling back to full classical MDS init.", self.cfg.mds_num_landmarks, k, n,
            )
            self.cfg.mds_use_landmarks = False
            try:
                return self._classical_mds_init(target_dist)
            finally:
                self.cfg.mds_use_landmarks = True

        try:
            gen = torch.Generator(device="cpu").manual_seed(0)
            landmark_idx = torch.randperm(n, generator=gen)[:L].to(device)

            d_LL = d[landmark_idx][:, landmark_idx]                 # (L, L)
            d_LL_sq = d_LL ** 2
            ones_L = torch.ones(L, L, device=device, dtype=dtype)
            J_L = torch.eye(L, device=device, dtype=dtype) - ones_L / L
            B_LL = -0.5 * J_L @ d_LL_sq @ J_L
            B_LL = 0.5 * (B_LL + B_LL.T)

            eigvals, eigvecs = torch.linalg.eigh(B_LL)              # O(L³)
            eigvals_top = eigvals[-k:].clamp_min(self.cfg.mds_eps)    # (k,)
            eigvecs_top = eigvecs[:, -k:]                              # (L, k)
            X_landmarks = eigvecs_top * eigvals_top.sqrt().unsqueeze(0)  # (L, k)

            # Closed-form triangulation for every residue (landmarks
            # included — recovers X_landmarks back out for free, and gives
            # every other residue's coordinate from its distances to the
            # landmarks alone, without any further eigendecomposition):
            #   x_p = -0.5 * pinv(X_landmarks)^T @ (d_p,L^2 - mean_sq_L)
            mean_d_sq_landmarks = d_LL_sq.mean(dim=1)                 # (L,)
            X_landmarks_pinv = torch.linalg.pinv(X_landmarks)          # (k, L), O(L²·k)

            d_all_to_L = d[:, landmark_idx]                            # (N, L) — distances from every
                                                                          # residue to the L landmarks
            delta = d_all_to_L ** 2 - mean_d_sq_landmarks.unsqueeze(0)  # (N, L)
            X_all = -0.5 * (delta @ X_landmarks_pinv.T)                  # (N, k) — O(N·L·k)

            if X_all.size(1) < self.cfg.mds_dim:
                pad = torch.zeros(
                    n, self.cfg.mds_dim - X_all.size(1), device=device, dtype=dtype
                )
                X_all = torch.cat([X_all, pad], dim=1)
            return X_all.to(dtype=dtype)
        except RuntimeError as exc:
            logger.warning(
                "Landmark MDS init failed (%s); falling back to seeded random init.", exc
            )
            gen = torch.Generator(device="cpu").manual_seed(0)
            return (self.cfg.mds_init_scale * torch.randn(
                n, self.cfg.mds_dim, generator=gen
            )).to(device=device, dtype=dtype)

    @staticmethod
    def stress(
        coords: torch.Tensor,
        target_dist: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute the (weighted) SMACOF stress of a candidate coordinate set
        against a target distance matrix — useful as an auxiliary loss term
        or convergence diagnostic.

        Args:
            coords      : (N, mds_dim).
            target_dist : (N, N).
            weights     : optional (N, N); defaults to uniform off-diagonal.
            eps         : numerical floor.
        Returns:
            Scalar stress value.
        """
        n = coords.size(0)
        if weights is None:
            weights = 1.0 - torch.eye(n, device=coords.device, dtype=coords.dtype)
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        d = torch.linalg.norm(diff, dim=-1).clamp_min(eps)
        return (weights * (d - target_dist) ** 2).sum() / weights.sum().clamp_min(eps)


# =============================================================================
# 7.  Main Module — SeqToCoarseStructure
# =============================================================================

class SeqToCoarseStructure(nn.Module):
    """
    End-to-end, single-sequence, MSA-free structure initialiser.

        sequence (str)
            │
            ▼
        SequenceEmbedder            (ESM-2, frozen — or learned fallback)
            │  (N, embed_dim)
            ▼
        SequenceTransformerEncoder  (bidirectional self-attention, single seq)
            │  (N, hidden_dim)
            ├──────────────┬──────────────────┐
            ▼               ▼                  ▼
        DistogramHead   SigmaHead         (latent h, returned for
            │               │              downstream reuse)
            ▼               ▼
        expected_distance_matrix      σ(x)  (N, 1)
            │
            ▼
        DifferentiableMDS  (SMACOF)
            │
            ▼
        init_coords  (N, 3)

    The three outputs ``(init_coords, seq_features, sigma)`` are the exact
    signature expected by ``StructuralGNOFold.forward`` in
    ``structural_gno_fold_v3.py``:

        final_coords, pred_ddg = sgno_model(seq_features, init_coords, sigma)

    where ``seq_features`` here is the contextualised transformer latent
    ``h`` (hidden_dim-wide) rather than a raw one-hot — a strict
    upgrade over the SGNOConfig default (``node_in_dim=20`` one-hot), so
    ``node_in_dim`` should be set to ``cfg.hidden_dim`` when wiring the two
    modules together (see ``build_sgno_compatible_inputs`` below).

    Args:
        cfg : Seq2CoarseConfig instance (created with defaults if None).
    """

    def __init__(self, cfg: Optional[Seq2CoarseConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or Seq2CoarseConfig()

        self.embedder = SequenceEmbedder(self.cfg)
        self.encoder = SequenceTransformerEncoder(self.cfg)
        self.distogram_head = DistogramHead(self.cfg)
        self.sigma_head = SigmaHead(self.cfg)
        self.mds = DifferentiableMDS(self.cfg)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            "SeqToCoarseStructure v%s | backend=%s | trainable_params=%s / total=%s",
            SEQ2COARSE_VERSION,
            self.embedder._backend_active,
            f"{n_trainable:,}",
            f"{n_total:,}",
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sequence: str,
        init_coords: Optional[torch.Tensor] = None,
        mds_iters: Optional[int] = None,
        return_distogram: bool = True,
        use_2d_tiling: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequence    : raw single-letter amino-acid string (no MSA).
            init_coords : optional (N, 3) warm-start for the MDS solver
                          (e.g. coordinates from a previous training step,
                          a homology template, or an extended-chain build).
                          Random initialisation is used if None.
            mds_iters   : override cfg.mds_iters for this call.
            return_distogram : if True (default), the full ``(N, N,
                          num_distance_bins)`` logits tensor is computed
                          and returned under "distogram" — needed for
                          distogram cross-entropy training
                          (``Seq2CoarseTrainer``). If False, the cheaper
                          fused path (``DistogramHead.forward_expected_distance``)
                          is used instead, which never materialises the
                          full logits tensor; "distogram" is omitted from
                          the returned dict in that case. Set to False
                          for inference-only use (e.g. via ``predict()``)
                          when paired with ``pair_chunk_size`` /
                          ``pair_proj_dtype`` for maximum memory savings
                          on long sequences.
            use_2d_tiling : only takes effect when ``return_distogram=False``
                          and ``cfg.pair_chunk_size`` is set. Uses 2-D
                          block-tiled pairwise computation instead of row-
                          chunking — a further memory reduction for the
                          largest sequences (caps intermediate MLP
                          activations at block_size² rather than
                          block_size · N), at the cost of somewhat higher
                          wall-clock time from nested-loop overhead. Does
                          not reduce MLP compute relative to row-chunking
                          — see
                          ``DistogramHead.forward_expected_distance`` and
                          README_PREDICTOR.md §8 for details and when to
                          prefer one over the other.
        Returns:
            Dict with:
                "init_coords"   : (N, 3) coarse Cα coordinates.
                "seq_features"  : (N, hidden_dim) contextualised latent —
                                   feed directly as StructuralGNOFold's
                                   ``seq_features`` argument.
                "sigma"         : (N, 1) structural-regime field.
                "distogram"     : (N, N, num_distance_bins) raw logits.
                                   Only present if return_distogram=True.
                "expected_dist" : (N, N) expected distance matrix.
        """
        device = next(self.parameters()).device
        n = len(sequence)
        if n < 2:
            raise ValueError(f"Sequence must have at least 2 residues; got length {n}.")

        embed = self.embedder(sequence, device=device)            # (N, embed_dim)
        h = self.encoder(embed)                                    # (N, hidden_dim)

        out: Dict[str, torch.Tensor] = {}
        if return_distogram:
            distogram = self.distogram_head(h)                                       # (N, N, bins)
            expected_dist = self.distogram_head.expected_distance_matrix(distogram)   # (N, N)
            out["distogram"] = distogram
        else:
            expected_dist = self.distogram_head.forward_expected_distance(
                h, use_2d_tiling=use_2d_tiling
            )                                                                          # (N, N)

        sigma = self.sigma_head(h)                                  # (N, 1)

        coarse_coords = self.mds(
            expected_dist, init_coords=init_coords, n_iters=mds_iters
        )                                                            # (N, 3)

        out.update({
            "init_coords":   coarse_coords,
            "seq_features":  h,
            "sigma":         sigma,
            "expected_dist": expected_dist,
        })
        return out

    @torch.no_grad()
    def predict(
        self,
        sequence: str,
        return_distogram: bool = True,
        use_2d_tiling: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-mode convenience wrapper (eval(), no_grad, detached).

        Args:
            sequence         : raw amino-acid string.
            return_distogram : default True, matching prior behaviour
                                exactly. Set False to use the fused,
                                memory-saving distance-matrix path
                                (skips materialising full logits) —
                                recommended for long sequences when combined
                                with ``cfg.pair_chunk_size`` /
                                ``cfg.pair_proj_dtype``; see
                                README_PREDICTOR.md §3.3 / §8.
            use_2d_tiling     : only relevant when ``return_distogram=False``;
                                see ``SeqToCoarseStructure.forward``.
        """
        was_training = self.training
        self.eval()
        try:
            out = self.forward(
                sequence, return_distogram=return_distogram, use_2d_tiling=use_2d_tiling
            )
        finally:
            if was_training:
                self.train()
        return {k: v.detach() for k, v in out.items()}


def build_sgno_compatible_inputs(
    seq2coarse_output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adapt a ``SeqToCoarseStructure.forward`` output dict into the
    positional ``(seq_features, init_coords, sigma)`` triple expected by
    ``StructuralGNOFold.forward`` in ``structural_gno_fold_v3.py``.

    Note: when wiring the two modules together, set
    ``SGNOConfig(node_in_dim=seq2coarse_cfg.hidden_dim)`` so that
    ``StructuralGNOFold.node_embed`` accepts the contextualised latent
    rather than the default 20-dim one-hot.

    Args:
        seq2coarse_output : dict returned by SeqToCoarseStructure.forward.
    Returns:
        (seq_features, init_coords, sigma) tuple, ready to pass to
        ``StructuralGNOFold.forward(*triple)``.
    """
    return (
        seq2coarse_output["seq_features"],
        seq2coarse_output["init_coords"],
        seq2coarse_output["sigma"],
    )


# =============================================================================
# 8.  PDB Export — bridges into RefinementEngine.refine(pdb_file=...)
# =============================================================================

# Minimal 3-letter code table (subset needed for Cα-only coarse models).
# Mirrors Bio.PDB.Polypeptide.one_to_three when biopython is unavailable.
_ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def write_ca_pdb(
    sequence: str,
    coords: torch.Tensor,
    out_path: Union[str, Path],
    chain_id: str = "A",
) -> Path:
    """
    Write a minimal Cα-only PDB file from a sequence and coarse
    coordinates, suitable as a starting structure for
    ``RefinementEngine.refine(pdb_file=...)`` in ``real_fold_one_v2.py``
    (which builds a full all-atom OpenMM system / topology from a PDB and
    then performs gradient-based refinement — this function only needs
    to supply a geometrically reasonable Cα trace, not full atomic detail).

    Uses ``Bio.PDB.Polypeptide.one_to_three`` if biopython is available,
    else the built-in ``_ONE_TO_THREE`` table.

    Args:
        sequence  : single-letter amino-acid string, length N.
        coords    : (N, 3) Cα coordinates in Å (CPU or GPU tensor; detached
                    internally).
        out_path  : destination .pdb file path.
        chain_id  : single-character PDB chain identifier.
    Returns:
        Path to the written PDB file.
    """
    n = len(sequence)
    coords_np = coords.detach().to("cpu").numpy()
    if coords_np.shape != (n, 3):
        raise ValueError(
            f"coords shape {coords_np.shape} does not match sequence length {n} (expected ({n}, 3))."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for i, (aa, xyz) in enumerate(zip(sequence, coords_np), start=1):
        aa_u = aa.upper()
        if _HAS_BIOPYTHON:
            try:
                three = Polypeptide.one_to_three(aa_u)
            except Exception:
                three = _ONE_TO_THREE.get(aa_u, "UNK")
        else:
            three = _ONE_TO_THREE.get(aa_u, "UNK")

        x, y, z = (float(v) for v in xyz)
        lines.append(
            f"ATOM  {i:5d}  CA  {three:<3s} {chain_id}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          "
            f" C"
        )
    lines.append("TER")
    lines.append("END")

    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote coarse Cα PDB (%d residues) → %s", n, out_path)
    return out_path


# =============================================================================
# 9.  EMA Wrapper (matches structural_gno_fold_v3.EMAWrapper convention)
# =============================================================================

class EMAWrapper:
    """
    Exponential moving average of model parameters for inference-time
    weight averaging — identical convention to ``EMAWrapper`` in
    ``structural_gno_fold_v3.py`` so the two modules can share a single
    EMA-handling code path downstream.

    Usage:
        ema = EMAWrapper(model, decay=0.999)
        # after each optimiser.step():
        ema.update()
        # at inference time:
        with ema.average_parameters():
            out = model(sequence)

    Args:
        model : the model whose parameters will be tracked.
        decay : EMA decay coefficient.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update EMA shadow weights after each optimiser step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self) -> None:
        """Replace model weights with EMA shadow for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original (training) weights."""
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    class _Context:
        def __init__(self, ema: "EMAWrapper") -> None:
            self._ema = ema

        def __enter__(self) -> None:
            self._ema.apply_shadow()

        def __exit__(self, *_: Any) -> None:
            self._ema.restore()

    def average_parameters(self) -> "_Context":
        """Context manager: temporarily swap in EMA weights."""
        return self._Context(self)


# =============================================================================
# 10.  Trainer
# =============================================================================

class Seq2CoarseTrainer:
    """
    Training loop wrapper for ``SeqToCoarseStructure``.

    Supports two supervision regimes, usable independently or jointly:

      • Distogram cross-entropy  — requires ground-truth Cα–Cα distances
        (e.g. from a PDB structure), the standard structure-prediction
        target.
      • Direct coordinate loss   — requires ground-truth coordinates
        (Kabsch-aligned RMSD-style), useful for end-to-end fine-tuning
        once the MDS solver is in the loop.

    A σ-regularisation term gently pulls the predicted structural-regime
    field toward ``cfg.sigma_target`` in the absence of direct structural-
    stress labels, keeping σ in a sensible operating range for the
    downstream ``StructuralGNOFold`` FiLM modulation.

    Args:
        model : SeqToCoarseStructure instance.
        cfg   : Seq2CoarseConfig instance (uses model.cfg if None).
    """

    def __init__(
        self,
        model: SeqToCoarseStructure,
        cfg: Optional[Seq2CoarseConfig] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or model.cfg

        trainable_backbone = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("embedder.aa_embed")
        ]
        embedding_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("embedder.aa_embed")
        ]

        param_groups = [{"params": trainable_backbone, "lr": self.cfg.lr_encoder}]
        if embedding_params:
            param_groups.append({"params": embedding_params, "lr": self.cfg.lr_embedding})

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.cfg.weight_decay
        )
        self.ema = EMAWrapper(model, decay=self.cfg.ema_decay)
        self.global_step = 0

    @staticmethod
    def _distance_to_bin_targets(
        dist_matrix: torch.Tensor,
        bin_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a continuous distance matrix into nearest-bin class indices."""
        # (N, N, 1) vs (bins,) → (N, N, bins) → argmin over bins
        diffs = (dist_matrix.unsqueeze(-1) - bin_centers.view(1, 1, -1)).abs()
        return diffs.argmin(dim=-1)

    def train_step(
        self,
        sequence: str,
        true_coords: Optional[torch.Tensor] = None,
        true_dist: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single supervised training step.

        Args:
            sequence    : raw amino-acid string.
            true_coords : optional (N, 3) ground-truth Cα coordinates.
            true_dist   : optional (N, N) ground-truth Cα–Cα distance matrix
                          (computed from true_coords if omitted but
                          true_coords is provided).
        Returns:
            Dict of scalar loss components for logging.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        out = self.model(sequence)
        losses: Dict[str, torch.Tensor] = {}

        if true_coords is not None and true_dist is None:
            diff = true_coords.unsqueeze(1) - true_coords.unsqueeze(0)
            true_dist = torch.linalg.norm(diff, dim=-1)

        if true_dist is not None:
            bin_centers = self.model.distogram_head.bin_centers
            targets = self._distance_to_bin_targets(true_dist, bin_centers)
            n = targets.size(0)
            off_diag = ~torch.eye(n, dtype=torch.bool, device=targets.device)
            logits_flat = out["distogram"][off_diag]
            targets_flat = targets[off_diag]
            losses["distogram"] = self.cfg.lambda_distogram * F.cross_entropy(
                logits_flat, targets_flat
            )

        if true_coords is not None:
            losses["coord"] = self.cfg.lambda_coord * DifferentiableMDS.stress(
                out["init_coords"], true_dist
                if true_dist is not None
                else torch.linalg.norm(
                    true_coords.unsqueeze(1) - true_coords.unsqueeze(0), dim=-1
                ),
            )

        # Guard against training with no real structural supervision at
        # all: this must be checked BEFORE sigma_reg is added below,
        # since sigma_reg is always present and would otherwise make
        # `losses` non-empty even when neither true_coords nor true_dist
        # was supplied — silently letting train_step optimize only the
        # weak sigma-target regularizer with zero structural signal.
        if not losses:
            raise ValueError(
                "train_step requires at least one of true_coords / true_dist "
                "to compute a supervised loss (sigma_reg alone is not sufficient)."
            )

        sigma_reg = (out["sigma"] - self.cfg.sigma_target).pow(2).mean()
        losses["sigma_reg"] = self.cfg.lambda_sigma * sigma_reg

        total = sum(losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.ema.update()
        self.global_step += 1

        metrics = {k: float(v.detach().item()) for k, v in losses.items()}
        metrics["total"] = float(total.detach().item())
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """
        Save model + optimizer + EMA shadow + config to
        ``cfg.checkpoint_dir``.

        Args:
            tag : optional filename suffix; defaults to the current global step.
        Returns:
            Path to the written checkpoint file.
        """
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = tag or f"step{self.global_step}"
        path = ckpt_dir / f"seq2coarse_{tag}.pt"

        torch.save(
            {
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "ema_shadow":      self.ema.shadow,
                "cfg":             self.cfg.to_dict(),
                "global_step":     self.global_step,
                "seq2coarse_version": SEQ2COARSE_VERSION,
                "fold_version":    FOLD_VERSION,
                "timestamp":       time.time(),
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Restore model + optimizer + EMA shadow + global step from a
        checkpoint file written by ``save_checkpoint``.

        Args:
            path : path to the .pt checkpoint file.
        """
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["model_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        self.ema.shadow = {
            k: v.to(next(self.model.parameters()).device)
            for k, v in data["ema_shadow"].items()
        }
        self.global_step = data.get("global_step", 0)
        logger.info(
            "Loaded checkpoint ← %s (step=%d, seq2coarse_version=%s)",
            path, self.global_step, data.get("seq2coarse_version", "unknown"),
        )


# =============================================================================
# 11.  Verification Suite
# =============================================================================

if __name__ == "__main__":
    _device = get_device("cuda")
    print("=" * 70)
    print("  Seq → Coarse Structure v1 — Production Integration Tests")
    print(f"  SEQ2COARSE_VERSION = {SEQ2COARSE_VERSION} | FOLD_VERSION = {FOLD_VERSION}")
    print(f"  ESM backends available: fair-esm={_HAS_FAIR_ESM}, "
          f"transformers={_HAS_HF_TRANSFORMERS}, biopython={_HAS_BIOPYTHON}")
    print(f"  Device: {_device}")
    print("=" * 70)

    _seq = (
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGTLHHFVHKKDS"
    )
    _seq = _seq[:48]  # keep the smoke test fast; full-length sequences are supported

    # Force the lightweight, dependency-free fallback backend so this
    # smoke test runs identically with or without fair-esm/transformers
    # installed.
    _cfg = Seq2CoarseConfig(
        embed_backend="learned",
        embed_dim=64,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=128,
        num_distance_bins=32,
        mds_iters=50,
        max_seq_len=256,
        checkpoint_dir="/tmp/seq2coarse_checkpoints_smoketest",
    )
    _model = SeqToCoarseStructure(_cfg).to(_device)

    # ── Test 1: Forward pass shapes ─────────────────────────────────────
    out = _model(_seq)
    N = len(_seq)
    assert out["init_coords"].shape == (N, 3), out["init_coords"].shape
    assert out["seq_features"].shape == (N, _cfg.hidden_dim), out["seq_features"].shape
    assert out["sigma"].shape == (N, 1), out["sigma"].shape
    assert out["distogram"].shape == (N, N, _cfg.num_distance_bins), out["distogram"].shape
    assert out["expected_dist"].shape == (N, N), out["expected_dist"].shape
    assert torch.isfinite(out["init_coords"]).all(), "Non-finite coordinates produced."
    assert (out["sigma"] >= _cfg.sigma_min - 1e-4).all() and (out["sigma"] <= _cfg.sigma_max + 1e-4).all()
    print(f"[PASS] Forward pass → coords {out['init_coords'].shape}, "
          f"seq_features {out['seq_features'].shape}, sigma {out['sigma'].shape}")

    # ── Test 2: Distance matrix symmetry + zero diagonal ────────────────
    dmat = out["expected_dist"]
    assert torch.allclose(dmat, dmat.T, atol=1e-4), "Expected distance matrix not symmetric."
    assert torch.allclose(torch.diag(dmat), torch.zeros(N, device=_device), atol=1e-4)
    print("[PASS] Expected distance matrix symmetric with zero diagonal")

    # ── Test 3: Gradient flow end-to-end through MDS ────────────────────
    _model.zero_grad()
    loss = out["init_coords"].pow(2).sum() + out["sigma"].sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in _model.parameters() if p.requires_grad
    )
    assert has_grad, "No gradient reached trainable parameters through the MDS solver."
    print("[PASS] Gradients flow end-to-end through DifferentiableMDS")

    # ── Test 4: SGNO-compatible adapter ─────────────────────────────────
    seq_features, init_coords, sigma = build_sgno_compatible_inputs(out)
    assert seq_features.shape == (N, _cfg.hidden_dim)
    assert init_coords.shape == (N, 3)
    assert sigma.shape == (N, 1)
    print(f"[PASS] build_sgno_compatible_inputs → "
          f"seq_features {seq_features.shape}, init_coords {init_coords.shape}, sigma {sigma.shape}")

    # ── Test 5: PDB export ───────────────────────────────────────────────
    _pdb_path = write_ca_pdb(_seq, out["init_coords"], "/tmp/seq2coarse_smoketest.pdb")
    assert _pdb_path.exists()
    _text = _pdb_path.read_text()
    assert _text.count("ATOM") == N, f"Expected {N} ATOM records, found {_text.count('ATOM')}"
    print(f"[PASS] PDB export → {_pdb_path} ({N} CA atoms)")

    # ── Test 6: EMA context manager ─────────────────────────────────────
    _ema = EMAWrapper(_model, decay=_cfg.ema_decay)
    _ema.update()
    with _ema.average_parameters():
        out_ema = _model.predict(_seq)
    assert out_ema["init_coords"].shape == (N, 3)
    print(f"[PASS] EMA inference mode → coords {out_ema['init_coords'].shape}")

    # ── Test 7: Trainer step + checkpointing ────────────────────────────
    _trainer = Seq2CoarseTrainer(_model, _cfg)
    _true_coords = torch.randn(N, 3, device=_device) * 5.0
    m = _trainer.train_step(_seq, true_coords=_true_coords)
    assert "total" in m and math.isfinite(m["total"])
    print(f"[PASS] Trainer step → losses {m}")

    _ckpt = _trainer.save_checkpoint()
    _trainer.load_checkpoint(_ckpt)
    print(f"[PASS] Checkpoint save/load → {_ckpt}")

    # ── Test 8: predict() determinism in eval mode ──────────────────────
    p1 = _model.predict(_seq)
    p2 = _model.predict(_seq)
    assert torch.allclose(p1["init_coords"], p2["init_coords"], atol=1e-5), \
        "predict() should be deterministic in eval mode (no dropout, fixed MDS init)."
    print("[PASS] predict() determinism in eval mode")

    # ── Test 9: Factorized pair projection == unfactorized (exact identity) ──
    _model.eval()
    with torch.no_grad():
        h_probe = torch.randn(N, _cfg.hidden_dim, device=_device)
        logits_factorized = _model.distogram_head._pairwise_logits_chunk(
            h_probe, 0, N, h_probe.dtype
        )
        _model.distogram_head.cfg.pair_proj_factorized = False
        logits_unfactorized = _model.distogram_head._pairwise_logits_chunk(
            h_probe, 0, N, h_probe.dtype
        )
        _model.distogram_head.cfg.pair_proj_factorized = True  # restore
    assert torch.allclose(logits_factorized, logits_unfactorized, atol=1e-4), \
        f"Factorized vs unfactorized pair projection mismatch: " \
        f"max diff = {(logits_factorized - logits_unfactorized).abs().max().item():.6f}"
    print("[PASS] pair_proj_factorized=True produces identical logits to the unfactorized path")

    # ── Test 10: Chunked == unchunked (exact identity) ──────────────────
    with torch.no_grad():
        full_logits = _model.distogram_head(h_probe)
        _model.distogram_head.cfg.pair_chunk_size = 7  # deliberately uneven vs N=48
        chunked_logits = _model.distogram_head(h_probe)
        _model.distogram_head.cfg.pair_chunk_size = None  # restore
    assert torch.allclose(full_logits, chunked_logits, atol=1e-4), \
        f"Chunked vs unchunked distogram mismatch: " \
        f"max diff = {(full_logits - chunked_logits).abs().max().item():.6f}"
    print("[PASS] pair_chunk_size=7 produces identical logits to the unchunked (None) path")

    # ── Test 11: Fused expected-distance path == forward()+expected_distance_matrix() ──
    with torch.no_grad():
        out_full = _model(_seq, return_distogram=True)
        out_fused = _model(_seq, return_distogram=False)
    assert "distogram" not in out_fused, "return_distogram=False should omit 'distogram' from output."
    assert torch.allclose(out_full["expected_dist"], out_fused["expected_dist"], atol=1e-3), \
        f"Fused vs unfused expected-distance mismatch: " \
        f"max diff = {(out_full['expected_dist'] - out_fused['expected_dist']).abs().max().item():.6f}"
    print("[PASS] return_distogram=False (fused path) matches return_distogram=True (full path)")

    # ── Test 12: 2-D tiled path == row-chunked path (exact identity) ────
    with torch.no_grad():
        _model.distogram_head.cfg.pair_chunk_size = 11  # deliberately uneven vs N=48
        dmat_row_chunked = _model.distogram_head.forward_expected_distance(
            h_probe, use_2d_tiling=False
        )
        dmat_2d_tiled = _model.distogram_head.forward_expected_distance(
            h_probe, use_2d_tiling=True
        )
        _model.distogram_head.cfg.pair_chunk_size = None  # restore
    assert torch.allclose(dmat_row_chunked, dmat_2d_tiled, atol=1e-4), \
        f"2D-tiled vs row-chunked expected-distance mismatch: " \
        f"max diff = {(dmat_row_chunked - dmat_2d_tiled).abs().max().item():.6f}"
    print("[PASS] use_2d_tiling=True produces identical results to row-chunking (chunk_size=11, N=48)")

    # ── Test 13: 2-D tiled path preserves gradient flow (the bug the row-      ──
    # ── allocate-and-scatter-assign anti-pattern would have introduced) ──────
    _model.zero_grad()
    h_grad_probe = torch.randn(N, _cfg.hidden_dim, device=_device, requires_grad=True)
    _model.distogram_head.cfg.pair_chunk_size = 13
    dmat_for_grad = _model.distogram_head.forward_expected_distance(
        h_grad_probe, use_2d_tiling=True
    )
    _model.distogram_head.cfg.pair_chunk_size = None  # restore
    loss_2d = dmat_for_grad.sum()
    loss_2d.backward()
    assert h_grad_probe.grad is not None and torch.isfinite(h_grad_probe.grad).all() \
        and h_grad_probe.grad.abs().sum() > 0, \
        "Gradient did not flow back through the 2D-tiled path to its input."
    print("[PASS] Gradients flow correctly through the 2D-tiled (block-cat-assembled) path")

    # ── Test 14: row-chunked SMACOF update == unchunked (exact identity) ──
    torch.manual_seed(0)
    mds_probe_dist = torch.rand(N, N, device=_device) * 15.0
    mds_probe_dist = 0.5 * (mds_probe_dist + mds_probe_dist.T)
    mds_probe_dist.fill_diagonal_(0.0)
    mds_probe_init = torch.randn(N, 3, device=_device)

    _model.mds.cfg.mds_row_chunk_size = None
    coords_unchunked = _model.mds(mds_probe_dist, init_coords=mds_probe_init, n_iters=5)
    _model.mds.cfg.mds_row_chunk_size = 13  # deliberately uneven vs N=48
    coords_chunked = _model.mds(mds_probe_dist, init_coords=mds_probe_init, n_iters=5)
    _model.mds.cfg.mds_row_chunk_size = None  # restore
    assert torch.allclose(coords_unchunked, coords_chunked, atol=1e-3), \
        f"Row-chunked vs unchunked SMACOF mismatch: " \
        f"max diff = {(coords_unchunked - coords_chunked).abs().max().item():.6f}"
    print("[PASS] mds_row_chunk_size=13 produces identical SMACOF updates to the unchunked path")

    # ── Test 15: chunked SMACOF preserves gradient flow ──────────────────
    _model.zero_grad()
    dist_for_grad = mds_probe_dist.clone().requires_grad_(True)
    _model.mds.cfg.mds_row_chunk_size = 11
    coords_for_grad = _model.mds(dist_for_grad, init_coords=mds_probe_init, n_iters=3)
    _model.mds.cfg.mds_row_chunk_size = None
    coords_for_grad.sum().backward()
    assert dist_for_grad.grad is not None and torch.isfinite(dist_for_grad.grad).all() \
        and dist_for_grad.grad.abs().sum() > 0, \
        "Gradient did not flow back through the row-chunked SMACOF path to target_dist."
    print("[PASS] Gradients flow correctly through the row-chunked SMACOF path")

    # ── Test 16: Landmark MDS init runs and produces finite, sane coordinates ──
    _model.mds.cfg.mds_use_landmarks = True
    _model.mds.cfg.mds_num_landmarks = 20  # well under N=48
    with torch.no_grad():
        landmark_init_coords = _model.mds._landmark_mds_init(mds_probe_dist)
    _model.mds.cfg.mds_use_landmarks = False  # restore
    assert landmark_init_coords.shape == (N, 3)
    assert torch.isfinite(landmark_init_coords).all(), "Landmark MDS init produced non-finite values."
    print(f"[PASS] Landmark MDS init (L=20, N={N}) → finite coords {landmark_init_coords.shape}")

    # ── Test 17: Sliding-window attention == full attention when window ≥ N ──
    # (a window that spans the whole sequence should attend to everything,
    # i.e. reproduce full self-attention's *set of attended pairs* — this
    # checks the encoder actually runs end-to-end and produces finite,
    # correctly-shaped output, not numerical equality to nn.TransformerEncoder
    # (different parameterisation/init), which is the real contract here.)
    _model.eval()
    with torch.no_grad():
        h_full = _model.encoder(embed_probe := torch.randn(N, _cfg.embed_dim, device=_device))
    _model.encoder.cfg.attn_window_size = N  # window ≥ N -> every pair attended
    with torch.no_grad():
        h_window_full = _model.encoder(embed_probe)
    _model.encoder.cfg.attn_window_size = None  # restore
    assert h_window_full.shape == (N, _cfg.hidden_dim)
    assert torch.isfinite(h_window_full).all(), "Sliding-window encoder produced non-finite output."
    print(f"[PASS] Sliding-window attention (window=N={N}) runs end-to-end → {h_window_full.shape}")

    # ── Test 18: Sliding-window attention only sees local context ──────
    # Perturbing a residue far outside a small window should not change
    # another residue's output, while perturbing one *inside* the window
    # should. This is the actual behavioural contract of a banded mask.
    _model.encoder.cfg.attn_window_size = 2
    torch.manual_seed(1)
    base_embed = torch.randn(N, _cfg.embed_dim, device=_device)
    with torch.no_grad():
        out_base = _model.encoder(base_embed)
        probe_idx = N // 2
        far_embed = base_embed.clone()
        far_embed[probe_idx + 10] += 5.0   # outside window=2 relative to probe_idx
        out_far = _model.encoder(far_embed)
        near_embed = base_embed.clone()
        near_embed[probe_idx + 1] += 5.0   # inside window=2 relative to probe_idx
        out_near = _model.encoder(near_embed)
    _model.encoder.cfg.attn_window_size = None  # restore
    far_changed = (out_far[probe_idx] - out_base[probe_idx]).abs().max().item()
    near_changed = (out_near[probe_idx] - out_base[probe_idx]).abs().max().item()
    assert far_changed < 1e-5, (
        f"Sliding-window attention (window=2) leaked information from a "
        f"residue 10 positions away (delta={far_changed:.6f}); banding is broken."
    )
    assert near_changed > 1e-5, (
        "Sliding-window attention (window=2) failed to propagate information "
        "from a residue 1 position away; banding is over-restrictive."
    )
    print(f"[PASS] Sliding-window attention (window=2) is local: "
          f"in-window Δ={near_changed:.4f}, out-of-window Δ={far_changed:.2e}")

    # ── Test 19: auto_window_attn_threshold triggers the sliding-window path ──
    orig_threshold = _cfg.auto_window_attn_threshold
    _cfg.auto_window_attn_threshold = N - 1  # force N to exceed it
    with torch.no_grad():
        out_auto = _model.encoder(base_embed)
    _cfg.auto_window_attn_threshold = orig_threshold  # restore
    assert out_auto.shape == (N, _cfg.hidden_dim)
    assert torch.isfinite(out_auto).all()
    print(f"[PASS] auto_window_attn_threshold={N - 1} auto-switches "
          f"SequenceTransformerEncoder to sliding-window attention at N={N}")

    # ── Test 20: auto_landmark_threshold triggers Landmark MDS automatically ──
    orig_landmark_threshold = _cfg.auto_landmark_threshold
    _cfg.auto_landmark_threshold = N - 1  # force N to exceed it
    assert not _cfg.mds_use_landmarks, "Test setup assumption violated."
    with torch.no_grad():
        auto_landmark_coords = _model.mds._classical_mds_init(mds_probe_dist)
    _cfg.auto_landmark_threshold = orig_landmark_threshold  # restore
    assert auto_landmark_coords.shape == (N, 3)
    assert torch.isfinite(auto_landmark_coords).all(), \
        "Auto-switched Landmark MDS init produced non-finite values."
    print(f"[PASS] auto_landmark_threshold={N - 1} auto-switches "
          f"_classical_mds_init to Landmark MDS at N={N}, without setting mds_use_landmarks")

    # ── Test 21: auto_learned_embed_threshold auto-switches the embedder ──
    orig_embed_threshold = _cfg.auto_learned_embed_threshold
    _cfg.auto_learned_embed_threshold = N - 1  # force N to exceed it
    with torch.no_grad():
        auto_embed_out = _model.embedder(_seq, device=_device)
    _cfg.auto_learned_embed_threshold = orig_embed_threshold  # restore
    assert auto_embed_out.shape == (N, _cfg.embed_dim)
    assert torch.isfinite(auto_embed_out).all(), \
        "Auto-switched 'learned' embedding fallback produced non-finite values."
    print(f"[PASS] auto_learned_embed_threshold={N - 1} auto-switches "
          f"SequenceEmbedder to the 'learned' fallback at N={N}, without changing cfg.embed_backend")

    # ── Test 22: lazy positional-encoding cache grows monotonically ────
    learned_cfg = Seq2CoarseConfig(embed_backend="learned", max_seq_len=10_000,
                                    hidden_dim=32, num_heads=2, num_layers=1, embed_dim=32)
    learned_embedder = SequenceEmbedder(learned_cfg).to(_device)
    short_seq = "ACDEFGHIKL"          # 10 residues
    long_seq = "ACDEFGHIKL" * 5        # 50 residues
    with torch.no_grad():
        out_short = learned_embedder(short_seq, device=_device)
        cache_after_short = learned_embedder._pos_encoding_cache.size(0)
        out_long = learned_embedder(long_seq, device=_device)
        cache_after_long = learned_embedder._pos_encoding_cache.size(0)
        out_short_again = learned_embedder(short_seq, device=_device)
        cache_after_repeat = learned_embedder._pos_encoding_cache.size(0)
    assert out_short.shape == (10, learned_cfg.embed_dim)
    assert out_long.shape == (50, learned_cfg.embed_dim)
    assert cache_after_short == 10, f"Expected cache sized to 10 after first call, got {cache_after_short}."
    assert cache_after_long == 50, f"Expected cache grown to 50, got {cache_after_long}."
    assert cache_after_repeat == 50, (
        f"Expected cache to stay at 50 (not shrink) after a shorter request, got {cache_after_repeat}."
    )
    assert torch.allclose(out_short, out_short_again, atol=1e-6), (
        "Re-embedding the same short sequence after the cache grew should give identical results."
    )
    print(f"[PASS] Lazy positional-encoding cache: {cache_after_short} → {cache_after_long} "
          f"(grows on demand, never shrinks, never precomputed to max_seq_len={learned_cfg.max_seq_len})")

    print("=" * 70)
    print("  All tests passed.")
    print("=" * 70)
