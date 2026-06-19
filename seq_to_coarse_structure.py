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

    Distogram
    ---------
    num_distance_bins : number of discrete Cα–Cα distance bins.
    min_distance      : lower edge of the first bin (Å).
    max_distance      : upper edge of the last bin (Å).

    Differentiable MDS (coarse 3-D embedding)
    ------------------------------------------
    mds_iters       : number of SMACOF stress-majorization iterations.
    mds_dim         : output embedding dimension (3 for Cα coordinates).
    mds_eps         : numerical floor to avoid division by ~0 distances.
    mds_init_scale  : scale of the random initial 3-D embedding.

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
    max_seq_len:    int   = 2048

    # Transformer encoder
    hidden_dim:  int   = 256
    num_heads:   int   = 8
    num_layers:  int   = 6
    ffn_dim:     int   = 1024
    dropout:     float = 0.1

    # Distogram
    num_distance_bins: int   = 64
    min_distance:      float = 2.0
    max_distance:       float = 40.0

    # Differentiable MDS
    mds_iters:      int   = 200
    mds_dim:        int   = 3
    mds_eps:        float = 1e-6
    mds_init_scale: float = 5.0

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
        assert self.num_distance_bins >= 2
        assert self.max_distance > self.min_distance > 0.0
        assert self.mds_iters >= 1
        assert self.mds_dim >= 1
        assert self.mds_eps > 0.0
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
            self.register_buffer(
                "pos_encoding",
                self._build_sinusoidal_table(cfg.max_seq_len, cfg.embed_dim),
                persistent=False,
            )
            self.input_proj: Optional[nn.Module] = None
        else:
            self.aa_embed = None  # type: ignore[assignment]
            self.pos_encoding = None  # type: ignore[assignment]
            # Project frozen ESM-2 hidden width → cfg.embed_dim.
            native_dim = self._esm_native_dim or cfg.embed_dim
            self.input_proj = nn.Sequential(
                nn.Linear(native_dim, cfg.embed_dim),
                nn.LayerNorm(cfg.embed_dim),
            )

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
        x = x + self.pos_encoding[:n].to(device)         # add positional signal
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
# 3.  Sequence Transformer Encoder
# =============================================================================

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

    def forward(
        self,
        x: torch.Tensor,                          # (N, embed_dim)
        padding_mask: Optional[torch.Tensor] = None,  # (N,) bool, True = pad
    ) -> torch.Tensor:
        """
        Args:
            x            : (N, embed_dim) per-residue embedding (single sequence).
            padding_mask : optional (N,) bool mask; True marks padded positions.
        Returns:
            (N, hidden_dim) contextualised per-residue latent.
        """
        h = self.input_norm(x)
        h = self.in_proj(h).unsqueeze(0)           # (1, N, hidden_dim) — batch size 1

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

    Memory scaling note: this head materialises an (N, N, 2*hidden_dim)
    pair tensor, i.e. O(N²·d) memory — the same asymptotic cost as any
    pairwise distogram approach (AlphaFold-1, trRosetta included). For
    very long sequences (N ≳ 1500 on typical GPU memory budgets), chunk
    the (i, j) pairs and call ``forward`` per-chunk, or reduce
    ``hidden_dim`` / process on CPU for an initial coarse pass.

    Args:
        cfg : Seq2CoarseConfig instance.
    """

    def __init__(self, cfg: Seq2CoarseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
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

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (N, hidden_dim) per-residue latent.
        Returns:
            (N, N, num_distance_bins) symmetrised distance-bin logits.
        """
        n = h.size(0)
        h_i = h.unsqueeze(1).expand(n, n, -1)      # (N, N, d)
        h_j = h.unsqueeze(0).expand(n, n, -1)      # (N, N, d)
        pair = torch.cat([h_i, h_j], dim=-1)        # (N, N, 2d)
        logits = self.pair_proj(pair)               # (N, N, bins)
        logits = 0.5 * (logits + logits.transpose(0, 1))  # enforce symmetry
        return logits

    def expected_distance_matrix(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert distogram logits into an *expected* (soft) distance matrix
        by taking the probability-weighted mean over bin centers — fully
        differentiable, unlike argmax bin selection.

        Args:
            logits : (N, N, num_distance_bins).
        Returns:
            (N, N) expected Cα–Cα distance matrix, symmetric, zero diagonal.
        """
        probs = F.softmax(logits, dim=-1)            # (N, N, bins)
        dmat = torch.einsum("ijb,b->ij", probs, self.bin_centers)
        n = dmat.size(0)
        dmat = dmat * (1.0 - torch.eye(n, device=dmat.device, dtype=dmat.dtype))
        dmat = 0.5 * (dmat + dmat.transpose(0, 1))
        return dmat


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

        # Guarantee differentiability through the loop regardless of caller context.
        w_sum = weights.sum(dim=1, keepdim=True).clamp_min(self.cfg.mds_eps)  # (N, 1)

        for _ in range(iters):
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

            X = (B @ X) / w_sum

        return X

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

        Args:
            target_dist : (N, N) target distance matrix.
        Returns:
            (N, mds_dim) initial coordinate guess.
        """
        n = target_dist.size(0)
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
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequence    : raw single-letter amino-acid string (no MSA).
            init_coords : optional (N, 3) warm-start for the MDS solver
                          (e.g. coordinates from a previous training step,
                          a homology template, or an extended-chain build).
                          Random initialisation is used if None.
            mds_iters   : override cfg.mds_iters for this call.
        Returns:
            Dict with:
                "init_coords"   : (N, 3) coarse Cα coordinates.
                "seq_features"  : (N, hidden_dim) contextualised latent —
                                   feed directly as StructuralGNOFold's
                                   ``seq_features`` argument.
                "sigma"         : (N, 1) structural-regime field.
                "distogram"     : (N, N, num_distance_bins) raw logits.
                "expected_dist" : (N, N) expected distance matrix.
        """
        device = next(self.parameters()).device
        n = len(sequence)
        if n < 2:
            raise ValueError(f"Sequence must have at least 2 residues; got length {n}.")

        embed = self.embedder(sequence, device=device)            # (N, embed_dim)
        h = self.encoder(embed)                                    # (N, hidden_dim)

        distogram = self.distogram_head(h)                         # (N, N, bins)
        expected_dist = self.distogram_head.expected_distance_matrix(distogram)  # (N, N)
        sigma = self.sigma_head(h)                                  # (N, 1)

        coarse_coords = self.mds(
            expected_dist, init_coords=init_coords, n_iters=mds_iters
        )                                                            # (N, 3)

        return {
            "init_coords":   coarse_coords,
            "seq_features":  h,
            "sigma":         sigma,
            "distogram":     distogram,
            "expected_dist": expected_dist,
        }

    @torch.no_grad()
    def predict(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Inference-mode convenience wrapper (eval(), no_grad, detached)."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward(sequence)
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

        sigma_reg = (out["sigma"] - self.cfg.sigma_target).pow(2).mean()
        losses["sigma_reg"] = self.cfg.lambda_sigma * sigma_reg

        if not losses:
            raise ValueError(
                "train_step requires at least one of true_coords / true_dist "
                "to compute a supervised loss (sigma_reg alone is not sufficient)."
            )

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

    print("=" * 70)
    print("  All tests passed.")
    print("=" * 70)
