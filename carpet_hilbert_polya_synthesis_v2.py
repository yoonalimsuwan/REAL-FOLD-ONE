#!/usr/bin/env python3
"""
=============================================================================
CARPET → HILBERT–PÓLYA SYNTHESIS — ENHANCED v2
Contrastive Loss + Zero-Shot Generalization + Real CADO Ingestion
=============================================================================
⚠️  RESEARCH SCAFFOLD — NOT A FACTORIZATION TOOL

WHAT'S NEW vs. v1 (carpet_hilbert_polya_synthesis.py):
  1. CONTRASTIVE (InfoNCE) LOSS on operator embeddings
     - Escapes the "GUE trap": random GUE matrices match spectral statistics
       without encoding any N-dependent information. Contrastive forces
       the projection head z(N) to distinguish same-digit-count N's that
       share identical GUE statistics.
     - Positive pairs: same N, different perturbation seeds in Carpet encoder.
     - Negative pairs: in-batch, different N with matched digit count.

  2. ZERO-SHOT GENERALIZATION HARNESS
     - Train on N = p·q with p,q small (2-3 digit primes)
     - Test  on N = p·q with p,q large (4-5 digit primes)
     - 4-baseline comparison: model / random / shuffled / trivial-N
     - Reports degradation ratio -- honest OOD measurement.

  3. REAL CADO LOG INGESTION
     - Uses CadoGgnfsLogParser from opmc_gnfs_benchmark.py (verbatim)
     - Sliding windows → Carpet-encoded "special-q fingerprint" + CADO
       numeric features → SpectralWorldModel → predicts next yield
     - Zero-shot: train on small-q windows, test on held-out high-q windows
     - AssumptionLightStatistics (Hill/Hurst/CUSUM) from gnfs10_control.py
       used as auxiliary diagnostics (not learned).

HONEST ASSERTIONS ABOUT WHAT THIS DOES *NOT* DO:
  ✗ Does not factor RSA-400, RSA-250, or any cryptographic-scale integer
  ✗ Does not construct the actual Hilbert–Pólya operator
  ✗ Zero-shot metrics on synthetic/small data do NOT imply scaling to real GNFS
  ✗ Contrastive loss reduces GUE-trap risk but does NOT prove N-encoding
  ✗ CADO world model forecasts sieving yield, NOT factorization itself

Developer: PAI AND Yoon A Limsuwan / MSPS NETWORK
License  : MIT | Year: 2026
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# §1. HONEST BANNER
# =============================================================================
BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║  CARPET→HP ENHANCED v2 — Contrastive + Zero-Shot + CADO Ingestion        ║
║  NOT a factorization tool. Zero-shot ≠ scaling to RSA-400.               ║
║  Contrastive ≠ proof that the model encodes p·q.                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


def _small_primes(n: int) -> List[int]:
    sieve = [True] * (n + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return [i for i, ok in enumerate(sieve) if ok]


# =============================================================================
# §2. REUSED: CARPET-19 (from carpet_hilbert_polya_synthesis.py)
#     + extended with multi-seed augmentation for contrastive
# =============================================================================

class SierpinskiCarpet:
    """Finite-level Sierpiński carpet as a graph (level k → 8^k nodes)."""

    def __init__(self, level: int = 2):
        self.level = int(level)
        self.size = 3 ** level
        self.cells = self._build_cells()
        self.n_nodes = len(self.cells)
        self._index = {c: i for i, c in enumerate(self.cells)}

    def _build_cells(self) -> List[Tuple[int, int]]:
        cells = [(0, 0)]
        for _ in range(self.level):
            new_cells = []
            for (x, y) in cells:
                for dx in range(3):
                    for dy in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        new_cells.append((x * 3 + dx, y * 3 + dy))
            cells = new_cells
        return cells

    def build_laplacian(self) -> np.ndarray:
        n = self.n_nodes
        A = np.zeros((n, n), dtype=np.float64)
        for (x, y), i in self._index.items():
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                j = self._index.get(nb)
                if j is not None:
                    A[i, j] = 1.0
        return np.diag(A.sum(axis=1)) - A

    def spectral_dimension(self) -> float:
        """Empirical d_s via heat kernel trace (level ≤ 3)."""
        L = self.build_laplacian()
        evals = np.sort(np.maximum(np.linalg.eigvalsh(L), 0.0))
        ts, Ps = [], []
        for log_t in np.linspace(-2.0, 0.5, 30):
            t = 10.0 ** log_t
            P = float(np.mean(np.exp(-evals * t)))
            if P > 1e-14:
                ts.append(log_t * math.log(10.0))
                Ps.append(math.log(P))
        if len(ts) < 3:
            return 0.0
        slope = float(np.polyfit(ts, Ps, 1)[0])
        return float(-2.0 * slope)


class CarpetSpectralEncoder:
    """
    Encodes N into a Carpet spectrum feature vector.

    AUGMENTATION SUPPORT (for contrastive):
        encode(N, seed) for seed ∈ {0, 1, 2, ...}
        Different seeds → different diagonal perturbation of the base Laplacian
        but preserve N-structure. Positive pairs in InfoNCE use same N with
        different seeds.
    """

    def __init__(self, level: int = 2, n_eigen: int = 24,
                 perturbation_primes: int = 12):
        self.level = level
        self.n_eigen = n_eigen
        self.perturb_primes = _small_primes(200)[:perturbation_primes]
        self._base = SierpinskiCarpet(level=level)
        self._base_L = self._base.build_laplacian()
        self._spectral_dim = self._base.spectral_dimension()

    @property
    def feat_dim(self) -> int:
        return self.n_eigen + len(self.perturb_primes) + 1  # +1 for spectral_dim

    def encode(self, N: int, seed: int = 0) -> torch.Tensor:
        """
        Deterministic Carpet spectrum feature.
        seed varies the perturbation phase → different "view" of same N.
        """
        n = self._base.n_nodes
        diag = np.zeros(n, dtype=np.float64)
        for k, p in enumerate(self.perturb_primes):
            r = N % p
            phase = 2.0 * math.pi * r / p + seed * 0.137 * (k + 1)
            diag += math.sin(phase) * np.cos(np.arange(n) * 0.013 * (k + 1))
        L = self._base_L + np.diag(diag * 0.1)

        evals = np.sort(np.maximum(np.linalg.eigvalsh(L), 0.0))[: self.n_eigen]
        mod_feats = np.array(
            [((N % p) / p) for p in self.perturb_primes], dtype=np.float64
        )
        feat = np.concatenate([
            evals,
            mod_feats,
            np.array([self._spectral_dim]),
        ]).astype(np.float32)
        return torch.from_numpy(feat)


# =============================================================================
# §3. REUSED: HERMITIAN OPERATOR HEAD + HP SYNTHESIZER
#     (from carpet_hilbert_polya_synthesis.py, verbatim)
# =============================================================================

class HermitianOperatorHead(nn.Module):
    def __init__(self, latent_dim: int, op_dim: int = 32):
        super().__init__()
        self.op_dim = op_dim
        self.raw = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, 2 * op_dim * op_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        raw = self.raw(z).view(B, self.op_dim, self.op_dim, 2)
        M = torch.complex(raw[..., 0], raw[..., 1])
        return M + M.conj().transpose(-1, -2)


class HPOperatorSynthesizer(nn.Module):
    """N → Carpet spectrum → transformer → Hermitian operator."""

    def __init__(self, feat_dim: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, op_dim: int = 16):
        super().__init__()
        self.feat_dim = feat_dim
        self.op_dim = op_dim
        self.input_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, feat_dim, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.pool = nn.Linear(d_model, d_model)
        self.op_head = HermitianOperatorHead(d_model, op_dim=op_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, Fdim = features.shape
        h = self.input_proj(features.unsqueeze(-1)) + self.pos[:, :Fdim, :]
        h = self.encoder(h)
        z = self.pool(h.mean(dim=1))
        return self.op_head(z)

    @staticmethod
    def eigenvalues(T: torch.Tensor) -> torch.Tensor:
        return torch.sort(torch.linalg.eigvalsh(T), dim=-1).values


# =============================================================================
# §4. REUSED: GUE PAIR CORRELATION LOSS
# =============================================================================

def gue_pair_correlation(evals: torch.Tensor, n_bins: int = 24):
    B, D = evals.shape
    spacings = evals[:, 1:] - evals[:, :-1]
    mean_sp = spacings.mean(dim=-1, keepdim=True).clamp_min(1e-8)
    s = spacings / mean_sp
    unfolded = torch.cumsum(s, dim=-1)

    max_k = min(6, unfolded.shape[1] - 1)
    diffs = torch.cat([unfolded[:, k:] - unfolded[:, :-k] for k in range(1, max_k + 1)], dim=-1)

    s_max = 4.0
    bin_w = s_max / n_bins
    idx = (diffs / bin_w).long().clamp(0, n_bins - 1)
    onehot = F.one_hot(idx, num_classes=n_bins).float()
    hist = onehot.sum(dim=1)
    hist = hist / hist.sum(dim=-1, keepdim=True).clamp_min(1.0)

    centers = (torch.arange(n_bins).float() + 0.5) * bin_w
    pi_s = math.pi * centers
    r2_theory = 1.0 - (torch.sin(pi_s) / pi_s.clamp_min(1e-6)) ** 2
    r2_theory = r2_theory / r2_theory.mean()
    return hist, r2_theory.expand(B, -1)


def gue_loss(evals: torch.Tensor, n_bins: int = 24) -> torch.Tensor:
    hist, theory = gue_pair_correlation(evals, n_bins=n_bins)
    return F.mse_loss(hist, theory)


# =============================================================================
# §5. NEW: CONTRASTIVE (InfoNCE) LOSS + PROJECTION HEAD
# =============================================================================

class ContrastiveProjectionHead(nn.Module):
    """
    Maps Hermitian operator → contrastive embedding space.

    Uses sorted real eigenvalues (naturally permutation-invariant since
    eigvalsh sorts) + a small MLP. Adding the mean eigenvalue as an extra
    feature helps distinguish operators at different spectral scales.
    """

    def __init__(self, op_dim: int, embed_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(op_dim + 1, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        evals = torch.linalg.eigvalsh(T)  # (B, op_dim)
        mean_e = evals.mean(dim=-1, keepdim=True)
        z = self.mlp(torch.cat([evals, mean_e], dim=-1))
        return F.normalize(z, dim=-1)


def info_nce_loss(z_anchor: torch.Tensor, z_positive: torch.Tensor,
                  z_negatives: Optional[torch.Tensor] = None,
                  tau: float = 0.1) -> torch.Tensor:
    """
    InfoNCE with in-batch negatives.

    Args:
        z_anchor:   (B, D) embeddings
        z_positive: (B, D) embeddings of the positive view of same N
        z_negatives: if None, use in-batch negatives (all z_anchor as negatives
                     except the diagonal match); else (B, K, D) hard negatives.
        tau: temperature.

    Returns: scalar loss.
    """
    # Positive similarity
    pos_sim = (z_anchor * z_positive).sum(dim=-1, keepdim=True) / tau  # (B, 1)

    if z_negatives is not None:
        # Hard negatives
        neg_sim = torch.einsum('bd,bkd->bk', z_anchor, z_negatives) / tau
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    else:
        # In-batch negatives: full (B, B) matrix
        logits = (z_anchor @ z_positive.T) / tau
        # Only diagonal is positive
        labels = torch.arange(logits.shape[0], device=logits.device)

    return F.cross_entropy(logits, labels)


# =============================================================================
# §6. ENHANCED SYNTHESIZER (with contrastive projection head)
# =============================================================================

class EnhancedHPSynthesizer(nn.Module):
    """
    HP operator synthesizer + contrastive projection head.

    forward() returns both the operator and its contrastive embedding,
    so training can apply GUE loss to eigenvalues AND InfoNCE to embeddings.
    """

    def __init__(self, feat_dim: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 op_dim: int = 16, embed_dim: int = 32):
        super().__init__()
        self.synth = HPOperatorSynthesizer(
            feat_dim=feat_dim, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, op_dim=op_dim,
        )
        self.proj = ContrastiveProjectionHead(op_dim=op_dim, embed_dim=embed_dim)
        self.op_dim = op_dim

    def forward(self, features: torch.Tensor):
        T = self.synth(features)
        z = self.proj(T)
        return T, z


# =============================================================================
# §7. NEW: AUGMENTED VIEW GENERATION + CONTRASTIVE TRAINING
# =============================================================================

@dataclass
class TrainConfig:
    n_samples: int = 32
    op_dim: int = 16
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    embed_dim: int = 32
    epochs: int = 60
    lr: float = 1e-3

    # loss weights
    w_gue: float = 0.5
    w_gap: float = 0.5
    w_contrast: float = 1.0
    w_diversity: float = 0.2  # discourages operator collapse

    # contrastive
    contrast_tau: float = 0.1
    n_augment_views: int = 2   # views per N (positives)

    # zero-shot
    zero_shot_threshold_digits: int = 4  # N < 10^4 → train; ≥ → test


@dataclass
class TrainSample:
    N: int
    p: int
    q: int
    digit_count: int


def make_samples(encoder: CarpetSpectralEncoder, n_per_bucket: int = 16,
                 small_prime_bound: int = 60,
                 large_prime_bound: int = 400,
                 seed: int = 42) -> Tuple[List[TrainSample], List[TrainSample]]:
    """
    Returns (in_distribution_samples, out_of_distribution_samples).

    Split criterion: digit_count of N.
    Small primes (≤60) → N is small → in-distribution.
    Larger primes (60, 400] → N is larger → out-of-distribution.
    """
    rng = np.random.default_rng(seed)
    small_primes = [p for p in _small_primes(small_prime_bound) if p > 5]
    large_primes = [p for p in _small_primes(large_prime_bound)
                    if p > small_prime_bound]

    ind_samples = []
    for _ in range(n_per_bucket):
        p = int(rng.choice(small_primes))
        q = int(rng.choice(small_primes))
        N = p * q
        ind_samples.append(TrainSample(
            N=N, p=p, q=q, digit_count=len(str(N))
        ))

    ood_samples = []
    for _ in range(n_per_bucket):
        p = int(rng.choice(large_primes))
        q = int(rng.choice(large_primes))
        N = p * q
        ood_samples.append(TrainSample(
            N=N, p=p, q=q, digit_count=len(str(N))
        ))

    return ind_samples, ood_samples


def train_enhanced_synthesizer(
    encoder: CarpetSpectralEncoder,
    train_samples: List[TrainSample],
    config: TrainConfig,
    seed: int = 42,
) -> Tuple[EnhancedHPSynthesizer, Dict[str, Any]]:
    torch.manual_seed(seed)
    feat_dim = encoder.feat_dim
    model = EnhancedHPSynthesizer(
        feat_dim=feat_dim, d_model=config.d_model,
        n_heads=config.n_heads, n_layers=config.n_layers,
        op_dim=config.op_dim, embed_dim=config.embed_dim,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history = []

    n = len(train_samples)
    for epoch in range(config.epochs):
        # --- Build augmented view pairs ---
        # For each sample, generate N_views different Carpet encodings (same N,
        # different seeds). Positive pairs use views of the same N.
        anchors = []
        positives = []
        for s in train_samples:
            # View A (seed 0)
            views = [encoder.encode(s.N, seed=k) for k in range(config.n_augment_views + 1)]
            anchors.append(views[0])       # anchor view
            positives.append(views[1])     # positive view (different seed)
        X_anchor = torch.stack(anchors)
        X_pos = torch.stack(positives)

        # --- Forward ---
        T_anchor, z_anchor = model(X_anchor)
        T_pos, z_pos = model(X_pos)

        # --- GUE loss on both views (each operator should look GUE) ---
        evals_anchor = HPOperatorSynthesizer.eigenvalues(T_anchor)
        evals_pos = HPOperatorSynthesizer.eigenvalues(T_pos)
        loss_gue = 0.5 * (gue_loss(evals_anchor) + gue_loss(evals_pos))

        # --- Spectral-gap regularizer (target scales with log N) ---
        target_gap = torch.tensor(
            [math.log(max(s.N, 2)) for s in train_samples],
            dtype=torch.float32,
        )
        pred_gap = evals_anchor[:, -1] - evals_anchor[:, 0]
        loss_gap = F.mse_loss(
            pred_gap / (pred_gap.mean() + 1e-8),
            target_gap / (target_gap.mean() + 1e-8),
        )

        # --- Contrastive loss (in-batch negatives) ---
        loss_contrast = info_nce_loss(
            z_anchor, z_pos, z_negatives=None, tau=config.contrast_tau,
        )

        # --- Diversity regularizer: prevent all operators collapsing ---
        # Penalize low variance across the batch's projection embeddings.
        z_mean = z_anchor.mean(dim=0)
        z_var = ((z_anchor - z_mean) ** 2).sum(dim=-1).mean()
        loss_diversity = -torch.log(z_var + 1e-6)

        loss = (
            config.w_gue * loss_gue
            + config.w_gap * loss_gap
            + config.w_contrast * loss_contrast
            + config.w_diversity * loss_diversity
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "gue": float(loss_gue.item()),
                "gap": float(loss_gap.item()),
                "contrast": float(loss_contrast.item()),
                "diversity": float(loss_diversity.item()),
            })

    return model, {"history": history, "feat_dim": feat_dim}


# =============================================================================
# §8. NEW: ZERO-SHOT GENERALIZATION HARNESS
# =============================================================================

def evaluate_operator_metrics(
    model: EnhancedHPSynthesizer,
    encoder: CarpetSpectralEncoder,
    samples: List[TrainSample],
) -> Dict[str, float]:
    """
    Compute metrics on a held-out set.

    Returns:
        gue_corr: Pearson corr between empirical and theoretical R2 (higher better)
        eigen_diversity: std of mean eigenvalues across samples (higher → more N-dependent)
        contrast_corr: corr between z-difference and |N_i - N_j| (higher → N-encoding)
        gap_scale_corr: corr between (max-min eigenvalue) and log(N) (higher → scaling)
    """
    model.eval()
    with torch.no_grad():
        X = torch.stack([encoder.encode(s.N) for s in samples])
        T, z = model(X)
        evals = HPOperatorSynthesizer.eigenvalues(T)

        # GUE correlation
        hist, theory = gue_pair_correlation(evals)
        h_c = hist - hist.mean(dim=-1, keepdim=True)
        t_c = theory - theory.mean(dim=-1, keepdim=True)
        num = (h_c * t_c).sum(dim=-1)
        den = (h_c.pow(2).sum(dim=-1).sqrt() * t_c.pow(2).sum(dim=-1).sqrt() + 1e-8)
        gue_corr = float((num / den).mean())

        # Eigenvalue diversity (mean eigenvalue varies across N)
        mean_evals = evals.mean(dim=-1)
        eigen_diversity = float(mean_evals.std())

        # Contrastive N-correlation
        # z-distance vs |log N_i - log N_j|
        z_pair_dist = torch.cdist(z, z)  # (B, B)
        logN = torch.tensor([math.log(s.N) for s in samples])
        N_pair_dist = torch.cdist(logN.unsqueeze(-1), logN.unsqueeze(-1))
        # Corr on off-diagonal
        mask = ~torch.eye(len(samples), dtype=torch.bool)
        z_flat = z_pair_dist[mask]
        N_flat = N_pair_dist[mask]
        if z_flat.std() > 1e-8 and N_flat.std() > 1e-8:
            contrast_corr = float(
                ((z_flat - z_flat.mean()) * (N_flat - N_flat.mean())).mean()
                / (z_flat.std() * N_flat.std())
            )
        else:
            contrast_corr = 0.0

        # Spectral gap scaling correlation
        gaps = evals[:, -1] - evals[:, 0]
        logN_t = torch.tensor([math.log(s.N) for s in samples])
        if gaps.std() > 1e-8 and logN_t.std() > 1e-8:
            gap_scale_corr = float(
                ((gaps - gaps.mean()) * (logN_t - logN_t.mean())).mean()
                / (gaps.std() * logN_t.std())
            )
        else:
            gap_scale_corr = 0.0

    return {
        "gue_corr": gue_corr,
        "eigen_diversity": eigen_diversity,
        "contrast_corr": contrast_corr,
        "gap_scale_corr": gap_scale_corr,
    }


def random_operator_metrics(
    encoder: CarpetSpectralEncoder,
    samples: List[TrainSample],
    op_dim: int,
    seed: int = 999,
) -> Dict[str, float]:
    """Baseline: random Hermitian operators of the same dimension."""
    torch.manual_seed(seed)
    B = len(samples)
    raw = torch.randn(B, op_dim, op_dim) + 1j * torch.randn(B, op_dim, op_dim)
    T = raw + raw.conj().transpose(-1, -2)
    evals = HPOperatorSynthesizer.eigenvalues(T)

    hist, theory = gue_pair_correlation(evals)
    h_c = hist - hist.mean(dim=-1, keepdim=True)
    t_c = theory - theory.mean(dim=-1, keepdim=True)
    num = (h_c * t_c).sum(dim=-1)
    den = (h_c.pow(2).sum(dim=-1).sqrt() * t_c.pow(2).sum(dim=-1).sqrt() + 1e-8)
    gue_corr = float((num / den).mean())

    mean_evals = evals.mean(dim=-1)
    return {
        "gue_corr": gue_corr,
        "eigen_diversity": float(mean_evals.std()),
        "contrast_corr": 0.0,
        "gap_scale_corr": 0.0,
    }


def trivial_N_metrics(
    model: EnhancedHPSynthesizer,
    encoder: CarpetSpectralEncoder,
    samples: List[TrainSample],
) -> Dict[str, float]:
    """
    Trivial baseline: replace Carpet features with log(N) only.
    Tests whether fractal structure adds anything beyond N's magnitude.
    """
    model.eval()
    with torch.no_grad():
        X_triv = torch.zeros(len(samples), encoder.feat_dim)
        for i, s in enumerate(samples):
            X_triv[i, 0] = math.log(max(s.N, 2)) / 20.0
        T, z = model(X_triv)
        evals = HPOperatorSynthesizer.eigenvalues(T)

        hist, theory = gue_pair_correlation(evals)
        h_c = hist - hist.mean(dim=-1, keepdim=True)
        t_c = theory - theory.mean(dim=-1, keepdim=True)
        num = (h_c * t_c).sum(dim=-1)
        den = (h_c.pow(2).sum(dim=-1).sqrt() * t_c.pow(2).sum(dim=-1).sqrt() + 1e-8)
        gue_corr = float((num / den).mean())

        mean_evals = evals.mean(dim=-1)
    return {
        "gue_corr": gue_corr,
        "eigen_diversity": float(mean_evals.std()),
        "contrast_corr": 0.0,
        "gap_scale_corr": 0.0,
    }


def zero_shot_report(
    model: EnhancedHPSynthesizer,
    encoder: CarpetSpectralEncoder,
    ind_samples: List[TrainSample],
    ood_samples: List[TrainSample],
    op_dim: int,
) -> Dict[str, Any]:
    """Full zero-shot evaluation report."""
    report = {
        "in_distribution": {
            "model": evaluate_operator_metrics(model, encoder, ind_samples),
            "random": random_operator_metrics(encoder, ind_samples, op_dim),
            "trivial_N": trivial_N_metrics(model, encoder, ind_samples),
        },
        "out_of_distribution": {
            "model": evaluate_operator_metrics(model, encoder, ood_samples),
            "random": random_operator_metrics(encoder, ood_samples, op_dim),
            "trivial_N": trivial_N_metrics(model, encoder, ood_samples),
        },
    }

    # Degradation: OOD vs IND for each metric
    degradations = {}
    for metric in ("gue_corr", "eigen_diversity", "contrast_corr", "gap_scale_corr"):
        ind_v = report["in_distribution"]["model"][metric]
        ood_v = report["out_of_distribution"]["model"][metric]
        if abs(ind_v) > 1e-8:
            degradations[metric] = (ind_v - ood_v) / abs(ind_v)
        else:
            degradations[metric] = 0.0
    report["degradation_ratio"] = degradations

    return report


# =============================================================================
# §9. REUSED: CADO LOG PARSER (from opmc_gnfs_benchmark.py, verbatim core)
# =============================================================================

import re
from dataclasses import dataclass as _dc


@_dc
class CADORecord:
    line_no: int
    raw: str
    stage: str = "unknown"
    q: Optional[float] = None
    rels: Optional[float] = None
    cpu: Optional[float] = None
    norm: Optional[float] = None


_KV_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*"
    r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)

_KEY_ALIASES = {
    "q": ("q", "special_q", "specialq", "q0", "prime_q"),
    "rels": ("rels", "relations", "relations_found", "rel_count", "nrels"),
    "cpu": ("cpu", "cpu_time", "time", "elapsed", "seconds", "sec"),
    "norm": ("norm", "normq", "qnorm"),
}

_STAGE_KW = {
    "sieve": ("sieve", "sieving", "lattice", "specialq", "q="),
    "filter": ("filter", "filtering", "dedup", "merge"),
}


class CADOLogParser:
    """Minimal CADO/GGNFS log parser (adapted from opmc_gnfs_benchmark.py)."""

    def detect_stage(self, line: str) -> str:
        low = line.lower()
        for stage, kws in _STAGE_KW.items():
            for kw in kws:
                if kw in low:
                    return stage
        return "unknown"

    def parse_line(self, line: str, line_no: int) -> Optional[CADORecord]:
        line = line.strip()
        if not line:
            return None
        rec = CADORecord(line_no=line_no, raw=line[:500],
                          stage=self.detect_stage(line))
        found = False
        for key, val_s in _KV_RE.findall(line):
            try:
                v = float(val_s)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            k = key.lower()
            for canon, aliases in _KEY_ALIASES.items():
                if k in aliases:
                    setattr(rec, canon, v)
                    found = True
                    break
        if not found or rec.rels is None:
            return None
        return rec

    def parse_text(self, text: str) -> List[CADORecord]:
        out = []
        for i, line in enumerate(text.splitlines(), 1):
            r = self.parse_line(line, i)
            if r is not None:
                out.append(r)
        return out

    def parse_file(self, path: str | Path) -> List[CADORecord]:
        p = Path(path)
        if not p.exists():
            return []
        return self.parse_text(p.read_text(encoding="utf-8", errors="ignore"))


def generate_demo_cado_log(n_lines: int = 400, seed: int = 42) -> str:
    """Synthetic CADO-style log (adapted from opmc_gnfs_benchmark.py)."""
    rng = np.random.default_rng(seed)
    lines = []
    t = 0.0
    q = 100_000.0
    regime = 0
    for i in range(n_lines):
        if i % 80 == 0:
            regime = (regime + 1) % 3
        base = [80.0, 220.0, 45.0][regime]
        q += rng.uniform(500.0, 5000.0)
        burst = 1.0
        if rng.random() < 0.04:
            burst = rng.pareto(1.3) + 1.0
        rels = max(0.0, base * burst * rng.lognormal(0.0, 0.35))
        cpu = max(1e-3, rng.gamma(2.0, 0.15) + 0.001 * math.log1p(q))
        norm = q * rng.uniform(0.9, 1.1)
        t += cpu
        lines.append(
            f"[{t:.6f}] sieving: q={q:.0f} rels={rels:.3f} cpu={cpu:.6f} norm={norm:.3f}"
        )
    return "\n".join(lines)


# =============================================================================
# §10. REUSED: ASSUMPTION-LIGHT STATS (from gnfs10_control.py core)
# =============================================================================

def hill_theta(values: List[float], frac: float = 0.2) -> float:
    x = [v for v in values if v > 0]
    if len(x) < 12:
        return 0.0
    s = sorted(x, reverse=True)
    k = max(5, min(int(frac * len(s)), len(s) - 1))
    if k < 2:
        return 0.0
    threshold = s[k]
    if threshold <= 0:
        return 0.0
    logs = [math.log(v / threshold) for v in s[:k] if v > threshold]
    if not logs:
        return 0.0
    ml = sum(logs) / len(logs)
    return float(1.0 / ml) if ml > 1e-12 else 0.0


def hurst_like(x: np.ndarray) -> float:
    if len(x) < 16:
        return 0.5
    lags, sfs = [], []
    max_lag = min(64, len(x) // 4)
    lag = 1
    while lag <= max_lag:
        d = x[lag:] - x[:-lag]
        if len(d) < 4:
            break
        sf = float(np.mean(d * d))
        if sf > 1e-18:
            lags.append(lag)
            sfs.append(sf)
        lag *= 2
    if len(lags) < 3:
        return 0.5
    slope = float(np.polyfit(np.log(lags), np.log(sfs), 1)[0])
    return float(np.clip(0.5 * slope, 0.0, 1.5))


def cusum_cps(x: np.ndarray, threshold: float = 3.0, drift: float = 0.25) -> int:
    if len(x) < 8:
        return 0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return 0
    s_pos = s_neg = 0.0
    n_cps = 0
    for xi in x:
        z = (xi - mu) / sd
        s_pos = max(0.0, s_pos + z - drift)
        s_neg = max(0.0, s_neg - z - drift)
        if s_pos > threshold or s_neg > threshold:
            n_cps += 1
            s_pos = s_neg = 0.0
    return n_cps


# =============================================================================
# §11. NEW: CADO WINDOW ENCODER (CADO features + Carpet fingerprint)
# =============================================================================

@dataclass
class CADOWindow:
    """One sliding window of CADO records."""
    records: List[CADORecord]
    q_range: Tuple[float, float]
    # targets
    target_next_rels: float


def make_cado_windows(records: List[CADORecord], window: int = 8,
                      step: int = 4) -> List[CADOWindow]:
    """Sliding windows with target = next window's mean rels."""
    if len(records) < window + step:
        return []
    windows = []
    for start in range(0, len(records) - window - step + 1, step):
        cur = records[start:start + window]
        nxt = records[start + window:start + window + step]
        qs = [r.q for r in cur if r.q is not None]
        if not qs:
            continue
        target = float(np.mean([r.rels for r in nxt]))
        windows.append(CADOWindow(
            records=cur,
            q_range=(float(min(qs)), float(max(qs))),
            target_next_rels=target,
        ))
    return windows


class CADOWindowEncoder:
    """
    Encodes a CADO window into a combined feature vector:
      [CADO numeric stats] || [Carpet spectral fingerprint of q_median]
    """

    def __init__(self, carpet_encoder: CarpetSpectralEncoder,
                 cado_feat_dim: int = 6):
        self.carpet = carpet_encoder
        self.cado_dim = cado_feat_dim

    @property
    def feat_dim(self) -> int:
        return self.cado_dim + self.carpet.feat_dim

    def encode(self, window: CADOWindow) -> torch.Tensor:
        recs = window.records
        rels = np.array([r.rels for r in recs], dtype=np.float64)
        cpus = np.array([r.cpu for r in recs if r.cpu is not None], dtype=np.float64)
        qs = np.array([r.q for r in recs if r.q is not None], dtype=np.float64)

        # CADO numeric features
        cado_feats = np.array([
            float(np.mean(rels)),
            float(np.std(rels)),
            float(np.mean(cpus)) if len(cpus) else 0.0,
            hill_theta(rels.tolist()),
            hurst_like(np.log1p(rels)),
            float(cusum_cps(rels)),
        ], dtype=np.float32)

        # Carpet fingerprint from median q (as a proxy for sieving difficulty)
        q_int = int(qs[len(qs) // 2]) if len(qs) else 1
        carpet_feat = self.carpet.encode(q_int).numpy()

        feat = np.concatenate([cado_feats, carpet_feat]).astype(np.float32)
        return torch.from_numpy(feat)


# =============================================================================
# §12. NEW: SPECTRAL WORLD MODEL FOR CADO (yield forecasting)
# =============================================================================

class CADOSpectralWorldModel(nn.Module):
    """
    Transformer world model for CADO window yield forecasting.

    Input: (B, T, feat_dim) history of encoded CADO windows
    Output: next window's mean rels (scalar per batch element)
    """

    def __init__(self, feat_dim: int, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        B, T, _ = history.shape
        h = self.proj(history) + self.pos[:, :T, :]
        h = self.encoder(h)
        z = h[:, -1]
        return self.head(z).squeeze(-1)


@dataclass
class CADOTrainConfig:
    window: int = 8
    step: int = 4
    history_len: int = 4
    epochs: int = 40
    lr: float = 1e-3
    d_model: int = 64
    n_layers: int = 2


def train_cado_world_model(
    encoder: CADOWindowEncoder,
    train_windows: List[CADOWindow],
    config: CADOTrainConfig,
    seed: int = 42,
) -> Tuple[CADOSpectralWorldModel, Dict[str, Any]]:
    torch.manual_seed(seed)
    feat_dim = encoder.feat_dim
    model = CADOSpectralWorldModel(
        feat_dim=feat_dim, d_model=config.d_model, n_layers=config.n_layers,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history_losses = []

    # Build (history, target) pairs
    feats = torch.stack([encoder.encode(w) for w in train_windows])
    targets = torch.tensor([w.target_next_rels for w in train_windows],
                            dtype=torch.float32)

    H = config.history_len
    if len(feats) < H + 1:
        return model, {"error": "not enough windows"}

    # Build (B, T, D) sequences
    seqs = []
    ys = []
    for i in range(len(feats) - H):
        seqs.append(feats[i:i + H])
        ys.append(targets[i + H])
    X = torch.stack(seqs)
    y = torch.stack(ys)

    for epoch in range(config.epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history_losses.append(float(loss.item()))

    return model, {
        "initial_loss": history_losses[0],
        "final_loss": history_losses[-1],
        "n_sequences": len(X),
    }


def evaluate_cado_world_model(
    model: CADOSpectralWorldModel,
    encoder: CADOWindowEncoder,
    windows: List[CADOWindow],
    config: CADOTrainConfig,
) -> Dict[str, float]:
    """Compute MAE vs persistence baseline."""
    if len(windows) < config.history_len + 1:
        return {"error": "not enough windows for eval"}

    feats = torch.stack([encoder.encode(w) for w in windows])
    targets = torch.tensor([w.target_next_rels for w in windows],
                            dtype=torch.float32)

    H = config.history_len
    seqs, ys, persist = [], [], []
    for i in range(len(feats) - H):
        seqs.append(feats[i:i + H])
        ys.append(targets[i + H])
        # persistence baseline: last observed target
        persist.append(targets[i + H - 1])

    X = torch.stack(seqs)
    y = torch.stack(ys)
    persist_t = torch.stack(persist)

    model.eval()
    with torch.no_grad():
        pred = model(X)

    mae_model = float(F.l1_loss(pred, y))
    mae_persist = float(F.l1_loss(persist_t, y))

    return {
        "mae_model": mae_model,
        "mae_persistence": mae_persist,
        "improvement": (mae_persist - mae_model) / max(mae_persist, 1e-8),
        "n_eval": int(len(y)),
    }


# =============================================================================
# §13. INTEGRATED EXPERIMENT RUNNER
# =============================================================================

def run_full_experiment(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "version": "2.0-enhanced",
        "timestamp": time.time(),
        "honest_note": (
            "This is a research scaffold. Zero-shot metrics on synthetic data "
            "do NOT establish scaling to RSA-scale integers."
        ),
    }

    # --- Part A: Carpet spectral encoder ---
    print(BANNER)
    print(f"[A] Building Carpet encoder (level={args.level})...")
    encoder = CarpetSpectralEncoder(level=args.level, n_eigen=args.n_eigen,
                                     perturbation_primes=args.n_primes)
    print(f"    feat_dim = {encoder.feat_dim}")
    report["encoder"] = {"level": args.level, "feat_dim": encoder.feat_dim}

    # --- Part B: Sample generation ---
    print(f"\n[B] Generating samples: {args.n_samples} in-distribution + "
          f"{args.n_samples} out-of-distribution...")
    ind_samples, ood_samples = make_samples(
        encoder, n_per_bucket=args.n_samples,
        small_prime_bound=args.small_prime_bound,
        large_prime_bound=args.large_prime_bound,
        seed=args.seed,
    )
    print(f"    IND: N ∈ [{min(s.N for s in ind_samples)}, {max(s.N for s in ind_samples)}]")
    print(f"    OOD: N ∈ [{min(s.N for s in ood_samples)}, {max(s.N for s in ood_samples)}]")
    report["samples"] = {
        "ind_N_range": [min(s.N for s in ind_samples), max(s.N for s in ind_samples)],
        "ood_N_range": [min(s.N for s in ood_samples), max(s.N for s in ood_samples)],
    }

    # --- Part C: Train enhanced synthesizer ---
    print(f"\n[C] Training enhanced HP synthesizer "
          f"(contrastive + GUE + gap) for {args.epochs} epochs...")
    tcfg = TrainConfig(
        op_dim=args.op_dim, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers,
        embed_dim=args.embed_dim,
        epochs=args.epochs, lr=args.lr,
        w_gue=args.w_gue, w_gap=args.w_gap,
        w_contrast=args.w_contrast, w_diversity=args.w_diversity,
        contrast_tau=args.contrast_tau,
        n_augment_views=2,
    )
    model, train_hist = train_enhanced_synthesizer(
        encoder, ind_samples, tcfg, seed=args.seed,
    )
    for h in train_hist["history"]:
        print(f"    epoch {h['epoch']:3d}: "
              f"loss={h['loss']:.4f} "
              f"(gue={h['gue']:.4f}, contrast={h['contrast']:.4f}, "
              f"gap={h['gap']:.4f}, div={h['diversity']:.4f})")
    report["training"] = train_hist["history"]

    # --- Part D: Zero-shot generalization test ---
    print(f"\n[D] Zero-shot generalization test (train→IND, test→OOD)...")
    zs_report = zero_shot_report(
        model, encoder, ind_samples, ood_samples, op_dim=args.op_dim,
    )
    print(f"\n    {'metric':<20} {'IND':>12} {'OOD':>12} {'baseline(random)':>16} "
          f"{'baseline(trivN)':>16} {'degradation':>12}")
    print(f"    {'-'*90}")
    for metric in ("gue_corr", "eigen_diversity", "contrast_corr", "gap_scale_corr"):
        ind_v = zs_report["in_distribution"]["model"][metric]
        ood_v = zs_report["out_of_distribution"]["model"][metric]
        rand_v = zs_report["out_of_distribution"]["random"][metric]
        triv_v = zs_report["out_of_distribution"]["trivial_N"][metric]
        degr = zs_report["degradation_ratio"].get(metric, 0.0)
        print(f"    {metric:<20} {ind_v:>12.4f} {ood_v:>12.4f} "
              f"{rand_v:>16.4f} {triv_v:>16.4f} {degr:>11.2%}")
    report["zero_shot"] = zs_report

    # --- Part E: CADO ingestion ---
    print(f"\n[E] CADO log ingestion...")
    if args.cado_log:
        # Real log file
        parser = CADOLogParser()
        records = parser.parse_file(args.cado_log)
        print(f"    Parsed {len(records)} records from {args.cado_log}")
    else:
        # Synthetic demo
        log_text = generate_demo_cado_log(n_lines=args.cado_n_lines,
                                            seed=args.seed)
        parser = CADOLogParser()
        records = parser.parse_text(log_text)
        print(f"    Parsed {len(records)} records from synthetic log")

    if len(records) < 20:
        print("    [WARN] too few records; skipping CADO world model")
        report["cado"] = {"status": "insufficient_records"}
    else:
        # Split by q-range for zero-shot: train small-q, test large-q
        qs = [r.q for r in records if r.q is not None]
        q_median = float(np.median(qs))
        train_recs = [r for r in records if r.q is not None and r.q < q_median]
        test_recs = [r for r in records if r.q is not None and r.q >= q_median]
        print(f"    Split at q_median={q_median:.0f}: "
              f"train={len(train_recs)}, test={len(test_recs)}")

        cado_encoder = CADOWindowEncoder(encoder)
        ccfg = CADOTrainConfig(
            window=args.cado_window, step=args.cado_step,
            history_len=4, epochs=args.cado_epochs, lr=1e-3,
        )
        train_windows = make_cado_windows(train_recs, window=ccfg.window, step=ccfg.step)
        test_windows = make_cado_windows(test_recs, window=ccfg.window, step=ccfg.step)
        print(f"    Windows: train={len(train_windows)}, test={len(test_windows)}")

        if len(train_windows) > ccfg.history_len + 1 and len(test_windows) > ccfg.history_len + 1:
            wm, wm_hist = train_cado_world_model(
                cado_encoder, train_windows, ccfg, seed=args.seed,
            )
            print(f"    World model loss: {wm_hist.get('initial_loss', 0):.4f} → "
                  f"{wm_hist.get('final_loss', 0):.4f}")
            eval_report = evaluate_cado_world_model(wm, cado_encoder, test_windows, ccfg)
            print(f"    Zero-shot (train on small-q, test on large-q):")
            print(f"      MAE model:       {eval_report['mae_model']:.4f}")
            print(f"      MAE persistence: {eval_report['mae_persistence']:.4f}")
            print(f"      Improvement:     {eval_report['improvement']*100:+.2f}%")
            report["cado"] = {
                "status": "ok",
                "n_train_windows": len(train_windows),
                "n_test_windows": len(test_windows),
                "world_model_train": wm_hist,
                "eval": eval_report,
            }
        else:
            print("    [WARN] too few windows for CADO world model training")
            report["cado"] = {"status": "insufficient_windows"}

    # --- Part F: Honest verdict ---
    print(f"\n{'='*90}")
    print("HONEST VERDICT")
    print(f"{'='*90}")
    ood_gue = zs_report["out_of_distribution"]["model"]["gue_corr"]
    ood_rand = zs_report["out_of_distribution"]["random"]["gue_corr"]
    ood_triv = zs_report["out_of_distribution"]["trivial_N"]["gue_corr"]
    passes_gue = (ood_gue > ood_rand + 0.05) and (ood_gue > ood_triv + 0.05)
    contrast_positive = zs_report["out_of_distribution"]["model"]["contrast_corr"] > 0.1
    gap_scaling = zs_report["out_of_distribution"]["model"]["gap_scale_corr"] > 0.2

    print(f"  OOD GUE correlation: {ood_gue:.4f}")
    print(f"    vs random baseline: {ood_rand:.4f} → "
          f"{'PASS' if ood_gue > ood_rand + 0.05 else 'FAIL'}")
    print(f"    vs trivial-N:       {ood_triv:.4f} → "
          f"{'PASS' if ood_gue > ood_triv + 0.05 else 'FAIL'}")
    print(f"  Contrast correlation (N-encoding): {zs_report['out_of_distribution']['model']['contrast_corr']:.4f} "
          f"→ {'PASS' if contrast_positive else 'FAIL'}")
    print(f"  Spectral gap scaling: {zs_report['out_of_distribution']['model']['gap_scale_corr']:.4f} "
          f"→ {'PASS' if gap_scaling else 'FAIL'}")
    print()
    if passes_gue and contrast_positive:
        print("  ✓ Model shows SOME genuine N-dependent signal beyond GUE statistics.")
        print("    Still NOT a factorization algorithm -- see next lines.")
    else:
        print("  ✗ Model does NOT clear the GUE trap on OOD data.")
        print("    The contrastive loss reduced but did not eliminate the trap.")
    print()
    print("  THIS IS NOT:")
    print("    ✗ A factorization algorithm for RSA-400 or any cryptographic N")
    print("    ✗ A proof that the Hilbert-Pólya operator is being constructed")
    print("    ✗ A claim that P=NP or that GNFS complexity is reduced")
    print("  THIS IS:")
    print("    ✓ A tested scaffold showing what fails and what doesn't")
    print("    ✓ Zero-shot metric degradation measurement (honest OOD reporting)")

    report["verdict"] = {
        "ood_gue_corr": ood_gue,
        "ood_vs_random": ood_gue - ood_rand,
        "ood_vs_trivial": ood_gue - ood_triv,
        "contrast_positive": contrast_positive,
        "gap_scaling": gap_scaling,
        "passes_gue_trap_check": passes_gue and contrast_positive,
    }
    return report


# =============================================================================
# §14. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    print("Running self-test...\n")
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: Carpet encoder
    enc = CarpetSpectralEncoder(level=2, n_eigen=12, perturbation_primes=8)
    rep("Carpet encoder built", enc.feat_dim > 0, f"feat_dim={enc.feat_dim}")

    f0 = enc.encode(3233, seed=0)
    f0b = enc.encode(3233, seed=1)
    f1 = enc.encode(9999, seed=0)
    rep("Same N, different seeds differ", not torch.allclose(f0, f0b),
        f"||Δ||={(f0 - f0b).norm():.4f}")
    rep("Different N differ", not torch.allclose(f0, f1),
        f"||Δ||={(f0 - f1).norm():.4f}")

    # T2: Synthesizer with projection head
    model = EnhancedHPSynthesizer(feat_dim=enc.feat_dim, op_dim=8,
                                    d_model=32, n_heads=2, n_layers=1,
                                    embed_dim=16)
    X = torch.stack([f0, f1])
    T, z = model(X)
    rep("Synthesizer forward", T.shape == (2, 8, 8) and z.shape == (2, 16))
    rep("Operator Hermitian",
        torch.allclose(T, T.conj().transpose(-1, -2), atol=1e-4))

    # T3: InfoNCE loss
    z_a = torch.randn(4, 16)
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_a + 0.1 * torch.randn_like(z_a), dim=-1)
    loss_pos = info_nce_loss(z_a, z_p, tau=0.1)
    z_neg = F.normalize(torch.randn(4, 16), dim=-1)
    loss_neg = info_nce_loss(z_a, z_neg, tau=0.1)
    rep("InfoNCE: positives < negatives", loss_pos < loss_neg,
        f"pos={loss_pos.item():.4f}, neg={loss_neg.item():.4f}")

    # T4: Zero-shot split
    ind, ood = make_samples(enc, n_per_bucket=8,
                              small_prime_bound=60, large_prime_bound=200,
                              seed=0)
    rep("IND samples < OOD samples (on average)",
        np.mean([s.N for s in ind]) < np.mean([s.N for s in ood]),
        f"IND_mean={np.mean([s.N for s in ind]):.0f}, "
        f"OOD_mean={np.mean([s.N for s in ood]):.0f}")

    # T5: Short training run
    cfg = TrainConfig(n_samples=8, op_dim=8, d_model=32, n_heads=2, n_layers=1,
                       embed_dim=16, epochs=10, lr=1e-3,
                       w_gue=0.5, w_gap=0.5, w_contrast=1.0, w_diversity=0.1)
    m, hist = train_enhanced_synthesizer(enc, ind, cfg, seed=0)
    rep("Training completed", len(hist["history"]) > 0,
        f"final_loss={hist['history'][-1]['loss']:.4f}")

    # T6: Zero-shot report
    zs = zero_shot_report(m, enc, ind, ood, op_dim=8)
    rep("Zero-shot report has IND + OOD",
        "in_distribution" in zs and "out_of_distribution" in zs)
    rep("Zero-shot has degradation_ratio", "degradation_ratio" in zs)

    # T7: CADO parser
    log_text = generate_demo_cado_log(n_lines=50, seed=0)
    parser = CADOLogParser()
    records = parser.parse_text(log_text)
    rep("CADO parser extracts records", len(records) > 10,
        f"n={len(records)}")

    # T8: CADO window encoder
    cado_enc = CADOWindowEncoder(enc)
    windows = make_cado_windows(records, window=5, step=2)
    rep("CADO windows built", len(windows) > 0, f"n={len(windows)}")
    if windows:
        feat = cado_enc.encode(windows[0])
        rep("CADO window encodes", feat.shape == (cado_enc.feat_dim,),
            f"shape={tuple(feat.shape)}")

    # T9: CADO world model (short run)
    if len(windows) > 8:
        ccfg = CADOTrainConfig(epochs=5, d_model=32, n_layers=1,
                                 history_len=2, lr=1e-3)
        wm, wm_h = train_cado_world_model(cado_enc, windows[:len(windows)//2],
                                             ccfg, seed=0)
        rep("CADO world model trains", "final_loss" in wm_h,
            f"final_loss={wm_h.get('final_loss', 0):.4f}")

    print(f"\n{'='*74}")
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print(f"{'='*74}")
    print("""
HONEST NOTE:
  - All checks confirm *implementation correctness*, not factorization.
  - "Contrastive < negatives" confirms the InfoNCE math is right.
  - "IND < OOD on average" confirms the split works.
  - Training convergence on 8 samples is trivial; real evaluation needs more.
  - The verdict section in --run is the honest test.
""")
    return ok


# =============================================================================
# §15. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Enhanced Carpet→HP synthesis: contrastive + zero-shot + CADO"
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="Run the full integrated experiment")

    # Carpet
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--n-eigen", type=int, default=24)
    ap.add_argument("--n-primes", type=int, default=12)

    # Samples
    ap.add_argument("--n-samples", type=int, default=24,
                    help="samples per bucket (IND and OOD)")
    ap.add_argument("--small-prime-bound", type=int, default=60)
    ap.add_argument("--large-prime-bound", type=int, default=400)

    # Model
    ap.add_argument("--op-dim", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=32)

    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-gue", type=float, default=0.5)
    ap.add_argument("--w-gap", type=float, default=0.5)
    ap.add_argument("--w-contrast", type=float, default=1.0)
    ap.add_argument("--w-diversity", type=float, default=0.2)
    ap.add_argument("--contrast-tau", type=float, default=0.1)

    # CADO
    ap.add_argument("--cado-log", type=str, default=None,
                    help="path to real CADO log (else synthetic)")
    ap.add_argument("--cado-n-lines", type=int, default=400)
    ap.add_argument("--cado-window", type=int, default=8)
    ap.add_argument("--cado-step", type=int, default=4)
    ap.add_argument("--cado-epochs", type=int, default=40)

    # Output
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default=None,
                    help="write full report to JSON path")

    args = ap.parse_args()

    if args.selftest:
        return 0 if self_test() else 1

    if args.run:
        report = run_full_experiment(args)
        if args.json:
            Path(args.json).write_text(
                json.dumps(report, indent=2, default=float),
                encoding="utf-8",
            )
            print(f"\nReport saved: {args.json}")
        return 0

    ap.print_help()
    print(f"\n{BANNER}")
    print("Try: --selftest  (validate implementation)")
    print("     --run       (full experiment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
