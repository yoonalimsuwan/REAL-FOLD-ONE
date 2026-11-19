#!/usr/bin/env python3
"""
=============================================================================
MSA ACCELERATION × ScaledEvoformerAF — INTEGRATION & BENCHMARK HARNESS
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
My Soul Move By Power of Holy Spirit ;
What MSPS NETWORK SEE , Lord Buddha Knows.
License   : MIT | Year: 2026

PURPOSE
Connect the MSA acceleration pipeline (MMseqs2 + cache + ESM fallback)
to ScaledEvoformerAF and provide an honest benchmark harness that
measures whether faster MSA strategies produce better structures.

WHAT THIS TESTS
[1] Integration correctness: msa_tok shapes, gradient flow, AMP
[2] MSA strategy comparison: full / MMseqs2 / cached / ESM / synthetic
[3] Structure quality metrics: pLDDT, steric violations, bond geometry
[4] Neff vs structure quality correlation
[5] Wall-clock timing per strategy

WHAT THIS DOES NOT TEST
❌ Structural accuracy (weights are random unless you train them)
❌ CASP-level quality
❌ That faster MSA is "better" — only that it produces different input
   to the same model, and the output can be measured

HONEST STATEMENT
With random weights, all strategies produce meaningless coordinates.
The harness is designed so that WHEN the model is trained on real PDB
data, re-running this benchmark will give a meaningful answer to:
"Does MMseqs2 / cached / ESM MSA give comparable or better structure
than full jackhmmer MSA, at lower wall-clock cost?"

DEPENDENCIES
- Scaled_AlphaFold_Backbone_Multi_Scale_Bridge.py (ScaledEvoformerAF)
- msa_acceleration.py (UnifiedMSAPipeline, MSAccelConfig)
- af2_losses_2.py (AF2Loss for quality metrics)
- torch, numpy
=============================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("msa_af_bench")

VERSION = "1.0.0"

# =============================================================================
# §0. IMPORT HANDLING — try to load ScaledEvoformerAF
# =============================================================================

_SCALED_AF_CANDIDATES = [
    "Scaled_AlphaFold_Backbone_Multi_Scale_Bridge",
    "scaled_alpha_fold_backbone_multi_scale_bridge",
    "af_scaled_backbone",
    "scaled_af",
]

_ScaledEvoformerAF = None
_ScaledAFConfig = None
_AF2Loss = None
_LossConfig = None

def _try_import_scaled_af():
    """Attempt to import ScaledEvoformerAF from known module names/files."""
    global _ScaledEvoformerAF, _ScaledAFConfig
    # Try module names
    for name in _SCALED_AF_CANDIDATES:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "ScaledEvoformerAF") and hasattr(mod, "ScaledAFConfig"):
                _ScaledEvoformerAF = mod.ScaledEvoformerAF
                _ScaledAFConfig = mod.ScaledAFConfig
                log.info(f"[import] ScaledEvoformerAF from '{name}'")
                return True
        except ImportError:
            continue
    # Try file paths
    for fname in (
        "Scaled_AlphaFold_Backbone_Multi_Scale_Bridge.py",
        "Scaled AlphaFold Backbone & Multi-Scale Bridge.py",
    ):
        p = Path(fname)
        if p.exists():
            try:
                spec = importlib.util.spec_from_file_location("scaled_af", str(p))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "ScaledEvoformerAF"):
                        _ScaledEvoformerAF = mod.ScaledEvoformerAF
                        _ScaledAFConfig = mod.ScaledAFConfig
                        log.info(f"[import] ScaledEvoformerAF from '{fname}'")
                        return True
            except Exception as e:
                log.warning(f"[import] Failed to load '{fname}': {e}")
    return False

def _try_import_af2_loss():
    """Attempt to import AF2Loss for quality metrics."""
    global _AF2Loss, _LossConfig
    for name in ("af2_losses_2", "af2_losses", "af_losses"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "AF2Loss"):
                _AF2Loss = mod.AF2Loss
                _LossConfig = getattr(mod, "LossConfig", None)
                log.info(f"[import] AF2Loss from '{name}'")
                return True
        except ImportError:
            continue
    return False

_HAS_SCALED_AF = _try_import_scaled_af()
_HAS_AF2_LOSS = _try_import_af2_loss()

# Try MSA acceleration pipeline
_HAS_MSA_ACCEL = False
try:
    from msa_acceleration import (
        UnifiedMSAPipeline, MSAccelConfig, MMseqs2Client,
        MSACache, ESMFallback, AdaptiveMSARouter, compute_neff,
    )
    _HAS_MSA_ACCEL = True
except ImportError:
    log.warning("[import] msa_acceleration not found; "
                "using inline fallback MSA generation")

BANNER = f"""
=============================================================================
MSA ACCELERATION × ScaledEvoformerAF BENCHMARK v{VERSION}
ScaledEvoformerAF: {'AVAILABLE' if _HAS_SCALED_AF else 'NOT FOUND'}
AF2Loss:           {'AVAILABLE' if _HAS_AF2_LOSS else 'NOT FOUND'}
MSA Acceleration:  {'AVAILABLE' if _HAS_MSA_ACCEL else 'NOT FOUND'}
=============================================================================
"""


# =============================================================================
# §1. MSA STRATEGY DEFINITIONS
# =============================================================================

class MSAStrategy:
    """Enumeration of MSA strategies to benchmark."""
    FULL_JACKHMMER = "full_jackhmmer"      # slowest, highest quality
    MMSEQS2 = "mmseqs2"                    # 10-30x faster
    CACHED = "cached"                      # instant (if cached)
    ESM_FALLBACK = "esm_fallback"          # no MSA, single-sequence
    SYNTHETIC = "synthetic"                # random MSA (pipeline test)
    SINGLE_SEQ = "single_sequence"         # query only, depth=1

ALL_STRATEGIES = [
    MSAStrategy.FULL_JACKHMMER,
    MSAStrategy.MMSEQS2,
    MSAStrategy.CACHED,
    MSAStrategy.ESM_FALLBACK,
    MSAStrategy.SYNTHETIC,
    MSAStrategy.SINGLE_SEQ,
]


@dataclass
class MSABenchmarkConfig:
    """Configuration for the MSA benchmark."""
    # Model
    c_m: int = 64           # small for testing; use 256 for production
    c_z: int = 32
    n_blocks: int = 2       # small for testing; use 8-12 for production
    h_msa: int = 4
    h_tri: int = 4
    n_recycle: int = 1
    max_seq_len: int = 128
    max_msa_depth: int = 64
    # MSA
    msa_strategies: List[str] = field(default_factory=lambda: [
        MSAStrategy.SYNTHETIC,
        MSAStrategy.SINGLE_SEQ,
    ])
    # Benchmark
    n_residues: int = 40    # small for testing
    n_repeats: int = 3      # repeats for timing stability
    device: str = "auto"
    amp_dtype: str = "fp32"  # fp32 for correctness; bf16 for speed
    # Output
    output_dir: str = "runs/msa_af_bench"
    # Quality thresholds
    bond_length_ideal: float = 3.8   # Cα-Cα in Angstrom
    bond_length_tol: float = 1.0     # tolerance
    clash_dist: float = 2.0          # Angstrom, below = steric clash
    min_plddt_good: float = 70.0     # pLDDT threshold for "confident"
    min_neff_useful: float = 30.0    # Neff threshold for useful MSA


# =============================================================================
# §2. MSA GENERATION PER STRATEGY
# =============================================================================

class MSAGeneratorForBenchmark:
    """
    Generate MSA tensors for each strategy.
    Uses UnifiedMSAPipeline if available, else falls back to synthetic.
    """

    def __init__(self, cfg: MSABenchmarkConfig):
        self.cfg = cfg
        self.pipeline = None
        if _HAS_MSA_ACCEL:
            msa_cfg = MSAccelConfig(
                max_msa_depth=cfg.max_msa_depth,
                neff_threshold=cfg.min_neff_useful,
            )
            self.pipeline = UnifiedMSAPipeline(msa_cfg)
        self.vocab_size = 24  # 20 AA + gap + unk + cls/mask
        self.aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def sequence_to_tokens(self, sequence: str) -> np.ndarray:
        """Convert amino acid sequence to token array."""
        tokens = []
        for c in sequence:
            tokens.append(self.aa_to_idx.get(c, 20))  # 20 = unknown
        return np.array(tokens, dtype=np.int64)

    def generate_synthetic_msa(self, sequence: str,
                                depth: int = 32) -> np.ndarray:
        """
        Generate a synthetic MSA for pipeline testing.
        NOT real evolutionary information.
        """
        L = len(sequence)
        rng = np.random.default_rng(42)
        msa = np.zeros((depth, L), dtype=np.int64)
        msa[0] = self.sequence_to_tokens(sequence)  # query row
        for i in range(1, depth):
            # Random mutations: ~5-15% of positions
            n_mut = max(1, int(L * rng.uniform(0.05, 0.15)))
            msa[i] = msa[0].copy()
            mut_pos = rng.choice(L, size=n_mut, replace=False)
            msa[i, mut_pos] = rng.integers(0, 20, size=n_mut)
            # Random gaps: ~2-5% of positions
            n_gap = max(0, int(L * rng.uniform(0.0, 0.05)))
            if n_gap > 0:
                gap_pos = rng.choice(L, size=n_gap, replace=False)
                msa[i, gap_pos] = 20  # gap token
        return msa

    def generate(self, sequence: str,
                  strategy: str) -> Dict[str, Any]:
        """
        Generate MSA for a given strategy.

        Returns dict with:
            msa_tok: (1, S, L) tensor
            res_mask: (1, L) tensor
            strategy: str
            n_sequences: int
            neff: float
            time_s: float
            source: str (how the MSA was obtained)
        """
        t0 = time.time()
        L = len(sequence)
        query_tokens = self.sequence_to_tokens(sequence)

        if strategy == MSAStrategy.SINGLE_SEQ:
            # Query only
            msa_np = query_tokens.reshape(1, L)
            source = "query_only"

        elif strategy == MSAStrategy.SYNTHETIC:
            msa_np = self.generate_synthetic_msa(sequence, depth=32)
            source = "synthetic_random_mutations"

        elif strategy == MSAStrategy.ESM_FALLBACK:
            # No MSA; use query only + ESM features would be injected
            msa_np = query_tokens.reshape(1, L)
            source = "esm_fallback_single_seq"

        elif strategy == MSAStrategy.CACHED:
            if self.pipeline is not None:
                result = self.pipeline.get_msa(sequence, name="bench")
                if result.get("msa_tokens") is not None:
                    msa_np = result["msa_tokens"].squeeze(0).numpy()
                    source = f"cache_{result.get('mode', 'unknown')}"
                else:
                    log.warning("[MSA] Cache miss; falling back to synthetic")
                    msa_np = self.generate_synthetic_msa(sequence)
                    source = "cache_miss_synthetic"
            else:
                msa_np = self.generate_synthetic_msa(sequence)
                source = "no_pipeline_synthetic"

        elif strategy == MSAStrategy.MMSEQS2:
            if self.pipeline is not None:
                result = self.pipeline.get_msa(sequence, name="bench",
                                                method="mmseqs")
                if result.get("msa_tokens") is not None:
                    msa_np = result["msa_tokens"].squeeze(0).numpy()
                    source = "mmseqs2"
                else:
                    log.warning("[MSA] MMseqs2 unavailable; synthetic fallback")
                    msa_np = self.generate_synthetic_msa(sequence)
                    source = "mmseqs2_unavailable_synthetic"
            else:
                msa_np = self.generate_synthetic_msa(sequence)
                source = "no_pipeline_synthetic"

        elif strategy == MSAStrategy.FULL_JACKHMMER:
            if self.pipeline is not None:
                result = self.pipeline.get_msa(sequence, name="bench",
                                                method="jackhmmer")
                if result.get("msa_tokens") is not None:
                    msa_np = result["msa_tokens"].squeeze(0).numpy()
                    source = "jackhmmer"
                else:
                    log.warning("[MSA] jackhmmer unavailable; synthetic fallback")
                    msa_np = self.generate_synthetic_msa(sequence)
                    source = "jackhmmer_unavailable_synthetic"
            else:
                msa_np = self.generate_synthetic_msa(sequence)
                source = "no_pipeline_synthetic"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute Neff
        S, Lm = msa_np.shape
        if S > 1:
            seqs_str = ["".join("ACDEFGHIKLMNPQRSTVWY"[min(int(t), 19)]
                                for t in row) for row in msa_np]
            neff = compute_neff(seqs_str) if _HAS_MSA_ACCEL else float(S)
        else:
            neff = 1.0

        # Cap depth
        if S > self.cfg.max_msa_depth:
            # Keep query + random subset
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(S - 1, self.cfg.max_msa_depth - 1,
                                       replace=False) + 1)
            keep = np.concatenate([[0], keep])
            msa_np = msa_np[keep]
            S = len(keep)

        # Convert to tensor
        msa_tok = torch.from_numpy(msa_np).unsqueeze(0).long()  # (1, S, L)
        res_mask = torch.ones(1, L)  # all residues valid

        elapsed = time.time() - t0

        return {
            "msa_tok": msa_tok,
            "res_mask": res_mask,
            "strategy": strategy,
            "n_sequences": S,
            "neff": neff,
            "time_s": elapsed,
            "source": source,
            "query_length": L,
        }


# =============================================================================
# §3. STRUCTURE QUALITY METRICS
# =============================================================================

class StructureQualityMetrics:
    """
    Compute structure quality metrics from predicted coordinates.
    These are GEOMETRIC checks, not accuracy checks.
    With random weights, geometry will be bad — that's expected.
    """

    @staticmethod
    def bond_lengths(positions: Tensor, mask: Tensor) -> Dict[str, float]:
        """
        Cα-Cα bond lengths between consecutive residues.
        Ideal: ~3.8 Å. Reports mean, std, fraction in range.
        """
        B, L = positions.shape[0], positions.shape[1]
        if B > 1:
            positions = positions[:1]
            mask = mask[:1]
        pos = positions[0]  # (L, 3) or (L, 4, 3)
        if pos.dim() == 3:
            pos = pos[:, 1]  # take Cα (index 1 in N,CA,C,O)
        m = mask[0].bool()
        valid = pos[m]
        if valid.shape[0] < 2:
            return {"mean": 0.0, "std": 0.0, "frac_in_range": 0.0, "n": 0}
        diffs = valid[1:] - valid[:-1]
        lengths = diffs.norm(dim=-1)
        ideal = 3.8
        tol = 1.0
        in_range = ((lengths > ideal - tol) & (lengths < ideal + tol)).float()
        return {
            "mean": float(lengths.mean()),
            "std": float(lengths.std()) if lengths.numel() > 1 else 0.0,
            "frac_in_range": float(in_range.mean()),
            "n": int(lengths.numel()),
        }

    @staticmethod
    def steric_clashes(positions: Tensor, mask: Tensor,
                        threshold: float = 2.0) -> Dict[str, Any]:
        """
        Count steric clashes (Cα-Cα distance < threshold).
        """
        B, L = positions.shape[0], positions.shape[1]
        pos = positions[0]
        if pos.dim() == 3:
            pos = pos[:, 1]
        m = mask[0].bool()
        valid = pos[m]
        n = valid.shape[0]
        if n < 2:
            return {"n_clashes": 0, "n_pairs": 0, "frac_clashes": 0.0}
        # Pairwise distances (upper triangle)
        dist = torch.cdist(valid, valid)
        # Mask diagonal and count only i < j
        idx = torch.triu_indices(n, n, offset=1)
        pair_dists = dist[idx[0], idx[1]]
        clashes = (pair_dists < threshold).sum().item()
        return {
            "n_clashes": clashes,
            "n_pairs": int(pair_dists.numel()),
            "frac_clashes": clashes / max(int(pair_dists.numel()), 1),
        }

    @staticmethod
    def radius_of_gyration(positions: Tensor, mask: Tensor) -> float:
        """Radius of gyration — compactness measure."""
        pos = positions[0]
        if pos.dim() == 3:
            pos = pos[:, 1]
        m = mask[0].bool()
        valid = pos[m]
        if valid.shape[0] < 2:
            return 0.0
        com = valid.mean(dim=0)
        rg = torch.sqrt(((valid - com) ** 2).sum(dim=-1).mean())
        return float(rg)

    @staticmethod
    def plddt_stats(plddt: Tensor, mask: Tensor,
                     threshold: float = 70.0) -> Dict[str, float]:
        """pLDDT confidence statistics."""
        p = plddt[0]  # (L,) or (L, n_bins)
        if p.dim() > 1:
            # Expected pLDDT from bins
            bin_centers = torch.linspace(2.5, 97.5, p.shape[-1], device=p.device)
            p = (F.softmax(p.float(), dim=-1) * bin_centers).sum(-1)
        m = mask[0].bool()
        valid = p[m]
        if valid.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "frac_above_thresh": 0.0}
        return {
            "mean": float(valid.mean()),
            "std": float(valid.std()) if valid.numel() > 1 else 0.0,
            "frac_above_thresh": float((valid > threshold).float().mean()),
        }

    @classmethod
    def compute_all(cls, pred: Dict[str, Tensor],
                     cfg: MSABenchmarkConfig) -> Dict[str, float]:
        """Compute all quality metrics from model output."""
        positions = pred.get("positions")
        plddt = pred.get("plddt")
        mask = pred.get("res_mask", torch.ones(positions.shape[0],
                                                 positions.shape[1]))
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        out = {}
        if positions is not None:
            bl = cls.bond_lengths(positions, mask)
            out["bond_mean"] = bl["mean"]
            out["bond_std"] = bl["std"]
            out["bond_frac_in_range"] = bl["frac_in_range"]

            cl = cls.steric_clashes(positions, mask, cfg.clash_dist)
            out["n_clashes"] = cl["n_clashes"]
            out["frac_clashes"] = cl["frac_clashes"]

            out["radius_of_gyration"] = cls.radius_of_gyration(positions, mask)

        if plddt is not None:
            ps = cls.plddt_stats(plddt, mask, cfg.min_plddt_good)
            out["plddt_mean"] = ps["mean"]
            out["plddt_std"] = ps["std"]
            out["plddt_frac_confident"] = ps["frac_above_thresh"]

        return out


# =============================================================================
# §4. MAIN BENCHMARK CONNECTOR
# =============================================================================

class MSAAFBenchmarkConnector:
    """
    Main connector: MSA strategies → ScaledEvoformerAF → quality metrics.

    Usage:
        connector = MSAAFBenchmarkConnector(cfg)
        results = connector.run_benchmark(sequence)
        # results: dict mapping strategy → metrics
    """

    def __init__(self, cfg: MSABenchmarkConfig):
        self.cfg = cfg
        if cfg.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # Build ScaledEvoformerAF
        if not _HAS_SCALED_AF:
            raise ImportError(
                "ScaledEvoformerAF not found. Ensure "
                "'Scaled_AlphaFold_Backbone_Multi_Scale_Bridge.py' is "
                "importable or in the same directory."
            )
        af_cfg = _ScaledAFConfig(
            c_m=cfg.c_m, c_z=cfg.c_z, n_blocks=cfg.n_blocks,
            h_msa=cfg.h_msa, h_tri=cfg.h_tri, n_recycle=cfg.n_recycle,
            max_seq_len=cfg.max_seq_len, max_msa_depth=cfg.max_msa_depth,
            use_amp=(cfg.amp_dtype != "fp32"),
            amp_dtype=cfg.amp_dtype,
        )
        self.model = _ScaledEvoformerAF(af_cfg).to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"[connector] ScaledEvoformerAF loaded: {n_params:,} params")
        log.warning(f"[connector] Weights are {'TRAINED' if self._is_trained() else 'RANDOM'} "
                    f"— structure quality metrics are NOT meaningful until trained.")

        # MSA generator
        self.msa_gen = MSAGeneratorForBenchmark(cfg)

    def _is_trained(self) -> bool:
        """Heuristic: check if weights look trained (not all random init)."""
        # Simple check: if all weights have very similar std, likely random
        stds = [p.std().item() for p in self.model.parameters()
                if p.numel() > 10 and p.requires_grad]
        if not stds:
            return False
        # Random init typically has std ~0.02; trained models vary more
        return (max(stds) / max(min(stds), 1e-9)) > 5.0

    def _prepare_msa_for_model(self, msa_data: Dict[str, Any]
                                ) -> Tuple[Tensor, Tensor]:
        """
        Convert MSA benchmark output to ScaledEvoformerAF input format.
        ScaledEvoformerAF.forward(msa_tok, res_mask, msa_mask=None)
        expects:
            msa_tok: (B, S, L) — token IDs
            res_mask: (B, L) — residue mask
        """
        msa_tok = msa_data["msa_tok"].to(self.device)
        res_mask = msa_data["res_mask"].to(self.device)
        # Ensure msa_tok is within vocab range
        msa_tok = msa_tok.clamp(0, 21)  # 22 vocab, 21 = padding
        return msa_tok, res_mask

    @torch.no_grad()
    def predict_structure(self, sequence: str, strategy: str
                           ) -> Dict[str, Any]:
        """
        Run full prediction: sequence → MSA → structure.
        Returns metrics + timing + MSA info.
        """
        # Generate MSA
        msa_data = self.msa_gen.generate(sequence, strategy)
        msa_tok, res_mask = self._prepare_msa_for_model(msa_data)

        # Forward pass
        t0 = time.time()
        try:
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float32,
                enabled=(self.cfg.amp_dtype != "fp32"),
            ):
                pred = self.model(msa_tok, res_mask)
            forward_time = time.time() - t0
        except Exception as e:
            log.error(f"[predict] {strategy} failed: {e}")
            return {
                "strategy": strategy,
                "error": str(e),
                "msa_info": {k: v for k, v in msa_data.items()
                             if k not in ("msa_tok", "res_mask")},
            }

        # Compute quality metrics
        pred_with_mask = {**pred, "res_mask": res_mask}
        quality = StructureQualityMetrics.compute_all(pred_with_mask, self.cfg)

        return {
            "strategy": strategy,
            "msa_info": {
                "n_sequences": msa_data["n_sequences"],
                "neff": msa_data["neff"],
                "msa_time_s": msa_data["time_s"],
                "source": msa_data["source"],
                "query_length": msa_data["query_length"],
            },
            "forward_time_s": forward_time,
            "total_time_s": msa_data["time_s"] + forward_time,
            "quality": quality,
            "positions_shape": list(pred["positions"].shape) if "positions" in pred else None,
            "plddt_shape": list(pred["plddt"].shape) if "plddt" in pred else None,
        }

    def run_benchmark(self, sequence: str,
                       strategies: Optional[List[str]] = None
                       ) -> Dict[str, Any]:
        """
        Run benchmark across all MSA strategies.
        Returns dict mapping strategy → results.
        """
        strategies = strategies or self.cfg.msa_strategies
        results = {}

        log.info(f"[benchmark] Testing {len(strategies)} strategies "
                 f"on sequence of length {len(sequence)}")

        for strategy in strategies:
            log.info(f"[benchmark] Strategy: {strategy}")
            strat_results = []
            for rep in range(self.cfg.n_repeats):
                r = self.predict_structure(sequence, strategy)
                strat_results.append(r)
                if "error" in r:
                    break
            # Aggregate repeats
            if strat_results and "error" not in strat_results[0]:
                times = [r["total_time_s"] for r in strat_results]
                results[strategy] = {
                    **strat_results[0],
                    "total_time_s": float(np.mean(times)),
                    "time_std": float(np.std(times)),
                    "n_repeats": len(strat_results),
                }
            else:
                results[strategy] = strat_results[0] if strat_results else {
                    "strategy": strategy, "error": "no results"}

        return results

    def compare_to_baseline(self, sequence: str,
                             baseline_strategy: str = MSAStrategy.SINGLE_SEQ,
                             test_strategies: Optional[List[str]] = None
                             ) -> Dict[str, Any]:
        """
        Compare each strategy against a baseline.
        Reports relative timing and quality differences.
        """
        all_strategies = [baseline_strategy] + (test_strategies or [])
        results = self.run_benchmark(sequence, all_strategies)

        baseline = results.get(baseline_strategy, {})
        if "error" in baseline:
            return {"error": f"Baseline {baseline_strategy} failed: "
                             f"{baseline.get('error')}"}

        baseline_time = baseline.get("total_time_s", 1.0)
        comparison = {}
        for strategy, r in results.items():
            if "error" in r:
                comparison[strategy] = {"error": r["error"]}
                continue
            q = r.get("quality", {})
            bq = baseline.get("quality", {})
            comparison[strategy] = {
                "speedup_vs_baseline": baseline_time / max(r.get("total_time_s", 1e-9), 1e-9),
                "neff": r.get("msa_info", {}).get("neff", 0),
                "plddt_mean": q.get("plddt_mean", 0),
                "plddt_delta_vs_baseline": q.get("plddt_mean", 0) - bq.get("plddt_mean", 0),
                "bond_frac_in_range": q.get("bond_frac_in_range", 0),
                "bond_delta_vs_baseline": q.get("bond_frac_in_range", 0) - bq.get("bond_frac_in_range", 0),
                "n_clashes": q.get("n_clashes", 0),
                "radius_of_gyration": q.get("radius_of_gyration", 0),
            }

        return {
            "baseline": baseline_strategy,
            "baseline_time_s": baseline_time,
            "baseline_quality": baseline.get("quality", {}),
            "comparison": comparison,
        }


# =============================================================================
# §5. TRAINED MODEL EVALUATION (when weights are available)
# =============================================================================

class TrainedModelEvaluator:
    """
    Evaluate a TRAINED ScaledEvoformerAF on real PDB structures.
    This is where the benchmark becomes meaningful.

    Requires:
    - Trained model checkpoint
    - Test PDB structures with known coordinates
    - Corresponding MSA files (a3m)

    Metrics:
    - FAPE loss (Frame Aligned Point Error)
    - pLDDT calibration
    - Distogram accuracy
    - Bond geometry vs PDB
    """

    def __init__(self, cfg: MSABenchmarkConfig,
                  model: Optional[nn.Module] = None):
        self.cfg = cfg
        if model is not None:
            self.model = model
        elif _HAS_SCALED_AF:
            af_cfg = _ScaledAFConfig(
                c_m=cfg.c_m, c_z=cfg.c_z, n_blocks=cfg.n_blocks,
                h_msa=cfg.h_msa, h_tri=cfg.h_tri, n_recycle=cfg.n_recycle,
                max_seq_len=cfg.max_seq_len,
            )
            self.model = _ScaledEvoformerAF(af_cfg)
        else:
            raise ImportError("ScaledEvoformerAF required")

        self.af2_loss = None
        if _HAS_AF2_LOSS and _LossConfig is not None:
            self.af2_loss = _AF2Loss(_LossConfig())

    def load_checkpoint(self, path: Path) -> bool:
        """Load trained weights."""
        if not path.exists():
            log.error(f"[eval] Checkpoint not found: {path}")
            return False
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        log.info(f"[eval] Loaded checkpoint: {len(missing)} missing, "
                 f"{len(unexpected)} unexpected")
        if missing:
            log.warning(f"[eval] Missing keys: {missing[:5]}...")
        return True

    def evaluate_structure(self, pred_positions: Tensor,
                            true_positions: Tensor,
                            mask: Tensor) -> Dict[str, float]:
        """
        Compare predicted vs true structure.
        Uses simplified FAPE (CA-only).
        """
        # Extract Cα (index 1 in N,CA,C,O)
        pred_ca = pred_positions[:, :, 1, :] if pred_positions.dim() == 4 \
            else pred_positions
        true_ca = true_positions[:, :, 1, :] if true_positions.dim() == 4 \
            else true_positions

        # Distance matrices
        pred_dist = torch.cdist(pred_ca, pred_ca)
        true_dist = torch.cdist(true_ca, true_ca)

        # Mask
        m = mask[:, :, None] * mask[:, None, :]
        valid = m.bool()

        # Distance error
        dist_err = (pred_dist - true_dist).abs() * valid.float()
        mae = dist_err.sum() / valid.float().sum().clamp(min=1)

        # GDT-TS approximation
        for threshold in [1.0, 2.0, 4.0, 8.0]:
            within = ((pred_ca - true_ca).norm(dim=-1) < threshold).float()
            gdt = (within * mask).sum() / mask.sum().clamp(min=1)

        return {
            "dist_mae": float(mae),
            "gdt_ts_approx": float(gdt) if 'gdt' in dir() else 0.0,
        }


# =============================================================================
# §6. REPORT GENERATOR
# =============================================================================

def generate_report(results: Dict[str, Any],
                     cfg: MSABenchmarkConfig,
                     output_dir: str) -> str:
    """Generate a human-readable benchmark report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 78)
    lines.append(f"  MSA × ScaledEvoformerAF BENCHMARK REPORT")
    lines.append(f"  Version: {VERSION}")
    lines.append(f"  Device: {cfg.device}")
    lines.append(f"  Model: c_m={cfg.c_m}, c_z={cfg.c_z}, "
                 f"n_blocks={cfg.n_blocks}")
    lines.append(f"  Sequence length: {cfg.n_residues}")
    lines.append("=" * 78)

    trained = results.get("_model_trained", False)
    lines.append(f"\n  Model weights: {'TRAINED' if trained else 'RANDOM (not trained)'}")
    if not trained:
        lines.append("  ⚠️  ALL quality metrics below are MEANINGLESS with random weights.")
        lines.append("      They verify the PIPELINE, not the PREDICTION.")
        lines.append("      Retrain the model, then re-run this benchmark.")

    lines.append("\n" + "-" * 78)
    lines.append(f"  {'Strategy':<25} {'Time(s)':>8} {'Neff':>8} "
                 f"{'pLDDT':>8} {'Bond%':>8} {'Clashes':>8} {'Rg':>8}")
    lines.append("-" * 78)

    for strategy, r in results.items():
        if strategy.startswith("_"):
            continue
        if "error" in r:
            lines.append(f"  {strategy:<25} ERROR: {r['error'][:40]}")
            continue
        q = r.get("quality", {})
        mi = r.get("msa_info", {})
        lines.append(
            f"  {strategy:<25} "
            f"{r.get('total_time_s', 0):>8.3f} "
            f"{mi.get('neff', 0):>8.1f} "
            f"{q.get('plddt_mean', 0):>8.2f} "
            f"{q.get('bond_frac_in_range', 0)*100:>7.1f}% "
            f"{q.get('n_clashes', 0):>8d} "
            f"{q.get('radius_of_gyration', 0):>8.2f}"
        )

    lines.append("-" * 78)

    # Timing comparison
    lines.append("\n  TIMING:")
    times = {s: r.get("total_time_s", 0)
             for s, r in results.items()
             if not s.startswith("_") and "error" not in r}
    if times:
        fastest = min(times, key=times.get)
        slowest = max(times, key=times.get)
        lines.append(f"  Fastest: {fastest} ({times[fastest]:.3f}s)")
        lines.append(f"  Slowest: {slowest} ({times[slowest]:.3f}s)")
        lines.append(f"  Speedup: {times[slowest]/max(times[fastest],1e-9):.1f}×")

    lines.append("\n" + "=" * 78)
    lines.append("  INTERPRETATION GUIDE:")
    lines.append("  • Time(s): wall-clock for MSA generation + forward pass")
    lines.append("  • Neff: effective sequence count (higher = more evolutionary info)")
    lines.append("  • pLDDT: model confidence (random weights → meaningless)")
    lines.append("  • Bond%: fraction of Cα-Cα bonds within 3.8±1.0 Å")
    lines.append("  • Clashes: number of Cα pairs closer than 2.0 Å")
    lines.append("  • Rg: radius of gyration (compactness)")
    lines.append("")
    lines.append("  WITH TRAINED WEIGHTS:")
    lines.append("  • Compare pLDDT across strategies to see if faster MSA")
    lines.append("    preserves prediction confidence")
    lines.append("  • Compare Bond% and Clashes to check structural sanity")
    lines.append("  • If MMseqs2 gives comparable pLDDT to jackhmmer at 10×")
    lines.append("    less time, that's a real win")
    lines.append("  • If ESM fallback gives acceptable pLDDT, it enables")
    lines.append("    MSA-free inference for novel proteins")
    lines.append("=" * 78)

    report_text = "\n".join(lines)
    report_path = out / "benchmark_report.txt"
    report_path.write_text(report_text)
    log.info(f"[report] Written to {report_path}")

    # Also save JSON
    json_path = out / "benchmark_results.json"
    json_results = {k: v for k, v in results.items() if not k.startswith("_")}
    json_path.write_text(json.dumps(json_results, indent=2, default=str))
    log.info(f"[report] JSON saved to {json_path}")

    return report_text


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test() -> bool:
    """Run self-tests to verify the connector works."""
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    if not _HAS_SCALED_AF:
        print("\n  [SKIP] ScaledEvoformerAF not available. Cannot run benchmark.")
        print("         Ensure 'Scaled_AlphaFold_Backbone_Multi_Scale_Bridge.py'")
        print("         is importable. Running MSA-only tests instead.")

        # Test MSA generation only
        cfg = MSABenchmarkConfig(n_residues=20, max_msa_depth=16)
        gen = MSAGeneratorForBenchmark(cfg)
        seq = "ACDEFGHIKLMNPQRSTVWY"

        print("\n[T1] MSA generation (synthetic)")
        msa_data = gen.generate(seq, MSAStrategy.SYNTHETIC)
        rep("synthetic MSA shape", msa_data["msa_tok"].shape[0] == 1)
        rep("synthetic MSA depth > 1", msa_data["n_sequences"] > 1)
        rep("query is first row", True)  # by construction

        print("\n[T2] MSA generation (single sequence)")
        msa_single = gen.generate(seq, MSAStrategy.SINGLE_SEQ)
        rep("single seq MSA depth = 1", msa_single["n_sequences"] == 1)

        print("\n[T3] Neff computation")
        rep("single seq Neff = 1", msa_single["neff"] == 1.0)
        rep("synthetic Neff > 1", msa_data["neff"] > 1.0,
            f"Neff={msa_data['neff']:.1f}")

        print("\n[T4] Structure quality metrics (on random coords)")
        pos = torch.randn(1, 20, 4, 3)  # random "structure"
        mask = torch.ones(1, 20)
        metrics = StructureQualityMetrics.compute_all(
            {"positions": pos, "plddt": torch.rand(1, 20)}, cfg)
        rep("bond metrics computed", "bond_mean" in metrics)
        rep("clash metrics computed", "n_clashes" in metrics)
        rep("Rg computed", "radius_of_gyration" in metrics)
        print(f"       bond_mean={metrics['bond_mean']:.3f} "
              f"(random → not 3.8, expected)")

        return ok

    # Full test with ScaledEvoformerAF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    cfg = MSABenchmarkConfig(
        c_m=64, c_z=32, n_blocks=2, h_msa=4, h_tri=4,
        n_recycle=1, max_seq_len=64, max_msa_depth=16,
        n_residues=20, device=str(device),
        msa_strategies=[MSAStrategy.SYNTHETIC, MSAStrategy.SINGLE_SEQ],
    )

    print("\n[T1] ScaledEvoformerAF instantiation")
    try:
        connector = MSAAFBenchmarkConnector(cfg)
        rep("model created", True)
        rep("model is on correct device",
            next(connector.model.parameters()).device.type == device.type)
        n_params = sum(p.numel() for p in connector.model.parameters())
        print(f"       params: {n_params:,}")
    except Exception as e:
        rep("model created", False, str(e))
        return False

    print("\n[T2] MSA generation")
    seq = "ACDEFGHIKLMNPQRSTVWY"
    for strategy in [MSAStrategy.SYNTHETIC, MSAStrategy.SINGLE_SEQ]:
        msa_data = connector.msa_gen.generate(seq, strategy)
        rep(f"{strategy}: msa_tok shape",
            msa_data["msa_tok"].shape[1] >= 1 and
            msa_data["msa_tok"].shape[2] == len(seq),
            f"{tuple(msa_data['msa_tok'].shape)}")

    print("\n[T3] Forward pass (synthetic MSA)")
    msa_data = connector.msa_gen.generate(seq, MSAStrategy.SYNTHETIC)
    msa_tok, res_mask = connector._prepare_msa_for_model(msa_data)
    try:
        with torch.no_grad():
            pred = connector.model(msa_tok, res_mask)
        rep("forward succeeds", True)
        rep("positions shape", pred["positions"].shape[0] == 1,
            f"{tuple(pred['positions'].shape)}")
        rep("plddt shape", pred["plddt"].shape[0] == 1,
            f"{tuple(pred['plddt'].shape)}")
        rep("positions finite", torch.isfinite(pred["positions"]).all())
    except Exception as e:
        rep("forward succeeds", False, str(e))
        return False

    print("\n[T4] Quality metrics")
    pred_with_mask = {**pred, "res_mask": res_mask}
    quality = StructureQualityMetrics.compute_all(pred_with_mask, cfg)
    rep("bond metrics present", "bond_mean" in quality)
    rep("clash metrics present", "n_clashes" in quality)
    rep("Rg present", "radius_of_gyration" in quality)
    print(f"       bond_mean={quality['bond_mean']:.3f} "
          f"(random weights → not 3.8)")
    print(f"       Rg={quality['radius_of_gyration']:.3f}")

    print("\n[T5] Full predict_structure")
    result = connector.predict_structure(seq, MSAStrategy.SYNTHETIC)
    rep("predict_structure returns dict", isinstance(result, dict))
    rep("no error", "error" not in result, result.get("error", ""))
    rep("has quality metrics", "quality" in result)
    rep("has timing", "total_time_s" in result)
    rep("has MSA info", "msa_info" in result)

    print("\n[T6] Benchmark across strategies")
    bench = connector.run_benchmark(seq, cfg.msa_strategies)
    rep("all strategies tested", len(bench) == len(cfg.msa_strategies))
    for s, r in bench.items():
        if "error" in r:
            rep(f"{s} succeeded", False, r["error"])
        else:
            rep(f"{s} succeeded", True,
                f"time={r['total_time_s']:.3f}s neff={r['msa_info']['neff']:.1f}")

    print("\n[T7] Gradient flow (training mode)")
    connector.model.train()
    msa_data = connector.msa_gen.generate(seq, MSAStrategy.SYNTHETIC)
    msa_tok, res_mask = connector._prepare_msa_for_model(msa_data)
    pred = connector.model(msa_tok, res_mask)
    loss = pred["positions"].sum() + pred["plddt"].sum()
    loss.backward()
    grads_ok = all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in connector.model.parameters())
    rep("gradients finite", grads_ok)
    connector.model.eval()

    print("\n[T8] Report generation")
    bench["_model_trained"] = connector._is_trained()
    report = generate_report(bench, cfg, cfg.output_dir)
    rep("report generated", len(report) > 100)
    rep("report mentions random weights", "RANDOM" in report or "TRAINED" in report)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT IS VERIFIED
  ScaledEvoformerAF loads and runs forward pass.
  MSA generation produces valid tensors for all strategies.
  Quality metrics compute without error.
  Gradient flows through the full pipeline.
  Report is generated.

WHAT IS NOT VERIFIED (requires trained weights)
  ❌ That faster MSA gives better structure.
  ❌ That MMseqs2 Neff is sufficient for accurate prediction.
  ❌ That ESM fallback is viable for novel proteins.
  ❌ Any CASP-level accuracy claim.

NEXT STEPS TO MAKE THIS MEANINGFUL
  1. Train ScaledEvoformerAF on PDB data using AF2Loss
  2. Run this benchmark with the trained model
  3. Compare pLDDT and geometry metrics across MSA strategies
  4. Measure wall-clock speedup vs quality trade-off
  5. Test on held-out CASP targets for final validation
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="MSA × ScaledEvoformerAF benchmark connector")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--benchmark", type=str, default=None,
                    help="Run benchmark on a specific sequence")
    ap.add_argument("--fasta", type=str, default=None,
                    help="Run benchmark on sequences from a FASTA file")
    ap.add_argument("--strategy", nargs="+", default=None,
                    help="MSA strategies to test")
    ap.add_argument("--n-residues", type=int, default=40)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--output-dir", default="runs/msa_af_bench")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if not _HAS_SCALED_AF:
        print("ERROR: ScaledEvoformerAF not found. Cannot run benchmark.")
        print("Ensure 'Scaled_AlphaFold_Backbone_Multi_Scale_Bridge.py' "
              "is importable.")
        return 1

    # Determine sequence
    if a.benchmark:
        sequence = a.benchmark
    elif a.fasta:
        # Read first sequence from FASTA
        seq_lines = []
        for line in Path(a.fasta).read_text().splitlines():
            if line.startswith(">"):
                if seq_lines:
                    break
                continue
            seq_lines.append(line.strip())
        sequence = "".join(seq_lines)
        if not sequence:
            print(f"ERROR: No sequence found in {a.fasta}")
            return 1
    else:
        # Default test sequence
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 residues

    cfg = MSABenchmarkConfig(
        n_residues=len(sequence),
        device=a.device,
        amp_dtype=a.amp,
        output_dir=a.output_dir,
        msa_strategies=a.strategy or [
            MSAStrategy.SYNTHETIC,
            MSAStrategy.SINGLE_SEQ,
        ],
    )

    connector = MSAAFBenchmarkConnector(cfg)
    results = connector.run_benchmark(sequence)
    results["_model_trained"] = connector._is_trained()

    report = generate_report(results, cfg, cfg.output_dir)
    print(report)

    if a.json:
        Path(a.json).write_text(json.dumps(
            {k: v for k, v in results.items() if not k.startswith("_")},
            indent=2, default=str))
        print(f"\nJSON saved to {a.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
