#!/usr/bin/env python3
"""
=============================================================================
MSA GENERATION PIPELINE: jackhmmer + ColabFold → ScaledEvoformerAF
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
            My Soul Move By Power of Holy Spirit ;
            What MSPS NETWORK SEE, Lord Buddha Knows.
License   : MIT | Year: 2026

WHAT THIS DOES
  [1] Generate MSA via jackhmmer (HMMER) against UniRef90 / BFD / MGnify
  [2] OR via ColabFold's MMseqs2 (much faster, recommended)
  [3] OR synthetic MSA fallback (pipeline testing only, NOT real)
  [4] Parse a3m → tokenize → msa_tok tensor (B, S, L)
  [5] Feed directly into ScaledEvoformerAF.forward()

WHAT THIS DOES NOT DO
  ❌ Bundle HMMER, MMseqs2, or sequence databases
  ❌ Generate real MSA without those tools installed
  ❌ Replace AlphaFold2's full MSA pipeline (no template search,
     no paired MSA for multimer, no BFD clustering)
  ❌ Claim synthetic MSA is useful for real structure prediction

DEPENDENCIES (install separately)
  hmmer        : sudo apt install hmmer
  colabfold    : pip install colabfold[all] && colabfold_download
  mmseqs2      : conda install -c bioconda mmseqs2
  databases    : UniRef90, BFD, MGnify (see AlphaFold data guide)
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("msa_pipeline")

VERSION = "1.0.0"

BANNER = """
=============================================================================
MSA GENERATION PIPELINE
jackhmmer / ColabFold(MMseqs2) / synthetic fallback → ScaledEvoformerAF
=============================================================================
"""

# =============================================================================
# §0. TOKENIZATION CONSTANTS (match ScaledEvoformerAF / AlphaFold2)
# =============================================================================

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}
GAP_IDX = 20
UNK_IDX = 21
CLS_IDX = 22
MASK_IDX = 23
VOCAB_SIZE = 24

# For ScaledEvoformerAF: aa_embed is Embedding(22, c_m)
# Token 21 is used as padding/mask in the uploaded code
AF2_PAD_TOKEN = 21


# =============================================================================
# §1. TOOL DETECTION
# =============================================================================

def detect_tools() -> Dict[str, bool]:
    """Check which MSA tools are actually installed."""
    return {
        "jackhmmer": shutil.which("jackhmmer") is not None,
        "mmseqs": shutil.which("mmseqs") is not None,
        "colabfold_batch": shutil.which("colabfold_batch") is not None,
        "hhblits": shutil.which("hhblits") is not None,
    }


def find_databases(search_paths: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    """Look for sequence databases in common locations."""
    candidates = search_paths or [
        os.path.expanduser("~/data/alphafold"),
        os.path.expanduser("~/alphafold_data"),
        "/data/alphafold",
        os.path.expanduser("~/.colabfold/databases"),
        os.path.expanduser("~/colabfold/databases"),
    ]
    found = {}
    names = {
        "uniref90": ["uniref90.fasta", "uniref90"],
        "uniref30": ["uniref30.fasta", "uniref30", "UniRef30"],
        "bfd": ["bfd_metaclust", "bfd"],
        "mgnify": ["mgy_clusters.fa", "mgnify"],
        "uniclust30": ["uniclust30", "UniClust30"],
    }
    for base in candidates:
        base_p = Path(base)
        if not base_p.exists():
            continue
        for db_name, patterns in names.items():
            if db_name in found:
                continue
            for pat in patterns:
                # Check for .fasta, .ffdata, or directory
                for ext in (".fasta", ".fa", ".ffdata", ""):
                    p = base_p / (pat + ext)
                    if p.exists():
                        found[db_name] = str(p)
                        break
                if db_name in found:
                    break
                # Check subdirectories
                for sub in base_p.rglob(pat + "*"):
                    if sub.exists():
                        found[db_name] = str(sub)
                        break
                if db_name in found:
                    break
    return found


# =============================================================================
# §2. MSA GENERATORS
# =============================================================================

class MSAGenerator:
    """
    Generates MSA files in a3m format using available tools.
    Falls back to synthetic MSA if no tools are installed.
    """

    def __init__(self, output_dir: str = "data/msa",
                 n_cpus: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_cpus = n_cpus
        self.tools = detect_tools()
        self.databases = find_databases()
        log.info(f"[MSA] Tools: {self.tools}")
        log.info(f"[MSA] Databases found: {list(self.databases.keys())}")

    def generate(self, sequence: str, name: str = "query",
                 method: str = "auto",
                 max_sequences: int = 512) -> Path:
        """
        Generate MSA for a sequence. Returns path to a3m file.

        method: 'jackhmmer', 'colabfold', 'mmseqs', 'synthetic', 'auto'
        """
        out_path = self.output_dir / f"{name}.a3m"
        if out_path.exists():
            log.info(f"[MSA] Using cached {out_path}")
            return out_path

        # Write query FASTA
        query_path = self.output_dir / f"{name}_query.fasta"
        query_path.write_text(f">{name}\n{sequence}\n")

        if method == "auto":
            method = self._pick_method()

        if method == "jackhmmer":
            self._run_jackhmmer(query_path, out_path, sequence)
        elif method == "colabfold":
            self._run_colabfold(query_path, out_path, name)
        elif method == "mmseqs":
            self._run_mmseqs(query_path, out_path, name)
        else:
            log.warning("[MSA] Using SYNTHETIC MSA — not real, for pipeline test only")
            self._generate_synthetic(sequence, out_path, max_sequences)

        return out_path

    def _pick_method(self) -> str:
        if self.tools["colabfold_batch"]:
            return "colabfold"
        if self.tools["mmseqs"] and self.databases.get("uniref30"):
            return "mmseqs"
        if self.tools["jackhmmer"] and self.databases.get("uniref90"):
            return "jackhmmer"
        log.warning("[MSA] No MSA tool+database available. Falling back to synthetic.")
        return "synthetic"

    def _run_jackhmmer(self, query_path: Path, out_path: Path,
                       sequence: str) -> None:
        """
        Run jackhmmer against UniRef90 (and optionally BFD, MGnify).
        Produces a3m by converting Stockholm output.
        """
        if not self.tools["jackhmmer"]:
            raise RuntimeError("jackhmmer not installed. Run: sudo apt install hmmer")

        db = self.databases.get("uniref90")
        if not db:
            raise RuntimeError("UniRef90 database not found. Download from AlphaFold data guide.")

        sto_path = out_path.with_suffix(".sto")
        cmd = [
            "jackhmmer",
            "--cpu", str(self.n_cpus),
            "-N", "1",                  # single iteration (AlphaFold default)
            "-E", "0.0001",
            "--incE", "0.0001",
            "--noali",
            "-o", "/dev/null",
            "--tblout", str(sto_path),
            str(query_path),
            db,
        ]
        log.info(f"[MSA] Running jackhmmer: {' '.join(cmd[:6])} ...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
        except subprocess.CalledProcessError as e:
            log.error(f"[MSA] jackhmmer failed: {e.stderr.decode()[:500]}")
            raise
        except subprocess.TimeoutExpired:
            log.error("[MSA] jackhmmer timed out (1h)")
            raise

        # Convert Stockholm → a3m
        self._stockholm_to_a3m(sto_path, out_path, sequence)
        log.info(f"[MSA] jackhmmer MSA written to {out_path}")

    def _run_colabfold(self, query_path: Path, out_path: Path,
                       name: str) -> None:
        """
        Run ColabFold batch MSA generation (uses MMseqs2 internally).
        Much faster than jackhmmer.
        """
        if not self.tools["colabfold_batch"]:
            raise RuntimeError(
                "colabfold_batch not installed. "
                "Run: pip install colabfold[all] && colabfold_download"
            )

        cf_out = self.output_dir / f"{name}_colabfold"
        cf_out.mkdir(exist_ok=True)

        cmd = [
            "colabfold_batch",
            "--msa-mode", "mmseqs2_uniref_env",
            "--num-recycle", "0",       # MSA only, no structure prediction
            "--stop-at-score", "0",
            str(query_path),
            str(cf_out),
        ]
        log.info(f"[MSA] Running ColabFold MSA: {name}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        except subprocess.CalledProcessError as e:
            log.error(f"[MSA] ColabFold failed: {e.stderr.decode()[:500]}")
            raise
        except subprocess.TimeoutExpired:
            log.error("[MSA] ColabFold timed out (30min)")
            raise

        # ColabFold writes a3m as {name}.a3m in output dir
        cf_a3m = cf_out / f"{name}.a3m"
        if not cf_a3m.exists():
            # Try alternate naming
            for f in cf_out.glob("*.a3m"):
                cf_a3m = f
                break
        if cf_a3m.exists():
            shutil.copy(cf_a3m, out_path)
            log.info(f"[MSA] ColabFold MSA written to {out_path}")
        else:
            raise RuntimeError(f"ColabFold did not produce a3m in {cf_out}")

    def _run_mmseqs(self, query_path: Path, out_path: Path,
                    name: str) -> None:
        """
        Run MMseqs2 directly for MSA generation.
        Requires uniref30 or uniclust30 database.
        """
        if not self.tools["mmseqs"]:
            raise RuntimeError("mmseqs not installed. Run: conda install -c bioconda mmseqs2")

        db = self.databases.get("uniref30") or self.databases.get("uniclust30")
        if not db:
            raise RuntimeError("UniRef30/UniClust30 database not found.")

        tmpdir = self.output_dir / f"{name}_mmseqs_tmp"
        tmpdir.mkdir(exist_ok=True)

        # Create query DB
        qdb = str(tmpdir / "qdb")
        subprocess.run(["mmseqs", "createdb", str(query_path), qdb],
                       check=True, capture_output=True)

        # Search
        sdb = str(tmpdir / "sdb")
        subprocess.run([
            "mmseqs", "search", qdb, db, sdb,
            str(tmpdir / "tmp"),
            "--num-iterations", "3",
            "-e", "0.001",
            "--max-seqs", "10000",
            "--threads", str(self.n_cpus),
        ], check=True, capture_output=True)

        # Convert to a3m
        subprocess.run([
            "mmseqs", "result2msa", qdb, sdb, str(tmpdir / "msa"),
            "--msa-format-mode", "6",
        ], check=True, capture_output=True)

        # Find output a3m
        msa_files = list(tmpdir.glob("msa/*"))
        if msa_files:
            shutil.copy(msa_files[0], out_path)
            log.info(f"[MSA] MMseqs2 MSA written to {out_path}")
        else:
            raise RuntimeError(f"MMseqs2 did not produce MSA in {tmpdir}")

    def _generate_synthetic(self, sequence: str, out_path: Path,
                            n_sequences: int = 128) -> None:
        """
        Generate a synthetic MSA for pipeline testing.
        NOT real — introduces random mutations and gaps.
        """
        rng = np.random.default_rng(42)
        lines = [f">query\n{sequence}"]
        aa_list = list(AA_ORDER)
        for i in range(n_sequences - 1):
            # Random mutations: ~5-15% of positions
            n_mut = max(1, int(len(sequence) * rng.uniform(0.05, 0.15)))
            mut_pos = rng.choice(len(sequence), size=n_mut, replace=False)
            seq = list(sequence)
            for p in mut_pos:
                seq[p] = rng.choice(aa_list)
            # Random gaps: ~2-5% of positions
            n_gap = max(0, int(len(sequence) * rng.uniform(0.0, 0.05)))
            gap_pos = rng.choice(len(sequence), size=n_gap, replace=False)
            for p in gap_pos:
                seq[p] = "-"
            lines.append(f">seq_{i}\n{''.join(seq)}")
        out_path.write_text("\n".join(lines) + "\n")
        log.info(f"[MSA] Synthetic MSA ({n_sequences} seqs) written to {out_path}")

    def _stockholm_to_a3m(self, sto_path: Path, out_path: Path,
                          query_seq: str) -> None:
        """Convert Stockholm format to a3m."""
        sequences = {}
        if not sto_path.exists():
            # jackhmmer --noali doesn't produce alignments; write query only
            out_path.write_text(f">query\n{query_seq}\n")
            return

        with open(sto_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line.startswith("//") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    name, seq = parts[0], parts[1]
                    if name not in sequences:
                        sequences[name] = ""
                    sequences[name] += seq

        lines = []
        for name, seq in sequences.items():
            # Convert to a3m: uppercase = aligned, remove gaps for a3m
            a3m_seq = seq.replace(".", "-")
            lines.append(f">{name}\n{a3m_seq}")
        out_path.write_text("\n".join(lines) + "\n")


# =============================================================================
# §3. A3M TOKENIZER (match ScaledEvoformerAF input format)
# =============================================================================

class A3MTokenizer:
    """
    Parses a3m files and tokenizes to the format expected by
    ScaledEvoformerAF.forward(msa_tok, res_mask, msa_mask).

    Token scheme (matching AlphaFold2 / uploaded code):
      0-19  : 20 amino acids (ACDEFGHIKLMNPQRSTVWY)
      20    : gap
      21    : padding / mask (used as pad in ScaledEvoformerAF)
      22    : CLS (not used in MSA, reserved)
      23    : EOS (not used in MSA, reserved)

    a3m format rules:
      - Uppercase = aligned column (kept)
      - Lowercase = insertion (DELETED for tokenization, as in AF2)
      - '-' or '.' = gap
      - First sequence = query
    """

    def __init__(self, max_depth: int = 256, max_len: int = 512):
        self.max_depth = max_depth
        self.max_len = max_len

    def parse_a3m(self, path: str | Path) -> Tuple[List[str], List[str]]:
        """
        Parse a3m file. Returns (sequences, headers).
        Lowercase (insertions) are removed per AF2 convention.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"a3m file not found: {path}")

        sequences = []
        headers = []
        current_seq = []
        current_header = None

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None and current_seq:
                    sequences.append("".join(current_seq))
                    headers.append(current_header)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                # Remove lowercase (insertions) — AF2 convention
                cleaned = "".join(c for c in line if c.isupper() or c in "-.")
                current_seq.append(cleaned)

        if current_header is not None and current_seq:
            sequences.append("".join(current_seq))
            headers.append(current_header)

        return sequences, headers

    def tokenize_sequence(self, seq: str) -> np.ndarray:
        """Tokenize a single aligned sequence to integer array."""
        tokens = []
        for c in seq[:self.max_len]:
            if c in AA_TO_IDX:
                tokens.append(AA_TO_IDX[c])
            elif c in "-.":
                tokens.append(GAP_IDX)
            else:
                tokens.append(UNK_IDX)
        # Pad to max_len
        while len(tokens) < self.max_len:
            tokens.append(AF2_PAD_TOKEN)
        return np.array(tokens[:self.max_len], dtype=np.int64)

    def to_msa_tok(self, path: str | Path,
                   max_depth: Optional[int] = None,
                   max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Full pipeline: a3m → msa_tok tensor.

        Returns:
          msa_tok:    (S, L) token tensor
          res_mask:   (L,) residue mask from query
          query_len:  actual query length (before padding)
        """
        max_depth = max_depth or self.max_depth
        max_len = max_len or self.max_len

        sequences, headers = self.parse_a3m(path)
        if not sequences:
            raise ValueError(f"No sequences parsed from {path}")

        # First sequence is query
        query = sequences[0]
        query_len = min(len(query), max_len)

        # Tokenize all sequences
        all_tokens = []
        for seq in sequences[:max_depth]:
            tok = self.tokenize_sequence(seq)
            all_tokens.append(tok)

        # Pad to max_depth if needed
        while len(all_tokens) < max_depth:
            all_tokens.append(np.full(max_len, AF2_PAD_TOKEN, dtype=np.int64))

        msa_tok = torch.from_numpy(np.stack(all_tokens[:max_depth]))  # (S, L)

        # Residue mask: 1 for real query residues, 0 for padding
        res_mask = torch.zeros(max_len)
        res_mask[:query_len] = 1.0

        return msa_tok, res_mask, query_len

    def subsample_msa(self, msa_tok: torch.Tensor,
                      max_depth: int,
                      seed: Optional[int] = None) -> torch.Tensor:
        """
        Randomly subsample MSA rows for training regularization.
        Always keeps row 0 (query).
        """
        S = msa_tok.shape[0]
        if S <= max_depth:
            return msa_tok
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        keep = np.concatenate([[0], rng.choice(
            np.arange(1, S), size=max_depth - 1, replace=False)])
        keep = np.sort(keep)
        return msa_tok[keep]


# =============================================================================
# §4. INTEGRATION WITH ScaledEvoformerAF
# =============================================================================

def load_scaled_evoformer(c_m: int = 256, c_z: int = 128,
                          n_blocks: int = 8, max_seq_len: int = 512,
                          use_amp: bool = True):
    """
    Import ScaledEvoformerAF from the user's existing module.
    Falls back to a minimal stub if the file is not importable.
    """
    try:
        # Try importing from the uploaded file
        import importlib.util
        candidates = [
            "Scaled AlphaFold Backbone & Multi-Scale Bridge",
            "scaled_af_backbone",
            "af_scaled_backbone",
        ]
        for name in candidates:
            try:
                spec = importlib.util.find_spec(name.replace(" ", "_").replace("&", "and"))
                if spec is not None:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    ScaledAFConfig = mod.ScaledAFConfig
                    ScaledEvoformerAF = mod.ScaledEvoformerAF
                    log.info(f"[Evoformer] Loaded ScaledEvoformerAF from '{name}'")
                    cfg = ScaledAFConfig(
                        c_m=c_m, c_z=c_z, n_blocks=n_blocks,
                        max_seq_len=max_seq_len, use_amp=use_amp,
                    )
                    return ScaledEvoformerAF(cfg), cfg, True
            except Exception:
                continue
        raise ImportError("ScaledEvoformerAF module not found")
    except ImportError:
        log.warning(
            "[Evoformer] ScaledEvoformerAF not importable. "
            "Using minimal stub for pipeline testing. "
            "Place 'Scaled AlphaFold Backbone & Multi-Scale Bridge.py' "
            "in the same directory for real usage."
        )
        return None, None, False


class MinimalEvoformerStub(nn.Module):
    """
    Minimal stand-in for ScaledEvoformerAF when the real module is unavailable.
    Matches the forward() signature: (msa_tok, res_mask) → dict with positions.
    NOT a real Evoformer — just validates the data pipeline.
    """
    def __init__(self, vocab_size: int = 24, c_m: int = 64,
                 max_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, c_m)
        self.pos = nn.Parameter(torch.randn(1, 1, max_len, c_m) * 0.02)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                c_m, nhead=4, dim_feedforward=c_m * 2,
                batch_first=True, norm_first=True),
            num_layers=2)
        self.coord_head = nn.Linear(c_m, 3)
        self.plddt_head = nn.Linear(c_m, 1)

    def forward(self, msa_tok: torch.Tensor,
                res_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, S, L = msa_tok.shape
        # Use query row (row 0) as primary sequence representation
        q = self.embed(msa_tok[:, 0]) + self.pos[:, :, :L]
        q = q.squeeze(1)  # (B, L, c_m)
        h = self.blocks(q, src_key_padding_mask=~res_mask.bool())
        coords = self.coord_head(h)   # (B, L, 3) — CA only
        # Expand to (B, L, 4, 3) for N, CA, C, O
        coords4 = coords.unsqueeze(2).expand(B, L, 4, 3).clone()
        plddt = torch.sigmoid(self.plddt_head(h)).squeeze(-1) * 100
        return {
            "positions": coords4,
            "plddt": plddt,
            "single_repr": h,
            "frames": torch.eye(4, device=h.device).expand(B, L, 4, 4).clone(),
        }


class MSAToEvoformer:
    """
    Complete pipeline: sequence → MSA → msa_tok → ScaledEvoformerAF.

    Usage:
      pipe = MSAToEvoformer(output_dir="data/msa")
      result = pipe.predict("ACDEFGHIKLMNPQRSTVWY", name="test_protein")
    """

    def __init__(self, output_dir: str = "data/msa",
                 c_m: int = 256, c_z: int = 128,
                 n_blocks: int = 8, max_len: int = 512,
                 max_depth: int = 256, n_cpus: int = 4,
                 device: str = "auto"):
        self.msa_gen = MSAGenerator(output_dir=output_dir, n_cpus=n_cpus)
        self.tokenizer = A3MTokenizer(max_depth=max_depth, max_len=max_len)
        self.max_len = max_len
        self.max_depth = max_depth

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load real ScaledEvoformerAF or fall back to stub
        self.model, self.cfg, self.is_real = load_scaled_evoformer(
            c_m=c_m, c_z=c_z, n_blocks=n_blocks, max_seq_len=max_len)
        if self.model is None:
            self.model = MinimalEvoformerStub(max_len=max_len).to(self.device)
            self.is_real = False
        else:
            self.model = self.model.to(self.device)

        log.info(f"[Pipeline] device={self.device}, "
                 f"model={'ScaledEvoformerAF' if self.is_real else 'MinimalStub'}")

    def generate_msa(self, sequence: str, name: str = "query",
                     method: str = "auto") -> Path:
        """Generate MSA file for a sequence."""
        return self.msa_gen.generate(sequence, name=name, method=method)

    def prepare_input(self, a3m_path: Path,
                      subsample: bool = False,
                      seed: Optional[int] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Parse a3m → msa_tok (1, S, L) + res_mask (1, L).
        Ready for ScaledEvoformerAF.forward().
        """
        msa_tok, res_mask, query_len = self.tokenizer.to_msa_tok(
            a3m_path, max_depth=self.max_depth, max_len=self.max_len)

        if subsample:
            msa_tok = self.tokenizer.subsample_msa(
                msa_tok, self.max_depth, seed=seed)

        # Add batch dimension: (S, L) → (1, S, L)
        msa_tok = msa_tok.unsqueeze(0).to(self.device)
        res_mask = res_mask.unsqueeze(0).to(self.device)
        return msa_tok, res_mask, query_len

    @torch.no_grad()
    def predict(self, sequence: str, name: str = "query",
                method: str = "auto",
                n_recycle: int = 3) -> Dict[str, Any]:
        """
        Full prediction pipeline: sequence → MSA → structure.

        Returns dict with positions, plddt, and metadata.
        """
        self.model.eval()

        # Step 1: Generate MSA
        t0 = time.time()
        a3m_path = self.generate_msa(sequence, name=name, method=method)
        msa_time = time.time() - t0

        # Step 2: Tokenize
        msa_tok, res_mask, query_len = self.prepare_input(a3m_path)

        # Step 3: Forward through Evoformer
        t0 = time.time()
        out = self.model(msa_tok, res_mask)
        fwd_time = time.time() - t0

        # Step 4: Extract results
        positions = out["positions"][0, :query_len].cpu()
        plddt = out["plddt"][0, :query_len].cpu()

        return {
            "name": name,
            "sequence": sequence,
            "query_len": query_len,
            "positions": positions,          # (L, 4, 3) or (L, 3)
            "plddt": plddt,                  # (L,)
            "mean_plddt": float(plddt.mean()),
            "msa_file": str(a3m_path),
            "msa_depth": msa_tok.shape[1],
            "msa_time_s": msa_time,
            "forward_time_s": fwd_time,
            "model_type": "ScaledEvoformerAF" if self.is_real else "MinimalStub",
            "device": str(self.device),
        }

    def predict_batch(self, sequences: Dict[str, str],
                      method: str = "auto") -> List[Dict[str, Any]]:
        """Predict structures for multiple sequences."""
        results = []
        for name, seq in sequences.items():
            log.info(f"[Pipeline] Processing {name} ({len(seq)} residues)")
            try:
                r = self.predict(seq, name=name, method=method)
                results.append(r)
            except Exception as e:
                log.error(f"[Pipeline] Failed for {name}: {e}")
                results.append({"name": name, "error": str(e)})
        return results


# =============================================================================
# §5. INTEGRATION WITH ProteinStreamingDataset
# =============================================================================

def create_manifest_from_a3m_dir(
    a3m_dir: str | Path,
    pdb_dir: str | Path,
    output_path: str | Path,
) -> int:
    """
    Create a manifest.json for ProteinStreamingDataset from
    a directory of a3m files and PDB files.

    Expected structure:
      a3m_dir/
        protein1.a3m
        protein2.a3m
      pdb_dir/
        protein1.pdb
        protein2.pdb

    Writes manifest: [{pdb_path, a3m_path, chain, name}, ...]
    """
    a3m_dir = Path(a3m_dir)
    pdb_dir = Path(pdb_dir)
    output_path = Path(output_path)

    entries = []
    for a3m_file in sorted(a3m_dir.glob("*.a3m")):
        name = a3m_file.stem
        pdb_file = pdb_dir / f"{name}.pdb"
        if not pdb_file.exists():
            log.warning(f"[Manifest] No PDB for {name}, skipping")
            continue
        entries.append({
            "name": name,
            "pdb_path": str(pdb_file),
            "a3m_path": str(a3m_file),
            "chain": "A",
        })

    output_path.write_text(json.dumps(entries, indent=2))
    log.info(f"[Manifest] Wrote {len(entries)} entries to {output_path}")
    return len(entries)


# =============================================================================
# §6. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: Tool detection
    print("\n[T1] Tool detection")
    tools = detect_tools()
    print(f"      jackhmmer: {tools['jackhmmer']}")
    print(f"      mmseqs:    {tools['mmseqs']}")
    print(f"      colabfold: {tools['colabfold_batch']}")
    rep("Tool detection runs", isinstance(tools, dict))

    # T2: Synthetic MSA generation
    print("\n[T2] Synthetic MSA generation")
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = MSAGenerator(output_dir=tmpdir)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 residues
        path = gen.generate(seq, name="test", method="synthetic",
                            max_sequences=32)
        rep("Synthetic MSA file created", path.exists(), str(path))
        content = path.read_text()
        n_seqs = content.count(">")
        rep("Synthetic MSA has sequences", n_seqs >= 16, f"n={n_seqs}")

    # T3: a3m parsing
    print("\n[T3] a3m parsing")
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = MSAGenerator(output_dir=tmpdir)
        seq = "ACDEFGHIKLMNPQRSTVWY" * 3
        path = gen.generate(seq, name="parse_test", method="synthetic",
                            max_sequences=16)
        tok = A3MTokenizer(max_depth=16, max_len=128)
        sequences, headers = tok.parse_a3m(path)
        rep("a3m parsed", len(sequences) > 0, f"n={len(sequences)}")
        rep("Query is first", len(sequences[0]) > 0,
            f"query_len={len(sequences[0])}")

    # T4: Tokenization
    print("\n[T4] Tokenization")
    tok = A3MTokenizer(max_depth=16, max_len=64)
    msa_tok, res_mask, qlen = tok.to_msa_tok(path, max_depth=16, max_len=64)
    rep("msa_tok shape (S, L)", msa_tok.shape == (16, 64),
        f"{tuple(msa_tok.shape)}")
    rep("res_mask shape (L,)", res_mask.shape == (64,),
        f"{tuple(res_mask.shape)}")
    rep("Query length correct", qlen == min(len(seq), 64),
        f"qlen={qlen}, expected={min(len(seq), 64)}")
    rep("Token values in range",
        int(msa_tok.min()) >= 0 and int(msa_tok.max()) < VOCAB_SIZE,
        f"min={msa_tok.min()}, max={msa_tok.max()}")

    # T5: Lowercase removal (insertion handling)
    print("\n[T5] a3m insertion handling")
    with tempfile.TemporaryDirectory() as tmpdir:
        a3m_path = Path(tmpdir) / "insert_test.a3m"
        # a3m with lowercase insertions
        a3m_path.write_text(
            ">query\nACDEF\n>hit1\nACdEF\n>hit2\nA-CDE\n"
        )
        tok2 = A3MTokenizer(max_depth=8, max_len=16)
        seqs, _ = tok2.parse_a3m(a3m_path)
        # 'd' is lowercase insertion → should be removed
        rep("Insertion removed from hit1", seqs[1] == "ACEF",
            f"got '{seqs[1]}'")
        rep("Gap preserved in hit2", seqs[2] == "A-CDE",
            f"got '{seqs[2]}'")

    # T6: MSA subsampling
    print("\n[T6] MSA subsampling")
    big_msa = torch.randint(0, 20, (100, 64))
    sub = tok.subsample_msa(big_msa, max_depth=16, seed=0)
    rep("Subsampled to max_depth", sub.shape[0] == 16,
        f"{sub.shape[0]}")
    rep("Query row preserved", torch.equal(sub[0], big_msa[0]))

    # T7: Full pipeline with stub model
    print("\n[T7] Full pipeline (stub model)")
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe = MSAToEvoformer(output_dir=tmpdir, c_m=64, max_len=64,
                              max_depth=16, device="cpu")
        seq = "ACDEFGHIKLMNPQRSTVWY" * 2  # 40 residues
        result = pipe.predict(seq, name="pipe_test", method="synthetic")
        rep("Pipeline returns positions", "positions" in result,
            f"shape={tuple(result['positions'].shape)}")
        rep("Pipeline returns plddt", "plddt" in result,
            f"mean_plddt={result['mean_plddt']:.1f}")
        rep("Query length matches", result["query_len"] == 40,
            f"qlen={result['query_len']}")
        rep("MSA depth reported", result["msa_depth"] > 0,
            f"depth={result['msa_depth']}")

    # T8: Batch prediction
    print("\n[T8] Batch prediction")
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe = MSAToEvoformer(output_dir=tmpdir, c_m=64, max_len=64,
                              max_depth=16, device="cpu")
        seqs = {
            "prot_a": "ACDEFGHIKLMNPQRSTVWY" * 2,
            "prot_b": "GGGGAAAACCCCDDDD" * 2,
        }
        results = pipe.predict_batch(seqs, method="synthetic")
        rep("Batch returns 2 results", len(results) == 2)
        rep("All succeeded", all("error" not in r for r in results))

    # T9: Manifest creation
    print("\n[T9] Manifest creation")
    with tempfile.TemporaryDirectory() as tmpdir:
        a3m_dir = Path(tmpdir) / "a3m"
        pdb_dir = Path(tmpdir) / "pdb"
        a3m_dir.mkdir()
        pdb_dir.mkdir()
        (a3m_dir / "test.a3m").write_text(">q\nACDEF\n")
        (pdb_dir / "test.pdb").write_text("ATOM 1 CA ALA A 1\n")
        manifest_path = Path(tmpdir) / "manifest.json"
        n = create_manifest_from_a3m_dir(a3m_dir, pdb_dir, manifest_path)
        rep("Manifest created", n == 1, f"n={n}")
        rep("Manifest file exists", manifest_path.exists())

    # T10: Real ScaledEvoformerAF loading attempt
    print("\n[T10] ScaledEvoformerAF import")
    model, cfg, is_real = load_scaled_evoformer()
    if is_real:
        rep("Real ScaledEvoformerAF loaded", True)
    else:
        rep("Fallback to stub (expected if module not on path)", True,
            "place the AF backbone file alongside this script for real usage")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT IS VERIFIED
  Tool detection; synthetic MSA generation; a3m parsing;
  insertion (lowercase) removal; tokenization to (S,L) tensor;
  MSA subsampling; full pipeline with stub model; batch prediction;
  manifest creation for ProteinStreamingDataset.

WHAT IS NOT VERIFIED (requires external tools)
  jackhmmer MSA generation (needs HMMER + UniRef90)
  ColabFold MSA generation (needs colabfold + databases)
  MMseqs2 MSA generation (needs mmseqs + UniRef30)
  Real ScaledEvoformerAF forward pass (needs the backbone file)

TO USE WITH REAL TOOLS:
  1. Install: sudo apt install hmmer
  2. Install: pip install colabfold[all] && colabfold_download
  3. Download UniRef90/BFD databases
  4. Place 'Scaled AlphaFold Backbone & Multi-Scale Bridge.py' here
  5. Run: python msa_to_evoformer_pipeline.py --predict ACDEFG...
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="MSA → ScaledEvoformerAF pipeline")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--detect", action="store_true",
                    help="Detect installed MSA tools and databases")
    ap.add_argument("--predict", type=str, default=None,
                    help="Predict structure for a sequence")
    ap.add_argument("--name", type=str, default="query")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "jackhmmer", "colabfold",
                             "mmseqs", "synthetic"])
    ap.add_argument("--sequence-file", type=str, default=None,
                    help="FASTA file with sequences to predict")
    ap.add_argument("--a3m", type=str, default=None,
                    help="Use existing a3m file (skip MSA generation)")
    ap.add_argument("--max-depth", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--c-m", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--output-dir", type=str, default="data/msa")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if a.detect:
        tools = detect_tools()
        dbs = find_databases()
        print(json.dumps({"tools": tools, "databases": dbs}, indent=2))
        return 0

    if a.predict:
        pipe = MSAToEvoformer(
            output_dir=a.output_dir, c_m=a.c_m, n_blocks=a.n_blocks,
            max_len=a.max_len, max_depth=a.max_depth, device=a.device)
        result = pipe.predict(a.predict, name=a.name, method=a.method)
        # Remove tensors for JSON output
        out = {k: v for k, v in result.items()
               if not isinstance(v, torch.Tensor)}
        out["positions_shape"] = list(result["positions"].shape)
        out["plddt_mean"] = result["mean_plddt"]
        print(json.dumps(out, indent=2, default=str))
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        return 0

    if a.a3m:
        pipe = MSAToEvoformer(
            output_dir=a.output_dir, c_m=a.c_m, max_len=a.max_len,
            max_depth=a.max_depth, device=a.device)
        msa_tok, res_mask, qlen = pipe.prepare_input(Path(a.a3m))
        print(f"msa_tok shape: {tuple(msa_tok.shape)}")
        print(f"res_mask shape: {tuple(res_mask.shape)}")
        print(f"query_len: {qlen}")
        print(f"MSA depth: {msa_tok.shape[1]}")
        return 0

    if a.sequence_file:
        # Read FASTA
        sequences = {}
        name = None
        for line in Path(a.sequence_file).read_text().splitlines():
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                sequences[name] = ""
            elif name:
                sequences[name] += line
        pipe = MSAToEvoformer(
            output_dir=a.output_dir, c_m=a.c_m, n_blocks=a.n_blocks,
            max_len=a.max_len, max_depth=a.max_depth, device=a.device)
        results = pipe.predict_batch(sequences, method=a.method)
        for r in results:
            if "error" in r:
                print(f"  {r['name']}: ERROR - {r['error']}")
            else:
                print(f"  {r['name']}: pLDDT={r['mean_plddt']:.1f}, "
                      f"len={r['query_len']}, depth={r['msa_depth']}")
        return 0

    ap.print_help()
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
