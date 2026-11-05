"""
=============================================================================
SESI FULL STAGE SCRIPTS + MSA-ENHANCED FOLDING INTEGRATION — v1.0.0
Complete 4-Stage Pipeline + MSAEnhancedFoldingSystem Bridge
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
=============================================================================
WHAT THIS MODULE DELIVERS
=============================================================================
  [1] MSAEnhancedFoldingBridge — direct wrapper around MSAEnhancedFoldingSystem
        • Parses .a3m/.sto/.fasta MSA files → (D, L) token tensor (vocab 23)
        • Loads pre-computed ESM-2 features (.pt) or generates mock
        • Calls MSAEnhancedFoldingSystem.forward(sequence, esm, msa) verbatim
        • Returns predicted_coords + MSA diagnostics (compression ratio,
          Hodge SNR, confidence, columns kept, consistency loss)

  [2] Stage1Designer   — chunked de novo sequence design (complete)
  [3] Stage2Folder     — SDA-ONE hierarchical folding + MSA enhancement
  [4] Stage3Refiner    — RefinementEngine OpenMM-ML refinement (k*=23)
  [5] Stage4Validator  — clash/RMSD/geometry validation + JSON summary
  [6] FullPipelineOrchestrator — runs all 4 stages end-to-end
  [7] write_stage_scripts() — emits 4 standalone runnable stage scripts

HONEST SCOPE:
  • MSA mode requires real MSA files; without them the bridge falls back to
    SINGLE_SEQUENCE mode automatically (via AdaptiveMSARouter).
  • ESM features require pre-computed .pt files; without them, mock random
    embeddings are used (clearly labeled).
  • MSAEnhancedFoldingSystem's folding_head is UNTRAINED by default — its
    predicted_coords are only structurally meaningful after training. For
    production structure, SDA-ONE + RefinementEngine is the physical path.
  • RefinementEngine requires OpenMM + OpenMM-ML + openmm-torch. Without
    them, Stage 3 degrades to a warning-only pass-through.
=============================================================================
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sesi.full_stage_scripts")
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    level=logging.INFO)

STAGE_SUITE_VERSION: str = "1.0.0"

# =============================================================================
# OPMC PAPER CONSTANTS
# =============================================================================
OPMC_C1 = 0.420000
OPMC_K2 = 12.5000
OPMC_L = 0.438520
OPMC_ETA_LATTICE = 0.99491
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n

def opmc_k_star(epsilon: float = 1e-8, D0: float = 1.0) -> int:
    """Theorem 11.1: k* = ⌈ln(ε/D₀)/ln L⌉ — independent of problem size N."""
    return math.ceil(math.log(epsilon / D0) / math.log(OPMC_L))

# =============================================================================
# Shared alphabet
# =============================================================================
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical, matches Designer
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_ALPHABET)}
# MSA vocab (23): 20 AA + gap(20) + unknown(21) + mask(22)
MSA_GAP, MSA_UNK, MSA_MASK = 20, 21, 22

def tokenize_sequence(sequence: str) -> torch.Tensor:
    return torch.tensor([AA_TO_IDX.get(c, 0) for c in sequence], dtype=torch.long)

def detokenize(tokens: torch.Tensor) -> str:
    return "".join(IDX_TO_AA.get(int(t), "X") for t in tokens.reshape(-1))

# =============================================================================
# Ecosystem imports (graceful)
# =============================================================================
def _try_import(module_candidates: List[str], names: List[str]):
    """Try importing `names` from the first importable module in candidates."""
    import importlib, importlib.util
    for cand in module_candidates:
        mod = None
        try:
            mod = importlib.import_module(cand)
        except ImportError:
            p = Path(cand + ".py") if not cand.endswith(".py") else Path(cand)
            for candidate_path in [p, Path(cand.replace("_", "_"))]:
                if candidate_path.exists():
                    spec = importlib.util.spec_from_file_location(cand, str(candidate_path))
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        break
        if mod is not None:
            try:
                return mod, {n: getattr(mod, n) for n in names}
            except AttributeError:
                continue
    return None, {}

_HAS_DESIGNER = False
_mod, _imp = _try_import(["denovo_sequence_designer_fixed"], ["DifferentiableStructuralDesigner"])
if _mod:
    DifferentiableStructuralDesigner = _imp["DifferentiableStructuralDesigner"]
    _HAS_DESIGNER = True

_HAS_AGENT = False
_mod, _imp = _try_import(["denovo_protein_calculus_agent_fixed"],
                          ["DeNovoProteinCalculusAgent", "ProteinCalculusConfig"])
if _mod:
    DeNovoProteinCalculusAgent = _imp["DeNovoProteinCalculusAgent"]
    ProteinCalculusConfig = _imp["ProteinCalculusConfig"]
    _HAS_AGENT = True

_HAS_SDA = False
_mod, _imp = _try_import(["structural_domain_assembly_one"],
                          ["StructuralDomainAssembly", "DomainAssemblyConfig"])
if _mod:
    StructuralDomainAssembly = _imp["StructuralDomainAssembly"]
    DomainAssemblyConfig = _imp["DomainAssemblyConfig"]
    _HAS_SDA = True

_HAS_MSA = False
_mod, _imp = _try_import(["MSA_ENHANCED_PROTEIN_FOLDING", "MSA_ENHANCED_PROTEIN_FOLDING (1)"],
                          ["MSAEnhancedConfig", "MSAFoldingMode", "MSAEnhancedFoldingSystem"])
if _mod:
    MSAEnhancedConfig = _imp["MSAEnhancedConfig"]
    MSAFoldingMode = _imp["MSAFoldingMode"]
    MSAEnhancedFoldingSystem = _imp["MSAEnhancedFoldingSystem"]
    _HAS_MSA = True

_HAS_REAL_FOLD = False
_mod, _imp = _try_import(["real_fold_one_v2", "real_fold_one_v2-5"],
                          ["RefinementEngine", "RefinementConfig", "openmm_solute_energy",
                           "validate_structure"])
if _mod:
    RefinementEngine = _imp["RefinementEngine"]
    RefinementConfig = _imp["RefinementConfig"]
    openmm_solute_energy = _imp["openmm_solute_energy"]
    validate_structure = _imp.get("validate_structure")
    _HAS_REAL_FOLD = True

logger.info(f"[imports] designer={_HAS_DESIGNER} agent={_HAS_AGENT} sda={_HAS_SDA} "
            f"msa={_HAS_MSA} real_fold={_HAS_REAL_FOLD}")


# =============================================================================
# [1] MSA PARSING + ESM LOADING
# =============================================================================
def parse_msa_file(path: os.PathLike | str, max_depth: int = 128) -> Optional[List[str]]:
    """Parse .a3m/.sto/.fasta MSA into a list of aligned sequences."""
    path = Path(path)
    if not path.exists():
        return None
    sequences: List[str] = []
    current: List[str] = []
    is_a3m = path.suffix.lower() == ".a3m"
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current)); current = []
                continue
            parts = line.split()
            seq_part = parts[-1] if parts else ""
            if is_a3m:
                seq_part = re.sub(r"[a-z]", "", seq_part)  # lowercase = insertions
            seq_part = seq_part.replace(".", "-").replace(" ", "")
            if seq_part:
                current.append(seq_part)
    if current:
        sequences.append("".join(current))
    if not sequences:
        return None
    sequences = sequences[:max_depth]
    max_len = max(len(s) for s in sequences)
    return [s.ljust(max_len, "-") for s in sequences]


def msa_to_token_tensor(sequences: List[str]) -> torch.Tensor:
    """Convert aligned sequences → (D, L) tensor in 23-token MSA vocab."""
    if not sequences:
        return torch.empty(0, 0, dtype=torch.long)
    D, L = len(sequences), len(sequences[0])
    tensor = torch.full((D, L), MSA_UNK, dtype=torch.long)
    for i, seq in enumerate(sequences):
        for j, c in enumerate(seq[:L]):
            if c in ("- ".):
                tensor[i, j] = MSA_GAP
            elif c in AA_TO_IDX:
                tensor[i, j] = AA_TO_IDX[c]
            else:
                tensor[i, j] = MSA_UNK
    return tensor


def load_esm_features(esm_path: Optional[str], sequence: str, esm_dim: int,
                        device: torch.device) -> torch.Tensor:
    """Load ESM-2 features from .pt, or return mock (clearly labeled)."""
    L = len(sequence)
    if esm_path and os.path.exists(esm_path):
        try:
            data = torch.load(esm_path, map_location="cpu")
            if isinstance(data, dict):
                if "representations" in data:
                    layers = sorted(data["representations"].keys())
                    esm = data["representations"][layers[-1]]
                else:
                    esm = data.get("embeddings", data.get("features"))
            else:
                esm = data
            if esm is not None:
                if esm.shape[0] > L:
                    esm = esm[1:L + 1]  # strip BOS/EOS
                elif esm.shape[0] < L:
                    pad = torch.zeros(L - esm.shape[0], esm.shape[1])
                    esm = torch.cat([esm, pad], dim=0)
                logger.info(f"[ESM] Loaded real features from {esm_path}: {tuple(esm.shape)}")
                return esm.unsqueeze(0).to(device)
        except Exception as e:
            logger.warning(f"[ESM] Failed to load {esm_path}: {e}")
    logger.warning("[ESM] Using MOCK random embeddings (no .pt file). Not real ESM-2.")
    return torch.randn(1, L, esm_dim, device=device)


# =============================================================================
# [2] MSAEnhancedFoldingBridge — DIRECT integration with MSAEnhancedFoldingSystem
# =============================================================================
@dataclass
class MSABridgeConfig:
    """Configuration for the MSA-enhanced folding bridge."""
    mode: str = "adaptive"             # single_sequence | msa_enhanced | adaptive
    esm_dim: int = 1280
    fused_dim: int = 256
    hodge_dim: int = 128
    max_msa_depth: int = 128
    max_seq_len_for_msa: int = 1000
    device: str = "auto"

    def to_msa_config(self) -> Any:
        if not _HAS_MSA:
            raise ImportError("MSA_ENHANCED_PROTEIN_FOLDING not available.")
        return MSAEnhancedConfig(
            max_msa_depth=self.max_msa_depth,
            hodge_dim=self.hodge_dim,
            esm_dim=self.esm_dim,
            fused_dim=self.fused_dim,
            max_seq_len_for_msa=self.max_seq_len_for_msa,
            device=self.device,
        )

    def to_msa_mode(self) -> Any:
        if not _HAS_MSA:
            raise ImportError("MSA_ENHANCED_PROTEIN_FOLDING not available.")
        return {
            "single_sequence": MSAFoldingMode.SINGLE_SEQUENCE,
            "msa_enhanced": MSAFoldingMode.MSA_ENHANCED,
            "adaptive": MSAFoldingMode.ADAPTIVE,
        }[self.mode]


class MSAEnhancedFoldingBridge:
    """
    [D] Direct bridge to MSAEnhancedFoldingSystem from MSA_ENHANCED_PROTEIN_FOLDING.

    Calls MSAEnhancedFoldingSystem.forward(sequence, esm_features, msa) verbatim.
    Returns predicted_coords plus the full MSA diagnostic set:
        use_msa, msa_decision_reason, msa_compress_compression_ratio,
        msa_compress_final_confidence, msa_hodge_signal_to_noise_ratio,
        msa_filter_n_kept, msa_fusion_consistency_loss, msa_fusion_final_fused

    HONEST: the folding_head inside MSAEnhancedFoldingSystem is untrained by
    default, so predicted_coords are only structurally meaningful after
    training. This bridge faithfully runs the real API and reports diagnostics.
    """

    def __init__(self, cfg: Optional[MSABridgeConfig] = None):
        if not _HAS_MSA:
            raise ImportError("MSAEnhancedFoldingBridge requires MSA_ENHANCED_PROTEIN_FOLDING.")
        self.cfg = cfg or MSABridgeConfig()
        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        self.system = MSAEnhancedFoldingSystem(
            self.cfg.to_msa_config(), mode=self.cfg.to_msa_mode()
        ).to(self.device)
        logger.info(f"[MSABridge] Initialized mode={self.cfg.mode} on {self.device}")

    def fold(self, sequence: str,
               esm_path: Optional[str] = None,
               msa_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run MSA-enhanced folding on a sequence.

        Args:
            sequence: amino acid sequence
            esm_path: optional path to pre-computed ESM-2 .pt
            msa_path: optional path to MSA file (.a3m/.sto/.fasta)
        """
        L = len(sequence)
        esm = load_esm_features(esm_path, sequence, self.cfg.esm_dim, self.device)

        msa_tensor = None
        if msa_path:
            seqs = parse_msa_file(msa_path, max_depth=self.cfg.max_msa_depth)
            if seqs:
                msa_tensor = msa_to_token_tensor(seqs).unsqueeze(0).to(self.device)  # (1,D,L)

        output = self.system(sequence, esm, msa=msa_tensor)

        result: Dict[str, Any] = {
            "use_msa": bool(output.get("use_msa", False)),
            "msa_decision_reason": output.get("msa_decision_reason", ""),
            "predicted_coords": output.get("predicted_coords"),
            "sequence_length": L,
        }
        if result["use_msa"]:
            result.update({
                "compression_ratio": float(output.get("msa_compress_compression_ratio", 0.0)),
                "final_confidence": float(output.get("msa_compress_final_confidence",
                                                     torch.tensor(0.0)).mean().item()),
                "hodge_snr": float(output.get("msa_hodge_signal_to_noise_ratio",
                                              torch.tensor(0.0)).mean().item()),
                "columns_kept": int(output.get("msa_filter_n_kept",
                                               torch.tensor(0)).sum().item()),
                "consistency_loss": float(output.get("msa_fusion_consistency_loss",
                                                     torch.tensor(0.0)).item()),
                "fused_features": output.get("msa_fusion_final_fused"),
            })
        return result


# =============================================================================
# [3] STAGE 1 — DE NOVO SEQUENCE DESIGN (complete)
# =============================================================================
@dataclass
class Stage1Config:
    n_residues: int = 100_000
    chunk_size: int = 10_000
    overlap: int = 100
    hidden_dim: int = 128
    lr: float = 1e-3
    optimize_steps: int = 50       # per-chunk viability optimization steps
    temperature: float = 1.0
    noise_scale: float = 0.1
    use_agent_gate: bool = True    # additionally gate with DeNovoProteinCalculusAgent
    output_dir: str = "./results/stage1_design"
    device: str = "auto"


class Stage1Designer:
    """
    [D] Chunked de novo sequence design.

    For each chunk: instantiate DifferentiableStructuralDesigner, optimize the
    sequence logits to maximize viability_score, then extract hard_tokens.
    Optionally gate each chunk through DeNovoProteinCalculusAgent for an
    independent viability/transition-probability readout.
    """

    def __init__(self, cfg: Optional[Stage1Config] = None):
        if not _HAS_DESIGNER:
            raise ImportError("Stage1Designer requires denovo_sequence_designer_fixed.")
        self.cfg = cfg or Stage1Config()
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if self.cfg.device == "auto" else torch.device(self.cfg.device))
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def _design_chunk(self, chunk_size: int) -> Dict[str, torch.Tensor]:
        designer = DifferentiableStructuralDesigner(
            seq_length=chunk_size, vocab_size=20, hidden_dim=self.cfg.hidden_dim,
            deterministic_eval=False,
        ).to(self.device)
        designer.train()
        optimizer = torch.optim.AdamW(designer.parameters(), lr=self.cfg.lr)

        best = None
        for step in range(self.cfg.optimize_steps):
            optimizer.zero_grad(set_to_none=True)
            out = designer(batch_size=1, temperature=self.cfg.temperature,
                           noise_scale=self.cfg.noise_scale)
            loss = -out["viability_score"].float().mean()  # maximize viability
            loss.backward()
            optimizer.step()
            if best is None or out["viability_score"].mean().item() > best["viability"]:
                best = {"tokens": out["hard_tokens"].detach().clone(),
                        "viability": out["viability_score"].mean().item()}

        del designer, optimizer
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return best

    def _agent_gate(self, tokens: torch.Tensor) -> Optional[float]:
        """Optional independent viability readout via DeNovoProteinCalculusAgent."""
        if not (self.cfg.use_agent_gate and _HAS_AGENT):
            return None
        try:
            agent_cfg = ProteinCalculusConfig(vocab_size=25, embed_dim=64,
                                              max_seq_len=tokens.shape[-1] + 8,
                                              n_vars=32, m_clauses=64, num_classes=1)
            agent = DeNovoProteinCalculusAgent(agent_cfg).to(self.device).eval()
            # pad tokens into agent vocab range (< 25)
            seq = tokens.reshape(1, -1).clamp(0, 19).to(self.device)
            with torch.no_grad():
                _, viability, transition = agent(seq, dt=0.01)
            return float(viability.mean().item())
        except Exception as e:
            logger.warning(f"[Stage1] agent gate skipped: {e}")
            return None

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        stride = cfg.chunk_size - cfg.overlap
        n_chunks = max(1, math.ceil((cfg.n_residues - cfg.overlap) / stride))
        logger.info(f"[Stage1] Designing {cfg.n_residues} residues in {n_chunks} chunks "
                    f"(chunk={cfg.chunk_size}, overlap={cfg.overlap})")

        chunk_files = []
        all_tokens = []
        for ci in range(n_chunks):
            start = ci * stride
            end = min(start + cfg.chunk_size, cfg.n_residues)
            size = end - start
            logger.info(f"[Stage1] chunk {ci+1}/{n_chunks}: residues {start}-{end} ({size})")

            best = self._design_chunk(size)
            tokens = best["tokens"].reshape(-1).cpu()
            agent_viability = self._agent_gate(tokens)

            path = os.path.join(cfg.output_dir, f"chunk_{ci:05d}.pt")
            torch.save({"tokens": tokens, "start": start, "end": end,
                        "viability": best["viability"],
                        "agent_viability": agent_viability}, path)
            chunk_files.append(path)
            all_tokens.append(tokens)
            logger.info(f"[Stage1]   viability={best['viability']:.4f} "
                        f"agent={agent_viability if agent_viability is not None else 'n/a'}")

        # Stitch (overlap: later chunk wins)
        full = torch.full((cfg.n_residues,), -1, dtype=torch.long)
        for ci, tok in enumerate(all_tokens):
            start = ci * stride
            L = tok.shape[0]
            full[start:start + L] = tok
        full_sequence = detokenize(full.clamp(min=0))

        seq_path = os.path.join(cfg.output_dir, "full_sequence.txt")
        with open(seq_path, "w") as f:
            f.write(full_sequence)

        return {"n_chunks": n_chunks, "chunk_files": chunk_files,
                "full_sequence": full_sequence, "sequence_path": seq_path}


# =============================================================================
# [4] STAGE 2 — HIERARCHICAL FOLDING (SDA-ONE + MSA enhancement)
# =============================================================================
@dataclass
class Stage2Config:
    input_dir: str = "./results/stage1_design"
    output_dir: str = "./results/stage2_fold"
    use_sda: bool = True
    use_msa_enhancement: bool = True
    msa_mode: str = "adaptive"
    esm_path: Optional[str] = None
    msa_path: Optional[str] = None
    esm_dim: int = 1280
    device: str = "auto"


class Stage2Folder:
    """
    [D] Hierarchical folding: SDA-ONE as the primary physical assembler,
    MSAEnhancedFoldingBridge as an optional evolutionary-enhancement pass.

    Flow:
      1. Load full sequence from Stage 1
      2. If use_msa_enhancement: run MSAEnhancedFoldingBridge → diagnostics +
         (untrained-head) coords + fused features
      3. If use_sda: run StructuralDomainAssembly(sequence) → init_coords
         (this is the physically-grounded structure for large sequences)
      4. Save coords, features, sigma, and MSA diagnostics
    """

    def __init__(self, cfg: Optional[Stage2Config] = None):
        self.cfg = cfg or Stage2Config()
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if self.cfg.device == "auto" else torch.device(self.cfg.device))
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        self.msa_bridge = None
        if self.cfg.use_msa_enhancement and _HAS_MSA:
            self.msa_bridge = MSAEnhancedFoldingBridge(MSABridgeConfig(
                mode=self.cfg.msa_mode, esm_dim=self.cfg.esm_dim, device=self.cfg.device))

        self.sda = None
        if self.cfg.use_sda and _HAS_SDA:
            self.sda = StructuralDomainAssembly(sda_cfg=DomainAssemblyConfig()).to(self.device)

    def _load_sequence(self) -> str:
        seq_path = os.path.join(self.cfg.input_dir, "full_sequence.txt")
        if os.path.exists(seq_path):
            with open(seq_path) as f:
                return f.read().strip()
        # fallback: reconstruct from chunks
        chunk_files = sorted(Path(self.cfg.input_dir).glob("chunk_*.pt"))
        tokens = []
        for cf in chunk_files:
            tokens.append(torch.load(cf)["tokens"])
        if not tokens:
            raise FileNotFoundError("No Stage 1 output found.")
        return detokenize(torch.cat(tokens).clamp(min=0))

    def run(self) -> Dict[str, Any]:
        sequence = self._load_sequence()
        N = len(sequence)
        logger.info(f"[Stage2] Folding sequence of length {N}")

        result: Dict[str, Any] = {"sequence_length": N}

        # --- MSA enhancement pass (optional) ---
        msa_out = None
        if self.msa_bridge is not None:
            try:
                msa_out = self.msa_bridge.fold(sequence, esm_path=self.cfg.esm_path,
                                                msa_path=self.cfg.msa_path)
                result["msa"] = {k: v for k, v in msa_out.items()
                                  if k not in ("predicted_coords", "fused_features")}
                result["msa_coords"] = (msa_out["predicted_coords"].detach().cpu()
                                          if msa_out.get("predicted_coords") is not None else None)
                logger.info(f"[Stage2] MSA pass: use_msa={msa_out['use_msa']} "
                            f"({msa_out['msa_decision_reason']})")
            except Exception as e:
                logger.warning(f"[Stage2] MSA pass failed: {e}")

        # --- SDA-ONE physical assembly (primary structure path) ---
        coords = None
        if self.sda is not None:
            with torch.no_grad():
                sda_out = self.sda(sequence)
            coords = sda_out["init_coords"].detach().cpu()
            result["seq_features"] = sda_out["seq_features"].detach().cpu()
            result["sigma"] = sda_out["sigma"].detach().cpu()
            logger.info(f"[Stage2] SDA-ONE assembled coords: {tuple(coords.shape)}")
        elif msa_out is not None and msa_out.get("predicted_coords") is not None:
            coords = msa_out["predicted_coords"].detach().cpu()
            logger.info("[Stage2] Using MSA-head coords (SDA-ONE unavailable).")

        if coords is None:
            raise RuntimeError("No folding backend available (need SDA-ONE or MSA).")

        torch.save({"coords": coords, "sequence": sequence,
                    "seq_features": result.get("seq_features"),
                    "sigma": result.get("sigma"),
                    "msa_diagnostics": result.get("msa")},
                   os.path.join(self.cfg.output_dir, "fold_result.pt"))
        result["coords"] = coords
        return result


# =============================================================================
# [5] STAGE 3 — OPENMM-ML REFINEMENT (k* = 23 via OPMC)
# =============================================================================
@dataclass
class Stage3Config:
    input_dir: str = "./results/stage2_fold"
    output_dir: str = "./results/stage3_refined"
    ml_potential: str = "ani2x"
    steps: Optional[int] = None       # None → OPMC k* = 23
    lr: float = 1e-4
    w_soc: float = 0.3
    device: str = "auto"


class Stage3Refiner:
    """
    [D] OpenMM-ML refinement via RefinementEngine.refine() from real_fold_one_v2-5.py.
    Iteration count defaults to OPMC k* = 23 (Theorem 11.1).
    """

    def __init__(self, cfg: Optional[Stage3Config] = None):
        if not _HAS_REAL_FOLD:
            raise ImportError("Stage3Refiner requires real_fold_one_v2-5.")
        self.cfg = cfg or Stage3Config()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.k_star = self.cfg.steps if self.cfg.steps else opmc_k_star()
        logger.info(f"[Stage3] Refinement steps = {self.k_star} "
                    f"(OPMC k*={opmc_k_star()} if auto)")

    def _write_ca_pdb(self, coords: torch.Tensor, sequence: str, path: str) -> None:
        coords_np = coords.detach().cpu().numpy()
        N = coords_np.shape[0]
        sequence = (sequence or "").ljust(N, "A")[:N]
        aa_1to3 = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS',
                   'I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN',
                   'R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR','X':'UNK'}
        with open(path, "w") as f:
            f.write("REMARK   1 SESI STAGE3 CA-TRACE\n")
            for i in range(N):
                x, y, z = coords_np[i]
                f.write(f"ATOM  {i+1:5d}  CA  {aa_1to3.get(sequence[i],'UNK'):>3s} A{i+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("END\n")

    def run(self) -> Dict[str, Any]:
        fold_pt = os.path.join(self.cfg.input_dir, "fold_result.pt")
        data = torch.load(fold_pt, map_location="cpu")
        coords, sequence = data["coords"], data["sequence"]

        tmp_pdb = os.path.join(self.cfg.output_dir, "stage2_input.pdb")
        self._write_ca_pdb(coords, sequence, tmp_pdb)

        refine_cfg = RefinementConfig(
            ml_potential=self.cfg.ml_potential,
            use_native_autograd=True,
            steps=self.k_star,
            lr=self.cfg.lr,
            w_soc=self.cfg.w_soc,
            device=self.cfg.device,
        )
        engine = RefinementEngine(refine_cfg)
        try:
            result = engine.refine(tmp_pdb, steps=self.k_star)
            refined = torch.from_numpy(result["solute_coords"]).float()
            out_pdb = os.path.join(self.cfg.output_dir, "refined.pdb")
            engine.export_pdb(result["solute_coords"], out_pdb)
            torch.save({"refined_coords": refined, "sequence": sequence,
                        "final_energy": result["final_energy"],
                        "initial_energy": result["initial_energy"],
                        "energy_history": result["energy_history"]},
                       os.path.join(self.cfg.output_dir, "refined.pt"))
            logger.info(f"[Stage3] Refined: E {result['initial_energy']:.3f} → "
                        f"{result['final_energy']:.3f} kcal/mol")
            return {"refined_coords": refined, "final_energy": result["final_energy"],
                    "initial_energy": result["initial_energy"], "steps": self.k_star}
        finally:
            engine.cleanup()


# =============================================================================
# [6] STAGE 4 — VALIDATION (clash / RMSD / geometry)
# =============================================================================
@dataclass
class Stage4Config:
    input_dir: str = "./results/stage3_refined"
    output_dir: str = "./results/validation"
    reference_pdb: Optional[str] = None


class Stage4Validator:
    """
    [D] Validation: self-consistency geometry (bond lengths, clashes) and,
    if a reference PDB is provided, RMSD via real_fold_one validate_structure.
    """

    def __init__(self, cfg: Optional[Stage4Config] = None):
        self.cfg = cfg or Stage4Config()
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def _geometry_metrics(self, coords: torch.Tensor) -> Dict[str, float]:
        c = coords.float()
        if c.shape[0] < 2:
            return {"n_residues": int(c.shape[0])}
        bonds = torch.norm(c[1:] - c[:-1], dim=-1)
        # clash: any pair < 2.0 Å (excluding sequential)
        if c.shape[0] <= 2000:
            d = torch.cdist(c, c)
            mask = torch.ones_like(d, dtype=torch.bool)
            mask.fill_diagonal_(False)
            for k in (-1, 1):
                idx = torch.arange(d.shape[0])
                j = idx + k
                ok = (j >= 0) & (j < d.shape[0])
                mask[idx[ok], j[ok]] = False
            n_clash = int((d[mask] < 2.0).sum().item())
        else:
            n_clash = -1  # skipped for very large
        return {
            "n_residues": int(c.shape[0]),
            "bond_mean": float(bonds.mean()),
            "bond_std": float(bonds.std()),
            "bond_min": float(bonds.min()),
            "bond_max": float(bonds.max()),
            "n_clashes": n_clash,
        }

    def run(self) -> Dict[str, Any]:
        refined_pt = os.path.join(self.cfg.input_dir, "refined.pt")
        data = torch.load(refined_pt, map_location="cpu")
        coords = data["refined_coords"]

        metrics = self._geometry_metrics(coords)
        metrics["final_energy"] = data.get("final_energy")
        metrics["initial_energy"] = data.get("initial_energy")

        if self.cfg.reference_pdb and _HAS_REAL_FOLD and validate_structure:
            tmp = os.path.join(self.cfg.output_dir, "to_validate.pdb")
            # write coords to pdb for validate_structure
            seq = data.get("sequence", "")
            self._write_pdb(coords, seq, tmp)
            try:
                val = validate_structure(tmp, reference_pdb=self.cfg.reference_pdb, steps=50)
                metrics["openmm_validation"] = val
            except Exception as e:
                logger.warning(f"[Stage4] validate_structure failed: {e}")

        out_path = os.path.join(self.cfg.output_dir, "validation_summary.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"[Stage4] Validation metrics → {out_path}")
        return metrics

    def _write_pdb(self, coords, sequence, path):
        coords_np = coords.detach().cpu().numpy()
        N = coords_np.shape[0]
        sequence = (sequence or "").ljust(N, "A")[:N]
        with open(path, "w") as f:
            for i in range(N):
                x, y, z = coords_np[i]
                f.write(f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            f.write("END\n")


# =============================================================================
# [7] FULL PIPELINE ORCHESTRATOR
# =============================================================================
class FullPipelineOrchestrator:
    """Runs Stage1 → Stage2(+MSA) → Stage3 → Stage4 in sequence."""

    def __init__(self, base_dir: str = "./results",
                 n_residues: int = 1000, use_msa: bool = True):
        self.base = base_dir
        self.n_residues = n_residues
        self.use_msa = use_msa

    def run(self) -> Dict[str, Any]:
        out = {}
        s1 = Stage1Designer(Stage1Config(
            n_residues=self.n_residues, chunk_size=min(500, self.n_residues),
            overlap=20, optimize_steps=10,
            output_dir=os.path.join(self.base, "stage1_design")))
        out["stage1"] = s1.run()

        s2 = Stage2Folder(Stage2Config(
            input_dir=os.path.join(self.base, "stage1_design"),
            output_dir=os.path.join(self.base, "stage2_fold"),
            use_msa_enhancement=self.use_msa))
        out["stage2"] = {k: v for k, v in s2.run().items()
                          if not isinstance(v, torch.Tensor)}

        if _HAS_REAL_FOLD:
            s3 = Stage3Refiner(Stage3Config(
                input_dir=os.path.join(self.base, "stage2_fold"),
                output_dir=os.path.join(self.base, "stage3_refined")))
            out["stage3"] = {k: v for k, v in s3.run().items()
                              if not isinstance(v, torch.Tensor)}
            s4 = Stage4Validator(Stage4Config(
                input_dir=os.path.join(self.base, "stage3_refined"),
                output_dir=os.path.join(self.base, "validation")))
            out["stage4"] = s4.run()
        else:
            out["stage3"] = {"skipped": "real_fold_one_v2 not available"}
            out["stage4"] = {"skipped": "real_fold_one_v2 not available"}
        return out


# =============================================================================
# [8] STANDALONE STAGE SCRIPT GENERATOR
# =============================================================================
def write_stage_scripts(output_dir: str = "./stage_scripts") -> Dict[str, str]:
    """Emit 4 complete standalone stage scripts matching the Slurm pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    files = {}

    files["sesi_100m_design_stage1.py"] = '''#!/usr/bin/env python3
"""Stage 1: De novo sequence design (chunked, distributed)."""
import argparse, os, torch, torch.distributed as dist
from sesi_full_stage_scripts_msa import Stage1Designer, Stage1Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-residues", type=int, default=100_000_000)
    p.add_argument("--chunk-size", type=int, default=10_000)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--output-dir", type=str, default="./results/stage1_design")
    args = p.parse_args()
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl"); rank = dist.get_rank()
    else:
        rank = 0
    cfg = Stage1Config(n_residues=args.n_residues, chunk_size=args.chunk_size,
                       overlap=args.overlap, output_dir=args.output_dir)
    designer = Stage1Designer(cfg)
    result = designer.run()
    if rank == 0:
        print(f"[stage1] designed {result['full_sequence'][:60]}... "
              f"({len(result['full_sequence'])} residues, {result['n_chunks']} chunks)")
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''

    files["sesi_100m_design_stage2.py"] = '''#!/usr/bin/env python3
"""Stage 2: Hierarchical folding (SDA-ONE + MSA enhancement)."""
import argparse, os, torch, torch.distributed as dist
from sesi_full_stage_scripts_msa import Stage2Folder, Stage2Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="./results/stage1_design")
    p.add_argument("--output-dir", type=str, default="./results/stage2_fold")
    p.add_argument("--msa-path", type=str, default=None)
    p.add_argument("--esm-path", type=str, default=None)
    p.add_argument("--no-msa", action="store_true")
    args = p.parse_args()
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
    cfg = Stage2Config(input_dir=args.input_dir, output_dir=args.output_dir,
                       use_msa_enhancement=not args.no_msa,
                       msa_path=args.msa_path, esm_path=args.esm_path)
    folder = Stage2Folder(cfg)
    result = folder.run()
    print(f"[stage2] folded {result['sequence_length']} residues; "
          f"msa_used={result.get('msa', {}).get('use_msa', 'n/a')}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''

    files["sesi_100m_design_stage3.py"] = '''#!/usr/bin/env python3
"""Stage 3: OpenMM-ML refinement (OPMC k* iterations)."""
import argparse, os, torch, torch.distributed as dist
from sesi_full_stage_scripts_msa import Stage3Refiner, Stage3Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="./results/stage2_fold")
    p.add_argument("--output-dir", type=str, default="./results/stage3_refined")
    p.add_argument("--ml-potential", type=str, default="ani2x")
    p.add_argument("--steps", type=int, default=0)  # 0 → OPMC k*
    args = p.parse_args()
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
    cfg = Stage3Config(input_dir=args.input_dir, output_dir=args.output_dir,
                       ml_potential=args.ml_potential,
                       steps=args.steps if args.steps > 0 else None)
    refiner = Stage3Refiner(cfg)
    result = refiner.run()
    print(f"[stage3] refined; E {result['initial_energy']:.3f} -> "
          f"{result['final_energy']:.3f} over {result['steps']} steps")
    if dist.is_available() and dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''

    files["sesi_100m_design_stage4.py"] = '''#!/usr/bin/env python3
"""Stage 4: Validation (clash / RMSD / geometry)."""
import argparse, json
from sesi_full_stage_scripts_msa import Stage4Validator, Stage4Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="./results/stage3_refined")
    p.add_argument("--output-dir", type=str, default="./results/validation")
    p.add_argument("--reference", type=str, default=None)
    args = p.parse_args()
    cfg = Stage4Config(input_dir=args.input_dir, output_dir=args.output_dir,
                       reference_pdb=args.reference)
    validator = Stage4Validator(cfg)
    metrics = validator.run()
    print("[stage4] validation metrics:")
    print(json.dumps(metrics, indent=2, default=str))

if __name__ == "__main__":
    main()
'''

    for name, content in files.items():
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            f.write(content)
        os.chmod(path, 0o755)
        files[name] = path
        logger.info(f"[scripts] wrote {path}")
    return files


# =============================================================================
# [9] SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 78)
    print(f"  SESI FULL STAGE SCRIPTS + MSA INTEGRATION v{STAGE_SUITE_VERSION} — Self-Test")
    print(f"  designer={_HAS_DESIGNER} agent={_HAS_AGENT} sda={_HAS_SDA} "
          f"msa={_HAS_MSA} real_fold={_HAS_REAL_FOLD}")
    print("=" * 78)

    # ---- T1: OPMC constants ----
    print("\n [T1] OPMC constants")
    assert opmc_k_star(1e-8) == 23, opmc_k_star(1e-8)
    assert OPMC_D_MN(4, 4) == 320
    print(f"   k*={opmc_k_star(1e-8)}, d(4,4)={OPMC_D_MN(4,4)}, L={OPMC_L}")
    print("[PASS] T1 OPMC constants match paper")

    # ---- T2: tokenization round-trip ----
    print("\n [T2] Tokenization")
    seq = "ACDEFGHIKLMNPQRSTVWY"
    toks = tokenize_sequence(seq)
    assert detokenize(toks) == seq
    print(f"   round-trip OK for {len(seq)} residues")
    print("[PASS] T2 tokenization")

    # ---- T3: MSA parsing ----
    print("\n [T3] MSA parsing")
    with tempfile.NamedTemporaryFile(suffix=".fasta", mode="w", delete=False) as f:
        f.write(">s1\nACDEFGHIKL\n>s2\nACDEFGH-KL\n>s3\nACDEFGHIKL\n")
        msa_path = f.name
    seqs = parse_msa_file(msa_path)
    assert seqs is not None and len(seqs) == 3
    msa_tok = msa_to_token_tensor(seqs)
    assert msa_tok.shape == (3, 10)
    assert (msa_tok[1] == MSA_GAP).any(), "gap token should be present"
    print(f"   parsed {len(seqs)} seqs, tensor {tuple(msa_tok.shape)}")
    os.unlink(msa_path)
    print("[PASS] T3 MSA parsing (with gap token)")

    # ---- T4: MSAEnhancedFoldingBridge ----
    print("\n [T4] MSAEnhancedFoldingBridge")
    if _HAS_MSA:
        bridge = MSAEnhancedFoldingBridge(MSABridgeConfig(mode="adaptive", esm_dim=64,
                                                             fused_dim=32, hodge_dim=32))
        out = bridge.fold("ACDEFGHIKLMNPQRSTVWY" * 2)  # 40 residues, no MSA/ESM files
        print(f"   use_msa={out['use_msa']} reason='{out['msa_decision_reason']}'")
        if out.get("predicted_coords") is not None:
            print(f"   predicted_coords={tuple(out['predicted_coords'].shape)}")
        print("[PASS] T4 MSAEnhancedFoldingBridge runs (adaptive routing)")
    else:
        print("[SKIP] T4 MSA_ENHANCED_PROTEIN_FOLDING not available")

    # ---- T5: Stage scripts generation ----
    print("\n [T5] Stage script generation")
    scripts = write_stage_scripts(output_dir="./_test_stage_scripts")
    assert len(scripts) == 4
    for name, path in scripts.items():
        assert os.path.exists(path)
        print(f"   {name}")
    import shutil; shutil.rmtree("./_test_stage_scripts", ignore_errors=True)
    print("[PASS] T5 four complete stage scripts generated")

    # ---- T6: Stage1 designer (tiny) ----
    print("\n [T6] Stage1Designer (tiny)")
    if _HAS_DESIGNER:
        s1 = Stage1Designer(Stage1Config(n_residues=60, chunk_size=30, overlap=5,
                                          optimize_steps=3, hidden_dim=32,
                                          use_agent_gate=False,
                                          output_dir="./_test_s1"))
        r1 = s1.run()
        assert len(r1["full_sequence"]) == 60
        print(f"   designed {len(r1['full_sequence'])} residues, {r1['n_chunks']} chunks")
        shutil.rmtree("./_test_s1", ignore_errors=True)
        print("[PASS] T6 Stage1Designer runs end-to-end")
    else:
        print("[SKIP] T6 designer not available")

    # ---- T7: geometry validator ----
    print("\n [T7] Stage4Validator geometry")
    coords = torch.randn(50, 3) * 5
    v = Stage4Validator(Stage4Config(output_dir="./_test_val"))
    gm = v._geometry_metrics(coords)
    print(f"   n_residues={gm['n_residues']}, bond_mean={gm['bond_mean']:.3f}, "
          f"n_clashes={gm['n_clashes']}")
    shutil.rmtree("./_test_val", ignore_errors=True)
    print("[PASS] T7 geometry metrics computed")

    print("\n" + "=" * 78)
    print("  SELF-TEST COMPLETE.")
    print("  HONEST REMINDERS:")
    print("    • MSA folding_head is UNTRAINED — predicted_coords need training to")
    print("      be structurally meaningful; SDA-ONE + RefinementEngine is the")
    print("      physically-grounded structure path.")
    print("    • ESM features are MOCK random unless a real .pt is supplied.")
    print("    • Stage3 requires OpenMM + OpenMM-ML + openmm-torch.")
    print("    • OPMC k*=23 bounds refinement iterations (Theorem 11.1); it does")
    print("      NOT make per-step cost O(1).")
    print("=" * 78)
