# =============================================================================
# REAL FOLD ONE — END-TO-END MSA-FREE PREDICTION PIPELINE
# Integration script: sequence → coarse structure → SGNO refinement →
#                      (optional) physics-based all-atom refinement
# =============================================================================
# Developer    : PAI , Yoon A Limsuwan / MSPS NETWORK
#                MY SOUL MOVE BY POWER OF HOLY SPIRIT
# Organization : MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# License      : MIT
# Year         : 2026
#
# AI Co-Developers:
#   - Claude (Anthropic) — pipeline design, two-tier integration, honest
#                            gap-flagging for the all-atom refinement step
#
# Description:
#   Wires together the three REAL FOLD ONE modules into a single
#   sequence-in → structure-out call, fully MSA-free at every stage:
#
#     Tier 1 (always available — pure PyTorch, no external deps beyond
#             the ecosystem files themselves):
#
#         sequence (str)
#             │
#             ▼
#         SeqToCoarseStructure   (seq_to_coarse_structure.py)
#             │  init_coords, seq_features, sigma
#             ▼
#         StructuralGNOFold      (structural_gno_fold_v3.py)
#             │  final_coords, pred_ddg
#             ▼
#         Cα-only PDB  (write_ca_pdb)
#
#     Tier 2 (optional — requires OpenMM + PDBFixer + side-chain
#             reconstruction; physics-based refinement of the Tier-1
#             output):
#
#         Cα-only PDB
#             │  PDBFixer: add missing heavy atoms + hydrogens
#             ▼
#         All-atom PDB
#             │
#             ▼
#         RefinementEngine.refine(pdb_file=...)   (real_fold_one_v2.py)
#             │
#             ▼
#         Final refined all-atom structure + energy trace
#
# ── IMPORTANT — KNOWN GAP, FLAGGED HONESTLY ─────────────────────────────
#   RefinementEngine._setup_system() calls OpenMMSystemBuilder.build_from_pdb(),
#   which only fills in *missing residues* (chain gaps) via
#   Modeller.addMissingResidues() and *missing hydrogens* via
#   Modeller.addHydrogens() — it does NOT reconstruct missing side-chain
#   heavy atoms within an existing residue. A Cα-only PDB (which is all
#   Tier 1 produces) has no side-chain heavy atoms at all, so handing it
#   to RefinementEngine directly will fail OpenMM's residue-template
#   matching.
#
#   evolution_one_v4.py imports `reconstruct_backbone` and
#   `build_sidechain_atoms` from a module named `one_core_evolution`
#   (no version suffix) — this module was NOT among the files provided
#   for this integration and its implementation is unknown to this
#   script. If that module exists in your environment, plug its
#   side-chain builder into `_add_sidechains()` below instead of the
#   PDBFixer fallback.
#
#   In its absence, this script uses **PDBFixer** (a standard, actively
#   maintained OpenMM companion package: pip install pdbfixer) to build
#   missing side-chain atoms, as the most reliable option to use without
#   fabricating ecosystem-specific functionality.
#
#   Tier 2 is therefore best-effort and clearly separated from Tier 1 —
#   if PDBFixer / OpenMM are unavailable, or if AI-surrogate-only
#   predictions are sufficient for your current purpose, Tier 1 alone is
#   a complete, fully differentiable, MSA-free prediction pipeline.
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

# =============================================================================
# Ecosystem imports — required, not optional, for this integration script
# =============================================================================
try:
    from seq_to_coarse_structure import (
        Seq2CoarseConfig,
        SeqToCoarseStructure,
        build_sgno_compatible_inputs,
        write_ca_pdb,
        get_device,
    )
except ImportError as exc:
    raise ImportError(
        "seq_to_coarse_structure.py must be importable (same directory or "
        "on PYTHONPATH) — this integration script cannot run without it."
    ) from exc

try:
    from structural_gno_fold_v3 import SGNOConfig, StructuralGNOFold
except ImportError as exc:
    raise ImportError(
        "structural_gno_fold_v3.py must be importable — this integration "
        "script cannot run without it."
    ) from exc

# Tier-2 dependency: optional.
try:
    from real_fold_one_v2 import RefinementEngine, RefinementConfig
    _HAS_REFINEMENT_ENGINE = True
except ImportError:
    _HAS_REFINEMENT_ENGINE = False
    RefinementEngine = None   # type: ignore[assignment]
    RefinementConfig = None   # type: ignore[assignment]

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    _HAS_PDBFIXER = True
except ImportError:
    _HAS_PDBFIXER = False
    PDBFixer = None   # type: ignore[assignment]
    PDBFile = None     # type: ignore[assignment]

PIPELINE_VERSION: str = "1.0.0"


# =============================================================================
# 1.  Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Top-level configuration for the end-to-end MSA-free pipeline.

    Args:
        seq2coarse_cfg   : Seq2CoarseConfig for the sequence → coarse-
                            structure stage. If None, a default config is
                            built with hidden_dim matching sgno_hidden_dim
                            (see __post_init__) so the two modules are
                            shape-compatible without manual wiring.
        sgno_cfg         : SGNOConfig for the refinement stage. If None,
                            a default config is built with
                            node_in_dim = seq2coarse_cfg.hidden_dim.
        run_tier2        : whether to attempt physics-based all-atom
                            refinement after the AI-surrogate stage.
                            Silently downgrades to Tier-1-only with a
                            warning if OpenMM / PDBFixer / RefinementEngine
                            are unavailable.
        tier2_steps      : optimisation steps for RefinementEngine.refine
                            if Tier 2 runs.
        device           : "auto" | "cuda" | "mps" | "cpu".
        output_dir       : directory for intermediate and final PDB files.
        sgno_checkpoint  : optional path to a pretrained StructuralGNOFold
                            checkpoint (as saved by its Trainer). If None,
                            randomly initialised weights are used — fine
                            for architecture/shape verification, not for
                            real structure prediction.
        seq2coarse_checkpoint : optional path to a pretrained
                            SeqToCoarseStructure checkpoint.
    """

    seq2coarse_cfg: Optional[Seq2CoarseConfig] = None
    sgno_cfg:       Optional[SGNOConfig]       = None

    run_tier2:   bool = False
    tier2_steps: int  = 600

    device:     str = "auto"
    output_dir: str = "./msa_free_pipeline_outputs"

    sgno_checkpoint:        Optional[str] = None
    seq2coarse_checkpoint:  Optional[str] = None

    def __post_init__(self) -> None:
        if self.seq2coarse_cfg is None:
            self.seq2coarse_cfg = Seq2CoarseConfig()
        if self.sgno_cfg is None:
            # Critical wiring point: StructuralGNOFold must accept the
            # contextualised transformer latent produced by
            # SeqToCoarseStructure, not the architecture's own default
            # 20-dim one-hot. See structural_gno_fold_v3.SGNOConfig.node_in_dim.
            self.sgno_cfg = SGNOConfig(node_in_dim=self.seq2coarse_cfg.hidden_dim)
        elif self.sgno_cfg.node_in_dim != self.seq2coarse_cfg.hidden_dim:
            raise ValueError(
                f"sgno_cfg.node_in_dim ({self.sgno_cfg.node_in_dim}) must equal "
                f"seq2coarse_cfg.hidden_dim ({self.seq2coarse_cfg.hidden_dim}) — "
                "StructuralGNOFold's node embedding expects the exact width of "
                "the seq_features tensor SeqToCoarseStructure produces."
            )


# =============================================================================
# 2.  Pipeline Orchestrator
# =============================================================================

class MSAFreePipeline:
    """
    End-to-end, single-sequence, MSA-free structure prediction pipeline.

    Tier 1 (AI surrogate, always available):
        sequence → SeqToCoarseStructure → StructuralGNOFold → Cα coordinates

    Tier 2 (physics-based refinement, optional / best-effort):
        Cα coordinates → side-chain reconstruction → RefinementEngine

    Args:
        cfg : PipelineConfig instance.
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.device = get_device(self.cfg.device if self.cfg.device != "auto" else "cuda")

        self.seq2coarse = SeqToCoarseStructure(self.cfg.seq2coarse_cfg).to(self.device)
        self.sgno = StructuralGNOFold(self.cfg.sgno_cfg).to(self.device)

        if self.cfg.seq2coarse_checkpoint:
            self._load_seq2coarse_checkpoint(self.cfg.seq2coarse_checkpoint)
        if self.cfg.sgno_checkpoint:
            self._load_sgno_checkpoint(self.cfg.sgno_checkpoint)

        self.seq2coarse.eval()
        self.sgno.eval()

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg.run_tier2 and not (_HAS_REFINEMENT_ENGINE and _HAS_PDBFIXER):
            missing = []
            if not _HAS_REFINEMENT_ENGINE:
                missing.append("real_fold_one_v2.RefinementEngine (+ OpenMM)")
            if not _HAS_PDBFIXER:
                missing.append("pdbfixer")
            warnings.warn(
                f"run_tier2=True but missing: {', '.join(missing)}. "
                "Falling back to Tier-1-only (AI surrogate) output."
            )
            self.cfg.run_tier2 = False

        logger.info(
            "MSAFreePipeline v%s ready | device=%s | tier2=%s",
            PIPELINE_VERSION, self.device, self.cfg.run_tier2,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_seq2coarse_checkpoint(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self.seq2coarse.load_state_dict(data["model_state"])
        logger.info("Loaded SeqToCoarseStructure checkpoint ← %s", path)

    def _load_sgno_checkpoint(self, path: str) -> None:
        # Matches the exact key used by structural_gno_fold_v3's own
        # Trainer.save_checkpoint (always "model_state").
        data = torch.load(path, map_location=self.device)
        self.sgno.load_state_dict(data["model_state"])
        logger.info("Loaded StructuralGNOFold checkpoint ← %s", path)

    # ------------------------------------------------------------------
    # Tier 1 — AI surrogate, sequence → coarse → refined Cα coordinates
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_tier1(self, sequence: str, name: str = "query") -> Dict[str, Any]:
        """
        Args:
            sequence : raw single-letter amino-acid string (no MSA).
            name     : identifier used for output filenames.
        Returns:
            Dict with:
                "coarse_coords" : (N, 3) — SeqToCoarseStructure output.
                "final_coords"  : (N, 3) — StructuralGNOFold-refined output.
                "pred_ddg"      : (1,)   — predicted ΔΔG (kcal/mol).
                "sigma"         : (N, 1) — structural-regime field.
                "coarse_pdb"    : Path to the coarse-stage Cα PDB.
                "refined_pdb"   : Path to the SGNO-refined Cα PDB.
        """
        n = len(sequence)
        logger.info("Tier 1 | sequence length=%d | name=%s", n, name)

        s2c_out = self.seq2coarse(sequence)
        seq_features, init_coords, sigma = build_sgno_compatible_inputs(s2c_out)

        final_coords, pred_ddg = self.sgno(seq_features, init_coords, sigma)

        coarse_pdb = write_ca_pdb(
            sequence, init_coords, self.output_dir / f"{name}_coarse.pdb"
        )
        refined_pdb = write_ca_pdb(
            sequence, final_coords, self.output_dir / f"{name}_sgno_refined.pdb"
        )

        logger.info(
            "Tier 1 complete | pred_ddg=%.3f kcal/mol | coarse→refined RMSD=%.3f Å",
            float(pred_ddg.item()),
            float(torch.linalg.norm(final_coords - init_coords, dim=-1).mean().item()),
        )

        return {
            "coarse_coords": init_coords,
            "final_coords":  final_coords,
            "pred_ddg":      pred_ddg,
            "sigma":         sigma,
            "coarse_pdb":    coarse_pdb,
            "refined_pdb":   refined_pdb,
        }

    # ------------------------------------------------------------------
    # Tier 2 — physics-based all-atom refinement (best-effort)
    # ------------------------------------------------------------------

    def _add_sidechains(self, ca_pdb_path: Path, name: str) -> Path:
        """
        Reconstruct side-chain heavy atoms + hydrogens from a Cα-only PDB
        using PDBFixer, producing an all-atom PDB that
        ``OpenMMSystemBuilder.build_from_pdb`` can parameterise.

        NOTE: PDBFixer reconstructs side chains via rotamer-library
        placement based on backbone geometry; it does not know anything
        about the predicted structure's actual side-chain packing, so
        Tier 2's energetic refinement partly compensates for this but
        results should be treated as an approximate physical relaxation
        of the Tier-1 backbone, not a definitive all-atom prediction.

        If a project-specific side-chain builder (e.g. the
        `build_sidechain_atoms` function referenced by
        `evolution_one_v4.py` from a `one_core_evolution` module not
        included in this integration) is available in your environment,
        replace this method's body with a call to that function instead.

        Args:
            ca_pdb_path : path to the Cα-only PDB (Tier 1 output).
            name        : identifier used for the output filename.
        Returns:
            Path to the all-atom PDB.
        """
        if not _HAS_PDBFIXER:
            raise RuntimeError(
                "PDBFixer not installed. Install with: pip install pdbfixer"
            )

        fixer = PDBFixer(filename=str(ca_pdb_path))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        out_path = self.output_dir / f"{name}_allatom.pdb"
        with open(out_path, "w") as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh)

        logger.info("Tier 2 | side-chain reconstruction → %s", out_path)
        return out_path

    def run_tier2(
        self,
        sequence: str,
        tier1_output: Dict[str, Any],
        name: str = "query",
    ) -> Optional[Dict[str, Any]]:
        """
        Run physics-based refinement on the Tier-1 output via
        RefinementEngine. Returns None (with a warning already issued at
        __init__ time) if the required dependencies are unavailable.

        Args:
            sequence     : raw amino-acid string (must match tier1_output).
            tier1_output : dict returned by run_tier1.
            name         : identifier used for output filenames.
        Returns:
            Dict with RefinementEngine.refine's return value
            ('solute_coords', 'energy_history', 'final_energy', ...),
            or None if Tier 2 is unavailable.
        """
        if not self.cfg.run_tier2:
            logger.info("Tier 2 skipped (disabled or dependencies unavailable).")
            return None

        allatom_pdb = self._add_sidechains(tier1_output["refined_pdb"], name)

        refine_cfg = RefinementConfig(device=str(self.device), steps=self.cfg.tier2_steps)
        engine = RefinementEngine(refine_cfg)
        try:
            result = engine.refine(str(allatom_pdb))
        finally:
            if hasattr(engine, "cleanup"):
                engine.cleanup()

        logger.info(
            "Tier 2 complete | final_energy=%.2f kcal/mol (Δ=%.2f from initial)",
            result["final_energy"], result["final_energy"] - result["initial_energy"],
        )
        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def predict(self, sequence: str, name: str = "query") -> Dict[str, Any]:
        """
        Run the full pipeline (Tier 1, then Tier 2 if enabled and available).

        Args:
            sequence : raw amino-acid string.
            name     : identifier used for output filenames.
        Returns:
            Dict merging Tier 1 output under "tier1" and Tier 2 output
            (or None) under "tier2".
        """
        tier1 = self.run_tier1(sequence, name=name)
        tier2 = self.run_tier2(sequence, tier1, name=name) if self.cfg.run_tier2 else None
        return {"tier1": tier1, "tier2": tier2}


# =============================================================================
# 3.  CLI Entrypoint
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="REAL FOLD ONE — end-to-end MSA-free structure prediction."
    )
    p.add_argument("sequence", type=str, help="Single-letter amino-acid sequence.")
    p.add_argument("--name", type=str, default="query", help="Output filename prefix.")
    p.add_argument("--output-dir", type=str, default="./msa_free_pipeline_outputs")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--tier2", action="store_true", help="Attempt physics-based all-atom refinement.")
    p.add_argument("--tier2-steps", type=int, default=600)
    p.add_argument("--seq2coarse-checkpoint", type=str, default=None)
    p.add_argument("--sgno-checkpoint", type=str, default=None)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = PipelineConfig(
        run_tier2=args.tier2,
        tier2_steps=args.tier2_steps,
        device=args.device,
        output_dir=args.output_dir,
        sgno_checkpoint=args.sgno_checkpoint,
        seq2coarse_checkpoint=args.seq2coarse_checkpoint,
    )
    pipeline = MSAFreePipeline(cfg)
    result = pipeline.predict(args.sequence, name=args.name)

    summary = {
        "name": args.name,
        "sequence_length": len(args.sequence),
        "pred_ddg_kcal_mol": float(result["tier1"]["pred_ddg"].item()),
        "coarse_pdb": str(result["tier1"]["coarse_pdb"]),
        "refined_pdb": str(result["tier1"]["refined_pdb"]),
        "tier2_ran": result["tier2"] is not None,
    }
    if result["tier2"] is not None:
        summary["tier2_final_energy_kcal_mol"] = result["tier2"]["final_energy"]
        summary["tier2_initial_energy_kcal_mol"] = result["tier2"]["initial_energy"]

    print(json.dumps(summary, indent=2))


# =============================================================================
# 4.  Verification Suite
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    print("=" * 70)
    print("  MSA-Free Pipeline — Integration Smoke Test")
    print(f"  PIPELINE_VERSION = {PIPELINE_VERSION}")
    print(f"  Tier-2 deps available: RefinementEngine={_HAS_REFINEMENT_ENGINE}, "
          f"PDBFixer={_HAS_PDBFIXER}")
    print("=" * 70)

    _seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK"[:48]

    _cfg = PipelineConfig(
        seq2coarse_cfg=Seq2CoarseConfig(
            embed_backend="learned",   # dependency-free for the smoke test
            embed_dim=64, hidden_dim=64, num_heads=4, num_layers=2,
            ffn_dim=128, num_distance_bins=32, mds_iters=50, max_seq_len=256,
        ),
        sgno_cfg=SGNOConfig(node_in_dim=64, hidden_dim=64, num_layers=3),
        run_tier2=False,   # Tier 2 requires OpenMM + PDBFixer + a real PDB target
        output_dir="/tmp/msa_free_pipeline_smoketest",
    )

    pipeline = MSAFreePipeline(_cfg)
    print(f"[PASS] Pipeline initialised | device={pipeline.device}")

    out = pipeline.run_tier1(_seq, name="smoketest")
    N = len(_seq)
    assert out["coarse_coords"].shape == (N, 3)
    assert out["final_coords"].shape == (N, 3)
    assert out["pred_ddg"].shape == (1,)
    assert out["sigma"].shape == (N, 1)
    assert out["coarse_pdb"].exists()
    assert out["refined_pdb"].exists()
    print(f"[PASS] Tier 1 end-to-end | coarse {out['coarse_coords'].shape} → "
          f"refined {out['final_coords'].shape}, ddg={out['pred_ddg'].item():.3f}")

    full = pipeline.predict(_seq, name="smoketest_full")
    assert full["tier1"] is not None
    assert full["tier2"] is None  # run_tier2=False in this smoke test
    print("[PASS] predict() — Tier 1 only (Tier 2 disabled as configured)")

    print("=" * 70)
    print("  All tests passed.")
    print(f"  Run with a real sequence via:")
    print(f"    python run_msa_free_pipeline.py <SEQUENCE> --name my_protein")
    print(f"  Add --tier2 to attempt physics-based all-atom refinement")
    print(f"  (requires: pip install openmm pdbfixer).")
    print("=" * 70)
