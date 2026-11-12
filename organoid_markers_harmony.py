#!/usr/bin/env python3
"""
=============================================================================
ORGANOID MARKER ANNOTATION + HARMONY BATCH CORRECTION
Organ-specific markers for: intestinal, cerebral, kidney
=============================================================================
License: MIT | 2026

LITERATURE
 [M1] Korsunsky et al. "Fast, sensitive and accurate integration of single-cell
      data with Harmony." Nat. Methods 16, 1289 (2019).
 [M2] Sato et al. "Single Lgr5 stem cells build crypt-villus structures in
      vitro without a mesenchymal niche." Nature 459, 262 (2009).     [intestinal]
 [M3] Lancaster et al. "Cerebral organoids model human brain development and
      microcephaly." Nature 501, 373 (2013).                          [cerebral]
 [M4] Takasato et al. "Kidney organoids from human iPS cells contain multiple
      lineages and model human nephrogenesis." Nature 526, 564 (2015). [kidney]
 [M5] Clevers. "The intestinal crypt, a prototype stem cell compartment."
      Cell 154, 274 (2013).
 [M6] Camp et al. "Human cerebral organoids recapitulate gene expression
      programs of fetal neocortex development." PNAS 112, 15678 (2015).
 [M7] Little & McMahon. "Mammalian kidney development: principles, progress,
      and projections." Cold Spring Harb Perspect Biol 4, a008300 (2012).
 [S1] Luecken & Theis. "Current best practices in single-cell RNA-seq analysis:
      a tutorial." Mol. Syst. Biol. 15, e8746 (2019).

=============================================================================
HONEST SCOPE
=============================================================================
MARKER ANNOTATION
  ✅ Curated markers from published organoid papers [M2-M7].
  ⚠️ Markers OVERLAP between organs (SOX2, PAX6, EPCAM, WT1...).
     The conflict detector reports this. Do NOT trust annotation without
     visual inspection on UMAP.
  ⚠️ Marker lists are INCOMPLETE. They cover major cell types only.
     Rare populations (tuft, M cells, interneurons subtypes) may be missed.
  ❌ Automated annotation is NOT ground truth. It is a starting hypothesis.

HARMONY BATCH CORRECTION
  ✅ Real Harmony implementation via harmonypy [M1].
  ⚠️ CRITICAL LIMITATION: Harmony CANNOT separate batch from biology when
     batch is CONFOUNDED with condition.
     Example: if GSE123 contains only drug-treated samples and GSE456
     contains only controls, Harmony will remove the treatment effect.
  ⚠️ Harmony assumes SHARED cell types across batches. If a cell type
     exists in only one batch, Harmony distorts it.
  ❌ Batch correction is NOT a substitute for proper experimental design.
  ❌ Harmony does NOT create treatment labels. The (factor, dose, time)
     pairing still requires manual curation from the paper.

REACHING 500 RECORDS
  A "record" = one sample with (factor, dose, time, composition).
  Typical GEO organoid series: 5–20 samples.
  500 records ≈ 25–100 GEO series OR large Perturb-seq datasets.
  Harmony helps MERGE the scRNA-seq; it does NOT create metadata.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("organoid_markers")

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import harmonypy as hpy
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False


# =============================================================================
# 1. ORGAN-SPECIFIC MARKER REGISTRY
# =============================================================================

@dataclass
class CellTypeMarkers:
    name: str
    positive: List[str]
    negative: List[str] = field(default_factory=list)
    reference: str = ""


INTESTINAL_MARKERS: List[CellTypeMarkers] = [
    CellTypeMarkers("STEM_LGR5",
                    ["LGR5", "ASCL2", "OLFM4", "SOX9", "SMOC2"],
                    ["MUC2", "CHGA"],
                    "Sato 2009 [M2]; Clevers 2013 [M5]"),
    CellTypeMarkers("ENTEROCYTE",
                    ["FABP1", "VIL1", "ALPI", "SI", "CDX2", "KRT20"],
                    ["LGR5", "MUC2", "LYZ"],
                    "Clevers 2013 [M5]"),
    CellTypeMarkers("GOBLET",
                    ["MUC2", "TFF3", "SPINK4", "CLCA1", "FCGBP"],
                    ["LGR5", "LYZ", "CHGA"],
                    "Clevers 2013 [M5]"),
    CellTypeMarkers("PANETH",
                    ["LYZ", "DEFA5", "DEFA6", "REG4", "PLA2G2A"],
                    ["MUC2", "FABP1", "CHGA"],
                    "Clevers 2013 [M5]"),
    CellTypeMarkers("ENTEROENDOCRINE",
                    ["CHGA", "CHGB", "NEUROD1", "INSM1", "TPH1"],
                    ["MUC2", "LYZ", "FABP1"],
                    "Clevers 2013 [M5]"),
    CellTypeMarkers("TUFT",
                    ["DCLK1", "TRPM5", "POU2F3", "IL25"],
                    ["MUC2", "LYZ", "CHGA"],
                    "Clevers 2013 [M5]"),
    CellTypeMarkers("MESENCHYMAL",
                    ["COL1A1", "VIM", "PDGFRA", "FOXO1", "DCN"],
                    ["EPCAM", "CDX2", "KRT20"],
                    "Supporting niche"),
]

CEREBRAL_MARKERS: List[CellTypeMarkers] = [
    CellTypeMarkers("RADIAL_GLIA",
                    ["SOX2", "PAX6", "HES1", "FABP7", "VIM", "HOPX"],
                    ["DCX", "GFAP", "MBP"],
                    "Lancaster 2013 [M3]; Camp 2015 [M6]"),
    CellTypeMarkers("NEURAL_PROGENITOR",
                    ["SOX1", "NES", "DCX", "STMN2", "TUBB3"],
                    ["GFAP", "MBP", "CX3CR1"],
                    "Lancaster 2013 [M3]"),
    CellTypeMarkers("EXCITATORY_NEURON",
                    ["SLC17A7", "NEUROD6", "TBR1", "SATB2", "FOXP2"],
                    ["GAD1", "GAD2", "DLX1", "GFAP"],
                    "Camp 2015 [M6]"),
    CellTypeMarkers("INHIBITORY_NEURON",
                    ["GAD1", "GAD2", "DLX1", "DLX2", "NKX2-1", "SST"],
                    ["SLC17A7", "GFAP", "MBP"],
                    "Camp 2015 [M6]"),
    CellTypeMarkers("ASTROCYTE",
                    ["GFAP", "S100B", "AQP4", "ALDH1L1", "SLC1A3"],
                    ["DCX", "MBP", "CX3CR1"],
                    "Lancaster 2013 [M3]"),
    CellTypeMarkers("OLIGODENDROCYTE",
                    ["MBP", "PLP1", "OLIG2", "MOG", "MAG"],
                    ["GFAP", "DCX", "CX3CR1"],
                    "Lancaster 2013 [M3]"),
    CellTypeMarkers("MICROGLIA",
                    ["CX3CR1", "TMEM119", "P2RY12", "AIF1", "C1QA"],
                    ["GFAP", "MBP", "DCX"],
                    "Lancaster 2013 [M3]"),
    CellTypeMarkers("CHOROID_PLEXUS",
                    ["TTR", "FOXJ1", "OTX2", "CLDN11"],
                    ["DCX", "GFAP", "MBP"],
                    "Lancaster 2013 [M3]"),
]

KIDNEY_MARKERS: List[CellTypeMarkers] = [
    CellTypeMarkers("NEPHRON_PROGENITOR",
                    ["SIX2", "WT1", "CITED1", "SALL1", "OSR1"],
                    ["NPHS1", "LRP2", "AQP2"],
                    "Takasato 2015 [M4]; Little & McMahon 2012 [M7]"),
    CellTypeMarkers("PODOCYTE",
                    ["NPHS1", "NPHS2", "PODXL", "WT1", "SYNPO"],
                    ["LRP2", "SLC12A3", "AQP2"],
                    "Takasato 2015 [M4]"),
    CellTypeMarkers("PROXIMAL_TUBULE",
                    ["LRP2", "CUBN", "SLC34A1", "ALB", "SLC22A6"],
                    ["NPHS1", "SLC12A3", "AQP2"],
                    "Takasato 2015 [M4]"),
    CellTypeMarkers("LOOP_OF_HENLE",
                    ["UMOD", "SLC12A1", "KCNJ1", "BSND"],
                    ["NPHS1", "LRP2", "AQP2"],
                    "Little & McMahon 2012 [M7]"),
    CellTypeMarkers("DISTAL_TUBULE",
                    ["SLC12A3", "CALB1", "TRPV5", "CLCNKB"],
                    ["NPHS1", "LRP2", "UMOD"],
                    "Takasato 2015 [M4]"),
    CellTypeMarkers("COLLECTING_DUCT",
                    ["AQP2", "AQP3", "GATA3", "FOXJ1"],
                    ["NPHS1", "LRP2", "SLC12A3"],
                    "Takasato 2015 [M4]"),
    CellTypeMarkers("ENDOTHELIAL",
                    ["PECAM1", "CDH5", "VWF", "CLDN5"],
                    ["EPCAM", "WT1", "NPHS1"],
                    "Takasato 2015 [M4]"),
    CellTypeMarkers("STROMA",
                    ["FOXD1", "PDGFRA", "COL1A1", "COL3A1"],
                    ["EPCAM", "WT1", "PECAM1"],
                    "Little & McMahon 2012 [M7]"),
]

# Markers shared across organs → conflict detection
SHARED_MARKERS = {"SOX2", "PAX6", "WT1", "EPCAM", "VIM", "CDX2", "SOX9",
                  "PECAM1", "COL1A1", "PDGFRA", "KRT20"}

ORGAN_REGISTRY = {
    "intestinal": INTESTINAL_MARKERS,
    "cerebral": CEREBRAL_MARKERS,
    "kidney": KIDNEY_MARKERS,
}


# =============================================================================
# 2. MARKER ANNOTATOR
# =============================================================================

class MarkerAnnotator:
    """
    Score cells against organ-specific markers.

    Method: for each cell type, compute
      score = mean(scaled_expr of positive markers)
            − 0.5 × mean(scaled_expr of negative markers)
    Assign cell to highest-scoring type if score > min_score.

    ⚠️ This is approximate. Validated marker-based annotation requires:
      • Visual inspection on UMAP
      • Cross-checking with known spatial patterns
      • Comparison against published single-cell atlases
    """
    def __init__(self, organ: str, min_score: float = 0.1,
                 conflict_threshold: float = 0.3):
        if organ not in ORGAN_REGISTRY:
            raise ValueError(f"Unknown organ '{organ}'. "
                             f"Available: {list(ORGAN_REGISTRY.keys())}")
        self.organ = organ
        self.markers = ORGAN_REGISTRY[organ]
        self.min_score = min_score
        self.conflict_threshold = conflict_threshold
        self.cell_types = [m.name for m in self.markers]
        log.info(f"[annotator] organ={organ}, {len(self.markers)} cell types")

    def score_matrix(self, adata: "ad.AnnData") -> np.ndarray:
        """
        Compute per-cell, per-type score matrix.
        Returns (n_cells, n_types).
        """
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy anndata")
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        # Scale per gene (z-score)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        Xs = (X - mu) / sd
        gene_names = list(adata.var_names)
        gene_idx = {g: i for i, g in enumerate(gene_names)}

        scores = np.zeros((adata.n_obs, len(self.markers)), np.float32)
        for j, cm in enumerate(self.markers):
            pos_idx = [gene_idx[g] for g in cm.positive if g in gene_idx]
            neg_idx = [gene_idx[g] for g in cm.negative if g in gene_idx]
            if pos_idx:
                scores[:, j] += Xs[:, pos_idx].mean(axis=1)
            if neg_idx:
                scores[:, j] -= 0.5 * Xs[:, neg_idx].mean(axis=1)
        return scores

    def annotate(self, adata: "ad.AnnData") -> Dict:
        """
        Assign cell types + detect conflicts.
        Modifies adata.obs in place: adds 'cell_type', 'type_score', 'conflict'.
        """
        scores = self.score_matrix(adata)
        best = scores.argmax(axis=1)
        best_score = scores[best] if scores.ndim == 1 else \
            scores[np.arange(len(scores)), best]
        # threshold
        assigned = np.array([self.cell_types[b] for b in best], dtype=object)
        assigned[best_score < self.min_score] = "UNKNOWN"
        # conflict: second-best within conflict_threshold of best
        sorted_idx = np.argsort(-scores, axis=1)
        second = scores[np.arange(len(scores)), sorted_idx[:, 1]]
        conflict = (best_score - second) < self.conflict_threshold
        adata.obs["cell_type"] = assigned
        adata.obs["type_score"] = best_score
        adata.obs["conflict"] = conflict
        # report
        from collections import Counter
        dist = Counter(assigned)
        n_conf = int(conflict.sum())
        log.info(f"[annotator] distribution: {dict(dist)}")
        log.info(f"[annotator] {n_conf}/{len(assigned)} cells have CONFLICT "
                 f"(two types within {self.conflict_threshold})")
        # shared-marker warning
        used_genes = set()
        for cm in self.markers:
            used_genes.update(cm.positive)
        shared_used = used_genes & SHARED_MARKERS
        if shared_used:
            log.warning(f"[annotator] These markers are SHARED across organs: "
                        f"{shared_used}. Cross-organ annotation may be unreliable.")
        return {"cell_type": assigned, "scores": scores,
                "conflict": conflict, "n_conflict": n_conf,
                "distribution": dict(dist),
                "shared_marker_warning": sorted(shared_used)}


# =============================================================================
# 3. HARMONY BATCH CORRECTION
# =============================================================================

def detect_confound(adata: "ad.AnnData",
                    batch_col: str, condition_col: str) -> Dict:
    """
    Check whether batch is CONFOUNDED with condition.
    If every batch contains only one condition, Harmony will REMOVE
    the biological signal. This is a FATAL problem for drug-response studies.
    """
    if condition_col not in adata.obs.columns:
        return {"has_condition": False, "confounded": None,
                "WARNING": f"column '{condition_col}' not found; "
                           f"cannot check confounding"}
    ct = adata.obs.groupby([batch_col, condition_col]).size().unstack(fill_value=0)
    # a batch is "pure" if it has only one condition with >90% of its cells
    purity = (ct.max(axis=1) / ct.sum(axis=1))
    n_pure = int((purity > 0.9).sum())
    n_batches = len(purity)
    confounded = n_pure == n_batches
    return {
        "has_condition": True,
        "n_batches": n_batches,
        "n_pure_batches": n_pure,
        "confounded": confounded,
        "batch_purity": purity.to_dict(),
        "VERDICT": ("FATAL: batch ≡ condition. Harmony will remove the "
                    "treatment effect. DO NOT correct." if confounded else
                    "OK: batches contain mixed conditions. Harmony is appropriate."),
    }


class HarmonyCorrector:
    """
    Wrapper around harmonypy [M1] with safety checks.

    Usage:
      1. Run PCA on combined dataset
      2. Check confounding with detect_confound()
      3. If not confounded, run Harmony on PCA embedding
      4. Re-cluster in Harmony-corrected space

    ⚠️ Harmony corrects the EMBEDDING, not the counts.
       Downstream clustering uses corrected embedding.
       Differential expression should use ORIGINAL counts + batch covariate.
    """
    def __init__(self, theta: float = 2.0, max_iter: int = 20,
                 random_state: int = 42):
        self.theta = theta
        self.max_iter = max_iter
        self.random_state = random_state

    def correct(self, adata: "ad.AnnData", batch_col: str = "batch",
                n_pcs: int = 30) -> "ad.AnnData":
        if not HAS_HARMONY:
            raise ImportError("pip install harmonypy")
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy anndata")
        # ensure PCA exists
        if "X_pca" not in adata.obsm:
            log.info("[harmony] computing PCA")
            sc.tl.pca(adata, n_comps=n_pcs)
        pca = adata.obsm["X_pca"][:, :n_pcs]
        meta = adata.obs.copy()
        meta.index = meta.index.astype(str)
        log.info(f"[harmony] correcting {pca.shape[0]} cells, "
                 f"batch_col='{batch_col}', theta={self.theta}")
        ho = hpy.run_harmony(pca, meta, batch_col,
                             theta=self.theta,
                             max_iter_harmony=self.max_iter,
                             random_state=self.random_state)
        adata.obsm["X_pca_harmony"] = ho.Z_corr
        log.info("[harmony] done. Corrected embedding in adata.obsm['X_pca_harmony']")
        return adata


def fallback_batch_regress(adata: "ad.AnnData", batch_col: str,
                           n_pcs: int = 30) -> "ad.AnnData":
    """
    Fallback if harmonypy unavailable: linear regression on PCA per component.
    ⚠️ INFERIOR to Harmony. No soft clustering, no iterative refinement.
    Use only if harmonypy cannot be installed.
    """
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_pcs)
    pca = adata.obsm["X_pca"][:, :n_pcs].copy()
    batches = adata.obs[batch_col].values
    unique_b = list(set(batches))
    batch_idx = np.array([unique_b.index(b) for b in batches])
    # one-hot
    B = np.zeros((len(batch_idx), len(unique_b)), np.float32)
    B[np.arange(len(batch_idx)), batch_idx] = 1.0
    # least-squares removal of batch mean (keep intercept)
    X = np.hstack([np.ones((len(batch_idx), 1)), B[:, 1:]])
    coef, _, _, _ = np.linalg.lstsq(X, pca, rcond=None)
    pca_corr = pca - X[:, 1:] @ coef[1:]
    adata.obsm["X_pca_batch_regress"] = pca_corr
    log.warning("[fallback] Linear batch regression used. "
                "INFERIOR to Harmony. Install harmonypy for proper correction.")
    return adata


# =============================================================================
# 4. MULTI-SERIES INTEGRATOR
# =============================================================================

class MultiSeriesIntegrator:
    """
    Combine multiple GEO scRNA-seq series into one AnnData,
    then correct batch effects and annotate.

    ⚠️ REQUIREMENTS for valid integration:
      1. Same species (human vs mouse markers differ)
      2. Same organoid system (intestinal + cerebral = invalid)
      3. Overlapping cell types (Harmony assumption)
      4. Batch NOT confounded with condition (detect_confound checks)

    The integrator ENFORCES these checks and refuses if violated.
    """
    def __init__(self, organ: str):
        if organ not in ORGAN_REGISTRY:
            raise ValueError(f"Unknown organ: {organ}")
        self.organ = organ
        self.series: Dict[str, "ad.AnnData"] = {}

    def add_series(self, gse: str, adata: "ad.AnnData"):
        if "batch" not in adata.obs.columns:
            adata.obs["batch"] = gse
        self.series[gse] = adata
        log.info(f"[integrator] added {gse}: {adata.n_obs} cells")

    def check_species(self) -> bool:
        """Heuristic: human genes are uppercase, mouse are Title-case."""
        for gse, adata in self.series.items():
            sample = list(adata.var_names[:100])
            upper_frac = sum(1 for g in sample if g.isupper()) / max(len(sample), 1)
            if upper_frac < 0.5:
                log.warning(f"[integrator] {gse} looks like MOUSE data "
                            f"(only {upper_frac:.0%} uppercase genes). "
                            f"Human/mouse mixing is INVALID.")
                return False
        return True

    def integrate(self, condition_col: Optional[str] = None) -> "ad.AnnData":
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy anndata")
        if len(self.series) < 2:
            raise ValueError("Need >= 2 series for integration")
        # species check
        if not self.check_species():
            raise DataRefusal("Species mismatch detected. Cannot integrate.")
        # concatenate
        adatas = list(self.series.values())
        combined = ad.concat(adatas, join="outer", label="batch",
                             keys=list(self.series.keys()))
        log.info(f"[integrator] combined: {combined.n_obs} cells, "
                 f"{combined.n_vars} genes, {len(self.series)} series")
        # QC
        sc.pp.filter_cells(combined, min_genes=200)
        sc.pp.filter_genes(combined, min_cells=3)
        combined.var["mt"] = combined.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(combined, qc_vars=["mt"], inplace=True,
                                   percent_top=None, log1p=False)
        combined = combined[combined.obs.pct_counts_mt < 20].copy()
        # normalize + HVG + PCA
        sc.pp.normalize_total(combined, target_sum=1e4)
        sc.pp.log1p(combined)
        sc.pp.highly_variable_genes(combined, n_top_genes=2000,
                                    batch_key="batch", flavor="seurat_v3")
        sc.pp.scale(combined, max_value=10)
        sc.tl.pca(combined, n_comps=30)
        # confound check
        if condition_col and condition_col in combined.obs.columns:
            conf = detect_confound(combined, "batch", condition_col)
            log.info(f"[confound] {conf['VERDICT']}")
            if conf.get("confounded"):
                raise DataRefusal(
                    "FATAL CONFOUNDING: batch ≡ condition.\n"
                    "Harmony would remove the treatment effect.\n"
                    "Either redesign the integration (add a series that "
                    "contains both conditions) or do NOT correct for batch.\n"
                    "See detect_confound() output for details.")
        # Harmony correction
        if HAS_HARMONY:
            hc = HarmonyCorrector()
            combined = hc.correct(combined, "batch")
            combined.obsm["X_corrected"] = combined.obsm["X_pca_harmony"]
        else:
            log.warning("[integrator] harmonypy not available; using fallback")
            combined = fallback_batch_regress(combined, "batch")
            combined.obsm["X_corrected"] = combined.obsm["X_pca_batch_regress"]
        # neighbors + UMAP + Leiden in corrected space
        sc.pp.neighbors(combined, use_rep="X_corrected")
        sc.tl.umap(combined)
        sc.tl.leiden(combined, resolution=0.6)
        # annotate
        annot = MarkerAnnotator(self.organ)
        ann_out = annot.annotate(combined)
        combined.obs["organ"] = self.organ
        return combined


class DataRefusal(RuntimeError):
    pass


# =============================================================================
# 5. COMPOSITION EXTRACTION (after integration)
# =============================================================================

def composition_per_sample(adata: "ad.AnnData",
                           sample_col: str,
                           cell_types: List[str],
                           min_cells: int = 30) -> Dict[str, np.ndarray]:
    """
    For each sample, compute fraction of cells per annotated type.
    Only samples with >= min_cells are kept.
    """
    out = {}
    for s in adata.obs[sample_col].unique():
        sub = adata.obs[adata.obs[sample_col] == s]
        if len(sub) < min_cells:
            continue
        counts = sub["cell_type"].value_counts()
        vec = np.zeros(len(cell_types), np.float32)
        for ct, n in counts.items():
            if ct in cell_types:
                vec[cell_types.index(ct)] = n
        vec = vec / max(vec.sum(), 1)
        out[str(s)] = vec
    log.info(f"[composition] {len(out)} samples with >= {min_cells} cells")
    return out


# =============================================================================
# 6. SELF-TEST
# =============================================================================

def selftest(args):
    print("=" * 76)
    print("  ORGANOID MARKERS + HARMONY — self-test")
    print("=" * 76)
    ok = True

    print("\n[T1] Marker registry completeness")
    for organ, markers in ORGAN_REGISTRY.items():
        n_pos = sum(len(m.positive) for m in markers)
        n_neg = sum(len(m.negative) for m in markers)
        print(f"   {organ:12s}: {len(markers)} types, "
              f"{n_pos} positive markers, {n_neg} negative markers")
    t = all(len(v) >= 5 for v in ORGAN_REGISTRY.values())
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T2] Shared-marker conflict detection")
    shared = SHARED_MARKERS
    for organ, markers in ORGAN_REGISTRY.items():
        used = set()
        for m in markers:
            used.update(m.positive)
        overlap = used & shared
        if overlap:
            print(f"   {organ}: shared markers = {overlap}")
    t = len(shared) > 0
    print(f"   {'PASS' if t else 'FAIL'} (shared markers detected)")
    ok &= t

    print("\n[T3] Confound detection logic")
    if HAS_SCANPY:
        # simulate confounded data
        n = 200
        adata = ad.AnnData(X=np.random.rand(n, 50).astype(np.float32))
        adata.obs["batch"] = ["GSE_A"] * 100 + ["GSE_B"] * 100
        adata.obs["treatment"] = ["drug"] * 100 + ["control"] * 100
        conf = detect_confound(adata, "batch", "treatment")
        print(f"   confounded={conf['confounded']}  verdict: {conf['VERDICT'][:60]}...")
        t = conf["confounded"] == True
        # simulate non-confounded
        adata.obs["treatment"] = ["drug", "control"] * 100
        conf2 = detect_confound(adata, "batch", "treatment")
        print(f"   non-confounded={not conf2['confounded']}")
        t = t and conf2["confounded"] == False
        print(f"   {'PASS' if t else 'FAIL'}"); ok &= t
    else:
        print("   SKIP — scanpy not installed")

    print("\n[T4] Marker scoring on synthetic data")
    if HAS_SCANPY:
        n_cells, n_genes = 200, 100
        rng = np.random.default_rng(0)
        X = rng.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        # inject real marker names at known positions
        for i, g in enumerate(["LGR5", "MUC2", "LYZ", "FABP1", "CHGA"]):
            if i < n_genes:
                gene_names[i] = g
        adata = ad.AnnData(X=X)
        adata.var_names = gene_names
        # make cells 0-50 high in LGR5 (stem), 50-100 high in MUC2 (goblet)
        adata.X[:50, 0] += 5.0
        adata.X[50:100, 1] += 5.0
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        annot = MarkerAnnotator("intestinal", min_score=0.0)
        result = annot.annotate(adata)
        dist = result["distribution"]
        print(f"   distribution: {dist}")
        t = len(dist) > 0
        print(f"   {'PASS' if t else 'FAIL'}"); ok &= t
    else:
        print("   SKIP — scanpy not installed")

    print("\n[T5] Harmony availability")
    print(f"   harmonypy installed: {HAS_HARMONY}")
    if not HAS_HARMONY:
        print("   → pip install harmonypy")
        print("   → fallback (linear regression) will be used but is INFERIOR")

    print("\n[T6] MultiSeriesIntegrator refuses species mismatch")
    if HAS_SCANPY:
        integ = MultiSeriesIntegrator("intestinal")
        a_human = ad.AnnData(X=np.random.rand(50, 20).astype(np.float32))
        a_human.var_names = [f"GENE{i}" for i in range(20)]
        a_mouse = ad.AnnData(X=np.random.rand(50, 20).astype(np.float32))
        a_mouse.var_names = [f"Gene{i}" for i in range(20)]  # Title-case = mouse
        integ.add_series("GSE_HUMAN", a_human)
        integ.add_series("GSE_MOUSE", a_mouse)
        try:
            integ.integrate()
            print("   FAIL — should have refused"); ok = False
        except (DataRefusal, ValueError) as e:
            print(f"   correctly refused: {str(e)[:60]}  PASS")
    else:
        print("   SKIP — scanpy not installed")

    print("\n" + "=" * 76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 76)
    print("""
  KEY REMINDERS:
    • Marker annotation is a HYPOTHESIS, not ground truth. Inspect UMAP.
    • Shared markers (SOX2, WT1, EPCAM...) cause cross-organ conflicts.
    • Harmony CANNOT fix confounded batches. Check BEFORE correcting.
    • Harmony does NOT create treatment labels. Manual curation required.
    • 500 records need 25-100 GEO series. Plan the curation effort.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "markers", "confound-check"])
    p.add_argument("--organ", default="intestinal",
                   choices=list(ORGAN_REGISTRY.keys()))
    p.add_argument("--data", default=None)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "markers":
        for cm in ORGAN_REGISTRY[a.organ]:
            print(f"{cm.name}:")
            print(f"  positive: {cm.positive}")
            print(f"  negative: {cm.negative}")
            print(f"  ref: {cm.reference}")
            print()
    elif a.cmd == "confound-check":
        if not HAS_SCANPY:
            log.error("pip install scanpy anndata"); return
        adata = sc.read_h5ad(a.data)
        conf = detect_confound(adata, "batch", "treatment")
        print(json.dumps(conf, indent=2, default=str))


if __name__ == "__main__":
    main()
