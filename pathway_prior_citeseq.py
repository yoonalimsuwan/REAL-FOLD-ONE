#!/usr/bin/env python3
"""
=============================================================================
PERTURB-seq PATHWAY PRIOR → T2 DOSE MODEL  +  PERTURB-CITE-seq (RNA+ADT)
=============================================================================
License: MIT | 2026

LITERATURE
 [W1] Janda et al. "Surrogate Wnt agonists..." Nature 545, 234 (2017).
 [W2] Silva et al. "De novo design of IL-2/IL-15 mimics." Nature 565, 186 (2019).
 [P1] Dixit et al. "Perturb-seq." Cell 167, 1853 (2016).
 [P2] Frangieh et al. "Perturb-CITE-seq." Nat. Genet. 53, 332 (2021).
 [P3] Replogle et al. "Genome-scale Perturb-seq." Cell 185, 2559 (2022).
 [C1] Stoeckius et al. "Simultaneous epitope and transcriptome measurement
      in single cells." Nat. Methods 14, 865 (2017).             [CITE-seq]
 [C2] Gayoso et al. "Joint probabilistic modeling of single-cell multi-omic
      data with totalVI." Nat. Biotechnol. 39, 272 (2021).
 [S1] Clevers & Watt. "Defining adult stem cell function." Cell 155, 502 (2013).
 [S2] Sato et al. "Intestinal organoids." Nature 459, 262 (2009).

=============================================================================
HOW THE PRIOR WORKS
=============================================================================
1. Perturb-seq KO of pathway genes → differential expression signature
   e.g., CTNNB1 KO: log2FC of AXIN2 = -2.1, MYC = -1.8, LGR5 = -1.5

2. Pathway aggregation: mean across KO genes in the pathway
   Wnt pathway signature = mean(CTNNB1_KO, APC_KO, AXIN1_KO, GSK3B_KO)

3. Direction prior: KO decreases gene X → activation should increase X
   prior[Wnt, AXIN2] = +1  (Wnt activation → AXIN2 up)
   prior[Wnt, CDX2] = 0   (no direct effect)

4. T2 regularization:
   L_prior = Σ_factors Σ_genes max(0, -prior[f,g] · predicted_change[f,g])
   Penalizes predictions that contradict the KO-derived direction.

=============================================================================
HONEST SCOPE
=============================================================================
✅ Pathway → gene mappings are from established literature (KEGG/Reactome).
✅ Perturb-seq KO signatures are real data.
✅ The prior constrains DIRECTION, not magnitude.

❌ KO ≠ partial dose reduction. A full knockout may have different effects
   than reducing the dose by 50%. The prior is a DIRECTIONAL constraint only.
❌ Pathway crosstalk (Wnt↔Notch, BMP↔Wnt) is NOT captured.
❌ CITE-seq covers ~100–300 proteins, NOT the whole proteome.
❌ Cell-type specificity: a pathway may have different effects in different
   cell types. The prior is an AVERAGE across the perturbed population.

⚠️  The prior must be ABLATED: run T2 with and without it.
    If the prior does not improve held-out prediction, remove it.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("pathway_prior")

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import scvi
    from scvi.model import TOTALVI
    HAS_TOTALVI = True
except ImportError:
    HAS_TOTALVI = False


# =============================================================================
# 1. PATHWAY DEFINITIONS (curated from literature)
# =============================================================================

@dataclass
class PathwayDefinition:
    name: str
    ko_genes: List[str]           # genes whose KO perturbs this pathway
    factors: List[str]            # exogenous factors that activate/inhibit
    target_genes: List[str]       # canonical transcriptional targets
    marker_genes: Dict[str, List[str]]  # fate markers affected
    direction_hint: str           # "activation" or "inhibition"

# Wnt/β-catenin: CTNNB1, APC, AXIN1/2, GSK3B, TCF7L2, LEF1
WNT = PathwayDefinition(
    name="WNT_BCATENIN",
    ko_genes=["CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B", "TCF7L2", "LEF1"],
    factors=["WNT3A", "CHIR"],
    target_genes=["AXIN2", "MYC", "CCND1", "LGR5", "ASCL2", "SOX9", "CD44"],
    marker_genes={
        "STEM_LGR5": ["LGR5", "ASCL2", "OLFM4", "SOX9", "SMOC2"],
        "ENTEROCYTE": ["FABP1", "VIL1", "ALPI", "SI"],
        "GOBLET": ["MUC2", "TFF3", "SPINK4"],
        "PANETH": ["LYZ", "DEFA5", "DEFA6"],
    },
    direction_hint="activation",
)

# BMP: SMAD1/5/8, BMPR1A/B, BMPR2
BMP = PathwayDefinition(
    name="BMP_SMAD158",
    ko_genes=["SMAD1", "SMAD5", "SMAD8", "BMPR1A", "BMPR1B", "BMPR2"],
    factors=["BMP4", "LDN193189"],
    target_genes=["ID1", "ID2", "ID3", "MSX1", "MSX2", "RUNX2"],
    marker_genes={
        "ENTEROCYTE": ["FABP1", "VIL1", "ALPI"],
        "GOBLET": ["MUC2", "TFF3"],
        "MESENCHYMAL": ["VIM", "COL1A1"],
    },
    direction_hint="activation",
)

# TGFβ/Activin: SMAD2/3/4, TGFBR1/2, ACVR1B
TGFB = PathwayDefinition(
    name="TGFB_ACTIVIN_SMAD23",
    ko_genes=["SMAD2", "SMAD3", "SMAD4", "TGFBR1", "TGFBR2", "ACVR1B"],
    factors=["ACTIVIN", "TGFB1", "SB431542"],
    target_genes=["SERPINE1", "CTGF", "SNAI1", "SNAI2", "TWIST1", "LEFTY1"],
    marker_genes={
        "ENDODERM": ["SOX17", "FOXA2", "GATA6"],
        "MESENCHYMAL": ["VIM", "COL1A1", "FN1"],
    },
    direction_hint="activation",
)

# FGF: FGFR1-4, RAS, RAF, MEK, ERK
FGF = PathwayDefinition(
    name="FGF_MAPK",
    ko_genes=["FGFR1", "FGFR2", "FGFR3", "FGFR4", "KRAS", "BRAF", "MAP2K1"],
    factors=["FGF2", "FGF4", "FGF9", "FGF10"],
    target_genes=["ETV4", "ETV5", "DUSP6", "SPRY2", "FOS", "JUN"],
    marker_genes={
        "NEURAL_PROGENITOR": ["SOX2", "NES", "PAX6"],
        "MESENCHYMAL": ["VIM", "PDGFRA"],
    },
    direction_hint="activation",
)

# Notch: NOTCH1/2, RBPJ, HES1, HEY1
NOTCH = PathwayDefinition(
    name="NOTCH",
    ko_genes=["NOTCH1", "NOTCH2", "RBPJ", "HES1", "HEY1", "PSEN1"],
    factors=["DAPT"],
    target_genes=["HES1", "HEY1", "HEY2", "HES5", "NRARP"],
    marker_genes={
        "GOBLET": ["MUC2", "TFF3"],
        "ENTEROENDOCRINE": ["CHGA", "NEUROD1"],
        "STEM_LGR5": ["LGR5", "ASCL2"],
    },
    direction_hint="activation",
)

# Hedgehog: SMO, PTCH1, GLI1/2/3
HEDGEHOG = PathwayDefinition(
    name="HEDGEHOG",
    ko_genes=["SMO", "PTCH1", "GLI1", "GLI2", "GLI3"],
    factors=["SHH", "SAG"],
    target_genes=["GLI1", "PTCH1", "HHIP", "MYCN", "BCL2"],
    marker_genes={
        "MESENCHYMAL": ["VIM", "PDGFRA"],
        "NEURAL_PROGENITOR": ["SOX2", "PAX6"],
    },
    direction_hint="activation",
)

# Retinoic acid: RARA/B/G, RXR
RA = PathwayDefinition(
    name="RETINOIC_ACID",
    ko_genes=["RARA", "RARB", "RARG", "RXRA", "RXRB", "CYP26A1"],
    factors=["RA"],
    target_genes=["RARB", "CYP26A1", "HOXA1", "HOXB1", "CRABP2"],
    marker_genes={
        "ENDODERM": ["SOX17", "FOXA2"],
        "ECTODERM": ["PAX6", "SOX1"],
    },
    direction_hint="activation",
)

PATHWAYS = [WNT, BMP, TGFB, FGF, NOTCH, HEDGEHOG, RA]

# Map factor names (from T2 FACTORS list) to pathway indices
FACTOR_TO_PATHWAY = {
    "WNT": "WNT_BCATENIN", "CHIR": "WNT_BCATENIN",
    "BMP4": "BMP_SMAD158", "LDN193189": "BMP_SMAD158",
    "ACTIVIN": "TGFB_ACTIVIN_SMAD23", "TGFB1": "TGFB_ACTIVIN_SMAD23",
    "SB431542": "TGFB_ACTIVIN_SMAD23",
    "FGF2": "FGF_MAPK", "FGF4": "FGF_MAPK", "FGF9": "FGF_MAPK", "FGF10": "FGF_MAPK",
    "NOTCH_AG": "NOTCH", "NOTCH_INH": "NOTCH", "DAPT": "NOTCH",
    "SHH": "HEDGEHOG",
    "RA": "RETINOIC_ACID",
}

# Fate categories (must match T2 FATES)
FATES = ["PLURIPOTENT", "ENDODERM", "MESODERM", "ECTODERM",
         "NEURAL_PROGENITOR", "NEURON", "GLIA",
         "INTESTINAL_EPITHELIUM", "GOBLET", "PANETH", "ENTEROCYTE",
         "HEPATOCYTE", "CHOLANGIOCYTE",
         "NEPHRON_PROGENITOR", "PODOCYTE", "TUBULE",
         "CARDIOMYOCYTE", "ENDOTHELIAL", "MESENCHYMAL", "UNKNOWN"]
F_IDX = {f: i for i, f in enumerate(FATES)}

# Map marker gene sets to fate categories
MARKER_TO_FATE = {
    "STEM_LGR5": "INTESTINAL_EPITHELIUM",
    "ENTEROCYTE": "ENTEROCYTE",
    "GOBLET": "GOBLET",
    "PANETH": "PANETH",
    "ENTEROENDOCRINE": "INTESTINAL_EPITHELIUM",
    "ENDODERM": "ENDODERM",
    "ECTODERM": "ECTODERM",
    "NEURAL_PROGENITOR": "NEURAL_PROGENITOR",
    "MESENCHYMAL": "MESENCHYMAL",
    "TUFT": "INTESTINAL_EPITHELIUM",
}


# =============================================================================
# 2. PATHWAY SIGNATURE EXTRACTION FROM PERTURB-seq
# =============================================================================

class PathwaySignatureExtractor:
    """
    Extract pathway-level signatures from Perturb-seq KO results.

    Input: per-guide differential expression (from PerturbationAggregator)
    Output: pathway → {gene: direction, magnitude} signature

    Direction is INFERRED from KO:
      KO decreases gene X → pathway activation should increase X
      So prior_direction[X] = +1 for that pathway.
    """
    def __init__(self, min_genes_per_pathway: int = 2):
        self.min_genes = min_genes_per_pathway

    def extract(
        self,
        perturb_results: Dict[str, Dict],  # guide_id → {target_gene, mean_effect, top_up, top_down}
        gene_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate KO signatures into pathway signatures.

        Returns: {pathway_name: {gene: direction_score}}
        direction_score > 0 means pathway activation upregulates that gene.
        """
        # Build gene → KO effect map
        # perturb_results values have 'target_gene' and 'mean_effect'
        # For a proper signature we need per-gene DE, not just mean effect.
        # Here we use the top_up/top_down lists as a proxy.
        pathway_sigs = {}
        for pw in PATHWAYS:
            ko_effects = []
            for guide_id, rec in perturb_results.items():
                if rec.get("target_gene") in pw.ko_genes:
                    ko_effects.append(rec)
            if len(ko_effects) < self.min_genes:
                log.warning(f"[extract] {pw.name}: only {len(ko_effects)} KO genes "
                            f"(< {self.min_genes}); skipping")
                continue
            # Aggregate: for each target gene, count up/down across KOs
            gene_scores = {}
            for rec in ko_effects:
                for g in rec.get("top_up", []):
                    gene_scores[g] = gene_scores.get(g, 0) + 1  # KO up → pathway inhibition upregulates
                for g in rec.get("top_down", []):
                    gene_scores[g] = gene_scores.get(g, 0) - 1  # KO down → pathway activation upregulates
            # Normalize
            n_ko = len(ko_effects)
            pathway_sigs[pw.name] = {g: s / n_ko for g, s in gene_scores.items()}
            log.info(f"[extract] {pw.name}: {len(gene_scores)} genes from {n_ko} KO guides")
        return pathway_sigs

    def to_prior_matrix(
        self,
        pathway_sigs: Dict[str, Dict[str, float]],
        n_factors: int,
        n_fates: int,
        factor_to_idx: Dict[str, int],
    ) -> np.ndarray:
        """
        Convert pathway signatures to a (n_factors, n_fates) prior matrix.

        prior[f, fate] = expected direction of effect of factor f on fate.
        +1: factor increases this fate fraction.
        -1: factor decreases this fate fraction.
         0: no prior information.
        """
        prior = np.zeros((n_factors, n_fates), np.float32)
        for pw in PATHWAYS:
            sig = pathway_sigs.get(pw.name)
            if sig is None:
                continue
            # Map this pathway to its factors
            pw_factors = [f for f in FACTOR_TO_PATHWAY if FACTOR_TO_PATHWAY[f] == pw.name]
            for f_name in pw_factors:
                if f_name not in factor_to_idx:
                    continue
                f_idx = factor_to_idx[f_name]
                # Map target genes to fates
                for marker_set, fate in MARKER_TO_FATE.items():
                    if fate not in F_IDX:
                        continue
                    fate_idx = F_IDX[fate]
                    # Look up genes in this marker set
                    genes = pw.marker_genes.get(marker_set, [])
                    if not genes:
                        continue
                    # Direction: mean KO effect on these genes
                    dir_scores = [sig.get(g, 0) for g in genes]
                    mean_dir = np.mean(dir_scores) if dir_scores else 0.0
                    # KO direction → activation direction (flip sign)
                    # KO decreases gene → activation increases gene
                    prior[f_idx, fate_idx] = -np.sign(mean_dir) * abs(mean_dir)
        return prior

    def prior_to_loss_weights(
        self, prior: np.ndarray, confidence_threshold: float = 0.1
    ) -> np.ndarray:
        """
        Convert prior to loss weights.
        High-confidence priors get weight 1; low-confidence get weight 0.
        """
        weights = np.abs(prior) >= confidence_threshold
        return weights.astype(np.float32)


# =============================================================================
# 3. T2 PRIOR REGULARIZATION LOSS
# =============================================================================

class T2PriorLoss:
    """
    Regularization loss that penalizes T2 predictions contradicting
    the Perturb-seq-derived prior.

    L_prior = Σ_f Σ_g w[f,g] · max(0, -prior[f,g] · predicted_change[f,g])

    where predicted_change[f,g] is the model's predicted effect of factor f
    on fate fraction g (relative to baseline).

    ⚠️ This constrains DIRECTION only, not magnitude.
    ⚠️ Must be ABLATED: run with weight=0 and weight>0 to see if it helps.
    """
    def __init__(self, prior: np.ndarray, weights: np.ndarray,
                 lambda_prior: float = 0.1):
        self.prior = prior
        self.weights = weights
        self.lambda_prior = lambda_prior

    def __call__(self, predicted_composition, baseline_composition,
                 factor_idx):
        """
        Args:
            predicted_composition: (B, n_fates) softmax output
            baseline_composition: (B, n_fates) baseline (e.g., no factor)
            factor_idx: (B,) index of factor for each sample
        Returns:
            scalar loss
        """
        # predicted_change[b, g] = predicted[b, g] - baseline[b, g]
        change = predicted_composition - baseline_composition
        # prior[factor_idx[b], g]
        B = factor_idx.shape[0]
        prior_sel = self.prior[factor_idx]      # (B, n_fates)
        w_sel = self.weights[factor_idx]        # (B, n_fates)
        # Violation: prior says +1 but model says negative (or vice versa)
        violation = np.maximum(0, -prior_sel * change)
        loss = (w_sel * violation).sum() / max(w_sel.sum(), 1)
        return self.lambda_prior * float(loss)


# =============================================================================
# 4. PERTURB-CITE-seq LOADER (RNA + ADT)
# =============================================================================

class PerturbCITELoader:
    """
    Load Perturb-CITE-seq data: RNA + surface protein (ADT) from same cells.

    Formats supported:
      • 10x Multiome: filtered_feature_bc_matrix with Gene Expression + Antibody Capture
      • Separate files: RNA matrix + ADT matrix with matching barcodes
      • h5ad with RNA in adata.X and ADT in adata.obsm or adata.obsp

    ⚠️ CITE-seq covers ~100–300 proteins, NOT the whole proteome.
    ⚠️ ADT and RNA have different noise models. Do NOT mix without normalization.
    """
    def __init__(self):
        self.rna = None
        self.adt = None
        self.obs = None

    def load_10x_multiome(self, path: Path) -> Tuple["ad.AnnData", "ad.AnnData"]:
        """Load 10x multiome (RNA + ADT) from a directory."""
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy")
        adata = sc.read_10x_mtx(str(path), gex_suffix="", cache=False)
        # Split RNA and ADT
        rna_mask = adata.var["feature_types"] == "Gene Expression"
        adt_mask = adata.var["feature_types"] == "Antibody Capture"
        rna = adata[:, rna_mask].copy()
        adt = adata[:, adt_mask].copy()
        log.info(f"[CITE] RNA: {rna.shape}, ADT: {adt.shape}")
        self.rna, self.adt = rna, adt
        return rna, adt

    def load_separate(self, rna_path: Path, adt_path: Path,
                      barcode_path: Optional[Path] = None):
        """Load RNA and ADT from separate files."""
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy")
        rna = sc.read_10x_mtx(str(rna_path)) if rna_path.is_dir() else \
              sc.read_csv(str(rna_path))
        adt = sc.read_10x_mtx(str(adt_path)) if adt_path.is_dir() else \
              sc.read_csv(str(adt_path))
        self.rna, self.adt = rna, adt
        return rna, adt

    def load_h5ad(self, rna_h5ad: Path, adt_obsm_key: str = "adt"):
        """Load from h5ad with ADT in obsm."""
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy")
        rna = sc.read_h5ad(str(rna_h5ad))
        if adt_obsm_key in rna.obsm:
            adt_arr = rna.obsm[adt_obsm_key]
            adt = ad.AnnData(
                X=adt_arr,
                obs=rna.obs.copy(),
                var=rna.var[[]][:adt_arr.shape[1]].copy() if adt_arr.shape[1] <= rna.var.shape[0] else None,
            )
        else:
            log.warning(f"[CITE] '{adt_obsm_key}' not in obsm; ADT unavailable")
            adt = None
        self.rna, self.adt = rna, adt
        return rna, adt

    def normalize(self, method: str = "dsb") -> Dict:
        """
        Normalize RNA and ADT separately.

        RNA: log-normalize (standard)
        ADT: DSB (denoised and scaled by background) or CLR

        ⚠️ DSB requires isotype control antibodies or background estimation.
           CLR is simpler but less robust.
        """
        if self.rna is None:
            raise ValueError("load data first")
        # RNA
        sc.pp.normalize_total(self.rna, target_sum=1e4)
        sc.pp.log1p(self.rna)
        log.info(f"[CITE] RNA normalized: {self.rna.shape}")
        # ADT
        if self.adt is not None:
            if method == "clr":
                # Centered log-ratio
                X = self.adt.X
                if hasattr(X, "toarray"):
                    X = X.toarray()
                X = np.asarray(X, dtype=np.float64)
                geo_mean = np.exp(np.log(X + 1).mean(axis=1, keepdims=True))
                clr = np.log((X + 1) / (geo_mean + 1e-8))
                self.adt.X = clr.astype(np.float32)
                log.info(f"[CITE] ADT CLR-normalized: {self.adt.shape}")
            else:
                # DSB requires isotype controls; fall back to CLR
                log.warning("[CITE] DSB requires isotype controls; falling back to CLR")
                self.normalize(method="clr")
        return {"rna": self.rna, "adt": self.adt}


# =============================================================================
# 5. MULTI-OMIC INTEGRATION (totalVI or manual)
# =============================================================================

class MultiOmicsIntegrator:
    """
    Integrate RNA + ADT for joint analysis.

    Option A: totalVI (scvi-tools) — probabilistic, handles both modalities.
    Option B: Manual concat + PCA — simpler, no totalVI required.

    ⚠️ totalVI requires raw counts (not log-normalized).
    ⚠️ ADT panel must be provided as a separate matrix.
    """
    def __init__(self, use_totalvi: bool = True):
        self.use_totalvi = use_totalvi and HAS_TOTALVI
        self.model = None

    def integrate(self, rna: "ad.AnnData", adt: "ad.AnnData",
                  batch_key: str = "batch",
                  n_latent: int = 30) -> "ad.AnnData":
        if self.use_totalvi:
            return self._totalvi(rna, adt, batch_key, n_latent)
        else:
            return self._manual_concat(rna, adt, n_latent)

    def _totalvi(self, rna, adt, batch_key, n_latent):
        """totalVI joint model."""
        adata = rna.copy()
        adata.obsm["protein_counts"] = adt.X
        TOTALVI.setup_anndata(
            adata,
            protein_expression_obsm_key="protein_counts",
            batch_key=batch_key if batch_key in adata.obs.columns else None,
        )
        self.model = TOTALVI(adata, latent_distribution="normal")
        self.model.train(max_epochs=200, early_stopping=True)
        adata.obsm["X_totalVI"] = self.model.get_latent_representation()
        return adata

    def _manual_concat(self, rna, adt, n_latent):
        """Simple concat + PCA. No probabilistic model."""
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy")
        adata = rna.copy()
        # Normalize ADT if not already
        adt_X = adt.X
        if hasattr(adt_X, "toarray"):
            adt_X = adt_X.toarray()
        adt_X = np.asarray(adt_X, dtype=np.float32)
        # Scale ADT to same magnitude as RNA
        adt_scaled = adt_X / (np.std(adt_X, axis=0, keepdims=True) + 1e-8)
        # Concatenate
        from scipy.sparse import hstack, csr_matrix
        rna_X = rna.X
        if hasattr(rna_X, "toarray"):
            rna_X = rna_X.toarray()
        combined = np.hstack([np.asarray(rna_X), adt_scaled])
        adata.obsm["X_combined"] = combined
        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_latent)
        adata.obsm["X_pca_multi"] = pca.fit_transform(combined)
        log.info(f"[integrate] manual concat: {combined.shape} → PCA {n_latent}")
        return adata


# =============================================================================
# 6. PROTEIN-BASED CELL TYPE ANNOTATION (CITE-seq advantage)
# =============================================================================

# Surface protein markers for common cell types
# These are MORE RELIABLE than RNA for surface markers
PROTEIN_MARKERS = {
    "STEM_LGR5": ["LGR5", "EPCAM", "CD44"],
    "ENTEROCYTE": ["EPCAM", "CDX2", "VIL1"],
    "GOBLET": ["MUC2", "EPCAM"],
    "PANETH": ["LYZ", "EPCAM"],
    "T_CELL": ["CD3", "CD4", "CD8"],
    "B_CELL": ["CD19", "CD20", "CD79A"],
    "MONOCYTE": ["CD14", "CD16"],
    "NK": ["CD56", "CD16"],
    "ENDOTHELIAL": ["CD31", "VWF"],
    "FIBROBLAST": ["PDGFRA", "COL1A1"],
    "EPITHELIAL": ["EPCAM", "KRT8", "KRT18"],
}


def protein_score_celltype(adt_matrix: np.ndarray,
                           protein_names: List[str],
                           markers: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Score cells against protein marker sets.
    More reliable than RNA for surface markers.

    Returns: {cell_type: score_array}
    """
    p_idx = {p: i for i, p in enumerate(protein_names)}
    scores = {}
    for ct, genes in markers.items():
        present = [p_idx[g] for g in genes if g in p_idx]
        if not present:
            continue
        scores[ct] = adt_matrix[:, present].mean(axis=1)
    return scores


# =============================================================================
# 7. SELF-TEST
# =============================================================================

def selftest(args) -> bool:
    print("=" * 76)
    print("  PATHWAY PRIOR + PERTURB-CITE-seq — self-test")
    print("=" * 76)
    ok = True

    print("\n[T1] Pathway definitions completeness")
    for pw in PATHWAYS:
        n_ko = len(pw.ko_genes)
        n_f = len(pw.factors)
        n_t = len(pw.target_genes)
        print(f"   {pw.name:<28s} KO genes={n_ko:2d}  factors={n_f}  targets={n_t}")
        t = n_ko >= 3 and n_f >= 1 and n_t >= 2
        ok &= t
    print(f"   {'PASS' if ok else 'FAIL'}")

    print("\n[T2] Factor → pathway mapping")
    n_mapped = len(FACTOR_TO_PATHWAY)
    n_unmapped = sum(1 for f in FACTOR_TO_PATHWAY if f not in
                     ["WNT","CHIR","BMP4","LDN193189","ACTIVIN","TGFB1","SB431542",
                      "FGF2","FGF4","FGF9","FGF10","NOTCH_AG","NOTCH_INH","DAPT","SHH","RA"])
    print(f"   {n_mapped} factors mapped to pathways")
    t = n_mapped >= 15
    ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T3] Pathway signature extraction (synthetic)")
    # Create synthetic perturb-seq results
    synth_results = {
        "guide_CTNNB1": {"target_gene": "CTNNB1", "mean_effect": 0.5,
                          "top_up": ["MYC", "CCND1"], "top_down": ["AXIN2", "LGR5"]},
        "guide_APC": {"target_gene": "APC", "mean_effect": 0.4,
                       "top_up": ["MYC"], "top_down": ["AXIN2", "ASCL2"]},
        "guide_SMAD4": {"target_gene": "SMAD4", "mean_effect": 0.3,
                         "top_up": ["ID1"], "top_down": ["SERPINE1"]},
    }
    extractor = PathwaySignatureExtractor(min_genes_per_pathway=1)
    sigs = extractor.extract(synth_results)
    print(f"   Extracted {len(sigs)} pathway signatures")
    for pw, sig in sigs.items():
        print(f"     {pw}: {len(sig)} genes")
    t = len(sigs) >= 1
    ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T4] Prior matrix construction")
    FACTORS_LIST = ["WNT", "CHIR", "BMP4", "LDN193189", "ACTIVIN", "TGFB1",
                    "SB431542", "FGF2", "FGF4", "FGF9", "FGF10",
                    "NOTCH_AG", "NOTCH_INH", "DAPT", "SHH", "RA", "OTHER"]
    F2I = {f: i for i, f in enumerate(FACTORS_LIST)}
    prior = extractor.to_prior_matrix(sigs, len(FACTORS_LIST), len(FATES), F2I)
    print(f"   Prior matrix shape: {prior.shape}")
    n_nonzero = int((prior != 0).sum())
    print(f"   Non-zero entries: {n_nonzero}")
    t = prior.shape == (len(FACTORS_LIST), len(FATES))
    ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T5] Prior loss computation")
    weights = extractor.prior_to_loss_weights(prior, confidence_threshold=0.05)
    loss_fn = T2PriorLoss(prior, weights, lambda_prior=0.1)
    # Synthetic prediction
    B = 4
    pred = np.random.dirichlet(np.ones(len(FATES)), size=B).astype(np.float32)
    base = np.random.dirichlet(np.ones(len(FATES)), size=B).astype(np.float32)
    factor_idx = np.array([0, 1, 2, 3])
    loss_val = loss_fn(pred, base, factor_idx)
    print(f"   Prior loss = {loss_val:.6f} (non-negative: {loss_val >= 0})")
    t = loss_val >= 0 and not np.isnan(loss_val)
    ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T6] Protein marker sets")
    n_ct = len(PROTEIN_MARKERS)
    n_proteins = len(set(p for markers in PROTEIN_MARKERS.values() for p in markers))
    print(f"   {n_ct} cell types, {n_proteins} unique protein markers")
    t = n_ct >= 8 and n_proteins >= 20
    ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T7] CITE-seq protein scoring")
    n_cells, n_proteins = 100, 30
    protein_names = list(PROTEIN_MARKERS["STEM_LGR5"][:3]) + \
                    list(PROTEIN_MARKERS["ENTEROCYTE"][:3]) + \
                    [f"PROT{i}" for i in range(n_proteins - 6)]
    adt_mat = np.random.rand(n_cells, n_proteins).astype(np.float32)
    scores = protein_score_celltype(adt_mat, protein_names, PROTEIN_MARKERS)
    print(f"   Scored {len(scores)} cell types")
    t = len(scores) >= 2
    ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T8] totalVI availability")
    print(f"   totalVI installed: {HAS_TOTALVI}")
    if not HAS_TOTALVI:
        print("   → pip install scvi-tools  (for probabilistic multi-omics)")
        print("   → Manual concat + PCA fallback is available")

    print("\n" + "=" * 76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 76)
    print("""
  KEY POINTS:
    • The prior constrains DIRECTION, not magnitude.
    • KO ≠ dose reduction. The prior is a directional constraint only.
    • Pathway crosstalk is NOT captured.
    • CITE-seq covers ~100-300 proteins, not the whole proteome.
    • Protein markers are MORE RELIABLE than RNA for surface markers.
    • The prior MUST be ablated: run T2 with and without it.
    • If the prior does not improve held-out prediction, remove it.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "extract-prior", "load-cite"])
    p.add_argument("--perturb-json", dest="perturb_json", default=None)
    p.add_argument("--cite-path", dest="cite_path", default=None)
    p.add_argument("--out", default="data/prior")
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "extract-prior":
        if not a.perturb_json:
            log.error("need --perturb-json"); return
        results = json.load(open(a.perturb_json))
        ext = PathwaySignatureExtractor()
        sigs = ext.extract(results)
        FACTORS_LIST = list(FACTOR_TO_PATHWAY.keys())
        F2I = {f: i for i, f in enumerate(FACTORS_LIST)}
        prior = ext.to_prior_matrix(sigs, len(FACTORS_LIST), len(FATES), F2I)
        weights = ext.prior_to_loss_weights(prior)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        np.save(Path(a.out) / "prior_matrix.npy", prior)
        np.save(Path(a.out) / "prior_weights.npy", weights)
        json.dump(sigs, open(Path(a.out) / "pathway_signatures.json", "w"), indent=2)
        log.info(f"Saved prior → {a.out}")
    elif a.cmd == "load-cite":
        if not a.cite_path:
            log.error("need --cite-path"); return
        loader = PerturbCITELoader()
        p = Path(a.cite_path)
        if p.is_dir():
            rna, adt = loader.load_10x_multiome(p)
        else:
            rna, adt = loader.load_h5ad(p)
        loader.normalize(method="clr")
        integrator = MultiOmicsIntegrator(use_totalvi=HAS_TOTALVI)
        result = integrator.integrate(loader.rna, loader.adt)
        log.info(f"Integrated: {result.shape}")


if __name__ == "__main__":
    main()
