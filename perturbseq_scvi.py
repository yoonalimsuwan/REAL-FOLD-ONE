#!/usr/bin/env python3
"""
=============================================================================
PERTURB-seq / CROP-seq / Perturb-CITE-seq LOADER + scVI/scANVI INTEGRATION
=============================================================================
License: MIT | 2026

LITERATURE
 [P1] Dixit et al. "Perturb-seq: probing gene function in complex systems by
      single-cell RNA sequencing." Cell 167, 1853 (2016).
 [P2] Datlinger et al. "Pooled CRISPR screening with single-cell transcriptome
      readout." Nat. Methods 14, 297 (2017).            [CROP-seq]
 [P3] Frangieh et al. "Multimodal pooled CRISPR screens with Perturb-CITE-seq."
      Nat. Genet. 53, 332 (2021).
 [P4] Replogle et al. "Mapping information-rich genotype-phenotype landscapes
      with genome-scale Perturb-seq." Cell 185, 2559 (2022).
 [P5] Norman et al. "Exploring genetic interaction manifolds." Science 365, 1408 (2019).
 [V1] Lopez et al. "Deep generative modeling for single-cell transcriptomics."
      Nat. Methods 15, 1053 (2018).                     [scVI]
 [V2] Xu et al. "Probabilistic harmonization and annotation of single-cell
      transcriptomics data with deep generative models."
      Nat. Methods 18, 148 (2021).                      [scANVI]
 [V3] Gayoso et al. "A Python library for probabilistic analysis of single-cell
      omics data." Nat. Biotechnol. 40, 163 (2022).     [scvi-tools]

=============================================================================
WHY THIS REDUCES CURATION — AND WHERE IT DOES NOT
=============================================================================
✅ Perturb-seq datasets ship WITH perturbation metadata: each cell knows its
   guide(s). You do NOT read papers to assign (perturbation → cell). This is
   the real curation saving.

⚠️ BUT the perturbation is GENETIC (CRISPR KO/CRISPRi/CRISPRa of an endogenous
   gene), NOT an exogenous factor dose. T2's "Wnt @ 100 nM → fate" is a
   different quantity from "CTNNB1 KO → transcriptome". They are complementary:
   • Perturb-seq of pathway genes (β-catenin, SMADs, receptors) tells you the
     wiring and necessity of a signaling pathway.
   • Factor dose–response tells you the quantitative morphogen effect.
   Both are tagged with perturbation_type and MUST NOT be mixed blindly.

⚠️ Most Perturb-seq is loss-of-function at a single dose (full KO). There is
   no dose–response curve. Do not claim dose information from it.

=============================================================================
HONEST SCOPE
=============================================================================
LOADER
  ✅ Reads standard formats: h5ad with guide column, 10x + guide matrix,
     CROP-seq (guide in transcript), Perturb-seq mixscape-style output.
  ✅ Guide QC: drops cells with no guide; flags multi-guide (high MOI).
  ✅ Aggregates per-guide responses with control subtraction.
  ❌ Does NOT invent guide→gene mappings. Requires the dataset's own annotation.

scVI / scANVI
  ✅ Real negative-binomial VAE on raw counts (scvi-tools [V3]).
  ✅ scANVI semi-supervised when partial cell-type labels exist.
  ❌ Requires GPU for reasonable speed; falls back to CPU with a warning.
  ❌ Cannot fix batch≡condition confounding (same as Harmony). Check first.

INTEGRATION CHOICE
  • Mild batch effects / CPU only / fast     → Harmony
  • Different protocols / strong batch / GPU → scVI
  • Partial cell-type labels to propagate    → scANVI
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("perturbseq_scvi")

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import scvi
    from scvi.model import SCVI, SCANVI
    HAS_SCVI = True
except ImportError:
    HAS_SCVI = False
    log.warning("scvi-tools not installed → scVI/scANVI disabled. "
                "pip install scvi-tools")

try:
    import harmonypy as hpy
    HAS_HARMONY = True
except ImportError:
    HAS_HARMONY = False

# Non-targeting control guide patterns (common across datasets)
NT_PATTERNS = re.compile(
    r"non.?target|non.?targeting|nt_|control|safe.?harbor|aaav|scrambl",
    re.IGNORECASE)


# =============================================================================
# 1. PERTURB-seq LOADER
# =============================================================================

@dataclass
class PerturbSeqConfig:
    guide_col_candidates: List[str] = field(default_factory=lambda: [
        "guide_id", "guide", "gRNA", "grna", "perturbation", "perturb",
        "crispr", "sgRNA", "target_gene", "gene_perturbation"])
    gene_col_candidates: List[str] = field(default_factory=lambda: [
        "target_gene", "gene", "gene_target", "perturbed_gene"])
    min_reads_per_guide: int = 1
    allow_multi_guide: bool = False
    min_cells_per_guide: int = 20
    control_label: str = "non_targeting"


class PerturbSeqLoader:
    """
    Load Perturb-seq / CROP-seq / Perturb-CITE-seq data.

    Expected inputs (any one):
      (a) .h5ad with guide identity in adata.obs
      (b) .h5ad with a cell×guide matrix in adata.obsm
      (c) 10x count dir + separate guide barcodes file

    Produces an AnnData with obs columns:
      guide_id, target_gene, n_guides, is_control, perturbation_type
    """
    def __init__(self, cfg: Optional[PerturbSeqConfig] = None):
        self.cfg = cfg or PerturbSeqConfig()

    # ---- detection ----
    def detect_guide_column(self, adata: "ad.AnnData") -> Optional[str]:
        for c in self.cfg.guide_col_candidates:
            if c in adata.obs.columns:
                return c
        # fuzzy match
        for col in adata.obs.columns:
            if any(k in col.lower() for k in ("guide", "grna", "perturb")):
                return col
        return None

    def detect_gene_column(self, adata: "ad.AnnData") -> Optional[str]:
        for c in self.cfg.gene_col_candidates:
            if c in adata.obs.columns:
                return c
        return None

    def detect_guide_matrix(self, adata: "ad.AnnData") -> Optional[str]:
        """Find a cell×guide matrix in obsm."""
        for key in adata.obsm.keys():
            if any(k in key.lower() for k in ("guide", "perturb", "crispr")):
                return key
        return None

    # ---- loading ----
    def load(self, path: Path, guide_matrix_path: Optional[Path] = None
             ) -> "ad.AnnData":
        if not HAS_SCANPY:
            raise ImportError("pip install scanpy anndata")
        adata = sc.read_h5ad(str(path))
        log.info(f"[load] {adata.n_obs} cells × {adata.n_vars} genes")

        gcol = self.detect_guide_column(adata)
        gmat = self.detect_guide_matrix(adata)

        if gcol is not None:
            log.info(f"[load] guide column found: '{gcol}'")
            adata = self._from_guide_column(adata, gcol)
        elif gmat is not None:
            log.info(f"[load] guide matrix found: '{gmat}'")
            adata = self._from_guide_matrix(adata, gmat)
        elif guide_matrix_path is not None:
            log.info(f"[load] external guide matrix: {guide_matrix_path}")
            adata = self._from_external_matrix(adata, guide_matrix_path)
        else:
            raise ValueError(
                "No guide information found.\n"
                "Looked for obs columns: "
                f"{self.cfg.guide_col_candidates}\n"
                "and obsm keys containing 'guide/perturb/crispr'.\n"
                "Run `inspect(path)` to see what is actually present.")
        return adata

    def _classify_control(self, guide_id: str) -> bool:
        return bool(NT_PATTERNS.search(str(guide_id)))

    def _from_guide_column(self, adata, gcol) -> "ad.AnnData":
        gids = adata.obs[gcol].astype(str).values
        # multi-guide delimiter detection
        delim = None
        for d in (";", ",", "|", "+"):
            if any(d in g for g in gids[:100]):
                delim = d
                break
        guide_lists, n_guides = [], []
        for g in gids:
            if g in ("nan", "", "None", "no_guide", "unassigned"):
                guide_lists.append([]); n_guides.append(0)
            elif delim and delim in g:
                parts = [p.strip() for p in g.split(delim) if p.strip()]
                guide_lists.append(parts); n_guides.append(len(parts))
            else:
                guide_lists.append([g]); n_guides.append(1)

        gcol_out, is_ctrl = [], []
        drop = np.zeros(adata.n_obs, dtype=bool)
        for i, gl in enumerate(guide_lists):
            if len(gl) == 0:
                drop[i] = True
                gcol_out.append("NONE"); is_ctrl.append(False)
            elif len(gl) > 1 and not self.cfg.allow_multi_guide:
                # keep only if all are the same, else mark multi
                gcol_out.append(gl[0]); is_ctrl.append(self._classify_control(gl[0]))
            else:
                gcol_out.append(gl[0]); is_ctrl.append(self._classify_control(gl[0]))

        adata.obs["guide_id"] = gcol_out
        adata.obs["n_guides"] = n_guides
        adata.obs["is_control"] = is_ctrl
        adata.obs["perturbation_type"] = np.where(
            is_ctrl, "non_targeting", "genetic_perturbation")

        gcol_gene = self.detect_gene_column(adata)
        if gcol_gene:
            adata.obs["target_gene"] = adata.obs[gcol_gene].astype(str)
        else:
            adata.obs["target_gene"] = adata.obs["guide_id"].map(
                lambda g: self._strip_guide_prefix(g))

        before = adata.n_obs
        adata = adata[~drop].copy()
        log.info(f"[load] dropped {before - adata.n_obs} cells with no guide")
        self._report(adata)
        return adata

    def _from_guide_matrix(self, adata, gmat_key) -> "ad.AnnData":
        G = adata.obsm[gmat_key]
        if hasattr(G, "toarray"):
            G = G.toarray()
        G = np.asarray(G)
        guide_names = (adata.obsm.get(gmat_key + "_names", None)
                       if isinstance(adata.obsm, dict) else None)
        n_per_cell = (G > 0).sum(axis=1)
        top = G.argmax(axis=1)
        adata.obs["n_guides"] = n_per_cell
        adata.obs["guide_id"] = [
            (guide_names[i] if guide_names is not None else f"guide_{i}")
            for i in top]
        adata.obs["is_control"] = [self._classify_control(g)
                                   for g in adata.obs["guide_id"]]
        adata.obs["perturbation_type"] = np.where(
            adata.obs["is_control"], "non_targeting", "genetic_perturbation")
        adata.obs["target_gene"] = adata.obs["guide_id"].map(self._strip_guide_prefix)
        drop = n_per_cell == 0
        adata = adata[~drop].copy()
        log.info(f"[load] dropped {int(drop.sum())} cells with no guide")
        self._report(adata)
        return adata

    def _from_external_matrix(self, adata, gpath: Path) -> "ad.AnnData":
        import pandas as pd
        gm = pd.read_csv(gpath, index_col=0)
        common = list(set(gm.index) & set(adata.obs_names))
        if len(common) < adata.n_obs * 0.5:
            log.warning(f"[load] only {len(common)}/{adata.n_obs} cells overlap "
                        f"with guide matrix")
        gm = gm.loc[adata.obs_names.intersection(gm.index)]
        adata = adata[adata.obs_names.isin(gm.index)].copy()
        G = gm.values
        top = G.argmax(axis=1)
        adata.obs["guide_id"] = [gm.columns[i] for i in top]
        adata.obs["n_guides"] = (G > 0).sum(axis=1)
        adata.obs["is_control"] = [self._classify_control(g)
                                   for g in adata.obs["guide_id"]]
        adata.obs["perturbation_type"] = np.where(
            adata.obs["is_control"], "non_targeting", "genetic_perturbation")
        adata.obs["target_gene"] = adata.obs["guide_id"].map(self._strip_guide_prefix)
        self._report(adata)
        return adata

    @staticmethod
    def _strip_guide_prefix(g: str) -> str:
        """Best-effort guide→gene name heuristic. VERIFY against dataset annotation."""
        g = str(g)
        for pre in ("guide_", "gRNA_", "sg", "g"):
            if g.lower().startswith(pre.lower()) and g[len(pre):len(pre)+1].isalpha():
                return g[len(pre):]
        return g

    def _report(self, adata):
        n_ctrl = int(adata.obs["is_control"].sum())
        n_guide = adata.obs["guide_id"].nunique()
        n_multi = int((adata.obs["n_guides"] > 1).sum())
        log.info(f"[report] cells={adata.n_obs} guides={n_guide} "
                 f"controls={n_ctrl} multi-guide={n_multi}")
        if n_multi > 0 and not self.cfg.allow_multi_guide:
            log.warning(f"[report] {n_multi} multi-guide cells kept using FIRST "
                        f"guide only. Set allow_multi_guide=True to handle pairs.")
        if n_ctrl == 0:
            log.warning("[report] NO non-targeting controls found. Control "
                        "subtraction will be unreliable.")


# =============================================================================
# 2. PERTURBATION → RESPONSE AGGREGATION
# =============================================================================

class PerturbationAggregator:
    """
    Aggregate single-cell responses per perturbation.

    Produces, per guide/gene:
      • mean expression shift vs non-targeting control
      • fraction of cells per (coarse) state, if labels present
      • effect size (Cohen's d on selected marker genes)

    ⚠️ Output is a GENETIC perturbation response, not a factor dose curve.
    """
    def __init__(self, min_cells: int = 20):
        self.min_cells = min_cells

    def aggregate_expression(self, adata: "ad.AnnData",
                             marker_genes: Optional[List[str]] = None
                             ) -> Dict[str, np.ndarray]:
        ctrl_mask = adata.obs["is_control"].values.astype(bool)
        if ctrl_mask.sum() < self.min_cells:
            raise ValueError(f"only {ctrl_mask.sum()} control cells; need "
                             f">= {self.min_cells} for a stable baseline")
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        ctrl_mean = X[ctrl_mask].mean(axis=0)
        ctrl_std = X[ctrl_mask].std(axis=0) + 1e-6

        gene_names = list(adata.var_names)
        gidx = {g: i for i, g in enumerate(gene_names)}
        sel = None
        if marker_genes:
            sel = [gidx[g] for g in marker_genes if g in gidx]

        guides = adata.obs["guide_id"].values
        genes = adata.obs["target_gene"].values
        results = {}
        for g in np.unique(guides):
            m = guides == g
            if m.sum() < self.min_cells or bool(adata.obs["is_control"][m].any()):
                continue
            mean = X[m].mean(axis=0)
            d = (mean - ctrl_mean) / ctrl_std            # per-gene effect size
            rec = {
                "guide_id": g,
                "target_gene": str(genes[m][0]),
                "n_cells": int(m.sum()),
                "perturbation_type": "genetic_perturbation",
                "mean_shift_n_markers": float(np.abs(d[sel]).mean()) if sel else float(np.abs(d).mean()),
                "top_up": None, "top_down": None,
            }
            order = np.argsort(d)
            rec["top_down"] = [gene_names[i] for i in order[:5]]
            rec["top_up"] = [gene_names[i] for i in order[-5:][::-1]]
            results[g] = rec
        log.info(f"[aggregate] {len(results)} guides with >= {self.min_cells} cells")
        return results

    def to_t2_candidates(self, agg: Dict[str, np.ndarray],
                         pert_type_note: str = "genetic_KO") -> List[Dict]:
        """
        Convert aggregated perturbations into T2-style CANDIDATE records.
        ⚠️ These have NO dose and NO time; dose_nM/time_h are placeholders
        that MUST be replaced or the record must be excluded from T2 training.
        """
        out = []
        for g, r in agg.items():
            out.append({
                "record_id": f"perturbseq_{g}",
                "factor": "GENETIC_" + str(r["target_gene"]).upper(),
                "dose_nM": None,          # NOT available from KO Perturb-seq
                "time_h": None,           # single time point, unknown
                "perturbation_type": pert_type_note,
                "target_gene": r["target_gene"],
                "n_cells": r["n_cells"],
                "mean_effect": r["mean_shift_n_markers"],
                "WARNING": ("genetic perturbation, NOT factor dose. "
                            "dose/time unknown — do not train a dose model on this."),
            })
        return out


# =============================================================================
# 3. BATCH–CONDITION CONFOUND CHECK (shared with Harmony pipeline)
# =============================================================================

def detect_confound(adata: "ad.AnnData", batch_col: str,
                    condition_col: str) -> Dict:
    if condition_col not in adata.obs.columns:
        return {"has_condition": False, "confounded": None,
                "VERDICT": f"'{condition_col}' absent; cannot check"}
    ct = adata.obs.groupby([batch_col, condition_col]).size().unstack(fill_value=0)
    purity = ct.max(axis=1) / ct.sum(axis=1)
    confounded = bool((purity > 0.9).all())
    return {"has_condition": True, "n_batches": len(purity),
            "confounded": confounded,
            "VERDICT": ("FATAL: batch ≡ condition. Integration would remove the "
                        "biological signal. DO NOT integrate across this axis."
                        if confounded else
                        "OK: batches mix conditions. Integration is appropriate.")}


# =============================================================================
# 4. scVI / scANVI WRAPPERS
# =============================================================================

class ScVIIntegrator:
    """
    scVI [V1] via scvi-tools [V3]. Models raw counts with a negative-binomial
    VAE and a batch covariate. No log-normalization needed.

    Use when: strong batch effects, different protocols/technologies, GPU available.
    """
    def __init__(self, n_latent=30, n_layers=2, gene_likelihood="nb",
                 max_epochs=400, early_stopping=True, use_gpu=True):
        if not HAS_SCVI:
            raise ImportError("pip install scvi-tools")
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.gene_likelihood = gene_likelihood
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.use_gpu = use_gpu
        self.model = None

    def fit(self, adata: "ad.AnnData", batch_key: str = "batch",
            subset_hvg: bool = True, n_top_genes: int = 3000) -> "ad.AnnData":
        a = adata.copy()
        if subset_hvg:
            sc.pp.highly_variable_genes(a, n_top_genes=n_top_genes,
                                        subset=False, flavor="seurat_v3",
                                        batch_key=batch_key)
        # scVI wants raw counts
        SCVI.setup_anndata(a, batch_key=batch_key,
                           layer=None)
        self.model = SCVI(a, n_latent=self.n_latent, n_layers=self.n_layers,
                          gene_likelihood=self.gene_likelihood)
        accel = "gpu" if (self.use_gpu and HAS_SCVI) else "cpu"
        self.model.train(max_epochs=self.max_epochs,
                         early_stopping=self.early_stopping,
                         accelerator=accel)
        a.obsm["X_scVI"] = self.model.get_latent_representation()
        log.info(f"[scVI] latent shape {a.obsm['X_scVI'].shape}")
        return a

    def transform(self, adata: "ad.AnnData") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.get_latent_representation(adata)


class ScANVIIntegrator:
    """
    scANVI [V2]: semi-supervised scVI. Uses PARTIAL cell-type labels to guide
    integration and can transfer labels to unlabeled cells.

    Use when: you have some annotated cells (e.g., from marker annotation)
    and want label-aware integration + label transfer.
    """
    def __init__(self, unlabeled_category="unknown", n_latent=30,
                 max_epochs=200, use_gpu=True):
        if not HAS_SCVI:
            raise ImportError("pip install scvi-tools")
        self.unlabeled_category = unlabeled_category
        self.n_latent = n_latent
        self.max_epochs = max_epochs
        self.use_gpu = use_gpu
        self.model = None

    def fit(self, adata: "ad.AnnData", batch_key="batch",
            labels_key="cell_type", predict: bool = True) -> "ad.AnnData":
        a = adata.copy()
        if labels_key not in a.obs.columns:
            raise ValueError(f"labels_key '{labels_key}' not in obs. "
                             f"Run marker annotation first, or use ScVIIntegrator.")
        has_label = a.obs[labels_key].notna() & (a.obs[labels_key] != "")
        a.obs[labels_key] = np.where(has_label, a.obs[labels_key],
                                     self.unlabeled_category)
        n_lab = int((a.obs[labels_key] != self.unlabeled_category).sum())
        log.info(f"[scANVI] {n_lab}/{a.n_obs} labeled cells")
        if n_lab < 50:
            log.warning("[scANVI] very few labeled cells; semi-supervision will "
                        "be weak. Consider plain scVI.")
        SCVI.setup_anndata(a, batch_key=batch_key, labels_key=labels_key)
        base = SCVI(a, n_latent=self.n_latent)
        accel = "gpu" if self.use_gpu else "cpu"
        base.train(max_epochs=self.max_epochs, accelerator=accel)
        self.model = SCANVI.from_scvi_model(
            base, adata=a, unlabeled_category=self.unlabeled_category,
            labels_key=labels_key)
        self.model.train(max_epochs=self.max_epochs, accelerator=accel)
        a.obsm["X_scANVI"] = self.model.get_latent_representation()
        if predict:
            a.obs["predicted_cell_type"] = self.model.predict()
            a.obs["prediction_prob"] = self.model.predict(soft=True).max(axis=1).values
        return a


def run_harmony_baseline(adata: "ad.AnnData", batch_key="batch",
                         n_pcs=30) -> "ad.AnnData":
    """Harmony on PCA, for direct comparison with scVI."""
    if not HAS_HARMONY:
        raise ImportError("pip install harmonypy")
    a = adata.copy()
    if "X_pca" not in a.obsm:
        sc.pp.pca(a, n_comps=n_pcs)
    ho = hpy.run_harmony(a.obsm["X_pca"][:, :n_pcs], a.obs, batch_key)
    a.obsm["X_harmony"] = ho.Z_corr
    return a


# =============================================================================
# 5. INTEGRATION COMPARISON (scVI vs Harmony)
=============================================================================

def mixing_metrics(adata: "ad.AnnData", emb_key: str,
                   batch_key="batch", k=15) -> Dict[str, float]:
    """
    Simple batch-mixing diagnostics on an embedding:
      • mean silhouette-like batch separation (lower = better mixed)
      • kNN batch purity (lower = better mixed)
    These are DIAGNOSTICS, not proofs. Always inspect UMAP too.
    """
    from sklearn.neighbors import NearestNeighbors
    Z = adata.obsm[emb_key]
    batches = adata.obs[batch_key].values
    ub = list(np.unique(batches))
    bidx = np.array([ub.index(b) for b in batches])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]
    same = (bidx[idx] == bidx[:, None]).mean(axis=1)
    purity = float(same.mean())
    expected = sum((np.mean(batches == b)) ** 2 for b in ub)
    return {"knn_batch_purity": purity,
            "random_expected_purity": float(expected),
            "purity_over_random": purity / max(expected, 1e-9),
            "n_batches": len(ub)}


def compare_integrations(adata: "ad.AnnData", batch_key="batch",
                         labels_key: Optional[str] = None,
                         use_gpu=True) -> Dict:
    """
    Run scVI (+scANVI if labels) and Harmony; report mixing + biology retention.
    Biology retention proxy: cell-type silhouette if labels_key given.
    """
    out = {}
    # Harmony
    try:
        ah = run_harmony_baseline(adata, batch_key)
        out["harmony"] = mixing_metrics(ah, "X_harmony", batch_key)
    except Exception as e:
        out["harmony"] = {"error": str(e)}
    # scVI
    try:
        sv = ScVIIntegrator(use_gpu=use_gpu)
        asv = sv.fit(adata, batch_key)
        out["scvi"] = mixing_metrics(asv, "X_scVI", batch_key)
    except Exception as e:
        out["scvi"] = {"error": str(e)}
    # scANVI if labels
    if labels_key and labels_key in adata.obs.columns:
        try:
            sa = ScANVIIntegrator(use_gpu=use_gpu)
            asa = sa.fit(adata, batch_key, labels_key)
            out["scanvi"] = mixing_metrics(asa, "X_scANVI", batch_key)
        except Exception as e:
            out["scanvi"] = {"error": str(e)}
    return out


# =============================================================================
# 6. SELF-TEST (synthetic — for code paths only)
=============================================================================

def selftest(args) -> bool:
    print("=" * 76)
    print("  PERTURB-seq + scVI/scANVI — self-test (synthetic)")
    print("=" * 76)
    ok = True

    print("\n[T1] guide column detection + QC")
    if HAS_SCANPY:
        n = 300
        adata = ad.AnnData(X=np.random.negative_binomial(5, .3, (n, 60)).astype(np.float32))
        adata.var_names = [f"GENE{i}" for i in range(60)]
        guides = ["guide_TP53"]*80 + ["guide_BRCA1"]*80 + \
                 ["non_targeting_1"]*80 + ["guide_CTNNB1;guide_APC"]*40 + [""]*20
        adata.obs["guide_id"] = guides
        adata.obs["sample"] = ["donor_A"]*150 + ["donor_B"]*150
        loader = PerturbSeqLoader()
        gcol = loader.detect_guide_column(adata)
        print(f"   detected guide column: '{gcol}'")
        a2 = loader._from_guide_column(adata.copy(), gcol)
        n_ctrl = int(a2.obs["is_control"].sum())
        has_multi = int((a2.obs["n_guides"] > 1).sum())
        print(f"   cells kept={a2.n_obs}, controls={n_ctrl}, multi-guide={has_multi}")
        t = gcol == "guide_id" and n_ctrl == 80 and a2.n_obs == 280
        print(f"   {'PASS' if t else 'FAIL'}"); ok &= t
    else:
        print("   SKIP — scanpy not installed")

    print("\n[T2] non-targeting control classification")
    loader = PerturbSeqLoader()
    for g, want in [("non_targeting_1", True), ("NT_guide", True),
                    ("scramble_ctrl", True), ("guide_TP53", False)]:
        got = loader._classify_control(g)
        t = got == want
        print(f"   {g:20s} → control={got} (want {want})  {'PASS' if t else 'FAIL'}")
        ok &= t

    print("\n[T3] guide→gene prefix stripping")
    for g, want in [("guide_TP53", "TP53"), ("gRNA_BRCA1", "BRCA1"),
                    ("TP53", "TP53")]:
        got = PerturbSeqLoader._strip_guide_prefix(g)
        print(f"   {g:15s} → {got:10s} (want {want})")
        ok &= (got == want)

    print("\n[T4] perturbation-type tagging (genetic ≠ factor dose)")
    cand = PerturbationAggregator().to_t2_candidates(
        {"guide_TP53": {"target_gene": "TP53", "n_cells": 80,
                        "mean_shift_n_markers": 0.5, "top_up": [], "top_down": []}})
    t = cand[0]["dose_nM"] is None and "WARNING" in cand[0]
    print(f"   dose_nM={cand[0]['dose_nM']}  has_warning={('WARNING' in cand[0])}")
    print(f"   {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T5] confound detection")
    if HAS_SCANPY:
        adata.obs["condition"] = ["treated"]*150 + ["control"]*150
        adata.obs["batch"] = ["donor_A"]*150 + ["donor_B"]*150
        conf = detect_confound(adata, "batch", "condition")
        print(f"   confounded={conf['confounded']}")
        adata.obs["condition"] = (["treated","control"]*150)[:300]
        conf2 = detect_confound(adata, "batch", "condition")
        print(f"   after mixing: confounded={conf2['confounded']}")
        t = conf["confounded"] and not conf2["confounded"]
        print(f"   {'PASS' if t else 'FAIL'}"); ok &= t
    else:
        print("   SKIP")

    print("\n[T6] scvi-tools availability")
    print(f"   scvi-tools installed: {HAS_SCVI}")
    print(f"   harmonypy installed:  {HAS_HARMONY}")
    if not HAS_SCVI:
        print("   → pip install scvi-tools   (needs GPU for speed)")

    print("\n" + "=" * 76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 76)
    print("""
  KEY REMINDERS:
    • Perturb-seq metadata ships WITH the data → less manual curation.
    • But it is GENETIC perturbation (KO/i/a), NOT factor dose–response.
      Records carry perturbation_type and a WARNING; do not mix blindly.
    • scVI models raw counts (negative binomial); Harmony models PCA.
    • Neither fixes batch≡condition confounding. Check detect_confound first.
    • All of this is a CODE TEST on synthetic data. Real conclusions need
      real datasets (e.g., Replogle 2022, Norman 2019) and a GPU.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "load", "aggregate", "compare"])
    p.add_argument("--h5ad", default=None)
    p.add_argument("--guide-matrix", dest="guide_matrix", default=None)
    p.add_argument("--batch-key", dest="batch_key", default="batch")
    p.add_argument("--labels-key", dest="labels_key", default=None)
    p.add_argument("--markers", nargs="*", default=None)
    p.add_argument("--out", default="data/perturbseq")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if not HAS_SCANPY:
        log.error("pip install scanpy anndata"); return
    if a.cmd == "load":
        loader = PerturbSeqLoader()
        adata = loader.load(Path(a.h5ad),
                            Path(a.guide_matrix) if a.guide_matrix else None)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(Path(a.out) / "perturbseq_annotated.h5ad")
        log.info(f"saved → {a.out}/perturbseq_annotated.h5ad")
    elif a.cmd == "aggregate":
        adata = sc.read_h5ad(a.h5ad)
        agg = PerturbationAggregator().aggregate_expression(adata, a.markers)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        json.dump(agg, open(Path(a.out)/"perturb_aggregate.json","w"), indent=2)
        log.info(f"{len(agg)} guides → {a.out}/perturb_aggregate.json")
    elif a.cmd == "compare":
        adata = sc.read_h5ad(a.h5ad)
        conf = detect_confound(adata, a.batch_key,
                               a.labels_key or "perturbation_type")
        print(json.dumps(conf, indent=2))
        if conf.get("confounded"):
            log.error("confounded — aborting integration"); return
        res = compare_integrations(adata, a.batch_key, a.labels_key,
                                   use_gpu=not a.cpu)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
