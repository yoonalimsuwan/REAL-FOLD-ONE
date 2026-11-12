#!/usr/bin/env python3
"""
=============================================================================
scRNA-seq → CELL-TYPE COMPOSITION  (standard bioinformatics, not novel math)
=============================================================================
License: MIT | 2026

LITERATURE (all standard tools, not this project's contributions)
 [S1] Luecken & Theis. "Current best practices in single-cell RNA-seq
      analysis: a tutorial." Mol. Syst. Biol. 15, e8746 (2019).
 [S2] Wolf, Angerer, Theis. "SCANPY: large-scale single-cell gene expression
      data analysis." Genome Biol. 19, 15 (2018).
 [S3] Stuart & Satija. "Integrative single-cell analysis."
      Nat. Rev. Genet. 20, 257 (2019).
 [S4] Korsunsky et al. "Fast, sensitive and accurate integration of
      single-cell data with Harmony." Nat. Methods 16, 1289 (2019).

WHAT THIS DOES
  1. Load a count matrix + cell/sample metadata
  2. QC, normalize, HVG, PCA, neighbors, Leiden clustering
  3. Cell-type annotation (marker scoring — you supply markers)
  4. Aggregate cells → composition per sample
  5. Parse GEO sample metadata to propose (factor, dose, time)
  6. Write a partial T2 JSONL for you to VERIFY and complete

HONEST SCOPE
  ✅ Standard, citable scRNA-seq workflow.
  ❌ This is NOT where the project's mathematics is tested. The math is
     tested downstream, in the MDOE predictor, on the labels this produces.
  ⚠️ Cell-type annotation is the error-prone step. Marker lists must be
     appropriate for the organoid system. Review the UMAP/clusters yourself.
  ⚠️ GEO metadata parsing is heuristic. Every proposed (factor, dose, time)
     MUST be checked against the paper before verification.
  ❌ No batch correction across series is done here. If you merge multiple
     GSE, use Harmony/BBKNN and treat batch as a covariate.
=============================================================================
"""
from __future__ import annotations
import argparse, gzip, json, logging, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("scrna")

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False
    log.warning("scanpy/anndata not installed → clustering disabled. "
                "pip install scanpy anndata")

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

FATES = ["PLURIPOTENT", "ENDODERM", "MESODERM", "ECTODERM",
         "NEURAL_PROGENITOR", "NEURON", "GLIA",
         "INTESTINAL_EPITHELIUM", "GOBLET", "PANETH", "ENTEROCYTE",
         "HEPATOCYTE", "CHOLANGIOCYTE",
         "NEPHRON_PROGENITOR", "PODOCYTE", "TUBULE",
         "CARDIOMYOCYTE", "ENDOTHELIAL", "MESENCHYMAL", "UNKNOWN"]

# Organoid-system marker examples — REPLACE/EXTEND for your system.
# These are representative, not exhaustive, and must be checked per tissue.
DEFAULT_MARKERS = {
    "PLURIPOTENT": ["NANOG", "POU5F1", "SOX2", "KLF4"],
    "ENDODERM": ["SOX17", "FOXA2", "GATA6", "CXCR4"],
    "MESODERM": ["TBXT", "MESP1", "PDGFRA", "KDR"],
    "ECTODERM": ["PAX6", "SOX1", "NES"],
    "NEURAL_PROGENITOR": ["SOX2", "PAX6", "HES1", "NES"],
    "NEURON": ["DCX", "TUBB3", "MAP2", "SYN1"],
    "GLIA": ["GFAP", "S100B", "AQP4"],
    "INTESTINAL_EPITHELIUM": ["EPCAM", "CDX2", "KRT20"],
    "GOBLET": ["MUC2", "SPINK4", "TFF3"],
    "PANETH": ["LYZ", "DEFA5", "REG4"],
    "ENTEROCYTE": ["FABP1", "VIL1", "ALPI"],
    "HEPATOCYTE": ["ALB", "HNF4A", "CYP3A4"],
    "CHOLANGIOCYTE": ["KRT19", "EPCAM", "SOX9"],
    "NEPHRON_PROGENITOR": ["SIX2", "WT1", "CITED1"],
    "PODOCYTE": ["NPHS1", "NPHS2", "WT1"],
    "TUBULE": ["SLC12A1", "UMOD", "AQP2"],
    "CARDIOMYOCYTE": ["TNNT2", "MYH6", "ACTC1"],
    "ENDOTHELIAL": ["PECAM1", "CDH5", "VWF"],
    "MESENCHYMAL": ["COL1A1", "VIM", "PDGFRB"],
}

FACTORS = ["WNT", "RSPO", "BMP4", "NOGGIN", "FGF2", "FGF4", "FGF9", "FGF10",
           "ACTIVIN", "NODAL", "TGFB1", "SHH", "RA", "EGF", "HGF", "VEGF",
           "IGF1", "NOTCH_AG", "NOTCH_INH", "CHIR", "SB431542", "LDN193189",
           "DAPT", "Y27632", "OTHER"]

# crude keyword → factor/dose/time extraction from GEO metadata strings
FACTOR_PATTERNS = {
    "CHIR": r"chir[- ]?99021|chir\b", "SB431542": r"sb[- ]?431542",
    "LDN193189": r"ldn[- ]?193189", "DAPT": r"\bdapt\b",
    "FGF2": r"fgf2|fgf-2|basic fgf", "FGF4": r"fgf4|fgf-4",
    "EGF": r"\begf\b", "BMP4": r"bmp4|bmp-4", "NOGGIN": r"noggin",
    "WNT": r"\bwnt|wnt3a", "RA": r"retinoic acid|\bra\b",
    "ACTIVIN": r"activin", "VEGF": r"vegf", "Y27632": r"y[- ]?27632|rock",
}
DOSE_PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(nm|µm|um|ng/?ml|µg/?ml)", re.I)
TIME_PAT = re.compile(r"(?:day|d)\s*(\d+(?:\.\d+)?)|(?:hour|hr|h)\s*(\d+(?:\.\d+)?)", re.I)


@dataclass
class ScrnaConfig:
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    max_mt_pct: float = 20.0
    n_top_genes: int = 2000
    n_pcs: int = 30
    leiden_res: float = 0.6
    score_quantile: float = 0.7     # fraction of top-scoring cells kept per type
    min_cells_per_sample: int = 30


def load_count_matrix(path: Path, sample_col: str = "sample_id") -> "ad.AnnData":
    """
    Load a count matrix. Supports 10x .h5, mtx dir, or csv/tsv with a
    metadata column. Adjust for your GEO deposit format.
    """
    if not HAS_SCANPY:
        raise ImportError("pip install scanpy anndata")
    p = Path(path)
    if p.suffix == ".h5":
        adata = sc.read_10x_h5(str(p))
    elif p.is_dir() and (p / "matrix.mtx").exists():
        adata = sc.read_10x_mtx(str(p))
    elif p.suffix in (".csv", ".tsv", ".txt"):
        import pandas as pd
        sep = "," if p.suffix == ".csv" else "\t"
        df = pd.read_csv(p, sep=sep, index_col=0)
        adata = ad.AnnData(X=df.values.T, obs=pd.DataFrame(index=df.columns),
                           var=pd.DataFrame(index=df.index))
    else:
        raise ValueError(f"unsupported format: {p}")
    log.info(f"[load] {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def qc_normalize(adata: "ad.AnnData", cfg: ScrnaConfig) -> "ad.AnnData":
    """Standard QC + log-normalization [S1,S2]."""
    sc.pp.filter_cells(adata, min_genes=cfg.min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells_per_gene)
    mt = adata.var_names.str.upper().str.startswith(("MT-", "MT."))
    adata.var["mt"] = mt.values
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)
    before = adata.n_obs
    adata = adata[adata.obs.pct_counts_mt < cfg.max_mt_pct].copy()
    log.info(f"[qc] {before} → {adata.n_obs} cells after MT filter")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def cluster(adata: "ad.AnnData", cfg: ScrnaConfig) -> "ad.AnnData":
    """HVG → PCA → neighbors → Leiden [S1,S2]."""
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_top_genes,
                                flavor="seurat", subset=False)
    X = adata[:, adata.var.highly_variable].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(cfg.n_pcs, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_pcs=min(cfg.n_pcs, adata.obsm["X_pca"].shape[1]))
    sc.tl.leiden(adata, resolution=cfg.leiden_res)
    sc.tl.umap(adata)
    log.info(f"[cluster] {adata.obs['leiden'].nunique()} clusters")
    return adata


def annotate_marker_score(adata: "ad.AnnData",
                          markers: Optional[Dict[str, List[str]]] = None,
                          cfg: Optional[ScrnaConfig] = None) -> "ad.AnnData":
    """
    Score each cell against marker sets; assign the best-scoring fate.
    Uses scanpy.tl.score_genes per fate. Cells with weak signal → UNKNOWN.
    ⚠️ Marker-based annotation is approximate. Validate on UMAP.
    """
    cfg = cfg or ScrnaConfig()
    markers = markers or DEFAULT_MARKERS
    names = [f for f in FATES if f in markers]
    for f in names:
        present = [g for g in markers[f] if g in adata.var_names]
        if not present:
            adata.obs[f"score_{f}"] = 0.0
            continue
        sc.tl.score_genes(adata, present, score_name=f"score_{f}", use_raw=False)
    score_cols = [f"score_{f}" for f in names]
    S = adata.obs[score_cols].values
    best = np.argmax(S, axis=1)
    best_score = S[np.arange(len(S)), best]
    thr = np.quantile(best_score[best_score > 0], cfg.score_quantile) \
          if (best_score > 0).any() else 0.0
    labels = np.array([names[i] for i in best], dtype=object)
    labels[best_score < max(thr * 0.5, 1e-3)] = "UNKNOWN"
    adata.obs["fate"] = labels
    log.info(f"[annotate] fate distribution: "
             f"{dict(zip(*np.unique(labels, return_counts=True)))}")
    return adata


def composition_per_sample(adata: "ad.AnnData",
                           sample_col: str,
                           cfg: Optional[ScrnaConfig] = None) -> Dict[str, np.ndarray]:
    """Aggregate cells → composition vector per sample."""
    cfg = cfg or ScrnaConfig()
    if sample_col not in adata.obs.columns:
        raise KeyError(f"sample_col '{sample_col}' not in adata.obs. "
                       f"Available: {list(adata.obs.columns)}")
    out = {}
    for s in adata.obs[sample_col].unique():
        sub = adata.obs[adata.obs[sample_col] == s]
        if len(sub) < cfg.min_cells_per_sample:
            log.warning(f"[composition] sample {s}: only {len(sub)} cells, skipping")
            continue
        counts = sub["fate"].value_counts()
        vec = np.zeros(len(FATES), np.float32)
        for f, c in counts.items():
            if f in FATES:
                vec[FATES.index(f)] = c
        vec = vec / max(vec.sum(), 1)
        out[str(s)] = vec
    log.info(f"[composition] {len(out)} samples with >= {cfg.min_cells_per_sample} cells")
    return out


def parse_geo_metadata(meta_path: Path) -> List[Dict]:
    """
    Parse a GEO series-matrix Sample_* table into per-sample metadata,
    then HEURISTICALLY propose factor/dose/time.

    ⚠️ Every proposal must be checked against the paper. This parser only
       reduces manual work; it does not replace reading the methods.
    """
    rows = []
    text = gzip.open(meta_path, "rt", errors="ignore").read() \
        if str(meta_path).endswith(".gz") else meta_path.read_text(errors="ignore")
    samples: Dict[str, Dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("!Sample_"):
            continue
        key = line.split("\t")[0][1:]
        vals = [x.strip('"') for x in line.split("\t")[1:]]
        # first occurrence defines the ordered sample list for _characteristics
        samples.setdefault(key, {})
        for i, v in enumerate(vals):
            samples[key][f"col{i}"] = samples[key].get(f"col{i}", "") + "|" + v
    # build a per-sample joined blob for regex search
    geo_ids = samples.get("Sample_geo_accession", {})
    n = len(geo_ids)
    for i in range(n):
        gsm = geo_ids.get(f"col{i}", "").split("|")[0]
        blob = " ".join(samples[k].get(f"col{i}", "") for k in samples).lower()
        rec = {"gsm": gsm, "factor": None, "dose_nM": None, "time_h": None}
        for f, pat in FACTOR_PATTERNS.items():
            if re.search(pat, blob):
                rec["factor"] = f
                break
        m = DOSE_PAT.search(blob)
        if m:
            val, unit = float(m.group(1)), m.group(2).lower()
            factor = 1.0
            if unit in ("µm", "um", "mm"):
                factor = 1000.0
            elif unit in ("ng/ml", "µg/ml"):
                factor = np.nan   # cannot convert mass→M without MW
            rec["dose_nM"] = val * factor if factor == factor else None
        t = TIME_PAT.search(blob)
        if t:
            if t.group(1):
                rec["time_h"] = float(t.group(1)) * 24.0
            elif t.group(2):
                rec["time_h"] = float(t.group(2))
        rows.append(rec)
    log.info(f"[geo] parsed {len(rows)} samples; "
             f"{sum(r['factor'] for r in rows)} with a detected factor")
    return rows


def emit_partial_t2(composition: Dict[str, np.ndarray],
                    geo_rows: List[Dict],
                    gse: str, doi_hint: str, out: Path):
    """
    Write an UNVERIFIED partial T2 JSONL. You must then:
      • confirm factor/dose/time against the paper
      • add the real DOI
      • set verified=true
    Records are written with verified=false on purpose.
    """
    geo_by_gsm = {r["gsm"]: r for r in geo_rows}
    n_written = 0
    with open(out, "a") as f:
        for sample_id, comp in composition.items():
            g = geo_by_gsm.get(sample_id, {})
            rec = {
                "record_id": f"{gse}_{sample_id}",
                "factor": g.get("factor") or "OTHER",
                "dose_nM": g.get("dose_nM") or -1.0,
                "time_h": g.get("time_h") or -1.0,
                "composition": comp.tolist(),
                "source_gse": gse,
                "sample_id": sample_id,
                "doi": doi_hint,
                "organoid_type": None,
                "cell_type_calling": "marker",
                "n_cells": None,
                "notes": "AUTO-GENERATED — verify before use",
                "verified": False,
            }
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    log.info(f"[emit] wrote {n_written} UNVERIFIED records → {out}")
    log.warning("[emit] All records have verified=false. "
                "Check factor/dose/time against the paper, add DOI, "
                "then set verified=true before training.")


def selftest(args) -> bool:
    print("=" * 72)
    print("  scRNA-seq → composition — self-test (synthetic)")
    print("=" * 72)
    ok = True
    if not HAS_SCANPY:
        print("  scanpy not installed — running reduced numpy checks only")
    rng = np.random.default_rng(0)

    print("\n[T1] synthetic composition sums to 1")
    comp = rng.dirichlet(np.ones(len(FATES)))
    t = abs(comp.sum() - 1.0) < 1e-6 and len(comp) == len(FATES)
    print(f"   sum={comp.sum():.6f}  {'PASS' if t else 'FAIL'}"); ok &= t

    print("\n[T2] dose/time regex")
    txt = "treated with CHIR-99021 3uM for day 3"
    f = "CHIR" if re.search(FACTOR_PATTERNS["CHIR"], txt) else None
    m = DOSE_PAT.search(txt)
    dose = float(m.group(1)) * (1000.0 if m.group(2).lower() in ("µm", "um") else 1.0) if m else None
    t2 = TIME_PAT.search(txt)
    th = float(t2.group(1)) * 24 if t2 and t2.group(1) else None
    t = f == "CHIR" and abs(dose - 3000.0) < 1e-6 and abs(th - 72.0) < 1e-6
    print(f"   factor={f} dose={dose}nM time={th}h  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T3] marker sets cover all FATES or are omitted")
    missing = [f for f in FATES if f != "UNKNOWN" and f not in DEFAULT_MARKERS]
    t = len(missing) == 0
    print(f"   missing markers: {missing}  {'PASS' if t else 'CHECK'}")
    ok &= t

    if HAS_SCANPY:
        print("\n[T4] end-to-end on synthetic counts")
        n_cells, n_genes = 300, 200
        X = rng.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"GENE{i}" for i in range(n_genes)]
        adata.obs["sample_id"] = [f"GSM{i % 3}" for i in range(n_cells)]
        # inject a couple of marker genes so scoring is non-trivial
        for f, genes in [("NEURON", ["GENE0"]), ("ENDODERM", ["GENE1"])]:
            pass
        adata = qc_normalize(adata, ScrnaConfig(min_genes_per_cell=1,
                                                min_cells_per_gene=1,
                                                n_top_genes=min(50, n_genes)))
        adata.obs["fate"] = rng.choice(["NEURON", "ENDODERM", "UNKNOWN"], n_cells)
        comp = composition_per_sample(adata, "sample_id",
                                      ScrnaConfig(min_cells_per_sample=10))
        t = len(comp) == 3 and all(abs(v.sum() - 1) < 1e-5 for v in comp.values())
        print(f"   {len(comp)} samples, all compositions sum to 1: "
              f"{'PASS' if t else 'FAIL'}")
        ok &= t
    else:
        print("\n[T4] SKIP — scanpy not installed")

    print("\n" + "=" * 72)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 72)
    print("""
  REMEMBER:
    • This is standard bioinformatics. The project's math is tested in the
      MDOE predictor ablation, on the labels this pipeline produces.
    • Cell-type annotation must be reviewed on UMAP before trusting it.
    • Every (factor, dose, time) proposal must be checked against the paper.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "run"])
    p.add_argument("--counts", default=None, help="path to counts (.h5/.mtx dir/.csv)")
    p.add_argument("--sample-col", dest="sample_col", default="sample_id")
    p.add_argument("--geo-matrix", dest="geo_matrix", default=None)
    p.add_argument("--gse", default="GSE000000")
    p.add_argument("--doi", default="")
    p.add_argument("--out", default="data/t2.jsonl")
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if not a.counts:
        log.error("need --counts for run"); return
    cfg = ScrnaConfig()
    adata = load_count_matrix(Path(a.counts), a.sample_col)
    adata = qc_normalize(adata, cfg)
    adata = cluster(adata, cfg)
    adata = annotate_marker_score(adata, DEFAULT_MARKERS, cfg)
    comp = composition_per_sample(adata, a.sample_col, cfg)
    geo_rows = parse_geo_metadata(Path(a.geo_matrix)) if a.geo_matrix else []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    emit_partial_t2(comp, geo_rows, a.gse, a.doi, Path(a.out))
    log.info("Done. Next: verify records, then validate + to-mdoe.")


if __name__ == "__main__":
    main()
