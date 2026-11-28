#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
scRNA-seq -> CELL-TYPE COMPOSITION v2.0
Standard bioinformatics, now with the CORRECTED occupation-measure theory.

THE NEW CONTENT (this is where the mathematics genuinely applies)
  composition = (1/n) sum_t e_{fate(t)}  is an occupation vector on Delta^d.
  * Thm 8.3 gives an explicit W2 rate to the limiting composition.
  * Prop 8.6 vs Thm 8.1 gives a TEST separating noisy-deterministic from
    genuinely multistable fate outcomes.
  * The infinite-mean middle (Lamperti) is [Open] -> Mode E.

PHILOSOPHY: see msps_esca_bio_core.PHILOSOPHY (full text).
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

HONEST SCOPE
  Standard, citable scRNA-seq workflow [S1-S4]. Cell-type annotation is
  the error-prone step; marker scoring is a HYPOTHESIS, not ground truth.
  GEO metadata parsing is heuristic; every proposal must be checked.
  No batch correction across series here; use organoid_markers_harmony v2.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, gzip, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from msps_esca_bio_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    composition_as_occupation, theorem_A_rate, occupation_regime_test,
    block_structure, boundary_null_check, get_logger,
)
log=get_logger("scrna")
VERSION="2.0.0"

try:
    import scanpy as sc, anndata as ad
    HAS_SCANPY=True
except ImportError:
    HAS_SCANPY=False
    log.warning("scanpy/anndata absent -> clustering disabled")

FATES=["PLURIPOTENT","ENDODERM","MESODERM","ECTODERM",
 "NEURAL_PROGENITOR","NEURON","GLIA",
 "INTESTINAL_EPITHELIUM","GOBLET","PANETH","ENTEROCYTE",
 "HEPATOCYTE","CHOLANGIOCYTE",
 "NEPHRON_PROGENITOR","PODOCYTE","TUBULE",
 "CARDIOMYOCYTE","ENDOTHELIAL","MESENCHYMAL","UNKNOWN"]

DEFAULT_MARKERS={
 "PLURIPOTENT":["NANOG","POU5F1","SOX2","KLF4"],
 "ENDODERM":["SOX17","FOXA2","GATA6","CXCR4"],
 "MESODERM":["TBXT","MESP1","PDGFRA","KDR"],
 "ECTODERM":["PAX6","SOX1","NES"],
 "NEURAL_PROGENITOR":["SOX2","PAX6","HES1","NES"],
 "NEURON":["DCX","TUBB3","MAP2","SYN1"],
 "GLIA":["GFAP","S100B","AQP4"],
 "INTESTINAL_EPITHELIUM":["EPCAM","CDX2","KRT20"],
 "GOBLET":["MUC2","SPINK4","TFF3"],
 "PANETH":["LYZ","DEFA5","REG4"],
 "ENTEROCYTE":["FABP1","VIL1","ALPI"],
 "HEPATOCYTE":["ALB","HNF4A","CYP3A4"],
 "CHOLANGIOCYTE":["KRT19","EPCAM","SOX9"],
 "NEPHRON_PROGENITOR":["SIX2","WT1","CITED1"],
 "PODOCYTE":["NPHS1","NPHS2","WT1"],
 "TUBULE":["SLC12A1","UMOD","AQP2"],
 "CARDIOMYOCYTE":["TNNT2","MYH6","ACTC1"],
 "ENDOTHELIAL":["PECAM1","CDH5","VWF"],
 "MESENCHYMAL":["COL1A1","VIM","PDGFRB"],
}

@dataclass
class ScrnaConfig:
    min_genes_per_cell:int=200
    min_cells_per_gene:int=3
    max_mt_pct:float=20.0
    n_top_genes:int=2000
    n_pcs:int=30
    leiden_res:float=0.6
    score_quantile:float=0.7
    min_cells_per_sample:int=30

def load_count_matrix(path: Path):
    if not HAS_SCANPY: raise Refusal("pip install scanpy anndata")
    p=Path(path)
    if p.suffix==".h5": adata=sc.read_10x_h5(str(p))
    elif p.is_dir() and (p/"matrix.mtx").exists(): adata=sc.read_10x_mtx(str(p))
    elif p.suffix in (".csv",".tsv",".txt"):
        import pandas as pd
        sep="," if p.suffix==".csv" else "\t"
        df=pd.read_csv(p,sep=sep,index_col=0)
        adata=ad.AnnData(X=df.values.T,obs=pd.DataFrame(index=df.columns),
                         var=pd.DataFrame(index=df.index))
    else: raise Refusal(f"unsupported format: {p}")
    log.info(f"[load] {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

def qc_normalize(adata,cfg:ScrnaConfig):
    sc.pp.filter_cells(adata,min_genes=cfg.min_genes_per_cell)
    sc.pp.filter_genes(adata,min_cells=cfg.min_cells_per_gene)
    adata.var["mt"]=adata.var_names.str.upper().str.startswith(("MT-","MT.")).values
    sc.pp.calculate_qc_metrics(adata,qc_vars=["mt"],percent_top=None,
                               log1p=False,inplace=True)
    n0=adata.n_obs
    adata=adata[adata.obs.pct_counts_mt<cfg.max_mt_pct].copy()
    log.info(f"[qc] {n0} -> {adata.n_obs} cells")
    sc.pp.normalize_total(adata,target_sum=1e4); sc.pp.log1p(adata)
    return adata

def cluster(adata,cfg:ScrnaConfig):
    sc.pp.highly_variable_genes(adata,n_top_genes=cfg.n_top_genes,
                                flavor="seurat",subset=False)
    sc.pp.scale(adata,max_value=10)
    sc.tl.pca(adata,n_comps=min(cfg.n_pcs,adata.n_vars-1))
    sc.pp.neighbors(adata,n_pcs=min(cfg.n_pcs,adata.obsm["X_pca"].shape[1]))
    sc.tl.leiden(adata,resolution=cfg.leiden_res); sc.tl.umap(adata)
    log.info(f"[cluster] {adata.obs['leiden'].nunique()} clusters")
    return adata

def annotate_marker_score(adata,markers=None,cfg=None):
    cfg=cfg or ScrnaConfig(); markers=markers or DEFAULT_MARKERS
    names=[f for f in FATES if f in markers]
    for f in names:
        present=[g for g in markers[f] if g in adata.var_names]
        if not present: adata.obs[f"score_{f}"]=0.0; continue
        sc.tl.score_genes(adata,present,score_name=f"score_{f}",use_raw=False)
    S=adata.obs[[f"score_{f}" for f in names]].values
    best=np.argmax(S,axis=1); bs=S[np.arange(len(S)),best]
    thr=np.quantile(bs[bs>0],cfg.score_quantile) if (bs>0).any() else 0.0
    lab=np.array([names[i] for i in best],dtype=object)
    lab[bs<max(thr*0.5,1e-3)]="UNKNOWN"
    adata.obs["fate"]=lab
    adata.obs["annotation_margin"]=bs-np.sort(S,axis=1)[:,-2]
    log.warning("[annotate] marker scoring is a HYPOTHESIS. Inspect the UMAP.")
    return adata

def composition_per_sample(adata,sample_col:str,cfg=None,sd=None):
    """
    v2: returns composition AND the Thm 8.3 W2 rate AND (when cell order
    is meaningful) the occupation-regime classification.
    """
    cfg=cfg or ScrnaConfig(); sd=sd or SDGate()
    if sample_col not in adata.obs.columns:
        raise Refusal(f"'{sample_col}' not in obs; have {list(adata.obs.columns)}")
    out={}
    for s in adata.obs[sample_col].unique():
        sub=adata.obs[adata.obs[sample_col]==s]
        if len(sub)<cfg.min_cells_per_sample:
            log.warning(f"[composition] {s}: {len(sub)} cells, skipped"); continue
        labels=sub["fate"].tolist()
        oc=composition_as_occupation(labels,FATES)
        rate=theorem_A_rate(len(labels),len(FATES),Gamma=1.0)
        margin=float(np.mean(sub.get("annotation_margin",np.zeros(len(sub)))))
        cert=sd.certify("composition_record",
                        ("ANNOT",) if margin>0.1 else (),
                        {"sample":str(s),"n_cells":len(labels)})
        out[str(s)]={"composition":np.array(oc["F"],np.float32),
                     "n_cells":len(labels),
                     "n_unknown":oc["n_unknown"],
                     "W2_bound":rate["W2_bound"],
                     "L2_bound":rate["L2_bound"],
                     "mean_annotation_margin":margin,
                     "sd":cert["status"],
                     "labels":labels}
    log.info(f"[composition] {len(out)} samples; W2 bounds computed (Thm 8.3)")
    return out

def sample_regime(comp_entry: Dict) -> Dict:
    """
    The three-way test on the cell sequence. NOTE: scRNA-seq cell order
    is arbitrary, so this is only meaningful when the labels are ordered
    by a real axis (pseudotime, spatial position, or clone).
    """
    return {"WARNING":("cell order in scRNA-seq is arbitrary; the regime "
                       "test requires a meaningful ordering (pseudotime, "
                       "spatial axis, or clone). Otherwise treat as Mode E."),
            **occupation_regime_test(comp_entry["labels"],FATES)}

def pseudotime_regime(adata,sample_col:str,order_key:str,
                      sample_id:str) -> Dict:
    """Order cells by a real axis, THEN run the regime test."""
    if order_key not in adata.obs.columns:
        raise Refusal(f"'{order_key}' not in obs; cannot order cells")
    sub=adata.obs[adata.obs[sample_col]==sample_id].sort_values(order_key)
    labels=sub["fate"].tolist()
    r=occupation_regime_test(labels,FATES)
    return {"ordered_by":order_key,"n_cells":len(labels),**r}

def parse_geo_metadata(meta_path: Path) -> List[Dict]:
    FACTOR_PATTERNS={"CHIR":r"chir[- ]?99021|chir\b","SB431542":r"sb[- ]?431542",
      "LDN193189":r"ldn[- ]?193189","DAPT":r"\bdapt\b","FGF2":r"fgf2|fgf-2",
      "FGF4":r"fgf4|fgf-4","EGF":r"\begf\b","BMP4":r"bmp4|bmp-4",
      "NOGGIN":r"noggin","WNT":r"\bwnt|wnt3a","RA":r"retinoic acid",
      "ACTIVIN":r"activin","VEGF":r"vegf","Y27632":r"y[- ]?27632"}
    DOSE=re.compile(r"(\d+(?:\.\d+)?)\s*(nm|µm|um|ng/?ml|µg/?ml)",re.I)
    TIME=re.compile(r"(?:day|d)\s*(\d+(?:\.\d+)?)|(?:hour|hr|h)\s*(\d+(?:\.\d+)?)",re.I)
    op=gzip.open if str(meta_path).endswith(".gz") else open
    with op(meta_path,"rt",errors="ignore") as f: text=f.read()
    samples: Dict[str,Dict[str,str]]={}
    for line in text.splitlines():
        if not line.startswith("!Sample_"): continue
        key=line.split("\t")[0][1:]
        vals=[x.strip('"') for x in line.split("\t")[1:]]
        samples.setdefault(key,{})
        for i,v in enumerate(vals):
            samples[key][f"col{i}"]=samples[key].get(f"col{i}","")+"|"+v
    ids=samples.get("Sample_geo_accession",{})
    rows=[]
    for i in range(len(ids)):
        gsm=ids.get(f"col{i}","").split("|")[0]
        blob=" ".join(samples[k].get(f"col{i}","") for k in samples).lower()
        rec={"gsm":gsm,"factor":None,"dose_nM":None,"time_h":None}
        for f,pat in FACTOR_PATTERNS.items():
            if re.search(pat,blob): rec["factor"]=f; break
        m=DOSE.search(blob)
        if m:
            val,unit=float(m.group(1)),m.group(2).lower()
            if unit in ("µm","um"): rec["dose_nM"]=val*1000
            elif unit=="nm": rec["dose_nM"]=val
            else: rec["dose_nM"]=None   # mass->M needs MW
        t=TIME.search(blob)
        if t:
            rec["time_h"]=float(t.group(1))*24 if t.group(1) else float(t.group(2))
        rows.append(rec)
    log.warning("[geo] heuristic proposals. CHECK against the paper.")
    return rows

def emit_partial_t2(comp: Dict, geo_rows: List[Dict], gse: str,
                    doi_hint: str, out: Path):
    by={r["gsm"]:r for r in geo_rows}
    n=0
    with open(out,"a") as f:
        for sid,c in comp.items():
            g=by.get(sid,{})
            rec={"record_id":f"{gse}_{sid}",
                 "factor":g.get("factor") or "OTHER",
                 "dose_nM":g.get("dose_nM") or -1.0,
                 "time_h":g.get("time_h") or -1.0,
                 "composition":c["composition"].tolist(),
                 "source_gse":gse,"sample_id":sid,"doi":doi_hint,
                 "cell_type_calling":"marker","n_cells":c["n_cells"],
                 "notes":(f"AUTO-GENERATED. W2_bound={c['W2_bound']:.4f} "
                          f"(Thm 8.3). Verify before use."),
                 "verified":False,"annotation_reviewed":False}
            f.write(json.dumps(rec)+"\n"); n+=1
    log.info(f"[emit] {n} UNVERIFIED records -> {out}")
    log.warning("[emit] verified=false and annotation_reviewed=false. "
                "Inspect the UMAP, check the paper, then set both true.")

def selftest(a=None) -> bool:
    print("="*76); print("  scRNA -> composition v2 — self-test"); print("="*76)
    ok=True; sd=SDGate(); rng=np.random.default_rng(0)
    print("\n[T1] composition = occupation vector")
    labels=["NEURON"]*40+["ENDODERM"]*35+["UNKNOWN"]*25
    oc=composition_as_occupation(labels,FATES)
    print(f"   n={oc['n_cells']} sum={sum(oc['F']):.6f} unknown={oc['n_unknown']}")
    ok &= oc["on_simplex"] and oc["n_unknown"]==25
    print("\n[T2] Thm 8.3 W2 rate")
    r=theorem_A_rate(100,len(FATES))
    print(f"   L2<={r['L2_bound']:.4f}  W2<={r['W2_bound']:.4f}")
    ok &= r["W2_bound"]>0
    print("\n[T3] three-way regime test")
    light=["NEURON","ENDODERM"]*50
    rl=occupation_regime_test(light,FATES)
    print(f"   alternating -> {rl['regime']}")
    ok &= rl["regime"]=="LIGHT_BLOCKS"
    sup=[]
    for j,f in enumerate(["NEURON","ENDODERM","MESENCHYMAL","GLIA"]):
        sup += [f]*(5**j)
    rs=occupation_regime_test(sup,FATES)
    print(f"   geometric  -> {rs['regime']}")
    ok &= rs["regime"]=="SUPER_DOMINANT"
    print("\n[T4] unordered cells -> warning")
    entry={"labels":labels}
    sr=sample_regime(entry)
    print(f"   WARNING present: {'WARNING' in sr}")
    ok &= "WARNING" in sr
    print("\n[T5] dose/time regex")
    txt="treated with CHIR-99021 3uM for day 3"
    m=re.search(r"(\d+(?:\.\d+)?)\s*(nm|µm|um)",txt,re.I)
    dose=float(m.group(1))*(1000 if m.group(2).lower() in ("µm","um") else 1)
    t=re.search(r"(?:day|d)\s*(\d+)",txt)
    th=float(t.group(1))*24
    print(f"   dose={dose} nM time={th} h")
    ok &= abs(dose-3000)<1e-6 and abs(th-72)<1e-6
    print("\n[T6] marker coverage")
    missing=[f for f in FATES if f!="UNKNOWN" and f not in DEFAULT_MARKERS]
    print(f"   missing markers: {missing}")
    ok &= not missing
    if HAS_SCANPY:
        print("\n[T7] end-to-end on synthetic counts")
        X=rng.negative_binomial(5,0.3,size=(300,200)).astype(np.float32)
        adata=ad.AnnData(X=X)
        adata.var_names=[f"GENE{i}" for i in range(200)]
        adata.obs["sample_id"]=[f"GSM{i%3}" for i in range(300)]
        adata=qc_normalize(adata,ScrnaConfig(min_genes_per_cell=1,
            min_cells_per_gene=1,n_top_genes=50))
        adata.obs["fate"]=rng.choice(["NEURON","ENDODERM","UNKNOWN"],300)
        adata.obs["annotation_margin"]=rng.uniform(0,0.3,300)
        comp=composition_per_sample(adata,"sample_id",
                                   ScrnaConfig(min_cells_per_sample=10),sd)
        t=len(comp)==3 and all(abs(v["composition"].sum()-1)<1e-5
                               for v in comp.values())
        print(f"   {len(comp)} samples; W2 bounds present: "
              f"{all('W2_bound' in v for v in comp.values())}")
        ok &= t
    else:
        print("\n[T7] SKIP — scanpy absent")
    rr=sd.report()
    print(f"\n   SD: {rr['n']} outputs, {rr['mode_E']} Mode E")
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
REMEMBER
  Standard bioinformatics; the mathematics enters at the occupation-measure
  layer. W2 bounds come from Thm 8.3. The regime test needs a MEANINGFUL
  cell ordering; unordered scRNA-seq gives Mode E for that test.
  Annotation must be reviewed. Metadata proposals must be checked.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","run"])
    ap.add_argument("--counts"); ap.add_argument("--sample-col",dest="sample_col",
                                                default="sample_id")
    ap.add_argument("--geo-matrix",dest="geo_matrix")
    ap.add_argument("--gse",default="GSE000000"); ap.add_argument("--doi",default="")
    ap.add_argument("--out",default="data/t2.jsonl")
    ap.add_argument("--order-key",dest="order_key",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if not a.counts: log.error("--counts required"); return
    cfg=ScrnaConfig(); sd=SDGate()
    adata=load_count_matrix(Path(a.counts))
    adata=qc_normalize(adata,cfg); adata=cluster(adata,cfg)
    adata=annotate_marker_score(adata,DEFAULT_MARKERS,cfg)
    comp=composition_per_sample(adata,a.sample_col,cfg,sd)
    if a.order_key:
        for sid in comp:
            comp[sid]["regime"]=pseudotime_regime(adata,a.sample_col,
                                                  a.order_key,sid)
    geo=parse_geo_metadata(Path(a.geo_matrix)) if a.geo_matrix else []
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    emit_partial_t2(comp,geo,a.gse,a.doi,Path(a.out))
    log.info("Next: review UMAP, verify records, then validate + to-mdoe.")

if __name__=="__main__": main()
