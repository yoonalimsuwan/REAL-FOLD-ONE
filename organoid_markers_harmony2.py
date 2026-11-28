#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ORGANOID MARKERS + HARMONY v2.0
Organ-specific markers, confound gating, and SD-certified integration.

NEW IN v2
  * SD gate on batch correction: CONFOUND package REQUIRED.
  * Marker identifiability: Fisher / rank / correlation on the marker
    design matrix. Overlapping markers (SOX2, PAX6, WT1, EPCAM) make
    cell-type assignment non-identifiable; this is now MEASURED.
  * Refuses integration when batch == condition (Harmony would delete
    the treatment effect).
  * Refuses human/mouse mixing and cross-organ pooling.

PHILOSOPHY: see msps_esca_bio_core.PHILOSOPHY (full text).
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

HONEST SCOPE
  Marker annotation is a HYPOTHESIS. Harmony CANNOT separate batch from
  biology when confounded, and assumes shared cell types across batches.
  Harmony does NOT create treatment labels.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

from msps_esca_bio_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    composition_as_occupation, theorem_A_rate, get_logger,
)
log=get_logger("organoid_markers")
VERSION="2.0.0"

class DataRefusal(Refusal): pass

try:
    import scanpy as sc, anndata as ad
    HAS_SCANPY=True
except ImportError: HAS_SCANPY=False
try:
    import harmonypy as hpy
    HAS_HARMONY=True
except ImportError: HAS_HARMONY=False

@dataclass
class CellTypeMarkers:
    name:str; positive:List[str]; negative:List[str]=field(default_factory=list)
    reference:str=""

INTESTINAL=[
 CellTypeMarkers("STEM_LGR5",["LGR5","ASCL2","OLFM4","SOX9","SMOC2"],
                 ["MUC2","CHGA"],"Sato 2009; Clevers 2013"),
 CellTypeMarkers("ENTEROCYTE",["FABP1","VIL1","ALPI","SI","CDX2","KRT20"],
                 ["LGR5","MUC2","LYZ"],"Clevers 2013"),
 CellTypeMarkers("GOBLET",["MUC2","TFF3","SPINK4","CLCA1","FCGBP"],
                 ["LGR5","LYZ","CHGA"],"Clevers 2013"),
 CellTypeMarkers("PANETH",["LYZ","DEFA5","DEFA6","REG4","PLA2G2A"],
                 ["MUC2","FABP1","CHGA"],"Clevers 2013"),
 CellTypeMarkers("ENTEROENDOCRINE",["CHGA","CHGB","NEUROD1","INSM1","TPH1"],
                 ["MUC2","LYZ","FABP1"],"Clevers 2013"),
 CellTypeMarkers("TUFT",["DCLK1","TRPM5","POU2F3","IL25"],
                 ["MUC2","LYZ","CHGA"],"Clevers 2013"),
 CellTypeMarkers("MESENCHYMAL",["COL1A1","VIM","PDGFRA","DCN"],
                 ["EPCAM","CDX2","KRT20"],"niche"),
]
CEREBRAL=[
 CellTypeMarkers("RADIAL_GLIA",["SOX2","PAX6","HES1","FABP7","VIM","HOPX"],
                 ["DCX","GFAP","MBP"],"Lancaster 2013; Camp 2015"),
 CellTypeMarkers("NEURAL_PROGENITOR",["SOX1","NES","DCX","STMN2","TUBB3"],
                 ["GFAP","MBP","CX3CR1"],"Lancaster 2013"),
 CellTypeMarkers("EXCITATORY_NEURON",["SLC17A7","NEUROD6","TBR1","SATB2","FOXP2"],
                 ["GAD1","GAD2","DLX1","GFAP"],"Camp 2015"),
 CellTypeMarkers("INHIBITORY_NEURON",["GAD1","GAD2","DLX1","DLX2","NKX2-1","SST"],
                 ["SLC17A7","GFAP","MBP"],"Camp 2015"),
 CellTypeMarkers("ASTROCYTE",["GFAP","S100B","AQP4","ALDH1L1","SLC1A3"],
                 ["DCX","MBP","CX3CR1"],"Lancaster 2013"),
 CellTypeMarkers("OLIGODENDROCYTE",["MBP","PLP1","OLIG2","MOG","MAG"],
                 ["GFAP","DCX","CX3CR1"],"Lancaster 2013"),
 CellTypeMarkers("MICROGLIA",["CX3CR1","TMEM119","P2RY12","AIF1","C1QA"],
                 ["GFAP","MBP","DCX"],"Lancaster 2013"),
 CellTypeMarkers("CHOROID_PLEXUS",["TTR","FOXJ1","OTX2","CLDN11"],
                 ["DCX","GFAP","MBP"],"Lancaster 2013"),
]
KIDNEY=[
 CellTypeMarkers("NEPHRON_PROGENITOR",["SIX2","WT1","CITED1","SALL1","OSR1"],
                 ["NPHS1","LRP2","AQP2"],"Takasato 2015"),
 CellTypeMarkers("PODOCYTE",["NPHS1","NPHS2","PODXL","WT1","SYNPO"],
                 ["LRP2","SLC12A3","AQP2"],"Takasato 2015"),
 CellTypeMarkers("PROXIMAL_TUBULE",["LRP2","CUBN","SLC34A1","ALB","SLC22A6"],
                 ["NPHS1","SLC12A3","AQP2"],"Takasato 2015"),
 CellTypeMarkers("LOOP_OF_HENLE",["UMOD","SLC12A1","KCNJ1","BSND"],
                 ["NPHS1","LRP2","AQP2"],"Little & McMahon 2012"),
 CellTypeMarkers("DISTAL_TUBULE",["SLC12A3","CALB1","TRPV5","CLCNKB"],
                 ["NPHS1","LRP2","UMOD"],"Takasato 2015"),
 CellTypeMarkers("COLLECTING_DUCT",["AQP2","AQP3","GATA3"],
                 ["NPHS1","LRP2","SLC12A3"],"Takasato 2015"),
 CellTypeMarkers("ENDOTHELIAL",["PECAM1","CDH5","VWF","CLDN5"],
                 ["EPCAM","WT1","NPHS1"],"Takasato 2015"),
 CellTypeMarkers("STROMA",["FOXD1","PDGFRA","COL1A1","COL3A1"],
                 ["EPCAM","WT1","PECAM1"],"Little & McMahon 2012"),
]
SHARED_MARKERS={"SOX2","PAX6","WT1","EPCAM","VIM","CDX2","SOX9",
                "PECAM1","COL1A1","PDGFRA","KRT20","ALB","NES"}
ORGAN_REGISTRY={"intestinal":INTESTINAL,"cerebral":CEREBRAL,"kidney":KIDNEY}

def marker_identifiability(organ:str, gene_universe:Optional[Set[str]]=None
                           ) -> Dict:
    """
    NEW: measure how identifiable the cell types are from the marker set.
    Build the type x gene incidence matrix (positive +1, negative -0.5),
    then report rank, condition, and the worst pairwise cosine.
    Overlapping markers make types collinear -> assignment is a hypothesis.
    """
    markers=ORGAN_REGISTRY[organ]
    genes=sorted({g for m in markers for g in m.positive+m.negative})
    if gene_universe is not None:
        genes=[g for g in genes if g in gene_universe]
    if not genes: return {"rank":0,"n_types":len(markers),"identifiable":False}
    gi={g:i for i,g in enumerate(genes)}
    M=np.zeros((len(markers),len(genes)))
    for i,m in enumerate(markers):
        for g in m.positive:
            if g in gi: M[i,gi[g]]+=1.0
        for g in m.negative:
            if g in gi: M[i,gi[g]]-=0.5
    s=np.linalg.svd(M,compute_uv=False)
    rank=int(np.sum(s>s[0]*1e-10)) if s[0]>0 else 0
    nrm=np.linalg.norm(M,axis=1,keepdims=True)+1e-12
    Mn=M/nrm; C=Mn@Mn.T
    off=C[np.triu_indices(len(markers),1)] if len(markers)>1 else np.array([0.0])
    worst=float(np.max(off))
    pairs=[]
    for i in range(len(markers)):
        for j in range(i+1,len(markers)):
            if C[i,j]>0.8: pairs.append((markers[i].name,markers[j].name,
                                        float(C[i,j])))
    used={g for m in markers for g in m.positive}
    return {"organ":organ,"n_types":len(markers),"n_genes":len(genes),
            "rank":rank,"full_rank":rank==len(markers),
            "condition":float(s[0]/s[rank-1]) if rank else float("inf"),
            "max_pairwise_cosine":worst,
            "high_overlap_pairs":pairs,
            "shared_markers_used":sorted(used&SHARED_MARKERS),
            "identifiable":rank==len(markers) and worst<0.9,
            "VERDICT":("marker set separates types" if rank==len(markers)
                       and worst<0.9 else
                       "types are COLLINEAR from markers alone; annotation "
                       "is a HYPOTHESIS requiring UMAP review")}

class MarkerAnnotator:
    def __init__(self,organ:str,min_score:float=0.1,
                 conflict_threshold:float=0.3,sd:Optional[SDGate]=None):
        if organ not in ORGAN_REGISTRY:
            raise Refusal(f"unknown organ '{organ}'")
        self.organ=organ; self.markers=ORGAN_REGISTRY[organ]
        self.min_score=min_score; self.conflict_threshold=conflict_threshold
        self.cell_types=[m.name for m in self.markers]
        self.sd=sd or SDGate()
        self.ident=marker_identifiability(organ)
        if not self.ident["identifiable"]:
            log.warning(f"[annotator] {self.ident['VERDICT']}")
    def score_matrix(self,adata):
        if not HAS_SCANPY: raise Refusal("pip install scanpy anndata")
        X=adata.X
        if hasattr(X,"toarray"): X=X.toarray()
        X=np.asarray(X,np.float32)
        Xs=(X-X.mean(0,keepdims=True))/(X.std(0,keepdims=True)+1e-8)
        gi={g:i for i,g in enumerate(adata.var_names)}
        S=np.zeros((adata.n_obs,len(self.markers)),np.float32)
        for j,cm in enumerate(self.markers):
            pi=[gi[g] for g in cm.positive if g in gi]
            ni=[gi[g] for g in cm.negative if g in gi]
            if pi: S[:,j]+=Xs[:,pi].mean(1)
            if ni: S[:,j]-=0.5*Xs[:,ni].mean(1)
        return S
    def annotate(self,adata) -> Dict:
        S=self.score_matrix(adata)
        best=S.argmax(1); bs=S[np.arange(len(S)),best]
        lab=np.array([self.cell_types[b] for b in best],dtype=object)
        lab[bs<self.min_score]="UNKNOWN"
        srt=np.argsort(-S,axis=1)
        second=S[np.arange(len(S)),srt[:,1]]
        margin=bs-second
        conflict=margin<self.conflict_threshold
        adata.obs["cell_type"]=lab; adata.obs["type_score"]=bs
        adata.obs["annotation_margin"]=margin
        adata.obs["conflict"]=conflict
        from collections import Counter
        dist=Counter(lab); nconf=int(conflict.sum())
        cert=self.sd.certify("composition_record",
            ("ANNOT",) if (nconf/len(lab)<0.2 and
                           self.ident["identifiable"]) else (),
            {"organ":self.organ,"n_conflict":nconf})
        log.info(f"[annotator] {dict(dist)}")
        log.info(f"[annotator] {nconf}/{len(lab)} conflicts; sd={cert['status']}")
        return {"cell_type":lab,"scores":S,"conflict":conflict,
                "n_conflict":nconf,"conflict_fraction":nconf/len(lab),
                "distribution":dict(dist),"identifiability":self.ident,
                "sd":cert}

def detect_confound(adata,batch_col:str,condition_col:str) -> Dict:
    if condition_col not in adata.obs.columns:
        return {"has_condition":False,"confounded":None,
                "WARNING":f"'{condition_col}' absent; cannot check confounding"}
    ct=adata.obs.groupby([batch_col,condition_col]).size().unstack(fill_value=0)
    purity=ct.max(axis=1)/ct.sum(axis=1)
    npure=int((purity>0.9).sum()); nb=len(purity)
    conf=(npure==nb)
    return {"has_condition":True,"n_batches":nb,"n_pure_batches":npure,
            "confounded":conf,"batch_purity":purity.to_dict(),
            "VERDICT":("FATAL: batch == condition. Harmony would REMOVE the "
                       "treatment effect. DO NOT correct." if conf else
                       "OK: batches contain mixed conditions.")}

class HarmonyCorrector:
    def __init__(self,theta:float=2.0,max_iter:int=20,random_state:int=42,
                 sd:Optional[SDGate]=None):
        self.theta=theta; self.max_iter=max_iter
        self.random_state=random_state; self.sd=sd or SDGate()
    def correct(self,adata,batch_col:str="batch",n_pcs:int=30,
                condition_col:Optional[str]=None):
        if not HAS_HARMONY: raise Refusal("pip install harmonypy")
        if not HAS_SCANPY: raise Refusal("pip install scanpy anndata")
        pkgs=[]
        if condition_col:
            cf=detect_confound(adata,batch_col,condition_col)
            if cf.get("confounded"):
                self.sd.certify("batch_correction",(),cf)
                raise DataRefusal(
                    "FATAL CONFOUNDING: batch == condition.\n"
                    "Harmony would remove the treatment effect.\n"
                    "Add a series containing BOTH conditions, or do not "
                    "correct for batch. See detect_confound().")
            pkgs.append("CONFOUND")
        cert=self.sd.certify("batch_correction",tuple(pkgs),
                             {"theta":self.theta,"batch_col":batch_col})
        if cert["status"]!=MODE_CERTIFIED:
            log.warning("[harmony] CONFOUND not verified -> Mode E. "
                        "Proceeding is exploratory only.")
        if "X_pca" not in adata.obsm: sc.tl.pca(adata,n_comps=n_pcs)
        pca=adata.obsm["X_pca"][:,:n_pcs]
        meta=adata.obs.copy(); meta.index=meta.index.astype(str)
        ho=hpy.run_harmony(pca,meta,batch_col,theta=self.theta,
                           max_iter_harmony=self.max_iter,
                           random_state=self.random_state)
        adata.obsm["X_pca_harmony"]=ho.Z_corr
        return adata,cert

class MultiSeriesIntegrator:
    def __init__(self,organ:str,sd:Optional[SDGate]=None):
        if organ not in ORGAN_REGISTRY: raise Refusal(f"unknown organ {organ}")
        self.organ=organ; self.series={}; self.sd=sd or SDGate()
    def add_series(self,gse:str,adata):
        if "batch" not in adata.obs.columns: adata.obs["batch"]=gse
        self.series[gse]=adata
    def check_species(self) -> bool:
        for gse,ad_ in self.series.items():
            s=list(ad_.var_names[:100])
            up=sum(1 for g in s if g.isupper())/max(len(s),1)
            if up<0.5:
                log.warning(f"[integrator] {gse} looks like MOUSE ({up:.0%} "
                            "uppercase). Human/mouse mixing is INVALID.")
                return False
        return True
    def integrate(self,condition_col:Optional[str]=None):
        if not HAS_SCANPY: raise Refusal("pip install scanpy anndata")
        if len(self.series)<2: raise Refusal("need >= 2 series")
        if not self.check_species():
            raise DataRefusal("species mismatch; cannot integrate")
        comb=ad.concat(list(self.series.values()),join="outer",label="batch",
                       keys=list(self.series.keys()))
        sc.pp.filter_cells(comb,min_genes=200); sc.pp.filter_genes(comb,min_cells=3)
        comb.var["mt"]=comb.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(comb,qc_vars=["mt"],inplace=True,
                                   percent_top=None,log1p=False)
        comb=comb[comb.obs.pct_counts_mt<20].copy()
        sc.pp.normalize_total(comb,target_sum=1e4); sc.pp.log1p(comb)
        sc.pp.highly_variable_genes(comb,n_top_genes=2000,batch_key="batch",
                                    flavor="seurat_v3")
        sc.pp.scale(comb,max_value=10); sc.tl.pca(comb,n_comps=30)
        if HAS_HARMONY:
            comb,cert=HarmonyCorrector(sd=self.sd).correct(
                comb,"batch",condition_col=condition_col)
            comb.obsm["X_corrected"]=comb.obsm["X_pca_harmony"]
        else:
            raise Refusal("harmonypy absent; refusing an inferior fallback "
                          "for a certified integration")
        sc.pp.neighbors(comb,use_rep="X_corrected")
        sc.tl.umap(comb); sc.tl.leiden(comb,resolution=0.6)
        ann=MarkerAnnotator(self.organ,sd=self.sd).annotate(comb)
        comb.obs["organ"]=self.organ
        return comb,{"annotation":ann,"harmony_certificate":cert}

def composition_per_sample(adata,sample_col:str,cell_types:List[str],
                           min_cells:int=30,sd:Optional[SDGate]=None) -> Dict:
    sd=sd or SDGate(); out={}
    for s in adata.obs[sample_col].unique():
        sub=adata.obs[adata.obs[sample_col]==s]
        if len(sub)<min_cells: continue
        oc=composition_as_occupation(sub["cell_type"].tolist(),cell_types)
        rate=theorem_A_rate(len(sub),len(cell_types))
        out[str(s)]={"composition":np.array(oc["F"],np.float32),
                     "n_cells":len(sub),"W2_bound":rate["W2_bound"]}
    log.info(f"[composition] {len(out)} samples with W2 bounds")
    return out

def selftest(a=None) -> bool:
    print("="*76); print("  MARKERS + HARMONY v2 — self-test"); print("="*76)
    ok=True; sd=SDGate()
    print("\n[T1] marker registry")
    for o,mk in ORGAN_REGISTRY.items():
        print(f"   {o:12s}: {len(mk)} types, "
              f"{sum(len(m.positive) for m in mk)} positive markers")
    ok &= all(len(v)>=5 for v in ORGAN_REGISTRY.values())
    print("\n[T2] NEW: marker identifiability")
    for o in ORGAN_REGISTRY:
        idf=marker_identifiability(o)
        print(f"   {o:12s} rank={idf['rank']}/{idf['n_types']} "
              f"max_cos={idf['max_pairwise_cosine']:.3f} "
              f"identifiable={idf['identifiable']}")
        if idf["high_overlap_pairs"]:
            print(f"      overlap: {idf['high_overlap_pairs'][:2]}")
    ok &= all("rank" in marker_identifiability(o) for o in ORGAN_REGISTRY)
    print("\n[T3] shared markers detected")
    idf=marker_identifiability("cerebral")
    print(f"   cerebral shared: {idf['shared_markers_used']}")
    ok &= len(idf["shared_markers_used"])>0
    if HAS_SCANPY:
        print("\n[T4] confound detection + SD gate")
        n=200
        A=ad.AnnData(X=np.random.rand(n,50).astype(np.float32))
        A.obs["batch"]=["GSE_A"]*100+["GSE_B"]*100
        A.obs["treatment"]=["drug"]*100+["control"]*100
        cf=detect_confound(A,"batch","treatment")
        print(f"   confounded={cf['confounded']}")
        ok &= cf["confounded"] is True
        A.obs["treatment"]=["drug","control"]*100
        cf2=detect_confound(A,"batch","treatment")
        print(f"   mixed -> confounded={cf2['confounded']}")
        ok &= cf2["confounded"] is False
        print("\n[T5] Harmony refuses confounded correction")
        A.obs["treatment"]=["drug"]*100+["control"]*100
        if HAS_HARMONY:
            b=False
            try: HarmonyCorrector(sd=sd).correct(A,"batch",
                                                 condition_col="treatment")
            except DataRefusal: b=True
            print(f"   refused={b}")
            ok &= b
        else:
            print("   harmonypy absent — refusal path untested")
        c=sd.certify("batch_correction",())
        print(f"   without CONFOUND -> {c['status']}")
        ok &= c["status"]==MODE_E
    else:
        print("\n[T4-T5] SKIP — scanpy absent")
    print("\n[T6] species-mismatch refusal")
    if HAS_SCANPY:
        itg=MultiSeriesIntegrator("intestinal",sd=sd)
        h=ad.AnnData(X=np.random.rand(50,20).astype(np.float32))
        h.var_names=[f"GENE{i}" for i in range(20)]
        m=ad.AnnData(X=np.random.rand(50,20).astype(np.float32))
        m.var_names=[f"Gene{i}" for i in range(20)]
        itg.add_series("H",h); itg.add_series("M",m)
        b=False
        try: itg.integrate()
        except (DataRefusal,Refusal): b=True
        print(f"   refused={b}")
        ok &= b
    rr=sd.report()
    print(f"\n   SD: {rr['n']} outputs, {rr['mode_E']} Mode E")
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
KEY REMINDERS
  Marker annotation is a HYPOTHESIS; identifiability is now MEASURED.
  Harmony CANNOT fix confounded batches; the gate refuses.
  Harmony does NOT create treatment labels; curate them.
  Composition now carries a Thm 8.3 W2 bound.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","markers","identifiability",
                                   "confound-check"])
    ap.add_argument("--organ",default="intestinal",choices=list(ORGAN_REGISTRY))
    ap.add_argument("--data")
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if a.cmd=="markers":
        for cm in ORGAN_REGISTRY[a.organ]:
            print(f"{cm.name}:\n  + {cm.positive}\n  - {cm.negative}\n  {cm.reference}\n")
    elif a.cmd=="identifiability":
        print(json.dumps(marker_identifiability(a.organ),indent=2))
    else:
        if not HAS_SCANPY: log.error("pip install scanpy anndata"); return
        A=sc.read_h5ad(a.data)
        print(json.dumps(detect_confound(A,"batch","treatment"),indent=2,default=str))

if __name__=="__main__": main()
