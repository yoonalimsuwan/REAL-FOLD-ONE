#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PERTURB-seq / CROP-seq / Perturb-CITE-seq + scVI/scANVI — v2.0
=============================================================================
BUGS FIXED
  1. isinstance(adata.obsm, dict) is ALWAYS False (obsm is AxisArrays), so
     guide names were NEVER retrieved. v2 uses membership on obsm/uns.
  2. aggregate_expression used adata.X without checking normalisation, so
     effect sizes could be dominated by library size. v2 requires a
     declared normalisation state.
  3. scANVI.from_scvi_model was called with labels_key, which recent
     scvi-tools versions do not accept. v2 handles both signatures.
  4. predict(soft=True).max(axis=1).values assumed a DataFrame. v2 uses
     np.asarray.
  5. Genetic KO vs factor dose was warned about but not ENFORCED.
     v2 refuses to emit dose fields from a KO dataset.
  6. Shared confound detection now comes from the core (one implementation).
  7. sklearn import inside mixing_metrics was unguarded.

PHILOSOPHY: see msps_bio_math_core.PHILOSOPHY.
HONEST SCOPE
  Perturb-seq ships guide metadata (less curation) but the perturbation is
  GENETIC, at a single dose (full KO). There is no dose-response curve.
  scVI/scANVI cannot fix batch == condition confounding.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from msps_bio_math_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, detect_confound,
    require_no_confound, composition_as_occupation, theorem_A_rate,
    get_logger,
)
log=get_logger("perturbseq_v2"); VERSION="2.0.0"
try:
    import scanpy as sc, anndata as ad
    HAS_SCANPY=True
except ImportError: HAS_SCANPY=False
try:
    import scvi
    from scvi.model import SCVI, SCANVI
    HAS_SCVI=True
except ImportError: HAS_SCVI=False
try:
    import harmonypy as hpy
    HAS_HARMONY=True
except ImportError: HAS_HARMONY=False
try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN=True
except ImportError: HAS_SKLEARN=False

NT=re.compile(r"non.?target|nt_|control|safe.?harbor|aaav|scrambl",re.I)

@dataclass
class Config:
    guide_cols:List[str]=field(default_factory=lambda:[
        "guide_id","guide","gRNA","grna","perturbation","perturb","crispr",
        "sgRNA","target_gene","gene_perturbation"])
    gene_cols:List[str]=field(default_factory=lambda:[
        "target_gene","gene","gene_target","perturbed_gene"])
    min_cells_per_guide:int=20
    allow_multi_guide:bool=False

class Loader:
    def __init__(self,cfg=None,sd:Optional[SDGate]=None):
        self.cfg=cfg or Config(); self.sd=sd or SDGate()
    def detect_guide_col(self,a):
        for c in self.cfg.guide_cols:
            if c in a.obs.columns: return c
        for c in a.obs.columns:
            if any(k in c.lower() for k in ("guide","grna","perturb")): return c
        return None
    def detect_guide_matrix(self,a):
        for k in a.obsm.keys():
            if any(t in k.lower() for t in ("guide","perturb","crispr")): return k
        return None
    def guide_names(self,a,key) -> Optional[List[str]]:
        """FIX: obsm is AxisArrays, NOT a dict. v1's isinstance check
        was always False so names were never found."""
        for container in (a.uns, a.obsm):
            for cand in (f"{key}_names", f"{key}_columns", key+"_var"):
                if cand in container:
                    v=container[cand]
                    return [str(x) for x in np.asarray(v).ravel()]
        return None
    def load(self,path:Path,guide_matrix:Optional[Path]=None):
        if not HAS_SCANPY: raise Refusal("pip install scanpy anndata")
        a=sc.read_h5ad(str(path))
        gc=self.detect_guide_col(a); gm=self.detect_guide_matrix(a)
        if gc: return self._from_col(a,gc)
        if gm: return self._from_matrix(a,gm)
        if guide_matrix: return self._from_external(a,guide_matrix)
        raise Refusal("no guide information found; run inspect() first")
    def _from_col(self,a,gc):
        vals=a.obs[gc].astype(str).values
        delim=next((d for d in (";",",","|","+")
                    if any(d in v for v in vals[:100])),None)
        gids=[]; ns=[]; drop=np.zeros(a.n_obs,bool)
        for i,v in enumerate(vals):
            if v in ("nan","","None","no_guide","unassigned"):
                drop[i]=True; gids.append("NONE"); ns.append(0); continue
            parts=[p.strip() for p in v.split(delim)] if delim else [v]
            parts=[p for p in parts if p]
            gids.append(parts[0] if parts else "NONE"); ns.append(len(parts))
        a.obs["guide_id"]=gids; a.obs["n_guides"]=ns
        a.obs["is_control"]=[bool(NT.search(str(g))) for g in gids]
        a.obs["perturbation_type"]=np.where(a.obs["is_control"],
                                            "non_targeting","genetic_perturbation")
        gene_col=next((c for c in self.cfg.gene_cols if c in a.obs.columns),None)
        a.obs["target_gene"]=(a.obs[gene_col].astype(str) if gene_col
                              else [self._strip(g) for g in gids])
        n0=a.n_obs; a=a[~drop].copy()
        log.info(f"dropped {n0-a.n_obs} cells with no guide")
        return a
    def _from_matrix(self,a,key):
        G=a.obsm[key]
        if hasattr(G,"toarray"): G=G.toarray()
        G=np.asarray(G)
        names=self.guide_names(a,key)
        if names is None:
            log.warning(f"no names for obsm['{key}']; using positional labels")
            names=[f"guide_{i}" for i in range(G.shape[1])]
        if len(names)!=G.shape[1]:
            raise Refusal(f"names {len(names)} != matrix columns {G.shape[1]}")
        n_per=(G>0).sum(1); top=G.argmax(1)
        a.obs["n_guides"]=n_per
        a.obs["guide_id"]=[names[i] for i in top]
        a.obs["is_control"]=[bool(NT.search(str(g))) for g in a.obs["guide_id"]]
        a.obs["perturbation_type"]=np.where(a.obs["is_control"],
                                            "non_targeting","genetic_perturbation")
        a.obs["target_gene"]=[self._strip(g) for g in a.obs["guide_id"]]
        return a[n_per>0].copy()
    def _from_external(self,a,path):
        import pandas as pd
        gm=pd.read_csv(path,index_col=0)
        common=a.obs_names.intersection(gm.index)
        if len(common)<a.n_obs*0.5:
            log.warning(f"only {len(common)}/{a.n_obs} cells overlap guide matrix")
        a=a[common].copy(); gm=gm.loc[common]
        G=gm.values; top=G.argmax(1)
        a.obs["guide_id"]=[gm.columns[i] for i in top]
        a.obs["n_guides"]=(G>0).sum(1)
        a.obs["is_control"]=[bool(NT.search(str(g))) for g in a.obs["guide_id"]]
        a.obs["perturbation_type"]=np.where(a.obs["is_control"],
                                            "non_targeting","genetic_perturbation")
        a.obs["target_gene"]=[self._strip(g) for g in a.obs["guide_id"]]
        return a
    @staticmethod
    def _strip(g):
        g=str(g)
        for p in ("guide_","gRNA_","sg","g"):
            if g.lower().startswith(p.lower()) and g[len(p):len(p)+1].isalpha():
                return g[len(p):]
        return g

class Aggregator:
    def __init__(self,min_cells=20,sd:Optional[SDGate]=None):
        self.min=min_cells; self.sd=sd or SDGate()
    def aggregate(self,a,marker_genes=None,normalised:bool=False):
        """FIX: normalisation state must be DECLARED, not assumed."""
        if not normalised:
            raise Refusal("aggregate requires log-normalised input. "
                          "Raw counts make the effect size library-size "
                          "dominated. Call sc.pp.normalize_total + log1p "
                          "and pass normalised=True.")
        ctrl=a.obs["is_control"].values.astype(bool)
        if ctrl.sum()<self.min:
            raise Refusal(f"only {ctrl.sum()} control cells; need >= {self.min}")
        X=a.X
        if hasattr(X,"toarray"): X=X.toarray()
        X=np.asarray(X,np.float32)
        cm=X[ctrl].mean(0); cs=X[ctrl].std(0)+1e-6
        genes=list(a.var_names); gi={g:i for i,g in enumerate(genes)}
        sel=[gi[g] for g in (marker_genes or []) if g in gi] or None
        guides=a.obs["guide_id"].values; tg=a.obs["target_gene"].values
        out={}
        for g in np.unique(guides):
            m=guides==g
            if m.sum()<self.min: continue
            if bool(a.obs["is_control"].values[m].all()): continue
            d=(X[m].mean(0)-cm)/cs
            order=np.argsort(d)
            out[g]={"guide_id":g,"target_gene":str(tg[m][0]),
                    "n_cells":int(m.sum()),
                    "perturbation_type":"genetic_perturbation",
                    "mean_effect":float(np.abs(d[sel]).mean()) if sel
                                  else float(np.abs(d).mean()),
                    "top_down":[genes[i] for i in order[:5]],
                    "top_up":[genes[i] for i in order[-5:][::-1]]}
        cert=self.sd.certify("perturb_response",("NORM",),{"n_guides":len(out)})
        return {"results":out,"certificate":cert}
    def to_t2_candidates(self,agg):
        """
        ENFORCED: a genetic KO has NO dose and NO time course.
        v2 refuses to emit placeholder numeric fields at all.
        """
        out=[]
        for g,r in agg.items():
            out.append({"record_id":f"perturbseq_{g}",
                        "factor":"GENETIC_"+str(r["target_gene"]).upper(),
                        "perturbation_type":"genetic_KO",
                        "target_gene":r["target_gene"],
                        "n_cells":r["n_cells"],
                        "mean_effect":r["mean_effect"],
                        "dose_nM":None,"time_h":None,
                        "verified":False,
                        "EXCLUDED_FROM_T2":True,
                        "REASON":("genetic perturbation at a single (full-KO) "
                                  "dose. There is no dose-response curve. "
                                  "Do NOT train a dose model on this.")})
        return out
    def require_dose_dataset(self,candidates):
        bad=[c["record_id"] for c in candidates if c.get("EXCLUDED_FROM_T2")]
        if bad:
            raise Refusal(f"{len(bad)} records are genetic KO with no dose. "
                          "They cannot enter a dose-response model. "
                          "Use them for the pathway PRIOR only.")
        return {"ok":True}

class ScVI:
    def __init__(self,n_latent=30,n_layers=2,max_epochs=400,use_gpu=True,
                 sd:Optional[SDGate]=None):
        if not HAS_SCVI: raise Refusal("pip install scvi-tools")
        self.n_latent=n_latent; self.n_layers=n_layers
        self.max_epochs=max_epochs; self.use_gpu=use_gpu
        self.sd=sd or SDGate(); self.model=None
    def fit(self,a,batch_key="batch",n_top_genes=3000):
        if "condition" in a.obs.columns:
            require_no_confound(a.obs,batch_key,"condition")
        X=a.X
        if hasattr(X,"toarray"): X=X.toarray()
        X=np.asarray(X)
        if not np.allclose(X,np.round(X),atol=1e-6):
            raise Refusal("scVI requires RAW COUNTS; input looks normalised")
        a=a.copy()
        sc.pp.highly_variable_genes(a,n_top_genes=n_top_genes,subset=False,
                                    flavor="seurat_v3",batch_key=batch_key)
        SCVI.setup_anndata(a,batch_key=batch_key)
        self.model=SCVI(a,n_latent=self.n_latent,n_layers=self.n_layers,
                        gene_likelihood="nb")
        self.model.train(max_epochs=self.max_epochs,early_stopping=True,
                         accelerator="gpu" if self.use_gpu else "cpu")
        a.obsm["X_scVI"]=self.model.get_latent_representation()
        return a

class ScANVI:
    def __init__(self,unlabeled="unknown",n_latent=30,max_epochs=200,
                 use_gpu=True):
        if not HAS_SCVI: raise Refusal("pip install scvi-tools")
        self.unlabeled=unlabeled; self.n_latent=n_latent
        self.max_epochs=max_epochs; self.use_gpu=use_gpu; self.model=None
    def fit(self,a,batch_key="batch",labels_key="cell_type",predict=True):
        if labels_key not in a.obs.columns:
            raise Refusal(f"'{labels_key}' absent; run marker annotation first")
        a=a.copy()
        has=a.obs[labels_key].notna()&(a.obs[labels_key]!="")
        a.obs[labels_key]=np.where(has,a.obs[labels_key],self.unlabeled)
        n_lab=int((a.obs[labels_key]!=self.unlabeled).sum())
        if n_lab<50:
            log.warning(f"only {n_lab} labelled cells; semi-supervision weak")
        SCVI.setup_anndata(a,batch_key=batch_key,labels_key=labels_key)
        base=SCVI(a,n_latent=self.n_latent)
        acc="gpu" if self.use_gpu else "cpu"
        base.train(max_epochs=self.max_epochs,accelerator=acc)
        # FIX: newer scvi-tools does not accept labels_key here
        try:
            self.model=SCANVI.from_scvi_model(base,unlabeled_category=self.unlabeled)
        except TypeError:
            self.model=SCANVI.from_scvi_model(base,adata=a,
                unlabeled_category=self.unlabeled,labels_key=labels_key)
        self.model.train(max_epochs=self.max_epochs,accelerator=acc)
        a.obsm["X_scANVI"]=self.model.get_latent_representation()
        if predict:
            a.obs["predicted_cell_type"]=self.model.predict()
            soft=self.model.predict(soft=True)
            a.obs["prediction_prob"]=np.asarray(soft).max(axis=1)
        return a

def mixing_metrics(a,emb_key,batch_key="batch",k=15):
    if not HAS_SKLEARN: raise Refusal("pip install scikit-learn")
    Z=a.obsm[emb_key]; b=a.obs[batch_key].values
    ub=list(np.unique(b)); bi=np.array([ub.index(x) for x in b])
    nn=NearestNeighbors(n_neighbors=k+1).fit(Z)
    _,idx=nn.kneighbors(Z); idx=idx[:,1:]
    purity=float((bi[idx]==bi[:,None]).mean(1).mean())
    exp=sum((np.mean(b==x))**2 for x in ub)
    return {"knn_batch_purity":purity,"random_expected":float(exp),
            "purity_over_random":purity/max(exp,1e-9),"n_batches":len(ub)}

def selftest(a_=None) -> bool:
    print("="*76); print("  PERTURB-seq v2 — self-test"); print("="*76)
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] GUIDE NAME RETRIEVAL FIXED")
    if HAS_SCANPY:
        n,g=60,20
        a=ad.AnnData(X=np.random.negative_binomial(5,.3,(n,g)).astype("float32"))
        a.var_names=[f"GENE{i}" for i in range(g)]
        G=np.zeros((n,4)); G[np.arange(n),np.arange(n)%4]=1
        a.obsm["guide_matrix"]=G
        a.uns["guide_matrix_names"]=["g_TP53","g_BRCA1","NT_1","g_CTNNB1"]
        ld=Loader()
        rep("obsm is NOT a dict",not isinstance(a.obsm,dict))
        names=ld.guide_names(a,"guide_matrix")
        rep("names retrieved from uns",names is not None and len(names)==4)
        a2=ld._from_matrix(a,"guide_matrix")
        rep("real guide names used","g_TP53" in set(a2.obs["guide_id"]))
        rep("control detected",int(a2.obs["is_control"].sum())>0)
    else:
        print("      SKIP — scanpy absent")

    print("\n[T2] NORMALISATION GUARD")
    if HAS_SCANPY:
        a=ad.AnnData(X=np.random.negative_binomial(5,.3,(80,20)).astype("float32"))
        a.var_names=[f"GENE{i}" for i in range(20)]
        a.obs["guide_id"]=["g_TP53"]*40+["NT_1"]*40
        a.obs["is_control"]=[False]*40+[True]*40
        a.obs["target_gene"]=["TP53"]*40+["NT"]*40
        ag=Aggregator(min_cells=10)
        b=False
        try: ag.aggregate(a,normalised=False)
        except Refusal: b=True
        rep("unnormalised input REFUSED",b)
        sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
        r=ag.aggregate(a,normalised=True)
        rep("normalised input accepted",len(r["results"])>0)
        rep("certificate present","certificate" in r)

        print("\n[T3] GENETIC KO EXCLUDED FROM DOSE MODEL")
        cands=ag.to_t2_candidates(r["results"])
        rep("dose is None",all(c["dose_nM"] is None for c in cands))
        rep("exclusion flag set",all(c["EXCLUDED_FROM_T2"] for c in cands))
        b=False
        try: ag.require_dose_dataset(cands)
        except Refusal: b=True
        rep("entering a dose model REFUSED",b)
    else:
        print("      SKIP")

    print("\n[T4] SHARED CONFOUND + SD")
    if HAS_SCANPY:
        a.obs["batch"]=["A"]*40+["B"]*40
        a.obs["condition"]=["treated"]*40+["control"]*40
        c=detect_confound(a.obs,"batch","condition")
        rep("confound detected",c["confounded"])
        b=False
        try: require_no_confound(a.obs,"batch","condition")
        except Exception: b=True
        rep("confounded integration REFUSED",b)
    sd=SDGate()
    rep("missing NORM -> Mode E",
        sd.certify("perturb_response",("VERIFIED",))["status"]==MODE_E)
    print("\n"+"="*76); print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  Guide names retrieved from uns/obsm (the isinstance(dict) bug).
  Normalisation state must be declared before aggregation.
  scANVI handles both from_scvi_model signatures; soft predict via asarray.
  Genetic KO records are EXCLUDED from dose models, not just warned.
  Shared confound detection; sklearn guarded; SD gate wired.
NOT CLAIMED
  Perturb-seq is genetic perturbation at a single dose. No dose curve.
  scVI/scANVI cannot fix batch == condition.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","load","aggregate"])
    ap.add_argument("--h5ad"); ap.add_argument("--guide-matrix")
    ap.add_argument("--out",default="data/perturbseq")
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if not HAS_SCANPY: log.error("pip install scanpy anndata"); return
    if a.cmd=="load":
        ld=Loader()
        a_=ld.load(Path(a.h5ad),Path(a.guide_matrix) if a.guide_matrix else None)
        Path(a.out).mkdir(parents=True,exist_ok=True)
        a_.write_h5ad(Path(a.out)/"annotated.h5ad")
    else:
        a_=sc.read_h5ad(a.h5ad)
        sc.pp.normalize_total(a_,target_sum=1e4); sc.pp.log1p(a_)
        r=Aggregator().aggregate(a_,normalised=True)
        Path(a.out).mkdir(parents=True,exist_ok=True)
        (Path(a.out)/"aggregate.json").write_text(json.dumps(r,indent=2,default=str))

if __name__=="__main__": main()
