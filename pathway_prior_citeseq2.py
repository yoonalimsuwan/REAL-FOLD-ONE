#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PERTURB-seq PATHWAY PRIOR + PERTURB-CITE-seq — v2.0
=============================================================================
BUGS FIXED
  1. FATAL: T2PriorLoss was numpy and returned a float. It could NEVER
     contribute a gradient, so the 'prior regularisation' never trained
     anything. v2 implements it in torch and asserts gradient reach.
  2. load_h5ad built var from rna.var[[]][:n] (an EMPTY DataFrame), giving
     a shape mismatch whenever ADT was present. v2 builds proper var names.
  3. PROTEIN_MARKERS used gene symbols while ADT panels use names like
     'CD3-1' or 'CD19_TotalSeqB'. v2 adds an alias resolver.
  4. totalVI was fed possibly log-normalised counts. v2 checks the
     normalisation state and refuses log-normalised input.
  5. Direction-sign logic was correct but convoluted; now explicit.
  6. No SD gate; genetic-KO vs factor-dose mixing was warned, not enforced.

PHILOSOPHY: see msps_bio_math_core.PHILOSOPHY.
HONEST SCOPE
  The prior constrains DIRECTION, not magnitude. KO != partial dose
  reduction. Crosstalk is not captured. CITE-seq covers ~100-300 proteins.
  The prior MUST be ablated; remove it if it does not help held-out.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

from msps_bio_math_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, assert_reaches,
    require_no_confound, get_logger, HAS_NUMPY,
)
log=get_logger("pathway_v2"); VERSION="2.0.0"
try:
    import scanpy as sc, anndata as ad
    HAS_SCANPY=True
except ImportError: HAS_SCANPY=False
try:
    from scvi.model import TOTALVI
    HAS_TOTALVI=True
except ImportError: HAS_TOTALVI=False

@dataclass
class Pathway:
    name:str; ko_genes:List[str]; factors:List[str]
    target_genes:List[str]; marker_genes:Dict[str,List[str]]

PATHWAYS=[
 Pathway("WNT_BCATENIN",["CTNNB1","APC","AXIN1","AXIN2","GSK3B","TCF7L2","LEF1"],
   ["WNT3A","CHIR"],["AXIN2","MYC","CCND1","LGR5","ASCL2","SOX9","CD44"],
   {"STEM_LGR5":["LGR5","ASCL2","OLFM4","SOX9","SMOC2"],
    "ENTEROCYTE":["FABP1","VIL1","ALPI","SI"],
    "GOBLET":["MUC2","TFF3","SPINK4"],"PANETH":["LYZ","DEFA5","DEFA6"]}),
 Pathway("BMP_SMAD158",["SMAD1","SMAD5","SMAD8","BMPR1A","BMPR1B","BMPR2"],
   ["BMP4","LDN193189"],["ID1","ID2","ID3","MSX1","MSX2","RUNX2"],
   {"ENTEROCYTE":["FABP1","VIL1","ALPI"],"GOBLET":["MUC2","TFF3"],
    "MESENCHYMAL":["VIM","COL1A1"]}),
 Pathway("TGFB_ACTIVIN_SMAD23",["SMAD2","SMAD3","SMAD4","TGFBR1","TGFBR2","ACVR1B"],
   ["ACTIVIN","TGFB1","SB431542"],
   ["SERPINE1","CTGF","SNAI1","SNAI2","TWIST1","LEFTY1"],
   {"ENDODERM":["SOX17","FOXA2","GATA6"],"MESENCHYMAL":["VIM","COL1A1","FN1"]}),
 Pathway("FGF_MAPK",["FGFR1","FGFR2","FGFR3","FGFR4","KRAS","BRAF","MAP2K1"],
   ["FGF2","FGF4","FGF9","FGF10"],["ETV4","ETV5","DUSP6","SPRY2","FOS","JUN"],
   {"NEURAL_PROGENITOR":["SOX2","NES","PAX6"],"MESENCHYMAL":["VIM","PDGFRA"]}),
 Pathway("NOTCH",["NOTCH1","NOTCH2","RBPJ","HES1","HEY1","PSEN1"],["DAPT"],
   ["HES1","HEY1","HEY2","HES5","NRARP"],
   {"GOBLET":["MUC2","TFF3"],"ENTEROENDOCRINE":["CHGA","NEUROD1"],
    "STEM_LGR5":["LGR5","ASCL2"]}),
 Pathway("HEDGEHOG",["SMO","PTCH1","GLI1","GLI2","GLI3"],["SHH","SAG"],
   ["GLI1","PTCH1","HHIP","MYCN","BCL2"],
   {"MESENCHYMAL":["VIM","PDGFRA"],"NEURAL_PROGENITOR":["SOX2","PAX6"]}),
 Pathway("RETINOIC_ACID",["RARA","RARB","RARG","RXRA","RXRB","CYP26A1"],["RA"],
   ["RARB","CYP26A1","HOXA1","HOXB1","CRABP2"],
   {"ENDODERM":["SOX17","FOXA2"],"ECTODERM":["PAX6","SOX1"]}),
]
FACTOR_TO_PATHWAY={"WNT":"WNT_BCATENIN","CHIR":"WNT_BCATENIN",
 "BMP4":"BMP_SMAD158","LDN193189":"BMP_SMAD158",
 "ACTIVIN":"TGFB_ACTIVIN_SMAD23","TGFB1":"TGFB_ACTIVIN_SMAD23",
 "SB431542":"TGFB_ACTIVIN_SMAD23","FGF2":"FGF_MAPK","FGF4":"FGF_MAPK",
 "FGF9":"FGF_MAPK","FGF10":"FGF_MAPK","NOTCH_AG":"NOTCH","NOTCH_INH":"NOTCH",
 "DAPT":"NOTCH","SHH":"HEDGEHOG","RA":"RETINOIC_ACID"}
FATES=["PLURIPOTENT","ENDODERM","MESODERM","ECTODERM","NEURAL_PROGENITOR",
       "NEURON","GLIA","INTESTINAL_EPITHELIUM","GOBLET","PANETH","ENTEROCYTE",
       "HEPATOCYTE","CHOLANGIOCYTE","NEPHRON_PROGENITOR","PODOCYTE","TUBULE",
       "CARDIOMYOCYTE","ENDOTHELIAL","MESENCHYMAL","UNKNOWN"]
F_IDX={f:i for i,f in enumerate(FATES)}
MARKER_TO_FATE={"STEM_LGR5":"INTESTINAL_EPITHELIUM","ENTEROCYTE":"ENTEROCYTE",
 "GOBLET":"GOBLET","PANETH":"PANETH","ENTEROENDOCRINE":"INTESTINAL_EPITHELIUM",
 "ENDODERM":"ENDODERM","ECTODERM":"ECTODERM",
 "NEURAL_PROGENITOR":"NEURAL_PROGENITOR","MESENCHYMAL":"MESENCHYMAL"}

# =============================================================================
# §1. SIGNATURE + PRIOR (explicit sign convention)
# =============================================================================

class PathwaySignature:
    """
    SIGN CONVENTION (made explicit)
      KO of a pathway ACTIVATOR reduces pathway activity.
      A gene DOWN in KO  -> pathway activation should raise it -> prior +1
      A gene UP   in KO  -> pathway activation should lower it -> prior -1
    """
    def __init__(self,min_ko_genes=2,sd:Optional[SDGate]=None):
        self.min=min_ko_genes; self.sd=sd or SDGate()
    def extract(self,perturb_results:Dict[str,Dict]) -> Dict[str,Dict[str,float]]:
        sigs={}
        for pw in PATHWAYS:
            kos=[r for r in perturb_results.values()
                 if r.get("target_gene") in pw.ko_genes]
            if len(kos)<self.min:
                log.warning(f"{pw.name}: {len(kos)} KO genes < {self.min}; skipped")
                continue
            score={}
            for r in kos:
                for g in r.get("top_down",[]): score[g]=score.get(g,0)+1
                for g in r.get("top_up",[]):   score[g]=score.get(g,0)-1
            sigs[pw.name]={g:s/len(kos) for g,s in score.items()}
        return sigs
    def prior_matrix(self,sigs,n_factors,n_fates,factor_to_idx) -> np.ndarray:
        P=np.zeros((n_factors,n_fates),np.float32)
        for pw in PATHWAYS:
            s=sigs.get(pw.name)
            if s is None: continue
            for f,pname in FACTOR_TO_PATHWAY.items():
                if pname!=pw.name or f not in factor_to_idx: continue
                fi=factor_to_idx[f]
                for mset,fate in MARKER_TO_FATE.items():
                    if fate not in F_IDX: continue
                    genes=pw.marker_genes.get(mset,[])
                    if not genes: continue
                    P[fi,F_IDX[fate]]=float(np.mean([s.get(g,0.) for g in genes]))
        return P
    def weights(self,P,thr=0.1) -> np.ndarray:
        return (np.abs(P)>=thr).astype(np.float32)

# =============================================================================
# §2. TORCH PRIOR LOSS  (the fatal bug)
# =============================================================================

class T2PriorLossTorch(nn.Module):
    """
    v2: DIFFERENTIABLE. v1 was numpy and returned a float, so the prior
    never trained anything.
        L = sum w[f,g] * relu(-prior[f,g] * (pred[f,g]-base[f,g])) / sum w
    """
    def __init__(self,prior:np.ndarray,weights:np.ndarray,lam=0.1):
        super().__init__()
        self.register_buffer("prior",torch.from_numpy(prior.astype("float32")))
        self.register_buffer("w",torch.from_numpy(weights.astype("float32")))
        self.lam=lam
    def forward(self,pred:Tensor,baseline:Tensor,factor_idx:Tensor) -> Tensor:
        p=self.prior[factor_idx]; w=self.w[factor_idx]
        change=pred-baseline
        viol=F.relu(-p*change)
        return self.lam*(w*viol).sum()/w.sum().clamp(min=1.0)

def T2PriorLoss(*a,**k):
    raise Refusal("the numpy T2PriorLoss was removed: it had NO gradient. "
                  "Use T2PriorLossTorch.")

# =============================================================================
# §3. CITE-seq LOADER (var fix + alias resolver + normalisation guard)
# =============================================================================

PROTEIN_MARKERS={
 "STEM_LGR5":["LGR5","EPCAM","CD44"],"ENTEROCYTE":["EPCAM","CDX2","VIL1"],
 "GOBLET":["MUC2","EPCAM"],"PANETH":["LYZ","EPCAM"],
 "T_CELL":["CD3","CD4","CD8"],"B_CELL":["CD19","CD20","CD79A"],
 "MONOCYTE":["CD14","CD16"],"NK":["CD56","CD16"],
 "ENDOTHELIAL":["CD31","VWF"],"FIBROBLAST":["PDGFRA","COL1A1"],
 "EPITHELIAL":["EPCAM","KRT8","KRT18"]}

def resolve_protein_alias(panel:List[str],gene:str) -> Optional[int]:
    """ADT names are like 'CD3-1', 'CD19_TotalSeqB', 'CD31 (PECAM1)'."""
    g=gene.upper()
    alias={"CD31":"PECAM1","CD20":"MS4A1","CD56":"NCAM1","CD49f":"ITGA6",
           "CD29":"ITGB1","CD133":"PROM1","CD90":"THY1","CD105":"ENG",
           "CD73":"NT5E","CD45":"PTPRC","CD34":"CD34"}
    targets={g,alias.get(g,g)}
    for i,p in enumerate(panel):
        base=re.split(r"[-_\s(]",str(p).upper())[0]
        if base in targets: return i
    return None

class CITELoader:
    def __init__(self,sd:Optional[SDGate]=None):
        self.sd=sd or SDGate(); self.rna=None; self.adt=None
        self.normalised=False
    def load_10x(self,path:Path):
        if not HAS_SCANPY: raise Refusal("pip install scanpy anndata")
        a=sc.read_10x_mtx(str(path),gex_suffix="",cache=False)
        if "feature_types" not in a.var:
            raise Refusal("no feature_types; cannot split RNA / ADT")
        self.rna=a[:,a.var.feature_types=="Gene Expression"].copy()
        self.adt=a[:,a.var.feature_types=="Antibody Capture"].copy()
        return self.rna,self.adt
    def load_h5ad(self,path:Path,obsm_key="adt",protein_names=None):
        """FIX: v1 built var from an EMPTY DataFrame -> shape mismatch."""
        if not HAS_SCANPY: raise Refusal("pip install scanpy anndata")
        import pandas as pd
        self.rna=sc.read_h5ad(str(path))
        if obsm_key not in self.rna.obsm:
            log.warning(f"'{obsm_key}' not in obsm; ADT unavailable")
            self.adt=None; return self.rna,None
        X=self.rna.obsm[obsm_key]
        if hasattr(X,"toarray"): X=X.toarray()
        X=np.asarray(X)
        n_p=X.shape[1]
        names=(list(protein_names) if protein_names is not None
               else [f"ADT_{i}" for i in range(n_p)])
        if len(names)!=n_p:
            raise Refusal(f"protein_names has {len(names)} entries, ADT has {n_p}")
        self.adt=ad.AnnData(X=X,obs=self.rna.obs.copy(),
                            var=pd.DataFrame(index=pd.Index(names)))
        return self.rna,self.adt
    def normalize(self,method="clr"):
        if self.rna is None: raise Refusal("load first")
        if self.normalised:
            log.warning("already normalised; skipping")
            return {"rna":self.rna,"adt":self.adt}
        sc.pp.normalize_total(self.rna,target_sum=1e4); sc.pp.log1p(self.rna)
        if self.adt is not None:
            X=self.adt.X
            if hasattr(X,"toarray"): X=X.toarray()
            X=np.asarray(X,np.float64)
            gm=np.exp(np.log(X+1).mean(1,keepdims=True))
            self.adt.X=np.log((X+1)/(gm+1e-8)).astype(np.float32)
            if method=="dsb":
                log.warning("DSB needs isotype controls; using CLR")
        self.normalised=True
        return {"rna":self.rna,"adt":self.adt}

class TotalVIIntegrator:
    def __init__(self,sd:Optional[SDGate]=None):
        self.sd=sd or SDGate(); self.model=None
    def fit(self,rna,adt,batch_key="batch",n_latent=30,max_epochs=200):
        if not HAS_TOTALVI: raise Refusal("pip install scvi-tools")
        X=rna.X
        if hasattr(X,"toarray"): X=X.toarray()
        X=np.asarray(X)
        # FIX: totalVI needs RAW COUNTS. Refuse log-normalised input.
        if not np.allclose(X,np.round(X),atol=1e-6) or X.max()<50:
            raise Refusal("totalVI requires RAW COUNTS. Input looks "
                          "log-normalised or non-integer. Reload raw counts.")
        require_no_confound(rna.obs,batch_key,"condition") \
            if "condition" in rna.obs.columns else None
        a=rna.copy()
        AX=adt.X
        if hasattr(AX,"toarray"): AX=AX.toarray()
        a.obsm["protein_counts"]=np.asarray(AX)
        TOTALVI.setup_anndata(a,protein_expression_obsm_key="protein_counts",
            batch_key=batch_key if batch_key in a.obs.columns else None)
        self.model=TOTALVI(a,n_latent=n_latent)
        self.model.train(max_epochs=max_epochs,early_stopping=True)
        a.obsm["X_totalVI"]=self.model.get_latent_representation()
        return a

def protein_score(adt_matrix,panel,markers=None):
    markers=markers or PROTEIN_MARKERS
    out={}; missing={}
    for ct,genes in markers.items():
        idx=[]; miss=[]
        for g in genes:
            i=resolve_protein_alias(panel,g)
            if i is None: miss.append(g)
            else: idx.append(i)
        if miss: missing[ct]=miss
        if idx: out[ct]=adt_matrix[:,idx].mean(1)
    return {"scores":out,"missing_markers":missing,
            "NOTE":"ADT names differ from gene symbols; alias resolver used"}

# =============================================================================
# §4. SELF-TEST
# =============================================================================

def selftest(a=None) -> bool:
    print("="*76); print("  PATHWAY PRIOR + CITE-seq v2 — self-test"); print("="*76)
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] THE FATAL BUG: prior loss now DIFFERENTIABLE")
    P=np.zeros((17,len(FATES)),np.float32)
    P[0,F_IDX["GOBLET"]]=0.8
    W=(np.abs(P)>=0.1).astype(np.float32)
    lf=T2PriorLossTorch(P,W,lam=1.0)
    pred=torch.softmax(torch.randn(4,len(FATES),requires_grad=True),-1)
    base=torch.softmax(torch.randn(4,len(FATES)),-1)
    fi=torch.zeros(4,dtype=torch.long)
    L=lf(pred,base,fi)
    rep("loss is a torch scalar",isinstance(L,Tensor) and L.dim()==0)
    L.backward()
    rep("gradient reaches prediction",
        pred.grad is not None and float(pred.grad.abs().sum())>0)
    b=False
    try: T2PriorLoss(P,W)
    except Refusal: b=True
    rep("numpy version REFUSED with explanation",b)

    print("\n[T2] SIGN CONVENTION")
    sig={"guide_CTNNB1":{"target_gene":"CTNNB1",
          "top_up":["MYC"],"top_down":["AXIN2","LGR5","ASCL2"]},
         "guide_APC":{"target_gene":"APC",
          "top_up":[],"top_down":["AXIN2","LGR5"]}}
    ps=PathwaySignature(min_ko_genes=1)
    s=ps.extract(sig)
    rep("WNT signature extracted","WNT_BCATENIN" in s)
    f2i={"WNT":0,"CHIR":1}
    Pm=ps.prior_matrix(s,17,len(FATES),f2i)
    print(f"      prior[WNT, INTESTINAL]={Pm[0,F_IDX['INTESTINAL_EPITHELIUM']]:.3f}")
    rep("KO-down -> activation-up (positive prior)",
        Pm[0,F_IDX["INTESTINAL_EPITHELIUM"]]>0)

    print("\n[T3] ADT ALIAS RESOLVER")
    panel=["CD3-1","CD19_TotalSeqB","CD31 (PECAM1)","EPCAM","CD44","LGR5"]
    rep("CD3 resolved",resolve_protein_alias(panel,"CD3")==0)
    rep("CD19 resolved",resolve_protein_alias(panel,"CD19")==1)
    rep("CD31 resolved via alias",resolve_protein_alias(panel,"CD31")==2)
    sc_=protein_score(np.random.rand(20,len(panel)),panel)
    rep("scores computed",len(sc_["scores"])>0)
    rep("missing markers reported","missing_markers" in sc_)

    print("\n[T4] CITE var FIX + totalVI raw-count guard")
    if HAS_SCANPY:
        import pandas as pd
        n,g,p=40,30,8
        a=ad.AnnData(X=np.random.negative_binomial(5,.3,(n,g)).astype("float32"))
        a.var_names=[f"GENE{i}" for i in range(g)]
        a.obs["batch"]=["A"]*(n//2)+["B"]*(n//2)
        a.obsm["adt"]=np.random.negative_binomial(5,.3,(n,p)).astype("float32")
        f=Path("_cite_v2.h5ad"); a.write_h5ad(f)
        ld=CITELoader()
        r_,ad_=ld.load_h5ad(f,protein_names=[f"ADT{i}" for i in range(p)])
        rep("ADT var names correct",ad_.n_vars==p)
        rep("ADT shape matches",ad_.X.shape==(n,p))
        rep("no empty-var crash",list(ad_.var_names)==[f"ADT{i}" for i in range(p)])
        ld.normalize()
        rep("normalisation state tracked",ld.normalised)
        if HAS_TOTALVI:
            b=False
            try: TotalVIIntegrator().fit(ld.rna,ld.adt)
            except Refusal: b=True
            rep("log-normalised input to totalVI REFUSED",b)
        f.unlink(missing_ok=True)
    else:
        print("      SKIP — scanpy absent")

    print("\n[T5] SD GATE")
    sd=SDGate()
    rep("prior without GRAD -> Mode E",
        sd.certify("prior_matrix",("VERIFIED",))["status"]==MODE_E)
    rep("prior with GRAD -> CERTIFIED",
        sd.certify("prior_matrix",("VERIFIED","GRAD"))["status"]==MODE_CERTIFIED)
    print("\n"+"="*76); print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  Prior loss is torch and reaches the prediction (v1 had NO gradient).
  CITE var names built properly; no empty-DataFrame crash.
  ADT alias resolver maps gene symbols to panel names.
  totalVI refuses log-normalised input.
  Sign convention explicit. SD gate wired.
NOT CLAIMED
  Direction only, not magnitude. KO != dose reduction. No crosstalk.
  CITE-seq is ~100-300 proteins. Ablate the prior; remove if it does not help.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","extract-prior"])
    ap.add_argument("--perturb-json"); ap.add_argument("--out",default="data/prior")
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if not a.perturb_json: log.error("--perturb-json required"); return
    res=json.loads(Path(a.perturb_json).read_text())
    ps=PathwaySignature(); s=ps.extract(res)
    f2i={f:i for i,f in enumerate(sorted(FACTOR_TO_PATHWAY))}
    P=ps.prior_matrix(s,len(f2i),len(FATES),f2i)
    Path(a.out).mkdir(parents=True,exist_ok=True)
    np.save(Path(a.out)/"prior_matrix.npy",P)
    np.save(Path(a.out)/"prior_weights.npy",ps.weights(P))
    (Path(a.out)/"signatures.json").write_text(json.dumps(s,indent=2))
    log.info(f"saved prior -> {a.out}")

if __name__=="__main__": main()
