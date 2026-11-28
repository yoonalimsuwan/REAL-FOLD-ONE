#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
T2 PRIOR-AS-WEIGHT-INIT + MULTI-OMICS — v2.0
=============================================================================
BUGS FIXED
  1. FATAL: protocol_ctx was torch.randn(...) on EVERY get_batch call, so
     train/val/test saw DIFFERENT contexts for the same sample and
     evaluation was non-deterministic. The ablation was invalid.
     v2 uses fixed_context() stored with the data.
  2. FATAL: RNAEncoder(None) / ProteinEncoder(None) hardcoded B=1, breaking
     concat for any batch > 1. v2 requires an explicit batch size.
  3. FactorPriorEmbedding silently truncated the prior when d_embed<N_FATE.
     Now warns and uses a learned projection initialised to preserve as
     much of the prior as the dimension allows.
  4. build_t2_batch used log1p(x/global_max), which is NOT library-size
     normalisation. v2 does per-cell normalisation and flags the state.
  5. No SD gate, no seed aggregation, no ablation integrity check.

PHILOSOPHY: see msps_bio_math_core.PHILOSOPHY.
HONEST SCOPE
  Prior init is an inductive bias, not a constraint. Multi-omics adds no
  information that is absent from the data. CITE-seq covers ~100-300
  proteins, not the proteome. Not clinical.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

from msps_bio_math_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    fixed_context, verify_ablation_arms, aggregate_seeds,
    composition_as_occupation, theorem_A_rate, get_logger,
)
log=get_logger("t2_prior_v2")
VERSION="2.0.0"

FACTORS=["WNT","CHIR","BMP4","LDN193189","ACTIVIN","TGFB1","SB431542",
         "FGF2","FGF4","FGF9","FGF10","NOTCH_AG","NOTCH_INH","DAPT",
         "SHH","RA","OTHER"]
N_FACTOR=len(FACTORS); F_IDX={f:i for i,f in enumerate(FACTORS)}
FATES=["PLURIPOTENT","ENDODERM","MESODERM","ECTODERM","NEURAL_PROGENITOR",
       "NEURON","GLIA","INTESTINAL_EPITHELIUM","GOBLET","PANETH","ENTEROCYTE",
       "HEPATOCYTE","CHOLANGIOCYTE","NEPHRON_PROGENITOR","PODOCYTE","TUBULE",
       "CARDIOMYOCYTE","ENDOTHELIAL","MESENCHYMAL","UNKNOWN"]
N_FATE=len(FATES)
PROTEIN_PANEL=["EPCAM","CDX2","CD44","LGR5","MUC2","LYZ","CHGA","CD31","VWF",
 "PDGFRA","COL1A1","VIM","CD3","CD4","CD8","CD19","CD14","CD56","KRT8",
 "KRT18","KRT20","CD133","CD49f","CD24","CD29","SSEA4","TRA-1-60","TRA-1-81",
 "CD34","CD45","CD73","CD90","CD105"]
N_PROTEIN=len(PROTEIN_PANEL)

# =============================================================================
# §1. PRIOR INIT
# =============================================================================

class PriorWeightInit:
    def __init__(self,prior:np.ndarray,scale=1.0,confidence=0.3):
        if prior.shape!=(N_FACTOR,N_FATE):
            raise Refusal(f"prior must be ({N_FACTOR},{N_FATE})")
        self.prior=prior.astype("float32")
        self.mask=(np.abs(self.prior)>=confidence).astype("float32")
        self.masked=self.prior*self.mask; self.scale=scale
    def layer(self) -> nn.Linear:
        L=nn.Linear(N_FACTOR,N_FATE,bias=False)
        with torch.no_grad():
            L.weight.copy_(torch.from_numpy(self.masked.T*self.scale))
        return L
    def summary(self):
        return {"n_active":int(self.mask.sum()),
                "coverage":float(self.mask.mean()),
                "factors_with_prior":int((self.mask.sum(1)>0).sum())}

class FactorPriorEmbedding(nn.Module):
    """v2: warns when d_embed < N_FATE instead of silently truncating."""
    def __init__(self,prior:np.ndarray,d_embed=32,scale=1.0):
        super().__init__()
        self.proj=nn.Linear(N_FATE,d_embed,bias=False)
        with torch.no_grad():
            W=torch.zeros(d_embed,N_FATE)
            for i in range(min(d_embed,N_FATE)): W[i,i]=1.0
            self.proj.weight.copy_(W)
        self.emb=nn.Embedding(N_FACTOR,d_embed)
        p=torch.from_numpy(prior.astype("float32"))
        with torch.no_grad():
            for f in range(N_FACTOR): self.emb.weight[f]=self.proj(p[f])*scale
        self.truncated = d_embed < N_FATE
        if self.truncated:
            log.warning(f"[prior-embed] d_embed={d_embed} < N_FATE={N_FATE}: "
                        f"{N_FATE-d_embed} fate directions are NOT represented "
                        "in the initial embedding. The layer can learn them, "
                        "but the prior does not carry them.")
    def forward(self,fidx): return self.emb(fidx)

# =============================================================================
# §2. ENCODERS — explicit batch size when missing
# =============================================================================

class ModalityEncoder(nn.Module):
    def __init__(self,d_in,d_out,d_hidden=None,kind="rna"):
        super().__init__()
        h=d_hidden or max(d_out*2,64)
        self.kind=kind
        self.net=nn.Sequential(nn.Linear(d_in,h),nn.LayerNorm(h),nn.GELU(),
                               nn.Dropout(0.1),nn.Linear(h,d_out),
                               nn.LayerNorm(d_out))
        self.missing=nn.Parameter(torch.randn(d_out)*0.01)
    def forward(self,x:Optional[Tensor],batch_size:int):
        """FIX: batch_size is REQUIRED. v1 hardcoded B=1 and crashed."""
        if batch_size<1: raise Refusal("batch_size must be >= 1")
        if x is None:
            return (self.missing.unsqueeze(0).expand(batch_size,-1),
                    torch.zeros(batch_size,device=self.missing.device))
        if x.shape[0]!=batch_size:
            raise Refusal(f"{self.kind} batch {x.shape[0]} != {batch_size}")
        return self.net(torch.nan_to_num(x)), torch.ones(x.shape[0],device=x.device)

class MultiOmicsFusion(nn.Module):
    def __init__(self,d_rna=128,d_prot=64,d_out=128):
        super().__init__()
        self.rp=nn.Linear(d_rna,d_out); self.pp=nn.Linear(d_prot,d_out)
        self.rg=nn.Sequential(nn.Linear(d_out,1),nn.Sigmoid())
        self.pg=nn.Sequential(nn.Linear(d_out,1),nn.Sigmoid())
        self.fuse=nn.Sequential(nn.Linear(2*d_out,d_out),nn.GELU(),
                                nn.LayerNorm(d_out))
    def forward(self,re,rp,pe,pp):
        a=self.rp(re); b=self.pp(pe)
        ga=self.rg(a).squeeze(-1)*rp; gb=self.pg(b).squeeze(-1)*pp
        return self.fuse(torch.cat([a*ga.unsqueeze(-1),
                                    b*gb.unsqueeze(-1)],-1))

# =============================================================================
# §3. MODEL
# =============================================================================

@dataclass
class T2Config:
    n_rna_genes:int=2000; n_proteins:int=N_PROTEIN
    d_rna:int=128; d_prot:int=64; d_fused:int=128
    d_hidden:int=256; d_ctx:int=128
    use_prior_init:bool=True; use_multiomics:bool=True
    prior_scale:float=1.0; prior_confidence:float=0.3; dropout:float=0.2

class T2Model(nn.Module):
    def __init__(self,cfg:T2Config,prior:Optional[np.ndarray]=None):
        super().__init__()
        self.cfg=cfg
        use_p = cfg.use_prior_init and prior is not None
        self.emb=(FactorPriorEmbedding(prior,32,cfg.prior_scale) if use_p
                  else nn.Embedding(N_FACTOR,32))
        self.dt=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,32))
        if cfg.use_multiomics:
            self.re=ModalityEncoder(cfg.n_rna_genes,cfg.d_rna,kind="rna")
            self.pe=ModalityEncoder(cfg.n_proteins,cfg.d_prot,kind="protein")
            self.fuse=MultiOmicsFusion(cfg.d_rna,cfg.d_prot,cfg.d_fused)
            od=cfg.d_fused
        else:
            self.re=self.pe=self.fuse=None; od=0
        self.cp=nn.Linear(cfg.d_ctx,64)
        self.trunk=nn.Sequential(nn.Linear(32+32+od+64,cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),nn.GELU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden,cfg.d_hidden),nn.LayerNorm(cfg.d_hidden),
            nn.GELU(),nn.Dropout(cfg.dropout))
        self.head=nn.Linear(cfg.d_hidden,N_FATE)
        self.prior_bias=(PriorWeightInit(prior,cfg.prior_scale,
                                         cfg.prior_confidence).layer()
                         if use_p else None)
    def forward(self,factor_idx,dose_nM,time_h,protocol_ctx,
                rna=None,protein=None):
        B=factor_idx.shape[0]
        parts=[self.emb(factor_idx),
               self.dt(torch.stack([torch.log10(dose_nM.clamp(min=1e-3)),
                                    torch.log2(time_h.clamp(min=1.))],-1)),
               self.cp(protocol_ctx)]
        info={}
        if self.cfg.use_multiomics:
            re_,rp=self.re(rna,B); pe_,pp=self.pe(protein,B)
            parts.append(self.fuse(re_,rp,pe_,pp))
            info={"n_with_rna":int(rp.sum()),"n_with_protein":int(pp.sum())}
        h=self.trunk(torch.cat(parts,-1)); lg=self.head(h)
        if self.prior_bias is not None:
            lg=lg+self.prior_bias(F.one_hot(factor_idx,N_FACTOR).float())
        comp=F.softmax(lg,-1)
        return {"composition":comp,"logits":lg,
                "entropy":-(comp*torch.log(comp+1e-9)).sum(-1),**info}

# =============================================================================
# §4. DATA — proper normalisation, fixed context
# =============================================================================

def build_batch(records:List[Dict],rna=None,protein=None,
                n_rna_genes=2000,d_ctx=128,seed=0,sd:Optional[SDGate]=None):
    B=len(records)
    comp=np.array([r["composition"] for r in records],np.float32)
    for i,c in enumerate(comp):
        if abs(c.sum()-1)>0.01:
            log.warning(f"record {i} sums to {c.sum():.4f}; renormalising")
            comp[i]=c/max(c.sum(),1e-8)
    out={"factor_idx":torch.tensor([F_IDX.get(r.get("factor","OTHER"),
                                                 N_FACTOR-1) for r in records]),
         "dose_nM":torch.tensor([r.get("dose_nM",1.) for r in records],dtype=torch.float32),
         "time_h":torch.tensor([r.get("time_h",24.) for r in records],dtype=torch.float32),
         "composition":torch.from_numpy(comp),
         "protocol_ctx":torch.from_numpy(fixed_context(B,d_ctx,seed))}
    norm_state="none"
    if rna is not None:
        X=np.asarray(rna,np.float32)
        if X.shape[1]>n_rna_genes:
            v=X.var(0); X=X[:,np.argsort(v)[-n_rna_genes:]]
        # FIX: per-cell library-size normalisation, then log1p
        lib=X.sum(1,keepdims=True); lib=np.maximum(lib,1.0)
        X=np.log1p(X/lib*1e4)
        out["rna"]=torch.from_numpy(X); out["n_rna_genes"]=X.shape[1]
        norm_state="per_cell_library_size_log1p"
    else:
        out["rna"]=None; out["n_rna_genes"]=n_rna_genes
    if protein is not None:
        P=np.asarray(protein,np.float32)
        gm=np.exp(np.log(P+1).mean(1,keepdims=True))
        out["protein"]=torch.from_numpy(np.log((P+1)/(gm+1e-8)).astype("float32"))
        norm_state += "+CLR"
    else: out["protein"]=None
    out["normalisation"]=norm_state
    if sd:
        pkgs=["VERIFIED"] if all(r.get("verified") for r in records) else []
        if norm_state!="none": pkgs.append("NORM")
        out["sd"]=sd.certify("perturb_response",tuple(pkgs),
                             {"normalisation":norm_state})
    rates=[theorem_A_rate(int(r.get("n_cells",0)) or 1,N_FATE)["W2_bound"]
           for r in records]
    out["W2_bound"]=torch.tensor(rates,dtype=torch.float32)
    return out

# =============================================================================
# §5. ABLATION
# =============================================================================

ARMS={"A_no_prior_no_omics":dict(use_prior_init=False,use_multiomics=False),
      "C_prior_init":dict(use_prior_init=True,use_multiomics=False),
      "D_omics_only":dict(use_prior_init=False,use_multiomics=True),
      "E_prior_init_omics":dict(use_prior_init=True,use_multiomics=True)}

def run_ablation(data,prior,epochs=60,seeds=(42,43,44),
                 out="runs/t2_prior_v2",device="cpu"):
    n=len(data["factor_idx"]); d_ctx=data["protocol_ctx"].shape[-1]
    ng=data.get("n_rna_genes",2000)
    probe={k:data[k][:4] for k in ("factor_idx","dose_nM","time_h","protocol_ctx")}
    probe["rna"]=data["rna"][:4] if data.get("rna") is not None else None
    probe["protein"]=data["protein"][:4] if data.get("protein") is not None else None
    chk=verify_ablation_arms(
        lambda f: T2Model(T2Config(n_rna_genes=ng,d_ctx=d_ctx,**f),prior),
        ARMS,probe)
    log.info(f"[ablation] {chk['n_arms']} arms verified distinct")
    rng=np.random.default_rng(42); perm=rng.permutation(n)
    ntr=int(.7*n); nva=int(.15*n)
    split={"train":perm[:ntr],"val":perm[ntr:ntr+nva],"test":perm[ntr+nva:]}
    if len(split["test"])<2: split["test"]=split["val"]
    dev=torch.device(device)
    for s in seeds:
        res={}
        for arm,fl in ARMS.items():
            torch.manual_seed(s)
            use_o=fl["use_multiomics"] and data.get("rna") is not None
            cfg=T2Config(n_rna_genes=ng,d_ctx=d_ctx,
                         use_prior_init=fl["use_prior_init"],use_multiomics=use_o)
            m=T2Model(cfg,prior).to(dev)
            opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
            def bt(idx):
                i=torch.as_tensor(idx,dtype=torch.long)
                return (data["factor_idx"][i].to(dev),data["dose_nM"][i].to(dev),
                        data["time_h"][i].to(dev),data["protocol_ctx"][i].to(dev),
                        data["rna"][i].to(dev) if use_o and data.get("rna") is not None else None,
                        data["protein"][i].to(dev) if use_o and data.get("protein") is not None else None)
            def kl(idx):
                i=torch.as_tensor(idx,dtype=torch.long)
                o=m(*bt(idx))
                return F.kl_div(o["composition"].log().clamp(min=-20),
                                data["composition"][i].to(dev),reduction="batchmean")
            best=float("inf"); state=None
            for ep in range(epochs):
                m.train()
                pp=np.random.default_rng(s+ep).permutation(len(split["train"]))
                ti=split["train"][pp]
                for st in range(0,len(ti),32):
                    L=kl(ti[st:st+32])
                    opt.zero_grad(); L.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
                if (ep+1)%max(1,epochs//5)==0:
                    m.eval()
                    with torch.no_grad(): v=float(kl(split["val"]))
                    if v<best:
                        best=v; state={k:x.detach().clone() for k,x in m.state_dict().items()}
            if state: m.load_state_dict(state)
            m.eval()
            with torch.no_grad(): tk=float(kl(split["test"]))
            res[arm]={"test_kl":tk,"best_val":best}
        d=Path(out)/f"seed{s}"; d.mkdir(parents=True,exist_ok=True)
        (d/"ablation.json").write_text(json.dumps(res,indent=2))
    agg=aggregate_seeds([str(Path(out)/f"seed{s}") for s in seeds])
    print("\n"+"="*76); print("  T2 ABLATION (fixed context, verified arms)")
    print("="*76)
    for arm,v in agg["per_arm"].items():
        print(f"  {arm:<24} {v['mean']:.5f} ± {v['sd']:.5f}  "
              f"{v.get('verdict','baseline')}")
    print("  SYNTHETIC data: code test, not a finding.")
    print("="*76)
    return agg

# =============================================================================
# §6. SELF-TEST
# =============================================================================

def selftest(a=None) -> bool:
    print("="*76); print("  T2 PRIOR + MULTIOMICS v2 — self-test"); print("="*76)
    torch.manual_seed(0); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] MISSING-MODALITY BATCH BUG FIXED")
    enc=ModalityEncoder(100,16,kind="rna")
    for B in (1,4,8):
        e_,p_=enc(None,B)
        rep(f"None with B={B} gives shape ({B},16)",tuple(e_.shape)==(B,16))
    b=False
    try: enc(None,0)
    except Refusal: b=True
    rep("batch_size=0 refused",b)
    b=False
    try: enc(torch.randn(4,100),8)
    except Refusal: b=True
    rep("batch mismatch refused",b)

    print("\n[T2] PRIOR INIT")
    prior=np.zeros((N_FACTOR,N_FATE),np.float32)
    prior[F_IDX["WNT"],FATES.index("INTESTINAL_EPITHELIUM")]=0.8
    prior[F_IDX["WNT"],FATES.index("GOBLET")]=-0.3
    pw=PriorWeightInit(prior,1.0,0.2)
    w=pw.layer().weight.detach().numpy()
    rep("weight = prior.T",bool(np.allclose(w,prior.T,atol=1e-6)))
    fe=FactorPriorEmbedding(prior,d_embed=8)
    rep("truncation flagged",fe.truncated)
    rep("full-dim not truncated",not FactorPriorEmbedding(prior,32).truncated)

    print("\n[T3] FIXED CONTEXT (the fatal bug)")
    recs=[{"factor":"WNT","dose_nM":100,"time_h":48,
           "composition":[1/N_FATE]*N_FATE,"verified":True,"n_cells":500}
          for _ in range(64)]
    b1=build_batch(recs,d_ctx=16,seed=0)
    b2=build_batch(recs,d_ctx=16,seed=0)
    rep("context reproducible",
        bool(np.allclose(b1["protocol_ctx"].numpy(),b2["protocol_ctx"].numpy())))
    rep("W2 bound present","W2_bound" in b1)

    print("\n[T4] NORMALISATION FIXED")
    rna=np.random.default_rng(0).negative_binomial(5,.3,(64,200)).astype("float32")
    b=build_batch(recs,rna=rna,n_rna_genes=100,d_ctx=16)
    rep("per-cell library-size norm","per_cell_library_size" in b["normalisation"])
    rep("HVG selection",b["n_rna_genes"]==100)

    print("\n[T5] MODEL FORWARD + ABLATION INTEGRITY")
    cfg=T2Config(n_rna_genes=100,d_ctx=16)
    m=T2Model(cfg,prior)
    o=m(b["factor_idx"][:4],b["dose_nM"][:4],b["time_h"][:4],
        b["protocol_ctx"][:4],b["rna"][:4],None)
    rep("composition on simplex",
        bool(torch.allclose(o["composition"].sum(-1),torch.ones(4),atol=1e-4)))
    o2=m(b["factor_idx"][:4],b["dose_nM"][:4],b["time_h"][:4],
         b["protocol_ctx"][:4],None,None)
    rep("both modalities missing works",tuple(o2["composition"].shape)==(4,N_FATE))
    probe={"factor_idx":b["factor_idx"][:4],"dose_nM":b["dose_nM"][:4],
           "time_h":b["time_h"][:4],"protocol_ctx":b["protocol_ctx"][:4],
           "rna":b["rna"][:4],"protein":None}
    chk=verify_ablation_arms(
        lambda f: T2Model(T2Config(n_rna_genes=100,d_ctx=16,**f),prior),ARMS,probe)
    rep("4 arms distinct",chk["all_arms_distinct"])

    print("\n[T6] GRADIENT REACHES PRIOR BIAS")
    L=F.kl_div(o["composition"].log().clamp(min=-20),
               b["composition"][:4],reduction="batchmean")
    L.backward()
    g=m.prior_bias.weight.grad
    rep("prior_bias has gradient",g is not None and bool(torch.isfinite(g).all()))
    rep("prior_bias gradient non-zero",float(g.abs().sum())>0)

    print("\n[T7] ABLATION RUNS")
    agg=run_ablation(b,prior,epochs=6,seeds=(42,43,44),out="runs/_t2p_v2_st")
    rep("4 arms x 3 seeds",len(agg["per_arm"])==4 and
        all(v["n_seeds"]==3 for v in agg["per_arm"].values()))
    import shutil; shutil.rmtree("runs/_t2p_v2_st",ignore_errors=True)
    print("\n"+"="*76); print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  protocol context is FIXED per sample (the ablation is now valid).
  missing-modality encoders take an explicit batch size.
  prior truncation is flagged. RNA uses per-cell library-size norm.
  ablation arms verified distinct; >=3 seeds; SD gate wired.
NOT CLAIMED
  Prior init is an inductive bias. Multi-omics creates no information.
  Synthetic metrics are a code test. Not clinical.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","ablation"])
    ap.add_argument("--prior"); ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--seeds",type=int,default=3); ap.add_argument("--out",default="runs/t2_prior_v2")
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    prior=np.load(a.prior) if a.prior else np.zeros((N_FACTOR,N_FATE),np.float32)
    recs=[{"factor":"WNT","dose_nM":100,"time_h":48,
           "composition":[1/N_FATE]*N_FATE,"verified":True,"n_cells":500}
          for _ in range(200)]
    rng=np.random.default_rng(0)
    b=build_batch(recs,rna=rng.negative_binomial(5,.3,(200,2000)).astype("float32"),
                  protein=rng.negative_binomial(5,.3,(200,N_PROTEIN)).astype("float32"))
    run_ablation(b,prior,epochs=a.epochs,seeds=tuple(range(42,42+a.seeds)),out=a.out)

if __name__=="__main__": main()
