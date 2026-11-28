#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
T1 -> T2 -> T3 CHAIN WITH LEARNABLE HILL COEFFICIENT — v2.0
=============================================================================
BUGS FIXED (two invalidated the ablation)
  1. FATAL: learnable_n=False only controlled self.hill, which the chain
     NEVER CALLED. The forward pass always used n_raw_head, so arms
     A (n=1 fixed) and B (learnable n) were IDENTICAL. v2 routes the flag
     into the forward pass and verify_ablation_arms() asserts the arms
     produce different outputs before any result is reported.
  2. quality loss used raw MSE on diameter in micrometres (~1e2-1e3) vs
     composition KL (~1), so quality dominated the objective by 100x.
     v2 uses log-diameter heteroscedastic NLL, comparable in scale.
  3. LearnableHillOccupancy.n_hill returned a CPU tensor when disabled
     -> device mismatch on CUDA. Now device-aware.
  4. The dead self.hill parameter sat in the optimizer with no gradient.
     Removed.
  5. evaluate() scored only composition KL, so model selection ignored
     the quality objective. Now scores the full loss.
  6. EC50/n correlation was only WARNED about. Now QUANTIFIED via Fisher
     information (msps_bio_math_core.hill_identifiability).

PHILOSOPHY: see msps_bio_math_core.PHILOSOPHY.
  The Tao is the Source. The Te is the Path. The Qi is the Proof.

HONEST SCOPE
  The optimised EC50 and n are TASK-SPECIFIC, not physical constants.
  T3 quality labels are scarce. T4 is NOT here — see msps_t4_preclinical.
  Not a clinical device. Not a transplant decision aid.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

from msps_bio_math_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    hill_identifiability, verify_ablation_arms, aggregate_seeds,
    fixed_context, get_logger,
)
log=get_logger("hill_chain_v2")
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
QUALITY_MARKERS=["LGR5","OLFM4","MUC2","LYZ","CHGA","KRT20","CDX2","EPCAM"]
N_MARKERS=len(QUALITY_MARKERS)

# =============================================================================
# §1. HILL — the flag now REACHES the forward pass
# =============================================================================

class HillOccupancy(nn.Module):
    """
    o = Emax D^n/(D^n+E^n).
    v2: n_source is explicit. Either 'fixed' (n=1), 'global' (one learned
    scalar), or 'per_sample' (predicted from sequence). The ablation flag
    selects among these IN THE FORWARD PASS.
    """
    def __init__(self,n_min=0.3,n_max=5.0,mode="per_sample"):
        super().__init__()
        if mode not in ("fixed","global","per_sample"):
            raise Refusal("mode must be fixed|global|per_sample")
        self.mode=mode; self.n_min=n_min; self.n_max=n_max
        if mode=="global":
            init=(1.0-n_min)/(n_max-n_min)
            self.n_raw=nn.Parameter(torch.tensor(math.log(init/(1-init))))
        else:
            self.register_buffer("n_raw",torch.tensor(0.0))
    def n_value(self, n_raw_external:Optional[Tensor],
                device, dtype) -> Tensor:
        if self.mode=="fixed":
            return torch.ones((),device=device,dtype=dtype)
        if self.mode=="global":
            return self.n_min+(self.n_max-self.n_min)*torch.sigmoid(self.n_raw)
        if n_raw_external is None:
            raise Refusal("per_sample mode requires n_raw from T1")
        return self.n_min+(self.n_max-self.n_min)*torch.sigmoid(n_raw_external)
    def forward(self,dose,ec50,emax,n_raw_external=None,eps=1e-9):
        n=self.n_value(n_raw_external,dose.device,dose.dtype)
        D=dose.clamp(min=eps); E=ec50.clamp(min=eps)
        u=D.pow(n); v=E.pow(n)
        return emax*u/(u+v), n

def hill_grad_ec50(dose,ec50,emax,n,eps=1e-9):
    D=dose.clamp(min=eps); E=ec50.clamp(min=eps)
    u=D.pow(n); v=E.pow(n)
    return -emax*n*u*E.pow(n-1)/(u+v).pow(2).clamp(min=eps**2)

def hill_grad_n(dose,ec50,emax,n,eps=1e-9):
    D=dose.clamp(min=eps); E=ec50.clamp(min=eps)
    u=D.pow(n); v=E.pow(n)
    return emax*u*v*(torch.log(D)-torch.log(E))/(u+v).pow(2).clamp(min=eps**2)

# =============================================================================
# §2. HEADS
# =============================================================================

@dataclass
class ChainConfig:
    d_seq:int=128; d_hidden:int=256; d_ctx:int=128
    dropout:float=0.2
    ec50_range:tuple=(1e-3,1e6)
    n_min:float=0.3; n_max:float=5.0
    hill_mode:str="per_sample"      # fixed | global | per_sample
    use_t3:bool=True
    w_composition:float=1.0; w_quality:float=0.5
    w_ec50_reg:float=0.01; w_n_reg:float=0.001
    n_prior:float=1.0

class T1Potency(nn.Module):
    def __init__(self,cfg:ChainConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=nn.Sequential(
            nn.Linear(cfg.d_seq,cfg.d_hidden),nn.LayerNorm(cfg.d_hidden),
            nn.GELU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden,cfg.d_hidden),nn.LayerNorm(cfg.d_hidden),
            nn.GELU())
        self.log_ec50=nn.Linear(cfg.d_hidden,1)
        self.emax=nn.Sequential(nn.Linear(cfg.d_hidden,1),nn.Sigmoid())
        # v2: only create the per-sample head when it is actually used
        self.n_head=(nn.Linear(cfg.d_hidden,1)
                     if cfg.hill_mode=="per_sample" else None)
    def forward(self,seq):
        h=self.enc(seq)
        return {"log_ec50":self.log_ec50(h).squeeze(-1),
                "emax":self.emax(h).squeeze(-1),
                "n_raw":(self.n_head(h).squeeze(-1)
                         if self.n_head is not None else None)}
    def ec50_reg(self,log_ec50):
        lo,hi=math.log(self.cfg.ec50_range[0]),math.log(self.cfg.ec50_range[1])
        return (F.relu(lo-log_ec50).pow(2)+F.relu(log_ec50-hi).pow(2)).mean()

class T2Composition(nn.Module):
    def __init__(self,cfg:ChainConfig):
        super().__init__()
        self.emb=nn.Embedding(N_FACTOR,32)
        d=32+3+cfg.d_ctx
        self.trunk=nn.Sequential(nn.Linear(d,cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),nn.GELU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden,cfg.d_hidden),nn.LayerNorm(cfg.d_hidden),
            nn.GELU())
        self.head=nn.Linear(cfg.d_hidden,N_FATE)
    def forward(self,fidx,occ,dose,time,ctx):
        extra=torch.stack([occ,torch.log10(dose.clamp(min=1e-3)),
                           torch.log2(time.clamp(min=1.))],-1)
        h=self.trunk(torch.cat([self.emb(fidx),extra,ctx],-1))
        lg=self.head(h)
        return {"composition":F.softmax(lg,-1),"logits":lg}

class T3Quality(nn.Module):
    def __init__(self,cfg:ChainConfig):
        super().__init__()
        d=N_FATE+cfg.d_ctx+2
        self.trunk=nn.Sequential(nn.Linear(d,cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),nn.GELU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden,cfg.d_hidden),nn.LayerNorm(cfg.d_hidden),
            nn.GELU())
        self.log_diam=nn.Linear(cfg.d_hidden,1)
        self.viab=nn.Sequential(nn.Linear(cfg.d_hidden,1),nn.Sigmoid())
        self.mark=nn.Sequential(nn.Linear(cfg.d_hidden,128),nn.GELU(),
                                nn.Linear(128,N_MARKERS),nn.Sigmoid())
        self.log_sd=nn.Linear(cfg.d_hidden,2)
    def forward(self,comp,ctx,dose,time):
        extra=torch.stack([torch.log10(dose.clamp(min=1e-3)),
                           torch.log2(time.clamp(min=1.))],-1)
        h=self.trunk(torch.cat([comp,ctx,extra],-1))
        sd=self.log_sd(h).clamp(-6,3)
        return {"log_diameter":self.log_diam(h).squeeze(-1),
                "diameter_um":self.log_diam(h).squeeze(-1).exp(),
                "viability":self.viab(h).squeeze(-1),
                "markers":self.mark(h),
                "log_sd_diam":sd[:,0],"log_sd_viab":sd[:,1]}

class Chain(nn.Module):
    """T1 -> Hill -> T2 -> T3. The hill_mode flag reaches the forward pass."""
    def __init__(self,cfg:ChainConfig):
        super().__init__()
        self.cfg=cfg
        self.t1=T1Potency(cfg)
        self.hill=HillOccupancy(cfg.n_min,cfg.n_max,cfg.hill_mode)
        self.t2=T2Composition(cfg)
        self.t3=T3Quality(cfg) if cfg.use_t3 else None
    def forward(self,seq_feat,factor_idx,dose_nM,time_h,protocol_ctx):
        t1=self.t1(seq_feat)
        ec50=torch.exp(t1["log_ec50"]).clamp(*self.cfg.ec50_range)
        occ,n=self.hill(dose_nM,ec50,t1["emax"],t1["n_raw"])
        t2=self.t2(factor_idx,occ,dose_nM,time_h,protocol_ctx)
        out={**{k:v for k,v in t1.items() if v is not None},
             **t2,"ec50_nM":ec50,"occupancy":occ,"n_hill":n}
        if self.t3 is not None:
            out.update(self.t3(t2["composition"],protocol_ctx,dose_nM,time_h))
        return out

# =============================================================================
# §3. LOSS — quality now on a comparable scale
# =============================================================================

class ChainLoss(nn.Module):
    def __init__(self,cfg:ChainConfig):
        super().__init__(); self.cfg=cfg
    def forward(self,out,targets):
        dev=out["ec50_nM"].device; z=torch.zeros((),device=dev); L={}
        if "composition" in targets:
            L["composition"]=F.kl_div(
                out["composition"].log().clamp(min=-20),
                targets["composition"],reduction="batchmean")
        else: L["composition"]=z
        q=z
        if self.cfg.use_t3:
            # FIX: log-diameter NLL, not raw micrometre MSE
            if "diameter_um" in targets:
                sd=out["log_sd_diam"].exp().clamp(1e-3,1e3)
                y=torch.log(targets["diameter_um"].clamp(min=1.))
                q=q+(0.5*((out["log_diameter"]-y)/sd).pow(2)
                     +out["log_sd_diam"]).mean()
            if "viability" in targets:
                q=q+F.mse_loss(out["viability"],targets["viability"])
            if "markers" in targets:
                q=q+F.mse_loss(out["markers"],targets["markers"])
        L["quality"]=q
        L["ec50_reg"]=self.t1.ec50_reg(out["log_ec50"])
        n=out["n_hill"]
        L["n_reg"]=(n-self.cfg.n_prior).pow(2).mean() if n.numel()>1 else \
                   (n-self.cfg.n_prior).pow(2)
        L["total"]=(self.cfg.w_composition*L["composition"]
                    +self.cfg.w_quality*L["quality"]
                    +self.cfg.w_ec50_reg*L["ec50_reg"]
                    +self.cfg.w_n_reg*L["n_reg"])
        return L

# =============================================================================
# §4. TRAIN + ABLATION (integrity-verified)
# =============================================================================

ABLATION_ARMS={
 "A_n1_noT3":  dict(hill_mode="fixed",      use_t3=False),
 "B_nL_noT3":  dict(hill_mode="per_sample", use_t3=False),
 "C_n1_T3":    dict(hill_mode="fixed",      use_t3=True),
 "D_nL_T3":    dict(hill_mode="per_sample", use_t3=True),
 "E_nGlobal_T3":dict(hill_mode="global",    use_t3=True),
}

def make_data(n=200,d_seq=128,d_ctx=128,seed=0):
    """SYNTHETIC SHAPE CHECK. Metrics are meaningless."""
    log.warning("SYNTHETIC — code test only, NOT a scientific result")
    g=torch.Generator().manual_seed(seed)
    return {"seq_feat":torch.randn(n,d_seq,generator=g),
            "factor_idx":torch.randint(0,N_FACTOR,(n,),generator=g),
            "dose_nM":torch.rand(n,generator=g)*1000+1,
            "time_h":torch.rand(n,generator=g)*200+1,
            "protocol_ctx":torch.from_numpy(fixed_context(n,d_ctx,seed)),
            "targets":{"composition":F.softmax(torch.randn(n,N_FATE,generator=g),-1),
                       "diameter_um":torch.rand(n,generator=g)*500+50,
                       "viability":torch.rand(n,generator=g),
                       "markers":torch.rand(n,N_MARKERS,generator=g)}}

def train(data,cfg,split,epochs=50,lr=1e-3,bs=32,seed=42,device="cpu"):
    torch.manual_seed(seed); dev=torch.device(device)
    model=Chain(cfg).to(dev); lf=ChainLoss(cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sel=lambda t,i: t[i].to(dev)
    def batch(idx):
        return (sel(data["seq_feat"],idx),sel(data["factor_idx"],idx),
                sel(data["dose_nM"],idx),sel(data["time_h"],idx),
                sel(data["protocol_ctx"],idx))
    def tg(idx):
        return {k:sel(v,idx) for k,v in data["targets"].items()
                if cfg.use_t3 or k=="composition"}
    def full_loss(idx):
        o=model(*batch(idx)); return float(lf(o,tg(idx))["total"])
    best=float("inf"); state=None
    for ep in range(epochs):
        model.train()
        perm=np.random.default_rng(seed+ep).permutation(len(split["train"]))
        ti=[split["train"][i] for i in perm]
        for s in range(0,len(ti),bs):
            bi=torch.tensor(ti[s:s+bs],dtype=torch.long)
            o=model(*batch(bi)); L=lf(o,tg(bi))
            opt.zero_grad(); L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (ep+1)%max(1,epochs//5)==0:
            model.eval()
            with torch.no_grad():
                vi=torch.tensor(split["val"],dtype=torch.long)
                v=full_loss(vi)     # FIX: full objective, not just KL
            if v<best:
                best=v; state={k:x.detach().clone() for k,x in model.state_dict().items()}
    if state: model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        te=torch.tensor(split["test"],dtype=torch.long)
        test_total=full_loss(te)
        o=model(*batch(te))
        test_kl=float(F.kl_div(o["composition"].log().clamp(min=-20),
                      data["targets"]["composition"][te].to(dev),
                      reduction="batchmean"))
    return {"test_total":test_total,"test_kl":test_kl,"best_val":best,
            "mean_n_hill":float(o["n_hill"].mean()),
            "hill_mode":cfg.hill_mode}

def run_ablation(data,epochs=50,seeds=(42,43,44),out="runs/hill_v2",
                 device="cpu",verify=True):
    n=len(data["seq_feat"])
    base=ChainConfig(d_seq=data["seq_feat"].shape[-1],
                     d_ctx=data["protocol_ctx"].shape[-1])
    if verify:
        probe={"seq_feat":data["seq_feat"][:4],
               "factor_idx":data["factor_idx"][:4],
               "dose_nM":data["dose_nM"][:4],
               "time_h":data["time_h"][:4],
               "protocol_ctx":data["protocol_ctx"][:4]}
        arms={k:{**v} for k,v in ABLATION_ARMS.items()}
        chk=verify_ablation_arms(
            lambda f: Chain(ChainConfig(d_seq=base.d_seq,d_ctx=base.d_ctx,
                                        hill_mode=f["hill_mode"],
                                        use_t3=f["use_t3"])),
            arms,probe)
        log.info(f"[ablation] integrity verified: {chk['n_arms']} distinct arms")
    rng=np.random.default_rng(42); perm=rng.permutation(n)
    ntr=int(.7*n); nva=int(.15*n)
    split={"train":perm[:ntr].tolist(),"val":perm[ntr:ntr+nva].tolist(),
           "test":perm[ntr+nva:].tolist()}
    if len(split["test"])<2: split["test"]=split["val"]
    for s in seeds:
        res={}
        for arm,fl in ABLATION_ARMS.items():
            cfg=ChainConfig(d_seq=base.d_seq,d_ctx=base.d_ctx,
                            hill_mode=fl["hill_mode"],use_t3=fl["use_t3"],
                            w_quality=base.w_quality if fl["use_t3"] else 0.)
            r=train(data,cfg,split,epochs=epochs,seed=s,device=device)
            res[arm]=r
            log.info(f"[{arm}] seed={s} total={r['test_total']:.4f} "
                     f"n={r['mean_n_hill']:.3f}")
        d=Path(out)/f"seed{s}"; d.mkdir(parents=True,exist_ok=True)
        (d/"ablation.json").write_text(json.dumps(res,indent=2))
    agg=aggregate_seeds([str(Path(out)/f"seed{s}") for s in seeds],
                        metric="test_total")
    print("\n"+"="*76)
    print("  ABLATION (integrity-verified, >=3 seeds)")
    print("="*76)
    for arm,v in agg["per_arm"].items():
        print(f"  {arm:<16} {v['mean']:.4f} ± {v['sd']:.4f}  "
              f"{v.get('verdict','baseline')}")
    print("\n  A verdict needs |effect| > 1 pooled sd. Otherwise INCONCLUSIVE.")
    print("  SYNTHETIC data: these numbers are a CODE TEST, not a finding.")
    print("="*76)
    return agg

# =============================================================================
# §5. SELF-TEST
# =============================================================================

def selftest(a=None) -> bool:
    print("="*76); print("  HILL CHAIN v2 — self-test"); print("="*76)
    torch.manual_seed(0); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] THE FATAL BUG IS FIXED: flag reaches forward pass")
    d=torch.tensor([1.,10.,100.]); e=torch.tensor([10.]*3); m=torch.ones(3)
    hf=HillOccupancy(mode="fixed"); hp=HillOccupancy(mode="per_sample")
    of,nf=hf(d,e,m)
    op,np_=hp(d,e,m,torch.zeros(3))
    print(f"      fixed n={float(nf):.3f}  per_sample n={float(np_.mean()):.3f}")
    rep("fixed gives n=1",abs(float(nf)-1.)<1e-6)
    rep("per_sample gives n!=1",abs(float(np_.mean())-1.)>1e-3)
    rep("outputs DIFFER",float((of-op).abs().max())>1e-6)
    cfg_f=ChainConfig(d_seq=32,d_hidden=32,d_ctx=16,hill_mode="fixed")
    cfg_p=ChainConfig(d_seq=32,d_hidden=32,d_ctx=16,hill_mode="per_sample")
    rep("fixed arm has NO n_head",Chain(cfg_f).t1.n_head is None)
    rep("per_sample arm HAS n_head",Chain(cfg_p).t1.n_head is not None)

    print("\n[T2] ABLATION INTEGRITY VERIFIED")
    data=make_data(64,32,16)
    probe={k:data[k][:4] for k in ("seq_feat","factor_idx","dose_nM",
                                   "time_h","protocol_ctx")}
    chk=verify_ablation_arms(
        lambda f: Chain(ChainConfig(d_seq=32,d_hidden=32,d_ctx=16,
                                    hill_mode=f["hill_mode"],use_t3=f["use_t3"])),
        ABLATION_ARMS,probe)
    rep("all 5 arms distinct",chk["all_arms_distinct"])
    print(f"      {chk['n_arms']} arms verified distinct")

    print("\n[T3] HILL IDENTIFIABILITY QUANTIFIED")
    hi=hill_identifiability([1.,10.,100.],10.,1.,1.5)
    print(f"      3 doses: corr(EC50,n)={hi['corr_EC50_n']:.4f} "
          f"identifiable={hi['jointly_identifiable']}")
    rep("3 doses NOT jointly identifiable",not hi["jointly_identifiable"])
    hi2=hill_identifiability([0.1,0.3,1,3,10,30,100,300,1000],10.,1.,1.)
    print(f"      9 doses: corr={hi2['corr_EC50_n']:.4f}")
    rep("wide range improves",abs(hi2["corr_EC50_n"])<abs(hi["corr_EC50_n"]))

    print("\n[T4] QUALITY LOSS ON COMPARABLE SCALE")
    cfg=ChainConfig(d_seq=32,d_hidden=32,d_ctx=16)
    mo=Chain(cfg); o=mo(data["seq_feat"][:4],data["factor_idx"][:4],
                        data["dose_nM"][:4],data["time_h"][:4],
                        data["protocol_ctx"][:4])
    tg={k:v[:4] for k,v in data["targets"].items()}
    L=ChainLoss(cfg)(o,tg)
    print(f"      composition={float(L['composition']):.3f} "
          f"quality={float(L['quality']):.3f}")
    rep("quality not 100x composition",
        float(L["quality"])<20*max(float(L["composition"]),0.1))
    rep("log-diameter used","log_diameter" in o)

    print("\n[T5] GRADIENT REACHES ALL HEADS")
    L["total"].backward()
    for name,params in (("t1",mo.t1.parameters()),("t2",mo.t2.parameters()),
                        ("t3",mo.t3.parameters())):
        gs=[p.grad for p in params if p.grad is not None]
        rep(f"{name} has finite grads",
            bool(gs) and all(torch.isfinite(g).all() for g in gs))
    gs=[p.grad for p in mo.t1.n_head.parameters() if p.grad is not None]
    rep("n_head receives gradient",bool(gs) and
        any(float(g.abs().sum())>0 for g in gs))

    print("\n[T6] FIXED CONTEXT")
    c1=data["protocol_ctx"].numpy()
    c2=fixed_context(64,16,0)
    rep("context reproducible",bool(np.allclose(c1,c2)))

    print("\n[T7] ABLATION RUNS")
    agg=run_ablation(data,epochs=6,seeds=(42,43,44),out="runs/_hill_v2_st")
    rep("5 arms reported",len(agg["per_arm"])==5)
    rep("3 seeds each",all(v["n_seeds"]==3 for v in agg["per_arm"].values()))
    import shutil; shutil.rmtree("runs/_hill_v2_st",ignore_errors=True)

    print("\n[T8] T4 IS NOT HERE")
    rep("no engraftment head in this module",not hasattr(mo,"t4"))
    print("      T4 -> msps_t4_preclinical (preclinical readiness only)")
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  hill_mode reaches the forward pass; arms are verified distinct.
  quality loss uses log-diameter NLL, comparable to composition KL.
  dead self.hill parameter removed; device-aware fixed n.
  model selection uses the FULL objective.
  EC50/n correlation quantified, not just warned.
  protocol context fixed per sample.

NOT CLAIMED
  Optimised EC50/n are task-specific, not physical constants.
  Synthetic metrics are a code test. T4 is not in this module.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","ablation","identifiability"])
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--out",default="runs/hill_v2")
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if a.cmd=="identifiability":
        print(json.dumps(hill_identifiability([1.,10.,100.],10.,1.,1.5),indent=2))
        return
    run_ablation(make_data(),epochs=a.epochs,
                 seeds=tuple(range(42,42+a.seeds)),out=a.out)

if __name__=="__main__": main()
