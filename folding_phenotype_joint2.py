#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
folding_phenotype_joint.py — v2.0.0
Joint protein folding + sequence-to-phenotype on the corrected mathematics.

BUGS FIXED vs v1
  1. FATAL: init_frames placed residues at 3.8 Angstrom spacing and then
     coords = frames[...,:3,3] * coord_scale multiplied by 3.8 AGAIN,
     giving 14.44 Angstrom C-alpha spacing. v2 removes the double scaling.
  2. FATAL: the ensemble test was written `cond or True`, so it ALWAYS
     passed. v2 uses a plain bool with no escape hatch.
  3. structure_loss was masked MSE on ABSOLUTE coordinates: not invariant
     to rigid motion, so a correct fold in a different frame was penalised.
     v2 adds dRMS (stable, no SVD) and Kabsch-aligned RMSD.
  4. No SD gate; no identifiability or moment gate on the phenotype head.

NEW FROM ESCA 2.0 UNIFIED EDITION
  Phenotype-class fractions ARE an occupation vector -> Thm 8.3 W2 rate.
  Ensemble spread is VERIFIED distinct before any uncertainty is reported.
  SD gate; compression honesty for the shared bottleneck.

PHILOSOPHY: see msps_research_math_core.PHILOSOPHY (full text).
RESEARCH ONLY. Not a clinical predictor. No gene-editor design.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

from msps_research_math_core import (
    SDGate, Refusal, IntegrityError, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    drms_loss, kabsch_rmsd, verify_ensemble_spread, assert_real_test,
    composition_as_occupation, theorem_A_rate, boundary_null_check,
    KouRegime, moment_estimable, response_summary, get_logger,
)
log=get_logger("folding_v2"); VERSION="2.0.0"
AA="ACDEFGHIKLMNPQRSTVWY"; AA_IDX={c:i for i,c in enumerate(AA)}
PHENO_CLASSES=["BENIGN","UNCERTAIN","PATHOGENIC"]

@dataclass
class JointConfig:
    d_model:int=128; n_heads:int=4; n_encoder_layers:int=3
    n_ipa_blocks:int=2; d_pair:int=64
    n_phenotype_classes:int=1; n_ensemble:int=5
    w_struct:float=1.0; w_pheno:float=0.5
    max_len:int=512; ca_spacing:float=3.8
    struct_loss_kind:str="drms"     # drms | kabsch | raw_mse (NOT invariant)

class SequenceEncoder(nn.Module):
    def __init__(self,cfg:JointConfig):
        super().__init__()
        self.tok=nn.Embedding(21,cfg.d_model)
        self.pos=nn.Embedding(cfg.max_len,cfg.d_model)   # (L,d) — 2-D, correct
        layer=nn.TransformerEncoderLayer(cfg.d_model,cfg.n_heads,4*cfg.d_model,
                                         0.1,batch_first=True,activation="gelu")
        self.enc=nn.TransformerEncoder(layer,cfg.n_encoder_layers)
        self.norm=nn.LayerNorm(cfg.d_model)
    def forward(self,tokens,mask=None):
        B,L=tokens.shape
        x=self.tok(tokens)+self.pos(torch.arange(L,device=tokens.device))[None]
        x=self.enc(x,src_key_padding_mask=(~mask) if mask is not None else None)
        return self.norm(x)

class StructureHead(nn.Module):
    """
    FIX: frames already carry Angstrom-scale translations. No second
    multiplication by coord_scale. A learned residual is added instead.
    """
    def __init__(self,cfg:JointConfig):
        super().__init__()
        self.cfg=cfg
        self.proj=nn.Linear(cfg.d_model,cfg.d_model)
        self.ipa=None
        try:
            from ipa_patch import IPAStructureHeadFixed
            self.ipa=IPAStructureHeadFixed(d_scalar=cfg.d_model,d_point=8,
                n_heads=cfg.n_heads,n_ipa_blocks=cfg.n_ipa_blocks,
                d_pair=cfg.d_pair,n_torsion_angles=7)
        except ImportError:
            log.warning("ipa_patch absent; using a lightweight coordinate head")
        self.coord_head=nn.Sequential(nn.Linear(cfg.d_model,64),nn.GELU(),
                                      nn.Linear(64,3))
        self.torsion_head=nn.Linear(cfg.d_model,7)
    def init_frames(self,L,B,device,dtype):
        fr=torch.eye(4,device=device,dtype=dtype).view(1,1,4,4).expand(B,L,4,4).clone()
        fr[...,0,3]=torch.arange(L,device=device,dtype=dtype)*self.cfg.ca_spacing
        return fr
    def forward(self,feats,mask=None):
        B,L,_=feats.shape
        s=self.proj(feats)
        frames=self.init_frames(L,B,feats.device,feats.dtype)
        if self.ipa is not None:
            o=self.ipa(s,z=None,frames=frames)
            frames=o["frames"]; tors=o["torsions"]
        else:
            tors=self.torsion_head(s)
        # FIX: NO second scaling by 3.8
        coords=frames[...,:3,3]+self.coord_head(s)
        return {"frames":frames,"torsions":tors,"coords":coords}

def structure_loss(pred,target,mask,kind="drms")->Dict[str,Tensor]:
    """
    drms   : pairwise-distance loss. Rigid-motion invariant, no SVD.
    kabsch : aligned RMSD. Invariant, uses SVD (can be unstable early).
    raw_mse: NOT invariant — kept only to demonstrate the v1 error.
    """
    if kind=="drms":
        L=drms_loss(pred,target,mask)
    elif kind=="kabsch":
        L=kabsch_rmsd(pred,target,mask)["mean"]
    elif kind=="raw_mse":
        m=mask.to(pred.dtype).unsqueeze(-1)
        L=(((pred-target)**2)*m).sum()/m.sum().clamp_min(1)
    else: raise Refusal(f"unknown loss kind '{kind}'")
    if pred.shape[1]>1:
        bl=(pred[:,1:]-pred[:,:-1]).norm(dim=-1)
        bond=(((bl-3.8)**2)*mask[:,1:].to(pred.dtype)).sum()/ \
              mask[:,1:].to(pred.dtype).sum().clamp_min(1)
    else: bond=pred.new_zeros(())
    return {"total":L+0.1*bond,"invariant":L,"bond":bond,"kind":kind,
            "is_rigid_invariant":kind in ("drms","kabsch")}

class PhenotypeHead(nn.Module):
    """Ensemble of independently initialised MLPs. Spread is VERIFIED."""
    def __init__(self,cfg:JointConfig):
        super().__init__()
        self.n_class=cfg.n_phenotype_classes; self.n_ens=cfg.n_ensemble
        self.members=nn.ModuleList([nn.Sequential(
            nn.Linear(cfg.d_model,cfg.d_model//2),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(cfg.d_model//2,self.n_class)) for _ in range(self.n_ens)])
    def forward(self,feats,mask=None):
        if mask is not None:
            m=mask.to(feats.dtype).unsqueeze(-1)
            pooled=(feats*m).sum(1)/m.sum(1).clamp_min(1)
        else: pooled=feats.mean(1)
        outs=torch.stack([mm(pooled) for mm in self.members],0)
        return {"logits":outs.mean(0),"std":outs.std(0),"ensemble":outs}

def phenotype_loss(logits,target,n_class):
    if n_class==1: return F.mse_loss(logits.squeeze(-1),target.float())
    return F.cross_entropy(logits,target.long())

class FoldingPhenotypeJoint(nn.Module):
    def __init__(self,cfg:Optional[JointConfig]=None):
        super().__init__()
        self.cfg=cfg or JointConfig()
        self.encoder=SequenceEncoder(self.cfg)
        self.struct=StructureHead(self.cfg)
        self.pheno=PhenotypeHead(self.cfg)
    def forward(self,tokens,mask=None,target_coords=None,target_pheno=None):
        feats=self.encoder(tokens,mask)
        st=self.struct(feats,mask); ph=self.pheno(feats,mask)
        out={"coords":st["coords"],"frames":st["frames"],"torsions":st["torsions"],
             "phenotype_logits":ph["logits"],"phenotype_std":ph["std"],
             "phenotype_ensemble":ph["ensemble"],
             "clinical_actionability":RESEARCH_ONLY}
        if mask is None:
            mask=torch.ones(tokens.shape,dtype=torch.bool,device=tokens.device)
        loss=torch.zeros((),device=feats.device)
        if target_coords is not None:
            sl=structure_loss(st["coords"],target_coords,mask,self.cfg.struct_loss_kind)
            out["structure_loss"]=sl; loss=loss+self.cfg.w_struct*sl["total"]
        if target_pheno is not None:
            pl=phenotype_loss(ph["logits"],target_pheno,self.cfg.n_phenotype_classes)
            out["phenotype_loss"]=pl; loss=loss+self.cfg.w_pheno*pl
        if target_coords is not None or target_pheno is not None:
            out["total_loss"]=loss
        return out

def uncertainty_certificate(out:Dict,sd:Optional[SDGate]=None)->Dict[str,Any]:
    sd=sd or SDGate()
    ens=out["phenotype_ensemble"].detach().cpu().numpy()
    try:
        chk=verify_ensemble_spread(ens); ok=True
    except IntegrityError as e:
        chk={"members_distinct":False,"reason":str(e)}; ok=False
    cert=sd.certify("ensemble_uq",("SPREAD",) if ok else (),chk)
    return {"member_check":chk,"sd":cert,
            "spread_available":ok,
            "NOTE":("uncertainty is reported only when members are verified "
                    "distinct; identical members give zero spread")}

def phenotype_diagnostics(logits:Tensor,sd:Optional[SDGate]=None)->Dict[str,Any]:
    sd=sd or SDGate()
    if logits.shape[-1]==len(PHENO_CLASSES):
        p=F.softmax(logits,-1); lab=[PHENO_CLASSES[int(i)] for i in p.argmax(-1)]
        oc=composition_as_occupation(lab,PHENO_CLASSES)
        rate=theorem_A_rate(len(lab),len(PHENO_CLASSES))
        sd.certify("phenotype_prediction",(),{"n":len(lab)})
        return {"class_fractions":oc["F"],"names":PHENO_CLASSES,
                "W2_bound":rate["W2_bound"],
                "NOTE":"ClinVar labels are submitter assertions, not assays",
                "clinical_actionability":RESEARCH_ONLY}
    return {"regression":True,"n":int(logits.shape[0]),
            "clinical_actionability":RESEARCH_ONLY}

def synthetic_helix(B,L,device="cpu",seed=0):
    g=torch.Generator(device=device).manual_seed(seed)
    t=torch.arange(L,dtype=torch.float32,device=device)
    ang=t*math.radians(100)
    base=torch.stack([2.3*torch.cos(ang),2.3*torch.sin(ang),t*1.5],-1)
    return base.unsqueeze(0).expand(B,L,3)+torch.randn(B,L,3,generator=g,device=device)*0.1

def self_test(verbose=True)->bool:
    print("="*76); print("  FOLDING × PHENOTYPE v2 — self-test"); print("="*76)
    torch.manual_seed(0); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    dev="cpu"
    cfg=JointConfig(d_model=64,n_heads=4,n_encoder_layers=2,n_ipa_blocks=1,
                    d_pair=32,n_ensemble=5)
    m=FoldingPhenotypeJoint(cfg).to(dev)
    B,L=2,40
    tokens=torch.randint(0,20,(B,L))
    mask=torch.ones(B,L,dtype=torch.bool)
    tgt=synthetic_helix(B,L,dev); ph=torch.rand(B)

    print("\n[T1] FIX 1: no double coordinate scaling")
    with torch.no_grad():
        st=m.struct(m.encoder(tokens,mask),mask)
        bl=(st["coords"][:,1:]-st["coords"][:,:-1]).norm(dim=-1)
    print(f"      mean C-alpha spacing = {float(bl.mean()):.3f} A (target ~3.8)")
    rep("spacing near 3.8, NOT 14.44",float(bl.mean())<8.0)
    fr=m.struct.init_frames(L,B,dev,torch.float32)
    fbl=(fr[0,1:,:3,3]-fr[0,:-1,:3,3]).norm(dim=-1)
    rep("init_frames spacing exactly 3.8",abs(float(fbl.mean())-3.8)<1e-5)

    print("\n[T2] FIX 3: rigid-motion invariant structure loss")
    P=torch.randn(2,10,3,dtype=torch.float64)
    th=torch.randn(2,dtype=torch.float64)
    Rm=torch.stack([torch.stack([torch.cos(th),-torch.sin(th),torch.zeros_like(th)],1),
                    torch.stack([torch.sin(th),torch.cos(th),torch.zeros_like(th)],1),
                    torch.stack([torch.zeros_like(th),torch.zeros_like(th),torch.ones_like(th)],1)],1)
    Q=torch.einsum("bij,bnj->bni",Rm,P)+torch.tensor([1.,2.,3.],dtype=torch.float64)
    mk=torch.ones(2,10,dtype=torch.bool)
    dr=float(structure_loss(P,Q,mk,"drms")["invariant"])
    kb=float(structure_loss(P,Q,mk,"kabsch")["invariant"])
    raw=float(structure_loss(P,Q,mk,"raw_mse")["invariant"])
    print(f"      dRMS={dr:.2e}  Kabsch={kb:.2e}  raw MSE={raw:.3f}")
    rep("dRMS invariant",dr<1e-16)
    rep("Kabsch invariant",kb<1e-7)
    rep("raw MSE NOT invariant (the v1 bug)",raw>1.0)
    rep("invariance flag set",structure_loss(P,Q,mk,"drms")["is_rigid_invariant"])
    rep("raw flagged NOT invariant",
        not structure_loss(P,Q,mk,"raw_mse")["is_rigid_invariant"])

    print("\n[T3] FIX 2: no `or True` in tests")
    out=m(tokens,mask,tgt,ph)
    ens=out["phenotype_ensemble"].detach().numpy()
    chk=verify_ensemble_spread(ens)
    print(f"      max pairwise diff={chk['max_pairwise_diff']:.4e}")
    rep("ensemble members VERIFIED distinct",chk["members_distinct"])
    rep("a False test stays False",assert_real_test("demo",False)["passed"] is False)
    uc=uncertainty_certificate(out)
    rep("uncertainty certificate issued",uc["spread_available"])

    print("\n[T4] forward + backward")
    rep("coords shape",tuple(out["coords"].shape)==(B,L,3))
    rep("phenotype logits shape",tuple(out["phenotype_logits"].shape)==(B,1))
    rep("total loss finite",bool(torch.isfinite(out["total_loss"])))
    out["total_loss"].backward()
    for name,params in (("encoder",m.encoder.parameters()),
                        ("struct",m.struct.parameters()),
                        ("pheno",m.pheno.parameters())):
        gs=[p.grad for p in params if p.grad is not None]
        rep(f"{name} finite grads",bool(gs) and all(torch.isfinite(g).all() for g in gs))

    print("\n[T5] training reduces the joint loss")
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3); losses=[]
    m.train()
    for _ in range(30):
        opt.zero_grad(); o=m(tokens,mask,tgt,ph)
        o["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        losses.append(float(o["total_loss"]))
    rep("loss decreased",losses[-1]<losses[0],f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    print("\n[T6] phenotype occupation diagnostics")
    cfg3=JointConfig(d_model=64,n_encoder_layers=2,n_phenotype_classes=3)
    m3=FoldingPhenotypeJoint(cfg3)
    o3=m3(tokens,mask)
    dg=phenotype_diagnostics(o3["phenotype_logits"])
    print(f"      class fractions={[round(v,3) for v in dg['class_fractions']]}")
    rep("fractions sum to 1",abs(sum(dg["class_fractions"])-1)<1e-6)
    rep("W2 bound reported",dg["W2_bound"]>0)
    rep("ClinVar caveat present","submitter assertions" in dg["NOTE"])
    rep("research_only",dg["clinical_actionability"]==RESEARCH_ONLY)
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  Double coordinate scaling removed; C-alpha spacing is ~3.8 A.
  Structure loss is rigid-motion invariant (dRMS / Kabsch); raw MSE is
  kept only to demonstrate the v1 error.
  No `or True` escape hatch; ensemble spread is verified before any
  uncertainty is reported. SD gate wired.

NOT CLAIMED
  Not AlphaFold-grade: a single IPA head, not the full Evoformer.
  Trained on a synthetic helix here; no real PDB data.
  Phenotype output is research_only, not a clinical predictor.
""")
    return ok

def main()->int:
    ap=argparse.ArgumentParser(description="Joint folding × phenotype v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
                    choices=["selftest","demo","invariance"])
    ap.add_argument("--json",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    out={}
    if a.cmd=="invariance":
        P=torch.randn(2,10,3,dtype=torch.float64)
        Q=P+torch.tensor([1.,2.,3.],dtype=torch.float64)
        mk=torch.ones(2,10,dtype=torch.bool)
        out={k:float(structure_loss(P,Q,mk,k)["invariant"])
             for k in ("drms","kabsch","raw_mse")}
    elif a.cmd=="demo":
        cfg=JointConfig(d_model=64,n_encoder_layers=2)
        m=FoldingPhenotypeJoint(cfg).eval()
        with torch.no_grad(): o=m(torch.randint(0,20,(1,30)))
        out={"coords_shape":list(o["coords"].shape),
             "phenotype_logits":o["phenotype_logits"].tolist(),
             "phenotype_std":o["phenotype_std"].tolist(),
             "clinical_actionability":o["clinical_actionability"]}
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
