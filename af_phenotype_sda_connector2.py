#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
af_phenotype_sda_connector.py — v2.0.0
AF-grade structure + ClinVar phenotype + SDA-ONE scale, corrected.

BUGS FIXED vs v1
  1. FATAL: pos_embed had shape (1,1,max_len,d). Added to m of shape
     (B,L,d) it broadcast to (1,B,L,d) — a 4-D tensor — and
     nn.MultiheadAttention then failed. v2 uses (1,max_len,d).
  2. FATAL: SDAOneScaleAdapter.compress_domain constructed a NEW
     nn.Linear on EVERY call with fresh random weights: untrained,
     unregistered, non-deterministic noise. v2 registers a persistent
     projection in __init__ and verify_persistent_projection() asserts it.
  3. FATAL: predict_large fed a (d_mn+4)=324-dim vector into a head
     expecting d_model+4=260. v2 projects to the head's input width.
  4. Two self-tests were written `cond or True` and always passed.
  5. JointAFPhenotypeLoss had a dead `sum(... for _ in [0])` line.

NEW FROM ESCA 2.0 UNIFIED EDITION
  SD gate; phenotype-class occupation fractions (Thm 8.3); verified
  ensemble spread; ClinVar circularity gate; moment gate on scores.

PHILOSOPHY: see msps_research_math_core.PHILOSOPHY (full text).
RESEARCH ONLY. Weights are RANDOM until trained. AlphaMissense is SOTA,
not this. ClinVar labels are submitter assertions. No gene editing.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
import urllib.parse, urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

from msps_research_math_core import (
    SDGate, Refusal, IntegrityError, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    verify_persistent_projection, verify_ensemble_spread, assert_real_test,
    verify_no_information_collapse, composition_as_occupation, theorem_A_rate,
    boundary_null_check, KouRegime, moment_estimable, response_summary,
    drms_loss, kabsch_rmsd, get_logger,
)
log=get_logger("af_sda_v2"); VERSION="2.0.0"
AA="ACDEFGHIKLMNPQRSTVWY"; VOCAB=24
PHENO_CLASSES=["BENIGN","PATHOGENIC"]

# =============================================================================
# §1. AF-GRADE ENCODER — pos_embed shape FIXED
# =============================================================================
class AFStructureEncoder(nn.Module):
    def __init__(self,d_model=256,c_z=128,n_blocks=4,n_heads=8,
                 n_structure_blocks=2,max_len=512):
        super().__init__()
        self.d_model=d_model; self.max_len=max_len
        self.aa_embed=nn.Embedding(VOCAB,d_model)
        # FIX 1: (1, max_len, d) — NOT (1,1,max_len,d)
        self.pos_embed=nn.Parameter(torch.randn(1,max_len,d_model)*0.02)
        self.z_init=nn.Linear(2*d_model+1,c_z)
        self.evo=nn.ModuleList([nn.ModuleDict({
            "msa_norm":nn.LayerNorm(d_model),
            "msa_attn":nn.MultiheadAttention(d_model,n_heads,batch_first=True,dropout=0.1),
            "pair_norm":nn.LayerNorm(c_z),"pair_proj":nn.Linear(c_z,c_z),
            "ffn":nn.Sequential(nn.Linear(d_model,2*d_model),nn.GELU(),
                                nn.Linear(2*d_model,d_model)),
            "ffn_norm":nn.LayerNorm(d_model)}) for _ in range(n_blocks)])
        self.struct=nn.ModuleList([nn.ModuleDict({
            "norm":nn.LayerNorm(d_model),
            "q":nn.Linear(d_model,d_model),"k":nn.Linear(d_model,d_model),
            "v":nn.Linear(d_model,d_model),
            "coord":nn.Linear(d_model,3)}) for _ in range(n_structure_blocks)])
        self.plddt=nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model,d_model//2),nn.GELU(),
                                 nn.Linear(d_model//2,50))
        self.distogram=nn.Sequential(nn.LayerNorm(c_z),nn.Linear(c_z,64))
        self.final=nn.LayerNorm(d_model)
    def forward(self,tokens,residue_mask=None)->Dict[str,Tensor]:
        B,L=tokens.shape
        if L>self.max_len: raise Refusal(f"L={L} > max_len={self.max_len}")
        if residue_mask is None:
            residue_mask=torch.ones(B,L,dtype=torch.bool,device=tokens.device)
        m=self.aa_embed(tokens)+self.pos_embed[:,:L]     # (B,L,d) — 3-D ✓
        assert m.dim()==3, f"single repr must be 3-D, got {m.dim()}"
        rel=torch.arange(L,device=tokens.device)
        d=(rel[:,None]-rel[None,:]).float().clamp(-32,32)/32.
        z=self.z_init(torch.cat([m[:,:,None].expand(B,L,L,-1),
                                 m[:,None,:].expand(B,L,L,-1),
                                 d[None,:,:,None].expand(B,L,L,1)],-1))
        kp=~residue_mask
        for blk in self.evo:
            mn=blk["msa_norm"](m)
            ao,_=blk["msa_attn"](mn,mn,mn,key_padding_mask=kp)
            m=m+ao
            z=z+blk["pair_proj"](blk["pair_norm"](z))
            m=m+blk["ffn"](blk["ffn_norm"](m))
        coords=torch.zeros(B,L,3,device=tokens.device,dtype=m.dtype)
        for sb in self.struct:
            sn=sb["norm"](m)
            q,k,v=sb["q"](sn),sb["k"](sn),sb["v"](sn)
            att=torch.einsum("bld,bmd->blm",q,k)/math.sqrt(q.shape[-1])
            att=att.masked_fill(~residue_mask[:,None,:],float("-inf"))
            att=F.softmax(att,-1)
            m=m+torch.einsum("blm,bmd->bld",att,v)
            coords=coords+sb["coord"](m)
        pl=self.plddt(m); dl=self.distogram(0.5*(z+z.transpose(1,2)))
        return {"single_repr":self.final(m),"pair_repr":z,"positions":coords,
                "plddt_logits":pl,"distogram_logits":dl,"residue_mask":residue_mask}

class AFPhenotypeJoint(nn.Module):
    def __init__(self,d_model=256,c_z=128,n_evo_blocks=4,n_heads=8,
                 n_structure_blocks=2,n_phenotype_classes=2,max_len=512,
                 n_ensemble=4):
        super().__init__()
        self.encoder=AFStructureEncoder(d_model,c_z,n_evo_blocks,n_heads,
                                        n_structure_blocks,max_len)
        self.d_model=d_model; self.n_class=n_phenotype_classes
        self.in_width=d_model+3+1
        self.members=nn.ModuleList([nn.Sequential(
            nn.Linear(self.in_width,d_model),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(d_model,d_model//2),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(d_model//2,n_phenotype_classes)) for _ in range(n_ensemble)])
    def features(self,tokens,mask=None):
        e=self.encoder(tokens,mask)
        bins=torch.linspace(2.5,97.5,50,device=tokens.device)
        pl=(F.softmax(e["plddt_logits"],-1)*bins).sum(-1)
        return e,pl
    def forward(self,tokens,residue_mask=None,true_coords=None)->Dict[str,Any]:
        e,pl=self.features(tokens,residue_mask)
        single=e["single_repr"]; coords=e["positions"]; mk=e["residue_mask"]
        pin=torch.cat([single,coords,pl.unsqueeze(-1)/100.],-1)
        mf=mk.to(single.dtype).unsqueeze(-1)
        pooled=(pin*mf).sum(1)/mf.sum(1).clamp_min(1)
        outs=torch.stack([mm(pooled) for mm in self.members],0)
        out={"phenotype_logits":outs.mean(0),"phenotype_std":outs.std(0),
             "phenotype_ensemble":outs,"positions":coords,"plddt":pl,
             "distogram_logits":e["distogram_logits"],"single_repr":single,
             "residue_mask":mk,"clinical_actionability":RESEARCH_ONLY}
        if true_coords is not None:
            out["structure_loss"]=drms_loss(coords,true_coords,mk)
        return out

class JointLoss(nn.Module):
    """FIX 5: dead `sum(... for _ in [0])` removed."""
    def __init__(self,w_pheno=1.0,w_struct=0.5,w_plddt=0.1):
        super().__init__(); self.w=(w_pheno,w_struct,w_plddt)
    def forward(self,pred,batch)->Dict[str,Tensor]:
        L={}; dev=pred["phenotype_logits"].device
        z=torch.zeros((),device=dev)
        L["phenotype"]=F.cross_entropy(pred["phenotype_logits"],batch["labels"]) \
                       if "labels" in batch else z
        L["structure"]=pred.get("structure_loss",z)
        L["plddt"]=F.cross_entropy(pred["plddt_logits"].reshape(-1,50),
                                   batch["plddt_targets"].reshape(-1).long()) \
                   if "plddt_targets" in batch else z
        L["total"]=self.w[0]*L["phenotype"]+self.w[1]*L["structure"]+self.w[2]*L["plddt"]
        return L

# =============================================================================
# §2. SDA-ONE ADAPTER — persistent projection, correct widths
# =============================================================================
class SDAOneScaleAdapter(nn.Module):
    """
    FIX 2: the compression projection is REGISTERED in __init__ and trained.
    FIX 3: the aggregated feature is projected to the phenotype head's
    input width, not fed at the wrong dimension.
    """
    def __init__(self,d_model=256,d_mn=320,domain_size=1000,head_in_width=None):
        super().__init__()
        self.domain_size=int(domain_size); self.d_mn=int(d_mn)
        self.d_model=int(d_model)
        self.head_in_width=int(head_in_width or (d_model+4))
        self.proj=nn.Linear(d_model,d_mn,bias=False)     # PERSISTENT
        self.to_head=nn.Linear(d_mn,self.head_in_width-4)
        verify_persistent_projection(self,"SDA compression")
    def segment(self,L:int)->List[Tuple[int,int]]:
        return [(s,min(s+self.domain_size,L)) for s in range(0,L,self.domain_size)]
    def compress(self,feat:Tensor)->Tensor:
        pooled=feat.mean(1)
        verify_no_information_collapse(pooled.detach().cpu().numpy(),"domain pooled")
        return self.proj(pooled)
    def predict_large(self,model:AFPhenotypeJoint,tokens,mask)->Dict[str,Any]:
        B,L=tokens.shape; doms=self.segment(L)
        if len(doms)==1: return model(tokens,mask)
        feats=[]
        for s,e in doms:
            o=model.encoder(tokens[:,s:e],mask[:,s:e] if mask is not None else None)
            feats.append(o["single_repr"])
        comp=[self.compress(f) for f in feats]
        glob=torch.stack(comp,1).mean(1)                 # (B,d_mn)
        head_in=torch.cat([self.to_head(glob),
                           torch.zeros(B,4,device=glob.device,dtype=glob.dtype)],-1)
        assert head_in.shape[-1]==self.head_in_width
        outs=torch.stack([mm(head_in) for mm in model.members],0)
        return {"phenotype_logits":outs.mean(0),"phenotype_std":outs.std(0),
                "phenotype_ensemble":outs,"n_domains":len(doms),
                "clinical_actionability":RESEARCH_ONLY,
                "NOTE":"domain aggregation is a research surrogate, not AF3"}

# =============================================================================
# §3. CLINVAR
# =============================================================================
class ClinVarClient:
    EUTILS="https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PATHO={"pathogenic","likely pathogenic","pathogenic/likely pathogenic"}
    BENIGN={"benign","likely benign","benign/likely benign"}
    def __init__(self,cache_dir="cache/clinvar",offline=True,api_key=None):
        self.cache=Path(cache_dir); self.cache.mkdir(parents=True,exist_ok=True)
        self.offline=offline; self.api_key=api_key or os.environ.get("NCBI_API_KEY")
        self._last=0.
    def _fetch(self,url):
        key=self.cache/(hashlib.sha256(url.encode()).hexdigest()[:32]+".json")
        if key.exists(): return key.read_text()
        if self.offline: return None
        dt=time.time()-self._last
        if dt<0.35: time.sleep(0.35-dt)
        self._last=time.time()
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"AF-SDA/2.0 (research)"})
            with urllib.request.urlopen(req,timeout=30) as r:
                body=r.read().decode("utf-8","replace")
            key.write_text(body); return body
        except Exception as e:
            log.warning(f"ClinVar fetch failed: {e}"); return None
    def fetch_gene(self,gene,retmax=200)->Dict[str,Any]:
        p={"db":"clinvar","term":f"{gene}[gene]","retmax":str(retmax),"retmode":"json"}
        if self.api_key: p["api_key"]=self.api_key
        body=self._fetch(f"{self.EUTILS}/esearch.fcgi?{urllib.parse.urlencode(p)}")
        if body is None:
            return {"status":"UNAVAILABLE","records":[],
                    "reason":"offline or unreachable; no fabricated labels"}
        try: ids=json.loads(body)["esearchresult"]["idlist"]
        except Exception: return {"status":"ERROR","records":[]}
        if not ids: return {"status":"OK","records":[],"n_ids":0}
        p2={"db":"clinvar","id":",".join(ids),"retmode":"json"}
        if self.api_key: p2["api_key"]=self.api_key
        b2=self._fetch(f"{self.EUTILS}/esummary.fcgi?{urllib.parse.urlencode(p2)}")
        if b2 is None: return {"status":"UNAVAILABLE","records":[]}
        recs=[]
        try:
            res=json.loads(b2).get("result",{})
            for uid in res.get("uids",[]):
                r=res.get(uid,{}); sig=r.get("clinical_significance",{})
                desc=(sig.get("description","") if isinstance(sig,dict) else str(sig)).strip().lower()
                if desc in self.PATHO: lab=1
                elif desc in self.BENIGN: lab=0
                else: continue
                vs=r.get("variation_set",[])
                recs.append({"clinvar_id":uid,"gene":gene,
                             "variation":vs[0].get("variation_name","") if vs else "",
                             "significance":desc,"label":lab,
                             "review_status":r.get("review_status","")})
        except Exception as e: return {"status":"ERROR","reason":str(e),"records":[]}
        return {"status":"OK","records":recs,"n_total":len(recs),
                "label_semantics":"submitter assertions, NOT functional assays",
                "CIRCULARITY":("training and testing on ClinVar risks circularity; "
                               "use held-out GENE splits and independent functional data")}

# =============================================================================
# §4. DIAGNOSTICS
# =============================================================================
def connector_diagnostics(out:Dict,sd:Optional[SDGate]=None)->Dict[str,Any]:
    sd=sd or SDGate()
    ens=out["phenotype_ensemble"].detach().cpu().numpy()
    try: chk=verify_ensemble_spread(ens); ok=True
    except IntegrityError as e: chk={"members_distinct":False,"reason":str(e)}; ok=False
    sd.certify("ensemble_uq",("SPREAD",) if ok else (),chk)
    lg=out["phenotype_logits"]
    lab=[PHENO_CLASSES[int(i)] for i in lg.argmax(-1)]
    oc=composition_as_occupation(lab,PHENO_CLASSES)
    rate=theorem_A_rate(len(lab),len(PHENO_CLASSES))
    sd.certify("phenotype_prediction",(),{"n":len(lab)})
    return {"member_check":chk,"class_fractions":oc["F"],"names":PHENO_CLASSES,
            "W2_bound":rate["W2_bound"],"sd_report":sd.report(),
            "clinical_actionability":RESEARCH_ONLY,
            "NOT_SOTA":"AlphaMissense (DeepMind 2023) is SOTA, not this scaffold"}

# =============================================================================
# §5. SELF-TEST
# =============================================================================
def self_test(verbose=True)->bool:
    print("="*76); print("  AF-PHENOTYPE-SDA v2 — self-test"); print("="*76)
    torch.manual_seed(0); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] FIX 1: pos_embed shape and 3-D single repr")
    enc=AFStructureEncoder(d_model=64,c_z=32,n_blocks=2,n_heads=2,
                           n_structure_blocks=1,max_len=64)
    rep("pos_embed is 3-D (1,max_len,d)",enc.pos_embed.dim()==3,
        f"shape={tuple(enc.pos_embed.shape)}")
    tokens=torch.randint(0,20,(2,32)); mask=torch.ones(2,32,dtype=torch.bool)
    o=enc(tokens,mask)
    rep("single_repr is 3-D (B,L,d)",o["single_repr"].dim()==3,
        f"shape={tuple(o['single_repr'].shape)}")
    rep("positions shape",tuple(o["positions"].shape)==(2,32,3))
    rep("distogram shape",tuple(o["distogram_logits"].shape)==(2,32,32,32))
    v1=torch.randn(1,1,32,64)
    rep("v1 (1,1,L,d) WOULD broadcast to 4-D",
        (torch.randn(2,32,64)+v1).dim()==4)

    print("\n[T2] joint model + loss (dead line removed)")
    m=AFPhenotypeJoint(d_model=64,c_z=32,n_evo_blocks=2,n_heads=2,
                       n_structure_blocks=1,max_len=64,n_ensemble=4)
    tc=torch.randn(2,32,3)
    out=m(tokens,mask,tc)
    rep("phenotype logits shape",tuple(out["phenotype_logits"].shape)==(2,2))
    rep("structure loss present","structure_loss" in out)
    L=JointLoss()(out,{"labels":torch.tensor([0,1])})
    rep("total loss finite",bool(torch.isfinite(L["total"])))
    L["total"].backward()
    gs=[p.grad for p in m.parameters() if p.grad is not None]
    rep("finite gradients",bool(gs) and all(torch.isfinite(g).all() for g in gs))

    print("\n[T3] FIX 2: persistent SDA projection")
    sda=SDAOneScaleAdapter(d_model=64,d_mn=48,domain_size=16,
                           head_in_width=m.in_width)
    rep("projection has registered parameters",
        verify_persistent_projection(sda)["n_registered_params"]>0)
    n_before=sum(p.numel() for p in sda.parameters())
    _=sda.compress(torch.randn(2,16,64)); _=sda.compress(torch.randn(2,16,64))
    rep("parameter count unchanged across calls",
        sum(p.numel() for p in sda.parameters())==n_before)
    class V1Style(nn.Module):
        def forward(self,x): return nn.Linear(64,48,bias=False)(x.mean(1))
    b=False
    try: verify_persistent_projection(V1Style(),"v1 compression")
    except IntegrityError: b=True
    rep("v1 fresh-Linear-per-call DETECTED",b)

    print("\n[T4] FIX 3: predict_large widths")
    long_tokens=torch.randint(0,20,(2,40))
    long_mask=torch.ones(2,40,dtype=torch.bool)
    r=sda.predict_large(m,long_tokens,long_mask)
    rep("multi-domain path runs",r.get("n_domains",1)>1,
        f"n_domains={r.get('n_domains')}")
    rep("logits width matches head",tuple(r["phenotype_logits"].shape)==(2,2))
    rep("head input width correct",sda.head_in_width==m.in_width)

    print("\n[T5] FIX 4: no `or True`")
    rep("a False test stays False",assert_real_test("demo",False)["passed"] is False)
    dg=connector_diagnostics(out)
    rep("ensemble VERIFIED distinct",dg["member_check"]["members_distinct"])
    rep("class fractions sum to 1",abs(sum(dg["class_fractions"])-1)<1e-6)
    rep("not-SOTA declared","AlphaMissense" in dg["NOT_SOTA"])

    print("\n[T6] ClinVar honesty")
    cv=ClinVarClient(offline=True)
    r=cv.fetch_gene("BRCA1")
    rep("offline => UNAVAILABLE, no fabricated labels",
        r["status"]=="UNAVAILABLE" and not r["records"])

    print("\n[T7] SD report")
    rr=dg["sd_report"]
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, {rr['mode_E']} Mode E")
    rep("Mode E retained",rr["mode_E"]>0)
    rep("research_only",dg["clinical_actionability"]==RESEARCH_ONLY)
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  pos_embed is (1,max_len,d); the single representation stays 3-D and
  MultiheadAttention receives the shape it expects.
  The SDA compression projection is registered in __init__ and trained;
  a fresh Linear per call is detected and refused.
  predict_large projects to the phenotype head's input width.
  No `or True` escape hatch. Dead loss line removed.

NOT CLAIMED
  Weights are RANDOM until trained on PDB. AlphaMissense is SOTA.
  ClinVar labels are submitter assertions; circularity is declared.
  SDA-ONE at 100M residues needs large GPU infrastructure.
  No gene-editor design. No clinical use.
""")
    return ok

def main()->int:
    ap=argparse.ArgumentParser(description="AF-phenotype-SDA connector v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
                    choices=["selftest","demo","clinvar","diagnostics"])
    ap.add_argument("--gene",default="BRCA1")
    ap.add_argument("--online",action="store_true")
    ap.add_argument("--json",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    out={}
    if a.cmd=="clinvar":
        out=ClinVarClient(offline=not a.online).fetch_gene(a.gene)
    elif a.cmd in ("demo","diagnostics"):
        m=AFPhenotypeJoint(d_model=64,c_z=32,n_evo_blocks=2,n_heads=2,
                           n_structure_blocks=1,max_len=64).eval()
        with torch.no_grad(): o=m(torch.randint(0,20,(2,32)))
        out=connector_diagnostics(o) if a.cmd=="diagnostics" else {
            "phenotype_logits":o["phenotype_logits"].tolist(),
            "phenotype_std":o["phenotype_std"].tolist(),
            "clinical_actionability":o["clinical_actionability"]}
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
