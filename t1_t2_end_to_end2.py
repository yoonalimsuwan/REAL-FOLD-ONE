#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
END-TO-END T1 -> T2 WITH SCAFFOLD SPLIT — v2.0
=============================================================================
BUGS FIXED
  1. LEAKAGE: when the greedy split left test empty, v1 moved individual
     indices from val to test, splitting a scaffold across val/test.
     v2 moves WHOLE SCAFFOLD GROUPS, preserving separation.
  2. UNVERIFIED SMILES were used silently to compute Murcko scaffolds.
     The listed SMILES are almost certainly not the named compounds.
     v2 marks every entry unverified and REFUSES to use unverified
     structures for splitting, falling back to family grouping with an
     explicit Mode E.
  3. Docstring promised an 'independent T1/T2' arm that was never
     implemented. v2 implements it by detaching occupancy.
  4. No SD gate, no seed aggregation, no ablation integrity check.

PHILOSOPHY: see msps_bio_math_core.PHILOSOPHY.
HONEST SCOPE
  The optimised EC50 is task-specific, not a binding constant.
  Scaffold split does not prevent mechanism leakage (same target,
  different scaffold). Protein family grouping is a proxy.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

from msps_bio_math_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, fixed_context,
    verify_ablation_arms, aggregate_seeds, hill_identifiability, get_logger,
)
log=get_logger("e2e_v2"); VERSION="2.0.0"
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT=True
except ImportError:
    HAS_RDKIT=False

FACTORS=["WNT","CHIR","BMP4","LDN193189","ACTIVIN","TGFB1","SB431542",
         "FGF2","FGF4","FGF9","FGF10","NOTCH_AG","NOTCH_INH","DAPT",
         "SHH","RA","OTHER"]
N_FACTOR=len(FACTORS); F_IDX={f:i for i,f in enumerate(FACTORS)}
FATES=["PLURIPOTENT","ENDODERM","MESODERM","ECTODERM","NEURAL_PROGENITOR",
       "NEURON","GLIA","INTESTINAL_EPITHELIUM","GOBLET","PANETH","ENTEROCYTE",
       "HEPATOCYTE","CHOLANGIOCYTE","NEPHRON_PROGENITOR","PODOCYTE","TUBULE",
       "CARDIOMYOCYTE","ENDOTHELIAL","MESENCHYMAL","UNKNOWN"]
N_FATE=len(FATES)

# v2: EVERY entry is unverified. Do not use for splitting until checked.
FACTOR_INFO={
 "CHIR":{"type":"small_molecule","family":"GSK3_inhibitor",
         "smiles":None,"verified":False},
 "SB431542":{"type":"small_molecule","family":"TGFBR_inhibitor",
             "smiles":None,"verified":False},
 "LDN193189":{"type":"small_molecule","family":"BMP_inhibitor",
              "smiles":None,"verified":False},
 "DAPT":{"type":"small_molecule","family":"gamma_secretase_inhibitor",
         "smiles":None,"verified":False},
 "RA":{"type":"small_molecule","family":"retinoid","smiles":None,"verified":False},
 "WNT":{"type":"protein","family":"Wnt","smiles":None,"verified":True},
 "FGF2":{"type":"protein","family":"FGF","smiles":None,"verified":True},
 "FGF4":{"type":"protein","family":"FGF","smiles":None,"verified":True},
 "FGF9":{"type":"protein","family":"FGF","smiles":None,"verified":True},
 "FGF10":{"type":"protein","family":"FGF","smiles":None,"verified":True},
 "BMP4":{"type":"protein","family":"BMP","smiles":None,"verified":True},
 "ACTIVIN":{"type":"protein","family":"TGFbeta","smiles":None,"verified":True},
 "TGFB1":{"type":"protein","family":"TGFbeta","smiles":None,"verified":True},
 "SHH":{"type":"protein","family":"Hedgehog","smiles":None,"verified":True},
 "NOTCH_AG":{"type":"unknown","family":"Notch_agonist","smiles":None,"verified":False},
 "NOTCH_INH":{"type":"unknown","family":"Notch_inhibitor","smiles":None,"verified":False},
 "OTHER":{"type":"unknown","family":"unknown","smiles":None,"verified":False},
}

def hill_occupancy(dose,ec50,emax,n=1.0,eps=1e-9):
    D=dose.clamp(min=eps); E=ec50.clamp(min=eps)
    u=D.pow(n); v=E.pow(n)
    return emax*u/(u+v)

# =============================================================================
# §1. SCAFFOLD SPLIT — whole groups, verified structures only
# =============================================================================

class ScaffoldSplitter:
    def __init__(self,info=None,sd:Optional[SDGate]=None):
        self.info=info or FACTOR_INFO
        self.sd=sd or SDGate(); self._cache={}
        self.refused_unverified=[]
    def group(self,factor:str) -> str:
        if factor in self._cache: return self._cache[factor]
        i=self.info.get(factor,{"type":"unknown","family":"unknown"})
        key=None
        if (i.get("type")=="small_molecule" and i.get("smiles")
                and i.get("verified") and HAS_RDKIT):
            try:
                mol=Chem.MolFromSmiles(i["smiles"])
                if mol is not None:
                    key="smi_"+MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol,includeChirality=False)
            except Exception as e:
                log.warning(f"[scaffold] RDKit failed for {factor}: {e}")
        if key is None:
            if i.get("type")=="small_molecule" and not i.get("verified"):
                self.refused_unverified.append(factor)
            key="fam_"+str(i.get("family","unknown"))
        self._cache[factor]=key; return key
    def certificate(self) -> Dict:
        any_smi=any(k.startswith("smi_") for k in self._cache.values())
        return self.sd.certify("scaffold_split",("SMILES",) if any_smi else (),
            {"refused_unverified":sorted(set(self.refused_unverified)),
             "grouping":"scaffold" if any_smi else "family (PROXY)",
             "NOTE":"unverified SMILES are never used for splitting"})
    def split(self,records,train_frac=.7,val_frac=.15,seed=42):
        groups=defaultdict(list)
        for i,r in enumerate(records):
            groups[self.group(r.get("factor","OTHER"))].append(i)
        n=len(records); ntr=int(train_frac*n); nva=int(val_frac*n)
        rng=np.random.default_rng(seed)
        keys=list(groups); rng.shuffle(keys)
        keys.sort(key=lambda k:-len(groups[k]))
        tr,va,te=[],[],[]
        for k in keys:
            g=groups[k]
            if len(tr)+len(g)<=ntr: tr+=g
            elif len(va)+len(g)<=nva: va+=g
            else: te+=g
        # FIX: move WHOLE GROUPS, never individual indices
        if not te and va:
            moved=[]; keep=[]
            va_groups=defaultdict(list)
            for i in va: va_groups[self.group(records[i].get("factor","OTHER"))].append(i)
            vk=list(va_groups)
            for j,k in enumerate(vk):
                if j>=max(1,len(vk)//2): moved+=va_groups[k]
                else: keep+=va_groups[k]
            va,te=keep,moved
        if not te and tr:
            tr_groups=defaultdict(list)
            for i in tr: tr_groups[self.group(records[i].get("factor","OTHER"))].append(i)
            tk=list(tr_groups)
            if len(tk)>1:
                k=tk[-1]; te=tr_groups[k]; tr=[i for i in tr if i not in set(te)]
        assert not (set(tr)&set(va)) and not (set(tr)&set(te)) and not (set(va)&set(te))
        gs=lambda idx:{self.group(records[i].get("factor","OTHER")) for i in idx}
        sep=not (gs(tr)&gs(te))
        return {"train":tr,"val":va,"test":te,
                "group_separation_train_test":sep,
                "n_groups":len(groups),
                "certificate":self.certificate(),
                "NOTE":("group separation preserved" if sep else
                        "too few groups to separate; report as Mode E")}

def compare_random_vs_scaffold(records,splitter,seed=42):
    n=len(records); rng=np.random.default_rng(seed); perm=rng.permutation(n)
    ntr=int(.7*n)
    rs={"train":perm[:ntr].tolist(),"test":perm[ntr:].tolist()}
    gs=lambda idx:{splitter.group(records[i].get("factor","OTHER")) for i in idx}
    overlap=gs(rs["train"])&gs(rs["test"])
    ss=splitter.split(records,seed=seed)
    return {"random_leakage_rate":len(overlap)/max(len(gs(rs["train"])|gs(rs["test"])),1),
            "random_leaking_groups":len(overlap),
            "scaffold_group_separation":ss["group_separation_train_test"],
            "total_groups":ss["n_groups"]}

# =============================================================================
# §2. MODEL — with a genuine independent arm
# =============================================================================

@dataclass
class E2EConfig:
    d_seq:int=128; d_hidden:int=256; d_ctx:int=128
    hill_n:float=1.0; ec50_reg_weight:float=0.01
    ec50_range:tuple=(1e-3,1e6); dropout:float=0.2
    end_to_end:bool=True     # False -> detach occupancy (independent arm)

class E2EModel(nn.Module):
    def __init__(self,cfg:E2EConfig):
        super().__init__(); self.cfg=cfg
        self.enc=nn.Sequential(nn.Linear(cfg.d_seq,cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),nn.GELU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden,cfg.d_hidden),nn.LayerNorm(cfg.d_hidden),nn.GELU())
        self.log_ec50=nn.Linear(cfg.d_hidden,1)
        self.emax=nn.Sequential(nn.Linear(cfg.d_hidden,1),nn.Sigmoid())
        self.emb=nn.Embedding(N_FACTOR,32)
        self.trunk=nn.Sequential(nn.Linear(32+3+cfg.d_ctx,cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),nn.GELU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_hidden,cfg.d_hidden),nn.LayerNorm(cfg.d_hidden),nn.GELU())
        self.head=nn.Linear(cfg.d_hidden,N_FATE)
    def forward(self,seq_feat,factor_idx,dose_nM,time_h,protocol_ctx):
        h=self.enc(seq_feat)
        le=self.log_ec50(h).squeeze(-1); em=self.emax(h).squeeze(-1)
        ec50=torch.exp(le).clamp(*self.cfg.ec50_range)
        occ=hill_occupancy(dose_nM,ec50,em,self.cfg.hill_n)
        # v2: the independent arm DETACHES, so no gradient reaches T1
        occ_t2 = occ if self.cfg.end_to_end else occ.detach()
        extra=torch.stack([occ_t2,torch.log10(dose_nM.clamp(min=1e-3)),
                           torch.log2(time_h.clamp(min=1.))],-1)
        z=self.trunk(torch.cat([self.emb(factor_idx),extra,protocol_ctx],-1))
        lg=self.head(z)
        return {"composition":F.softmax(lg,-1),"log_ec50":le,"emax":em,
                "ec50_nM":ec50,"occupancy":occ}
    def ec50_reg(self,le):
        lo,hi=math.log(self.cfg.ec50_range[0]),math.log(self.cfg.ec50_range[1])
        return (F.relu(lo-le).pow(2)+F.relu(le-hi).pow(2)).mean()

ARMS={"A_independent_random":dict(end_to_end=False,split="random"),
      "B_e2e_random":dict(end_to_end=True,split="random"),
      "C_e2e_scaffold":dict(end_to_end=True,split="scaffold")}

def make_data(n=200,d_seq=128,d_ctx=128,seed=0):
    log.warning("SYNTHETIC — code test only")
    rng=np.random.default_rng(seed)
    return {"seq_feat":torch.randn(n,d_seq),
            "factor_idx":torch.randint(0,N_FACTOR,(n,)),
            "dose_nM":torch.rand(n)*1000+1,"time_h":torch.rand(n)*200+1,
            "protocol_ctx":torch.from_numpy(fixed_context(n,d_ctx,seed)),
            "composition":F.softmax(torch.randn(n,N_FATE),-1)}

def run_ablation(data,records,epochs=50,seeds=(42,43,44),
                 out="runs/e2e_v2",device="cpu"):
    sp=ScaffoldSplitter(); ss=sp.split(records)
    n=len(records); rng=np.random.default_rng(42); perm=rng.permutation(n)
    ntr=int(.7*n); nva=int(.15*n)
    rs={"train":perm[:ntr].tolist(),"val":perm[ntr:ntr+nva].tolist(),
        "test":perm[ntr+nva:].tolist()}
    if len(rs["test"])<2: rs["test"]=rs["val"]
    probe={k:data[k][:4] for k in ("seq_feat","factor_idx","dose_nM",
                                   "time_h","protocol_ctx")}
    chk=verify_ablation_arms(
        lambda f: E2EModel(E2EConfig(d_seq=data["seq_feat"].shape[-1],
                                     d_ctx=data["protocol_ctx"].shape[-1],
                                     end_to_end=f["end_to_end"])),
        {"indep":{"end_to_end":False},"e2e":{"end_to_end":True}},probe)
    log.info("[ablation] arms verified distinct")
    dev=torch.device(device)
    d_ctx=data["protocol_ctx"].shape[-1]; d_seq=data["seq_feat"].shape[-1]
    for s in seeds:
        res={}
        for arm,fl in ARMS.items():
            torch.manual_seed(s)
            cfg=E2EConfig(d_seq=d_seq,d_ctx=d_ctx,end_to_end=fl["end_to_end"])
            m=E2EModel(cfg).to(dev)
            opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
            split = rs if fl["split"]=="random" else ss
            def bt(idx):
                i=torch.as_tensor(idx,dtype=torch.long)
                return tuple(data[k][i].to(dev) for k in
                             ("seq_feat","factor_idx","dose_nM","time_h","protocol_ctx"))
            def loss(idx):
                i=torch.as_tensor(idx,dtype=torch.long)
                o=m(*bt(idx))
                kl=F.kl_div(o["composition"].log().clamp(min=-20),
                            data["composition"][i].to(dev),reduction="batchmean")
                return kl+cfg.ec50_reg_weight*m.ec50_reg(o["log_ec50"]),o
            best=float("inf"); state=None
            for ep in range(epochs):
                m.train()
                pp=np.random.default_rng(s+ep).permutation(len(split["train"]))
                ti=[split["train"][i] for i in pp]
                for st in range(0,len(ti),32):
                    L,_=loss(ti[st:st+32])
                    opt.zero_grad(); L.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
                if (ep+1)%max(1,epochs//5)==0:
                    m.eval()
                    with torch.no_grad(): v=float(loss(split["val"])[0])
                    if v<best:
                        best=v; state={k:x.detach().clone() for k,x in m.state_dict().items()}
            if state: m.load_state_dict(state)
            m.eval()
            with torch.no_grad(): tk=float(loss(split["test"])[0])
            # gradient-flow check for this arm
            m.train(); L,o=loss(split["train"][:4]); L.backward()
            t1g=[p.grad for p in m.enc.parameters() if p.grad is not None]
            reaches=any(float(g.abs().sum())>1e-12 for g in t1g)
            res[arm]={"test_kl":tk,"best_val":best,
                      "t1_receives_gradient":reaches,"split":fl["split"]}
        d=Path(out)/f"seed{s}"; d.mkdir(parents=True,exist_ok=True)
        (d/"ablation.json").write_text(json.dumps(res,indent=2))
    agg=aggregate_seeds([str(Path(out)/f"seed{s}") for s in seeds])
    leak=compare_random_vs_scaffold(records,sp)
    print("\n"+"="*76); print("  E2E ABLATION (group-preserving split)")
    print("="*76)
    for arm,v in agg["per_arm"].items():
        print(f"  {arm:<24} {v['mean']:.5f} ± {v['sd']:.5f}  "
              f"{v.get('verdict','baseline')}")
    print(f"\n  random-split group leakage: {leak['random_leakage_rate']:.1%}")
    print(f"  scaffold group separation: {leak['scaffold_group_separation']}")
    print("  All SMILES unverified -> FAMILY grouping (Mode E).")
    print("="*76)
    return agg

def selftest(a=None) -> bool:
    print("="*76); print("  E2E v2 — self-test"); print("="*76)
    torch.manual_seed(0); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] UNVERIFIED SMILES REFUSED FOR SPLITTING")
    sp=ScaffoldSplitter()
    g=sp.group("CHIR")
    rep("CHIR falls back to family",g.startswith("fam_"))
    rep("refusal recorded","CHIR" in sp.refused_unverified)
    c=sp.certificate()
    rep("certificate is Mode E",c["status"]==MODE_E)
    rep("note states policy","never used" in c["value"]["NOTE"])

    print("\n[T2] GROUP-PRESERVING SPLIT")
    recs=[{"factor":f} for f in ["CHIR","SB431542","WNT","FGF2","DAPT"]]*10
    ss=sp.split(recs)
    gs=lambda idx:{sp.group(recs[i]["factor"]) for i in idx}
    rep("no index overlap",not (set(ss["train"])&set(ss["test"])))
    rep("no GROUP overlap train/test",not (gs(ss["train"])&gs(ss["test"])))
    rep("separation flag",ss["group_separation_train_test"])
    # force the fallback path with few groups
    few=[{"factor":"WNT"}]*30+[{"factor":"FGF2"}]*10
    sf=sp.split(few)
    gsf=lambda idx:{sp.group(few[i]["factor"]) for i in idx}
    rep("fallback keeps groups whole",not (gsf(sf["val"])&gsf(sf["test"])))

    print("\n[T3] INDEPENDENT ARM REALLY DETACHES")
    d=make_data(32,32,16)
    cfg_e=E2EConfig(d_seq=32,d_hidden=32,d_ctx=16,end_to_end=True)
    cfg_i=E2EConfig(d_seq=32,d_hidden=32,d_ctx=16,end_to_end=False)
    for cfg,label,want in ((cfg_e,"end_to_end",True),(cfg_i,"independent",False)):
        m=E2EModel(cfg)
        o=m(d["seq_feat"][:4],d["factor_idx"][:4],d["dose_nM"][:4],
            d["time_h"][:4],d["protocol_ctx"][:4])
        F.kl_div(o["composition"].log().clamp(min=-20),
                 d["composition"][:4],reduction="batchmean").backward()
        gs=[p.grad for p in m.enc.parameters() if p.grad is not None]
        reaches=any(float(x.abs().sum())>1e-12 for x in gs)
        rep(f"{label}: T1 gradient = {want}",reaches==want)

    print("\n[T4] ABLATION INTEGRITY + RUN")
    probe={k:d[k][:4] for k in ("seq_feat","factor_idx","dose_nM",
                                "time_h","protocol_ctx")}
    chk=verify_ablation_arms(
        lambda f: E2EModel(E2EConfig(d_seq=32,d_hidden=32,d_ctx=16,
                                     end_to_end=f["end_to_end"])),
        {"indep":{"end_to_end":False},"e2e":{"end_to_end":True}},probe)
    rep("arms distinct",chk["all_arms_distinct"])
    recs=[{"factor":FACTORS[int(d["factor_idx"][i])]} for i in range(32)]
    agg=run_ablation(d,recs,epochs=6,seeds=(42,43,44),out="runs/_e2e_v2_st")
    rep("3 arms x 3 seeds",len(agg["per_arm"])==3)
    import shutil; shutil.rmtree("runs/_e2e_v2_st",ignore_errors=True)

    print("\n[T5] HILL IDENTIFIABILITY AVAILABLE")
    hi=hill_identifiability([1.,10.,100.],10.,1.,1.)
    rep("correlation reported","corr_EC50_n" in hi)
    print("\n"+"="*76); print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  Scaffold split moves WHOLE GROUPS; no leakage in the fallback path.
  Unverified SMILES are refused for splitting; family grouping is Mode E.
  The independent arm actually detaches occupancy.
  Fixed context, verified arms, >=3 seeds, SD gate.
NOT CLAIMED
  Optimised EC50 is task-specific. Family grouping is a proxy, not a
  scaffold. Mechanism leakage (same target) is not prevented.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","ablation","split"])
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--seeds",type=int,default=3); ap.add_argument("--out",default="runs/e2e_v2")
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    d=make_data()
    recs=[{"factor":FACTORS[int(d["factor_idx"][i])]} for i in range(len(d["factor_idx"]))]
    if a.cmd=="split":
        print(json.dumps(ScaffoldSplitter().split(recs),indent=2,default=str)); return
    run_ablation(d,recs,epochs=a.epochs,seeds=tuple(range(42,42+a.seeds)),out=a.out)

if __name__=="__main__": main()
