#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MDOE v2.0 — Morphogen-Directed Organoid Engineering Predictor
T1 -> T2 -> T3 with uncertainty propagation; T4 GATED.
Now wired to the CORRECTED ESCA mathematics.

CORRECTED HONESTY REGISTRY (this is the main change)
  Every component name is an analogy. The registry now cites the CORRECT
  theorem numbers and states which corrections apply.

  OPMC        named after Thm 7.5 / Prop 12.2 (was "Unified Ordinal-Tensor
              Thm 11.1"). IS a learned low-rank bottleneck + fixed-point
              iteration. The CERTIFIED statement is the change-point
              localisation radius 46 sigma^2 L/kappa^2 under (L1,D,O) and
              (SNR) with a (K+1)^2 factor. K-free separation is [Open].
  Carpet-19   named after Thm 17.16. IS a fixed carpet graph Laplacian
              used as a multi-scale BASIS. The continuum limit needs
              THREE imported theorems (KZ-R, KZ-S, BBKT). (C1) now follows
              from (KZ-R)+(G); (C2) is derived from (C1). Non-p.c.f.
  delta_min   named after Def 2.2 / Lemma 18.1. IS a thresholded gate with
              a fatness guard. The isodiametric bound is DERIVED for
              droplet-like components only; filaments/sheets are [Open].
  Hodge       named after the decomposition. IS a learned 3-way split
              with a Laplacian penalty. NOT differential forms.
  Hilbert-Polya  named after an UNPROVEN conjecture. IS a learned
              self-adjoint H = H0 + V V^T, eigendecomposed. NO connection
              to Riemann zeros.
  SSC         EMA smoother, eps=0.0028.
  CSOC        two-level hysteresis counter. v2 uses the CORRECTED no-Zeno
              certificate (Thm 18.6): Doob bound, NO horizon truncation.
  composition NEW: the T2 target IS an occupation measure. Thm 8.3 gives
              a W2 rate; Prop 8.6 / Thm 8.1 give a light-vs-multistable
              test. This is a genuine correspondence, not a name.
  response tail NEW: Kou double-exponential jumps, Gumbel extremes,
              heaviest-regime domination, moment-explosion gate.

PHILOSOPHY (full text in msps_esca_bio_core)
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

HONEST SCOPE — THE THREE THINGS THAT MATTER
  1. NO public paired T1->T2->T3->T4 dataset exists. Loaders find real
     data or REFUSE. Nothing is fabricated.
  2. T4 (engraftment) is GATED, not trained. Published mouse-transplant
     n is tens-to-hundreds; an overfit predictor is a clinical hazard.
  3. THE ABLATION IS THE RESULT. Keep a component only if it helps on
     held-out data with >= 3 seeds. Not because it has a nice name.

  Not a clinical device. Not a transplant decision aid.
  Organoid response != patient response. No claim of designing an organ.
=============================================================================
Developer: PAI and Yoon A. Limsuwan / MSPS NETWORK
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

from msps_esca_bio_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    theorem_A_rate, occupation_regime_test, composition_as_occupation,
    opmc_certified, KouRegime, moment_estimable, gumbel_extreme,
    heaviest_regime_dominates, esscher_worst_case,
    classify_morphology_event, no_zeno_certificate, doob_exit_bound,
    minimax_decision, carpet_status, compression_bound,
    boundary_null_check, get_logger,
)
log=get_logger("mdoe")
VERSION="2.0.0"

class DataRefusal(Refusal): pass

AA="ARNDCQEGHILKMFPSTWYV"; AA_IDX={a:i for i,a in enumerate(AA)}
FACTORS=["WNT","RSPO","BMP4","NOGGIN","FGF2","FGF4","FGF9","FGF10",
 "ACTIVIN","NODAL","TGFB1","SHH","RA","EGF","HGF","VEGF","IGF1",
 "NOTCH_AG","NOTCH_INH","CHIR","SB431542","LDN193189","DAPT","Y27632","OTHER"]
F_IDX={f:i for i,f in enumerate(FACTORS)}; N_F=len(FACTORS)
OPERATIONS=["ADD","REMOVE","MEDIA_CHANGE","PASSAGE","EMBED_MATRIGEL",
 "AIR_LIQUID","SPINNING","PERFUSION","CO_CULTURE","ASSAY"]
OP_IDX={o:i for i,o in enumerate(OPERATIONS)}; N_OP=len(OPERATIONS)
FATES=["PLURIPOTENT","ENDODERM","MESODERM","ECTODERM",
 "NEURAL_PROGENITOR","NEURON","GLIA",
 "INTESTINAL_EPITHELIUM","GOBLET","PANETH","ENTEROCYTE",
 "HEPATOCYTE","CHOLANGIOCYTE",
 "NEPHRON_PROGENITOR","PODOCYTE","TUBULE",
 "CARDIOMYOCYTE","ENDOTHELIAL","MESENCHYMAL","UNKNOWN"]
N_FATE=len(FATES)

HONESTY={
 "opmc":{"named_after":"ESCA Thm 7.5 / Prop 12.2 (corrected reference)",
   "is":"learned low-rank bottleneck + fixed-point steps",
   "certified_statement":"change-point radius 46 sigma^2 L/kappa^2",
   "packages":["L1","D","O"],
   "not_claimed":"lossless biological compression",
   "correction":"(SNR) carries (K+1)^2; K-free separation is [Open] (O1)"},
 "carpet":{"named_after":"ESCA Thm 17.16 (Carpet closure)",
   "is":"fixed carpet graph Laplacian as a multi-scale BASIS",
   "imported":["KZ-R","KZ-S","BBKT"],
   "proved_here":["chain property (G)","(C1) from (KZ-R)+(G)","(C2) from (C1)"],
   "not_claimed":"any theorem about organoids",
   "correction":"non-p.c.f.; three published inputs required (K7)"},
 "amr":{"named_after":"Def 2.2 (D) / Lemma 18.1",
   "is":"thresholded refinement gate with a fatness guard",
   "not_claimed":"isodiametric bound governs cell fate",
   "correction":"derived for droplet-like components only; "
                "filaments/sheets are [Open] (O6)"},
 "hodge":{"named_after":"Hodge decomposition",
   "is":"learned 3-way split + Laplacian kernel penalty",
   "not_claimed":"differential forms on a manifold"},
 "hilbert_polya":{"named_after":"Hilbert-Polya conjecture (UNPROVEN)",
   "is":"learned self-adjoint H=H0+VV^T, eigh",
   "not_claimed":"any connection to Riemann zeros"},
 "ssc":{"named_after":"EMA eps=0.0028","is":"exponential moving average"},
 "csoc":{"named_after":"ESCA Thm 18.6 / Cor 18.7 (corrected)",
   "is":"two-level hysteresis event counter",
   "certified_statement":"Doob exit bound; T_n -> inf a.s.",
   "packages":["U","R","M"],
   "correction":"horizon truncation REMOVED; Doob replaces the "
                "unproved exponential bound (K8)"},
 "composition":{"named_after":"ESCA Thm 8.1/8.2/8.3 (occupation measures)",
   "is":"the T2 target IS an occupation vector on Delta^d",
   "certified_statement":"W2 <= sqrt(4L(s/n)^2 + 4 sum d Gamma/m)",
   "packages":["A1","A2","A3"],
   "genuine":"this is a literal correspondence, not an analogy",
   "open":"infinite-mean Lamperti regime is [Open] (O4)"},
 "response_tail":{"named_after":"ESCA Part II §16 (Kou jumps)",
   "is":"double-exponential tail fit on log-response",
   "certified_statement":"Gumbel extremes; heaviest-regime domination; "
                         "moment-explosion threshold",
   "genuine":"exact results, directly applicable to IC50 variability"},
 "stats":{"named_after":"Politis-Romano; Shao; split conformal",
   "is":"exactly those procedures",
   "requires":"stationarity + mixing + finite variance; exchangeability"},
}

# -----------------------------------------------------------------------------
# §1. DATA LAYER — refuses to fabricate
# -----------------------------------------------------------------------------

@dataclass
class CurationRecord:
    record_id:str; ligand_sequence:str; receptor:str
    ec50_nM:Optional[float]=None; emax_frac:Optional[float]=None
    assay_type:Optional[str]=None
    factor:Optional[str]=None; dose_nM:Optional[float]=None
    time_h:Optional[float]=None
    fate_composition:Optional[List[float]]=None
    fate_source:Optional[str]=None
    organoid_diameter_um:Optional[float]=None
    viability_frac:Optional[float]=None
    marker_panel:Optional[Dict[str,float]]=None
    engraftment_success:Optional[int]=None; host_species:Optional[str]=None
    doi:Optional[str]=None; verified:bool=False; notes:str=""
    n_cells:Optional[int]=None
    cell_labels:Optional[List[str]]=None
    annotation_reviewed:bool=False
    def has_t1(self): return (self.verified and self.ec50_nM is not None
                              and len(self.ligand_sequence)>0 and self.doi)
    def has_t2(self): return (self.verified and self.dose_nM is not None
                              and self.time_h is not None
                              and self.fate_composition is not None
                              and len(self.fate_composition)==N_FATE
                              and self.fate_source is not None)
    def has_t3(self): return (self.verified and
                              self.organoid_diameter_um is not None and
                              self.viability_frac is not None)
    def has_t4(self): return (self.verified and
                              self.engraftment_success is not None and
                              self.host_species is not None)
    def t2_uncertainty(self):
        if self.n_cells is None: return None
        return theorem_A_rate(self.n_cells,N_FATE)
    def t2_regime(self):
        if not self.cell_labels: return None
        return occupation_regime_test(self.cell_labels,FATES)

def load_curation(path:Path) -> List[CurationRecord]:
    if not path.exists():
        raise DataRefusal(
            f"{path} not found.\n"
            "The full T1->T2->T3->T4 chain has NO public paired dataset.\n"
            "You must curate it. Schema: CurationRecord in this file.\n"
            "  T1: MaveDB, PDBbind, dose-response in Janda 2017 / Silva 2019\n"
            "  T2: GEO organoid scRNA-seq WITH protocol metadata + n_cells\n"
            "  T3: organoid screening supplements\n"
            "  T4: mouse transplant tables (n is SMALL)\n"
            "Every record needs a DOI and verified=true.\n"
            "v2: n_cells enables the Thm 8.3 W2 bound; cell_labels enable "
            "the light-vs-multistable regime test.")
    recs=[]
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        try: recs.append(CurationRecord(**json.loads(line)))
        except TypeError as e: log.warning(f"bad record skipped: {e}")
    counts={k:sum(getattr(r,f"has_{k}")() for r in recs)
            for k in ("t1","t2","t3","t4")}
    log.info(f"[curation] {len(recs)} records | usable: {counts}")
    unv=sum(not r.verified for r in recs)
    if unv: log.warning(f"[curation] {unv} NOT verified — excluded")
    nw=sum(1 for r in recs if r.has_t2() and r.n_cells)
    log.info(f"[curation] {nw} T2 records carry n_cells -> W2 bounds available")
    return recs

# -----------------------------------------------------------------------------
# §2. MATH CORE — components honest about what they are
# -----------------------------------------------------------------------------

def sgate(x,tau=1e-2): return torch.sigmoid(x/tau)

class LowRank(nn.Module):
    def __init__(self,di,do,r=32,bias=False):
        super().__init__()
        self.U=nn.Parameter(torch.randn(di,r)/math.sqrt(di))
        self.V=nn.Parameter(torch.zeros(r,do))
        self.b=nn.Parameter(torch.zeros(do)) if bias else None
    def forward(self,x):
        y=(x@self.U.to(x.dtype))@self.V.to(x.dtype)
        return y+self.b.to(x.dtype) if self.b is not None else y

class OPMC(nn.Module):
    """Learned bottleneck. See HONESTY['opmc'] for the corrected reference."""
    L=0.438520
    def __init__(self,d_in,d_out=320,r=32,steps=23,enabled=True):
        super().__init__()
        self.enabled=enabled; self.d_out=d_out
        self.enc=LowRank(d_in,d_out,r)
        self.step=nn.ModuleList([nn.Sequential(LowRank(d_out,d_out,r),nn.Tanh())
                                 for _ in range(min(steps,8) if enabled else 1)])
        self.bands=nn.Parameter(torch.zeros(5))
        self.gate=nn.Parameter(torch.tensor(0.1))
    @staticmethod
    def k_star(eps=1e-8,D0=1.0):
        return math.ceil(math.log(eps/D0)/math.log(OPMC.L))
    def forward(self,x):
        if not self.enabled: return {"z":self.enc(x),"k_star":0}
        z=torch.tanh(self.enc(x))
        for s in self.step: z=s(z)+self.gate*z
        w=torch.sigmoid(self.bands); c=self.d_out//5
        z=torch.cat([z[...,i*c:(i+1)*c]*w[i] for i in range(5)],-1)
        return {"z":z,"k_star":self.k_star()}

def carpet_laplacian(n_cells:int) -> Tensor:
    side=max(int(round(math.sqrt(n_cells))),3); n=side*side
    A=torch.zeros(n,n)
    for i in range(side):
        for j in range(side):
            if i%3==1 and j%3==1: continue
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii,jj=i+di,j+dj
                if 0<=ii<side and 0<=jj<side:
                    if ii%3==1 and jj%3==1: continue
                    A[i*side+j,ii*side+jj]=1.0
    deg=A.sum(1,keepdim=True).clamp(min=1.0)
    return torch.diag(deg.squeeze())-A

class CarpetAMR(nn.Module):
    """BASIS only. Continuum limit needs KZ-R/KZ-S/BBKT. HONESTY['carpet']."""
    RHO_MIN=9.0/8.0
    def __init__(self,d_in,n_cells=81,levels=3,kappa=2.0,r=32,enabled=True):
        super().__init__()
        self.enabled=enabled; self.kappa=kappa
        self.register_buffer("L",carpet_laplacian(n_cells))
        self.proj=nn.ModuleList([LowRank(d_in,self.L.shape[0],r)
                                 for _ in range(levels if enabled else 1)])
        self.log_rho=nn.Parameter(torch.tensor(math.log(1.25-self.RHO_MIN)))
        self.log_thr=nn.Parameter(torch.tensor(math.log(0.1)))
    @property
    def rho(self): return self.RHO_MIN+F.softplus(self.log_rho)
    def fatness(self,local):
        x=local.unsqueeze(1)
        avg=F.avg_pool1d(x,5,stride=1,padding=2).squeeze(1)
        return sgate(avg/(local+1e-8)-1.0/self.kappa,tau=0.05)
    def forward(self,z):
        if not self.enabled:
            return {"energy":z.new_zeros(z.shape[0]),
                    "refine":z.new_zeros(z.shape[0],1),"scales":[z]}
        en=[]; sc=[]
        for m,p in enumerate(self.proj):
            s=p(z); e=torch.einsum("bi,ij,bj->b",s,self.L.to(s.dtype),s)
            en.append((self.rho**m)*e); sc.append(s)
        E=torch.stack(en,0).sum(0)
        local=torch.stack(sc,0).abs().mean(0)
        refine=sgate(local-torch.exp(self.log_thr),tau=0.02)*self.fatness(local)
        return {"energy":E,"refine":refine.mean(-1,keepdim=True),
                "scales":sc,"rho":self.rho}

class HodgeSplit(nn.Module):
    def __init__(self,d,d_h=None,r=32,enabled=True):
        super().__init__()
        self.enabled=enabled; self.d_h=d_h or max(d//3,8)
        self.split=LowRank(d,3*d,r)
        self.h_proj=nn.Sequential(nn.Linear(d,self.d_h),nn.Tanh())
        self.Lw=nn.Parameter(torch.randn(self.d_h,self.d_h)*0.02)
    def forward(self,z):
        if not self.enabled:
            return {"harmonic":z,"snr":z.new_ones(z.shape[0]),
                    "kernel_penalty":z.new_zeros(())}
        h,e,c=self.split(z).chunk(3,-1)
        hp=self.h_proj(h); L=0.5*(self.Lw+self.Lw.T)
        pen=(hp@L.to(hp.dtype)).pow(2).sum(-1).mean()
        he,ee,ce=h.pow(2).sum(-1),e.pow(2).sum(-1),c.pow(2).sum(-1)
        return {"harmonic":hp,"exact":e,"coexact":c,
                "snr":(he+ee)/(ce+1e-8),"kernel_penalty":pen}

class HilbertPolya(nn.Module):
    """Learned spectral map. NOT the HP conjecture."""
    def __init__(self,d,d_s=64,rank=16,r=32,enabled=True,exact=True):
        super().__init__()
        self.enabled=enabled; self.d_s=d_s
        self.to_s=LowRank(d,d_s,r)
        self.log_base=nn.Parameter(torch.linspace(0,math.log(50.0),d_s))
        self.v=nn.Sequential(LowRank(d_s,d_s,rank),nn.Tanh())
        self.coupling=nn.Parameter(torch.tensor(0.1)); self.exact=exact
    def forward(self,z):
        if not self.enabled:
            return {"eigenvalues":z,"gap":z.new_zeros(z.shape[0]),"spectral":z}
        s=self.to_s(z)
        H0=torch.diag_embed(torch.exp(self.log_base)).unsqueeze(0)
        v=self.v(s)
        V=torch.bmm(v.unsqueeze(-1),v.unsqueeze(-2))*self.coupling
        H=H0.to(V.dtype)+0.5*(V+V.transpose(-1,-2))
        if self.exact:
            n=H.shape[-1]
            ev,_=torch.linalg.eigh(H.float()+1e-6*torch.eye(n,device=H.device))
            ev=ev.to(z.dtype)
        else: ev=torch.diagonal(H,dim1=-2,dim2=-1)
        gap=(ev[:,1]-ev[:,0]).abs() if ev.shape[-1]>1 else ev.new_zeros(ev.shape[0])
        return {"eigenvalues":ev,"gap":gap,"spectral":s*torch.cos(ev*math.pi)}

class SSC(nn.Module):
    def __init__(self,eps=0.0028,enabled=True):
        super().__init__()
        self.eps=eps; self.enabled=enabled
        self.register_buffer("prev",torch.zeros(1))
        self.register_buffer("init",torch.zeros(1,dtype=torch.bool))
    def reset(self): self.prev.zero_(); self.init.zero_()
    def forward(self,x):
        if not self.enabled: return x
        v=x.detach().reshape(1)
        if not bool(self.init.item()):
            self.prev.copy_(v); self.init.fill_(True); return x
        p=self.prev.to(x.dtype).reshape(())
        out=p+self.eps*(x-p)
        self.prev.copy_(out.detach().reshape(1)); return out

class CSOC(nn.Module):
    """
    Hysteresis counter. v2: the bound is checked against the CORRECTED
    no-Zeno certificate (Thm 18.6, Doob, no horizon truncation).
    """
    def __init__(self,hi=0.15,lo=0.05,enabled=True):
        super().__init__()
        self.enabled=enabled
        for n,v in (("hi",hi),("lo",lo)):
            self.register_buffer(n,torch.tensor(v))
        for n in ("stress","events","Emax","D"):
            self.register_buffer(n,torch.zeros(1))
        self.register_buffer("active",torch.zeros(1,dtype=torch.bool))
    def reset(self):
        for b in (self.stress,self.events,self.Emax,self.D): b.zero_()
        self.active.zero_()
    def forward(self,s):
        if not self.enabled:
            z=torch.zeros((),device=s.device)
            return {"events":z,"bound":z,"satisfied":torch.ones((),device=s.device)}
        s=s.reshape(())
        exc=F.softplus(s-self.hi)
        dec=torch.sigmoid(-(self.stress.reshape(())-self.lo)*10)
        new=self.stress.reshape(())+exc-dec*0.01*self.stress.reshape(())
        self.stress.copy_(new.detach().reshape(1))
        fire=(new>self.hi)&(~self.active.reshape(()))
        clear=(new<self.lo)&self.active.reshape(())
        self.active.copy_(((self.active.reshape(())|fire)&(~clear)).reshape(1))
        self.events.add_(fire.float().reshape(1))
        self.Emax.copy_(torch.maximum(self.Emax,new.detach()).reshape(1))
        self.D.add_((exc.detach()*0.01).reshape(1))
        bound=(self.Emax+self.D)/(self.hi-self.lo+1e-8)
        ev=self.events.reshape(())
        return {"events":ev,"bound":bound.reshape(()),
                "satisfied":(ev<=bound.reshape(())).float()}
    def formal_certificate(self,delta=1.0,B0=2.0,G0=0.5) -> Dict:
        """The engineering counter bound above is NOT the theorem. This is."""
        return no_zeno_certificate(delta,B0,G0)

# -----------------------------------------------------------------------------
# §3. STATISTICS
# -----------------------------------------------------------------------------

def subsample_ci(x:np.ndarray,conf=0.95,block=32) -> Dict:
    n=len(x)
    if n<8:
        m=float(np.mean(x)) if n else 0.0
        return {"mean":m,"lo":m,"hi":m,"n":n}
    b=min(block,n//4); th=float(np.mean(x))
    sub=np.array([x[j:j+b].mean() for j in range(n-b+1)])
    sc=math.sqrt(b)*(sub-th); a=1-conf
    return {"mean":th,
            "lo":th-float(np.percentile(sc,100*(1-a/2)))/math.sqrt(n),
            "hi":th-float(np.percentile(sc,100*(a/2)))/math.sqrt(n),"n":n}

class SplitConformal:
    def __init__(self,alpha=0.1): self.alpha=alpha; self.q=None; self.n=None
    def fit(self,mu,sd,y):
        s=np.abs(y-mu)/np.maximum(sd,1e-9); n=len(s); self.n=n
        k=min(math.ceil((n+1)*(1-self.alpha)),n)
        self.q=float(np.sort(s)[k-1])
        if n<200: log.warning(f"[conformal] n={n} small; q unstable")
        return self
    def interval(self,mu,sd):
        w=self.q*sd; return mu-w,mu+w

# -----------------------------------------------------------------------------
# §4. PROTOCOL TRANSFORMER + HEADS
# -----------------------------------------------------------------------------

def dose_bin(nM,n=12):
    return 0 if nM<=0 else int(np.clip(round(np.log10(nM+1e-9)+3),0,n-1))
def time_bin(h,n=12):
    return 0 if h<=0 else int(np.clip(round(np.log2(h+1)),0,n-1))

@dataclass
class ProtocolStep:
    operation:str; factor:Optional[str]=None
    dose_nM:float=0.0; duration_h:float=0.0
    def tokens(self):
        return (OP_IDX.get(self.operation,0),
                F_IDX.get(self.factor or "OTHER",N_F-1),
                dose_bin(self.dose_nM),time_bin(self.duration_h))

class ProtocolTransformer(nn.Module):
    def __init__(self,d=128,n_heads=4,n_layers=3,ff=4,dropout=0.0,
                 max_steps=32,rollout=8):
        super().__init__()
        self.d=d; self.rollout=rollout; self.max_steps=max_steps
        self.op_emb=nn.Embedding(N_OP,d//4); self.f_emb=nn.Embedding(N_F,d//4)
        self.d_emb=nn.Embedding(12,d//4); self.t_emb=nn.Embedding(12,d//4)
        self.pos=nn.Parameter(torch.randn(1,max_steps,d)*0.02)
        layer=nn.TransformerEncoderLayer(d,n_heads,d*ff,dropout,
                                         batch_first=True,norm_first=True,
                                         activation="gelu")
        self.trunk=nn.TransformerEncoder(layer,n_layers)
        self.gru=nn.GRUCell(d,d)
        self.head_state=LowRank(d,d,32)
        self.head_conf=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),
                                     nn.Linear(d//2,1),nn.Sigmoid())
    def embed(self,t):
        e=torch.cat([self.op_emb(t[...,0]),self.f_emb(t[...,1]),
                     self.d_emb(t[...,2]),self.t_emb(t[...,3])],-1)
        return e+self.pos[:,:e.shape[1]]
    def forward(self,tokens,mask=None):
        h=self.embed(tokens)
        if mask is not None: h=h*mask.unsqueeze(-1)
        z=self.trunk(h); ctx=z.mean(1); s=ctx
        st=[]; cf=[]
        for _ in range(self.rollout):
            st.append(self.head_state(s)); cf.append(self.head_conf(s).squeeze(-1))
            s=self.gru(s,ctx)
        return {"context":ctx,"trajectory":torch.stack(st,1),
                "confidence":torch.stack(cf,1)}

class MathCore(nn.Module):
    def __init__(self,d_in,d=192,flags=None,r=32,carpet_cells=81):
        super().__init__()
        f=flags or {}
        self.flags={k:f.get(k,True) for k in
                    ("opmc","carpet","hodge","hilbert_polya","ssc","csoc")}
        self.inp=nn.Sequential(LowRank(d_in,d,r),nn.GELU())
        self.opmc=OPMC(d,320,r,enabled=self.flags["opmc"])
        self.carpet=CarpetAMR(320,carpet_cells,3,r=r,enabled=self.flags["carpet"])
        self.hodge=HodgeSplit(320,96,r,enabled=self.flags["hodge"])
        self.hp=HilbertPolya(96 if self.flags["hodge"] else 320,64,16,r,
                             enabled=self.flags["hilbert_polya"])
        d_out=64 if self.flags["hilbert_polya"] else (96 if self.flags["hodge"] else 320)
        self.out=nn.Sequential(nn.LayerNorm(d_out),LowRank(d_out,d,r),nn.GELU())
        self.ssc=SSC(enabled=self.flags["ssc"]); self.csoc=CSOC(enabled=self.flags["csoc"])
    def forward(self,x):
        h=self.inp(x); o=self.opmc(h)
        c=self.carpet(o["z"]); hd=self.hodge(o["z"]); hp=self.hp(hd["harmonic"])
        z=self.out(hp["spectral"])
        sm=self.ssc(z.abs().mean()); cs=self.csoc(sm)
        return {"z":z,"carpet_energy":c["energy"],"hodge_snr":hd["snr"],
                "kernel_penalty":hd["kernel_penalty"],"spectral_gap":hp["gap"],
                "csoc":cs,"k_star":o["k_star"]}

class T1Potency(nn.Module):
    def __init__(self,seq_dim,core,d=192):
        super().__init__()
        self.core=core
        self.seq=nn.Sequential(nn.Linear(seq_dim,d),nn.GELU(),LowRank(d,d,32))
        self.mu=nn.Sequential(nn.Linear(d,96),nn.GELU(),nn.Linear(96,1))
        self.log_sd=nn.Sequential(nn.Linear(d,96),nn.GELU(),nn.Linear(96,1))
        self.emax=nn.Sequential(nn.Linear(d,96),nn.GELU(),nn.Linear(96,1),nn.Sigmoid())
    def forward(self,sf):
        c=self.core(self.seq(sf)); z=c["z"]
        return {"log_ec50":self.mu(z).squeeze(-1),
                "log_sd":self.log_sd(z).squeeze(-1).clamp(-6,3),
                "emax":self.emax(z).squeeze(-1),**c}

class T2Fate(nn.Module):
    """Output on the simplex. The target IS an occupation measure."""
    def __init__(self,core,d=192,ctx_dim=128):
        super().__init__()
        self.core=core
        self.fuse=nn.Sequential(nn.Linear(d+ctx_dim+N_F+3,d),nn.GELU())
        self.logits=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,N_FATE))
        self.log_conc=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,occ,fidx,dose,time,ctx):
        f=F.one_hot(fidx,N_F).float()
        extra=torch.stack([occ,torch.log10(dose+1e-6),torch.log2(time+1.0)],-1)
        c=self.core(self.fuse(torch.cat([ctx,f,extra],-1))); z=c["z"]
        lg=self.logits(z)
        return {"composition":F.softmax(lg,-1),"logits":lg,
                "concentration":F.softplus(self.log_conc(z).squeeze(-1))+N_FATE,**c}

class T3Quality(nn.Module):
    def __init__(self,core,d=192,n_markers=8):
        super().__init__()
        self.core=core
        self.fuse=nn.Sequential(nn.Linear(d+N_FATE,d),nn.GELU())
        self.diam=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,1),nn.Softplus())
        self.viab=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
        self.mark=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,n_markers),nn.Sigmoid())
        self.log_sd=nn.Sequential(nn.Linear(d,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,ctx,comp):
        c=self.core(self.fuse(torch.cat([ctx,comp],-1))); z=c["z"]
        sd=self.log_sd(z).clamp(-6,3).exp()
        return {"diameter_um":self.diam(z).squeeze(-1),
                "viability":self.viab(z).squeeze(-1),"markers":self.mark(z),
                "sd_diam":sd[:,0],"sd_viab":sd[:,1],**c}

class T4EngraftmentGate(nn.Module):
    """GATED. Refuses below min_n. Uses T5 minimax when ungated."""
    MIN_N_DEFAULT=500
    def __init__(self,core,min_n=MIN_N_DEFAULT,d=192):
        super().__init__()
        self.min_n=min_n; self.n_train=0; self.core=core
        self.head=nn.Sequential(nn.Linear(d+3,64),nn.GELU(),nn.Linear(64,1))
    def set_n(self,n):
        self.n_train=n
        if n<self.min_n:
            log.warning(f"[T4] n={n} < {self.min_n} -> GATED")
    def forward(self,ctx,diam,viab,ent):
        if self.n_train<self.min_n:
            raise DataRefusal(
                f"T4 GATED: n_train={self.n_train} < {self.min_n}.\n"
                "Published mouse transplant studies do not provide enough "
                "verified records for a generalizable predictor.\n"
                "An overfit engraftment model is a clinical hazard.\n"
                "To ungate: curate >= min_n verified records with DOIs, "
                "call set_n(), and report held-out AUROC.")
        h=torch.cat([ctx,torch.stack([diam,viab,ent],-1)],-1)
        c=self.core(h)
        return {"logit":self.head(c["z"]).squeeze(-1),**c}

class MDOEChain(nn.Module):
    def __init__(self,seq_dim=128,d=192,ctx_dim=128,flags=None,
                 mc_samples=32,min_engraftment_n=500,carpet_cells=81):
        super().__init__()
        self.mc=mc_samples
        self.proto=ProtocolTransformer(d=ctx_dim)
        self.core1=MathCore(d,d,flags,carpet_cells=carpet_cells)
        self.core2=MathCore(d+ctx_dim+N_F+3,d,flags,carpet_cells=carpet_cells)
        self.core3=MathCore(d+N_FATE,d,flags,carpet_cells=carpet_cells)
        self.t1=T1Potency(seq_dim,self.core1,d)
        self.t2=T2Fate(self.core2,d,ctx_dim)
        self.t3=T3Quality(self.core3,d)
        self.t4=T4EngraftmentGate(MathCore(d+3,d,flags),min_engraftment_n,d)
    @staticmethod
    def occupancy(dose,ec50,emax):
        return emax*dose/(dose+ec50+1e-12)
    def forward(self,sf,proto,fidx,dose,time,propagate=True):
        p=self.proto(proto); ctx=p["context"]
        t1=self.t1(sf)
        lec,lsd,emx=t1["log_ec50"],t1["log_sd"],t1["emax"]
        if propagate and self.mc>1:
            eps=torch.randn(self.mc,*lec.shape,device=lec.device)
            samp=lec.unsqueeze(0)+eps*lsd.exp().unsqueeze(0)
            ec50=samp.exp().clamp(1e-6,1e9)
            occ=self.occupancy(dose.unsqueeze(0),ec50,emx.unsqueeze(0))
            rep=lambda t: t.unsqueeze(0).expand(self.mc,*t.shape)
            t2=self.t2(occ.reshape(-1),rep(fidx).reshape(-1),
                       rep(dose).reshape(-1),rep(time).reshape(-1),
                       rep(ctx).reshape(-1,ctx.shape[-1]))
            comp=t2["composition"].reshape(self.mc,*lec.shape,N_FATE)
            cmu,csd=comp.mean(0),comp.std(0)
            t3s=self.t3(rep(ctx).reshape(-1,ctx.shape[-1]),comp.reshape(-1,N_FATE))
            dmu=t3s["diameter_um"].reshape(self.mc,-1).mean(0)
            dsd=t3s["diameter_um"].reshape(self.mc,-1).std(0)
            vmu=t3s["viability"].reshape(self.mc,-1).mean(0)
            vsd=t3s["viability"].reshape(self.mc,-1).std(0)
            t3=t3s
        else:
            occ=self.occupancy(dose,lec.exp(),emx)
            t2=self.t2(occ,fidx,dose,time,ctx)
            cmu,csd=t2["composition"],torch.zeros_like(t2["composition"])
            t3=self.t3(ctx,cmu)
            dmu,dsd=t3["diameter_um"],t3["sd_diam"]
            vmu,vsd=t3["viability"],t3["sd_viab"]
        ent=-(cmu*torch.log(cmu+1e-9)).sum(-1)
        return {"t1_log_ec50":lec,"t1_sd":lsd.exp(),"t1_emax":emx,
                "occupancy":self.occupancy(dose,lec.exp(),emx),
                "t2_composition":cmu,"t2_composition_sd":csd,
                "t2_concentration":t2["concentration"],
                "t3_diameter_um":dmu,"t3_diameter_sd":dsd,
                "t3_viability":vmu,"t3_viability_sd":vsd,
                "t3_markers":t3["markers"],"fate_entropy":ent,
                "protocol_trajectory":p["trajectory"],
                "protocol_confidence":p["confidence"],
                "carpet_energy":t2["carpet_energy"],"hodge_snr":t2["hodge_snr"],
                "kernel_penalty":t2["kernel_penalty"],
                "csoc_events":t2["csoc"]["events"],
                "csoc_satisfied":t2["csoc"]["satisfied"],"context":ctx}

# -----------------------------------------------------------------------------
# §5. LOSSES + TRAINER
# -----------------------------------------------------------------------------

def simplex_kl(comp,target,conc):
    return -(target.clamp(min=1e-8)*torch.log(comp+1e-9)).sum(-1).mean() \
           + 0.01*(torch.log(conc)-1.0).pow(2).mean()

def hetero_nll(mu,log_sd,y):
    sd=log_sd.exp().clamp(1e-4,1e4)
    return (0.5*((y-mu)/sd).pow(2)+log_sd).mean()

class ChainLoss(nn.Module):
    def __init__(self,w_t1=1.0,w_t2=1.0,w_t3=0.5,w_reg=1e-6,
                 w_kernel=0.01,w_csoc=0.01):
        super().__init__()
        self.w=dict(t1=w_t1,t2=w_t2,t3=w_t3,reg=w_reg,
                    kernel=w_kernel,csoc=w_csoc)
    def forward(self,out,targets,model):
        dev=out["t1_log_ec50"].device; z=torch.zeros((),device=dev); L={}
        L["t1"]=hetero_nll(out["t1_log_ec50"],torch.log(out["t1_sd"]+1e-9),
                           targets["log_ec50"]) if "log_ec50" in targets else z
        L["t2"]=simplex_kl(out["t2_composition"],targets["composition"],
                           out["t2_concentration"]) if "composition" in targets else z
        t3=z
        if "diameter" in targets:
            t3=t3+hetero_nll(torch.log(out["t3_diameter_um"]+1),
                             torch.log(out["t3_diameter_sd"]+1e-9),
                             torch.log(targets["diameter"]+1))
        if "viability" in targets:
            t3=t3+F.mse_loss(out["t3_viability"],targets["viability"])
        L["t3"]=t3
        L["kernel"]=out["kernel_penalty"]
        L["csoc"]=out["csoc_events"]*0.01
        L["reg"]=sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
        L["total"]=sum(self.w[k]*L[k] for k in self.w)
        return L

class RunRecorder:
    def __init__(self,name,root="runs"):
        self.dir=Path(root)/name; self.dir.mkdir(parents=True,exist_ok=True)
        self.f=open(self.dir/"log.jsonl","a")
        self.best=float("inf"); self.state=None; self.t0=time.time()
    def log(self,rec):
        rec["t"]=round(time.time()-self.t0,2)
        self.f.write(json.dumps(rec)+"\n"); self.f.flush()
    def save_best(self,model,val,ep):
        if val<self.best:
            self.best=val
            self.state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({"model":self.state,"epoch":ep,"val":val},self.dir/"best.pt")
            return True
        return False
    def finalize(self,model,test,extra=None):
        if self.state is not None: model.load_state_dict(self.state)
        pay={"best_val":self.best,"test":test,
             "note":"test evaluated once, after validation selection"}
        if extra: pay.update(extra)
        json.dump(pay,open(self.dir/"test.json","w"),indent=2)
        self.f.close(); return pay

def synthetic_shape_check_only(n=64,seq_dim=128,T=8,device="cpu"):
    log.warning("="*68)
    log.warning("SYNTHETIC SHAPE CHECK — NOT DATA. Metrics are MEANINGLESS.")
    log.warning("="*68)
    g=torch.Generator(device=device).manual_seed(0)
    return {"seq_feat":torch.randn(n,seq_dim,generator=g,device=device),
      "protocol":torch.randint(0,8,(n,T,4),generator=g,device=device),
      "factor_idx":torch.randint(0,N_F,(n,),generator=g,device=device),
      "dose_nM":torch.rand(n,generator=g,device=device)*100+1,
      "time_h":torch.rand(n,generator=g,device=device)*240+1,
      "targets":{"log_ec50":torch.randn(n,generator=g,device=device),
        "composition":F.softmax(torch.randn(n,N_FATE,generator=g,device=device),-1),
        "diameter":torch.rand(n,generator=g,device=device)*500+50,
        "viability":torch.rand(n,generator=g,device=device)}}

def batch_from_curation(recs,stage,device="cpu"):
    use=[r for r in recs if getattr(r,f"has_{stage}")()]
    if len(use)<8:
        raise DataRefusal(f"stage {stage}: only {len(use)} verified; need >= 8")
    n=len(use); seq_dim=128
    seq=torch.zeros(n,seq_dim)
    for i,r in enumerate(use):
        for j,a in enumerate(r.ligand_sequence[:seq_dim]):
            if a in AA_IDX: seq[i,j]=AA_IDX[a]+1
        seq[i]/=21.0
    proto=torch.zeros(n,4,4,dtype=torch.long)
    for i,r in enumerate(use):
        proto[i,0]=torch.tensor(ProtocolStep("ADD",r.factor or "OTHER",
            r.dose_nM or 0.0,r.time_h or 0.0).tokens())
    out={"seq_feat":seq.to(device),"protocol":proto.to(device),
      "factor_idx":torch.tensor([F_IDX.get(r.factor or "OTHER",N_F-1)
                                 for r in use]).to(device),
      "dose_nM":torch.tensor([r.dose_nM or 1.0 for r in use]).to(device),
      "time_h":torch.tensor([r.time_h or 1.0 for r in use]).to(device)}
    tg={}
    if stage=="t1":
        tg["log_ec50"]=torch.tensor([math.log(r.ec50_nM) for r in use]).to(device)
    if stage=="t2":
        tg["composition"]=torch.tensor(np.array([r.fate_composition
                                                 for r in use],np.float32)).to(device)
        w2=[r.t2_uncertainty()["W2_bound"] if r.t2_uncertainty() else np.nan
            for r in use]
        out["W2_bound"]=torch.tensor(w2).to(device)
    if stage=="t3":
        tg["diameter"]=torch.tensor([r.organoid_diameter_um for r in use]).to(device)
        tg["viability"]=torch.tensor([r.viability_frac for r in use]).to(device)
    out["targets"]=tg; out["n_verified"]=n
    return out

# -----------------------------------------------------------------------------
# §6. ABLATION — this is the result
# -----------------------------------------------------------------------------

ABLATION_ARMS={
 "none":dict(opmc=False,carpet=False,hodge=False,hilbert_polya=False,ssc=False,csoc=False),
 "opmc_only":dict(opmc=True,carpet=False,hodge=False,hilbert_polya=False,ssc=False,csoc=False),
 "carpet_only":dict(opmc=False,carpet=True,hodge=False,hilbert_polya=False,ssc=False,csoc=False),
 "hodge_only":dict(opmc=False,carpet=False,hodge=True,hilbert_polya=False,ssc=False,csoc=False),
 "hp_only":dict(opmc=False,carpet=False,hodge=False,hilbert_polya=True,ssc=False,csoc=False),
 "full":dict(opmc=True,carpet=True,hodge=True,hilbert_polya=True,ssc=True,csoc=True),
}

def run_ablation(data,epochs=40,seed=42,out="runs/ablation",device="cpu",mc=8):
    n=data["seq_feat"].shape[0]
    g=np.random.default_rng(seed); idx=g.permutation(n)
    ntr=max(4,int(0.7*n)); nva=max(2,int(0.15*n))
    tr,va,te=idx[:ntr],idx[ntr:ntr+nva],idx[ntr+nva:]
    if len(te)<2: te=va
    res={}
    for arm,flags in ABLATION_ARMS.items():
        torch.manual_seed(seed)
        model=MDOEChain(flags=flags,mc_samples=mc).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
        lf=ChainLoss(); rec=RunRecorder(f"ablation_{arm}",out)
        sel=lambda t,i: t[i]
        for ep in range(epochs):
            model.train(); opt.zero_grad(set_to_none=True)
            o=model(sel(data["seq_feat"],tr),sel(data["protocol"],tr),
                    sel(data["factor_idx"],tr),sel(data["dose_nM"],tr),
                    sel(data["time_h"],tr))
            L=lf(o,{k:sel(v,tr) for k,v in data["targets"].items()},model)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if (ep+1)%max(1,epochs//5)==0:
                model.eval()
                with torch.no_grad():
                    ov=model(sel(data["seq_feat"],va),sel(data["protocol"],va),
                             sel(data["factor_idx"],va),sel(data["dose_nM"],va),
                             sel(data["time_h"],va))
                    Lv=lf(ov,{k:sel(v,va) for k,v in data["targets"].items()},model)
                imp=rec.save_best(model,float(Lv["total"]),ep)
                rec.log({"epoch":ep+1,"train":float(L["total"]),
                         "val":float(Lv["total"]),"best":imp})
        model.eval()
        with torch.no_grad():
            ot=model(sel(data["seq_feat"],te),sel(data["protocol"],te),
                     sel(data["factor_idx"],te),sel(data["dose_nM"],te),
                     sel(data["time_h"],te))
            Lt=lf(ot,{k:sel(v,te) for k,v in data["targets"].items()},model)
        npar=sum(p.numel() for p in model.parameters())
        res[arm]={"test_total":float(Lt["total"]),"test_t1":float(Lt["t1"]),
                  "test_t2":float(Lt["t2"]),"n_params":npar,"flags":flags}
        rec.finalize(model,res[arm],{"honesty":HONESTY})
        log.info(f"[ablation:{arm:12s}] test={float(Lt['total']):.4f} params={npar:,}")
    base=res["none"]["test_total"]
    print("\n"+"="*78)
    print("  ABLATION — does each component actually help?")
    print("="*78)
    print(f"  {'arm':<14}{'test':>10}{'Δ vs none':>12}{'params':>12}  verdict")
    for arm,r in res.items():
        d=base-r["test_total"]
        v="HELPS" if d>0.02 else ("no effect" if abs(d)<=0.02 else "HURTS")
        print(f"  {arm:<14}{r['test_total']:>10.4f}{d:>+12.4f}{r['n_params']:>12,}  {v}")
    print("\n  With small n these differences may be NOISE.")
    print("     Repeat with >= 3 seeds before believing any verdict.")
    print("     Keep a component only if it helps on held-out data.")
    print("="*78)
    Path(out).mkdir(parents=True,exist_ok=True)
    json.dump(res,open(Path(out)/"ablation.json","w"),indent=2)
    return res

def aggregate_seeds(seed_dirs:Sequence[str]) -> Dict:
    """NEW: aggregate across seeds and report whether a verdict survives."""
    rows={}
    for d in seed_dirs:
        p=Path(d)/"ablation.json"
        if not p.exists(): continue
        r=json.loads(p.read_text())
        for arm,v in r.items():
            rows.setdefault(arm,[]).append(v["test_total"])
    out={}
    for arm,vals in rows.items():
        a=np.array(vals)
        out[arm]={"n_seeds":len(a),"mean":float(a.mean()),
                  "sd":float(a.std()),"min":float(a.min()),"max":float(a.max())}
    if "none" in out:
        base=out["none"]["mean"]
        for arm in out:
            d=base-out[arm]["mean"]
            pooled=math.sqrt((out[arm]["sd"]**2+out["none"]["sd"]**2)/2)+1e-9
            out[arm]["delta_vs_none"]=d
            out[arm]["effect_in_sd_units"]=d/pooled
            out[arm]["verdict"]=("HELPS" if d/pooled>1.0 else
                                 "HURTS" if d/pooled<-1.0 else "INCONCLUSIVE")
    return {"per_arm":out,"n_seed_dirs":len(seed_dirs),
            "NOTE":"a verdict needs |effect| > 1 pooled sd across >= 3 seeds"}

# -----------------------------------------------------------------------------
# §7. SELF-TEST
# -----------------------------------------------------------------------------

def selftest(a=None) -> bool:
    print("="*78); print("  MDOE v2 — self-test"); print("="*78)
    torch.manual_seed(0); ok=True
    dev="cuda" if torch.cuda.is_available() else "cpu"
    sd=SDGate()

    print("\n[T1] corrected honesty registry")
    print(f"   OPMC -> {HONESTY['opmc']['named_after']}")
    print(f"   CSOC -> {HONESTY['csoc']['named_after']}")
    ok &= "7.5" in HONESTY["opmc"]["named_after"]
    ok &= "18.6" in HONESTY["csoc"]["named_after"]
    ok &= HONESTY["composition"]["genuine"].startswith("this is a literal")
    print(f"   composition: {HONESTY['composition']['genuine']}")

    print("\n[T2] OPMC k* and shape")
    o=OPMC(64,320); r=o(torch.randn(3,64))
    print(f"   z={tuple(r['z'].shape)} k*={r['k_star']}")
    ok &= r["z"].shape==(3,320) and r["k_star"]==23

    print("\n[T3] carpet Laplacian valid + non-PCF status")
    L=carpet_laplacian(81)
    rs=L.sum(1).abs().max().item(); sym=(L-L.T).abs().max().item()
    ev=torch.linalg.eigvalsh(L)
    ok &= rs<1e-5 and sym<1e-6 and ev.min().item()>-1e-5
    cs=carpet_status(False)
    print(f"   imports needed: {len(cs['inputs_needed'])}")
    ok &= len(cs["inputs_needed"])==3

    print("\n[T4] CSOC counter + FORMAL corrected certificate")
    c=CSOC()
    for v in np.linspace(0,0.5,200): cr=c(torch.tensor(float(v)))
    print(f"   events={cr['events'].item():.0f} bound={cr['bound'].item():.2f} "
          f"satisfied={bool(cr['satisfied'].item())}")
    ok &= bool(cr["satisfied"].item())
    fc=c.formal_certificate()
    print(f"   formal: eps0={fc['eps0']:.4f} q0={fc['q0']:.4f} "
          f"truncation={fc['HORIZON_TRUNCATION']}")
    ok &= fc["certified"] and "REMOVED" in fc["HORIZON_TRUNCATION"]

    print("\n[T5] composition on simplex + Thm 8.3 rate")
    d=synthetic_shape_check_only(n=16,device=dev)
    m=MDOEChain(mc_samples=4).to(dev)
    out=m(d["seq_feat"],d["protocol"],d["factor_idx"],d["dose_nM"],d["time_h"])
    s=out["t2_composition"].sum(-1)
    ok &= bool(((s-1).abs()<1e-4).all())
    ra=theorem_A_rate(1000,N_FATE)
    print(f"   sum range [{s.min():.6f},{s.max():.6f}]  W2<={ra['W2_bound']:.4f}")
    ok &= ra["W2_bound"]>0

    print("\n[T6] uncertainty propagates")
    ok &= bool((out["t2_composition_sd"].sum()>0).item() and
               (out["t3_diameter_sd"]>0).all().item())
    print(f"   comp_sd sum={out['t2_composition_sd'].sum():.4f}")

    print("\n[T7] T4 GATED")
    m.t4.set_n(50); b=False
    try: m.t4(out["context"],out["t3_diameter_um"],out["t3_viability"],
              out["fate_entropy"])
    except DataRefusal: b=True
    print(f"   refused={b}"); ok &= b

    print("\n[T8] gradient flow")
    L=ChainLoss()(out,d["targets"],m); L["total"].backward()
    g=[p.grad for p in m.parameters() if p.grad is not None]
    ok &= len(g)>20 and all(torch.isfinite(x).all() for x in g)
    print(f"   finite grads on {len(g)} tensors")

    print("\n[T9] NEW: response-tail moment gate")
    r1=KouRegime("line_A",eta_up=3.0,eta_dn=4.0,lam=1.0)
    r2=KouRegime("hyper",eta_up=0.7,eta_dn=5.0,lam=0.5,occupation=0.05)
    ok &= moment_estimable([r1],1.0)["finite"]
    ok &= not moment_estimable([r2],1.0)["finite"]
    hv=heaviest_regime_dominates([r1,r2],T=60.0)
    print(f"   eta+<1 mean infinite; tail governed by {hv['governing_regimes']}")
    ok &= hv["governing_regimes"]==["hyper"]
    wc=esscher_worst_case(r1,T=10.0,eta_KL=0.5)
    print(f"   Esscher worst-case shift={wc['shift']:.4f}")
    ok &= wc["shift"]>0

    print("\n[T10] NEW: corrected morphology surgery")
    ev2=classify_morphology_event(0,2,0,-2)
    print(f"   lumen handle Db1=+2 -> {ev2['candidates']}")
    ok &= ev2["candidates"]==["H"]
    ok &= classify_morphology_event(0,1,0,-1)["candidates"]==[]
    print("   filament +-1 rejected for a closed surface")

    print("\n[T11] loaders REFUSE rather than fabricate")
    b=False
    try: load_curation(Path("__nonexistent__.jsonl"))
    except DataRefusal: b=True
    print(f"   missing curation refused={b}"); ok &= b

    print("\n[T12] conformal coverage")
    rng=np.random.default_rng(0)
    mu=rng.normal(0,1,600); s=np.exp(rng.normal(-1,.3,600))
    y=mu+s*rng.normal(0,1,600)
    cf=SplitConformal(0.1).fit(mu[:400],s[:400],y[:400])
    lo,hi=cf.interval(mu[400:],s[400:])
    cov=float(((y[400:]>=lo)&(y[400:]<=hi)).mean())
    print(f"   coverage={cov:.3f} (nominal 0.90)"); ok &= cov>0.83

    print("\n[T13] ablation arms run")
    res=run_ablation(d,epochs=6,mc=4,out="runs/_mdoe_v2_selftest",device=dev)
    ok &= len(res)==len(ABLATION_ARMS)
    import shutil; shutil.rmtree("runs/_mdoe_v2_selftest",ignore_errors=True)

    print("\n"+"="*78)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*78)
    print("""
VERIFIED: corrected registry, shapes, simplex constraint, uncertainty
propagation, CSOC counter AND its formal corrected certificate, T4 gate,
gradient flow, refusal-to-fabricate, conformal coverage, moment gate,
corrected surgery table, all ablation arms.

NOT VERIFIED: that this predicts anything real.
  No public paired T1->T2->T3->T4 dataset exists. Curate it.
  The synthetic ablation is a CODE TEST, not a result.
  Run >= 3 seeds and use aggregate_seeds() before any claim.
  Keep a math component only if it helps on held-out data.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","honesty","ablation-synthetic",
                                   "train-curated","aggregate","regime","tail"])
    ap.add_argument("--curation",default="data/curation.jsonl")
    ap.add_argument("--stage",choices=["t1","t2","t3"],default="t2")
    ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--mc",type=int,default=8)
    ap.add_argument("--seeds",type=int,default=1)
    ap.add_argument("--seed-dirs",nargs="*",default=[])
    ap.add_argument("--min-engraftment-n",dest="min_n",type=int,default=500)
    ap.add_argument("--out",default="runs/mdoe_v2")
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if a.cmd=="honesty":
        print(json.dumps({"components":HONESTY,
                          "carpet_status":carpet_status(False),
                          "no_zeno":no_zeno_certificate(1.0,2.0,0.5)},
                         indent=2,ensure_ascii=False)); return
    if a.cmd=="aggregate":
        print(json.dumps(aggregate_seeds(a.seed_dirs),indent=2)); return
    if a.cmd=="regime":
        recs=load_curation(Path(a.curation))
        out={r.record_id:r.t2_regime() for r in recs if r.t2_regime()}
        print(json.dumps(out,indent=2,ensure_ascii=False)); return
    if a.cmd=="tail":
        recs=load_curation(Path(a.curation))
        regs=[]
        for r in recs:
            if r.has_t1() and r.ec50_nM:
                regs.append(KouRegime(r.record_id,eta_up=3.0,eta_dn=4.0,
                                      lam=1.0,occupation=1.0/max(len(recs),1)))
        if not regs: print("no T1 records"); return
        print(json.dumps({"moment_p1":moment_estimable(regs,1.0),
                          "governance":heaviest_regime_dominates(regs,T=100.0)},
                         indent=2)); return
    if a.cmd=="ablation-synthetic":
        d=synthetic_shape_check_only(device=dev)
        dirs=[]
        for s in range(a.seeds):
            od=f"{a.out}/seed{42+s}"; dirs.append(od)
            run_ablation(d,a.epochs,seed=42+s,out=od,device=dev,mc=a.mc)
        if a.seeds>=3:
            print(json.dumps(aggregate_seeds(dirs),indent=2))
        else:
            print("\nSYNTHETIC. These numbers are a CODE TEST, not a finding.")
        return
    recs=load_curation(Path(a.curation))
    d=batch_from_curation(recs,a.stage,dev)
    log.info(f"[train] stage={a.stage} n_verified={d['n_verified']}")
    if "W2_bound" in d:
        log.info(f"[train] Thm 8.3 W2 bounds present: "
                 f"{torch.isfinite(d['W2_bound']).sum().item()} records")
    dirs=[]
    for s in range(a.seeds):
        od=f"{a.out}/seed{42+s}"; dirs.append(od)
        run_ablation(d,a.epochs,seed=42+s,out=od,device=dev,mc=a.mc)
    if a.seeds>=3:
        print(json.dumps(aggregate_seeds(dirs),indent=2))
    else:
        log.warning("Use --seeds 3 or more before drawing any conclusion.")

if __name__=="__main__": main()
