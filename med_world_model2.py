#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
med_world_model.py — v2.0.0
Differentiable multiscale medical world model, now on the corrected math.

BUGS FIXED vs v1
  1. FATAL: EnsembleUncertainty set torch.manual_seed(i) but the model is
     DETERMINISTIC given (state, intervention), so every member was
     IDENTICAL -> std = 0 and every CI collapsed. v2 uses parameter
     perturbation and VERIFIES member spread before reporting any CI.
  2. run_research_scenario computed uncertainty from the INITIAL state
     while reporting the FINAL state. v2 uses the final state and says so.

MATHEMATICS PROPAGATED
  cell-fate composition IS an occupation measure -> Thm 8.3 W2 rate and
  the light / super-dominant / infinite-mean trichotomy (ordered only).
  BN check on the lineage taxonomy before pooling levels (K1).
  Kou moment-explosion gate before reporting any mean risk (Prop 16.14).
  Tumorigenicity arithmetic replaces a heuristic score alone.
  SD gate on every reported statistic; Mode E when unverified.

PHILOSOPHY: see msps_med_math_core.PHILOSOPHY (full text).
NOT a medical device. NO gene-editor design, wet-lab protocols, pathogen
enhancement, or autonomous clinical orders. Every output research_only.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, math, time, uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from msps_med_math_core import (
    SDGate, Refusal, ScopeRefusal, GovernanceRefusal, EnsembleIntegrityError,
    MODE_E, MODE_CERTIFIED, RESEARCH_ONLY, Governance, IRBRecord, PHIGuard,
    refuse_disallowed, classify_intent, composition_as_occupation,
    theorem_A_rate, occupation_regime_test, boundary_null_check,
    KouRegime, moment_estimable, require_finite_mean, response_summary,
    heaviest_regime_dominates, TumorigenicitySpec, engraftment_power,
    detectable_effect, TRANSLATION_GAP, classify_morphology,
    no_zeno_certificate, gibbs_conductance, verify_gibbs_symmetry,
    carpet_status, get_logger,
)
log=get_logger("med_world_v2"); VERSION="2.0.0"

LINEAGE=["iPSC","mesoderm","ectoderm","endoderm","progenitor","mature",
         "senescent","transformed"]
LINEAGE_HIERARCHY={"GERM_LAYER":["mesoderm","ectoderm","endoderm"],
                   "DIFFERENTIATED":["progenitor","mature"],
                   "ABERRANT":["senescent","transformed"]}

class Organ(str,Enum):
    BRAIN="brain"; HEART="heart"; LUNG="lung"; LIVER="liver"
    KIDNEY="kidney"; PANCREAS="pancreas"; VASCULAR="vascular"

class InterventionClass(str,Enum):
    OBSERVATION="observation"
    GROWTH_FACTOR_SCREEN="growth_factor_screen"
    SMALL_MOLECULE_SCREEN="small_molecule_screen"
    MECHANICAL_STIMULATION="mechanical_stimulation"
    DIFFERENTIATION_REVIEW="differentiation_review"
    SURGICAL_PLANNING_SIM="surgical_planning_sim"

# =============================================================================
# §1. KNOWLEDGE GRAPH (evidence, not causation)
# =============================================================================
EVIDENCE_WEIGHT={"curated":1.0,"clinical_review":.95,"gwas":.75,
                 "single_cell":.65,"animal":.55,"in_vitro":.45,
                 "text_mined":.30,"inferred":.25}

@dataclass
class EvidenceRecord:
    gene:str; variant:Optional[str]; phenotype:str; organ:Optional[str]
    effect:str; evidence_level:str; confidence:float; source:str
    developmental_stage:Optional[str]=None
    tissue_specificity:Optional[str]=None
    contradictions:List[str]=field(default_factory=list)
    clinical_actionability:str=RESEARCH_ONLY
    def to_dict(self): return asdict(self)

class KnowledgeGraph:
    def __init__(self): self.records=[]; self.unavailable=[]
    def add(self,r): self.records.append(r)
    def note_unavailable(self,src,reason):
        self.unavailable.append({"source":src,"reason":reason})
    def query(self,gene=None,phenotype=None,organ=None):
        out=self.records
        if gene: out=[r for r in out if r.gene==gene]
        if phenotype: out=[r for r in out if phenotype.lower() in r.phenotype.lower()]
        if organ: out=[r for r in out if r.organ==organ]
        return out
    def detect_contradictions(self):
        n=0
        for i,a in enumerate(self.records):
            for b in self.records[i+1:]:
                if (a.gene==b.gene and a.phenotype.lower()==b.phenotype.lower()
                    and {a.effect,b.effect}=={"risk_increase","benign"}):
                    m1=f"{b.source} reports {b.effect}"
                    m2=f"{a.source} reports {a.effect}"
                    if m1 not in a.contradictions: a.contradictions.append(m1); n+=1
                    if m2 not in b.contradictions: b.contradictions.append(m2); n+=1
        return n
    def confidence(self,recs):
        if not recs: return 0.0
        num=sum(EVIDENCE_WEIGHT.get(r.evidence_level,.3)*r.confidence for r in recs)
        den=sum(EVIDENCE_WEIGHT.get(r.evidence_level,.3) for r in recs)
        base=num/max(den,1e-12)
        pen=.10*sum(1 for r in recs if r.contradictions)
        return float(np.clip(base-pen,0.,1.))
    def report(self,gene=None,phenotype=None,organ=None):
        recs=self.query(gene,phenotype,organ)
        return {"records":[r.to_dict() for r in recs],"n_records":len(recs),
                "aggregate_confidence":self.confidence(recs),
                "confidence_semantics":"weighted evidence score, NOT P(pathogenic)",
                "n_contradictions":sum(len(r.contradictions) for r in recs),
                "sources_unavailable":self.unavailable,
                "clinical_actionability":RESEARCH_ONLY}
    @classmethod
    def demo(cls):
        kg=cls()
        for g,v,ph,o,e,l,c,s in (
          ("GENE_A","rs1","congenital_heart_defect","heart","risk_increase","curated",.9,"PMID:000001"),
          ("GENE_A","rs1","congenital_heart_defect","heart","benign","curated",.8,"PMID:000002"),
          ("GENE_A","rs2","congenital_heart_defect","heart","associated","gwas",.7,"PMID:000003"),
          ("GENE_B",None,"neural_tube_defect","brain","risk_increase","animal",.6,"PMID:000004")):
            kg.add(EvidenceRecord(g,v,ph,o,e,l,c,s))
        kg.note_unavailable("OMIM","requires licensed OMIM_API_KEY")
        kg.note_unavailable("DisGeNET","requires DISGENET_API_KEY")
        return kg

# =============================================================================
# §2. ONTOGENETIC WORLD MODEL + occupation diagnostics
# =============================================================================
class SSC(nn.Module):
    def __init__(self,eps=0.0028):
        super().__init__()
        self.eps=float(eps)
        self.register_buffer("prev",torch.zeros((),dtype=torch.float64))
        self.register_buffer("ready",torch.zeros((),dtype=torch.bool))
    def forward(self,x):
        s=x.reshape(()).to(torch.float64)
        if not bool(self.ready):
            self.prev=s.detach().clone(); self.ready.fill_(True); return s
        out=(1-self.eps)*self.prev+self.eps*s
        self.prev=out.detach().clone(); return out
    def reset(self): self.prev.zero_(); self.ready.zero_()

class CSOC(nn.Module):
    """Engineering counter bound. The THEOREM is no_zeno_certificate (K8)."""
    def __init__(self,hi=.20,lo=.06,rate=.05):
        super().__init__()
        assert hi>lo
        self.hi,self.lo,self.gap,self.rate=hi,lo,hi-lo,rate
        for n in ("acc","events","E_max","D"):
            self.register_buffer(n,torch.zeros((),dtype=torch.float64))
        self.register_buffer("armed",torch.ones((),dtype=torch.bool))
    def forward(self,signal):
        s=signal.reshape(()).to(torch.float64)
        ex=F.softplus(s-self.hi)
        dec=torch.sigmoid(-(self.acc-self.lo)*10.)
        rem=dec*self.rate*self.acc
        acc=self.acc+ex.detach()-rem
        fire=self.armed&(acc>self.hi); clear=(~self.armed)&(acc<self.lo)
        self.acc=acc.detach()
        self.armed=((self.armed&~fire)|clear).detach()
        self.events=(self.events+fire.to(torch.float64)).detach()
        self.E_max=torch.maximum(self.E_max,acc).detach()
        self.D=(self.D+rem).detach()
        bound=(self.E_max+self.D)/self.gap
        return {"gate":torch.sigmoid((acc-self.hi)*10.),
                "events":self.events,"bound":bound,
                "satisfied":self.events<=bound+1e-9}
    def formal_certificate(self,delta=.05,B0=2.,G0=.5):
        return no_zeno_certificate(delta,B0,G0)

class OntogeneticWorldModel(nn.Module):
    def __init__(self,n_fates=len(LINEAGE),obs_dim=8,hidden=64,device="cpu"):
        super().__init__()
        self.n_fates=n_fates; self.obs_dim=obs_dim
        self.state_dim=n_fates+4
        self.net=nn.Sequential(
            nn.Linear(self.state_dim+obs_dim,hidden),nn.GELU(),
            nn.Linear(hidden,hidden),nn.GELU(),
            nn.Linear(hidden,self.state_dim))
        self.ssc=SSC(); self.csoc=CSOC()
        self.to(device)
    def initial(self,batch=1,device="cpu"):
        f=torch.zeros(batch,self.n_fates,dtype=torch.float64,device=device)
        f[:,0]=1.
        r=torch.full((batch,4),.05,dtype=torch.float64,device=device)
        return torch.cat([f,r],-1)
    def intervention_embedding(self,cls,batch=1,device="cpu"):
        classes=list(InterventionClass)
        emb=torch.zeros(batch,8,device=device)
        emb[:,classes.index(cls)%8]=1.
        return emb
    def forward(self,state,obs,dt=.1):
        x=torch.cat([state,obs],-1)
        new=state+dt*self.net(x)
        fates=F.softmax(new[...,:self.n_fates],-1)
        risks=torch.sigmoid(new[...,self.n_fates:])
        s=torch.cat([fates,risks],-1)
        filt=self.ssc((s-state).abs().mean())
        return {"state":s,"fates":fates,"risks":risks,
                "ssc_signal":filt,"csoc":self.csoc(filt)}
    def composition_diagnostics(self,fates,ordered_labels=None):
        """NEW: composition IS an occupation measure."""
        p=fates.detach().cpu().numpy().reshape(-1)
        out={"composition":p.tolist(),"on_simplex":abs(p.sum()-1)<1e-6,
             "n_fates":len(p),
             "NOTE":("a softmax state is a MEAN composition, not an empirical "
                     "occupation of n cells; the Thm 8.3 rate needs n_cells")}
        bn=boundary_null_check(LINEAGE_HIERARCHY,
                               {**dict(zip(LINEAGE,p)),
                                **{k:sum(p[LINEAGE.index(c)] for c in v)
                                   for k,v in LINEAGE_HIERARCHY.items()}})
        out["taxonomy_BN"]=bn
        if ordered_labels is not None:
            out["regime"]=occupation_regime_test(ordered_labels,LINEAGE,ordered=True)
        else:
            out["regime"]=occupation_regime_test([],LINEAGE,ordered=False)
        return out

# =============================================================================
# §3. ENSEMBLE UQ — the identical-member bug FIXED
# =============================================================================
def subsample_ci(x,conf=.95,block=None):
    x=np.asarray(x,float).reshape(-1); n=len(x)
    if n<8:
        m=float(x.mean()) if n else 0.
        return {"mean":m,"lo":m,"hi":m,"n":n}
    b=block or min(max(2,int(n**(1/3))),n//2)
    th=float(x.mean())
    sub=np.array([x[i:i+b].mean() for i in range(n-b+1)])
    sc=math.sqrt(b)*(sub-th); a=1-conf
    lo=th-float(np.quantile(sc,1-a/2))/math.sqrt(n)
    hi=th-float(np.quantile(sc,a/2))/math.sqrt(n)
    return {"mean":th,"lo":min(lo,hi),"hi":max(lo,hi),"n":n,"block":b}

class EnsembleUncertainty:
    """
    FIX: parameter perturbation + VERIFIED member spread.
    A seed alone does not perturb a deterministic model.
    """
    def __init__(self,model,n_ensemble=8,perturb_scale=0.05,sd=None):
        self.model=model; self.n=n_ensemble; self.scale=perturb_scale
        self.sd=sd or SDGate()
        self._members=None
    def build_members(self,seed=0):
        import copy
        g=torch.Generator().manual_seed(seed)
        self._members=[]
        for _ in range(self.n):
            m=copy.deepcopy(self.model)
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad and p.numel()>0:
                        p.add_(torch.randn(p.shape,generator=g,
                                           dtype=p.dtype)*self.scale)
            m.eval(); self._members.append(m)
        return self._members
    def predict(self,state,obs,steps=10,dt=.1,seed=0):
        from msps_med_math_core import verify_ensemble_spread
        if self._members is None: self.build_members(seed)
        trajs=[]
        for m in self._members:
            s=state.clone(); tr=[]
            for _ in range(steps):
                s=m(s,obs,dt)["state"]; tr.append(s.detach().cpu().numpy())
            trajs.append(np.array(tr))
        T=np.array(trajs)                        # (E,T,B,D)
        try:
            chk=verify_ensemble_spread(T)
            spread_ok=True
        except EnsembleIntegrityError as e:
            chk={"members_distinct":False,"reason":str(e)}; spread_ok=False
        cert=self.sd.certify("ensemble_uq",("SPREAD",) if spread_ok else (),chk)
        final=T[:,-1,:,:]
        ci={}
        if spread_ok:
            for d in range(final.shape[-1]):
                ci[f"dim_{d}"]=subsample_ci(final[:,:,d].reshape(-1))
        return {"mean_trajectory":T.mean(0).tolist(),
                "std_trajectory":T.std(0).tolist(),
                "final_ci":ci,"n_ensemble":self.n,
                "member_check":chk,"sd":cert,
                "method":"parameter-perturbation ensemble + subsampling",
                "assumptions":"stationarity + mixing for the CI; NOT assumption-free",
                "status":MODE_CERTIFIED if spread_ok else MODE_E}

# =============================================================================
# §4. RISK — moment gate + tumorigenicity arithmetic
# =============================================================================
class OrganoidRiskModel:
    EVIDENCE_REQUIRED=(
      "residual undifferentiated iPSC assay (ddPCR LIN28A / flow TRA-1-60)",
      "karyotype / genomic stability","tumorigenicity in vivo spiking study",
      "immune matching + rejection model","vascular perfusion demonstration",
      "functional maturation benchmark vs adult tissue",
      "GMP batch consistency record","long-term stability follow-up",
      "GLP toxicology","IRB approval","regulator pathway identified")
    def __init__(self,kg,sd=None): self.kg=kg; self.sd=sd or SDGate()
    def score(self,state,organ,evidence_present=None,
              tumor_spec:Optional[TumorigenicitySpec]=None,
              response_regimes:Optional[List[KouRegime]]=None):
        s=state.detach().cpu().numpy().reshape(-1)
        nf=len(s)-4
        mat,vas,imm,tum=[float(x) for x in s[-4:]]
        recs=self.kg.query(organ=organ)
        conf=self.kg.confidence(recs); ev_risk=1.-conf
        risks={
          "tumorigenicity_residual_ipsc":float(np.clip(.6*tum+.3*ev_risk+.1*(1-mat),0,1)),
          "immune_rejection":float(np.clip(.5*imm+.3*ev_risk+.2*(1-vas),0,1)),
          "maturation_failure":float(np.clip(.8*(1-mat)+.2*ev_risk,0,1)),
          "vascularization_failure":float(np.clip(.8*(1-vas)+.2*ev_risk,0,1)),
          "innervation_failure":float(np.clip(.5*(1-mat)+.5*ev_risk,0,1)),
          "mechanical_integration_failure":float(np.clip(.4*(1-mat)+.3*(1-vas)+.3*ev_risk,0,1)),
          "gmp_manufacturing_variability":float(np.clip(.5*ev_risk+.5*(1-mat),0,1)),
          "long_term_instability":float(np.clip(.5*tum+.5*ev_risk,0,1))}
        present=set(evidence_present or [])
        missing=[e for e in self.EVIDENCE_REQUIRED if e not in present]
        out={"risk_scores":risks,
             "overall_research_risk":float(np.mean(list(risks.values()))),
             "evidence_confidence":conf,"evidence_present":sorted(present),
             "missing_evidence":missing,
             "evidence_completeness":1.-len(missing)/len(self.EVIDENCE_REQUIRED),
             "risk_semantics":"heuristic research score, NOT a probability",
             "decision":"NO_GO_NO_GO_ISSUED",
             "explicitly_not_provided":["transplantable","approved","go/no-go",
                                        "clinical recommendation"],
             "clinical_actionability":RESEARCH_ONLY}
        if tumor_spec is not None:
            out["tumorigenicity_arithmetic"]={
                "required":tumor_spec.required_limit(),
                "assay":tumor_spec.assay_adequacy()}
        if response_regimes:
            me=moment_estimable(response_regimes,1.0)
            out["response_moment_gate"]=me
            if me["finite"]:
                out["tail_governance"]=heaviest_regime_dominates(response_regimes)
            else:
                out["REFUSED_MEAN"]=me["REFUSED"]
        out["translation_gap"]=TRANSLATION_GAP
        return out

# =============================================================================
# §5. ORCHESTRATOR
# =============================================================================
class MedWorldModel:
    def __init__(self,kg=None,irb=None,device="cpu",out_dir="runs/med_world_v2"):
        self.out=Path(out_dir); self.out.mkdir(parents=True,exist_ok=True)
        self.gov=Governance(irb); self.sd=SDGate()
        self.kg=kg or KnowledgeGraph.demo()
        self.onto=OntogeneticWorldModel(device=device)
        self.risk=OrganoidRiskModel(self.kg,self.sd)
        self.uq=EnsembleUncertainty(self.onto,8,0.05,self.sd)
        self.device=device
    def run_scenario(self,organ:str,intervention:InterventionClass,
                     question:str,steps=10,uses_patient_data=False,
                     deid=None,evidence_present=None,
                     tumor_spec=None,response_regimes=None,
                     ordered_labels=None):
        d=self.gov.authorize(question,uses_patient_data,deid,"simulation")
        if not d["authorized"]:
            return {"status":"BLOCKED","decision":d}
        st=self.onto.initial(1,self.device)
        obs=torch.rand(1,self.onto.obs_dim,dtype=torch.float64,device=self.device)
        s=st; traj=[]
        for _ in range(steps):
            o=self.onto(s,obs); s=o["state"]
            traj.append({"fates":dict(zip(LINEAGE,[float(x) for x in o["fates"][0]])),
                         "risks":[float(x) for x in o["risks"][0]],
                         "csoc_events":int(o["csoc"]["events"])})
        comp=self.onto.composition_diagnostics(o["fates"],ordered_labels)
        risk=self.risk.score(s,organ,evidence_present,tumor_spec,response_regimes)
        # FIX 2: uncertainty from the FINAL state, explicitly declared
        unc=self.uq.predict(s,obs,steps=steps)
        self.sd.certify("taxonomy_pooling",
                        ("BN",) if comp["taxonomy_BN"]["BN_satisfied"] else (),
                        comp["taxonomy_BN"])
        rep={"scenario_id":str(uuid.uuid4())[:8],"organ":organ,
             "intervention":intervention.value,"trajectory":traj,
             "composition":comp,"risk":risk,"uncertainty":unc,
             "uncertainty_reference_state":"FINAL state (v1 used the initial)",
             "governance":d,"sd_report":self.sd.report(),
             "clinical_actionability":RESEARCH_ONLY,
             "clinician_review_required":True}
        (self.out/f"{rep['scenario_id']}.json").write_text(
            json.dumps(rep,indent=2,default=str))
        return rep
    def status(self):
        return {"version":VERSION,"lineage":LINEAGE,
                "governance":"IRB + de-id enforced; disallowed intents BLOCKED",
                "math":["occupation composition (Thm 8.3)","BN taxonomy (K1)",
                        "Kou moment gate","Doob no-Zeno (K8)",
                        "symmetric Gibbs conductance (K11)",
                        "corrected surgery table (K10)"],
                "ensemble":"parameter perturbation + VERIFIED spread",
                "carpet":carpet_status(False),
                "clinical_actionability":RESEARCH_ONLY}

# =============================================================================
# §6. SELF-TEST
# =============================================================================
def self_test(verbose=True)->bool:
    print("="*76); print("  MED WORLD MODEL v2 — self-test"); print("="*76)
    torch.manual_seed(0); np.random.seed(0); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] governance + scope")
    gov=Governance()
    rep("patient path BLOCKED without IRB",
        not gov.authorize("simulation",True)["authorized"])
    for txt,lbl in (("design a CRISPR Cas9 guide RNA","gene editor"),
                    ("give a wet lab transfection protocol","wet lab"),
                    ("enhance virulence and immune evasion","pathogen"),
                    ("should we operate on the patient","clinical order")):
        b=False
        try: refuse_disallowed(txt)
        except ScopeRefusal: b=True
        rep(f"{lbl} BLOCKED",b)
    irb=IRBRecord("P-001","Demo","2026-01-01","2099-12-31",["simulation"],
                  data_use_agreement=True,status="APPROVED")
    g2=Governance(irb)
    deid=g2.phi.deidentify({"patient_name":"Jane Doe","age":55,"sex":"female"})
    rep("IRB + de-id authorised",
        g2.authorize("simulation",True,deid,"simulation")["authorized"])

    print("\n[T2] knowledge graph")
    kg=KnowledgeGraph.demo(); n=kg.detect_contradictions()
    rep("contradictions detected",n>=2,f"n={n}")
    r=kg.report(organ="heart")
    rep("confidence reduced by contradictions",r["aggregate_confidence"]<.85,
        f"{r['aggregate_confidence']:.3f}")
    rep("confidence is NOT P(pathogenic)","NOT P(pathogenic)" in r["confidence_semantics"])
    rep("unavailable sources recorded",len(r["sources_unavailable"])==2)

    print("\n[T3] ontogenetic model + occupation diagnostics")
    mw=MedWorldModel(kg=kg,irb=irb)
    st=mw.onto.initial(2)
    obs=torch.rand(2,8,dtype=torch.float64)
    o=mw.onto(st,obs)
    rep("fates sum to 1",
        torch.allclose(o["fates"].sum(-1),torch.ones(2,dtype=torch.float64),atol=1e-9))
    rep("risks in [0,1]",bool(((o["risks"]>=0)&(o["risks"]<=1)).all()))
    rep("CSOC counter bound satisfied",bool(o["csoc"]["satisfied"]))
    comp=mw.onto.composition_diagnostics(o["fates"][0])
    rep("composition on simplex",comp["on_simplex"])
    rep("BN taxonomy checked","taxonomy_BN" in comp)
    rep("unordered regime -> MODE_E",comp["regime"]["status"]==MODE_E)
    fc=mw.onto.csoc.formal_certificate()
    rep("formal Doob certificate valid",fc["certified"])
    rep("horizon truncation REMOVED","REMOVED" in fc["HORIZON_TRUNCATION"])

    print("\n[T4] FIX 1: ensemble members are DISTINCT")
    uq=EnsembleUncertainty(mw.onto,6,0.05)
    u=uq.predict(mw.onto.initial(1),torch.rand(1,8,dtype=torch.float64),steps=5)
    chk=u["member_check"]
    print(f"      max pairwise diff={chk.get('max_pairwise_diff',0):.4e}  "
          f"mean spread={chk.get('mean_spread',0):.4e}")
    rep("members verified distinct",chk["members_distinct"])
    rep("CI reported (non-degenerate)",len(u["final_ci"])>0)
    rep("ensemble CERTIFIED",u["sd"]["status"]==MODE_CERTIFIED)
    # demonstrate the v1 failure mode
    uq0=EnsembleUncertainty(mw.onto,4,0.0)   # zero perturbation = v1 behaviour
    u0=uq0.predict(mw.onto.initial(1),torch.rand(1,8,dtype=torch.float64),steps=5)
    rep("zero-perturbation ensemble -> MODE_E",u0["sd"]["status"]==MODE_E)
    rep("no CI emitted for identical members",len(u0["final_ci"])==0)

    print("\n[T5] risk + moment gate + tumorigenicity")
    ts=TumorigenicitySpec(n_crit_cells=100.,dose_cells=1e7,safety_factor=1.,
                          assay_lod_fraction=1e-6,cells_interrogated=5e5,
                          positives_observed=0)
    regs=[KouRegime("common",eta_up=4.),
          KouRegime("hyper",eta_up=.7,occupation=.05)]
    risk=mw.risk.score(o["state"][0],"heart",["IRB approval"],ts,regs)
    rep("8 risk scores",len(risk["risk_scores"])==8)
    rep("missing evidence reported",len(risk["missing_evidence"])>5)
    rep("NO go/no-go issued",risk["decision"]=="NO_GO_NO_GO_ISSUED")
    rq=risk["tumorigenicity_arithmetic"]["required"]
    print(f"      p_max={rq['p_max_fraction']:.2e}  n_assay={rq['n_assay_required']:.0f}")
    rep("p_max = 1e-5",abs(rq["p_max_fraction"]-1e-5)<1e-12)
    rep("mean REFUSED for eta+<1","REFUSED_MEAN" in risk)
    rep("translation gap present",len(risk["translation_gap"])==6)
    regs_ok=[KouRegime("A",eta_up=3.),KouRegime("B",eta_up=2.,occupation=.1)]
    risk2=mw.risk.score(o["state"][0],"heart",None,None,regs_ok)
    rep("rare line governs tail when finite",
        risk2["tail_governance"]["governing_regimes"]==["B"])

    print("\n[T6] FIX 2: uncertainty uses the FINAL state")
    out=mw.run_scenario("heart",InterventionClass.GROWTH_FACTOR_SCREEN,
                        "simulate cardiac organoid maturation for research",
                        steps=5)
    rep("scenario completes",out["scenario_id"] is not None)
    rep("uncertainty reference is FINAL",
        "FINAL" in out["uncertainty_reference_state"])
    rep("research_only",out["clinical_actionability"]==RESEARCH_ONLY)
    b=False
    try:
        mw.run_scenario("heart",InterventionClass.OBSERVATION,
                        "design CRISPR Cas9 guide RNA for gene editing")
    except ScopeRefusal: b=True
    rep("disallowed scenario REFUSED",b)

    print("\n[T7] K11 symmetric Gibbs + carpet status")
    gs=verify_gibbs_symmetry([0.,.5,1.,1.5],2.)
    rep("corrected conductance symmetric",gs["is_symmetric"])
    rep("source conductance NOT symmetric",not gs["source_was_symmetric"])
    cp=carpet_status(False)
    rep("non-PCF needs 3 imported inputs",len(cp["inputs_needed"])==3)
    rr=mw.sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    rep("Mode E retained",rr["mode_E"]>0)
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  Ensemble members are genuinely perturbed and VERIFIED distinct; a
  zero-perturbation ensemble is detected and returns Mode E with no CI.
  Uncertainty is computed from the FINAL state, explicitly declared.

MATHEMATICS PROPAGATED
  composition = occupation measure (Thm 8.3 rate needs n_cells)
  BN lineage taxonomy (K1) · Kou moment gate (Prop 16.14)
  Doob no-Zeno (K8) · symmetric Gibbs conductance (K11)
  tumorigenicity p_max / n_assay arithmetic · translation-gap register

REFUSED
  Gene-editor design · wet-lab protocols · pathogen enhancement
  Autonomous clinical orders · transplant go/no-go · infinite means
""")
    return ok

def main()->int:
    ap=argparse.ArgumentParser(description="Med world model v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","status","scenario","risk","ensemble","moment"])
    ap.add_argument("--organ",default="heart")
    ap.add_argument("--steps",type=int,default=10)
    ap.add_argument("--eta-up",type=float,default=3.0)
    ap.add_argument("--json",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    mw=MedWorldModel(); out={}
    if a.cmd=="status": out=mw.status()
    elif a.cmd=="scenario":
        out=mw.run_scenario(a.organ,InterventionClass.GROWTH_FACTOR_SCREEN,
                            "simulate organoid maturation for research",a.steps)
    elif a.cmd=="risk":
        o=mw.onto(mw.onto.initial(1),torch.rand(1,8,dtype=torch.float64))
        out=mw.risk.score(o["state"][0],a.organ,None,
                          TumorigenicitySpec(),[KouRegime("x",eta_up=a.eta_up)])
    elif a.cmd=="ensemble":
        out=mw.uq.predict(mw.onto.initial(1),torch.rand(1,8,dtype=torch.float64),
                          steps=a.steps)
    elif a.cmd=="moment":
        out=moment_estimable([KouRegime("x",eta_up=a.eta_up)],1.)
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
