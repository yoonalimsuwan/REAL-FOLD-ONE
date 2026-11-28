#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sle_longitudinal.py — v2.0.0
SLE flare lead-time RESEARCH pipeline on the corrected ESCA mathematics.

BUGS FIXED vs v1
  1. RATE: the OLS-slope subsampling used tau_n = n^{3/2}, which assumes
     EQUALLY SPACED visits. Real SLE visits are irregular (spacing CV is
     measured and often > 0.5). v2 uses the design-based tau_n = sqrt(S_tt),
     S_tt = sum (t_i - tbar)^2, which is exact in both cases and reduces to
     n^{3/2}/sqrt(12) on a grid. Block normalisation uses the block's S_tt.
  2. Dead walrus `len(alerts_weeks := alerts_week)` removed.
  3. Supporting-evidence items were hardcoded as missing; now checked.
  4. No SD gate; no moment-estimability gate on heavy-tailed titres.

NEW FROM ESCA 2.0 UNIFIED EDITION
  * Disease-state fractions ARE an occupation vector -> Thm 8.3 W2 rate
    and the Prop 8.5/8.6 test separating a STATIONARY course from a
    genuinely BISTABLE one (time-ordered, so the test is admissible).
  * OPMC change points on biomarker series with the honest (SNR).
  * Kou moment gate: anti-dsDNA titres are heavy-tailed; if eta+ <= 1 the
    MEAN titre is INFINITE and must not be reported.
  * K8 Doob no-Zeno certificate for the alert sequence; the CSOC bound is
    an ENGINEERING bound on the counter, NOT the theorem.
  * BN check on a nested disease-activity taxonomy before pooling levels.

UNCHANGED AND CORRECT
  CSOC bound N(T) <= (E_max + D)/gap with D = ACTUAL removal. Verified.
  Medication log is a BLOCKING requirement: without it the model learns
  steroid tapers, not flare biology.

PHILOSOPHY: see msps_research_math_core.PHILOSOPHY (full text).
RESEARCH ONLY. No diagnosis, treatment, dosing, or steroid-taper advice.
SLE is a loss of immune tolerance, NOT a protein-folding disease.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from msps_research_math_core import (
    SDGate, Refusal, InfiniteMoment, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    subsample_ci, iid_ci_WRONG, tau_int_ms, coverage_study, stat_and_rate,
    composition_as_occupation, theorem_A_rate, occupation_regime_test,
    boundary_null_check, opmc_certified, KouRegime, fit_kou_tail,
    moment_estimable, response_summary, heaviest_regime_dominates,
    no_zeno_certificate, surgery_count_tail, detectable_effect,
    get_logger,
)
log=get_logger("sle_v2"); VERSION="2.0.0"

class Status:
    OK="OK"; BLOCKED="BLOCKED"; UNAVAILABLE="UNAVAILABLE"
    NOT_COMPARED="NOT_COMPARED"; INSUFFICIENT="INSUFFICIENT"

# =============================================================================
# §1. GOVERNANCE
# =============================================================================
class SafetyGate:
    BLOCKED={
     "treatment_advice":[r"\bprescribe\b",
       r"\bstart\s+(steroid|prednisolone|mycophenolate|cyclophosphamide|"
       r"belimumab|anifrolumab|rituximab)\b",
       r"\bincrease\s+(the\s+)?(steroid|prednisolone)\b",
       r"\btaper\b.*\b(steroid|prednisolone)\b",
       r"\bwhat\s+dose\b",r"\bshould\s+(i|we)\s+(treat|start|stop)\b",
       r"\brecommend\s+(treatment|therapy|drug)\b"],
     "gene_editor_design":[r"\bcrispr\b",r"\bcas9\b",r"\bguide\s*rna\b",
       r"\bgrna\b",r"\bbase\s*editor\b",r"\bprime\s*editor\b",
       r"\bcar[\s-]*t\s*(construct|design)\b",r"\bgene\s*edit(or|ing)\b",
       r"\bdelivery\s*vector\b"],
     "autonomous_clinical":[r"\bdiagnose\s*(the\s*)?patient\b",
       r"\bclinical\s*decision\b",r"\bapprove\s*treatment\b"],
    }
    def check(self,text:str)->Dict[str,Any]:
        low=text.lower()
        for intent,pats in self.BLOCKED.items():
            hits=[p for p in pats if re.search(p,low)]
            if hits:
                return {"allowed":False,"intent":intent,"blocked_patterns":hits,
                        "reason":f"BLOCKED intent={intent}. No treatment advice, "
                                 "dosing, gene-editor design, or clinical decisions."}
        return {"allowed":True,"intent":"research","blocked_patterns":[],
                "reason":"allowed research query"}

class PHIGuard:
    PATTERNS={"name_like":r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b",
      "mrn":r"\b(MRN|mrn)[:\s]*\d{4,}\b","ssn":r"\b\d{3}-\d{2}-\d{4}\b",
      "full_date":r"\b(19|20)\d{2}-\d{2}-\d{2}\b",
      "email":r"[\w.+-]+@[\w-]+\.[\w.]+","phone":r"\b\+?\d[\d\s().-]{8,}\d\b"}
    DIRECT={"name","patient_name","mrn","patient_id","national_id","address",
      "email","phone","birth_date","dob","date_of_birth","accession_number",
      "hospital_number"}
    QUASI={"age","sex","ethnicity","postcode","site","rare_diagnosis",
      "enrolment_date"}
    def scrub(self,text:str)->Tuple[str,List[str]]:
        out,rm=text,[]
        for lab,pat in self.PATTERNS.items():
            new=re.sub(pat,"[REDACTED]",out)
            if new!=out: rm.append(lab); out=new
        return out,rm
    def deidentify(self,rec:Dict)->Dict[str,Any]:
        rm,q,out=[],[],dict(rec)
        for k in list(out.keys()):
            if k.lower() in self.DIRECT: out.pop(k); rm.append(k)
            elif k.lower() in self.QUASI: q.append(k)
        if "age" in out:
            try:
                a=int(out.pop("age")); out["age_band"]=f"{(a//10)*10}-{(a//10)*10+9}"
                rm.append("age->age_band")
            except Exception: pass
        for k,v in list(out.items()):
            if isinstance(v,str):
                s,r=self.scrub(v)
                if r: out[k]=s; rm.append(f"{k}:{','.join(r)}")
        risk="low" if not q else ("medium" if len(q)<=3 else "high")
        return {"passed":risk!="high","removed_fields":rm,
                "residual_risk":risk,"quasi_identifiers":q,"clean_record":out}

@dataclass
class IRBRecord:
    protocol_id:str; institution:str; approval_date:str; expiry_date:str
    approved_uses:List[str]; consent_model:str="written"
    data_use_agreement:bool=False; status:str="PENDING"
    def active(self)->bool:
        if self.status!="APPROVED": return False
        try: return time.time()<time.mktime(time.strptime(self.expiry_date,"%Y-%m-%d"))
        except Exception: return False

class Governance:
    def __init__(self,irb:Optional[IRBRecord]=None):
        self.irb,self.phi,self.safety=irb,PHIGuard(),SafetyGate()
        self.audit:List[Dict]=[]
    def authorize(self,text:str,uses_patient_data:bool=False,
                  deid:Optional[Dict]=None,purpose:str="sle_research")->Dict:
        s=self.safety.check(text); d=dict(s)
        d.update({"authorized":False,"clinician_review_required":True,
                  "clinical_actionability":RESEARCH_ONLY})
        if not s["allowed"]: self._log(d,text); return d
        if uses_patient_data:
            for cond,reason in (
                (self.irb is None,"BLOCKED: no IRB record"),
                (self.irb is not None and not self.irb.active(),
                 "BLOCKED: IRB inactive or expired"),
                (self.irb is not None and purpose not in self.irb.approved_uses,
                 f"BLOCKED: purpose '{purpose}' not approved"),
                (self.irb is not None and not self.irb.data_use_agreement,
                 "BLOCKED: no data use agreement"),
                (not deid or not deid.get("passed"),
                 "BLOCKED: de-identification missing or failed")):
                if cond: d["reason"]=reason; self._log(d,text); return d
            d["authorized"]=True; d["reason"]="authorized for research use"
        else:
            d["authorized"]=True; d["reason"]="public/synthetic research query"
        self._log(d,text); return d
    def _log(self,d,text):
        self.audit.append({"ts":time.time(),
            "query_hash":hashlib.sha256(text.encode()).hexdigest()[:16],
            "authorized":d["authorized"],"reason":d["reason"]})

# =============================================================================
# §2. DATA SCHEMA
# =============================================================================
@dataclass
class AssayMeta:
    analyte:str; method:str; unit:str
    ref_low:Optional[float]=None; ref_high:Optional[float]=None
    site_id:str="site_anon"; lod:Optional[float]=None
    def complete(self)->bool: return bool(self.method and self.unit)

@dataclass
class Visit:
    t_week:float
    sledai2k:Optional[float]=None; anti_dsdna:Optional[float]=None
    c3_mg_dl:Optional[float]=None; c4_mg_dl:Optional[float]=None
    upcr_g_g:Optional[float]=None; ifn_score:Optional[float]=None
    alc_per_ul:Optional[float]=None; anti_c1q:Optional[float]=None
    prednisolone_mg_day:Optional[float]=None; hcq_mg_day:Optional[float]=None
    immunosuppressant:Optional[str]=None; biologic:Optional[str]=None
    infection_present:Optional[bool]=None
    upcr_method:Optional[str]=None

@dataclass
class FlareAdjudication:
    t_week:float; definition:str; severity:str; adjudicated_by:str
    infection_excluded:bool; organ_domain:str="unspecified"

DISEASE_STATES=["REMISION","LOW_ACTIVITY","MODERATE","HIGH","SEVERE"]
ACTIVITY_HIERARCHY={"ACTIVE":["LOW_ACTIVITY","MODERATE","HIGH","SEVERE"],
                    "HIGH_BURDEN":["HIGH","SEVERE"]}

def sledai_state(s:Optional[float]) -> str:
    if s is None: return "REMISION"
    if s<=2: return "REMISION"
    if s<=5: return "LOW_ACTIVITY"
    if s<=10: return "MODERATE"
    if s<=20: return "HIGH"
    return "SEVERE"

@dataclass
class PatientSeries:
    research_id:str; visits:List[Visit]
    flares:List[FlareAdjudication]=field(default_factory=list)
    assays:Dict[str,AssayMeta]=field(default_factory=dict)
    ifn_stratum:str="unknown"
    sledai_assessor_reliability:Optional[float]=None
    consent_scope:str=RESEARCH_ONLY
    def n(self)->int: return len(self.visits)
    def spacing_cv(self)->float:
        if self.n()<3: return float("nan")
        g=np.diff([v.t_week for v in self.visits]); m=float(g.mean())
        return float(g.std(ddof=1)/m) if m>0 else float("nan")
    def series(self,key:str)->Tuple[np.ndarray,np.ndarray]:
        t,y=[],[]
        for v in self.visits:
            val=getattr(v,key,None)
            if val is not None and np.isfinite(val):
                t.append(v.t_week); y.append(float(val))
        return np.array(t,float),np.array(y,float)
    def state_labels(self)->List[str]:
        return [sledai_state(v.sledai2k) for v in self.visits]

# =============================================================================
# §3. FEATURES  (correct rate, occupation, moment gate)
# =============================================================================
BIOMARKERS=("sledai2k","anti_dsdna","c3_mg_dl","c4_mg_dl","upcr_g_g",
            "ifn_score","alc_per_ul","anti_c1q")
ADVERSE_DIRECTION={"sledai2k":+1,"anti_dsdna":+1,"upcr_g_g":+1,
  "ifn_score":+1,"anti_c1q":+1,"c3_mg_dl":-1,"c4_mg_dl":-1,"alc_per_ul":-1}

def extract_features(ps:PatientSeries, window:int=8, conf:float=0.95,
                     sd:Optional[SDGate]=None) -> Dict[str,Any]:
    sd=sd or SDGate()
    out={"window":window,"conf":conf,"biomarkers":{}}
    for key in BIOMARKERS:
        t,y=ps.series(key)
        if len(y)==0:
            out["biomarkers"][key]={"status":Status.UNAVAILABLE}; continue
        t=t[-window:]; y=y[-window:]
        if len(t)>1: t=t-t[0]
        lvl=subsample_ci(t,y,"mean",conf=conf)
        trd=subsample_ci(t,y,"slope",conf=conf)
        ti=tau_int_ms(y); iid=iid_ci_WRONG(t,y,"slope",conf=conf)
        sd.certify("subsampling_ci",("RATE",),
                   {"stat":"slope","rate_name":trd.get("rate_name"),
                    "spacing_cv":trd.get("spacing_cv")})
        d=ADVERSE_DIRECTION.get(key,+1)
        adverse=(trd["status"]==MODE_CERTIFIED and trd["lo"]*d>0.0)
        # NEW: moment-estimability gate on the (possibly heavy-tailed) titre
        tail=None
        if len(y)>=8:
            try:
                reg=fit_kou_tail(np.log(np.maximum(y,1e-9)),name=key)
                tail=response_summary(np.log(np.maximum(y,1e-9)),regimes=[reg])
                tail["eta_up"]=reg.eta_up
                sd.certify("response_tail",("L1e",),
                           {"finite":tail["first_moment_finite"]})
            except Refusal: tail=None
        out["biomarkers"][key]={
          "status":Status.OK,"n_used":len(y),
          "level":lvl,"trend":trd,"tau_int":ti,
          "iid_trend_FOR_COMPARISON_ONLY":iid,
          "width_ratio_sub_over_iid":(trd.get("width",float("nan"))/
                                      max(iid.get("width",1e-30),1e-30)),
          "adverse_trend_significant":bool(adverse),"adverse_direction":d,
          "log_titre_tail":tail}
    out["spacing_cv"]=ps.spacing_cv()
    out["spacing_warning"]=bool(np.isfinite(out["spacing_cv"]) and
                                out["spacing_cv"]>0.5)
    out["RATE_NOTE"]=("slope uses tau_n=sqrt(S_tt); n^{3/2} is only valid "
                      "for equally spaced visits")
    return out

def disease_state_diagnostics(ps:PatientSeries,
                              sd:Optional[SDGate]=None) -> Dict[str,Any]:
    """
    NEW. The fraction of visits in each activity state IS an occupation
    vector. Visits are TIME-ORDERED, so the regime test is admissible.
    """
    sd=sd or SDGate()
    labels=ps.state_labels()
    oc=composition_as_occupation(labels,DISEASE_STATES)
    rate=theorem_A_rate(len(labels),len(DISEASE_STATES))
    comp={k:float(v) for k,v in zip(DISEASE_STATES,oc["F"])}
    for parent,kids in ACTIVITY_HIERARCHY.items():
        comp[parent]=sum(comp.get(c,0.) for c in kids)
    bn=boundary_null_check(ACTIVITY_HIERARCHY,comp)
    regime=occupation_regime_test(labels,DISEASE_STATES,ordered=True)
    sd.certify("taxonomy_pooling",("BN",) if bn["BN_satisfied"] else (),bn)
    sd.certify("occupation_regime",(),{"regime":regime["regime"]})
    return {"composition":oc["F"],"names":DISEASE_STATES,
            "n_visits":len(labels),"W2_bound":rate["W2_bound"],
            "taxonomy_BN":bn,"regime":regime,
            "interpretation":("LIGHT_BLOCKS => a stationary, noisy course; "
                              "SUPER_DOMINANT => genuinely bistable, so a "
                              "single mean activity level is NOT the limit")}

def biomarker_changepoints(ps:PatientSeries, key:str="anti_dsdna",
                           ell:int=4, sigma:float=1.0,
                           sd:Optional[SDGate]=None) -> Dict[str,Any]:
    sd=sd or SDGate()
    t,y=ps.series(key)
    if len(y)<12:
        return {"status":Status.INSUFFICIENT,"reason":"need >= 12 visits"}
    z=(y-y.mean())/max(y.std(),1e-9)
    r=opmc_certified(list(z),ell,sigma)
    sd.certify("changepoints",("L1","D","O") if r["status"]==MODE_CERTIFIED else (),
               {"K_hat":r["K_hat"]})
    return {"biomarker":key,**r,
            "boundary_weeks":[float(t[b]) for b in r["boundaries"] if b<len(t)],
            "NOTE":"strict (SNR) carries (K+1)^2; practical series often fail it"}

# =============================================================================
# §4. CSOC ALERT COUNTER  (bound unchanged, formal certificate added)
# =============================================================================
class CSOCAlertCounter:
    """
    N(T) <= (E_max + D)/gap with D = ACTUAL removal. ENGINEERING bound on
    the counter. The THEOREM about non-accumulating event times is the
    Doob no-Zeno certificate below, under (U),(R),(M).
    """
    def __init__(self,theta_hi=0.60,theta_lo=0.25,decay_rate=0.08,
                 min_interval_visits=2):
        if not theta_hi>theta_lo: raise Refusal("theta_hi > theta_lo required")
        self.hi,self.lo=float(theta_hi),float(theta_lo)
        self.gap=self.hi-self.lo; self.rate=float(decay_rate)
        self.min_interval=int(min_interval_visits)
        self.acc=0.0; self.armed=True; self.events=0
        self.E_max=0.0; self.D=0.0; self.D_audit=0.0
        self._last_fire=-10**9; self.history:List[Dict]=[]
    @staticmethod
    def _softplus(x): return math.log1p(math.exp(-abs(x)))+max(x,0.0)
    def step(self,signal:float,k:int)->Dict[str,Any]:
        s=float(signal)
        excess=self._softplus(s-self.hi)
        decay=1.0/(1.0+math.exp(-(self.acc-self.lo)*10.0))
        removal=decay*self.rate*self.acc
        self.acc+=excess-removal
        self.D+=removal; self.D_audit+=removal
        self.E_max=max(self.E_max,self.acc)
        if self.acc<self.lo: self.armed=True
        fire=(self.armed and self.acc>self.hi
              and (k-self._last_fire)>=self.min_interval)
        if fire:
            self.armed=False; self.events+=1; self._last_fire=int(k)
        bound=(self.E_max+self.D)/self.gap
        rec={"k":k,"signal":s,"acc":self.acc,"removal":removal,
             "fired":bool(fire),"events":self.events,"bound":bound,
             "satisfied":self.events<=bound+1e-9}
        self.history.append(rec); return rec
    def report(self)->Dict[str,Any]:
        bound=(self.E_max+self.D)/self.gap
        return {"events":self.events,"bound":bound,
                "satisfied":self.events<=bound+1e-9,
                "E_max":self.E_max,"D_actual":self.D,
                "D_audit_matches":abs(self.D-self.D_audit)<1e-12,
                "note":"ENGINEERING bound on the counter, NOT a clinical "
                       "guarantee and NOT the no-Zeno theorem"}
    def formal_certificate(self,delta=0.05,B0=2.0,G0=0.5)->Dict:
        return no_zeno_certificate(delta,B0,G0)

# =============================================================================
# §5. SCORE + STRATIFICATION
# =============================================================================
@dataclass
class ScoreWeights:
    anti_dsdna_trend:float=0.30; complement_trend:float=0.25
    upcr_trend:float=0.20; sledai_level:float=0.15; ifn_level:float=0.10
    sledai_ref:float=12.0; ifn_ref:float=10.0; trend_ref:float=0.05
    def declared(self): return asdict(self)

def flare_pressure(feat:Dict,w:Optional[ScoreWeights]=None)->Dict[str,Any]:
    w=w or ScoreWeights(); b=feat["biomarkers"]
    def sig_trend(key):
        r=b.get(key,{})
        if r.get("status")!=Status.OK: return 0.0
        tr=r["trend"]
        if tr["status"]!=MODE_CERTIFIED: return 0.0
        d=ADVERSE_DIRECTION.get(key,+1)
        cons=tr["lo"] if d>0 else -tr["hi"]
        return float(max(0.0,d*cons)/max(w.trend_ref,1e-9))
    def lvl(key,ref):
        r=b.get(key,{})
        if r.get("status")!=Status.OK: return 0.0
        lv=r["level"]
        if lv["status"]!=MODE_CERTIFIED: return 0.0
        return float(max(0.0,lv["estimate"])/max(ref,1e-9))
    parts={"anti_dsdna_trend":w.anti_dsdna_trend*min(1.,sig_trend("anti_dsdna")),
      "complement_trend":w.complement_trend*min(1.,0.5*(sig_trend("c3_mg_dl")
                                                        +sig_trend("c4_mg_dl"))),
      "upcr_trend":w.upcr_trend*min(1.,sig_trend("upcr_g_g")),
      "sledai_level":w.sledai_level*min(1.,lvl("sledai2k",w.sledai_ref)),
      "ifn_level":w.ifn_level*min(1.,lvl("ifn_score",w.ifn_ref))}
    score=float(sum(parts.values()))
    return {"score":min(1.,max(0.,score)),"components":parts,
            "weights_declared":w.declared(),
            "model_class":"transparent declared index, NOT fitted",
            "clinical_actionability":RESEARCH_ONLY}

def stratify(ps:PatientSeries,ifn_threshold:float=5.0)->Dict[str,Any]:
    t,y=ps.series("ifn_score")
    if len(y)==0:
        return {"stratum":"unknown","reason":"no interferon score available"}
    m=float(np.median(y))
    return {"stratum":"high" if m>=ifn_threshold else "low",
            "median_ifn":m,"threshold":ifn_threshold,"n_used":len(y)}

# =============================================================================
# §6. MISSING EVIDENCE
# =============================================================================
BLOCKING=[
 "IRB approval active and purpose within approved_uses",
 "de-identification passed with residual risk not high",
 "MEDICATION LOG: prednisolone dose at every visit",
 "immunosuppressant / biologic exposure recorded at every visit",
 "infection status assessed at every visit (flare mimic)",
 "flare ADJUDICATED by a declared definition, independent assessor",
 "anti-dsDNA assay method + unit + reference range per site",
 "C3 / C4 reference range per site",
 "SLEDAI-2K assessor reliability reported (kappa or ICC >= 0.7)",
]
SUPPORTING=[
 "hydroxychloroquine dose and adherence",
 "UPCR collection method (spot vs 24 h)",
 ">= 8 visits per patient with a declared target spacing",
 "follow-up duration >= 48 weeks",
 "missing-data mechanism documented (MCAR / MAR / MNAR)",
 "pre-declared prediction horizon and analysis plan",
 "site harmonisation plan for pooled anti-dsDNA",
 "renal biopsy status where nephritis is claimed",
]

def missing_evidence(ps:PatientSeries,gov:Dict,min_visits:int=8)->Dict[str,Any]:
    present=[]; mb=[]; ms=[]
    if gov.get("authorized"): present+=BLOCKING[:2]
    else: mb+=BLOCKING[:2]
    v=ps.visits; n=len(v)
    checks=[(n>0 and all(x.prednisolone_mg_day is not None for x in v),BLOCKING[2]),
            (n>0 and all(x.immunosuppressant is not None for x in v),BLOCKING[3]),
            (n>0 and all(x.infection_present is not None for x in v),BLOCKING[4]),
            (len(ps.flares)>0 and all(f.adjudicated_by and f.definition
                                      for f in ps.flares),BLOCKING[5]),
            (ps.assays.get("anti_dsdna") is not None and
             ps.assays["anti_dsdna"].complete(),BLOCKING[6]),
            (ps.assays.get("c3_mg_dl") is not None and
             ps.assays["c3_mg_dl"].complete(),BLOCKING[7]),
            (ps.sledai_assessor_reliability is not None and
             ps.sledai_assessor_reliability>=0.7,BLOCKING[8])]
    for ok,label in checks: (present if ok else mb).append(label)
    dur=(v[-1].t_week-v[0].t_week) if n>=2 else 0.0
    sup=[(all(x.hcq_mg_day is not None for x in v) if n else False,SUPPORTING[0]),
         (all(x.upcr_method is not None for x in v) if n else False,SUPPORTING[1]),
         (n>=min_visits,SUPPORTING[2]),(dur>=48,SUPPORTING[3])]
    for ok,label in sup: (present if ok else ms).append(label)
    ms+=SUPPORTING[4:]
    comp=len(present)/max(len(BLOCKING)+len(SUPPORTING),1)
    return {"status":Status.BLOCKED if mb else Status.OK,
            "n_visits":n,"follow_up_weeks":dur,"present":present,
            "missing_blocking":mb,"missing_supporting":ms,
            "evidence_completeness":comp,
            "study_usable_for_prediction_research":not mb,
            "decision":"NO_CLINICAL_DECISION_ISSUED",
            "explicitly_not_provided":["diagnosis","treatment","dosing",
                                       "steroid taper advice","prognosis"]}

# =============================================================================
# §7. VALIDATION
# =============================================================================
def horizon_validation(alerts_week:List[float],flares_week:List[float],
                       horizon_weeks:float=12.0,
                       follow_up_weeks:float=52.0)->Dict[str,Any]:
    if not flares_week:
        return {"status":Status.NOT_COMPARED,
                "reason":"no adjudicated flare reference supplied",
                "n_alerts":len(alerts_week)}
    tp=fp=0; leads=[]
    for a in alerts_week:
        nxt=[f for f in flares_week if a<f<=a+horizon_weeks]
        if nxt: tp+=1; leads.append(min(nxt)-a)
        else: fp+=1
    n_alert=len(alerts_week)
    covered=sum(1 for f in flares_week
                if any(f-horizon_weeks<=a<f for a in alerts_week))
    return {"status":Status.OK,"horizon_weeks":horizon_weeks,
            "n_alerts":n_alert,"n_flares":len(flares_week),
            "true_positive_alerts":tp,"false_positive_alerts":fp,
            "false_alarm_rate":fp/max(n_alert,1),
            "positive_predictive_value":tp/max(n_alert,1),
            "flare_sensitivity_at_horizon":covered/max(len(flares_week),1),
            "median_lead_time_weeks":float(np.median(leads)) if leads else None,
            "lead_time_iqr_weeks":([float(np.quantile(leads,.25)),
                                    float(np.quantile(leads,.75))] if leads else None),
            "follow_up_weeks":follow_up_weeks,
            "note":"research metrics; no clinical utility claimed"}

# =============================================================================
# §8. PIPELINE
# =============================================================================
class SLEPipeline:
    def __init__(self,irb:Optional[IRBRecord]=None,horizon_weeks:float=12.0,
                 window:int=8):
        self.gov=Governance(irb); self.sd=SDGate()
        self.horizon=float(horizon_weeks); self.window=int(window)
    def run(self,ps:PatientSeries,question:str="SLE flare lead-time research",
            uses_patient_data:bool=True,deid:Optional[Dict]=None)->Dict[str,Any]:
        d=self.gov.authorize(question,uses_patient_data,deid)
        if not d["authorized"]:
            return {"status":Status.BLOCKED,"governance":d}
        feat=extract_features(ps,self.window,sd=self.sd)
        score=flare_pressure(feat); stratum=stratify(ps)
        states=disease_state_diagnostics(ps,self.sd)
        cps=biomarker_changepoints(ps,sd=self.sd)
        ev=missing_evidence(ps,d)
        self.sd.certify("sle_study",
                        ("MEDS","ADJ") if not ev["missing_blocking"] else (),
                        {"n_blocking":len(ev["missing_blocking"])})
        counter=CSOCAlertCounter(); alerts=[]
        for k in range(3,ps.n()+1):
            sub=PatientSeries(ps.research_id,ps.visits[:k],[],ps.assays,
                              ps.ifn_stratum,ps.sledai_assessor_reliability)
            f=extract_features(sub,self.window)
            rec=counter.step(flare_pressure(f)["score"],k)
            if rec["fired"]: alerts.append(ps.visits[k-1].t_week)
        val=horizon_validation(alerts,[f.t_week for f in ps.flares],self.horizon,
                               (ps.visits[-1].t_week-ps.visits[0].t_week)
                               if ps.n()>=2 else 0.0)
        return {"report_id":str(uuid.uuid4())[:8],
                "status":Status.OK if not ev["missing_blocking"] else Status.BLOCKED,
                "research_id":ps.research_id,"governance":d,
                "features":feat,"flare_pressure":score,"ifn_stratum":stratum,
                "disease_state_occupation":states,"biomarker_changepoints":cps,
                "alert_bound":counter.report(),
                "alert_no_zeno_certificate":counter.formal_certificate(),
                "alert_weeks":alerts,"validation":val,"missing_evidence":ev,
                "sd_report":self.sd.report(),
                "clinical_actionability":RESEARCH_ONLY,
                "clinician_review_required":True,
                "disclaimer":("RESEARCH ONLY. Not a diagnostic device. No "
                              "treatment, dosing, or steroid-taper advice. SLE "
                              "management requires a rheumatologist and, for "
                              "nephritis, a nephrologist.")}

# =============================================================================
# §9. SYNTHETIC COHORT
# =============================================================================
def synthetic_patient(seed:int=0,n_visits:int=24,will_flare:bool=True,
                      flare_visit:int=16,ifn_high:bool=True,
                      with_meds:bool=True,irregular:bool=True)->PatientSeries:
    rng=np.random.default_rng(seed); visits=[]; t=0.0
    dsdna,c3,c4,upcr=25.,95.,22.,0.15
    for k in range(n_visits):
        t+=float(max(2.,rng.normal(6.,2. if irregular else 0.05)))
        pre=max(0,min(6,flare_visit-k)) if will_flare else 0
        drive=(6-pre)/6. if will_flare and k<=flare_visit else 0.
        if will_flare and k>flare_visit:
            drive=max(0.,1.-0.25*(k-flare_visit))
        dsdna=max(2.,0.75*dsdna+0.25*(25+180*drive)+rng.normal(0,8))
        c3=max(20.,0.75*c3+0.25*(95-45*drive)+rng.normal(0,4))
        c4=max(2.,0.75*c4+0.25*(22-12*drive)+rng.normal(0,1.5))
        upcr=max(0.02,0.7*upcr+0.3*(0.15+1.6*drive)+rng.normal(0,0.05))
        sledai=float(np.clip(round(2+12*drive+rng.normal(0,1)),0,40))
        ifn=float(np.clip((8. if ifn_high else 2.)+3*drive+rng.normal(0,1.),0,25))
        alc=float(np.clip(1400-600*drive+rng.normal(0,120),200,3000))
        pred=(20.+25.*drive if (will_flare and k>=flare_visit-1) else 7.5) \
             if with_meds else None
        visits.append(Visit(t_week=t,sledai2k=sledai,anti_dsdna=dsdna,
          c3_mg_dl=c3,c4_mg_dl=c4,upcr_g_g=upcr,ifn_score=ifn,alc_per_ul=alc,
          prednisolone_mg_day=(float(pred) if pred is not None else None),
          hcq_mg_day=200.,
          immunosuppressant="mycophenolate" if with_meds else None,
          biologic="none",infection_present=False,upcr_method="spot"))
    flares=[]
    if will_flare:
        flares.append(FlareAdjudication(visits[flare_visit].t_week,
          "SELENA-SLEDAI SFI","severe","assessor_anon_01",True,"renal"))
    assays={"anti_dsdna":AssayMeta("anti_dsdna","ELISA","IU/mL",0.,25.),
            "c3_mg_dl":AssayMeta("c3_mg_dl","immunoturbidimetry","mg/dL",90.,180.),
            "c4_mg_dl":AssayMeta("c4_mg_dl","immunoturbidimetry","mg/dL",10.,40.)}
    return PatientSeries(f"SYNTH_{seed:04d}",visits,flares,assays,
                         "high" if ifn_high else "low",0.85)

# =============================================================================
# §10. SELF-TEST
# =============================================================================
def self_test(verbose:bool=True)->bool:
    print("="*76); print("  SLE LONGITUDINAL v2 — self-test"); print("="*76)
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng=np.random.default_rng(0)

    print("\n[T1] governance")
    gov=Governance()
    rep("patient path BLOCKED without IRB",
        not gov.authorize("SLE research",True)["authorized"])
    for txt,lab in (("increase the prednisolone dose","treatment advice"),
                    ("what dose of mycophenolate","dosing"),
                    ("design a CAR-T construct for lupus","cell design"),
                    ("design a CRISPR guide RNA","gene editor")):
        rep(f"{lab} BLOCKED",not gov.authorize(txt)["authorized"])
    irb=IRBRecord("SLE-001","Demo","2026-01-01","2099-12-31",["sle_research"],
                  data_use_agreement=True,status="APPROVED")
    g2=Governance(irb)
    rep("IRB alone insufficient (no de-id)",
        not g2.authorize("SLE research",True)["authorized"])
    deid=g2.phi.deidentify({"patient_name":"Jane Doe","mrn":"MRN: 12345",
                            "age":34,"sex":"female"})
    rep("IRB + de-id authorised",g2.authorize("SLE research",True,deid)["authorized"])

    print("\n[T2] THE RATE FIX")
    t=np.arange(40,dtype=float); y=0.3*t+rng.normal(0,1,40)
    m=subsample_ci(t,y,"mean"); s=subsample_ci(t,y,"slope")
    print(f"      mean rate={m['rate_name']}  slope rate={s['rate_name']}")
    rep("mean rate sqrt(n)",m["rate_name"]=="sqrt(n)")
    rep("slope rate sqrt(S_tt) NOT n^1.5",s["rate_name"]=="sqrt(S_tt)")
    rep("slope CI brackets 0.3",s["lo"]<=0.3<=s["hi"],
        f"[{s['lo']:.4f},{s['hi']:.4f}]")
    ti=np.cumsum(rng.uniform(2.,10.,40)); yi=0.3*ti+rng.normal(0,3,40)
    si=subsample_ci(ti,yi,"slope")
    rep("irregular spacing detected",si["spacing_irregular"],
        f"CV={si['spacing_cv']:.3f}")
    rep("rate note explains the failure","n^{3/2} is only valid" in
        extract_features(synthetic_patient(1),8)["RATE_NOTE"])

    print("\n[T3] COVERAGE")
    for rho in (0.0,0.7):
        c=coverage_study(n_real=200,n=60,rho=rho,seed=1)
        print(f"      rho={rho:.1f} sub={c['coverage_subsampling']:.3f} "
              f"iid={c['coverage_iid']:.3f} ratio={c['width_ratio']:.2f}")
        if rho>=0.6:
            rep(f"rho={rho}: iid UNDERCOVERS",c["coverage_iid"]<0.85)
            rep(f"rho={rho}: subsampling wider",c["width_ratio"]>1.2)

    print("\n[T4] tau_int")
    for rho in (0.0,0.8):
        n=400; e=rng.normal(size=n); yy=np.empty(n); yy[0]=e[0]
        for i in range(1,n): yy[i]=rho*yy[i-1]+e[i]
        r=tau_int_ms(yy); th=0.5*(1+rho)/(1-rho)
        print(f"      rho={rho:.1f} tau={r['tau_int']:.2f} (AR1 {th:.2f})")
        rep(f"rho={rho}: within 40% of AR(1)",abs(r["tau_int"]-th)/th<0.40)

    print("\n[T5] CSOC bound + FORMAL no-Zeno certificate")
    c=CSOCAlertCounter(0.6,0.25,0.08,2)
    for k in range(600): c.step(0.35+0.45*abs(math.sin(0.09*k)),k)
    r=c.report()
    rep("N <= (E_max+D)/gap",r["satisfied"],f"N={r['events']} bound={r['bound']:.2f}")
    rep("D == D_audit (ACTUAL removal)",r["D_audit_matches"])
    rep("bound is NOT the theorem","NOT the no-Zeno theorem" in r["note"])
    fc=c.formal_certificate()
    rep("Doob certificate valid",fc["certified"])
    rep("horizon truncation REMOVED","REMOVED" in fc["HORIZON_TRUNCATION"])

    print("\n[T6] MEDICATION LOG BLOCKS")
    ps_ok=synthetic_patient(1,with_meds=True)
    ps_no=synthetic_patient(1,with_meds=False)
    ev_ok=missing_evidence(ps_ok,{"authorized":True})
    ev_no=missing_evidence(ps_no,{"authorized":True})
    rep("complete meds => not blocked on meds",
        not any("MEDICATION LOG" in m for m in ev_ok["missing_blocking"]))
    rep("absent meds => BLOCKED",
        any("MEDICATION LOG" in m for m in ev_no["missing_blocking"]))
    rep("absent meds => study not usable",
        not ev_no["study_usable_for_prediction_research"])
    rep("UPCR method now CHECKED (was hardcoded missing)",
        "UPCR collection method" in ev_ok["present"])

    print("\n[T7] NEW: disease-state occupation")
    st=disease_state_diagnostics(ps_ok)
    print(f"      composition={[round(v,3) for v in st['composition']]}")
    print(f"      regime={st['regime']['regime']}  W2<={st['W2_bound']:.4f}")
    rep("composition on simplex",abs(sum(st["composition"])-1)<1e-6)
    rep("time-ordered => regime test admissible",
        st["regime"]["status"] in (MODE_CERTIFIED,MODE_E))
    rep("BN taxonomy checked","taxonomy_BN" in st)
    rep("interpretation distinguishes bistable","bistable" in st["interpretation"])

    print("\n[T8] NEW: moment gate on heavy-tailed titre")
    f=extract_features(ps_ok,10)
    tail=f["biomarkers"]["anti_dsdna"]["log_titre_tail"]
    rep("log-titre tail analysed",tail is not None)
    if tail:
        print(f"      eta+={tail['eta_up']:.3f} mean finite={tail['first_moment_finite']}")
        rep("quantiles always offered","quantiles" in tail)
    heavy=[KouRegime("heavy",p=.6,eta_up=.7,eta_dn=5.,lam=1.)]
    rh=response_summary(rng.normal(0,.5,40),regimes=heavy)
    rep("eta+<1 mean REFUSED","REFUSED" in rh)

    print("\n[T9] OPMC change points")
    cp=biomarker_changepoints(ps_ok,sd=None)
    rep("change-point analysis runs","K_hat" in cp or cp["status"]==Status.INSUFFICIENT)
    if "K_hat" in cp:
        print(f"      K_hat={cp['K_hat']} status={cp['status']}")
        rep("strict SNR reported honestly","snr_ok" in cp)

    print("\n[T10] end-to-end")
    pipe=SLEPipeline(irb=irb,horizon_weeks=12.,window=8)
    out=pipe.run(ps_ok,deid=deid)
    rep("pipeline runs",out["status"] in (Status.OK,Status.BLOCKED))
    rep("research_only",out["clinical_actionability"]==RESEARCH_ONLY)
    rep("alert bound satisfied",out["alert_bound"]["satisfied"])
    rep("no-Zeno certificate present",out["alert_no_zeno_certificate"]["certified"])
    rr=out["sd_report"]
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    rep("Mode E retained",rr["mode_E"]>0)
    blocked=pipe.run(ps_ok,deid=None)
    rep("without de-id BLOCKED",blocked["status"]==Status.BLOCKED)
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  Slope subsampling uses tau_n = sqrt(S_tt), exact for irregular visits.
  Dead walrus removed. UPCR method is now checked, not hardcoded missing.
  SD gate on every reported statistic.

NEW
  Disease-state fractions as an occupation vector: Thm 8.3 W2 rate and
  the Prop 8.5/8.6 test separating a stationary course from a bistable one.
  Kou moment gate on log-titres; OPMC change points with honest (SNR).
  Doob no-Zeno certificate alongside the engineering counter bound.

UNCHANGED AND CORRECT
  CSOC bound with ACTUAL removal. Medication log BLOCKS the study.

NOT CLAIMED
  No clinical validation; synthetic data only. The score is a DECLARED
  index, not fitted. No diagnosis, treatment, dosing, or taper advice.
""")
    return ok

def main()->int:
    ap=argparse.ArgumentParser(description="SLE longitudinal research v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","coverage","demo","states","rate"])
    ap.add_argument("--rho",type=float,default=0.7)
    ap.add_argument("--n",type=int,default=60)
    ap.add_argument("--reals",type=int,default=300)
    ap.add_argument("--stat",default="mean",choices=["mean","slope"])
    ap.add_argument("--irregular",action="store_true")
    ap.add_argument("--no-meds",action="store_true")
    ap.add_argument("--json",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    out={}
    if a.cmd=="coverage":
        out=coverage_study(a.reals,a.n,a.rho,a.stat,irregular=a.irregular)
    elif a.cmd=="rate":
        t=np.arange(a.n,dtype=float)
        if a.irregular: t=np.cumsum(np.random.default_rng(0).uniform(2.,10.,a.n))
        y=0.2*t+np.random.default_rng(1).normal(0,2,a.n)
        out={"mean":subsample_ci(t,y,"mean"),"slope":subsample_ci(t,y,"slope")}
    elif a.cmd in ("demo","states"):
        irb=IRBRecord("SLE-DEMO","Demo","2026-01-01","2099-12-31",
                      ["sle_research"],data_use_agreement=True,status="APPROVED")
        pipe=SLEPipeline(irb)
        ps=synthetic_patient(3,with_meds=not a.no_meds)
        deid=pipe.gov.phi.deidentify({"age":34,"sex":"female"})
        if a.cmd=="states": out=disease_state_diagnostics(ps)
        else:
            r=pipe.run(ps,deid=deid)
            out={k:v for k,v in r.items() if k!="features"}
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
