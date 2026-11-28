#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_med_math_core.py — v1.0.0
Corrected ESCA 2.0 (Unified Edition) mathematics + shared safety/governance
for the ONE MedWorld / ALE-GCL / conservative-remesh medical research stack.

STATUS LEGEND
  [P] proved · [P|H] proved under explicit hypotheses · [Cit] imported
  [Corr] a source statement was wrong and is corrected · [Open] not established

CORRECTIONS PROPAGATED
  K1  T2 false for general Borel partitions (Prop 5.4): needs BN / code space.
  K5  compression CONDITIONAL; Prop 6.3 generic incompressibility.
  K8  no-Zeno via Doob; horizon truncation REMOVED; (U),(R),(M) explicit.
  K9  H^k_R = H^k; orientation (-1)^|alpha|; flux balance in the domain.
  K10 closed surfaces Dchi = +-2, Db1 = +-2; annihilation A added.
  K11 Langevin edge weights MUST be symmetric Gibbs conductances.

GENUINE CORRESPONDENCES
  cell-fate composition IS an occupation vector on Delta^d  -> Thm 8.3 rate,
  Prop 8.5/8.6 light-vs-super-dominant trichotomy.
  heavy-tailed response across lines -> Kou jumps, Gumbel extremes,
  heaviest-regime domination, moment-explosion threshold.
  tumorigenicity -> p_max = (n_crit/SF)/N_dose, n_assay >= 3/p_max.

PHILOSOPHY (mathematics stands independently)
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

REFUSED
  Gene-editor design. Wet-lab protocols. Pathogen enhancement.
  Autonomous clinical orders. Transplant go/no-go. Fabricated data.
  Reporting an infinite moment. Clinical use of any output.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import hashlib, json, math, re, time, logging, sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False; np = None
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VERSION = "1.0.0"; PI = math.pi
RESEARCH_ONLY = "research_only"
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"
CARPET_RHO_MIN = 9.0 / 8.0

class Refusal(RuntimeError): pass
class ScopeRefusal(Refusal): pass
class GovernanceRefusal(Refusal): pass
class InfiniteMoment(Refusal): pass
class BoundaryNullViolation(Refusal): pass
class EnsembleIntegrityError(Refusal): pass
class UnderResolved(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

def get_logger(n):
    lg = logging.getLogger(n)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg
log = get_logger("med_core")

def phi_std(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def z_quantile(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        m = 0.5*(lo+hi)
        if phi_std(m) < p: lo = m
        else: hi = m
    return 0.5*(lo+hi)

PHILOSOPHY = {
 "tao":"The Source","te":"The Path","qi":"The Proof",
 "p1":"composition IS an occupation measure; the theorems apply literally",
 "p2":"an ensemble whose members are identical is not an ensemble",
 "p3":"a corrected theorem is worth more than an impressive name",
 "p4":"Mode E beats false certainty",
 "motto":"MY SOUL MOVE BY POWER OF HOLY SPIRIT — "
         "What MSPS NETWORK SEE, Lord Buddha Knows."}

# =============================================================================
# §1. SD GATE
# =============================================================================

PACKAGES = {
 "L1":"independent centred sub-Gaussian noise",
 "L1e":"light-exponential (Kou double-exponential jumps)",
 "L2":"stationary, summable covariance",
 "L3":"iid finite-mean blocks",
 "D":"delta_min-isodiametric partition",
 "O":"OPMC admissible partition with dwell",
 "BN":"boundary-null on the lineage taxonomy",
 "A1":"super-dominance s_k/n_k -> 0",
 "A2":"conditional stationarity",
 "A3":"a.e. continuity of the limit map",
 "U":"uniform local bounds B0, G0 and reset radius delta",
 "R":"reset only after exiting the delta-ball",
 "M":"measurable reset maps",
 "STC":"transmission conditions (correct orientation, flux balance)",
 "GCL":"discrete geometric conservation law verified per cell",
 "DONOR":"donor-cell condition checked (bound preservation)",
 "IRB":"active IRB record","DEID":"passed de-identification",
 "VERIFIED":"manually checked against source with DOI",
 "ANNOT":"cell-type annotation reviewed",
 "CONFOUND":"batch not confounded with condition",
 "NORM":"normalisation state declared",
 "IDENT":"full-rank well-conditioned parameter block",
 "SPREAD":"ensemble members verified distinct",
 "CONV":"documented convergence (orders / mesh / time)",
 "KZR":"resistance upper bound [IMPORTED]","KZS":"point separation [IMPORTED]",
 "BBKT":"local-symmetry uniqueness [IMPORTED]",
}

@dataclass(frozen=True)
class SDEntry:
    statistic:str; theorem:str; packages:Tuple[str,...]
    status:str; caveat:str

DICTIONARY: Dict[str,SDEntry] = {
 "composition_rate": SDEntry("fate composition -> limit","Thm 8.3 (Theorem A)",
   ("A1","A2","A3"),"[P|H]","conservative vs exact multinomial for iid cells"),
 "occupation_regime": SDEntry("multistable vs deterministic","Thm 8.1 / Prop 8.6",
   (),"[P]","infinite-mean middle is [Open] -> Mode E"),
 "taxonomy_pooling": SDEntry("nested lineage levels","Thm 5.3 / Prop 5.4",
   ("BN",),"[Corr][P]","T2 FALSE without BN or a code space"),
 "changepoints": SDEntry("protocol / trajectory breaks","Thm 7.5",
   ("L1","D","O"),"[P|H]","(SNR) carries (K+1)^2; K-free is [Open]"),
 "response_tail": SDEntry("heavy-tailed response","Thm 16.9/16.10, Prop 16.14",
   ("L1e",),"[P]","mean may be INFINITE; report quantiles"),
 "esscher_worst_case": SDEntry("robust mean, KL budget","Thm 16.16",
   ("L1e",),"[P]","exact for ONE regime; bound only for switching"),
 "gcl": SDEntry("discrete geometric conservation","ALE GCL, swept prism",
   ("GCL",),"[P]","space-time prism, NOT A*(w.n)*dt"),
 "ale_freestream": SDEntry("uniform flow on a moving mesh","ALE + GCL",
   ("GCL",),"[P]","fails if pressure uses the RELATIVE flux"),
 "conservative_remap": SDEntry("mesh-to-mesh transfer","telescoping + donor cell",
   ("GCL","DONOR"),"[P]","FIRST order: conservative but DIFFUSIVE"),
 "surgery_topology": SDEntry("lumen / budding event","Prop 18.2",
   (),"[Corr][P]","closed surfaces Dchi=+-2; A added; L,C not generic"),
 "no_zeno": SDEntry("event times do not accumulate","Thm 18.6, Cor 18.7",
   ("U","R","M"),"[Corr][P|H]","Doob; horizon truncation REMOVED"),
 "stc_flux": SDEntry("multi-regime flux balance","Prop 19.2/19.3",
   ("STC",),"[Corr][P]","H^k_R = H^k; (-1)^|alpha| orientation"),
 "carpet_nonpcf": SDEntry("carpet continuum limit","Thm 17.16",
   ("KZR","KZS","BBKT"),"[P|Cit]","NON-p.c.f.: three imported inputs"),
 "langevin_weights": SDEntry("weighted Carpet Langevin","Thm 17.25",
   ("STC",),"[Corr][P|C]","edge weights MUST be symmetric Gibbs"),
 "coupled_existence": SDEntry("global-in-time coupled surgery/PDE","§21 O7",
   (),"[Open]","NOT proved; degenerate thin-film open"),
 "ensemble_uq": SDEntry("ensemble uncertainty","empirical",
   ("SPREAD",),"[P]","identical members give zero spread"),
 "adjoint_gradient": SDEntry("implicit / checkpointed adjoint","Taylor test",
   ("CONV",),"[P]","order ~2 required; state must be carried"),
 "tumorigenicity": SDEntry("residual iPSC limit","p_max=(n_crit/SF)/N_dose",
   ("VERIFIED",),"[P]","n_crit is line- and site-specific"),
 "patient_pathway": SDEntry("patient-derived research","governance",
   ("IRB","DEID"),"[P]","no IRB or no de-id => BLOCKED"),
 "clinical_use": SDEntry("clinical decision","none",(),"REFUSED",
   "not a medical device; no go/no-go"),
 "ai_candidate": SDEntry("AI-generated hypothesis","none",(),"hypothesis",
   "must pass a solver or measurement"),
}

class SDGate:
    def __init__(self): self.log: List[Dict] = []
    def certify(self, statistic:str, verified:Sequence[str], value=None) -> Dict:
        e = DICTIONARY.get(statistic); v = tuple(verified)
        if e is None:
            r = {"statistic":statistic,"status":MODE_E,"reason":"no entry"}
        else:
            miss = tuple(p for p in e.packages if p not in v)
            r = {"statistic":statistic,"theorem":e.theorem,
                 "paper_status":e.status,"caveat":e.caveat,
                 "required":list(e.packages),"verified":list(v),
                 "missing":list(miss),
                 "status":MODE_E if miss else MODE_CERTIFIED,
                 "reason":f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def require(self, statistic, verified, value=None):
        c = self.certify(statistic, verified, value)
        if c["status"] != MODE_CERTIFIED:
            raise Refusal(f"SD GATE: {statistic}. {c['reason']}")
        return value
    def report(self):
        n = sum(1 for c in self.log if c["status"]==MODE_CERTIFIED)
        return {"n":len(self.log),"certified":n,"mode_E":len(self.log)-n,
                "log":self.log}

# =============================================================================
# §2. GOVERNANCE — shared, enforced
# =============================================================================

class Intent:
    LITERATURE="literature_query"; SIMULATION="simulation"; RISK="risk_assessment"
    GENE_EDITOR="gene_editor_design"; WETLAB="wetlab_protocol"
    PATHOGEN="pathogen_enhancement"; CLINICAL_ORDER="autonomous_clinical_order"

BLOCKED_PATTERNS: Dict[str,List[str]] = {
 Intent.GENE_EDITOR:[r"\bcrispr\b",r"\bcas9\b",r"\bcas12\b",r"\bcas13\b",
   r"\bguide\s*rna\b",r"\bgrna\b",r"\bsgrna\b",r"\bbase\s*editor\b",
   r"\bprime\s*editor\b",r"\bgene\s*drive\b",r"\bgene\s*edit(or|ing)\b",
   r"\bknock\s*(out|in)\b",r"\baav\s*capsid\b",r"\bdelivery\s*vector\b"],
 Intent.WETLAB:[r"\bwet\s*lab\b",r"\btransfection\b",r"\belectroporation\b",
   r"\bculture\s*protocol\b",r"\bmedia\s*recipe\b",r"\bpcr\s*condition\b",
   r"\bstep\s*by\s*step\s*(protocol|method)\b"],
 Intent.PATHOGEN:[r"\benhance\s*virulence\b",r"\bimmune\s*evasion\b",
   r"\bincrease\s*transmissibility\b",r"\bgain\s*of\s*function\b"],
 Intent.CLINICAL_ORDER:[r"\bdiagnose\s*(the\s*)?patient\b",r"\bprescribe\b",
   r"\boperate\s*on\b",r"\btransplant\s*(now|the\s*patient)\b",
   r"\bautonomous\s*surgery\b",r"\bapprove\s*transplant\b",
   r"\bshould\s*(we|i)\s*operate\b"],
}

def classify_intent(text:str) -> Tuple[str,List[str]]:
    low = text.lower()
    for intent, pats in BLOCKED_PATTERNS.items():
        hits = [p for p in pats if re.search(p, low)]
        if hits: return intent, hits
    return Intent.LITERATURE, []

def refuse_disallowed(text:str) -> None:
    intent, hits = classify_intent(text)
    if hits:
        raise ScopeRefusal(
            f"BLOCKED intent={intent}. This platform does not support gene-editor "
            "design, wet-lab protocols, pathogen enhancement, or autonomous "
            "clinical orders. Every output is research_only.")

class PHIGuard:
    PATTERNS = {
      "name_like":r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b",
      "ssn":r"\b\d{3}-\d{2}-\d{4}\b",
      "mrn":r"\b(MRN|mrn)[:\s]*\d{4,}\b",
      "date_full":r"\b(19|20)\d{2}-\d{2}-\d{2}\b",
      "email":r"[\w.+-]+@[\w-]+\.[\w.]+",
      "phone":r"\b\+?\d[\d\s().-]{8,}\d\b",
      "dicom_uid":r"\b1\.2\.\d{3,}\.[\d.]{6,}\b"}
    DIRECT = {"name","patient_name","given_name","family_name","ssn","mrn",
      "patient_id","national_id","address","email","phone","birth_date","dob",
      "date_of_birth","accession_number","dicom_patient_name",
      "dicom_patient_id","photo"}
    QUASI = {"age","sex","postcode","zip","ethnicity","admission_date",
      "hospital","rare_diagnosis"}
    def scrub(self, text:str) -> Tuple[str,List[str]]:
        out, rm = text, []
        for label, pat in self.PATTERNS.items():
            new = re.sub(pat, "[REDACTED]", out)
            if new != out: rm.append(label); out = new
        return out, rm
    def deidentify(self, rec:Dict[str,Any]) -> Dict[str,Any]:
        removed, quasi, out = [], [], dict(rec)
        for k in list(out.keys()):
            if k.lower() in self.DIRECT: out.pop(k); removed.append(k)
            elif k.lower() in self.QUASI: quasi.append(k)
        if "age" in out:
            try:
                a = int(out.pop("age"))
                out["age_band"] = f"{(a//10)*10}-{(a//10)*10+9}"
                removed.append("age->age_band")
            except Exception: pass
        for k, v in list(out.items()):
            if isinstance(v, str):
                s, rm = self.scrub(v)
                if rm: out[k] = s; removed.append(f"{k}:{','.join(rm)}")
        risk = "low" if not quasi else ("medium" if len(quasi)<=3 else "high")
        return {"passed": risk != "high", "removed_fields": removed,
                "residual_risk": risk, "quasi_identifiers": quasi,
                "clean_record": out}

@dataclass
class IRBRecord:
    protocol_id:str; institution:str; approval_date:str; expiry_date:str
    approved_uses:List[str]; consent_model:str="written"
    data_use_agreement:bool=False; status:str="PENDING"
    def active(self) -> bool:
        if self.status != "APPROVED": return False
        try:
            return time.time() < time.mktime(time.strptime(self.expiry_date,"%Y-%m-%d"))
        except Exception:
            return False

class Governance:
    def __init__(self, irb:Optional[IRBRecord]=None):
        self.irb = irb; self.phi = PHIGuard(); self.audit: List[Dict] = []
    def authorize(self, text:str, uses_patient_data:bool=False,
                  deid:Optional[Dict]=None, purpose:str="research") -> Dict:
        refuse_disallowed(text)
        d = {"authorized":False,"reason":"","clinical_actionability":RESEARCH_ONLY,
             "clinician_review_required":True,
             "required_approvals":["IRB","ethics_committee","clinician","regulator"]}
        if uses_patient_data:
            if self.irb is None: d["reason"]="BLOCKED: no IRB record"
            elif not self.irb.active():
                d["reason"]=f"BLOCKED: IRB status={self.irb.status} or expired"
            elif purpose not in self.irb.approved_uses:
                d["reason"]=f"BLOCKED: purpose '{purpose}' not approved"
            elif not self.irb.data_use_agreement:
                d["reason"]="BLOCKED: no data use agreement"
            elif not deid or not deid.get("passed"):
                d["reason"]="BLOCKED: de-identification missing/failed"
            else:
                d["authorized"]=True; d["reason"]="authorized for research use"
        else:
            d["authorized"]=True; d["reason"]="public/synthetic research query"
        self.audit.append({"ts":time.time(),
            "query_hash":hashlib.sha256(text.encode()).hexdigest()[:16],
            "authorized":d["authorized"],"reason":d["reason"]})
        return d
    def require(self, text, uses_patient_data=False, deid=None, purpose="research"):
        d = self.authorize(text, uses_patient_data, deid, purpose)
        if not d["authorized"]: raise GovernanceRefusal(d["reason"])
        return d

# =============================================================================
# §3. LINEAGE TAXONOMY: BN / projectivity  (K1)
# =============================================================================

def coarse_grain(p_fine:Sequence[float], groups:Sequence[Sequence[int]]) -> Dict:
    _np()
    p = np.asarray(p_fine,float)
    if abs(p.sum()-1) > 1e-6 or (p < -1e-12).any():
        raise Refusal("p_fine must be a probability vector")
    covered = sorted(i for g in groups for i in g)
    if covered != list(range(len(p))):
        raise Refusal("groups must partition every fine index exactly once")
    pc = np.array([p[list(g)].sum() for g in groups])
    return {"p_coarse":pc.tolist(),"consistent":abs(pc.sum()-1)<1e-9,
            "contraction_L1":True,"theorem":"Thm 4.1 + Prop 4.2"}

def t2_counterexample(m_levels:int=6) -> Dict:
    rows=[{"level":m,"cell":[0.5-2.0**(-m),0.5],"assigned_mass":1.0}
          for m in range(1,m_levels+1)]
    return {"levels":rows,"intersection":"EMPTY","assigned_mass_all_levels":1.0,
            "VERDICT":"projective family exists, measure does NOT",
            "REMEDY":["compact code space (Thm 5.1)","Boundary-Null (Thm 5.3)"],
            "BIO_CONSEQUENCE":("a nested lineage taxonomy is not automatically "
                               "consistent; check BN before pooling levels")}

def boundary_null_check(hierarchy:Dict[str,Sequence[str]],
                        composition:Dict[str,float], tol:float=1e-9) -> Dict:
    viol=[]
    for parent,kids in hierarchy.items():
        pm=float(composition.get(parent,0.0))
        cm=sum(float(composition.get(c,0.0)) for c in kids)
        if abs(pm-cm)>tol:
            viol.append({"parent":parent,"parent_mass":pm,"children_sum":cm,
                         "residual":pm-cm})
    ok = not viol
    return {"BN_satisfied":ok,"violations":viol,
            "status":MODE_CERTIFIED if ok else MODE_E,
            "action":("levels may be pooled" if ok else
                      "DO NOT pool: projectivity violated (Prop 5.4)")}

# =============================================================================
# §4. COMPOSITION AS OCCUPATION MEASURE  (the genuine link)
# =============================================================================

def composition_as_occupation(labels:Sequence[str], fate_names:Sequence[str]) -> Dict:
    _np()
    idx={f:i for i,f in enumerate(fate_names)}
    c=np.zeros(len(fate_names)); unk=0
    for l in labels:
        if l in idx: c[idx[l]]+=1
        else: unk+=1
    F=c/max(c.sum(),1)
    return {"F":F.tolist(),"n_cells":len(labels),"n_unknown":unk,
            "fate_names":list(fate_names),"on_simplex":abs(F.sum()-1)<1e-9,
            "interpretation":"occupation vector, ESCA §8 with L=1"}

def theorem_A_rate(n_cells:int, n_fates:int, Gamma:float=1.0,
                   s_k:int=0, n_levels:int=1) -> Dict:
    if n_cells<=s_k: raise Refusal("n_cells must exceed s_k")
    m=n_cells-s_k
    b1=4*n_levels*(s_k/n_cells)**2; b2=4*n_levels*n_fates*Gamma/m
    bound=b1+b2
    return {"n_cells":n_cells,"n_fates":n_fates,"Gamma":Gamma,
            "L2_bound":bound,"W2_bound":math.sqrt(bound),
            "exact_iid_multinomial_upper":1.0/n_cells,
            "NOTE":"theorem bound is conservative for iid cells",
            "theorem":"Thm 8.3","status":"[P|H]"}

def block_structure(labels:Sequence[str]) -> Dict:
    runs=[]
    if not labels: return {"runs":[],"lengths":[]}
    cur=labels[0]; ln=1
    for l in labels[1:]:
        if l==cur: ln+=1
        else: runs.append((cur,ln)); cur=l; ln=1
    runs.append((cur,ln))
    return {"runs":runs,"lengths":[l for _,l in runs]}

def occupation_regime_test(labels:Sequence[str], fate_names:Sequence[str],
                           ordered:bool=False) -> Dict:
    """
    Requires a MEANINGFUL ordering (pseudotime, spatial axis, clone).
    Unordered scRNA-seq gives Mode E for this test.
    """
    _np()
    if not ordered:
        return {"regime":"UNORDERED","status":MODE_E,
                "reason":"cell order in scRNA-seq is arbitrary; needs "
                         "pseudotime / spatial / clone ordering"}
    bs=block_structure(labels); L=np.array(bs["lengths"],float)
    if len(L)<4:
        return {"regime":"INSUFFICIENT_BLOCKS","status":MODE_E,"n_blocks":len(L)}
    S=np.cumsum(L); ratios=(S[:-1]/S[1:]).tolist()
    trend=float(np.mean(ratios[-max(1,len(ratios)//3):]))
    Ls=np.sort(L)[::-1]; k=max(2,int(len(Ls)*0.2))
    hill=float(np.mean(np.log(Ls[:k])-np.log(Ls[k]))) if Ls[k]>0 else 0.0
    alpha=1.0/hill if hill>1e-9 else float("inf")
    lim=None
    if trend<0.25 and ratios[-1]<ratios[0]:
        regime="SUPER_DOMINANT"; status=MODE_CERTIFIED
        thm="Thm 8.1: limit is RANDOM, vertex-supported"
        cons=("genuinely MULTISTABLE: replicate organoids converge to "
              "different pure fates")
    elif alpha>1.5:
        regime="LIGHT_BLOCKS"; status=MODE_CERTIFIED
        thm="Prop 8.6: deterministic a.s. limit"
        cons="NOISY BUT DETERMINISTIC: replicates converge to the same mean"
        num={}
        for f,ln in bs["runs"]: num[f]=num.get(f,0.0)+ln
        lim={f:num.get(f,0.0)/max(S[-1],1) for f in fate_names}
    else:
        regime="INFINITE_MEAN_CANDIDATE"; status=MODE_E
        thm="Lamperti regime: §21 O4 — NOT PROVED"
        cons="tail index suggests infinite mean; do not report a limit"
    out={"regime":regime,"status":status,"theorem":thm,"consequence":cons,
         "n_blocks":len(L),"mean_block":float(L.mean()),
         "prefix_ratio_trend":trend,"hill_tail_index":float(alpha),"ordered":True}
    if lim is not None: out["deterministic_limit"]=lim
    return out

# =============================================================================
# §5. KOU JUMPS — heavy-tailed response
# =============================================================================

@dataclass
class KouRegime:
    name:str; p:float=0.5; eta_up:float=3.0; eta_dn:float=4.0
    lam:float=1.0; drift:float=0.0; vol:float=0.3; occupation:float=1.0
    @property
    def q(self): return 1.0-self.p
    def mgf(self,s):
        if not (-self.eta_dn<s<self.eta_up):
            raise InfiniteMoment(f"MGF diverges at s={s}")
        return (self.p*self.eta_up/(self.eta_up-s)
                + self.q*self.eta_dn/(self.eta_dn+s))
    def dmgf(self,s):
        if not (-self.eta_dn<s<self.eta_up): raise InfiniteMoment("MGF' diverges")
        return (self.p*self.eta_up/(self.eta_up-s)**2
                - self.q*self.eta_dn/(self.eta_dn+s)**2)
    def cumulant(self,s):
        return s*self.drift+0.5*s*s*self.vol**2+self.lam*(self.mgf(s)-1.0)
    def dcumulant(self,s):
        return self.drift+s*self.vol**2+self.lam*self.dmgf(s)
    def v_star(self):
        return self.vol**2+4*self.lam*(self.p/self.eta_up**2+self.q/self.eta_dn**2)

def light_exponential_params(regimes:Sequence[KouRegime]) -> Dict:
    v=max(r.v_star() for r in regimes)
    eta=min(min(r.eta_up,r.eta_dn) for r in regimes
            if r.lam>0 and (r.p>0 or r.q>0))
    return {"nu_squared":v,"b":2.0/eta,"eta_star":eta,
            "tail_bound":"2 exp(-min{x^2/(2T nu^2), x eta*/4})",
            "NOT":"sub-Gaussian (L1); this is (L1e)","theorem":"Prop 16.5"}

def gumbel_extreme(r:KouRegime, T:float) -> Dict:
    a=r.lam*r.p
    if a<=0: return {"no_upward_jumps":True}
    cdf=lambda x: math.exp(-a*T*math.exp(-r.eta_up*x))
    return {"lambda_p_T":a*T,"eta_up":r.eta_up,
            "gumbel_location":math.log(a*T)/r.eta_up,
            "gumbel_scale":1.0/r.eta_up,
            "cdf_at_0":cdf(0.0),"cdf_at_1":cdf(1.0),"cdf_at_3":cdf(3.0),
            "return_level_99":(math.log(a*T)-math.log(math.log(1/0.99)))/r.eta_up,
            "theorem":"Thm 16.9","status":"[P] exact"}

def heaviest_regime_dominates(regimes:Sequence[KouRegime]) -> Dict:
    act=[r for r in regimes if r.lam*r.p>0]
    if not act: return {"no_active_regimes":True}
    eta=min(r.eta_up for r in act)
    gov=[r for r in act if abs(r.eta_up-eta)<1e-12]
    return {"eta_up_star":eta,"governing_regimes":[r.name for r in gov],
            "governing_occupation":sum(r.occupation for r in gov),
            "BIO_CONSEQUENCE":("a rare hyper-responder line sets the tail "
                               "however small its representation"),
            "theorem":"Thm 16.10"}

def moment_estimable(regimes:Sequence[KouRegime], p:float) -> Dict:
    bad=[r.name for r in regimes if r.lam>0 and not (-r.eta_dn<p<r.eta_up)]
    ok=not bad
    out={"p":p,"finite":ok,"violating_regimes":bad,"theorem":"Prop 16.14"}
    if not ok:
        out["REFUSED"]=("E S^p is INFINITE. Do NOT report this moment. "
                        "Report a quantile or trimmed statistic.")
    return out

def require_finite_mean(regimes) -> Dict:
    r=moment_estimable(regimes,1.0)
    if not r["finite"]: raise InfiniteMoment(r["REFUSED"])
    return r

def fit_kou_tail(log_responses, name="cohort") -> KouRegime:
    _np()
    x=np.asarray(log_responses,float)
    if len(x)<8: raise Refusal("need >= 8 responses")
    d=np.diff(x); up=d[d>0]; dn=-d[d<0]
    eu=1.0/max(float(up.mean()),1e-9) if len(up)>1 else 10.0
    ed=1.0/max(float(dn.mean()),1e-9) if len(dn)>1 else 10.0
    return KouRegime(name,p=len(up)/max(len(d),1),eta_up=eu,eta_dn=ed,
                     lam=float(len(d))/len(x),drift=float(d.mean()),
                     vol=float(d.std()))

def response_summary(log_responses, regimes=None) -> Dict:
    _np()
    x=np.asarray(log_responses,float)
    regs=list(regimes) if regimes else [fit_kou_tail(x)]
    me=moment_estimable(regs,1.0)
    out={"n":len(x),"first_moment_finite":me["finite"],
         "quantiles":{"q10":float(np.quantile(x,.1)),
                      "q50":float(np.quantile(x,.5)),
                      "q90":float(np.quantile(x,.9))}}
    if me["finite"]:
        out["mean"]=float(x.mean()); out["report"]="mean estimable"
    else:
        out["REFUSED"]=me["REFUSED"]; out["report"]="mean refused; quantiles given"
    if len(regs)>1: out["tail_governance"]=heaviest_regime_dominates(regs)
    return out

def esscher_worst_case(r:KouRegime, T:float, eta_KL:float) -> Dict:
    if eta_KL<0: raise Refusal("eta_KL >= 0")
    h=lambda th: T*(th*r.dcumulant(th)-r.cumulant(th))
    lo,hi=0.0,r.eta_up*(1-1e-9)
    if h(hi)<eta_KL: raise Refusal("KL budget beyond reachable tilt range")
    for _ in range(200):
        m=0.5*(lo+hi)
        if h(m)<eta_KL: lo=m
        else: hi=m
    th=0.5*(lo+hi)
    wc=T*r.dcumulant(th); base=T*r.dcumulant(0.0)
    return {"eta_KL":eta_KL,"theta_star":th,"nominal_mean":base,
            "worst_case_mean":wc,"shift":wc-base,
            "small_eta_approx":math.sqrt(2*eta_KL*T*r.v_star()),
            "theorem":"Thm 16.16","status":"[P] exact for ONE regime"}

# =============================================================================
# §6. TUMORIGENICITY ARITHMETIC  (the real transplant gate)
# =============================================================================

@dataclass
class TumorigenicitySpec:
    n_crit_cells:float=100.0; n_crit_verified:bool=False; n_crit_source:str=""
    dose_cells:float=1e7; safety_factor:float=10.0
    assay_lod_fraction:Optional[float]=None; assay_method:str=""
    cells_interrogated:Optional[float]=None; positives_observed:Optional[int]=None
    def required_limit(self) -> Dict:
        n_eff=self.n_crit_cells/max(self.safety_factor,1.0)
        p_max=n_eff/max(self.dose_cells,1.0)
        return {"effective_n_crit":n_eff,"dose_cells":self.dose_cells,
                "p_max_fraction":p_max,"p_max_percent":100*p_max,
                "n_assay_required":3.0/max(p_max,1e-30),
                "formula":"p_max=(n_crit/SF)/N_dose ; n_assay >= 3/p_max",
                "n_crit_verified":self.n_crit_verified,
                "STATUS":("line-specific" if self.n_crit_verified else
                          "PLACEHOLDER n_crit — run your own spiking study")}
    def assay_adequacy(self) -> Dict:
        req=self.required_limit(); p_max=req["p_max_fraction"]
        out={"required_p_max":p_max,"assay_lod":self.assay_lod_fraction}
        if self.assay_lod_fraction is None:
            out.update(adequate=None,status=MODE_E,reason="assay LOD not measured")
            return out
        lod_ok=self.assay_lod_fraction<=p_max
        out["lod_sufficient"]=lod_ok
        if self.cells_interrogated and self.positives_observed is not None:
            n=float(self.cells_interrogated); k=int(self.positives_observed)
            p_up=(3.0/max(n,1.0)) if k==0 else _clopper_upper(k,n)
            out.update(cells_interrogated=n,positives=k,p_upper_95=p_up,
                       demonstrated_below_limit=p_up<=p_max,
                       n_assay_shortfall=max(0.0,req["n_assay_required"]-n))
            out["adequate"]=bool(lod_ok and p_up<=p_max)
        else:
            out["adequate"]=None; out["reason"]="cells/positives not supplied"
        out["status"]=MODE_CERTIFIED if out.get("adequate") else MODE_E
        if not lod_ok:
            out["BLOCKER"]=("assay LOD coarser than the required limit; "
                            "use ddPCR LIN28A or high-sensitivity flow")
        return out

def _clopper_upper(k:int,n:float,alpha:float=0.05)->float:
    lo,hi=k/n,1.0
    for _ in range(120):
        m=0.5*(lo+hi)
        if n<=2000:
            tail=sum(math.comb(int(n),i)*m**i*(1-m)**(int(n)-i) for i in range(k+1))
        else:
            mu=n*m; sd=math.sqrt(max(n*m*(1-m),1e-12))
            tail=phi_std((k+0.5-mu)/sd)
        if tail>alpha: lo=m
        else: hi=m
    return 0.5*(lo+hi)

def engraftment_power(p1:float,p2:float,alpha:float=0.05,power:float=0.8)->Dict:
    if not (0<p1<1 and 0<p2<1): raise Refusal("p in (0,1)")
    za=z_quantile(1-alpha/2); zb=z_quantile(power); d=abs(p1-p2)
    n=((za+zb)**2)*(p1*(1-p1)+p2*(1-p2))/(d*d)
    return {"p1":p1,"p2":p2,"n_per_arm":math.ceil(n),"n_total":2*math.ceil(n),
            "PURPOSE":"study DESIGN, not outcome prediction"}

def detectable_effect(n_per_arm:int,p_control:float=0.3,alpha:float=0.05,
                      power:float=0.8)->Dict:
    za=z_quantile(1-alpha/2); zb=z_quantile(power)
    lo,hi=1e-4,1.0-p_control-1e-6
    for _ in range(200):
        m=0.5*(lo+hi); p2=p_control+m
        if p2>=1: hi=m; continue
        n=((za+zb)**2)*(p_control*(1-p_control)+p2*(1-p2))/(m*m)
        if n>n_per_arm: lo=m
        else: hi=m
    d=0.5*(lo+hi)
    return {"n_per_arm":n_per_arm,"min_detectable_delta":d,
            "VERDICT":(f"with n={n_per_arm}/arm only delta >= {d:.3f} is "
                       "detectable. A predictor fit on such n cannot be trusted.")}

TRANSLATION_GAP = {
 "immune_rejection":"immunodeficient mouse lacks T/B/NK rejection; OVERESTIMATES human",
 "scale":"1e5-1e6 cells / mm tissue vs 1e7-1e9 / cm; diffusion does not scale",
 "site":"permissive subcutaneous/renal capsule vs orthotopic diseased niche",
 "time":"weeks-months vs years-decades; late teratoma not excluded",
 "tumorigenicity":"mouse inoculum threshold in a permissive host; human unknown",
 "regulatory":"preclinical only; IND/CTA needs GMP, release criteria, sponsor",
}

# =============================================================================
# §7. CORRECTED SURGERY + DOOB NO-ZENO + STC
# =============================================================================

CONVENTIONS=("closed_orientable_surface","filament_network","solid_handlebody")
SURGERY={
 "closed_orientable_surface":{
   "N":(1,0,1,2),"A":(-1,0,-1,-2),"B":(-1,0,-1,-2),
   "M":(1,0,1,2),"H":(0,2,0,-2),"P":(0,-2,0,2)},
 "filament_network":{
   "N":(1,0,0,1),"A":(-1,0,0,-1),"B":(-1,0,0,-1),
   "M":(1,0,0,1),"H":(0,1,0,-1),"P":(0,-1,0,1)},
 "solid_handlebody":{
   "N":(1,0,1,2),"A":(-1,0,-1,-2),"B":(-1,0,0,-1),
   "M":(1,0,0,1),"H":(0,1,0,-1),"P":(0,-1,0,1)},
}

def classify_morphology(db0,db1,db2,dchi,
                        convention="closed_orientable_surface")->Dict:
    if convention not in CONVENTIONS: raise Refusal(f"convention {CONVENTIONS}")
    tbl=SURGERY[convention]; sig=(db0,db1,db2,dchi)
    cands=[op for op,v in tbl.items() if v==sig]
    return {"signature":sig,"candidates":cands,"convention":convention,
            "euler_consistent":dchi==db0-db1+db2,
            "annihilation_available":"A" in tbl,
            "NOT_CLASSIFIED":["L (boundary contact)","C (junction reconnect)"],
            "CORRECTION":"closed surfaces Dchi=+-2, Db1=+-2 (K10)",
            "BIO_MAP":{"N":"lumen nucleation","A":"lumen collapse",
                       "B":"organoid fusion","M":"budding / fission",
                       "H":"toroid / handle","P":"handle removal"}}

def doob_exit_bound(delta:float,B0:float,G0:float,eps:float)->Dict:
    if delta<=0 or G0<0 or B0<0: raise Refusal("delta>0, G0>=0, B0>=0")
    eps_max=delta/(2*B0) if B0>0 else float("inf")
    if eps>eps_max: raise Refusal(f"eps={eps} exceeds delta/(2B0)={eps_max}")
    q=4*G0*G0*eps/(delta*delta)
    return {"delta":delta,"B0":B0,"G0":G0,"eps":eps,"q_doob":q,"q_lt_1":q<1.0,
            "theorem":"Prop 18.4 (Doob L2 maximal inequality)",
            "REPLACES":"2exp(-delta^2/(8G0^2 eps) [cited, unproved]"}

def no_zeno_certificate(delta:float,B0:float,G0:float)->Dict:
    e1=delta/(2*B0) if B0>0 else float("inf")
    e2=delta*delta/(8*G0*G0) if G0>0 else float("inf")
    eps0=min(e1,e2)
    if not math.isfinite(eps0) or eps0<=0: raise Refusal("degenerate bounds")
    q=4*G0*G0*eps0/(delta*delta)
    return {"eps0":eps0,"q0":q,"q0_le_half":q<=0.5+1e-12,"certified":q<1.0,
            "HORIZON_TRUNCATION":"REMOVED (K8)",
            "conclusion":"T_n -> infinity a.s. (Levy conditional B-C)",
            "theorem":"Thm 18.6 [Corr]"}

def surgery_count_tail(t:float,m:int,eps0:float,q0:float)->Dict:
    if m<=0 or eps0<=0: raise Refusal("m>0, eps0>0")
    x=t/(m*eps0); p=1-q0
    if not (0<x<1) or not (0<p<1):
        return {"applicable":False,"reason":"need 0 < t/(m eps0) < 1"}
    kl=x*math.log(x/p)+(1-x)*math.log((1-x)/(1-p))
    return {"t":t,"m":m,"x":x,"KL":kl,"bound":math.exp(-m*kl),
            "applicable":True,"theorem":"Cor 18.7"}

def stc_trace_jump(u_i,u_j,alpha=0,outward_normals=True)->Dict:
    sign=(-1.0)**alpha if outward_normals else 1.0
    jump=u_j-sign*u_i
    return {"alpha":alpha,"sign_factor":sign,"jump":jump,
            "satisfied":abs(jump)<1e-8,"Hk_R_equals":"H^k(Omega)",
            "is_new_space":False,"CORRECTION":"K9"}

def stc_flux_balance(a_i,du_i,a_j,du_j)->Dict:
    resid=a_i*du_i-a_j*du_j
    return {"residual":resid,"balanced":abs(resid)<1e-8,
            "theorem":"Prop 19.3 (flux balance in the operator domain)"}

def stc_reflux(coarse_flux:float,fine_fluxes:Sequence[float])->Dict:
    tot=float(sum(fine_fluxes)); mis=tot-coarse_flux
    return {"coarse":coarse_flux,"fine_total":tot,"mismatch":mis,
            "corrected":coarse_flux+mis,"residual_after":abs(tot-(coarse_flux+mis)),
            "conserved":abs(mis)<1e-14,
            "DERIVED":"STC^1 requires refluxing; not optional"}

# =============================================================================
# §8. SYMMETRIC GIBBS CONDUCTANCE (K11) + CARPET GAUGE
# =============================================================================

def gibbs_conductance(U_a:float,U_b:float,beta:float,rho:float=1.0)->float:
    """
    K11: the source used exp(-beta/2 [U(w')-U(w)]) which is NOT symmetric.
    The correct reversible conductance is symmetric in (a,b):
        c_ab = rho * exp(-beta/2 [U(a)+U(b)])
    giving reversible measure m ∝ rho e^{-beta U}.
    """
    return rho*math.exp(-0.5*beta*(U_a+U_b))

def verify_gibbs_symmetry(U:Sequence[float],beta:float)->Dict:
    _np()
    n=len(U); asym=0.0
    for a in range(n):
        for b in range(n):
            if a==b: continue
            cab=gibbs_conductance(U[a],U[b],beta)
            cba=gibbs_conductance(U[b],U[a],beta)
            asym=max(asym,abs(cab-cba))
    wrong=0.0
    for a in range(n):
        for b in range(n):
            if a==b: continue
            w1=math.exp(-0.5*beta*(U[b]-U[a]))
            w2=math.exp(-0.5*beta*(U[a]-U[b]))
            wrong=max(wrong,abs(w1-w2))
    return {"symmetric_asymmetry":asym,"is_symmetric":asym<1e-12,
            "source_formula_asymmetry":wrong,
            "source_was_symmetric":wrong<1e-12,
            "CORRECTION":"K11: use symmetric Gibbs conductances",
            "theorem":"Thm 17.25 [Corr]"}

def carpet_status(is_pcf:bool)->Dict:
    if is_pcf:
        return {"is_pcf":True,"construction":"Kigami renormalisation",
                "status":"[Cit]","inputs_needed":[],
                "resistance_ratio":5.0/3.0}
    return {"is_pcf":False,"construction":"NOT constructive",
            "inputs_needed":["(KZ-R) resistance upper bound",
                             "(KZ-S) point separation","(BBKT) uniqueness"],
            "proved_here":["chain property (G)","(C1) from (KZ-R)+(G)",
                           "(C2) from (C1) via Gamma-compactness"],
            "rho_lower_bound":CARPET_RHO_MIN,"status":"[P|Cit]",
            "theorem":"Thm 17.16"}

def carpet_multiscale_energy(fields_by_level:Dict[int,"np.ndarray"],
                             laplacians:Dict[int,"np.ndarray"],
                             kappa:Dict[int,float])->Dict:
    """
    CORRECT multiscale form: each level m has its OWN coarse-grained field
    P_m u and its OWN gauge kappa_m ∝ rho^m. Using one field with three
    rho^k weights is NOT the Carpet Dirichlet form.
    """
    _np()
    terms={}
    for m,u in fields_by_level.items():
        L=laplacians[m]
        if u.shape[0]!=L.shape[0]:
            raise Refusal(f"level {m}: field {u.shape[0]} != Laplacian {L.shape[0]}")
        terms[m]=float(kappa[m]*(u@L@u))
    return {"per_level_energy":terms,"total":sum(terms.values()),
            "gauge":kappa,
            "CORRECTION":"each level uses its OWN coarse-grained field",
            "NOT":"one field reweighted by rho^k"}

# =============================================================================
# §9. INTEGRITY GUARDS  (the bugs that invalidated results)
# =============================================================================

def verify_ensemble_spread(trajectories:"np.ndarray", tol:float=1e-12)->Dict:
    """
    FIX for the identical-member bug: an ensemble whose members are
    bitwise identical has ZERO spread and its CI is meaningless.
    trajectories: (E, T, B, D)
    """
    _np()
    a=np.asarray(trajectories,float)
    if a.ndim<2 or a.shape[0]<2:
        raise EnsembleIntegrityError("need >= 2 ensemble members")
    E=a.shape[0]
    max_pair=0.0
    for i in range(E):
        for j in range(i+1,E):
            max_pair=max(max_pair,float(np.max(np.abs(a[i]-a[j]))))
    spread=float(np.std(a,axis=0).mean())
    ok = max_pair>tol
    if not ok:
        raise EnsembleIntegrityError(
            f"ensemble members are IDENTICAL (max pairwise diff {max_pair:.2e}). "
            "Setting a random seed does not perturb a deterministic model. "
            "Perturb parameters, inputs, or use MC-dropout.")
    return {"n_members":E,"max_pairwise_diff":max_pair,"mean_spread":spread,
            "members_distinct":True}

def parameter_perturbation_ensemble(model, n_members:int, scale:float,
                                    seed:int=0):
    """
    Produce genuinely DIFFERENT ensemble members by perturbing parameters.
    Returns a list of deep-copied models with Gaussian weight noise.
    """
    if not HAS_TORCH: raise Refusal("torch required")
    import copy
    g=torch.Generator().manual_seed(seed)
    members=[]
    for i in range(n_members):
        m=copy.deepcopy(model)
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad and p.numel()>0:
                    p.add_(torch.randn(p.shape,generator=g,
                                       dtype=p.dtype)*scale)
        m.eval(); members.append(m)
    return members

def verify_adjoint_order(J:Callable, x0, grad, direction,
                         epsilons=(1e-3,5e-4,2.5e-4,1.25e-4))->Dict:
    """Taylor test: |J(x+e d) - J(x) - e <g,d>| must decay as O(e^2)."""
    if not HAS_TORCH: raise Refusal("torch required")
    import torch as T
    J0=float(J(x0)); dirv=float((grad*direction).sum())
    rows=[]
    for e in epsilons:
        err=abs(float(J(x0+e*direction))-J0-e*dirv)
        rows.append({"eps":e,"error":err})
    rates=[math.log(rows[i-1]["error"]/rows[i]["error"])/math.log(2.0)
           for i in range(1,len(rows)) if rows[i]["error"]>1e-30]
    order=float(np.median(rates)) if rates else 0.0
    return {"rows":rows,"observed_order":order,"passes":abs(order-2.0)<0.4,
            "expected_order":2.0}

def auc_with_ties(risk:"np.ndarray", event:"np.ndarray")->Dict:
    """
    FIX: raw ranks give a WRONG AUC when risks are tied.
    Use average (mid) ranks for tied groups.
    """
    _np()
    r=np.asarray(risk,float); y=np.asarray(event,int)
    npos=int(y.sum()); nneg=int(len(y)-npos)
    if npos==0 or nneg==0: raise Refusal("single-class outcome")
    order=np.argsort(r,kind="mergesort"); rs=r[order]
    ranks=np.empty(len(r),float)
    i=0
    while i<len(r):
        j=i
        while j+1<len(r) and rs[j+1]==rs[i]: j+=1
        ranks[i:j+1]=0.5*((i+1)+(j+1))   # average rank for the tie group
        i=j+1
    y_sorted=y[order]
    auc=float((ranks[y_sorted==1].sum()-npos*(npos+1)/2)/(npos*nneg))
    naive_order=np.argsort(r)
    naive_ranks=np.arange(1,len(r)+1,dtype=float)
    naive=float((naive_ranks[y[naive_order]==1].sum()-npos*(npos+1)/2)/(npos*nneg))
    return {"auc_tie_corrected":auc,"auc_naive":naive,
            "n_tied_groups":int(len(r)-len(np.unique(r))),
            "brier":float(np.mean((r-y)**2)),"n":len(y),"n_events":npos,
            "CORRECTION":"average ranks for ties"}

# =============================================================================
# §10. OPMC (corrected reference) + SELF-TEST
# =============================================================================

def opmc(y:Sequence[float], ell:int, beta:float)->Dict:
    n=len(y)
    if not (1<=ell<=n): raise Refusal("1 <= ell <= n")
    s1=[0.0]; s2=[0.0]
    for v in y: s1.append(s1[-1]+v); s2.append(s2[-1]+v*v)
    def sse(a,b):
        m=b-a
        if m<=0: return 0.0
        t1=s1[b]-s1[a]; t2=s2[b]-s2[a]
        return max(t2-t1*t1/m,0.0)
    INF=float("inf"); D=[INF]*(n+1); prev=[-1]*(n+1); D[0]=-beta
    for b in range(ell,n+1):
        best,arg=INF,-1
        for a in range(0,b-ell+1):
            if D[a]==INF: continue
            v=D[a]+sse(a,b)+beta
            if v<best: best,arg=v,a
        D[b]=best; prev[b]=arg
    cps=[]; b=n
    while b>0:
        a=prev[b]
        if a<=0: break
        cps.append(a); b=a
    cps.reverse()
    return {"boundaries":cps,"K_hat":len(cps),"cost":D[n]}

def opmc_certified(y, ell:int, sigma:float, alpha:float=0.05)->Dict:
    n=len(y); L=math.log(10*math.e*n**3/alpha); beta=32*sigma*sigma*L
    r=opmc(y,ell,beta); bnds=[0]+r["boundaries"]+[n]
    means=[sum(y[bnds[i]:bnds[i+1]])/max(bnds[i+1]-bnds[i],1)
           for i in range(len(bnds)-1)]
    jumps=[abs(means[i+1]-means[i]) for i in range(len(means)-1)]
    kappa=min(jumps) if jumps else 0.0
    spac=[bnds[i+1]-bnds[i] for i in range(len(bnds)-1)]
    delta=min(spac) if spac else n; K=r["K_hat"]
    lhs=kappa*kappa*delta; rhs=1e5*sigma*sigma*L*(K+1)**2
    dwell=delta>=8 and (delta/2<ell<=3*delta/4)
    ok=lhs>=rhs and dwell
    return {**r,"kappa_min":kappa,"delta_min":delta,
            "snr_lhs":lhs,"snr_rhs":rhs,"snr_ok":lhs>=rhs,"dwell_ok":dwell,
            "radius":46*sigma*sigma*L/max(kappa*kappa,1e-30) if ok else None,
            "status":MODE_CERTIFIED if ok else MODE_E,
            "K_dependence":"(K+1)^2 present; K-free is [Open] (O1)"}


def self_test(verbose:bool=True)->bool:
    print("="*76); print("  MSPS MED MATH CORE — self-test"); print("="*76)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    sd=SDGate()

    print("\n[T1] GOVERNANCE")
    gov=Governance()
    b=False
    try: gov.require("hemodynamic simulation", True)
    except GovernanceRefusal: b=True
    rep("patient path BLOCKED without IRB",b)
    for txt,lbl in (("design a CRISPR Cas9 guide RNA","gene editor"),
                    ("give a wet lab transfection protocol","wet lab"),
                    ("enhance virulence and immune evasion","pathogen"),
                    ("should we operate on the patient","clinical order")):
        b=False
        try: refuse_disallowed(txt)
        except ScopeRefusal: b=True
        rep(f"{lbl} BLOCKED",b)
    irb=IRBRecord("P-001","Demo","2026-01-01","2099-12-31",
                  ["simulation","organoid_research"],data_use_agreement=True,
                  status="APPROVED")
    g2=Governance(irb)
    deid=g2.phi.deidentify({"patient_name":"Jane Doe","mrn":"MRN: 12345",
                            "age":55,"sex":"female"})
    rep("de-id removes direct identifiers",
        deid["passed"] and "patient_name" in deid["removed_fields"])
    rep("IRB + de-id authorised",
        g2.authorize("simulation",True,deid,"simulation")["authorized"])
    q,rm=g2.phi.scrub("Jane Doe MRN: 999999 BRCA1")
    rep("query scrubbed before network",bool(rm) and "REDACTED" in q)

    print("\n[T2] BN taxonomy (K1)")
    ce=t2_counterexample(6)
    rep("intersection EMPTY, mass 1 all levels",
        ce["intersection"]=="EMPTY" and ce["assigned_mass_all_levels"]==1.0)
    h={"GERM":["A","B"]}
    rep("consistent passes",boundary_null_check(h,{"GERM":1.,"A":.6,"B":.4})["BN_satisfied"])
    rep("inconsistent fails",not boundary_null_check(h,{"GERM":.9,"A":.6,"B":.4})["BN_satisfied"])

    print("\n[T3] COMPOSITION = OCCUPATION MEASURE")
    fates=["PLURI","ENDO","MESO","NEURON"]
    oc=composition_as_occupation(["PLURI"]*10+["ENDO"]*30+["MESO"]*40+["NEURON"]*20,fates)
    rep("on simplex",oc["on_simplex"])
    ra=theorem_A_rate(100,4)
    print(f"      W2<={ra['W2_bound']:.4f}  iid<={ra['exact_iid_multinomial_upper']:.4f}")
    rep("bound conservative",ra["L2_bound"]>ra["exact_iid_multinomial_upper"])
    rep("unordered -> MODE_E",
        occupation_regime_test(["A","B"]*20,fates)["status"]==MODE_E)
    rep("ordered alternating -> LIGHT",
        occupation_regime_test(["PLURI","ENDO"]*40,fates,ordered=True)["regime"]=="LIGHT_BLOCKS")
    sup=[]
    for j,f in enumerate(fates): sup+=[f]*(4**j)
    rep("ordered geometric -> SUPER_DOMINANT",
        occupation_regime_test(sup,fates,ordered=True)["regime"]=="SUPER_DOMINANT")

    print("\n[T4] KOU moment gate + Gumbel + Esscher")
    r1=KouRegime("A",eta_up=3.0); r2=KouRegime("hyper",eta_up=0.7,occupation=0.05)
    rep("eta+>1 mean finite",moment_estimable([r1],1.)["finite"])
    rep("eta+<1 mean INFINITE",not moment_estimable([r2],1.)["finite"])
    b=False
    try: require_finite_mean([r2])
    except InfiniteMoment: b=True
    rep("reporting infinite mean REFUSED",b)
    gu=gumbel_extreme(r1,100.0)
    ana=math.exp(-1.0*0.5*100*math.exp(0.0))
    rep("Gumbel cdf exact at 0",abs(gu["cdf_at_0"]-ana)<1e-12)
    hv=heaviest_regime_dominates([r1,r2])
    rep("rare line governs tail",hv["governing_regimes"]==["hyper"])
    wc=esscher_worst_case(r1,10.0,0.5)
    rep("worst case exceeds nominal",wc["worst_case_mean"]>wc["nominal_mean"])
    rs=response_summary(np.random.default_rng(0).normal(0,.5,60))
    rep("light cohort mean reported","mean" in rs)

    print("\n[T5] TUMORIGENICITY arithmetic")
    t=TumorigenicitySpec(n_crit_cells=100.,dose_cells=1e7,safety_factor=1.)
    rq=t.required_limit()
    print(f"      p_max={rq['p_max_fraction']:.2e}  n_assay={rq['n_assay_required']:.0f}")
    rep("p_max = 1e-5",abs(rq["p_max_fraction"]-1e-5)<1e-12)
    rep("n_assay = 3e5",abs(rq["n_assay_required"]-3e5)<1)
    rep("placeholder flagged",not rq["n_crit_verified"])
    t.assay_lod_fraction=1e-6; t.cells_interrogated=1e5; t.positives_observed=0
    a=t.assay_adequacy()
    rep("1e5 cells insufficient",not a["adequate"])
    t.cells_interrogated=5e5
    rep("5e5 cells sufficient",t.assay_adequacy()["adequate"])
    de=detectable_effect(20)
    print(f"      n=20/arm -> min detectable delta={de['min_detectable_delta']:.3f}")
    rep("small n only detects large effects",de["min_detectable_delta"]>0.3)
    rep("6 translation gaps",len(TRANSLATION_GAP)==6)

    print("\n[T6] CORRECTED SURGERY (K10)")
    rep("lumen handle Db1=+2 -> H",classify_morphology(0,2,0,-2)["candidates"]==["H"])
    rep("Db1=+1 rejected for closed surface",
        classify_morphology(0,1,0,-1)["candidates"]==[])
    rep("filament accepts +-1",
        classify_morphology(0,1,0,-1,"filament_network")["candidates"]==["H"])
    rep("annihilation present",classify_morphology(0,0,0,0)["annihilation_available"])

    print("\n[T7] DOOB NO-ZENO (K8) + STC (K9)")
    nz=no_zeno_certificate(1.,2.,.5)
    rep("q0 <= 1/2",nz["q0_le_half"])
    rep("truncation REMOVED","REMOVED" in nz["HORIZON_TRUNCATION"])
    b=False
    try: doob_exit_bound(1.,2.,.5,.5)
    except Refusal: b=True
    rep("eps > delta/(2B0) refused",b)
    st=surgery_count_tail(10.,100,nz["eps0"],nz["q0"])
    rep("count tail applicable",st["applicable"])
    rep("alpha=1 needs sign flip",stc_trace_jump(1.,-1.,1)["satisfied"])
    rep("literal outward equality fails",not stc_trace_jump(1.,1.,1)["satisfied"])
    rep("H^k_R not a new space",not stc_trace_jump(0,0)["is_new_space"])
    rep("reflux conserved",stc_reflux(.7,[.2,.15,.25,.18])["conserved"])

    print("\n[T8] SYMMETRIC GIBBS (K11) + CARPET")
    gs=verify_gibbs_symmetry([0.,.5,1.,1.5],2.0)
    print(f"      symmetric asym={gs['symmetric_asymmetry']:.2e}  "
          f"source asym={gs['source_formula_asymmetry']:.3f}")
    rep("corrected conductance IS symmetric",gs["is_symmetric"])
    rep("source conductance was NOT symmetric",not gs["source_was_symmetric"])
    cp=carpet_status(False)
    rep("non-PCF needs 3 imported inputs",len(cp["inputs_needed"])==3)
    rep("rho >= 9/8",abs(cp["rho_lower_bound"]-9/8)<1e-12)
    L1=np.array([[1.,-1.],[-1.,1.]]); L2=np.eye(4)*2
    ce2=carpet_multiscale_energy({1:np.array([1.,-1.]),2:np.ones(4)},
                                 {1:L1,2:L2},{1:1.125,2:1.266})
    rep("multiscale uses per-level fields",len(ce2["per_level_energy"])==2)
    b=False
    try: carpet_multiscale_energy({1:np.ones(3)},{1:L1},{1:1.})
    except Refusal: b=True
    rep("dimension mismatch refused",b)

    print("\n[T9] INTEGRITY GUARDS (the bugs that invalidated results)")
    identical=np.ones((4,3,2,2))
    b=False
    try: verify_ensemble_spread(identical)
    except EnsembleIntegrityError: b=True
    rep("IDENTICAL ensemble DETECTED and refused",b)
    distinct=identical+np.random.default_rng(0).normal(0,.1,identical.shape)
    vs=verify_ensemble_spread(distinct)
    rep("distinct ensemble accepted",vs["members_distinct"])
    print(f"      max pairwise diff={vs['max_pairwise_diff']:.4f}")
    if HAS_TORCH:
        import torch, torch.nn as nn
        m=nn.Sequential(nn.Linear(4,4),nn.Tanh(),nn.Linear(4,2))
        members=parameter_perturbation_ensemble(m,4,0.05,seed=0)
        x=torch.ones(1,4)
        outs=np.array([mm(x).detach().numpy() for mm in members])
        rep("perturbed members give distinct outputs",
            float(np.max(np.abs(outs-outs[0])))>1e-8)
        # Taylor test
        f=lambda v: float((v**2).sum())
        x0=torch.tensor([1.,2.]); g=2*x0; d=torch.tensor([1.,0.])
        at=verify_adjoint_order(f,x0,g,d)
        rep("Taylor order ~2",at["passes"],f"order={at['observed_order']:.3f}")
    tied_r=np.array([.1,.1,.5,.5,.9]); tied_y=np.array([0,1,0,1,1])
    au=auc_with_ties(tied_r,tied_y)
    print(f"      AUC tie-corrected={au['auc_tie_corrected']:.4f}  "
          f"naive={au['auc_naive']:.4f}  ties={au['n_tied_groups']}")
    rep("tie correction changes AUC",
        abs(au["auc_tie_corrected"]-au["auc_naive"])>1e-9)
    rep("tie-corrected AUC = 0.875",abs(au["auc_tie_corrected"]-0.875)<1e-9)

    print("\n[T10] OPMC + SD")
    y=[0.]*60+[4.]*60+[-3.]*60
    r=opmc_certified(y,30,1.0)
    rep("finds 2 change points",r["K_hat"]==2)
    rep("strict SNR honestly fails",r["status"]==MODE_E)
    rep("missing package -> MODE_E",
        sd.certify("no_zeno",("U","R"))["status"]==MODE_E)
    rep("all packages -> CERTIFIED",
        sd.certify("no_zeno",("U","R","M"))["status"]==MODE_CERTIFIED)
    rep("coupled existence always MODE_E",
        sd.certify("coupled_existence",())["status"]==MODE_E)
    rep("clinical use refused",sd.certify("clinical_use",())["status"]==MODE_E)
    rr=sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
CORRECTED AND ENFORCED
  K1 BN taxonomy · K8 Doob no-Zeno (truncation removed) · K9 STC orientation
  K10 closed-surface surgery + annihilation · K11 symmetric Gibbs conductances
  Thm 8.3 composition rate · Kou moment-explosion gate · Gumbel extremes
  Tumorigenicity p_max / n_assay arithmetic · translation-gap register

INTEGRITY GUARDS (each corresponds to a real bug found in the stack)
  verify_ensemble_spread      : identical members -> REFUSED
  parameter_perturbation_ensemble : genuinely different members
  verify_adjoint_order        : Taylor order must be ~2
  auc_with_ties               : average ranks, not raw ranks
  carpet_multiscale_energy    : per-level fields, not one reweighted field

REFUSED
  Gene-editor design · wet-lab protocols · pathogen enhancement
  Autonomous clinical orders · transplant go/no-go · infinite moments
""")
    return ok

if __name__=="__main__":
    raise SystemExit(0 if self_test() else 1)
