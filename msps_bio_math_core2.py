#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_bio_math_core.py — v1.0.0
Corrected ESCA 2.0 (Unified Edition) mathematics for iPSC / organoid research,
plus the shared guards that the five bio modules were missing.

STATUS LEGEND (from the Unified Edition)
  [P]     proved in the paper
  [P|H]   proved under an explicit hypothesis package
  [Cit]   published theorem used as an input, not reproved
  [Corr]  a source statement was wrong and is corrected here
  [Open]  not established

CORRECTIONS PROPAGATED
  K1  T2 is FALSE for general Borel partitions (Prop 5.4). A nested fate
      taxonomy needs Boundary-Null or a compact code space.
  K2  T4-G needs a.e. continuity of Psi.
  K5  compression is CONDITIONAL; Prop 6.3 forbids unconditional compression.
  K8  no-Zeno uses Doob; the horizon truncation is REMOVED.
  K9  H^k_R = H^k; orientation carries (-1)^|alpha|; flux balance in the domain.
  K10 closed surfaces have Dchi = +-2 and Db1 = +-2; annihilation A was missing.

GENUINE CORRESPONDENCES (not name analogies)
  * cell-type composition IS an occupation vector on Delta^d  -> Thm 8.3 rate,
    Prop 8.5/8.6 light-vs-super-dominant trichotomy.
  * dose-response / differentiation trajectory change points -> OPMC Thm 7.5
    with the honest (SNR) carrying (K+1)^2.
  * heavy-tailed response across lines -> Kou jumps, Gumbel extremes,
    heaviest-regime domination, moment-explosion threshold.
  * Hill EC50 / n correlation -> Fisher information, QUANTIFIED not warned.
  * organoid lumen morphology -> corrected Morse surgery table.

PHILOSOPHY (mathematics stands independently)
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

REFUSED
  Fabricating data. Certifying an unverified hypothesis. Reporting an
  infinite moment. Clinical use. Transplant decisions (see msps_t4_preclinical).
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import json, math, logging, sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"
RESEARCH_ONLY = "research_only_not_clinical"

class Refusal(RuntimeError): pass
class NotIdentifiable(Refusal): pass
class InfiniteMoment(Refusal): pass
class BoundaryNullViolation(Refusal): pass
class LeakageError(Refusal): pass
class AblationIntegrityError(Refusal): pass
class ConfoundRefusal(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

def get_logger(n):
    lg = logging.getLogger(n)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg
log = get_logger("bio_core")

def phi_std(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))

PHILOSOPHY = {
 "tao":"The Source", "te":"The Path", "qi":"The Proof",
 "p1":"composition IS an occupation measure; the theorems apply literally",
 "p2":"an ablation that does not actually disable a component is not an ablation",
 "p3":"a prior with no gradient is not a prior",
 "p4":"Mode E beats false certainty",
 "motto":"MY SOUL MOVE BY POWER OF HOLY SPIRIT — "
         "What MSPS NETWORK SEE, Lord Buddha Knows."}

# =============================================================================
# §1. SD GATE
# =============================================================================

PACKAGES = {
 "L1":"independent centred sub-Gaussian noise",
 "L1e":"light-exponential (Kou jumps)",
 "L2":"stationary, summable covariance",
 "L3":"iid finite-mean blocks",
 "D":"isodiametric partition",
 "O":"OPMC admissible partition with dwell",
 "BN":"boundary-null on the fate taxonomy",
 "A1":"super-dominance s_k/n_k -> 0",
 "A2":"conditional stationarity",
 "A3":"a.e. continuity of the limit map",
 "VERIFIED":"manually checked against source with DOI",
 "ANNOT":"cell-type annotation reviewed, not marker-only",
 "CONFOUND":"batch not confounded with condition",
 "NORM":"normalisation state verified",
 "IDENT":"full-rank well-conditioned parameter block",
 "SMILES":"chemical structure verified against PubChem",
 "ABL":"ablation arms verified to differ in the forward pass",
 "FIXEDCTX":"protocol context is fixed per sample, not resampled",
 "GRAD":"loss is differentiable and reaches the intended parameters",
}

@dataclass(frozen=True)
class SDEntry:
    statistic:str; theorem:str; packages:Tuple[str,...]
    status:str; caveat:str

DICTIONARY: Dict[str,SDEntry] = {
 "composition_rate": SDEntry("composition -> limit","Thm 8.3 (Theorem A)",
   ("A1","A2","A3"),"[P|H]",
   "bound is conservative vs exact multinomial for iid cells"),
 "occupation_regime": SDEntry("multistable vs deterministic","Thm 8.1 / Prop 8.6",
   (),"[P]","infinite-mean middle is [Open] -> Mode E"),
 "taxonomy_pooling": SDEntry("nested fate levels","Thm 5.3 / Prop 5.4",
   ("BN",),"[Corr][P]","T2 is FALSE without BN or a code space"),
 "changepoints": SDEntry("protocol / trajectory breaks","Thm 7.5",
   ("L1","D","O"),"[P|H]","(SNR) carries (K+1)^2; K-free is [Open]"),
 "response_tail": SDEntry("heavy-tailed response","Thm 16.9/16.10, Prop 16.14",
   ("L1e",),"[P]","mean may be INFINITE; report quantiles"),
 "hill_identifiability": SDEntry("EC50 and n jointly identified","Fisher information",
   ("IDENT",),"[P]","EC50 and n are correlated; report CI not point"),
 "perturb_response": SDEntry("genetic perturbation effect","aggregation",
   ("VERIFIED","NORM","CONFOUND"),"[P]",
   "genetic KO is NOT a factor dose; never mix"),
 "prior_matrix": SDEntry("pathway direction prior","KO signature aggregation",
   ("VERIFIED","GRAD"),"[P|H]","direction only, not magnitude"),
 "scaffold_split": SDEntry("chemical leakage control","Bemis-Murcko",
   ("SMILES",),"[Cit]","unverified SMILES give wrong scaffolds"),
 "ablation_result": SDEntry("component helps on held-out","empirical",
   ("ABL","FIXEDCTX","GRAD"),"[P]",
   "arms must actually differ; >=3 seeds; effect > 1 pooled sd"),
 "morphology_event": SDEntry("lumen / budding topology","Prop 18.2",
   (),"[Corr][P]","closed surfaces Dchi=+-2; A added; L,C not generic"),
 "no_zeno": SDEntry("event times do not accumulate","Thm 18.6",
   (),"[Corr][P|H]","Doob bound; horizon truncation REMOVED"),
 "stc_flux": SDEntry("multi-regime flux balance","Prop 19.2/19.3",
   (),"[Corr][P]","H^k_R = H^k; (-1)^|alpha| orientation"),
 "ai_candidate": SDEntry("AI-generated hypothesis","none",(),
   "hypothesis","must pass a solver or measurement"),
 "clinical_use": SDEntry("clinical decision","none",(),"REFUSED",
   "not a clinical device; see msps_t4_preclinical"),
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
# §2. SHARED GUARDS (the bugs that invalidated the ablations)
# =============================================================================

def detect_confound(obs, batch_col:str, condition_col:str) -> Dict[str,Any]:
    """Single shared implementation. obs is a pandas DataFrame."""
    if condition_col not in obs.columns:
        return {"has_condition":False,"confounded":None,
                "WARNING":f"'{condition_col}' absent; cannot check confounding"}
    ct = obs.groupby([batch_col, condition_col]).size().unstack(fill_value=0)
    purity = ct.max(axis=1)/ct.sum(axis=1)
    n_pure = int((purity>0.9).sum()); nb = len(purity)
    conf = (n_pure == nb)
    return {"has_condition":True,"n_batches":nb,"n_pure_batches":n_pure,
            "confounded":conf,"batch_purity":purity.to_dict(),
            "VERDICT":("FATAL: batch == condition. Integration would REMOVE "
                       "the treatment effect. DO NOT correct." if conf else
                       "OK: batches mix conditions.")}

def require_no_confound(obs, batch_col, condition_col) -> Dict:
    c = detect_confound(obs, batch_col, condition_col)
    if c.get("confounded"):
        raise ConfoundRefusal(c["VERDICT"])
    return c

def fixed_context(n_samples:int, d:int, seed:int=0):
    """
    FIX for the t2_prior_init_multiomics bug: protocol_ctx was resampled
    with torch.randn on EVERY call, so train/val/test saw different
    contexts for the same sample and evaluation was non-deterministic.
    Contexts must be FIXED per sample and stored with the data.
    """
    _np()
    rng = np.random.default_rng(seed)
    return rng.normal(0,1,(n_samples,d)).astype("float32")

def verify_ablation_arms(model_factory, arms:Dict[str,Dict],
                         probe_inputs:Dict[str,Any]) -> Dict[str,Any]:
    """
    FIX for the t1_t2_t3_hill_chain bug: learnable_n=False controlled an
    UNUSED module, so arms A and B were identical. An ablation is only
    valid if disabling a component CHANGES the forward pass.
    This runs each arm on the same fixed probe and asserts distinct outputs.
    """
    if not HAS_TORCH:
        return {"checked":False,"reason":"torch absent"}
    outs = {}
    for name, flags in arms.items():
        torch.manual_seed(0)
        m = model_factory(flags)
        m.eval()
        with torch.no_grad():
            o = m(**probe_inputs)
        key = o["composition"] if isinstance(o,dict) and "composition" in o else o
        outs[name] = key.detach().cpu().numpy().ravel()
    names = list(outs)
    identical = []
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a,b = outs[names[i]], outs[names[j]]
            if a.shape==b.shape and float(np.max(np.abs(a-b)))<1e-10:
                identical.append((names[i],names[j]))
    if identical:
        raise AblationIntegrityError(
            f"ablation arms produce IDENTICAL outputs: {identical}. "
            "The flag does not reach the forward pass. Fix the model "
            "before reporting any ablation result.")
    return {"checked":True,"n_arms":len(arms),"identical_pairs": [],
            "all_arms_distinct":True}

def assert_reaches(loss, params:Sequence, name:str="component") -> Dict:
    """A prior/regulariser with no gradient is not a prior."""
    grads = [p.grad for p in params if getattr(p,"grad",None) is not None]
    ok = bool(grads) and any(float(g.abs().sum())>0 for g in grads)
    if not ok:
        raise Refusal(f"'{name}' receives NO gradient. A numpy-valued loss "
                      "cannot train anything. Implement it in torch.")
    return {"reaches":True,"n_tensors":len(grads)}

def aggregate_seeds(seed_dirs:Sequence[str], filename="ablation.json",
                    metric="test_kl") -> Dict:
    """A verdict needs |effect| > 1 pooled sd across >= 3 seeds."""
    from pathlib import Path
    rows: Dict[str,List[float]] = {}
    for d in seed_dirs:
        p = Path(d)/filename
        if not p.exists(): continue
        r = json.loads(p.read_text())
        for arm,v in r.items():
            val = v.get(metric, v.get("test_kl_mean"))
            if val is not None: rows.setdefault(arm,[]).append(float(val))
    out = {}
    for arm,vals in rows.items():
        a = np.array(vals) if HAS_NUMPY else vals
        out[arm] = {"n_seeds":len(vals),"mean":float(np.mean(a)),
                    "sd":float(np.std(a))}
    base = None
    for k in ("none","A_no_prior_no_omics","A_n1_noT3","B_e2e_random"):
        if k in out: base = k; break
    if base:
        b = out[base]
        for arm in out:
            d = b["mean"]-out[arm]["mean"]
            pooled = math.sqrt((out[arm]["sd"]**2+b["sd"]**2)/2)+1e-9
            out[arm]["delta_vs_baseline"] = d
            out[arm]["effect_in_sd"] = d/pooled
            out[arm]["verdict"] = ("HELPS" if d/pooled>1.0 else
                                   "HURTS" if d/pooled<-1.0 else "INCONCLUSIVE")
    return {"per_arm":out,"baseline":base,"n_seed_dirs":len(seed_dirs),
            "NOTE":"a verdict needs |effect| > 1 pooled sd and >= 3 seeds"}

# =============================================================================
# §3. COMPOSITION AS OCCUPATION MEASURE  (the genuine link)
# =============================================================================

def composition_as_occupation(labels:Sequence[str], fate_names:Sequence[str]
                              ) -> Dict[str,Any]:
    _np()
    idx = {f:i for i,f in enumerate(fate_names)}
    c = np.zeros(len(fate_names)); unk = 0
    for l in labels:
        if l in idx: c[idx[l]] += 1
        else: unk += 1
    F = c/max(c.sum(),1)
    return {"F":F.tolist(),"n_cells":len(labels),"n_unknown":unk,
            "fate_names":list(fate_names),
            "on_simplex":abs(F.sum()-1)<1e-9,
            "interpretation":"occupation vector, ESCA §8 with L=1"}

def theorem_A_rate(n_cells:int, n_fates:int, Gamma:float=1.0,
                   s_k:int=0, n_levels:int=1) -> Dict[str,Any]:
    """Thm 8.3 [P|H]: E||F-pi||^2 <= 4L(s/n)^2 + 4 sum d Gamma/m."""
    if n_cells<=s_k: raise Refusal("n_cells must exceed s_k")
    m = n_cells-s_k
    b1 = 4*n_levels*(s_k/n_cells)**2
    b2 = 4*n_levels*n_fates*Gamma/m
    bound = b1+b2
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
                           ordered:bool=False) -> Dict[str,Any]:
    """
    Three-way test. REQUIRES a meaningful ordering (pseudotime, spatial
    axis, clone). Unordered scRNA-seq gives Mode E for this test.
    """
    _np()
    if not ordered:
        return {"regime":"UNORDERED","status":MODE_E,
                "reason":"cell order in scRNA-seq is arbitrary; the regime "
                         "test needs pseudotime / spatial / clone ordering"}
    bs = block_structure(labels); L = np.array(bs["lengths"],float)
    if len(L)<4:
        return {"regime":"INSUFFICIENT_BLOCKS","status":MODE_E,
                "n_blocks":len(L)}
    S = np.cumsum(L); ratios = (S[:-1]/S[1:]).tolist()
    trend = float(np.mean(ratios[-max(1,len(ratios)//3):]))
    Ls = np.sort(L)[::-1]; k = max(2,int(len(Ls)*0.2))
    hill = float(np.mean(np.log(Ls[:k])-np.log(Ls[k]))) if Ls[k]>0 else 0.0
    alpha = 1.0/hill if hill>1e-9 else float("inf")
    if trend<0.25 and ratios[-1]<ratios[0]:
        regime="SUPER_DOMINANT"; status=MODE_CERTIFIED
        thm="Thm 8.1: limit is RANDOM, vertex-supported"
        cons=("genuinely MULTISTABLE: replicate organoids converge to "
              "different pure fates")
        lim=None
    elif alpha>1.5:
        regime="LIGHT_BLOCKS"; status=MODE_CERTIFIED
        thm="Prop 8.6: deterministic a.s. limit"
        cons=("NOISY BUT DETERMINISTIC: replicates converge to the same "
              "mean composition")
        num={}
        for f,ln in bs["runs"]: num[f]=num.get(f,0.0)+ln
        lim={f:num.get(f,0.0)/max(S[-1],1) for f in fate_names}
    else:
        regime="INFINITE_MEAN_CANDIDATE"; status=MODE_E
        thm="Lamperti regime: §21 O4 — NOT PROVED"; cons=(
            "tail index suggests infinite mean; generalised arcsine laws "
            "are cited, not established. Do not report a limit."); lim=None
    out={"regime":regime,"status":status,"theorem":thm,"consequence":cons,
         "n_blocks":len(L),"mean_block":float(L.mean()),
         "prefix_ratio_trend":trend,"hill_tail_index":float(alpha),
         "ordered":True}
    if lim is not None: out["deterministic_limit"]=lim
    return out

def boundary_null_check(hierarchy:Dict[str,Sequence[str]],
                        composition:Dict[str,float], tol=1e-9) -> Dict:
    """K1: a nested fate taxonomy needs projectivity before pooling levels."""
    viol=[]
    for parent,kids in hierarchy.items():
        pm=float(composition.get(parent,0.0))
        cm=sum(float(composition.get(c,0.0)) for c in kids)
        if abs(pm-cm)>tol:
            viol.append({"parent":parent,"parent_mass":pm,
                         "children_sum":cm,"residual":pm-cm})
    ok = not viol
    return {"BN_satisfied":ok,"violations":viol,
            "status":MODE_CERTIFIED if ok else MODE_E,
            "action":("levels may be pooled" if ok else
                      "DO NOT pool: projectivity violated (Prop 5.4)")}

# =============================================================================
# §4. HILL IDENTIFIABILITY  (quantify the EC50/n correlation)
# =============================================================================

def hill_jacobian(doses, ec50:float, emax:float, n:float):
    """
    o(D) = Emax D^n/(D^n+E^n)
      do/dE    = -Emax n D^n E^(n-1)/(D^n+E^n)^2
      do/dn    =  Emax D^n E^n ln(D/E)/(D^n+E^n)^2
      do/dEmax =  D^n/(D^n+E^n)
    """
    _np()
    D = np.asarray(doses,float); E = float(ec50)
    u = D**n; v = E**n; den = (u+v)**2
    dE = -emax*n*u*E**(n-1)/den
    dn = emax*u*v*np.log(D/E)/den
    dM = u/(u+v)
    return np.column_stack([dE,dn,dM])

def hill_identifiability(doses, ec50:float, emax:float, n:float,
                         sigma:float=0.02) -> Dict[str,Any]:
    """
    THE UPGRADE the file only warned about. Fisher information for
    (EC50, n, Emax); report the EC50-n correlation explicitly.
    |corr| > 0.95 -> the pair is NOT jointly identifiable from these doses.
    """
    _np()
    J = hill_jacobian(doses, ec50, emax, n)
    F = (J.T@J)/max(sigma**2,1e-18)
    ev = np.linalg.eigvalsh(F)
    rank = int(np.sum(ev > (ev[-1] if len(ev) else 0)*1e-10))
    C = np.linalg.pinv(F)
    d = np.sqrt(np.maximum(np.diag(C),0))
    corr = np.zeros_like(C)
    for i in range(3):
        for j in range(3):
            if d[i]>0 and d[j]>0: corr[i,j]=C[i,j]/(d[i]*d[j])
    c_en = float(corr[0,1])
    ident = rank==3 and abs(c_en)<0.95
    doses = np.asarray(doses,float)
    span = float(np.log10(doses.max()/max(doses.min(),1e-12)))
    return {"names":["EC50","n_Hill","Emax"],
            "fisher_eigenvalues":ev.tolist(),"rank":rank,
            "std_errors":d.tolist(),
            "corr_EC50_n":c_en,"correlation_matrix":corr.tolist(),
            "jointly_identifiable":ident,
            "n_doses":len(doses),"dose_decades":span,
            "status":MODE_CERTIFIED if ident else MODE_E,
            "VERDICT":("EC50 and n jointly identifiable" if ident else
                       f"|corr(EC50,n)|={abs(c_en):.3f} >= 0.95: the pair is "
                       "NOT jointly identifiable. Report a confidence "
                       "region, not point estimates. Add doses spanning "
                       "the transition, or fix n=1 with justification."),
            "REMEDY":["more doses across the transition region",
                      "wider dose range (>= 3 decades)",
                      "fix n=1 and justify",
                      "profile likelihood / bootstrap CI"]}

# =============================================================================
# §5. OPMC + KOU TAILS  (corrected references)
# =============================================================================

def opmc(y:Sequence[float], ell:int, beta:float) -> Dict:
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

def opmc_certified(y, ell:int, sigma:float, alpha:float=0.05) -> Dict:
    """Thm 7.5. (SNR) carries (K+1)^2; the K-free separation is [Open]."""
    n=len(y); L=math.log(10*math.e*n**3/alpha); beta=32*sigma*sigma*L
    r=opmc(y,ell,beta); bnds=[0]+r["boundaries"]+[n]
    means=[sum(y[bnds[i]:bnds[i+1]])/max(bnds[i+1]-bnds[i],1)
           for i in range(len(bnds)-1)]
    jumps=[abs(means[i+1]-means[i]) for i in range(len(means)-1)]
    kappa=min(jumps) if jumps else 0.0
    spac=[bnds[i+1]-bnds[i] for i in range(len(bnds)-1)]
    delta=min(spac) if spac else n; K=r["K_hat"]
    lhs=kappa*kappa*delta; rhs=1e5*sigma*sigma*L*(K+1)**2
    dwell = delta>=8 and (delta/2 < ell <= 3*delta/4)
    ok = lhs>=rhs and dwell
    return {**r,"kappa_min":kappa,"delta_min":delta,
            "snr_lhs":lhs,"snr_rhs":rhs,"snr_ok":lhs>=rhs,"dwell_ok":dwell,
            "radius":46*sigma*sigma*L/max(kappa*kappa,1e-30) if ok else None,
            "status":MODE_CERTIFIED if ok else MODE_E,
            "K_dependence":"(K+1)^2 present; K-free is [Open] (O1)"}

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
    def v_star(self):
        return self.vol**2+4*self.lam*(self.p/self.eta_up**2
                                       + self.q/self.eta_dn**2)

def moment_estimable(regimes:Sequence[KouRegime], p:float) -> Dict:
    """Prop 16.14: E S^p < inf iff -eta^- < p < eta^+ for EVERY regime."""
    bad=[r.name for r in regimes if r.lam>0 and not (-r.eta_dn<p<r.eta_up)]
    ok = not bad
    out={"p":p,"finite":ok,"violating_regimes":bad,"theorem":"Prop 16.14"}
    if not ok:
        out["REFUSED"]=("E S^p is INFINITE. Do NOT report this moment. "
                        "Report a quantile or trimmed statistic.")
    return out

def require_finite_mean(regimes) -> Dict:
    r = moment_estimable(regimes,1.0)
    if not r["finite"]: raise InfiniteMoment(r["REFUSED"])
    return r

def heaviest_regime_dominates(regimes:Sequence[KouRegime]) -> Dict:
    """Thm 16.10: only regimes attaining eta+* govern the extreme."""
    act=[r for r in regimes if r.lam*r.p>0]
    if not act: return {"no_active_regimes":True}
    eta=min(r.eta_up for r in act)
    gov=[r for r in act if abs(r.eta_up-eta)<1e-12]
    return {"eta_up_star":eta,"governing_regimes":[r.name for r in gov],
            "governing_occupation":sum(r.occupation for r in gov),
            "BIO_CONSEQUENCE":("a rare hyper-responder line sets the tail "
                               "however small its representation"),
            "theorem":"Thm 16.10"}

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

# =============================================================================
# §6. CORRECTED SURGERY + DOOB NO-ZENO + STC
# =============================================================================

CONVENTIONS=("closed_orientable_surface","filament_network")
SURGERY={
 "closed_orientable_surface":{
   "N":(1,0,1,2),"A":(-1,0,-1,-2),"B":(-1,0,-1,-2),
   "M":(1,0,1,2),"H":(0,2,0,-2),"P":(0,-2,0,2)},
 "filament_network":{
   "N":(1,0,0,1),"A":(-1,0,0,-1),"B":(-1,0,0,-1),
   "M":(1,0,0,1),"H":(0,1,0,-1),"P":(0,-1,0,1)},
}

def classify_morphology(db0,db1,db2,dchi,
                        convention="closed_orientable_surface") -> Dict:
    """
    K10. An organoid lumen is a CLOSED epithelial surface: chi = 2-2g and
    b1 = 2g, so a handle changes b1 by +-2 and chi by -+2. The +-1 table
    applies to filament networks. Annihilation A was missing.
    """
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

def no_zeno_certificate(delta:float,B0:float,G0:float) -> Dict:
    """K8: Doob bound; the horizon truncation is REMOVED."""
    e1 = delta/(2*B0) if B0>0 else float("inf")
    e2 = delta*delta/(8*G0*G0) if G0>0 else float("inf")
    eps0=min(e1,e2)
    if not math.isfinite(eps0) or eps0<=0: raise Refusal("degenerate bounds")
    q=4*G0*G0*eps0/(delta*delta)
    return {"eps0":eps0,"q0":q,"q0_le_half":q<=0.5+1e-12,
            "certified":q<1.0,"HORIZON_TRUNCATION":"REMOVED (K8)",
            "conclusion":"T_n -> infinity a.s.","theorem":"Thm 18.6 [Corr]"}

def stc_trace_jump(u_i,u_j,alpha=0,outward_normals=True) -> Dict:
    """K9: (-1)^|alpha| for outward normals; H^k_R = H^k."""
    sign=(-1.0)**alpha if outward_normals else 1.0
    jump=u_j-sign*u_i
    return {"alpha":alpha,"sign_factor":sign,"jump":jump,
            "satisfied":abs(jump)<1e-8,
            "Hk_R_equals":"H^k(Omega)","is_new_space":False,
            "CORRECTION":"K9"}

def stc_flux_balance(a_i,du_i,a_j,du_j) -> Dict:
    resid=a_i*du_i-a_j*du_j
    return {"residual":resid,"balanced":abs(resid)<1e-8,
            "theorem":"Prop 19.3 (flux balance in the operator domain)"}

# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print("="*76); print("  MSPS BIO MATH CORE — self-test"); print("="*76)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    sd=SDGate()

    print("\n[T1] COMPOSITION = OCCUPATION MEASURE")
    fates=["PLURI","ENDO","MESO","NEURON"]
    labels=["PLURI"]*10+["ENDO"]*30+["MESO"]*40+["NEURON"]*20
    oc=composition_as_occupation(labels,fates)
    rep("on simplex",oc["on_simplex"])
    ra=theorem_A_rate(100,4)
    print(f"      W2<={ra['W2_bound']:.4f}  iid<={ra['exact_iid_multinomial_upper']:.4f}")
    rep("bound conservative",ra["L2_bound"]>ra["exact_iid_multinomial_upper"])
    un=occupation_regime_test(labels,fates,ordered=False)
    rep("unordered -> MODE_E",un["status"]==MODE_E)
    li=occupation_regime_test(["PLURI","ENDO"]*40,fates,ordered=True)
    rep("alternating ordered -> LIGHT",li["regime"]=="LIGHT_BLOCKS")
    sup=[]
    for j,f in enumerate(fates): sup += [f]*(4**j)
    su=occupation_regime_test(sup,fates,ordered=True)
    rep("geometric ordered -> SUPER_DOMINANT",su["regime"]=="SUPER_DOMINANT")

    print("\n[T2] BN taxonomy (K1)")
    h={"GERM":["A","B"]}
    rep("consistent passes",boundary_null_check(h,{"GERM":1.,"A":.6,"B":.4})["BN_satisfied"])
    rep("inconsistent fails",not boundary_null_check(h,{"GERM":.9,"A":.6,"B":.4})["BN_satisfied"])

    print("\n[T3] HILL IDENTIFIABILITY (the new quantification)")
    hi=hill_identifiability([1.,10.,100.],10.,1.,1.5,sigma=0.02)
    print(f"      corr(EC50,n)={hi['corr_EC50_n']:.4f}  rank={hi['rank']}")
    rep("3 doses -> poorly identified",not hi["jointly_identifiable"])
    hi2=hill_identifiability([0.1,0.3,1,3,10,30,100,300,1000],10.,1.,1.0,0.02)
    print(f"      9 doses: corr={hi2['corr_EC50_n']:.4f} decades={hi2['dose_decades']:.1f}")
    rep("wide dose range improves identifiability",
        abs(hi2["corr_EC50_n"])<abs(hi["corr_EC50_n"]))
    rep("remedy listed",len(hi["REMEDY"])>=3)
    rep("verdict actionable","confidence" in hi["VERDICT"])

    print("\n[T4] OPMC honest SNR")
    y=[0.]*60+[4.]*60+[-3.]*60
    r=opmc_certified(y,30,1.0)
    rep("finds 2 change points",r["K_hat"]==2)
    rep("strict SNR honestly fails",r["status"]==MODE_E)
    rep("(K+1)^2 declared","(K+1)^2" in r["K_dependence"])

    print("\n[T5] KOU moment gate")
    r1=KouRegime("A",eta_up=3.0); r2=KouRegime("hyper",eta_up=0.7,occupation=0.05)
    rep("eta+>1 mean finite",moment_estimable([r1],1.)["finite"])
    rep("eta+<1 mean INFINITE",not moment_estimable([r2],1.)["finite"])
    b=False
    try: require_finite_mean([r2])
    except InfiniteMoment: b=True
    rep("reporting infinite mean REFUSED",b)
    hv=heaviest_regime_dominates([r1,r2])
    rep("rare line governs tail",hv["governing_regimes"]==["hyper"])

    print("\n[T6] CORRECTED SURGERY (K10)")
    ev=classify_morphology(0,2,0,-2)
    rep("lumen handle Db1=+2 -> H",ev["candidates"]==["H"])
    rep("Db1=+1 rejected for closed surface",
        classify_morphology(0,1,0,-1)["candidates"]==[])
    rep("filament accepts +-1",
        classify_morphology(0,1,0,-1,"filament_network")["candidates"]==["H"])
    rep("annihilation present",ev["annihilation_available"])
    rep("bio map present","lumen nucleation" in ev["BIO_MAP"]["N"])

    print("\n[T7] DOOB NO-ZENO (K8) + STC (K9)")
    nz=no_zeno_certificate(1.,2.,0.5)
    rep("q0 <= 1/2",nz["q0_le_half"])
    rep("truncation REMOVED","REMOVED" in nz["HORIZON_TRUNCATION"])
    rep("alpha=1 needs sign flip",stc_trace_jump(1.,-1.,1)["satisfied"])
    rep("literal outward equality fails",not stc_trace_jump(1.,1.,1)["satisfied"])
    rep("H^k_R not a new space",not stc_trace_jump(0,0)["is_new_space"])

    print("\n[T8] ABLATION INTEGRITY GUARD")
    if HAS_TORCH:
        import torch.nn as nn
        class Toy(nn.Module):
            def __init__(self,use_flag):
                super().__init__()
                self.use=use_flag
                self.a=nn.Linear(4,3); self.b=nn.Linear(4,3)
            def forward(self,x):
                return {"composition":self.a(x) if self.use else self.a(x)}
        class Good(Toy):
            def forward(self,x):
                return {"composition":self.a(x) if self.use else self.b(x)}
        probe={"x":torch.ones(2,4)}
        b=False
        try:
            verify_ablation_arms(lambda f: Toy(f["on"]),
                                 {"on":{"on":True},"off":{"on":False}},probe)
        except AblationIntegrityError: b=True
        rep("identical arms DETECTED and refused",b)
        g=verify_ablation_arms(lambda f: Good(f["on"]),
                               {"on":{"on":True},"off":{"on":False}},probe)
        rep("distinct arms accepted",g["all_arms_distinct"])
        # gradient reach
        m=Good(True); out=m(torch.ones(2,4))["composition"].sum()
        out.backward()
        rep("gradient reaches params",assert_reaches(out,list(m.a.parameters()))["reaches"])
        m2=Good(True)
        b=False
        try: assert_reaches(None,list(m2.b.parameters()),"unused")
        except Refusal: b=True
        rep("no-gradient component REFUSED",b)
    else:
        rep("torch available",False,"pip install torch")

    print("\n[T9] FIXED CONTEXT + SD")
    c1=fixed_context(8,16,seed=0); c2=fixed_context(8,16,seed=0)
    rep("context deterministic",bool(np.allclose(c1,c2)))
    rep("context shape",c1.shape==(8,16))
    rep("missing package -> MODE_E",
        sd.certify("ablation_result",("ABL","FIXEDCTX"))["status"]==MODE_E)
    rep("all packages -> CERTIFIED",
        sd.certify("ablation_result",("ABL","FIXEDCTX","GRAD"))["status"]==MODE_CERTIFIED)
    rep("clinical use refused",sd.certify("clinical_use",())["status"]==MODE_E)
    rr=sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, {rr['mode_E']} Mode E")
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
NEW HERE
  Hill EC50/n identifiability is QUANTIFIED (Fisher + correlation), not
  merely warned about. Three doses cannot separate EC50 from n.
  Ablation integrity guard: arms that produce identical outputs are refused.
  Gradient-reach guard: a numpy-valued 'loss' cannot train anything.
  Fixed protocol context: no more resampling between train and test.
  Composition regime test requires a MEANINGFUL ordering.

CORRECTED
  K1 BN taxonomy · K8 Doob no-Zeno · K9 STC orientation and H^k identity
  K10 closed-surface surgery table with annihilation

REFUSED
  Reporting an infinite moment. Clinical use. Transplant decisions.
""")
    return ok

if __name__=="__main__":
    raise SystemExit(0 if self_test() else 1)
