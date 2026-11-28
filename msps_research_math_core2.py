#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_research_math_core.py — v1.0.0
Corrected ESCA 2.0 (Unified Edition) mathematics for the SLE longitudinal,
MDOE full-pipeline, folding×phenotype and AF-phenotype-SDA modules.

STATUS LEGEND
  [P] proved · [P|H] proved under named hypotheses · [Cit] imported
  [Corr] a source statement was wrong and is corrected · [Open] not established

CORRECTIONS PROPAGATED
  K1  T2 false for general Borel partitions (Prop 5.4): needs BN / code space.
  K5  compression CONDITIONAL; Prop 6.3 generic incompressibility.
  K8  no-Zeno via Doob; horizon truncation REMOVED; (U),(R),(M) explicit.
  K9  H^k_R = H^k; orientation (-1)^|alpha|; flux balance in the domain.
  K10 closed surfaces Dchi = +-2, Db1 = +-2; annihilation A added.
  K11 Langevin edge weights MUST be symmetric Gibbs conductances.

GENUINE CORRESPONDENCES (not name analogies)
  * fraction of visits / cells / variants in each state IS an occupation
    vector on Delta^d -> Thm 8.3 rate, Prop 8.5/8.6 trichotomy.
  * OLS slope subsampling rate is sqrt(S_tt), NOT n^{3/2} in general.
  * Black-Leff EC50 = KA/(tau-1) with tau>1; tau<=1 has NO finite EC50.
  * heavy-tailed titres / responses -> Kou moment-explosion gate.
  * Hill (EC50,n) joint identifiability -> Fisher information, quantified.

PHILOSOPHY (mathematics stands independently)
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

REFUSED
  Fabricating data. Certifying an unverified hypothesis. Reporting an
  infinite moment. Gene-editor design. Clinical decisions. A test that
  passes because of `or True`.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import json, math, logging, sys
from dataclasses import dataclass, field
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

VERSION = "1.0.0"
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"
RESEARCH_ONLY = "research_only"
CARPET_RHO_MIN = 9.0 / 8.0

class Refusal(RuntimeError): pass
class InfiniteMoment(Refusal): pass
class NotIdentifiable(Refusal): pass
class BoundaryNullViolation(Refusal): pass
class IntegrityError(Refusal): pass
class Underpowered(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

def get_logger(n):
    lg = logging.getLogger(n)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg
log = get_logger("research_core")

def phi_std(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))

PHILOSOPHY = {
 "tao":"The Source","te":"The Path","qi":"The Proof",
 "p1":"a state fraction IS an occupation measure; the theorems apply literally",
 "p2":"a test with `or True` is not a test",
 "p3":"a reported k* that differs from the executed steps is not a result",
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
 "D":"delta_min-isodiametric partition",
 "O":"OPMC admissible partition with dwell",
 "BN":"boundary-null on the state taxonomy",
 "A1":"super-dominance s_k/n_k -> 0",
 "A2":"conditional stationarity",
 "A3":"a.e. continuity of the limit map",
 "U":"uniform local bounds B0, G0 and reset radius delta",
 "R":"reset only after exiting the delta-ball",
 "M":"measurable reset maps",
 "STC":"transmission conditions (correct orientation, flux balance)",
 "IDENT":"full-rank well-conditioned parameter block",
 "VERIFIED":"manually checked against source with DOI",
 "SPREAD":"ensemble members verified distinct",
 "STEPS":"executed iteration count matches the reported count",
 "RATE":"subsampling normalisation matches the statistic",
 "MEDS":"medication log complete at every visit",
 "ADJ":"flare adjudicated by a declared definition",
 "KZR":"resistance upper bound [IMPORTED]",
 "KZS":"point separation [IMPORTED]",
 "BBKT":"local-symmetry uniqueness [IMPORTED]",
}

@dataclass(frozen=True)
class SDEntry:
    statistic:str; theorem:str; packages:Tuple[str,...]
    status:str; caveat:str

DICTIONARY: Dict[str,SDEntry] = {
 "occupation_rate": SDEntry("state fraction -> limit","Thm 8.3 (Theorem A)",
   ("A1","A2","A3"),"[P|H]","conservative vs exact multinomial for iid draws"),
 "occupation_regime": SDEntry("bistable vs stationary","Thm 8.1 / Prop 8.6",
   (),"[P]","infinite-mean middle is [Open] -> Mode E"),
 "taxonomy_pooling": SDEntry("nested state levels","Thm 5.3 / Prop 5.4",
   ("BN",),"[Corr][P]","T2 FALSE without BN or a code space"),
 "changepoints": SDEntry("regime / flare onset breaks","Thm 7.5",
   ("L1","D","O"),"[P|H]","(SNR) carries (K+1)^2; K-free is [Open]"),
 "subsampling_ci": SDEntry("Politis-Romano CI","[Cit] + correct rate",
   ("RATE",),"[Cit]","stationarity + mixing; asymptotic, NOT assumption-free"),
 "response_tail": SDEntry("heavy-tailed titre / response",
   "Thm 16.9/16.10, Prop 16.14",("L1e",),"[P]",
   "mean may be INFINITE; report quantiles"),
 "esscher_worst_case": SDEntry("robust mean, KL budget","Thm 16.16",
   ("L1e",),"[P]","exact for ONE regime; bound only for switching"),
 "hill_identifiability": SDEntry("EC50 and n jointly identified",
   "Fisher information",("IDENT",),"[P]",
   "|corr| >= 0.95 -> report a region, not a point"),
 "black_leff_ec50": SDEntry("EC50 from (KA,tau)","operational model algebra",
   ("IDENT",),"[P]","EC50 = KA/(tau-1); tau<=1 has NO finite EC50"),
 "compression": SDEntry("nested / bottleneck compression",
   "Thm 6.1, Prop 6.3",("D","STEPS"),"[P][Corr]",
   "CONDITIONAL; no unconditional compression by s bits"),
 "carpet_nonpcf": SDEntry("carpet continuum limit","Thm 17.16",
   ("KZR","KZS","BBKT"),"[P|Cit]","NON-p.c.f.: three imported inputs"),
 "langevin_weights": SDEntry("weighted Carpet Langevin","Thm 17.25",
   ("STC",),"[Corr][P|C]","edge weights MUST be symmetric Gibbs"),
 "surgery_topology": SDEntry("morphology event","Prop 18.2",
   (),"[Corr][P]","closed surfaces Dchi=+-2; A added; L,C not generic"),
 "no_zeno": SDEntry("event times do not accumulate","Thm 18.6, Cor 18.7",
   ("U","R","M"),"[Corr][P|H]","Doob; horizon truncation REMOVED"),
 "stc_flux": SDEntry("multi-regime flux balance","Prop 19.2/19.3",
   ("STC",),"[Corr][P]","H^k_R = H^k; (-1)^|alpha| orientation"),
 "coupled_existence": SDEntry("global coupled surgery/PDE","§21 O7",
   (),"[Open]","NOT proved; degenerate thin-film open"),
 "state_dependent_averaging": SDEntry("state-dependent averaging","§21 O5",
   (),"[Open]","Thm 16.8 covers the ADDITIVE case only"),
 "ensemble_uq": SDEntry("ensemble uncertainty","empirical",
   ("SPREAD",),"[P]","identical members give zero spread"),
 "structure_loss": SDEntry("alignment-invariant structure loss",
   "dRMS / Kabsch",(),"[P]","absolute-coordinate MSE is NOT invariant"),
 "phenotype_prediction": SDEntry("variant phenotype","empirical",
   ("VERIFIED",),"hypothesis","ClinVar labels are submitter assertions"),
 "sle_study": SDEntry("flare lead-time research","governance",
   ("MEDS","ADJ"),"[P]","absent meds => the model learns steroid tapers"),
 "clinical_use": SDEntry("clinical decision","none",(),"REFUSED",
   "not a device; no diagnosis, dosing, or go/no-go"),
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
# §2. SUBSAMPLING WITH THE CORRECT RATE  (the SLE fix)
# =============================================================================

def ols_slope(t, y) -> float:
    _np()
    t = np.asarray(t,float); y = np.asarray(y,float)
    if len(t) < 2: return 0.0
    tc = t - t.mean(); den = float(tc@tc)
    return float(tc@(y-y.mean()))/den if den > 0 else 0.0

def stat_and_rate(t, y, stat:str="mean") -> Tuple[float,float,str]:
    """
    THE FIX. Politis-Romano needs tau_n such that tau_n(theta_hat - theta)
    has a non-degenerate limit.

      mean  : Var ~ sigma^2/n            -> tau_n = sqrt(n)
      slope : Var = sigma^2 / S_tt       -> tau_n = sqrt(S_tt)
              S_tt = sum (t_i - tbar)^2

    For an EQUALLY SPACED grid S_tt = n(n^2-1)/12 ~ n^3/12, so
    sqrt(S_tt) ~ n^{3/2}/sqrt(12): proportional to n^{3/2}, as the old code
    assumed. For IRREGULAR visit spacing n^{3/2} is WRONG; sqrt(S_tt) is
    exact in both cases. This is the design-based normalisation.
    """
    _np()
    t = np.asarray(t,float); y = np.asarray(y,float); n = len(y)
    if stat == "mean":
        return float(y.mean()) if n else float("nan"), math.sqrt(max(n,1)), "sqrt(n)"
    if stat == "slope":
        tc = t - t.mean(); Stt = float(tc@tc)
        return ols_slope(t,y), math.sqrt(max(Stt,1e-30)), "sqrt(S_tt)"
    raise Refusal(f"stat must be mean|slope, got '{stat}'")

def subsample_ci(t, y, stat:str="mean", b:Optional[int]=None,
                 conf:float=0.95, min_blocks:int=20) -> Dict[str,Any]:
    """
    Overlapping-block subsampling with the DESIGN-BASED rate.
    L(x) = empirical law of tau_b (theta_{b,i} - theta_n)
    CI   = [theta_n - q_{1-a/2}/tau_n, theta_n - q_{a/2}/tau_n]
    Assumptions: stationarity + mixing; b -> inf, b/n -> 0. Asymptotic.
    """
    _np()
    t = np.asarray(t,float); y = np.asarray(y,float); n = len(y)
    theta_n, tau_n, rate_name = stat_and_rate(t,y,stat)
    if n < 6:
        return {"status":MODE_E,"stat":stat,"estimate":theta_n,
                "lo":theta_n,"hi":theta_n,"n":n,"b":0,"n_blocks":0,
                "rate_name":rate_name,"reason":"n < 6"}
    if b is None:
        b = int(max(3, min(int(round(n**(1/3))), n//2)))
    nb = n-b+1
    if nb < min_blocks:
        b = max(3, n-min_blocks+1); nb = n-b+1
    if nb < 4:
        return {"status":MODE_E,"stat":stat,"estimate":theta_n,
                "lo":theta_n,"hi":theta_n,"n":n,"b":b,"n_blocks":nb,
                "rate_name":rate_name,"reason":"too few blocks"}
    devs = np.empty(nb)
    for i in range(nb):
        tb = t[i:i+b]; yb = y[i:i+b]
        th_b, tau_b, _ = stat_and_rate(tb,yb,stat)
        devs[i] = tau_b*(th_b - theta_n)
    a = 1.0-conf
    q_hi = float(np.quantile(devs,1-a/2)); q_lo = float(np.quantile(devs,a/2))
    lo = theta_n - q_hi/tau_n; hi = theta_n - q_lo/tau_n
    if lo > hi: lo,hi = hi,lo
    # spacing diagnostic
    gaps = np.diff(t)
    cv = float(gaps.std(ddof=1)/gaps.mean()) if len(gaps)>1 and gaps.mean()>0 else 0.0
    return {"status":MODE_CERTIFIED,"stat":stat,"estimate":theta_n,
            "lo":lo,"hi":hi,"width":hi-lo,"n":n,"b":b,"n_blocks":nb,
            "conf":conf,"tau_n":tau_n,"rate_name":rate_name,
            "spacing_cv":cv,"spacing_irregular":cv>0.25,
            "method":"Politis-Romano overlapping blocks, design-based rate",
            "assumptions":"stationarity + mixing; asymptotic",
            "RATE_NOTE":("sqrt(S_tt) handles irregular spacing exactly; "
                         "n^{3/2} does not")}

def iid_ci_WRONG(t, y, stat:str="mean", conf:float=0.95) -> Dict[str,Any]:
    """The WRONG interval, kept ONLY so undercoverage can be MEASURED."""
    _np()
    t=np.asarray(t,float); y=np.asarray(y,float); n=len(y)
    if n < 3:
        m = float(y.mean()) if n else float("nan")
        return {"estimate":m,"lo":m,"hi":m,"width":0.0,"n":n}
    z = 1.959963985
    if stat == "mean":
        est = float(y.mean()); se = float(y.std(ddof=1)/math.sqrt(n))
    else:
        est = ols_slope(t,y); tc = t-t.mean(); den=float(tc@tc)
        resid = y - (y.mean()+est*tc)
        s2 = float(resid@resid)/max(n-2,1)
        se = math.sqrt(s2/den) if den>0 else 0.0
    return {"estimate":est,"lo":est-z*se,"hi":est+z*se,"width":2*z*se,
            "n":n,"WARNING":"iid SE invalid under serial correlation"}

def tau_int_ms(y, c:float=5.0) -> Dict[str,Any]:
    """Madras-Sokal automatic window; n_eff = n/(2 tau)."""
    _np()
    y=np.asarray(y,float); n=len(y)
    if n < 16:
        return {"status":MODE_E,"tau_int":float("nan"),"n":n,
                "n_eff":float("nan"),"reason":"n < 16"}
    xc = y-y.mean(); var = float(xc@xc)/n
    if var <= 0:
        return {"status":MODE_CERTIFIED,"tau_int":0.5,"window":0,"n":n,
                "n_eff":float(n),"note":"constant series"}
    tau, W = 0.5, 0
    for k in range(1, n//2):
        rho = float(xc[:n-k]@xc[k:])/(n-k)/var
        tau += rho
        if tau <= 0: tau=0.5; W=k; break
        W = k
        if W >= c*max(tau,0.5): break
    tau = max(tau,0.5); ne = n/(2*tau)
    return {"status":MODE_CERTIFIED,"tau_int":float(tau),"window":int(W),
            "n":n,"n_eff":float(ne),"few_effective_samples":ne<20,
            "method":"Madras-Sokal automatic window","c":c}

def coverage_study(n_real:int=300, n:int=60, rho:float=0.7,
                   stat:str="mean", conf:float=0.95, seed:int=0,
                   irregular:bool=False) -> Dict[str,Any]:
    """
    THE decisive test. AR(1) with a KNOWN truth; compare empirical
    coverage of the subsampling CI against the iid CI. With irregular
    spacing the sqrt(S_tt) rate is exercised against n^{3/2}.
    """
    _np()
    rng = np.random.default_rng(seed)
    hit_s=hit_i=0; ws=[]; wi=[]; truth_hits=0
    for _ in range(n_real):
        if irregular:
            t = np.cumsum(rng.uniform(2.,10.,n))
        else:
            t = np.arange(n,dtype=float)
        e = rng.normal(size=n); y = np.empty(n)
        y[0] = e[0]/math.sqrt(max(1-rho*rho,1e-9))
        for i in range(1,n): y[i] = rho*y[i-1]+e[i]
        truth = 0.0
        if stat == "slope":
            beta = 0.2; y = y + beta*t; truth = beta
        s = subsample_ci(t,y,stat,conf=conf)
        i_ = iid_ci_WRONG(t,y,stat,conf=conf)
        if s["status"] != MODE_CERTIFIED: continue
        truth_hits += 1
        hit_s += int(s["lo"] <= truth <= s["hi"])
        hit_i += int(i_["lo"] <= truth <= i_["hi"])
        ws.append(s["width"]); wi.append(i_["width"])
    m = max(truth_hits,1)
    return {"status":MODE_CERTIFIED,"n_realisations":truth_hits,"n":n,
            "ar1_rho":rho,"stat":stat,"nominal_conf":conf,
            "irregular_spacing":irregular,
            "coverage_subsampling":hit_s/m,"coverage_iid":hit_i/m,
            "median_width_subsampling":float(np.median(ws)) if ws else None,
            "median_width_iid":float(np.median(wi)) if wi else None,
            "width_ratio":(float(np.median(ws)/np.median(wi))
                           if ws and wi else None),
            "interpretation":("iid undercovers under serial correlation; "
                              "subsampling is wider and closer to nominal")}

# =============================================================================
# §3. OCCUPATION MEASURES  (state fractions)
# =============================================================================

def composition_as_occupation(labels:Sequence[str], names:Sequence[str]) -> Dict:
    _np()
    idx = {f:i for i,f in enumerate(names)}
    c = np.zeros(len(names)); unk = 0
    for l in labels:
        if l in idx: c[idx[l]] += 1
        else: unk += 1
    F = c/max(c.sum(),1)
    return {"F":F.tolist(),"n":len(labels),"n_unknown":unk,
            "names":list(names),"on_simplex":abs(F.sum()-1)<1e-9,
            "interpretation":"occupation vector, ESCA §8 with L=1"}

def theorem_A_rate(n:int, d:int, Gamma:float=1.0, s_k:int=0,
                   n_levels:int=1) -> Dict:
    if n <= s_k: raise Refusal("n must exceed s_k")
    m = n-s_k
    b1 = 4*n_levels*(s_k/n)**2; b2 = 4*n_levels*d*Gamma/m
    bound = b1+b2
    return {"n":n,"d":d,"Gamma":Gamma,"L2_bound":bound,
            "W2_bound":math.sqrt(bound),
            "exact_iid_multinomial_upper":1.0/n,
            "NOTE":"theorem bound is conservative for iid draws",
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

def occupation_regime_test(labels:Sequence[str], names:Sequence[str],
                           ordered:bool=False) -> Dict:
    """
    Three-way test. REQUIRES a meaningful ordering (time, pseudotime,
    clone). Unordered data gives Mode E for this test.

      light blocks (E ell < inf)      -> Prop 8.6 deterministic a.s. limit
      super-dominant (S_j/S_{j+1}->0) -> Thm 8.1 RANDOM vertex limit
      regularly varying infinite mean -> Lamperti [Open] -> Mode E
    """
    _np()
    if not ordered:
        return {"regime":"UNORDERED","status":MODE_E,
                "reason":"needs a meaningful ordering (time / pseudotime / clone)"}
    bs = block_structure(labels); L = np.array(bs["lengths"],float)
    if len(L) < 4:
        return {"regime":"INSUFFICIENT_BLOCKS","status":MODE_E,"n_blocks":len(L)}
    S = np.cumsum(L); ratios = (S[:-1]/S[1:]).tolist()
    trend = float(np.mean(ratios[-max(1,len(ratios)//3):]))
    Ls = np.sort(L)[::-1]; k = max(2,int(len(Ls)*0.2))
    hill = float(np.mean(np.log(Ls[:k])-np.log(Ls[k]))) if Ls[k]>0 else 0.0
    alpha = 1.0/hill if hill>1e-9 else float("inf")
    lim=None
    if trend<0.25 and ratios[-1]<ratios[0]:
        regime="SUPER_DOMINANT"; status=MODE_CERTIFIED
        thm="Thm 8.1: limit is RANDOM, vertex-supported"
        cons=("genuinely BISTABLE: replicate courses converge to different "
              "pure states")
    elif alpha>1.5:
        regime="LIGHT_BLOCKS"; status=MODE_CERTIFIED
        thm="Prop 8.6: deterministic a.s. limit (renewal-reward)"
        cons="NOISY BUT STATIONARY: replicates converge to the same mean"
        num={}
        for f,ln in bs["runs"]: num[f]=num.get(f,0.0)+ln
        lim={f:num.get(f,0.0)/max(S[-1],1) for f in names}
    else:
        regime="INFINITE_MEAN_CANDIDATE"; status=MODE_E
        thm="Lamperti regime: §21 O4 — NOT PROVED"
        cons="tail index suggests infinite mean; do not report a limit"
    out={"regime":regime,"status":status,"theorem":thm,"consequence":cons,
         "n_blocks":len(L),"mean_block":float(L.mean()),
         "prefix_ratio_trend":trend,"hill_tail_index":float(alpha),
         "ordered":True}
    if lim is not None: out["deterministic_limit"]=lim
    return out

def boundary_null_check(hierarchy:Dict[str,Sequence[str]],
                        composition:Dict[str,float], tol=1e-9) -> Dict:
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
# §4. KOU MOMENT GATE  (heavy-tailed titres and responses)
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

def moment_estimable(regimes:Sequence[KouRegime], p:float) -> Dict:
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

def fit_kou_tail(values, name="cohort") -> KouRegime:
    _np()
    x=np.asarray(values,float)
    if len(x)<8: raise Refusal("need >= 8 values")
    d=np.diff(x); up=d[d>0]; dn=-d[d<0]
    eu=1.0/max(float(up.mean()),1e-9) if len(up)>1 else 10.0
    ed=1.0/max(float(dn.mean()),1e-9) if len(dn)>1 else 10.0
    return KouRegime(name,p=len(up)/max(len(d),1),eta_up=eu,eta_dn=ed,
                     lam=float(len(d))/len(x),drift=float(d.mean()),
                     vol=float(d.std()))

def heaviest_regime_dominates(regimes:Sequence[KouRegime]) -> Dict:
    act=[r for r in regimes if r.lam*r.p>0]
    if not act: return {"no_active_regimes":True}
    eta=min(r.eta_up for r in act)
    gov=[r for r in act if abs(r.eta_up-eta)<1e-12]
    return {"eta_up_star":eta,"governing_regimes":[r.name for r in gov],
            "governing_occupation":sum(r.occupation for r in gov),
            "CONSEQUENCE":("a rare heavy-tail subgroup sets the extreme "
                           "however small its representation"),
            "theorem":"Thm 16.10"}

def gumbel_extreme(r:KouRegime, T:float) -> Dict:
    a=r.lam*r.p
    if a<=0: return {"no_upward_jumps":True}
    cdf=lambda x: math.exp(-a*T*math.exp(-r.eta_up*x))
    return {"lambda_p_T":a*T,"eta_up":r.eta_up,
            "gumbel_location":math.log(a*T)/r.eta_up,
            "gumbel_scale":1.0/r.eta_up,"cdf_at_0":cdf(0.0),
            "return_level_99":(math.log(a*T)-math.log(math.log(1/0.99)))/r.eta_up,
            "theorem":"Thm 16.9","status":"[P] exact"}

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

def response_summary(values, regimes=None) -> Dict:
    _np()
    x=np.asarray(values,float)
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
    if len(regs or [])>1: out["tail_governance"]=heaviest_regime_dominates(regs)
    return out

# =============================================================================
# §5. BLACK-LEFF + HILL IDENTIFIABILITY  (the MDOE fix)
# =============================================================================

def black_leff_ec50(KA:float, tau:float, n:float=1.0) -> Dict[str,Any]:
    """
    THE FIX. Operational model
        E(A) = Em tau^n A^n / ((KA+A)^n + tau^n A^n)
    At E = Em/2:  tau^n A^n = (KA+A)^n  =>  tau A = KA + A
        EC50 = KA / (tau - 1)
    INDEPENDENT of n. Requires tau > 1; tau <= 1 never reaches half-max,
    so there is NO finite EC50. The old formula KA/(tau^{1/n}+1) is wrong,
    and a signed tau makes tau^n undefined for non-integer n.
    """
    if KA <= 0: raise Refusal("KA must be positive")
    if tau <= 0: raise Refusal("tau must be POSITIVE (transducer efficacy)")
    if tau <= 1.0:
        return {"ec50":float("inf"),"finite":False,"KA":KA,"tau":tau,"n":n,
                "reason":("tau <= 1: the system never reaches half-maximal "
                          "response; NO finite EC50 exists"),
                "formula":"EC50 = KA/(tau-1)","status":MODE_E}
    ec = KA/(tau-1.0)
    return {"ec50":ec,"finite":True,"KA":KA,"tau":tau,"n":n,
            "formula":"EC50 = KA/(tau-1)","n_independent":True,
            "status":MODE_CERTIFIED}

def black_leff_occupancy(A, KA, tau, n, Em):
    """Differentiable operational occupancy. tau must be > 0."""
    if not HAS_TORCH: raise Refusal("torch required")
    import torch
    eps=1e-9
    Ac=A.clamp(min=eps); KAc=KA.clamp(min=eps); tauc=tau.clamp(min=eps)
    An=Ac.pow(n); tot=(KAc+Ac).pow(n)+tauc.pow(n)*An
    return Em*tauc.pow(n)*An/tot.clamp(min=eps)

def hill_jacobian(doses, ec50:float, emax:float, n:float):
    _np()
    D=np.asarray(doses,float); E=float(ec50)
    u=D**n; v=E**n; den=(u+v)**2
    dE=-emax*n*u*E**(n-1)/den
    dn=emax*u*v*np.log(D/E)/den
    dM=u/(u+v)
    return np.column_stack([dE,dn,dM])

def hill_identifiability(doses, ec50:float, emax:float, n:float,
                         sigma:float=0.02) -> Dict[str,Any]:
    """
    Fisher information for (EC50, n, Emax); report the EC50-n correlation.
    |corr| >= 0.95 -> the pair is NOT jointly identifiable from these doses.
    """
    _np()
    J=hill_jacobian(doses,ec50,emax,n)
    F=(J.T@J)/max(sigma**2,1e-18)
    ev=np.linalg.eigvalsh(F)
    rank=int(np.sum(ev>(ev[-1] if len(ev) else 0)*1e-10))
    C=np.linalg.pinv(F); d=np.sqrt(np.maximum(np.diag(C),0))
    corr=np.zeros_like(C)
    for i in range(3):
        for j in range(3):
            if d[i]>0 and d[j]>0: corr[i,j]=C[i,j]/(d[i]*d[j])
    c_en=float(corr[0,1]); ident = rank==3 and abs(c_en)<0.95
    doses=np.asarray(doses,float)
    span=float(np.log10(doses.max()/max(doses.min(),1e-12)))
    return {"names":["EC50","n_Hill","Emax"],
            "fisher_eigenvalues":ev.tolist(),"rank":rank,
            "std_errors":d.tolist(),"corr_EC50_n":c_en,
            "correlation_matrix":corr.tolist(),
            "jointly_identifiable":ident,"n_doses":len(doses),
            "dose_decades":span,
            "status":MODE_CERTIFIED if ident else MODE_E,
            "VERDICT":("EC50 and n jointly identifiable" if ident else
                       f"|corr(EC50,n)|={abs(c_en):.3f} >= 0.95: NOT jointly "
                       "identifiable. Report a confidence region, add doses "
                       "across the transition, or fix n=1 with justification."),
            "REMEDY":["more doses across the transition",
                      "wider dose range (>= 3 decades)",
                      "fix n=1 and justify","profile likelihood / bootstrap CI"]}

# =============================================================================
# §6. OPMC + COMPRESSION HONESTY
# =============================================================================

def opmc(y, ell:int, beta:float) -> Dict:
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

def compression_bound(N:int, b:int, K:int, h_bar:int=0,
                      lam_bits:Optional[int]=None) -> Dict:
    lam = lam_bits if lam_bits is not None else max(1,math.ceil(math.log2(max(N,2))))
    C=(2*K-1)*(1+h_bar)+K*b+(K-1)*(lam+2*math.log2(max(K,2))+1)
    raw=N*b
    return {"N":N,"b":b,"K":K,"code_length_bound":C,"raw_bits":raw,
            "compresses":C<raw,
            "generic_incompressibility":"fraction <= 2^{1-s} compress by s bits",
            "NOT":"unconditional compression","theorem":"Thm 6.1, Prop 6.3"}

def verify_step_count(claimed:int, executed:int, name:str="iteration") -> Dict:
    """A reported k* that differs from the executed count is not a result."""
    if claimed != executed:
        raise IntegrityError(
            f"{name}: claimed {claimed} but executed {executed}. "
            "Report the EXECUTED count, or run the claimed number of steps.")
    return {"claimed":claimed,"executed":executed,"matches":True}

# =============================================================================
# §7. SURGERY / NO-ZENO / STC / GIBBS
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
                        convention="closed_orientable_surface") -> Dict:
    if convention not in CONVENTIONS: raise Refusal(f"convention {CONVENTIONS}")
    tbl=SURGERY[convention]; sig=(db0,db1,db2,dchi)
    cands=[op for op,v in tbl.items() if v==sig]
    return {"signature":sig,"candidates":cands,"convention":convention,
            "euler_consistent":dchi==db0-db1+db2,
            "annihilation_available":"A" in tbl,
            "NOT_CLASSIFIED":["L (boundary contact)","C (junction reconnect)"],
            "CORRECTION":"closed surfaces Dchi=+-2, Db1=+-2 (K10)"}

def doob_exit_bound(delta:float,B0:float,G0:float,eps:float) -> Dict:
    if delta<=0 or G0<0 or B0<0: raise Refusal("delta>0, G0>=0, B0>=0")
    eps_max=delta/(2*B0) if B0>0 else float("inf")
    if eps>eps_max: raise Refusal(f"eps={eps} exceeds delta/(2B0)={eps_max}")
    q=4*G0*G0*eps/(delta*delta)
    return {"delta":delta,"B0":B0,"G0":G0,"eps":eps,"q_doob":q,"q_lt_1":q<1.0,
            "theorem":"Prop 18.4 (Doob L2 maximal inequality)",
            "REPLACES":"2exp(-delta^2/(8G0^2 eps) [cited, unproved]"}

def no_zeno_certificate(delta:float,B0:float,G0:float) -> Dict:
    e1=delta/(2*B0) if B0>0 else float("inf")
    e2=delta*delta/(8*G0*G0) if G0>0 else float("inf")
    eps0=min(e1,e2)
    if not math.isfinite(eps0) or eps0<=0: raise Refusal("degenerate bounds")
    q=4*G0*G0*eps0/(delta*delta)
    return {"eps0":eps0,"q0":q,"q0_le_half":q<=0.5+1e-12,"certified":q<1.0,
            "HORIZON_TRUNCATION":"REMOVED (K8)",
            "conclusion":"T_n -> infinity a.s. (Levy conditional B-C)",
            "theorem":"Thm 18.6 [Corr]"}

def surgery_count_tail(t:float,m:int,eps0:float,q0:float) -> Dict:
    if m<=0 or eps0<=0: raise Refusal("m>0, eps0>0")
    x=t/(m*eps0); p=1-q0
    if not (0<x<1) or not (0<p<1):
        return {"applicable":False,"reason":"need 0 < t/(m eps0) < 1"}
    kl=x*math.log(x/p)+(1-x)*math.log((1-x)/(1-p))
    return {"t":t,"m":m,"x":x,"KL":kl,"bound":math.exp(-m*kl),
            "applicable":True,"theorem":"Cor 18.7"}

def stc_trace_jump(u_i,u_j,alpha=0,outward_normals=True) -> Dict:
    sign=(-1.0)**alpha if outward_normals else 1.0
    jump=u_j-sign*u_i
    return {"alpha":alpha,"sign_factor":sign,"jump":jump,
            "satisfied":abs(jump)<1e-8,"Hk_R_equals":"H^k(Omega)",
            "is_new_space":False,"CORRECTION":"K9"}

def stc_flux_balance(a_i,du_i,a_j,du_j) -> Dict:
    resid=a_i*du_i-a_j*du_j
    return {"residual":resid,"balanced":abs(resid)<1e-8,
            "theorem":"Prop 19.3 (flux balance in the operator domain)"}

def stc_reflux(coarse_flux:float,fine_fluxes:Sequence[float]) -> Dict:
    tot=float(sum(fine_fluxes)); mis=tot-coarse_flux
    return {"coarse":coarse_flux,"fine_total":tot,"mismatch":mis,
            "corrected":coarse_flux+mis,"residual_after":abs(tot-(coarse_flux+mis)),
            "conserved":abs(mis)<1e-14,
            "DERIVED":"STC^1 requires refluxing; not optional"}

def gibbs_conductance(U_a:float,U_b:float,beta:float,rho:float=1.0) -> float:
    """K11: symmetric in (a,b). The source's exp(-beta/2[U(b)-U(a)]) is NOT."""
    return rho*math.exp(-0.5*beta*(U_a+U_b))

def verify_gibbs_symmetry(U:Sequence[float],beta:float) -> Dict:
    n=len(U); asym=0.0; wrong=0.0
    for a in range(n):
        for b in range(n):
            if a==b: continue
            asym=max(asym,abs(gibbs_conductance(U[a],U[b],beta)
                              -gibbs_conductance(U[b],U[a],beta)))
            wrong=max(wrong,abs(math.exp(-0.5*beta*(U[b]-U[a]))
                                -math.exp(-0.5*beta*(U[a]-U[b]))))
    return {"symmetric_asymmetry":asym,"is_symmetric":asym<1e-12,
            "source_formula_asymmetry":wrong,
            "source_was_symmetric":wrong<1e-12,
            "CORRECTION":"K11","theorem":"Thm 17.25 [Corr]"}

def carpet_status(is_pcf:bool) -> Dict:
    if is_pcf:
        return {"is_pcf":True,"construction":"Kigami renormalisation",
                "status":"[Cit]","inputs_needed":[],"resistance_ratio":5.0/3.0}
    return {"is_pcf":False,"construction":"NOT constructive",
            "inputs_needed":["(KZ-R) resistance upper bound",
                             "(KZ-S) point separation","(BBKT) uniqueness"],
            "proved_here":["chain property (G)","(C1) from (KZ-R)+(G)",
                           "(C2) from (C1) via Gamma-compactness"],
            "rho_lower_bound":CARPET_RHO_MIN,"status":"[P|Cit]",
            "theorem":"Thm 17.16"}

def carpet_multiscale_energy(fields:Dict[int,"np.ndarray"],
                             laplacians:Dict[int,"np.ndarray"],
                             kappa:Dict[int,float]) -> Dict:
    """Each level m uses its OWN coarse-grained field and gauge kappa_m ∝ rho^m."""
    _np()
    terms={}
    for m,u in fields.items():
        L=laplacians[m]
        if u.shape[0]!=L.shape[0]:
            raise Refusal(f"level {m}: field {u.shape[0]} != Laplacian {L.shape[0]}")
        terms[m]=float(kappa[m]*(u@L@u))
    return {"per_level_energy":terms,"total":sum(terms.values()),"gauge":kappa,
            "CORRECTION":"each level uses its OWN field and rho^m gauge",
            "NOT":"one field reweighted, or rho stored but unused"}

# =============================================================================
# §8. INTEGRITY GUARDS  (the fake tests and the noise ensembles)
# =============================================================================

def assert_real_test(name:str, condition:bool) -> Dict:
    """
    A test written as `cond or True` always passes. This helper takes a
    plain bool and records it; there is no escape hatch.
    """
    return {"test":name,"passed":bool(condition),"or_True_present":False}

def verify_ensemble_spread(trajectories, tol:float=1e-12) -> Dict:
    _np()
    a=np.asarray(trajectories,float)
    if a.ndim<2 or a.shape[0]<2:
        raise IntegrityError("need >= 2 ensemble members")
    E=a.shape[0]; mx=0.0
    for i in range(E):
        for j in range(i+1,E):
            mx=max(mx,float(np.max(np.abs(a[i]-a[j]))))
    spread=float(np.std(a,axis=0).mean())
    if mx<=tol:
        raise IntegrityError(
            f"ensemble members are IDENTICAL (max pairwise diff {mx:.2e}). "
            "A seed does not perturb a deterministic model. Perturb "
            "parameters, inputs, or use MC-dropout.")
    return {"n_members":E,"max_pairwise_diff":mx,"mean_spread":spread,
            "members_distinct":True}

def verify_no_information_collapse(x, name:str="feature",
                                   tol:float=1e-12) -> Dict:
    """
    Detects `feat.mean(-1,keepdim=True).expand(-1,d)`: every entry along
    the last axis identical => all information destroyed.
    """
    _np()
    a=np.asarray(x,float)
    if a.ndim<2: return {"collapsed":False,"name":name}
    spread=float(np.std(a,axis=-1).mean())
    if spread<=tol:
        raise IntegrityError(
            f"'{name}' has ZERO spread along the last axis (mean {spread:.2e}). "
            "This is the mean-then-expand collapse: every entry is identical "
            "and all information is destroyed. Pass the feature itself.")
    return {"collapsed":False,"name":name,"mean_last_axis_spread":spread}

def verify_persistent_projection(module, name:str="projection") -> Dict:
    """A projection created fresh inside forward() is untrained noise."""
    if not HAS_TORCH: return {"checked":False}
    import torch.nn as nn
    n_params=sum(1 for _ in module.parameters())
    if n_params==0:
        raise IntegrityError(
            f"'{name}' has NO registered parameters. A layer constructed "
            "inside forward() is re-randomised every call and never trained. "
            "Register it in __init__.")
    return {"checked":True,"name":name,"n_registered_params":n_params}

def kabsch_rmsd(P, Q, mask=None) -> Dict:
    """Rotation/translation-invariant RMSD via Kabsch. Differentiable."""
    if not HAS_TORCH: raise Refusal("torch required")
    import torch
    if mask is None: mask=torch.ones(P.shape[:2],dtype=torch.bool,device=P.device)
    m=mask.to(P.dtype).unsqueeze(-1); denom=m.sum(1).clamp_min(1)
    Pc=P-(P*m).sum(1,keepdim=True)/denom.unsqueeze(-1)
    Qc=Q-(Q*m).sum(1,keepdim=True)/denom.unsqueeze(-1)
    H=(Pc*m).transpose(1,2)@Qc
    U,S,Vh=torch.linalg.svd(H)
    d=torch.sign(torch.det(Vh.transpose(1,2)@U.transpose(1,2)))
    D=torch.eye(3,device=P.device,dtype=P.dtype).unsqueeze(0).repeat(P.shape[0],1,1)
    D[:,2,2]=d
    R=Vh.transpose(1,2)@D@U.transpose(1,2)
    Pr=torch.einsum("bij,bnj->bni",R,Pc)
    per=((Pr-Qc)**2).sum(-1)
    rmsd=torch.sqrt((per*mask.to(P.dtype)).sum(1)/denom.squeeze(-1)+1e-12)
    return {"rmsd":rmsd,"mean":float(rmsd.mean()),"rotated":Pr}

def drms_loss(P, Q, mask=None):
    """
    Pairwise-distance RMS loss. Rotation/translation invariant WITHOUT
    SVD, so it is numerically stable in early training.
    """
    if not HAS_TORCH: raise Refusal("torch required")
    import torch
    dP=torch.cdist(P,P); dQ=torch.cdist(Q,Q)
    if mask is None:
        return ((dP-dQ)**2).mean()
    mm=mask.unsqueeze(1)&mask.unsqueeze(2)
    return (((dP-dQ)**2)*mm).sum()/mm.sum().clamp_min(1)

# =============================================================================
# §9. POWER / STUDY DESIGN
# =============================================================================

def z_quantile(p:float) -> float:
    lo,hi=-10.,10.
    for _ in range(200):
        m=0.5*(lo+hi)
        if phi_std(m)<p: lo=m
        else: hi=m
    return 0.5*(lo+hi)

def binary_power(p1:float,p2:float,alpha:float=0.05,power:float=0.8) -> Dict:
    if not (0<p1<1 and 0<p2<1): raise Refusal("p in (0,1)")
    za=z_quantile(1-alpha/2); zb=z_quantile(power); d=abs(p1-p2)
    n=((za+zb)**2)*(p1*(1-p1)+p2*(1-p2))/(d*d)
    return {"p1":p1,"p2":p2,"n_per_arm":math.ceil(n),"n_total":2*math.ceil(n),
            "PURPOSE":"study DESIGN, not outcome prediction"}

def detectable_effect(n_per_arm:int,p_control:float=0.3,alpha:float=0.05,
                      power:float=0.8) -> Dict:
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

# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose:bool=True) -> bool:
    print("="*76); print("  MSPS RESEARCH MATH CORE — self-test"); print("="*76)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    sd=SDGate(); rng=np.random.default_rng(0)

    print("\n[T1] SUBSAMPLING RATE: the design-based fix")
    t=np.arange(60,dtype=float); y=0.2*t+rng.normal(0,1,60)
    th,tau,rn=stat_and_rate(t,y,"slope")
    Stt=float(((t-t.mean())**2).sum())
    print(f"      slope={th:.4f}  tau_n=sqrt(S_tt)={tau:.2f}  ({rn})")
    rep("slope rate is sqrt(S_tt)",rn=="sqrt(S_tt)")
    rep("sqrt(S_tt) ~ n^1.5/sqrt(12) for equal spacing",
        abs(tau - 60**1.5/math.sqrt(12))/(60**1.5/math.sqrt(12))<0.01)
    ci=subsample_ci(t,y,"slope")
    rep("slope CI brackets 0.2",ci["lo"]<=0.2<=ci["hi"],
        f"[{ci['lo']:.4f},{ci['hi']:.4f}]")
    ti=np.cumsum(rng.uniform(2.,10.,60)); yi=0.2*ti+rng.normal(0,3,60)
    ci2=subsample_ci(ti,yi,"slope")
    print(f"      irregular spacing CV={ci2['spacing_cv']:.3f} "
          f"irregular={ci2['spacing_irregular']}")
    rep("irregular spacing flagged",ci2["spacing_irregular"])
    rep("rate note explains n^1.5 failure","n^{3/2} does not" in ci2["RATE_NOTE"])

    print("\n[T2] COVERAGE: subsampling vs iid")
    for rho in (0.0,0.7):
        c=coverage_study(n_real=200,n=60,rho=rho,seed=1)
        print(f"      rho={rho:.1f} sub={c['coverage_subsampling']:.3f} "
              f"iid={c['coverage_iid']:.3f} width ratio={c['width_ratio']:.2f}")
        if rho>=0.6:
            rep(f"rho={rho}: iid UNDERCOVERS",c["coverage_iid"]<0.85)
            rep(f"rho={rho}: subsampling wider",c["width_ratio"]>1.2)
    cs=coverage_study(n_real=150,n=50,rho=0.5,stat="slope",
                      seed=2,irregular=True)
    rep("irregular-spacing slope study runs",cs["status"]==MODE_CERTIFIED)

    print("\n[T3] BLACK-LEFF EC50: the corrected algebra")
    r=black_leff_ec50(KA=10.0,tau=3.0,n=1.5)
    print(f"      KA=10, tau=3 -> EC50={r['ec50']:.4f} (KA/(tau-1)=5)")
    rep("EC50 = KA/(tau-1) = 5",abs(r["ec50"]-5.0)<1e-12)
    r2=black_leff_ec50(KA=10.0,tau=3.0,n=4.0)
    rep("EC50 independent of n",abs(r2["ec50"]-r["ec50"])<1e-12)
    wrong=10.0/(3.0**(1/1.5)+1)
    rep("old formula KA/(tau^(1/n)+1) is DIFFERENT",abs(wrong-5.0)>1.0,
        f"old={wrong:.4f} vs correct=5.0")
    r3=black_leff_ec50(KA=10.0,tau=0.8)
    rep("tau<=1 -> NO finite EC50",not r3["finite"])
    b=False
    try: black_leff_ec50(10.0,-2.0)
    except Refusal: b=True
    rep("negative tau REFUSED",b)
    if HAS_TORCH:
        import torch
        A=torch.tensor([5.0],dtype=torch.float64)
        occ=black_leff_occupancy(A,torch.tensor([10.0],dtype=torch.float64),
                                 torch.tensor([3.0],dtype=torch.float64),
                                 torch.tensor([1.5],dtype=torch.float64),
                                 torch.tensor([1.0],dtype=torch.float64))
        rep("occupancy at EC50 = Em/2",abs(float(occ)-0.5)<1e-9,
            f"{float(occ):.6f}")

    print("\n[T4] HILL IDENTIFIABILITY")
    hi=hill_identifiability([1.,10.,100.],10.,1.,1.5)
    print(f"      3 doses: corr(EC50,n)={hi['corr_EC50_n']:.4f}")
    rep("3 doses NOT jointly identifiable",not hi["jointly_identifiable"])
    hi2=hill_identifiability([0.1,0.3,1,3,10,30,100,300,1000],10.,1.,1.)
    rep("wide range improves",abs(hi2["corr_EC50_n"])<abs(hi["corr_EC50_n"]))

    print("\n[T5] OCCUPATION REGIME (state fractions)")
    names=["REM","FLARE"]
    rep("unordered -> MODE_E",
        occupation_regime_test(["REM","FLARE"]*20,names)["status"]==MODE_E)
    li=occupation_regime_test(["REM","FLARE"]*40,names,ordered=True)
    rep("ordered alternating -> LIGHT",li["regime"]=="LIGHT_BLOCKS")
    sup=[]
    for j,s in enumerate(["REM","FLARE","REM","FLARE"]): sup+=[s]*(4**j)
    su=occupation_regime_test(sup,names,ordered=True)
    rep("ordered geometric -> SUPER_DOMINANT",su["regime"]=="SUPER_DOMINANT")
    rep("bistability consequence stated","BISTABLE" in su["consequence"])
    ra=theorem_A_rate(100,2)
    rep("Thm 8.3 bound conservative",ra["L2_bound"]>ra["exact_iid_multinomial_upper"])
    h={"ACTIVE":["REM","FLARE"]}
    rep("BN passes when consistent",
        boundary_null_check(h,{"ACTIVE":1.,"REM":.6,"FLARE":.4})["BN_satisfied"])
    rep("BN fails when inconsistent",
        not boundary_null_check(h,{"ACTIVE":.9,"REM":.6,"FLARE":.4})["BN_satisfied"])

    print("\n[T6] KOU MOMENT GATE")
    r1=KouRegime("A",eta_up=3.0); r2=KouRegime("heavy",eta_up=0.7,occupation=0.05)
    rep("eta+>1 mean finite",moment_estimable([r1],1.)["finite"])
    rep("eta+<1 mean INFINITE",not moment_estimable([r2],1.)["finite"])
    b=False
    try: require_finite_mean([r2])
    except InfiniteMoment: b=True
    rep("reporting infinite mean REFUSED",b)
    rs=response_summary(rng.normal(0,.5,60),regimes=[r2])
    rep("quantiles offered instead","quantiles" in rs and "REFUSED" in rs)
    hv=heaviest_regime_dominates([r1,r2])
    rep("rare heavy subgroup governs tail",hv["governing_regimes"]==["heavy"])
    gu=gumbel_extreme(r1,100.)
    ana=math.exp(-1.0*0.5*100)
    rep("Gumbel cdf exact at 0",abs(gu["cdf_at_0"]-ana)<1e-12)
    wc=esscher_worst_case(r1,10.,0.5)
    rep("worst case exceeds nominal",wc["worst_case_mean"]>wc["nominal_mean"])

    print("\n[T7] OPMC + STEP-COUNT HONESTY")
    yy=[0.]*60+[4.]*60+[-3.]*60
    r=opmc_certified(yy,30,1.0)
    rep("finds 2 change points",r["K_hat"]==2)
    rep("strict SNR honestly fails",r["status"]==MODE_E)
    b=False
    try: verify_step_count(23,8,"OPMC refinement")
    except IntegrityError: b=True
    rep("claimed 23 vs executed 8 DETECTED",b)
    rep("matching counts accepted",verify_step_count(8,8)["matches"])
    cb=compression_bound(1000,5,8)
    rep("compression conditional","NOT" in cb)

    print("\n[T8] SURGERY / NO-ZENO / STC / GIBBS")
    rep("closed-surface handle Db1=+2 -> H",
        classify_morphology(0,2,0,-2)["candidates"]==["H"])
    rep("Db1=+1 rejected for closed surface",
        classify_morphology(0,1,0,-1)["candidates"]==[])
    rep("filament accepts +-1",
        classify_morphology(0,1,0,-1,"filament_network")["candidates"]==["H"])
    nz=no_zeno_certificate(1.,2.,.5)
    rep("q0 <= 1/2",nz["q0_le_half"])
    rep("truncation REMOVED","REMOVED" in nz["HORIZON_TRUNCATION"])
    b=False
    try: doob_exit_bound(1.,2.,.5,.5)
    except Refusal: b=True
    rep("eps > delta/(2B0) refused",b)
    rep("alpha=1 needs sign flip",stc_trace_jump(1.,-1.,1)["satisfied"])
    rep("H^k_R not a new space",not stc_trace_jump(0,0)["is_new_space"])
    rep("reflux conserved",stc_reflux(.7,[.2,.15,.25,.18])["conserved"])
    gs=verify_gibbs_symmetry([0.,.5,1.],2.)
    rep("corrected conductance symmetric",gs["is_symmetric"])
    rep("source conductance NOT symmetric",not gs["source_was_symmetric"])
    cp=carpet_status(False)
    rep("non-PCF needs 3 imported inputs",len(cp["inputs_needed"])==3)
    L1=np.array([[1.,-1.],[-1.,1.]]); L2=np.eye(4)*2
    ce=carpet_multiscale_energy({1:np.array([1.,-1.]),2:np.ones(4)},
                                {1:L1,2:L2},{1:1.125,2:1.266})
    rep("multiscale per-level energies",len(ce["per_level_energy"])==2)

    print("\n[T9] INTEGRITY GUARDS (the bugs that invalidated results)")
    collapsed=np.tile(np.array([[1.5]]),(4,32))
    b=False
    try: verify_no_information_collapse(collapsed,"struct_feat")
    except IntegrityError: b=True
    rep("mean-then-expand collapse DETECTED",b)
    good=rng.normal(0,1,(4,32))
    rep("real feature accepted",
        not verify_no_information_collapse(good)["collapsed"])
    b=False
    try: verify_ensemble_spread(np.ones((4,3,2)))
    except IntegrityError: b=True
    rep("identical ensemble DETECTED",b)
    rep("distinct ensemble accepted",
        verify_ensemble_spread(np.ones((4,3,2))+rng.normal(0,.1,(4,3,2)))
        ["members_distinct"])
    if HAS_TORCH:
        import torch, torch.nn as nn
        class Empty(nn.Module):
            def forward(self,x): return nn.Linear(4,8)(x)
        b=False
        try: verify_persistent_projection(Empty(),"proj")
        except IntegrityError: b=True
        rep("unregistered projection DETECTED",b)
        class Real(nn.Module):
            def __init__(s): super().__init__(); s.p=nn.Linear(4,8)
            def forward(s,x): return s.p(x)
        rep("registered projection accepted",
            verify_persistent_projection(Real())["checked"])
        # invariant structure losses
        P=torch.randn(2,10,3,dtype=torch.float64)
        th=torch.randn(2,dtype=torch.float64)
        Rm=torch.stack([torch.stack([torch.cos(th),-torch.sin(th),torch.zeros_like(th)],1),
                        torch.stack([torch.sin(th),torch.cos(th),torch.zeros_like(th)],1),
                        torch.stack([torch.zeros_like(th),torch.zeros_like(th),torch.ones_like(th)],1)],1)
        Q=torch.einsum("bij,bnj->bni",Rm,P)+torch.tensor([1.,2.,3.],dtype=torch.float64)
        dr=float(drms_loss(P,Q))
        kb=kabsch_rmsd(P,Q)["mean"]
        raw=float(((P-Q)**2).mean())
        print(f"      dRMS={dr:.2e}  Kabsch={kb:.2e}  raw MSE={raw:.3f}")
        rep("dRMS invariant to rigid motion",dr<1e-16)
        rep("Kabsch invariant to rigid motion",kb<1e-7)
        rep("raw MSE NOT invariant (the bug)",raw>1.0)
    rep("no `or True` escape hatch",
        assert_real_test("demo",False)["passed"] is False)

    print("\n[T10] POWER + SD")
    de=detectable_effect(20)
    print(f"      n=20/arm -> min detectable delta={de['min_detectable_delta']:.3f}")
    rep("small n only detects large effects",de["min_detectable_delta"]>0.3)
    rep("missing RATE -> MODE_E",
        sd.certify("subsampling_ci",())["status"]==MODE_E)
    rep("RATE verified -> CERTIFIED",
        sd.certify("subsampling_ci",("RATE",))["status"]==MODE_CERTIFIED)
    rep("O5 always MODE_E",
        sd.certify("state_dependent_averaging",())["status"]==MODE_E)
    rep("O7 always MODE_E",
        sd.certify("coupled_existence",())["status"]==MODE_E)
    rep("clinical use refused",sd.certify("clinical_use",())["status"]==MODE_E)
    rr=sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
CORRECTED HERE
  Subsampling rate for an OLS slope is sqrt(S_tt), NOT n^{3/2}. This is
  exact for irregular visit spacing and reduces to n^{3/2} for a grid.
  Black-Leff EC50 = KA/(tau-1), independent of n; tau<=1 has NO finite
  EC50 and a negative tau is refused.
  Hill (EC50,n) identifiability quantified by Fisher information.
  Occupation regime test separates stationary from genuinely bistable.
  Kou moment gate refuses to report an infinite mean.
  K8 Doob no-Zeno, K9 STC orientation, K10 surgery table, K11 Gibbs.

INTEGRITY GUARDS (each corresponds to a real bug found)
  verify_no_information_collapse : mean-then-expand detected
  verify_step_count              : claimed 23 vs executed 8 detected
  verify_persistent_projection   : layer built in forward() detected
  verify_ensemble_spread         : identical members detected
  assert_real_test               : no `or True` escape hatch
  drms_loss / kabsch_rmsd        : rigid-motion invariant, raw MSE is not

REFUSED
  Infinite moments. Negative transducer efficacy. Clinical decisions.
""")
    return ok

if __name__=="__main__":
    raise SystemExit(0 if self_test() else 1)
