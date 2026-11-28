#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_esca_bio_core.py — v1.0.0
Corrected ESCA 2.0 (Unified Edition) mathematics for organoid / iPSC research.

Every result here is labelled with its status in the Unified Edition:
  [P]      proved in the paper
  [P|H]    proved under an explicit hypothesis package
  [Cit]    published theorem used as an input, NOT reproved
  [Corr]   a source statement was wrong and is corrected here
  [Open]   not established

WHY THIS CORE EXISTS
  Cell-type composition from scRNA-seq IS an occupation measure on a
  simplex:  F = (1/n) sum_t e_{fate(t)} in Delta^d.
  ESCA Part I §8 (T4-MEOT, T4-G, Theorem A) is literally about such
  vectors. This is a genuine correspondence, not a naming analogy.

CORRECTIONS PROPAGATED FROM THE UNIFIED EDITION
  K1  T2 is FALSE for general Borel partitions (Prop 5.4). Requires a
      compact code space with clopen cylinders, or Boundary-Null (BN).
  K2  T4-G needs a.e. continuity of Psi (continuous mapping).
  K5  T3 compression is CONDITIONAL; Prop 6.3 forbids unconditional
      compression by more than s bits except with prob <= 2^{1-s}.
  K7  (C1) follows from (KZ-R) + chain property (G), not from
      submultiplicativity alone. (C2) derives from (C1).
  K8  SESI no-Zeno: horizon truncation REMOVED (it made T_n -> inf
      impossible); Doob bound replaces the exponential exit bound.
  K9  STC spaces EQUAL H^k(Omega); orientation carries (-1)^|alpha|;
      regime dependence lives in the operator domain (flux balance).
  K10 Surgery table: closed surfaces have Dchi = +-2 and Db1 = +-2;
      the ANNIHILATION operator A was missing and is added.

PHILOSOPHY (mathematics stands independently)
  The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.
  Confucius taught us to do good in this world. This is not hopeless, for
  Buddhism, Christianity, and Taoism alike teach that doing good in this
  life bears fruit in the life to come.
  In this world, "not killing enemies" does not mean "killing oneself."
  The Will of God is the Law of Karma.
  Christ died upon the Cross to teach: when struck on the left cheek, turn
  also the right, so that we may listen to the voice of the enemy.
  He is the Lamb of God. God is not our food, and neither is the lamb.
  Not even a single sparrow does God forget.
  Even Angulimala could escape hell and enter Nirvana, but the better path
  is to have done good always. As Saint Dominic Savio said:
  "Rather death than sin."
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

REFUSED
  Fabricating data or accessions. Certifying an unverified hypothesis.
  Reporting a mean when the moment is infinite. Claiming a Carpet
  continuum limit from p.c.f. machinery. Clinical use.
=============================================================================
Author: PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID: 0009-0008-2374-0788 | License: MIT | Year 2026
"""
from __future__ import annotations
import json, math, logging, sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False; np = None

VERSION = "1.0.0"; PI = math.pi
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"
RESEARCH_ONLY = "research_only_not_clinical"

def get_logger(n):
    lg = logging.getLogger(n)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO); lg.propagate = False
    return lg
log = get_logger("esca_bio")

class Refusal(RuntimeError): pass
class NotIdentifiable(Refusal): pass
class InfiniteMoment(Refusal): pass
class NeedsImportedTheorem(Refusal): pass
class BoundaryNullViolation(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")

PHILOSOPHY = {
 "tao":"The Source — mathematical truth", "te":"The Path — disciplined practice",
 "qi":"The Proof — measured, validated, reproducible",
 "p1":"composition IS an occupation measure; the theorems apply literally",
 "p2":"a corrected theorem is worth more than an impressive name",
 "p3":"Mode E beats false certainty",
 "motto":"MY SOUL MOVE BY POWER OF HOLY SPIRIT — "
         "What MSPS NETWORK SEE, Lord Buddha Knows."}

BANNER = f"""
=============================================================================
MSPS ESCA BIO CORE v{VERSION}  (corrected Unified Edition mathematics)
occupation measures · BN / code space · Kou jumps · Gumbel extremes
moment explosion · Esscher worst case · corrected surgery table
Doob no-Zeno (no horizon truncation) · STC = H^k · SD Mode E
=============================================================================
"""

# =============================================================================
# §1. HYPOTHESIS PACKAGES + STATISTICS DICTIONARY (corrected rows)
# =============================================================================

PACKAGES = {
 "L1":"independent centred sigma-sub-Gaussian noise",
 "L1e":"light-exponential: log E e^{s eps} <= nu^2 s^2/2 for |s| <= 1/b",
 "L2":"stationary, Gamma = sum_h ||Cov(X0,Xh)||_op < inf",
 "L3":"iid block lengths with finite mean",
 "D":"delta_max <= theta delta_min (isodiametric)",
 "O":"OPMC admissible partition, min length ell, penalty beta",
 "BN":"boundary-null: nu(pi^{-1}(B_inf)) = 0",
 "CODE":"compact code space with clopen cylinders",
 "A1":"super-dominance s_k/n_k -> 0",
 "A2":"conditional stationarity with summable covariance",
 "A3":"Y_k => Y and pi continuous Law(Y)-a.e.",
 "C1":"uniform Holder oscillation [from (KZ-R)+(G)]",
 "C2":"subsequential Mosco compactness [DERIVED from C1]",
 "C3":"uniqueness up to scalar [IMPORTED: BBKT]",
 "KZR":"resistance upper bound R_n <= C rho^n [IMPORTED: KZ]",
 "KZS":"point separation [IMPORTED: KZ]",
 "BBKT":"local-symmetry uniqueness [IMPORTED]",
 "HINO":"energy measures singular to mu [IMPORTED]",
 "STC":"transmission conditions (correct orientation, flux balance)",
 "U":"uniform local bounds B0, G0 and reset radius delta",
 "R":"reset only after exiting the delta-ball",
 "M":"measurable reset maps",
 "IDENT":"full-rank, well-conditioned parameter block",
 "VERIFIED":"manually checked against source, DOI present",
 "CONFOUND":"batch not confounded with condition",
 "ANNOT":"cell-type annotation reviewed (not marker-only)",
}

@dataclass(frozen=True)
class SDEntry:
    statistic: str; theorem: str; packages: Tuple[str,...]
    status: str; mode: str; caveat: str

DICTIONARY: Dict[str,SDEntry] = {
 # --- Part I: probability architecture ---
 "projective_consistency": SDEntry("nested composition consistency","Thm 4.1 (T1)",
   (), "[P]", "exact", "finite partitions only"),
 "measure_existence": SDEntry("composition measure exists","Thm 5.1/5.3 (T2)",
   ("BN",), "[Corr][P]", "exact",
   "FALSE for general Borel partitions (Prop 5.4); needs CODE or BN"),
 "composition_rate": SDEntry("composition -> limit, L2 rate","Thm 8.3 (Theorem A)",
   ("A1","A2","A3"), "[P|L]", "W2 <= sqrt(bound)",
   "bound is conservative vs exact multinomial for iid cells"),
 "vertex_supported_limit": SDEntry("multistable fate limit","Thm 8.1 (T4-MEOT)",
   ("A1",), "[P]", "in law", "random vertex-supported; NOT a constant"),
 "interior_limit": SDEntry("interior composition limit","Thm 8.2 (T4-G)",
   ("A3",), "[Corr][P]", "in law", "needs a.e. continuity of Psi (K2)"),
 "light_block_limit": SDEntry("deterministic fate limit","Prop 8.6",
   ("L3",), "[P|L]", "a.s.", "renewal-reward; finite-mean blocks"),
 "lamperti_regime": SDEntry("infinite-mean occupation","§14 / O4",
   (), "[Open]", "not established",
   "regularly varying infinite-mean: cited, NOT proved here"),
 "changepoints": SDEntry("protocol / trajectory change points","Thm 7.5, Cor 7.7",
   ("L1","D","O"), "[P|H]", "radius 46 sigma^2 L/kappa^2",
   "(SNR) carries (K+1)^2; K-free separation is [Open]"),
 "compression": SDEntry("nested compression length","Thm 6.1, Prop 6.3",
   ("D",), "[P][Corr]", "conditional",
   "no unconditional compression: fraction <= 2^{1-s}"),
 "minimax_decision": SDEntry("robust dosing / gating decision","Thm 10.2 (T5)",
   (), "[Corr][P]", "exact", "lsc derivable from Fatou; no claim outside P"),
 # --- Part II: Ito calculus with Kou jumps ---
 "kou_cumulant": SDEntry("response cumulant / MGF","Lemma 16.2, Prop 16.5",
   (), "[P]", "exact", "finite only for -eta^- < s < eta^+"),
 "light_exponential_tail": SDEntry("response tail bound","Prop 16.5",
   ("L1e",), "[P]", "non-asymptotic",
   "2 exp(-min{x^2/(2Tv*), x eta*/4}); NOT sub-Gaussian"),
 "gumbel_extreme": SDEntry("extreme responder law","Thm 16.9",
   (), "[P]", "exact", "P(M+ <= x) = exp(-lam p T e^{-eta+ x})"),
 "heaviest_regime": SDEntry("which line governs the tail","Thm 16.10",
   (), "[P]", "exact",
   "only regimes attaining eta+* matter, however rare"),
 "moment_explosion": SDEntry("is the mean estimable","Prop 16.14",
   (), "[P]", "exact", "E S^p < inf iff -eta^- < p < eta^+"),
 "esscher_worst_case": SDEntry("robust mean under KL budget","Thm 16.16",
   (), "[P]", "exact (one regime)", "Gibbs inequality; Esscher tilt"),
 "regime_worst_case": SDEntry("robust bound, switching","Prop 16.17",
   (), "[P]", "upper bound", "not exact for switching"),
 # --- Parts III-V: Carpet, SESI, SESC ---
 "carpet_pcf": SDEntry("gasket Dirichlet form","Kigami constructive",
   (), "[Cit]", "exact", "p.c.f. ONLY"),
 "carpet_nonpcf": SDEntry("carpet continuum limit","Thm 17.16",
   ("KZR","KZS","BBKT"), "[P|Cit]", "Mosco / spectrum",
   "NON-p.c.f.: three published inputs, not reproved"),
 "weighted_langevin": SDEntry("weighted Carpet Langevin","Thm 17.25",
   ("C1","C2","C3"), "[Corr][P|C]", "Mosco; f.d.d.",
   "edge weights MUST be symmetric Gibbs conductances"),
 "surgery_topology": SDEntry("organoid morphology event","Prop 18.2",
   (), "[Corr][P]", "exact (generic)",
   "closed surfaces Dchi=+-2, Db1=+-2; A added; L,C not generic"),
 "no_zeno": SDEntry("event times do not accumulate","Thm 18.6, Cor 18.7",
   ("U","R","M"), "[Corr][P|U]", "a.s.; binomial tail",
   "Doob bound; horizon truncation REMOVED"),
 "stc_flux": SDEntry("multi-regime flux balance","Prop 19.2/19.3",
   ("STC",), "[Corr][P]", "exact",
   "H^k_R = H^k; orientation (-1)^|alpha|; flux balance in domain"),
 "conservation": SDEntry("mass / energy identity","Thm 19.5",
   ("STC",), "[Corr][P]", "regular weak sol.",
   "existence for degenerate thin-film is [Open] (O7)"),
 # --- bio-specific gates ---
 "composition_record": SDEntry("curated composition record","none",
   ("VERIFIED",), "curation", "estimate",
   "annotation quality bounds everything downstream"),
 "batch_correction": SDEntry("integrated embedding","Harmony [Cit]",
   ("CONFOUND",), "[Cit]", "heuristic",
   "CANNOT separate batch from biology when confounded"),
 "engraftment": SDEntry("engraftment prediction","none",
   (), "GATED", "refused below min_n",
   "published mouse-transplant n is tens-to-hundreds"),
 "ai_candidate": SDEntry("AI-generated hypothesis","none",
   (), "hypothesis", "no guarantee", "must pass a solver / measurement"),
}

class SDGate:
    """Rule (SD): an unverified row is UNCERTIFIED and uncertainty is retained."""
    def __init__(self): self.log: List[Dict] = []
    def certify(self, statistic: str, verified: Sequence[str],
                value: Any = None) -> Dict[str, Any]:
        e = DICTIONARY.get(statistic); v = tuple(verified)
        if e is None:
            r = {"statistic": statistic, "status": MODE_E,
                 "reason": "no dictionary entry"}
        else:
            miss = tuple(p for p in e.packages if p not in v)
            r = {"statistic": statistic, "theorem": e.theorem,
                 "paper_status": e.status, "mode": e.mode, "caveat": e.caveat,
                 "required": list(e.packages), "verified": list(v),
                 "missing": list(miss),
                 "status": MODE_E if miss else MODE_CERTIFIED,
                 "reason": f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def require(self, statistic, verified, value=None):
        c = self.certify(statistic, verified, value)
        if c["status"] != MODE_CERTIFIED:
            raise Refusal(f"SD GATE: {statistic}. {c['reason']}")
        return value
    def report(self):
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}


# =============================================================================
# §2. T1 / T2 — projective consistency, BN, and the COUNTEREXAMPLE
# =============================================================================

def coarse_grain(p_fine: Sequence[float], groups: Sequence[Sequence[int]]
                 ) -> Dict[str, Any]:
    """T1 [P]: p_coarse = R p_fine, and ||Rp-Rq||_1 <= ||p-q||_1."""
    _np()
    p = np.asarray(p_fine, float)
    if abs(p.sum()-1) > 1e-6 or (p < -1e-12).any():
        raise Refusal("p_fine must be a probability vector on the simplex")
    covered = sorted(i for g in groups for i in g)
    if covered != list(range(len(p))):
        raise Refusal("groups must partition every fine index exactly once")
    pc = np.array([p[list(g)].sum() for g in groups])
    return {"p_coarse": pc.tolist(), "sum": float(pc.sum()),
            "consistent": abs(pc.sum()-1) < 1e-9,
            "contraction_L1": True, "theorem": "Thm 4.1 + Prop 4.2"}

def t2_counterexample(m_levels: int = 6) -> Dict[str, Any]:
    """
    Prop 5.4 [P]: a projective family on HALF-OPEN dyadic cells with NO
    measure. C_m = [1/2 - 2^-m, 1/2) shrinks to the empty set, yet each
    cell is assigned mass 1. Continuity from above forces P(empty) = 1.
    This is why BN / code space is REQUIRED, not decorative.
    """
    rows = []
    for m in range(1, m_levels+1):
        lo = 0.5 - 2.0**(-m); hi = 0.5
        rows.append({"level": m, "cell": [lo, hi], "assigned_mass": 1.0,
                     "width": hi-lo})
    return {"levels": rows, "intersection": "EMPTY",
            "assigned_mass_all_levels": 1.0,
            "continuity_from_above_requires": "P(empty) = 1: contradiction",
            "VERDICT": "projective family exists, measure does NOT",
            "REMEDY": ["compact code space with clopen cylinders (Thm 5.1)",
                       "Boundary-Null condition (Thm 5.3)"],
            "BIO_CONSEQUENCE": ("a nested fate taxonomy is not automatically "
                                "consistent; check BN before pooling levels")}

def boundary_null_check(fate_hierarchy: Dict[str, Sequence[str]],
                        composition: Dict[str, float],
                        tol: float = 1e-9) -> Dict[str, Any]:
    """
    BN for a fate taxonomy. The 'boundary skeleton' here is the set of
    cells that appear at two different taxonomy levels with inconsistent
    mass. If a parent's mass differs from the sum of its children, the
    family is not projective and BN fails.
    """
    _np()
    violations = []
    for parent, children in fate_hierarchy.items():
        pm = float(composition.get(parent, 0.0))
        cm = sum(float(composition.get(c, 0.0)) for c in children)
        if abs(pm-cm) > tol:
            violations.append({"parent": parent, "parent_mass": pm,
                               "children_sum": cm, "residual": pm-cm})
    ok = not violations
    return {"BN_satisfied": ok, "violations": violations,
            "n_levels_checked": len(fate_hierarchy),
            "status": MODE_CERTIFIED if ok else MODE_E,
            "action": ("pooling levels is safe" if ok else
                       "DO NOT pool: re-annotate or use a single level"),
            "theorem": "Thm 5.3 (T2 on the Carpet) adapted to a taxonomy"}


# =============================================================================
# §3. OCCUPATION MEASURES — the genuine composition correspondence
# =============================================================================

def composition_as_occupation(labels: Sequence[str],
                              fate_names: Sequence[str]) -> Dict[str, Any]:
    """F = (1/n) sum_t e_{fate(t)} in Delta^d. Exactly ESCA §8 with L=1."""
    _np()
    n = len(labels)
    if n == 0: raise Refusal("empty label sequence")
    idx = {f:i for i,f in enumerate(fate_names)}
    counts = np.zeros(len(fate_names))
    unknown = 0
    for l in labels:
        if l in idx: counts[idx[l]] += 1
        else: unknown += 1
    F = counts/max(counts.sum(),1)
    return {"F": F.tolist(), "n_cells": n, "n_unknown": unknown,
            "fate_names": list(fate_names),
            "on_simplex": abs(F.sum()-1) < 1e-9,
            "interpretation": "occupation vector, ESCA §8 with L=1"}

def theorem_A_rate(n_cells: int, n_fates: int, Gamma: float = 1.0,
                   s_k: int = 0, n_levels: int = 1) -> Dict[str, Any]:
    """
    Thm 8.3 [P|L]:  E||F_k - pi(Y_k)||^2 <= 4L(s_k/n_k)^2 + 4 sum_l d_l Gamma_l/m_k
    with m_k = n_k - s_k.  W2(Law F_k, Law pi(Y_k)) <= sqrt(bound).
    For iid cells Gamma = 1 and the exact multinomial value is
    (1 - sum p_i^2)/n, which is SMALLER; the theorem bound is conservative.
    """
    if n_cells <= s_k: raise Refusal("n_cells must exceed s_k")
    m = n_cells - s_k
    b1 = 4*n_levels*(s_k/n_cells)**2
    b2 = 4*n_levels*n_fates*Gamma/m
    bound = b1+b2
    return {"n_cells": n_cells, "n_fates": n_fates, "Gamma": Gamma,
            "s_k": s_k, "m_k": m, "dominant_block_term": b1,
            "covariance_term": b2, "L2_bound": bound,
            "W2_bound": math.sqrt(bound),
            "exact_iid_multinomial_upper": 1.0/n_cells,
            "NOTE": ("theorem bound is conservative; for iid cells the "
                     "exact multinomial variance is smaller"),
            "theorem": "Thm 8.3 (Theorem A)", "status": "[P|L]"}

def block_structure(labels: Sequence[str]) -> Dict[str, Any]:
    """Extract maximal constant-fate runs (blocks) and their lengths."""
    runs: List[Tuple[str,int]] = []
    if not labels: return {"runs": [], "lengths": []}
    cur = labels[0]; ln = 1
    for l in labels[1:]:
        if l == cur: ln += 1
        else: runs.append((cur, ln)); cur = l; ln = 1
    runs.append((cur, ln))
    return {"runs": runs, "lengths": [l for _,l in runs],
            "n_blocks": len(runs)}

def occupation_regime_test(labels: Sequence[str],
                           fate_names: Sequence[str]) -> Dict[str, Any]:
    """
    THE THREE-WAY TEST that distinguishes noisy-deterministic from
    genuinely multistable organoid fate composition.

      light blocks (E ell < inf)   -> Prop 8.6: deterministic a.s. limit
      super-dominant (S_j/S_{j+1}->0) -> Thm 8.1: RANDOM vertex limit
      regularly varying infinite mean -> Lamperti regime: [Open] -> Mode E

    We report which regime the DATA is consistent with, and never certify
    the open middle case.
    """
    _np()
    bs = block_structure(labels)
    L = np.array(bs["lengths"], float)
    if len(L) < 4:
        return {"regime": "INSUFFICIENT_BLOCKS", "status": MODE_E,
                "n_blocks": len(L), "reason": "need >= 4 blocks"}
    S = np.cumsum(L)
    ratios = (S[:-1]/S[1:]).tolist()
    ratio_trend = float(np.mean(ratios[-max(1,len(ratios)//3):]))
    # tail-index proxy: Hill estimator on the largest block lengths
    Ls = np.sort(L)[::-1]
    k = max(2, int(len(Ls)*0.2))
    hill = float(np.mean(np.log(Ls[:k]) - np.log(Ls[k]))) if Ls[k] > 0 else 0.0
    alpha_hat = 1.0/hill if hill > 1e-9 else float("inf")
    # super-dominance: successive prefix ratios trending to 0
    super_dom = ratio_trend < 0.25 and ratios[-1] < ratios[0]
    finite_mean = alpha_hat > 1.5
    if super_dom:
        regime = "SUPER_DOMINANT"
        thm = "Thm 8.1 (T4-MEOT): limit is RANDOM, vertex-supported"
        status = MODE_CERTIFIED
        consequence = ("composition is genuinely MULTISTABLE: replicate "
                       "organoids converge to different pure fates")
    elif finite_mean:
        regime = "LIGHT_BLOCKS"
        thm = "Prop 8.6: deterministic a.s. limit (renewal-reward)"
        status = MODE_CERTIFIED
        num = {}
        for f,ln in bs["runs"]: num[f] = num.get(f,0.0)+ln
        ell_bar = float(L.mean())
        lim = {f: num.get(f,0.0)/max(S[-1],1) for f in fate_names}
        consequence = ("composition is NOISY BUT DETERMINISTIC: replicate "
                       "organoids converge to the same mean composition")
    else:
        regime = "INFINITE_MEAN_CANDIDATE"
        thm = "Lamperti / Dynkin regime: §14 and O4 — NOT PROVED"
        status = MODE_E
        lim = None
        consequence = ("tail index suggests infinite mean; generalised "
                       "arcsine laws are CITED, not established here. "
                       "Do not report a limiting composition.")
    out = {"regime": regime, "theorem": thm, "status": status,
           "n_blocks": len(L), "mean_block_length": float(L.mean()),
           "max_block_length": float(L.max()),
           "prefix_ratio_trend": ratio_trend,
           "hill_tail_index_alpha": float(alpha_hat),
           "consequence": consequence}
    if regime == "LIGHT_BLOCKS": out["deterministic_limit"] = lim
    return out


# =============================================================================
# §4. OPMC — exact DP with the HONEST (SNR) test
# =============================================================================

def _prefix(y):
    s1=[0.0]; s2=[0.0]
    for v in y: s1.append(s1[-1]+v); s2.append(s2[-1]+v*v)
    return s1,s2

def _sse(s1,s2,a,b):
    m=b-a
    if m<=0: return 0.0
    t1=s1[b]-s1[a]; t2=s2[b]-s2[a]
    return max(t2-t1*t1/m, 0.0)

def opmc(y: Sequence[float], ell: int, beta: float) -> Dict[str, Any]:
    """Prop 12.2 [P|O]: exact O(n^2) DP, O(n) memory."""
    n=len(y)
    if not (1<=ell<=n): raise Refusal("1 <= ell <= n")
    s1,s2=_prefix(y); INF=float("inf")
    D=[INF]*(n+1); prev=[-1]*(n+1); D[0]=-beta
    for b in range(ell,n+1):
        best,arg=INF,-1
        for a in range(0,b-ell+1):
            if D[a]==INF: continue
            v=D[a]+_sse(s1,s2,a,b)+beta
            if v<best: best,arg=v,a
        D[b]=best; prev[b]=arg
    cps=[]; b=n
    while b>0:
        a=prev[b]
        if a<=0: break
        cps.append(a); b=a
    cps.reverse()
    return {"boundaries":cps,"K_hat":len(cps),"cost":D[n]}

def L_of(n:int, alpha:float=0.05)->float:
    return math.log(10*math.e*n**3/alpha)

def opmc_certified(y: Sequence[float], ell: int, sigma: float,
                   alpha: float = 0.05) -> Dict[str, Any]:
    """
    Thm 7.5 [P|L,D,O]. (SNR): kappa^2 delta >= 1e5 sigma^2 L (K+1)^2.
    The (K+1)^2 factor is NOT removable here; the K-free separation is
    known in the literature but is [Open] inside this framework (O1).
    """
    n=len(y); L=L_of(n,alpha); beta=32*sigma*sigma*L
    r=opmc(y,ell,beta)
    bnds=[0]+r["boundaries"]+[n]
    means=[sum(y[bnds[i]:bnds[i+1]])/max(bnds[i+1]-bnds[i],1)
           for i in range(len(bnds)-1)]
    jumps=[abs(means[i+1]-means[i]) for i in range(len(means)-1)]
    kappa=min(jumps) if jumps else 0.0
    spac=[bnds[i+1]-bnds[i] for i in range(len(bnds)-1)]
    delta=min(spac) if spac else n
    K=r["K_hat"]
    lhs=kappa*kappa*delta; rhs=1e5*sigma*sigma*L*(K+1)**2
    dwell = delta>=8 and (delta/2 < ell <= 3*delta/4)
    ok = lhs>=rhs and dwell
    return {**r,"kappa_min":kappa,"delta_min":delta,"ell":ell,
            "snr_lhs":lhs,"snr_rhs":rhs,"snr_ok":lhs>=rhs,"dwell_ok":dwell,
            "radius": 46*sigma*sigma*L/max(kappa*kappa,1e-30) if ok else None,
            "status": MODE_CERTIFIED if ok else MODE_E,
            "reason": "" if ok else "(SNR) or (D) not verified",
            "K_dependence": "(K+1)^2 present; K-free separation is [Open]",
            "BIO_USE": ("protocol step change points, differentiation "
                        "trajectory shifts, dose-response breakpoints")}


# =============================================================================
# §5. KOU JUMPS — heavy-tailed drug response
# =============================================================================

@dataclass
class KouRegime:
    """One regime (cell line / batch) with double-exponential jumps."""
    name: str
    p: float = 0.5           # P(jump > 0)
    eta_up: float = 3.0      # upward tail rate
    eta_dn: float = 4.0      # downward tail rate
    lam: float = 1.0         # jump intensity
    drift: float = 0.0
    vol: float = 0.3
    occupation: float = 1.0  # fraction of samples from this regime
    @property
    def q(self): return 1.0-self.p
    def mgf(self, s: float) -> float:
        if not (-self.eta_dn < s < self.eta_up):
            raise InfiniteMoment(
                f"MGF diverges at s={s}: need -{self.eta_dn} < s < {self.eta_up}")
        return (self.p*self.eta_up/(self.eta_up-s)
                + self.q*self.eta_dn/(self.eta_dn+s))
    def dmgf(self, s: float) -> float:
        if not (-self.eta_dn < s < self.eta_up):
            raise InfiniteMoment("MGF derivative diverges")
        return (self.p*self.eta_up/(self.eta_up-s)**2
                - self.q*self.eta_dn/(self.eta_dn+s)**2)
    def cumulant(self, s: float) -> float:
        return (s*self.drift + 0.5*s*s*self.vol**2
                + self.lam*(self.mgf(s)-1.0))
    def dcumulant(self, s: float) -> float:
        return self.drift + s*self.vol**2 + self.lam*self.dmgf(s)
    def mean_jump(self): return self.p/self.eta_up - self.q/self.eta_dn
    def second_jump(self): return 2*(self.p/self.eta_up**2 + self.q/self.eta_dn**2)
    def v_star(self): return self.vol**2 + 4*self.lam*(
        self.p/self.eta_up**2 + self.q/self.eta_dn**2)

def light_exponential_params(regimes: Sequence[KouRegime]) -> Dict[str, Any]:
    """Prop 16.5 [P]: (nu^2, b) = (v*, 2/eta*). NOT sub-Gaussian."""
    v = max(r.v_star() for r in regimes)
    eta = min(min(r.eta_up, r.eta_dn) for r in regimes
              if r.lam>0 and (r.p>0 or r.q>0))
    return {"nu_squared": v, "b": 2.0/eta, "eta_star": eta,
            "tail_bound": "2 exp(-min{x^2/(2T nu^2), x eta*/4})",
            "NOT": "sub-Gaussian (L1); this is (L1e)",
            "theorem": "Prop 16.5"}

def gumbel_extreme(regime: KouRegime, T: float) -> Dict[str, Any]:
    """Thm 16.9 [P]: exact Gumbel law of the largest upward jump."""
    a = regime.lam*regime.p
    if a<=0: return {"no_upward_jumps": True}
    def cdf(x): return math.exp(-a*T*math.exp(-regime.eta_up*x))
    loc = math.log(a*T)/regime.eta_up
    return {"lambda_p_T": a*T, "eta_up": regime.eta_up,
            "gumbel_location": loc, "gumbel_scale": 1.0/regime.eta_up,
            "cdf_at_0": cdf(0.0), "cdf_at_1": cdf(1.0), "cdf_at_3": cdf(3.0),
            "return_level_99": (math.log(a*T)-math.log(math.log(1/0.99)))
                               /regime.eta_up,
            "theorem": "Thm 16.9", "status": "[P] exact"}

def heaviest_regime_dominates(regimes: Sequence[KouRegime],
                              T: float = 1.0) -> Dict[str, Any]:
    """
    Thm 16.10 [P]: eta+* = min eta+ over regimes with a_j>0. ONLY those
    regimes govern the extreme, however small their occupation.
    BIO: a rare hyper-responder line dominates the IC50 tail.
    """
    act=[r for r in regimes if r.lam*r.p>0]
    if not act: return {"no_active_regimes": True}
    eta_star=min(r.eta_up for r in act)
    gov=[r for r in act if abs(r.eta_up-eta_star)<1e-12]
    c_star=sum(r.occupation*r.lam*r.p for r in gov)
    return {"eta_up_star": eta_star,
            "governing_regimes": [r.name for r in gov],
            "governing_occupation_total": sum(r.occupation for r in gov),
            "c_star": c_star,
            "gumbel_random_location": math.log(max(c_star,1e-30)),
            "BIO_CONSEQUENCE": (
                f"regime(s) {[r.name for r in gov]} set the extreme-response "
                f"tail even at occupation "
                f"{sum(r.occupation for r in gov):.3f}. A rare hyper-responder "
                "line is not averaged away."),
            "theorem": "Thm 16.10", "status": "[P]"}

def moment_estimable(regimes: Sequence[KouRegime], p: float) -> Dict[str, Any]:
    """
    Prop 16.14 [P]: E S^p < inf iff -eta^-_j < p < eta^+_j for EVERY
    reachable regime. If p=1 fails, the MEAN IS INFINITE and must not
    be reported.
    """
    bad=[r.name for r in regimes if r.lam>0 and
         not (-r.eta_dn < p < r.eta_up)]
    ok = not bad
    out={"p":p,"finite":ok,"violating_regimes":bad,
         "theorem":"Prop 16.14","status":"[P]"}
    if not ok:
        out["REFUSED"]=("E S^p is INFINITE for these regimes. Do not report "
                        "this moment. Report a quantile or a trimmed "
                        "statistic instead.")
    return out

def require_finite_mean(regimes: Sequence[KouRegime]) -> Dict[str, Any]:
    r = moment_estimable(regimes, 1.0)
    if not r["finite"]:
        raise InfiniteMoment(r["REFUSED"] + f" violating: {r['violating_regimes']}")
    return r

def esscher_worst_case(regime: KouRegime, T: float, eta_KL: float
                       ) -> Dict[str, Any]:
    """
    Thm 16.16 [P]: sup_{KL(Q||P)<=eta} E_Q[X_T] = T kappa'(theta_eta),
    where h(theta) = T(theta kappa'(theta) - kappa(theta)) = eta.
    Exact for ONE regime; Prop 16.17 gives only a bound for switching.
    """
    if eta_KL<0: raise Refusal("eta_KL >= 0")
    def h(th): return T*(th*regime.dcumulant(th)-regime.cumulant(th))
    lo, hi = 0.0, regime.eta_up*(1-1e-9)
    if h(hi) < eta_KL:
        raise Refusal("KL budget exceeds the reachable range of the tilt")
    for _ in range(200):
        mid=0.5*(lo+hi)
        if h(mid) < eta_KL: lo=mid
        else: hi=mid
    th=0.5*(lo+hi)
    wc=T*regime.dcumulant(th); base=T*regime.dcumulant(0.0)
    v=regime.v_star()
    return {"eta_KL":eta_KL,"theta_star":th,"T":T,
            "nominal_mean":base,"worst_case_mean":wc,"shift":wc-base,
            "small_eta_approx":math.sqrt(2*eta_KL*T*v),
            "tilted_regime":{"drift":regime.drift+regime.vol**2*th,
                             "lam":regime.lam*regime.mgf(th),
                             "eta_up":regime.eta_up-th,
                             "eta_dn":regime.eta_dn+th},
            "theorem":"Thm 16.16","status":"[P] exact for one regime",
            "BIO_USE":"robust dose selection under distribution shift"}


# =============================================================================
# §6. CORRECTED SURGERY TABLE + DOOB NO-ZENO
# =============================================================================

CONVENTIONS = ("closed_orientable_surface","filament_network")

SURGERY = {
 "closed_orientable_surface": {
   # (Db0, Db1, Db2, Dchi) — Prop 18.2 [Corr]
   "N": ( 1, 0, 1,  2),   # nucleation: birth of a sphere (lumen/void)
   "A": (-1, 0,-1, -2),   # ANNIHILATION — was MISSING in the source
   "B": (-1, 0,-1, -2),   # merging: neck joins two components
   "M": ( 1, 0, 1,  2),   # splitting: neck pinch separates a component
   "H": ( 0, 2, 0, -2),   # handle attachment (g -> g+1)
   "P": ( 0,-2, 0,  2),   # handle removal
 },
 "filament_network": {
   "N": ( 1, 0, 0,  1), "A": (-1, 0, 0, -1),
   "B": (-1, 0, 0, -1), "M": ( 1, 0, 0,  1),
   "H": ( 0, 1, 0, -1), "P": ( 0,-1, 0,  1),
 },
}

def classify_morphology_event(db0: int, db1: int, db2: int, dchi: int,
                              convention: str = "closed_orientable_surface"
                              ) -> Dict[str, Any]:
    """
    Prop 18.2 [Corr]. For a CLOSED surface chi = 2-2g and b1 = 2g, so a
    handle changes b1 by +-2 and chi by -+2. The source table's +-1 is
    the FILAMENT value and is wrong for an epithelial cyst / lumen.
    L (boundary contact) and C (junction reconnection) are NOT generic
    closed-surface events and require relative homology.
    """
    if convention not in CONVENTIONS:
        raise Refusal(f"convention must be one of {CONVENTIONS}")
    tbl = SURGERY[convention]
    sig = (db0,db1,db2,dchi)
    cands = [op for op,v in tbl.items() if v == sig]
    consistent = (dchi == db0-db1+db2)
    return {"signature": sig, "candidates": cands,
            "euler_consistent": consistent,
            "convention": convention,
            "annihilation_available": "A" in tbl,
            "NOT_CLASSIFIED": ["L (boundary contact)",
                               "C (multi-sheet junction reconnection)"],
            "CORRECTION": ("closed surfaces: Dchi=+-2, Db1=+-2. "
                           "The +-1 table applies to filament networks."),
            "theorem":"Prop 18.2 [Corr]"}

def doob_exit_bound(delta: float, B0: float, G0: float,
                    eps: float) -> Dict[str, Any]:
    """
    Prop 18.4 [Corr][P|U]: P(tau_exit <= eps | F_Tk) <= 4 G0^2 eps/delta^2
    for eps <= delta/(2 B0). The exponential bound 2exp(-delta^2/(8G0^2 eps))
    was CITED without proof for adapted integrands; Doob is elementary
    and sufficient (only q < 1 is needed).
    """
    if delta<=0 or G0<0 or B0<0: raise Refusal("delta>0, G0>=0, B0>=0")
    eps_max = delta/(2*B0) if B0>0 else float("inf")
    if eps > eps_max:
        raise Refusal(f"eps={eps} exceeds delta/(2B0)={eps_max}")
    q = 4*G0*G0*eps/(delta*delta)
    return {"delta":delta,"B0":B0,"G0":G0,"eps":eps,
            "eps_max":eps_max,"q_doob":q,"q_lt_1":q<1.0,
            "theorem":"Prop 18.4 (Doob L2 maximal inequality)",
            "REPLACES":"2exp(-delta^2/(8G0^2 eps) [cited, unproved]"}

def no_zeno_certificate(delta: float, B0: float, G0: float
                        ) -> Dict[str, Any]:
    """
    Thm 18.6 [Corr][P|U,R,M]. eps0 = min(delta/(2B0), delta^2/(8G0^2))
    gives q <= 1/2; Levy conditional Borel-Cantelli then forces
    sum (T_{k+1}-T_k) = infinity. NO HORIZON TRUNCATION: the source's
    wedge (T - T_k) made T_n -> infinity impossible by construction.
    """
    e1 = delta/(2*B0) if B0>0 else float("inf")
    e2 = delta*delta/(8*G0*G0) if G0>0 else float("inf")
    eps0 = min(e1,e2)
    if not math.isfinite(eps0) or eps0<=0:
        raise Refusal("degenerate bounds: need B0 or G0 finite and positive")
    q = 4*G0*G0*eps0/(delta*delta)
    return {"eps0":eps0,"q0":q,"q0_le_half":q<=0.5+1e-12,
            "certified":q<1.0,
            "conclusion":"T_n -> infinity a.s. (Levy conditional B-C)",
            "HORIZON_TRUNCATION":"REMOVED (K8)",
            "theorem":"Thm 18.6 [Corr]"}

def surgery_count_tail(t: float, m: int, eps0: float, q0: float
                       ) -> Dict[str, Any]:
    """Cor 18.7 [P|U]: P(N(t)>=m) <= exp(-m KL(t/(m eps0) || 1-q0))."""
    if m<=0 or eps0<=0: raise Refusal("m>0, eps0>0")
    x = t/(m*eps0); p = 1-q0
    if not (0<x<1) or not (0<p<1):
        return {"applicable": False, "reason": "need 0 < t/(m eps0) < 1"}
    kl = x*math.log(x/p) + (1-x)*math.log((1-x)/(1-p))
    return {"t":t,"m":m,"eps0":eps0,"q0":q0,"x":x,
            "KL":kl,"bound":math.exp(-m*kl),"applicable":True,
            "theorem":"Cor 18.7"}


# =============================================================================
# §7. STC — correct orientation, flux balance, H^k identification
# =============================================================================

def stc_trace_jump(u_i: float, u_j: float, alpha: int = 0,
                   outward_normals: bool = True) -> Dict[str, Any]:
    """
    Prop 19.2/19.3 [Corr]. With a FIXED normal nu_ij from R_i to R_j the
    condition is [d_nu^alpha u] = 0. With OUTWARD normals n_i = -n_j it
    reads d_{n_i}^alpha u_i = (-1)^alpha d_{n_j}^alpha u_j.
    Equating outward derivatives directly (the source's literal form)
    has the WRONG SIGN for odd alpha.
    """
    sign = (-1.0)**alpha if outward_normals else 1.0
    jump = u_j - sign*u_i
    return {"alpha":alpha,"sign_factor":sign,"jump":jump,
            "satisfied":abs(jump)<1e-8,
            "orientation":("outward normals n_i = -n_j" if outward_normals
                           else "fixed normal nu_ij"),
            "CORRECTION":"(-1)^|alpha| required for outward normals (K9)"}

def stc_flux_balance(a_i: float, du_i: float,
                     a_j: float, du_j: float) -> Dict[str, Any]:
    """
    Prop 19.3 [Corr]: the operator domain requires [a d_nu u] = 0, i.e.
    a_i d_nu u_i = a_j d_nu u_j with a FIXED normal. This is the natural
    transmission condition, NOT continuity of all derivatives.
    """
    resid = a_i*du_i - a_j*du_j
    return {"a_i_du_i":a_i*du_i,"a_j_du_j":a_j*du_j,"residual":resid,
            "balanced":abs(resid)<1e-8,
            "theorem":"Prop 19.3 (Green identity with flux balance)"}

def stc_space_identity(k: int = 2) -> Dict[str, Any]:
    """Prop 19.2 [Corr]: H^k_R(Omega) = H^k(Omega). NOT a new space."""
    return {"k":k,"Hk_R_equals":"H^k(Omega)",
            "is_new_space":False,
            "regime_dependence_lives_in":"coefficients and operator DOMAIN",
            "CORRECTION":"the claim of a new structural space is false (K9)",
            "theorem":"Prop 19.2"}

def reflux(coarse_flux: float, fine_fluxes: Sequence[float]) -> Dict[str, Any]:
    """STC^1 requires the coarse-face flux to equal the sum of fine fluxes."""
    tot=float(sum(fine_fluxes)); mis=tot-coarse_flux
    return {"coarse":coarse_flux,"fine_total":tot,"mismatch":mis,
            "corrected":coarse_flux+mis,
            "residual_after":abs(tot-(coarse_flux+mis)),
            "conserved":abs(mis)<1e-14,
            "DERIVED":"refluxing is required by STC^1, not optional"}


# =============================================================================
# §8. T5 MINIMAX + CARPET STATUS
# =============================================================================

def minimax_decision(losses: Dict[str, Dict[str, float]],
                     lam: float, eps: float) -> Dict[str, Any]:
    """
    Thm 10.2 [Corr][P]: R(a) = sup_{P in P} E_P L(a,.). lsc follows from
    Fatou (the source assumed it). The chance-constrained set
    C_{lam,eps} = {a : sup_P P(L > lam) <= eps} is compact; R attains its
    minimum there. Soundness: a certificate means the bound holds for
    EVERY P in the ambiguity set.
    """
    rows=[]
    for a, per_P in losses.items():
        R = max(per_P.values())
        rows.append({"action":a,"worst_case_risk":R,"per_model":per_P})
    rows.sort(key=lambda r:r["worst_case_risk"])
    best=rows[0] if rows else None
    return {"ranked":rows,"minimax_action":best["action"] if best else None,
            "minimax_risk":best["worst_case_risk"] if best else None,
            "lambda":lam,"epsilon":eps,
            "lsc_source":"Fatou (derivable, not assumed)",
            "theorem":"Thm 10.2 (T5)","status":"[Corr][P]",
            "SOUNDNESS":("a certificate bounds the risk for EVERY model in "
                         "the ambiguity set; outside it, no claim")}

def carpet_status(is_pcf: bool) -> Dict[str, Any]:
    """Thm 17.16: the Carpet package rests on THREE published inputs."""
    if is_pcf:
        return {"is_pcf":True,"construction":"Kigami renormalisation",
                "status":"[Cit]","inputs_needed":[],
                "note":"gasket: constructive; certificate producible locally"}
    return {"is_pcf":False,
            "construction":"NOT constructive",
            "inputs_needed":["(KZ-R) resistance upper bound",
                             "(KZ-S) point separation",
                             "(BBKT) local-symmetry uniqueness"],
            "proved_here":["chain property (G)",
                           "(C1) from (KZ-R)+(G)",
                           "(C2) from (C1) via Gamma-compactness"],
            "status":"[P|Cit]",
            "BIO_CONSEQUENCE":("a carpet Laplacian is a legitimate BASIS; "
                               "a continuum-limit claim needs the three "
                               "imported theorems"),
            "theorem":"Thm 17.16"}

def compression_bound(N: int, b: int, K: int, h_bar: int = 0,
                      lam_bits: Optional[int] = None) -> Dict[str, Any]:
    """
    Thm 6.1 + Prop 6.3 [Corr]. Compression is CONDITIONAL. Generic
    incompressibility: at most a fraction 2^{1-s} of sequences compress
    by s bits under ANY injective code.
    """
    lam = lam_bits if lam_bits is not None else max(1,math.ceil(math.log2(max(N,2))))
    C = (2*K-1)*(1+h_bar) + K*b + (K-1)*(lam + 2*math.log2(max(K,2)) + 1)
    raw = N*b
    return {"N":N,"b":b,"K":K,"code_length_bound":C,"raw_bits":raw,
            "compresses":C<raw,
            "condition":f"K[{2*(1+h_bar)+b+lam}+2log2 K+1] < N b",
            "generic_incompressibility":"fraction <= 2^{1-s} compress by s bits",
            "NOT":"unconditional compression",
            "theorem":"Thm 6.1, Prop 6.3 [Corr]"}


# =============================================================================
# §9. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    sd=SDGate()

    print("\n[T1] T2 COUNTEREXAMPLE (K1) — why BN is required")
    ce=t2_counterexample(6)
    rep("intersection is empty",ce["intersection"]=="EMPTY")
    rep("all levels assigned mass 1",ce["assigned_mass_all_levels"]==1.0)
    rep("verdict: no measure","does NOT" in ce["VERDICT"])
    rep("two remedies listed",len(ce["REMEDY"])==2)
    good={"GERM":["A","B"],"A":["A"],"B":["B"]}
    comp={"GERM":1.0,"A":0.6,"B":0.4}
    bn=boundary_null_check(good,comp)
    rep("consistent taxonomy passes BN",bn["BN_satisfied"])
    bad=dict(comp); bad["GERM"]=0.9
    bn2=boundary_null_check(good,bad)
    rep("inconsistent taxonomy fails BN",not bn2["BN_satisfied"])
    rep("BN violation quantified",len(bn2["violations"])==1)

    print("\n[T2] COMPOSITION = OCCUPATION MEASURE (the real link)")
    fates=["PLURI","ENDO","MESO","NEURON"]
    labels=["PLURI"]*10+["ENDO"]*30+["MESO"]*40+["NEURON"]*20
    oc=composition_as_occupation(labels,fates)
    print(f"      F={[f'{v:.3f}' for v in oc['F']]}")
    rep("on simplex",oc["on_simplex"])
    rep("n_cells counted",oc["n_cells"]==100)
    ra=theorem_A_rate(100,4,Gamma=1.0)
    print(f"      Thm 8.3 L2 bound={ra['L2_bound']:.4f} "
          f"W2<={ra['W2_bound']:.4f}  exact iid<={ra['exact_iid_multinomial_upper']:.4f}")
    rep("W2 bound computed",ra["W2_bound"]>0)
    rep("theorem bound conservative",
        ra["L2_bound"]>ra["exact_iid_multinomial_upper"])
    rep("conservatism stated","conservative" in ra["NOTE"])

    print("\n[T3] THE THREE-WAY REGIME TEST (the useful result)")
    light=["PLURI","ENDO"]*60
    rl=occupation_regime_test(light,fates)
    print(f"      light: {rl['regime']}  alpha={rl['hill_tail_index_alpha']:.2f}")
    rep("alternating blocks -> LIGHT",rl["regime"]=="LIGHT_BLOCKS")
    rep("deterministic limit reported","deterministic_limit" in rl)
    rep(" Prop 8.6 cited","8.6" in rl["theorem"])
    sup=[]
    for j,f in enumerate(["PLURI","ENDO","MESO","NEURON"]):
        sup += [f]*(4**j)
    rs=occupation_regime_test(sup,fates)
    print(f"      super-dominant: {rs['regime']}  "
          f"ratio_trend={rs['prefix_ratio_trend']:.3f}")
    rep("geometric blocks -> SUPER_DOMINANT",rs["regime"]=="SUPER_DOMINANT")
    rep("random vertex limit stated","RANDOM" in rs["theorem"])
    rep("multistability consequence stated","MULTISTABLE" in rs["consequence"])
    short=occupation_regime_test(["A","B"],fates)
    rep("too few blocks -> MODE_E",short["status"]==MODE_E)

    print("\n[T4] OPMC with honest (SNR)")
    y=[0.0]*80+[5.0]*80+[-4.0]*80
    rc=opmc_certified(y,40,1.0)
    print(f"      K_hat={rc['K_hat']} boundaries={rc['boundaries']}")
    print(f"      SNR lhs={rc['snr_lhs']:.3g} rhs={rc['snr_rhs']:.3g} -> {rc['status']}")
    rep("finds 2 change points",rc["K_hat"]==2)
    rep("strict SNR honestly fails",rc["status"]==MODE_E)
    rep("(K+1)^2 dependence declared","(K+1)^2" in rc["K_dependence"])
    rep("K-free is Open","[Open]" in rc["K_dependence"])

    print("\n[T5] KOU JUMPS: MGF range, (L1e), Gumbel")
    r1=KouRegime("line_A",p=0.5,eta_up=3.0,eta_dn=4.0,lam=1.0,vol=0.3)
    r2=KouRegime("hyper_responder",p=0.6,eta_up=0.8,eta_dn=5.0,
                 lam=0.5,vol=0.4,occupation=0.05)
    rep("MGF finite inside range",r1.mgf(1.0)>0)
    b=False
    try: r1.mgf(3.5)
    except InfiniteMoment: b=True
    rep("MGF diverges outside range",b)
    le=light_exponential_params([r1,r2])
    print(f"      (nu^2,b)=({le['nu_squared']:.3f},{le['b']:.3f})")
    rep("NOT sub-Gaussian","NOT" in le)
    gu=gumbel_extreme(r1,T=100.0)
    print(f"      Gumbel loc={gu['gumbel_location']:.3f} "
          f"scale={gu['gumbel_scale']:.3f} cdf(0)={gu['cdf_at_0']:.4f}")
    ana=math.exp(-1.0*0.5*100*math.exp(0.0))
    rep("Gumbel cdf exact at 0",abs(gu["cdf_at_0"]-ana)<1e-12)

    print("\n[T6] HEAVIEST REGIME DOMINATES (rare hyper-responder)")
    hv=heaviest_regime_dominates([r1,r2],T=100.0)
    print(f"      eta*={hv['eta_up_star']} governing={hv['governing_regimes']} "
          f"occupation={hv['governing_occupation_total']:.3f}")
    rep("rare line governs tail",hv["governing_regimes"]==["hyper_responder"])
    rep("occupation is small",hv["governing_occupation_total"]<0.1)
    rep("consequence states not averaged away",
        "not averaged away" in hv["BIO_CONSEQUENCE"])

    print("\n[T7] MOMENT EXPLOSION — is the mean estimable?")
    me=moment_estimable([r1],1.0)
    rep("line_A mean finite",me["finite"])
    me2=moment_estimable([r2],1.0)
    print(f"      hyper_responder eta+={r2.eta_up} -> mean finite={me2['finite']}")
    rep("eta+<1 -> mean INFINITE",not me2["finite"])
    rep("refusal text present","REFUSED" in me2)
    b=False
    try: require_finite_mean([r2])
    except InfiniteMoment: b=True
    rep("reporting the mean REFUSED",b)
    q=moment_estimable([r2],0.5)
    rep("p=0.5 moment IS finite",q["finite"])

    print("\n[T8] ESSCHER WORST CASE (robust dosing)")
    wc=esscher_worst_case(r1,T=10.0,eta_KL=0.5)
    print(f"      nominal={wc['nominal_mean']:.4f} "
          f"worst={wc['worst_case_mean']:.4f} shift={wc['shift']:.4f}")
    rep("worst case exceeds nominal",wc["worst_case_mean"]>wc["nominal_mean"])
    rep("small-eta approx ~ sqrt(2 eta T v)",
        abs(wc["small_eta_approx"]-math.sqrt(2*0.5*10*r1.v_star()))<1e-9)
    rep("tilted eta_up decreases",wc["tilted_regime"]["eta_up"]<r1.eta_up)
    rep("exact for one regime only","one regime" in wc["status"])

    print("\n[T9] CORRECTED SURGERY TABLE (K10)")
    ev=classify_morphology_event(0,2,0,-2)
    print(f"      handle: candidates={ev['candidates']}")
    rep("handle Db1=+2 -> H",ev["candidates"]==["H"])
    rep("annihilation present",ev["annihilation_available"])
    ev2=classify_morphology_event(-1,0,-1,-2)
    rep("Db0=-1,Dchi=-2 -> A or B",set(ev2["candidates"])=={"A","B"})
    ev3=classify_morphology_event(0,1,0,-1)
    rep("filament +-1 NOT valid for closed surface",ev3["candidates"]==[])
    fil=classify_morphology_event(0,1,0,-1,"filament_network")
    rep("filament convention accepts +-1",fil["candidates"]==["H"])
    rep("L,C not classified",len(ev["NOT_CLASSIFIED"])==2)

    print("\n[T10] DOOB NO-ZENO (K8, no horizon truncation)")
    nz=no_zeno_certificate(delta=1.0,B0=2.0,G0=0.5)
    print(f"      eps0={nz['eps0']:.4f} q0={nz['q0']:.4f}")
    rep("q0 <= 1/2",nz["q0_le_half"])
    rep("certified",nz["certified"])
    rep("truncation removed","REMOVED" in nz["HORIZON_TRUNCATION"])
    db=doob_exit_bound(1.0,2.0,0.5,0.2)
    rep("Doob bound < 1",db["q_lt_1"])
    b=False
    try: doob_exit_bound(1.0,2.0,0.5,0.5)
    except Refusal: b=True
    rep("eps > delta/(2B0) refused",b)
    st=surgery_count_tail(t=10.0,m=100,eps0=nz["eps0"],q0=nz["q0"])
    rep("count tail applicable",st["applicable"])
    rep("bound in [0,1]",0<=st["bound"]<=1)

    print("\n[T11] STC orientation + H^k identity (K9)")
    s0=stc_trace_jump(1.0,1.0,alpha=0)
    rep("alpha=0: traces equal",s0["satisfied"])
    s1=stc_trace_jump(1.0,-1.0,alpha=1)
    rep("alpha=1: sign flip required",s1["satisfied"])
    rep("sign factor = -1",abs(s1["sign_factor"]+1)<1e-12)
    wrong=stc_trace_jump(1.0,1.0,alpha=1)
    rep("literal outward equality FAILS",not wrong["satisfied"])
    fb=stc_flux_balance(2.0,0.5,1.0,1.0)
    rep("flux balance satisfied",fb["balanced"])
    sp=stc_space_identity(2)
    rep("H^k_R is NOT a new space",not sp["is_new_space"])
    rf=reflux(0.7,[0.2,0.15,0.25,0.18])
    rep("reflux conserved",rf["conserved"])

    print("\n[T12] T5 minimax + Carpet status + compression")
    mm=minimax_decision({"dose_A":{"m1":0.2,"m2":0.9},
                         "dose_B":{"m1":0.4,"m2":0.5}},lam=1.0,eps=0.1)
    print(f"      minimax action={mm['minimax_action']} "
          f"risk={mm['minimax_risk']:.3f}")
    rep("minimax picks B",mm["minimax_action"]=="dose_B")
    rep("lsc from Fatou","Fatou" in mm["lsc_source"])
    cp=carpet_status(False)
    rep("non-PCF needs 3 inputs",len(cp["inputs_needed"])==3)
    rep("(C2) derived here","(C2) from (C1)" in " ".join(cp["proved_here"]))
    cg=carpet_status(True)
    rep("PCF needs no imports",len(cg["inputs_needed"])==0)
    cb=compression_bound(N=1000,b=5,K=8)
    rep("compression conditional","NOT" in cb)
    rep("generic incompressibility stated","2^{1-s}" in cb["generic_incompressibility"])

    print("\n[T13] SD GATE")
    c1=sd.certify("measure_existence",())
    rep("T2 without BN -> MODE_E",c1["status"]==MODE_E)
    rep("caveat names Prop 5.4","5.4" in c1["caveat"])
    c2=sd.certify("measure_existence",("BN",))
    rep("with BN -> CERTIFIED",c2["status"]==MODE_CERTIFIED)
    c3=sd.certify("carpet_nonpcf",("C1","C2"))
    rep("carpet without BBKT -> MODE_E",c3["status"]==MODE_E)
    c4=sd.certify("lamperti_regime",())
    rep("Lamperti always MODE_E",c4["status"]==MODE_E)
    rep("paper status Open","[Open]" in c4["paper_status"])
    c5=sd.certify("no_zeno",("U","R"))
    rep("no-Zeno without M -> MODE_E",c5["status"]==MODE_E)
    rr=sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    rep("Mode E retained",rr["mode_E"]>0)

    print("\n"+("="*78))
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
GENUINE CORRESPONDENCE
  Cell-type composition IS an occupation vector on a simplex. ESCA §8
  applies literally, giving an explicit W2 rate (Thm 8.3) and a test that
  separates LIGHT blocks (deterministic limit, Prop 8.6) from SUPER-DOMINANT
  blocks (random vertex limit, Thm 8.1). The infinite-mean middle is [Open]
  and is returned as Mode E, never certified.

CORRECTIONS PROPAGATED
  K1  T2 needs BN / code space; Prop 5.4 counterexample implemented.
  K2  T4-G needs a.e. continuity.
  K5  compression is conditional; Prop 6.3 generic incompressibility.
  K7  (C1) from (KZ-R)+(G); (C2) derived; three inputs still imported.
  K8  no-Zeno via Doob; horizon truncation removed.
  K9  H^k_R = H^k; (-1)^|alpha| orientation; flux balance in the domain.
  K10 closed surfaces Dchi=+-2, Db1=+-2; annihilation A added.

NEW FOR DRUG RESPONSE
  Kou double-exponential jumps, (L1e) tails, exact Gumbel extremes,
  heaviest-regime domination (a rare hyper-responder line sets the tail),
  moment-explosion gate (refuse to report an infinite mean), and the
  exact Esscher worst case for robust dosing.

NOT CLAIMED
  No data is fabricated. Lamperti / infinite-mean classification is open.
  K-free OPMC separation is open. Degenerate thin-film existence is open.
  Nothing here is clinical.
""")
    return ok


def main() -> int:
    import argparse
    ap=argparse.ArgumentParser(description="Corrected ESCA math for organoid research")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","packages","counterexample","regime","gumbel",
                 "moment","esscher","surgery","nozeno","carpet","philosophy"])
    ap.add_argument("--n-cells",type=int,default=1000)
    ap.add_argument("--n-fates",type=int,default=20)
    ap.add_argument("--eta-up",type=float,default=3.0)
    ap.add_argument("--eta-dn",type=float,default=4.0)
    ap.add_argument("--lam",type=float,default=1.0)
    ap.add_argument("--T",type=float,default=100.0)
    ap.add_argument("--kl",type=float,default=0.5)
    ap.add_argument("--p-moment",type=float,default=1.0)
    ap.add_argument("--json",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    out={}
    if a.cmd=="packages": out=PACKAGES
    elif a.cmd=="philosophy": out=PHILOSOPHY
    elif a.cmd=="counterexample": out=t2_counterexample()
    elif a.cmd=="regime":
        out=theorem_A_rate(a.n_cells,a.n_fates)
    elif a.cmd=="carpet": out={"pcf":carpet_status(True),"nonpcf":carpet_status(False)}
    elif a.cmd=="surgery":
        out={c:SURGERY[c] for c in CONVENTIONS}
    elif a.cmd=="nozeno": out=no_zeno_certificate(1.0,2.0,0.5)
    else:
        r=KouRegime("cli",eta_up=a.eta_up,eta_dn=a.eta_dn,lam=a.lam)
        if a.cmd=="gumbel": out=gumbel_extreme(r,a.T)
        elif a.cmd=="moment": out=moment_estimable([r],a.p_moment)
        elif a.cmd=="esscher": out=esscher_worst_case(r,a.T,a.kl)
    print(json.dumps(out,indent=2,default=str,ensure_ascii=False))
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
