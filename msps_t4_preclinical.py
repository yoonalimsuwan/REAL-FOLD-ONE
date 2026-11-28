#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_t4_preclinical.py — v1.0.0
T4 PRECLINICAL READINESS for iPSC / organoid-derived cell therapy.

=============================================================================
WHAT THIS MODULE IS AND IS NOT
=============================================================================
IS:
  * Tumorigenicity risk arithmetic: required residual-pluripotency limit
    and the assay sample size needed to demonstrate it.
  * Genomic stability QC: variant-allele-fraction detection limits.
  * Engraftment STUDY DESIGN: power analysis, not outcome prediction.
  * GMP release-criteria checklist with Mode E for unmeasured items.
  * HLA coverage estimation for allogeneic banks (rough, needs review).
  * Completed-study analysis: Kaplan-Meier, log-rank, Cox with CI.
  * Explicit documentation of the preclinical -> clinical translation gap.

IS NOT — AND REFUSES:
  [R1] An individual engraftment probability prediction.
  [R2] A transplant go / no-go decision.
  [R3] A clinical dose recommendation.
  [R4] Any claim of clinical readiness or regulatory approval.
  [R5] Ungating the MDOE T4 head. It stays gated.

WHY THE ENGRAFTMENT PREDICTOR STAYS GATED
  Published mouse engraftment studies give n in the tens-to-hundreds.
  A predictor fit on that overfits. Immunodeficient mice lack the immune
  rejection and HLA barriers that dominate human outcomes. No regulator
  accepts a model as evidence of engraftment. An overfit model used for
  a transplant decision can kill a patient.

  The real preparation for transplantation is not a predictor. It is
  tumorigenicity control, genomic stability, potency assays, and a
  properly powered study. That is what this module computes.

=============================================================================
THE CENTRAL SAFETY ARITHMETIC
=============================================================================
Residual undifferentiated hPSC in the graft can form teratoma. If the
residual fraction is p and the dose is N cells, the expected number of
transplanted pluripotent cells is p*N. Requiring that to stay below a
critical inoculum n_crit gives

    p_max = n_crit / N_dose

To DEMONSTRATE p < p_max by sampling n cells and observing zero positives,
the one-sided 95% upper bound (rule of three) is p_upper ~ 3/n, so

    n_assay >= 3 / p_max = 3 * N_dose / n_crit

Worked example: n_crit = 100, N_dose = 1e7
    p_max   = 1e-5  (0.001%)
    n_assay = 3e5   cells that must be interrogated

n_crit is LINE- AND SITE-SPECIFIC and must come from your own spiking
study. The default here is a conservative placeholder, clearly flagged.

PHILOSOPHY (mathematics stands independently)
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

  "Rather death than sin." The gate that protects a patient is not an
  obstacle to progress; it is the progress.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from msps_bio_math_core import (
    HAS_NUMPY, np, SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    get_logger, _np, phi_std, kou_tail if False else fit_kou_tail,
)
log = get_logger("t4_preclinical")
VERSION = "1.0.0"
PRECLINICAL_ONLY = "preclinical_research_only_not_a_transplant_decision"

class ClinicalRefusal(Refusal): pass
class ReleaseFailure(Refusal): pass
class UnderpoweredStudy(Refusal): pass

BANNER = f"""
=============================================================================
MSPS T4 PRECLINICAL READINESS v{VERSION}
tumorigenicity arithmetic · genomic stability · study POWER (not prediction)
GMP release checklist · HLA coverage · completed-study survival analysis
REFUSED: individual engraftment prediction, transplant go/no-go,
         clinical dose, clinical-readiness claims
=============================================================================
"""

# =============================================================================
# §0. THE HARD REFUSALS
# =============================================================================

def refuse_clinical_prediction(request:str) -> None:
    low = request.lower()
    blocked = {
      "predict engraftment":"n is too small; mouse != human; overfit is lethal",
      "transplant decision":"a model is not evidence of engraftment",
      "clinical dose":"requires actual PK/biodistribution and a regulator",
      "go no-go":"requires a named clinician and a regulator, not a model",
      "ungate t4":"the gate protects a patient",
      "พร้อมปลูกถ่าย":"ต้องผ่าน QC, power analysis และ regulator ก่อน",
      "ตัดสินใจปลูกถ่าย":"โมเดลไม่ใช่หลักฐานการ engraft",
      "ทำนาย engraftment":"n น้อยเกิน และ mouse ไม่ใช่ human",
    }
    for k,why in blocked.items():
        if k in low:
            raise ClinicalRefusal(
                f"REFUSED: '{k}' — {why}.\n"
                "This module computes PRECLINICAL READINESS: tumorigenicity "
                "limits, assay sample size, genomic QC, study power, and "
                "GMP release criteria. It does not predict a patient's "
                "engraftment and does not authorise a transplant.")

# =============================================================================
# §1. TUMORIGENICITY — the real gate
# =============================================================================

@dataclass
class TumorigenicitySpec:
    """
    n_crit is the minimum inoculum that formed a teratoma in YOUR OWN
    spiking study for THIS line and THIS implantation site. The default
    is a conservative placeholder and is flagged as unverified.
    """
    n_crit_cells: float = 100.0
    n_crit_verified: bool = False
    n_crit_source: str = ""
    dose_cells: float = 1e7
    assay_lod_fraction: Optional[float] = None   # measured LOD of your assay
    assay_method: str = ""                       # ddPCR LIN28A | flow TRA-1-60 | ...
    cells_interrogated: Optional[float] = None
    positives_observed: Optional[int] = None
    safety_factor: float = 10.0                  # extra margin on n_crit

    def required_limit(self) -> Dict[str,Any]:
        n_eff = self.n_crit_cells/max(self.safety_factor,1.0)
        p_max = n_eff/max(self.dose_cells,1.0)
        n_assay = 3.0/max(p_max,1e-30)
        return {"n_crit_cells":self.n_crit_cells,
                "safety_factor":self.safety_factor,
                "effective_n_crit":n_eff,
                "dose_cells":self.dose_cells,
                "p_max_fraction":p_max,"p_max_percent":100*p_max,
                "n_assay_required":n_assay,
                "formula":"p_max = (n_crit/SF)/N_dose ; n_assay >= 3/p_max",
                "n_crit_verified":self.n_crit_verified,
                "STATUS":("line-specific limit" if self.n_crit_verified else
                          "PLACEHOLDER n_crit — run your own spiking study"),
                "REQUIRED_STUDY":("spike graded numbers of undifferentiated "
                                  "hPSC into the differentiated product, "
                                  "implant into immunodeficient mice, and "
                                  "find the minimum teratoma-forming inoculum")}

    def assay_adequacy(self) -> Dict[str,Any]:
        req = self.required_limit()
        p_max = req["p_max_fraction"]
        out = {"required_p_max":p_max,"assay_method":self.assay_method,
               "assay_lod":self.assay_lod_fraction}
        if self.assay_lod_fraction is None:
            out["adequate"]=None; out["status"]=MODE_E
            out["reason"]="assay LOD not measured; cannot judge adequacy"
            return out
        lod_ok = self.assay_lod_fraction <= p_max
        out["lod_sufficient"]=lod_ok
        out["lod_margin_factor"]=p_max/max(self.assay_lod_fraction,1e-30)
        if self.cells_interrogated and self.positives_observed is not None:
            n = float(self.cells_interrogated); k = int(self.positives_observed)
            if k == 0:
                p_upper = 3.0/max(n,1.0)          # rule of three, ~95% one-sided
                method = "rule of three (0 positives)"
            else:
                # Clopper-Pearson upper bound via beta quantile approximation
                p_upper = _clopper_upper(k, n)
                method = "Clopper-Pearson upper bound"
            out.update({"cells_interrogated":n,"positives":k,
                        "p_upper_95":p_upper,"bound_method":method,
                        "demonstrated_below_limit":p_upper<=p_max,
                        "n_assay_required":req["n_assay_required"],
                        "n_assay_shortfall":max(0.0,req["n_assay_required"]-n)})
            out["adequate"] = bool(lod_ok and p_upper<=p_max)
        else:
            out["adequate"] = None
            out["reason"] = "cells_interrogated / positives not supplied"
        out["status"] = (MODE_CERTIFIED if out["adequate"] else MODE_E)
        if not lod_ok:
            out["BLOCKER"] = (f"assay LOD {self.assay_lod_fraction:.2e} is "
                              f"coarser than the required limit {p_max:.2e}. "
                              "A more sensitive assay is required (ddPCR for "
                              "LIN28A, or high-sensitivity flow).")
        return out

def _clopper_upper(k:int, n:float, alpha:float=0.05) -> float:
    """One-sided upper bound; bisection on the binomial tail."""
    if n<=0: return 1.0
    lo,hi = k/n, 1.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        tail = sum(math.comb(int(n),i)*mid**i*(1-mid)**(int(n)-i)
                   for i in range(0,k+1)) if n<=2000 else _norm_tail(k,n,mid)
        if tail > alpha: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def _norm_tail(k,n,p):
    mu=n*p; sd=math.sqrt(max(n*p*(1-p),1e-12))
    return phi_std((k+0.5-mu)/sd)

# =============================================================================
# §2. GENOMIC STABILITY
# =============================================================================

CULTURE_ADAPTED_LOCI = {
 "TP53":       {"risk":"dominant-negative, apoptosis escape","evidence":"strong"},
 "BCOR":       {"risk":"growth advantage in hPSC culture","evidence":"moderate"},
 "20q11.21":   {"risk":"BCL2L1 amplification, survival advantage","evidence":"strong"},
 "12p":        {"risk":"NANOG/GDF3 region gain","evidence":"moderate"},
 "17q":        {"risk":"growth advantage","evidence":"moderate"},
 "1q":         {"risk":"MDM4 region gain","evidence":"emerging"},
}

@dataclass
class GenomicQC:
    sequencing_depth: Optional[int] = None
    target_vaf: float = 0.01           # smallest VAF you must detect
    karyotype_done: bool = False
    cnv_array_done: bool = False
    snv_panel_done: bool = False
    passage_number: Optional[int] = None
    max_passage_validated: Optional[int] = None
    detected_variants: List[Dict] = field(default_factory=list)

    def depth_requirement(self) -> Dict[str,Any]:
        """To see VAF f you need ~3/f reads for >=95% detection (binomial)."""
        need = 3.0/max(self.target_vaf,1e-9)
        return {"target_vaf":self.target_vaf,
                "depth_required":need,
                "depth_supplied":self.sequencing_depth,
                "sufficient":(self.sequencing_depth is not None
                              and self.sequencing_depth>=need),
                "formula":"P(0 reads | VAF f, depth D) = (1-f)^D <= 0.05 "
                          "=> D >= 3/f",
                "NOTE":"heterozygous VAF f implies ~2f of cells carry the variant"}

    def assess(self) -> Dict[str,Any]:
        dr = self.depth_requirement()
        flags=[]
        for v in self.detected_variants:
            g = str(v.get("gene","")).upper()
            for locus,info in CULTURE_ADAPTED_LOCI.items():
                if locus.upper() in g or g in locus.upper():
                    flags.append({"variant":v,"locus":locus,**info})
        passage_ok = (self.passage_number is not None
                      and self.max_passage_validated is not None
                      and self.passage_number<=self.max_passage_validated)
        mods = [self.karyotype_done,self.cnv_array_done,self.snv_panel_done]
        out={"depth":dr,"culture_adapted_flags":flags,
             "n_flags":len(flags),"modalities_done":sum(mods),
             "passage_within_validated":passage_ok,
             "passage":self.passage_number,
             "max_validated":self.max_passage_validated}
        ok = (dr["sufficient"] and not flags and all(mods) and passage_ok)
        out["passes"]=ok
        out["status"]=MODE_CERTIFIED if ok else MODE_E
        if flags:
            out["BLOCKER"]=("culture-adapted driver variant(s) detected: "
                            f"{[f['locus'] for f in flags]}. A subclone with "
                            "a growth advantage can expand during culture. "
                            "Re-derive from an earlier passage or a new clone.")
        return out

# =============================================================================
# §3. STUDY POWER — design, NOT prediction
# =============================================================================

def z_quantile(p:float) -> float:
    """Inverse normal CDF by bisection (no scipy dependency)."""
    lo,hi=-10.0,10.0
    for _ in range(200):
        mid=0.5*(lo+hi)
        if phi_std(mid)<p: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)

def binary_engraftment_power(p1:float,p2:float,alpha:float=0.05,
                             power:float=0.8) -> Dict[str,Any]:
    """
    Two-arm binary outcome (engrafted / not). This is STUDY DESIGN.
    It answers 'how many animals do I need', never 'will this patient engraft'.
    """
    if not (0<p1<1 and 0<p2<1): raise Refusal("p in (0,1)")
    za=z_quantile(1-alpha/2); zb=z_quantile(power)
    d=abs(p1-p2)
    n=((za+zb)**2)*(p1*(1-p1)+p2*(1-p2))/(d*d)
    return {"p1":p1,"p2":p2,"delta":d,"alpha":alpha,"power":power,
            "n_per_arm":math.ceil(n),"n_total":2*math.ceil(n),
            "formula":"n = (z_a+z_b)^2 [p1(1-p1)+p2(1-p2)]/(p1-p2)^2",
            "PURPOSE":"study design, NOT outcome prediction",
            "NOTE":"add attrition allowance; cluster by litter/batch"}

def detectable_effect(n_per_arm:int,alpha:float=0.05,power:float=0.8,
                      p_control:float=0.3) -> Dict[str,Any]:
    """Invert: with n animals, what is the SMALLEST effect you can detect?"""
    za=z_quantile(1-alpha/2); zb=z_quantile(power)
    lo,hi=1e-4,1.0-p_control-1e-6
    for _ in range(200):
        mid=0.5*(lo+hi); p2=p_control+mid
        if p2>=1: hi=mid; continue
        n=((za+zb)**2)*(p_control*(1-p_control)+p2*(1-p2))/(mid*mid)
        if n>n_per_arm: lo=mid
        else: hi=mid
    d=0.5*(lo+hi)
    return {"n_per_arm":n_per_arm,"p_control":p_control,
            "min_detectable_delta":d,"p_treatment_needed":p_control+d,
            "VERDICT":(f"with n={n_per_arm} per arm you can only detect a "
                       f"delta >= {d:.3f}. Smaller true effects will be "
                       "missed. This is why a predictor fit on such n "
                       "cannot be trusted."),
            "PURPOSE":"honest limitation of the existing evidence base"}

def survival_logrank_events(hr:float,alpha:float=0.05,power:float=0.8
                            ) -> Dict[str,Any]:
    """Events needed for a log-rank test at hazard ratio hr."""
    if hr<=0 or abs(math.log(hr))<1e-9: raise Refusal("hr>0 and != 1")
    za=z_quantile(1-alpha/2); zb=z_quantile(power)
    d=4*(za+zb)**2/(math.log(hr)**2)
    return {"hazard_ratio":hr,"alpha":alpha,"power":power,
            "events_required":math.ceil(d),
            "formula":"d = 4(z_a+z_b)^2/ln(HR)^2",
            "NOTE":"events, not subjects; follow-up must be long enough"}

def refuse_underpowered(n_per_arm:int, p1:float, p2:float) -> Dict:
    r=binary_engraftment_power(p1,p2)
    if n_per_arm < r["n_per_arm"]:
        raise UnderpoweredStudy(
            f"n={n_per_arm}/arm is below the {r['n_per_arm']} needed to "
            f"detect {p1}->{p2} at 80% power. Reporting a null result from "
            "this study as 'no effect' is invalid. Report the detectable "
            "effect size instead.")
    return r

# =============================================================================
# §4. HLA COVERAGE (allogeneic bank, rough estimate)
# =============================================================================

def hla_bank_coverage(donor_homozygous_haplotypes:Sequence[Dict[str,float]],
                      loci:Sequence[str]=("A","B","DRB1")) -> Dict[str,Any]:
    """
    A homozygous donor presents one antigen per locus. A recipient is
    'covered' at that locus if they carry that antigen. Under a crude
    independence approximation the covered fraction is the product over
    loci of (1-(1-f_l)^2) for antigen frequency f_l.

    THIS IS A ROUGH ESTIMATE. Real matching requires haplotype frequencies,
    linkage disequilibrium, DPB1/DQB1, and immunology review. It is reported
    as Mode E until an immunologist signs off.
    """
    per_donor=[]
    for h in donor_homozygous_haplotypes:
        cov=1.0; per_locus={}
        for l in loci:
            f=float(h.get(l,0.0))
            c=1.0-(1.0-f)**2
            per_locus[l]={"antigen_freq":f,"covered_fraction":c}
            cov*=c
        per_donor.append({"haplotype":h,"per_locus":per_locus,
                          "covered_fraction":cov})
    union = 1.0
    for d in per_donor: union *= (1.0-d["covered_fraction"])
    return {"n_donors":len(per_donor),"per_donor":per_donor,
            "union_coverage_independent_approx":1.0-union,
            "loci":list(loci),
            "status":MODE_E,
            "CAVEAT":("independence approximation; ignores linkage "
                      "disequilibrium, DPB1/DQB1, and minor antigens. "
                      "Requires immunologist review. Not a matching report."),
            "NOT":"a clinical HLA match determination"}

# =============================================================================
# §5. GMP RELEASE CRITERIA
# =============================================================================

@dataclass
class ReleaseCriteria:
    identity_markers_verified: bool = False
    purity_target_fraction: Optional[float] = None
    purity_min_required: float = 0.80
    potency_assay_validated: bool = False
    potency_result: Optional[float] = None
    potency_min: Optional[float] = None
    sterility_passed: Optional[bool] = None
    mycoplasma_passed: Optional[bool] = None
    endotoxin_EU_per_mL: Optional[float] = None
    endotoxin_limit: float = 5.0
    tumorigenicity: Optional[Dict] = None
    genomic_qc: Optional[Dict] = None
    viability_fraction: Optional[float] = None
    viability_min: float = 0.70
    residual_reprogramming_transgene: Optional[bool] = None  # False = cleared
    responsible_person: str = ""

    def evaluate(self) -> Dict[str,Any]:
        rows=[]
        def add(name, ok, detail=""):
            rows.append({"criterion":name,
                         "status":("PASS" if ok is True else
                                   "FAIL" if ok is False else "NOT_MEASURED"),
                         "detail":detail})
        add("identity", self.identity_markers_verified or None)
        add("purity", None if self.purity_target_fraction is None else
            self.purity_target_fraction>=self.purity_min_required,
            f"{self.purity_target_fraction} vs >= {self.purity_min_required}")
        add("potency", None if not self.potency_assay_validated else
            (None if self.potency_result is None or self.potency_min is None
             else self.potency_result>=self.potency_min))
        add("sterility", self.sterility_passed)
        add("mycoplasma", self.mycoplasma_passed)
        add("endotoxin", None if self.endotoxin_EU_per_mL is None else
            self.endotoxin_EU_per_mL<=self.endotoxin_limit)
        add("viability", None if self.viability_fraction is None else
            self.viability_fraction>=self.viability_min)
        add("transgene_cleared",
            None if self.residual_reprogramming_transgene is None else
            not self.residual_reprogramming_transgene)
        if self.tumorigenicity:
            add("tumorigenicity", self.tumorigenicity.get("adequate"))
        if self.genomic_qc:
            add("genomic_stability", self.genomic_qc.get("passes"))
        add("responsible_person_named", bool(self.responsible_person))
        n_fail=sum(1 for r in rows if r["status"]=="FAIL")
        n_nm=sum(1 for r in rows if r["status"]=="NOT_MEASURED")
        return {"criteria":rows,"n_fail":n_fail,"n_not_measured":n_nm,
                "release_candidate":(n_fail==0 and n_nm==0),
                "status":(MODE_CERTIFIED if (n_fail==0 and n_nm==0) else MODE_E),
                "SCOPE":("a research checklist modelled on release-criteria "
                         "structure. Actual release is decided by the "
                         "manufacturer's Quality unit and the regulator."),
                "NOT":"a regulatory release authorisation"}

    def require_release_candidate(self) -> Dict:
        r=self.evaluate()
        if not r["release_candidate"]:
            raise ReleaseFailure(
                f"release checklist not complete: {r['n_fail']} FAIL, "
                f"{r['n_not_measured']} NOT_MEASURED. "
                "No preclinical batch may be described as ready.")
        return r

# =============================================================================
# §6. COMPLETED-STUDY ANALYSIS (not prediction)
# =============================================================================

def kaplan_meier(times:Sequence[float], events:Sequence[int]) -> Dict[str,Any]:
    """Analyse a COMPLETED study. Does not predict a new subject."""
    _np()
    t=np.asarray(times,float); e=np.asarray(events,int)
    order=np.argsort(t); t=t[order]; e=e[order]
    n=len(t); S=1.0; rows=[]; var=0.0
    i=0
    while i<n:
        j=i
        while j<n and t[j]==t[i]: j+=1
        at_risk=n-i; d=int(e[i:j].sum())
        if d>0:
            S*= (1.0-d/at_risk)
            if at_risk>d: var += d/(at_risk*(at_risk-d))
        rows.append({"time":float(t[i]),"at_risk":at_risk,"events":d,"S":S})
        i=j
    se = S*math.sqrt(var) if S>0 else 0.0
    return {"curve":rows,"S_final":S,"greenwood_se":se,
            "ci95":[max(0.0,S-1.96*se),min(1.0,S+1.96*se)],
            "n":n,"total_events":int(e.sum()),
            "SCOPE":"describes a completed cohort; NOT a prediction"}

def logrank_test(times_a,events_a,times_b,events_b) -> Dict[str,Any]:
    _np()
    ta=np.asarray(times_a,float); ea=np.asarray(events_a,int)
    tb=np.asarray(times_b,float); eb=np.asarray(events_b,int)
    all_t=np.unique(np.concatenate([ta[ea==1],tb[eb==1]]))
    O_minus_E=0.0; V=0.0
    for t in all_t:
        n1=int((ta>=t).sum()); n2=int((tb>=t).sum()); n=n1+n2
        d1=int(((ta==t)&(ea==1)).sum()); d2=int(((tb==t)&(eb==1)).sum())
        d=d1+d2
        if n<2 or d==0: continue
        E1=d*n1/n
        O_minus_E += d1-E1
        V += d*n1*n2*(n-d)/(n*n*(n-1))
    chi2 = (O_minus_E**2)/V if V>0 else 0.0
    p = math.erfc(math.sqrt(chi2/2)) if chi2>0 else 1.0
    return {"chi2":chi2,"p_value":p,"O_minus_E":O_minus_E,"variance":V,
            "significant":p<0.05,
            "SCOPE":"compares two completed arms; not a prediction"}

# =============================================================================
# §7. TRANSLATION GAP — explicit
# =============================================================================

TRANSLATION_GAP = {
 "immune_rejection":{
   "mouse":"immunodeficient (NSG/NOG) — no T/B/NK rejection",
   "human":"intact immunity; HLA mismatch drives rejection",
   "consequence":"mouse engraftment OVERESTIMATES human engraftment"},
 "scale":{
   "mouse":"~1e5–1e6 cells, mm-scale tissue",
   "human":"~1e7–1e9 cells, cm-scale; diffusion limits differ",
   "consequence":"vascularisation and hypoxia do not scale linearly"},
 "site":{
   "mouse":"often subcutaneous or renal capsule (permissive)",
   "human":"orthotopic, diseased, fibrotic, inflamed",
   "consequence":"niche signals differ qualitatively"},
 "time":{
   "mouse":"weeks-to-months follow-up",
   "human":"years-to-decades; late teratoma and immune events matter",
   "consequence":"short follow-up cannot exclude late events"},
 "tumorigenicity":{
   "mouse":"spiking studies give an inoculum threshold in a permissive host",
   "human":"threshold unknown; must be conservatively bounded",
   "consequence":"p_max must carry a safety factor, not a point estimate"},
 "regulatory":{
   "mouse":"preclinical evidence only",
   "human":"IND/CTA requires GMP, release criteria, biodistribution, "
           "tumorigenicity, and a named sponsor",
   "consequence":"no model output substitutes for any of these"},
}

def translation_gap_report() -> Dict[str,Any]:
    return {"gaps":TRANSLATION_GAP,
            "CONCLUSION":("mouse engraftment data cannot be extrapolated to "
                          "a human probability. This is why the T4 "
                          "engraftment head stays GATED."),
            "WHAT_TO_DO_INSTEAD":[
              "run a line-specific tumorigenicity spiking study",
              "measure your assay LOD and interrogate >= 3/p_max cells",
              "complete genomic stability QC at the intended passage",
              "validate a potency assay tied to mechanism",
              "power the pivotal study honestly (see binary_engraftment_power)",
              "assemble the GMP release checklist with a named Quality unit",
              "submit to the regulator; a model is not evidence"]}

# =============================================================================
# §8. ORCHESTRATOR
# =============================================================================

class T4Preclinical:
    def __init__(self, out_dir="runs/t4_preclinical"):
        self.out=Path(out_dir); self.out.mkdir(parents=True,exist_ok=True)
        self.sd=SDGate()
    def assess(self, tumor:TumorigenicitySpec, genomic:GenomicQC,
               release:ReleaseCriteria,
               study_n_per_arm:Optional[int]=None,
               p_control:float=0.3,p_treatment:float=0.6) -> Dict[str,Any]:
        refuse_clinical_prediction("preclinical readiness assessment")
        req=tumor.required_limit(); adq=tumor.assay_adequacy()
        gq=genomic.assess()
        release.tumorigenicity=adq; release.genomic_qc=gq
        rel=release.evaluate()
        study={}
        if study_n_per_arm is not None:
            study["detectable"]=detectable_effect(study_n_per_arm,
                                                  p_control=p_control)
            study["required"]=binary_engraftment_power(p_control,p_treatment)
            study["adequately_powered"]=(study_n_per_arm>=
                                         study["required"]["n_per_arm"])
        blockers=[]
        if not req["n_crit_verified"]:
            blockers.append("n_crit is a PLACEHOLDER — run a spiking study")
        if adq.get("adequate") is not True:
            blockers.append("tumorigenicity limit not demonstrated")
        if not gq["passes"]:
            blockers.append("genomic stability QC incomplete or flagged")
        if not rel["release_candidate"]:
            blockers.append(f"release checklist: {rel['n_fail']} FAIL, "
                            f"{rel['n_not_measured']} NOT_MEASURED")
        if study and not study["adequately_powered"]:
            blockers.append("pivotal study underpowered")
        return {"tumorigenicity":{"required":req,"assay":adq},
                "genomic":gq,"release":rel,"study_design":study,
                "blockers":blockers,"n_blockers":len(blockers),
                "preclinical_ready":not blockers,
                "translation_gap":translation_gap_report(),
                "scope":PRECLINICAL_ONLY,
                "REFUSED":["individual engraftment prediction",
                           "transplant go/no-go","clinical dose",
                           "clinical-readiness claim"],
                "NEXT":"a named clinician, a Quality unit, and a regulator — "
                       "not a model"}
    def report(self):
        rep={"version":VERSION,"scope":PRECLINICAL_ONLY,
             "culture_adapted_loci":CULTURE_ADAPTED_LOCI,
             "translation_gap":TRANSLATION_GAP,
             "sd":self.sd.report()}
        (self.out/"report.json").write_text(json.dumps(rep,indent=2,default=str))
        return rep

# =============================================================================
# §9. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print(BANNER); ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] CLINICAL PREDICTION REFUSED")
    b=0
    for r in ("predict engraftment","transplant decision","clinical dose",
              "go no-go","ungate t4","ตัดสินใจปลูกถ่าย","ทำนาย engraftment"):
        try: refuse_clinical_prediction(r)
        except ClinicalRefusal: b+=1
    rep("all clinical requests refused",b==7,f"{b}/7")

    print("\n[T2] TUMORIGENICITY ARITHMETIC (the real gate)")
    t=TumorigenicitySpec(n_crit_cells=100.,dose_cells=1e7,safety_factor=1.0)
    r=t.required_limit()
    print(f"      n_crit=100, dose=1e7 -> p_max={r['p_max_fraction']:.2e} "
          f"({r['p_max_percent']:.4f}%)  n_assay={r['n_assay_required']:.0f}")
    rep("p_max = 1e-5",abs(r["p_max_fraction"]-1e-5)<1e-12)
    rep("n_assay = 3e5",abs(r["n_assay_required"]-3e5)<1)
    rep("placeholder flagged",not r["n_crit_verified"])
    t10=TumorigenicitySpec(n_crit_cells=100.,dose_cells=1e7,safety_factor=10.)
    r10=t10.required_limit()
    print(f"      SF=10 -> p_max={r10['p_max_fraction']:.2e} "
          f"n_assay={r10['n_assay_required']:.0f}")
    rep("safety factor tightens limit",r10["p_max_fraction"]<r["p_max_fraction"])
    rep("dose scaling: 10x dose -> 10x n_assay",
        abs(TumorigenicitySpec(dose_cells=1e8,safety_factor=1.)
            .required_limit()["n_assay_required"]/r["n_assay_required"]-10)<1e-6)

    print("\n[T3] ASSAY ADEQUACY (LOD + rule of three)")
    t.assay_method="ddPCR LIN28A"; t.assay_lod_fraction=1e-6
    t.cells_interrogated=1e5; t.positives_observed=0
    a=t.assay_adequacy()
    print(f"      LOD={t.assay_lod_fraction:.0e} n=1e5 k=0 -> "
          f"p_upper={a['p_upper_95']:.2e} adequate={a['adequate']}")
    rep("rule of three used","rule of three" in a["bound_method"])
    rep("1e5 cells insufficient for 1e-5",not a["adequate"])
    rep("shortfall reported",a["n_assay_shortfall"]>0)
    t.cells_interrogated=5e5
    a2=t.assay_adequacy()
    print(f"      n=5e5 -> p_upper={a2['p_upper_95']:.2e} adequate={a2['adequate']}")
    rep("5e5 cells sufficient",a2["adequate"])
    t.assay_lod_fraction=1e-3
    a3=t.assay_adequacy()
    rep("coarse LOD BLOCKS",not a3["lod_sufficient"])
    rep("blocker names ddPCR","ddPCR" in a3.get("BLOCKER",""))

    print("\n[T4] GENOMIC STABILITY")
    g=GenomicQC(sequencing_depth=100,target_vaf=0.01,karyotype_done=True,
                cnv_array_done=True,snv_panel_done=True,
                passage_number=20,max_passage_validated=30)
    d=g.depth_requirement()
    print(f"      VAF 1% needs depth {d['depth_required']:.0f}, have 100")
    rep("depth 300 required for 1% VAF",abs(d["depth_required"]-300)<1)
    rep("depth 100 insufficient",not d["sufficient"])
    g.sequencing_depth=600
    rep("depth 600 sufficient",g.depth_requirement()["sufficient"])
    ga=g.assess(); rep("clean genome passes",ga["passes"])
    g.detected_variants=[{"gene":"TP53","vaf":0.08}]
    ga2=g.assess()
    rep("TP53 variant FLAGGED",ga2["n_flags"]==1)
    rep("TP53 blocks release",not ga2["passes"])
    rep("blocker explains subclone expansion","subclone" in ga2["BLOCKER"])
    g.detected_variants=[]; g.passage_number=50
    rep("over-passage fails",not g.assess()["passes"])

    print("\n[T5] STUDY POWER — design, not prediction")
    pw=binary_engraftment_power(0.3,0.6)
    print(f"      0.3->0.6 needs n={pw['n_per_arm']}/arm")
    rep("n computed",pw["n_per_arm"]>0)
    rep("purpose is design","NOT outcome prediction" in pw["PURPOSE"])
    de=detectable_effect(20,p_control=0.3)
    print(f"      n=20/arm -> min detectable delta={de['min_detectable_delta']:.3f}")
    rep("small n only detects large effects",de["min_detectable_delta"]>0.3)
    rep("verdict explains gating","cannot be trusted" in de["VERDICT"])
    b=False
    try: refuse_underpowered(20,0.3,0.6)
    except UnderpoweredStudy: b=True
    rep("underpowered study REFUSED",b)
    ev=survival_logrank_events(0.6)
    print(f"      HR=0.6 needs {ev['events_required']} events")
    rep("events not subjects","events, not subjects" in ev["NOTE"])

    print("\n[T6] HLA COVERAGE (rough, Mode E)")
    h=hla_bank_coverage([{"A":0.15,"B":0.10,"DRB1":0.12},
                         {"A":0.10,"B":0.20,"DRB1":0.08}])
    print(f"      2 homozygous donors -> union coverage "
          f"{h['union_coverage_independent_approx']:.3f}")
    rep("coverage in (0,1)",0<h["union_coverage_independent_approx"]<1)
    rep("always Mode E",h["status"]==MODE_E)
    rep("LD caveat present","linkage" in h["CAVEAT"])
    rep("not a match report","NOT" in h)

    print("\n[T7] GMP RELEASE CHECKLIST")
    rc=ReleaseCriteria()
    r0=rc.evaluate()
    rep("empty checklist NOT ready",not r0["release_candidate"])
    rep("unmeasured counted",r0["n_not_measured"]>0)
    b=False
    try: rc.require_release_candidate()
    except ReleaseFailure: b=True
    rep("incomplete release REFUSED",b)
    full=ReleaseCriteria(identity_markers_verified=True,
        purity_target_fraction=0.92,potency_assay_validated=True,
        potency_result=1.2,potency_min=1.0,sterility_passed=True,
        mycoplasma_passed=True,endotoxin_EU_per_mL=0.5,
        viability_fraction=0.88,residual_reprogramming_transgene=False,
        responsible_person="QA-01")
    t2=TumorigenicitySpec(n_crit_cells=100.,dose_cells=1e7,safety_factor=1.,
        n_crit_verified=True,n_crit_source="own spiking study 2026",
        assay_method="ddPCR LIN28A",assay_lod_fraction=1e-6,
        cells_interrogated=5e5,positives_observed=0)
    g2=GenomicQC(sequencing_depth=600,karyotype_done=True,cnv_array_done=True,
                 snv_panel_done=True,passage_number=20,max_passage_validated=30)
    full.tumorigenicity=t2.assay_adequacy(); full.genomic_qc=g2.assess()
    rf=full.evaluate()
    rep("complete checklist is a release CANDIDATE",rf["release_candidate"])
    rep("still not a regulatory authorisation","NOT" in rf)

    print("\n[T8] COMPLETED-STUDY ANALYSIS")
    ta=[1,2,3,4,5,6,7,8]; ea=[1,1,0,1,0,1,0,0]
    km=kaplan_meier(ta,ea)
    print(f"      S_final={km['S_final']:.3f} CI={km['ci95']}")
    rep("KM monotone decreasing",
        all(km["curve"][i]["S"]>=km["curve"][i+1]["S"]
            for i in range(len(km["curve"])-1)))
    rep("not a prediction","NOT a prediction" in km["SCOPE"])
    lr=logrank_test(ta,ea,[2,3,4,5,6,7,8,9],[1,1,1,1,0,1,0,1])
    rep("logrank runs","p_value" in lr)
    rep("compares completed arms","not a prediction" in lr["SCOPE"])

    print("\n[T9] FULL ASSESSMENT + TRANSLATION GAP")
    pipe=T4Preclinical()
    out=pipe.assess(t2,g2,full,study_n_per_arm=20)
    print(f"      blockers={out['n_blockers']} ready={out['preclinical_ready']}")
    rep("underpowered study blocks readiness",not out["preclinical_ready"])
    rep("study design present","detectable" in out["study_design"])
    out2=pipe.assess(t2,g2,full,study_n_per_arm=200)
    rep("adequate power removes that blocker",
        out2["study_design"]["adequately_powered"])
    rep("scope preclinical only",out["scope"]==PRECLINICAL_ONLY)
    rep("refusals listed",len(out["REFUSED"])==4)
    gap=out["translation_gap"]["gaps"]
    rep("immune rejection gap documented","immune_rejection" in gap)
    rep("mouse overestimates stated",
        "OVERESTIMATES" in gap["immune_rejection"]["consequence"])
    rep("6 gaps documented",len(gap)==6)
    rr=pipe.report()
    rep("report written",(pipe.out/"report.json").exists())
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
BUILT — the T4 that genuinely prepares for transplantation
  Tumorigenicity: p_max = (n_crit/SF)/N_dose and n_assay >= 3/p_max.
  Assay adequacy: measured LOD plus a rule-of-three / Clopper-Pearson bound.
  Genomic stability: depth >= 3/VAF, culture-adapted driver flags, passage cap.
  Study POWER: n per arm, minimum detectable effect, log-rank event count.
  HLA coverage: rough bank estimate, always Mode E pending immunology review.
  GMP release checklist: identity, purity, potency, safety, sterility.
  Completed-study survival analysis with Greenwood CI.
  Explicit six-part translation-gap report.

REFUSED — and this is the point
  Individual engraftment prediction. Transplant go/no-go. Clinical dose.
  Clinical-readiness claims. Ungating the MDOE T4 head.

WHY
  Mouse n is in the tens-to-hundreds; immunodeficient hosts lack rejection
  and HLA barriers. An overfit engraftment model used for a transplant
  decision can kill a patient. The real preparation is QC, a demonstrated
  tumorigenicity limit, and an honestly powered study — then a named
  clinician, a Quality unit, and a regulator.
""")
    return ok

def main() -> int:
    ap=argparse.ArgumentParser(description="T4 preclinical readiness")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","tumorigenicity","power","detectable","gap","report"])
    ap.add_argument("--n-crit",type=float,default=100.)
    ap.add_argument("--dose",type=float,default=1e7)
    ap.add_argument("--sf",type=float,default=10.)
    ap.add_argument("--p1",type=float,default=0.3)
    ap.add_argument("--p2",type=float,default=0.6)
    ap.add_argument("--n",type=int,default=20)
    ap.add_argument("--json",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    out={}
    if a.cmd=="tumorigenicity":
        out=TumorigenicitySpec(n_crit_cells=a.n_crit,dose_cells=a.dose,
                               safety_factor=a.sf).required_limit()
    elif a.cmd=="power": out=binary_engraftment_power(a.p1,a.p2)
    elif a.cmd=="detectable": out=detectable_effect(a.n,p_control=a.p1)
    elif a.cmd=="gap": out=translation_gap_report()
    elif a.cmd=="report": out=T4Preclinical().report()
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
