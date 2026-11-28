#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ORGANOID GEO LOADER v2.0 — screening + moment-estimability gate
=============================================================================
NEW IN v2
  * SD gate on the screening verdict.
  * MOMENT-ESTIMABILITY GATE (Prop 16.14): before reporting a mean IC50 /
    AUC across lines, fit a double-exponential tail and check whether the
    first moment is finite. If eta+ <= 1 the MEAN IS INFINITE and the
    module REFUSES to report it, offering a quantile instead.
  * Heaviest-regime diagnostic (Thm 16.10): which line governs the tail.
  * Still refuses to ship a pre-certified accession list.

PHILOSOPHY: see msps_esca_bio_core.PHILOSOPHY (full text).
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

HONEST SCOPE
  No accession is pre-certified. A CANDIDATE verdict means metadata is
  CONSISTENT with organoid drug response, not that usable data exists.
  Organoid response != patient response. Nothing here is clinical.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, gzip, json, re, urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np

from msps_esca_bio_core import (
    SDGate, Refusal, InfiniteMoment, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    KouRegime, moment_estimable, require_finite_mean,
    heaviest_regime_dominates, gumbel_extreme, get_logger,
)
log=get_logger("organoid_geo")
VERSION="2.0.0"

GEO_FTP="https://ftp.ncbi.nlm.nih.gov/geo/series"
ORGANOID_TERMS=["organoid","spheroid","3d culture","tumoroid",
                "patient-derived organoid","pdo"]
DRUG_TERMS=["drug","compound","treatment","treated","ic50","ec50","dose",
            "viability","sensitivity","response","inhibitor","chemotherap","screen"]
RESPONSE_TERMS=["ic50","ec50","auc","viability","response","sensitivity",
                "gi50","% inhibition","survival"]

def series_matrix_url(gse:str) -> str:
    gse=gse.upper().strip()
    if not re.fullmatch(r"GSE\d+",gse): raise Refusal(f"bad accession {gse}")
    stub=gse[:-3]+"nnn" if len(gse)>4 else "GSEnnn"
    return f"{GEO_FTP}/{stub}/{gse}/matrix/{gse}_series_matrix.txt.gz"

def fetch_series_matrix(gse:str,dest="data/geo") -> Optional[Path]:
    d=Path(dest); d.mkdir(parents=True,exist_ok=True)
    out=d/f"{gse.upper()}_series_matrix.txt.gz"
    if out.exists(): return out
    url=series_matrix_url(gse); log.info(f"[GEO] {url}")
    try:
        with urllib.request.urlopen(url,timeout=120) as r:
            out.write_bytes(r.read())
        return out
    except Exception as e:
        log.warning(f"[GEO] fetch failed: {e}")
        log.info(f"[GEO] browse https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}")
        out.unlink(missing_ok=True); return None

def parse_series_matrix(path:Path) -> Dict:
    meta:Dict[str,List[str]]={}; rows=[]; ids=None; in_tab=False
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",errors="ignore") as f:
        for line in f:
            line=line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"): in_tab=True; continue
            if line.startswith("!series_matrix_table_end"): in_tab=False; continue
            if in_tab:
                p=line.split("\t")
                if ids is None: ids=[x.strip('"') for x in p[1:]]
                else: rows.append((p[0].strip('"'),
                    [float(x) if x not in ("","NA","null") else np.nan
                     for x in p[1:]]))
                continue
            if line.startswith("!"):
                k=line.split("\t")[0][1:]
                meta.setdefault(k,[]).extend([x.strip('"') for x in line.split("\t")[1:]])
    X=np.array([r[1] for r in rows],np.float32) if rows else None
    return {"meta":meta,"sample_ids":ids or [],"expression":X,
            "probes":[r[0] for r in rows],
            "n_samples":len(ids or []),"n_features":len(rows)}

def screen(parsed:Dict,gse:str,sd:Optional[SDGate]=None) -> Dict:
    sd=sd or SDGate()
    m=parsed["meta"]
    blob=" ".join(" ".join(v) for v in m.values()).lower()
    chars=" ".join(m.get("Sample_characteristics_ch1",[])).lower()
    design=" ".join(m.get("Series_overall_design",[])).lower()
    summ=" ".join(m.get("Series_summary",[])).lower()
    org=[t for t in ORGANOID_TERMS if t in blob]
    drug=[t for t in DRUG_TERMS if t in (chars+" "+design+" "+summ)]
    resp=[t for t in RESPONSE_TERMS if t in chars]
    has_x=parsed["expression"] is not None and parsed["n_samples"]>0
    verdict="CANDIDATE" if (org and drug and has_x) else "REJECT"
    reasons=[]
    if not org: reasons.append("no organoid term")
    if not drug: reasons.append("no drug/treatment term in sample metadata")
    if not has_x: reasons.append("no expression matrix")
    if not resp: reasons.append("no numeric response term in characteristics "
                                "(may be in supplementary files)")
    cert=sd.certify("composition_record",(),{"accession":gse,"verdict":verdict})
    return {"accession":gse,"verdict":verdict,"reasons":reasons,
            "organoid_terms":org,"drug_terms":drug,"response_terms":resp,
            "n_samples":parsed["n_samples"],"n_features":parsed["n_features"],
            "sd_status":cert["status"],
            "NEXT_STEP":(f"Open https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
                         f"acc={gse} and confirm (1) organoid model "
                         "(2) drug panel (3) numeric response per sample. "
                         "Check supplementary files."),
            "NOT_CERTIFIED":"a CANDIDATE verdict is not a data guarantee"}

# -----------------------------------------------------------------------------
# NEW: response-tail analysis with a moment-estimability gate
# -----------------------------------------------------------------------------

def fit_kou_tail(log_responses:Sequence[float],name:str="cohort"
                 ) -> KouRegime:
    """
    Fit a double-exponential tail to log-response increments. Positive
    increments -> upward rate eta+, negative -> downward rate eta-.
    MLE for an exponential tail: eta = 1/mean(excess).
    """
    x=np.asarray(log_responses,float)
    if len(x)<8: raise Refusal("need >= 8 responses to fit a tail")
    d=np.diff(x)
    up=d[d>0]; dn=-d[d<0]
    eta_up=1.0/max(float(up.mean()),1e-9) if len(up)>1 else 10.0
    eta_dn=1.0/max(float(dn.mean()),1e-9) if len(dn)>1 else 10.0
    p=len(up)/max(len(d),1)
    lam=float(len(d))/max(len(x),1)
    return KouRegime(name,p=p,eta_up=eta_up,eta_dn=eta_dn,lam=lam,
                     drift=float(d.mean()),vol=float(d.std()),occupation=1.0)

def response_summary(log_responses:Sequence[float],
                     regimes:Optional[Sequence[KouRegime]]=None
                     ) -> Dict:
    """
    Prop 16.14 gate: report a MEAN only if the first moment is finite.
    Otherwise refuse and give quantiles.
    """
    x=np.asarray(log_responses,float)
    regs=list(regimes) if regimes else [fit_kou_tail(x)]
    me=moment_estimable(regs,1.0)
    out={"n":len(x),"regimes":[{"name":r.name,"eta_up":r.eta_up,
                                "eta_dn":r.eta_dn,"occupation":r.occupation}
                               for r in regs],
         "first_moment_finite":me["finite"],
         "quantiles":{"q10":float(np.quantile(x,0.10)),
                      "q50":float(np.quantile(x,0.50)),
                      "q90":float(np.quantile(x,0.90))}}
    if me["finite"]:
        out["mean_log_response"]=float(x.mean())
        out["report"]="mean is estimable (Prop 16.14 satisfied)"
    else:
        out["REFUSED"]=("E S^1 is INFINITE for "
                        f"{me['violating_regimes']}. Do NOT report a mean. "
                        "Use the quantiles above.")
        out["report"]="mean refused; quantiles reported instead"
    if len(regs)>1:
        out["tail_governance"]=heaviest_regime_dominates(regs,T=float(len(x)))
    return out

def require_reportable_mean(log_responses:Sequence[float]) -> Dict:
    r=response_summary(log_responses)
    if not r["first_moment_finite"]:
        raise InfiniteMoment(r["REFUSED"])
    return r

def build_dataset(parsed:Dict,response_map:Dict[str,float],
                  log_transform:bool=True,sd:Optional[SDGate]=None) -> Dict:
    sd=sd or SDGate()
    ids=parsed["sample_ids"]; X=parsed["expression"]
    if X is None: raise Refusal("no expression matrix")
    keep=[i for i,s in enumerate(ids) if s in response_map]
    if len(keep)<8: raise Refusal(f"only {len(keep)} samples with response")
    Xk=X[:,keep].T
    y=np.array([response_map[ids[i]] for i in keep],np.float32)
    if log_transform and np.nanmax(Xk)>100: Xk=np.log2(np.clip(Xk,1,None))
    mu=np.nanmean(Xk,0); s=np.nanstd(Xk,0)+1e-6
    Xk=np.nan_to_num((Xk-mu)/s,nan=0.0)
    tail=response_summary(np.log(np.maximum(y,1e-9)))
    cert=sd.certify("composition_record",(),{"n":len(keep)})
    return {"X":Xk,"y":y,"sample_ids":[ids[i] for i in keep],
            "response_tail":tail,"sd":cert["status"],
            "NOTE":"verify organoid provenance and response units"}

def selftest(a=None) -> bool:
    print("="*76); print("  ORGANOID GEO v2 — self-test"); print("="*76)
    ok=True; sd=SDGate(); rng=np.random.default_rng(0)
    print("\n[T1] accession validation")
    b=False
    try: series_matrix_url("NOT_A_GSE")
    except Refusal: b=True
    print(f"   bad accession refused={b}"); ok &= b
    print(f"   url={series_matrix_url('GSE123456')}")
    print("\n[T2] screening refuses to certify")
    fake={"meta":{"Series_title":["intestinal organoid drug screen"],
                  "Sample_characteristics_ch1":["treatment: CHIR 3uM"]},
          "sample_ids":["GSM1","GSM2"],"expression":np.zeros((5,2)),
          "n_samples":2,"n_features":5}
    sc=screen(fake,"GSE999999",sd)
    print(f"   verdict={sc['verdict']} sd={sc['sd_status']}")
    ok &= sc["sd_status"]==MODE_E
    print("\n[T3] NEW: moment-estimability gate")
    light=rng.normal(0,0.5,60)
    r1=response_summary(light)
    print(f"   light tails: mean finite={r1['first_moment_finite']}")
    ok &= r1["first_moment_finite"]
    heavy=np.concatenate([rng.normal(0,0.3,40), rng.exponential(3.0,20)])
    r2=KouRegime("heavy",p=0.6,eta_up=0.7,eta_dn=4.0,lam=1.0)
    r3=response_summary(heavy,regimes=[r2])
    print(f"   eta+=0.7 -> mean finite={r3['first_moment_finite']}")
    ok &= not r3["first_moment_finite"]
    print(f"   REFUSED present={'REFUSED' in r3}")
    ok &= "REFUSED" in r3
    print(f"   quantiles offered={'quantiles' in r3}")
    ok &= "quantiles" in r3
    b=False
    try: require_reportable_mean(heavy)
    except InfiniteMoment: b=True
    print("\n[T4] reporting an infinite mean raises")
    print(f"   raised={b}"); ok &= b
    print("\n[T5] heaviest-regime governance")
    rare=KouRegime("rare_hyper",p=0.5,eta_up=0.5,eta_dn=5.0,
                   lam=0.5,occupation=0.03)
    common=KouRegime("common",p=0.5,eta_up=4.0,eta_dn=4.0,
                     lam=1.0,occupation=0.97)
    hv=heaviest_regime_dominates([common,rare],T=60.0)
    print(f"   governing={hv['governing_regimes']} "
          f"occ={hv['governing_occupation_total']:.3f}")
    ok &= hv["governing_regimes"]==["rare_hyper"]
    rr=sd.report()
    print(f"\n   SD: {rr['n']} outputs, {rr['mode_E']} Mode E")
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
NO ACCESSION IS PRE-CERTIFIED HERE.
A CANDIDATE verdict means metadata is consistent with organoid drug
response, not that usable data exists. Confirm on the GEO page and in
the paper. And before reporting a mean response, check Prop 16.14:
if eta+ <= 1 the mean is INFINITE and must not be reported.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","screen","parse","tail"])
    ap.add_argument("--gse",nargs="*",default=[])
    ap.add_argument("--out",default="data/geo")
    ap.add_argument("--responses",help="JSON list of log-responses")
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if a.cmd=="tail":
        if not a.responses: log.error("--responses required"); return
        y=json.loads(Path(a.responses).read_text())
        print(json.dumps(response_summary(y),indent=2,default=str)); return
    reports=[]
    for g in a.gse:
        p=fetch_series_matrix(g,a.out)
        if p is None: reports.append({"accession":g,"verdict":"FETCH_FAILED"}); continue
        parsed=parse_series_matrix(p)
        if a.cmd=="parse":
            print(json.dumps({"accession":g,"n_samples":parsed["n_samples"],
                              "meta_keys":list(parsed["meta"])[:30]},indent=2))
        else:
            r=screen(parsed,g); reports.append(r)
            print(json.dumps(r,indent=2))
    if reports: Path(a.out,"screen.json").write_text(json.dumps(reports,indent=2))

if __name__=="__main__": main()
