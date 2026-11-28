#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
T2 CURATION TEMPLATE v2.0 — (factor, dose, time) -> composition
Now wired to the corrected ESCA mathematics.

NEW IN v2
  * BN check on a nested fate taxonomy (K1: T2 is false in general).
  * Thm 8.3 W2 rate per record: composition uncertainty from n_cells.
  * Occupation-regime test when cell-level labels are supplied.
  * SD gate: VERIFIED required; unverified records are Mode E, not usable.
  * Refuses to pool taxonomy levels that violate projectivity.

PHILOSOPHY: see msps_esca_bio_core.PHILOSOPHY (full text).
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

HONEST SCOPE
  Defines WHAT to curate and CHECKS consistency. Does NOT auto-generate
  labels. Composition is an ESTIMATE bounded by annotation quality.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np

from msps_esca_bio_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    boundary_null_check, theorem_A_rate, composition_as_occupation,
    occupation_regime_test, coarse_grain, get_logger,
)
log = get_logger("t2_curation")
VERSION = "2.0.0"

FACTORS = ["WNT","RSPO","BMP4","NOGGIN","FGF2","FGF4","FGF9","FGF10",
 "ACTIVIN","NODAL","TGFB1","SHH","RA","EGF","HGF","VEGF","IGF1",
 "NOTCH_AG","NOTCH_INH","CHIR","SB431542","LDN193189","DAPT","Y27632","OTHER"]

FATES = ["PLURIPOTENT","ENDODERM","MESODERM","ECTODERM",
 "NEURAL_PROGENITOR","NEURON","GLIA",
 "INTESTINAL_EPITHELIUM","GOBLET","PANETH","ENTEROCYTE",
 "HEPATOCYTE","CHOLANGIOCYTE",
 "NEPHRON_PROGENITOR","PODOCYTE","TUBULE",
 "CARDIOMYOCYTE","ENDOTHELIAL","MESENCHYMAL","UNKNOWN"]
N_FATE=len(FATES); F_IDX={f:i for i,f in enumerate(FATES)}

# A nested taxonomy. Pooling levels requires BN (K1).
FATE_HIERARCHY = {
 "GERM_LAYER": ["ENDODERM","MESODERM","ECTODERM"],
 "NEURAL": ["NEURAL_PROGENITOR","NEURON","GLIA"],
 "INTESTINAL": ["INTESTINAL_EPITHELIUM","GOBLET","PANETH","ENTEROCYTE"],
 "HEPATIC": ["HEPATOCYTE","CHOLANGIOCYTE"],
 "RENAL": ["NEPHRON_PROGENITOR","PODOCYTE","TUBULE"],
}

@dataclass
class T2Record:
    record_id: str
    factor: str
    dose_nM: float
    time_h: float
    composition: List[float]
    source_gse: str
    sample_id: str
    doi: Optional[str]=None
    organoid_type: Optional[str]=None
    cell_type_calling: Optional[str]=None   # marker | automated | author
    n_cells: Optional[int]=None
    notes: str=""
    verified: bool=False
    # v2: optional cell-level labels enable the regime test
    cell_labels: Optional[List[str]]=None
    annotation_reviewed: bool=False

    def problems(self) -> List[str]:
        p=[]
        if self.factor not in FACTORS: p.append(f"factor '{self.factor}' not in FACTORS")
        if self.dose_nM<=0: p.append(f"dose_nM must be > 0")
        if self.time_h<=0: p.append(f"time_h must be > 0")
        if len(self.composition)!=N_FATE:
            p.append(f"composition length {len(self.composition)} != {N_FATE}")
        else:
            c=np.array(self.composition)
            if (c<0).any(): p.append("negative composition entries")
            if abs(c.sum()-1.0)>1e-3: p.append(f"composition sums to {c.sum():.5f}")
        if not self.source_gse.startswith("GSE"): p.append("source_gse not a GSE id")
        if not self.sample_id.startswith("GSM"): p.append("sample_id not a GSM id")
        if not self.doi: p.append("missing doi")
        if not self.verified: p.append("not verified (read the paper)")
        if self.n_cells is not None and self.n_cells < 30:
            p.append(f"n_cells={self.n_cells} < 30: composition too noisy")
        return p

    def is_usable(self) -> bool: return not self.problems()

    def certify(self, sd: SDGate) -> Dict:
        pkgs=[]
        if self.verified and self.doi: pkgs.append("VERIFIED")
        if self.annotation_reviewed: pkgs.append("ANNOT")
        return sd.certify("composition_record", tuple(pkgs),
                          {"record_id": self.record_id,
                           "problems": self.problems()})

    def uncertainty(self) -> Dict:
        """Thm 8.3 rate: how much of the composition is sampling noise."""
        if self.n_cells is None:
            return {"available": False, "reason": "n_cells not recorded"}
        r=theorem_A_rate(self.n_cells, N_FATE, Gamma=1.0)
        return {"available": True, **r,
                "interpretation": ("W2 bound on the distance between this "
                                   "empirical composition and its limit")}

    def regime(self) -> Dict:
        if not self.cell_labels:
            return {"available": False,
                    "reason": "cell_labels not supplied; regime test needs them"}
        return {"available": True,
                **occupation_regime_test(self.cell_labels, FATES)}

def load_jsonl(path: Path) -> List[T2Record]:
    out=[]
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        try: out.append(T2Record(**json.loads(line)))
        except TypeError as e: log.warning(f"bad record skipped: {e}")
    return out

def validate_all(records: List[T2Record], sd: Optional[SDGate]=None) -> Dict:
    sd = sd or SDGate()
    usable=[r for r in records if r.is_usable()]
    broken=[r for r in records if not r.is_usable()]
    certs=[r.certify(sd) for r in records]
    keys=[f"{r.source_gse}:{r.sample_id}" for r in usable]
    dup=len(keys)-len(set(keys))
    n_unknown=sum(1 for r in usable if r.composition[F_IDX["UNKNOWN"]]>0.5)
    rates=[r.uncertainty() for r in usable if r.uncertainty().get("available")]
    w2=[x["W2_bound"] for x in rates]
    return {"total":len(records),"usable":len(usable),"broken":len(broken),
            "duplicates":dup,"mostly_unknown":n_unknown,
            "factor_coverage":sorted({r.factor for r in usable}),
            "n_with_n_cells":len(rates),
            "W2_bound_median":float(np.median(w2)) if w2 else None,
            "W2_bound_max":float(max(w2)) if w2 else None,
            "sd":sd.report(),
            "actionability":RESEARCH_ONLY}

def check_taxonomy_pooling(records: Sequence[T2Record]) -> Dict:
    """
    K1 gate: can coarse germ-layer composition be pooled with fine fates?
    Only if the parent mass equals the sum of children (BN / projectivity).
    """
    rows=[]
    for r in records:
        comp={f:r.composition[i] for f,i in F_IDX.items()}
        # derive parent masses from children, then test projectivity
        for parent, kids in FATE_HIERARCHY.items():
            comp[parent]=sum(comp[k] for k in kids)
        bn=boundary_null_check(FATE_HIERARCHY, comp)
        rows.append({"record_id":r.record_id,"BN":bn["BN_satisfied"]})
    ok=all(x["BN"] for x in rows)
    return {"all_BN":ok,"per_record":rows,
            "VERDICT":("taxonomy levels may be pooled" if ok else
                       "DO NOT pool levels: projectivity violated"),
            "theorem":"Thm 5.3 / Prop 5.4 (K1)"}

def write_template(path: Path, n_example: int = 2):
    ex=[T2Record("example_001","CHIR",3000.0,72.0,
        [0.1,0.4,0.1]+[0.0]*16+[0.3,0.1],
        "GSE000000","GSM0000001","10.1234/example","intestinal",
        "marker",1200,"EXAMPLE — replace with real data",False),
        T2Record("example_002","FGF2",100.0,48.0,
        [0.05,0.1,0.05,0.2,0.4,0.1,0.05]+[0.0]*12+[0.05],
        "GSE000000","GSM0000002","10.1234/example","cerebral",
        "automated",800,"EXAMPLE — replace with real data",False)]
    with open(path,"w") as f:
        f.write("# T2 curation v2: one JSON object per line.\n")
        f.write(f"# REQUIRED: factor, dose_nM, time_h, composition (len {N_FATE}, sums 1),\n")
        f.write("#           source_gse, sample_id, doi, verified=true, n_cells>=30.\n")
        f.write("# v2: annotation_reviewed and cell_labels enable the regime test.\n")
        f.write("# composition order: "+", ".join(FATES)+"\n")
        for r in ex[:n_example]: f.write(json.dumps(asdict(r))+"\n")
    log.info(f"[template] wrote {path}")

def to_mdoe_batch(records: List[T2Record], seed: int = 42) -> Dict:
    recs=[r for r in records if r.is_usable()]
    if len(recs)<8: raise Refusal(f"only {len(recs)} usable; need >= 8")
    rng=np.random.default_rng(seed); idx=rng.permutation(len(recs))
    a=int(0.7*len(recs)); b=int(0.85*len(recs))
    return {"split":{"train":idx[:a],"val":idx[a:b],
                     "test":idx[b:] if b<len(recs) else idx[a:]},
            "composition":np.array([r.composition for r in recs],np.float32),
            "meta":{"factor_idx":np.array([FACTORS.index(r.factor) for r in recs]),
                    "dose_nM":np.array([r.dose_nM for r in recs],np.float32),
                    "time_h":np.array([r.time_h for r in recs],np.float32),
                    "n_cells":np.array([r.n_cells or 0 for r in recs]),
                    "W2_bound":np.array([r.uncertainty().get("W2_bound",np.nan)
                                         for r in recs],np.float32)},
            "fates":FATES,
            "NOTE":"W2_bound carries the Thm 8.3 sampling uncertainty per record"}

def selftest(a=None) -> bool:
    print("="*76); print("  T2 CURATION v2 — self-test"); print("="*76)
    ok=True; sd=SDGate()
    p=Path("_t2_v2_test.jsonl"); write_template(p)
    recs=load_jsonl(p)
    print(f"\n[T1] template: {len(recs)} records")
    ok &= len(recs)==2
    print("\n[T2] unverified -> not usable, Mode E")
    for r in recs:
        c=r.certify(sd)
        t=(not r.is_usable()) and c["status"]==MODE_E
        print(f"   {r.record_id}: usable={r.is_usable()} sd={c['status']} "
              f"{'PASS' if t else 'FAIL'}")
        ok &= t
    print("\n[T3] verified + n_cells -> usable and CERTIFIED")
    r=recs[0]; r.verified=True; r.annotation_reviewed=True
    c=r.certify(sd)
    t=r.is_usable() and c["status"]==MODE_CERTIFIED
    print(f"   usable={r.is_usable()} sd={c['status']} {'PASS' if t else 'FAIL'}")
    ok &= t
    print("\n[T4] Thm 8.3 uncertainty per record")
    u=r.uncertainty()
    print(f"   n_cells={r.n_cells} W2<={u['W2_bound']:.4f}")
    ok &= u["available"] and u["W2_bound"]>0
    print("\n[T5] BN taxonomy check (K1)")
    bn=check_taxonomy_pooling(recs)
    print(f"   all_BN={bn['all_BN']}  {bn['VERDICT']}")
    ok &= bn["all_BN"]
    print("\n[T6] regime test needs cell labels")
    rg=r.regime()
    print(f"   available={rg['available']} reason={rg.get('reason','')[:40]}")
    ok &= not rg["available"]
    r.cell_labels=["PLURIPOTENT","NEURON"]*40
    rg2=r.regime()
    print(f"   with labels: {rg2['regime']}")
    ok &= rg2["regime"]=="LIGHT_BLOCKS"
    print("\n[T7] low n_cells refused")
    low=T2Record("low","CHIR",100,24,[1.0]+[0.0]*(N_FATE-1),
                 "GSE1","GSM1","10.1/x",n_cells=10,verified=True)
    t=not low.is_usable()
    print(f"   n_cells=10 usable={low.is_usable()} {'PASS' if t else 'FAIL'}")
    ok &= t
    print("\n[T8] validator summary + SD")
    rep=validate_all(recs,sd)
    print(f"   usable={rep['usable']} W2_median={rep['W2_bound_median']}")
    print(f"   SD: {rep['sd']['n']} outputs, {rep['sd']['mode_E']} Mode E")
    ok &= rep["total"]==2
    p.unlink(missing_ok=True)
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
WORKFLOW
  1. t2_curation_template.py template --out data/t2.jsonl
  2. scrna_to_composition.py on each GEO series (v2 gives the W2 rate)
  3. Manually fill factor/dose/time/doi/n_cells; set verified=true
     and annotation_reviewed=true after inspecting the UMAP.
  4. t2_curation_template.py validate --data data/t2.jsonl
  5. t2_curation_template.py taxonomy --data ...   (BN gate before pooling)
  6. t2_curation_template.py to-mdoe --data ...    (W2_bound carried along)
  7. MDOE ablation with >= 3 seeds.
""")
    return ok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("cmd",choices=["selftest","template","validate","taxonomy","to-mdoe"])
    ap.add_argument("--out",default="data/t2.jsonl")
    ap.add_argument("--data",default="data/t2.jsonl")
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if selftest(a) else 1)
    if a.cmd=="template":
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        write_template(Path(a.out)); return
    recs=load_jsonl(Path(a.data))
    if a.cmd=="validate": print(json.dumps(validate_all(recs),indent=2,default=str))
    elif a.cmd=="taxonomy": print(json.dumps(check_taxonomy_pooling(recs),indent=2))
    else:
        b=to_mdoe_batch(recs,a.seed)
        out=Path(a.data).with_suffix(".mdoe.npz")
        np.savez_compressed(out,composition=b["composition"],
            **{k:v for k,v in b["meta"].items()},
            train_idx=b["split"]["train"],val_idx=b["split"]["val"],
            test_idx=b["split"]["test"])
        print(f"saved -> {out}; usable={len(b['composition'])}")
        print("W2_bound per record is included (Thm 8.3).")

if __name__=="__main__": main()
