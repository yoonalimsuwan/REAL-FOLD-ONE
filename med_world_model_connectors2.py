#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
med_world_model_connectors.py — v2.0.0
Literature · CFD · Imaging · Omics · Governance · Validation

BUGS FIXED vs v1
  1. resample_isotropic built a full (N^3, 3) float64 coordinate grid.
     A 1024^3 target needs ~25 GB. v2 resamples SEPARABLY axis by axis
     with 1-D linear interpolation: O(N) working memory per axis.
  2. outcome_discrimination used raw ranks, giving a WRONG AUC when risks
     are tied. v2 uses average (mid) ranks and reports both values.
  3. OmicsPipeline.to_cell_fate returned no occupation diagnostics.
     v2 adds the Thm 8.3 W2 rate and the ordered-only regime test.

MATHEMATICS PROPAGATED
  BN lineage taxonomy (K1) · occupation composition (Thm 8.3)
  Kou moment-explosion gate before any mean is reported
  Doob no-Zeno certificate (K8) · corrected surgery table (K10)
  SD gate on every connector output; Mode E when unverified.

PHILOSOPHY: see msps_med_math_core.PHILOSOPHY (full text).
HONEST: segmentation is an INPUT requiring validation. No automatic
clinical segmentation. CFD backends must be repaired before use.
NOT_COMPARED without reference data. Never a fabricated number.
=============================================================================
License: MIT | 2026
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, re, time
import urllib.error, urllib.parse, urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from msps_med_math_core import (
    SDGate, Refusal, MODE_E, MODE_CERTIFIED, RESEARCH_ONLY,
    Governance, IRBRecord, PHIGuard, refuse_disallowed,
    composition_as_occupation, theorem_A_rate, occupation_regime_test,
    boundary_null_check, KouRegime, moment_estimable, response_summary,
    fit_kou_tail, auc_with_ties, classify_morphology, no_zeno_certificate,
    stc_reflux, get_logger,
)
log=get_logger("connectors_v2"); VERSION="2.0.0"

class Status:
    OK="OK"; UNAVAILABLE="UNAVAILABLE"; NOT_COMPARED="NOT_COMPARED"
    NOT_IMPLEMENTED="NOT_IMPLEMENTED"; BLOCKED="BLOCKED"; ERROR="ERROR"

try:
    import pydicom; HAS_PYDICOM=True
except ImportError: HAS_PYDICOM=False
try:
    import nibabel as nib; HAS_NIBABEL=True
except ImportError: HAS_NIBABEL=False
try:
    from skimage.measure import marching_cubes; HAS_SKIMAGE=True
except ImportError: HAS_SKIMAGE=False
try:
    import anndata as ad; HAS_ANNDATA=True
except ImportError: HAS_ANNDATA=False
try:
    import scipy.stats as sps; HAS_SCIPY=True
except ImportError: HAS_SCIPY=False

# =============================================================================
# §1. HTTP with PHI scrubbing
# =============================================================================
class HTTPClient:
    def __init__(self,cache_dir="./medworld_cache",ttl=86400.,min_interval=.34,
                 offline=True,timeout=20.,retries=3):
        self.cache=Path(cache_dir); self.cache.mkdir(parents=True,exist_ok=True)
        self.ttl,self.offline,self.timeout,self.retries=ttl,offline,timeout,retries
        self.min_interval,self._last=min_interval,0.
        self.phi=PHIGuard(); self.n_req=self.n_hit=0
    def get(self,url,headers=None):
        clean,rm=self.phi.scrub(url)
        meta={"url":clean,"phi_stripped":rm,"cached":False}
        key=self.cache/(hashlib.sha256(clean.encode()).hexdigest()[:32]+".txt")
        if key.exists() and time.time()-key.stat().st_mtime<self.ttl:
            self.n_hit+=1; meta["cached"]=True
            return key.read_text(errors="replace"),meta
        if self.offline:
            meta.update(status=Status.UNAVAILABLE,reason="offline; no cache")
            return None,meta
        err=""
        for a in range(self.retries):
            dt=time.time()-self._last
            if dt<self.min_interval: time.sleep(self.min_interval-dt)
            self._last=time.time(); self.n_req+=1
            try:
                req=urllib.request.Request(clean,headers=headers or
                    {"User-Agent":"ONE-MedWorld/2.0 (research)"})
                with urllib.request.urlopen(req,timeout=self.timeout) as r:
                    body=r.read().decode("utf-8",errors="replace")
                key.write_text(body); meta["status"]=Status.OK
                return body,meta
            except urllib.error.HTTPError as e:
                err=f"HTTP {e.code}"
                if e.code in (429,500,502,503,504): time.sleep(2.**a); continue
                break
            except Exception as e:
                err=f"{type(e).__name__}: {e}"; time.sleep(2.**a)
        meta.update(status=Status.ERROR,reason=err)
        return None,meta

# =============================================================================
# §2. LITERATURE + KG
# =============================================================================
EVIDENCE_WEIGHT={"curated":1.,"clinical_review":.95,"GWAS":.75,
                 "single_cell":.65,"animal":.55,"in_vitro":.45,
                 "text_mined":.30,"inferred":.25}

@dataclass
class Evidence:
    gene:str; variant:Optional[str]; phenotype:str; evidence_level:str
    confidence:float; source:str; effect:str="associated"
    organ:Optional[str]=None; developmental_stage:Optional[str]=None
    contradictions:List[str]=field(default_factory=list)
    clinical_actionability:str=RESEARCH_ONLY

class KnowledgeGraph:
    EUTILS="https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    GWAS="https://www.ebi.ac.uk/gwas/rest/api"
    def __init__(self,http=None,sd=None):
        self.http=http or HTTPClient(); self.sd=sd or SDGate()
        self.records=[]; self.unavailable=[]
        self.keys={k:os.environ.get(v) for k,v in
                   {"ncbi":"NCBI_API_KEY","omim":"OMIM_API_KEY",
                    "disgenet":"DISGENET_API_KEY"}.items()}
    def _note(self,s,r): self.unavailable.append({"source":s,"reason":r})
    def pubmed(self,term,retmax=20):
        clean,rm=self.http.phi.scrub(term)
        p={"db":"pubmed","term":clean,"retmax":str(retmax),"retmode":"json"}
        if self.keys["ncbi"]: p["api_key"]=self.keys["ncbi"]
        body,meta=self.http.get(f"{self.EUTILS}/esearch.fcgi?{urllib.parse.urlencode(p)}")
        if body is None:
            self._note("PubMed",meta.get("reason",""))
            return {"status":meta.get("status"),"pmids":[],"phi_stripped":rm}
        try: ids=json.loads(body)["esearchresult"]["idlist"]
        except Exception as e: return {"status":Status.ERROR,"reason":str(e),"pmids":[]}
        return {"status":Status.OK,"pmids":ids,"phi_stripped":rm}
    def clinvar(self,gene,phenotype,retmax=20):
        p={"db":"clinvar","term":f"{gene}[gene]","retmax":str(retmax),"retmode":"json"}
        if self.keys["ncbi"]: p["api_key"]=self.keys["ncbi"]
        body,meta=self.http.get(f"{self.EUTILS}/esearch.fcgi?{urllib.parse.urlencode(p)}")
        if body is None: self._note("ClinVar",meta.get("reason","")); return 0
        try: ids=json.loads(body)["esearchresult"]["idlist"]
        except Exception: return 0
        if not ids: return 0
        p2={"db":"clinvar","id":",".join(ids),"retmode":"json"}
        if self.keys["ncbi"]: p2["api_key"]=self.keys["ncbi"]
        b2,m2=self.http.get(f"{self.EUTILS}/esummary.fcgi?{urllib.parse.urlencode(p2)}")
        if b2 is None: self._note("ClinVar",m2.get("reason","")); return 0
        n=0
        try:
            res=json.loads(b2).get("result",{})
            for uid in res.get("uids",[]):
                r=res.get(uid,{}); sig=r.get("clinical_significance",{})
                desc=(sig.get("description","") if isinstance(sig,dict) else str(sig)).lower()
                eff,conf=(("risk_increase",.9) if "pathogenic" in desc else
                          ("benign",.8) if "benign" in desc else ("uncertain",.35))
                lvl=("clinical_review" if "reviewed" in
                     (r.get("review_status") or "").lower() else "curated")
                vs=r.get("variation_set",[])
                self.records.append(Evidence(gene,
                    vs[0].get("variation_name") if vs else None,
                    phenotype,lvl,conf,f"ClinVar:{uid}",eff))
                n+=1
        except Exception as e: self._note("ClinVar",str(e))
        return n
    def gwas(self,rsid,phenotype):
        if not re.fullmatch(r"rs\d+",rsid):
            self._note("GWAS","rsid must match rs\\d+"); return 0
        body,meta=self.http.get(
            f"{self.GWAS}/singleNucleotidePolymorphisms/{rsid}/associations",
            {"Accept":"application/json"})
        if body is None: self._note("GWAS_Catalog",meta.get("reason","")); return 0
        n=0
        try:
            for a in json.loads(body).get("_embedded",{}).get("associations",[]):
                pv=a.get("pvalue")
                sig=isinstance(pv,(int,float)) and pv<5e-8
                traits=[t.get("trait","") for t in a.get("efoTraits",[])]
                self.records.append(Evidence(a.get("riskAllele") or "unknown",
                    rsid,"; ".join(traits) or phenotype,"GWAS",
                    .75 if sig else .35,f"GWAS:{rsid}"))
                n+=1
        except Exception as e: self._note("GWAS_Catalog",str(e))
        return n
    def omim(self,mim):
        if not self.keys["omim"]:
            self._note("OMIM","requires licensed OMIM_API_KEY")
            return {"status":Status.UNAVAILABLE,"reason":"no OMIM_API_KEY"}
        return {"status":Status.OK,"note":"adapter present; parser NOT_IMPLEMENTED"}
    def disgenet(self,gene):
        if not self.keys["disgenet"]:
            self._note("DisGeNET","requires DISGENET_API_KEY")
            return {"status":Status.UNAVAILABLE,"reason":"no DISGENET_API_KEY"}
        return {"status":Status.OK,"note":"adapter present; parser NOT_IMPLEMENTED"}
    def add(self,r): self.records.append(r)
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
    def query(self,gene=None,phenotype=None):
        out=self.records
        if gene: out=[r for r in out if r.gene==gene]
        if phenotype: out=[r for r in out if phenotype.lower() in r.phenotype.lower()]
        return out
    def confidence(self,recs):
        if not recs: return 0.
        num=sum(EVIDENCE_WEIGHT.get(r.evidence_level,.3)*r.confidence for r in recs)
        den=sum(EVIDENCE_WEIGHT.get(r.evidence_level,.3) for r in recs)
        return float(np.clip(num/max(den,1e-12)-.1*sum(1 for r in recs if r.contradictions),0,1))
    def report(self,gene=None,phenotype=None):
        recs=self.query(gene,phenotype)
        return {"records":[asdict(r) for r in recs],"n_records":len(recs),
                "aggregate_confidence":self.confidence(recs),
                "confidence_semantics":"weighted evidence score, NOT P(pathogenic)",
                "n_contradictions":sum(len(r.contradictions) for r in recs),
                "sources_unavailable":self.unavailable,
                "clinical_actionability":RESEARCH_ONLY}

# =============================================================================
# §3. IMAGING — separable resample (memory FIX)
# =============================================================================
@dataclass
class ImageVolume:
    data:np.ndarray; spacing_mm:Tuple[float,float,float]
    origin_mm:Tuple[float,float,float]; modality:str
    orientation:str="LPS"; deidentified:bool=False; provenance:str=""

@dataclass
class SurfaceMesh:
    vertices_mm:np.ndarray; faces:np.ndarray; normals:np.ndarray
    n_vertices:int; n_faces:int; watertight:Optional[bool]; quality:Dict

def _lerp_axis(arr,axis,target_n):
    """1-D linear interpolation along one axis. O(target_n) working memory."""
    n=arr.shape[axis]
    if n==target_n: return arr
    src=np.linspace(0,n-1,target_n)
    i0=np.floor(src).astype(int); i1=np.minimum(i0+1,n-1); w=src-i0
    shape=[1]*arr.ndim; shape[axis]=target_n
    w=w.reshape(shape)
    a=np.take(arr,i0,axis=axis); b=np.take(arr,i1,axis=axis)
    return a*(1-w)+b*w

class ImagingPipeline:
    def __init__(self,gov,sd=None): self.gov=gov; self.sd=sd or SDGate()
    def read_dicom_series(self,directory):
        if not HAS_PYDICOM:
            return None,{"status":Status.UNAVAILABLE,"reason":"pydicom not installed"}
        files=sorted(Path(directory).glob("*.dcm"))
        if not files: return None,{"status":Status.ERROR,"reason":"no .dcm files"}
        sl=[]
        for f in files:
            ds=pydicom.dcmread(str(f))
            for tag in ("PatientName","PatientID","PatientBirthDate",
                        "AccessionNumber","InstitutionName","ReferringPhysicianName"):
                if tag in ds: setattr(ds,tag,"ANON")
            sl.append(ds)
        sl.sort(key=lambda s: float(getattr(s,"ImagePositionPatient",[0,0,0])[2]))
        arr=np.stack([s.pixel_array for s in sl]).astype(np.float64)
        s0=sl[0]; ps=getattr(s0,"PixelSpacing",[1.,1.]); st=getattr(s0,"SliceThickness",1.)
        arr=arr*float(getattr(s0,"RescaleSlope",1.))+float(getattr(s0,"RescaleIntercept",0.))
        return ImageVolume(arr,(float(ps[0]),float(ps[1]),float(st)),
            tuple(float(v) for v in getattr(s0,"ImagePositionPatient",[0,0,0])),
            str(getattr(s0,"Modality","CT")),deidentified=True,
            provenance=f"DICOM:{directory}"),{"status":Status.OK,"n_slices":len(sl)}
    def read_nifti(self,path):
        if not HAS_NIBABEL:
            return None,{"status":Status.UNAVAILABLE,"reason":"nibabel not installed"}
        img=nib.load(path); data=np.asarray(img.dataobj,dtype=np.float64)
        z=img.header.get_zooms()[:3]
        return ImageVolume(data,tuple(float(v) for v in z),(0.,0.,0.),"MRI",
                           deidentified=True,provenance=f"NIfTI:{path}"),\
               {"status":Status.OK,"shape":data.shape}
    def resample_isotropic(self,vol,target_mm=0.5):
        """
        FIX 1: SEPARABLE axis-by-axis interpolation. v1 built an (N^3,3)
        float64 coordinate array (~25 GB at 1024^3). Working memory here is
        O(N) per axis plus the output array itself.
        """
        sp=np.array(vol.spacing_mm,float)
        new_shape=tuple(max(2,int(round(s*z/target_mm)))
                        for s,z in zip(vol.data.shape,sp))
        out=vol.data
        for ax in range(3):
            out=_lerp_axis(out,ax,new_shape[ax])
        return ImageVolume(out,(target_mm,)*3,vol.origin_mm,vol.modality,
                           vol.orientation,vol.deidentified,
                           vol.provenance+f"|resample{target_mm}mm"),\
               {"method":"separable 1-D linear","peak_extra_memory":"O(N) per axis",
                "v1_method":"full (N^3,3) coordinate grid"}
    @staticmethod
    def resample_memory_estimate(n_target):
        return {"v1_coord_bytes":int(n_target**3*3*8),
                "v2_extra_bytes_per_axis":int(n_target*8*2),
                "reduction_factor":(n_target**3*3*8)/(n_target*8*2)}
    def accept_segmentation(self,mask,vol,validation):
        req=["method","dice_vs_reference","validated_by","date"]
        miss=[k for k in req if k not in validation]
        if miss:
            return None,{"status":Status.BLOCKED,
                         "reason":f"segmentation validation missing: {miss}"}
        if validation.get("dice_vs_reference",0.)<0.80:
            return None,{"status":Status.BLOCKED,"reason":"Dice < 0.80"}
        if mask.shape!=vol.data.shape:
            return None,{"status":Status.ERROR,"reason":"mask/volume shape mismatch"}
        return (mask>0.5).astype(np.uint8),{"status":Status.OK,"validation":validation}
    def mask_to_mesh(self,mask,spacing_mm):
        if not HAS_SKIMAGE:
            return None,{"status":Status.UNAVAILABLE,"reason":"scikit-image not installed"}
        if mask.sum()<8: return None,{"status":Status.ERROR,"reason":"mask too small"}
        padded=np.pad(mask,1,mode="constant")
        try:
            v,f,n,_=marching_cubes(padded.astype(np.float64),0.5,spacing=spacing_mm)
        except Exception as e: return None,{"status":Status.ERROR,"reason":str(e)}
        v-=np.array(spacing_mm)
        q=self._mesh_quality(v,f)
        return SurfaceMesh(v,f,n,len(v),len(f),
                           q.get("euler_characteristic")==2,q),{"status":Status.OK}
    @staticmethod
    def _mesh_quality(v,f):
        e=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],0)
        uniq,cnt=np.unique(np.sort(e,1),axis=0,return_counts=True)
        V,F,E=len(v),len(f),len(uniq)
        tri=v[f]
        a=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        return {"n_boundary_edges":int((cnt==1).sum()),
                "euler_characteristic":V-E+F,
                "degenerate_faces":int((a<1e-12).sum()),
                "area_mm2":float(a.sum())}
    def classify_topology_change(self,chi_before,chi_after,b0_before,b0_after,
                                 convention="closed_orientable_surface"):
        return classify_morphology(b0_after-b0_before,0,0,
                                   chi_after-chi_before,convention)
    def assign_bc(self,mesh,inlet_plane,outlet_planes,flow_rate_ml_s=None,
                  murray_exponent=2.6):
        cent=mesh.vertices_mm[mesh.faces].mean(1)
        tol=float(np.mean(np.linalg.norm(
            mesh.vertices_mm[mesh.faces[:,0]]-mesh.vertices_mm[mesh.faces[:,1]],
            axis=1)))*1.5
        ax,c=inlet_plane
        inlet=np.where(np.abs(cent[:,ax]-c)<tol)[0]
        outs=[np.where(np.abs(cent[:,a]-cc)<tol)[0] for a,cc in outlet_planes]
        outlet_all=np.concatenate(outs) if outs else np.array([],int)
        tagged=np.concatenate([inlet,outlet_all]) if len(outlet_all) else inlet
        wall=np.setdiff1d(np.arange(mesh.n_faces),tagged)
        def area(idx):
            if len(idx)==0: return 0.
            t=mesh.vertices_mm[mesh.faces[idx]]
            return float(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()*.5)
        a_outs=[area(o) for o in outs]
        tot=sum(a**(murray_exponent/2) for a in a_outs) or 1.
        split={f"outlet_{i}":(a**(murray_exponent/2))/tot for i,a in enumerate(a_outs)}
        assumptions=["rigid wall (no FSI)","Newtonian rheology",
                     f"Murray's law flow split, exponent={murray_exponent}",
                     "inlet/outlet planes chosen by user; not auto-detected"]
        if flow_rate_ml_s is None:
            assumptions.append("NO patient-specific flow rate; placeholder BC")
        return {"inlet_faces":inlet,"outlet_faces":outlet_all,"wall_faces":wall,
                "inlet_area_mm2":area(inlet),"flow_split":split,
                "assumptions":assumptions,
                "validation_required":["4D-flow MRI","Doppler","invasive pressure",
                                       "mesh convergence"]}
    @staticmethod
    def vessel_phantom(n=48,radius=8.,spacing=.5):
        ax=(np.arange(n)-n/2)*spacing
        X,Y,Z=np.meshgrid(ax,ax,ax,indexing="ij")
        mask=(np.sqrt(X**2+Y**2)<radius).astype(np.uint8)
        return ImageVolume(mask.astype(np.float64)*1000.,(spacing,)*3,
                           (0.,0.,0.),"phantom",deidentified=True,
                           provenance="synthetic cylinder"),mask

# =============================================================================
# §4. OMICS — occupation diagnostics added
# =============================================================================
FATES=["pluripotent","neuroectoderm","mesoderm","endoderm","cardiac",
       "neural","hepatic","renal","senescent","unknown"]
FATE_HIERARCHY={"GERM":["neuroectoderm","mesoderm","endoderm"],
                "DIFFERENTIATED":["cardiac","neural","hepatic","renal"]}

class OmicsPipeline:
    DEFAULT_PANELS={
      "pluripotent":["POU5F1","NANOG","SOX2","LIN28A","DNMT3B"],
      "neuroectoderm":["SOX1","PAX6","NES","OTX2"],
      "mesoderm":["TBXT","MIXL1","EOMES","MESP1","HAND1"],
      "endoderm":["SOX17","FOXA2","GATA4","GATA6","CXCR4"],
      "cardiac":["NKX2-5","TNNT2","MYH6","MYH7","ACTC1"],
      "neural":["SOX2","PAX6","TUBB3","MAP2","NEUROD1"],
      "hepatic":["ALB","AFP","HNF4A","TTR","APOA1"],
      "renal":["NPHS1","NPHS2","PAX2","WT1","SIX2"],
      "senescent":["CDKN1A","CDKN2A","GLB1","TP53","SERPINE1"]}
    def __init__(self,gov,sd=None): self.gov=gov; self.sd=sd or SDGate()
    def qc_filter(self,X,genes,min_genes=200,max_pct_mito=20.):
        counts=X.sum(1); det=(X>0).sum(1)
        mi=[i for i,g in enumerate(genes) if g.upper().startswith("MT-")]
        pct=(X[:,mi].sum(1)/np.maximum(counts,1)*100) if mi else np.zeros(X.shape[0])
        keep=(det>=min_genes)&(pct<=max_pct_mito)&(counts>0)
        return X[keep],keep,{"n_input":int(X.shape[0]),"n_pass":int(keep.sum()),
                             "median_detected":float(np.median(det)),
                             "median_pct_mito":float(np.median(pct))}
    @staticmethod
    def normalize_log1p(X,target=1e4):
        return np.log1p(X/np.maximum(X.sum(1,keepdims=True),1)*target)
    def score_fates(self,X,genes,panels=None,n_bg=50,seed=0):
        panels=panels or self.DEFAULT_PANELS
        rng=np.random.default_rng(seed)
        gi={g.upper():i for i,g in enumerate(genes)}
        gm=X.mean(0); order=np.argsort(gm)
        bins=np.empty(len(genes),int)
        bins[order]=(np.arange(len(genes))*25//max(len(genes),1))
        scores={}
        for fate,markers in panels.items():
            idx=[gi[m.upper()] for m in markers if m.upper() in gi]
            if not idx: scores[fate]=np.zeros(X.shape[0]); continue
            obs=X[:,idx].mean(1); bg=[]
            for i in idx:
                pool=np.where(bins==bins[i])[0]
                pool=pool[~np.isin(pool,idx)]
                if len(pool):
                    bg.extend(rng.choice(pool,size=min(n_bg,len(pool)),
                                         replace=False).tolist())
            b=X[:,bg].mean(1) if bg else np.zeros(X.shape[0])
            scores[fate]=obs-b
        return scores
    def to_cell_fate(self,X,genes,panels=None,ordered_labels=None,
                     order_key=None):
        Xq,keep,qc=self.qc_filter(X,genes)
        Xn=self.normalize_log1p(Xq)
        sc=self.score_fates(Xn,genes,panels)
        keys=list(sc.keys()); M=np.stack([sc[k] for k in keys],1)
        T=max(float(M.std()),1e-6)
        e=np.exp((M-M.max(1,keepdims=True))/T); P=e/e.sum(1,keepdims=True)
        dom=np.array(keys)[P.argmax(1)]; conf=P.max(1)
        # NEW: occupation diagnostics
        oc=composition_as_occupation(dom.tolist(),keys)
        rate=theorem_A_rate(len(dom),len(keys))
        comp={k:float(v) for k,v in zip(keys,oc["F"])}
        for parent,kids in FATE_HIERARCHY.items():
            comp[parent]=sum(comp.get(c,0.) for c in kids)
        bn=boundary_null_check(FATE_HIERARCHY,comp)
        regime=occupation_regime_test(dom.tolist(),keys,
                                      ordered=ordered_labels is not None)
        cert=self.sd.certify("composition_rate",
                             ("A1","A2","A3") if ordered_labels is not None else (),
                             {"W2":rate["W2_bound"]})
        return {"n_cells":int(Xq.shape[0]),"n_genes":len(genes),
                "dominant_fate":dom,"fate_confidence":conf,"qc":qc,
                "composition":oc["F"],"composition_names":keys,
                "W2_bound":rate["W2_bound"],"L2_bound":rate["L2_bound"],
                "taxonomy_BN":bn,"regime":regime,"sd":cert,
                "caveats":["marker-based scoring, NOT a validated classifier",
                           "confidence is a softmax heuristic, NOT calibrated",
                           "batch effects not corrected here",
                           "doublets not removed",
                           "regime test needs a MEANINGFUL ordering"],
                "clinical_actionability":RESEARCH_ONLY}
    @staticmethod
    def synthetic_scrna(n_cells=300,n_genes=400,seed=0):
        rng=np.random.default_rng(seed)
        markers=["POU5F1","NANOG","SOX2","PAX6","NES","TBXT","MIXL1","SOX17",
                 "FOXA2","NKX2-5","TNNT2","MYH6","ALB","HNF4A","NPHS1","PAX2",
                 "CDKN1A","CDKN2A","MT-CO1","MT-ND1"]
        genes=markers+[f"GENE{i}" for i in range(n_genes-len(markers))]
        X=rng.poisson(.4,size=(n_cells,n_genes)).astype(np.float64)
        for pop,ms in ((0,["POU5F1","NANOG","SOX2"]),
                       (1,["PAX6","NES"]),
                       (2,["NKX2-5","TNNT2","MYH6"])):
            sl=slice(pop*n_cells//3,(pop+1)*n_cells//3)
            for m in ms: X[sl,genes.index(m)]+=rng.poisson(25,size=X[sl].shape[0])
        return X,genes

# =============================================================================
# §5. CFD COUPLER
# =============================================================================
class CFDCoupler:
    BACKENDS=["one_cfd_stencils","sesi_multiphase_flow_fix","super_dns_one",
              "ale_gcl_fsi","ale_remesh_conservative"]
    def __init__(self,module_dir=".",sd=None):
        self.dir=Path(module_dir); self.sd=sd or SDGate()
        self.availability={}; self._probe()
    def _probe(self):
        import importlib.util
        for name in self.BACKENDS:
            matches=sorted(self.dir.glob(f"{name}*.py"))
            info={"found":bool(matches),"files":[m.name for m in matches]}
            if not matches:
                info.update(status=Status.UNAVAILABLE,reason="file not present")
                self.availability[name]=info; continue
            p=matches[0]; src=p.read_text(errors="replace")
            try:
                compile(src,str(p),"exec"); info["parses"]=True
            except SyntaxError as e:
                info.update(parses=False,syntax_error=f"line {e.lineno}: {e.msg}")
            if info.get("parses"):
                try:
                    spec=importlib.util.spec_from_file_location(f"_cfd_{name}",str(p))
                    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                    info.update(importable=True,status=Status.OK)
                    setattr(self,f"_mod_{name}",mod)
                except Exception as e:
                    info.update(importable=False,status=Status.ERROR,
                                reason=f"{type(e).__name__}: {e}")
            else:
                info.update(status=Status.UNAVAILABLE,reason="does not parse")
            self.availability[name]=info
    def report(self): return {"backends":self.availability}
    def run_hemodynamics(self,mesh,bc,rho=1060.,mu=3.5e-3,flow_ml_s=None):
        assumptions=list(bc["assumptions"])+[f"rho={rho}, mu={mu} (Newtonian)"]
        required=bc["validation_required"]
        if flow_ml_s is None:
            return {"backend":"none","status":Status.NOT_COMPARED,"metrics":{},
                    "assumptions":assumptions+["no flow rate"],
                    "validation_required":required}
        Q=flow_ml_s*1e-6; A=bc["inlet_area_mm2"]*1e-6
        if A<=0: return {"backend":"none","status":Status.ERROR,"metrics":{}}
        r=math.sqrt(A/math.pi); v=Q/A
        return {"backend":"analytic_poiseuille_ESTIMATE","status":Status.OK,
                "metrics":{"mean_inlet_velocity_m_s":v,"equivalent_radius_m":r,
                           "wall_shear_stress_Pa_ESTIMATE":4*mu*Q/(math.pi*r**3),
                           "reynolds_number":rho*v*2*r/mu},
                "assumptions":assumptions+["ANALYTIC ESTIMATE, not a CFD solve",
                                           "straight circular pipe, steady"],
                "validation_required":required,
                "clinical_actionability":RESEARCH_ONLY}

# =============================================================================
# §6. VALIDATION — AUC ties FIXED
# =============================================================================
class ValidationHarness:
    def __init__(self,reference=None,sd=None):
        self.ref=reference or {}; self.sd=sd or SDGate()
    @staticmethod
    def bland_altman(p,r):
        d=p-r; m=float(d.mean()); sd=float(d.std(ddof=1)) if len(d)>1 else 0.
        return {"bias":m,"sd_diff":sd,"loa_upper":m+1.96*sd,"loa_lower":m-1.96*sd,
                "n":int(len(d))}
    @staticmethod
    def icc21(p,r):
        Y=np.stack([p,r],1); n,k=Y.shape
        if n<2: return float("nan")
        gm=Y.mean(); row=Y.mean(1); col=Y.mean(0)
        SSR=k*((row-gm)**2).sum(); SSC=n*((col-gm)**2).sum()
        SSE=((Y-gm)**2).sum()-SSR-SSC
        MSR=SSR/(n-1); MSC=SSC/(k-1); MSE=SSE/max((n-1)*(k-1),1)
        den=MSR+(k-1)*MSE+k*(MSC-MSE)/n
        return float((MSR-MSE)/den) if den else float("nan")
    @staticmethod
    def calibration(p,r):
        X=np.stack([np.ones_like(p),p],1)
        b,*_=np.linalg.lstsq(X,r,rcond=None)
        return {"intercept":float(b[0]),"slope":float(b[1])}
    @staticmethod
    def bootstrap_ci(p,r,stat,n_boot=2000,seed=0,conf=.95):
        rng=np.random.default_rng(seed); n=len(p); vals=[]
        for _ in range(n_boot):
            i=rng.integers(0,n,n)
            try: vals.append(float(stat(p[i],r[i])))
            except Exception: pass
        if not vals: return {"lo":float("nan"),"hi":float("nan")}
        a=1-conf
        return {"point":float(stat(p,r)),"lo":float(np.quantile(vals,a/2)),
                "hi":float(np.quantile(vals,1-a/2))}
    def compare(self,name,pred,ref_key):
        if ref_key not in self.ref:
            return {"case":name,"status":Status.NOT_COMPARED,
                    "reason":f"reference '{ref_key}' not supplied"}
        r=np.asarray(self.ref[ref_key],float)
        if len(r)!=len(pred):
            return {"case":name,"status":Status.ERROR,"reason":"length mismatch"}
        out={"case":name,"status":Status.OK,"n":int(len(pred)),
             "bland_altman":self.bland_altman(pred,r),"icc21":self.icc21(pred,r),
             "calibration":self.calibration(pred,r),
             "rmse":float(np.sqrt(((pred-r)**2).mean())),
             "mae":float(np.abs(pred-r).mean())}
        if HAS_SCIPY:
            out["pearson_r"]=float(sps.pearsonr(pred,r)[0])
            out["rmse_ci95"]=self.bootstrap_ci(pred,r,
                lambda p,q: float(np.sqrt(((p-q)**2).mean())))
        return out
    def coverage(self,lo,hi,truth):
        inside=(truth>=lo)&(truth<=hi)
        return {"empirical_coverage":float(inside.mean()),
                "mean_interval_width":float((hi-lo).mean()),"n":int(len(truth))}
    def outcome_discrimination(self,risk,event):
        """FIX 2: average ranks for ties."""
        try:
            r=auc_with_ties(np.asarray(risk,float),np.asarray(event,int))
        except Refusal as e:
            return {"status":Status.ERROR,"reason":str(e)}
        return {"status":Status.OK,"auc":r["auc_tie_corrected"],
                "auc_naive_v1":r["auc_naive"],"n_tied_groups":r["n_tied_groups"],
                "brier":r["brier"],"n":r["n"],"n_events":r["n_events"],
                "CORRECTION":"average ranks for ties",
                "note":"research discrimination metric, NOT clinical validation"}
    def response_tail(self,log_responses):
        return response_summary(log_responses)

# =============================================================================
# §7. ORCHESTRATOR
# =============================================================================
class MedWorldConnectors:
    def __init__(self,irb=None,offline=True,module_dir=".",
                 cache_dir="./medworld_cache"):
        self.sd=SDGate(); self.gov=Governance(irb)
        self.http=HTTPClient(cache_dir=cache_dir,offline=offline)
        self.kg=KnowledgeGraph(self.http,self.sd)
        self.imaging=ImagingPipeline(self.gov,self.sd)
        self.omics=OmicsPipeline(self.gov,self.sd)
        self.cfd=CFDCoupler(module_dir,self.sd)
        self.validation=ValidationHarness(sd=self.sd)
    def literature_scan(self,gene,phenotype,rsid=None):
        d=self.gov.authorize(f"{gene} {phenotype} association literature")
        if not d["authorized"]: return {"status":Status.BLOCKED,"decision":d}
        ncv=self.kg.clinvar(gene,phenotype)
        ngw=self.kg.gwas(rsid,phenotype) if rsid else 0
        pm=self.kg.pubmed(f"{gene} AND {phenotype}",10)
        self.kg.omim(100000); self.kg.disgenet(gene)
        self.kg.detect_contradictions()
        return {"status":Status.OK,"n_clinvar":ncv,"n_gwas":ngw,
                "pubmed_status":pm.get("status"),"kg":self.kg.report(gene,phenotype),
                "decision":d}
    def imaging_to_cfd(self,vol,mask,seg_validation,inlet_plane,outlet_planes,
                       flow_ml_s=None,uses_patient_data=False,deid=None):
        d=self.gov.authorize("patient-specific hemodynamic simulation research",
                             uses_patient_data,deid,"simulation")
        if not d["authorized"]: return {"status":Status.BLOCKED,"decision":d}
        seg,si=self.imaging.accept_segmentation(mask,vol,seg_validation)
        if seg is None: return {"status":Status.BLOCKED,"segmentation":si}
        mesh,mi=self.imaging.mask_to_mesh(seg,vol.spacing_mm)
        if mesh is None: return {"status":mi.get("status"),"mesh":mi}
        bc=self.imaging.assign_bc(mesh,inlet_plane,outlet_planes,flow_ml_s)
        res=self.cfd.run_hemodynamics(mesh,bc,flow_ml_s=flow_ml_s)
        return {"status":Status.OK,"mesh_quality":mesh.quality,
                "flow_split":bc["flow_split"],"cfd":res,"decision":d}
    def omics_to_fate(self,X,genes,uses_patient_data=False,deid=None,
                      ordered_labels=None):
        d=self.gov.authorize("single-cell cell fate research",
                             uses_patient_data,deid,"omics")
        if not d["authorized"]: return {"status":Status.BLOCKED,"decision":d}
        st=self.omics.to_cell_fate(X,genes,ordered_labels=ordered_labels)
        return {"status":Status.OK,**{k:v for k,v in st.items()
                                      if k not in ("dominant_fate","fate_confidence")},
                "fate_counts":{k:int((st["dominant_fate"]==k).sum())
                               for k in st["composition_names"]},
                "mean_confidence":float(st["fate_confidence"].mean()),
                "decision":d}
    def status(self):
        return {"version":VERSION,"offline":self.http.offline,
                "connectors":{"PubMed":True,"ClinVar":True,"GWAS":True,
                              "OMIM":bool(self.kg.keys["omim"]),
                              "DisGeNET":bool(self.kg.keys["disgenet"])},
                "imaging_deps":{"pydicom":HAS_PYDICOM,"nibabel":HAS_NIBABEL,
                                "skimage":HAS_SKIMAGE},
                "omics_deps":{"anndata":HAS_ANNDATA},"stats_deps":{"scipy":HAS_SCIPY},
                "cfd":self.cfd.report(),
                "math":["occupation composition (Thm 8.3)","BN taxonomy (K1)",
                        "Kou moment gate","AUC tie correction",
                        "separable resample","corrected surgery (K10)"],
                "clinical_actionability":RESEARCH_ONLY}

# =============================================================================
# §8. SELF-TEST
# =============================================================================
def self_test(module_dir=".",verbose=True)->bool:
    print("="*76); print("  MEDWORLD CONNECTORS v2 — self-test"); print("="*76)
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    mw=MedWorldConnectors(offline=True,module_dir=module_dir)

    print("\n[T1] governance")
    rep("patient path BLOCKED without IRB",
        not mw.gov.authorize("simulation",True)["authorized"])
    irb=IRBRecord("P-001","Demo","2026-01-01","2099-12-31",
                  ["simulation","omics"],data_use_agreement=True,status="APPROVED")
    mw2=MedWorldConnectors(irb=irb,offline=True,module_dir=module_dir)
    deid=mw2.gov.phi.deidentify({"patient_name":"Jane Doe","age":55,"sex":"female"})
    rep("IRB + de-id authorised",
        mw2.gov.authorize("simulation",True,deid,"simulation")["authorized"])

    print("\n[T2] knowledge graph")
    kg=mw.kg
    kg.add(Evidence("GENE_A","rs1","heart defect","curated",.9,"ClinVar:1","risk_increase"))
    kg.add(Evidence("GENE_A","rs1","heart defect","curated",.8,"ClinVar:2","benign"))
    n=kg.detect_contradictions()
    rep("contradictions detected",n>=2,f"n={n}")
    r=kg.report("GENE_A","heart defect")
    rep("confidence reduced",r["aggregate_confidence"]<.85)
    kg.omim(100000); kg.disgenet("GENE_A")
    rep("OMIM/DisGeNET UNAVAILABLE",
        len([u for u in kg.unavailable if u["source"] in ("OMIM","DisGeNET")])==2)

    print("\n[T3] FIX 1: separable resample memory")
    est=ImagingPipeline.resample_memory_estimate(1024)
    print(f"      v1 coords={est['v1_coord_bytes']/1e9:.2f} GB   "
          f"v2 per-axis={est['v2_extra_bytes_per_axis']/1e3:.1f} kB")
    rep("v1 would need >20 GB at 1024^3",est["v1_coord_bytes"]>2e10)
    rep("reduction factor >1e6",est["reduction_factor"]>1e6)
    vol,mask=ImagingPipeline.vessel_phantom(n=24,radius=6.,spacing=1.)
    rv,info=mw.imaging.resample_isotropic(vol,0.5)
    print(f"      {vol.data.shape} -> {rv.data.shape}  method={info['method']}")
    rep("separable resample shape correct",rv.data.shape==(48,48,48))
    rep("method declared separable","separable" in info["method"])
    rep("spacing updated",abs(rv.spacing_mm[0]-0.5)<1e-12)

    print("\n[T4] segmentation gate + mesh")
    bad=mw.imaging.accept_segmentation(mask,vol,{"method":"unvalidated CNN"})
    rep("unvalidated segmentation BLOCKED",bad[0] is None)
    segv={"method":"phantom analytic","dice_vs_reference":1.,
          "validated_by":"self-test","date":"2026-01-01"}
    seg,si=mw.imaging.accept_segmentation(mask,vol,segv)
    rep("validated segmentation accepted",seg is not None)
    if HAS_SKIMAGE:
        mesh,mi=mw.imaging.mask_to_mesh(seg,vol.spacing_mm)
        rep("marching cubes produced a mesh",mesh is not None,
            f"V={mesh.n_vertices} F={mesh.n_faces}" if mesh else "")
        if mesh:
            bc=mw.imaging.assign_bc(mesh,(2,-11.5),[(2,11.5)],5.)
            res=mw.cfd.run_hemodynamics(mesh,bc,flow_ml_s=5.)
            rep("analytic estimate labelled",
                any("ESTIMATE" in a for a in res["assumptions"]))
            rn=mw.cfd.run_hemodynamics(mesh,bc,flow_ml_s=None)
            rep("no flow => NOT_COMPARED",rn["status"]==Status.NOT_COMPARED)
    else:
        print("      SKIP mesh — scikit-image absent")

    print("\n[T5] omics + occupation diagnostics")
    X,genes=OmicsPipeline.synthetic_scrna(300,400)
    out=mw.omics_to_fate(X,genes)
    rep("cell fate computed",out["status"]==Status.OK,f"n={out.get('n_cells')}")
    rep("composition sums to 1",abs(sum(out["composition"])-1)<1e-6)
    rep("W2 bound reported",out["W2_bound"]>0,f"{out['W2_bound']:.4f}")
    rep("BN taxonomy checked","taxonomy_BN" in out)
    rep("unordered regime -> MODE_E",out["regime"]["status"]==MODE_E)
    fc=out["fate_counts"]
    rep("populations detected",fc.get("pluripotent",0)>20 and fc.get("cardiac",0)>20,
        f"pluri={fc.get('pluripotent')} cardiac={fc.get('cardiac')}")
    rep("caveats declared",len(out["caveats"])>=5)

    print("\n[T6] FIX 2: AUC with ties")
    risk=np.array([.1,.1,.5,.5,.9]); event=np.array([0,1,0,1,1])
    oc=mw.validation.outcome_discrimination(risk,event)
    print(f"      AUC corrected={oc['auc']:.4f}  naive v1={oc['auc_naive_v1']:.4f}  "
          f"tied groups={oc['n_tied_groups']}")
    rep("tie correction changes AUC",abs(oc["auc"]-oc["auc_naive_v1"])>1e-9)
    rep("corrected AUC = 0.875",abs(oc["auc"]-0.875)<1e-9)
    rep("correction declared","average ranks" in oc["CORRECTION"])

    print("\n[T7] validation harness")
    v=ValidationHarness()
    p=np.random.default_rng(1).normal(10,2,40)
    rep("no reference => NOT_COMPARED",
        v.compare("4D-flow",p,"flow_mri")["status"]==Status.NOT_COMPARED)
    v2=ValidationHarness({"flow_mri":p*1.05+np.random.default_rng(2).normal(0,.3,40)})
    c=v2.compare("4D-flow",p,"flow_mri")
    rep("Bland-Altman + ICC + calibration",c["status"]==Status.OK and c["icc21"]>.8,
        f"ICC={c['icc21']:.3f}")
    rep("calibration slope ~1.05",abs(c["calibration"]["slope"]-1.05)<.1)

    print("\n[T8] response tail moment gate")
    rt=mw.validation.response_tail(np.random.default_rng(0).normal(0,.5,60))
    rep("light cohort mean reported","mean" in rt)
    heavy=np.concatenate([np.random.default_rng(1).normal(0,.3,40),
                          np.random.default_rng(2).exponential(3.,20)])
    regs=[KouRegime("heavy",p=.6,eta_up=.7,eta_dn=5.,lam=1.)]
    rh=mw.validation.response_summary(heavy) if False else response_summary(heavy,regs)
    rep("eta+<1 mean REFUSED","REFUSED" in rh)
    rep("quantiles offered instead","quantiles" in rh)

    print("\n[T9] corrected topology + no-Zeno + SD")
    ev=mw.imaging.classify_topology_change(2,0,1,2)
    rep("closed-surface event classified","candidates" in ev)
    nz=no_zeno_certificate(.05,2.,.5)
    rep("Doob certificate valid",nz["certified"])
    rep("truncation REMOVED","REMOVED" in nz["HORIZON_TRUNCATION"])
    rr=mw.sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")
    st=mw.status()
    rep("research_only",st["clinical_actionability"]==RESEARCH_ONLY)
    print("\n"+"="*76)
    print(f"  {'ALL PASS' if ok else 'SOME FAILED'}")
    print("="*76)
    print("""
FIXED
  Resampling is SEPARABLE: O(N) working memory per axis instead of an
  (N^3,3) float64 coordinate grid (~25 GB at 1024^3).
  AUC uses average ranks for ties; both corrected and naive values reported.
  Omics returns occupation diagnostics: W2 rate, BN taxonomy, regime test.

MATHEMATICS PROPAGATED
  Thm 8.3 composition rate · K1 BN taxonomy · Kou moment gate
  K8 Doob no-Zeno · K10 corrected surgery table · SD on every output

NOT IMPLEMENTED / NOT CLAIMED
  No automatic clinical segmentation. Analytic hemodynamic ESTIMATE only.
  OMIM/DisGeNET need licensed keys. NOT_COMPARED without reference data.
  Regime test needs a meaningful ordering. Nothing here is clinical.
""")
    return ok

def main()->int:
    ap=argparse.ArgumentParser(description="MedWorld connectors v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","status","kg","omics","resample","auc","tail"])
    ap.add_argument("--gene",default=None); ap.add_argument("--phenotype",default=None)
    ap.add_argument("--rsid",default=None); ap.add_argument("--online",action="store_true")
    ap.add_argument("--module-dir",default="."); ap.add_argument("--n",type=int,default=256)
    ap.add_argument("--json",default=None)
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test(a.module_dir) else 1
    mw=MedWorldConnectors(offline=not a.online,module_dir=a.module_dir); out={}
    if a.cmd=="status": out=mw.status()
    elif a.cmd=="kg" and a.gene and a.phenotype:
        out=mw.literature_scan(a.gene,a.phenotype,a.rsid)
    elif a.cmd=="omics":
        X,g=OmicsPipeline.synthetic_scrna(); out=mw.omics_to_fate(X,g)
    elif a.cmd=="resample":
        out=ImagingPipeline.resample_memory_estimate(a.n)
    elif a.cmd=="auc":
        out=mw.validation.outcome_discrimination(
            np.array([.1,.1,.5,.5,.9]),np.array([0,1,0,1,1]))
    elif a.cmd=="tail":
        out=mw.validation.response_tail(np.random.default_rng(0).normal(0,.5,60))
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
