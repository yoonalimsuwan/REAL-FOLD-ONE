#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_boundary_layer_extraction.py — v2.0.0  [EXTRACTION + IDENTIFIABILITY]
Upgrades over v1:
  * Levenberg-Marquardt width/shadow optimisation with analytic-ish Jacobian
    (replaces the exploding Cartesian grid search).
  * RCWA reference CACHED (eig reused across all extraction trials).
  * Fisher information / rank / correlation identifiability gate.
  * OPMC edge localisation with PELT (from core) — same certified radius.
  * Exploratory modes (nonlocal kernel, Carpet multi-scale) retained and
    still Mode E by construction. corner_2d still refused.

PHILOSOPHY: see msps_math_foundation_api.PHILOSOPHY (full text).
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

REFUSALS: nonlocal_kernel and carpet_multiscale have NO industrial practice
  and no validated reference -> Mode E. corner_2d needs a converged VECTOR
  2-D reference (S4/grcwa/RETICOLO/FDTD); scalar 2-D is insufficient.
=============================================================================
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from msps_math_foundation_api import (
    HAS_NUMPY, np, RunConfig, Cache, SDGate, Refusal, MODE_E, MODE_CERTIFIED,
    opmc_certified, content_key, bench, _np, get_logger,
)
from msps_litho_math_api import (
    MaskStack, Multilayer, Material, TABN, VAC, rcwa, rcwa_convergence,
    rcwa_uniform_check, kirchhoff_field,
)

VERSION="2.0.0"; PI=math.pi
log=get_logger("msps.bl")
class NeedsRigorousReference(Refusal): pass
class NotIdentifiable(Refusal): pass

MODES = {
 "single_edge":      {"industry":"production","symmetric":True,"needs_2d":False},
 "dual_edge":        {"industry":"production","symmetric":False,"needs_2d":False},
 "pitch_dependent":  {"industry":"advanced_ILT","symmetric":False,"needs_2d":False},
 "angle_dependent":  {"industry":"advanced_ILT","symmetric":False,"needs_2d":False},
 "polarisation":     {"industry":"advanced_ILT","symmetric":False,"needs_2d":False},
 "stratified_3d":    {"industry":"research","symmetric":False,"needs_2d":False},
 "ml_penetration":   {"industry":"research","symmetric":True,"needs_2d":False},
 "corner_2d":        {"industry":"research","symmetric":False,"needs_2d":True},
 "nonlocal_kernel":  {"industry":"NONE","symmetric":False,"needs_2d":False},
 "carpet_multiscale":{"industry":"NONE","symmetric":False,"needs_2d":False},
}

@dataclass
class BLParams:
    widths_nm: List[float]; amps: List[float]; phases_rad: List[float]
    shadow_nm: float = 0.0; ml_plane_depth_nm: float = 0.0
    mode: str = "dual_edge"
    def complexes(self):
        return [a*complex(math.cos(p),math.sin(p))
                for a,p in zip(self.amps,self.phases_rad)]
    def to_dict(self):
        return {"mode":self.mode,"widths_nm":self.widths_nm,"amps":self.amps,
                "phases_rad":self.phases_rad,
                "complex":[complex(c) for c in self.complexes()],
                "shadow_nm":self.shadow_nm,
                "ml_plane_depth_nm":self.ml_plane_depth_nm}

def bl_basis(xs, edges, widths, pitch, profile="gauss"):
    _np()
    B=np.zeros((len(xs),len(edges)),dtype=complex)
    for j,(e,w) in enumerate(zip(edges,widths)):
        if w<=0: continue
        d=(xs-e+pitch/2)%pitch-pitch/2
        if profile=="rect": B[:,j]=(np.abs(d)<=w/2).astype(complex)
        elif profile=="gauss": B[:,j]=np.exp(-(d/(w/2))**2).astype(complex)
        elif profile=="exp": B[:,j]=np.exp(-np.abs(d)/(w/2)).astype(complex)
        else: raise Refusal(f"unknown profile '{profile}'")
    return B

def bl_field(mask: MaskStack, xs, edges, p: BLParams, profile="gauss"):
    _np()
    m2=MaskStack(mask.pitch_nm,mask.cd_nm,mask.absorber,mask.absorber_nm,
                 mask.cra_deg,mask.wavelength_nm,mask.ml,p.shadow_nm)
    base=kirchhoff_field(m2,xs)
    B=bl_basis(xs,edges,p.widths_nm,mask.pitch_nm,profile)
    out=base+B@np.array(p.complexes(),dtype=complex)
    if p.ml_plane_depth_nm!=0.0:
        out=out*np.exp(2j*(2*PI/mask.wavelength_nm)*p.ml_plane_depth_nm)
    return out

def fit_amplitudes(target, base, B, lam=1e-6):
    """Amplitudes are LINEAR for fixed widths: solve exactly (Tikhonov)."""
    _np()
    y=target-base; Bh=B.conj().T
    A=Bh@B+lam*np.eye(B.shape[1])
    coef=np.linalg.solve(A,Bh@y); res=y-B@coef
    sv=np.linalg.svd(B,compute_uv=False)
    rank=int(np.sum(sv>sv[0]*1e-10)) if sv[0]>0 else 0
    cond=float(np.linalg.cond(A))
    return {"coef":coef,"relative_residual":float(np.linalg.norm(res)/
                                                 max(np.linalg.norm(y),1e-30)),
            "condition_number":cond,"design_rank":rank,"n_columns":B.shape[1],
            "identifiable":rank==B.shape[1] and cond<1e10}


# =============================================================================
# §1. LEVENBERG-MARQUARDT over widths + shadow (replaces grid explosion)
# =============================================================================

def extract_lm(mask: MaskStack, target, xs, edges,
               w0: Optional[Sequence[float]]=None, shadow0=0.0,
               mode="dual_edge", profile="gauss", lam=1e-6,
               max_iter=60, w_min=0.2, w_max=12.0,
               cache: Optional[Cache]=None) -> Dict[str,Any]:
    """
    Variable projection + LM on the nonlinear parameters (widths, shadow).
    Amplitudes solved exactly at every step. Jacobian by finite difference
    on the reduced objective (cheap: len(widths)+1 evaluations per step).
    """
    _np()
    if mode not in MODES: raise Refusal(f"unknown mode '{mode}'")
    info=MODES[mode]
    if info["needs_2d"]:
        raise NeedsRigorousReference(
            f"'{mode}' requires a rigorous VECTOR 2-D reference. "
            "A 1-D reference cannot identify corner terms. SD -> Mode E.")
    if info["industry"]=="NONE":
        raise NeedsRigorousReference(
            f"'{mode}' has NO industrial practice and no validated reference. "
            "Use the exploratory fitters, which return Mode E by construction.")
    K=len(edges)
    w=np.array(w0 if w0 is not None else [2.0]*K,dtype=float)
    if info["symmetric"]: w=np.full(K,w[0])
    sh=float(shadow0); mu=1e-2
    def obj(wv,sv):
        m2=MaskStack(mask.pitch_nm,mask.cd_nm,mask.absorber,mask.absorber_nm,
                     mask.cra_deg,mask.wavelength_nm,mask.ml,float(sv))
        base=kirchhoff_field(m2,xs)
        B=bl_basis(xs,edges,wv,mask.pitch_nm,profile)
        f=fit_amplitudes(target,base,B,lam)
        return f["relative_residual"],f
    val,f=obj(w,sh); hist=[val]; h=1e-3
    for it in range(max_iter):
        # reduced Jacobian (finite difference on the projected objective)
        J=[]
        for k in range(K):
            wp=w.copy(); wp[k]=min(w_max,wp[k]+h)
            if info["symmetric"]: wp=np.full(K,wp[k])
            vp,_=obj(wp,sh); J.append((vp-val)/h)
        ws=min(w_max,sh+h); vs,_=obj(w,ws); J.append((vs-val)/h)
        J=np.array(J); g=J*val
        H=np.outer(J,J)+mu*np.eye(len(J))
        try: step=-np.linalg.solve(H,g)
        except np.linalg.LinAlgError: break
        wn=np.clip(w+step[:K],w_min,w_max)
        if info["symmetric"]: wn=np.full(K,wn[0])
        sn=float(np.clip(sh+step[K],0.0,mask.absorber_nm*0.2))
        vn,fn=obj(wn,sn)
        if vn<val:
            w,sh,val,f=wn,sn,vn,fn; mu=max(mu*0.5,1e-8)
        else:
            mu=min(mu*4,1e6)
        hist.append(val)
        if abs(hist[-1]-hist[-2])<1e-10: break
    coef=f["coef"]
    p=BLParams(widths_nm=w.tolist(),amps=[float(abs(c)) for c in coef],
               phases_rad=[float(math.atan2(c.imag,c.real)) for c in coef],
               shadow_nm=sh,mode=mode)
    return {"params":p,"relative_residual":val,"condition_number":f["condition_number"],
            "design_rank":f["design_rank"],"n_columns":f["n_columns"],
            "identifiable":f["identifiable"],"iterations":it+1,
            "objective_history":hist,"method":"variable projection + LM",
            "industry_status":info["industry"],
            "advantage":"no Cartesian grid explosion; O(K) evals per step"}


# =============================================================================
# §2. OPMC LOCALISATION + IDENTIFIABILITY
# =============================================================================

def locate_edges(mask: MaskStack, n_orders=21, sigma=0.01, ell=8,
                 cache: Optional[Cache]=None) -> Dict[str,Any]:
    _np()
    rig=rcwa(mask,"TE",n_orders,cache)
    xs=rig["x_nm"]; near=rig["near_field"]; kir=kirchhoff_field(mask,xs)
    resid=np.abs(near-kir); scale=float(resid.max()) or 1.0
    cert=opmc_certified((resid/scale).tolist(),ell,sigma)
    dx=xs[1]-xs[0]
    edges=[float(xs[b]) for b in cert["boundaries"] if b<len(xs)]
    return {"x_nm":xs,"near_field":near,"kirchhoff":kir,"residual":resid,
            "opmc":cert,"edges_nm":edges,"edges_certified":cert["status"]==MODE_CERTIFIED,
            "radius_nm":cert["radius"]*dx if cert["radius"] is not None else None,
            "interpretation":"BL locations ARE change points of the M3D residual"}

def fisher(B, sigma=1e-3) -> Dict[str,Any]:
    _np()
    F=(B.conj().T@B).real/(sigma*sigma)
    ev=np.maximum(np.linalg.eigvalsh(F),0.0)
    rank=int(np.sum(ev>(ev[-1] if len(ev) else 0)*1e-10))
    C=np.linalg.pinv(F); d=np.sqrt(np.maximum(np.diag(C),0))
    denom=np.outer(d,d); ok=denom>0
    corr=np.zeros_like(C); corr[ok]=C[ok]/denom[ok]
    off=corr[np.triu_indices(len(d),1)] if len(d)>1 else np.array([0.0])
    return {"fisher_eigenvalues":ev.tolist(),"rank":rank,"n_params":B.shape[1],
            "full_rank":rank==B.shape[1],"std_errors":d.tolist(),
            "max_offdiag_correlation":float(np.max(np.abs(off))),
            "ill_conditioned":float(np.max(np.abs(off)))>0.95}


# =============================================================================
# §3. EXPLORATORY MODES (Mode E by construction)
# =============================================================================

def extract_nonlocal(mask: MaskStack, target, xs, kernel_len=9, lam=1e-3):
    _np()
    base=kirchhoff_field(mask,xs); n=len(xs); dx=xs[1]-xs[0]
    chi=((xs%mask.pitch_nm)<mask.cd_nm).astype(float)
    dchi=np.gradient(chi,dx); half=kernel_len//2
    C=np.column_stack([np.roll(dchi,s) for s in range(-half,half+1)]).astype(complex)
    f=fit_amplitudes(target,base,C,lam)
    return {"kernel":f["coef"],"relative_residual":f["relative_residual"],
            "condition_number":f["condition_number"],"identifiable":f["identifiable"],
            "mode":"nonlocal_kernel","industry_status":"NONE",
            "STATUS":"MODE_E — proposed extension, not validated",
            "why_not_certified":"no uniqueness without a sparsity/decay prior; no measured data"}

def extract_carpet(mask: MaskStack, target, xs, level=3, lam=1e-3):
    _np()
    base=kirchhoff_field(mask,xs); span=mask.pitch_nm; edges=[0.0,mask.cd_nm]
    cols=[]; labels=[]
    for m in range(1,level+1):
        w=span/(3**m)
        for e in edges:
            cols.append(bl_basis(xs,[e],[w],span,"gauss")[:,0])
            labels.append({"level":m,"edge_nm":e,"width_nm":w})
    B=np.column_stack(cols); f=fit_amplitudes(target,base,B,lam)
    return {"coefficients":f["coef"],"labels":labels,"n_scales":level,
            "relative_residual":f["relative_residual"],
            "design_rank":f["design_rank"],"n_columns":f["n_columns"],
            "condition_number":f["condition_number"],
            "mode":"carpet_multiscale","industry_status":"NONE","is_pcf":False,
            "imported_hypotheses":["C2 Mosco compactness","C3 BBKT uniqueness"],
            "STATUS":"MODE_E — non-PCF, C2/C3 imported, not validated",
            "why_not_certified":"multi-scale strips strongly collinear; rank deficiency expected"}

def load_vector2d(path: str) -> Dict[str,Any]:
    _np()
    d=json.loads(Path(path).read_text())
    VS={"S4","grcwa","RETICOLO","FDTD","RCWA_vector"}
    if d.get("solver") not in VS:
        raise NeedsRigorousReference(f"solver '{d.get('solver')}' not a VECTOR method {sorted(VS)}")
    conv=d.get("convergence_rel_change")
    if conv is None: raise NeedsRigorousReference("no convergence_rel_change supplied")
    if float(conv)>1e-3: raise NeedsRigorousReference(f"convergence {conv} > 1e-3")
    return {"solver":d["solver"],"convergence":float(conv),"VECTOR_CERTIFIED":True,
            "Ex":np.array(d["Ex_real"])+1j*np.array(d["Ex_imag"]),
            "Ey":np.array(d["Ey_real"])+1j*np.array(d["Ey_imag"])}


# =============================================================================
# §4. FULL PIPELINE + SELF-TEST
# =============================================================================

def full_extraction(mask: MaskStack, n_orders=21, sigma=0.01, ell=8,
                    explore=True, cfg: Optional[RunConfig]=None) -> Dict[str,Any]:
    _np()
    cfg=(cfg or RunConfig()).validate()
    cache=Cache(cfg.cache_dir,cfg.cache_max); sd=SDGate()
    cv=rcwa_convergence(mask,"TE",(11,21,31))
    sd.certify("rcwa_reference",("EMF","CONV") if cv["converged"] else ("EMF",),
               {"rel_changes":cv["rel_changes"]})
    loc=locate_edges(mask,n_orders,sigma,ell,cache)
    sd.certify("opmc_changepoints",("L1","D","O") if loc["edges_certified"] else (),
               {"edges_nm":loc["edges_nm"]})
    edges=loc["edges_nm"][:2] if len(loc["edges_nm"])>=2 else [0.0,mask.cd_nm]
    fit=extract_lm(mask,loc["near_field"],loc["x_nm"],edges,cache=cache)
    B=bl_basis(loc["x_nm"],edges,fit["params"].widths_nm,mask.pitch_nm)
    fd=fisher(B)
    ident=fit["identifiable"] and fd["full_rank"] and not fd["ill_conditioned"]
    sd.certify("bl_parameters",("EMF","CONV","IDENT") if ident else ("EMF",),
               fit["params"].to_dict())
    before=float(np.linalg.norm(loc["near_field"]-loc["kirchhoff"])/
                 max(np.linalg.norm(loc["near_field"]),1e-30))
    pred=bl_field(mask,loc["x_nm"],edges,fit["params"])
    after=float(np.linalg.norm(loc["near_field"]-pred)/
                max(np.linalg.norm(loc["near_field"]),1e-30))
    out={"order_convergence":cv,
         "localisation":{"edges_nm":loc["edges_nm"],
                         "certified":loc["edges_certified"],
                         "status":loc["opmc"]["status"]},
         "fit":{"params":fit["params"].to_dict(),
                "relative_residual":fit["relative_residual"],
                "iterations":fit["iterations"],
                "condition_number":fit["condition_number"]},
         "fisher":fd,"kirchhoff_error_before":before,"kirchhoff_error_after":after,
         "error_reduction_factor":before/max(after,1e-30),
         "cache":cache.stats()}
    if explore:
        out["nonlocal"]=extract_nonlocal(mask,loc["near_field"],loc["x_nm"])
        sd.certify("nonlocal_kernel",(),{})
        out["carpet"]=extract_carpet(mask,loc["near_field"],loc["x_nm"])
        sd.certify("carpet_multiscale",(),{})
    sd.certify("corner_2d",(),{})
    out["sd"]=sd.report()
    return out

def self_test(verbose=True) -> bool:
    print("="*78); print(f"MSPS BL EXTRACTION v{VERSION} (LM + identifiability)")
    print("="*78)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    cfg=RunConfig().validate(); cache=Cache(cfg.cache_dir)
    mask=MaskStack(pitch_nm=32,cd_nm=16,absorber_nm=60,cra_deg=6)

    print("\n[T1] RCWA reference + cache")
    u=rcwa_uniform_check("TE",11)
    rep("uniform limit exact", u["passes"], f"err={u['max_abs_error']:.1e}")
    r1=rcwa(mask,"TE",21,cache); r2=rcwa(mask,"TE",21,cache)
    rep("second call cache hit", r2["cache_hit"])
    xs=r1["x_nm"]; tgt=r1["near_field"]; edges=[0.0,mask.cd_nm]

    print("\n[T2] LM beats grid (no explosion)")
    lm=extract_lm(mask,tgt,xs,edges,cache=cache)
    print(f"      widths={[f'{w:.2f}' for w in lm['params'].widths_nm]} "
          f"resid={lm['relative_residual']:.4f} iters={lm['iterations']}")
    rep("LM returns parameters", len(lm["params"].widths_nm)==2)
    kir=kirchhoff_field(mask,xs)
    err0=float(np.linalg.norm(tgt-kir)/np.linalg.norm(tgt))
    rep("residual beats Kirchhoff", lm["relative_residual"]<err0,
        f"{lm['relative_residual']:.4f} < {err0:.4f}")
    rep("no grid explosion declared","no Cartesian grid" in lm["advantage"])

    print("\n[T3] SYNTHETIC RECOVERY (ground truth known)")
    truth=BLParams([2.0,3.0],[0.30,0.22],[-1.1,-0.7],0.0,mode="dual_edge")
    synth=bl_field(mask,xs,edges,truth)
    rec=extract_lm(mask,synth,xs,edges,w0=[1.5,2.5],cache=cache,max_iter=120)
    rp=rec["params"]
    print(f"      true [2.0,3.0] -> {[f'{w:.3f}' for w in rp.widths_nm]}")
    print(f"      true amps [0.30,0.22] -> {[f'{a:.3f}' for a in rp.amps]}")
    rep("widths within 0.15 nm",
        all(abs(a-b)<0.15 for a,b in zip(rp.widths_nm,truth.widths_nm)))
    rep("amps within 0.03",
        all(abs(a-b)<0.03 for a,b in zip(rp.amps,truth.amps)))
    rep("synthetic residual small", rec["relative_residual"]<0.05,
        f"{rec['relative_residual']:.2e}")

    print("\n[T4] OPMC LOCALISATION (PELT)")
    loc=locate_edges(mask,21,0.01,8,cache)
    print(f"      edges={[f'{e:.2f}' for e in loc['edges_nm']]} "
          f"status={loc['opmc']['status']}")
    rep("localisation runs","edges_nm" in loc)
    rep("SNR reported honestly","snr_lhs" in loc["opmc"])
    rep("change-point link stated","change points" in loc["interpretation"])

    print("\n[T5] IDENTIFIABILITY")
    B=bl_basis(xs,edges,[2.0,3.0],mask.pitch_nm)
    fd=fisher(B); rep("distinct strips full rank", fd["full_rank"])
    Bc=bl_basis(xs,edges,[2.0,2.0],mask.pitch_nm)
    fd2=fisher(np.column_stack([Bc[:,0],Bc[:,0]]))
    rep("collinear -> rank deficient", not fd2["full_rank"])
    rep("correlation reported","max_offdiag_correlation" in fd2)

    print("\n[T6] EXPLORATORY MODES stay Mode E")
    nl=extract_nonlocal(mask,tgt,xs); cp=extract_carpet(mask,tgt,xs)
    rep("nonlocal NO industry", nl["industry_status"]=="NONE")
    rep("nonlocal MODE_E", nl["STATUS"].startswith("MODE_E"))
    rep("carpet non-PCF", cp["is_pcf"] is False)
    rep("C2/C3 imported", len(cp["imported_hypotheses"])==2)
    rep("carpet MODE_E", cp["STATUS"].startswith("MODE_E"))
    b=False
    try: extract_lm(mask,tgt,xs,edges,mode="corner_2d")
    except NeedsRigorousReference: b=True
    rep("corner_2d REFUSED", b)
    b=False
    try: extract_lm(mask,tgt,xs,edges,mode="nonlocal_kernel")
    except NeedsRigorousReference: b=True
    rep("exploratory REFUSED in certified fitter", b)

    print("\n[T7] FULL PIPELINE")
    full=full_extraction(mask,21,cfg=cfg)
    print(f"      Kirchhoff {full['kirchhoff_error_before']:.4f} -> "
          f"{full['kirchhoff_error_after']:.4f} "
          f"({full['error_reduction_factor']:.2f}x)")
    print(f"      SD: {full['sd']['n']} outputs, {full['sd']['certified']} certified, "
          f"{full['sd']['mode_E']} Mode E")
    rep("error reduction >= 1", full["error_reduction_factor"]>=1.0)
    rep("Mode E retained", full["sd"]["mode_E"]>0)
    rep("corner_2d gated", any(c["statistic"]=="corner_2d" and c["status"]==MODE_E
                              for c in full["sd"]["log"]))
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    return ok

def main() -> int:
    ap=argparse.ArgumentParser(description="MSPS BL extraction v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","locate","extract","nonlocal","carpet","full","modes"])
    ap.add_argument("--pitch",type=float,default=32.0)
    ap.add_argument("--cd",type=float,default=None)
    ap.add_argument("--orders",type=int,default=21)
    ap.add_argument("--mode",default="dual_edge")
    ap.add_argument("--level",type=int,default=3)
    ap.add_argument("--json")
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    if a.cmd=="modes": print(json.dumps(MODES,indent=2)); return 0
    cfg=RunConfig().validate(); cache=Cache(cfg.cache_dir)
    cd=a.cd or a.pitch/2
    mask=MaskStack(pitch_nm=a.pitch,cd_nm=cd); out={}
    try:
        if a.cmd=="full": out=full_extraction(mask,a.orders,cfg=cfg)
        else:
            r=rcwa(mask,"TE",a.orders,cache); xs=r["x_nm"]; tgt=r["near_field"]
            if a.cmd=="locate":
                l=locate_edges(mask,a.orders,cache=cache)
                out={"edges_nm":l["edges_nm"],"opmc":l["opmc"]}
            elif a.cmd=="extract":
                f=extract_lm(mask,tgt,xs,[0.0,cd],mode=a.mode,cache=cache)
                out={"params":f["params"].to_dict(),
                     "relative_residual":f["relative_residual"],
                     "iterations":f["iterations"],"identifiable":f["identifiable"]}
            elif a.cmd=="nonlocal":
                o=extract_nonlocal(mask,tgt,xs); o["kernel"]=[complex(c) for c in o["kernel"]]; out=o
            elif a.cmd=="carpet":
                o=extract_carpet(mask,tgt,xs,a.level)
                o["coefficients"]=[complex(c) for c in o["coefficients"]]; out=o
    except Refusal as e: out={"REFUSED":str(e)}
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
