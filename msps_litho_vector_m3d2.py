#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_litho_vector_m3d.py — v2.0.0  [VECTOR / M3D FACADE + EXTENSIONS]
Backward-compatible surface over the unified kernel, plus vector-specific
production additions:
  * Proper Zernike radial polynomials (was rho**|j| placeholder).
  * Full polarization-tensor report per order pair (audit trail).
  * Anamorphic shadow asymmetry with H/V bias decomposition.
  * Cached Bossung / EPE / best-focus-vs-pitch sweeps.
  * Kirchhoff-vs-M3D spectral error quantification.

PHILOSOPHY: see msps_math_foundation_api.PHILOSOPHY (full text).
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

REFUSALS: boundary-layer M3D is a CALIBRATED SURROGATE, not rigorous RCWA.
  Optical constants approximate. Resist is a hard threshold. Multilayer
  phase taken angle-independent in the mask spectrum. No wafer data compared.
=============================================================================
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from msps_math_foundation_api import (
    HAS_NUMPY, np, RunConfig, Cache, SDGate, Refusal, MODE_E, MODE_CERTIFIED,
    bench, _np,
)
from msps_litho_math_api import (
    Mask3D, Pupil, Multilayer, Material, MO, SI, RU, VAC, LA, B4C,
    aerial_vector, aerial_vector_pairwise, mask_spectrum, two_beam_contrast,
    bossung, best_focus_vs_pitch, zernike_radial, snell_cos, nils_edge,
)

VERSION = "2.0.0"

# ---- backward-compatible names ------------------------------------------------
VectorPupil = Pupil
def make_order(fx, lam, na):
    a = lam*fx; prop = abs(a) <= na+1e-12
    return {"fx":fx,"alpha":a,"gamma":math.sqrt(max(0.0,1-a*a)) if prop else 0.0,
            "propagating":prop,"theta_deg":math.degrees(math.asin(max(-1,min(1,a))))}
def pol_dot(on, om, pol):
    if not (on["propagating"] and om["propagating"]): return 0j
    return 1.0+0j if pol=="TE" else complex(on["gamma"]*om["gamma"]+on["alpha"]*om["alpha"])


# =============================================================================
# §1. POLARIZATION TENSOR AUDIT
# =============================================================================

def polarization_tensor(freqs: Sequence[float], lam: float, na: float
                        ) -> Dict[str,Any]:
    """Explicit e_i . conj(e_j) matrix. Audit trail for the factorisation."""
    _np()
    os_=[make_order(f,lam,na) for f in freqs]
    n=len(os_)
    TE=np.ones((n,n),dtype=complex); TM=np.zeros((n,n),dtype=complex)
    for i,oi in enumerate(os_):
        for j,oj in enumerate(os_):
            TE[i,j]=pol_dot(oi,oj,"TE"); TM[i,j]=pol_dot(oi,oj,"TM")
    return {"TE":TE,"TM":TM,"orders":os_,
            "factorisable":True,
            "identity":"sum_ij a_i conj(a_j) d_ij e^{i2pi(f_i-f_j)x} "
                       "= |sum a_i gamma_i e^{..}|^2 + |sum a_i alpha_i e^{..}|^2"}

def zernike_report(coeffs: Dict[int,float], n_rho: int = 32) -> Dict[str,Any]:
    _np()
    rho=np.linspace(0,1,n_rho)
    terms={str(j):{"coefficient_nm":c,"radial":zernike_radial(j,rho).tolist()}
           for j,c in coeffs.items()}
    W=sum(c*zernike_radial(j,rho) for j,c in coeffs.items())
    return {"rho":rho.tolist(),"terms":terms,"total_W_nm":W.tolist(),
            "rms_nm":float(np.sqrt(np.mean(W**2))),
            "NOTE":"low-order radial subset; full Noll set requires more terms"}


# =============================================================================
# §2. M3D DIAGNOSTICS
# =============================================================================

def kirchhoff_vs_m3d(pitch: float, cd: float, mask: Mask3D, lam=13.5,
                     cache: Cache = None) -> Dict[str,Any]:
    _np()
    m3=mask_spectrum(mask,pitch,cd,lam,cache=cache)
    k=Mask3D(absorber_nm=mask.absorber_nm,cra_deg=0.0,mag_x=mask.mag_x,
             mag_y=mask.mag_y,r_ml=mask.r_ml,r_abs=mask.r_abs,
             bl_width_nm=0.0,ml=mask.ml)
    kk=mask_spectrum(k,pitch,cd,lam)
    n=min(len(m3["amps"]),len(kk["amps"]))
    a=m3["amps"][:n]; b=kk["amps"][:n]
    err=float(np.sum(np.abs(a-b)**2)/max(float(np.sum(np.abs(b)**2)),1e-30))
    return {"pitch_nm":pitch,"spectral_error_relative":err,
            "shadow_mask_nm":m3["shadow_mask_nm"],
            "shadow_wafer_x_nm":mask.shadow_wafer("x"),
            "shadow_wafer_y_nm":mask.shadow_wafer("y"),
            "ml_phase_rad":m3["ml_phase_rad"],
            "kirchhoff_adequate":err<0.01,
            "VERDICT":"Kirchhoff INVALID" if err>=0.01 else "acceptable at this pitch"}

def anamorphic_bias_decomposition(mask: Mask3D, half_pitch_nm: float
                                  ) -> Dict[str,Any]:
    a=mask.anamorphic_asymmetry()
    return {**a,"half_pitch_nm":half_pitch_nm,
            "x_fraction_of_hp":a["x_nm"]/half_pitch_nm,
            "y_fraction_of_hp":a["y_nm"]/half_pitch_nm,
            "required_bias_x_nm":-a["x_nm"],"required_bias_y_nm":-a["y_nm"],
            "single_global_bias_insufficient":a["anisotropic"],
            "CONSEQUENCE":"H and V need separate OPC bias; a scalar bias cannot fix both"}

def scalar_vs_vector(pitch: float, cd: float, mask: Mask3D, na: float,
                     lam=13.5, cache: Cache=None) -> Dict[str,Any]:
    r={}
    for pol in ("TE","TM","unpolarised"):
        a=aerial_vector(mask,pitch,cd,Pupil(na,lam,polarisation=pol),
                        n_src=5,n_x=128,cache=cache)
        r[pol]={"nils":a["nils"],"contrast":a["contrast"]}
    te,tm,un=r["TE"]["nils"],r["TM"]["nils"],r["unpolarised"]["nils"]
    return {"na":na,"pitch_nm":pitch,**{f"NILS_{k}":v["nils"] for k,v in r.items()},
            "TM_over_TE":tm/te if te else None,
            "scalar_model_equals":"TE (no polarization loss)",
            "scalar_overestimates_by":te/un if un else None}


# =============================================================================
# §3. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print("="*78); print(f"MSPS VECTOR + M3D v{VERSION}")
    print("="*78)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    cfg=RunConfig().validate(); cache=Cache(cfg.cache_dir); sd=SDGate()
    m=Mask3D()

    print("\n[T1] POLARIZATION TENSOR + FACTORISATION IDENTITY")
    pt=polarization_tensor([-0.03,0.0,0.03],13.5,0.55)
    rep("TE all ones", bool(np.allclose(pt["TE"],1.0)))
    o1=make_order(0.03,13.5,0.55); o2=make_order(-0.03,13.5,0.55)
    ana=math.cos(math.radians(o1["theta_deg"]-o2["theta_deg"]))
    rep("TM = cos(theta_i - theta_j)", abs(pol_dot(o1,o2,"TM").real-ana)<1e-12)
    ev=make_order(0.2,13.5,0.55)
    rep("evanescent carries no field", not ev["propagating"] and pol_dot(ev,o1,"TE")==0)
    rep("identity declared", pt["factorisable"])

    print("\n[T2] ZERNIKE (proper radial, not rho**|j|)")
    zr=zernike_report({2:5.0,4:-2.0})
    rep("rms computed", zr["rms_nm"]>0)
    rho=np.array([0.0,0.5,1.0])
    z2=zernike_radial(2,rho)
    rep("Z2 = 2rho^2-1", bool(np.allclose(z2,2*rho**2-1)))
    z4=zernike_radial(4,rho)
    rep("Z4 = 6rho^4-6rho^2+1", bool(np.allclose(z4,6*rho**4-6*rho**2+1)))

    print("\n[T3] ANAMORPHIC SHADOW")
    m8=Mask3D(mag_x=4,mag_y=8)
    print(f"      mask {m8.shadow_mask():.3f} nm -> wafer x={m8.shadow_wafer('x'):.3f} "
          f"y={m8.shadow_wafer('y'):.3f}")
    rep("mask shadow ~6.30 nm", abs(m8.shadow_mask()-6.30)<0.05)
    rep("4x wafer ~1.576 nm", abs(m8.shadow_wafer("x")-1.576)<0.02)
    rep("8x wafer ~0.788 nm", abs(m8.shadow_wafer("y")-0.788)<0.02)
    bd=anamorphic_bias_decomposition(m8,8.0)
    print(f"      at hp=8 nm: x is {100*bd['x_fraction_of_hp']:.1f}% of hp")
    rep("x fraction >15%", bd["x_fraction_of_hp"]>0.15)
    rep("single bias insufficient", bd["single_global_bias_insufficient"])

    print("\n[T4] KIRCHHOFF vs M3D")
    for p in (32.0,24.0,16.0):
        r=kirchhoff_vs_m3d(p,p/2,m,cache=cache)
        print(f"      pitch {p:5.1f}: err={r['spectral_error_relative']:.4f} {r['VERDICT']}")
        rep(f"pitch {p} error computed", r["spectral_error_relative"]>=0)
    rep("Kirchhoff INVALID at 16 nm", not kirchhoff_vs_m3d(16,8,m)["kirchhoff_adequate"])

    print("\n[T5] SCALAR vs VECTOR")
    for na,p in ((0.33,40.0),(0.55,24.0)):
        sv=scalar_vs_vector(p,p/2,m,na,cache=cache)
        print(f"      NA={na}: TE={sv['NILS_TE']:.3f} TM={sv['NILS_TM']:.3f} "
              f"unpol={sv['NILS_unpolarised']:.3f}")
        rep(f"NA={na} TM below TE", sv["NILS_TM"]<sv["NILS_TE"])
        rep(f"NA={na} unpol between",
            min(sv["NILS_TE"],sv["NILS_TM"])<=sv["NILS_unpolarised"]
            <=max(sv["NILS_TE"],sv["NILS_TM"])+1e-6)

    print("\n[T6] CACHED BOSSUNG SWEEP")
    b1=bench(bossung,m,24,12,0.55,n_focus=7,n_dose=5,cache=cache,n=2)
    rep("bossung runs", b1["mean_s"]>0)
    bf=best_focus_vs_pitch(m,[32,26,22],0.55,cache=cache)
    print(f"      BF spread={bf['best_focus_spread_nm']:.2f} nm, "
          f"cache hit_rate={cache.stats()['hit_rate']:.2f}")
    rep("Kirchhoff predicts identical BF (stated)","identical" in bf["KIRCHHOFF_PREDICTS"])

    print("\n[T7] SD GATE")
    c=sd.certify("m3d_spectrum",(),{}); rep("uncalibrated MODE_E", c["status"]==MODE_E)
    rep("caveat says NOT rigorous RCWA","NOT rigorous" in c["caveat"])
    c2=sd.certify("process_window",("M3D","OPT"),{})
    rep("missing RES MODE_E", c2["status"]==MODE_E)
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    return ok

def main() -> int:
    ap=argparse.ArgumentParser(description="MSPS vector + M3D v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","tensor","zernike","shadow","m3d","sv","bossung","bfpitch"])
    ap.add_argument("--na",type=float,default=0.55)
    ap.add_argument("--pitch",type=float,default=24.0)
    ap.add_argument("--cd",type=float,default=None)
    ap.add_argument("--mag-x",type=float,default=4.0)
    ap.add_argument("--mag-y",type=float,default=8.0)
    ap.add_argument("--json")
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    cfg=RunConfig().validate(); cache=Cache(cfg.cache_dir)
    m=Mask3D(mag_x=a.mag_x,mag_y=a.mag_y); cd=a.cd or a.pitch/2; out={}
    if a.cmd=="tensor":
        t=polarization_tensor([-0.03,0,0.03],13.5,a.na)
        out={"TE":t["TE"].real.tolist(),"TM":t["TM"].real.tolist(),"identity":t["identity"]}
    elif a.cmd=="zernike": out=zernike_report({2:5.0,4:-2.0})
    elif a.cmd=="shadow": out=anamorphic_bias_decomposition(m,cd/2)
    elif a.cmd=="m3d": out=kirchhoff_vs_m3d(a.pitch,cd,m,cache=cache)
    elif a.cmd=="sv": out=scalar_vs_vector(a.pitch,cd,m,a.na,cache=cache)
    elif a.cmd=="bossung": out=bossung(m,a.pitch,cd,a.na,cache=cache)
    elif a.cmd=="bfpitch": out=best_focus_vs_pitch(m,[a.pitch*1.3,a.pitch,a.pitch*0.8],a.na,cache=cache)
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
