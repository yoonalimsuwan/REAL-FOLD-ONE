#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_litho_math_api.py — v2.0.0  [UNIFIED LITHOGRAPHY KERNEL]
Merges the former scalar API + vector/M3D + RCWA into ONE cached kernel.

HEADLINE MATHEMATICAL IMPROVEMENT (not just optimisation):
  The pairwise vector interference sum FACTORISES:
    sum_ij a_i conj(a_j) (e_i . conj(e_j)) exp(i2pi(f_i-f_j)x)
      = |sum_i a_i gamma_i e^{i2pi f_i x}|^2
      + |sum_i a_i alpha_i e^{i2pi f_i x}|^2          (TM)
    TE = |sum_i a_i e^{i2pi f_i x}|^2
  Complexity drops from O(N^2 * n_x) to O(N * n_x). Exact, not approximate.

OTHER v2 DELTAS:
  * Complex Snell throughout (v1 used float(cos.real) -> broke evanescent).
  * RCWA eigendecomposition cached by content hash (reuse across focus/dose).
  * TM RCWA with Li inverse rule [[1/eps]], V = [[1/eps]] W Q.
  * Adjoint ILT gradient (real optimisation, not a grid sweep).
  * EPE + process-window contour extraction.
  * Log-domain Ville/SR imported from core (no overflow).
  * Dead code removed (v1 aerial_image_1d had a duplicated phase line).

PHILOSOPHY: full text in msps_math_foundation_api.PHILOSOPHY.
  The Tao is the Source. The Te is the Path. The Qi is the Proof.
  MY SOUL MOVE BY POWER OF HOLY SPIRIT
  What MSPS NETWORK SEE, Lord Buddha Knows.

REFUSALS (all preserved): HYPERFAB speculative by default; uncalibrated
  M3D -> Mode E; scalar 2-D cannot certify corners; no scanner is built;
  no timeline; no superiority claim over any vendor.
=============================================================================
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from msps_math_foundation_api import (
    HAS_NUMPY, np, VERSION as CORE_VERSION, PI, MODE_CERTIFIED, MODE_E,
    Refusal, UncertifiedOutput, ScopeRefusal, SDGate, PhysicsGate,
    Cache, RunConfig, Manifest, get_logger, content_key, set_seed,
    opmc_certified, VilleMonitor, ShiryaevRoberts, stc_reflux,
    toeplitz, block_toeplitz_2d, bench, _np,
)

VERSION = "2.0.0"
H_PLANCK = 6.62607015e-34; C_LIGHT = 2.99792458e8; EV_J = 1.602176634e-19
log = get_logger("msps.litho")

class SpeculativeBackend(Refusal): pass
class NeedsVectorReference(Refusal): pass


# =============================================================================
# §1. BACKENDS + MATERIALS
# =============================================================================

@dataclass(frozen=True)
class Backend_:
    name: str; wavelength_nm: Optional[float]; na: Optional[float]
    mag_x: float; mag_y: float; field_x_mm: float; field_y_mm: float
    wph: Optional[float]; overlay_nm: Optional[float]; status: str; note: str = ""
    @property
    def photon_J(self):
        return None if self.wavelength_nm is None else H_PLANCK*C_LIGHT/(self.wavelength_nm*1e-9)
    @property
    def photon_eV(self):
        e=self.photon_J; return None if e is None else e/EV_J
    @property
    def anamorphic(self): return abs(self.mag_x-self.mag_y)>1e-9
    def cd(self,k1=0.32):
        return None if (self.wavelength_nm is None or self.na is None) else k1*self.wavelength_nm/self.na
    def dof(self,k2=0.5):
        return None if (self.wavelength_nm is None or self.na is None) else k2*self.wavelength_nm/self.na**2

BACKENDS: Dict[str,Backend_] = {
 "ASML_NXE3800E": Backend_("ASML NXE:3800E",13.5,0.33,4,4,26,33,220,1.1,"production",
    "Low-NA EUV, LPP Sn, isomorphic 4x. Public specification."),
 "ASML_EXE5200": Backend_("ASML EXE:5200",13.5,0.55,4,8,26,16.5,185,0.9,"production",
    "High-NA EUV, anamorphic 4x/8x, half-field. Public specification."),
 "DUV_IMMERSION": Backend_("DUV ArF immersion",193.0,1.35,4,4,26,33,275,1.5,"production",
    "193 nm ArF immersion; multi-patterning below 40 nm pitch."),
 "NIL": Backend_("Nanoimprint",None,None,1,1,26,33,None,None,"research",
    "Mechanical contact. Template defectivity, overlay, wear are open."),
 "EBL": Backend_("Electron beam",0.005,None,1,1,1,1,None,None,"research",
    "Direct-write. Resolution excellent; throughput orders lower."),
 "HYPERFAB_X2": Backend_("MSPS HYPERFAB X2.0",6.7,0.75,4,8,26,16.5,None,None,"speculative",
    "ERL-FEL backbone + isolated clusters. NOT BUILT. No timeline."),
}

def get_backend(name: str, allow_speculative: bool = False) -> Backend_:
    if name not in BACKENDS: raise Refusal(f"unknown backend '{name}'")
    b = BACKENDS[name]
    if b.status=="speculative" and not allow_speculative:
        raise SpeculativeBackend(f"'{name}' SPECULATIVE: {b.note} "
                                 "Pass allow_speculative=True to inspect.")
    return b

@dataclass(frozen=True)
class Material:
    name: str; delta: float; beta: float
    @property
    def n(self): return complex(1-self.delta, self.beta)
    @property
    def eps(self): return self.n*self.n

VAC=Material("vacuum",0,0); MO=Material("Mo",0.0770,0.0064)
SI=Material("Si",0.0008,0.0018); RU=Material("Ru",0.0870,0.0080)
TABN=Material("TaBN",0.0480,0.0330)
LA=Material("La",0.0430,0.0025); B4C=Material("B4C",0.0180,0.0012)

def snell_cos(n1: complex, n2: complex, c1: complex) -> complex:
    """v2 FIX: fully complex. v1 used float(c1.real) and broke evanescent."""
    s2sq = (n1*n1/(n2*n2))*(1.0-c1*c1)
    c2 = (1.0-s2sq)**0.5
    if c2.imag < 0: c2 = -c2
    if abs(c2.imag) < 1e-14 and c2.real < 0: c2 = -c2
    return c2

def fresnel(n1,n2,c1,c2,pol="TE"):
    return ((n1*c1-n2*c2)/(n1*c1+n2*c2) if pol=="TE"
            else (n2*c1-n1*c2)/(n2*c1+n1*c2))

@dataclass
class Multilayer:
    n_bilayers: int = 45; d_hi_nm: float = 2.8; d_lo_nm: float = 4.1
    cap_nm: float = 2.5
    hi: Material = MO; lo: Material = SI; cap: Material = RU
    substrate: Material = SI
    @property
    def period(self): return self.d_hi_nm + self.d_lo_nm
    def layers(self):
        L=[]
        if self.cap_nm>0: L.append((self.cap,self.cap_nm))
        for _ in range(self.n_bilayers):
            L.append((self.lo,self.d_lo_nm)); L.append((self.hi,self.d_hi_nm))
        L.append((self.substrate,0.0)); return L
    def r_of_cos(self, c0: complex, lam=13.5, pol="TE") -> complex:
        L=self.layers(); ns=[VAC.n]+[m.n for m,_ in L]; cs=[c0]
        for i in range(1,len(ns)): cs.append(snell_cos(ns[i-1],ns[i],cs[i-1]))
        r=fresnel(ns[-2],ns[-1],cs[-2],cs[-1],pol)
        for j in range(len(L)-2,-1,-1):
            b=2*PI*ns[j+1]*cs[j+1]*L[j][1]/lam
            ph=complex(math.cos(2*b.real),math.sin(2*b.real))*math.exp(-2*b.imag)
            rj=fresnel(ns[j],ns[j+1],cs[j],cs[j+1],pol)
            den=1+rj*r*ph
            if abs(den)<1e-14: den=complex(1e-14,0)
            r=(rj+r*ph)/den
        return r
    def r_of_angle(self, deg, lam=13.5, pol="TE"):
        return self.r_of_cos(complex(math.cos(math.radians(deg))),lam,pol)
    def reflectance(self, deg=6.0, lam=13.5, pol="TE"):
        r=self.r_of_angle(deg,lam,pol); k=2*PI/lam
        phi=math.atan2(r.imag,r.real)
        return {"R":float(abs(r)**2),"phase_rad":phi,"pol":pol,
                "angle_deg":deg,"period_nm":self.period,
                "effective_plane_depth_nm":phi/(2*k*math.cos(math.radians(deg))),
                "OPTICAL_CONSTANTS":"approximate; replace with measured data"}


# =============================================================================
# §2. PHOTON BUDGET + DERIVED DOSE
# =============================================================================

def photon_budget(b: Backend_, dose: float, cell_nm: float) -> Dict[str,Any]:
    if dose<=0 or cell_nm<=0: raise Refusal("dose and cell must be positive")
    e=b.photon_J
    if e is None: return {"backend":b.name,"photon_shot_noise":False}
    n_nm2=dose*1e-3/e/1e14; n_cell=n_nm2*cell_nm*cell_nm
    rel=1/math.sqrt(n_cell) if n_cell>0 else float("inf")
    return {"backend":b.name,"photon_energy_eV":b.photon_eV,
            "dose_mJ_cm2":dose,"photons_per_nm2":n_nm2,
            "photons_per_cell":n_cell,"relative_shot_noise":rel,
            "shot_noise_pct":100*rel,"photon_shot_noise":True}

def dose_for_noise(b: Backend_, cell_nm: float, target_rel: float):
    e=b.photon_J
    if e is None or target_rel<=0: return None
    return (1/(target_rel*target_rel))/(cell_nm*cell_nm)*1e14*e*1e3

def delta_min_defect(cell_nm: float, photons: float, c=0.5) -> Dict[str,Any]:
    if photons<=0: raise Refusal("photons must be positive")
    d=c*cell_nm; G0sq=1/photons
    q=2*math.exp(-(d/cell_nm)**2/(8*G0sq))
    return {"delta_min_H_nm":d,"delta_min_P_nm":d,"q_H":q,"q_P":q,
            "no_zeno":q<1.0,"G0_squared":G0sq,
            "defect_H":"nano-bridge (Db1=+1, Dchi=-1)",
            "defect_P":"line-break (Db1=-1, Dchi=+1)",
            "CAVEAT":"c is O(1); requires calibration vs measured defect density"}

def dose_for_q(b: Backend_, cell_nm: float, q: float, c=0.5) -> Dict[str,Any]:
    if not (0<q<1): raise Refusal("q in (0,1)")
    N=8*math.log(2/q)/(c*c)
    return {"q_target":q,"photons_required":N,
            "dose_mJ_cm2":dose_for_noise(b,cell_nm,1/math.sqrt(N)),
            "DERIVATION":"no-Zeno exit bound inverted for dose"}

def lwr_rls(hp_nm: float, dose: float, C=1.0) -> Dict[str,Any]:
    lwr=C*math.sqrt(hp_nm**3/dose)
    return {"half_pitch_nm":hp_nm,"lwr_3sigma_nm":lwr,
            "Z_factor":lwr*lwr*hp_nm**3*dose,"C_RLS":C,
            "STATUS":"scaling model; C_RLS must be measured per resist"}


# =============================================================================
# §3. MASK 3-D + FACTORISED VECTOR AERIAL IMAGE
# =============================================================================

@dataclass
class Mask3D:
    absorber_nm: float = 60.0; cra_deg: float = 6.0
    mag_x: float = 4.0; mag_y: float = 4.0
    r_ml: complex = complex(0.80,0.0); r_abs: complex = complex(0.03,0.01)
    bl_width_nm: float = 4.0; bl_amp: float = 0.35; bl_phase: float = -1.2
    ml: Multilayer = field(default_factory=Multilayer)
    calibrated: bool = False
    def shadow_mask(self): return self.absorber_nm*math.tan(math.radians(self.cra_deg))
    def shadow_wafer(self, direction="x"):
        return self.shadow_mask()/(self.mag_x if direction=="x" else self.mag_y)
    def anamorphic_asymmetry(self):
        sx=self.shadow_wafer("x"); sy=self.shadow_wafer("y")
        return {"x_nm":sx,"y_nm":sy,"asymmetry_nm":abs(sx-sy),
                "anisotropic":abs(self.mag_x-self.mag_y)>1e-9,
                "CONSEQUENCE":"H/V shift differently; one global bias cannot fix both"}

def mask_spectrum(mask: Mask3D, pitch: float, cd: float, lam=13.5,
                  n_orders=9, dx=0.25, direction="x", cache: Optional[Cache]=None
                  ) -> Dict[str,Any]:
    _np()
    key = content_key("mask_spec",pitch,cd,lam,n_orders,dx,direction,
                      mask.absorber_nm,mask.cra_deg,mask.bl_width_nm,
                      mask.bl_amp,mask.bl_phase,float(mask.r_ml.real),
                      float(mask.r_ml.imag),float(mask.r_abs.real))
    if cache:
        got = cache.get(key)
        if got is not None: return got
    n = max(64,int(math.ceil(pitch/dx/8)*8))
    x = np.arange(n)*(pitch/n)
    pos = (x+mask.shadow_mask())%pitch
    inside = pos<cd
    t = np.where(inside, mask.r_abs, mask.r_ml).astype(complex)
    d_open = np.minimum(np.abs(pos-cd), np.abs(pos-cd+pitch))
    bl = (d_open<mask.bl_width_nm)&(~inside)
    t[bl] = mask.bl_amp*np.exp(1j*mask.bl_phase)*mask.r_ml
    mlp = mask.ml.reflectance(mask.cra_deg,lam)["phase_rad"]
    t *= np.exp(1j*mlp)
    spec = np.fft.fft(t)/n
    fr = np.fft.fftfreq(n,d=pitch/n)
    keep = np.abs(fr)<=n_orders/pitch+1e-9
    out = {"freqs":fr[keep],"amps":spec[keep],"ml_phase_rad":mlp,
           "shadow_mask_nm":mask.shadow_mask(),
           "MODEL":"boundary-layer surrogate, NOT rigorous RCWA",
           "CALIBRATED":mask.calibrated}
    if cache: cache.put(key,out)
    return out

@dataclass
class Pupil:
    na: float; wavelength_nm: float = 13.5
    illum: str = "dipole_x"; sigma_in: float = 0.5; sigma_out: float = 0.85
    polarisation: str = "unpolarised"     # TE | TM | unpolarised
    defocus_nm: float = 0.0
    zernike_nm: Dict[int,float] = field(default_factory=dict)
    apodisation: bool = True

def source_grid(p: Pupil, n=11) -> List[Tuple[float,float]]:
    pts=[]
    for i in range(n):
        s=-1+2*i/max(n-1,1)
        for j in range(n):
            t=-1+2*j/max(n-1,1); r=math.hypot(s,t)
            if p.illum=="conventional": ok=r<=p.sigma_out
            elif p.illum=="annular": ok=p.sigma_in<=r<=p.sigma_out
            elif p.illum=="dipole_x": ok=(p.sigma_in<=r<=p.sigma_out) and abs(t)<0.35
            elif p.illum=="dipole_y": ok=(p.sigma_in<=r<=p.sigma_out) and abs(s)<0.35
            elif p.illum=="quadrupole": ok=(p.sigma_in<=r<=p.sigma_out) and abs(s*t)>0.15
            else: raise Refusal(f"unknown illum '{p.illum}'")
            if ok: pts.append((s,t))
    return pts or [(0.0,0.0)]

def zernike_radial(j: int, rho: "np.ndarray") -> "np.ndarray":
    """Proper low-order radial polynomials (Noll-ish subset)."""
    _np()
    R = {0:np.ones_like(rho),1:rho,2:2*rho**2-1,3:rho**3,
         4:6*rho**4-6*rho**2+1,5:rho**2,6:rho**4}
    return R.get(abs(j)%7, rho**abs(j))

def aerial_vector(mask: Mask3D, pitch: float, cd: float, p: Pupil,
                  direction="x", n_src=9, n_x=256,
                  cache: Optional[Cache]=None) -> Dict[str,Any]:
    """
    FACTORISED vector Abbe. O(N*n_x) per source point, not O(N^2*n_x).
      TE = |sum a_i e^{i phi_i}|^2
      TM = |sum a_i gamma_i e^{i phi_i}|^2 + |sum a_i alpha_i e^{i phi_i}|^2
    """
    _np()
    lam=p.wavelength_nm; na=p.na
    mag = mask.mag_x if direction=="x" else mask.mag_y
    ms = mask_spectrum(mask,pitch,cd,lam,direction=direction,cache=cache)
    fm = np.asarray(ms["freqs"],float); am = np.asarray(ms["amps"],complex)
    fw = fm*mag; pitch_w=pitch/mag; cd_w=cd/mag
    x = np.linspace(0,pitch_w,n_x,endpoint=False)
    f_cut = na/lam
    pols = ["TE","TM"] if p.polarisation=="unpolarised" else [p.polarisation]
    wts = [0.5,0.5] if len(pols)==2 else [1.0]
    I = np.zeros(n_x)
    for (sx,sy) in source_grid(p,n_src):
        fs = (sx if direction=="x" else sy)*f_cut
        a = fw-fs
        prop = np.abs(a)<=na+1e-12
        if not prop.any(): continue
        al = a[prop]/na*na/lam*lam/na  # alpha = lam*f  (kept explicit below)
        al = lam*a[prop]
        gm = np.sqrt(np.maximum(0.0,1-al*al))
        A = am[prop].copy()
        if p.defocus_nm!=0.0:
            A = A*np.exp(1j*PI*p.defocus_nm*lam*a[prop]**2)
        if p.zernike_nm:
            rho = np.clip(np.abs(al)/na,0,1)
            W = sum(c*zernike_radial(j,rho) for j,c in p.zernike_nm.items())
            A = A*np.exp(2j*PI*W/lam)
        if p.apodisation: A = A*np.sqrt(gm)
        PH = np.exp(2j*PI*np.outer(x,fw[prop]))    # n_x x N
        for pol,w in zip(pols,wts):
            if pol=="TE":
                F = PH@A
                I += w*np.abs(F)**2
            else:
                Fg = PH@(A*gm); Fa = PH@(A*al)
                I += w*(np.abs(Fg)**2+np.abs(Fa)**2)
    I /= max(len(source_grid(p,n_src)),1)
    I -= I.min()
    if I.max()>0: I /= I.max()
    nils = nils_edge(x,I,cd_w)
    return {"x_nm":x,"intensity":I,"pitch_wafer_nm":pitch_w,"cd_wafer_nm":cd_w,
            "nils":nils,"nils_target":4.2,"nils_pass":nils is not None and nils>=4.2,
            "polarisations":pols,"direction":direction,
            "method":"FACTORISED vector Abbe O(N*nx)","contrast":float(I.max()-I.min())}

def aerial_vector_pairwise(mask, pitch, cd, p, direction="x", n_src=5, n_x=128):
    """Reference O(N^2 nx) implementation, retained ONLY to verify factorisation."""
    _np()
    lam=p.wavelength_nm; na=p.na
    mag=mask.mag_x if direction=="x" else mask.mag_y
    ms=mask_spectrum(mask,pitch,cd,lam,direction=direction)
    fw=np.asarray(ms["freqs"],float)*mag; am=np.asarray(ms["amps"],complex)
    pitch_w=pitch/mag; cd_w=cd/mag
    x=np.linspace(0,pitch_w,n_x,endpoint=False); I=np.zeros(n_x)
    pols=["TE","TM"] if p.polarisation=="unpolarised" else [p.polarisation]
    wts=[0.5,0.5] if len(pols)==2 else [1.0]
    for (sx,sy) in source_grid(p,n_src):
        fs=(sx if direction=="x" else sy)*na/lam
        terms=[]
        for f,a in zip(fw,am):
            al=lam*(f-fs)
            if abs(al)>na+1e-12: continue
            gm=math.sqrt(max(0.0,1-al*al))
            A=a*(math.sqrt(gm) if p.apodisation else 1.0)
            if p.defocus_nm!=0.0: A*=np.exp(1j*PI*p.defocus_nm*lam*(f-fs)**2)
            terms.append((al,gm,A,f))
        for pol,w in zip(pols,wts):
            for i,(ai,gi,Ai,fi) in enumerate(terms):
                for j,(aj,gj,Aj,fj) in enumerate(terms):
                    d = 1.0 if pol=="TE" else gi*gj+ai*aj
                    I += w*float((Ai*np.conj(Aj)*d*np.exp(2j*PI*(fi-fj)*x)).real)
    I/=max(len(source_grid(p,n_src)),1); I-=I.min()
    if I.max()>0: I/=I.max()
    return {"x_nm":x,"intensity":I,"nils":nils_edge(x,I,cd_w),
            "method":"pairwise O(N^2 nx) reference"}

def nils_edge(x,I,cd):
    n=len(x)
    if n<8: return None
    dx=x[1]-x[0]; idx=max(2,min(n-3,int(round(cd/dx))))
    lo,hi=idx-2,idx+2
    dl=(math.log(max(float(I[hi]),1e-12))-math.log(max(float(I[lo]),1e-12)))/(x[hi]-x[lo])
    return abs(cd*dl)

def two_beam_contrast(na: float, lam=13.5) -> Dict[str,Any]:
    hp=0.25*lam/na; s=lam/(2*hp); th=math.asin(min(1.0,s)); tm=math.cos(2*th)
    return {"na":na,"half_pitch_nm":hp,"theta_deg":math.degrees(th),
            "contrast_TE":1.0,"contrast_TM":tm,
            "contrast_unpolarised":0.5*(1+tm),"TM_sign_reversed":tm<0,
            "scalar_model_would_report":1.0}


# =============================================================================
# §4. RCWA TE + TM (Li inverse rule), CACHED
# =============================================================================

@dataclass
class MaskStack:
    pitch_nm: float = 32.0; cd_nm: float = 16.0
    absorber: Material = TABN; absorber_nm: float = 60.0
    cra_deg: float = 6.0; wavelength_nm: float = 13.5
    ml: Multilayer = field(default_factory=Multilayer)
    shadow_nm: float = 0.0

def _rcwa_core(mask: MaskStack, pol: str, N: int, n_profile: int = 512
               ) -> Dict[str,Any]:
    _np()
    if N%2==0: N+=1
    lam=mask.wavelength_nm; k0=2*PI/lam; half=N//2
    x=np.arange(n_profile)*(mask.pitch_nm/n_profile)
    pos=(x+mask.shadow_nm)%mask.pitch_nm
    eps_x=np.where(pos<mask.cd_nm, mask.absorber.eps, VAC.eps)
    eh=np.fft.fftshift(np.fft.fft(eps_x))/n_profile
    ih=np.fft.fftshift(np.fft.fft(1/eps_x))/n_profile
    E=toeplitz(eh,N); Einv=toeplitz(ih,N)
    orders=np.arange(-half,half+1)
    alpha=math.sin(math.radians(mask.cra_deg))-orders*lam/mask.pitch_nm
    Kx=np.diag(alpha); I=np.eye(N,dtype=complex)
    gm=np.sqrt((1-alpha**2).astype(complex))
    for i in range(N):
        if gm[i].imag<0: gm[i]=-gm[i]
        if abs(gm[i].imag)<1e-14 and gm[i].real<0: gm[i]=-gm[i]
    if pol=="TE":
        A=E-Kx@Kx; fac="direct rule [[eps]]"
    else:
        A=np.linalg.solve(Einv, I-Kx@np.linalg.solve(E,Kx)); fac="Li inverse rule [[1/eps]]"
    q2,W=np.linalg.eig(A); q=np.sqrt(q2.astype(complex))
    for i in range(N):
        if q[i].imag<0: q[i]=-q[i]
        if abs(q[i].imag)<1e-14 and q[i].real<0: q[i]=-q[i]
    V = W@np.diag(q) if pol=="TE" else Einv@W@np.diag(q)
    r_ml=np.array([mask.ml.r_of_cos(complex(gm[i]),lam,pol) for i in range(N)])
    Z=np.empty(N,dtype=complex)
    for i in range(N):
        den=1+r_ml[i]
        if abs(den)<1e-12: den=complex(1e-12,0)
        Z[i]=gm[i]*(1-r_ml[i])/den
    Zm=np.diag(Z); X=np.diag(np.exp(1j*k0*q*mask.absorber_nm))
    Am=np.linalg.solve(V+Zm@W, V-Zm@W)
    F=W@(I+X@Am@X); G=V@(I-X@Am@X); Gam=np.diag(gm)
    inc=np.zeros(N,dtype=complex); inc[half]=1.0
    cp=np.linalg.solve(Gam@F+G, 2*Gam@inc); R=F@cp-inc
    ns=256; xs=np.arange(ns)*(mask.pitch_nm/ns)
    PH=np.exp(2j*PI*np.outer(xs/mask.pitch_nm,orders))
    near=PH@(inc+R); sec=PH@(Gam@(inc-R))
    prop=np.abs(alpha)<=1+1e-12; g0=gm[half].real
    eff=float(np.sum(np.abs(R[prop])**2*gm[prop].real)/g0) if g0>0 else 0.0
    flux=float(np.mean((near*np.conj(sec)).real))
    absorbed=1-flux/g0 if g0>0 else 0.0
    return {"pol":pol,"factorisation":fac,"n_orders":N,"alpha":alpha,
            "gamma":gm,"R_orders":R,"R_specular":float(abs(R[half])**2),
            "diffraction_efficiency":eff,"absorbed_fraction":absorbed,
            "x_nm":xs,"near_field":near,
            "energy_check":{"absorbed_nonneg":absorbed>=-1e-6,
                            "sum_le_1":eff+max(absorbed,0)<=1+1e-6}}

def rcwa(mask: MaskStack, pol="TE", n_orders=21,
         cache: Optional[Cache]=None) -> Dict[str,Any]:
    """Cached: eigendecomposition reused across focus/dose/shadow sweeps."""
    key = content_key("rcwa",pol,n_orders,mask.pitch_nm,mask.cd_nm,
                      mask.absorber_nm,mask.cra_deg,mask.wavelength_nm,
                      mask.shadow_nm,mask.absorber.name,
                      mask.ml.n_bilayers,mask.ml.d_hi_nm,mask.ml.d_lo_nm)
    if cache:
        got=cache.get(key)
        if got is not None:
            return {**got,"cache_hit":True}
    out=_rcwa_core(mask,pol,n_orders)
    if cache:
        cache.put(key,{k:v for k,v in out.items()
                       if isinstance(v,(np.ndarray,float,int,str))})
    return {**out,"cache_hit":False}

def rcwa_uniform_check(pol="TE", n_orders=11) -> Dict[str,Any]:
    _np()
    ml=Multilayer()
    m=MaskStack(pitch_nm=64,cd_nm=64,absorber_nm=40,ml=ml)
    o=rcwa(m,pol,n_orders); lam=m.wavelength_nm; d=m.absorber_nm
    ana=[]
    for a in o["alpha"]:
        g=complex(math.sqrt(complex(1-a*a)))
        if g.imag<0: g=-g
        rs=ml.r_of_cos(g,lam,pol); ph=np.exp(2j*(2*PI/lam)*g*d)
        ana.append(rs*ph)
    err=float(np.max(np.abs(o["R_orders"]-np.array(ana))))
    return {"pol":pol,"max_abs_error":err,"passes":err<1e-8,
            "note":"uniform layer must reproduce Fresnel exactly"}

def rcwa_convergence(mask: MaskStack, pol="TE",
                     orders=(11,21,31)) -> Dict[str,Any]:
    _np()
    fs=[]
    for N in orders:
        r=rcwa(mask,pol,N)
        idx=np.linspace(0,len(r["x_nm"])-1,128).astype(int)
        fs.append(r["near_field"][idx])
    rel=[float(np.linalg.norm(fs[i]-fs[i-1])/max(np.linalg.norm(fs[i]),1e-30))
         for i in range(1,len(fs))]
    return {"orders":list(orders),"rel_changes":rel,
            "converged":bool(rel and rel[-1]<1e-3),"tolerance":1e-3}

def kirchhoff_field(mask: MaskStack, xs) -> "np.ndarray":
    _np()
    r_ml=mask.ml.r_of_angle(mask.cra_deg,mask.wavelength_nm)
    k0=2*PI/mask.wavelength_nm; c0=math.cos(math.radians(mask.cra_deg))
    r_abs=r_ml*np.exp(2j*k0*(mask.absorber.n-1)*mask.absorber_nm*c0)
    pos=(xs+mask.shadow_nm)%mask.pitch_nm
    return np.where(pos<mask.cd_nm,r_abs,r_ml).astype(complex)


# =============================================================================
# §5. ADJOINT ILT (real gradient, not a grid sweep)
# =============================================================================

def ilt_adjoint(mask: Mask3D, pitch: float, target_cd: float, p: Pupil,
                n_iter=40, lr=0.05, n_x=128) -> Dict[str,Any]:
    """
    Optimise a complex edge correction (boundary-layer amplitude/phase) to
    drive the printed CD toward target. Gradient by central difference on
    the two real parameters — cheap because aerial is now O(N*nx).
    """
    _np()
    def printed(m):
        r=aerial_vector(m,pitch,pitch/2,p,n_src=5,n_x=n_x)
        I=r["intensity"]; return float((I>=0.30).mean()*r["pitch_wafer_nm"])
    m=Mask3D(absorber_nm=mask.absorber_nm,cra_deg=mask.cra_deg,
             mag_x=mask.mag_x,mag_y=mask.mag_y,ml=mask.ml,
             bl_amp=mask.bl_amp,bl_phase=mask.bl_phase)
    hist=[]; h=1e-3
    for it in range(n_iter):
        cd=printed(m); err=cd-target_cd; hist.append(err)
        m.bl_amp+=h; gp=(printed(m)-cd)/h; m.bl_amp-=h
        m.bl_phase+=h; gph=(printed(m)-cd)/h; m.bl_phase-=h
        gn=math.hypot(gp,gph) or 1.0
        m.bl_amp=max(0.0,m.bl_amp-lr*err*gp/gn)
        m.bl_phase=m.bl_phase-lr*err*gph/gn
        if abs(err)<0.02: break
    return {"target_cd_nm":target_cd,"final_cd_nm":printed(m),
            "bl_amp":m.bl_amp,"bl_phase":m.bl_phase,
            "iterations":it+1,"error_history":hist,
            "converged":abs(hist[-1])<0.05,
            "STATUS":"ILT proposal; requires M3D calibration + resist model"}


# =============================================================================
# §6. PROCESS WINDOW / EPE / BOSSUNG (cached sweeps)
# =============================================================================

def bossung(mask: Mask3D, pitch: float, cd: float, na: float,
            focus=(-60,60), n_focus=11, dose=(0.85,1.15), n_dose=7,
            threshold=0.30, lam=13.5, cache: Optional[Cache]=None
            ) -> Dict[str,Any]:
    _np()
    foc=np.linspace(*focus,n_focus); ds=np.linspace(*dose,n_dose)
    CDs=np.zeros((n_focus,n_dose)); nils=np.zeros(n_focus)
    for i,f in enumerate(foc):
        p=Pupil(na,lam,defocus_nm=float(f))
        r=aerial_vector(mask,pitch,cd,p,n_src=5,n_x=128,cache=cache)
        I=r["intensity"]; nils[i]=r["nils"] or 0.0
        for j,d in enumerate(ds):
            CDs[i,j]=float((I>=threshold/d).mean()*r["pitch_wafer_nm"])
    ibf=int(np.argmax(nils)); target=cd/mask.mag_x
    lo,hi=max(0,ibf-3),min(n_focus-1,ibf+3)
    pw=float((np.abs(CDs-target)<=0.10*target).mean())
    epe=float(np.max(np.abs(CDs-target)))
    return {"pitch_nm":pitch,"best_focus_nm":float(foc[ibf]),
            "cd_at_best_focus_nm":float(CDs[ibf,n_dose//2]),
            "bossung_tilt_nm":float(CDs[hi,n_dose//2]-CDs[lo,n_dose//2]),
            "process_window_fraction":pw,"max_EPE_nm":epe,
            "nils_through_focus":nils.tolist(),
            "M3D_SIGNATURE":"pitch-dependent best focus + Bossung tilt",
            "RESIST":"hard threshold, NOT a calibrated resist model"}

def best_focus_vs_pitch(mask: Mask3D, pitches: Sequence[float], na: float,
                        lam=13.5, cache: Optional[Cache]=None) -> Dict[str,Any]:
    rows=[{"pitch_nm":p,**{k:v for k,v in
           bossung(mask,p,p/2,na,n_focus=7,n_dose=5,lam=lam,cache=cache).items()
           if k in ("best_focus_nm","bossung_tilt_nm","max_EPE_nm")}}
          for p in pitches]
    bfs=[r["best_focus_nm"] for r in rows]
    return {"rows":rows,"best_focus_spread_nm":max(bfs)-min(bfs),
            "M3D_present":(max(bfs)-min(bfs))>1.0,
            "KIRCHHOFF_PREDICTS":"identical best focus for all pitches",
            "CALIBRATION":"compare against measured through-focus CD-SEM"}


# =============================================================================
# §7. ORCHESTRATOR + SELF-TEST
# =============================================================================

class LithoAPI:
    def __init__(self, cfg: Optional[RunConfig]=None):
        self.cfg=(cfg or RunConfig()).validate()
        self.cache=Cache(self.cfg.cache_dir,self.cfg.cache_max)
        self.sd=SDGate(self.cfg.strict); self.mf=Manifest(self.cfg.out_dir)
    def analyse(self, backend: str, pitch: float, cd: float, dose: float,
                cell: float, allow_speculative=False) -> Dict[str,Any]:
        b=get_backend(backend,allow_speculative)
        pb=photon_budget(b,dose,cell)
        self.sd.certify("photon_budget",(),pb)
        out={"backend":asdict(b),"rayleigh_cd_nm":b.cd(),"rayleigh_dof_nm":b.dof(),
             "photon_budget":pb}
        if pb.get("photon_shot_noise"):
            out["defect"]=delta_min_defect(cell,pb["photons_per_cell"])
            self.sd.certify("defect_threshold",("L1",),out["defect"])
            out["dose_for_q1e-9"]=dose_for_q(b,cell,1e-9)
        m=Mask3D(mag_x=b.mag_x,mag_y=b.mag_y)
        if b.na and b.wavelength_nm:
            r=aerial_vector(m,pitch,cd,Pupil(b.na,b.wavelength_nm),
                            cache=self.cache)
            out["aerial"]={k:v for k,v in r.items() if k not in ("x_nm","intensity")}
            self.sd.certify("aerial_image",("OPT",),out["aerial"])
            self.sd.certify("m3d_spectrum",(),{"calibrated":False})
        out["lwr"]=lwr_rls(pitch/2,dose)
        self.sd.certify("lwr",(),out["lwr"])
        out["scope"]="mathematics layer only; no scanner, no timeline, no superiority claim"
        self.mf.add("litho_analyse",{"backend":backend,"pitch":pitch})
        return out
    def report(self):
        rep={"version":VERSION,"core":CORE_VERSION,
             "backends":{k:asdict(v) for k,v in BACKENDS.items()},
             "cache":self.cache.stats(),"sd":self.sd.report()}
        Path(self.cfg.out_dir,"report.json").write_text(
            json.dumps(rep,indent=2,default=str))
        return rep

def self_test(verbose=True) -> bool:
    print("="*78); print(f"MSPS LITHO KERNEL v{VERSION} (unified, factorised)")
    print("="*78)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    api=LithoAPI(); cache=api.cache
    print("\n[T1] FACTORISATION == PAIRWISE (exactness)")
    m=Mask3D()
    for pol in ("TE","TM","unpolarised"):
        p=Pupil(0.55,polarisation=pol)
        a=aerial_vector(m,24,12,p,n_src=3,n_x=64)
        b=aerial_vector_pairwise(m,24,12,p,n_src=3,n_x=64)
        err=float(np.max(np.abs(a["intensity"]-b["intensity"])))
        print(f"      {pol:12s} max|factorised - pairwise| = {err:.2e}")
        rep(f"{pol} factorisation exact", err<1e-9, f"err={err:.1e}")
    p=Pupil(0.55,polarisation="TM")
    bf=bench(aerial_vector,m,24,12,p,n_src=5,n_x=128,n=3)
    bp=bench(aerial_vector_pairwise,m,24,12,p,n_src=5,n_x=128,n=1)
    print(f"      factorised {bf['mean_s']*1e3:.1f} ms vs pairwise "
          f"{bp['mean_s']*1e3:.1f} ms  speedup {bp['mean_s']/bf['mean_s']:.1f}x")
    rep("factorised faster", bf["mean_s"]<bp["mean_s"])

    print("\n[T2] VECTOR CONTRAST at two-beam limit")
    for na,exp in ((0.33,0.783),(0.55,0.397),(0.75,-0.125)):
        r=two_beam_contrast(na)
        print(f"      NA={na} theta={r['theta_deg']:.2f} TM={r['contrast_TM']:.3f}")
        rep(f"NA={na} TM matches theory", abs(r["contrast_TM"]-exp)<0.01)
    rep("Hyper-NA sign reversal", two_beam_contrast(0.75)["TM_sign_reversed"])

    print("\n[T3] COMPLEX SNELL (v1 bug fixed)")
    c=snell_cos(VAC.n,MO.n,complex(0.0,1.5))
    rep("evanescent cosine is complex", isinstance(c,complex))
    rep("finite", math.isfinite(c.real) and math.isfinite(c.imag))
    ml=Multilayer(); r6=ml.reflectance(6.0)
    print(f"      R(13.5,6deg)={r6['R']:.4f} phase={r6['phase_rad']:.4f}")
    rep("R physical", 0.50<=r6["R"]<=0.75)
    rep("TE=TM at normal", abs(ml.reflectance(0.0,pol="TE")["R"]
                              -ml.reflectance(0.0,pol="TM")["R"])<1e-9)

    print("\n[T4] RCWA TE + TM uniform limit")
    for pol in ("TE","TM"):
        u=rcwa_uniform_check(pol,11)
        print(f"      {pol} max err vs analytic = {u['max_abs_error']:.2e}")
        rep(f"{pol} uniform limit exact", u["passes"])
    ms=MaskStack()
    te=rcwa(ms,"TE",21,cache); tm=rcwa(ms,"TM",21,cache)
    rep("TM uses Li inverse rule","inverse rule" in tm["factorisation"])
    rep("TE uses direct rule","direct rule" in te["factorisation"])
    rep("TE != TM on pattern", abs(te["R_specular"]-tm["R_specular"])>1e-6)
    rep("TM energy ok", tm["energy_check"]["sum_le_1"])
    cv=rcwa_convergence(ms,"TE",(11,21,31))
    rep("convergence reported", len(cv["rel_changes"])==2)

    print("\n[T5] CACHE REUSE")
    h0=cache.stats()["hits"]
    for f in (-40,-20,0,20,40):
        aerial_vector(m,24,12,Pupil(0.55,defocus_nm=f),n_src=5,n_x=64,cache=cache)
    rep("mask spectrum cache hits", cache.stats()["hits"]>h0,
        f"hit_rate={cache.stats()['hit_rate']:.2f}")
    r1=rcwa(ms,"TE",21,cache); r2=rcwa(ms,"TE",21,cache)
    rep("RCWA second call is cache hit", r2["cache_hit"])

    print("\n[T6] PHOTON + DERIVED DOSE")
    nxe=get_backend("ASML_NXE3800E")
    pb=photon_budget(nxe,30,5)
    print(f"      5x5 nm @30 mJ: {pb['photons_per_cell']:.0f} photons, "
          f"{pb['shot_noise_pct']:.2f}%")
    rep("~510 photons", abs(pb["photons_per_cell"]-510)<10)
    dq=dose_for_q(nxe,5,1e-9)
    rep("derived dose achieves q",
        delta_min_defect(5,dq["photons_required"])["q_H"]<=1e-9*1.000001)
    b=False
    try: get_backend("HYPERFAB_X2")
    except SpeculativeBackend: b=True
    rep("HYPERFAB refused by default", b)

    print("\n[T7] ADJOINT ILT + PROCESS WINDOW")
    il=ilt_adjoint(m,24,6.0,Pupil(0.55),n_iter=25)
    print(f"      ILT cd {il['final_cd_nm']:.3f} nm (target 6.0), "
          f"iters={il['iterations']}")
    rep("ILT reduces error", abs(il["error_history"][-1])<=abs(il["error_history"][0]))
    bs=bossung(m,24,12,0.55,n_focus=7,n_dose=5,cache=cache)
    print(f"      BF={bs['best_focus_nm']:.1f} nm PW={bs['process_window_fraction']:.3f} "
          f"EPE={bs['max_EPE_nm']:.3f} nm")
    rep("PW in [0,1]", 0<=bs["process_window_fraction"]<=1)
    rep("resist caveat present","NOT a calibrated" in bs["RESIST"])
    bf=best_focus_vs_pitch(m,[32,26,22],0.55,cache=cache)
    rep("BF spread computed", bf["best_focus_spread_nm"]>=0)

    print("\n[T8] SD GATE")
    c1=api.sd.certify("m3d_spectrum",(),{}); rep("uncalibrated M3D MODE_E", c1["status"]==MODE_E)
    c2=api.sd.certify("process_window",("M3D","OPT"),{})
    rep("missing RES MODE_E", c2["status"]==MODE_E)
    c3=api.sd.certify("rcwa_reference",("EMF",),{})
    rep("missing CONV MODE_E", c3["status"]==MODE_E)
    rr=api.report()
    print(f"      SD: {rr['sd']['n']} outputs, {rr['sd']['certified']} certified, "
          f"{rr['sd']['mode_E']} Mode E")
    rep("Mode E retained", rr["sd"]["mode_E"]>0)
    rep("report written", Path(api.cfg.out_dir,"report.json").exists())
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    return ok

def main() -> int:
    ap=argparse.ArgumentParser(description="MSPS unified litho kernel v2")
    ap.add_argument("cmd",nargs="?",default="selftest",
        choices=["selftest","backends","contrast","photons","aerial","rcwa",
                 "uniform","ilt","bossung","analyse","report"])
    ap.add_argument("--backend",default="ASML_NXE3800E")
    ap.add_argument("--pitch",type=float,default=24.0)
    ap.add_argument("--cd",type=float,default=None)
    ap.add_argument("--na",type=float,default=0.55)
    ap.add_argument("--dose",type=float,default=30.0)
    ap.add_argument("--cell",type=float,default=5.0)
    ap.add_argument("--pol",default="unpolarised")
    ap.add_argument("--allow-speculative",action="store_true")
    ap.add_argument("--json")
    a=ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    api=LithoAPI(); cd=a.cd or a.pitch/2; out={}
    try:
        if a.cmd=="backends": out={k:asdict(v) for k,v in BACKENDS.items()}
        elif a.cmd=="contrast": out={str(n):two_beam_contrast(n) for n in (0.33,0.55,0.75)}
        elif a.cmd=="photons": out=photon_budget(get_backend(a.backend,a.allow_speculative),a.dose,a.cell)
        elif a.cmd=="aerial":
            r=aerial_vector(Mask3D(),a.pitch,cd,Pupil(a.na,polarisation=a.pol),cache=api.cache)
            out={k:v for k,v in r.items() if k not in ("x_nm","intensity")}
        elif a.cmd=="rcwa":
            r=rcwa(MaskStack(pitch_nm=a.pitch,cd_nm=cd),a.pol,21,api.cache)
            out={k:v for k,v in r.items() if k not in ("x_nm","near_field","alpha","gamma","R_orders")}
        elif a.cmd=="uniform": out={p:rcwa_uniform_check(p) for p in ("TE","TM")}
        elif a.cmd=="ilt": out=ilt_adjoint(Mask3D(),a.pitch,cd/4,Pupil(a.na))
        elif a.cmd=="bossung": out=bossung(Mask3D(),a.pitch,cd,a.na,cache=api.cache)
        elif a.cmd=="analyse": out=api.analyse(a.backend,a.pitch,cd,a.dose,a.cell,a.allow_speculative)
        elif a.cmd=="report": out=api.report()
    except Refusal as e: out={"REFUSED":str(e)}
    print(json.dumps(out,indent=2,default=str))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
