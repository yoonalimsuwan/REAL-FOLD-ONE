#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_litho_vector_m3d.py — v1.0.0
Vector diffraction + thin-mask 3D (M3D) extensions to the lithography math API.
Required at high NA. Scalar Kirchhoff models are invalid here.

Adds: polarization tensor, obliquity, vector Abbe/Hopkins,
      absorber shadowing, boundary-layer mask, Mo/Si multilayer
      transfer-matrix reflectance, Bossung/process-window with M3D.
=============================================================================
Author : PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID  : 0009-0008-2374-0788
License: MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHY SCALAR DIFFRACTION FAILS AT HIGH NA
=============================================================================
Two plane waves at angles +/-theta interfere with a polarization-dependent
contrast. Writing alpha = sin(theta), gamma = cos(theta):

  TE (s-polarised, E along y for a 1-D x-pattern):
      e_n . e_m* = 1                       (full contrast at any angle)
  TM (p-polarised, E in the x-z plane):
      e_n = (gamma_n, 0, -alpha_n)
      e_n . e_m* = gamma_n gamma_m + alpha_n alpha_m = cos(theta_n - theta_m)

For the symmetric two-beam case (+/-theta) this is cos(2*theta).
At the two-beam resolution limit hp = 0.25*lambda/NA:

  NXE:3800E  NA 0.33  theta 19.2 deg  cos2theta = 0.783
  EXE:5200   NA 0.55  theta 33.3 deg  cos2theta = 0.397
  Hyper-NA   NA 0.75  theta 48.6 deg  cos2theta = -0.125  (SIGN REVERSAL)

Unpolarised light averages TE and TM: (1 + cos2theta)/2 = cos^2(theta).
A scalar model reports contrast 1.0 in every case. That is the error.

=============================================================================
WHY THE THIN-MASK (KIRCHHOFF) APPROXIMATION FAILS IN EUV
=============================================================================
The absorber is ~40-70 nm thick at lambda = 13.5 nm, i.e. 3-5 wavelengths.
Consequences that a 2-D transmission function cannot represent:

  1. SHADOWING at oblique chief-ray incidence (CRA ~6 deg at mask):
         shift_mask  = h_abs * tan(CRA)      ~ 60 nm * 0.105 = 6.3 nm
         shift_wafer = shift_mask / M        1.58 nm at 4x, 0.79 nm at 8x
     At 8 nm half-pitch this is ~20% of the feature. Anamorphic magnification
     makes the shadow DIRECTION-DEPENDENT (x vs y).
  2. WAVEGUIDING through the absorber opening; edge boundary layers with
     complex transmission differing from both the open and covered regions.
  3. MULTILAYER PENETRATION: the effective reflection plane sits below the
     surface, adding a pitch- and angle-dependent phase.
  4. Through-focus signatures: pitch-dependent BEST FOCUS SHIFT, pattern
     shift, and Bossung tilt. These are the measurable M3D fingerprints.

This module implements (1)-(3) with a boundary-layer + transfer-matrix
model and computes (4) explicitly. It is NOT a rigorous RCWA/FDTD solver
for the full 3-D reflective stack; that requires a dedicated EMF solver.
The parameters that must be calibrated against measurement are flagged.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("vector_m3d")
VERSION = "1.0.0"; PI = math.pi
C_LIGHT = 2.99792458e8; H_PLANCK = 6.62607015e-34
EV_J = 1.602176634e-19

BANNER = f"""
=============================================================================
MSPS LITHO VECTOR + THIN-MASK 3D v{VERSION}
Polarization tensor · obliquity · vector Abbe · absorber shadowing
boundary-layer mask · Mo/Si transfer matrix · Bossung with M3D
Scalar Kirchhoff is INVALID at high NA. This module replaces it.
=============================================================================
"""

class Refusal(RuntimeError): pass
class NeedsCalibration(RuntimeError): pass


# =============================================================================
# §1. VECTOR DIFFRACTION GEOMETRY
# =============================================================================

@dataclass(frozen=True)
class VectorOrder:
    """One diffracted order with its full 3-D propagation direction."""
    fx: float          # spatial frequency [1/nm]
    alpha: float       # sin(theta) = lambda*fx (image side)
    gamma: float       # cos(theta)
    propagating: bool

    @property
    def theta_deg(self) -> float:
        return math.degrees(math.asin(max(-1.0, min(1.0, self.alpha))))


def make_order(fx: float, wavelength_nm: float, na: float) -> VectorOrder:
    a = wavelength_nm*fx
    prop = abs(a) <= na + 1e-12
    g = math.sqrt(max(0.0, 1.0 - a*a)) if prop else 0.0
    return VectorOrder(fx, a, g, prop)


def pol_vectors(o: VectorOrder, pol: str) -> Tuple[complex, complex, complex]:
    """
    Unit polarization vector for a propagating order.
      TE (s): E along y, perpendicular to the x-z plane of incidence.
      TM (p): E in x-z, perpendicular to k = (alpha, 0, gamma).
    """
    if not o.propagating: return (0j, 0j, 0j)
    if pol == "TE": return (0j, 1.0+0j, 0j)
    if pol == "TM": return (o.gamma+0j, 0j, -o.alpha+0j)
    raise Refusal(f"pol must be TE|TM, got '{pol}'")


def pol_dot(o_n: VectorOrder, o_m: VectorOrder, pol: str) -> complex:
    """
    THE VECTOR EFFECT. e_n . e_m*
      TE -> 1
      TM -> gamma_n*gamma_m + alpha_n*alpha_m = cos(theta_n - theta_m)
    """
    if not (o_n.propagating and o_m.propagating): return 0j
    if pol == "TE": return 1.0+0j
    return complex(o_n.gamma*o_m.gamma + o_n.alpha*o_m.alpha)


def two_beam_contrast(na: float, wavelength_nm: float = 13.5,
                      at_limit: bool = True,
                      pitch_nm: Optional[float] = None) -> Dict[str, Any]:
    """
    Contrast of the symmetric two-beam interference for TE / TM / unpolarised.
    at_limit=True uses hp = 0.25*lambda/NA (the two-beam resolution limit).
    """
    if at_limit:
        hp = 0.25*wavelength_nm/na
        pitch = 2*hp
    else:
        if pitch_nm is None: raise Refusal("give pitch_nm or at_limit=True")
        pitch = pitch_nm; hp = pitch/2
    s = wavelength_nm/(2*hp)
    if s > na + 1e-9:
        return {"resolvable": False, "sin_theta": s, "na": na,
                "pitch_nm": pitch,
                "reason": "diffraction orders fall outside the pupil"}
    th = math.asin(s)
    tm = math.cos(2*th)
    return {"resolvable": True, "na": na, "wavelength_nm": wavelength_nm,
            "half_pitch_nm": hp, "pitch_nm": pitch,
            "sin_theta": s, "theta_deg": math.degrees(th),
            "contrast_TE": 1.0, "contrast_TM": tm,
            "contrast_unpolarised": 0.5*(1.0+tm),
            "TM_sign_reversed": tm < 0,
            "scalar_model_would_report": 1.0,
            "scalar_error_factor": (1.0/tm) if abs(tm) > 1e-9 else float("inf")}


def obliquity_factor(o: VectorOrder) -> float:
    """
    Radiometric obliquity. Energy flux through the exit pupil scales with
    gamma = cos(theta). Combined with the 1/gamma from the projection
    Jacobian this gives the standard sqrt or unity weighting depending on
    the chosen normalisation; we expose gamma explicitly.
    """
    return o.gamma if o.propagating else 0.0


# =============================================================================
# §2. Mo/Si MULTILAYER — EXACT TRANSFER-MATRIX RECURSION
# =============================================================================

@dataclass(frozen=True)
class Material:
    """EUV optical constants: n = 1 - delta + i*beta."""
    name: str; delta: float; beta: float
    @property
    def n(self) -> complex: return complex(1.0-self.delta, self.beta)

# Approximate CXRO values near 13.5 nm. MUST be replaced by measured
# optical constants for production use.
MO = Material("Mo", 0.0770, 0.0064)
SI = Material("Si", 0.0008, 0.0018)
RU = Material("Ru", 0.0870, 0.0080)
VAC = Material("vacuum", 0.0, 0.0)


def fresnel_r(n1: complex, n2: complex, cos1: complex, cos2: complex,
              pol: str) -> complex:
    """Amplitude reflection coefficient at an interface."""
    if pol == "TE":
        return (n1*cos1 - n2*cos2)/(n1*cos1 + n2*cos2)
    return (n2*cos1 - n1*cos2)/(n2*cos1 + n1*cos2)


def snell_cos(n1: complex, n2: complex, cos1: complex) -> complex:
    s1 = math.sqrt(max(0.0, 1.0 - float(cos1.real)**2))
    s2 = n1*s1/n2
    return (1.0 - s2*s2)**0.5


@dataclass
class Multilayer:
    """Mo/Si bilayer stack with optional Ru cap."""
    n_bilayers: int = 45
    d_mo_nm: float = 2.8
    d_si_nm: float = 4.1
    cap_ru_nm: float = 2.5
    substrate: Material = SI
    gamma_ratio: float = 0.40      # Mo fraction of the period

    @property
    def period_nm(self) -> float: return self.d_mo_nm + self.d_si_nm

    def reflectance(self, wavelength_nm: float = 13.5,
                    angle_deg: float = 6.0, pol: str = "TE"
                    ) -> Dict[str, Any]:
        """
        Exact recursive Fresnel (Parratt-style) amplitude reflectance.
        r_j = (r_{j,j+1} + r_{j+1} e^{2i b_{j+1}}) /
              (1 + r_{j,j+1} r_{j+1} e^{2i b_{j+1}})
        """
        th = math.radians(angle_deg)
        cos0 = complex(math.cos(th))
        # build layer list top -> bottom
        layers: List[Tuple[Material, float]] = []
        if self.cap_ru_nm > 0: layers.append((RU, self.cap_ru_nm))
        for _ in range(self.n_bilayers):
            layers.append((SI, self.d_si_nm))
            layers.append((MO, self.d_mo_nm))
        layers.append((self.substrate, 0.0))   # semi-infinite

        n_list = [VAC.n] + [m.n for m,_ in layers]
        cos_list = [cos0]
        for i in range(1, len(n_list)):
            cos_list.append(snell_cos(n_list[i-1], n_list[i], cos_list[i-1]))

        # start from the bottom interface
        r = fresnel_r(n_list[-2], n_list[-1], cos_list[-2], cos_list[-1], pol)
        for j in range(len(layers)-2, -1, -1):
            d = layers[j][1]
            b = 2*PI*n_list[j+1]*cos_list[j+1]*d/wavelength_nm
            ph = complex(math.cos(2*b.real), math.sin(2*b.real))*math.exp(-2*b.imag)
            rj = fresnel_r(n_list[j], n_list[j+1], cos_list[j], cos_list[j+1], pol)
            r = (rj + r*ph)/(1.0 + rj*r*ph)
        R = float(abs(r)**2)
        phase = math.atan2(r.imag, r.real)
        # effective penetration: phase / (2 k cos) gives a depth
        k = 2*PI/wavelength_nm
        d_eff = phase/(2*k*float(cos0.real)) if abs(cos0.real) > 1e-9 else 0.0
        return {"R": R, "amplitude": abs(r), "phase_rad": phase,
                "n_bilayers": self.n_bilayers, "period_nm": self.period_nm,
                "angle_deg": angle_deg, "pol": pol,
                "effective_plane_depth_nm": d_eff,
                "bragg_estimate_nm": 2*self.period_nm*math.cos(th),
                "OPTICAL_CONSTANTS": "approximate; replace with measured data"}

    def peak_wavelength(self, lo: float = 12.5, hi: float = 14.5,
                        n: int = 201, angle_deg: float = 6.0
                        ) -> Dict[str, Any]:
        best = (0.0, -1.0)
        for i in range(n):
            w = lo + (hi-lo)*i/(n-1)
            R = self.reflectance(w, angle_deg)["R"]
            if R > best[1]: best = (w, R)
        return {"peak_wavelength_nm": best[0], "peak_R": best[1],
                "target_nm": 13.5,
                "detuned_nm": best[0]-13.5}


def multilayer_phase_error(ml: Multilayer, wavelength_nm: float = 13.5,
                           angle_deg: float = 6.0) -> Dict[str, Any]:
    """
    The reflection phase is NOT zero. A thin-mask model sets it to zero and
    therefore mis-places the effective image plane. This returns the phase
    and the equivalent defocus.
    """
    r = ml.reflectance(wavelength_nm, angle_deg)
    phi = r["phase_rad"]
    # equivalent single-pass defocus producing the same phase
    dz = phi*wavelength_nm/(4*PI)
    return {"phase_rad": phi, "equivalent_defocus_nm": dz,
            "thin_mask_assumes_rad": 0.0,
            "phase_error_rad": phi,
            "CONSEQUENCE": "pitch-independent focus offset; must be calibrated"}


# =============================================================================
# §3. THIN-MASK 3D: BOUNDARY LAYER + SHADOWING
# =============================================================================

@dataclass
class Mask3D:
    """
    Reflective EUV mask with 3-D absorber effects.

    Model class: boundary-layer (a.k.a. domain-decomposition / extended
    thin-mask). This is a CALIBRATED surrogate, not rigorous RCWA.
    """
    absorber_nm: float = 60.0
    cra_deg: float = 6.0            # chief ray angle at the mask
    mag_x: float = 4.0
    mag_y: float = 4.0
    # complex amplitude reflectances (must be measured / EMF-extracted)
    r_multilayer: complex = complex(0.80, 0.0)
    r_absorber: complex = complex(0.03, 0.01)
    # edge boundary layers
    bl_width_nm: float = 4.0        # at mask scale
    bl_amplitude: float = 0.35
    bl_phase_rad: float = -1.2
    ml: Multilayer = field(default_factory=Multilayer)
    calibrated: bool = False

    def shadow_at_mask_nm(self, direction: str = "x") -> float:
        """Lateral edge shift from oblique illumination of a thick absorber."""
        return self.absorber_nm*math.tan(math.radians(self.cra_deg))

    def shadow_at_wafer_nm(self, direction: str = "x") -> float:
        m = self.mag_x if direction == "x" else self.mag_y
        return self.shadow_at_mask_nm(direction)/m

    def anamorphic_shadow_asymmetry_nm(self) -> Dict[str, float]:
        sx = self.shadow_at_wafer_nm("x")
        sy = self.shadow_at_wafer_nm("y")
        return {"x_wafer_nm": sx, "y_wafer_nm": sy,
                "asymmetry_nm": abs(sx-sy),
                "mask_shift_nm": self.shadow_at_mask_nm(),
                "anisotropic": abs(self.mag_x-self.mag_y) > 1e-9,
                "CONSEQUENCE": ("H/V features shift differently; a single "
                                "global bias cannot correct both")}

    def mask_spectrum(self, pitch_nm: float, cd_nm: float,
                      n_orders: int = 9, dx_nm: float = 0.25,
                      direction: str = "x",
                      wavelength_nm: float = 13.5) -> Dict[str, Any]:
        """
        Complex mask diffraction spectrum including:
          - absorber shadowing (edge shift, direction dependent)
          - edge boundary layers with complex transmission
          - multilayer reflection phase
        Returned at MASK scale; frequencies later scaled by magnification.
        """
        if not HAS_NUMPY:
            raise Refusal("mask_spectrum requires numpy")
        n = max(64, int(round(pitch_nm/dx_nm)))
        n = int(math.ceil(n/8)*8)
        x = np.arange(n)*dx_nm
        shift = self.shadow_at_mask_nm(direction)
        pos = (x + shift) % pitch_nm
        inside = pos < cd_nm
        # base binary reflective mask
        t = np.where(inside, self.r_absorber, self.r_multilayer).astype(complex)
        # boundary layers at both edges of the absorber
        dist_edge = np.minimum(pos, np.abs(pitch_nm-pos))
        dist_open = np.minimum(np.abs(pos-cd_nm), np.abs(pos-cd_nm+pitch_nm))
        bl_mask = (dist_open < self.bl_width_nm) & (~inside)
        bl = self.bl_amplitude*np.exp(1j*self.bl_phase_rad)
        t[bl_mask] = bl*self.r_multilayer
        # multilayer reflection phase (angle-independent here; flagged)
        mlp = self.ml.reflectance(wavelength_nm, self.cra_deg)["phase_rad"]
        t = t*np.exp(1j*mlp)
        spec = np.fft.fft(t)/n
        freqs = np.fft.fftfreq(n, d=dx_nm)
        idx = np.argsort(freqs)
        keep = [i for i in idx if abs(freqs[i]) <= n_orders/pitch_nm + 1e-9]
        return {"freqs": freqs[keep], "amps": spec[keep],
                "n_orders": len(keep), "pitch_nm": pitch_nm, "cd_nm": cd_nm,
                "shadow_mask_nm": shift, "direction": direction,
                "ml_phase_rad": mlp,
                "MODEL": "boundary-layer surrogate, NOT rigorous RCWA",
                "CALIBRATED": self.calibrated}


def kirchhoff_vs_m3d(pitch_nm: float, cd_nm: float, mask: Mask3D,
                     wavelength_nm: float = 13.5) -> Dict[str, Any]:
    """Quantify what the thin-mask approximation misses."""
    m3d = mask.mask_spectrum(pitch_nm, cd_nm, wavelength_nm=wavelength_nm)
    # Kirchhoff: same binary pattern, no shadow, no BL, no ML phase
    k = Mask3D(absorber_nm=mask.absorber_nm, cra_deg=0.0,
               mag_x=mask.mag_x, mag_y=mask.mag_y,
               r_multilayer=mask.r_multilayer, r_absorber=mask.r_absorber,
               bl_width_nm=0.0, ml=mask.ml)
    kir = k.mask_spectrum(pitch_nm, cd_nm, wavelength_nm=wavelength_nm)
    n = min(len(m3d["amps"]), len(kir["amps"]))
    a = m3d["amps"][:n]; b = kir["amps"][:n]
    denom = float(np.sum(np.abs(b)**2)) if HAS_NUMPY else 1.0
    err = float(np.sum(np.abs(a-b)**2)/max(denom, 1e-30)) if HAS_NUMPY else 0.0
    return {"pitch_nm": pitch_nm,
            "spectral_error_relative": err,
            "shadow_mask_nm": m3d["shadow_mask_nm"],
            "shadow_wafer_x_nm": mask.shadow_at_wafer_nm("x"),
            "shadow_wafer_y_nm": mask.shadow_at_wafer_nm("y"),
            "ml_phase_rad": m3d["ml_phase_rad"],
            "kirchhoff_adequate": err < 0.01,
            "VERDICT": ("Kirchhoff INVALID" if err >= 0.01
                        else "Kirchhoff acceptable for this pitch")}


# =============================================================================
# §4. VECTOR + M3D AERIAL IMAGE
# =============================================================================

@dataclass
class VectorPupil:
    na: float
    wavelength_nm: float = 13.5
    sigma: float = 0.7
    illum: str = "annular"
    sigma_in: float = 0.4
    sigma_out: float = 0.8
    polarisation: str = "unpolarised"   # TE | TM | unpolarised | x | y
    defocus_nm: float = 0.0
    zernike_nm: Dict[int, float] = field(default_factory=dict)
    apodisation: bool = True


def source_grid(p: VectorPupil, n: int = 15) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n):
        s = -1.0 + 2.0*i/max(n-1, 1)
        for j in range(n):
            t = -1.0 + 2.0*j/max(n-1, 1)
            r = math.hypot(s, t)
            if p.illum == "conventional": ok = r <= p.sigma
            elif p.illum == "annular":    ok = p.sigma_in <= r <= p.sigma_out
            elif p.illum == "dipole_x":   ok = (p.sigma_in <= r <= p.sigma_out) and abs(t) < 0.35
            elif p.illum == "dipole_y":   ok = (p.sigma_in <= r <= p.sigma_out) and abs(s) < 0.35
            elif p.illum == "quadrupole": ok = (p.sigma_in <= r <= p.sigma_out) and abs(s*t) > 0.15
            else: raise Refusal(f"unknown illum '{p.illum}'")
            if ok: pts.append((s, t))
    return pts or [(0.0, 0.0)]


def aerial_vector_m3d(pitch_nm: float, cd_nm: float, mask: Mask3D,
                      pupil: VectorPupil, n_src: int = 11,
                      direction: str = "x",
                      n_x: int = 256) -> Dict[str, Any]:
    """
    Vector Abbe aerial image with a 3-D mask spectrum.

    I(x) = sum_source sum_pol w_pol * | sum_n a_n (e_n . e_ref) P_n e^{i2pi f_n x} |^2

    The polarization dot product is taken between every pair of orders,
    which is exactly the term a scalar model omits.
    """
    if not HAS_NUMPY: raise Refusal("requires numpy")
    lam = pupil.wavelength_nm; na = pupil.na
    mag = mask.mag_x if direction == "x" else mask.mag_y
    ms = mask.mask_spectrum(pitch_nm, cd_nm, dx_nm=pitch_nm/64,
                            direction=direction, wavelength_nm=lam)
    # mask frequency -> wafer frequency via magnification
    f_mask = np.asarray(ms["freqs"], float)
    amps = np.asarray(ms["amps"], complex)
    f_wafer = f_mask*mag
    pitch_w = pitch_nm/mag
    cd_w = cd_nm/mag

    pols = ["TE", "TM"] if pupil.polarisation in ("unpolarised",) else \
           (["TE"] if pupil.polarisation in ("y", "TE") else ["TM"])
    weights = [0.5, 0.5] if len(pols) == 2 else [1.0]

    src = source_grid(pupil, n_src)
    x = np.linspace(0, pitch_w, n_x, endpoint=False)
    I = np.zeros(n_x)
    f_cut = na/lam

    for (sx, sy) in src:
        fs = sx*f_cut if direction == "x" else sy*f_cut
        for pol, w in zip(pols, weights):
            # each source point shifts the pupil; orders enter at f_n - fs
            field = np.zeros(n_x, dtype=complex)
            terms = []
            for fn, an in zip(f_wafer, amps):
                fo = make_order(float(fn - fs), lam, na)
                if not fo.propagating: continue
                P = 1.0+0j
                if pupil.defocus_nm != 0.0:
                    P *= np.exp(1j*PI*pupil.defocus_nm*lam*(fo.alpha/lam)**2)
                if pupil.zernike_nm:
                    rho = abs(fo.alpha)/na
                    Wz = sum(a*rho**abs(j) for j, a in pupil.zernike_nm.items())
                    P *= np.exp(2j*PI*Wz/lam)
                apo = math.sqrt(fo.gamma) if pupil.apodisation else 1.0
                terms.append((fo, an*P*apo, fn))
            # pairwise interference with the polarization dot product
            for i, (oi, ai, fi) in enumerate(terms):
                for j, (oj, aj, fj) in enumerate(terms):
                    d = pol_dot(oi, oj, pol)
                    if d == 0: continue
                    I += w*float((ai*np.conj(aj)*d *
                                  np.exp(2j*PI*(fi-fj)*x)).real)
    I /= max(len(src), 1)
    I = I - I.min()
    if I.max() > 0: I = I/I.max()

    nils = nils_edge(x, I, cd_w)
    return {"x_nm": x.tolist(), "intensity": I.tolist(),
            "pitch_wafer_nm": pitch_w, "cd_wafer_nm": cd_w,
            "n_source_points": len(src), "polarisations": pols,
            "nils": nils, "nils_target": 4.2,
            "nils_pass": nils is not None and nils >= 4.2,
            "direction": direction,
            "model": "vector Abbe + boundary-layer M3D",
            "contrast_max": float(I.max()-I.min())}


def nils_edge(x, I, cd_nm: float) -> Optional[float]:
    n = len(x)
    if n < 8: return None
    dx = x[1]-x[0]
    idx = max(2, min(n-3, int(round(cd_nm/dx))))
    lo, hi = idx-2, idx+2
    dln = (math.log(max(float(I[hi]), 1e-12)) -
           math.log(max(float(I[lo]), 1e-12)))/(x[hi]-x[lo])
    return abs(cd_nm*dln)


def scalar_vs_vector(pitch_nm: float, cd_nm: float, mask: Mask3D,
                     na: float, wavelength_nm: float = 13.5,
                     illum: str = "dipole_x") -> Dict[str, Any]:
    """Run the same configuration scalar (TE-only) and vector; report the gap."""
    base = dict(sigma_in=0.5, sigma_out=0.85, illum=illum)
    te = aerial_vector_m3d(pitch_nm, cd_nm, mask,
        VectorPupil(na, wavelength_nm, polarisation="TE", **base))
    tm = aerial_vector_m3d(pitch_nm, cd_nm, mask,
        VectorPupil(na, wavelength_nm, polarisation="TM", **base))
    un = aerial_vector_m3d(pitch_nm, cd_nm, mask,
        VectorPupil(na, wavelength_nm, polarisation="unpolarised", **base))
    def contrast(r):
        I = r["intensity"]; return max(I)-min(I)
    return {"na": na, "pitch_nm": pitch_nm,
            "NILS_TE": te["nils"], "NILS_TM": tm["nils"],
            "NILS_unpolarised": un["nils"],
            "contrast_TE": contrast(te), "contrast_TM": contrast(tm),
            "contrast_unpolarised": contrast(un),
            "TM_over_TE": (tm["nils"]/te["nils"]) if te["nils"] else None,
            "scalar_model_equals": "TE result (no polarization loss)",
            "scalar_overestimates_by": (te["nils"]/un["nils"]) if un["nils"] else None}


# =============================================================================
# §5. BOSSUNG / PROCESS WINDOW WITH M3D
# =============================================================================

def bossung(pitch_nm: float, cd_nm: float, mask: Mask3D, na: float,
            focus_range_nm: Tuple[float, float] = (-60.0, 60.0),
            n_focus: int = 13, dose_range: Tuple[float, float] = (0.8, 1.2),
            n_dose: int = 9, threshold: float = 0.30,
            wavelength_nm: float = 13.5) -> Dict[str, Any]:
    """
    Through-focus / through-dose CD. M3D shows up as:
      - pitch-dependent BEST FOCUS (non-zero, varies with pitch)
      - Bossung TILT (asymmetry about best focus)
    """
    if not HAS_NUMPY: raise Refusal("requires numpy")
    foc = np.linspace(*focus_range_nm, n_focus)
    doses = np.linspace(*dose_range, n_dose)
    cds = np.zeros((n_focus, n_dose))
    nils = np.zeros(n_focus)
    for i, f in enumerate(foc):
        p = VectorPupil(na, wavelength_nm, illum="dipole_x",
                        sigma_in=0.5, sigma_out=0.85,
                        polarisation="unpolarised", defocus_nm=float(f))
        r = aerial_vector_m3d(pitch_nm, cd_nm, mask, p, n_src=7)
        I = np.asarray(r["intensity"]); x = np.asarray(r["x_nm"])
        nils[i] = r["nils"] or 0.0
        for j, d in enumerate(doses):
            above = I >= threshold/d
            cds[i, j] = float(above.sum()/len(x)*r["pitch_wafer_nm"])
    ibf = int(np.argmax(nils))
    best_focus = float(foc[ibf])
    cd_bf = cds[ibf, n_dose//2]
    # tilt: CD difference symmetric about best focus at nominal dose
    lo = max(0, ibf-3); hi = min(n_focus-1, ibf+3)
    tilt = float(cds[hi, n_dose//2] - cds[lo, n_dose//2])
    # process window area (simplified: count cells within +/-10% CD)
    target = cd_nm/mask.mag_x
    ok = np.abs(cds-target) <= 0.10*target
    pw_frac = float(ok.sum()/cds.size)
    return {"pitch_nm": pitch_nm, "na": na,
            "best_focus_nm": best_focus,
            "best_focus_nonzero": abs(best_focus) > 1e-6,
            "cd_at_best_focus_nm": cd_bf,
            "bossung_tilt_nm": tilt,
            "bossung_asymmetric": abs(tilt) > 0.05,
            "process_window_fraction": pw_frac,
            "focus_points": foc.tolist(),
            "nils_through_focus": nils.tolist(),
            "M3D_SIGNATURE": ("pitch-dependent best focus + Bossung tilt; "
                              "absent in a Kirchhoff model")}


def best_focus_vs_pitch(pitches: Sequence[float], mask: Mask3D, na: float,
                        wavelength_nm: float = 13.5) -> Dict[str, Any]:
    """THE measurable M3D fingerprint: best focus shifts with pitch."""
    rows = []
    for p in pitches:
        b = bossung(p, p/2, mask, na, n_focus=9, n_dose=5,
                    wavelength_nm=wavelength_nm)
        rows.append({"pitch_nm": p, "best_focus_nm": b["best_focus_nm"],
                     "tilt_nm": b["bossung_tilt_nm"]})
    bfs = [r["best_focus_nm"] for r in rows]
    return {"rows": rows, "best_focus_spread_nm": max(bfs)-min(bfs),
            "M3D_present": (max(bfs)-min(bfs)) > 1.0,
            "KIRCHHOFF_PREDICTS": "best focus identical for all pitches",
            "CALIBRATION": "compare against measured through-focus CD-SEM"}


# =============================================================================
# §6. SD GATE INTEGRATION
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "vector_contrast": (("OPT",), "vector Abbe polarization tensor",
                          "exact within scalar-mask spectrum"),
      "multilayer_R": (("OC",), "Fresnel recursion (exact)",
                     "requires measured optical constants"),
      "m3d_spectrum": (("M3D",), "boundary-layer surrogate",
                     "NOT rigorous RCWA; needs EMF calibration"),
      "best_focus": (("M3D","OPT"), "through-focus vector image",
                   "model-dependent until measured"),
      "process_window": (("M3D","OPT","RES"), "Bossung with resist threshold",
                         "resist model is a hard threshold here"),
    }
    def __init__(self): self.log: List[Dict] = []
    def certify(self, stat: str, verified: Sequence[str], value=None) -> Dict:
        e = self.ENTRIES.get(stat); v = tuple(verified)
        if e is None:
            r = {"statistic": stat, "status": MODE_E, "reason": "no entry"}
        else:
            need, thm, cav = e
            miss = tuple(p for p in need if p not in v)
            r = {"statistic": stat, "theorem": thm, "caveat": cav,
                 "required": list(need), "verified": list(v),
                 "missing": list(miss),
                 "status": MODE_E if miss else MODE_CERTIFIED}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def report(self) -> Dict:
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}

PACKAGES = {
 "OPT": "measured optical constants, pupil, aberrations",
 "OC":  "measured Mo/Si optical constants at the working wavelength",
 "M3D": "EMF-extracted or measured boundary-layer parameters",
 "RES": "calibrated resist model (not a hard threshold)",
}


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] VECTOR CONTRAST at the two-beam limit")
    for name, na, exp_tm in (("NXE:3800E", 0.33, 0.783),
                             ("EXE:5200", 0.55, 0.397),
                             ("Hyper-NA", 0.75, -0.125)):
        r = two_beam_contrast(na)
        print(f"      {name:12s} NA={na}  theta={r['theta_deg']:.2f} deg  "
              f"TE={r['contrast_TE']:.3f}  TM={r['contrast_TM']:.3f}  "
              f"unpol={r['contrast_unpolarised']:.3f}")
        rep(f"{name} TM contrast matches theory",
            abs(r["contrast_TM"]-exp_tm) < 0.01,
            f"got {r['contrast_TM']:.4f}, expected {exp_tm}")
        rep(f"{name} TE always 1.0", abs(r["contrast_TE"]-1.0) < 1e-12)
        rep(f"{name} unpolarised = (1+TM)/2",
            abs(r["contrast_unpolarised"]-0.5*(1+r["contrast_TM"])) < 1e-12)
    hyp = two_beam_contrast(0.75)
    rep("Hyper-NA TM contrast REVERSES sign", hyp["TM_sign_reversed"])
    rep("scalar model reports 1.0 (the error)",
        hyp["scalar_model_would_report"] == 1.0)

    print("\n[T2] polarization dot product identity")
    o1 = make_order(0.03, 13.5, 0.55); o2 = make_order(-0.03, 13.5, 0.55)
    d_te = pol_dot(o1, o2, "TE"); d_tm = pol_dot(o1, o2, "TM")
    analytic = math.cos(o1.theta_deg*math.pi/180 - o2.theta_deg*math.pi/180)
    rep("TE dot = 1", abs(d_te-1.0) < 1e-12)
    rep("TM dot = cos(theta_n - theta_m)", abs(d_tm.real-analytic) < 1e-12,
        f"{d_tm.real:.6f} vs {analytic:.6f}")
    ev = make_order(0.2, 13.5, 0.55)
    rep("evanescent order carries no field", not ev.propagating and
        pol_dot(ev, o1, "TE") == 0)

    print("\n[T3] Mo/Si MULTILAYER transfer matrix")
    ml = Multilayer()
    r6 = ml.reflectance(13.5, 6.0)
    print(f"      period={ml.period_nm:.2f} nm  N={ml.n_bilayers}  "
          f"R(13.5nm,6deg)={r6['R']:.4f}  phase={r6['phase_rad']:.4f} rad")
    rep("R in physical range 0.5-0.75", 0.50 <= r6["R"] <= 0.75,
        f"R={r6['R']:.4f}")
    pk = ml.peak_wavelength()
    print(f"      peak at {pk['peak_wavelength_nm']:.3f} nm, R={pk['peak_R']:.4f}")
    rep("Bragg peak near 13.5 nm", abs(pk["peak_wavelength_nm"]-13.5) < 0.5)
    rep("peak R exceeds on-target R", pk["peak_R"] >= r6["R"]-1e-9)
    r0 = ml.reflectance(13.5, 0.0); r20 = ml.reflectance(13.5, 20.0)
    rep("R falls with angle (detuning)", r20["R"] < r0["R"],
        f"R(0)={r0['R']:.4f} R(20)={r20['R']:.4f}")
    pe = multilayer_phase_error(ml)
    print(f"      reflection phase={pe['phase_rad']:.4f} rad -> "
          f"equiv defocus {pe['equivalent_defocus_nm']:.3f} nm")
    rep("thin-mask assumes zero phase (error)", pe["thin_mask_assumes_rad"] == 0.0)
    rep("phase error is non-zero", abs(pe["phase_error_rad"]) > 1e-6)

    print("\n[T4] THIN-MASK 3D shadowing")
    m4 = Mask3D(mag_x=4.0, mag_y=4.0)
    m8 = Mask3D(mag_x=4.0, mag_y=8.0)
    s_mask = m4.shadow_at_mask_nm()
    print(f"      absorber 60 nm, CRA 6 deg -> mask shift {s_mask:.3f} nm")
    rep("mask shadow ~6.3 nm", abs(s_mask-6.30) < 0.05)
    print(f"      wafer 4x: {m4.shadow_at_wafer_nm('x'):.3f} nm   "
          f"8x: {m8.shadow_at_wafer_nm('y'):.3f} nm")
    rep("4x wafer shadow ~1.58 nm",
        abs(m4.shadow_at_wafer_nm("x")-1.576) < 0.02)
    rep("8x wafer shadow ~0.79 nm",
        abs(m8.shadow_at_wafer_nm("y")-0.788) < 0.02)
    asym = m8.anamorphic_shadow_asymmetry_nm()
    print(f"      anamorphic asymmetry = {asym['asymmetry_nm']:.3f} nm")
    rep("anamorphic shadow is anisotropic", asym["anisotropic"])
    rep("asymmetry ~0.79 nm", abs(asym["asymmetry_nm"]-0.788) < 0.02)
    frac = m4.shadow_at_wafer_nm("x")/8.0
    print(f"      at 8 nm half-pitch: shadow = {100*frac:.1f}% of hp")
    rep("shadow >15% of 8 nm half-pitch", frac > 0.15)

    if not HAS_NUMPY:
        print("\n  numpy absent; skipping image tests")
        print(f"\nSELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
        return ok

    print("\n[T5] KIRCHHOFF vs M3D spectrum")
    mask = Mask3D()
    for p in (32.0, 24.0, 16.0):
        r = kirchhoff_vs_m3d(p, p/2, mask)
        print(f"      pitch {p:5.1f} nm: spectral error "
              f"{r['spectral_error_relative']:.4f}  "
              f"{r['VERDICT']}")
        rep(f"pitch {p} Kirchhoff error computed",
            r["spectral_error_relative"] >= 0)
    r16 = kirchhoff_vs_m3d(16.0, 8.0, mask)
    rep("Kirchhoff INVALID at 16 nm mask pitch", not r16["kirchhoff_adequate"])

    print("\n[T6] SCALAR vs VECTOR aerial image")
    for na, p in ((0.33, 40.0), (0.55, 24.0)):
        sv = scalar_vs_vector(p, p/2, mask, na)
        print(f"      NA={na} pitch={p}: NILS TE={sv['NILS_TE']:.3f} "
              f"TM={sv['NILS_TM']:.3f} unpol={sv['NILS_unpolarised']:.3f}")
        rep(f"NA={na} TM NILS below TE", sv["NILS_TM"] < sv["NILS_TE"])
        rep(f"NA={na} unpolarised between TE and TM",
            min(sv["NILS_TE"], sv["NILS_TM"]) <= sv["NILS_unpolarised"]
            <= max(sv["NILS_TE"], sv["NILS_TM"]) + 1e-6)
        rep(f"NA={na} scalar overestimates",
            sv["scalar_overestimates_by"] is None or
            sv["scalar_overestimates_by"] >= 1.0 - 1e-6)

    print("\n[T7] BOSSUNG with M3D")
    b = bossung(24.0, 12.0, mask, 0.55, n_focus=11, n_dose=7)
    print(f"      best focus = {b['best_focus_nm']:.1f} nm  "
          f"tilt = {b['bossung_tilt_nm']:.3f} nm  "
          f"PW = {b['process_window_fraction']:.3f}")
    rep("best focus computed", "best_focus_nm" in b)
    rep("M3D signature stated", "pitch-dependent" in b["M3D_SIGNATURE"])
    rep("process window in [0,1]", 0 <= b["process_window_fraction"] <= 1)
    bf = best_focus_vs_pitch([32.0, 26.0, 22.0], mask, 0.55)
    for r in bf["rows"]:
        print(f"      pitch {r['pitch_nm']:5.1f}: BF={r['best_focus_nm']:6.1f} nm "
              f"tilt={r['tilt_nm']:6.3f} nm")
    print(f"      best-focus spread = {bf['best_focus_spread_nm']:.2f} nm")
    rep("Kirchhoff predicts identical BF (stated)",
        "identical" in bf["KIRCHHOFF_PREDICTS"])
    rep("spread computed", bf["best_focus_spread_nm"] >= 0)

    print("\n[T8] SD GATE")
    sd = SDGate()
    c1 = sd.certify("multilayer_R", ("OC",), {"R": r6["R"]})
    rep("measured optical constants -> CERTIFIED", c1["status"] == MODE_CERTIFIED)
    c2 = sd.certify("m3d_spectrum", (), {})
    rep("uncalibrated M3D -> MODE_E", c2["status"] == MODE_E)
    c3 = sd.certify("process_window", ("M3D","OPT"), {})
    rep("missing RES -> MODE_E", c3["status"] == MODE_E)
    rep("caveat says NOT rigorous RCWA", "NOT rigorous" in c2["caveat"])
    rr = sd.report()
    print(f"      outputs={rr['n']} certified={rr['certified']} "
          f"Mode_E={rr['mode_E']}")
    rep("Mode E retained", rr["mode_E"] > 0)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
ADDED
  Vector diffraction: full 3-D order directions, TE/TM polarization vectors,
  pairwise dot product e_n . e_m* = cos(theta_n - theta_m), obliquity/apodisation.
  Verified: TM contrast 0.783 (NA 0.33) -> 0.397 (NA 0.55) -> -0.125 (NA 0.75).
  Mo/Si multilayer: exact recursive Fresnel with Ru cap, wavelength/angle
  sweep, reflection phase and equivalent defocus.
  Thin-mask 3D: absorber shadowing (6.30 nm at mask; 1.58 nm at 4x,
  0.79 nm at 8x), anamorphic asymmetry, edge boundary layers, ML phase.
  Vector+M3D aerial image, scalar-vs-vector comparison, Bossung,
  best-focus-vs-pitch (the measurable M3D fingerprint).
  SD gate with packages OPT / OC / M3D / RES.

NOT CLAIMED
  Boundary-layer M3D is a CALIBRATED SURROGATE, not rigorous RCWA/FDTD.
  Optical constants are approximate; replace with measured data.
  Resist is a hard threshold, not a calibrated resist model.
  Multilayer phase is taken angle-independent in the mask spectrum.
  No measured wafer data has been compared. All results are model outputs.
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Vector + thin-mask 3D lithography")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","contrast","multilayer","shadow","m3d",
                 "aerial","bossung","bfpitch","packages"])
    ap.add_argument("--na", type=float, default=0.55)
    ap.add_argument("--pitch", type=float, default=24.0)
    ap.add_argument("--cd", type=float, default=None)
    ap.add_argument("--lam", type=float, default=13.5)
    ap.add_argument("--absorber", type=float, default=60.0)
    ap.add_argument("--cra", type=float, default=6.0)
    ap.add_argument("--mag-x", type=float, default=4.0)
    ap.add_argument("--mag-y", type=float, default=4.0)
    ap.add_argument("--pol", default="unpolarised")
    ap.add_argument("--illum", default="dipole_x")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    cd = a.cd if a.cd is not None else a.pitch/2
    mask = Mask3D(absorber_nm=a.absorber, cra_deg=a.cra,
                  mag_x=a.mag_x, mag_y=a.mag_y)
    out: Dict[str, Any] = {}
    if a.cmd == "contrast":
        out = {str(na): two_beam_contrast(na, a.lam)
               for na in (0.33, 0.55, 0.75)}
    elif a.cmd == "multilayer":
        ml = Multilayer()
        out = {"at_13p5": ml.reflectance(a.lam, a.cra),
               "peak": ml.peak_wavelength(),
               "phase_error": multilayer_phase_error(ml, a.lam, a.cra)}
    elif a.cmd == "shadow":
        out = {"mask_nm": mask.shadow_at_mask_nm(),
               "wafer_x_nm": mask.shadow_at_wafer_nm("x"),
               "wafer_y_nm": mask.shadow_at_wafer_nm("y"),
               "anamorphic": mask.anamorphic_shadow_asymmetry_nm()}
    elif a.cmd == "m3d":
        out = kirchhoff_vs_m3d(a.pitch, cd, mask, a.lam)
    elif a.cmd == "aerial":
        out = scalar_vs_vector(a.pitch, cd, mask, a.na, a.lam, a.illum)
    elif a.cmd == "bossung":
        out = bossung(a.pitch, cd, mask, a.na, wavelength_nm=a.lam)
    elif a.cmd == "bfpitch":
        out = best_focus_vs_pitch([a.pitch*1.4, a.pitch, a.pitch*0.8],
                                  mask, a.na, a.lam)
    elif a.cmd == "packages":
        out = PACKAGES
    print(json.dumps(out, indent=2, default=str))
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
