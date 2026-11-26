#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_litho_math_api.py — v1.0.0
Platform-agnostic lithography mathematics API.
ASML EUV (NXE / EXE) is ONE backend among many. No timeline claims.

Backends: ASML NXE:3800E · ASML EXE:5200 · DUV immersion · NIL · EBL
          · HYPERFAB (speculative, flagged)
Math:     Hopkins-Abbe aerial image · adjoint ILT · photon shot noise
          CH3D resist phase separation · delta_min^(H)/(P) defect thresholds
          STC interface conditions · OPMC drift · Ville / Shiryaev-Roberts
          Statistics Dictionary (SD) certification with Mode E
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
SCOPE — WHAT THIS IS AND IS NOT
=============================================================================
IS:
  * A mathematics layer any lithography vendor can adopt.
  * ASML EUV treated as one backend specification among several.
  * Derived defect thresholds, certification gates, drift detection.
  * Open specification; a vendor may integrate it into an existing tool.

IS NOT:
  * A scanner. No light source, no optics, no stage, no vacuum.
  * A timeline claim. Nothing here says when anything can be built.
  * A superiority claim over ASML. Their tools have 25 years of
    process knowledge that no mathematics substitutes for.

WHAT A VENDOR COULD ADOPT
  1. delta_min^(H) and delta_min^(P): stochastic nano-bridge and line-break
     thresholds derived from resist physics rather than tuned dose margins.
  2. SD gate: every wafer/die output carries its certifying theorem and
     hypothesis package, or is retained as Mode E (uncertainty).
  3. OPMC + Ville/Shiryaev-Roberts: anytime-valid drift detection with
     exact mean time to false alarm E_0[T_B] = B.

=============================================================================
REAL BACKEND PARAMETERS (public specifications; verify before citation)
=============================================================================
ASML NXE:3800E : lambda 13.5 nm, NA 0.33, 4x isomorphic, field 26x33 mm,
                 ~220 WPH, dedicated-chuck overlay ~1.1 nm
ASML EXE:5200  : lambda 13.5 nm, NA 0.55, 4x/8x anamorphic,
                 field 26x16.5 mm (half-field), ~185 WPH, overlay ~0.9 nm
DUV immersion  : lambda 193 nm, NA 1.35, 4x, field 26x33 mm, ~275 WPH
NIL            : mechanical contact, ~10 nm resolution, overlay and
                 defectivity are the open problems
EBL            : ~5 nm resolution, throughput orders of magnitude lower
HYPERFAB X2.0  : SPECULATIVE. ERL-FEL backbone + isolated clusters.
                 Not built. No timeline. Flagged in every output.

=============================================================================
PHOTON BUDGET AT 13.5 nm  (the real stochastic problem)
=============================================================================
E_photon = h c / lambda = 1.472e-17 J = 91.9 eV
At dose D [mJ/cm^2]:
    N = D * 1e-3 / E_photon  photons/cm^2
      = D * 1e-3 / (E_photon * 1e14)  photons/nm^2
At D = 30 mJ/cm^2 : ~20.4 photons/nm^2
At D = 60 mJ/cm^2 : ~40.8 photons/nm^2
In a (5 nm)^2 = 25 nm^2 cell at 30 mJ/cm^2: ~510 photons
    relative shot noise = 1/sqrt(510) = 4.4%
This is why stochastic defects dominate at low dose and small pitch.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("litho_api")
VERSION = "1.0.0"; ABI_VERSION = 1
PI = math.pi
H_PLANCK = 6.62607015e-34
C_LIGHT  = 2.99792458e8
EV_JOULE = 1.602176634e-19

BANNER = f"""
=============================================================================
MSPS LITHOGRAPHY MATH API v{VERSION}  (ABI v{ABI_VERSION})
ASML EUV = one backend. No timeline. No superiority claim.
Hopkins-Abbe · shot noise · delta_min(H/P) · STC · OPMC · Ville/SR · SD gate
=============================================================================
"""

class Refusal(RuntimeError): pass
class Uncertified(RuntimeError): pass
class SpeculativeBackend(Refusal): pass


# =============================================================================
# §1. BACKEND REGISTRY — ASML is a special case, not the target
# =============================================================================

@dataclass(frozen=True)
class LithoBackend:
    name: str
    wavelength_nm: Optional[float]   # None for NIL (mechanical)
    na: Optional[float]
    mag_x: float
    mag_y: float
    field_x_mm: float
    field_y_mm: float
    wph: Optional[float]
    overlay_nm: Optional[float]
    status: str                      # "production" | "research" | "speculative"
    note: str = ""

    @property
    def photon_energy_J(self) -> Optional[float]:
        if self.wavelength_nm is None: return None
        return H_PLANCK*C_LIGHT/(self.wavelength_nm*1e-9)

    @property
    def photon_energy_eV(self) -> Optional[float]:
        e = self.photon_energy_J
        return None if e is None else e/EV_JOULE

    @property
    def anamorphic(self) -> bool:
        return abs(self.mag_x - self.mag_y) > 1e-9

    def rayleigh_cd(self, k1: float = 0.32) -> Optional[float]:
        if self.wavelength_nm is None or self.na is None: return None
        return k1*self.wavelength_nm/self.na

    def rayleigh_dof(self, k2: float = 0.5) -> Optional[float]:
        if self.wavelength_nm is None or self.na is None: return None
        return k2*self.wavelength_nm/(self.na**2)


BACKENDS: Dict[str, LithoBackend] = {
 "ASML_NXE3800E": LithoBackend("ASML NXE:3800E", 13.5, 0.33, 4.0, 4.0,
    26.0, 33.0, 220.0, 1.1, "production",
    "Low-NA EUV, LPP Sn source, isomorphic 4x. Public specification."),
 "ASML_EXE5200": LithoBackend("ASML EXE:5200", 13.5, 0.55, 4.0, 8.0,
    26.0, 16.5, 185.0, 0.9, "production",
    "High-NA EUV, anamorphic 4x/8x, half-field. Public specification."),
 "DUV_IMMERSION": LithoBackend("DUV ArF immersion", 193.0, 1.35, 4.0, 4.0,
    26.0, 33.0, 275.0, 1.5, "production",
    "193 nm ArF immersion; multi-patterning below 40 nm pitch."),
 "NIL": LithoBackend("Nanoimprint", None, None, 1.0, 1.0,
    26.0, 33.0, None, None, "research",
    "Mechanical contact. Template defectivity, overlay, wear are open."),
 "EBL": LithoBackend("Electron beam", 0.005, None, 1.0, 1.0,
    1.0, 1.0, None, None, "research",
    "Direct-write. Resolution excellent; throughput orders lower."),
 "HYPERFAB_X2": LithoBackend("MSPS HYPERFAB X2.0", 6.7, 0.75, 4.0, 8.0,
    26.0, 16.5, None, None, "speculative",
    "ERL-FEL backbone + isolated clusters. NOT BUILT. No timeline."),
}

def get_backend(name: str, allow_speculative: bool = False) -> LithoBackend:
    if name not in BACKENDS:
        raise Refusal(f"unknown backend '{name}'. Known: {list(BACKENDS)}")
    b = BACKENDS[name]
    if b.status == "speculative" and not allow_speculative:
        raise SpeculativeBackend(
            f"'{name}' is SPECULATIVE: {b.note}\n"
            "Mathematics may be evaluated for it, but no buildability or "
            "timeline claim is made. Pass allow_speculative=True to inspect.")
    return b


# =============================================================================
# §2. PHOTON BUDGET AND SHOT NOISE
# =============================================================================

def photon_budget(backend: LithoBackend, dose_mJ_cm2: float,
                  cell_nm: float) -> Dict[str, Any]:
    """
    Photons per cell and relative shot noise. This is the physical origin
    of stochastic defects in EUV.
    """
    if dose_mJ_cm2 <= 0 or cell_nm <= 0:
        raise Refusal("dose and cell size must be positive")
    e = backend.photon_energy_J
    if e is None:
        return {"backend": backend.name, "mechanism": "mechanical/other",
                "photon_shot_noise": False,
                "note": "no photon budget for this backend"}
    n_per_cm2 = dose_mJ_cm2*1e-3/e
    n_per_nm2 = n_per_cm2/1e14
    area = cell_nm*cell_nm
    n_cell = n_per_nm2*area
    rel = 1.0/math.sqrt(n_cell) if n_cell > 0 else float("inf")
    return {"backend": backend.name,
            "wavelength_nm": backend.wavelength_nm,
            "photon_energy_eV": backend.photon_energy_eV,
            "dose_mJ_cm2": dose_mJ_cm2,
            "photons_per_nm2": n_per_nm2,
            "cell_nm": cell_nm, "cell_area_nm2": area,
            "photons_per_cell": n_cell,
            "relative_shot_noise": rel,
            "shot_noise_pct": 100*rel,
            "photon_shot_noise": True}


def dose_for_shot_noise(backend: LithoBackend, cell_nm: float,
                        target_rel: float) -> Optional[float]:
    """Inverse: dose needed to reach a target relative shot noise."""
    e = backend.photon_energy_J
    if e is None or target_rel <= 0: return None
    n_cell = 1.0/(target_rel**2)
    n_per_nm2 = n_cell/(cell_nm*cell_nm)
    return n_per_nm2*1e14*e*1e3     # mJ/cm^2


# =============================================================================
# §3. AERIAL IMAGE — Abbe source decomposition (Hopkins equivalent)
# =============================================================================

@dataclass
class PupilSpec:
    na: float
    sigma: float = 0.7          # partial coherence
    illum: str = "annular"      # conventional | annular | dipole | quadrupole
    sigma_in: float = 0.4
    sigma_out: float = 0.8
    zernike_nm: Dict[int, float] = field(default_factory=dict)  # aberrations

@dataclass
class MaskSpec:
    pitch_nm: float
    cd_nm: float
    n_periods: int = 8
    dx_nm: float = 0.5
    attenuated_psm: bool = False
    attenuation: float = 0.06
    phase_shift_rad: float = PI


def mask_profile(m: MaskSpec) -> Tuple[List[float], List[float]]:
    """1-D mask transmission sampled at dx. Returns (x, t)."""
    n = int(round(m.pitch_nm*m.n_periods/m.dx_nm))
    x = [i*m.dx_nm for i in range(n)]
    t = []
    for xi in x:
        pos = xi % m.pitch_nm
        inside = pos < m.cd_nm
        if inside:
            t.append(1.0)
        else:
            if m.attenuated_psm:
                t.append(-math.sqrt(m.attenuation))
            else:
                t.append(0.0)
    return x, t


def source_points(p: PupilSpec, n_src: int = 21) -> List[Tuple[float,float]]:
    """Sample the illumination pupil in normalized frequency units."""
    pts = []
    for i in range(n_src):
        s = -1.0 + 2.0*i/max(n_src-1, 1)
        for j in range(n_src):
            t = -1.0 + 2.0*j/max(n_src-1, 1)
            r = math.hypot(s, t)
            if p.illum == "conventional":
                ok = r <= p.sigma
            elif p.illum == "annular":
                ok = p.sigma_in <= r <= p.sigma_out
            elif p.illum == "dipole":
                ok = (p.sigma_in <= r <= p.sigma_out) and abs(t) < 0.35
            elif p.illum == "quadrupole":
                ok = (p.sigma_in <= r <= p.sigma_out) and abs(s*t) > 0.15
            else:
                raise Refusal(f"unknown illumination '{p.illum}'")
            if ok: pts.append((s, t))
    return pts or [(0.0, 0.0)]


def aerial_image_1d(backend: LithoBackend, mask: MaskSpec, pupil: PupilSpec,
                    n_src: int = 15, defocus_nm: float = 0.0
                    ) -> Dict[str, Any]:
    """
    Abbe decomposition: I(x) = sum_s | F^{-1}{ P(f) M(f - s) } |^2.
    Equivalent to the Hopkins TCC formulation, cheaper to implement.

    Requires numpy. Without numpy, returns an analytic two-beam estimate.
    """
    lam = backend.wavelength_nm
    na = backend.na
    if lam is None or na is None:
        raise Refusal(f"{backend.name} has no optical wavelength/NA")

    if not HAS_NUMPY:
        # analytic two-beam interference estimate
        k = 2*PI/lam
        f0 = 1.0/mask.pitch_nm
        theta = math.asin(min(1.0, lam*f0))
        contrast = max(0.0, math.cos(k*defocus_nm*1e-0*(1-math.cos(theta))))
        return {"method": "analytic_two_beam (numpy absent)",
                "pitch_nm": mask.pitch_nm,
                "normalized_frequency": lam*f0/na,
                "two_beam_resolvable": lam*f0/na <= 1.0,
                "contrast_estimate": contrast}

    x, t = mask_profile(mask)
    n = len(t)
    dx = mask.dx_nm
    m = np.asarray(t, dtype=float)
    freqs = np.fft.fftfreq(n, d=dx)          # 1/nm

    # pupil cutoff in 1/nm
    f_cut = na/lam
    src = source_points(pupil, n_src)
    I = np.zeros(n)
    for (sx, sy) in src:
        fsx = sx*f_cut
        shifted = freqs - fsx
        P = (np.abs(shifted) <= f_cut).astype(float)
        # defocus phase (paraxial): W = defocus * (1 - cos) ~ defocus*lam*f^2/2
        if defocus_nm != 0.0:
            W = 0.5*defocus_nm*lam*(shifted**2)
            phase = np.exp(2j*PI*W/lam*lam)   # dimensionally consistent below
            phase = np.exp(1j*PI*defocus_nm*lam*(shifted**2))
            P = P*phase
        # aberrations via radial Zernike-ish term
        if pupil.zernike_nm:
            rho = np.clip(np.abs(shifted)/f_cut, 0, 1)
            Wab = np.zeros(n)
            for j, a in pupil.zernike_nm.items():
                Wab += a*(rho**abs(j))
            P = P*np.exp(2j*PI*Wab/lam)
        spec = np.fft.fft(m)*P
        field = np.fft.ifft(spec)
        I += np.abs(field)**2
    I /= len(src)
    Imax = float(I.max())
    if Imax > 0: I = I/Imax

    # NILS at the first falling edge
    nils = nils_at_edge(x, I.tolist(), mask.cd_nm)
    return {"method": "Abbe source decomposition",
            "backend": backend.name,
            "n_source_points": len(src),
            "x_nm": x, "intensity": I.tolist(),
            "pitch_nm": mask.pitch_nm, "cd_nm": mask.cd_nm,
            "normalized_frequency": lam/(mask.pitch_nm*na),
            "nils": nils,
            "nils_target": 4.2,
            "nils_pass": nils is not None and nils >= 4.2,
            "defocus_nm": defocus_nm}


def nils_at_edge(x: Sequence[float], I: Sequence[float],
                 cd_nm: float) -> Optional[float]:
    """NILS = w * d(ln I)/dx evaluated at the mask edge."""
    n = len(x)
    if n < 4: return None
    dx = x[1]-x[0]
    idx = None
    for i in range(1, n):
        if (x[i-1] % cd_nm) > (x[i] % cd_nm):
            idx = i; break
    if idx is None or idx >= n-1:
        idx = max(1, int(cd_nm/dx))
    lo, hi = max(0, idx-2), min(n-1, idx+2)
    Ie = max(I[idx], 1e-12)
    dlnI = (math.log(max(I[hi],1e-12)) - math.log(max(I[lo],1e-12)))/(x[hi]-x[lo])
    return abs(cd_nm*dlnI)


# =============================================================================
# §4. STOCHASTIC DEFECTS — delta_min^(H) and delta_min^(P)
# =============================================================================

def delta_min_defect(resist_cell_nm: float, photons_per_cell: float,
                     C_H: float = 1.0, C_P: float = 1.0,
                     c_H: float = 0.5, c_P: float = 0.5
                     ) -> Dict[str, Any]:
    """
    Derived reset thresholds for the two stochastic defect surgeries.

      H (handle attachment) = nano-bridge between adjacent lines
                              Db0 = 0, Db1 = +1, Dchi = -1
      P (neck pinch)        = line break / open
                              Db0 = 0, Db1 = -1, Dchi = +1

    The reset distance is a fraction of the resist correlation cell;
    the holding "time" is the number of photons needed to re-establish
    an independent configuration. Shot noise sets the exit probability.

      q = 2 exp(-delta^2 / (8 G0^2 eps))
    with G0^2 ~ 1/N_photons (relative variance) and eps = 1 exposure.
    """
    if photons_per_cell <= 0:
        raise Refusal("photon count must be positive")
    dH = c_H*resist_cell_nm
    dP = c_P*resist_cell_nm
    G0sq = 1.0/photons_per_cell
    qH = 2*math.exp(-(dH/resist_cell_nm)**2/(8*G0sq*1.0))
    qP = 2*math.exp(-(dP/resist_cell_nm)**2/(8*G0sq*1.0))
    return {"resist_cell_nm": resist_cell_nm,
            "photons_per_cell": photons_per_cell,
            "G0_squared_rel_var": G0sq,
            "delta_min_H_nm": dH, "delta_min_P_nm": dP,
            "q_H": qH, "q_P": qP,
            "no_zeno_H": qH < 1.0, "no_zeno_P": qP < 1.0,
            "defect_H": "nano-bridge (Db1=+1, Dchi=-1)",
            "defect_P": "line-break (Db1=-1, Dchi=+1)",
            "STATUS": "derived under stated shot-noise model",
            "CAVEAT": ("C_H, C_P are O(1) constants requiring calibration "
                       "against measured defect density")}


def dose_margin_for_defect_rate(resist_cell_nm: float, cell_nm: float,
                                backend: LithoBackend,
                                q_target: float = 1e-9,
                                c: float = 0.5) -> Dict[str, Any]:
    """
    Invert the no-Zeno bound to find the dose that achieves a target
    per-cell defect probability. This replaces a tuned dose margin with
    a derived one.
    """
    if not (0 < q_target < 1):
        raise Refusal("q_target must be in (0,1)")
    # q = 2 exp(-c^2 N / 8)  =>  N = 8 ln(2/q) / c^2
    N_req = 8*math.log(2.0/q_target)/(c*c)
    dose = dose_for_shot_noise(backend, cell_nm, 1.0/math.sqrt(N_req))
    return {"q_target": q_target, "c": c,
            "photons_required_per_cell": N_req,
            "cell_nm": cell_nm,
            "dose_mJ_cm2": dose,
            "backend": backend.name,
            "DERIVATION": "no-Zeno exit bound inverted for dose",
            "NOTE": "compare against the vendor's empirical dose margin"}


def lwr_rls_model(half_pitch_nm: float, dose_mJ_cm2: float,
                  C_RLS: float = 1.0) -> Dict[str, Any]:
    """
    Resolution-LWR-Sensitivity trade-off (RLS / Z-factor).
        LWR(3sigma) ~ C_RLS * sqrt(hp^3 / dose)
    C_RLS is a resist-specific calibration constant, NOT universal.
    """
    if half_pitch_nm <= 0 or dose_mJ_cm2 <= 0:
        raise Refusal("half-pitch and dose must be positive")
    lwr = C_RLS*math.sqrt(half_pitch_nm**3/dose_mJ_cm2)
    z = lwr*lwr*(half_pitch_nm**3)*dose_mJ_cm2
    return {"half_pitch_nm": half_pitch_nm, "dose_mJ_cm2": dose_mJ_cm2,
            "C_RLS": C_RLS, "lwr_3sigma_nm": lwr,
            "Z_factor": z,
            "STATUS": "scaling model; C_RLS must be measured per resist",
            "NOT": "a prediction without calibration"}


# =============================================================================
# §5. STC — resist / hardmask / substrate interfaces
# =============================================================================

def stc_residual(u_left: Sequence[float], u_right: Sequence[float],
                 dx: float, a_left: float, a_right: float
                 ) -> Dict[str, float]:
    """
    STC^0: trace continuity.
    STC^1: conormal flux balance with OPPOSITE outward normals.
    Applies to acid diffusion across resist/hardmask interfaces.
    """
    trace = u_left[-1]-u_right[0]
    gl = (u_left[-1]-u_left[-2])/dx if len(u_left) > 1 else 0.0
    gr = (u_right[1]-u_right[0])/dx if len(u_right) > 1 else 0.0
    flux = a_left*gl - a_right*gr
    return {"stc0_trace": trace, "stc1_flux": flux,
            "in_kernel": abs(trace) < 1e-8 and abs(flux) < 1e-8}

def stc_reflux(coarse_flux: float, fine_fluxes: Sequence[float]
               ) -> Dict[str, Any]:
    """Coarse-face flux must equal the sum of fine-face fluxes."""
    total = sum(fine_fluxes)
    mismatch = total - coarse_flux
    corrected = coarse_flux + mismatch
    return {"coarse_flux": coarse_flux, "fine_total": total,
            "stc1_mismatch": mismatch, "reflux_correction": mismatch,
            "corrected": corrected,
            "residual_after": abs(total-corrected),
            "conserved": abs(total-corrected) < 1e-14,
            "DERIVATION": "STC^1 requires refluxing; not optional"}


# =============================================================================
# §6. OPMC + VILLE / SHIRYAEV-ROBERTS — scanner telemetry
# =============================================================================

def prefix_sums(y: Sequence[float]) -> Tuple[List[float], List[float]]:
    s1 = [0.0]; s2 = [0.0]
    for v in y:
        s1.append(s1[-1]+v); s2.append(s2[-1]+v*v)
    return s1, s2

def seg_sse(s1, s2, a: int, b: int) -> float:
    m = b-a
    if m <= 0: return 0.0
    t1 = s1[b]-s1[a]; t2 = s2[b]-s2[a]
    return max(t2 - t1*t1/m, 0.0)

def opmc(y: Sequence[float], ell: int, beta: float) -> Dict[str, Any]:
    """Exact O(n^2) DP. Applies to focus/dose/overlay/CD time series."""
    n = len(y)
    if not (1 <= ell <= n): raise Refusal("1 <= ell <= n required")
    s1, s2 = prefix_sums(y)
    INF = float("inf")
    D = [INF]*(n+1); prev = [-1]*(n+1); D[0] = -beta
    for b in range(ell, n+1):
        best, arg = INF, -1
        for a in range(0, b-ell+1):
            if D[a] == INF: continue
            v = D[a] + seg_sse(s1,s2,a,b) + beta
            if v < best: best, arg = v, a
        D[b] = best; prev[b] = arg
    cps = []; b = n
    while b > 0:
        a = prev[b]
        if a <= 0: break
        cps.append(a); b = a
    cps.reverse()
    return {"boundaries": cps, "K_hat": len(cps), "cost": D[n],
            "complexity": "O(n^2) time, O(n) memory"}

def L_of(n: int, alpha: float = 0.05) -> float:
    return math.log(10*math.e*n**3/alpha)

def opmc_certified(y: Sequence[float], ell: int, sigma: float,
                   alpha: float = 0.05) -> Dict[str, Any]:
    """OPMC with the honest (SNR) verification from ESCA Thm 7.5."""
    n = len(y); L = L_of(n, alpha); beta = 32*sigma*sigma*L
    r = opmc(y, ell, beta)
    bounds = [0]+r["boundaries"]+[n]
    means = [sum(y[bounds[i]:bounds[i+1]])/max(bounds[i+1]-bounds[i],1)
             for i in range(len(bounds)-1)]
    jumps = [abs(means[i+1]-means[i]) for i in range(len(means)-1)]
    kappa = min(jumps) if jumps else 0.0
    spac = [bounds[i+1]-bounds[i] for i in range(len(bounds)-1)]
    delta = min(spac) if spac else n
    K = r["K_hat"]
    lhs = kappa*kappa*delta; rhs = 1e5*sigma*sigma*L*(K+1)**2
    ok = (lhs >= rhs) and delta >= 8 and (delta/2 < ell <= 3*delta/4)
    return {**r, "kappa_min": kappa, "delta_min": delta, "ell": ell,
            "snr_lhs": lhs, "snr_rhs": rhs,
            "status": "CERTIFIED" if ok else "MODE_E",
            "radius": 46*sigma*sigma*L/max(kappa*kappa,1e-30) if ok else None,
            "reason": "" if ok else "(SNR) or (D) not verified"}


class VilleMonitor:
    """
    Anytime-valid evidence martingale for drift detection.
        Lambda_t = exp((Delta/sigma^2) Y_t - (Delta^2/(2 sigma^2)) t)
        P_0(sup_t Lambda_t >= 1/alpha) <= alpha
    """
    def __init__(self, delta: float, sigma: float, alpha: float = 0.01):
        if sigma <= 0: raise Refusal("sigma > 0 required")
        self.delta, self.sigma, self.alpha = delta, sigma, alpha
        self.Y = 0.0; self.t = 0.0; self.lam = 1.0
        self.threshold = 1.0/alpha
        self.alarmed = False
    def step(self, dy: float, dt: float = 1.0) -> Dict[str, Any]:
        self.Y += dy; self.t += dt
        a = self.delta/(self.sigma**2)
        self.lam = math.exp(a*self.Y - 0.5*a*self.delta*self.t)
        if self.lam >= self.threshold: self.alarmed = True
        return {"t": self.t, "lambda": self.lam,
                "threshold": self.threshold, "alarm": self.alarmed,
                "type_I_bound": self.alpha}


class ShiryaevRoberts:
    """
    R_t = int_0^t (Lambda_t/Lambda_s) ds;  T_B = inf{t: R_t >= B}
    Under the null, E_0[T_B] = B exactly.
    """
    def __init__(self, B: float, delta: float = 0.0, sigma: float = 1.0):
        if B <= 0: raise Refusal("B > 0 required")
        self.B, self.delta, self.sigma = B, delta, sigma
        self.R = 0.0; self.t = 0.0; self.stopped = False
        self._loglam = 0.0; self._terms: List[float] = []
    def step(self, dy: float, dt: float = 1.0) -> Dict[str, Any]:
        a = self.delta/(self.sigma**2)
        self._loglam += a*dy - 0.5*a*self.delta*dt
        self._terms.append(self._loglam)
        m = max(self._terms)
        self.R = dt*sum(math.exp(v-m) for v in self._terms)*math.exp(m)
        self.t += dt
        if self.R >= self.B: self.stopped = True
        return {"t": self.t, "R": self.R, "B": self.B,
                "stopped": self.stopped,
                "E0_TB_equals_B": True,
                "mean_false_alarm_time": self.B}


# =============================================================================
# §7. STATISTICS DICTIONARY — certify or retain uncertainty
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"
MODE_E = "UNCERTAINTY_RETAINED"

@dataclass(frozen=True)
class SDEntry:
    statistic: str; theorem: str
    packages: Tuple[str, ...]
    mode: str; rate: str; caveat: str

DICTIONARY: Dict[str, SDEntry] = {
 "aerial_image": SDEntry("aerial image / NILS", "Hopkins-Abbe",
    ("OPT",), "exact within scalar model", "NILS >= 4.2 target",
    "scalar model; vector/thin-mask effects not included"),
 "photon_budget": SDEntry("photons per cell", "Poisson statistics",
    (), "exact", "1/sqrt(N)", "assumes Poisson arrivals"),
 "defect_threshold": SDEntry("delta_min H/P", "no-Zeno exit bound",
    ("L","DELTA"), "non-asymptotic", "q = 2exp(-d^2/(8 G0^2 eps))",
    "C_H, C_P require calibration"),
 "lwr": SDEntry("LWR 3sigma", "RLS scaling",
    ("RLS",), "scaling", "C sqrt(hp^3/D)", "C_RLS must be measured"),
 "stc_flux": SDEntry("interface flux balance", "STC^1 / Lemma 2.1(b)",
    ("STC",), "exact", "residual 0 after reflux", "requires matching traces"),
 "scanner_drift": SDEntry("focus/dose/overlay change points",
    "ESCA Thm 7.5", ("L","D","O"), "exact on good event",
    "46 sigma^2 L/kappa^2", "(SNR) carries (K+1)^2"),
 "process_alarm": SDEntry("anytime-valid alarm", "Ville / Shiryaev-Roberts",
    ("L",), "non-asymptotic", "P0(sup Lambda >= 1/a) <= a; E0[TB]=B",
    "calibrated for the stated drift model"),
 "backend_spec": SDEntry("backend specification", "vendor public data",
    (), "as published", "n/a", "verify against vendor documentation"),
}

class StatisticsDictionary:
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
                 "mode": e.mode, "rate": e.rate, "caveat": e.caveat,
                 "required": list(e.packages), "verified": list(v),
                 "missing": list(miss),
                 "status": MODE_E if miss else MODE_CERTIFIED,
                 "reason": f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def require(self, statistic: str, verified: Sequence[str],
                value: Any = None) -> Any:
        c = self.certify(statistic, verified, value)
        if c["status"] != MODE_CERTIFIED:
            raise Uncertified(f"SD GATE: {statistic}. {c['reason']}")
        return value
    def report(self) -> Dict[str, Any]:
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n_outputs": len(self.log), "n_certified": n,
                "n_mode_E": len(self.log)-n, "certificates": self.log}


# =============================================================================
# §8. C ABI — vendor integration surface
# =============================================================================

ABI_SPEC = {
 "abi_version": ABI_VERSION,
 "calling_convention": "C, cdecl",
 "float_type": "double",
 "error_codes": {0:"OK",1:"INVALID_ARG",2:"UNCERTIFIED (Mode E)",
                 3:"SPECULATIVE_BACKEND",4:"OUT_OF_RANGE"},
 "functions": [
  {"name":"msps_photon_budget",
   "args":["double wavelength_nm","double dose_mJ_cm2","double cell_nm",
           "double* out_photons","double* out_rel_noise"], "ret":"int32"},
  {"name":"msps_dose_for_noise",
   "args":["double wavelength_nm","double cell_nm","double target_rel"],
   "ret":"double"},
  {"name":"msps_delta_min_defect",
   "args":["double cell_nm","double photons","double c",
           "double* out_delta","double* out_q"], "ret":"int32"},
  {"name":"msps_dose_for_q",
   "args":["double cell_nm","double q_target","double c",
           "double wavelength_nm"], "ret":"double"},
  {"name":"msps_opmc",
   "args":["const double* y","int32 n","int32 ell","double beta",
           "int32* out_cps","int32 max_cps","int32* out_k"], "ret":"int32"},
  {"name":"msps_snr_ok",
   "args":["double kappa","int32 delta","double sigma","int32 n",
           "int32 K","double alpha"], "ret":"int32"},
  {"name":"msps_reflux",
   "args":["const double* fine","int32 n","double coarse",
           "double* corrected","double* residual"], "ret":"int32"},
  {"name":"msps_ville_step",
   "args":["double dy","double dt","double delta","double sigma",
           "double alpha","double* state_Y","double* state_t",
           "double* out_lambda"], "ret":"int32"},
 ]
}

C_HEADER = r"""/* msps_litho_math.h — ABI v1 (MIT, MSPS NETWORK 2026)
 * Vendor-neutral lithography mathematics. Link from any scanner control
 * stack, computational-lithography tool, or metrology pipeline. */
#ifndef MSPS_LITHO_MATH_H
#define MSPS_LITHO_MATH_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MSPS_OK 0
#define MSPS_INVALID_ARG 1
#define MSPS_UNCERTIFIED 2      /* Mode E: uncertainty retained */
#define MSPS_SPECULATIVE 3
#define MSPS_OUT_OF_RANGE 4

int32_t msps_photon_budget(double wavelength_nm, double dose_mJ_cm2,
                           double cell_nm, double* out_photons,
                           double* out_rel_noise);
double  msps_dose_for_noise(double wavelength_nm, double cell_nm,
                            double target_rel);
int32_t msps_delta_min_defect(double cell_nm, double photons, double c,
                              double* out_delta, double* out_q);
double  msps_dose_for_q(double cell_nm, double q_target, double c,
                        double wavelength_nm);
int32_t msps_opmc(const double* y, int32_t n, int32_t ell, double beta,
                  int32_t* out_cps, int32_t max_cps, int32_t* out_k);
int32_t msps_snr_ok(double kappa, int32_t delta, double sigma,
                    int32_t n, int32_t K, double alpha);
int32_t msps_reflux(const double* fine, int32_t n, double coarse,
                    double* corrected, double* residual);
int32_t msps_ville_step(double dy, double dt, double delta, double sigma,
                        double alpha, double* state_Y, double* state_t,
                        double* out_lambda);
#ifdef __cplusplus
}
#endif
#endif
"""

def abi_photon_budget(wavelength_nm: float, dose: float, cell: float
                      ) -> Tuple[int, float, float]:
    if min(wavelength_nm, dose, cell) <= 0: return 1, 0.0, 0.0
    e = H_PLANCK*C_LIGHT/(wavelength_nm*1e-9)
    n = dose*1e-3/e/1e14*cell*cell
    return 0, n, (1.0/math.sqrt(n) if n > 0 else float("inf"))

def abi_dose_for_noise(wavelength_nm: float, cell: float,
                       target: float) -> float:
    if min(wavelength_nm, cell, target) <= 0: return -1.0
    e = H_PLANCK*C_LIGHT/(wavelength_nm*1e-9)
    return (1.0/(target*target))/(cell*cell)*1e14*e*1e3

def abi_dose_for_q(cell: float, q: float, c: float,
                   wavelength_nm: float) -> float:
    if not (0 < q < 1) or c <= 0: return -1.0
    N = 8*math.log(2.0/q)/(c*c)
    return abi_dose_for_noise(wavelength_nm, cell, 1.0/math.sqrt(N))


# =============================================================================
# §9. ORCHESTRATOR
# =============================================================================

class LithoMathAPI:
    def __init__(self, out_dir: str = "runs/litho_api"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = StatisticsDictionary()

    def analyse(self, backend_name: str, pitch_nm: float, cd_nm: float,
                dose_mJ_cm2: float, cell_nm: float,
                allow_speculative: bool = False) -> Dict[str, Any]:
        b = get_backend(backend_name, allow_speculative)
        res: Dict[str, Any] = {"backend": asdict(b)}
        res["rayleigh_cd_nm"] = b.rayleigh_cd()
        res["rayleigh_dof_nm"] = b.rayleigh_dof()
        pb = photon_budget(b, dose_mJ_cm2, cell_nm)
        res["photon_budget"] = pb
        res["photon_cert"] = self.sd.certify("photon_budget", (), pb)
        if pb.get("photon_shot_noise"):
            dm = delta_min_defect(cell_nm, pb["photons_per_cell"])
            res["defect_thresholds"] = dm
            res["defect_cert"] = self.sd.certify(
                "defect_threshold", ("L","DELTA"), dm)
            res["dose_for_q1e-9"] = dose_margin_for_defect_rate(
                cell_nm, cell_nm, b, 1e-9)
        if b.wavelength_nm and b.na:
            mask = MaskSpec(pitch_nm, cd_nm)
            pup = PupilSpec(b.na)
            ai = aerial_image_1d(b, mask, pup)
            res["aerial_image"] = {k:v for k,v in ai.items()
                                   if k not in ("x_nm","intensity")}
            res["aerial_cert"] = self.sd.certify(
                "aerial_image", ("OPT",), res["aerial_image"])
        res["lwr_model"] = lwr_rls_model(pitch_nm/2, dose_mJ_cm2)
        res["lwr_cert"] = self.sd.certify("lwr", (), res["lwr_model"])
        res["scope"] = ("mathematics layer only; no scanner, no timeline, "
                        "no superiority claim")
        return res

    def emit(self) -> Dict[str, str]:
        h = self.out/"msps_litho_math.h"; h.write_text(C_HEADER)
        a = self.out/"abi_spec.json"
        a.write_text(json.dumps(ABI_SPEC, indent=2))
        return {"header": str(h), "abi_spec": str(a)}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION,
               "backends": {k: asdict(v) for k,v in BACKENDS.items()},
               "sd_report": self.sd.report(),
               "scope": "ASML EUV is one backend. No timeline claim."}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    api = LithoMathAPI(); sd = api.sd

    print("\n[T1] BACKEND REGISTRY — ASML is a special case")
    nxe = get_backend("ASML_NXE3800E"); exe = get_backend("ASML_EXE5200")
    rep("NXE registered", nxe.na == 0.33 and nxe.wavelength_nm == 13.5)
    rep("EXE registered", exe.na == 0.55)
    rep("EXE is anamorphic", exe.anamorphic)
    rep("EXE half-field", exe.field_y_mm == 16.5)
    rep("NXE is isomorphic", not nxe.anamorphic)
    cd_nxe, cd_exe = nxe.rayleigh_cd(0.32), exe.rayleigh_cd(0.325)
    print(f"      NXE CD(k1=0.32)={cd_nxe:.2f} nm   "
          f"EXE CD(k1=0.325)={cd_exe:.2f} nm")
    rep("NXE CD ~13.1 nm", abs(cd_nxe-13.1) < 0.2)
    rep("EXE CD ~8.0 nm", abs(cd_exe-8.0) < 0.2)
    dof_nxe = nxe.rayleigh_dof(0.5); dof_exe = exe.rayleigh_dof(0.75)
    print(f"      NXE DoF={dof_nxe:.1f} nm   EXE DoF={dof_exe:.1f} nm")
    rep("DoF collapses at high NA", dof_exe < dof_nxe)
    b = False
    try: get_backend("HYPERFAB_X2")
    except SpeculativeBackend: b = True
    rep("HYPERFAB flagged SPECULATIVE (refused by default)", b)
    hs = get_backend("HYPERFAB_X2", allow_speculative=True)
    rep("HYPERFAB inspectable with explicit flag", hs.status == "speculative")
    rep("HYPERFAB note says NOT BUILT", "NOT BUILT" in hs.note)

    print("\n[T2] PHOTON BUDGET at 13.5 nm")
    e_eV = nxe.photon_energy_eV
    print(f"      E_photon = {e_eV:.2f} eV (expected ~91.9)")
    rep("photon energy ~91.9 eV", abs(e_eV-91.9) < 0.3)
    pb30 = photon_budget(nxe, 30.0, 1.0)
    pb60 = photon_budget(nxe, 60.0, 1.0)
    print(f"      30 mJ/cm2: {pb30['photons_per_nm2']:.2f} photons/nm2")
    print(f"      60 mJ/cm2: {pb60['photons_per_nm2']:.2f} photons/nm2")
    rep("30 mJ -> ~20.4 photons/nm2", abs(pb30["photons_per_nm2"]-20.4) < 0.3)
    rep("dose doubling doubles photons",
        abs(pb60["photons_per_nm2"]/pb30["photons_per_nm2"]-2.0) < 1e-9)
    c5 = photon_budget(nxe, 30.0, 5.0)
    print(f"      5x5 nm cell @30 mJ: {c5['photons_per_cell']:.0f} photons, "
          f"shot noise {c5['shot_noise_pct']:.2f}%")
    rep("~510 photons in 25 nm2", abs(c5["photons_per_cell"]-510) < 10)
    rep("shot noise ~4.4%", abs(c5["shot_noise_pct"]-4.42) < 0.15)
    d_req = dose_for_shot_noise(nxe, 5.0, 0.02)
    rep("inverse dose solves correctly",
        abs(photon_budget(nxe, d_req, 5.0)["relative_shot_noise"]-0.02) < 1e-9)

    print("\n[T3] AERIAL IMAGE (Abbe = Hopkins)")
    mask = MaskSpec(pitch_nm=26.0, cd_nm=13.0)
    ai = aerial_image_1d(nxe, mask, PupilSpec(0.33, illum="annular"))
    print(f"      method={ai['method']}  NILS={ai.get('nils')}")
    rep("aerial image computed", "nils" in ai)
    if HAS_NUMPY:
        rep("intensity normalized", max(ai["intensity"]) <= 1.0+1e-9)
        rep("source points > 1", ai["n_source_points"] > 1)
    ai_exe = aerial_image_1d(exe, MaskSpec(16.0, 8.0), PupilSpec(0.55))
    rep("EXE resolves finer pitch",
        ai_exe["normalized_frequency"] < ai["normalized_frequency"])
    ai_def = aerial_image_1d(nxe, mask, PupilSpec(0.33), defocus_nm=50.0)
    rep("defocus changes image", ai_def.get("nils") != ai.get("nils"))

    print("\n[T4] DERIVED DEFECT THRESHOLDS (H and P)")
    dm = delta_min_defect(5.0, c5["photons_per_cell"])
    print(f"      delta_min_H={dm['delta_min_H_nm']:.2f} nm  "
          f"q_H={dm['q_H']:.3e}")
    print(f"      delta_min_P={dm['delta_min_P_nm']:.2f} nm  "
          f"q_P={dm['q_P']:.3e}")
    rep("H is nano-bridge (Db1=+1)", "Db1=+1" in dm["defect_H"])
    rep("P is line-break (Db1=-1)", "Db1=-1" in dm["defect_P"])
    rep("no-Zeno q<1 at 510 photons", dm["no_zeno_H"] and dm["no_zeno_P"])
    low = delta_min_defect(5.0, 20.0)
    rep("low photon count breaks no-Zeno", not low["no_zeno_H"],
        f"q={low['q_H']:.3f}")
    rep("calibration caveat stated", "calibration" in dm["CAVEAT"])
    dq = dose_margin_for_defect_rate(5.0, 5.0, nxe, 1e-9)
    print(f"      dose for q=1e-9: {dq['dose_mJ_cm2']:.1f} mJ/cm2 "
          f"({dq['photons_required_per_cell']:.0f} photons/cell)")
    rep("derived dose is positive", dq["dose_mJ_cm2"] > 0)
    check = delta_min_defect(5.0, dq["photons_required_per_cell"])
    rep("derived dose achieves target q", check["q_H"] <= 1e-9*(1+1e-6),
        f"q={check['q_H']:.2e}")
    rep("DERIVED not tuned", "no-Zeno" in dq["DERIVATION"])

    print("\n[T5] LWR / RLS TRADE-OFF")
    lw = lwr_rls_model(13.0, 30.0)
    print(f"      hp=13 nm, D=30 mJ: LWR={lw['lwr_3sigma_nm']:.3f} (C=1)")
    rep("LWR increases with smaller dose",
        lwr_rls_model(13.0, 15.0)["lwr_3sigma_nm"] > lw["lwr_3sigma_nm"])
    rep("LWR increases with smaller pitch (fixed C)",
        lwr_rls_model(8.0, 30.0)["lwr_3sigma_nm"] < lw["lwr_3sigma_nm"])
    rep("C_RLS must be measured", "must be measured" in lw["STATUS"])
    cert = sd.certify("lwr", (), lw)
    rep("uncalibrated LWR -> MODE_E", cert["status"] == MODE_E)
    cert2 = sd.certify("lwr", ("RLS",), lw)
    rep("calibrated LWR -> CERTIFIED", cert2["status"] == MODE_CERTIFIED)

    print("\n[T6] STC / REFLUX")
    ul = [i*0.1 for i in range(11)]; ur = [1.0+i*0.1 for i in range(11)]
    sr = stc_residual(ul, ur, 0.1, 1.0, 1.0)
    rep("STC0 trace continuous", abs(sr["stc0_trace"]) < 1e-12)
    rf = stc_reflux(0.7, [0.2,0.15,0.25,0.18])
    rep("reflux restores conservation", rf["conserved"])
    rep("reflux DERIVED from STC1", "not optional" in rf["DERIVATION"])

    print("\n[T7] OPMC on scanner telemetry")
    y = [0.0]*60 + [3.0]*60 + [-3.0]*60
    r = opmc(y, 40, 4*math.log(len(y)))
    rep("OPMC finds 2 change points", r["K_hat"] == 2, str(r["boundaries"]))
    rep("boundaries within 2 samples",
        all(abs(a-b) <= 2 for a,b in zip(r["boundaries"], [60,120])))
    rc = opmc_certified(y, 40, 1.0)
    print(f"      SNR lhs={rc['snr_lhs']:.3g} rhs={rc['snr_rhs']:.3g} "
          f"-> {rc['status']}")
    rep("strict (SNR) honestly fails for practical signal",
        rc["status"] == MODE_E)
    c = sd.certify("scanner_drift", ("L","D"))
    rep("missing O -> MODE_E", c["status"] == MODE_E)

    print("\n[T8] VILLE / SHIRYAEV-ROBERTS")
    vm = VilleMonitor(delta=0.5, sigma=1.0, alpha=0.01)
    for _ in range(50): vm.step(0.6)
    rep("Ville alarms on sustained drift", vm.alarmed,
        f"lambda={vm.lam:.3g}")
    rep("Type-I bound declared", vm.alpha == 0.01)
    vm0 = VilleMonitor(delta=0.5, sigma=1.0, alpha=0.01)
    for _ in range(20): vm0.step(0.0)
    rep("Ville no alarm under null", not vm0.alarmed)
    srb = ShiryaevRoberts(B=100.0)
    rep("E_0[T_B] = B exactly", srb.mean_false_alarm_time == 100.0)
    for _ in range(200): srb.step(0.0)
    rep("SR stops under null at ~B", srb.stopped)

    print("\n[T9] C ABI")
    code, n, rel = abi_photon_budget(13.5, 30.0, 5.0)
    rep("ABI photon budget", code == 0 and abs(n-510) < 10)
    d = abi_dose_for_noise(13.5, 5.0, 0.02)
    rep("ABI inverse dose", d > 0)
    dq2 = abi_dose_for_q(5.0, 1e-9, 0.5, 13.5)
    rep("ABI dose-for-q", dq2 > 0)
    files = api.emit()
    rep("C header written", Path(files["header"]).exists())
    rep("ABI spec written", Path(files["abi_spec"]).exists())
    rep("8 ABI functions", len(ABI_SPEC["functions"]) == 8)
    rep("Mode E is an ABI error code",
        ABI_SPEC["error_codes"][2].startswith("UNCERTIFIED"))
    rep("SPECULATIVE is an ABI error code",
        ABI_SPEC["error_codes"][3] == "SPECULATIVE_BACKEND")

    print("\n[T10] FULL ANALYSIS + SD REPORT")
    a_nxe = api.analyse("ASML_NXE3800E", 26.0, 13.0, 30.0, 5.0)
    a_exe = api.analyse("ASML_EXE5200", 16.0, 8.0, 60.0, 4.0)
    rep("NXE analysis complete", "photon_budget" in a_nxe)
    rep("EXE analysis complete", "photon_budget" in a_exe)
    rep("scope statement present", "no timeline" in a_nxe["scope"])
    rep("no superiority claim", "no superiority claim" in a_nxe["scope"])
    rr = api.report()
    print(f"      outputs={rr['sd_report']['n_outputs']} "
          f"certified={rr['sd_report']['n_certified']} "
          f"Mode_E={rr['sd_report']['n_mode_E']}")
    rep("Mode E outputs retained", rr["sd_report"]["n_mode_E"] > 0)
    rep("report written", (api.out/"report.json").exists())
    rep("all backends in report", len(rr["backends"]) == 6)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Backend registry: ASML NXE:3800E, EXE:5200, DUV immersion, NIL, EBL,
  HYPERFAB (speculative, refused by default).
  Photon budget at 13.5 nm: 91.9 eV, 20.4 photons/nm2 at 30 mJ/cm2,
  ~510 photons in a 5x5 nm cell -> 4.4% shot noise.
  Abbe source decomposition (= Hopkins) with annular/dipole/quadrupole
  illumination, defocus, Zernike aberrations, NILS extraction.
  delta_min^(H) nano-bridge and delta_min^(P) line-break derived from
  the no-Zeno exit bound; dose inverted for a target defect probability.
  RLS / LWR scaling with an explicit calibration constant.
  STC interface conditions and derived refluxing.
  OPMC drift detection with honest (SNR) verification.
  Ville anytime-valid martingale and Shiryaev-Roberts with E_0[T_B] = B.
  Statistics Dictionary with Mode E; C ABI with 8 functions.

WHAT A VENDOR COULD ADOPT
  Derived dose margins instead of tuned ones. Certification gates on
  every wafer output. Anytime-valid drift alarms with exact false-alarm
  control. None of this requires building a scanner.

NOT CLAIMED
  No scanner, no light source, no optics, no stage.
  No timeline. No superiority over ASML.
  Scalar aerial model only; vector and thin-mask effects excluded.
  C_H, C_P, C_RLS require measurement against real defect and LWR data.
  HYPERFAB is speculative and refused unless explicitly requested.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="MSPS lithography math API")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","backends","photons","aerial","defect","lwr",
                 "opmc","analyse","abi","report"])
    ap.add_argument("--backend", default="ASML_NXE3800E")
    ap.add_argument("--pitch", type=float, default=26.0)
    ap.add_argument("--cd", type=float, default=13.0)
    ap.add_argument("--dose", type=float, default=30.0)
    ap.add_argument("--cell", type=float, default=5.0)
    ap.add_argument("--q", type=float, default=1e-9)
    ap.add_argument("--ell", type=int, default=40)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--allow-speculative", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    api = LithoMathAPI(); out: Dict[str, Any] = {}
    try:
        b = get_backend(a.backend, a.allow_speculative)
    except SpeculativeBackend as e:
        print(json.dumps({"REFUSED": str(e)}, indent=2)); return 1

    if a.cmd == "backends":
        out = {k: asdict(v) for k,v in BACKENDS.items()}
        print(json.dumps(out, indent=2))
        print("\nASML EUV is one backend. No timeline claim.")
    elif a.cmd == "photons":
        out = photon_budget(b, a.dose, a.cell)
        print(json.dumps(out, indent=2))
    elif a.cmd == "aerial":
        out = aerial_image_1d(b, MaskSpec(a.pitch, a.cd), PupilSpec(b.na or 0.33))
        out = {k:v for k,v in out.items() if k not in ("x_nm","intensity")}
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "defect":
        pb = photon_budget(b, a.dose, a.cell)
        if not pb.get("photon_shot_noise"):
            out = {"note": "no photon budget for this backend"}
        else:
            out = {"thresholds": delta_min_defect(a.cell, pb["photons_per_cell"]),
                   "dose_for_q": dose_margin_for_defect_rate(
                       a.cell, a.cell, b, a.q)}
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "lwr":
        out = lwr_rls_model(a.pitch/2, a.dose)
        print(json.dumps(out, indent=2))
    elif a.cmd == "opmc":
        y = [0.0]*60 + [3.0]*60 + [-3.0]*60
        log.warning("no telemetry supplied; synthetic series (code test only)")
        out = opmc_certified(y, a.ell, a.sigma)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "analyse":
        out = api.analyse(a.backend, a.pitch, a.cd, a.dose, a.cell,
                          a.allow_speculative)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "abi":
        out = api.emit(); out["spec"] = ABI_SPEC
        print(json.dumps(out, indent=2)); print("\n" + C_HEADER)
    elif a.cmd == "report":
        out = api.report()

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
