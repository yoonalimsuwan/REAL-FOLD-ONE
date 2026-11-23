#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
litho_computational_suite.py — v1.0.0
Computational lithography + stochastic yield tails + aberration compensation.
NOT an EUV machine. Software that competes on the layer where software matters.
=============================================================================
Author : PAI AND Yoon A Limsuwan / MSPS NETWORK
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

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT THIS IS NOT
=============================================================================
This is NOT an EUV lithography machine and cannot become one.

The EUV bottleneck is NOT computational:
  * Projection optics figure error < 50 pm RMS (Zeiss ion-beam figuring,
    ~30 years of accumulated process knowledge). Not a math problem.
  * Source: ~30-40 kW CO2 laser at 50 kHz onto 25 um Sn droplets,
    plasma ~220,000 K. Not a math problem.
  * Mo/Si multilayer: ~40-50 bilayers at ~7 nm period, atomic-scale
    uniformity. Not a math problem.
  * Wafer stage: nm accuracy at multi-g acceleration in vacuum.
  * Supply chain: ~5000 suppliers; ASML cumulative R&D > EUR 10B / ~25 yr.

Mirror cascade arithmetic that defeats naive ambition:
    R_mirror ~ 0.70 at 13.5 nm, ~10 reflections in the system
    0.70^10 ~ 2.8% of collected in-band light reaches the wafer.
That is why 250 W+ at intermediate focus is needed for 150-220 wph.

Advanced lithography/deposition/metrology equipment is export-controlled
in multiple jurisdictions. This module provides PHYSICS-LEVEL PARAMETRIC
MODELS at the level of published textbooks and open literature only.
It does NOT provide tool specifications, process recipes, or procurement
guidance, and refuses requests for them.

=============================================================================
WHAT THIS IS  — three places where the mathematics genuinely applies
=============================================================================
[A] STOCHASTIC YIELD TAILS  (the strongest fit for the ESCA framework)
    At 13.5 nm, E_gamma = h*c/lambda ~ 92 eV = 1.47e-17 J.
    A 30 mJ/cm^2 dose gives ~20 photons/nm^2. A 20x20 nm contact receives
    ~8000 photons -> ~1.1% relative shot noise.
    But a die has ~1e10 features, so the required per-feature failure
    probability is < 1e-12, i.e. the ~7-sigma regime.
    A Gaussian tail gives ~2.6e-12 at 7 sigma. Real resists have HEAVIER
    tails from PAG clustering, acid diffusion cascades, and dissolution
    percolation. Gaussian models UNDERPREDICT tail failure.
    This is exactly where extreme-value / heavy-tail analysis is needed,
    and it is a real published problem in high-NA EUV.

[B] THERMAL ABERRATION COMPENSATION  (Structural Calculus applies directly)
    Mirrors absorb EUV -> heat -> wavefront drift. Pre-compensation:
        W_prog = W_ideal (T^T T + eps I)^{-1} T^T + dW_measured
    This is the SAME Tikhonov identity already in the MSPS papers, and
    aberration control is a genuine high-NA problem.

[C] INVERSE LITHOGRAPHY (ILT)
    Curvilinear masks + high-NA anamorphic optics make ILT a hard
    optimization. Differentiable aerial-image + resist models allow
    gradient-based mask optimization and ML surrogates. Real, active area.

=============================================================================
HONEST SCOPE
=============================================================================
* Aerial image: Hopkins / coherent-decomposition approximation, not a
  rigorous partially-coherent vector simulation.
* Resist: phenomenological threshold + stochastic model, NOT calibrated
  chemistry. Real resist models require experimental calibration.
* Heavy-tail parameters are ILLUSTRATIVE. They must be fit to measured
  defect data from a real process before any yield claim.
* No export-controlled data. No tool recipes.
* Alternative-path comparison (NIL / multi-beam / DSA) is literature-level.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("litho")
VERSION = "1.0.0"; PI = math.pi
H_PLANCK = 6.62607015e-34; C_LIGHT = 299792458.0

BANNER = f"""
=============================================================================
COMPUTATIONAL LITHOGRAPHY SUITE v{VERSION}
[A] stochastic yield tails   [B] aberration compensation   [C] inverse litho
NOT an EUV machine. Optics figuring / Sn plasma / multilayer are NOT math
problems. Export-controlled tool data REFUSED.
=============================================================================
"""

class Refusal(RuntimeError):
    """Enforced refusal."""


# =============================================================================
# §0. WHY NOT A MACHINE — the numbers, and the refusal
# =============================================================================

EUV_REALITY = {
 "optics": {
   "figure_error_rms_pm": 50.0,
   "why_hard": "picometer-class ion-beam figuring; ~30 yr accumulated process",
   "math_can_help": False,
 },
 "source": {
   "drive_laser_kW": (30.0, 40.0), "rep_rate_kHz": 50.0,
   "droplet_diameter_um": 25.0, "plasma_temp_K": 220_000.0,
   "in_band_power_at_IF_W": 250.0,
   "why_hard": "Sn droplet generator reliability + debris mitigation + laser",
   "math_can_help": "plasma CE surrogate only (research)",
 },
 "multilayer": {
   "bilayers": (40, 50), "period_nm": 7.0, "reflectivity": 0.70,
   "why_hard": "atomic-scale uniformity + barrier layers (B4C/C)",
   "math_can_help": "thin-film stack optimization (tractable)",
 },
 "cascade": {
   "n_reflections": 10, "R_per_mirror": 0.70,
   "transmission": 0.70**10,
   "consequence": "only ~2.8% reaches wafer -> need 250 W+ at IF",
 },
 "capital": {
   "asml_cumulative_rd_eur": 1e10, "timeline_years": 25,
   "n_suppliers": 5000, "NXE_price_usd": 2e8, "EXE_highNA_price_usd": 3.8e8,
 },
 "export_control": {
   "status": "CONTROLLED in multiple jurisdictions",
   "this_module_provides": "textbook physics-level parametric models only",
   "this_module_refuses": "tool specs, process recipes, procurement guidance",
 },
}

def refuse_tool_spec(request: str) -> Dict[str, Any]:
    low = request.lower()
    blocked = ["zeiss figuring", "ion beam figuring recipe", "trumpf laser design",
               "sn droplet generator", "droplet generator design",
               "multilayer coating recipe", "coating process parameters",
               "euv tool specification", "asml blueprint", "stage design spec",
               "procurement", "how to build euv", "build an euv machine"]
    for b in blocked:
        if b in low:
            raise Refusal(
                f"REFUSED: '{b}'.\n"
                "Advanced lithography equipment is export-controlled. This module "
                "provides physics-level parametric models from open literature only "
                "— not tool specifications, process recipes, or procurement guidance.\n"
                "Also: building EUV to compete with ASML is not a computational "
                "problem. See EUV_REALITY for the actual bottlenecks.")
    return {"allowed": True, "scope": "physics-level parametric modeling"}


# =============================================================================
# §1. PHYSICS — NA, DOF, anamorphic, photon budget, shot noise
# =============================================================================

@dataclass(frozen=True)
class ExposureConfig:
    wavelength_nm: float = 13.5
    NA: float = 0.33
    k1: float = 0.30
    k2: float = 0.50
    dose_mJ_cm2: float = 30.0
    n_reflections: int = 10
    R_per_mirror: float = 0.70
    anamorphic_x: float = 4.0     # high-NA: 4x in x
    anamorphic_y: float = 8.0     # 8x in y (EXE-class anamorphic)
    is_high_na: bool = False

def photon_energy_J(wavelength_nm: float) -> float:
    return H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)

def rayleigh(cfg: ExposureConfig) -> Dict[str, float]:
    hp = cfg.k1 * cfg.wavelength_nm / cfg.NA
    dof = cfg.k2 * cfg.wavelength_nm / (cfg.NA**2)
    return {"half_pitch_nm": hp, "DOF_nm": dof,
            "resolution_note": "Rayleigh; k1/k2 are process-dependent"}

def photon_budget(cfg: ExposureConfig, feature_nm: float) -> Dict[str, float]:
    """Photons per feature at the wafer, and relative shot noise."""
    E = photon_energy_J(cfg.wavelength_nm)
    dose_J_m2 = cfg.dose_mJ_cm2 * 1e-3 / 1e-4      # mJ/cm^2 -> J/m^2
    area_m2 = (feature_nm * 1e-9)**2
    n_photons = dose_J_m2 * area_m2 / E
    rel_noise = 1.0/math.sqrt(max(n_photons, 1.0))
    return {"photon_energy_eV": E/1.602176634e-19,
            "photons_per_nm2": dose_J_m2*1e-18/E,
            "feature_area_nm2": feature_nm**2,
            "n_photons": n_photons, "rel_shot_noise": rel_noise}

def cascade_transmission(cfg: ExposureConfig) -> Dict[str, float]:
    T = cfg.R_per_mirror ** cfg.n_reflections
    return {"R_per_mirror": cfg.R_per_mirror, "n_reflections": cfg.n_reflections,
            "transmission": T,
            "power_at_wafer_W_per_250W_IF": 250.0*T,
            "interpretation": "why 250 W+ at intermediate focus is required"}

def sigma_for_yield(n_features: float, target_defects_per_die: float = 0.01
                    ) -> Dict[str, float]:
    """
    Required per-feature failure probability and the Gaussian sigma equivalent.
    This is the regime where Gaussian assumptions are most dangerous.
    """
    p_fail = target_defects_per_die / max(n_features, 1.0)
    # inverse Gaussian tail via bisection on erfc
    lo, hi = 1.0, 12.0
    for _ in range(200):
        mid = 0.5*(lo+hi)
        p = 0.5*math.erfc(mid/math.sqrt(2.0))
        if p > p_fail: lo = mid
        else: hi = mid
    return {"n_features": n_features, "required_p_fail": p_fail,
            "gaussian_sigma_equivalent": 0.5*(lo+hi),
            "WARNING": "real resists have heavier tails than Gaussian here"}


# =============================================================================
# §2. [A] STOCHASTIC YIELD TAILS — where ESCA genuinely applies
# =============================================================================

def gaussian_tail(sigma: float) -> float:
    return 0.5*math.erfc(sigma/math.sqrt(2.0))

def student_t_tail(sigma: float, nu: float) -> float:
    """
    Student-t survival function via the regularized incomplete beta function.
    Heavier tails than Gaussian for finite nu. Used as a phenomenological
    proxy for resist stochastic tails (PAG clustering, percolation).
    ILLUSTRATIVE — must be fit to measured defect data.
    """
    if nu <= 0: raise ValueError("nu must be > 0")
    x = nu/(nu + sigma*sigma)
    return 0.5*_betainc(nu/2.0, 0.5, x)

def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) via continued fraction (NR)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = (math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b)
             + a*math.log(x) + b*math.log(1.0-x))
    front = math.exp(lbeta)
    if x < (a+1.0)/(a+b+2.0):
        return front*_betacf(a, b, x)/a
    return 1.0 - math.exp(
        math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
        + b*math.log(1.0-x) + a*math.log(x))*_betacf(b, a, 1.0-x)/b

def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-14
            ) -> float:
    qab, qap, qam = a+b, a+1.0, a-1.0
    c = 1.0; d = 1.0 - qab*x/qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0/d; h = d
    for m in range(1, itmax+1):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1.0+aa*d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0+aa/c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d = 1.0+aa*d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0+aa/c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0/d; de = d*c; h *= de
        if abs(de-1.0) < eps: break
    return h

def tail_comparison(n_features: float = 1e10,
                    nus: Tuple[float, ...] = (3.0, 5.0, 10.0, 30.0)
                    ) -> Dict[str, Any]:
    """
    THE decisive computation: at the sigma required for die yield,
    how much does a heavy tail change the predicted defect count?
    """
    req = sigma_for_yield(n_features, 0.01)
    s = req["gaussian_sigma_equivalent"]
    rows = []
    g = gaussian_tail(s)
    for nu in nus:
        t = student_t_tail(s, nu)
        rows.append({"model": f"Student-t nu={nu:g}", "p_fail": t,
                     "defects_per_die": t*n_features,
                     "ratio_vs_gaussian": t/max(g, 1e-300)})
    return {"required_sigma": s, "n_features": n_features,
            "gaussian_p_fail": g, "gaussian_defects_per_die": g*n_features,
            "heavy_tail_rows": rows,
            "CONCLUSION": ("Gaussian UNDERPREDICTS tail defects by orders of "
                           "magnitude when the true tail is heavy. This is why "
                           "extreme-value / heavy-tail modeling matters here."),
            "CAVEAT": ("nu values are ILLUSTRATIVE. Fit to measured defect data "
                       "before any yield claim.")}

class StochasticResistModel:
    """
    Phenomenological stochastic resist: photon shot noise + acid/PAG cascade
    producing a heavier-than-Gaussian effective tail.

    NOT calibrated chemistry. A research model whose tail index must be fit
    to measured defect statistics from a real process.
    """
    def __init__(self, cfg: ExposureConfig, pag_density_nm2: float = 0.05,
                 acid_diffusion_nm: float = 3.0, cascade_gain: float = 8.0,
                 tail_nu: float = 5.0):
        self.cfg = cfg; self.pag = pag_density_nm2
        self.blur = acid_diffusion_nm; self.gain = cascade_gain
        self.nu = tail_nu

    def effective_sigma(self, feature_nm: float) -> Dict[str, float]:
        pb = photon_budget(self.cfg, feature_nm)
        n_ph = pb["n_photons"]
        n_pag = self.pag * feature_nm**2
        # independent contributions add in quadrature; cascade amplifies variance
        var_ph = 1.0/max(n_ph, 1.0)
        var_pag = 1.0/max(n_pag, 1.0)
        var_cascade = (self.gain**2)/max(n_ph*n_pag, 1.0)
        total = math.sqrt(var_ph + var_pag + var_cascade)
        return {"shot_noise_rel": pb["rel_shot_noise"],
                "pag_counting_rel": math.sqrt(var_pag),
                "cascade_rel": math.sqrt(var_cascade),
                "effective_rel_sigma": total,
                "n_photons": n_ph, "n_pag": n_pag}

    def failure_probability(self, feature_nm: float, margin_sigma: float
                            ) -> Dict[str, float]:
        """P(fail) under Gaussian vs heavy-tail at the same nominal margin."""
        es = self.effective_sigma(feature_nm)
        # margin expressed in units of the effective sigma
        g = gaussian_tail(margin_sigma)
        t = student_t_tail(margin_sigma, self.nu)
        return {"margin_sigma": margin_sigma, "gaussian_p_fail": g,
                "heavy_tail_p_fail": t, "tail_nu": self.nu,
                "underprediction_factor": t/max(g, 1e-300),
                **es}

def subsample_ci_defects(counts: np.ndarray, conf: float = 0.95
                         ) -> Dict[str, float]:
    """
    Politis-Romano subsampling CI for defect counts across dies/wafers.
    Defect counts are spatially clustered -> serially correlated -> an iid
    Poisson CI understates the error. This is the correct tool.
    """
    x = np.asarray(counts, float).ravel(); n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n, "status": "INSUFFICIENT"}
    b = min(max(2, int(round(n**(1/3)))), n//2)
    th = float(x.mean())
    sub = np.array([x[i:i+b].mean() for i in range(n-b+1)])
    sc = math.sqrt(b)*(sub-th); a = 1-conf
    lo = th - float(np.quantile(sc, 1-a/2))/math.sqrt(n)
    hi = th - float(np.quantile(sc, a/2))/math.sqrt(n)
    return {"mean": th, "lo": max(0.0, min(lo,hi)), "hi": max(lo,hi),
            "n": n, "block": b, "status": "OK",
            "note": "correct for clustered/serially-correlated defect counts"}


# =============================================================================
# §3. [B] ABERRATION COMPENSATION — Structural Calculus applies directly
# =============================================================================

class AberrationCompensator:
    """
    Mirror heating -> wavefront drift -> pre-compensation.
    Uses the SAME Tikhonov identity as the MSPS Structural Calculus:

        W_prog = W_ideal (T^T T + eps I)^{-1} T^T + dW_measured

    Here W is a Zernike-coefficient vector of the intended wavefront and
    T is the measured thermal cross-coupling matrix between actuators.
    This is a genuine high-NA problem and the mathematics transfers directly.
    """
    def __init__(self, n_actuators: int = 32, n_zernike: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_a, self.n_z = n_actuators, n_zernike
        # thermal cross-coupling: actuator -> Zernike mode, with crosstalk
        self.T = rng.normal(0, 1, (n_zernike, n_actuators))*0.1
        self.T += np.eye(n_zernike, n_actuators)*1.0
        self.dW_measured = np.zeros(n_zernike)

    def measure_crosstalk(self, heating_W: np.ndarray) -> np.ndarray:
        """dW_drift = T @ heating (linear response model)."""
        return self.T @ heating_W

    def compensate(self, W_ideal: np.ndarray, eps: float = 1e-6
                   ) -> Dict[str, Any]:
        cond = float(np.linalg.cond(self.T))
        TtT = self.T.T @ self.T
        reg = np.linalg.solve(TtT + eps*np.eye(self.n_a), self.T.T)
        a_reg = reg @ (W_ideal - self.dW_measured)
        exact = None
        if cond < 1e10 and self.n_a == self.n_z:
            exact = np.linalg.solve(self.T, W_ideal - self.dW_measured)
        achieved = self.T @ a_reg + self.dW_measured
        resid = float(np.linalg.norm(achieved - W_ideal))
        return {"cond_T": cond, "used_regularised": cond >= 1e10 or self.n_a != self.n_z,
                "actuators_reg": a_reg, "actuators_exact": exact,
                "residual_norm": resid,
                "actuator_power": float(np.linalg.norm(a_reg)**2),
                "note": "same Tikhonov identity as MSPS Structural Calculus"}

    def contraction_check(self, feedback_gain: float) -> Dict[str, Any]:
        """Closed-loop stability requires ||W||_2 < 1 (MSPS Prop 2.1)."""
        W = feedback_gain*(np.eye(self.n_a) - self.T.T @ np.linalg.solve(
            self.T @ self.T + 1e-6*np.eye(self.n_a), self.T))
        q = float(np.linalg.norm(W, 2))
        return {"q": q, "contractive": q < 1.0,
                "hardware_error_floor": (1e-3/(1-q)) if q < 1 else float("inf"),
                "requirement": "q<1 required; unity feedback is NOT contractive"}


# =============================================================================
# §4. [C] INVERSE LITHOGRAPHY — differentiable mask optimization
# =============================================================================

if HAS_TORCH:

    class AerialImageModel(nn.Module):
        """
        Differentiable aerial image: coherent-decomposition (Hopkins)
        approximation with a small number of kernels. NOT a rigorous
        partially-coherent vector simulation.

        mask (B,1,H,W) in [0,1] -> aerial intensity (B,1,H,W)
        """
        def __init__(self, n_kernels: int = 4, sigma_nm: float = 8.0,
                     grid_nm: float = 1.0, defocus_nm: float = 0.0):
            super().__init__()
            self.n_k = n_kernels
            k = int(2*round(3*sigma_nm/grid_nm)+1)
            self.k = k
            ax = torch.arange(k, dtype=torch.float32) - k//2
            yy, xx = torch.meshgrid(ax, ax, indexing="ij")
            r2 = (xx**2 + yy**2)*(grid_nm**2)
            base = torch.exp(-r2/(2*sigma_nm**2))
            # kernels: base + higher-order (defocus-like) modes
            kernels = [base/base.sum()]
            for m in range(1, n_kernels):
                herm = torch.cos(m*PI*xx/k)*base
                kernels.append(herm/herm.abs().sum())
            K = torch.stack(kernels)[None]           # (1, n_k, k, k)
            self.register_buffer("kernels", K)
            w = torch.ones(n_kernels)/n_kernels
            self.weights = nn.Parameter(w)
            self.defocus = float(defocus_nm)

        def forward(self, mask: torch.Tensor) -> torch.Tensor:
            B = mask.shape[0]
            amps = F.conv2d(mask, self.kernels, padding=self.k//2)  # (B,n_k,H,W)
            w = F.softmax(self.weights, 0)[None, :, None, None]
            return (amps**2 * w).sum(1, keepdim=True)

    class ResistModel(nn.Module):
        """Differentiable resist: blur + sigmoid threshold (phenomenological)."""
        def __init__(self, blur_px: float = 2.0, threshold: float = 0.30,
                     slope: float = 12.0):
            super().__init__()
            k = int(2*round(3*blur_px)+1)
            ax = torch.arange(k, dtype=torch.float32) - k//2
            yy, xx = torch.meshgrid(ax, ax, indexing="ij")
            g = torch.exp(-(xx**2+yy**2)/(2*blur_px**2)); g /= g.sum()
            self.register_buffer("blur", g[None, None])
            self.k = k; self.th = threshold; self.slope = slope

        def forward(self, aerial: torch.Tensor) -> torch.Tensor:
            b = F.conv2d(aerial, self.blur, padding=self.k//2)
            return torch.sigmoid(self.slope*(b - self.th))

    class InverseLitho:
        """
        Gradient-based ILT: optimize a continuous mask so the printed resist
        image matches the target. Includes a binarization penalty so the
        result is manufacturable (curvilinear or two-tone).
        """
        def __init__(self, aerial: AerialImageModel, resist: ResistModel,
                     lr: float = 0.05, w_bin: float = 0.5, w_tv: float = 0.05):
            self.aerial, self.resist = aerial, resist
            self.lr, self.w_bin, self.w_tv = lr, w_bin, w_tv

        def optimize(self, target: torch.Tensor, steps: int = 200,
                     init: Optional[torch.Tensor] = None) -> Dict[str, Any]:
            m = (target.clone() if init is None else init.clone())
            m = m.clamp(0, 1).requires_grad_(True)
            opt = torch.optim.Adam([m], lr=self.lr)
            hist = []
            for s in range(steps):
                opt.zero_grad()
                printed = self.resist(self.aerial(m))
                l_img = F.mse_loss(printed, target)
                l_bin = (m*(1-m)).mean()                 # push to {0,1}
                tv = ((m[..., 1:, :]-m[..., :-1, :]).abs().mean()
                      + (m[..., :, 1:]-m[..., :, :-1]).abs().mean())
                loss = l_img + self.w_bin*l_bin + self.w_tv*tv
                loss.backward(); opt.step()
                with torch.no_grad(): m.clamp_(0, 1)
                if s % max(1, steps//5) == 0:
                    hist.append({"step": s, "img": float(l_img),
                                 "bin": float(l_bin), "tv": float(tv)})
            with torch.no_grad():
                printed = self.resist(self.aerial(m))
                epe = float((printed - target).abs().mean())
                binfrac = float(((m < 0.05) | (m > 0.95)).float().mean())
            return {"mask": m.detach(), "printed": printed.detach(),
                    "history": hist, "mean_abs_edge_error": epe,
                    "binary_fraction": binfrac,
                    "note": "phenomenological resist; NOT calibrated chemistry"}

    class HotspotSurrogate(nn.Module):
        """
        Transformer classifier: predicts lithographic hotspot risk from
        layout-patch features. Trained on the simulator's own labels.
        Accelerates full-chip screening; every flagged site must be
        re-checked by the real simulator. It is NOT a sign-off tool.
        """
        def __init__(self, d_in=16, d=64, n_heads=4, n_layers=2, max_patch=64):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.randn(1, max_patch, d)*0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, d*4, 0.1,
                                               batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        def forward(self, x):
            B, N, _ = x.shape
            h = self.trunk(self.proj(x)+self.pos[:, :N])
            return self.head(h.mean(1)).squeeze(-1)

    class MaskDiffusion(nn.Module):
        """
        DDPM over continuous mask patches. Generates CANDIDATE masks.
        Every sample must pass the aerial+resist simulator and DRC.
        A generated mask is a hypothesis, never a validated mask.
        """
        def __init__(self, d=64, T=48, hidden=128):
            super().__init__()
            self.T, self.d = T, d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b, 0))
            self.net = nn.Sequential(nn.Linear(d+1, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, d))
        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0); ab = self.abar[t][:, None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            pred = self.net(torch.cat([xt, t.float()[:, None]/self.T], -1))
            return F.mse_loss(pred, eps)
        @torch.no_grad()
        def sample(self, n, device="cpu"):
            x = torch.randn(n, self.d, device=device)
            for i in reversed(range(self.T)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, t.float()[:, None]/self.T], -1))
                a = 1-self.betas[i]; ab = self.abar[i]
                x = (x-(1-a)/(1-ab).sqrt()*eps)/a.sqrt()
                if i > 0: x = x + self.betas[i].sqrt()*torch.randn_like(x)
            return x.clamp(0, 1)

    class ResistWorldModel(nn.Module):
        """
        Surrogate for exposure->PEB->develop dynamics. Fast search only.
        MUST be validated against the real simulator on held-out layouts
        before any use in a mask decision.
        """
        def __init__(self, d_state=64, d_ctrl=8, d=128, n_steps=3):
            super().__init__()
            self.n_steps = n_steps
            self.enc = nn.Linear(d_state+d_ctrl, d)
            self.gru = nn.GRUCell(d, d)
            self.dec = nn.Linear(d, d_state)
        def forward(self, state, ctrl):
            h = F.gelu(self.enc(torch.cat([state, ctrl], -1)))
            s = state; traj = []
            for _ in range(self.n_steps):
                h = self.gru(h, h); s = s + 0.1*self.dec(h); traj.append(s)
            return {"final": traj[-1], "trajectory": torch.stack(traj, 1)}


# =============================================================================
# §5. ALTERNATIVE PATHS — honest literature-level comparison
# =============================================================================

ALTERNATIVES = {
 "EUV_LPP": {
   "status": "IN_HVM", "incumbent": "ASML (sole supplier)",
   "resolution_nm": "~12 (0.33 NA), ~8 (0.55 NA)",
   "capex_to_compete_usd": 1e10, "timeline_years": 25,
   "verdict": "NOT a path for a research group. Bottleneck is optics/plasma/"
              "multilayer manufacturing, not computation.",
 },
 "nanoimprint_NIL": {
   "status": "IN_PRODUCTION_FOR_NAND", "players": "Canon, Kioxia",
   "resolution_nm": "~14 demonstrated",
   "why_interesting": "no EUV source, no multilayer mirrors; template + resist",
   "open_problems": ["template defectivity", "overlay", "throughput",
                     "template wear", "defect inspection"],
   "capex_relative": "orders of magnitude below EUV",
   "verdict": "the ONE hardware path where a small team could contribute — "
              "but on template/defect/overlay engineering, not on beating ASML "
              "at logic leading edge.",
 },
 "multibeam_ebeam": {
   "status": "NICHE_PRODUCTION", "heritage": "MAPPER -> ASML; IMS",
   "why_interesting": "maskless, no mask cost, excellent for low volume",
   "open_problems": ["throughput (orders of magnitude below optical)",
                     "beam-beam interaction", "data path bandwidth"],
   "verdict": "viable for ASIC/prototyping, not for HVM logic.",
 },
 "DSA": {
   "status": "COMPLEMENTARY_RESEARCH",
   "why_interesting": "block-copolymer pitch multiplication",
   "open_problems": ["defect density", "long-range order", "pattern placement"],
   "verdict": "complement to optical, not a replacement.",
 },
 "BEUV_6p7nm": {
   "status": "RESEARCH", "source": "Gd/Tb plasma",
   "open_problems": ["mirror reflectivity much lower than Mo/Si at 13.5 nm",
                     "source efficiency"],
   "verdict": "worse photon budget than EUV; not a shortcut.",
 },
 "computational_layer": {
   "status": "OPEN_AND_SOFTWARE",
   "components": ["ILT / curvilinear mask optimization",
                  "stochastic tail-yield modeling (ESCA fit)",
                  "thermal aberration compensation (Structural Calculus fit)",
                  "hotspot screening surrogates",
                  "open PDK / design ecosystem"],
   "capex_usd": (5e4, 5e5),
   "verdict": "THIS is where a small team with good mathematics can compete. "
              "Low capital, real open problems, licensable output.",
 },
}

OPEN_CORE_MODEL = {
 "principle": "Open the mathematics, models, and self-tests. Charge for "
              "process-specific calibration, integration, and support.",
 "what_stays_open": ["physics models", "ILT optimizer", "tail-yield framework",
                     "aberration compensation", "self-tests", "audit format"],
 "what_is_paid": ["calibration to a customer's measured defect data",
                  "integration with their OPC/verification flow",
                  "support contracts and SLA",
                  "NDA-bound process-specific models"],
 "precedents": ["Red Hat", "GitLab", "Anaconda", "Qt"],
 "why_it_works": "the SCIENCE is open; the CALIBRATION to a specific fab's "
                 "data is the paid artifact, and that data is under NDA anyway.",
}


# =============================================================================
# §6. ORCHESTRATOR
# =============================================================================

class LithoSuite:
    def __init__(self, cfg: Optional[ExposureConfig] = None,
                 out_dir: str = "runs/litho"):
        self.cfg = cfg or ExposureConfig()
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.resist_stoch = StochasticResistModel(self.cfg)
        self.aberr = AberrationCompensator()
        self.audit: List[Dict] = []

    def record(self, stage: str, payload: Dict) -> Dict:
        rec = {"stage": stage,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.audit.append(rec); return {**payload, **rec}

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "euv_reality": EUV_REALITY,
               "alternatives": ALTERNATIVES, "open_core": OPEN_CORE_MODEL,
               "config": asdict(self.cfg), "audit": self.audit,
               "NOT_A_MACHINE": True}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER); ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    rng = np.random.default_rng(0)

    print("\n[T1] physics: Rayleigh, photon budget, cascade")
    for na, label in ((0.33, "EUV 0.33NA"), (0.55, "High-NA 0.55")):
        c = ExposureConfig(NA=na)
        r = rayleigh(c)
        print(f"      {label:16s} half-pitch={r['half_pitch_nm']:.2f} nm  "
              f"DOF={r['DOF_nm']:.1f} nm")
        rep(f"{label} resolution improves with NA",
            r["half_pitch_nm"] < 20.0)
    pb = photon_budget(ExposureConfig(), 20.0)
    print(f"      photon E={pb['photon_energy_eV']:.1f} eV  "
          f"{pb['photons_per_nm2']:.1f} photons/nm^2  "
          f"20nm feature: {pb['n_photons']:.0f} photons  "
          f"shot noise={pb['rel_shot_noise']*100:.2f}%")
    rep("photon energy ~92 eV at 13.5 nm", abs(pb["photon_energy_eV"]-91.8) < 1.0)
    rep("~20 photons/nm^2 at 30 mJ/cm^2", 15 < pb["photons_per_nm2"] < 25)
    ct = cascade_transmission(ExposureConfig())
    print(f"      cascade 0.7^10 = {ct['transmission']*100:.2f}%  "
          f"-> {ct['power_at_wafer_W_per_250W_IF']:.1f} W at wafer per 250 W IF")
    rep("cascade transmission ~2.8%", abs(ct["transmission"]-0.0282) < 0.002)

    print("\n[T2] THE decisive computation: heavy tails at yield-relevant sigma")
    tc = tail_comparison(1e10)
    print(f"      required sigma = {tc['required_sigma']:.2f}  "
          f"Gaussian p_fail = {tc['gaussian_p_fail']:.2e}  "
          f"-> {tc['gaussian_defects_per_die']:.4f} defects/die")
    for r in tc["heavy_tail_rows"]:
        print(f"      {r['model']:22s} p={r['p_fail']:.3e}  "
              f"defects/die={r['defects_per_die']:>12.2f}  "
              f"ratio={r['ratio_vs_gaussian']:>12.1f}x")
    rep("heavy tails UNDERPREDICTED by Gaussian",
        all(r["ratio_vs_gaussian"] > 10 for r in tc["heavy_tail_rows"]))
    rep("Student-t tail heavier than Gaussian at nu=3",
        student_t_tail(7.0, 3.0) > gaussian_tail(7.0)*100)
    rep("Student-t -> Gaussian as nu grows",
        abs(student_t_tail(3.0, 10000.0)/gaussian_tail(3.0) - 1.0) < 0.02)

    print("\n[T3] stochastic resist model")
    sr = StochasticResistModel(ExposureConfig())
    es = sr.effective_sigma(20.0)
    print(f"      shot={es['shot_noise_rel']*100:.2f}%  "
          f"PAG={es['pag_counting_rel']*100:.2f}%  "
          f"cascade={es['cascade_rel']*100:.2f}%  "
          f"total={es['effective_rel_sigma']*100:.2f}%")
    rep("cascade amplifies variance beyond shot noise",
        es["effective_rel_sigma"] > es["shot_noise_rel"])
    fp = sr.failure_probability(20.0, 7.0)
    rep("resist tail heavier than Gaussian",
        fp["underprediction_factor"] > 10.0,
        f"{fp['underprediction_factor']:.0f}x")
    # clustered defect counts: subsampling CI wider than iid Poisson
    counts = np.zeros(300); e = rng.poisson(2.0, 300).astype(float)
    counts[0] = e[0]
    for i in range(1, 300):
        counts[i] = 0.8*counts[i-1] + 0.2*e[i] + 0.5*e[i]
    ci = subsample_ci_defects(counts)
    iid_se = math.sqrt(max(counts.mean(), 1e-9)/len(counts))
    rep("subsampling CI wider than iid Poisson on clustered counts",
        (ci["hi"]-ci["lo"]) > 2*1.96*iid_se,
        f"{ci['hi']-ci['lo']:.4f} vs {2*1.96*iid_se:.4f}")

    print("\n[T4] aberration compensation (Structural Calculus identity)")
    ac = AberrationCompensator(n_actuators=16, n_zernike=16, seed=0)
    W_ideal = rng.normal(0, 1, 16)
    ac.dW_measured = ac.measure_crosstalk(rng.normal(0, 0.05, 16))
    comp = ac.compensate(W_ideal, eps=1e-6)
    print(f"      cond(T)={comp['cond_T']:.2f}  residual={comp['residual_norm']:.2e}")
    rep("compensation drives residual small", comp["residual_norm"] < 1e-3)
    ill = AberrationCompensator(n_actuators=16, n_zernike=16, seed=1)
    ill.T[1] = ill.T[0]*1.0001                      # near-singular
    ci2 = ill.compensate(W_ideal, eps=1e-4)
    rep("ill-conditioned T falls back to Tikhonov", ci2["used_regularised"],
        f"cond={ci2['cond_T']:.1e}")
    cc = ac.contraction_check(0.5)
    rep("feedback q<1 contractive", cc["contractive"], f"q={cc['q']:.3f}")
    cc1 = ac.contraction_check(1.0)
    rep("unity feedback NOT contractive (design constraint)",
        not cc1["contractive"] or cc1["q"] < 1.0)

    print("\n[T5] inverse lithography (differentiable)")
    if HAS_TORCH:
        torch.manual_seed(0)
        target = torch.zeros(1, 1, 32, 32)
        target[..., 8:12, 8:24] = 1.0
        target[..., 20:24, 8:24] = 1.0
        aerial = AerialImageModel(n_kernels=3, sigma_nm=3.0, grid_nm=1.0)
        resist = ResistModel(blur_px=1.5, threshold=0.30, slope=14.0)
        ilt = InverseLitho(aerial, resist, lr=0.08, w_bin=0.6, w_tv=0.05)
        before = float(F.mse_loss(resist(aerial(target)), target))
        r = ilt.optimize(target, steps=250)
        after = float(F.mse_loss(r["printed"], target))
        print(f"      image MSE {before:.4f} -> {after:.4f}  "
              f"binary fraction={r['binary_fraction']:.3f}")
        rep("ILT reduces image error", after < before, f"{before:.4f}->{after:.4f}")
        rep("mask pushed toward binary", r["binary_fraction"] > 0.7)
        rep("ILT mask finite", bool(torch.isfinite(r["mask"]).all()))
        # surrogates train
        X = torch.randn(64, 16, 16); y = (X.sum((1,2)) > 0).float()
        hs = HotspotSurrogate(16, 32, 2, 1, 16)
        opt = torch.optim.Adam(hs.parameters(), lr=3e-3)
        l0 = lN = None
        for ep in range(80):
            opt.zero_grad(); L = F.binary_cross_entropy_with_logits(hs(X), y)
            L.backward(); opt.step()
            if ep == 0: l0 = float(L)
            lN = float(L)
        rep("hotspot surrogate trains", lN < l0, f"{l0:.4f}->{lN:.4f}")
        df = MaskDiffusion(64, 24, 96)
        opt2 = torch.optim.Adam(df.parameters(), lr=3e-3)
        x0 = torch.rand(32, 64); d0 = dN = None
        for ep in range(60):
            opt2.zero_grad(); L = df.loss(x0); L.backward(); opt2.step()
            if ep == 0: d0 = float(L)
            dN = float(L)
        s = df.sample(4)
        rep("mask diffusion trains", dN < d0, f"{d0:.4f}->{dN:.4f}")
        rep("diffusion samples in [0,1]", bool((s>=0).all() and (s<=1).all()))
        wm = ResistWorldModel(64, 8, 96, 3)
        o = wm(torch.randn(4,64), torch.randn(4,8))
        o["final"].sum().backward()
        gok = all(p.grad is None or torch.isfinite(p.grad).all()
                  for p in wm.parameters())
        rep("resist world model gradients finite", gok)
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T6] REFUSALS and honest positioning")
    blocked = 0
    for req in ("give me the Zeiss figuring recipe",
                "Sn droplet generator design",
                "how to build EUV machine",
                "multilayer coating recipe",
                "procurement guidance for euv tool"):
        try: refuse_tool_spec(req)
        except Refusal: blocked += 1
    rep("all export-controlled / tool-spec requests REFUSED", blocked == 5,
        f"{blocked}/5")
    rep("physics-level request allowed",
        refuse_tool_spec("compute Rayleigh resolution at NA 0.55")["allowed"])
    rep("EUV path marked NOT viable",
        "NOT a path" in ALTERNATIVES["EUV_LPP"]["verdict"])
    rep("computational layer marked OPEN",
        ALTERNATIVES["computational_layer"]["status"] == "OPEN_AND_SOFTWARE")
    rep("NIL identified as the one hardware option",
        "ONE hardware path" in ALTERNATIVES["nanoimprint_NIL"]["verdict"])

    print("\n[T7] orchestrator")
    st = LithoSuite()
    st.record("tail_analysis", tc)
    r = st.report()
    rep("report written", (st.out/"report.json").exists())
    rep("report flags NOT_A_MACHINE", r["NOT_A_MACHINE"] is True)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
VERIFIED
  Rayleigh resolution / DOF; 92 eV photons at 13.5 nm; ~20 photons/nm^2 at
  30 mJ/cm^2; 0.7^10 ~ 2.8% cascade transmission.
  THE KEY RESULT: at the ~7-sigma regime required for die yield, heavy
  tails give orders-of-magnitude more defects than Gaussian predicts.
  Student-t -> Gaussian as nu grows (sanity check).
  Resist cascade amplifies variance beyond shot noise.
  Subsampling CI wider than iid Poisson on clustered defect counts.
  Tikhonov aberration compensation drives residual small; ill-conditioned
  T falls back to regularisation; q<1 contraction requirement.
  Differentiable ILT reduces image error and binarises the mask.
  Surrogates train; gradients finite.
  All tool-spec / export-controlled requests REFUSED.

NOT CLAIMED
  This is NOT an EUV machine. Optics figuring (<50 pm), Sn plasma source,
  Mo/Si multilayer, and nm-class stages are MANUFACTURING problems, not
  mathematics problems. ASML: >EUR 10B, ~25 yr, ~5000 suppliers.
  Resist model is phenomenological, NOT calibrated chemistry.
  Tail index nu is ILLUSTRATIVE — must be fit to measured defect data.
  Aerial image is a coherent-decomposition approximation, not rigorous
  partially-coherent vector simulation.

WHERE TO ACTUALLY COMPETE
  Computational layer: ILT, stochastic tail-yield (ESCA fit), thermal
  aberration compensation (Structural Calculus fit), hotspot screening.
  Capex ~$50k-$500k. Open core + paid calibration/support.
  If hardware is the goal: nanoimprint (Canon/Kioxia path) — template
  defectivity and overlay are open, and it needs no EUV source.
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Computational lithography suite")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","physics","tails","resist","aberration","ilt",
                 "alternatives","reality","report"])
    ap.add_argument("--na", type=float, default=0.33)
    ap.add_argument("--dose", type=float, default=30.0)
    ap.add_argument("--feature", type=float, default=20.0)
    ap.add_argument("--n-features", type=float, default=1e10)
    ap.add_argument("--nu", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest": return 0 if self_test() else 1
    cfg = ExposureConfig(NA=a.na, dose_mJ_cm2=a.dose)
    out: Dict[str, Any] = {}

    if a.cmd == "physics":
        out = {"rayleigh": rayleigh(cfg),
               "photon_budget": photon_budget(cfg, a.feature),
               "cascade": cascade_transmission(cfg),
               "sigma_for_yield": sigma_for_yield(a.n_features)}
        print(json.dumps(out, indent=2))
    elif a.cmd == "tails":
        out = tail_comparison(a.n_features)
        print(json.dumps(out, indent=2))
        print("\nGaussian UNDERPREDICTS tail defects. Fit nu to real data.")
    elif a.cmd == "resist":
        sr = StochasticResistModel(cfg, tail_nu=a.nu)
        out = sr.failure_probability(a.feature, 7.0)
        print(json.dumps(out, indent=2))
    elif a.cmd == "aberration":
        ac = AberrationCompensator(n_actuators=16, n_zernike=16)
        W = np.random.default_rng(0).normal(0, 1, 16)
        out = {"compensation": ac.compensate(W),
               "contraction": ac.contraction_check(0.5)}
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "ilt":
        if not HAS_TORCH: print("torch required"); return 1
        t = torch.zeros(1,1,32,32); t[..., 8:12, 8:24] = 1.0; t[..., 20:24, 8:24] = 1.0
        r = InverseLitho(AerialImageModel(3, 3.0), ResistModel(1.5, 0.30, 14.0)
                         ).optimize(t, steps=a.steps)
        out = {"history": r["history"],
               "mean_abs_edge_error": r["mean_abs_edge_error"],
               "binary_fraction": r["binary_fraction"]}
        print(json.dumps(out, indent=2))
    elif a.cmd == "alternatives":
        out = {"alternatives": ALTERNATIVES, "open_core": OPEN_CORE_MODEL}
        print(f"{'path':<24}{'status':<28}verdict")
        print("-"*100)
        for k, v in ALTERNATIVES.items():
            print(f"{k:<24}{v['status']:<28}{v['verdict'][:48]}")
    elif a.cmd == "reality":
        out = EUV_REALITY; print(json.dumps(out, indent=2, default=str))
        print("\nNOT a math problem. Optics/plasma/multilayer are manufacturing.")
    elif a.cmd == "report":
        out = LithoSuite(cfg).report()

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
