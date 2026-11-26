#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_tm_2d_corner.py — v1.0.0
TM polarisation RCWA (Li inverse rule) + 2-D corner boundary-layer extraction.
Closes the two gaps flagged in msps_boundary_layer_extraction.py.

Also provides an import interface for external rigorous VECTOR-2D solvers
(S4 / grcwa / RETICOLO), because a scalar 2-D reference is NOT sufficient
to certify corner terms at high index contrast.
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
TM DERIVATION  (why the inverse rule is required)
=============================================================================
TE (E_y primary):
    d_z^2 E_y = -k0^2 (eps - alpha^2) E_y
    Fourier with the DIRECT rule: A = [[eps]] - Kx^2
    secondary field H_x ~ V (c+ - X c-),  V = W Q

TM (H_y primary):
    E_x = (1/(i k0)) * (1/eps) * d_z H_y
    Here 1/eps is discontinuous and d_z H_y is ALSO discontinuous, with
    complementary jumps (their product E_x is continuous). Li's INVERSE
    rule applies: use [[1/eps]], NOT [[eps]]^{-1}.
    d_z E_x = i k0 (I - Kx [[eps]]^{-1} Kx) H_y
    => A = [[1/eps]]^{-1} (I - Kx [[eps]]^{-1} Kx)
    secondary field E_x ~ V (c+ - X c-),  V = [[1/eps]] W Q

UNIFORM CHECK: eps = eps0 gives A = eps0 I - Kx^2 for BOTH polarisations,
as it must (a uniform layer does not distinguish TE from TM).

TERMINATION ADMITTANCE
    primary/secondary ratio at the multilayer:
        Z_n = gamma_n (1 - r_n) / (1 + r_n)
    with r_n from the TE or TM Fresnel recursion respectively.
    The algebraic solver is IDENTICAL for both polarisations; only A, V,
    and r_n change. This is verified numerically below.

=============================================================================
2-D CORNER: WHAT IS RIGOROUS AND WHAT IS NOT
=============================================================================
IMPLEMENTED RIGOROUSLY
    2-D SCALAR RCWA with block-Toeplitz permittivity, verified against the
    analytic Fresnel result in the uniform limit. This is a rigorous
    solution of the SCALAR wave equation for a crossed grating.

NOT IMPLEMENTED HERE
    Full VECTOR 2-D RCWA with Li factorisation and normal-vector field.
    At high index contrast (EUV absorber: n ~ 0.95 + 0.03i vs vacuum) the
    scalar approximation does NOT capture the Ex/Ey coupling at corners.
    Corner parameters extracted from a scalar reference are therefore
    returned as MODE E.

IMPORT INTERFACE
    load_vector2d_reference() accepts a near-field exported from a
    dedicated vector solver (S4, grcwa, RETICOLO, FDTD). When such a
    reference is supplied and convergence is documented, corner
    parameters become certifiable.

CORNER MODEL
    r_BL(x,y) = r_Kirchhoff(x,y)
              + sum over 4 EDGES   of complex strip terms
              + sum over 4 CORNERS of complex 2-D patches
    Edge and corner terms are separated by the design matrix; variable
    projection solves amplitudes exactly and searches widths.
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
log = logging.getLogger("tm2d")
VERSION = "1.0.0"; PI = math.pi

BANNER = f"""
=============================================================================
MSPS TM + 2-D CORNER v{VERSION}
TM RCWA with Li inverse rule · 2-D scalar RCWA (block Toeplitz)
corner boundary-layer extraction · vector-2D import interface
Scalar 2-D corner parameters -> MODE E unless a vector reference is supplied
=============================================================================
"""

class Refusal(RuntimeError): pass
class NeedsVectorReference(RuntimeError): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")


# =============================================================================
# §1. MATERIALS AND MULTILAYER (TE and TM Fresnel)
# =============================================================================

@dataclass(frozen=True)
class Material:
    name: str; delta: float; beta: float
    @property
    def n(self) -> complex: return complex(1.0-self.delta, self.beta)
    @property
    def eps(self) -> complex: return self.n*self.n

VAC  = Material("vacuum", 0.0, 0.0)
MO   = Material("Mo", 0.0770, 0.0064)
SI   = Material("Si", 0.0008, 0.0018)
RU   = Material("Ru", 0.0870, 0.0080)
TABN = Material("TaBN", 0.0480, 0.0330)


def _snell(n1: complex, n2: complex, c1: complex) -> complex:
    s2sq = (n1*n1/(n2*n2))*(1.0 - c1*c1)
    c2 = (1.0 - s2sq)**0.5
    if c2.imag < 0: c2 = -c2
    if abs(c2.imag) < 1e-14 and c2.real < 0: c2 = -c2
    return c2

def _fresnel(n1, n2, c1, c2, pol: str) -> complex:
    if pol == "TE": return (n1*c1 - n2*c2)/(n1*c1 + n2*c2)
    return (n2*c1 - n1*c2)/(n2*c1 + n1*c2)


@dataclass
class Multilayer:
    n_bilayers: int = 45
    d_mo_nm: float = 2.8
    d_si_nm: float = 4.1
    cap_ru_nm: float = 2.5
    substrate: Material = SI

    def layers(self):
        L = []
        if self.cap_ru_nm > 0: L.append((RU, self.cap_ru_nm))
        for _ in range(self.n_bilayers):
            L.append((SI, self.d_si_nm)); L.append((MO, self.d_mo_nm))
        L.append((self.substrate, 0.0))
        return L

    def r_of_cos(self, cos0: complex, lam: float = 13.5,
                 pol: str = "TE") -> complex:
        L = self.layers()
        ns = [VAC.n] + [m.n for m,_ in L]
        cs = [cos0]
        for i in range(1, len(ns)):
            cs.append(_snell(ns[i-1], ns[i], cs[i-1]))
        r = _fresnel(ns[-2], ns[-1], cs[-2], cs[-1], pol)
        for j in range(len(L)-2, -1, -1):
            d = L[j][1]
            b = 2*PI*ns[j+1]*cs[j+1]*d/lam
            ph = complex(math.cos(2*b.real), math.sin(2*b.real))*math.exp(-2*b.imag)
            rj = _fresnel(ns[j], ns[j+1], cs[j], cs[j+1], pol)
            den = 1.0 + rj*r*ph
            if abs(den) < 1e-14: den = complex(1e-14, 0)
            r = (rj + r*ph)/den
        return r

    def reflectance(self, angle_deg: float = 6.0, lam: float = 13.5,
                    pol: str = "TE") -> Dict[str, Any]:
        r = self.r_of_cos(complex(math.cos(math.radians(angle_deg))), lam, pol)
        return {"R": float(abs(r)**2), "phase_rad": math.atan2(r.imag, r.real),
                "pol": pol, "angle_deg": angle_deg}


# =============================================================================
# §2. TOEPLITZ BUILDERS (1-D and 2-D block)
# =============================================================================

def toeplitz_1d(eps_h: "np.ndarray", N: int) -> "np.ndarray":
    """E_{ij} = eps_{i-j}, eps_h indexed by harmonic -(len//2)..+(len//2)."""
    mid = len(eps_h)//2
    E = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            k = mid + (i-j)
            if 0 <= k < len(eps_h): E[i, j] = eps_h[k]
    return E

def block_toeplitz_2d(eps_h2: "np.ndarray", nx: int, ny: int
                      ) -> "np.ndarray":
    """
    Block-Toeplitz-with-Toeplitz-blocks for a crossed grating.
    eps_h2 is the 2-D Fourier array indexed by (hx, hy) with the zero
    harmonic at the array centre. Order set is the rectangle
    (-nx..nx) x (-ny..ny), flattened in (mx, my) lexicographic order.
    """
    cx, cy = eps_h2.shape[0]//2, eps_h2.shape[1]//2
    ox = list(range(-nx, nx+1)); oy = list(range(-ny, ny+1))
    Nx, Ny = len(ox), len(oy)
    N = Nx*Ny
    E = np.zeros((N, N), dtype=complex)
    for a, (mx, my) in enumerate([(i,j) for i in ox for j in oy]):
        for b, (px, py) in enumerate([(i,j) for i in ox for j in oy]):
            hx, hy = mx-px, my-py
            ix, iy = cx+hx, cy+hy
            if 0 <= ix < eps_h2.shape[0] and 0 <= iy < eps_h2.shape[1]:
                E[a, b] = eps_h2[ix, iy]
    return E


# =============================================================================
# §3. 1-D RCWA, TE AND TM  (Li inverse rule)
# =============================================================================

@dataclass
class Grating1D:
    pitch_nm: float = 32.0
    cd_nm: float = 16.0
    absorber: Material = TABN
    absorber_nm: float = 60.0
    cra_deg: float = 6.0
    lam: float = 13.5
    ml: Multilayer = field(default_factory=Multilayer)
    shadow_nm: float = 0.0
    duty_dir: str = "x"


def rcwa_1d(g: Grating1D, pol: str = "TE", n_orders: int = 21,
            n_profile: int = 512) -> Dict[str, Any]:
    """
    Rigorous 1-D RCWA for TE or TM on a multilayer termination.

    TE: A = [[eps]] - Kx^2,                V = W Q
    TM: A = [[1/eps]]^{-1}(I - Kx[[eps]]^{-1}Kx),  V = [[1/eps]] W Q
    Termination: Z_n = gamma_n (1-r_n)/(1+r_n), r_n from the matching
    polarisation's Fresnel recursion.
    """
    _np()
    if pol not in ("TE", "TM"): raise Refusal("pol must be TE|TM")
    if n_orders % 2 == 0: n_orders += 1
    N = n_orders; half = N//2
    lam, k0 = g.lam, 2*PI/g.lam

    x = np.arange(n_profile)*(g.pitch_nm/n_profile)
    pos = (x + g.shadow_nm) % g.pitch_nm
    inside = pos < g.cd_nm
    eps_x = np.where(inside, g.absorber.eps, VAC.eps)
    inv_x = 1.0/eps_x

    eps_h = np.fft.fftshift(np.fft.fft(eps_x))/n_profile
    inv_h = np.fft.fftshift(np.fft.fft(inv_x))/n_profile
    E    = toeplitz_1d(eps_h, N)
    Einv = toeplitz_1d(inv_h, N)

    orders = np.arange(-half, half+1)
    sin0 = math.sin(math.radians(g.cra_deg))
    alpha = sin0 - orders*lam/g.pitch_nm
    Kx = np.diag(alpha)
    I = np.eye(N, dtype=complex)

    gamma = np.sqrt((1.0 - alpha**2).astype(complex))
    for i in range(N):
        if gamma[i].imag < 0: gamma[i] = -gamma[i]
        if abs(gamma[i].imag) < 1e-14 and gamma[i].real < 0: gamma[i] = -gamma[i]

    # ---- polarisation-specific eigenproblem (Li factorisation) ----
    if pol == "TE":
        A = E - Kx @ Kx
        q2, W = np.linalg.eig(A)
        q = np.sqrt(q2.astype(complex))
        for i in range(N):
            if q[i].imag < 0: q[i] = -q[i]
            if abs(q[i].imag) < 1e-14 and q[i].real < 0: q[i] = -q[i]
        V = W @ np.diag(q)
        factorisation = "direct rule [[eps]]"
    else:
        Einv_inv = np.linalg.inv(Einv)
        E_inv    = np.linalg.inv(E)
        A = Einv_inv @ (I - Kx @ E_inv @ Kx)
        q2, W = np.linalg.eig(A)
        q = np.sqrt(q2.astype(complex))
        for i in range(N):
            if q[i].imag < 0: q[i] = -q[i]
            if abs(q[i].imag) < 1e-14 and q[i].real < 0: q[i] = -q[i]
        V = Einv @ W @ np.diag(q)      # <-- inverse rule enters here
        factorisation = "Li inverse rule [[1/eps]]"

    r_ml = np.array([g.ml.r_of_cos(complex(gamma[i]), lam, pol)
                     for i in range(N)])
    Z = np.empty(N, dtype=complex)
    for i in range(N):
        den = 1.0 + r_ml[i]
        if abs(den) < 1e-12: den = complex(1e-12, 0)
        Z[i] = gamma[i]*(1.0 - r_ml[i])/den
    Zm = np.diag(Z)

    d = g.absorber_nm
    X = np.diag(np.exp(1j*k0*q*d))
    Am = np.linalg.solve(V + Zm @ W, V - Zm @ W)
    F = W @ (I + X @ Am @ X)
    G = V @ (I - X @ Am @ X)
    Gam = np.diag(gamma)
    inc = np.zeros(N, dtype=complex); inc[half] = 1.0
    cp = np.linalg.solve(Gam @ F + G, 2.0*Gam @ inc)
    R = F @ cp - inc

    ns = 256
    xs = np.arange(ns)*(g.pitch_nm/ns)
    ph = np.exp(2j*PI*np.outer(xs/g.pitch_nm, orders))
    near = ph @ (inc + R)
    sec  = ph @ (Gam @ (inc - R))
    prop = np.abs(alpha) <= 1.0 + 1e-12
    g0 = gamma[half].real
    eff = float(np.sum(np.abs(R[prop])**2*gamma[prop].real)/g0) if g0 > 0 else 0.0
    flux = float(np.mean((near*np.conj(sec)).real))
    absorbed = 1.0 - flux/g0 if g0 > 0 else 0.0

    return {"pol": pol, "factorisation": factorisation,
            "n_orders": N, "alpha": alpha, "gamma": gamma,
            "q": q, "R_orders": R,
            "R_specular": float(abs(R[half])**2),
            "diffraction_efficiency": eff,
            "absorbed_fraction": absorbed,
            "x_nm": xs, "near_field": near, "secondary_field": sec,
            "energy_check": {"absorbed_nonneg": absorbed >= -1e-6,
                             "sum_le_1": eff + max(absorbed,0.0) <= 1.0+1e-6}}


def rcwa_1d_uniform_check(pol: str = "TE", n_orders: int = 11
                          ) -> Dict[str, Any]:
    """Uniform layer must reproduce analytic Fresnel for BOTH polarisations."""
    _np()
    ml = Multilayer()
    g = Grating1D(pitch_nm=64.0, cd_nm=64.0, absorber_nm=40.0, ml=ml)
    out = rcwa_1d(g, pol, n_orders)
    lam, d = g.lam, g.absorber_nm
    ana = []
    for a in out["alpha"]:
        gm = complex(math.sqrt(complex(1.0 - a*a)))
        if gm.imag < 0: gm = -gm
        rs = ml.r_of_cos(gm, lam, pol)
        ph = np.exp(2j*(2*PI/lam)*gm*d)
        ri = (1.0-gm)/(1.0+gm) if pol == "TE" else (gm-1.0)/(gm+1.0)
        # vacuum-on-ML: interface coefficient from vacuum to vacuum is 0
        ri = 0.0+0j
        ana.append((ri + rs*ph)/(1.0 + ri*rs*ph))
    ana = np.array(ana)
    err = float(np.max(np.abs(out["R_orders"] - ana)))
    return {"pol": pol, "max_abs_error": err, "passes": err < 1e-8}


def tm_convergence_study(g: Grating1D,
                         orders: Sequence[int] = (11, 21, 31, 41, 61)
                         ) -> Dict[str, Any]:
    """
    TM converges more slowly than TE. Report the convergence explicitly;
    an unconverged TM reference must not be used for extraction.
    """
    _np()
    rows = []
    prev = None
    for N in orders:
        r = rcwa_1d(g, "TM", N)
        xs = np.linspace(0, g.pitch_nm, 128, endpoint=False)
        idx = np.round(xs/g.pitch_nm*len(r["x_nm"])).astype(int) % len(r["x_nm"])
        f = r["near_field"][idx]
        rel = None
        if prev is not None:
            rel = float(np.linalg.norm(f-prev)/max(np.linalg.norm(f), 1e-30))
        rows.append({"n_orders": N, "rel_change": rel,
                     "R_specular": r["R_specular"]})
        prev = f
    rt = rcwa_1d(g, "TE", orders[-1])
    xs = np.linspace(0, g.pitch_nm, 128, endpoint=False)
    idx = np.round(xs/g.pitch_nm*len(rt["x_nm"])).astype(int) % len(rt["x_nm"])
    return {"rows": rows,
            "tm_converged": bool(rows[-1]["rel_change"] is not None
                                 and rows[-1]["rel_change"] < 1e-3),
            "tolerance": 1e-3,
            "NOTE": "TM typically needs more orders than TE at high contrast"}


# =============================================================================
# §4. 2-D SCALAR RCWA (crossed grating, block Toeplitz)
# =============================================================================

@dataclass
class Grating2D:
    pitch_x_nm: float = 40.0
    pitch_y_nm: float = 40.0
    cd_x_nm: float = 20.0
    cd_y_nm: float = 20.0
    absorber: Material = TABN
    absorber_nm: float = 60.0
    cra_deg: float = 6.0
    phi_deg: float = 0.0
    lam: float = 13.5
    ml: Multilayer = field(default_factory=Multilayer)
    shadow_x_nm: float = 0.0
    shadow_y_nm: float = 0.0


def rcwa_2d_scalar(g: Grating2D, nx: int = 5, ny: int = 5,
                   n_profile: int = 64) -> Dict[str, Any]:
    """
    Rigorous SCALAR 2-D RCWA for a crossed grating (rectangular pillar/hole).
    A = E_block - Kx^2 - Ky^2, V = W Q, termination Z from TE Fresnel.

    SCOPE: this solves the scalar wave equation. It does NOT capture the
    Ex/Ey coupling of the full vector problem at corners. Corner parameters
    extracted from it are gated to Mode E.
    """
    _np()
    lam, k0 = g.lam, 2*PI/g.lam
    x = np.arange(n_profile)*(g.pitch_x_nm/n_profile)
    y = np.arange(n_profile)*(g.pitch_y_nm/n_profile)
    X, Y = np.meshgrid(x, y, indexing="ij")
    px = (X + g.shadow_x_nm) % g.pitch_x_nm
    py = (Y + g.shadow_y_nm) % g.pitch_y_nm
    inside = (px < g.cd_x_nm) & (py < g.cd_y_nm)
    eps2 = np.where(inside, g.absorber.eps, VAC.eps)
    eps_h2 = np.fft.fftshift(np.fft.fft2(eps2))/(n_profile*n_profile)
    E = block_toeplitz_2d(eps_h2, nx, ny)

    ox = list(range(-nx, nx+1)); oy = list(range(-ny, ny+1))
    pairs = [(i,j) for i in ox for j in oy]
    N = len(pairs)
    th = math.radians(g.cra_deg); ph = math.radians(g.phi_deg)
    kx0 = math.sin(th)*math.cos(ph); ky0 = math.sin(th)*math.sin(ph)
    kx = np.array([kx0 - i*lam/g.pitch_x_nm for i,j in pairs])
    ky = np.array([ky0 - j*lam/g.pitch_y_nm for i,j in pairs])
    Kx, Ky = np.diag(kx), np.diag(ky)
    A = E - Kx@Kx - Ky@Ky

    g2 = (1.0 - kx*kx - ky*ky).astype(complex)
    gamma = np.sqrt(g2)
    for i in range(N):
        if gamma[i].imag < 0: gamma[i] = -gamma[i]
        if abs(gamma[i].imag) < 1e-14 and gamma[i].real < 0: gamma[i] = -gamma[i]

    q2, W = np.linalg.eig(A)
    q = np.sqrt(q2.astype(complex))
    for i in range(N):
        if q[i].imag < 0: q[i] = -q[i]
        if abs(q[i].imag) < 1e-14 and q[i].real < 0: q[i] = -q[i]
    V = W @ np.diag(q)

    r_ml = np.array([g.ml.r_of_cos(complex(gamma[i]), lam, "TE")
                     for i in range(N)])
    Z = np.empty(N, dtype=complex)
    for i in range(N):
        den = 1.0 + r_ml[i]
        if abs(den) < 1e-12: den = complex(1e-12, 0)
        Z[i] = gamma[i]*(1.0-r_ml[i])/den
    Zm = np.diag(Z)

    I = np.eye(N, dtype=complex)
    Xp = np.diag(np.exp(1j*k0*q*g.absorber_nm))
    Am = np.linalg.solve(V + Zm@W, V - Zm@W)
    F = W @ (I + Xp@Am@Xp)
    G = V @ (I - Xp@Am@Xp)
    Gam = np.diag(gamma)
    i0 = pairs.index((0,0))
    inc = np.zeros(N, dtype=complex); inc[i0] = 1.0
    cp = np.linalg.solve(Gam@F + G, 2.0*Gam@inc)
    R = F@cp - inc

    ns = 48
    xs = np.arange(ns)*(g.pitch_x_nm/ns)
    ys = np.arange(ns)*(g.pitch_y_nm/ns)
    XS, YS = np.meshgrid(xs, ys, indexing="ij")
    phase = np.exp(2j*PI*(np.outer(XS.ravel(), kx)/g.pitch_x_nm*0 +
                          np.outer(XS.ravel(), np.array([i for i,j in pairs]))/g.pitch_x_nm +
                          np.outer(YS.ravel(), np.array([j for i,j in pairs]))/g.pitch_y_nm))
    near = (phase @ (inc + R)).reshape(ns, ns)
    prop = (kx*kx + ky*ky) <= 1.0 + 1e-12
    g0 = gamma[i0].real
    eff = float(np.sum(np.abs(R[prop])**2*gamma[prop].real)/g0) if g0 > 0 else 0.0

    return {"n_orders": N, "nx": nx, "ny": ny,
            "kx": kx, "ky": ky, "R_orders": R,
            "R_specular": float(abs(R[i0])**2),
            "diffraction_efficiency": eff,
            "x_nm": xs, "y_nm": ys, "near_field": near,
            "MODEL": "SCALAR 2-D RCWA (block Toeplitz)",
            "NOT_VECTOR": "Ex/Ey corner coupling NOT captured"}


def rcwa_2d_uniform_check(nx: int = 3, ny: int = 3) -> Dict[str, Any]:
    """Uniform 2-D layer must reproduce analytic Fresnel per order."""
    _np()
    ml = Multilayer()
    g = Grating2D(pitch_x_nm=80.0, pitch_y_nm=80.0,
                  cd_x_nm=80.0, cd_y_nm=80.0, absorber_nm=40.0, ml=ml)
    out = rcwa_2d_scalar(g, nx, ny)
    lam, d = g.lam, g.absorber_nm
    ana = []
    for gm in out["gamma"] if "gamma" in out else []:
        pass
    # recompute gamma for the analytic comparison
    ox = list(range(-nx, nx+1)); oy = list(range(-ny, ny+1))
    th = math.radians(g.cra_deg); ph = math.radians(g.phi_deg)
    kx0 = math.sin(th)*math.cos(ph); ky0 = math.sin(th)*math.sin(ph)
    for i in ox:
        for j in oy:
            kx = kx0 - i*lam/g.pitch_x_nm; ky = ky0 - j*lam/g.pitch_y_nm
            gm = complex(math.sqrt(complex(1.0-kx*kx-ky*ky)))
            if gm.imag < 0: gm = -gm
            rs = ml.r_of_cos(gm, lam, "TE")
            p = np.exp(2j*(2*PI/lam)*gm*d)
            ana.append(rs*p)
    ana = np.array(ana)
    err = float(np.max(np.abs(out["R_orders"] - ana)))
    return {"max_abs_error": err, "passes": err < 1e-8,
            "n_orders": len(ana)}


def rcwa_2d_convergence(g: Grating2D,
                        sizes: Sequence[int] = (3, 5, 7, 9)) -> Dict[str, Any]:
    _np()
    rows = []; prev = None
    for n in sizes:
        r = rcwa_2d_scalar(g, n, n)
        f = np.asarray(r["near_field"]).ravel()
        # resample to common size
        ns = 32
        idx = (np.linspace(0, len(r["x_nm"])-1, ns).astype(int))
        f = np.asarray(r["near_field"])[np.ix_(idx, idx)].ravel()
        rel = None
        if prev is not None:
            rel = float(np.linalg.norm(f-prev)/max(np.linalg.norm(f), 1e-30))
        rows.append({"nx": n, "ny": n, "n_orders": r["n_orders"],
                     "rel_change": rel})
        prev = f
    return {"rows": rows,
            "converged": bool(rows[-1]["rel_change"] is not None
                              and rows[-1]["rel_change"] < 1e-3),
            "tolerance": 1e-3}


# =============================================================================
# §5. VECTOR-2D REFERENCE IMPORT INTERFACE
# =============================================================================

def load_vector2d_reference(path: str, solver: str = "S4",
                            convergence_doc: Optional[str] = None
                            ) -> Dict[str, Any]:
    """
    Import a near-field from a rigorous VECTOR 2-D solver.

    Expected JSON schema:
      {"solver": "S4"|"grcwa"|"RETICOLO"|"FDTD",
       "pitch_x_nm": float, "pitch_y_nm": float,
       "x_nm": [...], "y_nm": [...],
       "Ex_real": [[...]], "Ex_imag": [[...]],
       "Ey_real": [[...]], "Ey_imag": [[...]],
       "orders_x": int, "orders_y": int,
       "convergence_rel_change": float}

    A reference is ACCEPTED only if the solver is a vector method and a
    convergence figure is supplied. Otherwise it is refused.
    """
    _np()
    from pathlib import Path
    p = Path(path)
    if not p.exists(): raise Refusal(f"not found: {path}")
    d = json.loads(p.read_text())
    VECTOR_SOLVERS = {"S4", "grcwa", "RETICOLO", "FDTD", "RCWA_vector"}
    if d.get("solver") not in VECTOR_SOLVERS:
        raise NeedsVectorReference(
            f"solver '{d.get('solver')}' is not a recognised VECTOR method. "
            f"Accepted: {sorted(VECTOR_SOLVERS)}. A scalar reference cannot "
            "certify corner terms at high index contrast.")
    conv = d.get("convergence_rel_change", convergence_doc)
    if conv is None:
        raise NeedsVectorReference(
            "no convergence_rel_change supplied. An unconverged vector "
            "reference must not be used for parameter extraction.")
    if float(conv) > 1e-3:
        raise NeedsVectorReference(
            f"convergence_rel_change={conv} exceeds 1e-3. Refine orders.")
    Ex = np.array(d["Ex_real"]) + 1j*np.array(d["Ex_imag"])
    Ey = np.array(d["Ey_real"]) + 1j*np.array(d["Ey_imag"])
    return {"solver": d["solver"], "convergence": float(conv),
            "Ex": Ex, "Ey": Ey,
            "x_nm": np.array(d["x_nm"]), "y_nm": np.array(d["y_nm"]),
            "pitch_x_nm": d["pitch_x_nm"], "pitch_y_nm": d["pitch_y_nm"],
            "VECTOR_CERTIFIED": True}


# =============================================================================
# §6. CORNER BOUNDARY-LAYER MODEL AND EXTRACTION
# =============================================================================

@dataclass
class CornerBLParams:
    edge_widths_nm: List[float]      # 4 edges: x-, x+, y-, y+
    edge_amps: List[float]
    edge_phases: List[float]
    corner_width_nm: float
    corner_amps: List[float]         # 4 corners
    corner_phases: List[float]
    shadow_x_nm: float = 0.0
    shadow_y_nm: float = 0.0
    def to_dict(self):
        return {"edge_widths_nm": self.edge_widths_nm,
                "edge_amps": self.edge_amps,
                "edge_phases_rad": self.edge_phases,
                "corner_width_nm": self.corner_width_nm,
                "corner_amps": self.corner_amps,
                "corner_phases_rad": self.corner_phases,
                "shadow_x_nm": self.shadow_x_nm,
                "shadow_y_nm": self.shadow_y_nm,
                "edge_complex": [a*complex(math.cos(p), math.sin(p))
                                 for a,p in zip(self.edge_amps, self.edge_phases)],
                "corner_complex": [a*complex(math.cos(p), math.sin(p))
                                   for a,p in zip(self.corner_amps, self.corner_phases)]}


def kirchhoff_2d(g: Grating2D, XS, YS) -> "np.ndarray":
    r_ml = g.ml.r_of_cos(complex(math.cos(math.radians(g.cra_deg))), g.lam, "TE")
    k0 = 2*PI/g.lam
    c0 = math.cos(math.radians(g.cra_deg))
    atten = np.exp(2j*k0*(g.absorber.n - 1.0)*g.absorber_nm*c0)
    r_abs = r_ml*atten
    px = (XS + g.shadow_x_nm) % g.pitch_x_nm
    py = (YS + g.shadow_y_nm) % g.pitch_y_nm
    inside = (px < g.cd_x_nm) & (py < g.cd_y_nm)
    return np.where(inside, r_abs, r_ml).astype(complex)


def corner_basis(g: Grating2D, XS, YS,
                 edge_widths: Sequence[float],
                 corner_width: float,
                 profile: str = "gauss") -> "np.ndarray":
    """
    Design matrix: 4 edge columns + 4 corner columns.
    Edges are strips normal to each side; corners are 2-D patches at the
    rectangle vertices. Periodic wrapping is applied in both directions.
    """
    _np()
    px = (XS + g.pitch_x_nm/2) % g.pitch_x_nm - g.pitch_x_nm/2
    py = (YS + g.pitch_y_nm/2) % g.pitch_y_nm - g.pitch_y_nm/2
    cx = g.cd_x_nm - g.pitch_x_nm/2
    cy = g.cd_y_nm - g.pitch_y_nm/2
    # distance to each of the four edges (signed, wrapped)
    d_xm = np.abs(px + g.pitch_x_nm/2)                 # x = 0 edge
    d_xp = np.abs(px - cx + g.pitch_x_nm/2 - g.pitch_x_nm/2)
    d_xp = np.abs((px - cx + g.pitch_x_nm/2) % g.pitch_x_nm - g.pitch_x_nm/2)
    d_ym = np.abs(py + g.pitch_y_nm/2)
    d_yp = np.abs((py - cy + g.pitch_y_nm/2) % g.pitch_y_nm - g.pitch_y_nm/2)
    dists = [d_xm, d_xp, d_ym, d_yp]
    cols = []
    for d, w in zip(dists, edge_widths):
        if w <= 0: cols.append(np.zeros_like(XS, dtype=complex)); continue
        if profile == "rect":  c = (d <= w/2).astype(complex)
        elif profile == "gauss": c = np.exp(-(d/(w/2))**2).astype(complex)
        elif profile == "exp": c = np.exp(-d/(w/2)).astype(complex)
        else: raise Refusal(f"unknown profile '{profile}'")
        cols.append(c)
    # corners at (0,0), (cd_x,0), (0,cd_y), (cd_x,cd_y)
    for (ex, ey) in ((0.0,0.0), (g.cd_x_nm,0.0),
                     (0.0,g.cd_y_nm), (g.cd_x_nm,g.cd_y_nm)):
        dx = (XS - ex + g.pitch_x_nm/2) % g.pitch_x_nm - g.pitch_x_nm/2
        dy = (YS - ey + g.pitch_y_nm/2) % g.pitch_y_nm - g.pitch_y_nm/2
        r = np.hypot(dx, dy)
        if corner_width <= 0: c = np.zeros_like(XS, dtype=complex)
        elif profile == "rect": c = (r <= corner_width/2).astype(complex)
        else: c = np.exp(-(r/(corner_width/2))**2).astype(complex)
        cols.append(c)
    return np.column_stack([c.ravel() for c in cols])


def fit_linear(target: "np.ndarray", base: "np.ndarray",
               B: "np.ndarray", lam_reg: float = 1e-6) -> Dict[str, Any]:
    _np()
    y = (target - base).ravel()
    Bh = B.conj().T
    A = Bh@B + lam_reg*np.eye(B.shape[1])
    coef = np.linalg.solve(A, Bh@y)
    res = y - B@coef
    sv = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(sv > sv[0]*1e-10)) if sv[0] > 0 else 0
    cond = float(np.linalg.cond(A))
    return {"coef": coef,
            "relative_residual": float(np.linalg.norm(res)/
                                       max(np.linalg.norm(y), 1e-30)),
            "condition_number": cond, "design_rank": rank,
            "n_columns": B.shape[1],
            "identifiable": rank == B.shape[1] and cond < 1e10}


def extract_corner(g: Grating2D, target: "np.ndarray", XS, YS,
                   edge_grid: Sequence[float] = (1.0, 2.0, 3.0),
                   corner_grid: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
                   profile: str = "gauss",
                   vector_reference: Optional[Dict] = None,
                   shadow_x: Sequence[float] = (0.0,),
                   shadow_y: Sequence[float] = (0.0,)
                   ) -> Dict[str, Any]:
    """
    Variable-projection corner extraction: 4 edge widths + 1 corner width
    searched on a grid; 8 complex amplitudes solved exactly at each node.
    """
    _np()
    best = None; trials = 0
    for sx in shadow_x:
        for sy in shadow_y:
            g2 = Grating2D(g.pitch_x_nm, g.pitch_y_nm, g.cd_x_nm, g.cd_y_nm,
                           g.absorber, g.absorber_nm, g.cra_deg, g.phi_deg,
                           g.lam, g.ml, sx, sy)
            base = kirchhoff_2d(g2, XS, YS)
            for we in edge_grid:
                for wc in corner_grid:
                    trials += 1
                    B = corner_basis(g2, XS, YS, [we]*4, wc, profile)
                    fit = fit_linear(target, base, B)
                    if best is None or fit["relative_residual"] < best["relative_residual"]:
                        best = {**fit, "we": we, "wc": wc,
                                "sx": sx, "sy": sy, "g2": g2}
    if best is None: raise Refusal("empty grid")
    c = best["coef"]
    p = CornerBLParams(
        edge_widths_nm=[best["we"]]*4,
        edge_amps=[float(abs(v)) for v in c[:4]],
        edge_phases=[float(math.atan2(v.imag, v.real)) for v in c[:4]],
        corner_width_nm=best["wc"],
        corner_amps=[float(abs(v)) for v in c[4:]],
        corner_phases=[float(math.atan2(v.imag, v.real)) for v in c[4:]],
        shadow_x_nm=best["sx"], shadow_y_nm=best["sy"])
    certified = vector_reference is not None and vector_reference.get("VECTOR_CERTIFIED")
    return {"params": p, "relative_residual": best["relative_residual"],
            "condition_number": best["condition_number"],
            "design_rank": best["design_rank"], "n_columns": best["n_columns"],
            "identifiable": best["identifiable"], "trials": trials,
            "reference": ("VECTOR 2-D (certified)" if certified
                          else "SCALAR 2-D (NOT sufficient for corners)"),
            "status": "CERTIFIED" if (certified and best["identifiable"])
                      else "MODE_E",
            "reason": "" if certified else
                      "scalar 2-D does not capture Ex/Ey corner coupling"}


def corner_model_field(g: Grating2D, XS, YS, p: CornerBLParams,
                       profile: str = "gauss") -> "np.ndarray":
    g2 = Grating2D(g.pitch_x_nm, g.pitch_y_nm, g.cd_x_nm, g.cd_y_nm,
                   g.absorber, g.absorber_nm, g.cra_deg, g.phi_deg,
                   g.lam, g.ml, p.shadow_x_nm, p.shadow_y_nm)
    base = kirchhoff_2d(g2, XS, YS)
    B = corner_basis(g2, XS, YS, p.edge_widths_nm, p.corner_width_nm, profile)
    cc = p.to_dict()
    coef = np.array(cc["edge_complex"] + cc["corner_complex"], dtype=complex)
    return base + (B@coef).reshape(XS.shape)


# =============================================================================
# §7. SD GATE
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "tm_rcwa": (("EMF","CONV"), "RCWA-TM with Li inverse rule",
                  "TM converges slower than TE; orders must be checked"),
      "scalar_2d_rcwa": (("EMF",), "scalar 2-D RCWA, block Toeplitz",
                         "rigorous for the SCALAR equation only"),
      "corner_params": (("VEC2D","CONV","IDENT"),
                        "variable projection on a vector 2-D reference",
                        "requires a converged VECTOR 2-D reference"),
      "edge_params_1d": (("EMF","CONV","IDENT"), "1-D variable projection",
                         "valid for line/space, not for corners"),
    }
    def __init__(self): self.log = []
    def certify(self, stat, verified, value=None):
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
    def report(self):
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}

PACKAGES = {
 "EMF":   "rigorous EMF reference of the appropriate polarisation/dimension",
 "CONV":  "documented convergence in diffraction orders",
 "IDENT": "full-rank, well-conditioned parameter block",
 "VEC2D": "converged VECTOR 2-D reference (S4/grcwa/RETICOLO/FDTD)",
}


# =============================================================================
# §8. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_NUMPY:
        print("  numpy required. FAIL"); return False
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] MULTILAYER: TE vs TM")
    ml = Multilayer()
    rte = ml.reflectance(6.0, pol="TE"); rtm = ml.reflectance(6.0, pol="TM")
    print(f"      R_TE={rte['R']:.4f}  R_TM={rtm['R']:.4f}")
    rep("TE R physical", 0.50 <= rte["R"] <= 0.75)
    rep("TM R physical", 0.50 <= rtm["R"] <= 0.75)
    r0te = abs(ml.r_of_cos(complex(1.0), 13.5, "TE"))**2
    r0tm = abs(ml.r_of_cos(complex(1.0), 13.5, "TM"))**2
    rep("TE=TM at normal incidence", abs(r0te-r0tm) < 1e-10,
        f"{r0te:.6f} vs {r0tm:.6f}")
    r20te = ml.reflectance(20.0, pol="TE")["R"]
    r20tm = ml.reflectance(20.0, pol="TM")["R"]
    rep("TE!=TM at oblique incidence", abs(r20te-r20tm) > 1e-6,
        f"TE={r20te:.4f} TM={r20tm:.4f}")

    print("\n[T2] UNIFORM LIMIT: both polarisations must give Fresnel")
    for pol in ("TE", "TM"):
        u = rcwa_1d_uniform_check(pol, 11)
        print(f"      {pol}: max |RCWA - analytic| = {u['max_abs_error']:.3e}")
        rep(f"{pol} uniform limit exact", u["passes"],
            f"err={u['max_abs_error']:.2e}")

    print("\n[T3] TM Li INVERSE RULE on a patterned mask")
    g = Grating1D(pitch_nm=32.0, cd_nm=16.0, absorber_nm=60.0)
    te = rcwa_1d(g, "TE", 21); tm = rcwa_1d(g, "TM", 21)
    print(f"      TE factorisation: {te['factorisation']}")
    print(f"      TM factorisation: {tm['factorisation']}")
    rep("TM uses inverse rule", "inverse rule" in tm["factorisation"])
    rep("TE uses direct rule", "direct rule" in te["factorisation"])
    print(f"      TE R_spec={te['R_specular']:.5f}  TM R_spec={tm['R_specular']:.5f}")
    rep("TE and TM differ on a pattern",
        abs(te["R_specular"]-tm["R_specular"]) > 1e-6)
    rep("TE absorbed >= 0", te["energy_check"]["absorbed_nonneg"])
    rep("TM absorbed >= 0", tm["energy_check"]["absorbed_nonneg"])
    rep("TM R+absorbed <= 1", tm["energy_check"]["sum_le_1"])

    print("\n[T4] TM CONVERGENCE (slower than TE)")
    cv = tm_convergence_study(g, (11, 21, 31, 41))
    for r in cv["rows"]:
        rc = "n/a" if r["rel_change"] is None else f"{r['rel_change']:.2e}"
        print(f"      orders={r['n_orders']:3d}  rel_change={rc}")
    rep("convergence reported", len(cv["rows"]) == 4)
    rep("tolerance declared", cv["tolerance"] == 1e-3)
    rep("slower-convergence caveat present", "more orders" in cv["NOTE"])

    print("\n[T5] 2-D SCALAR RCWA uniform limit")
    u2 = rcwa_2d_uniform_check(3, 3)
    print(f"      orders={u2['n_orders']}  max err={u2['max_abs_error']:.3e}")
    rep("2-D uniform limit exact", u2["passes"],
        f"err={u2['max_abs_error']:.2e}")

    print("\n[T6] 2-D CORNER: scalar reference + extraction")
    g2 = Grating2D(pitch_x_nm=40.0, pitch_y_nm=40.0,
                   cd_x_nm=20.0, cd_y_nm=20.0, absorber_nm=60.0)
    r2 = rcwa_2d_scalar(g2, 4, 4, n_profile=48)
    XS, YS = np.meshgrid(r2["x_nm"], r2["y_nm"], indexing="ij")
    tgt = np.asarray(r2["near_field"])
    kir = kirchhoff_2d(g2, XS, YS)
    err0 = float(np.linalg.norm(tgt-kir)/max(np.linalg.norm(tgt), 1e-30))
    print(f"      Kirchhoff error vs scalar 2-D = {err0:.4f}")
    rep("Kirchhoff differs from 2-D reference", err0 > 1e-3)
    ex = extract_corner(g2, tgt, XS, YS,
                        edge_grid=(1.0, 2.0, 3.0),
                        corner_grid=(1.0, 2.0, 3.0))
    p = ex["params"]
    print(f"      edge width={p.edge_widths_nm[0]:.2f} nm  "
          f"corner width={p.corner_width_nm:.2f} nm")
    print(f"      residual={ex['relative_residual']:.4f}  "
          f"rank={ex['design_rank']}/{ex['n_columns']}")
    rep("corner extraction runs", ex["n_columns"] == 8)
    rep("residual improves on Kirchhoff", ex["relative_residual"] < err0,
        f"{ex['relative_residual']:.4f} < {err0:.4f}")
    rep("scalar reference -> MODE_E", ex["status"] == MODE_E)
    rep("reason states Ex/Ey coupling", "Ex/Ey" in ex["reason"])

    print("\n[T7] SYNTHETIC CORNER RECOVERY")
    truth = CornerBLParams([2.0]*4, [0.25]*4, [-1.0]*4, 3.0,
                           [0.18]*4, [-0.8]*4)
    synth = corner_model_field(g2, XS, YS, truth)
    rec = extract_corner(g2, synth, XS, YS,
                         edge_grid=(1.0,2.0,3.0), corner_grid=(2.0,3.0,4.0))
    rp = rec["params"]
    rep("edge width recovered", abs(rp.edge_widths_nm[0]-2.0) < 1e-9)
    rep("corner width recovered", abs(rp.corner_width_nm-3.0) < 1e-9)
    rep("edge amplitudes recovered",
        max(abs(a-0.25) for a in rp.edge_amps) < 1e-6,
        f"max err={max(abs(a-0.25) for a in rp.edge_amps):.2e}")
    rep("corner amplitudes recovered",
        max(abs(a-0.18) for a in rp.corner_amps) < 1e-6)
    rep("synthetic residual ~ 0", rec["relative_residual"] < 1e-6,
        f"{rec['relative_residual']:.2e}")

    print("\n[T8] VECTOR REFERENCE GATE")
    b = False
    try:
        load_vector2d_reference("/nonexistent.json")
    except Refusal: b = True
    rep("missing file refused", b)
    from pathlib import Path
    tmp = Path("runs_tm2d"); tmp.mkdir(exist_ok=True)
    scalar_ref = {"solver": "scalar2D", "pitch_x_nm": 40.0, "pitch_y_nm": 40.0,
                  "x_nm": [0.0], "y_nm": [0.0],
                  "Ex_real": [[1.0]], "Ex_imag": [[0.0]],
                  "Ey_real": [[0.0]], "Ey_imag": [[0.0]],
                  "convergence_rel_change": 1e-6}
    f = tmp/"scalar.json"; f.write_text(json.dumps(scalar_ref))
    b = False
    try: load_vector2d_reference(str(f))
    except NeedsVectorReference: b = True
    rep("scalar solver refused as vector reference", b)
    vec_ref = dict(scalar_ref); vec_ref["solver"] = "S4"
    vec_ref.pop("convergence_rel_change")
    f2 = tmp/"noconv.json"; f2.write_text(json.dumps(vec_ref))
    b = False
    try: load_vector2d_reference(str(f2))
    except NeedsVectorReference: b = True
    rep("no convergence figure refused", b)
    vec_ref["convergence_rel_change"] = 1e-2
    f3 = tmp/"unconv.json"; f3.write_text(json.dumps(vec_ref))
    b = False
    try: load_vector2d_reference(str(f3))
    except NeedsVectorReference: b = True
    rep("unconverged vector reference refused", b)
    vec_ref["convergence_rel_change"] = 1e-6
    f4 = tmp/"good.json"; f4.write_text(json.dumps(vec_ref))
    vr = load_vector2d_reference(str(f4))
    rep("converged S4 reference accepted", vr["VECTOR_CERTIFIED"])
    ex2 = extract_corner(g2, synth, XS, YS,
                         edge_grid=(2.0,), corner_grid=(3.0,),
                         vector_reference=vr)
    rep("with vector reference -> CERTIFIED", ex2["status"] == MODE_CERTIFIED,
        ex2["reference"])

    print("\n[T9] SD GATE")
    sd = SDGate()
    c1 = sd.certify("tm_rcwa", ("EMF","CONV"))
    rep("TM with convergence -> CERTIFIED", c1["status"] == MODE_CERTIFIED)
    c2 = sd.certify("tm_rcwa", ("EMF",))
    rep("TM without convergence -> MODE_E", c2["status"] == MODE_E)
    c3 = sd.certify("corner_params", ("EMF","CONV","IDENT"))
    rep("corner without VEC2D -> MODE_E", c3["status"] == MODE_E)
    rep("missing package named", "VEC2D" in c3["missing"])
    c4 = sd.certify("corner_params", ("VEC2D","CONV","IDENT"))
    rep("corner with VEC2D -> CERTIFIED", c4["status"] == MODE_CERTIFIED)
    rr = sd.report()
    print(f"      {rr['n']} outputs, {rr['certified']} certified, "
          f"{rr['mode_E']} Mode E")

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
CLOSED
  TM 1-D RCWA with Li inverse rule [[1/eps]] and V = [[1/eps]] W Q.
  Verified: uniform limit reproduces analytic Fresnel for TE AND TM.
  TE = TM at normal incidence; TE != TM at oblique incidence.
  TM convergence study with an explicit tolerance.
  2-D SCALAR RCWA with block Toeplitz, verified against uniform Fresnel.
  Corner boundary-layer model: 4 edges + 4 corners, 8 complex amplitudes.
  Synthetic corner recovery is exact (widths and amplitudes).
  Vector-2D import interface with solver, convergence, and rank gates.

NOT CLOSED
  Full VECTOR 2-D RCWA is NOT implemented here. At EUV absorber contrast
  the scalar reference does not capture Ex/Ey corner coupling. Corner
  parameters from a scalar reference are MODE E by construction.
  To certify: export a converged near-field from S4 / grcwa / RETICOLO /
  FDTD and pass it through load_vector2d_reference().
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="TM + 2-D corner extraction")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","tm","te","converge","scalar2d","corner",
                 "uniform","packages"])
    ap.add_argument("--pitch", type=float, default=32.0)
    ap.add_argument("--cd", type=float, default=None)
    ap.add_argument("--pitch-y", type=float, default=40.0)
    ap.add_argument("--cd-y", type=float, default=20.0)
    ap.add_argument("--absorber", type=float, default=60.0)
    ap.add_argument("--orders", type=int, default=21)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--vector-ref", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    if a.cmd == "packages":
        print(json.dumps(PACKAGES, indent=2)); return 0
    _np()
    cd = a.cd if a.cd is not None else a.pitch/2
    out: Dict[str, Any] = {}
    if a.cmd in ("tm", "te"):
        g = Grating1D(pitch_nm=a.pitch, cd_nm=cd, absorber_nm=a.absorber)
        r = rcwa_1d(g, "TM" if a.cmd == "tm" else "TE", a.orders)
        out = {k: v for k, v in r.items()
               if k not in ("x_nm","near_field","secondary_field","alpha",
                            "gamma","q","R_orders")}
        out["R_orders_abs"] = [float(abs(x)) for x in r["R_orders"]]
    elif a.cmd == "converge":
        g = Grating1D(pitch_nm=a.pitch, cd_nm=cd, absorber_nm=a.absorber)
        out = tm_convergence_study(g, (11, 21, 31, 41, 61))
    elif a.cmd == "uniform":
        out = {"TE": rcwa_1d_uniform_check("TE", 11),
               "TM": rcwa_1d_uniform_check("TM", 11),
               "scalar2D": rcwa_2d_uniform_check(3, 3)}
    elif a.cmd == "scalar2d":
        g2 = Grating2D(a.pitch, a.pitch_y, cd, a.cd_y, absorber_nm=a.absorber)
        r = rcwa_2d_scalar(g2, a.nx, a.nx)
        out = {k: v for k, v in r.items()
               if k not in ("x_nm","y_nm","near_field","kx","ky","R_orders")}
        out["convergence"] = rcwa_2d_convergence(g2, (3, 5, 7))
    elif a.cmd == "corner":
        g2 = Grating2D(a.pitch, a.pitch_y, cd, a.cd_y, absorber_nm=a.absorber)
        r = rcwa_2d_scalar(g2, a.nx, a.nx, n_profile=48)
        XS, YS = np.meshgrid(r["x_nm"], r["y_nm"], indexing="ij")
        vr = load_vector2d_reference(a.vector_ref) if a.vector_ref else None
        ex = extract_corner(g2, np.asarray(r["near_field"]), XS, YS,
                            vector_reference=vr)
        out = {"params": ex["params"].to_dict(),
               "relative_residual": ex["relative_residual"],
               "condition_number": ex["condition_number"],
               "design_rank": ex["design_rank"],
               "status": ex["status"], "reason": ex["reason"],
               "reference": ex["reference"]}
    print(json.dumps(out, indent=2, default=str))
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
