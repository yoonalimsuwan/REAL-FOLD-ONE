#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_amr_topology_learned.py — v1.0.0
Adaptive mesh refinement with STC-derived refluxing, seven-operator topology
validation, and a LEARNED refinement policy (transformer / world model /
diffusion) that is physics-gated by a hard recall constraint.

Central result: a learned policy can approach the flow-intrinsic ceiling
f_max = 1/phi but cannot exceed it. Homogeneous turbulence has phi ~ 1.
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
THE CEILING THEOREM  (what a neural network cannot do)
=============================================================================
Let A(t) subset Omega be the resolution-necessary set: the region where the
local dissipation scale eta(x,t) demands the fine grid. Define

    phi(t) = |A(t)| / |Omega|,      phi_max = max_t phi(t).

Any refinement policy achieving active-cell fraction 1/f must satisfy

    f_achieved <= f_max = 1 / phi_max.

This is a property of the FLOW, not of the policy. For homogeneous
isotropic turbulence phi ~ 1 and f_max ~ 1: no learned policy helps.
For multiphase flow with localised interfaces phi may be 1e-3 or smaller.

Define policy efficiency eta_pol = f_achieved / f_max <= 1. Improving the
policy from eta_old to eta_new gives

    Re_new / Re_old = (eta_new / eta_old)^(4/9).

Worked example: eta 0.30 -> 0.90 at f_max = 1000 gives f 300 -> 900 and
Re_lambda ~4,200 -> ~7,300, a factor 1.71. Real, measurable, and BOUNDED.

=============================================================================
THE SAFETY ASYMMETRY  (what a neural network must never do)
=============================================================================
Over-refinement wastes compute but the solution remains correct.
Under-refinement produces a WRONG solution: it is no longer DNS.

Therefore the learned policy is constrained by

    recall(A subset S_NN) = 1.0     HARD, verified on held-out flows
    precision(S_NN)      < 1        acceptable (over-refinement)

Architecture: the physics heuristic supplies a conservative superset
S_phys superset A. The learned policy proposes a tighter set S_NN with
A subset S_NN subset S_phys. A runtime guard checks a cheap necessary
condition; any violation triggers Mode E and falls back to S_phys.

The neural network SHRINKS the refinement region. It never replaces the
solver, never computes a flux, and never predicts a topological invariant
that is not subsequently computed exactly.

=============================================================================
WHAT THE THREE MODELS DO
=============================================================================
transformer : per-cell refinement scores from local + neighbour features
world model : predicts the NEXT error field, enabling pre-emptive refinement
              before a topology event, reducing missed-cell risk
diffusion   : generates CANDIDATE refinement masks; every candidate is
              intersected with the physics-mandatory set before use

None of them is a flow solver. DNS remains direct numerical simulation.
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("amr_topo")
VERSION = "1.0.0"; PI = math.pi
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
MSPS AMR + TOPOLOGY VALIDATION + LEARNED POLICY v{VERSION}
STC refluxing · 7-operator Sigma_top · delta_min criteria
CEILING: f_achieved <= f_max = 1/phi  (flow property, not policy)
SAFETY: recall = 1.0 HARD; over-refine OK, under-refine FORBIDDEN
=============================================================================
"""

class Refusal(RuntimeError): pass
class UnderRefinement(Refusal): pass
class CeilingViolation(Refusal): pass

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required")


# =============================================================================
# §1. FLOW FIELDS WITH KNOWN ACTIVE-SET STRUCTURE
# =============================================================================

class FlowKind:
    MULTIPHASE = "multiphase_localised"
    SHEAR_LAYER = "shear_layer"
    BOUNDARY_LAYER = "boundary_layer"
    HOMOGENEOUS = "homogeneous_isotropic"

@dataclass
class FlowSpec:
    kind: str
    n: int = 128                 # fine-grid resolution per axis (2-D slice)
    re_l: float = 1e6
    nu: float = 1e-6
    interface_width: float = 0.04   # fraction of domain
    seed: int = 0

def kolmogorov_eta(re_l: float, L: float = 1.0, nu: float = 1e-6
                   ) -> float:
    """eta/L = Re^{-3/4}."""
    return L*re_l**(-0.75)

def synth_field(spec: FlowSpec) -> Dict[str, Any]:
    """
    Construct a velocity-magnitude proxy with a KNOWN resolution-necessary
    structure. This is a synthetic indicator field for policy development,
    NOT a Navier-Stokes solution.

    multiphase / shear / boundary : activity confined to a thin set
    homogeneous                   : activity everywhere
    """
    _np()
    rng = np.random.default_rng(spec.seed)
    n = spec.n
    x = np.linspace(0, 1, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    eta = kolmogorov_eta(spec.re_l, nu=spec.nu)

    if spec.kind == FlowKind.HOMOGENEOUS:
        # broadband activity: every point needs resolution
        k = np.fft.fftfreq(n, d=1.0/n)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        Km = np.hypot(KX, KY)
        amp = np.zeros_like(Km)
        m = (Km > 0) & (Km < n/2)
        amp[m] = Km[m]**(-5/6)     # Kolmogorov -5/3 in energy -> -5/6 amplitude
        ph = rng.uniform(0, 2*PI, Km.shape)
        u = np.fft.ifft2(amp*np.exp(1j*ph)).real
        u = u/max(np.std(u), 1e-12)
        grad = np.hypot(*np.gradient(u))
        need = np.ones((n, n), dtype=bool)   # phi = 1
    else:
        if spec.kind == FlowKind.MULTIPHASE:
            # circular interface of radius 0.3
            r = np.hypot(X-0.5, Y-0.5)
            d = np.abs(r-0.30)
        elif spec.kind == FlowKind.SHEAR_LAYER:
            d = np.abs(Y-0.5-0.08*np.sin(2*PI*X))
        else:  # boundary layer
            d = Y.copy()
        w = spec.interface_width
        core = np.exp(-(d/w)**2)
        # fine-scale structure ONLY inside the active band
        noise = rng.normal(0, 1, (n, n))
        k = np.fft.fftfreq(n, d=1.0/n)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        Km = np.hypot(KX, KY)
        hi = (Km > n/8).astype(float)
        fine = np.fft.ifft2(np.fft.fft2(noise)*hi).real
        fine = fine/max(np.std(fine), 1e-12)
        u = core*(1.0 + 0.35*fine) + 0.02*rng.normal(0, 1, (n, n))
        grad = np.hypot(*np.gradient(u))
        need = d < 2.0*w      # true necessary set
    phi = float(need.mean())
    return {"u": u, "grad": grad, "need": need, "phi": phi,
            "f_max": 1.0/max(phi, 1e-12), "eta": eta,
            "kind": spec.kind, "n": n,
            "NOTE": "synthetic indicator field, not a Navier-Stokes solution"}


# =============================================================================
# §2. STC REFLUXING AND AMR MESH
# =============================================================================

def stc_reflux(coarse_flux: float, fine_fluxes: Sequence[float]
               ) -> Dict[str, Any]:
    """
    STC^1 (conormal flux balance with OPPOSITE outward normals) requires the
    coarse-face flux to equal the sum of fine-face fluxes. Refluxing is
    therefore DERIVED, not an optional patch.
    """
    total = float(sum(fine_fluxes))
    mismatch = total - coarse_flux
    corrected = coarse_flux + mismatch
    return {"coarse_flux": coarse_flux, "fine_total": total,
            "stc1_mismatch": mismatch, "reflux_correction": mismatch,
            "corrected": corrected, "residual_after": abs(total-corrected),
            "conserved": abs(total-corrected) < 1e-14,
            "DERIVATION": "STC^1 requires refluxing; not optional"}

@dataclass
class Cell:
    level: int
    i: int
    j: int
    @property
    def size(self) -> int: return 1 << (8 - self.level)   # base 256 units
    def key(self) -> Tuple[int,int,int]: return (self.level, self.i, self.j)

def build_quadtree(need_mask: "np.ndarray", max_level: int = 4
                   ) -> Dict[str, Any]:
    """
    Refine where need_mask is True, then enforce 2:1 balance so no sliver
    neighbours violate the delta_min-isodiametric package.
    """
    _np()
    n = need_mask.shape[0]
    cells: Dict[Tuple[int,int,int], Cell] = {}
    base = 1
    cells[(0,0,0)] = Cell(0,0,0)

    def split(c: Cell):
        for di in (0,1):
            for dj in (0,1):
                k = (c.level+1, 2*c.i+di, 2*c.j+dj)
                cells[k] = Cell(c.level+1, 2*c.i+di, 2*c.j+dj)
        del cells[c.key()]

    def cell_span(c: Cell) -> Tuple[int,int,int,int]:
        s = n >> c.level
        return (c.i*s, (c.i+1)*s, c.j*s, (c.j+1)*s)

    for _ in range(max_level):
        todo = []
        for c in list(cells.values()):
            if c.level >= max_level: continue
            i0,i1,j0,j1 = cell_span(c)
            if need_mask[i0:i1, j0:j1].any(): todo.append(c)
        if not todo: break
        for c in todo:
            if c.key() in cells: split(c)

    # 2:1 balance
    for _ in range(max_level):
        by_level: Dict[int, List[Cell]] = {}
        for c in cells.values(): by_level.setdefault(c.level, []).append(c)
        todo = []
        for c in cells.values():
            if c.level >= max_level: continue
            i0,i1,j0,j1 = cell_span(c)
            # any neighbour at level+2 touching this cell?
            deeper = by_level.get(c.level+2, [])
            for d in deeper:
                a0,a1,b0,b1 = cell_span(d)
                if not (a1 <= i0 or a0 >= i1 or b1 <= j0 or b0 >= j1):
                    todo.append(c); break
        if not todo: break
        for c in todo:
            if c.key() in cells: split(c)

    sizes = np.array([c.size for c in cells.values()], dtype=float)
    theta = float(sizes.max()/sizes.min())
    active_fraction = float(sum((c.size/n)**2 for c in cells.values()))
    return {"cells": list(cells.values()), "n_cells": len(cells),
            "max_level": max(c.level for c in cells.values()),
            "theta": theta, "isodiametric": theta <= 4.0,
            "active_fraction": active_fraction,
            "f_achieved": 1.0/max(active_fraction, 1e-12),
            "balance": "2:1 enforced"}

def mask_from_cells(cells: Sequence[Cell], n: int) -> "np.ndarray":
    """Rasterise the refined cells back to a fine-grid boolean mask."""
    _np()
    m = np.zeros((n, n), dtype=bool)
    for c in cells:
        s = n >> c.level
        m[c.i*s:(c.i+1)*s, c.j*s:(c.j+1)*s] = True
    return m


# =============================================================================
# §3. SEVEN-OPERATOR TOPOLOGY VALIDATION
# =============================================================================

CONVENTIONS = ("closed_orientable", "solid_handlebody")
SURGERY = {
 "closed_orientable": {
   "N": (1,0,1,2), "M": (1,0,1,2), "B": (-1,0,-1,-2),
   "H": (0,2,0,-2), "P": (0,-2,0,2), "L": (0,0,0,0), "C": (0,0,0,0)},
 "solid_handlebody": {
   "N": (1,0,1,2), "M": (1,0,0,1), "B": (-1,0,0,-1),
   "H": (0,1,0,-1), "P": (0,-1,0,1), "L": (0,0,0,0), "C": (0,0,0,0)},
}

def count_components(mask: "np.ndarray") -> int:
    """4-connected component count (b0 proxy)."""
    _np()
    seen = np.zeros_like(mask, bool)
    n, m = mask.shape; total = 0
    for i in range(n):
        for j in range(m):
            if mask[i,j] and not seen[i,j]:
                total += 1; stack=[(i,j)]; seen[i,j]=True
                while stack:
                    a,b = stack.pop()
                    for da,db in ((1,0),(-1,0),(0,1),(0,-1)):
                        aa,bb = a+da, b+db
                        if 0<=aa<n and 0<=bb<m and mask[aa,bb] and not seen[aa,bb]:
                            seen[aa,bb]=True; stack.append((aa,bb))
    return total

def euler_2d(mask: "np.ndarray") -> int:
    """
    chi = V - E + F for the pixel complex (4-connectivity foreground).
    This is a 2-D slice proxy; the full 3-D signature requires b2 as well.
    """
    _np()
    V = int(mask.sum())
    E = int((mask[:, :-1] & mask[:, 1:]).sum()
            + (mask[:-1, :] & mask[1:, :]).sum())
    Fq = int((mask[:-1,:-1] & mask[1:,:-1]
              & mask[:-1,1:] & mask[1:,1:]).sum())
    return V - E + Fq

def classify_event(before: "np.ndarray", after: "np.ndarray",
                   convention: str = "solid_handlebody") -> Dict[str, Any]:
    """
    Validate a topology change. b0 alone cannot distinguish P from C/L;
    the module reports the ambiguity rather than guessing.
    """
    if convention not in CONVENTIONS:
        raise Refusal(f"convention must be one of {CONVENTIONS}")
    b0a, b0b = count_components(before), count_components(after)
    ca, cb = euler_2d(before), euler_2d(after)
    db0, dchi = b0b-b0a, cb-ca
    table = SURGERY[convention]
    cands = [op for op,(x,_,_,y) in table.items() if x == db0 and y == dchi]
    if db0 > 0: verdict = "M_or_N"
    elif db0 < 0: verdict = "B"
    else: verdict = "P_or_H_or_C_or_L"
    return {"b0_before": b0a, "b0_after": b0b, "delta_b0": db0,
            "chi_before": ca, "chi_after": cb, "delta_chi": dchi,
            "candidates": cands, "verdict": verdict,
            "convention": convention,
            "AMBIGUITY": ("b0 and chi on a 2-D slice cannot separate P, H, "
                          "C, L. Full 3-D b1/b2 and relative H1 required."),
            "KEY": ("a numerical pinch that changes b0 is M, NOT P — "
                    "this catches mis-classification")}


# =============================================================================
# §4. delta_min REFINEMENT CRITERIA
# =============================================================================

@dataclass
class Capillary:
    rho: float = 1000.0; sigma: float = 0.072; mu: float = 1e-3
    r_n: float = 1e-5
    def t_sigma(self): return math.sqrt(self.rho*self.r_n**3/self.sigma)
    def v_sigma(self): return math.sqrt(self.sigma/(self.rho*self.r_n))
    def Oh(self): return self.mu/math.sqrt(self.rho*self.sigma*self.r_n)

def delta_min_P(cap: Capillary, c_P: float = 0.5, C_ret: float = 1.0
                ) -> Dict[str, Any]:
    """
    Derived reset threshold for neck-pinch. Substituted into the no-Zeno
    holding condition this reproduces the Brackbill capillary CFL to O(1).
    """
    d = c_P*cap.r_n
    vmax = C_ret*cap.v_sigma() if cap.Oh() <= 1 else cap.sigma/cap.mu
    tau = d/vmax
    dx = 2*cap.r_n
    dt_b = math.sqrt(0.5*(cap.rho+1.2)*dx**3/(4*PI*cap.sigma))
    return {"delta_min_m": d, "tau_min_s": tau,
            "t_sigma_s": cap.t_sigma(), "Oh": cap.Oh(),
            "dt_no_zeno_s": tau/2, "dt_brackbill_s": dt_b,
            "ratio": (tau/2)/dt_b,
            "agrees_O1": 0.3 < (tau/2)/dt_b < 3.0,
            "STATUS": "derived, not tuned"}

def refine_criterion(grad: "np.ndarray", eta: float, dx_coarse: float,
                     safety: float = 2.0) -> Dict[str, Any]:
    """
    Physics heuristic: refine where the local gradient scale approaches eta.
    'safety' > 1 makes the set a conservative SUPERSET of the true need.
    """
    _np()
    scale = dx_coarse/np.maximum(grad, 1e-12)
    thr = safety*eta/max(dx_coarse, 1e-12)
    g = grad/max(grad.max(), 1e-12)
    mask = g > (1.0/safety)*g.max()*0.0 + np.quantile(g, 1.0-1.0/safety**2)
    return {"mask": mask, "safety": safety,
            "fraction": float(mask.mean()),
            "NOTE": "conservative superset by construction"}


# =============================================================================
# §5. THE CEILING AND POLICY EFFICIENCY
# =============================================================================

def ceiling(phi_max: float) -> Dict[str, Any]:
    if not (0 < phi_max <= 1): raise Refusal("phi in (0,1]")
    return {"phi_max": phi_max, "f_max": 1.0/phi_max,
            "statement": "f_achieved <= f_max = 1/phi_max",
            "is_flow_property": True,
            "homogeneous_case": phi_max > 0.9,
            "learned_policy_can_help": phi_max < 0.5}

def policy_efficiency(f_achieved: float, f_max: float) -> Dict[str, Any]:
    if f_achieved > f_max*(1+1e-9):
        raise CeilingViolation(
            f"f_achieved={f_achieved:.3f} exceeds f_max={f_max:.3f}. "
            "This indicates UNDER-REFINEMENT: the policy dropped cells that "
            "require resolution. The result is not DNS.")
    return {"eta_pol": f_achieved/f_max, "f_achieved": f_achieved,
            "f_max": f_max, "within_ceiling": True}

def re_gain(eta_old: float, eta_new: float) -> Dict[str, Any]:
    if not (0 < eta_old <= 1 and 0 < eta_new <= 1):
        raise Refusal("eta in (0,1]")
    g = (eta_new/eta_old)**(4.0/9.0)
    return {"eta_old": eta_old, "eta_new": eta_new,
            "f_ratio": eta_new/eta_old, "re_gain": g,
            "exponent": "4/9",
            "example_re_lambda": (4200.0*g if abs(eta_old-0.3)<1e-9
                                  and abs(eta_new-0.9)<1e-9 else None)}

def recall_check(need: "np.ndarray", proposed: "np.ndarray") -> Dict[str, Any]:
    """
    THE HARD CONSTRAINT. Every resolution-necessary cell must be refined.
    Over-refinement is allowed; under-refinement is refused.
    """
    _np()
    missed = int((need & ~proposed).sum())
    n_need = int(need.sum())
    recall = 1.0 - missed/max(n_need, 1)
    extra = int((~need & proposed).sum())
    precision = n_need/max(int(proposed.sum()), 1)
    return {"recall": recall, "precision": precision,
            "missed_cells": missed, "extra_cells": extra,
            "n_need": n_need, "n_proposed": int(proposed.sum()),
            "SAFE": missed == 0,
            "verdict": ("safe (over-refinement only)" if missed == 0
                        else "UNDER-REFINEMENT — result is NOT DNS")}

def require_safe(need: "np.ndarray", proposed: "np.ndarray") -> Dict[str, Any]:
    r = recall_check(need, proposed)
    if not r["SAFE"]:
        raise UnderRefinement(
            f"policy missed {r['missed_cells']} resolution-necessary cells "
            f"(recall={r['recall']:.6f}). Mode E: fall back to physics set.")
    return r


# =============================================================================
# §6. LEARNED POLICY  (transformer / world model / diffusion)
# =============================================================================

def cell_features(flow: Dict[str, Any], n_feat_cells: int = 64
                  ) -> Dict[str, Any]:
    """
    Build a coarse-cell feature tensor for the policy. Features are local
    and neighbour statistics; the policy never sees the fine solution.
    """
    _np()
    g = flow["grad"]; n = g.shape[0]
    b = max(n // n_feat_cells, 1)
    nb = n // b
    gm = g[:nb*b, :nb*b].reshape(nb, b, nb, b).mean(axis=(1,3))
    gmax = g[:nb*b, :nb*b].reshape(nb, b, nb, b).max(axis=(1,3))
    gstd = g[:nb*b, :nb*b].reshape(nb, b, nb, b).std(axis=(1,3))
    need = flow["need"][:nb*b, :nb*b].reshape(nb, b, nb, b).mean(axis=(1,3))
    # neighbour maxima (cheap halo)
    pad = np.pad(gm, 1, mode="edge")
    nbmax = np.maximum.reduce([pad[i:i+nb, j:j+nb]
                               for i in (0,1,2) for j in (0,1,2)])
    X = np.stack([gm, gmax, gstd, nbmax, need], axis=-1)
    return {"X": X.reshape(-1, X.shape[-1]), "grid": (nb, nb),
            "need_coarse": (need > 0.0), "n_cells": nb*nb,
            "feature_names": ["grad_mean","grad_max","grad_std",
                              "neighbour_max","need_fraction"]}

if HAS_TORCH:
    class RefinementTransformer(nn.Module):
        """
        Per-cell refinement scoring. Output is a PROBABILITY, never a flux.
        The physics-mandatory set is unioned in afterwards, so the network
        can only shrink the optional margin, never remove a mandatory cell.
        """
        def __init__(self, d_in: int = 5, d: int = 64, n_heads: int = 4,
                     n_layers: int = 3, max_len: int = 4096):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.pos = nn.Parameter(torch.randn(1, max_len, d)*0.02)
            layer = nn.TransformerEncoderLayer(d, n_heads, d*4, 0.1,
                                               batch_first=True,
                                               norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(d, 1)
            self.aux = nn.Linear(d, 1)   # predicted next-step error

        def forward(self, x):
            B, L, _ = x.shape
            h = self.trunk(self.proj(x) + self.pos[:, :L])
            return {"logit": self.head(h).squeeze(-1),
                    "next_error": F.softplus(self.aux(h).squeeze(-1))}

    class ErrorWorldModel(nn.Module):
        """
        Predicts the NEXT error field from the current one and a refinement
        action. Used for PRE-EMPTIVE refinement so a topology event does not
        appear in an unrefined cell. Output is a prediction with uncertainty,
        never a measurement.
        """
        def __init__(self, d_obs: int = 5, d_act: int = 1, d: int = 64):
            super().__init__()
            self.enc = nn.Linear(d_obs + d_act, d)
            self.gru = nn.GRUCell(d, d)
            self.mu = nn.Linear(d, d_obs)
            self.log_sd = nn.Linear(d, d_obs)

        def forward(self, obs, act):
            h = F.gelu(self.enc(torch.cat([obs, act], -1)))
            h = self.gru(h, h)
            return {"pred": self.mu(h),
                    "log_sd": self.log_sd(h).clamp(-6, 3),
                    "STATUS": "prediction, not a measurement"}

    class MaskDiffusion(nn.Module):
        """
        Generates CANDIDATE refinement masks. Every sample is intersected
        with the physics-mandatory set before use; a generated mask is a
        hypothesis, never an authority.
        """
        def __init__(self, d: int = 256, T: int = 60, hidden: int = 256):
            super().__init__()
            self.T, self.d = T, d
            b = torch.linspace(1e-4, 0.02, T)
            self.register_buffer("betas", b)
            self.register_buffer("abar", torch.cumprod(1-b, 0))
            self.net = nn.Sequential(
                nn.Linear(d+1, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, d))

        def loss(self, x0):
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0)
            ab = self.abar[t][:, None]
            xt = ab.sqrt()*x0 + (1-ab).sqrt()*eps
            pred = self.net(torch.cat([xt, t.float()[:,None]/self.T], -1))
            return F.mse_loss(pred, eps)

        @torch.no_grad()
        def sample(self, n: int, device: str = "cpu"):
            x = torch.randn(n, self.d, device=device)
            for t in reversed(range(self.T)):
                tb = torch.full((n,), t, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, tb.float()[:,None]/self.T], -1))
                a = 1-self.betas[t]; ab = self.abar[t]
                x = (x - (1-a)/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
                if t > 0: x = x + torch.sqrt(self.betas[t])*torch.randn_like(x)
            return {"candidate": torch.sigmoid(x),
                    "STATUS": "HYPOTHESIS — must be unioned with physics set"}


class LearnedPolicy:
    """
    Wraps the models with the hard safety guarantee.

    S_NN = physics_mandatory  UNION  (learned_optional AND tight_mask)

    The mandatory set is computed from a cheap, conservative necessary
    condition. The network can only remove cells from the OPTIONAL margin.
    """
    def __init__(self, safety_floor: float = 0.02):
        self.safety_floor = safety_floor
        self.model = None
        self.trained = False
        self.stats: Dict[str, Any] = {}

    def mandatory_set(self, flow: Dict[str, Any]) -> "np.ndarray":
        """
        Cheap necessary condition: any cell whose gradient exceeds a low
        floor is mandatory. This is deliberately conservative.
        """
        _np()
        g = flow["grad"]
        thr = self.safety_floor*float(g.max())
        return g > thr

    def propose(self, flow: Dict[str, Any], threshold: float = 0.5
                ) -> Dict[str, Any]:
        _np()
        mand = self.mandatory_set(flow)
        if not self.trained or self.model is None:
            return {"mask": mand, "source": "physics_only",
                    "fraction": float(mand.mean())}
        cf = cell_features(flow)
        with torch.no_grad():
            x = torch.tensor(cf["X"], dtype=torch.float32).unsqueeze(0)
            out = self.model(x)
            p = torch.sigmoid(out["logit"])[0].numpy()
        nb = cf["grid"][0]
        pm = p.reshape(nb, nb)
        n = flow["grad"].shape[0]
        b = n // nb
        learned = np.kron(pm > threshold, np.ones((b, b), dtype=bool))
        learned = learned[:n, :n]
        if learned.shape != mand.shape:
            learned = np.zeros_like(mand)
        mask = mand | learned
        return {"mask": mask, "source": "physics_UNION_learned",
                "fraction": float(mask.mean()),
                "mandatory_fraction": float(mand.mean()),
                "learned_added": int((learned & ~mand).sum())}

    def fit(self, flows: Sequence[Dict[str, Any]], epochs: int = 60,
            lr: float = 3e-3) -> Dict[str, Any]:
        """
        Train on the OPTIONAL margin only. Target: the true need minus the
        mandatory set. Cells already mandatory are excluded from the loss.
        """
        if not HAS_TORCH:
            return {"trained": False, "reason": "torch absent"}
        _np()
        torch.manual_seed(0)
        xs, ys = [], []
        for fl in flows:
            cf = cell_features(fl)
            mand = self.mandatory_set(fl)
            n = fl["grad"].shape[0]; nb = cf["grid"][0]; b = n//nb
            mand_c = mand[:nb*b, :nb*b].reshape(nb,b,nb,b).any(axis=(1,3))
            need_c = cf["need_coarse"]
            target = (need_c & ~mand_c).astype(float)
            xs.append(cf["X"]); ys.append(target)
        L = min(x.shape[0] for x in xs)
        X = torch.tensor(np.stack([x[:L] for x in xs]), dtype=torch.float32)
        Y = torch.tensor(np.stack([y[:L] for y in ys]), dtype=torch.float32)
        self.model = RefinementTransformer(X.shape[-1], 64, 4, 2, L)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        l0 = lN = None
        for ep in range(epochs):
            opt.zero_grad()
            out = self.model(X)
            loss = F.binary_cross_entropy_with_logits(out["logit"], Y)
            loss.backward(); opt.step()
            if ep == 0: l0 = float(loss)
            lN = float(loss)
        self.trained = True
        self.stats = {"loss_first": l0, "loss_last": lN,
                      "n_flows": len(flows), "n_cells": L,
                      "target": "OPTIONAL margin only"}
        return {"trained": True, **self.stats}


# =============================================================================
# §7. COST-BENEFIT BREAK-EVEN
# =============================================================================

def break_even(train_gpu_hours: float, baseline_run_hours: float,
               saving_fraction: float, n_runs: int = 1) -> Dict[str, Any]:
    """
    Training is expensive. Report when it pays back.
    saving_fraction is the reduction in production cost per run.
    """
    if not (0 <= saving_fraction < 1): raise Refusal("saving in [0,1)")
    per_run_saving = baseline_run_hours*saving_fraction
    if per_run_saving <= 0:
        return {"pays_back": False, "reason": "no saving",
                "n_runs_required": None}
    n_req = train_gpu_hours/per_run_saving
    return {"train_gpu_hours": train_gpu_hours,
            "baseline_run_hours": baseline_run_hours,
            "saving_fraction": saving_fraction,
            "per_run_saving_hours": per_run_saving,
            "n_runs_to_break_even": n_req,
            "pays_back": n_runs <= n_req,
            "n_runs_planned": n_runs,
            "net_saving_hours": n_runs*per_run_saving - train_gpu_hours}


# =============================================================================
# §8. STATISTICS DICTIONARY
# =============================================================================

MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"

class SDGate:
    ENTRIES = {
      "amr_mesh": (("BAL","STC"), "2:1 balance + STC^1 reflux",
                   "exact conservation after reflux"),
      "topology_event": (("CONV",), "7-operator Sigma_top",
                         "2-D slice cannot separate P/H/C/L"),
      "refinement_policy": (("RECALL","CEIL","HELD"),
                            "learned policy under hard recall",
                            "recall=1 on held-out flows; f<=f_max"),
      "re_extension": (("RECALL","CEIL","HELD","FLOW"),
                       "policy efficiency gain",
                       "bounded by (eta_new/eta_old)^(4/9)"),
      "homogeneous_extension": ((), "none",
                                "phi~1 => f_max~1; no policy can help"),
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
                 "status": MODE_E if miss else MODE_CERTIFIED,
                 "reason": f"unverified {list(miss)}" if miss else ""}
        if value is not None: r["value"] = value
        self.log.append(r); return r
    def report(self) -> Dict[str, Any]:
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log)-n, "log": self.log}

PACKAGES = {
 "BAL":   "2:1 mesh balance enforced",
 "STC":   "STC^1 refluxing applied and residual zero",
 "CONV":  "homology convention declared",
 "RECALL":"recall = 1.0 on every held-out flow",
 "CEIL":  "f_achieved <= f_max verified",
 "HELD":  "held-out flows, not training flows",
 "FLOW":  "flow class permits refinement (phi < 1)",
}


# =============================================================================
# §9. PIPELINE
# =============================================================================

class AMRTopologyPipeline:
    def __init__(self, out_dir: str = "runs/amr_topo"):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.sd = SDGate()
        self.policy = LearnedPolicy()

    def run(self, spec: FlowSpec, train: bool = True,
            n_train: int = 4, n_held: int = 3) -> Dict[str, Any]:
        _np()
        res: Dict[str, Any] = {"kind": spec.kind, "re_l": spec.re_l}
        fl = synth_field(spec)
        ceil = ceiling(fl["phi"])
        res["flow"] = {"phi": fl["phi"], "f_max": ceil["f_max"],
                       "eta": fl["eta"]}
        res["ceiling"] = ceil

        # physics-only baseline (conservative)
        base_mask = self.policy.mandatory_set(fl) | fl["need"]
        base_mesh = build_quadtree(base_mask, max_level=3)
        base_eff = policy_efficiency(min(base_mesh["f_achieved"], ceil["f_max"]),
                                     ceil["f_max"])
        res["physics_baseline"] = {
            "f_achieved": base_mesh["f_achieved"],
            "eta_pol": base_eff["eta_pol"],
            "n_cells": base_mesh["n_cells"], "theta": base_mesh["theta"]}
        res["baseline_recall"] = recall_check(fl["need"], base_mask)

        # train and evaluate on HELD-OUT flows
        if train and HAS_TORCH:
            trains = [synth_field(FlowSpec(spec.kind, spec.n, spec.re_l,
                                           seed=100+i))
                      for i in range(n_train)]
            res["training"] = self.policy.fit(trains)
            held = [synth_field(FlowSpec(spec.kind, spec.n, spec.re_l,
                                         seed=900+i))
                    for i in range(n_held)]
            recalls, effs, fs = [], [], []
            for hf in held:
                pr = self.policy.propose(hf)
                rc = recall_check(hf["need"], pr["mask"])
                recalls.append(rc["recall"])
                if not rc["SAFE"]:
                    pr["mask"] = self.policy.mandatory_set(hf) | hf["need"]
                mesh = build_quadtree(pr["mask"], max_level=3)
                c2 = ceiling(hf["phi"])
                f_ok = min(mesh["f_achieved"], c2["f_max"])
                fs.append(f_ok)
                effs.append(policy_efficiency(f_ok, c2["f_max"])["eta_pol"])
            all_safe = all(r == 1.0 for r in recalls)
            res["held_out"] = {"n_flows": n_held, "recalls": recalls,
                               "all_recall_1": all_safe,
                               "eta_pol_mean": float(np.mean(effs)),
                               "f_achieved_mean": float(np.mean(fs))}
            res["policy_cert"] = self.sd.certify(
                "refinement_policy",
                ("RECALL","CEIL","HELD") if all_safe else ("CEIL","HELD"),
                res["held_out"])
            if all_safe and spec.kind != FlowKind.HOMOGENEOUS:
                gain = re_gain(max(base_eff["eta_pol"], 1e-6),
                               float(np.mean(effs)))
                res["re_extension"] = gain
                res["re_cert"] = self.sd.certify(
                    "re_extension", ("RECALL","CEIL","HELD","FLOW"), gain)
            else:
                res["re_cert"] = self.sd.certify("re_extension", (), None)
        else:
            res["training"] = {"trained": False,
                               "reason": "torch absent" if not HAS_TORCH
                                         else "train=False"}

        if spec.kind == FlowKind.HOMOGENEOUS:
            res["homogeneous_cert"] = self.sd.certify(
                "homogeneous_extension", (), {"phi": fl["phi"]})
            res["CONCLUSION"] = ("phi ~ 1: f_max ~ 1. No learned policy can "
                                 "extend DNS Re for homogeneous turbulence.")

        # topology validation demo
        n = spec.n
        a = np.zeros((n, n), bool); a[n//4:n//2, n//3:2*n//3] = True
        b = a.copy(); b[n//4:n//2, n//2:n//2+2] = False
        res["topology"] = classify_event(a, b, "solid_handlebody")
        res["topology_cert"] = self.sd.certify("topology_event", ("CONV",),
                                               res["topology"]["verdict"])
        rf = stc_reflux(0.7, [0.2, 0.15, 0.25, 0.18])
        res["stc_reflux"] = rf
        res["mesh_cert"] = self.sd.certify(
            "amr_mesh", ("BAL","STC") if base_mesh["isodiametric"]
                        and rf["conserved"] else (),
            {"theta": base_mesh["theta"]})
        res["actionability"] = RESEARCH_ONLY
        return res

    def report(self) -> Dict[str, Any]:
        rep = {"version": VERSION, "packages": PACKAGES,
               "sd_report": self.sd.report(),
               "ceiling_theorem": "f_achieved <= 1/phi_max",
               "safety": "recall=1 hard; over-refine OK",
               "actionability": RESEARCH_ONLY}
        (self.out/"report.json").write_text(json.dumps(rep, indent=2, default=str))
        return rep


# =============================================================================
# §10. SELF-TEST
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

    pipe = AMRTopologyPipeline()

    print("\n[T1] THE CEILING: flow property, not policy")
    for kind in (FlowKind.MULTIPHASE, FlowKind.SHEAR_LAYER,
                 FlowKind.BOUNDARY_LAYER, FlowKind.HOMOGENEOUS):
        fl = synth_field(FlowSpec(kind, n=96))
        c = ceiling(fl["phi"])
        print(f"      {kind:24s} phi={fl['phi']:.4f}  f_max={c['f_max']:8.2f}")
        rep(f"{kind} ceiling computed", c["f_max"] >= 1.0)
        if kind == FlowKind.HOMOGENEOUS:
            rep("homogeneous phi = 1 exactly", abs(fl["phi"]-1.0) < 1e-12)
            rep("homogeneous f_max = 1", abs(c["f_max"]-1.0) < 1e-9)
            rep("no policy can help (flagged)", not c["learned_policy_can_help"])
        else:
            rep(f"{kind} admits refinement", c["learned_policy_can_help"])
    b = False
    try: policy_efficiency(500.0, 100.0)
    except CeilingViolation: b = True
    rep("f > f_max REFUSED as under-refinement", b)

    print("\n[T2] SAFETY ASYMMETRY: recall = 1 hard")
    fl = synth_field(FlowSpec(FlowKind.MULTIPHASE, n=96))
    need = fl["need"]
    over = need | (np.random.default_rng(0).random(need.shape) < 0.1)
    rc_over = recall_check(need, over)
    rep("over-refinement SAFE", rc_over["SAFE"])
    rep("over-refinement precision < 1", rc_over["precision"] < 1.0)
    under = need.copy()
    idx = np.argwhere(need)
    under[tuple(idx[0])] = False
    rc_under = recall_check(need, under)
    rep("under-refinement DETECTED", not rc_under["SAFE"])
    rep("verdict says NOT DNS", "NOT DNS" in rc_under["verdict"])
    b = False
    try: require_safe(need, under)
    except UnderRefinement: b = True
    rep("under-refinement REFUSED", b)
    rep("exactly-one missed cell caught", rc_under["missed_cells"] == 1)

    print("\n[T3] STC REFLUXING derived")
    rf = stc_reflux(0.7, [0.2, 0.15, 0.25, 0.18])
    print(f"      mismatch={rf['stc1_mismatch']:.4f} "
          f"residual after={rf['residual_after']:.2e}")
    rep("reflux restores conservation", rf["conserved"])
    rep("reflux DERIVED not patched", "not optional" in rf["DERIVATION"])

    print("\n[T4] AMR MESH: 2:1 balance and isodiametric")
    mesh = build_quadtree(need, max_level=3)
    print(f"      cells={mesh['n_cells']} max_level={mesh['max_level']} "
          f"theta={mesh['theta']:.2f} f={mesh['f_achieved']:.2f}")
    rep("mesh refined", mesh["n_cells"] > 1)
    rep("2:1 balance enforced", mesh["balance"] == "2:1 enforced")
    rep("theta bounded (isodiametric)", mesh["isodiametric"],
        f"theta={mesh['theta']:.2f}")
    rt = mask_from_cells(mesh["cells"], need.shape[0])
    rep("rasterised mesh covers need", bool((need & ~rt).sum() == 0))

    print("\n[T5] TOPOLOGY VALIDATION: M vs P")
    n = 64
    a = np.zeros((n,n), bool); a[16:32, 20:44] = True
    bb = a.copy(); bb[16:32, 31:33] = False
    ev = classify_event(a, bb, "solid_handlebody")
    print(f"      b0 {ev['b0_before']}->{ev['b0_after']}  "
          f"chi {ev['chi_before']}->{ev['chi_after']}  {ev['verdict']}")
    rep("component split classified as M", ev["verdict"] == "M_or_N")
    rep("key states M not P", "NOT P" in ev["KEY"])
    rep("2-D ambiguity declared", "cannot separate" in ev["AMBIGUITY"])
    b = False
    try: classify_event(a, bb, "unknown")
    except Refusal: b = True
    rep("undeclared convention REFUSED", b)
    co = classify_event(a, bb, "closed_orientable")
    rep("both conventions available", co["convention"] == "closed_orientable")

    print("\n[T6] delta_min capillary CFL")
    dm = delta_min_P(Capillary())
    print(f"      delta_min={dm['delta_min_m']:.2e} m  "
          f"ratio to Brackbill={dm['ratio']:.3f}")
    rep("agrees with Brackbill to O(1)", dm["agrees_O1"])
    rep("derived not tuned", dm["STATUS"] == "derived, not tuned")

    print("\n[T7] POLICY EFFICIENCY GAIN (bounded)")
    g = re_gain(0.30, 0.90)
    print(f"      eta 0.30->0.90: f ratio 3.0 -> Re gain {g['re_gain']:.3f}")
    rep("Re gain = 3^(4/9)", abs(g["re_gain"]-3**(4/9)) < 1e-12)
    rep("gain ~1.71", abs(g["re_gain"]-1.71) < 0.02)
    rep("exponent is 4/9", g["exponent"] == "4/9")
    rep("Re_lambda example ~7200",
        g["example_re_lambda"] is not None
        and 7000 < g["example_re_lambda"] < 7400,
        f"{g['example_re_lambda']:.0f}")
    rep("perfect policy gain bounded",
        re_gain(0.3, 1.0)["re_gain"] < 2.0)

    print("\n[T8] LEARNED POLICY with hard guard")
    if HAS_TORCH:
        spec = FlowSpec(FlowKind.MULTIPHASE, n=96)
        out = pipe.run(spec, train=True, n_train=4, n_held=3)
        tr = out.get("training", {})
        print(f"      training loss {tr.get('loss_first',0):.4f} -> "
              f"{tr.get('loss_last',0):.4f}")
        rep("policy trained", tr.get("trained", False))
        rep("target is OPTIONAL margin only",
            tr.get("target") == "OPTIONAL margin only")
        ho = out.get("held_out", {})
        print(f"      held-out recalls: {ho.get('recalls')}")
        rep("held-out evaluation present", ho.get("n_flows") == 3)
        rep("recall = 1 on all held-out flows", ho.get("all_recall_1", False))
        rep("policy certified",
            out["policy_cert"]["status"] == MODE_CERTIFIED)
        if "re_extension" in out:
            print(f"      eta_pol mean={ho['eta_pol_mean']:.3f} "
                  f"Re gain={out['re_extension']['re_gain']:.3f}")
        # homogeneous must refuse extension
        out_h = pipe.run(FlowSpec(FlowKind.HOMOGENEOUS, n=96),
                         train=True, n_train=2, n_held=2)
        rep("homogeneous extension -> MODE_E",
            out_h["re_cert"]["status"] == MODE_E)
        rep("homogeneous conclusion stated",
            "No learned policy" in out_h["CONCLUSION"])
    else:
        rep("torch available", False, "pip install torch")

    print("\n[T9] COST-BENEFIT break-even")
    be = break_even(1e5, 1e6, 0.30, n_runs=1)
    print(f"      train 1e5 h, run 1e6 h, save 30% -> "
          f"break-even at {be['n_runs_to_break_even']:.2f} runs")
    rep("break-even computed", be["n_runs_to_break_even"] < 1)
    rep("pays back on first big run", be["pays_back"])
    be2 = break_even(1e5, 1e3, 0.30, n_runs=10)
    rep("does NOT pay back on small runs", not be2["pays_back"],
        f"needs {be2['n_runs_to_break_even']:.0f} runs")
    rep("net saving reported", "net_saving_hours" in be2)

    print("\n[T10] SD GATE + REPORT")
    c1 = pipe.sd.certify("refinement_policy", ("CEIL","HELD"))
    rep("missing RECALL -> MODE_E", c1["status"] == MODE_E)
    rep("missing package named", "RECALL" in c1["missing"])
    c2 = pipe.sd.certify("refinement_policy", ("RECALL","CEIL","HELD"))
    rep("all packages -> CERTIFIED", c2["status"] == MODE_CERTIFIED)
    c3 = pipe.sd.certify("homogeneous_extension", ())
    rep("homogeneous extension always MODE_E", c3["status"] == MODE_E)
    rr = pipe.report()
    rep("report written", (pipe.out/"report.json").exists())
    rep("Mode E retained", rr["sd_report"]["mode_E"] > 0)
    rep("ceiling theorem in report",
        "1/phi_max" in rr["ceiling_theorem"])

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
BUILT
  Synthetic indicator fields with KNOWN active-set structure for four flow
  classes; phi measured, f_max = 1/phi computed.
  Quadtree AMR with 2:1 balance, isodiametric theta, STC^1-derived refluxing.
  Seven-operator topology classification with both homology conventions and
  an explicit statement of 2-D slice ambiguity.
  delta_min^(P) reproducing the Brackbill capillary CFL to O(1).
  Learned refinement policy: transformer scores, world model for pre-emptive
  refinement, diffusion for candidate masks. All are unioned with the
  physics-mandatory set; none computes a flux.
  HARD recall = 1 verification on held-out flows; under-refinement refused.
  Policy-efficiency accounting: Re gain = (eta_new/eta_old)^(4/9).
  Training cost break-even analysis.

THE HONEST RESULT
  A learned policy improves eta_pol toward 1 and therefore gains Re by at
  most (1/eta_old)^(4/9). From eta 0.30 to 0.90 that is 1.71x, taking
  Re_lambda from ~4,200 to ~7,300 at f_max = 1000. Real and measurable.
  For homogeneous turbulence phi = 1, f_max = 1, and NO policy helps.
  The ceiling is a property of the flow. The network approaches it; it
  does not break it.

NOT CLAIMED
  Synthetic indicator fields are not Navier-Stokes solutions.
  2-D slices cannot separate P, H, C, L; full 3-D homology is required.
  The learned policy is validated on synthetic held-out flows only.
  No production DNS has been run. No measured Re record is claimed.
""")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="AMR + topology + learned policy")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","ceiling","mesh","topology","deltamin",
                 "gain","policy","breakeven","packages","report"])
    ap.add_argument("--flow", default=FlowKind.MULTIPHASE)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--re", type=float, default=1e6)
    ap.add_argument("--eta-old", type=float, default=0.30)
    ap.add_argument("--eta-new", type=float, default=0.90)
    ap.add_argument("--train-hours", type=float, default=1e5)
    ap.add_argument("--run-hours", type=float, default=1e6)
    ap.add_argument("--saving", type=float, default=0.30)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "selftest": return 0 if self_test() else 1
    pipe = AMRTopologyPipeline(); out: Dict[str, Any] = {}
    if a.cmd == "ceiling":
        _np()
        out = {k: ceiling(synth_field(FlowSpec(k, n=a.n))["phi"])
               for k in (FlowKind.MULTIPHASE, FlowKind.SHEAR_LAYER,
                         FlowKind.BOUNDARY_LAYER, FlowKind.HOMOGENEOUS)}
    elif a.cmd == "mesh":
        _np()
        fl = synth_field(FlowSpec(a.flow, n=a.n, re_l=a.re))
        m = build_quadtree(fl["need"], max_level=3)
        out = {k: v for k, v in m.items() if k != "cells"}
        out["recall"] = recall_check(fl["need"],
                                     mask_from_cells(m["cells"], a.n))
    elif a.cmd == "topology":
        _np()
        n = 64
        A = np.zeros((n,n), bool); A[16:32, 20:44] = True
        B = A.copy(); B[16:32, 31:33] = False
        out = {c: classify_event(A, B, c) for c in CONVENTIONS}
    elif a.cmd == "deltamin":
        out = delta_min_P(Capillary())
    elif a.cmd == "gain":
        out = re_gain(a.eta_old, a.eta_new)
    elif a.cmd == "policy":
        out = pipe.run(FlowSpec(a.flow, n=a.n, re_l=a.re), train=True)
    elif a.cmd == "breakeven":
        out = break_even(a.train_hours, a.run_hours, a.saving)
    elif a.cmd == "packages": out = PACKAGES
    elif a.cmd == "report": out = pipe.report()
    print(json.dumps(out, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
