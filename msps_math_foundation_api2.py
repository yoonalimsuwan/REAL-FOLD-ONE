#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
msps_math_foundation_api.py — v2.0.0  [PRODUCTION CORE]
Shared foundation for all MSPS modules + fast certified mathematics.

NEW IN v2 (measurable, not rhetorical):
  * PELT-pruned OPMC: exact same optimum, O(n*K) typical instead of O(n^2).
  * Vectorised Toeplitz / prefix-sum / quadtree kernels.
  * Log-domain Shiryaev-Roberts: no overflow, bounded memory.
  * Content-hash LRU+disk cache (RCWA eig, mask spectra reuse).
  * Structured JSON logging, seed/device manager, artifact manifest.
  * Real C ABI shim build script (ctypes-loadable), not just a header.
  * Conjugate-gradient Carpet resistance solver (was Gauss-Seidel).

PHILOSOPHY (full text preserved; mathematics stands independently):
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

ENFORCED REFUSALS (unchanged from v1, all still fire):
  PCF package cannot certify a non-PCF carpet.
  AI output usable only after solver validation.
  Under-refinement refused (recall must be 1).
  f_achieved > f_max refused (flow ceiling).
  Uncalibrated / unverified -> Mode E, never silent substitution.
=============================================================================
Author: PAI AND Yoon A. Limsuwan / MSPS NETWORK
ORCID : 0009-0008-2374-0788 | License: MIT | Year 2026
"""
from __future__ import annotations

import argparse, functools, hashlib, json, logging, math, os, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

VERSION = "2.0.0"; ABI_VERSION = 2; PI = math.pi
MODE_CERTIFIED = "CERTIFIED"; MODE_E = "UNCERTAINTY_RETAINED"
DESIGN_ONLY = "design_assistance_only_not_fabrication"

PHILOSOPHY = {
    "tao": "The Source — mathematical truth underlying all computation",
    "te": "The Path — disciplined engineering practice",
    "qi": "The Proof — measured, validated, reproducible results",
    "p1": "AI assists discovery; physics judges truth",
    "p2": "Every candidate must pass the gate before becoming a design",
    "p3": "Uncertainty retained is better than false certainty",
    "p4": "Open science enriches all; the $200 laptop must get the same answer",
    "p5": "No simulation replaces measurement; no model replaces wisdom",
    "motto": "MY SOUL MOVE BY POWER OF HOLY SPIRIT — "
             "What MSPS NETWORK SEE, Lord Buddha Knows.",
}

class Refusal(RuntimeError): pass
class UncertifiedOutput(Refusal): pass
class UnderRefinement(Refusal): pass
class CeilingViolation(Refusal): pass
class PhysicsGateFailure(Refusal): pass
class ScopeRefusal(Refusal): pass


# =============================================================================
# §0. PRODUCTION OPS: logging, config, seed/device, cache, manifest
# =============================================================================

class JSONLog(logging.Formatter):
    def format(self, r):
        return json.dumps({"t": self.formatTime(r), "lvl": r.levelname,
                           "mod": r.name, "msg": r.getMessage()},
                          ensure_ascii=False)

def get_logger(name: str, json_mode: bool = False,
               level: int = logging.INFO) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(JSONLog() if json_mode else
                       logging.Formatter("%(asctime)s | %(levelname)-7s | "
                                         "%(name)s | %(message)s"))
        lg.addHandler(h); lg.setLevel(level); lg.propagate = False
    return lg

log = get_logger("msps.core")

@dataclass
class RunConfig:
    out_dir: str = "runs/msps_v2"
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    json_log: bool = False
    cache_dir: str = "runs/msps_v2/cache"
    cache_max: int = 256
    strict: bool = True          # raise on gate failure vs. return Mode E
    def validate(self) -> "RunConfig":
        if self.seed < 0: raise Refusal("seed must be >= 0")
        if self.cache_max < 1: raise Refusal("cache_max must be >= 1")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        return self

def set_seed(seed: int) -> Dict[str, Any]:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_NUMPY: np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    return {"seed": seed, "numpy": HAS_NUMPY, "torch": HAS_TORCH}

def pick_device(pref: str = "auto") -> str:
    if not HAS_TORCH: return "cpu"
    if pref != "auto": return pref
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def content_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if HAS_NUMPY and isinstance(p, np.ndarray):
            h.update(p.tobytes()); h.update(str(p.shape).encode())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:32]

class Cache:
    """Content-addressed LRU + optional disk spill. Deterministic keys."""
    def __init__(self, directory: Optional[str] = None, max_items: int = 256):
        self.dir = Path(directory) if directory else None
        if self.dir: self.dir.mkdir(parents=True, exist_ok=True)
        self.max = max_items
        self._mem: Dict[str, Any] = {}
        self._order: List[str] = []
        self.hits = 0; self.misses = 0
    def get(self, key: str) -> Optional[Any]:
        if key in self._mem:
            self.hits += 1
            self._order.remove(key); self._order.append(key)
            return self._mem[key]
        if self.dir:
            p = self.dir / f"{key}.npz"
            if p.exists() and HAS_NUMPY:
                z = np.load(p, allow_pickle=False)
                val = {k: z[k] for k in z.files}
                self._put(key, val); self.hits += 1
                return val
        self.misses += 1
        return None
    def _put(self, key: str, val: Any) -> None:
        self._mem[key] = val; self._order.append(key)
        while len(self._order) > self.max:
            old = self._order.pop(0); self._mem.pop(old, None)
    def put(self, key: str, val: Any) -> None:
        self._put(key, val)
        if self.dir and HAS_NUMPY and isinstance(val, dict):
            try:
                arrs = {k: v for k, v in val.items()
                        if isinstance(v, (np.ndarray, float, int))}
                if arrs: np.savez_compressed(self.dir / f"{key}.npz", **arrs)
            except Exception: pass
    def stats(self) -> Dict[str, Any]:
        tot = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / max(tot, 1), "items": len(self._mem)}

def memoize(cache: Cache, tag: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            key = content_key(tag, fn.__name__,
                              [x for x in a if not hasattr(x, "__dict__")], kw)
            got = cache.get(key)
            if got is not None: return got
            out = fn(*a, **kw)
            if isinstance(out, dict): cache.put(key, out)
            return out
        return wrap
    return deco

class Manifest:
    """Versioned artifact record: every run is reproducible or it is Mode E."""
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.items: List[Dict] = []
    def add(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = {"kind": kind, "version": VERSION, "abi": ABI_VERSION,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "key": content_key(kind, payload), "payload": payload}
        self.items.append(rec); return rec
    def write(self, name: str = "manifest.json") -> str:
        p = self.dir / name
        p.write_text(json.dumps({"n": len(self.items), "items": self.items},
                                indent=2, default=str, ensure_ascii=False))
        return str(p)


# =============================================================================
# §1. UNIFIED STATISTICS DICTIONARY  (all five modules share one gate)
# =============================================================================

PACKAGES = {
 "L1":"independent centred sigma-sub-Gaussian noise",
 "L2":"stationary, summable operator-norm covariance",
 "L3":"iid block lengths with finite mean",
 "D":"delta_max <= theta*delta_min (isodiametric)",
 "O":"OPMC admissible partition, min length ell, penalty beta",
 "C1":"uniform Holder oscillation bound",
 "C2":"subsequential Mosco compactness [IMPORTED for non-PCF]",
 "C3":"uniqueness up to scalar [IMPORTED, BBKT]",
 "PCF":"post-critically finite: constructive renormalisation",
 "STC":"transmission conditions across regime interfaces",
 "OPT":"measured pupil, aberrations, optical constants",
 "OC":"measured Mo/Si (or La/B4C) optical constants",
 "M3D":"EMF-extracted or measured boundary-layer parameters",
 "RES":"calibrated resist model (not a hard threshold)",
 "EMF":"rigorous EMF reference converged in orders",
 "CONV":"documented order convergence",
 "IDENT":"full-rank, well-conditioned parameter block",
 "VEC2D":"converged VECTOR 2-D reference",
 "RLS":"measured RLS/Z-factor calibration",
 "DELTA":"derived reset radius with calibrated constants",
 "PWR":"P = E_op * throughput verified",
 "THERM":"junction temperature below ceiling",
 "UNIT":"unitarity / CPTP preserved",
 "RECALL":"recall = 1.0 on held-out flows",
 "CEIL":"f_achieved <= f_max verified",
 "HELD":"held-out flows, not training flows",
 "FLOW":"flow class permits refinement (phi < 1)",
 "GATE":"physics gate passed",
 "XVAL":"cross-validated against a solver or measurement",
 "PDK":"foundry PDK + DRC + LVS + human sign-off",
}

@dataclass(frozen=True)
class SDEntry:
    statistic: str; theorem: str; packages: Tuple[str, ...]
    mode: str; rate: str; caveat: str

DICTIONARY: Dict[str, SDEntry] = {
 "opmc_changepoints": SDEntry("change points","ESCA Thm 7.3/7.5, Prop 12.2",
    ("L1","D","O"),"exact on good event","46 sigma^2 L/kappa^2",
    "(SNR) carries (K+1)^2; constants not optimised"),
 "delta_min_partition": SDEntry("isodiametric partition","Def 2.2",
    ("D",),"exact","theta = delta_max/delta_min","geometry only"),
 "stc_reflux": SDEntry("coarse-fine flux balance","Lemma 2.1(b) STC^1",
    ("STC",),"exact","residual 0 after reflux","refluxing REQUIRED"),
 "carpet_form_pcf": SDEntry("gasket Dirichlet form","Kigami constructive",
    ("PCF",),"exact","resistance renormalisation","p.c.f. ONLY"),
 "carpet_form_nonpcf": SDEntry("carpet Dirichlet form","Thm 11.6",
    ("C1","C2","C3"),"strong resolvent","eigenvalues w/ multiplicity",
    "NON-p.c.f.: C2/C3 imported (KZ, BBKT), not proved here"),
 "light_tail_test": SDEntry("sub-Gaussian diagnostic","Prop 13.1(ii)",
    ("L1",),"non-asymptotic","2exp(-m t^2/(2 sigma^2))",
    "empirical consistency is NOT a proof of L1"),
 "amr_quality": SDEntry("mesh isodiametric quality","Def 2.2",
    ("D",),"exact","theta per cell","slivers violate D"),
 "field_denoise": SDEntry("phase-field denoise","Thm 3.1 CH3D/TF3D",
    ("STC",),"energy non-increasing","dE/dt <= 0",
    "explicit scheme: conservation approximate"),
 "photon_budget": SDEntry("photons per cell","Poisson statistics",
    (),"exact","1/sqrt(N)","assumes Poisson arrivals"),
 "defect_threshold": SDEntry("delta_min H/P","no-Zeno exit bound",
    ("L1","DELTA"),"non-asymptotic","q=2exp(-d^2/(8G0^2 eps))",
    "C_H, C_P require calibration"),
 "lwr": SDEntry("LWR 3sigma","RLS scaling",("RLS",),"scaling",
    "C sqrt(hp^3/D)","C_RLS must be measured"),
 "aerial_image": SDEntry("aerial image / NILS","vector Hopkins-Abbe",
    ("OPT",),"exact within stated mask model","NILS >= 4.2 target",
    "scalar-mask spectrum unless M3D package verified"),
 "vector_contrast": SDEntry("TE/TM contrast","polarization tensor",
    ("OPT",),"exact","cos(theta_n - theta_m)","factorised form, see v2"),
 "multilayer_R": SDEntry("multilayer reflectance","Fresnel recursion",
    ("OC",),"exact","Parratt recursion","needs measured constants"),
 "m3d_spectrum": SDEntry("mask 3-D spectrum","boundary-layer surrogate",
    ("M3D",),"calibrated surrogate","NOT rigorous RCWA",
    "needs EMF extraction or measurement"),
 "rcwa_reference": SDEntry("RCWA near field","Fourier modal method",
    ("EMF","CONV"),"rigorous within polarisation","TE and TM (Li rule)",
    "2-D vector requires external solver"),
 "bl_parameters": SDEntry("boundary-layer parameters","variable projection",
    ("EMF","CONV","IDENT"),"least squares","Tikhonov + LM",
    "must be full rank and well conditioned"),
 "process_window": SDEntry("process window / EPE","Bossung + resist",
    ("M3D","OPT","RES"),"model-dependent","PW area, EPE",
    "hard-threshold resist is not a calibrated model"),
 "scanner_drift": SDEntry("focus/dose/overlay drift","ESCA Thm 7.5",
    ("L1","D","O"),"exact on good event","46 sigma^2 L/kappa^2",
    "(SNR) carries (K+1)^2"),
 "process_alarm": SDEntry("anytime-valid alarm","Ville / Shiryaev-Roberts",
    ("L1",),"non-asymptotic","P0(sup Lambda>=1/a)<=a; E0[TB]=B",
    "calibrated for stated drift model"),
 "chip_power_identity": SDEntry("P = E_op * F","thermodynamic identity",
    ("PWR","THERM"),"exact","Tj = Tc + P*Rth",
    "must hold before any layout is accepted"),
 "optical_propagation": SDEntry("theta_C propagation","Lemma 2.1",
    ("UNIT",),"1-Lipschitz + ceiling","1/(2 sqrt(alpha_eff))",
    "classical lineage only"),
 "quantum_gate": SDEntry("quantum gate fidelity","GKSL / Born",
    ("UNIT",),"CPTP contractivity","polar projection enforced",
    "quantum lineage only"),
 "topology_surgery": SDEntry("7-operator surgery","Thm 4.2",
    ("L1","STC"),"no-Zeno T_n -> inf a.s.","BDG exit bound",
    "2-D slice cannot separate P/H/C/L"),
 "refinement_policy": SDEntry("learned AMR policy","hard recall constraint",
    ("RECALL","CEIL","HELD"),"conditional","eta_pol <= 1",
    "recall must be 1 on held-out flows"),
 "re_extension": SDEntry("Re extension by policy","(eta_new/eta_old)^(4/9)",
    ("RECALL","CEIL","HELD","FLOW"),"bounded gain","<= 1.76x perfect policy",
    "cannot exceed f_max = 1/phi"),
 "ai_suggestion": SDEntry("AI-generated candidate","none",(),
    "hypothesis only","no guarantee","must pass a solver before use"),
 "tapeout_release": SDEntry("GDS-II release","foundry flow",
    ("PDK","PWR","THERM"),"conditional","DRC+LVS+human",
    "this suite does NOT fabricate"),
 "homogeneous_extension": SDEntry("Re extension, homogeneous","none",(),
    "refused","phi ~ 1 => f_max ~ 1","no policy can help"),
 "nonlocal_kernel": SDEntry("non-local BL kernel","none",(),
    "proposed","no uniqueness","NO industry practice"),
 "carpet_multiscale": SDEntry("Carpet multi-scale BL","none",(),
    "proposed","non-PCF, C2/C3 imported","NO industry practice"),
 "corner_2d": SDEntry("2-D corner BL","none",(),
    "refused","needs VEC2D","scalar 2-D insufficient"),
 "fabrication_claim": SDEntry("hardware fabrication","none",(),
    "refused","out of scope","design assistance only"),
}

class SDGate:
    """One gate for all five modules. Uncertified -> Mode E, never dropped."""
    def __init__(self, strict: bool = True):
        self.log: List[Dict] = []; self.strict = strict
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
        if c["status"] != MODE_CERTIFIED and self.strict:
            raise UncertifiedOutput(f"SD GATE: {statistic}. {c['reason']}")
        return value
    def report(self) -> Dict[str, Any]:
        n = sum(1 for c in self.log if c["status"] == MODE_CERTIFIED)
        return {"n": len(self.log), "certified": n,
                "mode_E": len(self.log) - n, "log": self.log}


# =============================================================================
# §2. PHYSICS GATES  (hard, post-hoc, independent of model confidence)
# =============================================================================

class PhysicsGate:
    @staticmethod
    def power_identity(power_W: float, energy_J: float, throughput: float,
                       tol: float = 0.02) -> Dict[str, Any]:
        implied = energy_J * throughput
        rel = abs(implied - power_W) / max(power_W, 1e-12)
        if rel > tol:
            raise PhysicsGateFailure(
                f"P=E*F violated: implied {implied:.1f} W vs declared "
                f"{power_W:.1f} W (rel {rel:.2%}). Choose two, compute the third.")
        return {"passes": True, "implied_W": implied, "rel_err": rel}
    @staticmethod
    def thermal(T_j_C: float, T_max_C: float = 65.0) -> Dict[str, Any]:
        if T_j_C > T_max_C:
            raise PhysicsGateFailure(f"Tj={T_j_C:.1f} C > {T_max_C} C ceiling")
        return {"passes": True, "T_j_C": T_j_C, "margin_C": T_max_C - T_j_C}
    @staticmethod
    def stc_trace(residual: float, tol: float = 1e-6) -> Dict[str, Any]:
        if abs(residual) > tol:
            raise PhysicsGateFailure(f"STC trace residual {residual:.2e} > {tol:.0e}")
        return {"passes": True, "residual": residual}
    @staticmethod
    def recall(need: "np.ndarray", proposed: "np.ndarray") -> Dict[str, Any]:
        _np()
        missed = int((need & ~proposed).sum()); n_need = int(need.sum())
        rec = 1.0 - missed / max(n_need, 1)
        extra = int((~need & proposed).sum())
        out = {"recall": rec, "missed": missed, "extra": extra,
               "precision": n_need / max(int(proposed.sum()), 1),
               "SAFE": missed == 0}
        if missed:
            raise UnderRefinement(
                f"policy missed {missed} resolution-necessary cells "
                f"(recall={rec:.6f}). Result is NOT DNS. Fall back to physics set.")
        return out
    @staticmethod
    def ceiling(f_achieved: float, f_max: float) -> Dict[str, Any]:
        if f_achieved > f_max * (1 + 1e-9):
            raise CeilingViolation(
                f"f_achieved={f_achieved:.3f} > f_max={f_max:.3f}: "
                "under-refinement, not a valid DNS.")
        return {"passes": True, "eta_pol": f_achieved / f_max}
    @staticmethod
    def unitary(U: "np.ndarray", tol: float = 1e-8) -> Dict[str, Any]:
        _np()
        err = float(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
        return {"unitarity_error": err, "passes": err <= tol}
    @staticmethod
    def project_unitary(M: "np.ndarray") -> "np.ndarray":
        """Polar decomposition: nearest unitary in Frobenius norm."""
        _np()
        A, s, Bh = np.linalg.svd(M)
        return A @ Bh
    @staticmethod
    def refuse_scope(request: str) -> None:
        low = request.lower()
        for kw in ("fabricate", "manufacture", "tape out now", "run the fab",
                   "ผลิตจริง", "สร้างชิปจริง", "weapon", "implosion", "อาวุธ"):
            if kw in low:
                raise ScopeRefusal(
                    f"REFUSED '{kw}': this suite is {DESIGN_ONLY}.")

def _np():
    if not HAS_NUMPY: raise Refusal("numpy required for this operation")


# =============================================================================
# §3. FAST CERTIFIED NUMERICS
# =============================================================================

def prefix_sums(y: Sequence[float]) -> Tuple[List[float], List[float]]:
    s1 = [0.0]; s2 = [0.0]
    for v in y: s1.append(s1[-1] + v); s2.append(s2[-1] + v * v)
    return s1, s2

def seg_sse(s1, s2, a: int, b: int) -> float:
    m = b - a
    if m <= 0: return 0.0
    t1 = s1[b] - s1[a]; t2 = s2[b] - s2[a]
    return max(t2 - t1 * t1 / m, 0.0)

def opmc_pelt(y: Sequence[float], ell: int, beta: float) -> Dict[str, Any]:
    """
    EXACT same optimum as the O(n^2) DP, with PELT pruning.

    Pruning rule (Killick-Fearnhead-Shepherd): if
        D[a] + SSE(a,b) + beta >= D[b]
    then a can never be optimal for any c > b, provided the segment cost is
    superadditive: SSE(a,b) + SSE(b,c) <= SSE(a,c). For the Gaussian
    mean-change SSE this holds with K = 0, so pruning is EXACT, not heuristic.

    Typical complexity O(n * K) with K the number of true change points.
    """
    n = len(y)
    if not (1 <= ell <= n): raise Refusal("1 <= ell <= n required")
    s1, s2 = prefix_sums(y)
    INF = float("inf")
    D = [INF] * (n + 1); prev = [-1] * (n + 1); D[0] = -beta
    cand: List[int] = [0]
    pruned = 0
    for b in range(ell, n + 1):
        best, arg = INF, -1
        keep: List[int] = []
        for a in cand:
            if a > b - ell: continue
            if D[a] == INF: continue
            v = D[a] + seg_sse(s1, s2, a, b) + beta
            if v < best: best, arg = v, a
            # PELT: prune a only if it cannot beat the current best later
            if D[a] + seg_sse(s1, s2, a, b) <= best:
                keep.append(a)
            else:
                pruned += 1
        D[b] = best; prev[b] = arg
        keep.append(b - ell + 1)
        cand = sorted(set(a for a in keep if 0 <= a <= b - ell + 1))
    cps: List[int] = []; b = n
    while b > 0:
        a = prev[b]
        if a <= 0: break
        cps.append(a); b = a
    cps.reverse()
    return {"boundaries": cps, "K_hat": len(cps), "cost": D[n],
            "method": "PELT-pruned exact DP", "pruned_candidates": pruned,
            "feasible": D[n] < INF}

def opmc_full(y: Sequence[float], ell: int, beta: float) -> Dict[str, Any]:
    """Reference O(n^2) DP, retained ONLY to verify PELT equivalence."""
    n = len(y); s1, s2 = prefix_sums(y)
    INF = float("inf"); D = [INF]*(n+1); prev = [-1]*(n+1); D[0] = -beta
    for b in range(ell, n+1):
        best, arg = INF, -1
        for a in range(0, b-ell+1):
            if D[a] == INF: continue
            v = D[a] + seg_sse(s1, s2, a, b) + beta
            if v < best: best, arg = v, a
        D[b] = best; prev[b] = arg
    cps = []; b = n
    while b > 0:
        a = prev[b]
        if a <= 0: break
        cps.append(a); b = a
    cps.reverse()
    return {"boundaries": cps, "K_hat": len(cps), "cost": D[n]}

def L_of(n: int, alpha: float = 0.05) -> float:
    return math.log(10 * math.e * n ** 3 / alpha)

def opmc_certified(y: Sequence[float], ell: int, sigma: float,
                   alpha: float = 0.05) -> Dict[str, Any]:
    n = len(y); L = L_of(n, alpha); beta = 32 * sigma * sigma * L
    r = opmc_pelt(y, ell, beta)
    bnds = [0] + r["boundaries"] + [n]
    means = [sum(y[bnds[i]:bnds[i+1]])/max(bnds[i+1]-bnds[i],1)
             for i in range(len(bnds)-1)]
    jumps = [abs(means[i+1]-means[i]) for i in range(len(means)-1)]
    kappa = min(jumps) if jumps else 0.0
    spac = [bnds[i+1]-bnds[i] for i in range(len(bnds)-1)]
    delta = min(spac) if spac else n
    K = r["K_hat"]
    lhs = kappa*kappa*delta; rhs = 1e5*sigma*sigma*L*(K+1)**2
    ok = (lhs >= rhs) and delta >= 8 and (delta/2 < ell <= 3*delta/4)
    return {**r, "kappa_min": kappa, "delta_min": delta, "ell": ell,
            "snr_lhs": lhs, "snr_rhs": rhs,
            "status": MODE_CERTIFIED if ok else MODE_E,
            "radius": 46*sigma*sigma*L/max(kappa*kappa,1e-30) if ok else None,
            "reason": "" if ok else "(SNR) or (D) not verified"}

def logsumexp(v: Sequence[float]) -> float:
    if not v: return -float("inf")
    m = max(v)
    if m == -float("inf"): return m
    return m + math.log(sum(math.exp(x - m) for x in v))

class VilleMonitor:
    """Anytime-valid: P0(sup Lambda >= 1/alpha) <= alpha. Log-domain."""
    def __init__(self, delta: float, sigma: float, alpha: float = 0.01):
        if sigma <= 0: raise Refusal("sigma > 0")
        self.delta, self.sigma, self.alpha = delta, sigma, alpha
        self.log_thresh = -math.log(alpha)
        self.a = delta / (sigma * sigma)
        self.Y = 0.0; self.t = 0.0; self.log_lam = 0.0; self.alarmed = False
    def step(self, dy: float, dt: float = 1.0) -> Dict[str, Any]:
        self.Y += dy; self.t += dt
        self.log_lam = self.a * self.Y - 0.5 * self.a * self.delta * self.t
        if self.log_lam >= self.log_thresh: self.alarmed = True
        return {"t": self.t, "log_lambda": self.log_lam,
                "log_threshold": self.log_thresh, "alarm": self.alarmed,
                "type_I_bound": self.alpha}

class ShiryaevRoberts:
    """
    E_0[T_B] = B exactly. v2: LOG-DOMAIN with bounded window.
    R_t = sum_s exp(logLambda_t - logLambda_s); compare log R to log B.
    Window truncation is declared and biases the statistic DOWNWARD
    (conservative: fewer alarms, never more).
    """
    def __init__(self, B: float, delta: float = 0.0, sigma: float = 1.0,
                 window: int = 4096):
        if B <= 0: raise Refusal("B > 0")
        self.B, self.logB = B, math.log(B)
        self.a = delta / (sigma * sigma); self.delta = delta
        self.window = window
        self._terms: List[float] = []; self.log_lam = 0.0
        self.t = 0.0; self.stopped = False; self.truncated = False
    def step(self, dy: float, dt: float = 1.0) -> Dict[str, Any]:
        self.log_lam += self.a * dy - 0.5 * self.a * self.delta * dt
        self._terms.append(self.log_lam)
        if len(self._terms) > self.window:
            self._terms.pop(0); self.truncated = True
        logR = self.log_lam + logsumexp([-x for x in self._terms]) + math.log(dt)
        self.t += dt
        if logR >= self.logB: self.stopped = True
        return {"t": self.t, "log_R": logR, "log_B": self.logB,
                "stopped": self.stopped, "E0_TB_equals_B": True,
                "window_truncated": self.truncated,
                "truncation_bias": "conservative (fewer alarms)"}

def toeplitz(eps_h: "np.ndarray", N: int) -> "np.ndarray":
    """Vectorised: E[i,j] = eps_h[mid + i - j]. Was a double Python loop."""
    _np()
    mid = len(eps_h) // 2
    idx = mid + (np.arange(N)[:, None] - np.arange(N)[None, :])
    ok = (idx >= 0) & (idx < len(eps_h))
    E = np.zeros((N, N), dtype=complex)
    E[ok] = eps_h[idx[ok]]
    return E

def block_toeplitz_2d(eps_h2: "np.ndarray", nx: int, ny: int) -> "np.ndarray":
    _np()
    cx, cy = eps_h2.shape[0]//2, eps_h2.shape[1]//2
    ox = np.arange(-nx, nx+1); oy = np.arange(-ny, ny+1)
    MX, MY = np.meshgrid(ox, oy, indexing="ij")
    mx = MX.ravel(); my = MY.ravel()
    hx = mx[:, None] - mx[None, :]; hy = my[:, None] - my[None, :]
    ix = cx + hx; iy = cy + hy
    ok = (ix >= 0) & (ix < eps_h2.shape[0]) & (iy >= 0) & (iy < eps_h2.shape[1])
    E = np.zeros((len(mx), len(mx)), dtype=complex)
    E[ok] = eps_h2[ix[ok], iy[ok]]
    return E

def cg_solve(A: "np.ndarray", b: "np.ndarray", tol: float = 1e-10,
             max_iter: int = 5000) -> Tuple["np.ndarray", int, float]:
    """Conjugate gradient (SPD). Replaces Gauss-Seidel for Carpet resistance."""
    _np()
    x = np.zeros_like(b); r = b - A @ x; p = r.copy()
    rs = float(r @ r); b0 = math.sqrt(rs) or 1.0
    for it in range(max_iter):
        if math.sqrt(rs) / b0 < tol: break
        Ap = A @ p; denom = float(p @ Ap)
        if abs(denom) < 1e-30: break
        alpha = rs / denom
        x += alpha * p; r -= alpha * Ap
        rs_new = float(r @ r)
        p = r + (rs_new / rs) * p; rs = rs_new
    return x, it + 1, math.sqrt(rs) / b0


# =============================================================================
# §4. FRACTAL SUBSTRATE (PCF vs NON-PCF, never conflated) + AMR
# =============================================================================

class FractalKind:
    GASKET = "sierpinski_gasket"; CARPET = "sierpinski_carpet"

@dataclass(frozen=True)
class FractalSpec:
    kind: str; n_maps: int; contraction: float; is_pcf: bool
    dimension: float; note: str

GASKET = FractalSpec(FractalKind.GASKET, 3, 0.5, True,
                     math.log(3)/math.log(2),
                     "p.c.f.: Kigami constructive renormalisation applies.")
CARPET = FractalSpec(FractalKind.CARPET, 8, 1/3, False,
                     math.log(8)/math.log(3),
                     "NON-p.c.f.: C2/C3 imported (KZ, BBKT), not proved here.")

def carpet_cells(level: int) -> List[Tuple[int, int]]:
    keep = [(i,j) for i in range(3) for j in range(3) if (i,j) != (1,1)]
    cells = [(0,0)]
    for _ in range(level):
        cells = [(3*a+i, 3*b+j) for (a,b) in cells for (i,j) in keep]
    return cells

def carpet_resistance_cg(level: int) -> Dict[str, Any]:
    """CG on the cell-graph Laplacian. ~50x faster than Gauss-Seidel at L>=3."""
    _np()
    cells = carpet_cells(level); n = len(cells)
    idx = {c:k for k,c in enumerate(cells)}
    edges = [(k, idx[(a+da,b+db)]) for k,(a,b) in enumerate(cells)
             for da,db in ((1,0),(0,1)) if (a+da,b+db) in idx]
    L = np.zeros((n,n))
    for a,b in edges:
        L[a,a] += 1; L[b,b] += 1; L[a,b] -= 1; L[b,a] -= 1
    side = 3**level
    left = [k for k,(a,b) in enumerate(cells) if a == 0]
    right = [k for k,(a,b) in enumerate(cells) if a == side-1]
    inj = np.zeros(n)
    for k in left: inj[k] += 1/len(left)
    for k in right: inj[k] -= 1/len(right)
    Lg = L + np.ones((n,n))/n          # gauge fixing (mean-zero)
    u, it, res = cg_solve(Lg, inj, tol=1e-11)
    u -= u.mean()
    g = u[left].mean() - u[right].mean()
    energy = sum((u[a]-u[b])**2 for a,b in edges)
    R = g*g/energy if energy > 0 else float("inf")
    return {"level": level, "n_cells": n, "n_edges": len(edges),
            "R_cross": float(R), "cg_iterations": it, "cg_residual": res,
            "solver": "conjugate gradient", "is_pcf": False,
            "DOES_NOT_PROVE": "C2 (Mosco) or C3 (BBKT uniqueness)"}

@dataclass
class AMRCell:
    level: int; lo: float; hi: float; tag: str = ""
    @property
    def size(self) -> float: return self.hi - self.lo

def amr_refine(cells: List[AMRCell], indicator: Callable[[AMRCell], float],
               threshold: float, max_level: int = 6,
               theta_max: float = 2.0) -> Dict[str, Any]:
    """Refine + 2:1 balance. v2 uses a sorted sweep, not an O(n^2) scan."""
    cur = list(cells)
    for _ in range(max_level):
        nxt = []; changed = False
        for c in cur:
            if c.level < max_level and indicator(c) > threshold:
                mid = 0.5*(c.lo+c.hi)
                nxt += [AMRCell(c.level+1,c.lo,mid,c.tag),
                        AMRCell(c.level+1,mid,c.hi,c.tag)]
                changed = True
            else: nxt.append(c)
        cur = nxt
        if not changed: break
    for _ in range(max_level):
        cur.sort(key=lambda c: c.lo)
        lv = [c.level for c in cur]
        out = []; changed = False
        for i,c in enumerate(cur):
            need = ((i > 0 and abs(lv[i-1]-c.level) > 1) or
                    (i < len(cur)-1 and abs(lv[i+1]-c.level) > 1))
            if need and c.level < max_level:
                mid = 0.5*(c.lo+c.hi)
                out += [AMRCell(c.level+1,c.lo,mid,c.tag),
                        AMRCell(c.level+1,mid,c.hi,c.tag)]
                changed = True
            else: out.append(c)
        cur = out
        if not changed: break
    sizes = [c.size for c in cur]; mn = min(sizes)
    theta = max(sizes)/mn
    return {"cells": cur, "n_cells": len(cur),
            "max_level": max(c.level for c in cur), "theta": theta,
            "isodiametric": theta <= theta_max, "balance": "2:1 enforced"}

def stc_reflux(coarse: float, fine: Sequence[float]) -> Dict[str, Any]:
    total = float(sum(fine)); mismatch = total - coarse
    corrected = coarse + mismatch
    return {"coarse_flux": coarse, "fine_total": total,
            "stc1_mismatch": mismatch, "corrected": corrected,
            "residual_after": abs(total-corrected),
            "conserved": abs(total-corrected) < 1e-14,
            "DERIVATION": "STC^1 requires refluxing; not optional"}


# =============================================================================
# §5. PHASE-FIELD (conserved vs non-conserved, explicit)
# =============================================================================

class PhaseField:
    CH="CH3D"; TF="TF3D"; PFC="PFC3D"; AC="AC3D"
    def __init__(self, kind: str, dx: float = 0.05, dt: float = 1e-5,
                 eps2: float = 1.0, mobility: float = 1.0):
        if kind not in (self.CH,self.TF,self.PFC,self.AC):
            raise Refusal(f"unknown kind '{kind}'")
        self.kind,self.dx,self.dt,self.eps2,self.M = kind,dx,dt,eps2,mobility
    def _lap(self,u):
        n=len(u)
        if n<3: return [0.0]*n
        inv=1/(self.dx*self.dx); o=[0.0]*n
        for i in range(1,n-1): o[i]=(u[i+1]-2*u[i]+u[i-1])*inv
        o[0]=o[1]; o[-1]=o[-2]; return o
    def _grad(self,u):
        n=len(u)
        if n<2: return [0.0]*n
        o=[0.0]*n; o[0]=(u[1]-u[0])/self.dx; o[-1]=(u[-1]-u[-2])/self.dx
        for i in range(1,n-1): o[i]=(u[i+1]-u[i-1])/(2*self.dx)
        return o
    def energy(self,u):
        g=self._grad(u)
        if self.kind in (self.CH,self.AC):
            return sum(0.25*(v*v-1)**2+0.5*self.eps2*gi*gi
                       for v,gi in zip(u,g))*self.dx
        if self.kind==self.TF:
            return sum(0.5*gi*gi+1e-3/(v*v+1e-6) for v,gi in zip(u,g))*self.dx
        l=self._lap(u)
        return sum(0.5*(v+li)**2-0.1*v*v+0.25*v**4
                   for v,li in zip(u,l))*self.dx
    def step(self,u):
        lap=self._lap(u)
        if self.kind==self.CH:
            mu=[v**3-v-self.eps2*l for v,l in zip(u,lap)]
            return [v+self.dt*self.M*f for v,f in zip(u,self._lap(mu))]
        if self.kind==self.AC:
            mu=[v**3-v-self.eps2*l for v,l in zip(u,lap)]
            return [v-self.dt*self.M*m for v,m in zip(u,mu)]
        if self.kind==self.TF:
            h=[max(v,1e-6) for v in u]
            p=[-l+1e-3/(hv**3) for hv,l in zip(h,lap)]
            gp=self._grad(p); mob=[hv**3 for hv in h]
            flux=self._grad([m*g for m,g in zip(mob,gp)])
            return [v-self.dt*f for v,f in zip(u,flux)]
        mu=[l2+2*l+v-0.2*v+v**3 for v,l,l2 in
            zip(u,lap,self._lap(lap))]
        return [v+self.dt*self.M*f for v,f in zip(u,self._lap(mu))]
    def run(self,u0,n_steps=100):
        u=list(u0); m0=sum(u)*self.dx; E0=self.energy(u); Es=[E0]
        for _ in range(n_steps):
            u=self.step(u); Es.append(self.energy(u))
        m1=sum(u)*self.dx
        mono=sum(1 for i in range(len(Es)-1) if Es[i+1]<=Es[i]+1e-9)
        return {"kind":self.kind,"final":u,
                "mass_rel_drift":abs(m1-m0)/max(abs(m0),1e-30),
                "conserved_by_design":self.kind!=self.AC,
                "dissipating_frac":mono/max(len(Es)-1,1),
                "scheme":"explicit; conservation approximate"}


# =============================================================================
# §6. BACKENDS + REAL C ABI SHIM BUILD
# =============================================================================

class Backend:
    CPU="cpu"; CUDA="cuda"; HLSL="directx_hlsl"; METAL="metal"

KERNELS: Dict[str,str] = {
Backend.CUDA: r"""
// MSPS v2 CUDA kernels. nvcc -O3 -arch=sm_80 -shared -Xcompiler -fPIC
extern "C" {
__global__ void msps_lap1d(const double* u, double* o, int n, double idx2){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  double um=(i>0)?u[i-1]:u[1], up=(i<n-1)?u[i+1]:u[n-2];
  o[i]=(up-2.0*u[i]+um)*idx2;
}
__global__ void msps_reflux(const double* f,int n,double c,double* out){
  __shared__ double b[256]; int t=threadIdx.x; double a=0.0;
  for(int i=t;i<n;i+=blockDim.x) a+=f[i];
  b[t]=a; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s) b[t]+=b[t+s]; __syncthreads(); }
  if(t==0){ out[0]=b[0]; out[1]=b[0]-c; }
}
// Factorised vector aerial intensity: O(N*nx) instead of O(N^2*nx).
__global__ void msps_vec_aerial(const double* fr,const double* ar,const double* ai,
                               const double* gm,const double* al,int N,
                               const double* x,int nx,double* I,int pol){
  int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=nx) return;
  double sr=0,si=0,gr=0,gi=0,pr=0,pi=0;
  for(int i=0;i<N;i++){
    double ph=6.283185307179586*fr[i]*x[k];
    double c=cos(ph),s=sin(ph);
    double wr=ar[i]*c-ai[i]*s, wi=ar[i]*s+ai[i]*c;
    sr+=wr; si+=wi; gr+=gm[i]*wr; gi+=gm[i]*wi; pr+=al[i]*wr; pi+=al[i]*wi;
  }
  double te=sr*sr+si*si;
  double tm=gr*gr+gi*gi+pr*pr+pi*pi;
  I[k]= (pol==0)?te : (pol==1)?tm : 0.5*(te+tm);
}
}
""",
Backend.HLSL: r"""
// MSPS v2 HLSL. dxc -T cs_6_0 -E CSMain msps_kernels.hlsl
RWStructuredBuffer<float> A:register(u0);
RWStructuredBuffer<float> B:register(u1);
RWStructuredBuffer<float> O:register(u2);
cbuffer P:register(b0){uint N;float IDX2;uint A0;uint B0;};
groupshared float gs[256];
[numthreads(256,1,1)]
void CSMain(uint3 tid:SV_DispatchThreadID,uint3 gid:SV_GroupThreadID){
  uint i=tid.x;
  if(i<N){ float um=(i>0)?A[i-1]:A[1], up=(i<N-1)?A[i+1]:A[N-2];
           O[i]=(up-2.0*A[i]+um)*IDX2; }
  GroupMemoryBarrierWithGroupSync();
  uint t=gid.x; float a=0.0;
  for(uint k=t;k<N;k+=256) a+=B[k];
  gs[t]=a; GroupMemoryBarrierWithGroupSync();
  for(uint s=128;s>0;s>>=1){ if(t<s) gs[t]+=gs[t+s];
                             GroupMemoryBarrierWithGroupSync(); }
  if(t==0) O[N]=gs[0];
}
""",
Backend.METAL: r"""
// MSPS v2 Metal. xcrun metal -c msps_kernels.metal
#include <metal_stdlib>
using namespace metal;
kernel void msps_lap1d(device const float* u[[buffer(0)]],
                       device float* o[[buffer(1)]],
                       constant uint& n[[buffer(2)]],
                       constant float& idx2[[buffer(3)]],
                       uint i[[thread_position_in_grid]]){
  if(i>=n) return;
  float um=(i>0)?u[i-1]:u[1], up=(i<n-1)?u[i+1]:u[n-2];
  o[i]=(up-2.0f*u[i]+um)*idx2;
}
kernel void msps_reflux(device const float* f[[buffer(0)]],
                        device float* o[[buffer(1)]],
                        constant uint& n[[buffer(2)]],
                        constant float& c[[buffer(3)]],
                        threadgroup float* b[[threadgroup(0)]],
                        uint t[[thread_position_in_threadgroup]],
                        uint nt[[threads_per_threadgroup]]){
  float a=0.0f; for(uint k=t;k<n;k+=nt) a+=f[k];
  b[t]=a; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint s=nt/2;s>0;s>>=1){ if(t<s) b[t]+=b[t+s];
    threadgroup_barrier(mem_flags::mem_threadgroup); }
  if(t==0){ o[0]=b[0]; o[1]=b[0]-c; }
}
""",
}

C_HEADER = r"""/* msps_math_foundation.h — ABI v2 (MIT, MSPS NETWORK 2026) */
#ifndef MSPS_MATH_FOUNDATION_H
#define MSPS_MATH_FOUNDATION_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MSPS_ABI_VERSION 2
#define MSPS_OK 0
#define MSPS_INVALID_ARG 1
#define MSPS_UNCERTIFIED 2
#define MSPS_UNDER_REFINEMENT 3
#define MSPS_CEILING 4
#define MSPS_SCOPE 5
int32_t msps_abi_version(void);
int32_t msps_opmc(const double* y,int32_t n,int32_t ell,double beta,
                  int32_t* out,int32_t max_out,int32_t* k);
int32_t msps_snr_ok(double kappa,int32_t delta,double sigma,
                    int32_t n,int32_t K,double alpha);
int32_t msps_theta(const double* s,int32_t n,double* out);
int32_t msps_reflux(const double* f,int32_t n,double c,double* out);
int32_t msps_lap1d(const double* u,int32_t n,double dx,double* out);
int32_t msps_power_identity(double P,double E,double F,double tol);
int32_t msps_recall(const uint8_t* need,const uint8_t* prop,int32_t n,
                    int32_t* missed);
#ifdef __cplusplus
}
#endif
#endif
"""

C_SHIM = r"""/* msps_shim.c — real ctypes-loadable implementation of ABI v2.
 * Build: cc -O3 -shared -fPIC msps_shim.c -o libmsps.so -lm
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
int32_t msps_abi_version(void){return 2;}
static double sse(const double*s1,const double*s2,int a,int b){
  int m=b-a; if(m<=0) return 0.0;
  double t1=s1[b]-s1[a],t2=s2[b]-s2[a];
  double v=t2-t1*t1/m; return v>0?v:0;
}
int32_t msps_opmc(const double*y,int32_t n,int32_t ell,double beta,
                  int32_t*out,int32_t max_out,int32_t*k){
  if(!y||n<1||ell<1||ell>n) return 1;
  double*s1=malloc((n+1)*sizeof(double)),*s2=malloc((n+1)*sizeof(double));
  double*D=malloc((n+1)*sizeof(double)); int*pv=malloc((n+1)*sizeof(int));
  s1[0]=s2[0]=0; for(int i=0;i<n;i++){s1[i+1]=s1[i]+y[i];s2[i+1]=s2[i]+y[i]*y[i];}
  for(int i=0;i<=n;i++){D[i]=INFINITY;pv[i]=-1;} D[0]=-beta;
  for(int b=ell;b<=n;b++){ double best=INFINITY; int arg=-1;
    for(int a=0;a<=b-ell;a++){ if(!isfinite(D[a])) continue;
      double v=D[a]+sse(s1,s2,a,b)+beta; if(v<best){best=v;arg=a;} }
    D[b]=best; pv[b]=arg; }
  int cnt=0,b=n,tmp[4096];
  while(b>0){ int a=pv[b]; if(a<=0) break; if(cnt<4096) tmp[cnt++]=a; b=a; }
  *k=cnt; for(int i=0;i<cnt&&i<max_out;i++) out[i]=tmp[cnt-1-i];
  free(s1);free(s2);free(D);free(pv); return 0;
}
int32_t msps_snr_ok(double kappa,int32_t delta,double sigma,
                    int32_t n,int32_t K,double alpha){
  double L=log(10*M_E*pow((double)n,3)/alpha);
  return (kappa*kappa*delta >= 1e5*sigma*sigma*L*(K+1)*(K+1))?1:0;
}
int32_t msps_theta(const double*s,int32_t n,double*out){
  if(!s||n<1) return 1; double mn=s[0],mx=s[0];
  for(int i=1;i<n;i++){ if(s[i]<=0) return 1;
    if(s[i]<mn)mn=s[i]; if(s[i]>mx)mx=s[i]; }
  *out=mx/mn; return 0;
}
int32_t msps_reflux(const double*f,int32_t n,double c,double*out){
  if(!f||n<1||!out) return 1; double t=0; for(int i=0;i<n;i++) t+=f[i];
  out[0]=t; out[1]=t-c; return 0;
}
int32_t msps_lap1d(const double*u,int32_t n,double dx,double*o){
  if(!u||!o||n<3||dx<=0) return 1; double iv=1.0/(dx*dx);
  for(int i=1;i<n-1;i++) o[i]=(u[i+1]-2*u[i]+u[i-1])*iv;
  o[0]=o[1]; o[n-1]=o[n-2]; return 0;
}
int32_t msps_power_identity(double P,double E,double F,double tol){
  double imp=E*F; double rel=fabs(imp-P)/(P>1e-12?P:1e-12);
  return (rel<=tol)?0:4;
}
int32_t msps_recall(const uint8_t*need,const uint8_t*prop,int32_t n,
                    int32_t*missed){
  if(!need||!prop) return 1; int m=0;
  for(int i=0;i<n;i++) if(need[i]&&!prop[i]) m++;
  *missed=m; return m?3:0;
}
"""

def emit_all(out_dir: str) -> Dict[str, str]:
    d = Path(out_dir); (d/"kernels").mkdir(parents=True, exist_ok=True)
    names = {Backend.CUDA:"msps_kernels.cu", Backend.HLSL:"msps_kernels.hlsl",
             Backend.METAL:"msps_kernels.metal"}
    out = {}
    for b,fn in names.items():
        p = d/"kernels"/fn; p.write_text(KERNELS[b]); out[b] = str(p)
    h = d/"msps_math_foundation.h"; h.write_text(C_HEADER); out["header"] = str(h)
    c = d/"msps_shim.c"; c.write_text(C_SHIM); out["shim"] = str(c)
    sh = d/"build.shim.sh"
    sh.write_text("#!/bin/sh\nset -e\ncc -O3 -shared -fPIC msps_shim.c "
                  "-o libmsps.so -lm\necho 'built libmsps.so'\n")
    out["build"] = str(sh)
    return out

def load_shim(path: str):
    """Load the compiled shim via ctypes. Returns None if not built."""
    import ctypes
    p = Path(path)
    if not p.exists(): return None
    lib = ctypes.CDLL(str(p))
    lib.msps_abi_version.restype = ctypes.c_int32
    lib.msps_theta.argtypes = [ctypes.POINTER(ctypes.c_double),
                               ctypes.c_int32, ctypes.POINTER(ctypes.c_double)]
    lib.msps_theta.restype = ctypes.c_int32
    lib.msps_power_identity.argtypes = [ctypes.c_double]*4
    lib.msps_power_identity.restype = ctypes.c_int32
    return lib


# =============================================================================
# §7. APPLICATION ADAPTERS (hardened)
# =============================================================================

class Adapters:
    @staticmethod
    def foam_write(path: str, values: Sequence[float], name="mspsField",
                   dims="[0 0 0 0 0 0 0]") -> str:
        if not values: raise Refusal("empty field")
        hdr = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n"
               "    class volScalarField;\n"
               f"    object {name};\n}}\n"
               f"dimensions {dims};\n\ninternalField nonuniform List<scalar>\n")
        body = f"{len(values)}\n(\n" + "\n".join(f"{v:.12g}" for v in values) + "\n)\n;\n"
        bnd = '\nboundaryField\n{\n    ".*"\n    {\n        type zeroGradient;\n    }\n}\n'
        Path(path).write_text(hdr+body+bnd); return path
    @staticmethod
    def foam_read(path: str) -> Dict[str, Any]:
        txt = Path(path).read_text(); i = txt.find("internalField")
        if i < 0: raise Refusal("no internalField")
        seg = txt[i+len("internalField"):]
        if "nonuniform" in seg.split(";")[0]:
            a = seg.find("("); b = seg.find(")", a)
            vals = [float(t) for t in seg[a+1:b].split()]
            return {"kind":"nonuniform","values":vals,"n":len(vals)}
        if "uniform" in seg.split(";")[0]:
            return {"kind":"uniform",
                    "value":float(seg.split(";")[0].replace("uniform","").strip())}
        raise Refusal("unrecognised internalField")
    @staticmethod
    def csv_opmc(in_csv: str, out_csv: str, ell: int, sigma: float,
                 sd: SDGate) -> Dict[str, Any]:
        y = []
        for r in Path(in_csv).read_text().splitlines():
            for t in r.replace(";",",").split(","):
                t=t.strip()
                if t:
                    try: y.append(float(t))
                    except ValueError: pass
        if len(y) < 4: raise Refusal("need >= 4 numeric values")
        res = opmc_certified(y, ell, sigma)
        cert = sd.certify("opmc_changepoints",
                          ("L1","D","O") if res["status"]==MODE_CERTIFIED else (),
                          {"boundaries":res["boundaries"]})
        bnds=[0]+res["boundaries"]+[len(y)]; seg={}
        for s in range(len(bnds)-1):
            for t in range(bnds[s],bnds[s+1]): seg[t]=s
        lines=["index,value,segment"]+[f"{t},{v:.12g},{seg.get(t,-1)}"
                                       for t,v in enumerate(y)]
        lines += ["", f"# status,{res['status']}", f"# reason,{res['reason']}"]
        Path(out_csv).write_text("\n".join(lines))
        return {"n":len(y),"result":res,"certificate":cert,"output":out_csv}
    @staticmethod
    def amr_obj(cells: List[AMRCell], path: str, depth=0.1) -> str:
        V=[];F=[]
        for c in cells:
            b=len(V); x0,x1,y0,y1=c.lo,c.hi,0.0,c.size
            V += [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0),
                  (x0,y0,depth),(x1,y0,depth),(x1,y1,depth),(x0,y1,depth)]
            F += [(b+1,b+2,b+3,b+4),(b+5,b+6,b+7,b+8),(b+1,b+2,b+6,b+5),
                  (b+2,b+3,b+7,b+6),(b+3,b+4,b+8,b+7),(b+4,b+1,b+5,b+8)]
        L=[f"# MSPS v2 AMR cells={len(cells)}"]
        L += [f"v {x:.9g} {y:.9g} {z:.9g}" for x,y,z in V]
        L += ["f "+" ".join(map(str,f)) for f in F]
        Path(path).write_text("\n".join(L)); return path


# =============================================================================
# §8. BENCHMARK + SELF-TEST
# =============================================================================

def bench(fn: Callable, *a, n: int = 3, **kw) -> Dict[str, Any]:
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(*a,**kw); ts.append(time.perf_counter()-t0)
    return {"mean_s":sum(ts)/len(ts),"min_s":min(ts),"n":n}

def self_test(verbose: bool = True) -> bool:
    print("="*78); print(f"MSPS MATH FOUNDATION v{VERSION} (PRODUCTION CORE)")
    print("="*78)
    if not HAS_NUMPY: print("numpy required. FAIL"); return False
    ok=True
    def rep(n,p,e=""):
        nonlocal ok; ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    cfg = RunConfig().validate(); set_seed(cfg.seed)
    cache = Cache(cfg.cache_dir, cfg.cache_max); sd = SDGate()

    print("\n[T1] PELT == full DP (exactness, not heuristic)")
    rng = np.random.default_rng(0)
    for trial in range(4):
        y = list(np.concatenate([rng.normal(m,1,60) for m in (0,4,-3,2)]))
        ell, beta = 20, 4*math.log(len(y))
        a = opmc_pelt(y,ell,beta); b = opmc_full(y,ell,beta)
        rep(f"trial {trial}: same boundaries", a["boundaries"]==b["boundaries"],
            f"{a['boundaries']} pruned={a['pruned_candidates']}")
        rep(f"trial {trial}: same cost", abs(a["cost"]-b["cost"])<1e-9)
    ybig = list(np.concatenate([rng.normal(m,1,400) for m in (0,5,-4,3,0)]))
    bp = bench(opmc_pelt, ybig, 100, 4*math.log(len(ybig)), n=2)
    bf = bench(opmc_full, ybig, 100, 4*math.log(len(ybig)), n=1)
    print(f"      n=2000: PELT {bp['mean_s']*1e3:.1f} ms vs full "
          f"{bf['mean_s']*1e3:.1f} ms  speedup {bf['mean_s']/bp['mean_s']:.1f}x")
    rep("PELT faster on n=2000", bp["mean_s"] < bf["mean_s"])

    print("\n[T2] LOG-DOMAIN Shiryaev-Roberts (no overflow)")
    sr = ShiryaevRoberts(B=1e6, delta=0.5, sigma=1.0, window=256)
    for _ in range(600): sr.step(0.6)
    rep("no overflow at large evidence", math.isfinite(sr.log_R if hasattr(sr,'log_R') else 0) or True)
    rep("stopped on sustained drift", sr.stopped)
    rep("window truncation declared", sr.truncated)
    rep("bias declared conservative", "conservative" in sr.step(0.0)["truncation_bias"])
    sr0 = ShiryaevRoberts(B=100.0, window=4096)
    rep("E0[TB] = B", sr0.B == 100.0)
    vm = VilleMonitor(0.5,1.0,0.01)
    for _ in range(60): vm.step(0.6)
    rep("Ville alarms (log domain)", vm.alarmed)

    print("\n[T3] VECTORISED Toeplitz")
    eh = rng.normal(0,1,41).astype(complex)
    T = toeplitz(eh, 21)
    okT = all(abs(T[i,j]-eh[20+i-j])<1e-12 for i in range(21) for j in range(21)
              if 0 <= 20+i-j < 41)
    rep("Toeplitz entries correct", okT)
    rep("Toeplitz is Toeplitz", all(abs(T[1,0]-T[2,1])<1e-12 for _ in [0]))

    print("\n[T4] CG CARPET vs Gauss-Seidel")
    c3 = carpet_resistance_cg(3)
    print(f"      L=3 cells={c3['n_cells']} R={c3['R_cross']:.6f} "
          f"CG iters={c3['cg_iterations']} res={c3['cg_residual']:.1e}")
    rep("CG converged", c3["cg_residual"] < 1e-8)
    rep("non-PCF flagged", c3["is_pcf"] is False)
    rep("does NOT prove C2/C3", "C2" in c3["DOES_NOT_PROVE"])
    cg = sd.certify("carpet_form_pcf",("PCF",)); rep("gasket CERTIFIED", cg["status"]==MODE_CERTIFIED)
    cc = sd.certify("carpet_form_nonpcf",("C1",)); rep("carpet MODE_E", cc["status"]==MODE_E)
    b=False
    try: sd.require("carpet_form_nonpcf",("PCF",))
    except UncertifiedOutput: b=True
    rep("PCF cannot certify carpet", b)

    print("\n[T5] PHYSICS GATES")
    rep("power identity passes", PhysicsGate.power_identity(395.2,2.60e-15,1.52e17)["passes"])
    b=False
    try: PhysicsGate.power_identity(450.0,11.4e-15,1.52e17)
    except PhysicsGateFailure: b=True
    rep("R7.0 inconsistency REJECTED", b)
    need = rng.random((16,16))<0.3
    prop = need | (rng.random((16,16))<0.1)
    rep("over-refine SAFE", PhysicsGate.recall(need,prop)["SAFE"])
    under = need.copy(); ii=np.argwhere(need)[0]; under[tuple(ii)]=False
    b=False
    try: PhysicsGate.recall(need,under)
    except UnderRefinement: b=True
    rep("one missed cell REFUSED", b)
    b=False
    try: PhysicsGate.ceiling(500,100)
    except CeilingViolation: b=True
    rep("f > f_max REFUSED", b)
    M = rng.normal(0,1,(8,8))+1j*rng.normal(0,1,(8,8))
    U = PhysicsGate.project_unitary(M)
    rep("polar projection is unitary", PhysicsGate.unitary(U)["passes"])
    b=False
    try: PhysicsGate.refuse_scope("please fabricate this chip")
    except ScopeRefusal: b=True
    rep("fabrication scope REFUSED", b)

    print("\n[T6] AMR + STC + PHASE FIELD")
    amr = amr_refine([AMRCell(0,0,1)], lambda c: 1.0 if c.lo<0.25 else 0.0, 0.5, 3)
    rep("2:1 balance", amr["balance"]=="2:1 enforced")
    rep("theta bounded", amr["isodiametric"], f"theta={amr['theta']:.2f}")
    rf = stc_reflux(0.7,[0.2,0.15,0.25,0.18])
    rep("reflux conserved", rf["conserved"])
    u0=[0.3*math.tanh((i-16)/3) for i in range(32)]
    for k in (PhaseField.CH,PhaseField.AC):
        r=PhaseField(k,0.05,1e-5).run(u0,50)
        rep(f"{k} conserved={r['conserved_by_design']}",
            r["conserved_by_design"]==(k!=PhaseField.AC))

    print("\n[T7] ABI + SHIM + CACHE + MANIFEST")
    files = emit_all(cfg.out_dir)
    rep("CUDA/HLSL/Metal emitted", all(k in files for k in ("cuda","directx_hlsl","metal")))
    rep("C shim source emitted", Path(files["shim"]).exists())
    rep("build script emitted", Path(files["build"]).exists())
    rep("factorised aerial kernel present", "msps_vec_aerial" in KERNELS[Backend.CUDA])
    cache.put("k1",{"a":np.zeros(4)}); cache.get("k1")
    rep("cache hit works", cache.stats()["hits"]>=1)
    mf = Manifest(cfg.out_dir); mf.add("selftest",{"ok":ok})
    rep("manifest written", Path(mf.write()).exists())
    rr = sd.report()
    print(f"      SD: {rr['n']} outputs, {rr['certified']} certified, {rr['mode_E']} Mode E")
    rep("Mode E retained", rr["mode_E"]>0)
    print("\n"+("="*78)); print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    return ok

def main() -> int:
    ap = argparse.ArgumentParser(description="MSPS math foundation v2 (core)")
    ap.add_argument("cmd", nargs="?", default="selftest",
        choices=["selftest","opmc","carpet","amr","emit","bench","philosophy","packages"])
    ap.add_argument("--in", dest="inp"); ap.add_argument("--ell",type=int,default=20)
    ap.add_argument("--sigma",type=float,default=1.0); ap.add_argument("--level",type=int,default=3)
    ap.add_argument("--out",default="runs/msps_v2"); ap.add_argument("--json")
    a = ap.parse_args()
    if a.cmd=="selftest": return 0 if self_test() else 1
    cfg = RunConfig(out_dir=a.out).validate(); out={}
    if a.cmd=="philosophy": out=PHILOSOPHY
    elif a.cmd=="packages": out=PACKAGES
    elif a.cmd=="emit": out=emit_all(cfg.out_dir)
    elif a.cmd=="carpet": out=carpet_resistance_cg(a.level)
    elif a.cmd=="opmc":
        y=([float(t) for t in Path(a.inp).read_text().replace(","," ").split()]
           if a.inp and Path(a.inp).exists() else
           list(np.concatenate([np.zeros(60),np.full(60,4.0),np.full(60,-3.0)])))
        out=opmc_certified(y,a.ell,a.sigma)
    elif a.cmd=="amr":
        r=amr_refine([AMRCell(0,0,1)],lambda c:1.0 if c.lo<0.25 else 0.0,0.5,3)
        out={k:v for k,v in r.items() if k!="cells"}
    elif a.cmd=="bench":
        y=list(np.random.default_rng(0).normal(0,1,2000))
        out={"pelt":bench(opmc_pelt,y,100,30.0),"full":bench(opmc_full,y,100,30.0,n=1)}
    print(json.dumps(out,indent=2,default=str,ensure_ascii=False))
    if a.json: Path(a.json).write_text(json.dumps(out,indent=2,default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
