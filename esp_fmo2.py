#!/usr/bin/env python3
"""
=============================================================================
ESP CHARGE FITTING (MK / CHELPG / RESP) + FMO2 WITH ESP EMBEDDING
=============================================================================
License: MIT | 2026

LITERATURE
 [E1] Besler, Merz, Kollman. "Atomic charges derived from semiempirical
      methods." J. Comput. Chem. 11, 431 (1990).            [MK ESP]
 [E2] Singh & Kollman. "An approach to computing electrostatic charges for
      molecules." J. Comput. Chem. 5, 129 (1984).
 [E3] Breneman & Wiberg. "Determining atom-centered monopoles from molecular
      electrostatic potentials." J. Comput. Chem. 11, 361 (1990). [CHELPG]
 [E4] Bayly, Cieplak, Cornell, Kollman. "A well-behaved electrostatic potential
      based method using charge restraints for deriving atomic charges: the
      RESP model." J. Phys. Chem. 97, 10269 (1993).          [RESP]
 [E5] Bondi. "van der Waals volumes and radii." J. Phys. Chem. 68, 441 (1964).
 [F1] Fedorov & Kitaura. "Extending the power of quantum chemistry to large
      systems with the fragment molecular orbital method."
      J. Phys. Chem. A 111, 6904 (2007).
 [F2] Nakano et al. "Fragment molecular orbital method: use of approximate
      electrostatic potentials as external potential."
      Chem. Phys. Lett. 351, 475 (2002).
 [F3] Fedorov, Kitaura. "The importance of three-body terms in the fragment
      molecular orbital method." J. Chem. Phys. 120, 6832 (2004).

=============================================================================
WHY ESP INSTEAD OF MULLIKEN
=============================================================================
Mulliken charges:
  • partition the density matrix by basis-function overlap
  • STRONGLY basis-set dependent; can be qualitatively wrong with diffuse sets
  • no relation to the observable electrostatic potential

ESP-fitted charges:
  • reproduce the QM electrostatic potential outside the vdW surface
  • far weaker basis-set dependence
  • the physically meaningful quantity for electrostatic embedding
  → This is what FMO needs [F2]. RESP [E4] adds a hyperbolic restraint to
    tame the ill-conditioning for buried atoms.

WHAT THIS STILL DOES NOT FIX
  ❌ FMO2 fragmentation error. Cutting a covalent or dative bond is an
     approximation regardless of how good the embedding charges are.
     At metal–ligand dative bonds the charge transfer is large and FMO2
     error can be tens of kcal/mol. ALWAYS compare against the
     unfragmented cluster; this file reports that difference.
  ❌ Three-body (FMO3) terms [F3] are not included.
  ❌ Hybrid-orbital projection / bond-detachment atoms of production FMO
     (GAMESS) are not implemented; link-H capping is used instead.
  ❌ Two-stage RESP with methyl/methylene equivalence constraints is NOT
     implemented (single-stage only). Equivalent-H symmetry is not enforced.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pyscf import gto, scf, dft, mp
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("esp_fmo2")

BOHR_ANG = 0.529177210903
HARTREE_KCAL = 627.5094740631
HARTREE_EV = 27.211386245988

# [E5] Bondi vdW radii (Å). Metals are rough estimates — flagged.
VDW = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,"P":1.80,"S":1.80,
       "Cl":1.75,"Br":1.85,"I":1.98,
       "ZN":2.01,"FE":2.05,"MG":1.73,"MN":2.05,"CU":1.96,"CA":2.31,
       "NI":1.97,"CO":2.00,"CD":2.18}
Z_NUM = {"H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"Cl":17,"Br":35,"I":53,
         "ZN":30,"FE":26,"MG":12,"MN":25,"CU":29,"CA":20,"NI":28,"CO":27,"CD":48}


# =============================================================================
# 1. FITTING GRID  (MK shells / CHELPG cube)
# =============================================================================

def fibonacci_sphere(n: int) -> np.ndarray:
    """Near-uniform points on unit sphere (no external dependency)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = math.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta)*np.sin(phi),
                     np.sin(theta)*np.sin(phi), np.cos(phi)], axis=-1)


def mk_grid(els: List[str], xyz_ang: np.ndarray,
            shells=(1.4, 1.6, 1.8, 2.0), n_per_shell: int = 120,
            max_points: int = 6000, seed: int = 0) -> np.ndarray:
    """
    Merz–Kollman / Connolly-style shells [E1,E2].
    Points at s×vdW around each atom; points inside ANY atom's innermost
    shell are discarded. Returns Å.
    """
    radii = np.array([VDW.get(e, 1.7) for e in els])
    inner = radii * shells[0]
    unit = fibonacci_sphere(n_per_shell)
    pts = []
    for s in shells:
        for i in range(len(els)):
            pts.append(xyz_ang[i] + unit * (radii[i] * s))
    P = np.concatenate(pts, axis=0)
    d = np.linalg.norm(P[:, None, :] - xyz_ang[None, :, :], axis=-1)
    keep = (d >= inner[None, :] - 1e-8).all(axis=1)
    P = P[keep]
    if len(P) > max_points:
        rng = np.random.default_rng(seed)
        P = P[rng.choice(len(P), max_points, replace=False)]
        log.info(f"[MK] subsampled to {max_points} points (cost control)")
    return P


def chelpg_grid(els: List[str], xyz_ang: np.ndarray, spacing: float = 0.4,
                outer: float = 3.0, max_points: int = 6000,
                seed: int = 0) -> np.ndarray:
    """
    CHELPG cubic grid [E3]: exclude points inside vdW surface,
    keep points within `outer` Å of some atom. Returns Å.
    """
    radii = np.array([VDW.get(e, 1.7) for e in els])
    lo = xyz_ang.min(0) - outer; hi = xyz_ang.max(0) + outer
    axes = [np.arange(lo[k], hi[k] + spacing, spacing) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d = np.linalg.norm(G[:, None, :] - xyz_ang[None, :, :], axis=-1)
    keep = (d >= radii[None, :]).all(axis=1) & (d.min(axis=1) <= outer)
    P = G[keep]
    if len(P) > max_points:
        rng = np.random.default_rng(seed)
        P = P[rng.choice(len(P), max_points, replace=False)]
        log.info(f"[CHELPG] subsampled to {max_points} points")
    return P


# =============================================================================
# 2. QM ELECTROSTATIC POTENTIAL ON THE GRID
# =============================================================================

def qm_esp(mol, dm: np.ndarray, pts_ang: np.ndarray) -> np.ndarray:
    """
    V_QM(r) = Σ_A Z_A/|r-R_A| − Σ_pq P_pq ⟨p| 1/|r-R| |q⟩   (atomic units)

    Uses PySCF int1e_grids. Falls back to a numerical density-grid ESP
    if that integral engine is unavailable.
    """
    pts_bohr = np.asarray(pts_ang, dtype=np.float64) / BOHR_ANG
    # nuclear part
    R = mol.atom_coords()                       # Bohr
    Zn = mol.atom_charges()
    dn = np.linalg.norm(pts_bohr[:, None, :] - R[None, :, :], axis=-1)
    v_nuc = (Zn[None, :] / np.maximum(dn, 1e-12)).sum(axis=1)
    # electronic part
    try:
        ints = mol.intor("int1e_grids", grids=pts_bohr)   # (npts, nao, nao)
        v_elec = -np.einsum("kij,ji->k", ints, dm)
        return v_nuc + v_elec
    except Exception as exc:
        log.warning(f"[ESP] int1e_grids unavailable ({exc}); "
                    f"falling back to numerical density integration")
    from pyscf.dft import numint
    ni = numint.NumInt()
    grids = mol.grids if hasattr(mol, "grids") else None
    from pyscf.dft.gen_grid import Grids
    g = Grids(mol); g.level = 3; g.build()
    ao = ni.eval_ao(mol, g.coords)
    rho = ni.eval_rho(mol, ao, dm)
    w = g.weights * rho                          # electron count per point
    dq = np.linalg.norm(pts_bohr[:, None, :] - g.coords[None, :, :], axis=-1)
    v_elec = -(w[None, :] / np.maximum(dq, 1e-8)).sum(axis=1)
    return v_nuc + v_elec


# =============================================================================
# 3. CONSTRAINED FIT  (ESP / RESP)
# =============================================================================

@dataclass
class ESPConfig:
    scheme: str = "resp"          # esp | resp
    grid: str = "mk"              # mk | chelpg
    resp_a: float = 0.0005        # restraint strength [E4]
    resp_b: float = 0.1           # hyperbolic tightness [E4]
    resp_iters: int = 25
    resp_tol: float = 1e-6
    ridge: float = 1e-8
    n_per_shell: int = 120
    max_points: int = 6000
    chelpg_spacing: float = 0.4
    chelpg_outer: float = 3.0


def fit_esp_charges(els: List[str], xyz_ang: np.ndarray,
                    v_qm: np.ndarray, pts_ang: np.ndarray,
                    total_charge: float, cfg: ESPConfig) -> Dict:
    """
    Constrained least squares:
        min ||A q − V||²  s.t.  Σq = Q
    A_kA = 1/|r_k − R_A|  (Bohr)

    RESP [E4] adds a hyperbolic restraint a(√(q²+b²) − b), applied by
    iteratively reweighting the diagonal by a/√(q²+b²).
    """
    R = np.asarray(xyz_ang) / BOHR_ANG
    P = np.asarray(pts_ang) / BOHR_ANG
    d = np.linalg.norm(P[:, None, :] - R[None, :, :], axis=-1)
    A = 1.0 / np.maximum(d, 1e-10)
    n = len(els)
    AtA = A.T @ A
    Atb = A.T @ v_qm
    ones = np.ones(n)

    def solve(diag_extra):
        M = np.zeros((n+1, n+1))
        M[:n, :n] = AtA + np.diag(diag_extra) + cfg.ridge*np.eye(n)
        M[:n, n] = ones; M[n, :n] = ones
        rhs = np.concatenate([Atb, [total_charge]])
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return sol[:n]

    q = solve(np.zeros(n))
    hist = []
    if cfg.scheme == "resp":
        for it in range(cfg.resp_iters):
            extra = cfg.resp_a / np.sqrt(q*q + cfg.resp_b**2)
            q_new = solve(extra)
            dq = float(np.abs(q_new - q).max())
            hist.append(dq); q = q_new
            if dq < cfg.resp_tol:
                break
    resid = A @ q - v_qm
    rrms = float(np.sqrt(np.mean(resid**2)) /
                 max(np.sqrt(np.mean(v_qm**2)), 1e-12))
    return {
        "charges": q.tolist(),
        "sum_charges": float(q.sum()),
        "target_charge": float(total_charge),
        "charge_error": float(abs(q.sum() - total_charge)),
        "rrms": rrms,                     # relative RMS ESP fit error
        "n_points": int(len(P)),
        "scheme": cfg.scheme, "grid": cfg.grid,
        "resp_history": hist,
        "resp_converged": bool(hist and hist[-1] < cfg.resp_tol)
                          if cfg.scheme == "resp" else None,
        "NOTE": ("single-stage RESP; equivalent-H symmetry NOT enforced"
                 if cfg.scheme == "resp" else "unrestrained ESP"),
    }


def mulliken_charges(mf) -> np.ndarray:
    return np.asarray(mf.mulliken_pop(verbose=0)[1])


# =============================================================================
# 4. SINGLE POINT + CHARGES FOR ONE FRAGMENT
# =============================================================================

def basis_for(el: str) -> str:
    return "lanl2dz" if el in ("ZN","FE","CU","MN","NI","CO","CD") else "6-31g*"

def ecp_for(el: str) -> Optional[str]:
    return "lanl2dz" if el in ("ZN","FE","CU","MN","NI","CO","CD") else None


def fragment_calc(els: List[str], xyz_ang: np.ndarray, charge: int, mult: int,
                  mm_coords: Optional[np.ndarray] = None,
                  mm_charges: Optional[np.ndarray] = None,
                  method: str = "b3lyp", esp_cfg: Optional[ESPConfig] = None,
                  verbose: int = 0) -> Optional[Dict]:
    """
    DFT/HF single point, optionally in an external point-charge field.
    Returns energy, ESP charges, Mulliken charges, and the fragment's
    interaction energy with the external field (for FMO internal energies).
    """
    if not HAS_PYSCF:
        raise ImportError("pip install pyscf")
    esp_cfg = esp_cfg or ESPConfig()
    try:
        atom = [[els[i], tuple(map(float, xyz_ang[i]))] for i in range(len(els))]
        basis = {e: basis_for(e) for e in set(els)}
        ecp = {e: ecp_for(e) for e in set(els) if ecp_for(e)}
        mol = gto.M(atom=atom, basis=basis, ecp=ecp or None,
                    charge=charge, spin=mult-1, unit="Angstrom", verbose=verbose)
        mf = dft.UKS(mol) if mult > 1 else dft.RKS(mol)
        if method.lower() in ("hf", "rhf", "uhf"):
            mf = scf.UHF(mol) if mult > 1 else scf.RHF(mol)
        else:
            mf.xc = method
        mf.conv_tol = 1e-9; mf.max_cycle = 150
        if mm_coords is not None and mm_charges is not None and len(mm_charges):
            from pyscf import qmmm
            mf = qmmm.mm_charge(mf, np.asarray(mm_coords)/BOHR_ANG,
                                np.asarray(mm_charges))
        e = float(mf.kernel())
        if not mf.converged:
            log.warning("[frag] SCF did not converge — discarded")
            return None
        dm = mf.make_rdm1()
        if isinstance(dm, (list, tuple)) or (hasattr(dm, "ndim") and dm.ndim == 3):
            dm = np.sum(dm, axis=0)
        pts = (mk_grid(els, xyz_ang, n_per_shell=esp_cfg.n_per_shell,
                       max_points=esp_cfg.max_points)
               if esp_cfg.grid == "mk" else
               chelpg_grid(els, xyz_ang, esp_cfg.chelpg_spacing,
                           esp_cfg.chelpg_outer, esp_cfg.max_points))
        if len(pts) < len(els) + 2:
            log.warning("[frag] too few ESP points — falling back to Mulliken")
            q_esp = mulliken_charges(mf)
            fit = {"rrms": float("nan"), "n_points": len(pts),
                   "scheme": "mulliken_fallback"}
        else:
            v = qm_esp(mol, dm, pts)
            fit = fit_esp_charges(els, xyz_ang, v, pts, float(charge), esp_cfg)
            q_esp = np.array(fit["charges"])
        q_mul = mulliken_charges(mf)
        # interaction with external point charges (point-charge approximation)
        e_int = 0.0
        if mm_coords is not None and mm_charges is not None and len(mm_charges):
            dext = np.linalg.norm(
                (xyz_ang[:, None, :] - np.asarray(mm_coords)[None, :, :])/BOHR_ANG,
                axis=-1)
            phi = (np.asarray(mm_charges)[None, :] /
                   np.maximum(dext, 1e-10)).sum(axis=1)
            e_int = float(np.dot(q_esp, phi))
        return {"energy": e, "esp": q_esp, "mulliken": q_mul,
                "esp_fit": fit, "e_int_ext": e_int,
                "n_basis": int(mol.nao_nr())}
    except Exception as exc:
        log.warning(f"[frag] failed: {type(exc).__name__}: {exc}")
        return None


# =============================================================================
# 5. FMO2 WITH ESP EMBEDDING (SCC) + INTERNAL-ENERGY CORRECTION
# =============================================================================

def fmo2_esp(fragments: List[Tuple[List[str], np.ndarray, int, int]],
             method: str = "b3lyp", scc_iters: int = 12,
             scc_tol: float = 1e-4, esp_cfg: Optional[ESPConfig] = None,
             max_pair_atoms: int = 200,
             charge_scheme: str = "esp") -> Dict:
    """
    FMO2 [F1,F2]:
        E = Σ_I E'_I + Σ_{I>J} (E'_IJ − E'_I − E'_J) + Σ_{I>J} E^es_IJ

    E' = E − (interaction with external point charges)   [internal energy]
    E^es_IJ = classical charge–charge interaction between fragments I and J

    Embedding charges: ESP (default) or Mulliken (for comparison).
    SCC loop iterates monomer charges to self-consistency.

    ⚠️ Point-charge internal-energy correction is an approximation; production
       FMO uses the full embedding-potential matrix trace.
    ⚠️ No FMO3 [F3], no hybrid-orbital projection, no dimer-es approximation.
    """
    n = len(fragments)
    if n == 0:
        return {"error": "no fragments"}
    esp_cfg = esp_cfg or ESPConfig()
    key = "esp" if charge_scheme == "esp" else "mulliken"
    Q = [np.zeros(len(f[0])) for f in fragments]
    E = [0.0]*n; Eint = [0.0]*n
    hist = []; converged = False

    for it in range(scc_iters):
        newQ, newE, newI = [], [], []
        ok = True
        for i, (els, xyz, q, m) in enumerate(fragments):
            cc, qq = [], []
            for j in range(n):
                if j == i: continue
                cc.append(fragments[j][1]); qq.append(Q[j])
            mc = np.concatenate(cc) if cc else None
            mq = np.concatenate(qq) if qq else None
            r = fragment_calc(els, xyz, q, m, mc, mq, method, esp_cfg)
            if r is None:
                ok = False; break
            newQ.append(r[key]); newE.append(r["energy"])
            newI.append(r["e_int_ext"])
        if not ok:
            return {"error": f"monomer failed at SCC iter {it}",
                    "scc_converged": False}
        dq = max(float(np.abs(newQ[i]-Q[i]).max()) for i in range(n))
        hist.append(dq); Q, E, Eint = newQ, newE, newI
        log.info(f"[FMO2-{key}] SCC {it+1}: Δq_max={dq:.3e}")
        if dq < scc_tol:
            converged = True; break

    Eprime = [E[i] - Eint[i] for i in range(n)]
    # classical inter-fragment electrostatics
    Ees = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(
                (fragments[i][1][:, None, :] -
                 fragments[j][1][None, :, :])/BOHR_ANG, axis=-1)
            Ees += float((Q[i][:, None]*Q[j][None, :] /
                          np.maximum(d, 1e-10)).sum())

    dimer_corr = 0.0; npair = 0; skipped = 0
    for i in range(n):
        for j in range(i+1, n):
            ei, xi, qi, mi = fragments[i]
            ej, xj, qj, mj = fragments[j]
            els = ei+ej; xyz = np.concatenate([xi, xj])
            if len(els) > max_pair_atoms:
                skipped += 1; continue
            q = qi+qj
            ne = sum(Z_NUM.get(e, 0) for e in els) - q
            m = 1 if ne % 2 == 0 else 2
            cc, qq = [], []
            for k in range(n):
                if k in (i, j): continue
                cc.append(fragments[k][1]); qq.append(Q[k])
            mc = np.concatenate(cc) if cc else None
            mq = np.concatenate(qq) if qq else None
            r = fragment_calc(els, xyz, q, m, mc, mq, method, esp_cfg)
            if r is None:
                skipped += 1; continue
            Ep_ij = r["energy"] - r["e_int_ext"]
            dimer_corr += Ep_ij - Eprime[i] - Eprime[j]
            npair += 1

    E1 = float(sum(Eprime)) + Ees
    E2 = E1 + dimer_corr
    return {
        "E_FMO1_hartree": E1,
        "E_FMO2_hartree": E2,
        "dimer_correction_kcal": float(dimer_corr*HARTREE_KCAL),
        "classical_es_hartree": Ees,
        "charge_scheme": key,
        "n_fragments": n, "n_dimers": npair, "n_dimers_skipped": skipped,
        "scc_history": hist, "scc_converged": converged,
        "charges": [q.tolist() for q in Q],
        "esp_rrms": None,
        "WARNING": ("FMO2 fragmentation error remains. Compare to the "
                    "unfragmented cluster. No FMO3, no HOP projection."),
    }


# =============================================================================
# 6. SELF-TEST
# =============================================================================

def selftest(args):
    print("="*76); print("  ESP / RESP / FMO2 — self-test"); print("="*76)
    ok = True
    els = ["O","H","H"]
    xyz = np.array([[0.0,0.0,0.117],[0.0,0.757,-0.469],[0.0,-0.757,-0.469]])

    print("\n[T1] grid generation excludes interior points")
    P = mk_grid(els, xyz, n_per_shell=60, max_points=2000)
    radii = np.array([VDW[e] for e in els])
    dmin = np.linalg.norm(P[:, None]-xyz[None], axis=-1)
    inside = (dmin < radii*1.4 - 1e-6).any(axis=1).sum()
    print(f"   MK points={len(P)}  interior violations={inside}")
    t = len(P) > 20 and inside == 0; ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T2] CHELPG grid")
    Pc = chelpg_grid(els, xyz, spacing=0.5, outer=2.5, max_points=2000)
    dc = np.linalg.norm(Pc[:, None]-xyz[None], axis=-1)
    t = len(Pc) > 20 and (dc < radii).any(axis=1).sum() == 0
    ok &= t; print(f"   points={len(Pc)}  {'PASS' if t else 'FAIL'}")

    print("\n[T3] constrained fit conserves total charge exactly")
    rng = np.random.default_rng(0)
    q_true = np.array([-0.7, 0.35, 0.35])
    Rb = xyz/BOHR_ANG; Pb = P/BOHR_ANG
    A = 1.0/np.linalg.norm(Pb[:, None]-Rb[None], axis=-1)
    v = A @ q_true
    cfg = ESPConfig(scheme="esp")
    fit = fit_esp_charges(els, xyz, v, P, 0.0, cfg)
    err = fit["charge_error"]; rrms = fit["rrms"]
    t = err < 1e-8 and rrms < 1e-6
    print(f"   |Σq−Q|={err:.2e}  RRMS={rrms:.2e}")
    print(f"   recovered q={np.round(fit['charges'],5)}")
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T4] RESP restraint shrinks buried-atom charges")
    v_noisy = v + rng.normal(0, 1e-3, len(v))
    f_esp = fit_esp_charges(els, xyz, v_noisy, P, 0.0, ESPConfig(scheme="esp"))
    f_resp = fit_esp_charges(els, xyz, v_noisy, P, 0.0,
                             ESPConfig(scheme="resp", resp_a=0.001))
    n_esp = np.linalg.norm(f_esp["charges"])
    n_resp = np.linalg.norm(f_resp["charges"])
    t = n_resp <= n_esp + 1e-6 and f_resp["charge_error"] < 1e-8
    print(f"   ||q||_ESP={n_esp:.5f}  ||q||_RESP={n_resp:.5f}")
    print(f"   RESP iters={len(f_resp['resp_history'])} "
          f"converged={f_resp['resp_converged']}")
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    if HAS_PYSCF:
        print("\n[T5] real QM ESP: Mulliken vs ESP for water")
        r = fragment_calc(els, xyz, 0, 1, method="b3lyp",
                          esp_cfg=ESPConfig(scheme="resp"))
        if r:
            qm, qe = r["mulliken"], r["esp"]
            print(f"   Mulliken: {np.round(qm,4)}")
            print(f"   RESP ESP: {np.round(qe,4)}")
            print(f"   ESP fit RRMS = {r['esp_fit']['rrms']:.4f} "
                  f"on {r['esp_fit']['n_points']} points")
            t = abs(qe.sum()) < 1e-6 and r["esp_fit"]["rrms"] < 0.2
            ok &= t; print(f"   {'PASS' if t else 'CHECK'}")
            print("   NOTE: literature SPC/TIP3P-like O charge ≈ −0.8;")
            print("         ESP-derived values are method/basis dependent.")
        else:
            print("   FAIL"); ok = False

        print("\n[T6] FMO2 ESP vs Mulliken on 2 water fragments")
        w2 = xyz + np.array([3.0, 0.0, 0.0])
        frags = [(els, xyz, 0, 1), (els, w2, 0, 1)]
        for scheme in ("mulliken", "esp"):
            res = fmo2_esp(frags, method="b3lyp", scc_iters=6,
                           charge_scheme=scheme,
                           esp_cfg=ESPConfig(n_per_shell=40, max_points=800))
            if "error" in res:
                print(f"   {scheme}: {res['error']}"); ok = False; continue
            print(f"   {scheme:9s} E_FMO2={res['E_FMO2_hartree']:.6f} Ha  "
                  f"ΔE_dim={res['dimer_correction_kcal']:+.3f} kcal/mol  "
                  f"SCC={res['scc_converged']}")
        # reference: unfragmented dimer
        d_all = fragment_calc(els+els, np.concatenate([xyz, w2]), 0, 1,
                              method="b3lyp",
                              esp_cfg=ESPConfig(n_per_shell=30, max_points=600))
        if d_all:
            print(f"   cluster (unfragmented) E = {d_all['energy']:.6f} Ha")
            print("   → the FMO2−cluster gap IS the fragmentation error.")
    else:
        print("\n[T5,T6] SKIP — pip install pyscf")

    print("\n" + "="*76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*76)
    print("""
  FIXED: Mulliken basis-set dependence → ESP/RESP potential-fitted charges.
  NOT FIXED: FMO2 fragmentation error, missing FMO3, missing HOP projection,
             two-stage RESP equivalence constraints.
  ALWAYS report the FMO2 − cluster difference alongside the FMO2 energy.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest"])
    a = p.parse_args()
    raise SystemExit(0 if selftest(a) else 1)


if __name__ == "__main__":
    main()
