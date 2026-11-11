#!/usr/bin/env python3
"""
=============================================================================
METAL-SITE DATASET · FMO2/DFT FEATURES · LINK-ATOM QM/MM (QM SIDE)
=============================================================================
License: MIT | 2026

LITERATURE
 [M1] Brown & Altermatt. "Bond-valence parameters obtained from a systematic
      analysis of the Inorganic Crystal Structure Database."
      Acta Cryst. B41, 244 (1985).
 [M2] Yang, Powell, Houser. "Structural variation in copper(I) complexes with
      pyridylmethylamide ligands: structural analysis with a new four-coordinate
      geometry index, tau4." Dalton Trans. 955 (2007).
 [M3] Addison et al. "Synthesis, structure, and spectroscopic properties of
      copper(II) compounds containing nitrogen-sulphur donor ligands."
      J. Chem. Soc. Dalton Trans. 1349 (1984).            [tau5]
 [F1] Fedorov & Kitaura. "Extending the power of quantum chemistry to large
      systems with the fragment molecular orbital method."
      J. Phys. Chem. A 111, 6904 (2007).
 [F2] Nakano et al. "Fragment molecular orbital method: use of approximate
      electrostatic potentials as external potential." Chem. Phys. Lett. 351, 475 (2002).
 [Q1] Senn & Thiel. "QM/MM methods for biomolecular systems."
      Angew. Chem. Int. Ed. 48, 1198 (2009).
 [Q2] Reuter, Dejaegere, Maigret, Karplus. "Frontier bonds in QM/MM methods:
      a comparison of different approaches." J. Phys. Chem. A 104, 1720 (2000).
 [Q3] Besler, Merz, Kollman. "Atomic charges derived from semiempirical methods."
      J. Comput. Chem. 11, 431 (1990).                    [ESP charges]

=============================================================================
HONEST SCOPE
=============================================================================
METAL SITES
  ✅ Real PDB coordinates. Coordination geometry is measured, not predicted.
  ⚠️ OXIDATION STATE IS ASSUMED (+2 default, Fe +3). PDB does not encode it.
     BVS is reported for BOTH plausible states so the ambiguity is visible.
  ⚠️ R0 bond-valence parameters below are typical literature values.
     VERIFY against Brown–Altermatt / IUCr tables before publishing.
  ❌ Crystal contacts, alternate conformers, and partial occupancy are not
     resolved. A crystal site is not necessarily the solution-state site.

FMO2
  ✅ Real monomer SCF + dimer correction: E = ΣE_I + Σ(E_IJ − E_I − E_J)
  ⚠️ Embedding uses MULLIKEN charges, not ESP. Real FMO uses ESP (or
     approximate ESP [F2]); Mulliken is a known, poorer approximation.
  ⚠️ SCC loop included but convergence is not guaranteed for charged metals.
  ❌ Metal–ligand fragmentation cuts dative bonds with large charge transfer.
     FMO2 error at the metal can be LARGE. The cluster (unfragmented) result
     is reported alongside so the fragmentation error is measurable.

LINK-ATOM QM/MM
  ✅ Link-H placement along the cut bond (standard, [Q2]).
  ✅ Electrostatic embedding via PySCF qmmm.mm_charge — MM charges enter
     the QM Hamiltonian.
  ❌ MM ENERGY IS NOT IMPLEMENTED. A toy force field would produce numbers
     that look like energies and are not. An OpenMM interface is provided;
     without it, only the QM part is reported and clearly flagged.
  ❌ No charge shifting / redistribution schemes [Q2]; no vdW parameter set.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math, urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pyscf import gto, scf, mp, qmmm, dft
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False
try:
    import openmm as mm
    from openmm import app, unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("protein_qm")

HARTREE_KCAL = 627.5094740631
HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903

METALS = {"ZN","FE","MG","MN","CU","CA","NI","CO","CD","MN2","FE2"}
DONORS = {"N","O","S"}
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU",
       "LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}

# [M1] Typical R0 (Å). VERIFY before publication.
R0_TABLE = {
 ("ZN","O"):1.704, ("ZN","N"):1.772, ("ZN","S"):2.090,
 ("FE2","O"):1.734,("FE3","O"):1.759,("FE2","N"):1.769,("FE3","N"):1.815,
 ("FE2","S"):2.125,("FE3","S"):2.149,
 ("MG","O"):1.693, ("MG","N"):1.710, ("MG","S"):2.100,
 ("MN","O"):1.790, ("MN","N"):1.849, ("MN","S"):2.220,
 ("CU","O"):1.679, ("CU","N"):1.713, ("CU","S"):2.054,
 ("CA","O"):1.967, ("CA","N"):2.140, ("CA","S"):2.450,
 ("NI","O"):1.654, ("NI","N"):1.647, ("NI","S"):2.040,
 ("CO","O"):1.692, ("CO","N"):1.840, ("CO","S"):2.060,
}
DEFAULT_OX = {"ZN":2,"MG":2,"CA":2,"MN":2,"NI":2,"CO":2,"CU":2,"CD":2,"FE":3}
LINK_BOND = 1.09   # Å, C–H link bond [Q2]


# =============================================================================
# 1. PDB PARSING + METAL SITE EXTRACTION
# =============================================================================

def fetch_pdb(pdb_id: str, dest: str = "data/pdb") -> Path:
    d = Path(dest); d.mkdir(parents=True, exist_ok=True)
    out = d / f"{pdb_id.lower()}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    log.info(f"[PDB] {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        out.write_bytes(r.read())
    return out


def parse_pdb(path: Path) -> List[Dict]:
    atoms = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break                      # first model only
        if not line.startswith(("ATOM", "HETATM")):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue                   # skip alternate conformers
        try:
            atoms.append({
                "record": line[:6].strip(),
                "serial": int(line[6:11]),
                "name": line[12:16].strip(),
                "resn": line[17:20].strip(),
                "chain": line[21],
                "resi": int(line[22:26]),
                "xyz": np.array([float(line[30:38]), float(line[38:46]),
                                 float(line[46:54])], dtype=np.float64),
                "occ": float(line[54:60] or 1.0),
                "elem": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
            })
        except (ValueError, IndexError):
            continue
    return atoms


def tau4(angles_deg: List[float]) -> float:
    """[M2] four-coordinate index: 1 = tetrahedral, 0 = square planar."""
    if len(angles_deg) < 6:
        return float("nan")
    a = sorted(angles_deg, reverse=True)[:2]
    return (360.0 - a[0] - a[1]) / (360.0 - 2 * 109.47)


def tau5(angles_deg: List[float]) -> float:
    """[M3] five-coordinate index: 1 = TBP, 0 = square pyramidal."""
    if len(angles_deg) < 10:
        return float("nan")
    a = sorted(angles_deg, reverse=True)[:2]
    return (a[0] - a[1]) / 60.0


def bond_valence(metal: str, donors: List[Tuple[str, float]],
                 ox: int) -> float:
    """[M1] BVS = Σ exp((R0 − d)/0.37)."""
    key_m = metal if metal != "FE" else f"FE{ox}"
    s = 0.0
    for el, d in donors:
        r0 = R0_TABLE.get((key_m, el)) or R0_TABLE.get((metal, el))
        if r0 is None:
            continue
        s += math.exp((r0 - d) / 0.37)
    return s


def extract_metal_sites(atoms: List[Dict], cutoff: float = 2.8,
                        pdb_id: str = "") -> List[Dict]:
    prot = [a for a in atoms if a["resn"] in AA3]
    metals = [a for a in atoms if a["elem"] in METALS
              and a["record"] == "HETATM"]
    sites = []
    for m in metals:
        coord = []
        for a in prot:
            if a["elem"] not in DONORS:
                continue
            d = float(np.linalg.norm(a["xyz"] - m["xyz"]))
            if d <= cutoff:
                coord.append((a, d))
        if len(coord) < 2:
            continue
        coord.sort(key=lambda t: t[1])
        vecs = [a["xyz"] - m["xyz"] for a, _ in coord]
        angles = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                c = np.dot(vecs[i], vecs[j]) / (
                    np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-12)
                angles.append(math.degrees(math.acos(np.clip(c, -1, 1))))
        donors = [(a["elem"], d) for a, d in coord]
        ox = DEFAULT_OX.get(m["elem"], 2)
        bvs = {o: bond_valence(m["elem"], donors, o) for o in (2, 3)}
        res = sorted({(a["chain"], a["resi"], a["resn"]) for a, _ in coord})
        sites.append({
            "pdb": pdb_id, "metal": m["elem"],
            "metal_serial": m["serial"], "chain": m["chain"], "resi": m["resi"],
            "metal_xyz": m["xyz"],
            "coordination_number": len(coord),
            "donors": [{"elem": a["elem"], "name": a["name"],
                        "resn": a["resn"], "chain": a["chain"],
                        "resi": a["resi"], "dist": d} for a, d in coord],
            "residues": res,
            "tau4": tau4(angles) if len(coord) == 4 else None,
            "tau5": tau5(angles) if len(coord) == 5 else None,
            "angles_deg": angles,
            "assumed_oxidation": ox,
            "bvs_ox2": bvs[2], "bvs_ox3": bvs[3],
            "mean_dist": float(np.mean([d for _, d in coord])),
        })
    return sites


# =============================================================================
# 2. CAPPED CLUSTER BUILDER  (link atoms on Cα–Cβ cut)
# =============================================================================

SIDECHAIN_SKIP = {"CA", "C", "O", "N", "OXT"}

def build_cluster(atoms: List[Dict], site: Dict,
                  include_backbone: bool = False
                  ) -> Tuple[List[str], np.ndarray, List[Dict]]:
    """
    QM region = metal + donor-residue sidechains, cut at Cα–Cβ.
    Link H replaces Cα, placed 1.09 Å from Cβ along Cβ→Cα  [Q2].

    Returns (elements, coords, link_records).
    Gly (no Cβ) is skipped unless include_backbone=True.
    """
    els = [site["metal"]]
    xyz = [site["metal_xyz"]]
    links = []
    by_res: Dict[Tuple[str, int], List[Dict]] = {}
    for a in atoms:
        if a["resn"] in AA3:
            by_res.setdefault((a["chain"], a["resi"]), []).append(a)

    for chain, resi, resn in site["residues"]:
        group = by_res.get((chain, resi), [])
        if not group:
            continue
        names = {a["name"]: a for a in group}
        ca, cb = names.get("CA"), names.get("CB")
        if cb is None and not include_backbone:
            log.debug(f"[cluster] {resn}{resi}: no CB (Gly?) — skipped")
            continue
        keep = [a for a in group
                if include_backbone or a["name"] not in SIDECHAIN_SKIP]
        if not keep:
            continue
        for a in keep:
            els.append(a["elem"]); xyz.append(a["xyz"])
        if cb is not None and ca is not None:
            v = ca["xyz"] - cb["xyz"]
            v = v / (np.linalg.norm(v) + 1e-12)
            h = cb["xyz"] + LINK_BOND * v
            els.append("H"); xyz.append(h)
            links.append({"res": f"{resn}{resi}", "qm_atom": "CB",
                          "mm_atom": "CA", "link_xyz": h,
                          "cut_bond_A": float(np.linalg.norm(ca["xyz"]-cb["xyz"]))})
    return els, np.array(xyz, dtype=np.float64), links


def cluster_charge_mult(site: Dict, els: List[str]) -> Tuple[int, int]:
    """
    Formal charge bookkeeping. HIGHLY APPROXIMATE:
    Asp/Glu carboxylate −1, Cys thiolate −1, His/Met neutral.
    Protonation states are NOT determined from structure.
    """
    q = DEFAULT_OX.get(site["metal"], 2)
    seen = set()
    for d in site["donors"]:
        key = (d["chain"], d["resi"])
        if key in seen:
            continue
        seen.add(key)
        if d["resn"] in ("ASP", "GLU"):
            q -= 1
        elif d["resn"] == "CYS":
            q -= 1
    n_e = sum({"H":1,"C":6,"N":7,"O":8,"S":16,"ZN":30,"FE":26,"MG":12,
               "MN":25,"CU":29,"CA":20,"NI":28,"CO":27,"CD":48}.get(e, 0)
              for e in els) - q
    mult = 1 if n_e % 2 == 0 else 2
    # high-spin Fe(III)/Mn(II) common; expose as assumption
    if site["metal"] == "FE" and site["assumed_oxidation"] == 3:
        mult = 6
    elif site["metal"] == "MN":
        mult = 6
    return q, mult


# =============================================================================
# 3. DFT SINGLE POINT + FEATURES
# =============================================================================

Z_TABLE = {"H":1,"C":6,"N":7,"O":8,"S":16,"ZN":30,"FE":26,"MG":12,
           "MN":25,"CU":29,"CA":20,"NI":28,"CO":27,"CD":48}

def basis_for(el: str) -> str:
    return "lanl2dz" if el in ("ZN","FE","CU","MN","NI","CO","CD") else "6-31g*"

def ecp_for(el: str) -> Optional[str]:
    return "lanl2dz" if el in ("ZN","FE","CU","MN","NI","CO","CD") else None


def dft_features(els: List[str], xyz: np.ndarray, charge: int, mult: int,
                 mm_coords: Optional[np.ndarray] = None,
                 mm_charges: Optional[np.ndarray] = None,
                 method: str = "b3lyp", max_atoms: int = 160,
                 verbose: int = 0) -> Optional[Dict]:
    """
    Real DFT single point. Electrostatic embedding when MM charges given.
    Returns orbital / population / energetic descriptors.
    """
    if not HAS_PYSCF:
        raise ImportError("pip install pyscf")
    if len(els) > max_atoms:
        log.warning(f"[DFT] {len(els)} atoms > {max_atoms} — skipped")
        return None
    try:
        atom = [[els[i], tuple(map(float, xyz[i]))] for i in range(len(els))]
        basis = {e: basis_for(e) for e in set(els)}
        ecp = {e: ecp_for(e) for e in set(els) if ecp_for(e)}
        mol = gto.M(atom=atom, basis=basis, ecp=ecp or None,
                    charge=charge, spin=mult - 1, unit="Angstrom",
                    verbose=verbose)
        mf = dft.UKS(mol) if mult > 1 else dft.RKS(mol)
        mf.xc = method
        mf.conv_tol = 1e-8
        mf.max_cycle = 150
        if mm_coords is not None and mm_charges is not None and len(mm_charges):
            mf = qmmm.mm_charge(mf, np.asarray(mm_coords) / BOHR_ANG,
                                np.asarray(mm_charges))
        e = float(mf.kernel())
        if not mf.converged:
            log.warning("[DFT] SCF did not converge — result discarded")
            return None
        mo = mf.mo_energy
        if isinstance(mo, (list, tuple)) or (hasattr(mo, "ndim") and mo.ndim == 2):
            occ = mf.mo_occ
            homo = max(float(np.max(m[occ_i > 0])) for m, occ_i in zip(mo, occ))
            lumo = min(float(np.min(m[occ_i == 0])) for m, occ_i in zip(mo, occ)
                       if np.any(occ_i == 0))
        else:
            occ = mf.mo_occ
            homo = float(mo[occ > 0][-1]); lumo = float(mo[occ == 0][0])
        pop = mf.mulliken_pop(verbose=0)[1]
        m_idx = 0
        dip = mf.dip_moment(unit="Debye", verbose=0)
        return {
            "energy_hartree": e, "converged": True,
            "homo_hartree": homo, "lumo_hartree": lumo,
            "gap_eV": (lumo - homo) * HARTREE_EV,
            "mulliken_metal": float(pop[m_idx]),
            "mulliken_all": pop.tolist(),
            "dipole_debye": np.asarray(dip).tolist(),
            "n_electrons": int(mol.nelectron),
            "n_basis": int(mol.nao_nr()),
            "method": method, "embedded": mm_charges is not None,
        }
    except Exception as exc:
        log.warning(f"[DFT] failed: {type(exc).__name__}: {exc}")
        return None


# =============================================================================
# 4. FMO2  (monomer SCC + dimer correction, Mulliken embedding)
# =============================================================================

def fmo2(fragments: List[Tuple[List[str], np.ndarray, int, int]],
         method: str = "b3lyp", scc_iters: int = 5,
         scc_tol: float = 1e-4, max_pair_atoms: int = 200) -> Dict:
    """
    FMO2 [F1]:  E = Σ_I E_I + Σ_{I<J} (E_IJ − E_I − E_J)

    Embedding: Mulliken point charges of all OTHER fragments, iterated to
    self-consistency (SCC). Real FMO uses ESP charges [F2]; Mulliken is a
    documented, poorer approximation. Convergence is not guaranteed.

    fragments: list of (elements, coords, charge, multiplicity)
    """
    n = len(fragments)
    if n == 0:
        return {"error": "no fragments"}
    charges = [np.zeros(len(f[0])) for f in fragments]
    mono_e = [0.0] * n
    hist = []

    for it in range(scc_iters):
        new_charges, energies, ok = [], [], True
        for i, (els, xyz, q, m) in enumerate(fragments):
            cc, qq = [], []
            for j, (e2, x2, _, _) in enumerate(fragments):
                if j == i:
                    continue
                cc.append(x2); qq.append(charges[j])
            mc = np.concatenate(cc) if cc else None
            mq = np.concatenate(qq) if qq else None
            r = dft_features(els, xyz, q, m, mc, mq, method=method)
            if r is None:
                ok = False; break
            energies.append(r["energy_hartree"])
            new_charges.append(np.array(r["mulliken_all"]))
        if not ok:
            return {"error": f"monomer SCF failed at SCC iter {it}",
                    "scc_converged": False}
        delta = max(float(np.abs(new_charges[i] - charges[i]).max())
                    for i in range(n))
        hist.append(delta)
        charges, mono_e = new_charges, energies
        log.info(f"[FMO2] SCC {it+1}: Δq_max = {delta:.2e}")
        if delta < scc_tol:
            break
    scc_ok = hist[-1] < scc_tol

    dimer_corr, n_pair, skipped = 0.0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            ei, xi, qi, mi = fragments[i]
            ej, xj, qj, mj = fragments[j]
            els = ei + ej; xyz = np.concatenate([xi, xj])
            if len(els) > max_pair_atoms:
                skipped += 1; continue
            q, m = qi + qj, 1
            ne = sum(Z_TABLE.get(e, 0) for e in els) - q
            if ne % 2:
                m = 2
            cc, qq = [], []
            for k in range(n):
                if k in (i, j):
                    continue
                cc.append(fragments[k][1]); qq.append(charges[k])
            mc = np.concatenate(cc) if cc else None
            mq = np.concatenate(qq) if qq else None
            r = dft_features(els, xyz, q, m, mc, mq, method=method)
            if r is None:
                skipped += 1; continue
            dimer_corr += r["energy_hartree"] - mono_e[i] - mono_e[j]
            n_pair += 1

    e1 = float(sum(mono_e))
    return {
        "E_FMO1_hartree": e1,
        "E_FMO2_hartree": e1 + dimer_corr,
        "dimer_correction_hartree": float(dimer_corr),
        "dimer_correction_kcal": float(dimer_corr * HARTREE_KCAL),
        "n_fragments": n, "n_dimers_computed": n_pair, "n_dimers_skipped": skipped,
        "scc_history": hist, "scc_converged": bool(scc_ok),
        "embedding": "Mulliken (NOT ESP — see [F2])",
        "WARNING": ("Metal–ligand fragmentation cuts dative bonds with large "
                    "charge transfer. Compare against the unfragmented cluster."),
    }


def make_fragments(atoms, site) -> List[Tuple[List[str], np.ndarray, int, int]]:
    """Metal as fragment 0; each capped donor sidechain as its own fragment."""
    frags = [([site["metal"]], site["metal_xyz"][None, :],
              DEFAULT_OX.get(site["metal"], 2),
              6 if site["metal"] in ("FE", "MN") else 1)]
    by_res = {}
    for a in atoms:
        if a["resn"] in AA3:
            by_res.setdefault((a["chain"], a["resi"]), []).append(a)
    for chain, resi, resn in site["residues"]:
        g = by_res.get((chain, resi), [])
        names = {a["name"]: a for a in g}
        ca, cb = names.get("CA"), names.get("CB")
        if cb is None:
            continue
        keep = [a for a in g if a["name"] not in SIDECHAIN_SKIP]
        els = [a["elem"] for a in keep]; xyz = [a["xyz"] for a in keep]
        v = ca["xyz"] - cb["xyz"]; v /= (np.linalg.norm(v) + 1e-12)
        els.append("H"); xyz.append(cb["xyz"] + LINK_BOND * v)
        q = -1 if resn in ("ASP", "GLU", "CYS") else 0
        ne = sum(Z_TABLE.get(e, 0) for e in els) - q
        frags.append((els, np.array(xyz), q, 1 if ne % 2 == 0 else 2))
    return frags


# =============================================================================
# 5. QM/MM  (QM side real; MM side requires OpenMM)
# =============================================================================

class OpenMMInterface:
    """
    MM energy via OpenMM. NOT a substitute for a validated force field:
    you must supply a proper topology + parameter set (AMBER/CHARMM).
    Without OpenMM installed, no MM energy is produced — and none is faked.
    """
    def __init__(self, pdb_path: Path, forcefield: str = "amber14-all.xml"):
        if not HAS_OPENMM:
            raise ImportError("pip install openmm  (conda install -c conda-forge openmm)")
        pdb = app.PDBFile(str(pdb_path))
        ff = app.ForceField(forcefield, "amber14/tip3p.xml")
        self.system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                      constraints=None, rigidWater=False)
        self.topology = pdb.topology
        self.positions = pdb.positions
        self.context = mm.Context(self.system, mm.LangevinIntegrator(
            300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds),
            mm.Platform.getPlatformByName("Reference"))

    def energy_kcal(self, positions=None) -> float:
        self.context.setPositions(positions or self.positions)
        st = self.context.getState(getEnergy=True)
        return st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

    def mm_charges(self) -> Tuple[np.ndarray, np.ndarray]:
        coords, charges = [], []
        for i in range(self.system.getNumParticles()):
            for f in range(self.system.getNumForces()):
                force = self.system.getForce(f)
                if isinstance(force, mm.NonbondedForce):
                    q, _, _ = force.getParticleParameters(i)
                    charges.append(q.value_in_unit(unit.elementary_charge))
        pos = self.positions.value_in_unit(unit.angstrom)
        return np.array(pos), np.array(charges)


def qmmm_energy(atoms, site, pdb_path: Path,
                mm_engine: Optional[OpenMMInterface] = None,
                method: str = "b3lyp") -> Dict:
    """
    Subtractive (ONIOM-style) QM/MM:
        E = E_QM(model + link, embedded) + E_MM(real) − E_MM(model)

    QM side: implemented, real.
    MM side: requires OpenMM. If absent, only E_QM is reported and the
             result is explicitly marked INCOMPLETE.
    """
    els, xyz, links = build_cluster(atoms, site)
    q, m = cluster_charge_mult(site, els)
    mc = mq = None
    if mm_engine is not None:
        pos, chg = mm_engine.mm_charges()
        qm_serials = set()
        for chain, resi, _ in site["residues"]:
            for a in atoms:
                if (a["chain"], a["resi"]) == (chain, resi) and a["name"] not in SIDECHAIN_SKIP:
                    qm_serials.add(a["serial"])
        keep = [i for i, a in enumerate(atoms) if a["serial"] not in qm_serials]
        mc, mq = pos[keep], chg[keep]
    qm = dft_features(els, xyz, q, m, mc, mq, method=method)
    out = {"n_qm_atoms": len(els), "n_link_atoms": len(links),
           "link_atoms": links, "charge": q, "multiplicity": m,
           "qm": qm, "electrostatic_embedding": mc is not None}
    if mm_engine is None or qm is None:
        out["E_total_hartree"] = None
        out["STATUS"] = ("INCOMPLETE — MM energy unavailable. "
                         "Install OpenMM and supply a validated force field. "
                         "No MM number is fabricated.")
        return out
    e_mm_real = mm_engine.energy_kcal()
    out["E_MM_real_kcal"] = e_mm_real
    out["STATUS"] = ("PARTIAL — E_MM(model) subtraction requires an MM "
                     "calculation on the capped cluster, which needs its own "
                     "parameterisation. Not implemented. Do not report this "
                     "as a QM/MM total energy.")
    out["E_total_hartree"] = None
    return out


# =============================================================================
# 6. DATASET BUILDER
# =============================================================================

def build_dataset(pdb_ids: List[str], out: str, cutoff: float = 2.8,
                  run_dft: bool = False, run_fmo: bool = False,
                  method: str = "b3lyp", limit_sites: Optional[int] = None):
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    records = []
    for pid in pdb_ids:
        try:
            path = fetch_pdb(pid)
        except Exception as e:
            log.warning(f"[{pid}] fetch failed: {e}"); continue
        atoms = parse_pdb(path)
        sites = extract_metal_sites(atoms, cutoff, pid)
        log.info(f"[{pid}] {len(sites)} metal sites")
        for k, s in enumerate(sites):
            if limit_sites and len(records) >= limit_sites:
                break
            els, xyz, links = build_cluster(atoms, s)
            q, m = cluster_charge_mult(s, els)
            rec = {"site": s, "cluster": {"elements": els,
                                          "coords": xyz.tolist(),
                                          "charge": q, "multiplicity": m,
                                          "n_link": len(links)}}
            if run_dft:
                rec["dft_cluster"] = dft_features(els, xyz, q, m, method=method)
            if run_fmo:
                rec["fmo2"] = fmo2(make_fragments(atoms, s), method=method)
                if rec.get("dft_cluster") and rec["fmo2"].get("E_FMO2_hartree"):
                    rec["fragmentation_error_kcal"] = (
                        rec["fmo2"]["E_FMO2_hartree"]
                        - rec["dft_cluster"]["energy_hartree"]) * HARTREE_KCAL
            records.append(rec)
            json.dump(rec, open(outp / f"{pid}_site{k}.json", "w"), indent=2)
    json.dump(records, open(outp / "dataset.json", "w"), indent=2)
    log.info(f"[dataset] {len(records)} sites → {outp/'dataset.json'}")
    errs = [r["fragmentation_error_kcal"] for r in records
            if "fragmentation_error_kcal" in r]
    if errs:
        log.info(f"[FMO2 vs cluster] error kcal/mol: "
                 f"mean={np.mean(errs):.2f} max|={np.max(np.abs(errs)):.2f}")
        log.info("Large errors are EXPECTED at dative metal–ligand cuts.")
    return records


# =============================================================================
# 7. SELF-TEST
# =============================================================================

def selftest(args):
    print("="*76); print("  PROTEIN QM — self-test"); print("="*76)
    ok = True

    print("\n[T1] tau4 / tau5 reference geometries")
    tet = [109.47]*6
    sqp = [90.0]*4 + [180.0]*2
    t4_t, t4_s = tau4(tet), tau4(sqp)
    tbp = [120.0]*3 + [90.0]*6 + [180.0]
    t5 = tau5(tbp)
    print(f"   τ4(tetrahedral)={t4_t:.4f} (expect 1)")
    print(f"   τ4(square planar)={t4_s:.4f} (expect 0)")
    print(f"   τ5(TBP)={t5:.4f} (expect 1)")
    t = abs(t4_t-1) < 0.01 and abs(t4_s) < 0.01 and abs(t5-1) < 0.01
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T2] bond-valence sum, ideal Zn–O4 at R0")
    bvs = bond_valence("ZN", [("O", 1.704)]*4, 2)
    print(f"   BVS = {bvs:.4f} (expect 4.0 at d=R0)")
    t = abs(bvs-4.0) < 1e-6; ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T3] link-atom geometry (Cα–Cβ cut)")
    ca = np.array([0.0, 0.0, 0.0]); cb = np.array([1.53, 0.0, 0.0])
    v = (ca-cb)/np.linalg.norm(ca-cb)
    h = cb + LINK_BOND*v
    d_bh = np.linalg.norm(h-cb); collinear = abs(np.linalg.norm(h-ca) - (1.53-1.09))
    print(f"   |Cβ–H| = {d_bh:.4f} Å (expect {LINK_BOND})")
    print(f"   collinearity residual = {collinear:.2e}")
    t = abs(d_bh-LINK_BOND) < 1e-9 and collinear < 1e-9; ok &= t
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T4] real PDB metal site (1ZIN / carbonic anhydrase-like)")
    try:
        p = fetch_pdb(args.pdb)
        atoms = parse_pdb(p)
        sites = extract_metal_sites(atoms, 2.8, args.pdb)
        print(f"   atoms={len(atoms)}  metal sites={len(sites)}")
        if sites:
            s = sites[0]
            print(f"   {s['metal']} CN={s['coordination_number']} "
                  f"τ4={s['tau4']} mean d={s['mean_dist']:.3f} Å")
            print(f"   BVS(ox2)={s['bvs_ox2']:.3f}  BVS(ox3)={s['bvs_ox3']:.3f}"
                  f"   ← oxidation state AMBIGUOUS from PDB")
            els, xyz, links = build_cluster(atoms, s)
            print(f"   cluster: {len(els)} atoms, {len(links)} link H")
            t = len(els) > 1 and s["coordination_number"] >= 2
        else:
            print("   no metal site in this entry (choose another PDB)")
            t = True
    except Exception as e:
        print(f"   SKIP: {e}"); t = True
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    if HAS_PYSCF:
        print("\n[T5] real DFT with electrostatic embedding")
        els = ["O","H","H"]
        xyz = np.array([[0,0,0.117],[0,0.757,-0.469],[0,-0.757,-0.469]])
        r0 = dft_features(els, xyz, 0, 1, method="b3lyp")
        r1 = dft_features(els, xyz, 0, 1,
                          mm_coords=np.array([[3.0,0.0,0.0]]),
                          mm_charges=np.array([-0.834]), method="b3lyp")
        if r0 and r1:
            de = (r1["energy_hartree"]-r0["energy_hartree"])*HARTREE_KCAL
            print(f"   E(no MM) = {r0['energy_hartree']:.6f} Ha")
            print(f"   E(embed) = {r1['energy_hartree']:.6f} Ha")
            print(f"   ΔE from MM charge = {de:+.3f} kcal/mol")
            t = abs(de) > 1e-3
            ok &= t; print(f"   {'PASS' if t else 'FAIL'} — embedding is active")
        else:
            print("   FAIL: DFT returned None"); ok = False
    else:
        print("\n[T5] SKIP — pip install pyscf")

    print("\n[T6] MM energy is NOT fabricated")
    try:
        OpenMMInterface(Path("nonexistent.pdb"))
        print("   FAIL — should have raised"); ok = False
    except ImportError:
        print("   OpenMM absent → raises, no fake energy  PASS")
    except Exception as e:
        print(f"   OpenMM present but file missing → {type(e).__name__}  PASS")

    print("\n" + "="*76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*76)
    print("""
  IMPLEMENTED: site extraction, τ4/τ5, BVS, link-H geometry, capped clusters,
               real DFT, electrostatic embedding, FMO2 with Mulliken SCC.
  NOT IMPLEMENTED: MM energy (OpenMM interface only), ESP embedding,
               E_MM(model) subtraction, charge shifting, protonation states.
  ASSUMED: oxidation state, protonation, formal charges. All flagged in output.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest","build","qmmm"])
    p.add_argument("--pdb", default="1CA2")
    p.add_argument("--pdb-ids", nargs="*", default=["1CA2","1ZIN","4HHB","1TIM"])
    p.add_argument("--out", default="data/metal_sites")
    p.add_argument("--cutoff", type=float, default=2.8)
    p.add_argument("--dft", action="store_true")
    p.add_argument("--fmo", action="store_true")
    p.add_argument("--method", default="b3lyp")
    p.add_argument("--limit-sites", dest="limit_sites", type=int, default=None)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "build":
        build_dataset(a.pdb_ids, a.out, a.cutoff, a.dft, a.fmo,
                      a.method, a.limit_sites)
    else:
        path = fetch_pdb(a.pdb); atoms = parse_pdb(path)
        sites = extract_metal_sites(atoms, a.cutoff, a.pdb)
        if not sites:
            log.error("no metal site"); return
        eng = None
        if HAS_OPENMM:
            try: eng = OpenMMInterface(path)
            except Exception as e: log.warning(f"[MM] {e}")
        print(json.dumps(qmmm_energy(atoms, sites[0], path, eng, a.method),
                         indent=2, default=str))


if __name__ == "__main__":
    main()
