#!/usr/bin/env python3
"""
=============================================================================
YEE-EDGE CONVENTION SENSITIVITY + MULTI-CELL CASCADE
for the Balsara interior-closure / coarse-circulation controllability question
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

LITERATURE
 [B1] Balsara (2001) JCP 174, 614.
 [B2] Balsara (2004) ApJS 151, 149.
 [T1] Toth & Roe (2002) JCP 180, 736.
 [Y1] Yee (1966) J. Comput. Phys. 1, 104.

PURPOSE
  The earlier single-cell test used ONE circulation convention (MAC face-plane
  line integral). A conclusion about whether the 5 interior DOF can control
  coarse circulation is NOT convention-independent unless tested. This file:

  1. Implements TWO conventions:
       mac_face : face-plane line integral using MAC face values with
                  trapezoidal weights (previous convention)
       yee_edge : Yee-style edge velocities obtained by interpolating MAC
                  face velocities to edges/corners, then summing edge
                  segments around the coarse face [Y1]
  2. Builds the divergence constraint matrix C_div and circulation matrix
     C_circ numerically by perturbation.
  3. Computes N = nullspace(C_div) and M = C_circ @ N.
       rank(M) == rank(C_circ)  -> all independent coarse circulations are
                                   controllable by interior closure
       rank(M) <  rank(C_circ)  -> genuine obstruction under that convention
  4. Repeats over multi-cell cascades (1x1x1, 2x1x1, 2x2x1, 2x2x2, ...)
     to test whether shared coarse faces/edges change the conclusion.

WHAT THIS IS NOT
  ❌ Not a full nonlinear AMR time-integration test.
  ❌ Not a proof that a particular Balsara/Toth-Roe scheme is superior.
  ❌ Not a validation against experiments or published benchmarks.
  ❌ Not a claim of "state of the art" or "paradigm shift".

HONEST INTERPRETATION
  The matrices are built with all KNOWN coarse-face values set to zero.
  That is legitimate for controllability because the known face prolongation
  contributes an affine offset; it does not change the linear span that the
  interior unknowns can affect.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Tuple

import numpy as np

VERSION = "1.0.0"

# Tangent axes for a face normal n, chosen cyclically.
TANGENTS = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


# =============================================================================
# 0. FIELD LAYOUT FOR A PATCH OF COARSE CELLS (refine ratio 2)
# =============================================================================

def fine_dims(coarse: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return tuple(2 * c for c in coarse)


def build_fields(F: Tuple[int, int, int]) -> List[np.ndarray]:
    """
    MAC fields on the fine patch:
      u : (Fx+1, Fy, Fz)
      v : (Fx, Fy+1, Fz)
      w : (Fx, Fy, Fz+1)
    """
    Fx, Fy, Fz = F
    u = np.zeros((Fx + 1, Fy, Fz), dtype=float)
    v = np.zeros((Fx, Fy + 1, Fz), dtype=float)
    w = np.zeros((Fx, Fy, Fz + 1), dtype=float)
    return [u, v, w]


def build_unknowns(F: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
    """
    Interior unknowns for a 2:1 Balsara prolongation over the patch.

    A fine face is UNKNOWN iff it is not on any coarse face plane:
      u unknown at i odd
      v unknown at j odd
      w unknown at k odd

    Returns list of (component, i, j, k).
    """
    Fx, Fy, Fz = F
    unk: List[Tuple[int, int, int, int]] = []
    for i in range(1, Fx, 2):
        for j in range(Fy):
            for k in range(Fz):
                unk.append((0, i, j, k))
    for i in range(Fx):
        for j in range(1, Fy, 2):
            for k in range(Fz):
                unk.append((1, i, j, k))
    for i in range(Fx):
        for j in range(Fy):
            for k in range(1, Fz, 2):
                unk.append((2, i, j, k))
    return unk


def set_unknown(fields: List[np.ndarray], unk: Tuple[int, int, int, int],
                val: float) -> None:
    comp, i, j, k = unk
    fields[comp][i, j, k] = val


# =============================================================================
# 1. DIVERGENCE CONSTRAINT MATRIX
# =============================================================================

def build_div_matrix(F: Tuple[int, int, int],
                     unknowns: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    C_div maps the interior unknowns to the divergence of every fine cell.
    Known coarse-face values are held at zero.
    """
    Fx, Fy, Fz = F
    colmap = {u: c for c, u in enumerate(unknowns)}
    n_rows = Fx * Fy * Fz
    A = np.zeros((n_rows, len(unknowns)), dtype=float)
    r = 0
    for i in range(Fx):
        for j in range(Fy):
            for k in range(Fz):
                terms = (
                    (0, i + 1, j, k, +1.0),
                    (0, i,     j, k, -1.0),
                    (1, i, j + 1, k, +1.0),
                    (1, i, j,     k, -1.0),
                    (2, i, j, k + 1, +1.0),
                    (2, i, j, k,     -1.0),
                )
                for comp, ii, jj, kk, s in terms:
                    c = colmap.get((comp, ii, jj, kk))
                    if c is not None:
                        A[r, c] += s
                r += 1
    return A


# =============================================================================
# 2. CIRCULATION CONVENTIONS
# =============================================================================

def node_value_yee(fields: List[np.ndarray], comp: int,
                   nodes: List[int], F: Tuple[int, int, int]) -> float:
    """
    Interpolate MAC component `comp` to a full grid node (Yee-edge style).
    The component is already located at face nodes along its own axis and
    at cell centers along the two transverse axes. Transverse interpolation
    averages the adjacent centers. At domain boundaries one-sided values are
    used.
    """
    arr = fields[comp]
    idx = [0, 0, 0]
    idx[comp] = nodes[comp]
    trans = [ax for ax in range(3) if ax != comp]
    centers = []
    for ax in trans:
        n = nodes[ax]
        L = F[ax]
        if n <= 0:
            centers.append([0])
        elif n >= L:
            centers.append([L - 1])
        else:
            centers.append([n - 1, n])
    total = 0.0
    count = 0
    for c0 in centers[0]:
        for c1 in centers[1]:
            ii = idx.copy()
            ii[trans[0]] = c0
            ii[trans[1]] = c1
            total += arr[tuple(ii)]
            count += 1
    return total / float(count)


def node_value_mac(fields: List[np.ndarray], comp: int, node_comp: int,
                   centers: Dict[int, int]) -> float:
    """
    MAC face-plane convention: take the MAC face value at the given
    component-axis face index and at explicitly chosen transverse centers.
    """
    arr = fields[comp]
    idx = [0, 0, 0]
    idx[comp] = node_comp
    for ax, c in centers.items():
        idx[ax] = c
    return arr[tuple(idx)]


def edge_integral(fields: List[np.ndarray], F: Tuple[int, int, int],
                  comp: int, start: int, end: int,
                  fixed_nodes: Dict[int, int],
                  mac_centers: Dict[int, int],
                  convention: str) -> float:
    """
    Integrate component `comp` along its axis from node `start` to `end`.
    The integral is the sum of segment values. Each segment is the average
    of its endpoint values, giving trapezoidal weights over face nodes.
    """
    total = 0.0
    for p in range(start, end):
        if convention == "yee":
            nodes0 = [0, 0, 0]
            nodes1 = [0, 0, 0]
            for ax, val in fixed_nodes.items():
                nodes0[ax] = val
                nodes1[ax] = val
            nodes0[comp] = p
            nodes1[comp] = p + 1
            v0 = node_value_yee(fields, comp, nodes0, F)
            v1 = node_value_yee(fields, comp, nodes1, F)
        else:
            v0 = node_value_mac(fields, comp, p, mac_centers)
            v1 = node_value_mac(fields, comp, p + 1, mac_centers)
        total += 0.5 * (v0 + v1)
    return total


def coarse_face_circulation(fields: List[np.ndarray],
                            F: Tuple[int, int, int],
                            cell: Tuple[int, int, int],
                            axis: int,
                            side: int,
                            convention: str) -> float:
    """
    Circulation around one coarse face.

    axis : face normal, 0/1/2
    side : -1 for the low face of the coarse cell, +1 for the high face
    convention : "mac_face" | "yee"
    """
    s = [2 * cell[ax] for ax in range(3)]
    e = [s[ax] + 2 for ax in range(3)]
    a, b = TANGENTS[axis]

    n_node = s[axis] if side == -1 else e[axis]
    # MAC face-plane convention uses the cell-adjacent center on the
    # normal side; Yee interpolation uses nodes and averages both sides.
    n_center = s[axis] if side == -1 else e[axis] - 1

    # Edge 1: +a at b = b0
    fixed_yee = {b: s[b], axis: n_node}
    mac_centers = {axis: n_center, b: s[b]}
    E1 = edge_integral(fields, F, a, s[a], e[a], fixed_yee, mac_centers,
                       convention)

    # Edge 2: +b at a = a1
    fixed_yee = {a: e[a], axis: n_node}
    mac_centers = {axis: n_center, a: e[a] - 1}
    E2 = edge_integral(fields, F, b, s[b], e[b], fixed_yee, mac_centers,
                       convention)

    # Edge 3: -a at b = b1
    fixed_yee = {b: e[b], axis: n_node}
    mac_centers = {axis: n_center, b: e[b] - 1}
    E3 = edge_integral(fields, F, a, s[a], e[a], fixed_yee, mac_centers,
                       convention)

    # Edge 4: -b at a = a0
    fixed_yee = {a: s[a], axis: n_node}
    mac_centers = {axis: n_center, a: s[a]}
    E4 = edge_integral(fields, F, b, s[b], e[b], fixed_yee, mac_centers,
                       convention)

    return E1 + E2 - E3 - E4


# =============================================================================
# 3. CIRCULATION CONSTRAINT MATRIX
# =============================================================================

def build_circ_constraints(coarse: Tuple[int, int, int]) -> List[Dict]:
    """All per-coarse-cell face circulations (6 per cell)."""
    cons = []
    for cx in range(coarse[0]):
        for cy in range(coarse[1]):
            for cz in range(coarse[2]):
                cell = (cx, cy, cz)
                for axis in range(3):
                    for side in (-1, +1):
                        cons.append({"cell": cell, "axis": axis, "side": side})
    return cons


def build_circ_matrix(F: Tuple[int, int, int],
                      coarse: Tuple[int, int, int],
                      unknowns: List[Tuple[int, int, int, int]],
                      convention: str) -> np.ndarray:
    """
    Build C_circ by perturbation: set one interior unknown to 1, all known
    coarse faces remain 0, then evaluate every coarse-face circulation.
    """
    cons = build_circ_constraints(coarse)
    C = np.zeros((len(cons), len(unknowns)), dtype=float)
    for col, unk in enumerate(unknowns):
        fields = build_fields(F)
        set_unknown(fields, unk, 1.0)
        for row, c in enumerate(cons):
            C[row, col] = coarse_face_circulation(fields, F, c["cell"],
                                                  c["axis"], c["side"],
                                                  convention)
    return C


# =============================================================================
# 4. SVD UTILITIES
# =============================================================================

def svd_tol(A: np.ndarray) -> float:
    if A.size == 0 or min(A.shape) == 0:
        return 1e-12
    s = np.linalg.svd(A, compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 1e-12
    return max(1e-10, max(A.shape) * np.finfo(float).eps * s[0] * 10.0)


def rank_of(A: np.ndarray) -> int:
    if A.size == 0 or min(A.shape) == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    tol = svd_tol(A)
    return int((s > tol).sum())


def nullspace_of(A: np.ndarray) -> Tuple[np.ndarray, int]:
    """Returns (nullspace basis, rank)."""
    if A.size == 0:
        return np.zeros((A.shape[1], 0)), 0
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    tol = svd_tol(A)
    rank = int((s > tol).sum())
    N = vh[rank:].T
    return N, rank


# =============================================================================
# 5. ANALYSIS
# =============================================================================

def analyze_patch(coarse: Tuple[int, int, int], convention: str) -> Dict:
    t0 = time.perf_counter()
    F = fine_dims(coarse)
    unknowns = build_unknowns(F)
    n_unknown = len(unknowns)
    n_cells = coarse[0] * coarse[1] * coarse[2]

    # Divergence constraints
    C_div = build_div_matrix(F, unknowns)
    N, rank_div = nullspace_of(C_div)
    nullity = N.shape[1]

    # Circulation constraints
    C_circ = build_circ_matrix(F, coarse, unknowns, convention)
    rank_circ = rank_of(C_circ)

    # Controllability of circulation by the divergence-free interior closure
    if nullity == 0:
        rank_ctrl = 0
    else:
        M = C_circ @ N
        rank_ctrl = rank_of(M)

    deficiency = max(0, rank_circ - rank_ctrl)
    overconstraint = max(0, rank_circ - nullity)

    if rank_circ == 0:
        verdict = "NO CIRCULATION FUNCTIONALS DETECTED"
    elif deficiency == 0 and overconstraint == 0:
        verdict = "FULL CONTROL: interior closure can match all independent circulations"
    elif overconstraint > 0:
        verdict = ("OVERCONSTRAINED: more independent circulation functionals "
                   "than interior DOF")
    else:
        verdict = ("OBSTRUCTION: interior closure cannot control all "
                   "independent circulations")

    return {
        "convention": convention,
        "coarse_patch": coarse,
        "fine_patch": F,
        "n_coarse_cells": n_cells,
        "n_unknowns": n_unknown,
        "expected_nullity_5N": 5 * n_cells,
        "rank_div": rank_div,
        "expected_rank_div_7N": 7 * n_cells,
        "nullity": nullity,
        "rank_circ": rank_circ,
        "rank_controlled": rank_ctrl,
        "deficiency": deficiency,
        "overconstraint": overconstraint,
        "verdict": verdict,
        "elapsed_s": time.perf_counter() - t0,
    }


def print_table(rows: List[Dict]) -> None:
    print("=" * 112)
    print(f"{'convention':<10}{'patch':<10}{'cells':>6}{'unk':>6}"
          f"{'rankDiv':>9}{'null':>6}{'rankCirc':>10}{'ctrl':>6}"
          f"{'def':>5}  verdict")
    print("-" * 112)
    for r in rows:
        patch = "x".join(map(str, r["coarse_patch"]))
        print(f"{r['convention']:<10}{patch:<10}{r['n_coarse_cells']:>6}"
              f"{r['n_unknowns']:>6}{r['rank_div']:>9}{r['nullity']:>6}"
              f"{r['rank_circ']:>10}{r['rank_controlled']:>6}"
              f"{r['deficiency']:>5}  {r['verdict']}")
    print("=" * 112)


def compare_conventions(rows: List[Dict]) -> None:
    mac = [r for r in rows if r["convention"] == "mac_face"]
    yee = [r for r in rows if r["convention"] == "yee"]
    print("\nCONVENTION SENSITIVITY")
    print("-" * 78)
    stable = True
    for rm, ry in zip(mac, yee):
        same = (rm["deficiency"] == ry["deficiency"]
                and rm["rank_controlled"] == ry["rank_controlled"])
        stable &= same
        tag = "SAME" if same else "DIFFERENT"
        print(f"  patch {'x'.join(map(str, rm['coarse_patch'])):<8} "
              f"mac_ctrl={rm['rank_controlled']}/{rm['rank_circ']}  "
              f"yee_ctrl={ry['rank_controlled']}/{ry['rank_circ']}  -> {tag}")
    print("-" * 78)
    if stable:
        print("  Conclusion is STABLE across the two circulation conventions tested.")
    else:
        print("  Conclusion is CONVENTION-DEPENDENT. Any claim must state the")
        print("  convention explicitly; do NOT generalize beyond it.")


# =============================================================================
# 6. SELF-TEST / DRIVER
# =============================================================================

def self_test(patches: List[Tuple[int, int, int]],
              conventions: Tuple[str, ...] = ("mac_face", "yee")) -> bool:
    print("=" * 112)
    print(f"  YEE-EDGE SENSITIVITY + MULTI-CELL CASCADE v{VERSION}")
    print("=" * 112)
    ok = True

    rows: List[Dict] = []
    for conv in conventions:
        for patch in patches:
            r = analyze_patch(patch, conv)
            rows.append(r)
            print(f"  done {conv:<9} patch {'x'.join(map(str, patch)):<8} "
                  f"in {r['elapsed_s']:.2f}s")

    print()
    print_table(rows)
    compare_conventions(rows)

    # Basic sanity assertions
    print("\nSANITY CHECKS")
    print("-" * 78)
    for r in rows:
        exp_null = r["expected_nullity_5N"]
        exp_rank = r["expected_rank_div_7N"]
        t1 = r["nullity"] == exp_null
        t2 = r["rank_div"] == exp_rank
        ok &= t1 and t2
        print(f"  {r['convention']:<9} {'x'.join(map(str, r['coarse_patch'])):<8} "
              f"rank_div={r['rank_div']} (expect {exp_rank}) "
              f"nullity={r['nullity']} (expect {exp_null})  "
              f"[{'PASS' if t1 and t2 else 'FAIL'}]")
    print("-" * 78)

    print("""
HOW TO READ THE TABLE
  rankDiv   : rank of divergence constraints on interior unknowns.
  null      : divergence-free interior DOF; expected 5 per coarse cell.
  rankCirc  : independent coarse-circulation functionals visible to the
              interior unknowns under the chosen convention.
  ctrl      : rank of C_circ restricted to the divergence-free nullspace.
  def       : rankCirc - ctrl. Zero means all independent circulation
              constraints are controllable by an interior closure.

INTERPRETATION
  • If def = 0 for both conventions and all cascades:
      the earlier claim "no interior closure can preserve coarse circulation"
      is NOT supported by these conventions.
  • If def > 0 for one convention only:
      the conclusion is convention-dependent and must be stated as such.
  • If def > 0 for both and stable under cascade:
      there is a genuine linear obstruction under both conventions, closer
      to the Toth-Roe motivation, but still not a proof that Toth-Roe is
      the unique remedy.

WHAT IS STILL NOT PROVEN
  ❌ This does not prove nonlinear AMR accuracy or stability.
  ❌ This does not validate against Toth-Roe's published formulas.
  ❌ This does not establish a paradigm shift.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest", choices=["selftest"])
    p.add_argument("--quick", action="store_true",
                   help="run smaller cascade only")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    if a.quick:
        patches = [(1, 1, 1), (2, 1, 1), (2, 2, 1)]
    else:
        patches = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2), (3, 3, 3)]

    ok = self_test(patches)

    if a.json:
        rows = []
        for patch in patches:
            for conv in ("mac_face", "yee"):
                rows.append(analyze_patch(patch, conv))
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"json -> {a.json}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
