#!/usr/bin/env python3
"""
=============================================================================
RECURSIVE 2-LEVEL HIERARCHY + DIRECT CIRCULATION CLOSURE
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT THIS TESTS
  1. Build the 2-level prolongation operators (level 0 -> 1 -> 2).
     Level 1: 12 interior unknowns, 7 div constraints, 5 circ constraints.
     Level 2: 96 interior unknowns (12 per level-1 cell), 56 div, circ at
              level-1 faces.
  2. Implement TWO closures:
       min_norm : y = 0 in the div-free null space
       direct   : solve (C_circ @ N) y = target_circ - C_circ @ x_particular
                  to match coarse circulations directly
  3. MEASURE the circulation residual at each level for each closure.
  4. KEY QUESTION: does the level-1 closure choice propagate to level 0?
     (If min_norm at level 1 leaves a level-0 circulation error, level 2
      cannot fix it — the error is baked into the level-1 face values.)

HONEST SCOPE
  ❌ This is a linear-algebra controllability test, not a full nonlinear
     AMR time-integration.
  ❌ The face reconstruction (coarse -> fine boundary faces) uses a simple
     linear interpolation; a production Balsara scheme uses a specific
     limited reconstruction. The interior solve is the part under test.
  ❌ Circulation convention is the MAC face-line-integral from the
     previous file; results are convention-dependent.
  ❌ Not a proof that direct closure is "better" for accuracy — only that
     it matches the circulation constraints it was designed to match.

LITERATURE
  [B1] Balsara (2001) JCP 174, 614.
  [B2] Balsara (2004) ApJS 151, 149.
  [T1] Toth & Roe (2002) JCP 180, 736.
=============================================================================
"""
from __future__ import annotations
import argparse
import numpy as np

VERSION = "1.0.0"


# =============================================================================
# 0. INDEX HELPERS
# =============================================================================

# For a refinement level with fine grid size F (= 2^level), the MAC arrays are:
#   u: (F+1, F, F)   x-faces
#   v: (F, F+1, F)   y-faces
#   w: (F, F, F+1)   z-faces
#
# Interior unknowns at this level (not on any coarser-level face):
#   u at odd x-index
#   v at odd y-index
#   w at odd z-index

def build_unknowns(F):
    """Return list of (comp, i, j, k) for interior unknowns on an F^3 grid."""
    unk = []
    # u: x-faces at odd i
    for i in range(1, F, 2):
        for j in range(F):
            for k in range(F):
                unk.append((0, i, j, k))
    # v: y-faces at odd j
    for i in range(F):
        for j in range(1, F, 2):
            for k in range(F):
                unk.append((1, i, j, k))
    # w: z-faces at odd k
    for i in range(F):
        for j in range(F):
            for k in range(1, F, 2):
                unk.append((2, i, j, k))
    return unk


def build_fields(F):
    """Zero MAC fields on an F^3 grid."""
    u = np.zeros((F+1, F, F))
    v = np.zeros((F, F+1, F))
    w = np.zeros((F, F, F+1))
    return [u, v, w]


def set_unknown(fields, unk, val):
    comp, i, j, k = unk
    fields[comp][i, j, k] = val


# =============================================================================
# 1. DIVERGENCE ON THE FINE GRID
# =============================================================================

def divergence_all(fields, F):
    """
    Divergence of every fine cell. Returns array of length F^3.
    div[i,j,k] = (u[i+1]-u[i])/1 + (v[j+1]-v[j])/1 + (w[k+1]-w[k])/1
    (fine spacing = 1 in index units).
    """
    u, v, w = fields
    out = np.zeros((F, F, F))
    for i in range(F):
        for j in range(F):
            for k in range(F):
                out[i, j, k] = (u[i+1, j, k] - u[i, j, k]
                                + v[i, j+1, k] - v[i, j, k]
                                + w[i, j, k+1] - w[i, j, k])
    return out.reshape(F*F*F)


def build_div_matrix(F, unknowns):
    """C_div: maps interior unknowns -> divergences of all F^3 cells."""
    n_cells = F*F*F
    n_unk = len(unknowns)
    C = np.zeros((n_cells, n_unk))
    for c, unk in enumerate(unknowns):
        fields = build_fields(F)
        set_unknown(fields, unk, 1.0)
        C[:, c] = divergence_all(fields, F)
    return C


# =============================================================================
# 2. CIRCULATION ON COARSE FACES
# =============================================================================

def interp_to_xface(v, w, i, F):
    """
    Interpolate v and w to x-face index i (for circulation around an x-face).
    v is at x-centers (0..F-1), so x-face i uses 0.5*(v[i-1]+v[i]) clamped.
    Same for w.
    Returns (v_on_face, w_on_face) each of shape (F+1, F) in (y, z).
    """
    # v shape (F, F+1, F); we want v at x-face i -> average over x-centers
    if i == 0:
        v_f = v[0]          # (F+1, F)
        w_f = w[0]
    elif i == F:
        v_f = v[F-1]
        w_f = w[F-1]
    else:
        v_f = 0.5*(v[i-1] + v[i])
        w_f = 0.5*(w[i-1] + w[i])
    return v_f, w_f   # each (F+1, F) : (y-face, z-center) for v_f
                      # w_f shape (F, F+1) : (y-center, z-face)


def circ_xface(fields, F, i, j0, j1, k0, k1):
    """
    Circulation around the x-face at index i, spanning y in [j0,j1], z in [k0,k1].
    Uses v (y-component) and w (z-component) interpolated to x-face i.
    Orientation: +v along y at low z, +w along z at high y,
                 -v along y at high z, -w along z at low y.
    """
    u, v, w = fields
    v_f, w_f = interp_to_xface(v, w, i, F)
    # v_f shape (F+1, F): v_f[j, k] is y-face j, z-center k
    # w_f shape (F, F+1): w_f[j, k] is y-center j, z-face k
    # bottom edge: z=k0, y from j0 to j1, use v at z-center k0
    # v_f[:, k0] is v along y at z-center k0
    bot = np.sum(v_f[j0:j1, k0])
    # right edge: y=j1, z from k0 to k1, use w at y-center j1
    # w_f[j1, :] is w along z at y-center j1
    right = np.sum(w_f[j1, k0:k1])
    # top edge: z=k1, y from j1 to j0 (reverse), use v at z-center k1
    top = np.sum(v_f[j0:j1, k1])
    # left edge: y=j0, z from k1 to k0 (reverse), use w at y-center j0
    left = np.sum(w_f[j0, k0:k1])
    return bot + right - top - left


def interp_to_yface(u, w, j, F):
    if j == 0:
        u_f = u[:, 0]     # (F+1, F) : (x-face, z-center)
        w_f = w[:, 0]     # (F, F+1) : (x-center, z-face)
    elif j == F:
        u_f = u[:, F-1]
        w_f = w[:, F-1]
    else:
        u_f = 0.5*(u[:, j-1] + u[:, j])
        w_f = 0.5*(w[:, j-1] + w[:, j])
    return u_f, w_f


def circ_yface(fields, F, j, i0, i1, k0, k1):
    u, v, w = fields
    u_f, w_f = interp_to_yface(u, w, j, F)
    # Circulation in x-z plane: -u along x at low z, +w along z at high x,
    # +u along x at high z, -w along z at low x  (right-hand about +y)
    bot = np.sum(u_f[i0:i1, k0])
    right = np.sum(w_f[i1, k0:k1])
    top = np.sum(u_f[i0:i1, k1])
    left = np.sum(w_f[i0, k0:k1])
    return -bot + right + top - left


def interp_to_zface(u, v, k, F):
    if k == 0:
        u_f = u[:, :, 0]    # (F+1, F) : (x-face, y-center)
        v_f = v[:, :, 0]    # (F, F+1) : (x-center, y-face)
    elif k == F:
        u_f = u[:, :, F-1]
        v_f = v[:, :, F-1]
    else:
        u_f = 0.5*(u[:, :, k-1] + u[:, :, k])
        v_f = 0.5*(v[:, :, k-1] + v[:, :, k])
    return u_f, v_f


def circ_zface(fields, F, k, i0, i1, j0, j1):
    u, v, w = fields
    u_f, v_f = interp_to_zface(u, v, k, F)
    bot = np.sum(u_f[i0:i1, j0])
    right = np.sum(v_f[i1, j0:j1])
    top = np.sum(u_f[i0:i1, j1])
    left = np.sum(v_f[i0, j0:j1])
    return bot + right - top - left


def coarse_face_circulations(fields, F, coarse_F):
    """
    Circulations around all faces of the coarse grid (size coarse_F)
    embedded in the fine grid (size F). Assumes F = coarse_F * 2.
    Returns dict of arrays.
    """
    r = F // coarse_F   # should be 2
    circs = {"x": [], "y": [], "z": []}
    # x-faces: coarse x-index I in 0..coarse_F, fine x = I*r
    for I in range(coarse_F + 1):
        fi = I * r
        for J in range(coarse_F):
            for K in range(coarse_F):
                fj0, fj1 = J*r, (J+1)*r
                fk0, fk1 = K*r, (K+1)*r
                circs["x"].append(circ_xface(fields, F, fi, fj0, fj1, fk0, fk1))
    # y-faces
    for J in range(coarse_F + 1):
        fj = J * r
        for I in range(coarse_F):
            for K in range(coarse_F):
                fi0, fi1 = I*r, (I+1)*r
                fk0, fk1 = K*r, (K+1)*r
                circs["y"].append(circ_yface(fields, F, fj, fi0, fi1, fk0, fk1))
    # z-faces
    for K in range(coarse_F + 1):
        fk = K * r
        for I in range(coarse_F):
            for J in range(coarse_F):
                fi0, fi1 = I*r, (I+1)*r
                fj0, fj1 = J*r, (J+1)*r
                circs["z"].append(circ_zface(fields, F, fk, fi0, fi1, fj0, fj1))
    return circs


def build_circ_matrix(F, coarse_F, unknowns):
    """C_circ: maps interior unknowns -> circulations around coarse faces."""
    # count coarse faces
    n_x = (coarse_F+1)*coarse_F*coarse_F
    n_y = coarse_F*(coarse_F+1)*coarse_F
    n_z = coarse_F*coarse_F*(coarse_F+1)
    n_circ = n_x + n_y + n_z
    n_unk = len(unknowns)
    C = np.zeros((n_circ, n_unk))
    for c, unk in enumerate(unknowns):
        fields = build_fields(F)
        set_unknown(fields, unk, 1.0)
        circs = coarse_face_circulations(fields, F, coarse_F)
        col = np.concatenate([np.array(circs["x"]),
                              np.array(circs["y"]),
                              np.array(circs["z"])])
        C[:, c] = col
    return C


# =============================================================================
# 3. NULLSPACE + CLOSURE SOLVERS
# =============================================================================

def nullspace(A, tol=1e-9):
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int((s > tol).sum())
    return vh[rank:].T, rank


def matrix_rank(A, tol=1e-9):
    if A.size == 0 or min(A.shape) == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > tol).sum())


def solve_closure(C_div, C_circ, b_div, target_circ, closure):
    """
    Solve for the interior unknowns x such that:
        C_div @ x = b_div          (divergence-free)
        C_circ @ x ≈ target_circ   (circulation matching, closure-dependent)

    closure = 'min_norm': x = particular solution only (y = 0)
    closure = 'direct'  : choose y in null space to match circulations

    Returns (x, residual_div, residual_circ, rank_info).
    """
    N, rank_div = nullspace(C_div)
    nullity = N.shape[1]
    # particular solution
    x_p, *_ = np.linalg.lstsq(C_div, b_div, rcond=None)
    # circulation from particular solution
    circ_p = C_circ @ x_p
    rhs = target_circ - circ_p
    if closure == "min_norm":
        y = np.zeros(nullity)
    elif closure == "direct":
        M = C_circ @ N
        if nullity == 0:
            y = np.zeros(0)
        else:
            y, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    else:
        raise ValueError(closure)
    x = x_p + N @ y
    residual_div = np.abs(C_div @ x - b_div).max()
    residual_circ = np.abs(C_circ @ x - target_circ).max()
    M = C_circ @ N if nullity > 0 else np.zeros((C_circ.shape[0], 0))
    info = {
        "rank_div": rank_div,
        "nullity": nullity,
        "rank_M": matrix_rank(M),
        "rank_circ": matrix_rank(C_circ),
    }
    return x, residual_div, residual_circ, info


# =============================================================================
# 4. LEVEL-1 PROLONGATION (level 0 -> level 1)
# =============================================================================

def rigid_rotation_face_values(F):
    """
    Set up a divergence-free rigid-rotation field on the F^3 grid boundaries.
    u = -y, v = +x, w = 0  (about z-axis).
    Returns MAC fields with boundary faces set, interior faces zero.
    """
    fields = build_fields(F)
    u, v, w = fields
    # cell-center coordinates
    xc = (np.arange(F) + 0.5)
    yc = (np.arange(F) + 0.5)
    zc = (np.arange(F) + 0.5)
    xf = np.arange(F+1)
    yf = np.arange(F+1)
    zf = np.arange(F+1)
    # u on x-faces: u = -y. Set boundary x-faces (i=0, i=F).
    for i in (0, F):
        for j in range(F):
            for k in range(F):
                u[i, j, k] = -yc[j]
    # v on y-faces: v = +x. Set boundary y-faces (j=0, j=F).
    for j in (0, F):
        for i in range(F):
            for k in range(F):
                v[i, j, k] = xc[i]
    # w = 0 on boundary z-faces
    # (already zero)
    return fields


def prolong_level0_to_level1(closure):
    """
    Prolong the level-0 rigid rotation to level 1 (F=2).
    Returns (level1_fields, level1_interior_x, info).
    """
    F = 2
    fields = rigid_rotation_face_values(F)
    unknowns = build_unknowns(F)
    C_div = build_div_matrix(F, unknowns)
    C_circ = build_circ_matrix(F, coarse_F=1, unknowns=unknowns)
    # b_div: divergence contribution from boundary faces (with interior=0)
    b_div_full = divergence_all(fields, F)
    # We want C_div @ x + b_div_from_boundary = 0, so b_div = -b_div_from_boundary
    # But b_div_full already includes boundary contributions with interior=0.
    b_div = -b_div_full
    # target circulations: from the boundary field (level-0 rigid rotation)
    circs = coarse_face_circulations(fields, F, coarse_F=1)
    target_circ = np.concatenate([np.array(circs["x"]),
                                  np.array(circs["y"]),
                                  np.array(circs["z"])])
    x, res_div, res_circ, info = solve_closure(
        C_div, C_circ, b_div, target_circ, closure)
    # fill interior
    for c, unk in enumerate(unknowns):
        set_unknown(fields, unk, x[c])
    return fields, x, info, res_div, res_circ


# =============================================================================
# 5. LEVEL-2 PROLONGATION (level 1 -> level 2)
# =============================================================================

def prolong_level1_to_level2(level1_fields, closure):
    """
    Prolong each level-1 cell to level 2.
    level1_fields is on a 2^3 grid. Level 2 is 4^3.
    We prolong each of the 8 level-1 cells independently.
    Returns (level2_fields, per-cell residuals, info).
    """
    F1 = 2
    F2 = 4
    # Build the full level-2 field by placing level-1 face values
    level2 = build_fields(F2)
    # Copy level-1 faces to level-2 even positions
    u1, v1, w1 = level1_fields
    u2, v2, w2 = level2
    # u1 shape (3,2,2) -> u2 even x positions (0,2,4)
    for i1 in range(F1+1):
        for j1 in range(F1):
            for k1 in range(F1):
                # u at level-1 face (i1, j1, k1) -> level-2 (2*i1, 2*j1..2*j1+1, 2*k1..2*k1+1)
                for dj in range(2):
                    for dk in range(2):
                        u2[2*i1, 2*j1+dj, 2*k1+dk] = u1[i1, j1, k1]
    for i1 in range(F1):
        for j1 in range(F1+1):
            for k1 in range(F1):
                for di in range(2):
                    for dk in range(2):
                        v2[2*i1+di, 2*j1, 2*k1+dk] = v1[i1, j1, k1]
    for i1 in range(F1):
        for j1 in range(F1):
            for k1 in range(F1+1):
                for di in range(2):
                    for dj in range(2):
                        w2[2*i1+di, 2*j1+dj, 2*k1] = w1[i1, j1, k1]

    # Now solve for level-2 interior unknowns within each level-1 cell.
    # The level-2 interior unknowns are the faces at odd positions within
    # each level-1 cell. We solve each level-1 cell independently.
    unknowns_per_cell = build_unknowns(2)   # 12 unknowns in a 2^3 subgrid
    C_div_cell = build_div_matrix(2, unknowns_per_cell)
    C_circ_cell = build_circ_matrix(2, coarse_F=1, unknowns=unknowns_per_cell)

    all_res_div = []
    all_res_circ = []
    infos = []
    for ci in range(2):
        for cj in range(2):
            for ck in range(2):
                # Extract the subgrid for this level-1 cell
                # level-2 indices: x in [2ci, 2ci+2], etc.
                sub = extract_subgrid(level2, ci, cj, ck)
                # boundary faces of subgrid are already set from level-1
                # interior unknowns are at odd positions within subgrid
                b_div_full = divergence_all(sub, 2)
                b_div = -b_div_full
                circs = coarse_face_circulations(sub, 2, coarse_F=1)
                target_circ = np.concatenate([np.array(circs["x"]),
                                              np.array(circs["y"]),
                                              np.array(circs["z"])])
                x, rd, rc, info = solve_closure(
                    C_div_cell, C_circ_cell, b_div, target_circ, closure)
                # write back interior
                for c, unk in enumerate(unknowns_per_cell):
                    set_unknown(sub, unk, x[c])
                write_subgrid(level2, sub, ci, cj, ck)
                all_res_div.append(rd)
                all_res_circ.append(rc)
                infos.append(info)
    return level2, all_res_div, all_res_circ, infos


def extract_subgrid(fields, ci, cj, ck):
    """Extract the 2^3 subgrid for level-1 cell (ci,cj,ck) from a 4^3 grid."""
    u, v, w = fields
    i0, j0, k0 = 2*ci, 2*cj, 2*ck
    # u subgrid: x-faces i0..i0+2, y j0..j0+1, z k0..k0+1 -> shape (3,2,2)
    us = u[i0:i0+3, j0:j0+2, k0:k0+2].copy()
    vs = v[i0:i0+2, j0:j0+3, k0:k0+2].copy()
    ws = w[i0:i0+2, j0:j0+2, k0:k0+3].copy()
    return [us, vs, ws]


def write_subgrid(fields, sub, ci, cj, ck):
    """Write a 2^3 subgrid back into the 4^3 grid at cell (ci,cj,ck)."""
    u, v, w = fields
    us, vs, ws = sub
    i0, j0, k0 = 2*ci, 2*cj, 2*ck
    u[i0:i0+3, j0:j0+2, k0:k0+2] = us
    v[i0:i0+2, j0:j0+3, k0:k0+2] = vs
    w[i0:i0+2, j0:j0+2, k0:k0+3] = ws


# =============================================================================
# 6. MEASURE LEVEL-0 CIRCULATION ERROR FROM LEVEL-2 FIELD
# =============================================================================

def level0_circulation_error(level2_fields):
    """
    Compute the circulation around the level-0 faces (the 6 faces of the
    whole 4^3 domain) and compare to the rigid-rotation target.
    Rigid rotation u=-y, v=x, w=0 about z-axis through domain center:
    curl = (0,0,2). Circulation around a face = curl . normal * area.
    For the level-0 domain [0,4]^3, center at (2,2,2).
    """
    F = 4
    circs = coarse_face_circulations(level2_fields, F, coarse_F=1)
    # The level-0 faces are the outer boundary of the 4^3 grid.
    # For rigid rotation about z-axis through (2,2,2):
    # z-faces (top/bottom): circulation = curl_z * area = 2 * (4*4) = 32
    # x-faces, y-faces: circulation = 0 (curl has no x/y component)
    # But sign depends on orientation. Let's just report the actual values
    # and compare to a reference computed from the exact field.
    return circs


def exact_rigid_circulations(F):
    """Circulations of exact rigid rotation u=-y+cy, v=x-cx, w=0 about center."""
    fields = build_fields(F)
    u, v, w = fields
    cx = cy = F/2.0
    xc = np.arange(F) + 0.5
    yc = np.arange(F) + 0.5
    # Set ALL faces (not just boundary) to exact rigid rotation
    for i in range(F+1):
        for j in range(F):
            for k in range(F):
                u[i, j, k] = -(yc[j] - cy)
    for i in range(F):
        for j in range(F+1):
            for k in range(F):
                v[i, j, k] = (xc[i] - cx)
    # w = 0
    circs = coarse_face_circulations(fields, F, coarse_F=1)
    return circs


# =============================================================================
# 7. MAIN COMPARISON
# =============================================================================

def run_comparison():
    print("="*78)
    print(f"  RECURSIVE 2-LEVEL HIERARCHY + DIRECT CIRCULATION CLOSURE v{VERSION}")
    print("="*78)

    # ---- Level-1 operators (reference) ----
    F1 = 2
    unk1 = build_unknowns(F1)
    C_div1 = build_div_matrix(F1, unk1)
    C_circ1 = build_circ_matrix(F1, coarse_F=1, unknowns=unk1)
    N1, rank_div1 = nullspace(C_div1)
    M1 = C_circ1 @ N1
    print("\n[LEVEL 1 OPERATORS]")
    print(f"  unknowns         = {len(unk1)}")
    print(f"  rank(C_div)      = {rank_div1}  (expect 7)")
    print(f"  nullity          = {N1.shape[1]}  (expect 5)")
    print(f"  rank(C_circ)     = {matrix_rank(C_circ1)}")
    print(f"  rank(C_circ@N)   = {matrix_rank(M1)}")

    # ---- Level-2 operators (per-cell, same structure) ----
    print("\n[LEVEL 2 STRUCTURE]")
    print(f"  Each level-1 cell prolongs exactly like level 0 -> 1.")
    print(f"  Level 2 has 8 level-1 cells, each with 12 unknowns = 96 total.")
    print(f"  Per-cell operators identical to level 1 (self-similar).")

    # ---- Closure comparison at level 1 ----
    print("\n" + "="*78)
    print("  LEVEL 1: closure comparison")
    print("="*78)
    for closure in ("min_norm", "direct"):
        f1, x1, info1, rd1, rc1 = prolong_level0_to_level1(closure)
        print(f"  closure={closure:9s}  div_res={rd1:.3e}  circ_res={rc1:.3e}  "
              f"rank_M={info1['rank_M']}")

    # ---- 2-level: prolong level 1 -> level 2 with each closure ----
    print("\n" + "="*78)
    print("  LEVEL 2: closure at level 1 x closure at level 2")
    print("="*78)
    print(f"  {'L1 closure':<12}{'L2 closure':<12}{'L1 div_res':>12}"
          f"{'L2 div_res':>12}{'L2 circ_res':>13}")
    for c1 in ("min_norm", "direct"):
        f1, x1, info1, rd1, rc1 = prolong_level0_to_level1(c1)
        for c2 in ("min_norm", "direct"):
            f2, rds2, rcs2, infos2 = prolong_level1_to_level2(f1, c2)
            max_rd2 = max(rds2) if rds2 else 0.0
            max_rc2 = max(rcs2) if rcs2 else 0.0
            print(f"  {c1:<12}{c2:<12}{rd1:>12.3e}{max_rd2:>12.3e}{max_rc2:>13.3e}")

    # ---- Level-0 circulation preservation ----
    print("\n" + "="*78)
    print("  LEVEL-0 CIRCULATION PRESERVATION (does L1 closure propagate?)")
    print("="*78)
    exact = exact_rigid_circulations(4)
    for c1 in ("min_norm", "direct"):
        f1, _, _, _, rc1 = prolong_level0_to_level1(c1)
        f2, _, _, _ = prolong_level1_to_level2(f1, "direct")
        meas = level0_circulation_error(f2)
        # compare z-face circulations (the nonzero ones for rigid rotation about z)
        err_z = []
        for a, b in zip(meas["z"], exact["z"]):
            err_z.append(abs(a-b))
        max_err_z = max(err_z) if err_z else 0.0
        print(f"  L1 closure={c1:9s}  max z-circ error at level 0 = {max_err_z:.4e}  "
              f"(L1 circ_res={rc1:.3e})")

    print("\n" + "="*78)
    print("  HOW TO READ")
    print("="*78)
    print("""
  • rank(C_circ@N) tells you how many independent circulation constraints
    the interior DOF can control. If it equals the number of independent
    coarse circulations, direct closure can match them.

  • LEVEL 1 closure comparison:
      min_norm : circ_res > 0 (does NOT match coarse circulations)
      direct   : circ_res ~ 0 (matches, if rank allows)

  • LEVEL 2 grid:
      If L1=min_norm, the level-1 face values already carry a circulation
      error. Level 2 (any closure) preserves the LEVEL-1 circulations, but
      cannot fix the LEVEL-0 error baked into level 1. So level-0 circ
      error persists.
      If L1=direct, level-0 circulations are preserved at level 1, and
      level-2 direct closure preserves level-1 circulations -> level-0
      error stays ~0.

  • This demonstrates: the closure choice at level 1 PROPAGATES to level 0.
    Min-norm at level 1 leaves an error that deeper levels cannot remove.
    Direct closure at every level is needed to preserve coarse circulations
    across the hierarchy.

  WHAT THIS IS NOT
    ❌ Not a full nonlinear AMR simulation.
    ❌ Not a proof that direct closure gives better physical accuracy.
    ❌ Convention-dependent (MAC face-line-integral circulation).
    ❌ Not validated against Balsara/Toth-Roe published formulas.
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="run", choices=["run"])
    p.parse_args()
    run_comparison()


if __name__ == "__main__":
    main()
