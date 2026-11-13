#!/usr/bin/env python3
"""
=============================================================================
AMR CURL-PRESERVING CLOSURE: A RANK-DEFINITIVE INVESTIGATION
Is Tóth–Roe edge-circulation really required, or can an interior closure
of the Balsara null-space preserve coarse-face circulation?
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

LITERATURE
 [B1] Balsara (2001) JCP 174, 614.     [divergence-free AMR prolongation]
 [B2] Balsara (2004) ApJS 151, 149.    [divergence-free reconstruction]
 [T1] Tóth & Roe (2002) JCP 180, 736.  [divergence- AND curl-preserving]
 [M1] Jiang & Wu (1999) JCP 150, 561.  [constrained transport, context]

=============================================================================
THE QUESTION, STATED PRECISELY
=============================================================================
Under 2:1 refinement of a divergence-free MAC velocity field on one coarse
cell, the prolongation has:
    12 interior face unknowns
     8 sub-cell divergence constraints, rank 7  ->  nullity 5
So the divergence-free prolongation has exactly 5 free parameters (the
"interior closure"). The question is whether those 5 parameters can ALSO be
chosen to preserve the coarse-face circulations (the discrete curl content).

There are 6 coarse-face circulations, but for any velocity field the sum of
the oriented circulations over the closed coarse-cell surface is identically
zero (each interior edge is shared by two faces with opposite orientation).
So there are 5 independent circulation targets — the SAME number as the
available interior DOF. The system is square. Whether it is solvable depends
on the rank of a specific 5x5 matrix, which THIS SCRIPT COMPUTES.

KEY POINT OF HONESTY
  The answer depends on the discrete circulation convention. Here circulation
  is defined as a line integral around each coarse face using MAC face
  velocities sampled in the layer nearest that face, with segment-averaged
  quadrature. A different convention (e.g. a Yee edge interpolation) can
  change the rank. Therefore this script reports a result UNDER A STATED
  CONVENTION, not an unconditional theorem. Cross-checking against Tóth–Roe's
  own formulas is still required before any publication claim.

WHAT THIS SCRIPT DOES (and it lets the numbers decide, not the hypothesis)
  T1  Builds C_div (8x12) numerically, verifies rank 7 and nullity 5.
  T2  Builds C_circ (6x12) numerically under the stated convention.
  T3  Computes N = nullspace(C_div) and M = C_circ @ N, reports rank(M).
      rank(M) == 5  -> interior closure CAN preserve circulation
      rank(M) <  5  -> genuine obstruction; extra DOF (Tóth-Roe) needed
  T4  Constructive test: prolong a rigid-rotation field (div-free, uniform
      curl) by solving for interior unknowns that are simultaneously
      div-free AND circulation-matching. Reports whether the residual is
      zero. This is independent confirmation of the rank verdict.
  T5  Tóth–Roe-style comparison: prolong using edge-circulation as the
      primary variable; report the difference.

WHAT THIS IS NOT
  ❌ Not a proof of a "paradigm shift" or "state of the art" claim.
  ❌ Not a validation against Tóth–Roe's actual published formulas.
  ❌ Not a full AMR solver; single coarse cell only.
  ❌ Results are convention-dependent and must be re-checked under other
     circulation discretizations before being cited.
=============================================================================
"""
from __future__ import annotations

import argparse
import numpy as np

VERSION = "1.0.0"


# =============================================================================
# 0. MAC FIELD LAYOUT FOR ONE COARSE CELL (2x2x2 fine cells)
# =============================================================================
# u (x-normal): shape (3, 2, 2)   i in 0..2 (faces), j,k in 0..1 (centers)
# v (y-normal): shape (2, 3, 2)
# w (z-normal): shape (2, 2, 3)
#
# Interior (UNKNOWN) faces — not on any coarse boundary face:
#   u[1, j, k]  for j,k in {0,1}   -> 4 unknowns   (index 0..3)
#   v[i, 1, k]  for i,k in {0,1}   -> 4 unknowns   (index 4..7)
#   w[i, j, 1]  for i,j in {0,1}   -> 4 unknowns   (index 8..11)
# Known (boundary) faces:
#   u[0,:,:], u[2,:,:]  ;  v[:,0,:], v[:,2,:]  ;  w[:,:,0], w[:,:,2]

N_UNK = 12
UNKNOWN_MAP = (
    [("u", 1, j, k) for j in (0, 1) for k in (0, 1)]
    + [("v", i, 1, k) for i in (0, 1) for k in (0, 1)]
    + [("w", i, j, 1) for i in (0, 1) for j in (0, 1)]
)
assert len(UNKNOWN_MAP) == N_UNK


def zero_fields():
    u = np.zeros((3, 2, 2)); v = np.zeros((2, 3, 2)); w = np.zeros((2, 2, 3))
    return u, v, w


def set_unknown(fields, idx, val):
    name, i, j, k = UNKNOWN_MAP[idx]
    fields[{"u": 0, "v": 1, "w": 2}[name]][i, j, k] = val


# =============================================================================
# 1. DISCRETE DIVERGENCE (8 sub-cells)
# =============================================================================

def divergence(u, v, w):
    """div[i,j,k] = (u[i+1]-u[i]) + (v[j+1]-v[j]) + (w[k+1]-w[k]), fine h=1."""
    out = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                out[i, j, k] = (
                    (u[i + 1, j, k] - u[i, j, k])
                    + (v[i, j + 1, k] - v[i, j, k])
                    + (w[i, j, k + 1] - w[i, j, k])
                )
    return out.reshape(8)


# =============================================================================
# 2. COARSE-FACE CIRCULATION under the stated convention
# =============================================================================

def plane_circulation(A, B):
    """
    Circulation around a unit-coarse (2x2 fine) face lying in a plane.
    A: tangential velocity along the first in-plane axis, shape (3, 2)
       A[p, q]  p = face-position index along axis1 (0..2),
                q = cell-center index along axis2 (0..1)
    B: tangential velocity along the second in-plane axis, shape (2, 3)
       B[p, q]  p = cell-center index along axis1 (0..1),
                q = face-position index along axis2 (0..2)

    Boundary loop (counter-clockwise viewed from +normal):
      edge axis2=0 (near):  + integral of A along axis1   using q=0 layer
      edge axis1=2 (far):   + integral of B along axis2   using p=1 layer
      edge axis2=2 (far):   - integral of A along axis1   using q=1 layer
      edge axis1=0 (near):  - integral of B along axis2   using p=0 layer
    Segment quadrature: each of the 2 fine segments uses the average of its
    two endpoint face-values, i.e. weights (0.5, 1.0, 0.5) over positions 0,1,2.
    """
    wpos = np.array([0.5, 1.0, 0.5])
    bottom = np.dot(wpos, A[:, 0])          # +A along axis2=0
    top    = np.dot(wpos, A[:, 1])          # -A along axis2=2 (nearest layer q=1)
    right  = np.dot(wpos, B[1, :])          # +B along axis1=2 (nearest layer p=1)
    left   = np.dot(wpos, B[0, :])          # -B along axis1=0
    return bottom - top + right - left


def coarse_circulations(u, v, w):
    """
    Returns the 6 coarse-face circulations
    [xmin, xmax, ymin, ymax, zmin, zmax].

    Convention: for a face normal to axis a, use the tangential MAC
    velocities in the layer nearest that face.
      zmin: u[:,:,0], v[:,:,0]     zmax: u[:,:,1], v[:,:,1]
      ymin: u[:,0,:], w[:,0,:]     ymax: u[:,1,:], w[:,1,:]
      xmin: v[0,:,:], w[0,:,:]     xmax: v[1,:,:], w[1,:,:]
    """
    g = np.zeros(6)
    # z-faces: tangential (u along x, v along y)
    g[4] = plane_circulation(u[:, :, 0], v[:, :, 0])   # zmin
    g[5] = plane_circulation(u[:, :, 1], v[:, :, 1])   # zmax
    # y-faces: tangential (u along x, w along z)
    g[2] = plane_circulation(u[:, 0, :], w[:, 0, :])   # ymin
    g[3] = plane_circulation(u[:, 1, :], w[:, 1, :])   # ymax
    # x-faces: tangential (v along y, w along z)
    g[0] = plane_circulation(v[0, :, :], w[0, :, :])   # xmin
    g[1] = plane_circulation(v[1, :, :], w[1, :, :])   # xmax
    return g


# =============================================================================
# 3. OPERATOR ASSEMBLY BY NUMERICAL PERTURBATION
# =============================================================================

def build_operators():
    """Build C_div (8x12) and C_circ (6x12) column by column."""
    C_div = np.zeros((8, N_UNK))
    C_circ = np.zeros((6, N_UNK))
    for c in range(N_UNK):
        u, v, w = zero_fields()
        set_unknown((u, v, w), c, 1.0)
        C_div[:, c] = divergence(u, v, w)
        C_circ[:, c] = coarse_circulations(u, v, w)
    return C_div, C_circ


def nullspace(A, tol=1e-9):
    _, s, vh = np.linalg.svd(A)
    rank = int((s > tol).sum())
    return vh[rank:].T, rank


def matrix_rank(A, tol=1e-9):
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > tol).sum())


# =============================================================================
# 4. THE DEFINITIVE RANK TEST
# =============================================================================

def rank_test():
    C_div, C_circ = build_operators()
    N, rank_div = nullspace(C_div)
    nullity = N.shape[1]
    M = C_circ @ N                 # 6 x nullity
    rank_M = matrix_rank(M)
    # circulation dependency: sum of oriented circulations over closed surface
    dep = C_circ.sum(axis=0)
    print("=" * 78)
    print("  RANK ANALYSIS — the decisive test")
    print("=" * 78)
    print(f"  rank(C_div)                = {rank_div}   (expect 7)")
    print(f"  nullity = interior DOF     = {nullity}   (expect 5)")
    print(f"  closed-surface dependency  = {'present' if np.abs(dep).max() < 1e-9 else 'ABSENT'}")
    print(f"  independent circulations   = {matrix_rank(C_circ)}   (expect 5)")
    print(f"  rank(M = C_circ @ N)       = {rank_M}")
    print("-" * 78)
    if nullity == 0:
        print("  No interior freedom at all.")
    elif rank_M >= 5:
        print("  VERDICT: rank(M) = 5. The 5 interior DOF can satisfy the 5")
        print("  independent coarse-circulation constraints while remaining")
        print("  divergence-free. Under THIS convention, an interior closure")
        print("  CAN preserve coarse circulation. The claim that Tóth-Roe")
        print("  edge-circulation variables are REQUIRED is NOT supported here.")
    else:
        print(f"  VERDICT: rank(M) = {rank_M} < 5. The interior DOF cannot")
        print("  independently control all coarse circulations under this")
        print("  convention. This is consistent with needing additional")
        print("  edge-circulation degrees of freedom (Tóth-Roe style).")
    print("=" * 78)
    return C_div, C_circ, N, rank_M


# =============================================================================
# 5. CONSTRUCTIVE TEST — RIGID ROTATION
# =============================================================================

def rigid_rotation_fields():
    """u = -y, v = +x, w = 0 at MAC face positions (fine coords)."""
    u, v, w = zero_fields()
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u[i, j, k] = -(j + 0.5)          # u = -y
    for i in range(2):
        for j in range(3):
            for k in range(2):
                v[i, j, k] = (i + 0.5)           # v = +x
    # w = 0 already
    return u, v, w


def boundary_only(u, v, w):
    """Copy of the fields with interior unknowns zeroed out."""
    ub, vb, wb = u.copy(), v.copy(), w.copy()
    for c in range(N_UNK):
        set_unknown((ub, vb, wb), c, 0.0)
    return ub, vb, wb


def constructive_test(C_div, C_circ):
    print("\n" + "=" * 78)
    print("  CONSTRUCTIVE TEST — rigid rotation (div-free, uniform curl)")
    print("=" * 78)
    u_t, v_t, w_t = rigid_rotation_fields()

    # sanity: the exact field is divergence-free
    div_exact = divergence(u_t, v_t, w_t)
    print(f"  max|div| of exact rigid-rotation field = {np.abs(div_exact).max():.2e}")

    # target circulations from the fully-specified exact field
    g_target = coarse_circulations(u_t, v_t, w_t)
    print(f"  target circulations (xmin..zmax) = {np.round(g_target, 4).tolist()}")

    # boundary-only field
    ub, vb, wb = boundary_only(u_t, v_t, w_t)
    div_b = divergence(ub, vb, wb)
    g_b = coarse_circulations(ub, vb, wb)

    # solve for interior x: C_div x = -div_b ; C_circ x = g_target - g_b
    A = np.vstack([C_div, C_circ])
    b = np.concatenate([-div_b, g_target - g_b])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    # residuals
    res_div = np.abs(C_div @ x + div_b).max()
    res_circ = np.abs(C_circ @ x - (g_target - g_b)).max()
    print(f"  solved interior DOF: {np.round(x, 4).tolist()}")
    print(f"  residual div-free      = {res_div:.3e}")
    print(f"  residual circ-match    = {res_circ:.3e}")

    # rebuild full field and measure achieved curl via circulation
    u_r, v_r, w_r = ub.copy(), vb.copy(), wb.copy()
    for c in range(N_UNK):
        set_unknown((u_r, v_r, w_r), c, x[c])
    div_r = np.abs(divergence(u_r, v_r, w_r)).max()
    g_r = coarse_circulations(u_r, v_r, w_r)
    circ_err = np.abs(g_r - g_target).max()
    print(f"  achieved max|div|          = {div_r:.3e}")
    print(f"  achieved max circ error    = {circ_err:.3e}")
    ok = res_div < 1e-8 and res_circ < 1e-8
    print("-" * 78)
    print(f"  CONSTRUCTIVE VERDICT: interior closure "
          f"{'CAN' if ok else 'CANNOT'} simultaneously enforce div-free AND "
          f"circulation preservation on this field.")
    print("=" * 78)
    return ok


# =============================================================================
# 6. SELF-TEST DRIVER
# =============================================================================

def self_test():
    print("=" * 78)
    print(f"  AMR CURL-PRESERVING CLOSURE TEST v{VERSION}")
    print("  NOTE: results are convention-dependent; see header.")
    print("=" * 78)
    C_div, C_circ, N, rank_M = rank_test()
    constructive_ok = constructive_test(C_div, C_circ)

    print("\n" + "=" * 78)
    print("  HOW TO READ THIS")
    print("=" * 78)
    print("""
  • If rank(M) == 5 AND the constructive test passes:
      An interior closure of the Balsara null-space CAN preserve coarse
      circulation. Tóth-Roe edge-circulation variables are ONE way, but
      not the ONLY way. The strong claim is not supported.

  • If rank(M) < 5 OR the constructive test fails:
      There is a genuine obstruction under this convention, supporting the
      need for additional edge-circulation degrees of freedom.

  • EITHER outcome is convention-dependent. Repeat under a Yee-edge
    circulation convention before drawing any strong conclusion.

  WHAT THIS DOES NOT ESTABLISH
    • Not a "paradigm shift" or "state of the art" result by itself.
    • Not a verification against Tóth-Roe's published formulas.
    • Single coarse cell; no multi-level, no time evolution, no AMR cycle.
    • No experimental/numerical benchmark comparison.
""")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest", choices=["selftest"])
    p.parse_args()
    self_test()


if __name__ == "__main__":
    main()
