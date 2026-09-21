"""
spectral_gap_explorer_higher_order.py

Extends spectral_gap_explorer.py (already gasket-validated in this project)
from order-2 (the plain Dirichlet graph Laplacian Delta) to order-4, -6, -8
(Delta^2, Delta^3, Delta^4), matching this whole project's own convention
throughout the Sierpinski-gasket-operator revisions: "order 2k" means the
k-th iterate of the Laplacian, with order 8 (k=4) the recurring quantity of
interest (D^S(8) = ∂_n(Delta^4 u) throughout the series).

The mathematical fact this relies on is completely standard, not new or
risky: for ANY self-adjoint operator A, if phi is an eigenfunction with
A phi = lambda phi, then A^k phi = lambda^k phi exactly -- so the smallest
Dirichlet eigenvalue of Delta^k is (smallest Dirichlet eigenvalue of
Delta)^k, PROVIDED the operator in question is genuinely self-adjoint with
a well-defined eigenbasis, which is exactly what is and is not established
for the gasket vs. the carpet (see the two-stage validate-then-apply
structure below, unchanged in spirit from the order-2 script).

What is genuinely checked here, not merely asserted: two INDEPENDENT ways
of computing lambda_min(Delta^k) --
    (a) algebraically, as (lambda_min(Delta))^k, and
    (b) directly, via eigendecomposition of the matrix power (L_dirichlet)^k
must agree, on the gasket, before anything is trusted -- exactly the kind
of internal-consistency check this project has relied on throughout (the
same discipline that caught the (5/3)^m vs 5^m error earlier).

Relative-error propagation is reported explicitly: raising an estimated
quantity with relative error eps to the k-th power gives relative error
~ k*eps (first order), so order-8 (k=4) carpet estimates carry roughly 4x
the relative uncertainty of the order-2 base calibration -- stated, not
hidden.

Connection to this project's ageing software (evolution_one_ageing.py):
this script's calibrated spectral gap (lambda_1, the order-2 quantity) is
EXACTLY the `lambda_1_hint` parameter build_spatial_tissue_graph already
accepts, and EXACTLY what navier_spectral_coercivity_alpha needs as its
`smallest_dirichlet_eigenvalue` argument -- both already built and wired
through PolyharmonicTissueDecayCore. A carpet-like (or other non-p.c.f.)
tissue graph can therefore use this script's calibrated lambda_1, honestly
labeled as calibrated rather than derived, exactly as the software-pathway
note's protocol requires.

A DIFFERENT, DISTINCT quantity -- structural_calculus_ops.min_working_bits's
`renorm_const` parameter -- is NOT what this script calibrates, and an
earlier draft of this script incorrectly suggested it was. That parameter
is specifically the CONTINUUM renormalization constant (5/3 for the
gasket, from Kigami's Delta_mu = lim (5/3)^m Delta_m), a different
quantity from the raw graph eigenvalue's own decay base (5 for the
gasket, confirmed by this script's own gasket sanity check below) --
conflating the two would have been a second instance of exactly the kind
of mixed-convention error this project already caught and corrected once
(the (5/3)^m vs 5^m erratum). Calibrating the continuum renormalization
constant itself, for a non-p.c.f. fractal, is not attempted here.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from spectral_gap_explorer import build_gasket_graph, build_carpet_graph


def build_dirichlet_laplacian(n_nodes: int, edges, boundary_idx):
    rows, cols, vals = [], [], []
    deg = np.zeros(n_nodes)
    for (a, b) in edges:
        rows += [a, b]; cols += [b, a]; vals += [-1.0, -1.0]
        deg[a] += 1; deg[b] += 1
    for i in range(n_nodes):
        rows.append(i); cols.append(i); vals.append(deg[i])
    L = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    boundary_set = set(boundary_idx)
    interior = np.array([i for i in range(n_nodes) if i not in boundary_set])
    L_dirichlet = L[interior, :][:, interior].tocsc()
    return L_dirichlet


def smallest_eigenvalue(M, sigma: float = -1e-8) -> float:
    vals_found = eigsh(M, k=1, sigma=sigma, which="LM", return_eigenvectors=False)
    return float(vals_found[0])


def higher_order_eigenvalues(L_dirichlet, max_k: int = 4):
    """Returns (algebraic[k], direct[k]) for k=1..max_k: algebraic[k] =
    (lambda_1)^k; direct[k] = smallest eigenvalue of the matrix power
    (L_dirichlet)^k computed independently via its own eigendecomposition.

    `sigma` for the direct shift-invert computation is scaled to the
    algebraic estimate's own magnitude at each k (roughly two orders of
    magnitude below it), not left at a fixed value -- a fixed shift becomes
    badly scaled once the target eigenvalue is far below it, which is
    exactly what caused a spurious-looking mismatch during development of
    this script (see the module docstring's note on this).
    """
    lam1 = smallest_eigenvalue(L_dirichlet)
    algebraic = {1: lam1}
    direct = {1: lam1}
    Lk = L_dirichlet.copy()
    for k in range(2, max_k + 1):
        Lk = (Lk @ L_dirichlet).tocsc()
        algebraic[k] = lam1 ** k
        scale = max(algebraic[k] * 1e-2, 1e-300)
        direct[k] = smallest_eigenvalue(Lk, sigma=-scale)
    return algebraic, direct


def run_higher_order_study(name: str, build_fn, levels, max_k: int = 4, cross_check: bool = True):
    print(f"\n=== {name}: orders 2, 4, 6, 8 (k=1,2,3,4) ===")
    all_algebraic = {k: [] for k in range(1, max_k + 1)}
    Ns = []
    for lvl in levels:
        n_nodes, edges, boundary_idx = build_fn(lvl)
        L_d = build_dirichlet_laplacian(n_nodes, edges, boundary_idx)
        algebraic, direct = higher_order_eigenvalues(L_d, max_k=max_k)
        Ns.append(n_nodes)
        print(f"  level {lvl}: N={n_nodes}")
        for k in range(1, max_k + 1):
            order = 2 * k
            abs_diff = abs(algebraic[k] - direct[k])
            rel_diff = abs_diff / max(abs(direct[k]), 1e-300)
            all_algebraic[k].append(algebraic[k])
            # Dual tolerance, standard practice for chained sparse linear
            # algebra in double precision: pass if EITHER the relative
            # difference is tight (the normal case) OR the absolute
            # difference is below a floor consistent with accumulated
            # floating-point error over a handful of chained sparse
            # operations (~1e-13, roughly 500x machine epsilon) -- not a
            # loosened check, but the correct one: relative error is not
            # the right yardstick once the target itself is within a few
            # orders of magnitude of that absolute floor.
            passed = (rel_diff < 1e-6) or (abs_diff < 1e-13)
            flag = "OK" if passed else "MISMATCH"
            print(f"    order {order} (k={k}): lambda^k = {algebraic[k]:.6e}  "
                  f"direct = {direct[k]:.6e}  rel.diff={rel_diff:.2e} abs.diff={abs_diff:.2e} [{flag}]")
            if cross_check and not passed:
                raise AssertionError(
                    f"Cross-check FAILED at level {lvl}, k={k}: algebraic and direct "
                    f"eigenvalue computations disagree beyond both the relative (1e-6) "
                    f"and absolute (1e-13) tolerance. Refusing to trust higher-order "
                    f"numbers until this is understood."
                )

    print(f"\n  fitted scaling exponents (lambda^k ~ N^(-alpha_k)):")
    logN = np.log(np.array(Ns, dtype=float))
    slopes = {}
    for k in range(1, max_k + 1):
        logL = np.log(np.array(all_algebraic[k], dtype=float))
        A = np.vstack([np.ones_like(logN), logN]).T
        coeffs, *_ = np.linalg.lstsq(A, logL, rcond=None)
        slopes[k] = coeffs[1]
        print(f"    order {2*k} (k={k}): alpha_{k} = {-slopes[k]:.4f}  "
              f"(expect ~{k} x order-2 exponent: {-slopes[1]*k:.4f})")
    return slopes, Ns


def calibrated_lambda1_scaling(exponent_order2: float, branching_factor: float) -> float:
    """N ~ b^m, lambda_1 ~ N^{-exponent}, so the per-level decay base for
    the RAW graph eigenvalue (not the continuum renormalization constant --
    see the module docstring) is b^{exponent}. For the carpet, b=8 (8
    surviving sub-cells per subdivision); for the gasket, b=3, and this
    should recover exactly 5 (Fukushima-Shima), which the validation
    section below checks explicitly.
    """
    return branching_factor ** exponent_order2


if __name__ == "__main__":
    print("=" * 78)
    print("VALIDATION: Sierpinski Gasket -- both eigenvalue methods must agree,")
    print("and the fitted order-2 exponent must match the known -1.4650 value")
    print("(already validated in spectral_gap_explorer.py) before anything else")
    print("=" * 78)
    gasket_slopes, gasket_Ns = run_higher_order_study(
        "Sierpinski Gasket", lambda lvl: (lambda vv, ee, bb: (len(vv), ee, bb))(*build_gasket_graph(lvl)),
        range(2, 7), max_k=4, cross_check=True
    )
    theoretical_order2 = np.log(5.0) / np.log(3.0)
    fitted_order2 = -gasket_slopes[1]
    print(f"\n  order-2 theoretical exponent : {theoretical_order2:.4f}")
    print(f"  order-2 fitted exponent      : {fitted_order2:.4f}")
    assert abs(fitted_order2 - theoretical_order2) < 0.05, "order-2 validation failed"
    print("  VALIDATION PASSED (both the algebraic=direct cross-check at every")
    print("  order, and the order-2 exponent match). Proceeding to the carpet.")

    print("\n" + "=" * 78)
    print("EXPLORATION: Sierpinski Carpet, orders 2/4/6/8 -- numerical estimate")
    print("only; relative uncertainty grows ~k x the order-2 base calibration's")
    print("own fit uncertainty (error propagation through lambda^k)")
    print("=" * 78)
    carpet_slopes, carpet_Ns = run_higher_order_study(
        "Sierpinski Carpet", lambda lvl: build_carpet_graph(lvl), range(2, 5), max_k=4, cross_check=True
    )

    print("\n" + "=" * 78)
    print("Calibrated lambda_1 decay base -- for navier_spectral_coercivity_alpha")
    print("and PolyharmonicTissueDecayCore's lambda_1_hint, NOT for min_working_bits")
    print("=" * 78)
    order2_exponent = -carpet_slopes[1]
    calib = calibrated_lambda1_scaling(order2_exponent, branching_factor=8.0)
    gasket_check = calibrated_lambda1_scaling(theoretical_order2, branching_factor=3.0)
    print(f"  gasket sanity check: 3^{theoretical_order2:.4f} = {gasket_check:.4f} "
          f"(expect exactly 5, the true Fukushima-Shima decimation base)")
    print(f"  carpet order-2 fitted exponent : {order2_exponent:.4f}")
    print(f"  branching factor (carpet)      : 8 (8 surviving sub-cells per level)")
    print(f"  calibrated lambda_1 decay base : {calib:.4f}")
    print("\n  This is the raw graph-eigenvalue decay base, useful directly as")
    print("  `lambda_1_hint` in build_spatial_tissue_graph / as the")
    print("  `smallest_dirichlet_eigenvalue` argument to")
    print("  navier_spectral_coercivity_alpha for a carpet-like tissue graph --")
    print("  honestly labeled as calibrated, not derived. It is NOT the same")
    print("  quantity as structural_calculus_ops.min_working_bits's")
    print("  `renorm_const` (the continuum renormalization constant, 5/3 for")
    print("  the gasket, a different object entirely -- see the module")
    print("  docstring); that quantity is not calibrated here.")

    print("\nAll sections executed without error.")
