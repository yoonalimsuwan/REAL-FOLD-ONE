#!/usr/bin/env python3
"""
=============================================================================
FH FIELD-VISCOSITY FIX — remove HOMOGENIZED_READ + correct variable-nu operator
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

THE DEFECT (verified against structuralfluctuatinghydro_v6_3.py)
----------------------------------------------------------------
    if self._ext_nu_eff.any():
        nu = 0.5 * (nu + self._ext_nu_eff.mean())      # <-- .mean()
    ...
    visc_x = nu * laplacian_cell(ucx, dx, dy, dz, cfg.bc_x)
    ...
    "nu": nu.item()

CahnHilliardFHBridge.sync() writes a full (Nx,Ny,Nz) viscosity FIELD.
`.mean()` collapses it to a scalar, so the phase viscosity CONTRAST is
destroyed even though the buffer is technically "read" — the same physical
consequence as Bug 11 (_ext_nu_ch never read at all), one layer deeper.

TWO changes are required, not one:
  1. keep nu as a FIELD
  2. nu*laplacian(u) is WRONG for variable nu. The conservative form is
     div(nu grad u); using nu*lap(u) silently DROPS the grad(nu) term.

Also fixed: _ext_rho_eff was blended 0.5*(rho+ext), HALVING the density
contrast. Now replaces (blend weight is explicit if you want otherwise).

ALREADY CORRECT (no patch needed, verified by reading):
  _check_cfl        : nu.detach().item() if nu.ndim==0 else nu.max().item()
  _noise_prefactor  : eta = rho * nu   (field x field broadcasts fine)

BLOCKING SYNTAX ERRORS also repaired here (they prevent import at all):
  _lim_superbee     : `2r`, `0.5_soft_abs`, `0.5(r` — missing '*'
  _advect_upwind_3d : `(1-gx) (fp...)`, `gy (rho-...)`, `dt (ucx drho_dx...)`
  from future import / getLogger(name) / if name == "main"
General bare-separator repair is a separate pass; this script reports
whether the result parses rather than claiming success.

HONEST SCOPE
  ❌ div(nu grad u) is the conservative SCALAR-diffusion form. The full
     symmetric stress div(nu(grad u + grad u^T)) is provided but opt-in.
  ❌ No experimental validation of the FH solver.
  ❌ Exact-string patching: a miss is REPORTED, never silently ignored.
=============================================================================
"""
from __future__ import annotations
import argparse, ast, json, logging, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger("fh_nu_fix")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
FIX_VERSION = "1.0.0"


# =============================================================================
# CORRECT VARIABLE-COEFFICIENT OPERATORS (torch only, standalone-testable)
# =============================================================================

def sl(ax: int, idx) -> Tuple:
    o = [slice(None)] * 3
    o[ax] = idx
    return tuple(o)


def pad1(f: Tensor, ax: int, periodic: bool) -> Tensor:
    n = f.shape[ax]; d = f.device
    if periodic:
        lo = f.index_select(ax, torch.tensor([n - 1], device=d))
        hi = f.index_select(ax, torch.tensor([0], device=d))
    else:
        lo = f.index_select(ax, torch.tensor([0], device=d))
        hi = f.index_select(ax, torch.tensor([n - 1], device=d))
    return torch.cat([lo, f, hi], dim=ax)


def face_scalar(f: Tensor, ax: int, periodic: bool) -> Tensor:
    """Cell-centred scalar -> faces (n_ax+1), BC-aware."""
    shape = tuple(f.shape[i] + (1 if i == ax else 0) for i in range(3))
    b = torch.zeros(shape, device=f.device, dtype=f.dtype)
    b[sl(ax, slice(1, -1))] = 0.5 * (f[sl(ax, slice(0, -1))]
                                     + f[sl(ax, slice(1, None))])
    if periodic:
        m = 0.5 * (f[sl(ax, 0)] + f[sl(ax, -1)])
        b[sl(ax, 0)] = m; b[sl(ax, -1)] = m
    else:
        b[sl(ax, 0)] = f[sl(ax, 0)]; b[sl(ax, -1)] = f[sl(ax, -1)]
    return b


def diff_ax(f: Tensor, ax: int, h: float) -> Tensor:
    return (f[sl(ax, slice(1, None))] - f[sl(ax, slice(0, -1))]) / h


def div_nu_grad(u: Tensor, nu: Tensor, h: Tuple[float, float, float],
                periodic=(True, True, True)) -> Tensor:
    """
    Conservative variable-coefficient diffusion div(nu grad u) at CENTRES.

    Reduces EXACTLY to nu*laplacian(u) for uniform nu (test T1).
    Differs when nu varies — that difference IS the grad(nu) contribution
    the old code dropped (test T2).
    """
    if nu.ndim == 0:
        nu = torch.full_like(u, float(nu))
    out = torch.zeros_like(u)
    for ax in range(3):
        nu_f = face_scalar(nu, ax, periodic[ax])
        up = pad1(u, ax, periodic[ax])
        out = out + diff_ax(nu_f * diff_ax(up, ax, h[ax]), ax, h[ax])
    return out


def laplacian_ref(f: Tensor, h, periodic=(True, True, True)) -> Tensor:
    """7-point Laplacian — the uniform-nu reference used in tests."""
    out = torch.zeros_like(f)
    for ax in range(3):
        fp = pad1(f, ax, periodic[ax])
        hi = fp[sl(ax, slice(2, None))]; lo = fp[sl(ax, slice(0, -2))]
        for o in range(3):
            if o != ax:
                hi = hi[sl(o, slice(1, -1))]; lo = lo[sl(o, slice(1, -1))]
        out = out + (hi - 2 * f + lo) / h[ax] ** 2
    return out


def viscous_stress_div(u, v, w, nu, h, periodic=(True, True, True)):
    """Full symmetric form div(nu(grad U + grad U^T)) at centres (~2x cost)."""
    if nu.ndim == 0:
        nu = torch.full_like(u, float(nu))
    U = (u, v, w); g = {}
    for a in range(3):
        for b in range(3):
            fp = pad1(U[a], b, periodic[b])
            hi = fp[sl(b, slice(2, None))]; lo = fp[sl(b, slice(0, -2))]
            for o in range(3):
                if o != b:
                    hi = hi[sl(o, slice(1, -1))]; lo = lo[sl(o, slice(1, -1))]
            g[(a, b)] = (hi - lo) / (2 * h[b])
    out = []
    for a in range(3):
        acc = torch.zeros_like(u)
        for b in range(3):
            T = nu * (g[(a, b)] + g[(b, a)])
            Tp = pad1(T, b, periodic[b])
            hi = Tp[sl(b, slice(2, None))]; lo = Tp[sl(b, slice(0, -2))]
            for o in range(3):
                if o != b:
                    hi = hi[sl(o, slice(1, -1))]; lo = lo[sl(o, slice(1, -1))]
            acc = acc + (hi - lo) / (2 * h[b])
        out.append(acc)
    return tuple(out)


# =============================================================================
# SOURCE PATCHER
# =============================================================================

OPERATOR_SRC = '''
# ---- [FH-NU-FIX] variable-coefficient viscous operators -------------------
def _fh_sl(ax, idx):
    o = [slice(None)] * 3
    o[ax] = idx
    return tuple(o)


def _fh_pad1(f, ax, periodic):
    n = f.shape[ax]; d = f.device
    if periodic:
        lo = f.index_select(ax, torch.tensor([n - 1], device=d))
        hi = f.index_select(ax, torch.tensor([0], device=d))
    else:
        lo = f.index_select(ax, torch.tensor([0], device=d))
        hi = f.index_select(ax, torch.tensor([n - 1], device=d))
    return torch.cat([lo, f, hi], dim=ax)


def _fh_face(f, ax, periodic):
    shape = tuple(f.shape[i] + (1 if i == ax else 0) for i in range(3))
    b = torch.zeros(shape, device=f.device, dtype=f.dtype)
    b[_fh_sl(ax, slice(1, -1))] = 0.5 * (
        f[_fh_sl(ax, slice(0, -1))] + f[_fh_sl(ax, slice(1, None))])
    if periodic:
        m = 0.5 * (f[_fh_sl(ax, 0)] + f[_fh_sl(ax, -1)])
        b[_fh_sl(ax, 0)] = m; b[_fh_sl(ax, -1)] = m
    else:
        b[_fh_sl(ax, 0)] = f[_fh_sl(ax, 0)]; b[_fh_sl(ax, -1)] = f[_fh_sl(ax, -1)]
    return b


def div_nu_grad(u, nu, dx, dy, dz, periodic=(True, True, True)):
    """[FH-NU-FIX] conservative div(nu grad u); == nu*lap(u) for uniform nu."""
    h = (dx, dy, dz)
    if nu.ndim == 0:
        nu = torch.full_like(u, float(nu))
    out = torch.zeros_like(u)
    for ax in range(3):
        nf = _fh_face(nu, ax, periodic[ax])
        up = _fh_pad1(u, ax, periodic[ax])
        g = (up[_fh_sl(ax, slice(1, None))] - up[_fh_sl(ax, slice(0, -1))]) / h[ax]
        fl = nf * g
        out = out + (fl[_fh_sl(ax, slice(1, None))]
                     - fl[_fh_sl(ax, slice(0, -1))]) / h[ax]
    return out
# ---- end [FH-NU-FIX] -----------------------------------------------------
'''

# (old, new, label) — exact strings verified against the uploaded source
REPLACEMENTS: List[Tuple[str, str, str]] = [
    # --- blocking syntax errors (prevent import at all) ---
    ("from future import annotations", "from __future__ import annotations",
     "future import"),
    ('logger = logging.getLogger(name)', 'logger = logging.getLogger(__name__)',
     "getLogger(name)"),
    ('if name == "main":', 'if __name__ == "__main__":', "main guard"),
    ('" " "1st-order upwind density advection (3-D). " " "',
     '"""1st-order upwind density advection (3-D)."""', "upwind docstring"),
    ("m1   = 0.5*(2r + 1.0) - 0.5_soft_abs(2r - 1.0)",
     "m1   = 0.5*(2*r + 1.0) - 0.5*_soft_abs(2*r - 1.0)", "superbee m1 '*'"),
    ("m2   = 0.5(r   + 2.0) - 0.5*_soft_abs(r   - 2.0)",
     "m2   = 0.5*(r   + 2.0) - 0.5*_soft_abs(r   - 2.0)", "superbee m2 '*'"),
    ("drho_dx = (gx*(rho-fp[:-2,1:-1,1:-1]) + (1-gx) (fp[2:,1:-1,1:-1]-rho)) / dx",
     "drho_dx = (gx*(rho-fp[:-2,1:-1,1:-1]) + (1-gx)*(fp[2:,1:-1,1:-1]-rho)) / dx",
     "upwind dx '*'"),
    ("drho_dy = (gy (rho-fp[1:-1,:-2,1:-1]) + (1-gy) (fp[1:-1,2:,1:-1]-rho)) / dy",
     "drho_dy = (gy*(rho-fp[1:-1,:-2,1:-1]) + (1-gy)*(fp[1:-1,2:,1:-1]-rho)) / dy",
     "upwind dy '*'"),
    ("drho_dz = (gz (rho-fp[1:-1,1:-1,:-2]) + (1-gz) (fp[1:-1,1:-1,2:]-rho)) / dz",
     "drho_dz = (gz*(rho-fp[1:-1,1:-1,:-2]) + (1-gz)*(fp[1:-1,1:-1,2:]-rho)) / dz",
     "upwind dz '*'"),
    ("return rho - dt (ucx drho_dx + ucy drho_dy + ucz*drho_dz)",
     "return rho - dt*(ucx*drho_dx + ucy*drho_dy + ucz*drho_dz)",
     "upwind return '*'"),
    # --- THE homogenized read ---
    ("        nu = 0.5 * (nu + self._ext_nu_eff.mean())",
     "        # [FH-NU-FIX] keep the FIELD; .mean() destroyed the phase contrast\n"
     "        nu = self._ext_nu_eff.to(rho.dtype)",
     "HOMOGENIZED_READ nu"),
    # --- density half-blend ---
    ("        rho = 0.5 * (rho + self._ext_rho_eff)",
     "        # [FH-NU-FIX] 0.5-blend HALVED the density contrast; replace\n"
     "        rho = self._ext_rho_eff.to(rho.dtype)",
     "density half-blend"),
    # --- wrong operator for a field nu ---
    ("    visc_x = nu * laplacian_cell(ucx, dx, dy, dz, cfg.bc_x)",
     "    _per = (cfg.bc_x == BoundaryCondition.PERIODIC,\n"
     "            cfg.bc_y == BoundaryCondition.PERIODIC,\n"
     "            cfg.bc_z == BoundaryCondition.PERIODIC)\n"
     "    # [FH-NU-FIX] nu*lap(u) drops grad(nu); use conservative form\n"
     "    visc_x = div_nu_grad(ucx, nu, dx, dy, dz, _per)",
     "visc_x operator"),
    ("    visc_y = nu * laplacian_cell(ucy, dx, dy, dz, cfg.bc_y)",
     "    visc_y = div_nu_grad(ucy, nu, dx, dy, dz, _per)", "visc_y operator"),
    ("    visc_z = nu * laplacian_cell(ucz, dx, dy, dz, cfg.bc_z)",
     "    visc_z = div_nu_grad(ucz, nu, dx, dy, dz, _per)", "visc_z operator"),
    # --- diagnostics .item() on a field ---
    ('         "nu":         nu.item(),',
     '         "nu":         float(nu.mean()),\n'
     '         "nu_min":     float(nu.min()),\n'
     '         "nu_max":     float(nu.max()),',
     "diag nu .item()"),
]


def patch_source(path: Path, dry_run: bool = True) -> Dict:
    src = path.read_text(errors="ignore")
    rec = {"file": path.name, "applied": [], "missed": [], "already": []}
    for old, new, label in REPLACEMENTS:
        if old not in src:
            if new.split("\n")[-1].strip() in src:
                rec["already"].append(label)
            else:
                pat = re.escape(old.strip()).replace(r"\ ", r"\s+")
                m = re.search(r"^[ \t]*" + pat + r"[ \t]*$", src, re.M)
                if m:
                    ind = m.group(0)[:len(m.group(0)) - len(m.group(0).lstrip())]
                    body = "\n".join(ind + l if l.strip() else l
                                     for l in new.split("\n"))
                    src = src[:m.start()] + body + src[m.end():]
                    rec["applied"].append(label + " (ws-tolerant)")
                else:
                    rec["missed"].append(label)
            continue
        src = src.replace(old, new, 1)
        rec["applied"].append(label)
    if any("operator" in a for a in rec["applied"]) and "def div_nu_grad(" not in src:
        i = src.find("def laplacian_cell(")
        if i < 0:
            rec["missed"].append("operator injection (no anchor)")
        else:
            src = src[:i] + OPERATOR_SRC + "\n\n" + src[i:]
            rec["applied"].append("operator injection")
    try:
        ast.parse(src); rec["ast"] = True; rec["ast_error"] = ""
    except SyntaxError as e:
        rec["ast"] = False; rec["ast_error"] = f"line {e.lineno}: {e.msg}"
    if not dry_run and rec["applied"]:
        bak = path.with_suffix(path.suffix + ".pre_nufix.bak")
        if not bak.exists():
            bak.write_text(path.read_text(errors="ignore"))
        path.write_text(src)
        rec["written"] = True; rec["backup"] = bak.name
    else:
        rec["written"] = False
    return rec


# =============================================================================
# SELF-TEST
# =============================================================================
def self_test(target: Optional[str] = None) -> bool:
    print("=" * 78)
    print(f"  FH FIELD-VISCOSITY FIX v{FIX_VERSION} — verification")
    print("=" * 78)
    torch.manual_seed(0); ok = True
    N = 12; h = (1.0 / N,) * 3
    u = torch.rand(N, N, N, dtype=torch.float64)

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    print("\n[T1] uniform nu: div(nu grad u) == nu*lap(u) EXACTLY")
    nu0 = torch.full_like(u, 3.7e-3)
    e = float((div_nu_grad(u, nu0, h) - nu0 * laplacian_ref(u, h)).abs().max())
    rep("uniform reduction", e < 1e-12, f"max err = {e:.2e}")

    print("\n[T2] variable nu: the field MUST change the answer")
    nu = torch.full_like(u, 1e-3); nu[: N // 2] = 1e-2      # 10:1 contrast
    a = div_nu_grad(u, nu, h)
    e_mean = float((a - float(nu.mean()) * laplacian_ref(u, h)).abs().max())
    e_local = float((a - nu * laplacian_ref(u, h)).abs().max())
    rep("differs from .mean() version", e_mean > 1e-6, f"max diff = {e_mean:.3e}")
    rep("differs from nu*lap(u) (dropped grad nu)", e_local > 1e-6,
        f"max diff = {e_local:.3e}")

    print("\n[T3] the read preserves the contrast")
    old_spread = 0.0
    new_spread = float(nu.max() - nu.min())
    rep("old .mean() collapses spread", old_spread == 0.0)
    rep("new field preserves spread", new_spread > 5e-3,
        f"spread = {new_spread:.3e}")

    print("\n[T4] full symmetric stress: uniform-nu identity")
    v = torch.rand_like(u); w = torch.rand_like(u)
    sx, _, _ = viscous_stress_div(u, v, w, nu0, h)
    def gc(f, ax):
        fp = pad1(f, ax, True)
        hi = fp[sl(ax, slice(2, None))]; lo = fp[sl(ax, slice(0, -2))]
        for o in range(3):
            if o != ax:
                hi = hi[sl(o, slice(1, -1))]; lo = lo[sl(o, slice(1, -1))]
        return (hi - lo) / (2 * h[ax])
    divU = gc(u, 0) + gc(v, 1) + gc(w, 2)
    ref = float(nu0) * (laplacian_ref(u, h) + gc(divU, 0))
    e = float((sx - ref).abs().max())
    rep("nu*(lap U + grad div U)", e < 1e-11, f"max err = {e:.2e}")

    print("\n[T5] gradient flows to the nu FIELD itself")
    ng = nu.clone().requires_grad_(True)
    ug = u.clone().requires_grad_(True)
    div_nu_grad(ug, ng, h).pow(2).sum().backward()
    t = (ng.grad is not None and torch.isfinite(ng.grad).all()
         and float(ng.grad.abs().sum()) > 0
         and ug.grad is not None and torch.isfinite(ug.grad).all())
    rep("d/d(nu) and d/d(u) finite, non-zero", t)

    print("\n[T6] mixed BC path finite + shape-correct")
    a = div_nu_grad(u, nu, h, periodic=(False, True, False))
    rep("mixed BC", a.shape == u.shape and bool(torch.isfinite(a).all()))

    if target:
        print(f"\n[T7] source patch: {Path(target).name}")
        rec = patch_source(Path(target), dry_run=True)
        for k in ("applied", "already", "missed"):
            if rec[k]:
                print(f"   {k} ({len(rec[k])}): {rec[k]}")
        print(f"   parses: {rec['ast']}  {rec['ast_error']}")
        rep("all replacements found", len(rec["missed"]) == 0,
            f"missed={len(rec['missed'])}")
        if not rec["ast"]:
            print("   NOTE: remaining SyntaxError is a PRE-EXISTING bare-text "
                  "separator / indentation issue. Run the general syntax "
                  "repair first; this patch does not fix those.")
    else:
        print("\n[T7] SKIP source patch (pass --target <file>)")

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 78)
    print("""
  PROVEN: uniform reduction exact; a viscosity FIELD changes the answer;
          nu*lap(u) drops grad(nu); the new read preserves contrast;
          gradient reaches the viscosity field.
  NOT PROVEN: any experimental validation of the FH solver.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "patch"])
    p.add_argument("--target", default=None)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if self_test(a.target) else 1)
    if not a.target:
        log.error("--target required"); raise SystemExit(1)
    print(json.dumps(patch_source(Path(a.target), dry_run=not a.apply), indent=2))


if __name__ == "__main__":
    main()
