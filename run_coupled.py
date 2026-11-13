#!/usr/bin/env python3
"""
=============================================================================
RUN COUPLED — all five ONE-Ecosystem modules + the rewritten VOF solver
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

COUPLING GRAPH
  structural_interface_utils ──> structural_langevin (InterfaceDetector)
  one_core ──> SSC / CSOCBase / bridges ──> all solvers
  structural_langevin ──LangevinFHBridge──> structuralfluctuatinghydro
  structural_langevin ──LangevinDNSBridge──> super_dns
  structural_cahn_hilliard ──CahnHilliardDNSBridge──> super_dns
  sesi_multiphase_flow (VOF) ── shares one_cfd_stencils with FH

Every stage reports availability honestly. A missing module degrades that
stage to SKIP, it does not silently produce a no-op result.
=============================================================================
"""
from __future__ import annotations
import argparse, importlib, json, logging, math, sys
from pathlib import Path
from typing import Any, Dict

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("coupled")


def try_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="grid size per axis")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    N, NS = a.n, a.steps
    dev = torch.device(a.device)
    torch.manual_seed(0)

    report: Dict[str, Any] = {"grid": N, "steps": NS, "stages": {}}
    print("=" * 78)
    print("  RUN COUPLED — ONE Ecosystem + VOF")
    print("=" * 78)

    # ---- 0. one_core ----
    core, err = try_import("one_core")
    if core is None:
        print(f"\n[0] one_core: UNAVAILABLE — {err}")
        print("    cp one_core_v3-6.py one_core.py   then re-run")
        report["stages"]["one_core"] = {"available": False, "error": err}
    else:
        print(f"\n[0] one_core v{core.ONE_VERSION}: OK")
        report["stages"]["one_core"] = {"available": True,
                                        "version": core.ONE_VERSION}

    # ---- 1. interface utils ----
    iu, err = try_import("structural_interface_utils")
    if iu is None:
        print(f"[1] interface_utils: SKIP — {err}")
        report["stages"]["interface_utils"] = {"available": False}
    else:
        coords = torch.randn(200, 3) * 10.0
        sc = iu.soft_interface_score(coords, 8.0, 4.0, mode="auto")
        print(f"[1] interface_utils: OK  score shape {tuple(sc.shape)} "
              f"range [{float(sc.min()):.3f}, {float(sc.max()):.3f}]")
        report["stages"]["interface_utils"] = {"available": True,
                                               "n": int(sc.numel())}

    # ---- 2. stencils ----
    st_mod, err = try_import("one_cfd_stencils")
    if st_mod is None:
        print(f"[2] one_cfd_stencils: MISSING — {err}")
        report["stages"]["stencils"] = {"available": False}
        raise SystemExit(1)
    spec = st_mod.GridSpec(nx=N, ny=N, nz=N, dtype="float64",
                           device=str(dev))
    ops = st_mod.MACOps(spec)
    print(f"[2] one_cfd_stencils v{st_mod.STENCIL_VERSION}: OK")
    report["stages"]["stencils"] = {"available": True,
                                    "version": st_mod.STENCIL_VERSION}

    # ---- 3. Langevin ----
    lang_mod, err = try_import("structural_langevin_v3_2")
    if lang_mod is None:
        lang_mod, err = try_import("structural_langevin_v3-2")
    lang = None
    if lang_mod is None:
        print(f"[3] Langevin: SKIP — {err}")
        report["stages"]["langevin"] = {"available": False}
    else:
        lang = lang_mod.AdvancedStructuralLangevin(mass=1.0, dt=0.002,
                                                   base_temp=300.0)
        coords = torch.randn(20, 3, requires_grad=True) * 5.0
        vel = torch.randn(20, 3) * 0.5

        def harmonic(x):
            E = 0.5 * (x ** 2).sum()
            return E, -torch.autograd.grad(E, x, retain_graph=True)[0]

        x, v, T, sig = lang.full_step(coords, vel, harmonic)
        print(f"[3] Langevin BAOAB: OK  T={T:.2f} K  sigma={sig:.4f}")
        report["stages"]["langevin"] = {"available": True, "T": T, "sigma": sig}
        coords, vel = x.detach(), v.detach()

    # ---- 4. FH ----
    fh_mod, err = try_import("structuralfluctuatinghydro_v6_3")
    fh = None
    if fh_mod is None:
        print(f"[4] FH: SKIP — {err}")
        report["stages"]["fh"] = {"available": False}
    else:
        cfg = fh_mod.SolverConfig(Nx=N, Ny=N, Nz=N, dt=1e-4,
                                  enable_fluctuations=False, cfl_limit=0.0,
                                  dtype=torch.float64, advection_scheme="upwind")
        fh = fh_mod.StructuralFluctuatingHydro(cfg)
        rho, ux, uy, uz, p = fh.initialize_taylor_green()
        if lang is not None and core is not None:
            br = core.LangevinFHBridge(lang, fh)
            br.sync(coords, vel)
            print("    LangevinFHBridge.sync -> _ext_sigma, _ext_mask")
        for _ in range(NS):
            rho, ux, uy, uz, p, diag = fh.step(rho, ux, uy, uz, p)
        print(f"[4] FH: OK  div_max={diag['div_max']:.3e} KE={diag['ke']:.4e}")
        report["stages"]["fh"] = {"available": True,
                                  "div_max": diag["div_max"], "ke": diag["ke"]}
        # cross-check stencils vs FH
        pp = torch.rand(spec.cc_shape, dtype=torch.float64)
        gx, gy, gz = ops.grad_cc_to_faces(pp)
        fx, fy, fz = fh_mod.grad_p(pp, spec.dx, spec.dy, spec.dz)
        e = float((gx[1:-1] - fx[1:-1]).abs().max())
        print(f"    stencil vs FH grad interior diff = {e:.2e}")
        report["stages"]["stencil_vs_fh_grad_diff"] = e

    # ---- 5. compressible DNS ----
    dns_mod, err = try_import("super_dns_one_v6_3_8")
    if dns_mod is None:
        dns_mod, err = try_import("super_dns_one_v6_3-8")
    dns = None
    if dns_mod is None:
        print(f"[5] super_dns: SKIP — {err}")
        report["stages"]["dns"] = {"available": False}
    else:
        try:
            cfg = dns_mod.CFDConfig(nx=N, ny=N, nz=N, steps=NS, device=a.device,
                                    dtype=torch.float64, flux_scheme="ausm",
                                    advection_scheme="muscl",
                                    muscl_limiter="van_leer", cfl=0.3, Re=100.0)
            dns = dns_mod.CompressibleSolver(cfg)
            dns.initialize("taylor_green")
            if lang is not None and core is not None:
                core.LangevinDNSBridge(lang, dns).sync(coords)
                print("    LangevinDNSBridge.sync -> _ext_sigma")
            for _ in range(NS):
                dns.step(dt=1e-4)
            ke = float(dns.kinetic_energy())
            print(f"[5] super_dns: OK  KE={ke:.6f}")
            report["stages"]["dns"] = {"available": True, "ke": ke}
        except Exception as ex:
            print(f"[5] super_dns: FAIL — {type(ex).__name__}: {ex}")
            report["stages"]["dns"] = {"available": True, "error": str(ex)}

    # ---- 6. Cahn-Hilliard -> DNS ----
    ch_mod, err = try_import("structural_cahn_hilliard_3d_v3_2")
    if ch_mod is None:
        ch_mod, err = try_import("structural_cahn_hilliard_3d_v3-2")
    if ch_mod is None or dns is None or core is None:
        print(f"[6] CH->DNS: SKIP")
        report["stages"]["ch_to_dns"] = {"available": False}
    else:
        try:
            chcfg = ch_mod.CahnHilliardConfig(dx=spec.dx, dt=1e-6,
                                              laplacian="conv3d",
                                              device=a.device)
            ch = ch_mod.StructuralCahnHilliard3D(chcfg)
            u = torch.rand(N, N, N, dtype=torch.float64, device=dev) * 0.2 - 0.1
            br = core.CahnHilliardDNSBridge(ch, dns, korteweg_strength=1e-4)
            u = ch.step(u)
            br.sync(u)
            fx = float(dns._ext_fx.abs().max())
            print(f"[6] CH->DNS: OK  max|f_korteweg|={fx:.3e}")
            report["stages"]["ch_to_dns"] = {"available": True, "max_force": fx}
        except Exception as ex:
            print(f"[6] CH->DNS: FAIL — {type(ex).__name__}: {ex}")
            report["stages"]["ch_to_dns"] = {"available": True, "error": str(ex)}

    # ---- 7. VOF on the shared core ----
    vof_mod, err = try_import("sesi_multiphase_flow")
    if vof_mod is None:
        print(f"[7] VOF: SKIP — {err}")
        report["stages"]["vof"] = {"available": False}
    else:
        cfg = vof_mod.VOFConfig(nx=N, ny=N, nz=N, bc=("periodic",) * 3,
                                gravity=(0, 0, 0), sigma=0.0,
                                vof_scheme="upwind", dtype="float64",
                                device=a.device, poisson_tol=1e-10)
        s = vof_mod.MultiphaseNavierStokes3D(cfg)
        stt = s.initial_state(alpha_fn=lambda X, Y, Z: (
            (X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2 < 0.04).double())
        stt.u += 0.5
        m0 = float(stt.alpha.sum())
        for _ in range(NS):
            stt = s.step(stt)
        d = s.diagnostics(stt)
        rel = abs(d["alpha_mass"] - m0) / max(m0, 1e-30)
        print(f"[7] VOF v{vof_mod.VOF_VERSION}: OK  mass rel={rel:.2e} "
              f"div={d['max_divergence']:.2e} cg={d['poisson_iters']}")
        report["stages"]["vof"] = {"available": True, "mass_rel": rel,
                                   "div_max": d["max_divergence"],
                                   "poisson_converged": d["poisson_converged"]}

    Path("coupled_report.json").write_text(json.dumps(report, indent=2))
    n_ok = sum(1 for v in report["stages"].values()
               if isinstance(v, dict) and v.get("available") and "error" not in v)
    print("\n" + "=" * 78)
    print(f"  stages OK: {n_ok}/{len(report['stages'])}")
    print("  report: coupled_report.json")
    print("=" * 78)
    print("""
  HONEST NOTES
    • A SKIP means the module was not importable, NOT that coupling worked.
    • Bridges write buffers; whether the solver CONSUMES them is verified
      separately (one_core's Bug-11 history is exactly this failure mode).
    • Nothing here is trained or validated against experiment.
""")


if __name__ == "__main__":
    main()
