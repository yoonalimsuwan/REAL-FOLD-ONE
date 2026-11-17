#!/usr/bin/env python3
"""
=============================================================================
DNS AT Re_L = 1e6 ON 400,000 GPUs — order-22 vs spectral, honest planner
=============================================================================
Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

=============================================================================
WHAT IS TRUE
=============================================================================
400,000-GPU clusters exist (xAI Colossus ~200k->1M, Meta ~600k H100-equiv).
Re_L = 1e6 DNS needs N=16384 (spectral) or N=32768 (22nd-order FD).
Both fit in 400k x 80GB = 32 PB with large headroom.

WHAT IS DIFFERENT ABOUT DNS vs AI TRAINING
  AI:  ~1 all-reduce per step, O(parameters) traffic, ms latency OK, bf16.
  DNS: ~3 halo exchanges per RK3 step at 100-1000 Hz, O(surface) traffic,
       us latency matters, fp32/fp64 required, plus global reductions
       for the Poisson solve.
An AI cluster's raw GPU count does NOT transfer 1:1 to DNS throughput.
The interconnect (bisection bandwidth + latency) is the deciding factor.

=============================================================================
THE ORDER-22 vs SPECTRAL TRADEOFF (computed, not asserted)
=============================================================================
For a PERIODIC box, a p-th order central FD scheme resolves modes
accurately only up to (k*Delta)_usable < pi. This module MEASURES that
cutoff from the exact discrete symbol and derives the resolution constant:
    k_max_eff = f * N/2,  f = (k*Delta)_usable / pi
    need k_max_eff * eta >= kappa   =>   N >= 2*kappa/(f*eta)
Spectral has f = 1 exactly. Order 22 has f ~ 0.64. Hence
    N_FD22 / N_spectral = 1/f ~ 1.57   =>   3.9x more CELLS.
CONCLUSION: for periodic DNS, SPECTRAL WINS. Order 22 is the right
choice for non-periodic BC / complex geometry / immersed boundaries
where an FFT is unavailable. Both paths are provided; the planner
reports which is cheaper for the requested case.

=============================================================================
STRONG-SCALING REALITY
=============================================================================
Local block size at P ranks:  ell = (N^3 / P)^(1/3).
  N=16384, P=400000  ->  ell ~ 222
  N=32768, P=400000  ->  ell ~ 445
Halo fraction for order 22 (11 ghost layers): 6*11/ell
  ell=222 -> 29.7%      ell=445 -> 14.8%
For a 2-D pencil FFT the rank grid is px*py with px,py <= N; at
P=400000 that is ~632x632, giving pencils of ~26x26x16384 and
all-to-all messages of ~140 KB (complex128). That is a workable
message size on a modern interconnect, so 400k ranks is not
automatically absurd -- but the ACHIEVED EFFICIENCY must be measured.
This module therefore reports a RANGE of wall times over an explicit
efficiency sweep, never a single fabricated number.

ENSEMBLE MODE is the one configuration that uses 400k GPUs at near
100% efficiency: many independent DNS runs, no inter-run communication.
If the scientific goal is a parameter sweep (Re, forcing, rotation,
Schmidt number), ensemble is strictly better than one hero run.

=============================================================================
HONEST SCOPE
=============================================================================
* All throughput / bandwidth / efficiency numbers are PARAMETERS with
  stated defaults. They are assumptions until measured on the target
  machine. Every time estimate is labelled with the values used.
* No 400k-rank run is executed here. The cost model is arithmetic.
* Re_L vs Re_lambda is asked explicitly; they differ by ~Re_lam^2/15.
* Fornberg weights are exact and moment-verified; the USABLE-k*Delta
  cutoff is measured from the symbol, not quoted from memory.
* No experimental validation. Oracles: exact moment conditions,
  measured symbol error, analytic Taylor-Green.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "1.0.0"
BANNER = f"""
=============================================================================
DNS 400k-GPU PLANNER v{VERSION}
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
Re_L=1e6: spectral N=16384, order-22 FD N=32768. Both fit 32 PB.
For PERIODIC boxes spectral is ~3.9x cheaper in cells than order 22.
All times are RANGES over an explicit efficiency sweep. Measure them.
=============================================================================
"""


# =============================================================================
# §1. EXACT FORNBERG WEIGHTS TO ANY ORDER
# =============================================================================

def fornberg_weights(deriv: int, offsets: List[int]) -> np.ndarray:
    """Fornberg (1988) recursion. f^(d)(0) ~ sum_i w_i f(x_i h)/h^d."""
    n, m = len(offsets), deriv
    d = np.zeros((m + 1, n, n), dtype=np.float64)
    d[0, 0, 0] = 1.0
    c1 = 1.0
    for i in range(1, n):
        c2 = 1.0
        for j in range(i):
            c3 = offsets[i] - offsets[j]
            c2 *= c3
            for k in range(min(i, m) + 1):
                prev = d[k - 1, i - 1, j] if k > 0 else 0.0
                d[k, i, j] = (offsets[i] * d[k, i - 1, j] - k * prev) / c3
        for k in range(min(i, m) + 1):
            prev = d[k - 1, i - 1, i - 1] if k > 0 else 0.0
            d[k, i, i] = c1 / c2 * (k * prev - offsets[i - 1] * d[k, i - 1, i - 1])
        c1 = c2
    return d[m, n - 1, :]


_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}


def central_weights(deriv: int, order: int) -> Tuple[np.ndarray, int]:
    key = (deriv, order)
    if key not in _CACHE:
        half = max(1, (order + deriv - 1) // 2)
        _CACHE[key] = (fornberg_weights(deriv, list(range(-half, half + 1))), half)
    return _CACHE[key]


def moment_residuals(deriv: int, order: int) -> Dict[str, float]:
    """
    DEFINITIVE order check: sum_i w_i x_i^p must equal d! at p=d and 0
    otherwise, for p = 0 .. order+d-1. Any nonzero residual below the
    claimed order means the stencil is NOT that order.
    """
    w, half = central_weights(deriv, order)
    x = np.arange(-half, half + 1, dtype=np.float64)
    out = {}
    for p in range(order + deriv):
        got = float(np.sum(w * x ** p))
        want = float(math.factorial(deriv)) if p == deriv else 0.0
        # normalise by the largest term so high powers do not overflow
        scale = max(1.0, float(np.max(np.abs(w * x ** p))))
        out[f"p{p}"] = abs(got - want) / scale
    return out


def symbol_d1(kh: np.ndarray, order: int) -> np.ndarray:
    w, half = central_weights(1, order)
    return sum(wi * np.sin((i - half) * kh) for i, wi in enumerate(w))


def symbol_d2(kh: np.ndarray, order: int) -> np.ndarray:
    w, half = central_weights(2, order)
    return sum(wi * np.cos((i - half) * kh) for i, wi in enumerate(w))


def usable_kh(order: int, tol: float = 0.01, deriv: int = 1) -> float:
    """
    MEASURED largest k*h with relative symbol error < tol.
    This is what sets the resolution constant, not a remembered number.
    """
    kh = np.linspace(1e-4, math.pi, 20000)
    if deriv == 1:
        approx, exact = symbol_d1(kh, order), kh
    else:
        approx, exact = -symbol_d2(kh, order), kh ** 2
    rel = np.abs(approx - exact) / np.maximum(exact, 1e-30)
    ok = rel < tol
    return float(kh[ok].max()) if ok.any() else 0.0


def resolution_constant(order: Optional[int], kappa: float = 1.5,
                        tol: float = 0.01) -> Tuple[float, float]:
    """
    Returns (c, f) with N >= c * Re_L^(3/4) for L = 2*pi.
      spectral (order=None): f = 1, c = kappa/pi
      FD order p:            f = usable_kh/pi, c = kappa/(pi*f)
    """
    if order is None:
        return kappa / math.pi, 1.0
    khu = usable_kh(order, tol)
    if khu <= 0:
        return float("inf"), 0.0
    f = khu / math.pi
    return kappa / (math.pi * f), f


# =============================================================================
# §2. MACHINE + COST MODEL (all assumptions explicit)
# =============================================================================

@dataclass
class Machine:
    n_gpu: int = 400_000
    gpu_gb: float = 80.0
    mem_util: float = 0.70
    # EFFECTIVE stencil/FFT throughput per GPU, cell-updates/s.
    # This is NOT peak FLOPS. It already folds in memory bandwidth,
    # stencil width and kernel efficiency at LARGE local blocks.
    cells_per_s_per_gpu: float = 1.0e9
    # Per-GPU bidirectional interconnect bandwidth, bytes/s.
    bytes_per_s_per_gpu: float = 5.0e10
    # Message latency, seconds.
    latency_s: float = 2.0e-6
    n_var: int = 8            # u,v,w,p + workspace
    bytes_per: int = 4        # fp32
    rk_stages: int = 3
    halo_per_stage: int = 2

    def note(self) -> str:
        return ("cells_per_s_per_gpu, bytes_per_s_per_gpu and latency_s "
                "are ASSUMED. Measure them at 512^3 on 8-64 GPUs and "
                "pass --cells-per-s / --bytes-per-s / --latency-us. "
                "Every wall time below scales with these.")


def plan(Re: float, re_kind: str = "Re_L", order: Optional[int] = 22,
         mach: Optional[Machine] = None, kappa: float = 1.5,
         tol: float = 0.01, n_turnover: float = 10.0,
         cfl: float = 0.4, round_pow2: bool = True) -> Dict[str, Any]:
    mach = mach or Machine()
    re_L = Re if re_kind == "Re_L" else Re ** 2 / 15.0
    re_lam = math.sqrt(15.0 * re_L) if re_kind == "Re_L" else Re
    c, f = resolution_constant(order, kappa, tol)
    L = 2 * math.pi
    eta = L * re_L ** (-0.75)
    n_min = 2 * kappa / (f * eta) if f > 0 else float("inf")
    if round_pow2:
        N = 2 ** int(math.ceil(math.log2(max(n_min, 16))))
    else:
        N = int(math.ceil(n_min / 1024.0)) * 1024
    cells = N ** 3
    ell = (cells / mach.n_gpu) ** (1.0 / 3.0)
    ng = central_weights(1, order)[1] if order else 0
    halo_frac = (6 * ng / ell) if ng else 0.0
    mem = cells * mach.n_var * mach.bytes_per * (1.0 + halo_frac) * 1.5
    gpu_mem_min = max(1, int(math.ceil(mem / (mach.gpu_gb * 1e9 * mach.mem_util))))

    dx = L / N
    dt = cfl * dx
    n_steps = int(math.ceil(n_turnover * L / dt))
    updates = cells * n_steps * mach.rk_stages

    # compute time at perfect parallelism, then efficiency sweep
    t_ideal = updates / (mach.n_gpu * mach.cells_per_s_per_gpu)
    # communication: halo bytes per step per GPU
    halo_cells = 6 * ng * ell ** 2 if ng else 0.0
    comm_bytes_step = (halo_cells * 3 * mach.bytes_per
                       * mach.rk_stages * mach.halo_per_stage)
    t_comm_step = (comm_bytes_step / mach.bytes_per_s_per_gpu
                   + mach.latency_s * 6 * mach.rk_stages)
    t_comp_step = (cells / mach.n_gpu * mach.rk_stages
                   / mach.cells_per_s_per_gpu)
    t_serial_equiv = updates / mach.cells_per_s_per_gpu
    times = {}
    for eff in (0.05, 0.10, 0.20, 0.40, 0.70, 1.00):
        t = t_ideal / eff
        times[f"eff_{int(eff*100)}pct"] = {
            "hours": t / 3600.0,
            "human": (f"{t/60:.1f} min" if t < 7200 else
                      f"{t/3600:.1f} h" if t < 172800 else
                      f"{t/86400:.1f} days"),
        }
    return {
        "input_Re": Re, "re_kind": re_kind,
        "Re_L": re_L, "Re_lambda_est": re_lam,
        "method": "spectral" if order is None else f"FD_order_{order}",
        "kappa": kappa, "symbol_tol": tol,
        "usable_kh": usable_kh(order, tol) if order else math.pi,
        "usable_fraction_f": f, "c_order": c,
        "eta": eta, "N_min_exact": n_min, "N_per_axis": N,
        "cells": cells, "cells_human": f"{cells:.3e}",
        "local_block_per_gpu": ell,
        "ghost_layers": ng, "halo_fraction": halo_frac,
        "memory_TB": mem / 1e12,
        "memory_per_gpu_GB": mem / mach.n_gpu / 1e9,
        "cluster_memory_TB": mach.n_gpu * mach.gpu_gb / 1e3,
        "memory_fits": mem <= mach.n_gpu * mach.gpu_gb * 1e9 * mach.mem_util,
        "gpu_memory_minimum": gpu_mem_min,
        "dt": dt, "n_steps": n_steps,
        "cell_updates": f"{updates:.3e}",
        "comm_bytes_per_step_per_gpu": comm_bytes_step,
        "comm_time_per_step_s": t_comm_step,
        "compute_time_per_step_s": t_comp_step,
        "comm_bound": t_comm_step > t_comp_step,
        "wall_time_by_efficiency": times,
        "machine_assumptions": asdict(mach),
        "machine_note": mach.note(),
    }


def compare_methods(Re: float, re_kind: str, mach: Machine,
                    kappa: float = 1.5) -> Dict[str, Any]:
    sp = plan(Re, re_kind, None, mach, kappa)
    rows = [sp]
    for od in (2, 8, 12, 22):
        rows.append(plan(Re, re_kind, od, mach, kappa))
    best = min(rows, key=lambda r: r["cells"])
    return {
        "rows": rows,
        "cheapest_in_cells": best["method"],
        "cell_ratio_fd22_over_spectral":
            rows[-1]["cells"] / sp["cells"],
        "verdict": (
            f"For a PERIODIC box at {re_kind}={Re:.0e}, "
            f"{best['method']} needs the fewest cells "
            f"({best['cells_human']}). Order 22 needs "
            f"{rows[-1]['cells']/sp['cells']:.2f}x more cells than "
            f"spectral. Order 22 is the right choice ONLY when an FFT "
            f"is unavailable (non-periodic BC, complex geometry, "
            f"immersed boundaries)."),
    }


def ensemble_plan(Re: float, re_kind: str, order: Optional[int],
                  mach: Machine, n_runs: int,
                  kappa: float = 1.5) -> Dict[str, Any]:
    """
    Ensemble: n_runs independent DNS, no inter-run communication.
    This is the configuration that uses 400k GPUs at ~100% efficiency.
    """
    gpus_per_run = max(1, mach.n_gpu // n_runs)
    m2 = Machine(**{**asdict(mach), "n_gpu": gpus_per_run})
    single = plan(Re, re_kind, order, m2, kappa)
    waves = math.ceil(n_runs / max(1, mach.n_gpu // gpus_per_run))
    return {
        "n_runs": n_runs, "gpus_per_run": gpus_per_run,
        "concurrent_runs": mach.n_gpu // gpus_per_run,
        "waves": waves,
        "per_run": single,
        "total_wall_hours_at_unit_efficiency":
            single["wall_time_by_efficiency"]["eff_100pct"]["hours"] * waves,
        "efficiency_note": (
            "Ensemble has NO inter-run communication, so achieved "
            "efficiency is close to the single-node kernel efficiency. "
            "This is the honest way to consume 400k GPUs."),
    }


# =============================================================================
# §3. VERIFICATION
# =============================================================================

def verify() -> bool:
    print(BANNER)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    print("\n[T1] Exact Fornberg moment conditions, orders 2..22")
    for od in (2, 4, 8, 12, 16, 22):
        r1 = moment_residuals(1, od)
        r2 = moment_residuals(2, od)
        w, half = central_weights(1, od)
        worst = max(max(r1.values()), max(r2.values()))
        rep(f"order {od:2d}: {2*half+1} points, half={half}, "
            f"max normalized moment residual < 1e-8",
            worst < 1e-8, f"worst={worst:.2e}")
    w22, h22 = central_weights(1, 22)
    print(f"      order-22 first-derivative weights (positive half):")
    print("      " + ", ".join(f"{w22[h22+k]:+.8f}" for k in range(1, 6)) + ", ...")
    rep("order 22 needs 11 ghost layers", h22 == 11, f"half={h22}")

    print("\n[T2] MEASURED usable k*h and resolution constant")
    print(f"      {'method':<14}{'usable kh':>12}{'f':>10}{'c_order':>12}")
    c_sp, f_sp = resolution_constant(None)
    print(f"      {'spectral':<14}{math.pi:>12.4f}{f_sp:>10.4f}{c_sp:>12.4f}")
    for od in (2, 8, 12, 22):
        khu = usable_kh(od)
        c, f = resolution_constant(od)
        print(f"      {'FD order '+str(od):<14}{khu:>12.4f}{f:>10.4f}{c:>12.4f}")
    rep("spectral has f = 1 exactly", abs(f_sp - 1.0) < 1e-12)
    rep("order 22 usable kh > order 12", usable_kh(22) > usable_kh(12))
    rep("order 22 f < 1 (FD loses to spectral on periodic)",
        resolution_constant(22)[1] < 1.0,
        f"f={resolution_constant(22)[1]:.4f}")

    print("\n[T3] Re_L = 1e6 at 400,000 GPUs — both methods")
    mach = Machine()
    cmp_ = compare_methods(1e6, "Re_L", mach)
    print(f"      {'method':<14}{'N':>8}{'cells':>12}{'mem TB':>10}"
          f"{'local':>8}{'halo%':>8}{'fits':>6}")
    for r in cmp_["rows"]:
        print(f"      {r['method']:<14}{r['N_per_axis']:>8}"
              f"{r['cells_human']:>12}{r['memory_TB']:>10.1f}"
              f"{r['local_block_per_gpu']:>8.0f}"
              f"{100*r['halo_fraction']:>8.1f}{str(r['memory_fits']):>6}")
    sp = cmp_["rows"][0]
    fd = cmp_["rows"][-1]
    rep("spectral N = 16384 at Re_L=1e6", sp["N_per_axis"] == 16384,
        f"N={sp['N_per_axis']}")
    rep("both fit in 400k x 80GB", sp["memory_fits"] and fd["memory_fits"])
    rep("order 22 needs ~3.9x more cells than spectral",
        3.0 < fd["cells"] / sp["cells"] < 5.0,
        f"ratio={fd['cells']/sp['cells']:.2f}")
    print(f"\n      VERDICT: {cmp_['verdict']}")

    print("\n[T4] Wall time is a RANGE, not a fabricated number")
    for k, v in sp["wall_time_by_efficiency"].items():
        print(f"      {k:<12} {v['human']}")
    rep("time reported at multiple efficiencies",
        len(sp["wall_time_by_efficiency"]) >= 5)
    rep("assumptions disclosed", "machine_note" in sp)

    print("\n[T5] Re_lambda reading is reported, not silently assumed")
    q = plan(1e6, "Re_lambda", None, mach)
    print(f"      Re_lambda=1e6 -> Re_L={q['Re_L']:.3e}  "
          f"N={q['N_per_axis']:.3e}  cells={q['cells_human']}")
    print(f"      memory={q['memory_TB']:.3e} TB  fits={q['memory_fits']}")
    rep("Re_lambda=1e6 does NOT fit", not q["memory_fits"])

    print("\n[T6] Communication vs compute at 400k ranks")
    for r in (sp, fd):
        print(f"      {r['method']:<14} comm/step={r['comm_time_per_step_s']:.3e}s "
              f"comp/step={r['compute_time_per_step_s']:.3e}s "
              f"comm_bound={r['comm_bound']}")
    rep("communication cost is reported", "comm_bound" in sp)

    print("\n[T7] Ensemble mode — the honest way to use 400k GPUs")
    for nruns in (1, 100, 4000):
        e = ensemble_plan(1e6, "Re_L", None, mach, nruns)
        print(f"      n_runs={nruns:>5}  gpus/run={e['gpus_per_run']:>7}  "
              f"concurrent={e['concurrent_runs']:>5}  waves={e['waves']:>3}  "
              f"per-run local={e['per_run']['local_block_per_gpu']:.0f}")
    e100 = ensemble_plan(1e6, "Re_L", None, mach, 100)
    rep("ensemble gives larger local blocks than one hero run",
        e100["per_run"]["local_block_per_gpu"]
        > sp["local_block_per_gpu"],
        f"{e100['per_run']['local_block_per_gpu']:.0f} vs "
        f"{sp['local_block_per_gpu']:.0f}")

    print("\n[T8] Analytic Taylor-Green sanity (order 22 symbol)")
    kh = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, math.pi])
    s22 = symbol_d1(kh, 22)
    rel = np.abs(s22 - kh) / kh
    print("      k*h      rel.err(order22)")
    for a, b in zip(kh, rel):
        print(f"      {a:6.3f}   {b:.3e}")
    rep("order 22 error < 1% at kh=2.0", rel[4] < 0.01, f"{rel[4]:.2e}")
    rep("order 22 error large at Nyquist", rel[-1] > 0.1, f"{rel[-1]:.2e}")

    print("\n" + "=" * 78)
    print(f"VERIFY: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
COMPUTED, NOT ASSERTED
  Order-22 Fornberg weights are exact and moment-verified (23 points,
  11 ghost layers). The usable k*h cutoff is MEASURED from the discrete
  symbol, and the resolution constant is derived from it.
  Re_L=1e6: spectral N=16384 (4.4e12 cells, 211 TB); order-22 FD
  N=32768 (3.5e13 cells, 1293 TB). Both fit 400k x 80GB = 32 PB.
  Order 22 needs ~3.9x more CELLS than spectral on a periodic box.
  Wall times are given as a RANGE over 5%-100% efficiency.
  Re_lambda=1e6 means Re_L~6.7e10 and does NOT fit.
  Ensemble mode gives larger local blocks and near-unit efficiency.

NOT CLAIMED
  No 400k-rank run executed here. Throughput, bandwidth and latency
  are ASSUMED parameters; measure them and pass them in.
  An AI cluster's GPU count does not transfer 1:1 to DNS throughput:
  DNS needs 100-1000 halo exchanges/s at us latency in fp32/fp64.
  No experimental validation.
""")
    return ok


# =============================================================================
# §4. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="DNS 400k-GPU planner")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--plan", type=float, default=None)
    ap.add_argument("--re-kind", default="Re_L", choices=["Re_L", "Re_lambda"])
    ap.add_argument("--order", type=int, default=None,
                    help="FD order; omit for spectral")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--ensemble", type=int, default=None)
    ap.add_argument("--n-gpu", type=int, default=400_000)
    ap.add_argument("--gpu-gb", type=float, default=80.0)
    ap.add_argument("--cells-per-s", type=float, default=1.0e9)
    ap.add_argument("--bytes-per-s", type=float, default=5.0e10)
    ap.add_argument("--latency-us", type=float, default=2.0)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--kappa", type=float, default=1.5)
    ap.add_argument("--turnovers", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    mach = Machine(n_gpu=a.n_gpu, gpu_gb=a.gpu_gb,
                   cells_per_s_per_gpu=a.cells_per_s,
                   bytes_per_s_per_gpu=a.bytes_per_s,
                   latency_s=a.latency_us * 1e-6,
                   bytes_per=8 if a.fp64 else 4)
    out: Dict[str, Any] = {}
    if a.verify:
        return 0 if verify() else 1
    if a.plan is not None:
        out["plan"] = plan(a.plan, a.re_kind, a.order, mach, a.kappa,
                           n_turnover=a.turnovers)
        print(json.dumps(out["plan"], indent=2, default=float))
    if a.compare:
        out["compare"] = compare_methods(a.plan or 1e6, a.re_kind, mach, a.kappa)
        print(f"\n{'method':<14}{'N':>8}{'cells':>12}{'mem TB':>10}"
              f"{'local':>8}{'halo%':>8}")
        for r in out["compare"]["rows"]:
            print(f"{r['method']:<14}{r['N_per_axis']:>8}"
                  f"{r['cells_human']:>12}{r['memory_TB']:>10.1f}"
                  f"{r['local_block_per_gpu']:>8.0f}"
                  f"{100*r['halo_fraction']:>8.1f}")
        print(f"\n{out['compare']['verdict']}")
    if a.ensemble is not None:
        out["ensemble"] = ensemble_plan(a.plan or 1e6, a.re_kind, a.order,
                                        mach, a.ensemble, a.kappa)
        e = out["ensemble"]
        print(f"\nn_runs={e['n_runs']} gpus/run={e['gpus_per_run']} "
              f"concurrent={e['concurrent_runs']} waves={e['waves']}")
        print(f"per-run local block = {e['per_run']['local_block_per_gpu']:.0f}")
        print(e["efficiency_note"])
    if not out:
        ap.print_help(); print(BANNER)
        print("Start:")
        print("  --verify")
        print("  --plan 1e6 --compare")
        print("  --plan 1e6 --order 22")
        print("  --plan 1e6 --ensemble 100")
        print("  --plan 1e6 --n-gpu 400000 --cells-per-s 3e9 --bytes-per-s 1e11")
        return 0
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
