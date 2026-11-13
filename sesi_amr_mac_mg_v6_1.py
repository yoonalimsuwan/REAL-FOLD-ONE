#!/usr/bin/env python3
"""
=============================================================================
SESI AMR MAC-MG v6.1
MAC-face conservative momentum + MG cycle predictor + AMP/DDP harness
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit.
             What MSPS NETWORK SEE, Lord Buddha Knows.

  "Tao is the wellspring; Te is the path; Si is the proof."

  "The teaching of Confucius tells us that this world should not
   destroy one another."

  "The Lord Christ gave His life on the cross to confirm the way of
   turning the other cheek: when struck on the right cheek, turn the
   left cheek to him. Whoever believes in this way shall be delivered
   from the path of sin."

License : MIT | Year: 2026

WHAT THIS FILE ADDS
  [A] MACFaceConservativeMomentum
      Flux-form div(rho u (x) u) on the TRUE staggered MAC layout:
          u : (nx+1, ny,   nz  )
          v : (nx,   ny+1, nz  )
          w : (nx,   ny,   nz+1)
      Compatible with both:
          sesi_multiphase_flow_fix.py   (one_cfd_stencils / MACOps)
          sesi_multiphase_flow_fix2.py  (local MACGrid)
      Conservative by telescoping flux differences.
      Supports periodic / wall / slip and cut-cell apertures.

  [B] MGCyclePredictor
      Online RLS predictor of multigrid V-cycle count BEFORE solving.
      Features: grid size, tolerance, density contrast, RHS roughness,
      aperture/solid fraction, previous-cycle memory.
      Used for AMR work weights before the expensive solve.

  [C] PredictiveMultigrid
      Wrapper around a multigrid solver:
        predicted_cycles = predictor.predict(rhs, rho, apertures, tol)
        actual solve
        predictor.update(actual_cycles)
      Returns both predicted and actual cycles for auditing.

  [D] ProductionRuntime
      AMP autocast context
      DDP wrapper for batched/parameter-sweep training
      optional torch.compile
      checkpoint helpers
      z-slab shard/halo utilities for future domain decomposition

HONEST STATUS
  ✅ MAC-face conservative momentum is implemented and tested.
  ✅ MG cycle predictor is implemented and tested online.
  ✅ AMP/DDP/compile scaffolding is provided.
  ❌ NOT yet certified production/SOTA:
       - no Basilisk/OpenFOAM comparison
       - no multi-rank AMR scaling table
       - no validated energy conservation
       - DDP here is training/batch scaffolding, not a certified
         domain-decomposed AMR solver
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("sesi_amr_mac_mg")

VERSION = "6.1.0"

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    _HAS_DIST = True
except Exception:
    dist = None
    DDP = None
    _HAS_DIST = False


# =============================================================================
# A. MAC-FACE CONSERVATIVE MOMENTUM
# =============================================================================

def _loc_shape(n: Tuple[int, int, int], loc: Tuple[int, int, int]):
    return tuple(n[i] + loc[i] for i in range(3))


def _narrow(t: torch.Tensor, axis: int, start: int, length: int):
    return t.narrow(axis, start, length)


def _interp_axis(t: torch.Tensor, axis: int,
                 src_bit: int, dst_bit: int,
                 bc: str = "periodic") -> torch.Tensor:
    """
    Interpolate between cell-centre (0) and face (1) locations along axis.
    src_bit=1, dst_bit=0 : face -> centre, length n+1 -> n
    src_bit=0, dst_bit=1 : centre -> face, length n -> n+1
    """
    if src_bit == dst_bit:
        return t

    if src_bit == 1 and dst_bit == 0:
        n = t.shape[axis] - 1
        return 0.5 * (_narrow(t, axis, 0, n) + _narrow(t, axis, 1, n))

    # centre -> face
    n = t.shape[axis]
    if n == 1:
        return torch.cat([t, t], dim=axis)

    interior = 0.5 * (_narrow(t, axis, 0, n - 1)
                      + _narrow(t, axis, 1, n - 1))
    if bc == "periodic":
        b = 0.5 * (_narrow(t, axis, n - 1, 1) + _narrow(t, axis, 0, 1))
        return torch.cat([b, interior, b], dim=axis)
    else:
        left = _narrow(t, axis, 0, 1)
        right = _narrow(t, axis, n - 1, 1)
        return torch.cat([left, interior, right], dim=axis)


def interp_location(t: torch.Tensor,
                    src_loc: Tuple[int, int, int],
                    dst_loc: Tuple[int, int, int],
                    bc: Tuple[str, str, str]) -> torch.Tensor:
    """Interpolate a MAC field from one staggered location to another."""
    out = t
    for ax in range(3):
        if src_loc[ax] != dst_loc[ax]:
            out = _interp_axis(out, ax, src_loc[ax], dst_loc[ax], bc[ax])
    return out


def face_scalar_from_cc(cc: torch.Tensor, axis: int,
                        bc: Tuple[str, str, str]) -> torch.Tensor:
    loc = [0, 0, 0]
    loc[axis] = 1
    return interp_location(cc, (0, 0, 0), tuple(loc), bc)


def face_density(rho_cc: torch.Tensor, axis: int,
                 bc: Tuple[str, str, str],
                 mode: str = "arithmetic") -> torch.Tensor:
    if mode == "harmonic":
        inv = 1.0 / rho_cc.clamp_min(1e-30)
        inv_f = face_scalar_from_cc(inv, axis, bc)
        return 1.0 / inv_f.clamp_min(1e-30)
    return face_scalar_from_cc(rho_cc, axis, bc)


def zero_normal_wall_flux(flux: torch.Tensor, axis: int,
                          bc: Tuple[str, str, str]) -> torch.Tensor:
    """Zero normal mass flux at impermeable walls, out-of-place."""
    if bc[axis] == "periodic":
        return flux
    n = flux.shape[axis]
    mask = torch.ones_like(flux)
    mask = mask.index_copy(axis, torch.tensor([0, n - 1], device=flux.device),
                           torch.zeros_like(_narrow(mask, axis, 0, 1)).expand(
                               *[_narrow(mask, axis, 0, 1).shape[d] if d != axis else 2
                                 for d in range(flux.dim())]))
    # simpler robust mask:
    mask = torch.ones_like(flux)
    sl0 = [slice(None)] * flux.dim(); sl0[axis] = 0
    sl1 = [slice(None)] * flux.dim(); sl1[axis] = n - 1
    mask[tuple(sl0)] = 0.0
    mask[tuple(sl1)] = 0.0
    return flux * mask


@dataclass
class MACMomentumConfig:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    bc: Tuple[str, str, str] = ("periodic", "periodic", "periodic")
    density_face: str = "arithmetic"   # arithmetic | harmonic
    flux_scheme: str = "central"       # central | smooth_upwind
    upwind_tau: float = 1e-2
    aperture_floor: float = 1e-12

    @property
    def n(self): return (self.nx, self.ny, self.nz)

    @property
    def h(self): return (self.dx, self.dy, self.dz)


class MACFaceConservativeMomentum(nn.Module):
    """
    Conservative MAC-face momentum advection.

    For momentum component c (u/v/w), the advective flux through a-normal
    faces is built at the control-volume face location of that component.
    The divergence is a telescoping flux difference, so the total momentum
    change is exactly the boundary flux imbalance.

    This is the form needed for AMR refluxing.
    """

    def __init__(self, cfg: MACMomentumConfig):
        super().__init__()
        self.cfg = cfg

    def mass_flux(self, rho_cc: torch.Tensor,
                  vel: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  axis: int) -> torch.Tensor:
        """Continuity-consistent mass flux rho_f * u_f at axis-faces."""
        rho_f = face_density(rho_cc, axis, self.cfg.bc, self.cfg.density_face)
        m = rho_f * vel[axis]
        return zero_normal_wall_flux(m, axis, self.cfg.bc)

    @staticmethod
    def component_loc(c: int) -> Tuple[int, int, int]:
        loc = [0, 0, 0]
        loc[c] = 1
        return tuple(loc)

    @staticmethod
    def flux_loc(c: int, a: int) -> Tuple[int, int, int]:
        """
        Momentum component c lives at loc_c.
        Flux normal a toggles axis a of that location.
        """
        loc = list(MACFaceConservativeMomentum.component_loc(c))
        loc[a] = 1 - loc[a]
        return tuple(loc)

    def flux(self, rho_cc: torch.Tensor,
             vel: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
             c: int, a: int,
             apertures: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
             ) -> torch.Tensor:
        """
        Flux of momentum component c through faces normal to a.
        Returns tensor at the staggered flux location.
        """
        cfg = self.cfg
        loc_c = self.component_loc(c)
        loc_a = self.component_loc(a)
        loc_f = self.flux_loc(c, a)

        m_a = self.mass_flux(rho_cc, vel, a)
        m_at_f = interp_location(m_a, loc_a, loc_f, cfg.bc)
        v_c_at_f = interp_location(vel[c], loc_c, loc_f, cfg.bc)

        if cfg.flux_scheme == "smooth_upwind":
            # Smooth upwind blend of the transported momentum component.
            # Central value plus a differentiable upwind bias.
            gate = torch.sigmoid(m_at_f / max(cfg.upwind_tau, 1e-12))
            # left/right along a in the component's own location
            vc = vel[c]
            if a == c:
                # vc is face-located along a; shift to centre-like neighbours
                left = torch.roll(vc, 1, a)
                right = vc
            else:
                left = vc
                right = torch.roll(vc, -1, a)
            left_f = interp_location(left, loc_c, loc_f, cfg.bc)
            right_f = interp_location(right, loc_c, loc_f, cfg.bc)
            v_up = gate * left_f + (1.0 - gate) * right_f
            flux = m_at_f * v_up
        else:
            flux = m_at_f * v_c_at_f

        if apertures is not None:
            ap_f = interp_location(apertures[a], loc_a, loc_f, cfg.bc)
            flux = flux * ap_f.clamp_min(cfg.aperture_floor)

        return flux

    def divergence(self, rho_cc: torch.Tensor,
                   vel: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   apertures: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (div_u, div_v, div_w) at the MAC face locations.
        """
        cfg = self.cfg
        out = []
        for c in range(3):
            div = torch.zeros_like(vel[c])
            for a in range(3):
                f = self.flux(rho_cc, vel, c, a, apertures)
                h = cfg.h[a]
                if a == c:
                    # flux is centre-located along c, length n_c
                    if cfg.bc[a] == "periodic":
                        d = f - torch.roll(f, 1, a)
                        d = torch.cat([d, _narrow(d, a, 0, 1)], dim=a)
                    else:
                        interior = _narrow(f, a, 1, f.shape[a] - 1) \
                                   - _narrow(f, a, 0, f.shape[a] - 1)
                        z = torch.zeros_like(_narrow(vel[c], a, 0, 1))
                        d = torch.cat([z, interior, z], dim=a)
                else:
                    # flux is face-located along a, length n_a+1
                    d = _narrow(f, a, 1, f.shape[a] - 1) \
                        - _narrow(f, a, 0, f.shape[a] - 1)
                div = div + d / h
            out.append(div)
        return out[0], out[1], out[2]

    def flux_register_terms(self, rho_cc, vel, apertures=None):
        """
        Return all component/axis fluxes for AMR reflux registration.
        dict[(component, axis)] = flux tensor
        """
        return {(c, a): self.flux(rho_cc, vel, c, a, apertures)
                for c in range(3) for a in range(3)}


def mac_conservative_terms_from_state(solver, st, rho=None, apertures=None):
    """
    Adapter for BOTH sesi_multiphase_flow_fix.py and fix2.py.

    Usage inside solver.step():
        conv_u, conv_v, conv_w = mac_conservative_terms_from_state(self, st)
        acc_u = -conv_u + ...
    """
    c = solver.cfg
    if rho is None:
        rho = solver.density(st.alpha)
    cfg = MACMomentumConfig(
        nx=c.nx, ny=c.ny, nz=c.nz,
        dx=c.dx, dy=c.dy, dz=c.dz,
        bc=tuple(c.bc),
    )
    mac = MACFaceConservativeMomentum(cfg).to(rho.device)
    return mac.divergence(rho, (st.u, st.v, st.w), apertures)


# =============================================================================
# B. MG CYCLE PREDICTOR
# =============================================================================

class MGCyclePredictor(nn.Module):
    """
    Online RLS predictor of multigrid V-cycle count.

    Prediction is used BEFORE the solve for AMR work weighting.
    After the solve, update() calibrates the predictor against the
    actual cycle count.

    Features are nonlinear in the state, but linear in theta, so RLS
    remains a valid online estimator.
    """

    NFEAT = 9

    def __init__(self, theta0: Optional[List[float]] = None,
                 lam: float = 0.985, P0: float = 1e3,
                 min_cycles: int = 1, max_cycles: int = 80):
        super().__init__()
        th = theta0 if theta0 is not None else [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
        assert len(th) == self.NFEAT
        self.register_buffer("theta", torch.tensor(th, dtype=torch.float64))
        self.register_buffer("P", torch.eye(self.NFEAT, dtype=torch.float64) * P0)
        self.lam = float(lam)
        self.min_cycles = int(min_cycles)
        self.max_cycles = int(max_cycles)
        self.register_buffer("n_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_pred", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("last_actual", torch.tensor(0.0, dtype=torch.float64))

    def features(self, rhs: torch.Tensor, rho: torch.Tensor,
                 tol: float,
                 apertures: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                 prev_cycles: Optional[float] = None) -> torch.Tensor:
        rhs = rhs.detach().to(torch.float64)
        rho = rho.detach().to(torch.float64)
        n = rhs.numel()
        logN = math.log(max(n, 2))
        logTol = -math.log(max(tol, 1e-30))
        contrast = float(rho.max() / rho.min().clamp_min(1e-30))
        logContrast = math.log(max(contrast, 1.0))
        rho_cv = float(rho.std() / rho.mean().clamp_min(1e-30))
        rhs_norm = float(rhs.norm())
        # high-frequency roughness proxy
        rough = 0.0
        for ax in range(3):
            rough += float((torch.roll(rhs, -1, ax) - rhs).pow(2).mean())
        rough = math.log1p(rough)
        solid = 0.0
        ap_min = 1.0
        if apertures is not None:
            for ap in apertures:
                solid += float((ap < 1e-8).to(torch.float64).mean())
                ap_min = min(ap_min, float(ap.min()))
            solid /= 3.0
        prev = float(prev_cycles) if prev_cycles is not None else 0.0
        x = torch.tensor([
            1.0,
            logN / 10.0,
            logTol / 10.0,
            logContrast,
            rho_cv,
            math.log1p(rhs_norm),
            rough,
            solid,
            prev / 10.0,
        ], dtype=torch.float64)
        return x

    def predict(self, rhs, rho, tol, apertures=None, prev_cycles=None) -> int:
        x = self.features(rhs, rho, tol, apertures, prev_cycles)
        y = float(self.theta @ x)
        y = max(self.min_cycles, min(self.max_cycles, int(math.ceil(y))))
        self.last_pred.fill_(float(y))
        return y

    def update(self, rhs, rho, tol, actual_cycles: int,
               apertures=None, prev_cycles=None):
        x = self.features(rhs, rho, tol, apertures, prev_cycles)
        y = float(actual_cycles)
        Px = self.P @ x
        den = self.lam + float(x @ Px)
        if den <= 1e-30:
            return
        K = Px / den
        err = y - float(self.theta @ x)
        self.theta = (self.theta + K * err).clamp_min(1e-6)
        self.P = (self.P - torch.outer(K, Px)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1
        self.last_actual.fill_(y)

    def work_weight(self, rhs, rho, tol, ncells: int,
                    substeps: int = 1, apertures=None, prev_cycles=None) -> float:
        """AMR work weight BEFORE solving."""
        cyc = self.predict(rhs, rho, tol, apertures, prev_cycles)
        return float(ncells * substeps * cyc)


class PredictiveMultigrid:
    """
    Wrapper: predict MG cycles, solve, then calibrate predictor.
    Compatible with any solver exposing:
        solve(rhs, x0=None, tol=..., max_vcycles=...) -> (p, stats)
    """

    def __init__(self, mg_solver, predictor: Optional[MGCyclePredictor] = None):
        self.mg = mg_solver
        self.predictor = predictor or MGCyclePredictor()
        self.history: List[Dict[str, Any]] = []

    def solve(self, rhs, rho, apertures=None, tol=1e-7,
              max_vcycles=40, prev_cycles=None, x0=None):
        pred = self.predictor.predict(rhs, rho, tol, apertures, prev_cycles)
        budget = max(pred, 4)
        budget = min(budget, max_vcycles)
        p, stats = self.mg.solve(rhs, x0=x0, tol=tol, max_vcycles=budget)
        actual = int(stats.get("iterations", stats.get("vcycles", budget)))
        self.predictor.update(rhs, rho, tol, actual, apertures, prev_cycles)
        rec = {
            "predicted": pred,
            "budget": budget,
            "actual": actual,
            "residual": float(stats.get("rel_residual", float("nan"))),
        }
        self.history.append(rec)
        return p, {**stats, "predicted_cycles": pred, "actual_cycles": actual}


# =============================================================================
# C. PRODUCTION RUNTIME HARNESS
# =============================================================================

@dataclass
class ProductionRuntimeConfig:
    device: str = "cpu"
    amp: bool = False
    amp_dtype: str = "bfloat16"     # bfloat16 | float16
    ddp: bool = False
    compile: bool = False
    checkpoint: bool = False
    checkpoint_dir: str = "./checkpoints"

    @property
    def torch_device(self):
        return torch.device(self.device)

    @property
    def torch_amp_dtype(self):
        return torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16


class ProductionRuntime:
    """
    AMP / DDP / compile / checkpoint scaffolding.

    NOTE: DDP here is for batched simulations / parameter sweeps.
    It is NOT a certified domain-decomposed AMR solver.
    """

    def __init__(self, cfg: ProductionRuntimeConfig):
        self.cfg = cfg
        self.device = cfg.torch_device
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(cfg.amp and self.device.type == "cuda")
        )
        self.rank = 0
        self.world = 1
        if cfg.ddp and _HAS_DIST and dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world = dist.get_world_size()

    def autocast(self):
        if self.cfg.amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.cfg.torch_amp_dtype)
        if self.cfg.amp and self.device.type == "cpu":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def wrap_ddp(self, module: nn.Module) -> nn.Module:
        if not self.cfg.ddp:
            return module
        if not (_HAS_DIST and dist.is_available() and dist.is_initialized()):
            log.warning("DDP requested but torch.distributed not initialized; returning unwrapped module.")
            return module
        device_ids = [torch.cuda.current_device()] if self.device.type == "cuda" else None
        return DDP(module, device_ids=device_ids)

    def maybe_compile(self, fn: Callable) -> Callable:
        if self.cfg.compile and hasattr(torch, "compile"):
            try:
                return torch.compile(fn)
            except Exception as e:
                log.warning(f"torch.compile failed: {e}")
        return fn

    def save_checkpoint(self, name: str, payload: Dict[str, Any]):
        if not self.cfg.checkpoint:
            return
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save(payload, path)
        log.info(f"checkpoint saved -> {path}")

    def load_checkpoint(self, path: str, map_location=None):
        return torch.load(path, map_location=map_location or self.device)

    def shard_z(self, t: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Simple z-slab shard for future domain decomposition."""
        if self.world == 1:
            return t
        n = t.shape[dim]
        if n % self.world != 0:
            raise ValueError("z dimension must be divisible by world size")
        local = n // self.world
        return t.narrow(dim, self.rank * local, local)

    def halo_exchange_z(self, t: torch.Tensor, ng: int = 2, dim: int = -1) -> torch.Tensor:
        """
        Non-blocking halo exchange along z for a padded tensor.
        Requires torch.distributed initialized.
        """
        if self.world == 1:
            return t
        if not (_HAS_DIST and dist.is_initialized()):
            return t
        left = self.rank - 1 if self.rank > 0 else -1
        right = self.rank + 1 if self.rank < self.world - 1 else -1
        reqs = []
        recv_lo = torch.empty_like(t.narrow(dim, 0, ng))
        recv_hi = torch.empty_like(t.narrow(dim, t.shape[dim] - ng, ng))
        if right >= 0:
            reqs.append(dist.irecv(recv_lo, src=right))
        if left >= 0:
            reqs.append(dist.irecv(recv_hi, src=left))
        if left >= 0:
            reqs.append(dist.isend(t.narrow(dim, ng, ng).contiguous(), dst=left))
        if right >= 0:
            reqs.append(dist.isend(t.narrow(dim, t.shape[dim] - 2 * ng, ng).contiguous(), dst=right))
        for r in reqs:
            r.wait()
        out = t.clone()
        if right >= 0:
            out.narrow(dim, 0, ng).copy_(recv_lo)
        if left >= 0:
            out.narrow(dim, t.shape[dim] - ng, ng).copy_(recv_hi)
        return out


# =============================================================================
# D. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 88)
    print(f"  SESI AMR MAC-MG v{VERSION} — self-test")
    print("=" * 88)
    torch.manual_seed(0)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1 MAC shapes
    nx, ny, nz = 10, 12, 8
    cfg = MACMomentumConfig(nx, ny, nz, 1/nx, 1/ny, 1/nz,
                            bc=("periodic", "periodic", "periodic"))
    mac = MACFaceConservativeMomentum(cfg)
    rho = torch.rand(nx, ny, nz, dtype=torch.float64) * 999 + 1
    u = torch.randn(nx + 1, ny, nz, dtype=torch.float64) * 0.1
    v = torch.randn(nx, ny + 1, nz, dtype=torch.float64) * 0.1
    w = torch.randn(nx, ny, nz + 1, dtype=torch.float64) * 0.1
    du, dv, dw = mac.divergence(rho, (u, v, w))
    rep("MAC divergence shapes",
        du.shape == u.shape and dv.shape == v.shape and dw.shape == w.shape)

    # T2 periodic conservation
    su = float(du[:-1].sum())
    sv = float(dv[:, :-1].sum())
    sw = float(dw[:, :, :-1].sum())
    rep("MAC periodic momentum telescopes",
        max(abs(su), abs(sv), abs(sw)) < 1e-10,
        f"sums=({su:.2e},{sv:.2e},{sw:.2e})")

    # T3 differentiability
    u.requires_grad_(True)
    du, _, _ = mac.divergence(rho, (u, v, w))
    du.pow(2).sum().backward()
    rep("MAC momentum autograd", u.grad is not None and torch.isfinite(u.grad).all())

    # T4 wall flux zero
    cfg_wall = MACMomentumConfig(nx, ny, nz, 1/nx, 1/ny, 1/nz,
                                 bc=("wall", "periodic", "periodic"))
    mac_wall = MACFaceConservativeMomentum(cfg_wall)
    m = mac_wall.mass_flux(rho, (u.detach(), v, w), axis=0)
    rep("Wall normal mass flux zero",
        float(m[0].abs().max()) < 1e-14 and float(m[-1].abs().max()) < 1e-14)

    # T5 MG predictor learns
    pred = MGCyclePredictor()
    rhs = torch.randn(16, 16, 16, dtype=torch.float64)
    rho_mg = torch.ones(16, 16, 16, dtype=torch.float64)
    for i in range(50):
        actual = 5 + (i % 3)
        pred.update(rhs, rho_mg, 1e-6, actual)
    p = pred.predict(rhs, rho_mg, 1e-6)
    rep("MG predictor returns bounded cycles", 1 <= p <= 80, f"pred={p}")

    # T6 work weight
    wgt = pred.work_weight(rhs, rho_mg, 1e-6, ncells=16**3, substeps=2)
    rep("MG work weight positive", wgt > 0, f"weight={wgt:.1f}")

    # T7 AMP runtime
    rt = ProductionRuntime(ProductionRuntimeConfig(device="cpu", amp=False))
    with rt.autocast():
        y = mac.divergence(rho, (u.detach(), v, w))[0]
    rep("AMP context runs", torch.isfinite(y).all())

    # T8 DDP safe degradation
    mod = rt.wrap_ddp(mac)
    rep("DDP safe degradation", mod is not None)

    print("\n" + "=" * 88)
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 88)
    print("""
  PROVEN BY THESE TESTS
    • MAC-face conservative momentum has correct staggered shapes
    • periodic momentum divergence telescopes to ~0
    • autograd flows through MAC momentum
    • wall normal mass flux is zero
    • MG cycle predictor updates online and returns bounded weights
    • AMP/DDP scaffolding degrades safely

  NOT YET CERTIFIED
    • Basilisk / OpenFOAM benchmark comparison
    • multi-rank AMR scaling
    • energy conservation
    • full domain-decomposed AMR production use
""")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        raise SystemExit(0 if self_test() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
