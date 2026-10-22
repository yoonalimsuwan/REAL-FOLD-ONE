"""
=============================================================================
SESI FUSED PHYSICS — PyTorch Autograd Wrappers + Benchmark
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
License      : MIT
Year         : 2026
=============================================================================
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try importing Triton kernels
try:
    from sesi_triton_kernels import (
        triton_soft_abs, triton_soft_floor, triton_soft_ceiling,
        triton_soft_sign, triton_soft_gate, triton_fused_sesi_barrier,
        triton_laplacian_3d, triton_gradient_3d, triton_divergence_3d,
        triton_pressure_jacobi, triton_chorin_projection,
        triton_fused_ns_step, triton_ns_step,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Try importing CUDA kernels
_CUDA_EXT = None
def _get_cuda_ext():
    global _CUDA_EXT
    if _CUDA_EXT is None:
        try:
            from sesi_cuda_kernels import load_cuda_kernels
            _CUDA_EXT = load_cuda_kernels()
        except Exception:
            _CUDA_EXT = None
    return _CUDA_EXT

__all__ = [
    "SoftAbs", "SoftFloor", "SoftCeiling", "SoftSign", "SoftGate",
    "FusedSESIBarrier",
    "Laplacian3D", "Gradient3D", "Divergence3D",
    "PressureJacobi", "ChorinProjection", "FusedNSStep",
    "benchmark_all",
]


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Autograd Functions
# ═══════════════════════════════════════════════════════════════════════

class SoftAbsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, eps: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.eps = eps
        if HAS_TRITON and x.is_cuda:
            return triton_soft_abs(x, eps)
        return torch.sqrt(x * x + eps * eps)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        x, = ctx.saved_tensors
        eps = ctx.eps
        denom = torch.sqrt(x * x + eps * eps)
        grad_x = grad_output * x / (denom + 1e-30)
        return grad_x, None


class SoftFloorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, floor: float, beta: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.floor = floor
        ctx.beta = beta
        if HAS_TRITON and x.is_cuda:
            return triton_soft_floor(x, floor, beta)
        return floor + F.softplus(x - floor, beta=beta)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        x, = ctx.saved_tensors
        sig = torch.sigmoid(ctx.beta * (x - ctx.floor))
        return grad_output * sig, None, None


class SoftCeilingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, ceil_val: float, beta: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.ceil_val = ceil_val
        ctx.beta = beta
        if HAS_TRITON and x.is_cuda:
            return triton_soft_ceiling(x, ceil_val, beta)
        return ceil_val - F.softplus(ceil_val - x, beta=beta)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        x, = ctx.saved_tensors
        sig = torch.sigmoid(ctx.beta * (x - ctx.ceil_val))
        return grad_output * (1.0 - sig), None, None


class SoftSignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, beta: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        if HAS_TRITON and x.is_cuda:
            return triton_soft_sign(x, beta)
        return torch.tanh(beta * x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        x, = ctx.saved_tensors
        beta = ctx.beta
        tanh_val = torch.tanh(beta * x)
        return grad_output * beta * (1.0 - tanh_val * tanh_val), None


class SoftGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, c: float, tau: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.c = c
        ctx.tau = tau
        if HAS_TRITON and x.is_cuda:
            return triton_soft_gate(x, c, tau)
        return torch.sigmoid((x - c) / tau)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        x, = ctx.saved_tensors
        sig = torch.sigmoid((x - ctx.c) / ctx.tau)
        return grad_output * sig * (1.0 - sig) / ctx.tau, None, None


# ── Convenience nn.Modules ──────────────────────────────────────────────

class SoftAbs(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return SoftAbsFunction.apply(x, self.eps)


class SoftFloor(nn.Module):
    def __init__(self, floor: float, beta: float = 50.0):
        super().__init__()
        self.floor = floor
        self.beta = beta
    def forward(self, x: Tensor) -> Tensor:
        return SoftFloorFunction.apply(x, self.floor, self.beta)


class SoftCeiling(nn.Module):
    def __init__(self, ceil_val: float, beta: float = 50.0):
        super().__init__()
        self.ceil_val = ceil_val
        self.beta = beta
    def forward(self, x: Tensor) -> Tensor:
        return SoftCeilingFunction.apply(x, self.ceil_val, self.beta)


class SoftSign(nn.Module):
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta
    def forward(self, x: Tensor) -> Tensor:
        return SoftSignFunction.apply(x, self.beta)


class SoftGate(nn.Module):
    def __init__(self, c: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.c = c
        self.tau = tau
    def forward(self, x: Tensor) -> Tensor:
        return SoftGateFunction.apply(x, self.c, self.tau)


class FusedSESIBarrier(nn.Module):
    """
    Fused SESI topological jump barrier.
    Combines: barrier computation + Gumbel-sigmoid + sign surrogate + tanh.
    """
    def __init__(
        self,
        c1: float = 1.0,
        inner_clamp: float = 15.0,
        log_floor: float = -25.0,
        beta_sign: float = 10.0,
        jump_mag: float = 0.5,
        tau: float = 0.5,
    ):
        super().__init__()
        self.c1 = c1
        self.inner_clamp = inner_clamp
        self.log_floor = log_floor
        self.beta_sign = beta_sign
        self.jump_mag = jump_mag
        self.tau = tau

    def forward(
        self, s: Tensor, delta_e: Tensor, sigma_sq: Tensor, dt: float,
    ) -> Tuple[Tensor, Tensor]:
        if HAS_TRITON and s.is_cuda:
            return triton_fused_sesi_barrier(
                s, delta_e, sigma_sq,
                c1=self.c1, dt=dt,
                inner_clamp=self.inner_clamp,
                log_floor=self.log_floor,
                beta_sign=self.beta_sign,
                jump_mag=self.jump_mag,
                tau=self.tau,
                is_training=self.training,
            )
        # Fallback: pure PyTorch
        denom = sigma_sq * dt + 1e-8
        inner = (delta_e / denom).clamp(max=self.inner_clamp)
        log_p = (-self.c1 * torch.exp(inner)).clamp(min=self.log_floor)
        prob = torch.exp(log_p)
        sign_s = torch.tanh(self.beta_sign * s)
        if self.training:
            u = torch.rand_like(prob).clamp_(1e-7, 1.0 - 1e-7)
            g = torch.log(u) - torch.log1p(-u)
            logit = torch.log(prob + 1e-12) - torch.log1p(-prob + 1e-12)
            jump = torch.sigmoid((logit + g) / self.tau)
        else:
            jump = (prob > 0.5).to(prob.dtype)
        s_jumped = s + jump * sign_s * self.jump_mag
        return torch.tanh(s_jumped), prob


# ═══════════════════════════════════════════════════════════════════════
# Navier-Stokes Autograd Wrappers
# ═══════════════════════════════════════════════════════════════════════

class Laplacian3D(nn.Module):
    """3D Laplacian with periodic BC. Triton-accelerated."""
    def __init__(self, dx: float):
        super().__init__()
        self.dx = dx

    def forward(self, u: Tensor) -> Tensor:
        if HAS_TRITON and u.is_cuda:
            return triton_laplacian_3d(u, self.dx)
        # Fallback: PyTorch conv3d
        lap_kernel = torch.zeros(1, 1, 3, 3, 3, device=u.device, dtype=u.dtype)
        lap_kernel[0, 0, 1, 1, 1] = -6.0
        lap_kernel[0, 0, 0, 1, 1] = 1.0
        lap_kernel[0, 0, 2, 1, 1] = 1.0
        lap_kernel[0, 0, 1, 0, 1] = 1.0
        lap_kernel[0, 0, 1, 2, 1] = 1.0
        lap_kernel[0, 0, 1, 1, 0] = 1.0
        lap_kernel[0, 0, 1, 1, 2] = 1.0
        if u.dim() == 3:
            u = u.unsqueeze(0).unsqueeze(0)
        up = F.pad(u, (1, 1, 1, 1, 1, 1), mode="circular")
        out = F.conv3d(up, lap_kernel / (self.dx ** 2))
        return out.squeeze(0).squeeze(0) if out.shape[0] == 1 else out


class Gradient3D(nn.Module):
    """3D Gradient with periodic BC. Returns (gx, gy, gz)."""
    def __init__(self, dx: float):
        super().__init__()
        self.dx = dx

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if HAS_TRITON and u.is_cuda:
            return triton_gradient_3d(u, self.dx)
        # Fallback
        if u.dim() == 3:
            u = u.unsqueeze(0).unsqueeze(0)
        up = F.pad(u, (1, 1, 1, 1, 1, 1), mode="circular")
        inv_2dx = 1.0 / (2.0 * self.dx)
        gx = (up[:, :, 2:, 1:-1, 1:-1] - up[:, :, :-2, 1:-1, 1:-1]) * inv_2dx
        gy = (up[:, :, 1:-1, 2:, 1:-1] - up[:, :, 1:-1, :-2, 1:-1]) * inv_2dx
        gz = (up[:, :, 1:-1, 1:-1, 2:] - up[:, :, 1:-1, 1:-1, :-2]) * inv_2dx
        return gx.squeeze(0).squeeze(0), gy.squeeze(0).squeeze(0), gz.squeeze(0).squeeze(0)


class Divergence3D(nn.Module):
    """3D Divergence with periodic BC."""
    def __init__(self, dx: float):
        super().__init__()
        self.dx = dx

    def forward(self, ux: Tensor, uy: Tensor, uz: Tensor) -> Tensor:
        if HAS_TRITON and ux.is_cuda:
            return triton_divergence_3d(ux, uy, uz, self.dx)
        inv_2dx = 1.0 / (2.0 * self.dx)
        if ux.dim() == 3:
            ux, uy, uz = ux.unsqueeze(0), uy.unsqueeze(0), uz.unsqueeze(0)
        ux_p = F.pad(ux, (1, 1, 1, 1, 1, 1), mode="circular")
        uy_p = F.pad(uy, (1, 1, 1, 1, 1, 1), mode="circular")
        uz_p = F.pad(uz, (1, 1, 1, 1, 1, 1), mode="circular")
        div = (
            (ux_p[:, 2:, 1:-1, 1:-1] - ux_p[:, :-2, 1:-1, 1:-1])
            + (uy_p[:, 1:-1, 2:, 1:-1] - uy_p[:, 1:-1, :-2, 1:-1])
            + (uz_p[:, 1:-1, 1:-1, 2:] - uz_p[:, 1:-1, 1:-1, :-2])
        ) * inv_2dx
        return div.squeeze(0)


class FusedNSStep(nn.Module):
    """
    Complete incompressible Navier-Stokes step (Chorin projection).
    Fused: advection + diffusion + forcing → pressure solve → projection.
    """
    def __init__(self, dx: float, nu: float, n_pressure_iter: int = 40):
        super().__init__()
        self.dx = dx
        self.nu = nu
        self.n_pressure_iter = n_pressure_iter

    def forward(self, u: Tensor, f: Tensor, dt: float) -> Tuple[Tensor, Tensor]:
        """
        Args:
            u: (B, 3, D, H, W) velocity
            f: (B, 3, D, H, W) forcing
            dt: time step
        Returns:
            u_next: (B, 3, D, H, W)
            p: (B, D, H, W)
        """
        if HAS_TRITON and u.is_cuda:
            return triton_ns_step(
                u, f, self.dx, dt, self.nu,
                n_pressure_iter=self.n_pressure_iter,
            )

        # Fallback: pure PyTorch
        B, _, D, H, W = u.shape

        # Step 1: u_star = u + dt * (-(u·∇)u + ν∇²u + f)
        lap = Laplacian3D(self.dx)
        grad = Gradient3D(self.dx)

        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        lap_u = torch.stack([
            lap(ux), lap(uy), lap(uz)
        ], dim=1)

        # Advection (central differences)
        g_ux = grad(ux)
        g_uy = grad(uy)
        g_uz = grad(uz)
        adv_x = -(ux.unsqueeze(1) * g_ux[0] + uy.unsqueeze(1) * g_ux[1] + uz.unsqueeze(1) * g_ux[2])
        adv_y = -(ux.unsqueeze(1) * g_uy[0] + uy.unsqueeze(1) * g_uy[1] + uz.unsqueeze(1) * g_uy[2])
        adv_z = -(ux.unsqueeze(1) * g_uz[0] + uy.unsqueeze(1) * g_uz[1] + uz.unsqueeze(1) * g_uz[2])
        adv = torch.cat([adv_x, adv_y, adv_z], dim=1)

        u_star = u + dt * (adv + self.nu * lap_u + f)

        # Step 2: Pressure solve
        div = Divergence3D(self.dx)
        rhs = div(u_star[:, 0], u_star[:, 1], u_star[:, 2]) / dt
        p = torch.zeros(B, D, H, W, device=u.device, dtype=u.dtype)
        # Simple Jacobi in PyTorch
        for _ in range(self.n_pressure_iter):
            p_pad = F.pad(p.unsqueeze(1), (1, 1, 1, 1, 1, 1), mode="circular")
            p = (
                p_pad[:, :, 2:, 1:-1, 1:-1] + p_pad[:, :, :-2, 1:-1, 1:-1]
                + p_pad[:, :, 1:-1, 2:, 1:-1] + p_pad[:, :, 1:-1, :-2, 1:-1]
                + p_pad[:, :, 1:-1, 1:-1, 2:] + p_pad[:, :, 1:-1, 1:-1, :-2]
                - self.dx ** 2 * rhs.unsqueeze(1)
            ).squeeze(1) / 6.0

        # Step 3: Projection
        g_p = grad(p)
        u_next = u_star - dt * torch.cat([
            g_p[0].unsqueeze(1), g_p[1].unsqueeze(1), g_p[2].unsqueeze(1)
        ], dim=1)

        return u_next, p


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════

def benchmark_all(
    device: str = "cuda",
    grid: int = 32,
    batch: int = 4,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    """
    Benchmark Triton vs PyTorch native for all kernels.
    Returns dict of timings.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return {}

    dev = torch.device(device)
    results = {}

    def time_fn(fn, *args, **kwargs):
        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1000  # ms

    # ── C^∞ Surrogates ──────────────────────────────────────────────
    x = torch.randn(batch, grid, grid, grid, device=dev, requires_grad=True)

    # soft_abs
    t_pt = time_fn(lambda: torch.sqrt(x * x + 1e-8))
    if HAS_TRITON:
        t_tr = time_fn(lambda: triton_soft_abs(x, 1e-4))
        results["soft_abs"] = {"pytorch_ms": t_pt, "triton_ms": t_tr,
                                "speedup": t_pt / max(t_tr, 1e-9)}

    # soft_floor
    t_pt = time_fn(lambda: 0.0 + F.softplus(x - 0.0, beta=50.0))
    if HAS_TRITON:
        t_tr = time_fn(lambda: triton_soft_floor(x, 0.0, 50.0))
        results["soft_floor"] = {"pytorch_ms": t_pt, "triton_ms": t_tr,
                                  "speedup": t_pt / max(t_tr, 1e-9)}

    # ── Navier-Stokes ─────────────────────────────────────────────────
    u = torch.randn(batch, 3, grid, grid, grid, device=dev)
    f = torch.randn_like(u)
    dx = 1.0 / grid
    dt = 1e-3
    nu = 1e-4

    # Laplacian
    t_pt = time_fn(lambda: F.pad(
        u[:, :1], (1, 1, 1, 1, 1, 1), mode="circular"
    ))
    if HAS_TRITON:
        t_tr = time_fn(lambda: triton_laplacian_3d(u[:, :1], dx))
        results["laplacian_3d"] = {"pytorch_ms": t_pt, "triton_ms": t_tr,
                                    "speedup": t_pt / max(t_tr, 1e-9)}

    # Fused NS step
    if HAS_TRITON:
        t_tr = time_fn(lambda: triton_fused_ns_step(u, f, dx, dt, nu))
        results["fused_ns_step"] = {"triton_ms": t_tr}

    # Complete NS step (with pressure solve)
    if HAS_TRITON:
        t_tr = time_fn(lambda: triton_ns_step(u, f, dx, dt, nu, n_pressure_iter=20))
        results["full_ns_step"] = {"triton_ms": t_tr}

    # ── Print results ─────────────────────────────────────────────────
    print("=" * 72)
    print(" SESI Fused Physics Kernels — Benchmark Results")
    print("=" * 72)
    print(f"  Device: {dev}, Grid: {grid}³, Batch: {batch}")
    print(f"  Warmup: {n_warmup}, Iterations: {n_iters}")
    print("-" * 72)
    for name, r in results.items():
        pt = r.get("pytorch_ms", float("nan"))
        tr = r.get("triton_ms", float("nan"))
        sp = r.get("speedup", float("nan"))
        print(f"  {name:<25s}  PT={pt:8.3f}ms  TR={tr:8.3f}ms  "
              f"speedup={sp:6.2f}x")
    print("=" * 72)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    """Verify gradient flow through all kernels."""
    print("=" * 72)
    print(" SESI Fused Physics Kernels — Smoke Test")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Triton available: {HAS_TRITON}")

    # ── C^∞ Surrogates ──────────────────────────────────────────────
    x = torch.randn(4, 16, 16, 16, device=device, requires_grad=True)

    sa = SoftAbs(eps=1e-4)
    out = sa(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"  [PASS] SoftAbs: grad norm = {x.grad.norm().item():.6e}")
    x.grad = None

    sf = SoftFloor(floor=0.0, beta=50.0)
    out = sf(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"  [PASS] SoftFloor: grad norm = {x.grad.norm().item():.6e}")
    x.grad = None

    ss = SoftSign(beta=10.0)
    out = ss(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"  [PASS] SoftSign: grad norm = {x.grad.norm().item():.6e}")
    x.grad = None

    # ── Fused SESI Barrier ──────────────────────────────────────────
    s = torch.randn(4, 16, 16, 16, device=device, requires_grad=True)
    de = torch.rand(4, 16, 16, 16, device=device)
    ssq = torch.rand(4, 16, 16, 16, device=device) + 0.1

    barrier = FusedSESIBarrier().to(device)
    barrier.train()
    out, prob = barrier(s, de, ssq, dt=0.01)
    loss = out.pow(2).mean()
    loss.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    print(f"  [PASS] FusedSESIBarrier: grad norm = {s.grad.norm().item():.6e}")

    # ── Navier-Stokes ─────────────────────────────────────────────────
    B, D, H, W = 2, 16, 16, 16
    u = torch.randn(B, 3, D, H, W, device=device, requires_grad=True)
    f = torch.randn_like(u)

    ns = FusedNSStep(dx=1.0 / D, nu=1e-4, n_pressure_iter=10).to(device)
    u_next, p = ns(u, f, dt=1e-3)
    loss = u_next.pow(2).mean() + p.pow(2).mean()
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    print(f"  [PASS] FusedNSStep: grad norm = {u.grad.norm().item():.6e}")
    print(f"         u_next shape: {tuple(u_next.shape)}, p shape: {tuple(p.shape)}")

    # Check divergence-free (approximate)
    with torch.no_grad():
        div = Divergence3D(1.0 / D)(u_next[:, 0], u_next[:, 1], u_next[:, 2])
        max_div = div.abs().max().item()
    print(f"         max |∇·u_next| = {max_div:.6e} (should be small)")

    print("=" * 72)
    print("  All smoke tests passed.")
    print("=" * 72)


if __name__ == "__main__":
    smoke_test()
    print()
    benchmark_all()
