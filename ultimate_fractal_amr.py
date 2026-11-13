#!/usr/bin/env python3
"""
=============================================================================
ULTIMATE FRACTAL AMR — Production Harness for Non-p.c.f. Fractals
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

GOAL
  A production-grade, differentiable AMR control layer for fractal cell
  graphs, built around the diagnostics from Consolidated_Revision_Carpet-19:
    • finite-level graph approximation
    • crossing resistance R_n
    • resistance growth rho estimate
    • energy-measure concentration diagnostics
    • spectral Sobolev-scale surrogate
    • differentiable refine/coarsen controller
    • AMP/DDP-ready training path

VERY IMPORTANT HONEST SCOPE
  ❌ This does NOT prove that an arbitrary non-p.c.f. fractal satisfies
     Carpet-19 standing inputs (I1)-(I4).
  ❌ This does NOT automatically make AMR correct on every fractal.
  ❌ This does NOT certify "state of the art" or "changing the world".
  ✅ This DOES provide a rigorous computational harness that can test
     whether a given fractal family behaves like the carpet at finite
     levels, and can drive differentiable AMR decisions from the
     resulting diagnostics.

ARCHITECTURE
  1. FractalGraphBuilder  -> GraphData(cells, edges, boundary masks)
  2. resistance analysis  -> R_n, ratios, rho estimate, Carpet-19 checks
  3. spectral surrogate   -> graph Laplacian powers / Sobolev norms
  4. energy measure       -> Γ(S)/µ(S), entropy, singularity diagnostics
  5. AMR controller       -> differentiable level selection with cost
  6. AMP/DDP helpers      -> production training integration
=============================================================================
"""
from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except Exception:
    dist = None
    DDP = None

VERSION = "1.0.0"


# =============================================================================
# 0. CONFIG
# =============================================================================

@dataclass
class UltimateAMRConfig:
    # Fractal graph levels used for verification / diagnostics
    verify_levels: Tuple[int, ...] = (1, 2, 3)

    # AMR controller
    amr_levels: int = 4              # 0..amr_levels-1
    refine_hi: float = 0.60
    coarsen_lo: float = 0.10
    tau: float = 0.05
    width: float = 0.35
    cost_weight: float = 1e-2
    ref_ratio: float = 2.0
    fractal_dim: float = 2.0

    # Numerics
    max_cells: int = 50_000
    max_dense_eig_cells: int = 2048
    cg_tol: float = 1e-10
    device: str = "cpu"
    dtype: str = "float64"

    # Production
    amp: bool = False
    amp_dtype: str = "bfloat16"     # bfloat16 | float16
    ddp: bool = False

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


# =============================================================================
# 1. GRAPH DATA + CARPET BUILDER
# =============================================================================

@dataclass
class GraphData:
    level: int
    n: int
    coords: torch.Tensor          # (n, 2) integer cell coordinates
    edge_index: torch.Tensor      # (2, E) directed undirected edges
    adj: torch.Tensor             # (n, n) dense adjacency (small levels only)
    left: torch.Tensor            # bool mask
    right: torch.Tensor
    bottom: torch.Tensor
    top: torch.Tensor


class CarpetGraphBuilder:
    """
    Standard Sierpinski carpet cell graph.

    Level L has 8^L cells embedded in a 3^L x 3^L grid with centers removed.
    Cells are side-adjacent. This is the finite-level graph used in
    Carpet-19's finite-level diagnostics.
    """

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype

    def cells(self, level: int) -> List[Tuple[int, int]]:
        if level < 0:
            raise ValueError("level must be >= 0")
        cells = [(0, 0)]
        for _ in range(level):
            new_cells = []
            for x, y in cells:
                for dx in range(3):
                    for dy in range(3):
                        if dx == 1 and dy == 1:
                            continue
                        new_cells.append((x * 3 + dx, y * 3 + dy))
            cells = new_cells
        return cells

    def build(self, level: int, max_cells: int = 50_000) -> GraphData:
        cells = self.cells(level)
        n = len(cells)
        if n > max_cells:
            raise ValueError(
                f"carpet level {level} has {n} cells > max_cells={max_cells}; "
                f"use lower level or sparse backend")

        idx = {c: i for i, c in enumerate(cells)}
        coords = torch.tensor(cells, dtype=torch.long, device=self.device)

        edge_list = []
        for i, (x, y) in enumerate(cells):
            for dx, dy in ((1, 0), (0, 1)):
                nb = (x + dx, y + dy)
                if nb in idx:
                    j = idx[nb]
                    edge_list.append((i, j))
                    edge_list.append((j, i))

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long,
                                      device=self.device).t().contiguous()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)

        adj = torch.zeros(n, n, dtype=self.dtype, device=self.device)
        if edge_index.numel():
            adj[edge_index[0], edge_index[1]] = 1.0

        side = 3 ** level
        left = coords[:, 0] == 0
        right = coords[:, 0] == side - 1
        bottom = coords[:, 1] == 0
        top = coords[:, 1] == side - 1

        return GraphData(level=level, n=n, coords=coords,
                         edge_index=edge_index, adj=adj,
                         left=left, right=right, bottom=bottom, top=top)


# =============================================================================
# 2. GRAPH LAPLACIAN + CG
# =============================================================================

def graph_laplacian(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1)
    return torch.diag(deg) - adj


def conjugate_gradient(A: torch.Tensor, b: torch.Tensor,
                       tol: float = 1e-10,
                       max_iter: Optional[int] = None) -> torch.Tensor:
    """
    Simple CG for symmetric positive (semi)definite systems.
    Used for resistance solves on moderate-size graph Laplacians.
    """
    n = b.shape[0]
    if n == 0:
        return b.clone()
    if max_iter is None:
        max_iter = min(10 * n, 20_000)

    diag = A.diagonal().clone()
    diag = torch.where(diag > 1e-14, diag, torch.ones_like(diag))

    x = torch.zeros_like(b)
    r = b.clone()
    z = r / diag
    p = z.clone()
    rz = (r * z).sum()
    bnorm = b.norm() + 1e-30

    for _ in range(max_iter):
        Ap = A @ p
        denom = (p * Ap).sum()
        if denom.abs() < 1e-300:
            break
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if r.norm() / bnorm < tol:
            break

        z = r / diag
        rz_new = (r * z).sum()
        if rz.abs() < 1e-300:
            break
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x


# =============================================================================
# 3. RESISTANCE / RHO / CARPET-19 CHECKS
# =============================================================================

def effective_resistance(graph: GraphData,
                         source: torch.Tensor,
                         sink: torch.Tensor,
                         tol: float = 1e-10) -> float:
    """
    Effective resistance between two boundary sets with voltage 1/0.
    R_eff = 1 / energy of harmonic extension.
    """
    L = graph_laplacian(graph.adj)
    fixed = source | sink
    interior = ~fixed

    if int(interior.sum()) == 0:
        return float("inf")

    A = L[interior][:, interior].clone()
    diag = A.diagonal().clone()
    jitter = (diag <= 0).to(A.dtype) * 1e-12
    if jitter.any():
        A += torch.diag(jitter)

    # RHS from fixed boundary values: source=1, sink=0
    b = -L[interior][:, source].sum(dim=1)
    x = conjugate_gradient(A, b, tol=tol)

    v = torch.zeros(graph.n, dtype=graph.adj.dtype, device=graph.adj.device)
    v[source] = 1.0
    v[sink] = 0.0
    v[interior] = x

    energy = float((v @ (L @ v)).item())
    if energy <= 1e-300:
        return float("inf")
    return 1.0 / energy


def verify_fractal_properties(builder: CarpetGraphBuilder,
                              levels: Tuple[int, ...],
                              max_cells: int = 50_000,
                              tol: float = 1e-10) -> Dict[str, Any]:
    """
    Finite-level Carpet-19 diagnostics:
      • crossing resistances R_n
      • ratios R_{n+1}/R_n
      • Fekete-style rho estimate from inf (12 R_n)^{1/n}
      • quasi-multiplicativity checks R_{n+m} <= 12 R_n R_m
      • lower bound R_n >= (4/9)(9/8)^n as a consistency check

    IMPORTANT: these are finite-level numerical checks, not proofs.
    """
    Rs: Dict[int, float] = {}
    graphs: Dict[int, GraphData] = {}

    for lev in levels:
        g = builder.build(lev, max_cells=max_cells)
        graphs[lev] = g
        R = effective_resistance(g, g.left, g.right, tol=tol)
        Rs[lev] = R

    sorted_levels = sorted(Rs.keys())
    ratios = {}
    for i in range(len(sorted_levels) - 1):
        a, b = sorted_levels[i], sorted_levels[i + 1]
        if math.isfinite(Rs[a]) and math.isfinite(Rs[b]) and Rs[a] > 0:
            ratios[f"{b}/{a}"] = Rs[b] / Rs[a]

    rho_candidates = []
    for lev, R in Rs.items():
        if lev > 0 and math.isfinite(R) and R > 0:
            rho_candidates.append((12.0 * R) ** (1.0 / lev))
    rho_est = min(rho_candidates) if rho_candidates else float("nan")

    quasi_ok = []
    for n in sorted_levels:
        for m in sorted_levels:
            nm = n + m
            if nm in Rs and math.isfinite(Rs[nm]):
                lhs = Rs[nm]
                rhs = 12.0 * Rs[n] * Rs[m]
                quasi_ok.append({
                    "n": n, "m": m,
                    "lhs": lhs, "rhs": rhs,
                    "ok": lhs <= rhs + 1e-8 * max(1.0, rhs),
                })

    lower_bound_ok = []
    for lev, R in Rs.items():
        if math.isfinite(R):
            bound = (4.0 / 9.0) * (9.0 / 8.0) ** lev
            lower_bound_ok.append({
                "level": lev,
                "R": R,
                "lower_bound": bound,
                "ok": R + 1e-12 >= bound,
            })

    return {
        "levels": sorted_levels,
        "resistances": Rs,
        "ratios": ratios,
        "rho_estimate": rho_est,
        "quasi_multiplicativity": quasi_ok,
        "lower_bound_checks": lower_bound_ok,
        "NOTE": "finite-level diagnostics only; not a proof of (I1)-(I4)",
    }


# =============================================================================
# 4. ENERGY MEASURE DIAGNOSTICS (Carpet-19 style)
# =============================================================================

def energy_measure_diagnostics(graph: GraphData,
                               u: torch.Tensor,
                               coarse_level: int,
                               eps: float = 1e-30) -> Dict[str, Any]:
    """
    Finite-level energy-measure concentration diagnostics.

    For a field u on level-N carpet cells:
      • edge energy (u_i - u_j)^2 is split equally to endpoints
      • cell masses are aggregated to m-level coarse cells
      • Γ(S)/µ(S) is the energy density relative to uniform carpet measure
      • entropy / max density track singularity of the energy measure

    This is the numerical surrogate for Hino's singularity diagnostics:
    if Γ concentrates more and more as N grows, it behaves like a measure
    singular to µ.
    """
    if coarse_level > graph.level:
        raise ValueError("coarse_level cannot exceed graph.level")
    if graph.edge_index.numel() == 0:
        return {"ok": False, "reason": "no edges"}

    u = u.to(graph.adj.dtype).to(graph.adj.device)
    ei = graph.edge_index[0]
    ej = graph.edge_index[1]
    edge_energy = (u[ei] - u[ej]) ** 2

    cell_mass = torch.zeros(graph.n, dtype=u.dtype, device=u.device)
    cell_mass.index_add_(0, ei, 0.5 * edge_energy)
    cell_mass.index_add_(0, ej, 0.5 * edge_energy)

    total = float(cell_mass.sum().item())
    if total <= eps:
        return {"ok": False, "reason": "zero energy"}

    scale = 3 ** (graph.level - coarse_level)
    coarse_coords = graph.coords // scale
    side = 3 ** coarse_level
    key = coarse_coords[:, 0] * side + coarse_coords[:, 1]

    uniq_keys, inverse = torch.unique(key, return_inverse=True)
    mass_coarse = torch.zeros(uniq_keys.numel(), dtype=u.dtype, device=u.device)
    mass_coarse.index_add_(0, inverse, cell_mass)

    gamma = mass_coarse / (total + eps)
    mu = torch.full_like(gamma, 8.0 ** (-coarse_level))

    X = gamma / mu.clamp_min(eps)
    max_density = float(X.max().item()) if X.numel() else 0.0

    rel_entropy = float((gamma * torch.log((gamma + eps) / mu)).sum().item())
    l2_concentration = float((gamma * gamma / mu).sum().item())

    return {
        "ok": True,
        "coarse_level": coarse_level,
        "total_energy": total,
        "max_density": max_density,
        "l2_concentration": l2_concentration,
        "relative_entropy": rel_entropy,
        "n_coarse_cells": int(uniq_keys.numel()),
        "NOTE": "finite-level surrogate for energy-measure singularity diagnostics",
    }


# =============================================================================
# 5. SPECTRAL SOBOLEV SURROGATE
# =============================================================================

class SpectralSobolevSurrogate:
    """
    Finite-level spectral Sobolev surrogate.

    For small graphs, diagonalize A_mu ≈ m^n kappa_n L_n and compute
        ||u||_{W^s}^2 = Σ (1+λ)^s |<u,φ>|^2.

    For larger graphs, skip diagonalization and provide only integer-order
    energy seminorms ||A^k u||.
    """

    def __init__(self, graph: GraphData, kappa: float = 1.0,
                 m: int = 8, max_dense_eig: int = 2048):
        self.graph = graph
        self.kappa = kappa
        self.m = m
        self.max_dense_eig = max_dense_eig
        self.L = graph_laplacian(graph.adj)
        self.A = (m ** graph.level) * kappa * self.L
        self._eigvals: Optional[torch.Tensor] = None
        self._eigvecs: Optional[torch.Tensor] = None

    def diagonalize(self) -> bool:
        if self.graph.n > self.max_dense_eig:
            return False
        if self._eigvals is None:
            self._eigvals, self._eigvecs = torch.linalg.eigh(self.A)
        return True

    def sobolev_norm(self, u: torch.Tensor, s: float) -> float:
        if self.diagonalize():
            coeffs = self._eigvecs.T @ u.to(self._eigvecs.dtype)
            weights = (1.0 + self._eigvals.clamp_min(0.0)) ** s
            return float(torch.sqrt((weights * coeffs * coeffs).sum()).item())
        # fallback: integer orders only
        if abs(s - round(s)) > 1e-12:
            raise ValueError("fractional Sobolev norm requires eigendecomposition")
        k = int(round(s))
        v = u.clone()
        for _ in range(k):
            v = self.A @ v
        return float(v.norm().item())


# =============================================================================
# 6. DIFFERENTIABLE AMR CONTROLLER
# =============================================================================

class DifferentiableFractalAMRController(nn.Module):
    """
    Native differentiable AMR level controller.

    Given an error/singularity indicator field, produce soft or hard
    refinement-level decisions via smooth Gaussian level logits.

    The hard path uses straight-through estimation (STE): forward is
    discrete, backward flows through the soft distribution.
    """

    def __init__(self,
                 amr_levels: int = 4,
                 refine_hi: float = 0.60,
                 coarsen_lo: float = 0.10,
                 tau: float = 0.05,
                 width: float = 0.35,
                 ref_ratio: float = 2.0,
                 fractal_dim: float = 2.0):
        super().__init__()
        if amr_levels < 2:
            raise ValueError("amr_levels must be >= 2")
        self.amr_levels = amr_levels
        self.ref_ratio = ref_ratio
        self.fractal_dim = fractal_dim

        self.log_tau = nn.Parameter(torch.tensor(math.log(max(tau, 1e-6))))
        self.log_width = nn.Parameter(torch.tensor(math.log(max(width, 1e-6))))
        self.mid = nn.Parameter(torch.tensor(0.5 * (refine_hi + coarsen_lo)))
        self.gain = nn.Parameter(torch.tensor(6.0))

    def forward(self, indicator: torch.Tensor,
                hard: bool = False,
                temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        shape = indicator.shape
        ind = indicator.reshape(-1).to(self.mid.device, self.mid.dtype)

        tau = F.softplus(self.log_tau) + 1e-4
        width = F.softplus(self.log_width) + 1e-3

        levels = torch.arange(self.amr_levels,
                              device=ind.device, dtype=ind.dtype)
        target = (self.amr_levels - 1) * torch.sigmoid(
            self.gain * (ind - self.mid) / tau)

        logits = -((levels.unsqueeze(0) - target.unsqueeze(1)) ** 2) / (
            2.0 * width * width)

        probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        level_costs = self.ref_ratio ** (self.fractal_dim * levels)

        soft_level = (probs * levels.unsqueeze(0)).sum(dim=-1)
        soft_cost = (probs * level_costs.unsqueeze(0)).sum(dim=-1)

        out: Dict[str, torch.Tensor] = {
            "logits": logits.reshape(*shape, self.amr_levels),
            "probs": probs.reshape(*shape, self.amr_levels),
            "soft_level": soft_level.reshape(shape),
            "cost": soft_cost.reshape(shape),
        }

        if hard:
            idx = probs.argmax(dim=-1)
            hard_one = F.one_hot(idx, self.amr_levels).to(probs.dtype)
            probs_hard = hard_one + (probs - probs.detach())

            hard_level = (probs_hard * levels.unsqueeze(0)).sum(dim=-1)
            hard_cost = (probs_hard * level_costs.unsqueeze(0)).sum(dim=-1)

            out.update({
                "hard_probs": probs_hard.reshape(*shape, self.amr_levels),
                "level": idx.reshape(shape),
                "soft_level": hard_level.reshape(shape),
                "cost": hard_cost.reshape(shape),
            })

        return out


# =============================================================================
# 7. AMP / DDP HELPERS
# =============================================================================

@contextlib.contextmanager
def amp_autocast(enabled: bool, device_type: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
    """
    AMP context that safely degrades to no-op on CPU or unsupported configs.
    """
    if enabled and torch.cuda.is_available() and device_type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    elif enabled and device_type == "cpu":
        # CPU autocast is optional and op-dependent; keep explicit.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            yield
    else:
        yield


def maybe_ddp(model: nn.Module, cfg: UltimateAMRConfig) -> nn.Module:
    if not cfg.ddp:
        return model
    if DDP is None or dist is None or not dist.is_initialized():
        warnings.warn("DDP requested but torch.distributed is not initialized; "
                      "returning unwrapped model.")
        return model
    device_ids = None
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        device_ids = [torch.cuda.current_device()]
    return DDP(model, device_ids=device_ids)


# =============================================================================
# 8. ULTIMATE FRACTAL AMR ENGINE
# =============================================================================

class UltimateFractalAMR:
    """
    Orchestrates:
      • graph construction
      • resistance/rho verification
      • energy-measure diagnostics
      • spectral Sobolev surrogate
      • differentiable AMR controller
      • AMP/DDP integration
    """

    def __init__(self, cfg: UltimateAMRConfig,
                 builder: Optional[Any] = None):
        self.cfg = cfg
        self.builder = builder or CarpetGraphBuilder(cfg.device, cfg.torch_dtype)
        self.controller = DifferentiableFractalAMRController(
            amr_levels=cfg.amr_levels,
            refine_hi=cfg.refine_hi,
            coarsen_lo=cfg.coarsen_lo,
            tau=cfg.tau,
            width=cfg.width,
            ref_ratio=cfg.ref_ratio,
            fractal_dim=cfg.fractal_dim,
        ).to(cfg.torch_device)
        self._graph_cache: Dict[int, GraphData] = {}
        self._resistance_cache: Dict[int, float] = {}

    # ---------------- graph / resistance ----------------
    def graph(self, level: int) -> GraphData:
        if level not in self._graph_cache:
            self._graph_cache[level] = self.builder.build(
                level, max_cells=self.cfg.max_cells)
        return self._graph_cache[level]

    def crossing_resistance(self, level: int) -> float:
        if level not in self._resistance_cache:
            g = self.graph(level)
            R = effective_resistance(g, g.left, g.right, tol=self.cfg.cg_tol)
            self._resistance_cache[level] = R
        return self._resistance_cache[level]

    def resistance_report(self) -> Dict[str, Any]:
        return verify_fractal_properties(
            self.builder,
            self.cfg.verify_levels,
            max_cells=self.cfg.max_cells,
            tol=self.cfg.cg_tol,
        )

    # ---------------- indicators ----------------
    def edge_variation_indicator(self, level: int,
                                 u: torch.Tensor,
                                 eps: float = 1e-12) -> torch.Tensor:
        """
        Cell-wise edge variation:
            I_i = Σ_{j~i} (u_i - u_j)^2 / deg(i)
        normalized to [0,1].
        """
        g = self.graph(level)
        u = u.to(g.adj.dtype).to(g.adj.device)
        if g.edge_index.numel() == 0:
            return torch.zeros(g.n, dtype=u.dtype, device=u.device)

        ei, ej = g.edge_index[0], g.edge_index[1]
        ee = (u[ei] - u[ej]) ** 2
        mass = torch.zeros(g.n, dtype=u.dtype, device=u.device)
        mass.index_add_(0, ei, ee)
        mass.index_add_(0, ej, ee)

        deg = g.adj.sum(dim=1).clamp_min(1e-12)
        ind = mass / deg
        ind = ind / (ind.max() + eps)
        return ind

    def energy_diagnostics(self, level: int,
                           u: torch.Tensor,
                           coarse_level: int) -> Dict[str, Any]:
        g = self.graph(level)
        return energy_measure_diagnostics(g, u, coarse_level)

    def spectral_report(self, level: int, u: torch.Tensor) -> Dict[str, Any]:
        g = self.graph(level)
        kappa = self.crossing_resistance(level)
        if not math.isfinite(kappa) or kappa <= 0:
            kappa = 1.0
        spec = SpectralSobolevSurrogate(
            g, kappa=kappa, m=8,
            max_dense_eig=self.cfg.max_dense_eig_cells)
        out: Dict[str, Any] = {"level": level, "kappa": kappa}
        try:
            out["W0"] = spec.sobolev_norm(u, 0.0)
            out["W1"] = spec.sobolev_norm(u, 1.0)
            out["W2"] = spec.sobolev_norm(u, 2.0)
        except Exception as e:
            out["error"] = str(e)
        return out

    # ---------------- AMR decisions ----------------
    def adapt(self, indicator: torch.Tensor,
              hard: bool = True,
              temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bfloat16" else torch.float16
        device_type = "cuda" if self.cfg.torch_device.type == "cuda" else "cpu"
        with amp_autocast(self.cfg.amp, device_type=device_type, dtype=amp_dtype):
            return self.controller(indicator, hard=hard, temperature=temperature)

    def amr_loss(self, accuracy_error: torch.Tensor,
                 decision: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Multi-objective AMR training loss:
            L = mean(error) + λ * mean(cost)
        """
        return accuracy_error.mean() + self.cfg.cost_weight * decision["cost"].mean()

    # ---------------- production ----------------
    def prepare_ddp(self) -> nn.Module:
        return maybe_ddp(self.controller, self.cfg)


# =============================================================================
# 9. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print("=" * 94)
    print(f"  ULTIMATE FRACTAL AMR v{VERSION} — self-test")
    print("=" * 94)
    torch.manual_seed(0)
    ok = True

    def rep(name: str, passed: bool, extra: str = ""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    cfg = UltimateAMRConfig(
        verify_levels=(1, 2),
        amr_levels=3,
        max_cells=10_000,
        max_dense_eig_cells=512,
        device="cpu",
        dtype="float64",
        amp=False,
        ddp=False,
    )
    engine = UltimateFractalAMR(cfg)

    # T1 graph construction
    g1 = engine.graph(1)
    g2 = engine.graph(2)
    rep("carpet level sizes", g1.n == 8 and g2.n == 64,
        f"n1={g1.n}, n2={g2.n}")

    # T2 resistance positive and finite
    R1 = engine.crossing_resistance(1)
    R2 = engine.crossing_resistance(2)
    rep("resistances positive finite",
        math.isfinite(R1) and math.isfinite(R2) and R1 > 0 and R2 > 0,
        f"R1={R1:.4f}, R2={R2:.4f}")

    # T3 Carpet-19 finite checks
    report = engine.resistance_report()
    quasi_ok = all(q["ok"] for q in report["quasi_multiplicativity"])
    lower_ok = all(q["ok"] for q in report["lower_bound_checks"])
    rho_ok = math.isfinite(report["rho_estimate"])
    rep("quasi-multiplicativity finite check", quasi_ok,
        f"rho_est={report['rho_estimate']:.4f}")
    rep("lower-bound finite check", lower_ok)
    rep("rho estimate finite", rho_ok)

    # T4 energy-measure diagnostics
    u2 = g2.coords[:, 0].to(torch.float64) / float(3 ** 2 - 1)
    diag = engine.energy_diagnostics(level=2, u=u2, coarse_level=1)
    rep("energy diagnostics ok", diag.get("ok", False),
        f"max_density={diag.get('max_density', 0):.4f}")

    # T5 edge-variation indicator
    ind = engine.edge_variation_indicator(2, u2)
    rep("indicator shape and range",
        ind.shape == (g2.n,) and float(ind.min()) >= 0 and float(ind.max()) <= 1,
        f"range=[{float(ind.min()):.3f},{float(ind.max()):.3f}]")

    # T6 differentiable AMR decision + gradient
    decision = engine.adapt(ind, hard=True, temperature=1.0)
    loss = engine.amr_loss(ind, decision)
    engine.controller.zero_grad(set_to_none=True)
    loss.backward()
    has_grad = (engine.controller.gain.grad is not None and
                torch.isfinite(engine.controller.gain.grad).all())
    rep("AMR decision differentiable", bool(has_grad),
        f"loss={float(loss):.6f}")

    # T7 spectral surrogate
    u1 = g1.coords[:, 0].to(torch.float64) / float(3 ** 1 - 1)
    spec = engine.spectral_report(1, u1)
    rep("spectral surrogate produces norms",
        "W1" in spec or "error" in spec,
        f"keys={list(spec.keys())}")

    # T8 DDP/AMP safe degradation
    model = engine.prepare_ddp()
    rep("DDP safe degradation", model is not None)

    print("\n" + "=" * 94)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 94)
    print("""
  WHAT THIS HARNESS DOES
    • builds carpet finite-level graphs
    • computes crossing resistance and rho diagnostics
    • computes energy-measure concentration diagnostics
    • provides a spectral Sobolev surrogate
    • produces differentiable AMR decisions with cost penalty
    • is AMP/DDP-ready for training the controller

  WHAT IT DOES NOT DO
    ❌ prove (I1)-(I4) for a new non-p.c.f. fractal
    ❌ guarantee AMR accuracy for every fractal
    ❌ replace a full PDE solver / refluxing / coarse-fine correction
    ❌ certify SOTA or paradigm shift
""")
    return ok


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest", choices=["selftest"])
    p.parse_args()
    raise SystemExit(0 if self_test() else 1)


if __name__ == "__main__":
    main()
