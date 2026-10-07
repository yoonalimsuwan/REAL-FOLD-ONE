# =============================================================================
# Nanobot CRISPR-Cas Differentiable Kinetics Module (SESI)
# SUPER DNS ONE Cluster / ONE Ecosystem — Production Release
# =============================================================================
# Developer : PAI, Yoon A Limsuwan / MSPS NETWORK
# License   : MIT · Year 2026 · Version 2.0.0-PROD
# Target    : CRISPR-Cas9/12/13 RNP Nuclear Translocation & Cleavage Kinetics
#
# Native fully-differentiable · DDP · AMP · torch.compile · checkpointing
# =============================================================================
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler

logger = logging.getLogger("crispr_cas")


__all__ = [
    "CRISPRConfig",
    "NanobotCRISPRCasEditingModule",
    "initialize_distributed",
    "cleanup_distributed",
    "wrap_ddp",
    "train_step",
    "EMA",
    "save_checkpoint",
    "load_checkpoint",
    "enable_fast_math",
]


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class CRISPRConfig:
    # --- Geometry / time -----------------------------------------------------
    dx: float = 1e-5
    dt: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # --- Kinetic constants (initial values) ----------------------------------
    k_nuc_import_init: float = 0.015       # nuclear import rate
    v_max_cleavage_init: float = 2.5e-3    # max cleavage velocity
    k_m_affinity_init: float = 1.2e-4      # Michaelis affinity Km
    c1_off_target_init: float = 0.85       # Gumbel sharpness
    sigma_sq_init: float = 0.02            # quenched-noise variance

    learnable_kinetics: bool = True        # False → frozen buffers

    # --- Physics constants ---------------------------------------------------
    baseline_temp_K: float = 310.15        # 37 °C
    off_target_barrier: float = 3.0        # thermodynamic mismatch gap
    barrier_floor: float = 0.01
    barrier_clamp: float = 30.0            # caps inner exp → NaN-free

    # --- Numerics ------------------------------------------------------------
    eps: float = 1e-8
    softplus_beta: float = 20.0            # β for smooth positivity

    # --- Performance ---------------------------------------------------------
    compile: bool = False
    compile_mode: str = "max-autotune"
    use_checkpointing: bool = False

    # --- Distributed ---------------------------------------------------------
    distributed: bool = False
    backend: str = "nccl"
    seed: int = 42


# =============================================================================
# Distributed helpers
# =============================================================================
def initialize_distributed(cfg: CRISPRConfig) -> None:
    if not cfg.distributed or dist.is_initialized():
        return
    if cfg.device.startswith("cuda"):
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend=cfg.backend)
    logger.info("DDP init: rank=%d world=%d backend=%s",
                dist.get_rank(), dist.get_world_size(), cfg.backend)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _seeded_rand_like(x: torch.Tensor, step: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x.device)
    g.manual_seed(seed + 7_919 * get_rank() + 31 * step)
    return torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype)


# =============================================================================
# Main module
# =============================================================================
class NanobotCRISPRCasEditingModule(nn.Module):
    """
    Fully differentiable CRISPR-Cas editing kinetic solver.

    Couples:
        • Temperature-dependent Cas9/12/13 activation
        • ATP-dependent RNP nuclear translocation
        • Michaelis-Menten on-target DNA cleavage
        • Double-exponential off-target probability bound (No-Zeno math)

    All numerical operations are gradient-safe (softplus / sigmoid / clamped
    double-exp), batch-aware, and DDP/AMP/torch.compile/checkpoint ready.
    """

    def __init__(self, cfg: Optional[CRISPRConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CRISPRConfig()
        c = self.cfg
        dev = torch.device(c.device)

        def _make(name: str, init: float) -> torch.Tensor:
            t = torch.tensor([float(init)], device=dev, dtype=c.dtype)
            if c.learnable_kinetics:
                return nn.Parameter(t)
            self.register_buffer(name, t, persistent=False)
            return None

        # Learnable or frozen kinetic constants (dtype-aligned).
        if c.learnable_kinetics:
            self.k_nuc_import  = nn.Parameter(torch.tensor([c.k_nuc_import_init],
                                                           device=dev, dtype=c.dtype))
            self.v_max_cleavage = nn.Parameter(torch.tensor([c.v_max_cleavage_init],
                                                            device=dev, dtype=c.dtype))
            self.k_m_affinity  = nn.Parameter(torch.tensor([c.k_m_affinity_init],
                                                           device=dev, dtype=c.dtype))
            self.c1_off_target = nn.Parameter(torch.tensor([c.c1_off_target_init],
                                                           device=dev, dtype=c.dtype))
            self.sigma_sq      = nn.Parameter(torch.tensor([c.sigma_sq_init],
                                                           device=dev, dtype=c.dtype))
        else:
            self.register_buffer("k_nuc_import",
                torch.tensor([c.k_nuc_import_init], device=dev, dtype=c.dtype),
                persistent=False)
            self.register_buffer("v_max_cleavage",
                torch.tensor([c.v_max_cleavage_init], device=dev, dtype=c.dtype),
                persistent=False)
            self.register_buffer("k_m_affinity",
                torch.tensor([c.k_m_affinity_init], device=dev, dtype=c.dtype),
                persistent=False)
            self.register_buffer("c1_off_target",
                torch.tensor([c.c1_off_target_init], device=dev, dtype=c.dtype),
                persistent=False)
            self.register_buffer("sigma_sq",
                torch.tensor([c.sigma_sq_init], device=dev, dtype=c.dtype),
                persistent=False)

        # Per-rank step counter (never synced, keeps state_dict portable).
        self.register_buffer("step", torch.tensor(0, dtype=torch.long),
                             persistent=False)

        # Optional compilation of the inner implementation.
        if c.compile:
            logger.info("torch.compile(mode=%s) on _forward_impl", c.compile_mode)
            self._impl: Callable[..., Any] = torch.compile(
                self._forward_impl, mode=c.compile_mode)
        else:
            self._impl = self._forward_impl

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _smooth_pos(x: torch.Tensor, floor: float, beta: float) -> torch.Tensor:
        """
        Smooth, strictly-positive projection:
            y ≈ max(x, floor)   with C^∞ behaviour and non-zero gradient
            for all x within (floor − O(1/β), floor + O(1/β)).
        """
        return F.softplus(x - floor, beta=beta) + floor

    # ------------------------------------------------------- core computation
    def _forward_impl(
        self,
        rnp_concentration: torch.Tensor,    # [B,1,Z,Y,X]
        target_dna_density: torch.Tensor,   # [B,1,Z,Y,X]
        temperature_field: torch.Tensor,    # [B,1,Z,Y,X]
        local_atp_energy: torch.Tensor,     # [B,1,Z,Y,X]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        c = self.cfg
        step = int(self.step.item())

        # ---------------------------------------------------------------------
        # 1. Temperature-dependent Cas activation
        #    Smooth sigmoid saturation in (0,1). Non-zero gradient everywhere.
        # ---------------------------------------------------------------------
        thermal_activation = torch.sigmoid(
            (temperature_field - c.baseline_temp_K) * 0.5)
        active_rnp = rnp_concentration * (1.0 + thermal_activation)

        # ---------------------------------------------------------------------
        # 2. ATP-dependent nuclear translocation (active transport)
        #    atp_gate ∈ (0,1) via sigmoid; positivity via smooth softplus.
        # ---------------------------------------------------------------------
        atp_gate = torch.sigmoid(local_atp_energy - 1.0)
        nuc_flux = self.k_nuc_import * active_rnp * atp_gate
        nuclear_rnp = self._smooth_pos(
            active_rnp - nuc_flux * c.dt, floor=c.eps,
            beta=c.softplus_beta,
        )

        # ---------------------------------------------------------------------
        # 3. Michaelis-Menten on-target cleavage
        #       V = Vmax · [RNP] · [DNA] / (Km + [RNP])
        # ---------------------------------------------------------------------
        denom = self.k_m_affinity + nuclear_rnp + c.eps
        cleavage_rate = (self.v_max_cleavage * nuclear_rnp
                         * target_dna_density) / denom

        # FIX: softplus(x - eps, beta)+eps carries a bias of up to ln2/beta (~0.035 at
        # beta=20): with ZERO RNP and target < ~0.2 it returned NEGATIVE successful_edits.
        # Exact first-order update instead: edits = DNA * (1 - exp(-k_eff*dt)) in [0, DNA].
        # NOTE: `edited_dna` is the REMAINING unedited target (name kept for API compat).
        k_eff = self.v_max_cleavage * nuclear_rnp / denom
        successful_edits = target_dna_density * (-torch.expm1(-k_eff * c.dt))
        edited_dna = target_dna_density - successful_edits

        # ---------------------------------------------------------------------
        # 4. Off-target bound (double-exponential No-Zeno)
        #       P_off ≤ exp(−C1 · exp(ΔE / (σ² · dt)))
        #    ΔE smoothed via softplus; inner exp clamped to avoid NaN / Inf.
        # ---------------------------------------------------------------------
        delta_e = self._smooth_pos(
            c.off_target_barrier - local_atp_energy, floor=c.barrier_floor,
            beta=c.softplus_beta,
        )
        inner = torch.clamp(
            delta_e / (self.sigma_sq * c.dt + c.eps),
            min=0.0, max=c.barrier_clamp,
        )
        log_off_target_prob = -self.c1_off_target * torch.exp(inner)
        off_target_prob = torch.exp(log_off_target_prob)
        off_target_damage = off_target_prob * nuclear_rnp * c.dt

        # ---------------------------------------------------------------------
        # 5. Telemetry (all per-voxel; consumers reduce as needed)
        # ---------------------------------------------------------------------
        metrics: Dict[str, torch.Tensor] = {
            "cleavage_flux":          cleavage_rate,
            "thermal_activation_map": thermal_activation,
            "thermal_enhancement":    thermal_activation,       # legacy key
            "nuclear_rnp":            nuclear_rnp,
            "off_target_risk_map":    off_target_damage,
            "off_target_prob":        off_target_prob,
            "log_off_target_prob":    log_off_target_prob,       # grad-safe
        }

        with torch.no_grad():
            self.step.add_(1)

        return edited_dna, successful_edits, metrics

    # ---------------------------------------------------------------- forward
    def forward(self, *args: Any, **kwargs: Any
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cfg.use_checkpointing and self.training:
            return checkpoint(self._impl, *args,
                              use_reentrant=False, **kwargs)
        return self._impl(*args, **kwargs)


# =============================================================================
# DDP wrapper & training utilities
# =============================================================================
def wrap_ddp(model: nn.Module, cfg: CRISPRConfig,
             find_unused_parameters: bool = False) -> nn.Module:
    """
    Wrap in DDP when cfg.distributed is True. If `learnable_kinetics=False`
    the model has zero parameters — DDP is a no-op but harmless.
    """
    if not cfg.distributed:
        return model
    if not dist.is_initialized():
        raise RuntimeError("Call initialize_distributed(cfg) first.")
    local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
    if torch.cuda.is_available():
        model = model.to(torch.device(f"cuda:{local_rank}"))
    return DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=True,
    )


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]],
                      torch.Tensor],
    grad_accum_steps: int = 1,
    micro_step: int = 0,
    max_grad_norm: float = 1.0,
    amp_dtype: Optional[torch.dtype] = None,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    """One AMP-aware training iteration with grad accumulation + clipping."""
    model.train()
    enabled = amp_dtype is not None and torch.cuda.is_available()

    with autocast(device_type="cuda", dtype=amp_dtype, enabled=enabled):
        out = model(**batch)
        loss = loss_fn(out) / grad_accum_steps

    if scaler is not None and enabled:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    metrics = {"loss": float(loss.detach()) * grad_accum_steps}

    if (micro_step + 1) % grad_accum_steps == 0:
        if scaler is not None and enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

    return metrics


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scaler: Optional[GradScaler], epoch: int, step: int,
                    ema: Optional[EMA] = None) -> None:
    if not is_main_process():
        return
    raw = model.module if isinstance(model, DDP) else model
    state: Dict[str, Any] = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch, "step": step,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        state["ema"] = ema.shadow
    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scaler: Optional[GradScaler] = None,
                    ema: Optional[EMA] = None) -> Dict[str, int]:
    raw = model.module if isinstance(model, DDP) else model
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        device = next(raw.parameters()).device if any(
            p.requires_grad for p in raw.parameters()) else torch.device("cpu")
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
    return {"epoch": ckpt.get("epoch", 0), "step": ckpt.get("step", 0)}


def enable_fast_math() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Demo (torchrun --standalone --nproc_per_node=N crispr_cas_v2.py)
# =============================================================================
def _demo() -> None:
    cfg = CRISPRConfig(
        dt=1e-4, dx=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        distributed=False,
        learnable_kinetics=True,
        compile=False,
    )
    enable_fast_math()
    initialize_distributed(cfg)

    model = wrap_ddp(NanobotCRISPRCasEditingModule(cfg).to(cfg.device), cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-4)
    scaler = GradScaler("cuda", enabled=False)

    B, Z, Y, X = 2, 16, 16, 16
    dev, dt = cfg.device, cfg.dtype
    batch = {
        "rnp_concentration":  torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.5,
        "target_dna_density": torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.8 + 0.2,
        "temperature_field":  torch.full((B, 1, Z, Y, X), 313.15, device=dev, dtype=dt)
                              + torch.randn(B, 1, Z, Y, X, device=dev, dtype=dt) * 0.5,
        "local_atp_energy":   torch.rand(B, 1, Z, Y, X, device=dev, dtype=dt) * 2.0,
    }

    def loss_fn(out):
        edited_dna, edits, metrics = out
        # Supervise: maximise on-target edits, minimise off-target damage.
        return (-edits.mean()
                + 0.5 * metrics["off_target_risk_map"].mean()
                + 0.01 * metrics["log_off_target_prob"].mean())

    for step in range(5):
        m = train_step(model, opt, scaler, batch, loss_fn,
                       amp_dtype=None, micro_step=step)
        if is_main_process():
            print(f"[step {step}] loss={m['loss']:.6f}")

    cleanup_distributed()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    _demo()
