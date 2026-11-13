#!/usr/bin/env python3
"""
=============================================================================
PRODUCTION TRAINER for world_model_protein_design.py + reflective_freedom_engine.py
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT IS REAL HERE
  ✅ JSONL log per epoch (train/val metrics, LR, grad-norm, timing)
  ✅ Checkpoint selected on VALIDATION only; optimizer/sched/scaler state saved
  ✅ Test split evaluated EXACTLY ONCE, guarded by TestGuard
  ✅ Split indices persisted to split.json (reproducible)
  ✅ AMP (bf16/fp16) + DDP + gradient accumulation + clipping
  ✅ Parameter-usage audit BEFORE training; unused params excluded from optimizer
  ✅ Resume from checkpoint
  ✅ Data provenance stamped into every artifact

WHAT IS NOT REAL
  ❌ No biological or psychological dataset ships with these engines.
     Default data is SYNTHETIC and every artifact says so.
  ❌ reflective_freedom_engine's loss has NO external ground truth.
     Its objective is self-defined (maximize its own RF score). Without
     external labels, val/test numbers are OBJECTIVE VALUES, not accuracy.
     The source paper states RSMF is not empirically validated.
  ❌ Passing this trainer means the plumbing works, not that the model is right.

USAGE
  python train_engines.py --engine wmpd --epochs 40 --seeds 1
  python train_engines.py --engine rfe  --epochs 40 --external-targets
  python train_engines.py --engine wmpd --audit-only
  python train_engines.py --engine wmpd --resume runs/wmpd_xxx/last.pt
=============================================================================
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, logging, math, os, time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("trainer")

SYNTHETIC_WARNING = """
===============================================================================
  DATA PROVENANCE: SYNTHETIC
  Targets are a fixed random projection of the inputs plus noise.
  There is NO biology and NO psychology in this data.
  Metrics validate the TRAINING PLUMBING only.
  Do not report these numbers as model performance.
===============================================================================
"""


# =============================================================================
# APPARATUS: seeding, device, AMP, DDP
# =============================================================================

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> Tuple[torch.device, str]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        tag = "AMD_ROCM" if any(k in name for k in ("amd", "instinct", "radeon")) else "NVIDIA_CUDA"
        return torch.device("cuda"), tag
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "APPLE_MPS"
    return torch.device("cpu"), "CPU"


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def setup_ddp() -> Tuple[torch.device, str, int, int]:
    dev, tag = get_device()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        lr = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
        if dev.type == "cuda":
            torch.cuda.set_device(lr)
            dev = torch.device("cuda", lr)
        backend = "nccl" if tag in ("NVIDIA_CUDA", "AMD_ROCM") else "gloo"
        dist.init_process_group(backend=backend)
        return dev, tag, dist.get_rank(), dist.get_world_size()
    return dev, tag, 0, 1


def cleanup_ddp() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


@contextlib.contextmanager
def amp_ctx(dev: torch.device, precision: str):
    if precision == "fp32" or dev.type != "cuda":
        yield
        return
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        yield


def make_scaler(dev: torch.device, precision: str):
    enabled = (precision == "fp16" and dev.type == "cuda")
    try:
        return torch.amp.GradScaler(dev.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# =============================================================================
# INTEGRITY: test-set guard + split persistence
# =============================================================================

class TestGuard:
    """
    Prevents test-set peeking. `check()` raises until `finalize()` is called.
    finalize() is invoked exactly once, after validation-based selection.
    """
    def __init__(self):
        self._finalized = False
        self._test_calls = 0

    def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("TestGuard.finalize() called twice")
        self._finalized = True

    def check(self) -> None:
        if not self._finalized:
            raise RuntimeError(
                "TEST SET ACCESSED BEFORE FINALIZATION. "
                "Select on validation, then finalize(), then evaluate test once.")
        if self._test_calls >= 1:
            raise RuntimeError("TEST SET EVALUATED MORE THAN ONCE.")
        self._test_calls += 1


@dataclass
class Split:
    train: List[int]
    val: List[int]
    test: List[int]
    seed: int
    n_total: int

    @classmethod
    def make(cls, n: int, seed: int, frac=(0.7, 0.15, 0.15)) -> "Split":
        if n < 20:
            raise ValueError(f"n={n} too small for a 3-way split; need >= 20")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        a = int(frac[0] * n)
        b = int((frac[0] + frac[1]) * n)
        tr, va, te = idx[:a].tolist(), idx[a:b].tolist(), idx[b:].tolist()
        if min(len(tr), len(va), len(te)) < 2:
            raise ValueError(f"degenerate split: {len(tr)}/{len(va)}/{len(te)}")
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        return cls(tr, va, te, seed, n)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


# =============================================================================
# ARTIFACTS
# =============================================================================

class RunArtifacts:
    """
    runs/<name>/
      config.json      full config + engine + provenance
      split.json       exact indices (reproducible)
      audit.json       unused / nonfinite parameter names
      log.jsonl        one record per epoch
      best.pt          validation-selected
      last.pt          most recent (for resume)
      test.json        written ONCE
    """
    def __init__(self, root: str, name: str):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._logf = None
        self._test_written = False

    def write_json(self, fname: str, obj: Any) -> None:
        if not is_main():
            return
        (self.dir / fname).write_text(json.dumps(obj, indent=2, default=str))

    def open_log(self) -> None:
        if is_main() and self._logf is None:
            self._logf = open(self.dir / "log.jsonl", "a")

    def log(self, rec: Dict[str, Any]) -> None:
        if not is_main():
            return
        self.open_log()
        self._logf.write(json.dumps(rec, default=str) + "\n")
        self._logf.flush()

    def save_ckpt(self, fname: str, payload: Dict[str, Any]) -> None:
        if is_main():
            torch.save(payload, self.dir / fname)

    def write_test(self, obj: Any) -> None:
        if self._test_written:
            raise RuntimeError("test.json already written")
        self.write_json("test.json", obj)
        self._test_written = True

    def close(self) -> None:
        if self._logf is not None:
            self._logf.close()
            self._logf = None


# =============================================================================
# PARAMETER AUDIT (the v2 fix made operational)
# =============================================================================

def audit_parameters(model: nn.Module, loss: Tensor) -> Dict[str, List[str]]:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    unused, nonfinite, used = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            unused.append(n)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(n)
        else:
            used.append(n)
    model.zero_grad(set_to_none=True)
    return {"unused": unused, "nonfinite": nonfinite, "used": used}


def build_optimizer(model: nn.Module, used_names: List[str],
                    lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Optimize ONLY parameters that the audit proved reachable.
    This makes DDP(find_unused_parameters=False) safe by construction.
    """
    keep = set(used_names)
    params = [p for n, p in model.named_parameters() if n in keep and p.requires_grad]
    if not params:
        raise RuntimeError("audit found zero reachable parameters")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# =============================================================================
# SYNTHETIC DATA (clearly labelled, fixed target map across arms)
# =============================================================================

TARGET_SEED = 20260101   # independent of model seed → fair ablation


def fixed_target_map(d_in: int, d_out: int, device) -> Tensor:
    g = torch.Generator(device="cpu").manual_seed(TARGET_SEED)
    W = torch.randn(d_in, d_out, generator=g) / math.sqrt(d_in)
    return W.to(device)


def synthetic_composition(feats: Tensor, W: Tensor, signal: float) -> Tensor:
    """
    Generic smooth target: softmax(feats @ W / temperature) blended with noise.
    Deliberately INDEPENDENT of any Hodge internals — no rigging.
    """
    logits = feats @ W
    clean = F.softmax(logits, dim=-1)
    noise = F.softmax(torch.randn_like(logits) * 2.0, dim=-1)
    return signal * clean + (1.0 - signal) * noise


# =============================================================================
# ENGINE ADAPTERS
# =============================================================================

class EngineAdapter(Protocol):
    primary: str
    lower_is_better: bool
    def build_model(self, device) -> nn.Module: ...
    def make_batch(self, idx: Tensor, device) -> Dict[str, Tensor]: ...
    def forward(self, model: nn.Module, batch: Dict[str, Tensor]) -> Dict[str, Any]: ...
    def loss(self, out: Dict[str, Any], batch: Dict[str, Tensor],
             model: nn.Module) -> Dict[str, Tensor]: ...
    def metrics(self, out: Dict[str, Any], batch: Dict[str, Tensor]) -> Dict[str, Tensor]: ...


class WMPDAdapter:
    """world_model_protein_design.py (v2, dead branch fixed)."""
    primary = "kl"
    lower_is_better = True

    def __init__(self, n_samples: int = 256, signal: float = 0.7,
                 d_model: int = 64, d_action: int = 32, d_mn: int = 64,
                 d_hodge: int = 32, d_seq: int = 48, d_hidden: int = 64,
                 w_snr: float = 0.01, w_kernel: float = 0.05):
        self.n = n_samples
        self.signal = signal
        self.dims = dict(d_model=d_model, d_action=d_action, d_mn=d_mn,
                         d_hodge=d_hodge, d_seq=d_seq, d_hidden=d_hidden)
        self.w_snr, self.w_kernel = w_snr, w_kernel
        self._W = None
        self._loss_fn = None

    def build_model(self, device) -> nn.Module:
        from world_model_protein_design import (
            WorldModelProteinDesignEngine, EngineConfig, EngineLoss)
        cfg = EngineConfig(**self.dims, carpet_n_cells=36, carpet_n_levels=3,
                           rank=16, rollout_steps=4, n_res=24,
                           w_snr=self.w_snr, w_kernel=self.w_kernel)
        model = WorldModelProteinDesignEngine(cfg).to(device)
        self._loss_fn = EngineLoss(cfg)
        self._cfg = cfg
        return model

    def _ensure_W(self, device):
        if self._W is None:
            from world_model_protein_design import N_FATE
            self._W = fixed_target_map(self.dims["d_model"], N_FATE, device)

    def make_batch(self, idx: Tensor, device) -> Dict[str, Tensor]:
        from world_model_protein_design import N_FACTOR, N_FATE
        self._ensure_W(device)
        g = torch.Generator(device="cpu").manual_seed(int(idx.sum().item()) + 7)
        B = len(idx)
        state = torch.randn(B, self.dims["d_model"], generator=g).to(device)
        action = torch.randn(B, self.dims["d_action"], generator=g).to(device)
        fi = torch.randint(0, N_FACTOR, (B,), generator=g).to(device)
        dose = (torch.rand(B, generator=g) * 100 + 1).to(device)
        tm = (torch.rand(B, generator=g) * 100 + 1).to(device)
        tgt = synthetic_composition(state, self._W, self.signal)
        return {"state": state, "action": action, "factor_idx": fi,
                "dose_nM": dose, "time_h": tm, "target_comp": tgt}

    def forward(self, model, batch):
        return model(batch["state"], batch["action"], batch["factor_idx"],
                     batch["dose_nM"], batch["time_h"], generate_structure=False)

    def loss(self, out, batch, model):
        return self._loss_fn(out, target_comp=batch["target_comp"], model=model)

    def metrics(self, out, batch):
        kl = F.kl_div(out["composition"].log().clamp(min=-20),
                      batch["target_comp"], reduction="batchmean")
        return {"kl": kl,
                "snr": out["snr"].mean(),
                "kernel_penalty": out["kernel_penalty"].mean(),
                "viability_mae": (out["viability"] - 0.85).abs().mean()}


class RFEAdapter:
    """
    reflective_freedom_engine.py (v2).

    ⚠️ Its native loss has NO external ground truth. Two modes:
      external=False → primary metric is the self-defined OBJECTIVE.
                       This is NOT accuracy. Artifacts are stamped accordingly.
      external=True  → a synthetic external label is supplied so a supervised
                       metric exists. Still synthetic, but at least the target
                       is not defined by the model itself.
    """
    lower_is_better = True

    def __init__(self, n_samples: int = 256, signal: float = 0.7,
                 external: bool = False,
                 neural_dim: int = 64, desire_dim: int = 32,
                 value_dim: int = 48, hodge_dim: int = 24, action_dim: int = 16,
                 w_kernel: float = 0.05, w_value: float = 0.1):
        self.n = n_samples
        self.signal = signal
        self.external = external
        self.primary = "ext_mse" if external else "objective"
        self.dims = dict(neural_state_dim=neural_dim, desire_dim=desire_dim,
                         value_dim=value_dim, hodge_dim=hodge_dim,
                         action_dim=action_dim)
        self.w_kernel, self.w_value = w_kernel, w_value
        self._W = None
        self._loss_fn = None

    def build_model(self, device) -> nn.Module:
        from reflective_freedom_engine import (
            ReflectiveFreedomEngine, ReflectiveFreedomConfig, ReflectiveFreedomLoss)
        cfg = ReflectiveFreedomConfig(**self.dims, rank=16,
                                      w_kernel=self.w_kernel, w_value=self.w_value)
        self._loss_fn = ReflectiveFreedomLoss(cfg)
        self._cfg = cfg
        return ReflectiveFreedomEngine(cfg).to(device)

    def make_batch(self, idx: Tensor, device) -> Dict[str, Tensor]:
        d = self.dims["neural_state_dim"]
        if self._W is None:
            self._W = fixed_target_map(d, 1, device)
        g = torch.Generator(device="cpu").manual_seed(int(idx.sum().item()) + 13)
        B = len(idx)
        mk = lambda: torch.randn(B, d, generator=g).to(device)
        b = {"stimulus": mk(), "context": mk(), "neural_state": mk(),
             "action": torch.randn(B, self.dims["action_dim"], generator=g).to(device)}
        # Always exercise optional modules → all params reachable → DDP-safe
        b["policy_vector"] = b["neural_state"]
        b["explore_paths"] = True
        if self.external:
            clean = torch.sigmoid(b["neural_state"] @ self._W).squeeze(-1)
            noise = torch.rand(B, generator=g).to(device)
            b["ext_target"] = self.signal * clean + (1 - self.signal) * noise
        return b

    def forward(self, model, batch):
        return model(batch["stimulus"], batch["context"], batch["neural_state"],
                     batch["action"], policy_vector=batch["policy_vector"],
                     explore_paths=batch["explore_paths"])

    def loss(self, out, batch, model):
        L = self._loss_fn(out, model=model)
        if self.external and "ext_target" in batch:
            pred = out["autonomy_reflective_freedom"]
            ext = F.mse_loss(pred, batch["ext_target"])
            L = dict(L)
            L["ext"] = ext
            L["total"] = L["total"] + ext
        return L

    def metrics(self, out, batch):
        m = {"snr": out["value_snr"].mean(),
             "kernel_penalty": out["value_kernel_penalty"].mean(),
             "rf": out["autonomy_reflective_freedom"].mean(),
             "akrasia": out["akrasia_score"].mean()}
        if self.external and "ext_target" in batch:
            m["ext_mse"] = F.mse_loss(out["autonomy_reflective_freedom"],
                                      batch["ext_target"])
        return m


# =============================================================================
# TRAINER
# =============================================================================

@dataclass
class TrainerConfig:
    engine: str = "wmpd"
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    grad_accum: int = 1
    precision: str = "fp32"
    seed: int = 42
    n_samples: int = 256
    signal: float = 0.7
    patience: int = 12
    out_root: str = "runs"
    name: Optional[str] = None
    resume: Optional[str] = None
    external_targets: bool = False
    w_snr: float = 0.01
    w_kernel: float = 0.05
    audit_only: bool = False


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.dev, self.tag, self.rank, self.world = setup_ddp()
        seed_all(cfg.seed + self.rank)

        if cfg.engine == "wmpd":
            self.adapter = WMPDAdapter(cfg.n_samples, cfg.signal,
                                       w_snr=cfg.w_snr, w_kernel=cfg.w_kernel)
        elif cfg.engine == "rfe":
            self.adapter = RFEAdapter(cfg.n_samples, cfg.signal,
                                      external=cfg.external_targets,
                                      w_kernel=cfg.w_kernel)
        else:
            raise ValueError(f"unknown engine {cfg.engine}")

        name = cfg.name or f"{cfg.engine}_{time.strftime('%Y%m%d_%H%M%S')}_s{cfg.seed}"
        self.art = RunArtifacts(cfg.out_root, name)
        self.split = Split.make(cfg.n_samples, cfg.seed)
        self.guard = TestGuard()

        self.model = self.adapter.build_model(self.dev)
        self.provenance = {
            "data": "SYNTHETIC",
            "target_seed": TARGET_SEED,
            "signal_strength": cfg.signal,
            "engine": cfg.engine,
            "external_targets": cfg.external_targets,
            "rfe_objective_is_self_defined": (cfg.engine == "rfe" and not cfg.external_targets),
        }
        self.art.write_json("config.json",
                            {"trainer": asdict(cfg), "provenance": self.provenance,
                             "device": str(self.dev), "backend": self.tag,
                             "world_size": self.world})
        self.art.write_json("split.json", asdict(self.split))

        if is_main():
            print(SYNTHETIC_WARNING)
            if self.provenance["rfe_objective_is_self_defined"]:
                log.warning("RFE without external targets: the objective is "
                            "SELF-DEFINED. val/test numbers are NOT accuracy.")
            log.info(f"run dir: {self.art.dir}")
            log.info(f"split: {len(self.split.train)}/{len(self.split.val)}/"
                     f"{len(self.split.test)}  digest={self.split.digest()}")

        # ---- audit BEFORE optimizer/DDP ----
        idx = torch.tensor(self.split.train[:cfg.batch_size])
        batch = self.adapter.make_batch(idx, self.dev)
        out = self.adapter.forward(self.model, batch)
        losses = self.adapter.loss(out, batch, self.model)
        audit = audit_parameters(self.model, losses["total"])
        self.art.write_json("audit.json", audit)
        if is_main():
            log.info(f"audit: used={len(audit['used'])} "
                     f"unused={len(audit['unused'])} nonfinite={len(audit['nonfinite'])}")
            if audit["nonfinite"]:
                raise RuntimeError(f"non-finite gradients: {audit['nonfinite'][:8]}")
            if audit["unused"]:
                log.info(f"excluding {len(audit['unused'])} unreachable params "
                         f"from optimizer → DDP(find_unused_parameters=False) safe")
        if cfg.audit_only:
            self.art.close()
            cleanup_ddp()
            return

        self.opt = build_optimizer(self.model, audit["used"], cfg.lr, cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.01)
        self.scaler = make_scaler(self.dev, cfg.precision)

        if is_dist():
            ids = [self.dev.index] if self.dev.type == "cuda" else None
            self.model = DDP(self.model, device_ids=ids,
                             find_unused_parameters=False,
                             gradient_as_bucket_view=True)

        self.start_epoch = 0
        self.best = float("inf") if self.adapter.lower_is_better else -float("inf")
        self.best_epoch = -1
        self.epochs_no_improve = 0
        if cfg.resume:
            self._resume(cfg.resume)

    def _raw(self) -> nn.Module:
        m = self.model
        while isinstance(m, DDP):
            m = m.module
        return m

    def _resume(self, path: str) -> None:
        ck = torch.load(path, map_location=self.dev)
        if ck["split_digest"] != self.split.digest():
            raise RuntimeError("resume checkpoint has a DIFFERENT split — "
                               "continuing would contaminate the test set")
        self._raw().load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.scaler.load_state_dict(ck["scaler"])
        self.start_epoch = ck["epoch"] + 1
        self.best = ck["best"]
        self.best_epoch = ck["best_epoch"]
        log.info(f"resumed from {path} at epoch {self.start_epoch}")

    def _ckpt(self, epoch: int) -> Dict[str, Any]:
        return {"model": self._raw().state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "epoch": epoch, "best": self.best, "best_epoch": self.best_epoch,
                "split_digest": self.split.digest(),
                "config": asdict(self.cfg), "provenance": self.provenance}

    def _iterate(self, indices: List[int], train: bool) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train(train)
        order = np.random.default_rng(cfg.seed + (0 if train else 1)).permutation(len(indices))
        agg: Dict[str, List[float]] = {}
        total, nb = 0.0, 0
        gnorms: List[float] = []

        for s in range(0, len(order), cfg.batch_size):
            sel = [indices[order[i]] for i in range(s, min(s + cfg.batch_size, len(order)))]
            idx = torch.tensor(sel)
            batch = self.adapter.make_batch(idx, self.dev)

            with amp_ctx(self.dev, cfg.precision):
                out = self.adapter.forward(self.model, batch)
                losses = self.adapter.loss(out, batch, self._raw())
                mets = self.adapter.metrics(out, batch)

            if train:
                self.scaler.scale(losses["total"] / cfg.grad_accum).backward()
                if (nb + 1) % cfg.grad_accum == 0:
                    self.scaler.unscale_(self.opt)
                    gn = torch.nn.utils.clip_grad_norm_(
                        [p for g in self.opt.param_groups for p in g["params"]],
                        cfg.grad_clip)
                    gnorms.append(float(gn))
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                total += float(losses["total"].detach())
            else:
                total += float(losses["total"].detach())

            for k, v in mets.items():
                agg.setdefault(k, []).append(float(v.detach()))
            for k, v in losses.items():
                if isinstance(v, Tensor) and v.numel() == 1:
                    agg.setdefault(f"loss_{k}", []).append(float(v.detach()))
            nb += 1

        res = {k: float(np.mean(v)) for k, v in agg.items()}
        res["objective"] = total / max(nb, 1)
        res["grad_norm"] = float(np.mean(gnorms)) if gnorms else 0.0
        return res

    def fit(self) -> Dict[str, Any]:
        if self.cfg.audit_only:
            return {"audit_only": True}
        cfg = self.cfg
        primary = self.adapter.primary
        t0 = time.time()

        for ep in range(self.start_epoch, cfg.epochs):
            tr = self._iterate(self.split.train, train=True)
            with torch.no_grad():
                va = self._iterate(self.split.val, train=False)
            self.sched.step()

            key = va.get(primary, va["objective"])
            improved = key < self.best
            if improved:
                self.best, self.best_epoch = key, ep
                self.epochs_no_improve = 0
                self.art.save_ckpt("best.pt", self._ckpt(ep))
            else:
                self.epochs_no_improve += 1
            self.art.save_ckpt("last.pt", self._ckpt(ep))

            self.art.log({"epoch": ep + 1, "lr": self.sched.get_last_lr()[0],
                          "elapsed_s": round(time.time() - t0, 2),
                          "train": tr, "val": va,
                          "primary": primary, "val_primary": key,
                          "best": self.best, "improved": improved})
            if is_main() and ((ep + 1) % max(1, cfg.epochs // 10) == 0 or improved):
                log.info(f"ep {ep+1:4d} | train {tr['objective']:.5f} | "
                         f"val {primary} {key:.5f} {'*' if improved else ''}")

            if self.epochs_no_improve >= cfg.patience:
                if is_main():
                    log.info(f"early stop at epoch {ep+1} (patience {cfg.patience})")
                break

        # ---- TEST: exactly once, after selection ----
        best_path = self.art.dir / "best.pt"
        if best_path.exists():
            ck = torch.load(best_path, map_location=self.dev)
            self._raw().load_state_dict(ck["model"])
        self.guard.finalize()
        self.guard.check()
        with torch.no_grad():
            te = self._iterate(self.split.test, train=False)

        payload = {"engine": cfg.engine, "primary_metric": primary,
                   "best_epoch": self.best_epoch + 1, "best_val": self.best,
                   "test": te, "split": asdict(self.split),
                   "provenance": self.provenance,
                   "seed": cfg.seed,
                   "NOTE": ("Test evaluated once, after validation-based selection. "
                            "Data is SYNTHETIC — these are not model performance numbers.")}
        self.art.write_test(payload)
        self.art.close()

        if is_main():
            print("\n" + "=" * 76)
            print(f"  RUN COMPLETE — {cfg.engine}  seed={cfg.seed}")
            print("=" * 76)
            print(f"  best epoch      : {self.best_epoch + 1}")
            print(f"  best val {primary:<8}: {self.best:.6f}")
            print(f"  TEST {primary:<8}: {te.get(primary, float('nan')):.6f}")
            print(f"  test kernel_pen : {te.get('kernel_penalty', float('nan')):.6f}")
            print(f"  artifacts       : {self.art.dir}")
            print(f"  provenance      : {self.provenance['data']}")
            if self.provenance["rfe_objective_is_self_defined"]:
                print("  ⚠️  RFE objective is SELF-DEFINED. Not accuracy.")
            print("=" * 76)
        cleanup_ddp()
        return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["wmpd", "rfe"], default="wmpd")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", dest="n_samples", type=int, default=256)
    p.add_argument("--signal", type=float, default=0.7)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--grad-accum", dest="grad_accum", type=int, default=1)
    p.add_argument("--out-root", dest="out_root", default="runs")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--external-targets", dest="external_targets", action="store_true")
    p.add_argument("--w-snr", dest="w_snr", type=float, default=0.01)
    p.add_argument("--w-kernel", dest="w_kernel", type=float, default=0.05)
    p.add_argument("--audit-only", dest="audit_only", action="store_true")
    a = p.parse_args()
    Trainer(TrainerConfig(**vars(a))).fit()


if __name__ == "__main__":
    main()
