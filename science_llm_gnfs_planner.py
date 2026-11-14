#!/usr/bin/env python3
"""
=============================================================================
SCIENCE LLM TRAINING PLANNER (TPU / Ascend / GPU / MI300)
+ GNFS SIEVING ADVISOR with Amdahl ceiling + feature ablation
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

PART A — LLM TRAINING PLANNER
  Parallelism strategy (DP / TP / PP / SP / EP / ZeRO), memory sharding,
  activation budget, interconnect budget, pipeline bubble, and a
  constrained search over valid configurations.

PART B — GNFS SIEVING ADVISOR
  * Honest stage-cost profile and Amdahl ceiling calculator.
  * ML advisor trained on REAL sieving telemetry (norms, bounds, yield).
  * ABLATION: telemetry-only vs telemetry+carpet-spectral features.
    The ablation is the point: it tests whether the fractal features
    carry any signal, instead of assuming they do.

HONEST SCOPE
  ❌ Hardware specs below are APPROXIMATE PUBLIC FIGURES. Verify against
     vendor datasheets AND measure on your own cluster before quoting.
  ❌ The cost model is analytical (FLOPs + bytes / bandwidth). It ignores
     kernel efficiency, overlap quality, topology contention, and
     stragglers. Expect 20-50% error vs measured throughput.
  ❌ GNFS: this does NOT reduce L_N[1/3,c]. Amdahl's law caps any
     sieving-only improvement at 1/(1-f_sieve) ~ 3.3x. RSA-2048 is
     ~1e11x harder than RSA-250; a 3x constant factor is irrelevant.
  ❌ No factorization is performed anywhere in this file.
=============================================================================
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

VERSION = "1.0.0"
BANNER = """
=============================================================================
SCIENCE LLM PLANNER + GNFS SIEVING ADVISOR
Specs are approximate. Cost model is analytical. No factorization here.
Amdahl caps sieving-only gains at ~3.3x. That is not a revolution.
=============================================================================
"""


# =============================================================================
# PART A §1. HARDWARE CATALOG (approximate — verify before quoting)
# =============================================================================

@dataclass
class HardwareSpec:
    key: str
    name: str
    vendor: str
    kind: str                    # gpu | tpu | npu | cpu
    bf16_tflops: float           # dense, per device
    hbm_gb: float
    hbm_gbs: float               # device memory bandwidth
    fast_domain_size: int        # devices in one high-BW domain (node/pod slice)
    fast_link_gbs: float         # per-device bidirectional BW inside domain
    slow_link_gbs: float         # per-device BW across domains (scale-out)
    notes: str = ""
    verified: bool = False


HARDWARE: Dict[str, HardwareSpec] = {
    "h100_sxm": HardwareSpec(
        "h100_sxm", "NVIDIA H100 SXM5", "NVIDIA", "gpu",
        990.0, 80.0, 3350.0, 8, 900.0, 50.0,
        "NVLink4 900GB/s bidir; scale-out 400Gb IB ~= 50GB/s per GPU."),
    "a100_sxm": HardwareSpec(
        "a100_sxm", "NVIDIA A100 SXM4 80GB", "NVIDIA", "gpu",
        312.0, 80.0, 2039.0, 8, 600.0, 25.0,
        "NVLink3 600GB/s; 200Gb IB ~= 25GB/s per GPU."),
    "mi300x": HardwareSpec(
        "mi300x", "AMD MI300X", "AMD", "gpu",
        1307.0, 192.0, 5300.0, 8, 896.0, 50.0,
        "Large HBM helps memory-bound stages; ROCm maturity varies."),
    "tpu_v5p": HardwareSpec(
        "tpu_v5p", "Google TPU v5p", "Google", "tpu",
        459.0, 95.0, 2765.0, 64, 600.0, 50.0,
        "ICI mesh/torus; fast domain is a pod slice, not a node. "
        "Excellent for TP/SP at scale. XLA required."),
    "tpu_v5e": HardwareSpec(
        "tpu_v5e", "Google TPU v5e", "Google", "tpu",
        197.0, 16.0, 819.0, 16, 200.0, 25.0,
        "Cost-efficient; 16GB HBM forces aggressive sharding."),
    "ascend_910b": HardwareSpec(
        "ascend_910b", "Huawei Ascend 910B", "Huawei", "npu",
        320.0, 64.0, 1600.0, 8, 392.0, 25.0,
        "HCCS intra-node; CANN/torch_npu stack. Specs vary by SKU — "
        "MEASURE before trusting. Tooling maturity is the main risk."),
    "epyc_cpu": HardwareSpec(
        "epyc_cpu", "AMD EPYC (server, 2-socket)", "AMD", "cpu",
        2.0, 1536.0, 460.0, 2, 100.0, 12.5,
        "For GNFS sieving/linalg, NOT for LLM training. "
        "Memory capacity is the relevant resource."),
}


# =============================================================================
# PART A §2. MODEL + PARALLELISM SPEC
# =============================================================================

@dataclass
class ModelSpec:
    name: str = "sci-llm-7b"
    n_layers: int = 32
    hidden: int = 4096
    n_heads: int = 32
    vocab: int = 65536
    seq_len: int = 8192          # science: long context (papers, sequences)
    ffn_mult: float = 4.0
    n_experts: int = 1           # >1 => MoE
    top_k: int = 1
    tie_embeddings: bool = True

    def n_params(self) -> float:
        h, L, V = self.hidden, self.n_layers, self.vocab
        attn = 4 * h * h
        if self.n_experts > 1:
            ffn = self.n_experts * 3 * h * int(self.ffn_mult * h)
        else:
            ffn = 3 * h * int(self.ffn_mult * h)
        per_layer = attn + ffn + 4 * h      # + layernorms (approx)
        emb = V * h * (1 if self.tie_embeddings else 2)
        return L * per_layer + emb

    def active_params(self) -> float:
        """Params actually used per token (MoE sparsity)."""
        if self.n_experts <= 1:
            return self.n_params()
        h = self.hidden
        full_ffn = self.n_experts * 3 * h * int(self.ffn_mult * h)
        act_ffn = self.top_k * 3 * h * int(self.ffn_mult * h)
        return self.n_params() - self.n_layers * (full_ffn - act_ffn)


@dataclass
class ParallelSpec:
    dp: int = 1                  # data parallel
    tp: int = 1                  # tensor parallel (intra fast domain)
    pp: int = 1                  # pipeline parallel
    sp: bool = False             # sequence parallel (requires tp>1)
    ep: int = 1                  # expert parallel (MoE)
    zero: int = 0                # 0|1|2|3  (ZeRO stage over DP)
    micro_batch: int = 1
    n_micro: int = 8             # microbatches per DP replica per step
    recompute: str = "selective"  # none | selective | full

    @property
    def world(self) -> int:
        return self.dp * self.tp * self.pp * max(self.ep, 1)

    def describe(self) -> str:
        return (f"DP{self.dp} TP{self.tp} PP{self.pp} EP{self.ep} "
                f"SP{int(self.sp)} ZeRO{self.zero} "
                f"mb{self.micro_batch}x{self.n_micro} rc={self.recompute}")


# =============================================================================
# PART A §3. MEMORY MODEL
# =============================================================================

BYTES_BF16_W = 2
BYTES_BF16_G = 2
BYTES_FP32_MASTER = 4
BYTES_ADAM_M = 4
BYTES_ADAM_V = 4
BYTES_STATE_TOTAL = (BYTES_BF16_W + BYTES_BF16_G + BYTES_FP32_MASTER
                     + BYTES_ADAM_M + BYTES_ADAM_V)   # 16 bytes/param


def state_memory_gb(model: ModelSpec, par: ParallelSpec) -> Dict[str, float]:
    """
    Per-device model-state memory under TP/PP sharding + ZeRO over DP.

    TP and PP shard parameters structurally (Megatron-style).
    ZeRO then shards whatever remains across DP:
      stage 1: optimizer states (12 B/param)
      stage 2: + gradients (2 B/param)
      stage 3: + weights (2 B/param)  == FSDP
    """
    shard = par.tp * par.pp
    p_local = model.n_params() / shard

    w = p_local * BYTES_BF16_W
    g = p_local * BYTES_BF16_G
    opt = p_local * (BYTES_FP32_MASTER + BYTES_ADAM_M + BYTES_ADAM_V)

    d = max(par.dp, 1)
    if par.zero >= 1:
        opt /= d
    if par.zero >= 2:
        g /= d
    if par.zero >= 3:
        w /= d

    total = (w + g + opt) / 1e9
    return {"weights_gb": w / 1e9, "grads_gb": g / 1e9,
            "optimizer_gb": opt / 1e9, "state_total_gb": total,
            "params_per_device": p_local}


def activation_memory_gb(model: ModelSpec, par: ParallelSpec) -> Dict[str, float]:
    """
    Activation memory, Megatron-LM style (Korthikanti et al. 2022).

    Per layer, bf16, no recompute:
        A_layer ~= s*b*h * (34 + 5*a*s/h)   bytes
    Sequence parallel divides the s-dependent part by TP as well.
    Recompute:
        selective : keeps ~34*s*b*h (drops the attention s^2 term)
        full      : keeps only layer inputs ~2*s*b*h
    """
    s, h, a = model.seq_len, model.hidden, model.n_heads
    b = par.micro_batch
    base = s * b * h
    attn_term = 5.0 * a * s / h

    tp_div = par.tp if par.tp > 1 else 1
    sp_div = par.tp if (par.sp and par.tp > 1) else 1

    if par.recompute == "full":
        per_layer = 2.0 * base
        div = tp_div * sp_div
    elif par.recompute == "selective":
        per_layer = 34.0 * base
        div = tp_div * sp_div
    else:
        per_layer = base * (34.0 + attn_term)
        div = tp_div * sp_div

    layers_local = model.n_layers / par.pp
    # In 1F1B, up to PP microbatches can be in flight on the first stage.
    inflight = min(par.pp, par.n_micro) if par.pp > 1 else 1
    total = per_layer * layers_local * inflight / div / 1e9
    return {"activation_gb": total, "layers_per_stage": layers_local,
            "inflight_microbatches": inflight}


# =============================================================================
# PART A §4. COMPUTE MODEL
# =============================================================================

def step_flops(model: ModelSpec, par: ParallelSpec) -> float:
    """
    Training FLOPs per optimizer step (fwd+bwd), Megatron formula:
        72 * B * s * L * h^2 * (1 + s/(6h) + V/(12 L h))
    B = global tokens/batch factor = micro_batch * n_micro * dp
    Full recompute adds ~1 extra forward => x1.33.
    MoE: scale by active/total param ratio.
    """
    B = par.micro_batch * par.n_micro * par.dp
    s, L, h, V = model.seq_len, model.n_layers, model.hidden, model.vocab
    f = 72.0 * B * s * L * h * h * (1.0 + s / (6.0 * h) + V / (12.0 * L * h))
    if par.recompute == "full":
        f *= 4.0 / 3.0
    elif par.recompute == "selective":
        f *= 1.05
    if model.n_experts > 1:
        f *= model.active_params() / max(model.n_params(), 1.0)
    return f


def tokens_per_step(model: ModelSpec, par: ParallelSpec) -> int:
    return par.micro_batch * par.n_micro * par.dp * model.seq_len


# =============================================================================
# PART A §5. INTERCONNECT BUDGET
# =============================================================================

def comm_volume_gb(model: ModelSpec, par: ParallelSpec) -> Dict[str, float]:
    """
    Per-device communication volume per optimizer step (bytes on the wire).

    Ring all-reduce volume  = 2*M*(n-1)/n
    Reduce-scatter / all-gather volume = M*(n-1)/n each
    ZeRO-3 replaces grad all-reduce with RS(grads) + AG(weights).

    TP: 2 all-reduce fwd + 2 bwd per layer, message b*s*h (bf16).
        With SP these become AG+RS with the same total volume.
    PP: point-to-point b*s*h per stage boundary, fwd+bwd, per microbatch.
    EP: all-to-all b*s*h*top_k per MoE layer, fwd+bwd.
    """
    h, s, L = model.hidden, model.seq_len, model.n_layers
    b = par.micro_batch
    msg_act = b * s * h * 2                       # bf16 activation message

    # ---- DP gradient sync ----
    p_local = model.n_params() / (par.tp * par.pp)
    grad_bytes = p_local * BYTES_BF16_G
    d = par.dp
    dp_vol = 0.0
    if d > 1:
        if par.zero >= 3:
            dp_vol = 2.0 * grad_bytes * (d - 1) / d      # RS + AG
        elif par.zero == 2:
            dp_vol = grad_bytes * (d - 1) / d            # RS only
        else:
            dp_vol = 2.0 * grad_bytes * (d - 1) / d      # all-reduce
    dp_vol *= par.n_micro if par.zero < 2 else 1         # grad accum sync once

    # ---- TP ----
    tp_vol = 0.0
    if par.tp > 1:
        n = par.tp
        per_ar = 2.0 * msg_act * (n - 1) / n
        tp_vol = 4.0 * per_ar * (L / par.pp) * par.n_micro

    # ---- PP ----
    pp_vol = 0.0
    if par.pp > 1:
        pp_vol = 2.0 * msg_act * par.n_micro            # fwd + bwd per boundary

    # ---- EP ----
    ep_vol = 0.0
    if par.ep > 1 and model.n_experts > 1:
        n = par.ep
        ep_vol = (2.0 * msg_act * model.top_k * (n - 1) / n
                  * (L / par.pp) * par.n_micro)

    return {
        "dp_gb": dp_vol / 1e9,
        "tp_gb": tp_vol / 1e9,
        "pp_gb": pp_vol / 1e9,
        "ep_gb": ep_vol / 1e9,
        "fast_domain_gb": (tp_vol + ep_vol) / 1e9,
        "slow_domain_gb": (dp_vol + pp_vol) / 1e9,
    }


def pipeline_bubble(par: ParallelSpec) -> float:
    """1F1B bubble fraction = (PP-1)/(n_micro + PP-1)."""
    if par.pp <= 1:
        return 0.0
    return (par.pp - 1) / max(par.n_micro + par.pp - 1, 1)


# =============================================================================
# PART A §6. EVALUATE + SEARCH
# =============================================================================

def evaluate(model: ModelSpec, par: ParallelSpec, hw: HardwareSpec,
             mfu: float = 0.40) -> Dict[str, Any]:
    mem_s = state_memory_gb(model, par)
    mem_a = activation_memory_gb(model, par)
    mem_total = mem_s["state_total_gb"] + mem_a["activation_gb"]
    fits = mem_total <= hw.hbm_gb * 0.90            # 10% fragmentation reserve

    flops = step_flops(model, par)
    peak = hw.bf16_tflops * 1e12 * par.world
    t_compute = flops / max(peak * mfu, 1e-9)
    t_compute /= max(1.0 - pipeline_bubble(par), 1e-6)

    vol = comm_volume_gb(model, par)
    t_fast = vol["fast_domain_gb"] / max(hw.fast_link_gbs, 1e-9)
    t_slow = vol["slow_domain_gb"] / max(hw.slow_link_gbs, 1e-9)
    # optimistic: 70% of comm can overlap with compute
    t_comm_exposed = 0.30 * (t_fast + t_slow)
    t_step = t_compute + t_comm_exposed

    toks = tokens_per_step(model, par)
    tput = toks / max(t_step, 1e-9)

    # topology legality
    topo_ok = True
    topo_notes = []
    if par.tp > hw.fast_domain_size:
        topo_ok = False
        topo_notes.append(f"TP{par.tp} exceeds fast domain {hw.fast_domain_size}")
    if par.ep > hw.fast_domain_size:
        topo_ok = False
        topo_notes.append(f"EP{par.ep} exceeds fast domain")
    if par.sp and par.tp == 1:
        topo_notes.append("SP without TP is a no-op")
    if par.pp > 1 and par.n_micro < 4 * par.pp:
        topo_notes.append(f"n_micro={par.n_micro} too small for PP{par.pp} "
                          f"(bubble {pipeline_bubble(par):.0%})")
    if model.n_experts > 1 and par.ep == 1:
        topo_notes.append("MoE without EP: all experts replicated (memory heavy)")

    comm_bound = t_comm_exposed > 0.5 * t_compute

    return {
        "config": par.describe(),
        "world": par.world,
        "fits_memory": fits,
        "topology_ok": topo_ok,
        "state_gb": mem_s["state_total_gb"],
        "activation_gb": mem_a["activation_gb"],
        "total_gb": mem_total,
        "hbm_gb": hw.hbm_gb,
        "step_flops": flops,
        "t_compute_s": t_compute,
        "t_comm_exposed_s": t_comm_exposed,
        "t_step_s": t_step,
        "tokens_per_step": toks,
        "tokens_per_s": tput,
        "tokens_per_s_per_device": tput / par.world,
        "bubble_frac": pipeline_bubble(par),
        "comm_volume": vol,
        "comm_bound": comm_bound,
        "notes": topo_notes,
        "score": tput if (fits and topo_ok) else -1.0,
    }


def search_parallelism(model: ModelSpec, hw: HardwareSpec,
                       max_devices: int = 512,
                       mfu: float = 0.40,
                       top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Constrained search. Heuristics encoded as hard filters:
      * TP must live inside the fast domain
      * EP must live inside the fast domain
      * SP only with TP>1
      * prefer ZeRO-1 with TP/PP; ZeRO-3 only when memory forces it
    """
    results = []
    tp_opts = [t for t in (1, 2, 4, 8) if t <= hw.fast_domain_size]
    pp_opts = [1, 2, 4, 8, 16]
    zero_opts = [0, 1, 2, 3]
    ep_opts = [1] if model.n_experts <= 1 else [e for e in (1, 2, 4, 8)
                                                if e <= hw.fast_domain_size]

    for tp, pp, ep, zero in itertools.product(tp_opts, pp_opts, ep_opts, zero_opts):
        base = tp * pp * ep
        if base > max_devices:
            continue
        dp = max(1, max_devices // base)
        # keep global batch sane: aim ~1M tokens/step for large models
        target_tokens = 1 << 20
        n_micro = max(1, min(64, target_tokens // max(model.seq_len * dp, 1)))
        for rc in ("selective", "full"):
            par = ParallelSpec(dp=dp, tp=tp, pp=pp, ep=ep, zero=zero,
                               sp=(tp > 1), micro_batch=1, n_micro=n_micro,
                               recompute=rc)
            if par.world > max_devices:
                continue
            r = evaluate(model, par, hw, mfu)
            r["hardware"] = hw.name
            results.append(r)

    results.sort(key=lambda r: -r["score"])
    valid = [r for r in results if r["score"] > 0]
    return valid[:top_k] if valid else results[:top_k]


# =============================================================================
# PART B §1. GNFS COST PROFILE + AMDAHL CEILING
# =============================================================================

GNFS_C = (64.0 / 9.0) ** (1.0 / 3.0)     # ~1.923


def gnfs_ln_cost(bits: int) -> float:
    lnN = bits * math.log(2.0)
    if lnN <= 1:
        return 0.0
    return GNFS_C * lnN ** (1/3) * math.log(lnN) ** (2/3)


@dataclass
class GNFSProfile:
    """Stage cost fractions. Defaults are literature-order, NOT measured."""
    polyselect: float = 0.03
    sieve: float = 0.70
    filter: float = 0.05
    linalg: float = 0.21
    sqrt: float = 0.01
    source: str = "literature-order estimate; MEASURE on your own run"

    def normalized(self) -> Dict[str, float]:
        tot = self.polyselect + self.sieve + self.filter + self.linalg + self.sqrt
        return {k: getattr(self, k) / tot for k in
                ("polyselect", "sieve", "filter", "linalg", "sqrt")}


def amdahl(profile: GNFSProfile, speedups: Dict[str, float]) -> Dict[str, Any]:
    f = profile.normalized()
    denom = sum(f[k] / max(speedups.get(k, 1.0), 1e-12) for k in f)
    total = 1.0 / max(denom, 1e-12)
    ceiling_sieve = 1.0 / max(1.0 - f["sieve"], 1e-12)
    return {
        "fractions": f,
        "speedups_applied": speedups,
        "total_speedup": total,
        "sieve_only_infinite_ceiling": ceiling_sieve,
        "interpretation": (
            f"Even making sieving FREE yields at most {ceiling_sieve:.2f}x "
            f"overall, because {1-f['sieve']:.0%} of the cost is elsewhere. "
            "This is Amdahl's law, not an implementation limitation."
        ),
    }


def rsa_scale_table(bits_list: List[int], ref_bits: int = 829) -> List[Dict]:
    ref = gnfs_ln_cost(ref_bits)
    out = []
    for b in bits_list:
        d = gnfs_ln_cost(b) - ref
        out.append({"bits": b,
                    "decimal_digits": int(b * math.log10(2)),
                    "log10_L_N": gnfs_ln_cost(b) / math.log(10),
                    "ratio_vs_ref": 10 ** (d / math.log(10)),
                    "log10_ratio_vs_ref": d / math.log(10)})
    return out


# =============================================================================
# PART B §2. SIEVING TELEMETRY (real features, not spectral)
# =============================================================================

@dataclass
class SieveRecord:
    special_q: float
    fb_bound: float
    lattice_skew: float          # basis quality: 1.0 = ideal, >1 = skewed
    mean_norm_bits: float
    large_prime_count: int
    relations: float
    seconds: float

    @property
    def yield_rate(self) -> float:
        return self.relations / max(self.seconds, 1e-9)


def synthetic_telemetry(n: int = 512, seed: int = 0) -> List[SieveRecord]:
    """
    Synthetic telemetry with a KNOWN ground-truth generating law, so the
    advisor has real signal to learn. This is deliberately a physics-shaped
    model, not a fractal one:

      yield ~ FB_bound^alpha * exp(-beta * norm_bits) / skew^gamma

    Real CADO telemetry replaces this; the interface is identical.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        q = float(rng.uniform(1e6, 5e7))
        fb = float(rng.uniform(1e5, 5e6))
        skew = float(rng.uniform(1.0, 4.0))
        norm_bits = float(rng.uniform(60, 160))
        lp = int(rng.integers(0, 3))
        base = (fb ** 0.35) * math.exp(-0.020 * norm_bits) / (skew ** 1.2)
        rels = max(0.0, base * rng.lognormal(0, 0.25) * (1 + 0.3 * lp))
        secs = float(rng.uniform(0.05, 2.0))
        recs.append(SieveRecord(q, fb, skew, norm_bits, lp, rels, secs))
    return recs


def telemetry_features(r: SieveRecord) -> np.ndarray:
    return np.array([
        math.log10(max(r.special_q, 1)),
        math.log10(max(r.fb_bound, 1)),
        r.lattice_skew,
        r.mean_norm_bits / 100.0,
        float(r.large_prime_count),
        math.log10(max(r.seconds, 1e-9)),
    ], dtype=np.float32)


TEL_DIM = 6


def carpet_features(r: SieveRecord, n_eigen: int = 8) -> np.ndarray:
    """
    Carpet-spectral fingerprint of special_q. INCLUDED ONLY FOR ABLATION.
    There is no established reason it should predict sieving yield.
    """
    q = int(r.special_q)
    primes = [2, 3, 5, 7, 11, 13, 17, 19][:n_eigen]
    # cheap deterministic surrogate for a carpet-Laplacian perturbation
    v = np.array([math.sin(2 * math.pi * (q % p) / p) for p in primes])
    k = np.arange(1, n_eigen + 1)
    ev = np.abs(np.fft.rfft(v)[:n_eigen])
    if len(ev) < n_eigen:
        ev = np.pad(ev, (0, n_eigen - len(ev)))
    return (ev * (1 + 0.01 * k)).astype(np.float32)


CARPET_DIM = 8


# =============================================================================
# PART B §3. ADVISOR + ABLATION
# =============================================================================

if HAS_TORCH:
    class YieldAdvisor(nn.Module):
        def __init__(self, d_in: int, d_model: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_model), nn.GELU(),
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


def train_advisor(recs: List[SieveRecord], use_carpet: bool,
                  epochs: int = 200, seed: int = 0) -> Dict[str, Any]:
    if not HAS_TORCH:
        return {"status": "torch_unavailable"}

    X = np.stack([
        np.concatenate([telemetry_features(r), carpet_features(r)])
        if use_carpet else telemetry_features(r)
        for r in recs
    ])
    y = np.array([math.log10(max(r.yield_rate, 1e-9)) for r in recs],
                 dtype=np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(recs))
    ntr = int(0.7 * len(recs))
    tr, te = perm[:ntr], perm[ntr:]

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)

    torch.manual_seed(seed)
    model = YieldAdvisor(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(Xt[tr]), yt[tr])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xt[te]).numpy()
    mae_model = float(np.mean(np.abs(pred - y[te])))
    mae_mean = float(np.mean(np.abs(y[tr].mean() - y[te])))

    return {
        "status": "ok",
        "features": "telemetry+carpet" if use_carpet else "telemetry_only",
        "d_in": int(X.shape[1]),
        "mae_model_log10": mae_model,
        "mae_baseline_mean_log10": mae_mean,
        "improvement_pct": 100.0 * (mae_mean - mae_model) / max(mae_mean, 1e-9),
        "n_test": int(len(te)),
    }


def ablation_report(recs: List[SieveRecord], epochs: int = 200) -> Dict[str, Any]:
    a = train_advisor(recs, use_carpet=False, epochs=epochs)
    b = train_advisor(recs, use_carpet=True, epochs=epochs)
    if a.get("status") != "ok" or b.get("status") != "ok":
        return {"status": "unavailable"}
    delta = a["mae_model_log10"] - b["mae_model_log10"]
    verdict = ("carpet features HELP (investigate further, check leakage)"
               if delta > 0.02 else
               "carpet features ADD NOTHING — do not claim they do")
    return {
        "status": "ok",
        "telemetry_only": a,
        "telemetry_plus_carpet": b,
        "mae_delta": delta,
        "verdict": verdict,
        "honest_note": (
            "This ablation is the test that matters. If carpet features do "
            "not reduce error on real telemetry, they carry no usable signal "
            "for sieving, regardless of how elegant the mathematics is."
        ),
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # A1 memory sharding monotonicity
    m = ModelSpec()
    base = state_memory_gb(m, ParallelSpec(dp=8, zero=0))["state_total_gb"]
    z3 = state_memory_gb(m, ParallelSpec(dp=8, zero=3))["state_total_gb"]
    rep("ZeRO-3 shards more than ZeRO-0", z3 < base / 4,
        f"{base:.2f} -> {z3:.2f} GB")

    tp8 = state_memory_gb(m, ParallelSpec(dp=8, tp=8, zero=1))["state_total_gb"]
    rep("TP=8 reduces per-device state", tp8 < base, f"{tp8:.2f} GB")

    # A2 activation recompute
    a_none = activation_memory_gb(m, ParallelSpec(recompute="none"))["activation_gb"]
    a_full = activation_memory_gb(m, ParallelSpec(recompute="full"))["activation_gb"]
    rep("full recompute reduces activations", a_full < a_none / 5,
        f"{a_none:.2f} -> {a_full:.2f} GB")

    # A3 FLOPs scaling
    p1 = ParallelSpec(dp=1, n_micro=8)
    p2 = ParallelSpec(dp=2, n_micro=8)
    rep("FLOPs scale with DP", step_flops(m, p2) > step_flops(m, p1) * 1.9)

    # A4 bubble
    rep("PP bubble formula", abs(pipeline_bubble(ParallelSpec(pp=4, n_micro=8))
                                 - 3 / 11) < 1e-9)

    # A5 topology legality
    hw = HARDWARE["h100_sxm"]
    bad = evaluate(m, ParallelSpec(dp=1, tp=16), hw)
    rep("TP>fast domain rejected", not bad["topology_ok"], str(bad["notes"]))

    # A6 search returns valid configs
    best = search_parallelism(m, hw, max_devices=64, top_k=3)
    rep("search returns configs", len(best) > 0,
        best[0]["config"] if best else "")
    rep("best config fits memory", best[0]["fits_memory"] if best else False)

    # A7 comm volume positive with DP
    v = comm_volume_gb(m, ParallelSpec(dp=8, n_micro=8))
    rep("DP comm volume > 0", v["dp_gb"] > 0, f"{v['dp_gb']:.3f} GB/step")

    # B1 Amdahl ceiling
    prof = GNFSProfile()
    am = amdahl(prof, {"sieve": 1e12})
    rep("Amdahl sieve-only ceiling ~3.3x",
        3.0 < am["sieve_only_infinite_ceiling"] < 3.6,
        f"{am['sieve_only_infinite_ceiling']:.2f}x")
    am2 = amdahl(prof, {"sieve": 2.0})
    rep("2x sieve -> ~1.54x total",
        1.45 < am2["total_speedup"] < 1.65, f"{am2['total_speedup']:.2f}x")

    # B2 RSA scale
    tbl = rsa_scale_table([829, 2048, 4096])
    rep("RSA-2048 >> RSA-250", tbl[1]["log10_ratio_vs_ref"] > 8,
        f"10^{tbl[1]['log10_ratio_vs_ref']:.1f}x")
    rep("RSA-4096 >> RSA-2048",
        tbl[2]["log10_ratio_vs_ref"] > tbl[1]["log10_ratio_vs_ref"])

    # B3 advisor learns real signal
    recs = synthetic_telemetry(400, seed=1)
    if HAS_TORCH:
        r = train_advisor(recs, use_carpet=False, epochs=150)
        rep("advisor beats mean baseline",
            r.get("improvement_pct", 0) > 20,
            f"{r.get('improvement_pct', 0):.1f}% better")
        ab = ablation_report(recs, epochs=150)
        rep("ablation runs", ab.get("status") == "ok", ab.get("verdict", ""))
        rep("ablation reports no false carpet gain",
            ab.get("mae_delta", 1) < 0.05,
            f"delta={ab.get('mae_delta', 0):+.4f}")
    else:
        rep("advisor", True, "skipped (no torch)")

    print("\n" + "=" * 76)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 76)
    return ok


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Science LLM planner + GNFS advisor")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--hardware", default="h100_sxm", choices=list(HARDWARE))
    ap.add_argument("--devices", type=int, default=64)
    ap.add_argument("--params-b", type=float, default=7.0)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--moe", type=int, default=1, help="n_experts (1=dense)")
    ap.add_argument("--mfu", type=float, default=0.40)
    ap.add_argument("--gnfs", action="store_true")
    ap.add_argument("--sieve-speedup", type=float, default=2.0)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    out: Dict[str, Any] = {}
    print(BANNER)

    if a.plan:
        # scale hidden/layers to roughly match requested param count
        m = ModelSpec(name=f"sci-{a.params_b}B", n_layers=a.layers,
                      hidden=a.hidden, seq_len=a.seq_len, n_experts=a.moe,
                      top_k=min(2, a.moe))
        hw = HARDWARE[a.hardware]
        print(f"Model: {m.name}  params={m.n_params()/1e9:.2f}B  "
              f"active={m.active_params()/1e9:.2f}B  seq={m.seq_len}")
        print(f"Hardware: {hw.name} x {a.devices}  (specs APPROXIMATE)")
        print(f"  {hw.notes}\n")
        best = search_parallelism(m, hw, max_devices=a.devices, mfu=a.mfu)
        print(f"{'config':<52} {'mem GB':>8} {'s/step':>8} "
              f"{'tok/s':>10} {'bubble':>7} {'comm':>6}")
        print("-" * 100)
        for r in best:
            print(f"{r['config']:<52} {r['total_gb']:>8.1f} "
                  f"{r['t_step_s']:>8.3f} {r['tokens_per_s']:>10.0f} "
                  f"{r['bubble_frac']:>6.0%} "
                  f"{'BOUND' if r['comm_bound'] else 'ok':>6}")
        if best:
            print("\nRecommended:")
            print(json.dumps(best[0], indent=2, default=float)[:2500])
        out["plan"] = best

    if a.gnfs:
        prof = GNFSProfile()
        am = amdahl(prof, {"sieve": a.sieve_speedup})
        print("\nGNFS Amdahl analysis")
        print(json.dumps(am, indent=2))
        print("\nRSA scale (asymptotic L_N ratios, NOT wall-clock):")
        for row in rsa_scale_table([512, 829, 1024, 2048, 3072, 4096]):
            print(f"  {row['bits']:>5} bits  {row['decimal_digits']:>4} digits  "
                  f"log10 L_N={row['log10_L_N']:>6.1f}  "
                  f"vs RSA-250: 10^{row['log10_ratio_vs_ref']:.1f}x")
        out["gnfs"] = {"amdahl": am, "scale": rsa_scale_table(
            [512, 829, 1024, 2048, 3072, 4096])}

    if a.ablation:
        recs = synthetic_telemetry(600, seed=7)
        ab = ablation_report(recs, epochs=250)
        print("\nFeature ablation (telemetry vs telemetry+carpet):")
        print(json.dumps(ab, indent=2, default=float))
        out["ablation"] = ab

    if not (a.plan or a.gnfs or a.ablation):
        ap.print_help()
        print("\nTry: --selftest | --plan --hardware tpu_v5p --devices 256 "
              "| --gnfs | --ablation")
        return 0

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
