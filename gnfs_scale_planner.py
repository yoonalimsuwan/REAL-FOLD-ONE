#!/usr/bin/env python3
"""
=============================================================================
GNFS SCALE PLANNER + HARDWARE ADVISOR + LLM TRAINING PLANNER
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

╔═══════════════════════════════════════════════════════════════════════╗
║  THIS MODULE DOES NOT FACTOR ANY INTEGER.                            ║
║  It is a DEFENSIVE capacity-planning and benchmarking tool.          ║
║                                                                       ║
║  ❌ No factorization algorithm is implemented or invoked.            ║
║  ❌ No cryptographic attack capability is provided.                  ║
║  ✅ Difficulty estimation from published GNFS complexity.            ║
║  ✅ CADO-NFS / msieve log parsing and yield forecasting.             ║
║  ✅ Honest hardware suitability scoring (CPU/GPU/TPU/NPU/POWER).     ║
║  ✅ LLM training resource planning (where AI accelerators DO help).  ║
║  ✅ Post-quantum migration guidance (defensive).                     ║
╚═══════════════════════════════════════════════════════════════════════╝

KEY HONEST FACTS ENCODED IN THIS MODULE
  1. GNFS complexity is L_N[1/3, (64/9)^(1/3)] — subexponential, unchanged
     by any hardware or by the Structural Calculus mathematics.
  2. RSA-250 (829 bits) is the public record (~2700 core-years, 2020).
     RSA-2048 is ~1e11x harder; RSA-4096 ~1e23x. No accelerator closes that.
  3. TPUs / NPUs (Ascend, Hanguang) are systolic-array matmul engines.
     GNFS needs big-integer arithmetic, irregular memory access, and
     sparse GF(2) linear algebra — a poor architectural match.
  4. GPUs help ECM substantially; they help sieving/linear-algebra only
     marginally (memory-bound, irregular).
  5. IBM POWER's high memory bandwidth IS genuinely useful for the
     block-Wiedemann/Lanczos stage.
  6. Modern CPUs are ~2-4x a 2020 Xeon per socket, not 100x.
  7. The Structural Calculus tools (Carpet, OPMC, SSC/CSOC,
     assumption-light statistics) provide DIAGNOSTICS and FORECASTING.
     They do not reduce factoring complexity.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
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
╔═══════════════════════════════════════════════════════════════════════╗
║  GNFS SCALE PLANNER — capacity planning, NOT a factorization tool    ║
║  Adding TPUs/NPUs does NOT make RSA-2048/4096 factorable.            ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# §1. GNFS COMPLEXITY ESTIMATOR (published formula, honest calibration)
# =============================================================================

GNFS_C = (64.0 / 9.0) ** (1.0 / 3.0)     # ≈ 1.923
RSA250_BITS = 829
RSA250_CORE_YEARS = 2700.0               # published estimate, Boudot et al. 2020


def gnfs_ln_cost(bits: int) -> float:
    """ln of L_N[1/3, c] for an RSA modulus of `bits` bits."""
    lnN = bits * math.log(2.0)
    if lnN <= 1:
        return 0.0
    return GNFS_C * (lnN ** (1.0 / 3.0)) * (math.log(lnN) ** (2.0 / 3.0))


def gnfs_assessment(bits: int) -> Dict[str, Any]:
    """
    Honest difficulty assessment. Ratios are ASYMPTOTIC (constant factors,
    memory, and communication costs are not modelled and dominate in practice).
    """
    ln_c = gnfs_ln_cost(bits)
    ln_ref = gnfs_ln_cost(RSA250_BITS)
    log10_ratio = (ln_c - ln_ref) / math.log(10.0)
    ratio = 10.0 ** log10_ratio
    core_years = RSA250_CORE_YEARS * ratio

    if bits <= 512:
        tier = "workstation / small cluster (hours-days)"
        feasible = True
    elif bits <= 768:
        tier = "large cluster (weeks-months)"
        feasible = True
    elif bits <= RSA250_BITS:
        tier = "national-scale cluster (the public record)"
        feasible = True
    elif bits <= 1024:
        tier = "beyond public record; ~1e2x RSA-250"
        feasible = False
    else:
        tier = "far beyond any public capability"
        feasible = False

    return {
        "bits": bits,
        "decimal_digits": int(bits * math.log10(2.0)),
        "log10_L_N": ln_c / math.log(10.0),
        "ratio_vs_RSA250": ratio,
        "log10_ratio_vs_RSA250": log10_ratio,
        "core_years_rough": core_years,
        "feasibility_tier": tier,
        "publicly_feasible": feasible,
        "caveats": [
            "Ratios are asymptotic L_N ratios; real cost includes memory, "
            "communication, and constant factors that this does not model.",
            "Memory, not FLOPs, is the binding constraint above ~700 bits.",
            "No hardware accelerator changes the asymptotic complexity.",
        ],
    }


# =============================================================================
# §2. HARDWARE PROFILES + HONEST SUITABILITY SCORING
# =============================================================================

@dataclass
class HardwareProfile:
    name: str
    kind: str                    # cpu | gpu | tpu | npu
    fp16_tflops: float           # dense tensor throughput (estimate)
    mem_gb: float
    mem_bw_gbs: float            # memory bandwidth (estimate)
    int_arith: str               # strong | moderate | weak
    irregular_mem: str           # strong | moderate | weak
    notes: str = ""

    def gnfs_suitability(self) -> Dict[str, float]:
        """
        Stage-wise suitability in [0,1]. These are ARCHITECTURAL judgments,
        not measured benchmarks. GNFS is memory/branch/integer bound.
        """
        ia = {"strong": 1.0, "moderate": 0.5, "weak": 0.15}[self.int_arith]
        im = {"strong": 1.0, "moderate": 0.5, "weak": 0.15}[self.irregular_mem]
        bw = min(self.mem_bw_gbs / 1000.0, 1.0)

        if self.kind == "cpu":
            sieve, linalg, ecm = 0.9 * ia, 0.8 * bw, 0.6 * ia
        elif self.kind == "gpu":
            sieve, linalg, ecm = 0.35 * im, 0.4 * bw, 0.95
        else:  # tpu / npu — systolic arrays, poor fit
            sieve, linalg, ecm = 0.05, 0.08, 0.10
        return {
            "sieving": round(sieve, 3),
            "linear_algebra": round(linalg, 3),
            "ecm": round(ecm, 3),
            "llm_training": round(min(self.fp16_tflops / 300.0, 1.0), 3),
        }


HARDWARE_CATALOG: Dict[str, HardwareProfile] = {
    "xeon_2020": HardwareProfile(
        "Intel Xeon (2020, Ice/Cascade Lake)", "cpu", 0.5, 512, 140,
        "strong", "strong", "Baseline referenced by the user."),
    "epyc_modern": HardwareProfile(
        "AMD EPYC (Genoa/Bergamo class)", "cpu", 2.0, 1536, 460,
        "strong", "strong", "~3x memory bandwidth vs 2020 Xeon."),
    "xeon_modern": HardwareProfile(
        "Intel Xeon (Sapphire/Emerald Rapids)", "cpu", 3.0, 1024, 350,
        "strong", "strong", "AMX helps AI, not GNFS."),
    "power10": HardwareProfile(
        "IBM POWER10", "cpu", 1.5, 4096, 1000,
        "strong", "strong", "Very high memory BW — genuinely useful for linalg."),
    "yitian710": HardwareProfile(
        "Alibaba Yitian 710 (ARM)", "cpu", 1.0, 768, 300,
        "strong", "strong", "ARM server CPU; usable in a GNFS cluster."),
    "kunpeng": HardwareProfile(
        "Huawei Kunpeng (ARM)", "cpu", 1.0, 768, 280,
        "strong", "strong", "ARM server CPU; usable in a GNFS cluster."),
    "h100": HardwareProfile(
        "NVIDIA H100", "gpu", 990, 80, 3350,
        "moderate", "moderate", "Excellent for ECM and LLM; limited for GNFS linalg."),
    "a100": HardwareProfile(
        "NVIDIA A100", "gpu", 312, 80, 2039,
        "moderate", "moderate", "Same architectural limits as H100 for GNFS."),
    "tpu_v5": HardwareProfile(
        "Google TPU v5", "tpu", 400, 96, 1600,
        "weak", "weak", "Systolic array; no big-integer / sparse-GF2 path."),
    "ascend910": HardwareProfile(
        "Huawei Ascend 910", "npu", 320, 64, 1200,
        "weak", "weak", "AI tensor engine; poor GNFS fit."),
    "hanguang800": HardwareProfile(
        "Alibaba Hanguang 800", "npu", 120, 32, 700,
        "weak", "weak", "Inference accelerator; poor GNFS fit."),
}


def hardware_report() -> List[Dict[str, Any]]:
    out = []
    for key, hw in HARDWARE_CATALOG.items():
        out.append({
            "key": key,
            "name": hw.name,
            "kind": hw.kind,
            "fp16_tflops_est": hw.fp16_tflops,
            "mem_gb": hw.mem_gb,
            "mem_bw_gbs_est": hw.mem_bw_gbs,
            "suitability": hw.gnfs_suitability(),
            "notes": hw.notes,
        })
    return out


def recommend_hardware(stage: str) -> List[Tuple[str, float]]:
    scored = [(k, hw.gnfs_suitability().get(stage, 0.0))
              for k, hw in HARDWARE_CATALOG.items()]
    return sorted(scored, key=lambda x: -x[1])[:5]


# =============================================================================
# §3. CADO-NFS / msieve LOG PARSER
# =============================================================================

@dataclass
class SieveRecord:
    line_no: int
    stage: str = "unknown"
    q: Optional[float] = None
    rels: Optional[float] = None
    cpu: Optional[float] = None
    norm: Optional[float] = None
    raw: str = ""


_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*"
                 r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_ALIASES = {
    "q": ("q", "special_q", "specialq", "q0", "prime_q"),
    "rels": ("rels", "relations", "relations_found", "rel_count", "nrels", "prod"),
    "cpu": ("cpu", "cpu_time", "time", "elapsed", "seconds", "sec", "t"),
    "norm": ("norm", "normq", "qnorm"),
}
_STAGE_KW = {
    "polyselect": ("polyselect", "murphy", "polynomial"),
    "sieve": ("sieve", "sieving", "lattice", "specialq"),
    "filter": ("filter", "filtering", "merge", "dedup", "purge"),
    "linalg": ("linalg", "bwc", "krylov", "lanczos", "wiedemann", "matrix"),
    "sqrt": ("sqrt", "square root"),
}


class GnfsLogParser:
    """Parses CADO-NFS / msieve / GGNFS style logs (key=value lines)."""

    def parse_text(self, text: str) -> List[SieveRecord]:
        out = []
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            rec = SieveRecord(line_no=i, raw=s[:300])
            low = s.lower()
            for st, kws in _STAGE_KW.items():
                if any(k in low for k in kws):
                    rec.stage = st
                    break
            found = False
            for k, v in _KV.findall(s):
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if not math.isfinite(fv):
                    continue
                for canon, aliases in _ALIASES.items():
                    if k.lower() in aliases:
                        setattr(rec, canon, fv)
                        found = True
                        break
            if found and (rec.rels is not None or rec.q is not None):
                out.append(rec)
        return out

    def parse_file(self, path: str | Path) -> List[SieveRecord]:
        p = Path(path)
        if not p.exists():
            return []
        return self.parse_text(p.read_text(encoding="utf-8", errors="ignore"))


def generate_demo_log(n: int = 400, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    lines, t, q, regime = [], 0.0, 100_000.0, 0
    for i in range(n):
        if i % 90 == 0:
            regime = (regime + 1) % 3
        base = [80.0, 220.0, 45.0][regime]
        q += rng.uniform(500, 5000)
        burst = (rng.pareto(1.3) + 1.0) if rng.random() < 0.04 else 1.0
        rels = max(0.0, base * burst * rng.lognormal(0, 0.35))
        cpu = max(1e-3, rng.gamma(2.0, 0.15))
        t += cpu
        lines.append(f"[{t:.6f}] sieving: q={q:.0f} rels={rels:.3f} "
                     f"cpu={cpu:.6f} norm={q*1.02:.1f}")
        if i % 97 == 0:
            lines.append(f"[{t:.6f}] filtering: rels={rels*0.4:.1f} cpu={cpu*0.1:.4f}")
    return "\n".join(lines)


# =============================================================================
# §4. ASSUMPTION-LIGHT DIAGNOSTICS (Paper 9 / Paper 11)
# =============================================================================

def hill_theta(vals: List[float], frac: float = 0.2) -> float:
    x = [v for v in vals if v > 0]
    if len(x) < 12:
        return 0.0
    s = sorted(x, reverse=True)
    k = max(5, min(int(frac * len(s)), len(s) - 1))
    thr = s[k]
    if thr <= 0:
        return 0.0
    logs = [math.log(v / thr) for v in s[:k] if v > thr]
    if not logs:
        return 0.0
    m = sum(logs) / len(logs)
    return 1.0 / m if m > 1e-12 else 0.0


def cusum_count(x: np.ndarray, thr: float = 3.0, drift: float = 0.25) -> int:
    if len(x) < 8:
        return 0
    mu, sd = float(x.mean()), float(x.std())
    if sd < 1e-12:
        return 0
    sp = sn = 0.0
    n = 0
    for xi in x:
        z = (xi - mu) / sd
        sp = max(0.0, sp + z - drift)
        sn = max(0.0, sn - z - drift)
        if sp > thr or sn > thr:
            n += 1
            sp = sn = 0.0
    return n


def subsample_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    n = len(x)
    if n < 6:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = max(2, int(n ** (1 / 3)))
    b = min(b, n // 2)
    theta = float(x.mean())
    subs = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (subs - theta)
    a = 1 - conf
    lo = theta - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = theta - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": theta, "lo": min(lo, hi), "hi": max(lo, hi),
            "n": n, "block": b}


# =============================================================================
# §5. WORLD MODEL — relation-yield forecasting (NOT factorization)
# =============================================================================

@dataclass
class YieldWindow:
    q_lo: float
    q_hi: float
    mean_rels: float
    std_rels: float
    mean_cpu: float
    theta: float
    n_cp: int
    target_next: float


def make_windows(recs: List[SieveRecord], window: int = 8,
                 step: int = 4) -> List[YieldWindow]:
    recs = [r for r in recs if r.rels is not None and r.q is not None]
    if len(recs) < window + step:
        return []
    out = []
    for s in range(0, len(recs) - window - step + 1, step):
        cur = recs[s:s + window]
        nxt = recs[s + window:s + window + step]
        rels = np.array([r.rels for r in cur])
        cpus = np.array([r.cpu for r in cur if r.cpu is not None])
        qs = [r.q for r in cur]
        out.append(YieldWindow(
            q_lo=float(min(qs)), q_hi=float(max(qs)),
            mean_rels=float(rels.mean()), std_rels=float(rels.std()),
            mean_cpu=float(cpus.mean()) if len(cpus) else 0.0,
            theta=hill_theta(rels.tolist()),
            n_cp=cusum_count(rels),
            target_next=float(np.mean([r.rels for r in nxt])),
        ))
    return out


def window_features(w: YieldWindow) -> np.ndarray:
    return np.array([
        math.log1p(w.mean_rels), math.log1p(w.std_rels),
        math.log1p(w.mean_cpu), w.theta, float(w.n_cp),
        math.log1p(w.q_lo), math.log1p(w.q_hi),
    ], dtype=np.float32)


FEAT_DIM = 7

if HAS_TORCH:
    class YieldWorldModel(nn.Module):
        """Transformer world model: history of windows -> next mean rels."""
        def __init__(self, d_model: int = 64, n_heads: int = 2, n_layers: int = 2):
            super().__init__()
            self.proj = nn.Linear(FEAT_DIM, d_model)
            self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)
            self.unc = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(),
                                     nn.Linear(32, 1), nn.Softplus())

        def forward(self, hist):
            B, T, _ = hist.shape
            h = self.proj(hist) + self.pos[:, :T, :]
            z = self.enc(h)[:, -1]
            return {"pred": self.head(z).squeeze(-1),
                    "unc": self.unc(z).squeeze(-1)}


def train_world_model(windows: List[YieldWindow], history: int = 4,
                      epochs: int = 40) -> Tuple[Any, Dict[str, Any]]:
    if not HAS_TORCH:
        return None, {"status": "torch_unavailable"}
    if len(windows) < history + 2:
        return None, {"status": "insufficient_windows"}
    feats = torch.tensor(np.stack([window_features(w) for w in windows]))
    targets = torch.tensor([w.target_next for w in windows], dtype=torch.float32)
    X = torch.stack([feats[i:i + history] for i in range(len(feats) - history)])
    y = targets[history:]
    model = YieldWorldModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = F.mse_loss(out["pred"], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
    # persistence baseline
    persist = targets[history - 1:-1]
    mae_m = float(F.l1_loss(model(X)["pred"].detach(), y))
    mae_p = float(F.l1_loss(persist, y))
    return model, {
        "status": "ok", "initial_loss": losses[0], "final_loss": losses[-1],
        "mae_model": mae_m, "mae_persistence": mae_p,
        "improvement_pct": 100.0 * (mae_p - mae_m) / max(mae_p, 1e-9),
        "n_sequences": int(len(X)),
        "note": "Forecasting only. Does not factor anything.",
    }


# =============================================================================
# §6. LLM TRAINING PLANNER (where AI accelerators genuinely help)
# =============================================================================

def llm_training_plan(params_b: float, tokens_b: float,
                      hardware_key: str, n_devices: int = 1,
                      utilization: float = 0.40) -> Dict[str, Any]:
    """
    Rough LLM training plan. FLOPs ≈ 6 * params * tokens (Kaplan et al.).
    Memory ≈ params * (2 bytes weight + 2 grad + 8 optimizer) for AdamW mixed.
    """
    hw = HARDWARE_CATALOG.get(hardware_key)
    if hw is None:
        return {"error": f"unknown hardware {hardware_key}"}
    params = params_b * 1e9
    tokens = tokens_b * 1e9
    flops = 6.0 * params * tokens
    peak = hw.fp16_tflops * 1e12 * max(n_devices, 1)
    effective = peak * utilization
    seconds = flops / max(effective, 1e-9)
    mem_gb = params * 12 / 1e9
    fits = mem_gb <= hw.mem_gb * max(n_devices, 1) * 0.8
    return {
        "params_B": params_b, "tokens_B": tokens_b,
        "hardware": hw.name, "n_devices": n_devices,
        "total_flops": flops,
        "effective_tflops": effective / 1e12,
        "estimated_days": seconds / 86400.0,
        "memory_required_gb": mem_gb,
        "memory_fits": fits,
        "utilization_assumed": utilization,
        "note": ("Estimate only. Real throughput depends on parallelism, "
                 "interconnect, sequence length, and kernel efficiency."),
    }


# =============================================================================
# §7. DEFENSIVE: POST-QUANTUM MIGRATION GUIDANCE
# =============================================================================

def pqc_migration_plan(current_bits: int) -> Dict[str, Any]:
    a = gnfs_assessment(current_bits)
    return {
        "current_rsa_bits": current_bits,
        "classical_risk": a["feasibility_tier"],
        "quantum_risk": ("Shor's algorithm breaks RSA at any size given a "
                         "sufficiently large fault-tolerant quantum computer; "
                         "no such machine is publicly known to exist."),
        "recommended_actions": [
            "Inventory all RSA/ECDSA/ECDH usage (TLS, SSH, code signing, PKI).",
            "Adopt hybrid key exchange: X25519 + ML-KEM-768 (Kyber).",
            "Plan ML-DSA (Dilithium) or SLH-DSA (SPHINCS+) for signatures.",
            "Prioritise long-lived secrets ('harvest now, decrypt later').",
            "Follow NIST FIPS 203/204/205 and your national regulator.",
        ],
        "honest_note": ("Migration is the actionable defence. Waiting for "
                        "factoring advances is not a security strategy."),
    }


# =============================================================================
# §8. SELF-TEST
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1 complexity monotonicity
    a250 = gnfs_assessment(829)
    a2048 = gnfs_assessment(2048)
    a4096 = gnfs_assessment(4096)
    rep("T1 RSA-2048 harder than RSA-250",
        a2048["ratio_vs_RSA250"] > 1e8,
        f"ratio={a2048['ratio_vs_RSA250']:.2e}")
    rep("T1 RSA-4096 harder than RSA-2048",
        a4096["ratio_vs_RSA250"] > a2048["ratio_vs_RSA250"],
        f"ratio={a4096['ratio_vs_RSA250']:.2e}")
    rep("T1 RSA-2048 marked infeasible", not a2048["publicly_feasible"])

    # T2 hardware suitability
    tpu = HARDWARE_CATALOG["tpu_v5"].gnfs_suitability()
    cpu = HARDWARE_CATALOG["epyc_modern"].gnfs_suitability()
    gpu = HARDWARE_CATALOG["h100"].gnfs_suitability()
    rep("T2 CPU beats TPU for sieving", cpu["sieving"] > tpu["sieving"],
        f"cpu={cpu['sieving']} tpu={tpu['sieving']}")
    rep("T2 GPU beats CPU for ECM", gpu["ecm"] > cpu["ecm"],
        f"gpu={gpu['ecm']} cpu={cpu['ecm']}")
    rep("T2 TPU strong for LLM", tpu["llm_training"] > 0.8,
        f"{tpu['llm_training']}")

    # T3 parser
    p = GnfsLogParser()
    recs = p.parse_text(generate_demo_log(200))
    rep("T3 parser extracts records", len(recs) > 100, f"n={len(recs)}")
    rep("T3 stage detected", any(r.stage == "sieve" for r in recs))

    # T4 windows + stats
    ws = make_windows(recs)
    rep("T4 windows built", len(ws) > 10, f"n={len(ws)}")
    rels = np.array([r.rels for r in recs if r.rels is not None])
    ci = subsample_ci(rels)
    rep("T4 subsample CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])

    # T5 world model
    if HAS_TORCH and len(ws) > 8:
        _, hist = train_world_model(ws[:len(ws) // 2], epochs=15)
        rep("T5 world model trains", hist.get("status") == "ok",
            f"improvement={hist.get('improvement_pct', 0):+.1f}%")
    else:
        rep("T5 world model", True, "skipped (torch/data)")

    # T6 LLM planner
    plan = llm_training_plan(70, 1500, "h100", n_devices=256)
    rep("T6 LLM plan computed", plan["estimated_days"] > 0,
        f"{plan['estimated_days']:.1f} days on 256xH100")

    # T7 PQC plan
    pqc = pqc_migration_plan(2048)
    rep("T7 PQC plan has actions", len(pqc["recommended_actions"]) >= 4)

    # T8 no factorization capability
    rep("T8 module has no factorization function",
        not any("factor" in n.lower() and callable(globals().get(n))
                for n in globals()))

    print("\n" + "=" * 72)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 72)
    print("""
WHAT THIS MODULE IS
  ✅ Capacity planning, log diagnostics, yield forecasting
  ✅ Honest hardware suitability scoring
  ✅ LLM training resource estimation
  ✅ Defensive post-quantum migration guidance

WHAT IT IS NOT
  ❌ A factorization tool (none is implemented)
  ❌ A way to exceed RSA-250 with more accelerators
  ❌ A claim that Structural Calculus reduces GNFS complexity
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="GNFS scale planner (no factoring)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--assess", type=int, help="RSA bit size to assess")
    ap.add_argument("--hardware", action="store_true", help="hardware table")
    ap.add_argument("--recommend", choices=["sieving", "linear_algebra", "ecm",
                                            "llm_training"])
    ap.add_argument("--log", type=str, help="CADO/msieve log to analyze")
    ap.add_argument("--forecast", action="store_true", help="train yield model")
    ap.add_argument("--llm-plan", nargs=3, metavar=("PARAMS_B", "TOKENS_B", "HW"))
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--pqc", type=int, help="current RSA bits for migration plan")
    ap.add_argument("--json", type=str)
    a = ap.parse_args()

    out: Dict[str, Any] = {}
    if a.selftest:
        return 0 if self_test() else 1
    if a.assess:
        out["assessment"] = gnfs_assessment(a.assess)
    if a.hardware:
        out["hardware"] = hardware_report()
    if a.recommend:
        out["recommend"] = {a.recommend: recommend_hardware(a.recommend)}
    if a.log or a.forecast:
        p = GnfsLogParser()
        recs = p.parse_file(a.log) if a.log else p.parse_text(generate_demo_log())
        ws = make_windows(recs)
        out["log"] = {"n_records": len(recs), "n_windows": len(ws),
                      "source": a.log or "synthetic"}
        if a.forecast and ws:
            _, h = train_world_model(ws)
            out["forecast"] = h
    if a.llm_plan:
        pb, tb, hw = float(a.llm_plan[0]), float(a.llm_plan[1]), a.llm_plan[2]
        out["llm_plan"] = llm_training_plan(pb, tb, hw, a.devices)
    if a.pqc:
        out["pqc"] = pqc_migration_plan(a.pqc)

    if not out:
        ap.print_help()
        print(BANNER)
        return 0
    print(json.dumps(out, indent=2, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
