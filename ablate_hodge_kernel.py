#!/usr/bin/env python3
"""
=============================================================================
ABLATION: does the Hodge kernel penalty actually help?
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

THE QUESTION, STATED PRECISELY
  v2 added:  kernel_penalty = ||h L||²   with L = (W + Wᵀ)/2
  and a loss term  w_kernel · kernel_penalty  (minimized).

  Two SEPARATE questions must not be conflated:
    Q1 (mechanism) Does minimizing it actually reduce ||hL||²?
    Q2 (utility)   Does reducing ||hL||² improve the HELD-OUT TASK metric?

  Q1 can be YES while Q2 is NO. That is a common and publishable outcome:
  the regularizer does what it says, but the task does not care.

ARMS
  A  w_snr=0     w_kernel=0     no Hodge regularization
  B  w_snr>0     w_kernel=0     SNR only
  C  w_snr=0     w_kernel>0     kernel penalty only
  D  w_snr>0     w_kernel>0     both (v2 default)

FAIRNESS CONTROLS
  • Identical data, identical split, identical target map across arms.
    Target generator uses a FIXED seed independent of the model seed.
  • Same model seed per (arm, seed) pair → paired comparison.
  • The synthetic target is a generic random projection of the INPUT.
    It is deliberately NOT constructed to reward low kernel penalty.

STATISTICS
  • Paired per-seed deltas vs arm A.
  • Bootstrap 95% CI on the mean paired delta (resample seeds).
  • Sign consistency across seeds.
  • Verdict: "evidence of effect" only if CI excludes 0 AND signs agree.

HONEST EXPECTATION
  On synthetic targets there is no reason for the penalty to help.
  The scientifically correct default outcome is NO EVIDENCE OF EFFECT.
  A "gain" on 3–5 seeds is more likely noise than signal. Use >= 10 seeds
  and real data before claiming anything.

USAGE
  python ablate_hodge_kernel.py --engine wmpd --seeds 5 --epochs 40
  python ablate_hodge_kernel.py --engine rfe  --seeds 5 --external-targets
=============================================================================
"""
from __future__ import annotations

import argparse, json, logging, math, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from train_engines import Trainer, TrainerConfig, is_main, cleanup_ddp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("ablate")

ARMS: Dict[str, Dict[str, float]] = {
    "A_none":        {"w_snr": 0.0,  "w_kernel": 0.0},
    "B_snr_only":    {"w_snr": 0.01, "w_kernel": 0.0},
    "C_kernel_only": {"w_snr": 0.0,  "w_kernel": 0.05},
    "D_both":        {"w_snr": 0.01, "w_kernel": 0.05},
}
BASELINE = "A_none"


# =============================================================================
# MECHANISM CHECK: is the penalty term reachable, and does it move?
# =============================================================================

def mechanism_check(engine: str, w_kernel: float, device: str = "cpu") -> Dict[str, Any]:
    """
    Verify the v2 fix operationally:
      • L_weight receives gradient iff w_kernel > 0 (or w_snr > 0 via snr path)
      • Minimizing kernel_penalty alone actually decreases it
    """
    import torch.optim as optim
    if engine == "wmpd":
        from world_model_protein_design import (
            WorldModelProteinDesignEngine, EngineConfig, EngineLoss, N_FATE)
        cfg = EngineConfig(d_model=64, d_action=32, d_mn=64, d_hodge=32,
                           d_seq=48, d_hidden=64, carpet_n_cells=36,
                           carpet_n_levels=3, rank=16, rollout_steps=4,
                           w_snr=0.0, w_kernel=w_kernel)
        model = WorldModelProteinDesignEngine(cfg).to(device)
        loss_fn = EngineLoss(cfg)
        B = 4
        batch = dict(state=torch.randn(B, 64, device=device),
                     action=torch.randn(B, 32, device=device),
                     factor_idx=torch.zeros(B, dtype=torch.long, device=device),
                     dose_nM=torch.ones(B, device=device) * 10,
                     time_h=torch.ones(B, device=device) * 24)
        out = model(batch["state"], batch["action"], batch["factor_idx"],
                    batch["dose_nM"], batch["time_h"])
        tgt = torch.rand(B, N_FATE, device=device).softmax(-1)
        losses = loss_fn(out, target_comp=tgt, model=model)
        model.zero_grad(set_to_none=True)
        losses["total"].backward()
        Lw = model.hodge.L_weight
        grad_norm = float(Lw.grad.norm()) if Lw.grad is not None else 0.0

        # descent test on the penalty alone
        z = torch.randn(8, 64, device=device)
        opt = optim.SGD(model.hodge.parameters(), lr=0.05)
        p0 = model.hodge(z)["kernel_penalty"].mean().item()
        for _ in range(60):
            opt.zero_grad()
            model.hodge(z)["kernel_penalty"].mean().backward()
            opt.step()
        p1 = model.hodge(z)["kernel_penalty"].mean().item()
    else:
        from reflective_freedom_engine import (
            ReflectiveFreedomEngine, ReflectiveFreedomConfig, ReflectiveFreedomLoss)
        cfg = ReflectiveFreedomConfig(neural_state_dim=64, desire_dim=32,
                                      value_dim=48, hodge_dim=24, action_dim=16,
                                      rank=16, w_kernel=w_kernel, w_value=0.0)
        model = ReflectiveFreedomEngine(cfg).to(device)
        loss_fn = ReflectiveFreedomLoss(cfg)
        B = 4
        batch = dict(stimulus=torch.randn(B, 64, device=device),
                     context=torch.randn(B, 64, device=device),
                     neural_state=torch.randn(B, 64, device=device),
                     action=torch.randn(B, 16, device=device))
        out = model(batch["stimulus"], batch["context"], batch["neural_state"],
                    batch["action"], policy_vector=batch["neural_state"],
                    explore_paths=True)
        losses = loss_fn(out, model=model)
        model.zero_grad(set_to_none=True)
        losses["total"].backward()
        Lw = model.value.L_weight
        grad_norm = float(Lw.grad.norm()) if Lw.grad is not None else 0.0

        z = torch.randn(8, 64, device=device)
        opt = optim.SGD(model.value.parameters(), lr=0.05)
        p0 = model.value(z)["kernel_penalty"].mean().item()
        for _ in range(60):
            opt.zero_grad()
            model.value(z)["kernel_penalty"].mean().backward()
            opt.step()
        p1 = model.value(z)["kernel_penalty"].mean().item()

    return {"w_kernel": w_kernel,
            "L_weight_grad_norm": grad_norm,
            "penalty_before": p0, "penalty_after": p1,
            "penalty_reduced": p1 < p0,
            "Q1_mechanism_works": p1 < p0}


# =============================================================================
# STATISTICS
# =============================================================================

def bootstrap_ci(deltas: np.ndarray, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = deltas[rng.integers(0, len(deltas), len(deltas))].mean()
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def paired_stats(base: np.ndarray, arm: np.ndarray) -> Dict[str, Any]:
    """Lower-is-better: negative delta = arm better than baseline."""
    d = arm - base
    lo, hi = bootstrap_ci(d)
    n_better = int((d < 0).sum())
    pooled = math.sqrt((base.var(ddof=1) + arm.var(ddof=1)) / 2) if len(d) > 1 else float("nan")
    effect = float(d.mean() / pooled) if pooled and pooled > 1e-12 else 0.0
    ci_excludes_zero = (not math.isnan(lo)) and (lo > 0 or hi < 0)
    signs_agree = (n_better == len(d)) or (n_better == 0)
    if ci_excludes_zero and signs_agree:
        verdict = "EVIDENCE OF EFFECT"
    elif math.isnan(lo):
        verdict = "TOO FEW SEEDS"
    else:
        verdict = "NO EVIDENCE OF EFFECT"
    return {"mean_delta": float(d.mean()), "std_delta": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "ci95_low": lo, "ci95_high": hi,
            "n_seeds": len(d), "n_better_than_baseline": n_better,
            "effect_size": effect, "per_seed_delta": d.tolist(),
            "verdict": verdict}


# =============================================================================
# ABLATION DRIVER
# =============================================================================

def run_ablation(engine: str, seeds: List[int], epochs: int,
                 n_samples: int, signal: float, batch_size: int,
                 precision: str, external: bool, out_root: str,
                 patience: int) -> Dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(out_root) / f"ablation_{engine}_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mech = {"kernel_on": mechanism_check(engine, 0.05, device),
            "kernel_off": mechanism_check(engine, 0.0, device)}
    (root / "mechanism.json").write_text(json.dumps(mech, indent=2))
    print("\n" + "=" * 78)
    print("  Q1 — MECHANISM CHECK (does the penalty do what it claims?)")
    print("=" * 78)
    for k, v in mech.items():
        print(f"  {k:<12} w_kernel={v['w_kernel']:.3f}  "
              f"|grad L|={v['L_weight_grad_norm']:.4e}  "
              f"penalty {v['penalty_before']:.5f} → {v['penalty_after']:.5f}  "
              f"reduced={v['penalty_reduced']}")

    results: Dict[str, Dict[str, List[float]]] = {a: {} for a in ARMS}
    primary: Optional[str] = None

    for seed in seeds:
        for arm, w in ARMS.items():
            name = f"{arm}_s{seed}"
            cfg = TrainerConfig(engine=engine, epochs=epochs,
                                batch_size=batch_size, seed=seed,
                                n_samples=n_samples, signal=signal,
                                precision=precision, patience=patience,
                                out_root=str(root), name=name,
                                external_targets=external,
                                w_snr=w["w_snr"], w_kernel=w["w_kernel"])
            tr = Trainer(cfg)
            primary = tr.adapter.primary
            payload = tr.fit()
            te = payload["test"]
            for k, v in te.items():
                if isinstance(v, (int, float)):
                    results[arm].setdefault(k, []).append(float(v))
            log.info(f"[{arm} seed={seed}] test {primary}="
                     f"{te.get(primary, float('nan')):.6f} "
                     f"kernel={te.get('kernel_penalty', float('nan')):.6f}")

    # ---- paired comparison vs baseline ----
    stats: Dict[str, Any] = {}
    for arm in ARMS:
        if arm == BASELINE:
            continue
        base = np.array(results[BASELINE][primary])
        a = np.array(results[arm][primary])
        stats[arm] = paired_stats(base, a)

    summary = {"engine": engine, "primary_metric": primary,
               "seeds": seeds, "epochs": epochs,
               "provenance": "SYNTHETIC",
               "arms": {a: {k: {"mean": float(np.mean(v)),
                                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                "values": v}
                            for k, v in d.items()} for a, d in results.items()},
               "paired_vs_baseline": stats,
               "mechanism": mech}
    (root / "ablation.json").write_text(json.dumps(summary, indent=2))

    # ---- report ----
    print("\n" + "=" * 78)
    print(f"  Q2 — UTILITY: does the penalty improve held-out {primary}?")
    print(f"  engine={engine}  seeds={len(seeds)}  data=SYNTHETIC")
    print("=" * 78)
    print(f"  {'arm':<16}{primary:>12}{'± std':>10}{'kernel_pen':>13}")
    for arm, d in results.items():
        m = float(np.mean(d[primary]))
        s = float(np.std(d[primary], ddof=1)) if len(seeds) > 1 else 0.0
        kp = float(np.mean(d.get("kernel_penalty", [float('nan')])))
        print(f"  {arm:<16}{m:>12.6f}{s:>10.6f}{kp:>13.6f}")

    print(f"\n  PAIRED vs {BASELINE}  (negative Δ = better)")
    print(f"  {'arm':<16}{'mean Δ':>12}{'95% CI':>24}{'better':>9}  verdict")
    for arm, st in stats.items():
        ci = f"[{st['ci95_low']:+.5f}, {st['ci95_high']:+.5f}]"
        print(f"  {arm:<16}{st['mean_delta']:>+12.6f}{ci:>24}"
              f"{st['n_better_than_baseline']:>4}/{st['n_seeds']:<4}  {st['verdict']}")

    print("\n" + "=" * 78)
    print("  INTERPRETATION")
    print("=" * 78)
    q1 = mech["kernel_on"]["Q1_mechanism_works"]
    any_effect = any(st["verdict"] == "EVIDENCE OF EFFECT" for st in stats.values())
    print(f"  Q1 mechanism (penalty reduces ||hL||²) : {'YES' if q1 else 'NO'}")
    print(f"  Q2 utility   (improves held-out task)  : "
          f"{'EVIDENCE' if any_effect else 'NO EVIDENCE'}")
    if q1 and not any_effect:
        print("\n  → The regularizer works mechanically but the task does not benefit.")
        print("    This is a legitimate, reportable result. Do NOT keep the term")
        print("    just because it is mathematically appealing.")
    if not q1:
        print("\n  → The penalty is not even doing its own job. Investigate before")
        print("    interpreting any task metric.")
    print(f"""
  CAVEATS
    • Data is SYNTHETIC. No biology, no psychology.
    • {len(seeds)} seeds gives a WEAK bootstrap CI. Use >= 10 for any claim.
    • The target map is a generic random projection, deliberately NOT
      constructed to favour low kernel penalty.
    • A gain here would justify collecting REAL data, not a publication.
  artifacts: {root}
""")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["wmpd", "rfe"], default="wmpd")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", dest="seed_start", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-samples", dest="n_samples", type=int, default=256)
    p.add_argument("--signal", type=float, default=0.7)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--external-targets", dest="external_targets", action="store_true")
    p.add_argument("--out-root", dest="out_root", default="ablations")
    p.add_argument("--mechanism-only", dest="mechanism_only", action="store_true")
    a = p.parse_args()

    if a.mechanism_only:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(json.dumps({"kernel_on": mechanism_check(a.engine, 0.05, dev),
                          "kernel_off": mechanism_check(a.engine, 0.0, dev)}, indent=2))
        return

    run_ablation(a.engine, list(range(a.seed_start, a.seed_start + a.seeds)),
                 a.epochs, a.n_samples, a.signal, a.batch_size,
                 a.precision, a.external_targets, a.out_root, a.patience)


if __name__ == "__main__":
    main()
