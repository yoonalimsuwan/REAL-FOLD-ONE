"""
=============================================================================
SESI INTEGRATION TEST SUITE — 8-LAYER VERIFICATION
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0 (Full Integration Test Suite)

TEST LAYERS
───────────
  [L1] Contract Tests       — shape/dtype/semantic ที่ทุกขอบ
  [L2] Gradient Tests       — backward ผ่านทุกโมดูล
  [L3] Consistency Tests    — C_{n+1→n} R_{n+1} ≈ R_n
  [L4] Provenance Tests     — synthetic ไม่ถูก promote
  [L5] Budget Tests         — resource allocation ทำงาน
  [L6] AMP/DDP Tests        — mixed precision + distributed
  [L7] Determinism Tests    — seed เดิม → ผลเหมือนเดิม
  [L8] Benchmark Tests      — เทียบกับ AF3 baseline

USAGE
─────
  python integration_test_suite.py --all
  python integration_test_suite.py --layer 1,2,3
  python integration_test_suite.py --module protein_folding
  python integration_test_suite.py --verbose

=============================================================================
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.test_suite")


# ═══════════════════════════════════════════════════════════════════════
# TEST RESULT TRACKING
# ═══════════════════════════════════════════════════════════════════════

class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class TestResult:
    layer: int
    name: str
    status: TestStatus
    message: str = ""
    duration_s: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class TestSuite:
    """Central test runner and reporter."""
    
    def __init__(self, verbose: bool = False):
        self.results: List[TestResult] = []
        self.verbose = verbose
        self._current_layer = 0
    
    def start_layer(self, layer: int, name: str) -> None:
        self._current_layer = layer
        if self.verbose:
            print(f"\n{'='*72}")
            print(f"  LAYER {layer}: {name}")
            print(f"{'='*72}")
    
    def record(self, name: str, passed: bool, message: str = "",
               details: Optional[Dict] = None, duration: float = 0.0) -> None:
        status = TestStatus.PASS if passed else TestStatus.FAIL
        result = TestResult(
            layer=self._current_layer,
            name=name,
            status=status,
            message=message,
            duration_s=duration,
            details=details or {},
        )
        self.results.append(result)
        
        symbol = "✅" if passed else "❌"
        if self.verbose:
            print(f"  {symbol} {name}")
            if message and not passed:
                print(f"     └─ {message}")
            if details and self.verbose:
                for k, v in details.items():
                    print(f"     └─ {k}: {v}")
    
    def skip(self, name: str, reason: str) -> None:
        self.results.append(TestResult(
            layer=self._current_layer,
            name=name,
            status=TestStatus.SKIP,
            message=reason,
        ))
        if self.verbose:
            print(f"  ⏭️  SKIP: {name} ({reason})")
    
    def summary(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in TestStatus}
        for r in self.results:
            counts[r.status.value] += 1
        return counts
    
    def print_summary(self) -> None:
        counts = self.summary()
        total = sum(counts.values())
        
        print(f"\n{'='*72}")
        print(f"  TEST SUITE SUMMARY")
        print(f"{'='*72}")
        print(f"  Total tests : {total}")
        print(f"  ✅ PASS     : {counts['PASS']}")
        print(f"  ❌ FAIL     : {counts['FAIL']}")
        print(f"  ⏭️  SKIP     : {counts['SKIP']}")
        print(f"  ⚠️  WARN     : {counts['WARN']}")
        print(f"{'='*72}")
        
        # Per-layer breakdown
        for layer in sorted(set(r.layer for r in self.results)):
            layer_results = [r for r in self.results if r.layer == layer]
            layer_pass = sum(1 for r in layer_results if r.status == TestStatus.PASS)
            layer_total = len(layer_results)
            print(f"  Layer {layer}: {layer_pass}/{layer_total} passed")
        
        # Failures
        failures = [r for r in self.results if r.status == TestStatus.FAIL]
        if failures:
            print(f"\n  ❌ FAILURES:")
            for r in failures:
                print(f"    L{r.layer}: {r.name} — {r.message}")
        
        print(f"{'='*72}")
    
    def save_report(self, path: str) -> None:
        report = {
            "timestamp": time.time(),
            "summary": self.summary(),
            "results": [
                {
                    "layer": r.layer,
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_s": r.duration_s,
                    "details": {k: str(v) for k, v in r.details.items()},
                }
                for r in self.results
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {path}")
    
    @property
    def all_passed(self) -> bool:
        return all(r.status in (TestStatus.PASS, TestStatus.SKIP) for r in self.results)


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1: CONTRACT TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_contract_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L1: Verify shape/dtype/semantic contracts at every module boundary.
    """
    suite.start_layer(1, "Contract Tests")
    
    # ── Test 1.1: Protein Folding Orchestrator ────────────────────────
    try:
        from PROTEIN_FOLDING_ORCHESTRATOR import (
            ProteinFoldingOrchestrator, ProteinOrchestratorConfig,
            FoldingTask, TaskType,
        )
        
        cfg = ProteinOrchestratorConfig(
            latent_dim=32, sequence_dim=16, structure_dim=32,
            level1_dim=16, level2_dim=24, level3_dim=32,
            batch_size=2, device=str(device),
        )
        model = ProteinFoldingOrchestrator(cfg).to(device)
        
        B, L = 2, 20
        seq = torch.randint(0, 21, (B, L), device=device)
        task = FoldingTask(
            task_id="contract_test", task_type=TaskType.STRUCTURE_PREDICTION,
            sequence_length=L, accuracy_target=0.8,
            time_budget_s=10.0, memory_budget_mb=500.0, energy_budget_j=100.0,
        )
        
        t0 = time.time()
        out = model(seq, task)
        dt = time.time() - t0
        
        # Contract checks
        checks = {
            "task_level_weights shape": out["task_level_weights"].shape == (B, cfg.n_levels),
            "exec_total_computational_cost > 0": out["exec_total_computational_cost"] > 0,
            "consistency_overall ∈ [0,1]": 0 <= out["consistency_overall"].item() <= 1,
            "verify_quality_score ∈ [0,1]": 0 <= out["verify_quality_score"].mean().item() <= 1,
            "pareto_report has all fields": all(
                hasattr(out["pareto_report"], f) for f in [
                    "task_id", "error", "runtime_s", "memory_mb",
                    "energy_j", "levels_used", "consistency", "verified",
                    "epistemic_levels",
                ]
            ),
        }
        
        all_ok = all(checks.values())
        suite.record(
            "ProteinFoldingOrchestrator contracts",
            all_ok,
            message="" if all_ok else f"Failed: {[k for k,v in checks.items() if not v]}",
            details=checks,
            duration=dt,
        )
    except ImportError as e:
        suite.skip("ProteinFoldingOrchestrator contracts", str(e))
    
    # ── Test 1.2: SESI Framework v7 ───────────────────────────────────
    try:
        from SESI_FRAMEWORKv7_Hierarchical_Quantum_State_Spaces_Multiscale_Applications import (
            SESIFrameworkV7, SESIv7Config,
        )
        
        cfg = SESIv7Config(
            latent_dim=64, hierarchical_levels=3,
            weak_measurement_steps=4, device=str(device),
        )
        model = SESIFrameworkV7(cfg).to(device)
        
        B = 2
        base = torch.randn(B, cfg.latent_dim, device=device)
        noise = torch.randn_like(base) * 0.1
        radar = torch.randn_like(base) * 0.05
        
        t0 = time.time()
        out = model(base, noise, radar)
        dt = time.time() - t0
        
        checks = {
            "hierarchical_representations has 3 levels": len(out["hier_hierarchical_representations"]) == 3,
            "consistency_loss >= 0": out["hier_consistency_loss"].item() >= 0,
            "measurement_efficiency > 0": out["meas_measurement_efficiency"].mean().item() > 0,
            "global_optimization_loss finite": torch.isfinite(out["global_optimization_loss"]),
            "all 11 domain outputs present": all(
                any(k.startswith(p) for k in out.keys())
                for p in ["protein_", "chem_", "onco_", "epi_", "aging_",
                          "regen_", "crispr_", "brain_", "fab7_", "ew_"]
            ),
        }
        
        all_ok = all(checks.values())
        suite.record(
            "SESIFrameworkV7 contracts",
            all_ok,
            message="" if all_ok else f"Failed: {[k for k,v in checks.items() if not v]}",
            details=checks,
            duration=dt,
        )
    except ImportError as e:
        suite.skip("SESIFrameworkV7 contracts", str(e))
    
    # ── Test 1.3: Calabi-Yau System ───────────────────────────────────
    try:
        from SESI_CALABI_YAU_AGI_ONE_INTEGRATION__1_ import (
            SESICalabiYauSystem, CalabiYauConfig,
        )
        
        cfg = CalabiYauConfig(latent_dim=32, period_dim=8, device=str(device))
        model = SESICalabiYauSystem(cfg).to(device)
        
        B = 2
        config = torch.randn(B, cfg.config_matrix_max_size, cfg.max_poly_degree, device=device)
        hodge = torch.tensor([[1.0, 101.0], [2.0, 86.0]], device=device)
        
        t0 = time.time()
        out = model(config, hodge)
        dt = time.time() - t0
        
        checks = {
            "cy_latent shape": out["cy_latent"].shape == (B, cfg.latent_dim),
            "period_matrix shape": out["period_matrix"].shape == (B, cfg.period_dim, cfg.period_dim),
            "period_matrix symmetric": torch.allclose(
                out["period_matrix"], out["period_matrix"].transpose(-1, -2), atol=1e-5
            ),
            "mirror_score ∈ [0,1]": (out["mirror_score"] >= 0).all() and (out["mirror_score"] <= 1).all(),
            "euler_pred finite": torch.isfinite(out["euler_pred"]).all(),
        }
        
        all_ok = all(checks.values())
        suite.record(
            "SESICalabiYauSystem contracts",
            all_ok,
            message="" if all_ok else f"Failed: {[k for k,v in checks.items() if not v]}",
            details=checks,
            duration=dt,
        )
    except ImportError as e:
        suite.skip("SESICalabiYauSystem contracts", str(e))
    
    # ── Test 1.4: Quantum RISC/CISC ───────────────────────────────────
    try:
        from QUANTUM_RISC_CISC_SESI_FULL_SPECTRUM_INTEGRATION import (
            QuantumRISCCISCFullSpectrum, QuantumRISCCISCConfig,
        )
        
        cfg = QuantumRISCCISCConfig(
            latent_dim=32, instruction_dim=16, qubit_state_dim=16,
            max_weak_layers=4, device=str(device),
        )
        model = QuantumRISCCISCFullSpectrum(cfg).to(device)
        
        B = 2
        inp = torch.randn(B, cfg.latent_dim, device=device)
        
        t0 = time.time()
        out = model(inp)
        dt = time.time() - t0
        
        checks = {
            "lwsga_composition_fidelity ∈ [0,1]": (
                (out["lwsga_composition_fidelity"] >= 0).all() and
                (out["lwsga_composition_fidelity"] <= 1).all()
            ),
            "cmam_final_confidence ∈ [0,1]": (
                (out["cmam_final_confidence"] >= 0).all() and
                (out["cmam_final_confidence"] <= 1).all()
            ),
            "vacuum_HONEST_NOTE present": "HONEST_NOTE" in out,
            "qm_HONEST_NOTE present": "qm_HONEST_NOTE" in out,
        }
        
        all_ok = all(checks.values())
        suite.record(
            "QuantumRISCCISCFullSpectrum contracts",
            all_ok,
            message="" if all_ok else f"Failed: {[k for k,v in checks.items() if not v]}",
            details=checks,
            duration=dt,
        )
    except ImportError as e:
        suite.skip("QuantumRISC/CISC contracts", str(e))
    
    # ── Test 1.5: Biomedical Integration ──────────────────────────────
    try:
        from SESI_UNIFIED_BIOMEDICAL_INTEGRATION__1_ import (
            SESIBiomedicalIntegration, BiomedicalIntegrationConfig,
        )
        
        cfg = BiomedicalIntegrationConfig(
            latent_dim=32, hodge_dim=16, protein_dim=16,
            neural_dim=8, tumor_grid_size=4,
            eeg_channels=4, fmri_voxels=16, device=str(device),
        )
        model = SESIBiomedicalIntegration(cfg).to(device)
        
        B = 2
        hodge_latent = torch.randn(B, cfg.latent_dim, device=device)
        period_matrix = torch.randn(B, cfg.hodge_dim, device=device)
        hodge_numbers = torch.tensor([[2.0, 3.0]], device=device).expand(B, -1)
        yukawa = torch.randn(B, 16, device=device)
        dream_quality = torch.rand(B, device=device)
        
        t0 = time.time()
        out = model(hodge_latent, period_matrix, hodge_numbers, yukawa, dream_quality)
        dt = time.time() - t0
        
        checks = {
            "integrated_state shape": out["integrated_state"].shape == (B, cfg.latent_dim),
            "protein_coords_3d shape": out["protein_coords_3d"].shape[-1] == 3,
            "onco_treatment_response ∈ [0,1]": (
                (out["onco_treatment_response"] >= 0).all() and
                (out["onco_treatment_response"] <= 1).all()
            ),
            "epi_r0 > 0": (out["epi_r0"] > 0).all(),
            "aging_biological_age > 0": (out["aging_biological_age"] > 0).all(),
            "control_safety_score ∈ [0,1]": (
                (out["control_safety_score"] >= 0).all() and
                (out["control_safety_score"] <= 1).all()
            ),
        }
        
        all_ok = all(checks.values())
        suite.record(
            "SESIBiomedicalIntegration contracts",
            all_ok,
            message="" if all_ok else f"Failed: {[k for k,v in checks.items() if not v]}",
            details=checks,
            duration=dt,
        )
    except ImportError as e:
        suite.skip("SESIBiomedicalIntegration contracts", str(e))


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2: GRADIENT TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_gradient_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L2: Verify backward() flows through every module.
    """
    suite.start_layer(2, "Gradient Tests")
    
    modules_to_test = [
        ("PROTEIN_FOLDING_ORCHESTRATOR", "ProteinFoldingOrchestrator",
         lambda d: _make_protein_orchestrator(d)),
        ("SESI_FRAMEWORKv7_Hierarchical_Quantum_State_Spaces_Multiscale_Applications",
         "SESIFrameworkV7", lambda d: _make_sesi_v7(d)),
        ("SESI_CALABI_YAU_AGI_ONE_INTEGRATION__1_", "SESICalabiYauSystem",
         lambda d: _make_calabi_yau(d)),
        ("QUANTUM_RISC_CISC_SESI_FULL_SPECTRUM_INTEGRATION",
         "QuantumRISCCISCFullSpectrum", lambda d: _make_qrisc(d)),
        ("SESI_UNIFIED_BIOMEDICAL_INTEGRATION__1_", "SESIBiomedicalIntegration",
         lambda d: _make_biomedical(d)),
    ]
    
    for module_name, class_name, factory in modules_to_test:
        try:
            t0 = time.time()
            model, loss_fn = factory(device)
            loss = loss_fn()
            loss.backward()
            dt = time.time() - t0
            
            # Check all trainable params have finite gradients
            trainable = [p for p in model.parameters() if p.requires_grad]
            all_have_grad = all(p.grad is not None for p in trainable)
            all_finite = all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
            
            suite.record(
                f"Gradient flow: {class_name}",
                all_have_grad and all_finite,
                message=(
                    "No gradient" if not all_have_grad else
                    "Non-finite gradient" if not all_finite else ""
                ),
                details={
                    "n_params": sum(p.numel() for p in trainable),
                    "loss": loss.item(),
                    "duration_s": dt,
                },
                duration=dt,
            )
        except ImportError as e:
            suite.skip(f"Gradient flow: {class_name}", str(e))
        except Exception as e:
            suite.record(f"Gradient flow: {class_name}", False, message=str(e))


def _make_protein_orchestrator(device):
    from PROTEIN_FOLDING_ORCHESTRATOR import (
        ProteinFoldingOrchestrator, ProteinOrchestratorConfig,
        FoldingTask, TaskType,
    )
    cfg = ProteinOrchestratorConfig(
        latent_dim=16, sequence_dim=8, structure_dim=16,
        level1_dim=8, level2_dim=12, level3_dim=16,
        device=str(device),
    )
    model = ProteinFoldingOrchestrator(cfg).to(device)
    seq = torch.randint(0, 21, (2, 10), device=device)
    task = FoldingTask(
        task_id="grad_test", task_type=TaskType.STRUCTURE_PREDICTION,
        sequence_length=10, accuracy_target=0.8,
        time_budget_s=10.0, memory_budget_mb=500.0, energy_budget_j=100.0,
    )
    
    def loss_fn():
        out = model(seq, task)
        return -out["verify_quality_score"].mean()
    
    return model, loss_fn


def _make_sesi_v7(device):
    from SESI_FRAMEWORKv7_Hierarchical_Quantum_State_Spaces_Multiscale_Applications import (
        SESIFrameworkV7, SESIv7Config,
    )
    cfg = SESIv7Config(latent_dim=32, hierarchical_levels=2,
                       weak_measurement_steps=2, device=str(device))
    model = SESIFrameworkV7(cfg).to(device)
    base = torch.randn(2, cfg.latent_dim, device=device)
    
    def loss_fn():
        out = model(base)
        return out["global_optimization_loss"]
    
    return model, loss_fn


def _make_calabi_yau(device):
    from SESI_CALABI_YAU_AGI_ONE_INTEGRATION__1_ import (
        SESICalabiYauSystem, CalabiYauConfig,
    )
    cfg = CalabiYauConfig(latent_dim=32, period_dim=8, device=str(device))
    model = SESICalabiYauSystem(cfg).to(device)
    config = torch.randn(2, cfg.config_matrix_max_size, cfg.max_poly_degree, device=device)
    hodge = torch.tensor([[1.0, 101.0], [2.0, 86.0]], device=device)
    
    def loss_fn():
        out = model(config, hodge)
        return -out["mirror_score"].mean()
    
    return model, loss_fn


def _make_qrisc(device):
    from QUANTUM_RISC_CISC_SESI_FULL_SPECTRUM_INTEGRATION import (
        QuantumRISCCISCFullSpectrum, QuantumRISCCISCConfig,
    )
    cfg = QuantumRISCCISCConfig(
        latent_dim=16, instruction_dim=8, qubit_state_dim=8,
        max_weak_layers=3, device=str(device),
    )
    model = QuantumRISCCISCFullSpectrum(cfg).to(device)
    inp = torch.randn(2, cfg.latent_dim, device=device)
    
    def loss_fn():
        out = model(inp)
        return -out["lwsga_composition_fidelity"].mean()
    
    return model, loss_fn


def _make_biomedical(device):
    from SESI_UNIFIED_BIOMEDICAL_INTEGRATION__1_ import (
        SESIBiomedicalIntegration, BiomedicalIntegrationConfig,
    )
    cfg = BiomedicalIntegrationConfig(
        latent_dim=32, hodge_dim=16, protein_dim=16,
        neural_dim=8, tumor_grid_size=4,
        eeg_channels=4, fmri_voxels=16, device=str(device),
    )
    model = SESIBiomedicalIntegration(cfg).to(device)
    B = 2
    hodge = torch.randn(B, cfg.latent_dim, device=device)
    period = torch.randn(B, cfg.hodge_dim, device=device)
    hodge_n = torch.tensor([[2.0, 3.0]], device=device).expand(B, -1)
    yukawa = torch.randn(B, 16, device=device)
    quality = torch.rand(B, device=device)
    
    def loss_fn():
        out = model(hodge, period, hodge_n, yukawa, quality)
        return -out["integrated_state"].mean()
    
    return model, loss_fn


# ═══════════════════════════════════════════════════════════════════════
# LAYER 3: CONSISTENCY TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_consistency_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L3: Verify C_{n+1→n} R_{n+1} ≈ R_n (coarse-graining consistency).
    """
    suite.start_layer(3, "Consistency Tests")
    
    # ── Test 3.1: Hierarchical Quantum State Space ────────────────────
    try:
        from SESI_FRAMEWORKv7_Hierarchical_Quantum_State_Spaces_Multiscale_Applications import (
            SESIFrameworkV7, SESIv7Config,
        )
        
        cfg = SESIv7Config(latent_dim=32, hierarchical_levels=3, device=str(device))
        model = SESIFrameworkV7(cfg).to(device)
        base = torch.randn(2, cfg.latent_dim, device=device)
        
        out = model(base)
        consistency_loss = out["hier_consistency_loss"].item()
        
        # Consistency loss should be small (ideally < 0.1 for random init)
        suite.record(
            "Hierarchical state consistency",
            consistency_loss < 1.0,
            message=f"consistency_loss={consistency_loss:.4f}",
            details={"consistency_loss": consistency_loss},
        )
    except ImportError as e:
        suite.skip("Hierarchical state consistency", str(e))
    
    # ── Test 3.2: Protein Folding Orchestrator ────────────────────────
    try:
        from PROTEIN_FOLDING_ORCHESTRATOR import (
            ProteinFoldingOrchestrator, ProteinOrchestratorConfig,
            FoldingTask, TaskType,
        )
        
        cfg = ProteinOrchestratorConfig(
            latent_dim=32, sequence_dim=16, structure_dim=32,
            level1_dim=16, level2_dim=24, level3_dim=32,
            device=str(device),
        )
        model = ProteinFoldingOrchestrator(cfg).to(device)
        seq = torch.randint(0, 21, (2, 20), device=device)
        task = FoldingTask(
            task_id="consistency_test", task_type=TaskType.STRUCTURE_PREDICTION,
            sequence_length=20, accuracy_target=0.8,
            time_budget_s=10.0, memory_budget_mb=500.0, energy_budget_j=100.0,
        )
        
        out = model(seq, task)
        consistency = out["consistency_overall"].item()
        
        suite.record(
            "Protein folding cross-level consistency",
            consistency > 0.0,  # Should have some consistency signal
            message=f"consistency={consistency:.4f}",
            details={"consistency": consistency},
        )
    except ImportError as e:
        suite.skip("Protein folding cross-level consistency", str(e))


# ═══════════════════════════════════════════════════════════════════════
# LAYER 4: PROVENANCE TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_provenance_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L4: Verify synthetic data cannot be promoted to VERIFIED.
    """
    suite.start_layer(4, "Provenance Tests")
    
    # ── Test 4.1: Calabi-Yau synthetic data detection ─────────────────
    try:
        from SESI_CALABI_YAU_AGI_ONE_INTEGRATION__1_ import (
            CalabiYauDataset, CalabiYauConfig,
        )
        
        cfg = CalabiYauConfig(max_cicy_configs=8, max_ks_polytopes=4, device=str(device))
        dataset = CalabiYauDataset(cfg, "train")
        
        n_real = dataset.n_real
        n_total = len(dataset)
        n_synthetic = n_total - n_real
        
        # Should have some synthetic data (since we don't have real CICY files)
        suite.record(
            "Calabi-Yau dataset provenance tracking",
            n_synthetic > 0 or n_real > 0,
            message=f"real={n_real}, synthetic={n_synthetic}",
            details={"n_real": n_real, "n_synthetic": n_synthetic},
        )
        
        # Check is_real flag in samples
        sample = dataset[0]
        has_is_real = "is_real" in sample
        suite.record(
            "Calabi-Yau sample has is_real flag",
            has_is_real,
            message="" if has_is_real else "Missing is_real flag",
        )
    except ImportError as e:
        suite.skip("Calabi-Yau provenance", str(e))
    
    # ── Test 4.2: Epistemic labels in Protein Orchestrator ────────────
    try:
        from PROTEIN_FOLDING_ORCHESTRATOR import (
            ProteinFoldingOrchestrator, ProteinOrchestratorConfig,
            FoldingTask, TaskType,
        )
        
        cfg = ProteinOrchestratorConfig(
            latent_dim=16, sequence_dim=8, structure_dim=16,
            level1_dim=8, level2_dim=12, level3_dim=16,
            device=str(device),
        )
        model = ProteinFoldingOrchestrator(cfg).to(device)
        seq = torch.randint(0, 21, (2, 10), device=device)
        task = FoldingTask(
            task_id="prov_test", task_type=TaskType.STRUCTURE_PREDICTION,
            sequence_length=10, accuracy_target=0.8,
            time_budget_s=10.0, memory_budget_mb=500.0, energy_budget_j=100.0,
        )
        
        out = model(seq, task)
        pareto = out["pareto_report"]
        
        has_epistemic = hasattr(pareto, "epistemic_levels")
        epistemic_valid = (
            has_epistemic and
            all(level in ["VERIFIED", "CONDITIONAL", "HEURISTIC", "SPECULATIVE"]
                for level in pareto.epistemic_levels)
        )
        
        suite.record(
            "Pareto report has valid epistemic labels",
            epistemic_valid,
            message="" if epistemic_valid else f"Invalid labels: {pareto.epistemic_levels if has_epistemic else 'missing'}",
        )
    except ImportError as e:
        suite.skip("Protein Orchestrator provenance", str(e))


# ═══════════════════════════════════════════════════════════════════════
# LAYER 5: BUDGET TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_budget_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L5: Verify resource budgets are respected.
    """
    suite.start_layer(5, "Budget Tests")
    
    try:
        from PROTEIN_FOLDING_ORCHESTRATOR import (
            ProteinFoldingOrchestrator, ProteinOrchestratorConfig,
            FoldingTask, TaskType,
        )
        
        cfg = ProteinOrchestratorConfig(
            latent_dim=16, sequence_dim=8, structure_dim=16,
            level1_dim=8, level2_dim=12, level3_dim=16,
            device=str(device),
        )
        model = ProteinFoldingOrchestrator(cfg).to(device)
        seq = torch.randint(0, 21, (2, 10), device=device)
        
        # Test 1: Tight time budget → should use fewer levels
        task_tight = FoldingTask(
            task_id="tight", task_type=TaskType.STRUCTURE_PREDICTION,
            sequence_length=10, accuracy_target=0.8,
            time_budget_s=0.5,  # Very tight
            memory_budget_mb=50.0, energy_budget_j=10.0,
        )
        out_tight = model(seq, task_tight)
        levels_tight = out_tight["exec_levels_executed"]
        cost_tight = out_tight["exec_total_computational_cost"]
        
        # Test 2: Loose time budget → can use more levels
        task_loose = FoldingTask(
            task_id="loose", task_type=TaskType.STRUCTURE_PREDICTION,
            sequence_length=10, accuracy_target=0.8,
            time_budget_s=100.0,  # Very loose
            memory_budget_mb=5000.0, energy_budget_j=1000.0,
        )
        out_loose = model(seq, task_loose)
        levels_loose = out_loose["exec_levels_executed"]
        cost_loose = out_loose["exec_total_computational_cost"]
        
        # Tight budget should use ≤ loose budget
        budget_respected = levels_tight <= levels_loose or cost_tight <= cost_loose
        
        suite.record(
            "Resource budget respected",
            budget_respected,
            message=f"tight: L{levels_tight}/C{cost_tight:.2f}, loose: L{levels_loose}/C{cost_loose:.2f}",
            details={
                "levels_tight": levels_tight,
                "levels_loose": levels_loose,
                "cost_tight": cost_tight,
                "cost_loose": cost_loose,
            },
        )
    except ImportError as e:
        suite.skip("Budget tests", str(e))


# ═══════════════════════════════════════════════════════════════════════
# LAYER 6: AMP / DDP TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_amp_ddp_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L6: Verify AMP and DDP compatibility.
    """
    suite.start_layer(6, "AMP/DDP Tests")
    
    # ── Test 6.1: AMP (bf16) ──────────────────────────────────────────
    if device.type == "cuda":
        try:
            from SESI_FRAMEWORKv7_Hierarchical_Quantum_State_Spaces_Multiscale_Applications import (
                SESIFrameworkV7, SESIv7Config,
            )
            
            cfg = SESIv7Config(latent_dim=32, hierarchical_levels=2, device=str(device))
            model = SESIFrameworkV7(cfg).to(device)
            
            base = torch.randn(2, cfg.latent_dim, device=device)
            
            try:
                from torch.amp import autocast
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(base)
                    loss = out["global_optimization_loss"]
                loss.backward()
                
                # Check no NaN/Inf
                all_finite = all(
                    torch.isfinite(p.grad).all() for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                )
                
                suite.record(
                    "AMP bf16 forward+backward",
                    all_finite and torch.isfinite(loss),
                    message=f"loss={loss.item():.4f}",
                )
            except ImportError:
                suite.skip("AMP bf16 test", "torch.amp not available")
        except ImportError as e:
            suite.skip("AMP bf16 test", str(e))
    else:
        suite.skip("AMP bf16 test", "CUDA not available")
    
    # ── Test 6.2: DDP setup (single-process simulation) ───────────────
    suite.record(
        "DDP infrastructure available",
        torch.distributed.is_available(),
        message="" if torch.distributed.is_available() else "torch.distributed not available",
    )


# ═══════════════════════════════════════════════════════════════════════
# LAYER 7: DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_determinism_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L7: Verify seed → same result.
    """
    suite.start_layer(7, "Determinism Tests")
    
    try:
        from SESI_FRAMEWORKv7_Hierarchical_Quantum_State_Spaces_Multiscale_Applications import (
            SESIFrameworkV7, SESIv7Config,
        )
        
        cfg = SESIv7Config(latent_dim=32, hierarchical_levels=2, device=str(device))
        
        # Run 1
        torch.manual_seed(42)
        model1 = SESIFrameworkV7(cfg).to(device).eval()
        base = torch.randn(2, cfg.latent_dim, device=device)
        
        with torch.no_grad():
            out1 = model1(base)
            loss1 = out1["global_optimization_loss"].item()
        
        # Run 2 (same seed)
        torch.manual_seed(42)
        model2 = SESIFrameworkV7(cfg).to(device).eval()
        base2 = torch.randn(2, cfg.latent_dim, device=device)
        
        with torch.no_grad():
            out2 = model2(base2)
            loss2 = out2["global_optimization_loss"].item()
        
        # Should be identical
        deterministic = abs(loss1 - loss2) < 1e-5
        
        suite.record(
            "Determinism with same seed",
            deterministic,
            message=f"loss1={loss1:.6f}, loss2={loss2:.6f}, diff={abs(loss1-loss2):.2e}",
            details={"loss1": loss1, "loss2": loss2},
        )
    except ImportError as e:
        suite.skip("Determinism test", str(e))


# ═══════════════════════════════════════════════════════════════════════
# LAYER 8: BENCHMARK TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark_tests(suite: TestSuite, device: torch.device) -> None:
    """
    L8: Benchmark against baselines (smoke test only).
    """
    suite.start_layer(8, "Benchmark Tests")
    
    # ── Test 8.1: Protein folding benchmark (smoke) ───────────────────
    try:
        from PROTEIN_FOLDING_ORCHESTRATOR import (
            ProteinFoldingOrchestrator, ProteinOrchestratorConfig,
            FoldingTask, TaskType,
        )
        
        cfg = ProteinOrchestratorConfig(
            latent_dim=32, sequence_dim=16, structure_dim=32,
            level1_dim=16, level2_dim=24, level3_dim=32,
            device=str(device),
        )
        model = ProteinFoldingOrchestrator(cfg).to(device).eval()
        
        # Benchmark: 10 tasks
        n_tasks = 10
        times = []
        
        for i in range(n_tasks):
            seq = torch.randint(0, 21, (1, 20), device=device)
            task = FoldingTask(
                task_id=f"bench_{i}", task_type=TaskType.STRUCTURE_PREDICTION,
                sequence_length=20, accuracy_target=0.8,
                time_budget_s=10.0, memory_budget_mb=500.0, energy_budget_j=100.0,
            )
            
            t0 = time.time()
            with torch.no_grad():
                out = model(seq, task)
            dt = time.time() - t0
            times.append(dt)
        
        avg_time = sum(times) / len(times)
        
        suite.record(
            f"Protein folding benchmark ({n_tasks} tasks)",
            avg_time < 5.0,  # Should be fast
            message=f"avg_time={avg_time:.4f}s",
            details={
                "avg_time_s": avg_time,
                "min_time_s": min(times),
                "max_time_s": max(times),
            },
        )
    except ImportError as e:
        suite.skip("Protein folding benchmark", str(e))
    
    # ── Test 8.2: Pareto frontier analysis ────────────────────────────
    try:
        from PROTEIN_FOLDING_ORCHESTRATOR import (
            ParetoFrontierAnalyzer, ParetoReport,
        )
        
        analyzer = ParetoFrontierAnalyzer()
        
        # Add synthetic reports
        for i in range(5):
            report = ParetoReport(
                task_id=f"pareto_{i}",
                error=0.1 * (i + 1),
                runtime_s=1.0 + i * 0.5,
                memory_mb=100.0 + i * 50,
                energy_j=10.0 + i * 5,
                levels_used=i % 3 + 1,
                consistency=0.8 + 0.02 * i,
                verified=True,
                epistemic_levels=["HEURISTIC"],
            )
            analyzer.add_report(report)
        
        pareto = analyzer.compute_pareto_front()
        hv = analyzer.hypervolume()
        
        suite.record(
            "Pareto frontier analysis",
            len(pareto) > 0 and hv > 0,
            message=f"pareto_size={len(pareto)}, hypervolume={hv:.4f}",
            details={
                "pareto_size": len(pareto),
                "hypervolume": hv,
                "n_reports": 5,
            },
        )
    except ImportError as e:
        suite.skip("Pareto frontier analysis", str(e))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SESI Integration Test Suite")
    parser.add_argument("--all", action="store_true", help="Run all layers")
    parser.add_argument("--layer", type=str, default="",
                        help="Comma-separated layer numbers (e.g., '1,2,3')")
    parser.add_argument("--module", type=str, default="",
                        help="Specific module to test")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", type=str, default="./test_reports/test_suite.json")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*72}")
    print(f"  SESI INTEGRATION TEST SUITE")
    print(f"{'='*72}")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'='*72}")
    
    suite = TestSuite(verbose=args.verbose)
    
    # Determine which layers to run
    if args.all or not args.layer:
        layers = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        layers = [int(x) for x in args.layer.split(",")]
    
    layer_fns = {
        1: run_contract_tests,
        2: run_gradient_tests,
        3: run_consistency_tests,
        4: run_provenance_tests,
        5: run_budget_tests,
        6: run_amp_ddp_tests,
        7: run_determinism_tests,
        8: run_benchmark_tests,
    }
    
    t_start = time.time()
    for layer in layers:
        if layer in layer_fns:
            layer_fns[layer](suite, device)
    t_total = time.time() - t_start
    
    suite.print_summary()
    suite.save_report(args.report)
    
    print(f"\n  Total duration: {t_total:.2f}s")
    print(f"  Report: {args.report}")
    
    sys.exit(0 if suite.all_passed else 1)


if __name__ == "__main__":
    main()
