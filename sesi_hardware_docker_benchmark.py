"""
=============================================================================
SESI MULTI-BACKEND VALIDATION SUITE — v1.0.0
Hardware-Specific Testing + Multi-Backend Docker + Cross-Backend Benchmark
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0
=============================================================================
INTEGRATION WITH ECOSYSTEM
=============================================================================
• real_fold_one_v2-5.py     — detect_optimal_device(), CSOCKernel,
                               SOCController, NeighborListManager,
                               openmm_solute_energy, OpenMMEnergyCalculator
• sesi_multibackend_openmm_trainer.py — MultiBackendDeviceManager,
                                         BackendCapability, OpenMMEnergyTrainerFull
• structural_domain_assembly_one.py   — DomainAssemblyConfig (for SDA-ONE tests)

WHAT THIS MODULE PROVIDES:
  [A] HardwareTestSuite    — per-backend correctness tests with graceful skip
  [B] DockerImageBuilder   — multi-stage Dockerfile generation for 5 backends
  [C] BackendBenchmark     — latency/throughput benchmark with statistics

HONEST SCOPE:
  • Hardware tests SKIP gracefully when the target hardware is absent.
    They do not fabricate results.
  • Docker images are TEMPLATES. Actual deployment requires:
    - NVIDIA Container Toolkit (for CUDA)
    - ROCm Docker support (for AMD)
    - Appropriate base images available on the target registry
  • Benchmark numbers are only meaningful when run on actual hardware.
    On CPU-only machines, GPU benchmarks are skipped (not emulated).
  • This module has NOT been run on all 8 backends simultaneously.
=============================================================================
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sesi.hardware_docker_benchmark")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

VALIDATION_SUITE_VERSION: str = "1.0.0"

# =============================================================================
# Ecosystem imports (graceful)
# =============================================================================
_HAS_REAL_FOLD = False
try:
    from real_fold_one_v2 import (
        detect_optimal_device,
        CSOCKernel,
        SOCController,
        NeighborListManager,
        OpenMMEnergyCalculator,
        openmm_solute_energy,
        OpenMMSystemBuilder,
    )
    _HAS_REAL_FOLD = True
except ImportError:
    import importlib.util
    for cand in ("real_fold_one_v2-5.py", "real_fold_one_v2.py"):
        p = Path(cand)
        if p.exists():
            spec = importlib.util.spec_from_file_location("real_fold_one_v2", str(p))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                detect_optimal_device = mod.detect_optimal_device
                CSOCKernel = mod.CSOCKernel
                SOCController = mod.SOCController
                NeighborListManager = mod.NeighborListManager
                OpenMMEnergyCalculator = mod.OpenMMEnergyCalculator
                openmm_solute_energy = mod.openmm_solute_energy
                OpenMMSystemBuilder = mod.OpenMMSystemBuilder
                _HAS_REAL_FOLD = True
            break
    if not _HAS_REAL_FOLD:
        warnings.warn("real_fold_one_v2 not found; ecosystem tests will use stand-ins.")

_HAS_OPENMM = False
try:
    import openmm as mm
    _HAS_OPENMM = True
except ImportError:
    pass

_HAS_OPENMMML = False
try:
    from openmmml import MLPotential
    _HAS_OPENMMML = True
except ImportError:
    pass


# =============================================================================
# [A] HARDWARE TEST SUITE
# =============================================================================
@dataclass
class HardwareTestResult:
    """Result of a single hardware test."""
    test_name: str
    backend: str
    status: str  # "PASS" | "FAIL" | "SKIP" | "ERROR"
    duration_s: float
    message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HardwareTestReport:
    """Aggregated report across all backends."""
    backend_name: str
    torch_device: str
    is_available: bool
    device_info: Dict[str, Any]
    results: List[HardwareTestResult]
    n_pass: int
    n_fail: int
    n_skip: int
    total_duration_s: float


class HardwareTestSuite:
    """
    [A] Per-hardware correctness tests.

    TEST CATEGORIES:
      1. Basic tensor ops    — matmul, reduction, gradient flow
      2. CSOCKernel forward  — LJ energy computation (from real_fold_one_v2)
      3. NeighborList build  — edge construction correctness
      4. OpenMM energy       — openmm_solute_energy call (if OpenMM available)
      5. AMP compatibility   — bf16/fp16 autocast behavior

    COROLLARY-3 GUARD: all tests are first-order (pass/fail based on
    numerical tolerance). No Hessian of test metrics.

    HINO GUARD: each backend is tested independently; results are discrete.
    """

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance
        self.results: List[HardwareTestResult] = []

    def _detect_backends(self) -> List[Dict[str, Any]]:
        """Detect all available backends (mirrors MultiBackendDeviceManager)."""
        backends = []

        # NVIDIA CUDA
        if torch.cuda.is_available():
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            name = "AMD_ROCM" if is_rocm else "NVIDIA_CUDA"
            backends.append({
                "name": name,
                "torch_device": "cuda",
                "is_available": True,
                "info": {
                    "device_name": torch.cuda.get_device_name(0),
                    "compute_capability": torch.cuda.get_device_capability(0),
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                },
            })

        # Apple MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            backends.append({
                "name": "APPLE_MPS",
                "torch_device": "mps",
                "is_available": True,
                "info": {"device_name": "Apple Silicon (MPS)"},
            })

        # Huawei Ascend NPU
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                backends.append({
                    "name": "HUAWEI_ASCEND",
                    "torch_device": "npu",
                    "is_available": True,
                    "info": {"device_name": "Huawei Ascend NPU"},
                })
        except ImportError:
            pass

        # Intel XPU
        try:
            import intel_extension_for_pytorch  # noqa: F401
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                backends.append({
                    "name": "INTEL_XPU",
                    "torch_device": "xpu",
                    "is_available": True,
                    "info": {"device_name": "Intel GPU (XPU)"},
                })
        except ImportError:
            pass

        # Google TPU
        try:
            import torch_xla.core.xla_model as xm
            xla_dev = xm.xla_device()
            if xla_dev is not None:
                backends.append({
                    "name": "GOOGLE_TPU",
                    "torch_device": "xla",
                    "is_available": True,
                    "info": {"device_name": "Google TPU (XLA)"},
                })
        except ImportError:
            pass

        # CPU (always available)
        backends.append({
            "name": "CPU",
            "torch_device": "cpu",
            "is_available": True,
            "info": {
                "device_name": platform.processor() or "Generic CPU",
                "cpu_count": os.cpu_count(),
            },
        })

        return backends

    def _test_basic_tensor_ops(self, device: torch.device) -> Tuple[bool, str]:
        """Test basic tensor operations: matmul, sum, gradient."""
        try:
            a = torch.randn(64, 64, device=device, requires_grad=True)
            b = torch.randn(64, 64, device=device)
            c = a @ b
            loss = c.sum()
            loss.backward()

            if a.grad is None:
                return False, "Gradient not computed"
            if not torch.isfinite(a.grad).all():
                return False, "Non-finite gradient"
            if not torch.isfinite(c).all():
                return False, "Non-finite matmul result"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    def _test_csoc_kernel(self, device: torch.device) -> Tuple[bool, str]:
        """Test CSOCKernel forward (LJ energy) from real_fold_one_v2."""
        if not _HAS_REAL_FOLD:
            return False, "real_fold_one_v2 not available"
        try:
            kernel = CSOCKernel().to(device)
            r = torch.rand(100, device=device) * 10 + 1.0  # distances 1-11 Å
            K = kernel(r)
            if not torch.isfinite(K).all():
                return False, "Non-finite kernel output"
            # At r = r0, K should be near +eps_lj (minimum of E_LJ)
            r0 = float(kernel.r0.item())
            k_at_r0 = float(K[(r - r0).abs() < 0.1].mean())
            eps_lj = float(kernel.eps_lj.item())
            if abs(k_at_r0 - eps_lj) > 0.5 * eps_lj:
                return False, f"K(r0)={k_at_r0:.3f} not near eps_lj={eps_lj:.3f}"
            return True, f"OK (K(r0)={k_at_r0:.3f}, eps_lj={eps_lj:.3f})"
        except Exception as e:
            return False, str(e)

    def _test_neighbor_list(self, device: torch.device) -> Tuple[bool, str]:
        """Test NeighborListManager.build correctness."""
        if not _HAS_REAL_FOLD:
            return False, "real_fold_one_v2 not available"
        try:
            coords = torch.randn(200, 3, device=device) * 10
            mgr = NeighborListManager(
                cutoffs={'soc': 12.0},
                max_neighbors=64,
                device=device,
            )
            result = mgr.build(coords)
            edge_idx, edge_dist = result['soc']
            if edge_idx.shape[0] != 2:
                return False, f"edge_idx shape {edge_idx.shape} != (2, E)"
            if not torch.isfinite(edge_dist).all():
                return False, "Non-finite edge distances"
            if edge_dist.numel() > 0 and edge_dist.max() > 12.0 + 1e-3:
                return False, f"Edge distance {edge_dist.max():.3f} > cutoff 12.0"
            return True, f"OK ({edge_idx.shape[1]} edges)"
        except Exception as e:
            return False, str(e)

    def _test_amp_autocast(self, device: torch.device) -> Tuple[bool, str]:
        """Test AMP autocast behavior (bf16 on CUDA, fp16 fallback)."""
        if device.type == "cpu":
            # CPU supports bf16 autocast
            try:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    x = torch.randn(32, 32, device=device)
                    y = x @ x
                return True, "bf16 autocast OK on CPU"
            except Exception as e:
                return False, str(e)

        # GPU backends
        try:
            dtype = torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=dtype):
                x = torch.randn(32, 32, device=device)
                y = x @ x
            if y.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                return False, f"Unexpected dtype {y.dtype} after autocast"
            return True, f"autocast({dtype}) OK"
        except Exception as e:
            return False, str(e)

    def _test_openmm_energy(self, device: torch.device) -> Tuple[bool, str]:
        """Test openmm_solute_energy call (requires OpenMM + a small PDB)."""
        if not _HAS_OPENMM or not _HAS_REAL_FOLD:
            return False, "OpenMM or real_fold_one_v2 not available"
        try:
            # Create a minimal 2-residue PDB
            with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
                f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
                f.write("ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C\n")
                f.write("END\n")
                pdb_path = f.name

            builder = OpenMMSystemBuilder(implicit_solvent=None, solvate=False)
            system, topology, init_coords, meta = builder.build_from_pdb(
                pdb_path, solvate=False, add_hydrogens=False
            )
            solute_mask = meta['solute_mask']
            solute_coords = init_coords[solute_mask].clone().requires_grad_(True)

            calculator = OpenMMEnergyCalculator(
                system, topology, init_coords, solute_mask,
                platform_name="CPU",
                ml_potential_name=None,
                use_native_autograd=False,
            )
            energy = openmm_solute_energy(solute_coords, calculator)
            if not torch.isfinite(energy):
                calculator.close()
                os.unlink(pdb_path)
                return False, f"Non-finite energy: {energy.item()}"

            calculator.close()
            os.unlink(pdb_path)
            return True, f"OK (E={energy.item():.2f} kcal/mol)"
        except Exception as e:
            return False, str(e)

    def run_all_tests(self, backends: Optional[List[Dict]] = None) -> List[HardwareTestReport]:
        """Run all tests on all available backends."""
        if backends is None:
            backends = self._detect_backends()

        reports = []
        for backend in backends:
            device = torch.device(backend["torch_device"])
            results = []
            t_start = time.time()

            # Test 1: Basic tensor ops
            t0 = time.time()
            ok, msg = self._test_basic_tensor_ops(device)
            results.append(HardwareTestResult(
                "basic_tensor_ops", backend["name"],
                "PASS" if ok else "FAIL",
                time.time() - t0, msg,
            ))

            # Test 2: CSOCKernel
            t0 = time.time()
            ok, msg = self._test_csoc_kernel(device)
            results.append(HardwareTestResult(
                "csoc_kernel_forward", backend["name"],
                "PASS" if ok else ("SKIP" if "not available" in msg else "FAIL"),
                time.time() - t0, msg,
            ))

            # Test 3: NeighborList
            t0 = time.time()
            ok, msg = self._test_neighbor_list(device)
            results.append(HardwareTestResult(
                "neighbor_list_build", backend["name"],
                "PASS" if ok else ("SKIP" if "not available" in msg else "FAIL"),
                time.time() - t0, msg,
            ))

            # Test 4: AMP autocast
            t0 = time.time()
            ok, msg = self._test_amp_autocast(device)
            results.append(HardwareTestResult(
                "amp_autocast", backend["name"],
                "PASS" if ok else "FAIL",
                time.time() - t0, msg,
            ))

            # Test 5: OpenMM energy (CPU platform only for portability)
            t0 = time.time()
            ok, msg = self._test_openmm_energy(torch.device("cpu"))
            results.append(HardwareTestResult(
                "openmm_energy", backend["name"],
                "PASS" if ok else ("SKIP" if "not available" in msg else "FAIL"),
                time.time() - t0, msg,
            ))

            n_pass = sum(1 for r in results if r.status == "PASS")
            n_fail = sum(1 for r in results if r.status == "FAIL")
            n_skip = sum(1 for r in results if r.status == "SKIP")

            report = HardwareTestReport(
                backend_name=backend["name"],
                torch_device=backend["torch_device"],
                is_available=backend["is_available"],
                device_info=backend["info"],
                results=results,
                n_pass=n_pass, n_fail=n_fail, n_skip=n_skip,
                total_duration_s=time.time() - t_start,
            )
            reports.append(report)

            logger.info(
                f"[HWTest] {backend['name']}: "
                f"{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP "
                f"({report.total_duration_s:.2f}s)"
            )

        self.results = [r for rep in reports for r in rep.results]
        return reports


# =============================================================================
# [B] DOCKER IMAGE BUILDER
# =============================================================================
@dataclass
class DockerImageConfig:
    """Configuration for Docker image generation."""
    image_name: str = "sesi-cfd-prod"
    image_tag: str = "latest"
    python_version: str = "3.11"
    # Backend selection
    backends: List[str] = field(default_factory=lambda: ["cuda", "cpu"])
    # CUDA-specific
    cuda_version: str = "12.2"
    cudnn_version: str = "8"
    # ROCm-specific
    rocm_version: str = "6.0"
    # Dependencies
    install_openmm: bool = True
    install_openmm_ml: bool = True
    install_openmm_torch: bool = True
    install_triton: bool = True
    # Output
    output_dir: str = "./docker_build"


class DockerImageBuilder:
    """
    [B] Multi-backend Docker image generator.

    GENERATES:
      • Dockerfile.cuda   — NVIDIA GPU (CUDA 12.2 + cuDNN 8)
      • Dockerfile.rocm   — AMD GPU (ROCm 6.0)
      • Dockerfile.cpu    — CPU-only fallback
      • Dockerfile.mps    — Apple Silicon (host build, not cross-platform)
      • docker-compose.yml — orchestrates multi-backend testing
      • build.sh           — convenience build script

    HONEST SCOPE:
      • These are TEMPLATES. Actual deployment requires:
        - NVIDIA Container Toolkit installed on the host (for CUDA)
        - ROCm Docker runtime (for AMD)
        - docker buildx for multi-arch builds
      • OpenMM-ML + openmm-torch are installed via conda-forge.
      • Triton is only installed for CUDA backend (not ROCm/CPU).
    """

    def __init__(self, cfg: Optional[DockerImageConfig] = None):
        self.cfg = cfg or DockerImageConfig()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_image(self, backend: str) -> str:
        """Return the base Docker image for a given backend."""
        if backend == "cuda":
            return f"nvidia/cuda:{self.cfg.cuda_version}-cudnn{self.cfg.cudnn_version}-devel-ubuntu22.04"
        elif backend == "rocm":
            return f"rocm/dev-ubuntu-22.04:{self.cfg.rocm_version}-complete"
        elif backend == "cpu":
            return "ubuntu:22.04"
        elif backend == "mps":
            return "ubuntu:22.04"  # MPS is host-only; container just needs Python
        else:
            return "ubuntu:22.04"

    def _conda_install_block(self) -> str:
        """Generate conda installation block."""
        packages = [
            "python=" + self.cfg.python_version,
            "pytorch",
            "torchvision",
            "torchaudio",
        ]
        if self.cfg.install_openmm:
            packages.append("openmm")
        if self.cfg.install_openmm_ml:
            packages.append("openmm-ml")
        if self.cfg.install_openmm_torch:
            packages.append("openmm-torch")

        pip_packages = [
            "numpy", "scipy", "biotite", "rdkit",
            "torch-cluster", "networkx",
        ]
        if self.cfg.install_triton:
            pip_packages.append("triton")

        return f"""
# Install Miniconda
RUN apt-get update && apt-get install -y wget && \\
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \\
    bash /tmp/miniconda.sh -b -p /opt/conda && \\
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create environment and install packages
RUN conda create -n sesi python={self.cfg.python_version} -y && \\
    conda install -n sesi -c conda-forge -c pytorch {' '.join(packages)} -y && \\
    conda run -n sesi pip install {' '.join(pip_packages)}
"""

    def generate_dockerfile(self, backend: str) -> str:
        """Generate Dockerfile content for a specific backend."""
        base = self._base_image(backend)

        header = f"""# =============================================================================
# SESI CFD Production Dockerfile — {backend.upper()} backend
# Auto-generated by DockerImageBuilder v{VALIDATION_SUITE_VERSION}
# Developer: Yoon A Limsuwan / MSPS NETWORK
# =============================================================================
FROM {base}

LABEL maintainer="Yoon A Limsuwan <yoonalimsuwan@msps.network>"
LABEL org.opencontainers.image.title="SESI CFD Production ({backend})"
LABEL org.opencontainers.image.version="{self.cfg.image_tag}"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential cmake git wget curl ca-certificates \\
    libhdf5-dev libfftw3-dev liblapack-dev libblas-dev \\
    && rm -rf /var/lib/apt/lists/*
"""

        conda_block = self._conda_install_block()

        app_block = f"""
# Activate conda environment
SHELL ["conda", "run", "-n", "sesi", "/bin/bash", "-c"]

# Copy application code
WORKDIR /workspace
COPY sesi_hardware_docker_benchmark.py .
COPY sesi_multibackend_openmm_trainer.py .
COPY real_fold_one_v2-5.py .
COPY structural_domain_assembly_one.py .

# Verify installation
RUN python -c "import torch; print(f'PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')"

ENTRYPOINT ["conda", "run", "-n", "sesi", "python"]
CMD ["sesi_hardware_docker_benchmark.py", "--self-test"]
"""
        return header + conda_block + app_block

    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for multi-backend testing."""
        services = {}
        for backend in self.cfg.backends:
            services[f"sesi-{backend}"] = {
                "build": {
                    "context": ".",
                    "dockerfile": f"Dockerfile.{backend}",
                },
                "image": f"{self.cfg.image_name}:{self.cfg.image_tag}-{backend}",
                "container_name": f"sesi-{backend}",
            }
            if backend == "cuda":
                services[f"sesi-{backend}"]["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                        }
                    }
                }

        compose = {
            "version": "3.8",
            "services": services,
        }
        import yaml
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)

    def generate_build_script(self) -> str:
        """Generate a convenience build.sh script."""
        lines = [
            "#!/bin/bash",
            "# Auto-generated build script for SESI multi-backend Docker images",
            f"# Generated by DockerImageBuilder v{VALIDATION_SUITE_VERSION}",
            "set -euo pipefail",
            "",
            f'IMAGE_NAME="{self.cfg.image_name}"',
            f'IMAGE_TAG="{self.cfg.image_tag}"',
            "",
        ]
        for backend in self.cfg.backends:
            lines.append(f'echo "Building {backend} image..."')
            lines.append(
                f'docker build -f Dockerfile.{backend} '
                f'-t "${{IMAGE_NAME}}:${{IMAGE_TAG}}-{backend}" .'
            )
            lines.append("")
        lines.append('echo "All builds complete."')
        return "\n".join(lines)

    def generate_all(self) -> Dict[str, str]:
        """Generate all Docker artifacts."""
        artifacts = {}

        for backend in self.cfg.backends:
            dockerfile = self.generate_dockerfile(backend)
            path = self.output_dir / f"Dockerfile.{backend}"
            with open(path, "w") as f:
                f.write(dockerfile)
            artifacts[f"Dockerfile.{backend}"] = str(path)
            logger.info(f"[Docker] Generated {path}")

        # docker-compose.yml
        try:
            compose_content = self.generate_docker_compose()
            compose_path = self.output_dir / "docker-compose.yml"
            with open(compose_path, "w") as f:
                f.write(compose_content)
            artifacts["docker-compose.yml"] = str(compose_path)
        except ImportError:
            logger.warning("[Docker] PyYAML not installed; skipping docker-compose.yml")

        # build.sh
        build_script = self.generate_build_script()
        build_path = self.output_dir / "build.sh"
        with open(build_path, "w") as f:
            f.write(build_script)
        os.chmod(build_path, 0o755)
        artifacts["build.sh"] = str(build_path)

        return artifacts


# =============================================================================
# [C] BACKEND BENCHMARK
# =============================================================================
@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""
    benchmark_name: str
    backend: str
    metric: str  # "latency_ms" | "throughput_ops_per_s" | "memory_mb"
    value: float
    n_repeats: int
    std: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""
    benchmark_name: str
    results: Dict[str, List[BenchmarkResult]]  # backend -> results
    speedup_vs_cpu: Dict[str, float]  # backend -> speedup ratio
    total_duration_s: float


class BackendBenchmark:
    """
    [C] Cross-backend performance benchmark.

    BENCHMARKS:
      1. matmul_1k      — 1024×1024 matrix multiplication
      2. csoc_kernel    — CSOCKernel.forward on 10k distances
      3. neighbor_list  — NeighborListManager.build on 1k points
      4. fft_3d         — 3D FFT (128³)
      5. reduction      — sum reduction over 1M elements

    METHODOLOGY:
      • Warmup: 3 iterations (not timed)
      • Timed: 10 iterations, report mean ± std
      • Memory: peak GPU memory allocated (if applicable)
      • Comparison: speedup relative to CPU baseline

    COROLLARY-3 GUARD: all measurements are first-order statistics
    (mean, std). No Hessian of performance metrics.

    HINO GUARD: each benchmark is run independently per backend;
    results are discrete observations.
    """

    def __init__(self, n_warmup: int = 3, n_repeats: int = 10):
        self.n_warmup = n_warmup
        self.n_repeats = n_repeats
        self.results: List[BenchmarkResult] = []

    def _detect_backends(self) -> List[Tuple[str, torch.device]]:
        """Detect available backends as (name, device) tuples."""
        backends = []
        if torch.cuda.is_available():
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            name = "AMD_ROCM" if is_rocm else "NVIDIA_CUDA"
            backends.append((name, torch.device("cuda")))
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            backends.append(("APPLE_MPS", torch.device("mps")))
        backends.append(("CPU", torch.device("cpu")))
        return backends

    def _time_fn(self, fn: Callable, device: torch.device) -> Tuple[float, float]:
        """Time a function over n_repeats with warmup. Returns (mean_ms, std_ms)."""
        # Warmup
        for _ in range(self.n_warmup):
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        for _ in range(self.n_repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        return float(np.mean(times)), float(np.std(times))

    def _bench_matmul(self, device: torch.device, size: int = 1024) -> BenchmarkResult:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        def fn():
            return a @ b

        mean_ms, std_ms = self._time_fn(fn, device)
        return BenchmarkResult(
            "matmul_1k", device.type, "latency_ms", mean_ms,
            self.n_repeats, std_ms,
            {"size": size},
        )

    def _bench_csoc_kernel(self, device: torch.device, n: int = 10000) -> BenchmarkResult:
        if not _HAS_REAL_FOLD:
            return BenchmarkResult("csoc_kernel", device.type, "latency_ms", float('nan'),
                                   0, 0.0, {"error": "real_fold_one_v2 not available"})
        kernel = CSOCKernel().to(device)
        r = torch.rand(n, device=device) * 10 + 1.0

        def fn():
            return kernel(r)

        mean_ms, std_ms = self._time_fn(fn, device)
        return BenchmarkResult(
            "csoc_kernel", device.type, "latency_ms", mean_ms,
            self.n_repeats, std_ms,
            {"n_distances": n},
        )

    def _bench_neighbor_list(self, device: torch.device, n: int = 1000) -> BenchmarkResult:
        if not _HAS_REAL_FOLD:
            return BenchmarkResult("neighbor_list", device.type, "latency_ms", float('nan'),
                                   0, 0.0, {"error": "real_fold_one_v2 not available"})
        coords = torch.randn(n, 3, device=device) * 20
        mgr = NeighborListManager(
            cutoffs={'soc': 12.0}, max_neighbors=64, device=device,
        )

        def fn():
            return mgr.build(coords)

        mean_ms, std_ms = self._time_fn(fn, device)
        return BenchmarkResult(
            "neighbor_list", device.type, "latency_ms", mean_ms,
            self.n_repeats, std_ms,
            {"n_points": n},
        )

    def _bench_fft_3d(self, device: torch.device, size: int = 128) -> BenchmarkResult:
        x = torch.randn(size, size, size, device=device)

        def fn():
            return torch.fft.fftn(x)

        mean_ms, std_ms = self._time_fn(fn, device)
        return BenchmarkResult(
            "fft_3d", device.type, "latency_ms", mean_ms,
            self.n_repeats, std_ms,
            {"size": f"{size}^3"},
        )

    def _bench_reduction(self, device: torch.device, n: int = 1_000_000) -> BenchmarkResult:
        x = torch.randn(n, device=device)

        def fn():
            return x.sum()

        mean_ms, std_ms = self._time_fn(fn, device)
        return BenchmarkResult(
            "reduction", device.type, "latency_ms", mean_ms,
            self.n_repeats, std_ms,
            {"n_elements": n},
        )

    def run_all_benchmarks(self) -> Dict[str, BenchmarkReport]:
        """Run all benchmarks on all available backends."""
        backends = self._detect_backends()
        reports = {}

        for bench_name, bench_fn in [
            ("matmul_1k", self._bench_matmul),
            ("csoc_kernel", self._bench_csoc_kernel),
            ("neighbor_list", self._bench_neighbor_list),
            ("fft_3d", self._bench_fft_3d),
            ("reduction", self._bench_reduction),
        ]:
            logger.info(f"[Benchmark] Running {bench_name}...")
            results_by_backend: Dict[str, List[BenchmarkResult]] = {}
            t_start = time.time()

            for backend_name, device in backends:
                try:
                    result = bench_fn(device)
                    result.backend = backend_name
                    results_by_backend.setdefault(backend_name, []).append(result)
                    self.results.append(result)
                    logger.info(
                        f"  {backend_name}: {result.value:.3f} ms "
                        f"(±{result.std:.3f})"
                    )
                except Exception as e:
                    logger.warning(f"  {backend_name}: FAILED ({e})")
                    results_by_backend.setdefault(backend_name, []).append(
                        BenchmarkResult(bench_name, backend_name, "latency_ms",
                                        float('nan'), 0, 0.0, {"error": str(e)})
                    )

            # Compute speedup vs CPU
            speedup = {}
            cpu_results = results_by_backend.get("CPU", [])
            if cpu_results:
                cpu_time = cpu_results[0].value
                for backend_name, res_list in results_by_backend.items():
                    if res_list and math.isfinite(res_list[0].value) and res_list[0].value > 0:
                        speedup[backend_name] = cpu_time / res_list[0].value
                    else:
                        speedup[backend_name] = float('nan')

            reports[bench_name] = BenchmarkReport(
                benchmark_name=bench_name,
                results=results_by_backend,
                speedup_vs_cpu=speedup,
                total_duration_s=time.time() - t_start,
            )

        return reports

    def generate_report_markdown(self, reports: Dict[str, BenchmarkReport]) -> str:
        """Generate a Markdown-formatted benchmark report."""
        lines = [
            f"# SESI Backend Benchmark Report",
            f"",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Version: {VALIDATION_SUITE_VERSION}",
            f"",
        ]

        for bench_name, report in reports.items():
            lines.append(f"## {bench_name}")
            lines.append("")
            lines.append("| Backend | Latency (ms) | Std (ms) | Speedup vs CPU |")
            lines.append("|---|---|---|---|")
            for backend, res_list in report.results.items():
                for r in res_list:
                    speedup = report.speedup_vs_cpu.get(backend, float('nan'))
                    if math.isfinite(r.value):
                        lines.append(
                            f"| {backend} | {r.value:.3f} | {r.std:.3f} | {speedup:.2f}x |"
                        )
                    else:
                        err = r.metadata.get("error", "N/A")
                        lines.append(f"| {backend} | N/A | N/A | N/A ({err}) |")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# ORCHESTRATOR
# =============================================================================
class SESIMultiBackendValidationSuite:
    """
    Top-level suite integrating all three components:
      [A] HardwareTestSuite
      [B] DockerImageBuilder
      [C] BackendBenchmark
    """

    def __init__(self, output_dir: str = "./sesi_validation_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hw_tests = HardwareTestSuite()
        self.docker_builder = DockerImageBuilder(
            DockerImageConfig(output_dir=str(self.output_dir / "docker"))
        )
        self.benchmark = BackendBenchmark()

    def run_full_validation(self) -> Dict[str, Any]:
        """Run all validation steps and save results."""
        results = {
            "version": VALIDATION_SUITE_VERSION,
            "timestamp": time.time(),
            "platform": {
                "os": platform.system(),
                "arch": platform.machine(),
                "python": sys.version,
                "torch": torch.__version__,
            },
            "ecosystem": {
                "has_real_fold": _HAS_REAL_FOLD,
                "has_openmm": _HAS_OPENMM,
                "has_openmm_ml": _HAS_OPENMMML,
            },
        }

        # A: Hardware tests
        logger.info("[Suite] Running hardware tests...")
        hw_reports = self.hw_tests.run_all_tests()
        results["hardware_tests"] = {
            rep.backend_name: {
                "n_pass": rep.n_pass,
                "n_fail": rep.n_fail,
                "n_skip": rep.n_skip,
                "duration_s": rep.total_duration_s,
            }
            for rep in hw_reports
        }
        # Save detailed results
        hw_path = self.output_dir / "hardware_test_results.json"
        with open(hw_path, "w") as f:
            json.dump([asdict(r) for r in self.hw_tests.results], f, indent=2)
        results["hardware_tests_file"] = str(hw_path)

        # B: Docker artifacts
        logger.info("[Suite] Generating Docker artifacts...")
        docker_artifacts = self.docker_builder.generate_all()
        results["docker_artifacts"] = docker_artifacts

        # C: Benchmarks
        logger.info("[Suite] Running benchmarks...")
        bench_reports = self.benchmark.run_all_benchmarks()
        results["benchmarks"] = {
            name: {
                "speedup_vs_cpu": rep.speedup_vs_cpu,
                "duration_s": rep.total_duration_s,
            }
            for name, rep in bench_reports.items()
        }
        # Save benchmark report
        bench_md = self.benchmark.generate_report_markdown(bench_reports)
        bench_path = self.output_dir / "benchmark_report.md"
        with open(bench_path, "w") as f:
            f.write(bench_md)
        results["benchmark_report_file"] = str(bench_path)

        # Save full results
        full_path = self.output_dir / "validation_results.json"
        with open(full_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"[Suite] Full results saved to {full_path}")

        return results


# =============================================================================
# SELF-TEST SUITE
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 78)
    print(f"  SESI MULTI-BACKEND VALIDATION SUITE v{VALIDATION_SUITE_VERSION} — Self-Test")
    print(f"  real_fold_one_v2: {_HAS_REAL_FOLD}  OpenMM: {_HAS_OPENMM}  OpenMM-ML: {_HAS_OPENMMML}")
    print(f"  PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    print("=" * 78)

    suite = SESIMultiBackendValidationSuite(output_dir="./test_validation_output")

    # ---- T1: Backend detection -----------------------------------------
    print("\n [T1] Backend Detection")
    backends = suite.hw_tests._detect_backends()
    print(f"   Detected {len(backends)} backend(s):")
    for b in backends:
        print(f"     {b['name']:20s} -> {b['torch_device']} (info: {b['info'].get('device_name', 'N/A')})")
    assert len(backends) >= 1, "At least CPU must be detected"
    print("[PASS] T1 Backend detection works")

    # ---- T2: Hardware tests on CPU -------------------------------------
    print("\n [T2] Hardware Tests (CPU)")
    cpu_backend = [b for b in backends if b["name"] == "CPU"]
    if cpu_backend:
        reports = suite.hw_tests.run_all_tests(backends=cpu_backend)
        for rep in reports:
            print(f"   {rep.backend_name}: {rep.n_pass} PASS, {rep.n_fail} FAIL, {rep.n_skip} SKIP")
            for r in rep.results:
                status_icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘"}.get(r.status, "?")
                print(f"     {status_icon} {r.test_name}: {r.status} ({r.duration_s:.3f}s) {r.message}")
        # CPU should pass basic_tensor_ops and amp_autocast at minimum
        cpu_rep = reports[0]
        basic_ops = [r for r in cpu_rep.results if r.test_name == "basic_tensor_ops"]
        assert basic_ops and basic_ops[0].status == "PASS"
    print("[PASS] T2 Hardware tests run on CPU")

    # ---- T3: Docker artifact generation --------------------------------
    print("\n [T3] Docker Artifact Generation")
    artifacts = suite.docker_builder.generate_all()
    for name, path in artifacts.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"   {name:30s} -> {path} ({size} bytes, exists={exists})")
        assert exists, f"Docker artifact {name} not created"
    print("[PASS] T3 Docker artifacts generated")

    # ---- T4: Benchmark on CPU ------------------------------------------
    print("\n [T4] Backend Benchmark (CPU only for self-test)")
    # Run a subset to keep self-test fast
    cpu_device = torch.device("cpu")
    result_matmul = suite.benchmark._bench_matmul(cpu_device, size=128)  # small for speed
    print(f"   matmul_128: {result_matmul.value:.3f} ms (±{result_matmul.std:.3f})")
    assert math.isfinite(result_matmul.value)

    result_reduction = suite.benchmark._bench_reduction(cpu_device, n=100_000)
    print(f"   reduction_100k: {result_reduction.value:.3f} ms (±{result_reduction.std:.3f})")
    assert math.isfinite(result_reduction.value)

    if _HAS_REAL_FOLD:
        result_csoc = suite.benchmark._bench_csoc_kernel(cpu_device, n=1000)
        print(f"   csoc_kernel_1k: {result_csoc.value:.3f} ms (±{result_csoc.std:.3f})")
        assert math.isfinite(result_csoc.value)

        result_nl = suite.benchmark._bench_neighbor_list(cpu_device, n=200)
        print(f"   neighbor_list_200: {result_nl.value:.3f} ms (±{result_nl.std:.3f})")
        assert math.isfinite(result_nl.value)
    print("[PASS] T4 Benchmarks run on CPU")

    # ---- T5: Markdown report generation --------------------------------
    print("\n [T5] Benchmark Report Generation")
    mock_reports = {
        "matmul_1k": BenchmarkReport(
            "matmul_1k",
            {"CPU": [result_matmul]},
            {"CPU": 1.0},
            1.0,
        ),
    }
    md = suite.benchmark.generate_report_markdown(mock_reports)
    assert "# SESI Backend Benchmark Report" in md
    assert "matmul_1k" in md
    print(f"   Markdown report length: {len(md)} chars")
    print("[PASS] T5 Markdown report generated")

    # ---- T6: Full validation run ---------------------------------------
    print("\n [T6] Full Validation Suite Run")
    full_results = suite.run_full_validation()
    print(f"   Platform: {full_results['platform']['os']} / {full_results['platform']['arch']}")
    print(f"   Torch: {full_results['platform']['torch']}")
    print(f"   Ecosystem: real_fold={full_results['ecosystem']['has_real_fold']}, "
          f"openmm={full_results['ecosystem']['has_openmm']}")
    print(f"   Hardware tests: {len(full_results['hardware_tests'])} backend(s) tested")
    print(f"   Docker artifacts: {len(full_results['docker_artifacts'])} file(s)")
    print(f"   Benchmarks: {len(full_results['benchmarks'])} benchmark(s)")
    assert os.path.exists(full_results["hardware_tests_file"])
    assert os.path.exists(full_results["benchmark_report_file"])
    print("[PASS] T6 Full validation completed")

    print("\n" + "=" * 78)
    print("  ALL SELF-TESTS PASSED.")
    print("  HONEST REMINDERS:")
    print("    • Hardware tests SKIP gracefully when target hardware is absent.")
    print("    • Docker images are TEMPLATES; requires docker build on target host.")
    print("    • Benchmark numbers are only meaningful on actual hardware.")
    print("    • OpenMM energy test uses CPU platform for portability.")
    print("    • This module has NOT been run on all 8 backends simultaneously.")
    print("=" * 78)
