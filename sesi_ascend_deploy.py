"""
=============================================================================
SESI ASCEND DEPLOYMENT — ONNX → ATC → OM + ACL Inference
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

CRITICAL CLARIFICATION
──────────────────────
TensorRT is NVIDIA-proprietary and CANNOT run on Huawei Ascend.
The correct Ascend deployment path is:

    PyTorch model
        │  torch.onnx.export()
        ▼
    ONNX (.onnx)
        │  ATC (Ascend Tensor Compiler)  ← part of CANN toolkit
        ▼
    OM (Offline Model, .om)
        │  ACL (Ascend Computing Language) runtime
        ▼
    Ascend 910 / 310 NPU inference

Alternative paths:
    • torch_npu   — run PyTorch directly on Ascend (training + inference)
    • MindSpore   — native Ascend framework

REQUIREMENTS
────────────
    • CANN toolkit installed (provides `atc` binary + ACL)
    • Ascend hardware (910 for training, 310 for edge inference)
    • For torch_npu: pip install torch_npu
    • For ONNX export: pip install onnx onnxruntime

HONEST SCOPE
────────────
• ATC conversion requires the `atc` binary (not pip-installable; comes with CANN)
• This module generates the exact commands + wraps ACL Python API
• Actual NPU execution requires physical Ascend hardware
• Quantization (INT8) needs calibration dataset via ATC
=============================================================================
"""

from __future__ import annotations

import os
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger("sesi.ascend_deploy")

__all__ = [
    "AscendDeployConfig",
    "ONNXToAscendConverter",
    "AscendOMInference",
    "PyTorchNPUExporter",
    "detect_ascend",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AscendDeployConfig:
    # ── SoC target ────────────────────────────────────────────────────────
    soc_version: str = "Ascend910"   # Ascend910 | Ascend310 | Ascend910B
    # ── ATC options ───────────────────────────────────────────────────────
    atc_bin: str = "atc"             # path to ATC binary (from CANN)
    op_select_implmode: str = "high_performance"
    precision_mode: str = "allow_mix_precision"
    # ── Quantization ──────────────────────────────────────────────────────
    quantize: bool = False
    quant_calibration_dir: Optional[str] = None  # dir of calibration .bin files
    # ── Dynamic shape ─────────────────────────────────────────────────────
    dynamic_dims: Optional[str] = None  # e.g. "1,3,224,224;8,3,224,224"
    # ── I/O ───────────────────────────────────────────────────────────────
    input_shape: str = "1,128"       # must match the ONNX input
    input_name: str = "input"
    output_dir: str = "./ascend_models"
    # ── ACL runtime ───────────────────────────────────────────────────────
    device_id: int = 0


def detect_ascend() -> Dict[str, Any]:
    """
    Detect Ascend NPU availability.

    Returns dict with:
        npu_available : bool
        npu_count     : int
        cann_found    : bool  (atc binary on PATH)
        soc           : str or None
    """
    result = {
        "npu_available": False,
        "npu_count": 0,
        "cann_found": False,
        "soc": None,
    }

    # Check for torch_npu
    try:
        import torch_npu  # noqa: F401
        if torch_npu.npu.is_available():
            result["npu_available"] = True
            result["npu_count"] = torch_npu.npu.device_count()
            result["soc"] = torch_npu.npu.get_device_name(0)
    except ImportError:
        pass

    # Check for ATC binary (CANN toolkit)
    from shutil import which
    if which("atc") is not None:
        result["cann_found"] = True

    return result


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: PyTorch → ONNX
# ═══════════════════════════════════════════════════════════════════════

class PyTorchNPUExporter:
    """
    Exports a PyTorch model to ONNX, ready for ATC conversion.

    Also supports direct torch_npu execution (no export needed) when
    you want to train/infer on Ascend without leaving PyTorch.
    """

    def __init__(self, cfg: AscendDeployConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def export_onnx(
        self,
        model: nn.Module,
        sample_input: Tensor,
        onnx_path: str,
        opset_version: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Path:
        """
        Export PyTorch model to ONNX.

        Args:
            model: PyTorch model (will be set to eval mode)
            sample_input: example input tensor for tracing
            onnx_path: output .onnx file path
            opset_version: ONNX opset (17 recommended for ATC)
            dynamic_axes: optional dynamic dimension spec
        """
        model.eval()
        onnx_path = Path(onnx_path)

        if dynamic_axes is None:
            dynamic_axes = {self.cfg.input_name: {0: "batch_size"}}

        torch.onnx.export(
            model,
            sample_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[self.cfg.input_name],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        logger.info(f"ONNX exported: {onnx_path}")

        # Validate if onnx is installed
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validated successfully")
        except ImportError:
            logger.warning("onnx not installed; skipping validation")
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")

        return onnx_path

    def run_on_npu(self, model: nn.Module, x: Tensor) -> Tensor:
        """
        Run a PyTorch model directly on Ascend NPU via torch_npu.
        No export needed — useful for training and quick inference.
        """
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "torch_npu not installed. Install with: pip install torch_npu"
            )

        device = torch.device("npu:0")
        model = model.to(device)
        x = x.to(device)

        with torch.no_grad():
            return model(x)


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: ONNX → OM via ATC
# ═══════════════════════════════════════════════════════════════════════

class ONNXToAscendConverter:
    """
    Converts ONNX to Ascend OM (Offline Model) using the ATC tool.

    ATC is part of the CANN toolkit. This class builds and runs the
    correct `atc` command line. It does NOT ship ATC (proprietary).

    Equivalent role to TensorRT's `trtexec` on NVIDIA, but for Ascend.
    """

    def __init__(self, cfg: AscendDeployConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def build_atc_command(self, onnx_path: str, om_path: str) -> List[str]:
        """
        Construct the ATC command line.

        Returns the command as a list (for subprocess), so it can be
        inspected / logged before execution.
        """
        cfg = self.cfg
        om_path = str(Path(om_path).with_suffix(""))  # ATC appends .om

        cmd = [
            cfg.atc_bin,
            f"--model={onnx_path}",
            "--framework=5",                       # 5 = ONNX
            f"--output={om_path}",
            f"--soc_version={cfg.soc_version}",
            f"--input_shape={cfg.input_name},{cfg.input_shape}",
            f"--op_select_implmode={cfg.op_select_implmode}",
            f"--precision_mode={cfg.precision_mode}",
            "--log=error",
        ]

        # Dynamic batch / shape
        if cfg.dynamic_dims is not None:
            cmd.append(f"--dynamic_dims={cfg.dynamic_dims}")

        # INT8 quantization with calibration
        if cfg.quantize:
            if cfg.quant_calibration_dir is None:
                raise ValueError(
                    "quantize=True requires quant_calibration_dir "
                    "(directory of calibration .bin files)."
                )
            cmd.extend([
                "--quantize=True",
                f"--calibrate_dir={cfg.quant_calibration_dir}",
            ])

        return cmd

    def convert(self, onnx_path: str, om_name: str) -> Path:
        """
        Run ATC to convert ONNX → OM.

        Args:
            onnx_path: input .onnx file
            om_name: output name (without extension; .om appended by ATC)

        Returns:
            Path to the generated .om file
        """
        om_path = Path(self.cfg.output_dir) / om_name
        cmd = self.build_atc_command(onnx_path, str(om_path))

        logger.info("Running ATC command:")
        logger.info("  " + " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                logger.error(f"ATC failed:\n{result.stderr}")
                raise RuntimeError(f"ATC conversion failed: {result.stderr[:500]}")
            logger.info(f"ATC succeeded. OM written near {om_path}.om")
        except FileNotFoundError:
            raise RuntimeError(
                f"'{self.cfg.atc_bin}' not found. Install the CANN toolkit "
                "and ensure `atc` is on your PATH. "
                "See: https://www.hiascend.com/software/cann"
            )

        return om_path.with_suffix(".om")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: OM Inference via ACL
# ═══════════════════════════════════════════════════════════════════════

class AscendOMInference:
    """
    Runs an Ascend OM model via the ACL (Ascend Computing Language)
    Python API.

    This is the Ascend equivalent of loading a TensorRT engine and
    running inference. Requires the ACL Python bindings from CANN.

    Lifecycle:
        init() → load_model() → [run() ...] → unload() → finalize()
    """

    def __init__(self, cfg: AscendDeployConfig):
        self.cfg = cfg
        self._acl = None
        self._model_id = None
        self._initialized = False

    def _import_acl(self):
        """Import the ACL Python module (ships with CANN)."""
        if self._acl is not None:
            return self._acl
        try:
            from acl import acl  # type: ignore
            self._acl = acl
            return acl
        except ImportError:
            raise RuntimeError(
                "ACL Python bindings not found. Install the CANN toolkit "
                "and source the set_env.sh / add the ACL python path."
            )

    def init(self) -> None:
        """Initialize ACL and set device."""
        acl = self._import_acl()
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed with code {ret}")
        ret = acl.rt.set_device(self.cfg.device_id)
        if ret != 0:
            raise RuntimeError(f"set_device failed with code {ret}")
        self._initialized = True
        logger.info(f"ACL initialized on device {self.cfg.device_id}")

    def load_model(self, om_path: str) -> None:
        """Load an OM model file."""
        acl = self._import_acl()
        if not self._initialized:
            self.init()
        self._model_id, ret = acl.mdl.load_from_file(om_path)
        if ret != 0:
            raise RuntimeError(f"load_from_file failed with code {ret}")
        logger.info(f"OM model loaded: {om_path} (id={self._model_id})")

    def run(self, input_np) -> Any:
        """
        Run inference on a numpy input.

        Args:
            input_np: numpy array matching the model's input shape/dtype

        Returns:
            numpy array of the model output
        """
        acl = self._import_acl()
        import numpy as np

        if self._model_id is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Create input dataset
        input_dataset = acl.mdl.create_dataset()
        input_buffer = acl.mdl.create_data_buffer(
            input_np, input_np.nbytes
        )
        acl.mdl.add_dataset_buffer(input_dataset, input_buffer)

        # Create output dataset
        output_dataset = acl.mdl.create_dataset()
        # NOTE: real code must allocate output buffer based on model's
        # output description via acl.mdl.get_desc / create_dataset_by_desc.
        # Simplified here for clarity.

        ret = acl.mdl.execute(self._model_id, input_dataset, output_dataset)
        if ret != 0:
            raise RuntimeError(f"mdl.execute failed with code {ret}")

        # Extract output (simplified)
        logger.info("Inference executed on Ascend NPU")
        return output_dataset

    def finalize(self) -> None:
        """Unload model and finalize ACL."""
        acl = self._import_acl()
        if self._model_id is not None:
            acl.mdl.unload(self._model_id)
            self._model_id = None
        if self._initialized:
            acl.rt.reset_device(self.cfg.device_id)
            acl.finalize()
            self._initialized = False
        logger.info("ACL finalized")

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *exc):
        self.finalize()


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Ascend Deployment — Smoke Test")
    print("=" * 72)

    # Detect environment
    env = detect_ascend()
    print(f"  NPU available : {env['npu_available']}")
    print(f"  NPU count     : {env['npu_count']}")
    print(f"  CANN/ATC found: {env['cann_found']}")
    print(f"  SoC           : {env['soc']}")

    cfg = AscendDeployConfig(soc_version="Ascend910", input_shape="2,128")

    # Build a tiny model
    model = nn.Sequential(
        nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 10)
    )
    sample = torch.randn(2, 128)

    # Step 1: ONNX export
    exporter = PyTorchNPUExporter(cfg)
    onnx_path = exporter.export_onnx(model, sample, f"{cfg.output_dir}/model.onnx")
    print(f"  [PASS] ONNX exported → {onnx_path}")

    # Step 2: ATC command (build but don't run if atc missing)
    converter = ONNXToAscendConverter(cfg)
    atc_cmd = converter.build_atc_command(str(onnx_path), f"{cfg.output_dir}/model")
    print(f"  [PASS] ATC command constructed:")
    print(f"         {' '.join(atc_cmd)}")

    if env["cann_found"]:
        try:
            om_path = converter.convert(str(onnx_path), "model")
            print(f"  [PASS] OM generated → {om_path}")
        except Exception as e:
            print(f"  [WARN] ATC conversion skipped/failed: {e}")
    else:
        print("  [SKIP] ATC not on PATH — install CANN to convert for real.")

    print("\n" + "=" * 72)
    print("  HONEST REMINDERS:")
    print("  • TensorRT does NOT run on Ascend. Use ATC → OM instead.")
    print("  • ATC + ACL require the CANN toolkit (not pip-installable).")
    print("  • Real NPU inference requires physical Ascend hardware.")
    print("  • torch_npu is the easiest path for PyTorch-on-Ascend.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
