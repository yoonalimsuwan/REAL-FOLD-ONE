"""
=============================================================================
SESI APPLE SILICON DEPLOYMENT + SURROGATE-ORCHESTRATOR BRIDGE v1.0
CoreML Export for Apple Silicon × OpenFOAM Surrogate ↔ SESICarpetGlobalOrchestrator
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026
Version      : 1.0.0

PART A — CoreML Export for Apple Silicon
────────────────────────────────────────
The FNO surrogate uses torch.fft.rfft2/irfftn. coremltools has LIMITED
FFT support, so full-FNO conversion may fail or fall back to CPU.

Solution: provide a CoreML-compatible variant that replaces the spectral
layer with a learnable depthwise-separable convolution approximation.
This is an APPROXIMATION for edge deployment, not the exact FNO.

Deployment targets (MLComputeUnits):
  • ALL            → Neural Engine + GPU + CPU (fastest, op-dependent)
  • CPU_AND_GPU    → skip Neural Engine
  • CPU_ONLY       → reference / debugging

PART B — Surrogate ↔ Orchestrator Bridge
────────────────────────────────────────
Connects DifferentiableCFDSurrogate to SESICarpetGlobalOrchestrator's
CFD domain, replacing the lumped BroadCFDModule with a full-field
differentiable CFD engine while preserving the orchestrator's output
contract (pressure / velocity / turbulence keys).

HONEST SCOPE
────────────
• CoreML conversion requires `coremltools` (pip install coremltools).
• Not every op maps to the Neural Engine; verify placement with
  `mlmodel.get_spec()` or Xcode's CoreML instrument.
• The CoreML-safe surrogate is a conv APPROXIMATION of the spectral
  operator — validate accuracy against the FNO before trusting it.
• The bridge is fully differentiable; gradients flow orchestrator →
  surrogate → physics loss, enabling end-to-end design optimization.
• Real Apple Silicon inference requires macOS + CoreML runtime.

=============================================================================
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger("sesi.apple_bridge")

# ── Optional imports ──────────────────────────────────────────────────────
try:
    import coremltools as ct
    HAS_COREMLTOOLS = True
except ImportError:
    HAS_COREMLTOOLS = False
    ct = None

# Try to import the real surrogate; fall back to a local minimal version
try:
    from sesi_openfoam_surrogate import (
        DifferentiableCFDSurrogate,
        OpenFOAMSurrogateConfig,
        SpectralConv2d,
    )
    HAS_SURROGATE = True
except ImportError:
    HAS_SURROGATE = False

__all__ = [
    "CoreMLExportConfig",
    "CoreMLSafeSpectralConv2d",
    "CoreMLCompatibleCFDSurrogate",
    "CoreMLExporter",
    "SurrogateOrchestratorBridge",
    "CFDOrchestratorAdapter",
    "attach_surrogate_to_orchestrator",
]


# ═══════════════════════════════════════════════════════════════════════
# Local fallback definitions (used only if sesi_openfoam_surrogate absent)
# ═══════════════════════════════════════════════════════════════════════

if not HAS_SURROGATE:
    @dataclass
    class OpenFOAMSurrogateConfig:
        grid_h: int = 32
        grid_w: int = 32
        in_channels: int = 3
        out_channels: int = 3
        fourier_modes: int = 8
        fno_width: int = 16
        n_fno_layers: int = 3

    class SpectralConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, modes1, modes2):
            super().__init__()
            self.in_ch, self.out_ch = in_ch, out_ch
            self.modes1, self.modes2 = modes1, modes2
            self.scale = 1.0 / (in_ch * out_ch)
            self.w1 = nn.Parameter(self.scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
            self.w2 = nn.Parameter(self.scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

        def forward(self, x):
            B, C, H, W = x.shape
            x_ft = torch.fft.rfft2(x)
            out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1 = min(self.modes1, H); m2 = min(self.modes2, W // 2 + 1)
            out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
            return torch.fft.irfft2(out_ft, s=(H, W))

    class DifferentiableCFDSurrogate(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.lift = nn.Linear(cfg.in_channels, cfg.fno_width)
            self.specs = nn.ModuleList([
                SpectralConv2d(cfg.fno_width, cfg.fno_width, cfg.fourier_modes, cfg.fourier_modes)
                for _ in range(cfg.n_fno_layers)])
            self.pws = nn.ModuleList([nn.Conv2d(cfg.fno_width, cfg.fno_width, 1) for _ in range(cfg.n_fno_layers)])
            self.proj = nn.Sequential(nn.Conv2d(cfg.fno_width, cfg.fno_width, 1), nn.GELU(),
                                      nn.Conv2d(cfg.fno_width, cfg.out_channels, 1))

        def forward(self, x):
            x = x.permute(0, 2, 3, 1)
            x = self.lift(x)
            x = x.permute(0, 3, 1, 2)
            for spec, pw in zip(self.specs, self.pws):
                x = F.gelu(spec(x) + pw(x))
            return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════
# PART A: CoreML EXPORT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoreMLExportConfig:
    output_dir: str = "./coreml_models"
    model_name: str = "sesi_cfd_surrogate"
    opset: int = 17
    # Input spec
    grid_h: int = 32
    grid_w: int = 32
    in_channels: int = 3
    # Dynamic batch range
    dynamic_batch: Tuple[int, int] = (1, 8)
    # Quantization
    quantize: bool = False              # int8 weight-only quantization
    quantize_bits: int = 8
    # Compute units hint stored in metadata (applied at load time on device)
    compute_units: str = "all"          # all | cpu_and_gpu | cpu_only


class CoreMLSafeSpectralConv2d(nn.Module):
    """
    CoreML-friendly replacement for SpectralConv2d.

    The true FNO spectral layer uses rfft2/irfftn + complex einsum, which
    coremltools does not reliably map to the Neural Engine. This module
    approximates the global spectral mixing with a stack of:
       • depthwise conv (local mixing)
       • 1x1 conv (channel mixing)
       • large-kernel dilated conv (wider receptive field ≈ low-freq modes)

    HONEST: this is an approximation. Validate against the FNO before use.
    """

    def __init__(self, channels: int, modes: int = 8):
        super().__init__()
        # depthwise local mixing
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        # wide receptive field to mimic low-frequency spectral modes
        self.dilated = nn.Conv2d(channels, channels, kernel_size=3,
                                 padding=modes // 2, dilation=max(1, modes // 4), groups=channels)
        # channel mixing
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        local = self.dw(x)
        wide = self.dilated(x)
        return self.pw(F.gelu(local + wide))


class CoreMLCompatibleCFDSurrogate(nn.Module):
    """
    FNO-structured surrogate with CoreML-safe spectral layers (no FFT).

    Same lift → N×(spectral + pointwise) → project skeleton as
    DifferentiableCFDSurrogate, but every op is CoreML/ANE-friendly.
    """

    def __init__(self, cfg: OpenFOAMSurrogateConfig):
        super().__init__()
        self.cfg = cfg
        self.lift = nn.Conv2d(cfg.in_channels, cfg.fno_width, kernel_size=1)
        self.specs = nn.ModuleList([
            CoreMLSafeSpectralConv2d(cfg.fno_width, cfg.fourier_modes)
            for _ in range(cfg.n_fno_layers)
        ])
        self.pws = nn.ModuleList([
            nn.Conv2d(cfg.fno_width, cfg.fno_width, kernel_size=1)
            for _ in range(cfg.n_fno_layers)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, cfg.fno_width) for _ in range(cfg.n_fno_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(cfg.fno_width, cfg.fno_width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(cfg.fno_width, cfg.out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, in_channels, H, W) → (B, out_channels, H, W)."""
        h = self.lift(x)
        for spec, pw, norm in zip(self.specs, self.pws, self.norms):
            h = norm(spec(h) + pw(h))
            h = F.gelu(h)
        return self.proj(h)


class CoreMLExporter:
    """
    Exports a CFD surrogate to CoreML for Apple Silicon.

    Strategy:
      1. Prefer the CoreML-compatible (conv-only) surrogate for reliable
         Neural Engine placement.
      2. Trace with torch.jit.trace using a representative example input.
      3. Convert via coremltools with dynamic batch range.
      4. Optionally apply int8 weight-only quantization.
      5. Save as .mlpackage and record compute-unit metadata.

    HONEST: if you pass the FFT-based DifferentiableCFDSurrogate, conversion
    may fail or produce a CPU-fallback model. Use CoreMLCompatibleCFDSurrogate
    for production edge deployment.
    """

    def __init__(self, cfg: CoreMLExportConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _example_input(self, device: torch.device) -> Tensor:
        return torch.randn(1, self.cfg.in_channels,
                           self.cfg.grid_h, self.cfg.grid_w, device=device)

    def export(self, model: nn.Module) -> Dict[str, Any]:
        """
        Convert model to CoreML. Returns dict with path + honest report.
        """
        result = {
            "success": False,
            "path": None,
            "used_coreml_safe_variant": isinstance(model, CoreMLCompatibleCFDSurrogate),
            "notes": [],
        }

        if not HAS_COREMLTOOLS:
            result["notes"].append(
                "coremltools not installed. Install with: pip install coremltools. "
                "Returning dry-run report only."
            )
            logger.warning(result["notes"][-1])
            return result

        model.eval()
        device = torch.device("cpu")  # trace on CPU for CoreML
        model = model.to(device)
        example = self._example_input(device)

        # 1. Trace
        try:
            traced = torch.jit.trace(model, example, strict=False)
        except Exception as e:
            result["notes"].append(f"torch.jit.trace failed: {e}")
            logger.error(result["notes"][-1])
            return result

        # 2. Build input spec with dynamic batch
        lo, hi = self.cfg.dynamic_batch
        input_shape = ct.Shape(
            shape=(ct.RangeDim(lo, hi), self.cfg.in_channels,
                   self.cfg.grid_h, self.cfg.grid_w)
        )
        mlinput = ct.TensorType(name="input", shape=input_shape)

        # 3. Convert
        try:
            mlmodel = ct.convert(
                traced,
                inputs=[mlinput],
                minimum_deployment_target=ct.target.iOS16,
                convert_to="mlprogram",   # MLX-backed, best for Apple Silicon
            )
        except Exception as e:
            result["notes"].append(
                f"coremltools conversion failed: {e}. If the model uses FFT, "
                "switch to CoreMLCompatibleCFDSurrogate."
            )
            logger.error(result["notes"][-1])
            return result

        # 4. Optional quantization
        if self.cfg.quantize:
            try:
                op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype="int8",
                    granularity="per_tensor",
                )
                config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
                mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
                result["notes"].append("Applied int8 weight-only quantization.")
            except Exception as e:
                result["notes"].append(f"Quantization skipped: {e}")

        # 5. Attach metadata
        mlmodel.author = "PAI & Yoon A Limsuwan / MSPS NETWORK"
        mlmodel.short_description = "SESI differentiable CFD surrogate (CoreML-safe variant)"
        mlmodel.user_defined_metadata["compute_units_hint"] = self.cfg.compute_units
        mlmodel.user_defined_metadata["quantized"] = str(self.cfg.quantize)

        # 6. Save
        out_path = Path(self.cfg.output_dir) / f"{self.cfg.model_name}.mlpackage"
        mlmodel.save(str(out_path))
        result["success"] = True
        result["path"] = str(out_path)
        result["notes"].append(
            f"Saved {out_path}. Load on device with compute_units="
            f"ct.ComputeUnit.{self.cfg.compute_units.upper()} for Neural Engine."
        )
        logger.info("CoreML export succeeded → %s", out_path)
        return result

    @staticmethod
    def load_for_inference(mlpackage_path: str, compute_units: str = "all"):
        """
        Load a .mlpackage on macOS with the requested compute units.
        compute_units: 'all' | 'cpu_and_gpu' | 'cpu_only'
        """
        if not HAS_COREMLTOOLS:
            raise RuntimeError("coremltools required to load CoreML models.")
        cu_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }
        return ct.models.MLModel(mlpackage_path, compute_units=cu_map[compute_units])


# ═══════════════════════════════════════════════════════════════════════
# PART B: SURROGATE ↔ ORCHESTRATOR BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class SurrogateOrchestratorBridge(nn.Module):
    """
    Differentiable bridge between an orchestrator latent vector and the
    full-field CFD surrogate.

    latent (B, latent_dim)
        → latent_to_field → (B, in_channels, H, W)
        → surrogate        → (B, out_channels, H, W)  [Ux, Uy, p]

    Everything is autograd-native, so gradients flow from any downstream
    objective (drag, mixing, pressure drop) back to the latent/design.
    """

    def __init__(self, cfg: OpenFOAMSurrogateConfig, latent_dim: int,
                 coarse: int = 8):
        super().__init__()
        self.cfg = cfg
        self.coarse = coarse
        # Project latent to a small coarse grid, then upsample differentiably
        self.coarse_grid = nn.Sequential(
            nn.Linear(latent_dim, cfg.in_channels * coarse * coarse),
            nn.GELU(),
        )
        self.upsample = nn.Upsample(size=(cfg.grid_h, cfg.grid_w),
                                    mode="bilinear", align_corners=False)
        # Build surrogate (use CoreML-safe if requested externally)
        self.surrogate = DifferentiableCFDSurrogate(cfg)

    def forward(self, latent: Tensor) -> Tensor:
        """
        Args:
            latent: (B, latent_dim)
        Returns:
            flow field: (B, out_channels, H, W)
        """
        B = latent.shape[0]
        coarse = self.coarse_grid(latent)                      # (B, C*h*w)
        coarse = coarse.view(B, self.cfg.in_channels, self.coarse, self.coarse)
        field_in = self.upsample(coarse)                       # (B, C, H, W)
        return self.surrogate(field_in)                        # (B, out_C, H, W)


class CFDOrchestratorAdapter(nn.Module):
    """
    Drop-in replacement for BroadCFDModule inside SESICarpetGlobalOrchestrator.

    Accepts the orchestrator's encoded feature tensor (B, feature_dim) and
    returns a dict matching BroadCFDModule's output contract:
        pressure            (B,)
        velocity            (B, 3)
        turbulence_intensity(B,)
    plus the full predicted flow field and physics diagnostics.
    """

    def __init__(self, feature_dim: int, surrogate_cfg: OpenFOAMSurrogateConfig):
        super().__init__()
        self.bridge = SurrogateOrchestratorBridge(surrogate_cfg, latent_dim=feature_dim)
        self.cfg = surrogate_cfg

        # Small heads to produce the lumped quantities the orchestrator expects
        self._pool_pressure = nn.AdaptiveAvgPool2d(1)

    def forward(self, flow_field_input: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            flow_field_input: (B, feature_dim) encoded orchestrator features
        Returns:
            dict with pressure, velocity, turbulence_intensity, full_field
        """
        field = self.bridge(flow_field_input)      # (B, 3, H, W) → [Ux, Uy, p]
        ux = field[:, 0]
        uy = field[:, 1]
        p = field[:, 2]

        # Lumped outputs (match BroadCFDModule contract)
        pressure = p.mean(dim=(1, 2))                          # (B,)
        velocity = torch.stack([ux.mean(dim=(1, 2)),
                                uy.mean(dim=(1, 2)),
                                torch.zeros_like(ux.mean(dim=(1, 2)))], dim=-1)  # (B,3)
        speed_mag = torch.sqrt(ux.pow(2) + uy.pow(2))
        turbulence_intensity = speed_mag.std(dim=(1, 2)) / (speed_mag.mean(dim=(1, 2)) + 1e-6)

        return {
            "pressure": pressure,
            "velocity": velocity,
            "turbulence_intensity": turbulence_intensity,
            "full_field": field,
            # keep carpet-math hooks (filled by orchestrator if it wants them)
        }


def attach_surrogate_to_orchestrator(
    orchestrator: Any,
    feature_dim: int,
    surrogate_cfg: Optional[OpenFOAMSurrogateConfig] = None,
    use_coreml_safe: bool = False,
) -> CFDOrchestratorAdapter:
    """
    Replace the orchestrator's CFD domain module with the differentiable
    surrogate adapter, preserving gradient flow.

    Args:
        orchestrator: SESICarpetGlobalOrchestrator instance (has .cfd attribute)
        feature_dim: dimension of the orchestrator's encoded features
        surrogate_cfg: OpenFOAMSurrogateConfig (defaults created if None)
        use_coreml_safe: if True, swap the surrogate inside the bridge to the
                         CoreML-compatible conv variant.

    Returns:
        The installed CFDOrchestratorAdapter (already set as orchestrator.cfd).
    """
    if surrogate_cfg is None:
        surrogate_cfg = OpenFOAMSurrogateConfig()

    adapter = CFDOrchestratorAdapter(feature_dim, surrogate_cfg)

    if use_coreml_safe:
        # Swap in the conv-only surrogate for ANE-friendly deployment
        adapter.bridge.surrogate = CoreMLCompatibleCFDSurrogate(surrogate_cfg)
        logger.info("Installed CoreML-safe surrogate into orchestrator CFD adapter.")

    # Move adapter to the orchestrator's device if detectable
    try:
        orch_device = next(orchestrator.parameters()).device
        adapter = adapter.to(orch_device)
    except StopIteration:
        pass

    if hasattr(orchestrator, "cfd"):
        orchestrator.cfd = adapter
        logger.info("Orchestrator.cfd replaced with differentiable surrogate adapter.")
    else:
        logger.warning("Orchestrator has no .cfd attribute; returning adapter unattached.")

    return adapter


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Apple Silicon Deployment + Surrogate-Orchestrator Bridge")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  coremltools available: {HAS_COREMLTOOLS}")
    torch.manual_seed(42)

    cfg = OpenFOAMSurrogateConfig(
        grid_h=32, grid_w=32, in_channels=3, out_channels=3,
        fourier_modes=8, fno_width=16, n_fno_layers=3,
    )

    # ── Test 1: CoreML-safe surrogate forward/backward ──────────────────
    print("\n  ── Test 1: CoreML-safe surrogate ──")
    safe_model = CoreMLCompatibleCFDSurrogate(cfg).to(device)
    x = torch.randn(2, cfg.in_channels, cfg.grid_h, cfg.grid_w, device=device, requires_grad=True)
    y = safe_model(x)
    y.sum().backward()
    print(f"    Output shape: {tuple(y.shape)}")
    print(f"    Gradient flows: {x.grad is not None and torch.isfinite(x.grad).all()}")
    assert y.shape == (2, cfg.out_channels, cfg.grid_h, cfg.grid_w)

    # ── Test 2: CoreML export (dry-run if coremltools missing) ─────────
    print("\n  ── Test 2: CoreML export ──")
    export_cfg = CoreMLExportConfig(quantize=False)
    exporter = CoreMLExporter(export_cfg)
    report = exporter.export(safe_model)
    print(f"    Success: {report['success']}")
    print(f"    CoreML-safe variant used: {report['used_coreml_safe_variant']}")
    for note in report["notes"]:
        print(f"    • {note}")

    # ── Test 3: Surrogate ↔ Orchestrator bridge ────────────────────────
    print("\n  ── Test 3: Surrogate-Orchestrator bridge ──")
    feature_dim = 64
    adapter = CFDOrchestratorAdapter(feature_dim, cfg).to(device)
    latent = torch.randn(2, feature_dim, device=device, requires_grad=True)
    out = adapter(latent)
    loss = out["pressure"].mean() + out["turbulence_intensity"].mean()
    loss.backward()
    print(f"    pressure shape        : {tuple(out['pressure'].shape)}")
    print(f"    velocity shape        : {tuple(out['velocity'].shape)}")
    print(f"    turbulence shape      : {tuple(out['turbulence_intensity'].shape)}")
    print(f"    full_field shape      : {tuple(out['full_field'].shape)}")
    print(f"    Gradient to latent    : {latent.grad is not None and torch.isfinite(latent.grad).all()}")

    # ── Test 4: attach to a mock orchestrator ──────────────────────────
    print("\n  ── Test 4: Attach to orchestrator ──")
    class MockOrchestrator(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfd = nn.Identity()
    mock = MockOrchestrator()
    installed = attach_surrogate_to_orchestrator(mock, feature_dim, cfg, use_coreml_safe=True)
    test_out = mock.cfd(torch.randn(2, feature_dim, device=next(installed.parameters()).device))
    print(f"    orchestrator.cfd is now: {type(mock.cfd).__name__}")
    print(f"    CFD output keys        : {list(test_out.keys())}")

    print("\n" + "=" * 72)
    print("  SMOKE TEST COMPLETE")
    print("=" * 72)
    print("  HONEST REMINDERS:")
    print("  • FFT-based FNO does NOT reliably convert to CoreML/ANE.")
    print("  • Use CoreMLCompatibleCFDSurrogate (conv approximation) for edge.")
    print("  • Validate CoreML-safe accuracy against the FNO before trusting it.")
    print("  • Neural Engine placement is op-dependent — verify in Xcode.")
    print("  • Real Apple Silicon inference requires macOS + CoreML runtime.")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
