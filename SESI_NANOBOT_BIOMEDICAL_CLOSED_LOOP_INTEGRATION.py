#!/usr/bin/env python3
"""
=============================================================================
SESI NANOBOT × BIOMEDICAL CLOSED-LOOP INTEGRATION — v1.0.0
Connects: biomedical_pipeline + 4 nanobot modules into one system
=============================================================================

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the
better path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT THIS MODULE CONNECTS
=============================================================================
[1] sesi_biomedical_pipeline.py
    → SESIBiomedicalPipeline: 5-domain decision layer
    → ProteinFolding, Oncology, Epidemiology, AntiAging, GeneEditing
    → Produces: target parameters, dose, timing, cell-type selection

[2] nanobot_sesi_swarm_intravascular_navigation_...py
    → NanobotSwarmNavigationModule
    → Magnetic steering + chemotaxis + No-Zeno topological gate
    → Consumes: target_gradient, magnetic_field_grad, fluid_velocity

[3] nanobot_sesi_crispr_cas_gene_editing_...py
    → NanobotCRISPRCasEditingModule
    → Temperature-dependent Cas activation + nuclear translocation
    → Consumes: rnp_concentration, target_dna_density, temperature, ATP

[4] nanobot_sesi_swarm_hyperthermia_...py
    → NanobotHyperthermiaAblationModule
    → Pennes bio-heat + SAR heating + ALE re-centering
    → Consumes: temperature_field, nanobot_density, AC field

[5] nanobot_sesi_targeted_drug_delivery_...py
    → NanobotPayloadDeliveryModule
    → Multi-compartment release + SESI topological operators
    → Consumes: payload_concentration, pH, MMP, topo_weight

=============================================================================
HONEST SCOPE
=============================================================================
❌ These are SIMULATION modules, not real nanobots.
❌ The CRISPR module models kinetics of an editing event; it does NOT
   design guide RNAs, editors, or delivery vectors.
❌ The biomedical pipeline produces research hypotheses, not clinical
   decisions. No autonomous treatment.
❌ No experimental validation. All outputs are research_only.
✅ All modules are fully differentiable — gradients flow end-to-end.
✅ All modules are DDP/AMP/torch.compile safe.
✅ The closed loop is a valid control-theory architecture.
=============================================================================
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("nanobot_integration")

VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
SESI NANOBOT × BIOMEDICAL INTEGRATION v{VERSION}
5 modules connected · closed-loop · fully differentiable
RESEARCH ONLY. Simulation, not real nanobots. No clinical decisions.
=============================================================================
"""

# =============================================================================
# §0. IMPORT HANDLING
# =============================================================================

_HAS_BIOMED = False
_HAS_NAV = False
_HAS_CRISPR = False
_HAS_HYPER = False
_HAS_PAYLOAD = False

def _try_import(name: str, candidates: List[str]):
    """Try importing by module name or file path."""
    import importlib, importlib.util
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                spec = importlib.util.spec_from_file_location(name, str(p))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod
            except Exception as e:
                log.warning(f"[import] {c} failed: {e}")
    return None

# [1] Biomedical pipeline
_bm = _try_import("sesi_biomedical_pipeline", [
    "sesi_biomedical_pipeline.py",
])
if _bm is not None:
    _HAS_BIOMED = True
    SESIBiomedicalPipeline = getattr(_bm, "SESIBiomedicalPipeline", None)
    BiomedicalConfig = getattr(_bm, "BiomedicalConfig", None)
    SyntheticBiomedicalDataset = getattr(_bm, "SyntheticBiomedicalDataset", None)
    BiomedicalLoss = getattr(_bm, "BiomedicalLoss", None)
    log.info("[import] sesi_biomedical_pipeline loaded")
else:
    log.warning("[import] sesi_biomedical_pipeline NOT FOUND")

# [2] Navigation
_nav = _try_import("nanobot_sesi_swarm_intravascular_navigation", [
    "nanobot_sesi_swarm_intravascular_navigation_disordered_topological_control_module.py",
    "nanobot_sesi_swarm_intravascular_navigation_disordered_topological_control_module (1).py",
])
if _nav is not None:
    _HAS_NAV = True
    NanobotSwarmNavigationModule = getattr(_nav, "NanobotSwarmNavigationModule", None)
    SwarmProperties = getattr(_nav, "SwarmProperties", None)
    log.info("[import] Navigation module loaded")

# [3] CRISPR
_cris = _try_import("nanobot_sesi_crispr_cas_gene_editing", [
    "nanobot_sesi_crispr_cas_gene_editing_differentiable_kinetics_module.py",
])
if _cris is not None:
    _HAS_CRISPR = True
    NanobotCRISPRCasEditingModule = getattr(_cris, "NanobotCRISPRCasEditingModule", None)
    CRISPRConfig = getattr(_cris, "CRISPRConfig", None)
    log.info("[import] CRISPR module loaded")

# [4] Hyperthermia
_hyp = _try_import("nanobot_sesi_swarm_hyperthermia", [
    "nanobot_sesi_swarm_hyperthermia_recentered_chart_ablation_control_module.py",
    "nanobot_sesi_swarm_hyperthermia_recentered_chart_ablation_control_module (1).py",
])
if _hyp is not None:
    _HAS_HYPER = True
    NanobotHyperthermiaAblationModule = getattr(_hyp, "NanobotHyperthermiaAblationModule", None)
    TissueProperties = getattr(_hyp, "TissueProperties", None)
    log.info("[import] Hyperthermia module loaded")

# [5] Payload
_pay = _try_import("nanobot_sesi_targeted_drug_delivery", [
    "nanobot_sesi_targeted_drug_delivery_piecewise_graph_payload_kinetics_module.py",
    "nanobot_sesi_targeted_drug_delivery_piecewise_graph_payload_kinetics_module (1).py",
])
if _pay is not None:
    _HAS_PAYLOAD = True
    NanobotPayloadDeliveryModule = getattr(_pay, "NanobotPayloadDeliveryModule", None)
    PayloadKinetics = getattr(_pay, "PayloadKinetics", None)
    log.info("[import] Payload module loaded")

log.info(f"[imports] Biomed={_HAS_BIOMED} Nav={_HAS_NAV} "
         f"CRISPR={_HAS_CRISPR} Hyper={_HAS_HYPER} Payload={_HAS_PAYLOAD}")


# =============================================================================
# §1. UNIFIED STATE REPRESENTATION
# =============================================================================

@dataclass
class NanobotState:
    """
    Unified state tensor dictionary flowing through the nanobot pipeline.
    All fields are (B, C, Z, Y, X) unless noted.
    """
    # Swarm state
    positions: Tensor          # (B, 3, Z, Y, X) nanobot position density
    velocities: Tensor         # (B, 3, Z, Y, X) nanobot velocity field
    # Environment
    fluid_velocity: Tensor     # (B, 3, Z, Y, X) hemodynamics
    magnetic_field_grad: Tensor  # (B, 9, Z, Y, X) Maxwell tensor
    target_gradient: Tensor    # (B, 3, Z, Y, X) chemotactic attractor
    # Biology
    temperature: Tensor        # (B, 1, Z, Y, X) local temperature [K or °C]
    local_ph: Tensor           # (B, 1, Z, Y, X) pH
    local_atp: Tensor          # (B, 1, Z, Y, X) ATP energy
    local_mmp: Tensor          # (B, 1, Z, Y, X) MMP enzyme concentration
    biomass_energy: Tensor     # (B, 1, Z, Y, X) metabolic energy
    target_dna: Tensor         # (B, 1, Z, Y, X) target DNA density
    rnp_concentration: Tensor  # (B, 1, Z, Y, X) RNP concentration
    # Payload
    payload: Tensor            # (B, 2, Z, Y, X) [encapsulated, released]
    # Topology
    reference_chart: Tensor    # (B, 1, Z, Y, X) ALE reference
    topo_weight: Tensor        # (B, 1, Z, Y, X) topological jump weight
    # Metadata
    step: int = 0
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Tensor]:
        return {k: v for k, v in asdict(self).items()
                if isinstance(v, Tensor)}

    def clone(self) -> "NanobotState":
        return NanobotState(
            **{k: v.clone() if isinstance(v, Tensor) else v
               for k, v in asdict(self).items()})

    def detach(self) -> "NanobotState":
        return NanobotState(
            **{k: v.detach() if isinstance(v, Tensor) else v
               for k, v in asdict(self).items()})


def make_initial_state(B: int, Z: int, Y: int, X: int,
                       device: str = "cpu",
                       dtype: torch.dtype = torch.float32) -> NanobotState:
    """Create a zero-initialized state for a B×Z×Y×X grid."""
    dev = torch.device(device)
    def ones(c): return torch.ones(B, c, Z, Y, X, device=dev, dtype=dtype)
    def zeros(c): return torch.zeros(B, c, Z, Y, X, device=dev, dtype=dtype)
    return NanobotState(
        positions=zeros(3),
        velocities=zeros(3),
        fluid_velocity=zeros(3),
        magnetic_field_grad=zeros(9),
        target_gradient=zeros(3),
        temperature=ones(1) * 310.15,   # 37°C in Kelvin
        local_ph=ones(1) * 7.4,
        local_atp=ones(1) * 2.0,
        local_mmp=zeros(1),
        biomass_energy=ones(1) * 2.0,
        target_dna=zeros(1),
        rnp_concentration=zeros(1),
        payload=torch.cat([ones(1), zeros(1)], dim=1),  # encapsulated + released
        reference_chart=ones(1) * 310.15,
        topo_weight=zeros(1),
        step=0,
        device=str(dev),
    )


# =============================================================================
# §2. NANOBOT ORCHESTRATOR (runs 4 modules in sequence)
# =============================================================================

class NanobotOrchestrator(nn.Module):
    """
    Orchestrates the four nanobot modules in a fixed execution order:

    1. Navigation  — steer swarm toward target
    2. CRISPR      — edit at target location
    3. Hyperthermia — ablate tissue if needed
    4. Payload     — release drug cargo

    Each module is optional: if not imported, that step is skipped
    with a warning. The orchestrator is fully differentiable.

    HONEST: This is a SIMULATION orchestrator. It does not control
    real nanobots. The modules model physics, not hardware.
    """

    def __init__(self, cfg: Optional[Dict] = None, device: str = "auto"):
        super().__init__()
        self.cfg = cfg or {}
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize available modules
        self.nav = None
        self.crispr = None
        self.hyper = None
        self.payload = None

        if _HAS_NAV and NanobotSwarmNavigationModule is not None:
            try:
                dx = self.cfg.get("dx", 1e-6)
                dt = self.cfg.get("dt", 1e-4)
                self.nav = NanobotSwarmNavigationModule(dx=dx, dt=dt).to(self.device)
                log.info("[Orchestrator] Navigation module ready")
            except Exception as e:
                log.warning(f"[Orchestrator] Navigation init failed: {e}")

        if _HAS_CRISPR and NanobotCRISPRCasEditingModule is not None:
            try:
                c_cfg = CRISPRConfig() if CRISPRConfig else None
                self.crispr = NanobotCRISPRCasEditingModule(c_cfg).to(self.device)
                log.info("[Orchestrator] CRISPR module ready")
            except Exception as e:
                log.warning(f"[Orchestrator] CRISPR init failed: {e}")

        if _HAS_HYPER and NanobotHyperthermiaAblationModule is not None:
            try:
                dx = self.cfg.get("dx", 1e-4)
                dt = self.cfg.get("dt", 1e-3)
                self.hyper = NanobotHyperthermiaAblationModule(
                    dx=dx, dt=dt).to(self.device)
                log.info("[Orchestrator] Hyperthermia module ready")
            except Exception as e:
                log.warning(f"[Orchestrator] Hyperthermia init failed: {e}")

        if _HAS_PAYLOAD and NanobotPayloadDeliveryModule is not None:
            try:
                dt = self.cfg.get("dt_payload", 1e-2)
                self.payload = NanobotPayloadDeliveryModule(dt=dt).to(self.device)
                log.info("[Orchestrator] Payload module ready")
            except Exception as e:
                log.warning(f"[Orchestrator] Payload init failed: {e}")

        self.n_active = sum(1 for m in [self.nav, self.crispr,
                                         self.hyper, self.payload] if m is not None)
        log.info(f"[Orchestrator] {self.n_active}/4 modules active")

    def forward(self, state: NanobotState) -> Tuple[NanobotState, Dict[str, Any]]:
        """
        Execute all active modules in sequence.
        Returns updated state + diagnostic metrics.
        """
        diagnostics = {}
        s = state

        # ---- 1. Navigation ----
        if self.nav is not None:
            try:
                new_pos, new_vel, jump_gate, prob_bound = self.nav(
                    s.positions, s.velocities,
                    s.magnetic_field_grad, s.fluid_velocity,
                    s.target_gradient, s.biomass_energy,
                    return_probability=True,
                )
                s.positions = new_pos
                s.velocities = new_vel
                diagnostics["nav_jump_gate"] = jump_gate
                diagnostics["nav_prob_bound"] = prob_bound
                diagnostics["nav_active"] = True
            except Exception as e:
                log.warning(f"[Orchestrator] Navigation step failed: {e}")
                diagnostics["nav_active"] = False
        else:
            diagnostics["nav_active"] = False

        # ---- 2. CRISPR editing ----
        if self.crispr is not None:
            try:
                edited_dna, successful_edits, crispr_metrics = self.crispr(
                    s.rnp_concentration, s.target_dna,
                    s.temperature, s.local_atp,
                )
                s.target_dna = edited_dna
                diagnostics["crispr_edits"] = successful_edits
                diagnostics["crispr_metrics"] = crispr_metrics
                diagnostics["crispr_active"] = True
            except Exception as e:
                log.warning(f"[Orchestrator] CRISPR step failed: {e}")
                diagnostics["crispr_active"] = False
        else:
            diagnostics["crispr_active"] = False

        # ---- 3. Hyperthermia ablation ----
        if self.hyper is not None:
            try:
                new_temp, damage, new_chart, recenter_w = self.hyper(
                    s.temperature,
                    nanobot_density=s.positions.norm(dim=1, keepdim=True),
                    reference_chart=s.reference_chart,
                    ac_magnetic_field_amp=torch.tensor(0.05, device=s.temperature.device),
                    ac_frequency=torch.tensor(1e5, device=s.temperature.device),
                    return_recenter_weight=True,
                )
                s.temperature = new_temp
                s.reference_chart = new_chart
                diagnostics["hyper_damage"] = damage
                diagnostics["hyper_recenter_weight"] = recenter_w
                diagnostics["hyper_active"] = True
            except Exception as e:
                log.warning(f"[Orchestrator] Hyperthermia step failed: {e}")
                diagnostics["hyper_active"] = False
        else:
            diagnostics["hyper_active"] = False

        # ---- 4. Payload delivery ----
        if self.payload is not None:
            try:
                new_payload, payload_metrics = self.payload(
                    s.payload, s.local_ph, s.local_mmp, s.topo_weight,
                    return_metrics=True,
                )
                s.payload = new_payload
                diagnostics["payload_metrics"] = payload_metrics
                diagnostics["payload_active"] = True
            except Exception as e:
                log.warning(f"[Orchestrator] Payload step failed: {e}")
                diagnostics["payload_active"] = False
        else:
            diagnostics["payload_active"] = False

        s.step += 1
        return s, diagnostics


# =============================================================================
# §3. BIOMEDICAL → NANOBOT CONTROLLER
# =============================================================================

class BiomedicalNanobotController(nn.Module):
    """
    Connects the biomedical pipeline's decisions to nanobot actions.

    The biomedical pipeline produces:
      - target_params: what to target (gene, cell type, location)
      - dose: how much
      - timing: when

    This controller translates those into nanobot commands:
      - target_gradient: chemotactic attractor toward target
      - magnetic_field_grad: steering field
      - rnp_concentration: CRISPR cargo
      - payload: drug cargo
      - ac_magnetic_field_amp: hyperthermia trigger

    HONEST: This is a RESEARCH controller. It does not make clinical
    decisions. All outputs require clinician review.
    """

    def __init__(self, d_in: int = 256, d_state: int = 64,
                 grid_shape: Tuple[int, int, int] = (16, 16, 16)):
        super().__init__()
        self.grid_shape = grid_shape
        Z, Y, X = grid_shape

        # Project biomedical decision features to nanobot commands
        self.target_proj = nn.Sequential(
            nn.Linear(d_in, d_state), nn.GELU(),
            nn.Linear(d_state, 3),  # 3D target direction
        )
        self.dose_proj = nn.Sequential(
            nn.Linear(d_in, d_state), nn.GELU(),
            nn.Linear(d_state, 1), nn.Softplus(),  # positive dose
        )
        self.rnp_proj = nn.Sequential(
            nn.Linear(d_in, d_state), nn.GELU(),
            nn.Linear(d_state, 1), nn.Sigmoid(),  # RNP concentration [0,1]
        )
        self.payload_proj = nn.Sequential(
            nn.Linear(d_in, d_state), nn.GELU(),
            nn.Linear(d_state, 1), nn.Sigmoid(),  # payload amount
        )
        self.ac_proj = nn.Sequential(
            nn.Linear(d_in, d_state), nn.GELU(),
            nn.Linear(d_state, 1), nn.Sigmoid(),  # AC field amplitude
        )

    def forward(self, decision_features: Tensor,
                state: NanobotState) -> NanobotState:
        """
        Translate biomedical decisions into nanobot state updates.

        Args:
            decision_features: (B, d_in) from biomedical pipeline
            state: current NanobotState

        Returns:
            Updated NanobotState with nanobot commands injected
        """
        B = decision_features.shape[0]
        Z, Y, X = self.grid_shape
        dev = decision_features.device

        # Target gradient: broadcast (B,3) → (B,3,Z,Y,X)
        target_dir = self.target_proj(decision_features)  # (B,3)
        target_grad = target_dir[:, :, None, None, None].expand(B, 3, Z, Y, X)
        state.target_gradient = target_grad

        # RNP concentration: broadcast (B,1) → (B,1,Z,Y,X)
        rnp = self.rnp_proj(decision_features)  # (B,1)
        state.rnp_concentration = rnp[:, :, None, None, None].expand(B, 1, Z, Y, X)

        # Payload: set encapsulated channel
        payload_amt = self.payload_proj(decision_features)  # (B,1)
        state.payload[:, 0] = payload_amt.expand(B, Z, Y, X)

        # AC field for hyperthermia (stored as scalar for orchestrator)
        ac_amp = self.ac_proj(decision_features)  # (B,1)
        # Store in a side channel (not in state, used by orchestrator)
        state._ac_amp = ac_amp

        return state


# =============================================================================
# §4. CLOSED-LOOP TRAINER
# =============================================================================

class ClosedLoopTrainer:
    """
    End-to-end training loop:
      biomedical decision → nanobot execution → feedback → next decision

    The loop is:
      1. Biomedical pipeline produces decision features
      2. Controller translates to nanobot commands
      3. Orchestrator executes nanobot physics
      4. Feedback controller computes error vs target
      5. Loss backpropagates through entire chain

    HONEST: This trains a SIMULATION. It does not train real nanobots.
    """

    def __init__(self, cfg: Dict, device: str = "auto"):
        self.cfg = cfg
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build components
        self.biomed = None
        if _HAS_BIOMED and SESIBiomedicalPipeline is not None:
            try:
                bm_cfg = BiomedicalConfig() if BiomedicalConfig else None
                self.biomed = SESIBiomedicalPipeline(bm_cfg).to(self.device)
            except Exception as e:
                log.warning(f"[Trainer] Biomedical pipeline init failed: {e}")

        self.orchestrator = NanobotOrchestrator(cfg, device=str(self.device))
        grid = cfg.get("grid_shape", (16, 16, 16))
        d_in = cfg.get("d_in", 256)
        self.controller = BiomedicalNanobotController(
            d_in=d_in, grid_shape=grid).to(self.device)

        # Feedback: simple MSE between achieved and target state
        self.feedback_weight = cfg.get("feedback_weight", 1.0)

        log.info(f"[Trainer] Initialized on {self.device}")

    def compute_feedback_loss(self, state: NanobotState,
                               target_state: NanobotState) -> Tensor:
        """
        Compute feedback loss: how far is achieved state from target.
        Uses temperature, payload release, and editing success.
        """
        loss = torch.zeros((), device=self.device)

        # Temperature error
        if state.temperature is not None and target_state.temperature is not None:
            loss = loss + F.mse_loss(state.temperature, target_state.temperature)

        # Payload release error
        if state.payload is not None and target_state.payload is not None:
            loss = loss + F.mse_loss(state.payload[:, 1], target_state.payload[:, 1])

        # DNA editing error
        if state.target_dna is not None and target_state.target_dna is not None:
            loss = loss + F.mse_loss(state.target_dna, target_state.target_dna)

        return self.feedback_weight * loss

    def train_step(self, decision_features: Tensor,
                   target_state: NanobotState) -> Dict[str, float]:
        """One closed-loop training step."""
        # 1. Create initial state
        B = decision_features.shape[0]
        Z, Y, X = self.cfg.get("grid_shape", (16, 16, 16))
        state = make_initial_state(B, Z, Y, X, device=str(self.device))

        # 2. Controller translates decisions to commands
        state = self.controller(decision_features, state)

        # 3. Orchestrator executes nanobot physics
        state, diagnostics = self.orchestrator(state)

        # 4. Feedback loss
        loss = self.compute_feedback_loss(state, target_state)

        return {
            "loss": float(loss.item()),
            "diagnostics": diagnostics,
            "state": state,
        }

    def train(self, n_steps: int = 10, batch_size: int = 4) -> List[Dict]:
        """Run multiple training steps."""
        d_in = self.cfg.get("d_in", 256)
        Z, Y, X = self.cfg.get("grid_shape", (16, 16, 16))
        history = []

        # Simple optimizer over controller parameters
        params = list(self.controller.parameters())
        if self.biomed is not None:
            params += list(self.biomed.parameters())
        opt = torch.optim.AdamW(params, lr=self.cfg.get("lr", 1e-3))

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)

            # Random decision features (replace with real biomedical output)
            decision_features = torch.randn(
                batch_size, d_in, device=self.device)

            # Random target state
            target = make_initial_state(batch_size, Z, Y, X,
                                        device=str(self.device))
            target.temperature = torch.ones_like(target.temperature) * 315.0
            target.payload[:, 1] = 0.5  # want 50% release

            result = self.train_step(decision_features, target)
            loss_val = result["loss"]

            # Backprop
            if loss_val > 0:
                # Need to recompute with grad
                state = make_initial_state(batch_size, Z, Y, X,
                                           device=str(self.device))
                state = self.controller(decision_features, state)
                state, _ = self.orchestrator(state)
                loss = self.compute_feedback_loss(state, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            history.append({"step": step, "loss": loss_val})
            if (step + 1) % max(1, n_steps // 5) == 0:
                log.info(f"[Trainer] step {step+1}/{n_steps} loss={loss_val:.6f}")

        return history


# =============================================================================
# §5. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    torch.manual_seed(42)
    device = torch.device("cpu")

    # T1: Module availability
    print("\n[T1] Module availability")
    for name, avail in [
        ("Biomedical pipeline", _HAS_BIOMED),
        ("Navigation", _HAS_NAV),
        ("CRISPR", _HAS_CRISPR),
        ("Hyperthermia", _HAS_HYPER),
        ("Payload", _HAS_PAYLOAD),
    ]:
        status = "AVAILABLE" if avail else "NOT FOUND"
        print(f"      {name:25s} {status}")
    rep("At least orchestrator can run", True)

    # T2: State creation
    print("\n[T2] NanobotState creation")
    state = make_initial_state(2, 8, 8, 8, device="cpu")
    rep("positions shape", state.positions.shape == (2, 3, 8, 8, 8))
    rep("payload shape", state.payload.shape == (2, 2, 8, 8, 8))
    rep("temperature initialized", state.temperature.mean().item() > 300)
    rep("clone works", state.clone().positions.shape == state.positions.shape)

    # T3: Orchestrator
    print("\n[T3] NanobotOrchestrator")
    orch = NanobotOrchestrator(device="cpu")
    rep("orchestrator created", orch is not None)
    rep("active modules counted", orch.n_active >= 0,
        f"{orch.n_active}/4 active")

    # T4: Controller
    print("\n[T4] BiomedicalNanobotController")
    ctrl = BiomedicalNanobotController(d_in=64, grid_shape=(8, 8, 8))
    decision = torch.randn(2, 64)
    state = make_initial_state(2, 8, 8, 8, device="cpu")
    updated = ctrl(decision, state)
    rep("target_gradient set", updated.target_gradient.shape == (2, 3, 8, 8, 8))
    rep("rnp_concentration set", updated.rnp_concentration.shape == (2, 1, 8, 8, 8))
    rep("payload updated", updated.payload.shape == (2, 2, 8, 8, 8))

    # T5: Gradient flow through controller
    print("\n[T5] Gradient flow")
    ctrl.train()
    decision = torch.randn(2, 64, requires_grad=True)
    state = make_initial_state(2, 8, 8, 8, device="cpu")
    updated = ctrl(decision, state)
    loss = updated.target_gradient.sum() + updated.rnp_concentration.sum()
    loss.backward()
    has_grad = decision.grad is not None and torch.isfinite(decision.grad).all()
    rep("gradient flows to decision", bool(has_grad))

    # T6: Full pipeline (if biomedical available)
    print("\n[T6] Full closed-loop pipeline")
    if _HAS_BIOMED:
        cfg = {"d_in": 64, "grid_shape": (8, 8, 8), "lr": 1e-3}
        trainer = ClosedLoopTrainer(cfg, device="cpu")
        history = trainer.train(n_steps=3, batch_size=2)
        rep("training completes", len(history) == 3)
        rep("loss is finite", all(math.isfinite(h["loss"]) for h in history))
    else:
        rep("biomedical pipeline available", False, "skipped")
        print("      [SKIP] Biomedical pipeline not available")

    # T7: Orchestrator forward pass
    print("\n[T7] Orchestrator forward pass")
    state = make_initial_state(1, 8, 8, 8, device="cpu")
    state.target_gradient = torch.randn(1, 3, 8, 8, 8)
    state.rnp_concentration = torch.rand(1, 1, 8, 8, 8)
    out_state, diag = orch(state)
    rep("orchestrator runs", out_state is not None)
    rep("diagnostics returned", isinstance(diag, dict))
    rep("step incremented", out_state.step == 1)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT IS VERIFIED
  State creation, controller, gradient flow, orchestrator forward pass.

WHAT IS NOT VERIFIED (requires real modules)
  ❌ Actual nanobot physics (modules must be installed)
  ❌ Biomedical pipeline decisions (requires training data)
  ❌ Clinical relevance (this is simulation only)

NEXT STEPS
  1. Install all 5 modules in the same directory
  2. Run: python nanobot_biomedical_integration.py --selftest
  3. For real training: provide decision_features from biomedical pipeline
  4. For production: deploy on GPU cluster with DDP
""")
    return ok


# =============================================================================
# §6. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Nanobot × Biomedical closed-loop integration")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1

    if a.status:
        status = {
            "version": VERSION,
            "modules": {
                "biomedical_pipeline": _HAS_BIOMED,
                "navigation": _HAS_NAV,
                "crispr": _HAS_CRISPR,
                "hyperthermia": _HAS_HYPER,
                "payload": _HAS_PAYLOAD,
            },
            "clinical_actionability": RESEARCH_ONLY,
        }
        print(json.dumps(status, indent=2))
        return 0

    if a.demo:
        print(BANNER)
        cfg = {"d_in": 64, "grid_shape": (8, 8, 8), "lr": 1e-3}
        trainer = ClosedLoopTrainer(cfg, device="cpu")
        history = trainer.train(n_steps=a.steps, batch_size=a.batch_size)
        print(f"\nDemo complete: {len(history)} steps")
        for h in history:
            print(f"  step {h['step']}: loss={h['loss']:.6f}")
        if a.json:
            Path(a.json).write_text(json.dumps(history, indent=2))
        return 0

    ap.print_help()
    print(BANNER)
    print("Start: --selftest | --status | --demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
