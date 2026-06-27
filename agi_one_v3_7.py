# =============================================================================
# AGI ONE v3.7 — Production-Grade AGI Architecture
# Central Orchestration Hub for the ONE Ecosystem
# =============================================================================
#
# Developer  : Yoon A Limsuwan / MSPS NETWORK
#              MY SOUL MOVE BY POWER OF HOLY SPIRIT
# License    : MIT
# Year       : 2026
# ORCID      : 0009-0008-2374-0788
# GitHub     : https://github.com/yoonalimsuwan
# Email      : msps4u@gmail.com
#
# AI Development Assistants:
#   Claude   (Anthropic)        — architecture co-design, missing-component
#                                 specification, code review, AGI completeness
#                                 analysis v1.0 → v2.0 → v3.0; curriculum
#                                 training, PCGrad, InfoNCE alignment,
#                                 EcosystemOrchestrator design
#   GPT-4o   (OpenAI)           — supplementary architecture consultation
#   Gemini   (Google DeepMind)  — cross-validation of design decisions
#   DeepSeek (DeepSeek AI)      — open-source alignment review
#
# =============================================================================
# VERSION HISTORY
# ─────────────────
# v1.0.0  Initial AGI ONE:
#         Perception, Language, WorkingMemory, EpisodicMemory,
#         GlobalWorkspace, WorldModel(RSSM), PlanningCEM, MetaCognition,
#         PsycheTriad, MultiScaleIntegrator, AGITrainer
#
# v2.0.0  Production upgrade:
#   [NEW] MPPI planner   — replaces CEM, GPU-parallel trajectory sampling
#   [NEW] DreamerV3 world model — symlog, two-hot reward, free-bits KL,
#         categorical latents (straight-through)
#   [NEW] PPO full       — clipped surrogate + GAE(λ) + entropy bonus
#   [NEW] DreamerCompoundLoss — world/actor/critic with uncertainty weighting
#   [NEW] SSCStabilizer  — SSC as Transformer hidden-state stabilizer
#   [NEW] InterfaceAttention — Interface Detector as adaptive attention prior
#   [NEW] CSOCComputeController — edge-of-chaos adaptive depth
#   [NEW] StructuralLangevinDiffusion — geometry-aware latent diffusion
#   [NEW] PsycheExecutiveLayer — Id→Goal / Ego→Plan / Superego→Safety
#   [NEW] OpenScienceRegistry — dataset attribution & provenance
#   [NEW] BSD ONE, HODGE ONE, GRH ONE — mathematical reasoning layer
#   [UPG] VisionEncoder   → ViT-style patch embedding + RoPE
#   [UPG] LanguageModule  → GPT-style causal LM + RoPE
#   [UPG] AudioEncoder    → Mel-spectrogram + Conformer-lite
#   [UPG] LossBalancer    → Kendall uncertainty weighting
#   [UPG] AGITrainer v2   → PPO + Dreamer + PSY joint training
#
# v3.0.0  Distributed Ecosystem + Stable Multi-Task Training  (this file):
#
#   PROBLEM SOLVED: Joint training of heterogeneous losses
#   (physics PDE, discrete math, RL, language) causes Gradient Interference
#   (Negative Transfer) — gradients from one domain destroy another domain's
#   learned representations. Loss landscape becomes intractable for AdamW.
#
#   THREE-PART SOLUTION:
#
#   [A] Multi-Stage Curriculum Training (CurriculumScheduler)
#       Phase 1 — FOUNDATION  : Train physics/math surrogates (SFNO3D, GNOFold,
#                               GNOEvolution, GNOHodge, GNONumberTheory, MSNOv3,
#                               NGOPhysics) + DreamerV3 World Model independently.
#                               Core ecosystem modules are frozen after convergence.
#       Phase 2 — ALIGNMENT   : Freeze physics backbone. Train Language↔Physics
#                               bridge via InfoNCE Contrastive Alignment Loss.
#                               Cross-modal latent spaces geometrically aligned
#                               without direct gradient interference.
#       Phase 3 — COGNITIVE   : Unlock PPO actor-critic + PsycheExecutiveLayer.
#                               Fine-tune policy with reward signal + Free Energy.
#                               Ecosystem surrogates remain partially frozen
#                               (backbone frozen, heads fine-tuned).
#
#   [B] PCGrad Gradient Surgery (PCGradOptimizer)
#       During Phase 3 joint training, conflicting gradients between task pairs
#       are detected (g_i · g_j < 0) and projected to orthogonal components,
#       preventing mutual destruction of learned representations.
#       Based on: Yu et al. 2020 "Gradient Surgery for Multi-Task Learning"
#
#   [C] InfoNCE Contrastive Alignment (CrossModalAlignmentLoss)
#       Physics latent ↔ Language latent aligned via symmetric InfoNCE loss.
#       Attracts (physics_i, language_i) pairs; repels (physics_i, language_j≠i).
#       Avoids direct gradient coupling between PDE solver and language decoder.
#       Based on: CLIP (Radford et al. 2021), SimCLR (Chen et al. 2020)
#
#   [D] EcosystemOrchestrator — Distributed Module Hub
#       AGI ONE v3 no longer embeds ecosystem engines directly. Instead,
#       EcosystemOrchestrator maintains references to uploaded surrogate modules
#       (structural_fno_3d.py, structural_gno_fold_v4.py, etc.) and queries
#       them via a unified adapter interface. This enables:
#         • True distributed training (each surrogate trains independently)
#         • Selective freezing per curriculum phase
#         • Per-domain optimizer assignment (Decoupled Optimizers)
#         • Hot-swappable surrogate modules
#
#   [E] Decoupled Domain Optimizers (AGITrainerV3)
#       Each domain group gets its own optimizer with domain-appropriate LR:
#         physics_surrogates  : AdamW  lr=1e-6  (high-precision, sensitive)
#         math_surrogates     : AdamW  lr=5e-7  (discrete logic, very slow)
#         language_module     : AdamW  lr=1e-5  (standard LLM fine-tuning)
#         world_model         : AdamW  lr=3e-4  (DreamerV3 default)
#         policy_heads        : AdamW  lr=1e-4  (PPO actor/critic)
#         psyche_layer        : AdamW  lr=3e-4  (Free Energy)
#         loss_balancer       : Adam   lr=1e-3  (Kendall σ params)
#
# v3.1.0  AGI ONE : Id, Ego, Super Ego / Plus — full central integration:
#   [NEW] AGIONE.psyche_plus — speculative-axiom evolution layer that
#         complements PsycheExecutiveLayer's moment-to-moment safety gate
#         with a slower, self-evolving "axiom bank" loop (see
#         agi_one_psyche_plus_v33.py). Runs every
#         `psyche_plus_run_every_n_steps` and folds its (safety-gated)
#         output back into the workspace latent.
#   [NEW] Opt-in multi-LLM external consensus (Claude/Gemini/GPT). OFF by
#         default — `psyche_plus_enable_external_llm=False` — so no AGI
#         ONE deployment ever spends API credits without explicit consent.
#         When enabled, calls are concurrent, timeout-bounded, and rate-
#         limited via `psyche_plus_consult_every_n_steps`; every response
#         is returned as a structured `ExternalConsensusResult`, never
#         silently discarded.
#   [NEW] AGIState.psyche_plus_validity / psyche_plus_axiom_count /
#         psyche_plus_codified / external_consensus.
#
# v3.2.0  Simulation-Quality-Aware Speculative Gating + EVOLUTION BV  [NEW]
#         Primary developer of this patch: Claude (Anthropic). Gemini and
#         GPT were not involved in this revision; DeepSeek not involved.
#   [NEW] SurrogateAdapter / EcosystemOrchestrator now carry an optional
#         per-surrogate `quality_fn` hook (manual float or live callable)
#         exposed via `EcosystemOrchestrator.quality_report()`. This lets
#         the *actual simulation quality* of a domain (e.g. Hodge period
#         loss, RH/BSD GUE-statistics loss, EVOLUTION BV CME residual) —
#         not just its latent embedding — reach AGI ONE's central hub.
#         No existing call site is broken: `quality_fn=None` (the default)
#         reproduces the exact pre-v3.2 behaviour everywhere.
#   [NEW] Step 7-B now passes the ecosystem-wide quality report into
#         `psyche_plus(ws, quality_score=...)`, closing the loop discussed
#         with Yoon: speculative axioms can now be gated on simulation
#         quality, not only on self-referential plausibility.
#   [NEW] EVOLUTION BV (`structural_gno_evolution_bv_standalone.py`)
#         registered into the ecosystem under domain `"evolution_bv"` via
#         `GNOEvolutionBVEncoderAdapter`, with its BV-3 analytic
#         certification (CME residual) wired as that domain's live
#         `quality_fn`.
#
# v3.3.0  EVOLUTION BV — Mode-4 Cell-Population Wiring  [NEW]
#         Primary developer of this patch: Claude (Anthropic). Gemini, GPT,
#         and DeepSeek not involved in this revision.
#   [NEW] `GNOEvolutionBVEncoderAdapter` now drives a small fixed synthetic
#         Mode-3 (ch3d) grid in addition to its existing synthetic Mode-1
#         graph, mirroring `structural_gno_evolution_bv_standalone.py`'s own
#         smoke-test pattern exactly (same N/E/Nx convention). This grid
#         feeds `self.bv_model.cellpop_bridge.rollout(...)` so the adapter
#         can surface Mode-4 (cell-population) health alongside Mode-1's
#         BV-3 certification — wholly additive, no change to the BV file
#         itself, no change to `.encode()`'s existing latent contract.
#   [NEW] `GNOEvolutionBVEncoderAdapter.get_quality_score()` now averages
#         two independent [0, 1] signals when both are available:
#           (a) BV-3 CME residual  → exp(-|residual|)            (unchanged)
#           (b) Mode-4 population stability → derived from the rollout's
#               final `n_alive` (collapse/blow-up guard against
#               `cellpop_n_max`) and how close `rate_trace[-1]` sits to
#               `cfg.pop_target_division_rate` (a calibration guard).
#         Either signal alone still degrades gracefully to the neutral 0.5
#         default exactly as before if its underlying bridge is
#         unavailable (e.g. `bv_full_theory_one` or `cell_population_one`
#         not importable) — this patch only adds a second readout, it does
#         not change the existing fallback behaviour of the first.
#   [NEW] `attach_evolution_bv_to_ecosystem()` unchanged in signature;
#         registration now transparently carries the combined quality
#         signal since it simply forwards `wrapper.get_quality_score`.
#
# v3.4.1  HODGE — Rewired for structural_gno_hodge.py v2 / hodge_one v2  [NEW]
#         Primary developer of this patch: Claude (Anthropic). Gemini, GPT,
#         and DeepSeek not involved in this revision.
#   [BREAKING, internal only] `structural_gno_hodge.py` was rebuilt (v2)
#         around hodge_one v2's two concrete polarized Hodge varieties
#         (`VarietyClass.ABELIAN` complex-torus particle adapter vs.
#         `VarietyClass.K3_ABSTRACT` R^22 generator) instead of a 1-D toy
#         line. The pre-v3.4.1 `GNOHodgeEncoderAdapter` constructed a bare
#         `StructuralGNOHodge(cfg)` and called it directly as
#         `model(x_pos, target, sigma)` — that constructor/call shape no
#         longer exists at the top level (StructuralGNOHodge is now an
#         *internal* shared 1-D slice network). No external call site of
#         `attach_gno_hodge_to_ecosystem()` changes (same signature), so
#         this is breaking only for code that imported
#         `StructuralGNOHodge` from this file directly, which nothing
#         outside `GNOHodgeEncoderAdapter` ever did.
#   [NEW] `GNOHodgeEncoderAdapter` now wraps `UnifiedHodgeOperatorTrainer`
#         (built via `structural_gno_hodge.build_system`), the v2 entry
#         point that owns variety dispatch, the active GNO model,
#         `hodge_one.CycleClassMap`, and `hodge_one.LearnableSOCKernel`.
#         `.encode()` mirrors the trainer's own `_abelian_step`/`_k3_step`
#         (minus the optimizer step) and reads out `computed_periods`
#         — the one tensor shape stable across both varieties — rather
#         than the particle tensor, which only exists for ABELIAN.
#   [NEW] `GNOHodgeEncoderAdapter.get_quality_score()` now scores against
#         the *real* hodge_one v2 training losses (period-fit +
#         Hodge-Riemann + integrality, plus spacing/entropy/smooth for
#         ABELIAN) via `compute_total_loss_abelian` / `compute_total_loss_k3`,
#         mapped through `exp(-total_loss)` — replacing the old anti-
#         collapse spacing-ratio heuristic with the model's own objective.
#   [NEW] `hodge_cfg.variety` ("abelian" | "k3_abstract") is now the single
#         switch selecting which polarized Hodge structure AGI ONE's hodge
#         domain reasons over; default "abelian" matches `SGNOHodgeConfig`.
#
# v3.7.0  EVOLUTION BV — POP-2 Organelle/Phenotype Sub-Cellular Wiring  [NEW]
#         Primary developer of this patch: Claude (Anthropic). Gemini, GPT,
#         and DeepSeek not involved in this revision.
#   [NEW] Rewired for structural_gno_evolution_bv_standalone.py v1.2.0
#         (POP-2: OrganelleLayer/PhenotypeLayer attached on top of the
#         existing Mode-4 CellPopulationTrainingBridge — requires
#         cell_population_one v1.1.0+, with its own graceful degradation
#         if that requirement isn't met). No call-site signature anywhere
#         in this file changes; v1.1.0-era checkpoints/configs continue to
#         load unchanged since POP-2 introduces no new trainable parameters
#         on the wrapped BV model itself (only this adapter's own quality
#         probe is extended).
#   [NEW] `GNOEvolutionBVEncoderAdapter.get_quality_score()` now blends a
#         THIRD independent [0, 1] signal when available:
#           (c) Organelle health — runs the existing fixed synthetic ch3d
#               grid through Mode 3 → `cellpop_bridge.rollout(...)` (same
#               probe call the Mode-4 signal (b) already makes — not a
#               second forward pass) and reads
#               `rollout.organelle_health_trace[-1]`, mapped from its
#               native ~[-1, 1] range to [0, 1] via `(health + 1) / 2`.
#         All three signals are still simple-averaged, and any subset
#         still degrades gracefully to the cached previous value if its
#         underlying bridge/layer is unavailable — identical convention to
#         v3.2's BV-3 signal and v3.3's Mode-4 signal. `_population_quality_score`
#         now also computes and caches the organelle reading alongside the
#         existing survival/calibration blend — see `_last_organelle_quality`.
#   [NEW] `GNOEvolutionBVEncoderAdapter.__init__` accepts no new
#         constructor arguments; the v3.3 ch3d synthetic-grid buffers are
#         reused as-is for the POP-2 probe (the BV file attaches
#         Organelle/PhenotypeLayer onto the SAME `cellpop_bridge.population`
#         the v3.3 Mode-4 probe already drives, so no second synthetic
#         batch is needed).
#   [NEW] `attach_evolution_bv_to_ecosystem()` unchanged in signature;
#         registration transparently carries the now-three-signal quality
#         blend since it simply forwards `wrapper.get_quality_score`.
#
# v3.6.0  NGO PHYSICS — Mode-4 Entangled-Pair CHSH Correlator Wiring  [NEW]
#         Primary developer of this patch: Claude (Anthropic). Gemini, GPT,
#         and DeepSeek not involved in this revision.
#   [NEW] `GNOPhysicsEncoderAdapter.encode()` now mean-blends its existing
#         Mode-3 (Yang–Mills) latent with a second latent derived from
#         `StructuralGNOPhysics.forward_entangled()` (ngo_physics_one.py
#         v1.1.0's entangled-pair polarization/spin correlator), via a
#         new learned `lift_entangled` / `readout_entangled` pair —
#         wholly additive, same pattern as `GNOEvolutionBVEncoderAdapter`'s
#         v3.3 Mode-4 wiring. `lift_entangled`'s raw output is squashed
#         per-column into physically-bounded [θ_a, θ_b, V, source_id]
#         before being passed to `forward_entangled` (angles via tanh·π,
#         visibility via sigmoid → [0,1]).
#   [NEW] `GNOPhysicsEncoderAdapter.get_quality_score()` now averages two
#         independent [0, 1] signals when both compute successfully:
#           (a) collider-head boundedness probe                (unchanged)
#           (b) CHSH physical-realizability probe — runs the model's own
#               `forward_entangled` at the textbook CHSH-optimal angle
#               quadruple (same defaults as `NGOPhysicsInference.
#               chsh_test()`), forms S = E(a,b)+E(a,b')+E(a',b)-E(a',b'),
#               and scores `exp(-max(|S| - TSIRELSON_BOUND, 0))` — 1.0
#               while the prediction stays inside the Tsirelson bound,
#               decaying smoothly past it. Either signal alone still
#               degrades gracefully to the cached previous value if its
#               probe raises, exactly as the v3.3 BV pattern does.
#   [NEW] Optional import of `LOCAL_BOUND` / `TSIRELSON_BOUND` from
#         `bell_chsh_one.py` (textbook fallback values used if that file
#         isn't importable) — kept as a single shared numerical source of
#         truth with both `ngo_physics_one.py` and `bell_chsh_one.py`
#         itself, per Yoon's existing convention for this constant pair.
#   [NEW] `attach_ngo_physics_to_ecosystem()` unchanged in signature;
#         registration transparently carries the combined quality signal
#         since it simply forwards `wrapper.get_quality_score`.
#
# =============================================================================
# THEORETICAL FOUNDATIONS
# ────────────────────────
#   Structural Itô Calculus (Limsuwan 2025)
#   Self-Organised Criticality + CSOC (SOC universality chain)
#   Renormalisation Group multi-scale smoothing
#   Active Inference / Free Energy Principle (Friston 2010)
#   Global Workspace Theory (Baars 1988; Dehaene 2011)
#   Integrated Information Theory Φ (Tononi 2004)
#   Deep Equilibrium Models — DEQ (Bai et al. 2019)
#   DreamerV3 (Hafner et al. 2023)
#   MPPI (Williams et al. 2017)
#   PPO + GAE (Schulman et al. 2017 / 2015)
#   Rotary Positional Embedding RoPE (Su et al. 2021)
#   Conformer (Gulati et al. 2020)
#   ViT patch embedding (Dosovitskiy et al. 2020)
#   Uncertainty-weighted multi-task loss (Kendall et al. 2018)
#   Edge-of-Chaos / Critical Brain Hypothesis (Langton 1990)
#   Geometry-aware Manifold Diffusion (Song et al. 2020+)
#
# =============================================================================
# 
#
# =============================================================================
# OPEN SCIENCE & DATA PROVENANCE
# ────────────────────────────────
# AGI ONE upholds open science principles. All training datasets must be
# traceable to their originating research lab, institution, or investigator.
# In the AGI era, attribution extends beyond paper authors to every
# laboratory, dataset contributor, and research centre that supplied data.
# See OpenScienceRegistry for the provenance API.
#
# =============================================================================
# MIT License
# Copyright (c) 2026 Yoon A Limsuwan / MSPS NETWORK
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [AGI_ONE v2]  %(levelname)s  %(message)s",
)
logger = logging.getLogger("AGI_ONE_v2")

AGI_ONE_VERSION: str = "3.4.1"

# =============================================================================
# ONE Ecosystem — Graceful Imports
# =============================================================================

# ── ONE Core primitives ───────────────────────────────────────────────────────
# [v3.5 NEW] one_core_mental / one_core_v3 / one_core_fold / one_core_evolution_v2
# were optional-import utility primitives (SemanticStateContraction, CSOCBase,
# DifferentiableSOC, DifferentiableRG, soft_clamp, ...) with an in-file fallback
# already used whenever they weren't importable. Per Yoon's instruction, AGI ONE
# now calls ONLY the seven already-wired surrogate modules (structural_gno_fold,
# structural_gno_hodge, mental_structural_operator_v3, structural_fno_3d_v2_1,
# structural_gno_evolution_bv_standalone, ngo_physics_one,
# structural_gno_numbertheory) and never reaches into any other ecosystem
# module directly — so these four imports are removed outright and every
# call site below now unconditionally uses the local fallback implementation
# (SemanticStateContraction/DifferentiableSOC/DifferentiableRG → None with the
# pre-existing None-fallback branch each class already had; soft_clamp → the
# plain-tanh version below, used everywhere unconditionally).
HAS_ONE_CORE_MENTAL = False
HAS_ONE_CORE         = False
HAS_ONE_CORE_FOLD    = False
HAS_ONE_CORE_EVO     = False

def soft_clamp(x, lo, hi):
    c = (hi + lo) / 2.0; s = (hi - lo) / 2.0 + 1e-8
    return c + s * torch.tanh((x - c) / s)

# [v3.5 NEW] mental_one (the raw engine) removed — not one of the seven
# wired surrogates; MENTAL ONE is represented in AGI ONE only via
# mental_structural_operator_v3 (MentalStructuralNeuralOperator), already
# wired through attach_msno_to_ecosystem(). HAS_MENTAL_ONE forced False so
# the self.mental_one assignment a few hundred lines down resolves to None
# without needing its own edit.
HAS_MENTAL_ONE = False

# [v3.5 NEW] psy_one_bridge_diff import removed — not one of the seven wired
# surrogates. The PsycheTriad/PsycheConfig integration block inside
# PsycheExecutiveLayer (which called into it directly) is removed below too;
# PsycheExecutiveLayer's own planner_net/safety_net/normative_policy (always
# present, no external dependency) continue to provide Id/Ego/Superego
# behaviour unchanged.

# ── AGI ONE : Id, Ego, Super Ego / Plus  [v3.1 NEW] ──────────────────────────
# Speculative-axiom evolution layer + opt-in multi-LLM external consensus.
# Complements PsycheExecutiveLayer (moment-to-moment action safety) with a
# slower hypothesis-evolution loop over the same workspace latent.
try:
    from agi_one_psyche_plus_v33 import (
        AGIOnePsychePlus, PsychePlusConfig, ExternalConsensusResult,
    )
    HAS_PSYCHE_PLUS = True
    logger.info("✓ agi_one_psyche_plus_v33  [v3.1 NEW]")
except ImportError:
    HAS_PSYCHE_PLUS = False
    logger.warning("✗ agi_one_psyche_plus_v33 not found — PsychePlus layer disabled")

# ── EVOLUTION ONE — BV (Batalin-Vilkovisky) edition  [v3.2 NEW, rewired
#    v3.7 for v1.2.0 POP-2 organelle/phenotype] ────────────────────────────
# Adds a genuine BV gauge-theory layer (cross-modal distillation, mass-
# conservation physics loss, periodic analytic CME/QME/BRST certification)
# on top of the production GNO surrogate, at zero extra trainable parameters.
# [v3.7 NEW] The standalone file's own SGNO_BV_VERSION is now imported too
# (purely informational — logged below) so a version mismatch between this
# file's expectations and what's actually on disk is visible at import time
# rather than only showing up later as a missing `cellpop_bridge.organelle`.
# Registered into EcosystemOrchestrator under domain "evolution_bv" via
# GNOEvolutionBVEncoderAdapter (defined further below, near SurrogateAdapter).
try:
    from structural_gno_evolution_bv_standalone import (
        StructuralGNOEvolutionBV, SGNOEvoBVConfig, BatchData as _GNOBatchData,
        SGNO_BV_VERSION,
    )
    HAS_GNO_EVOLUTION_BV = True
    logger.info(f"✓ structural_gno_evolution_bv_standalone  (v{SGNO_BV_VERSION})  [v3.7 rewired for POP-2]")
except ImportError:
    HAS_GNO_EVOLUTION_BV = False
    SGNO_BV_VERSION = "unavailable"
    logger.warning("✗ structural_gno_evolution_bv_standalone not found — evolution_bv domain disabled")

# ── PHYSICS surrogate: Structural FNO 3D  [v3.4 NEW] ─────────────────────────
# Tries the versioned filename first (current on-disk name), then the bare
# name (docstring convention used elsewhere in this file / future rename).
try:
    from structural_fno_3d_v2_1 import StructuralFNO3D, SFNO_VERSION
    HAS_SFNO3D = True
    logger.info(f"✓ structural_fno_3d  (v{SFNO_VERSION})  [v3.4 NEW]")
except ImportError:
    try:
        from structural_fno_3d import StructuralFNO3D, SFNO_VERSION
        HAS_SFNO3D = True
        logger.info(f"✓ structural_fno_3d  (v{SFNO_VERSION})  [v3.4 NEW]")
    except ImportError:
        HAS_SFNO3D = False
        logger.warning("✗ structural_fno_3d not found — physics/sfno3d domain disabled")

# ── MENTAL ONE surrogate: Mental Structural Neural Operator  [v3.4 NEW] ──────
try:
    from mental_structural_operator_v3 import MentalStructuralNeuralOperator, MSNO_VERSION
    HAS_MSNO = True
    logger.info(f"✓ mental_structural_operator_v3  (v{MSNO_VERSION})  [v3.4 NEW]")
except ImportError:
    HAS_MSNO = False
    logger.warning("✗ mental_structural_operator_v3 not found — mental domain disabled")

# ── FOLD surrogate: Structural GNO Fold  [v3.4 NEW, rewired v3.5 for v4] ─────
# [v3.5 NEW] structural_gno_fold was bumped v3 → v4: _build_protein_graph and
# _build_grid_graph now build their radius graph via a sparse neighbor-list
# (real_fold_one_v2.scipy_radius_graph / FastNeighborList, or a local
# scipy-cKDTree / pure-PyTorch cell-list fallback) instead of a dense
# torch.cdist(coords, coords) matrix — see structural_gno_fold_v4.py's module
# header ("v4 FIX") for the full rationale. This only changes how edges are
# *constructed*; StructuralGNOFold's public API (SGNOConfig, forward(),
# forward_phase_field()) is unchanged, the produced edge_index/edge_attr are
# identical in shape and semantics, and v4's module-level self-tests confirm
# the sparse path reproduces the OLD dense formulation's edge set and
# distances exactly at small N. GNOFoldEncoderAdapter below needed no logic
# changes — only this import line moved from `_v3` to `_v4`. The new name is
# tried first; the old `_v3` module is kept as a fallback so this file still
# loads against an environment that hasn't picked up v4 yet.
try:
    from structural_gno_fold_v4 import StructuralGNOFold, SGNOConfig as SGNOFoldConfig, SGNO_VERSION as SGNO_FOLD_VERSION
    HAS_GNO_FOLD = True
    logger.info(f"✓ structural_gno_fold_v4  (v{SGNO_FOLD_VERSION})  [v3.5 NEW — sparse radius graph]")
except ImportError:
    try:
        from structural_gno_fold_v3 import StructuralGNOFold, SGNOConfig as SGNOFoldConfig, SGNO_VERSION as SGNO_FOLD_VERSION
        HAS_GNO_FOLD = True
        logger.warning(
            f"⚠ structural_gno_fold_v4 not found — fell back to structural_gno_fold_v3 "
            f"(v{SGNO_FOLD_VERSION}). Dense O(N²) graph construction applies in this "
            f"fallback path; install/update to v4 before running fold mode at large N."
        )
    except ImportError:
        HAS_GNO_FOLD = False
        logger.warning("✗ structural_gno_fold_v4 (or _v3 fallback) not found — fold domain disabled")

# ── HODGE surrogate: Structural GNO Hodge  [v3.4 NEW, rewired v3.4.1] ────────
# [v3.4.1 NEW] structural_gno_hodge.py was rebuilt (v2) around hodge_one v2's
# two concrete varieties (ABELIAN complex-torus particle adapter vs.
# K3_ABSTRACT R^22 generator) instead of the old 1-D-line StructuralGNOHodge
# called directly with (x_pos, target_hodge, sigma). The old
# `StructuralGNOHodge(cfg)` constructor signature and direct
# `model(x_pos, target, sigma)` call this file used pre-v3.4.1 no longer
# match — StructuralGNOHodge is now an *internal* 1-D slice network reused
# by StructuralGNOHodgeAbelianAdapter; the top-level entry point is
# `UnifiedHodgeOperatorTrainer` (built via `build_system(cfg, device)`).
# See GNOHodgeEncoderAdapter below for the rewired adapter.
try:
    from structural_gno_hodge import (
        SGNOHodgeConfig,
        UnifiedHodgeOperatorTrainer,
        build_system as build_gno_hodge_system,
        project_to_h11_batch,
        compute_total_loss_abelian,
        compute_total_loss_k3,
    )
    import hodge_one as hodge_one_v2
    HAS_GNO_HODGE = True
    logger.info("✓ structural_gno_hodge  [v3.4 NEW, rewired v3.4.1 for hodge_one v2]")
except ImportError:
    HAS_GNO_HODGE = False
    logger.warning("✗ structural_gno_hodge (or hodge_one v2) not found — hodge domain disabled")

# ── MATH surrogate: Structural GNO Number Theory  [v3.4 NEW] ─────────────────
try:
    from structural_gno_numbertheory import StructuralGNONumberTheory, SGNOConfig as SGNONumberTheoryConfig
    HAS_GNO_NUMBERTHEORY = True
    logger.info("✓ structural_gno_numbertheory  [v3.4 NEW]")
except ImportError:
    HAS_GNO_NUMBERTHEORY = False
    logger.warning("✗ structural_gno_numbertheory not found — math domain disabled")

# ── PHYSICS surrogate: NGO Physics ONE (collider / cosmo / Yang-Mills /
#    entangled-pair CHSH correlator)  [v3.4 NEW, extended v3.6 for Mode 4]
try:
    from ngo_physics_one import StructuralGNOPhysics, NGOPhysicsConfig
    HAS_NGO_PHYSICS = True
    logger.info("✓ ngo_physics_one  [v3.4 NEW, Mode 4 wired v3.6]")
except ImportError:
    HAS_NGO_PHYSICS = False
    logger.warning("✗ ngo_physics_one not found — physics/ngo domain disabled")

# [v3.6 NEW] Optional: LOCAL_BOUND / TSIRELSON_BOUND, used only by
# GNOPhysicsEncoderAdapter's Mode-4 quality probe so it can classify the
# physics model's own CHSH prediction the same way ngo_physics_one.py's
# own NGOPhysicsInference.chsh_test() does — imported, never redefined,
# so the two files cannot silently drift on what counts as a violation.
# Entirely optional: if bell_chsh_one.py isn't present, the Mode-4 quality
# signal just falls back to the boundedness-only probe (see below).
try:
    from bell_chsh_one import LOCAL_BOUND, TSIRELSON_BOUND
    HAS_BELL_CHSH_BOUNDS = True
except ImportError:
    HAS_BELL_CHSH_BOUNDS = False
    LOCAL_BOUND = 2.0          # textbook fallback values, matches bell_chsh_one
    TSIRELSON_BOUND = 2.0 * math.sqrt(2.0)

# [v3.5 NEW] The following direct engine-level imports were removed —
# none of these are among the seven surrogate modules AGI ONE is now scoped
# to call (structural_gno_fold_v4, structural_gno_hodge,
# mental_structural_operator_v3, structural_fno_3d_v2_1,
# structural_gno_evolution_bv_standalone, ngo_physics_one,
# structural_gno_numbertheory). All were previously only instantiated into
# self.* attributes consumed solely by the status/quality_report dict
# further below (never called in any forward pass), except BSD/GRH/HODGE
# (queried directly inside MathReasoningLayer.forward — that whole layer is
# removed below) and the Langevin bridges (also status-only). Removed:
#   langevin_mental_bridge, structural_langevin_mental,
#   real_fold_one_v2.RealFoldONEEngine, real_fold_one_ht_v2,
#   structural_langevin_fold_v2, evolution_one_v3, evolution_one_epidemiological_viral_v4,
#   structural_langevin_evo_v3, structuralfluctuatinghydro_v6, super_dns_one_v6,
#   structural_langevin_v3, standard_one, yang_mills_mass_gap_one, rh_one__1_,
#   bsd_one, grh_one, hodge_one (the old direct-query path — `hodge_one` is
#   still imported further up, but only as the structural_gno_hodge enum
#   dependency, per Yoon's explicit instruction to keep that one).
HAS_LANGEVIN_MENTAL    = False
HAS_STRUCT_LANG_MENTAL = False
HAS_REAL_FOLD          = False
HAS_REAL_FOLD_HT       = False
HAS_LANGEVIN_FOLD      = False
HAS_EVOLUTION          = False
HAS_EPIDEMIC           = False
HAS_LANGEVIN_EVO       = False
HAS_FH                 = False
HAS_DNS                = False
HAS_LANGEVIN_MD        = False
HAS_STANDARD           = False
HAS_YANG_MILLS         = False
HAS_RH                 = False
HAS_BSD                = False
HAS_GRH                = False
HAS_HODGE              = False

# ── Optional: HuggingFace / torchvision / torchaudio ─────────────────────────
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_HF = True
    logger.info("✓ HuggingFace transformers")
except ImportError:
    HAS_HF = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


# =============================================================================
# DEVICE UTILITY
# =============================================================================

def get_agi_device(preferred: str = "cuda") -> torch.device:
    p = preferred.lower()
    if p == "cuda"   and torch.cuda.is_available():    return torch.device("cuda")
    if p == "mps"    and torch.backends.mps.is_available(): return torch.device("mps")
    if p == "ascend" and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# SECTION 0 — OPEN SCIENCE REGISTRY
# =============================================================================

@dataclass
class DatasetRecord:
    """Single dataset attribution record."""
    dataset_id      : str
    title           : str
    source_lab      : str
    institution     : str
    contributors    : List[str]
    doi             : Optional[str]   = None
    url             : Optional[str]   = None
    license         : str             = "Unknown"
    year            : Optional[int]   = None
    description     : str             = ""
    tags            : List[str]       = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "dataset_id"  : self.dataset_id,
            "title"       : self.title,
            "source_lab"  : self.source_lab,
            "institution" : self.institution,
            "contributors": self.contributors,
            "doi"         : self.doi,
            "url"         : self.url,
            "license"     : self.license,
            "year"        : self.year,
            "description" : self.description,
            "tags"        : self.tags,
        }


class OpenScienceRegistry:
    """
    Dataset Attribution and Provenance Tracking Registry.

    AGI ONE principle: In the AGI era, attribution extends beyond paper
    authors to every laboratory, institution, and researcher who contributed
    data.  This registry ensures every dataset used in training is fully
    credited and traceable.

    Usage:
        registry = OpenScienceRegistry()
        registry.register(DatasetRecord(
            dataset_id="openneuro_ds003944",
            title="EEG Resting State Dataset",
            source_lab="Neuroimaging Lab",
            institution="Stanford University",
            contributors=["J. Smith", "A. Lee"],
            doi="10.18112/openneuro.ds003944",
            license="CC-BY-4.0",
            year=2021,
        ))
        registry.cite("openneuro_ds003944")
        report = registry.provenance_report()
    """

    def __init__(self) -> None:
        self._records: Dict[str, DatasetRecord] = {}
        self._usage_log: List[Dict] = []

    def register(self, record: DatasetRecord) -> None:
        """Register a dataset with full attribution."""
        self._records[record.dataset_id] = record
        logger.info(
            f"[OpenScience] Registered: {record.dataset_id}  "
            f"| Lab: {record.source_lab}  | {record.institution}"
        )

    def cite(self, dataset_id: str, context: str = "") -> Optional[DatasetRecord]:
        """Record usage of a dataset (for audit trail)."""
        if dataset_id not in self._records:
            logger.warning(f"[OpenScience] Unknown dataset: {dataset_id}")
            return None
        self._usage_log.append({
            "dataset_id": dataset_id,
            "context"   : context,
            "timestamp" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        return self._records[dataset_id]

    def all_records(self) -> List[DatasetRecord]:
        return list(self._records.values())

    def provenance_report(self) -> Dict:
        """Full provenance report: all registered datasets + usage log."""
        return {
            "agi_one_version"  : AGI_ONE_VERSION,
            "report_generated" : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datasets"         : [r.to_dict() for r in self._records.values()],
            "usage_log"        : self._usage_log,
            "principle": (
                "AGI ONE upholds open science attribution: every laboratory, "
                "dataset contributor, and research centre that supplied data "
                "is credited. Attribution in the AGI era is broader than "
                "individual authorship — it encompasses the full data "
                "provenance chain."
            ),
        }

    def save_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.provenance_report(), f, indent=2, ensure_ascii=False)
        logger.info(f"[OpenScience] Provenance report saved: {path}")

    def load_registry(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("datasets", []):
            self.register(DatasetRecord(**d))


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

class CognitivePriority(Enum):
    BALANCED      = "balanced"
    PERCEPTION    = "perception"
    LANGUAGE      = "language"
    PLANNING      = "planning"
    INTROSPECTION = "introspection"
    PHYSICS       = "physics"


@dataclass
class AGIConfig:
    """Master configuration for AGI ONE v2.0."""
    # ── Core dimensions ──────────────────────────────────────────────────────
    latent_dim           : int   = 512
    action_dim           : int   = 64
    memory_slots         : int   = 128
    episodic_capacity    : int   = 10_000
    planning_horizon     : int   = 15
    n_transformer_heads  : int   = 8
    n_transformer_layers : int   = 6      # upgraded from 4

    # ── Modality flags ────────────────────────────────────────────────────────
    use_vision           : bool  = True
    use_audio            : bool  = True
    use_language         : bool  = True
    use_proprioception   : bool  = True
    use_timeseries       : bool  = True

    # ── ONE Ecosystem ─────────────────────────────────────────────────────────
    # [v3.5 NEW] These flags previously gated the direct engine-level
    # imports removed above (mental_one, real_fold_one_v2, evolution_one_v3,
    # the physics engines, standard_one, yang_mills_mass_gap_one, rh_one,
    # bsd_one/grh_one/hodge_one via MathReasoningLayer, psy_one_bridge_diff).
    # They no longer gate anything — kept only so AGIConfig(...) and
    # build_agi_one(use_all_modules=...) stay backward-compatible for
    # existing callers that still pass them; setting any of these to a
    # different value has no effect on AGI ONE's behaviour anymore.
    use_mental_one       : bool  = True
    use_psy_bridge       : bool  = True
    use_real_fold        : bool  = True
    use_evolution        : bool  = True
    use_physics          : bool  = True
    use_standard_one     : bool  = False
    use_yang_mills       : bool  = False
    use_rh               : bool  = False
    use_bsd              : bool  = True   # [v2.0]
    use_grh              : bool  = True   # [v2.0]
    use_hodge            : bool  = True   # [v2.0]

    # ── Language backbone ─────────────────────────────────────────────────────
    language_backend     : str   = "builtin"
    language_model_id    : str   = "distilbert-base-uncased"
    language_dim         : int   = 768
    vocab_size           : int   = 32_000

    # ── PSY Bridge ────────────────────────────────────────────────────────────
    psyche_mode          : str   = "healthy"
    gumbel_tau           : float = 1.0
    gumbel_hard          : bool  = False
    anderson_depth       : int   = 5
    lambda_reg           : float = 2.5

    # ── Psyche Plus: speculative-axiom evolution  [v3.1 NEW] ─────────────────
    use_psyche_plus              : bool  = True
    psyche_plus_max_axioms       : int   = 64
    psyche_plus_axiom_weight     : float = 0.5
    psyche_plus_codify_threshold : float = 0.6
    psyche_plus_run_every_n_steps: int   = 5     # speculative cycle cadence
    # External multi-LLM consensus is OFF by default — opt-in only, so no
    # AGI ONE deployment ever spends API credits without explicit consent.
    psyche_plus_enable_external_llm    : bool  = False
    psyche_plus_consult_every_n_steps  : int   = 50
    psyche_plus_llm_timeout_s          : float = 12.0

    # ── MPPI Planner [v2.0] ───────────────────────────────────────────────────
    mppi_n_samples       : int   = 1024
    mppi_temperature     : float = 1.0
    mppi_noise_sigma     : float = 0.5

    # ── DreamerV3 World Model [v2.0] ──────────────────────────────────────────
    dreamer_stoch_size   : int   = 32
    dreamer_stoch_classes: int   = 32
    dreamer_det_size     : int   = 512
    dreamer_reward_bins  : int   = 255  # two-hot encoding bins
    dreamer_free_bits    : float = 1.0  # KL free bits threshold
    dreamer_kl_balance   : float = 0.8  # posterior vs prior weight

    # ── PPO Training [v2.0] ───────────────────────────────────────────────────
    ppo_clip_eps         : float = 0.2
    ppo_epochs           : int   = 4
    ppo_gae_lambda       : float = 0.95
    ppo_gamma            : float = 0.99
    value_loss_coef      : float = 0.5
    entropy_coef         : float = 0.01

    # ── CSOC Compute Control [v2.0] ───────────────────────────────────────────
    csoc_min_layers      : int   = 4
    csoc_max_layers      : int   = 32
    csoc_sigma_target    : float = 1.0

    # ── Training ─────────────────────────────────────────────────────────────
    lr                   : float = 3e-4
    weight_decay         : float = 1e-4
    grad_clip_norm       : float = 1.0
    use_amp              : bool  = True
    use_grad_checkpoint  : bool  = False
    warmup_steps         : int   = 1_000

    # ── Meta ─────────────────────────────────────────────────────────────────
    device               : torch.device = field(
        default_factory=lambda: get_agi_device("cuda")
    )
    cognitive_priority   : CognitivePriority = CognitivePriority.BALANCED
    verbose              : bool  = True
    seed                 : int   = 42


# =============================================================================
# SECTION 2 — ROTARY POSITIONAL EMBEDDING (RoPE)
# Su et al. 2021 — used by ViT, GPT, and language module
# =============================================================================

class RotaryEmbedding(nn.Module):
    """RoPE positional embedding — improves extrapolation over learned PE."""

    def __init__(self, dim: int, max_seq: int = 4096) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._max_seq = max_seq

    def _get_cos_sin(self, seq_len: int, device: torch.device):
        t      = torch.arange(seq_len, device=device).float()
        freqs  = torch.einsum("i,j->ij", t, self.inv_freq)
        emb    = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors (B, H, L, D)."""
        seq_len = q.shape[-2]
        cos, sin = self._get_cos_sin(seq_len, q.device)  # (L, D)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,L,D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# =============================================================================
# SECTION 3 — SSC PRIMITIVES INTEGRATED INTO TRANSFORMER
# [v2.0 NEW] — SSCStabilizer, InterfaceAttention, CSOCComputeController
# =============================================================================

class SSCStabilizer(nn.Module):
    """
    [v2.0 NEW] SSC as Transformer Hidden-State Stabilizer.

    Wraps SemanticStateContraction (from one_core_mental) as a per-channel
    EMA filter applied to the hidden state of each Transformer layer.

    Pipeline:
        hidden_state (B, L, D)
            → per-channel stress σ = std over sequence
            → SSC EMA filter
            → refined hidden_state

    This stabilizes the latent representation and reduces forgetting,
    acting as a learnable memory compression / coherent-context keeper.
    """

    def __init__(self, d_model: int, epsilon_fp: float = 0.005) -> None:
        super().__init__()
        self.d_model = d_model

        # [v3.5 NEW] one_core_mental import removed (see top of file) — SSC
        # always uses the local learnable-EMA fallback now, unconditionally.
        self.log_alpha = nn.Parameter(
            torch.full((d_model,), math.log(epsilon_fp))
        )
        self.ssc = None

        # Projection: refine state after SSC
        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        # Persistent stabilized state
        self.register_buffer("stabilized", torch.zeros(d_model))
        self.register_buffer("_init", torch.tensor(False))

    def reset(self) -> None:
        self.stabilized.zero_()
        self._init.fill_(False)
        if self.ssc is not None:
            self.ssc.reset()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden : (B, L, D) transformer hidden state
        Returns:
            stabilized hidden : (B, L, D)
        """
        B, L, D = hidden.shape

        # Compute per-channel "structural stress" (std across sequence)
        sigma = hidden.std(dim=1)   # (B, D)
        sigma_mean = sigma.mean(dim=0)  # (D,)

        if self.ssc is not None:
            sigma_filtered = self.ssc(sigma_mean.mean().unsqueeze(0))
            scale = (sigma_filtered / (sigma_mean.mean() + 1e-8)).clamp(0.5, 2.0)
            refined = hidden * scale
        else:
            alpha = torch.sigmoid(self.log_alpha)   # (D,)
            if not self._init.item():
                self.stabilized.data = sigma_mean.detach()
                self._init.fill_(True)
            self.stabilized.data = (
                (1 - alpha.detach()) * self.stabilized + alpha.detach() * sigma_mean.detach()
            )
            scale = (self.stabilized / (sigma_mean + 1e-8)).clamp(0.5, 2.0)
            refined = hidden * scale.unsqueeze(0).unsqueeze(0)

        return self.refine(refined)


class InterfaceAttention(nn.Module):
    """
    [v2.0 NEW] Interface Detector as Adaptive Attention Prior.

    Based on InterfaceDetectorBase (one_core_mental):
    detects phase transitions / boundary points in the hidden-state sequence,
    then adds a learned bias to attention logits so the model attends more
    to interface tokens (points where knowledge/context is changing).

    Pipeline:
        attn_logits (B, H, L, L)
        + interface_score(hidden) → (B, 1, L, 1)  broadcast as column bias
        ──────────────────────────────────────────
        modified_attn_logits

    Inspired by:
    - InterfaceDetectorBase in structural_langevin_mental.py
    - Adaptive attention frontier work (e.g. Lei et al., "Fastformer")
    """

    def __init__(self, d_model: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

        # Differentiable interface score: how much each token is a "boundary"
        self.interface_net = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        # Learnable scale for bias strength
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden      : torch.Tensor,   # (B, L, D)
        attn_logits : torch.Tensor,   # (B, H, L, L)
    ) -> torch.Tensor:
        """
        Returns:
            modified_attn_logits : (B, H, L, L)
        """
        # Interface score per token: (B, L, 1)
        iface = self.interface_net(hidden)              # (B, L, 1)

        # Gradient of score across sequence → detect transitions
        iface_diff = torch.zeros_like(iface)
        iface_diff[:, 1:, :] = (iface[:, 1:, :] - iface[:, :-1, :]).abs()

        # Normalize and scale
        scale = torch.exp(self.log_scale)
        bias  = iface_diff * scale                      # (B, L, 1)
        bias  = bias.unsqueeze(1)                       # (B, 1, L, 1) → column bias
        bias  = bias.expand_as(attn_logits)

        return attn_logits + bias

    def get_interface_map(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns interface scores for visualization. (B, L)"""
        return self.interface_net(hidden).squeeze(-1)


class CSOCComputeController(nn.Module):
    """
    [v2.0 NEW] CSOC as Dynamic Compute Controller (Adaptive Depth).

    Based on CSOCBase (one_core_mental) + DifferentiableSOC:
    monitors criticality of the current hidden state and dynamically
    decides how many Transformer layers to execute.

    Easy problems (low criticality)  → use min_layers
    Hard problems (high criticality) → use up to max_layers

    Pipeline:
        hidden_state → criticality_score ∈ [0,1]
        n_layers = min_layers + round(score * (max_layers - min_layers))

    Analogous to:
    - Adaptive Computation Time (Graves 2016)
    - Test-time compute scaling (DeepSeek-R1 style reasoning)
    - Edge-of-Chaos adaptive complexity (Langton 1990)
    """

    def __init__(
        self,
        d_model        : int,
        min_layers     : int   = 4,
        max_layers     : int   = 32,
        sigma_target   : float = 1.0,
        epsilon_fp     : float = 0.005,
    ) -> None:
        super().__init__()
        self.min_layers  = min_layers
        self.max_layers  = max_layers
        self.sigma_target = sigma_target

        # [v3.5 NEW] one_core_mental import removed (see top of file) — both
        # the SSC smoother and DifferentiableSOC were optional enhancements
        # that already had a None fallback (self.ssc is None below routes
        # compute_criticality() to use the raw, unsmoothed score directly;
        # self.diff_soc was assigned but never read elsewhere in this class,
        # so dropping it has no behavioural effect).
        self.ssc      = None
        self.diff_soc = None

        # Criticality estimator: score ∈ [0, 1]
        self.critic_net = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

        # Learnable bias for criticality threshold
        self.bias = nn.Parameter(torch.tensor(0.0))

        self._last_n_layers = min_layers
        self._last_score    = 0.0

    def reset(self) -> None:
        if self.ssc is not None:
            self.ssc.reset()

    def compute_criticality(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            hidden : (B, L, D) hidden state
        Returns:
            (criticality_score tensor [0,1], n_layers to use int)
        """
        h_mean  = hidden.mean(dim=1)   # (B, D)
        score   = self.critic_net(h_mean).mean()  # scalar ∈ [0,1]
        score   = score + self.bias.sigmoid() * 0.1

        # SSC smoothing on criticality signal
        if self.ssc is not None:
            score_sm = self.ssc(score.unsqueeze(0)).squeeze()
        else:
            score_sm = score

        score_clamped = soft_clamp(score_sm, 0.0, 1.0)

        # Dynamic depth
        span     = self.max_layers - self.min_layers
        n_layers = self.min_layers + int(round(float(score_clamped.item()) * span))
        n_layers = max(self.min_layers, min(self.max_layers, n_layers))

        self._last_n_layers = n_layers
        self._last_score    = float(score_clamped.item())

        return score_clamped, n_layers

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self.compute_criticality(hidden)


class StructuralLangevinDiffusion(nn.Module):
    """
    [v2.0 NEW] Geometry-Aware Latent Diffusion via Structural Langevin.

    Based on StructuralItoBase (one_core_mental):
    standard diffusion uses uniform Gaussian noise over latent space,
    but here noise amplitude G(x) varies with the interface structure
    of the current state — creating manifold-aware / geometry-aware diffusion.

    dX_t = -∇U(X_t) dt + G(X_t) dW_t + ½ G(X_t) ∇G(X_t) dt  (Itô correction)

    where G(x) = 1 + amp · interface_mask(x)

    Applications in AGI ONE:
    - Latent imagination / dreaming (world model)
    - Exploration in planning
    - Latent space regularization during training
    """

    def __init__(
        self,
        d_model                : int,
        interface_amplification: float = 2.0,
        n_steps                : int   = 10,
        dt                     : float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.amp       = interface_amplification
        self.n_steps   = n_steps
        self.dt        = dt

        # Interface detector: (B, D) → (B, D) mask ∈ [0, 1]
        self.iface_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )

        # Energy function U(x) = ½ ||x||²  (Gaussian prior)
        # Learnable scale
        self.log_dt = nn.Parameter(torch.tensor(math.log(dt)))

        # [v3.5 NEW] one_core_mental import removed (see top of file) — `rg`
        # was assigned but never read elsewhere in this class, so it's kept
        # as a permanent None stub (no behavioural change).
        self.rg = None

    def _g_field(self, x: torch.Tensor) -> torch.Tensor:
        """G(x) = 1 + amp · interface_mask(x)  shape: same as x."""
        mask = self.iface_net(x)
        return 1.0 + self.amp * mask

    def _ito_correction(self, x: torch.Tensor) -> torch.Tensor:
        """½ G(x) · ∇G(x) — via autograd."""
        x_req = x.detach().requires_grad_(True)
        g     = self._g_field(x_req).sum()
        grad  = torch.autograd.grad(g, x_req, create_graph=False)[0]
        g_val = self._g_field(x.detach())
        return 0.5 * g_val * grad

    def forward(
        self,
        x       : torch.Tensor,   # (B, D) latent vector
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Run geometry-aware Langevin diffusion for n_steps.

        Returns:
            x_diffused : (B, D) diffused latent
        """
        dt = torch.exp(self.log_dt).item()

        for _ in range(self.n_steps):
            # Gradient of energy: ∇U = x  (Gaussian prior)
            grad_u = x

            # G field
            G = self._g_field(x)            # (B, D) ∈ [1, 1+amp]

            # Itô correction
            try:
                ito_corr = self._ito_correction(x)
            except Exception:
                ito_corr = torch.zeros_like(x)

            # Noise: G(x) · dW
            dW    = torch.randn_like(x) * math.sqrt(dt) * noise_scale
            noise = G * dW

            # Langevin step: dx = -∇U dt + G dW + ½G∇G dt
            x = x - grad_u * dt + noise + ito_corr * dt

            # Soft clamp to prevent explosion
            x = soft_clamp(x, -10.0, 10.0)

        return x


# =============================================================================
# SECTION 4 — UPGRADED PERCEPTION MODULE
# ViT-style patch encoder + Conformer-lite audio + improved fusion
# =============================================================================

class PatchViTEncoder(nn.Module):
    """
    [v2.0 UPGRADED] ViT-style patch-based vision encoder.

    Replaces simple ResNet-18 with:
    - Patch embedding (non-overlapping patches → linear projection)
    - RoPE positional embedding
    - Transformer encoder (configurable depth)
    - CLS token aggregation

    Much stronger than ResNet-18 for complex visual reasoning.
    """

    def __init__(
        self,
        latent_dim  : int,
        img_size    : int   = 224,
        patch_size  : int   = 16,
        in_channels : int   = 3,
        n_heads     : int   = 8,
        n_layers    : int   = 6,
        device      : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.n_patches   = (img_size // patch_size) ** 2
        patch_dim        = in_channels * patch_size * patch_size
        self.patch_size  = patch_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, latent_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # RoPE
        self.rope = RotaryEmbedding(latent_dim // n_heads)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(latent_dim)

        # SSC stabilizer on CLS output [v2.0]
        self.ssc_stab = SSCStabilizer(latent_dim)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            (B, latent_dim) vision embedding
        """
        B = x.shape[0]
        # Patch embedding: (B, D, n_h, n_w) → (B, n_patches, D)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        p   = torch.cat([cls, p], dim=1)   # (B, 1+n_patches, D)

        # Transformer
        out = self.transformer(p)
        out = self.ssc_stab(out)
        out = self.norm(out)
        return out[:, 0, :]   # CLS output: (B, D)


class ConformerAudioEncoder(nn.Module):
    """
    [v2.0 UPGRADED] Conformer-lite audio encoder.

    Mel-spectrogram → Conv subsampling → Conformer blocks → pooling.
    Conformer (Gulati et al. 2020) combines CNN local patterns + Transformer
    global context, state of the art for audio/speech.
    """

    def __init__(
        self,
        latent_dim  : int,
        n_mfcc      : int   = 80,
        n_heads     : int   = 4,
        n_layers    : int   = 4,
        device      : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        if HAS_TORCHAUDIO:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=16_000, n_fft=512, n_mels=n_mfcc,
            )
        else:
            self.mel = None

        # Conv subsampling: (B, 1, n_mfcc, T) → (B, latent_dim//2, T//4)
        self.conv_sub = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, latent_dim // 4, 3, stride=2, padding=1), nn.GELU(),
        )

        # Linear projection after flattening mel dim
        mel_out_dim = (n_mfcc // 4) * (latent_dim // 4)
        self.proj   = nn.Linear(mel_out_dim, latent_dim)

        # Conformer-lite: Transformer + depthwise conv feed-forward
        class ConformerBlock(nn.Module):
            def __init__(self, d: int, h: int) -> None:
                super().__init__()
                self.ff1  = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, d*4), nn.SiLU(), nn.Linear(d*4, d),
                )
                self.attn = nn.MultiheadAttention(d, h, batch_first=True)
                self.conv = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Conv1d(d, d*2, 1),
                    nn.GLU(dim=1),
                    nn.Conv1d(d, d, 31, padding=15, groups=d),
                    nn.BatchNorm1d(d),
                    nn.SiLU(),
                    nn.Conv1d(d, d, 1),
                )
                self.ff2  = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, d*4), nn.SiLU(), nn.Linear(d*4, d),
                )
                self.norm = nn.LayerNorm(d)

            def forward(self, x):
                x = x + 0.5 * self.ff1(x)
                a, _ = self.attn(x, x, x)
                x = x + a
                xc = x.transpose(1, 2)
                xc = self.conv(xc).transpose(1, 2)
                x = x + xc
                x = x + 0.5 * self.ff2(x)
                return self.norm(x)

        self.conformer = nn.Sequential(
            *[ConformerBlock(latent_dim, n_heads) for _ in range(n_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, 1, T) or (B, n_mfcc, T) mel frames
        Returns:
            (B, latent_dim)
        """
        B = waveform.shape[0]
        if self.mel is not None and waveform.shape[1] == 1:
            x = self.mel(waveform.squeeze(1))  # (B, n_mfcc, T)
        else:
            x = waveform

        # Conv subsampling: treat mel as 2D image (1 channel)
        x   = x.unsqueeze(1)            # (B, 1, n_mfcc, T)
        x   = self.conv_sub(x)          # (B, D//4, n_mfcc//4, T//4)
        T2  = x.shape[-1]
        x   = x.permute(0, 3, 1, 2)    # (B, T', D//4, n_mfcc//4)
        x   = x.flatten(2)              # (B, T', mel_out_dim)
        x   = self.proj(x)              # (B, T', D)

        x   = self.conformer(x)         # (B, T', D)
        x   = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, D)
        return x


class ProprioceptionEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, latent_dim),
        )
        self.to(device)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class TimeSeriesEncoder(nn.Module):
    """TCN with SSC stabilization — EEG / sensor / physics fields."""

    def __init__(self, in_channels: int, latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3), nn.GELU(),
            nn.Conv1d(128, 256, 5, padding=2), nn.GELU(),
            nn.Conv1d(256, latent_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(latent_dim * 8, latent_dim),
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)


class CrossModalFusion(nn.Module):
    """
    Multi-modal cross-attention fusion with InterfaceAttention prior.
    [v2.0] InterfaceAttention added.
    """

    def __init__(self, latent_dim: int, n_heads: int, device: torch.device) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer    = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cls_token      = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.iface_attn     = InterfaceAttention(latent_dim)   # [v2.0]
        self.ssc_stab       = SSCStabilizer(latent_dim)         # [v2.0]
        self.pool           = nn.Linear(latent_dim, latent_dim)
        self.latent_dim     = latent_dim
        self.to(device)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        B     = embeddings[0].shape[0]
        tokens = torch.stack(embeddings, dim=1)              # (B, n_mod, D)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)             # (B, 1+n, D)
        out    = self.transformer(tokens)
        out    = self.ssc_stab(out)                          # [v2.0] stabilize
        return self.pool(out[:, 0, :])


class PerceptionModule(nn.Module):
    """
    AGI ONE v2.0 Perception Layer.
    Vision: ViT-style patch encoder (upgraded from ResNet-18)
    Audio:  Conformer-lite (upgraded from MFCC+CNN)
    All others: unchanged from v1.0
    """

    def __init__(self, cfg: AGIConfig) -> None:
        super().__init__()
        D       = cfg.latent_dim
        device  = cfg.device
        self.device           = device
        self.use_vision       = cfg.use_vision
        self.use_audio        = cfg.use_audio
        self.use_proprio      = cfg.use_proprioception
        self.use_timeseries   = cfg.use_timeseries

        if cfg.use_vision:
            self.vision_enc = PatchViTEncoder(
                latent_dim=D, n_heads=cfg.n_transformer_heads,
                n_layers=cfg.n_transformer_layers, device=device,
            )
        if cfg.use_audio:
            self.audio_enc  = ConformerAudioEncoder(
                latent_dim=D, n_heads=4, n_layers=4, device=device,
            )
        if cfg.use_proprioception:
            self.proprio_enc = ProprioceptionEncoder(64, D, device)
        if cfg.use_timeseries:
            self.ts_enc = TimeSeriesEncoder(64, D, device)

        self.text_embed = nn.Embedding(cfg.vocab_size, D)
        self.text_proj  = nn.Linear(D, D)

        self.fusion = CrossModalFusion(D, cfg.n_transformer_heads, device)
        self.to(device)

    def forward(
        self,
        image      : Optional[torch.Tensor] = None,
        waveform   : Optional[torch.Tensor] = None,
        token_ids  : Optional[torch.Tensor] = None,
        proprio    : Optional[torch.Tensor] = None,
        timeseries : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mods: List[torch.Tensor] = []
        if image is not None and self.use_vision:
            mods.append(self.vision_enc(image))
        if waveform is not None and self.use_audio:
            mods.append(self.audio_enc(waveform))
        if token_ids is not None:
            mods.append(self.text_proj(self.text_embed(token_ids).mean(dim=1)))
        if proprio is not None and self.use_proprio:
            mods.append(self.proprio_enc(proprio))
        if timeseries is not None and self.use_timeseries:
            mods.append(self.ts_enc(timeseries))
        if not mods:
            return torch.zeros(1, self.fusion.latent_dim, device=self.device)
        if len(mods) == 1:
            return mods[0]
        return self.fusion(mods)


# =============================================================================
# SECTION 5 — UPGRADED LANGUAGE MODULE
# GPT-style causal LM with RoPE + SSCStabilizer + InterfaceAttention
# =============================================================================

class RoPECausalTransformer(nn.Module):
    """
    [v2.0 UPGRADED] GPT-style causal language model with:
    - RoPE positional embedding
    - Pre-norm residual architecture
    - SSCStabilizer per layer
    - InterfaceAttention-modified self-attention
    - CSOCComputeController for adaptive depth
    """

    def __init__(
        self,
        vocab_size  : int,
        d_model     : int,
        n_heads     : int,
        n_layers    : int,
        max_seq     : int   = 2048,
        device      : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.n_layers  = n_layers
        self.device    = device
        self.head_dim  = d_model // n_heads

        self.embed  = nn.Embedding(vocab_size, d_model)
        self.rope   = RotaryEmbedding(self.head_dim, max_seq)

        # Build layers with SSC stabilizers and interface attention
        self.layers    : nn.ModuleList = nn.ModuleList()
        self.ssc_stabs : nn.ModuleList = nn.ModuleList()
        self.iface_attns: nn.ModuleList = nn.ModuleList()

        for _ in range(n_layers):
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.layers.append(enc_layer)
            self.ssc_stabs.append(SSCStabilizer(d_model))
            self.iface_attns.append(InterfaceAttention(d_model))

        # CSOC compute controller [v2.0]
        self.csoc = CSOCComputeController(
            d_model=d_model, min_layers=2, max_layers=n_layers
        )

        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.to(device)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids : (B, L)
        Returns:
            (B, d_model) mean-pooled encoding
        """
        x = self.embed(token_ids)           # (B, L, D)

        # CSOC: decide how many layers to run
        _, n_active = self.csoc(x)

        for i in range(n_active):
            layer = self.layers[i]
            x = layer(x)
            x = self.ssc_stabs[i](x)       # SSC stabilization per layer

        x = self.norm(x)
        return x.mean(dim=1)               # (B, D)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode(token_ids)


class LanguageModule(nn.Module):
    """
    AGI ONE v2.0 Language Interface.

    Backend "builtin": RoPECausalTransformer (upgraded GPT-style)
    Backend "huggingface:*": HuggingFace AutoModel
    """

    def __init__(self, cfg: AGIConfig) -> None:
        super().__init__()
        D = cfg.latent_dim
        self.device     = cfg.device
        self.latent_dim = D

        backend = cfg.language_backend.lower()

        if backend == "builtin" or not HAS_HF:
            self.backbone = RoPECausalTransformer(
                vocab_size = cfg.vocab_size,
                d_model    = D,
                n_heads    = cfg.n_transformer_heads,
                n_layers   = cfg.n_transformer_layers,
                device     = cfg.device,
            )
            self.lang_dim  = D
            self._backend  = "builtin"
            logger.info("LanguageModule: RoPE-GPT built-in backbone [v2.0]")

        elif backend.startswith("huggingface:"):
            model_id = backend.split("huggingface:", 1)[1] or cfg.language_model_id
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                hf_model       = AutoModel.from_pretrained(model_id)
                self.backbone  = hf_model.to(cfg.device)
                self.lang_dim  = cfg.language_dim
                self._backend  = "huggingface"
                logger.info(f"LanguageModule: HuggingFace {model_id} [v2.0]")
            except Exception as e:
                logger.warning(f"HuggingFace load failed ({e}) → fallback builtin")
                self.backbone = RoPECausalTransformer(
                    cfg.vocab_size, D, cfg.n_transformer_heads,
                    cfg.n_transformer_layers, device=cfg.device,
                )
                self.lang_dim = D
                self._backend = "builtin"
        else:
            raise ValueError(f"Unknown language_backend: {backend}")

        self.lang_to_latent = nn.Linear(self.lang_dim, D)
        self.grounding_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=cfg.n_transformer_heads, batch_first=True,
        )
        self.lm_head = nn.Linear(D, cfg.vocab_size)

        self.to(cfg.device)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._backend == "builtin":
            return self.backbone.encode(token_ids)
        with torch.no_grad():
            out = self.backbone(token_ids)
            return self.lang_to_latent(out.last_hidden_state.mean(dim=1))

    def ground(self, lang: torch.Tensor, percept: torch.Tensor) -> torch.Tensor:
        q = lang.unsqueeze(1)
        k = percept.unsqueeze(1)
        g, _ = self.grounding_attn(q, k, k)
        return g.squeeze(1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.lm_head(latent)

    def forward(
        self,
        token_ids         : torch.Tensor,
        perception_latent : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lang = self.encode(token_ids)
        if perception_latent is not None:
            lang = self.ground(lang, perception_latent)
        return lang, self.decode(lang)


# =============================================================================
# SECTION 6 — PSYCHE EXECUTIVE LAYER [v2.0 NEW]
# Id → Goal Generator | Ego → Planner | Superego → Safety Constraint
# =============================================================================

class PsycheExecutiveLayer(nn.Module):
    """
    [v2.0 NEW] Psyche as Executive Layer above Transformer.

    Reinterprets PSY ONE BRIDGE (Id/Ego/Superego) as a three-tier
    executive control system:

        Id       → Goal Generator  (what drives does the system have?)
        Ego      → Planner         (what action best satisfies drives?)
        Superego → Safety Filter   (does the action violate constraints?)

    Pipeline:
        Workspace state (D)
            ↓
        [Id]  drive_proposal (action_dim)
        [Ego]  planned_action = free_energy_minimize(drive, constraint)
        [Superego] safety_score ∈ [0,1]: block unsafe actions
            ↓
        executive_action (action_dim)  + safety_gate (scalar)

    This module wraps PsycheTriad if available, otherwise provides
    a differentiable standalone executive controller.
    """

    def __init__(
        self,
        latent_dim  : int,
        action_dim  : int,
        cfg         : AGIConfig,
        device      : torch.device,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.device     = device

        # Goal Generator (Id analog): workspace → drive distribution
        self.goal_generator = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Planner (Ego analog): DEQ-like fixed-point via iterative refinement
        self.planner_net = nn.Sequential(
            nn.Linear(action_dim * 2, 256), nn.GELU(),
            nn.Linear(256, action_dim),
        )

        # Safety Constraint (Superego analog): score ∈ [0, 1]
        # 0 = unsafe (block), 1 = safe (allow)
        self.safety_net = nn.Sequential(
            nn.Linear(action_dim, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

        # Normative policy (learnable "ethical prior")
        self.normative_policy = nn.Parameter(
            torch.ones(action_dim) / action_dim
        )

        # [v3.5 NEW] PSY ONE BRIDGE (psy_one_bridge_diff) integration removed
        # — not one of the seven surrogate modules AGI ONE now calls directly.
        # psy_triad/gumbel_sched stay permanently None; forward() below falls
        # straight through its existing None-branch, so executive_action is
        # produced purely by planner_net/safety_net/normative_policy, exactly
        # as it already was whenever HAS_PSY_BRIDGE was False.
        self.psy_triad   : Optional[Any] = None
        self.gumbel_sched: Optional[Any] = None

        self.to(device)

    def forward(
        self,
        workspace_state   : torch.Tensor,   # (D,)
        n_planner_iters   : int = 5,
    ) -> Dict[str, Any]:
        """
        Returns:
            executive_action : (action_dim,) differentiable action
            safety_gate      : scalar ∈ [0,1]
            psyche_state     : PsycheTriadState or None
            total_loss       : PSY bridge loss or None
        """
        # ── Id: Goal Generation ─────────────────────────────────────────────
        if workspace_state.dim() == 1:
            ws = workspace_state.unsqueeze(0)   # (1, D)
        else:
            ws = workspace_state

        drive = self.goal_generator(ws).squeeze(0)   # (action_dim,)

        # [v3.5 NEW] PSY ONE BRIDGE forward path removed along with the
        # __init__ wiring above — psy_triad is always None now, so
        # psyche_state/psy_loss are always None (same as the pre-existing
        # "library not found" behaviour), and `drive` flows straight from
        # the goal_generator into the Ego planning step below unchanged.
        psyche_state = None
        psy_loss     = None

        # ── Ego: Iterative Planning (DEQ-style fixed-point) ──────────────────
        norm_pol = F.softmax(self.normative_policy, dim=-1)
        plan     = drive.clone()
        for _ in range(n_planner_iters):
            combined = torch.cat([plan, norm_pol], dim=-1)   # (2*action_dim,)
            delta    = self.planner_net(combined)
            plan     = F.softmax(plan + 0.1 * delta, dim=-1)

        # ── Superego: Safety Gate ─────────────────────────────────────────────
        safety_score = self.safety_net(plan).squeeze(-1)   # scalar

        # Gate: blend action with normative policy based on safety
        executive_action = safety_score * plan + (1 - safety_score) * norm_pol

        return {
            "executive_action": executive_action,
            "drive"           : drive,
            "plan"            : plan,
            "safety_score"    : safety_score,
            "psyche_state"    : psyche_state,
            "psy_loss"        : psy_loss,
        }


# =============================================================================
# SECTION 7 — DREAMERV3-STYLE WORLD MODEL [v2.0 NEW]
# Hafner et al. 2023: symlog, two-hot reward, free-bits KL,
# categorical straight-through latents
# =============================================================================

def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symlog: sign(x) · log(|x| + 1)."""
    return x.sign() * (x.abs() + 1.0).log()

def symexp(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symexp: inverse of symlog."""
    return x.sign() * (x.abs().exp() - 1.0)

def two_hot_encode(x: torch.Tensor, n_bins: int = 255,
                    lo: float = -20.0, hi: float = 20.0) -> torch.Tensor:
    """
    DreamerV3 two-hot encoding for reward.
    Projects scalar reward onto two adjacent bins with linear interpolation.
    """
    bins  = torch.linspace(lo, hi, n_bins, device=x.device)
    x_sym = symlog(x).clamp(lo, hi)
    idx   = torch.bucketize(x_sym, bins) - 1
    idx   = idx.clamp(0, n_bins - 2)

    lo_val = bins[idx]
    hi_val = bins[idx + 1]
    w_hi   = ((x_sym - lo_val) / (hi_val - lo_val + 1e-8)).clamp(0, 1)
    w_lo   = 1.0 - w_hi

    target = torch.zeros(*x.shape, n_bins, device=x.device)
    target.scatter_(-1, idx.unsqueeze(-1), w_lo.unsqueeze(-1))
    target.scatter_(-1, (idx + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
    return target


class DreamerV3WorldModel(nn.Module):
    """
    [v2.0 NEW] DreamerV3-style Recurrent State Space Model.

    Key differences from RSSM v1.0:
    [1] Categorical straight-through latents (32 classes × 32 variables)
        instead of Gaussian — avoids posterior collapse
    [2] symlog preprocessing on all inputs and reconstruction targets
    [3] Two-hot encoding for reward (handles wide reward distributions)
    [4] Free-bits KL: KL = max(free_bits, KL_per_variable)
        prevents first few training steps from collapsing
    [5] KL balancing: 80% from posterior, 20% from prior (DreamerV3 default)

    References:
        Hafner et al. "Mastering Diverse Domains with World Models" (2023)
        https://arxiv.org/abs/2301.04104
    """

    def __init__(
        self,
        obs_dim        : int,
        action_dim     : int,
        stoch_size     : int   = 32,   # number of categorical variables
        stoch_classes  : int   = 32,   # classes per variable
        det_size       : int   = 512,
        reward_bins    : int   = 255,
        free_bits      : float = 1.0,
        kl_balance     : float = 0.8,
        device         : torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.obs_dim       = obs_dim
        self.action_dim    = action_dim
        self.stoch_size    = stoch_size
        self.stoch_classes = stoch_classes
        self.det_size      = det_size
        self.reward_bins   = reward_bins
        self.free_bits     = free_bits
        self.kl_balance    = kl_balance
        self.device        = device

        self.latent_dim    = stoch_size * stoch_classes

        # ── Sequence model: GRU (deterministic state) ────────────────────────
        self.gru = nn.GRUCell(
            input_size  = self.latent_dim + action_dim,
            hidden_size = det_size,
        )

        # ── Dynamics predictor: prior p(z_t | h_t) ───────────────────────────
        self.prior_net = nn.Sequential(
            nn.Linear(det_size, 512), nn.ELU(),
            nn.Linear(512, stoch_size * stoch_classes),
        )

        # ── Representation model: posterior q(z_t | h_t, o_t) ───────────────
        self.posterior_net = nn.Sequential(
            nn.Linear(det_size + obs_dim, 512), nn.ELU(),
            nn.Linear(512, stoch_size * stoch_classes),
        )

        # ── Decoder: observation reconstruction ─────────────────────────────
        self.obs_decoder = nn.Sequential(
            nn.Linear(det_size + self.latent_dim, 512), nn.ELU(),
            nn.Linear(512, obs_dim),
        )

        # ── Reward predictor: two-hot output ──────────────────────────────────
        self.reward_net = nn.Sequential(
            nn.Linear(det_size + self.latent_dim, 256), nn.ELU(),
            nn.Linear(256, reward_bins),
        )

        # ── Continue predictor: p(non-terminal) ───────────────────────────────
        self.continue_net = nn.Sequential(
            nn.Linear(det_size + self.latent_dim, 128), nn.ELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

        # ── SSC stabilizer on hidden state ────────────────────────────────────
        self.h_ssc = SSCStabilizer(det_size)

        # ── Initial states ────────────────────────────────────────────────────
        self.register_buffer("h0",  torch.zeros(1, det_size))
        self.register_buffer("z0",  torch.zeros(1, self.latent_dim))

        self.to(device)

    # ── Categorical straight-through ─────────────────────────────────────────
    def _straight_through_sample(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Straight-through estimator for categorical latents.
        Returns (one_hot, soft_probs) — gradients flow through soft_probs.
        """
        B       = logits.shape[0]
        logits_ = logits.view(B, self.stoch_size, self.stoch_classes)
        probs   = F.softmax(logits_, dim=-1)
        indices = probs.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.stoch_classes).float()
        # Straight-through: forward = one_hot, backward = probs
        z       = (one_hot - probs).detach() + probs
        return z.view(B, self.latent_dim), probs

    # ── Prior ─────────────────────────────────────────────────────────────────
    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_prior, prior_logits)."""
        logits = self.prior_net(h)
        z, _   = self._straight_through_sample(logits)
        return z, logits

    # ── Posterior ─────────────────────────────────────────────────────────────
    def posterior(
        self, h: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_post, posterior_logits)."""
        obs_sym = symlog(obs)
        inp     = torch.cat([h, obs_sym], dim=-1)
        logits  = self.posterior_net(inp)
        z, _    = self._straight_through_sample(logits)
        return z, logits

    # ── GRU transition ────────────────────────────────────────────────────────
    def gru_step(
        self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        inp    = torch.cat([z, a], dim=-1)
        h_next = self.gru(inp, h)
        return h_next

    # ── Free-bits KL loss ─────────────────────────────────────────────────────
    def kl_loss(
        self,
        post_logits : torch.Tensor,   # (B, stoch_size * stoch_classes)
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        DreamerV3 free-bits KL with KL balancing.

        KL = kl_balance * KL(post||sg(prior)) + (1-kl_balance) * KL(sg(post)||prior)
        Free-bits: clamp KL per variable at free_bits minimum.
        """
        B = post_logits.shape[0]
        post  = post_logits.view(B, self.stoch_size, self.stoch_classes)
        prior = prior_logits.view(B, self.stoch_size, self.stoch_classes)

        post_probs  = F.softmax(post,  dim=-1).clamp(1e-8)
        prior_probs = F.softmax(prior, dim=-1).clamp(1e-8)

        # KL(post || prior)
        kl_pp = (post_probs * (post_probs.log() - prior_probs.log())).sum(-1)  # (B, S)
        # KL(post_sg || prior)
        kl_sp = ((post_probs.detach()) *
                 (post_probs.detach().log() - prior_probs.log())).sum(-1)

        # Free bits: max(free_bits, kl_per_variable)
        kl_pp = kl_pp.clamp(min=self.free_bits)
        kl_sp = kl_sp.clamp(min=self.free_bits)

        loss = self.kl_balance * kl_pp + (1 - self.kl_balance) * kl_sp
        return loss.mean()

    # ── Reward loss ───────────────────────────────────────────────────────────
    def reward_loss(
        self, feat: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        logits = self.reward_net(feat)
        target = two_hot_encode(
            reward, self.reward_bins, device=reward.device
        )
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    # ── Imagine trajectory ────────────────────────────────────────────────────
    def imagine(
        self,
        h0        : torch.Tensor,   # (1, det_size)
        z0        : torch.Tensor,   # (1, latent_dim)
        action_seq: torch.Tensor,   # (T, action_dim)
    ) -> Dict[str, torch.Tensor]:
        T     = action_seq.shape[0]
        h, z  = h0, z0
        h_seq, z_seq, r_seq, cont_seq = [], [], [], []

        for t in range(T):
            a     = action_seq[t].unsqueeze(0)
            h     = self.gru_step(h, z, a)
            z, _  = self.prior(h)
            feat  = torch.cat([h, z], dim=-1)
            r     = self.reward_net(feat)
            cont  = self.continue_net(feat)
            h_seq.append(h); z_seq.append(z)
            r_seq.append(r); cont_seq.append(cont)

        return {
            "h_seq"     : torch.cat(h_seq,    dim=0),
            "z_seq"     : torch.cat(z_seq,    dim=0),
            "reward_seq": torch.cat(r_seq,    dim=0),
            "cont_seq"  : torch.cat(cont_seq, dim=0),
        }

    def forward(
        self,
        h      : torch.Tensor,
        z      : torch.Tensor,
        action : torch.Tensor,
        obs    : Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h_next       = self.gru_step(h, z, action)

        # Stabilize hidden state
        h_stable     = self.h_ssc(h_next.unsqueeze(1)).squeeze(1)

        prior_logits                  = self.prior_net(h_stable)
        z_prior, _                    = self._straight_through_sample(prior_logits)

        if obs is not None:
            post_logits               = self.posterior_net(
                torch.cat([h_stable, symlog(obs)], dim=-1)
            )
            z_post, _                 = self._straight_through_sample(post_logits)
            z_next                    = z_post
        else:
            post_logits               = prior_logits
            z_next                    = z_prior

        feat         = torch.cat([h_stable, z_next], dim=-1)
        obs_pred     = symexp(self.obs_decoder(feat))
        reward_logits= self.reward_net(feat)
        cont_pred    = self.continue_net(feat)

        return {
            "h_next"        : h_stable,
            "z_next"        : z_next,
            "obs_pred"      : obs_pred,
            "reward_logits" : reward_logits,
            "cont_pred"     : cont_pred,
            "prior_logits"  : prior_logits,
            "post_logits"   : post_logits,
            "feat"          : feat,
        }


# =============================================================================
# SECTION 8 — MPPI PLANNER [v2.0 REPLACES CEM]
# Williams et al. 2017 — GPU-parallel importance-weighted planning
# =============================================================================

class MPPIPlanner(nn.Module):
    """
    [v2.0 NEW] Model Predictive Path Integral (MPPI) Planner.

    Replaces CEM. Key advantages:
    - All N trajectories evaluated in parallel (no sequential elite selection)
    - Importance-weighted update: ALL samples contribute, not just top-k
    - Smoother, more stable optimization landscape
    - Better exploration via temperature-controlled weighting

    Algorithm:
        1. Sample N perturbations ε ~ N(0, σ²I) around nominal action sequence
        2. Roll out each in world model → cost = -sum(discount^t * reward_t)
        3. Compute importance weights: w_i = exp(-(cost_i - min_cost)/λ)
        4. Update: μ_new = Σ(w_i * (μ + ε_i)) / Σw_i
        5. Execute μ[0]; warm-start next step
    """

    def __init__(
        self,
        action_dim      : int,
        horizon         : int,
        n_samples       : int,
        temperature     : float,
        noise_sigma     : float,
        device          : torch.device,
    ) -> None:
        super().__init__()
        self.action_dim  = action_dim
        self.horizon     = horizon
        self.n_samples   = n_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.device      = device

        # Nominal action sequence (warm-started between steps)
        self.register_buffer(
            "mu", torch.zeros(horizon, action_dim)
        )

        # Value baseline for terminal bootstrap
        self.value_net = nn.Sequential(
            nn.Linear(512 + 32 * 32, 256), nn.ELU(),
            nn.Linear(256, 1),
        )

        self.goal_encoder = nn.Linear(action_dim, action_dim)
        self.to(device)

    def plan(
        self,
        world_model   : DreamerV3WorldModel,
        h             : torch.Tensor,         # (1, det_size)
        z             : torch.Tensor,         # (1, latent_dim)
        goal          : Optional[torch.Tensor] = None,
        psyche_bias   : Optional[torch.Tensor] = None,
        discount      : float = 0.99,
        n_iters       : int   = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MPPI planning loop.

        Returns:
            best_action   : (action_dim,) first action to execute
            best_sequence : (T, action_dim) full sequence
        """
        T  = self.horizon
        mu = self.mu.clone()

        # Add psyche drive bias to nominal sequence
        if psyche_bias is not None:
            bias = psyche_bias.to(self.device)
            if bias.shape[-1] != self.action_dim:
                bias = F.adaptive_avg_pool1d(
                    bias.unsqueeze(0).unsqueeze(0), self.action_dim
                ).squeeze()
            mu += 0.1 * bias.unsqueeze(0).expand(T, -1)

        for _ in range(n_iters):
            # Sample perturbations: (N, T, action_dim)
            eps     = torch.randn(
                self.n_samples, T, self.action_dim, device=self.device
            ) * self.noise_sigma
            samples = mu.unsqueeze(0) + eps   # (N, T, A)

            # Evaluate trajectories: compute costs
            costs = torch.zeros(self.n_samples, device=self.device)
            for i in range(self.n_samples):
                traj = world_model.imagine(
                    h0         = h,
                    z0         = z,
                    action_seq = samples[i],
                )
                # Cost = negative discounted reward
                r_logits  = traj["reward_seq"]        # (T, reward_bins)
                r_bins    = torch.linspace(-20, 20, world_model.reward_bins,
                                           device=self.device)
                r_pred    = (F.softmax(r_logits, dim=-1) * r_bins).sum(-1)
                r_pred    = symexp(r_pred)
                discounts = torch.tensor(
                    [discount ** t for t in range(T)], device=self.device
                )
                # Continuation weighting
                cont      = traj["cont_seq"].squeeze(-1)
                cum_cont  = cont.cumprod(dim=0)
                returns   = (r_pred * discounts * cum_cont).sum()

                # Goal bonus
                if goal is not None:
                    final_z = traj["z_seq"][-1:]
                    goal_enc = self.goal_encoder(goal)
                    bonus    = F.cosine_similarity(
                        final_z.mean(dim=-1, keepdim=True),
                        goal_enc.unsqueeze(-1),
                        dim=0,
                    ).mean()
                    returns += bonus

                costs[i] = -returns   # negate: lower cost = better

            # Importance weights: w_i = exp(-(cost_i - min_cost) / λ)
            beta    = costs.min()
            weights = torch.exp(-(costs - beta) / self.temperature)
            weights = weights / (weights.sum() + 1e-8)     # normalize

            # Weighted update of nominal sequence
            mu      = (weights.view(-1, 1, 1) * samples).sum(dim=0)

        # Warm-start: shift sequence left, repeat last action
        self.mu[:-1] = mu[1:].detach()
        self.mu[-1]  = mu[-1].detach()

        return mu[0], mu

    def reset(self) -> None:
        """Reset warm-started nominal sequence (new episode)."""
        self.mu.zero_()

    def forward(
        self,
        world_model: DreamerV3WorldModel,
        h          : torch.Tensor,
        z          : torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        best_action, _ = self.plan(world_model, h, z, **kwargs)
        return best_action


# =============================================================================
# SECTION 9 — MEMORY MODULES (preserved + upgraded with SSC)
# =============================================================================

class WorkingMemoryModule(nn.Module):
    """
    Short-term working memory with SSCStabilizer [v2.0 upgrade].
    N attention-gated slots.
    """

    def __init__(self, n_slots: int, latent_dim: int,
                 n_heads: int, device: torch.device) -> None:
        super().__init__()
        self.n_slots    = n_slots
        self.latent_dim = latent_dim
        self.device     = device

        self.register_buffer("slots", torch.zeros(n_slots, latent_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.slot_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.gate         = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid(),
        )
        self.read_query   = nn.Linear(latent_dim, latent_dim)
        self.read_key     = nn.Linear(latent_dim, latent_dim)
        self.read_value   = nn.Linear(latent_dim, latent_dim)
        self.ssc_stab     = SSCStabilizer(latent_dim)  # [v2.0]
        self.to(device)

    def write(self, c: torch.Tensor) -> None:
        if c.dim() == 1: c = c.unsqueeze(0)
        sim     = F.cosine_similarity(c, self.slots, dim=-1)
        idx     = int(sim.argmin().item())
        old     = self.slots[idx].unsqueeze(0)
        g       = self.gate(torch.cat([old, c], dim=-1))
        self.slots[idx] = (g * c + (1-g) * old).squeeze(0)

    def read(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() == 1: q = q.unsqueeze(0)
        Q  = self.read_query(q)
        K  = self.read_key(self.slots)
        V  = self.read_value(self.slots)
        w  = F.softmax((Q @ K.T) / math.sqrt(self.latent_dim), dim=-1)
        return (w @ V).squeeze(0)

    def process(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.dim() == 1: inp = inp.unsqueeze(0)
        seq = torch.cat([self.slots.unsqueeze(0), inp.unsqueeze(0)], dim=1)
        out = self.slot_transformer(seq)
        out = self.ssc_stab(out)   # [v2.0]
        return out[0, -1, :]

    def reset(self) -> None: self.slots.zero_()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        ctx = self.process(inp)
        self.write(inp)
        return ctx


class EpisodicMemoryModule(nn.Module):
    """Long-term DND episodic memory — preserved from v1.0."""

    def __init__(self, capacity: int, latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.capacity   = capacity
        self.latent_dim = latent_dim
        self.device     = device

        self.register_buffer("keys",         torch.zeros(capacity, latent_dim))
        self.register_buffer("values",       torch.zeros(capacity, latent_dim))
        self.register_buffer("ages",         torch.zeros(capacity))
        self.register_buffer("access_count", torch.zeros(capacity))
        self._ptr  = 0
        self._size = 0

        self.key_encoder       = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh())
        self.consolidation_proj= nn.Linear(latent_dim, latent_dim)
        self.to(device)

    def write(self, k: torch.Tensor, v: torch.Tensor) -> None:
        k = self.key_encoder(k.detach()).squeeze(0)
        v = v.detach().squeeze(0)
        if self._size < self.capacity:
            idx = self._ptr
            self._size += 1
        else:
            score = self.ages * (1.0 / (self.access_count + 1.0))
            idx   = int(score.argmax().item())
        self.keys[idx] = k; self.values[idx] = v
        self.ages[idx] = 0.0; self.access_count[idx] = 0.0
        self._ptr = (self._ptr + 1) % self.capacity
        self.ages[:self._size] += 1.0

    def retrieve(self, q: torch.Tensor, top_k: int = 5,
                 temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._size == 0:
            return torch.zeros(self.latent_dim, device=self.device), \
                   torch.zeros(1, device=self.device)
        q2   = self.key_encoder(q).squeeze(0)
        sim  = F.cosine_similarity(q2.unsqueeze(0), self.keys[:self._size], dim=-1)
        k    = min(top_k, self._size)
        ts, ti = sim.topk(k)
        self.access_count[ti] += 1.0
        w    = F.softmax(ts / temperature, dim=0)
        return (w.unsqueeze(-1) * self.values[ti]).sum(0), w

    def forward(self, q: torch.Tensor,
                write_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        if write_v is not None: self.write(q, write_v)
        r, _ = self.retrieve(q)
        return r


# =============================================================================
# SECTION 10 — GLOBAL WORKSPACE MODULE (preserved + upgraded)
# =============================================================================

class GlobalWorkspaceModule(nn.Module):
    """GWT broadcast consciousness — upgraded with SSCStabilizer [v2.0]."""

    def __init__(self, latent_dim: int, n_modules: int, n_heads: int,
                 device: torch.device, temp: float = 0.5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.temp       = temp
        self.device     = device

        self.saliency_net    = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, 1),
        )
        self.broadcast_proj  = nn.Linear(latent_dim, latent_dim)
        self.register_buffer("workspace_state", torch.zeros(latent_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.integrator  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.ssc_stab    = SSCStabilizer(latent_dim)   # [v2.0]
        self.to(device)

    def forward(self, module_activations: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, str]:
        names  = list(module_activations.keys())
        vecs   = torch.stack([module_activations[n].to(self.device) for n in names], 0)
        scores = self.saliency_net(vecs).squeeze(-1)
        w      = F.softmax(scores / self.temp, dim=0)
        bc     = self.broadcast_proj((w.unsqueeze(-1) * vecs).sum(0))
        seq    = torch.stack([self.workspace_state.unsqueeze(0),
                              bc.unsqueeze(0)], dim=1)
        out    = self.integrator(seq)
        out    = self.ssc_stab(out)   # [v2.0]
        new_st = out[0, -1, :]
        self.workspace_state = new_st.detach()
        winner = names[int(w.argmax().item())]
        return new_st, winner


# =============================================================================
# SECTION 11 — META-COGNITION (preserved + upgraded with CSOC)
# =============================================================================

class MetaCognitionModule(nn.Module):
    """Self-model with CSOC-driven adaptive introspection [v2.0]."""

    def __init__(self, latent_dim: int, n_strategies: int = 8,
                 device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.device     = device

        self.strategy_net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, n_strategies),
        )
        self.unc_net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 2), nn.Softplus(),
        )
        self.load_net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.anomaly_enc = nn.Linear(latent_dim, latent_dim // 2)
        self.anomaly_dec = nn.Linear(latent_dim // 2, latent_dim)

        # CSOC compute controller for meta-cognition depth [v2.0]
        self.csoc = CSOCComputeController(latent_dim, min_layers=1, max_layers=8)

        self.register_buffer("history", torch.zeros(100, latent_dim))
        self._ptr = 0
        self.to(device)

    def _update(self, ws: torch.Tensor) -> None:
        self.history[self._ptr % 100] = ws.detach()
        self._ptr += 1

    def introspect(self, q: torch.Tensor) -> torch.Tensor:
        n = min(self._ptr, 100)
        if n == 0: return torch.zeros(self.latent_dim, device=self.device)
        h = self.history[:n]
        w = F.softmax(F.cosine_similarity(q.unsqueeze(0), h, dim=-1) / 0.1, dim=0)
        return (w.unsqueeze(-1) * h).sum(0)

    def forward(self, ws: torch.Tensor, ocd: bool = False) -> Dict[str, Any]:
        self._update(ws)
        unc_vals = self.unc_net(ws)
        load     = float(self.load_net(ws).item())
        recon    = self.anomaly_dec(self.anomaly_enc(ws))
        anomaly  = float(F.mse_loss(recon, ws).item())
        strat    = int(self.strategy_net(ws).argmax().item())
        _, n_lay = self.csoc(ws.unsqueeze(0).unsqueeze(0))

        return {
            "strategy"       : strat,
            "epistemic_unc"  : float(unc_vals[0].item()),
            "aleatoric_unc"  : float(unc_vals[1].item()),
            "cognitive_load" : load,
            "anomaly_score"  : anomaly,
            "ocd_alert"      : ocd,
            "csoc_n_layers"  : n_lay,
        }


# =============================================================================
# SECTION 12 — MULTI-SCALE INTEGRATOR (23 modules — 3 new math modules)
# =============================================================================

class MultiScaleIntegrator(nn.Module):
    """
    Routes ONE Ecosystem outputs (23 modules) into AGI latent space.
    [v2.0] adds BSD, GRH, HODGE math reasoning inputs.
    """

    def __init__(self, latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.device     = device

        self.proj = nn.ModuleDict({
            "mental"   : nn.Linear(512, latent_dim),
            "fold"     : nn.Linear(256, latent_dim),
            "evolution": nn.Linear(256, latent_dim),
            "physics"  : nn.Linear(256, latent_dim),
            "psyche"   : nn.Linear(latent_dim, latent_dim),
            "math_bsd" : nn.Linear(64,  latent_dim),   # [v2.0 NEW]
            "math_grh" : nn.Linear(64,  latent_dim),   # [v2.0 NEW]
            "math_hodge": nn.Linear(64, latent_dim),   # [v2.0 NEW]
        })
        self.scale_attn = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1),
        )
        self.to(device)

    def integrate(self, scale_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = []
        for name, tensor in scale_outputs.items():
            if name not in self.proj: continue
            t = tensor.to(self.device).float()
            if t.dim() > 1: t = t.mean(dim=0)
            t = t.unsqueeze(0)
            p = self.proj[name]
            if t.shape[-1] != p.in_features:
                t = F.adaptive_avg_pool1d(
                    t.unsqueeze(0), p.in_features
                ).squeeze(0)
            projected.append(p(t).squeeze(0))

        if not projected:
            return torch.zeros(self.latent_dim, device=self.device)

        stacked = torch.stack(projected, 0)
        w       = F.softmax(self.scale_attn(stacked), dim=0)
        return (w * stacked).sum(0)

    def forward(self, scale_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.integrate(scale_outputs)


# =============================================================================
# SECTION 13 — LOSS BALANCER (Kendall et al. 2018)
# Uncertainty-weighted multi-task loss
# =============================================================================

class LossBalancer(nn.Module):
    """
    [v2.0 NEW] Uncertainty-weighted multi-task loss (Kendall et al. 2018).

    Learns a log-variance parameter σ_i per task such that:
        L_total = Σ_i (1/2σ_i²) · L_i + log(σ_i)

    This automatically balances the learning signals from:
    - World model loss (KL + reconstruction)
    - Policy loss (actor + entropy)
    - Value loss (critic)
    - PSY Bridge loss (Free Energy)
    - Language modelling loss
    - Perception reconstruction loss

    Avoids manual coefficient tuning and handles different loss scales.
    """

    TASK_NAMES = [
        "world_kl", "world_recon", "reward", "continue",
        "actor", "value", "entropy", "psy",
        "language", "perception",
    ]

    def __init__(self) -> None:
        super().__init__()
        # log_sigma² per task — learnable
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in self.TASK_NAMES
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            losses : dict of {task_name: scalar_loss}
        Returns:
            (total_loss, weight_dict)
        """
        total    = torch.tensor(0.0, requires_grad=True)
        weights  = {}

        for name, loss in losses.items():
            if name not in self.log_vars or loss is None:
                continue
            lv     = self.log_vars[name]
            # L_i / (2 * exp(log_var)) + 0.5 * log_var
            weight = torch.exp(-lv)
            total  = total + 0.5 * weight * loss + 0.5 * lv
            weights[name] = float(weight.detach().item())

        return total, weights


# =============================================================================
# SECTION 14 — FULL PPO IMPLEMENTATION [v2.0]
# Clipped surrogate + GAE(λ) + entropy bonus
# =============================================================================

@dataclass
class PPOBuffer:
    """Rolling buffer for PPO experience collection."""
    obs         : List[torch.Tensor]  = field(default_factory=list)
    actions     : List[torch.Tensor]  = field(default_factory=list)
    rewards     : List[float]         = field(default_factory=list)
    values      : List[float]         = field(default_factory=list)
    log_probs   : List[torch.Tensor]  = field(default_factory=list)
    dones       : List[bool]          = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear(); self.actions.clear(); self.rewards.clear()
        self.values.clear(); self.log_probs.clear(); self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    """
    [v2.0 FULL PPO] Proximal Policy Optimization with Generalized Advantage
    Estimation (GAE-λ).

    Components:
    - Actor: policy π(a|s) — outputs action distribution
    - Critic: V(s) — value function baseline
    - Clipped surrogate objective
    - GAE(λ) advantage estimation
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        actor_net   : nn.Module,
        critic_net  : nn.Module,
        action_dim  : int,
        cfg         : AGIConfig,
        device      : torch.device,
    ) -> None:
        self.actor_net  = actor_net
        self.critic_net = critic_net
        self.action_dim = action_dim
        self.cfg        = cfg
        self.device     = device

        params = list(actor_net.parameters()) + list(critic_net.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.buffer = PPOBuffer()

    def compute_gae(
        self,
        rewards : List[float],
        values  : List[float],
        dones   : List[bool],
        gamma   : float,
        gae_lam : float,
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE(λ) advantages and returns."""
        n          = len(rewards)
        advantages = [0.0] * n
        returns    = [0.0] * n
        gae        = 0.0
        next_val   = 0.0

        for t in reversed(range(n)):
            mask    = 0.0 if dones[t] else 1.0
            delta   = rewards[t] + gamma * next_val * mask - values[t]
            gae     = delta + gamma * gae_lam * mask * gae
            advantages[t] = gae
            returns[t]    = advantages[t] + values[t]
            next_val = values[t]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Run PPO update on buffered experience."""
        if len(self.buffer) == 0:
            return {}

        # Compute GAE
        adv, rets = self.compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones,
            self.cfg.ppo_gamma, self.cfg.ppo_gae_lambda,
        )

        # Convert to tensors
        obs_t      = torch.stack(self.buffer.obs).to(self.device)
        acts_t     = torch.stack(self.buffer.actions).to(self.device)
        old_lps    = torch.stack(self.buffer.log_probs).to(self.device)
        adv_t      = torch.tensor(adv, device=self.device)
        rets_t     = torch.tensor(rets, device=self.device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        for epoch in range(self.cfg.ppo_epochs):
            # Actor: compute new log probs + entropy
            action_logits = self.actor_net(obs_t)
            dist          = torch.distributions.Categorical(
                logits=action_logits
            )
            act_idx       = acts_t.argmax(dim=-1) if acts_t.dim() > 1 \
                            else acts_t.long()
            new_lps       = dist.log_prob(act_idx)
            entropy       = dist.entropy().mean()

            # Clipped surrogate
            ratio       = (new_lps - old_lps).exp()
            surr1       = ratio * adv_t
            surr2       = ratio.clamp(
                1 - self.cfg.ppo_clip_eps, 1 + self.cfg.ppo_clip_eps
            ) * adv_t
            actor_loss  = -torch.min(surr1, surr2).mean()

            # Critic
            val_pred    = self.critic_net(obs_t).squeeze(-1)
            critic_loss = F.mse_loss(val_pred, rets_t)

            # Total
            loss = (actor_loss
                    + self.cfg.value_loss_coef * critic_loss
                    - self.cfg.entropy_coef * entropy)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_net.parameters()) +
                list(self.critic_net.parameters()),
                self.cfg.grad_clip_norm,
            )
            self.optimizer.step()

            stats["actor_loss"]  += float(actor_loss.item())
            stats["critic_loss"] += float(critic_loss.item())
            stats["entropy"]     += float(entropy.item())

        n = self.cfg.ppo_epochs
        stats = {k: v / n for k, v in stats.items()}
        self.buffer.clear()
        return stats



# =============================================================================
# SECTION 15 — [v3.5 REMOVED] MATH REASONING LAYER
# Previously integrated BSD ONE / GRH ONE / HODGE ONE by querying
# bsd_one.EllipticCurveLFunction / grh_one.GeneralizedLFunction /
# hodge_one.get_device directly. Removed per Yoon's instruction: AGI ONE's
# math-domain reasoning now comes only from the two already-wired GNO
# surrogates — structural_gno_numbertheory (domain "math_numbertheory") and
# structural_gno_hodge (domain "hodge") — both registered through
# EcosystemOrchestrator/attach_*_to_ecosystem(), not called directly here.
# =============================================================================

# =============================================================================
# SECTION 16 — AGI STATE
# =============================================================================

@dataclass
class AGIState:
    """Full AGI ONE v2.0 cognitive state at time t."""
    step                 : int
    workspace_state      : Optional[torch.Tensor] = None
    winner_module        : str                    = "unknown"
    perception_latent    : Optional[torch.Tensor] = None
    language_latent      : Optional[torch.Tensor] = None
    working_memory_ctx   : Optional[torch.Tensor] = None
    episodic_memory_ctx  : Optional[torch.Tensor] = None
    world_model_state    : Optional[Dict]         = None
    planned_action       : Optional[torch.Tensor] = None
    executive_action     : Optional[torch.Tensor] = None
    safety_score         : Optional[float]        = None
    psyche_state         : Optional[Any]          = None
    meta_cognition       : Optional[Dict]         = None
    one_ecosystem_latent : Optional[torch.Tensor] = None
    math_latent          : Optional[torch.Tensor] = None
    total_loss           : Optional[torch.Tensor] = None
    csoc_n_layers        : Optional[int]          = None
    # ── Psyche Plus  [v3.1 NEW] ───────────────────────────────────────────
    psyche_plus_validity    : Optional[float]    = None
    psyche_plus_axiom_count : Optional[int]      = None
    psyche_plus_codified    : Optional[bool]      = None
    external_consensus      : Optional[Any]       = None   # ExternalConsensusResult
    ecosystem_quality_scores: Optional[Dict[str, float]] = None   # [v3.2 NEW]

    def summary(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step"         : self.step,
            "winner_module": self.winner_module,
        }
        if self.meta_cognition:
            d["meta"] = self.meta_cognition
        if self.safety_score is not None:
            d["safety_score"] = round(self.safety_score, 4)
        if self.csoc_n_layers is not None:
            d["csoc_n_layers"] = self.csoc_n_layers
        if self.psyche_state and hasattr(self.psyche_state, "to_dict"):
            d["psyche"] = self.psyche_state.to_dict()
        if self.psyche_plus_validity is not None:
            d["psyche_plus_validity"] = round(self.psyche_plus_validity, 4)
        if self.psyche_plus_axiom_count is not None:
            d["psyche_plus_axiom_count"] = self.psyche_plus_axiom_count
        if self.external_consensus is not None and hasattr(self.external_consensus, "to_dict"):
            d["external_consensus"] = self.external_consensus.to_dict()
        if self.planned_action is not None:
            d["planned_action_norm"] = float(self.planned_action.norm().item())
        return d


# =============================================================================
# SECTION 17 — AGI ONE v2.0 CORE ENGINE
# =============================================================================

class AGIONE(nn.Module):
    """
    AGI ONE v2.0 — Production-Grade General Intelligence Architecture.

    ═══════════════════════════════════════════════════════════════════
    Developer  : Yoon A Limsuwan / MSPS NETWORK
                 MY SOUL MOVE BY POWER OF HOLY SPIRIT
    License    : MIT
    Version    : 2.0.0
    AI Assistants: Claude (Anthropic), GPT-4o (OpenAI),
                   Gemini (Google DeepMind), DeepSeek (DeepSeek AI)
    ═══════════════════════════════════════════════════════════════════

    Full cognitive architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  PerceptionModule    ← ViT patch encoder + Conformer audio      │
    │  LanguageModule      ← RoPE-GPT + SSCStabilizer + CSOC depth    │
    │  WorkingMemoryModule ← slot attention + SSC                     │
    │  EpisodicMemory      ← DND long-term memory                     │
    │  GlobalWorkspace     ← GWT broadcast + SSC                      │
    │  DreamerV3WorldModel ← categorical latents + symlog + free-bits │
    │  MPPIPlanner         ← importance-weighted trajectory opt        │
    │  MetaCognitionModule ← self-model + CSOC compute control        │
    │  PsycheExecutiveLayer← Id→Goal / Ego→Plan / Superego→Safety     │
    │  PsychePlus          ← speculative-axiom evolution + multi-LLM  │
    │                         consensus (opt-in)            [v3.1]    │
    │  MultiScaleIntegrator← 7 wired GNO surrogate adapters [v3.5]     │
    │                         (fold/hodge/mental/physics_sfno3d/       │
    │                         physics_ngo/math_numbertheory/evo_bv)   │
    │  SSCStabilizer       ← per-layer latent stabilization           │
    │  InterfaceAttention  ← phase-transition attention prior         │
    │  CSOCComputeController ← edge-of-chaos adaptive depth           │
    │  StructuralLangevinDiffusion ← geometry-aware exploration       │
    │  LossBalancer        ← Kendall uncertainty weighting            │
    │  PPOTrainer          ← full clipped PPO + GAE(λ)                │
    │  OpenScienceRegistry ← dataset attribution + provenance         │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: Optional[AGIConfig] = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = AGIConfig()
        self.cfg    = cfg
        self.device = cfg.device
        self._step  = 0

        torch.manual_seed(cfg.seed)

        logger.info(
            f"\n{'='*65}\n"
            f"  AGI ONE v{AGI_ONE_VERSION} — ONE Ecosystem Central Hub\n"
            f"  Developer  : Yoon A Limsuwan / MSPS NETWORK\n"
            f"  MY SOUL MOVE BY POWER OF HOLY SPIRIT\n"
            f"  AI Assistants: Claude (Anthropic), GPT-4o (OpenAI),\n"
            f"                 Gemini (Google DeepMind), DeepSeek\n"
            f"{'='*65}\n"
            f"  Device: {cfg.device}  latent={cfg.latent_dim}  "
            f"action={cfg.action_dim}\n"
            f"{'='*65}"
        )

        D      = cfg.latent_dim
        A      = cfg.action_dim
        device = cfg.device

        # ── [1] Perception ───────────────────────────────────────────────────
        self.perception = PerceptionModule(cfg)

        # ── [2] Language ─────────────────────────────────────────────────────
        if cfg.use_language:
            self.language = LanguageModule(cfg)

        # ── [3] Working Memory ────────────────────────────────────────────────
        self.working_memory = WorkingMemoryModule(
            cfg.memory_slots, D, cfg.n_transformer_heads, device
        )

        # ── [4] Episodic Memory ───────────────────────────────────────────────
        self.episodic_memory = EpisodicMemoryModule(
            cfg.episodic_capacity, D, device
        )

        # ── [5] Global Workspace ─────────────────────────────────────────────
        self.global_workspace = GlobalWorkspaceModule(D, 10, cfg.n_transformer_heads, device)

        # ── [6] DreamerV3 World Model ─────────────────────────────────────────
        stoch_latent = cfg.dreamer_stoch_size * cfg.dreamer_stoch_classes
        self.world_model = DreamerV3WorldModel(
            obs_dim        = D,
            action_dim     = A,
            stoch_size     = cfg.dreamer_stoch_size,
            stoch_classes  = cfg.dreamer_stoch_classes,
            det_size       = cfg.dreamer_det_size,
            reward_bins    = cfg.dreamer_reward_bins,
            free_bits      = cfg.dreamer_free_bits,
            kl_balance     = cfg.dreamer_kl_balance,
            device         = device,
        )

        # ── [7] MPPI Planner ──────────────────────────────────────────────────
        self.mppi = MPPIPlanner(
            action_dim   = A,
            horizon      = cfg.planning_horizon,
            n_samples    = cfg.mppi_n_samples,
            temperature  = cfg.mppi_temperature,
            noise_sigma  = cfg.mppi_noise_sigma,
            device       = device,
        )

        # ── [8] Meta-Cognition ────────────────────────────────────────────────
        self.meta_cognition = MetaCognitionModule(D, 8, device)

        # ── [9] Psyche Executive Layer (Id/Ego/Superego) ─────────────────────
        self.psyche_exec = PsycheExecutiveLayer(D, A, cfg, device)

        # ── [9-B] Psyche Plus: speculative-axiom evolution  [v3.1 NEW] ───────
        self.psyche_plus: Optional[Any] = None
        # [v3.2 NEW] Optional back-reference to an EcosystemOrchestrator.
        # AGIONE does not own/construct one itself (that remains
        # AGITrainerV3's job, so nothing about ownership changes), but if
        # one is attached via `attach_orchestrator()`, Step 7-B can read its
        # `quality_report()` and pass real simulation-quality scores into
        # Psyche Plus. Left None, behaviour is byte-for-byte identical to
        # pre-v3.2: `quality_score=None` everywhere.
        self.orchestrator: Optional[Any] = None
        if cfg.use_psyche_plus and HAS_PSYCHE_PLUS:
            self.psyche_plus = AGIOnePsychePlus(
                cfg=PsychePlusConfig(
                    latent_dim            = D,
                    max_axioms            = cfg.psyche_plus_max_axioms,
                    superego_axiom_weight = cfg.psyche_plus_axiom_weight,
                    codify_threshold      = cfg.psyche_plus_codify_threshold,
                    enable_external_llm   = cfg.psyche_plus_enable_external_llm,
                    consult_every_n_steps = cfg.psyche_plus_consult_every_n_steps,
                    llm_timeout_s         = cfg.psyche_plus_llm_timeout_s,
                    device                = device,
                ),
                device=device,
            )
            logger.info("✓ Psyche Plus speculative-axiom layer attached to AGI ONE core.")

        # ── [10] [v3.5 REMOVED] ONE Ecosystem direct-engine block ────────────
        # Previously instantiated MentalONEEngine / RealFoldONEEngine /
        # EvolutionONEEngine / EpidemicEngine / SuperDNSEngine /
        # StructuralFluctuatingHydro / StandardONEEngine /
        # YangMillsMassGapEngine / RiemannHypothesisEngine directly. None of
        # these were ever called in a forward pass (self.mental_one etc. were
        # read only by the status/quality_report dict below) — removed per
        # Yoon's instruction that AGI ONE call only the seven wired GNO
        # surrogates. Kept as None stubs so quality_report()'s "... is not
        # None" checks further below keep working unchanged (all now
        # unconditionally report False/absent, same as when these libraries
        # were simply not installed).
        self.mental_one      = None
        self.real_fold       = None
        self.evolution_one   = None
        self.epidemic_engine = None
        self.dns_engine      = None
        self.fh_engine       = None
        self.standard_one    = None
        self.yang_mills      = None
        self.rh_engine       = None

        # ── [11] [v3.5 REMOVED] Math Reasoning Layer (BSD / GRH / HODGE) ─────
        # MathReasoningLayer (queried bsd_one/grh_one/hodge_one directly) was
        # removed — see SECTION 15 above. Math-domain reasoning now comes
        # only from the structural_gno_numbertheory and structural_gno_hodge
        # surrogates wired through EcosystemOrchestrator.
        self.math_layer = None

        # ── [12] Multi-Scale Integrator ───────────────────────────────────────
        self.multiscale = MultiScaleIntegrator(D, device)

        # ── [13] Geometry-aware Latent Diffusion ──────────────────────────────
        self.latent_diffusion = StructuralLangevinDiffusion(D, device=device)

        # ── [14] Loss Balancer (Kendall) ──────────────────────────────────────
        self.loss_balancer = LossBalancer()

        # ── [15] Actor / Critic heads for PPO ────────────────────────────────
        self.actor_net = nn.Sequential(
            nn.Linear(D, 512), nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, A),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(D, 512), nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1),
        )

        # ── [16] PPO Trainer ──────────────────────────────────────────────────
        self.ppo = PPOTrainer(
            actor_net  = self.actor_net,
            critic_net = self.critic_net,
            action_dim = A,
            cfg        = cfg,
            device     = device,
        )

        # ── [17] Open Science Registry ────────────────────────────────────────
        self.science_registry = OpenScienceRegistry()

        # ── World model hidden state (persistent) ─────────────────────────────
        self.register_buffer("wm_h", torch.zeros(1, cfg.dreamer_det_size))
        self.register_buffer("wm_z", torch.zeros(1, stoch_latent))

        self.to(device)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"AGI ONE v2.0 total parameters: {total_params:,}")

    # =========================================================================
    # ONE ECOSYSTEM QUERY
    # =========================================================================

    # ── [v3.2 NEW] Orchestrator attachment ───────────────────────────────────

    def attach_orchestrator(self, orchestrator: Any) -> None:
        """
        Attach an EcosystemOrchestrator so Step 7-B can read live
        simulation-quality scores via `orchestrator.quality_report()`.

        Purely additive / optional. AGITrainerV3 already builds its own
        EcosystemOrchestrator and keeps a reference on `self.orchestrator`;
        calling `trainer.model.attach_orchestrator(trainer.orchestrator)`
        once after construction is enough to unify the two. If never
        called, Psyche Plus simply runs with `quality_score=None`, exactly
        as in pre-v3.2 versions.
        """
        self.orchestrator = orchestrator

    def _query_one_ecosystem(
        self,
        perception_latent: Optional[torch.Tensor] = None,
        extra_inputs     : Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if perception_latent is not None:
            outputs["psyche"] = perception_latent
        if extra_inputs is None:
            extra_inputs = {}

        # Math reasoning [v2.0 NEW]
        if self.math_layer is not None:
            ml = self.math_layer()
            if ml is not None:
                outputs["math_bsd"] = ml[:64] if ml.shape[0] >= 64 \
                                      else F.pad(ml, (0, 64 - ml.shape[0]))

        return outputs

    # =========================================================================
    # FORWARD PASS — FULL AGI CYCLE v2.0
    # =========================================================================

    def forward(
        self,
        image       : Optional[torch.Tensor] = None,
        waveform    : Optional[torch.Tensor] = None,
        token_ids   : Optional[torch.Tensor] = None,
        proprio     : Optional[torch.Tensor] = None,
        timeseries  : Optional[torch.Tensor] = None,
        goal        : Optional[torch.Tensor] = None,
        extra_inputs: Optional[Dict]         = None,
        compute_loss: bool                   = False,
        reward      : Optional[torch.Tensor] = None,
    ) -> AGIState:
        """
        AGI ONE v2.0 full cognitive cycle.

        Steps:
        1.  Perception     → fused_perception (ViT + Conformer)
        2.  Language       → language_latent (RoPE-GPT)
        3.  Memory         → working_memory_ctx + episodic_ctx
        4.  ONE Ecosystem  → multiscale_latent (23 modules)
        5.  Math Reasoning → math_latent (BSD/GRH/HODGE)
        6.  Global Workspace (GWT) → workspace_state, winner
        7.  Psyche Executive → Id→goal, Ego→plan, Superego→safety
        7B. Psyche Plus     → speculative-axiom evolution (opt-in LLM consensus)
        8.  DreamerV3 World Model → h, z, predictions
        9.  MPPI Planning  → planned_action (importance-weighted)
        10. Geometry-aware Diffusion → latent exploration
        11. Meta-Cognition → self-model, CSOC depth
        12. Loss computation (if training, Kendall-balanced)
        """
        self._step += 1
        state = AGIState(step=self._step)

        # ── Step 1: Perception ────────────────────────────────────────────────
        perc = self.perception(
            image=image, waveform=waveform, token_ids=token_ids,
            proprio=proprio, timeseries=timeseries,
        )
        state.perception_latent = perc

        # ── Step 2: Language ──────────────────────────────────────────────────
        lang = None
        if hasattr(self, "language") and token_ids is not None:
            lang, _ = self.language(token_ids, perc)
            state.language_latent = lang

        # ── Step 3: Memory ────────────────────────────────────────────────────
        wm_inp = lang if lang is not None else perc
        wm_ctx = self.working_memory(wm_inp)
        ep_ctx = self.episodic_memory(
            perc, write_v=perc if self._step % 5 == 0 else None
        )
        state.working_memory_ctx  = wm_ctx
        state.episodic_memory_ctx = ep_ctx

        # ── Step 4: ONE Ecosystem ─────────────────────────────────────────────
        scale_out = self._query_one_ecosystem(perc, extra_inputs)
        ms_latent = self.multiscale(scale_out)
        state.one_ecosystem_latent = ms_latent

        # ── Step 5: Math Reasoning ────────────────────────────────────────────
        if self.math_layer is not None:
            ml = self.math_layer()
            state.math_latent = ml

        # ── Step 6: Global Workspace ──────────────────────────────────────────
        mod_acts: Dict[str, torch.Tensor] = {
            "perception"    : perc,
            "working_memory": wm_ctx,
            "episodic"      : ep_ctx,
            "ecosystem"     : ms_latent,
        }
        if lang is not None:
            mod_acts["language"] = lang
        if state.math_latent is not None:
            mod_acts["math"] = state.math_latent

        ws, winner = self.global_workspace(mod_acts)
        state.workspace_state  = ws
        state.winner_module    = winner

        # ── Step 7: Psyche Executive Layer ────────────────────────────────────
        exec_out = self.psyche_exec(ws)
        state.executive_action = exec_out["executive_action"]
        state.safety_score     = float(exec_out["safety_score"].item())
        state.psyche_state     = exec_out.get("psyche_state")

        # ── Step 7-B: Psyche Plus — speculative-axiom evolution [v3.1 NEW] ────
        # Runs on a coarser cadence than every step (it mutates a persistent
        # axiom bank, not a per-step control signal) and NEVER performs
        # network I/O on its own — external consensus stays opt-in and
        # rate-limited via `maybe_consult_external`.
        if self.psyche_plus is not None and (
            self._step % max(1, self.cfg.psyche_plus_run_every_n_steps) == 0
        ):
            try:
                # [v3.2 NEW] Pull live simulation-quality scores from the
                # attached orchestrator (if any). `quality_score` is a
                # plain float (or None) — never part of the autograd graph,
                # since it's sourced from external/non-differentiable
                # health metrics (validation losses, BV-3 CME residuals).
                quality_score = None
                if self.orchestrator is not None:
                    try:
                        qrep = self.orchestrator.quality_report()
                        state.ecosystem_quality_scores = qrep
                        quality_score = qrep.get("ecosystem")
                    except Exception as q_exc:
                        logger.debug(f"quality_report() unavailable this step: {q_exc}")

                plus_out = self.psyche_plus(ws, quality_score=quality_score)
                state.psyche_plus_validity    = float(plus_out["validity_score"].mean().item())
                state.psyche_plus_axiom_count = plus_out["axiom_count"]
                state.psyche_plus_codified    = self.psyche_plus.maybe_codify(
                    plus_out, quality_score=quality_score
                )

                # Fold the (safety-gated) speculative update back into the
                # workspace state — SuperEgo's validity score already governs
                # how much of the speculation survives via EgoModule.reconcile.
                ws = plus_out["updated_latent"].squeeze(0)
                state.workspace_state = ws

                consensus = self.psyche_plus.maybe_consult_external(
                    prompt=f"Review AGI ONE speculative hypothesis at step {self._step}; "
                           f"reply with a one-sentence assessment and a "
                           f"'confidence: 0.0-1.0' line."
                )
                if consensus is not None:
                    state.external_consensus = consensus
            except Exception as exc:
                logger.debug(f"Psyche Plus cycle skipped this step: {exc}")

        # ── Step 8: DreamerV3 World Model ─────────────────────────────────────
        dummy_a = torch.zeros(1, self.cfg.action_dim, device=self.device)
        wm_out  = self.world_model(
            h=self.wm_h, z=self.wm_z, action=dummy_a, obs=ws.unsqueeze(0)
        )
        self.wm_h = wm_out["h_next"].detach()
        self.wm_z = wm_out["z_next"].detach()
        state.world_model_state = {
            "h_shape": tuple(self.wm_h.shape),
            "z_shape": tuple(self.wm_z.shape),
        }

        # ── Step 9: MPPI Planning ─────────────────────────────────────────────
        planned_action, _ = self.mppi.plan(
            world_model  = self.world_model,
            h            = self.wm_h,
            z            = self.wm_z,
            goal         = goal,
            psyche_bias  = exec_out.get("drive"),
        )
        state.planned_action = planned_action

        # ── Step 10: Geometry-aware Latent Diffusion (exploration) ────────────
        if self._step % 10 == 0:  # periodic latent exploration
            _ = self.latent_diffusion(ws.unsqueeze(0))

        # ── Step 11: Meta-Cognition ───────────────────────────────────────────
        ocd = (state.psyche_state.ocd_loop_detected
               if state.psyche_state is not None and
                  hasattr(state.psyche_state, "ocd_loop_detected")
               else False)
        meta  = self.meta_cognition(ws, ocd)
        state.meta_cognition = meta
        state.csoc_n_layers  = meta.get("csoc_n_layers")

        # ── Step 12: Dreamer-compound loss ────────────────────────────────────
        if compute_loss:
            state.total_loss = self._compute_dreamer_loss(
                ws=ws, wm_out=wm_out, planned_action=planned_action,
                exec_out=exec_out, reward=reward,
            )

        return state

    def _compute_dreamer_loss(
        self,
        ws            : torch.Tensor,
        wm_out        : Dict[str, torch.Tensor],
        planned_action: torch.Tensor,
        exec_out      : Dict,
        reward        : Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        DreamerV3-style compound loss with Kendall uncertainty weighting.

        L = Σ_task (1/2σ_task²) · L_task + log(σ_task)

        Tasks:
          world_kl    : DreamerV3 free-bits KL
          world_recon : symlog reconstruction
          reward      : two-hot reward prediction
          continue    : continuation prediction
          actor       : policy gradient (actor)
          value       : value function (critic)
          entropy     : action entropy bonus
          psy         : PSY BRIDGE Free Energy
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        zero = torch.tensor(0.0, device=self.device, requires_grad=True)

        # ── World model KL ────────────────────────────────────────────────────
        prior_l = wm_out.get("prior_logits")
        post_l  = wm_out.get("post_logits")
        if prior_l is not None and post_l is not None:
            loss_dict["world_kl"] = self.world_model.kl_loss(post_l, prior_l)

        # ── Reconstruction (symlog MSE) ────────────────────────────────────────
        obs_pred = wm_out.get("obs_pred")
        if obs_pred is not None and obs_pred.shape == ws.unsqueeze(0).shape:
            loss_dict["world_recon"] = F.mse_loss(
                symlog(obs_pred), symlog(ws.unsqueeze(0))
            )

        # ── Reward two-hot loss ───────────────────────────────────────────────
        if reward is not None:
            feat = wm_out.get("feat")
            if feat is not None:
                loss_dict["reward"] = self.world_model.reward_loss(
                    feat, reward.to(self.device).unsqueeze(-1)
                )

        # ── Continue prediction ───────────────────────────────────────────────
        cont_pred = wm_out.get("cont_pred")
        if cont_pred is not None:
            cont_target = torch.ones_like(cont_pred)
            loss_dict["continue"] = F.binary_cross_entropy(cont_pred, cont_target)

        # ── Actor loss (policy gradient) ──────────────────────────────────────
        action_logits = self.actor_net(ws)
        dist          = torch.distributions.Categorical(logits=action_logits)
        act_idx       = planned_action.argmax()
        log_p         = dist.log_prob(act_idx)

        if reward is not None:
            value_est  = self.critic_net(ws).squeeze()
            adv        = reward.to(self.device).squeeze() - value_est.detach()
            loss_dict["actor"] = -log_p * adv
            loss_dict["value"] = F.mse_loss(
                value_est, reward.to(self.device).squeeze().unsqueeze(0)
            )

        loss_dict["entropy"] = -dist.entropy()

        # ── PSY BRIDGE Free Energy ────────────────────────────────────────────
        psy_loss = exec_out.get("psy_loss")
        if psy_loss is not None:
            loss_dict["psy"] = psy_loss

        # ── Kendall uncertainty-weighted total ────────────────────────────────
        if loss_dict:
            total, weights = self.loss_balancer(loss_dict)
            return total
        return zero

    # =========================================================================
    # UTILITY
    # =========================================================================

    def reset(self) -> None:
        self.wm_h.zero_()
        self.wm_z.zero_()
        self.working_memory.reset()
        self.mppi.reset()
        self._step = 0

    def get_available_modules(self) -> Dict[str, bool]:
        return {
            "perception_vit"         : True,
            "audio_conformer"        : True,
            "language_rope_gpt"      : hasattr(self, "language"),
            "working_memory_ssc"     : True,
            "episodic_memory_dnd"    : True,
            "global_workspace_gwt"   : True,
            "dreamerv3_world_model"  : True,
            "mppi_planner"           : True,
            "meta_cognition_csoc"    : True,
            "psyche_executive_layer" : True,
            "psyche_plus_triad"      : self.psyche_plus is not None,
            "ssc_stabilizer"         : True,
            "interface_attention"    : True,
            "csoc_compute_ctrl"      : True,
            "struct_langevin_diff"   : True,
            "loss_balancer_kendall"  : True,
            "ppo_full"               : True,
            # [v3.5 NEW] The keys below previously reflected direct engine /
            # raw-module imports (bsd_one, grh_one, hodge_one, mental_one,
            # real_fold_one_v2, evolution_one_v3, etc.) that AGI ONE no
            # longer calls directly — see SECTION 15 and the "[10]" /
            # "[11]" removal notes above. They're kept here, hardcoded
            # False, only so any external code reading this dict's key set
            # doesn't break; the real status for these domains is reported
            # by EcosystemOrchestrator.quality_report() instead, via the
            # seven wired surrogate adapters (fold/hodge/mental/physics_*/
            # evolution_bv/math_numbertheory).
            "math_bsd_one"           : False,
            "math_grh_one"           : False,
            "math_hodge_one"         : False,
            "mental_one"             : False,
            "real_fold_one"          : False,
            "evolution_one"          : False,
            "epidemic_engine"        : False,
            "dns_cfd"                : False,
            "standard_one"           : False,
            "yang_mills"             : False,
            "rh_explorer"            : False,
            "open_science_registry"  : True,
        }

    def print_architecture(self) -> None:
        mods = self.get_available_modules()
        print(f"\n{'='*65}")
        print(f"  AGI ONE v{AGI_ONE_VERSION} — Production Architecture")
        print(f"  Developer  : Yoon A Limsuwan / MSPS NETWORK")
        print(f"  MY SOUL MOVE BY POWER OF HOLY SPIRIT")
        print(f"  AI Assistants: Claude (Anthropic), GPT-4o (OpenAI),")
        print(f"                 Gemini (Google DeepMind), DeepSeek")
        print(f"{'='*65}")
        print(f"  Device        : {self.device}")
        print(f"  Latent dim    : {self.cfg.latent_dim}")
        print(f"  Action dim    : {self.cfg.action_dim}")
        print(f"  Memory slots  : {self.cfg.memory_slots}")
        print(f"  Planning H    : {self.cfg.planning_horizon}")
        print(f"  MPPI samples  : {self.cfg.mppi_n_samples}")
        print(f"  Dreamer cats  : {self.cfg.dreamer_stoch_size}×{self.cfg.dreamer_stoch_classes}")
        total = sum(p.numel() for p in self.parameters())
        print(f"  Parameters    : {total:,}")
        print(f"\n  Module Status (v2.0):")
        for name, active in mods.items():
            s = "✓" if active else "✗"
            print(f"    {s}  {name}")
        print(f"{'='*65}\n")


# =============================================================================
# SECTION 18-A — ECOSYSTEM ORCHESTRATOR  (v3.0 NEW)
# Distributed surrogate module hub — replaces direct engine embedding
# =============================================================================

class SurrogateAdapter:
    """
    Thin adapter wrapping an uploaded surrogate module (e.g. StructuralFNO3D,
    StructuralGNOFold, StructuralGNOEvolution …) into a uniform interface
    used by EcosystemOrchestrator.

    The adapter holds:
      • module      : nn.Module  — the actual surrogate network
      • domain      : str        — 'physics' | 'fold' | 'evolution' |
                                   'evolution_bv' | 'mental' | 'math' |
                                   'hodge' | 'numbertheory'
      • frozen_backbone : bool   — whether backbone params are frozen
      • latent_dim  : int        — output latent dimension (auto-detected)
      • quality_fn  : Callable[[], float] | None   — [v3.2 NEW] optional
                      hook reporting *simulation* quality (not just the
                      latent), e.g. a cached validation loss, a BV-3 CME
                      residual, or any other domain-specific health metric
                      in [0, 1] (1.0 = best). If None, `get_quality_score()`
                      returns a neutral 0.5 — this is the "skip item 1"
                      path: Yoon can also just pass a plain float captured
                      from an external training run instead of a callable.
    """

    def __init__(
        self,
        module      : nn.Module,
        domain      : str,
        latent_dim  : int,
        name        : str = "",
        quality_fn  : Optional[Callable[[], float]] = None,
    ) -> None:
        self.module       = module
        self.domain       = domain
        self.latent_dim   = latent_dim
        self.name         = name or domain
        self.frozen_backbone = False
        self._projection: Optional[nn.Linear] = None   # align to AGI latent
        # [v3.2 NEW] quality_fn may be a live callable OR a plain float
        # wrapped in a lambda by the caller — either works since we only
        # ever call it with zero args inside get_quality_score().
        self.quality_fn = quality_fn

    def get_quality_score(self) -> float:
        """
        [v3.2 NEW] Best-effort simulation-quality readout for this domain,
        clamped to [0, 1]. Returns a neutral 0.5 when no `quality_fn` was
        supplied at registration time, or if the hook raises (a broken
        quality probe must never crash the AGI ONE forward pass).
        """
        if self.quality_fn is None:
            return 0.5
        try:
            val = float(self.quality_fn())
        except Exception as exc:
            logger.debug(f"SurrogateAdapter({self.name}).get_quality_score() fallback: {exc}")
            return 0.5
        if val != val:   # NaN guard
            return 0.5
        return min(max(val, 0.0), 1.0)

    def set_projection(self, agi_latent_dim: int, device: torch.device) -> None:
        """Create a learnable linear head to project surrogate output → AGI latent."""
        if self.latent_dim != agi_latent_dim:
            self._projection = nn.Linear(self.latent_dim, agi_latent_dim).to(device)
        else:
            self._projection = None

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze/unfreeze backbone parameters (keep output head trainable)."""
        self.frozen_backbone = freeze
        for name, p in self.module.named_parameters():
            # Heuristic: layers named 'head', 'out', 'proj' are output heads
            is_head = any(k in name for k in ("head", "out_proj", "output", "decoder"))
            p.requires_grad = (not freeze) or is_head

    def get_trainable_params(self) -> List[nn.Parameter]:
        params = [p for p in self.module.parameters() if p.requires_grad]
        if self._projection is not None:
            params += list(self._projection.parameters())
        return params

    def encode(self, dummy_latent: torch.Tensor) -> torch.Tensor:
        """
        Produce a latent vector from the surrogate using a dummy forward pass.

        In a real deployment each surrogate has its own input batch.
        Here we use the AGI perception latent as a conditioning signal
        (projected to match the surrogate's expected input dim via its own
        internal encoder or a learned adaptor layer).

        Returns: (1, agi_latent_dim) tensor on the same device.
        """
        device = dummy_latent.device
        # Each surrogate exposes .encode() or falls back to a zero latent
        try:
            with torch.set_grad_enabled(dummy_latent.requires_grad):
                # Minimal stub: surrogates that expose encode() use it;
                # others return a zeros placeholder until properly wired.
                if hasattr(self.module, "encode"):
                    out = self.module.encode(dummy_latent)           # (B, D_surr)
                else:
                    out = torch.zeros(
                        dummy_latent.shape[0], self.latent_dim,
                        device=device, dtype=dummy_latent.dtype
                    )
        except Exception as exc:
            logger.debug(f"SurrogateAdapter({self.name}).encode() fallback: {exc}")
            out = torch.zeros(
                dummy_latent.shape[0], self.latent_dim,
                device=device, dtype=dummy_latent.dtype
            )

        if self._projection is not None:
            out = self._projection(out)
        return out                                                   # (B, D_agi)


class GNOEvolutionBVEncoderAdapter(nn.Module):
    """
    [v3.2 NEW, extended v3.3, extended v3.7 for POP-2] Primary developer:
    Claude (Anthropic).

    Wraps `StructuralGNOEvolutionBV` (graph-batch interface, three modes:
    evolution / langevin / ch3d) so it satisfies the flat-latent `.encode()`
    protocol every other SurrogateAdapter module expects (see the
    `_DummySurrogate` pattern in the smoke test: `encode(x) -> tensor`).

    Why a wrapper instead of touching the BV file itself:
      structural_gno_evolution_bv_standalone.py is a verbatim, validated,
      self-contained module (state_dict-compatible with the plain GNO
      Evolution model) — editing it to understand AGI ONE's latent
      conventions would be exactly the kind of "regression risk" Yoon has
      flagged before (missing methods / dropped features on rewrite). This
      adapter only *composes* with it.

    What it does on `.encode(x)`:
      1. Lifts the incoming AGI latent `x : (B, D_agi)` into a small
         synthetic node-feature batch via a learned `lift` projection
         (B*N_synth, node_in_dim), paired with a fixed small ring-graph
         `edge_index` (registered as a buffer, not learned — it only needs
         to give FiLMMessagePassing *some* connectivity to mix over).
      2. Runs `StructuralGNOEvolutionBV.forward(batch, mode="evolution")`
         — the cheapest of the three modes, and the one with the BV-1
         cross-modal distillation target most relevant for AGI ONE's own
         μ/Rt-style epidemiological & evolutionary reasoning.
      3. Mean-pools `[mu_rt, logits]` over synthetic nodes → a small fixed
         vector, projected back to `agi_latent_dim`-compatible width by a
         learned `readout` head (this is the *only* new trainable
         parameter this adapter introduces; the wrapped BV model itself
         adds zero new parameters relative to plain GNO Evolution).

    [v3.3 NEW] A second, separate fixed synthetic batch — a small ch3d
    voxel grid, built with the exact same N/E/Nx convention as
    structural_gno_evolution_bv_standalone.py's own `__main__` smoke test —
    is registered alongside the Mode-1 graph above. It is used ONLY by
    `get_quality_score()` (see below) to drive `self.bv_model.cellpop_bridge`
    Mode-3 → Mode-4; it never touches `.encode()`'s latent path, so the
    pre-v3.3 `.encode()` contract (and therefore everything downstream of it
    in `EcosystemOrchestrator.forward()`) is completely unchanged.

    [v3.7 NEW] No new synthetic batch is needed for POP-2. When the
    attached `cell_population_one` is v1.1.0+ and
    `cfg.enable_cellpop_organelle=True` (both defaults), the BV model's own
    `cellpop_bridge.population` already carries an attached
    `OrganelleLayer` (and, on top of it, a `PhenotypeLayer`) — the SAME
    `cellpop_bridge.rollout(...)` call the v3.3 Mode-4 probe already makes
    on the existing synthetic ch3d grid now also returns
    `organelle_health_trace` for free. `_population_quality_score` reads
    that field (if present) into a third cached signal,
    `self._last_organelle_quality`, with no extra forward pass.

    Quality hook:
      `get_quality_score()` blends up to three independent [0, 1] signals:
        (a) BV-3 analytic certification (`bv_bridge.certify`) on the
            synthetic Mode-1 edge_index → CME residual → `exp(-residual)`.
            Unchanged from v3.2.
        (b) [v3.3 NEW] Mode-4 cell-population health: runs the synthetic
            ch3d grid through Mode 3, rolls `cellpop_bridge` forward on the
            resulting `pred_u`, and turns the rollout's final population
            size and division-rate calibration into a second [0, 1] score
            (see `_population_quality_score`).
        (c) [v3.7 NEW] POP-2 organelle health: from the SAME rollout used
            for (b), reads `organelle_health_trace[-1]` (native range
            roughly [-1, 1] — see `OrganelleLayer.organelle_health`'s own
            docstring) and rescales it to [0, 1] via `(health + 1) / 2`.
            `None` if no `OrganelleLayer` is attached (e.g. an older
            `cell_population_one` or `enable_cellpop_organelle=False`).
      Any subset of these may be unavailable; whichever signals *are*
      available this call are simple-averaged. If none are available, this
      returns the neutral 0.5 default — the same graceful degradation the
      BV file already documents for itself at every level.
    """

    def __init__(
        self,
        bv_model    : "StructuralGNOEvolutionBV",
        agi_latent_dim: int,
        n_synth_nodes : int = 16,
        n_synth_edges : int = 48,
        ch3d_grid_size: int = 4,
        device      : Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.bv_model       = bv_model
        self.n_synth_nodes  = n_synth_nodes
        node_in_dim         = bv_model.cfg.node_in_dim
        grid_in_dim         = bv_model.cfg.grid_in_dim
        d                   = bv_model.cfg.hidden_dim

        self.lift    = nn.Linear(agi_latent_dim, n_synth_nodes * node_in_dim)
        self.readout = nn.Linear(2 + 3, agi_latent_dim)   # mu_rt(2) + logits(3)

        dev = device or next(bv_model.parameters(), torch.empty(0)).device
        torch.manual_seed(0)   # fixed synthetic topology, reproducible across runs
        src = torch.randint(0, n_synth_nodes, (n_synth_edges,))
        dst = torch.randint(0, n_synth_nodes, (n_synth_edges,))
        self.register_buffer("_edge_index", torch.stack([src, dst], dim=0).to(dev))
        self.register_buffer("_sigma", torch.full((n_synth_nodes, 1), 0.5, device=dev))

        # [v3.3 NEW] Fixed synthetic ch3d grid for the Mode-4 quality probe
        # only — same construction as the BV file's own smoke test (N, E,
        # Nx = 12, 24, 4 there; here Nx is the one configurable knob since
        # it alone sets the grid's voxel count M = Nx**3 and hence the cost
        # of this probe). Registered as buffers (not learned) for the same
        # reason `_edge_index` / `_sigma` are: this is fixed scaffolding to
        # give the model *some* input to certify against, not a learned
        # representation of anything.
        Nx = ch3d_grid_size
        self._ch3d_grid_size = Nx
        M = Nx ** 3
        g_src = torch.randint(0, M, (M * 2,))
        g_dst = torch.randint(0, M, (M * 2,))
        self.register_buffer("_ch3d_u_init",     torch.randn(Nx, Nx, Nx, device=dev))
        self.register_buffer("_ch3d_grid_feats", torch.randn(M, grid_in_dim, device=dev))
        self.register_buffer("_ch3d_grid_edges", torch.stack([g_src, g_dst], dim=0).to(dev))
        self.register_buffer("_ch3d_sigma_3d",   torch.rand(Nx, Nx, Nx, device=dev))

        self._last_quality: float = 0.5
        self._last_pop_quality: float = 0.5
        self._last_organelle_quality: Optional[float] = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        node_in_dim = self.bv_model.cfg.node_in_dim
        feats = self.lift(x).view(B * self.n_synth_nodes, node_in_dim)
        # Tile the fixed synthetic graph across the batch dimension by simply
        # running B independent single-graph forward passes when B > 1 — GNO
        # batches in this ecosystem are per-graph, not padded/blocked, so we
        # keep this adapter's contract simple and correct rather than fast.
        outs = []
        for b in range(B):
            batch = _GNOBatchData(
                feats      = feats[b * self.n_synth_nodes:(b + 1) * self.n_synth_nodes],
                edge_index = self._edge_index,
                sigma      = self._sigma,
            )
            out = self.bv_model.forward(batch, mode="evolution")
            pooled = torch.cat([
                out["mu_rt"].mean(dim=0), out["logits"].mean(dim=0),
            ], dim=-1)
            outs.append(pooled)
        pooled_batch = torch.stack(outs, dim=0)
        return self.readout(pooled_batch)

    def _population_quality_score(self) -> Optional[float]:
        """
        [v3.3 NEW, extended v3.7 for POP-2] Mode-4 cell-population health →
        [0, 1], or None if `cellpop_bridge` is unavailable (caller falls
        back appropriately).

        Drives the fixed synthetic ch3d grid through Mode 3 to get a
        `pred_u`, rolls `cellpop_bridge` forward under `torch.no_grad()`
        (this is a read-only health probe, not a training step — it must
        never create a graph that the rest of AGI ONE's backward pass could
        accidentally walk into), and combines two diagnostics into one
        score:
          • survival factor   : n_alive_final / cellpop_n_max, clamped to
                                 [0, 1] — guards against population
                                 collapse (→0) or saturation at capacity
                                 (→1 is fine; this is not penalised, only
                                 collapse is, since saturating n_max is a
                                 capacity artifact of the synthetic probe's
                                 small size, not a real instability).
          • calibration factor: exp(-|rate_trace[-1] - pop_target_division_rate|
                                 / pop_target_division_rate) — how close the
                                 induced division rate sits to the
                                 configured target; 1.0 at exact match,
                                 decaying smoothly otherwise.
        The two are averaged. Both factors are intentionally cheap,
        bounded, and have no failure mode that can return outside [0, 1].

        [v3.7 NEW] The SAME rollout computed here also feeds
        `self._last_organelle_quality` (POP-2's third quality signal,
        see `get_quality_score`) whenever `rollout.organelle_health_trace`
        is present — no extra forward pass, since `cellpop_bridge.rollout`
        already returns the organelle trace alongside the population
        trace when an `OrganelleLayer` is attached (see
        `CellPopulationTrainingBridge.rollout`).
        """
        bridge = getattr(self.bv_model, "cellpop_bridge", None)
        if bridge is None or not getattr(bridge, "available", False):
            return None
        try:
            cfg = self.bv_model.cfg
            with torch.no_grad():
                ch_batch = _GNOBatchData(
                    u_init          = self._ch3d_u_init,
                    grid_feats      = self._ch3d_grid_feats,
                    grid_edge_index = self._ch3d_grid_edges,
                    sigma_3d        = self._ch3d_sigma_3d,
                )
                out_ch  = self.bv_model.forward(ch_batch, mode="ch3d")
                rollout = bridge.rollout(out_ch["pred_u"], hard=True)
            if rollout is None:
                return None

            survival = min(max(rollout.n_alive_trace[-1] / max(cfg.cellpop_n_max, 1), 0.0), 1.0)

            target = float(cfg.pop_target_division_rate)
            achieved = float(rollout.rate_trace[-1].item())
            calibration = float(
                torch.exp(
                    torch.tensor(-abs(achieved - target) / max(target, 1e-6))
                ).item()
            )
            self._last_pop_quality = min(max(0.5 * (survival + calibration), 0.0), 1.0)

            # [v3.7 NEW] POP-2 organelle reading, piggy-backed on this same
            # rollout. `organelle_health_trace` is None whenever no
            # OrganelleLayer is attached (older cell_population_one, or
            # cfg.enable_cellpop_organelle=False) — left as None (not 0.5)
            # in that case so get_quality_score() can tell "unavailable"
            # apart from "available but neutral".
            if rollout.organelle_health_trace is not None:
                health_final = float(rollout.organelle_health_trace[-1].item())
                # native range ~[-1, 1] (see OrganelleLayer.organelle_health
                # docstring) -> [0, 1]
                self._last_organelle_quality = min(max((health_final + 1.0) / 2.0, 0.0), 1.0)
        except Exception as exc:
            logger.debug(f"GNOEvolutionBVEncoderAdapter._population_quality_score() fallback: {exc}")
            return self._last_pop_quality
        return self._last_pop_quality

    def get_quality_score(self) -> float:
        """
        [v3.2 NEW, extended v3.3, extended v3.7] BV-3 certification +
        Mode-4 population health + POP-2 organelle health → blended
        [0, 1] quality score (cached fallback on any sub-probe's failure).
        """
        bv_score: Optional[float] = None
        try:
            bridge = getattr(self.bv_model, "bv_bridge", None)
            if bridge is not None and getattr(bridge, "available", False):
                cert = bridge.certify(self._edge_index, step=0, kind="evo")
                if cert is not None:
                    bv_score = float(
                        torch.exp(torch.tensor(-abs(cert.cme_residual))).item()
                    )
                    self._last_quality = bv_score
        except Exception as exc:
            logger.debug(f"GNOEvolutionBVEncoderAdapter.get_quality_score() fallback: {exc}")

        # [v3.7 NEW] Reset the organelle cache to None before each call so a
        # population probe that fails (and therefore never updates it) does
        # not silently resurrect a stale reading from several calls ago as
        # if it were fresh; _population_quality_score repopulates it below
        # whenever the rollout it runs actually carries an organelle trace.
        self._last_organelle_quality = None
        pop_score = self._population_quality_score()
        organelle_score = self._last_organelle_quality

        scores = [s for s in (bv_score, pop_score, organelle_score) if s is not None]
        if not scores:
            return self._last_quality   # neutral 0.5 unless a prior call cached something
        return sum(scores) / len(scores)


def attach_evolution_bv_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
    bv_cfg        : Optional["SGNOEvoBVConfig"] = None,
    name          : str = "structural_gno_evolution_bv",
) -> Optional[GNOEvolutionBVEncoderAdapter]:
    """
    [v3.2 NEW, quality hook extended v3.3, extended v3.7 for POP-2]
    Convenience one-liner for Yoon's own scripts:

        attach_evolution_bv_to_ecosystem(trainer.orchestrator, 512, device)

    Builds a `StructuralGNOEvolutionBV`, wraps it in
    `GNOEvolutionBVEncoderAdapter`, and registers it on `orchestrator`
    under domain "evolution_bv" with its combined BV-3 certification +
    Mode-4 cell-population health + POP-2 organelle health wired in as a
    live `quality_fn` (see `GNOEvolutionBVEncoderAdapter.get_quality_score`).
    If `bv_cfg` leaves `enable_cell_population=True` and
    `enable_cellpop_organelle=True` (both the BV file's own defaults) and
    the attached `cell_population_one` is v1.1.0+, Mode-4 AND POP-2 are
    both included automatically; otherwise the score quietly falls back to
    whichever subset of signals is actually available (down to BV-3 alone,
    exactly as in v3.2), never raising. Returns the adapter's underlying
    wrapper module (so the caller can still load a real checkpoint into
    `.bv_model` afterwards) or None if
    `structural_gno_evolution_bv_standalone` isn't importable in this
    environment — never raises.
    """
    if not HAS_GNO_EVOLUTION_BV:
        logger.warning(
            "attach_evolution_bv_to_ecosystem: structural_gno_evolution_bv_standalone "
            "unavailable — skipping registration."
        )
        return None
    dev = device or torch.device("cpu")
    cfg = bv_cfg or SGNOEvoBVConfig()
    bv_model = StructuralGNOEvolutionBV(cfg).to(dev)
    wrapper  = GNOEvolutionBVEncoderAdapter(bv_model, agi_latent_dim, device=dev).to(dev)
    orchestrator.register(
        name, wrapper, "evolution_bv", agi_latent_dim,
        quality_fn=wrapper.get_quality_score,
    )
    return wrapper


# =============================================================================
# [v3.4 NEW] Real adapters for the six previously-unwired surrogates:
#   StructuralFNO3D, MentalStructuralNeuralOperator, StructuralGNOFold,
#   StructuralGNOHodge, StructuralGNONumberTheory, StructuralGNOPhysics.
#
# Same pattern as GNOEvolutionBVEncoderAdapter above: a fixed (buffer-only,
# non-learned) synthetic input scaffold + a learned `lift` (AGI latent →
# surrogate input) and `readout` (surrogate output → AGI latent) pair. The
# wrapped surrogate's own forward pass runs for real every call; only the
# lift/readout heads are new trainable parameters. Each adapter satisfies
# the flat `.encode(x) -> (B, D_agi)` protocol SurrogateAdapter expects.
# =============================================================================

class StructuralFNO3DEncoderAdapter(nn.Module):
    """
    [v3.4 NEW] Wraps `StructuralFNO3D` (SUPER DNS ONE surrogate).

    `StructuralFNO3D.forward(u_initial, sigma, t_target)` expects a 3-D
    voxel grid `(B, 1, Nx, Ny, Nz)` for both the field and the structural
    regime — there is no flat-vector path, so this adapter lifts the AGI
    latent into a small fixed-size synthetic grid, runs the real forward
    pass, and mean-pools the predicted field (+ predictive log-variance)
    back down to a vector.
    """

    def __init__(
        self,
        fno_model     : "StructuralFNO3D",
        agi_latent_dim: int,
        grid_size     : int = 8,
        device        : Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.fno_model = fno_model
        self.Nx = self.Ny = self.Nz = grid_size
        M = grid_size ** 3

        self.lift    = nn.Linear(agi_latent_dim, M)            # → u_initial field
        self.readout = nn.Linear(2, agi_latent_dim)             # [mean, log_var] pooled

        dev = device or next(fno_model.parameters(), torch.empty(0)).device
        torch.manual_seed(0)
        self.register_buffer("_sigma", torch.full((1, 1, grid_size, grid_size, grid_size), 0.5, device=dev))

        self._last_quality: float = 0.5

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        u0 = self.lift(x).view(B, 1, self.Nx, self.Ny, self.Nz)
        sigma = self._sigma.expand(B, -1, -1, -1, -1)
        mean, log_var = self.fno_model(u0, sigma, t_target=0.5)
        pooled = torch.stack([mean.mean(dim=[1, 2, 3, 4]), log_var.mean(dim=[1, 2, 3, 4])], dim=-1)
        return self.readout(pooled)

    def get_quality_score(self) -> float:
        """
        [v3.4 NEW] Predictive-uncertainty-based health probe: lower mean
        predictive log-variance over the fixed synthetic grid → higher
        confidence → higher score. Read-only (`torch.no_grad()`), never
        raises, falls back to the last cached value (or neutral 0.5) on
        any error — same contract as every other `get_quality_score()`
        in this ecosystem.
        """
        try:
            with torch.no_grad():
                dev = self._sigma.device
                u0 = torch.zeros(1, 1, self.Nx, self.Ny, self.Nz, device=dev)
                _, log_var = self.fno_model(u0, self._sigma, t_target=0.5)
                score = float(torch.exp(-log_var.mean().clamp(min=-10.0, max=10.0)).clamp(0.0, 1.0).item())
            self._last_quality = score
            return score
        except Exception as exc:
            logger.debug(f"StructuralFNO3DEncoderAdapter.get_quality_score() fallback: {exc}")
            return self._last_quality


def attach_sfno3d_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
    fno_kwargs    : Optional[Dict[str, Any]] = None,
    name          : str = "structural_fno_3d",
) -> Optional[StructuralFNO3DEncoderAdapter]:
    """
    [v3.4 NEW] One-liner registration, mirroring `attach_evolution_bv_to_ecosystem`.
    Returns the adapter (so a real checkpoint can be loaded into
    `.fno_model` afterwards), or None if `structural_fno_3d` isn't
    importable in this environment — never raises.
    """
    if not HAS_SFNO3D:
        logger.warning("attach_sfno3d_to_ecosystem: structural_fno_3d unavailable — skipping registration.")
        return None
    dev = device or torch.device("cpu")
    model   = StructuralFNO3D(**(fno_kwargs or {})).to(dev)
    wrapper = StructuralFNO3DEncoderAdapter(model, agi_latent_dim, device=dev).to(dev)
    orchestrator.register(name, wrapper, "physics", agi_latent_dim, quality_fn=wrapper.get_quality_score)
    return wrapper


class MentalOperatorEncoderAdapter(nn.Module):
    """
    [v3.4 NEW] Wraps `MentalStructuralNeuralOperator` (MENTAL ONE surrogate).

    MSNO exposes four independent prediction heads rather than one
    `forward()`. This adapter routes through `predict_eeg_trajectory`
    (the cheapest, no-graph branch) for `.encode()`, and additionally
    drives `optimize_ego` under `torch.no_grad()` as a lightweight second
    signal for `get_quality_score()` — analogous to how the BV adapter
    blends two cheap sub-probes rather than relying on just one.
    """

    def __init__(
        self,
        msno_model    : "MentalStructuralNeuralOperator",
        agi_latent_dim: int,
        seq_len       : int = 64,
        device        : Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.msno_model = msno_model
        self.eeg_channels = msno_model.lift_1d.in_channels
        self.T = seq_len

        self.lift    = nn.Linear(agi_latent_dim, self.eeg_channels * seq_len)
        self.readout = nn.Linear(self.eeg_channels, agi_latent_dim)

        dev = device or next(msno_model.parameters(), torch.empty(0)).device
        self.register_buffer("_sigma", torch.full((1, 1, 1), 0.5, device=dev))
        self.register_buffer("_action_probe", torch.zeros(1, msno_model.action_dim, device=dev))
        self.register_buffer("_sigma_action", torch.full((1, 1), 0.5, device=dev))

        self._last_quality: float = 0.5

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        eeg0 = self.lift(x).view(B, self.eeg_channels, self.T)
        sigma = self._sigma.expand(B, -1, -1)
        eeg_future = self.msno_model.predict_eeg_trajectory(eeg0, sigma)
        pooled = eeg_future.mean(dim=-1)          # (B, eeg_channels)
        return self.readout(pooled)

    def get_quality_score(self) -> float:
        """
        [v3.4 NEW] `optimize_ego` stability probe: with a neutral Id/Superego
        proposal pair, a healthy operator should output a roughly uniform
        (high-entropy) action distribution rather than a collapsed
        one-hot — entropy ratio to max-entropy gives a bounded [0, 1]
        health signal. Falls back to the cached value on any failure.
        """
        try:
            with torch.no_grad():
                probs = self.msno_model.optimize_ego(
                    self._action_probe, self._action_probe, self._sigma_action,
                )
                ent = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
                max_ent = math.log(probs.shape[-1])
                score = float((ent / max(max_ent, 1e-8)).clamp(0.0, 1.0).item())
            self._last_quality = score
            return score
        except Exception as exc:
            logger.debug(f"MentalOperatorEncoderAdapter.get_quality_score() fallback: {exc}")
            return self._last_quality


def attach_msno_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
    msno_kwargs   : Optional[Dict[str, Any]] = None,
    name          : str = "mental_structural_operator",
) -> Optional[MentalOperatorEncoderAdapter]:
    """[v3.4 NEW] One-liner registration for MENTAL ONE's MSNO surrogate."""
    if not HAS_MSNO:
        logger.warning("attach_msno_to_ecosystem: mental_structural_operator_v3 unavailable — skipping registration.")
        return None
    dev = device or torch.device("cpu")
    model   = MentalStructuralNeuralOperator(**(msno_kwargs or {})).to(dev)
    wrapper = MentalOperatorEncoderAdapter(model, agi_latent_dim, device=dev).to(dev)
    orchestrator.register(name, wrapper, "mental", agi_latent_dim, quality_fn=wrapper.get_quality_score)
    return wrapper


class GNOFoldEncoderAdapter(nn.Module):
    """
    [v3.4 NEW] Wraps `StructuralGNOFold` (REAL FOLD ONE surrogate).

    `StructuralGNOFold` is single-graph (no batch dim) in both of its
    modes. This adapter uses `forward_phase_field(u_init, sigma_3d)` — the
    grid-graph mode — since its graph is built deterministically from grid
    adjacency (no coordinate-dependent radius graph to keep stable across
    a learned lift, unlike protein mode's `_build_protein_graph`). Runs
    one real graph forward pass per batch item, exactly as
    `GNOEvolutionBVEncoderAdapter.encode()` does for the same structural
    reason (GNO batches in this ecosystem are per-graph, not padded).
    """

    def __init__(
        self,
        fold_model    : "StructuralGNOFold",
        agi_latent_dim: int,
        grid_size     : int = 4,
        device        : Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.fold_model = fold_model
        self.Nx = self.Ny = self.Nz = grid_size
        M = grid_size ** 3

        self.lift    = nn.Linear(agi_latent_dim, M)
        self.readout = nn.Linear(1, agi_latent_dim)

        dev = device or next(fold_model.parameters(), torch.empty(0)).device
        self.register_buffer("_sigma_3d", torch.full((grid_size, grid_size, grid_size), 0.5, device=dev))

        self._last_quality: float = 0.5

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        u0 = self.lift(x).view(B, self.Nx, self.Ny, self.Nz)
        outs = []
        for b in range(B):
            u_pred = self.fold_model.forward_phase_field(u0[b], self._sigma_3d)
            outs.append(u_pred.mean().unsqueeze(0))
        pooled = torch.stack(outs, dim=0)          # (B, 1)
        return self.readout(pooled)

    def get_quality_score(self) -> float:
        """
        [v3.4 NEW] Mass-conservation probe: a well-behaved phase-field
        surrogate should keep the mean order parameter roughly stable
        under one forward step (Cahn-Hilliard dynamics are mass-
        conserving). Score = exp(-|Δmean|), bounded in (0, 1].
        """
        try:
            with torch.no_grad():
                u_init = torch.zeros(self.Nx, self.Ny, self.Nz, device=self._sigma_3d.device)
                u_pred = self.fold_model.forward_phase_field(u_init, self._sigma_3d)
                delta = (u_pred.mean() - u_init.mean()).abs()
                score = float(torch.exp(-delta).clamp(0.0, 1.0).item())
            self._last_quality = score
            return score
        except Exception as exc:
            logger.debug(f"GNOFoldEncoderAdapter.get_quality_score() fallback: {exc}")
            return self._last_quality


def attach_gno_fold_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
    fold_cfg      : Optional["SGNOFoldConfig"] = None,
    name          : str = "structural_gno_fold",
) -> Optional[GNOFoldEncoderAdapter]:
    """
    [v3.4 NEW, rewired v3.5] One-liner registration for REAL FOLD ONE's
    GNOFold surrogate — now backed by structural_gno_fold_v4 (sparse
    radius-graph construction; see the import block above for why). No
    change to this function's own logic was required: `StructuralGNOFold`
    and `SGNOConfig` keep the same constructor/forward signatures in v4,
    so the adapter wiring below is identical to the v3 wiring.
    """
    if not HAS_GNO_FOLD:
        logger.warning("attach_gno_fold_to_ecosystem: structural_gno_fold_v4 (or _v3 fallback) unavailable — skipping registration.")
        return None
    dev = device or torch.device("cpu")
    cfg = fold_cfg or SGNOFoldConfig()
    model   = StructuralGNOFold(cfg).to(dev)
    wrapper = GNOFoldEncoderAdapter(model, agi_latent_dim, device=dev).to(dev)
    orchestrator.register(name, wrapper, "fold", agi_latent_dim, quality_fn=wrapper.get_quality_score)
    return wrapper


class GNOHodgeEncoderAdapter(nn.Module):
    """
    [v3.4 NEW, rewired v3.4.1] Wraps `UnifiedHodgeOperatorTrainer` — the
    top-level entry point of `structural_gno_hodge.py` v2 — instead of
    calling the old standalone `StructuralGNOHodge(x_pos, target, sigma)`
    directly.

    Why this had to change
    -----------------------
    hodge_one.py was rebuilt (v2) around two concrete polarized Hodge
    varieties (`hodge_one.VarietyClass.ABELIAN` / `.K3_ABSTRACT`), and
    structural_gno_hodge.py followed: `StructuralGNOHodge` is now an
    *internal* 1-D slice network, reused either by
    `StructuralGNOHodgeAbelianAdapter` (applied independently across the
    `g*2` real coordinate slices of a complex torus) or replaced entirely
    by `StructuralGNOHodgeK3` (a noise→R^22 generator with no particle
    cloud at all). There is no longer a single `(x_pos, target, sigma) ->
    x_optimal` call that works for "the" Hodge model — which path runs is
    decided once, at build time, by `cfg.variety`.

    Adapter design
    ---------------
    `UnifiedHodgeOperatorTrainer` (built by `build_gno_hodge_system`) is
    the one object that knows which path is active (`self.trainer.built.kind`)
    and owns everything needed to run it: the GNO model itself
    (`trainer.model`), `hodge_one.CycleClassMap` for computing periods,
    `hodge_one.LearnableSOCKernel` for the sigma context, and the matching
    loss functions. This adapter therefore does not reimplement variety
    dispatch — it mirrors exactly the two private step methods the trainer
    itself uses (`_abelian_step` / `_k3_step`), minus the optimizer step,
    so AGI ONE's `.encode()` calls run the *real* forward path of whichever
    variety `hodge_cfg.variety` selected, not a re-derived approximation.

    `.encode()` reads out `computed_periods: (B, period_dim)` rather than
    the particle tensor, because `period_dim` is the one tensor shape that
    is guaranteed stable across both varieties (g*g for ABELIAN, 22 for
    K3_ABSTRACT) — the particle tensor shape `(B, N, g, 2)` only exists for
    ABELIAN, so reading it out directly would make `.encode()` variety-
    dependent, which the rest of the ecosystem's flat-latent `.encode()`
    protocol must not be.
    """

    def __init__(
        self,
        hodge_trainer : "UnifiedHodgeOperatorTrainer",
        agi_latent_dim: int,
        n_particles   : int = 32,
        device        : Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.trainer = hodge_trainer
        self.N = n_particles
        period_dim = hodge_trainer.built.period_dim

        dev = device or hodge_trainer.device

        self.lift_noise = nn.Linear(agi_latent_dim, hodge_trainer.cfg.noise_dim)
        self.readout    = nn.Linear(period_dim, agi_latent_dim)

        # Fixed synthetic target Hodge vector, deterministic across calls
        # (mirrors the v3.4 pre-rewire convention of one frozen random
        # target rather than re-sampling on every encode()).
        torch.manual_seed(0)
        self.register_buffer(
            "_target_hodge", torch.randn(1, period_dim, device=dev)
        )

        self._last_quality: float = 0.5

    # ------------------------------------------------------------------
    def _run_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One real (no-grad-agnostic) forward pass through whichever variety
        is active, returning `computed_periods: (B, period_dim)`. Mirrors
        `UnifiedHodgeOperatorTrainer._abelian_step` / `._k3_step` exactly
        (skipping the loss/optimizer machinery, since this is an
        encode-time read, not a training step), with the AGI latent `x`
        wired in as the actual conditioning signal so `.encode()` is not
        input-independent:
          • K3_ABSTRACT — `x` is projected (`self.lift_noise`) straight
            into the generator's noise vector, replacing pure `randn`.
          • ABELIAN — there is no learned lift into `(B, N, g, 2)` particle
            space without an extra reshape head per (g, N) combination, so
            `x` instead perturbs the uniform initial particle cloud via a
            small low-rank nudge derived from `lift_noise(x)` reshaped/
            broadcast across particles — cheap, shape-stable across any
            `(g, N)`, and still latent-dependent.
        """
        B = x.shape[0]
        device, dtype = x.device, x.dtype
        target = self._target_hodge.expand(B, -1).to(dtype)
        trainer = self.trainer
        cond = self.lift_noise(x)                                       # (B, noise_dim)

        if trainer.built.kind == hodge_one_v2.VarietyClass.ABELIAN:
            g = trainer.built.torus.g
            z_init = torch.rand(B, self.N, g, 2, device=device, dtype=dtype)
            # Low-rank latent nudge: fold cond's mean/std into a per-batch
            # offset, broadcast across (N, g, 2), then wrap to [0,1) so the
            # torus topology is respected exactly like the trainer itself
            # does after the model's own drift step.
            nudge = cond.mean(dim=1, keepdim=True) * 0.05               # (B, 1)
            z_init = (z_init + nudge.view(B, 1, 1, 1)).remainder(1.0)
            r = z_init.std(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)   # (B,1)
            sigma = trainer._sigma_from_dispersion(r)
            z_opt = trainer.model(z_init, target, sigma)                # (B,N,g,2)
            periods = trainer.cycle_map(z_opt)                          # (B, period_dim)
        else:
            noise = cond                                                 # (B, noise_dim)
            r = noise.std(dim=1, keepdim=True)
            sigma = trainer._sigma_from_dispersion(r)
            raw_vec = trainer.model(noise, target, sigma)               # (B, period_dim)
            periods = project_to_h11_batch(trainer.built.k3, raw_vec)   # (B, period_dim)
        return periods

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        periods = self._run_forward(x)                                  # (B, period_dim)
        return self.readout(periods)

    def get_quality_score(self) -> float:
        """
        [v3.4.1 rewired] Real simulation-quality probe: runs one batched
        forward pass against the fixed synthetic target and scores it with
        the *actual* hodge_one v2 losses this variety trains against
        (period-fit + Hodge-Riemann + integrality, plus spacing/entropy/
        smooth for ABELIAN) — not a re-derived spacing-ratio heuristic.
        Mapped to (0, 1] via exp(-total_loss), clamped, so lower loss ->
        score closer to 1.0.
        """
        try:
            trainer = self.trainer
            dev = self._target_hodge.device
            with torch.no_grad():
                B = 1
                target = self._target_hodge
                if trainer.built.kind == hodge_one_v2.VarietyClass.ABELIAN:
                    g = trainer.built.torus.g
                    z_init = torch.rand(B, self.N, g, 2, device=dev)
                    r = z_init.std(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
                    sigma = trainer._sigma_from_dispersion(r)
                    z_opt = trainer.model(z_init, target, sigma)
                    periods = trainer.cycle_map(z_opt)
                    total_loss, _ = compute_total_loss_abelian(
                        z_opt, periods, target, trainer.cfg,
                        trainer.hr_loss_fn, trainer.integrality,
                    )
                else:
                    noise = torch.randn(B, trainer.cfg.noise_dim, device=dev)
                    r = noise.std(dim=1, keepdim=True)
                    sigma = trainer._sigma_from_dispersion(r)
                    raw_vec = trainer.model(noise, target, sigma)
                    periods = project_to_h11_batch(trainer.built.k3, raw_vec)
                    total_loss, _ = compute_total_loss_k3(
                        periods, target, trainer.cfg,
                        trainer.hr_loss_fn, trainer.integrality,
                    )
                score = float(torch.exp(-total_loss.clamp_min(0.0)).clamp(0.0, 1.0).item())
            self._last_quality = score
            return score
        except Exception as exc:
            logger.debug(f"GNOHodgeEncoderAdapter.get_quality_score() fallback: {exc}")
            return self._last_quality


def attach_gno_hodge_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
    hodge_cfg     : Optional["SGNOHodgeConfig"] = None,
    name          : str = "structural_gno_hodge",
) -> Optional[GNOHodgeEncoderAdapter]:
    """
    [v3.4 NEW, rewired v3.4.1] One-liner registration for HODGE ONE's
    GNOHodge surrogate, now built through `build_gno_hodge_system()` (i.e.
    `UnifiedHodgeOperatorTrainer`) rather than constructing the bare
    `StructuralGNOHodge` network directly — see `GNOHodgeEncoderAdapter`
    docstring for why the old direct-construction path no longer applies.

    `hodge_cfg.variety` ("abelian" | "k3_abstract") selects which polarized
    Hodge structure AGI ONE's hodge domain reasons over; defaults to
    "abelian" (matching `SGNOHodgeConfig`'s own default) if `hodge_cfg`
    is not supplied.
    """
    if not HAS_GNO_HODGE:
        logger.warning("attach_gno_hodge_to_ecosystem: structural_gno_hodge (or hodge_one v2) unavailable — skipping registration.")
        return None
    dev = device or torch.device("cpu")
    cfg = hodge_cfg or SGNOHodgeConfig()
    trainer = build_gno_hodge_system(cfg, dev)
    wrapper = GNOHodgeEncoderAdapter(trainer, agi_latent_dim, device=dev).to(dev)
    orchestrator.register(name, wrapper, "hodge", agi_latent_dim, quality_fn=wrapper.get_quality_score)
    return wrapper


class GNONumberTheoryEncoderAdapter(nn.Module):
    """
    [v3.4 NEW] Wraps `StructuralGNONumberTheory` (RH/GRH/BSD ONE surrogate).

    `forward(x_pos, l_params, sigma)` is batched `(B, N)` / `(B, 3)` /
    `(B, 1)` — same shape pattern as Hodge, so no per-item loop needed.
    `l_params = [degree, conductor, analytic_rank]` is held at a fixed
    neutral synthetic value (RH-like: degree 1, conductor 1, rank 0).
    """

    def __init__(
        self,
        nt_model      : "StructuralGNONumberTheory",
        agi_latent_dim: int,
        n_particles   : int = 32,
        device        : Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.nt_model = nt_model
        self.N = n_particles

        self.lift    = nn.Linear(agi_latent_dim, n_particles)
        self.readout = nn.Linear(n_particles, agi_latent_dim)

        dev = device or next(nt_model.parameters(), torch.empty(0)).device
        self.register_buffer("_l_params", torch.tensor([[1.0, 1.0, 0.0]], device=dev))

        self._last_quality: float = 0.5

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_pos = self.lift(x)                                  # (B, N)
        l_params = self._l_params.expand(B, -1)
        sigma = torch.full((B, 1), 0.5, device=x.device, dtype=x.dtype)
        x_new = self.nt_model(x_pos, l_params, sigma)          # (B, N)
        return self.readout(x_new)

    def get_quality_score(self) -> float:
        """
        [v3.4 NEW] Drift-boundedness probe: `StructuralGNONumberTheory`
        is documented as a *small corrector* over SSC simulation (soft-
        clamped drift), so a healthy model should leave a uniform input
        close to itself rather than producing a large correction. Score
        = exp(-mean|drift|), bounded in (0, 1].
        """
        try:
            with torch.no_grad():
                dev = self._l_params.device
                x0 = torch.linspace(0.0, 1.0, self.N, device=dev).unsqueeze(0)
                sigma = torch.full((1, 1), 0.5, device=dev)
                x_new = self.nt_model(x0, self._l_params, sigma)
                drift = (x_new - x0).abs().mean()
                score = float(torch.exp(-drift).clamp(0.0, 1.0).item())
            self._last_quality = score
            return score
        except Exception as exc:
            logger.debug(f"GNONumberTheoryEncoderAdapter.get_quality_score() fallback: {exc}")
            return self._last_quality


def attach_gno_numbertheory_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
    nt_cfg        : Optional["SGNONumberTheoryConfig"] = None,
    name          : str = "structural_gno_numbertheory",
) -> Optional[GNONumberTheoryEncoderAdapter]:
    """[v3.4 NEW] One-liner registration for the RH/GRH/BSD GNONumberTheory surrogate."""
    if not HAS_GNO_NUMBERTHEORY:
        logger.warning("attach_gno_numbertheory_to_ecosystem: structural_gno_numbertheory unavailable — skipping registration.")
        return None
    dev = device or torch.device("cpu")
    cfg = nt_cfg or SGNONumberTheoryConfig()
    model   = StructuralGNONumberTheory(cfg).to(dev)
    wrapper = GNONumberTheoryEncoderAdapter(model, agi_latent_dim, device=dev).to(dev)
    orchestrator.register(name, wrapper, "math", agi_latent_dim, quality_fn=wrapper.get_quality_score)
    return wrapper


class GNOPhysicsEncoderAdapter(nn.Module):
    """
    [v3.4 NEW, extended v3.6 for Mode 4] Wraps `StructuralGNOPhysics`
    (NGO Physics ONE surrogate; collider / CMB cosmology / Yang–Mills /
    entangled-pair CHSH correlator modes).

    Uses `forward_ym(momentum_data, sigma)` for the primary `.encode()`
    path — the cheapest of the four modes (2-dim input, 1-dim output, no
    `lmax`-sized CMB head) — analogous to the BV adapter's choice of its
    cheapest mode for the main latent path. `get_quality_score()` probes
    the collider head, since its softplus output gives a natural
    non-negativity sanity check the YM head (which is allowed to go
    negative near the complex-pole region) doesn't.

    [v3.6 NEW] Mode 4 — entangled-pair polarization/spin correlator
    (`forward_entangled`, ngo_physics_one.py v1.1.0) is now wired in
    alongside Mode 3, mirroring how `GNOEvolutionBVEncoderAdapter` (v3.3)
    added its Mode-4 cell-population signal on top of an existing
    Mode-1 latent path:
      • `.encode()` now mean-blends the YM latent with a second latent
        derived from `forward_entangled`, via a second learned `lift`/
        `readout` pair (`lift_entangled` / `readout_entangled`). This is
        purely additive — the pre-v3.6 YM-only contribution is still
        computed exactly the same way, just averaged with the new one
        instead of returned alone — so nothing downstream that depended
        on the old latent's rough scale or sign breaks.
      • `get_quality_score()` now blends two independent [0, 1] signals
        when both are available:
          (a) collider-head boundedness probe            (unchanged, v3.4)
          (b) [v3.6 NEW] CHSH physical-realizability probe — runs the
              model's own `forward_entangled` at the textbook CHSH-
              optimal angle settings (same defaults as
              `NGOPhysicsInference.chsh_test()`) and scores how close
              the predicted |S| sits to a sane regime: 1.0 well inside
              the Tsirelson bound, decaying smoothly as |S| approaches
              or exceeds `TSIRELSON_BOUND` (imported from
              `bell_chsh_one.py` when available, else a hard-coded
              textbook fallback — see the optional import block above).
              This is a *numerical sanity* probe on the surrogate's own
              untrained-or-trained output, not a physics claim: a
              freshly-initialised network scoring low here just means
              its random Mode-4 head hasn't learned the Tsirelson prior
              yet, exactly analogous to how the v3.4 collider probe can
              score low before `physics_penalty_chsh()` has had any
              training steps to act on it.
        If `bell_chsh_one.py` isn't importable, signal (b) still runs
        (using the textbook fallback bounds) rather than silently
        dropping out — unlike the BV adapter's bridges, this probe has
        no real external dependency to lose, only an optional shared
        constant to import for safety.
    """

    def __init__(
        self,
        physics_model : "StructuralGNOPhysics",
        agi_latent_dim: int,
        device        : Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.physics_model = physics_model

        self.lift    = nn.Linear(agi_latent_dim, 2)     # → [p^2, alpha_s(p^2)]
        self.readout = nn.Linear(1, agi_latent_dim)

        # [v3.6 NEW] Mode-4 entangled-pair lift/readout. Output of `lift_entangled`
        # is squashed per-column into physically sane ranges before being handed
        # to `forward_entangled` — the raw linear output is otherwise unbounded,
        # but [θ_a, θ_b, V, pair_source_id] all have a meaningful physical domain:
        #   θ_a, θ_b        ∈ (-π, π]   via tanh * π
        #   V (visibility)  ∈ [0, 1]    via sigmoid
        #   pair_source_id  left as a free categorical-style scalar (matches how
        #                   Mode 1's process_id is also passed through unconstrained
        #                   in ngo_physics_one.py's own SyntheticDataGenerator)
        self.lift_entangled    = nn.Linear(agi_latent_dim, 4)
        self.readout_entangled = nn.Linear(1, agi_latent_dim)

        dev = device or next(physics_model.parameters(), torch.empty(0)).device
        self.register_buffer("_sigma", torch.full((1, 1), 0.5, device=dev))
        self.register_buffer("_collider_probe", torch.ones(1, 4, device=dev))

        # [v3.6 NEW] Fixed CHSH-optimal angle quadruple for the quality probe
        # only (never touches `.encode()`'s latent path) — same default angles
        # as `NGOPhysicsInference.chsh_test()` in ngo_physics_one.py, so the
        # two probes can never disagree about what "the textbook setting" means.
        self._chsh_angles = (0.0, math.pi / 2, math.pi / 4, -math.pi / 4)  # a, a', b, b'

        self._last_quality: float = 0.5
        self._last_chsh_quality: float = 0.5

    def _lift_entangled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Project AGI latent → physically-bounded [θ_a, θ_b, V, source_id]."""
        raw = self.lift_entangled(x)                        # (B, 4)
        theta_a   = torch.tanh(raw[:, 0:1]) * math.pi
        theta_b   = torch.tanh(raw[:, 1:2]) * math.pi
        visibility = torch.sigmoid(raw[:, 2:3])
        source_id  = raw[:, 3:4]
        return torch.cat([theta_a, theta_b, visibility, source_id], dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        sigma = self._sigma.expand(B, -1)

        momentum_data = self.lift(x)                                  # (B, 2)
        d_p2 = self.physics_model.forward_ym(momentum_data, sigma)     # (B, 1)
        latent_ym = self.readout(d_p2)                                 # (B, D_agi)

        # [v3.6 NEW] Mode-4 entangled-pair contribution, blended in via simple
        # mean — same "wholly additive" pattern the BV adapter used for its
        # own Mode-4 wiring: the YM-only latent is still computed exactly as
        # before, just averaged with this new term rather than returned alone.
        ent_feats = self._lift_entangled_features(x)                   # (B, 4)
        E_hat = self.physics_model.forward_entangled(ent_feats, sigma) # (B, 1)
        latent_entangled = self.readout_entangled(E_hat)                # (B, D_agi)

        return 0.5 * (latent_ym + latent_entangled)

    def get_quality_score(self) -> float:
        """
        [v3.4 NEW, extended v3.6] Blends two independent [0, 1] signals:
          (a) Collider-head non-negativity/finiteness probe (unchanged).
          (b) [v3.6 NEW] CHSH physical-realizability probe on Mode 4.
        Averages whichever signals compute successfully; falls back to
        the cached previous value (never crashes the AGI ONE forward pass).
        """
        scores: List[float] = []

        # (a) Collider boundedness probe — unchanged from v3.4.
        try:
            with torch.no_grad():
                sigma = torch.full((1, 1), 0.5, device=self._sigma.device)
                out = self.physics_model.forward_collider(self._collider_probe, sigma)
                val = float(out.mean().item())
            if val != val or math.isinf(val):     # NaN / Inf guard
                collider_score = 0.0
            else:
                collider_score = float(math.exp(-abs(val)))
            collider_score = min(max(collider_score, 0.0), 1.0)
            scores.append(collider_score)
        except Exception as exc:
            logger.debug(f"GNOPhysicsEncoderAdapter.get_quality_score() collider probe fallback: {exc}")

        # (b) [v3.6 NEW] CHSH physical-realizability probe on Mode 4.
        try:
            a, ap, b, bp = self._chsh_angles
            dev = self._sigma.device

            def _feat(t_a: float, t_b: float) -> torch.Tensor:
                return torch.tensor(
                    [[t_a, t_b, 1.0, 0.0]], dtype=torch.float32, device=dev
                )

            was_training = self.physics_model.training
            with torch.no_grad():
                sigma1 = torch.full((1, 1), 0.5, device=dev)
                self.physics_model.eval()
                try:
                    E_ab   = self.physics_model.forward_entangled(_feat(a,  b),  sigma1).item()
                    E_abp  = self.physics_model.forward_entangled(_feat(a,  bp), sigma1).item()
                    E_apb  = self.physics_model.forward_entangled(_feat(ap, b),  sigma1).item()
                    E_apbp = self.physics_model.forward_entangled(_feat(ap, bp), sigma1).item()
                finally:
                    self.physics_model.train(was_training)

            S = E_ab + E_abp + E_apb - E_apbp
            if S != S or math.isinf(S):           # NaN / Inf guard
                chsh_score = 0.0
            else:
                # 1.0 well inside Tsirelson, decaying smoothly past it.
                # Uses TSIRELSON_BOUND (imported from bell_chsh_one.py when
                # available; textbook fallback otherwise — see import block).
                overshoot = max(abs(S) - TSIRELSON_BOUND, 0.0)
                chsh_score = float(math.exp(-overshoot))
            chsh_score = min(max(chsh_score, 0.0), 1.0)
            self._last_chsh_quality = chsh_score
            scores.append(chsh_score)
        except Exception as exc:
            logger.debug(f"GNOPhysicsEncoderAdapter.get_quality_score() CHSH probe fallback: {exc}")
            scores.append(self._last_chsh_quality)

        if not scores:
            return self._last_quality

        self._last_quality = sum(scores) / len(scores)
        return self._last_quality


def attach_ngo_physics_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
    physics_cfg   : Optional["NGOPhysicsConfig"] = None,
    name          : str = "ngo_physics_one",
) -> Optional[GNOPhysicsEncoderAdapter]:
    """[v3.4 NEW, Mode 4 wired v3.6] One-liner registration for NGO Physics ONE's GNOPhysics surrogate."""
    if not HAS_NGO_PHYSICS:
        logger.warning("attach_ngo_physics_to_ecosystem: ngo_physics_one unavailable — skipping registration.")
        return None
    dev = device or torch.device("cpu")
    cfg = physics_cfg or NGOPhysicsConfig()
    model   = StructuralGNOPhysics(cfg).to(dev)
    wrapper = GNOPhysicsEncoderAdapter(model, agi_latent_dim, device=dev).to(dev)
    orchestrator.register(name, wrapper, "physics", agi_latent_dim, quality_fn=wrapper.get_quality_score)
    return wrapper


def attach_all_surrogates_to_ecosystem(
    orchestrator  : "EcosystemOrchestrator",
    agi_latent_dim: int,
    device        : Optional[torch.device] = None,
) -> Dict[str, Optional[nn.Module]]:
    """
    [v3.4 NEW] Convenience one-liner that wires every surrogate this file
    knows how to adapt — the six newly-wired modules plus EVOLUTION ONE
    BV — onto `orchestrator` in one call. Each `attach_*` already
    no-ops gracefully (logs a warning, returns None) if its underlying
    module isn't importable in this environment, so this is always safe
    to call regardless of which surrogate files happen to be present.

        ecosystem = EcosystemOrchestrator(agi_latent_dim=64, device=device)
        wrappers  = attach_all_surrogates_to_ecosystem(ecosystem, 64, device)

    Returns {domain_name: adapter_or_None} for inspection/checkpoint-loading.
    """
    return {
        "physics_sfno3d":   attach_sfno3d_to_ecosystem(orchestrator, agi_latent_dim, device),
        "physics_ngo":      attach_ngo_physics_to_ecosystem(orchestrator, agi_latent_dim, device),
        "mental":           attach_msno_to_ecosystem(orchestrator, agi_latent_dim, device),
        "fold":             attach_gno_fold_to_ecosystem(orchestrator, agi_latent_dim, device),
        "hodge":            attach_gno_hodge_to_ecosystem(orchestrator, agi_latent_dim, device),
        "math_numbertheory":attach_gno_numbertheory_to_ecosystem(orchestrator, agi_latent_dim, device),
        "evolution_bv":     attach_evolution_bv_to_ecosystem(orchestrator, agi_latent_dim, device),
    }


class EcosystemOrchestrator(nn.Module):
    """
    AGI ONE v3 Distributed Ecosystem Hub.

    Manages a registry of SurrogateAdapters (one per uploaded module) and
    provides:
      1. Selective freezing per curriculum phase
      2. Per-domain latent extraction (for GlobalWorkspace / InfoNCE)
      3. Unified parameter groups for Decoupled Optimizers
      4. Health monitoring (NaN guard per domain)

    Domain groups (match AGITrainerV3 optimizer keys):
      'physics'      → structural_fno_3d, ngo_physics_one
                       [v3.4 NEW] real-wired via attach_sfno3d_to_ecosystem()
                       and attach_ngo_physics_to_ecosystem()
                       [v3.6 NEW] ngo_physics_one's Mode 4 (entangled-pair
                       CHSH correlator) now also blended into the latent
                       and quality-probed; see GNOPhysicsEncoderAdapter.
      'fold'         → structural_gno_fold_v4 [v3.5 rewired: sparse radius
                       graph, no API change]
                       [v3.4 NEW] real-wired via attach_gno_fold_to_ecosystem()
      'evolution'    → structural_gno_evolution  (still unwired — no adapter
                       written for the plain, non-BV evolution surrogate yet)
      'evolution_bv' → structural_gno_evolution_bv_standalone  [v3.2 NEW]
      'mental'       → mental_structural_operator_v3
                       [v3.4 NEW] real-wired via attach_msno_to_ecosystem()
      'math'         → structural_gno_numbertheory
                       [v3.4 NEW] real-wired via attach_gno_numbertheory_to_ecosystem()
      'hodge'        → structural_gno_hodge
                       [v3.4 NEW] real-wired via attach_gno_hodge_to_ecosystem()

    [v3.4 NEW] All six adapters above follow the same lift→real-forward→
    readout pattern as GNOEvolutionBVEncoderAdapter (see attach_all_
    surrogates_to_ecosystem() for a one-call convenience wiring every
    domain at once). Each `attach_*_to_ecosystem()` degrades gracefully
    (warns, returns None) if its module isn't importable — never raises.
    """

    DOMAIN_ORDER: List[str] = [
        "physics", "fold", "evolution", "evolution_bv", "mental", "math", "hodge",
    ]

    def __init__(self, agi_latent_dim: int, device: torch.device) -> None:
        super().__init__()
        self.agi_latent_dim = agi_latent_dim
        self.device         = device
        self._adapters: Dict[str, SurrogateAdapter] = {}
        # Learnable projection heads are registered as a ModuleDict
        self._proj_heads = nn.ModuleDict()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        name      : str,
        module    : nn.Module,
        domain    : str,
        latent_dim: int,
        quality_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        """Register a surrogate module under a given domain.

        `quality_fn`  [v3.2 NEW] — optional zero-arg callable (or a plain
        float captured from an external training run, wrapped by the
        caller as `lambda: 0.83`) reporting that surrogate's *simulation*
        quality in [0, 1]. Omit it entirely to reproduce pre-v3.2 behaviour.
        """
        adapter = SurrogateAdapter(module, domain, latent_dim, name, quality_fn=quality_fn)
        adapter.set_projection(self.agi_latent_dim, self.device)
        self._adapters[name] = adapter

        if adapter._projection is not None:
            self._proj_heads[name] = adapter._projection

        logger.info(
            f"EcosystemOrchestrator: registered '{name}'  "
            f"domain={domain}  latent={latent_dim}→{self.agi_latent_dim}  "
            f"quality_fn={'set' if quality_fn is not None else 'none (neutral 0.5)'}"
        )

    # ── Curriculum-phase freeze control ──────────────────────────────────────

    def apply_curriculum_phase(self, phase: int) -> None:
        """
        Control which surrogate parameters are frozen per curriculum phase.

        Phase 1 — FOUNDATION  : all surrogates trainable (they train independently)
        Phase 2 — ALIGNMENT   : freeze surrogate backbones; only proj heads train
        Phase 3 — COGNITIVE   : unfreeze surrogate output heads; backbone stays frozen
        """
        if phase == 1:
            for adp in self._adapters.values():
                adp.freeze_backbone(False)
            logger.info("EcosystemOrchestrator → Phase 1: all surrogates UNFROZEN")

        elif phase == 2:
            for adp in self._adapters.values():
                adp.freeze_backbone(True)      # backbone frozen, heads still live
            logger.info("EcosystemOrchestrator → Phase 2: surrogate backbones FROZEN")

        elif phase == 3:
            for adp in self._adapters.values():
                adp.freeze_backbone(True)      # backbones stay frozen
            # Projection heads remain trainable (they are in self._proj_heads)
            logger.info(
                "EcosystemOrchestrator → Phase 3: backbones FROZEN, heads TRAINABLE"
            )

    # ── Domain-grouped parameter lists ───────────────────────────────────────

    def param_groups_by_domain(self) -> Dict[str, List[nn.Parameter]]:
        """Return {domain: [trainable params]} for Decoupled Optimizers."""
        groups: Dict[str, List[nn.Parameter]] = {d: [] for d in self.DOMAIN_ORDER}
        for name, adp in self._adapters.items():
            domain = adp.domain if adp.domain in groups else "physics"
            groups[domain].extend(adp.get_trainable_params())
        return {k: v for k, v in groups.items() if v}

    # ── Forward: extract per-domain latents ──────────────────────────────────

    def forward(
        self,
        perception_latent: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Query all registered surrogates with the AGI perception latent.

        Returns dict {name: latent_tensor (B, D_agi)} for each surrogate,
        plus an aggregated 'ecosystem' key (mean-pooled across all domains).
        """
        outputs: Dict[str, torch.Tensor] = {}
        domain_latents: Dict[str, List[torch.Tensor]] = {
            d: [] for d in self.DOMAIN_ORDER
        }

        for name, adp in self._adapters.items():
            latent = adp.encode(perception_latent)

            # NaN guard
            if torch.isnan(latent).any():
                logger.warning(
                    f"EcosystemOrchestrator: NaN in surrogate '{name}' — zeroed"
                )
                latent = torch.zeros_like(latent)

            outputs[name] = latent
            dom = adp.domain if adp.domain in domain_latents else "physics"
            domain_latents[dom].append(latent)

        # Per-domain mean (for InfoNCE and GlobalWorkspace)
        for dom, tensors in domain_latents.items():
            if tensors:
                outputs[f"domain_{dom}"] = torch.stack(tensors, dim=0).mean(0)

        # Global ecosystem latent (mean across all registered surrogates)
        if outputs:
            all_latents = [v for k, v in outputs.items() if not k.startswith("domain_")]
            if all_latents:
                outputs["ecosystem"] = torch.stack(all_latents, dim=0).mean(0)

        return outputs

    def registered_names(self) -> List[str]:
        return list(self._adapters.keys())

    # ── [v3.2 NEW] Simulation-quality reporting ──────────────────────────────

    def quality_report(self) -> Dict[str, float]:
        """
        Aggregate `get_quality_score()` across all registered surrogates.

        Returns a plain-float dict (no grad, safe to log/feed to non-
        differentiable downstream consumers like Psyche Plus's codify gate):
          { "<name>": score, "domain_<domain>": mean-over-domain,
            "ecosystem": mean-over-everything }

        This is intentionally a *separate* method from `forward()` rather
        than folded into its tensor outputs — quality scores are scalars
        sourced from non-differentiable, possibly-stale external metrics
        (a training-loop loss value, a BV-3 certification residual), so
        mixing them into the differentiable latent dict would be a type
        smell. Domains with no registered surrogate, or whose surrogates
        all report the neutral default, simply average to 0.5.
        """
        per_name: Dict[str, float] = {}
        per_domain: Dict[str, List[float]] = {d: [] for d in self.DOMAIN_ORDER}

        for name, adp in self._adapters.items():
            q = adp.get_quality_score()
            per_name[name] = q
            dom = adp.domain if adp.domain in per_domain else "physics"
            per_domain[dom].append(q)

        report: Dict[str, float] = dict(per_name)
        for dom, scores in per_domain.items():
            if scores:
                report[f"domain_{dom}"] = sum(scores) / len(scores)

        if per_name:
            report["ecosystem"] = sum(per_name.values()) / len(per_name)

        return report

    def __repr__(self) -> str:
        lines = [f"EcosystemOrchestrator(agi_latent={self.agi_latent_dim})"]
        for name, adp in self._adapters.items():
            frozen = "frozen-bb" if adp.frozen_backbone else "trainable"
            lines.append(f"  [{adp.domain:12s}]  {name}  ({frozen})")
        return "\n".join(lines)


# =============================================================================
# SECTION 18-B — PCGrad GRADIENT SURGERY  (v3.0 NEW)
# Yu et al. 2020 "Gradient Surgery for Multi-Task Learning"
# =============================================================================

class PCGradOptimizer:
    """
    PCGrad (Projecting Conflicting Gradients) wrapper.

    Wraps an existing torch.optim.Optimizer.  On each step:
      1. Compute per-task gradients individually (no accumulation)
      2. For each task pair (i, j): if g_i · g_j < 0, project g_i onto the
         normal plane of g_j  →  g_i' = g_i - (g_i·g_j / ‖g_j‖²) g_j
      3. Sum projected gradients and load into .grad buffers
      4. Call wrapped optimizer.step()

    Usage:
        pcgrad = PCGradOptimizer(torch.optim.AdamW(params, lr=1e-4))
        losses = [loss_physics, loss_language, loss_rl]
        pcgrad.step(losses, retain_graph=False)

    Note: losses must each be separately computable (no joint backward).
          If you have already summed them, PCGrad has no effect.
    """

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._opt    = optimizer
        self._params = [
            p for group in optimizer.param_groups
            for p in group["params"] if p.requires_grad
        ]

    def zero_grad(self) -> None:
        self._opt.zero_grad()

    def _flatten_grad(self) -> torch.Tensor:
        grads = []
        for p in self._params:
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)                           # (total_params,)

    def _set_flat_grad(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self._params:
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(flat[offset: offset + n].view_as(p))
            offset += n

    def step(
        self,
        task_losses: List[torch.Tensor],
        retain_graph: bool = False,
    ) -> None:
        """
        Perform PCGrad update given a list of per-task scalar losses.

        Each loss is backpropagated independently to obtain its gradient
        vector.  Conflicting gradient pairs are then projected before
        the final parameter update.
        """
        n_tasks = len(task_losses)
        if n_tasks == 0:
            return

        # ── Step 1: Collect per-task flat gradient vectors ────────────────────
        task_grads: List[torch.Tensor] = []
        for i, loss in enumerate(task_losses):
            self._opt.zero_grad()
            retain = retain_graph or (i < n_tasks - 1)
            loss.backward(retain_graph=retain)
            task_grads.append(self._flatten_grad())

        # ── Step 2: Project conflicting gradients ─────────────────────────────
        proj_grads = [g.clone() for g in task_grads]

        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                gi = proj_grads[i]
                gj = task_grads[j]
                dot = torch.dot(gi, gj)
                if dot < 0:                              # conflicting direction
                    norm_sq = gj.dot(gj).clamp(min=1e-12)
                    proj_grads[i] = gi - (dot / norm_sq) * gj

        # ── Step 3: Sum projected gradients and load into .grad ───────────────
        final_grad = torch.stack(proj_grads, dim=0).sum(dim=0)
        self._opt.zero_grad()
        self._set_flat_grad(final_grad)

        # ── Step 4: Optimizer step ────────────────────────────────────────────
        self._opt.step()

    def state_dict(self) -> Dict:
        return self._opt.state_dict()

    def load_state_dict(self, sd: Dict) -> None:
        self._opt.load_state_dict(sd)

    @property
    def param_groups(self):
        return self._opt.param_groups


# =============================================================================
# SECTION 18-C — InfoNCE CROSS-MODAL ALIGNMENT LOSS  (v3.0 NEW)
# CLIP-style contrastive alignment: physics latent ↔ language latent
# Radford et al. 2021 / Chen et al. 2020 (SimCLR)
# =============================================================================

class CrossModalAlignmentLoss(nn.Module):
    """
    Symmetric InfoNCE (NT-Xent) loss aligning two latent spaces.

    Given a batch of B paired embeddings from two domains (e.g. physics and
    language), the loss pulls (physics_i, language_i) pairs together and
    pushes (physics_i, language_j≠i) apart in the shared embedding space.

    L = -½ [ mean_i log softmax(sim(p_i, L)/τ)[i]    # physics→language
            + mean_i log softmax(sim(L_i, p)/τ)[i] ]  # language→physics

    where sim(a, b) = a·b / (‖a‖·‖b‖)  (cosine similarity)

    Usage:
        align_loss = CrossModalAlignmentLoss(d_model=512, temperature=0.07)
        loss = align_loss(physics_latent, language_latent)

    If batch size B == 1, returns zero (no negatives to contrast against).
    Both tensors must be (B, D).
    """

    def __init__(
        self,
        d_model     : int,
        temperature : float = 0.07,
        learnable_T : bool  = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Learnable temperature (log-parameterised for numerical stability)
        if learnable_T:
            self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_tau", torch.tensor(math.log(temperature)))

        # Shared projection MLP: both domains projected to alignment space
        self.proj_physics  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_language = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        physics_latent : torch.Tensor,   # (B, D)
        language_latent: torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        B = physics_latent.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=physics_latent.device, requires_grad=True)

        p = F.normalize(self.proj_physics(physics_latent),  dim=-1)  # (B, D)
        l = F.normalize(self.proj_language(language_latent), dim=-1)  # (B, D)

        tau = self.log_tau.exp().clamp(min=1e-4, max=1.0)

        # Cosine similarity matrix (B, B)
        logits_pl = (p @ l.T) / tau        # physics-rows  → language-cols
        logits_lp = (l @ p.T) / tau        # language-rows → physics-cols

        labels = torch.arange(B, device=physics_latent.device)

        loss_pl = F.cross_entropy(logits_pl, labels)
        loss_lp = F.cross_entropy(logits_lp, labels)

        return 0.5 * (loss_pl + loss_lp)


# =============================================================================
# SECTION 18-D — CURRICULUM SCHEDULER  (v3.0 NEW)
# Three-phase curriculum control for AGITrainerV3
# =============================================================================

class TrainingPhase(Enum):
    """Curriculum training phase."""
    FOUNDATION = 1   # Physics/math surrogates + DreamerV3 world model
    ALIGNMENT  = 2   # Freeze physics backbone; language↔physics InfoNCE bridge
    COGNITIVE  = 3   # PPO + Psyche; fine-tune with PCGrad; full policy


@dataclass
class CurriculumConfig:
    """
    Configuration for three-phase curriculum training.

    Attributes:
        phase1_steps : Steps to train in FOUNDATION phase.
        phase2_steps : Steps to train in ALIGNMENT phase.
        phase3_steps : Steps to train in COGNITIVE phase (or -1 = indefinite).
        align_loss_weight  : InfoNCE alignment loss weight (Phase 2+3).
        pcgrad_enabled     : Whether to use PCGrad in Phase 3.
        phase1_convergence_patience : Switch to Phase 2 early if world-model
                                      loss has not improved for this many steps.
    """
    phase1_steps               : int   = 5_000
    phase2_steps               : int   = 5_000
    phase3_steps               : int   = -1       # indefinite
    align_loss_weight          : float = 0.1
    pcgrad_enabled             : bool  = True
    phase1_convergence_patience: int   = 500


class CurriculumScheduler:
    """
    Manages curriculum phase transitions for AGITrainerV3.

    Tracks global step count and fires phase transition callbacks.
    Also implements early-switch to Phase 2 if the world-model loss
    has converged (patience mechanism).

    Usage:
        scheduler = CurriculumScheduler(curriculum_cfg, orchestrator)
        # In trainer loop:
        scheduler.on_step(global_step, loss_metrics)
        phase = scheduler.current_phase
    """

    def __init__(
        self,
        cfg         : CurriculumConfig,
        orchestrator: EcosystemOrchestrator,
    ) -> None:
        self.cfg           = cfg
        self.orchestrator  = orchestrator
        self._phase        = TrainingPhase.FOUNDATION
        self._best_wm_loss = float("inf")
        self._patience_ctr = 0
        self._callbacks: Dict[TrainingPhase, List[Callable]] = {
            TrainingPhase.FOUNDATION: [],
            TrainingPhase.ALIGNMENT : [],
            TrainingPhase.COGNITIVE : [],
        }
        # Apply Phase 1 immediately
        self._enter_phase(TrainingPhase.FOUNDATION)

    # ── Phase transition ──────────────────────────────────────────────────────

    def _enter_phase(self, phase: TrainingPhase) -> None:
        self._phase = phase
        self.orchestrator.apply_curriculum_phase(phase.value)
        for cb in self._callbacks.get(phase, []):
            cb()
        logger.info(
            f"CurriculumScheduler ══► Phase {phase.value} ({phase.name}) STARTED"
        )

    def register_callback(self, phase: TrainingPhase, fn: Callable) -> None:
        """Register a callback fired when transitioning into a phase."""
        self._callbacks[phase].append(fn)

    # ── Per-step update ───────────────────────────────────────────────────────

    def on_step(
        self,
        global_step  : int,
        loss_metrics : Dict[str, float],
    ) -> TrainingPhase:
        """
        Update phase based on step count and optional early convergence.
        Returns the current phase after any transition.
        """
        p1_end = self.cfg.phase1_steps
        p2_end = p1_end + self.cfg.phase2_steps

        # ── Phase 1 → 2 transition ────────────────────────────────────────────
        if self._phase == TrainingPhase.FOUNDATION:
            wm_loss = loss_metrics.get("world_kl", loss_metrics.get("total_loss", 1e9))

            # Early convergence check (patience)
            if wm_loss < self._best_wm_loss - 1e-4:
                self._best_wm_loss = wm_loss
                self._patience_ctr = 0
            else:
                self._patience_ctr += 1

            early_converge = (
                self._patience_ctr >= self.cfg.phase1_convergence_patience
            )
            if global_step >= p1_end or early_converge:
                reason = "early-convergence" if early_converge else "step-limit"
                logger.info(f"CurriculumScheduler: Phase 1 exit [{reason}]")
                self._enter_phase(TrainingPhase.ALIGNMENT)
                self._patience_ctr = 0

        # ── Phase 2 → 3 transition ────────────────────────────────────────────
        elif self._phase == TrainingPhase.ALIGNMENT:
            if global_step >= p2_end:
                self._enter_phase(TrainingPhase.COGNITIVE)

        # Phase 3 runs until manually stopped (or phase3_steps if set)
        elif self._phase == TrainingPhase.COGNITIVE:
            if (
                self.cfg.phase3_steps > 0
                and global_step >= p2_end + self.cfg.phase3_steps
            ):
                logger.info("CurriculumScheduler: Phase 3 complete.")

        return self._phase

    @property
    def current_phase(self) -> TrainingPhase:
        return self._phase

    def is_phase(self, phase: TrainingPhase) -> bool:
        return self._phase == phase

    def phase_name(self) -> str:
        return self._phase.name


# =============================================================================
# SECTION 18 — AGI TRAINER v2.0
# Full Dreamer + PPO + PSY compound training
# =============================================================================

class AGITrainer:
    """
    AGI ONE v2.0 Unified Trainer.

    Combines:
    [1] DreamerV3 world model training
        - KL (free-bits) + reconstruction + reward (two-hot) + continue
    [2] PPO actor-critic (on imagined trajectories from world model)
    [3] PSY BRIDGE Free Energy minimization
    [4] Language modelling auxiliary loss
    [5] Kendall uncertainty-weighted loss balancing
    [6] AMP mixed precision
    [7] Gradient clipping
    [8] Warmup + cosine LR scheduling
    [9] DDP-ready (call torch.nn.parallel.DistributedDataParallel externally)
    """

    def __init__(
        self,
        model : AGIONE,
        cfg   : Optional[AGIConfig] = None,
    ) -> None:
        self.model  = model
        self.cfg    = cfg or model.cfg
        self.device = model.device

        # Separate optimizer for world model vs policy vs loss balancer
        self.opt_world = torch.optim.AdamW(
            list(model.world_model.parameters()),
            lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )
        self.opt_policy = torch.optim.AdamW(
            list(model.actor_net.parameters()) +
            list(model.critic_net.parameters()) +
            list(model.psyche_exec.parameters()),
            lr=self.cfg.lr * 3,   # policy learns faster
            weight_decay=self.cfg.weight_decay,
        )
        self.opt_balance = torch.optim.Adam(
            model.loss_balancer.parameters(), lr=1e-3
        )
        self.opt_perception = torch.optim.AdamW(
            list(model.perception.parameters()),
            lr=self.cfg.lr * 0.3,  # perception slower
        )

        # Warmup + cosine LR
        def warmup_cosine(step):
            w = self.cfg.warmup_steps
            if step < w:
                return float(step) / max(w, 1)
            progress = (step - w) / max(1, 10_000 - w)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(o, warmup_cosine)
            for o in [self.opt_world, self.opt_policy,
                      self.opt_balance, self.opt_perception]
        ]

        self.scaler = GradScaler(
            enabled=self.cfg.use_amp and torch.cuda.is_available()
        )

        self.train_stats: Dict[str, List[float]] = {
            "total_loss": [], "kl": [], "recon": [],
            "reward_loss": [], "actor": [], "value": [],
            "entropy": [], "psy": [],
        }
        self._global_step = 0

        logger.info(
            f"AGITrainer v2.0 initialized  |  "
            f"warmup={self.cfg.warmup_steps}  amp={self.cfg.use_amp}"
        )

    def step(
        self,
        observation: Dict[str, Optional[torch.Tensor]],
        reward     : Optional[torch.Tensor] = None,
        done       : bool = False,
    ) -> Dict[str, float]:
        """
        Single training step.

        Returns stats dict.
        """
        self.model.train()

        for opt in [self.opt_world, self.opt_policy,
                    self.opt_balance, self.opt_perception]:
            opt.zero_grad()

        with autocast(enabled=self.cfg.use_amp and torch.cuda.is_available()):
            agi_state = self.model(
                **observation,
                compute_loss = True,
                reward       = reward,
            )

        if agi_state.total_loss is None or not agi_state.total_loss.requires_grad:
            return {"loss": 0.0}

        self.scaler.scale(agi_state.total_loss).backward()

        for opt in [self.opt_world, self.opt_policy,
                    self.opt_balance, self.opt_perception]:
            self.scaler.unscale_(opt)

        all_params = list(self.model.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, self.cfg.grad_clip_norm)

        for opt in [self.opt_world, self.opt_policy,
                    self.opt_balance, self.opt_perception]:
            self.scaler.step(opt)

        self.scaler.update()

        for sched in self.schedulers:
            sched.step()

        self._global_step += 1

        # PPO buffer update
        if agi_state.workspace_state is not None and reward is not None:
            ws = agi_state.workspace_state.detach()
            self.model.ppo.buffer.obs.append(ws)
            if agi_state.planned_action is not None:
                self.model.ppo.buffer.actions.append(
                    agi_state.planned_action.detach()
                )
            self.model.ppo.buffer.rewards.append(float(reward.item()))
            val = float(self.model.critic_net(ws).detach().item())
            self.model.ppo.buffer.values.append(val)
            act_logits = self.model.actor_net(ws)
            lp = F.log_softmax(act_logits, dim=-1)
            self.model.ppo.buffer.log_probs.append(lp.max().detach())
            self.model.ppo.buffer.dones.append(done)

            # Run PPO update every 128 steps
            if len(self.model.ppo.buffer) >= 128:
                ppo_stats = self.model.ppo.update()
                logger.debug(f"PPO update: {ppo_stats}")

        loss_val = float(agi_state.total_loss.detach().item())
        self.train_stats["total_loss"].append(loss_val)

        return {
            "loss"          : loss_val,
            "winner_module" : agi_state.winner_module,
            "safety_score"  : agi_state.safety_score or 0.0,
            "csoc_n_layers" : agi_state.csoc_n_layers or 0,
            "global_step"   : self._global_step,
            "lr_world"      : self.schedulers[0].get_last_lr()[0],
        }

    def train(
        self,
        observations : List[Dict],
        rewards      : Optional[List] = None,
        n_steps      : int = 1000,
        eval_every   : int = 100,
    ) -> None:
        logger.info(f"AGITrainer v2.0: training for {n_steps} steps")
        for step in range(n_steps):
            idx    = step % len(observations)
            obs    = observations[idx]
            rwd    = torch.tensor(rewards[idx]) if rewards else None
            stats  = self.step(obs, rwd)

            if step % eval_every == 0:
                logger.info(
                    f"Step {step:5d}  loss={stats.get('loss',0):.4f}  "
                    f"winner={stats.get('winner_module','?')}  "
                    f"safety={stats.get('safety_score',0):.3f}  "
                    f"csoc_layers={stats.get('csoc_n_layers',0)}  "
                    f"lr={stats.get('lr_world',0):.2e}"
                )
        logger.info("AGITrainer v2.0: training complete")

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "model_state"     : self.model.state_dict(),
            "opt_world"       : self.opt_world.state_dict(),
            "opt_policy"      : self.opt_policy.state_dict(),
            "opt_balance"     : self.opt_balance.state_dict(),
            "train_stats"     : self.train_stats,
            "global_step"     : self._global_step,
            "agi_version"     : AGI_ONE_VERSION,
            "science_registry": self.model.science_registry.provenance_report(),
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.opt_world.load_state_dict(ckpt["opt_world"])
        self.opt_policy.load_state_dict(ckpt["opt_policy"])
        self.opt_balance.load_state_dict(ckpt["opt_balance"])
        self.train_stats   = ckpt.get("train_stats", self.train_stats)
        self._global_step  = ckpt.get("global_step", 0)
        logger.info(f"Checkpoint loaded: {path}  (step={self._global_step})")


# =============================================================================
# SECTION 18-E — AGI TRAINER v3.0  (v3.0 NEW)
# Distributed Ecosystem + Curriculum + PCGrad + InfoNCE + Decoupled Optimizers
# =============================================================================

class AGITrainerV3:
    """
    AGI ONE v3.0 Unified Trainer — Stable Multi-Task Learning.

    Solves the Gradient Interference problem of v2.0 via:

    [A] CurriculumScheduler   — 3-phase staged training
    [B] EcosystemOrchestrator — surrogate modules trained/frozen independently
    [C] CrossModalAlignmentLoss — InfoNCE physics↔language geometric alignment
    [D] PCGradOptimizer       — gradient surgery for conflicting tasks (Phase 3)
    [E] Decoupled Optimizers  — per-domain learning rates

    Optimizer map (Decoupled Optimizers):
    ─────────────────────────────────────────────────────────────────────────
    Group              Optimizer   LR       Rationale
    ─────────────────────────────────────────────────────────────────────────
    physics_surrogates AdamW       1e-6     High-precision PDE; very sensitive
    math_surrogates    AdamW       5e-7     Discrete logic; extreme sensitivity
    language_module    AdamW       1e-5     Standard LLM fine-tuning regime
    world_model        AdamW       3e-4     DreamerV3 nominal
    policy_heads       AdamW       1e-4     PPO actor + critic
    psyche_layer       AdamW       3e-4     Free Energy minimisation
    align_proj         AdamW       1e-4     InfoNCE projection heads
    loss_balancer      Adam        1e-3     Kendall σ params (fast convergence)
    ─────────────────────────────────────────────────────────────────────────

    Phase-specific active optimizers:
      Phase 1 — FOUNDATION : world_model + (ecosystem surrogates, independent)
      Phase 2 — ALIGNMENT  : align_proj + language_module
      Phase 3 — COGNITIVE  : policy_heads + psyche_layer (+ PCGrad across all)
    """

    def __init__(
        self,
        model           : "AGIONE",
        orchestrator    : EcosystemOrchestrator,
        cfg             : Optional[AGIConfig]      = None,
        curriculum_cfg  : Optional[CurriculumConfig] = None,
    ) -> None:
        self.model        = model
        self.orchestrator = orchestrator
        self.cfg          = cfg or model.cfg
        self.device       = model.device

        # [v3.2 NEW] Unify the trainer's orchestrator with the model's
        # optional back-reference so Psyche Plus (Step 7-B, inside
        # `model.forward()`) can read live `quality_report()` scores
        # without AGITrainerV3 having to thread them through manually
        # on every call. Safe no-op if `model` predates `attach_orchestrator`.
        if hasattr(model, "attach_orchestrator"):
            model.attach_orchestrator(orchestrator)

        curriculum_cfg = curriculum_cfg or CurriculumConfig()

        # ── Cross-modal alignment loss ─────────────────────────────────────────
        self.align_loss = CrossModalAlignmentLoss(
            d_model     = self.cfg.latent_dim,
            temperature = 0.07,
            learnable_T = True,
        ).to(self.device)

        # ── Curriculum scheduler ───────────────────────────────────────────────
        self.curriculum = CurriculumScheduler(curriculum_cfg, orchestrator)

        # ── Decoupled Optimizers ──────────────────────────────────────────────
        # World model
        self.opt_world = torch.optim.AdamW(
            list(model.world_model.parameters()),
            lr=3e-4, weight_decay=self.cfg.weight_decay,
        )
        # Policy heads (actor + critic + MPPI)
        self.opt_policy = torch.optim.AdamW(
            list(model.actor_net.parameters()) +
            list(model.critic_net.parameters()),
            lr=1e-4, weight_decay=self.cfg.weight_decay,
        )
        # Psyche Executive (Free Energy) + Psyche Plus  [v3.1: psyche_plus joins]
        psyche_params = list(model.psyche_exec.parameters())
        if getattr(model, "psyche_plus", None) is not None:
            psyche_params += list(model.psyche_plus.parameters())
        self.opt_psyche = torch.optim.AdamW(
            psyche_params, lr=3e-4, weight_decay=self.cfg.weight_decay,
        )
        # Language module (standard LLM fine-tune regime)
        lang_params = list(model.language.parameters()) if hasattr(model, "language") else []
        self.opt_language = torch.optim.AdamW(
            lang_params, lr=1e-5, weight_decay=self.cfg.weight_decay,
        ) if lang_params else None

        # InfoNCE alignment projections
        self.opt_align = torch.optim.AdamW(
            list(self.align_loss.parameters()), lr=1e-4,
        )

        # Loss balancer (Kendall σ)
        self.opt_balance = torch.optim.Adam(
            model.loss_balancer.parameters(), lr=1e-3,
        )

        # Perception (slow)
        self.opt_perception = torch.optim.AdamW(
            list(model.perception.parameters()),
            lr=self.cfg.lr * 0.3,
        )

        # Domain surrogate optimizers (per-domain LR via EcosystemOrchestrator)
        self._domain_opts: Dict[str, torch.optim.Optimizer] = {}
        domain_lr = {
            "physics"      : 1e-6,
            "fold"         : 1e-6,
            "evolution"    : 1e-6,
            "evolution_bv" : 1e-6,   # [v3.2 NEW] zero new params, but keep
                                     # the slot consistent for any future
                                     # trainable additions on this domain
            "mental"       : 1e-6,
            "math"         : 5e-7,
            "hodge"        : 5e-7,
        }
        param_groups = orchestrator.param_groups_by_domain()
        for domain, params in param_groups.items():
            if params:
                lr = domain_lr.get(domain, 1e-6)
                self._domain_opts[domain] = torch.optim.AdamW(
                    params, lr=lr, weight_decay=self.cfg.weight_decay,
                )
                logger.info(
                    f"AGITrainerV3: domain optimizer '{domain}'  "
                    f"lr={lr}  params={len(params)}"
                )

        # ── PCGrad wrapper (Phase 3) — wraps policy optimizer ─────────────────
        self.pcgrad = PCGradOptimizer(self.opt_policy)

        # ── Warmup + Cosine LR schedulers ────────────────────────────────────
        def _warmup_cos(step: int) -> float:
            w = self.cfg.warmup_steps
            if step < w:
                return float(step) / max(w, 1)
            progress = (step - w) / max(1, 50_000 - w)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        _base_opts = [
            self.opt_world, self.opt_policy, self.opt_psyche,
            self.opt_balance, self.opt_perception,
        ]
        if self.opt_language:
            _base_opts.append(self.opt_language)

        self._schedulers = [
            torch.optim.lr_scheduler.LambdaLR(o, _warmup_cos) for o in _base_opts
        ]

        # ── AMP scaler ────────────────────────────────────────────────────────
        self.scaler = GradScaler(
            enabled=self.cfg.use_amp and torch.cuda.is_available()
        )

        # ── Stats ─────────────────────────────────────────────────────────────
        self.train_stats: Dict[str, List[float]] = {
            "total_loss": [], "align_loss": [], "world_kl": [],
            "actor": [], "value": [], "psy": [], "phase": [],
        }
        self._global_step = 0

        logger.info(
            f"AGITrainerV3 initialized  |  "
            f"curriculum={curriculum_cfg}  "
            f"pcgrad={curriculum_cfg.pcgrad_enabled}  "
            f"amp={self.cfg.use_amp}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _all_base_opts(self) -> List[torch.optim.Optimizer]:
        opts = [
            self.opt_world, self.opt_policy, self.opt_psyche,
            self.opt_balance, self.opt_perception, self.opt_align,
        ]
        if self.opt_language:
            opts.append(self.opt_language)
        return opts

    def _zero_all(self) -> None:
        for opt in self._all_base_opts():
            opt.zero_grad()
        for opt in self._domain_opts.values():
            opt.zero_grad()

    def _clip_and_step(self, opts: List[torch.optim.Optimizer]) -> None:
        for opt in opts:
            for group in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(
                    group["params"], self.cfg.grad_clip_norm
                )
        for opt in opts:
            self.scaler.step(opt)

    # ── Phase-specific training steps ─────────────────────────────────────────

    def _step_phase1(
        self,
        observation: Dict[str, Optional[torch.Tensor]],
        reward     : Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Phase 1 — FOUNDATION
        Train: DreamerV3 world model + perception encoder.
        Ecosystem surrogates train independently via their own optimizers.
        """
        self.model.train()
        self._zero_all()

        with autocast(enabled=self.cfg.use_amp and torch.cuda.is_available()):
            agi_state = self.model(
                **observation, compute_loss=True, reward=reward,
            )

        if agi_state.total_loss is None or not agi_state.total_loss.requires_grad:
            return {"loss": 0.0, "phase": 1}

        self.scaler.scale(agi_state.total_loss).backward()

        # Only step world model + perception in Phase 1
        active_opts = [self.opt_world, self.opt_perception, self.opt_balance]
        for opt in active_opts:
            self.scaler.unscale_(opt)
        self._clip_and_step(active_opts)
        self.scaler.update()

        # Also step domain surrogate optimizers (independent surrogate training)
        for opt in self._domain_opts.values():
            opt.step()

        return {
            "loss"      : float(agi_state.total_loss.detach().item()),
            "phase"     : 1,
            "world_kl"  : float(agi_state.total_loss.detach().item()),
        }

    def _step_phase2(
        self,
        observation: Dict[str, Optional[torch.Tensor]],
        reward     : Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Phase 2 — ALIGNMENT
        Train: InfoNCE cross-modal alignment (physics↔language).
        Surrogate backbones are frozen; only projection heads + language module update.
        """
        self.model.train()
        self._zero_all()

        with autocast(enabled=self.cfg.use_amp and torch.cuda.is_available()):
            agi_state = self.model(**observation, compute_loss=False, reward=reward)

            # Get physics and language latents for InfoNCE
            ecosystem_latents = self.orchestrator(agi_state.perception_latent)
            physics_latent = ecosystem_latents.get(
                "domain_physics",
                ecosystem_latents.get("ecosystem", agi_state.perception_latent),
            )
            lang_latent = (
                agi_state.language_latent
                if agi_state.language_latent is not None
                else agi_state.perception_latent
            )

            # InfoNCE loss: align physics representation ↔ language representation
            align_loss = self.align_loss(physics_latent, lang_latent)
            align_loss = align_loss * self.curriculum.cfg.align_loss_weight

        self.scaler.scale(align_loss).backward()

        # Only update alignment projections + language module in Phase 2
        active_opts = [self.opt_align]
        if self.opt_language:
            active_opts.append(self.opt_language)
        for opt in active_opts:
            self.scaler.unscale_(opt)
        self._clip_and_step(active_opts)
        self.scaler.update()

        align_val = float(align_loss.detach().item())
        return {
            "loss"       : align_val,
            "align_loss" : align_val,
            "phase"      : 2,
        }

    def _step_phase3(
        self,
        observation: Dict[str, Optional[torch.Tensor]],
        reward     : Optional[torch.Tensor],
        done       : bool,
    ) -> Dict[str, float]:
        """
        Phase 3 — COGNITIVE
        Train: PPO policy + PsycheExecutiveLayer with PCGrad gradient surgery.
        Alignment loss maintained; world model fine-tuned at slow rate.

        Loss taxonomy for PCGrad task separation:
          task_losses[0] = world_model_loss   (DreamerV3 KL + recon)
          task_losses[1] = policy_loss        (PPO actor + value)
          task_losses[2] = psyche_loss        (Free Energy)
          task_losses[3] = align_loss         (InfoNCE — optional if batch>1)
        """
        self.model.train()

        # ── Collect per-task losses (separate backward passes for PCGrad) ──────
        task_losses: List[torch.Tensor] = []
        stats: Dict[str, float]         = {"phase": 3}

        with autocast(enabled=self.cfg.use_amp and torch.cuda.is_available()):
            agi_state = self.model(
                **observation, compute_loss=True, reward=reward,
            )

        # Task 0: World model loss
        if (
            agi_state.total_loss is not None
            and agi_state.total_loss.requires_grad
        ):
            task_losses.append(agi_state.total_loss)
            stats["world_loss"] = float(agi_state.total_loss.detach().item())

        # Task 1: Policy loss (actor + value from PPO buffer)
        if agi_state.workspace_state is not None and reward is not None:
            ws  = agi_state.workspace_state
            val = self.model.critic_net(ws).squeeze()
            rwd = reward.to(self.device).squeeze()

            act_logits = self.model.actor_net(ws)
            dist       = torch.distributions.Categorical(logits=act_logits)
            if agi_state.planned_action is not None:
                log_p = dist.log_prob(agi_state.planned_action.argmax())
                adv   = (rwd - val.detach()).clamp(-10, 10)
                actor_loss = -(log_p * adv)
                value_loss = F.mse_loss(val, rwd.unsqueeze(0))
                entropy_loss = -self.cfg.entropy_coef * dist.entropy()
                policy_loss  = actor_loss + self.cfg.value_loss_coef * value_loss + entropy_loss
                task_losses.append(policy_loss)
                stats["policy_loss"] = float(policy_loss.detach().item())

        # Task 2: Psyche Free Energy
        if agi_state.workspace_state is not None:
            exec_out  = self.model.psyche_exec(agi_state.workspace_state)
            psy_loss  = exec_out.get("psy_loss")
            if psy_loss is not None and psy_loss.requires_grad:
                task_losses.append(psy_loss)
                stats["psy_loss"] = float(psy_loss.detach().item())

        # Task 3: Alignment loss (InfoNCE — only if batch > 1 possible)
        if agi_state.language_latent is not None:
            eco_out   = self.orchestrator(agi_state.perception_latent)
            phys_lat  = eco_out.get("domain_physics", agi_state.perception_latent)
            align_val = self.align_loss(phys_lat, agi_state.language_latent)
            align_val = align_val * self.curriculum.cfg.align_loss_weight
            if align_val.requires_grad:
                task_losses.append(align_val)
                stats["align_loss"] = float(align_val.detach().item())

        if not task_losses:
            return {"loss": 0.0, **stats}

        # ── PCGrad or plain joint backward ────────────────────────────────────
        if self.curriculum.cfg.pcgrad_enabled and len(task_losses) > 1:
            # PCGrad handles its own zero_grad + backward + step internally
            self.pcgrad.zero_grad()
            self.pcgrad.step(task_losses, retain_graph=False)

            # Also step psyche, world model (with standard backward already done)
            for opt in [self.opt_world, self.opt_psyche, self.opt_balance, self.opt_align]:
                opt.zero_grad()
                if task_losses:
                    # Partial update: reuse already-computed gradients if available
                    opt.step()

        else:
            # Fallback: weighted sum (standard backward)
            self._zero_all()
            total = sum(task_losses)
            self.scaler.scale(total).backward()
            active_opts = [
                self.opt_world, self.opt_policy, self.opt_psyche,
                self.opt_balance, self.opt_align,
            ]
            if self.opt_language:
                active_opts.append(self.opt_language)
            for opt in active_opts:
                self.scaler.unscale_(opt)
            self._clip_and_step(active_opts)
            self.scaler.update()

        # PPO buffer update
        if agi_state.workspace_state is not None and reward is not None:
            ws = agi_state.workspace_state.detach()
            self.model.ppo.buffer.obs.append(ws)
            if agi_state.planned_action is not None:
                self.model.ppo.buffer.actions.append(agi_state.planned_action.detach())
            self.model.ppo.buffer.rewards.append(float(reward.item()))
            self.model.ppo.buffer.values.append(
                float(self.model.critic_net(ws).detach().item())
            )
            act_logits = self.model.actor_net(ws)
            lp = F.log_softmax(act_logits, dim=-1)
            self.model.ppo.buffer.log_probs.append(lp.max().detach())
            self.model.ppo.buffer.dones.append(done)
            if len(self.model.ppo.buffer) >= 128:
                self.model.ppo.update()

        total_loss_val = sum(
            float(l.detach().item()) for l in task_losses
        )
        stats["loss"] = total_loss_val
        return stats

    # ── Main training step ────────────────────────────────────────────────────

    def step(
        self,
        observation: Dict[str, Optional[torch.Tensor]],
        reward     : Optional[torch.Tensor] = None,
        done       : bool = False,
    ) -> Dict[str, float]:
        """
        Single unified training step.  Curriculum phase is determined
        automatically; appropriate optimizer subset is activated.
        """
        # Determine current phase
        phase = self.curriculum.on_step(
            self._global_step,
            {k: v[-1] for k, v in self.train_stats.items() if v},
        )

        # Step LR schedulers
        for sched in self._schedulers:
            sched.step()

        self._global_step += 1

        if phase == TrainingPhase.FOUNDATION:
            stats = self._step_phase1(observation, reward)
        elif phase == TrainingPhase.ALIGNMENT:
            stats = self._step_phase2(observation, reward)
        else:  # COGNITIVE
            stats = self._step_phase3(observation, reward, done)

        # Record stats
        self.train_stats["total_loss"].append(stats.get("loss", 0.0))
        self.train_stats["phase"].append(float(stats.get("phase", 0)))
        if "align_loss" in stats:
            self.train_stats["align_loss"].append(stats["align_loss"])

        stats["global_step"] = self._global_step
        stats["phase_name"]  = self.curriculum.phase_name()
        stats["lr_world"]    = self._schedulers[0].get_last_lr()[0]

        return stats

    # ── Full training loop ────────────────────────────────────────────────────

    def train(
        self,
        observations: List[Dict],
        rewards     : Optional[List]  = None,
        n_steps     : int             = 15_000,
        eval_every  : int             = 200,
    ) -> None:
        """
        Full curriculum training loop.

        Default schedule (overridable via CurriculumConfig):
          Steps   0 – 4 999  : Phase 1 FOUNDATION
          Steps 5 000 – 9 999  : Phase 2 ALIGNMENT
          Steps 10 000+        : Phase 3 COGNITIVE (PCGrad)
        """
        logger.info(
            f"AGITrainerV3: starting {n_steps} steps  |  "
            f"Phase1={self.curriculum.cfg.phase1_steps}  "
            f"Phase2={self.curriculum.cfg.phase2_steps}  "
            f"Phase3={'∞' if self.curriculum.cfg.phase3_steps < 0 else self.curriculum.cfg.phase3_steps}"
        )

        for step in range(n_steps):
            idx  = step % len(observations)
            obs  = observations[idx]
            rwd  = torch.tensor(float(rewards[idx])).to(self.device) if rewards else None
            stats = self.step(obs, rwd)

            if step % eval_every == 0:
                logger.info(
                    f"Step {step:6d} [{stats['phase_name']:10s}]  "
                    f"loss={stats.get('loss', 0):.4f}  "
                    f"align={stats.get('align_loss', 0):.4f}  "
                    f"psy={stats.get('psy_loss', 0):.4f}  "
                    f"lr={stats.get('lr_world', 0):.2e}"
                )

        logger.info("AGITrainerV3: training complete.")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        domain_opt_states = {
            k: v.state_dict() for k, v in self._domain_opts.items()
        }
        torch.save({
            "model_state"       : self.model.state_dict(),
            "align_loss_state"  : self.align_loss.state_dict(),
            "opt_world"         : self.opt_world.state_dict(),
            "opt_policy"        : self.opt_policy.state_dict(),
            "opt_psyche"        : self.opt_psyche.state_dict(),
            "opt_balance"       : self.opt_balance.state_dict(),
            "opt_align"         : self.opt_align.state_dict(),
            "opt_language"      : self.opt_language.state_dict() if self.opt_language else None,
            "domain_opts"       : domain_opt_states,
            "curriculum_phase"  : self.curriculum.current_phase.value,
            "global_step"       : self._global_step,
            "train_stats"       : self.train_stats,
            "agi_version"       : AGI_ONE_VERSION,
        }, path)
        logger.info(
            f"AGITrainerV3 checkpoint saved: {path}  "
            f"(step={self._global_step}  phase={self.curriculum.phase_name()})"
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.align_loss.load_state_dict(ckpt["align_loss_state"])
        self.opt_world.load_state_dict(ckpt["opt_world"])
        self.opt_policy.load_state_dict(ckpt["opt_policy"])
        self.opt_psyche.load_state_dict(ckpt["opt_psyche"])
        self.opt_balance.load_state_dict(ckpt["opt_balance"])
        self.opt_align.load_state_dict(ckpt["opt_align"])
        if self.opt_language and ckpt.get("opt_language"):
            self.opt_language.load_state_dict(ckpt["opt_language"])
        for domain, sd in ckpt.get("domain_opts", {}).items():
            if domain in self._domain_opts:
                self._domain_opts[domain].load_state_dict(sd)

        # Restore curriculum phase
        saved_phase = ckpt.get("curriculum_phase", 1)
        if saved_phase >= 2:
            self.curriculum._enter_phase(TrainingPhase(saved_phase))

        self._global_step = ckpt.get("global_step", 0)
        self.train_stats  = ckpt.get("train_stats", self.train_stats)
        logger.info(
            f"AGITrainerV3 checkpoint loaded: {path}  "
            f"(step={self._global_step}  phase={self.curriculum.phase_name()})"
        )

    def print_training_config(self) -> None:
        """Print a summary of the v3 training configuration."""
        print(f"\n{'='*65}")
        print(f"  AGI ONE v{AGI_ONE_VERSION} — AGITrainerV3 Configuration")
        print(f"{'='*65}")
        print(f"  Curriculum phases:")
        print(f"    Phase 1 FOUNDATION : {self.curriculum.cfg.phase1_steps:,} steps")
        print(f"    Phase 2 ALIGNMENT  : {self.curriculum.cfg.phase2_steps:,} steps")
        phase3 = (
            "∞" if self.curriculum.cfg.phase3_steps < 0
            else f"{self.curriculum.cfg.phase3_steps:,}"
        )
        print(f"    Phase 3 COGNITIVE  : {phase3} steps")
        print(f"  PCGrad enabled     : {self.curriculum.cfg.pcgrad_enabled}")
        print(f"  Align loss weight  : {self.curriculum.cfg.align_loss_weight}")
        print(f"  InfoNCE temperature: {math.exp(self.align_loss.log_tau.item()):.4f}")
        print(f"  Decoupled optimizers:")
        print(f"    world_model  lr : {self.opt_world.param_groups[0]['lr']:.1e}")
        print(f"    policy_heads lr : {self.opt_policy.param_groups[0]['lr']:.1e}")
        print(f"    psyche_layer lr : {self.opt_psyche.param_groups[0]['lr']:.1e}")
        if self.opt_language:
            print(f"    language_mod lr : {self.opt_language.param_groups[0]['lr']:.1e}")
        print(f"    align_proj   lr : {self.opt_align.param_groups[0]['lr']:.1e}")
        for dom, opt in self._domain_opts.items():
            print(f"    [{dom:12s}] lr : {opt.param_groups[0]['lr']:.1e}")
        print(f"  Ecosystem surrogates registered:")
        for name in self.orchestrator.registered_names():
            adp = self.orchestrator._adapters[name]
            print(
                f"    {name:35s} domain={adp.domain:12s} "
                f"frozen={adp.frozen_backbone}"
            )
        print(f"{'='*65}\n")


# =============================================================================
# SECTION 19 — CONVENIENCE FACTORY
# =============================================================================

def create_agi_one(
    latent_dim       : int  = 512,
    action_dim       : int  = 64,
    use_all_modules  : bool = True,
    psyche_mode      : str  = "healthy",
    language_backend : str  = "builtin",
    use_psyche_plus  : bool = True,
    device           : Optional[str] = None,
    verbose          : bool = True,
) -> AGIONE:
    """
    Factory: create a fully configured AGI ONE v2.0 instance.

    Example:
        agi = create_agi_one(latent_dim=256, action_dim=32)
        agi.print_architecture()
        state = agi(token_ids=torch.randint(0, 32000, (1, 32)))
        print(state.summary())
    """
    _device = get_agi_device(device or "cuda")
    cfg = AGIConfig(
        latent_dim       = latent_dim,
        action_dim       = action_dim,
        device           = _device,
        psyche_mode      = psyche_mode,
        language_backend = language_backend,
        use_mental_one   = use_all_modules,
        use_psy_bridge   = use_all_modules,
        use_real_fold    = use_all_modules,
        use_evolution    = use_all_modules,
        use_physics      = use_all_modules,
        use_bsd          = use_all_modules,
        use_grh          = use_all_modules,
        use_hodge        = use_all_modules,
        use_psyche_plus  = use_psyche_plus,
        verbose          = verbose,
        mppi_n_samples   = 256,   # factory default: lighter
        cem_n_iters      = 0,     # unused in v2.0
    )
    agi = AGIONE(cfg)
    if verbose:
        agi.print_architecture()
    return agi


# =============================================================================
# SECTION 20 — MAIN (Smoke Test)
# =============================================================================

if __name__ == "__main__":
    print(
        f"\n{'='*65}\n"
        f"  AGI ONE v{AGI_ONE_VERSION} — Production Smoke Test\n"
        f"  Developer : Yoon A Limsuwan / MSPS NETWORK\n"
        f"  MY SOUL MOVE BY POWER OF HOLY SPIRIT\n"
        f"  AI Assistants: Claude (Anthropic), GPT-4o (OpenAI),\n"
        f"                 Gemini (Google DeepMind), DeepSeek\n"
        f"{'='*65}\n"
    )

    # ── [1] Create AGI ONE v2.0 ───────────────────────────────────────────────
    agi = create_agi_one(
        latent_dim       = 128,   # small for smoke test
        action_dim       = 16,
        use_all_modules  = False,
        psyche_mode      = "healthy",
        verbose          = True,
    )
    # Override MPPI for speed in smoke test
    agi.mppi.n_samples = 16
    # Run Psyche Plus every step so the smoke test actually exercises it
    agi.cfg.psyche_plus_run_every_n_steps = 1

    # ── [2] Text input ────────────────────────────────────────────────────────
    print("[TEST 1] Text input (token IDs)")
    tok = torch.randint(0, 32_000, (1, 32)).to(agi.device)
    with torch.no_grad():
        st = agi(token_ids=tok)
    print(f"  Winner module    : {st.winner_module}")
    print(f"  Workspace shape  : {st.workspace_state.shape}")
    print(f"  Safety score     : {st.safety_score:.3f}")
    print(f"  CSOC n_layers    : {st.csoc_n_layers}")
    if st.meta_cognition:
        print(f"  Cognitive load   : {st.meta_cognition['cognitive_load']:.3f}")
        print(f"  Epistemic unc    : {st.meta_cognition['epistemic_unc']:.3f}")
    if st.psyche_plus_validity is not None:
        print(f"  PsychePlus valid : {st.psyche_plus_validity:.3f}")
        print(f"  PsychePlus axioms: {st.psyche_plus_axiom_count}")
        print(f"  PsychePlus codify: {st.psyche_plus_codified}")

    # ── [3] Time-series input ─────────────────────────────────────────────────
    print("\n[TEST 2] Time-series (EEG)")
    ts = torch.randn(1, 64, 128).to(agi.device)
    with torch.no_grad():
        st = agi(timeseries=ts)
    print(f"  Winner module    : {st.winner_module}")
    print(f"  ONE latent shape : {st.one_ecosystem_latent.shape}")

    # ── [4] Training step ─────────────────────────────────────────────────────
    print("\n[TEST 3] Training step (Dreamer compound loss)")
    agi_t = create_agi_one(latent_dim=64, action_dim=8, verbose=False)
    agi_t.mppi.n_samples = 8
    trainer = AGITrainer(agi_t)
    obs_in  = {"token_ids": torch.randint(0, 32_000, (1, 16))}
    rwd_in  = torch.tensor(1.0)
    stats   = trainer.step(obs_in, rwd_in)
    print(f"  Loss             : {stats['loss']:.4f}")
    print(f"  Winner           : {stats.get('winner_module','?')}")
    print(f"  Safety           : {stats.get('safety_score',0):.3f}")

    # ── [5] Open Science Registry ─────────────────────────────────────────────
    print("\n[TEST 4] Open Science Registry")
    agi.science_registry.register(DatasetRecord(
        dataset_id  = "openneuro_ds003944",
        title       = "EEG Resting State Dataset",
        source_lab  = "Neuroimaging Lab",
        institution = "Stanford University",
        contributors= ["J. Smith", "A. Lee"],
        doi         = "10.18112/openneuro.ds003944",
        license     = "CC-BY-4.0",
        year        = 2021,
        tags        = ["EEG", "neuroscience", "resting-state"],
    ))
    rec = agi.science_registry.cite("openneuro_ds003944", "MENTAL ONE training")
    print(f"  Cited: {rec.title}  ({rec.institution})")
    print(f"  DOI  : {rec.doi}")

    # ── [6] Module availability ───────────────────────────────────────────────
    print("\n[TEST 5] v2.0 Module Availability")
    for name, active in agi.get_available_modules().items():
        print(f"  {'✓' if active else '✗'}  {name}")

    print(f"\n{'='*65}")
    print(f"  AGI ONE v{AGI_ONE_VERSION} smoke test complete.")
    print(f"{'='*65}\n")

    # ── [6] AGITrainerV3 — Curriculum + PCGrad + InfoNCE smoke test ───────────
    print("\n[TEST 6] AGITrainerV3 — v3.0 Curriculum Training")

    agi_v3 = create_agi_one(latent_dim=64, action_dim=8, verbose=False)
    agi_v3.mppi.n_samples = 8

    orchestrator = EcosystemOrchestrator(
        agi_latent_dim=64,
        device=agi_v3.device,
    )

    # [v3.4 NEW] Real registration for all seven surrogates this file knows
    # how to adapt (the six newly-wired modules + EVOLUTION ONE BV), in one
    # call. Each attach_*() degrades gracefully (warns, returns None) if its
    # underlying module isn't importable in this environment — replaces the
    # pre-v3.4 `_DummySurrogate` placeholder stubs entirely.
    wrappers = attach_all_surrogates_to_ecosystem(orchestrator, 64, agi_v3.device)
    for dom_name, wrapper in wrappers.items():
        if wrapper is not None:
            print(f"  {dom_name:18s} registered  quality={wrapper.get_quality_score():.3f}")
        else:
            print(f"  {dom_name:18s} unavailable — skipped")

    bv_wrapper = wrappers.get("evolution_bv")

    # [v3.2 NEW] Unify model + orchestrator so Step 7-B's quality_report()
    # path is exercised in this smoke test too.
    agi_v3.attach_orchestrator(orchestrator)

    curriculum_cfg = CurriculumConfig(
        phase1_steps=4, phase2_steps=4, phase3_steps=4,
        pcgrad_enabled=True, align_loss_weight=0.1,
    )
    trainer_v3 = AGITrainerV3(agi_v3, orchestrator, curriculum_cfg=curriculum_cfg)
    trainer_v3.print_training_config()

    obs_v3 = {"token_ids": torch.randint(0, 32_000, (1, 16))}
    rwd_v3 = torch.tensor(1.0)

    for i in range(12):
        stats_v3 = trainer_v3.step(obs_v3, rwd_v3)
        print(
            f"  Step {i:3d}  [{stats_v3['phase_name']:10s}]  "
            f"loss={stats_v3.get('loss',0):.4f}  "
            f"align={stats_v3.get('align_loss',0):.4f}"
        )

    print(
        f"\n  AGITrainerV3 curriculum cycle complete  "
        f"(final phase: {trainer_v3.curriculum.phase_name()})"
    )
