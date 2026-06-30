# AGI ONE

**Central orchestration hub of the ONE Ecosystem** — a PyTorch-based AGI architecture that unifies perception, language, world-modeling, planning, a psyche/safety layer, and a distributed hub of scientific surrogate models (protein folding, cancer/epidemiology, computational psychiatry, fluid dynamics, particle physics, and number-theoretic conjecture modules) under one cognitive engine.

This repository documents two companion modules:

| File | Role |
|---|---|
| `agi_one_v3_8.py` | Main AGI ONE engine (v3.8) — perception/language/world-model/planning core, the Ecosystem Orchestrator, curriculum training, and gradient-conflict-aware multi-task training |
| `agi_one_psyche_plus_v33.py` | **Psyche Plus** (v3.3) — Id/Ego/Super-Ego speculative-axiom evolution layer with optional multi-LLM consensus, plugged into the main engine as a slow, self-evolving safety/reasoning loop |

---

## Developer

**Yoon A Limsuwan** / MSPS NETWORK
ORCID: `0009-0008-2374-0788` · GitHub: [yoonalimsuwan](https://github.com/yoonalimsuwan) · Email: msps4u@gmail.com
License: MIT

**AI development assistants** (credited per the project's standing convention): Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google DeepMind), DeepSeek AI — contributing to architecture co-design, code review, and revision passes across versions, as noted per-version in each file's changelog.

---

## 1. `agi_one_v3_8.py` — AGI ONE Core Engine

### What it is
A production-grade, modular AGI architecture (the file itself is organized into 20 numbered sections) combining:

- **Perception** — ViT-style patch vision encoder, Conformer-lite audio encoder, proprioception and time-series encoders, fused via cross-modal attention.
- **Language** — GPT-style causal transformer with RoPE.
- **Memory** — working memory and episodic memory modules feeding a **Global Workspace** (a Global-Workspace-Theory-style attention bottleneck).
- **World model** — a DreamerV3-style model (symlog encoding, two-hot reward, free-bits KL, categorical latents with straight-through estimation).
- **Planning** — an MPPI (Model Predictive Path Integral) planner, replacing an earlier CEM-based planner.
- **Meta-cognition** — tracks cognitive load and epistemic uncertainty, feeding a CSOC (Controlled Self-Organized Criticality) compute controller that adapts model depth at the edge of chaos.
- **Psyche Executive Layer** — a moment-to-moment Id→Goal / Ego→Plan / Superego→Safety gate (distinct from, and complementary to, the slower Psyche Plus module below).
- **Training** — full PPO (clipped surrogate + GAE(λ) + entropy bonus), a Dreamer-style compound loss with Kendall uncertainty weighting, and two trainers (`AGITrainer` for joint single-stage training, `AGITrainerV3` for multi-stage curriculum training).

### The core problem it solves
Jointly training heterogeneous losses — physics PDEs, discrete math, reinforcement learning, and language — causes **gradient interference (negative transfer)**, where gradients from one domain corrupt representations learned for another. AGI ONE v3.0+ addresses this with three coordinated mechanisms:

1. **Multi-stage curriculum training** (`CurriculumScheduler`)
   - *Phase 1 — Foundation*: physics/math surrogates and the DreamerV3 world model train independently, then freeze.
   - *Phase 2 — Alignment*: the physics backbone is frozen; a language↔physics bridge is trained via InfoNCE contrastive alignment, so cross-modal latent spaces align geometrically without direct gradient coupling.
   - *Phase 3 — Cognitive*: the PPO actor-critic and Psyche Executive Layer are unlocked and fine-tuned against reward + free-energy signals, with surrogate backbones still frozen.

2. **PCGrad gradient surgery** (`PCGradOptimizer`) — during Phase 3, conflicting task gradients (where `g_i · g_j < 0`) are detected and projected onto each other's orthogonal complement, based on Yu et al. 2020.

3. **InfoNCE cross-modal alignment** (`CrossModalAlignmentLoss`) — a symmetric contrastive loss (CLIP/SimCLR-style) that attracts matched physics↔language latent pairs and repels mismatched ones, again avoiding direct gradient coupling.

Each domain group additionally gets its own optimizer and learning rate (`AGITrainerV3`): physics and math surrogates train at very low LR (1e-6 / 5e-7), the language module at standard LLM fine-tuning LR (1e-5), and the world model / policy heads / psyche layer at higher RL-typical LRs (1e-4–3e-4).

### Ecosystem Orchestrator
Rather than embedding scientific modules directly, AGI ONE v3+ keeps `EcosystemOrchestrator` as a hub that holds adapter references to externally maintained surrogate modules from the wider ONE Ecosystem, including:

- `GNOFoldEncoderAdapter` — REAL FOLD ONE (protein structure)
- `GNOEvolutionBVEncoderAdapter` — EVOLUTION ONE / BV cancer-evolution surrogate (with Mode-3/Mode-4 cell-population and organelle/phenotype wiring as of v3.7)
- `MentalOperatorEncoderAdapter` — MENTAL ONE (computational psychiatry)
- `StructuralFNO3DEncoderAdapter` — SUPER DNS ONE (turbulence/fluid dynamics)
- `GNOPhysicsEncoderAdapter` — STANDARD ONE (particle physics, including Yang–Mills and Bell/CHSH entangled-pair correlator modes as of v3.6)
- `GNOHodgeEncoderAdapter`, `GNONumberTheoryEncoderAdapter` — HODGE ONE / RH / BSD / GRH mathematical-conjecture modules

Each adapter exposes a uniform `.encode()` contract and an optional `quality_fn` hook so a domain's *actual simulation quality* (e.g. a CME residual, a GUE-statistics loss) — not just its latent embedding — can reach the central hub via `EcosystemOrchestrator.quality_report()`. This lets downstream gating (notably Psyche Plus's codification gate) condition on real simulation reliability rather than self-referential plausibility alone. Every quality hook is additive and defaults to `None`, reproducing prior behavior exactly when unused.

### Triadic Coherence Analyzer (v3.8, new in this file)
Where the orchestrator previously only mean-pooled the EVOLUTION BV, REAL FOLD ONE, and MENTAL ONE latents into one shared ecosystem vector — discarding pairwise structure — `TriadicCoherenceAnalyzer` adds:

- **`pairwise_geometry()`** — a gradient-free read-out of centered cosine similarity and normalized distance for each of the three domain pairs, plus a composite `triadic_coherence` score in `[-1, 1]`.
- A learnable **3-way InfoNCE** generalization of `CrossModalAlignmentLoss` covering all 3 domain pairs with its own projection heads and temperature (no parameter sharing with the physics↔language alignment loss).
- **`quality_weighted_gate()`** — scales the 3-way alignment loss by the geometric mean of the three domains' `quality_report()` scores, so a single unreliable surrogate suppresses (rather than just dilutes) the triadic alignment signal.

This feature is opt-in and defaults to off (`CurriculumConfig.triadic_bv_fold_mental_enabled = False`); it is purely additive and does not alter any existing call path.

### Version history (high level)
- **v1.0** — initial architecture: perception, language, memory, global workspace, RSSM world model, CEM planning, meta-cognition, psyche triad.
- **v2.0** — production upgrade: MPPI planner, DreamerV3 world model, full PPO, Kendall-weighted compound loss, SSC stabilizer, CSOC compute controller, Psyche Executive Layer, Open Science Registry, math reasoning layer (BSD/HODGE/GRH ONE).
- **v3.0** — distributed ecosystem + curriculum training: `EcosystemOrchestrator`, PCGrad, InfoNCE alignment, decoupled domain optimizers, `AGITrainerV3`.
- **v3.1–v3.7** — progressive wiring of Psyche Plus, simulation-quality-aware gating, EVOLUTION BV (including organelle/phenotype sub-cellular Mode-4 wiring), HODGE rewiring, and NGO Physics Mode-4 entangled-pair CHSH correlators.
- **v3.8** — Triadic Coherence Analyzer for tight BV ↔ Fold ↔ Mental cross-domain analysis (described above).

### Quick start
```python
from agi_one_v3_8 import create_agi_one, AGITrainer, AGITrainerV3, EcosystemOrchestrator

agi = create_agi_one(latent_dim=128, action_dim=16, verbose=True)

# Single forward pass on text
import torch
tok = torch.randint(0, 32_000, (1, 32))
state = agi(token_ids=tok)
print(state.winner_module, state.safety_score)
```
Running the file directly (`python agi_one_v3_8.py`) executes a full smoke test: text/time-series forward passes, a Dreamer-style training step, Open Science Registry dataset citation, module-availability report, and a 12-step `AGITrainerV3` curriculum cycle with all available surrogates attached.

---

## 2. `agi_one_psyche_plus_v33.py` — Psyche Plus (Id / Ego / Super-Ego)

### What it is
A complementary, **slower-timescale** cognitive layer to the main engine's moment-to-moment Psyche Executive Layer. Where the executive layer gates each step, Psyche Plus runs every `psyche_plus_run_every_n_steps` and maintains a persistent, **self-evolving "axiom bank"**:

- **`IdModule`** — generates speculative latent proposals with a softplus-reparameterized noise scale (so it can never go negative or explode) and passes speculation through a `soft_clamp`.
- **`SuperEgoModule`** — verifies proposals by blending three signals: a learned linear verifier, cosine similarity against the nearest stored axiom in the bank, and (optionally) a live ecosystem `quality_score` from `EcosystemOrchestrator.quality_report()`.
- **`EgoModule`** — mediates between Id's proposal and Super-Ego's verdict to produce the layer's output.
- **`AGIOnePsychePlus`** — ties the triad together; its output is folded back into the AGI ONE workspace latent, safety-gated.
- **Axiom memory** — a bounded ring buffer (not a simple list), updated under `torch.no_grad()`, so it is checkpoint-safe and distinguishes genuinely codified axioms from random initialization filler.

### Dual-gated codification
`maybe_codify()` only writes a new speculative axiom into the bank if it clears **both**:
1. a validity threshold (Super-Ego's blended score), **and**
2. `quality_codify_min` — a minimum live ecosystem simulation-quality bar (toggleable via `require_quality_for_codify`, default on).

This prevents low-quality simulation output from silently accumulating as "verified" axioms. Incoming vectors are also validated with `torch.isfinite(...).all()` before being written; non-finite speculations are rejected and logged rather than corrupting the bank.

### Optional multi-LLM external consensus
Psyche Plus can optionally consult external LLMs (Claude / Gemini / GPT) as a structured, opt-in side-channel:
- **Off by default** (`enable_external_llm=False`) — no deployment spends API credits without explicit developer consent.
- When enabled: calls run **concurrently** (`ThreadPoolExecutor`), are **timeout-bounded**, and are **rate-limited** (`consult_every_n_steps`).
- Every call returns a structured `ExternalConsensusResult` (never silently discarded), with bounded retry/backoff on the live-tier path.
- Credentials are read only from explicit arguments or environment variables (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`) — never hardcoded or logged. SDK bindings are each wrapped in `try/except ImportError`, matching the ONE Ecosystem's optional-dependency convention.

### Robustness (v3.3 hardening pass)
The v3.3 revision (primary developer: Claude) added a full correctness/robustness pass: `quality_score` sanitization via `_safe_unit_scalar()` (NaN/Inf/out-of-range clamped and logged once), a re-clamped `quality_weight` property so weights can never sum above 1, empty-batch guards, full config validation in `PsychePlusConfig.__post_init__`, a de-duplicated multi-provider login path (`MultiLLMAuthManager`), and clearer logging on all-failed external-consensus fallbacks.

### Version history (high level)
- **v1.0** — prototype: stub multi-LLM "auth," standalone Id/Ego/Super-Ego triad, not yet wired into the central engine.
- **v2.0** — production hardening: real axiom-bank usage, ring-buffer memory, reparameterized Id noise, real SDK bindings behind opt-in flags, structured `ExternalConsensusResult`, full integration hooks into AGI ONE's central engine.
- **v3.2** — simulation-quality-aware gating: `quality_score` blending in `SuperEgoModule`, dual-gate codification.
- **v3.3** — full robustness/correctness/production-readiness pass (sanitization, validation, deduplication, retry/backoff).

### Quick start
```python
from agi_one_psyche_plus_v33 import AGIOnePsychePlus, PsychePlusConfig

cfg = PsychePlusConfig(enable_external_llm=False)  # safe default, no API calls
psyche_plus = AGIOnePsychePlus(cfg)

out = psyche_plus(latent, quality_score=0.82)  # quality_score optional
psyche_plus.maybe_codify(out, quality_score=0.82)
```

In `agi_one_v3_8.py`, this module is wired in automatically when `AGIConfig.use_psyche_plus=True`; no manual integration is required.

---

## Notes on scope

These two files are part of a much larger codebase (the **ONE Ecosystem**, 70,000+ lines of PyTorch across protein folding, cancer genomics, computational psychiatry, fluid dynamics, particle physics, and mathematical-conjecture surrogate clusters). This README covers only the central orchestration engine and its psyche/safety layer; surrogate modules referenced here (e.g. `structural_gno_fold_v4.py`, `structural_gno_evolution_bv_standalone.py`) are documented separately within their own clusters.
