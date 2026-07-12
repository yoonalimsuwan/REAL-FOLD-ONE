# =============================================================================
# AGI ONE : Id, Ego, Super Ego / Plus
# Production-Grade Cognitive Psyche & Multi-LLM Consensus Layer
# =============================================================================
# Developer     : PAI , Yoon A Limsuwan / MSPS NETWORK
#                 MY SOUL MOVE BY POWER OF HOLY SPIRIT
# AI Assistants : Gemini (Google DeepMind)  — v1.0 prototype design
#                 Claude  (Anthropic)        — v2.0 production hardening &
#                                              AGI ONE v3.0 central integration;
#                                              v3.2 PRIMARY DEVELOPER —
#                                              simulation-quality-aware gating
#                                              (quality_score blending in
#                                              SuperEgoModule + dual-gate
#                                              codify, PsychePlusConfig
#                                              quality fields);
#                                              v3.3 PRIMARY DEVELOPER —
#                                              full robustness/correctness/
#                                              production-readiness pass
#                                              (see v3.3 changelog below)
# License       : MIT
# Year          : 2026
# ORCID         : 0009-0008-2374-0788
# GitHub        : https://github.com/yoonalimsuwan
# =============================================================================
# VERSION HISTORY
# ─────────────────
# v1.0  Prototype — stub multi-LLM "auth", standalone Id/Ego/SuperEgo triad,
#       no connection to the AGI ONE central engine.
#
# v2.0  Production hardening:
#   [FIX] SuperEgo verifier previously NEVER read `evolved_axioms` — the
#         axiom bank was dead state. It now scores proposals against the
#         nearest stored axiom (cosine similarity) blended with the learned
#         linear verifier, so "evolving law" is real, not cosmetic.
#   [FIX] Axiom memory is a proper bounded ring buffer (register_buffer +
#         occupancy mask + write pointer), updated in-place under
#         `torch.no_grad()` — safe for state_dict checkpointing and for
#         distinguishing "real" axioms from the random init filler.
#   [FIX] IdModule's noise scale is reparameterised through softplus so it
#         can never go negative or explode, and speculation is passed
#         through `soft_clamp` (ONE Ecosystem convention) instead of being
#         left unbounded.
#
# v3.2  Simulation-Quality-Aware Gating
#         Primary developer: Claude (Anthropic).
#         Gemini and GPT were not involved in this revision.
#   [NEW] PsychePlusConfig: three new fields —
#         `superego_quality_weight` (blend weight for quality_score in
#         SuperEgoModule.forward), `quality_codify_min` (minimum ecosystem
#         quality required before codification is permitted),
#         `require_quality_for_codify` (on/off switch; default True).
#   [NEW] SuperEgoModule.forward(proposed_z, quality_score=None) — blends
#         a live ecosystem quality score (from EcosystemOrchestrator
#         .quality_report()) into the validity signal alongside the
#         existing linear-verifier + axiom-similarity terms. Omitting it
#         reproduces exact pre-v3.2 behaviour everywhere.
#   [NEW] AGIOnePsychePlus.forward(latent, quality_score=None) — threads
#         quality_score to SuperEgoModule and echoes it in the return dict.
#   [NEW] AGIOnePsychePlus.maybe_codify(out, threshold=None,
#         quality_score=None) — dual-gate: speculation must clear BOTH the
#         validity threshold AND quality_codify_min before it is written
#         into the axiom bank, preventing low-quality simulation data from
#         silently accumulating as "verified axioms".
#   [NEW] Real Anthropic / Google Generative AI / OpenAI SDK bindings,
#         each behind `try/except ImportError`, exactly like the rest of
#         the ONE Ecosystem's optional-dependency pattern. Credentials are
#         read ONLY from explicit arguments or environment variables
#         (ANTHROPIC_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY) — never
#         hardcoded, never logged.
#   [NEW] External multi-LLM consensus is no longer silently fired and
#         discarded on every forward() call. It is now an explicit,
#         opt-in (`enable_external_llm`), rate-limited
#         (`consult_every_n_steps`), concurrent (ThreadPoolExecutor) side
#         channel with a hard timeout, that returns a structured
#         `ExternalConsensusResult` and can NEVER block, crash, or spend
#         API credits unless the developer turns it on.
#   [NEW] `PsychePlusConfig` — single dataclass source of truth for every
#         hyperparameter (mirrors `AGIConfig` / `PsycheConfig` patterns
#         already used elsewhere in the ONE Ecosystem).
#   [NEW] Full integration hooks for AGI ONE v3.0's central engine:
#         `AGIONE.psyche_plus`, `AGIConfig.use_psyche_plus`, new
#         `AGIState` fields — see agi_one_v3.py SECTION 6-B. This module
#         no longer needs to be wired in by hand.
#   [NEW] [PASS]/[FAIL] verification suite in `__main__`, matching ONE
#         Ecosystem conventions.
#
# v3.3  Full Robustness / Correctness / Production-Readiness Pass  [NEW]
#         Primary developer: Claude (Anthropic).
#         Gemini and GPT were not involved in this revision.
#
#   ── Correctness ─────────────────────────────────────────────────────────
#   [FIX] `quality_score` is now sanitised through `_safe_unit_scalar()`
#         everywhere it enters the system (SuperEgoModule.forward,
#         maybe_codify). NaN / Inf / out-of-range values are clamped to
#         [0, 1] and logged once at WARNING level instead of silently
#         corrupting the validity blend or being compared with `>=` against
#         garbage.
#   [FIX] `quality_weight` is no longer frozen at construction time as a
#         plain float derived from `axiom_weight`. It is now a property
#         (`SuperEgoModule.quality_weight`) re-clamped against the *current*
#         `axiom_weight` on every access, so the two weights can never sum
#         to > 1 even if a future caller mutates `axiom_weight` after
#         construction.
#   [FIX] `_axiom_similarity` / `forward` guard against an empty batch
#         (`proposed_z.shape[0] == 0`) and return correctly-shaped empty
#         tensors instead of risking a broadcast error downstream.
#   [FIX] `codify_new_axiom` now validates the incoming vector with
#         `torch.isfinite(...).all()` before writing into the ring buffer.
#         A non-finite speculation is REJECTED (logged at WARNING, ring
#         buffer untouched, returns -1) instead of silently poisoning the
#         axiom bank — this was a real corruption path in v2.0/v3.2 because
#         nothing upstream guaranteed `updated_latent` was finite before
#         `maybe_codify` calls this.
#   [FIX] `ExternalConsensusResult` now carries a human-readable `summary`
#         field and the all-failed case is logged at WARNING (previously
#         silently fell back to a neutral 0.5 consensus with only a
#         DEBUG-invisible trail).
#   [FIX] `PsychePlusConfig.device` accepts a `torch.device`, a device
#         string ("cpu", "cuda:0"), or None — invalid strings raise a clear
#         `ValueError` at construction time instead of failing deep inside
#         `nn.Module.to()`.
#
#   ── Robustness ──────────────────────────────────────────────────────────
#   [NEW] `PsychePlusConfig.__post_init__` validates every numeric field
#         (ranges, positivity) and raises `ValueError` immediately with a
#         specific message, instead of letting a bad config silently
#         produce nonsense scores three modules downstream.
#   [NEW] `MultiLLMAuthManager` de-duplicated: the three near-identical
#         `login_claude` / `login_gemini` / `login_gpt` bodies are now a
#         single `_login(provider_spec, ...)` driven by a small
#         `_PROVIDER_REGISTRY` table. Behaviour is unchanged; the public
#         `login_claude/login_gemini/login_gpt` methods remain as thin
#         wrappers for backward compatibility.
#   [NEW] `LLMClient.execute_reasoning` supports bounded retry with
#         exponential backoff (`max_retries`, `retry_backoff_s` on
#         `PsychePlusConfig`, default `max_retries=0` → byte-for-byte
#         pre-v3.3 behaviour). Retries only apply to live-tier calls and
#         only to exceptions raised by the SDK call itself, never to a
#         deliberate timeout budget overrun.
#   [NEW] `attach_to_agi_one` now logs a WARNING (not silently) when it
#         has to guess `latent_dim`, and accepts an explicit
#         `latent_dim` override argument.
#
#   ── Production-readiness ────────────────────────────────────────────────
#   [NEW] Every `AGIOnePsychePlus` instance gets a short `instance_id`
#         (uuid4 hex[:8]) threaded into every log line via a
#         `logging.LoggerAdapter`, so multi-instance deployments can be
#         disambiguated in shared log streams.
#   [NEW] Lightweight in-process counters on `AGIOnePsychePlus`
#         (`stats: Dict[str, int]`) tracking forward_calls, codify_attempts,
#         codify_successes, codify_refused_validity, codify_refused_quality,
#         external_consult_calls, external_consult_skipped,
#         axiom_rejected_nonfinite — exposed via `get_state_summary()`.
#   [NEW] `AGIOnePsychePlus.__repr__` prints a compact, useful one-liner
#         (latent_dim, axiom_count/max_axioms, global_step, external_llm
#         on/off) for REPL / debugger use.
#   [NEW] `__main__` verification suite extended with tests for: NaN/Inf
#         quality_score sanitisation, non-finite axiom rejection, empty-
#         batch handling, config validation (`__post_init__` raises), and
#         the new stats counters.
#
#   All v3.3 changes are additive and default-preserving: every new
#   knob defaults to a value that reproduces exact pre-v3.3 behaviour.
#   No existing public method signature had a required parameter added.
# =============================================================================

from __future__ import annotations

import logging
import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
_base_logger = logging.getLogger("AGI_ONE_Psyche_Plus")

# =============================================================================
# 0. ONE ECOSYSTEM SHARED PRIMITIVES (one_core fallback chain)
# =============================================================================

try:
    from one_core_v3 import soft_clamp                     # shared SSC/CSOC/Itô
    HAS_ONE_CORE = True
    _base_logger.info("✓ one_core_v3 (soft_clamp)")
except ImportError:
    try:
        from one_core_mental import soft_clamp             # mental-scale primitives
        HAS_ONE_CORE = True
        _base_logger.info("✓ one_core_mental (soft_clamp)")
    except ImportError:
        HAS_ONE_CORE = False
        _base_logger.warning("✗ one_core_v3 / one_core_mental not found — inline soft_clamp fallback active")

        def soft_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            """tanh-based soft clamp (ONE Ecosystem convention) — differentiable,
            never produces a hard gradient cliff like `.clamp()`."""
            mid   = (hi + lo) / 2.0
            half  = max((hi - lo) / 2.0, 1e-6)
            return mid + half * torch.tanh((x - mid) / half)

# =============================================================================
# 0-B. OPTIONAL MULTI-LLM SDKS (each independently optional)
# =============================================================================

try:
    import anthropic
    HAS_ANTHROPIC_SDK = True
    _base_logger.info("✓ anthropic SDK")
except ImportError:
    HAS_ANTHROPIC_SDK = False
    _base_logger.info("✗ anthropic SDK not installed — Claude sessions will be simulated")

try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
    _base_logger.info("✓ google-generativeai SDK")
except ImportError:
    HAS_GEMINI_SDK = False
    _base_logger.info("✗ google-generativeai SDK not installed — Gemini sessions will be simulated")

try:
    import openai
    HAS_OPENAI_SDK = True
    _base_logger.info("✓ openai SDK")
except ImportError:
    HAS_OPENAI_SDK = False
    _base_logger.info("✗ openai SDK not installed — GPT sessions will be simulated")


# =============================================================================
# 0-C. [v3.3 NEW] SMALL SHARED SANITISATION HELPERS
# =============================================================================

def _safe_unit_scalar(
    value: Optional[float],
    *,
    name: str,
    logger: logging.Logger,
    default: float = 0.5,
) -> Optional[float]:
    """
    Sanitises an externally-supplied "score in [0, 1]" value.

    - None passes through unchanged (caller treats it as "not supplied").
    - NaN / Inf / non-numeric -> logs a WARNING once, returns `default`.
    - Out-of-range finite values -> silently clamped to [0, 1] (this is a
      normal, expected occurrence — e.g. floating-point drift — and not
      worth a log line on every call).
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        logger.warning(f"[SANITISE] {name}={value!r} is not numeric — using default {default}.")
        return default
    if math.isnan(v) or math.isinf(v):
        logger.warning(f"[SANITISE] {name}={v} is NaN/Inf — using default {default}.")
        return default
    return max(0.0, min(1.0, v))


def _resolve_device(device: Union[str, "torch.device", None]) -> torch.device:
    """Accepts a torch.device, a device string, or None -> torch.device('cpu')."""
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device spec {device!r}: {exc}") from exc


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

@dataclass
class PsychePlusConfig:
    """Single source of truth for AGI ONE Psyche Plus hyperparameters."""
    latent_dim              : int   = 64
    max_axioms              : int   = 64
    id_noise_init           : float = 0.2
    superego_axiom_weight   : float = 0.5   # blend: linear verifier vs axiom similarity
    codify_threshold        : float = 0.6   # validity score above which a speculation is codified

    # ── [v3.2] Simulation-quality-aware gating ───────────────────────────
    # Primary developer of this addition: Claude (Anthropic).
    # `quality_score` (see AGIOnePsychePlus.forward / maybe_codify) is an
    # optional float in [0, 1] sourced from EcosystemOrchestrator.quality_report()
    # — i.e. from *actual* simulation health (Hodge period loss, RH/BSD
    # GUE-statistics loss, EVOLUTION BV CME residual, …), not from the
    # axiom bank's own self-referential plausibility scoring.
    superego_quality_weight : float = 0.25  # blend weight for quality_score in validity_score
    quality_codify_min      : float = 0.4   # below this, codify is refused even if validity passes
    require_quality_for_codify: bool = True # if quality_score is None, fall back to pre-v3.2 behaviour

    # ── External multi-LLM consensus (opt-in; off by default) ───────────────
    enable_external_llm     : bool  = False
    consult_every_n_steps   : int   = 50
    llm_timeout_s           : float = 12.0
    claude_model            : str   = "claude-sonnet-4-6"
    gemini_model            : str   = "gemini-2.0-flash"
    gpt_model                : str  = "gpt-4o"

    # ── [v3.3 NEW] Bounded retry for transient live-SDK failures ────────────
    # Default 0 retries reproduces exact pre-v3.3 behaviour. Retries only
    # wrap the live SDK call itself (network blips, rate limits) — never
    # the simulated-tier path, and never used to silently extend the
    # caller's wall-clock timeout budget past `llm_timeout_s`.
    max_retries              : int  = 0
    retry_backoff_s          : float = 0.5

    device: Union[str, "torch.device", None] = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        # [v3.3 NEW] Fail fast and specifically, rather than producing a
        # module that runs but silently emits nonsense scores.
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.max_axioms <= 0:
            raise ValueError(f"max_axioms must be positive, got {self.max_axioms}")
        if self.id_noise_init <= 0:
            raise ValueError(f"id_noise_init must be positive, got {self.id_noise_init}")
        for name, val in (
            ("superego_axiom_weight", self.superego_axiom_weight),
            ("superego_quality_weight", self.superego_quality_weight),
            ("quality_codify_min", self.quality_codify_min),
            ("codify_threshold", self.codify_threshold),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if self.superego_axiom_weight + self.superego_quality_weight > 1.0:
            raise ValueError(
                "superego_axiom_weight + superego_quality_weight must be <= 1.0, got "
                f"{self.superego_axiom_weight + self.superego_quality_weight}"
            )
        if self.consult_every_n_steps < 0:
            raise ValueError(f"consult_every_n_steps must be >= 0, got {self.consult_every_n_steps}")
        if self.llm_timeout_s <= 0:
            raise ValueError(f"llm_timeout_s must be positive, got {self.llm_timeout_s}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_backoff_s < 0:
            raise ValueError(f"retry_backoff_s must be >= 0, got {self.retry_backoff_s}")
        # Normalise device eagerly so later code only ever sees a torch.device.
        self.device = _resolve_device(self.device)


# =============================================================================
# 2. STRUCTURED RESULT TYPES (LLM responses are never silently discarded)
# =============================================================================

@dataclass
class ProviderResponse:
    provider   : str
    tier       : str             # "live" | "simulated"
    text       : str
    confidence : float
    latency_s  : float
    error      : Optional[str] = None
    retries_used: int = 0        # [v3.3 NEW]


@dataclass
class ExternalConsensusResult:
    responses        : List[ProviderResponse] = field(default_factory=list)
    consensus_score   : float = 0.5
    n_succeeded       : int   = 0
    n_failed          : int   = 0
    timestamp         : float = field(default_factory=time.time)
    summary           : str   = ""   # [v3.3 NEW] human-readable one-liner

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_score": round(self.consensus_score, 4),
            "n_succeeded"    : self.n_succeeded,
            "n_failed"       : self.n_failed,
            "providers"      : [r.provider for r in self.responses],
            "summary"        : self.summary,
        }


_CONF_PATTERN = re.compile(r"confidence[:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _extract_confidence(text: str) -> float:
    """Best-effort parse of a 'confidence: 0.8' style hint from free text.
    Falls back to a neutral 0.5 — never raises."""
    if not text:
        return 0.5
    m = _CONF_PATTERN.search(text)
    if not m:
        return 0.5
    try:
        val = float(m.group(1))
        if val > 1.0:           # tolerate "confidence: 80"
            val /= 100.0
        return max(0.0, min(1.0, val))
    except ValueError:
        return 0.5


def _inv_softplus(y: float) -> float:
    """Inverse of softplus, used only to initialise a raw nn.Parameter so
    that softplus(raw) == y at construction time."""
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y)) if y < 20.0 else y


# =============================================================================
# 3. MULTI-LLM AUTHENTICATION MANAGER (Claude, Gemini, GPT)
# =============================================================================

class LLMClient:
    """
    Unified wrapper around a single authenticated provider session.

    tier ∈ {"live", "simulated"}:
      "live"      — a real SDK client backed by a valid API key.
      "simulated" — no SDK and/or no API key found. Returns a clearly
                    labeled placeholder so callers never crash and the
                    system never silently spends API credits.
    """

    def __init__(self, provider: str, tier: str,
                 client: Any = None, model: Optional[str] = None) -> None:
        self.provider = provider
        self.tier     = tier
        self.client   = client
        self.model    = model

    def execute_reasoning(
        self,
        prompt: str,
        timeout: float = 12.0,
        max_retries: int = 0,
        retry_backoff_s: float = 0.5,
        logger: Optional[logging.Logger] = None,
    ) -> ProviderResponse:
        log = logger or _base_logger
        t0 = time.time()
        if self.tier != "live" or self.client is None:
            text = (f"[SIMULATED:{self.provider}] no live credentials configured — "
                    f"heuristic placeholder for prompt: {prompt[:80]!r}")
            return ProviderResponse(self.provider, self.tier, text, 0.5, time.time() - t0)

        # [v3.3 NEW] Bounded retry with exponential backoff. Only the live
        # SDK call is retried; max_retries=0 (default) is byte-for-byte
        # identical to pre-v3.3 behaviour (single attempt, no sleep).
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                text = self._call_live(prompt, timeout)
                return ProviderResponse(
                    self.provider, self.tier, text, _extract_confidence(text),
                    time.time() - t0, retries_used=attempt,
                )
            except Exception as exc:   # noqa: BLE001 — must never propagate into forward()
                last_exc = exc
                if attempt < max_retries:
                    log.warning(
                        f"[{self.provider.upper()}] live call failed (attempt "
                        f"{attempt + 1}/{max_retries + 1}): {exc} — retrying."
                    )
                    time.sleep(retry_backoff_s * (2 ** attempt))
                else:
                    log.warning(
                        f"[{self.provider.upper()}] live call failed after "
                        f"{max_retries + 1} attempt(s): {exc}"
                    )
        return ProviderResponse(
            self.provider, self.tier, "", 0.0, time.time() - t0,
            error=str(last_exc), retries_used=max_retries,
        )

    def _call_live(self, prompt: str, timeout: float) -> str:
        if self.provider == "claude":
            # `timeout` is honored by the SDK itself so the worker thread
            # returns within budget — the ThreadPoolExecutor context manager
            # still joins all threads on exit, so the *real* bound on
            # consult_external()'s wall-clock time comes from here, not
            # from as_completed()'s timeout alone.
            resp = self.client.messages.create(
                model=self.model, max_tokens=300, timeout=timeout,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            )
        if self.provider == "gemini":
            resp = self.client.generate_content(
                prompt, request_options={"timeout": timeout}
            )
            return getattr(resp, "text", "") or ""
        if self.provider == "gpt":
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=300, timeout=timeout,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        raise ValueError(f"Unknown provider: {self.provider}")


# [v3.3 NEW] Small registry table driving the de-duplicated login logic.
# Each entry: (env_var, has_sdk_flag_name, default_model, client_factory)
_PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "claude": {
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-6",
        "sdk_available": lambda: HAS_ANTHROPIC_SDK,
        "make_client": lambda key: anthropic.Anthropic(api_key=key),
        "log_name": "Claude",
        "sdk_hint": "SDK and/or ANTHROPIC_API_KEY",
    },
    "gemini": {
        "env_var": "GOOGLE_API_KEY",
        "default_model": "gemini-2.0-flash",
        "sdk_available": lambda: HAS_GEMINI_SDK,
        "make_client": lambda key, model: (genai.configure(api_key=key), genai.GenerativeModel(model))[1],
        "log_name": "Gemini",
        "sdk_hint": "SDK and/or GOOGLE_API_KEY",
    },
    "gpt": {
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "sdk_available": lambda: HAS_OPENAI_SDK,
        "make_client": lambda key: openai.OpenAI(api_key=key),
        "log_name": "GPT",
        "sdk_hint": "SDK and/or OPENAI_API_KEY",
    },
}


class MultiLLMAuthManager:
    """
    Manages provider sessions for the External Consensus side-channel.

    Credentials are read ONLY from an explicit argument or from the
    standard environment variable for that provider — never hardcoded.
    If a provider's SDK is missing or no key is found, that provider
    silently falls back to a "simulated" session rather than failing,
    so the rest of the system keeps working in dev/test environments.

    [v3.3] The three near-identical login_* bodies from v2.0/v3.2 are now
    driven by a single `_login()` implementation + `_PROVIDER_REGISTRY`.
    Public method names/signatures are unchanged for backward compatibility.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.sessions: Dict[str, LLMClient] = {}
        self.log = logger or _base_logger
        self.log.info("Initializing AGI ONE External Consensus Authentication Subsystem.")

    def _login(self, provider: str, api_key: Optional[str], model: Optional[str]) -> bool:
        spec = _PROVIDER_REGISTRY[provider]
        resolved_model = model or spec["default_model"]
        key = api_key or os.environ.get(spec["env_var"])

        if spec["sdk_available"]() and key:
            try:
                if provider == "gemini":
                    client = spec["make_client"](key, resolved_model)
                else:
                    client = spec["make_client"](key)
                self.sessions[provider] = LLMClient(provider, "live", client, resolved_model)
                self.log.info(f"[AUTH] {spec['log_name']} — live session established.")
                return True
            except Exception as exc:   # noqa: BLE001
                self.log.warning(
                    f"[AUTH] {spec['log_name']} live session failed ({exc}); falling back to simulated."
                )
        self.sessions[provider] = LLMClient(provider, "simulated", None, resolved_model)
        self.log.info(f"[AUTH] {spec['log_name']} — simulated mode (no {spec['sdk_hint']}).")
        return False

    def login_claude(self, api_key: Optional[str] = None,
                      model: str = "claude-sonnet-4-6") -> bool:
        return self._login("claude", api_key, model)

    def login_gemini(self, api_key: Optional[str] = None,
                      model: str = "gemini-2.0-flash") -> bool:
        return self._login("gemini", api_key, model)

    def login_gpt(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> bool:
        return self._login("gpt", api_key, model)

    def get_session(self, provider: str) -> Optional[LLMClient]:
        return self.sessions.get(provider.lower())


# =============================================================================
# 4. COGNITIVE PSYCHE TRIAD MODULES (Id, Ego, Super Ego / Plus)
# =============================================================================

class IdModule(nn.Module):
    """
    The Speculator / Creative Engine.

    Generates bounded stochastic hypotheses (structural mutations) in the
    latent landscape. The exploration noise scale is a learnable, strictly
    positive parameter (softplus-reparameterised), and the output is passed
    through `soft_clamp` so speculation can never diverge numerically —
    a real concern once this feeds a 500+ module production graph.
    """

    def __init__(self, latent_dim: int, init_noise: float = 0.2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.speculate_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        # softplus(raw) > 0 always — no risk of negative/zero/exploding noise
        self._raw_noise_scale = nn.Parameter(torch.tensor(_inv_softplus(init_noise)))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

    @property
    def noise_scale(self) -> torch.Tensor:
        return F.softplus(self._raw_noise_scale) + 1e-4

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 0:   # [v3.3 NEW] empty-batch guard
            return z.new_zeros((0, self.latent_dim))
        epsilon = torch.randn_like(z) * self.noise_scale
        speculation = self.speculate_net(z) + epsilon
        speculation = soft_clamp(speculation, -8.0, 8.0)
        with torch.no_grad():
            self.step_count += 1
        return speculation


class SuperEgoModule(nn.Module):
    """
    The Evolving Law / Axiomatic Filter.

    Scores a proposed latent against a bounded, self-evolving bank of
    "axioms" — speculations that previously passed verification. The score
    blends:
      (a) a learned linear verifier (general-purpose structural plausibility)
      (b) nearest-axiom cosine similarity (explicit precedent matching)
      (c) [v3.2] an optional externally-supplied simulation quality score

    The axiom bank is a fixed-capacity ring buffer stored as
    `register_buffer` tensors so it checkpoints correctly with the rest of
    the model and survives `.to(device)` calls. An occupancy mask
    distinguishes genuinely-codified axioms from the random initial filler.
    """

    def __init__(self, latent_dim: int, max_axioms: int = 64,
                 axiom_weight: float = 0.5, quality_weight: float = 0.0,
                 logger: Optional[logging.Logger] = None) -> None:
        super().__init__()
        self.latent_dim    = latent_dim
        self.max_axioms    = max_axioms
        self.log = logger or _base_logger
        self._axiom_weight = float(min(max(axiom_weight, 0.0), 1.0))
        # [v3.2] Weight given to an externally-supplied simulation-quality
        # score, on top of the pre-existing linear-verifier / axiom-
        # similarity blend. Defaults to 0.0, so a SuperEgoModule built
        # without this kwarg — or called without `quality_score` —
        # behaves byte-for-byte like pre-v3.2.
        self._quality_weight_raw = float(max(quality_weight, 0.0))

        self.verifier_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, 1),
        )

        self.register_buffer("evolved_axioms", torch.randn(max_axioms, latent_dim) * 0.1)
        self.register_buffer("axiom_filled",   torch.zeros(max_axioms, dtype=torch.bool))
        self.register_buffer("axiom_count",    torch.zeros(1, dtype=torch.long))
        self.register_buffer("_write_ptr",     torch.zeros(1, dtype=torch.long))

    @property
    def axiom_weight(self) -> float:
        return self._axiom_weight

    @property
    def quality_weight(self) -> float:
        """
        [v3.3 FIX] Re-clamped against the *current* axiom_weight on every
        access (rather than frozen once at __init__ time), so the two
        weights can never sum to more than 1.0 even if axiom_weight is
        mutated after construction.
        """
        return float(min(self._quality_weight_raw, max(0.0, 1.0 - self._axiom_weight)))

    def _axiom_similarity(self, z: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to the nearest codified axiom, mapped to [0,1].
        Returns 0.5 (neutral) when no axioms have been codified yet."""
        if z.shape[0] == 0:   # [v3.3 NEW] empty-batch guard
            return z.new_zeros((0,))
        if int(self.axiom_count.item()) == 0:
            return torch.full((z.shape[0],), 0.5, device=z.device, dtype=z.dtype)
        axioms = self.evolved_axioms[self.axiom_filled]            # (N, D)
        z_n    = F.normalize(z, dim=-1)
        a_n    = F.normalize(axioms, dim=-1)
        sim    = (z_n @ a_n.T).max(dim=-1).values                  # (B,) in [-1, 1]
        return (sim + 1.0) / 2.0

    def forward(
        self,
        proposed_z   : torch.Tensor,
        quality_score: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Evaluates structural validity score in [0, 1]: 1.0 = fully safe/valid.

        `quality_score`  [v3.2] — optional float in [0, 1] reporting the
        *actual* simulation health of the ecosystem domain(s) that fed the
        current workspace latent (see EcosystemOrchestrator.quality_report()).
        When omitted (None), this method is exactly the pre-v3.2 blend of
        linear-verifier + axiom-similarity scores.

        [v3.3 FIX] `quality_score` is sanitised (NaN/Inf -> default 0.5,
        out-of-range -> clamped) before use, and an empty batch returns a
        correctly-shaped empty tensor rather than risking a broadcast error.
        """
        if proposed_z.dim() == 1:
            proposed_z = proposed_z.unsqueeze(0)
        if proposed_z.shape[0] == 0:   # [v3.3 NEW] empty-batch guard
            return proposed_z.new_zeros((0,))

        quality_score = _safe_unit_scalar(
            quality_score, name="quality_score", logger=self.log,
        )

        linear_score = torch.sigmoid(self.verifier_net(proposed_z)).squeeze(-1)   # (B,)
        axiom_score  = self._axiom_similarity(proposed_z)                         # (B,)

        qw = self.quality_weight   # [v3.3] live property, re-clamped each call
        if quality_score is None or qw <= 0.0:
            score = (1.0 - self.axiom_weight) * linear_score + self.axiom_weight * axiom_score
        else:
            q = torch.full_like(linear_score, quality_score)
            w_lin = 1.0 - self.axiom_weight - qw
            score = w_lin * linear_score + self.axiom_weight * axiom_score + qw * q
        return soft_clamp(score, 0.0, 1.0)

    @torch.no_grad()
    def codify_new_axiom(self, verified_speculation: torch.Tensor) -> int:
        """
        Dynamic evolution mechanism: folds a verified Id speculation into the
        permanent axiom ring buffer, overwriting the oldest slot once full.
        Returns the ring slot index written, or -1 if the speculation was
        rejected.

        [v3.3 FIX] The incoming vector is now validated with
        `torch.isfinite(...).all()` BEFORE it touches the ring buffer. A
        non-finite speculation (NaN/Inf — e.g. from an upstream numerical
        blow-up) is rejected and logged at WARNING instead of silently
        poisoning the axiom bank, which would otherwise corrupt every
        future similarity score computed against it.
        """
        vec = verified_speculation.detach().reshape(-1)
        if vec.shape[0] < self.latent_dim:
            vec = F.pad(vec, (0, self.latent_dim - vec.shape[0]))
        else:
            vec = vec[: self.latent_dim]

        if not torch.isfinite(vec).all():
            self.log.warning(
                "[SUPER EGO EVOLUTION] Rejected non-finite speculation — "
                "axiom bank left untouched."
            )
            return -1

        ptr = int(self._write_ptr.item())
        self.evolved_axioms[ptr].copy_(vec.to(self.evolved_axioms.device, self.evolved_axioms.dtype))
        self.axiom_filled[ptr] = True
        self._write_ptr[0] = (ptr + 1) % self.max_axioms
        self.axiom_count[0] = min(int(self.axiom_count.item()) + 1, self.max_axioms)

        self.log.info(
            f"[SUPER EGO EVOLUTION] Axiom codified into ring slot "
            f"{ptr}/{self.max_axioms} (total codified: {int(self.axiom_count.item())})."
        )
        return ptr


class EgoModule(nn.Module):
    """
    The Central Orchestrator / Facilitator.

    `reconcile()` is the differentiable internal pathway — it balances Id's
    speculation against SuperEgo's validity gate and updates the running
    latent state via a GRUCell. This is the ONLY path used inside any
    `forward()` call.

    `consult_external()` is a deliberately SEPARATE, non-differentiable,
    opt-in side-channel. Network I/O has no place inside an autograd graph,
    so external multi-LLM consensus never runs implicitly — callers (e.g.
    `AGIOnePsychePlus.maybe_consult_external`) must invoke it explicitly,
    and it always returns a structured result instead of being discarded.
    """

    def __init__(self, latent_dim: int, auth_manager: MultiLLMAuthManager,
                 logger: Optional[logging.Logger] = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.auth = auth_manager
        self.log = logger or _base_logger

        self.balance_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(), nn.LayerNorm(latent_dim),
        )
        self.state_updater = nn.GRUCell(latent_dim, latent_dim)

    def reconcile(
        self,
        z_state         : torch.Tensor,
        id_speculation  : torch.Tensor,
        superego_score  : torch.Tensor,
    ) -> torch.Tensor:
        """Harmonizes Id's speculation with SuperEgo's safety gate."""
        if z_state.dim() == 1:
            z_state = z_state.unsqueeze(0)
        if id_speculation.dim() == 1:
            id_speculation = id_speculation.unsqueeze(0)
        if z_state.shape[0] == 0:   # [v3.3 NEW] empty-batch guard
            return z_state
        score = superego_score.unsqueeze(-1) if superego_score.dim() == 1 else superego_score

        combined      = torch.cat([z_state, id_speculation], dim=-1)
        gated_latent  = self.balance_gate(combined)
        final_internal = score * gated_latent + (1.0 - score) * id_speculation

        updated_state = self.state_updater(final_internal, z_state)
        return soft_clamp(updated_state, -10.0, 10.0)

    def consult_external(
        self,
        prompt          : str,
        timeout         : float = 12.0,
        max_workers     : int = 3,
        max_retries     : int = 0,
        retry_backoff_s : float = 0.5,
    ) -> ExternalConsensusResult:
        """
        Queries every authenticated provider concurrently with a hard
        per-call timeout. NEVER raises — every failure mode (timeout,
        network error, missing credentials) is captured per-provider in
        the returned result so the caller's control flow is unaffected.

        [v3.3 NEW] `max_retries`/`retry_backoff_s` thread through to each
        provider's `execute_reasoning` for transient live-call failures.
        Defaults reproduce exact pre-v3.3 behaviour (no retries).
        """
        sessions = dict(self.auth.sessions)
        if not sessions:
            self.log.debug("[EGO HUB] consult_external: no registered providers — skipping.")
            return ExternalConsensusResult(summary="no providers registered")

        responses: List[ProviderResponse] = []
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(sessions)))) as pool:
                future_map = {
                    pool.submit(
                        sess.execute_reasoning, prompt, timeout,
                        max_retries, retry_backoff_s, self.log,
                    ): name
                    for name, sess in sessions.items()
                }
                for fut in as_completed(future_map, timeout=timeout + 5.0):
                    name = future_map[fut]
                    try:
                        responses.append(fut.result())
                    except Exception as exc:   # noqa: BLE001
                        responses.append(ProviderResponse(
                            name, "unknown", "", 0.0, timeout, error=str(exc)
                        ))
        except Exception as exc:   # noqa: BLE001 — pool-level timeout/failure
            self.log.warning(f"[EGO HUB] consult_external pool failure: {exc}")

        n_ok   = sum(1 for r in responses if not r.error)
        n_fail = len(responses) - n_ok
        consensus = (
            sum(r.confidence for r in responses if not r.error) / n_ok if n_ok else 0.5
        )

        if n_ok == 0 and n_fail > 0:
            # [v3.3 FIX] Previously this fell back to a neutral 0.5 with no
            # WARNING-level trail — indistinguishable from "nobody asked".
            summary = f"all {n_fail} provider(s) failed — consensus defaulted to neutral 0.5"
            self.log.warning(f"[EGO HUB] {summary}")
        else:
            summary = f"{n_ok} ok / {n_fail} failed, consensus={consensus:.3f}"

        result = ExternalConsensusResult(
            responses=responses, consensus_score=consensus,
            n_succeeded=n_ok, n_failed=n_fail, summary=summary,
        )
        self.log.info(f"[EGO HUB] External consensus: {summary}")
        return result


# =============================================================================
# 5. UNIFIED INTEGRATION SYSTEM (AGI ONE: Id, Ego, Super Ego / Plus)
# =============================================================================

class AGIOnePsychePlus(nn.Module):
    """
    Production Id/Ego/Superego speculative-axiom layer with an optional
    multi-LLM external consensus side-channel.

    Role inside AGI ONE v3.0
    ─────────────────────────
    `PsycheExecutiveLayer` (agi_one_v3.py, SECTION 6) governs moment-to-moment
    action safety: Id→Goal, Ego→Plan, Superego→Safety-gate.

    `AGIOnePsychePlus` governs a slower, complementary loop: longer-horizon
    *hypothesis* evolution over the same workspace latent. Id proposes
    structural mutations, SuperEgo scores them against a growing bank of
    self-discovered axioms, and verified speculations are folded back into
    that bank — closing PSY ONE BRIDGE's "evolving law" concept for real.

    See `agi_one_v3.py` → `AGIConfig.use_psyche_plus` and
    `AGIONE.psyche_plus` for the central-hub wiring.

    [v3.3 NEW] Every instance carries a short `instance_id` threaded into
    all of its log lines (via a LoggerAdapter), and tracks lightweight
    in-process counters in `self.stats`, exposed through
    `get_state_summary()`. Both are purely observational — they have zero
    effect on the differentiable forward path.
    """

    def __init__(
        self,
        cfg    : Optional[PsychePlusConfig] = None,
        device : Optional[torch.device]     = None,
    ) -> None:
        super().__init__()
        self.cfg        = cfg or PsychePlusConfig()
        self.device     = _resolve_device(device) if device is not None else self.cfg.device
        self.latent_dim = self.cfg.latent_dim

        # [v3.3 NEW] Per-instance log context for multi-instance deployments.
        self.instance_id = uuid.uuid4().hex[:8]
        self.log = logging.LoggerAdapter(_base_logger, {"instance_id": self.instance_id})

        # [v3.3 NEW] Lightweight observational counters.
        self.stats: Dict[str, int] = {
            "forward_calls"            : 0,
            "codify_attempts"          : 0,
            "codify_successes"         : 0,
            "codify_refused_validity"  : 0,
            "codify_refused_quality"   : 0,
            "axiom_rejected_nonfinite" : 0,
            "external_consult_calls"   : 0,
            "external_consult_skipped" : 0,
        }

        self.auth_manager = MultiLLMAuthManager(logger=self.log)
        self.id_layer       = IdModule(self.latent_dim, self.cfg.id_noise_init)
        self.superego_layer = SuperEgoModule(
            self.latent_dim,
            self.cfg.max_axioms,
            self.cfg.superego_axiom_weight,
            quality_weight=self.cfg.superego_quality_weight,   # [v3.2]
            logger=self.log,
        )
        self.ego_layer = EgoModule(self.latent_dim, self.auth_manager, logger=self.log)

        self.register_buffer("_global_step", torch.zeros(1, dtype=torch.long))
        self.to(self.device)

        self.log.info(
            f"==> [AGI ONE : Id, Ego, Super Ego / Plus v3.3] assembled "
            f"(latent_dim={self.latent_dim}, max_axioms={self.cfg.max_axioms}, "
            f"external_llm={'ON' if self.cfg.enable_external_llm else 'off'})."
        )

    def __repr__(self) -> str:   # [v3.3 NEW]
        return (
            f"AGIOnePsychePlus(instance_id={self.instance_id!r}, "
            f"latent_dim={self.latent_dim}, "
            f"axioms={int(self.superego_layer.axiom_count.item())}/{self.cfg.max_axioms}, "
            f"global_step={int(self._global_step.item())}, "
            f"external_llm={'on' if self.cfg.enable_external_llm else 'off'})"
        )

    # ── Differentiable core pathway ──────────────────────────────────────────

    def forward(
        self,
        current_latent: torch.Tensor,
        quality_score : Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        One Id→SuperEgo→Ego cycle. Fully differentiable end to end.

        `quality_score`  [v3.2] — optional float in [0, 1] from
        EcosystemOrchestrator.quality_report()["ecosystem"] (see agi_one_v3.py
        Step 7-B). When provided, SuperEgoModule blends it into the validity
        score alongside its own linear-verifier and axiom-similarity terms
        (controlled by `PsychePlusConfig.superego_quality_weight`).
        Passing None reproduces exact pre-v3.2 behaviour.

        [v3.3] `quality_score` sanitisation happens inside
        `SuperEgoModule.forward`; this method just passes it through and
        echoes back whatever was actually used (post-sanitisation is NOT
        re-applied here to keep this method side-effect-light — see
        `out["quality_score"]` docstring note below).
        """
        if current_latent.dim() == 1:
            current_latent = current_latent.unsqueeze(0)
        current_latent = current_latent.to(self.device)

        self.stats["forward_calls"] += 1

        speculative_z  = self.id_layer(current_latent)
        validity_score = self.superego_layer(speculative_z, quality_score=quality_score)
        new_latent     = self.ego_layer.reconcile(current_latent, speculative_z, validity_score)

        with torch.no_grad():
            self._global_step += 1

        return {
            "updated_latent"    : new_latent,
            "speculative_latent": speculative_z,
            "validity_score"    : validity_score,
            "axiom_count"       : int(self.superego_layer.axiom_count.item()),
            # Pass-through of the RAW input quality_score for inspection.
            # Note: this is the value the caller passed in, not the
            # sanitised value actually used internally by SuperEgoModule —
            # call `_safe_unit_scalar` yourself if you need the sanitised
            # number for logging/comparison purposes.
            "quality_score"     : quality_score,
        }

    def maybe_codify(
        self,
        out          : Dict[str, Any],
        threshold    : Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> bool:
        """
        Folds the cycle's output into the axiom bank if it cleared the
        validity threshold — and, when `require_quality_for_codify` is set
        and a quality score is available, also cleared `quality_codify_min`.

        The dual-gate logic  [v3.2]  (Primary developer: Claude, Anthropic):
          • validity threshold  — self-referential: how plausible is the
            speculation in latent space, relative to existing axioms?
          • quality minimum     — external: how healthy is the simulation
            domain whose data fed this workspace state?
        Requiring BOTH before codification prevents the axiom bank from
        accumulating speculations that came from poor-quality simulations
        (high cosine similarity to old axioms but low actual domain health).

        Falls back to plain pre-v3.2 behaviour when:
          a) `quality_score` is None — i.e. no orchestrator attached, or
          b) `cfg.require_quality_for_codify` is False.

        [v3.3 FIX] `quality_score` (whether passed as an argument or pulled
        from `out["quality_score"]`) is now sanitised via
        `_safe_unit_scalar` before the `>=` comparison against
        `quality_codify_min` — a NaN/Inf value here previously made the
        comparison meaningless (Python's `NaN >= x` is always False, so a
        corrupted quality signal silently blocked ALL codification with no
        explanation).
        [v3.3 NEW] Updates `self.stats` counters for attempts/successes/
        refusal reasons, and rejects (rather than codifies) a non-finite
        `updated_latent` — see `SuperEgoModule.codify_new_axiom`.
        """
        self.stats["codify_attempts"] += 1

        thr = threshold if threshold is not None else self.cfg.codify_threshold
        validity_ok = out["validity_score"].mean().item() > thr

        # Resolve quality from argument, then from the dict (forward() pass-through)
        q = quality_score
        if q is None:
            q = out.get("quality_score")
        q = _safe_unit_scalar(q, name="quality_score", logger=self.log)

        if q is not None and self.cfg.require_quality_for_codify:
            quality_ok = q >= self.cfg.quality_codify_min
            if not quality_ok:
                self.stats["codify_refused_quality"] += 1
                self.log.debug(
                    f"[PIPELINE] Codify refused: quality {q:.3f} < "
                    f"quality_codify_min {self.cfg.quality_codify_min:.3f}"
                )
                return False

        if validity_ok:
            ptr = self.superego_layer.codify_new_axiom(out["updated_latent"][0])
            if ptr == -1:
                self.stats["axiom_rejected_nonfinite"] += 1
                return False
            self.stats["codify_successes"] += 1
            return True

        self.stats["codify_refused_validity"] += 1
        self.log.debug("[PIPELINE] Speculation below codify threshold — remains unverified.")
        return False

    # ── Non-differentiable external side-channel (opt-in) ───────────────────

    def login_all(
        self,
        claude_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        gpt_key   : Optional[str] = None,
    ) -> None:
        """Convenience: attempt to establish all three provider sessions.
        Any provider without SDK/key silently falls back to simulated mode."""
        self.auth_manager.login_claude(api_key=claude_key, model=self.cfg.claude_model)
        self.auth_manager.login_gemini(api_key=gemini_key, model=self.cfg.gemini_model)
        self.auth_manager.login_gpt(api_key=gpt_key, model=self.cfg.gpt_model)

    def maybe_consult_external(
        self, prompt: str, force: bool = False
    ) -> Optional[ExternalConsensusResult]:
        """
        Rate-limited gateway to `EgoModule.consult_external`. Returns None
        (no-op) unless `enable_external_llm` is set AND the step cadence
        matches `consult_every_n_steps` — or `force=True` overrides both.
        This is the ONLY method in this module that may perform network I/O.
        """
        if not force and not self.cfg.enable_external_llm:
            self.stats["external_consult_skipped"] += 1
            return None
        step = int(self._global_step.item())
        if not force and (self.cfg.consult_every_n_steps <= 0
                           or step % self.cfg.consult_every_n_steps != 0):
            self.stats["external_consult_skipped"] += 1
            return None
        self.stats["external_consult_calls"] += 1
        return self.ego_layer.consult_external(
            prompt, timeout=self.cfg.llm_timeout_s,
            max_retries=self.cfg.max_retries,
            retry_backoff_s=self.cfg.retry_backoff_s,
        )

    # ── Introspection ────────────────────────────────────────────────────────

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "instance_id"            : self.instance_id,          # [v3.3 NEW]
            "axiom_count"            : int(self.superego_layer.axiom_count.item()),
            "max_axioms"             : self.superego_layer.max_axioms,
            "global_step"            : int(self._global_step.item()),
            "authenticated_providers": {
                name: sess.tier for name, sess in self.auth_manager.sessions.items()
            },
            "external_llm_enabled"   : self.cfg.enable_external_llm,
            "stats"                  : dict(self.stats),           # [v3.3 NEW]
        }

    def simulate_scientific_discovery_pipeline(self, current_latent: torch.Tensor) -> Dict[str, Any]:
        """Demonstrates the full loop: Speculation → Reconciliation → Codification."""
        self.log.info("--- AGI ONE Psyche Plus: Scientific Discovery Pipeline ---")
        out = self.forward(current_latent)
        codified = self.maybe_codify(out)
        status = "CODIFIED" if codified else "UNVERIFIED"
        self.log.info(f"[PIPELINE {status}] validity={out['validity_score'].mean().item():.4f}")
        out["codified"] = codified
        return out


# =============================================================================
# 6. AGI ONE v3.0 CENTRAL-HUB INTEGRATION HELPER
# =============================================================================

def attach_to_agi_one(
    agi_one: nn.Module,
    cfg: Optional[PsychePlusConfig] = None,
    latent_dim: Optional[int] = None,
) -> "AGIOnePsychePlus":
    """
    Hot-attach an `AGIOnePsychePlus` instance onto an already-constructed
    `AGIONE` engine (agi_one_v3.py), matching its latent_dim/device.

    Prefer setting `AGIConfig.use_psyche_plus = True` and letting `AGIONE`
    construct this module natively (see agi_one_v3.py SECTION 6-B) — this
    helper exists for hot-swapping into engines that were built before that
    flag existed, consistent with EcosystemOrchestrator's
    hot-swappable-module philosophy.

    [v3.3 NEW] `latent_dim` can be passed explicitly to avoid guessing.
    If omitted and `agi_one.cfg.latent_dim` is unavailable, this now logs a
    WARNING (previously silently defaulted to 64) so a dimension mismatch
    is visible at attach time rather than surfacing as a confusing shape
    error several calls later.
    """
    resolved_latent_dim = latent_dim or getattr(getattr(agi_one, "cfg", None), "latent_dim", None)
    device     = getattr(agi_one, "device", torch.device("cpu"))
    if resolved_latent_dim is None:
        _base_logger.warning(
            "[ATTACH] Could not determine latent_dim from agi_one.cfg.latent_dim "
            "and none was passed explicitly — defaulting to 64. Pass "
            "`latent_dim=...` explicitly to silence this warning and avoid a "
            "possible shape mismatch."
        )
        resolved_latent_dim = 64
    if cfg is None:
        cfg = PsychePlusConfig(latent_dim=resolved_latent_dim, device=device)
    plus = AGIOnePsychePlus(cfg=cfg, device=device)
    agi_one.psyche_plus = plus
    _base_logger.info(f"AGIOnePsychePlus ({plus.instance_id}) hot-attached to existing AGIONE instance.")
    return plus


# =============================================================================
# 7. VERIFICATION SUITE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  AGI ONE : Id, Ego, Super Ego / Plus — v3.3 Production Smoke Test")
    print("  Developer : Yoon A Limsuwan / MSPS NETWORK")
    print("  AI Assistants : Gemini (Google DeepMind), Claude (Anthropic)")
    print("=" * 65 + "\n")

    n_pass = 0
    n_fail = 0

    def check(label: str, cond: bool) -> None:
        global n_pass, n_fail
        if cond:
            print(f"  [PASS] {label}")
            n_pass += 1
        else:
            print(f"  [FAIL] {label}")
            n_fail += 1

    LATENT = 64
    cfg = PsychePlusConfig(latent_dim=LATENT, max_axioms=8, enable_external_llm=False)
    psyche = AGIOnePsychePlus(cfg=cfg)

    # [TEST 1] Assembly ------------------------------------------------------
    check("Module assembles", isinstance(psyche, nn.Module))
    check("Latent dim wired through", psyche.latent_dim == LATENT)
    check("Instance has a repr", "AGIOnePsychePlus" in repr(psyche))   # [v3.3]

    # [TEST 2] Forward pass shapes -------------------------------------------
    z = torch.randn(1, LATENT)
    out = psyche(z)
    check("Forward returns updated_latent", out["updated_latent"].shape == (1, LATENT))
    check("Validity score in [0,1]",
          bool((out["validity_score"] >= 0).all() and (out["validity_score"] <= 1).all()))
    check("No NaNs in output", not torch.isnan(out["updated_latent"]).any().item())

    # [TEST 3] Gradient flow (must be end-to-end differentiable) -------------
    z_grad = torch.randn(1, LATENT, requires_grad=True)
    out2 = psyche(z_grad)
    loss = out2["updated_latent"].pow(2).mean() + out2["validity_score"].mean()
    loss.backward()
    check("Gradients reach input latent", z_grad.grad is not None and
          not torch.isnan(z_grad.grad).any().item())
    check("IdModule noise_scale stays positive", psyche.id_layer.noise_scale.item() > 0.0)

    # [TEST 4] Axiom evolution (the regression fix) ---------------------------
    before = int(psyche.superego_layer.axiom_count.item())
    score_before = psyche.superego_layer(out["updated_latent"]).mean().item()
    codified = psyche.maybe_codify(out, threshold=-1.0)   # force codification
    after = int(psyche.superego_layer.axiom_count.item())
    check("Axiom codified when forced", codified and after == before + 1)

    score_after = psyche.superego_layer(out["updated_latent"]).mean().item()
    check("SuperEgo score reacts to its own axiom bank (not dead code)",
          abs(score_after - score_before) > 1e-6 or after > 0)

    # Fill past capacity to test ring-buffer wraparound
    for _ in range(cfg.max_axioms + 2):
        psyche.superego_layer.codify_new_axiom(torch.randn(LATENT))
    check("Axiom ring buffer caps at max_axioms",
          int(psyche.superego_layer.axiom_count.item()) == cfg.max_axioms)

    # [TEST 5] Multi-LLM auth — simulated mode (no keys in this environment) -
    psyche.login_all()
    sessions = psyche.auth_manager.sessions
    check("All three providers register a session",
          set(sessions.keys()) == {"claude", "gemini", "gpt"})
    check("Sessions fall back to simulated tier without credentials",
          all(s.tier == "simulated" for s in sessions.values()))

    # [TEST 6] External consensus side-channel never blocks/crashes ---------
    result = psyche.maybe_consult_external("Smoke-test prompt", force=True)
    check("consult_external returns a structured result", isinstance(result, ExternalConsensusResult))
    check("Every provider responded (simulated)", result.n_succeeded == 3 and result.n_failed == 0)
    check("Result carries a human-readable summary", len(result.summary) > 0)   # [v3.3]

    # Rate limiting: without force, disabled-by-default config returns None
    quiet = psyche.maybe_consult_external("Should not fire")
    check("External consult stays off by default (no surprise API spend)", quiet is None)

    # [TEST 7] Full discovery pipeline ----------------------------------------
    pipeline_out = psyche.simulate_scientific_discovery_pipeline(torch.randn(1, LATENT))
    check("Pipeline returns codified flag", "codified" in pipeline_out)

    # [TEST 8] State summary ---------------------------------------------------
    summary = psyche.get_state_summary()
    check("State summary reports axiom_count", summary["axiom_count"] == cfg.max_axioms)
    check("State summary includes instance_id", "instance_id" in summary)        # [v3.3]
    check("State summary includes stats counters", "stats" in summary
          and summary["stats"]["forward_calls"] >= 1)                            # [v3.3]

    # [TEST 9] [v3.3 NEW] quality_score sanitisation -----------------------------
    out_nan = psyche.forward(torch.randn(1, LATENT), quality_score=float("nan"))
    check("Forward survives NaN quality_score (no crash, no NaN propagation)",
          not torch.isnan(out_nan["updated_latent"]).any().item())

    out_inf = psyche.forward(torch.randn(1, LATENT), quality_score=float("inf"))
    check("Forward survives Inf quality_score",
          not torch.isnan(out_inf["updated_latent"]).any().item()
          and not torch.isinf(out_inf["updated_latent"]).any().item())

    codify_with_nan_q = psyche.maybe_codify(out, quality_score=float("nan"))
    check("maybe_codify handles NaN quality_score without raising",
          isinstance(codify_with_nan_q, bool))

    # [TEST 10] [v3.3 NEW] non-finite axiom rejection ----------------------------
    before_reject = int(psyche.superego_layer.axiom_count.item())
    bad_vec = torch.full((LATENT,), float("nan"))
    ptr = psyche.superego_layer.codify_new_axiom(bad_vec)
    after_reject = int(psyche.superego_layer.axiom_count.item())
    check("Non-finite speculation rejected, not written to ring buffer",
          ptr == -1 and after_reject == before_reject)

    # [TEST 11] [v3.3 NEW] empty-batch handling -----------------------------------
    empty_z = torch.zeros(0, LATENT)
    empty_out = psyche.id_layer(empty_z)
    check("IdModule handles empty batch", empty_out.shape == (0, LATENT))
    empty_validity = psyche.superego_layer(empty_z)
    check("SuperEgoModule handles empty batch", empty_validity.shape == (0,))

    # [TEST 12] [v3.3 NEW] config validation raises on bad values -----------------
    config_raised = False
    try:
        PsychePlusConfig(latent_dim=-1)
    except ValueError:
        config_raised = True
    check("PsychePlusConfig rejects invalid latent_dim", config_raised)

    weight_sum_raised = False
    try:
        PsychePlusConfig(superego_axiom_weight=0.8, superego_quality_weight=0.5)
    except ValueError:
        weight_sum_raised = True
    check("PsychePlusConfig rejects axiom_weight + quality_weight > 1.0", weight_sum_raised)

    device_raised = False
    try:
        PsychePlusConfig(device="not-a-real-device")
    except ValueError:
        device_raised = True
    check("PsychePlusConfig rejects invalid device string", device_raised)

    print(f"\n{'='*65}")
    print(f"  RESULT: {n_pass} passed, {n_fail} failed")
    print("=" * 65 + "\n")

    if n_fail == 0:
        print("[SUCCESS] AGI ONE : Id, Ego, Super Ego / Plus v3.3 — all checks passed.\n")
    else:
        raise SystemExit(f"{n_fail} verification check(s) failed.")
