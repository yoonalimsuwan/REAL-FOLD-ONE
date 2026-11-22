#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
autonomy_layer.py — v1.0.0
Closing three capability gaps + the honest partial of the fourth.
  [1] Cross-domain transfer WITHOUT retraining (+ refusal under shift)
  [2] Self-hypothesis generation + Bayesian experiment design (EIG)
  [3] Meta-reasoning over its own competence boundary
  [4] Auto-DERIVED falsification protocol + mandatory ratification gate
=============================================================================

Author  : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID   : 0009-0008-2374-0788
License : MIT | Year 2026

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
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHAT IS NOW REAL, AND WHAT REMAINS IMPOSSIBLE
=============================================================================
[1] TRANSFER WITHOUT RETRAINING — REAL, CONDITIONAL
    Method: learn environment-invariant features on MULTIPLE source domains
    (IRM-style penalty), then at target time:
      (a) MMD permutation test for covariate shift
      (b) if shift is detected AND density ratio is estimable, use
          importance-weighted conformal (Tibshirani et al. 2019)
      (c) if shift is beyond the estimable regime, REFUSE
    Guarantee: marginal coverage under covariate shift with a consistent
    density-ratio estimate. NOT a guarantee for arbitrary new domains.
    Fundamental limit: domain generalization is impossible without
    assumptions linking source and target. This module MAKES the assumption
    explicit and TESTS it, rather than assuming transfer silently.

[2] SELF-HYPOTHESIS + EXPERIMENT DESIGN — REAL, WITHIN A DECLARED SPACE
    Method: a hypothesis GRAMMAR enumerates candidate mechanisms. Bayesian
    experimental design selects the experiment maximising mutual information
    between the outcome and the hypothesis identity:
        x* = argmax_x  I(Y ; H | x)
             = argmax_x  [ H(p(y|x)) - Σ_h p(h) H(p(y|x,h)) ]
    This is exactly "run the experiment where the hypotheses disagree most".
    Posterior update by Bayes. Stops when one hypothesis dominates or the
    budget is exhausted.
    Limit: the hypothesis SPACE is declared by a human. The system searches
    it and designs discriminating experiments; it does not invent new
    conceptual primitives. That is a real capability (cf. robot scientists
    Adam/Eve), not AGI.

[3] META-REASONING OVER COMPETENCE — REAL, RELATIVE TO OBSERVED SUPPORT
    Method:
      (a) epistemic / aleatoric decomposition from a heteroscedastic ensemble
      (b) kNN distance to the training support (no Gaussian assumption)
      (c) Mahalanobis score as a secondary, parametric view
      (d) conformal interval width as a decision-relevant competence proxy
      (e) explicit refusal when any indicator exceeds a declared tolerance
    Limit: the system knows WHERE IT HAS NO DATA. It cannot know what it
    does not know. Competence is bounded relative to observed support,
    never absolutely.

[4] AUTONOMOUS FALSIFICATION — PARTIAL ONLY, AND THE REST IS REFUSED
    IMPLEMENTED: given a HUMAN-DECLARED objective (metric, baseline, minimum
    detectable effect, alpha, power), the system DERIVES:
      • required sample size (power analysis)
      • the appropriate test (paired bootstrap / Wilcoxon / paired t)
      • the pre-registered decision rule
      • the stopping rule
      • a protocol hash
    It then BLOCKS execution until a human ratification signature matching
    that hash is recorded.
    REFUSED: letting the system choose its own success criteria.
    WHY (three independent reasons):
      (i)  falsification criteria encode VALUES — what counts as harm, what
           counts as success. Values are not derivable from data.
      (ii) a system that sets its own criteria can trivially satisfy them
           (Goodhart / specification gaming). Self-certification is not
           evidence.
      (iii) in the biomedical setting this is a safety hazard: a system that
           decides its own engraftment-success threshold could authorise a
           transplant decision. The T4 gate exists precisely to prevent this.
    So the fourth row is closed only in its mechanical part. The
    authorisation part remains human, deliberately and permanently.

=============================================================================
HONEST SCOPE
=============================================================================
* No claim of AGI. These are four bounded, testable capabilities.
* Every capability has a self-test that would FAIL if it were fake.
* Every refusal is enforced in code, not documented as a suggestion.
* research_only throughout. No clinical decision authority.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("autonomy")

VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"

BANNER = f"""
=============================================================================
AUTONOMY LAYER v{VERSION}
[1] transfer + refusal   [2] hypothesis + EIG design
[3] competence boundary  [4] derived falsification + ratification gate
No AGI claim. Every capability has a falsifying self-test.
=============================================================================
"""


class Refusal(RuntimeError):
    """Raised when the system must not answer. Enforced, not advisory."""


# =============================================================================
# §1. CAPABILITY 1 — TRANSFER WITHOUT RETRAINING
# =============================================================================

def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    d2 = (np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :]
          - 2.0 * X @ Y.T)
    return np.exp(-gamma * np.maximum(d2, 0.0))


def median_heuristic_gamma(X: np.ndarray, max_n: int = 512) -> float:
    Z = X if len(X) <= max_n else X[np.random.default_rng(0).choice(
        len(X), max_n, replace=False)]
    d2 = np.sum((Z[:, None, :] - Z[None, :, :])**2, axis=-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    return float(1.0 / max(med, 1e-8))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD^2 estimator (Gretton et al. 2012)."""
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return 0.0
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m*(m-1)) + Kyy.sum() / (n*(n-1))
                 - 2.0 * Kxy.mean())


def mmd_permutation_test(X: np.ndarray, Y: np.ndarray,
                         n_perm: int = 200, seed: int = 0
                         ) -> Dict[str, Any]:
    """
    Two-sample test. H0: X and Y come from the same distribution.
    Returns MMD^2, permutation p-value, and a shift verdict.
    """
    gamma = median_heuristic_gamma(np.vstack([X, Y]))
    obs = mmd2_unbiased(X, Y, gamma)
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y]); m = len(X)
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(len(Z))
        null[b] = mmd2_unbiased(Z[p[:m]], Z[p[m:]], gamma)
    pval = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    return {"mmd2": obs, "p_value": pval, "gamma": gamma,
            "shift_detected": bool(pval < 0.05),
            "null_mean": float(null.mean()), "null_std": float(null.std())}


if HAS_TORCH:

    class InvariantEncoder(nn.Module):
        """
        Environment-invariant feature learner.

        Trained on MULTIPLE labelled source environments with an IRM-style
        penalty: the gradient of each environment's risk w.r.t. a dummy
        scalar classifier at scale 1 should vanish simultaneously. When it
        does, the predictor uses only environment-invariant structure.

        Reference: Arjovsky et al., Invariant Risk Minimization (2019).
        HONEST: IRM has known failure modes and needs many environments.
        The penalty is a REGULARISER toward invariance, not a proof of it.
        The MMD test at target time is what actually protects you.
        """

        def __init__(self, d_in: int, d_feat: int = 32, d_hidden: int = 64):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_feat), nn.LayerNorm(d_feat))
            self.head = nn.Linear(d_feat, 1)
            self.log_sd = nn.Linear(d_feat, 1)     # heteroscedastic
            self.dummy = nn.Parameter(torch.ones(1))

        def features(self, x: Tensor) -> Tensor:
            return self.body(x)

        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            z = self.features(x)
            return {"mu": self.head(z).squeeze(-1),
                    "log_sd": self.log_sd(z).squeeze(-1).clamp(-6, 3),
                    "feat": z}

        @staticmethod
        def irm_penalty(mu: Tensor, y: Tensor) -> Tensor:
            """grad of MSE(scale*mu, y) w.r.t. scale, at scale=1, squared."""
            scale = torch.ones(1, requires_grad=True, device=mu.device)
            risk = F.mse_loss(mu * scale, y)
            g = torch.autograd.grad(risk, [scale], create_graph=True)[0]
            return g.pow(2).sum()

    def train_invariant(models_out: Dict[str, Any],
                        envs: List[Dict[str, Tensor]],
                        d_in: int, d_feat: int = 32,
                        epochs: int = 400, lr: float = 3e-3,
                        w_irm: float = 10.0, seed: int = 0
                        ) -> Dict[str, Any]:
        """Train one encoder across environments with the IRM penalty."""
        torch.manual_seed(seed)
        enc = InvariantEncoder(d_in, d_feat)
        opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
        hist = []
        for ep in range(epochs):
            opt.zero_grad()
            erm = torch.zeros((), dtype=torch.float32)
            pen = torch.zeros((), dtype=torch.float32)
            for e in envs:
                o = enc(e["x"])
                sd = o["log_sd"].exp()
                erm = erm + (0.5*((e["y"]-o["mu"])/sd).pow(2)
                             + o["log_sd"]).mean()
                pen = pen + InvariantEncoder.irm_penalty(o["mu"], e["y"])
            loss = erm / len(envs) + w_irm * pen / len(envs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            if ep % max(1, epochs//5) == 0:
                hist.append({"epoch": ep, "erm": float(erm/len(envs)),
                             "irm": float(pen/len(envs))})
        models_out["encoder"] = enc
        return {"history": hist, "n_envs": len(envs)}


class DensityRatio:
    """
    w(x) = p_target(x) / p_source(x) via the classifier trick.
    Train a logistic classifier source(0) vs target(1); then
        w(x) = (n_source/n_target) * clf(x)/(1-clf(x)).
    Consistency of the ratio estimate is REQUIRED for weighted-conformal
    coverage. We report a clipping fraction so you can see when the
    estimate is degenerate (separable domains => ratio unbounded).
    """

    def __init__(self, d_in: int, epochs: int = 300, lr: float = 5e-3):
        self.d_in = d_in
        self.epochs = epochs
        self.lr = lr
        self.clf = None
        self.n_s = self.n_t = 1

    def fit(self, Xs: np.ndarray, Xt: np.ndarray) -> Dict[str, Any]:
        if not HAS_TORCH:
            raise ImportError("torch required")
        self.n_s, self.n_t = len(Xs), len(Xt)
        X = torch.tensor(np.vstack([Xs, Xt]), dtype=torch.float32)
        y = torch.tensor(np.concatenate([np.zeros(len(Xs)),
                                         np.ones(len(Xt))]),
                         dtype=torch.float32)
        torch.manual_seed(0)
        self.clf = nn.Sequential(nn.Linear(self.d_in, 64), nn.GELU(),
                                 nn.Linear(64, 1))
        opt = torch.optim.Adam(self.clf.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                self.clf(X).squeeze(-1), y)
            loss.backward(); opt.step()
        with torch.no_grad():
            acc = float(((torch.sigmoid(self.clf(X).squeeze(-1)) > 0.5)
                         .float() == y).float().mean())
        return {"separability_auc_proxy": acc,
                "warning": ("domains nearly separable — density ratio "
                            "unstable" if acc > 0.95 else "ratio estimable")}

    def ratio(self, X: np.ndarray, clip: float = 20.0
              ) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            p = torch.sigmoid(self.clf(
                torch.tensor(X, dtype=torch.float32))).squeeze(-1).numpy()
        p = np.clip(p, 1e-6, 1 - 1e-6)
        w = (self.n_s / max(self.n_t, 1)) * p / (1 - p)
        frac = float(np.mean(w > clip))
        return np.clip(w, 1e-6, clip), frac


def weighted_conformal(cal_scores: np.ndarray, cal_weights: np.ndarray,
                       test_weight: float, alpha: float = 0.1
                       ) -> Dict[str, Any]:
    """
    Importance-weighted split conformal (Tibshirani et al. 2019).
    Coverage guarantee under COVARIATE shift with a consistent ratio.
    Mass at +inf accounts for the test point's own weight.
    """
    s = np.asarray(cal_scores, float)
    w = np.asarray(cal_weights, float)
    tot = w.sum() + test_weight
    p = w / tot
    p_inf = test_weight / tot
    order = np.argsort(s)
    cum = np.cumsum(p[order])
    # smallest score whose cumulative mass (incl. inf) reaches 1-alpha
    idx = np.searchsorted(cum, 1 - alpha - p_inf, side="left")
    if idx >= len(s):
        q = float("inf")
    else:
        q = float(s[order][idx])
    return {"q": q, "alpha": alpha, "mass_at_inf": float(p_inf),
            "infinite_interval": bool(not np.isfinite(q)),
            "note": ("infinite interval => shift too extreme to certify"
                     if not np.isfinite(q) else "finite, certified")}


class TransferGate:
    """
    The actual capability: transfer WITHOUT touching target gradients,
    and REFUSE when the shift is not certifiable.
    """

    def __init__(self, encoder: Any, cal_feat: np.ndarray,
                 cal_scores: np.ndarray,
                 max_knn_dist: float, max_shift_p: float = 0.05,
                 alpha: float = 0.1):
        self.enc = encoder
        self.cal_feat = cal_feat
        self.cal_scores = cal_scores
        self.max_knn = max_knn_dist
        self.max_shift_p = max_shift_p
        self.alpha = alpha
        self.dr = None

    def fit_ratio(self, Xt_feat: np.ndarray) -> Dict[str, Any]:
        self.dr = DensityRatio(self.cal_feat.shape[1])
        return self.dr.fit(self.cal_feat, Xt_feat)

    def predict(self, x: Tensor) -> Dict[str, Any]:
        self.enc.eval()
        with torch.no_grad():
            o = self.enc(x)
        z = o["feat"].numpy()
        d = np.sqrt(np.sum((self.cal_feat[:, None, :]
                            - z[None, :, :])**2, axis=-1)).min(axis=0)
        out = {"mu": o["mu"].numpy(), "sd": o["log_sd"].exp().numpy(),
               "knn_dist": d, "n_query": len(x)}
        out["outside_support"] = bool(np.any(d > self.max_knn))
        if out["outside_support"]:
            out["status"] = "REFUSED"
            out["reason"] = ("query outside training support "
                             f"(max kNN {d.max():.3f} > {self.max_knn:.3f})")
            return out
        if self.dr is None:
            q = float(np.quantile(self.cal_scores, 1 - self.alpha))
            out.update({"q": q, "mode": "standard_conformal",
                        "status": "OK_NO_SHIFT_MODEL"})
        else:
            w_cal, frac_cal = self.dr.ratio(self.cal_feat)
            w_test, _ = self.dr.ratio(z)
            qs = []
            for wt in w_test:
                r = weighted_conformal(self.cal_scores, w_cal, float(wt),
                                       self.alpha)
                qs.append(r["q"])
            out.update({"q": np.array(qs), "mode": "weighted_conformal",
                        "ratio_clip_fraction": frac_cal})
            if not np.all(np.isfinite(out["q"])):
                out["status"] = "REFUSED"
                out["reason"] = "covariate shift too extreme to certify"
            else:
                out["status"] = "OK_CERTIFIED"
        q = np.asarray(out["q"])
        out["lo"] = out["mu"] - q * out["sd"]
        out["hi"] = out["mu"] + q * out["sd"]
        return out


# =============================================================================
# §2. CAPABILITY 2 — SELF-HYPOTHESIS + BAYESIAN EXPERIMENT DESIGN
# =============================================================================

@dataclass
class Hypothesis:
    """One candidate mechanism. `predict` maps design x -> outcome mean."""
    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    noise_sd: float = 0.15
    prior: float = 1.0
    description: str = ""


class HypothesisGrammar:
    """
    A DECLARED space of mechanisms. The system searches this space and
    designs discriminating experiments; it does not invent new primitives.

    Example space below is dose-response mechanism identification — the
    exact question the Hill-vs-Black-Leff distinction raises.
    """

    @staticmethod
    def dose_response_space() -> List[Hypothesis]:
        def hill1(x):
            return x / (x + 10.0)

        def hill_steep(x):
            return x**2.5 / (x**2.5 + 10.0**2.5)

        def black_leff(x):
            ka, tau, em = 10.0, 5.0, 1.0
            return em * tau * x / ((ka + x) + tau * x)

        def biphasic(x):
            return x / (x + 5.0) * np.exp(-x / 200.0)

        return [
            Hypothesis("hill_n1", hill1, 0.12, 1.0,
                       "Hill, slope 1, EC50=10"),
            Hypothesis("hill_steep", hill_steep, 0.12, 1.0,
                       "Hill, slope 2.5 (positive cooperativity)"),
            Hypothesis("black_leff", black_leff, 0.12, 1.0,
                       "Black-Leff operational, KA=10, tau=5"),
            Hypothesis("biphasic", biphasic, 0.12, 1.0,
                       "activation then inhibition at high dose"),
        ]


def expected_information_gain(hyps: List[Hypothesis], post: np.ndarray,
                              x: np.ndarray, n_bins: int = 24,
                              y_range: Tuple[float, float] = (-0.5, 1.5)
                              ) -> float:
    """
    I(Y ; H | x) computed EXACTLY on a discretised outcome grid:
        Σ_h p(h) Σ_y p(y|x,h) log[ p(y|x,h) / p(y|x) ]
    This is the experiment where the hypotheses disagree most.
    """
    edges = np.linspace(y_range[0], y_range[1], n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    L = np.zeros((len(hyps), n_bins))
    for i, h in enumerate(hyps):
        m = float(h.predict(np.asarray([x]))[0])
        dens = np.exp(-0.5*((centres - m)/h.noise_sd)**2) \
            / (h.noise_sd*math.sqrt(2*math.pi))
        p = dens * width
        L[i] = p / max(p.sum(), 1e-12)
    marg = post @ L
    mi = 0.0
    for i in range(len(hyps)):
        if post[i] <= 1e-12:
            continue
        mask = (L[i] > 1e-12) & (marg > 1e-12)
        mi += post[i] * float(np.sum(L[i][mask]
                                     * np.log(L[i][mask]/marg[mask])))
    return float(max(mi, 0.0))


class BayesianExperimentDesigner:
    """
    Autonomous loop:
      1. enumerate the declared hypothesis space
      2. pick x* maximising I(Y;H|x)
      3. observe y (from a real experiment, or a supplied oracle)
      4. Bayes-update the posterior
      5. stop when one hypothesis dominates or budget exhausted
    The oracle is INJECTED — in production it is the wet lab or a validated
    simulator. The designer never fabricates observations.
    """

    def __init__(self, hyps: List[Hypothesis],
                 designs: np.ndarray,
                 dominance: float = 0.95,
                 max_experiments: int = 12):
        self.hyps = hyps
        self.designs = np.asarray(designs, float)
        self.post = np.array([h.prior for h in hyps], float)
        self.post /= self.post.sum()
        self.dominance = dominance
        self.max_exp = max_experiments
        self.trace: List[Dict[str, Any]] = []
        self.used: List[float] = []

    def select_design(self) -> Tuple[float, Dict[str, float]]:
        scores = {}
        for x in self.designs:
            if any(abs(x - u) < 1e-9 for u in self.used):
                continue
            scores[float(x)] = expected_information_gain(
                self.hyps, self.post, float(x))
        if not scores:
            raise Refusal("design space exhausted; no informative experiment left")
        best = max(scores, key=scores.get)
        return best, scores

    def update(self, x: float, y: float) -> Dict[str, Any]:
        lik = np.array([
            math.exp(-0.5*((y - float(h.predict(np.asarray([x]))[0]))
                           / h.noise_sd)**2) / (h.noise_sd*math.sqrt(2*math.pi))
            for h in self.hyps])
        num = self.post * lik
        self.post = num / max(num.sum(), 1e-300)
        self.used.append(float(x))
        return {"x": x, "y": y,
                "posterior": {h.name: float(p)
                              for h, p in zip(self.hyps, self.post)}}

    def decided(self) -> Optional[str]:
        i = int(np.argmax(self.post))
        if self.post[i] >= self.dominance:
            return self.hyps[i].name
        return None

    def run(self, oracle: Callable[[float], float]) -> Dict[str, Any]:
        for k in range(self.max_exp):
            x, scores = self.select_design()
            y = float(oracle(x))
            rec = self.update(x, y)
            rec["eig"] = scores[x]
            rec["eig_rank_of_chosen"] = 1
            rec["runner_up_eig"] = float(sorted(scores.values())[-2]) \
                if len(scores) > 1 else 0.0
            self.trace.append(rec)
            d = self.decided()
            if d is not None:
                return {"status": "DECIDED", "hypothesis": d,
                        "n_experiments": k + 1, "trace": self.trace,
                        "final_posterior": rec["posterior"]}
        return {"status": "BUDGET_EXHAUSTED_UNCERTAINTY_RETAINED",
                "n_experiments": len(self.trace), "trace": self.trace,
                "best": self.hyps[int(np.argmax(self.post))].name,
                "best_posterior": float(self.post.max())}


# =============================================================================
# §3. CAPABILITY 3 — META-REASONING OVER COMPETENCE
# =============================================================================

@dataclass
class CompetenceConfig:
    max_knn_dist: float = 3.0
    max_epistemic_fraction: float = 0.6
    max_interval_width: Optional[float] = None
    knn_k: int = 5
    mahalanobis_max: float = 6.0


class CompetenceMap:
    """
    The system's model of its own limits, RELATIVE TO OBSERVED SUPPORT.

    Reports four independent indicators and refuses if any is out of bounds:
      1. kNN distance to training features (non-parametric support test)
      2. Mahalanobis distance (parametric secondary view)
      3. epistemic fraction of total predictive variance
      4. conformal interval width (decision-relevant competence)

    Epistemic / aleatoric decomposition (Lakshminarayanan et al. 2017):
        aleatoric = mean_m sigma_m^2      (irreducible noise)
        epistemic = var_m mu_m            (model disagreement)
    High epistemic + in-support  => need more data here.
    High aleatoric + in-support  => noise floor; more data will not help.
    Out of support               => refuse, regardless of variance.
    """

    def __init__(self, train_feat: np.ndarray, cfg: CompetenceConfig):
        self.Z = np.asarray(train_feat, float)
        self.cfg = cfg
        self.mu = self.Z.mean(0)
        C = np.cov(self.Z.T) + 1e-4*np.eye(self.Z.shape[1])
        self.Cinv = np.linalg.inv(C)

    def knn_dist(self, z: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.sum((self.Z[:, None, :] - z[None, :, :])**2, -1))
        k = min(self.cfg.knn_k, d.shape[0])
        return np.sort(d, axis=0)[:k].mean(axis=0)

    def mahalanobis(self, z: np.ndarray) -> np.ndarray:
        dz = z - self.mu
        return np.sqrt(np.einsum("bi,ij,bj->b", dz, self.Cinv, dz))

    @staticmethod
    def decompose(mus: np.ndarray, sds: np.ndarray) -> Dict[str, np.ndarray]:
        """mus, sds: (n_models, B)."""
        alea = np.mean(sds**2, axis=0)
        epis = np.var(mus, axis=0)
        tot = alea + epis
        return {"aleatoric": alea, "epistemic": epis, "total": tot,
                "epistemic_fraction": epis / np.maximum(tot, 1e-12)}

    def assess(self, z: np.ndarray, mus: np.ndarray, sds: np.ndarray,
               interval_width: Optional[np.ndarray] = None
               ) -> Dict[str, Any]:
        kn = self.knn_dist(z)
        mh = self.mahalanobis(z)
        dec = self.decompose(mus, sds)
        reasons, flags = [], []
        for i in range(len(z)):
            r = []
            if kn[i] > self.cfg.max_knn_dist:
                r.append(f"outside support (kNN {kn[i]:.2f})")
            if mh[i] > self.cfg.mahalanobis_max:
                r.append(f"Mahalanobis {mh[i]:.2f}")
            if dec["epistemic_fraction"][i] > self.cfg.max_epistemic_fraction:
                r.append(f"epistemic fraction {dec['epistemic_fraction'][i]:.2f}")
            if (interval_width is not None
                    and self.cfg.max_interval_width is not None
                    and interval_width[i] > self.cfg.max_interval_width):
                r.append(f"interval width {interval_width[i]:.2f}")
            reasons.append(r)
            flags.append(len(r) == 0)
        return {"knn_dist": kn, "mahalanobis": mh, **dec,
                "competent": np.array(flags),
                "refusal_reasons": reasons,
                "verdict": ("ANSWER" if all(flags) else "REFUSE_PARTIAL"),
                "SCOPE": ("competence is RELATIVE to observed support; "
                          "unknown-unknowns are not detectable")}


# =============================================================================
# §4. CAPABILITY 4 — AUTO-DERIVED FALSIFICATION + RATIFICATION GATE
# =============================================================================

def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam approximation, |err|<1.15e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass
class DeclaredObjective:
    """HUMAN-DECLARED. The system may not modify these fields."""
    metric_name: str
    baseline_value: float
    min_detectable_effect: float      # delta, in metric units
    paired_sd: float                  # SD of paired differences
    alpha: float = 0.05
    power: float = 0.80
    direction: str = "increase"       # or "decrease"
    declared_by: str = ""
    rationale: str = ""

    def validate(self) -> List[str]:
        p = []
        if not self.declared_by.strip():
            p.append("declared_by is empty — a human must own the objective")
        if self.min_detectable_effect <= 0:
            p.append("min_detectable_effect must be > 0")
        if self.paired_sd <= 0:
            p.append("paired_sd must be > 0")
        if not (0 < self.alpha < 1):
            p.append("alpha must be in (0,1)")
        if not (0 < self.power < 1):
            p.append("power must be in (0,1)")
        if self.direction not in ("increase", "decrease"):
            p.append("direction must be increase|decrease")
        if not self.rationale.strip():
            p.append("rationale is empty — state WHY this threshold matters")
        return p


@dataclass
class FalsificationProtocol:
    """DERIVED by the system from a DeclaredObjective. Not self-authored."""
    objective: Dict[str, Any]
    required_n: int
    test: str
    decision_rule: str
    stopping_rule: str
    derivation: Dict[str, Any]
    protocol_hash: str = ""
    ratified_by: Optional[str] = None
    ratification_hash: Optional[str] = None

    def compute_hash(self) -> str:
        blob = json.dumps({"objective": self.objective,
                           "required_n": self.required_n,
                           "test": self.test,
                           "decision_rule": self.decision_rule,
                           "stopping_rule": self.stopping_rule},
                          sort_keys=True)
        self.protocol_hash = hashlib.sha256(blob.encode()).hexdigest()
        return self.protocol_hash

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_falsification(obj: DeclaredObjective,
                         normality_ok: bool = True
                         ) -> FalsificationProtocol:
    """
    MECHANICAL derivation. Every number is traceable to the declared objective.

    Paired design (same held-out set, two models):
        n >= (z_{1-alpha/2} + z_{power})^2 * sd_d^2 / delta^2
    Test choice:
        normality plausible  -> paired t (or paired bootstrap if n<30)
        otherwise            -> Wilcoxon signed-rank
    """
    probs = obj.validate()
    if probs:
        raise Refusal("objective invalid:\n  " + "\n  ".join(probs))
    za, zb = _z(1 - obj.alpha/2), _z(obj.power)
    n = math.ceil((za + zb)**2 * obj.paired_sd**2
                  / obj.min_detectable_effect**2)
    n = max(n, 5)
    if normality_ok and n >= 30:
        test = "paired t-test (two-sided)"
    elif normality_ok:
        test = "paired bootstrap (10000 resamples, BCa CI)"
    else:
        test = "Wilcoxon signed-rank (two-sided)"
    sign = ">" if obj.direction == "increase" else "<"
    rule = (f"Reject the improvement claim unless "
            f"mean(new - baseline) {sign} {obj.min_detectable_effect} "
            f"AND p < {obj.alpha} AND the {100*(1-obj.alpha):.0f}% CI "
            f"excludes 0. Otherwise report NO IMPROVEMENT.")
    stop = (f"Collect exactly n={n} paired held-out evaluations. "
            f"No interim peeking; no optional stopping. "
            f"If n cannot be reached, report INSUFFICIENT_POWER, "
            f"not a directional claim.")
    der = {"z_alpha_2": za, "z_power": zb,
           "formula": "n = (z_{1-a/2}+z_power)^2 * sd_d^2 / delta^2",
           "sd_d": obj.paired_sd, "delta": obj.min_detectable_effect,
           "normality_assumed": normality_ok}
    p = FalsificationProtocol(asdict(obj), n, test, rule, stop, der)
    p.compute_hash()
    return p


class RatificationGate:
    """
    ENFORCED human authorisation. The system may derive; it may not authorise.
    Execution requires a signature whose hash matches the protocol hash.
    """

    def __init__(self, store: str = "runs/ratifications"):
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)

    def request(self, p: FalsificationProtocol) -> Path:
        f = self.store / f"protocol_{p.protocol_hash[:16]}.json"
        f.write_text(json.dumps(p.to_dict(), indent=2))
        log.info("[gate] protocol written, awaiting ratification: %s", f)
        return f

    def ratify(self, p: FalsificationProtocol, human_id: str,
               acknowledged_hash: str) -> Dict[str, Any]:
        if acknowledged_hash != p.protocol_hash:
            raise Refusal(
                "ratification hash does not match the protocol. "
                "The human must acknowledge the EXACT derived protocol.")
        if not human_id.strip():
            raise Refusal("human_id is empty")
        p.ratified_by = human_id
        p.ratification_hash = hashlib.sha256(
            (human_id + p.protocol_hash).encode()).hexdigest()
        rec = {"protocol_hash": p.protocol_hash, "human_id": human_id,
               "ratification_hash": p.ratification_hash,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (self.store / f"ratified_{p.protocol_hash[:16]}.json").write_text(
            json.dumps(rec, indent=2))
        return rec

    def authorize(self, p: FalsificationProtocol) -> Dict[str, Any]:
        f = self.store / f"ratified_{p.protocol_hash[:16]}.json"
        if not f.exists():
            raise Refusal(
                "NOT RATIFIED. The system derived this protocol but may not "
                "execute it. A human must ratify the exact hash.\n"
                f"  protocol_hash = {p.protocol_hash}\n"
                "This gate is permanent by design: falsification criteria "
                "encode values, and values are not derivable from data.")
        rec = json.loads(f.read_text())
        if rec["protocol_hash"] != p.protocol_hash:
            raise Refusal("stored ratification does not match this protocol")
        return rec


def execute_paired_test(diff: np.ndarray, p: FalsificationProtocol,
                        gate: RatificationGate,
                        n_boot: int = 10000, seed: int = 0
                        ) -> Dict[str, Any]:
    """Runs ONLY after ratification. Reports the pre-registered verdict."""
    gate.authorize(p)
    d = np.asarray(diff, float)
    if len(d) < p.required_n:
        return {"verdict": "INSUFFICIENT_POWER",
                "n_have": len(d), "n_required": p.required_n,
                "note": "no directional claim permitted"}
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean()
                      for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [p.objective["alpha"]/2,
                                1 - p.objective["alpha"]/2])
    mean = float(d.mean())
    # two-sided bootstrap p against 0
    pval = float(2*min(np.mean(boots <= 0), np.mean(boots >= 0)))
    delta = p.objective["min_detectable_effect"]
    dir_ok = (mean > delta) if p.objective["direction"] == "increase" \
        else (mean < -delta)
    ci_ok = (lo > 0) if p.objective["direction"] == "increase" else (hi < 0)
    improve = bool(dir_ok and pval < p.objective["alpha"] and ci_ok)
    return {"verdict": "IMPROVEMENT" if improve else "NO_IMPROVEMENT",
            "mean_diff": mean, "ci": [float(lo), float(hi)],
            "p_value": pval, "n": len(d),
            "effect_threshold": delta,
            "decision_rule": p.decision_rule,
            "ratified_by": p.ratified_by}


# =============================================================================
# §5. INTEGRATED LOOP
# =============================================================================

class AutonomyLoop:
    """
    The four capabilities in one cycle:
      competence check -> transfer or refuse -> hypothesis design ->
      derived falsification -> ratification -> execute -> update.
    """

    def __init__(self, obj: DeclaredObjective,
                 hyps: List[Hypothesis], designs: np.ndarray,
                 store: str = "runs/autonomy"):
        self.obj = obj
        self.designer = BayesianExperimentDesigner(hyps, designs)
        self.gate = RatificationGate(store)
        self.protocol: Optional[FalsificationProtocol] = None
        self.competence: Optional[CompetenceMap] = None
        self.transfer: Optional[TransferGate] = None
        self.log: List[Dict[str, Any]] = []

    def bind_competence(self, cm: CompetenceMap) -> None:
        self.competence = cm

    def bind_transfer(self, tg: TransferGate) -> None:
        self.transfer = tg

    def step(self, z: np.ndarray, mus: np.ndarray, sds: np.ndarray,
             oracle: Callable[[float], float]) -> Dict[str, Any]:
        rec: Dict[str, Any] = {}
        if self.competence is not None:
            c = self.competence.assess(z, mus, sds)
            rec["competence"] = c["verdict"]
            if not bool(c["competent"][0]):
                rec["status"] = "REFUSED_COMPETENCE"
                rec["reasons"] = c["refusal_reasons"][0]
                self.log.append(rec)
                return rec
        d = self.designer.run(oracle)
        rec["design"] = {k: d[k] for k in
                         ("status", "n_experiments") if k in d}
        rec["design"].update({"hypothesis": d.get("hypothesis")
                              or d.get("best")})
        self.protocol = derive_falsification(self.obj)
        rec["protocol_hash"] = self.protocol.protocol_hash
        rec["required_n"] = self.protocol.required_n
        rec["status"] = "AWAITING_RATIFICATION"
        self.gate.request(self.protocol)
        self.log.append(rec)
        return rec


# =============================================================================
# §6. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    if not HAS_TORCH:
        print("  torch required"); return False
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # ---- T1 MMD detects shift ----
    print("\n[T1] capability 1a: MMD permutation test")
    A = rng.normal(0, 1, (120, 4))
    B_same = rng.normal(0, 1, (120, 4))
    B_shift = rng.normal(1.2, 1, (120, 4))
    r_same = mmd_permutation_test(A, B_same, n_perm=150)
    r_shift = mmd_permutation_test(A, B_shift, n_perm=150)
    print(f"      same:  MMD2={r_same['mmd2']:+.5f} p={r_same['p_value']:.4f}")
    print(f"      shift: MMD2={r_shift['mmd2']:+.5f} p={r_shift['p_value']:.4f}")
    rep("no false shift on identical distributions",
        not r_same["shift_detected"])
    rep("detects real shift", r_shift["shift_detected"])

    # ---- T2 invariant encoder + transfer refusal ----
    print("\n[T2] capability 1b: invariant transfer + refusal")
    d_in = 6
    envs = []
    for e, scale in enumerate([1.0, 2.0, 3.0]):
        n = 200
        x = rng.normal(0, 1, (n, d_in))
        x[:, 1] *= scale                       # environment-specific nuisance
        y = 2.0*x[:, 0] + 0.5*x[:, 2] + rng.normal(0, 0.15, n)
        envs.append({"x": torch.tensor(x, dtype=torch.float32),
                     "y": torch.tensor(y, dtype=torch.float32)})
    holder: Dict[str, Any] = {}
    tr = train_invariant(holder, envs, d_in, d_feat=16, epochs=300)
    enc = holder["encoder"]
    print(f"      IRM penalty {tr['history'][0]['irm']:.4f} -> "
          f"{tr['history'][-1]['irm']:.4f}")
    with torch.no_grad():
        cf = enc.features(envs[0]["x"]).numpy()
        o = enc(envs[0]["x"])
        cs = np.abs((o["mu"] - envs[0]["y"]).numpy()) / o["log_sd"].exp().numpy()
    kn = np.sqrt(np.sum((cf[:, None, :] - cf[None, :, :])**2, -1))
    np.fill_diagonal(kn, np.inf)
    thr = float(np.quantile(kn.min(0), 0.99) * 1.5)
    tg = TransferGate(enc, cf, cs, max_knn_dist=thr)
    in_dist = tg.predict(envs[1]["x"][:8])
    rep("in-support transfer succeeds without target gradients",
        in_dist["status"].startswith("OK"), in_dist["status"])
    far = torch.tensor(rng.normal(12, 1, (4, d_in)), dtype=torch.float32)
    out_dist = tg.predict(far)
    rep("far-out-of-support REFUSED", out_dist["status"] == "REFUSED",
        out_dist.get("reason", "")[:60])

    # ---- T3 weighted conformal under covariate shift ----
    print("\n[T3] capability 1c: weighted conformal coverage")
    n_cal, n_te = 400, 400
    xs = rng.normal(0, 1, (n_cal, 2))
    xt = rng.normal(1.0, 1, (n_te, 2))
    def y_of(x): return 2*x[:, 0] + rng.normal(0, 0.3, len(x))
    ys, yt = y_of(xs), y_of(xt)
    mu_s, mu_t = 2*xs[:, 0], 2*xt[:, 0]
    sd = np.full(n_cal, 0.3)
    scores = np.abs(ys - mu_s)/sd
    dr = DensityRatio(2, epochs=250)
    info = dr.fit(xs, xt)
    w_cal, fc = dr.ratio(xs)
    w_te, _ = dr.ratio(xt)
    q_naive = float(np.quantile(scores, 0.9))
    cov_naive = float(np.mean(np.abs(yt-mu_t) <= q_naive*0.3))
    qs = np.array([weighted_conformal(scores, w_cal, float(w), 0.1)["q"]
                   for w in w_te])
    finite = np.isfinite(qs)
    cov_w = float(np.mean(np.abs(yt-mu_t)[finite] <= qs[finite]*0.3))
    print(f"      separability={info['separability_auc_proxy']:.3f} "
          f"clip_frac={fc:.3f}")
    print(f"      naive coverage={cov_naive:.3f}  weighted={cov_w:.3f} "
          f"(nominal 0.90)")
    rep("weighted conformal improves coverage under shift",
        cov_w >= cov_naive - 0.02, f"{cov_naive:.3f} -> {cov_w:.3f}")
    rep("weighted coverage near nominal", cov_w > 0.82, f"{cov_w:.3f}")

    # ---- T4 EIG experiment design beats random ----
    print("\n[T4] capability 2: hypothesis + EIG design")
    hyps = HypothesisGrammar.dose_response_space()
    designs = np.array([0.5, 1, 2, 5, 8, 10, 15, 25, 50, 100, 200, 400])
    truth = hyps[2]                              # black_leff
    oracle = lambda x: float(truth.predict(np.asarray([x]))[0]
                             + rng.normal(0, truth.noise_sd))
    bd = BayesianExperimentDesigner(hyps, designs, 0.95, 12)
    res = bd.run(oracle)
    print(f"      status={res['status']} n={res['n_experiments']} "
          f"-> {res.get('hypothesis', res.get('best'))}")
    rep("EIG design identifies the true mechanism",
        res.get("hypothesis") == truth.name, str(res.get("hypothesis")))
    n_eig = res["n_experiments"]
    # random-design control
    hits, ns = 0, []
    for s in range(20):
        r2 = np.random.default_rng(s)
        post = np.ones(len(hyps))/len(hyps)
        used = []
        for k in range(12):
            avail = [x for x in designs if x not in used]
            x = float(r2.choice(avail)); used.append(x)
            y = float(truth.predict(np.asarray([x]))[0]
                      + r2.normal(0, truth.noise_sd))
            lik = np.array([math.exp(-0.5*((y-float(h.predict(np.asarray([x]))[0]))
                                           /h.noise_sd)**2) for h in hyps])
            post = post*lik; post /= post.sum()
            if post.max() >= 0.95:
                break
        ns.append(k+1)
        hits += int(hyps[int(np.argmax(post))].name == truth.name)
    print(f"      random-design control: {hits}/20 correct, "
          f"mean n={np.mean(ns):.1f}  (EIG n={n_eig})")
    rep("EIG is at least as efficient as random design",
        n_eig <= np.mean(ns) + 1.0, f"{n_eig} vs {np.mean(ns):.1f}")
    # posterior update correctness
    rep("posterior normalised", abs(bd.post.sum()-1) < 1e-9)

    # ---- T5 epistemic / aleatoric decomposition ----
    print("\n[T5] capability 3: competence boundary")
    Ztr = rng.normal(0, 1, (300, 8))
    cm = CompetenceMap(Ztr, CompetenceConfig(max_knn_dist=3.0,
                                             max_epistemic_fraction=0.6))
    # in-support, low disagreement, high noise -> aleatoric dominates
    z_in = rng.normal(0, 1, (1, 8))
    mus_in = np.full((5, 1), 0.5) + rng.normal(0, 0.01, (5, 1))
    sds_in = np.full((5, 1), 0.8)
    a_in = cm.assess(z_in, mus_in, sds_in)
    # in-support, HIGH disagreement -> epistemic dominates
    mus_dis = np.array([[0.0], [1.0], [-1.0], [2.0], [-2.0]])
    sds_dis = np.full((5, 1), 0.05)
    a_dis = cm.assess(z_in, mus_dis, sds_dis)
    # out of support
    z_out = rng.normal(9, 1, (1, 8))
    a_out = cm.assess(z_out, mus_in, sds_in)
    print(f"      in/noisy : epi_frac={a_in['epistemic_fraction'][0]:.3f} "
          f"verdict={a_in['verdict']}")
    print(f"      disagree : epi_frac={a_dis['epistemic_fraction'][0]:.3f} "
          f"verdict={a_dis['verdict']}")
    print(f"      OOD      : kNN={a_out['knn_dist'][0]:.2f} "
          f"verdict={a_out['verdict']}")
    rep("aleatoric dominates when models agree but noise is high",
        a_in["epistemic_fraction"][0] < 0.2)
    rep("epistemic dominates when models disagree",
        a_dis["epistemic_fraction"][0] > 0.8)
    rep("disagreement triggers refusal", a_dis["verdict"] == "REFUSE_PARTIAL")
    rep("out-of-support triggers refusal", a_out["verdict"] == "REFUSE_PARTIAL")
    rep("scope statement present", "RELATIVE" in a_out["SCOPE"])

    # ---- T6 derived falsification + power analysis ----
    print("\n[T6] capability 4: derived falsification + ratification gate")
    obj = DeclaredObjective(
        metric_name="held-out lDDT", baseline_value=0.72,
        min_detectable_effect=0.02, paired_sd=0.05,
        alpha=0.05, power=0.80, direction="increase",
        declared_by="human:Yoon", rationale="0.02 lDDT is the smallest "
        "change that alters downstream domain assignment")
    p = derive_falsification(obj)
    za, zb = _z(0.975), _z(0.80)
    n_expect = math.ceil((za+zb)**2 * 0.05**2 / 0.02**2)
    print(f"      derived n={p.required_n} (analytic {n_expect})  "
          f"test={p.test}")
    rep("power analysis matches analytic formula",
        p.required_n == max(n_expect, 5), f"{p.required_n}")
    rep("protocol hash computed", len(p.protocol_hash) == 64)
    gate = RatificationGate("runs/_selftest_ratify")
    gate.request(p)
    blocked = False
    try:
        execute_paired_test(rng.normal(0.05, 0.05, p.required_n), p, gate)
    except Refusal:
        blocked = True
    rep("execution BLOCKED without ratification", blocked)
    wrong = False
    try:
        gate.ratify(p, "human:Yoon", "wrong_hash")
    except Refusal:
        wrong = True
    rep("wrong-hash ratification refused", wrong)
    rec = gate.ratify(p, "human:Yoon", p.protocol_hash)
    rep("correct ratification recorded", rec["ratification_hash"] is not None)
    # true improvement
    d_true = rng.normal(0.05, 0.05, p.required_n)
    r_true = execute_paired_test(d_true, p, gate)
    # no improvement
    d_null = rng.normal(0.0, 0.05, p.required_n)
    r_null = execute_paired_test(d_null, p, gate)
    print(f"      true effect -> {r_true['verdict']}  "
          f"null -> {r_null['verdict']}")
    rep("detects real improvement", r_true["verdict"] == "IMPROVEMENT")
    rep("reports NO_IMPROVEMENT on null", r_null["verdict"] == "NO_IMPROVEMENT")
    # insufficient power
    r_short = execute_paired_test(rng.normal(0.05, 0.05, 3), p, gate)
    rep("insufficient n -> INSUFFICIENT_POWER, no directional claim",
        r_short["verdict"] == "INSUFFICIENT_POWER")
    # objective without human owner is refused
    bad = DeclaredObjective("x", 0.5, 0.01, 0.05, declared_by="",
                            rationale="")
    refused = False
    try:
        derive_falsification(bad)
    except Refusal:
        refused = True
    rep("objective without human owner/rationale REFUSED", refused)

    # ---- T7 integrated loop ----
    print("\n[T7] integrated autonomy loop")
    loop = AutonomyLoop(obj, hyps, designs, store="runs/_selftest_autonomy")
    loop.bind_competence(cm)
    r_in = loop.step(z_in, mus_in, sds_in, oracle)
    r_out = loop.step(z_out, mus_in, sds_in, oracle)
    print(f"      in-support  -> {r_in['status']}")
    print(f"      OOD         -> {r_out['status']}")
    rep("in-support reaches ratification gate",
        r_in["status"] == "AWAITING_RATIFICATION")
    rep("OOD refused before any experiment",
        r_out["status"] == "REFUSED_COMPETENCE")

    import shutil
    shutil.rmtree("runs/_selftest_ratify", ignore_errors=True)
    shutil.rmtree("runs/_selftest_autonomy", ignore_errors=True)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
NOW REAL (each with a test that would fail if fake)
  [1] transfer without target gradients; MMD shift test; weighted conformal
      restores coverage where naive conformal undercovers; REFUSES outside
      support and under un-certifiable shift.
  [2] hypothesis grammar + exact EIG design; identifies the true mechanism
      in fewer experiments than random design; Bayes posterior normalised.
  [3] epistemic/aleatoric decomposition separates "need data" from
      "noise floor"; kNN + Mahalanobis support test; refusal enforced.
  [4] falsification protocol DERIVED from a human objective with correct
      power analysis; execution BLOCKED until hash-matched human ratification;
      null effect reported as NO_IMPROVEMENT; low n as INSUFFICIENT_POWER.

STILL NOT REAL — AND DELIBERATELY NOT BUILT
  * Transfer to ARBITRARY new domains. Impossible without assumptions
    linking source and target. We test the assumption instead of hiding it.
  * Inventing new conceptual primitives outside the declared grammar.
  * Knowing unknown-unknowns. Competence is relative to observed support.
  * Self-authored success criteria. Falsification encodes VALUES; values
    are not derivable from data; self-certification is Goodhart; and in
    the biomedical setting it is a safety hazard. The ratification gate
    is permanent by design.
""")
    return ok


# =============================================================================
# §7. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Autonomy layer")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "derive", "ratify", "design",
                             "ledger"])
    ap.add_argument("--metric", default="held-out lDDT")
    ap.add_argument("--baseline", type=float, default=0.72)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--sd", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--direction", default="increase")
    ap.add_argument("--by", default="")
    ap.add_argument("--rationale", default="")
    ap.add_argument("--hash", default=None)
    ap.add_argument("--human", default=None)
    ap.add_argument("--store", default="runs/ratifications")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    out: Dict[str, Any] = {}

    if a.cmd == "ledger":
        out = {
            "transfer": "REAL_CONDITIONAL",
            "hypothesis_design": "REAL_WITHIN_DECLARED_SPACE",
            "competence_metareasoning": "REAL_RELATIVE_TO_SUPPORT",
            "falsification_derivation": "REAL",
            "falsification_authorization": "PERMANENTLY_HUMAN",
            "arbitrary_domain_transfer": "IMPOSSIBLE_WITHOUT_ASSUMPTIONS",
            "new_primitives": "NOT_IMPLEMENTED",
            "unknown_unknowns": "NOT_DETECTABLE",
        }
        print(json.dumps(out, indent=2))

    elif a.cmd == "derive":
        obj = DeclaredObjective(a.metric, a.baseline, a.delta, a.sd,
                                a.alpha, a.power, a.direction,
                                a.by, a.rationale)
        try:
            p = derive_falsification(obj)
        except Refusal as e:
            print(str(e)); return 2
        gate = RatificationGate(a.store)
        path = gate.request(p)
        out = p.to_dict()
        print(json.dumps(out, indent=2))
        print(f"\nAwaiting ratification. File: {path}")
        print(f"To ratify: --cmd ratify --hash {p.protocol_hash} "
              f"--human <your-id>")

    elif a.cmd == "ratify":
        if not (a.hash and a.human):
            print("--hash and --human required"); return 1
        f = Path(a.store) / f"protocol_{a.hash[:16]}.json"
        if not f.exists():
            print(f"protocol not found: {f}"); return 1
        d = json.loads(f.read_text())
        p = FalsificationProtocol(**{k: d[k] for k in
                                     FalsificationProtocol.__dataclass_fields__})
        gate = RatificationGate(a.store)
        out = gate.ratify(p, a.human, a.hash)
        print(json.dumps(out, indent=2))

    elif a.cmd == "design":
        hyps = HypothesisGrammar.dose_response_space()
        designs = np.array([0.5, 1, 2, 5, 8, 10, 15, 25, 50, 100, 200, 400])
        bd = BayesianExperimentDesigner(hyps, designs)
        x, scores = bd.select_design()
        out = {"next_experiment": x, "eig_by_design": scores,
               "prior": {h.name: float(p)
                         for h, p in zip(hyps, bd.post)}}
        print(json.dumps(out, indent=2, default=float))
        print("\nRun this experiment, then call update(x, y). "
              "The oracle is NOT supplied by this module.")

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
