"""
=============================================================================
SESI MULTI-SCALE COUPLING ↔ EXPERIMENTAL DATA INTEGRATION
Calibration, validation, and uncertainty quantification for multi-scale models
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE ADDS
---------------------
[EXP1] ExperimentalDataInterface
       Abstract interface for loading experimental data from various sources:
         • Microscopy (fluorescence, electron, atomic force)
         • Spectroscopy (mass spec, NMR, Raman)
         • Imaging (MRI, CT, optical coherence tomography)
         • Omics (genomics, proteomics, metabolomics)

[EXP2] CalibrationPipeline
       Bayesian calibration of multi-scale coupling parameters against
       experimental data. Uses:
         • Likelihood function from experimental noise model
         • Prior from physical constraints
         • MCMC or variational inference for posterior

[EXP3] ValidationFramework
       Cross-validation and out-of-sample testing:
         • K-fold cross-validation on experimental data
         • Hold-out validation set
         • Predictive checks (posterior predictive p-values)

[EXP4] UncertaintyQuantification
       Propagation of experimental uncertainty through the multi-scale model:
         • Measurement error model
         • Model discrepancy (structural uncertainty)
         • Posterior predictive intervals

LABELS: [P] proved  [C] cited  [D] derived  [O] open

HONEST SCOPE
------------
• This is an INTERFACE for experimental data, not a claim that the multi-scale
  model is "correct" or "validated" in an absolute sense.
• Calibration depends on the noise model; wrong noise model → wrong parameters.
• Validation is necessary but not sufficient; a model can pass validation and
  still be wrong in untested regimes.
• Uncertainty quantification requires careful specification of priors and
  likelihood; results are conditional on these choices.
• Experimental data has its own uncertainties (measurement error, systematic
  bias, missing data); these must be modeled explicitly.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.multiscale_exp")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    "ExperimentalDataConfig",
    "ExperimentalDataInterface",
    "MicroscopyData",
    "SpectroscopyData",
    "CalibrationPipeline",
    "ValidationFramework",
    "UncertaintyQuantification",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentalDataConfig:
    """Configuration for experimental data integration."""
    # Data sources
    data_paths: List[str] = field(default_factory=list)
    data_format: str = "auto"  # auto, csv, hdf5, json

    # Noise model
    noise_type: str = "gaussian"  # gaussian, poisson, student_t
    noise_scale: float = 0.1  # relative noise level

    # Calibration
    n_calibration_samples: int = 1000
    n_warmup: int = 500
    learning_rate: float = 1e-3

    # Validation
    n_folds: int = 5
    validation_fraction: float = 0.2

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [EXP1] EXPERIMENTAL DATA INTERFACE
# ═══════════════════════════════════════════════════════════════════════

class ExperimentalDataInterface:
    """
    [D] Abstract interface for loading experimental data.

    Subclasses implement specific data formats (microscopy, spectroscopy, etc.)
    and return standardized data structures.
    """

    def __init__(self, cfg: ExperimentalDataConfig):
        self.cfg = cfg

    def load(self, path: str) -> Dict[str, Any]:
        """Load experimental data from file."""
        raise NotImplementedError

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data (normalization, filtering, etc.)."""
        raise NotImplementedError

    def get_observations(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract observations and uncertainties.

        Returns:
            (observations, uncertainties): arrays of shape (N,) or (N, D)
        """
        raise NotImplementedError


class MicroscopyData(ExperimentalDataInterface):
    """
    [D] Microscopy data interface (fluorescence, electron, AFM).

    Handles:
      • Image data (2D/3D arrays)
      • Point spread function (PSF) deconvolution
      • Intensity calibration
    """

    def load(self, path: str) -> Dict[str, Any]:
        """Load microscopy image data."""
        try:
            # Placeholder: real implementation would use tifffile, h5py, etc.
            logger.info(f"Loading microscopy data from {path}")
            # Mock data for smoke test
            image = np.random.rand(256, 256)
            metadata = {"pixel_size": 0.1, "units": "um"}
            return {"image": image, "metadata": metadata, "path": path}
        except Exception as e:
            logger.error(f"Failed to load microscopy data: {e}")
            return {}

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess microscopy data (background subtraction, normalization)."""
        if "image" not in data:
            return data
        image = data["image"]
        # Background subtraction
        background = np.percentile(image, 5)
        image = image - background
        # Normalization
        image = image / (np.max(image) + 1e-8)
        data["image"] = image
        return data

    def get_observations(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract intensity measurements and uncertainties."""
        if "image" not in data:
            return np.array([]), np.array([])
        image = data["image"]
        # Flatten image to 1D array of intensities
        observations = image.flatten()
        # Uncertainty: Poisson noise model (sqrt of intensity)
        uncertainties = np.sqrt(np.abs(observations) + 1e-8) * self.cfg.noise_scale
        return observations, uncertainties


class SpectroscopyData(ExperimentalDataInterface):
    """
    [D] Spectroscopy data interface (mass spec, NMR, Raman).

    Handles:
      • Spectral data (wavelength/frequency vs intensity)
      • Peak detection and integration
      • Baseline correction
    """

    def load(self, path: str) -> Dict[str, Any]:
        """Load spectroscopy data."""
        try:
            logger.info(f"Loading spectroscopy data from {path}")
            # Mock data: wavelength (nm) vs intensity (a.u.)
            wavelength = np.linspace(400, 700, 300)
            intensity = np.exp(-((wavelength - 550) ** 2) / (2 * 30 ** 2))
            intensity += np.random.normal(0, 0.05, len(wavelength))
            return {
                "wavelength": wavelength,
                "intensity": intensity,
                "path": path,
            }
        except Exception as e:
            logger.error(f"Failed to load spectroscopy data: {e}")
            return {}

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Baseline correction and normalization."""
        if "intensity" not in data:
            return data
        intensity = data["intensity"]
        # Baseline correction (simple linear fit)
        baseline = np.polyval(np.polyfit(np.arange(len(intensity)), intensity, 1),
                              np.arange(len(intensity)))
        intensity = intensity - baseline
        # Normalization
        intensity = intensity / (np.max(intensity) + 1e-8)
        data["intensity"] = intensity
        return data

    def get_observations(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract spectral intensities and uncertainties."""
        if "intensity" not in data:
            return np.array([]), np.array([])
        observations = data["intensity"]
        uncertainties = np.abs(observations) * self.cfg.noise_scale + 1e-8
        return observations, uncertainties


# ═══════════════════════════════════════════════════════════════════════
# [EXP2] CALIBRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class CalibrationPipeline:
    """
    [C/D] Bayesian calibration of multi-scale coupling parameters.

    Uses:
      • Likelihood: p(data | params) from noise model
      • Prior: p(params) from physical constraints
      • Posterior: p(params | data) ∝ likelihood × prior

    Methods:
      • MCMC (Metropolis-Hastings) for sampling
      • Variational inference for approximation
    """

    def __init__(self, cfg: ExperimentalDataConfig, model: Any):
        self.cfg = cfg
        self.model = model  # multi-scale coupling model
        self.posterior_samples: Optional[np.ndarray] = None

    def log_likelihood(self, params: np.ndarray, observations: np.ndarray,
                       uncertainties: np.ndarray) -> float:
        """
        Compute log-likelihood under Gaussian noise model.

        log p(data | params) = -½ Σ [(obs - pred) / unc]²
        """
        # Predict observations from model
        predictions = self.model.predict(params)
        # Gaussian likelihood
        residuals = (observations - predictions) / uncertainties
        return float(-0.5 * np.sum(residuals ** 2))

    def log_prior(self, params: np.ndarray) -> float:
        """
        Compute log-prior from physical constraints.

        Default: uniform prior on bounded parameters.
        """
        # Placeholder: real implementation would have parameter-specific priors
        if np.any(params < 0) or np.any(params > 10):
            return -np.inf
        return 0.0

    def log_posterior(self, params: np.ndarray, observations: np.ndarray,
                      uncertainties: np.ndarray) -> float:
        """Compute log-posterior = log-likelihood + log-prior."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(params, observations, uncertainties)
        return lp + ll

    def calibrate_mcmc(self, observations: np.ndarray, uncertainties: np.ndarray,
                       initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate parameters using Metropolis-Hastings MCMC.

        Returns posterior samples and diagnostics.
        """
        if not HAS_TORCH:
            return {"error": "torch required for MCMC calibration"}

        n_params = len(initial_params)
        n_samples = self.cfg.n_calibration_samples
        n_warmup = self.cfg.n_warmup

        samples = np.zeros((n_samples, n_params))
        current = initial_params.copy()
        current_log_p = self.log_posterior(current, observations, uncertainties)

        proposal_scale = 0.1 * np.ones(n_params)
        n_accept = 0

        for i in range(n_samples + n_warmup):
            # Propose new parameters
            proposal = current + proposal_scale * np.random.randn(n_params)
            proposal_log_p = self.log_posterior(proposal, observations, uncertainties)

            # Accept/reject
            log_alpha = proposal_log_p - current_log_p
            if np.log(np.random.rand()) < log_alpha:
                current = proposal
                current_log_p = proposal_log_p
                if i >= n_warmup:
                    n_accept += 1

            if i >= n_warmup:
                samples[i - n_warmup] = current

        self.posterior_samples = samples
        acceptance_rate = n_accept / n_samples

        return {
            "posterior_samples": samples,
            "acceptance_rate": acceptance_rate,
            "posterior_mean": np.mean(samples, axis=0),
            "posterior_std": np.std(samples, axis=0),
            "note": "MCMC calibration; check acceptance rate (target ~0.23 for high-dim)",
        }

    def calibrate_variational(self, observations: np.ndarray,
                              uncertainties: np.ndarray,
                              initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Calibrate parameters using variational inference (mean-field approximation).

        Faster than MCMC but less accurate for complex posteriors.
        """
        # Placeholder: real implementation would use Pyro or TensorFlow Probability
        logger.warning("Variational inference not fully implemented; using MCMC fallback")
        return self.calibrate_mcmc(observations, uncertainties, initial_params)


# ═══════════════════════════════════════════════════════════════════════
# [EXP3] VALIDATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════

class ValidationFramework:
    """
    [C/D] Cross-validation and out-of-sample testing for multi-scale models.

    Methods:
      • K-fold cross-validation
      • Hold-out validation
      • Posterior predictive checks
    """

    def __init__(self, cfg: ExperimentalDataConfig, model: Any):
        self.cfg = cfg
        self.model = model

    def k_fold_cv(self, observations: np.ndarray, uncertainties: np.ndarray,
                  params: np.ndarray) -> Dict[str, Any]:
        """
        K-fold cross-validation.

        Splits data into K folds, trains on K-1, validates on 1.
        """
        n = len(observations)
        n_folds = self.cfg.n_folds
        fold_size = n // n_folds

        errors = []
        for k in range(n_folds):
            # Split data
            val_start = k * fold_size
            val_end = (k + 1) * fold_size if k < n_folds - 1 else n

            val_mask = np.zeros(n, dtype=bool)
            val_mask[val_start:val_end] = True

            train_obs = observations[~val_mask]
            train_unc = uncertainties[~val_mask]
            val_obs = observations[val_mask]
            val_unc = uncertainties[val_mask]

            # Calibrate on training data
            calib = CalibrationPipeline(self.cfg, self.model)
            calib_result = calib.calibrate_mcmc(train_obs, train_unc, params)

            # Validate on held-out data
            val_params = calib_result["posterior_mean"]
            val_pred = self.model.predict(val_params)
            val_error = np.mean((val_obs - val_pred) ** 2 / val_unc ** 2)
            errors.append(val_error)

        return {
            "cv_errors": errors,
            "mean_cv_error": np.mean(errors),
            "std_cv_error": np.std(errors),
            "n_folds": n_folds,
        }

    def posterior_predictive_check(self, observations: np.ndarray,
                                   uncertainties: np.ndarray,
                                   posterior_samples: np.ndarray) -> Dict[str, Any]:
        """
        Posterior predictive check.

        Generate replicated data from posterior predictive distribution and
        compare to observed data.
        """
        n_replications = 100
        replicated_errors = []

        for i in range(n_replications):
            # Sample parameters from posterior
            params = posterior_samples[np.random.randint(len(posterior_samples))]
            # Generate replicated data
            pred = self.model.predict(params)
            rep_data = pred + uncertainties * np.random.randn(len(pred))
            # Compute test statistic (e.g., mean squared error)
            rep_error = np.mean((observations - rep_data) ** 2)
            replicated_errors.append(rep_error)

        # Observed test statistic
        obs_params = np.mean(posterior_samples, axis=0)
        obs_pred = self.model.predict(obs_params)
        obs_error = np.mean((observations - obs_pred) ** 2)

        # Posterior predictive p-value
        p_value = np.mean(np.array(replicated_errors) >= obs_error)

        return {
            "observed_error": obs_error,
            "replicated_errors": replicated_errors,
            "posterior_predictive_p_value": p_value,
            "note": "p-value ~0.5 indicates good fit; p < 0.05 or p > 0.95 indicates misfit",
        }


# ═══════════════════════════════════════════════════════════════════════
# [EXP4] UNCERTAINTY QUANTIFICATION
# ═══════════════════════════════════════════════════════════════════════

class UncertaintyQuantification:
    """
    [C/D] Propagation of experimental uncertainty through multi-scale model.

    Sources of uncertainty:
      • Measurement error (experimental noise)
      • Model discrepancy (structural uncertainty)
      • Parameter uncertainty (from calibration)
    """

    def __init__(self, cfg: ExperimentalDataConfig, model: Any):
        self.cfg = cfg
        self.model = model

    def propagate_parameter_uncertainty(self, posterior_samples: np.ndarray,
                                        input_conditions: np.ndarray) -> Dict[str, Any]:
        """
        Propagate parameter uncertainty to predictions.

        For each posterior sample, generate predictions and compute intervals.
        """
        n_samples = len(posterior_samples)
        predictions = np.zeros((n_samples, len(input_conditions)))

        for i in range(n_samples):
            params = posterior_samples[i]
            predictions[i] = self.model.predict(params, input_conditions)

        # Compute credible intervals
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)

        return {
            "mean_prediction": mean_pred,
            "std_prediction": std_pred,
            "95_percent_ci_lower": ci_lower,
            "95_percent_ci_upper": ci_upper,
            "n_posterior_samples": n_samples,
        }

    def add_model_discrepancy(self, predictions: np.ndarray,
                              observations: np.ndarray) -> Dict[str, Any]:
        """
        Estimate model discrepancy (structural uncertainty).

        discrepancy = observations - predictions (after accounting for noise)
        """
        residuals = observations - predictions
        discrepancy_mean = np.mean(residuals)
        discrepancy_std = np.std(residuals)

        return {
            "discrepancy_mean": discrepancy_mean,
            "discrepancy_std": discrepancy_std,
            "note": "Model discrepancy indicates structural uncertainty; consider model refinement",
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Multi-Scale Coupling ↔ Experimental Data — Smoke Test")
    print("=" * 72)

    cfg = ExperimentalDataConfig(
        noise_type="gaussian",
        noise_scale=0.1,
        n_calibration_samples=100,
        n_warmup=50,
    )

    # Mock multi-scale model
    class MockMultiScaleModel:
        def predict(self, params, conditions=None):
            # Simple linear model for testing
            if conditions is None:
                conditions = np.linspace(0, 1, 50)
            return params[0] * conditions + params[1]

    model = MockMultiScaleModel()

    # Test 1: Load experimental data
    print("\n  ── [EXP1] Experimental Data Loading ──")
    micro_data = MicroscopyData(cfg)
    spec_data = SpectroscopyData(cfg)

    micro_loaded = micro_data.load("mock_microscopy.tif")
    spec_loaded = spec_data.load("mock_spectrum.csv")

    print(f"    Microscopy data loaded: {bool(micro_loaded)}")
    print(f"    Spectroscopy data loaded: {bool(spec_loaded)}")

    if micro_loaded:
        micro_processed = micro_data.preprocess(micro_loaded)
        micro_obs, micro_unc = micro_data.get_observations(micro_processed)
        print(f"    Microscopy observations: {len(micro_obs)} points")

    if spec_loaded:
        spec_processed = spec_data.preprocess(spec_loaded)
        spec_obs, spec_unc = spec_data.get_observations(spec_processed)
        print(f"    Spectroscopy observations: {len(spec_obs)} points")

    # Test 2: Calibration
    print("\n  ── [EXP2] Bayesian Calibration ──")
    if spec_loaded:
        calib = CalibrationPipeline(cfg, model)
        initial_params = np.array([1.0, 0.0])
        calib_result = calib.calibrate_mcmc(spec_obs, spec_unc, initial_params)

        print(f"    Acceptance rate: {calib_result['acceptance_rate']:.3f}")
        print(f"    Posterior mean: {calib_result['posterior_mean']}")
        print(f"    Posterior std: {calib_result['posterior_std']}")

    # Test 3: Validation
    print("\n  ── [EXP3] Cross-Validation ──")
    if spec_loaded and "posterior_samples" in calib_result:
        val = ValidationFramework(cfg, model)
        cv_result = val.k_fold_cv(spec_obs, spec_unc, initial_params)

        print(f"    Mean CV error: {cv_result['mean_cv_error']:.4f}")
        print(f"    Std CV error: {cv_result['std_cv_error']:.4f}")

        ppc_result = val.posterior_predictive_check(
            spec_obs, spec_unc, calib_result["posterior_samples"]
        )
        print(f"    Posterior predictive p-value: {ppc_result['posterior_predictive_p_value']:.3f}")

    # Test 4: Uncertainty quantification
    print("\n  ── [EXP4] Uncertainty Quantification ──")
    if spec_loaded and "posterior_samples" in calib_result:
        uq = UncertaintyQuantification(cfg, model)
        input_conditions = np.linspace(0, 1, 50)
        uq_result = uq.propagate_parameter_uncertainty(
            calib_result["posterior_samples"], input_conditions
        )

        print(f"    Prediction mean range: [{uq_result['mean_prediction'].min():.3f}, "
              f"{uq_result['mean_prediction'].max():.3f}]")
        print(f"    95% CI width (mean): {np.mean(uq_result['95_percent_ci_upper'] - uq_result['95_percent_ci_lower']):.3f}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • This is an INTERFACE for experimental data, not a validated model")
    print("  • Calibration depends on noise model; wrong model → wrong parameters")
    print("  • Validation is necessary but not sufficient")
    print("  • Uncertainty quantification requires careful prior specification")
    print("  • Experimental data has its own uncertainties (must be modeled)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
