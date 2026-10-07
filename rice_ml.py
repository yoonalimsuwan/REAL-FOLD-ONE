"""
rice_ml.py  - proper acentric Rice (crystallographic ML) amplitude likelihood.

The original `compute_xray_maximum_likelihood` is documented as "Rice / ML" but its
`nll_xray` is 0.5*sum(((|Fo|-|Fc|)/sigma)^2): a Gaussian amplitude least-squares
residual. The true acentric Rice density, for observed amplitude Fo given a model
amplitude Fc, is
    p(Fo | Fc) = (2 Fo / S) * exp(-(Fo^2 + D^2 Fc^2) / S) * I0(2 Fo D Fc / S)
with D = sigma_A-like model-quality factor in (0,1] and S = variance of the
unexplained structure factor (>0). Normalised over Fo in [0, inf).
(Also note: the "bulk solvent" there scales F_calc by (1 + k exp(-B s^2/4)); real
solvent models add k*exp(-B s^2/4)*F_mask from a separate mask structure factor.)
"""
import torch


def rice_acentric_nll(f_obs: torch.Tensor, f_calc: torch.Tensor,
                      d: torch.Tensor, s: torch.Tensor,
                      eps: float = 1e-12) -> torch.Tensor:
    """-log p(Fo | Fc, D, S), elementwise. Uses log I0(x) = log(i0e(x)) + x (stable)."""
    fo = f_obs.clamp_min(0)
    arg = 2.0 * fo * d * f_calc / s
    log_i0 = torch.log(torch.special.i0e(arg)) + arg
    return -(torch.log(2.0 * fo.clamp_min(eps) / s)
             - (fo ** 2 + (d * f_calc) ** 2) / s + log_i0)
