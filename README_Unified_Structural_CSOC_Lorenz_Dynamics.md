## Unified Structural CSOC-Lorenz Dynamics
A production-grade, fully differentiable PyTorch implementation of the Unified Structural CSOC-Lorenz Dynamics framework. This module bridges Structural Geometric Measure Theory (Structural GMT), Controlled Self-Organized Criticality (CSOC), and spatiotemporal chaotic systems.
Key Features
 * Spectral 8th-Order Polyharmonic Regularization: Computes the 8th-order polyharmonic operator (\Delta^4_{\mathbb{R}}) efficiently in the Fourier domain, reducing spatial computations to O(N \log N) complexity.
 * Semantic-State Contraction (SSC) Control: Fully vectorized implementation of the deterministic control law \mathcal{H}_{SSC}(X) for precise fixed-point stabilization.
 * Learnable Universality Tuning: Features dynamic parameters for the spatial redistribution kernel K_\alpha(r) to study universality class transitions and scale invariance.
 * Native Autograd & Hardware Acceleration: Designed for end-to-end gradient-based optimization and fully compatible with torch.compile(mode="max-autotune") for kernel fusion.
Mathematical Architecture
The module models a coupled structural Lorenz system across three primary evolution equations:
 * Avalanche State Evolution (X): Captures continuous chaotic dynamics coupled with discrete structural phase transitions.
 * Fractal Interface Strain (Y): Regularized via the 8th-order structural polyharmonic operator (\Delta^4_{\mathbb{R}}) to guarantee uniform higher-order up-to-interface regularity.
 * Learnable CSOC Parameter Control (Z): Integrates spatial redistribution kernels and Semantic-State Contraction (SSC) to maintain the system at its critical fixed point (e_{FP} = 1.64 \times 10^{-4}).
Installation & Quick Start
Prerequisites
 * Python 3.10+
 * PyTorch 2.0+
Usage Example
import torch
from model import UnifiedCSOCLorenzDynamics

# Initialize the production model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedCSOCLorenzDynamics(grid_size=64).to(device)

# Create a batch of spatial state tensors: shape (Batch, Channels [X, Y, Z], Height, Width)
state = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)

# Perform a single continuous-time step update
next_state = model(state, dt=0.01)

# Backpropagation test (fully differentiable graph)
loss = next_state.sum()
loss.backward()
print("Gradients successfully propagated through CSOC-Lorenz dynamics.")

Learnable Parameters
 * alpha: Universality tuning parameter initialized near \alpha \approx 2.8.
 * a1, a2, a3, a4: SSC control coefficients initialized to target fixed-point stability values (a_1 = a_4 = 3 \times 10^{-4}).
 * length_scale: Spatial interaction length scale l.
 * sigma, rho, beta_lorenz: Structural Lorenz system coupling constants.
Citation & License
If you utilize this framework in your computational pipeline or research, please cite the underlying structural dynamics manuscript. Licensed under the MIT License.
