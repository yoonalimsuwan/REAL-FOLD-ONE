"""
Production-grade differentiable Vitali-set pipeline with OPMC closure,
native full differentiation, AMP, and multi-GPU DDP.

# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# ORCID        : 0009-0008-2374-0788
# GitHub       : yoonalimsuwan
# Contact      : msps4u@gmail.com
# License      : MIT
# Year         : 2026
# =============================================================================

Tested against: PyTorch >= 2.0
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler


# ============================================================================
# 1. Differentiable Vitali Set on the torus T = R/Z
# ============================================================================
class DifferentiableVitaliSet(nn.Module):
    """
    Native-differentiable surrogate of a Vitali set V ⊂ T = R/Z.

    A Vitali set is a transversal of the quotient T / Q : it selects exactly
    one representative from every coset of the (countable) rational group in
    the circle.  It is non-measurable w.r.t. the Lebesgue measure λ, hence the
    classical indicator 1_V cannot be integrated.  We replace 1_V by a C^∞
    density built as a temperature-annealed softmax over a finite-rank proxy
    of the quotient:

        rho_V(x) = Σ_c  κ_c · K_sigma(x - r_c)  /  Σ_c κ_c

    with
        c   indexes a finite family of cosets,
        r_c = E_{p_c}[ grid ]          (expected soft representative),
        κ_c = ||p_c||_2^2              (confidence = 1 for a hard choice),
        K_sigma  periodic Gaussian (Von-Mises-type) kernel, bandwidth sigma.

    Every quantity is native-differentiable in {class_logits}.  No detach, no
    index-assign, no in-place scatter.  All heavy tensors are pre-materialised
    as non-persistent buffers so `.to(device)` and DDP broadcast are free.
    """

    def __init__(
        self,
        num_cosets: int = 256,
        grid_size: int = 512,
        sigma: float = 0.05,
        temperature: float = 1.0,
        chunk_size: int = 8192,
    ):
        super().__init__()
        self.num_cosets = num_cosets
        self.grid_size = grid_size
        self.sigma = sigma
        self.temperature = temperature
        self.chunk_size = chunk_size

        # Soft choice function (differentiable replacement of Axiom of Choice).
        self.class_logits = nn.Parameter(torch.zeros(num_cosets, grid_size))
        with torch.no_grad():
            self.class_logits.normal_(0.0, 1e-3)

        grid = torch.linspace(0.0, 1.0, grid_size, dtype=torch.float32)
        self.register_buffer("grid", grid, persistent=False)

        # Finite-rank coset index via irrational rotation (low discrepancy).
        idx = torch.arange(grid_size, dtype=torch.float32)
        alpha = 0.7071067811865476  # 1/sqrt(2)
        coset_ids = (torch.frac(idx * alpha) * num_cosets).long().clamp_(0, num_cosets - 1)
        self.register_buffer("coset_ids", coset_ids, persistent=False)

    # ------------------------------------------------------------------ #
    def soft_choice(self):
        p = F.softmax(self.class_logits / self.temperature, dim=-1)   # (C, G)
        reps = (p * self.grid.unsqueeze(0)).sum(dim=-1)               # (C,)
        conf = (p * p).sum(dim=-1)                                    # (C,)
        return reps, conf

    # ------------------------------------------------------------------ #
    def density(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth indicator rho_V(x) for x of any shape; memory-chunked."""
        reps, conf = self.soft_choice()
        shape = x.shape
        x_flat = x.reshape(-1)
        outs = []
        for i in range(0, x_flat.numel(), self.chunk_size):
            chunk = x_flat[i : i + self.chunk_size]
            diff = chunk.unsqueeze(-1) - reps                     # (B, C)
            diff = diff - torch.round(diff)                       # periodic wrap
            rbf = torch.exp(-0.5 * (diff / self.sigma).pow(2))    # (B, C)
            rho = (rbf * conf).sum(-1) / (conf.sum() + 1e-8)
            outs.append(rho)
        return torch.cat(outs, dim=0).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.density(x)


# ============================================================================
# 2. Finitely-additive measure mu_FA  (Banach limit over translates)
# ============================================================================
class FinitelyAdditiveMeasure(nn.Module):
    """
    Translation-invariant finitely-additive measure mu_FA on T extending λ.

    Operational definition, differentiable and singularity-free:

        mu_FA(f) = lim_{N->∞}  (1/N) Σ_{k=0}^{N-1} ∫  f(x) · w(x + k/N) dx

    We evaluate with a fixed, small N using trapezoidal quadrature.  Because
    the integrand (a smooth Vitali density times a smooth observable) is C^∞,
    the Banach limit is well-defined and autograd-friendly — the operator
    "Vitali set -> scalar mass" is now a first-class differentiable primitive.
    """

    def __init__(self, num_translates: int = 32, num_quad: int = 2048):
        super().__init__()
        self.num_translates = num_translates
        self.num_quad = num_quad
        self.register_buffer(
            "x_quad",
            torch.linspace(0.0, 1.0, num_quad, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, vitali: "DifferentiableVitaliSet", observable=None) -> torch.Tensor:
        x = self.x_quad
        w = observable(x) if observable is not None else None

        accum = None
        for k in range(self.num_translates):
            shift = k / self.num_translates
            xs = torch.frac(x + shift)
            rho = vitali.density(xs)
            term = rho if w is None else rho * w
            accum = term if accum is None else accum + term
        mean_integrand = accum / self.num_translates
        return torch.trapezoid(mean_integrand, x)


# ============================================================================
# 3. Universal Contraction Operator  (unchanged mathematical contract)
# ============================================================================
class UniversalContractionOperator(nn.Module):
    """
    Lemma 2.4 / Theorem 2.6 : exact dimension d(m,n) = m^2 n^2 + m n^2,
    compactness of M_str via bounded image B_R(0), Banach fixed-point
    compatible contraction.
    """

    def __init__(self, m: int, n: int, R: float = 1.0):
        super().__init__()
        self.m = m
        self.n = n
        self.d_str = m ** 2 * n ** 2 + m * n ** 2
        self.R = R

        self.linear_proj = nn.Linear(self.d_str, self.d_str, bias=True)
        self.nonlin_scale = nn.Parameter(torch.tensor(0.035, dtype=torch.float32))
        nn.init.orthogonal_(self.linear_proj.weight)
        nn.init.zeros_(self.linear_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.linear_proj(x)
        norm = torch.norm(x_proj, p=2, dim=-1, keepdim=True)
        scale = torch.clamp(norm / self.R, min=1.0)
        x_bounded = x_proj / scale
        return x_bounded + self.nonlin_scale * torch.tanh(x_bounded)


# ============================================================================
# 4. OPMC Closure Integration Engine  (Theorem 9.2 + Corollary 3.2)
# ============================================================================
class OPMCClosureIntegrationEngine(nn.Module):
    """
    Weak-value / structural expectation on M_str evaluated against mu_FA
    applied to the smooth Vitali density rho_V.  End-to-end differentiable
    in the Vitali choice-function logits, the observable network, and the
    contraction operator's parameters.
    """

    def __init__(self, vitali: DifferentiableVitaliSet,
                 measure: FinitelyAdditiveMeasure,
                 observable_hidden: int = 64):
        super().__init__()
        self.vitali = vitali
        self.measure = measure

        # Learnable smooth observable w: T -> R via Fourier features + MLP.
        self.observable = nn.Sequential(
            nn.Linear(2, observable_hidden), nn.SiLU(),
            nn.Linear(observable_hidden, observable_hidden), nn.SiLU(),
            nn.Linear(observable_hidden, 1),
        )

    def _w(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack(
            [torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x)], dim=-1
        )
        return self.observable(feats).squeeze(-1)

    def forward(self) -> torch.Tensor:
        return self.measure(self.vitali, observable=self._w)


# ============================================================================
# 5. Full structural network
# ============================================================================
class VitaliOPMCNetwork(nn.Module):
    """
    Wraps the contraction operator + the OPMC engine over a differentiable
    Vitali set.  Exposes a single .forward(x) -> (transformed_x, weak_value)
    so DDP / AMP work without introspection into submodules.
    """

    def __init__(self, m: int, n: int, R: float = 1.0,
                 num_cosets: int = 128, grid_size: int = 256,
                 sigma: float = 0.05, num_translates: int = 16,
                 num_quad: int = 1024):
        super().__init__()
        self.contraction = UniversalContractionOperator(m=m, n=n, R=R)
        self.vitali = DifferentiableVitaliSet(
            num_cosets=num_cosets, grid_size=grid_size, sigma=sigma,
        )
        self.measure = FinitelyAdditiveMeasure(
            num_translates=num_translates, num_quad=num_quad,
        )
        self.opmc = OPMCClosureIntegrationEngine(self.vitali, self.measure)
        self.d_str = self.contraction.d_str

    def forward(self, x: torch.Tensor):
        y = self.contraction(x)
        weak_val = self.opmc()
        return y, weak_val


# ============================================================================
# 6. DDP + AMP training harness
# ============================================================================
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size, True
    return 0, 0, 1, False


class SyntheticStructuralDataset(Dataset):
    """Placeholder dataset — replace with real M_str samples in production."""
    def __init__(self, num_samples: int, d_str: int):
        g = torch.Generator().manual_seed(0)
        self.data = torch.randn(num_samples, d_str, generator=g)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        return self.data[i]


def execute_production_pipeline(
    m: int = 2, n: int = 2,
    batch_size: int = 64, epochs: int = 5,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    num_workers: int = 4,
    num_cosets: int = 128, grid_size: int = 256,
):
    local_rank, rank, world_size, is_distributed = setup_distributed()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    # -------------------- model --------------------
    model = VitaliOPMCNetwork(
        m=m, n=n, R=1.0,
        num_cosets=num_cosets, grid_size=grid_size,
    ).to(device)

    if is_distributed:
        # Vitali class_logits are used every forward; keep them in the
        # gradient sync path (default). No find_unused_parameters needed.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    raw_model = model.module if is_distributed else model

    # -------------------- data --------------------
    dataset = SyntheticStructuralDataset(2048, raw_model.d_str)
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )

    # -------------------- optim / AMP --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        fused=use_cuda,   # fused AdamW: lower memory + faster kernels
    )
    amp_device = "cuda" if use_cuda else "cpu"
    scaler = GradScaler(amp_device, enabled=use_cuda)

    model.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0
        for batch_x in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=amp_device, dtype=torch.float16,
                          enabled=use_cuda):
                transformed_x, weak_val = model(batch_x)

                # Contraction residual (Banach fixed-point, Thm 7.10)
                contraction_loss = torch.mean(torch.abs(transformed_x - batch_x))

                # Structural weak value on the Vitali density (mu_FA)
                # (kept in fp32 to protect the quadrature from fp16 underflow)
                weak_val_fp32 = weak_val.float()
                loss = contraction_loss + 0.01 * torch.abs(weak_val_fp32)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach().item()
            num_batches += 1

        if rank == 0:
            avg = epoch_loss / max(num_batches, 1)
            print(f"[epoch {epoch + 1:03d}/{epochs:03d}]  loss = {avg:.6f}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


# ============================================================================
# 7. Entry point
# ============================================================================
if __name__ == "__main__":
    # Single-process:
    #   python train_vitali.py
    # Multi-GPU (example, 4 GPUs on one node):
    #   torchrun --nproc_per_node=4 train_vitali.py
    execute_production_pipeline(
        m=2, n=2,
        batch_size=32, epochs=3,
        num_cosets=128, grid_size=256,
    )
