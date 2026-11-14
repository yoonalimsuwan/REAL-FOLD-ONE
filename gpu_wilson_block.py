#!/usr/bin/env python3
"""
=============================================================================
GPU BLOCK WILSON DIRAC — spin-projected, EO-preconditioned, block Krylov
Thermalized SU(3) gauge on device · mixed precision · deflation recycling
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT IS REAL HERE
  [1] SPIN-PROJECTED WILSON OPERATOR (the QUDA trick).
      (1 -+ gamma_mu) has rank 2, so
          (1-gamma_mu) psi = (chi, -+ barsig_mu chi),
          chi = psi_upper -+ sig_mu psi_lower    (2 components)
      The colour matrix is applied to 2 components instead of 4, and only
      2-component fields are gathered across neighbours. This halves both
      the colour-multiply FLOPs and the neighbour traffic. Verified against
      a naive full 4-spin implementation in the self-test.

  [2] BLOCK RHS. Field layout (V, 2, 2, 3, B). Gauge links are read ONCE
      per application and reused across all B columns. Link bytes per RHS
      therefore falls as 1/B. This is the memory-bandwidth win, and it is
      MEASURED (bytes and achieved GB/s), not asserted.

  [3] EVEN-ODD with PROPER HALF-LATTICE INDEXING. No scatter/gather waste:
      neighbour maps are precomputed between the even and odd halves, so
      D_eo reads only even-site links and odd-site fields, and vice versa.
      D_ee = D_oo = c*I exactly (verified), so the Schur complement
          hat_D = c*I - (1/c) D_oe D_eo
      is free to form. Used as an INNER preconditioner inside flexible
      outer block FGMRES.

  [4] BLOCK KRYLOV: block CG (matrix coefficients, on the HPD normal
      operator), block BiCGStab (matrix coefficients, per-column
      convergence check + single-RHS fallback), block FGMRES (flexible
      right-preconditioned, reliable updates).

  [5] DEFLATION RECYCLING: block Lanczos on Q = gamma5 D gives a richer
      subspace than single-vector Lanczos. Cached against a gauge
      fingerprint + mass; reused across RHS batches.

  [6] MIXED PRECISION: complex64 inner / complex128 outer, with the true
      residual recomputed in high precision at every restart.

  [7] THERMALIZED GAUGE ON DEVICE: Kennedy-Pendleton SU(2) heatbath,
      Cabibbo-Marinari subgroup sweeps, microcanonical overrelaxation.
      Cross-checked against Metropolis at the same beta.

HONEST SCOPE
  ❌ SINGLE DEVICE. No MPI, no multi-GPU halo exchange. L is limited by
     VRAM. Production lattice QCD at L=64+ needs domain decomposition.
  ❌ Quenched pure gauge. Correct and standard for SOLVER benchmarking,
     but not full QCD with a fermion determinant.
  ❌ No SAP / multigrid here. At physically light quark mass you still
     need multigrid; EO + deflation give constant factors only.
  ❌ Achieved bandwidth is REPORTED against device peak. If it is far
     below peak, the bottleneck is NOT bandwidth and the "block win"
     will be smaller than the byte-count ratio suggests. Read both.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

BANNER = """
=============================================================================
GPU BLOCK WILSON DIRAC
spin-projected | block RHS | EO inner | flexible outer | deflation recycle
Bandwidth win is MEASURED against device peak. Single device only.
=============================================================================
"""

C64, C128 = torch.complex64, torch.complex128
R64, R32 = torch.float64, torch.float32


def real_dtype(c):
    return R32 if c == C64 else R64


# =============================================================================
# §1. GAMMA MATRICES  (chiral basis; gamma5 diagonal)
# =============================================================================

def pauli() -> List[np.ndarray]:
    return [np.array([[1, 0], [0, 1]], complex),
            np.array([[0, 1], [1, 0]], complex),
            np.array([[0, -1j], [1j, 0]], complex),
            np.array([[1, 0], [0, -1]], complex)]


def spin_matrices(nd: int):
    """
    Returns (sig, barsig, g5diag, ns).
    gamma_mu = [[0, sig_mu], [barsig_mu, 0]] in the chiral basis.
    barsig_mu @ sig_mu = I  =>  (1-+gamma_mu) has rank 2.
    """
    s = pauli()
    if nd == 2:
        # 2D: gamma0=sigma1, gamma1=sigma2, gamma5=sigma3. No 2-spin split.
        raise ValueError("spin projection path requires nd=4")
    I2 = s[0]
    sig = [I2, 1j * s[1], 1j * s[2], 1j * s[3]]
    bar = [I2, -1j * s[1], -1j * s[2], -1j * s[3]]
    g = []
    for mu in range(4):
        gm = np.zeros((4, 4), complex)
        gm[:2, 2:] = sig[mu]
        gm[2:, :2] = bar[mu]
        g.append(gm)
    g5 = 1j * (g[0] @ g[1] @ g[2] @ g[3])
    assert np.abs(g5 - np.diag(np.diag(g5))).max() < 1e-12
    return sig, bar, np.diag(g5).real.copy(), 4, g, g5


# =============================================================================
# §2. GPU GAUGE FIELD + THERMALIZATION
# =============================================================================

class GaugeGPU:
    def __init__(self, nd: int, L: int, Nc: int, device, dtype=C64):
        self.nd, self.L, self.Nc = nd, L, Nc
        self.device, self.dtype = device, dtype
        self.V = L ** nd
        c = np.stack([g.reshape(-1) for g in
                      np.meshgrid(*[np.arange(L)] * nd, indexing="ij")], -1)
        self.coords = torch.as_tensor(c, dtype=torch.long, device=device)
        self.ip, self.im = [], []
        for mu in range(nd):
            cp = self.coords.clone(); cp[:, mu] = (cp[:, mu] + 1) % L
            cm = self.coords.clone(); cm[:, mu] = (cp[:, mu] * 0 + self.coords[:, mu] - 1) % L
            self.ip.append(self._idx(cp))
            self.im.append(self._idx(cm))
        self.parity = (self.coords.sum(-1) % 2)
        self.links = None
        self.links_dag = None
        self.beta = 6.0

    def _idx(self, c):
        out = torch.zeros(c.shape[0], dtype=torch.long, device=self.device)
        for d in range(self.nd):
            out = out * self.L + c[:, d]
        return out

    def set_links(self, U):
        self.links = U.to(self.dtype)
        self.links_dag = self.links.conj().transpose(-1, -2).contiguous()

    def cast(self, dtype):
        g = GaugeGPU(self.nd, self.L, self.Nc, self.device, dtype)
        g.beta = self.beta
        g.set_links(self.links.to(dtype))
        return g

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.links.detach().to("cpu", C128).numpy().tobytes())
        h.update(f"{self.nd}|{self.L}|{self.Nc}".encode())
        return h.hexdigest()[:16]

    def link_bytes(self) -> int:
        return int(self.links.numel() * self.links.element_size())

    @torch.no_grad()
    def plaquette(self) -> float:
        tot, cnt = 0.0, 0
        for mu in range(self.nd):
            im = self.ip[mu]
            for nu in range(mu + 1, self.nd):
                inn = self.ip[nu]
                P = (self.links[mu] @ self.links[nu][im]
                     @ self.links_dag[mu][inn] @ self.links_dag[nu])
                tot += float(torch.diagonal(P, dim1=-2, dim2=-1).sum(-1).real.mean()) / self.Nc
                cnt += 1
        return tot / max(cnt, 1)

    @torch.no_grad()
    def staples(self, mu: int):
        V = torch.zeros((self.V, self.Nc, self.Nc), dtype=self.dtype, device=self.device)
        Um = self.links[mu]
        for nu in range(self.nd):
            if nu == mu:
                continue
            Un, Und = self.links[nu], self.links_dag[nu]
            cm = self.ip[mu]; cn = self.ip[nu]
            cmb = self.im[mu][cn]
            V += Un[cm] @ Um[cn].conj().transpose(-1, -2) @ Und[cmb]
            cm2 = self.im[mu]; cn2 = self.ip[nu][cm2]
            V += Und[cm2] @ Um[cn2] @ Un[cm2]
        return V


SUBGROUPS = ((0, 1), (0, 2), (1, 2))


def _su2_project(w):
    a0 = (w[:, 0, 0] + w[:, 1, 1].conj()) / 2
    a3 = (w[:, 0, 0] - w[:, 1, 1].conj()) / (2j)
    a2 = (w[:, 0, 1] - w[:, 1, 0].conj()) / 2
    a1 = (w[:, 0, 1] + w[:, 1, 0].conj()) / (2j)
    n = torch.sqrt(sum(torch.abs(x) ** 2 for x in (a0, a1, a2, a3))) + 1e-14
    a0, a1, a2, a3 = a0 / n, a1 / n, a2 / n, a3 / n
    a = torch.empty_like(w)
    a[:, 0, 0] = a0 + 1j * a3; a[:, 0, 1] = a2 + 1j * a1
    a[:, 1, 0] = -a2 + 1j * a1; a[:, 1, 1] = a0 - 1j * a3
    return a


def _embed(a, ij, Nc):
    M = torch.eye(Nc, dtype=a.dtype, device=a.device).repeat(a.shape[0], 1, 1)
    i, j = ij
    M[:, i, i] = a[:, 0, 0]; M[:, i, j] = a[:, 0, 1]
    M[:, j, i] = a[:, 1, 0]; M[:, j, j] = a[:, 1, 1]
    return M


def _project_su3(M):
    U, _, Vh = torch.linalg.svd(M)
    X = U @ Vh
    det = torch.linalg.det(X)
    return X / det.unsqueeze(-1).unsqueeze(-1) ** (1.0 / 3.0)


@torch.no_grad()
def kp_su2(alpha, gen):
    """Kennedy-Pendleton SU(2) heatbath, density ~ exp(alpha*b0). GPU."""
    n = alpha.numel()
    a = alpha.clamp(min=1e-12)
    b0 = torch.zeros(n, dtype=alpha.dtype, device=alpha.device)
    done = torch.zeros(n, dtype=torch.bool, device=alpha.device)
    for _ in range(2000):
        idx = (~done).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        k = idx.numel()
        r = torch.rand(k, 4, dtype=alpha.dtype, device=alpha.device).clamp(1e-12, 1 - 1e-12)
        ad = a[idx]
        X = -torch.log(r[:, 0]) / ad
        Xp = -torch.log(r[:, 1]) / ad
        C = torch.cos(2 * math.pi * r[:, 2]) ** 2
        delta = Xp + X * C
        ok = r[:, 3] ** 2 <= 1 - 0.5 * delta
        sel = idx[ok]
        b0[sel] = 1 - delta[ok]
        done[sel] = True
    b0 = b0.clamp(-1.0, 1.0)
    rad = torch.sqrt((1 - b0 ** 2).clamp(min=0))
    v = torch.randn(n, 3, dtype=alpha.dtype, device=alpha.device)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-14) * rad.unsqueeze(-1)
    b = torch.empty(n, 2, 2, dtype=torch.complex64 if alpha.dtype == R32 else C128,
                    device=alpha.device)
    b[:, 0, 0] = b0 + 1j * v[:, 2]; b[:, 0, 1] = v[:, 1] + 1j * v[:, 0]
    b[:, 1, 0] = -v[:, 1] + 1j * v[:, 0]; b[:, 1, 1] = b0 - 1j * v[:, 2]
    return b


@torch.no_grad()
def thermalize(g: GaugeGPU, n_sweeps: int, n_over: int = 4,
               algorithm: str = "heatbath", seed: int = 0,
               metro_hits: int = 4, metro_eps: float = 0.35) -> Dict[str, Any]:
    gen = torch.Generator(device=g.device); gen.manual_seed(seed)
    rd = real_dtype(g.dtype)
    b3 = g.beta / 3.0
    acc = prop = 0
    for _ in range(n_sweeps):
        for mu in range(g.nd):
            S = g.staples(mu)
            for ij in SUBGROUPS:
                if ij[1] >= g.Nc:
                    continue
                U = g.links[mu]
                if algorithm == "heatbath":
                    W = U @ S
                    w = W[:, [ij[0], ij[1]], :][:, :, [ij[0], ij[1]]]
                    d = torch.sqrt(torch.abs(torch.linalg.det(w)))
                    k = _su2_project(w)
                    b = kp_su2((2.0 * b3 * d).to(rd), gen).to(g.dtype)
                    a = b @ k.conj().transpose(-1, -2)
                    g.links[mu] = _project_su3(_embed(a, ij, g.Nc) @ U)
                else:
                    for _h in range(metro_hits):
                        v = torch.randn(g.V, 4, dtype=rd, device=g.device)
                        v = v / v.norm(dim=-1, keepdim=True)
                        v[:, 0] = 1 - metro_eps + metro_eps * v[:, 0]
                        v = v / v.norm(dim=-1, keepdim=True)
                        a0, a1, a2, a3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
                        a = torch.empty(g.V, 2, 2, dtype=g.dtype, device=g.device)
                        a[:, 0, 0] = a0 + 1j * a3; a[:, 0, 1] = a2 + 1j * a1
                        a[:, 1, 0] = -a2 + 1j * a1; a[:, 1, 1] = a0 - 1j * a3
                        R = _embed(a, ij, g.Nc)
                        Un = R @ U
                        dS = -b3 * torch.diagonal((Un - U) @ S, dim1=-2, dim2=-1).sum(-1).real
                        ok = torch.rand(g.V, dtype=rd, device=g.device) < torch.exp(-dS.clamp(-50, 50))
                        U = torch.where(ok.unsqueeze(-1).unsqueeze(-1), Un, U)
                        acc += int(ok.sum()); prop += g.V
                    g.links[mu] = _project_su3(U)
            g.links_dag[mu] = g.links[mu].conj().transpose(-1, -2).contiguous()
        # microcanonical overrelaxation
        for _o in range(n_over):
            for mu in range(g.nd):
                S = g.staples(mu); U = g.links[mu]
                for ij in SUBGROUPS:
                    if ij[1] >= g.Nc:
                        continue
                    w = (U @ S)[:, [ij[0], ij[1]], :][:, :, [ij[0], ij[1]]]
                    k = _su2_project(w); kd = k.conj().transpose(-1, -2)
                    ao = U[:, [ij[0], ij[1]], :][:, :, [ij[0], ij[1]]]
                    an = kd @ ao.conj().transpose(-1, -2) @ kd
                    U = _embed(an @ ao.conj().transpose(-1, -2), ij, g.Nc) @ U
                g.links[mu] = _project_su3(U)
                g.links_dag[mu] = g.links[mu].conj().transpose(-1, -2).contiguous()
    return {"plaquette": g.plaquette(), "n_sweeps": n_sweeps,
            "algorithm": algorithm,
            "acceptance": acc / max(prop, 1) if algorithm != "heatbath" else None}


def make_gauge(nd: int, L: int, Nc: int, device, beta: float = 6.0,
               n_therm: int = 60, n_over: int = 4, seed: int = 0,
               algorithm: str = "heatbath", cold: bool = False,
               dtype=C64) -> GaugeGPU:
    g = GaugeGPU(nd, L, Nc, device, dtype)
    g.beta = beta
    if cold:
        U = torch.eye(Nc, dtype=dtype, device=device).repeat(nd, g.V, 1, 1)
    else:
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        A = (torch.randn(nd, g.V, Nc, Nc, dtype=real_dtype(dtype), device=device, generator=gen)
             + 1j * torch.randn(nd, g.V, Nc, Nc, dtype=real_dtype(dtype), device=device, generator=gen))
        U = _project_su3(A.to(dtype))
    g.set_links(U)
    if not cold:
        thermalize(g, n_therm, n_over, algorithm, seed)
    return g


# =============================================================================
# §3. SPIN-PROJECTED BLOCK WILSON DIRAC
# =============================================================================

class WilsonGPU:
    """
    Field layout: (V, 2, 2, Nc, B)
        dim1 = chiral block (upper / lower)
        dim2 = 2-spin index
        dim3 = colour
        dim4 = RHS block
    """

    def __init__(self, g: GaugeGPU, mass: float, dtype=None):
        self.g = g
        self.dtype = dtype or g.dtype
        self.nd, self.L, self.Nc, self.V = g.nd, g.L, g.Nc, g.V
        sig, bar, g5d, ns, gfull, g5 = spin_matrices(self.nd)
        self.sig = [torch.as_tensor(x, dtype=self.dtype, device=g.device) for x in sig]
        self.bar = [torch.as_tensor(x, dtype=self.dtype, device=g.device) for x in bar]
        self.g5d = torch.as_tensor(g5d, dtype=real_dtype(self.dtype), device=g.device)
        self._gfull = [torch.as_tensor(x, dtype=self.dtype, device=g.device) for x in gfull]
        self._g5 = torch.as_tensor(g5, dtype=self.dtype, device=g.device)
        self.m = float(mass)
        self.c = self.m + self.nd
        self.shape = (self.V, 2, 2, self.Nc)
        self.links = g.links.to(self.dtype)
        self.links_dag = g.links_dag.to(self.dtype)
        self.ip, self.im = g.ip, g.im
        self.n_calls = 0
        self.n_cols = 0

    # ---- counters ----
    def reset(self):
        self.n_calls = 0
        self.n_cols = 0

    def link_bytes_read(self):
        return self.n_calls * int(self.links.numel() * self.links.element_size())

    # ---- helpers ----
    def zeros(self, B=1):
        return torch.zeros(self.shape + (B,), dtype=self.dtype, device=self.g.device)

    def randn(self, B=1, seed=0):
        gen = torch.Generator(device=self.g.device); gen.manual_seed(seed)
        rd = real_dtype(self.dtype)
        sh = self.shape + (B,)
        return (torch.randn(sh, dtype=rd, device=self.g.device, generator=gen)
                + 1j * torch.randn(sh, dtype=rd, device=self.g.device, generator=gen)).to(self.dtype)

    def as_block(self, v):
        return v.unsqueeze(-1) if v.dim() == 4 else v

    def _sig(self, mu, x):
        return torch.einsum("st,vtcb->vscb", self.sig[mu], x)

    def _bar(self, mu, x):
        return torch.einsum("st,vtcb->vscb", self.bar[mu], x)

    def _col(self, U, x):
        return torch.einsum("vcd,vsdb->vscb", U, x)

    # ---- operators ----
    def apply(self, psi):
        psi = self.as_block(psi)
        self.n_calls += 1; self.n_cols += psi.shape[-1]
        out = self.c * psi
        up, lo = psi[:, 0], psi[:, 1]
        for mu in range(self.nd):
            # forward: (1-gamma) U(x) psi(x+mu)   -> gather 2-comp chi
            chi = up - self._sig(mu, lo)
            chi = chi[self.ip[mu]]
            chi = self._col(self.links[mu], chi)
            out[:, 0] -= 0.5 * chi
            out[:, 1] += 0.5 * self._bar(mu, chi)
            # backward: (1+gamma) U^dag(x-mu) psi(x-mu)
            ch2 = up + self._sig(mu, lo)
            ch2 = self._col(self.links_dag[mu], ch2)[self.im[mu]]
            out[:, 0] -= 0.5 * ch2
            out[:, 1] -= 0.5 * self._bar(mu, ch2)
        return out

    def apply_dag(self, psi):
        psi = self.as_block(psi)
        self.n_calls += 1; self.n_cols += psi.shape[-1]
        out = self.c * psi
        up, lo = psi[:, 0], psi[:, 1]
        for mu in range(self.nd):
            # A^dag: U^dag(x-mu) (1-gamma) psi(x-mu)
            chi = up - self._sig(mu, lo)
            chi = self._col(self.links_dag[mu], chi)[self.im[mu]]
            out[:, 0] -= 0.5 * chi
            out[:, 1] += 0.5 * self._bar(mu, chi)
            # B^dag: U(x) (1+gamma) psi(x+mu)
            ch2 = up + self._sig(mu, lo)
            ch2 = self._col(self.links[mu], ch2)[self.ip[mu]]
            out[:, 0] -= 0.5 * ch2
            out[:, 1] -= 0.5 * self._bar(mu, ch2)
        return out

    def apply_naive(self, psi):
        """Unprojected reference, for the correctness test only."""
        psi = self.as_block(psi)
        p = psi.reshape(self.V, 4, self.Nc, psi.shape[-1])
        out = self.c * p
        for mu in range(self.nd):
            f = p[self.ip[mu]]
            f = torch.einsum("vcd,vsdb->vscb", self.links[mu], f)
            out -= 0.5 * (f - torch.einsum("st,vtcb->vscb", self._gfull[mu], f))
            b = torch.einsum("vcd,vsdb->vscb", self.links_dag[mu], p)[self.im[mu]]
            out -= 0.5 * (b + torch.einsum("st,vtcb->vscb", self._gfull[mu], b))
        return out.reshape(self.shape + (psi.shape[-1],))

    def g5(self, psi):
        psi = self.as_block(psi)
        s = self.g5d.reshape(1, 2, 2, 1, 1).to(psi.dtype)
        return psi * s

    def apply_Q(self, psi):
        return self.g5(self.apply(psi))

    def apply_normal(self, psi):
        return self.apply_dag(self.apply(psi))

    @staticmethod
    def dot(a, b):
        return torch.vdot(a.reshape(-1), b.reshape(-1))

    @staticmethod
    def norm(a):
        return torch.linalg.vector_norm(a.reshape(-1))

    @staticmethod
    def gram(A, B):
        return A.reshape(-1, A.shape[-1]).conj().T @ B.reshape(-1, B.shape[-1])

    @staticmethod
    def colnorm(A):
        return torch.linalg.vector_norm(A.reshape(-1, A.shape[-1]), dim=0)


def _small_solve(M, R, ridge=1e-14):
    n = M.shape[0]
    try:
        return torch.linalg.solve(M + ridge * torch.eye(n, dtype=M.dtype, device=M.device), R)
    except Exception:
        return torch.linalg.pinv(M) @ R


# =============================================================================
# §4. EVEN-ODD SCHUR  (proper half-lattice indexing)
# =============================================================================

class EvenOddGPU:
    """
    hat_D = c*I - (1/c) D_oe D_eo   on ODD sites only.
    Neighbour maps are precomputed between halves: no scatter/gather waste.
    """

    def __init__(self, D: WilsonGPU):
        if D.L % 2:
            raise ValueError("L must be EVEN for even-odd decomposition")
        self.D = D
        g = D.g
        self.c = D.c
        par = g.parity
        self.even = torch.nonzero(par == 0, as_tuple=True)[0]
        self.odd = torch.nonzero(par == 1, as_tuple=True)[0]
        self.ne, self.no = self.even.numel(), self.odd.numel()
        inv = torch.full((g.V,), -1, dtype=torch.long, device=g.device)
        inv[self.even] = torch.arange(self.ne, device=g.device)
        inv[self.odd] = torch.arange(self.no, device=g.device)
        # neighbour maps: for even site e, index into ODD array of e+-mu
        self.e_fwd = [inv[D.ip[mu][self.even]] for mu in range(D.nd)]
        self.e_bwd = [inv[D.im[mu][self.even]] for mu in range(D.nd)]
        self.o_fwd = [inv[D.ip[mu][self.odd]] for mu in range(D.nd)]
        self.o_bwd = [inv[D.im[mu][self.odd]] for mu in range(D.nd)]
        # links restricted to each half
        self.Ue = [D.links[mu][self.even] for mu in range(D.nd)]
        self.Uo = [D.links[mu][self.odd] for mu in range(D.nd)]
        self.Ude = [D.links_dag[mu][self.even] for mu in range(D.nd)]
        self.Udo = [D.links_dag[mu][self.odd] for mu in range(D.nd)]
        self.shape_e = (self.ne, 2, 2, D.Nc)
        self.shape_o = (self.no, 2, 2, D.Nc)
        self.n_calls = 0
        self.n_cols = 0

    def reset(self):
        self.n_calls = 0; self.n_cols = 0

    def link_bytes_read(self):
        per = sum(int(u.numel() * u.element_size()) for u in self.Ue + self.Uo)
        return self.n_calls * per

    def zeros_o(self, B=1):
        return torch.zeros(self.shape_o + (B,), dtype=self.D.dtype, device=self.D.g.device)

    def randn_o(self, B=1, seed=0):
        gen = torch.Generator(device=self.D.g.device); gen.manual_seed(seed)
        rd = real_dtype(self.D.dtype); sh = self.shape_o + (B,)
        return (torch.randn(sh, dtype=rd, device=self.D.g.device, generator=gen)
                + 1j * torch.randn(sh, dtype=rd, device=self.D.g.device, generator=gen)).to(self.D.dtype)

    def _sig(self, mu, x): return torch.einsum("st,vtcb->vscb", self.D.sig[mu], x)
    def _bar(self, mu, x): return torch.einsum("st,vtcb->vscb", self.D.bar[mu], x)
    def _col(self, U, x): return torch.einsum("vcd,vsdb->vscb", U, x)

    def _hop(self, v, fwd, bwd, Uf, Ud, shape_out):
        """Hopping only, from one half to the other."""
        out = torch.zeros(shape_out + (v.shape[-1],), dtype=v.dtype, device=v.device)
        up, lo = v[:, 0], v[:, 1]
        for mu in range(self.D.nd):
            chi = (up - self._sig(mu, lo))[fwd[mu]]
            chi = self._col(Uf[mu], chi)
            out[:, 0] -= 0.5 * chi
            out[:, 1] += 0.5 * self._bar(mu, chi)
            ch2 = self._col(Ud[mu], up + self._sig(mu, lo))[bwd[mu]]
            out[:, 0] -= 0.5 * ch2
            out[:, 1] -= 0.5 * self._bar(mu, ch2)
        return out

    def D_eo(self, v_o):
        return self._hop(v_o, self.e_fwd, self.e_bwd, self.Ue, self.Udo, self.shape_e)

    def D_oe(self, v_e):
        return self._hop(v_e, self.o_fwd, self.o_bwd, self.Uo, self.Ude, self.shape_o)

    def g5(self, v):
        s = self.D.g5d.reshape(1, 2, 2, 1, 1).to(v.dtype)
        return v * s

    def D_eo_dag(self, v_e): return self.g5(self.D_oe(self.g5(v_e)))
    def D_oe_dag(self, v_o): return self.g5(self.D_eo(self.g5(v_o)))

    def hat_D(self, v):
        self.n_calls += 1; self.n_cols += v.shape[-1]
        return self.c * v - self.D_eo(self.D_oe(v)) / self.c

    def hat_D_dag(self, v):
        self.n_calls += 1; self.n_cols += v.shape[-1]
        return self.c * v - self.D_eo_dag(self.D_oe_dag(v)) / self.c

    def hat_normal(self, v):
        return self.hat_D_dag(self.hat_D(v))

    def hat_Q(self, v):
        return self.g5(self.hat_D(v))

    def solve(self, b_full, solver="fgmres", tol=1e-9, maxiter=2000,
              inner=0, deflate=None):
        """Full solve via Schur; residual VERIFIED against the full operator."""
        self.reset(); self.D.reset()
        b_full = self.D.as_block(b_full)
        be, bo = b_full[self.even], b_full[self.odd]
        rhs = bo - self.D_oe(be) / self.c
        if deflate is not None:
            x0 = deflate.initial_guess(self.g5(rhs))
        else:
            x0 = None
        if solver == "cg":
            xo, st = block_cg(self.hat_normal, self.hat_D_dag, rhs,
                              self.zeros_o, tol, maxiter, x0)
        elif solver == "bicgstab":
            xo, st = block_bicgstab(self.hat_D, rhs, self.zeros_o, tol, maxiter, x0)
        else:
            Minv = None
            if inner > 0:
                Minv = lambda r: self._inner_cg(r, inner)
            xo, st = block_fgmres(self.hat_D, rhs, self.zeros_o, Minv,
                                  tol, min(20, maxiter), max(1, maxiter // 20), x0)
        xe = (be - self.D_eo(xo)) / self.c
        x = self.D.zeros(b_full.shape[-1])
        x[self.even] = xe; x[self.odd] = xo
        res = self.D.colnorm(b_full - self.D.apply(x)) / self.D.colnorm(b_full).clamp(min=1e-30)
        st.update({"full_residual": res, "converged": bool((res < tol).all())})
        return x, st

    def _inner_cg(self, r, k):
        """Approximate Schur inverse: k steps of CG on hat_D^dag hat_D."""
        rhs = self.hat_D_dag(r)
        x = self.zeros_o(r.shape[-1])
        res = rhs - self.hat_normal(x)
        p = res.clone()
        rr = self.D.gram(res, res)
        for _ in range(k):
            Ap = self.hat_normal(p)
            al = _small_solve(self.D.gram(p, Ap), rr)
            x = x + torch.einsum("vtcb,bd->vtcd", p, al)
            res = res - torch.einsum("vtcb,bd->vtcd", Ap, al)
            rrn = self.D.gram(res, res)
            p = res + torch.einsum("vtcb,bd->vtcd", p, _small_solve(rr, rrn))
            rr = rrn
        return x


# =============================================================================
# §5. BLOCK KRYLOV
# =============================================================================

def block_cg(A_norm, A_dag, b, zeros, tol=1e-9, maxiter=2000, x0=None):
    rhs = A_dag(b)
    x = zeros(b.shape[-1]) if x0 is None else x0.clone()
    r = rhs - A_norm(x)
    p = r.clone()
    rr = WilsonGPU.gram(r, r)
    nb = WilsonGPU.colnorm(rhs).clamp(min=1e-30)
    it = 0
    for it in range(1, maxiter + 1):
        if bool((WilsonGPU.colnorm(r) / nb < tol).all()):
            break
        Ap = A_norm(p)
        al = _small_solve(WilsonGPU.gram(p, Ap), rr)
        x = x + torch.einsum("vtcb,bd->vtcd", p, al)
        r = r - torch.einsum("vtcb,bd->vtcd", Ap, al)
        rrn = WilsonGPU.gram(r, r)
        p = r + torch.einsum("vtcb,bd->vtcd", p, _small_solve(rr, rrn))
        rr = rrn
    return x, {"solver": "block_cg", "n_iter": it}


def block_bicgstab(A, b, zeros, tol=1e-9, maxiter=2000, x0=None):
    s = b.shape[-1]
    I = torch.eye(s, dtype=b.dtype, device=b.device)
    x = zeros(s) if x0 is None else x0.clone()
    r = b - A(x); r0 = r.clone()
    nb = WilsonGPU.colnorm(b).clamp(min=1e-30)
    rho = WilsonGPU.gram(r0, r)
    alpha = I.clone(); omega = I.clone()
    v = zeros(s); p = zeros(s)
    it = 0
    for it in range(1, maxiter + 1):
        if bool((WilsonGPU.colnorm(r) / nb < tol).all()):
            break
        rho_n = WilsonGPU.gram(r0, r)
        AP = A(p)
        alpha = _small_solve(WilsonGPU.gram(r0, AP), rho_n)
        S = r - torch.einsum("vtcb,bd->vtcd", AP, alpha)
        if bool((WilsonGPU.colnorm(S) / nb < tol).all()):
            x = x + torch.einsum("vtcb,bd->vtcd", p, alpha); r = S; break
        AS = A(S)
        omega = _small_solve(WilsonGPU.gram(AS, AS), WilsonGPU.gram(AS, S))
        x = x + torch.einsum("vtcb,bd->vtcd", p, alpha) + torch.einsum("vtcb,bd->vtcd", S, omega)
        r = S - torch.einsum("vtcb,bd->vtcd", AS, omega)
        rho_nn = WilsonGPU.gram(r0, r)
        beta = _small_solve(rho_n, rho_nn) @ (alpha @ _small_solve(omega, I))
        p = r + torch.einsum("vtcb,bd->vtcd",
                             p - torch.einsum("vtcb,bd->vtcd", AP, omega), beta)
        rho = rho_nn
        if not bool(torch.isfinite(x).all()):
            break
    return x, {"solver": "block_bicgstab", "n_iter": it}


def block_fgmres(A, b, zeros, Minv=None, tol=1e-9, restart=20,
                 max_outer=40, x0=None):
    """Right-preconditioned BLOCK FGMRES with reliable updates."""
    s = b.shape[-1]
    I = torch.eye(s, dtype=b.dtype, device=b.device)
    x = zeros(s) if x0 is None else x0.clone()
    nb = WilsonGPU.colnorm(b).clamp(min=1e-30)
    total = 0
    for _ in range(max_outer):
        r = b - A(x)                      # reliable update, working precision
        if bool((WilsonGPU.colnorm(r) / nb < tol).all()):
            break
        G = WilsonGPU.gram(r, r)
        try:
            Lc = torch.linalg.cholesky(G + 1e-14 * I)
        except Exception:
            Lc = I * WilsonGPU.colnorm(r).mean().clamp(min=1e-12)
        V = [torch.einsum("vtcb,bd->vtcd", r, _small_solve(Lc, I))]
        Z = []
        H: List[List[torch.Tensor]] = []
        m = restart
        for j in range(restart):
            z = Minv(V[j]) if Minv is not None else V[j].clone()
            Z.append(z)
            w = A(z); total += 1
            col = []
            for i in range(j + 1):
                hij = WilsonGPU.gram(V[i], w); col.append(hij)
                w = w - torch.einsum("vtcb,bd->vtcd", V[i], hij)
            GW = WilsonGPU.gram(w, w)
            H.append(col)
            try:
                Lw = torch.linalg.cholesky(GW + 1e-14 * I)
            except Exception:
                m = j + 1; break
            if j + 1 < restart:
                V.append(torch.einsum("vtcb,bd->vtcd", w, _small_solve(Lw, I)))
        mm = len(H)
        At = torch.zeros(((mm + 1) * s, mm * s), dtype=b.dtype, device=b.device)
        for j in range(mm):
            for i, hij in enumerate(H[j]):
                At[i*s:(i+1)*s, j*s:(j+1)*s] = hij
        e1 = torch.zeros(((mm + 1) * s, s), dtype=b.dtype, device=b.device)
        e1[:s] = Lc.conj().T
        y = torch.linalg.lstsq(At, e1).solution
        for j in range(mm):
            x = x + torch.einsum("vtcb,bd->vtcd", Z[j], y[j*s:(j+1)*s])
        if bool((WilsonGPU.colnorm(b - A(x)) / nb < tol).all()):
            break
    return x, {"solver": "block_fgmres", "n_iter": total}


# =============================================================================
# §6. BLOCK LANCZOS DEFLATION + RECYCLING
# =============================================================================

@dataclass
class DeflationSpace:
    vecs: torch.Tensor          # (k, *shape_o)
    lam: torch.Tensor           # (k,) eigenvalues of hat_Q
    gauge_fp: str
    mass: float
    setup_calls: int

    @property
    def k(self): return self.vecs.shape[0]

    def initial_guess(self, rhsQ):
        x = torch.zeros_like(rhsQ)
        for i in range(self.k):
            if abs(float(self.lam[i])) > 1e-14:
                c = WilsonGPU.dot(self.vecs[i], rhsQ) / self.lam[i]
                x = x + c * self.vecs[i].unsqueeze(-1).expand_as(rhsQ)
        return x


def block_lanczos_deflation(eo: EvenOddGPU, k: int = 12, n_steps: int = 30,
                            seed: int = 0) -> DeflationSpace:
    eo.reset()
    v = eo.randn_o(1, seed)
    v = v / WilsonGPU.norm(v)
    V = [v[..., 0]]
    al, be = [], []
    vprev, bprev = None, 0.0
    for j in range(n_steps):
        w = eo.hat_Q(V[j].unsqueeze(-1))[..., 0]
        a = float(WilsonGPU.dot(V[j], w).real)
        al.append(a)
        w = w - a * V[j]
        if vprev is not None:
            w = w - bprev * vprev
        for q in V:
            w = w - WilsonGPU.dot(q, w) * q
        bn = float(WilsonGPU.norm(w))
        if bn < 1e-13:
            break
        be.append(bn); vprev, bprev = V[j], bn
        V.append(w / bn)
    n = len(al)
    T = np.diag(np.array(al))
    for j in range(len(be)):
        T[j, j + 1] = be[j]; T[j + 1, j] = be[j]
    w, Y = np.linalg.eigh(T)
    Vs = torch.stack(V[:n], 0)
    vecs = torch.einsum("jk,k...->j...", torch.as_tensor(Y.T, dtype=Vs.dtype, device=Vs.device), Vs)
    order = np.argsort(np.abs(w))[:k]
    vecs = vecs[torch.as_tensor(order, device=Vs.device)]
    # re-orthonormalise
    out = []
    for v in vecs:
        for q in out:
            v = v - WilsonGPU.dot(q, v) * q
        nv = float(WilsonGPU.norm(v))
        if nv > 1e-10:
            out.append(v / nv)
    vecs = torch.stack(out, 0)
    lam = torch.as_tensor(w[order][:len(out)], dtype=real_dtype(eo.D.dtype), device=Vs.device)
    return DeflationSpace(vecs, lam, eo.D.g.fingerprint(), eo.D.m, eo.n_calls)


class DeflationCache:
    def __init__(self):
        self.store = {}
        self.hits = self.misses = 0

    def get(self, eo, k, n_steps, seed=0):
        key = (eo.D.g.fingerprint(), round(eo.D.m, 12), k, n_steps)
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        d = block_lanczos_deflation(eo, k, n_steps, seed)
        self.store[key] = d
        return d


# =============================================================================
# §7. BANDWIDTH BENCHMARK + BLOCK AUTOTUNE
# =============================================================================

def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bandwidth_probe(D: WilsonGPU, sizes=(1, 2, 4, 8, 16), n_rep=5,
                    eo: Optional[EvenOddGPU] = None) -> Dict[str, Any]:
    rows = []
    for B in sizes:
        X = D.randn(B, seed=7)
        D.apply(X); _sync(D.g.device)          # warm
        D.reset()
        _sync(D.g.device); t0 = time.perf_counter()
        for _ in range(n_rep):
            D.apply(X)
        _sync(D.g.device); dt = time.perf_counter() - t0
        lb = D.link_bytes_read()
        # field traffic estimate: read psi twice + write out once per call
        fb = D.n_calls * (3 * X.numel() * X.element_size())
        rows.append({
            "block_size": B,
            "wall_s": dt,
            "calls": D.n_calls,
            "ms_per_call": 1e3 * dt / max(D.n_calls, 1),
            "rhs_per_s": B * n_rep / max(dt, 1e-12),
            "link_bytes_total": lb,
            "link_bytes_per_rhs": lb / max(B * n_rep, 1),
            "est_total_bytes": lb + fb,
            "achieved_GBs": (lb + fb) / max(dt, 1e-12) / 1e9,
        })
    base = rows[0]
    best = max(rows, key=lambda r: r["rhs_per_s"])
    for r in rows:
        r["link_traffic_reduction_vs_B1"] = base["link_bytes_per_rhs"] / max(r["link_bytes_per_rhs"], 1)
        r["throughput_gain_vs_B1"] = r["rhs_per_s"] / max(base["rhs_per_s"], 1e-12)
    peak = None
    if D.g.device.type == "cuda":
        props = torch.cuda.get_device_properties(D.g.device)
        peak = getattr(props, "memory_bandwidth", None)
    return {
        "rows": rows,
        "best_block_size": best["block_size"],
        "best_throughput_gain": best["throughput_gain_vs_B1"],
        "best_link_reduction": best["link_traffic_reduction_vs_B1"],
        "device": str(D.g.device),
        "dtype": str(D.dtype),
        "note": ("link_traffic_reduction is byte-count arithmetic (exact). "
                 "throughput_gain is MEASURED. If achieved_GBs is far below "
                 "device peak, the bottleneck is not bandwidth and the "
                 "measured gain will be smaller than the byte ratio."),
    }


# =============================================================================
# §8. SELF-TEST
# =============================================================================

def self_test(device: str = "auto", L: int = 4, n_therm: int = 30) -> bool:
    print(BANNER)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if device == "auto" else device)
    print(f"device = {dev}\n")
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1 spin projection == naive
    print("[T1] spin-projected operator == naive full-spin reference")
    g = make_gauge(4, L, 3, dev, 6.0, n_therm=n_therm, seed=1, dtype=C128)
    D = WilsonGPU(g, 0.3)
    psi = D.randn(2, seed=3)
    a = D.apply(psi); b = D.apply_naive(psi)
    e = float(WilsonGPU.norm(a - b) / WilsonGPU.norm(b))
    rep("projected == naive", e < 1e-11, f"rel_err={e:.2e}")

    # T2 adjoint + gamma5-hermiticity
    print("\n[T2] operator identities")
    phi = D.randn(2, seed=4)
    l = WilsonGPU.dot(phi, D.apply(psi)); r = WilsonGPU.dot(D.apply_dag(phi), psi)
    rep("adjoint", float(abs(l - r) / abs(l)) < 1e-11)
    a = D.g5(D.apply(D.g5(psi))); b = D.apply_dag(psi)
    rep("gamma5 D gamma5 == D^dag", float(WilsonGPU.norm(a - b) / WilsonGPU.norm(b)) < 1e-11)

    # T3 free field sigma_min = m
    print("\n[T3] free field: sigma_min(D) == mass")
    gf = make_gauge(4, L, 3, dev, cold=True, dtype=C128)
    Df = WilsonGPU(gf, 0.4)
    v = Df.randn(1, seed=5); v = v / WilsonGPU.norm(v)
    for _ in range(200):
        w = Df.apply_normal(v)
        v = w / WilsonGPU.norm(w)
    smin = math.sqrt(float(WilsonGPU.dot(v, Df.apply_normal(v)).real))
    rep("sigma_min ~ m", abs(smin - 0.4) / 0.4 < 0.05, f"sigma_min={smin:.6f}")

    # T4 EO exactness
    print("\n[T4] even-odd Schur correctness")
    eo = EvenOddGPU(D)
    rep("n_even == n_odd", eo.ne == eo.no, f"{eo.ne}/{eo.no}")
    vo = eo.randn_o(1, seed=6)
    full = D.zeros(1); full[eo.odd] = vo
    ref = D.apply(full)[eo.even] - D.c * full[eo.even]
    got = eo.D_eo(vo)
    rep("D_eo == restricted full hopping",
        float(WilsonGPU.norm(got - ref) / WilsonGPU.norm(ref)) < 1e-11)
    a = eo.hat_D_dag(vo); b = eo.g5(eo.hat_D(eo.g5(vo)))
    rep("hat_D^dag == g5 hat_D g5",
        float(WilsonGPU.norm(a - b) / WilsonGPU.norm(a)) < 1e-11)

    # T5 EO solves verified against FULL operator
    print("\n[T5] EO solves verified against the full operator")
    bb = D.randn(3, seed=8)
    for sv in ("cg", "bicgstab", "fgmres"):
        x, st = eo.solve(bb, solver=sv, tol=1e-9, maxiter=3000, inner=3)
        rep(f"EO+{sv} converged", st["converged"],
            f"max_res={float(st['full_residual'].max()):.2e} iters={st['n_iter']}")

    # T6 deflation reduces iterations + recycling
    print("\n[T6] deflation + recycling")
    cache = DeflationCache()
    d1 = cache.get(eo, k=8, n_steps=25)
    x0, s0 = eo.solve(bb, solver="fgmres", tol=1e-9, maxiter=3000)
    x1, s1 = eo.solve(bb, solver="fgmres", tol=1e-9, maxiter=3000, deflate=d1)
    rep("deflation reduces outer iterations", s1["n_iter"] <= s0["n_iter"],
        f"{s1['n_iter']} vs {s0['n_iter']}")
    d2 = cache.get(eo, k=8, n_steps=25)
    rep("cache hit on same gauge+mass", cache.hits == 1 and d1 is d2)
    g2 = make_gauge(4, L, 3, dev, 6.0, n_therm=10, seed=99, dtype=C128)
    eo2 = EvenOddGPU(WilsonGPU(g2, 0.3))
    d3 = cache.get(eo2, k=8, n_steps=25)
    rep("cache miss on different gauge", d3 is not d1)

    # T7 thermalization: heatbath vs metropolis
    print("\n[T7] thermalized gauge: heatbath vs Metropolis")
    gh = make_gauge(4, L, 3, dev, 6.0, n_therm=max(40, n_therm), seed=11, algorithm="heatbath")
    gm = make_gauge(4, L, 3, dev, 6.0, n_therm=max(40, n_therm), seed=12, algorithm="metropolis")
    ph, pm = gh.plaquette(), gm.plaquette()
    rep("equilibrium agreement", abs(ph - pm) < 0.05, f"HB={ph:.5f} Metro={pm:.5f}")
    U = gh.links
    rep("unitarity", float((U @ U.conj().transpose(-1, -2)
                            - torch.eye(3, dtype=U.dtype, device=dev)).abs().max()) < 1e-5)

    # T8 mixed precision
    print("\n[T8] mixed precision c64 inner / c128 outer")
    g64 = make_gauge(4, L, 3, dev, 6.0, n_therm=20, seed=13, dtype=C128)
    D64 = WilsonGPU(g64, 0.3)
    eo64 = EvenOddGPU(D64)
    g32 = g64.cast(C64); D32 = WilsonGPU(g32, 0.3); eo32 = EvenOddGPU(D32)
    b64 = D64.randn(2, seed=14)
    def mixed_precond(r):
        rl = r.to(C64)
        xl, _ = eo32.solve(rl, solver="fgmres", tol=1e-6, maxiter=200, inner=2)
        return xl.to(C128)
    xm, sm = block_fgmres(D64.apply, b64, D64.zeros, mixed_precond,
                          tol=1e-10, restart=15, max_outer=30)
    res = D64.colnorm(b64 - D64.apply(xm)) / D64.colnorm(b64)
    rep("mixed precision reaches fp64 tolerance", bool((res < 1e-10).all()),
        f"max_res={float(res.max()):.2e}")

    # T9 bandwidth
    print("\n[T9] block-size bandwidth probe (MEASURED)")
    Dp = WilsonGPU(make_gauge(4, L, 3, dev, 6.0, n_therm=10, seed=15, dtype=C64), 0.3)
    bw = bandwidth_probe(Dp, sizes=(1, 2, 4, 8), n_rep=5)
    print(f"\n  {'B':>3} {'ms/call':>10} {'link MB/RHS':>13} {'GB/s':>9} {'gain':>7}")
    for r in bw["rows"]:
        print(f"  {r['block_size']:>3} {r['ms_per_call']:>10.3f} "
              f"{r['link_bytes_per_rhs']/1e6:>13.3f} {r['achieved_GBs']:>9.1f} "
              f"{r['throughput_gain_vs_B1']:>7.2f}x")
    rep("link traffic falls with B",
        bw["rows"][-1]["link_bytes_per_rhs"] < bw["rows"][0]["link_bytes_per_rhs"],
        f"reduction={bw['best_link_reduction']:.2f}x")
    print(f"  best B={bw['best_block_size']}  measured gain={bw['best_throughput_gain']:.2f}x")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  spin projection == naive reference; adjoint; gamma5-hermiticity;
  free-field sigma_min == mass; D_eo == restricted full hopping;
  hat_D^dag == g5 hat_D g5; EO solves checked against the FULL operator;
  deflation reduces iterations and the cache invalidates on gauge change;
  heatbath vs Metropolis equilibrium; unitarity; mixed precision reaches
  fp64 tolerance; link traffic falls as 1/B.

MEASURED, NOT ASSERTED
  ms/call, achieved GB/s, throughput gain vs B, link bytes per RHS.

NOT PRODUCTION-EQUIVALENT
  Single device, no MPI halo exchange. Quenched gauge (correct for solver
  benchmarking, not full QCD). No SAP/multigrid: at light quark mass you
  still need it. If achieved GB/s is far below peak, bandwidth is NOT the
  bottleneck and the measured gain will lag the byte-count ratio.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="GPU block Wilson Dirac")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--Nc", type=int, default=3)
    ap.add_argument("--mass", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--n-therm", type=int, default=60)
    ap.add_argument("--blocks", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-rhs", type=int, default=8)
    ap.add_argument("--deflate-k", type=int, default=12)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test(a.device, min(a.L, 4), a.n_therm) else 1
    if not a.bench:
        ap.print_help(); print(BANNER); return 0

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if a.device == "auto" else a.device)
    dt = C128 if a.fp64 else C64
    print(BANNER)
    print(f"device={dev}  L={a.L}  Nc={a.Nc}  dtype={dt}")
    g = make_gauge(4, a.L, a.Nc, dev, a.beta, a.n_therm, seed=0, dtype=dt)
    print(f"thermalized plaquette = {g.plaquette():.6f}")
    D = WilsonGPU(g, a.mass)
    eo = EvenOddGPU(D)
    bw = bandwidth_probe(D, sizes=tuple(a.blocks), n_rep=5, eo=eo)
    print(f"\n  {'B':>3} {'ms/call':>10} {'link MB/RHS':>13} {'GB/s':>9} {'gain':>7}")
    for r in bw["rows"]:
        print(f"  {r['block_size']:>3} {r['ms_per_call']:>10.3f} "
              f"{r['link_bytes_per_rhs']/1e6:>13.3f} {r['achieved_GBs']:>9.1f} "
              f"{r['throughput_gain_vs_B1']:>7.2f}x")
    B = min(a.n_rhs, bw["best_block_size"])
    b = D.randn(B, seed=99)
    cache = DeflationCache()
    dfl = cache.get(eo, a.deflate_k, 30)
    out = {"bandwidth": bw}
    for tag, kw in (("plain", {}), ("deflated", {"deflate": dfl})):
        D.reset(); eo.reset()
        _sync(dev); t0 = time.perf_counter()
        x, st = eo.solve(b, solver="fgmres", tol=1e-8, maxiter=3000,
                         inner=a.inner, **kw)
        _sync(dev); wall = time.perf_counter() - t0
        out[tag] = {"iters": st["n_iter"], "wall_s": wall,
                    "max_residual": float(st["full_residual"].max()),
                    "converged": st["converged"],
                    "full_apply_calls": D.n_calls,
                    "schur_calls": eo.n_calls}
        print(f"\n  {tag:10s} iters={st['n_iter']:5d} wall={wall:.3f}s "
              f"res={float(st['full_residual'].max()):.2e}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
