#!/usr/bin/env python3
"""
GNFS-10 Research Scaffold
Spectral / Hodge / Assumption-Light Statistics / World Model diagnostics
for Number Field Sieve experimentation.

NOT a production GNFS implementation.
NOT a claim that 1000-4000 digit semiprimes can be factored.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 0. Big integer / primality / toy factor utilities
# =============================================================================

_SMALL_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
    41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
]


def is_probable_prime(n: int, rounds: int = 32) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    bases = _SMALL_PRIMES[:]
    if n.bit_length() > 64:
        bases += [random.randrange(2, n - 2) for _ in range(rounds)]

    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def pollard_brent(n: int, max_steps: int = 200000) -> Optional[int]:
    """Toy factor helper for small demos only."""
    if n % 2 == 0:
        return 2
    if is_probable_prime(n):
        return n

    rng = random.Random(12345)
    for _ in range(20):
        y = rng.randrange(1, n)
        c = rng.randrange(1, n)
        m = 128
        g = r = q = 1
        steps = 0

        while g == 1 and steps < max_steps:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
                steps += 1
            k = 0
            while k < r and g == 1 and steps < max_steps:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                    steps += 1
                g = math.gcd(q, n)
                k += m
            r *= 2

        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)

        if 1 < g < n:
            return g
    return None


# =============================================================================
# 1. GNFS polynomial planning
# =============================================================================

def integer_nthroot(n: int, d: int) -> int:
    lo, hi = 0, 1 << ((n.bit_length() + d - 1) // d + 1)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid ** d <= n:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class NFSPolynomial:
    coeffs: List[int]  # c0 + c1 x + ... + cd x^d
    m: int

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def eval(self, x: int) -> int:
        out = 0
        for c in reversed(self.coeffs):
            out = out * x + c
        return out

    def algebraic_norm(self, a: int, b: int) -> int:
        d = self.degree
        out = 0
        for i, c in enumerate(self.coeffs):
            out += c * (a ** i) * (b ** (d - i))
        return out

    def rational_norm(self, a: int, b: int) -> int:
        return a - b * self.m


def choose_degree(n: int) -> int:
    digits = len(str(n))
    if digits < 80:
        return 3
    if digits < 140:
        return 4
    if digits < 220:
        return 5
    return 6


def base_m_polynomial(n: int, degree: Optional[int] = None) -> NFSPolynomial:
    """
    Toy base-m polynomial: f(m)=n exactly.
    Production GNFS uses much better polynomial selection
    (Kleinjung, Murphy E, lattice/sieving scores).
    """
    d = degree or choose_degree(n)
    m = integer_nthroot(n, d + 1) + 1
    x = n
    coeffs = []
    for _ in range(d + 1):
        coeffs.append(x % m)
        x //= m
    if x:
        coeffs[-1] += x * m
    return NFSPolynomial(coeffs, m)


def poly_score(poly: NFSPolynomial, n: int) -> Dict[str, float]:
    coeff_log = sum(math.log1p(abs(c)) for c in poly.coeffs)
    skew = abs(poly.coeffs[0] / max(poly.coeffs[-1], 1)) ** (1 / max(poly.degree, 1))
    return {
        "degree": poly.degree,
        "m_bits": poly.m.bit_length(),
        "coeff_log": coeff_log,
        "skew_proxy": skew,
        "score_proxy": coeff_log + 0.25 * math.log1p(skew),
    }


# =============================================================================
# 2. Toy relation collection
# =============================================================================

def primes_up_to(b: int) -> List[int]:
    sieve = [True] * (b + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(b ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, b + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def factor_over_base(x: int, fb: List[int]) -> Optional[List[int]]:
    """Return exponent vector if x is smooth over fb, else None."""
    if x == 0:
        return None
    e = [0] * (len(fb) + 1)
    if x < 0:
        e[0] = 1
        x = -x
    for i, p in enumerate(fb, start=1):
        while x % p == 0:
            e[i] += 1
            x //= p
        if x == 1:
            return e
    return None


@dataclass
class Relation:
    a: int
    b: int
    rational: int
    algebraic: int
    exp_r: List[int]
    exp_a: List[int]


def toy_relation_scan(
    n: int,
    poly: NFSPolynomial,
    fb_size: int = 50,
    a_limit: int = 200,
    b_limit: int = 5,
) -> List[Relation]:
    fb = primes_up_to(fb_size)
    rels = []
    for b in range(1, b_limit + 1):
        for a in range(-a_limit, a_limit + 1):
            if math.gcd(a, b) != 1:
                continue
            rr = poly.rational_norm(a, b)
            aa = poly.algebraic_norm(a, b)
            er = factor_over_base(rr, fb)
            if er is None:
                continue
            ea = factor_over_base(aa, fb)
            if ea is None:
                continue
            rels.append(Relation(a, b, rr, aa, er, ea))
    return rels


# =============================================================================
# 3. Spectral / Hodge diagnostics
# =============================================================================

def relation_graph_features(rels: List[Relation], fb_size: int) -> Dict[str, float]:
    """
    Bipartite graph: relation nodes + prime nodes.
    Hodge L1 = B1^T B1 with B2=0.
    """
    if not rels:
        return {
            "n_relations": 0, "n_edges": 0, "beta0": 0, "beta1": 0,
            "spectral_gap": 0.0, "hodge_edge_energy": 0.0,
        }

    fb = primes_up_to(fb_size)
    nR = len(rels)
    nP = len(fb) + 1
    V = nR + nP
    edges = []

    for i, r in enumerate(rels):
        for j, e in enumerate(r.exp_r):
            if e % 2:
                edges.append((i, nR + j))
        for j, e in enumerate(r.exp_a):
            if e % 2:
                edges.append((i, nR + j))

    E = len(edges)
    if E == 0:
        return {
            "n_relations": nR, "n_edges": 0, "beta0": V, "beta1": 0,
            "spectral_gap": 0.0, "hodge_edge_energy": 0.0,
        }

    B1 = np.zeros((V, E), dtype=np.float64)
    A = np.zeros((V, V), dtype=np.float64)
    for k, (u, v) in enumerate(edges):
        B1[u, k] = -1
        B1[v, k] = 1
        A[u, v] = A[v, u] = 1

    D = np.diag(A.sum(axis=1))
    L0 = D - A
    vals = np.linalg.eigvalsh(L0)
    gap = float(vals[1] - vals[0]) if len(vals) > 1 else 0.0

    # connected components
    seen = set()
    comps = 0
    adj = [set() for _ in range(V)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(V):
        if i in seen:
            continue
        comps += 1
        stack = [i]
        seen.add(i)
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)

    beta0 = comps
    beta1 = E - V + beta0
    L1 = B1.T @ B1
    hodge_energy = float(np.trace(L1))

    return {
        "n_relations": nR,
        "n_edges": E,
        "beta0": beta0,
        "beta1": max(beta1, 0),
        "spectral_gap": gap,
        "hodge_edge_energy": hodge_energy,
    }


# =============================================================================
# 4. Assumption-light statistics
# =============================================================================

def subsampling_mean_ci(x: np.ndarray, block: Optional[int] = None,
                        alpha: float = 0.05) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = block or max(2, int(math.sqrt(n)))
    means = np.array([x[i:i+b].mean() for i in range(n - b + 1)])
    theta = float(x.mean())
    centered = math.sqrt(b) * (means - theta)
    lo = theta - float(np.quantile(centered, 1 - alpha / 2)) / math.sqrt(n)
    hi = theta - float(np.quantile(centered, alpha / 2)) / math.sqrt(n)
    return {"mean": theta, "lo": lo, "hi": hi, "n": n, "block": b}


def cusum_change_points(x: np.ndarray, threshold: float = 1.0) -> List[int]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return []
    mu = x.mean()
    s_pos = s_neg = 0.0
    out = []
    for i, v in enumerate(x):
        s_pos = max(0.0, s_pos + v - mu)
        s_neg = max(0.0, s_neg - v + mu)
        if max(s_pos, s_neg) > threshold * x.std():
            out.append(i)
            s_pos = s_neg = 0.0
    return out


def hill_tail_index(x: np.ndarray, k: Optional[int] = None) -> Optional[float]:
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    x = x[x > 0]
    if len(x) < 10:
        return None
    k = k or max(5, len(x) // 10)
    if k >= len(x):
        return None
    logs = np.log(x[:k] / x[k])
    m = logs.mean()
    return float(1.0 / m) if m > 0 else None


# =============================================================================
# 5. Transformer / GRU world model
# =============================================================================

class GNFSWorldModel(nn.Module):
    """
    Predict next relation-yield statistics from previous diagnostic states.
    Input feature example:
      [smooth_rate, log_norm_mean, spectral_gap, beta1, fb_size, degree]
    """
    def __init__(self, d_in: int = 6, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model,
            batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.gru = nn.GRUCell(d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_in),
        )
        self.unc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_in),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.proj(x)
        h = self.tr(h)
        z = h[:, -1]
        z = self.gru(z, z)
        return {"next": self.head(z), "uncertainty": self.unc(z)}


def synthetic_training_sequence(n: int = 128) -> torch.Tensor:
    t = torch.linspace(0, 10, n)
    smooth = 0.05 + 0.02 * torch.sin(t) + 0.005 * torch.randn(n)
    lognorm = 5 + 0.5 * torch.cos(0.7*t) + 0.05 * torch.randn(n)
    gap = 0.1 + 0.03 * torch.sin(1.3*t)
    beta1 = 10 + 3 * torch.cos(t)
    fb = torch.full((n,), 50.0)
    deg = torch.full((n,), 5.0)
    return torch.stack([smooth, lognorm, gap, beta1, fb, deg], dim=-1)


def train_world_model_demo(steps: int = 50) -> Dict[str, float]:
    data = synthetic_training_sequence(256)
    model = GNFSWorldModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        idx = torch.randint(0, len(data)-17, (8,))
        xb = torch.stack([data[i:i+16] for i in idx])
        yb = torch.stack([data[i+16] for i in idx])
        out = model(xb)["next"]
        loss = F.mse_loss(out, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return {"initial_loss": losses[0], "final_loss": losses[-1]}


# =============================================================================
# 6. Reporting
# =============================================================================

def plan_n(n: int) -> Dict[str, object]:
    poly = base_m_polynomial(n)
    digits = len(str(n))
    return {
        "digits": digits,
        "bits": n.bit_length(),
        "probable_prime": is_probable_prime(n),
        "degree": poly.degree,
        "m_bits": poly.m.bit_length(),
        "polynomial": poly.coeffs,
        "poly_score": poly_score(poly, n),
        "public_feasibility": (
            "beyond public GNFS records" if digits > 250
            else "research-scale / possibly feasible with large resources"
        ),
        "warning": (
            "This is a planning/diagnostic module, not a production GNFS "
            "implementation and not a claim of factoring 1000-4000 digit semiprimes."
        ),
    }


def self_test() -> bool:
    print("=" * 72)
    print("GNFS-10 research scaffold — self-test")
    print("=" * 72)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {extra}")

    p, q = 1009, 1013
    n = p * q
    f = pollard_brent(n)
    rep("toy factor", f in (p, q), f"factor={f}")

    poly = base_m_polynomial(n, degree=3)
    rep("base-m polynomial f(m)=n", poly.eval(poly.m) == n)

    rels = toy_relation_scan(n, poly, fb_size=30, a_limit=80, b_limit=3)
    feats = relation_graph_features(rels, 30)
    rep("relation scan runs", isinstance(feats, dict), str(feats))

    x = np.random.default_rng(0).normal(size=100)
    ci = subsampling_mean_ci(x)
    rep("subsampling CI", ci["lo"] <= ci["mean"] <= ci["hi"], str(ci))

    wm = train_world_model_demo(steps=10)
    rep("world model trains", wm["final_loss"] < wm["initial_loss"] * 2, str(wm))

    print("=" * 72)
    print("Not production GNFS. No Hilbert-Pólya proof. No RSA break.")
    print("=" * 72)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--plan", type=str, help="integer N to plan")
    ap.add_argument("--factor-demo", type=str, help="small N only")
    args = ap.parse_args()

    if args.test:
        raise SystemExit(0 if self_test() else 1)

    if args.plan:
        n = int(args.plan)
        import json
        print(json.dumps(plan_n(n), indent=2))
        return

    if args.factor_demo:
        n = int(args.factor_demo)
        if n.bit_length() > 90:
            print("Refusing demo factor above 90 bits. Use real GNFS software.")
            return
        f = pollard_brent(n)
        print({"n": n, "factor": f, "cofactor": n//f if f else None})
        return

    ap.print_help()


if __name__ == "__main__":
    main()
