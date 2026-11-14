#!/usr/bin/env python3
"""
=============================================================================
ECM STAGE 2 (Baby-Step Giant-Step + batched gcd)
+ integration bridge for sci_factor_v2.py
+ GMP-ECM auto-delegation
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT THIS ADDS OVER STAGE-1-ONLY ECM
  Stage 1 finds a factor p when the group order #E(F_p) is B1-smooth.
  Stage 2 extends coverage to group orders that are B1-smooth EXCEPT
  for ONE prime q in (B1, B2]. This is the standard continuation and
  is what makes ECM practical for 40-60 digit factors.

  Method implemented here:
    * Montgomery x-only arithmetic (correct xz ladder)
    * Suyama parametrization (sigma -> curve + point)
    * Stage 2 via Baby-Step Giant-Step with a primorial wheel:
        q = j*D +/- d,  gcd(d, W) = 1,  d <= D/2
      Since x(P) = x(-P) on Montgomery curves, [q]Q = O  iff
      x([jD]Q) == x([d]Q). Collisions are detected by accumulating
      the cross-products (Xg*Zb - Xb*Zg) mod N and taking ONE gcd
      per chunk (batched gcd), with backtracking when gcd == N.
    * This is the "standard continuation with BSGS + batched gcd".
      It is NOT the full subquadratic polynomial-evaluation
      continuation of GMP-ECM. For serious work, GMP-ECM is used
      automatically when the `ecm` binary is available.

HONEST SCOPE — RSA-300/350
  * ECM finds FACTORS up to ~60-70 digits (with enough curves, in
    GMP-ECM/GPU). It does NOT factor a balanced semiprime whose two
    factors are each 150+ digits.
  * If "RSA-300/350" means 300-350 DECIMAL DIGITS (~1000-1160 bits),
    neither ECM nor any publicly demonstrated GNFS run can factor it.
    RSA-250 (829 bits) required ~2700 core-years; 1000+ bits is
    ~1e6-1e8 times harder.
  * If "RSA-300/350" means 300-350 BITS (~90-105 digits), GNFS on a
    cluster is feasible (hours-days). ECM helps only if one factor
    is small (<= ~40-50 digits).
  * This module REFUSES demo factorization above MAX_DEMO_BITS and
    prints an honest feasibility report instead.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("ecm_stage2")

VERSION = "1.0.0"
MAX_DEMO_BITS = 220          # Python ECM demo guard; use GMP-ECM/GNFS beyond
WHEEL_W = 210                # 2*3*5*7
WHEEL_PHI = 48               # phi(210)
GIANT_D = 210                # giant-step stride (multiple of WHEEL_W)

BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  ECM STAGE 2 (BSGS + batched gcd) — finds FACTORS, not balanced RSA  ║
║  Auto-delegates to GMP-ECM when available. Honest about RSA-300/350. ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# §0. Primes / primality
# =============================================================================

def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, int(math.isqrt(n)) + 1):
        if s[i]:
            s[i*i::i] = b"\x00" * (((n - i*i) // i) + 1)
    return [i for i in range(n + 1) if s[i]]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small:
        return True
    for p in small:
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = small if n < 3_317_044_064_679_887_385_961_981 else \
        (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# =============================================================================
# §1. Montgomery x-only arithmetic (correct)
# =============================================================================

def xadd(xp: int, zp: int, xq: int, zq: int,
         xm: int, zm: int, n: int) -> Tuple[int, int]:
    """Differential addition: given P, Q, P-Q, return P+Q (x-only)."""
    u = (xp - zp) * (xq + zq) % n
    v = (xp + zp) * (xq - zq) % n
    return (zm * (u + v) * (u + v) % n,
            xm * (u - v) * (u - v) % n)


def xdbl(x: int, z: int, a24: int, n: int) -> Tuple[int, int]:
    a = (x + z) * (x + z) % n
    b = (x - z) * (x - z) % n
    c = (a - b) % n
    return (a * b % n, c * (b + a24 * c) % n)


def ladder(k: int, x: int, z: int, a24: int, n: int) -> Tuple[int, int]:
    """Montgomery ladder: [k](x:z)."""
    if k == 0:
        return (1, 0)
    x1, z1 = x, z
    x2, z2 = xdbl(x, z, a24, n)
    for bit in bin(k)[3:]:
        if bit == "1":
            x1, z1 = xadd(x1, z1, x2, z2, x, z, n)
            x2, z2 = xdbl(x2, z2, a24, n)
        else:
            x2, z2 = xadd(x2, z2, x1, z1, x, z, n)
            x1, z1 = xdbl(x1, z1, a24, n)
    return x1, z1


def suyama_curve(n: int, sigma: int
                 ) -> Tuple[Optional[Tuple[int, int, int]], Optional[int]]:
    """
    Suyama parametrization. Returns ((x0, z0, a24), None) or (None, factor).
    """
    u = (sigma * sigma - 5) % n
    v = (4 * sigma) % n
    x0 = pow(u, 3, n)
    z0 = pow(v, 3, n)
    num = pow(v - u, 3, n) * (3 * u + v) % n
    den = 4 * x0 * v % n
    g = math.gcd(den, n)
    if 1 < g < n:
        return None, g
    if g == n or den == 0:
        return None, None
    try:
        inv_den = pow(den, -1, n)
        inv4 = pow(4, -1, n)
    except ValueError:
        return None, None
    A = (num * inv_den - 2) % n
    a24 = (A + 2) * inv4 % n
    return (x0, z0, a24), None


# =============================================================================
# §2. ECM Stage 1
# =============================================================================

def ecm_stage1(n: int, B1: int, sigma: int
               ) -> Tuple[Optional[Tuple[int, int, int]], Optional[int]]:
    """
    Returns ((Qx, Qz, a24), None) on success (no factor yet),
    or (None, factor) if a factor is found, or (None, None) on bad curve.
    """
    curve, f = suyama_curve(n, sigma)
    if f is not None:
        return None, f
    if curve is None:
        return None, None
    x, z, a24 = curve
    for p in sieve_primes(B1):
        pe = p
        while pe * p <= B1:
            pe *= p
        x, z = ladder(pe, x, z, a24, n)
        g = math.gcd(z, n)
        if 1 < g < n:
            return None, g
        if g == n:
            return None, None
    return (x, z, a24), None


# =============================================================================
# §3. ECM Stage 2 — BSGS + batched gcd
# =============================================================================

def _baby_steps(Qx: int, Qz: int, a24: int, n: int,
                D: int, W: int) -> List[Tuple[int, int, int]]:
    """
    Baby steps [d]Q for d in [1, D/2] with gcd(d, W) = 1.
    Built sequentially with 1 xadd per step (using [d-2]Q as difference).
    Returns list of (d, Xd, Zd).
    """
    half = D // 2
    out: List[Tuple[int, int, int]] = []
    # P_{d-2}, P_{d-1}
    prev2 = (1, 0)          # identity
    prev1 = (Qx, Qz)        # [1]Q
    if math.gcd(1, W) == 1:
        out.append((1, Qx, Qz))
    for d in range(2, half + 1):
        if d == 2:
            cur = xdbl(Qx, Qz, a24, n)
        else:
            cur = xadd(prev1[0], prev1[1], Qx, Qz,
                       prev2[0], prev2[1], n)
        prev2, prev1 = prev1, cur
        if math.gcd(d, W) == 1:
            out.append((d, cur[0], cur[1]))
    return out


def ecm_stage2_bsgs(n: int, Qx: int, Qz: int, a24: int,
                    B1: int, B2: int,
                    D: int = GIANT_D, W: int = WHEEL_W,
                    gcd_chunk: int = 2048) -> Optional[int]:
    """
    Standard continuation via Baby-Step Giant-Step.

    Covers primes q in (B1, B2] with q = j*D +/- d, gcd(d,W)=1, d<=D/2.
    Since x(P)=x(-P), [q]Q = O  iff  x([jD]Q) == x([d]Q).
    Collisions detected by batched cross-product gcd with backtracking.

    NOTE: This is O((B2-B1)/D * #babies) cheap modular mults with ONE gcd
    per chunk. It is NOT GMP-ECM's subquadratic polynomial-evaluation
    continuation. For B2 > ~1e7, use GMP-ECM (auto-delegated below).
    """
    if B2 <= B1:
        return None
    babies = _baby_steps(Qx, Qz, a24, n, D, W)
    if not babies:
        return None

    j_lo = max(1, B1 // D)
    j_hi = (B2 + D - 1) // D + 1

    # G_j = [j*D]Q, built sequentially
    G1 = ladder(D, Qx, Qz, a24, n)
    G_prev2 = (1, 0)
    G_prev1 = G1
    # advance to j_lo
    if j_lo == 1:
        Gj = G1
    else:
        Gj = ladder(j_lo * D, Qx, Qz, a24, n)
        G_prev1 = ladder((j_lo - 1) * D, Qx, Qz, a24, n) if j_lo >= 2 else (1, 0)
        G_prev2 = ladder((j_lo - 2) * D, Qx, Qz, a24, n) if j_lo >= 3 else (1, 0)

    prod = 1
    chunk: List[int] = []

    def flush() -> Optional[int]:
        nonlocal prod, chunk
        if not chunk:
            return None
        g = math.gcd(prod, n)
        if 1 < g < n:
            chunk = []; prod = 1
            return g
        if g == n:
            # backtrack: find the individual diff that gives a proper factor
            for dv in chunk:
                gg = math.gcd(dv, n)
                if 1 < gg < n:
                    chunk = []; prod = 1
                    return gg
        chunk = []; prod = 1
        return None

    for j in range(j_lo, j_hi + 1):
        if j >= j_lo + 2:
            Gj = xadd(G_prev1[0], G_prev1[1], G1[0], G1[1],
                      G_prev2[0], G_prev2[1], n)
            G_prev2, G_prev1 = G_prev1, Gj
        elif j == j_lo + 1 and j_lo >= 1:
            Gj = xadd(G_prev1[0], G_prev1[1], G1[0], G1[1],
                      G_prev2[0], G_prev2[1], n) if j_lo >= 2 else xdbl(G1[0], G1[1], a24, n)
            G_prev2, G_prev1 = G_prev1, Gj

        Xg, Zg = Gj
        if Zg == 0:
            g = math.gcd(Xg, n)
            if 1 < g < n:
                return g
            continue
        for (_d, Xb, Zb) in babies:
            dv = (Xg * Zb - Xb * Zg) % n
            if dv == 0:
                continue
            prod = prod * dv % n
            chunk.append(dv)
            if len(chunk) >= gcd_chunk:
                f = flush()
                if f is not None:
                    return f
                if prod == 0:
                    break
        if prod == 0:
            f = flush()
            if f is not None:
                return f
    return flush()


# =============================================================================
# §4. Full ECM (stage 1 + stage 2), multi-curve
# =============================================================================

@dataclass
class ECMConfig:
    B1: int = 11_000
    B2: int = 1_870_000        # ~ t20-ish default (modest for Python)
    curves: int = 20
    seed: int = 12345
    use_gmp: bool = True
    gmp_timeout: float = 120.0


# GMP-ECM t-level presets (B1, recommended B2, curves) — for delegation
T_LEVELS = {
    20: (11_000, 1_870_000, 74),
    25: (50_000, 12_750_000, 221),
    30: (250_000, 128_992_510, 430),
    35: (1_000_000, 1_045_566_240, 904),
    40: (3_000_000, 5_706_890_290, 2350),
    45: (11_000_000, 31_770_803_020, 4480),
    50: (43_000_000, 201_633_903_050, 7553),
    55: (110_000_000, 776_278_396_540, 17769),
    60: (260_000_000, 3_011_414_953_060, 42017),
}


def gmp_ecm_available() -> bool:
    return shutil.which("ecm") is not None


def gmp_ecm(n: int, B1: int, curves: int, B2: Optional[int] = None,
            timeout: float = 300.0) -> Optional[int]:
    """Delegate to GMP-ECM (real, fast). Returns a factor or None."""
    if not gmp_ecm_available():
        return None
    cmd = ["ecm", "-q", "-c", str(curves)]
    if B2:
        cmd += ["-B2", str(B2)]
    cmd += [str(B1)]
    try:
        r = subprocess.run(cmd, input=f"{n}\n", capture_output=True,
                           text=True, timeout=timeout)
        for line in r.stdout.splitlines():
            for tok in line.replace(",", " ").split():
                if tok.isdigit():
                    f = int(tok)
                    if 1 < f < n and n % f == 0 and is_prime(f):
                        return f
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return None


def ecm(n: int, cfg: ECMConfig) -> Dict[str, Any]:
    """
    Full ECM: stage 1 + stage 2 BSGS, multi-curve.
    Auto-delegates to GMP-ECM when available (much faster).
    """
    t0 = time.time()
    if n % 2 == 0:
        return {"factor": 2, "method": "trivial", "elapsed_s": 0.0}
    if is_prime(n):
        return {"factor": None, "method": "prime_input", "elapsed_s": 0.0}

    if cfg.use_gmp and gmp_ecm_available():
        f = gmp_ecm(n, cfg.B1, cfg.curves, cfg.B2, cfg.gmp_timeout)
        if f:
            return {"factor": f, "method": "gmp-ecm",
                    "elapsed_s": time.time() - t0}

    rng = random.Random(cfg.seed)
    for c in range(cfg.curves):
        sigma = rng.randrange(6, max(7, n - 1))
        Q, f = ecm_stage1(n, cfg.B1, sigma)
        if f:
            return {"factor": f, "method": "python-ecm-stage1",
                    "curve": c, "sigma": sigma,
                    "elapsed_s": time.time() - t0}
        if Q is None:
            continue
        Qx, Qz, a24 = Q
        f = ecm_stage2_bsgs(n, Qx, Qz, a24, cfg.B1, cfg.B2)
        if f:
            return {"factor": f, "method": "python-ecm-stage2-bsgs",
                    "curve": c, "sigma": sigma,
                    "elapsed_s": time.time() - t0}
    return {"factor": None, "method": "python-ecm",
            "curves_tried": cfg.curves, "elapsed_s": time.time() - t0}


# =============================================================================
# §5. Integration with sci_factor_v2.py
# =============================================================================

def install_into_sci_factor_v2(verbose: bool = True) -> bool:
    """
    Monkey-patch sci_factor_v2.Factorizer to use this ECM (stage 1+2).
    Returns True if patched, False if module not importable.
    """
    try:
        import sci_factor_v2 as sf
    except ImportError:
        if verbose:
            log.warning("sci_factor_v2 not importable; skipping patch")
        return False

    orig_ecm = getattr(sf.Factorizer, "ecm", None)

    def ecm_patched(self, N: int, B1: int = 11_000, curves: int = 20,
                    timeout: float = 60.0):
        cfg = ECMConfig(B1=B1, B2=max(B1 * 100, 1_870_000),
                        curves=curves, use_gmp=True, gmp_timeout=timeout)
        res = ecm(N, cfg)
        return res.get("factor")

    sf.Factorizer.ecm = ecm_patched
    sf.Factorizer.ecm_stage2 = ecm_patched
    if verbose:
        log.info("Patched sci_factor_v2.Factorizer.ecm -> stage1+stage2 ECM")
    return True


# =============================================================================
# §6. Honest RSA feasibility report
# =============================================================================

def rsa_feasibility(size: int, unit: str = "bits") -> Dict[str, Any]:
    """
    Honest feasibility for a balanced semiprime of the given size.
    unit: 'bits' or 'digits' (decimal).
    """
    bits = size if unit == "bits" else int(size * math.log2(10))
    digits = int(bits * math.log10(2))
    factor_digits = digits // 2
    lnN = bits * math.log(2)
    c = (64.0 / 9.0) ** (1.0 / 3.0)
    lnL = c * (lnN ** (1/3)) * (math.log(lnN) ** (2/3))
    log10_ops = lnL / math.log(10)

    ecm_ok = factor_digits <= 60
    gnfs_note = (
        "feasible on a cluster (hours-days)" if bits <= 512 else
        "large cluster / months" if bits <= 768 else
        "beyond public record (RSA-250 = 829 bits, ~2700 core-years)"
        if bits <= 1024 else
        "far beyond any public capability (~1e6-1e8x harder than RSA-250)"
    )
    return {
        "input": f"{size} {unit}",
        "bits": bits, "decimal_digits": digits,
        "balanced_factor_digits": factor_digits,
        "ecm_can_find_factor": ecm_ok,
        "ecm_note": ("ECM finds factors <= ~60-70 digits with enough curves; "
                     "a balanced semiprime with 150+ digit factors is out of reach"),
        "gnfs_log10_ops_estimate": round(log10_ops, 1),
        "gnfs_feasibility": gnfs_note,
        "recommendation": (
            "use GMP-ECM (GPU) for small factors; CADO-NFS/msieve for GNFS"
        ),
    }


# =============================================================================
# §7. Benchmark: stage1-only vs stage1+stage2
# =============================================================================

def benchmark(n_curves: int = 8, seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)
    primes = sieve_primes(200_000)
    rows = []
    for _ in range(6):
        p = rng.choice(primes[1000:5000])
        q = rng.choice(primes[5000:20000])
        N = p * q
        cfg1 = ECMConfig(B1=2000, B2=2000, curves=n_curves,
                         seed=seed, use_gmp=False)
        cfg2 = ECMConfig(B1=2000, B2=200_000, curves=n_curves,
                         seed=seed, use_gmp=False)
        r1 = ecm(N, cfg1)
        r2 = ecm(N, cfg2)
        rows.append({
            "N": N, "p": p, "q": q,
            "stage1_found": r1["factor"] is not None,
            "stage12_found": r2["factor"] is not None,
            "stage1_s": round(r1["elapsed_s"], 3),
            "stage12_s": round(r2["elapsed_s"], 3),
        })
    s1 = sum(r["stage1_found"] for r in rows)
    s12 = sum(r["stage12_found"] for r in rows)
    return {
        "rows": rows,
        "stage1_hits": s1, "stage12_hits": s12,
        "note": ("stage 2 covers group orders smooth except one prime in "
                 "(B1,B2]; improvement depends on the factor, not guaranteed"),
    }


# =============================================================================
# §8. Self-test
# =============================================================================

def self_test() -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: ladder consistency on a small prime field
    n = 1009
    curve, f = suyama_curve(n, 11)
    rep("T1 Suyama on prime gives curve", curve is not None and f is None)
    if curve:
        x, z, a24 = curve
        P2 = xdbl(x, z, a24, n)
        P3 = xadd(P2[0], P2[1], x, z, x, z, n)
        P3b = ladder(3, x, z, a24, n)
        rep("T1 [3]P via ladder == via add",
            (P3[0] * P3b[1] - P3b[0] * P3[1]) % n == 0)

    # T2: ECM finds a known factor
    p, q = 1_000_003, 1_000_033
    N = p * q
    res = ecm(N, ECMConfig(B1=5000, B2=500_000, curves=15, use_gmp=False))
    rep("T2 ECM finds factor of 13-digit semiprime",
        res["factor"] in (p, q), f"factor={res['factor']}")

    # T3: stage 2 runs and returns proper factor or None
    Q, f = ecm_stage1(N, 2000, 17)
    if Q:
        f2 = ecm_stage2_bsgs(N, Q[0], Q[1], Q[2], 2000, 100_000)
        rep("T3 stage2 returns proper factor or None",
            f2 is None or (1 < f2 < N and N % f2 == 0), f"f2={f2}")
    else:
        rep("T3 stage1 already found factor", f in (p, q), f"f={f}")

    # T4: GMP-ECM availability reported honestly
    rep("T4 GMP-ECM availability reported",
        isinstance(gmp_ecm_available(), bool),
        f"available={gmp_ecm_available()}")

    # T5: RSA feasibility honest
    r300b = rsa_feasibility(300, "bits")
    r300d = rsa_feasibility(300, "digits")
    rep("T5 300-bit: GNFS feasible, ECM only if small factor",
        "cluster" in r300b["gnfs_feasibility"], r300b["gnfs_feasibility"])
    rep("T5 300-digit: beyond public capability",
        "beyond" in r300d["gnfs_feasibility"], r300d["gnfs_feasibility"])
    rep("T5 300-digit ECM cannot (balanced)",
        r300d["ecm_can_find_factor"] is False)

    # T6: integration patch
    patched = install_into_sci_factor_v2(verbose=False)
    rep("T6 sci_factor_v2 patch attempted",
        isinstance(patched, bool), f"patched={patched}")

    # T7: benchmark runs
    b = benchmark(n_curves=4)
    rep("T7 benchmark runs", "rows" in b,
        f"stage1={b['stage1_hits']}, stage1+2={b['stage12_hits']}")

    print("\n" + "=" * 72)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 72)
    print("""
HONEST REMINDERS
  * ECM finds FACTORS up to ~60-70 digits (GMP-ECM/GPU, many curves).
  * A balanced semiprime with 150+ digit factors is NOT an ECM target.
  * RSA-300/350 DECIMAL DIGITS (~1000-1160 bits): not factorable by any
    public method. RSA-300/350 BITS (~90-105 digits): GNFS on a cluster.
  * This Python ECM is educational; GMP-ECM is auto-used when present.
""")
    return ok


# =============================================================================
# §9. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ECM stage 2 + sci_factor_v2 bridge")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--factor", type=str, default=None)
    ap.add_argument("--B1", type=int, default=11_000)
    ap.add_argument("--B2", type=int, default=1_870_000)
    ap.add_argument("--curves", type=int, default=20)
    ap.add_argument("--no-gmp", action="store_true")
    ap.add_argument("--rsa-check", type=int, default=None)
    ap.add_argument("--unit", choices=["bits", "digits"], default="bits")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--install", action="store_true",
                    help="patch sci_factor_v2 and exit")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test() else 1
    if a.install:
        ok = install_into_sci_factor_v2()
        print("patched" if ok else "sci_factor_v2 not found")
        return 0 if ok else 1
    if a.rsa_check is not None:
        print(json.dumps(rsa_feasibility(a.rsa_check, a.unit), indent=2))
        return 0
    if a.benchmark:
        print(json.dumps(benchmark(), indent=2, default=str))
        return 0
    if a.factor:
        N = int(a.factor)
        if N.bit_length() > MAX_DEMO_BITS and not gmp_ecm_available():
            print(f"N has {N.bit_length()} bits > demo guard {MAX_DEMO_BITS} "
                  f"and GMP-ECM not available.")
            print(json.dumps(rsa_feasibility(N.bit_length(), "bits"), indent=2))
            return 1
        cfg = ECMConfig(B1=a.B1, B2=a.B2, curves=a.curves,
                        use_gmp=not a.no_gmp)
        res = ecm(N, cfg)
        print(json.dumps(res, indent=2, default=str))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=2, default=str)
        return 0
    ap.print_help()
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
