#!/usr/bin/env python3
"""
=============================================================================
BUFFER ACCUMULATOR — fix silent overwrite collisions between bridges
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

VERIFIED COLLISIONS (from one_core_v3-6.py)
-------------------------------------------
  DNS _ext_fx/_ext_fy/_ext_fz : CahnHilliardDNSBridge + SeismicDNSBridge
  FH  _ext_mask               : LangevinFHBridge      + CahnHilliardFHBridge
Both pairs use OVERWRITE semantics. Whichever syncs LAST wins; the other
contribution is silently discarded. SeismicDNSBridge's own docstring
acknowledges this ("combine the contributions externally") but nothing
enforces or detects it.

WHY ONE RULE IS NOT ENOUGH
--------------------------
  forces / volumetric sources  -> SUM is correct
      _ext_fx/fy/fz [N/m^3], _ext_q [W/m^3], _ext_mdot*
  material properties          -> SUM IS UNPHYSICAL
      _ext_rho_ch/_ext_rho_eff, _ext_nu_ch/_ext_nu_eff
      Two densities do not add. Requires an explicit PRIORITY; equal
      priority RAISES rather than silently picking one.
  signals / masks              -> MAX or MEAN, declared per buffer

HONEST SCOPE
  ❌ Wall-mode PyrolysisDNSBridge writes bc_objects[face], not a solver
     buffer. Not interceptable here; the hub enforces ONE owner per face.
  ❌ Unknown buffers default to REQUIRE — refuse until a policy is declared.
  ❌ Accumulation cannot make an ill-posed coupling well-posed. Two
     densities for one cell is a MODELING conflict; surfaced, not solved.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

log = logging.getLogger("buffer_acc")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
ACC_VERSION = "1.0.0"


class Combine(str, Enum):
    SUM = "sum"; MAX = "max"; MEAN = "mean"
    PRIORITY = "priority"; REQUIRE = "require"


class BufferConflict(RuntimeError):
    pass


DEFAULT_POLICY: Dict[str, Combine] = {
    "_ext_fx": Combine.SUM, "_ext_fy": Combine.SUM, "_ext_fz": Combine.SUM,
    "_ext_q": Combine.SUM,
    "_ext_mdot": Combine.SUM, "_ext_mdot_u": Combine.SUM,
    "_ext_mdot_v": Combine.SUM, "_ext_mdot_w": Combine.SUM,
    "_ext_mdot_e": Combine.SUM, "_ext_mdot_Z": Combine.SUM,
    "_ext_rho_ch": Combine.PRIORITY, "_ext_rho_eff": Combine.PRIORITY,
    "_ext_nu_ch": Combine.PRIORITY, "_ext_nu_eff": Combine.PRIORITY,
    "_ext_sigma": Combine.MAX, "_ext_mask": Combine.MAX,
}


@dataclass
class Contribution:
    source: str
    value: Tensor
    priority: int = 0


class _Shadow:
    """Stand-in target: tracked buffers stored locally, rest forwarded."""
    def __init__(self, real: Any, tracked: List[str]):
        object.__setattr__(self, "_real", real)
        buf = {}
        for n in tracked:
            if hasattr(real, n):
                v = getattr(real, n)
                buf[n] = v.clone() if isinstance(v, Tensor) else v
        object.__setattr__(self, "_buf", buf)
        object.__setattr__(self, "_touched", set())

    def __getattr__(self, k):
        buf = object.__getattribute__(self, "_buf")
        if k in buf:
            return buf[k]
        return getattr(object.__getattribute__(self, "_real"), k)

    def __setattr__(self, k, v):
        buf = object.__getattribute__(self, "_buf")
        touched = object.__getattribute__(self, "_touched")
        if k in buf or k.startswith("_ext_"):
            buf[k] = v; touched.add(k)
        else:
            setattr(object.__getattribute__(self, "_real"), k, v)


class BufferAccumulator:
    def __init__(self, solver: Any, policy: Optional[Dict[str, Combine]] = None):
        self.solver = solver
        self.policy = dict(DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)
        self._pending: Dict[str, List[Contribution]] = {}
        self._history: List[Dict] = []

    def rule(self, buf: str) -> Combine:
        return self.policy.get(buf, Combine.REQUIRE)

    def stage(self, buf: str, value: Tensor, source: str, priority: int = 0):
        if self.rule(buf) is Combine.REQUIRE:
            raise BufferConflict(
                f"buffer '{buf}' has no declared combine policy. Additive "
                f"sources use SUM; material properties use PRIORITY (never "
                f"sum). Declare it: acc.policy['{buf}'] = Combine.SUM")
        self._pending.setdefault(buf, []).append(
            Contribution(source, value, priority))

    def _combine(self, buf: str, cs: List[Contribution]) -> Tensor:
        r = self.rule(buf)
        if r is Combine.SUM:
            out = cs[0].value.clone()
            for c in cs[1:]:
                out = out + c.value
            return out
        if r is Combine.MAX:
            out = cs[0].value.clone()
            for c in cs[1:]:
                out = torch.maximum(out, c.value)
            return out
        if r is Combine.MEAN:
            return sum(c.value for c in cs) / len(cs)
        if r is Combine.PRIORITY:
            top = max(c.priority for c in cs)
            win = [c for c in cs if c.priority == top]
            if len(win) > 1:
                raise BufferConflict(
                    f"buffer '{buf}': {len(win)} sources at equal priority "
                    f"{top} ({[c.source for c in win]}). This is a MATERIAL "
                    f"PROPERTY — summing is unphysical and silent selection "
                    f"is unsafe. Assign distinct priorities.")
            return win[0].value.clone()
        raise BufferConflict(f"unhandled rule {r}")

    def commit(self) -> Dict[str, Any]:
        rep = {}
        for buf, cs in self._pending.items():
            val = self._combine(buf, cs)
            setattr(self.solver, buf, val)
            rep[buf] = {"rule": self.rule(buf).value,
                        "sources": [c.source for c in cs],
                        "priorities": [c.priority for c in cs],
                        "norm": float(val.norm())}
        self._history.append(rep); self._pending.clear()
        return rep

    def reset(self): self._pending.clear()
    def provenance(self): return list(self._history)


class BridgeHub:
    """Wrap UNMODIFIED one_core bridges via a shadow target."""
    def __init__(self, solver: Any, policy=None):
        self.solver = solver
        self.acc = BufferAccumulator(solver, policy)
        self._bridges: List[Dict] = []
        self._wall_owners: Dict[str, str] = {}

    def register(self, bridge: Any, target: str, name: str, priority: int = 0):
        if not hasattr(bridge, target):
            raise ValueError(f"bridge {name} has no attribute '{target}'")
        if getattr(bridge, "target", None) == "wall":
            face = getattr(bridge, "wall_face", None)
            if face in self._wall_owners:
                raise BufferConflict(
                    f"wall face '{face}' already owned by "
                    f"'{self._wall_owners[face]}'. Wall BC writes are NOT "
                    f"accumulable (one physical state per face). Use "
                    f"target='volumetric' if superposition is needed.")
            self._wall_owners[face] = name
            self._bridges.append(dict(bridge=bridge, target=target, name=name,
                                      priority=priority, wall=True))
            return
        self._bridges.append(dict(bridge=bridge, target=target, name=name,
                                  priority=priority, wall=False))

    def sync_all(self) -> Dict[str, Any]:
        tracked = [k for k in self.acc.policy if hasattr(self.solver, k)]
        self.acc.reset()
        for rec in self._bridges:
            br, tgt, name, pri = (rec["bridge"], rec["target"],
                                  rec["name"], rec["priority"])
            if rec["wall"]:
                br.sync(); continue
            real = getattr(br, tgt)
            shadow = _Shadow(real, tracked)
            setattr(br, tgt, shadow)
            try:
                br.sync()
            finally:
                setattr(br, tgt, real)
            touched = object.__getattribute__(shadow, "_touched")
            buf = object.__getattribute__(shadow, "_buf")
            for k in touched:
                if isinstance(buf[k], Tensor):
                    self.acc.stage(k, buf[k], source=name, priority=pri)
        return self.acc.commit()


# =============================================================================
def self_test() -> bool:
    print("=" * 78)
    print(f"  BUFFER ACCUMULATOR v{ACC_VERSION} — verification")
    print("=" * 78)
    torch.manual_seed(0); ok = True; N = 6

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    class S:
        def __init__(self):
            self.device = torch.device("cpu"); self.time = 0.0
            z = torch.zeros(N, N, N, dtype=torch.float64)
            for k in ("_ext_fx", "_ext_fy", "_ext_fz", "_ext_q",
                      "_ext_rho_ch", "_ext_nu_ch", "_ext_sigma", "_ext_mask"):
                setattr(self, k, z.clone())

    class B:  # mimics one_core sync(): direct attribute writes
        def __init__(self, s, fx=None, rho=None, mask=None):
            self.dns = s; self._fx, self._rho, self._mask = fx, rho, mask
        def sync(self):
            if self._fx is not None:   self.dns._ext_fx = self._fx
            if self._rho is not None:  self.dns._ext_rho_ch = self._rho
            if self._mask is not None: self.dns._ext_mask = self._mask

    s = S()
    fx_ch = torch.full((N, N, N), 2.0, dtype=torch.float64)
    fx_se = torch.full((N, N, N), 3.0, dtype=torch.float64)

    print("\n[T1] the OLD behaviour really does lose a contribution")
    s._ext_fx = fx_ch.clone(); s._ext_fx = fx_se.clone()
    rep("overwrite discards CH", float(s._ext_fx.mean()) == 3.0,
        f"got {float(s._ext_fx.mean()):.1f}; CH's 2.0 gone")

    print("\n[T2] forces SUM + provenance")
    acc = BufferAccumulator(s)
    acc.stage("_ext_fx", fx_ch, "CH"); acc.stage("_ext_fx", fx_se, "Seismic")
    info = acc.commit()
    rep("sum = 5.0", float(s._ext_fx.mean()) == 5.0)
    rep("provenance", info["_ext_fx"]["sources"] == ["CH", "Seismic"],
        str(info["_ext_fx"]["sources"]))

    print("\n[T3] material property must NOT sum")
    ra = torch.full((N, N, N), 1000.0, dtype=torch.float64)
    rb = torch.full((N, N, N), 1.0, dtype=torch.float64)
    acc.reset()
    acc.stage("_ext_rho_ch", ra, "CH", 0); acc.stage("_ext_rho_ch", rb, "Other", 0)
    try:
        acc.commit(); rep("equal-priority conflict raises", False, "no raise")
    except BufferConflict as e:
        rep("equal-priority conflict raises", True, str(e).splitlines()[0][:60])
    rep("density NOT summed to 1001", float(s._ext_rho_ch.mean()) != 1001.0)

    print("\n[T4] explicit PRIORITY resolves deterministically")
    acc.reset()
    acc.stage("_ext_rho_ch", ra, "CH", 10); acc.stage("_ext_rho_ch", rb, "Other", 0)
    acc.commit()
    rep("higher priority wins", float(s._ext_rho_ch.mean()) == 1000.0)

    print("\n[T5] masks MAX not SUM (the FH collision)")
    m1 = torch.full((N, N, N), 0.3, dtype=torch.float64)
    m2 = torch.full((N, N, N), 0.8, dtype=torch.float64)
    acc.reset(); acc.stage("_ext_mask", m1, "Lang"); acc.stage("_ext_mask", m2, "CH")
    acc.commit()
    rep("max = 0.8 (not 1.1)", abs(float(s._ext_mask.mean()) - 0.8) < 1e-12)

    print("\n[T6] unknown buffer refuses")
    acc.reset()
    try:
        acc.stage("_ext_brand_new", m1, "A"); rep("unknown raises", False)
    except BufferConflict:
        rep("unknown raises", True)

    print("\n[T7] BridgeHub wraps UNMODIFIED bridges")
    s2 = S()
    b1, b2 = B(s2, fx=fx_ch, mask=m1), B(s2, fx=fx_se, mask=m2)
    hub = BridgeHub(s2)
    hub.register(b1, "dns", "CH"); hub.register(b2, "dns", "Seismic")
    inf = hub.sync_all()
    rep("hub sums forces", float(s2._ext_fx.mean()) == 5.0)
    rep("hub maxes masks", abs(float(s2._ext_mask.mean()) - 0.8) < 1e-12)
    rep("bridges restored", b1.dns is s2 and b2.dns is s2)
    rep("both sources recorded", set(inf["_ext_fx"]["sources"]) == {"CH", "Seismic"})

    print("\n[T8] reset clears pending")
    acc.reset(); acc.stage("_ext_fx", fx_ch, "CH"); acc.reset()
    rep("nothing committed", acc.commit() == {})

    print("\n[T9] gradient survives accumulation")
    a = fx_ch.clone().requires_grad_(True); b = fx_se.clone().requires_grad_(True)
    acc.reset(); acc.stage("_ext_fx", a, "CH"); acc.stage("_ext_fx", b, "Seismic")
    acc.commit(); s._ext_fx.pow(2).sum().backward()
    rep("both sources get gradient",
        a.grad is not None and b.grad is not None
        and float(a.grad.abs().sum()) > 0 and float(b.grad.abs().sum()) > 0)

    print("\n" + "=" * 78)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 78)
    print("""
  PROVEN: overwrite loses data; forces superpose; material properties refuse
          to sum and refuse silent selection; unmodified bridges work.
  NOT SOLVED: wall-mode BC writes (one-owner-per-face enforced instead);
              ill-posed couplings (surfaced, not resolved).
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="selftest",
                   choices=["selftest", "policy"])
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if self_test() else 1)
    print(json.dumps({k: v.value for k, v in DEFAULT_POLICY.items()}, indent=2))


if __name__ == "__main__":
    main()
