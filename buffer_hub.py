#!/usr/bin/env python3
"""
BUFFER HUB — collision-safe accumulation for one_core bridges.
Material properties REFUSE to sum; _ext_mask collision is explicit.
Developer: PAI and Yoon A Limsuwan / MSPS NETWORK | MIT | 2026
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import torch
from torch import Tensor


class Combine(str, Enum):
    SUM="sum"; MAX="max"; MEAN="mean"; PRIORITY="priority"; REQUIRE="require"


class BufferConflict(RuntimeError): pass


# _ext_mask collides between LangevinFHBridge and CahnHilliardFHBridge.
# MAX is the declared resolution: an interface is present if EITHER says so.
POLICY: Dict[str, Combine] = {
    "_ext_fx":Combine.SUM, "_ext_fy":Combine.SUM, "_ext_fz":Combine.SUM,
    "_ext_q":Combine.SUM,
    "_ext_mdot":Combine.SUM, "_ext_mdot_u":Combine.SUM, "_ext_mdot_v":Combine.SUM,
    "_ext_mdot_w":Combine.SUM, "_ext_mdot_e":Combine.SUM, "_ext_mdot_Z":Combine.SUM,
    "_ext_rho_ch":Combine.PRIORITY, "_ext_rho_eff":Combine.PRIORITY,
    "_ext_nu_ch":Combine.PRIORITY,  "_ext_nu_eff":Combine.PRIORITY,
    "_ext_sigma":Combine.MAX, "_ext_mask":Combine.MAX,
}
KNOWN_COLLISIONS = [
    ("DNS", "_ext_fx/_ext_fy/_ext_fz",
     ["CahnHilliardDNSBridge", "SeismicDNSBridge"], Combine.SUM),
    ("FH", "_ext_mask",
     ["LangevinFHBridge", "CahnHilliardFHBridge"], Combine.MAX),
]


@dataclass
class Contribution:
    source: str; value: Tensor; priority: int = 0


class _Shadow:
    def __init__(self, real, tracked):
        object.__setattr__(self,"_real",real)
        buf = {n: (getattr(real,n).clone()
                   if isinstance(getattr(real,n),Tensor) else getattr(real,n))
               for n in tracked if hasattr(real,n)}
        object.__setattr__(self,"_buf",buf)
        object.__setattr__(self,"_touched",set())
    def __getattr__(self,k):
        b = object.__getattribute__(self,"_buf")
        return b[k] if k in b else getattr(object.__getattribute__(self,"_real"),k)
    def __setattr__(self,k,v):
        b = object.__getattribute__(self,"_buf")
        t = object.__getattribute__(self,"_touched")
        if k in b or k.startswith("_ext_"): b[k]=v; t.add(k)
        else: setattr(object.__getattribute__(self,"_real"),k,v)


class BufferHub:
    def __init__(self, solver, policy=None):
        self.solver = solver
        self.policy = dict(POLICY); self.policy.update(policy or {})
        self._pending: Dict[str,List[Contribution]] = {}
        self._bridges: List[Dict] = []
        self._wall_owner: Dict[str,str] = {}
        self._history: List[Dict] = []

    def rule(self,b): return self.policy.get(b, Combine.REQUIRE)

    def stage(self, buf, value, source, priority=0):
        if self.rule(buf) is Combine.REQUIRE:
            raise BufferConflict(
                f"'{buf}' has no declared policy. Additive sources use SUM; "
                f"material properties use PRIORITY (never sum). Declare it.")
        self._pending.setdefault(buf,[]).append(Contribution(source,value,priority))

    def _combine(self, buf, cs):
        r = self.rule(buf)
        if r is Combine.SUM:
            o = cs[0].value.clone()
            for c in cs[1:]: o = o + c.value
            return o
        if r is Combine.MAX:
            o = cs[0].value.clone()
            for c in cs[1:]: o = torch.maximum(o, c.value)
            return o
        if r is Combine.MEAN:
            return sum(c.value for c in cs)/len(cs)
        top = max(c.priority for c in cs)
        win = [c for c in cs if c.priority==top]
        if len(win)>1:
            raise BufferConflict(
                f"'{buf}': {len(win)} sources at equal priority {top} "
                f"({[c.source for c in win]}). MATERIAL PROPERTY — summing is "
                f"unphysical and silent selection is unsafe. Assign distinct "
                f"priorities or resolve the modeling conflict yourself.")
        return win[0].value.clone()

    def commit(self):
        rep={}
        for buf,cs in self._pending.items():
            val=self._combine(buf,cs); setattr(self.solver,buf,val)
            rep[buf]={"rule":self.rule(buf).value,
                      "sources":[c.source for c in cs],
                      "priorities":[c.priority for c in cs]}
        self._history.append(rep); self._pending.clear(); return rep

    def reset(self): self._pending.clear()

    def register(self, bridge, target, name, priority=0):
        if not hasattr(bridge,target):
            raise ValueError(f"bridge {name} has no '{target}'")
        if getattr(bridge,"target",None)=="wall":
            f=getattr(bridge,"wall_face",None)
            if f in self._wall_owner:
                raise BufferConflict(
                    f"wall face '{f}' already owned by '{self._wall_owner[f]}'. "
                    f"Wall BC writes are NOT accumulable (one physical state "
                    f"per face). Use target='volumetric' for superposition.")
            self._wall_owner[f]=name
            self._bridges.append(dict(b=bridge,t=target,n=name,p=priority,wall=True))
            return
        self._bridges.append(dict(b=bridge,t=target,n=name,p=priority,wall=False))

    def sync_all(self):
        tracked=[k for k in self.policy if hasattr(self.solver,k)]
        self.reset()
        for r in self._bridges:
            if r["wall"]: r["b"].sync(); continue
            real=getattr(r["b"],r["t"]); sh=_Shadow(real,tracked)
            setattr(r["b"],r["t"],sh)
            try: r["b"].sync()
            finally: setattr(r["b"],r["t"],real)
            for k in object.__getattribute__(sh,"_touched"):
                v=object.__getattribute__(sh,"_buf")[k]
                if isinstance(v,Tensor): self.stage(k,v,r["n"],r["p"])
        return self.commit()


def self_test() -> bool:
    print("="*78); print("  BUFFER HUB — verification"); print("="*78)
    ok=True; N=6
    def rep(n,p,e=""):
        nonlocal ok; ok&=bool(p); print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")
    class S:
        def __init__(s):
            s.device=torch.device("cpu"); s.time=0.0
            z=torch.zeros(N,N,N,dtype=torch.float64)
            for k in POLICY: setattr(s,k,z.clone())
    class B:
        def __init__(s,sol,**kw): s.dns=sol; s.fh=sol; s.kw=kw
        def sync(s):
            for k,v in s.kw.items(): setattr(s.dns,k,v)
    s=S()
    fx1=torch.full((N,N,N),2.0,dtype=torch.float64)
    fx2=torch.full((N,N,N),3.0,dtype=torch.float64)
    m1=torch.full((N,N,N),0.3,dtype=torch.float64)
    m2=torch.full((N,N,N),0.8,dtype=torch.float64)
    ra=torch.full((N,N,N),1000.0,dtype=torch.float64)
    rb=torch.full((N,N,N),1.0,dtype=torch.float64)

    print("\n[H1] old overwrite loses a contribution")
    s._ext_fx=fx1.clone(); s._ext_fx=fx2.clone()
    rep("CH's 2.0 discarded", float(s._ext_fx.mean())==3.0)

    print("\n[H2] forces SUM through the hub (unmodified bridges)")
    s2=S(); hub=BufferHub(s2)
    hub.register(B(s2,_ext_fx=fx1,_ext_mask=m1),"dns","CH")
    hub.register(B(s2,_ext_fx=fx2,_ext_mask=m2),"dns","Seismic")
    info=hub.sync_all()
    rep("sum = 5.0", float(s2._ext_fx.mean())==5.0)
    rep("both sources recorded",
        set(info["_ext_fx"]["sources"])=={"CH","Seismic"})

    print("\n[H3] _ext_mask collision resolved by MAX, not SUM")
    rep("mask = 0.8 (not 1.1)", abs(float(s2._ext_mask.mean())-0.8)<1e-12,
        f"got {float(s2._ext_mask.mean()):.3f}")
    rep("policy is MAX", info["_ext_mask"]["rule"]=="max")

    print("\n[H4] material property REFUSES to sum at equal priority")
    s3=S(); h3=BufferHub(s3)
    h3.stage("_ext_rho_ch",ra,"CH",0); h3.stage("_ext_rho_ch",rb,"Other",0)
    try:
        h3.commit(); rep("equal-priority raises", False, "no raise")
    except BufferConflict as e:
        rep("equal-priority raises", True, str(e).splitlines()[0][:58])
    rep("density NOT summed to 1001", float(s3._ext_rho_ch.mean())!=1001.0)

    print("\n[H5] explicit priority resolves deterministically")
    h3.reset(); h3.stage("_ext_rho_ch",ra,"CH",10); h3.stage("_ext_rho_ch",rb,"O",0)
    h3.commit()
    rep("higher priority wins", float(s3._ext_rho_ch.mean())==1000.0)

    print("\n[H6] unknown buffer refuses until declared")
    h3.reset()
    try:
        h3.stage("_ext_new",m1,"A"); rep("unknown raises", False)
    except BufferConflict: rep("unknown raises", True)

    print("\n[H7] gradient survives accumulation")
    a=fx1.clone().requires_grad_(True); b=fx2.clone().requires_grad_(True)
    h3.reset(); h3.stage("_ext_fx",a,"CH"); h3.stage("_ext_fx",b,"Seis")
    h3.commit(); s3._ext_fx.pow(2).sum().backward()
    rep("both sources get gradient",
        a.grad is not None and b.grad is not None
        and float(a.grad.abs().sum())>0 and float(b.grad.abs().sum())>0)

    print("\n  KNOWN COLLISIONS (declared, not silent):")
    for solver,buf,bridges,rule in KNOWN_COLLISIONS:
        print(f"    {solver:4s} {buf:26s} {bridges}  -> {rule.value}")
    print("\n"+"="*78)
    print(f"  {'✅ ALL CHECKS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*78)
    return ok


def main():
    p=argparse.ArgumentParser()
    p.add_argument("cmd",nargs="?",default="selftest",
                   choices=["selftest","policy"])
    a=p.parse_args()
    if a.cmd=="selftest": raise SystemExit(0 if self_test() else 1)
    print(json.dumps({k:v.value for k,v in POLICY.items()},indent=2))


if __name__=="__main__": main()
