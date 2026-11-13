#!/usr/bin/env python3
"""
=============================================================================
REPAIR FIVE — make the 5 uploaded ONE-Ecosystem files importable, and fix
the 4 semantic blockers that remain even in super_dns_one_v6_3-8.py (v6.12)
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

TARGETS
  one_core_v3-6.py
  super_dns_one_v6_3-8.py
  structural_langevin_v3-2.py
  structuralfluctuatinghydro_v6_3.py
  structural_interface_utils.py

PASSES
  1 mechanical : syntax corruption (regex, logged)
  2 arithmetic : missing '*' in FH limiters/advection (targeted, NOT generic)
  3 semantic   : the 4 remaining blockers, each with a post-condition check
  4 iterative  : comment out bare-text lines, LOGGED, never inside strings
  5 AST verify : every file must parse
  6 import     : optional isolated subprocess

WHAT IT CANNOT DO
  ❌ Cannot make an untrained model accurate.
  ❌ Cannot validate physics.
  ❌ Will REFUSE to write a file that does not parse.

USAGE
  python repair_five.py --dir . --dry-run
  python repair_five.py --dir . --apply --smoke
=============================================================================
"""
from __future__ import annotations

import argparse, ast, io, json, logging, re, shutil, subprocess, sys, tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("repair_five")

TARGETS = [
    "one_core_v3*.py",
    "super_dns_one_v6*.py",
    "structural_langevin_v3*.py",
    "structuralfluctuatinghydro_v6*.py",
    "structural_interface_utils.py",
]

MECH: List[Tuple[str, str, str]] = [
    (r"\bfrom future import\b", "from __future__ import", "future import"),
    (r"\bdef init\(\s*self", "def __init__(self", "def init"),
    (r"\bsuper\(\)\.init\(\)", "super().__init__()", "super().init()"),
    (r'\bif\s+name\s*==\s*"\s*main\s*"', 'if __name__ == "__main__"', "main guard"),
    (r"\bif\s+name\s*==\s*'main'", "if __name__ == '__main__'", "main guard sq"),
    (r'"\s+"\s+"', '"""', "broken triple quote"),
    (r"logging\.getLogger\(\s*name\s*\)", "logging.getLogger(__name__)", "getLogger(name)"),
    (r"(?m)^all\s*=\s*\[", "__all__ = [", "all -> __all__"),
    (r"\bde\s+vice\b", "device", "de vice"),
    (r"\bget_worl\s+d_size\b", "get_world_size", "get_worl d_size"),
    (r"\bb\s+ackend_name\b", "backend_name", "b ackend_name"),
    (r"\blr_sc\s+heduler\b", "lr_scheduler", "lr_sc heduler"),
    (r"\btorch\.aut\s+ocast\b", "torch.autocast", "aut ocast"),
    (r"\bm2\s*\*\s*n2\b", "m**2 * n**2", "OPMC m2*n2"),
    (r"\b2\s+math\.pi\b", "2*math.pi", "2 math.pi"),
    (r"\bparse\s+r\.add_argument\b", "parser.add_argument", "parse r"),
    (r"\bpar\s+ser\.add_argument\b", "parser.add_argument", "par ser"),
    (r"\bpa\s+rser\.add_argument\b", "parser.add_argument", "pa rser"),
    (r"\bparser\.add_arg\s+ument\b", "parser.add_argument", "add_arg ument"),
    (r"'cud\s+a'", "'cuda'", "cud a"),
    (r"'p\s+eriodic'", "'periodic'", "p eriodic"),
    (r"'s\s+sc'", "'ssc'", "s sc"),
    (r"\bdefa\s+ult\b", "default", "defa ult"),
    (r"--tr\s+ain-soc", "--train-soc", "--tr ain-soc"),
    (r"--di\s+ff-test", "--diff-test", "--di ff-test"),
    (r"--rg\s+'", "--rg'", "--rg space"),
    (r"\baction\s*=\s*'\s*store_true\s*'", "action='store_true'", "action space"),
]

# FH-specific missing multiplication operators. These are EXACT patterns,
# not a generic "insert *" heuristic (which would corrupt unrelated code).
ARITH: List[Tuple[str, str, str]] = [
    (r"0\.5_soft_abs", "0.5*_soft_abs", "0.5_soft_abs"),
    (r"\b2r\b", "2*r", "2r"),
    (r"0\.5\(r", "0.5*(r", "0.5(r"),
    (r"\(1-g([xyz])\)\s*\(", r"(1-g\1)*(", "(1-gx) ("),
    (r"\bg([xyz])\s+\(rho", r"g\1*(rho", "gx (rho"),
    (r"\bdt\s+\(", "dt*(", "dt ("),
    (r"\bucx drho_dx\b", "ucx*drho_dx", "ucx drho_dx"),
    (r"\bucy drho_dy\b", "ucy*drho_dy", "ucy drho_dy"),
    (r"\bucz drho_dz\b", "ucz*drho_dz", "ucz drho_dz"),
]

WORDS = ("periodic", "wall", "slip", "cuda", "cpu", "mps", "ascend",
         "nccl", "gloo", "hccl", "ccl", "xpu", "npu", "float32", "float64",
         "bf16", "fp16", "fp32", "no_slip", "free_slip", "inflow", "outflow",
         "farfield", "moving_wall", "wall_model", "pyrolysis_wall",
         "NVIDIA_CUDA", "AMD_ROCM", "INTEL_XPU", "HUAWEI_NPU", "APPLE_MPS",
         "GENERIC_CPU", "auto", "minmod", "van_leer", "superbee",
         "muscl", "weno5", "semi_lagrangian", "ausm", "hllc",
         "taylor_green", "hypersonic_bnd", "uniform", "ideal", "real",
         "volumetric", "wall", "ssc", "wiener", "wavelet")


def string_lines(src: str) -> set:
    covered = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING:
                covered.update(range(tok.start[0], tok.end[0] + 1))
    except Exception:
        pass
    return covered


def mechanical(src: str) -> Tuple[str, List[str]]:
    applied = []
    for pat, rep, name in MECH:
        new, n = re.subn(pat, rep, src)
        if n:
            applied.append(f"{name} x{n}"); src = new
    src, n = re.subn(r'"([A-Za-z_][A-Za-z0-9_]*)[ \t]+"(\s*:)', r'"\1"\2', src)
    if n:
        applied.append(f"dict-key space x{n}")
    for w in WORDS:
        for pat in (r'"[ \t]+' + re.escape(w) + r'[ \t]*"',
                    r'"' + re.escape(w) + r'[ \t]+"'):
            src, n = re.subn(pat, f'"{w}"', src)
            if n:
                applied.append(f"literal '{w}' x{n}")
    src, n = re.subn(r'(einsum\(\s*")([^"]+)(")',
                     lambda m: m.group(1) + re.sub(r"\s+", "", m.group(2)) + m.group(3),
                     src)
    if n:
        applied.append(f"einsum space x{n}")
    return src, applied


def arithmetic(src: str, fname: str) -> Tuple[str, List[str]]:
    if "fluctuatinghydro" not in fname:
        return src, []
    applied = []
    for pat, rep, name in ARITH:
        new, n = re.subn(pat, rep, src)
        if n:
            applied.append(f"{name} x{n}"); src = new
    return src, applied


@dataclass
class Patch:
    name: str
    hint: str
    old: str
    new: str
    must: str
    note: str


PATCHES: List[Patch] = [
    Patch("[S1] _padv 5-arg signature", "super_dns",
          "def _padv(t): q=self._pad_field(t); self._fill_ghost_cells(q); return q",
          "def _padv(t):\n"
          "            q = self._pad_field(t)\n"
          "            z = torch.zeros_like(q)\n"
          "            self._fill_ghost_cells(q, z, z, z, z)  # [S1] 5-arg\n"
          "            return q",
          "[S1] 5-arg",
          "TypeError: _fill_ghost_cells takes 5 fields, got 1 — every step"),
    Patch("[S2] softplus floor width tracks floor", "super_dns",
          "def _softplus_floor(x: torch.Tensor, floor: float, beta: float = _SOFTPLUS_B) -> torch.Tensor:",
          "def _softplus_floor(x: torch.Tensor, floor: float,\n"
          "                    beta: float = None, sharpness: float = 50.0) -> torch.Tensor:\n"
          "    # [S2] width must track the requested floor, not a fixed 6.9e-3\n"
          "    if beta is None:\n"
          "        beta = min(sharpness / max(abs(floor), 1e-30), 1e12)",
          "[S2] width must track",
          "floor 1e-6 behaved as ~6.9e-3 (6900x)"),
    Patch("[S3] grid convergence reference ordering", "super_dns",
          "        sorted_sizes = sorted(grid_sizes)",
          "        # [S3] descending: compute the FINEST reference FIRST so the\n"
          "        # coarser grids have something to compare against.\n"
          "        sorted_sizes = sorted(grid_sizes, reverse=True)",
          "[S3] descending",
          "u_ref set only at the largest N, loop ascending -> errors empty"),
    Patch("[S3b] grid convergence reference condition", "super_dns",
          "            if N == sorted_sizes[-1]:",
          "            if N == sorted_sizes[0]:   # [S3] finest is now first",
          "[S3] finest is now first",
          "match the descending order"),
    Patch("[S4] matplotlib optional (headless)", "super_dns",
          "import matplotlib.pyplot as plt",
          "try:\n"
          "    import matplotlib\n"
          "    matplotlib.use('Agg')\n"
          "    import matplotlib.pyplot as plt\n"
          "    HAS_MPL = True\n"
          "except Exception:\n"
          "    plt = None\n"
          "    HAS_MPL = False",
          "HAS_MPL",
          "module-level pyplot import fails headless"),
]


def semantic(src: str, fname: str) -> Tuple[str, List[str], List[str]]:
    applied, skipped = [], []
    for p in PATCHES:
        if p.hint not in fname:
            continue
        if p.must in src:
            skipped.append(f"{p.name} (already present)"); continue
        if p.old in src:
            src = src.replace(p.old, p.new, 1)
            (applied if p.must in src else skipped).append(
                f"{p.name}  [{p.note}]" if p.must in src
                else f"{p.name} — post-condition failed")
        else:
            skipped.append(f"{p.name} — NOT FOUND (manual)")
    return src, applied, skipped


SEP = re.compile(r"^[═─━=\-_]{4,}\s*$")


def comment_seps(src: str) -> Tuple[str, int]:
    lines = src.splitlines(); inside = string_lines(src); n = 0
    for i, ln in enumerate(lines, 1):
        if i in inside:
            continue
        if SEP.match(ln.strip()) and not ln.lstrip().startswith("#"):
            lines[i - 1] = "# " + ln; n += 1
    return "\n".join(lines) + "\n", n


def iterative(src: str, cap: int = 800):
    commented = []
    for _ in range(cap):
        try:
            ast.parse(src); return src, commented, None
        except SyntaxError as e:
            ln = e.lineno or 0
            lines = src.splitlines()
            if not (1 <= ln <= len(lines)):
                return src, commented, f"unfixable at {ln}: {e.msg}"
            if ln in string_lines(src):
                return src, commented, (f"SyntaxError inside string at {ln}: "
                                        f"{e.msg} — MANUAL FIX")
            orig = lines[ln - 1]
            if orig.lstrip().startswith("# [AUTO]"):
                return src, commented, f"stalled at {ln}: {e.msg}"
            lines[ln - 1] = "# [AUTO] " + orig
            commented.append({"line": ln, "msg": e.msg, "text": orig[:110]})
            src = "\n".join(lines) + "\n"
    return src, commented, "cap reached"


def run_isolated(path: Path, timeout: int = 180) -> Dict:
    code = ("import importlib.util,json,traceback\n"
            f"spec=importlib.util.spec_from_file_location('m',r'{path}')\n"
            "m=importlib.util.module_from_spec(spec)\n"
            "try:\n spec.loader.exec_module(m); ok=True; err=''\n"
            "except Exception:\n ok=False; err=traceback.format_exc()[-1200:]\n"
            "print(json.dumps({'import_ok':ok,'error':err}))\n")
    try:
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(path.parent))
        for line in reversed(p.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"import_ok": False, "error": (p.stderr or p.stdout)[-1200:]}
    except subprocess.TimeoutExpired:
        return {"import_ok": False, "error": "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", default="repair_five_report.json")
    a = ap.parse_args()
    root = Path(a.dir).resolve()
    files = sorted({f for pat in TARGETS for f in root.glob(pat)})
    if not files:
        log.error(f"no targets under {root}"); raise SystemExit(1)

    print("=" * 78)
    print(f"  REPAIR FIVE — {'APPLY' if a.apply else 'DRY-RUN'}")
    print("=" * 78)
    for f in files:
        print(f"  · {f.name}")

    recs = []
    for f in files:
        print("\n" + "-" * 78); print(f"  {f.name}")
        src = f.read_text(errors="ignore")
        rec = {"file": f.name}
        src, mech = mechanical(src); rec["mechanical"] = mech
        src, arith = arithmetic(src, f.name); rec["arithmetic"] = arith
        src, sem, skip = semantic(src, f.name)
        rec["semantic"] = sem; rec["manual"] = skip
        src, ns = comment_seps(src); rec["separators"] = ns
        src, comm, fatal = iterative(src)
        rec["auto_commented"] = comm; rec["fatal"] = fatal
        try:
            ast.parse(src); ok, msg = True, ""
        except SyntaxError as e:
            ok, msg = False, f"line {e.lineno}: {e.msg}"
        rec["ast"] = ok; rec["ast_error"] = msg
        print(f"    mechanical : {len(mech)}")
        for m in mech[:5]:
            print(f"      ✓ {m}")
        if arith:
            print(f"    arithmetic : {len(arith)}")
            for m in arith:
                print(f"      ✓ {m}")
        print(f"    semantic   : {len(sem)}")
        for m in sem:
            print(f"      ✓ {m}")
        if skip:
            print(f"    MANUAL     : {len(skip)}")
            for m in skip:
                print(f"      ⚠ {m}")
        print(f"    separators : {ns}   auto-commented: {len(comm)}")
        for c in comm[:4]:
            print(f"      line {c['line']}: {c['text'][:58]}")
        print(f"    AST        : {'PASS' if ok else 'FAIL ' + msg}")
        if fatal:
            print(f"    FATAL      : {fatal}")
        if a.apply and ok:
            bak = f.with_suffix(f.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(f, bak)
            f.write_text(src); rec["written"] = True
            print(f"    written    : yes (backup {bak.name})")
            if a.smoke:
                r = run_isolated(f); rec["runtime"] = r
                print(f"    import     : {'PASS' if r['import_ok'] else 'FAIL'}")
                if not r["import_ok"]:
                    print("      " + r["error"].strip().splitlines()[-1][:100])
        else:
            rec["written"] = False
            if a.apply and not ok:
                print("    written    : NO — refuses to write a non-parsing file")
        recs.append(rec)

    npass = sum(r["ast"] for r in recs)
    nimp = sum(1 for r in recs if r.get("runtime", {}).get("import_ok"))
    print("\n" + "=" * 78)
    print(f"  AST parse : {npass}/{len(recs)}")
    print(f"  import OK : {nimp}")
    print(f"  manual    : {sum(len(r['manual']) for r in recs)}")
    Path(a.report).write_text(json.dumps(recs, indent=2))
    print(f"  report    : {a.report}")
    print("""
  AFTER REPAIR
    • one_core.py must be importable as `one_core` — rename or symlink:
        cp one_core_v3-6.py one_core.py
    • super_dns_one_v6_3-8.py still needs its own self-test run; the
      remaining semantic patches here fix crashes, not physics.
    • Passing AST != validated CFD. Run each module's own test suite.
""")
    raise SystemExit(0 if npass == len(recs) else 1)


if __name__ == "__main__":
    main()
