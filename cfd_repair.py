#!/usr/bin/env python3
"""
=============================================================================
CFD REPAIR — mechanical + targeted semantic fixes for the 5 CFD files
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT IT DOES
  Pass 1  mechanical : syntax corruption (regex, safe, logged)
  Pass 2  semantic   : targeted patches with post-condition checks
  Pass 3  iterative  : comment out bare-text lines, LOGGED
  Pass 4  AST verify : every file must parse
  Pass 5  import     : optional, isolated subprocess

WHAT IT CANNOT DO
  ❌ Cannot create one_core.py (absent). super_dns cannot import without it.
  ❌ Cannot rewrite the face-shape logic in sesi_multiphase_flow.py.
     Use one_cfd_core.MACGrid instead.
  ❌ Cannot add the missing VOF advection to vof_sota.
     Use one_cfd_core.advect_alpha instead.
  ❌ Cannot validate physics. Passing AST != correct CFD.

USAGE
  python cfd_repair.py --dir . --dry-run
  python cfd_repair.py --dir . --apply
  python cfd_repair.py --dir . --apply --smoke
=============================================================================
"""
from __future__ import annotations

import argparse, ast, io, json, logging, re, shutil, subprocess, sys, tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("cfd_repair")

TARGETS = [
    "sesi_multiphase_vof_sota.py",
    "sesi_multiphase_flow_vof.py",
    "sesi_multiphase_flow.py",
    "structural_cahn_hilliard_3d*.py",
    "super_dns_one_v6*.py",
]

MECH: List[Tuple[str, str, str]] = [
    (r"\bfrom future import\b", "from __future__ import", "future import"),
    (r"\bdef init\(\s*self", "def __init__(self", "def init"),
    (r"\bsuper\(\)\.init\(\)", "super().__init__()", "super().init()"),
    (r'\bif\s+name\s*==\s*"\s*main\s*"', 'if __name__ == "__main__"', "main guard"),
    (r"\bif\s+name\s*==\s*'main'", "if __name__ == '__main__'", "main guard sq"),
    (r'"\s+"\s+"', '"""', "broken triple quote"),
    (r"logging\.getLogger\(\s*name\s*\)", "logging.getLogger(__name__)", "getLogger(name)"),
    (r"\bm2\s*\*\s*n2\b", "m**2 * n**2", "OPMC m2*n2"),
    (r"\b2\s+math\.pi\b", "2*math.pi", "2 math.pi"),
    (r"\bparse\s+r\.add_argument\b", "parser.add_argument", "parse r"),
    (r"\bpar\s+ser\.add_argument\b", "parser.add_argument", "par ser"),
    (r"\bparser\.add_arg\s+ument\b", "parser.add_argument", "add_arg ument"),
    (r"\bpa\s+rser\.add_argument\b", "parser.add_argument", "pa rser"),
    (r"\bget_worl\s+d_size\b", "get_world_size", "get_worl d_size"),
    (r"\bde\s+vice\b", "device", "de vice"),
    (r"\bb\s+ackend_name\b", "backend_name", "b ackend_name"),
    (r"\blr_sc\s+heduler\b", "lr_scheduler", "lr_sc heduler"),
    (r"\btorch\.aut\s+ocast\b", "torch.autocast", "aut ocast"),
    (r"'cud\s+a'", "'cuda'", "cud a"),
    (r"'p\s+eriodic'", "'periodic'", "p eriodic"),
    (r"'s\s+sc'", "'ssc'", "s sc"),
    (r"\bdefa\s+ult\b", "default", "defa ult"),
    (r"--tr\s+ain-soc", "--train-soc", "--tr ain-soc"),
    (r"--di\s+ff-test", "--diff-test", "--di ff-test"),
    (r"--rg\s+'", "--rg'", "--rg space"),
    (r"\baction\s*=\s*'\s*store_true\s*'", "action='store_true'", "action space"),
    (r"\bchoices\s*=\s*\[", "choices=[", "choices space"),
]

WORDS = ("wall", "slip", "periodic", "plic", "cicsam", "geometric", "upwind",
         "height_function", "smoothed_alpha", "filtered_alpha", "minmod",
         "central", "opmc_cg", "cg", "fft", "rk2_heun", "newtonian",
         "power_law", "carreau", "bingham", "laminar", "smagorinsky",
         "cpu", "cuda", "float32", "float64", "explicit", "imex",
         "conv3d", "roll", "no_slip", "free_slip", "inflow", "outflow",
         "brackbill", "smoothed", "nccl", "gloo")


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
    if n: applied.append(f"dict-key space x{n}")
    for w in WORDS:
        for pat in (r'"[ \t]+' + re.escape(w) + r'[ \t]*"',
                    r'"' + re.escape(w) + r'[ \t]+"'):
            src, n = re.subn(pat, f'"{w}"', src)
            if n: applied.append(f"literal '{w}' x{n}")
    src, n = re.subn(r'(einsum\(\s*")([^"]+)(")',
                     lambda m: m.group(1) + re.sub(r"\s+", "", m.group(2)) + m.group(3),
                     src)
    if n: applied.append(f"einsum space x{n}")
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
    Patch("[B1] face_beta inversion (sota)", "vof_sota",
          "hx = 2.0 / (1.0 / rho[:-1].clamp_min(1e-30) + 1.0 / rho[1:].clamp_min(1e-30))",
          "hx = 0.5 * (1.0 / rho[:-1].clamp_min(1e-30) + 1.0 / rho[1:].clamp_min(1e-30))  # [B1] beta=1/rho_face",
          "[B1] beta=1/rho_face",
          "projection used rho instead of 1/rho"),
    Patch("[B1] face_beta inversion (flow)", "sesi_multiphase_flow.py",
          "hx = 2.0 / (1.0 / rho[:-1].clamp_min(1e-30) + 1.0 / rho[1:].clamp_min(1e-30))",
          "hx = 0.5 * (1.0 / rho[:-1].clamp_min(1e-30) + 1.0 / rho[1:].clamp_min(1e-30))  # [B1] beta=1/rho_face",
          "[B1] beta=1/rho_face",
          "same inversion"),
    Patch("self.nz -> self.cfg.nz", "vof_sota",
          "return mask.reshape(self.cfg.nx, self.cfg.ny, self.nz).to(dt_)",
          "return mask.reshape(self.cfg.nx, self.cfg.ny, self.cfg.nz).to(dt_)",
          "self.cfg.ny, self.cfg.nz)",
          "AttributeError in mask builder"),
    Patch("bc_objects dict", "super_dns",
          "self.bc_z_min = _bc(cfg.bc_z_min); self.bc_z_max = _bc(cfg.bc_z_max)",
          "self.bc_z_min = _bc(cfg.bc_z_min); self.bc_z_max = _bc(cfg.bc_z_max)\n"
          "        # [FIX] _fill_ghost_cells indexes bc_objects[face]; build it.\n"
          "        self.bc_objects = {\n"
          "            'xmin': self.bc_x_min, 'xmax': self.bc_x_max,\n"
          "            'ymin': self.bc_y_min, 'ymax': self.bc_y_max,\n"
          "            'zmin': self.bc_z_min, 'zmax': self.bc_z_max,\n"
          "        }",
          "self.bc_objects = {",
          "AttributeError: bc_objects never created"),
    Patch("_padv arity", "super_dns",
          "def _padv(t): q=self._pad_field(t); self._fill_ghost_cells(q); return q",
          "def _padv(t):\n"
          "            q = self._pad_field(t)\n"
          "            z = torch.zeros_like(q)\n"
          "            self._fill_ghost_cells(q, z, z, z, z)  # [FIX] 5-arg signature\n"
          "            return q",
          "[FIX] 5-arg signature",
          "TypeError: 1 arg vs 5"),
    Patch("dead if-False drag", "sesi_multiphase_flow.py",
          "sp = torch.sqrt(ucc ** 2 + 0) if False else vel.abs()",
          "sp = vel.abs()  # [FIX] removed dead branch referencing undefined ucc",
          "[FIX] removed dead branch",
          "dead code referencing undefined name"),
    Patch("soft_floor beta scaling", "super_dns",
          "def _softplus_floor(x: torch.Tensor, floor: float, beta: float = _SOFTPLUS_B) -> torch.Tensor:",
          "def _softplus_floor(x: torch.Tensor, floor: float, beta: float = None) -> torch.Tensor:\n"
          "    # [B5] width must track the requested floor, not a fixed 6.9e-3\n"
          "    if beta is None:\n"
          "        beta = min(50.0 / max(abs(floor), 1e-30), 1e12)",
          "[B5] width must track",
          "floor 1e-6 behaved as 6.9e-3"),
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
            if p.must in src:
                applied.append(f"{p.name}  [{p.note}]")
            else:
                skipped.append(f"{p.name} — post-condition failed")
        else:
            skipped.append(f"{p.name} — NOT FOUND (manual)")
    return src, applied, skipped


SEP = re.compile(r"^[═─━=\-_]{4,}\s*$")


def comment_seps(src: str) -> Tuple[str, int]:
    lines = src.splitlines(); inside = string_lines(src); n = 0
    for i, ln in enumerate(lines, 1):
        if i in inside: continue
        if SEP.match(ln.strip()) and not ln.lstrip().startswith("#"):
            lines[i - 1] = "# " + ln; n += 1
    return "\n".join(lines) + "\n", n


def iterative(src: str, cap: int = 600):
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


def run_isolated(path: Path, timeout: int = 120) -> Dict:
    code = ("import importlib.util,json,traceback\n"
            f"spec=importlib.util.spec_from_file_location('m',r'{path}')\n"
            "m=importlib.util.module_from_spec(spec)\n"
            "try:\n spec.loader.exec_module(m); ok=True; err=''\n"
            "except Exception:\n ok=False; err=traceback.format_exc()[-1200:]\n"
            "print(json.dumps({'import_ok':ok,'error':err}))\n")
    try:
        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        for line in reversed(p.stdout.strip().splitlines()):
            if line.startswith("{"): return json.loads(line)
        return {"import_ok": False, "error": (p.stderr or p.stdout)[-1200:]}
    except subprocess.TimeoutExpired:
        return {"import_ok": False, "error": "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", default="cfd_repair_report.json")
    a = ap.parse_args()
    root = Path(a.dir).resolve()
    files = sorted({f for pat in TARGETS for f in root.glob(pat)})
    if not files:
        log.error(f"no targets under {root}"); raise SystemExit(1)

    print("=" * 78)
    print(f"  CFD REPAIR — {'APPLY' if a.apply else 'DRY-RUN'}")
    print("=" * 78)
    for f in files: print(f"  · {f.name}")

    recs = []
    for f in files:
        print("\n" + "-" * 78); print(f"  {f.name}")
        src = f.read_text(errors="ignore")
        rec = {"file": f.name}
        src, mech = mechanical(src); rec["mechanical"] = mech
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
        for m in mech[:5]: print(f"      ✓ {m}")
        print(f"    semantic   : {len(sem)}")
        for m in sem: print(f"      ✓ {m}")
        if skip:
            print(f"    MANUAL     : {len(skip)}")
            for m in skip: print(f"      ⚠ {m}")
        print(f"    separators : {ns}   auto-commented: {len(comm)}")
        print(f"    AST        : {'PASS' if ok else 'FAIL ' + msg}")
        if fatal: print(f"    FATAL      : {fatal}")
        if a.apply and ok:
            bak = f.with_suffix(f.suffix + ".bak")
            if not bak.exists(): shutil.copy2(f, bak)
            f.write_text(src); rec["written"] = True
            print(f"    written    : yes (backup {bak.name})")
            if a.smoke:
                r = run_isolated(f); rec["runtime"] = r
                print(f"    import     : {'PASS' if r['import_ok'] else 'FAIL'}")
                if not r["import_ok"]:
                    print("      " + r["error"].strip().splitlines()[-1][:100])
        else:
            rec["written"] = False
        recs.append(rec)

    npass = sum(r["ast"] for r in recs)
    print("\n" + "=" * 78)
    print(f"  AST parse : {npass}/{len(recs)}")
    print(f"  manual    : {sum(len(r['manual']) for r in recs)}")
    Path(a.report).write_text(json.dumps(recs, indent=2))
    print(f"  report    : {a.report}")
    print("""
  STILL BROKEN AFTER REPAIR (needs one_cfd_core or a rewrite)
    • sesi_multiphase_flow.py  face shapes, strain rate, _viscous_* stencils
    • sesi_multiphase_vof_sota.py  missing VOF advection, HF curvature
    • sesi_multiphase_flow_vof.py  geometric flux is upwind, curvature formula
    • super_dns_one_v6_3.py  requires one_core.py (ABSENT from corpus)
    • structural_cahn_hilliard  PFC IMEX denominator signs
""")
    raise SystemExit(0 if npass == len(recs) else 1)


if __name__ == "__main__":
    main()
