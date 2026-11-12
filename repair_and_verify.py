#!/usr/bin/env python3
"""
=============================================================================
REPAIR AND VERIFY — bug-fix sweep for the ONE/SESI/iPSC/MSA file family
=============================================================================
Developer : PAI and Yoon A Limsuwan / MSPS NETWORK
License   : MIT | Year: 2026

WHAT THIS DOES
  Pass 1  Mechanical repair   : syntax-level corruption (regex, safe)
  Pass 2  Semantic patches    : real logic bugs (targeted, each verified)
  Pass 3  Iterative syntax fix: comment out bare-text lines, LOGGED
  Pass 4  AST verification    : every file must parse
  Pass 5  Import + smoke      : optional, per-file, isolated subprocess

WHAT THIS CANNOT DO
  ❌ Cannot create CFD / multiphase solvers. They do not exist in the corpus.
  ❌ Cannot make learnable-Gaussian "quantum chemistry" into real QC.
     That needs PySCF/libcint integrals, not a patch.
  ❌ Cannot make random SU(3) links into real lattice gauge configurations.
     That needs Monte Carlo, not a patch.
  ❌ Cannot train anything. Weights stay random after repair.
  ❌ Cannot fix generate_benchmark_script's nested triple quotes automatically.
     A manual replacement is supplied separately.

USAGE
  python repair_and_verify.py --dir . --dry-run
  python repair_and_verify.py --dir . --apply
  python repair_and_verify.py --dir . --apply --smoke
=============================================================================
"""
from __future__ import annotations

import argparse, ast, io, json, logging, re, shutil, subprocess, sys, time
import tokenize
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("repair")

TARGETS = [
    "ONE_ECOSYSTEMv40*.py",
    "SESI_GNFS9*.py",
    "iPSC_ORGAN_REGENERATION*.py",
    "MSA_ENHANCED_PROTEIN_FOLDING*.py",
    "sesi_lite_msa_af3*.py",
]


# =============================================================================
# PASS 1 — MECHANICAL REPAIR
# =============================================================================

MECH_RULES: List[Tuple[str, str, str]] = [
    (r"\bfrom future import\b", "from __future__ import", "future import"),
    (r"\bdef init\(\s*self", "def __init__(self", "def init -> __init__"),
    (r"\bsuper\(\)\.init\(\)", "super().__init__()", "super().init()"),
    (r'\bif name == "main"', 'if __name__ == "__main__"', "main guard (dq)"),
    (r"\bif name == 'main'", "if __name__ == '__main__'", "main guard (sq)"),
    (r"(?m)^all\s*=\s*\[", "__all__ = [", "all -> __all__"),
    (r"\bde vice\b", "device", "split identifier: device"),
    (r"\bget_worl d_size\b", "get_world_size", "split: get_world_size"),
    (r"\bb ackend_name\b", "backend_name", "split: backend_name"),
    (r"\blr_sc heduler\b", "lr_scheduler", "split: lr_scheduler"),
    (r"\btorch\.aut ocast\b", "torch.autocast", "split: autocast"),
    (r"\bnn\.Mod ule\b", "nn.Module", "split: Module"),
    (r'"\s+"\s+"', '"""', 'broken triple-quote delimiter'),
]

DEVICE_WORDS = ("cuda", "cpu", "nccl", "gloo", "hccl", "ccl", "xpu", "npu",
                "mps", "bf16", "fp16", "fp32", "auto", "NVIDIA_CUDA",
                "AMD_ROCM", "INTEL_XPU", "HUAWEI_NPU", "APPLE_MPS",
                "GENERIC_CPU", "BAIDU_KUNLUN")


def string_token_lines(src: str) -> set:
    """Line numbers covered by STRING tokens (so we never edit inside them)."""
    covered = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING:
                covered.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return covered


def mechanical_repair(src: str) -> Tuple[str, List[str]]:
    applied = []
    for pat, rep, name in MECH_RULES:
        new, n = re.subn(pat, rep, src)
        if n:
            applied.append(f"{name} ×{n}")
            src = new

    # dict keys with trailing whitespace:  "output_dim ":  ->  "output_dim":
    src, n = re.subn(r'"([A-Za-z_][A-Za-z0-9_]*)[ \t]+"(\s*:)', r'"\1"\2', src)
    if n:
        applied.append(f"dict-key trailing space ×{n}")

    # device / backend literals with stray whitespace
    for w in DEVICE_WORDS:
        src, n = re.subn(r'"[ \t]+' + re.escape(w) + r'[ \t]*"', f'"{w}"', src)
        if n:
            applied.append(f"device literal '{w}' ×{n}")
        src, n = re.subn(r'"' + re.escape(w) + r'[ \t]+"', f'"{w}"', src)
        if n:
            applied.append(f"device literal '{w}' (trail) ×{n}")

    # einsum subscripts must contain no whitespace
    def _fix_einsum(m):
        return m.group(1) + re.sub(r"\s+", "", m.group(2)) + m.group(3)
    src, n = re.subn(r'(einsum\(\s*")([^"]+)(")', _fix_einsum, src)
    if n:
        applied.append(f"einsum subscript whitespace ×{n}")

    return src, applied


# =============================================================================
# PASS 2 — SEMANTIC PATCHES (each with a post-condition check)
# =============================================================================

@dataclass
class Patch:
    name: str
    file_hint: str            # substring of filename, or "*"
    pattern: str              # regex
    replacement: str
    must_contain: str         # post-condition substring
    flags: int = re.DOTALL
    note: str = ""


PATCHES: List[Patch] = [

    # ---- iPSC / sesi_lite : trailing-space PARAM keys already handled in Pass 1,
    #      but the lookup fallback also needs the stripped default.
    Patch(
        name="protein-type default lookup",
        file_hint="sesi_lite",
        pattern=r'protein_type\s*=\s*"default\s+"',
        replacement='protein_type = "default"',
        must_contain='protein_type = "default"',
        note="KeyError on every ProteinTypeSpecificRefiner construction",
    ),

    # ---- sesi_lite : IPA proj_out dimension mismatch when z is None
    Patch(
        name="IPA concat: always include pair term",
        file_hint="sesi_lite",
        pattern=(r"if z is not None:\s*\n\s*z_out = z\.mean\(dim=2\)\s*\n"
                 r"\s*concat = torch\.cat\(\[v_scalar_out, z_out, v_point_out, v_point_norms\], dim=-1\)\s*\n"
                 r"\s*else:\s*\n"
                 r"\s*concat = torch\.cat\(\[v_scalar_out, v_point_out, v_point_norms\], dim=-1\)"),
        replacement=(
            "# [PATCHED] proj_out always expects the pair term; supply zeros when z is None\n"
            "        if z is not None:\n"
            "            z_out = z.mean(dim=2)\n"
            "        else:\n"
            "            z_out = s.new_zeros(B, L, self.d_pair)\n"
            "        concat = torch.cat(\n"
            "            [v_scalar_out, z_out, v_point_out, v_point_norms], dim=-1)"
        ),
        must_contain="z_out = s.new_zeros(B, L, self.d_pair)",
        note="shape mismatch on every IPA call with z=None (all self-tests)",
    ),

    # ---- sesi_lite : IPA inverse transform used the transposed tensor
    Patch(
        name="IPA local-frame inverse transform",
        file_hint="sesi_lite",
        pattern=(r'v_point_local = torch\.einsum\("blij,bhlkj->bhlki",\s*rot_inv,\s*v_point_global\)'),
        replacement=(
            "# [PATCHED] v_point_global was transposed to (B,H,p,L,3); use the\n"
            "        # aggregated output, transpose back to (B,H,L,p,3), subtract translation.\n"
            "        v_point_back = v_point_out.transpose(2, 3) - trans.unsqueeze(2).unsqueeze(2)\n"
            '        v_point_local = torch.einsum("blij,bhlkj->bhlki", rot_inv, v_point_back)'
        ),
        must_contain="v_point_back = v_point_out.transpose(2, 3)",
        note="wrong index layout: l matched n_points instead of L",
    ),

    # ---- v40 : Hodge SNR used the kernel penalty in the numerator
    Patch(
        name="Hodge SNR direction",
        file_hint="ONE_ECOSYSTEM",
        pattern=(r"h_energy = lap\.pow\(2\)\.sum\(dim=-1\)\s*\n"
                 r"\s*e_energy = exact\.pow\(2\)\.sum\(dim=-1\)\s*\n"
                 r"\s*c_energy = coexact\.pow\(2\)\.sum\(dim=-1\)\s*\n"
                 r"\s*snr = \(h_energy \+ e_energy\) / \(c_energy \+ 1e-8\)"),
        replacement=(
            "# [PATCHED] SNR must use the harmonic NORM, not its Laplacian penalty.\n"
            "        kernel_penalty = lap.pow(2).sum(dim=-1)   # minimize -> h near ker Δ\n"
            "        h_energy = hp.pow(2).sum(dim=-1)          # signal magnitude\n"
            "        e_energy = exact.pow(2).sum(dim=-1)\n"
            "        c_energy = coexact.pow(2).sum(dim=-1)\n"
            "        snr = (h_energy + e_energy) / (c_energy + 1e-8)"
        ),
        must_contain="kernel_penalty = lap.pow(2).sum(dim=-1)",
        note="maximizing SNR pushed harmonic AWAY from ker Δ (inverted intent)",
    ),
    Patch(
        name="Hodge expose kernel_penalty",
        file_hint="ONE_ECOSYSTEM",
        pattern=r'"harmonic_energy": h_energy,',
        replacement='"harmonic_energy": h_energy,\n            "kernel_penalty": kernel_penalty,',
        must_contain='"kernel_penalty": kernel_penalty,',
        note="make the new penalty reachable by the loss",
    ),

    # ---- v40 : hierarchical pool can receive n_groups == 0
    Patch(
        name="hierarchical pool zero-group guard",
        file_hint="ONE_ECOSYSTEM",
        pattern=r"g = min\(n_groups, N\)",
        replacement="g = max(1, min(n_groups, N))  # [PATCHED] avoid N//0",
        must_contain="g = max(1, min(n_groups, N))",
        note="n_proteins//8 can be 0 -> ZeroDivisionError",
    ),

    # ---- v30 : view() requires divisibility by 4
    Patch(
        name="stem-cell protein view divisibility",
        file_hint="SESI_GNFS9",
        pattern=(r"N = domains\.shape\[1\]\s*\n"
                 r"\s*grouped = domains\.view\(B, N // 4, 4 \* self\.cfg\.d_mn\)"),
        replacement=(
            "N = domains.shape[1]\n"
            "        # [PATCHED] pad to a multiple of 4 before view\n"
            "        pad = (-N) % 4\n"
            "        if pad:\n"
            "            domains = F.pad(domains, (0, 0, 0, pad))\n"
            "            N = domains.shape[1]\n"
            "        grouped = domains.reshape(B, N // 4, 4 * self.cfg.d_mn)"
        ),
        must_contain="pad = (-N) % 4",
        note="RuntimeError whenever n_proteins % 4 != 0",
    ),

    # ---- v30 : CRISPR designer input width mismatch
    Patch(
        name="CRISPR disease-embedding width",
        file_hint="SESI_GNFS9",
        pattern=(r"disease_emb = disease_emb\.unsqueeze\(0\)\.expand\(mutation_state\.shape\[0\], -1\)\s*\n"
                 r"\s*# Design guides\s*\n"
                 r"\s*combined = torch\.cat\(\[mutation_state, disease_emb\], dim=-1\)"),
        replacement=(
            "disease_emb = disease_emb.unsqueeze(0).expand(mutation_state.shape[0], -1)\n"
            "        # [PATCHED] crispr_designer expects d_mn + n_driver_genes,\n"
            "        # but disease_emb has width len(DISEASE_CATALOG). Project it.\n"
            "        if not hasattr(self, '_disease_proj'):\n"
            "            self._disease_proj = nn.Linear(\n"
            "                disease_emb.shape[-1], self.cfg.d_mn\n"
            "            ).to(disease_emb.device, disease_emb.dtype)\n"
            "        disease_feat = self._disease_proj(disease_emb)\n"
            "        combined = torch.cat([mutation_state, disease_feat], dim=-1)"
        ),
        must_contain="disease_feat = self._disease_proj(disease_emb)",
        note="cat width 200+4 vs expected 320+200 -> RuntimeError",
    ),

    # ---- v30 / iPSC : SSC buffer shape cannot hold a batch
    Patch(
        name="SSC buffer batch broadcast (v30)",
        file_hint="SESI_GNFS9",
        pattern=r'self\.register_buffer\("prev_state", torch\.zeros\(dim\)\)',
        replacement='self.register_buffer("prev_state", torch.zeros(1, dim))  # [PATCHED] batch-safe',
        must_contain='torch.zeros(1, dim))  # [PATCHED] batch-safe',
        note="copy_((B,dim)) into (dim,) fails for B>1",
    ),

    # ---- iPSC : in-place spectral weighting breaks autograd
    Patch(
        name="OPMC spectral weighting out-of-place",
        file_hint="iPSC_ORGAN",
        pattern=(r"chunk_size = self\.d_mn // 5\s*\n"
                 r"\s*for i in range\(5\):\s*\n"
                 r"\s*start = i \* chunk_size\s*\n"
                 r"\s*end = start \+ chunk_size\s*\n"
                 r"\s*Z\[:, start:end\] = Z\[:, start:end\] \* self\.spectral_weights\[i\]"),
        replacement=(
            "# [PATCHED] out-of-place: in-place slice assignment on a graph tensor\n"
            "        # raises a version-counter / leaf-grad error under autograd.\n"
            "        chunk_size = self.d_mn // 5\n"
            "        parts = [Z[:, i*chunk_size:(i+1)*chunk_size] * self.spectral_weights[i]\n"
            "                 for i in range(5)]\n"
            "        Z = torch.cat(parts, dim=-1)"
        ),
        must_contain="Z = torch.cat(parts, dim=-1)",
        note="autograd RuntimeError on every OPMC forward",
    ),

    # ---- MSA : fusion input width is hodge//4, not hodge//8
    Patch(
        name="MSA fusion input width",
        file_hint="MSA_ENHANCED",
        pattern=r"nn\.Linear\(cfg\.hodge_dim // 8 \+ cfg\.esm_dim, cfg\.fused_dim\)",
        replacement=("nn.Linear(cfg.hodge_dim // 4 + cfg.esm_dim, cfg.fused_dim)  "
                     "# [PATCHED] signal = cat(harmonic, exact) = hodge//4"),
        must_contain="cfg.hodge_dim // 4 + cfg.esm_dim",
        note="harmonic(hodge//8) + exact(hodge//8) = hodge//4, not hodge//8",
    ),

    # ---- MSA : hard threshold kills the gradient
    Patch(
        name="MSA delta_min soft gate",
        file_hint="MSA_ENHANCED",
        pattern=r"above_delta_min = \(conservation > self\.threshold\)\.float\(\)",
        replacement=("above_delta_min = torch.sigmoid(  "
                     "# [PATCHED] C^∞ gate; hard compare had zero gradient\n"
                     "            (conservation - self.threshold) / 0.02)"),
        must_contain="above_delta_min = torch.sigmoid(",
        note="hard boolean cast made the filter non-differentiable",
    ),
]


def apply_semantic(src: str, fname: str) -> Tuple[str, List[str], List[str]]:
    applied, skipped = [], []
    for p in PATCHES:
        if p.file_hint != "*" and p.file_hint not in fname:
            continue
        if p.must_contain in src:
            skipped.append(f"{p.name} (already present)")
            continue
        new, n = re.subn(p.pattern, p.replacement, src, flags=p.flags)
        if n and p.must_contain in new:
            applied.append(f"{p.name} ×{n}  [{p.note}]")
            src = new
        else:
            skipped.append(f"{p.name} — PATTERN NOT FOUND (needs manual patch)")
    return src, applied, skipped


# =============================================================================
# PASS 3 — ITERATIVE SYNTAX REPAIR (bare-text lines)
# =============================================================================

SEP_RE = re.compile(r"^[═─━=\-_]{4,}\s*$")


def comment_separators(src: str) -> Tuple[str, int]:
    lines = src.splitlines()
    inside = string_token_lines(src)
    n = 0
    for i, ln in enumerate(lines, start=1):
        if i in inside:
            continue
        if SEP_RE.match(ln.strip()) and not ln.lstrip().startswith("#"):
            lines[i - 1] = "# " + ln
            n += 1
    return "\n".join(lines) + ("\n" if src.endswith("\n") else ""), n


def iterative_syntax_repair(src: str, max_iter: int = 400
                            ) -> Tuple[str, List[Dict], Optional[str]]:
    """
    Comment out lines that Python cannot parse, one at a time, LOGGING EACH.
    Refuses to touch lines inside string tokens.
    """
    commented: List[Dict] = []
    for _ in range(max_iter):
        try:
            ast.parse(src)
            return src, commented, None
        except SyntaxError as e:
            ln = e.lineno or 0
            lines = src.splitlines()
            if not (1 <= ln <= len(lines)):
                return src, commented, f"unfixable SyntaxError at line {ln}: {e.msg}"
            inside = string_token_lines(src)
            if ln in inside:
                return src, commented, (
                    f"SyntaxError inside a string literal at line {ln}: {e.msg}. "
                    f"MANUAL FIX REQUIRED (likely nested triple quotes).")
            original = lines[ln - 1]
            if original.lstrip().startswith("# [AUTO]"):
                return src, commented, f"stalled at line {ln}: {e.msg}"
            lines[ln - 1] = "# [AUTO] " + original
            commented.append({"line": ln, "msg": e.msg, "text": original[:120]})
            src = "\n".join(lines) + "\n"
    return src, commented, "iteration cap reached"


# =============================================================================
# PASS 4/5 — VERIFY
# =============================================================================

def ast_ok(src: str) -> Tuple[bool, str]:
    try:
        ast.parse(src)
        return True, ""
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def run_isolated(path: Path, timeout: int = 180) -> Dict:
    """Import + run the file's own smoke test in a subprocess."""
    code = (
        "import importlib.util, sys, json, traceback\n"
        f"spec = importlib.util.spec_from_file_location('m', r'{path}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "try:\n"
        "    spec.loader.exec_module(m)\n"
        "    ok = True; err = ''\n"
        "except Exception:\n"
        "    ok = False; err = traceback.format_exc()[-1500:]\n"
        "print(json.dumps({'import_ok': ok, 'error': err}))\n"
    )
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=timeout)
        for line in reversed(p.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
        return {"import_ok": False, "error": (p.stderr or p.stdout)[-1500:]}
    except subprocess.TimeoutExpired:
        return {"import_ok": False, "error": f"timeout >{timeout}s"}


# =============================================================================
# DRIVER
# =============================================================================

def collect_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for pat in TARGETS:
        out.extend(sorted(root.glob(pat)))
    return sorted(set(out))


def repair_file(path: Path, apply: bool, smoke: bool) -> Dict:
    original = path.read_text(errors="ignore")
    rec: Dict = {"file": path.name, "bytes_before": len(original)}

    src, mech = mechanical_repair(original)
    rec["mechanical"] = mech

    src, sem, sem_skip = apply_semantic(src, path.name)
    rec["semantic_applied"] = sem
    rec["semantic_needs_manual"] = sem_skip

    src, nsep = comment_separators(src)
    rec["separators_commented"] = nsep

    src, commented, fatal = iterative_syntax_repair(src)
    rec["auto_commented_lines"] = commented
    rec["fatal"] = fatal

    ok, msg = ast_ok(src)
    rec["ast_parse"] = ok
    rec["ast_error"] = msg
    rec["bytes_after"] = len(src)

    if apply and ok:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(src)
        rec["written"] = True
        rec["backup"] = backup.name
        if smoke:
            rec["runtime"] = run_isolated(path)
    elif apply and not ok:
        rec["written"] = False
        rec["reason"] = "refused to write a file that does not parse"
    else:
        rec["written"] = False

    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".")
    p.add_argument("--apply", action="store_true", help="write repairs (default: dry-run)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="import each repaired file")
    p.add_argument("--report", default="repair_report.json")
    a = p.parse_args()

    root = Path(a.dir).resolve()
    files = collect_files(root)
    if not files:
        log.error(f"no target files under {root}")
        print("\nExpected filenames matching:")
        for t in TARGETS:
            print(f"  {t}")
        raise SystemExit(1)

    mode = "APPLY" if a.apply else "DRY-RUN"
    print("=" * 78)
    print(f"  REPAIR AND VERIFY — {mode}")
    print("=" * 78)
    print(f"  root: {root}")
    print(f"  files found: {len(files)}")
    for f in files:
        print(f"    · {f.name}")

    records = []
    for f in files:
        print("\n" + "-" * 78)
        print(f"  {f.name}")
        rec = repair_file(f, a.apply, a.smoke)
        records.append(rec)
        print(f"    mechanical fixes : {len(rec['mechanical'])}")
        for m in rec["mechanical"][:6]:
            print(f"      ✓ {m}")
        print(f"    semantic patches : {len(rec['semantic_applied'])}")
        for m in rec["semantic_applied"]:
            print(f"      ✓ {m}")
        if rec["semantic_needs_manual"]:
            print(f"    NEEDS MANUAL     : {len(rec['semantic_needs_manual'])}")
            for m in rec["semantic_needs_manual"]:
                print(f"      ⚠ {m}")
        print(f"    separators       : {rec['separators_commented']}")
        print(f"    auto-commented   : {len(rec['auto_commented_lines'])}")
        for c in rec["auto_commented_lines"][:5]:
            print(f"      line {c['line']}: {c['text'][:60]}")
        print(f"    AST parse        : {'PASS' if rec['ast_parse'] else 'FAIL ' + rec['ast_error']}")
        if rec.get("fatal"):
            print(f"    FATAL            : {rec['fatal']}")
        if rec.get("written"):
            print(f"    written          : yes (backup: {rec['backup']})")
        if "runtime" in rec:
            r = rec["runtime"]
            print(f"    import           : {'PASS' if r['import_ok'] else 'FAIL'}")
            if not r["import_ok"]:
                print("      " + r["error"].strip().splitlines()[-1][:100])

    n_parse = sum(r["ast_parse"] for r in records)
    n_written = sum(bool(r.get("written")) for r in records)
    n_manual = sum(len(r["semantic_needs_manual"]) for r in records)
    n_runtime = sum(1 for r in records if r.get("runtime", {}).get("import_ok"))

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  files                : {len(records)}")
    print(f"  AST parse            : {n_parse}/{len(records)}")
    print(f"  written              : {n_written}")
    print(f"  import OK            : {n_runtime}")
    print(f"  patches needing hand : {n_manual}")
    Path(a.report).write_text(json.dumps(records, indent=2))
    print(f"  report               : {a.report}")
    print("""
  NOT FIXED BY THIS SCRIPT (and cannot be)
    • CFD solvers            — absent from the corpus
    • Multiphase flow        — absent from the corpus
    • Organoid module        — absent (only a docstring mention)
    • "quantum chemistry"    — learnable Gaussians, not libcint integrals
    • "lattice gauge"        — random links, not Monte Carlo configurations
    • training               — all weights remain random after repair
    • nested triple quotes   — see the manual patch supplied with this script

  A passing AST parse means the file is SYNTACTICALLY valid.
  It does not mean the model is correct, trained, or validated.
""")
    raise SystemExit(0 if n_parse == len(records) else 1)


if __name__ == "__main__":
    main()
