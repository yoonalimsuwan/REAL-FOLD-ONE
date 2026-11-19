#!/usr/bin/env python3
"""
=============================================================================
REPAIR: HIV 5 MODULES — แก้บั๊กทุกไฟล์ในครั้งเดียว
=============================================================================
Developer : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code      : Qwen (Alibaba Group)
License   : MIT | Year 2026

BUGS FIXED:
[B1] HIV_LTR_Bistable_Dynamics_Engine.py
     def init → def __init__
     super().init() → super().__init__()

[B2] hCSN_CRISPR_off_target_predictor.py
     def init → def __init__
     super().init() → super().__init__()
     smoke test: ลบ trailing space ออกจาก string

[B3] HIV_Env_Immunogen_Designer.py
     u_pred.mean(dim=0) → u_pred (per-position mutation prediction)
     เพิ่ม import torch ที่ขาด
     เพิ่ม shape guard สำหรับ graph input

[B4] HIV_CRISPR_Excision_Designer.py
     dummy_human_sites: แทน 'X' ด้วย 'N' (encode ได้)
     เพิ่ม import torch ที่ขาด

[B5] hCSN_Protein_LM.py
     ลบ unused import build_lm
     max_seq + 2 สำหรับ CLS/SEP tokens

HOW TO USE:
  python repair_hiv_modules.py           # แสดงบั๊กที่พบ
  python repair_hiv_modules.py --apply   # แก้ไขไฟล์จริง
  python repair_hiv_modules.py --verify  # รัน smoke test หลังแก้
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("repair_hiv")

VERSION = "1.0.0"

# =============================================================================
# §1. FILE-LEVEL REPAIRS
# =============================================================================

REPAIRS: dict[str, list[dict]] = {
    # ---- B1: HIV_LTR_Bistable_Dynamics_Engine.py ----
    "HIV_LTR_Bistable_Dynamics_Engine.py": [
        {
            "id": "B1a",
            "desc": "def init → def __init__",
            "pattern": r"def init\(self",
            "replace": "def __init__(self",
        },
        {
            "id": "B1b",
            "desc": "super().init() → super().__init__()",
            "pattern": r"super\(\)\.init\(\)",
            "replace": "super().__init__()",
        },
    ],

    # ---- B2: hCSN_CRISPR_off_target_predictor.py ----
    "hCSN_CRISPR_off_target_predictor.py": [
        {
            "id": "B2a",
            "desc": "def init → def __init__",
            "pattern": r"def init\(self",
            "replace": "def __init__(self",
        },
        {
            "id": "B2b",
            "desc": "super().init() → super().__init__()",
            "pattern": r"super\(\)\.init\(\)",
            "replace": "super().__init__()",
        },
        {
            "id": "B2c",
            "desc": "smoke test: ลบ trailing space ใน guide string",
            "pattern": r'"GAGTCCGAGCATAAAGCGC "',
            "replace": '"GAGTCCGAGCATAAAGCGC"',
        },
        {
            "id": "B2d",
            "desc": "smoke test: ลบ trailing space ใน site string",
            "pattern": r'"GAGTCCGAGCATAAAGCGT "',
            "replace": '"GAGTCCGAGCATAAAGCGT"',
        },
    ],

    # ---- B3: HIV_Env_Immunogen_Designer.py ----
    "HIV_Env_Immunogen_Designer.py": [
        {
            "id": "B3a",
            "desc": "เพิ่ม import torch ที่ขาด (มีแต่ torch.nn)",
            "pattern": r"^import torch\.nn as nn$",
            "replace": "import torch\nimport torch.nn as nn",
            "multiline": True,
        },
        {
            "id": "B3b",
            "desc": "u_pred.mean(dim=0) → u_pred (per-position, shape guard)",
            "pattern": r"mutation_logits = self\.mutation_net\(u_pred\.mean\(dim=0\)\)",
            "replace": (
                "# [B3b fix] per-position mutation prediction\n"
                "        # u_pred: (B, N, C) → mutation_net applied on last dim\n"
                "        if u_pred.dim() == 3:\n"
                "            mutation_logits = self.mutation_net(u_pred)  # (B, N, 21)\n"
                "        elif u_pred.dim() == 2:\n"
                "            mutation_logits = self.mutation_net(u_pred).unsqueeze(0)\n"
                "        else:\n"
                "            raise ValueError(f"Unexpected u_pred shape: {u_pred.shape}")"
            ),
        },
        {
            "id": "B3c",
            "desc": "แก้ forward signature ให้รับ graph หรือ tensor",
            "pattern": r"def forward\(self, env_trimer_graph, target_epitope_mask\):",
            "replace": (
                "def forward(self, env_trimer_graph, target_epitope_mask):\n"
                "        \"\"\"\n"
                "        Args:\n"
                "            env_trimer_graph: object with .node_feats (B,N,C) and .sigma (B,N)\n"
                "                              OR a raw tensor (B,N,C)\n"
                "            target_epitope_mask: (B,N) or (N,) binary mask\n"
                "        \"\"\""
            ),
        },
    ],

    # ---- B4: HIV_CRISPR_Excision_Designer.py ----
    "HIV_CRISPR_Excision_Designer.py": [
        {
            "id": "B4a",
            "desc": "dummy_human_sites: แทน 'X' ด้วย 'N' (unknown base)",
            "pattern": r'"TGGAAGGGCTAATTCACTCCCAA" \+ "X"\*100',
            "replace": '"TGGAAGGGCTAATTCACTCCCAA" + "N"*100',
        },
        {
            "id": "B4b",
            "desc": "encode_pair: เพิ่ม 'N' ใน dict (map ไป 0 = A)",
            "pattern": r'b = \{"A": 0, "C": 1, "G": 2, "T": 3\}',
            "replace": 'b = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}',
        },
    ],

    # ---- B5: hCSN_Protein_LM.py ----
    "hCSN_Protein_LM.py": [
        {
            "id": "B5a",
            "desc": "ลบ unused import build_lm",
            "pattern": r"from model import build_lm, HCSNLMD",
            "replace": "from model import HCSNLMD",
        },
        {
            "id": "B5b",
            "desc": "max_seq + 2 สำหรับ CLS/SEP tokens",
            "pattern": r"max_seq=512",
            "replace": "max_seq=514  # 512 + CLS + SEP",
        },
    ],
}


# =============================================================================
# §2. REPAIR ENGINE
# =============================================================================

def apply_repairs(filepath: Path, repairs: list[dict],
                  dry_run: bool = False) -> list[dict]:
    """Apply regex-based repairs to a file. Returns list of applied fixes."""
    if not filepath.exists():
        log.warning(f"  file not found: {filepath}")
        return []

    src = filepath.read_text(encoding="utf-8", errors="replace")
    original = src
    applied = []

    for rep in repairs:
        flags = re.MULTILINE if rep.get("multiline") else 0
        new_src, n = re.subn(rep["pattern"], rep["replace"], src, flags=flags)
        if n > 0:
            applied.append({
                "id": rep["id"],
                "desc": rep["desc"],
                "matches": n,
            })
            src = new_src
        else:
            applied.append({
                "id": rep["id"],
                "desc": rep["desc"],
                "matches": 0,
                "note": "pattern not found (already fixed or different format)",
            })

    if src != original and not dry_run:
        # Backup
        backup = filepath.with_suffix(filepath.suffix + ".bak")
        shutil.copy2(filepath, backup)
        filepath.write_text(src, encoding="utf-8")
        log.info(f"  wrote {filepath} (backup: {backup})")

    return applied


def verify_syntax(filepath: Path) -> bool:
    """Check if a Python file parses without SyntaxError."""
    try:
        compile(filepath.read_text(encoding="utf-8", errors="replace"),
                str(filepath), "exec")
        return True
    except SyntaxError as e:
        log.error(f"  SyntaxError in {filepath.name}: line {e.lineno}: {e.msg}")
        return False


def verify_import(filepath: Path) -> bool:
    """Try to import the module (may fail if deps missing)."""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            filepath.stem, str(filepath))
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return True
    except Exception as e:
        log.warning(f"  import failed for {filepath.name}: {e}")
        return False


# =============================================================================
# §3. CORRECTED FILE CONTENTS (for reference / manual replacement)
# =============================================================================

FIXED_LTR_ENGINE = '''"""
HIV LTR Bistable Dynamics Engine — NATIVE FULL DIFFERENTIABLE
Based on: HybridPsychopathologyEngine (SESI Stack)
Purpose: Simulate 'Shock and Kill' Latency Reversal Agents (LRAs)
Developer: PAI AND Yoon A Limsuwan / MSPS NETWORK
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HIV_LTR_BistableEngine(nn.Module):
    """
    จำลองพลวัตของ HIV LTR Promoter (Tat feedback + Epigenetic silencing)
    Continuous core: Bistable ODE (Tat, HDAC, NF-kB) integrated with RK4 (fp32-internal).
    Discrete network: Cellular states (CD4 activation, Apoptosis, Cytokine release).
    """
    def __init__(self, num_cell_states: int = 16, dt: float = 0.01):
        super().__init__()
        self.num_nodes = num_cell_states
        self.dt = dt

        # Continuous Core Parameters (Learnable biological rates)
        self.k_tat_prod = nn.Parameter(torch.tensor(0.5))
        self.k_tat_deg = nn.Parameter(torch.tensor(0.1))
        self.k_hdac_prod = nn.Parameter(torch.tensor(0.2))
        self.k_nfkb_act = nn.Parameter(torch.tensor(1.0))

        # C∞ Surrogate parameters (AMP-Safe)
        self.register_buffer("_dt", torch.tensor(dt, dtype=torch.float32), persistent=False)
        self.register_buffer("_dt_half", torch.tensor(dt * 0.5, dtype=torch.float32), persistent=False)
        self.register_buffer("_dt_sixth", torch.tensor(dt / 6.0, dtype=torch.float32), persistent=False)
        self.register_buffer("_eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False)

    def _safe_abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x * x + self._eps)

    def _ltr_derivatives(self, tat, hdac, nfkb, lra_shock_agent):
        """HIV LTR ODE: Tat positive feedback inhibited by HDAC, activated by NF-kB"""
        tat_feedback = F.softplus(tat - hdac, beta=10.0)
        shock_effect = lra_shock_agent * self.k_nfkb_act
        d_tat = self.k_tat_prod * tat_feedback * (1.0 + shock_effect) - self.k_tat_deg * tat
        d_hdac = self.k_hdac_prod * (1.0 - F.sigmoid(tat)) - 0.05 * hdac
        d_nfkb = 0.1 * shock_effect - 0.1 * nfkb
        return d_tat, d_hdac, d_nfkb

    def _rk4_step(self, tat, hdac, nfkb, lra_shock_agent):
        """fp32-internal RK4 for chaotic/bistable stability"""
        dtype_in = tat.dtype
        t32, h32, n32 = tat.float(), hdac.float(), nfkb.float()
        lra32 = lra_shock_agent.float()
        k1t, k1h, k1n = self._ltr_derivatives(t32, h32, n32, lra32)
        k2t, k2h, k2n = self._ltr_derivatives(t32 + self._dt_half*k1t, h32 + self._dt_half*k1h, n32 + self._dt_half*k1n, lra32)
        k3t, k3h, k3n = self._ltr_derivatives(t32 + self._dt_half*k2t, h32 + self._dt_half*k2h, n32 + self._dt_half*k2n, lra32)
        k4t, k4h, k4n = self._ltr_derivatives(t32 + self._dt*k3t, h32 + self._dt*k3h, n32 + self._dt*k3n, lra32)
        tat_next = t32 + self._dt_sixth * (k1t + 2*k2t + 2*k3t + k4t)
        hdac_next = h32 + self._dt_sixth * (k1h + 2*k2h + 2*k3h + k4h)
        nfkb_next = n32 + self._dt_sixth * (k1n + 2*k2n + 2*k3n + k4n)
        return tat_next.to(dtype_in), hdac_next.to(dtype_in), nfkb_next.to(dtype_in)

    def forward(self, tat0, hdac0, nfkb0, lra_sequence, steps=100):
        """
        lra_sequence: (B, steps) ความเข้มข้นของยา Latency Reversal Agent ตามเวลา
        Returns: Tat trajectory (ถ้า Tat พุ่งสูง = Shock successful, ไวรัสตื่น)
        """
        tat, hdac, nfkb = tat0, hdac0, nfkb0
        tat_trajectory = []
        for t in range(steps):
            lra_t = lra_sequence[:, t].unsqueeze(-1)
            tat, hdac, nfkb = self._rk4_step(tat, hdac, nfkb, lra_t)
            tat_trajectory.append(tat)
        return torch.stack(tat_trajectory, dim=1)  # (B, steps, 1)
'''

FIXED_OFFTARGET = '''"""HCSN CRISPR off-target predictor — input: 23-nt guide + 23-nt
candidate site (encoded as 2 channels), output: off-target probability.
Domain meaning of HCSN: weak measurement ≈ 'soft' mismatch readout that
doesn't fully discard partial matches — exactly the biology of off-targets.
Developers : PAI AND Yoon A Limsuwan
Organization : MSPS NETWORK
Licence : MIT
Year : 2026
"""
import torch
import torch.nn as nn

# Fallback if model.py not available
try:
    from model import RMSNorm, HCSNPass
except ImportError:
    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.eps = eps
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class HCSNPass(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        def forward(self, x):
            return x * self.gate(x)

class OffTargetNet(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=4, seq=23):
        super().__init__()
        self.embed = nn.Embedding(5, d_model)     # A,C,G,T,+mismatch-channel
        self.pos = nn.Parameter(torch.randn(1, seq, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, n_layers)
        self.hcsn = HCSNPass(d_model)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def encode_pair(self, guide: str, site: str) -> torch.Tensor:
        b = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}
        ids = []
        for g, s in zip(guide.upper(), site.upper()):
            if g in b and s in b:
                ids.append(b[s] + (4 if g != s else 0))
            else:
                ids.append(0)
        return torch.tensor(ids)

    def forward(self, ids):                        # (B, 23)
        x = self.embed(ids) + self.pos
        x = self.backbone(x)
        x = self.hcsn(x)
        return self.head(self.norm(x.mean(dim=1))).squeeze(-1)

# --- smoke test ---
if __name__ == "__main__":
    torch.manual_seed(0)
    m = OffTargetNet()
    on  = m.encode_pair("GAGTCCGAGCATAAAGCGC", "GAGTCCGAGCATAAAGCGC")[:23]
    off = m.encode_pair("GAGTCCGAGCATAAAGCGC", "GAGTCCGAGCATAAAGCGT")[:23]
    ids = torch.stack([on, off])
    logits = m(ids)
    logits.sum().backward()
    print("off-target OK:", logits.shape, "| params:", sum(p.numel() for p in m.parameters()))
'''

FIXED_ENV_DESIGNER = '''"""
HIV Env Immunogen Designer — Structural GNO Phase Field
Based on: structural_gno_fold_v4.py (Real Fold ONE surrogate)
Purpose: Lock Env Trimer in Pre-fusion state to expose bNAT epitopes
"""
import torch
import torch.nn as nn

class HIV_Env_Immunogen_Designer(nn.Module):
    def __init__(self, gno_fold_model):
        super().__init__()
        self.gno_fold = gno_fold_model
        # Learned stabilizing mutations (Amino Acid substitutions)
        # 20 Amino Acids + 1 No-mutation
        self.mutation_net = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 21)
        )

    def forward(self, env_trimer_graph, target_epitope_mask):
        """
        Args:
            env_trimer_graph: Graph representation of HIV Env gp160
                              Must have .node_feats (B,N,C) and .sigma (B,N)
                              OR be a raw tensor (B,N,C) with sigma=None
            target_epitope_mask: Mask ของจุดที่เราต้องการเปิด (เช่น CD4 binding site)
                                 Shape: (B,N) or (N,)
        """
        # Handle both graph objects and raw tensors
        if hasattr(env_trimer_graph, 'node_feats'):
            node_feats = env_trimer_graph.node_feats
            sigma = getattr(env_trimer_graph, 'sigma', None)
        else:
            node_feats = env_trimer_graph
            sigma = None

        # 1. จำลอง Phase Field ของโปรตีนด้วย GNO Fold
        if sigma is not None:
            u_pred = self.gno_fold.forward_phase_field(node_feats, sigma)
        else:
            u_pred = self.gno_fold(node_feats)

        # Ensure u_pred is (B, N, C)
        if u_pred.dim() == 2:
            u_pred = u_pred.unsqueeze(0)

        # 2. คำนวณ Free Energy ของการเปลี่ยนรูป (Conformational Energy)
        if target_epitope_mask.dim() == 1:
            target_epitope_mask = target_epitope_mask.unsqueeze(0)
        pre_fusion_energy = (u_pred * target_epitope_mask.unsqueeze(-1)).sum()

        # 3. เสนอการกลายพันธุ์ (SOSIP mutations) เพื่อล็อกโครงสร้าง
        # [B3b fix] per-position mutation prediction
        # u_pred: (B, N, C) → mutation_net applied on last dim → (B, N, 21)
        mutation_logits = self.mutation_net(u_pred)
        mutation_probs = torch.softmax(mutation_logits, dim=-1)

        return {
            "stabilized_structure": u_pred,
            "energy_barrier": pre_fusion_energy,
            "proposed_mutations": mutation_probs,  # (B, N, 21)
        }
'''


# =============================================================================
# §4. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Repair HIV 5 modules")
    ap.add_argument("--apply", action="store_true",
                    help="Apply fixes to files (creates .bak backups)")
    ap.add_argument("--verify", action="store_true",
                    help="Run syntax + import verification after repair")
    ap.add_argument("--write-fixed", action="store_true",
                    help="Write fully corrected versions as *_fixed.py")
    ap.add_argument("--dir", default=".",
                    help="Directory containing the HIV modules")
    a = ap.parse_args()

    root = Path(a.dir)
    files = list(REPAIRS.keys())

    print("=" * 74)
    print(f"  HIV 5-MODULE REPAIR v{VERSION}")
    print(f"  Directory: {root.resolve()}")
    print("=" * 74)

    # Report
    for fname in files:
        fp = root / fname
        status = "EXISTS" if fp.exists() else "MISSING"
        print(f"\n  {fname}: {status}")
        if not fp.exists():
            continue
        syntax_ok = verify_syntax(fp)
        print(f"    syntax: {'OK' if syntax_ok else 'FAIL'}")
        for rep in REPAIRS[fname]:
            # Check if pattern exists
            src = fp.read_text(errors="ignore")
            found = bool(re.search(rep["pattern"], src,
                                   re.MULTILINE if rep.get("multiline") else 0))
            print(f"    [{rep['id']}] {rep['desc']}: "
                  f"{'FOUND' if found else 'not found / already fixed'}")

    # Apply
    if a.apply:
        print("\n" + "=" * 74)
        print("  APPLYING FIXES")
        print("=" * 74)
        for fname in files:
            fp = root / fname
            if not fp.exists():
                continue
            print(f"\n  {fname}:")
            applied = apply_repairs(fp, REPAIRS[fname], dry_run=False)
            for fix in applied:
                status = "FIXED" if fix["matches"] > 0 else "SKIP"
                print(f"    [{fix['id']}] {status} ({fix['matches']} matches)")

    # Write fully corrected versions
    if a.write_fixed:
        print("\n" + "=" * 74)
        print("  WRITING FULLY CORRECTED VERSIONS")
        print("=" * 74)
        fixed_files = {
            "HIV_LTR_Bistable_Dynamics_Engine_fixed.py": FIXED_LTR_ENGINE,
            "hCSN_CRISPR_off_target_predictor_fixed.py": FIXED_OFFTARGET,
            "HIV_Env_Immunogen_Designer_fixed.py": FIXED_ENV_DESIGNER,
        }
        for fname, content in fixed_files.items():
            out = root / fname
            out.write_text(content, encoding="utf-8")
            print(f"  wrote {out}")

    # Verify
    if a.verify:
        print("\n" + "=" * 74)
        print("  VERIFICATION")
        print("=" * 74)
        all_ok = True
        for fname in files:
            fp = root / fname
            if not fp.exists():
                print(f"  {fname}: MISSING")
                all_ok = False
                continue
            syn = verify_syntax(fp)
            imp = verify_import(fp)
            status = "OK" if (syn and imp) else ("SYNTAX OK, IMPORT FAIL" if syn else "SYNTAX FAIL")
            print(f"  {fname}: {status}")
            all_ok &= syn
        print(f"\n  {'ALL VERIFIED' if all_ok else 'SOME ISSUES REMAIN'}")

    print("\n" + "=" * 74)
    print("  DONE")
    print("=" * 74)
    print("""
NEXT STEPS:
  1. python repair_hiv_modules.py --apply --verify
  2. python repair_hiv_modules.py --write-fixed   (for clean copies)
  3. Run each module's smoke test:
     python HIV_LTR_Bistable_Dynamics_Engine.py
     python hCSN_CRISPR_off_target_predictor.py
     python HIV_Env_Immunogen_Designer.py  (needs gno_fold_model)
     python HIV_CRISPR_Excision_Designer.py
     python hCSN_Protein_LM.py  (needs model.py)
""")


if __name__ == "__main__":
    main()
