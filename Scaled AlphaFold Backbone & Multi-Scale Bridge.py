#!/usr/bin/env python3
"""
=============================================================================
AF-SCALED BACKBONE: Unified Multi-Scale Protein & Organoid Engine
=============================================================================
Integrates:
  1. evoformer_fape_2.py  -> Scaled Evoformer Trunk (AMP + Checkpointing)
  2. af2_losses_2.py      -> Joint FAPE + Distogram + pLDDT + Violation Loss
  3. msa_pipeline.py      -> Real MSA / Neff weighting integration
  4. protein_qm_2.py      -> Structural handoff to Metal-Site DFT/FMO2
  5. MDOE Predictor       -> Structural embedding bridge for T1 Potency

License: MIT | 2026
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, math, os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader

# Import existing modules from the workspace
from evoformer_fape_2 import (
    EvoformerBlock, StructureHead, frames_from_backbone,
    load_structure, load_a3m, fetch_pdb, AA_IDX, BB_ATOMS
)
from af2_losses_2 import AF2Loss, LossConfig
from msa_pipeline import MSAConfig, parse_a3m, msa_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
log = logging.getLogger("af_scaled")


# =============================================================================
# 1. SCALED CONFIGURATION
# =============================================================================

@dataclass
class ScaledAFConfig:
    # Model dimensions (Scaled up from toy defaults)
    c_m: int = 256
    c_z: int = 128
    n_blocks: int = 12
    h_msa: int = 8
    h_tri: int = 4
    n_recycle: int = 3
    max_seq_len: int = 512
    max_msa_depth: int = 256

    # Training & Memory Scaling
    use_amp: bool = True
    amp_dtype: str = "bfloat16"      # bfloat16 preferred on Ampere+ GPUs
    use_checkpointing: bool = True   # Trade compute for O(1) block activation memory
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad: float = 1.0

    # Loss configuration
    loss_cfg: LossConfig = LossConfig()


# =============================================================================
# 2. SCALED EVOFORMER ARCHITECTURE (With Auxiliary Heads & Checkpointing)
# =============================================================================

class ScaledEvoformerAF(nn.Module):
    """
    Production-scaled Evoformer backbone supporting:
      - Gradient checkpointing across Evoformer blocks
      - Mixed-precision stability (FP32 LayerNorms/Losses)
      - Full auxiliary heads (Distogram & pLDDT) for af2_losses_2.py
      - Downstream feature extraction for QM and MDOE T1
    """
    def __init__(self, cfg: ScaledAFConfig):
        super().__init__()
        self.cfg = cfg
        self.aa_embed = nn.Embedding(22, cfg.c_m)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1, cfg.max_seq_len, cfg.c_m) * 0.02
        )
        self.z_init = nn.Linear(2 * cfg.c_m + 1, cfg.c_z)

        # Deep Evoformer Stack
        self.blocks = nn.ModuleList([
            EvoformerBlock(cfg.c_m, cfg.c_z, cfg.h_msa, cfg.h_tri)
            for _ in range(cfg.n_blocks)
        ])

        # Recycling projections
        self.recycle_m = nn.LayerNorm(cfg.c_m)
        self.recycle_z = nn.LayerNorm(cfg.c_z)
        self.prev_m = nn.Linear(cfg.c_m, cfg.c_m)
        self.prev_z = nn.Linear(cfg.c_z, cfg.c_z)

        # Primary Structure Head
        self.structure_head = StructureHead(cfg.c_m, cfg.c_z)

        # Auxiliary Heads (Required for af2_losses_2.py)
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(cfg.c_z),
            nn.Linear(cfg.c_z, cfg.loss_cfg.n_dist_bins)
        )
        self.plddt_head = nn.Sequential(
            nn.LayerNorm(cfg.c_m),
            nn.Linear(cfg.c_m, cfg.c_m // 2),
            nn.GELU(),
            nn.Linear(cfg.c_m // 2, cfg.loss_cfg.n_plddt_bins)
        )

    def _run_block(self, blk: nn.Module, m: Tensor, z: Tensor,
                   msa_mask: Tensor, res_mask: Tensor) -> Tuple[Tensor, Tensor]:
        if self.cfg.use_checkpointing and self.training:
            return checkpoint(blk, m, z, msa_mask, res_mask, use_reentrant=False)
        return blk(m, z, msa_mask, res_mask)

    def forward(self, msa_tok: Tensor, res_mask: Tensor,
                msa_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        B, S, N = msa_tok.shape
        if msa_mask is None:
            msa_mask = (msa_tok != 21).float() * res_mask[:, None, :]

        # Initial embeddings
        m0 = self.aa_embed(msa_tok) + self.pos_embed[:, :, :N]
        q = m0[:, 0]
        rel = torch.arange(N, device=msa_tok.device)
        d = (rel[:, None] - rel[None, :]).float().clamp(-32, 32) / 32.0

        z0 = self.z_init(torch.cat([
            q[:, :, None].expand(B, N, N, -1),
            q[:, None, :].expand(B, N, N, -1),
            d[None, ..., None].expand(B, N, N, 1)
        ], dim=-1))

        frames = torch.eye(4, device=msa_tok.device, dtype=torch.float32)
        frames = frames[None, None].expand(B, N, 4, 4).clone()

        pm, pz = None, None
        n_it = (
            self.cfg.n_recycle if not self.training
            else int(torch.randint(0, self.cfg.n_recycle + 1, (1,)).item())
        )

        # Recycling loop (Gradients enabled only on final pass)
        for it in range(n_it + 1):
            is_final = (it == n_it)
            with torch.set_grad_enabled(self.training and is_final):
                mi, zi = m0, z0
                if pm is not None:
                    mi = mi + self.prev_m(self.recycle_m(pm))[:, None]
                    zi = zi + self.prev_z(self.recycle_z(pz))

                m, z = mi, zi
                for blk in self.blocks:
                    m, z = self._run_block(blk, m, z, msa_mask, res_mask)

                # Keep structure coordinates in FP32 for geometric precision
                with torch.amp.autocast(device_type=m.device.type, enabled=False):
                    frames, pos, single_h = self.structure_head(
                        m.float(), z.float(), frames.float()
                    )

                pm = m[:, 0].detach()
                pz = z.detach()

        # Symmetrize pair representation for distogram
        z_sym = 0.5 * (z + z.transpose(1, 2))
        distogram_logits = self.distogram_head(z_sym)
        plddt_logits = self.plddt_head(single_h)

        # Expected pLDDT score [0, 100]
        binw = 100.0 / self.cfg.loss_cfg.n_plddt_bins
        centers = (torch.arange(self.cfg.loss_cfg.n_plddt_bins, device=m.device).float() + 0.5) * binw
        plddt_score = (F.softmax(plddt_logits.float(), dim=-1) * centers).sum(-1)

        return {
            "frames": frames,
            "positions": pos,
            "distogram_logits": distogram_logits,
            "plddt_logits": plddt_logits,
            "plddt": plddt_score,
            "single_repr": single_h,       # Ready for MDOE T1 / downstream
            "pair_repr": z_sym,            # Ready for contact/interface analysis
            "n_recycle": n_it
        }


# =============================================================================
# 3. STREAMING DATASET FOR LARGE-SCALE PDB + MSA
# =============================================================================

class ProteinStreamingDataset(Dataset):
    """
    Lazy-loading dataset for scaling to thousands of PDB + A3M pairs
    without exhausting system RAM.
    """
    def __init__(self, manifest: List[Dict[str, str]], max_len: int = 512,
                 max_msa_depth: int = 256):
        self.manifest = manifest
        self.max_len = max_len
        self.max_msa_depth = max_msa_depth

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = self.manifest[idx]
        pdb_path = Path(item["pdb_path"])
        if not pdb_path.exists():
            return None

        st = load_structure(pdb_path, chain=item.get("chain"), max_len=self.max_len)
        if st is None:
            return None

        a3m_path = Path(item["a3m_path"]) if item.get("a3m_path") else None
        msa = None
        if a3m_path and a3m_path.exists():
            msa = load_a3m(a3m_path, max_depth=self.max_msa_depth, max_len=st["length"])

        if msa is None:
            msa = np.array([[AA_IDX.get(c, 20) for c in st["sequence"]]], dtype=np.int64)

        # Random MSA subsampling during training for regularization
        if msa.shape[0] > 1:
            n_keep = min(msa.shape[0], self.max_msa_depth)
            perm = np.concatenate([[0], np.random.permutation(np.arange(1, msa.shape[0]))[:n_keep - 1]])
            msa = msa[perm]

        st["msa"] = msa
        return st


def scaled_collate(batch: List[Optional[Dict]]) -> Optional[Dict[str, Tensor]]:
    valid = [b for b in batch if b is not None]
    if not valid:
        return None

    L = max(b["length"] for b in valid)
    S = max(b["msa"].shape[0] for b in valid)
    B = len(valid)

    tok = torch.full((B, S, L), 21, dtype=torch.long)
    rm = torch.zeros(B, L, dtype=torch.float32)
    am = torch.zeros(B, L, 4, dtype=torch.float32)
    at = torch.zeros(B, L, 4, 3, dtype=torch.float32)

    for i, b in enumerate(valid):
        n, s = b["length"], b["msa"].shape[0]
        tok[i, :s, :n] = torch.from_numpy(b["msa"])
        rm[i, :n] = torch.from_numpy(b["res_mask"])
        am[i, :n] = torch.from_numpy(b["atom_mask"])
        at[i, :n] = torch.from_numpy(b["atoms"])

    return {"msa_tok": tok, "res_mask": rm, "atom_mask": am, "atoms": at}


# =============================================================================
# 4. MULTI-SCALE BRIDGES (AF -> QM & AF -> MDOE)
# =============================================================================

class AFToMDOEBridge(nn.Module):
    """
    Replaces flat sequence encoding in MDOE T1Potency with rich 3D-aware
    structural representations extracted from the ScaledEvoformerAF backbone.
    """
    def __init__(self, af_model: ScaledEvoformerAF, out_dim: int = 128,
                 freeze_af: bool = True):
        super().__init__()
        self.af = af_model
        if freeze_af:
            for p in self.af.parameters():
                p.requires_grad = False

        c_m = af_model.cfg.c_m
        c_z = af_model.cfg.c_z
        # Pool single (mean + max) + pair global summary + mean pLDDT
        self.proj = nn.Sequential(
            nn.LayerNorm(c_m * 2 + c_z + 1),
            nn.Linear(c_m * 2 + c_z + 1, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, msa_tok: Tensor, res_mask: Tensor) -> Dict[str, Tensor]:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.af.parameters())):
            af_out = self.af(msa_tok, res_mask)

        mask_exp = res_mask.unsqueeze(-1)
        denom = mask_exp.sum(dim=1).clamp(min=1.0)

        single = af_out["single_repr"]
        s_mean = (single * mask_exp).sum(dim=1) / denom
        s_max = single.masked_fill(mask_exp == 0, -1e9).max(dim=1)[0]

        pair = af_out["pair_repr"]
        p_mask = res_mask[:, :, None, None] * res_mask[:, None, :, None]
        p_mean = (pair * p_mask).sum(dim=(1, 2)) / p_mask.sum(dim=(1, 2)).clamp(min=1.0)

        plddt_mean = ((af_out["plddt"] * res_mask).sum(dim=1) / denom.squeeze(-1)).unsqueeze(-1) / 100.0

        feat = torch.cat([s_mean, s_max, p_mean, plddt_mean], dim=-1)
        return {
            "seq_feat": self.proj(feat),      # Drop-in replacement for MDOE T1 input
            "mean_plddt": plddt_mean * 100.0,
            "predicted_atoms": af_out["positions"]
        }


def export_af_structure_for_qm(sequence: str, positions: Tensor, plddt: Tensor,
                               out_pdb: Path, plddt_cutoff: float = 70.0) -> Dict:
    """
    Exports predicted backbone coordinates from AF into a PDB file with
    pLDDT stored in the B-factor column, ready for `protein_qm_2.py`
    metal-site extraction and QM/MM cluster building.
    """
    AA1to3 = {
        "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
        "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
        "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"
    }
    pos_np = positions.detach().cpu().numpy()
    plddt_np = plddt.detach().cpu().numpy()
    lines = []
    serial = 1
    high_conf_res = 0

    for i, aa in enumerate(sequence):
        resn = AA1to3.get(aa, "UNK")
        conf = float(plddt_np[i])
        if conf >= plddt_cutoff:
            high_conf_res += 1
        for j, atom_name in enumerate(BB_ATOMS):
            x, y, z = pos_np[i, j]
            elem = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d}  {atom_name:<3s} {resn:3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{conf:6.2f}           {elem:>2s}"
            )
            serial += 1
    lines.append("END")
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text("\n".join(lines) + "\n")

    return {
        "pdb_path": str(out_pdb),
        "length": len(sequence),
        "high_conf_fraction": high_conf_res / max(len(sequence), 1),
        "ready_for_qm": (high_conf_res / max(len(sequence), 1)) >= 0.7
    }


# =============================================================================
# 5. SCALED TRAINING LOOP (AMP + Gradient Accumulation + AF2Loss)
# =============================================================================

def train_scaled(model: ScaledEvoformerAF, loader: DataLoader,
                 epochs: int, out_dir: str, device: torch.device):
    cfg = model.cfg
    criterion = AF2Loss(cfg.loss_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(cfg.use_amp and amp_dtype == torch.float16))

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    log_file = open(outp / "train_scaled.jsonl", "w")

    model.to(device)
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0

        for step, batch in enumerate(loader):
            if batch is None:
                continue
            msa_tok = batch["msa_tok"].to(device, non_blocking=True)
            res_mask = batch["res_mask"].to(device, non_blocking=True)
            atom_mask = batch["atom_mask"].to(device, non_blocking=True)
            true_atoms = batch["atoms"].to(device, non_blocking=True)
            true_frames = frames_from_backbone(true_atoms)

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=cfg.use_amp
            ):
                pred = model(msa_tok, res_mask)

            # Compute geometric & stereochemical losses in FP32
            with torch.amp.autocast(device_type=device.type, enabled=False):
                true_dict = {
                    "frames": true_frames.float(),
                    "atoms": true_atoms.float(),
                    "atom_mask": atom_mask.float()
                }
                losses = criterion(pred, true_dict, mask=res_mask.float())
                loss = losses["total"] / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(losses["total"].detach())
            steps += 1

        scheduler.step()
        avg_loss = running_loss / max(steps, 1)
        rec = {"epoch": ep + 1, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()
        log.info(f"Epoch {ep+1:03d}/{epochs:03d} | Loss: {avg_loss:.4f} | LR: {rec['lr']:.2e}")

    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, outp / "af_scaled.pt")
    log_file.close()


# =============================================================================
# 6. SELF-TEST & VERIFICATION
# =============================================================================

def selftest():
    print("=" * 76)
    print("  AF-SCALED BACKBONE & MULTI-SCALE BRIDGE — self-test")
    print("=" * 76)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ScaledAFConfig(
        c_m=64, c_z=32, n_blocks=3, h_msa=4, h_tri=4,
        n_recycle=1, max_seq_len=64, use_amp=False
    )
    model = ScaledEvoformerAF(cfg).to(device)
    print(f"[T1] ScaledEvoformerAF initialized ({sum(p.numel() for p in model.parameters()):,} params)")

    B, S, L = 2, 8, 24
    msa_tok = torch.randint(0, 20, (B, S, L), device=device)
    res_mask = torch.ones(B, L, device=device)
    true_atoms = torch.randn(B, L, 4, 3, device=device)
    true_frames = frames_from_backbone(true_atoms)

    model.train()
    pred = model(msa_tok, res_mask)
    criterion = AF2Loss(cfg.loss_cfg).to(device)
    losses = criterion(pred, {"frames": true_frames, "atoms": true_atoms}, mask=res_mask)
    losses["total"].backward()

    grads_ok = all(
        torch.isfinite(p.grad).all()
        for p in model.parameters() if p.grad is not None
    )
    print(f"[T2] Joint AF2Loss (FAPE+Dist+pLDDT+Viol) = {losses['total'].item():.4f} | Grads finite: {grads_ok}")
    assert grads_ok

    # Test Bridge to MDOE T1
    bridge = AFToMDOEBridge(model, out_dim=128, freeze_af=True).to(device)
    mdoe_in = bridge(msa_tok, res_mask)
    assert mdoe_in["seq_feat"].shape == (B, 128)
    print(f"[T3] AF -> MDOE T1 Bridge output shape: {tuple(mdoe_in['seq_feat'].shape)} | Mean pLDDT: {mdoe_in['mean_plddt'].mean():.1f}")

    # Test Export to QM
    tmp_pdb = Path("_test_af_to_qm.pdb")
    qm_meta = export_af_structure_for_qm("A" * L, pred["positions"][0], pred["plddt"][0], tmp_pdb)
    assert tmp_pdb.exists()
    tmp_pdb.unlink(missing_ok=True)
    print(f"[T4] AF -> QM PDB export verified (ready_for_qm={qm_meta['ready_for_qm']})")

    print("=" * 76)
    print("  ✅ ALL SCALING & BRIDGE TESTS PASSED")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["selftest"], default="selftest", nargs="?")
    args = parser.parse_args()
    selftest()
```

### สิ่งที่สคริปต์นี้เพิ่มเข้ามาเพื่อขยายสเกล:
1. **รองรับการฝึกขนาดใหญ่ (`ScaledEvoformerAF`):** เปิดใช้งาน **Gradient Checkpointing** และ **Automatic Mixed Precision (AMP)** ทำให้คุณสามารถเพิ่มจำนวนบล็อก (`n_blocks`) และความยาวลำดับกรดอะมิโน (`max_seq_len`) ได้โดยไม่ติดปัญหาหน่วยความจำ GPU เต็ม และยังคงความแม่นยำทางเรขาคณิตของโครงสร้างในระดับ FP32
2. **เชื่อมต่อฟังก์ชันสูญเสียครบวงจร (`AF2Loss`):** เพิ่ม `distogram_head` และ `plddt_head` เข้าไปในโครงสร้าง Evoformer เพื่อคำนวณ Loss ทั้ง 4 ส่วน (FAPE, Distogram, pLDDT, และ Violation) พร้อมกัน
3. **สะพานเชื่อมข้ามสเกล (`AFToMDOEBridge` และ `export_af_structure_for_qm`):**
   * ดึง Embedding เชิงโครงสร้าง 3 มิติ (`single_repr`, `pair_repr`, `plddt`) ส่งตรงไปยังโมเดลทำนายความแรงของลิแกนด์ (**T1Potency**) ใน `Morphogen_Directed_Organoid_Engineering_Predictor.py`
   * ส่งออกพิกัดโครงสร้างพร้อมค่าความเชื่อมั่นรายกรดอะมิโนไปเป็นไฟล์ PDB เพื่อส่งต่อให้ **`protein_qm_2.py`** คำนวณระดับควอนตัมต่อไป

หากต้องการปรับแต่งขนาดพารามิเตอร์สำหรับคลัสเตอร์ GPU เฉพาะรุ่น หรือเขียนสคริปต์สร้างไฟล์ `manifest.json` เพื่อดึงข้อมูล PDB และ A3M จำนวนมากเข้าสู่ `ProteinStreamingDataset` แจ้งได้เลยครับ!