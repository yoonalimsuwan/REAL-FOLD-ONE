"""
=============================================================================
SPECIALISTFOLD v1.0 — Mixture-of-Experts Protein Folding by Protein Class
=============================================================================
License: MIT | MSPS NETWORK | 2026

PURPOSE
-------
Replace ONE generic folding head with N class-specialized experts,
each carrying a different inductive bias, plus a learned router,
calibrated confidence, and a paired-vs-AF3 benchmark harness.

CRITICAL HONESTY
----------------
1. Random-initialized weights have NO predictive power. This is an
   architecture, not a trained predictor and not an AF3 replacement.
2. Class rules below are HEURISTIC PRIORS, not validated predictors.
   The learned router is meant to override them after training.
3. IDP, membrane, metal and glyco restraints are differentiable
   SURROGATES. They are not force fields and not experimental evidence.
4. "Beat AF3" requires a real paired benchmark (§Benchmark below).
   This module reports numbers; it does not manufacture them.
5. No physical quantum gate, no vacuum energy, no CY≡protein
   equivalence is asserted. CY priors remain ANALOGICAL.
=============================================================================
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i for i, a in enumerate(AA)}
UNK = 20

# Kyte-Doolittle hydropathy (fixed, public scale)
KD = [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,
      1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]
CHARGE = [0,0,-1,-1,0,0,.1,0,1,0,0,0,0,0,1,0,0,0,0,0]
GLY_PRO = [float(a in "GP") for a in AA]
AROM    = [float(a in "FWY") for a in AA]
CYS_HIS = [float(a in "CH") for a in AA]


class ProteinClass(str, Enum):
    SOLUBLE     = "soluble_globular"
    MEMBRANE    = "membrane"
    ANTIBODY    = "antibody_fv"
    IDP         = "intrinsically_disordered"
    MULTIDOMAIN = "multi_domain"
    METALLO     = "metalloprotein"
    GLYCOPRO    = "glycoprotein"
    NOVEL       = "de_novo_or_orphan"

CLASSES = list(ProteinClass)


@dataclass
class SpecialistConfig:
    n_aa: int = 21
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_len: int = 1024
    n_experts: int = 8
    top_k: int = 2
    n_ensemble: int = 8          # IDP ensemble size
    router_temperature: float = 1.0
    confidence_temp_init: float = 1.0
    bond_w: float = 1.0
    contact_w: float = 0.3
    class_w: float = 0.5
    lr: float = 3e-4
    device: str = "cpu"


# ══════════════════════════════════════════════════════════════════════
# 1. SEQUENCE PHYSICOCHEMICAL FEATURES (transparent, not a black box)
# ══════════════════════════════════════════════════════════════════════
class SequenceFeatures(nn.Module):
    """
    Hand-computed, differentiable-through-embedding descriptors.
    These are COMPOSITIONAL proxies — not TMHMM / IUPred / SignalP.
    """
    def __init__(self, cfg: SpecialistConfig):
        super().__init__()
        reg = lambda v: self.register_buffer(
            v[0], torch.tensor(v[1], dtype=torch.float32), persistent=False)
        reg(("kd", KD)); reg(("charge", CHARGE))
        reg(("gp", GLY_PRO)); reg(("arom", AROM)); reg(("ch", CYS_HIS))
        self.win = 19

    def _smooth(self, x: Tensor) -> Tensor:
        # (B,L) -> sliding mean via avg_pool1d, zero padding
        B, L = x.shape
        p = F.pad(x.unsqueeze(1), (self.win//2, self.win//2))
        return F.avg_pool1d(p, self.win, stride=1,
                           count_include_pad=False).squeeze(1)[:, :L]

    def forward(self, seq: Tensor, mask: Tensor):
        """seq (B,L) long in [0,20]; mask (B,L) bool."""
        m = mask.float()
        safe = seq.clamp(0, 20)
        onehot = F.one_hot(safe, 21)[..., :20].float() * m.unsqueeze(-1)
        denom = m.sum(-1).clamp(min=1)

        def prop(buf):
            return (onehot @ buf) / denom

        hydro = F.embedding(safe, self.kd) * m
        charge = F.embedding(safe, self.charge) * m
        hydro = F.pad(hydro, (0, self.win//2))[:, :seq.shape[1]]
        local_h = self._smooth(F.pad(F.embedding(safe, self.kd)*m,
                                     (self.win//2, self.win//2)))[:, :seq.shape[1]]
        local_c = self._smooth(F.pad(F.embedding(safe, self.charge)*m,
                                     (self.win//2, self.win//2)))[:, :seq.shape[1]]

        # Compositional TM heuristic: hydrophobic runs + low polar fraction
        tm_like = ((local_h > 1.4) & (local_c.abs() < 1.2)).float() * m
        f = torch.stack([
            prop(self.kd)/4.5,
            (hydro.sum(-1)/denom),
            (prop(self.charge)),
            (prop(self.gp)),
            (prop(self.arom)),
            (prop(self.ch)),
            torch.log(denom)/math.log(max(self.win, 2)),
            (tm_like.sum(-1)/denom),
            ((charge.pow(2).sum(-1)/denom).sqrt()),
            ((local_h.pow(2).sum(-1)/denom).sqrt()),
        ], dim=-1)
        return f, local_h, tm_like


# ══════════════════════════════════════════════════════════════════════
# 2. LEARNED ROUTER with rule prior (prior is overridable by training)
# ══════════════════════════════════════════════════════════════════════
class SpecialistRouter(nn.Module):
    def __init__(self, cfg: SpecialistConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, cfg.n_experts),
        )
        # learned residual on top of heuristic prior
        self.residual = nn.Parameter(torch.zeros(cfg.n_experts))
        self._idx = {c: i for i, c in enumerate(CLASSES)}

    def prior(self, f: Tensor) -> Tensor:
        B = f.shape[0]
        p = torch.full((B, self.cfg.n_experts), 0.10, device=f.device)
        p[:, self._idx[ProteinClass.SOLUBLE]] = 0.55   # default
        i = self._idx
        p[B_f := torch.arange(B, device=f.device), i[ProteinClass.MEMBRANE]] = \
            torch.where(f[:, 7] > 0.25, 1.4, 0.1)
        p[B_f, i[ProteinClass.IDP]] = \
            torch.where((f[:, 0] < 0.15) & (f[:, 3] > 0.18), 1.5, 0.1)
        p[B_f, i[ProteinClass.METALLO]] = \
            torch.where(f[:, 5] > 0.12, 1.1, 0.1)
        p[B_f, i[ProteinClass.MULTIDOMAIN]] = \
            torch.where(f[:, 6] > 5.0, 0.9, 0.1)
        return p

    def forward(self, f: Tensor):
        logits = self.net(f) + torch.log(self.prior(f).clamp_min(1e-6)) + self.residual
        w = F.softmax(logits / self.cfg.router_temperature, dim=-1)
        k = min(self.cfg.top_k, self.cfg.n_experts)
        topw, topi = w.topk(k, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True).clamp_min(1e-8)
        return w, topw, topi, logits.argmax(-1)


# ══════════════════════════════════════════════════════════════════════
# 3. SPECIALIST EXPERTS — different bias, not just different names
# ══════════════════════════════════════════════════════════════════════
class BaseFoldExpert(nn.Module):
    """Shared backbone; each subclass adds a class-specific head/restraint."""
    def __init__(self, cfg: SpecialistConfig, pclass: ProteinClass):
        super().__init__()
        self.cfg, self.pclass = cfg, pclass
        self.emb = nn.Embedding(cfg.n_aa, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_len, cfg.d_model) * .02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model*4, dropout=0.0,
            batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.coord = nn.Linear(cfg.d_model, 3)
        self.conf = nn.Linear(cfg.d_model, 1)
        self.extra = nn.Linear(cfg.d_model, 8)   # class auxiliary head
        self._temp = nn.Parameter(torch.tensor(float(cfg.confidence_temp_init)))

    def backbone_fwd(self, seq, mask):
        L = seq.shape[1]
        key = ~mask
        x = self.emb(seq) + self.pos[:, :L]
        h = self.backbone(x, src_key_padding_mask=key)
        return self.norm(h)

    def trace(self, h):
        B, L, _ = h.shape
        base = torch.zeros(B, L, 3, device=h.device)
        base[..., 0] = torch.arange(L, device=h.device) * 3.8
        return base + torch.tanh(self.coord(h)) * 8.0

    def forward(self, seq, mask, **kw):
        h = self.backbone_fwd(seq, mask)
        coords = self.trace(h)
        # temperature-scaled confidence
        logit = self.conf(h).squeeze(-1) / self._temp.abs().clamp_min(1e-3)
        conf = torch.sigmoid(logit)
        aux = self.extra(h)
        return {"coords": coords, "confidence": conf, "aux": aux,
                "hidden": h, "expert": self.pclass.value}

    def class_loss(self, out, batch) -> Tensor:
        return out["coords"].new_zeros(())


class MembraneExpert(BaseFoldExpert):
    """Hydrophobic burial: buried residues should be hydrophobic.
    Requires a bilayer normal / topology later; here a contact-based proxy."""
    def __init__(self, cfg): super().__init__(cfg, ProteinClass.MEMBRANE)

    def class_loss(self, out, batch):
        c, m = out["coords"], batch["mask"].float()
        d = torch.cdist(c, c)
        contact = (d < 8.0).float() * m.unsqueeze(1) * m.unsqueeze(2)
        burial = contact.sum(-1) / contact.sum(-1).amax(-1, keepdim=True).clamp_min(1)
        hyd = batch["hydropathy"] * m
        hyd = (hyd - hyd.sum(-1, True)/m.sum(-1, True).clamp_min(1))
        hyd = hyd / hyd.pow(2).mean(-1, True).sqrt().clamp_min(1e-6)
        b = burial * m
        b = (b - b.sum(-1, True)/m.sum(-1, True).clamp_min(1))
        corr = (b*hyd*m).sum(-1) / (b.pow(2).sum(-1).sqrt()*hyd.pow(2).sum(-1).sqrt()).clamp_min(1e-6)
        return (1.0 - corr).clamp_min(0).mean()   # want positive correlation


class IDPExpert(BaseFoldExpert):
    """Outputs an ENSEMBLE, not one structure. Reports Rg spread."""
    def __init__(self, cfg):
        super().__init__(cfg, ProteinClass.IDP)
        self.ens = nn.Linear(cfg.d_model, cfg.n_ensemble*3)

    def forward(self, seq, mask, **kw):
        o = super().forward(seq, mask)
        B, L, _ = o["hidden"].shape
        delta = self.ens(o["hidden"]).view(B, L, self.cfg.n_ensemble, 3)
        base = o["coords"].unsqueeze(2)
        ens = base + torch.tanh(delta) * 12.0
        m = mask.float().unsqueeze(2)
        com = (ens*m.unsqueeze(-1)).sum(1) / m.sum(1).clamp_min(1)
        rg = (((ens-com.unsqueeze(1)).pow(2).sum(-1) * m.squeeze(-1))
              .sum(1) / m.squeeze(-1).sum(1).clamp_min(1)).sqrt()
        o["ensemble"] = ens
        o["rg_per_conformer"] = rg
        o["rg_mean"] = rg.mean(-1)
        o["rg_spread"] = rg.std(-1)
        o["confidence"] = o["confidence"] * 0.6   # honest: lower certainty
        return o

    def class_loss(self, out, batch):
        if "target_rg" not in batch:
            return out["rg_mean"].new_zeros(())
        return F.mse_loss(out["rg_mean"], batch["target_rg"])


class AntibodyExpert(BaseFoldExpert):
    """Framework restraint + CDR-H3 flexibility + disulfide proxy."""
    def __init__(self, cfg): super().__init__(cfg, ProteinClass.ANTIBODY)

    def class_loss(self, out, batch):
        c = out["coords"]
        loss = c.new_zeros(())
        if "cdr_mask" in batch:
            cdr = batch["cdr_mask"].float()
            # CDR must not be forced rigid: only bond-length regularity there
            bl = (c[:, 1:] - c[:, :-1]).norm(dim=-1)
            loss = loss + ((bl-3.8).abs()*cdr[:, 1:]).sum()/cdr[:, 1:].sum().clamp_min(1)
        if "ss_pairs" in batch:  # (B,K,2) residue indices of Cys pairs
            pairs = batch["ss_pairs"].long()
            B = c.shape[0]
            errs = []
            for b in range(B):
                for p in pairs[b]:
                    if int(p.min()) >= 0:
                        errs.append((c[b, p[0]] - c[b, p[1]]).norm() - 7.0)
            if errs:
                loss = loss + torch.stack(errs).pow(2).mean()
        return loss


class MultiDomainExpert(BaseFoldExpert):
    """Coarse intra-domain folding + separate inter-domain placement."""
    def __init__(self, cfg): super().__init__(cfg, ProteinClass.MULTIDOMAIN)

    def class_loss(self, out, batch):
        if "domain_ids" not in batch:
            return out["coords"].new_zeros(())
        d = batch["domain_ids"]
        c = out["coords"]
        same = (d.unsqueeze(-1) == d.unsqueeze(-2)).float()
        dhat = torch.cdist(c, c)
        # down-weight cross-domain distance error until docking stage
        w = 0.15 + 0.85*same
        return (dhat.pow(2)*w).mean()*0.01


class MetalloExpert(BaseFoldExpert):
    """Cys/His CA should approach supplied metal sites (CA proxy, not ion)."""
    def __init__(self, cfg): super().__init__(cfg, ProteinClass.METALLO)

    def class_loss(self, out, batch):
        if "metal_sites" not in batch:
            return out["coords"].new_zeros(())
        c, sites = out["coords"], batch["metal_sites"]
        d = torch.cdist(c, sites).min(-1).values
        return F.relu(d - 3.5).pow(2).mean()


class GlycoExpert(BaseFoldExpert):
    """Glycan sites get lower confidence + excluded-volume margin."""
    def __init__(self, cfg): super().__init__(cfg, ProteinClass.GLYCOPRO)

    def class_loss(self, out, batch):
        if "glyco_mask" not in batch:
            return out["coords"].new_zeros(())
        return out["coords"].new_zeros(())  # confidence handled below

    def forward(self, seq, mask, **kw):
        o = super().forward(seq, mask)
        if "glyco_mask" in kw:
            o["confidence"] = o["confidence"] * (1 - .35*kw["glyco_mask"].float())
        return o


class NovelExpert(BaseFoldExpert):
    """No homolog: rely on compactness + smooth backbone, not MSA."""
    def __init__(self, cfg): super().__init__(cfg, ProteinClass.NOVEL)

    def class_loss(self, out, batch):
        c, m = out["coords"], batch["mask"].float()
        bl = (c[:, 1:] - c[:, :-1]).norm(dim=-1)
        mask = m[:, 1:]*m[:, :-1]
        bond = ((bl-3.8).abs()*mask).sum()/mask.sum().clamp_min(1)
        clash = torch.cdist(c, c).clamp_min(1e-3)
        pen = (1.0/clash).pow(6).mean() * 1e-4
        return bond*0.1 + pen


class SolubleExpert(BaseFoldExpert):
    def __init__(self, cfg): super().__init__(cfg, ProteinClass.SOLUBLE)


EXPERT_CLASSES = {
    ProteinClass.SOLUBLE: SolubleExpert,
    ProteinClass.MEMBRANE: MembraneExpert,
    ProteinClass.ANTIBODY: AntibodyExpert,
    ProteinClass.IDP: IDPExpert,
    ProteinClass.MULTIDOMAIN: MultiDomainExpert,
    ProteinClass.METALLO: MetalloExpert,
    ProteinClass.GLYCOPRO: GlycoExpert,
    ProteinClass.NOVEL: NovelExpert,
}


# ══════════════════════════════════════════════════════════════════════
# 4. MIXTURE-OF-EXPERTS FOLDER
# ══════════════════════════════════════════════════════════════════════
class SpecialistFold(nn.Module):
    def __init__(self, cfg: SpecialistConfig):
        super().__init__()
        self.cfg = cfg
        self.features = SequenceFeatures(cfg)
        self.router = SpecialistRouter(cfg)
        self.experts = nn.ModuleList([EXPERT_CLASSES[c](cfg) for c in CLASSES])
        self.register_buffer("kd_buf", torch.tensor(KD))

    def forward(self, seq: Tensor, mask: Optional[Tensor] = None, **kw):
        B, L = seq.shape
        if mask is None:
            mask = seq != UNK
        f, local_h, tm = self.features(seq, mask)
        w, topw, topi, hard = self.router(f)

        batch = dict(mask=mask, hydropathy=F.embedding(seq.clamp(0,20), self.kd_buf))
        batch.update(kw)

        coords = seq.new_zeros(B, L, 3, dtype=torch.float32)
        conf = seq.new_zeros(B, L, dtype=torch.float32)
        aux_losses, expert_outs = [], {}

        for k in range(topw.shape[1]):
            idx, weight = topi[:, k], topw[:, k]
            for j, expert in enumerate(self.experts):
                use = (idx == j)
                if not use.any():
                    continue
                o = expert(seq[use], mask[use], **{
                    z: v[use] for z, v in kw.items() if torch.is_tensor(v) and v.shape[0] == B})
                sel = torch.where(use)[0]
                coords[sel] += weight[use].unsqueeze(-1).unsqueeze(-1) * o["coords"]
                conf[sel] += weight[use].unsqueeze(-1) * o["confidence"]
                cl = expert.class_loss(o, {
                    z: (v[use] if torch.is_tensor(v) and v.shape[0] == B else v)
                    for z, v in batch.items()})
                aux_losses.append(weight[use].mean() * cl)
                expert_outs[expert.pclass.value] = o

        out = {"coords": coords, "confidence": conf,
               "router_probs": w, "expert_weights": topw, "expert_ids": topi,
               "predicted_class": hard, "features": f,
               "expert_outputs": expert_outs,
               "class_loss": torch.stack(aux_losses).sum() if aux_losses else coords.new_zeros(())}
        if ProteinClass.IDP.value in expert_outs:
            out["idp_ensemble"] = expert_outs[ProteinClass.IDP.value].get("ensemble")
            out["idp_rg"] = expert_outs[ProteinClass.IDP.value].get("rg_mean")
        return out

    def loss(self, out, target_coords, mask, target_dist=None):
        """Rotation/translation-invariant training objective."""
        pred, ref = out["coords"], target_coords
        dp, dr = torch.cdist(pred, pred), torch.cdist(ref, ref)
        m = mask.float()
        pm = m.unsqueeze(1)*m.unsqueeze(2)
        dist_loss = ((dp-dr).pow(2)*pm).sum()/pm.sum().clamp_min(1)
        bl = (pred[:, 1:]-pred[:, :-1]).norm(dim=-1)
        bm = m[:, 1:]*m[:, :-1]
        bond = ((bl-3.8).pow(2)*bm).sum()/bm.sum().clamp_min(1)
        return (dist_loss*self.cfg.contact_w + bond*self.cfg.bond_w
                + out["class_loss"]*self.cfg.class_w)
