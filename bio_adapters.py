#!/usr/bin/env python3
"""
=============================================================================
rMD17 · ClinVar (open) · CRISPR Doench-2016  —  REAL DATA ADAPTERS
=============================================================================
License: MIT | 2026

LITERATURE
 [R1] Christensen & von Lilienfeld. MLST 1, 045018 (2020).      [rMD17]
 [C1] Landrum et al. "ClinVar: improvements to accessing data."
      Nucleic Acids Res. 48, D835 (2020).
 [C2] Doench et al. "Optimized sgRNA design to maximize activity and minimize
      off-target effects of CRISPR-Cas9." Nat. Biotechnol. 34, 184 (2016).
 [C3] Kim et al. "SpCas9 activity prediction by DeepSpCas9." Sci.Adv.5(2019)
 [C4] Tsai et al. "GUIDE-seq enables genome-wide profiling of off-target
      cleavage by CRISPR-Cas nucleases." Nat. Biotechnol. 33, 187 (2015).

=============================================================================
HONEST SCOPE — READ BEFORE USING ANY OUTPUT
=============================================================================
rMD17
  ✅ real DFT energies + forces (PBE/def2-SVP), kcal/mol
  ❌ 10 molecules only. Not a general-purpose potential.

ClinVar
  ✅ fully open, no authorization needed
  ⚠️ LABELS ARE CLINICAL ASSERTIONS, NOT GROUND TRUTH.
     A model trained on ClinVar learns to predict ClinVar's curation,
     including its errors and ascertainment bias.
  ⚠️ MUST split by GENE, not by variant. Variants in the same gene share
     annotation patterns → random splits leak badly.
  ⚠️ Extreme class imbalance; benign/pathogenic ratios differ per gene.
  ❌ Not a diagnostic. Not for clinical use.

TCGA
  ⚠️ Controlled-access data (BAM, germline) requires dbGaP authorization.
     This file only touches OPEN-access public somatic MAF summaries.
  ❌ No patient-level re-identification. No clinical decision support.

CRISPR (Doench 2016)
  ✅ real measured efficacy, ~2.6k guides — SMALL.
  ❌ On-target efficacy ≠ off-target safety. Separate datasets needed
     (GUIDE-seq / CIRCLE-seq) for off-target.
  ❌ Any designed guide requires experimental validation and biosafety review.
=============================================================================
"""
from __future__ import annotations
import argparse, gzip, json, logging, math, urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import h5py; HAS_H5=True
except ImportError:
    HAS_H5=False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("bio")

KCALMOL_EV = 1/23.060547830619
Z2I = {1:0,6:1,7:2,8:3,9:4,16:5,17:6}


# =============================================================================
# 1. rMD17 — schema-probing loader with unit verification
# =============================================================================

def rmd17_probe(path: Path) -> Dict:
    if not HAS_H5: raise ImportError("pip install h5py")
    with h5py.File(path,"r") as f:
        rep = {"file": path.name, "keys": list(f.keys()), "datasets": {}}
        def v(n,o):
            if isinstance(o,h5py.Dataset):
                rep["datasets"][n] = {"shape": list(o.shape), "dtype": str(o.dtype)}
        f.visititems(v)
    return rep


def rmd17_load(path: Path, limit: Optional[int]=None) -> Dict:
    """
    rMD17 native units: energies kcal/mol, forces kcal/mol/Å, coords Å.
    Converted to eV, eV/Å. Verified by dimensional sanity check.
    """
    if not HAS_H5: raise ImportError("pip install h5py")
    with h5py.File(path,"r") as f:
        ks = list(f.keys())
        def pick(*names):
            for n in names:
                if n in f: return n
            raise KeyError(f"none of {names} in {ks}. Run rmd17_probe first.")
        Z = np.asarray(f[pick("nuclear_charges","z","atomic_numbers")]).reshape(-1)
        R = np.asarray(f[pick("coords","coordinates")], dtype=np.float64)
        E = np.asarray(f[pick("energies","energy")], dtype=np.float64).reshape(-1)
        Fo= np.asarray(f[pick("forces","force")], dtype=np.float64)
    if limit: R,E,Fo = R[:limit],E[:limit],Fo[:limit]
    if R.ndim==2: R = R.reshape(len(E), len(Z), 3)
    Fo = Fo.reshape(R.shape)
    E_eV, F_eV = E*KCALMOL_EV, Fo*KCALMOL_EV
    fm = np.linalg.norm(F_eV,axis=-1)
    sanity = {"force_mean_eV_A": float(fm.mean()),
              "force_max_eV_A": float(fm.max()),
              "E_per_atom_eV": float((E_eV/len(Z)).mean())}
    warn=[]
    if fm.max()>200: warn.append("force>200 eV/Å — unit conversion likely wrong")
    if 0<fm.mean()<0.002: warn.append("force<0.002 eV/Å — over-converted?")
    log.info(f"[rMD17] {len(E)} conf, {len(Z)} atoms | "
             f"<F>={sanity['force_mean_eV_A']:.4f} eV/Å")
    for w in warn: log.warning(f"[rMD17] {w}")
    return {"z": np.array([Z2I[int(z)] for z in Z]),
            "Z_raw": Z, "R": R.astype(np.float32),
            "E": E_eV.astype(np.float32), "F": F_eV.astype(np.float32),
            "n_atoms": np.full(len(E), len(Z)),
            "level": "PBE/def2-SVP", "sanity": sanity, "warnings": warn}


def rmd17_split(d: Dict, n_train=950, n_val=50, seed=42) -> Dict:
    """Common rMD17 protocol: 950 train / 50 val / rest test [R1]."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(d["E"]))
    out={}
    for name, idx in (("train",p[:n_train]),
                      ("val",p[n_train:n_train+n_val]),
                      ("test",p[n_train+n_val:])):
        out[name] = {k:(v[idx] if isinstance(v,np.ndarray) and v.ndim and
                        len(v)==len(d["E"]) else v) for k,v in d.items()}
        out[name]["n"] = len(idx)
    log.info(f"[rMD17] split {n_train}/{n_val}/{len(p)-n_train-n_val} seed={seed}")
    log.info("[rMD17] Different seeds/splits are NOT comparable to published tables.")
    return out


# =============================================================================
# 2. ClinVar — open access, GENE-LEVEL split, honest labels
# =============================================================================

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

BENIGN = {"Benign","Likely benign","Benign/Likely benign"}
PATHO  = {"Pathogenic","Likely pathogenic","Pathogenic/Likely pathogenic"}
# review status → stars (higher = more trustworthy)
STARS = {"practice guideline":4,
         "reviewed by expert panel":3,
         "criteria provided, multiple submitters, no conflicts":2,
         "criteria provided, conflicting classifications":1,
         "criteria provided, single submitter":1,
         "criteria provided, conflicting interpretations":1,
         "no assertion criteria provided":0,
         "no assertion provided":0,
         "no classification provided":0,
         "no classification for the individual variant":0}


def clinvar_download(dest="data/clinvar") -> Path:
    d=Path(dest); d.mkdir(parents=True,exist_ok=True)
    out=d/"variant_summary.txt.gz"
    if out.exists() and out.stat().st_size>1_000_000:
        log.info(f"[ClinVar] cached {out.stat().st_size/1e6:.1f} MB"); return out
    log.info(f"[ClinVar] downloading {CLINVAR_URL}")
    with urllib.request.urlopen(CLINVAR_URL, timeout=300) as r, open(out,"wb") as f:
        tot=0
        while True:
            c=r.read(1<<20)
            if not c: break
            f.write(c); tot+=len(c)
            if tot%(50<<20)==0: log.info(f"  {tot/1e6:.0f} MB")
    log.info(f"[ClinVar] {out.stat().st_size/1e6:.1f} MB")
    return out


def clinvar_load(path: Path, assembly="GRCh38", min_stars=1,
                 limit: Optional[int]=None) -> List[Dict]:
    """
    Parse ClinVar variant_summary. Keeps SNVs with confident binary labels.
    LABEL = clinical assertion, NOT biological ground truth.
    """
    recs=[]; skipped={"not_snv":0,"ambiguous":0,"low_review":0,"other_assembly":0}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",errors="ignore") as f:
        header=f.readline().lstrip("#").rstrip("\n").split("\t")
        H={h:i for i,h in enumerate(header)}
        need=["GeneSymbol","ClinicalSignificance","ReviewStatus","Assembly",
              "Type","Chromosome","PositionVCF","ReferenceAlleleVCF",
              "AlternateAlleleVCF","Name"]
        for n in need:
            if n not in H: raise KeyError(f"ClinVar column '{n}' missing; got {header[:20]}")
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)<len(header): continue
            if p[H["Assembly"]]!=assembly:
                skipped["other_assembly"]+=1; continue
            if p[H["Type"]]!="single nucleotide variant":
                skipped["not_snv"]+=1; continue
            cs=p[H["ClinicalSignificance"]]
            if cs in BENIGN: y=0
            elif cs in PATHO: y=1
            else:
                skipped["ambiguous"]+=1; continue
            st=STARS.get(p[H["ReviewStatus"]].strip(),0)
            if st<min_stars:
                skipped["low_review"]+=1; continue
            ref,alt=p[H["ReferenceAlleleVCF"]],p[H["AlternateAlleleVCF"]]
            if len(ref)!=1 or len(alt)!=1 or ref not in "ACGT" or alt not in "ACGT":
                skipped["not_snv"]+=1; continue
            recs.append({"gene":p[H["GeneSymbol"]], "chrom":p[H["Chromosome"]],
                         "pos":int(p[H["PositionVCF"]]) if p[H["PositionVCF"]].isdigit() else -1,
                         "ref":ref,"alt":alt,"label":y,"stars":st,
                         "name":p[H["Name"]][:120]})
            if limit and len(recs)>=limit: break
    log.info(f"[ClinVar] kept {len(recs)} SNVs | skipped {skipped}")
    if recs:
        pos=sum(r["label"] for r in recs)
        log.info(f"[ClinVar] pathogenic {pos} ({100*pos/len(recs):.1f}%) / "
                 f"benign {len(recs)-pos}")
        log.info(f"[ClinVar] genes: {len(set(r['gene'] for r in recs))}")
    log.warning("[ClinVar] Labels are CLINICAL ASSERTIONS with ascertainment bias. "
                "A model learns ClinVar's curation, not biological truth.")
    return recs


def gene_split(recs: List[Dict], frac=(0.7,0.15,0.15), seed=42) -> Dict:
    """
    GENE-LEVEL split. Random variant splits leak: variants in one gene share
    annotation patterns and curation history. This is the single most common
    error in ClinVar ML papers.
    """
    genes = sorted(set(r["gene"] for r in recs))
    rng = np.random.default_rng(seed)
    rng.shuffle(genes)
    n=len(genes); ntr=int(frac[0]*n); nva=int(frac[1]*n)
    gtr,gva,gte = set(genes[:ntr]), set(genes[ntr:ntr+nva]), set(genes[ntr+nva:])
    out = {k:[r for r in recs if r["gene"] in g]
           for k,g in (("train",gtr),("val",gva),("test",gte))}
    for k,v in out.items():
        pos=sum(r["label"] for r in v)
        log.info(f"[gene-split] {k}: {len(v)} variants, {len(set(r['gene'] for r in v))} genes, "
                 f"{100*pos/max(len(v),1):.1f}% pathogenic")
    assert not (gtr & gva) and not (gtr & gte) and not (gva & gte)
    log.info("[gene-split] ✓ zero gene overlap between splits")
    return out


BASE_IDX={"A":0,"C":1,"G":2,"T":3}

def clinvar_features(r: Dict, flank: int = 0) -> np.ndarray:
    """
    Minimal, honest feature set. NO sequence context (we don't fetch a genome).
    To do this properly you must fetch ±k bp reference context from a genome
    FASTA — otherwise the model cannot learn sequence-dependent effects.
    """
    v = np.zeros(4+4+3+1, dtype=np.float32)
    v[BASE_IDX[r["ref"]]] = 1.0
    v[4+BASE_IDX[r["alt"]]] = 1.0
    ti = {(a,b) for a in "ACGT" for b in "ACGT" if a!=b}
    pur = set("AG")
    v[8] = float((r["ref"] in pur) == (r["alt"] in pur))   # transition
    v[9] = float(r["ref"] in pur and r["alt"] not in pur)
    v[10]= float(r["ref"] not in pur and r["alt"] in pur)
    v[11]= r["stars"]/4.0
    return v


class ClinVarMLP(nn.Module):
    def __init__(self, d_in=12, h=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,h),nn.SiLU(),nn.Dropout(0.2),
                               nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1))
    def forward(self,x): return self.net(x).squeeze(-1)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUROC (no sklearn dependency)."""
    order=np.argsort(p); r=np.empty(len(p),float); r[order]=np.arange(1,len(p)+1)
    npos=y.sum(); nneg=len(y)-npos
    if npos==0 or nneg==0: return float("nan")
    return float((r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    order=np.argsort(-p); ys=y[order]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys)
    prec=tp/np.maximum(tp+fp,1); rec=tp/max(ys.sum(),1)
    return float(np.sum(np.diff(np.concatenate([[0],rec]))*prec))


def train_clinvar(args):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    p=Path(args.data)
    if not p.exists(): p=clinvar_download(str(Path(args.data).parent))
    recs=clinvar_load(p, min_stars=args.min_stars, limit=args.limit)
    if len(recs)<100: log.error("too few records"); return
    sp=gene_split(recs, seed=args.seed)
    def mk(rs):
        X=torch.tensor(np.stack([clinvar_features(r) for r in rs]))
        y=torch.tensor([r["label"] for r in rs],dtype=torch.float32)
        return X.to(dev),y.to(dev)
    Xtr,ytr=mk(sp["train"]); Xva,yva=mk(sp["val"]); Xte,yte=mk(sp["test"])
    model=ClinVarMLP().to(dev)
    pos_w=(ytr.numel()-ytr.sum())/ytr.sum().clamp(min=1)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-1; best_state=None
    for ep in range(args.epochs):
        model.train(); perm=torch.randperm(len(Xtr),device=dev)
        for s in range(0,len(perm),args.batch_size):
            i=perm[s:s+args.batch_size]
            loss=F.binary_cross_entropy_with_logits(model(Xtr[i]),ytr[i],pos_weight=pos_w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (ep+1)%max(1,args.epochs//10)==0:
            model.eval()
            with torch.no_grad():
                pv=torch.sigmoid(model(Xva)).cpu().numpy()
            a=auroc(yva.cpu().numpy(),pv)
            log.info(f"ep {ep+1:4d} | val AUROC {a:.4f}")
            if a>best: best=a; best_state={k:v.clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): pt=torch.sigmoid(model(Xte)).cpu().numpy()
    yt=yte.cpu().numpy()
    print("\n"+"="*74)
    print("  ClinVar — GENE-LEVEL SPLIT (no leakage)")
    print("="*74)
    print(f"  train/val/test variants : {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(f"  min review stars        : {args.min_stars}")
    print(f"  test AUROC              : {auroc(yt,pt):.4f}")
    print(f"  test AUPRC              : {auprc(yt,pt):.4f}")
    print(f"  test prevalence         : {yt.mean():.3f}")
    print("\n  ⚠️  WHAT THIS NUMBER MEANS")
    print("     Agreement with ClinVar clinical assertions on UNSEEN GENES.")
    print("     It does NOT mean the model detects pathogenicity.")
    print("  ⚠️  Features here are minimal (ref/alt/type/stars only).")
    print("     Real performance requires reference sequence context,")
    print("     conservation, and consequence annotation.")
    print("  ❌ NOT a diagnostic. NOT for clinical use.")
    print("="*74)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    json.dump({"auroc":auroc(yt,pt),"auprc":auprc(yt,pt),
               "n_test":len(yt),"min_stars":args.min_stars,
               "split":"gene-level","seed":args.seed}, open(args.out,"w"), indent=2)


# =============================================================================
# 3. CRISPR on-target efficiency — Doench 2016 (Rule Set 2)
# =============================================================================

DOENCH_URLS = [
    "https://raw.githubusercontent.com/MicrosoftResearch/Azimuth/master/azimuth/data/Doench_et_al_2016.csv",
]

class GuideCNN(nn.Module):
    """1D CNN over one-hot 30-mer. Small model for a small dataset."""
    def __init__(self, n_ch=4, width=64):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(n_ch,width,5,padding=2),nn.SiLU(),
            nn.Conv1d(width,width,5,padding=2),nn.SiLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Linear(width,64),nn.SiLU(),
                                nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x):            # x: (B,4,30)
        return self.head(self.conv(x).squeeze(-1)).squeeze(-1)


def load_doench(path: Path, limit=None) -> Tuple[np.ndarray,np.ndarray]:
    """
    Doench 2016: 30-mer = 4 bp upstream + 20 bp guide + 3 bp PAM + 3 bp downstream.
    Label: measured efficacy (continuous). We keep the raw regression target.
    """
    import csv
    seqs=[]; ys=[]
    with open(path,newline="",encoding="utf-8",errors="ignore") as f:
        rd=csv.DictReader(f)
        cols=rd.fieldnames or []
        # column names vary by release; probe
        sq=next((c for c in cols if "30" in c or "sequence" in c.lower()
                 or "sgrna" in c.lower()), None)
        sc=next((c for c in cols if "score" in c.lower() or "efficacy" in c.lower()
                 or "activity" in c.lower()), None)
        if sq is None or sc is None:
            raise KeyError(f"cannot find sequence/score columns in {cols}")
        log.info(f"[Doench] sequence col='{sq}' score col='{sc}'")
        for row in rd:
            s=(row[sq] or "").strip().upper()
            try: y=float(row[sc])
            except (TypeError,ValueError): continue
            if len(s)!=30 or any(c not in "ACGT" for c in s): continue
            seqs.append(s); ys.append(y)
            if limit and len(seqs)>=limit: break
    if not seqs: raise RuntimeError("no valid 30-mers parsed")
    X=np.zeros((len(seqs),4,30),np.float32)
    for i,s in enumerate(seqs):
        for j,c in enumerate(s): X[i,BASE_IDX[c],j]=1.
    y=np.array(ys,np.float32)
    log.info(f"[Doench] {len(seqs)} guides | score mean={y.mean():.3f} std={y.std():.3f}")
    log.warning("[Doench] ~2.6k guides is SMALL. Expect modest correlation. "
                "On-target efficacy ≠ off-target safety.")
    return X,y


def train_crispr(args):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    p=Path(args.data)
    if not p.exists():
        log.error(f"{p} not found.\n"
                  f"Download Doench 2016 Supplementary Table from:\n"
                  f"  https://www.nature.com/articles/nbt.3437 (Supplementary)\n"
                  f"or try: {DOENCH_URLS[0]}\n"
                  f"Save as CSV with a 30-mer column and a score column.")
        return
    X,y=load_doench(p, limit=args.limit)
    rng=np.random.default_rng(args.seed); perm=rng.permutation(len(y))
    ntr=int(0.7*len(y)); nva=int(0.15*len(y))
    itr,iva,ite=perm[:ntr],perm[ntr:ntr+nva],perm[ntr+nva:]
    Xt=torch.tensor(X).to(dev); yt=torch.tensor(y).to(dev)
    ymu,ysd=float(yt[itr].mean()),float(yt[itr].std())
    model=GuideCNN().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=1e9; bs=None
    for ep in range(args.epochs):
        model.train(); order=rng.permutation(ntr)
        for s in range(0,ntr,args.batch_size):
            i=itr[order[s:s+args.batch_size]]
            pred=model(Xt[i]); tgt=(yt[i]-ymu)/ysd
            loss=F.mse_loss(pred,tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (ep+1)%max(1,args.epochs//10)==0:
            model.eval()
            with torch.no_grad():
                pv=model(Xt[iva]).cpu().numpy()*ysd+ymu
            mse=float(np.mean((pv-y[iva])**2))
            log.info(f"ep {ep+1:4d} | val MSE {mse:.5f}")
            if mse<best: best=mse; bs={k:v.clone() for k,v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad(): pt=model(Xt[ite]).cpu().numpy()*ysd+ymu
    from scipy.stats import spearmanr, pearsonr
    sp=spearmanr(pt,y[ite]).correlation; pr=pearsonr(pt,y[ite])[0]
    print("\n"+"="*74)
    print("  CRISPR on-target efficacy — Doench 2016")
    print("="*74)
    print(f"  n train/val/test : {ntr}/{nva}/{len(ite)}")
    print(f"  test Spearman ρ  : {sp:.4f}")
    print(f"  test Pearson r   : {pr:.4f}")
    print(f"  test RMSE        : {np.sqrt(np.mean((pt-y[ite])**2)):.4f}")
    print("\n  ⚠️  Published Rule Set 2 reports Spearman ≈ 0.4–0.5 on held-out")
    print("     genes. Verify against the paper's exact split before comparing.")
    print("  ⚠️  This is ON-TARGET efficacy only.")
    print("     Off-target requires GUIDE-seq / CIRCLE-seq [C4].")
    print("  ❌ Any designed guide needs experimental validation + biosafety review.")
    print("="*74)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"ymu":ymu,"ysd":ysd,
                "spearman":float(sp)}, args.out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("cmd",choices=["rmd17-probe","rmd17-load","clinvar","crispr"])
    p.add_argument("--data",default="data/clinvar/variant_summary.txt.gz")
    p.add_argument("--out",default="ckpt/bio.json")
    p.add_argument("--molecule",default="benzene")
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--min-stars",dest="min_stars",type=int,default=1)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch-size",dest="batch_size",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.cmd=="rmd17-probe":
        print(json.dumps(rmd17_probe(Path(a.data)),indent=2))
    elif a.cmd=="rmd17-load":
        d=rmd17_load(Path(a.data),a.limit); rmd17_split(d,seed=a.seed)
    elif a.cmd=="clinvar": train_clinvar(a)
    else: train_crispr(a)


if __name__=="__main__":
    main()
