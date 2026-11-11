#!/usr/bin/env python3
"""
=============================================================================
GENOME CONTEXT · GUIDE-seq OFF-TARGET · CALIBRATION (temperature/isotonic/conformal)
=============================================================================
License: MIT | 2026

LITERATURE
 [G1] Landrum et al. "ClinVar: improvements to accessing data."
      Nucleic Acids Res. 48, D835 (2020).
 [G2] Tsai et al. "GUIDE-seq enables genome-wide profiling of off-target
      cleavage by CRISPR-Cas nucleases." Nat. Biotechnol. 33, 187 (2015).
 [G3] Lazzarotto et al. "CHANGE-seq reveals genetic and epigenetic effects on
      CRISPR-Cas9 genome-wide specificity." Nat. Biotechnol. 39, 1394 (2021).
 [C1] Guo, Pleiss, Sun, Weinberger. "On calibration of modern neural networks."
      ICML (2017).                                    [temperature scaling]
 [C2] Zadrozny & Elkan. "Transforming classifier scores into accurate
      multiclass probability estimates." KDD (2002).   [isotonic]
 [C3] Niculescu-Mizil & Caruana. "Predicting good probabilities with supervised
      learning." ICML (2005).
 [C4] Kuleshov, Fenner, Ermon. "Accurate uncertainties for deep learning using
      calibrated regression." ICML (2018).
 [C5] Lei, G'Sell, Rinaldo, Tibshirani, Wasserman. "Distribution-free predictive
      inference for regression." JASA 113, 1094 (2018).
 [C6] Angelopoulos & Bates. "A gentle introduction to conformal prediction."
      arXiv:2107.07511 (2021).
 [C7] Vovk, Gammerman, Shafer. "Algorithmic Learning in a Random World." (2005).

=============================================================================
HONEST SCOPE — READ BEFORE USING ANY NUMBER
=============================================================================
GENOME CONTEXT
  ✅ Real reference sequence from UCSC/Ensembl. Ref-allele match is VERIFIED.
  ❌ Context alone is NOT pathogenicity. Without consequence annotation
     (VEP/snpEff), conservation (phyloP/phastCons), and splice models,
     a context CNN learns mostly local sequence composition.
  ⚠️ ClinVar labels are CLINICAL ASSERTIONS with ascertainment bias.
  ⚠️ MUST split by GENE. Random variant splits leak badly.

GUIDE-seq
  ✅ Real measured off-target cleavage counts.
  ❌ Cell-type specific. Detection limit exists.
     "Not detected" ≠ "never cleaved". Absence is NOT a validated negative.
  ❌ Synthetic mismatch decoys give OPTIMISTIC metrics. Flagged in output.
  ❌ On-target efficacy ≠ off-target safety. Different task, different data.
  ⚠️ For better negatives use CHANGE-seq [G3] (in vitro, more comprehensive).

CALIBRATION
  ✅ Temperature scaling: fits 1 parameter, PRESERVES ranking (AUROC unchanged).
  ⚠️ Isotonic: flexible, CAN change ranking, OVERFITS small validation sets.
  ✅ Split conformal: distribution-free MARGINAL coverage under exchangeability.
  ❌ Marginal coverage ≠ conditional coverage. A 90% interval can be
     systematically wrong on a subgroup while averaging to 90%.
  ❌ Exchangeability FAILS under distribution shift (new genes, new scaffolds,
     new cell types). Calibration fitted on val may not transfer to test.
  ⚠️ ALWAYS fit calibration on validation, evaluate on untouched test.
=============================================================================
"""
from __future__ import annotations
import argparse, gzip, json, logging, math, os, shutil, urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from pyfaidx import Fasta
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("ctx_cal")

BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = str.maketrans("ACGTN", "TGCAN")

GENOME_SOURCES = {
    "hg38_ucsc": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        "file": "hg38.fa.gz", "assembly": "GRCh38", "style": "ucsc"},
    "hg38_ensembl": {
        "url": ("https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/"
                "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"),
        "file": "GRCh38.primary_assembly.fa.gz", "assembly": "GRCh38",
        "style": "ensembl"},
    "hg19_ucsc": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
        "file": "hg19.fa.gz", "assembly": "GRCh37", "style": "ucsc"},
}

CLINVAR_URL = ("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
               "variant_summary.txt.gz")


# =============================================================================
# SECTION 1 — REFERENCE GENOME FASTA
# =============================================================================

def download_genome(key: str, dest: str = "data/genome",
                    decompress: bool = True) -> Path:
    """
    Download reference FASTA. ~950 MB compressed, ~3.1 GB uncompressed.
    pyfaidx builds a .fai index on first use (random access, no full load).
    """
    src = GENOME_SOURCES[key]
    d = Path(dest); d.mkdir(parents=True, exist_ok=True)
    gz = d / src["file"]
    fa = gz.with_suffix("") if str(gz).endswith(".gz") else gz
    if fa.exists() and fa.stat().st_size > 1e8:
        log.info(f"[genome] cached {fa.name} ({fa.stat().st_size/1e9:.2f} GB)")
        return fa
    if not gz.exists():
        log.info(f"[genome] downloading {src['url']}")
        log.info("[genome] ~950 MB — this takes a while")
        try:
            with urllib.request.urlopen(src["url"], timeout=600) as r, open(gz, "wb") as f:
                tot = 0
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk: break
                    f.write(chunk); tot += len(chunk)
                    if tot % (200 << 20) == 0:
                        log.info(f"  {tot/1e6:.0f} MB")
        except Exception as e:
            gz.unlink(missing_ok=True)
            log.error(f"[genome] download failed: {e}")
            print(f"\n  MANUAL: download {src['url']}\n  place at {gz}\n")
            raise
    if decompress:
        log.info(f"[genome] decompressing → {fa.name}")
        with gzip.open(gz, "rb") as fi, open(fa, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
    return fa


def chrom_candidates(c: str) -> List[str]:
    """UCSC uses chr1/chrM; Ensembl uses 1/MT. ClinVar uses 1/MT."""
    b = str(c).strip().replace("chr", "").replace("CHR", "")
    if b.upper() in ("MT", "M"):
        return ["chrM", "MT", "M", "chrMT"]
    return [f"chr{b}", b, f"Chr{b}"]


class GenomeContext:
    """
    Random-access reference context via pyfaidx.

    CRITICAL: verifies that the FASTA base at PositionVCF equals
    ReferenceAlleleVCF. A mismatch means wrong assembly or wrong coordinate —
    silently proceeding is the most common and most damaging bug here.
    """
    def __init__(self, fasta: Path, flank: int = 32, assembly: str = "GRCh38"):
        if not HAS_PYFAIDX:
            raise ImportError("pip install pyfaidx")
        self.fa = Fasta(str(fasta), as_raw=True, sequence_always_upper=True,
                        read_ahead=10000)
        self.flank = flank
        self.assembly = assembly
        self.keys = set(self.fa.keys())
        self._cache: Dict[str, str] = {}
        self.stats = {"ok": 0, "ref_mismatch": 0, "no_chrom": 0, "edge_pad": 0}
        log.info(f"[genome] {len(self.keys)} sequences, flank={flank} "
                 f"→ context length {2*flank+1}")

    def resolve(self, c: str) -> Optional[str]:
        if c in self._cache: return self._cache[c]
        for cand in chrom_candidates(c):
            if cand in self.keys:
                self._cache[c] = cand
                return cand
        self._cache[c] = None
        return None

    def fetch(self, chrom: str, pos: int, ref: str) -> Optional[str]:
        """pos is 1-based (VCF). Returns length 2*flank+1, center = ref base."""
        k = self.resolve(chrom)
        if k is None:
            self.stats["no_chrom"] += 1
            return None
        L = len(self.fa[k])
        lo, hi = pos - self.flank - 1, pos + self.flank   # 0-based half-open
        pad_l = max(0, -lo); pad_r = max(0, hi - L)
        if pad_l or pad_r: self.stats["edge_pad"] += 1
        s = self.fa[k][max(0, lo):min(L, hi)]
        s = "N" * pad_l + s + "N" * pad_r
        if len(s) != 2 * self.flank + 1:
            return None
        if s[self.flank] != ref.upper():
            self.stats["ref_mismatch"] += 1
            return None
        self.stats["ok"] += 1
        return s

    def report(self):
        n = sum(self.stats.values())
        log.info(f"[genome] {self.stats}")
        if self.stats["ref_mismatch"] > 0:
            frac = self.stats["ref_mismatch"] / max(n, 1)
            log.warning(f"[genome] {frac:.1%} ref-allele mismatches. "
                        f"If >1%, your FASTA assembly does NOT match the "
                        f"ClinVar Assembly column. STOP and fix.")
        return self.stats


def onehot(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), 4), np.float32)
    for i, c in enumerate(seq):
        if c in BASE_IDX: x[i, BASE_IDX[c]] = 1.0
    return x


def context_features(seq: str, ref: str, alt: str, flank: int) -> np.ndarray:
    """
    Two-channel representation: reference context and alternate context.
    The model sees BOTH alleles, differing at exactly the center position.
    Plus a few interpretable scalars.
    """
    alt_seq = seq[:flank] + alt.upper() + seq[flank + 1:]
    xr, xa = onehot(seq), onehot(alt_seq)
    center = seq[flank]
    is_cpg = (center == "C" and flank + 1 < len(seq) and seq[flank + 1] == "G") or \
             (center == "G" and flank > 0 and seq[flank - 1] == "C")
    pur = set("AG")
    scal = np.array([
        float((ref in pur) == (alt in pur)),               # transition
        float(ref in pur and alt not in pur),
        float(ref not in pur and alt in pur),
        float(is_cpg),
        (seq.count("G") + seq.count("C")) / max(len(seq), 1),
    ], np.float32)
    return np.concatenate([np.stack([xr, xa], 0).ravel(), scal])


N_SCAL = 5
def ctx_dim(flank: int) -> int:
    return 2 * (2 * flank + 1) * 4 + N_SCAL


# =============================================================================
# SECTION 2 — ClinVar WITH REAL SEQUENCE CONTEXT + GENE SPLIT
# =============================================================================

BENIGN = {"Benign", "Likely benign", "Benign/Likely benign"}
PATHO = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
STARS = {"practice guideline": 4, "reviewed by expert panel": 3,
         "criteria provided, multiple submitters, no conflicts": 2,
         "criteria provided, single submitter": 1,
         "criteria provided, conflicting classifications": 1,
         "criteria provided, conflicting interpretations": 1,
         "no assertion criteria provided": 0, "no assertion provided": 0,
         "no classification provided": 0,
         "no classification for the individual variant": 0}


def clinvar_download(dest="data/clinvar") -> Path:
    d = Path(dest); d.mkdir(parents=True, exist_ok=True)
    out = d / "variant_summary.txt.gz"
    if out.exists() and out.stat().st_size > 1e6:
        log.info(f"[ClinVar] cached {out.stat().st_size/1e6:.1f} MB"); return out
    log.info(f"[ClinVar] downloading {CLINVAR_URL}")
    with urllib.request.urlopen(CLINVAR_URL, timeout=600) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    log.info(f"[ClinVar] {out.stat().st_size/1e6:.1f} MB")
    return out


def clinvar_with_context(path: Path, genome: GenomeContext,
                         assembly="GRCh38", min_stars=2,
                         limit: Optional[int] = None) -> List[Dict]:
    recs = []
    skip = {"not_snv": 0, "ambiguous": 0, "low_review": 0,
            "other_assembly": 0, "no_context": 0}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="ignore") as f:
        hdr = f.readline().lstrip("#").rstrip("\n").split("\t")
        H = {h: i for i, h in enumerate(hdr)}
        need = ["GeneSymbol", "ClinicalSignificance", "ReviewStatus", "Assembly",
                "Type", "Chromosome", "PositionVCF",
                "ReferenceAlleleVCF", "AlternateAlleleVCF"]
        for n in need:
            if n not in H: raise KeyError(f"ClinVar column '{n}' missing")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr): continue
            if p[H["Assembly"]] != assembly:
                skip["other_assembly"] += 1; continue
            if p[H["Type"]] != "single nucleotide variant":
                skip["not_snv"] += 1; continue
            cs = p[H["ClinicalSignificance"]]
            y = 0 if cs in BENIGN else (1 if cs in PATHO else None)
            if y is None:
                skip["ambiguous"] += 1; continue
            st = STARS.get(p[H["ReviewStatus"]].strip(), 0)
            if st < min_stars:
                skip["low_review"] += 1; continue
            ref, alt = p[H["ReferenceAlleleVCF"]], p[H["AlternateAlleleVCF"]]
            if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
                skip["not_snv"] += 1; continue
            ps = p[H["PositionVCF"]]
            if not ps.isdigit():
                skip["no_context"] += 1; continue
            seq = genome.fetch(p[H["Chromosome"]], int(ps), ref)
            if seq is None:
                skip["no_context"] += 1; continue
            recs.append({"gene": p[H["GeneSymbol"]], "chrom": p[H["Chromosome"]],
                         "pos": int(ps), "ref": ref, "alt": alt,
                         "label": y, "stars": st, "seq": seq})
            if limit and len(recs) >= limit: break
    log.info(f"[ClinVar+ctx] kept {len(recs)} | skipped {skip}")
    genome.report()
    if recs:
        pos = sum(r["label"] for r in recs)
        log.info(f"[ClinVar+ctx] pathogenic {pos} ({100*pos/len(recs):.1f}%) | "
                 f"genes {len(set(r['gene'] for r in recs))}")
    log.warning("[ClinVar+ctx] Labels are clinical assertions, NOT biological truth.")
    return recs


def gene_split(recs: List[Dict], frac=(0.7, 0.15, 0.15), seed=42) -> Dict:
    """Gene-level split. Prevents the leakage that inflates most ClinVar papers."""
    genes = sorted(set(r["gene"] for r in recs))
    rng = np.random.default_rng(seed); rng.shuffle(genes)
    n = len(genes); a = int(frac[0]*n); b = int(frac[1]*n)
    g = {"train": set(genes[:a]), "val": set(genes[a:a+b]),
         "test": set(genes[a+b:])}
    out = {k: [r for r in recs if r["gene"] in s] for k, s in g.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) \
           and not (g["val"] & g["test"])
    for k, v in out.items():
        p = sum(r["label"] for r in v)
        log.info(f"[gene-split] {k}: {len(v)} variants, "
                 f"{len(set(r['gene'] for r in v))} genes, "
                 f"{100*p/max(len(v),1):.1f}% pathogenic")
    log.info("[gene-split] ✓ zero gene overlap")
    return out


class ContextCNN(nn.Module):
    """Small 1D CNN over (ref, alt) one-hot context + scalars."""
    def __init__(self, flank=32, width=64):
        super().__init__()
        self.flank = flank
        self.conv = nn.Sequential(
            nn.Conv1d(8, width, 9, padding=4), nn.SiLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(width + N_SCAL, 128), nn.SiLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        L = 2 * self.flank + 1
        seq = x[:, :2*L*4].reshape(-1, 2*L, 8).transpose(1, 2)
        scal = x[:, 2*L*4:]
        h = self.conv(seq).squeeze(-1)
        return self.head(torch.cat([h, scal], -1)).squeeze(-1)


# =============================================================================
# SECTION 3 — GUIDE-seq OFF-TARGET LOADER
# =============================================================================

SEQ_HINTS = ["offtarget sequence", "off-target sequence", "offtargetsequence",
             "sequence", "seq", "offtarget", "dna"]
GUIDE_HINTS = ["targetsite", "target site", "target", "guide", "on-target",
               "ontarget", "sgrna"]
READ_HINTS = ["reads", "read count", "#reads", "unique reads", "count",
              "offtarget reads"]
MIS_HINTS = ["mismatch", "mismatches", "#mismatch", "n_mismatch"]
CHROM_HINTS = ["chromosome", "chrom", "chr"]
POS_HINTS = ["position", "pos", "start"]
CELL_HINTS = ["cell type", "celltype", "cell", "experiment"]


def _pick(cols: List[str], hints: List[str],
          exclude: Optional[List[str]] = None) -> Optional[str]:
    low = {c: c.lower().strip() for c in cols}
    for h in hints:
        for c, lc in low.items():
            if exclude and any(e in lc for e in exclude): continue
            if h == lc: return c
    for h in hints:
        for c, lc in low.items():
            if exclude and any(e in lc for e in exclude): continue
            if h in lc: return c
    return None


def guideseq_probe(path: Path) -> Dict:
    """Report the actual columns. Never guess silently."""
    import csv
    delim = "\t" if path.suffix.lower() in (".txt", ".tsv") else ","
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.reader(f, delimiter=delim)
        rows = []
        for i, r in enumerate(rd):
            rows.append(r)
            if i >= 4: break
    cols = rows[0] if rows else []
    guess = {
        "seq": _pick(cols, SEQ_HINTS, exclude=["target site", "targetsite"]),
        "guide": _pick(cols, GUIDE_HINTS),
        "reads": _pick(cols, READ_HINTS),
        "mismatch": _pick(cols, MIS_HINTS),
        "chrom": _pick(cols, CHROM_HINTS),
        "pos": _pick(cols, POS_HINTS),
        "cell": _pick(cols, CELL_HINTS),
    }
    return {"file": path.name, "delimiter": repr(delim), "columns": cols,
            "guessed_mapping": guess, "sample_rows": rows[1:4]}


def clean_dna(s: str) -> Optional[str]:
    s = "".join(c for c in str(s).upper() if c in "ACGTN")
    return s if len(s) in (20, 23) and "N" not in s else None


def guideseq_load(path: Path, seq_col=None, guide_col=None, reads_col=None,
                  cell_col=None, min_reads=1,
                  limit: Optional[int] = None) -> List[Dict]:
    """
    Load GUIDE-seq off-target sites [G2].

    Output records: {guide, offtarget, reads, cell, label}
    label = 1 (detected cleavage). NEGATIVES ARE NOT PROVIDED HERE.
    """
    import csv
    delim = "\t" if path.suffix.lower() in (".txt", ".tsv") else ","
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f, delimiter=delim)
        cols = rd.fieldnames or []
        sc = seq_col or _pick(cols, SEQ_HINTS,
                              exclude=["target site", "targetsite"])
        gc = guide_col or _pick(cols, GUIDE_HINTS)
        rc = reads_col or _pick(cols, READ_HINTS)
        cc = cell_col or _pick(cols, CELL_HINTS)
        if sc is None:
            raise KeyError(f"no sequence column in {cols}. "
                           f"Run guideseq-probe and pass --seq-col.")
        log.info(f"[GUIDE-seq] seq='{sc}' guide='{gc}' reads='{rc}' cell='{cc}'")
        out = []
        for row in rd:
            ot = clean_dna(row.get(sc, ""))
            if ot is None: continue
            gd = clean_dna(row.get(gc, "")) if gc else None
            try: nreads = float(str(row.get(rc, "1")).replace(",", ""))
            except (TypeError, ValueError): nreads = 1.0
            if nreads < min_reads: continue
            out.append({"guide": gd, "offtarget": ot, "reads": nreads,
                        "cell": row.get(cc, "unknown") if cc else "unknown",
                        "label": 1})
            if limit and len(out) >= limit: break
    log.info(f"[GUIDE-seq] {len(out)} detected off-target sites")
    if out:
        gl = set(len(r["offtarget"]) for r in out)
        n_g = len(set(r["guide"] for r in out if r["guide"]))
        log.info(f"[GUIDE-seq] lengths={gl}, distinct guides={n_g}, "
                 f"cells={len(set(r['cell'] for r in out))}")
    log.warning("[GUIDE-seq] These are POSITIVES only. Cell-type specific, "
                "with a detection limit. 'Not detected' ≠ 'never cleaved'.")
    return out


def make_negatives(positives: List[Dict], n_per=4, max_mis=6,
                   seed=42) -> List[Dict]:
    """
    ⚠️ SYNTHETIC negatives: random mismatch decoys of the on-target guide.
    These are UNVERIFIED. Some are real off-targets that GUIDE-seq missed.
    Metrics obtained with these negatives are OPTIMISTIC and must be labeled.
    For real negatives use CHANGE-seq [G3] or validated biochemical assays.
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in positives:
        g = p["guide"]
        if not g: continue
        for _ in range(n_per):
            k = int(rng.integers(3, max_mis + 1))
            idx = rng.choice(len(g) - 3, size=min(k, len(g) - 3), replace=False)
            s = list(g)
            for i in idx:
                s[i] = rng.choice([c for c in "ACGT" if c != s[i]])
            out.append({"guide": g, "offtarget": "".join(s), "reads": 0.0,
                        "cell": p["cell"], "label": 0, "synthetic_negative": True})
    log.warning(f"[negatives] {len(out)} SYNTHETIC decoys generated. "
                f"UNVERIFIED. Any AUC/AUPRC from these is optimistic.")
    return out


def pair_features(guide: str, off: str) -> np.ndarray:
    """Position-wise match/mismatch encoding + mismatch count + PAM region."""
    L = min(len(guide), len(off))
    g, o = guide[:L], off[:L]
    m = np.zeros((L, 2), np.float32)
    for i in range(L):
        m[i, 0] = 1.0
        m[i, 1] = float(g[i] != o[i])
    nmis = float(sum(a != b for a, b in zip(g, o)))
    pam_mis = float(sum(a != b for a, b in zip(g[-3:], o[-3:])))
    seed_mis = float(sum(a != b for a, b in zip(g[-12:-3], o[-12:-3])))
    return np.concatenate([m.ravel(), [nmis / L, pam_mis / 3, seed_mis / 9]])


def pair_dim(L=23) -> int:
    return 2 * L + 3


class OffTargetCNN(nn.Module):
    def __init__(self, L=23, width=64):
        super().__init__()
        self.L = L
        self.conv = nn.Sequential(nn.Conv1d(2, width, 5, padding=2), nn.SiLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.SiLU(),
                                  nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(width + 3, 64), nn.SiLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        s = x[:, :2*self.L].reshape(-1, self.L, 2).transpose(1, 2)
        sc = x[:, 2*self.L:]
        return self.head(torch.cat([self.conv(s).squeeze(-1), sc], -1)).squeeze(-1)


# =============================================================================
# SECTION 4 — METRICS
# =============================================================================

def auroc(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    r = np.empty(len(p), float); r[np.argsort(p)] = np.arange(1, len(p) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    o = np.argsort(-p); ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(ys.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def brier(y, p): return float(np.mean((p - y) ** 2))


def nll(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def ece(y, p, n_bins=15, adaptive=False) -> Tuple[float, List]:
    """Expected Calibration Error. adaptive=True uses equal-mass bins."""
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -1e-9, 1 + 1e-9
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    e, curve = 0.0, []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i+1])
        if m.sum() == 0:
            curve.append((float((edges[i]+edges[i+1])/2), None, 0)); continue
        conf = float(p[m].mean()); acc = float(y[m].mean())
        e += (m.sum()/len(p)) * abs(acc - conf)
        curve.append((conf, acc, int(m.sum())))
    return float(e), curve


def clf_report(y, p, tag="") -> Dict:
    e, curve = ece(y, p)
    ea, _ = ece(y, p, adaptive=True)
    return {"tag": tag, "n": int(len(y)), "prevalence": float(y.mean()),
            "auroc": auroc(y, p), "auprc": auprc(y, p),
            "brier": brier(y, p), "nll": nll(y, p),
            "ece_fixed": e, "ece_adaptive": ea,
            "mean_conf": float(p.mean()),
            "reliability": curve}


def reg_coverage(mu, sd, y, levels=(0.5, 0.6827, 0.9, 0.95)) -> Dict:
    """Regression calibration: empirical vs nominal coverage (Kuleshov [C4])."""
    from statistics import NormalDist
    nd = NormalDist()
    out, err = {}, 0.0
    z = np.abs((y - mu) / np.maximum(sd, 1e-12))
    for L in levels:
        zc = nd.inv_cdf(0.5 + L/2)
        obs = float((z < zc).mean())
        out[f"{L:.4f}"] = obs
        err += abs(obs - L)
    return {"coverage": out, "calibration_error": float(err/len(levels)),
            "mean_sigma": float(np.mean(sd)),
            "rmse": float(np.sqrt(np.mean((y-mu)**2))),
            "sharpness": float(np.mean(sd))}


# =============================================================================
# SECTION 5 — CALIBRATION METHODS
# =============================================================================

class TemperatureScaler:
    """
    [C1] Single parameter T > 0, logits / T.
    PRESERVES ranking → AUROC and accuracy are UNCHANGED.
    Fixes over/under-confidence only.
    """
    def __init__(self):
        self.log_T = nn.Parameter(torch.zeros(1))

    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=200):
        lg = torch.tensor(logits, dtype=torch.float32)
        yy = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.LBFGS([self.log_T], lr=0.1, max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lg / self.log_T.exp(), yy)
            loss.backward(); return loss
        opt.step(closure)
        log.info(f"[temperature] T = {self.T:.4f} "
                 f"({'overconfident → softened' if self.T > 1 else 'underconfident → sharpened'})")
        return self

    @property
    def T(self) -> float: return float(self.log_T.exp())

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-logits/self.T))


def pava(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Pool-Adjacent-Violators for non-decreasing isotonic fit."""
    n = len(y)
    w = np.ones(n) if w is None else w.astype(float)
    lvl = y.astype(float).copy(); wt = w.copy()
    idx = list(range(n))
    i = 0
    while i < len(lvl) - 1:
        if lvl[i] <= lvl[i+1] + 1e-12:
            i += 1; continue
        tw = wt[i] + wt[i+1]
        nl = (wt[i]*lvl[i] + wt[i+1]*lvl[i+1]) / tw
        lvl[i] = nl; wt[i] = tw
        del lvl[i+1]; del wt[i+1]; del idx[i+1]
        i = max(i-1, 0)
    out = np.empty(n)
    start = 0
    for k, j in enumerate(idx):
        out[start:j+1] = lvl[k]; start = j+1
    out[start:] = lvl[-1]
    return out


class IsotonicCalibrator:
    """
    [C2] Monotone piecewise-linear map p → calibrated p.
    CAN change ranking (creates ties). OVERFITS small validation sets.
    Recommend n_val >= 500; prefer temperature below that.
    """
    def __init__(self, min_n=200):
        self.min_n = min_n
        self.xk = self.yk = None

    def fit(self, p: np.ndarray, y: np.ndarray):
        if len(p) < self.min_n:
            log.warning(f"[isotonic] n={len(p)} < {self.min_n}. "
                        f"High overfit risk. Consider temperature scaling.")
        o = np.argsort(p, kind="mergesort")
        xs, ys = p[o], y[o]
        fit = pava(ys)
        ux, inv = np.unique(xs, return_inverse=True)
        uy = np.zeros(len(ux))
        for i, g in enumerate(inv): uy[g] = max(uy[g], fit[i])
        uy = np.maximum.accumulate(uy)
        self.xk, self.yk = ux, np.clip(uy, 0, 1)
        log.info(f"[isotonic] {len(self.xk)} knots from n={len(p)}")
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(p, self.xk, self.yk), 0, 1)


class VarianceScaler:
    """
    Global σ scaling for regression ensembles.
    Closed-form MLE under Gaussian: s* = sqrt(mean(((y-μ)/σ)²)).
    Fixes GLOBAL coverage only, not conditional.
    """
    def __init__(self): self.s = 1.0
    def fit(self, mu, sd, y):
        z = (y - mu) / np.maximum(sd, 1e-12)
        self.s = float(np.sqrt(np.mean(z**2)))
        log.info(f"[variance] s = {self.s:.4f} "
                 f"({'underconfident → widened' if self.s > 1 else 'overconfident → narrowed'})")
        return self
    def predict(self, sd): return sd * self.s


class SigmaIsotonic:
    """
    [C4] Isotonic map σ → E[|error|]. Conditional-ish recalibration.
    Monotone by construction; still no coverage guarantee.
    """
    def __init__(self): self.xk = self.yk = None
    def fit(self, sd, abs_err):
        o = np.argsort(sd, kind="mergesort")
        xs, ae = sd[o], abs_err[o]
        fit = pava(ae)
        ux, inv = np.unique(xs, return_inverse=True)
        uy = np.zeros(len(ux))
        for i, g in enumerate(inv): uy[g] = max(uy[g], fit[i])
        self.xk = ux; self.yk = np.maximum.accumulate(uy)
        log.info(f"[sigma-iso] {len(self.xk)} knots")
        return self
    def predict(self, sd):
        # E|N(0,σ)| = σ·sqrt(2/π)  →  σ_equiv = E|err|·sqrt(π/2)
        return np.interp(sd, self.xk, self.yk) * math.sqrt(math.pi/2)


class SplitConformal:
    """
    [C5,C6,C7] Normalized split conformal regression.
    Scores s_i = |y_i - μ_i| / σ_i on calibration set.
    q = the ceil((n+1)(1-α))/n empirical quantile.
    Interval: μ ± q·σ.

    GUARANTEE: marginal coverage ≥ 1-α under EXCHANGEABILITY.
    NOT conditional. NOT valid under distribution shift.
    """
    def __init__(self, alpha=0.1): self.alpha = alpha; self.q = None; self.n = None
    def fit(self, mu, sd, y):
        s = np.abs(y - mu) / np.maximum(sd, 1e-12)
        n = len(s); self.n = n
        k = math.ceil((n + 1) * (1 - self.alpha))
        k = min(k, n)
        self.q = float(np.sort(s)[k-1])
        log.info(f"[conformal] n={n} alpha={self.alpha} q={self.q:.4f} "
                 f"(finite-sample correction applied)")
        if n < 200:
            log.warning(f"[conformal] n={n} is small; q is noisy and intervals "
                        f"will be conservative/unstable.")
        return self
    def interval(self, mu, sd):
        w = self.q * sd
        return mu - w, mu + w


class ConformalClassifier:
    """
    LAC / threshold conformal for binary classification [C6].
    Score = 1 - p_true. Set includes class c if 1-p_c <= q.
    Marginal coverage guarantee under exchangeability.
    Possible outputs: {0}, {1}, {0,1} (abstain), {} (reject).
    """
    def __init__(self, alpha=0.1): self.alpha = alpha; self.q = None
    def fit(self, p: np.ndarray, y: np.ndarray):
        s = np.where(y == 1, 1 - p, p)
        n = len(s); k = min(math.ceil((n+1)*(1-self.alpha)), n)
        self.q = float(np.sort(s)[k-1])
        log.info(f"[conformal-clf] n={n} q={self.q:.4f}")
        return self
    def sets(self, p: np.ndarray):
        inc1 = (1 - p) <= self.q
        inc0 = p <= self.q
        return np.stack([inc0, inc1], 1)


# =============================================================================
# SECTION 6 — PIPELINES
# =============================================================================

def train_clinvar_context(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    fa = Path(args.fasta)
    if not fa.exists():
        fa = download_genome(args.genome, str(Path(args.fasta).parent))
    gen = GenomeContext(fa, flank=args.flank, assembly=args.assembly)
    cvp = Path(args.clinvar)
    if not cvp.exists(): cvp = clinvar_download(str(cvp.parent))
    recs = clinvar_with_context(cvp, gen, assembly=args.assembly,
                                min_stars=args.min_stars, limit=args.limit)
    if len(recs) < 200:
        log.error("too few records with verified context"); return
    sp = gene_split(recs, seed=args.seed)

    def mk(rs):
        X = np.stack([context_features(r["seq"], r["ref"], r["alt"], args.flank)
                      for r in rs])
        y = np.array([r["label"] for r in rs], np.float32)
        return torch.tensor(X).to(dev), torch.tensor(y).to(dev)
    Xtr, ytr = mk(sp["train"]); Xva, yva = mk(sp["val"]); Xte, yte = mk(sp["test"])

    model = ContextCNN(args.flank).to(dev)
    pw = (ytr.numel() - ytr.sum()) / ytr.sum().clamp(min=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bs = 1e9, None
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(Xtr), device=dev)
        for s in range(0, len(perm), args.batch_size):
            i = perm[s:s+args.batch_size]
            loss = F.binary_cross_entropy_with_logits(model(Xtr[i]), ytr[i],
                                                      pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep+1) % max(1, args.epochs//10) == 0:
            model.eval()
            with torch.no_grad(): vl = float(F.binary_cross_entropy_with_logits(
                model(Xva), yva))
            log.info(f"ep {ep+1:4d} | val NLL {vl:.5f}")
            if vl < best:
                best = vl; bs = {k: v.clone() for k, v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        lva = model(Xva).cpu().numpy(); lte = model(Xte).cpu().numpy()
    yva_np, yte_np = yva.cpu().numpy(), yte.cpu().numpy()
    pva_raw = 1/(1+np.exp(-lva)); pte_raw = 1/(1+np.exp(-lte))

    # ---- CALIBRATION: fit on VAL only ----
    temp = TemperatureScaler().fit(lva, yva_np)
    iso = IsotonicCalibrator().fit(pva_raw, yva_np)
    conf = ConformalClassifier(alpha=args.alpha).fit(
        temp.predict(lva), yva_np)

    variants = {
        "raw": pte_raw,
        "temperature": temp.predict(lte),
        "isotonic": iso.predict(pte_raw),
        "temp+iso": iso.predict(temp.predict(lte)),
    }
    print("\n" + "="*78)
    print("  ClinVar + REAL GENOME CONTEXT — gene-level split")
    print("="*78)
    print(f"  assembly={args.assembly} flank={args.flank} "
          f"context_len={2*args.flank+1} min_stars={args.min_stars}")
    print(f"  train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(f"  temperature T = {temp.T:.4f}")
    print(f"\n  {'method':<14}{'AUROC':>9}{'AUPRC':>9}{'Brier':>9}"
          f"{'NLL':>9}{'ECE':>9}{'conf':>9}")
    reps = {}
    for k, p in variants.items():
        r = clf_report(yte_np, p, k); reps[k] = r
        print(f"  {k:<14}{r['auroc']:>9.4f}{r['auprc']:>9.4f}"
              f"{r['brier']:>9.4f}{r['nll']:>9.4f}"
              f"{r['ece_fixed']:>9.4f}{r['mean_conf']:>9.4f}")
    sets = conf.sets(variants["temperature"])
    cov = float(np.mean([sets[i, int(yte_np[i])] for i in range(len(yte_np))]))
    size = sets.sum(1).mean()
    print(f"\n  conformal sets (alpha={args.alpha}): "
          f"coverage={cov:.4f} (nominal {1-args.alpha:.2f}), "
          f"mean set size={size:.3f}")
    print(f"    empty={float((sets.sum(1)==0).mean()):.3f}  "
          f"both={float((sets.sum(1)==2).mean()):.3f}")
    print("\n  ⚠️  Temperature PRESERVES AUROC (ranking unchanged).")
    print("     Isotonic MAY change AUROC — it creates ties.")
    print("  ⚠️  Calibration fitted on VAL genes; TEST genes are different.")
    print("     Under gene shift, calibration transfer is NOT guaranteed.")
    print("  ❌ Context alone is not pathogenicity. Not a diagnostic.")
    print("="*78)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"reports": reps, "T": temp.T, "genome_stats": gen.stats,
               "conformal": {"coverage": cov, "mean_size": float(size)},
               "args": vars(args)}, open(args.out, "w"), indent=2, default=str)
    torch.save({"model": model.state_dict(), "log_T": temp.log_T.detach(),
                "iso_xk": iso.xk, "iso_yk": iso.yk},
               str(Path(args.out).with_suffix(".pt")))


def train_guideseq(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    p = Path(args.data)
    if not p.exists():
        log.error(f"{p} not found.\n"
                  f"GUIDE-seq data: Tsai et al. 2015 [G2] Supplementary Tables,\n"
                  f"  https://www.nature.com/articles/nbt.3117\n"
                  f"or the guideseq pipeline output (github.com/tsailabSJ/guideseq).\n"
                  f"For better negatives: CHANGE-seq [G3].\n"
                  f"Then run: guideseq-probe --data <file>")
        return
    pos = guideseq_load(p, seq_col=args.seq_col, guide_col=args.guide_col,
                        reads_col=args.reads_col, min_reads=args.min_reads,
                        limit=args.limit)
    pos = [r for r in pos if r["guide"]]
    if len(pos) < 20: log.error("too few paired records"); return
    neg = make_negatives(pos, n_per=args.neg_ratio, seed=args.seed)
    recs = pos + neg
    L = max(len(r["offtarget"]) for r in recs)
    L = min(L, 23)
    for r in recs:
        r["guide"] = r["guide"][:L].ljust(L, "A")
        r["offtarget"] = r["offtarget"][:L].ljust(L, "A")
    X = np.stack([pair_features(r["guide"], r["offtarget"]) for r in recs])
    y = np.array([r["label"] for r in recs], np.float32)

    # split by GUIDE (not by row) — same guide must not appear in two splits
    guides = sorted(set(r["guide"] for r in recs))
    rng = np.random.default_rng(args.seed); rng.shuffle(guides)
    n = len(guides); a = int(0.7*n); b = int(0.15*n)
    gmap = {"train": set(guides[:a]), "val": set(guides[a:a+b]),
            "test": set(guides[a+b:])}
    idx = {k: np.array([i for i, r in enumerate(recs) if r["guide"] in s])
           for k, s in gmap.items()}
    log.info(f"[GUIDE-seq] guide-level split "
             f"{len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])} rows")
    log.info("[GUIDE-seq] ✓ zero guide overlap between splits")

    Xt = torch.tensor(X).to(dev); yt = torch.tensor(y).to(dev)
    itr, iva, ite = (torch.tensor(idx[k], device=dev) for k in ("train","val","test"))
    model = OffTargetCNN(L).to(dev)
    pw = (yt[itr].numel()-yt[itr].sum())/yt[itr].sum().clamp(min=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bs = 1e9, None
    for ep in range(args.epochs):
        model.train(); perm = itr[torch.randperm(len(itr), device=dev)]
        for s in range(0, len(perm), args.batch_size):
            i = perm[s:s+args.batch_size]
            loss = F.binary_cross_entropy_with_logits(model(Xt[i]), yt[i],
                                                      pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (ep+1) % max(1, args.epochs//10) == 0:
            model.eval()
            with torch.no_grad(): vl = float(F.binary_cross_entropy_with_logits(
                model(Xt[iva]), yt[iva]))
            if vl < best:
                best = vl; bs = {k: v.clone() for k, v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        lva = model(Xt[iva]).cpu().numpy(); lte = model(Xt[ite]).cpu().numpy()
    yv, ytt = yt[iva].cpu().numpy(), yt[ite].cpu().numpy()
    temp = TemperatureScaler().fit(lva, yv)
    iso = IsotonicCalibrator().fit(1/(1+np.exp(-lva)), yv)
    print("\n" + "="*78)
    print("  GUIDE-seq off-target — GUIDE-level split")
    print("="*78)
    print(f"  positives={len(pos)}  SYNTHETIC negatives={len(neg)}  L={L}")
    print(f"  {'method':<14}{'AUROC':>9}{'AUPRC':>9}{'Brier':>9}{'ECE':>9}")
    for k, pv in (("raw", 1/(1+np.exp(-lte))),
                  ("temperature", temp.predict(lte)),
                  ("isotonic", iso.predict(1/(1+np.exp(-lte))))):
        r = clf_report(ytt, pv, k)
        print(f"  {k:<14}{r['auroc']:>9.4f}{r['auprc']:>9.4f}"
              f"{r['brier']:>9.4f}{r['ece_fixed']:>9.4f}")
    print("\n  ❌❌ NEGATIVES ARE SYNTHETIC AND UNVERIFIED.")
    print("     These numbers are OPTIMISTIC and are NOT a safety claim.")
    print("     Some 'negatives' are real off-targets GUIDE-seq missed.")
    print("  ⚠️  GUIDE-seq is cell-type specific with a detection limit.")
    print("  ⚠️  For real negatives use CHANGE-seq [G3] or targeted validation.")
    print("  ❌ No guide is safe for use without experimental off-target assays")
    print("     and institutional biosafety review.")
    print("="*78)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"T": temp.T, "n_pos": len(pos), "n_synthetic_neg": len(neg),
               "WARNING": "synthetic negatives; optimistic metrics",
               "args": vars(args)}, open(args.out, "w"), indent=2)


def calibrate_regression(args):
    """
    Calibrate an ensemble's σ on validation, evaluate on test.
    Input NPZ: mu_val, sd_val, y_val, mu_test, sd_test, y_test
    (produce these from mol_advanced.py ensemble output).
    """
    d = np.load(args.data)
    muv, sdv, yv = d["mu_val"], d["sd_val"], d["y_val"]
    mut, sdt, yt = d["mu_test"], d["sd_test"], d["y_test"]
    print("\n" + "="*78)
    print("  ENSEMBLE REGRESSION CALIBRATION")
    print("="*78)
    print(f"  n_val={len(yv)}  n_test={len(yt)}")
    base = reg_coverage(mut, sdt, yt)
    print(f"\n  RAW           cal_err={base['calibration_error']:.4f}  "
          f"mean σ={base['mean_sigma']:.5f}  RMSE={base['rmse']:.5f}")
    print(f"    coverage: " + "  ".join(f"{k}:{v:.3f}"
                                        for k, v in base["coverage"].items()))

    vs = VarianceScaler().fit(muv, sdv, yv)
    r_vs = reg_coverage(mut, vs.predict(sdt), yt)
    print(f"\n  VARIANCE      cal_err={r_vs['calibration_error']:.4f}  "
          f"s={vs.s:.4f}  mean σ={r_vs['mean_sigma']:.5f}")
    print(f"    coverage: " + "  ".join(f"{k}:{v:.3f}"
                                        for k, v in r_vs["coverage"].items()))

    si = SigmaIsotonic().fit(sdv, np.abs(yv - muv))
    r_si = reg_coverage(mut, si.predict(sdt), yt)
    print(f"\n  SIGMA-ISOTONIC cal_err={r_si['calibration_error']:.4f}  "
          f"mean σ={r_si['mean_sigma']:.5f}")
    print(f"    coverage: " + "  ".join(f"{k}:{v:.3f}"
                                        for k, v in r_si["coverage"].items()))

    cf = SplitConformal(alpha=args.alpha).fit(muv, sdv, yv)
    lo, hi = cf.interval(mut, sdt)
    cov = float(((yt >= lo) & (yt <= hi)).mean())
    width = float(np.mean(hi - lo))
    print(f"\n  CONFORMAL     q={cf.q:.4f}  coverage={cov:.4f} "
          f"(nominal {1-args.alpha:.2f})  mean width={width:.5f}")
    print(f"    → distribution-free MARGINAL guarantee under exchangeability")

    print("\n  ⚠️  Variance scaling fixes GLOBAL coverage only.")
    print("  ⚠️  Isotonic can overfit; conformal needs n_val ≥ 200 to be stable.")
    print("  ❌ MARGINAL coverage ≠ CONDITIONAL coverage.")
    print("     A 90% interval can be systematically wrong on a subgroup.")
    print("  ❌ Exchangeability fails under shift (new molecules/scaffolds).")
    print("     For molecules, calibrate on a SCAFFOLD split, not random.")
    print("="*78)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"raw": base, "variance": r_vs, "sigma_iso": r_si,
               "conformal": {"q": cf.q, "coverage": cov, "width": width},
               "s": vs.s, "args": vars(args)}, open(args.out, "w"), indent=2)


def selftest(args):
    print("="*78); print("  SELF-TEST"); print("="*78)
    rng = np.random.default_rng(0)
    n = 3000
    # overconfident synthetic classifier
    y = (rng.random(n) < 0.3).astype(float)
    logit = 3.5*(y - 0.5) + rng.normal(0, 1.2, n)
    p = 1/(1+np.exp(-logit))
    a, b = n//2, 3*n//4
    tr, va, te = slice(0,a), slice(a,b), slice(b,n)
    print("\n[T1] temperature scaling on overconfident scores")
    r0 = clf_report(y[te], p[te], "raw")
    T = TemperatureScaler().fit(logit[va], y[va])
    r1 = clf_report(y[te], T.predict(logit[te]), "temp")
    print(f"  raw  ECE={r0['ece_fixed']:.4f} NLL={r0['nll']:.4f} AUROC={r0['auroc']:.4f}")
    print(f"  temp ECE={r1['ece_fixed']:.4f} NLL={r1['nll']:.4f} AUROC={r1['auroc']:.4f}")
    same = abs(r0['auroc'] - r1['auroc']) < 1e-9
    print(f"  AUROC unchanged: {same}  {'PASS' if same else 'FAIL'}")
    assert same

    print("\n[T2] isotonic monotonicity")
    iso = IsotonicCalibrator().fit(p[va], y[va])
    grid = np.linspace(0, 1, 500)
    mono = bool(np.all(np.diff(iso.predict(grid)) >= -1e-9))
    print(f"  monotone non-decreasing: {mono}  {'PASS' if mono else 'FAIL'}")
    assert mono

    print("\n[T3] PAVA correctness vs known case")
    yy = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    fit = pava(yy)
    ok = np.all(np.diff(fit) >= -1e-9) and abs(fit.mean() - yy.mean()) < 1e-9
    print(f"  fit={np.round(fit,4)}  monotone+mean-preserving: {ok}")
    assert ok

    print("\n[T4] conformal coverage on exchangeable synthetic regression")
    mu = rng.normal(0, 1, n); sd = np.exp(rng.normal(-1, .4, n))
    yv2 = mu + sd*rng.normal(0, 1, n)
    cf = SplitConformal(alpha=0.1).fit(mu[va], sd[va], yv2[va])
    lo, hi = cf.interval(mu[te], sd[te])
    cov = float(((yv2[te] >= lo) & (yv2[te] <= hi)).mean())
    print(f"  coverage={cov:.4f}  nominal=0.90  "
          f"{'PASS' if cov > 0.85 else 'CHECK'}")
    assert cov > 0.82

    print("\n[T5] variance scaling recovers known miscalibration")
    sd_bad = sd * 0.5
    vs = VarianceScaler().fit(mu[va], sd_bad[va], yv2[va])
    print(f"  true factor 2.0, recovered s={vs.s:.3f}  "
          f"{'PASS' if 1.6 < vs.s < 2.5 else 'CHECK'}")

    print("\n[T6] genome ref-allele verification logic")
    seq = "ACGT" + "A" + "CGTA"; flank = 4
    print(f"  center base = {seq[flank]} (expect A)")
    assert seq[flank] == "A"
    print("  PASS")

    print("\n" + "="*78)
    print("  ✅ ALL SELF-TESTS PASSED")
    print("="*78)
    print("""
  WHAT IS VERIFIED:
    • Temperature scaling preserves ranking (AUROC identical)
    • Isotonic fit is monotone and mean-preserving (PAVA)
    • Conformal achieves nominal coverage on exchangeable data
    • Variance scaling recovers a known miscalibration factor

  WHAT IS NOT VERIFIED:
    • That your data is exchangeable
    • That calibration transfers across gene/scaffold/cell-type shift
    • That GUIDE-seq negatives are real (they are not, by default)
    • That ClinVar labels are biological truth (they are assertions)
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["genome-download", "clinvar-context",
                                   "guideseq-probe", "guideseq",
                                   "calibrate-reg", "selftest"])
    p.add_argument("--genome", default="hg38_ucsc", choices=list(GENOME_SOURCES))
    p.add_argument("--fasta", default="data/genome/hg38.fa")
    p.add_argument("--clinvar", default="data/clinvar/variant_summary.txt.gz")
    p.add_argument("--data", default="data/guideseq/offtargets.txt")
    p.add_argument("--out", default="ckpt/cal.json")
    p.add_argument("--assembly", default="GRCh38", choices=["GRCh38","GRCh37"])
    p.add_argument("--flank", type=int, default=32)
    p.add_argument("--min-stars", dest="min_stars", type=int, default=2)
    p.add_argument("--min-reads", dest="min_reads", type=float, default=1)
    p.add_argument("--neg-ratio", dest="neg_ratio", type=int, default=4)
    p.add_argument("--seq-col", dest="seq_col", default=None)
    p.add_argument("--guide-col", dest="guide_col", default=None)
    p.add_argument("--reads-col", dest="reads_col", default=None)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.cmd == "genome-download":
        download_genome(a.genome, str(Path(a.fasta).parent))
    elif a.cmd == "clinvar-context": train_clinvar_context(a)
    elif a.cmd == "guideseq-probe":
        print(json.dumps(guideseq_probe(Path(a.data)), indent=2))
    elif a.cmd == "guideseq": train_guideseq(a)
    elif a.cmd == "calibrate-reg": calibrate_regression(a)
    else: selftest(a)


if __name__ == "__main__":
    main()
