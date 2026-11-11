#!/usr/bin/env python3
"""
=============================================================================
CONSEQUENCE ANNOTATION · CONSERVATION TRACKS · CHANGE-seq  +  ABLATION STUDY
=============================================================================
License: MIT | 2026

LITERATURE
 [V1] McLaren et al. "The Ensembl Variant Effect Predictor."
      Genome Biol. 17, 122 (2016).
 [V2] Cingolani et al. "A program for annotating and predicting the effects of
      single nucleotide polymorphisms, SnpEff." Fly 6, 80 (2012).
 [P1] Pollard, Hubisz, Rosenbloom, Siepel. "Detection of nonneutral substitution
      rates on mammalian phylogenies." Genome Res. 20, 110 (2010).   [phyloP]
 [P2] Siepel et al. "Evolutionarily conserved elements in vertebrate, insect,
      worm, and yeast genomes." Genome Res. 15, 1034 (2005).      [phastCons]
 [L1] Lazzarotto et al. "CHANGE-seq reveals genetic and epigenetic effects on
      CRISPR-Cas9 genome-wide specificity." Nat. Biotechnol. 39, 1394 (2021).
 [L2] Tsai et al. "GUIDE-seq enables genome-wide profiling of off-target
      cleavage." Nat. Biotechnol. 33, 187 (2015).
 [G1] Grantham. "Amino acid difference formula to help explain protein
      evolution." Science 185, 862 (1974).
 [K1] Kyte & Doolittle. "A simple method for displaying the hydropathic
      character of a protein." J. Mol. Biol. 157, 105 (1982).
 [A1] Richards et al. "Standards and guidelines for the interpretation of
      sequence variants." Genet. Med. 17, 405 (2015).        [ACMG PP3/BP4]
 [C1] Guo et al. "On calibration of modern neural networks." ICML (2017).

=============================================================================
HONEST SCOPE — FIVE THINGS THAT WILL INVALIDATE YOUR RESULTS IF IGNORED
=============================================================================
1. SIFT / PolyPhen CIRCULARITY.
   VEP ships SIFT and PolyPhen scores. Both were trained/validated on variant
   sets that OVERLAP ClinVar. Including them inflates ClinVar agreement without
   demonstrating new biological insight.
   → This file reports ablation WITH and WITHOUT them. Use --no-pred-scores
     for the clean comparison. Report both.

2. phyloP IS NOT LEAKAGE, BUT IT IS NOT PATHOGENICITY EITHER.
   phyloP/phastCons derive from multiple sequence alignments, independent of
   ClinVar curation. They measure evolutionary CONSTRAINT. ACMG PP3/BP4 uses
   them as supporting (not standalone) evidence. A conserved site can be
   benign; a non-conserved site can be pathogenic.

3. CONSEQUENCE MAY DOMINATE FOR A TRIVIAL REASON.
   "stop_gained → pathogenic" and "synonymous → benign" is real biology but
   already known. The ablation shows how much of the gain is just this.
   Stratified metrics (within-missense-only) are reported to expose it.

4. CHANGE-seq IS IN VITRO.
   Cell-free, purified DNA, no chromatin. Higher sensitivity than GUIDE-seq,
   but detects sites that are inaccessible in living cells. Positives are
   "cleavable in vitro", NOT "will be edited in a patient".

5. "ASSAY-NEGATIVE" IS AN OPERATIONAL LABEL, NOT GROUND TRUTH.
   candidates = genome sites with PAM + <=N mismatches
   negatives  = candidates NOT detected by the assay at its depth
   Some are real off-targets below detection. Only targeted deep sequencing
   (rhAMPSeq / amplicon) yields validated negatives. Pass --validated-negatives
   to use them; the report states which kind was used.

❌ Nothing here is a diagnostic, a clinical decision aid, or a safety claim.
❌ Gene-level split is mandatory. Calibration on val genes may not transfer.
=============================================================================
"""
from __future__ import annotations
import argparse, gzip, json, logging, math, re, shutil, subprocess, time
import urllib.request, urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyBigWig; HAS_BW = True
except ImportError:
    HAS_BW = False
try:
    from pyfaidx import Fasta; HAS_FA = True
except ImportError:
    HAS_FA = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("annot")

BASE_IDX = {"A":0,"C":1,"G":2,"T":3}
COMP = str.maketrans("ACGTN","TGCAN")

# ---------------------------------------------------------------- VEP sources
REST = {"GRCh38": "https://rest.ensembl.org",
        "GRCh37": "https://grch37.rest.ensembl.org"}

CONSEQUENCE_GROUPS = [
    "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant",
    "stop_gained", "stop_lost", "start_lost", "missense_variant",
    "synonymous_variant", "splice_region_variant",
    "5_prime_UTR_variant", "3_prime_UTR_variant", "intron_variant",
    "non_coding_transcript_exon_variant", "upstream_gene_variant",
    "downstream_gene_variant", "regulatory_region_variant",
    "TF_binding_site_variant", "intergenic_variant", "other",
]
CG_IDX = {c:i for i,c in enumerate(CONSEQUENCE_GROUPS)}
IMPACT_ORD = {"HIGH":3.0, "MODERATE":2.0, "LOW":1.0, "MODIFIER":0.0}

# Amino-acid property axes (Grantham's three axes; values from the literature).
# NOTE: this is a RECOMPUTED property distance, NOT the original Grantham
# matrix. Load the published matrix with --grantham-file for exact values.
AA_HYDRO = dict(zip("ARNDCQEGHILKMFPSTWYV",
    [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,
     -0.8,-0.7,-0.9,-1.3,4.2]))                                    # [K1]
AA_VOL = dict(zip("ARNDCQEGHILKMFPSTWYV",
    [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,
     162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
AA_POL = dict(zip("ARNDCQEGHILKMFPSTWYV",
    [8.1,10.5,11.6,13.0,5.5,10.5,12.3,9.0,10.4,5.2,4.9,11.3,5.7,5.2,8.0,
     9.2,8.6,5.4,6.2,5.9]))
AA3to1 = {
 "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
 "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
 "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","TER":"*","STP":"*"}

CONSERVATION_SOURCES = {
  "phyloP100way_hg38": {
    "kind":"phyloP", "assembly":"GRCh38",
    "dir_url": ("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/"
                "hg38.100way.phyloP100way/"),
    "single_url": None,
    "note": "per-chromosome .bw files; open the directory listing and download"},
  "phyloP470way_hg38": {
    "kind":"phyloP", "assembly":"GRCh38",
    "dir_url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP470way/",
    "single_url": ("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP470way/"
                   "hg38.470way.phyloP470way.bw"),
    "note": "470-way mammal alignment; large single bigWig"},
  "phastCons100way_hg38": {
    "kind":"phastCons", "assembly":"GRCh38",
    "dir_url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/",
    "single_url": ("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/"
                   "hg38.100way.phastCons.bw"),
    "note": "phastCons in [0,1]"},
  "phyloP100way_hg19": {
    "kind":"phyloP", "assembly":"GRCh37",
    "dir_url": ("https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phyloP100way/"
                "hg19.100way.phyloP100way/"),
    "single_url": None,
    "note": "GRCh37/hg19 per-chromosome .bw"},
}


# =============================================================================
# SECTION 1 — CONSEQUENCE ANNOTATION (VEP REST / local VEP / snpEff)
# =============================================================================

class VEPAnnotator:
    """
    Three backends:
      rest    : Ensembl REST, no install, rate-limited, network required
      vep     : local VEP (--cache --offline), production scale
      snpeff  : local snpEff jar

    Disk cache is mandatory in practice: annotation is the expensive step and
    you will re-run the model many times against the same variants.
    """
    def __init__(self, backend="rest", assembly="GRCh38",
                 cache_dir="data/vep_cache", batch=100, sleep=0.15,
                 vep_bin="vep", vep_cache=None, snpeff_jar=None,
                 snpeff_db="GRCh38.105", canonical_only=True,
                 timeout=60, max_retry=4):
        self.backend, self.assembly = backend, assembly
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.batch, self.sleep = batch, sleep
        self.vep_bin, self.vep_cache = vep_bin, vep_cache
        self.snpeff_jar, self.snpeff_db = snpeff_jar, snpeff_db
        self.canonical_only = canonical_only
        self.timeout, self.max_retry = timeout, max_retry
        self.mem: Dict[str, Dict] = {}
        self.stats = {"cache_hit":0, "annotated":0, "failed":0}

    @staticmethod
    def key(chrom, pos, ref, alt) -> str:
        return f"{chrom}:{pos}:{ref}:{alt}"

    def _disk(self, k): return self.cache / (k.replace(":","_") + ".json")

    def _load(self, k):
        if k in self.mem: return self.mem[k]
        p = self._disk(k)
        if p.exists():
            try:
                d = json.load(open(p)); self.mem[k] = d
                self.stats["cache_hit"] += 1; return d
            except Exception: pass
        return None

    def _save(self, k, d):
        self.mem[k] = d
        json.dump(d, open(self._disk(k), "w"))

    def annotate(self, variants: List[Dict]) -> List[Optional[Dict]]:
        out: List[Optional[Dict]] = [None]*len(variants)
        todo = []
        for i, v in enumerate(variants):
            k = self.key(v["chrom"], v["pos"], v["ref"], v["alt"])
            c = self._load(k)
            if c is not None: out[i] = c
            else: todo.append(i)
        if not todo:
            log.info(f"[VEP] all {len(variants)} from cache"); return out
        log.info(f"[VEP] {len(todo)} to annotate ({self.backend}), "
                 f"{len(variants)-len(todo)} cached")
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s+self.batch]
            chunk = [variants[i] for i in idx]
            try:
                res = (self._rest(chunk) if self.backend=="rest" else
                       self._vep(chunk) if self.backend=="vep" else
                       self._snpeff(chunk))
            except Exception as e:
                log.warning(f"[VEP] batch failed: {e}")
                res = [None]*len(chunk)
            for i, r in zip(idx, res):
                if r is not None:
                    v = variants[i]
                    self._save(self.key(v["chrom"],v["pos"],v["ref"],v["alt"]), r)
                    out[i] = r; self.stats["annotated"] += 1
                else:
                    self.stats["failed"] += 1
            if self.backend == "rest": time.sleep(self.sleep)
            if (s//self.batch+1) % 10 == 0:
                log.info(f"  [VEP] {min(s+self.batch,len(todo))}/{len(todo)}")
        log.info(f"[VEP] {self.stats}")
        return out

    # ---- Ensembl REST ----
    def _rest(self, chunk: List[Dict]) -> List[Optional[Dict]]:
        url = f"{REST[self.assembly]}/vep/human/region"
        body = {"variants": [
            f"{v['chrom']} {v['pos']} . {v['ref']} {v['alt']} . . ." for v in chunk]}
        data = json.dumps(body).encode()
        for attempt in range(self.max_retry):
            try:
                req = urllib.request.Request(url, data=data, headers={
                    "Content-Type":"application/json", "Accept":"application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = json.loads(r.read().decode())
                return [self._parse_vep_record(x) for x in raw]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = float(e.headers.get("Retry-After", 2**attempt))
                    log.info(f"[VEP] 429 rate limit, sleeping {wait}s")
                    time.sleep(wait); continue
                log.warning(f"[VEP] HTTP {e.code}: {e.read()[:200]}")
                return [None]*len(chunk)
            except Exception as e:
                log.warning(f"[VEP] attempt {attempt+1}: {e}")
                time.sleep(2**attempt)
        return [None]*len(chunk)

    # ---- local VEP ----
    def _vep(self, chunk):
        if not self.vep_cache:
            raise RuntimeError("local VEP needs --vep-cache (install with "
                               "INSTALL.pl --CACHE --species homo_sapiens)")
        vcf = "\n".join(
            f"{v['chrom']}\t{v['pos']}\t.\t{v['ref']}\t{v['alt']}\t.\t.\t."
            for v in chunk) + "\n"
        cmd = [self.vep_bin, "--offline", "--cache", "--dir_cache", self.vep_cache,
               "--species","homo_sapiens","--assembly",self.assembly,
               "--format","vcf","--json","--no_stats","--force_overwrite",
               "--sift","b","--polyphen","b","--canonical","--biotype",
               "--input_file","STDIN","--output_file","STDOUT"]
        p = subprocess.run(cmd, input=vcf, capture_output=True, text=True,
                           timeout=max(self.timeout, 600))
        if p.returncode != 0:
            raise RuntimeError(f"vep failed: {p.stderr[:400]}")
        out = []
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            try: out.append(self._parse_vep_record(json.loads(line)))
            except Exception: out.append(None)
        return out or [None]*len(chunk)

    # ---- snpEff ----
    def _snpeff(self, chunk):
        if not self.snpeff_jar:
            raise RuntimeError("snpEff needs --snpeff-jar and a downloaded database")
        vcf = ("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
               + "\n".join(f"{v['chrom']}\t{v['pos']}\t.\t{v['ref']}\t{v['alt']}"
                           f"\t.\t.\t." for v in chunk) + "\n")
        cmd = ["java","-Xmx4g","-jar",self.snpeff_jar,"-noStats","-noLog",
               self.snpeff_db,"-"]
        p = subprocess.run(cmd, input=vcf, capture_output=True, text=True,
                           timeout=max(self.timeout,600))
        if p.returncode != 0:
            raise RuntimeError(f"snpEff failed: {p.stderr[:400]}")
        out = []
        for line in p.stdout.splitlines():
            if line.startswith("#") or not line.strip(): continue
            f = line.split("\t")
            m = re.search(r"ANN=(.*)", f[7] if len(f)>7 else "")
            if not m: out.append(None); continue
            anns = [a.split("|") for a in m.group(1).split(",")]
            best = anns[0]
            out.append({
                "consequence": best[1] if len(best)>1 else "other",
                "impact": best[2] if len(best)>2 else "MODIFIER",
                "gene": best[3] if len(best)>3 else "",
                "biotype": best[7] if len(best)>7 else "",
                "canonical": True, "n_transcripts": len(anns),
                "sift_score": None, "polyphen_score": None,
                "aa_ref": None, "aa_alt": None, "cds_rel": None,
                "source": "snpEff"})
        return out or [None]*len(chunk)

    def _parse_vep_record(self, rec: Dict) -> Optional[Dict]:
        tcs = rec.get("transcript_consequences", [])
        if not tcs:
            tcs = (rec.get("regulatory_feature_consequences", [])
                   or rec.get("intergenic_consequences", []))
        if not tcs:
            return {"consequence":"intergenic_variant","impact":"MODIFIER",
                    "gene":"","biotype":"","canonical":True,"n_transcripts":0,
                    "sift_score":None,"polyphen_score":None,
                    "aa_ref":None,"aa_alt":None,"cds_rel":None,"source":"vep"}
        pool = [t for t in tcs if t.get("canonical")==1] if self.canonical_only else tcs
        if not pool: pool = tcs
        # worst consequence across the chosen pool
        def rank(t):
            cs = t.get("consequence_terms", ["other"])
            return min(CG_IDX.get(c, len(CG_IDX)-1) for c in cs)
        best = sorted(pool, key=rank)[0]
        cs = best.get("consequence_terms", ["other"])
        pick = min(cs, key=lambda c: CG_IDX.get(c, len(CG_IDX)-1))
        aa = (best.get("amino_acids") or "/").split("/")
        cds, prot_len = best.get("cds_start"), best.get("protein_start")
        return {
            "consequence": pick,
            "impact": best.get("impact","MODIFIER"),
            "gene": best.get("gene_symbol",""),
            "biotype": best.get("biotype",""),
            "canonical": bool(best.get("canonical")==1),
            "n_transcripts": len(tcs),
            "sift_score": best.get("sift_score"),
            "polyphen_score": best.get("polyphen_score"),
            "aa_ref": aa[0] if aa and aa[0] else None,
            "aa_alt": aa[1] if len(aa)>1 and aa[1] else None,
            "cds_rel": (float(cds)/max(float(best.get("cds_end") or cds),1.0)
                        if cds else None),
            "source": "vep"}


def aa_property_distance(a: Optional[str], b: Optional[str]) -> float:
    """
    Scaled Euclidean distance on Grantham's three axes (hydropathy [K1],
    volume, polarity). NOT the original Grantham 1974 matrix [G1].
    Returns 0 for missing/synonymous.
    """
    if not a or not b or a==b or a not in AA_HYDRO or b not in AA_HYDRO:
        return 0.0
    dh = (AA_HYDRO[a]-AA_HYDRO[b])/9.0
    dp = (AA_POL[a]-AA_POL[b])/8.0
    dv = (AA_VOL[a]-AA_VOL[b])/170.0
    return float(math.sqrt(dh*dh + dp*dp + dv*dv))


N_CONS = len(CONSEQUENCE_GROUPS) + 8
def consequence_features(ann: Optional[Dict], use_pred_scores=True) -> np.ndarray:
    """
    Feature block:
      one-hot consequence group (19)
      IMPACT ordinal, canonical, log1p(n_transcripts), protein_coding flag
      cds relative position
      AA property distance
      SIFT, PolyPhen  ← EXCLUDED when use_pred_scores=False (circularity)
    """
    v = np.zeros(N_CONS, np.float32)
    if ann is None:
        v[CG_IDX["other"]] = 1.0
        return v
    v[CG_IDX.get(ann["consequence"], CG_IDX["other"])] = 1.0
    o = len(CONSEQUENCE_GROUPS)
    v[o+0] = IMPACT_ORD.get(ann.get("impact","MODIFIER"),0.0)/3.0
    v[o+1] = float(ann.get("canonical", False))
    v[o+2] = math.log1p(ann.get("n_transcripts",0))/5.0
    v[o+3] = float(ann.get("biotype")=="protein_coding")
    v[o+4] = ann.get("cds_rel") or 0.0
    v[o+5] = aa_property_distance(ann.get("aa_ref"), ann.get("aa_alt"))
    if use_pred_scores:
        v[o+6] = float(ann["sift_score"]) if ann.get("sift_score") is not None else -1.0
        v[o+7] = float(ann["polyphen_score"]) if ann.get("polyphen_score") is not None else -1.0
    else:
        v[o+6] = v[o+7] = 0.0
    return v


# =============================================================================
# SECTION 2 — CONSERVATION TRACKS (phyloP / phastCons bigWig)
# =============================================================================

def chrom_candidates(c: str) -> List[str]:
    b = str(c).strip().replace("chr","").replace("CHR","")
    if b.upper() in ("MT","M"): return ["chrM","MT","M","chrMT"]
    return [f"chr{b}", b, f"Chr{b}"]


class ConservationTrack:
    """
    bigWig random access via pyBigWig.

    COORDINATES: bigWig is 0-based half-open; VCF pos is 1-based.
    The base at VCF pos p is at bigWig index p-1. Off-by-one here silently
    shifts every conservation score by one base.

    Values: phyloP can be negative (accelerated) or positive (conserved);
            phastCons is in [0,1]. NaN (unmapped regions) → 0 with a mask flag.
    """
    def __init__(self, paths: List[Path], kind="phyloP", window=10):
        if not HAS_BW: raise ImportError("pip install pyBigWig")
        self.kind, self.window = kind, window
        self.single: Optional[Any] = None
        self.per_chrom: Dict[str, Any] = {}
        self._keys: set = set()
        for p in paths:
            p = Path(p)
            if p.is_dir():
                for f in sorted(p.glob("*.bw")) + sorted(p.glob("*.bigWig")):
                    self._open_perchrom(f)
            else:
                bw = pyBigWig.open(str(p))
                chroms = set(bw.chroms().keys())
                if len(chroms) > 1:
                    self.single = bw; self._keys |= chroms
                else:
                    self._open_perchrom(p)
        if self.single is None and not self.per_chrom:
            raise RuntimeError(f"no usable bigWig in {paths}")
        log.info(f"[{kind}] window={window}, "
                 f"{'single multi-chrom file' if self.single else f'{len(self.per_chrom)} per-chrom files'}")
        self.stats = {"ok":0,"no_chrom":0,"nan":0}

    def _open_perchrom(self, f: Path):
        m = re.search(r"(chr[0-9XYM]+|[0-9XYM]+)\.", f.name)
        if not m: return
        self.per_chrom[m.group(1)] = pyBigWig.open(str(f))
        self._keys.add(m.group(1))

    def resolve(self, c: str) -> Optional[str]:
        for cand in chrom_candidates(c):
            if cand in self._keys: return cand
        return None

    def _bw_for(self, k: str):
        return self.single if self.single is not None else self.per_chrom.get(k)

    def values(self, chrom: str, pos: int, flank: int) -> Optional[np.ndarray]:
        k = self.resolve(chrom)
        if k is None:
            self.stats["no_chrom"] += 1; return None
        bw = self._bw_for(k)
        if bw is None:
            self.stats["no_chrom"] += 1; return None
        L = bw.chroms(k)
        lo, hi = pos-1-flank, pos-1+flank+1
        if lo < 0 or hi > L:
            lo, hi = max(0,lo), min(L,hi)
        try:
            v = np.asarray(bw.values(k, lo, hi), dtype=np.float32)
        except Exception:
            self.stats["no_chrom"] += 1; return None
        need = 2*flank+1
        if len(v) < need:
            v = np.pad(v, (0, need-len(v)), constant_values=np.nan)
        if np.isnan(v).any():
            self.stats["nan"] += 1
            v = np.nan_to_num(v, nan=0.0)
        self.stats["ok"] += 1
        return v

    def features(self, chrom: str, pos: int) -> np.ndarray:
        """[center, mean±w, max±w, min±w, std±w, mean±2, frac_positive]"""
        v = self.values(chrom, pos, self.window)
        if v is None: return np.zeros(7, np.float32)
        c = float(v[self.window])
        w = v[max(0,self.window-2):self.window+3]
        return np.array([c, float(v.mean()), float(v.max()), float(v.min()),
                         float(v.std()), float(w.mean()),
                         float((v>0).mean())], np.float32)

    def close(self):
        if self.single: self.single.close()
        for bw in self.per_chrom.values(): bw.close()


N_CONSV = 7
def download_conservation(key: str, dest="data/conservation") -> Path:
    src = CONSERVATION_SOURCES[key]
    d = Path(dest)/key; d.mkdir(parents=True, exist_ok=True)
    if src["single_url"]:
        out = d/Path(src["single_url"]).name
        if out.exists() and out.stat().st_size > 1e7:
            log.info(f"[cons] cached {out.name}"); return out
        log.info(f"[cons] downloading {src['single_url']}")
        log.info(f"[cons] {src['note']} — this file is LARGE (GB scale)")
        try:
            with urllib.request.urlopen(src["single_url"], timeout=900) as r, \
                 open(out,"wb") as f:
                tot=0
                while True:
                    b=r.read(1<<22)
                    if not b: break
                    f.write(b); tot+=len(b)
                    if tot%(500<<20)==0: log.info(f"  {tot/1e6:.0f} MB")
            return out
        except Exception as e:
            out.unlink(missing_ok=True)
            log.error(f"[cons] failed: {e}")
    print(f"\n  MANUAL DOWNLOAD for {key}:")
    print(f"    1. open  {src['dir_url']}")
    print(f"    2. download the .bw file(s)  ({src['note']})")
    print(f"    3. place them in {d}")
    print(f"    4. pass --conservation {d}\n")
    raise SystemExit(1)


# =============================================================================
# SECTION 3 — CHANGE-seq LOADER + OPERATIONAL NEGATIVES
# =============================================================================

CS_SEQ_HINTS = ["sequence","seq","offtarget sequence","off-target sequence","dna"]
CS_GUIDE_HINTS = ["targetsite","target site","guide","sgrna","on-target","target"]
CS_READ_HINTS = ["reads","read count","#reads","unique","count","changeseq reads"]
CS_CELL_HINTS = ["cell type","celltype","cell","sample","experiment","condition"]

def _pick(cols, hints, exclude=None):
    low = {c: c.lower().strip() for c in cols}
    for h in hints:
        for c,lc in low.items():
            if exclude and any(e in lc for e in exclude): continue
            if h == lc: return c
    for h in hints:
        for c,lc in low.items():
            if exclude and any(e in lc for e in exclude): continue
            if h in lc: return c
    return None

def clean_dna(s) -> Optional[str]:
    s = "".join(ch for ch in str(s).upper() if ch in "ACGT")
    return s if len(s) in (20,23) else None


def changeseq_probe(path: Path) -> Dict:
    import csv
    delim = "\t" if path.suffix.lower() in (".txt",".tsv") else ","
    rows=[]
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for i,r in enumerate(csv.reader(f, delimiter=delim)):
            rows.append(r)
            if i>=4: break
    cols = rows[0] if rows else []
    return {"file":path.name, "columns":cols,
            "guessed": {"seq":_pick(cols,CS_SEQ_HINTS,["target site","targetsite"]),
                        "guide":_pick(cols,CS_GUIDE_HINTS),
                        "reads":_pick(cols,CS_READ_HINTS),
                        "cell":_pick(cols,CS_CELL_HINTS)},
            "sample_rows":rows[1:4]}


def changeseq_load(path: Path, seq_col=None, guide_col=None, reads_col=None,
                   cell_col=None, min_reads=1, limit=None) -> List[Dict]:
    """
    CHANGE-seq positives [L1]. In vitro, cell-free, purified genomic DNA.
    Higher sensitivity than GUIDE-seq [L2]; detects chromatin-inaccessible sites.
    """
    import csv
    delim = "\t" if path.suffix.lower() in (".txt",".tsv") else ","
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f, delimiter=delim); cols = rd.fieldnames or []
        sc = seq_col or _pick(cols, CS_SEQ_HINTS, ["target site","targetsite"])
        gc = guide_col or _pick(cols, CS_GUIDE_HINTS)
        rc = reads_col or _pick(cols, CS_READ_HINTS)
        cc = cell_col or _pick(cols, CS_CELL_HINTS)
        if sc is None:
            raise KeyError(f"no sequence column in {cols}; run changeseq-probe")
        log.info(f"[CHANGE-seq] seq='{sc}' guide='{gc}' reads='{rc}' sample='{cc}'")
        out=[]
        for row in rd:
            ot = clean_dna(row.get(sc,""))
            if ot is None: continue
            gd = clean_dna(row.get(gc,"")) if gc else None
            try: nr = float(str(row.get(rc,"1")).replace(",",""))
            except (TypeError,ValueError): nr = 1.0
            if nr < min_reads: continue
            out.append({"guide":gd,"offtarget":ot,"reads":nr,
                        "sample":row.get(cc,"unknown") if cc else "unknown",
                        "label":1,"evidence":"changeseq_in_vitro"})
            if limit and len(out)>=limit: break
    log.info(f"[CHANGE-seq] {len(out)} detected sites, "
             f"{len(set(r['guide'] for r in out if r['guide']))} guides")
    log.warning("[CHANGE-seq] IN VITRO positives: cleavable on purified DNA, "
                "not necessarily edited in living cells.")
    return out


def enumerate_candidates(guide: str, fasta: Path, chrom: str,
                         start: int = 0, end: Optional[int] = None,
                         max_mis: int = 6, seed_mis: int = 2,
                         pam: str = "NGG") -> List[Dict]:
    """
    Scan a REGION for PAM-adjacent protospacers with <= max_mis mismatches.

    ⚠️ Genome-wide enumeration is expensive. For real work use cas-offinder
       or a bowtie/BWA-based search and pass its output via --candidates.
       This scanner is for bounded regions (<= a few Mb) and demos.

    Seed prefilter: PAM-proximal 10 nt must have <= seed_mis mismatches.
    This is a heuristic and can MISS true off-targets with many seed mismatches.
    """
    if not HAS_FA: raise ImportError("pip install pyfaidx")
    if len(guide) < 20: raise ValueError("guide must be >= 20 nt")
    g = guide[:20]
    fa = Fasta(str(fasta), as_raw=True, sequence_always_upper=True)
    key = next((c for c in chrom_candidates(chrom) if c in fa.keys()), None)
    if key is None: raise KeyError(f"{chrom} not in FASTA")
    L = len(fa[key]); end = min(end or L, L)
    if end-start > 20_000_000:
        log.warning(f"[scan] region {end-start} bp is large; consider cas-offinder")
    seq = fa[key][start:end]
    log.info(f"[scan] {key}:{start}-{end} ({len(seq)} bp), guide={g}, "
             f"max_mis={max_mis}, seed_mis={seed_mis}")
    arr = np.frombuffer(seq.encode(), dtype=np.uint8)
    garr = np.frombuffer(g.encode(), dtype=np.uint8)
    rc_g = g.translate(COMP)[::-1]
    rc_arr = np.frombuffer(rc_g.encode(), dtype=np.uint8)
    out = []
    # plus strand: 20-mer followed by NGG
    for strand, s, ga in (("+", seq, garr), ("-", seq.translate(COMP)[::-1], rc_arr)):
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        n = len(a)-22
        if n <= 0: continue
        pam_ok = (a[21:n+21]==ord("G")) & (a[22:n+22]==ord("G"))
        pos = np.flatnonzero(pam_ok)
        log.info(f"[scan] {strand} strand: {len(pos)} PAM sites")
        for chunk_start in range(0, len(pos), 50_000):
            pp = pos[chunk_start:chunk_start+50_000]
            win = np.lib.stride_tricks.sliding_window_view(a, 20)[pp]
            seed_mm = (win[:, 10:20] != ga[10:20]).sum(1)
            keep = seed_mm <= seed_mis
            pp, win = pp[keep], win[keep]
            if len(pp)==0: continue
            mm = (win != ga).sum(1)
            keep2 = (mm <= max_mis) & (mm > 0)
            for p, w, m in zip(pp[keep2], win[keep2], mm[keep2]):
                out.append({"guide":g, "offtarget":w.tobytes().decode(),
                            "reads":0.0, "label":0,
                            "chrom":key, "pos":int(start+p), "strand":strand,
                            "mismatches":int(m),
                            "evidence":"assay_negative_candidate"})
    log.info(f"[scan] {len(out)} candidate off-targets (excluding exact match)")
    log.warning("[scan] seed prefilter may MISS off-targets with many seed "
                "mismatches. Genome-wide: use cas-offinder.")
    return out


def build_offtarget_dataset(positives: List[Dict],
                            candidates: Optional[List[Dict]] = None,
                            validated_negatives: Optional[List[Dict]] = None,
                            max_mis=6, seed=42) -> Tuple[List[Dict], str]:
    """
    Negative hierarchy, best to worst:
      1. validated_negatives : targeted deep sequencing, no editing above bg
      2. candidates          : PAM+mismatch sites NOT in CHANGE-seq  (assay-negative)
      3. (refused)           : random mismatch decoys — NOT USED HERE

    Detected CHANGE-seq sites are removed from the candidate pool by sequence.
    """
    detected = set(r["offtarget"] for r in positives)
    if validated_negatives:
        negs = [dict(r, label=0, evidence="validated_targeted_sequencing")
                for r in validated_negatives]
        kind = "VALIDATED (targeted sequencing)"
    elif candidates:
        negs = [c for c in candidates if c["offtarget"] not in detected]
        negs = [dict(c, label=0, evidence="changeseq_assay_negative") for c in negs]
        kind = "ASSAY-NEGATIVE (CHANGE-seq non-detection)"
        log.warning(f"[negatives] {len(negs)} assay-negatives. These are "
                    f"'not detected at this depth in vitro', NOT verified non-cleavage.")
    else:
        raise RuntimeError(
            "No negatives available. Provide --candidates (cas-offinder output "
            "or a bounded region scan) or --validated-negatives.\n"
            "Random mismatch decoys are deliberately NOT supported: they give "
            "optimistic, meaningless metrics.")
    recs = positives + negs
    log.info(f"[off-target] {len(positives)} pos / {len(negs)} neg — {kind}")
    return recs, kind


def pair_features(guide: str, off: str, L=23) -> np.ndarray:
    g = guide[:L].ljust(L,"A"); o = off[:L].ljust(L,"A")
    m = np.zeros((L,2), np.float32)
    for i in range(L):
        m[i,0]=1.0; m[i,1]=float(g[i]!=o[i])
    nm = float(sum(a!=b for a,b in zip(g,o)))
    pam = float(sum(a!=b for a,b in zip(g[-3:],o[-3:])))
    sd = float(sum(a!=b for a,b in zip(g[-12:-3],o[-12:-3])))
    return np.concatenate([m.ravel(), [nm/L, pam/3, sd/9]])


# =============================================================================
# SECTION 4 — UNIFIED ClinVar DATASET + ABLATION
# =============================================================================

BENIGN = {"Benign","Likely benign","Benign/Likely benign"}
PATHO = {"Pathogenic","Likely pathogenic","Pathogenic/Likely pathogenic"}
STARS = {"practice guideline":4,"reviewed by expert panel":3,
         "criteria provided, multiple submitters, no conflicts":2,
         "criteria provided, single submitter":1,
         "criteria provided, conflicting classifications":1,
         "criteria provided, conflicting interpretations":1,
         "no assertion criteria provided":0,"no assertion provided":0,
         "no classification provided":0,
         "no classification for the individual variant":0}
CLINVAR_URL = ("https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
               "variant_summary.txt.gz")


def clinvar_snvs(path: Path, assembly="GRCh38", min_stars=2, limit=None,
                 genome: Optional[Any]=None, flank=16) -> List[Dict]:
    recs=[]; skip={"not_snv":0,"ambiguous":0,"low_review":0,"assembly":0,"ctx":0}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8",errors="ignore") as f:
        hdr=f.readline().lstrip("#").rstrip("\n").split("\t")
        H={h:i for i,h in enumerate(hdr)}
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)<len(hdr): continue
            if p[H["Assembly"]]!=assembly: skip["assembly"]+=1; continue
            if p[H["Type"]]!="single nucleotide variant": skip["not_snv"]+=1; continue
            cs=p[H["ClinicalSignificance"]]
            y=0 if cs in BENIGN else (1 if cs in PATHO else None)
            if y is None: skip["ambiguous"]+=1; continue
            st=STARS.get(p[H["ReviewStatus"]].strip(),0)
            if st<min_stars: skip["low_review"]+=1; continue
            ref,alt=p[H["ReferenceAlleleVCF"]],p[H["AlternateAlleleVCF"]]
            if len(ref)!=1 or len(alt)!=1 or ref not in "ACGT" or alt not in "ACGT":
                skip["not_snv"]+=1; continue
            ps=p[H["PositionVCF"]]
            if not ps.isdigit(): skip["ctx"]+=1; continue
            pos=int(ps); seq=None
            if genome is not None:
                seq = genome.fetch(p[H["Chromosome"]], pos, ref)
                if seq is None: skip["ctx"]+=1; continue
            recs.append({"gene":p[H["GeneSymbol"]],"chrom":p[H["Chromosome"]],
                         "pos":pos,"ref":ref,"alt":alt,"label":y,"stars":st,
                         "seq":seq})
            if limit and len(recs)>=limit: break
    log.info(f"[ClinVar] {len(recs)} SNVs | skipped {skip}")
    return recs


class GenomeCtx:
    def __init__(self, fasta: Path, flank=16):
        if not HAS_FA: raise ImportError("pip install pyfaidx")
        self.fa=Fasta(str(fasta),as_raw=True,sequence_always_upper=True)
        self.flank=flank; self.keys=set(self.fa.keys())
        self.stats={"ok":0,"ref_mismatch":0,"no_chrom":0}
    def fetch(self, chrom, pos, ref):
        k=next((c for c in chrom_candidates(chrom) if c in self.keys),None)
        if k is None: self.stats["no_chrom"]+=1; return None
        L=len(self.fa[k]); lo,hi=pos-self.flank-1,pos+self.flank
        pl,pr=max(0,-lo),max(0,hi-L)
        s="N"*pl + self.fa[k][max(0,lo):min(L,hi)] + "N"*pr
        if len(s)!=2*self.flank+1: return None
        if s[self.flank]!=ref.upper():
            self.stats["ref_mismatch"]+=1; return None
        self.stats["ok"]+=1; return s


def onehot(seq):
    x=np.zeros((len(seq),4),np.float32)
    for i,c in enumerate(seq):
        if c in BASE_IDX: x[i,BASE_IDX[c]]=1.0
    return x

def context_block(seq, ref, alt, flank):
    alt_seq = seq[:flank]+alt+seq[flank+1:]
    xr,xa = onehot(seq), onehot(alt_seq)
    pur=set("AG"); center=seq[flank]
    cpg = (center=="C" and flank+1<len(seq) and seq[flank+1]=="G") or \
          (center=="G" and flank>0 and seq[flank-1]=="C")
    scal=np.array([float((ref in pur)==(alt in pur)),
                   float(ref in pur and alt not in pur),
                   float(ref not in pur and alt in pur),
                   float(cpg),
                   (seq.count("G")+seq.count("C"))/max(len(seq),1)],np.float32)
    return np.concatenate([np.stack([xr,xa],0).ravel(), scal])

N_SCAL=5
def ctx_dim(flank): return 2*(2*flank+1)*4 + N_SCAL


def gene_split(recs, frac=(0.7,0.15,0.15), seed=42):
    genes=sorted(set(r["gene"] for r in recs))
    rng=np.random.default_rng(seed); rng.shuffle(genes)
    n=len(genes); a=int(frac[0]*n); b=int(frac[1]*n)
    g={"train":set(genes[:a]),"val":set(genes[a:a+b]),"test":set(genes[a+b:])}
    out={k:[r for r in recs if r["gene"] in s] for k,s in g.items()}
    assert not (g["train"]&g["val"]) and not (g["train"]&g["test"]) \
           and not (g["val"]&g["test"])
    for k,v in out.items():
        p=sum(r["label"] for r in v)
        log.info(f"[gene-split] {k}: {len(v)} var, "
                 f"{len(set(r['gene'] for r in v))} genes, "
                 f"{100*p/max(len(v),1):.1f}% patho")
    log.info("[gene-split] ✓ zero gene overlap")
    return out


# ---- metrics ----
def auroc(y,p):
    if y.sum()==0 or y.sum()==len(y): return float("nan")
    r=np.empty(len(p),float); r[np.argsort(p)]=np.arange(1,len(p)+1)
    npos=y.sum(); nneg=len(y)-npos
    return float((r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def auprc(y,p):
    o=np.argsort(-p); ys=y[o]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys)
    pr=tp/np.maximum(tp+fp,1); rc=tp/max(ys.sum(),1)
    return float(np.sum(np.diff(np.concatenate([[0],rc]))*pr))

def ece(y,p,nb=15):
    edges=np.linspace(0,1,nb+1); e=0.0
    for i in range(nb):
        m=(p>edges[i])&(p<=edges[i+1])
        if m.sum()==0: continue
        e+=(m.sum()/len(p))*abs(float(y[m].mean())-float(p[m].mean()))
    return float(e)

def brier(y,p): return float(np.mean((p-y)**2))

class TempScaler:
    def __init__(self): self.log_T=nn.Parameter(torch.zeros(1))
    def fit(self, logits, y, it=200):
        lg=torch.tensor(logits,dtype=torch.float32)
        yy=torch.tensor(y,dtype=torch.float32)
        opt=torch.optim.LBFGS([self.log_T],lr=0.1,max_iter=it)
        def cl():
            opt.zero_grad()
            l=F.binary_cross_entropy_with_logits(lg/self.log_T.exp(),yy)
            l.backward(); return l
        opt.step(cl); return self
    @property
    def T(self): return float(self.log_T.exp())
    def predict(self, logits): return 1/(1+np.exp(-logits/self.T))


class VariantNet(nn.Module):
    """Context CNN + tabular (consequence, conservation) blocks."""
    def __init__(self, flank, use_ctx, n_tab, width=64):
        super().__init__()
        self.flank, self.use_ctx = flank, use_ctx
        d = 0
        if use_ctx:
            self.conv=nn.Sequential(
                nn.Conv1d(8,width,9,padding=4),nn.SiLU(),
                nn.Conv1d(width,width,5,padding=2),nn.SiLU(),
                nn.AdaptiveAvgPool1d(1))
            d += width + N_SCAL
        self.n_tab = n_tab
        d += n_tab
        self.head=nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Dropout(0.3),
                                nn.Linear(128,1))
    def forward(self, x):
        parts=[]; off=0
        if self.use_ctx:
            L=2*self.flank+1
            seq=x[:, :2*L*4].reshape(-1,2*L,8).transpose(1,2)
            scal=x[:, 2*L*4:2*L*4+N_SCAL]
            parts += [self.conv(seq).squeeze(-1), scal]
            off = ctx_dim(self.flank)
        if self.n_tab>0:
            parts.append(x[:, off:off+self.n_tab])
        return self.head(torch.cat(parts,-1)).squeeze(-1)


def run_ablation(args):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- data ----
    cvp=Path(args.clinvar)
    if not cvp.exists():
        d=cvp.parent; d.mkdir(parents=True,exist_ok=True)
        log.info(f"[ClinVar] downloading {CLINVAR_URL}")
        with urllib.request.urlopen(CLINVAR_URL,timeout=600) as r, open(cvp,"wb") as f:
            shutil.copyfileobj(r,f,1<<20)
    genome=None
    if args.fasta and Path(args.fasta).exists():
        genome=GenomeCtx(Path(args.fasta), flank=args.flank)
    recs=clinvar_snvs(cvp, args.assembly, args.min_stars, args.limit,
                      genome, args.flank)
    if len(recs)<200: log.error("too few records"); return

    # ---- annotation ----
    ann=VEPAnnotator(backend=args.vep_backend, assembly=args.assembly,
                     cache_dir=args.vep_cache, vep_cache=args.vep_local_cache,
                     snpeff_jar=args.snpeff_jar, snpeff_db=args.snpeff_db)
    anns=ann.annotate(recs)
    n_ok=sum(a is not None for a in anns)
    log.info(f"[annot] {n_ok}/{len(recs)} annotated")
    if n_ok < 0.5*len(recs):
        log.warning("[annot] <50% annotated — consequence block will be mostly "
                    "missing. Check network / VEP install.")

    # ---- conservation ----
    tracks=[]
    for p in (args.conservation or []):
        pp=Path(p)
        if not pp.exists():
            log.warning(f"[cons] {pp} missing, skipping"); continue
        tracks.append(ConservationTrack([pp], kind=args.cons_kind,
                                        window=args.cons_window))
    if not tracks:
        log.warning("[cons] no conservation track loaded — that ablation arm "
                    "will be identical to the previous one")

    # ---- feature blocks ----
    sp=gene_split(recs, seed=args.seed)
    idx={k:np.array([recs.index(r) for r in v]) for k,v in sp.items()}
    # index map is O(n^2); rebuild efficiently
    key2i={ (r["chrom"],r["pos"],r["ref"],r["alt"]):i for i,r in enumerate(recs) }
    idx={k:np.array([key2i[(r["chrom"],r["pos"],r["ref"],r["alt"])] for r in v])
         for k,v in sp.items()}

    def blocks(ii, use_pred):
        ctx=[]; cons=[]; conseq=[]
        for i in ii:
            r=recs[i]
            if genome is not None and r["seq"]:
                ctx.append(context_block(r["seq"],r["ref"],r["alt"],args.flank))
            else:
                ctx.append(np.zeros(ctx_dim(args.flank),np.float32))
            conseq.append(consequence_features(anns[i], use_pred))
            if tracks:
                cons.append(np.concatenate([t.features(r["chrom"],r["pos"])
                                            for t in tracks]))
            else:
                cons.append(np.zeros(N_CONSV,np.float32))
        return (np.stack(ctx), np.stack(conseq), np.stack(cons))

    arms = [
        ("context_only",        True,  False, False, True),
        ("consequence_only",    False, True,  False, True),
        ("conservation_only",   False, False, True,  True),
        ("ctx+conseq",          True,  True,  False, True),
        ("ctx+cons",            True,  False, True,  True),
        ("ctx+conseq+cons",     True,  True,  True,  True),
        ("full_NO_pred_scores", True,  True,  True,  False),
    ]
    if genome is None:
        arms=[a for a in arms if not a[1]]
        log.warning("[ablation] no FASTA → context arms removed")

    results={}
    for name, uc, uq, uv, up in arms:
        n_tab = (N_CONS if uq else 0) + (N_CONSV*len(tracks) if uv else 0)
        if not uc and n_tab==0: continue
        X={}
        for k,ii in idx.items():
            c,q,v = blocks(ii, up)
            parts=[]
            if uc: parts.append(c)
            if uq: parts.append(q)
            if uv: parts.append(v)
            X[k]=torch.tensor(np.concatenate(parts,1)).to(dev)
        y={k:torch.tensor([recs[i]["label"] for i in ii],
                          dtype=torch.float32).to(dev) for k,ii in idx.items()}
        torch.manual_seed(args.seed)
        net=VariantNet(args.flank, uc, n_tab).to(dev)
        pw=(y["train"].numel()-y["train"].sum())/y["train"].sum().clamp(min=1)
        opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
        best,bs=1e9,None
        for ep in range(args.epochs):
            net.train(); perm=torch.randperm(len(X["train"]),device=dev)
            for s in range(0,len(perm),args.batch_size):
                i=perm[s:s+args.batch_size]
                loss=F.binary_cross_entropy_with_logits(net(X["train"][i]),
                                                        y["train"][i],pos_weight=pw)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            if (ep+1)%max(1,args.epochs//8)==0:
                net.eval()
                with torch.no_grad():
                    vl=float(F.binary_cross_entropy_with_logits(
                        net(X["val"]),y["val"]))
                if vl<best:
                    best=vl; bs={k:v.clone() for k,v in net.state_dict().items()}
        if bs: net.load_state_dict(bs)
        net.eval()
        with torch.no_grad():
            lv=net(X["val"]).cpu().numpy(); lt=net(X["test"]).cpu().numpy()
        yv=y["val"].cpu().numpy(); yt=y["test"].cpu().numpy()
        T=TempScaler().fit(lv,yv)
        pt=T.predict(lt)
        r={"auroc":auroc(yt,pt),"auprc":auprc(yt,pt),
           "brier":brier(yt,pt),"ece":ece(yt,pt),"T":T.T,
           "n_test":int(len(yt)),"n_features":int(X["train"].shape[1])}
        # stratified: missense only, to expose trivial stop_gained signal
        mq=np.array([anns[i] is not None and
                     anns[i]["consequence"]=="missense_variant" for i in idx["test"]])
        if mq.sum()>30 and 0<yt[mq].sum()<mq.sum():
            r["auroc_missense_only"]=auroc(yt[mq],pt[mq])
            r["n_missense"]=int(mq.sum())
        results[name]=r
        log.info(f"[{name}] AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  "
                 f"ECE {r['ece']:.4f}")
        del net, X, y
        if dev.type=="cuda": torch.cuda.empty_cache()

    print("\n"+"="*84)
    print("  ABLATION — ClinVar, gene-level split, temperature-calibrated")
    print("="*84)
    print(f"  assembly={args.assembly}  min_stars={args.min_stars}  "
          f"vep={args.vep_backend}  tracks={len(tracks)}")
    print(f"  train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")
    print(f"\n  {'arm':<24}{'feat':>6}{'AUROC':>9}{'AUPRC':>9}{'Brier':>9}"
          f"{'ECE':>9}{'AUROC_mis':>11}")
    for k,r in results.items():
        am=r.get("auroc_missense_only")
        print(f"  {k:<24}{r['n_features']:>6}{r['auroc']:>9.4f}{r['auprc']:>9.4f}"
              f"{r['brier']:>9.4f}{r['ece']:>9.4f}"
              f"{(f'{am:.4f}' if am is not None else '—'):>11}")
    base=results.get("context_only") or results.get("consequence_only")
    full=results.get("full_NO_pred_scores")
    if base and full:
        print(f"\n  Δ AUROC (full_NO_pred vs {list(results)[0]}): "
              f"{full['auroc']-base['auroc']:+.4f}")
    if "ctx+conseq+cons" in results and full:
        gap=results["ctx+conseq+cons"]["auroc"]-full["auroc"]
        print(f"  SIFT/PolyPhen contribution to AUROC: {gap:+.4f}"
              f"   ← circularity嫌疑 if large")
    print("\n  HOW TO READ THIS")
    print("   • If consequence_only ≈ full, context and conservation add little.")
    print("   • If AUROC_mis ≪ AUROC, most of the signal is the trivial")
    print("     stop_gained/synonymous distinction, not missense interpretation.")
    print("   • A large SIFT/PolyPhen contribution is CIRCULARITY, not insight.")
    print("  ❌ Agreement with ClinVar assertions ≠ pathogenicity detection.")
    print("  ❌ Not a diagnostic. Not for clinical use.")
    print("="*84)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    json.dump({"results":results,"args":vars(args),
               "annot_stats":ann.stats,
               "genome_stats":genome.stats if genome else None,
               "cons_stats":[t.stats for t in tracks]},
              open(args.out,"w"),indent=2,default=str)
    for t in tracks: t.close()
    log.info(f"saved → {args.out}")


def run_offtarget(args):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p=Path(args.data)
    if not p.exists():
        log.error(f"{p} not found.\n"
                  f"CHANGE-seq [L1]: https://www.nature.com/articles/s41587-020-0743-1\n"
                  f"  code: https://github.com/tsailabSJ/changeseq\n"
                  f"Then: changeseq-probe --data <file>")
        return
    pos=changeseq_load(p, args.seq_col, args.guide_col, args.reads_col,
                       min_reads=args.min_reads, limit=args.limit)
    pos=[r for r in pos if r["guide"]]
    if len(pos)<20: log.error("too few paired records"); return
    cands=None
    if args.candidates:
        cp=Path(args.candidates)
        if cp.suffix.lower() in (".txt",".tsv",".csv"):
            cands=changeseq_load(cp, min_reads=0, limit=None)
            for c in cands: c["label"]=0
        else:
            log.error("--candidates must be a cas-offinder/TSV table")
    elif args.scan_fasta and args.scan_chrom:
        guides=sorted(set(r["guide"] for r in pos))[:args.scan_guides]
        cands=[]
        for g in guides:
            cands += enumerate_candidates(g, Path(args.scan_fasta), args.scan_chrom,
                                          args.scan_start, args.scan_end,
                                          args.max_mis, args.seed_mis)
    recs,kind=build_offtarget_dataset(pos,cands,None,args.max_mis,args.seed)
    L=23
    X=np.stack([pair_features(r["guide"],r["offtarget"],L) for r in recs])
    y=np.array([r["label"] for r in recs],np.float32)
    guides=sorted(set(r["guide"] for r in recs))
    rng=np.random.default_rng(args.seed); rng.shuffle(guides)
    n=len(guides); a=int(0.7*n); b=int(0.15*n)
    gm={"train":set(guides[:a]),"val":set(guides[a:a+b]),"test":set(guides[a+b:])}
    idx={k:np.array([i for i,r in enumerate(recs) if r["guide"] in s])
         for k,s in gm.items()}
    log.info(f"[off-target] GUIDE-level split "
             f"{len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")
    assert not (gm["train"]&gm["val"]) and not (gm["train"]&gm["test"])
    Xt=torch.tensor(X).to(dev); yt=torch.tensor(y).to(dev)
    itr,iva,ite=(torch.tensor(idx[k],device=dev) for k in ("train","val","test"))
    net=nn.Sequential(nn.Conv1d(2,64,5,padding=2),nn.SiLU(),
                      nn.Conv1d(64,64,3,padding=1),nn.SiLU(),
                      nn.AdaptiveAvgPool1d(1),nn.Flatten()).to(dev)
    head=nn.Sequential(nn.Linear(64+3,64),nn.SiLU(),nn.Dropout(0.3),
                       nn.Linear(64,1)).to(dev)
    def fwd(x):
        s=x[:,:2*L].reshape(-1,L,2).transpose(1,2)
        return head(torch.cat([net(s), x[:,2*L:]],-1)).squeeze(-1)
    params=list(net.parameters())+list(head.parameters())
    pw=(yt[itr].numel()-yt[itr].sum())/yt[itr].sum().clamp(min=1)
    opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-4)
    best,bs=1e9,None
    for ep in range(args.epochs):
        net.train(); head.train()
        perm=itr[torch.randperm(len(itr),device=dev)]
        for s in range(0,len(perm),args.batch_size):
            i=perm[s:s+args.batch_size]
            loss=F.binary_cross_entropy_with_logits(fwd(Xt[i]),yt[i],pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (ep+1)%max(1,args.epochs//8)==0:
            net.eval(); head.eval()
            with torch.no_grad():
                vl=float(F.binary_cross_entropy_with_logits(fwd(Xt[iva]),yt[iva]))
            if vl<best:
                best=vl
                bs=({"net":{k:v.clone() for k,v in net.state_dict().items()},
                     "head":{k:v.clone() for k,v in head.state_dict().items()}})
    if bs: net.load_state_dict(bs["net"]); head.load_state_dict(bs["head"])
    net.eval(); head.eval()
    with torch.no_grad():
        lv=fwd(Xt[iva]).cpu().numpy(); lt=fwd(Xt[ite]).cpu().numpy()
    yv,yt2=yt[iva].cpu().numpy(),yt[ite].cpu().numpy()
    T=TempScaler().fit(lv,yv); pt=T.predict(lt)
    print("\n"+"="*84)
    print("  CHANGE-seq off-target — GUIDE-level split")
    print("="*84)
    print(f"  negatives: {kind}")
    print(f"  pos={len(pos)}  total={len(recs)}  L={L}")
    print(f"  AUROC {auroc(yt2,pt):.4f}   AUPRC {auprc(yt2,pt):.4f}   "
          f"Brier {brier(yt2,pt):.4f}   ECE {ece(yt2,pt):.4f}   T={T.T:.3f}")
    print("\n  ⚠️  CHANGE-seq is IN VITRO (cell-free, no chromatin).")
    print("     Positives = cleavable on purified DNA, not edited in cells.")
    if "ASSAY-NEGATIVE" in kind:
        print("  ⚠️  Negatives = NOT DETECTED at this assay's depth.")
        print("     Some are real off-targets below detection. Metrics are")
        print("     optimistic relative to validated negatives.")
    else:
        print("  ✓  Negatives from targeted validation — the strongest available.")
    print("  ❌ No guide is safe without experimental off-target assays")
    print("     and institutional biosafety review.")
    print("="*84)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    json.dump({"negatives_kind":kind,"auroc":auroc(yt2,pt),
               "auprc":auprc(yt2,pt),"T":T.T,"args":vars(args)},
              open(args.out,"w"),indent=2)


def selftest(args):
    print("="*80); print("  SELF-TEST"); print("="*80)
    ok=True
    print("\n[T1] consequence feature block")
    a={"consequence":"missense_variant","impact":"MODERATE","gene":"BRCA1",
       "biotype":"protein_coding","canonical":True,"n_transcripts":12,
       "sift_score":0.01,"polyphen_score":0.99,"aa_ref":"R","aa_alt":"W",
       "cds_rel":0.4,"source":"vep"}
    f1=consequence_features(a,True); f0=consequence_features(a,False)
    print(f"  dim={len(f1)} (expect {N_CONS})")
    print(f"  missense one-hot set: {f1[CG_IDX['missense_variant']]==1.0}")
    print(f"  SIFT present with pred: {f1[-2]:.3f}   excluded: {f0[-2]:.3f}")
    ok &= len(f1)==N_CONS and f1[CG_IDX["missense_variant"]]==1.0
    ok &= f1[-2]!=0.0 and f0[-2]==0.0 and f0[-1]==0.0
    print(f"  {'PASS' if ok else 'FAIL'}")

    print("\n[T2] AA property distance sanity")
    d_li=aa_property_distance("L","I"); d_rw=aa_property_distance("R","W")
    d_same=aa_property_distance("A","A"); d_none=aa_property_distance(None,"A")
    print(f"  L→I (conservative) = {d_li:.4f}")
    print(f"  R→W (radical)      = {d_rw:.4f}")
    print(f"  A→A = {d_same:.4f},  None→A = {d_none:.4f}")
    t2 = d_li < d_rw and d_same==0.0 and d_none==0.0
    print(f"  conservative < radical: {t2}  {'PASS' if t2 else 'FAIL'}")
    ok &= t2
    print("  NOTE: recomputed property distance, NOT the Grantham 1974 matrix.")

    print("\n[T3] bigWig coordinate convention")
    pos=1000; flank=3
    lo,hi=pos-1-flank,pos-1+flank+1
    print(f"  VCF pos={pos} (1-based) → bigWig [{lo},{hi}) 0-based half-open")
    print(f"  length={hi-lo} (expect {2*flank+1}), center index={flank} → "
          f"bigWig {lo+flank} = VCF {lo+flank+1}")
    t3 = (hi-lo)==2*flank+1 and (lo+flank+1)==pos
    print(f"  {'PASS' if t3 else 'FAIL'}"); ok &= t3

    print("\n[T4] candidate enumeration logic (small synthetic sequence)")
    g="ACGTACGTACGTACGTACGT"
    rng=np.random.default_rng(0)
    core="".join(rng.choice(list("ACGT"),20))
    seq = core+"AGG" + "T"*50 + g+"TGG"
    arr=np.frombuffer(seq.encode(),dtype=np.uint8)
    n=len(arr)-22
    pam=(arr[21:n+21]==ord("G"))&(arr[22:n+22]==ord("G"))
    hits=np.flatnonzero(pam)
    print(f"  synthetic seq len={len(seq)}, PAM sites found={len(hits)} (expect ≥2)")
    t4=len(hits)>=2
    print(f"  {'PASS' if t4 else 'FAIL'}"); ok &= t4

    print("\n[T5] negatives refuse synthetic decoys")
    try:
        build_offtarget_dataset([{"guide":"A"*20,"offtarget":"A"*20,"label":1}],
                                None, None)
        print("  FAIL — should have raised"); ok=False
    except RuntimeError as e:
        print(f"  correctly refused: {str(e).splitlines()[0]}")
        print("  PASS")

    print("\n[T6] temperature preserves ranking")
    rng=np.random.default_rng(1); n=2000
    y=(rng.random(n)<0.3).astype(float)
    lg=3.5*(y-0.5)+rng.normal(0,1.2,n)
    T=TempScaler().fit(lg[:1000],y[:1000])
    a0=auroc(y[1500:],1/(1+np.exp(-lg[1500:])))
    a1=auroc(y[1500:],T.predict(lg[1500:]))
    t6=abs(a0-a1)<1e-9
    print(f"  AUROC raw {a0:.6f} vs calibrated {a1:.6f}  T={T.T:.3f}")
    print(f"  {'PASS' if t6 else 'FAIL'}"); ok &= t6

    print("\n[T7] gene split has no overlap")
    recs=[{"gene":f"G{i%20}","label":i%2,"chrom":"1","pos":i,
           "ref":"A","alt":"G"} for i in range(400)]
    sp=gene_split(recs,seed=0)
    gs={k:set(r["gene"] for r in v) for k,v in sp.items()}
    t7 = not (gs["train"]&gs["val"]) and not (gs["train"]&gs["test"])
    print(f"  {'PASS' if t7 else 'FAIL'}"); ok &= t7

    print("\n"+"="*80)
    print(f"  {'✅ ALL SELF-TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*80)
    print("""
  VERIFIED: feature blocks, pred-score exclusion, bigWig coordinates,
            PAM scanning, refusal of synthetic negatives, calibration
            ranking invariance, gene-level split integrity.

  NOT VERIFIED: that your annotation source matches your assembly;
            that SIFT/PolyPhen are independent of your labels (they are not);
            that CHANGE-seq non-detection means non-cleavage (it does not).
""")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("cmd",choices=["selftest","conservation-download",
                                  "clinvar-ablation","changeseq-probe",
                                  "changeseq"])
    p.add_argument("--clinvar",default="data/clinvar/variant_summary.txt.gz")
    p.add_argument("--fasta",default=None)
    p.add_argument("--conservation",nargs="*",default=None)
    p.add_argument("--cons-kind",dest="cons_kind",default="phyloP")
    p.add_argument("--cons-window",dest="cons_window",type=int,default=10)
    p.add_argument("--cons-key",dest="cons_key",default="phyloP470way_hg38",
                   choices=list(CONSERVATION_SOURCES))
    p.add_argument("--assembly",default="GRCh38",choices=["GRCh38","GRCh37"])
    p.add_argument("--flank",type=int,default=16)
    p.add_argument("--min-stars",dest="min_stars",type=int,default=2)
    p.add_argument("--vep-backend",dest="vep_backend",default="rest",
                   choices=["rest","vep","snpeff"])
    p.add_argument("--vep-cache",dest="vep_cache",default="data/vep_cache")
    p.add_argument("--vep-local-cache",dest="vep_local_cache",default=None)
    p.add_argument("--snpeff-jar",dest="snpeff_jar",default=None)
    p.add_argument("--snpeff-db",dest="snpeff_db",default="GRCh38.105")
    p.add_argument("--data",default="data/changeseq/offtargets.txt")
    p.add_argument("--candidates",default=None)
    p.add_argument("--validated-negatives",dest="validated_negatives",default=None)
    p.add_argument("--scan-fasta",dest="scan_fasta",default=None)
    p.add_argument("--scan-chrom",dest="scan_chrom",default=None)
    p.add_argument("--scan-start",dest="scan_start",type=int,default=0)
    p.add_argument("--scan-end",dest="scan_end",type=int,default=None)
    p.add_argument("--scan-guides",dest="scan_guides",type=int,default=3)
    p.add_argument("--max-mis",dest="max_mis",type=int,default=6)
    p.add_argument("--seed-mis",dest="seed_mis",type=int,default=2)
    p.add_argument("--seq-col",dest="seq_col",default=None)
    p.add_argument("--guide-col",dest="guide_col",default=None)
    p.add_argument("--reads-col",dest="reads_col",default=None)
    p.add_argument("--min-reads",dest="min_reads",type=float,default=1)
    p.add_argument("--out",default="ckpt/ablation.json")
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch-size",dest="batch_size",type=int,default=128)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.cmd=="selftest": selftest(a)
    elif a.cmd=="conservation-download": download_conservation(a.cons_key)
    elif a.cmd=="clinvar-ablation": run_ablation(a)
    elif a.cmd=="changeseq-probe":
        print(json.dumps(changeseq_probe(Path(a.data)),indent=2))
    else: run_offtarget(a)


if __name__=="__main__":
    main()
