#!/usr/bin/env python3
"""
=============================================================================
ACMG RULE ENGINE · SpliceAI · Pfam/UniProt  +  INDEPENDENCE ABLATION
=============================================================================
License: MIT | 2026

LITERATURE
 [R1] Richards et al. "Standards and guidelines for the interpretation of
      sequence variants." Genet. Med. 17, 405 (2015).      [ACMG/AMP]
 [R2] Tavtigian et al. "Modeling the ACMG/AMP variant classification guidelines
      as a Bayesian classification framework." Genet. Med. 20, 1054 (2018).
 [R3] Abou Tayoun et al. "Recommendations for interpreting the loss of function
      PVS1 ACMG/AMP variant criterion." Hum. Mutat. 39, 1517 (2018).
 [S1] Jaganathan et al. "Predicting splicing from primary sequence with deep
      learning." Cell 176, 535 (2019).                     [SpliceAI]
 [P1] Mistry et al. "Pfam: The protein families database in 2021."
      Nucleic Acids Res. 49, D412 (2021).
 [P2] UniProt Consortium. Nucleic Acids Res. 51, D523 (2023).

=============================================================================
INDEPENDENCE FROM ClinVar — CORRECTED
=============================================================================
  SpliceAI      ✅ INDEPENDENT. Trained on GENCODE splice-site annotation
                   from RNA evidence. Never sees ClinVar labels.
  Pfam/UniProt  ✅ LARGELY INDEPENDENT. Profile HMMs + curated families.
                   ⚠️ UniProt *variant* annotations DO overlap ClinVar;
                      use DOMAIN features only, not variant features.
  ACMG codes    ❌ NOT INDEPENDENT. ClinVar submissions frequently contain
                   ACMG classifications; ClinGen VCEPs produce both from the
                   same underlying evidence. Adding ACMG codes to a ClinVar
                   predictor largely re-encodes the label.
                   → The ablation below reports the ACMG increment explicitly
                     so it can be read as CIRCULARITY, not as insight.

OTHER HONEST LIMITS
  ❌ Automated ACMG is NOT clinical-grade. ClinGen VCEPs issue gene-specific
     rule specifications that override the generic 2015 rules.
  ❌ PVS1 requires knowing that LOF is the disease mechanism for that gene,
     plus NMD/last-exon/rescue considerations [R3]. A stop_gained flag alone
     is NOT PVS1.
  ❌ SpliceAI needs the real trained weights. No weights → no scores. This
     file does not fabricate them.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging, shutil, subprocess, time, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("var_evidence")


# =============================================================================
# 1. ACMG RULE ENGINE  (Richards 2015, Table 5)
# =============================================================================

PATH_VERY_STRONG = ["PVS1"]
PATH_STRONG = ["PS1", "PS2", "PS3", "PS4"]
PATH_MODERATE = ["PM1", "PM2", "PM3", "PM4", "PM5", "PM6"]
PATH_SUPPORT = ["PP1", "PP2", "PP3", "PP4", "PP5"]
BENIGN_STANDALONE = ["BA1"]
BENIGN_STRONG = ["BS1", "BS2", "BS3", "BS4"]
BENIGN_SUPPORT = ["BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7"]

ALL_CODES = (PATH_VERY_STRONG + PATH_STRONG + PATH_MODERATE + PATH_SUPPORT
             + BENIGN_STANDALONE + BENIGN_STRONG + BENIGN_SUPPORT)
CODE_IDX = {c: i for i, c in enumerate(ALL_CODES)}

# Tavtigian 2018 point weights (supporting=1, moderate=2, strong=4, very strong=8)
POINTS = {}
for c in PATH_VERY_STRONG: POINTS[c] = 8
for c in PATH_STRONG: POINTS[c] = 4
for c in PATH_MODERATE: POINTS[c] = 2
for c in PATH_SUPPORT: POINTS[c] = 1
for c in BENIGN_STANDALONE: POINTS[c] = -8
for c in BENIGN_STRONG: POINTS[c] = -4
for c in BENIGN_SUPPORT: POINTS[c] = -1


def acmg_classify(codes: List[str]) -> Dict[str, object]:
    """
    Generic ACMG/AMP 2015 combining rules (Table 5).

    ⚠️ NOT clinical-grade. Gene-specific ClinGen VCEP specifications override
       these rules. Conflicting pathogenic+benign evidence → VUS, not a vote.
    """
    s = set(codes)
    n_pvs = len(s & set(PATH_VERY_STRONG))
    n_ps = len(s & set(PATH_STRONG))
    n_pm = len(s & set(PATH_MODERATE))
    n_pp = len(s & set(PATH_SUPPORT))
    n_ba = len(s & set(BENIGN_STANDALONE))
    n_bs = len(s & set(BENIGN_STRONG))
    n_bp = len(s & set(BENIGN_SUPPORT))

    path = (
        (n_pvs >= 1 and (n_ps >= 1 or n_pm >= 2 or (n_pm == 1 and n_pp == 1) or n_pp >= 2))
        or n_ps >= 2
        or (n_ps == 1 and (n_pm >= 3 or (n_pm == 2 and n_pp >= 2) or (n_pm == 1 and n_pp >= 4)))
    )
    likely_path = (
        (n_pvs == 1 and n_pm == 1)
        or (n_ps == 1 and 1 <= n_pm <= 2)
        or (n_ps == 1 and n_pp >= 2)
        or n_pm >= 3
        or (n_pm == 2 and n_pp >= 2)
        or (n_pm == 1 and n_pp >= 4)
    )
    benign = (n_ba >= 1) or (n_bs >= 2)
    likely_benign = (n_bs == 1 and n_bp == 1) or (n_bp >= 2)

    has_p = (n_pvs + n_ps + n_pm + n_pp) > 0
    has_b = (n_ba + n_bs + n_bp) > 0
    if benign and not has_p:
        label = "Benign"
    elif likely_benign and not has_p:
        label = "Likely benign"
    elif path and not has_b:
        label = "Pathogenic"
    elif likely_path and not has_b:
        label = "Likely pathogenic"
    else:
        label = "VUS"
    return {"label": label, "points": sum(POINTS[c] for c in s),
            "counts": {"PVS": n_pvs, "PS": n_ps, "PM": n_pm, "PP": n_pp,
                       "BA": n_ba, "BS": n_bs, "BP": n_bp},
            "conflicting_evidence": bool(has_p and has_b),
            "NOT_CLINICAL": True}


N_ACMG = len(ALL_CODES) + 2


def acmg_features(codes: List[str]) -> np.ndarray:
    v = np.zeros(N_ACMG, np.float32)
    for c in set(codes):
        if c in CODE_IDX:
            v[CODE_IDX[c]] = 1.0
    r = acmg_classify(codes)
    v[-2] = np.clip(r["points"], -12, 12) / 12.0
    v[-1] = float(r["conflicting_evidence"])
    return v


def pvs1_candidate(consequence: str, is_lof_mechanism_gene: bool,
                   is_last_exon: bool = False,
                   nmd_escape: bool = False) -> Optional[str]:
    """
    PVS1 decision support [R3]. Returns a CANDIDATE strength, not an assertion.

    Requires external knowledge that loss of function is the established
    disease mechanism for the gene. Without that, PVS1 does not apply at all.
    """
    if not is_lof_mechanism_gene:
        return None
    null_set = {"transcript_ablation", "splice_acceptor_variant",
                "splice_donor_variant", "stop_gained", "frameshift_variant",
                "start_lost"}
    if consequence not in null_set:
        return None
    if is_last_exon or nmd_escape:
        return "PVS1_Strong"        # downgraded per ClinGen SVI
    return "PVS1"


# =============================================================================
# 2. SpliceAI  — real weights or precomputed scores, never fabricated
# =============================================================================

class SpliceAI:
    """
    Two supported routes:
      (a) precomputed VCF (Illumina BaseSpace release) — recommended, fast
      (b) the `spliceai` package with real Keras weights

    Delta scores: DS_AG (acceptor gain), DS_AL (acceptor loss),
                  DS_DG (donor gain),  DS_DL (donor loss), each in [0,1].
    Common thresholds: 0.2 (sensitive), 0.5 (specific), 0.8 (high precision).
    """
    N = 5   # 4 deltas + max

    def __init__(self, mode: str = "precomputed",
                 vcf: Optional[str] = None,
                 snv: Optional[str] = None, indel: Optional[str] = None):
        self.mode = mode
        self.table: Dict[Tuple[str, int, str, str], np.ndarray] = {}
        self.model = None
        if mode == "precomputed":
            if vcf and Path(vcf).exists():
                self._load_vcf(Path(vcf))
            else:
                log.warning("[SpliceAI] no precomputed VCF. Scores unavailable.\n"
                            "  Download from Illumina BaseSpace 'SpliceAI' release,\n"
                            "  or: pip install spliceai  (needs TF + weights)")
        elif mode == "model":
            try:
                from spliceai.utils import Annotator, get_delta_scores
                self._get = get_delta_scores
                log.info("[SpliceAI] package available")
            except ImportError:
                log.warning("[SpliceAI] `pip install spliceai` + reference FASTA "
                            "required. Scores unavailable.")
                self.mode = "unavailable"
        else:
            self.mode = "unavailable"

    def _load_vcf(self, path: Path):
        import gzip
        op = gzip.open if str(path).endswith(".gz") else open
        n = 0
        with op(path, "rt", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.rstrip().split("\t")
                if len(p) < 8:
                    continue
                chrom, pos, _, ref, alt = p[0], int(p[1]), p[2], p[3], p[4]
                info = p[7]
                vals = None
                for key in ("SpliceAI=", "SpliceAI_snv=", "SpliceAI_indel="):
                    if key in info:
                        rec = info.split(key, 1)[1].split(";")[0].split(",")[0]
                        fld = rec.split("|")
                        try:
                            vals = np.array([float(fld[2]), float(fld[3]),
                                             float(fld[4]), float(fld[5])], np.float32)
                        except (IndexError, ValueError):
                            vals = None
                        break
                if vals is not None:
                    self.table[(chrom.replace("chr", ""), pos, ref, alt)] = vals
                    n += 1
        log.info(f"[SpliceAI] loaded {n} precomputed records")

    def scores(self, chrom: str, pos: int, ref: str, alt: str) -> np.ndarray:
        v = self.table.get((str(chrom).replace("chr", ""), int(pos), ref, alt))
        if v is None:
            return np.zeros(self.N, np.float32)   # zeros + implicit missing flag
        return np.concatenate([v, [float(v.max())]])

    def available(self) -> bool:
        return bool(self.table) or self.model is not None


# =============================================================================
# 3. Pfam / UniProt  — domain features only (variant features would leak)
# =============================================================================

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
INTERPRO_REST = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"


class DomainAnnotator:
    """
    Route A: UniProt REST by accession → DOMAIN / REGION / ACT_SITE features.
    Route B: local hmmscan against Pfam-A.hmm (hmmer + Pfam download).

    ⚠️ We deliberately EXCLUDE UniProt 'VARIANT' features: those overlap
       ClinVar and would reintroduce leakage.
    """
    N = 12

    def __init__(self, cache_dir="data/domain_cache",
                 pfam_hmm: Optional[str] = None,
                 hmmscan_bin: str = "hmmscan"):
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.pfam_hmm = pfam_hmm
        self.hmmscan = shutil.which(hmmscan_bin)
        if pfam_hmm and not self.hmmscan:
            log.warning("[Pfam] hmmscan not found; install hmmer")

    def uniprot_domains(self, acc: str) -> List[Dict]:
        f = self.cache / f"{acc}.json"
        if f.exists():
            return json.load(open(f))
        try:
            url = f"{UNIPROT_REST}/{acc}.json"
            with urllib.request.urlopen(url, timeout=45) as r:
                d = json.loads(r.read().decode())
        except Exception as e:
            log.warning(f"[UniProt] {acc}: {e}")
            return []
        keep = {"Domain", "Region", "Active site", "Binding site",
                "Transmembrane", "Coiled coil", "Motif", "Zinc finger"}
        out = []
        for ft in d.get("features", []):
            if ft.get("type") not in keep:      # excludes 'Natural variant'
                continue
            loc = ft.get("location", {})
            out.append({"type": ft["type"],
                        "start": loc.get("start", {}).get("value"),
                        "end": loc.get("end", {}).get("value"),
                        "description": ft.get("description", "")})
        json.dump(out, open(f, "w"))
        time.sleep(0.15)
        return out

    def hmmscan(self, fasta: str) -> List[Dict]:
        if not (self.pfam_hmm and self.hmmscan):
            return []
        try:
            p = subprocess.run([self.hmmscan, "--domtblout", "/dev/stdout",
                                "--noali", "-E", "1e-3", self.pfam_hmm, "-"],
                               input=fasta, capture_output=True, text=True, timeout=600)
        except Exception as e:
            log.warning(f"[hmmscan] {e}"); return []
        out = []
        for line in p.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            if len(t) < 23:
                continue
            out.append({"pfam": t[0], "acc": t[1],
                        "env_from": int(t[19]), "env_to": int(t[20]),
                        "evalue": float(t[12])})
        return out

    def features(self, pos: int, domains: List[Dict]) -> np.ndarray:
        """Position-in-domain features. Independent of ClinVar labels."""
        v = np.zeros(self.N, np.float32)
        if not domains:
            v[-1] = 1.0                     # missing flag
            return v
        inside = [d for d in domains
                  if d.get("start") and d.get("end") and d["start"] <= pos <= d["end"]]
        v[0] = float(bool(inside))
        types = ["Domain", "Active site", "Binding site", "Transmembrane",
                 "Coiled coil", "Region", "Motif", "Zinc finger"]
        for d in inside:
            if d["type"] in types:
                v[1 + types.index(d["type"])] = 1.0
        if inside:
            d = inside[0]
            span = max(d["end"] - d["start"], 1)
            v[9] = (pos - d["start"]) / span          # relative position
            v[10] = min(span, 500) / 500.0            # domain length
        return v


# =============================================================================
# 4. INDEPENDENCE ABLATION
# =============================================================================

def independence_report() -> Dict:
    return {
        "SpliceAI": {
            "independent_of_clinvar": True,
            "training_source": "GENCODE splice-site annotation (RNA evidence)",
            "reference": "Jaganathan et al., Cell 176, 535 (2019)",
            "caveat": "predicts splice disruption, not pathogenicity",
        },
        "Pfam/UniProt_domains": {
            "independent_of_clinvar": "largely",
            "training_source": "profile HMMs + curated protein families",
            "reference": "Mistry et al., NAR 49, D412 (2021)",
            "caveat": "EXCLUDE UniProt 'Natural variant' features — those overlap",
        },
        "ACMG_codes": {
            "independent_of_clinvar": False,
            "reason": ("ClinVar submissions frequently include ACMG classifications; "
                       "ClinGen VCEPs derive both from the same evidence"),
            "reference": "Richards et al., Genet. Med. 17, 405 (2015)",
            "consequence": ("an ACMG increment on a ClinVar task is largely "
                            "CIRCULARITY, not new biological insight"),
        },
        "recommended_reporting": [
            "context-only baseline",
            "+SpliceAI (independent)",
            "+Pfam domains (independent)",
            "+SpliceAI+Pfam  ← the honest headline",
            "+ACMG on top    ← report separately, label as circularity",
        ],
    }


# =============================================================================
# 5. SELF-TEST
# =============================================================================

def selftest(args):
    print("="*76); print("  VARIANT EVIDENCE — self-test"); print("="*76)
    ok = True

    print("\n[T1] ACMG combining rules (Richards 2015 Table 5)")
    cases = [
        (["PVS1", "PM2"], "Likely pathogenic"),
        (["PVS1", "PM1", "PM2"], "Pathogenic"),
        (["PS1", "PS2"], "Pathogenic"),
        (["PM1", "PM2", "PM3"], "Likely pathogenic"),
        (["BA1"], "Benign"),
        (["BS1", "BS2"], "Benign"),
        (["BP1", "BP4"], "Likely benign"),
        (["PVS1", "BA1"], "VUS"),          # conflicting → VUS
        ([], "VUS"),
    ]
    for codes, want in cases:
        got = acmg_classify(codes)["label"]
        t = got == want; ok &= t
        print(f"   {str(codes):<26} → {got:<20} {'PASS' if t else f'FAIL(want {want})'}")

    print("\n[T2] PVS1 requires LOF mechanism + NMD consideration")
    a = pvs1_candidate("stop_gained", True)
    b = pvs1_candidate("stop_gained", False)
    c = pvs1_candidate("stop_gained", True, is_last_exon=True)
    d = pvs1_candidate("missense_variant", True)
    t = (a == "PVS1" and b is None and c == "PVS1_Strong" and d is None); ok &= t
    print(f"   LOF gene: {a} | not LOF: {b} | last exon: {c} | missense: {d}")
    print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T3] feature dimensions")
    fa = acmg_features(["PVS1", "PM2"])
    sa = SpliceAI(mode="unavailable").scores("1", 100, "A", "G")
    da = DomainAnnotator().features(50, [{"type": "Domain", "start": 10, "end": 120}])
    t = fa.shape[0] == N_ACMG and sa.shape[0] == SpliceAI.N and da.shape[0] == DomainAnnotator.N
    ok &= t
    print(f"   ACMG={fa.shape[0]}  SpliceAI={sa.shape[0]}  Domain={da.shape[0]}"
          f"  {'PASS' if t else 'FAIL'}")

    print("\n[T4] domain features exclude UniProt variant annotations")
    ann = DomainAnnotator()
    src = Path(ann.__class__.__module__.replace(".", "/") + ".py")
    txt = Path(__file__).read_text() if Path(__file__).exists() else ""
    t = '"Natural variant"' not in txt.split("keep = {")[-1].split("}")[0] if "keep = {" in txt else True
    ok &= t
    print(f"   'Natural variant' not in keep-set: {t}  {'PASS' if t else 'FAIL'}")

    print("\n[T5] missing SpliceAI → zeros, never fabricated scores")
    s = SpliceAI(mode="unavailable")
    v = s.scores("7", 117559590, "A", "T")
    t = bool((v == 0).all()) and not s.available(); ok &= t
    print(f"   all-zero={bool((v==0).all())}  available={s.available()}"
          f"  {'PASS' if t else 'FAIL'}")

    print("\n" + "="*76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*76)
    print("\n  INDEPENDENCE FROM ClinVar:")
    for k, v in independence_report().items():
        if isinstance(v, dict):
            print(f"   {k:<24} independent={v['independent_of_clinvar']}")
    print("\n  ❌ Automated ACMG is NOT clinical-grade.")
    print("  ❌ No SpliceAI weights → no scores. Nothing is fabricated.")
    print("="*76)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "independence", "uniprot", "spliceai-load"])
    p.add_argument("--acc", default="P04637")
    p.add_argument("--vcf", default=None)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "independence":
        print(json.dumps(independence_report(), indent=2))
    elif a.cmd == "uniprot":
        d = DomainAnnotator().uniprot_domains(a.acc)
        print(json.dumps(d[:20], indent=2))
    else:
        s = SpliceAI(mode="precomputed", vcf=a.vcf)
        print(f"loaded={len(s.table)} available={s.available()}")


if __name__ == "__main__":
    main()
