#!/usr/bin/env python3
"""
=============================================================================
ORGANOID DRUG-RESPONSE — GEO SERIES-MATRIX LOADER + VERIFICATION WORKFLOW
=============================================================================
License: MIT | 2026

LITERATURE
 [G1] Barrett et al. "NCBI GEO: archive for functional genomics data sets —
      update." Nucleic Acids Res. 41, D991 (2013).
 [G2] Clough & Barrett. "The Gene Expression Omnibus database."
      Methods Mol. Biol. 1418, 93 (2016).
 [O1] Drost & Clevers. "Organoids in cancer research."
      Nat. Rev. Cancer 18, 407 (2018).
 [O2] Veninga & Voest. "Tumor organoids: Opportunities and challenges to
      guide precision cancer treatment." Cancer Cell 39, 1190 (2021).

=============================================================================
HONEST SCOPE — WHY THERE IS NO ACCESSION LIST HERE
=============================================================================
  ❌ This file does NOT ship a list of GSE accessions claimed to contain
     organoid drug-response data. I cannot verify accession contents without
     querying GEO, and a fabricated list would send you after data that does
     not exist — the most expensive kind of wrong.
  ✅ Instead: a real series-matrix parser, a metadata-driven SCREEN that
     decides whether a given accession actually has paired
     (expression, drug, response) data, and a registry you fill after checking.

  ⚠️ Most organoid GEO series are expression-only, with no drug response.
     The screen will reject them. That is the correct behaviour.
  ⚠️ Drug response in GEO is often in the *supplementary* files or in the
     paper, not in the series matrix. The screen reports this explicitly.
  ❌ Nothing here is clinical. Organoid response ≠ patient response.
=============================================================================
"""
from __future__ import annotations
import argparse, gzip, json, logging, re, urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("organoid_geo")

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

ORGANOID_TERMS = ["organoid", "spheroid", "3d culture", "tumoroid",
                  "patient-derived organoid", "pdo"]
DRUG_TERMS = ["drug", "compound", "treatment", "treated", "ic50", "ec50",
              "dose", "viability", "sensitivity", "response", "inhibitor",
              "chemotherap", "screen"]
RESPONSE_TERMS = ["ic50", "ec50", "auc", "viability", "response",
                  "sensitivity", "gi50", "% inhibition", "survival"]


def series_matrix_url(gse: str) -> str:
    gse = gse.upper().strip()
    if not re.fullmatch(r"GSE\d+", gse):
        raise ValueError(f"bad accession: {gse}")
    stub = gse[:-3] + "nnn" if len(gse) > 4 else "GSEnnn"
    return f"{GEO_FTP}/{stub}/{gse}/matrix/{gse}_series_matrix.txt.gz"


def fetch_series_matrix(gse: str, dest="data/geo") -> Optional[Path]:
    d = Path(dest); d.mkdir(parents=True, exist_ok=True)
    out = d / f"{gse.upper()}_series_matrix.txt.gz"
    if out.exists():
        return out
    url = series_matrix_url(gse)
    log.info(f"[GEO] {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            out.write_bytes(r.read())
        return out
    except Exception as e:
        log.warning(f"[GEO] fetch failed: {e}")
        log.info(f"[GEO] browse: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}")
        out.unlink(missing_ok=True)
        return None


def parse_series_matrix(path: Path) -> Dict:
    """
    GEO SOFT series-matrix format:
      !Series_*   : study metadata
      !Sample_*   : per-sample metadata (tab-separated, one column per GSM)
      !series_matrix_table_begin / _end : expression block
    """
    meta: Dict[str, List[str]] = {}
    rows, ids = [], None
    in_table = False
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True; continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False; continue
            if in_table:
                p = line.split("\t")
                if ids is None:
                    ids = [x.strip('"') for x in p[1:]]
                else:
                    rows.append((p[0].strip('"'),
                                 [float(x) if x not in ("", "NA", "null") else np.nan
                                  for x in p[1:]]))
                continue
            if line.startswith("!"):
                key = line.split("\t")[0][1:]
                vals = [x.strip('"') for x in line.split("\t")[1:]]
                meta.setdefault(key, []).extend(vals)
    X = np.array([r[1] for r in rows], dtype=np.float32) if rows else None
    probes = [r[0] for r in rows]
    return {"meta": meta, "sample_ids": ids or [],
            "expression": X, "probes": probes,
            "n_samples": len(ids or []), "n_features": len(probes)}


def screen(parsed: Dict, gse: str) -> Dict:
    """
    Decide whether this accession plausibly has PAIRED
    (expression, drug, response) organoid data.
    """
    m = parsed["meta"]
    blob = " ".join(" ".join(v) for v in m.values()).lower()
    title = " ".join(m.get("Series_title", [])).lower()
    summary = " ".join(m.get("Series_summary", [])).lower()
    design = " ".join(m.get("Series_overall_design", [])).lower()
    chars = " ".join(m.get("Sample_characteristics_ch1", [])).lower()

    org = [t for t in ORGANOID_TERMS if t in blob]
    drug = [t for t in DRUG_TERMS if t in (chars + " " + design + " " + summary)]
    resp = [t for t in RESPONSE_TERMS if t in chars]

    has_expr = parsed["expression"] is not None and parsed["n_samples"] > 0
    verdict = ("CANDIDATE — verify manually"
               if (org and drug and has_expr) else "REJECT")
    reasons = []
    if not org: reasons.append("no organoid term in metadata")
    if not drug: reasons.append("no drug/treatment term in sample metadata")
    if not has_expr: reasons.append("no expression matrix")
    if not resp: reasons.append("no numeric response term in characteristics "
                                "(response may be in supplementary files)")
    return {
        "accession": gse, "verdict": verdict, "reasons": reasons,
        "organoid_terms": org, "drug_terms": drug, "response_terms": resp,
        "n_samples": parsed["n_samples"], "n_features": parsed["n_features"],
        "title": title[:200],
        "NEXT_STEP": ("Open https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
                      + gse + " and confirm: (1) organoid model, (2) drug panel, "
                      "(3) numeric response per sample. Check supplementary files."),
    }


def build_dataset(parsed: Dict, response_map: Dict[str, float],
                  log_transform: bool = True) -> Dict:
    """
    Assemble (X, y) ONLY after you have manually verified the accession and
    extracted a per-sample response. response_map: GSM → numeric response.
    """
    ids = parsed["sample_ids"]
    X = parsed["expression"]
    if X is None:
        raise ValueError("no expression matrix")
    keep = [i for i, s in enumerate(ids) if s in response_map]
    if len(keep) < 8:
        raise ValueError(f"only {len(keep)} samples with response — too few")
    Xk = X[:, keep].T
    y = np.array([response_map[ids[i]] for i in keep], np.float32)
    if log_transform and np.nanmax(Xk) > 100:
        Xk = np.log2(np.clip(Xk, 1, None))
    mu = np.nanmean(Xk, 0); sd = np.nanstd(Xk, 0) + 1e-6
    Xk = np.nan_to_num((Xk - mu) / sd, nan=0.0)
    log.info(f"[dataset] {Xk.shape[0]} samples × {Xk.shape[1]} features")
    return {"X": Xk, "y": y, "sample_ids": [ids[i] for i in keep],
            "NOTE": "Verify organoid provenance and response units before use."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["screen", "parse"])
    p.add_argument("--gse", nargs="+", required=True)
    p.add_argument("--out", default="data/geo")
    a = p.parse_args()
    reports = []
    for g in a.gse:
        path = fetch_series_matrix(g, a.out)
        if path is None:
            reports.append({"accession": g, "verdict": "FETCH_FAILED"}); continue
        parsed = parse_series_matrix(path)
        if a.cmd == "parse":
            print(json.dumps({"accession": g,
                              "n_samples": parsed["n_samples"],
                              "n_features": parsed["n_features"],
                              "meta_keys": list(parsed["meta"])[:30]}, indent=2))
        else:
            r = screen(parsed, g); reports.append(r)
            print(json.dumps(r, indent=2))
    Path(a.out, "screen.json").write_text(json.dumps(reports, indent=2))
    print("\n" + "="*72)
    print("  NO ACCESSION IS PRE-CERTIFIED HERE.")
    print("  A CANDIDATE verdict means 'metadata is consistent with organoid")
    print("  drug response' — NOT that the data exists in usable form.")
    print("  Confirm on the GEO page and in the paper before building a dataset.")
    print("="*72)


if __name__ == "__main__":
    main()
