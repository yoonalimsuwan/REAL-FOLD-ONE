#!/usr/bin/env python3
"""
=============================================================================
T2 CURATION TEMPLATE — (factor, dose, time) → cell-type composition
=============================================================================
License: MIT | 2026

PURPOSE
  Provide a strict schema + validator for the curated T2 dataset.
  Every record must be VERIFIED (has DOI, manually checked) before use.

HONEST SCOPE
  ✅ This defines WHAT to curate and CHECKS consistency.
  ❌ It does NOT auto-generate labels. Pairing treatment → composition
     requires reading the GEO metadata and the source paper.
  ⚠️ Cell-type composition from scRNA-seq depends on annotation quality.
     Treat every composition as an ESTIMATE, not ground truth.
=============================================================================
"""
from __future__ import annotations
import argparse, json, logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("t2_curation")

# Signalling factors actually used in organoid protocols
FACTORS = ["WNT", "RSPO", "BMP4", "NOGGIN", "FGF2", "FGF4", "FGF9", "FGF10",
           "ACTIVIN", "NODAL", "TGFB1", "SHH", "RA", "EGF", "HGF", "VEGF",
           "IGF1", "NOTCH_AG", "NOTCH_INH", "CHIR", "SB431542", "LDN193189",
           "DAPT", "Y27632", "OTHER"]

# Coarse, marker-definable fate categories (must match MDOE FATES)
FATES = ["PLURIPOTENT", "ENDODERM", "MESODERM", "ECTODERM",
         "NEURAL_PROGENITOR", "NEURON", "GLIA",
         "INTESTINAL_EPITHELIUM", "GOBLET", "PANETH", "ENTEROCYTE",
         "HEPATOCYTE", "CHOLANGIOCYTE",
         "NEPHRON_PROGENITOR", "PODOCYTE", "TUBULE",
         "CARDIOMYOCYTE", "ENDOTHELIAL", "MESENCHYMAL", "UNKNOWN"]
N_FATE = len(FATES)
F_IDX = {f: i for i, f in enumerate(FATES)}


@dataclass
class T2Record:
    """
    One curated (factor, dose, time) → composition record.

    REQUIRED for a record to be USABLE:
      • factor is in FACTORS
      • dose_nM > 0, time_h > 0
      • composition has length N_FATE, non-negative, sums to ~1
      • source_gse is a valid GSE accession
      • doi is present
      • verified == True  (you read the paper and checked)
    """
    record_id: str
    # treatment
    factor: str
    dose_nM: float
    time_h: float
    # outcome
    composition: List[float]          # length N_FATE, sums to 1
    # provenance
    source_gse: str
    sample_id: str                    # e.g. GSM number
    doi: Optional[str] = None
    organoid_type: Optional[str] = None   # intestinal / cerebral / kidney ...
    cell_type_calling: Optional[str] = None  # "marker" | "automated" | "author"
    n_cells: Optional[int] = None
    notes: str = ""
    verified: bool = False

    # ---- validation ----
    def problems(self) -> List[str]:
        p = []
        if self.factor not in FACTORS:
            p.append(f"factor '{self.factor}' not in FACTORS")
        if self.dose_nM <= 0:
            p.append(f"dose_nM must be > 0 (got {self.dose_nM})")
        if self.time_h <= 0:
            p.append(f"time_h must be > 0 (got {self.time_h})")
        if len(self.composition) != N_FATE:
            p.append(f"composition length {len(self.composition)} != {N_FATE}")
        else:
            c = np.array(self.composition)
            if (c < 0).any():
                p.append("composition has negative entries")
            if abs(c.sum() - 1.0) > 1e-3:
                p.append(f"composition sums to {c.sum():.5f}, expected ~1.0")
        if not self.source_gse.startswith("GSE"):
            p.append(f"source_gse '{self.source_gse}' is not a GSE accession")
        if not self.sample_id.startswith("GSM"):
            p.append(f"sample_id '{self.sample_id}' is not a GSM id")
        if not self.doi:
            p.append("missing doi")
        if not self.verified:
            p.append("not verified (you must read the paper)")
        return p

    def is_usable(self) -> bool:
        return len(self.problems()) == 0


def load_jsonl(path: Path) -> List[T2Record]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(T2Record(**json.loads(line)))
    return out


def validate_all(records: List[T2Record]) -> Dict:
    usable, broken = [], []
    for r in records:
        if r.is_usable():
            usable.append(r)
        else:
            broken.append(r)
    # duplicate check
    keys = [f"{r.source_gse}:{r.sample_id}" for r in usable]
    dup = len(keys) - len(set(keys))
    # composition sanity
    n_unknown = 0
    for r in usable:
        if r.composition[F_IDX["UNKNOWN"]] > 0.5:
            n_unknown += 1
    return {
        "total": len(records),
        "usable": len(usable),
        "broken": len(broken),
        "duplicates": dup,
        "mostly_unknown": n_unknown,
        "factor_coverage": sorted({r.factor for r in usable}),
        "organoid_types": sorted({r.organoid_type for r in usable if r.organoid_type}),
    }


def write_template(path: Path, n_example: int = 3):
    """Write a starter JSONL with example records (marked unverified)."""
    examples = [
        T2Record(
            record_id="example_001",
            factor="CHIR", dose_nM=3000.0, time_h=72.0,
            composition=[0.1, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1],
            source_gse="GSE000000", sample_id="GSM0000001",
            doi="10.1234/example", organoid_type="intestinal",
            cell_type_calling="marker", n_cells=1200,
            notes="EXAMPLE — replace with real data", verified=False),
        T2Record(
            record_id="example_002",
            factor="FGF2", dose_nM=100.0, time_h=48.0,
            composition=[0.05, 0.1, 0.05, 0.2, 0.4, 0.1, 0.05,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05],
            source_gse="GSE000000", sample_id="GSM0000002",
            doi="10.1234/example", organoid_type="cerebral",
            cell_type_calling="automated", n_cells=800,
            notes="EXAMPLE — replace with real data", verified=False),
    ]
    with open(path, "w") as f:
        f.write("# T2 curation: one JSON object per line.\n")
        f.write("# REQUIRED: factor, dose_nM, time_h, composition "
                f"(length {N_FATE}, sums to 1),\n")
        f.write("#           source_gse, sample_id, doi, verified=true.\n")
        f.write("# composition order: " + ", ".join(FATES) + "\n")
        for r in examples[:n_example]:
            f.write(json.dumps(asdict(r)) + "\n")
    log.info(f"[template] wrote {path} with {min(n_example, len(examples))} examples")


def to_mdoe_batch(records: List[T2Record], seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Convert usable records into the arrays MDOE's train-curated expects.
    Splits 70/15/15 by record (NOT by gene here — T2 has no gene split).
    """
    recs = [r for r in records if r.is_usable()]
    if len(recs) < 8:
        raise ValueError(f"only {len(recs)} usable records; need >= 8")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))
    a = int(0.7 * len(recs)); b = int(0.85 * len(recs))
    split = {"train": idx[:a], "val": idx[a:b], "test": idx[b:] if b < len(recs) else idx[a:]}
    comp = np.array([r.composition for r in recs], np.float32)
    meta = {
        "factor_idx": np.array([FACTORS.index(r.factor) for r in recs], np.int64),
        "dose_nM": np.array([r.dose_nM for r in recs], np.float32),
        "time_h": np.array([r.time_h for r in recs], np.float32),
    }
    return {"split": split, "composition": comp, "meta": meta, "fates": FATES}


def selftest(args):
    print("=" * 72)
    print("  T2 CURATION TEMPLATE — self-test")
    print("=" * 72)
    ok = True

    print("\n[T1] template generation")
    p = Path("_t2_template_test.jsonl")
    write_template(p)
    recs = load_jsonl(p)
    t = len(recs) == 2
    print(f"   wrote {len(recs)} example records  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T2] unverified examples are flagged, not usable")
    for r in recs:
        probs = r.problems()
        t = not r.is_usable() and any("not verified" in x for x in probs)
        print(f"   {r.record_id}: usable={r.is_usable()} "
              f"problems={len(probs)}  {'PASS' if t else 'FAIL'}")
        ok &= t

    print("\n[T3] a corrected record becomes usable")
    r = recs[0]
    r.verified = True
    t = r.is_usable()
    print(f"   after verified=True: usable={r.is_usable()}  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T4] bad composition is caught")
    bad = T2Record("bad", "CHIR", 100, 24, [0.5, 0.5] + [0.0]*(N_FATE-2),
                   "GSE1", "GSM1", "10.1/x", verified=True)
    t = not bad.is_usable()
    print(f"   sums to {sum(bad.composition)}: usable={bad.is_usable()}  "
          f"{'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T5] validator summary")
    rep = validate_all(recs)
    print(f"   {rep}")
    t = rep["total"] == 2
    ok &= t

    p.unlink(missing_ok=True)
    print("\n" + "=" * 72)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("=" * 72)
    print("""
  WORKFLOW TO BUILD THE REAL DATASET
  1. python t2_curation_template.py template --out data/t2.jsonl
  2. Run scrna_to_composition.py on each GEO series to get compositions.
  3. Manually fill: factor, dose_nM, time_h, doi, verified=true.
     (Read the paper. Do not guess.)
  4. python t2_curation_template.py validate --data data/t2.jsonl
  5. python t2_curation_template.py to-mdoe --data data/t2.jsonl
  6. Feed the MDOE batch into the ablation with >= 3 seeds.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "template", "validate", "to-mdoe"])
    p.add_argument("--out", default="data/t2.jsonl")
    p.add_argument("--data", default="data/t2.jsonl")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "template":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        write_template(Path(a.out))
    elif a.cmd == "validate":
        recs = load_jsonl(Path(a.data))
        print(json.dumps(validate_all(recs), indent=2))
    else:
        recs = load_jsonl(Path(a.data))
        batch = to_mdoe_batch(recs, a.seed)
        out = Path(a.data).with_suffix(".mdoe.npz")
        np.savez_compressed(out,
                            composition=batch["composition"],
                            factor_idx=batch["meta"]["factor_idx"],
                            dose_nM=batch["meta"]["dose_nM"],
                            time_h=batch["meta"]["time_h"],
                            train_idx=batch["split"]["train"],
                            val_idx=batch["split"]["val"],
                            test_idx=batch["split"]["test"])
        print(f"saved → {out}")
        print(f"usable records: {sum(1 for r in recs if r.is_usable())}")
        print(f"fates: {batch['fates']}")


if __name__ == "__main__":
    main()
