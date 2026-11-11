#!/usr/bin/env python3
"""
=============================================================================
REAL MSA: MMseqs2 + UniRef / ColabFold API  →  Evoformer input
=============================================================================
License: MIT | 2026

LITERATURE
 [M1] Steinegger & Söding. "MMseqs2 enables sensitive protein sequence
      searching for the analysis of massive data sets."
      Nat. Biotechnol. 35, 1026 (2017).
 [M2] Mirdita et al. "ColabFold: making protein folding accessible to all."
      Nat. Methods 19, 679 (2022).
 [M3] Suzek et al. "UniRef clusters: a comprehensive and scalable alternative
      for improving sequence similarity searches."
      Bioinformatics 31, 926 (2015).
 [M4] Steinegger & Söding. "Clustering huge protein sequence sets in linear
      time." Nat. Commun. 9, 2542 (2018).
 [A1] Jumper et al. Nature 596, 583 (2021).  [Evoformer, MSA row/col attention]

=============================================================================
WHY THIS MATTERS
=============================================================================
Without a real MSA the Evoformer's row/column attention has depth 1 and
degenerates to a pair transformer. Coevolution — the dominant signal behind
AF2's accuracy — is ABSENT. This file supplies the real thing.

HONEST SCOPE
  ✅ Real MMseqs2 search against real UniRef/ColabFold databases.
  ✅ Neff (effective sequence count) computed the standard way.
  ❌ A deep MSA does NOT guarantee accurate structure. It is necessary,
     not sufficient. Orphan sequences have no homologs to find.
  ❌ ColabFold public API is rate-limited and shared; be considerate.
  ❌ Database download is tens to hundreds of GB. Plan disk and bandwidth.
  ❌ This does not make the model AF2. Templates, recycling schedule,
     distillation data and training scale are still missing.
=============================================================================
"""
from __future__ import annotations
import argparse, io, json, logging, os, shutil, subprocess, tarfile, time
import urllib.request, urllib.parse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("msa")

AA = "ARNDCQEGHILKMFPSTWYV"
AA_IDX = {a: i for i, a in enumerate(AA)}
GAP_IDX, UNK_IDX = 20, 21
COLABFOLD_API = "https://api.colabfold.com"


@dataclass
class MSAConfig:
    max_depth: int = 512
    max_len: int = 512
    identity_weight_threshold: float = 0.8   # for Neff
    min_coverage: float = 0.5
    mmseqs_bin: str = "mmseqs"
    sensitivity: float = 7.5
    num_iterations: int = 3
    max_seqs: int = 10000
    e_value: float = 1e-3
    colabfold_mode: str = "env"              # env | uniref
    poll_seconds: float = 15.0
    poll_timeout: float = 1800.0


# =============================================================================
# 1. a3m PARSING
# =============================================================================

def parse_a3m(text: str, cfg: MSAConfig) -> Tuple[np.ndarray, List[str]]:
    """
    a3m: lowercase = insertion relative to query; '-' = gap; '.' = gap.
    Insertions are stripped so all rows share the query length.
    """
    seqs, names = [], []
    name = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:]
            continue
        if line.startswith("#"):
            continue
        s = "".join(c for c in line if not c.islower())
        s = s.replace(".", "-")
        if name is not None:
            seqs.append(s); names.append(name); name = None
    if not seqs:
        return np.zeros((0, 0), np.int64), []
    L = len(seqs[0])
    keep = [(s, nm) for s, nm in zip(seqs, names) if len(s) == L]
    if not keep:
        log.warning("[a3m] no rows match query length")
        return np.zeros((0, 0), np.int64), []
    L = min(L, cfg.max_len)
    out = np.full((min(len(keep), cfg.max_depth), L), GAP_IDX, np.int64)
    for i, (s, _) in enumerate(keep[:cfg.max_depth]):
        for j, c in enumerate(s[:L]):
            out[i, j] = AA_IDX.get(c.upper(), UNK_IDX if c != "-" else GAP_IDX)
    return out, [nm for _, nm in keep[:cfg.max_depth]]


# =============================================================================
# 2. MSA QUALITY METRICS  (Neff, coverage, conservation)
# =============================================================================

def msa_metrics(msa: np.ndarray, cfg: MSAConfig) -> Dict:
    """
    Neff = Σ_i w_i,  w_i = 1/|{j : identity(i,j) > threshold}|
    Standard sequence weighting to discount redundant homologs.
    """
    S, L = msa.shape
    if S == 0 or L == 0:
        return {"depth": 0, "length": 0, "neff": 0.0, "neff_per_sqrtL": 0.0,
                "coverage": 0.0, "mean_conservation": 0.0}
    valid = (msa < 20)
    same = (msa[:, None, :] == msa[None, :, :]) & valid[:, None, :] & valid[None, :, :]
    n_valid_i = valid.sum(1)
    ident = same.sum(-1) / np.maximum(
        np.minimum(n_valid_i[:, None], n_valid_i[None, :]), 1)
    np.fill_diagonal(ident, 1.0)
    w = 1.0 / np.maximum((ident > cfg.identity_weight_threshold).sum(1), 1)
    neff = float(w.sum())
    cov = float(valid.mean())
    # per-column conservation = 1 − normalized entropy over 20 AA
    cons = []
    for j in range(L):
        col = msa[:, j]
        col = col[col < 20]
        if len(col) == 0:
            cons.append(0.0); continue
        cnt = np.bincount(col, minlength=20).astype(float)
        p = cnt / cnt.sum(); p = p[p > 0]
        H = -(p*np.log(p)).sum() / np.log(20)
        cons.append(1.0 - H)
    return {
        "depth": int(S), "length": int(L),
        "neff": neff,
        "neff_per_sqrtL": neff / max(np.sqrt(L), 1.0),
        "coverage": cov,
        "mean_conservation": float(np.mean(cons)),
        "conservation": np.array(cons).tolist(),
        " Evoformer_meaningful": neff >= 10.0,
    }


# =============================================================================
# 3. LOCAL MMseqs2
# =============================================================================

def mmseqs_available(cfg: MSAConfig) -> bool:
    return shutil.which(cfg.mmseqs_bin) is not None


def mmseqs_db_instructions():
    print("""
  LOCAL MMseqs2 SETUP
  -------------------
  # install
  conda install -c conda-forge -c bioconda mmseqs2
  # or: https://github.com/soedinglab/MMseqs2/releases

  # download + prepare a database (LARGE — plan disk)
  mmseqs databases UniRef30  data/db/uniref30  tmp --threads 8
  # alternatives: UniRef50, UniRef100, ColabFoldDB, PDB70

  # then run:
  python msa_pipeline.py mmseqs --seq <SEQUENCE> \\
      --db data/db/uniref30 --out data/msa

  Disk/bandwidth: UniRef30 is tens of GB; ColabFoldDB is larger.
  Verify the current database names with `mmseqs databases --help`.
""")


def mmseqs_search(sequence: str, db: str, out_dir: str,
                  cfg: MSAConfig) -> Optional[np.ndarray]:
    """
    Real MMseqs2 search → result2msa → a3m.
    Uses the documented easy-search + result2msa path.
    """
    if not mmseqs_available(cfg):
        log.error("mmseqs not found"); mmseqs_db_instructions(); return None
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    tmp = d/"tmp"; tmp.mkdir(exist_ok=True)
    qfa = d/"query.fasta"
    qfa.write_text(f">query\n{sequence}\n")
    # create query DB and search
    cmds = [
        [cfg.mmseqs_bin, "createdb", str(qfa), str(d/"qdb")],
        [cfg.mmseqs_bin, "search", str(d/"qdb"), db, str(d/"res"), str(tmp),
         "--num-iterations", str(cfg.num_iterations),
         "-s", str(cfg.sensitivity),
         "--max-seqs", str(cfg.max_seqs),
         "-e", str(cfg.e_value), "--threads", "4"],
        [cfg.mmseqs_bin, "result2msa", str(d/"qdb"), db, str(d/"res"),
         str(d/"msaDB"), "--msa-format-mode", "6", "--threads", "4"],
        [cfg.mmseqs_bin, "convert2fasta", str(d/"msaDB"), str(d/"out.a3m")],
    ]
    for c in cmds:
        log.info("[mmseqs] " + " ".join(c[:3]) + " ...")
        p = subprocess.run(c, capture_output=True, text=True)
        if p.returncode != 0:
            log.error(f"[mmseqs] failed: {p.stderr[:500]}")
            log.info("If result2msa options differ in your version, run "
                     "`mmseqs result2msa --help` and adjust.")
            return None
    a3m = d/"out.a3m"
    if not a3m.exists():
        log.error("[mmseqs] no a3m produced"); return None
    msa, names = parse_a3m(a3m.read_text(), cfg)
    log.info(f"[mmseqs] {msa.shape[0]} sequences")
    return msa


# =============================================================================
# 4. ColabFold PUBLIC API
# =============================================================================

def colabfold_msa(sequence: str, out_dir: str,
                  cfg: MSAConfig) -> Optional[np.ndarray]:
    """
    ColabFold MMseqs2 server [M2]. Shared, rate-limited — be considerate.
    Flow: POST /ticket/msa → poll /ticket/{id} → GET /result/download/{id}
    """
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    cache = d / f"cf_{abs(hash(sequence))}.a3m"
    if cache.exists():
        log.info("[colabfold] cache hit")
        return parse_a3m(cache.read_text(), cfg)[0]
    body = urllib.parse.urlencode(
        {"q": sequence, "mode": cfg.colabfold_mode}).encode()
    try:
        req = urllib.request.Request(f"{COLABFOLD_API}/ticket/msa", data=body,
                                     headers={"Content-Type":
                                              "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=60) as r:
            ticket = json.loads(r.read().decode())
    except Exception as e:
        log.error(f"[colabfold] submit failed: {e}"); return None
    tid = ticket.get("id")
    if not tid:
        log.error(f"[colabfold] no ticket id: {ticket}"); return None
    log.info(f"[colabfold] ticket {tid}; polling every {cfg.poll_seconds}s")
    t0 = time.time()
    while time.time() - t0 < cfg.poll_timeout:
        time.sleep(cfg.poll_seconds)
        try:
            with urllib.request.urlopen(f"{COLABFOLD_API}/ticket/{tid}",
                                        timeout=60) as r:
                st = json.loads(r.read().decode())
        except Exception as e:
            log.warning(f"[colabfold] poll error: {e}"); continue
        status = st.get("status", "UNKNOWN")
        log.info(f"[colabfold] {status} ({time.time()-t0:.0f}s)")
        if status == "COMPLETE":
            break
        if status in ("ERROR", "FAILED"):
            log.error("[colabfold] job failed"); return None
    else:
        log.error("[colabfold] timeout"); return None
    try:
        with urllib.request.urlopen(f"{COLABFOLD_API}/result/download/{tid}",
                                    timeout=300) as r:
            blob = r.read()
    except Exception as e:
        log.error(f"[colabfold] download failed: {e}"); return None
    a3m_text = None
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.name.endswith(".a3m") or "uniref" in m.name:
                    a3m_text = tf.extractfile(m).read().decode("utf-8", "ignore")
                    break
    except Exception as e:
        log.error(f"[colabfold] untar failed: {e}"); return None
    if a3m_text is None:
        log.error("[colabfold] no a3m in archive"); return None
    cache.write_text(a3m_text)
    return parse_a3m(a3m_text, cfg)[0]


# =============================================================================
# 5. SELF-TEST
# =============================================================================

def selftest(args):
    print("="*76); print("  MSA PIPELINE — self-test"); print("="*76)
    cfg = MSAConfig(max_depth=64, max_len=64)
    ok = True

    print("\n[T1] a3m insertion stripping")
    txt = ">q\nACDEFGHIK\n>h1\nACDEFGHIK\n>h2\naACDEFGHiIK\n>h3\nACD-FGHIK\n"
    msa, names = parse_a3m(txt, cfg)
    t = msa.shape == (4, 9) and msa[2].tolist() == msa[0].tolist()
    print(f"   shape={msa.shape}  insertions stripped={t}")
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T2] gap token handling")
    t = msa[3, 3] == GAP_IDX
    print(f"   gap index={msa[3,3]} (expect {GAP_IDX})  {'PASS' if t else 'FAIL'}")
    ok &= t

    print("\n[T3] Neff discounts redundant sequences")
    identical = np.tile(msa[0], (20, 1))
    m_id = msa_metrics(identical, cfg)
    diverse = np.random.default_rng(0).integers(0, 20, (20, 9))
    m_dv = msa_metrics(diverse, cfg)
    print(f"   20 identical: depth={m_id['depth']} Neff={m_id['neff']:.2f}")
    print(f"   20 random:    depth={m_dv['depth']} Neff={m_dv['neff']:.2f}")
    t = m_id["neff"] < 2.0 and m_dv["neff"] > 10.0
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T4] depth-1 warns Evoformer is degenerate")
    m1 = msa_metrics(msa[:1], cfg)
    t = not m1[" Evoformer_meaningful"]
    print(f"   Neff={m1['neff']:.2f}  meaningful={m1[' Evoformer_meaningful']}")
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T5] conservation on fully conserved column")
    mc = msa_metrics(identical, cfg)
    t = mc["mean_conservation"] > 0.99
    print(f"   conservation={mc['mean_conservation']:.4f} (expect ≈1)")
    ok &= t; print(f"   {'PASS' if t else 'FAIL'}")

    print("\n[T6] tooling availability")
    print(f"   mmseqs2 installed: {mmseqs_available(cfg)}")
    if not mmseqs_available(cfg):
        mmseqs_db_instructions()

    print("\n" + "="*76)
    print(f"  {'✅ ALL TESTS PASSED' if ok else '❌ SOME FAILED'}")
    print("="*76)
    print("""
  A real MSA is NECESSARY for the Evoformer to do what AF2 does.
  It is NOT SUFFICIENT: templates, training scale, recycling schedule and
  distillation data are still required. Orphans have no homologs to find.
""")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["selftest", "mmseqs", "colabfold", "setup"])
    p.add_argument("--seq", default=None)
    p.add_argument("--fasta", default=None)
    p.add_argument("--db", default="data/db/uniref30")
    p.add_argument("--out", default="data/msa")
    p.add_argument("--mode", default="env")
    a = p.parse_args()
    cfg = MSAConfig(colabfold_mode=a.mode)
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest(a) else 1)
    if a.cmd == "setup":
        mmseqs_db_instructions(); return
    seq = a.seq
    if a.fasta:
        seq = "".join(l.strip() for l in Path(a.fasta).read_text().splitlines()
                      if not l.startswith(">"))
    if not seq:
        log.error("provide --seq or --fasta"); return
    msa = (mmseqs_search(seq, a.db, a.out, cfg) if a.cmd == "mmseqs"
           else colabfold_msa(seq, a.out, cfg))
    if msa is None:
        raise SystemExit(1)
    m = msa_metrics(msa, cfg)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.save(Path(a.out)/"msa.npy", msa)
    json.dump({k: v for k, v in m.items() if k != "conservation"},
              open(Path(a.out)/"metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "conservation"}, indent=2))


if __name__ == "__main__":
    main()
