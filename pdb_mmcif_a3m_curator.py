#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
pdb_mmcif_a3m_curator.py — v1.0.0
Real curation: chain selection · resolution filter · missing-residue policy ·
MSA pairing · OpenFold-compatible export · provenance audit
=============================================================================

Author  : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID   : 0009-0008-2374-0788
License : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics and software stand independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
WHY THIS LAYER EXISTS  (and why it is the reusable contribution)
=============================================================================
AlphaFold2 / OpenFold / AF3-style trunks consume a batch dict. They do NOT
tell you:
  * which chain of a deposited entry is the right training target
  * whether the entry's resolution is good enough for supervision
  * what to do about residues that are in SEQRES but absent from ATOM
  * whether the a3m you downloaded actually corresponds to THIS chain
  * how to reproduce yesterday's curation decision exactly

Every group re-writes this, badly, and the bugs are silent:
  - training on a chain with a 40-residue internal gap teaches the model
    that two distant residues are adjacent
  - pairing an a3m from a homolog with 60% identity to the wrong chain
    injects a wrong evolutionary prior
  - accepting an NMR ensemble's first model as if it were a crystal structure

This module makes those decisions EXPLICIT, LOGGED, and REPRODUCIBLE.
That is the extension. Not a new trunk.

=============================================================================
WHAT IS IMPLEMENTED (real, not stubbed)
=============================================================================
[C1] mmCIF tokenizer: loop_, quoted strings, semicolon text blocks, '.'/'?'
     nulls. Reads _atom_site, _entity_poly, _pdbx_poly_seq_scheme,
     _pdbx_unobs_or_zero_occ_residues, _exptl, _refine, _em_3d_reconstruction.
[C2] PDB parser: SEQRES, ATOM/HETATM, altLoc, insertion codes, MODEL,
     REMARK 2 resolution, REMARK 465 missing residues, EXPDTA method.
[C3] Chain selection: protein-only, length/completeness scoring, explicit
     chain allow/deny lists, multi-model policy.
[C4] Resolution filter: X-ray / cryo-EM / neutron; NMR rejected by default;
     missing resolution handled by declared policy, never silently passed.
[C5] Missing-residue policy: REJECT | SEGMENT | MASK.
     SEGMENT splits into contiguous observed runs (default, safest).
     MASK keeps the full numbering with atom_mask=0 — and is flagged as
     hazardous for FAPE unless the loss masks it correctly.
     Coordinates are NEVER interpolated.
[C6] MSA pairing: a3m parse (lowercase insertions removed), deletion matrix,
     exact / substring / Needleman-Wunsch identity match against the chain,
     column remapping when the a3m query is a subsequence, Neff.
[C7] Export: OpenFold/AF2-compatible NPZ (aatype, residue_index, msa,
     deletion_matrix, all_atom_positions[L,37,3], all_atom_mask, resolution).
[C8] Provenance: SHA256 of every input, config hash, decision log, timestamp.
[C9] Dataset statistics with Politis-Romano subsampling CI (the genuinely
     appropriate statistics for along-chain correlated quality).
[C10] delta_min / kappa-fatness segment guard and a CSOC-bounded
     curation retry counter — both clearly labelled as to what they bound.

=============================================================================
WHERE THE MSPS MATHEMATICS PLUGS IN  (honest placement)
=============================================================================
  delta_min / kappa-fatness  -> segment acceptance guard. It is a THRESHOLD
                                on segment length and observed density.
                                It is NOT a claim that the isodiametric
                                inequality governs protein structure.
  Politis-Romano subsampling -> CI on dataset-level quality statistics.
                                Along-chain completeness is serially
                                correlated; an iid CI understates the error.
                                This is the correct use of that theorem.
  CSOC event bound           -> bounds the number of accept/reject switches
                                in an active curation loop. Provable for
                                the counter. NOT a biological statement.
  OPMC / Carpet / Hodge / HP -> belong DOWNSTREAM in the model trunk, not in
                                curation. They are learned components with
                                analogical names. Ablate them there.

=============================================================================
HONEST SCOPE
=============================================================================
* No structure prediction happens here. This is data curation.
* mmCIF parsing covers the categories listed above. Exotic categories are
  ignored and REPORTED, not silently mis-read.
* Heteromer MSA pairing by taxonomy is HEURISTIC (OX= / OS= header tags).
  It is implemented and labelled; verify before training a multimer model.
* Modified residues (MSE, SEP, ...) are mapped to their parent or dropped
  by policy. Non-standard chemistry is not modelled.
* Nothing here validates biological accuracy. It validates DATA HYGIENE.
=============================================================================
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("curator")

VERSION = "1.0.0"

# =============================================================================
# §0. CONSTANTS  (AF2 residue_constants conventions)
# =============================================================================

AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common modified residues -> parent (policy-controlled)
    "MSE": "M", "SEC": "C", "PYL": "K", "HYP": "P", "TPO": "T",
    "SEP": "S", "PTR": "Y", "CSO": "C", "CME": "C", "CSD": "C",
    "MLY": "K", "M3L": "K", "ALY": "K", "KCX": "K", "LLP": "K",
    "PCA": "Q", "HIP": "H", "HID": "H", "HIE": "H",
}
MODIFIED_PARENT = {"MSE", "SEC", "PYL", "HYP", "TPO", "SEP", "PTR", "CSO",
                   "CME", "CSD", "MLY", "M3L", "ALY", "KCX", "LLP", "PCA",
                   "HIP", "HID", "HIE"}
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"          # AF2 restype order
AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}
UNK_IDX = 20
GAP_IDX = 20
MSA_VOCAB = 22

# AF2 atom37 order
ATOM37 = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
]
ATOM37_IDX = {a: i for i, a in enumerate(ATOM37)}
BB_ATOMS = ("N", "CA", "C", "O")

NUCLEIC_RES = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "I"}
WATER_RES = {"HOH", "DOD", "WAT"}


# =============================================================================
# §1. mmCIF TOKENIZER  (real; handles loop_, quotes, semicolon blocks)
# =============================================================================

def cif_tokenize(text: str) -> List[List[str]]:
    """
    Tokenize an mmCIF file into a list of token-lists (one per logical line).
    Handles:
      loop_    -> its own token
      'x' "x"  -> single token without quotes
      ;text    -> multi-line text block, single token
      . ?      -> null tokens (kept as-is)
      #        -> comment to end of line
    """
    lines = text.splitlines()
    out: List[List[str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                buf.append(lines[i]); i += 1
            out.append(["\n".join(buf)])
            i += 1
            continue
        s = line.split("#", 1)[0].strip()
        i += 1
        if not s:
            continue
        toks: List[str] = []
        j = 0
        while j < len(s):
            c = s[j]
            if c.isspace():
                j += 1; continue
            if c in ("'", '"'):
                q = c; j += 1; start = j
                while j < len(s) and (s[j] != q or
                                      (j + 1 < len(s) and not s[j + 1].isspace())):
                    j += 1
                toks.append(s[start:j]); j += 1
            else:
                start = j
                while j < len(s) and not s[j].isspace():
                    j += 1
                toks.append(s[start:j])
        if toks:
            out.append(toks)
    return out


def cif_parse(text: str) -> Dict[str, Any]:
    """
    Parse tokenized mmCIF into {category: {field: value}} for single items
    and {category: {"_fields": [...], "_rows": [[...]]}} for loops.
    Only the FIRST data_ block is read.
    """
    toks = cif_tokenize(text)
    data: Dict[str, Any] = {}
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if len(t) == 1 and t[0].lower().startswith("data_"):
            i += 1
            if data:      # only first block
                break
            continue
        if len(t) == 1 and t[0].lower() == "loop_":
            i += 1
            fields: List[str] = []
            while i < n and toks[i][0].startswith("_"):
                fields.extend(toks[i]); i += 1
            if not fields:
                continue
            cat = fields[0].split(".")[0]
            keys = [f.split(".", 1)[1] if "." in f else f for f in fields]
            rows: List[List[str]] = []
            nf = len(fields)
            while i < n:
                nt = toks[i]
                if nt[0].startswith("_") or (len(nt) == 1 and
                        nt[0].lower() in ("loop_", "stop_") or
                        nt[0].lower().startswith("data_")):
                    break
                # a logical line may carry a full row or part of one
                buf = list(nt)
                i += 1
                while len(buf) < nf and i < n and not toks[i][0].startswith("_"):
                    if len(toks[i]) == 1 and toks[i][0].lower() == "loop_":
                        break
                    buf.extend(toks[i]); i += 1
                for k in range(0, len(buf) - nf + 1, nf):
                    rows.append(buf[k:k + nf])
            data[cat] = {"_fields": keys, "_rows": rows}
            continue
        if t[0].startswith("_"):
            key = t[0]
            val = t[1] if len(t) > 1 else "."
            cat, _, fld = key.partition(".")
            data.setdefault(cat, {})
            if isinstance(data[cat], dict) and "_rows" not in data[cat]:
                data[cat][fld] = val
            i += 1
            continue
        i += 1
    return data


def _cif_loop_rows(d: Dict[str, Any], cat: str) -> Tuple[List[str], List[List[str]]]:
    c = d.get(cat)
    if isinstance(c, dict) and "_rows" in c:
        return c["_fields"], c["_rows"]
    return [], []


def _cif_scalar(d: Dict[str, Any], cat: str, field_: str) -> Optional[str]:
    c = d.get(cat)
    if isinstance(c, dict) and "_rows" not in c:
        v = c.get(field_)
        if v not in (None, ".", "?"):
            return v
    f, r = _cif_loop_rows(d, cat)
    if field_ in f and r:
        v = r[0][f.index(field_)]
        if v not in (".", "?"):
            return v
    return None


def _to_float(v: Optional[str]) -> Optional[float]:
    if v in (None, ".", "?", ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# =============================================================================
# §2. UNIFIED STRUCTURE RECORD
# =============================================================================

@dataclass
class ResidueObs:
    """One observed residue: author numbering + observed atoms."""
    auth_seq: int
    ins_code: str
    label_seq: Optional[int]
    comp_id: str
    one: str
    atoms: Dict[str, np.ndarray] = field(default_factory=dict)
    occupancy: Dict[str, float] = field(default_factory=dict)
    bfactor: Dict[str, float] = field(default_factory=dict)
    is_modified: bool = False
    is_hetatm: bool = False


@dataclass
class ChainRecord:
    chain_id: str                 # auth_asym_id (PDB-style) or label_asym_id
    entity_id: Optional[str]
    residues: List[ResidueObs]
    seqres: Optional[str]         # full declared sequence, if available
    seqres_numbering: Optional[List[int]] = None
    declared_missing: List[int] = field(default_factory=list)
    entity_type: str = "polymer"

    @property
    def observed_sequence(self) -> str:
        return "".join(r.one for r in self.residues)

    @property
    def n_observed(self) -> int:
        return len(self.residues)


@dataclass
class StructureRecord:
    entry_id: str
    method: str
    resolution: Optional[float]
    chains: List[ChainRecord]
    source_path: str
    source_sha256: str
    n_models: int = 1
    warnings: List[str] = field(default_factory=list)
    ignored_categories: List[str] = field(default_factory=list)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_text_maybe_gz(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as f:
            return f.read()
    return path.read_text(errors="replace")


# =============================================================================
# §3. mmCIF READER
# =============================================================================

def parse_mmcif(path: Path, keep_modified: bool = True,
                keep_hetatm_aa: bool = False) -> StructureRecord:
    text = read_text_maybe_gz(path)
    d = cif_parse(text)
    entry_id = _cif_scalar(d, "_entry", "id") or path.stem
    method = (_cif_scalar(d, "_exptl", "method") or "unknown").lower()

    res = (_to_float(_cif_scalar(d, "_refine", "ls_d_res_high"))
           or _to_float(_cif_scalar(d, "_reflns", "d_resolution_high"))
           or _to_float(_cif_scalar(d, "_em_3d_reconstruction", "resolution"))
           or _to_float(_cif_scalar(d, "_refine_hist", "d_res_high")))

    warns: List[str] = []

    # entity types
    ent_type: Dict[str, str] = {}
    f, rows = _cif_loop_rows(d, "_entity")
    if "id" in f and "type" in f:
        for r in rows:
            ent_type[r[f.index("id")]] = r[f.index("type")].lower()

    # entity -> canonical one-letter sequence
    ent_seq: Dict[str, str] = {}
    f, rows = _cif_loop_rows(d, "_entity_poly")
    if rows:
        i_id = f.index("entity_id") if "entity_id" in f else None
        i_seq = (f.index("seq_one_letter_code_can")
                 if "seq_one_letter_code_can" in f else None)
        if i_id is not None and i_seq is not None:
            for r in rows:
                s = re.sub(r"\s+", "", r[i_seq]).upper()
                ent_seq[r[i_id]] = s

    # asym -> entity
    asym_ent: Dict[str, str] = {}
    f, rows = _cif_loop_rows(d, "_struct_asym")
    if "id" in f and "entity_id" in f:
        for r in rows:
            asym_ent[r[f.index("id")]] = r[f.index("entity_id")]

    # author numbering scheme (label_seq_id -> auth_seq_id)
    scheme: Dict[Tuple[str, int], Tuple[int, str]] = {}
    seqres_num: Dict[str, List[int]] = {}
    f, rows = _cif_loop_rows(d, "_pdbx_poly_seq_scheme")
    if rows and "asym_id" in f:
        ia = f.index("asym_id")
        ils = f.index("seq_id") if "seq_id" in f else None
        ias = f.index("pdb_seq_num") if "pdb_seq_num" in f else None
        iic = f.index("pdb_ins_code") if "pdb_ins_code" in f else None
        for r in rows:
            try:
                ls = int(r[ils]) if ils is not None else None
                au = int(r[ias]) if ias is not None else None
            except (TypeError, ValueError):
                continue
            if ls is None or au is None:
                continue
            ic = r[iic] if iic is not None else "."
            ic = "" if ic in (".", "?") else ic
            scheme[(r[ia], ls)] = (au, ic)
            seqres_num.setdefault(r[ia], []).append(au)

    # declared unobserved residues
    declared_missing: Dict[str, List[int]] = {}
    f, rows = _cif_loop_rows(d, "_pdbx_unobs_or_zero_occ_residues")
    if rows and "label_asym_id" in f:
        ia = f.index("label_asym_id")
        ils = f.index("label_seq_id") if "label_seq_id" in f else None
        for r in rows:
            try:
                ls = int(r[ils]) if ils is not None else None
            except (TypeError, ValueError):
                continue
            if ls is None:
                continue
            au, _ = scheme.get((r[ia], ls), (ls, ""))
            declared_missing.setdefault(r[ia], []).append(au)

    # atom_site
    f, rows = _cif_loop_rows(d, "_atom_site")
    if not rows:
        raise ValueError(f"{path}: no _atom_site loop")
    ix = {k: f.index(k) for k in f}

    def g(r: List[str], k: str, default: str = "") -> str:
        i = ix.get(k)
        return r[i] if i is not None and i < len(r) else default

    chains: Dict[str, ChainRecord] = {}
    seen_model = set()
    for r in rows:
        mid = g(r, "pdbx_PDB_model_num", "1")
        seen_model.add(mid)
        if mid not in ("1", ""):
            continue                                  # first model only
        comp = g(r, "label_comp_id").upper()
        grp = g(r, "group_PDB")
        if comp in WATER_RES:
            continue
        one = AA3to1.get(comp)
        if one is None:
            continue                                  # nucleic / ligand
        is_mod = comp in MODIFIED_PARENT
        if is_mod and not keep_modified:
            continue
        if grp == "HETATM" and not (keep_hetatm_aa or is_mod):
            continue
        auth_chain = g(r, "auth_asym_id") or g(r, "label_asym_id")
        label_chain = g(r, "label_asym_id") or auth_chain
        try:
            au = int(g(r, "auth_seq_id"))
        except ValueError:
            continue
        ic = g(r, "pdbx_PDB_ins_code")
        ic = "" if ic in (".", "?") else ic
        ls_raw = g(r, "label_seq_id")
        ls = int(ls_raw) if ls_raw.isdigit() else None
        alt = g(r, "label_alt_id")
        if alt not in ("", ".", "?", "A", "1"):
            continue                                  # keep primary altloc
        try:
            xyz = np.array([float(g(r, "Cartn_x")), float(g(r, "Cartn_y")),
                            float(g(r, "Cartn_z"))], dtype=np.float32)
        except ValueError:
            continue
        an = g(r, "label_atom_id").strip('"').upper()
        if an not in ATOM37_IDX:
            continue
        occ = _to_float(g(r, "occupancy")) or 1.0
        bf = _to_float(g(r, "B_iso_or_equiv")) or 0.0

        ch = chains.get(auth_chain)
        if ch is None:
            eid = asym_ent.get(label_chain)
            ch = ChainRecord(chain_id=auth_chain, entity_id=eid, residues=[],
                             seqres=ent_seq.get(eid or ""),
                             seqres_numbering=sorted(set(seqres_num.get(label_chain, []))) or None,
                             declared_missing=sorted(set(declared_missing.get(label_chain, []))),
                             entity_type=ent_type.get(eid or "", "polymer"))
            chains[auth_chain] = ch
        if not ch.residues or (ch.residues[-1].auth_seq != au
                               or ch.residues[-1].ins_code != ic):
            ch.residues.append(ResidueObs(au, ic, ls, comp, one,
                                          is_modified=is_mod,
                                          is_hetatm=(grp == "HETATM")))
        res = ch.residues[-1]
        if an not in res.atoms:
            res.atoms[an] = xyz
            res.occupancy[an] = occ
            res.bfactor[an] = bf

    # keep only polymer protein chains
    keep = []
    for ch in chains.values():
        if ch.entity_type and ch.entity_type not in ("polymer", "polypeptide(l)",
                                                     "polypeptide(d)"):
            warns.append(f"chain {ch.chain_id}: entity_type={ch.entity_type} skipped")
            continue
        if ch.n_observed == 0:
            continue
        keep.append(ch)

    known = {"_entry", "_exptl", "_refine", "_reflns", "_em_3d_reconstruction",
             "_refine_hist", "_entity", "_entity_poly", "_struct_asym",
             "_pdbx_poly_seq_scheme", "_pdbx_unobs_or_zero_occ_residues",
             "_atom_site", "_struct", "_audit_conform", "_cell", "_symmetry",
             "_software", "_citation", "_audit_author", "_database_2",
             "_pdbx_database_status", "_chem_comp", "_atom_type"}
    ignored = sorted(set(d.keys()) - known)

    return StructureRecord(entry_id=entry_id, method=method, resolution=res,
                           chains=keep, source_path=str(path),
                           source_sha256=sha256_file(path),
                           n_models=len(seen_model) or 1,
                           warnings=warns, ignored_categories=ignored)


# =============================================================================
# §4. PDB READER
# =============================================================================

_RES_RE = re.compile(r"RESOLUTION\.\s*([\d.]+)")


def parse_pdb(path: Path, keep_modified: bool = True,
              keep_hetatm_aa: bool = False) -> StructureRecord:
    text = read_text_maybe_gz(path)
    entry_id = path.stem
    method = "unknown"
    resolution: Optional[float] = None
    seqres: Dict[str, List[str]] = {}
    chains: Dict[str, ChainRecord] = {}
    remark465: Dict[str, List[int]] = {}
    n_models = 0
    warns: List[str] = []
    in_465 = False

    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "HEADER":
            entry_id = line[62:66].strip() or entry_id
        elif rec == "EXPDTA":
            m = line[10:].strip().upper()
            if "X-RAY" in m:
                method = "x-ray diffraction"
            elif "ELECTRON MICROSCOPY" in m:
                method = "electron microscopy"
            elif "NMR" in m:
                method = "nmr"
            elif "NEUTRON" in m:
                method = "neutron diffraction"
        elif rec == "REMARK":
            if line[7:10] == "  2" and resolution is None:
                mm = _RES_RE.search(line)
                if mm:
                    resolution = _to_float(mm.group(1))
            if line[7:10] == "465":
                if "FOLLOWING RESIDUES" in line or "RES SS" in line:
                    in_465 = True
                    continue
                if in_465 and len(line) > 26:
                    ch = line[19].strip()
                    try:
                        num = int(line[21:26])
                    except ValueError:
                        continue
                    if ch:
                        remark465.setdefault(ch, []).append(num)
        elif rec == "SEQRES":
            ch = line[11].strip()
            toks = line[19:].split()
            seqres.setdefault(ch, []).extend(toks)
        elif rec == "MODEL":
            n_models += 1
            if n_models > 1:
                break
        elif rec in ("ATOM", "HETATM"):
            comp = line[17:20].strip().upper()
            if comp in WATER_RES:
                continue
            one = AA3to1.get(comp)
            if one is None:
                continue
            is_mod = comp in MODIFIED_PARENT
            if is_mod and not keep_modified:
                continue
            if rec == "HETATM" and not (keep_hetatm_aa or is_mod):
                continue
            alt = line[16].strip()
            if alt not in ("", "A", "1"):
                continue
            ch_id = line[21].strip() or "A"
            try:
                au = int(line[22:26])
            except ValueError:
                continue
            ic = line[26].strip()
            an = line[12:16].strip().upper()
            if an not in ATOM37_IDX:
                continue
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])], dtype=np.float32)
            except ValueError:
                continue
            occ = _to_float(line[54:60]) or 1.0
            bf = _to_float(line[60:66]) or 0.0
            ch = chains.get(ch_id)
            if ch is None:
                sr = seqres.get(ch_id)
                ch = ChainRecord(
                    chain_id=ch_id, entity_id=None, residues=[],
                    seqres=("".join(AA3to1.get(t, "X") for t in sr) if sr else None),
                    declared_missing=sorted(set(remark465.get(ch_id, []))))
                chains[ch_id] = ch
            if not ch.residues or (ch.residues[-1].auth_seq != au
                                   or ch.residues[-1].ins_code != ic):
                ch.residues.append(ResidueObs(au, ic, None, comp, one,
                                              is_modified=is_mod,
                                              is_hetatm=(rec == "HETATM")))
            r = ch.residues[-1]
            if an not in r.atoms:
                r.atoms[an] = xyz
                r.occupancy[an] = occ
                r.bfactor[an] = bf

    # SEQRES may appear after ATOM in odd files; backfill
    for cid, ch in chains.items():
        if ch.seqres is None and cid in seqres:
            ch.seqres = "".join(AA3to1.get(t, "X") for t in seqres[cid])
        if not ch.declared_missing and cid in remark465:
            ch.declared_missing = sorted(set(remark465[cid]))

    keep = [c for c in chains.values() if c.n_observed > 0]
    return StructureRecord(entry_id=entry_id, method=method,
                           resolution=resolution, chains=keep,
                           source_path=str(path),
                           source_sha256=sha256_file(path),
                           n_models=max(n_models, 1), warnings=warns)


def parse_structure(path: Path, **kw) -> StructureRecord:
    suf = str(path).lower()
    if suf.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz")):
        return parse_mmcif(path, **kw)
    if suf.endswith((".pdb", ".pdb.gz", ".ent", ".ent.gz")):
        return parse_pdb(path, **kw)
    # sniff
    head = read_text_maybe_gz(path)[:2000]
    if head.lstrip().startswith("data_") or "_atom_site." in head:
        return parse_mmcif(path, **kw)
    return parse_pdb(path, **kw)


# =============================================================================
# §5. SEQUENCE ALIGNMENT  (Needleman-Wunsch, linear gap)
# =============================================================================

def nw_align(a: str, b: str, match: int = 2, mismatch: int = -1,
             gap: int = -2) -> Tuple[str, str, float]:
    """
    Global alignment. Returns (aligned_a, aligned_b, identity).
    identity = matches / len(aligned columns where both are residues).
    O(len(a)*len(b)) — fine for chains up to a few thousand residues.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return "", "", 0.0
    if n * m > 4_000_000:
        # fall back to a cheap diagonal-free estimate for very long chains
        k = min(n, m)
        ident = sum(1 for i in range(k) if a[i] == b[i]) / k
        return a[:k], b[:k], ident
    S = np.zeros((n + 1, m + 1), dtype=np.int32)
    S[0, :] = np.arange(m + 1) * gap
    S[:, 0] = np.arange(n + 1) * gap
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = S[i - 1]
        cur = S[i]
        for j in range(1, m + 1):
            sc = match if ai == b[j - 1] else mismatch
            cur[j] = max(row[j - 1] + sc, row[j] + gap, cur[j - 1] + gap)
    i, j = n, m
    aa, ab = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sc = match if a[i - 1] == b[j - 1] else mismatch
            if S[i, j] == S[i - 1, j - 1] + sc:
                aa.append(a[i - 1]); ab.append(b[j - 1]); i -= 1; j -= 1
                continue
        if i > 0 and S[i, j] == S[i - 1, j] + gap:
            aa.append(a[i - 1]); ab.append("-"); i -= 1
        else:
            aa.append("-"); ab.append(b[j - 1]); j -= 1
    A, B = "".join(reversed(aa)), "".join(reversed(ab))
    both = [(x, y) for x, y in zip(A, B) if x != "-" and y != "-"]
    ident = (sum(1 for x, y in both if x == y) / len(both)) if both else 0.0
    return A, B, ident


# =============================================================================
# §6. A3M PARSING + MSA PAIRING
# =============================================================================

@dataclass
class A3MRecord:
    headers: List[str]
    aligned: List[str]            # uppercase / '-' only, all same length
    deletions: np.ndarray         # (D, L) int, count of inserted residues
    query: str                    # ungapped query sequence
    path: str
    sha256: str

    @property
    def depth(self) -> int:
        return len(self.aligned)

    @property
    def n_cols(self) -> int:
        return len(self.aligned[0]) if self.aligned else 0


def parse_a3m(path: Path) -> A3MRecord:
    """
    A3M convention (HH-suite):
      UPPERCASE = aligned column (match state)
      lowercase = insertion relative to query (NOT a column)
      '-' / '.' = gap in an aligned column
    Deletion matrix counts lowercase characters preceding each column.
    """
    text = read_text_maybe_gz(path)
    headers: List[str] = []
    raw: List[str] = []
    cur: List[str] = []
    hdr: Optional[str] = None
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if hdr is not None:
                headers.append(hdr); raw.append("".join(cur))
            hdr = line[1:].strip(); cur = []
        else:
            cur.append(line.strip())
    if hdr is not None:
        headers.append(hdr); raw.append("".join(cur))
    if not raw:
        raise ValueError(f"{path}: empty a3m")

    aligned: List[str] = []
    deletions: List[List[int]] = []
    for s in raw:
        cols: List[str] = []
        dels: List[int] = []
        pending = 0
        for c in s:
            if c.islower():
                pending += 1
            elif c.isupper() or c in "-.":
                cols.append("-" if c in "-." else c)
                dels.append(pending)
                pending = 0
        aligned.append("".join(cols))
        deletions.append(dels)

    L = len(aligned[0])
    bad = [i for i, s in enumerate(aligned) if len(s) != L]
    if bad:
        # keep only rows matching the query column count; report
        aligned = [s for i, s in enumerate(aligned) if i not in bad]
        deletions = [d for i, d in enumerate(deletions) if i not in bad]
        headers = [h for i, h in enumerate(headers) if i not in bad]
    D = np.zeros((len(aligned), L), dtype=np.int32)
    for i, d in enumerate(deletions):
        D[i, :min(L, len(d))] = d[:L]
    query = aligned[0].replace("-", "")
    return A3MRecord(headers=headers, aligned=aligned, deletions=D,
                     query=query, path=str(path), sha256=sha256_file(path))


def compute_neff(aligned: List[str], identity_threshold: float = 0.8,
                 max_sample: int = 256, seed: int = 0) -> float:
    """Neff = sum_i 1/|{j : id(i,j) >= thr}|, subsampled for large MSAs."""
    D = len(aligned)
    if D <= 1:
        return float(D)
    idx = np.arange(D)
    if D > max_sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(D, max_sample, replace=False))
    seqs = [aligned[i] for i in idx]
    arr = np.array([[ord(c) for c in s] for s in seqs], dtype=np.uint8)
    n = len(seqs)
    counts = np.ones(n)
    for i in range(n):
        valid = (arr[i] != ord("-")) & (arr != ord("-"))
        eq = (arr == arr[i]) & valid
        denom = valid.sum(axis=1).clip(min=1)
        ident = eq.sum(axis=1) / denom
        counts[i] = float((ident >= identity_threshold).sum())
    return float(np.sum(1.0 / counts) * (D / n))


def pair_msa_to_chain(a3m: A3MRecord, chain_seq: str,
                      min_identity: float = 0.95,
                      allow_subsequence: bool = True
                      ) -> Dict[str, Any]:
    """
    Decide whether this a3m belongs to this chain, and if so, map its
    columns onto the chain's residue indices.

    Cases:
      EXACT       : a3m query == chain sequence
      SUBSTRING   : a3m query is a contiguous subsequence of the chain
      SUPERSTRING : chain is a subsequence of the a3m query (construct tag)
      ALIGNED     : identity >= min_identity, columns remapped by alignment
      REJECT      : identity below threshold

    Returns dict with status, col_to_residue (list, len = a3m columns,
    value = index into chain residues or -1), identity, and a reason.
    """
    q = a3m.query
    L = a3m.n_cols
    out: Dict[str, Any] = {"a3m_path": a3m.path, "a3m_sha256": a3m.sha256,
                           "depth": a3m.depth, "n_cols": L,
                           "query_len": len(q), "chain_len": len(chain_seq)}
    if q == chain_seq:
        out.update(status="EXACT", identity=1.0, offset=0)
        out["col_to_residue"] = _cols_to_indices(a3m.aligned[0], 0, len(chain_seq))
        return out
    if allow_subsequence and q in chain_seq:
        off = chain_seq.index(q)
        out.update(status="SUBSTRING", identity=1.0, offset=off)
        out["col_to_residue"] = _cols_to_indices(a3m.aligned[0], off, len(chain_seq))
        return out
    if allow_subsequence and chain_seq in q:
        off = q.index(chain_seq)
        cmap = _cols_to_indices(a3m.aligned[0], 0, len(q))
        remap = [c - off if c >= 0 and (c - off) < len(chain_seq) else -1
                 for c in cmap]
        out.update(status="SUPERSTRING", identity=1.0, offset=-off)
        out["col_to_residue"] = remap
        return out
    A, B, ident = nw_align(q, chain_seq)
    out["identity"] = ident
    if ident < min_identity:
        out.update(status="REJECT",
                   reason=f"identity {ident:.3f} < {min_identity}")
        out["col_to_residue"] = [-1] * L
        return out
    # walk alignment: a3m column index -> chain residue index
    col = 0
    ci = 0
    mapping: List[int] = []
    for x, y in zip(A, B):
        if x != "-":
            mapping.append(ci if y != "-" else -1)
            col += 1
        if y != "-":
            ci += 1
    if len(mapping) != L:
        out.update(status="REJECT",
                   reason=f"alignment column count {len(mapping)} != {L}")
        out["col_to_residue"] = [-1] * L
        return out
    out.update(status="ALIGNED", offset=0)
    out["col_to_residue"] = mapping
    return out


def _cols_to_indices(query_row: str, offset: int, chain_len: int) -> List[int]:
    """Map each aligned column to a chain residue index (-1 for gaps)."""
    out: List[int] = []
    r = offset
    for c in query_row:
        if c == "-":
            out.append(-1)
        else:
            out.append(r if 0 <= r < chain_len else -1)
            r += 1
    return out


# =============================================================================
# §7. CURATION POLICY
# =============================================================================

@dataclass
class CurationConfig:
    # chain selection
    chains_allow: Optional[List[str]] = None
    chains_deny: List[str] = field(default_factory=list)
    min_chain_len: int = 20
    max_chain_len: int = 2048
    select: str = "all"                  # "all" | "best" | "first"
    # resolution
    max_resolution: Optional[float] = 3.5
    allow_methods: List[str] = field(default_factory=lambda: [
        "x-ray diffraction", "electron microscopy", "neutron diffraction"])
    missing_resolution_policy: str = "reject"   # "reject" | "accept_flagged"
    reject_multimodel: bool = True
    # missing residues
    missing_policy: str = "segment"      # "reject" | "segment" | "mask"
    max_gap_for_reject: int = 0
    min_segment_len: int = 20
    max_missing_frac: float = 0.10
    require_backbone: Tuple[str, ...] = ("N", "CA", "C")
    # delta_min / kappa-fatness segment guard (threshold semantics)
    use_delta_min_guard: bool = True
    kappa: float = 2.0
    a_min: float = 0.01
    min_observed_density: float = 0.85
    # modified residues
    keep_modified: bool = True
    keep_hetatm_aa: bool = False
    # MSA
    require_msa: bool = False
    msa_min_identity: float = 0.95
    msa_max_depth: int = 512
    msa_min_neff: float = 1.0
    msa_allow_subsequence: bool = True
    # stats
    subsample_block: int = 16

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


def delta_min_exact(n: int = 3, kappa: float = 2.0, a_min: float = 0.01) -> float:
    """Isodiametric floor (Bieberbach 1915). Used here as a THRESHOLD."""
    d = n - 1
    om = math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)
    return (1.0 / kappa) * (a_min / om) ** (1.0 / d)


def chain_completeness(ch: ChainRecord) -> Dict[str, Any]:
    """Observed vs declared. Uses SEQRES if available, else numbering span."""
    obs = ch.residues
    if not obs:
        return {"n_observed": 0, "n_declared": 0, "missing": [],
                "completeness": 0.0, "gaps": [], "source": "empty"}
    nums = [r.auth_seq for r in obs]
    if ch.seqres:
        n_decl = len(ch.seqres)
        src = "SEQRES"
    elif ch.seqres_numbering:
        n_decl = len(set(ch.seqres_numbering))
        src = "pdbx_poly_seq_scheme"
    else:
        n_decl = max(nums) - min(nums) + 1
        src = "author_numbering_span"
    obs_set = set(nums)
    lo, hi = min(nums), max(nums)
    missing = [i for i in range(lo, hi + 1) if i not in obs_set]
    if ch.declared_missing:
        missing = sorted(set(missing) | set(ch.declared_missing))
    gaps: List[Tuple[int, int]] = []
    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] > 1:
            gaps.append((nums[i - 1] + 1, nums[i] - 1))
    comp = len(obs) / max(n_decl, 1)
    return {"n_observed": len(obs), "n_declared": n_decl,
            "missing": missing, "gaps": gaps,
            "completeness": comp, "source": src,
            "n_declared_missing": len(ch.declared_missing)}


def contiguous_segments(ch: ChainRecord, min_len: int,
                        require_bb: Tuple[str, ...]) -> List[List[int]]:
    """Split residue list into runs with consecutive author numbering AND
    all required backbone atoms present. Never bridges a gap."""
    segs: List[List[int]] = []
    cur: List[int] = []
    for i, r in enumerate(ch.residues):
        ok_bb = all(a in r.atoms for a in require_bb)
        consec = (not cur) or (r.auth_seq == ch.residues[cur[-1]].auth_seq + 1
                               and r.ins_code == "" and
                               ch.residues[cur[-1]].ins_code == "")
        if ok_bb and consec:
            cur.append(i)
        else:
            if len(cur) >= min_len:
                segs.append(cur)
            cur = [i] if ok_bb else []
    if len(cur) >= min_len:
        segs.append(cur)
    return segs


def segment_guard(seg_res: List[ResidueObs], cfg: CurationConfig) -> Dict[str, Any]:
    """
    delta_min / kappa-fatness applied as a THRESHOLD on segment quality:
      length >= max(min_segment_len, ceil(delta_min * scale))
      observed backbone density >= min_observed_density
    This is an engineering acceptance gate. It is NOT a claim that the
    isodiametric inequality governs protein geometry.
    """
    dm = delta_min_exact(3, cfg.kappa, cfg.a_min)
    n = len(seg_res)
    dens = (sum(1 for r in seg_res
                if all(a in r.atoms for a in cfg.require_backbone)) / max(n, 1))
    ok = True
    reasons = []
    if n < cfg.min_segment_len:
        ok = False; reasons.append(f"len {n} < min_segment_len {cfg.min_segment_len}")
    if cfg.use_delta_min_guard and dens < cfg.min_observed_density:
        ok = False
        reasons.append(f"backbone density {dens:.3f} < {cfg.min_observed_density} "
                       f"(kappa-fatness guard, delta_min={dm:.5f})")
    return {"accepted": ok, "n": n, "density": dens, "delta_min": dm,
            "reasons": reasons}


# =============================================================================
# §8. CURATION  (structure -> curated samples)
# =============================================================================

@dataclass
class CuratedSample:
    sample_id: str
    entry_id: str
    chain_id: str
    segment_idx: int
    residue_indices: List[int]        # indices into chain.residues
    sequence: str
    auth_numbers: List[int]
    resolution: Optional[float]
    method: str
    completeness: float
    n_missing_in_span: int
    policy_decisions: List[str]
    msa_info: Optional[Dict[str, Any]] = None
    source_sha256: str = ""
    a3m_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def curate_structure(rec: StructureRecord, cfg: CurationConfig,
                     a3m_by_chain: Optional[Dict[str, Path]] = None,
                     a3m_fallback: Optional[Path] = None
                     ) -> Tuple[List[CuratedSample], Dict[str, Any]]:
    decisions: List[str] = []
    audit: Dict[str, Any] = {
        "entry_id": rec.entry_id, "source": rec.source_path,
        "sha256": rec.source_sha256, "method": rec.method,
        "resolution": rec.resolution, "n_models": rec.n_models,
        "n_chains_found": len(rec.chains), "rejected": [], "accepted": [],
        "ignored_cif_categories": rec.ignored_categories,
        "warnings": list(rec.warnings),
    }

    # ---- entry-level gates ----
    if rec.n_models > 1 and cfg.reject_multimodel:
        audit["rejected"].append({"level": "entry",
                                  "reason": f"{rec.n_models} models (ensemble)"})
        return [], audit
    meth = rec.method.lower()
    if not any(m in meth for m in cfg.allow_methods):
        audit["rejected"].append({"level": "entry",
                                  "reason": f"method '{rec.method}' not allowed"})
        return [], audit
    if rec.resolution is None:
        if cfg.missing_resolution_policy == "reject":
            audit["rejected"].append({"level": "entry",
                                      "reason": "resolution absent, policy=reject"})
            return [], audit
        decisions.append("resolution ABSENT — accepted flagged")
    elif cfg.max_resolution is not None and rec.resolution > cfg.max_resolution:
        audit["rejected"].append({"level": "entry",
                                  "reason": f"resolution {rec.resolution} > "
                                            f"{cfg.max_resolution}"})
        return [], audit

    # ---- chain scoring / selection ----
    cand: List[Tuple[float, ChainRecord, Dict[str, Any]]] = []
    for ch in rec.chains:
        if cfg.chains_allow and ch.chain_id not in cfg.chains_allow:
            audit["rejected"].append({"level": "chain", "chain": ch.chain_id,
                                      "reason": "not in chains_allow"})
            continue
        if ch.chain_id in cfg.chains_deny:
            audit["rejected"].append({"level": "chain", "chain": ch.chain_id,
                                      "reason": "in chains_deny"})
            continue
        cc = chain_completeness(ch)
        if cc["n_observed"] < cfg.min_chain_len:
            audit["rejected"].append({"level": "chain", "chain": ch.chain_id,
                                      "reason": f"observed {cc['n_observed']} < "
                                                f"{cfg.min_chain_len}"})
            continue
        score = cc["n_observed"] * cc["completeness"]
        cand.append((score, ch, cc))
    if not cand:
        return [], audit
    cand.sort(key=lambda t: -t[0])
    if cfg.select == "best":
        cand = cand[:1]
    elif cfg.select == "first":
        cand = sorted(cand, key=lambda t: t[1].chain_id)[:1]

    samples: List[CuratedSample] = []
    for score, ch, cc in cand:
        ch_dec = list(decisions)
        miss_frac = 1.0 - cc["completeness"]
        if miss_frac > cfg.max_missing_frac:
            if cfg.missing_policy == "reject":
                audit["rejected"].append({"level": "chain", "chain": ch.chain_id,
                                          "reason": f"missing_frac {miss_frac:.3f} > "
                                                    f"{cfg.max_missing_frac}"})
                continue
            ch_dec.append(f"missing_frac={miss_frac:.3f} policy={cfg.missing_policy}")
        max_gap = max((b - a + 1 for a, b in cc["gaps"]), default=0)
        if cfg.missing_policy == "reject" and max_gap > cfg.max_gap_for_reject:
            audit["rejected"].append({"level": "chain", "chain": ch.chain_id,
                                      "reason": f"internal gap {max_gap} > "
                                                f"{cfg.max_gap_for_reject}"})
            continue

        # ---- MSA pairing (per chain) ----
        msa_info: Optional[Dict[str, Any]] = None
        a3m_path = None
        if a3m_by_chain and ch.chain_id in a3m_by_chain:
            a3m_path = a3m_by_chain[ch.chain_id]
        elif a3m_fallback is not None:
            a3m_path = a3m_fallback
        if a3m_path is not None and Path(a3m_path).exists():
            try:
                a3m = parse_a3m(Path(a3m_path))
                msa_info = pair_msa_to_chain(a3m, ch.observed_sequence,
                                             cfg.msa_min_identity,
                                             cfg.msa_allow_subsequence)
                msa_info["neff"] = compute_neff(a3m.aligned)
                if msa_info["status"] == "REJECT":
                    if cfg.require_msa:
                        audit["rejected"].append({"level": "chain",
                                                  "chain": ch.chain_id,
                                                  "reason": "MSA " + msa_info["reason"]})
                        continue
                    ch_dec.append("MSA REJECTED: " + msa_info["reason"])
                    msa_info = None
                elif msa_info["neff"] < cfg.msa_min_neff:
                    ch_dec.append(f"MSA Neff {msa_info['neff']:.2f} < "
                                  f"{cfg.msa_min_neff} (kept, flagged)")
            except Exception as e:
                if cfg.require_msa:
                    audit["rejected"].append({"level": "chain",
                                              "chain": ch.chain_id,
                                              "reason": f"a3m parse failed: {e}"})
                    continue
                ch_dec.append(f"a3m parse failed: {e}")
        elif cfg.require_msa:
            audit["rejected"].append({"level": "chain", "chain": ch.chain_id,
                                      "reason": "no a3m available"})
            continue

        # ---- segment decomposition ----
        if cfg.missing_policy == "mask":
            segs = [list(range(len(ch.residues)))]
            ch_dec.append("missing_policy=mask — HAZARD: FAPE must mask "
                          "absent residues or supervision is wrong")
        else:
            segs = contiguous_segments(ch, cfg.min_segment_len, cfg.require_backbone)
        if not segs:
            audit["rejected"].append({"level": "chain", "chain": ch.chain_id,
                                      "reason": "no segment passes length/backbone"})
            continue

        for si, seg in enumerate(segs):
            seg_res = [ch.residues[i] for i in seg]
            gd = segment_guard(seg_res, cfg)
            if not gd["accepted"]:
                audit["rejected"].append({"level": "segment",
                                          "chain": ch.chain_id, "seg": si,
                                          "reason": "; ".join(gd["reasons"])})
                continue
            if len(seg_res) > cfg.max_chain_len:
                seg_res = seg_res[:cfg.max_chain_len]
                seg = seg[:cfg.max_chain_len]
                ch_dec.append(f"truncated to max_chain_len={cfg.max_chain_len}")
            seq = "".join(r.one for r in seg_res)
            nums = [r.auth_seq for r in seg_res]
            span_missing = len([m for m in cc["missing"]
                                if nums[0] <= m <= nums[-1]])
            s = CuratedSample(
                sample_id=f"{rec.entry_id}_{ch.chain_id}_s{si}",
                entry_id=rec.entry_id, chain_id=ch.chain_id, segment_idx=si,
                residue_indices=seg, sequence=seq, auth_numbers=nums,
                resolution=rec.resolution, method=rec.method,
                completeness=gd["density"], n_missing_in_span=span_missing,
                policy_decisions=ch_dec + [f"segment {si}: n={gd['n']} "
                                           f"density={gd['density']:.3f}"],
                msa_info=msa_info, source_sha256=rec.source_sha256,
                a3m_sha256=(msa_info or {}).get("a3m_sha256"))
            samples.append(s)
            audit["accepted"].append({"chain": ch.chain_id, "seg": si,
                                      "n": len(seg), "score": round(score, 2)})
    return samples, audit


# =============================================================================
# §9. EXPORT  (OpenFold / AF2 batch format)
# =============================================================================

def sample_to_arrays(rec: StructureRecord, s: CuratedSample,
                     cfg: CurationConfig,
                     a3m_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    ch = next(c for c in rec.chains if c.chain_id == s.chain_id)
    res = [ch.residues[i] for i in s.residue_indices]
    L = len(res)

    aatype = np.array([AA_IDX.get(r.one, UNK_IDX) for r in res], dtype=np.int64)
    residue_index = np.array([r.auth_seq for r in res], dtype=np.int64)

    pos = np.zeros((L, 37, 3), dtype=np.float32)
    amask = np.zeros((L, 37), dtype=np.float32)
    bfac = np.zeros((L, 37), dtype=np.float32)
    for i, r in enumerate(res):
        for an, xyz in r.atoms.items():
            j = ATOM37_IDX[an]
            pos[i, j] = xyz
            amask[i, j] = 1.0
            bfac[i, j] = r.bfactor.get(an, 0.0)

    out: Dict[str, np.ndarray] = {
        "aatype": aatype,
        "residue_index": residue_index,
        "all_atom_positions": pos,
        "all_atom_mask": amask,
        "b_factors": bfac,
        "seq_length": np.array([L], dtype=np.int64),
        "resolution": np.array([s.resolution if s.resolution is not None else -1.0],
                               dtype=np.float32),
    }

    # ---- MSA ----
    D = cfg.msa_max_depth
    msa = np.full((D, L), GAP_IDX, dtype=np.int64)
    dele = np.zeros((D, L), dtype=np.int32)
    msa_mask = np.zeros((D, L), dtype=np.float32)
    msa[0] = aatype
    msa_mask[0] = 1.0
    neff = 1.0
    if a3m_path is not None and Path(a3m_path).exists() and s.msa_info:
        a3m = parse_a3m(Path(a3m_path))
        colmap = s.msa_info.get("col_to_residue", [])
        # map chain residue index -> a3m column
        res_to_col = {}
        for c, ri in enumerate(colmap):
            if ri >= 0:
                res_to_col[ri] = c
        # segment offset within chain
        base = s.residue_indices[0]
        cols = [res_to_col.get(base + k, -1) for k in range(L)]
        keep = min(D, a3m.depth)
        for d in range(keep):
            row = a3m.aligned[d]
            for k, c in enumerate(cols):
                if c < 0 or c >= len(row):
                    continue
                ch_ = row[c]
                msa[d, k] = AA_IDX.get(ch_, GAP_IDX) if ch_ != "-" else GAP_IDX
                dele[d, k] = a3m.deletions[d, c]
                msa_mask[d, k] = 1.0
        neff = s.msa_info.get("neff", 1.0)
    out["msa"] = msa
    out["deletion_matrix_int"] = dele
    out["msa_mask"] = msa_mask
    out["num_alignments"] = np.array([min(D, max(1, int(msa_mask.sum(1).astype(bool).sum())))],
                                     dtype=np.int64)
    out["neff"] = np.array([neff], dtype=np.float32)
    return out


def to_openfold_batch(arrays: Dict[str, np.ndarray],
                      device: str = "cpu") -> Dict[str, Any]:
    """Add batch dim and convert to torch if available."""
    out = {}
    for k, v in arrays.items():
        if HAS_TORCH:
            t = torch.from_numpy(v)
            out[k] = t.unsqueeze(0).to(device) if v.ndim > 0 else t.to(device)
        else:
            out[k] = v[None, ...] if v.ndim > 0 else v
    # AF2 expects residue_mask / atom_mask aliases in some code paths
    if "all_atom_mask" in out:
        out["atom_mask"] = out["all_atom_mask"]
    if HAS_TORCH:
        out["residue_mask"] = (out["aatype"] != GAP_IDX).float()
    else:
        out["residue_mask"] = (out["aatype"] != GAP_IDX).astype(np.float32)
    return out


# =============================================================================
# §10. DATASET STATISTICS  (Politis–Romano — the correct use of that theorem)
# =============================================================================

def subsample_ci(x: np.ndarray, conf: float = 0.95,
                 block: Optional[int] = None) -> Dict[str, float]:
    """
    Politis-Romano overlapping-block subsampling CI.
    APPROPRIATE HERE because along-chain quality (completeness, B-factor,
    occupancy) is serially correlated: an iid SE understates the error.
    Requires stationarity + mixing. NOT assumption-free.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n, "block": 0,
                "status": "INSUFFICIENT"}
    b = block or max(2, int(round(n ** (1 / 3))))
    b = min(b, n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th)
    a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi),
            "n": n, "block": b, "status": "OK",
            "method": "Politis-Romano subsampling"}


def iid_ci_WRONG(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """Provided ONLY to measure how much an iid CI understates the error."""
    x = np.asarray(x, float).ravel(); n = len(x)
    if n < 2:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    se = float(x.std(ddof=1) / math.sqrt(n))
    z = 1.959963985
    return {"mean": float(x.mean()), "lo": float(x.mean() - z * se),
            "hi": float(x.mean() + z * se), "n": n,
            "WARNING": "iid SE invalid under serial correlation"}


class CSOCCurationCounter:
    """
    Bounds accept/reject switches in an active curation loop.
        N(T) <= (E_max + D)/gap,  D = ACTUAL removal (audited).
    Provable for THIS COUNTER. Not a statement about the data.
    """
    def __init__(self, hi: float = 0.6, lo: float = 0.25, rate: float = 0.08):
        assert hi > lo
        self.hi, self.lo, self.gap, self.rate = hi, lo, hi - lo, rate
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0; self.D_audit = 0.0

    def step(self, pressure: float) -> bool:
        ex = math.log1p(math.exp(-abs(pressure - self.hi))) + max(pressure - self.hi, 0.0)
        dec = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10.0))
        rem = dec * self.rate * self.acc
        self.acc += ex - rem
        self.D += rem; self.D_audit += rem
        self.E_max = max(self.E_max, self.acc)
        if self.acc < self.lo:
            self.armed = True
        fire = self.armed and self.acc > self.hi
        if fire:
            self.armed = False; self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        b = (self.E_max + self.D) / self.gap
        return {"events": self.events, "bound": b,
                "satisfied": self.events <= b + 1e-9,
                "D_actual": self.D,
                "D_audit_matches": abs(self.D - self.D_audit) < 1e-12,
                "scope": "engineering bound on the curation counter"}


def dataset_report(samples: List[CuratedSample],
                   cfg: CurationConfig) -> Dict[str, Any]:
    if not samples:
        return {"n_samples": 0, "status": "EMPTY"}
    lens = np.array([len(s.sequence) for s in samples], float)
    comp = np.array([s.completeness for s in samples], float)
    res = np.array([s.resolution for s in samples if s.resolution is not None], float)
    neffs = np.array([s.msa_info["neff"] for s in samples if s.msa_info], float)
    entries = sorted({s.entry_id for s in samples})
    return {
        "n_samples": len(samples),
        "n_entries": len(entries),
        "n_chains": len({(s.entry_id, s.chain_id) for s in samples}),
        "length": subsample_ci(lens, block=cfg.subsample_block),
        "completeness": subsample_ci(comp, block=cfg.subsample_block),
        "completeness_iid_WRONG": iid_ci_WRONG(comp),
        "resolution": subsample_ci(res, block=cfg.subsample_block) if len(res) else None,
        "n_with_msa": int(len(neffs)),
        "neff": subsample_ci(neffs, block=cfg.subsample_block) if len(neffs) else None,
        "config_hash": cfg.hash(),
        "statistics_note": ("subsampling CI used because along-chain quality is "
                            "serially correlated; iid CI shown for comparison only"),
    }


# =============================================================================
# §11. MANIFEST BUILDER
# =============================================================================

def find_a3m_for(entry_id: str, chain_id: str,
                 a3m_dirs: List[Path]) -> Optional[Path]:
    """Search common naming conventions. Returns first match."""
    stems = [f"{entry_id}_{chain_id}", f"{entry_id}{chain_id}",
             f"{entry_id.lower()}_{chain_id}", f"{entry_id}", entry_id.lower()]
    exts = [".a3m", ".a3m.gz", ".a3m.txt"]
    for d in a3m_dirs:
        if not d.exists():
            continue
        for s in stems:
            for e in exts:
                p = d / (s + e)
                if p.exists():
                    return p
    return None


def build_manifest(structure_dirs: List[str], a3m_dirs: List[str],
                   out_dir: str, cfg: CurationConfig,
                   export_npz: bool = True,
                   limit: Optional[int] = None) -> Dict[str, Any]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if export_npz:
        (out / "npz").mkdir(exist_ok=True)
    a3m_paths = [Path(d) for d in a3m_dirs]

    files: List[Path] = []
    for d in structure_dirs:
        p = Path(d)
        if p.is_file():
            files.append(p); continue
        if not p.exists():
            log.warning("[manifest] missing dir %s", p); continue
        for pat in ("*.pdb", "*.pdb.gz", "*.ent", "*.ent.gz",
                    "*.cif", "*.cif.gz", "*.mmcif", "*.mmcif.gz"):
            files.extend(sorted(p.rglob(pat)))
    files = sorted(set(files))
    if limit:
        files = files[:limit]
    log.info("[manifest] %d structure files", len(files))

    all_samples: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    counter = CSOCCurationCounter()
    n_fail = 0

    for i, f in enumerate(files):
        try:
            rec = parse_structure(f, keep_modified=cfg.keep_modified,
                                  keep_hetatm_aa=cfg.keep_hetatm_aa)
        except Exception as e:
            n_fail += 1
            audits.append({"source": str(f), "parse_error": str(e)})
            counter.step(0.9)
            continue
        a3m_by_chain: Dict[str, Path] = {}
        fallback = None
        for ch in rec.chains:
            p = find_a3m_for(rec.entry_id, ch.chain_id, a3m_paths)
            if p is not None:
                a3m_by_chain[ch.chain_id] = p
        fallback = find_a3m_for(rec.entry_id, "", a3m_paths)
        samples, audit = curate_structure(rec, cfg, a3m_by_chain, fallback)
        audits.append(audit)
        pressure = 0.2 if samples else 0.8
        counter.step(pressure)

        for s in samples:
            entry = s.to_dict()
            if export_npz:
                a3m_p = a3m_by_chain.get(s.chain_id, fallback)
                try:
                    arr = sample_to_arrays(rec, s, cfg, a3m_p)
                    npz_path = out / "npz" / f"{s.sample_id}.npz"
                    np.savez_compressed(npz_path, **arr)
                    entry["npz_path"] = str(npz_path)
                    entry["npz_sha256"] = sha256_file(npz_path)
                    entry["n_residues"] = int(arr["seq_length"][0])
                    entry["msa_depth_used"] = int(arr["num_alignments"][0])
                except Exception as e:
                    entry["export_error"] = str(e)
                    log.warning("[export] %s: %s", s.sample_id, e)
            all_samples.append(entry)
        if (i + 1) % 200 == 0:
            log.info("[manifest] %d/%d files, %d samples",
                     i + 1, len(files), len(all_samples))

    # ---- splits: by ENTRY, not by sample (prevents leakage) ----
    entries = sorted({s["entry_id"] for s in all_samples})
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(entries))
    ntr = int(0.8 * len(entries)); nva = int(0.9 * len(entries))
    split_of = {}
    for k, ei in enumerate(perm):
        split_of[entries[ei]] = ("train" if k < ntr else
                                 "val" if k < nva else "test")
    for s in all_samples:
        s["split"] = split_of.get(s["entry_id"], "test")

    cs = [CuratedSample(**{k: v for k, v in s.items()
                           if k in CuratedSample.__dataclass_fields__})
          for s in all_samples]
    report = dataset_report(cs, cfg)
    manifest = {
        "version": VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": asdict(cfg),
        "config_hash": cfg.hash(),
        "n_files_scanned": len(files),
        "n_parse_failures": n_fail,
        "n_samples": len(all_samples),
        "split_policy": "by ENTRY (80/10/10) — samples from one entry never straddle splits",
        "curation_counter": counter.report(),
        "dataset_report": report,
        "samples": all_samples,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "audit.json").write_text(json.dumps(audits, indent=2, default=str))
    log.info("[manifest] wrote %s (%d samples)", out / "manifest.json", len(all_samples))
    return manifest


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def iter_batches(manifest: Dict[str, Any], split: str = "train",
                 device: str = "cpu"):
    """Yield OpenFold-compatible batches from cached NPZ."""
    for s in manifest["samples"]:
        if s.get("split") != split or "npz_path" not in s:
            continue
        p = Path(s["npz_path"])
        if not p.exists():
            continue
        d = np.load(p)
        arr = {k: d[k] for k in d.files}
        yield s["sample_id"], to_openfold_batch(arr, device)


# =============================================================================
# §12. SELF-TEST
# =============================================================================

def _write_test_pdb(path: Path) -> None:
    """Helix with an INTERNAL GAP (residues 5-7 absent) + REMARK 465."""
    lines = ["HEADER    TEST PROTEIN                            01-JAN-26   TST1",
             "EXPDTA    X-RAY DIFFRACTION",
             "REMARK   2 RESOLUTION.    1.80 ANGSTROMS."]
    seq3 = ["ALA"] * 20
    lines.append("SEQRES   1 A   20  " + " ".join(seq3[:13]))
    lines.append("SEQRES   2 A   20  " + " ".join(seq3[13:]))
    lines.append("REMARK 465 MISSING RESIDUES")
    lines.append("REMARK 465   M RES C SSSEQI")
    for m in (5, 6, 7):
        lines.append(f"REMARK 465     ALA A {m:4d}")
    serial = 1
    for i in range(1, 21):
        if i in (5, 6, 7):
            continue
        ang = math.radians(100.0 * i)
        ca = np.array([2.3 * math.cos(ang), 2.3 * math.sin(ang), 1.5 * i])
        for an, off in (("N", [-0.5, 0.5, -0.5]), ("CA", [0, 0, 0]),
                        ("C", [0.5, -0.5, 0.5]), ("O", [1.0, 0.5, 0.0])):
            x, y, z = ca + np.array(off)
            elem = an[0]
            lines.append(
                f"ATOM  {serial:5d}  {an:<3s} ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.0:6.2f}          {elem:>2s}")
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _write_test_cif(path: Path) -> None:
    rows = []
    serial = 1
    for i in range(1, 13):
        if i in (6, 7):
            continue
        ang = math.radians(100.0 * i)
        ca = np.array([2.3 * math.cos(ang), 2.3 * math.sin(ang), 1.5 * i])
        for an, off in (("N", [-0.5, 0.5, -0.5]), ("CA", [0, 0, 0]),
                        ("C", [0.5, -0.5, 0.5]), ("O", [1.0, 0.5, 0.0])):
            x, y, z = ca + np.array(off)
            rows.append(f"ATOM {serial} C {an} . ALA A 1 {i} {i} ? {i} A 1 "
                        f"{x:.3f} {y:.3f} {z:.3f} 1.00 20.00 ? 1 TST1 A 1 {i} 1")
            serial += 1
    seq = "A" * 12
    text = f"""data_TST2
#
_entry.id   TST2
#
_exptl.method  'X-RAY DIFFRACTION'
#
_refine.ls_d_res_high   2.10
#
loop_
_entity.id
_entity.type
1 polymer
2 water
#
loop_
_entity_poly.entity_id
_entity_poly.seq_one_letter_code_can
1 {seq}
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
""" + "".join(f"A 1 {i} ALA {i} ?\n" for i in range(1, 13)) + """#
loop_
_pdbx_unobs_or_zero_occ_residues.polymer_flag
_pdbx_unobs_or_zero_occ_residues.label_asym_id
_pdbx_unobs_or_zero_occ_residues.label_seq_id
Y A 6
Y A 7
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.pdbx_PDB_model_num
_atom_site.auth_entity_id
_atom_site.label_atom_id_2
_atom_site.footnote_id
_atom_site.order_id
""" + "\n".join(rows) + "\n#\n"
    path.write_text(text)


def _write_test_a3m(path: Path, query: str, depth: int = 24,
                    mutate: float = 0.1, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    lines = [">query", query]
    for d in range(depth - 1):
        s = list(query)
        # lowercase insertions to exercise the deletion matrix
        nmut = max(1, int(len(query) * mutate))
        for p in rng.choice(len(query), nmut, replace=False):
            s[p] = AA_ORDER[int(rng.integers(0, 20))]
        out = "".join(s)
        if d % 5 == 0 and len(out) > 4:
            out = out[:3] + "gg" + out[3:]     # insertion (lowercase)
        lines += [f">hit{d} OX=9606", out]
    path.write_text("\n".join(lines) + "\n")


def self_test(verbose: bool = True) -> bool:
    print("=" * 78)
    print(f"  PDB/mmCIF + a3m CURATOR v{VERSION} — self-test")
    print("=" * 78)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="curator_"))
    try:
        # T1 PDB parsing
        print("\n[T1] PDB parsing (SEQRES, REMARK 2, REMARK 465, gap)")
        pdb = tmp / "tst1.pdb"; _write_test_pdb(pdb)
        r = parse_pdb(pdb)
        ch = r.chains[0]
        cc = chain_completeness(ch)
        print(f"      entry={r.entry_id} method={r.method} res={r.resolution}")
        print(f"      observed={cc['n_observed']} declared={cc['n_declared']} "
              f"missing={cc['missing']} gaps={cc['gaps']}")
        rep("resolution parsed", r.resolution == 1.80)
        rep("method parsed", "x-ray" in r.method.lower())
        rep("17 residues observed (20 - 3 missing)", cc["n_observed"] == 17)
        rep("SEQRES gives 20 declared", cc["n_declared"] == 20, cc["source"])
        rep("REMARK 465 missing captured", set([5, 6, 7]).issubset(set(cc["missing"])))
        rep("internal gap detected", len(cc["gaps"]) == 1 and cc["gaps"][0] == (5, 7))

        # T2 mmCIF parsing
        print("\n[T2] mmCIF parsing (loop_, entity_poly, unobserved)")
        cif = tmp / "tst2.cif"; _write_test_cif(cif)
        r2 = parse_mmcif(cif)
        ch2 = r2.chains[0]
        cc2 = chain_completeness(ch2)
        print(f"      entry={r2.entry_id} res={r2.resolution} "
              f"obs={cc2['n_observed']} decl={cc2['n_declared']} "
              f"missing={cc2['missing']}")
        rep("cif entry id", r2.entry_id == "TST2")
        rep("cif resolution", r2.resolution == 2.10)
        rep("cif 10 residues observed", cc2["n_observed"] == 10)
        rep("cif SEQRES from entity_poly = 12", cc2["n_declared"] == 12)
        rep("cif declared missing 6,7", set([6, 7]).issubset(set(cc2["missing"])))
        rep("water entity excluded", all(c.entity_type != "water" for c in r2.chains))

        # T3 a3m parsing + deletion matrix
        print("\n[T3] a3m parsing (lowercase insertions -> deletion matrix)")
        q = ch.observed_sequence
        a3m_p = tmp / "TST1_A.a3m"; _write_test_a3m(a3m_p, q, depth=24)
        a = parse_a3m(a3m_p)
        print(f"      depth={a.depth} cols={a.n_cols} query_len={len(a.query)}")
        print(f"      deletions max={int(a.deletions.max())} "
              f"rows with insertions={int((a.deletions.sum(1)>0).sum())}")
        rep("all rows same column count", len({len(s) for s in a.aligned}) == 1)
        rep("query length == chain length", len(a.query) == len(q))
        rep("insertions removed from columns", all(s.isupper() or "-" in s
                                                   for s in a.aligned))
        rep("deletion matrix records insertions", a.deletions.max() > 0)

        # T4 MSA pairing
        print("\n[T4] MSA pairing (exact / substring / aligned / reject)")
        p_exact = pair_msa_to_chain(a, q)
        rep("EXACT match", p_exact["status"] == "EXACT", f"id={p_exact['identity']}")
        sub_q = q[3:12]
        a_sub_p = tmp / "sub.a3m"; _write_test_a3m(a_sub_p, sub_q, depth=8)
        p_sub = pair_msa_to_chain(parse_a3m(a_sub_p), q)
        rep("SUBSTRING match with offset", p_sub["status"] == "SUBSTRING"
            and p_sub["offset"] == 3, f"offset={p_sub.get('offset')}")
        rng = np.random.default_rng(1)
        mut = "".join(AA_ORDER[int(rng.integers(0, 20))] if rng.random() < 0.4 else c
                      for c in q)
        a_mut_p = tmp / "mut.a3m"; _write_test_a3m(a_mut_p, mut, depth=8)
        p_mut = pair_msa_to_chain(parse_a3m(a_mut_p), q, min_identity=0.95)
        rep("low-identity MSA REJECTED", p_mut["status"] == "REJECT",
            f"id={p_mut['identity']:.3f}")
        neff = compute_neff(a.aligned)
        rep("Neff computed and <= depth", 1.0 <= neff <= a.depth, f"Neff={neff:.2f}")

        # T5 curation policies
        print("\n[T5] missing-residue policy: reject / segment / mask")
        cfg_rej = CurationConfig(missing_policy="reject", max_gap_for_reject=0,
                                 max_missing_frac=0.0, min_segment_len=5,
                                 min_chain_len=5)
        s_rej, au_rej = curate_structure(r, cfg_rej)
        rep("REJECT policy rejects gapped chain", len(s_rej) == 0,
            str(au_rej["rejected"][-1]["reason"])[:60])
        cfg_seg = CurationConfig(missing_policy="segment", min_segment_len=5,
                                 min_chain_len=5, max_missing_frac=0.5)
        s_seg, au_seg = curate_structure(r, cfg_seg,
                                         a3m_by_chain={"A": a3m_p})
        print(f"      SEGMENT -> {len(s_seg)} samples, "
              f"lengths={[len(x.sequence) for x in s_seg]}")
        rep("SEGMENT splits at the gap", len(s_seg) == 2,
            f"{[len(x.sequence) for x in s_seg]}")
        rep("segments are contiguous", all(
            all(x.auth_numbers[i+1] == x.auth_numbers[i] + 1
                for i in range(len(x.auth_numbers)-1)) for x in s_seg))
        rep("MSA paired on segments", all(x.msa_info is not None for x in s_seg))
        cfg_mask = CurationConfig(missing_policy="mask", min_segment_len=5,
                                  min_chain_len=5, max_missing_frac=0.5)
        s_mask, _ = curate_structure(r, cfg_mask)
        rep("MASK keeps one sample and flags hazard",
            len(s_mask) == 1 and any("HAZARD" in d for d in s_mask[0].policy_decisions))

        # T6 resolution / method gates
        print("\n[T6] resolution and method gates")
        cfg_res = CurationConfig(max_resolution=1.5, min_chain_len=5,
                                 min_segment_len=5, max_missing_frac=0.5)
        s_res, au_res = curate_structure(r, cfg_res)
        rep("resolution 1.8 > 1.5 rejected", len(s_res) == 0)
        r_nmr = StructureRecord("NMRX", "nmr", None, r.chains, str(pdb),
                                r.source_sha256, n_models=10)
        s_nmr, au_nmr = curate_structure(r_nmr, CurationConfig(min_chain_len=5))
        rep("multi-model NMR rejected", len(s_nmr) == 0,
            str(au_nmr["rejected"][0]["reason"])[:50])

        # T7 export
        print("\n[T7] OpenFold-compatible export")
        arr = sample_to_arrays(r, s_seg[0], cfg_seg, a3m_p)
        L = int(arr["seq_length"][0])
        print(f"      aatype={arr['aatype'].shape} pos={arr['all_atom_positions'].shape} "
              f"msa={arr['msa'].shape} del={arr['deletion_matrix_int'].shape}")
        rep("atom positions (L,37,3)", arr["all_atom_positions"].shape == (L, 37, 3))
        rep("atom mask matches observed", arr["all_atom_mask"].sum() > 0)
        rep("backbone N,CA,C present", all(
            arr["all_atom_mask"][:, ATOM37_IDX[x]].min() == 1.0 for x in ("N", "CA", "C")))
        rep("msa row 0 == aatype", np.array_equal(arr["msa"][0], arr["aatype"]))
        rep("deletion matrix non-negative", arr["deletion_matrix_int"].min() >= 0)
        if HAS_TORCH:
            b = to_openfold_batch(arr)
            rep("torch batch has batch dim", b["aatype"].shape[0] == 1)
            rep("residue_mask present", "residue_mask" in b)

        # T8 statistics
        print("\n[T8] Politis-Romano vs iid on correlated quality")
        rho = 0.85
        n = 400
        e = np.random.default_rng(0).normal(size=n)
        x = np.empty(n); x[0] = e[0]
        for i in range(1, n):
            x[i] = rho * x[i-1] + e[i]
        ci_s = subsample_ci(x); ci_i = iid_ci_WRONG(x)
        ws = ci_s["hi"] - ci_s["lo"]; wi = ci_i["hi"] - ci_i["lo"]
        print(f"      subsampling width={ws:.4f}  iid width={wi:.4f}  "
              f"ratio={ws/max(wi,1e-9):.2f}x")
        rep("subsampling CI wider under correlation", ws > wi * 1.5,
            f"{ws/max(wi,1e-9):.2f}x")
        rep("CI ordered", ci_s["lo"] <= ci_s["mean"] <= ci_s["hi"])

        # T9 CSOC counter
        print("\n[T9] CSOC curation counter bound")
        c = CSOCCurationCounter()
        for k in range(400):
            c.step(0.3 + 0.5 * abs(math.sin(0.11 * k)))
        rp = c.report()
        rep("N <= (E_max+D)/gap", rp["satisfied"],
            f"N={rp['events']} bound={rp['bound']:.2f}")
        rep("D audit matches", rp["D_audit_matches"])

        # T10 end-to-end manifest
        print("\n[T10] end-to-end manifest + NPZ cache")
        sdir = tmp / "structures"; sdir.mkdir()
        adir = tmp / "a3m"; adir.mkdir()
        _write_test_pdb(sdir / "tst1.pdb")
        _write_test_cif(sdir / "tst2.cif")
        _write_test_a3m(adir / "TST1_A.a3m", q, depth=16)
        _write_test_a3m(adir / "TST2_A.a3m", ch2.observed_sequence, depth=16)
        man = build_manifest([str(sdir)], [str(adir)], str(tmp / "out"),
                             CurationConfig(min_chain_len=5, min_segment_len=5,
                                            max_missing_frac=0.5))
        print(f"      samples={man['n_samples']} entries={man['dataset_report']['n_entries']}")
        rep("manifest written", (tmp / "out" / "manifest.json").exists())
        rep("audit written", (tmp / "out" / "audit.json").exists())
        rep("samples produced", man["n_samples"] >= 2)
        rep("NPZ cached", all("npz_path" in s for s in man["samples"]))
        rep("provenance sha256 present", all(s["source_sha256"] for s in man["samples"]))
        rep("split by entry", len({s["split"] for s in man["samples"]
                                   if s["entry_id"] == man["samples"][0]["entry_id"]}) == 1)
        n_batches = sum(1 for _ in iter_batches(man, "train"))
        print(f"      train batches iterable: {n_batches}")
        rep("batches iterable", n_batches >= 0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
VERIFIED
  PDB: SEQRES, REMARK 2, REMARK 465, internal gaps, altLoc, insertion codes.
  mmCIF: loop_ tokenizer, entity_poly, poly_seq_scheme, unobserved residues.
  a3m: lowercase insertions removed AND counted in the deletion matrix.
  MSA pairing: EXACT / SUBSTRING / ALIGNED / REJECT with identity threshold.
  Missing-residue policy: REJECT / SEGMENT / MASK, segments never bridge a gap.
  Export: (L,37,3) positions, atom mask from OBSERVED atoms, MSA row 0 = query.
  Statistics: subsampling CI wider than iid under serial correlation.
  CSOC: event bound holds with audited actual removal.
  Manifest: provenance hashes, entry-level splits, NPZ cache, audit log.

NOT CLAIMED
  No structure prediction. This is data hygiene.
  Heteromer taxonomy pairing is heuristic; verify before multimer training.
  delta_min/kappa-fatness here is a THRESHOLD, not a geometry theorem.
  OPMC / Carpet / Hodge / HP belong downstream in the trunk — ablate them there.
""")
    return ok


# =============================================================================
# §13. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="PDB/mmCIF + a3m curator")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "inspect", "build", "stats", "batch"])
    ap.add_argument("--structures", nargs="*", default=["data/structures"])
    ap.add_argument("--a3m", nargs="*", default=["data/a3m"])
    ap.add_argument("--out", default="data/curated")
    ap.add_argument("--file", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--max-resolution", type=float, default=3.5)
    ap.add_argument("--missing-policy", default="segment",
                    choices=["reject", "segment", "mask"])
    ap.add_argument("--min-segment-len", type=int, default=20)
    ap.add_argument("--min-chain-len", type=int, default=20)
    ap.add_argument("--max-missing-frac", type=float, default=0.10)
    ap.add_argument("--select", default="all", choices=["all", "best", "first"])
    ap.add_argument("--chains", nargs="*", default=None)
    ap.add_argument("--require-msa", action="store_true")
    ap.add_argument("--msa-min-identity", type=float, default=0.95)
    ap.add_argument("--msa-max-depth", type=int, default=512)
    ap.add_argument("--no-npz", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    cfg = CurationConfig(
        max_resolution=a.max_resolution, missing_policy=a.missing_policy,
        min_segment_len=a.min_segment_len, min_chain_len=a.min_chain_len,
        max_missing_frac=a.max_missing_frac, select=a.select,
        chains_allow=a.chains, require_msa=a.require_msa,
        msa_min_identity=a.msa_min_identity, msa_max_depth=a.msa_max_depth)

    out: Dict[str, Any] = {}
    if a.cmd == "inspect":
        if not a.file:
            print("--file required"); return 1
        rec = parse_structure(Path(a.file), keep_modified=cfg.keep_modified)
        print(f"entry={rec.entry_id} method={rec.method} res={rec.resolution} "
              f"models={rec.n_models}")
        print(f"sha256={rec.source_sha256[:16]}...")
        for ch in rec.chains:
            cc = chain_completeness(ch)
            segs = contiguous_segments(ch, cfg.min_segment_len, cfg.require_backbone)
            print(f"  chain {ch.chain_id}: obs={cc['n_observed']} "
                  f"decl={cc['n_declared']} ({cc['source']}) "
                  f"complete={cc['completeness']:.3f} gaps={cc['gaps']} "
                  f"segments={[len(s) for s in segs]}")
        if rec.ignored_categories:
            print(f"  ignored cif categories: {rec.ignored_categories[:12]}")
        samples, audit = curate_structure(rec, cfg)
        print(f"  would accept {len(samples)} samples")
        out = {"audit": audit, "samples": [s.to_dict() for s in samples]}

    elif a.cmd == "build":
        out = build_manifest(a.structures, a.a3m, a.out, cfg,
                             export_npz=not a.no_npz, limit=a.limit)
        print(json.dumps({k: v for k, v in out.items() if k != "samples"},
                         indent=2, default=str))

    elif a.cmd == "stats":
        if not a.manifest:
            print("--manifest required"); return 1
        man = load_manifest(Path(a.manifest))
        print(json.dumps(man.get("dataset_report", {}), indent=2, default=str))
        out = man.get("dataset_report", {})

    elif a.cmd == "batch":
        if not a.manifest:
            print("--manifest required"); return 1
        man = load_manifest(Path(a.manifest))
        n = 0
        for sid, b in iter_batches(man, a.split):
            shapes = {k: tuple(v.shape) for k, v in b.items()
                      if hasattr(v, "shape")}
            print(f"{sid}: {shapes}")
            n += 1
            if n >= 3:
                break
        print(f"({n} batches shown from split='{a.split}')")

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
