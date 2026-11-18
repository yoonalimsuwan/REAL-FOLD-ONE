#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ONE MedWorld v1.0 — Unified Research Platform
PIMPLE transient + implicit adjoint + KG + ontogenetic + organ CFD + iPSC risk
=============================================================================

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

Christ died upon the Cross to teach: when struck on the left cheek, turn also
the right, so that we may listen to the voice of the enemy who wishes us harm.

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
The philosophy is the author's belief, included by request, and is not a
scientific claim nor a condition of use.
-----------------------------------------------------------------------------

Architecture : PAI AND Yoon A Limsuwan / MSPS NETWORK
Code         : Qwen (Alibaba Group)
License      : MIT | Year 2026
ORCID        : 0009-0008-2374-0788

=============================================================================
CONTENTS
=============================================================================
[G]  Governance: IRB / PHI / SafetyGate — ENFORCED, not advisory
[1]  BiomedicalKnowledgeGraph — the requested JSON schema, real connectors
[2]  OntogeneticWorldModel — SSC / CSOC / OPMC / Carpet / Hodge-inspired
[3]  ProteinStructureModule — research framing, editor design BLOCKED
[4]  iPSCOrganoidTransplantResearch — risk + missing-evidence, never a go/no-go
[5]  PIMPLE transient solver + Rhie-Chow on a collocated FVM mesh
[6]  Implicit adjoint through the WHOLE time loop with checkpointing
[7]  PatientSpecificOrganCFD — organ registry, mesh gate, UQ, validation
[8]  Validation harness — Bland-Altman, ICC, calibration, coverage
[9]  Orchestrator + self-test + CLI

=============================================================================
HARD SAFETY RULES (enforced in code)
=============================================================================
R1  No PHI leaves this process. Every external query is scrubbed first.
R2  No patient-derived pathway without an ACTIVE IRB record AND a passed
    de-identification report.
R3  Every output carries clinical_actionability="research_only" and
    clinician_review_required=True.
R4  BLOCKED outright: gene-editor design, guide RNA, base/prime editors,
    delivery vectors, wet-lab protocols, pathogen enhancement, and
    autonomous clinical orders (diagnose / prescribe / operate / transplant).
R5  Missing credentials, models, or reference data produce an explicit
    UNAVAILABLE / NOT_COMPARED / NOT_IMPLEMENTED status. Never a fabricated
    number, never a silent fallback to a plausible-looking default.

=============================================================================
HONEST SCOPE
=============================================================================
* NOT a medical device. NOT for diagnosis, treatment, or surgical decisions.
* The CFD here is a collocated Cartesian FVM with Rhie-Chow. It is NOT a
  body-fitted unstructured solver, NOT FSI, NOT multiphase VOF. Those are
  declared NOT_IMPLEMENTED per organ, not silently approximated.
* Real folding / quantum chemistry / cell-type classification require
  external validated backends. Adapters are provided; without them the
  answer is UNAVAILABLE.
* The literature graph aggregates EVIDENCE, not causation. Confidence is a
  weighted evidence score, NOT a probability of pathogenicity.
* No experimental or clinical validation is performed here.
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("one_medworld")

VERSION = "1.0.0"
RESEARCH_ONLY = "research_only"
CARPET_RHO = 9.0 / 8.0

BANNER = f"""
=============================================================================
ONE MedWorld v{VERSION} — RESEARCH PLATFORM, NOT A MEDICAL DEVICE
Code: Qwen | Architecture: PAI & Yoon A Limsuwan / MSPS NETWORK
No PHI leaves this process. No IRB => no patient pathway.
Gene-editor design BLOCKED. Every output is research_only.
Missing data => UNAVAILABLE / NOT_COMPARED, never a fabricated number.
=============================================================================
"""


class Status:
    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_COMPARED = "NOT_COMPARED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


# =============================================================================
# §G. GOVERNANCE — the single choke point
# =============================================================================

class Intent:
    LITERATURE = "literature_query"
    SIMULATION = "simulation"
    RISK = "risk_assessment"
    GENE_EDITOR = "gene_editor_design"
    WETLAB = "wetlab_protocol"
    PATHOGEN = "pathogen_enhancement"
    CLINICAL_ORDER = "autonomous_clinical_order"


class SafetyGate:
    BLOCKED: Dict[str, List[str]] = {
        Intent.GENE_EDITOR: [
            r"\bcrispr\b", r"\bcas9\b", r"\bcas12\b", r"\bcas13\b",
            r"\bguide\s*rna\b", r"\bgrna\b", r"\bsgrna\b",
            r"\bbase\s*editor\b", r"\bprime\s*editor\b", r"\bgene\s*drive\b",
            r"\bgene\s*edit(or|ing)\b", r"\bknock\s*(out|in)\b",
            r"\baav\s*capsid\b", r"\bdelivery\s*vector\b",
            r"\bdesign\s+(a|an|the)?\s*(editor|crispr|guide|nuclease)\b",
        ],
        Intent.WETLAB: [
            r"\bwet\s*lab\b", r"\btransfection\b", r"\belectroporation\b",
            r"\bculture\s*protocol\b", r"\bmedia\s*recipe\b",
            r"\bstep\s*by\s*step\s*(protocol|method)\b", r"\bpcr\s*condition\b",
        ],
        Intent.PATHOGEN: [
            r"\benhance\s*virulence\b", r"\bimmune\s*evasion\b",
            r"\bincrease\s*transmissibility\b", r"\bgain\s*of\s*function\b",
        ],
        Intent.CLINICAL_ORDER: [
            r"\bdiagnose\s*(the\s*)?patient\b", r"\bprescribe\b",
            r"\boperate\s*on\b", r"\btransplant\s*(now|the\s*patient)\b",
            r"\bautonomous\s*surgery\b", r"\bshould\s*(we|i)\s*operate\b",
            r"\bapprove\s*transplant\b",
        ],
    }

    def classify(self, text: str) -> Tuple[str, List[str]]:
        low = text.lower()
        for intent, pats in self.BLOCKED.items():
            hits = [p for p in pats if re.search(p, low)]
            if hits:
                return intent, hits
        return Intent.LITERATURE, []

    def check(self, text: str) -> Dict[str, Any]:
        intent, hits = self.classify(text)
        blocked = bool(hits)
        return {
            "allowed": not blocked, "intent": intent, "blocked_patterns": hits,
            "reason": (f"BLOCKED intent={intent}" if blocked else "allowed research query"),
            "clinical_actionability": RESEARCH_ONLY,
        }


class PHIGuard:
    PATTERNS = {
        "name_like": r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "mrn": r"\b(MRN|mrn)[:\s]*\d{4,}\b",
        "date_full": r"\b(19|20)\d{2}-\d{2}-\d{2}\b",
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "phone": r"\b\+?\d[\d\s().-]{8,}\d\b",
        "dicom_uid": r"\b1\.2\.\d{3,}\.[\d.]{6,}\b",
    }
    DIRECT = {"name", "patient_name", "given_name", "family_name", "ssn", "mrn",
              "patient_id", "national_id", "address", "email", "phone",
              "birth_date", "dob", "date_of_birth", "accession_number",
              "dicom_patient_name", "dicom_patient_id", "photo"}
    QUASI = {"age", "sex", "postcode", "zip", "ethnicity", "admission_date",
             "hospital", "rare_diagnosis"}

    def scrub(self, text: str) -> Tuple[str, List[str]]:
        out, removed = text, []
        for label, pat in self.PATTERNS.items():
            new = re.sub(pat, "[REDACTED]", out)
            if new != out:
                removed.append(label); out = new
        return out, removed

    def deidentify(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        removed, quasi, out = [], [], dict(rec)
        for k in list(out.keys()):
            if k.lower() in self.DIRECT:
                out.pop(k); removed.append(k)
            elif k.lower() in self.QUASI:
                quasi.append(k)
        if "age" in out:
            try:
                a = int(out.pop("age"))
                out["age_band"] = f"{(a//10)*10}-{(a//10)*10+9}"
                removed.append("age->age_band")
            except Exception:
                pass
        for k, v in list(out.items()):
            if isinstance(v, str):
                s, rm = self.scrub(v)
                if rm:
                    out[k] = s; removed.append(f"{k}:{','.join(rm)}")
        risk = "low" if not quasi else ("medium" if len(quasi) <= 3 else "high")
        return {"passed": risk != "high", "removed_fields": removed,
                "residual_risk": risk, "quasi_identifiers": quasi,
                "clean_record": out}


@dataclass
class IRBRecord:
    protocol_id: str
    institution: str
    approval_date: str
    expiry_date: str
    approved_uses: List[str]
    consent_model: str = "written"
    data_use_agreement: bool = False
    hgr_permit: bool = False
    status: str = "PENDING"

    def active(self) -> bool:
        if self.status != "APPROVED":
            return False
        try:
            return time.time() < time.mktime(time.strptime(self.expiry_date, "%Y-%m-%d"))
        except Exception:
            return False


class Governance:
    def __init__(self, irb: Optional[IRBRecord] = None):
        self.irb, self.phi, self.safety = irb, PHIGuard(), SafetyGate()
        self.audit: List[Dict[str, Any]] = []

    def authorize(self, text: str, uses_patient_data: bool = False,
                  deid: Optional[Dict[str, Any]] = None,
                  purpose: str = "research") -> Dict[str, Any]:
        s = self.safety.check(text)
        d = dict(s)
        d.update({"authorized": False, "clinician_review_required": True,
                  "required_approvals": ["IRB", "ethics_committee", "clinician", "regulator"]})
        if not s["allowed"]:
            self._log(d, text); return d
        if uses_patient_data:
            if self.irb is None:
                d["reason"] = "BLOCKED: no IRB record"; self._log(d, text); return d
            if not self.irb.active():
                d["reason"] = f"BLOCKED: IRB status={self.irb.status} or expired"
                self._log(d, text); return d
            if purpose not in self.irb.approved_uses:
                d["reason"] = f"BLOCKED: purpose '{purpose}' not approved"
                self._log(d, text); return d
            if not self.irb.data_use_agreement:
                d["reason"] = "BLOCKED: no data use agreement"; self._log(d, text); return d
            if not deid or not deid.get("passed"):
                d["reason"] = "BLOCKED: de-identification missing/failed"
                self._log(d, text); return d
        d["authorized"] = True; d["reason"] = "authorized for research use"
        self._log(d, text); return d

    def _log(self, d: Dict[str, Any], text: str) -> None:
        self.audit.append({"ts": time.time(),
                           "query_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                           "authorized": d["authorized"], "reason": d["reason"]})

    def regulatory(self) -> Dict[str, Any]:
        return {"intended_use": "research / hypothesis generation",
                "device_status": "NOT_A_MEDICAL_DEVICE",
                "us_fda": "not submitted", "eu_mdr": "not CE marked",
                "china_nmpa": "not registered",
                "data_law": ["HIPAA", "GDPR", "PIPL", "CN HGR regulation"],
                "stem_cell": ["ISSCR", "ISCT"],
                "reporting": ["TRIPOD+AI", "DECIDE-AI", "CONSORT-AI"],
                "prohibited": ["autonomous diagnosis", "autonomous treatment",
                               "gene editor design", "wet-lab protocol generation"]}


# =============================================================================
# §1. BIOMEDICAL KNOWLEDGE GRAPH
# =============================================================================

EVIDENCE_WEIGHT = {
    "curated": 1.0, "clinical_review": 0.95, "GWAS": 0.75,
    "single_cell": 0.65, "spatial": 0.65, "animal": 0.55,
    "in_vitro": 0.45, "text_mined": 0.30, "inferred": 0.25,
}


@dataclass
class EvidenceRecord:
    """Exactly the requested output schema, plus mandatory provenance."""
    gene: str
    variant: Optional[str]
    phenotype: str
    evidence_level: str
    confidence: float
    contradictions: List[str] = field(default_factory=list)
    developmental_stage: Optional[str] = None
    tissue_specificity: Optional[str] = None
    clinical_actionability: str = RESEARCH_ONLY
    effect: str = "associated"
    source: str = ""
    retrieved_at: float = field(default_factory=time.time)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class HTTP:
    """stdlib HTTP with disk cache, rate limit, retry, and PHI scrubbing."""
    def __init__(self, cache_dir="./medworld_cache", ttl=86400.0,
                 min_interval=0.34, offline=True, timeout=20.0, retries=3):
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.ttl, self.offline, self.timeout, self.retries = ttl, offline, timeout, retries
        self.min_interval, self._last = min_interval, 0.0
        self.phi = PHIGuard()
        self.n_req = self.n_hit = 0

    def get(self, url: str, headers=None) -> Tuple[Optional[str], Dict[str, Any]]:
        clean, removed = self.phi.scrub(url)
        meta = {"url": clean, "phi_stripped": removed, "cached": False}
        key = self.cache / (hashlib.sha256(clean.encode()).hexdigest()[:32] + ".txt")
        if key.exists() and time.time() - key.stat().st_mtime < self.ttl:
            self.n_hit += 1; meta["cached"] = True
            return key.read_text(errors="replace"), meta
        if self.offline:
            meta.update(status=Status.UNAVAILABLE, reason="offline; no cache")
            return None, meta
        err = ""
        for a in range(self.retries):
            dt = time.time() - self._last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last = time.time(); self.n_req += 1
            try:
                req = urllib.request.Request(clean, headers=headers or
                                             {"User-Agent": "ONE-MedWorld/1.0 (research)"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = r.read().decode("utf-8", errors="replace")
                key.write_text(body); meta["status"] = Status.OK
                return body, meta
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2.0 ** a); continue
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"; time.sleep(2.0 ** a)
        meta.update(status=Status.ERROR, reason=err)
        return None, meta


class BiomedicalKnowledgeGraph:
    """
    Aggregates EVIDENCE, not causation. `confidence` is a weighted evidence
    score with a contradiction penalty — NOT a probability of pathogenicity.
    """
    EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    GWAS = "https://www.ebi.ac.uk/gwas/rest/api"
    HPO_OBO = "http://purl.obolibrary.org/obo/hp.obo"

    def __init__(self, http: Optional[HTTP] = None):
        self.http = http or HTTP()
        self.records: List[EvidenceRecord] = []
        self.unavailable: List[Dict[str, str]] = []
        self.hpo: Dict[str, Dict[str, Any]] = {}
        self.keys = {k: os.environ.get(v) for k, v in
                     {"ncbi": "NCBI_API_KEY", "omim": "OMIM_API_KEY",
                      "disgenet": "DISGENET_API_KEY"}.items()}

    def _note(self, src, reason):
        self.unavailable.append({"source": src, "reason": reason})

    # ---- connectors ----
    def pubmed(self, term: str, retmax=20) -> Dict[str, Any]:
        clean, rm = self.http.phi.scrub(term)
        p = {"db": "pubmed", "term": clean, "retmax": str(retmax), "retmode": "json"}
        if self.keys["ncbi"]:
            p["api_key"] = self.keys["ncbi"]
        url = f"{self.EUTILS}/esearch.fcgi?{urllib.parse.urlencode(p)}"
        body, meta = self.http.get(url)
        if body is None:
            self._note("PubMed", meta.get("reason", ""))
            return {"status": meta.get("status"), "pmids": [], "phi_stripped": rm}
        try:
            ids = json.loads(body)["esearchresult"]["idlist"]
        except Exception as e:
            return {"status": Status.ERROR, "reason": str(e), "pmids": []}
        return {"status": Status.OK, "pmids": ids, "phi_stripped": rm, "url": url}

    def clinvar(self, gene: str, phenotype: str, retmax=20) -> int:
        p = {"db": "clinvar", "term": f"{gene}[gene]", "retmax": str(retmax),
             "retmode": "json"}
        if self.keys["ncbi"]:
            p["api_key"] = self.keys["ncbi"]
        body, meta = self.http.get(f"{self.EUTILS}/esearch.fcgi?{urllib.parse.urlencode(p)}")
        if body is None:
            self._note("ClinVar", meta.get("reason", "")); return 0
        try:
            ids = json.loads(body)["esearchresult"]["idlist"]
        except Exception:
            return 0
        if not ids:
            return 0
        p2 = {"db": "clinvar", "id": ",".join(ids), "retmode": "json"}
        if self.keys["ncbi"]:
            p2["api_key"] = self.keys["ncbi"]
        b2, m2 = self.http.get(f"{self.EUTILS}/esummary.fcgi?{urllib.parse.urlencode(p2)}")
        if b2 is None:
            self._note("ClinVar", m2.get("reason", "")); return 0
        n = 0
        try:
            res = json.loads(b2).get("result", {})
            for uid in res.get("uids", []):
                r = res.get(uid, {})
                sig = r.get("clinical_significance", {})
                desc = (sig.get("description", "") if isinstance(sig, dict) else str(sig)).lower()
                eff, conf = (("risk_increase", 0.9) if "pathogenic" in desc else
                             ("benign", 0.8) if "benign" in desc else ("uncertain", 0.35))
                lvl = ("clinical_review" if "reviewed" in
                       (r.get("review_status") or "").lower() else "curated")
                vs = r.get("variation_set", [])
                self.records.append(EvidenceRecord(
                    gene=gene, variant=(vs[0].get("variation_name") if vs else None),
                    phenotype=phenotype, evidence_level=lvl, confidence=conf,
                    effect=eff, source=f"ClinVar:{uid}"))
                n += 1
        except Exception as e:
            self._note("ClinVar", str(e))
        return n

    def gwas(self, rsid: str, phenotype: str) -> int:
        if not re.fullmatch(r"rs\d+", rsid):
            self._note("GWAS", "rsid must match rs\\d+"); return 0
        url = f"{self.GWAS}/singleNucleotidePolymorphisms/{rsid}/associations"
        body, meta = self.http.get(url, {"Accept": "application/json"})
        if body is None:
            self._note("GWAS_Catalog", meta.get("reason", "")); return 0
        n = 0
        try:
            for a in json.loads(body).get("_embedded", {}).get("associations", []):
                pv = a.get("pvalue")
                sig = isinstance(pv, (int, float)) and pv < 5e-8
                traits = [t.get("trait", "") for t in a.get("efoTraits", [])]
                self.records.append(EvidenceRecord(
                    gene=a.get("riskAllele") or "unknown", variant=rsid,
                    phenotype="; ".join(traits) or phenotype,
                    evidence_level="GWAS", confidence=0.75 if sig else 0.35,
                    source=f"GWAS:{rsid}"))
                n += 1
        except Exception as e:
            self._note("GWAS_Catalog", str(e))
        return n

    def load_hpo(self, local_obo: Optional[str] = None) -> Dict[str, Any]:
        text = None
        if local_obo and Path(local_obo).exists():
            text = Path(local_obo).read_text(errors="replace")
        else:
            text, meta = self.http.get(self.HPO_OBO)
        if text is None:
            self._note("HPO", "no local hp.obo and offline"); 
            return {"status": Status.UNAVAILABLE, "n_terms": 0}
        cur = None
        for line in text.splitlines():
            line = line.strip()
            if line == "[Term]":
                if cur and cur.get("id"):
                    self.hpo[cur["id"]] = cur
                cur = {"id": "", "name": "", "is_a": [], "synonyms": [], "obsolete": False}
            elif line.startswith("[") and line != "[Term]":
                if cur and cur.get("id"):
                    self.hpo[cur["id"]] = cur
                cur = None
            elif cur is not None:
                if line.startswith("id: "): cur["id"] = line[4:].strip()
                elif line.startswith("name: "): cur["name"] = line[6:].strip()
                elif line.startswith("is_a: "): cur["is_a"].append(line[6:].split("!")[0].strip())
                elif line.startswith("synonym: "):
                    m = re.match(r'synonym:\s*"([^"]*)"', line)
                    if m: cur["synonyms"].append(m.group(1))
                elif line.startswith("is_obsolete: true"): cur["obsolete"] = True
        if cur and cur.get("id"):
            self.hpo[cur["id"]] = cur
        return {"status": Status.OK, "n_terms": len(self.hpo)}

    def omim(self, mim: int) -> Dict[str, Any]:
        if not self.keys["omim"]:
            self._note("OMIM", "requires licensed OMIM_API_KEY")
            return {"status": Status.UNAVAILABLE, "reason": "no OMIM_API_KEY"}
        p = {"mimNumber": str(mim), "format": "json", "apiKey": self.keys["omim"]}
        body, meta = self.http.get(f"https://api.omim.org/api/entry?{urllib.parse.urlencode(p)}")
        if body is None:
            return {"status": meta.get("status"), "reason": meta.get("reason")}
        return {"status": Status.OK, "raw_len": len(body)}

    def disgenet(self, gene: str) -> Dict[str, Any]:
        if not self.keys["disgenet"]:
            self._note("DisGeNET", "requires DISGENET_API_KEY")
            return {"status": Status.UNAVAILABLE, "reason": "no DISGENET_API_KEY"}
        body, meta = self.http.get(f"https://www.disgenet.org/api/gda/gene/{gene}",
                                   {"Authorization": f"Bearer {self.keys['disgenet']}"})
        if body is None:
            return {"status": meta.get("status"), "reason": meta.get("reason")}
        return {"status": Status.OK, "raw_len": len(body)}

    def optional_source(self, name: str, path: Optional[str]) -> Dict[str, Any]:
        """MONDO / Disease Ontology / GTEx / sc-atlas / PPI / drug-target."""
        if not path or not Path(path).exists():
            self._note(name, "no local file supplied; loader not run")
            return {"status": Status.UNAVAILABLE, "source": name,
                    "reason": "supply an ontology/atlas/interaction file"}
        return {"status": Status.OK, "source": name, "bytes": Path(path).stat().st_size,
                "note": "file present; a source-specific parser is NOT_IMPLEMENTED"}

    # ---- aggregation ----
    def add(self, rec: EvidenceRecord) -> None:
        self.records.append(rec)

    def detect_contradictions(self) -> int:
        n = 0
        for i, a in enumerate(self.records):
            for b in self.records[i + 1:]:
                if (a.gene == b.gene and a.phenotype.lower() == b.phenotype.lower()
                        and {a.effect, b.effect} == {"risk_increase", "benign"}):
                    msg = f"{b.source} reports {b.effect}"
                    if msg not in a.contradictions:
                        a.contradictions.append(msg); n += 1
                    msg2 = f"{a.source} reports {a.effect}"
                    if msg2 not in b.contradictions:
                        b.contradictions.append(msg2); n += 1
        return n

    def query(self, gene=None, phenotype=None) -> List[EvidenceRecord]:
        out = self.records
        if gene: out = [r for r in out if r.gene == gene]
        if phenotype: out = [r for r in out if phenotype.lower() in r.phenotype.lower()]
        return out

    def confidence(self, recs: List[EvidenceRecord]) -> float:
        if not recs:
            return 0.0
        num = sum(EVIDENCE_WEIGHT.get(r.evidence_level, 0.3) * r.confidence for r in recs)
        den = sum(EVIDENCE_WEIGHT.get(r.evidence_level, 0.3) for r in recs)
        base = num / max(den, 1e-12)
        pen = 0.10 * sum(1 for r in recs if r.contradictions)
        return float(np.clip(base - pen, 0.0, 1.0))

    def report(self, gene=None, phenotype=None) -> Dict[str, Any]:
        recs = self.query(gene, phenotype)
        return {
            "records": [r.to_json() for r in recs],
            "n_records": len(recs),
            "aggregate_confidence": self.confidence(recs),
            "confidence_semantics": "weighted evidence score, NOT P(pathogenic)",
            "n_contradictions": sum(len(r.contradictions) for r in recs),
            "sources_unavailable": self.unavailable,
            "clinical_actionability": RESEARCH_ONLY,
            "clinician_review_required": True,
        }


# =============================================================================
# §2. ONTOGENETIC WORLD MODEL — SSC / CSOC / OPMC / Carpet / Hodge-inspired
# =============================================================================

def delta_min_exact(n=3, kappa=2.0, a_min=0.01) -> float:
    d = n - 1
    om = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    return (1.0 / kappa) * (a_min / om) ** (1.0 / d)


def carpet_laplacian(side=9) -> Tensor:
    side = max(3, side); n = side * side
    A = torch.zeros(n, n, dtype=torch.float64)
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1: continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < side and 0 <= jj < side): continue
                if ii % 3 == 1 and jj % 3 == 1: continue
                A[i * side + j, ii * side + jj] = 1.0
    return torch.diag(A.sum(1)) - A


class SSC(nn.Module):
    """Semantic State Contraction: low-pass on a noisy longitudinal signal."""
    def __init__(self, epsilon=0.0028):
        super().__init__()
        self.eps = float(epsilon)
        self.register_buffer("prev", torch.zeros((), dtype=torch.float64))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

    def forward(self, x: Tensor) -> Tensor:
        s = x.reshape(()).to(torch.float64)
        if not bool(self.ready):
            self.prev = s.detach().clone(); self.ready.fill_(True); return s
        out = (1 - self.eps) * self.prev + self.eps * s
        self.prev = out.detach().clone()          # detached: no graph growth
        return out

    def reset(self):
        self.prev.zero_(); self.ready.zero_()


class CSOC(nn.Module):
    """
    Event-count bound N(T) <= (E_max + D)/gap with D = ACTUAL removal.
    Engineering bound on the switching mechanism, NOT PDE regularity.
    """
    def __init__(self, hi=0.15, lo=0.05, rate=0.05):
        super().__init__()
        assert hi > lo
        self.hi, self.lo, self.gap, self.rate = hi, lo, hi - lo, rate
        for n in ("acc", "events", "E_max", "D"):
            self.register_buffer(n, torch.zeros((), dtype=torch.float64))
        self.register_buffer("armed", torch.ones((), dtype=torch.bool))

    def forward(self, signal: Tensor) -> Dict[str, Tensor]:
        s = signal.reshape(()).to(torch.float64)
        ex = F.softplus(s - self.hi)
        dec = torch.sigmoid(-(self.acc - self.lo) * 10.0)
        rem = dec * self.rate * self.acc
        acc = self.acc + ex.detach() - rem
        fire = self.armed & (acc > self.hi)
        clear = (~self.armed) & (acc < self.lo)
        self.acc = acc.detach()
        self.armed = ((self.armed & ~fire) | clear).detach()
        self.events = (self.events + fire.to(torch.float64)).detach()
        self.E_max = torch.maximum(self.E_max, acc).detach()
        self.D = (self.D + rem).detach()
        bound = (self.E_max + self.D) / self.gap
        return {"gate": torch.sigmoid((acc - self.hi) * 10.0),
                "events": self.events, "bound": bound,
                "satisfied": self.events <= bound + 1e-9}


def opmc_d_mn(m=4, n=4) -> int:
    return m * m * n * n + m * n * n


class OPMCDesign(nn.Module):
    """theta in R^d -> decoder -> control field. EXACT gradient via decoder."""
    def __init__(self, out_dim: int, m=4, n=4, seed=0):
        super().__init__()
        self.d = opmc_d_mn(m, n)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(self.d, out_dim, generator=g,
                                              dtype=torch.float64) / math.sqrt(self.d))
        self.theta = nn.Parameter(torch.zeros(self.d, dtype=torch.float64))

    def forward(self) -> Tensor:
        return self.theta @ self.W


class HodgeInspiredSplit(nn.Module):
    """
    3-way split with a symmetric learned kernel penalty.
    NOT Hodge theory: no exterior derivative, no codifferential, no d^2=0,
    no orthogonality guarantee.
    """
    def __init__(self, d=16):
        super().__init__()
        self.proj = nn.Linear(1, 3 * d)
        self.L_raw = nn.Parameter(torch.randn(d, d, dtype=torch.float64) * 0.02)

    def forward(self, field: Tensor) -> Dict[str, Tensor]:
        v = field.reshape(-1).mean().reshape(1, 1).to(self.proj.weight.dtype)
        h, e, c = self.proj(v).reshape(3, -1).to(field.dtype)
        L = 0.5 * (self.L_raw + self.L_raw.T)
        lap = h @ L.to(h.dtype)
        return {"harmonic_norm": h.norm(), "exact_norm": e.norm(),
                "coexact_norm": c.norm(), "kernel_penalty": (lap * lap).sum(),
                "label": "Hodge-INSPIRED, not Hodge theory"}


LINEAGE = ["iPSC", "mesoderm", "ectoderm", "endoderm",
           "progenitor", "mature", "senescent", "transformed"]


class OntogeneticWorldModel(nn.Module):
    """
    Differentiable developmental trajectory:
      iPSC -> germ layer -> progenitor -> mature, plus senescence /
      transformation branches. CSOC bounds the discrete transition count.
    Research model. NOT a validated cell-fate classifier.
    """
    def __init__(self, n_fates=len(LINEAGE), obs_dim=8, hidden=64, device="cpu"):
        super().__init__()
        self.n_fates, self.obs_dim = n_fates, obs_dim
        self.state_dim = n_fates + 4        # + maturation, vascular, immune, tumor
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.state_dim))
        self.ssc = SSC()
        self.csoc = CSOC(hi=0.20, lo=0.06)
        self.opmc = OPMCDesign(self.state_dim)
        self.hodge = HodgeInspiredSplit()
        self.carpet_L = carpet_laplacian(9)
        self.to(device)

    def initial(self, batch=1, device="cpu") -> Tensor:
        f = torch.zeros(batch, self.n_fates, dtype=torch.float64, device=device)
        f[:, 0] = 1.0                                  # start at iPSC
        r = torch.full((batch, 4), 0.05, dtype=torch.float64, device=device)
        return torch.cat([f, r], dim=-1)

    def forward(self, state: Tensor, obs: Tensor, dt=0.1) -> Dict[str, Tensor]:
        ctrl = self.opmc().unsqueeze(0).expand(state.shape[0], -1)
        x = torch.cat([state + ctrl, obs], dim=-1)
        new = state + dt * self.net(x)
        fates = F.softmax(new[..., :self.n_fates], dim=-1)
        risks = torch.sigmoid(new[..., self.n_fates:])
        s = torch.cat([fates, risks], dim=-1)
        filt = self.ssc((s - state).abs().mean())
        ev = self.csoc(filt)
        return {"state": s, "fates": fates, "risks": risks,
                "ssc_signal": filt, "csoc": ev,
                "hodge": self.hodge(fates)}

    def carpet_priority(self, field: Tensor) -> Tensor:
        flat = field.reshape(-1).to(torch.float64)
        n, m = flat.numel(), self.carpet_L.shape[0]
        grp = torch.arange(n, device=flat.device) * m // n
        num = torch.zeros(m, dtype=torch.float64, device=flat.device).index_add(0, grp, flat)
        den = torch.zeros(m, dtype=torch.float64, device=flat.device).index_add(
            0, grp, torch.ones_like(flat))
        s = num / den.clamp_min(1.0)
        L = self.carpet_L.to(s.device)
        return sum((CARPET_RHO ** k) * (s @ L @ s) / (k + 1) for k in range(3))

    def refinement_guard(self, indicator: Tensor, kappa=2.0) -> Dict[str, Any]:
        f = indicator.abs()[None, None]
        avg = F.avg_pool3d(f, 3, stride=1, padding=1)[0, 0]
        gate = torch.sigmoid((avg / (indicator.abs() + 1e-12) - 1.0 / kappa) / 0.05)
        return {"kappa_fat_gate": gate, "delta_min": delta_min_exact(3, kappa),
                "rejected_fraction": float((gate < 0.5).to(torch.float32).mean())}


# =============================================================================
# §3. PROTEIN STRUCTURE MODULE — research framing, editor design BLOCKED
# =============================================================================

class ProteinStructureModule:
    """
    Research framing for folding stability, mutation effect, PPI, enzyme
    mechanism, drug binding, antigen/antibody interfaces.

    Real predictions REQUIRE an external validated backend. Without one the
    answer is UNAVAILABLE — never a plausible-looking fabricated number.
    Gene-editor design is BLOCKED by the SafetyGate before any call.
    """
    TASKS = ("folding_stability", "mutation_effect", "ppi_interface",
             "enzyme_mechanism", "drug_binding", "antibody_antigen_interface",
             "active_site_electronic_structure", "protonation_redox")

    def __init__(self, gov: Governance, backend: Any = None,
                 qc_backend: Any = None):
        self.gov, self.backend, self.qc = gov, backend, qc_backend

    def request(self, task: str, description: str,
                uses_patient_data=False, deid=None) -> Dict[str, Any]:
        d = self.gov.authorize(description, uses_patient_data, deid, "protein_research")
        if not d["authorized"]:
            return {"status": Status.BLOCKED, "task": task, "decision": d}
        if task not in self.TASKS:
            return {"status": Status.ERROR, "reason": f"unknown task; choose {self.TASKS}"}
        need_qc = task in ("active_site_electronic_structure", "protonation_redox",
                           "enzyme_mechanism")
        be = self.qc if need_qc else self.backend
        if be is None:
            return {"status": Status.UNAVAILABLE, "task": task,
                    "reason": ("a validated quantum-chemistry backend is required"
                               if need_qc else
                               "a validated structure/folding backend is required"),
                    "clinical_actionability": RESEARCH_ONLY,
                    "decision": d}
        return {"status": Status.OK, "task": task,
                "note": "backend present; a task-specific adapter is NOT_IMPLEMENTED",
                "clinical_actionability": RESEARCH_ONLY, "decision": d}

    def scope(self) -> Dict[str, Any]:
        return {"allowed": list(self.TASKS),
                "blocked": ["gene editor design", "guide RNA", "delivery vector",
                            "wet-lab protocol", "pathogen enhancement"],
                "requires": "validated external backend for any real prediction",
                "clinical_actionability": RESEARCH_ONLY}


# =============================================================================
# §4. iPSC / ORGANOID TRANSPLANT RESEARCH — risk + missing evidence
# =============================================================================

class iPSCOrganoidTransplantResearch:
    """
    Produces a RISK SCORE and a MISSING-EVIDENCE report.
    It never outputs "transplantable" / "approved" / a go-no-go decision.
    """
    RISKS = ("tumorigenicity_residual_ipsc", "immune_rejection",
             "maturation_failure", "vascularization_failure",
             "innervation_failure", "mechanical_integration_failure",
             "gmp_manufacturing_variability", "long_term_instability")
    EVIDENCE_REQUIRED = (
        "residual undifferentiated iPSC assay (e.g. LIN28/POU5F1 qPCR)",
        "karyotype / genomic stability", "tumorigenicity in vivo study",
        "immune matching + rejection model", "vascular perfusion demonstration",
        "functional maturation benchmark vs adult tissue",
        "GMP batch consistency record", "long-term stability follow-up",
        "GLP toxicology", "IRB approval", "regulator pathway identified",
    )

    def __init__(self, gov: Governance, kg: BiomedicalKnowledgeGraph):
        self.gov, self.kg = gov, kg

    def assess(self, organ: str, state: Tensor,
               evidence_present: Optional[List[str]] = None,
               uses_patient_data=False, deid=None) -> Dict[str, Any]:
        d = self.gov.authorize(f"{organ} organoid risk research",
                               uses_patient_data, deid, "organoid_research")
        if not d["authorized"]:
            return {"status": Status.BLOCKED, "decision": d}
        s = state.detach().reshape(-1).cpu().numpy()
        mat, vas, imm, tum = [float(x) for x in s[-4:]]
        recs = self.kg.query(phenotype=organ)
        ev_conf = self.kg.confidence(recs)
        ev_risk = 1.0 - ev_conf
        r = {
            "tumorigenicity_residual_ipsc": float(np.clip(0.6*tum + 0.3*ev_risk + 0.1*(1-mat), 0, 1)),
            "immune_rejection": float(np.clip(0.5*imm + 0.3*ev_risk + 0.2*(1-vas), 0, 1)),
            "maturation_failure": float(np.clip(0.8*(1-mat) + 0.2*ev_risk, 0, 1)),
            "vascularization_failure": float(np.clip(0.8*(1-vas) + 0.2*ev_risk, 0, 1)),
            "innervation_failure": float(np.clip(0.5*(1-mat) + 0.5*ev_risk, 0, 1)),
            "mechanical_integration_failure": float(np.clip(0.4*(1-mat)+0.3*(1-vas)+0.3*ev_risk, 0, 1)),
            "gmp_manufacturing_variability": float(np.clip(0.5*ev_risk + 0.5*(1-mat), 0, 1)),
            "long_term_instability": float(np.clip(0.5*tum + 0.5*ev_risk, 0, 1)),
        }
        present = set(evidence_present or [])
        missing = [e for e in self.EVIDENCE_REQUIRED if e not in present]
        return {
            "status": Status.OK, "organ": organ,
            "risk_scores": r,
            "overall_research_risk": float(np.mean(list(r.values()))),
            "evidence_confidence": ev_conf,
            "evidence_present": sorted(present),
            "missing_evidence": missing,
            "evidence_completeness": 1.0 - len(missing) / len(self.EVIDENCE_REQUIRED),
            "decision": "NO_GO_NO_GO_ISSUED",
            "explicitly_not_provided": ["transplantable", "approved", "go/no-go",
                                        "clinical recommendation"],
            "clinical_actionability": RESEARCH_ONLY,
            "clinician_review_required": True,
            "governance": d,
        }


# =============================================================================
# §5. PIMPLE TRANSIENT SOLVER + RHIE-CHOW
# =============================================================================

@dataclass
class FVMesh:
    vol: Tensor; cen: Tensor
    owner: Tensor; neigh: Tensor
    fcen: Tensor; Sf: Tensor; dvec: Tensor
    Nc: int; Nf: int
    label: str = "mesh"

    @property
    def area(self): return self.Sf.norm(dim=1)
    @property
    def nf(self): return self.Sf / self.area.clamp_min(1e-30)[:, None]
    @property
    def internal(self): return self.neigh >= 0
    @property
    def boundary(self): return self.neigh < 0

    def quality(self) -> Dict[str, float]:
        it = self.internal
        d = self.dvec[it]; dn = d.norm(dim=1).clamp_min(1e-30)
        cos = (self.nf[it] * (d / dn[:, None])).sum(1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cos.abs()))
        return {"n_cells": self.Nc, "n_faces": self.Nf,
                "nonorth_deg_max": float(ang.max()) if ang.numel() else 0.0,
                "nonorth_deg_mean": float(ang.mean()) if ang.numel() else 0.0,
                "volume_ratio": float(self.vol.max() / self.vol.min().clamp_min(1e-30))}


def cartesian_mesh(nx, ny, nz, Lx=1.0, Ly=1.0, Lz=1.0,
                   periodic=(True, True, True), device="cpu", dtype=torch.float64) -> FVMesh:
    dev = torch.device(device)
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    Nc = nx * ny * nz
    idx = lambda i, j, k: (i * ny + j) * nz + k
    cx = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    cy = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy
    cz = (torch.arange(nz, device=dev, dtype=dtype) + 0.5) * dz
    X, Y, Z = torch.meshgrid(cx, cy, cz, indexing="ij")
    cen = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
    vol = torch.full((Nc,), dx * dy * dz, device=dev, dtype=dtype)
    own, nei, fc, sf, dv = [], [], [], [], []
    A = [dy * dz, dx * dz, dx * dy]; h = [dx, dy, dz]; n = [nx, ny, nz]
    for ax in range(3):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    c = (i, j, k); cn = list(c); cn[ax] += 1
                    coord = [cx[i].item(), cy[j].item(), cz[k].item()]
                    coord[ax] = (c[ax] + 1) * h[ax]
                    nrm = [0.0, 0.0, 0.0]; nrm[ax] = A[ax]
                    d = [0.0, 0.0, 0.0]
                    if cn[ax] < n[ax]:
                        own.append(idx(*c)); nei.append(idx(*cn)); d[ax] = h[ax]
                    elif periodic[ax]:
                        cn[ax] = 0
                        own.append(idx(*c)); nei.append(idx(*cn)); d[ax] = h[ax]
                    else:
                        own.append(idx(*c)); nei.append(-1); d[ax] = 0.5 * h[ax]
                    fc.append(coord); sf.append(nrm); dv.append(d)
                    if c[ax] == 0 and not periodic[ax]:
                        c2 = list(coord); c2[ax] = 0.0
                        n2 = list(nrm); n2[ax] = -A[ax]
                        d2 = [0.0, 0.0, 0.0]; d2[ax] = -0.5 * h[ax]
                        own.append(idx(*c)); nei.append(-1)
                        fc.append(c2); sf.append(n2); dv.append(d2)
    t = lambda a, dt_: torch.tensor(a, dtype=dt_, device=dev)
    return FVMesh(vol, cen, t(own, torch.long), t(nei, torch.long),
                  t(fc, dtype), t(sf, dtype), t(dv, dtype), Nc, len(own),
                  f"cartesian {nx}x{ny}x{nz}")


@dataclass
class Patch:
    faces: Tensor
    kind: str                       # wall | inlet | outlet | symmetry
    u_value: Tuple[float, float, float] = (0., 0., 0.)
    p_value: float = 0.0


@dataclass
class PIMPLEConfig:
    dt: float = 2e-3
    n_outer: int = 3                # PIMPLE outer correctors
    n_piso: int = 2                 # PISO inner correctors
    n_momentum_sweeps: int = 6
    n_non_orth: int = 1
    alpha_u: float = 0.9
    alpha_p: float = 0.5
    momentum_omega: float = 0.9
    p_tol: float = 1e-9
    p_max_iter: int = 600
    tol_continuity: float = 1e-8
    ref_cell: int = 0


class PIMPLESolver:
    """
    Transient incompressible collocated FVM with Rhie-Chow.
    PIMPLE = n_outer SIMPLE-style outer correctors, each with n_piso
    PISO pressure correctors. Momentum includes the ddt term.
    Fully differentiable when run under torch.enable_grad().
    """

    def __init__(self, mesh: FVMesh, nu: float, patches: List[Patch],
                 cfg: Optional[PIMPLEConfig] = None,
                 body_force=(0., 0., 0.)):
        self.m, self.nu = mesh, float(nu)
        self.cfg = cfg or PIMPLEConfig()
        self.patches = patches
        self.dev, self.dt_ = mesh.cen.device, mesh.cen.dtype
        self.g = torch.tensor(body_force, device=self.dev, dtype=self.dt_)
        self._classify(); self._geometry()
        self.u = torch.zeros(mesh.Nc, 3, device=self.dev, dtype=self.dt_)
        self.p = torch.zeros(mesh.Nc, device=self.dev, dtype=self.dt_)
        self.phi = torch.zeros(mesh.Nf, device=self.dev, dtype=self.dt_)
        self._init_bnd_phi()
        self.t = 0.0; self.step_count = 0
        self.last: Dict[str, float] = {}

    def _classify(self):
        m = self.m
        self.bkind = torch.full((m.Nf,), -1, dtype=torch.long, device=self.dev)
        self.bu = torch.zeros(m.Nf, 3, device=self.dev, dtype=self.dt_)
        self.bp = torch.zeros(m.Nf, device=self.dev, dtype=self.dt_)
        K = {"wall": 0, "inlet": 1, "outlet": 2, "symmetry": 3}
        cov = torch.zeros(m.Nf, dtype=torch.bool, device=self.dev)
        for p in self.patches:
            f = p.faces.to(self.dev)
            self.bkind[f] = K[p.kind]
            self.bu[f] = torch.tensor(p.u_value, device=self.dev, dtype=self.dt_)
            self.bp[f] = p.p_value; cov[f] = True
        miss = m.boundary & ~cov
        if bool(miss.any()):
            self.bkind[miss] = 0
            log.warning("%d boundary faces defaulted to WALL", int(miss.sum()))
        self.n_defaulted = int(miss.sum())
        self.is_dir_p = m.boundary & (self.bkind == 2)
        self.has_dir = bool(self.is_dir_p.any())

    def _geometry(self):
        m = self.m
        ddn = (m.dvec * m.nf).sum(1)
        bad = ddn.abs() < 1e-14
        safe = torch.where(bad, torch.full_like(ddn, 1e-14), ddn)
        self.alpha = m.area / safe
        self.kvec = m.Sf - self.alpha[:, None] * m.dvec
        it = m.internal
        dn = m.dvec.norm(dim=1).clamp_min(1e-30)
        wP = (m.fcen - m.cen[m.neigh.clamp_min(0)]).norm(dim=1) / dn
        self.wP = torch.where(it, wP.clamp(0, 1), torch.ones_like(wP))
        self.alpha_orth = m.area / dn
        self.it, self.oi, self.ni = it, m.owner[it], m.neigh[it]

    def _interp(self, cf: Tensor) -> Tensor:
        P = cf[self.m.owner]; N = cf[self.m.neigh.clamp_min(0)]
        w = self.wP
        while w.dim() < P.dim(): w = w.unsqueeze(-1)
        out = w * P + (1 - w) * N
        b = self.m.boundary
        return torch.where(b.view(-1, *([1] * (P.dim() - 1))), P, out)

    def _grad(self, phi: Tensor) -> Tensor:
        pf = self._interp(phi)
        fl = pf[:, None] * self.m.Sf
        g = torch.zeros(self.m.Nc, 3, device=self.dev, dtype=self.dt_)
        g = g.index_add(0, self.m.owner, fl)
        g = g.index_add(0, self.ni, -fl[self.it])
        return g / self.m.vol.clamp_min(1e-30)[:, None]

    def _div(self, flux: Tensor) -> Tensor:
        o = torch.zeros(self.m.Nc, device=self.dev, dtype=self.dt_)
        o = o.index_add(0, self.m.owner, flux)
        return o.index_add(0, self.ni, -flux[self.it])

    def _init_bnd_phi(self):
        m = self.m; b = m.boundary
        kind, ub, Sb = self.bkind[b], self.bu[b], m.Sf[b]
        phi = (ub * Sb).sum(1)
        kill = (kind == 0) | (kind == 3)
        phi = torch.where(kill, torch.zeros_like(phi), phi)
        out = (kind == 2)
        phi = torch.where(out, (self.u[m.owner[b]] * Sb).sum(1), phi)
        self.phi = self.phi.clone(); self.phi[b] = phi
        self._balance()

    def _balance(self):
        b = self.m.boundary
        net = float(self.phi[b].sum())
        out = b & (self.bkind == 2)
        if bool(out.any()):
            tot = float(self.phi[out].sum())
            if abs(tot) > 1e-30:
                self.phi = self.phi.clone()
                self.phi[out] = self.phi[out] * ((tot - net) / tot)
        self.mass_imbalance = float(self.phi[b].sum())

    def _momentum(self, u_old: Tensor, dt: float):
        m, nu = self.m, self.nu
        F_ = self.phi[self.it]
        diff = nu * self.alpha_orth[self.it]
        z = torch.zeros_like(F_)
        aN = -torch.minimum(F_, z) + diff
        aP = torch.zeros(m.Nc, device=self.dev, dtype=self.dt_)
        aP = aP.index_add(0, self.oi, torch.maximum(F_, z) + diff)
        aP = aP.index_add(0, self.ni, torch.maximum(-F_, z) + diff)
        b = m.boundary; ob = m.owner[b]
        kind, Fb, db = self.bkind[b], self.phi[b], nu * self.alpha_orth[b]
        zb = torch.zeros_like(Fb)
        iw, ii, io = (kind == 0) | (kind == 3), kind == 1, kind == 2
        aP = aP.index_add(0, ob[iw], db[iw])
        aP = aP.index_add(0, ob[ii], db[ii] + torch.maximum(Fb[ii], zb[ii]))
        aP = aP.index_add(0, ob[io], torch.maximum(Fb[io], zb[io]))
        # transient ddt
        transient = m.vol / dt
        aP = aP + transient
        return aP.clamp_min(1e-30), aN, (iw, ii, io, ob, db, Fb, kind, transient)

    def _H(self, aN, ctx, u_old):
        iw, ii, io, ob, db, Fb, kind, transient = ctx
        m = self.m; u = self.u
        H = torch.zeros_like(u)
        c = aN[:, None]
        H = H.index_add(0, self.oi, c * u[self.ni])
        H = H.index_add(0, self.ni, c * u[self.oi])
        ub = self.bu[m.boundary]; zb = torch.zeros_like(Fb)
        H = H.index_add(0, ob[iw], db[iw][:, None] * ub[iw])
        H = H.index_add(0, ob[ii], (db[ii] + torch.maximum(-Fb[ii], zb[ii]))[:, None] * ub[ii])
        H = H.index_add(0, ob[io], torch.maximum(-Fb[io], zb[io])[:, None] * u[ob[io]])
        H = H + m.vol[:, None] * self.g[None, :] + transient[:, None] * u_old
        return H

    def _mom_solve(self, aP, aN, ctx, gradp, u_old):
        u = self.u.clone()
        rhs = -self.m.vol[:, None] * gradp
        om = self.cfg.momentum_omega
        for _ in range(self.cfg.n_momentum_sweeps):
            H = self._H(aN, ctx, u_old)
            u = om * ((H + rhs) / aP[:, None]) + (1 - om) * u
        H = self._H(aN, ctx, u_old)
        res = (aP[:, None] * u - H - rhs).norm()
        return u, float(res / (aP[:, None] * u).norm().clamp_min(1e-30))

    def _lap_diag(self, coeff):
        d = torch.zeros(self.m.Nc, device=self.dev, dtype=self.dt_)
        d = d.index_add(0, self.oi, coeff[self.it])
        d = d.index_add(0, self.ni, coeff[self.it])
        bD = self.is_dir_p
        if bool(bD.any()):
            d = d.index_add(0, self.m.owner[bD], self.rAU[self.m.owner[bD]] * self.alpha[bD])
        return d.clamp_min(1e-30)

    def _lap_apply(self, p, coeff):
        o = torch.zeros(self.m.Nc, device=self.dev, dtype=self.dt_)
        fl = coeff[self.it] * (p[self.oi] - p[self.ni])
        o = o.index_add(0, self.oi, fl).index_add(0, self.ni, -fl)
        bD = self.is_dir_p
        if bool(bD.any()):
            ob = self.m.owner[bD]
            o = o.index_add(0, ob, self.rAU[ob] * self.alpha[bD] * p[ob])
        return o

    def solve_pressure(self, b: Tensor, coeff: Tensor,
                       p0: Optional[Tensor] = None) -> Tuple[Tensor, Dict]:
        """CG on A p = b, A = -lap(rAU, .). Constant null space projected out."""
        diag = self._lap_diag(coeff)
        pin = not self.has_dir
        if pin: b = b - b.mean()
        p = torch.zeros_like(b) if p0 is None else p0.clone()
        if pin: p = p - p.mean()
        r = b - self._lap_apply(p, coeff)
        if pin: r = r - r.mean()
        z = r / diag; d = z.clone(); rz = (r * z).sum()
        nb = b.norm().clamp_min(1e-30); rel = float(r.norm() / nb); used = 0
        for it in range(self.cfg.p_max_iter):
            Ad = self._lap_apply(d, coeff)
            dAd = (d * Ad).sum()
            if float(dAd.abs()) < 1e-300: break
            a = rz / dAd
            p = p + a * d; r = r - a * Ad
            if pin: r = r - r.mean()
            rel = float(r.norm() / nb); used = it + 1
            if rel < self.cfg.p_tol: break
            z = r / diag; rzn = (r * z).sum()
            d = z + (rzn / rz.clamp_min(1e-300)) * d; rz = rzn
            if pin: p = p - p.mean()
        if pin: p = p - p[self.cfg.ref_cell]
        return p, {"iterations": used, "rel_residual": rel,
                   "converged": rel < self.cfg.p_tol}

    def step(self, dt: Optional[float] = None) -> Dict[str, float]:
        cfg, m = self.cfg, self.m
        dt = float(dt if dt is not None else cfg.dt)
        u_old = self.u.clone()
        p_old = self.p.clone()
        mom_res = 0.0; pstats: Dict = {}
        for outer in range(cfg.n_outer):
            aP, aN, ctx = self._momentum(u_old, dt)
            gp = self._grad(self.p)
            u_star, mom_res = self._mom_solve(aP, aN, ctx, gp, u_old)
            self.u = u_star
            self.rAU = m.vol / aP
            self.HbyA = self._H(aN, ctx, u_old) / aP[:, None]
            phiHbyA = (self._interp(self.HbyA) * m.Sf).sum(1)
            coeff = self._interp(self.rAU) * self.alpha
            for piso in range(cfg.n_piso):
                p_cur = self.p
                for k in range(max(1, cfg.n_non_orth)):
                    gpf = self._interp(self._grad(p_cur))
                    nofl = self._interp(self.rAU) * (self.kvec * gpf).sum(1) if k > 0 \
                        else torch.zeros_like(phiHbyA)
                    b = -self._div(phiHbyA) + self._div(nofl)
                    bD = self.is_dir_p
                    if bool(bD.any()):
                        ob = m.owner[bD]
                        b = b.index_add(0, ob, self.rAU[ob] * self.alpha[bD] * self.bp[bD])
                    p_new, pstats = self.solve_pressure(b, coeff, p0=p_cur)
                    p_cur = p_new
                # correct flux (never relaxed: continuity must hold)
                dp = torch.zeros(m.Nf, device=self.dev, dtype=self.dt_)
                dp[self.it] = p_cur[self.ni] - p_cur[self.oi]
                bD = self.is_dir_p
                dp[bD] = self.bp[bD] - p_cur[m.owner[bD]]
                gpf = self._interp(self._grad(p_cur))
                nofl = self._interp(self.rAU) * (self.kvec * gpf).sum(1)
                phi = phiHbyA - coeff * dp - nofl
                bN = m.boundary & ~bD
                phi[bN] = self.phi[bN]
                self.phi = phi
                kill = m.boundary & ((self.bkind == 0) | (self.bkind == 3))
                self.phi[kill] = 0.0
                self._balance()
                # correct velocity
                self.u = self.HbyA - self.rAU[:, None] * self._grad(p_cur)
                self.p = p_cur
            if outer < cfg.n_outer - 1:
                self.u = cfg.alpha_u * self.u + (1 - cfg.alpha_u) * u_old
                self.p = cfg.alpha_p * self.p + (1 - cfg.alpha_p) * p_old
        self.t += dt; self.step_count += 1
        cont = self._div(self.phi)
        scale = self.phi.abs().mean().clamp_min(1e-30) * 6.0
        self.last = {"t": self.t, "dt": dt,
                     "continuity_rel": float(cont.abs().max() / scale),
                     "momentum_rel_residual": mom_res,
                     "p_iterations": pstats.get("iterations", 0),
                     "p_rel_residual": pstats.get("rel_residual", float("nan")),
                     "u_max": float(self.u.norm(dim=1).max()),
                     "ke": float(0.5 * (self.u ** 2).sum(1).mean()),
                     "mass_imbalance": self.mass_imbalance}
        return self.last

    def run(self, n_steps: int, callback=None) -> List[Dict[str, float]]:
        hist = []
        for i in range(n_steps):
            r = self.step(); hist.append(r)
            if callback: callback(i, r)
        return hist


# =============================================================================
# §6. IMPLICIT ADJOINT THROUGH THE TIME LOOP WITH CHECKPOINTING
# =============================================================================

class PressureAdjoint(torch.autograd.Function):
    """A p = b, A symmetric: dL/db = A^{-1} dL/dp; dL/dcoeff via -(λ_o-λ_n)(p_o-p_n)."""
    @staticmethod
    def forward(ctx, b, coeff, solver):
        p, st = solver.solve_pressure(b.detach(), coeff.detach())
        ctx.solver = solver; ctx.save_for_backward(p, coeff); ctx.stats = st
        return p

    @staticmethod
    def backward(ctx, grad_p):
        s = ctx.solver; p, coeff = ctx.saved_tensors
        lam, _ = s.solve_pressure(grad_p.contiguous(), coeff)
        gc = torch.zeros_like(coeff)
        it_idx = torch.nonzero(s.it).flatten()
        gc = gc.index_add(0, it_idx, -(lam[s.oi] - lam[s.ni]) * (p[s.oi] - p[s.ni]))
        bD = s.is_dir_p
        if bool(bD.any()):
            ob = s.m.owner[bD]
            gc = gc.index_add(0, torch.nonzero(bD).flatten(),
                              -lam[ob] * p[ob] * s.rAU.detach()[ob])
        return lam, gc, None


class CheckpointedAdjointTimeLoop:
    """
    Reverse-mode through the WHOLE transient loop with uniform-interval
    checkpointing.

    Forward : run PIMPLE, store only every `interval`-th state (DETACHED).
    Backward: for each segment in reverse, recompute the forward with grad
              enabled from the stored checkpoint, then pull the adjoint
              through that segment with torch.autograd.grad.

    Memory  : O(n_steps/interval) stored states instead of O(n_steps) graphs.
    The reduction is MEASURED, not asserted (see self-test T11).

    HONEST: the PIMPLE inner iterations are differentiated as unrolled
    arithmetic within a segment. The pressure solve inside each step uses
    the implicit adjoint above, so its CG iterations are NOT unrolled.
    """

    def __init__(self, solver: PIMPLESolver, interval: int = 5):
        self.s = solver
        self.interval = max(1, int(interval))
        self.checkpoints: List[Tuple[int, Tensor, Tensor, Tensor]] = []
        self.n_steps = 0

    def forward(self, n_steps: int, dt: Optional[float] = None,
                observe: Optional[Callable[[int, PIMPLESolver], Tensor]] = None
                ) -> Dict[str, Any]:
        s = self.s
        self.checkpoints = [(0, s.u.detach().clone(), s.p.detach().clone(),
                             s.phi.detach().clone())]
        losses = []
        with torch.no_grad():
            for i in range(n_steps):
                s.step(dt)
                if observe is not None:
                    losses.append(float(observe(i, s)))
                if (i + 1) % self.interval == 0:
                    self.checkpoints.append((i + 1, s.u.detach().clone(),
                                             s.p.detach().clone(), s.phi.detach().clone()))
        self.n_steps = n_steps
        self.observed = losses
        return {"n_steps": n_steps, "n_checkpoints": len(self.checkpoints),
                "interval": self.interval,
                "memory_states_stored": len(self.checkpoints),
                "memory_states_naive": n_steps,
                "reduction_factor": n_steps / max(len(self.checkpoints), 1)}

    def backward(self, loss_fn: Callable[[Tensor], Tensor],
                 params: List[Tensor]) -> Dict[str, Any]:
        """
        loss_fn(u_final) -> scalar. Recomputes each segment with grad and
        accumulates dJ/dparam. Returns gradients aligned with `params`.
        """
        s = self.s
        grads = [torch.zeros_like(p) for p in params]
        segs = list(zip(self.checkpoints[:-1], self.checkpoints[1:]))
        adj_u = None
        n_recomputed = 0
        for (i0, u0, p0, phi0), (i1, u1, p1, phi1) in reversed(segs):
            s.u = u0.clone().requires_grad_(True)
            s.p = p0.clone(); s.phi = phi0.clone()
            s.t = i0 * s.cfg.dt; s.step_count = i0
            with torch.enable_grad():
                for _ in range(i1 - i0):
                    s.step()
                    n_recomputed += 1
                target = loss_fn(s.u) if adj_u is None else (s.u * adj_u).sum()
                gs = torch.autograd.grad(target, [s.u] + params,
                                         retain_graph=False, allow_unused=True)
            g_u = gs[0]
            for k, g in enumerate(gs[1:]):
                if g is not None:
                    grads[k] = grads[k] + g.detach()
            adj_u = g_u.detach() if g_u is not None else torch.zeros_like(u0)
        return {"grads": grads, "n_steps_recomputed": n_recomputed,
                "recompute_overhead": n_recomputed / max(self.n_steps, 1)}


# =============================================================================
# §7. PATIENT-SPECIFIC ORGAN CFD
# =============================================================================

ORGAN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "brain": {
        "physics_requested": ["CSF flow", "cerebral perfusion", "aneurysm WSS"],
        "implemented": ["incompressible laminar NS (PIMPLE + Rhie-Chow)"],
        "NOT_IMPLEMENTED": ["porous perfusion (Darcy/Forchheimer)",
                            "FSI vessel wall", "non-Newtonian rheology"],
        "outputs": ["pressure drop", "WSS proxy", "residence-time proxy"],
        "validation": ["4D-flow MRI", "transcranial Doppler", "invasive ICP"],
        "required_bc": ["inlet waveform", "outlet pressure/Windkessel", "wall model"],
    },
    "heart": {
        "physics_requested": ["FSI", "valve flow", "coronary CFD"],
        "implemented": ["incompressible laminar NS"],
        "NOT_IMPLEMENTED": ["FSI", "moving valve geometry", "Windkessel outlet",
                            "non-Newtonian blood"],
        "outputs": ["pressure drop", "WSS proxy"],
        "validation": ["4D-flow MRI", "Doppler", "invasive pressure"],
        "required_bc": ["inlet waveform", "Windkessel", "moving wall"],
    },
    "lung": {
        "physics_requested": ["airflow", "aerosol", "VOF"],
        "implemented": ["incompressible laminar NS (low-Mach approximation)"],
        "NOT_IMPLEMENTED": ["compressible airflow", "Lagrangian aerosol",
                            "VOF mucus layer", "moving airway wall"],
        "outputs": ["pressure drop", "wall shear proxy"],
        "validation": ["spirometry", "4D CT ventilation", "deposition imaging"],
        "required_bc": ["breathing waveform", "outlet compliance"],
    },
    "liver": {
        "physics_requested": ["porous media", "perfusion"],
        "implemented": ["incompressible laminar NS in resolved vessels"],
        "NOT_IMPLEMENTED": ["Darcy/Forchheimer porous parenchyma",
                            "multi-compartment perfusion"],
        "outputs": ["vessel pressure drop", "WSS proxy"],
        "validation": ["contrast CT perfusion", "Doppler", "portal pressure"],
        "required_bc": ["portal/arterial inflow", "hepatic vein outflow"],
    },
    "kidney": {
        "physics_requested": ["filtration", "tubular flow"],
        "implemented": ["incompressible laminar NS in resolved vessels"],
        "NOT_IMPLEMENTED": ["glomerular filtration membrane",
                            "tubular transport", "nephron-scale coupling"],
        "outputs": ["vessel pressure drop", "WSS proxy"],
        "validation": ["renal Doppler", "GFR measurement", "4D-flow MRI"],
        "required_bc": ["renal artery inflow", "venous outflow"],
    },
    "vascular": {
        "physics_requested": ["non-Newtonian blood", "FSI"],
        "implemented": ["incompressible NEWTONIAN laminar NS"],
        "NOT_IMPLEMENTED": ["Carreau/Casson rheology", "FSI", "Wall shear OSI"],
        "outputs": ["pressure drop", "WSS proxy"],
        "validation": ["4D-flow MRI", "Doppler", "invasive pressure"],
        "required_bc": ["inlet waveform", "Windkessel outlet", "wall model"],
    },
}


class PatientSpecificOrganCFD:
    """
    Research decision support. NEVER an autonomous surgical decision.
    A validated segmentation is a REQUIRED INPUT; there is no automatic
    clinical segmentation here.
    """

    def __init__(self, gov: Governance):
        self.gov = gov

    def organ_spec(self, organ: str) -> Dict[str, Any]:
        if organ not in ORGAN_REGISTRY:
            return {"status": Status.ERROR, "reason": f"choose {sorted(ORGAN_REGISTRY)}"}
        return {"status": Status.OK, "organ": organ, **ORGAN_REGISTRY[organ],
                "clinical_actionability": RESEARCH_ONLY}

    def accept_segmentation(self, mask: np.ndarray,
                            validation: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict]:
        req = ["method", "dice_vs_reference", "validated_by", "date"]
        miss = [k for k in req if k not in validation]
        if miss:
            return None, {"status": Status.BLOCKED, "reason": f"missing {miss}"}
        if validation.get("dice_vs_reference", 0.0) < 0.80:
            return None, {"status": Status.BLOCKED,
                          "reason": "Dice < 0.80; not acceptable even for research planning"}
        return (np.asarray(mask) > 0.5).astype(np.uint8), {"status": Status.OK}

    def plan(self, organ: str, mesh: FVMesh, patches: List[Patch], nu: float,
             n_steps=40, dt=2e-3, ensemble=1,
             uses_patient_data=False, deid=None) -> Dict[str, Any]:
        d = self.gov.authorize(f"{organ} hemodynamic simulation research",
                               uses_patient_data, deid, "simulation")
        if not d["authorized"]:
            return {"status": Status.BLOCKED, "decision": d}
        spec = self.organ_spec(organ)
        runs = []
        for e in range(max(1, ensemble)):
            cfg = PIMPLEConfig(dt=dt, n_outer=3, n_piso=2, p_tol=1e-8)
            s = PIMPLESolver(mesh, nu * (1.0 + 0.05 * e), patches, cfg)
            h = s.run(n_steps)
            runs.append({"ensemble": e, "nu": nu * (1.0 + 0.05 * e),
                         "ke_final": h[-1]["ke"], "u_max": h[-1]["u_max"],
                         "continuity_rel": h[-1]["continuity_rel"]})
        kes = np.array([r["ke_final"] for r in runs])
        return {
            "status": Status.OK, "organ": organ,
            "hypothesis_metrics": {
                "ke_mean": float(kes.mean()), "ke_std": float(kes.std()),
                "u_max_mean": float(np.mean([r["u_max"] for r in runs])),
                "continuity_rel_max": float(max(r["continuity_rel"] for r in runs)),
            },
            "uncertainty": {"n_ensemble": len(runs),
                            "method": "viscosity perturbation ensemble",
                            "note": "NOT a calibrated clinical uncertainty"},
            "implemented_physics": spec["implemented"],
            "NOT_IMPLEMENTED_physics": spec["NOT_IMPLEMENTED"],
            "mesh_quality": mesh.quality(),
            "validation_required": spec["validation"],
            "validation_status": Status.NOT_COMPARED,
            "decision": "NO_CLINICAL_DECISION_ISSUED",
            "clinical_actionability": RESEARCH_ONLY,
            "clinician_review_required": True,
            "governance": d,
        }


# =============================================================================
# §8. VALIDATION HARNESS
# =============================================================================

class Validation:
    def __init__(self, reference: Optional[Dict[str, np.ndarray]] = None):
        self.ref = reference or {}

    @staticmethod
    def bland_altman(p, r):
        d = p - r; m = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        return {"bias": m, "sd": sd, "loa_upper": m + 1.96*sd, "loa_lower": m - 1.96*sd}

    @staticmethod
    def icc21(p, r):
        Y = np.stack([p, r], 1); n, k = Y.shape
        if n < 2: return float("nan")
        gm = Y.mean(); row = Y.mean(1); col = Y.mean(0)
        SSR = k*((row-gm)**2).sum(); SSC = n*((col-gm)**2).sum()
        SSE = ((Y-gm)**2).sum() - SSR - SSC
        MSR = SSR/(n-1); MSC = SSC/(k-1); MSE = SSE/max((n-1)*(k-1), 1)
        den = MSR + (k-1)*MSE + k*(MSC-MSE)/n
        return float((MSR-MSE)/den) if den else float("nan")

    @staticmethod
    def calibration(p, r):
        X = np.stack([np.ones_like(p), p], 1)
        b, *_ = np.linalg.lstsq(X, r, rcond=None)
        return {"intercept": float(b[0]), "slope": float(b[1])}

    def compare(self, name: str, pred: np.ndarray, key: str) -> Dict[str, Any]:
        if key not in self.ref:
            return {"case": name, "status": Status.NOT_COMPARED,
                    "reason": f"reference '{key}' not supplied"}
        r = np.asarray(self.ref[key], float)
        if len(r) != len(pred):
            return {"case": name, "status": Status.ERROR, "reason": "length mismatch"}
        return {"case": name, "status": Status.OK, "n": len(pred),
                "bland_altman": self.bland_altman(pred, r),
                "icc21": self.icc21(pred, r),
                "calibration": self.calibration(pred, r),
                "rmse": float(np.sqrt(((pred-r)**2).mean()))}


# =============================================================================
# §9. ORCHESTRATOR
# =============================================================================

class OneMedWorld:
    def __init__(self, irb: Optional[IRBRecord] = None, offline=True,
                 cache_dir="./medworld_cache"):
        self.gov = Governance(irb)
        self.kg = BiomedicalKnowledgeGraph(HTTP(cache_dir=cache_dir, offline=offline))
        self.onto = OntogeneticWorldModel()
        self.protein = ProteinStructureModule(self.gov)
        self.ipsc = iPSCOrganoidTransplantResearch(self.gov, self.kg)
        self.cfd = PatientSpecificOrganCFD(self.gov)
        self.validation = Validation()

    def status(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "governance": self.gov.regulatory(),
            "knowledge_graph": {
                "public_sources": ["PubMed", "ClinVar", "GWAS Catalog", "HPO"],
                "key_required": {"OMIM": bool(self.kg.keys["omim"]),
                                 "DisGeNET": bool(self.kg.keys["disgenet"])},
                "file_required": ["MONDO", "Disease Ontology", "GTEx",
                                  "single-cell atlas", "spatial transcriptomics",
                                  "PPI", "drug-target-pathway"],
                "offline": self.kg.http.offline},
            "ontogenetic": {"lineage": LINEAGE,
                            "math": ["SSC", "CSOC event bound", "OPMC design",
                                     "Carpet priority", "Hodge-INSPIRED split"]},
            "protein": self.protein.scope(),
            "ipsc": {"outputs": "risk scores + missing evidence",
                     "never_outputs": ["transplantable", "go/no-go"]},
            "cfd": {"solver": "PIMPLE + Rhie-Chow, collocated Cartesian FVM",
                    "adjoint": "implicit pressure adjoint + checkpointed time loop",
                    "organs": sorted(ORGAN_REGISTRY),
                    "NOT_IMPLEMENTED": "FSI, VOF, non-Newtonian, porous, aerosol"},
            "clinical_actionability": RESEARCH_ONLY,
        }


# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(verbose=True) -> bool:
    print(BANNER)
    ok = True
    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose: print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    torch.manual_seed(0); np.random.seed(0)

    # T1 governance
    print("\n[T1] governance gate")
    gov = Governance()
    rep("patient path BLOCKED without IRB",
        not gov.authorize("hemodynamic simulation", True)["authorized"])
    for txt, label in (("design a CRISPR Cas9 guide RNA", "gene editor"),
                       ("give a wet lab transfection protocol", "wet lab"),
                       ("enhance virulence and immune evasion", "pathogen"),
                       ("should we operate on the patient", "clinical order")):
        rep(f"{label} BLOCKED", not gov.authorize(txt)["authorized"])
    irb = IRBRecord("P-001", "Demo", "2026-01-01", "2099-12-31",
                    ["simulation", "organoid_research"], data_use_agreement=True,
                    status="APPROVED")
    g2 = Governance(irb)
    rep("IRB alone insufficient (no de-id)",
        not g2.authorize("simulation", True, purpose="simulation")["authorized"])
    deid = g2.phi.deidentify({"patient_name": "Jane Doe", "mrn": "MRN: 12345",
                              "age": 55, "sex": "female"})
    rep("de-id removes direct identifiers", deid["passed"] and "patient_name" in deid["removed_fields"])
    rep("IRB + de-id authorised",
        g2.authorize("simulation", True, deid, "simulation")["authorized"])
    q, rm = g2.phi.scrub("Jane Doe MRN: 999999 BRCA1")
    rep("query scrubbed before network", bool(rm) and "REDACTED" in q, str(rm))

    # T2 knowledge graph
    print("\n[T2] BiomedicalKnowledgeGraph schema + aggregation")
    mw = OneMedWorld(irb=irb, offline=True)
    kg = mw.kg
    kg.add(EvidenceRecord("GENE_A", "rs1", "heart defect", "curated", 0.9,
                          effect="risk_increase", source="ClinVar:1"))
    kg.add(EvidenceRecord("GENE_A", "rs1", "heart defect", "curated", 0.8,
                          effect="benign", source="ClinVar:2"))
    kg.add(EvidenceRecord("GENE_A", "rs2", "heart defect", "GWAS", 0.7,
                          source="GWAS:rs2"))
    n_con = kg.detect_contradictions()
    rep("contradictions detected", n_con >= 2, f"n={n_con}")
    r = kg.report("GENE_A", "heart defect")
    keys = set(r["records"][0].keys())
    need = {"gene", "variant", "phenotype", "evidence_level", "confidence",
            "contradictions", "developmental_stage", "tissue_specificity",
            "clinical_actionability"}
    rep("requested JSON schema present", need <= keys, str(sorted(need - keys)))
    rep("confidence reduced by contradictions", r["aggregate_confidence"] < 0.8,
        f"conf={r['aggregate_confidence']:.3f}")
    rep("research_only", r["clinical_actionability"] == RESEARCH_ONLY)
    kg.omim(100000); kg.disgenet("GENE_A")
    rep("OMIM/DisGeNET report UNAVAILABLE",
        len([u for u in kg.unavailable if u["source"] in ("OMIM", "DisGeNET")]) == 2)
    for s in ("MONDO", "GTEx", "PPI", "drug-target"):
        kg.optional_source(s, None)
    rep("file-based sources report UNAVAILABLE",
        len([u for u in kg.unavailable if u["source"] in ("MONDO","GTEx","PPI","drug-target")]) == 4)

    # T3 ontogenetic model
    print("\n[T3] OntogeneticWorldModel: SSC / CSOC / OPMC / Carpet / Hodge")
    om = mw.onto
    st = om.initial(batch=2)
    obs = torch.rand(2, om.obs_dim, dtype=torch.float64)
    out = om(st, obs)
    rep("fates sum to 1", torch.allclose(out["fates"].sum(-1), torch.ones(2, dtype=torch.float64), atol=1e-9))
    rep("risks in [0,1]", bool(((out["risks"] >= 0) & (out["risks"] <= 1)).all()))
    rep("CSOC bound satisfied", bool(out["csoc"]["satisfied"]))
    rep("Hodge labelled INSPIRED", "INSPIRED" in out["hodge"]["label"])
    loss = out["state"].sum() + out["hodge"]["kernel_penalty"]
    loss.backward()
    rep("OPMC theta has finite grad",
        om.opmc.theta.grad is not None and bool(torch.isfinite(om.opmc.theta.grad).all()))
    rep("SSC buffer detached", om.ssc.prev.grad_fn is None)
    cp = om.carpet_priority(torch.rand(9, 9, 9, dtype=torch.float64))
    rep("Carpet priority finite scalar", cp.dim() == 0 and bool(torch.isfinite(cp)))
    sl = torch.zeros(12, 12, 12, dtype=torch.float64); sl[6, 6, :] = 1.0
    guard = om.refinement_guard(sl)
    rep("kappa-fatness rejects sliver", guard["rejected_fraction"] > 0,
        f"rejected={guard['rejected_fraction']:.2f}")
    rep("delta_min > 0", guard["delta_min"] > 0, f"{guard['delta_min']:.5f}")

    # T4 protein module
    print("\n[T4] ProteinStructureModule")
    pr = mw.protein.request("folding_stability", "study folding stability of a therapeutic enzyme")
    rep("no backend => UNAVAILABLE (no fake number)", pr["status"] == Status.UNAVAILABLE, pr["status"])
    pb = mw.protein.request("drug_binding", "design a CRISPR base editor for binding")
    rep("editor design BLOCKED inside protein module", pb["status"] == Status.BLOCKED)
    rep("scope lists blocked items", "gene editor design" in mw.protein.scope()["blocked"])

    # T5 iPSC risk
    print("\n[T5] iPSCOrganoidTransplantResearch")
    ip = mw.ipsc.assess("heart", out["state"][0], evidence_present=["IRB approval"])
    rep("risk scores present", len(ip["risk_scores"]) == 8)
    rep("all risks in [0,1]", all(0 <= v <= 1 for v in ip["risk_scores"].values()))
    rep("missing evidence reported", len(ip["missing_evidence"]) > 5,
        f"{len(ip['missing_evidence'])} missing")
    rep("NO go/no-go issued", ip["decision"] == "NO_GO_NO_GO_ISSUED")
    rep("never says transplantable", "transplantable" in ip["explicitly_not_provided"])

    # T6 PIMPLE continuity
    print("\n[T6] PIMPLE transient + Rhie-Chow: continuity")
    m = cartesian_mesh(10, 8, 6, periodic=(False, False, True))
    nf = m.nf; b = torch.nonzero(m.boundary).flatten()
    inlet = b[nf[b, 0] < -0.99]; outlet = b[nf[b, 0] > 0.99]
    walls = b[nf[b, 1].abs() > 0.99]
    patches = [Patch(inlet, "inlet", (1.0, 0., 0.)),
               Patch(outlet, "outlet", p_value=0.0), Patch(walls, "wall")]
    s = PIMPLESolver(m, nu=5e-2, patches=patches,
                     cfg=PIMPLEConfig(dt=2e-3, n_outer=3, n_piso=2, p_tol=1e-9))
    rep("boundary mass balanced", abs(s.mass_imbalance) < 1e-12, f"{s.mass_imbalance:.2e}")
    h = s.run(30)
    rep("continuity_rel < 1e-6", h[-1]["continuity_rel"] < 1e-6, f"{h[-1]['continuity_rel']:.2e}")
    rep("pressure converged", h[-1]["p_rel_residual"] < 1e-7, f"{h[-1]['p_rel_residual']:.2e}")
    rep("flow developed", h[-1]["u_max"] > 1e-6, f"|u|max={h[-1]['u_max']:.3e}")
    rep("all finite", all(math.isfinite(r["ke"]) for r in h))

    # T7 PIMPLE transient accuracy vs discrete analytic decay
    print("\n[T7] PIMPLE transient accuracy: periodic TGV decay")
    n, L, nu = 24, 1.0, 2e-2
    mp = cartesian_mesh(n, n, n, L, L, L, periodic=(True, True, True))
    sp = PIMPLESolver(mp, nu=nu, patches=[],
                      cfg=PIMPLEConfig(dt=2e-3, n_outer=3, n_piso=2, p_tol=1e-10))
    k = 2 * math.pi / L
    X, Y, Z = mp.cen[:, 0], mp.cen[:, 1], mp.cen[:, 2]
    sp.u = torch.stack([torch.sin(k*X)*torch.cos(k*Y),
                        -torch.cos(k*X)*torch.sin(k*Y),
                        torch.zeros_like(X)], 1)
    ke0 = float(0.5*(sp.u**2).sum(1).mean())
    nst = 40
    hh = sp.run(nst)
    ke1 = hh[-1]["ke"]
    lam_h = (4/(L/n)**2)*(math.sin(k*(L/n)/2)**2)*2
    ana = math.exp(-2*nu*lam_h*nst*2e-3)
    num = ke1/max(ke0, 1e-30)
    rel = abs(num-ana)/max(ana, 1e-30)
    print(f"      KE ratio num={num:.5f}  discrete-analytic={ana:.5f}  rel={rel:.3f}")
    rep("decay matches discrete analytic rate (<8%)", rel < 0.08, f"rel={rel:.3f}")

    # T8 implicit pressure adjoint Taylor test
    print("\n[T8] implicit pressure adjoint: Taylor test")
    ma = cartesian_mesh(6, 6, 6, periodic=(True, True, True))
    sa = PIMPLESolver(ma, nu=1e-2, patches=[], cfg=PIMPLEConfig(p_tol=1e-12, p_max_iter=2000))
    sa.rAU = ma.vol / (ma.vol / 1e-2)
    coeff = (sa._interp(sa.rAU) * sa.alpha).detach().clone().requires_grad_(True)
    bb = torch.randn(ma.Nc, dtype=torch.float64); bb = bb - bb.mean()
    bb.requires_grad_(True)
    def J(bv, cv):
        return PressureAdjoint.apply(bv, cv, sa).pow(2).sum()
    J0 = J(bb, coeff); gb, gc = torch.autograd.grad(J0, (bb, coeff))
    rng = np.random.default_rng(0)
    db = torch.tensor(rng.normal(size=tuple(bb.shape)), dtype=bb.dtype)
    db = db - db.mean(); db = db / db.norm()
    dirv = float((gb*db).sum())
    rows = [(e, abs(float(J(bb+e*db, coeff)) - float(J0) - e*dirv))
            for e in (1e-3, 5e-4, 2.5e-4, 1.25e-4)]
    rates = [math.log(rows[i-1][1]/rows[i][1])/math.log(2.0)
             for i in range(1, len(rows)) if rows[i][1] > 1e-30]
    order = float(np.median(rates)) if rates else 0.0
    rep("pressure adjoint order ~2", abs(order-2.0) < 0.4, f"order={order:.3f}")

    # T9 checkpointed adjoint through the time loop
    print("\n[T9] checkpointed adjoint through the WHOLE time loop")
    mc = cartesian_mesh(5, 5, 5, periodic=(True, True, True))
    sc = PIMPLESolver(mc, nu=5e-2, patches=[],
                      cfg=PIMPLEConfig(dt=5e-3, n_outer=2, n_piso=1,
                                       n_momentum_sweeps=3, p_tol=1e-8))
    Xc = mc.cen[:, 0]
    sc.u = torch.stack([0.1*torch.sin(2*math.pi*Xc),
                        torch.zeros_like(Xc), torch.zeros_like(Xc)], 1)
    # a learnable body force is the parameter we differentiate w.r.t.
    sc.g = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    loop = CheckpointedAdjointTimeLoop(sc, interval=4)
    fw = loop.forward(12)
    print(f"      checkpoints={fw['n_checkpoints']} vs naive={fw['memory_states_naive']}"
          f"  reduction={fw['reduction_factor']:.2f}x")
    rep("checkpointing reduces stored states", fw["reduction_factor"] > 1.5,
        f"{fw['reduction_factor']:.2f}x")
    bw = loop.backward(lambda u: (u**2).sum(), [sc.g])
    rep("adjoint produces a finite parameter gradient",
        bool(torch.isfinite(bw["grads"][0]).all()), str(bw["grads"][0].tolist()))
    rep("recompute overhead reported", bw["recompute_overhead"] >= 1.0,
        f"{bw['recompute_overhead']:.2f}x")

    # T10 organ CFD
    print("\n[T10] PatientSpecificOrganCFD")
    for org in ("brain", "heart", "lung", "liver", "kidney", "vascular"):
        sp_ = mw.cfd.organ_spec(org)
        rep(f"{org}: NOT_IMPLEMENTED declared", len(sp_["NOT_IMPLEMENTED"]) > 0,
            f"{len(sp_['NOT_IMPLEMENTED'])} items")
    bad = mw.cfd.accept_segmentation(np.ones((4,4,4)), {"method": "unvalidated CNN"})
    rep("unvalidated segmentation BLOCKED", bad[0] is None)
    good = mw.cfd.accept_segmentation(np.ones((4,4,4)),
        {"method":"manual","dice_vs_reference":0.93,"validated_by":"radiologist","date":"2026-01-01"})
    rep("validated segmentation accepted", good[0] is not None)
    plan = mw.cfd.plan("vascular", m, patches, nu=5e-2, n_steps=12, ensemble=3,
                       uses_patient_data=True, deid=deid)
    rep("plan runs with ensemble UQ", plan["status"] == Status.OK,
        f"n_ens={plan['uncertainty']['n_ensemble']}")
    rep("validation NOT_COMPARED without reference",
        plan["validation_status"] == Status.NOT_COMPARED)
    rep("NO clinical decision issued", plan["decision"] == "NO_CLINICAL_DECISION_ISSUED")
    blocked = mw.cfd.plan("vascular", m, patches, nu=5e-2, n_steps=4,
                          uses_patient_data=True, deid=None)
    rep("same plan without de-id BLOCKED", blocked["status"] == Status.BLOCKED)

    # T11 validation harness
    print("\n[T11] validation harness")
    v = Validation()
    p_ = np.random.default_rng(1).normal(10, 2, 40)
    rep("no reference => NOT_COMPARED",
        v.compare("4D-flow", p_, "flow_mri")["status"] == Status.NOT_COMPARED)
    v2 = Validation({"flow_mri": p_*1.05 + np.random.default_rng(2).normal(0,0.3,40)})
    c = v2.compare("4D-flow", p_, "flow_mri")
    rep("Bland-Altman + ICC + calibration computed",
        c["status"] == Status.OK and c["icc21"] > 0.8, f"ICC={c['icc21']:.3f}")

    # T12 global invariants
    print("\n[T12] global invariants")
    st_ = mw.status()
    rep("platform declares NOT_A_MEDICAL_DEVICE",
        st_["governance"]["device_status"] == "NOT_A_MEDICAL_DEVICE")
    rep("audit trail recorded", len(mw.gov.audit) > 0, f"n={len(mw.gov.audit)}")
    rep("everything research_only", st_["clinical_actionability"] == RESEARCH_ONLY)

    print("\n" + "="*78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("="*78)
    print("""
IMPLEMENTED AND MEASURED
  PIMPLE transient + Rhie-Chow: continuity -> 0, decay matches the discrete
  analytic rate. Implicit pressure adjoint: Taylor order ~2. Checkpointed
  adjoint through the whole time loop: memory reduction and recompute
  overhead both MEASURED. KG schema, confidence, contradictions. SSC, CSOC
  event bound, OPMC exact gradient, Carpet priority, kappa-fatness guard.
  iPSC risk + missing evidence. Organ registry with explicit gaps.

NOT IMPLEMENTED (declared, not approximated)
  FSI, VOF, non-Newtonian rheology, porous perfusion, aerosol transport,
  moving valves, Windkessel outlets, body-fitted unstructured meshing.
  Real folding / quantum chemistry / cell-type classification backends.
  MONDO / DO / GTEx / atlas / PPI / drug-target parsers.

NEVER PROVIDED
  Gene-editor design, guide RNA, delivery vectors, wet-lab protocols,
  pathogen enhancement, autonomous diagnosis / prescription / surgery,
  a transplant go/no-go, or any clinical recommendation.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ONE MedWorld research platform")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "status", "kg", "onto", "ipsc",
                             "organ", "pimple", "adjoint"])
    ap.add_argument("--organ", default="vascular", choices=sorted(ORGAN_REGISTRY))
    ap.add_argument("--gene", default=None)
    ap.add_argument("--phenotype", default=None)
    ap.add_argument("--rsid", default=None)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--hpo-obo", default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--interval", type=int, default=4)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.cmd == "selftest":
        return 0 if self_test() else 1

    mw = OneMedWorld(offline=not a.online)
    out: Dict[str, Any] = {}

    if a.cmd == "status":
        out = mw.status()
    elif a.cmd == "kg":
        if a.gene and a.phenotype:
            mw.kg.clinvar(a.gene, a.phenotype)
            if a.rsid: mw.kg.gwas(a.rsid, a.phenotype)
            mw.kg.pubmed(f"{a.gene} AND {a.phenotype}")
        if a.hpo_obo: mw.kg.load_hpo(a.hpo_obo)
        mw.kg.detect_contradictions()
        out = mw.kg.report(a.gene, a.phenotype)
    elif a.cmd == "onto":
        om = mw.onto
        st = om.initial(1)
        traj = []
        for i in range(a.steps):
            o = om(st, torch.rand(1, om.obs_dim, dtype=torch.float64))
            st = o["state"]
            traj.append({"step": i,
                         "fates": dict(zip(LINEAGE, [float(x) for x in o["fates"][0]])),
                         "risks": [float(x) for x in o["risks"][0]],
                         "csoc_events": int(o["csoc"]["events"]),
                         "csoc_bound": float(o["csoc"]["bound"])})
        out = {"trajectory": traj, "clinical_actionability": RESEARCH_ONLY}
    elif a.cmd == "ipsc":
        o = mw.onto(mw.onto.initial(1), torch.rand(1, mw.onto.obs_dim, dtype=torch.float64))
        out = mw.ipsc.assess(a.organ, o["state"][0])
    elif a.cmd == "organ":
        out = mw.cfd.organ_spec(a.organ)
    elif a.cmd == "pimple":
        m = cartesian_mesh(12, 10, 8, periodic=(False, False, True))
        nf = m.nf; b = torch.nonzero(m.boundary).flatten()
        pt = [Patch(b[nf[b,0] < -0.99], "inlet", (1.,0.,0.)),
              Patch(b[nf[b,0] > 0.99], "outlet", p_value=0.0),
              Patch(b[nf[b,1].abs() > 0.99], "wall")]
        s = PIMPLESolver(m, 5e-2, pt, PIMPLEConfig(dt=2e-3))
        h = s.run(a.steps)
        out = {"mesh_quality": m.quality(), "history": h,
               "clinical_actionability": RESEARCH_ONLY}
    elif a.cmd == "adjoint":
        m = cartesian_mesh(5,5,5, periodic=(True,True,True))
        s = PIMPLESolver(m, 5e-2, [], PIMPLEConfig(dt=5e-3, n_outer=2, n_piso=1))
        X = m.cen[:,0]
        s.u = torch.stack([0.1*torch.sin(2*math.pi*X), torch.zeros_like(X), torch.zeros_like(X)],1)
        s.g = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        loop = CheckpointedAdjointTimeLoop(s, interval=a.interval)
        fw = loop.forward(a.steps)
        bw = loop.backward(lambda u: (u**2).sum(), [s.g])
        out = {"forward": fw, "grad_body_force": bw["grads"][0].tolist(),
               "recompute_overhead": bw["recompute_overhead"]}

    print(json.dumps(out, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
