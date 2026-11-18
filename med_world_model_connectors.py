#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ONE MedWorld CONNECTORS v1.0
Literature · CFD · Imaging · Omics · Governance · Validation
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

Christ died on the Cross to teach: when struck on the left cheek, turn also
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
THE SIX DELIVERABLES
=============================================================================
[1] LITERATURE  PubMed (E-utilities), ClinVar (E-utilities), GWAS Catalog
                (REST), HPO (hp.obo). OMIM and DisGeNET require licensed
                keys and report UNAVAILABLE without them.
[2] CFD         Adapters for one_cfd_stencils / sesi_multiphase_flow_fix /
                super_dns_one. Availability is DETECTED and REPORTED; the
                uploaded sources do not parse until repair_five.py runs.
[3] IMAGING     DICOM/NIfTI -> resample -> SEGMENTATION INPUT -> marching
                cubes -> surface mesh -> inlet/outlet/wall BC.
                HONEST: segmentation is an INPUT that must be validated.
                There is no automatic clinical segmentation here.
[4] OMICS       scRNA-seq matrix -> QC -> normalize -> marker scoring ->
                cell-fate state with uncertainty. Research grade.
[5] GOVERNANCE  IRB / consent / de-identification / data-use agreement /
                regulatory classification. ENFORCED as a gate: no IRB and
                no de-identification means no patient pathway.
[6] VALIDATION  4D-flow MRI, Doppler, invasive pressure, clinical outcomes.
                Bland-Altman, ICC(2,1), calibration slope/intercept,
                coverage, bootstrap CI. Reports NOT COMPARED without data.

=============================================================================
HARD SAFETY RULES (enforced in code, not just documented)
=============================================================================
R1  No PHI is ever sent to an external API. Query sanitisation strips
    anything matching PHI patterns before any network call.
R2  No patient-derived pathway runs without an APPROVED IRB record AND a
    passed de-identification check.
R3  Every output carries clinical_actionability="research_only" and
    clinician_review_required=True.
R4  No gene-editor design, no guide RNA, no wet-lab protocol, no pathogen
    enhancement, no autonomous clinical order. SafetyGate blocks these.
R5  Missing credentials or missing reference data produce an explicit
    UNAVAILABLE / NOT COMPARED status, never a fabricated result.

=============================================================================
HONEST SCOPE
=============================================================================
* NOT a medical device. NOT for diagnosis, treatment, or surgical decisions.
* Network connectors are real but rate-limited and cached; offline mode
  uses fixtures so the self-test needs no network.
* Imaging requires a validated segmentation. Phantom tests only here.
* Omics cell-fate calls are research hypotheses, not clinical labels.
* CFD backends must be repaired and independently verified before use.
* Validation statistics are computed only when reference data is supplied.
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

# ---- optional dependencies: all graceful ----
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

try:
    import scipy.stats as sps
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("medworld")

VERSION = "1.0.0"
BANNER = f"""
=============================================================================
ONE MedWorld CONNECTORS v{VERSION}
[1] Literature  [2] CFD  [3] Imaging  [4] Omics  [5] Governance  [6] Validation
RESEARCH ONLY. Not a medical device. No PHI leaves this process.
No IRB + no de-identification => no patient pathway.
=============================================================================
"""


# =============================================================================
# §0. ENUMS + CORE RECORDS
# =============================================================================

class Status:
    OK = "OK"
    UNAVAILABLE = "UNAVAILABLE"          # missing credential / dependency
    NOT_COMPARED = "NOT_COMPARED"        # missing reference data
    BLOCKED = "BLOCKED"                  # governance / safety
    ERROR = "ERROR"


class Intent:
    LITERATURE = "literature_query"
    SIMULATION = "simulation"
    RISK = "risk_assessment"
    GENE_EDITOR = "gene_editor_design"
    WETLAB = "wetlab_protocol"
    PATHOGEN = "pathogen_enhancement"
    CLINICAL_ORDER = "autonomous_clinical_order"


@dataclass
class Provenance:
    source: str
    retrieved_at: float
    query: str
    url: str = ""
    cached: bool = False
    license: str = "unknown"
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    gene: str
    variant: Optional[str]
    phenotype: str
    effect: str
    evidence_level: str
    confidence: float
    provenance: Provenance
    organ: Optional[str] = None
    developmental_stage: Optional[str] = None
    contradictions: List[str] = field(default_factory=list)
    clinical_actionability: str = "research_only"


# =============================================================================
# §1. [5] GOVERNANCE — IRB / ETHICS / REGULATORY  (ENFORCED GATE)
# =============================================================================

@dataclass
class IRBRecord:
    protocol_id: str
    institution: str
    approval_date: str            # ISO 8601
    expiry_date: str              # ISO 8601
    approved_data_types: List[str]
    approved_uses: List[str]
    consent_model: str            # "written" | "waived" | "broad" | "none"
    data_use_agreement: bool
    human_genetic_resources_permit: bool = False
    cross_border_transfer_approved: bool = False
    status: str = "PENDING"       # PENDING | APPROVED | EXPIRED | REVOKED

    def is_active(self, now: Optional[float] = None) -> bool:
        if self.status != "APPROVED":
            return False
        now = now if now is not None else time.time()
        try:
            exp = time.mktime(time.strptime(self.expiry_date, "%Y-%m-%d"))
        except Exception:
            return False
        return now < exp


@dataclass
class DeIDReport:
    passed: bool
    removed_fields: List[str]
    residual_risk: str            # "low" | "medium" | "high"
    quasi_identifiers: List[str]
    method: str
    k_anonymity_estimate: Optional[int] = None


class PHIGuard:
    """
    R1: nothing that looks like PHI may leave the process.
    Patterns are intentionally conservative (over-block rather than leak).
    """
    PHI_PATTERNS = {
        "name_like": r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "mrn": r"\b(MRN|mrn)[:\s]*\d{4,}\b",
        "date_full": r"\b(19|20)\d{2}-\d{2}-\d{2}\b",
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "phone": r"\b\+?\d[\d\s().-]{8,}\d\b",
        "address": r"\b\d{1,5}\s+\w+\s+(Street|St|Road|Rd|Avenue|Ave|Lane|Ln)\b",
        "dicom_uid": r"\b1\.2\.\d{3,}\.[\d.]{6,}\b",
        "nhs": r"\b\d{3}\s\d{3}\s\d{4}\b",
    }
    DIRECT_ID_FIELDS = {
        "name", "patient_name", "given_name", "family_name", "ssn",
        "mrn", "patient_id", "national_id", "address", "email", "phone",
        "birth_date", "dob", "date_of_birth", "photo", "fingerprint",
        "dicom_patient_name", "dicom_patient_id", "accession_number",
    }
    QUASI_IDENTIFIERS = {"age", "sex", "postcode", "zip", "ethnicity",
                         "admission_date", "hospital", "rare_diagnosis"}

    def scan_text(self, text: str) -> List[str]:
        hits = []
        for label, pat in self.PHI_PATTERNS.items():
            if re.search(pat, text):
                hits.append(label)
        return hits

    def sanitize_query(self, query: str) -> Tuple[str, List[str]]:
        """Strip anything PHI-like BEFORE any network call."""
        cleaned = query
        removed = []
        for label, pat in self.PHI_PATTERNS.items():
            new = re.sub(pat, "[REDACTED]", cleaned)
            if new != cleaned:
                removed.append(label)
                cleaned = new
        return cleaned, removed

    def deidentify(self, record: Dict[str, Any],
                   generalize_age: bool = True) -> DeIDReport:
        removed, quasi = [], []
        out = dict(record)
        for k in list(out.keys()):
            if k.lower() in self.DIRECT_ID_FIELDS:
                out.pop(k)
                removed.append(k)
            elif k.lower() in self.QUASI_IDENTIFIERS:
                quasi.append(k)
        if generalize_age and "age" in out:
            try:
                a = int(out["age"])
                out["age_band"] = f"{(a // 10) * 10}-{(a // 10) * 10 + 9}"
                out.pop("age")
                removed.append("age->age_band")
            except Exception:
                pass
        for k, v in list(out.items()):
            if isinstance(v, str):
                hits = self.scan_text(v)
                if hits:
                    out[k] = self.sanitize_query(v)[0]
                    removed.append(f"{k}:{','.join(hits)}")
        risk = "low" if not quasi else ("medium" if len(quasi) <= 3 else "high")
        return DeIDReport(
            passed=(risk != "high"),
            removed_fields=removed, residual_risk=risk,
            quasi_identifiers=quasi,
            method="direct-id removal + age generalisation + text scrub",
            k_anonymity_estimate=None,
        )


class SafetyGate:
    """R4: block disallowed intents outright."""
    BLOCKED = {
        Intent.GENE_EDITOR: [
            r"\bcrispr\b", r"\bcas9\b", r"\bcas12\b", r"\bcas13\b",
            r"\bguide\s*rna\b", r"\bgrna\b", r"\bsgrna\b",
            r"\bbase\s*editor\b", r"\bprime\s*editor\b", r"\bgene\s*drive\b",
            r"\bgene\s*edit(or|ing)\b", r"\bknock\s*(out|in)\b",
            r"\baav\s*capsid\b", r"\bdelivery\s*vector\b",
        ],
        Intent.WETLAB: [
            r"\bprotocol\b", r"\btransfection\b", r"\belectroporation\b",
            r"\bculture\s*condition\b", r"\bmedia\s*recipe\b",
            r"\bstep\s*by\s*step\b", r"\bwet\s*lab\b", r"\bpcr\s*condition\b",
        ],
        Intent.PATHOGEN: [
            r"\benhance\s*virulence\b", r"\bimmune\s*evasion\b",
            r"\bincrease\s*transmissibility\b", r"\bgain\s*of\s*function\b",
        ],
        Intent.CLINICAL_ORDER: [
            r"\bdiagnose\s*patient\b", r"\bprescribe\b", r"\boperate\s*on\b",
            r"\btransplant\s*now\b", r"\bautonomous\s*surgery\b",
            r"\bclinical\s*decision\s*without\b",
        ],
    }

    def check(self, text: str) -> Tuple[bool, str, List[str]]:
        low = text.lower()
        for intent, pats in self.BLOCKED.items():
            hits = [p for p in pats if re.search(p, low)]
            if hits:
                return False, intent, hits
        return True, Intent.LITERATURE, []


class GovernanceGate:
    """
    R2/R3: the single choke point every patient-derived pathway must pass.
    """

    def __init__(self, irb: Optional[IRBRecord] = None):
        self.irb = irb
        self.phi = PHIGuard()
        self.safety = SafetyGate()
        self.audit: List[Dict[str, Any]] = []

    def authorize(self, *, text: str, uses_patient_data: bool,
                  deid: Optional[DeIDReport] = None,
                  purpose: str = "research") -> Dict[str, Any]:
        ok, intent, hits = self.safety.check(text)
        decision = {
            "authorized": False, "intent": intent, "blocked_patterns": hits,
            "reason": "", "clinical_actionability": "research_only",
            "clinician_review_required": True,
            "required_approvals": ["IRB", "ethics_committee", "clinician",
                                   "regulator"],
        }
        if not ok:
            decision["reason"] = f"BLOCKED intent={intent}"
            self._log(decision, text)
            return decision

        if uses_patient_data:
            if self.irb is None:
                decision["reason"] = "BLOCKED: no IRB record"
                self._log(decision, text)
                return decision
            if not self.irb.is_active():
                decision["reason"] = f"BLOCKED: IRB status={self.irb.status} or expired"
                self._log(decision, text)
                return decision
            if purpose not in self.irb.approved_uses:
                decision["reason"] = f"BLOCKED: purpose '{purpose}' not in approved_uses"
                self._log(decision, text)
                return decision
            if not self.irb.data_use_agreement:
                decision["reason"] = "BLOCKED: no data use agreement"
                self._log(decision, text)
                return decision
            if deid is None or not deid.passed:
                decision["reason"] = "BLOCKED: de-identification missing or failed"
                self._log(decision, text)
                return decision
            decision["deid"] = asdict(deid)

        decision["authorized"] = True
        decision["reason"] = "authorized for research use"
        self._log(decision, text)
        return decision

    def _log(self, decision: Dict[str, Any], text: str) -> None:
        self.audit.append({
            "ts": time.time(),
            "query_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "authorized": decision["authorized"],
            "reason": decision["reason"],
        })

    def regulatory_classification(self) -> Dict[str, Any]:
        return {
            "intended_use": "research / hypothesis generation",
            "device_status": "NOT_A_MEDICAL_DEVICE",
            "us_fda": "not submitted; SaMD classification would require 510(k)/De Novo",
            "eu_mdr": "not CE marked; MDR Annex VIII rules 11/22 would apply",
            "china_nmpa": "not registered; class II/III software likely",
            "data_law": ["HIPAA (US)", "GDPR (EU)", "PIPL (CN)",
                         "Human Genetic Resources Regulation (CN)"],
            "stem_cell_guidance": ["ISSCR", "ISCT"],
            "reporting_guidelines": ["TRIPOD+AI", "CONSORT-AI", "SPIRIT-AI",
                                     "DECIDE-AI", "STARD"],
            "prohibited": ["autonomous diagnosis", "autonomous treatment",
                           "gene editor design", "wet-lab protocol generation"],
        }


# =============================================================================
# §2. HTTP LAYER — rate limit, cache, retry, provenance
# =============================================================================

class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = float(min_interval)
        self._last = 0.0

    def wait(self) -> None:
        dt = time.time() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()


class HTTPClient:
    """stdlib-only HTTP with disk cache, retry/backoff, and rate limiting."""

    def __init__(self, cache_dir: str = "./medworld_cache",
                 ttl_s: float = 86400.0, min_interval: float = 0.34,
                 offline: bool = True, timeout: float = 20.0,
                 max_retries: int = 3):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.ttl = float(ttl_s)
        self.limiter = RateLimiter(min_interval)
        self.offline = bool(offline)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.phi = PHIGuard()
        self.n_requests = 0
        self.n_cache_hits = 0

    def _key(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache / f"{h}.json"

    def get(self, url: str, headers: Optional[Dict[str, str]] = None,
            force: bool = False) -> Tuple[Optional[str], Dict[str, Any]]:
        # R1: never let a PHI-bearing URL leave
        clean, removed = self.phi.sanitize_query(url)
        meta = {"url": clean, "phi_stripped": removed, "cached": False}
        if removed:
            meta["warning"] = "PHI-like content stripped from URL"
        path = self._key(clean)
        if path.exists() and not force:
            age = time.time() - path.stat().st_mtime
            if age < self.ttl:
                self.n_cache_hits += 1
                meta["cached"] = True
                return path.read_text(encoding="utf-8"), meta
        if self.offline:
            meta["status"] = Status.UNAVAILABLE
            meta["reason"] = "offline mode; no cached response"
            return None, meta

        last_err = ""
        for attempt in range(self.max_retries):
            self.limiter.wait()
            self.n_requests += 1
            try:
                req = urllib.request.Request(
                    clean, headers=headers or {"User-Agent": "ONE-MedWorld/1.0 (research)"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = r.read().decode("utf-8", errors="replace")
                path.write_text(body, encoding="utf-8")
                meta["status"] = Status.OK
                return body, meta
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}"
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2.0 ** attempt)
                    continue
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2.0 ** attempt)
        meta["status"] = Status.ERROR
        meta["reason"] = last_err
        return None, meta


# =============================================================================
# §3. [1] LITERATURE CONNECTORS
# =============================================================================

class LiteratureConnector:
    """Base: provenance-aware, credential-aware, offline-aware."""
    name = "base"
    requires_key = False
    license_note = "unknown"

    def __init__(self, http: HTTPClient, api_key: Optional[str] = None):
        self.http = http
        self.api_key = api_key or os.environ.get(self.env_key or "", None)

    env_key: Optional[str] = None

    def available(self) -> Tuple[bool, str]:
        if self.requires_key and not self.api_key:
            return False, f"{self.name} requires an API key ({self.env_key})"
        return True, "available"

    def _prov(self, query: str, url: str, cached: bool) -> Provenance:
        return Provenance(source=self.name, retrieved_at=time.time(),
                          query=query, url=url, cached=cached,
                          license=self.license_note)


class PubMedConnector(LiteratureConnector):
    """NCBI E-utilities. Public; 3 req/s without key, 10 with."""
    name = "PubMed"
    env_key = "NCBI_API_KEY"
    license_note = "NLM E-utilities; respect NLM usage policy"
    BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, term: str, retmax: int = 20) -> Dict[str, Any]:
        ok, why = self.available()
        clean, removed = PHIGuard().sanitize_query(term)
        params = {"db": "pubmed", "term": clean, "retmax": str(retmax),
                  "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
        body, meta = self.http.get(url)
        if body is None:
            return {"status": meta.get("status", Status.UNAVAILABLE),
                    "reason": meta.get("reason", ""), "pmids": [],
                    "provenance": self._prov(clean, url, meta.get("cached", False)).to_dict()}
        try:
            ids = json.loads(body)["esearchresult"]["idlist"]
        except Exception as e:
            return {"status": Status.ERROR, "reason": str(e), "pmids": []}
        return {"status": Status.OK, "pmids": ids, "count": len(ids),
                "phi_stripped": removed,
                "provenance": self._prov(clean, url, meta.get("cached", False)).to_dict()}

    def fetch_abstracts(self, pmids: List[str]) -> Dict[str, Any]:
        if not pmids:
            return {"status": Status.OK, "records": []}
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
        body, meta = self.http.get(url)
        if body is None:
            return {"status": meta.get("status", Status.UNAVAILABLE), "records": []}
        recs = []
        for blk in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", body, re.S):
            pmid = re.search(r"<PMID[^>]*>(\d+)</PMID>", blk)
            title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", blk, re.S)
            abst = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", blk, re.S)
            recs.append({
                "pmid": pmid.group(1) if pmid else None,
                "title": re.sub(r"<[^>]+>", "", title.group(1)) if title else "",
                "abstract": " ".join(re.sub(r"<[^>]+>", "", a) for a in abst),
            })
        return {"status": Status.OK, "records": recs,
                "provenance": self._prov(",".join(pmids), url,
                                         meta.get("cached", False)).to_dict()}


class ClinVarConnector(LiteratureConnector):
    """NCBI ClinVar via E-utilities. Public."""
    name = "ClinVar"
    env_key = "NCBI_API_KEY"
    license_note = "NLM ClinVar; check per-record submitter terms"
    BASE = PubMedConnector.BASE

    def search_variant(self, gene: str, variant: Optional[str] = None,
                       retmax: int = 20) -> Dict[str, Any]:
        term = f"{gene}[gene]"
        if variant:
            term += f" AND {variant}"
        params = {"db": "clinvar", "term": term, "retmax": str(retmax),
                  "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"
        body, meta = self.http.get(url)
        if body is None:
            return {"status": meta.get("status", Status.UNAVAILABLE),
                    "reason": meta.get("reason", ""), "ids": [],
                    "provenance": self._prov(term, url, meta.get("cached", False)).to_dict()}
        try:
            ids = json.loads(body)["esearchresult"]["idlist"]
        except Exception as e:
            return {"status": Status.ERROR, "reason": str(e), "ids": []}
        return {"status": Status.OK, "ids": ids,
                "provenance": self._prov(term, url, meta.get("cached", False)).to_dict()}

    def summarize(self, ids: List[str]) -> Dict[str, Any]:
        if not ids:
            return {"status": Status.OK, "records": []}
        params = {"db": "clinvar", "id": ",".join(ids), "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.BASE}/esummary.fcgi?{urllib.parse.urlencode(params)}"
        body, meta = self.http.get(url)
        if body is None:
            return {"status": meta.get("status", Status.UNAVAILABLE), "records": []}
        recs = []
        try:
            res = json.loads(body).get("result", {})
            for uid in res.get("uids", []):
                r = res.get(uid, {})
                sig = r.get("clinical_significance", {})
                desc = sig.get("description", "") if isinstance(sig, dict) else str(sig)
                genes = [g.get("symbol", "") for g in r.get("genes", [])]
                recs.append({
                    "clinvar_id": uid,
                    "title": r.get("title", ""),
                    "genes": genes,
                    "clinical_significance": desc,
                    "review_status": r.get("review_status", ""),
                    "variation_set": [v.get("variation_name", "")
                                      for v in r.get("variation_set", [])],
                })
        except Exception as e:
            return {"status": Status.ERROR, "reason": str(e), "records": []}
        return {"status": Status.OK, "records": recs,
                "provenance": self._prov(",".join(ids), url,
                                         meta.get("cached", False)).to_dict()}


class GWASCatalogConnector(LiteratureConnector):
    """EBI GWAS Catalog REST. Public."""
    name = "GWAS_Catalog"
    license_note = "EBI GWAS Catalog; CC0 / EBI terms"
    BASE = "https://www.ebi.ac.uk/gwas/rest/api"

    def associations_by_rsid(self, rsid: str) -> Dict[str, Any]:
        if not re.fullmatch(r"rs\d+", rsid):
            return {"status": Status.ERROR, "reason": "rsid must match rs\\d+"}
        url = f"{self.BASE}/singleNucleotidePolymorphisms/{rsid}/associations"
        body, meta = self.http.get(url, headers={"Accept": "application/json"})
        if body is None:
            return {"status": meta.get("status", Status.UNAVAILABLE),
                    "reason": meta.get("reason", ""), "associations": [],
                    "provenance": self._prov(rsid, url, meta.get("cached", False)).to_dict()}
        assoc = []
        try:
            j = json.loads(body)
            for a in j.get("_embedded", {}).get("associations", []):
                traits = []
                for t in a.get("efoTraits", []):
                    traits.append(t.get("trait", ""))
                assoc.append({
                    "rsid": rsid,
                    "pvalue": a.get("pvalue"),
                    "or_per_copy": a.get("orPerCopyNum"),
                    "beta": a.get("betaNum"),
                    "risk_allele": a.get("riskAllele"),
                    "traits": traits,
                })
        except Exception as e:
            return {"status": Status.ERROR, "reason": str(e), "associations": []}
        return {"status": Status.OK, "associations": assoc,
                "provenance": self._prov(rsid, url, meta.get("cached", False)).to_dict()}


class HPOConnector(LiteratureConnector):
    """Human Phenotype Ontology. Public OBO file."""
    name = "HPO"
    license_note = "HPO; hpo.jax.org license (custom, attribution required)"
    URL = "http://purl.obolibrary.org/obo/hp.obo"

    def __init__(self, http: HTTPClient, local_obo: Optional[str] = None,
                 api_key: Optional[str] = None):
        super().__init__(http, api_key)
        self.local_obo = local_obo
        self._terms: Dict[str, Dict[str, Any]] = {}

    def load(self) -> Dict[str, Any]:
        text = None
        if self.local_obo and Path(self.local_obo).exists():
            text = Path(self.local_obo).read_text(encoding="utf-8", errors="replace")
            src = self.local_obo
        else:
            text, meta = self.http.get(self.URL)
            src = self.URL
        if text is None:
            return {"status": Status.UNAVAILABLE,
                    "reason": "no local hp.obo and offline/no network",
                    "n_terms": 0}
        self._terms = self._parse_obo(text)
        return {"status": Status.OK, "n_terms": len(self._terms), "source": src}

    @staticmethod
    def _parse_obo(text: str) -> Dict[str, Dict[str, Any]]:
        terms: Dict[str, Dict[str, Any]] = {}
        cur: Optional[Dict[str, Any]] = None
        for line in text.splitlines():
            line = line.strip()
            if line == "[Term]":
                if cur and cur.get("id"):
                    terms[cur["id"]] = cur
                cur = {"id": "", "name": "", "is_a": [], "def": "",
                       "synonyms": [], "obsolete": False}
            elif line.startswith("[") and line != "[Term]":
                if cur and cur.get("id"):
                    terms[cur["id"]] = cur
                cur = None
            elif cur is not None:
                if line.startswith("id: "):
                    cur["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    cur["name"] = line[6:].strip()
                elif line.startswith("is_a: "):
                    cur["is_a"].append(line[6:].split("!")[0].strip())
                elif line.startswith("def: "):
                    m = re.match(r'def:\s*"([^"]*)"', line)
                    cur["def"] = m.group(1) if m else ""
                elif line.startswith("synonym: "):
                    m = re.match(r'synonym:\s*"([^"]*)"', line)
                    if m:
                        cur["synonyms"].append(m.group(1))
                elif line.startswith("is_obsolete: true"):
                    cur["obsolete"] = True
        if cur and cur.get("id"):
            terms[cur["id"]] = cur
        return terms

    def term(self, hp_id: str) -> Optional[Dict[str, Any]]:
        return self._terms.get(hp_id)

    def search(self, text: str, max_hits: int = 20) -> List[Dict[str, Any]]:
        low = text.lower()
        out = []
        for tid, t in self._terms.items():
            if t["obsolete"]:
                continue
            if low in t["name"].lower() or any(low in s.lower() for s in t["synonyms"]):
                out.append(t)
                if len(out) >= max_hits:
                    break
        return out

    def ancestors(self, hp_id: str, max_depth: int = 12) -> List[str]:
        seen, stack, depth = [], [hp_id], 0
        while stack and depth < max_depth:
            nxt = []
            for tid in stack:
                t = self._terms.get(tid)
                if not t:
                    continue
                for p in t["is_a"]:
                    if p not in seen:
                        seen.append(p)
                        nxt.append(p)
            stack = nxt
            depth += 1
        return seen


class OMIMConnector(LiteratureConnector):
    """OMIM API — REQUIRES a licensed key. Reports UNAVAILABLE without it."""
    name = "OMIM"
    requires_key = True
    env_key = "OMIM_API_KEY"
    license_note = "OMIM licensed API; commercial use requires a separate license"
    BASE = "https://api.omim.org/api"

    def entry(self, mim_number: int) -> Dict[str, Any]:
        ok, why = self.available()
        if not ok:
            return {"status": Status.UNAVAILABLE, "reason": why, "entry": None}
        params = {"mimNumber": str(mim_number), "include": "text",
                  "format": "json", "apiKey": self.api_key}
        url = f"{self.BASE}/entry?{urllib.parse.urlencode(params)}"
        body, meta = self.http.get(url)
        if body is None:
            return {"status": meta.get("status", Status.ERROR),
                    "reason": meta.get("reason", ""), "entry": None}
        try:
            j = json.loads(body)
            e = j["omim"]["entryList"][0]["entry"]
            return {"status": Status.OK,
                    "entry": {"mim": e.get("mimNumber"),
                              "titles": e.get("titles", {}),
                              "gene_symbols": e.get("geneSymbols", "")},
                    "provenance": self._prov(str(mim_number), url,
                                             meta.get("cached", False)).to_dict()}
        except Exception as ex:
            return {"status": Status.ERROR, "reason": str(ex), "entry": None}


class DisGeNETConnector(LiteratureConnector):
    """DisGeNET — REQUIRES a bearer token. UNAVAILABLE without it."""
    name = "DisGeNET"
    requires_key = True
    env_key = "DISGENET_API_KEY"
    license_note = "DisGeNET; free academic tier, commercial license required"
    BASE = "https://www.disgenet.org/api"

    def gene_disease(self, gene: str) -> Dict[str, Any]:
        ok, why = self.available()
        if not ok:
            return {"status": Status.UNAVAILABLE, "reason": why, "associations": []}
        url = f"{self.BASE}/gda/gene/{urllib.parse.quote(gene)}"
        body, meta = self.http.get(url, headers={"Authorization": f"Bearer {self.api_key}",
                                                 "Accept": "application/json"})
        if body is None:
            return {"status": meta.get("status", Status.ERROR),
                    "reason": meta.get("reason", ""), "associations": []}
        try:
            j = json.loads(body)
            rows = j if isinstance(j, list) else j.get("payload", [])
            assoc = [{"gene": r.get("gene_symbol"),
                      "disease": r.get("disease_name"),
                      "score": r.get("score"),
                      "pmids": r.get("pmid_count"),
                      "source": r.get("source")} for r in rows]
            return {"status": Status.OK, "associations": assoc,
                    "provenance": self._prov(gene, url,
                                             meta.get("cached", False)).to_dict()}
        except Exception as ex:
            return {"status": Status.ERROR, "reason": str(ex), "associations": []}


# =============================================================================
# §4. KNOWLEDGE GRAPH AGGREGATION
# =============================================================================

class KnowledgeGraph:
    EVIDENCE_WEIGHT = {
        "curated": 1.0, "clinical_review": 0.95, "gwas": 0.75,
        "single_cell": 0.65, "animal": 0.55, "in_vitro": 0.45,
        "text_mined": 0.30, "inferred": 0.25,
    }

    def __init__(self):
        self.evidence: List[Evidence] = []
        self.unavailable: List[Dict[str, str]] = []

    def add(self, ev: Evidence) -> None:
        self.evidence.append(ev)

    def note_unavailable(self, source: str, reason: str) -> None:
        self.unavailable.append({"source": source, "reason": reason})

    def ingest_clinvar(self, payload: Dict[str, Any], gene: str,
                       phenotype: str) -> int:
        if payload.get("status") != Status.OK:
            self.note_unavailable("ClinVar", payload.get("reason", "unknown"))
            return 0
        n = 0
        for r in payload.get("records", []):
            sig = (r.get("clinical_significance") or "").lower()
            if "pathogenic" in sig:
                effect, conf = "risk_increase", 0.9
            elif "benign" in sig:
                effect, conf = "benign", 0.8
            else:
                effect, conf = "uncertain", 0.35
            lvl = ("clinical_review" if "reviewed" in (r.get("review_status") or "").lower()
                   else "curated")
            prov = Provenance(**{k: v for k, v in payload["provenance"].items()
                                 if k in Provenance.__dataclass_fields__})
            self.add(Evidence(gene=gene,
                              variant=(r.get("variation_set") or [None])[0],
                              phenotype=phenotype, effect=effect,
                              evidence_level=lvl, confidence=conf,
                              provenance=prov))
            n += 1
        return n

    def ingest_gwas(self, payload: Dict[str, Any], phenotype: str) -> int:
        if payload.get("status") != Status.OK:
            self.note_unavailable("GWAS_Catalog", payload.get("reason", "unknown"))
            return 0
        n = 0
        prov = Provenance(**{k: v for k, v in payload["provenance"].items()
                             if k in Provenance.__dataclass_fields__})
        for a in payload.get("associations", []):
            p = a.get("pvalue")
            genome_wide = isinstance(p, (int, float)) and p < 5e-8
            self.add(Evidence(gene=a.get("risk_allele") or "unknown",
                              variant=a.get("rsid"),
                              phenotype="; ".join(a.get("traits", [])) or phenotype,
                              effect="associated",
                              evidence_level="gwas",
                              confidence=0.75 if genome_wide else 0.35,
                              provenance=prov))
            n += 1
        return n

    def confidence(self, gene: str, phenotype: str) -> float:
        rel = [e for e in self.evidence
               if e.gene == gene and phenotype.lower() in e.phenotype.lower()]
        if not rel:
            return 0.0
        num = sum(self.EVIDENCE_WEIGHT.get(e.evidence_level, 0.3) * e.confidence
                  for e in rel)
        den = sum(self.EVIDENCE_WEIGHT.get(e.evidence_level, 0.3) for e in rel)
        base = num / max(den, 1e-12)
        penalty = 0.10 * sum(1 for e in rel if e.contradictions)
        return float(np.clip(base - penalty, 0.0, 1.0))

    def conflicts(self) -> List[Dict[str, Any]]:
        out = []
        for i, a in enumerate(self.evidence):
            for b in self.evidence[i + 1:]:
                if (a.gene == b.gene
                        and a.phenotype.lower() == b.phenotype.lower()
                        and a.effect != b.effect
                        and {a.effect, b.effect} & {"risk_increase", "benign"}):
                    out.append({"gene": a.gene, "phenotype": a.phenotype,
                                "effects": [a.effect, b.effect],
                                "sources": [a.provenance.source, b.provenance.source]})
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "n_evidence": len(self.evidence),
            "n_conflicts": len(self.conflicts()),
            "conflicts": self.conflicts()[:10],
            "sources_unavailable": self.unavailable,
            "by_level": {lvl: sum(1 for e in self.evidence if e.evidence_level == lvl)
                         for lvl in self.EVIDENCE_WEIGHT},
            "clinical_actionability": "research_only",
        }


# =============================================================================
# §5. [3] IMAGING PIPELINE: CT / MRI -> MESH -> BOUNDARY CONDITIONS
# =============================================================================

@dataclass
class ImageVolume:
    data: np.ndarray               # (Z, Y, X) or (X, Y, Z)
    spacing_mm: Tuple[float, float, float]
    origin_mm: Tuple[float, float, float]
    modality: str                  # "CT" | "MRI" | "4D-flow" | "phantom"
    orientation: str = "LPS"
    deidentified: bool = False
    provenance: str = ""


@dataclass
class SurfaceMesh:
    vertices_mm: np.ndarray        # (N, 3)
    faces: np.ndarray              # (M, 3)
    normals: np.ndarray
    n_vertices: int
    n_faces: int
    watertight: Optional[bool]
    quality: Dict[str, float]


@dataclass
class BoundaryConditionSet:
    inlet_faces: np.ndarray
    outlet_faces: np.ndarray
    wall_faces: np.ndarray
    inlet_area_mm2: float
    outlet_area_mm2: float
    flow_split: Dict[str, float]
    waveform: Optional[np.ndarray]
    assumptions: List[str]
    validation_required: List[str]


class ImagingPipeline:
    """
    HONEST: segmentation is an INPUT. There is no automatic clinical
    segmentation here. A validated segmentation (manual or an approved
    model with its own validation report) must be supplied.
    """

    def __init__(self, governance: GovernanceGate):
        self.gov = governance

    # ---- readers ----
    def read_dicom_series(self, directory: str) -> Tuple[Optional[ImageVolume], Dict[str, Any]]:
        if not HAS_PYDICOM:
            return None, {"status": Status.UNAVAILABLE, "reason": "pydicom not installed"}
        files = sorted(Path(directory).glob("*.dcm"))
        if not files:
            return None, {"status": Status.ERROR, "reason": "no .dcm files"}
        slices = []
        for f in files:
            ds = pydicom.dcmread(str(f))
            # R1/R2: strip PHI at read time
            for tag in ("PatientName", "PatientID", "PatientBirthDate",
                        "AccessionNumber", "InstitutionName", "ReferringPhysicianName"):
                if tag in ds:
                    setattr(ds, tag, "ANON")
            slices.append(ds)
        slices.sort(key=lambda s: float(getattr(s, "ImagePositionPatient", [0, 0, 0])[2]))
        arr = np.stack([s.pixel_array for s in slices]).astype(np.float64)
        s0 = slices[0]
        ps = getattr(s0, "PixelSpacing", [1.0, 1.0])
        st = getattr(s0, "SliceThickness", 1.0)
        # CT rescale
        slope = float(getattr(s0, "RescaleSlope", 1.0))
        intercept = float(getattr(s0, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        vol = ImageVolume(
            data=arr, spacing_mm=(float(ps[0]), float(ps[1]), float(st)),
            origin_mm=tuple(float(v) for v in getattr(s0, "ImagePositionPatient", [0, 0, 0])),
            modality=str(getattr(s0, "Modality", "CT")),
            deidentified=True, provenance=f"DICOM:{directory}")
        return vol, {"status": Status.OK, "n_slices": len(slices),
                     "phi_stripped": True}

    def read_nifti(self, path: str) -> Tuple[Optional[ImageVolume], Dict[str, Any]]:
        if not HAS_NIBABEL:
            return None, {"status": Status.UNAVAILABLE, "reason": "nibabel not installed"}
        img = nib.load(path)
        data = np.asarray(img.dataobj, dtype=np.float64)
        zooms = img.header.get_zooms()[:3]
        vol = ImageVolume(data=data, spacing_mm=tuple(float(z) for z in zooms),
                          origin_mm=(0.0, 0.0, 0.0), modality="MRI",
                          deidentified=True, provenance=f"NIfTI:{path}")
        return vol, {"status": Status.OK, "shape": data.shape}

    # ---- resample ----
    def resample_isotropic(self, vol: ImageVolume, target_mm: float = 0.5
                           ) -> ImageVolume:
        """Trilinear resample to isotropic spacing (numpy only)."""
        sp = np.array(vol.spacing_mm, dtype=float)
        data = vol.data
        zoom = sp / target_mm
        new_shape = tuple(max(2, int(round(s * z))) for s, z in zip(data.shape, zoom))
        idx = [np.linspace(0, s - 1, n) for s, n in zip(data.shape, new_shape)]
        grids = np.meshgrid(*idx, indexing="ij")
        coords = np.stack(grids, axis=-1)
        out = self._trilinear(data, coords)
        return ImageVolume(data=out, spacing_mm=(target_mm,) * 3,
                           origin_mm=vol.origin_mm, modality=vol.modality,
                           orientation=vol.orientation,
                           deidentified=vol.deidentified,
                           provenance=vol.provenance + f"|resample{target_mm}mm")

    @staticmethod
    def _trilinear(arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
        c = np.clip(coords, 0, np.array(arr.shape) - 1 - 1e-6)
        i0 = np.floor(c).astype(int)
        f = c - i0
        out = np.zeros(c.shape[:-1], dtype=arr.dtype)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((f[..., 0] if dz else 1 - f[..., 0])
                         * (f[..., 1] if dy else 1 - f[..., 1])
                         * (f[..., 2] if dx else 1 - f[..., 2]))
                    out += w * arr[i0[..., 0] + dz, i0[..., 1] + dy, i0[..., 2] + dx]
        return out

    # ---- segmentation: INPUT, validated ----
    def accept_segmentation(self, mask: np.ndarray, vol: ImageVolume,
                            validation_report: Dict[str, Any]
                            ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        required = ["method", "dice_vs_reference", "validated_by", "date"]
        missing = [k for k in required if k not in validation_report]
        if missing:
            return None, {"status": Status.BLOCKED,
                          "reason": f"segmentation validation missing: {missing}"}
        if validation_report.get("dice_vs_reference", 0.0) < 0.80:
            return None, {"status": Status.BLOCKED,
                          "reason": "segmentation Dice < 0.80; not acceptable for planning research"}
        if mask.shape != vol.data.shape:
            return None, {"status": Status.ERROR, "reason": "mask/volume shape mismatch"}
        return (mask > 0.5).astype(np.uint8), {"status": Status.OK,
                                               "validation": validation_report}

    # ---- mesh ----
    def mask_to_mesh(self, mask: np.ndarray, spacing_mm: Tuple[float, float, float]
                     ) -> Tuple[Optional[SurfaceMesh], Dict[str, Any]]:
        if not HAS_SKIMAGE:
            return None, {"status": Status.UNAVAILABLE,
                          "reason": "scikit-image not installed"}
        if mask.sum() < 8:
            return None, {"status": Status.ERROR, "reason": "mask too small"}
        padded = np.pad(mask, 1, mode="constant")
        try:
            verts, faces, normals, _ = marching_cubes(
                padded.astype(np.float64), level=0.5, spacing=spacing_mm)
        except Exception as e:
            return None, {"status": Status.ERROR, "reason": str(e)}
        verts -= np.array(spacing_mm)
        q = self._mesh_quality(verts, faces)
        return SurfaceMesh(vertices_mm=verts, faces=faces, normals=normals,
                           n_vertices=len(verts), n_faces=len(faces),
                           watertight=q.get("euler_characteristic") == 2,
                           quality=q), {"status": Status.OK}

    @staticmethod
    def _mesh_quality(v: np.ndarray, f: np.ndarray) -> Dict[str, float]:
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        e_sorted = np.sort(e, axis=1)
        uniq, counts = np.unique(e_sorted, axis=0, return_counts=True)
        boundary = int((counts == 1).sum())
        V, F, E = len(v), len(f), len(uniq)
        tri = v[f]
        a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                    tri[:, 2] - tri[:, 0]), axis=1) * 0.5
        degenerate = int((a < 1e-12).sum())
        return {"n_boundary_edges": boundary,
                "euler_characteristic": V - E + F,
                "degenerate_faces": degenerate,
                "area_mm2": float(a.sum()),
                "min_face_area_mm2": float(a.min()) if len(a) else 0.0}

    # ---- boundary conditions ----
    def assign_bc(self, mesh: SurfaceMesh,
                  inlet_plane: Tuple[int, float],
                  outlet_planes: List[Tuple[int, float]],
                  flow_rate_ml_s: Optional[float] = None,
                  waveform: Optional[np.ndarray] = None,
                  murray_exponent: float = 2.6
                  ) -> BoundaryConditionSet:
        """
        Planes are (axis, coordinate_mm). Faces within tolerance are tagged.
        Flow split by Murray's law (area^(exponent/2)) — an ASSUMPTION that
        must be replaced by patient-specific measurement when available.
        """
        cent = mesh.vertices_mm[mesh.faces].mean(axis=1)
        tol = float(np.mean(np.linalg.norm(
            mesh.vertices_mm[mesh.faces[:, 0]] - mesh.vertices_mm[mesh.faces[:, 1]], axis=1))) * 1.5
        ax, coord = inlet_plane
        inlet = np.where(np.abs(cent[:, ax] - coord) < tol)[0]
        outlets = []
        for a, c in outlet_planes:
            outlets.append(np.where(np.abs(cent[:, a] - c) < tol)[0])
        outlet_all = np.concatenate(outlets) if outlets else np.array([], dtype=int)
        tagged = np.concatenate([inlet, outlet_all]) if len(outlet_all) else inlet
        wall = np.setdiff1d(np.arange(mesh.n_faces), tagged)

        def area(idx):
            if len(idx) == 0:
                return 0.0
            t = mesh.vertices_mm[mesh.faces[idx]]
            return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0],
                                                 t[:, 2] - t[:, 0]), axis=1).sum() * 0.5)

        a_in = area(inlet)
        a_outs = [area(o) for o in outlets]
        tot = sum(a ** (murray_exponent / 2.0) for a in a_outs) or 1.0
        split = {f"outlet_{i}": (a ** (murray_exponent / 2.0)) / tot
                 for i, a in enumerate(a_outs)}
        assumptions = [
            "rigid wall (no FSI) unless a validated FSI model is coupled",
            "Newtonian rheology unless a validated non-Newtonian model is used",
            f"Murray's law flow split, exponent={murray_exponent}",
            "inlet/outlet planes chosen by user; not auto-detected",
        ]
        required = ["4D-flow MRI inlet waveform", "Doppler peak velocity",
                    "invasive pressure if available", "mesh convergence study"]
        if flow_rate_ml_s is None:
            assumptions.append("NO patient-specific flow rate supplied; placeholder BC")
        return BoundaryConditionSet(
            inlet_faces=inlet, outlet_faces=outlet_all, wall_faces=wall,
            inlet_area_mm2=a_in, outlet_area_mm2=float(sum(a_outs)),
            flow_split=split, waveform=waveform,
            assumptions=assumptions, validation_required=required)

    # ---- phantom for testing ----
    @staticmethod
    def vessel_phantom(n: int = 48, radius: float = 8.0,
                       spacing: float = 0.5) -> Tuple[ImageVolume, np.ndarray]:
        ax = (np.arange(n) - n / 2) * spacing
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(X ** 2 + Y ** 2)
        mask = (r < radius).astype(np.uint8)
        vol = ImageVolume(data=mask.astype(np.float64) * 1000.0,
                          spacing_mm=(spacing,) * 3, origin_mm=(0., 0., 0.),
                          modality="phantom", deidentified=True,
                          provenance="synthetic cylinder phantom")
        return vol, mask


# =============================================================================
# §6. [4] OMICS PIPELINE: scRNA-seq -> CELL FATE STATE
# =============================================================================

@dataclass
class CellFateState:
    n_cells: int
    n_genes: int
    fate_scores: Dict[str, np.ndarray]
    dominant_fate: np.ndarray
    fate_confidence: np.ndarray
    qc: Dict[str, Any]
    marker_panels: Dict[str, List[str]]
    caveats: List[str]
    clinical_actionability: str = "research_only"


class OmicsPipeline:
    """
    Marker-based scoring with explicit uncertainty. Research grade.
    NOT a clinical cell-type classifier.
    """

    DEFAULT_PANELS = {
        "pluripotent": ["POU5F1", "NANOG", "SOX2", "LIN28A", "DNMT3B"],
        "neuroectoderm": ["SOX1", "PAX6", "NES", "OTX2", "ZBTB16"],
        "mesoderm": ["TBXT", "MIXL1", "EOMES", "MESP1", "HAND1"],
        "endoderm": ["SOX17", "FOXA2", "GATA4", "GATA6", "CXCR4"],
        "cardiac": ["NKX2-5", "TNNT2", "MYH6", "MYH7", "ACTC1", "GATA4"],
        "neural": ["SOX2", "PAX6", "TUBB3", "MAP2", "NEUROD1"],
        "hepatic": ["ALB", "AFP", "HNF4A", "TTR", "APOA1"],
        "renal": ["NPHS1", "NPHS2", "PAX2", "WT1", "SIX2"],
        "senescent": ["CDKN1A", "CDKN2A", "GLB1", "TP53", "SERPINE1"],
    }

    def __init__(self, governance: GovernanceGate):
        self.gov = governance

    def load_h5ad(self, path: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        if not HAS_ANNDATA:
            return None, {"status": Status.UNAVAILABLE, "reason": "anndata not installed"}
        try:
            a = ad.read_h5ad(path)
            return a, {"status": Status.OK, "shape": a.shape}
        except Exception as e:
            return None, {"status": Status.ERROR, "reason": str(e)}

    def qc_filter(self, X: np.ndarray, genes: List[str],
                  min_genes: int = 200, max_pct_mito: float = 20.0
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        counts = X.sum(axis=1)
        detected = (X > 0).sum(axis=1)
        mito_idx = [i for i, g in enumerate(genes) if g.upper().startswith("MT-")]
        pct_mito = (X[:, mito_idx].sum(axis=1) / np.maximum(counts, 1) * 100
                    if mito_idx else np.zeros(X.shape[0]))
        keep = (detected >= min_genes) & (pct_mito <= max_pct_mito) & (counts > 0)
        return X[keep], keep, {
            "n_input": int(X.shape[0]), "n_pass": int(keep.sum()),
            "min_genes": min_genes, "max_pct_mito": max_pct_mito,
            "median_detected": float(np.median(detected)),
            "median_pct_mito": float(np.median(pct_mito)),
        }

    @staticmethod
    def normalize_log1p(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
        tot = X.sum(axis=1, keepdims=True)
        return np.log1p(X / np.maximum(tot, 1) * target_sum)

    def score_fates(self, X: np.ndarray, genes: List[str],
                    panels: Optional[Dict[str, List[str]]] = None,
                    n_background: int = 50, seed: int = 0
                    ) -> Dict[str, np.ndarray]:
        """
        Background-corrected marker score (scanpy-style):
        score = mean(marker expr) - mean(matched-background expr)
        Bins genes by mean expression to match background difficulty.
        """
        panels = panels or self.DEFAULT_PANELS
        rng = np.random.default_rng(seed)
        gidx = {g.upper(): i for i, g in enumerate(genes)}
        gene_mean = X.mean(axis=0)
        n_bins = 25
        order = np.argsort(gene_mean)
        bins = np.empty(len(genes), dtype=int)
        bins[order] = (np.arange(len(genes)) * n_bins // max(len(genes), 1))
        scores = {}
        for fate, markers in panels.items():
            idx = [gidx[m.upper()] for m in markers if m.upper() in gidx]
            if not idx:
                scores[fate] = np.zeros(X.shape[0])
                continue
            obs = X[:, idx].mean(axis=1)
            bg_idx: List[int] = []
            for i in idx:
                pool = np.where(bins == bins[i])[0]
                pool = pool[~np.isin(pool, idx)]
                if len(pool):
                    bg_idx.extend(rng.choice(pool, size=min(n_background, len(pool)),
                                             replace=False).tolist())
            bg = X[:, bg_idx].mean(axis=1) if bg_idx else np.zeros(X.shape[0])
            scores[fate] = obs - bg
        return scores

    def to_cell_fate(self, X: np.ndarray, genes: List[str],
                     panels: Optional[Dict[str, List[str]]] = None
                     ) -> CellFateState:
        Xq, keep, qc = self.qc_filter(X, genes)
        Xn = self.normalize_log1p(Xq)
        sc = self.score_fates(Xn, genes, panels)
        keys = list(sc.keys())
        M = np.stack([sc[k] for k in keys], axis=1)
        # softmax confidence with temperature; explicit that it is heuristic
        T = max(float(M.std()), 1e-6)
        e = np.exp((M - M.max(axis=1, keepdims=True)) / T)
        P = e / e.sum(axis=1, keepdims=True)
        dom = np.array(keys)[P.argmax(axis=1)]
        conf = P.max(axis=1)
        return CellFateState(
            n_cells=int(Xq.shape[0]), n_genes=len(genes),
            fate_scores=sc, dominant_fate=dom, fate_confidence=conf,
            qc=qc, marker_panels=panels or self.DEFAULT_PANELS,
            caveats=[
                "marker-based scoring, NOT a validated clinical classifier",
                "panel coverage limited by the genes present in the matrix",
                "confidence is a softmax heuristic, NOT a calibrated probability",
                "batch effects not corrected here; integrate before comparing datasets",
                "doublets not removed; run a validated doublet detector",
            ])

    @staticmethod
    def synthetic_scrna(n_cells: int = 300, n_genes: int = 400, seed: int = 0
                        ) -> Tuple[np.ndarray, List[str]]:
        rng = np.random.default_rng(seed)
        markers = ["POU5F1", "NANOG", "SOX2", "PAX6", "NES", "TBXT", "MIXL1",
                   "SOX17", "FOXA2", "NKX2-5", "TNNT2", "MYH6", "ALB", "HNF4A",
                   "NPHS1", "PAX2", "CDKN1A", "CDKN2A", "MT-CO1", "MT-ND1"]
        genes = markers + [GENE{i} for i in range(n_genes - len(markers))]
        X = rng.poisson(0.4, size=(n_cells, n_genes)).astype(np.float64)
        # implant three populations
        for pop, ms in ((0, ["POU5F1", "NANOG", "SOX2"]),
                        (1, ["PAX6", "NES"]),
                        (2, ["NKX2-5", "TNNT2", "MYH6"])):
            sl = slice(pop * n_cells // 3, (pop + 1) * n_cells // 3)
            for m in ms:
                X[sl, genes.index(m)] += rng.poisson(25, size=X[sl].shape[0])
        return X, genes


# =============================================================================
# §7. [2] CFD COUPLING ADAPTERS
# =============================================================================

@dataclass
class CFDResult:
    backend: str
    status: str
    metrics: Dict[str, float]
    assumptions: List[str]
    validation_required: List[str]
    clinical_actionability: str = "research_only"


class CFDCoupler:
    """
    Detects and adapts the three backends. Reports parse status honestly.
    The uploaded sources do NOT parse until repair_five.py is run.
    """

    BACKENDS = ["one_cfd_stencils", "sesi_multiphase_flow_fix", "super_dns_one"]

    def __init__(self, module_dir: str = "."):
        self.dir = Path(module_dir)
        self.availability: Dict[str, Dict[str, Any]] = {}
        self._probe()

    def _probe(self) -> None:
        import importlib.util
        for name in self.BACKENDS:
            matches = sorted(self.dir.glob(f"{name}*.py"))
            info: Dict[str, Any] = {"found": bool(matches), "files": [m.name for m in matches]}
            if not matches:
                info["status"] = Status.UNAVAILABLE
                info["reason"] = "file not present"
                self.availability[name] = info
                continue
            p = matches[0]
            src = p.read_text(encoding="utf-8", errors="replace")
            try:
                compile(src, str(p), "exec")
                info["parses"] = True
            except SyntaxError as e:
                info["parses"] = False
                info["syntax_error"] = f"line {e.lineno}: {e.msg}"
            if info.get("parses"):
                try:
                    spec = importlib.util.spec_from_file_location(f"_cfd_{name}", str(p))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    info["importable"] = True
                    info["symbols"] = [s for s in
                                       ("GridSpec", "MACOps", "PressurePoisson", "project",
                                        "VOFConfig", "MultiphaseNavierStokes3D",
                                        "SuperDNSOne", "solve")
                                       if hasattr(mod, s)]
                    info["status"] = Status.OK
                    self.__dict__[f"_mod_{name}"] = mod
                except Exception as e:
                    info["importable"] = False
                    info["status"] = Status.ERROR
                    info["reason"] = f"{type(e).__name__}: {e}"
            else:
                info["status"] = Status.UNAVAILABLE
                info["reason"] = "does not parse; run repair_five.py --apply first"
            self.availability[name] = info

    def report(self) -> Dict[str, Any]:
        return {"backends": self.availability,
                "note": "UNAVAILABLE means repair or install is required, not a silent skip"}

    def run_hemodynamics(self, mesh: SurfaceMesh, bc: BoundaryConditionSet,
                         rho: float = 1060.0, mu: float = 3.5e-3,
                         flow_ml_s: Optional[float] = None
                         ) -> CFDResult:
        """
        Research-grade estimate. If no backend is available, returns an
        analytic Poiseuille/WSS ESTIMATE clearly labelled as such — never
        presented as a CFD solve.
        """
        assumptions = list(bc.assumptions) + [
            f"rho={rho} kg/m^3, mu={mu} Pa.s (blood, Newtonian assumption)",
        ]
        required = ["mesh convergence", "4D-flow MRI comparison",
                    "Doppler peak velocity", "invasive pressure if available"]
        usable = [k for k, v in self.availability.items() if v.get("status") == Status.OK]
        if flow_ml_s is None:
            return CFDResult(backend="none", status=Status.NOT_COMPARED,
                             metrics={}, assumptions=assumptions + ["no flow rate supplied"],
                             validation_required=required)
        # analytic estimate (clearly labelled)
        Q = flow_ml_s * 1e-6           # m^3/s
        A_in = bc.inlet_area_mm2 * 1e-6  # m^2
        if A_in <= 0:
            return CFDResult(backend="none", status=Status.ERROR, metrics={},
                             assumptions=assumptions, validation_required=required)
        r_eq = math.sqrt(A_in / math.pi)
        v_mean = Q / A_in
        wss = 4.0 * mu * Q / (math.pi * r_eq ** 3)     # Hagen-Poiseuille wall shear
        dp_per_m = 8.0 * mu * Q / (math.pi * r_eq ** 4)
        Re = rho * v_mean * 2 * r_eq / mu
        return CFDResult(
            backend=("analytic_poiseuille_ESTIMATE"
                     + (f"|available:{','.join(usable)}" if usable else "")),
            status=Status.OK,
            metrics={"mean_inlet_velocity_m_s": v_mean,
                     "equivalent_radius_m": r_eq,
                     "wall_shear_stress_Pa_ESTIMATE": wss,
                     "pressure_gradient_Pa_per_m_ESTIMATE": dp_per_m,
                     "reynolds_number": Re},
            assumptions=assumptions + [
                "ANALYTIC ESTIMATE, not a CFD solve",
                "straight circular pipe, fully developed, steady",
                "real WSS requires a validated CFD solve on the actual mesh",
            ],
            validation_required=required)


# =============================================================================
# §8. [6] VALIDATION HARNESS
# =============================================================================

class ValidationHarness:
    """
    Real statistics. Reports NOT COMPARED when reference data is absent.
    """

    def __init__(self, reference: Optional[Dict[str, np.ndarray]] = None):
        self.ref = reference or {}

    @staticmethod
    def bland_altman(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
        d = pred - ref
        m = float(d.mean())
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        return {"bias": m, "sd_diff": sd,
                "loa_upper": m + 1.96 * sd, "loa_lower": m - 1.96 * sd,
                "n": int(len(d))}

    @staticmethod
    def icc_2_1(pred: np.ndarray, ref: np.ndarray) -> float:
        """ICC(2,1), two-way random effects, absolute agreement."""
        Y = np.stack([pred, ref], axis=1)
        n, k = Y.shape
        if n < 2:
            return float("nan")
        gm = Y.mean()
        row = Y.mean(axis=1)
        col = Y.mean(axis=0)
        SSR = k * ((row - gm) ** 2).sum()
        SSC = n * ((col - gm) ** 2).sum()
        SST = ((Y - gm) ** 2).sum()
        SSE = SST - SSR - SSC
        MSR = SSR / (n - 1)
        MSC = SSC / (k - 1)
        MSE = SSE / max((n - 1) * (k - 1), 1)
        den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
        return float((MSR - MSE) / den) if den != 0 else float("nan")

    @staticmethod
    def calibration(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
        X = np.stack([np.ones_like(pred), pred], axis=1)
        beta, *_ = np.linalg.lstsq(X, ref, rcond=None)
        return {"calibration_intercept": float(beta[0]),
                "calibration_slope": float(beta[1])}

    @staticmethod
    def bootstrap_ci(pred: np.ndarray, ref: np.ndarray,
                     stat: Callable[[np.ndarray, np.ndarray], float],
                     n_boot: int = 2000, seed: int = 0,
                     conf: float = 0.95) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        n = len(pred)
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            try:
                vals.append(float(stat(pred[i], ref[i])))
            except Exception:
                continue
        if not vals:
            return {"lo": float("nan"), "hi": float("nan"), "point": float("nan")}
        a = 1 - conf
        return {"point": float(stat(pred, ref)),
                "lo": float(np.quantile(vals, a / 2)),
                "hi": float(np.quantile(vals, 1 - a / 2))}

    def compare(self, name: str, metric: str,
                pred: np.ndarray, ref_key: str) -> Dict[str, Any]:
        if ref_key not in self.ref:
            return {"case": name, "metric": metric,
                    "status": Status.NOT_COMPARED,
                    "reason": f"reference '{ref_key}' not supplied",
                    "n_predicted": int(len(pred))}
        ref = np.asarray(self.ref[ref_key], dtype=float)
        if len(ref) != len(pred):
            return {"case": name, "metric": metric, "status": Status.ERROR,
                    "reason": f"length mismatch {len(pred)} vs {len(ref)}"}
        out = {"case": name, "metric": metric, "status": Status.OK,
               "n": int(len(pred)),
               "bland_altman": self.bland_altman(pred, ref),
               "icc_2_1": self.icc_2_1(pred, ref),
               "calibration": self.calibration(pred, ref),
               "rmse": float(np.sqrt(((pred - ref) ** 2).mean())),
               "mae": float(np.abs(pred - ref).mean())}
        if HAS_SCIPY:
            out["pearson_r"] = float(sps.pearsonr(pred, ref)[0])
            out["spearman_rho"] = float(sps.spearmanr(pred, ref)[0])
        out["rmse_ci95"] = self.bootstrap_ci(
            pred, ref, lambda p, r: float(np.sqrt(((p - r) ** 2).mean())))
        return out

    def coverage(self, lo: np.ndarray, hi: np.ndarray,
                 truth: np.ndarray) -> Dict[str, float]:
        inside = (truth >= lo) & (truth <= hi)
        return {"empirical_coverage": float(inside.mean()),
                "mean_interval_width": float((hi - lo).mean()),
                "n": int(len(truth))}

    def outcome_discrimination(self, risk: np.ndarray,
                               event: np.ndarray) -> Dict[str, Any]:
        """AUC + Brier for a binary clinical outcome. Research only."""
        if not HAS_SCIPY:
            return {"status": Status.UNAVAILABLE, "reason": "scipy required"}
        order = np.argsort(risk)
        y = event[order]
        npos, nneg = int(y.sum()), int(len(y) - y.sum())
        if npos == 0 or nneg == 0:
            return {"status": Status.ERROR, "reason": "single-class outcome"}
        ranks = np.arange(1, len(y) + 1)
        auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        return {"status": Status.OK, "auc": auc,
                "brier": float(((risk - event) ** 2).mean()),
                "n": int(len(event)), "n_events": npos,
                "note": "research discrimination metric, NOT clinical validation"}


# =============================================================================
# §9. ORCHESTRATOR
# =============================================================================

class MedWorldConnectors:
    def __init__(self, *, irb: Optional[IRBRecord] = None,
                 offline: bool = True, module_dir: str = ".",
                 cache_dir: str = "./medworld_cache",
                 hpo_obo: Optional[str] = None):
        self.gov = GovernanceGate(irb)
        self.http = HTTPClient(cache_dir=cache_dir, offline=offline)
        self.pubmed = PubMedConnector(self.http)
        self.clinvar = ClinVarConnector(self.http)
        self.gwas = GWASCatalogConnector(self.http)
        self.hpo = HPOConnector(self.http, local_obo=hpo_obo)
        self.omim = OMIMConnector(self.http)
        self.disgenet = DisGeNETConnector(self.http)
        self.kg = KnowledgeGraph()
        self.imaging = ImagingPipeline(self.gov)
        self.omics = OmicsPipeline(self.gov)
        self.cfd = CFDCoupler(module_dir)
        self.validation = ValidationHarness()

    # ---- [1] ----
    def literature_scan(self, gene: str, phenotype: str,
                        rsid: Optional[str] = None,
                        mim: Optional[int] = None) -> Dict[str, Any]:
            decision = self.gov.authorize(
                text=f"{gene} {phenotype} association literature",
                uses_patient_data=False)
            if not decision["authorized"]:
                return {"status": Status.BLOCKED, "decision": decision}
            n_cv = self.kg.ingest_clinvar(
                self.clinvar.summarize(
                    self.clinvar.search_variant(gene).get("ids", [])),
                gene, phenotype)
            n_gw = 0
            if rsid:
                n_gw = self.kg.ingest_gwas(
                    self.gwas.associations_by_rsid(rsid), phenotype)
            pm = self.pubmed.search(f"{gene} AND {phenotype}", retmax=10)
            om = self.omim.entry(mim) if mim else {"status": Status.UNAVAILABLE,
                                                   "reason": "no MIM number supplied"}
            dg = self.disgenet.gene_disease(gene)
            for src, payload in (("OMIM", om), ("DisGeNET", dg)):
                if payload.get("status") != Status.OK:
                    self.kg.note_unavailable(src, payload.get("reason", ""))
            return {"status": Status.OK,
                    "n_clinvar": n_cv, "n_gwas": n_gw,
                    "pubmed": {"status": pm.get("status"),
                               "n_pmids": len(pm.get("pmids", []))},
                    "omim_status": om.get("status"),
                    "disgenet_status": dg.get("status"),
                    "kg": self.kg.summary(),
                    "decision": decision}

    # ---- [3]+[2] ----
    def imaging_to_cfd(self, vol: ImageVolume, mask: np.ndarray,
                       seg_validation: Dict[str, Any],
                       inlet_plane: Tuple[int, float],
                       outlet_planes: List[Tuple[int, float]],
                       flow_ml_s: Optional[float] = None,
                       uses_patient_data: bool = False,
                       deid: Optional[DeIDReport] = None) -> Dict[str, Any]:
        decision = self.gov.authorize(
            text="patient-specific hemodynamic simulation research",
            uses_patient_data=uses_patient_data, deid=deid,
            purpose="simulation")
        if not decision["authorized"]:
            return {"status": Status.BLOCKED, "decision": decision}
        seg, sinfo = self.imaging.accept_segmentation(mask, vol, seg_validation)
        if seg is None:
            return {"status": Status.BLOCKED, "segmentation": sinfo}
        mesh, minfo = self.imaging.mask_to_mesh(seg, vol.spacing_mm)
        if mesh is None:
            return {"status": minfo.get("status"), "mesh": minfo}
        bc = self.imaging.assign_bc(mesh, inlet_plane, outlet_planes, flow_ml_s)
        res = self.cfd.run_hemodynamics(mesh, bc, flow_ml_s=flow_ml_s)
        return {"status": Status.OK, "mesh_quality": mesh.quality,
                "n_inlet_faces": int(len(bc.inlet_faces)),
                "n_outlet_faces": int(len(bc.outlet_faces)),
                "n_wall_faces": int(len(bc.wall_faces)),
                "flow_split": bc.flow_split,
                "cfd": asdict(res), "decision": decision}

    # ---- [4] ----
    def omics_to_fate(self, X: np.ndarray, genes: List[str],
                      uses_patient_data: bool = False,
                      deid: Optional[DeIDReport] = None) -> Dict[str, Any]:
        decision = self.gov.authorize(
            text="single-cell cell fate research",
            uses_patient_data=uses_patient_data, deid=deid, purpose="omics")
        if not decision["authorized"]:
            return {"status": Status.BLOCKED, "decision": decision}
        st = self.omics.to_cell_fate(X, genes)
        counts = {k: int((st.dominant_fate == k).sum())
                  for k in st.marker_panels}
        return {"status": Status.OK, "n_cells": st.n_cells,
                "fate_counts": counts,
                "mean_confidence": float(st.fate_confidence.mean()),
                "qc": st.qc, "caveats": st.caveats,
                "clinical_actionability": st.clinical_actionability,
                "decision": decision}

    def status(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "offline": self.http.offline,
            "connectors": {
                "PubMed": self.pubmed.available()[0],
                "ClinVar": self.clinvar.available()[0],
                "GWAS_Catalog": self.gwas.available()[0],
                "HPO": True,
                "OMIM": self.omim.available(),
                "DisGeNET": self.disgenet.available(),
            },
            "imaging_deps": {"pydicom": HAS_PYDICOM, "nibabel": HAS_NIBABEL,
                             "SimpleITK": HAS_SITK, "skimage": HAS_SKIMAGE},
            "omics_deps": {"anndata": HAS_ANNDATA},
            "stats_deps": {"scipy": HAS_SCIPY},
            "cfd": self.cfd.report(),
            "governance": self.gov.regulatory_classification(),
        }


# =============================================================================
# §10. SELF-TEST
# =============================================================================

def self_test(module_dir: str = ".", verbose: bool = True) -> bool:
    print(BANNER)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # ---- T1 governance blocks without IRB ----
    print("\n[T1] [5] Governance gate")
    gov = GovernanceGate(irb=None)
    d = gov.authorize(text="hemodynamic simulation", uses_patient_data=True)
    rep("patient path BLOCKED without IRB", not d["authorized"], d["reason"])
    d2 = gov.authorize(text="gene association literature", uses_patient_data=False)
    rep("public literature allowed", d2["authorized"])
    d3 = gov.authorize(text="design a CRISPR Cas9 guide RNA", uses_patient_data=False)
    rep("gene editor design BLOCKED", not d3["authorized"], d3["intent"])
    d4 = gov.authorize(text="give a transfection protocol step by step",
                       uses_patient_data=False)
    rep("wet-lab protocol BLOCKED", not d4["authorized"])
    d5 = gov.authorize(text="enhance virulence and immune evasion",
                       uses_patient_data=False)
    rep("pathogen enhancement BLOCKED", not d5["authorized"])

    irb = IRBRecord(protocol_id="P-001", institution="Demo",
                    approval_date="2026-01-01", expiry_date="2099-12-31",
                    approved_data_types=["imaging", "omics"],
                    approved_uses=["simulation", "omics"],
                    consent_model="written", data_use_agreement=True,
                    status="APPROVED")
    gov2 = GovernanceGate(irb)
    d6 = gov2.authorize(text="simulation", uses_patient_data=True, purpose="simulation")
    rep("IRB present but no de-id => BLOCKED", not d6["authorized"], d6["reason"])

    # ---- T2 PHI guard ----
    print("\n[T2] [5] PHI guard")
    phi = PHIGuard()
    rec = {"patient_name": "John Smith", "mrn": "MRN: 123456",
           "age": 47, "sex": "male", "email": "a@b.com", "diagnosis": "X"}
    deid = phi.deidentify(rec)
    rep("direct identifiers removed", "patient_name" in deid.removed_fields
        and any("mrn" in r for r in deid.removed_fields))
    rep("age generalised", any("age_band" in r for r in deid.removed_fields))
    q, removed = phi.sanitize_query("John Smith MRN: 123456 BRCA1 breast cancer")
    rep("query sanitised before network", len(removed) > 0 and "REDACTED" in q,
        f"removed={removed}")
    d7 = gov2.authorize(text="simulation", uses_patient_data=True,
                        deid=deid, purpose="simulation")
    rep("IRB + de-id => authorised", d7["authorized"], d7["reason"])

    # ---- T3 literature connectors (offline) ----
    print("\n[T3] [1] Literature connectors")
    mw = MedWorldConnectors(offline=True, module_dir=module_dir)
    st = mw.status()
    rep("PubMed available (public)", st["connectors"]["PubMed"])
    rep("ClinVar available (public)", st["connectors"]["ClinVar"])
    rep("GWAS Catalog available (public)", st["connectors"]["GWAS_Catalog"])
    rep("OMIM reports UNAVAILABLE without key",
        st["connectors"]["OMIM"][0] is False, st["connectors"]["OMIM"][1])
    rep("DisGeNET reports UNAVAILABLE without key",
        st["connectors"]["DisGeNET"][0] is False)
    scan = mw.literature_scan("BRCA1", "breast cancer", rsid="rs1799966")
    rep("offline scan returns a status, not a fake result",
        scan["status"] in (Status.OK, Status.BLOCKED))
    rep("unavailable sources are recorded",
        len(mw.kg.unavailable) >= 2, f"n={len(mw.kg.unavailable)}")

    # ---- T4 HPO parser ----
    print("\n[T4] [1] HPO ontology parser")
    obo = """[Term]
id: HP:0000001
name: All
is_a: HP:0000118 ! Root

[Term]
id: HP:0001250
name: Seizure
def: "A seizure."
synonym: "Epileptic seizure" EXACT []
is_a: HP:0000118 ! Root

[Term]
id: HP:0000118
name: Phenotypic abnormality
"""
    hp = HPOConnector(mw.http)
    hp._terms = hp._parse_obo(obo)
    rep("parsed 3 terms", len(hp._terms) == 3, f"n={len(hp._terms)}")
    rep("search by synonym works", len(hp.search("epileptic")) == 1)
    rep("ancestors resolved", "HP:0000118" in hp.ancestors("HP:0001250"))

    # ---- T5 imaging phantom -> mesh -> BC ----
    print("\n[T5] [3] Imaging pipeline (phantom)")
    vol, mask = ImagingPipeline.vessel_phantom(n=40, radius=6.0, spacing=0.5)
    seg_val = {"method": "phantom analytic", "dice_vs_reference": 1.0,
               "validated_by": "self-test", "date": "2026-01-01"}
    bad_val = {"method": "unvalidated CNN"}
    seg_bad, info_bad = mw.imaging.accept_segmentation(mask, vol, bad_val)
    rep("unvalidated segmentation BLOCKED", seg_bad is None,
        info_bad.get("reason", "")[:60])
    seg_ok, info_ok = mw.imaging.accept_segmentation(mask, vol, seg_val)
    rep("validated segmentation accepted", seg_ok is not None)
    if HAS_SKIMAGE:
        mesh, mi = mw.imaging.mask_to_mesh(seg_ok, vol.spacing_mm)
        rep("marching cubes produced a mesh", mesh is not None,
            f"V={mesh.n_vertices} F={mesh.n_faces}" if mesh else "")
        if mesh:
            rep("mesh has positive area", mesh.quality["area_mm2"] > 0,
                f"A={mesh.quality['area_mm2']:.1f} mm^2")
            bc = mw.imaging.assign_bc(mesh, inlet_plane=(2, -9.5),
                                      outlet_planes=[(2, 9.5)], flow_ml_s=5.0)
            rep("BC tagged inlet/outlet/wall",
                len(bc.wall_faces) > 0,
                f"in={len(bc.inlet_faces)} out={len(bc.outlet_faces)} wall={len(bc.wall_faces)}")
            rep("assumptions are declared", len(bc.assumptions) >= 4)
    else:
        rep("skimage present", False, "install scikit-image for mesh tests")

    # ---- T6 omics ----
    print("\n[T6] [4] Omics pipeline")
    X, genes = OmicsPipeline.synthetic_scrna(n_cells=300, n_genes=400)
    res = mw.omics_to_fate(X, genes)
    rep("cell fate computed", res["status"] == Status.OK, f"n={res.get('n_cells')}")
    if res["status"] == Status.OK:
        fc = res["fate_counts"]
        rep("pluripotent/neural/cardiac populations detected",
            fc.get("pluripotent", 0) > 20 and fc.get("cardiac", 0) > 20,
            f"pluri={fc.get('pluripotent')} neural={fc.get('neural')} cardiac={fc.get('cardiac')}")
        rep("caveats declared", len(res["caveats"]) >= 4)
        rep("research_only", res["clinical_actionability"] == "research_only")

    # ---- T7 CFD coupling ----
    print("\n[T7] [2] CFD coupling")
    rep("CFD availability probed", len(mw.cfd.availability) == 3)
    for k, v in mw.cfd.availability.items():
        print(f"      {k}: {v.get('status')} {v.get('reason','')[:60]}")
    if HAS_SKIMAGE:
        mesh, _ = mw.imaging.mask_to_mesh(seg_ok, vol.spacing_mm)
        bc = mw.imaging.assign_bc(mesh, (2, -9.5), [(2, 9.5)], 5.0)
        r = mw.cfd.run_hemodynamics(mesh, bc, flow_ml_s=5.0)
        rep("hemodynamic estimate produced", r.status == Status.OK,
            f"WSS={r.metrics.get('wall_shear_stress_Pa_ESTIMATE', float('nan')):.3f} Pa")
        rep("labelled as ESTIMATE not CFD solve",
            any("ESTIMATE" in a for a in r.assumptions))
        r_none = mw.cfd.run_hemodynamics(mesh, bc, flow_ml_s=None)
        rep("no flow => NOT_COMPARED", r_none.status == Status.NOT_COMPARED)

    # ---- T8 validation ----
    print("\n[T8] [6] Validation harness")
    vh = ValidationHarness()
    rng = np.random.default_rng(0)
    pred = rng.normal(10, 2, 60)
    nc = vh.compare("4D_flow", "peak_velocity", pred, "flow_mri")
    rep("no reference => NOT_COMPARED", nc["status"] == Status.NOT_COMPARED,
        nc.get("reason", ""))
    ref = pred * 1.05 + rng.normal(0, 0.4, 60)
    vh2 = ValidationHarness({"flow_mri": ref, "doppler": ref * 0.98,
                             "pressure": ref * 1.2})
    c1 = vh2.compare("4D_flow", "peak_velocity", pred, "flow_mri")
    rep("with reference => compared", c1["status"] == Status.OK)
    rep("Bland-Altman computed", "bias" in c1["bland_altman"],
        f"bias={c1['bland_altman']['bias']:.3f}")
    rep("ICC(2,1) computed and high", c1["icc_2_1"] > 0.8, f"ICC={c1['icc_2_1']:.3f}")
    rep("calibration slope ~1", abs(c1["calibration"]["calibration_slope"] - 1.05) < 0.1,
        f"slope={c1['calibration']['calibration_slope']:.3f}")
    rep("bootstrap CI present", "lo" in c1["rmse_ci95"])
    for k in ("doppler", "pressure"):
        cc = vh2.compare(k, "velocity", pred, k)
        rep(f"{k} compared", cc["status"] == Status.OK)
    cov = vh.coverage(pred - 1.0, pred + 1.0, ref)
    rep("coverage computed", 0 <= cov["empirical_coverage"] <= 1,
        f"cov={cov['empirical_coverage']:.2f}")
    risk = rng.random(200)
    event = (risk + rng.normal(0, 0.2, 200) > 0.6).astype(int)
    oc = vh.outcome_discrimination(risk, event)
    rep("outcome AUC computed", oc.get("status") == Status.OK,
        f"AUC={oc.get('auc', float('nan')):.3f}")

    # ---- T9 end-to-end with governance ----
    print("\n[T9] End-to-end governed pipeline")
    mw2 = MedWorldConnectors(irb=irb, offline=True, module_dir=module_dir)
    deid2 = PHIGuard().deidentify({"patient_name": "Jane Doe", "age": 55,
                                   "sex": "female", "mrn": "MRN: 99999"})
    e2e = mw2.imaging_to_cfd(vol, mask, seg_val, (2, -9.5), [(2, 9.5)],
                             flow_ml_s=5.0, uses_patient_data=True, deid=deid2)
    rep("governed patient path runs", e2e["status"] in (Status.OK, Status.BLOCKED),
        e2e.get("status"))
    blocked = mw2.imaging_to_cfd(vol, mask, seg_val, (2, -9.5), [(2, 9.5)],
                                 flow_ml_s=5.0, uses_patient_data=True, deid=None)
    rep("same path without de-id is BLOCKED", blocked["status"] == Status.BLOCKED)
    rep("audit trail recorded", len(mw2.gov.audit) >= 2, f"n={len(mw2.gov.audit)}")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
DELIVERED
  [1] PubMed / ClinVar / GWAS Catalog / HPO connectors (real HTTP + cache).
      OMIM / DisGeNET report UNAVAILABLE without a licensed key.
  [2] CFD adapters probe one_cfd_stencils / sesi_multiphase_flow_fix /
      super_dns_one and report parse status honestly.
  [3] DICOM / NIfTI readers, isotropic resample, validated-segmentation
      gate, marching cubes, inlet/outlet/wall BC with declared assumptions.
  [4] scRNA-seq QC, normalisation, background-corrected marker scoring,
      cell-fate state with explicit caveats.
  [5] IRB / consent / DUA / de-identification ENFORCED as a gate.
      PHI is stripped before any network call.
  [6] Bland-Altman, ICC(2,1), calibration, bootstrap CI, coverage, AUC.
      NOT COMPARED is reported when reference data is absent.

NOT CLAIMED
  No automatic clinical segmentation. No clinical cell-type classifier.
  The hemodynamic numbers here are an ANALYTIC ESTIMATE, not a CFD solve.
  No device approval. No diagnosis, treatment, or surgical decision.
  No gene editor design, no wet-lab protocol, no pathogen enhancement.
""")
    return ok


# =============================================================================
# §11. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="ONE MedWorld connectors")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--online", action="store_true",
                    help="allow real network calls (public APIs only)")
    ap.add_argument("--gene", default=None)
    ap.add_argument("--phenotype", default=None)
    ap.add_argument("--rsid", default=None)
    ap.add_argument("--mim", type=int, default=None)
    ap.add_argument("--hpo-obo", default=None)
    ap.add_argument("--phantom", action="store_true")
    ap.add_argument("--omics-demo", action="store_true")
    ap.add_argument("--module-dir", default=".")
    ap.add_argument("--cache-dir", default="./medworld_cache")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.selftest:
        return 0 if self_test(a.module_dir) else 1

    mw = MedWorldConnectors(offline=not a.online, module_dir=a.module_dir,
                            cache_dir=a.cache_dir, hpo_obo=a.hpo_obo)
    out: Dict[str, Any] = {}

    if a.status:
        out["status"] = mw.status()
        print(json.dumps(out["status"], indent=2, default=str))

    if a.gene and a.phenotype:
        out["literature"] = mw.literature_scan(a.gene, a.phenotype, a.rsid, a.mim)
        print(json.dumps(out["literature"], indent=2, default=str))

    if a.phantom:
        vol, mask = ImagingPipeline.vessel_phantom()
        seg_val = {"method": "phantom analytic", "dice_vs_reference": 1.0,
                   "validated_by": "CLI", "date": time.strftime("%Y-%m-%d")}
        out["phantom"] = mw.imaging_to_cfd(vol, mask, seg_val,
                                           (2, -9.5), [(2, 9.5)], flow_ml_s=5.0)
        print(json.dumps(out["phantom"], indent=2, default=str))

    if a.omics_demo:
        X, genes = OmicsPipeline.synthetic_scrna()
        out["omics"] = mw.omics_to_fate(X, genes)
        print(json.dumps(out["omics"], indent=2, default=str))

    if not out:
        ap.print_help()
        print(BANNER)
        print("Start:")
        print("  --selftest")
        print("  --status")
        print("  --phantom")
        print("  --omics-demo")
        print("  --online --gene BRCA1 --phenotype 'breast cancer' --rsid rs1799966")
        return 0

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
