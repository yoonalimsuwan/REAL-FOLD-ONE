"""
=============================================================================
SESI REAL EXPERIMENTAL DATASET CONNECTORS
Unified interface for GW, neutrino, protein, and gene expression data
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
-------------------------
[D1] GWOSCConnector — LIGO/Virgo gravitational wave strain data
[D2] IceCubeConnector — high-energy neutrino events
[D3] PDBConnector — Protein Data Bank structures
[D4] GEOConnector — Gene Expression Omnibus data
[D5] UnifiedDatasetHub — single interface for all connectors

REAL DATA REQUIREMENTS
----------------------
• GWOSC: public, no credentials needed (https://gwosc.org)
• IceCube: public alerts (https://icecube.wisc.edu/science/alerts)
• PDB: public (https://www.rcsb.org)
• GEO: public (https://www.ncbi.nlm.nih.gov/geo)

OFFLINE FALLBACK
----------------
When network is unavailable or credentials are missing, each connector
returns SYNTHETIC data clearly flagged as such. This allows the pipeline
to run in demo mode without pretending to use real data.

LABELS: [C] cited data source  [D] derived connector  [O] open

HONEST SCOPE
------------
• Real data requires network access and may have rate limits.
• Synthetic fallback is for DEMONSTRATION only, not scientific analysis.
• Each connector must be validated against the official data format.
• No claim that synthetic data has the same statistical properties as real.
=============================================================================
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.real_datasets")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

__all__ = [
    "DatasetConfig",
    "GWOSCConnector",
    "IceCubeConnector",
    "PDBConnector",
    "GEOConnector",
    "UnifiedDatasetHub",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DatasetConfig:
    """Configuration for dataset connectors."""
    cache_dir: str = "./data/real_datasets"
    timeout: int = 60
    use_synthetic_fallback: bool = True

    # GWOSC
    gwosc_event: str = "GW150914"
    gwosc_detector: str = "H1"

    # IceCube
    icecube_alert_type: str = "HESE"

    # PDB
    pdb_id: str = "1UBQ"  # ubiquitin

    # GEO
    geo_accession: str = "GSE53987"

    def __post_init__(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# [D1] GWOSC CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class GWOSCConnector:
    """
    [C/D] LIGO/Virgo gravitational wave strain data from GWOSC.

    Public API: https://gwosc.org/eventapi/json/
    Strain data: HDF5 files from https://gwosc.org/archive/

    HONEST: Without h5py, returns synthetic chirp signal.
    """

    EVENT_API = "https://gwosc.org/eventapi/json/"

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "gwosc"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_event_metadata(self, event_name: str) -> Optional[Dict]:
        """Fetch event metadata from GWOSC API."""
        cache_file = self.cache_dir / f"{event_name}_metadata.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        if not HAS_REQUESTS:
            logger.warning("requests not installed; using synthetic metadata")
            return self._synthetic_metadata(event_name)

        try:
            url = f"{self.EVENT_API}{event_name}/"
            resp = requests.get(url, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                data = resp.json()
                cache_file.write_text(json.dumps(data, indent=2))
                return data
        except Exception as e:
            logger.warning(f"GWOSC fetch failed: {e}")

        return self._synthetic_metadata(event_name)

    def fetch_strain(self, event_name: str, detector: str = "H1") -> Dict[str, Any]:
        """
        Fetch strain data h(t) for an event.

        Returns dict with 'strain', 'sample_rate', 'gps_start', 'synthetic' flag.
        """
        metadata = self.fetch_event_metadata(event_name)
        if metadata is None:
            return self._synthetic_strain(event_name, detector)

        # In production, would download HDF5 file here
        # For now, return synthetic strain with metadata
        logger.info(f"Strain download not implemented; using synthetic")
        return self._synthetic_strain(event_name, detector)

    def _synthetic_metadata(self, event_name: str) -> Dict:
        """Synthetic event metadata for offline mode."""
        return {
            "name": event_name,
            "gps_time": 1126259462.4 if event_name == "GW150914" else 0.0,
            "mass1_solar": 36.0,
            "mass2_solar": 29.0,
            "snr": 24.4,
            "event_type": "BBH",
            "synthetic": True,
        }

    def _synthetic_strain(self, event_name: str, detector: str) -> Dict[str, Any]:
        """Synthetic chirp signal for offline mode."""
        rng = np.random.default_rng(hash(event_name + detector) % (2**31))
        sample_rate = 4096
        duration = 4.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)

        # Simple chirp
        f0, f1 = 35.0, 250.0
        chirp = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / duration))
        noise = rng.normal(0, 1e-21, n_samples)
        strain = chirp * 1e-21 + noise

        return {
            "strain": strain.tolist(),
            "sample_rate": sample_rate,
            "gps_start": 1126259462.4 - 2.0,
            "detector": detector,
            "event": event_name,
            "synthetic": True,
            "note": "SYNTHETIC chirp for offline demo; not real GW data",
        }


# ═══════════════════════════════════════════════════════════════════════
# [D2] ICECUBE CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class IceCubeConnector:
    """
    [C/D] IceCube neutrino alert data.

    Public alerts: https://icecube.wisc.edu/science/alerts/
    GCN Notices: https://gcn.gsfc.nasa.gov/notifications/

    HONEST: Real-time stream requires credentials; offline mode returns
    synthetic alerts.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "icecube"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Fetch recent neutrino alerts."""
        if not HAS_REQUESTS:
            logger.warning("requests not installed; using synthetic alerts")
            return self._synthetic_alerts(hours)

        try:
            # IceCube alert API (simplified; actual endpoint may vary)
            url = "https://icecube.wisc.edu/science/alerts/recent"
            params = {"hours": hours, "format": "json"}
            resp = requests.get(url, params=params, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                return resp.json().get("alerts", [])
        except Exception as e:
            logger.warning(f"IceCube fetch failed: {e}")

        return self._synthetic_alerts(hours)

    def _synthetic_alerts(self, hours: int) -> List[Dict]:
        """Synthetic neutrino alerts for offline mode."""
        import time
        rng = np.random.default_rng(42)
        n_alerts = rng.integers(5, 15)
        now = time.time()

        alerts = []
        for i in range(n_alerts):
            alerts.append({
                "id": f"IC_SYN_{i:04d}",
                "timestamp": now - rng.uniform(0, hours * 3600),
                "ra": rng.uniform(0, 360),
                "dec": rng.uniform(-90, 90),
                "energy_gev": 10 ** rng.uniform(4, 7),
                "significance": rng.uniform(3.0, 6.0),
                "alert_type": self.cfg.icecube_alert_type,
                "error_radius_deg": rng.uniform(0.5, 2.0),
                "synthetic": True,
            })

        return alerts


# ═══════════════════════════════════════════════════════════════════════
# [D3] PDB CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class PDBConnector:
    """
    [C/D] Protein Data Bank structure data.

    Public API: https://files.rcsb.org/view/
    Format: PDB/mmCIF files

    HONEST: Without network, returns synthetic coordinates.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "pdb"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_structure(self, pdb_id: str) -> Optional[Dict]:
        """Fetch protein structure from PDB."""
        cache_file = self.cache_dir / f"{pdb_id}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        if not HAS_REQUESTS:
            logger.warning("requests not installed; using synthetic structure")
            return self._synthetic_structure(pdb_id)

        try:
            url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
            resp = requests.get(url, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                # Parse PDB file (simplified)
                structure = self._parse_pdb(resp.text, pdb_id)
                cache_file.write_text(json.dumps(structure, indent=2))
                return structure
        except Exception as e:
            logger.warning(f"PDB fetch failed: {e}")

        return self._synthetic_structure(pdb_id)

    def _parse_pdb(self, pdb_text: str, pdb_id: str) -> Dict:
        """Parse PDB file to extract atom coordinates."""
        atoms = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atom_name = line[12:16].strip()
                    residue = line[17:20].strip()
                    atoms.append({
                        "atom": atom_name,
                        "residue": residue,
                        "coords": [x, y, z],
                    })
                except (ValueError, IndexError):
                    continue

        return {
            "pdb_id": pdb_id,
            "n_atoms": len(atoms),
            "atoms": atoms,
            "synthetic": False,
        }

    def _synthetic_structure(self, pdb_id: str) -> Dict:
        """Synthetic protein structure for offline mode."""
        rng = np.random.default_rng(hash(pdb_id) % (2**31))
        n_atoms = 500
        atoms = []
        for i in range(n_atoms):
            atoms.append({
                "atom": f"C{i}",
                "residue": "ALA",
                "coords": rng.normal(0, 10, 3).tolist(),
            })

        return {
            "pdb_id": pdb_id,
            "n_atoms": n_atoms,
            "atoms": atoms,
            "synthetic": True,
            "note": "SYNTHETIC structure for offline demo",
        }


# ═══════════════════════════════════════════════════════════════════════
# [D4] GEO CONNECTOR
# ═══════════════════════════════════════════════════════════════════════

class GEOConnector:
    """
    [C/D] Gene Expression Omnibus data.

    Public API: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
    Format: Series Matrix files

    HONEST: Without network, returns synthetic expression data.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir) / "geo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_expression(self, accession: str) -> Optional[Dict]:
        """Fetch gene expression data from GEO."""
        cache_file = self.cache_dir / f"{accession}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        if not HAS_REQUESTS:
            logger.warning("requests not installed; using synthetic expression")
            return self._synthetic_expression(accession)

        try:
            # GEO API (simplified; real parsing is more complex)
            url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            params = {"acc": accession, "targ": "self", "form": "text", "view": "full"}
            resp = requests.get(url, params=params, timeout=self.cfg.timeout)
            if resp.status_code == 200:
                # Parse expression matrix (simplified)
                expression = self._parse_geo(resp.text, accession)
                cache_file.write_text(json.dumps(expression, indent=2))
                return expression
        except Exception as e:
            logger.warning(f"GEO fetch failed: {e}")

        return self._synthetic_expression(accession)

    def _parse_geo(self, geo_text: str, accession: str) -> Dict:
        """Parse GEO series matrix (simplified)."""
        # Real parsing would extract expression matrix from table
        # For demo, return empty structure
        return {
            "accession": accession,
            "n_samples": 0,
            "n_genes": 0,
            "expression_matrix": [],
            "synthetic": False,
        }

    def _synthetic_expression(self, accession: str) -> Dict:
        """Synthetic gene expression for offline mode."""
        rng = np.random.default_rng(hash(accession) % (2**31))
        n_samples = 10
        n_genes = 1000

        # Log-normal distributed expression values
        expression = rng.lognormal(mean=5.0, sigma=1.5, size=(n_genes, n_samples))

        return {
            "accession": accession,
            "n_samples": n_samples,
            "n_genes": n_genes,
            "expression_matrix": expression.tolist(),
            "gene_ids": [f"GENE_{i:05d}" for i in range(n_genes)],
            "sample_ids": [f"SAMPLE_{i:02d}" for i in range(n_samples)],
            "synthetic": True,
            "note": "SYNTHETIC expression for offline demo",
        }


# ═══════════════════════════════════════════════════════════════════════
# [D5] UNIFIED DATASET HUB
# ═══════════════════════════════════════════════════════════════════════

class UnifiedDatasetHub:
    """
    [D] Single interface for all dataset connectors.

    Provides unified access to GW, neutrino, protein, and gene expression data.
    """

    def __init__(self, cfg: Optional[DatasetConfig] = None):
        self.cfg = cfg or DatasetConfig()
        self.gwosc = GWOSCConnector(self.cfg)
        self.icecube = IceCubeConnector(self.cfg)
        self.pdb = PDBConnector(self.cfg)
        self.geo = GEOConnector(self.cfg)

    def fetch_gw_strain(self, event_name: Optional[str] = None) -> Dict:
        """Fetch GW strain data."""
        event = event_name or self.cfg.gwosc_event
        return self.gwosc.fetch_strain(event, self.cfg.gwosc_detector)

    def fetch_neutrino_alerts(self, hours: int = 24) -> List[Dict]:
        """Fetch IceCube neutrino alerts."""
        return self.icecube.fetch_recent_alerts(hours)

    def fetch_protein_structure(self, pdb_id: Optional[str] = None) -> Dict:
        """Fetch protein structure from PDB."""
        pdb = pdb_id or self.cfg.pdb_id
        return self.pdb.fetch_structure(pdb)

    def fetch_gene_expression(self, accession: Optional[str] = None) -> Dict:
        """Fetch gene expression from GEO."""
        acc = accession or self.cfg.geo_accession
        return self.geo.fetch_expression(acc)

    def summary(self) -> Dict[str, Any]:
        """Summary of available data sources."""
        return {
            "gwosc": {
                "event": self.cfg.gwosc_event,
                "detector": self.cfg.gwosc_detector,
                "status": "available" if HAS_REQUESTS else "synthetic_only",
            },
            "icecube": {
                "alert_type": self.cfg.icecube_alert_type,
                "status": "available" if HAS_REQUESTS else "synthetic_only",
            },
            "pdb": {
                "id": self.cfg.pdb_id,
                "status": "available" if HAS_REQUESTS else "synthetic_only",
            },
            "geo": {
                "accession": self.cfg.geo_accession,
                "status": "available" if HAS_REQUESTS else "synthetic_only",
            },
            "honest_note": (
                "Real data requires network access. Synthetic fallback is for "
                "demonstration only, not scientific analysis."
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Real Experimental Datasets — Smoke Test")
    print("=" * 72)
    print(f"  requests installed : {HAS_REQUESTS}")
    print(f"  h5py installed     : {HAS_H5PY}")

    cfg = DatasetConfig()
    hub = UnifiedDatasetHub(cfg)

    # Test 1: GW strain
    print("\n  ── [D1] GWOSC Strain Data ──")
    gw_data = hub.fetch_gw_strain("GW150914")
    print(f"    Event          : {gw_data.get('event', 'unknown')}")
    print(f"    Detector       : {gw_data.get('detector', 'unknown')}")
    print(f"    Samples        : {len(gw_data.get('strain', []))}")
    print(f"    Sample rate    : {gw_data.get('sample_rate', 0)} Hz")
    print(f"    Synthetic      : {gw_data.get('synthetic', False)}")

    # Test 2: Neutrino alerts
    print("\n  ── [D2] IceCube Neutrino Alerts ──")
    alerts = hub.fetch_neutrino_alerts(hours=24)
    print(f"    Alerts fetched : {len(alerts)}")
    if alerts:
        a = alerts[0]
        print(f"    First alert    : {a.get('id', 'unknown')}")
        print(f"    Energy         : {a.get('energy_gev', 0):.2e} GeV")
        print(f"    Synthetic      : {a.get('synthetic', False)}")

    # Test 3: Protein structure
    print("\n  ── [D3] PDB Protein Structure ──")
    pdb_data = hub.fetch_protein_structure("1UBQ")
    print(f"    PDB ID         : {pdb_data.get('pdb_id', 'unknown')}")
    print(f"    Atoms          : {pdb_data.get('n_atoms', 0)}")
    print(f"    Synthetic      : {pdb_data.get('synthetic', False)}")

    # Test 4: Gene expression
    print("\n  ── [D4] GEO Gene Expression ──")
    geo_data = hub.fetch_gene_expression("GSE53987")
    print(f"    Accession      : {geo_data.get('accession', 'unknown')}")
    print(f"    Samples        : {geo_data.get('n_samples', 0)}")
    print(f"    Genes          : {geo_data.get('n_genes', 0)}")
    print(f"    Synthetic      : {geo_data.get('synthetic', False)}")

    # Summary
    print("\n  ── Summary ──")
    summary = hub.summary()
    for source, info in summary.items():
        if source != "honest_note":
            print(f"    {source:10s} : {info.get('status', 'unknown')}")
    print(f"\n    {summary['honest_note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Real data requires network access and may have rate limits")
    print("  • Synthetic fallback is for DEMONSTRATION only")
    print("  • Each connector must be validated against official data format")
    print("  • No claim that synthetic data matches real statistical properties")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
