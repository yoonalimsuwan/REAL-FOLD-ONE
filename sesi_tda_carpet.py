"""
=============================================================================
SESI TOPOLOGICAL DATA ANALYSIS (TDA) FOR CARPET
Persistent homology and topological features of Sierpinski carpet
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE EXPLORES
--------------------------
[C] Persistent homology — topological features across scales
[D] Vietoris-Rips complexes on carpet graph
[O] Connection between topological invariants and spectral structure

TOPOLOGICAL FEATURES OF CARPET
-------------------------------
The Sierpinski carpet has:
  • Hausdorff dimension d_f = log(8)/log(3) ≈ 1.893
  • Non-trivial topology at all scales (holes at every level)
  • Connected but not simply connected

Persistent homology can quantify:
  • Betti numbers β_0 (connected components), β_1 (loops), β_2 (voids)
  • Persistence diagrams showing topological features across scales
  • Topological entropy and complexity measures

CONNECTION TO SPECTRAL STRUCTURE (SPECULATIVE)
-----------------------------------------------
Carpet-19 establishes spectral dimension d_s ≈ 1.80 and spectral Sobolev
scale W^s. The question is whether topological invariants (Betti numbers,
persistence) correlate with spectral properties (eigenvalues, heat kernel).

This is [O] speculative — no rigorous connection has been established.

LABELS: [C] cited TDA method  [D] derived implementation  [O] open

HONEST SCOPE
------------
• TDA is a powerful tool for shape analysis, but its connection to the
  spectral theory of Carpet-19 is [O] not established.
• Persistent homology computations can be expensive for large complexes.
• The carpet's fractal structure requires careful treatment of scale.
• Results are exploratory; no claims of new theorems.
=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sesi.tda_carpet")

try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    import persim
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

# Import carpet graph from previous modules
try:
    from sesi_carpet_operator import CarpetGraph
    HAS_CARPET = True
except ImportError:
    HAS_CARPET = False

__all__ = [
    "TDAConfig",
    "CarpetPointCloud",
    "VietorisRipsComplex",
    "PersistentHomology",
    "TopologicalFeatures",
    "TDACarpetAnalyzer",
]


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TDAConfig:
    """Configuration for TDA analysis."""
    carpet_level: int = 3
    max_dimension: int = 1  # compute H_0, H_1
    max_radius: float = 1.0
    n_radii: int = 50

    # Persistence
    min_persistence: float = 0.01  # ignore features with persistence < this

    device: str = "cpu"


# ═══════════════════════════════════════════════════════════════════════
# [D] CARPET POINT CLOUD
# ═══════════════════════════════════════════════════════════════════════

class CarpetPointCloud:
    """
    [D] Generate point cloud from Sierpinski carpet graph.

    Each cell center becomes a point in ℝ².
    """

    def __init__(self, cfg: TDAConfig):
        self.cfg = cfg
        if not HAS_CARPET:
            raise RuntimeError("CarpetGraph not available")

    def generate(self) -> np.ndarray:
        """Generate point cloud from carpet cells."""
        graph = CarpetGraph(self.cfg.carpet_level)
        points = np.array(graph.cells, dtype=float)

        # Normalize to [0, 1]²
        side = 3 ** self.cfg.carpet_level
        points = points / side

        return points


# ═══════════════════════════════════════════════════════════════════════
# [D] VIETORIS-RIPS COMPLEX
# ═══════════════════════════════════════════════════════════════════════

class VietorisRipsComplex:
    """
    [C/D] Construct Vietoris-Rips complex from point cloud.

    For a set of points X and radius r, the Vietoris-Rips complex VR(X, r)
    has:
      • Vertices: points in X
      • Edges: pairs of points within distance r
      • k-simplices: (k+1)-tuples of points pairwise within distance r
    """

    def __init__(self, points: np.ndarray):
        self.points = points
        self.n_points = len(points)

    def distance_matrix(self) -> np.ndarray:
        """Compute pairwise distance matrix."""
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(self.points))

    def adjacency_at_radius(self, radius: float) -> np.ndarray:
        """Binary adjacency matrix at given radius."""
        D = self.distance_matrix()
        return (D <= radius).astype(int)

    def n_edges_at_radius(self, radius: float) -> int:
        """Count edges at given radius."""
        A = self.adjacency_at_radius(radius)
        return int(np.sum(A) // 2)  # divide by 2 for symmetry


# ═══════════════════════════════════════════════════════════════════════
# [C] PERSISTENT HOMOLOGY
# ═══════════════════════════════════════════════════════════════════════

class PersistentHomology:
    """
    [C] Compute persistent homology using ripser.

    Persistent homology tracks topological features (connected components,
    loops, voids) across scales. Features are born at some radius and die
    at another; the persistence (death - birth) measures significance.
    """

    def __init__(self, cfg: TDAConfig):
        self.cfg = cfg

    def compute(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Compute persistent homology.

        Returns dict with persistence diagrams for each dimension.
        """
        if not HAS_RIPSER:
            logger.warning("ripser not installed; using simplified persistence")
            return self._simplified_persistence(points)

        try:
            # Compute persistence diagrams
            result = ripser.ripser(
                points,
                maxdim=self.cfg.max_dimension,
                thresh=self.cfg.max_radius,
            )

            diagrams = result["dgms"]
            return {
                "persistence_diagrams": diagrams,
                "n_features": [len(d) for d in diagrams],
                "status": "computed",
            }
        except Exception as e:
            logger.warning(f"ripser failed: {e}")
            return self._simplified_persistence(points)

    def _simplified_persistence(self, points: np.ndarray) -> Dict[str, Any]:
        """Simplified persistence computation (H_0 only)."""
        vr = VietorisRipsComplex(points)

        # Track connected components across radii
        radii = np.linspace(0, self.cfg.max_radius, self.cfg.n_radii)
        n_components = []

        for r in radii:
            A = vr.adjacency_at_radius(r)
            # Count connected components via BFS
            n_comp = self._count_components(A)
            n_components.append(n_comp)

        # Simplified "persistence diagram" for H_0
        # (In reality, would track birth/death of each component)
        diagram_h0 = np.array([[0.0, r] for r in radii if n_components[radii == r][0] > 1])

        return {
            "persistence_diagrams": [diagram_h0],
            "n_features": [len(diagram_h0)],
            "status": "simplified",
            "note": "Only H_0 computed; install ripser for full persistent homology",
        }

    def _count_components(self, adjacency: np.ndarray) -> int:
        """Count connected components via BFS."""
        n = len(adjacency)
        visited = np.zeros(n, dtype=bool)
        n_components = 0

        for i in range(n):
            if not visited[i]:
                n_components += 1
                # BFS
                queue = [i]
                while queue:
                    node = queue.pop(0)
                    if visited[node]:
                        continue
                    visited[node] = True
                    neighbors = np.where(adjacency[node] > 0)[0]
                    queue.extend(neighbors.tolist())

        return n_components


# ═══════════════════════════════════════════════════════════════════════
# [D] TOPOLOGICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════

class TopologicalFeatures:
    """
    [D] Extract topological features from persistence diagrams.

    Features include:
      • Betti numbers (number of features at each dimension)
      • Persistence entropy
      • Topological complexity measures
    """

    def __init__(self, cfg: TDAConfig):
        self.cfg = cfg

    def betti_numbers(self, diagrams: List[np.ndarray]) -> Dict[int, int]:
        """Compute Betti numbers from persistence diagrams."""
        betti = {}
        for dim, diagram in enumerate(diagrams):
            # Filter by minimum persistence
            if len(diagram) > 0:
                persistence = diagram[:, 1] - diagram[:, 0]
                significant = persistence > self.cfg.min_persistence
                betti[dim] = int(np.sum(significant))
            else:
                betti[dim] = 0
        return betti

    def persistence_entropy(self, diagram: np.ndarray) -> float:
        """
        Compute persistence entropy of a diagram.

        H = -Σ p_i log(p_i), where p_i = persistence_i / Σ persistence
        """
        if len(diagram) == 0:
            return 0.0

        persistence = diagram[:, 1] - diagram[:, 0]
        persistence = persistence[persistence > 0]

        if len(persistence) == 0:
            return 0.0

        p = persistence / persistence.sum()
        entropy = -np.sum(p * np.log(p + 1e-10))
        return float(entropy)

    def topological_summary(self, diagrams: List[np.ndarray]) -> Dict[str, Any]:
        """Compute topological summary statistics."""
        betti = self.betti_numbers(diagrams)
        entropies = [self.persistence_entropy(d) for d in diagrams]

        return {
            "betti_numbers": betti,
            "persistence_entropies": entropies,
            "total_features": sum(betti.values()),
            "max_entropy": max(entropies) if entropies else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════
# [D] TDA CARPET ANALYZER
# ═══════════════════════════════════════════════════════════════════════

class TDACarpetAnalyzer:
    """
    [D] Full TDA analysis pipeline for Sierpinski carpet.

    Combines:
      • Point cloud generation
      • Vietoris-Rips complex construction
      • Persistent homology computation
      • Topological feature extraction
    """

    def __init__(self, cfg: Optional[TDAConfig] = None):
        self.cfg = cfg or TDAConfig()
        self.point_cloud = CarpetPointCloud(self.cfg)
        self.persistence = PersistentHomology(self.cfg)
        self.features = TopologicalFeatures(self.cfg)

    def analyze(self) -> Dict[str, Any]:
        """Run full TDA analysis."""
        # Generate point cloud
        points = self.point_cloud.generate()
        logger.info(f"Generated {len(points)} points from carpet level {self.cfg.carpet_level}")

        # Compute persistent homology
        ph_result = self.persistence.compute(points)

        if ph_result["status"] == "computed":
            diagrams = ph_result["persistence_diagrams"]
        else:
            diagrams = ph_result.get("persistence_diagrams", [])

        # Extract topological features
        summary = self.features.topological_summary(diagrams)

        # Compare with spectral dimension (speculative)
        d_f = np.log(8) / np.log(3)  # Hausdorff dimension
        d_s = 2 * d_f / (np.log(8 * 1.25) / np.log(3))  # spectral dimension (ρ≈1.25)

        return {
            "n_points": len(points),
            "carpet_level": self.cfg.carpet_level,
            "persistence_status": ph_result["status"],
            "topological_summary": summary,
            "spectral_comparison": {
                "hausdorff_dimension": d_f,
                "spectral_dimension": d_s,
                "betti_0": summary["betti_numbers"].get(0, 0),
                "betti_1": summary["betti_numbers"].get(1, 0),
                "note": (
                    "Connection between topological and spectral dimensions is [O] "
                    "speculative; no rigorous relationship established."
                ),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI TDA for Carpet — Smoke Test")
    print("=" * 72)
    print(f"  ripser installed   : {HAS_RIPSER}")
    print(f"  persim installed   : {HAS_PERSIM}")
    print(f"  carpet module      : {HAS_CARPET}")

    if not HAS_CARPET:
        print("  [SKIP] CarpetGraph not available")
        return

    cfg = TDAConfig(carpet_level=2, max_dimension=1, max_radius=0.5)
    analyzer = TDACarpetAnalyzer(cfg)

    result = analyzer.analyze()

    print(f"\n  ── Point Cloud ──")
    print(f"    Carpet level   : {result['carpet_level']}")
    print(f"    N points       : {result['n_points']}")

    print(f"\n  ── Persistent Homology ──")
    print(f"    Status         : {result['persistence_status']}")
    summary = result["topological_summary"]
    print(f"    Betti numbers  : {summary['betti_numbers']}")
    print(f"    Total features : {summary['total_features']}")
    print(f"    Max entropy    : {summary['max_entropy']:.4f}")

    print(f"\n  ── Spectral Comparison (Speculative) ──")
    spec = result["spectral_comparison"]
    print(f"    Hausdorff dim  : {spec['hausdorff_dimension']:.3f}")
    print(f"    Spectral dim   : {spec['spectral_dimension']:.3f}")
    print(f"    β_0            : {spec['betti_0']}")
    print(f"    β_1            : {spec['betti_1']}")
    print(f"    {spec['note']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • TDA is exploratory; connection to spectral theory is [O]")
    print("  • Persistent homology requires ripser for full computation")
    print("  • Carpet's fractal structure requires careful scale treatment")
    print("  • Results are topological features, not new theorems")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
