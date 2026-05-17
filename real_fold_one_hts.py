# =============================================================================
# REAL FOLD ONE HTS — High‑Throughput Mutation Scanning & Epistasis Engine
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# Unified mutation scanning for proteins, DNA, RNA, and mixed multimers.
# Uses REAL FOLD ONE (SOC‑controlled physics) as backend.
# Features:
#   • Full single‑mutation scan (all positions × all monomers)
#   • Double‑mutation epistasis scanning (all monomer pairs)
#   • Auto‑detect sequence type (protein, DNA, RNA)
#   • Local relaxation window for speed
#   • Multi‑GPU parallel evaluation
#   • CSV/JSON export, mutational landscape heatmap, ΔΔG distributions
#   • Position‑specific profile plots
#   • Epistasis distribution and additivity plots
# =============================================================================

import os, sys, json, time, random, argparse, logging, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Plotting imports – only if generating reports
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from tqdm import tqdm

import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("REAL_FOLD_HTS")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

# -----------------------------------------------------------------------------
# Import REAL FOLD ONE engine (must be in Python path)
# -----------------------------------------------------------------------------
try:
    from real_fold_one import (
        RefinementEngine,
        RefinementConfig,
        detect_sequence_type,
        load_structure,
        save_structure,
        AA_VOCAB,
        AA_TO_ID,
        build_dna_helix,
        HAS_BIOTITE,
    )
    REAL_FOLD_ONE_OK = True
except ImportError as e:
    logger.error("REAL FOLD ONE not found. Please install real_fold_one.py in the same directory.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------
PROTEIN_ALPHABET = [aa for aa in AA_VOCAB if aa != 'X']
DNA_ALPHABET = ['A', 'C', 'G', 'T']
RNA_ALPHABET = ['A', 'C', 'G', 'U']

def get_alphabet(seq: str) -> List[str]:
    seq_type = detect_sequence_type(seq)
    if seq_type == 'protein':
        return PROTEIN_ALPHABET
    elif seq_type == 'dna':
        return DNA_ALPHABET
    elif seq_type == 'rna':
        return RNA_ALPHABET
    else:
        return PROTEIN_ALPHABET

def is_transition(old: str, new: str, seq_type: str) -> bool:
    if seq_type in ('dna', 'rna'):
        transitions = {('A','G'),('G','A'),('C','T'),('T','C'),
                       ('A','U'),('U','A'),('C','U'),('U','C')}
        return (old, new) in transitions
    return False

def is_transversion(old: str, new: str, seq_type: str) -> bool:
    if seq_type in ('dna', 'rna'):
        return not is_transition(old, new, seq_type) and old != new
    return False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class HTSConfig:
    pdb_file: Optional[str] = None
    sequence: Optional[str] = None
    chain_id: Optional[str] = None
    output_dir: str = "./hts_output"
    ddg_threshold: float = 0.5
    relaxation_steps: int = 30
    relaxation_window: int = 3
    use_gpu: bool = True
    num_gpus: int = 1
    scan_full: bool = False
    mutation_list: Optional[List[Tuple[int, str]]] = None
    scan_epistasis: bool = False
    epistasis_pairs: Optional[List[Tuple[int, int, int, int]]] = None
    max_epistasis_pairs: int = 1000
    lr: float = 1e-4
    steps: int = 600
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# HTS Analyzer
# -----------------------------------------------------------------------------
class HTSAnalyzer:
    def __init__(self, cfg: HTSConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if cfg.use_gpu else "cpu")
        self.engine = None
        self.sequences = []
        self.chain_types = []
        self.full_seq = ""
        self.ca_coords = None
        self.wt_energy = None
        self.chain_boundaries = []
        self.per_chain_seq = []
        self.per_chain_types = []

    def load_structure(self):
        """Load initial structure from PDB or from sequence."""
        if self.cfg.pdb_file and os.path.exists(self.cfg.pdb_file):
            if not HAS_BIOTITE:
                raise ImportError("biotite is required to read PDB files.")
            data = load_structure(self.cfg.pdb_file, chain=self.cfg.chain_id)
            self.ca_coords = torch.tensor(data['coords'], dtype=torch.float32, device=self.device)
            self.full_seq = data['sequence']
            chain_ids = data.get('chain_ids', ['A'] * len(self.full_seq))
            boundaries = [0]
            current = chain_ids[0]
            for i, cid in enumerate(chain_ids[1:], start=1):
                if cid != current:
                    boundaries.append(i)
                    current = cid
            boundaries.append(len(self.full_seq))
            self.chain_boundaries = boundaries[1:-1]
            start = 0
            for end in boundaries[1:]:
                seq = self.full_seq[start:end]
                self.sequences.append(seq)
                self.chain_types.append(detect_sequence_type(seq))
                start = end
        elif self.cfg.sequence:
            self.full_seq = self.cfg.sequence.upper()
            seq_type = detect_sequence_type(self.full_seq)
            if seq_type == 'protein':
                raise ValueError("For protein, please provide a PDB file (initial structure).")
            self.sequences = [self.full_seq]
            self.chain_types = [seq_type]
            self.chain_boundaries = []
            if seq_type == 'rna':
                self.ca_coords = build_dna_helix(self.full_seq, rise=2.8, twist=32.7, radius=9.0)
            else:
                self.ca_coords = build_dna_helix(self.full_seq, rise=3.38, twist=36.0, radius=8.0)
            self.ca_coords = self.ca_coords.to(self.device)
        else:
            raise ValueError("Must provide either --pdb or --seq")

        # Create refinement engine (per process, we will recreate in workers later)
        eng_cfg = RefinementConfig(device=str(self.device), lr=self.cfg.lr, steps=self.cfg.steps)
        self.engine = RefinementEngine(eng_cfg)
        # Compute WT energy once on main device
        self.wt_energy = self.engine.compute_energy(
            self.ca_coords, self.full_seq,
            chain_types=[ct for seq, ct in zip(self.sequences, self.chain_types) for _ in seq],
            mask=None, chi=None, alpha=None
        )
        logger.info(f"WT energy: {self.wt_energy:.4f} kcal/mol")
        logger.info(f"Loaded {len(self.full_seq)} residues, chains: {self.chain_types}")

    def _global_pos(self, chain_idx: int, pos_in_chain: int) -> int:
        offset = sum(len(s) for s in self.sequences[:chain_idx])
        return offset + pos_in_chain

    def _compute_ddg_single_worker(self, task):
        """Worker function for parallel evaluation of a single mutation."""
        chain_idx, pos_in_chain, new_monomer, gpu_id = task
        device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
        # Create engine for this device
        eng_cfg = RefinementConfig(device=str(device), lr=self.cfg.lr, steps=self.cfg.steps)
        engine = RefinementEngine(eng_cfg)
        coords = self.ca_coords.to(device)
        full_seq = self.full_seq
        glob_pos = self._global_pos(chain_idx, pos_in_chain)
        wt = full_seq[glob_pos]
        if wt == new_monomer:
            return {'chain': chain_idx, 'pos_in_chain': pos_in_chain, 'global_pos': glob_pos,
                    'wt': wt, 'mut': new_monomer, 'ddg': 0.0, 'type': 'self', 'e_wt': self.wt_energy, 'e_mut': self.wt_energy}
        mut_seq = full_seq[:glob_pos] + new_monomer + full_seq[glob_pos+1:]
        chain_types_flat = [ct for seq, ct in zip(self.sequences, self.chain_types) for _ in seq]
        if self.cfg.relaxation_steps > 0:
            coords_mut, e_mut = engine.relax_local(
                coords, mut_seq,
                positions=[glob_pos],
                steps=self.cfg.relaxation_steps,
                window=self.cfg.relaxation_window,
                chain_types=chain_types_flat,
                mask=None, chi=None, alpha=None
            )
        else:
            e_mut = engine.compute_energy(coords, mut_seq, chain_types=chain_types_flat)
        ddg = e_mut - self.wt_energy
        seq_type = self.chain_types[chain_idx]
        mut_type = 'transition' if is_transition(wt, new_monomer, seq_type) else \
                   'transversion' if is_transversion(wt, new_monomer, seq_type) else 'mutation'
        return {
            'chain': chain_idx,
            'pos_in_chain': pos_in_chain,
            'global_pos': glob_pos,
            'wt': wt,
            'mut': new_monomer,
            'ddg': ddg,
            'type': mut_type,
            'e_wt': self.wt_energy,
            'e_mut': e_mut,
        }

    def compute_ddg_single(self, chain_idx: int, pos_in_chain: int, new_monomer: str) -> Dict:
        """Return ΔΔG for a single mutation (single‑threaded convenience)."""
        return self._compute_ddg_single_worker((chain_idx, pos_in_chain, new_monomer, -1))

    def scan_all_single(self) -> List[Dict]:
        """Scan all possible single mutations using multi‑GPU parallelism."""
        tasks = []
        for chain_idx, seq in enumerate(self.sequences):
            alphabet = get_alphabet(seq)
            for pos in range(len(seq)):
                wt = seq[pos]
                for new in alphabet:
                    if new == wt:
                        continue
                    tasks.append((chain_idx, pos, new, 0))  # gpu_id will be assigned later
        if not tasks:
            return []
        total = len(tasks)
        logger.info(f"Evaluating {total} single mutations...")
        results = []
        if self.cfg.use_gpu and torch.cuda.is_available():
            num_gpus = self.cfg.num_gpus if self.cfg.num_gpus > 0 else torch.cuda.device_count()
            num_gpus = min(num_gpus, total)
            # Assign tasks to GPUs
            gpu_tasks = [[] for _ in range(num_gpus)]
            for i, task in enumerate(tasks):
                gpu_id = i % num_gpus
                gpu_tasks[gpu_id].append(task[:3] + (gpu_id,))
            # Use ProcessPoolExecutor with multiprocessing to spawn workers
            # Since the engine is not picklable, we use a different approach:
            # Spawn processes using torch.multiprocessing.spawn
            # For simplicity, we'll use a sequential fallback if there are issues.
            # Instead, we'll implement a simple pool using multiprocessing.Pool
            # with a global function.
            # We'll rewrite to use ProcessPoolExecutor with a worker initializer that sets the GPU.
            # But we need to pass the entire HTSAnalyzer's data to the worker.
            # We'll implement a standalone worker function that reconstructs necessary parts.
            # For now, we'll use a simple approach: loop over GPUs sequentially.
            # Multi‑GPU parallelism is complex; we'll implement a basic version:
            # spawn processes per GPU, each processing its batch.
            # We'll use torch.multiprocessing.spawn for each GPU.
            results = self._run_parallel(gpu_tasks)
        else:
            # Single process (CPU or single GPU)
            for task in tqdm(tasks, desc="Single mutations"):
                res = self._compute_ddg_single_worker(task[:3] + (-1,))
                results.append(res)
        return results

    def _run_parallel(self, gpu_tasks: List[List]) -> List[Dict]:
        """Run tasks in parallel across multiple GPUs using torch.multiprocessing."""
        # We need to spawn a process for each GPU.
        # Each process will set its own device and process its list.
        manager = mp.Manager()
        result_list = manager.list()
        processes = []
        for gpu_id, tasks in enumerate(gpu_tasks):
            if not tasks:
                continue
            p = mp.Process(target=self._gpu_worker, args=(gpu_id, tasks, result_list))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # Convert manager.list to ordinary list
        return list(result_list)

    def _gpu_worker(self, gpu_id: int, tasks: List, result_list):
        """Worker function for a single GPU."""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device(f"cuda:0")  # after setting visible devices, only one GPU
        # Re‑initialize the engine and data on this device
        eng_cfg = RefinementConfig(device=str(device), lr=self.cfg.lr, steps=self.cfg.steps)
        engine = RefinementEngine(eng_cfg)
        # Copy coords to this device
        coords = self.ca_coords.to(device)
        for task in tasks:
            chain_idx, pos_in_chain, new_monomer, _ = task
            res = self._compute_ddg_single_worker((chain_idx, pos_in_chain, new_monomer, gpu_id))
            result_list.append(res)

    def compute_ddg_double(self, chain1, pos1, new1, chain2, pos2, new2) -> Dict:
        """Compute ΔΔG for double mutation (both positions)."""
        gp1 = self._global_pos(chain1, pos1)
        gp2 = self._global_pos(chain2, pos2)
        if gp1 > gp2:
            gp1, gp2 = gp2, gp1
            new1, new2 = new2, new1
        mut_seq = list(self.full_seq)
        mut_seq[gp1] = new1
        mut_seq[gp2] = new2
        mut_seq = "".join(mut_seq)
        coords_mut = self.ca_coords.clone()
        chain_types_flat = [ct for seq, ct in zip(self.sequences, self.chain_types) for _ in seq]
        if self.cfg.relaxation_steps > 0:
            coords_mut, e_dbl = self.engine.relax_local(
                coords_mut, mut_seq,
                positions=[gp1, gp2],
                steps=self.cfg.relaxation_steps,
                window=self.cfg.relaxation_window,
                chain_types=chain_types_flat
            )
        else:
            e_dbl = self.engine.compute_energy(coords_mut, mut_seq, chain_types=chain_types_flat)
        ddg = e_dbl - self.wt_energy
        return {
            'chain1': chain1, 'pos1': pos1, 'mut1': new1,
            'chain2': chain2, 'pos2': pos2, 'mut2': new2,
            'ddg_double': ddg,
            'e_mut': e_dbl,
        }

    def scan_epistasis(self, pairs: Optional[List[Tuple[int,int,int,int]]] = None) -> List[Dict]:
        """Scan epistasis for given pairs (or auto‑generate)."""
        if pairs is None:
            pairs = []
            total_pairs = 0
            for c1, seq1 in enumerate(self.sequences):
                for c2, seq2 in enumerate(self.sequences):
                    if c2 < c1:
                        continue
                    for p1 in range(len(seq1)):
                        for p2 in range(len(seq2)):
                            if c1 == c2 and abs(p1 - p2) < 2:
                                continue
                            total_pairs += 1
            max_pairs = min(total_pairs, self.cfg.max_epistasis_pairs)
            # Randomly sample pairs
            sampled = set()
            while len(sampled) < max_pairs:
                c1 = random.randint(0, len(self.sequences)-1)
                c2 = random.randint(c1, len(self.sequences)-1)
                seq1 = self.sequences[c1]
                seq2 = self.sequences[c2]
                p1 = random.randint(0, len(seq1)-1)
                p2 = random.randint(0, len(seq2)-1)
                if c1 == c2 and abs(p1 - p2) < 2:
                    continue
                sampled.add((c1, p1, c2, p2))
            pairs = list(sampled)

        results = []
        for c1, p1, c2, p2 in tqdm(pairs, desc="Epistasis pairs"):
            seq1 = self.sequences[c1]
            seq2 = self.sequences[c2]
            wt1 = seq1[p1]
            wt2 = seq2[p2]
            alph1 = get_alphabet(seq1)
            alph2 = get_alphabet(seq2)
            # Choose first non‑wt for each as representative mutation
            mut1 = next((m for m in alph1 if m != wt1), wt1)
            mut2 = next((m for m in alph2 if m != wt2), wt2)
            if mut1 == wt1 or mut2 == wt2:
                continue
            # Compute singles (may be cached or recomputed; for simplicity recompute)
            d1 = self.compute_ddg_single(c1, p1, mut1)
            d2 = self.compute_ddg_single(c2, p2, mut2)
            dbl = self.compute_ddg_double(c1, p1, mut1, c2, p2, mut2)
            additive = d1['ddg'] + d2['ddg']
            epistasis = dbl['ddg_double'] - additive
            results.append({
                'chain1': c1, 'pos1': p1, 'mut1': mut1, 'wt1': wt1,
                'chain2': c2, 'pos2': p2, 'mut2': mut2, 'wt2': wt2,
                'ddg1': d1['ddg'], 'ddg2': d2['ddg'],
                'ddg_double': dbl['ddg_double'], 'ddg_additive': additive,
                'epistasis': epistasis,
                'significant': abs(epistasis) > self.cfg.ddg_threshold,
            })
        return results

    def run(self):
        """Main entry point."""
        self.load_structure()
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        if self.cfg.scan_full:
            logger.info("Starting full single‑mutation scan...")
            scan_data = self.scan_all_single()
            self._generate_scan_reports(scan_data)

        if self.cfg.scan_epistasis:
            logger.info("Starting epistasis scan...")
            epi_data = self.scan_epistasis(self.cfg.epistasis_pairs)
            self._generate_epistasis_reports(epi_data)

        # Save summary JSON
        summary = {
            'wt_energy': self.wt_energy,
            'n_mutations': len(scan_data) if self.cfg.scan_full else 0,
            'n_epistasis': len(epi_data) if self.cfg.scan_epistasis else 0,
        }
        with open(Path(self.cfg.output_dir) / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved in {self.cfg.output_dir}")

    # -------------------------------------------------------------------------
    # Plotting utilities
    # -------------------------------------------------------------------------
    def _generate_scan_reports(self, data: List[Dict]):
        if not data:
            return
        out = Path(self.cfg.output_dir)
        df = pd.DataFrame(data)
        df.to_csv(out / "single_mutations.csv", index=False)

        if not HAS_PLOTTING:
            logger.warning("Matplotlib/seaborn not installed; skipping plots.")
            return

        # ΔΔG distribution
        plt.figure(figsize=(8,4))
        sns.histplot(df['ddg'], bins=50, kde=True)
        plt.axvline(0, color='red', linestyle='--')
        plt.axvline(-self.cfg.ddg_threshold, color='orange', linestyle=':')
        plt.axvline(self.cfg.ddg_threshold, color='orange', linestyle=':')
        plt.title("ΔΔG Distribution")
        plt.xlabel("ΔΔG (kcal/mol)")
        plt.tight_layout()
        plt.savefig(out / "ddg_distribution.png", dpi=200)
        plt.close()

        # Mutational landscape heatmap
        seq = self.full_seq
        all_muts = sorted(set(df['mut']))
        pos_list = sorted(set(df['global_pos']))
        if len(all_muts) > 0 and len(pos_list) > 0:
            mat = np.zeros((len(pos_list), len(all_muts)))
            for _, row in df.iterrows():
                i = pos_list.index(row['global_pos'])
                j = all_muts.index(row['mut'])
                mat[i, j] = row['ddg']
            # Mask wild‑type
            mask = np.zeros_like(mat, dtype=bool)
            for i, pos in enumerate(pos_list):
                wt = seq[pos]
                if wt in all_muts:
                    j = all_muts.index(wt)
                    mask[i, j] = True
            mat_masked = np.ma.array(mat, mask=mask)
            plt.figure(figsize=(max(6, len(all_muts)*0.4), max(6, len(pos_list)*0.2)))
            sns.heatmap(mat_masked, cmap='coolwarm', center=0,
                        xticklabels=all_muts, yticklabels=[f"{pos+1}{seq[pos]}" for pos in pos_list],
                        cbar_kws={'label': 'ΔΔG (kcal/mol)'})
            plt.title("Mutational landscape")
            plt.xlabel("Mutant")
            plt.ylabel("Position (WT)")
            plt.tight_layout()
            plt.savefig(out / "mutational_landscape.png", dpi=300)
            plt.close()

        # Position profile
        pos_mean = df.groupby('global_pos')['ddg'].agg(['mean', 'std', 'count'])
        if len(pos_mean) > 0:
            plt.figure(figsize=(max(10, len(pos_mean)*0.2), 5))
            x = pos_mean.index
            plt.fill_between(x, pos_mean['mean'] - pos_mean['std'], pos_mean['mean'] + pos_mean['std'], alpha=0.3)
            plt.plot(x, pos_mean['mean'], 'o-', markersize=3)
            plt.axhline(0, color='black', linestyle='--')
            plt.xlabel("Residue index")
            plt.ylabel("ΔΔG (kcal/mol)")
            plt.title("Position‑wise mutation tolerance")
            plt.tight_layout()
            plt.savefig(out / "position_profile.png", dpi=200)
            plt.close()

    def _generate_epistasis_reports(self, data: List[Dict]):
        if not data:
            return
        out = Path(self.cfg.output_dir)
        df = pd.DataFrame(data)
        df.to_csv(out / "epistasis.csv", index=False)

        if not HAS_PLOTTING:
            return

        # Epistasis histogram
        plt.figure(figsize=(8,4))
        sns.histplot(df['epistasis'], bins=50, kde=True)
        plt.axvline(0, color='black', linestyle='--')
        plt.axvline(-self.cfg.ddg_threshold, color='red', linestyle=':')
        plt.axvline(self.cfg.ddg_threshold, color='red', linestyle=':')
        plt.title("Epistasis (ε) Distribution")
        plt.xlabel("ε (kcal/mol)")
        plt.tight_layout()
        plt.savefig(out / "epistasis_distribution.png", dpi=200)
        plt.close()

        # Scatter plot additive vs double
        plt.figure(figsize=(6,6))
        plt.scatter(df['ddg_additive'], df['ddg_double'], alpha=0.5, s=10)
        lim = max(df['ddg_additive'].abs().max(), df['ddg_double'].abs().max()) + 0.5
        plt.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
        plt.xlabel("ΔΔG additive (kcal/mol)")
        plt.ylabel("ΔΔG double mutant (kcal/mol)")
        plt.title("Additivity vs Double mutant ΔΔG")
        plt.tight_layout()
        plt.savefig(out / "additivity_scatter.png", dpi=200)
        plt.close()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="REAL FOLD ONE HTS – Mutation Scanning & Epistasis")
    parser.add_argument('--pdb', type=str, help='Input PDB/mmCIF file')
    parser.add_argument('--seq', type=str, help='Sequence (auto‑detect type; DNA/RNA only)')
    parser.add_argument('--chain', type=str, default=None, help='Chain ID')
    parser.add_argument('--output', '-o', type=str, default='./hts_output', help='Output directory')
    parser.add_argument('--scan', action='store_true', help='Full single‑mutation scan')
    parser.add_argument('--epistasis', action='store_true', help='Epistasis scan')
    parser.add_argument('--relax_steps', type=int, default=30, help='Local relaxation steps per mutant')
    parser.add_argument('--window', type=int, default=3, help='Relaxation window size (±)')
    parser.add_argument('--ddg_threshold', type=float, default=0.5, help='Significance threshold (kcal/mol)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use in parallel')
    parser.add_argument('--max_epi_pairs', type=int, default=500, help='Max epistasis pairs to evaluate')
    args = parser.parse_args()

    cfg = HTSConfig(
        pdb_file=args.pdb,
        sequence=args.seq,
        chain_id=args.chain,
        output_dir=args.output,
        ddg_threshold=args.ddg_threshold,
        relaxation_steps=args.relax_steps,
        relaxation_window=args.window,
        use_gpu=args.gpu,
        num_gpus=args.num_gpus,
        scan_full=args.scan,
        scan_epistasis=args.epistasis,
        max_epistasis_pairs=args.max_epi_pairs,
    )

    analyzer = HTSAnalyzer(cfg)
    analyzer.run()

if __name__ == "__main__":
    main()
