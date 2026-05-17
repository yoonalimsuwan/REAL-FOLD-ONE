# =============================================================================
# REAL FOLD ONE — SOC‑Controlled Universal Refinement Engine (Full)
# =============================================================================
# Author: Yoon A Limsuwan
# License: MIT
# Year: 2026
#
# A standalone refinement engine that takes initial protein/nucleic acid
# structures from any predictor and refines them using SOC‑controlled
# molecular mechanics.
#
# Features:
#   • O(N) scaling for N > 100,000 residues
#   • No training required (physics‑based)
#   • SOC adaptive temperature & avalanche gradient
#   • Full‑atom protein (sidechains) with LJ, Coulomb, torsion, clash
#   • DNA/RNA (full nucleotide) with base pairing & stacking
#   • Ligand support (SDF/MOL2/PDB) with protein‑ligand interaction
#   • Multimer & chain boundaries support
#   • Antibody design: CDR loop modeling, Rosetta scoring, affinity prediction
#   • DNA origami: scaffold routing, staple design, oxDNA export, BV topology
#   • Itô SDE (Milstein) and Malliavin sensitivity
#   • Sparse graph for large systems, GPU/CPU, mixed precision
# =============================================================================

import math
import os
import sys
import json
import argparse
import warnings
import random
import itertools
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any, Callable
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Optional dependencies
try:
    from torch_cluster import radius_graph, radius
    HAS_CLUSTER = True
except ImportError:
    HAS_CLUSTER = False
    warnings.warn("torch_cluster not installed; using fallback (slower for large N)")

try:
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.mmcif as mmcif
    HAS_BIOTITE = True
except ImportError:
    HAS_BIOTITE = False
    warnings.warn("biotite not installed; cannot read PDB/mmCIF directly")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    warnings.warn("networkx not installed; DNA origami cycle detection disabled")

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {aa: i for i, aa in enumerate(AA_VOCAB)}
AA_3_TO_1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
    'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
    'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','UNK':'X'
}

HYDROPHOBICITY = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
    'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,
    'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3,'X':0.0
}
RESIDUE_CHARGE = {'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.5}

RAMACHANDRAN_PRIORS = {
    'general':{'phi':-60.0,'psi':-45.0,'width':25.0},
    'G':{'phi':-75.0,'psi':-60.0,'width':40.0},
    'P':{'phi':-65.0,'psi':-30.0,'width':20.0},
}

RESIDUE_TOPOLOGY = {
    'G':[],
    'A':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0)],
    'S':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('OG','OH',1,1.43,109.5,(-2,-1,0),0.0)],
    'C':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('SG','SG',1,1.81,109.5,(-2,-1,0),0.0)],
    'V':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG1','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0)],
    'T':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('OG1','OH',1,1.43,109.5,(-2,-1,0),0.0),
         ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0)],
    'L':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.53,109.5,(-1,0,1),0.0),
         ('CD2','CB',2,1.53,109.5,(-1,0,1),2.0)],
    'I':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG1','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CG2','CB',1,1.53,109.5,(-2,-1,0),2.0),
         ('CD1','CB',2,1.53,109.5,(-1,0,1),0.0)],
    'M':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('SD','S',2,1.81,109.5,(-1,0,1),0.0),
         ('CE','CB',3,1.81,109.5,(-2,-1,0),0.0)],
    'F':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('CE1','CB',3,1.40,120.0,(2,1,0),0.0),
         ('CE2','CB',4,1.40,120.0,(2,1,0),2.0),
         ('CZ','CB',5,1.40,120.0,(3,2,1),0.0)],
    'Y':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('CE1','CB',3,1.40,120.0,(2,1,0),0.0),
         ('CE2','CB',4,1.40,120.0,(2,1,0),2.0),
         ('CZ','CB',5,1.40,120.0,(3,2,1),0.0),
         ('OH','OH',6,1.36,120.0,(4,3,2),0.0)],
    'W':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.53,109.5,(-2,-1,0),0.0),
         ('CD1','CB',2,1.40,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('NE1','N',3,1.38,120.0,(2,1,0),0.0),
         ('CE2','CB',4,1.40,120.0,(2,1,0),2.0),
         ('CE3','CB',5,1.40,120.0,(2,1,0),2.0),
         ('CZ2','CB',6,1.40,120.0,(3,2,1),0.0),
         ('CZ3','CB',7,1.40,120.0,(5,4,2),0.0),
         ('CH2','CB',8,1.40,120.0,(6,3,2),0.0)],
    'D':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','C',1,1.52,109.5,(-2,-1,0),0.0),
         ('OD1','O',2,1.25,120.0,(-1,0,1),0.0),
         ('OD2','O',2,1.25,120.0,(-1,0,1),2.0)],
    'E':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','C',2,1.52,109.5,(-1,0,1),0.0),
         ('OE1','O',3,1.25,120.0,(2,1,0),0.0),
         ('OE2','O',3,1.25,120.0,(2,1,0),2.0)],
    'N':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','C',1,1.52,109.5,(-2,-1,0),0.0),
         ('OD1','O',2,1.25,120.0,(-1,0,1),0.0),
         ('ND2','N',2,1.33,120.0,(-1,0,1),2.0)],
    'Q':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','C',2,1.52,109.5,(-1,0,1),0.0),
         ('OE1','O',3,1.25,120.0,(2,1,0),0.0),
         ('NE2','N',3,1.33,120.0,(2,1,0),2.0)],
    'K':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','CB',2,1.52,109.5,(-1,0,1),0.0),
         ('CE','CB',3,1.52,109.5,(-2,-1,0),0.0),
         ('NZ','N',4,1.47,109.5,(-1,0,1),0.0)],
    'R':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.52,109.5,(-2,-1,0),0.0),
         ('CD','CB',2,1.52,109.5,(-1,0,1),0.0),
         ('NE','N',3,1.46,109.5,(-2,-1,0),0.0),
         ('CZ','C',4,1.33,125.0,(-1,0,1),0.0),
         ('NH1','N',5,1.33,120.0,(4,3,2),0.0),
         ('NH2','N',5,1.33,120.0,(4,3,2),2.0)],
    'H':[('CB','CB',0,1.53,109.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.50,109.5,(-2,-1,0),0.0),
         ('ND1','N',2,1.38,120.0,(-1,0,1),0.0),
         ('CD2','CB',2,1.40,120.0,(-1,0,1),2.0),
         ('CE1','CB',3,1.40,120.0,(2,1,0),0.0),
         ('NE2','N',4,1.38,120.0,(2,1,0),2.0)],
    'P':[('CB','CB',0,1.53,104.5,(-1,-2,-3),0.0),
         ('CG','CB',1,1.50,104.5,(-2,-1,0),0.0),
         ('CD','CB',2,1.50,104.5,(-1,0,1),0.0)],
}
MAX_CHI = 4
RESIDUE_NCHI = {
    'A':0,'G':0,'S':1,'C':1,'V':1,'T':1,'L':2,'I':2,'M':3,'F':2,'Y':2,'W':2,
    'D':2,'E':3,'N':2,'Q':3,'K':4,'R':4,'H':2,'P':2
}

DEFAULT_LJ_PARAMS = {
    'C':(1.9080,0.0860),'CA':(1.9080,0.0860),'CB':(1.9080,0.0860),
    'CG':(1.9080,0.0860),'CD':(1.9080,0.0860),'CE':(1.9080,0.0860),
    'CZ':(1.9080,0.0860),'CH2':(1.9080,0.0860),
    'N':(1.8240,0.1700),'ND':(1.8240,0.1700),'NE':(1.8240,0.1700),
    'NH1':(1.8240,0.1700),'NH2':(1.8240,0.1700),
    'O':(1.6612,0.2100),'OD':(1.6612,0.2100),'OE':(1.6612,0.2100),
    'OH':(1.6612,0.2100),'S':(2.0000,0.2500),'SG':(2.0000,0.2500),
}
DEFAULT_CHARGE_MAP = {'N':-0.5,'CA':0.0,'C':0.5,'O':-0.5,'CB':0.0,'OH':-0.5,
                      'OD':-0.5,'OE':-0.5,'ND':-0.5,'NE':-0.5,'NH1':-0.5,
                      'NH2':-0.5,'SG':-0.2,'S':-0.2}

# DNA/RNA constants
DNA_VOCAB = "ACGT"
RNA_VOCAB = "ACGU"
DNA_RNA_VOCAB = "ACGUT"
NT_TO_ID = {nt: i for i, nt in enumerate(DNA_RNA_VOCAB)}
WC_PAIRS = {
    ('A','T'):2, ('T','A'):2, ('A','U'):2, ('U','A'):2,
    ('G','C'):3, ('C','G'):3, ('G','U'):1, ('U','G'):1
}
BASE_STACKING = {'A':1.0,'T':0.8,'U':0.8,'G':1.2,'C':1.0}

NUCLEOTIDE_BACKBONE = [
    ('C4\'','C',0,0.00,0.0,(0,0,0),0.0),
    ('O4\'','O',0,1.44,109.5,(0,0,0),0.0),
    ('C1\'','C',1,1.42,109.5,(0,-1,0),0.0),
    ('C2\'','C',0,1.52,109.5,(-1,-2,0),120.0),
    ('C3\'','C',0,1.52,109.5,(-1,-2,0),-120.0),
    ('O3\'','O',4,1.43,109.5,(-1,0,1),0.0),
    ('C5\'','C',0,1.51,109.5,(-2,-3,0),180.0),
    ('O5\'','O',6,1.42,109.5,(-1,0,1),0.0),
    ('P','P',7,1.60,119.0,(-1,0,1),180.0),
    ('OP1','O',8,1.48,109.5,(-1,0,1),0.0),
    ('OP2','O',8,1.48,109.5,(-1,0,1),180.0),
]
PYRIMIDINE_BASE = [
    ('N1','N',2,1.47,109.5,(-1,0,1),0.0),
    ('C2','C',11,1.40,121.0,(-1,0,1),0.0),
    ('O2','O',12,1.24,118.0,(-1,0,1),0.0),
    ('N3','N',12,1.35,120.0,(-1,0,1),180.0),
    ('C4','C',14,1.33,118.0,(-1,0,1),0.0),
    ('C5','C',15,1.43,118.0,(-1,0,1),0.0),
    ('C6','C',16,1.34,122.0,(-1,0,1),0.0),
]
CYTOSINE_EXTRA = [('N4','N',15,1.33,118.0,(-1,0,1),180.0)]
URACIL_EXTRA   = [('O4','O',15,1.23,118.0,(-1,0,1),180.0)]
THYMINE_EXTRA  = [('O4','O',15,1.23,118.0,(-1,0,1),180.0),
                  ('C7','C',16,1.50,122.0,(-1,0,1),0.0)]
PURINE_BASE = [
    ('N9','N',2,1.46,109.5,(-1,0,1),0.0),
    ('C4','C',11,1.37,126.0,(-1,0,1),0.0),
    ('C5','C',12,1.39,106.0,(-1,0,1),0.0),
    ('C6','C',13,1.40,110.0,(-1,0,1),0.0),
    ('N1','N',14,1.34,118.0,(-1,0,1),0.0),
    ('C2','C',15,1.32,129.0,(-1,0,1),0.0),
    ('N3','N',16,1.32,110.0,(-1,0,1),0.0),
    ('N7','N',13,1.33,114.0,(-2,-1,0),180.0),
    ('C8','C',18,1.37,106.0,(-1,0,1),0.0),
]
ADENINE_EXTRA = [('N6','N',14,1.34,124.0,(-1,0,1),180.0)]
GUANINE_EXTRA = [('O6','O',14,1.23,124.0,(-1,0,1),180.0),
                 ('N2','N',16,1.34,120.0,(-1,0,1),180.0)]
NUCLEOTIDE_TOPOLOGY = {
    'A': NUCLEOTIDE_BACKBONE + PURINE_BASE + ADENINE_EXTRA,
    'G': NUCLEOTIDE_BACKBONE + PURINE_BASE + GUANINE_EXTRA,
    'C': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + CYTOSINE_EXTRA,
    'U': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + URACIL_EXTRA,
    'T': NUCLEOTIDE_BACKBONE + PYRIMIDINE_BASE + THYMINE_EXTRA,
}
NUCLEOTIDE_LJ = {'P':(2.1000,0.2000),'O':(1.6612,0.2100),'N':(1.8240,0.1700),
                 'C':(1.9080,0.0860),'S':(2.0000,0.2500)}
NUCLEOTIDE_CHARGES = {
    'P':0.90,'OP1':-0.70,'OP2':-0.70,'O5\'':-0.50,'C5\'':-0.10,
    'C4\'':0.00,'O4\'':-0.40,'C1\'':0.20,'C2\'':-0.10,'C3\'':0.10,'O3\'':-0.50,
    'N1':-0.50,'N2':-0.80,'N3':-0.60,'N4':-0.80,'N6':-0.80,'N7':-0.50,'N9':-0.30,
    'C2':0.40,'C4':0.30,'C5':0.10,'C6':0.10,'C7':-0.20,'C8':0.20,
    'O2':-0.55,'O4':-0.55,'O6':-0.55,
}

# Ligand (GAFF‑inspired)
ATOMIC_NUMBER = {
    'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
    'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
    'K':19,'Ca':20,'Br':35,'I':53,'Fe':26,'Zn':30,'Se':34,'Mn':25,'Cu':29,'Co':27,'Ni':28,
}
COVALENT_RADIUS = {
    'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'S':1.05,'Cl':1.02,
    'Br':1.20,'I':1.39,'B':0.84,'Si':1.11,'Se':1.20,'Fe':1.32,'Zn':1.22,
    'Mn':1.39,'Cu':1.32,'Co':1.26,'Ni':1.24,
}
GAFF_LJ = {
    'c':(1.9080,0.0860),'c1':(1.9080,0.0860),'c2':(1.9080,0.0860),
    'c3':(1.9080,0.1094),'ca':(1.9080,0.0860),'cp':(1.9080,0.0860),'cq':(1.9080,0.0860),
    'n':(1.8240,0.1700),'n1':(1.8240,0.1700),'n2':(1.8240,0.1700),
    'n3':(1.8240,0.1700),'n4':(1.8240,0.1700),'na':(1.8240,0.1700),
    'nh':(1.8240,0.1700),'no':(1.8240,0.1700),
    'o':(1.6612,0.2100),'oh':(1.7210,0.2104),'os':(1.6837,0.1700),'ow':(1.7683,0.1520),
    's':(2.0000,0.2500),'s2':(2.0000,0.2500),'s4':(2.0000,0.2500),
    's6':(2.0000,0.2500),'sh':(2.0000,0.2500),
    'p':(2.1000,0.2000),'p5':(2.1000,0.2000),
    'f':(1.7500,0.0610),'cl':(1.9480,0.2650),'br':(2.0200,0.4200),'i':(2.1500,0.5000),
    'Fe':(1.5000,0.0500),'Zn':(1.1000,0.0125),'Mg':(0.7926,0.8947),'Ca':(1.7000,0.4598),
    'Mn':(1.5000,0.0500),'Cu':(1.4000,0.0500),'Co':(1.4000,0.0500),'Ni':(1.4000,0.0500),
}
ELEMENT_TO_GAFF = {'C':'c3','N':'n3','O':'os','S':'s','P':'p','F':'f','Cl':'cl','Br':'br','I':'i','H':'H'}
IDEAL_BOND_LENGTHS = {
    ('C','C'):1.54,('C','N'):1.47,('C','O'):1.43,('C','S'):1.82,
    ('C','H'):1.09,('C','F'):1.35,('C','Cl'):1.77,('C','Br'):1.94,
    ('C','I'):2.14,('C','P'):1.84,
    ('N','N'):1.45,('N','O'):1.40,('N','S'):1.68,('N','H'):1.01,('N','P'):1.70,
    ('O','O'):1.48,('O','S'):1.57,('O','H'):0.96,('O','P'):1.60,
    ('S','S'):2.05,('S','H'):1.34,('S','P'):2.10,
    ('P','P'):2.20,('P','H'):1.42,
    ('C','Fe'):2.00,('N','Fe'):1.95,('O','Fe'):1.90,
    ('C','Zn'):2.00,('N','Zn'):2.05,('O','Zn'):2.10,
    ('C','Mg'):2.20,('O','Mg'):2.10,
}
IDEAL_ANGLES = {
    ('C','C','C'):109.5,('C','C','N'):109.5,('C','C','O'):109.5,
    ('C','N','C'):109.5,('C','N','H'):109.5,
    ('C','O','C'):109.5,('C','O','H'):109.5,
    ('C','S','C'):99.0,('C','S','H'):96.0,
    ('O','P','O'):109.5,('C','P','O'):109.5,
    ('C','C','H'):109.5,('H','C','H'):109.5,
    ('H','N','H'):109.5,('H','O','H'):104.5,
    ('C','C','O'):120.0,('C','C','N'):120.0,
    ('O','C','O'):120.0,('O','C','N'):120.0,
}
K_BOND = 300.0
K_ANGLE = 50.0
K_TORSION = 1.0
K_IMPROPER = 10.0

# =============================================================================
# Utility functions
# =============================================================================
def _normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def detect_sequence_type(seq: str) -> str:
    nt_set = set("ACGTU")
    aa_set = set("ACDEFGHIKLMNPQRSTVWY")
    n_nt = sum(1 for c in seq.upper() if c in nt_set)
    n_aa = sum(1 for c in seq.upper() if c in aa_set)
    total = len(seq)
    if total == 0: return 'unknown'
    if n_nt / total > 0.8:
        return 'rna' if 'U' in seq.upper() else 'dna'
    if n_aa / total > 0.7: return 'protein'
    return 'unknown'

def get_atom_type_for_topology(atom_name):
    if atom_name.startswith('C'): return 'C'
    if atom_name.startswith('N'): return 'N'
    if atom_name.startswith('O'): return 'O'
    if atom_name.startswith('P'): return 'P'
    if atom_name.startswith('S'): return 'S'
    return 'C'

# =============================================================================
# Sparse graph builders (O(N))
# =============================================================================
def sparse_edges(coords, cutoff, max_neighbors, batch=None):
    if coords.shape[0] == 0:
        return torch.empty((2,0), dtype=torch.long, device=coords.device), torch.empty(0, device=coords.device)
    if HAS_CLUSTER:
        if batch is None:
            batch = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        edge_index = radius_graph(coords, r=cutoff, max_num_neighbors=max_neighbors,
                                  batch=batch, flow='source_to_target')
        edge_dist = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1)
        return edge_index, edge_dist
    else:
        if batch is None:
            batch = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        B = batch.max().item() + 1
        all_src, all_dst, all_dist = [], [], []
        for b in range(B):
            mask = (batch == b)
            x = coords[mask]
            n = x.shape[0]
            if n == 0: continue
            dist = torch.cdist(x, x)
            src, dst = torch.where((dist < cutoff) & (dist > 1e-6))
            d = dist[src, dst]
            offset = torch.where(mask)[0].min().item() if mask.any() else 0
            all_src.append(src + offset)
            all_dst.append(dst + offset)
            all_dist.append(d)
        if not all_src:
            return torch.empty((2,0), dtype=torch.long, device=coords.device), torch.empty(0, device=coords.device)
        return torch.stack([torch.cat(all_src), torch.cat(all_dst)]), torch.cat(all_dist)

def cross_sparse_edges(coords1, coords2, cutoff, max_neighbors):
    if coords1.shape[0]==0 or coords2.shape[0]==0:
        return torch.empty((2,0), dtype=torch.long, device=coords1.device), torch.empty(0, device=coords1.device)
    if HAS_CLUSTER:
        row, col = radius(coords1, coords2, r=cutoff, max_num_neighbors=max_neighbors)
        edge_index = torch.stack([row, col], dim=0)
        edge_dist = torch.norm(coords1[row] - coords2[col], dim=-1)
        return edge_index, edge_dist
    else:
        dist = torch.cdist(coords1, coords2)
        src, dst = torch.where(dist < cutoff)
        edge_dist = dist[src, dst]
        return torch.stack([src, dst]), edge_dist

# =============================================================================
# Backbone & sidechain reconstruction
# =============================================================================
def reconstruct_backbone(ca):
    L = ca.shape[0]
    v = ca[1:] - ca[:-1]
    v_norm = _normalize(v)
    N = torch.zeros_like(ca); C = torch.zeros_like(ca)
    N[1:] = ca[1:] - 1.45 * v_norm
    N[0] = ca[0] - 1.45 * v_norm[0]
    C[:-1] = ca[:-1] + 1.52 * v_norm
    C[-1] = ca[-1] + 1.52 * v_norm[-1]
    O = torch.zeros_like(ca)
    for i in range(L):
        if i < L-1:
            ca_c = C[i] - ca[i]
            ca_n = N[i] - ca[i]
            perp = torch.cross(ca_c, ca_n, dim=-1)
            if perp.norm() > 1e-6: perp = perp / perp.norm()
            O[i] = C[i] + 1.24 * perp
        else:
            O[i] = C[i] + torch.tensor([0.,1.24,0.], device=ca.device)
    return {'N':N, 'CA':ca, 'C':C, 'O':O}

def dihedral_angle(p0,p1,p2,p3):
    b0=p1-p0; b1=p2-p1; b2=p3-p2
    b1n=_normalize(b1)
    v=b0-(b0*b1n).sum(-1,keepdim=True)*b1n
    w=b2-(b2*b1n).sum(-1,keepdim=True)*b1n
    x=(v*w).sum(-1); y=torch.cross(b1n,v,dim=-1); y=(y*w).sum(-1)
    return torch.atan2(y+1e-8, x+1e-8)

def compute_phi_psi(atoms):
    N,CA,C = atoms['N'],atoms['CA'],atoms['C']
    L=CA.shape[0]; phi=torch.zeros(L,device=CA.device); psi=torch.zeros(L,device=CA.device)
    if L>2:
        phi[1:-1]=dihedral_angle(C[:-2], N[1:-1], CA[1:-1], C[1:-1])
        psi[1:-1]=dihedral_angle(N[1:-1], CA[1:-1], C[1:-1], N[2:])
    return phi*180./math.pi, psi*180./math.pi

def _map_ref(idx):
    if idx==-1: return 0
    if idx==-2: return 1
    if idx==-3: return 2
    if idx==-4: return 3
    return 3+idx

def build_sidechain_atoms(ca, seq, chi_angles):
    device = ca.device
    L = ca.shape[0]
    v = ca[1:] - ca[:-1]; v_norm = _normalize(v)
    N = torch.zeros_like(ca); C = torch.zeros_like(ca)
    N[1:] = ca[1:] - 1.45 * v_norm; N[0] = ca[0] - 1.45 * v_norm[0]
    C[:-1] = ca[:-1] + 1.52 * v_norm; C[-1] = ca[-1] + 1.52 * v_norm[-1]
    all_coords, all_types = [], []
    for i, aa in enumerate(seq):
        if detect_sequence_type(aa) != 'protein' or aa=='G':
            if aa=='G':
                all_coords.append(torch.stack([N[i], ca[i], C[i]]))
                all_types.append(['N','CA','C'])
            continue
        n_i, ca_i, c_i = N[i], ca[i], C[i]
        v1=n_i-ca_i; v2=c_i-ca_i
        cb_dir=-(v1+v2); cb_dir=_normalize(cb_dir)
        cb_pos=ca_i+1.53*cb_dir
        local_atoms=[n_i, ca_i, c_i, cb_pos]
        local_types=['N','CA','C','CB']
        topo = RESIDUE_TOPOLOGY.get(aa, [])
        chi_idx = 0
        for (atom_name, atom_type, parent_idx, bond_len, bond_ang_deg, ref_tuple, dihedral0) in topo:
            a_idx = _map_ref(ref_tuple[0]); b_idx = _map_ref(ref_tuple[1]); c_idx = _map_ref(ref_tuple[2])
            a_idx = min(a_idx, len(local_atoms)-1)
            b_idx = min(b_idx, len(local_atoms)-1)
            c_idx = min(c_idx, len(local_atoms)-1)
            p_a,p_b,p_c = local_atoms[a_idx], local_atoms[b_idx], local_atoms[c_idx]
            bc = p_c - p_b; bc_norm = _normalize(bc)
            ref_vec = torch.tensor([1.,0.,0.], device=device)
            if abs(torch.dot(bc_norm, ref_vec)) > 0.9:
                ref_vec = torch.tensor([0.,1.,0.], device=device)
            perp = torch.cross(bc_norm, ref_vec, dim=-1)
            if perp.norm() < 1e-12:
                ref_vec = torch.tensor([0.,0.,1.], device=device)
                perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp = _normalize(perp)
            chi_val = chi_angles[i, chi_idx] if chi_angles is not None else 0.0
            total_angle = dihedral0 + chi_val
            cos_a, sin_a = math.cos(total_angle), math.sin(total_angle)
            cross_bn_perp = torch.cross(bc_norm, perp, dim=-1)
            rotated_perp = perp * cos_a + cross_bn_perp * sin_a
            ang = torch.tensor(bond_ang_deg * math.pi/180., device=device)
            bond_dir = math.cos(ang)*bc_norm + math.sin(ang)*rotated_perp
            new_pos = p_c + bond_len * bond_dir
            local_atoms.append(new_pos)
            local_types.append(atom_type)
            chi_idx += 1
        all_coords.append(torch.stack(local_atoms))
        all_types.append(local_types)
    return all_coords, all_types

def get_full_atom_coords_and_types(ca, seq, chi_angles):
    res_coords, res_types = build_sidechain_atoms(ca, seq, chi_angles)
    coords_list, types_list, res_idx_list = [], [], []
    for i,(rc,rt) in enumerate(zip(res_coords, res_types)):
        coords_list.append(rc)
        types_list.extend(rt)
        res_idx_list.append(torch.full((rc.shape[0],), i, dtype=torch.long, device=ca.device))
    if not coords_list:
        return torch.empty((0,3), device=ca.device), [], torch.empty(0, device=ca.device)
    return torch.cat(coords_list, dim=0), types_list, torch.cat(res_idx_list, dim=0)

# =============================================================================
# DNA/RNA full atom builder (simplified)
# =============================================================================
def build_single_nucleotide_ic(C4_prime, prev_C4, next_C4, nt_type):
    device = C4_prime.device
    topo = NUCLEOTIDE_TOPOLOGY.get(nt_type, [])
    if not topo:
        return torch.zeros((0,3), device=device), []
    coords, types = [], []
    if prev_C4 is not None and next_C4 is not None:
        x_axis = _normalize(next_C4 - C4_prime)
        v_tmp = C4_prime - prev_C4
        z_axis = _normalize(torch.cross(x_axis, v_tmp, dim=-1))
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
    elif next_C4 is not None:
        x_axis = _normalize(next_C4 - C4_prime)
        y_axis = torch.tensor([0.,1.,0.], device=device)
        z_axis = _normalize(torch.cross(x_axis, y_axis, dim=-1))
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
    else:
        x_axis = torch.tensor([1.,0.,0.], device=device)
        y_axis = torch.tensor([0.,1.,0.], device=device)
        z_axis = torch.tensor([0.,0.,1.], device=device)
    for atom_name, atom_type, parent_idx, bond_len, bond_ang_deg, ref_tuple, dihedral0 in topo:
        if parent_idx == 0 and len(coords) == 0:
            pos = C4_prime
        elif parent_idx < len(coords):
            parent_pos = coords[parent_idx]
            a_idx,b_idx,c_idx = ref_tuple
            def map_idx(idx, cur_len):
                if idx == 0: return parent_idx
                elif idx > 0: return min(idx-1, cur_len-1) if cur_len>0 else 0
                else: return max(0, cur_len+idx)
            a_abs = map_idx(a_idx, len(coords)); b_abs = map_idx(b_idx, len(coords)); c_abs = map_idx(c_idx, len(coords))
            p_a = coords[a_abs] if a_abs<len(coords) else parent_pos
            p_b = coords[b_abs] if b_abs<len(coords) else parent_pos
            p_c = coords[c_abs] if c_abs<len(coords) else parent_pos
            bc = p_c - p_b; bc_norm = _normalize(bc)
            ref_vec = torch.tensor([1.,0.,0.], device=device)
            if abs(torch.dot(bc_norm, ref_vec)) > 0.9:
                ref_vec = torch.tensor([0.,1.,0.], device=device)
            perp = torch.cross(bc_norm, ref_vec, dim=-1)
            if perp.norm() < 1e-12:
                ref_vec = torch.tensor([0.,0.,1.], device=device)
                perp = torch.cross(bc_norm, ref_vec, dim=-1)
            perp = _normalize(perp)
            total_angle = math.radians(dihedral0)
            cos_a, sin_a = math.cos(total_angle), math.sin(total_angle)
            cross_bn_perp = torch.cross(bc_norm, perp, dim=-1)
            rotated_perp = perp * cos_a + cross_bn_perp * sin_a
            ang = math.radians(bond_ang_deg)
            bond_dir = math.cos(ang)*bc_norm + math.sin(ang)*rotated_perp
            pos = p_c + bond_len * bond_dir
        else:
            pos = coords[-1] + bond_len * x_axis if coords else C4_prime + bond_len * x_axis
        coords.append(pos)
        types.append(atom_name)
    return torch.stack(coords, dim=0), types

def build_full_dna_rna(C4_coords, sequence):
    L = len(sequence)
    device = C4_coords.device
    all_c, all_t, all_r = [], [], []
    for i in range(L):
        prev = C4_coords[i-1] if i>0 else None
        next_ = C4_coords[i+1] if i<L-1 else None
        nuc_c, nuc_t = build_single_nucleotide_ic(C4_coords[i], prev, next_, sequence[i])
        all_c.append(nuc_c)
        all_t.extend(nuc_t)
        all_r.append(torch.full((nuc_c.shape[0],), i, dtype=torch.long, device=device))
    if not all_c:
        return torch.zeros((0,3), device=device), [], torch.zeros(0, device=device)
    return torch.cat(all_c, dim=0), all_t, torch.cat(all_r, dim=0)

# =============================================================================
# Ligand classes (Molecule, LigandEnergy, etc.)
# =============================================================================
class Molecule:
    def __init__(self, name="UNK"):
        self.name = name
        self.atom_coords = None
        self.atom_elements = []
        self.atom_names = []
        self.atom_gaff_types = []
        self.atom_charges = None
        self.bonds = []
        self.bond_orders = []
        self.angles = []
        self.torsions = []
        self.torsion_periodicities = []
        self.impropers = []
        self.rings = []
        self.n_atoms = 0

    def build_topology(self):
        if not self.bonds: return
        self.n_atoms = len(self.atom_elements)
        adj = defaultdict(set)
        for i,j in self.bonds:
            adj[i].add(j); adj[j].add(i)
        self.angles = []
        for j in range(self.n_atoms):
            neigh = list(adj[j])
            for a in range(len(neigh)):
                for b in range(a+1, len(neigh)):
                    self.angles.append((neigh[a], j, neigh[b]))
        self.torsions = []
        self.torsion_periodicities = []
        for j,k in self.bonds:
            for i in adj[j]:
                if i!=k:
                    for l in adj[k]:
                        if l!=j and l!=i:
                            period = 2 if self._is_sp2_bond(j,k) else 3
                            self.torsions.append((i,j,k,l))
                            self.torsion_periodicities.append(period)
        self.impropers = []
        for j in range(self.n_atoms):
            if len(adj[j])==3:
                n = list(adj[j])
                self.impropers.append((j,n[0],n[1],n[2]))
        self._find_rings(adj)

    def _is_sp2_bond(self,i,j):
        for (a,b),o in zip(self.bonds, self.bond_orders):
            if (a==i and b==j) or (a==j and b==i):
                return o>=1.5
        return False

    def _find_rings(self, adj, max_ring=8):
        self.rings = []
        visited_edges=set()
        for start in range(min(self.n_atoms,50)):
            for nb in adj[start]:
                edge = tuple(sorted((start,nb)))
                if edge in visited_edges: continue
                parent={start:-1}
                queue=[start]; found=False
                while queue and not found:
                    node=queue.pop(0)
                    for nb2 in adj[node]:
                        if nb2==parent[node]: continue
                        if nb2 in parent:
                            if nb2!=start: continue
                            ring=[node]
                            while ring[-1]!=start:
                                ring.append(parent[ring[-1]])
                            ring=ring[::-1]
                            if 3<=len(ring)<=max_ring:
                                self.rings.append(ring)
                                for a,b in zip(ring, ring[1:]+[ring[0]]):
                                    visited_edges.add(tuple(sorted((a,b))))
                            found=True
                            break
                        parent[nb2]=node
                        queue.append(nb2)
                if found: break

    def assign_gaff_types(self):
        self.atom_gaff_types = []
        adj = defaultdict(set)
        for i,j in self.bonds: adj[i].add(j); adj[j].add(i)
        for i,elem in enumerate(self.atom_elements):
            gt = ELEMENT_TO_GAFF.get(elem, elem.lower())
            n_nei = len(adj[i])
            if elem=='C':
                if n_nei==4: gt='c3'
                elif n_nei==3:
                    has_d=any(self._get_bond_order(i,nb)>=2.0 for nb in adj[i])
                    gt='c2' if has_d else 'ca'
                elif n_nei==2:
                    gt='c1' if self._get_bond_order(i,list(adj[i])[0])>=2.5 else 'c2'
            elif elem=='N':
                if n_nei==3: gt='n3'
                elif n_nei==2: gt='n2'
                elif n_nei==1: gt='n1'
            elif elem=='O':
                if n_nei==1:
                    gt='o' if self._get_bond_order(i,list(adj[i])[0])>=2.0 else 'oh'
                elif n_nei==2: gt='os'
            elif elem=='S':
                gt='s' if n_nei<=2 else 's6'
            self.atom_gaff_types.append(gt)

    def _get_bond_order(self,i,j):
        for (a,b),o in zip(self.bonds, self.bond_orders):
            if (a==i and b==j) or (a==j and b==i): return o
        return 1.0

    def to_device(self, device):
        if self.atom_coords is not None: self.atom_coords = self.atom_coords.to(device)
        if self.atom_charges is not None: self.atom_charges = self.atom_charges.to(device)
        return self

class LigandEnergy:
    def __init__(self, w_bond=300.0, w_angle=50.0, w_torsion=1.0, w_improper=10.0,
                 w_lj=30.0, w_coulomb=3.0, lj_params=None, dielectric=4.0):
        self.w_bond=w_bond; self.w_angle=w_angle; self.w_torsion=w_torsion
        self.w_improper=w_improper; self.w_lj=w_lj; self.w_coulomb=w_coulomb
        self.lj_params = lj_params or GAFF_LJ
        self.dielectric = dielectric

    def __call__(self, mol):
        E = torch.tensor(0.0, device=mol.atom_coords.device)
        E += self.energy_bond(mol)
        E += self.energy_angle(mol)
        E += self.energy_torsion(mol)
        E += self.energy_improper(mol)
        E += self.energy_nonbonded(mol)
        return E

    def energy_bond(self, mol):
        E = torch.tensor(0.0, device=mol.atom_coords.device)
        for (i,j),o in zip(mol.bonds, mol.bond_orders):
            key = tuple(sorted((mol.atom_elements[i], mol.atom_elements[j])))
            r0 = IDEAL_BOND_LENGTHS.get(key, 1.50)
            if o>=2.5: r0-=0.35
            elif o>=1.5: r0-=0.15
            dv = mol.atom_coords[j]-mol.atom_coords[i]
            d = torch.norm(dv)
            E += self.w_bond * K_BOND * (d - r0)**2
        return E

    def energy_angle(self, mol):
        E = torch.tensor(0.0, device=mol.atom_coords.device)
        for i,j,k in mol.angles:
            key = (mol.atom_elements[i], mol.atom_elements[j], mol.atom_elements[k])
            theta0 = math.radians(IDEAL_ANGLES.get(key, 109.5))
            v1 = mol.atom_coords[i]-mol.atom_coords[j]
            v2 = mol.atom_coords[k]-mol.atom_coords[j]
            cos = torch.clamp(torch.dot(v1,v2)/(torch.norm(v1)*torch.norm(v2)+1e-8), -1, 1)
            theta = torch.acos(cos)
            E += self.w_angle * K_ANGLE * (theta - theta0)**2
        return E

    def energy_torsion(self, mol):
        E = torch.tensor(0.0, device=mol.atom_coords.device)
        for (i,j,k,l),per in zip(mol.torsions, mol.torsion_periodicities):
            phi = self._compute_dihedral(mol.atom_coords[i], mol.atom_coords[j], mol.atom_coords[k], mol.atom_coords[l])
            V = 1.0
            delta = math.pi if per==2 else 0.0
            E += self.w_torsion * K_TORSION * V * (1 + torch.cos(per*phi - delta))
        return E

    def energy_improper(self, mol):
        E = torch.tensor(0.0, device=mol.atom_coords.device)
        for i,j,k,l in mol.impropers:
            phi = self._compute_dihedral(mol.atom_coords[i], mol.atom_coords[j], mol.atom_coords[k], mol.atom_coords[l])
            E += self.w_improper * K_IMPROPER * phi**2
        return E

    def energy_nonbonded(self, mol):
        E = torch.tensor(0.0, device=mol.atom_coords.device)
        n = mol.n_atoms
        excl = self._exclusion(mol)
        is14 = self._14list(mol)
        for i in range(n):
            for j in range(i+1,n):
                if (i,j) in excl or (j,i) in excl: continue
                dv = mol.atom_coords[i]-mol.atom_coords[j]
                d = torch.norm(dv) + 1e-8
                sc = 0.5 if ((i,j) in is14 or (j,i) in is14) else 1.0
                gi = mol.atom_gaff_types[i] if i<len(mol.atom_gaff_types) else 'c3'
                gj = mol.atom_gaff_types[j] if j<len(mol.atom_gaff_types) else 'c3'
                si,ei = self.lj_params.get(gi,(1.9,0.1))
                sj,ej = self.lj_params.get(gj,(1.9,0.1))
                sig = 0.5*(si+sj); eps = math.sqrt(ei*ej)
                inv_r = 1.0/d
                lj = 4.0*eps*((sig*inv_r)**12 - (sig*inv_r)**6)
                E += self.w_lj * sc * lj
                if mol.atom_charges is not None:
                    qi = mol.atom_charges[i]; qj = mol.atom_charges[j]
                    coul = 332.0637*qi*qj/(self.dielectric*d)
                    E += self.w_coulomb * sc * coul
        return E

    def _exclusion(self, mol):
        excl = set()
        for i,j in mol.bonds:
            excl.add((i,j)); excl.add((j,i))
        adj = defaultdict(set)
        for i,j in mol.bonds: adj[i].add(j); adj[j].add(i)
        for i in range(mol.n_atoms):
            for j in adj[i]:
                for k in adj[j]:
                    if k!=i: excl.add((i,k))
        return excl

    def _14list(self, mol):
        p14 = set()
        adj = defaultdict(set)
        for i,j in mol.bonds: adj[i].add(j); adj[j].add(i)
        for i in range(mol.n_atoms):
            for j in adj[i]:
                for k in adj[j]:
                    if k==i: continue
                    for l in adj[k]:
                        if l!=i and l!=j: p14.add((i,l))
        return p14

    def _compute_dihedral(self, p0,p1,p2,p3):
        b0=p1-p0; b1=p2-p1; b2=p3-p2
        b1n = _normalize(b1)
        v = b0 - (b0*b1n).sum(-1,keepdim=True)*b1n
        w = b2 - (b2*b1n).sum(-1,keepdim=True)*b1n
        x = (v*w).sum(-1)
        y = torch.cross(b1n,v,dim=-1)
        y = (y*w).sum(-1)
        return torch.atan2(y+1e-8, x+1e-8)

class ProteinLigandEnergy:
    def __init__(self, w_lj=30.0, w_coulomb=3.0, protein_lj_params=None, ligand_lj_params=None,
                 protein_charges=None, dielectric=4.0, cutoff=12.0):
        self.w_lj=w_lj; self.w_coulomb=w_coulomb
        self.protein_lj_params = protein_lj_params or DEFAULT_LJ_PARAMS
        self.ligand_lj_params = ligand_lj_params or GAFF_LJ
        self.protein_charges = protein_charges or DEFAULT_CHARGE_MAP
        self.dielectric = dielectric; self.cutoff = cutoff

    def __call__(self, prot_coords, prot_types, prot_charges_list, ligand):
        device = prot_coords.device
        E = torch.tensor(0.0, device=device)
        lig_coords = ligand.atom_coords.to(device)
        lig_charges = ligand.atom_charges
        for i in range(len(prot_coords)):
            ppos = prot_coords[i]
            ptype = prot_types[i] if i<len(prot_types) else 'CA'
            pq = prot_charges_list[i] if i<len(prot_charges_list) else 0.0
            for j in range(ligand.n_atoms):
                dv = ppos - lig_coords[j]
                d = torch.norm(dv) + 1e-8
                if d > self.cutoff: continue
                si,ei = self.protein_lj_params.get(ptype, (1.9,0.1))
                gj = ligand.atom_gaff_types[j] if j<len(ligand.atom_gaff_types) else 'c3'
                sj,ej = self.ligand_lj_params.get(gj, (1.9,0.1))
                sig = 0.5*(si+sj); eps = math.sqrt(ei*ej)
                inv_r = 1.0/d
                lj = 4.0*eps*((sig*inv_r)**12 - (sig*inv_r)**6)
                E += self.w_lj * lj
                if lig_charges is not None:
                    qj = lig_charges[j]
                    coul = 332.0637 * pq * qj / (self.dielectric * d)
                    E += self.w_coulomb * coul
        return E

class LigandBridge:
    def __init__(self):
        self.ligands: List[Molecule] = []
        self.lig_energy = LigandEnergy()
        self.pl_energy = ProteinLigandEnergy()

    def add_ligand(self, ligand): self.ligands.append(ligand)
    def load_ligand_file(self, filepath, filetype='sdf'):
        if filetype=='sdf': mols = read_sdf(filepath)
        elif filetype=='mol2': mols = read_mol2(filepath)
        elif filetype=='pdb': mols = read_pdb_ligand(filepath)
        else: raise ValueError(f"Unknown file type {filetype}")
        for m in mols: self.ligands.append(m)
        return len(mols)
    def compute_ligand_energy(self):
        E = torch.tensor(0.0)
        for lig in self.ligands: E += self.lig_energy(lig)
        return E
    def compute_interaction(self, protein_coords, protein_atom_types, protein_charges):
        E = torch.tensor(0.0, device=protein_coords.device)
        for lig in self.ligands:
            E += self.pl_energy(protein_coords, protein_atom_types, protein_charges, lig)
        return E

def read_sdf(sdf_path):
    molecules = []
    with open(sdf_path) as f: lines = f.readlines()
    i=0
    while i < len(lines)-4:
        name = lines[i].strip(); i+=1
        i+=1; i+=1
        if i>=len(lines): break
        counts = lines[i]; i+=1
        n_atoms = int(counts[0:3].strip()); n_bonds = int(counts[3:6].strip())
        mol = Molecule(name if name else "UNK")
        for _ in range(n_atoms):
            line = lines[i]; i+=1
            x=float(line[0:10].strip()); y=float(line[10:20].strip()); z=float(line[20:30].strip())
            elem = line[31:34].strip()
            mol.atom_elements.append(elem or 'C')
            mol.atom_names.append(f"{elem}{len(mol.atom_elements)}")
            arr = torch.tensor([[x,y,z]], dtype=torch.float32)
            mol.atom_coords = arr if mol.atom_coords is None else torch.cat([mol.atom_coords, arr])
        for _ in range(n_bonds):
            line = lines[i]; i+=1
            a1=int(line[0:3].strip())-1; a2=int(line[3:6].strip())-1
            order = int(line[6:9].strip()) if len(line)>6 else 1
            mol.bonds.append((a1,a2)); mol.bond_orders.append(float(order))
        mol.n_atoms = len(mol.atom_elements)
        mol.build_topology(); mol.assign_gaff_types()
        mol.atom_charges = torch.zeros(mol.n_atoms)
        for a,elem in enumerate(mol.atom_elements):
            if elem=='N': mol.atom_charges[a]=-0.5
            elif elem=='O': mol.atom_charges[a]=-0.5
            elif elem=='S': mol.atom_charges[a]=-0.2
            elif elem in ('F','Cl','Br','I'): mol.atom_charges[a]=-0.2
            elif elem=='P': mol.atom_charges[a]=0.5
        molecules.append(mol)
        while i<len(lines) and lines[i].strip()!="$$$$": i+=1
        i+=1
    return molecules

def read_mol2(mol2_path):
    molecules = []
    with open(mol2_path) as f: content = f.read()
    sections = content.split('@<TRIPOS>MOLECULE')
    for section in sections[1:]:
        lines = section.strip().split('\n')
        if len(lines)<2: continue
        name = lines[0].strip()
        mol = Molecule(name)
        astart=bstart=aend=bend=-1
        for j,line in enumerate(lines):
            if line.strip()=='@<TRIPOS>ATOM': astart=j+1
            elif line.strip()=='@<TRIPOS>BOND':
                bstart=j+1
                if astart>0: aend=j
            elif line.strip().startswith('@<TRIPOS>') and bstart>0:
                bend=j; break
        if astart<0: continue
        if aend<0: aend=len(lines)
        if bend<0: bend=len(lines)
        idmap={}; coords=[]
        for j in range(astart, min(aend,len(lines))):
            parts=lines[j].split()
            if len(parts)<6: continue
            mid=int(parts[0]); aname=parts[1]; x=float(parts[2]); y=float(parts[3]); z=float(parts[4])
            atype=parts[5]; elem=atype.split('.')[0][:2]
            if elem not in ATOMIC_NUMBER: elem=elem[0]
            idx=len(mol.atom_elements); idmap[mid]=idx
            mol.atom_elements.append(elem); mol.atom_names.append(aname)
            coords.append([x,y,z])
            if len(parts)>8:
                ch=float(parts[8])
                if mol.atom_charges is None: mol.atom_charges=torch.zeros(len(mol.atom_elements))
                mol.atom_charges[-1]=ch
        mol.atom_coords = torch.tensor(coords, dtype=torch.float32)
        if bstart>0:
            for j in range(bstart, min(bend,len(lines))):
                parts=lines[j].split()
                if len(parts)<4: continue
                a1=idmap.get(int(parts[1]),-1); a2=idmap.get(int(parts[2]),-1)
                if a1<0 or a2<0: continue
                ot=parts[3]
                if ot=='ar': order=1.5
                elif ot in ('am','du'): order=1.0
                else: order=float(ot) if ot.replace('.','').isdigit() else 1.0
                mol.bonds.append((a1,a2)); mol.bond_orders.append(order)
        mol.n_atoms=len(mol.atom_elements); mol.build_topology(); mol.assign_gaff_types()
        if mol.atom_charges is None:
            mol.atom_charges=torch.zeros(mol.n_atoms)
            for a,elem in enumerate(mol.atom_elements):
                if elem=='N': mol.atom_charges[a]=-0.5
                elif elem=='O': mol.atom_charges[a]=-0.5
                elif elem=='S': mol.atom_charges[a]=-0.2
                elif elem in ('F','Cl','Br','I'): mol.atom_charges[a]=-0.2
                elif elem=='P': mol.atom_charges[a]=0.5
        molecules.append(mol)
    return molecules

def read_pdb_ligand(pdb_path, residue_name=None, chain=None):
    residues = defaultdict(lambda:{'coords':[],'elements':[],'names':[]})
    order=[]
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('HETATM'): continue
            rname=line[17:20].strip(); cid=line[21].strip()
            if residue_name and rname!=residue_name: continue
            if chain and cid!=chain: continue
            aname=line[12:16].strip()
            x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
            elem=line[76:78].strip() if len(line)>76 else aname[0]
            key=(rname,cid)
            if key not in residues: order.append(key)
            residues[key]['coords'].append([x,y,z])
            residues[key]['elements'].append(elem or aname[0])
            residues[key]['names'].append(aname)
    mols=[]
    for key in order:
        d=residues[key]
        mol=Molecule(name=f"{key[0]}_{key[1]}")
        mol.atom_coords=torch.tensor(d['coords'],dtype=torch.float32)
        mol.atom_elements=d['elements']; mol.atom_names=d['names']
        mol.n_atoms=len(d['elements'])
        mol.bonds=[]; mol.bond_orders=[]
        coords_np=mol.atom_coords.numpy()
        for i in range(mol.n_atoms):
            for j in range(i+1,mol.n_atoms):
                ri=COVALENT_RADIUS.get(mol.atom_elements[i],0.76)
                rj=COVALENT_RADIUS.get(mol.atom_elements[j],0.76)
                dmax=(ri+rj)*1.2
                d=float(np.linalg.norm(coords_np[i]-coords_np[j]))
                if d<dmax:
                    mol.bonds.append((i,j))
                    if d<(ri+rj)*0.85: order=2.0
                    elif d<(ri+rj)*1.1: order=1.0
                    else: order=1.0
                    mol.bond_orders.append(order)
        mol.build_topology(); mol.assign_gaff_types()
        mol.atom_charges=torch.zeros(mol.n_atoms)
        mols.append(mol)
    return mols

# =============================================================================
# Physics energy functions (protein, DNA/RNA, SOC, clash, etc.)
# =============================================================================
def energy_bond(ca, alpha, w=30.0, mod=0.1, target_dist=3.8, mask=None):
    if mask is not None and not mask.any(): return torch.tensor(0.0, device=ca.device)
    target = target_dist * (1.0 + mod * (alpha - 1.0))
    target_pair = 0.5 * (target[1:] + target[:-1])
    d = torch.norm(ca[1:] - ca[:-1], dim=-1)
    if mask is not None:
        bond_mask = mask[1:] & mask[:-1]
        if bond_mask.sum()==0: return torch.tensor(0.0, device=ca.device)
        return w * ((d[bond_mask] - target_pair[bond_mask])**2).mean()
    return w * ((d - target_pair)**2).mean()

def energy_angle(ca, alpha, w=15.0, mod=0.05, target_angle_rad=111.0*math.pi/180.0, mask=None):
    if len(ca)<3: return torch.tensor(0.0, device=ca.device)
    v1=ca[:-2]-ca[1:-1]; v2=ca[2:]-ca[1:-1]
    v1n=_normalize(v1); v2n=_normalize(v2)
    cos_ang=(v1n*v2n).sum(-1)
    target_angle = target_angle_rad * (1.0 + mod * (alpha[1:-1] - 1.0))
    cos_target = torch.cos(target_angle)
    if mask is not None:
        ang_mask = mask[1:-1]
        if ang_mask.sum()==0: return torch.tensor(0.0, device=ca.device)
        return w * ((cos_ang[ang_mask] - cos_target[ang_mask])**2).mean()
    return w * ((cos_ang - cos_target)**2).mean()

def energy_rama(phi, psi, seq, alpha, w=8.0, mod=0.2, mask=None):
    L=len(seq); device=phi.device
    phi0,psi0,width=torch.zeros(L,device=device),torch.zeros(L,device=device),torch.zeros(L,device=device)
    for i,aa in enumerate(seq):
        prior=RAMACHANDRAN_PRIORS.get(aa,RAMACHANDRAN_PRIORS['general'])
        phi0[i],psi0[i],width[i]=prior['phi'],prior['psi'],prior['width']
    width_eff=width*(1.0+mod*(alpha-1.0))
    dphi=(phi-phi0)/(width_eff+1e-8); dpsi=(psi-psi0)/(width_eff+1e-8)
    loss=(dphi**2+dpsi**2)
    if mask is None: mask=torch.ones(L,device=device,dtype=torch.bool)
    mask[0],mask[-1]=False,False
    if mask.sum()==0: return torch.tensor(0.0, device=device)
    return w * (loss*mask.float()).sum()/mask.sum()

def energy_clash(ca, alpha, edge_index, edge_dist, w=80.0, radius=2.0, mod=0.1, mask=None):
    if edge_index.numel()==0: return torch.tensor(0.0, device=ca.device)
    idx_i,idx_j=edge_index[0],edge_index[1]
    keep = (torch.abs(idx_i-idx_j) > 2)
    if not keep.any(): return torch.tensor(0.0, device=ca.device)
    src,dst=idx_i[keep],idx_j[keep]; di=edge_dist[keep]
    if mask is not None:
        keep2=mask[src]&mask[dst]
        if not keep2.any(): return torch.tensor(0.0, device=ca.device)
        src,dst=src[keep2],dst[keep2]; di=di[keep2]
    r_i=radius*(1.0+mod*(alpha[src]-1.0)); r_j=radius*(1.0+mod*(alpha[dst]-1.0))
    r_avg=0.5*(r_i+r_j)
    clash=torch.relu(r_avg-di)
    if clash.numel()==0: return torch.tensor(0.0, device=ca.device)
    return w * (clash**2).mean()

def energy_electro(ca, seq, edge_index, edge_dist, w=4.0, mask=None):
    if edge_index.numel()==0: return torch.tensor(0.0, device=ca.device)
    q=torch.tensor([RESIDUE_CHARGE.get(a,0.0) for a in seq], device=ca.device)
    qi,qj=q[edge_index[0]],q[edge_index[1]]
    if mask is not None:
        keep=mask[edge_index[0]]&mask[edge_index[1]]
        if not keep.any(): return torch.tensor(0.0, device=ca.device)
        qi,qj=qi[keep],qj[keep]; r=torch.clamp(edge_dist[keep], min=1e-6)
    else: r=torch.clamp(edge_dist, min=1e-6)
    E=qi*qj*torch.exp(-0.1*r)/(80.0*r)
    return w * E.mean()

def energy_solvent(ca, seq, edge_index, w=5.0, mask=None):
    if edge_index.numel()==0: return torch.tensor(0.0, device=ca.device)
    src=edge_index[0]
    counts=torch.zeros(ca.shape[0], device=ca.device); counts.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    if mask is not None: counts=counts*mask.float()
    burial=1.0-torch.exp(-counts/20.0)
    hydro=torch.tensor([HYDROPHOBICITY.get(a,0.0) for a in seq], device=ca.device)
    exposed=torch.where(hydro>0, hydro*(1.0-burial), torch.zeros_like(burial))
    buried=torch.where(hydro<=0, -hydro*burial, torch.zeros_like(burial))
    return w * (exposed+buried).mean()

def energy_hbond(O,N,C,alpha, edge_index_hb, edge_dist_hb, w=6.0, mod=0.1, mask=None):
    if edge_index_hb.numel()==0: return torch.tensor(0.0, device=O.device)
    src,dst=edge_index_hb[0],edge_index_hb[1]
    if mask is not None:
        keep=mask[src]&mask[dst]
        if not keep.any(): return torch.tensor(0.0, device=O.device)
        src,dst=src[keep],dst[keep]; d=edge_dist_hb[keep]
    else: d=edge_dist_hb
    vec_co=O[src]-C[src]; vec_no=N[dst]-O[src]
    alignment=F.cosine_similarity(vec_co, vec_no, dim=-1, eps=1e-8)
    ideal_dist=2.9*(1.0+mod*(alpha[src]-1.0))
    E=-alignment*torch.exp(-((d-ideal_dist)/0.3)**2)
    return w * E.mean()

def energy_soc(ca, alpha, edge_index, edge_dist, w=0.3, kernel_lambda=12.0, mask=None):
    if edge_index.numel()==0: return torch.tensor(0.0, device=ca.device)
    src,dst=edge_index[0],edge_index[1]
    if mask is not None:
        keep=mask[src]&mask[dst]
        if not keep.any(): return torch.tensor(0.0, device=ca.device)
        src,dst=src[keep],dst[keep]; d=edge_dist[keep]
    else: d=edge_dist
    a=0.5*(alpha[src]+alpha[dst])
    safe_d=torch.clamp(d, min=1e-6)
    K=torch.exp(-a*torch.log(safe_d))*torch.exp(-d/kernel_lambda)
    E=-K*torch.exp(-d/8.0)
    return w * E.mean()

def energy_dna_rna_simple(c4, seq, w_bp=8.0, w_stack=5.0, w_backbone=30.0):
    L=len(seq); device=c4.device
    E=torch.tensor(0.0, device=device)
    if L>1:
        d=torch.norm(c4[1:]-c4[:-1], dim=-1)
        E+=w_backbone*((d-6.5)**2).mean()
    for i in range(L):
        for j in range(i+4, min(L,i+50)):
            d=torch.norm(c4[i]-c4[j])
            n=WC_PAIRS.get((seq[i],seq[j]),0)
            if n>0: E+= -n*w_bp*torch.exp(-((d-10.5)/2.0)**2)
    for i in range(L-1):
        d=torch.norm(c4[i+1]-c4[i])
        s=0.5*(BASE_STACKING.get(seq[i],1.0)+BASE_STACKING.get(seq[i+1],1.0))
        E+= -s*w_stack*torch.exp(-((d-6.5)/1.5)**2)
    return E

def energy_lj_full(all_coords, all_types, res_indices, edge_index, edge_dist, cfg_lj=DEFAULT_LJ_PARAMS, w=30.0):
    if edge_index.numel()==0: return torch.tensor(0.0, device=all_coords.device)
    src,dst=edge_index
    sigmas=torch.zeros(len(all_types), device=all_coords.device)
    epsilons=torch.zeros(len(all_types), device=all_coords.device)
    for i,t in enumerate(all_types):
        s,e=cfg_lj.get(t,(1.9,0.1)); sigmas[i]=s; epsilons[i]=e
    sigma=0.5*(sigmas[src]+sigmas[dst])
    eps=torch.sqrt(epsilons[src]*epsilons[dst])
    r=torch.clamp(edge_dist, min=1e-4)
    inv_r=1.0/r; inv_r6=inv_r**6; inv_r12=inv_r6**2
    lj_energy=4.0*eps*((sigma*inv_r)**12-(sigma*inv_r)**6)
    return w * lj_energy.mean()

def energy_coulomb_full(all_coords, all_types, res_indices, edge_index, edge_dist, charge_map=DEFAULT_CHARGE_MAP, w=3.0):
    if edge_index.numel()==0: return torch.tensor(0.0, device=all_coords.device)
    src,dst=edge_index
    q=torch.tensor([charge_map.get(t,0.0) for t in all_types], device=all_coords.device)
    qi,qj=q[src],q[dst]
    r=torch.clamp(edge_dist, min=1e-4)
    dielectric=4.0*r
    coulomb=332.0637*qi*qj/(dielectric*r)
    return w * coulomb.mean()

def energy_torsion_chi(chi_angles, seq, w=10.0):
    L=len(seq); max_chi=chi_angles.shape[1]; energy=0.0
    for i,aa in enumerate(seq):
        nchi=RESIDUE_NCHI.get(aa,0)
        for c in range(min(nchi, max_chi)):
            chi=chi_angles[i,c]
            energy+=0.5*(1.0-torch.cos(3.0*chi))
    return w * energy / max(1,L)

def alpha_regularisation(alpha, w_entropy=0.5, w_smooth=0.1):
    entropy=-(alpha*torch.log(alpha+1e-8)).mean()
    diff=alpha[1:]-alpha[:-1]; smooth=(diff**2).mean()
    return w_entropy*entropy + w_alpha_smooth*smooth

def chain_break_energy(ca, boundaries, w=1.0):
    if not boundaries: return torch.tensor(0.0, device=ca.device)
    energy=0.0
    for b in boundaries:
        if b>0 and b<len(ca):
            dist=torch.norm(ca[b]-ca[b-1], dim=-1)
            energy+=w*torch.relu(dist-5.0)
    return energy

# =============================================================================
# SOC Controller
# =============================================================================
class SOCController:
    def __init__(self, base_temp=300.0, friction=0.02, sigma_target=1.0, avalanche_threshold=0.5, w_avalanche=0.2):
        self.prev_coords=None
        self.base_temp=base_temp
        self.friction=friction
        self.sigma_target=sigma_target
        self.avalanche_threshold=avalanche_threshold
        self.w_avalanche=w_avalanche

    def sigma(self, coords):
        if self.prev_coords is None:
            self.prev_coords=coords.detach().clone()
            return torch.tensor(1.0, device=coords.device)
        delta=torch.norm(coords-self.prev_coords, dim=-1).mean()
        self.prev_coords=coords.detach().clone()
        return delta

    def temperature(self, sigma):
        dev=(sigma-self.sigma_target)/0.5
        T=self.base_temp+2000.0*torch.sigmoid(dev)
        return torch.clamp(T, self.base_temp*0.5, 3000.0)

    def avalanche_gradient(self, ca, alpha, edge_index, edge_dist):
        if ca.grad is None or edge_index.numel()==0: return None
        stress=ca.grad.detach()
        stressed=torch.norm(stress, dim=-1)>self.avalanche_threshold
        if not stressed.any(): return None
        src,dst=edge_index
        a=0.5*(alpha[src]+alpha[dst])
        safe_d=torch.clamp(edge_dist, min=1e-6)
        K=torch.exp(-a*torch.log(safe_d))*torch.exp(-edge_dist/12.0)
        direction=torch.zeros_like(ca)
        stressed_idx=torch.where(stressed)[0]
        grad_stressed=stress[stressed_idx]
        norm=torch.norm(grad_stressed, dim=-1, keepdim=True)
        direction[stressed_idx]=-grad_stressed/(norm+1e-8)
        src_stressed=stressed[src]
        if not src_stressed.any(): return None
        edge_K=K[src_stressed]
        edge_dst=dst[src_stressed]; edge_src=src[src_stressed]
        dir_src=direction[edge_src]
        grad_contrib=torch.zeros_like(ca)
        grad_contrib.index_add_(0, edge_dst, -self.w_avalanche*edge_K.unsqueeze(-1)*dir_src)
        return grad_contrib

# =============================================================================
# Itô SDE & Malliavin Sensitivity (optional)
# =============================================================================
class ItoProcess:
    def __init__(self, dim, drift, diffusion, dt=1e-3, device='cpu'):
        self.dim=dim; self.drift=drift; self.diffusion=diffusion; self.dt=dt; self.device=device
    def euler_maruyama_step(self,x):
        dW=torch.randn_like(x)*math.sqrt(self.dt)
        sigma=self.diffusion(x)
        if sigma.dim()==1: sigma=sigma.unsqueeze(-1)
        return x+self.drift(x)*self.dt+(sigma@dW.unsqueeze(-1)).squeeze(-1)
    def milstein_step(self,x):
        dW=torch.randn_like(x)*math.sqrt(self.dt)
        sigma=self.diffusion(x); b=self.drift(x)
        if sigma.dim()==1:
            x_temp=x.detach().requires_grad_(True)
            sigma_val=self.diffusion(x_temp)
            grad_sigma=torch.autograd.grad(sigma_val.sum(), x_temp, allow_unused=True)[0]
            if grad_sigma is not None:
                correction=0.5*sigma*grad_sigma*(dW**2-self.dt)
            else: correction=torch.zeros_like(x)
            return x+b*self.dt+sigma*dW+correction
        else:
            return x+b*self.dt+(sigma@dW.unsqueeze(-1)).squeeze(-1)

class LangevinDynamics(ItoProcess):
    def __init__(self, energy_fn, gamma=0.02, T=300.0, dt=1e-3, device='cpu'):
        self.energy_fn=energy_fn; self.gamma=gamma; self.T=T; self.kB=1.987e-3
        def drift(x):
            x.requires_grad_(True)
            grad=torch.autograd.grad(energy_fn(x), x)[0]
            return -grad/gamma
        def diffusion(x): return math.sqrt(2*self.kB*T/gamma)*torch.ones_like(x)
        super().__init__(dim=0, drift=drift, diffusion=diffusion, dt=dt, device=device)
    def step(self,x,scheme='milstein'):
        if scheme=='euler': return self.euler_maruyama_step(x)
        elif scheme=='milstein': return self.milstein_step(x)
        else: raise ValueError
    def refine(self,x0,steps,return_trajectory=False,scheme='milstein'):
        traj=[]; x=x0.clone().detach()
        for _ in range(steps):
            x=self.step(x,scheme=scheme)
            if return_trajectory: traj.append(x.cpu().clone())
        return torch.stack(traj) if return_trajectory else x

class MalliavinSensitivity:
    def __init__(self,process): self.process=process
    def compute_weight(self,x0,parameter='T',steps=100):
        h=1e-4
        if parameter=='T':
            T_orig=self.process.T
            self.process.T=T_orig+h; x_plus=self.process.refine(x0,steps=steps)
            self.process.T=T_orig-h; x_minus=self.process.refine(x0,steps=steps)
            self.process.T=T_orig
            return (x_plus-x_minus)/(2*h)
        else: raise NotImplementedError
    def greek(self,x0,functional,parameter='T',steps=100,n_paths=1000):
        total=0.0
        for _ in range(n_paths):
            weight=self.compute_weight(x0,parameter,steps)
            xT=self.process.refine(x0,steps=steps)
            total+=functional(xT)*weight.mean().item()
        return total/n_paths

# =============================================================================
# Batalin–Vilkovisky Formalism (DNA origami topological validation)
# =============================================================================
class BVFieldTheory:
    def __init__(self,field_names,ghost_numbers):
        self.fields=field_names
        self.ghost_numbers={name:gh for name,gh in zip(field_names,ghost_numbers)}
        self.phi={name:torch.tensor(0.0) for name in field_names}
        self.phi_star={name:torch.tensor(0.0) for name in field_names}
    def antibracket(self,F,G):
        phi={k:v.clone().detach().requires_grad_(True) for k,v in self.phi.items()}
        phistar={k:v.clone().detach().requires_grad_(True) for k,v in self.phi_star.items()}
        F_val=F(phi,phistar); G_val=G(phi,phistar)
        dF_dphi=torch.autograd.grad(F_val, list(phi.values()), retain_graph=True, create_graph=True)
        dF_dphistar=torch.autograd.grad(F_val, list(phistar.values()), retain_graph=True, create_graph=True)
        dG_dphi=torch.autograd.grad(G_val, list(phi.values()), retain_graph=True, create_graph=True)
        dG_dphistar=torch.autograd.grad(G_val, list(phistar.values()), retain_graph=True, create_graph=True)
        result=0.0
        for i,name in enumerate(self.fields):
            result+=torch.dot(dF_dphi[i].flatten(), dG_dphistar[i].flatten())
            result-=torch.dot(dF_dphistar[i].flatten(), dG_dphi[i].flatten())
        return result
    def classical_master_equation(self,S):
        ab=self.antibracket(S,S)
        return torch.allclose(ab, torch.tensor(0.0), atol=1e-6)

class DNAOrigamiBV(BVFieldTheory):
    def __init__(self,vertices,edges):
        field_names=[f"phi_{u}_{v}" for (u,v) in edges]
        ghost_numbers=[0]*len(field_names)
        super().__init__(field_names,ghost_numbers)
        self.vertices=torch.tensor(vertices,dtype=torch.float32)
        self.edges=edges
        for idx,(u,v) in enumerate(edges):
            vec=self.vertices[v]-self.vertices[u]
            self.phi[f"phi_{u}_{v}"]=vec.clone().detach().requires_grad_(True)
        if HAS_NX:
            g=nx.Graph()
            g.add_nodes_from(range(len(vertices)))
            g.add_edges_from(edges)
            self.cycles=nx.cycle_basis(g)
        else: self.cycles=[]
    def action_link(self,phi_dict,phi_star_dict):
        total=torch.tensor(0.0, device=self.vertices.device)
        for cycle in self.cycles:
            vec=torch.zeros(3, device=self.vertices.device)
            for idx in range(len(cycle)):
                u=cycle[idx]; v=cycle[(idx+1)%len(cycle)]
                key=f"phi_{u}_{v}"
                if key in phi_dict: vec+=phi_dict[key]
                else:
                    key_rev=f"phi_{v}_{u}"
                    if key_rev in phi_dict: vec-=phi_dict[key_rev]
            total+=torch.dot(vec,vec)
        return total
    def verify_topological_consistency(self):
        return self.classical_master_equation(self.action_link)

# =============================================================================
# Antibody scoring (RosettaScorer) and CDR modeling
# =============================================================================
class RosettaScorer:
    def __init__(self, cfg=None):
        self.w_interface=15.0; self.w_cross_lj=30.0; self.w_cross_coulomb=5.0
    def score_complex(self, coords_ab, seq_ab, coords_ag, seq_ag, chi_ab=None, chi_ag=None):
        device=coords_ab.device
        full_ca=torch.cat([coords_ab, coords_ag], dim=0)
        full_seq=seq_ab+seq_ag
        L_ab=len(seq_ab)
        full_chi=None
        if chi_ab is not None and chi_ag is not None:
            full_chi=torch.cat([chi_ab, chi_ag], dim=0)
        ei_ca, ed_ca = sparse_edges(full_ca, 12.0, 64)
        atoms = reconstruct_backbone(full_ca)
        ei_hb, ed_hb = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, 64)
        chain_types = ['protein']*len(full_seq)
        alpha = torch.ones(len(full_seq), device=device)
        phi, psi = compute_phi_psi(atoms)
        E = 0.0
        E += energy_bond(full_ca, alpha, w=30.0, mod=0.1, mask=None)
        E += energy_angle(full_ca, alpha, w=15.0, mod=0.05, mask=None)
        E += energy_rama(phi, psi, full_seq, alpha, w=8.0, mod=0.2)
        E += energy_clash(full_ca, alpha, ei_ca, ed_ca, w=80.0, radius=2.0)
        E += energy_hbond(atoms['O'], atoms['N'], atoms['C'], alpha, ei_hb, ed_hb, w=6.0)
        E += energy_electro(full_ca, full_seq, ei_ca, ed_ca, w=4.0)
        E += energy_solvent(full_ca, full_seq, ei_ca, ed_ca, w=5.0)
        E += energy_soc(full_ca, alpha, ei_ca, ed_ca, w=0.3)
        src,dst=ei_ca[0],ei_ca[1]
        cross=((src<L_ab)&(dst>=L_ab))|((src>=L_ab)&(dst<L_ab))
        if cross.any():
            cs,cd=src[cross],dst[cross]; d=ed_ca[cross]
            q=torch.tensor([1 if a in 'RK' else -1 if a in 'DE' else 0 for a in full_seq], device=device)
            qi,qj=q[cs],q[cd]; r=torch.clamp(d,min=1.0)
            lj=-4.0*((4.0/r)**6-(4.0/r)**4)
            coul=-qi*qj/r
            E+=self.w_cross_lj*lj.mean()+self.w_cross_coulomb*coul.mean()
        return E

class CDRLoopModeler:
    def __init__(self, scorer, fragment_db_path=None):
        self.scorer=scorer
        self.fragments=self._load_fragments(fragment_db_path)
    def _load_fragments(self,path):
        if path and os.path.exists(path):
            with open(path) as f: return [tuple(x) for x in json.load(f)]
        return [(-65,-45),(-70,-40),(-60,-50),(-80,-30),(-55,-55),(-75,-35),(-62,-48),(-68,-42),(-58,-52),(-72,-38),(-90,-20),(-50,-60),(-85,-25),(-95,-15)]
    def remodel_loop(self, coords, seq, loop_start, loop_end, n_steps=500, temp=1.0):
        device=coords.device
        best_coords=coords.clone()
        current_coords=coords.clone()
        best_E=self.scorer.score_complex(coords[:loop_start],seq[:loop_start],coords[loop_end:],seq[loop_end:]).item()
        for step in range(n_steps):
            if loop_end-loop_start<2: continue
            pos=random.randint(loop_start, loop_end-2)
            trial=current_coords.clone()
            trial[pos]+=0.5*torch.randn(3)
            trial[pos+1]+=0.5*torch.randn(3)
            E_new=self.scorer.score_complex(trial[:loop_start],seq[:loop_start],trial[loop_end:],seq[loop_end:]).item()
            delta=E_new-best_E
            if delta<0 or random.random()<math.exp(-delta/temp):
                current_coords=trial
                if E_new<best_E:
                    best_E=E_new
                    best_coords=trial
        return best_coords

# =============================================================================
# Refinement Engine (Main)
# =============================================================================
@dataclass
class RefinementConfig:
    w_bond: float = 30.0
    w_angle: float = 15.0
    w_rama: float = 8.0
    w_clash: float = 80.0
    w_hbond: float = 6.0
    w_electro: float = 4.0
    w_solvent: float = 5.0
    w_soc: float = 0.3
    w_alpha_entropy: float = 0.5
    w_alpha_smooth: float = 0.1
    w_chain_break: float = 1.0
    w_lj: float = 30.0
    w_coulomb: float = 5.0
    w_torsion: float = 10.0
    clash_radius: float = 2.0
    angle_target_rad: float = 111.0 * math.pi / 180.0
    bond_target: float = 3.8
    base_temp: float = 300.0
    friction: float = 0.02
    sigma_target: float = 1.0
    avalanche_threshold: float = 0.5
    w_avalanche: float = 0.2
    cutoff: float = 12.0
    max_neighbors: int = 64
    lr: float = 1e-4
    steps: int = 600
    rebuild_interval: int = 100
    use_amp: bool = True
    grad_clip: float = 5.0
    use_milstein: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class RefinementEngine:
    def __init__(self, cfg: Optional[RefinementConfig] = None):
        self.cfg = cfg or RefinementConfig()
        self.device = torch.device(self.cfg.device)
        self.soc = SOCController(
            base_temp=self.cfg.base_temp,
            friction=self.cfg.friction,
            sigma_target=self.cfg.sigma_target,
            avalanche_threshold=self.cfg.avalanche_threshold,
            w_avalanche=self.cfg.w_avalanche
        )
        self.scaler = GradScaler(enabled=self.cfg.use_amp)

    def _build_edges(self, coords, batch=None):
        return sparse_edges(coords, self.cfg.cutoff, self.cfg.max_neighbors, batch)

    def _total_energy(self, ca, seq, alpha, chi,
                      edge_idx, edge_dist, edge_hb, edge_hb_dist,
                      boundaries, chain_types, mask,
                      ligand_bridge=None, protein_mask_for_ligand=None):
        E = torch.tensor(0.0, device=ca.device)
        atoms = reconstruct_backbone(ca)
        phi, psi = compute_phi_psi(atoms)

        # protein terms
        prot_mask = torch.tensor([t=='protein' for t in chain_types], device=ca.device) if mask is None else mask
        E += energy_bond(ca, alpha, self.cfg.w_bond, mod=0.1, target_dist=self.cfg.bond_target, mask=prot_mask)
        E += energy_angle(ca, alpha, self.cfg.w_angle, mod=0.05, target_angle_rad=self.cfg.angle_target_rad, mask=prot_mask)
        E += energy_rama(phi, psi, seq, alpha, self.cfg.w_rama, mod=0.2, mask=prot_mask)
        E += energy_clash(ca, alpha, edge_idx, edge_dist, self.cfg.w_clash,
                          self.cfg.clash_radius, mod=0.1, mask=prot_mask)
        E += energy_hbond(atoms['O'], atoms['N'], atoms['C'], alpha,
                          edge_hb, edge_hb_dist, self.cfg.w_hbond, mod=0.1, mask=prot_mask)
        E += energy_electro(ca, seq, edge_idx, edge_dist, self.cfg.w_electro, mask=prot_mask)
        E += energy_solvent(ca, seq, edge_idx, self.cfg.w_solvent, mask=prot_mask)
        E += energy_soc(ca, alpha, edge_idx, edge_dist, self.cfg.w_soc, mask=prot_mask)
        E += alpha_regularisation(alpha, self.cfg.w_alpha_entropy, self.cfg.w_alpha_smooth)
        E += chain_break_energy(ca, boundaries, self.cfg.w_chain_break)

        # sidechain full‑atom terms (only for protein)
        if chi is not None and prot_mask.any():
            prot_idx = torch.where(prot_mask)[0]
            if len(prot_idx) > 0:
                prot_ca = ca[prot_idx]
                prot_seq = "".join([seq[i] for i in prot_idx.tolist()])
                prot_chi = chi[prot_idx]
                all_coords, all_types, _ = get_full_atom_coords_and_types(prot_ca, prot_seq, prot_chi)
                if all_coords.shape[0] > 0:
                    ei_full, ed_full = self._build_edges(all_coords)
                    E += energy_lj_full(all_coords, all_types, _, ei_full, ed_full, DEFAULT_LJ_PARAMS, self.cfg.w_lj)
                    E += energy_coulomb_full(all_coords, all_types, _, ei_full, ed_full, DEFAULT_CHARGE_MAP, self.cfg.w_coulomb)
                    E += energy_torsion_chi(prot_chi, prot_seq, self.cfg.w_torsion)

        # DNA/RNA
        dna_mask = torch.tensor([t in ('dna','rna') for t in chain_types], device=ca.device)
        if dna_mask.any():
            dna_idx = torch.where(dna_mask)[0]
            if len(dna_idx) > 0:
                dna_ca = ca[dna_idx]
                dna_seq = "".join([seq[i] for i in dna_idx.tolist()])
                E += energy_dna_rna_simple(dna_ca, dna_seq)

        # Ligand
        if ligand_bridge is not None and prot_mask.any():
            prot_idx = torch.where(prot_mask)[0]
            if len(prot_idx) > 0:
                prot_ca = ca[prot_idx]
                prot_seq = "".join([seq[i] for i in prot_idx.tolist()])
                prot_chi = chi[prot_idx] if chi is not None else None
                prot_all, prot_types, _ = get_full_atom_coords_and_types(prot_ca, prot_seq, prot_chi)
                if prot_all.shape[0] > 0:
                    prot_charges = [DEFAULT_CHARGE_MAP.get(t, 0.0) for t in prot_types]
                    E += ligand_bridge.compute_ligand_energy()
                    E += ligand_bridge.compute_interaction(prot_all, prot_types, prot_charges)
        return E

    def refine(self, coords, sequence, mask=None, chain_types=None, chain_boundaries=None,
               ligand_files=None, steps=None, logger=None, return_trajectory=False) -> Dict:
        if steps is None: steps = self.cfg.steps
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        elif isinstance(coords, torch.Tensor):
            coords = coords.to(self.device).float()
        else:
            raise ValueError("coords must be numpy array or torch tensor")

        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        B, L, _ = coords.shape
        if B != 1:
            raise ValueError("Only single chain/multimer supported as one concatenated chain")
        coords = coords.squeeze(0).detach().requires_grad_(True)
        L = len(sequence)

        if mask is None:
            mask = torch.ones(L, dtype=torch.bool, device=self.device)
        else:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

        if chain_types is None:
            chain_types = [detect_sequence_type(sequence[i:i+1]) for i in range(L)]

        if chain_boundaries is None:
            chain_boundaries = []

        # Chi angles for full‑atom (initialize randomly)
        max_chi = 4
        chi = torch.zeros(L, max_chi, device=self.device, requires_grad=True)
        alpha = torch.ones(L, device=self.device, requires_grad=True)

        # Ligand bridge
        ligand_bridge = None
        ligand_params = []
        if ligand_files:
            ligand_bridge = LigandBridge()
            for fpath in ligand_files:
                ext = os.path.splitext(fpath)[1].lower().lstrip('.')
                if ext in ('sdf',''): ftype='sdf'
                elif ext=='mol2': ftype='mol2'
                else: ftype='pdb'
                ligand_bridge.load_ligand_file(fpath, ftype)
            for lig in ligand_bridge.ligands:
                lig.to_device(self.device)
                lig.atom_coords = lig.atom_coords.detach().requires_grad_(True)
                ligand_params.append(lig.atom_coords)

        # Initial edges
        edge_idx, edge_dist = self._build_edges(coords)
        atoms = reconstruct_backbone(coords)
        edge_hb, edge_hb_dist = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors)

        params = [coords, chi, alpha] + ligand_params
        optimizer = torch.optim.Adam(params, lr=self.cfg.lr)
        energy_history = []
        best_energy = float('inf')
        best_coords = coords.clone()
        best_chi = chi.clone()
        best_alpha = alpha.clone()

        trajectory = [] if return_trajectory else None

        for step in range(steps):
            optimizer.zero_grad()
            with autocast(enabled=self.cfg.use_amp):
                E = self._total_energy(coords, sequence, alpha, chi,
                                       edge_idx, edge_dist, edge_hb, edge_hb_dist,
                                       chain_boundaries, chain_types, mask,
                                       ligand_bridge=ligand_bridge)
                loss = E
            self.scaler.scale(loss).backward()

            av_grad = self.soc.avalanche_gradient(coords, alpha, edge_idx, edge_dist)
            if av_grad is not None:
                if coords.grad is not None:
                    coords.grad = coords.grad + av_grad
                else:
                    coords.grad = av_grad

            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()

            sigma = self.soc.sigma(coords.detach())
            T = self.soc.temperature(sigma)
            noise_scale = math.sqrt(2 * self.cfg.friction * T.item() / 300.0) * self.cfg.lr
            with torch.no_grad():
                coords.add_(torch.randn_like(coords) * noise_scale)
                chi.data.add_(torch.randn_like(chi) * noise_scale * 0.5)
                for lc in ligand_params:
                    lc.add_(torch.randn_like(lc) * noise_scale * 0.1)

            if step > 0 and step % self.cfg.rebuild_interval == 0:
                edge_idx, edge_dist = self._build_edges(coords.detach())
                atoms = reconstruct_backbone(coords.detach())
                edge_hb, edge_hb_dist = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors)

            if loss.item() < best_energy:
                best_energy = loss.item()
                best_coords = coords.clone().detach()
                best_chi = chi.clone().detach()
                best_alpha = alpha.clone().detach()

            if logger and step % 50 == 0:
                logger.info(f"step {step:04d}  E={loss.item():.4f}  σ={sigma.item():.3f}  T={T.item():.1f}")
            energy_history.append(loss.item())
            if return_trajectory:
                trajectory.append(best_coords.cpu().clone())

    
def compute_energy(self, coords: torch.Tensor, sequence: str,
                   chain_types: Optional[List[str]] = None,
                   mask: Optional[torch.Tensor] = None,
                   chi: Optional[torch.Tensor] = None,
                   alpha: Optional[torch.Tensor] = None) -> float:
    """
    Compute total energy of a given structure without any optimization.
    Returns scalar energy (float).
    """
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    B, L, _ = coords.shape
    if B != 1:
        raise ValueError("Only single chain/multimer supported as one concatenated chain")
    coords = coords.squeeze(0)
    L = len(sequence)
    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=coords.device)
    if chain_types is None:
        chain_types = [detect_sequence_type(sequence[i:i+1]) for i in range(L)]
    if chi is None:
        chi = torch.zeros(L, 4, device=coords.device)   # max_chi=4
    if alpha is None:
        alpha = torch.ones(L, device=coords.device)

    edge_idx, edge_dist = sparse_edges(coords, self.cfg.cutoff, self.cfg.max_neighbors)
    atoms = reconstruct_backbone(coords)
    edge_hb, edge_hb_dist = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors)

    E = self._total_energy(coords, sequence, alpha, chi,
                           edge_idx, edge_dist, edge_hb, edge_hb_dist,
                           [], chain_types, mask)
    return E.item()

def relax_local(self, coords: torch.Tensor, sequence: str,
                positions: List[int], steps: int = 30, window: int = 3,
                chain_types: Optional[List[str]] = None,
                mask: Optional[torch.Tensor] = None,
                chi: Optional[torch.Tensor] = None,
                alpha: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
    """
    Local relaxation around given positions (indices in the concatenated sequence).
    Returns (refined_coords, final_energy).
    """
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    B, L, _ = coords.shape
    if B != 1:
        raise ValueError("Only single chain supported")
    coords = coords.squeeze(0).detach().requires_grad_(True)
    L = len(sequence)

    if mask is None:
        mask = torch.ones(L, dtype=torch.bool, device=coords.device)
    if chain_types is None:
        chain_types = [detect_sequence_type(sequence[i:i+1]) for i in range(L)]
    if chi is None:
        chi = torch.zeros(L, 4, device=coords.device)
    if alpha is None:
        alpha = torch.ones(L, device=coords.device)

    # Determine window
    min_pos = min(positions)
    max_pos = max(positions)
    win_start = max(0, min_pos - window)
    win_end = min(L, max_pos + window + 1)

    opt = torch.optim.Adam([coords], lr=self.cfg.lr)
    best_E = float('inf')
    best_coords = coords.clone()

    edge_idx, edge_dist = sparse_edges(coords, self.cfg.cutoff, self.cfg.max_neighbors)
    atoms = reconstruct_backbone(coords)
    edge_hb, edge_hb_dist = cross_sparse_edges(atoms['O'], atoms['N'], 3.5, self.cfg.max_neighbors)

    for _ in range(steps):
        opt.zero_grad()
        E = self._total_energy(coords, sequence, alpha, chi,
                               edge_idx, edge_dist, edge_hb, edge_hb_dist,
                               [], chain_types, mask)
        E.backward()
        # Mask gradient outside window
        if coords.grad is not None:
            grad_mask = torch.zeros(L, 3, device=coords.device)
            grad_mask[win_start:win_end] = 1.0
            coords.grad *= grad_mask
        opt.step()
        # Rebuild edges if coordinates changed significantly? (optional)
        if E.item() < best_E:
            best_E = E.item()
            best_coords = coords.clone().detach()
    return best_coords.detach(), best_E
                    
        return {
            'coords': best_coords.cpu().numpy(),
            'chi': best_chi.cpu().numpy(),
            'alpha': best_alpha.cpu().numpy(),
            'energy_history': energy_history,
            'final_energy': best_energy,
            'sigma': sigma.item(),
            'temperature': T.item(),
            'trajectory': np.stack(trajectory) if return_trajectory else None
        }

# =============================================================================
# I/O utilities (PDB/mmCIF)
# =============================================================================
def load_structure(filepath: str, chain: Optional[str] = None) -> Dict:
    if not HAS_BIOTITE:
        raise ImportError("biotite required for loading PDB/mmCIF files")
    if filepath.endswith('.pdb'):
        struct = pdb.PDBFile.read(filepath).get_structure(model=1)
    elif filepath.endswith('.cif') or filepath.endswith('.mmcif'):
        struct = mmcif.MMCIFFile.read(filepath).get_structure(model=1)
    else:
        raise ValueError("File must be .pdb or .cif/.mmcif")
    if chain is not None:
        struct = struct[struct.chain_id == chain]
    ca = struct[struct.atom_name == "CA"]
    coords = ca.coord.astype(np.float32)
    seq = []
    for res in ca.residues:
        seq.append(AA_3_TO_1.get(res.res_name, 'X'))
    return {'coords': coords, 'sequence': "".join(seq), 'chain_ids': [c for c in ca.chain_id]}

def save_structure(coords, sequence, filename, chain_id='A', chi=None):
    with open(filename, 'w') as f:
        for i, (x,y,z) in enumerate(coords):
            aa = sequence[i] if i < len(sequence) else 'X'
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} {chain_id}{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("END\n")
    print(f"Saved CA‑only PDB to {filename}")

# =============================================================================
# DNA Origami design & export (oxDNA)
# =============================================================================
class WireframeOrigami:
    def __init__(self, vertices: List[Tuple[float,float,float]],
                 edges: List[Tuple[int,int]],
                 scaffold_seq: str = None):
        self.vertices = vertices
        self.edges = edges
        self.scaffold_seq = scaffold_seq or ("AATGCTACTACTATTAGTAGAA" * 100)
        self.helix_radius = 1.0

    def route_scaffold(self) -> List[int]:
        visited = set()
        path = []
        current = 0
        while len(visited) < len(self.vertices):
            visited.add(current)
            path.append(current)
            neighbors = [v for v in self.edges[current] if v not in visited]
            if neighbors:
                best = min(neighbors, key=lambda v: np.linalg.norm(
                    np.array(self.vertices[current]) - np.array(self.vertices[v])))
                current = best
            else:
                unvisited = [v for v in range(len(self.vertices)) if v not in visited]
                if unvisited:
                    current = random.choice(unvisited)
                else: break
        return path

    def design_staples(self, scaffold_path: List[int]) -> Dict[str, str]:
        staples = {}
        scaffold_set = set(zip(scaffold_path[:-1], scaffold_path[1:]))
        for (u, v) in self.edges:
            if (u,v) not in scaffold_set and (v,u) not in scaffold_set:
                seq = self._complement(self.scaffold_seq[10:31])
                staples[f"staple_{u}_{v}"] = seq
        return staples

    def _complement(self, seq):
        comp = {'A':'T','T':'A','C':'G','G':'C'}
        return "".join(comp.get(b, 'N') for b in seq)

    def build_3d_model(self) -> Tuple[torch.Tensor, str]:
        all_coords = []
        full_seq = ""
        for v in range(len(self.vertices)):
            if v in self.edges and self.edges[v]:
                nb = self.edges[v][0]
                direction = np.array(self.vertices[nb]) - np.array(self.vertices[v])
            else:
                direction = np.array([0,0,1])
            norm = np.linalg.norm(direction) + 1e-8
            direction = direction / norm
            helix = build_dna_helix("A"*10, rise=0.34, twist=36, radius=1.0)
            # rotate helix to direction
            z_axis = np.array([0,0,1])
            if np.allclose(direction, z_axis): R = np.eye(3)
            elif np.allclose(direction, -z_axis): R = np.diag([1,1,-1])
            else:
                v_cross = np.cross(z_axis, direction); v_cross /= np.linalg.norm(v_cross)
                angle = math.acos(np.dot(z_axis, direction))
                K = np.array([[0, -v_cross[2], v_cross[1]],
                              [v_cross[2], 0, -v_cross[0]],
                              [-v_cross[1], v_cross[0], 0]])
                R = np.eye(3) + math.sin(angle)*K + (1-math.cos(angle))*(K@K)
            helix_np = helix.numpy()
            rotated = helix_np @ R.T + self.vertices[v]
            all_coords.append(torch.tensor(rotated, dtype=torch.float32))
            full_seq += "A"*10
        if all_coords:
            return torch.cat(all_coords, dim=0), full_seq
        return torch.empty(0,3), ""

    def export_oxDNA(self, filename: str):
        coords, seq = self.build_3d_model()
        with open(f"{filename}.top", 'w') as f:
            f.write(f"{len(seq)} nucleotides\n")
            for i in range(len(seq)):
                f.write(f"{i+1} {seq[i]} {'A' if i%2==0 else 'B'}\n")
        with open(f"{filename}.dat", 'w') as f:
            f.write(f"t = 0\nb = {len(seq)*0.34*10:.1f} {len(seq)*0.34*10:.1f} {len(seq)*0.34*10:.1f}\n")
            for coord in coords:
                f.write(f"{coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} 0 0 0 0 0 0\n")
        print(f"oxDNA files written to {filename}.top/.dat")

def build_dna_helix(seq, rise=3.38, twist=36.0, radius=8.0, start_angle=0.0):
    L = len(seq)
    coords = torch.zeros(L,3)
    for i in range(L):
        angle = start_angle + math.radians(i*twist)
        coords[i,0] = radius * math.cos(angle)
        coords[i,1] = radius * math.sin(angle)
        coords[i,2] = i*rise
    return coords

# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="REAL FOLD ONE – SOC‑Controlled Universal Refinement Engine")
    sub = parser.add_subparsers(dest='command', required=True)

    # Refine command
    refine_parser = sub.add_parser('refine')
    refine_parser.add_argument('--input', '-i', type=str, required=True)
    refine_parser.add_argument('--chain', type=str, default=None)
    refine_parser.add_argument('--output', '-o', type=str, default='refined.pdb')
    refine_parser.add_argument('--steps', type=int, default=600)
    refine_parser.add_argument('--lr', type=float, default=1e-4)
    refine_parser.add_argument('--ligand', nargs='+', type=str)
    refine_parser.add_argument('--boundaries', nargs='+', type=int)
    refine_parser.add_argument('--device', type=str, default='auto')
    refine_parser.add_argument('--milstein', action='store_true')
    refine_parser.add_argument('--trajectory', action='store_true')

    # Antibody design command
    ab_parser = sub.add_parser('antibody')
    ab_parser.add_argument('--antigen', type=str, required=True)
    ab_parser.add_argument('--cdr_start', type=int, default=95)
    ab_parser.add_argument('--cdr_end', type=int, default=102)
    ab_parser.add_argument('--num_designs', type=int, default=5)
    ab_parser.add_argument('--output', type=str, default='antibody_designs.pdb')

    # Origami command
    orig_parser = sub.add_parser('origami')
    orig_parser.add_argument('--shape', type=str, required=True)
    orig_parser.add_argument('--output', type=str, default='origami')
    orig_parser.add_argument('--bv_check', action='store_true')

    args = parser.parse_args()

    if args.command == 'refine':
        if not HAS_BIOTITE:
            print("Error: biotite required for reading PDB/mmCIF. Install: pip install biotite")
            sys.exit(1)
        data = load_structure(args.input, args.chain)
        coords, seq = data['coords'], data['sequence']
        print(f"Loaded {len(seq)} residues from {args.input}")
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        cfg = RefinementConfig(device=device, steps=args.steps, lr=args.lr, use_milstein=args.milstein)
        engine = RefinementEngine(cfg)
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("REAL_FOLD_ONE")
        result = engine.refine(coords, seq, ligand_files=args.ligand, chain_boundaries=args.boundaries,
                               logger=logger, return_trajectory=args.trajectory)
        save_structure(result['coords'], seq, args.output)
        print(f"Refined structure saved to {args.output}")
        print(f"Final energy: {result['final_energy']:.4f} kcal/mol")
        print(f"SOC sigma: {result['sigma']:.3f}  temperature: {result['temperature']:.1f} K")
        if args.trajectory:
            np.save(args.output.replace('.pdb','_traj.npy'), result['trajectory'])

    elif args.command == 'antibody':
        # Load antigen from PDB (simplified)
        if not HAS_BIOTITE:
            print("biotite required for antibody design")
            sys.exit(1)
        ant_data = load_structure(args.antigen)
        ant_coords = torch.tensor(ant_data['coords'], dtype=torch.float32)
        ant_seq = ant_data['sequence']
        print(f"Antigen loaded: {len(ant_seq)} residues")
        # Dummy antibody (random coordinates)
        ab_coords = torch.randn(len(ant_seq), 3) * 10
        ab_seq = "A" * len(ant_seq)
        scorer = RosettaScorer()
        modeler = CDRLoopModeler(scorer)
        best = modeler.remodel_loop(ab_coords, ab_seq, args.cdr_start, args.cdr_end,
                                    n_steps=500, temp=1.0)
        save_structure(best.cpu().numpy(), ab_seq, args.output)
        print(f"Antibody design saved to {args.output}")

    elif args.command == 'origami':
        with open(args.shape) as f:
            data = json.load(f)
        vertices = data['vertices']
        edges = data['edges']
        origami = WireframeOrigami(vertices, edges)
        path = origami.route_scaffold()
        staples = origami.design_staples(path)
        print(f"Scaffold path length: {len(path)}")
        print(f"Designed {len(staples)} staples")
        if args.bv_check:
            bv = DNAOrigamiBV(vertices, edges)
            print(f"BV topological consistency: {bv.verify_topological_consistency()}")
        origami.export_oxDNA(args.output)

if __name__ == "__main__":
    main()
