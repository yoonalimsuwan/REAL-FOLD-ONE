# Quantum-Chemistry-Grade Training Guide

**Real data. Real architectures. Real benchmarks. Honest scope.**

---

## 1. What "chemical accuracy" actually means

| Claim | Requires |
|---|---|
| "1 kcal/mol vs QM9" | Match B3LYP/6-31G(2df,p) labels on held-out QM9 |
| "1 kcal/mol vs experiment" | Match experimental atomization energies (much harder) |
| "CCSD(T) accuracy" | Train on CCSD(T) labels (ANI-1ccx, not QM9) |

**QM9 labels are DFT, not reality.** Beating QM9 MAE means reproducing that
functional, not solving chemistry. State which you mean, every time.

---

## 2. QM9 — the real dataset

- **133,885** molecules, ≤9 heavy atoms (C,N,O,F), relaxed geometries
- **3,054** "uncharacterized" (geometry check failed) → exclude → **130,831**
- Method: **B3LYP/6-31G(2df,p)**
- **NO FORCES.** Energies at minima only.

### Standard split (must match to compare)
```
train 110,000 | val 10,000 | test ~10,831
~~~
Different splits ⇒ **not comparable**. Report your seed.

### Targets and units
| Target | Raw unit | Report as |
|---|---|---|
| u0, u298, h298, g298, zpve, homo, lumo, gap | Hartree | eV or meV |
| mu | Debye | Debye |
| alpha | Bohr³ | Bohr³ |
| cv | cal/mol/K | cal/mol/K |

---

## 3. Published baselines — the numbers to beat

QM9, 110k train, MAE:

| Target | SchNet [A1] | PhysNet [A3] | PaiNN [A2] |
|---|---|---|---|
| U0 | 14 meV | 8 meV | **5.9 meV** |
| H298 | 14 meV | — | **6.0 meV** |
| G298 | 14 meV | — | **5.5 meV** |
| gap | 63 meV | — | **45.7 meV** |
| μ | 0.033 D | — | **0.012 D** |
| α | 0.235 a₀³ | — | **0.045 a₀³** |

5.9 meV ≈ 0.14 kcal/mol. **That is the bar.** Reproducing it is real work.

---

## 4. Running

```bash
pip install numpy torch

# QM9 (downloads ~80 MB compressed)
python qm9_schnet_painn.py --arch painn  --target u0 --epochs 300
python qm9_schnet_painn.py --arch schnet --target u0 --epochs 300
python qm9_schnet_painn.py --arch painn  --target gap --epochs 300

# quick smoke (subset)
python qm9_schnet_painn.py --limit 5000 --epochs 20 --n-train 4000 --n-val 500

# lattice gauge: VALIDATE FIRST
python lattice_heatbath_rk3.py validate
python lattice_heatbath_rk3.py generate --L 8 --beta 6.0 --algorithm heatbath
```

**Always run `validate` before trusting any lattice result.**

---

## 5. Forces — QM9 cannot do this

| Dataset | E | F | Level | Use |
|---|---|---|---|---|
| QM9 | ✅ | ❌ | B3LYP | energies only |
| MD17 / rMD17 [V1] | ✅ | ✅ | PBE+vdW / CCSD | E+F, small molecules |
| ANI-1x | ✅ | ✅ | ωB97X/6-31G(d) | E+F, 5M conformations |
| ANI-1ccx | ✅ | ❌ | CCSD(T)*/CBS | transfer learning |

```bash
# rMD17
wget https://figshare.com/ndownloader/files/23950376 -O rmd17.tar.bz2
```
Train with `loss = w_E·MSE(E) + w_F·MSE(F)`, typically `w_F ≫ w_E`.

---

## 6. Real data registry — other domains

### ✅ Has real public data

| Domain | Dataset | Access | Label |
|---|---|---|---|
| Protein structure | PDB | rcsb.org, open | 3D coords |
| Protein–ligand | PDBbind | registration | Kd/Ki |
| Folding benchmark | CASP15/16 | predictioncenter.org | blind GDT_TS |
| Protein design | ProteinMPNN sets | github | sequence recovery |
| Cancer genomics | TCGA | portal.gdc.cancer.gov | mutation, expression, survival |
| Drug response | GDSC / CCLE | cancerrxgene.org | IC50 |
| Dependency | DepMap | depmap.org | CRISPR essentiality |
| Variants | ClinVar | ncbi.nlm.nih.gov | pathogenicity |
| CRISPR on-target | Doench 2016 | Microsoft supplement | efficiency |
| CRISPR off-target | GUIDE-seq, CIRCLE-seq | published supplements | off-target counts |
| Base editing | DeepHF, BE-Hive | github | edit outcome |
| Pathogen genomes | Nextstrain / GISAID | registration | sequence + phylogeny |
| Epi curves | WHO / JHU | open | case counts |
| Methylation age | GEO (Horvath) | ncbi.nlm.nih.gov/geo | CpG β + age |
| Aging genes | HAGR, CellAge | genomics.senescence.info | curated |
| EEG | TUH EEG Corpus | isip.piconepress.com | clinical labels |
| BCI | BCI Competition, PhysioNet | open | task labels |

### ⚠️ Partial / no validated label

- **Rejuvenation outcome** — methylation age is measurable; "reversal of aging"
  as a training target does not exist as validated ground truth.
- **Organ-level design** — no dataset of designed organs with function labels.

### ❌ No data exists

- **"Quantum brain-wave footprint in the universe"** — no observable, no
  measurement, no dataset. A loader cannot be written for something unmeasured.

### 🚫 Not provided

- **ICBM / hypersonic interception** — weapons targeting. Out of scope.

---

## 7. Ethics and governance

| Data | Requirement |
|---|---|
| TCGA controlled access | dbGaP authorization, DUA |
| GISAID | registration + attribution terms |
| Human EEG / clinical | IRB / ethics approval |
| Patient genomes | consent, HIPAA/GDPR |
| CRISPR design | institutional biosafety review |

Any clinical output is a **research prediction**, not a diagnosis or therapy.

---

## 8. Validation checklist

Before claiming anything:

- [ ] Held-out test set never seen during training or model selection
- [ ] Split seed and sizes reported
- [ ] Compared to published baseline **on the same split**
- [ ] Confidence intervals reported (subsampled, not naive SEM)
- [ ] Ablation: does the new component actually help?
- [ ] Independent reproduction by someone else
- [ ] For lattice: gauge invariance + heatbath/Metropolis agreement
- [ ] For chemistry: state the reference level of theory

---

## 9. References

~~~
[D1] Ramakrishnan et al. Sci. Data 1, 140022 (2014).
[D2] Ruddigkeit et al. J. Chem. Inf. Model. 52, 2864 (2012).
[A1] Schütt et al. NeurIPS 30 (2017). SchNet.
[A2] Schütt, Unke, Gastegger. ICML (2021). PaiNN.
[A3] Unke & Meuwly. JCTC 15, 3678 (2019). PhysNet.
[V1] Christensen & von Lilienfeld. MLST 1, 045018 (2020). rMD17.
[LG1] Wilson. Phys. Rev. D 10, 2445 (1974).
[LG3] Cabibbo & Marinari. Phys. Lett. B 119, 387 (1982).
[LG4] Kennedy & Pendleton. Phys. Lett. B 156, 393 (1985).
[LG5] Lüscher. JHEP 08 (2010) 071. arXiv:1006.4518.
[LG6] Gattringer & Lang. Quantum Chromodynamics on the Lattice. Springer (2010).
[ST1] Politis & Romano. Ann. Statist. 22, 2031 (1994).
```
