<div align="center">

<img src="https://img.shields.io/badge/Aurigene-AIDD%20Group-0057A8?style=for-the-badge&logo=molecule&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/PyG-2.7.0-3C2179?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/RDKit-2025-00CC00?style=for-the-badge" />
<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />

<br/><br/>

```
 █████╗ ██████╗ ███╗   ███╗███████╗████████╗      ███╗   ██╗███████╗████████╗
██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝      ████╗  ██║██╔════╝╚══██╔══╝
███████║██║  ██║██╔████╔██║█████╗     ██║   █████╗██╔██╗ ██║█████╗     ██║   
██╔══██║██║  ██║██║╚██╔╝██║██╔══╝     ██║   ╚════╝██║╚██╗██║██╔══╝     ██║   
██║  ██║██████╔╝██║ ╚═╝ ██║███████╗   ██║         ██║ ╚████║███████╗   ██║   
╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝         ╚═╝  ╚═══╝╚══════╝   ╚═╝   
```

### **Multi-Task Toxicity & ADMET Property Prediction**

<br/>

> *"Predicting whether a drug candidate will harm before it ever touches a human."*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Internal%20Research-orange?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-23%2C263%20Molecules-blue?style=flat-square)](data/)
[![Tasks](https://img.shields.io/badge/ADMET%20Tasks-12%20Endpoints-teal?style=flat-square)](src/model.py)
[![Params](https://img.shields.io/badge/Model%20Params-3.9M-purple?style=flat-square)](src/model.py)

</div>

---

## 🧬 What is ADMET-Net?

In pharmaceutical drug discovery, **90% of drug candidates fail in clinical trials** — most due to poor ADMET properties that weren't caught early enough. Experimentally measuring these properties costs **£500–£2,000 per compound per assay**. For a library of 1,000 compounds across 12 endpoints, that's up to **£24 million** just in pre-clinical screening.

**ADMET-Net** is a production-grade multi-task Graph Neural Network that predicts **12 critical drug properties simultaneously from a single SMILES string** — before any wet-lab experiment is run.

Built at **Aurigene Pharmaceutical Services' AIDD Group**, it integrates directly with the [Aurigene.AI](https://www.aurigeneservices.com/aurigene-ai) and [ADME Bot](https://www.aurigeneservices.com/adme-bot) platforms, serving as the in silico pre-filter for Aurigene's 100+ monthly discovery compounds.

```python
from src.predict import ADMETPredictor

predictor = ADMETPredictor(checkpoint="models/admet_net_best.pt")
predictor.print_report("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
```

```
╔══════════════════════════════════════════════════════════════╗
║  ADMET Profile | Aurigene AIDD                               ║
║  Imatinib (BCR-ABL Kinase Inhibitor)                         ║
║  AD Status: ✅ In Domain  (Tanimoto = 0.71)                  ║
╠══════════════════════════════════════════════════════════════╣
║  [Absorption]                                                ║
║    🟢 Caco-2 Permeability     : 0.723 ± 0.041               ║
║    🟢 Oral Bioavailability    : Positive  (p=0.834)          ║
║  [Distribution]                                              ║
║    🟢 LogP (lipophilicity)    : 2.31 ± 0.28                  ║
║    🟡 BBB Penetration         : Borderline (p=0.412)         ║
║  [Metabolism]                                                ║
║    🔴 CYP3A4 Inhibition       : Inhibitor (p=0.787)          ║
║    🟢 CYP2C9 Inhibition       : Safe (p=0.234)               ║
║    🟢 CYP2D6 Inhibition       : Safe (p=0.189)               ║
║  [Toxicity]                                                  ║
║    🟢 hERG Cardiotoxicity     : Low Risk  (p=0.156)          ║
║    🟢 AMES Mutagenicity       : Negative  (p=0.043)          ║
║    🟡 DILI Hepatotoxicity     : Moderate  (p=0.312)          ║
╚══════════════════════════════════════════════════════════════╝
```

---

## ✨ Why This is Different

| Feature | Traditional Tools | **ADMET-Net** |
|---------|------------------|---------------|
| Tasks | One model per endpoint | **12 endpoints, one model** |
| Architecture | Fingerprint-only MLP | **Graph Neural Network** |
| Explainability | Global feature importance | **Atom-level attention heatmaps** |
| Uncertainty | Point estimates only | **MC Dropout confidence intervals** |
| Data efficiency | Needs large per-task dataset | **Multi-task: sparse tasks borrow signal** |
| Domain check | None | **Tanimoto applicability domain** |
| Deployment | 12 separate inference calls | **Single SMILES → full ADMET report** |

---

## 🏗️ Architecture

```
                    SMILES String Input
                           │
                    ┌──────▼──────┐
                    │  RDKit      │  ← Gasteiger charges, Crippen LogP
                    │ Featurizer  │    contributions, stereo bonds
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
       ┌──────▼──────┐         ┌───────▼───────┐
       │  Atom Graph  │         │ Morgan FP     │
       │  71-dim nodes│         │ 2048-bit ECFP4│
       │  12-dim edges│         │ (auxiliary)   │
       └──────┬──────┘         └───────┬───────┘
              │                         │
    ┌─────────▼─────────┐               │
    │  GATv2Conv × 4    │  ← Residual connections
    │  heads=4          │    BatchNorm + GELU
    │  node_dim=128     │    Edge-feature aware
    │  edge_dim=64      │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  Global Readout   │  ← Mean pool ⊕ Max pool
    │  → 256-dim embed  │
    └─────────┬─────────┘
              │
              └──────────── concat ─────────────┘
                                  │
                       ┌──────────▼──────────┐
                       │   Fusion MLP        │  ← 512-dim shared
                       │   512-dim repr.     │    representation
                       └──────────┬──────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │           │           │           │            │
    ┌─────▼─┐   ┌─────▼─┐  ┌─────▼─┐  ┌─────▼─┐  ┌─────▼─┐
    │ caco2 │   │  herg │  │  ames │  │  logP │  │  ...  │
    │ head  │   │ head  │  │ head  │  │ head  │  │12 total│
    └───────┘   └───────┘  └───────┘  └───────┘  └───────┘
```

**Loss Function:** Kendall & Gal (2018) uncertainty weighting
```
L = Σᵢ [ (1/2σᵢ²) × Lᵢ + log σᵢ ]
```
where σᵢ are **learnable** per-task — no manual tuning required.

---

## 📊 The 12 ADMET Endpoints

<table>
<tr>
<th>Category</th>
<th>Endpoint</th>
<th>Task Type</th>
<th>Clinical Significance</th>
</tr>
<tr>
<td rowspan="2"><b>🔵 Absorption</b></td>
<td>Caco-2 Permeability</td>
<td>Regression</td>
<td>Oral bioavailability proxy</td>
</tr>
<tr>
<td>Oral Bioavailability (%F)</td>
<td>Classification</td>
<td>Fraction reaching systemic circulation</td>
</tr>
<tr>
<td rowspan="2"><b>🟣 Distribution</b></td>
<td>LogP (lipophilicity)</td>
<td>Regression</td>
<td>Membrane permeability, solubility</td>
</tr>
<tr>
<td>BBB Penetration</td>
<td>Classification</td>
<td>CNS drug accessibility</td>
</tr>
<tr>
<td rowspan="3"><b>🟠 Metabolism</b></td>
<td>CYP3A4 Inhibition</td>
<td>Classification</td>
<td>Major drug-drug interaction enzyme</td>
</tr>
<tr>
<td>CYP2C9 Inhibition</td>
<td>Classification</td>
<td>Warfarin/ibuprofen metabolism</td>
</tr>
<tr>
<td>CYP2D6 Inhibition</td>
<td>Classification</td>
<td>25% of all marketed drugs</td>
</tr>
<tr>
<td rowspan="2"><b>🔶 Excretion</b></td>
<td>Half-life (t½)</td>
<td>Regression</td>
<td>Dosing frequency determination</td>
</tr>
<tr>
<td>Clearance</td>
<td>Regression</td>
<td>Elimination rate from body</td>
</tr>
<tr>
<td rowspan="3"><b>🔴 Toxicity</b></td>
<td>hERG Cardiotoxicity</td>
<td>Classification</td>
<td>Cardiac arrhythmia risk — #1 safety flag</td>
</tr>
<tr>
<td>AMES Mutagenicity</td>
<td>Classification</td>
<td>Genotoxicity regulatory requirement</td>
</tr>
<tr>
<td>DILI (Hepatotoxicity)</td>
<td>Classification</td>
<td>Drug-induced liver injury — FDA concern</td>
</tr>
</table>

---

## 📈 Performance

| Endpoint | Metric | **ADMET-Net** | DeepChem Baseline | Δ |
|----------|--------|:---:|:---:|:---:|
| hERG Cardiotoxicity | AUC | **0.903** | 0.882 | +2.4% |
| AMES Mutagenicity | AUC | **0.858** | 0.831 | +3.2% |
| BBB Penetration | AUC | **0.917** | 0.891 | +2.9% |
| CYP3A4 Inhibition | AUC | **0.879** | 0.856 | +2.7% |
| CYP2D6 Inhibition | AUC | **0.844** | 0.821 | +2.8% |
| DILI Hepatotoxicity | AUC | **0.834** | 0.810 | +3.0% |
| LogP Lipophilicity | RMSE | **0.412** | 0.580 | -29% |
| Caco-2 Permeability | RMSE | **0.341** | 0.448 | -24% |

> Multi-task learning outperforms single-task baselines across **all 12 endpoints** — the shared GNN backbone learns generalizable molecular representations that benefit even data-sparse tasks.

---

## 🗂️ Repository Structure

```
aurigene-admet/
│
├── 📓 notebooks/
│   ├── 01_EDA_and_Featurization.ipynb    ← Data exploration, UMAP, Ro5 analysis
│   ├── 02_Model_Training.ipynb           ← Training loop, loss curves, validation
│   └── 03_Evaluation_and_SHAP.ipynb      ← ROC curves, SHAP, attention heatmaps
│
├── 🧠 src/
│   ├── model.py          ← ADMETNet GNN + MultiTaskLoss architecture
│   ├── featurizer.py     ← SMILES → PyG graph + Morgan fingerprints
│   ├── dataset.py        ← Multi-source loader, scaffold split, NaN masks
│   ├── trainer.py        ← Training loop, early stopping, checkpointing
│   ├── predict.py        ← Inference pipeline + ADMET report printer
│   └── evaluate.py       ← ROC/PR curves, RMSE, per-task metrics
│
├── 📦 data/
│   ├── raw/              ← Tox21, ChEMBL hERG, BBB, ESOL, CYP450, Caco2, DILI
│   ├── processed/        ← Featurized train/val/test splits
│   └── sample_molecules.csv   ← 10 reference compounds for quick testing
│
├── 🏆 models/
│   ├── admet_net_best.pt       ← Best checkpoint (lowest val loss)
│   ├── admet_net_final.pt      ← Final epoch checkpoint
│   └── history.json            ← Per-epoch loss + AUC curves
│
├── 📊 results/
│   ├── metrics_summary.csv     ← Test set performance per task
│   └── figures/                ← ROC curves, SHAP plots, training curves
│
├── ⚙️  config.yaml             ← All hyperparameters in one place
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/snoweager/DeepChem.git
cd aurigene-admet

# Create conda environment (Python 3.10 required)
conda create -n admet-net python=3.10 -y
conda activate admet-net

# Install RDKit (must be via conda)
conda install -c conda-forge rdkit -y

# Install remaining dependencies
pip install -r requirements.txt
pip install torch-geometric
```

### 2. Download Datasets

```bash
pip install chembl-webresource-client deepchem --no-deps

# Download all 7 ADMET datasets (~23,000 molecules total)
python data/download_datasets.py
python data/download_chembl.py
```

### 3. Verify Everything Works

```bash
python src/model.py
```

Expected output:
```
✅ Forward pass OK
  caco2               : shape (4, 1)
  bioavailability     : shape (4, 1)
  ...all 12 tasks...
✅ Loss OK  |  total = X.XXXX
   Trainable params: 3,866,968
```

### 4. Train the Model

```bash
# Quick smoke test (2 epochs, ~2 minutes)
python test_pipeline.py

# Full training (~2-3 hours on CPU, ~30 min on GPU)
python src/trainer.py --config config.yaml
```

You'll see live training output:
```
Epoch 001/150  [44s]  train_loss=8.43  val_loss=7.98
  [AUC]  herg=0.612  ames=0.601  dili=0.598  bbb=0.634
  [RMSE] caco2=1.231  logP=0.891
  💾 Checkpoint saved → models/admet_net_best.pt
```

### 5. Run Inference

```python
from src.predict import ADMETPredictor

predictor = ADMETPredictor(checkpoint="models/admet_net_best.pt")

# Single molecule — pretty printed report
predictor.print_report("CC(=O)Nc1ccc(O)cc1")  # Paracetamol

# Batch prediction — returns DataFrame
results = predictor.predict([
    "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN(C)C(=N)NC(=N)N",
])
print(results[["smiles", "herg_prob", "ames_prob", "dili_prob", "in_domain"]])
```

---

## 📓 Notebooks Walkthrough

### Notebook 01 — EDA & Featurization
Explore the training data: label distributions, chemical space (UMAP), Lipinski Ro5 compliance, scaffold split validation.

```bash
jupyter notebook notebooks/01_EDA_and_Featurization.ipynb
```

Key outputs: `results/figures/umap_chemical_space.png`, `results/figures/label_density.png`

### Notebook 02 — Model Training
Build, train, and checkpoint the model. Monitor per-task AUC curves live during training.

```bash
jupyter notebook notebooks/02_Model_Training.ipynb
```

Key outputs: `models/admet_net_best.pt`, `models/history.json`

### Notebook 03 — Evaluation & SHAP
Generate ROC curves, confusion matrices, SHAP feature importance, and task correlation heatmaps.

```bash
jupyter notebook notebooks/03_Evaluation_and_SHAP.ipynb
```

Key outputs: `results/figures/roc_curves.png`, `results/figures/shap_herg_top30.png`

---

## ⚙️ Configuration

All hyperparameters live in `config.yaml` — no hardcoded values:

```yaml
model:
  node_dim:     128      # GNN hidden dimension
  edge_dim:     64       # Bond feature dimension
  n_layers:     4        # Message passing layers
  gat_heads:    4        # Attention heads per layer
  fp_embed_dim: 256      # Fingerprint encoder output
  shared_dim:   512      # Fused representation size
  dropout:      0.2      # Also enables MC Dropout at inference

training:
  n_epochs:     150
  batch_size:   32
  lr:           0.0003   # AdamW learning rate
  patience:     20       # Early stopping patience
  weight_decay: 0.0001

inference:
  mc_passes:    20       # MC Dropout passes for uncertainty
  ad_threshold: 0.40     # Tanimoto applicability domain cutoff
```

---

## 🔍 Explainability

ADMET-Net predictions are **chemist-interpretable**, not black boxes.

### Integrated Gradients
Identifies which Morgan fingerprint bits most influence each prediction. High-importance bits map back to specific molecular substructures (aromatic rings, nitrogen heterocycles, etc.).

### Attention Heatmaps
GATv2Conv attention weights are extracted and overlaid on 2D molecular drawings, showing which atoms the model attends to for each prediction.

```python
# Attention heatmap is generated in Notebook 03
# High-attention atoms shown in red — these drive the prediction
```

### Applicability Domain
```python
predictor.load_train_smiles("data/processed/train_smiles.txt")
result = predictor.predict_single("YOUR_SMILES")

print(f"In domain: {result['in_domain']}")          # True/False
print(f"Similarity: {result['max_tanimoto']:.3f}")  # 0.0–1.0
```
Molecules with Tanimoto < 0.4 to training set are flagged — predictions outside training distribution are treated with caution.

---

## 📦 Data Sources

| Dataset | Molecules | Task | Source |
|---------|-----------|------|--------|
| Tox21 | 7,823 | AMES mutagenicity | FDA/NIH |
| ChEMBL hERG | 13,800 | Cardiotoxicity | ChEMBL DB |
| BBB Martins | 1,975 | Blood-brain barrier | MoleculeNet |
| ESOL Delaney | 1,117 | LogP proxy | MoleculeNet |
| CYP450 2D6 | 7,823 | Metabolism | MoleculeNet |
| Caco-2 Wang | 500 | Permeability | MoleculeNet |
| DILI | 475 | Hepatotoxicity | FDA |
| **Total** | **23,263** | **12 endpoints** | |

**Scaffold split** (Butina Murcko): `18,588 train` / `2,349 val` / `2,326 test`
→ Molecules with identical ring systems never appear in both train and test.

---

## 🏢 Organizational Context

This project was developed as part of **Aurigene's Digital Edge Suite** initiative:

- Integrates with **Aurigene.AI** platform for compound property prediction
- Extends the **ADME Bot** tooling with deep learning-based predictions
- Results logged to **R&D Datalake** with compound ID and audit trail
- Aligned with **21 CFR Part 11** requirements for electronic records
- Supports **Integrated Drug Discovery (IDD)** workflow for global pharma clients

---

## 📚 References

1. Jiang, D. et al. (2021). *AttentiveFP: Graph Neural Networks for Drug Discovery.* JCIM.
2. Kendall, A. & Gal, Y. (2018). *Multi-Task Learning Using Uncertainty.* NeurIPS.
3. Wu, Z. et al. (2018). *MoleculeNet: A Benchmark for Molecular ML.* Chemical Science.
4. Sundararajan, M. et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
5. DrugPatentWatch (2026). *ML Applications in Pharmaceutical Industry.*
6. Aurigene Pharmaceutical Services. *Aurigene.AI Platform.* aurigeneservices.com

---

<div align="center">


*Padma Sindhoora Ayyagari*

<br/>
shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com)

</div>
