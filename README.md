# AetherCell: A Generative Engine for Virtual Cell Perturbation and Drug Discovery

> **Deep generative foundation model for transcriptomic perturbation prediction with natural language interface**

[![License](https://img.shields.io/badge/License-MIT%20(Research%20Only)-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Weights-HuggingFace-yellow.svg)](https://huggingface.co/liwenyuan99/AetherCell)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Integrated-purple.svg)](https://claude.ai/claude-code)

Official implementation of *"AetherCell: A generative engine for virtual cell perturbation and in vivo drug discovery"*

AetherCell is a deep generative foundation model that unifies context-rich clinical RNA-seq with perturbation-dense L1000 assays into a shared, platform-aligned transcriptomic manifold. By implementing a **Specificity-Driven Learning Framework**, AetherCell suppresses non-specific stress responses (Type II Failure) to recover low-frequency, mechanism-specific biological signals.

![AetherCell Framework](https://github.com/4Ueyez0nly/AetherCell/blob/main/AetherCell-framework.png)

*Fig. 1 — AetherCell's architecture and application. (a) Construction of the unified transcriptomic manifold. (b) Integration of multi-modal foundation models. (c) Versatile downstream applications.*

---

## Core Capabilities

### 1. Transcriptome Prediction — Drug & Gene Perturbation Effects

Predict how drugs or gene modifications affect cellular gene expression across platforms.

| Input | Output |
|-------|--------|
| Drug SMILES + Cell line | 978 landmark genes (L1000) or 10,085 genes (RNA-seq) |
| Gene perturbation target | Expression fold changes + top DEGs |

**Supported perturbation types:**
- Drug treatment (`drug`)
- Gene knockdown via shRNA (`sh`)
- Gene overexpression (`oe`)
- Gene knockout via CRISPR (`xpr`)

**Models:** 8 specialized predictors + 2 VAE encoders + MoLFormer molecular encoder

---

### 2. IC50 Prediction (AC-RP) — Cancer Drug Sensitivity

Predict drug sensitivity or resistance on specific cancer cell lines.

| Input | Output |
|-------|--------|
| Drug SMILES + Cancer cell line | Sensitivity (sensitive/resistant) + probability score |

**Model:** AetherCell-Response Prediction (AC-RP), transfer-learned from the transcriptome predictor.

---

### 3. Drug Repurposing (AetherCell-DR) — Disease-Drug Matching

Discover existing drugs that could treat new diseases using a Phenotype-Knowledge Mixture of Experts (PK-MoE) system.

| Input | Output |
|-------|--------|
| Disease name | Ranked drug candidates with MOE / TE / KG scores |

**Knowledge base:** 2,576 diseases × 7,957 drugs
**Web platform:** Under development — will be released upon publication.

---

## Model Architecture

### Hierarchical Satellite-Backbone Design

**Step 1: Global Manifold Construction (Backbone VAE)**

A deep Backbone β-VAE trained on 519,609 RNA-seq samples. Uses 8-block deep residual stacks to compress the whole transcriptome into a 256-dimensional latent biological state space.

**Step 2: Platform Interface Anchoring (Satellite VAE)**

A specialized "Satellite" VAE for the L1000 platform (978 landmark genes). Uses **Probabilistic Manifold Anchoring** to align the L1000 interface into the global RNA-seq manifold, correcting platform-specific biases while preserving biological variance.

**Step 3: Specificity-Driven Generative Modeling**

Predicts the latent transition vector (Δz) induced by perturbations. A multi-objective framework with **Latent Specificity Loss (L_spec)** filters generic stress centroids and prioritizes mechanism-specific driver signals.

```
Drug SMILES → MoLFormer Encoder → Drug Embedding (256-dim)
                                          ↓
                             Cross-Attention Fusion
                                          ↓
Cell Line → Cell Embedding (128-dim) → Δz Prediction
                                          ↓
              Backbone VAE Decoder → Transcriptomic Output
```

---

## Installation

### System Requirements

- **OS:** Linux, macOS, or Windows
- **Python:** 3.8 or higher
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** 5 GB free space
- **GPU:** Optional (CPU inference supported)

### Conda Environment (Recommended)

```bash
conda env create -f environments.yml
conda activate aethercell
```

### pip Install

```bash
python -m venv aethercell-env
source aethercell-env/bin/activate   # Linux/macOS
# aethercell-env\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

## Training

### Step 1: Train Backbone VAE (RNA-seq)

```bash
python train_RNAvae.py
```

### Step 2: Train Satellite VAE (L1000)

```bash
python train_Lvae.py
```

### Step 3: Train Perturbation Generative Module

```bash
python train_aethercell_drug.py   # Drug perturbation
python train_aethercell_sh.py     # shRNA knockdown
```

### Step 4: Train Drug Sensitivity Model (AC-RP)

```bash
python train_aethercellRP.py
```

---

## Inference & Virtual Experiments

### 1. Latent Shift Generation (Δz)

Generate the mechanism-specific latent transition vector induced by a perturbation:

```
aethercell_delta_z_inference.ipynb
```

### 2. Virtual Perturbation Simulation

```bash
# Predict effect of a compound
python inference_compound_perturbed.py

# Predict effect of gene knockdown
python inference_knockdown_perturbed.py
```

### 3. Python API

**Transcriptome Prediction**

```python
from models.transcriptome_prediction.transcriptome_inference import TranscriptomePredictor

predictor = TranscriptomePredictor(
    model_type='l1000',    # or 'bulk_rnaseq'
    perturbation='drug',   # or 'sh', 'oe', 'xpr'
    device='cpu'           # or 'cuda'
)

result = predictor.predict(
    drug_smiles='CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    cell_line='MCF7'
)

print(result['top_genes'][0])   # {'gene': ..., 'fold_change': ...}
print(result['expression'])     # 978-dim vector
```

**IC50 Prediction**

```python
from models.ic50_prediction.ic50_inference import IC50Predictor

predictor = IC50Predictor(device='cpu')

result = predictor.predict(
    drug_smiles='...',
    cell_line='A549'
)

print(result['prediction'])   # 'sensitive' or 'resistant'
print(result['probability'])  # 0.0 – 1.0
print(result['confidence'])   # 'high', 'medium', 'low'
```

**Drug Repurposing**

```python
from models.moe_repurposing.moe_inference import MoEPredictor

predictor = MoEPredictor()

results = predictor.predict_for_disease(
    disease_name='Alzheimer disease',
    top_n=10
)

for _, row in results.iterrows():
    print(f"{row['drugbank_id']}: MOE={row['moe_score']:.4f}")
```

---

## Natural Language Interface (Claude Code)

AetherCell integrates with [Claude Code](https://claude.ai/claude-code), allowing direct natural language interaction with all models — no coding required.

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Launch in project directory
cd aethercell-drug-discovery-v1.0.0
claude
```

**Example queries:**
- *"What genes does aspirin affect in MCF7 cells?"*
- *"Is doxorubicin effective against A549 lung cancer cells?"*
- *"Find drugs that could potentially treat Alzheimer's disease."*

---

## Data Availability

| Dataset | Source | Size |
|---------|--------|------|
| Bulk RNA-seq pre-training | TCGA, CCLE, GEO | 519,609 samples |
| L1000 perturbation data | CMap LINCS project | ~1.3M standardized pairs |

Processed training datasets: [Zenodo](https://zenodo.org/records/18295255)

---

## Pre-trained Models

Pre-trained weights for the Backbone VAE, Satellite VAE, and generative modules are available at:

**[Hugging Face — liwenyuan99/AetherCell](https://huggingface.co/liwenyuan99/AetherCell)**

| Model | Description | Size |
|-------|-------------|------|
| Backbone VAE | Global RNA-seq manifold | ~500 MB |
| Satellite VAE | L1000 platform interface | ~200 MB |
| Perturbation predictors (×8) | Drug/sh/oe/xpr × L1000/RNA-seq | ~1.5 GB |
| MoLFormer encoder | Molecular transformer | ~300 MB |
| AC-RP (IC50) | Drug sensitivity model | ~279 MB |
| PK-MoE (DR) | Drug repurposing ensemble | ~1.2 GB |

**Total: ~4.0 GB**

---

## Supported Cell Lines (Partial)

| Cell Line | Cancer Type | Tissue |
|-----------|-------------|--------|
| MCF7 | Breast cancer | Mammary gland |
| A549 | Lung cancer | Lung |
| HepG2 | Hepatocellular carcinoma | Liver |
| HeLa | Cervical cancer | Cervix |
| K562 | Chronic myeloid leukemia | Bone marrow |
| PC3 | Prostate cancer | Prostate |
| HCT116 | Colorectal cancer | Colon |
| MDAMB231 | Triple-negative breast cancer | Mammary gland |

90+ cell lines supported in total.

---

## Limitations

1. **Gene perturbation predictions** (sh/oe/xpr): require pre-computed gene embeddings; accuracy may vary for rare targets.
2. **Cell line coverage**: best performance on common lines (MCF7, A549, HepG2); lower accuracy on unseen cell lines.
3. **Drug repurposing**: predictions require experimental validation; not all diseases have transcriptome data (KG-only mode used as fallback).

---

## Disclaimer

> **FOR RESEARCH USE ONLY**
>
> This software is NOT validated for clinical use, NOT intended for medical diagnosis or treatment decisions, and NOT approved by FDA or equivalent regulatory agencies. All predictions must be experimentally validated before any downstream application.

---

## Citation

This work is currently under review for publication. If you use AetherCell in your research, please cite:

```bibtex
@software{aethercell2025,
  title   = {AetherCell: A generative engine for virtual cell perturbation and in vivo drug discovery},
  author  = {[Authors will be added upon publication]},
  year    = {2025},
  note    = {Manuscript under review},
  url     = {https://huggingface.co/liwenyuan99/AetherCell}
}
```

---

## License

**MIT License with Research Use Restriction**

- Academic and research use: Allowed
- Commercial use: Not allowed
- Clinical applications: Not allowed
- Redistribution of model weights separately: Not allowed

See [LICENSE](LICENSE) for full terms.

---

## Acknowledgments

- **L1000 Gene Expression:** [LINCS program](https://lincsproject.org/)
- **MoLFormer:** [IBM MoLFormer molecular transformer](https://github.com/IBM/molformer)
- **DrugBank:** [Drug-disease knowledge base](https://go.drugbank.com/)
- **Claude Code:** [Anthropic AI](https://claude.ai/claude-code)
- Pre-training data: TCGA, CCLE, GEO, CMap LINCS

---

*Last updated: March 2026 | Version: 2.0*
