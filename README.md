# AetherCell: A Generative Engine for Virtual Cell Perturbation and Drug Discovery

> Repository accompanying *AetherCell: A generative engine for virtual cell perturbation and in vivo drug discovery*.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Weights-HuggingFace-yellow.svg)](https://huggingface.co/liwenyuan99/AetherCell)
[![Agent-Ready](https://img.shields.io/badge/Agent--Ready-Skills%20Included-00b8d9.svg)](https://huggingface.co/liwenyuan99/AetherCell)
[![Web Demo](https://img.shields.io/badge/Web%20Demo-Drug%20Screening-success.svg)](http://101.32.8.25/)

**Resources**

| Resource | Link |
|----------|------|
| Preprint | [bioRxiv 10.64898/2026.03.13.710968](https://www.biorxiv.org/content/10.64898/2026.03.13.710968v1) |
| Model weights | [Hugging Face — liwenyuan99/AetherCell](https://huggingface.co/liwenyuan99/AetherCell) |
| Processed datasets | [Zenodo 10.5281/zenodo.18295255](https://zenodo.org/records/18295255) |
| Web demo | [http://101.32.8.25/](http://101.32.8.25/) |

![AetherCell Framework](AetherCell-framework.png)

---

## Overview

AetherCell is a generative modelling framework that aligns context-rich bulk RNA-seq data with perturbation-dense L1000 data in a shared latent space. The framework supports three core research tasks:

1. **Virtual perturbation modelling** — predict transcriptomic changes from drug treatment, gene knockdown (shRNA), overexpression (OE), or knockout (CRISPR/Cas9).
2. **Drug response prediction** — estimate drug sensitivity (IC50 / AUC) on cancer cell lines.
3. **Drug repurposing** — rank FDA-approved drug candidates for a given disease via a Mixture-of-Experts model fusing transcriptomic similarity and knowledge-graph reasoning.

Beyond a static model release, AetherCell is packaged as an **Agent-Ready** engine: it ships with executable Skills that can be invoked via natural language through [Claude Code](https://claude.ai/claude-code), making the full prediction pipeline accessible without writing code.

For architecture details, training objectives, and benchmark results, please refer to the manuscript.

---

## Repository structure

```
AetherCell/
├── README.md                              # This file
├── environment.yml                        # Conda environment specification
├── LICENSE                                # AetherCell Research License v1.0
├── aethercell-drug-discovery-v1.0.0/      # Inference package (download from Hugging Face)
│   ├── models/
│   │   ├── transcriptome_prediction/      #   Perturbation predictors + MolFormer + VAE weights
│   │   ├── ic50_prediction/               #   Drug response prediction model
│   │   └── moe_repurposing/               #   Drug repurposing MoE model + disease/drug data
│   └── .claude/skills/                    #   Skill definitions for Claude Code agent mode
└── src/                                   # Training-scale inference scripts (advanced)
    ├── INFERENCE_README.md                #   Documentation for src scripts
    ├── inference_delta_z.py               #   Batch delta-z generation (drug)
    ├── inference_perturbed_expression.py  #   Batch expression prediction (drug)
    ├── inference_knockdown_perturbed.py   #   Batch expression prediction (shRNA)
    ├── aethercell_delta_z_inference.ipynb #   Interactive notebook (all modes)
    └── ...                                #   Model definitions and data loaders
```

---

## Prerequisites

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate aethercell
```

### 2. Download model weights from Hugging Face

The inference package `aethercell-drug-discovery-v1.0.0` (~4.3 GB) contains all
pre-trained weights required by both the Python API and the Claude Code agent mode.

**Download from:** [https://huggingface.co/liwenyuan99/AetherCell](https://huggingface.co/liwenyuan99/AetherCell)

Place the extracted `aethercell-drug-discovery-v1.0.0/` directory at the repository root so
that the file tree matches the structure shown above.

After extraction, verify that the following weight files are present:

| Component | Path | Size |
|-----------|------|------|
| LINCS VAE | `models/transcriptome_prediction/L1000_vae.pt` | 14 MB |
| RNA VAE | `models/transcriptome_prediction/RNA_vae.pt` | 611 MB |
| Perturbation predictors (8 total) | `models/transcriptome_prediction/predictor_{L,R}_{drug,sh,oe,xpr}.pt` | 135–299 MB each |
| MolFormer | `models/transcriptome_prediction/molformer/` | ~200 MB |
| IC50 predictor | `models/ic50_prediction/ddp_predictor.pt` | 280 MB |
| MoE repurposing model | `models/moe_repurposing/standalone_expert_model.pt` | 394 MB |
| MoE static data | `models/moe_repurposing/data_sub/static_data.h5` | ~1 GB |

> **Note:** Only the weight package from Hugging Face is needed for the Python API
> and Agent modes below.  The processed training datasets on Zenodo are required
> only for the batch inference scripts in `src/` (see [Advanced usage](#advanced-usage-batch-inference-scripts)).

---

## Access modes

AetherCell provides four ways to use the models, ordered by ease of use.

### Mode 1 — Web demo

The fastest way to test the drug-screening pipeline:

**[http://101.32.8.25/](http://101.32.8.25/)**

The demo runs AetherCell's screening pipeline with an LLM-generated interpretive
report.  Daily API quota is limited.

---

### Mode 2 — Agent Skills via Claude Code

For interactive, natural-language-driven analysis without writing code.

```bash
cd aethercell-drug-discovery-v1.0.0
claude                          # launch Claude Code
```

Example prompts:

```
Predict the transcriptomic effect of Aspirin and perform pathway enrichment.
SMILES: CC(=O)Oc1ccccc1C(=O)O
```

```
Find the top 20 drug repurposing candidates for Alzheimer disease.
```

```
Predict IC50 of this compound on A549 cells: CC(C)Cc1ccc(cc1)C(C)C(O)=O
```

Three built-in Skills are provided:
- **Transcriptome Prediction** — drug/gene perturbation to DEGs and pathway enrichment
- **IC50 Prediction** — drug sensitivity on cancer cell lines
- **Drug Repurposing** — disease-centric candidate ranking

Each Skill automates model loading, inference, and result formatting.

---

### Mode 3 — Python API

For programmatic use, batch experiments, and integration into custom pipelines.
All examples below assume the working directory is `aethercell-drug-discovery-v1.0.0/`.

#### Transcriptome prediction

```python
from models.transcriptome_prediction.transcriptome_inference import TranscriptomePredictor

predictor = TranscriptomePredictor(
    model_type='l1000',        # 'l1000' (978 genes) or 'bulk_rnaseq' (10085 genes)
    perturbation='drug',       # 'drug', 'shrna', 'oe', or 'xpr'
    device='cpu'
)

result = predictor.predict(
    drug_smiles='CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
    cell_line='MCF7'
)

# result['expression']  — predicted expression array
# result['delta']       — fold changes vs. control
# result['top_genes']   — top 20 differentially expressed genes
```

#### Drug response prediction

```python
from models.ic50_prediction.ic50_inference import IC50Predictor

predictor = IC50Predictor(device='cpu')
result = predictor.predict(drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(O)=O', cell_line='A549')

print(result['prediction'])   # 'sensitive' or 'resistant'
print(result['probability'])  # sensitivity probability (0–1)
```

#### Drug repurposing

```python
from models.moe_repurposing.moe_inference import MoEPredictor

predictor = MoEPredictor()
results = predictor.predict_for_disease('Alzheimer disease', top_n=10)

print(results[['drugbank_id', 'moe_score', 'te_score', 'kg_score']])
```

#### Supported perturbation types

| Perturbation | `perturbation` parameter | Input |
|-------------|-------------------------|-------|
| Drug treatment | `'drug'` | SMILES string |
| Gene knockdown (shRNA) | `'shrna'` | Gene symbol or ENSG ID |
| Gene overexpression | `'oe'` | Gene symbol or ENSG ID |
| Gene knockout (CRISPR) | `'xpr'` | Gene symbol or ENSG ID |

---

### Advanced usage: batch inference scripts

The `src/` directory provides training-scale inference scripts intended for
researchers who need to generate predictions over large custom datasets (e.g.,
all drugs × all cell lines).  These scripts require the original processed
datasets from Zenodo in addition to the model weights.

**Processed datasets:** [https://zenodo.org/records/18295255](https://zenodo.org/records/18295255)

See [`src/INFERENCE_README.md`](src/INFERENCE_README.md) for complete setup
instructions, required file lists, and metadata format specifications.

---

## Reproducibility resources

### Data

| Dataset | Source | Scale |
|---------|--------|-------|
| Bulk RNA-seq pre-training | TCGA, CCLE, GEO | 519,609 samples |
| L1000 perturbation data | CMap / LINCS | ~1.3 M standardised pairs |

Processed datasets are archived at [Zenodo](https://zenodo.org/records/18295255).

### Model assets

Pre-trained weights are hosted at [Hugging Face](https://huggingface.co/liwenyuan99/AetherCell).

| Model asset | Description |
|-------------|-------------|
| Perturbation predictors | Drug / shRNA / OE / XPR, in both L1000 and Bulk RNA-seq output modes |
| AC-RP | Drug response prediction |
| PK-MoE | Disease-centric drug repurposing |

---

## Limitations

- Prediction accuracy may vary across perturbation classes, cell lines, and biological contexts.
- Gene perturbation tasks depend on the availability and quality of gene-level embeddings.
- Drug repurposing outputs are hypothesis-generating and require downstream experimental validation.
- The web demo is subject to limited daily LLM API quota and may experience service interruptions.

---

## Responsible use

> **FOR RESEARCH USE ONLY.**
> This repository and its associated model assets are intended for non-commercial
> academic research.  They are **not** validated for clinical use, diagnosis,
> patient stratification, or treatment decision-making.  Any biological or
> therapeutic hypothesis generated by the system should be independently evaluated
> and experimentally validated.

See [LICENSE](LICENSE) for the full AetherCell Research License v1.0.

---

## Citation

If you use AetherCell in your research, please cite:

```bibtex
@article{li2026aethercell,
  title   = {AetherCell: A Generative Engine for Virtual Cell Perturbation and In Vivo Drug Discovery},
  author  = {Li, Wenyuan and Chen, Yang and Peng, Zhaoyi and Xiang, Lei and Wang, Dong and Xie, Zhi},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.03.13.710968},
  url     = {https://www.biorxiv.org/content/10.64898/2026.03.13.710968v1}
}
```

Use of this repository, model weights, outputs, or derivative models in any
publication, preprint, report, benchmark, presentation, or public release
requires citation of the above preprint in accordance with the license terms.
