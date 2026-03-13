# AetherCell: A Generative Engine for Virtual Cell Perturbation and Drug Discovery

> **Repository accompanying a manuscript under peer review.** AetherCell is a generative modelling framework for virtual cell perturbation, drug response prediction, and drug repurposing from transcriptomic data.

[![License](https://img.shields.io/badge/License-MIT%20(Research%20Only)-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Weights-HuggingFace-yellow.svg)](https://huggingface.co/liwenyuan99/AetherCell)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Integrated-purple.svg)](https://claude.ai/claude-code)
[![Web Demo](https://img.shields.io/badge/Web%20Demo-Drug%20Screening-success.svg)](http://101.32.8.25/)

**Manuscript title:** *AetherCell: A generative engine for virtual cell perturbation and in vivo drug discovery*

**Resources**
- **Model weights:** [Hugging Face — liwenyuan99/AetherCell](https://huggingface.co/liwenyuan99/AetherCell)
- **Interactive workflow:** [Claude Code](https://claude.ai/claude-code)
- **Web demo:** [http://101.32.8.25/](http://101.32.8.25/)
- **Processed datasets:** [Zenodo](https://zenodo.org/records/18295255)

![AetherCell Framework](https://github.com/4Ueyez0nly/AetherCell/blob/main/AetherCell-framework.png)

---

## Overview

AetherCell is designed to support three related tasks from transcriptomic and molecular inputs:

1. **Virtual perturbation modelling** for drug and gene perturbations
2. **Drug response prediction** for cancer cell lines
3. **Drug repurposing** for disease-oriented candidate ranking

The framework aligns context-rich bulk RNA-seq data with perturbation-dense L1000 data in a shared latent space, enabling perturbation modelling across data regimes and downstream screening workflows.

This repository is intended as a research-facing entry point rather than a full methodological exposition. For architecture, training objectives, benchmark design, and quantitative results, please refer to the associated manuscript.

---

## Why this repository is structured this way

For a project positioned as a serious biomedical research contribution, the repository should help readers do three things quickly:

- understand the scientific scope
- reproduce the main access pathways
- evaluate the practical usability of the system
AetherCell demonstrates a development pattern we believe is especially useful in biomedicine: pairing a domain model with a task-oriented interaction layer. In this repository, that means the model is accessible through both a programmable API and a natural-language workflow, making it easier for non-specialist users to test core capabilities without rewriting infrastructure.

---

## Access modes

### 1. Web demo

The fastest way to test the core drug-screening workflow is the online demo:

**[http://101.32.8.25/](http://101.32.8.25/)**

The demo exposes AetherCell’s screening pipeline and uses an LLM to generate an end-to-end report with interpretable supporting rationale.

> **Note:** daily LLM API quota on the website is limited.

### 2. Python API

For local use, batch experiments, and integration into custom research pipelines, AetherCell can be called through Python.

> **Important:** using the Python API requires downloading the model files from **Hugging Face** first.

### 3. Claude Code workflow

AetherCell can also be used through a natural-language workflow in [Claude Code](https://claude.ai/claude-code), which is useful for rapid task execution, interactive exploration, and report-oriented usage.

> **Important:** Claude Code mode also requires downloading the model files from **Hugging Face** first.

---

## Core research tasks

| Task | Input | Output |
|------|------|--------|
| Virtual perturbation | Drug SMILES + cell line, or gene perturbation target | Predicted transcriptomic response, fold changes, top DEGs |
| Drug response prediction | Drug SMILES + cancer cell line | Sensitivity / resistance estimate |
| Drug repurposing | Disease name | Ranked candidate drugs |

**Supported perturbation types**
- Drug treatment (`drug`)
- Gene knockdown via shRNA (`sh`)
- Gene overexpression (`oe`)
- Gene knockout via CRISPR (`xpr`)

---

## Quick start

### Environment setup

```bash
conda env create -f environments.yml
conda activate aethercell
```

or

```bash
python -m venv aethercell-env
source aethercell-env/bin/activate   # Linux/macOS
# aethercell-env\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Download model weights

Before using either the **Python API** or **Claude Code workflow**, download the required weights from:

**[https://huggingface.co/liwenyuan99/AetherCell](https://huggingface.co/liwenyuan99/AetherCell)**

---

## Minimal usage examples

### Transcriptome prediction

```python
from models.transcriptome_prediction.transcriptome_inference import TranscriptomePredictor

predictor = TranscriptomePredictor(
    model_type='l1000',
    perturbation='drug',
    device='cpu'
)

result = predictor.predict(
    drug_smiles='CC(=O)Oc1ccccc1C(=O)O',
    cell_line='MCF7'
)
```

### Drug response prediction

```python
from models.ic50_prediction.ic50_inference import IC50Predictor

predictor = IC50Predictor(device='cpu')
result = predictor.predict(drug_smiles='...', cell_line='A549')
```

### Drug repurposing

```python
from models.moe_repurposing.moe_inference import MoEPredictor

predictor = MoEPredictor()
results = predictor.predict_for_disease('Alzheimer disease', top_n=10)
```

### Claude Code

```bash
npm install -g @anthropic-ai/claude-code
cd aethercell-drug-discovery-v1.0.0
claude
```

Example prompts:
- *What genes does aspirin affect in MCF7 cells?*
- *Is doxorubicin effective against A549 lung cancer cells?*
- *Find drugs that could potentially treat Alzheimer's disease.*

---

## Reproducibility resources

### Data

| Dataset | Source | Scale |
|---------|--------|-------|
| Bulk RNA-seq pre-training | TCGA, CCLE, GEO | 519,609 samples |
| L1000 perturbation data | CMap LINCS project | ~1.3M standardized pairs |

Processed datasets: [Zenodo](https://zenodo.org/records/18295255)

### Model assets

Pre-trained weights are available at [Hugging Face](https://huggingface.co/liwenyuan99/AetherCell).

| Model asset | Description |
|-------------|-------------|
| Perturbation predictors | Drug / sh / oe / xpr perturbation modules |
| AC-RP | Drug response prediction |
| PK-MoE | Drug repurposing system |

---

## Limitations

- Performance may vary across perturbation classes, cell lines, and biological contexts.
- Gene perturbation tasks rely on the availability and quality of perturbation-specific representations.
- Drug repurposing outputs are hypothesis-generating and require downstream experimental validation.
- The web demo depends on limited daily LLM API quota and is not guaranteed to provide uninterrupted service.

---

## Responsible use

> **FOR RESEARCH USE ONLY**
>
> This repository and its associated model assets are intended for research use. They are **not** validated for clinical use, diagnosis, patient stratification, or treatment decision-making. Any biological or therapeutic hypothesis generated by the system should be independently evaluated and experimentally validated.

---

## Citation

This work is currently under peer review. Please cite the associated manuscript or repository as appropriate.

```bibtex
@software{aethercell2026,
  title   = {AetherCell: A generative engine for virtual cell perturbation and in vivo drug discovery},
  author  = {[Authors will be added upon publication]},
  year    = {2026},
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
