# AetherCell：A generative engine for virtual cell perturbation and in vivo drug discovery
Official implementation of "AetherCell: A generative engine for virtual cell perturbation and in vivo drug discovery"
AetherCell is a deep generative foundation model that unifies context-rich clinical RNA-seq with perturbation-dense L1000 assays into a shared, platform-aligned transcriptomic manifold. By implementing a Specificity-Driven Learning Framework, AetherCell effectively suppresses non-specific stress responses (Type II Failure) to recover low-frequency, mechanism-specific biological signals.
![Image text](https://github.com/4Ueyez0nly/AetherCell/blob/main/AetherCell-framework.png)  
Fig. 1 AetherCell’s architecture and application. a Construction of the unified transcriptomic manifold. b Integration of multi-modal foundation models. c Versatile downstream applications.

### Setup and dependencies
environments.yml contains environment of this project.
```
Bash
conda env create -f environments.yml
conda activate aethercell
```
### Core Architecture & Training
AetherCell follows a hierarchical Satellite-Backbone architecture.
1.Backbone (Part 1): A deep $\beta$-VAE trained on 519,609 whole-transcriptome samples using 8-block deep residual stacks to resolve non-linear dependencies in a 256-dimensional latent bottleneck.Satellite (Part 2): A specialized interface that anchors L1000 measurements (978 landmark genes) into the global RNA-seq manifold via Probabilistic Manifold Anchoring.
[cite_start]AetherCell employs a hierarchical three-step strategy to bridge technical platforms and resolve mechanistic signatures[cite: 330, 420]:

### **Step 1: Global Manifold Construction (Backbone VAE)**
[cite_start]A deep Backbone Variational Autoencoder ($\beta$-VAE) is trained on a comprehensive corpus of 519,609 RNA-seq samples[cite: 332, 543]. [cite_start]It utilizes 8-block deep residual stacks to resolve complex non-linear dependencies, compressing the whole transcriptome into a 256-dimensional latent biological state space[cite: 331, 334, 335].

### **Step 2: Platform Interface Anchoring (Satellite VAE)**
[cite_start]A specialized "Satellite" VAE is designed for the L1000 platform (978 landmark genes)[cite: 338, 340]. [cite_start]Using a **Probabilistic Manifold Anchoring** strategy, the L1000 interface is anchored into the pre-established global RNA-seq manifold, correcting for platform-specific biases while preserving intrinsic biological variance[cite: 169, 171, 387].

### **Step 3: Specificity-Driven Generative Modeling**
[cite_start]The conditional generative module is trained to predict the latent transition vector ($\Delta z$) induced by unseen perturbations[cite: 175, 403]. [cite_start]By employing a multi-objective framework with a **Latent Specificity Loss ($\mathcal{L}_{spec}$)**, the model explicitly filters out generic stress centroids and prioritizes mechanism-specific driver signals[cite: 196, 444, 448].

---
### Inference and Virtual Experiments
We provide demos to demonstrate the cross-platform generalization and "virtual laboratory" capabilities of AetherCell.
1. Latent Shift Generator ($\Delta z$)Generate the mechanism-specific latent transition vector induced by a perturbation.
```
python latent_gen.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --cell_line "PANC-1"
```
2. Virtual Perturbation Simulation (Transcriptomic Prediction)
Predict changes from chemical or genetic inputs.
```
# Predict effect of an compound
python demo_perturb.py --mode chemical --input "drug_smiles.txt"
# Predict effect of gene knockdown
python demo_perturb.py --mode genetic --target "BRCA1" --type "knockdown"
```
4. Downstream Task: Drug Sensitivity (AC-RP)A inference script for predicting IC50 by integrating $h_{context}$ and $\Delta z_{pred}$.Bashpython drp_inference.py --cell_line "MCF7" --drug "Paclitaxel"
```
python drp_inference.py --cell_line "MCF7"
```
### 🌐 AetherCell-DR Web Platform (Under Development)
We are building a user-friendly web-based portal for AetherCell-DR (Drug Repurposing). This platform will host our Phenotype-Knowledge Mixture of Experts (PK-MoE) system, allowing users to:
Perform in silico drug-disease association screening.
Visualize mechanistically grounded reasoning chains.
The web portal link will be released upon official publication.
