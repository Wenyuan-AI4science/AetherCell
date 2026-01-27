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
### Phase I: Manifold Alignment and Backbone Construction
```
python train_aethercell_phase1.py
```
### Phase II: Specificity-Driven Generative Training (incorporating L_spec)
python train_aethercell_phase2.py

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
python drp_inference.py --cell_line "MCF7" --drug "Paclitaxel"
```
### 🌐 AetherCell-DR Web Platform (Under Development)
We are building a user-friendly web-based portal for AetherCell-DR (Drug Repurposing). This platform will host our Phenotype-Knowledge Mixture of Experts (PK-MoE) system, allowing users to:
Perform in silico drug-disease association screening.
Visualize mechanistically grounded reasoning chains.
The web portal link will be released upon official publication.
