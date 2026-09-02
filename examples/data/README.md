# Included downstream reproduction inputs

These CSV files are small model-output tables copied from the study's server-side analysis directories on 2026-09-02. They are included so reviewers can execute metric and ranking code without downloading multi-gigabyte checkpoints.

| File | Rows | Role |
|---|---:|---|
| `ac_rp_predictions.csv` | 90,379 | AC-RP continuous response predictions and targets |
| `synergy_predictions.csv` | 2,622 | drug-pair synergy scores and labels |
| `cdx_predictions.csv` | 819 | before/after gene-perturbation IC50 predictions |
| `tcga_predictions.csv` | 3,276 | TCGA drug-response scores and labels |
| `ac_dr_predictions.csv` | 2,103 | AC-DR MoE/TE/KG candidate scores |

These files are outputs, not training data. Full processed training inputs are obtained from the checksum-pinned Zenodo downloader. End-to-end model inference uses the checksum-pinned Hugging Face model package.
