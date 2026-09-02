# Included downstream reproduction inputs

These CSV files are small model-output tables copied from the study's server-side analysis directories on 2026-09-02. They are included so reviewers can execute metric and ranking code without downloading multi-gigabyte checkpoints.

| File | Rows | Role |
|---|---:|---|
| `ac_rp_predictions.csv` | 90,379 | AC-RP continuous response predictions and targets |
| `synergy_predictions.csv` | 2,622 | drug-pair synergy scores and labels |
| `cdx_predictions.csv` | 819 | before/after gene-perturbation IC50 predictions |
| `tcga_predictions.csv` | 3,276 | TCGA drug-response scores and labels |
| `ac_dr_predictions.csv` | 2,103 | AC-DR MoE/TE/KG candidate scores |
| `api_context_examples.npz` | 3 samples | real A549 RNA (10085), matched L1000 control (978), and token arrays for deterministic API smoke tests |

The CSV files are outputs, not training data. Full processed training inputs are obtained from the checksum-pinned Zenodo downloader. End-to-end model inference uses the checksum-pinned Hugging Face model package.

`api_context_examples.npz` is a small, pickle-free subset of the official
processed compound-perturbation data. Sample IDs, cell IDs, perturbation IDs,
arrays, shapes, and dtypes are retained so reviewers can verify provenance. It is
provided only as a deterministic A549 context example; users should supply their
own correctly normalized profile in the identical gene order for other cells.
