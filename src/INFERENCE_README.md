# AetherCell — src Inference Scripts

This directory contains the **batch inference scripts** for AetherCell.
They are intended for large-scale prediction on custom datasets, as opposed to the
single-sample API in `aethercell-drug-discovery-v1.0.0/`.

---

## Scripts Overview

| Script | Perturbation | Output |
|--------|-------------|--------|
| `inference_delta_z.py` | Drug (compound) | 256-dim latent delta_z vectors |
| `inference_perturbed_expression.py` | Drug (compound) | 978-dim predicted post-treatment L1000 expression |
| `inference_knockdown_perturbed.py` | shRNA knockdown | 978-dim predicted post-knockdown L1000 expression |
| `aethercell_delta_z_inference.ipynb` | Drug **or** shRNA × L1000 **or** RNAseq control | 256-dim delta_z vectors (all four mode combinations) |

### `inference_delta_z.py`
Loads a trained drug perturbation model and encodes each (drug, cell line) pair
into a 256-dimensional **delta_z** vector — the directional shift in latent space
caused by the drug perturbation. Useful for drug similarity analysis, clustering,
and as features for downstream tasks (e.g. IC50 prediction).

**Outputs saved to `../result/delta_z/`:**
- `delta_z_predictions.npy` — raw array of shape `(N_samples, 256)`
- `delta_z_predictions.csv` — metadata + all 256 delta_z columns

### `inference_perturbed_expression.py`
Generates the predicted **post-treatment L1000 gene expression profile** (978 genes)
for each (drug, cell line) pair. Also saves control expression and the
expression delta (perturbed − control) for downstream DEG analysis.

**Outputs saved to `../result/perturbed_expression/`:**
- `perturbed_expression.npy` — predicted post-treatment expression `(N, 978)`
- `control_expression.npy` — input control expression `(N, 978)`
- `delta_expression.npy` — difference `(N, 978)`
- `metadata.csv` — sample / perturbation identifiers
- `summary_statistics.csv` — mean, std, abs-mean of delta

### `inference_knockdown_perturbed.py`
Same output format as `inference_perturbed_expression.py`, but for **shRNA gene
knockdown** perturbations. Uses PPI and sequence embeddings as gene representations
instead of drug SMILES.

**Outputs saved to `../result/perturbed_expression_sh/`:**
- Same file structure as above, plus `det_plate` and `cell_id` columns in metadata.

### `aethercell_delta_z_inference.ipynb`
A self-contained Jupyter notebook that covers all four inference modes:

| `PERTURBATION_MODE` | `CONTROL_SOURCE` | Description |
|---------------------|-----------------|-------------|
| `"drug"` | `"L1000"` | Standard drug inference with L1000 control |
| `"drug"` | `"RNAseq"` | Drug inference using RNAseq as control (no L1000 needed) |
| `"shrna"` | `"L1000"` | shRNA inference with L1000 control |
| `"shrna"` | `"RNAseq"` | shRNA inference using RNAseq as control (no L1000 needed) |

The notebook also includes optional PCA visualisation of the resulting delta_z
embeddings and a utility cell for tokenising new drug SMILES on the fly.

---

## Step 1 — Install the Environment

```bash
conda env create -f ../environment.yml
conda activate aethercell
```

---

## Step 2 — Download Required Files

### 2.1 Model Weights — Hugging Face

**Repository:** [https://huggingface.co/liwenyuan99/AetherCell](https://huggingface.co/liwenyuan99/AetherCell)

Download the following weight files and note where you save them.
You will set the corresponding paths in Step 3.

| File | Used by | Variable in script |
|------|--------|-------------------|
| LINCS VAE checkpoint (e.g. `epoch_184.pt`) | all scripts | `best_model_path_LINCSVAE` |
| RNA VAE checkpoint (`best_model.pt`) | all scripts | `best_model_path_RNAVAE` |
| Drug perturbation model checkpoint | `inference_delta_z.py`, `inference_perturbed_expression.py`, notebook (drug mode) | `trained_model_path` / `DRUG_MODEL_CKPT` |
| shRNA perturbation model checkpoint | `inference_knockdown_perturbed.py`, notebook (shrna mode) | `trained_model_path` / `SHRNA_MODEL_CKPT` |

The **MolFormer** tokeniser and weights are already bundled in the repository at:
```
aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/molformer/
```
Point `molformer_path` / `MOLFORMER_PATH` to that directory.

---

### 2.2 Processed Datasets — Zenodo

**Record:** [https://zenodo.org/records/18295255](https://zenodo.org/records/18295255)

#### Required for drug inference scripts

| File | Description | Variable in script |
|------|------------|-------------------|
| `RNAseq.parquet` | RNA-seq expression matrix, shape `(genes, cell_lines)` | `RNA_parquet_path` / `RNASEQ_PARQUET` |
| `L1000_ctrl.npy` | L1000 control expression matrix, shape `(N_ctrl, 978)` | `L1000_ctrl_npy` / `L1000_CTRL_NPY` |
| `ctrl_idx_map.pkl` | Dict mapping `control_id → row index` in `L1000_ctrl.npy` | `ctrl_idx_map_path` / `CTRL_IDX_MAP_PKL` |
| `drug_input_ids.npy` | Pre-tokenised SMILES input IDs, shape `(N_drugs, 160)` | `drug_input_ids_npy` / `DRUG_INPUT_IDS_NPY` |
| `drug_attention_mask.npy` | Corresponding attention masks, shape `(N_drugs, 160)` | `drug_attention_mask_npy` / `DRUG_ATTN_MASK_NPY` |
| `drug_idx_map.pkl` | Dict mapping `pert_id → row index` in drug arrays | `drug_idx_map_path` / `DRUG_IDX_MAP_PKL` |
| Inference metadata CSV | CSV with columns: `sample_id`, `pert_id`, `cell_iname`, `representative_control` | `inference_meta_csv` / `META_CSV` |

#### Additional files for shRNA inference scripts

| File | Description | Variable in script |
|------|------------|-------------------|
| `L1000_exp.npy` | L1000 experimental (perturbed) expression, shape `(N_exp, 978)` | `L1000_exp_npy` |
| `exp_idx_map.pkl` | Dict mapping `sample_id → row index` in `L1000_exp.npy` | `exp_idx_map_path` |
| `ensg_PPI_emb.csv` | PPI-based gene embeddings, indexed by Ensembl gene ID | `sh_embed_PPI_csv` / `SH_PPI_CSV` |
| `emb_tokens_first_all.npy` | Sequence embeddings, shape `(N_genes, 1152)` | `sh_embed_seq_npy` / `SH_SEQ_NPY` |
| `id2idx_ensg_seq2_all.pkl` | Dict mapping `gene_ensg → row index` in sequence embeddings | `sh_embed_seq_pkl` / `SH_SEQ_PKL` |
| shRNA metadata CSV | CSV with columns: `sample_id`, `gene_ensg`, `cell_iname`, `representative_control`, `pert_type`, `det_plate` | `inference_meta_csv` / `META_CSV` |

> **RNAseq-control mode (notebook only):** when `CONTROL_SOURCE = "RNAseq"`,
> the L1000 files (`L1000_ctrl.npy`, `ctrl_idx_map.pkl`, `L1000_exp.npy`,
> `exp_idx_map.pkl`) are **not** needed.

---

## Step 3 — Configure Paths

Open the script you want to run and edit the **Configuration** section near the
top of `main()`. Replace every placeholder path with the actual location of the
file on your machine.

Example (drug inference):
```python
# Model paths
best_model_path_LINCSVAE = "/your/path/to/epoch_184.pt"
best_model_path_RNAVAE   = "/your/path/to/best_model.pt"
molformer_path           = "../aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/molformer"
trained_model_path       = "/your/path/to/drug_model/best_model.pt"

# Data paths
inference_meta_csv        = "/your/path/to/inference_meta.csv"
L1000_ctrl_npy            = "/your/path/to/L1000_ctrl.npy"
ctrl_idx_map_path         = "/your/path/to/ctrl_idx_map.pkl"
RNA_parquet_path          = "/your/path/to/RNAseq.parquet"
drug_input_ids_npy        = "/your/path/to/drug_input_ids.npy"
drug_attention_mask_npy   = "/your/path/to/drug_attention_mask.npy"
drug_idx_map_path         = "/your/path/to/drug_idx_map.pkl"

# Output directory
save_dir = "../result/perturbed_expression"
```

For the notebook, edit only the **USER CONFIGURATION** cell (Section 2).

---

## Step 4 — Prepare Your Metadata CSV

The metadata CSV tells the script which samples to run inference on.

### Drug inference metadata

Required columns:

| Column | Description | Example |
|--------|------------|---------|
| `sample_id` | Unique identifier for this sample | `test_sample_001` |
| `pert_id` | Drug identifier matching `drug_idx_map.pkl` | `BRD-K12345678` |
| `cell_iname` | Cell line name matching `RNAseq.parquet` columns | `A549` |
| `representative_control` | Control sample ID matching `ctrl_idx_map.pkl` | `ctrl_001` |

### shRNA inference metadata

Additional required columns:

| Column | Description |
|--------|------------|
| `gene_ensg` | Ensembl gene ID matching embedding files (e.g. `ENSG00000141510`) |
| `pert_type` | Perturbation type string (e.g. `trt_sh`) |
| `det_plate` | Plate identifier |

---

## Step 5 — Run Inference

```bash
cd src

# Generate delta_z embeddings (drug)
python inference_delta_z.py

# Generate perturbed expression profiles (drug)
python inference_perturbed_expression.py

# Generate perturbed expression profiles (shRNA knockdown)
python inference_knockdown_perturbed.py
```

For the notebook:
```bash
jupyter notebook aethercell_delta_z_inference.ipynb
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: <pert_id>` | Drug not in `drug_idx_map.pkl` | Verify all `pert_id` values in your metadata exist in the map |
| `KeyError: <cell_iname>` | Cell line not in `RNAseq.parquet` | Check column names in the parquet file (case-sensitive) |
| `KeyError: <control_id>` | Control not in `ctrl_idx_map.pkl` | Verify all `representative_control` values in your metadata |
| `KeyError: <gene_ensg>` | Gene not in PPI / seq embedding files | Check `ensg_PPI_emb.csv` index and `id2idx_ensg_seq2_all.pkl` keys |
| CUDA out of memory | Batch size too large | Reduce `batch_size` (e.g. from 256 to 64) |
| `RuntimeError: weights_only` | Unexpected in patched scripts; use PyTorch ≥ 2.0 | All `torch.load` calls already use `weights_only=False` |

> **Windows note:** if the DataLoader hangs, set `num_workers=0` in the `DataLoader` call.
