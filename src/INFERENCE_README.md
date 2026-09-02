# Portable batch inference

The historical inference scripts have been retained as compatibility entry points, but they no longer contain workstation or server paths. All three delegate to the tested `aethercell-batch-infer` command.

They retain the original scientific roles while sharing one portable CLI:

| Script | Perturbation | Primary scientific output |
|---|---|---|
| `inference_delta_z.py` | Drug | 256-dimensional latent displacement vectors for similarity, clustering, and downstream modelling |
| `inference_perturbed_expression.py` | Drug | 978-dimensional post-treatment L1000 expression and change from control |
| `inference_knockdown_perturbed.py` | shRNA knockdown | 978-dimensional post-knockdown L1000 expression and change from control |
| `aethercell_delta_z_inference.ipynb` | Drug or shRNA | Interactive delta-z inference, inspection, and optional PCA visualisation |

The notebook continues to document four combinations: drug × L1000 control,
drug × RNA-seq control, shRNA × L1000 control, and shRNA × RNA-seq control.
The command-line workflow below is the tested path for reviewer reproduction.

Run the preflight checker first:

```bash
pip install -e ".[model,test]"
aethercell-doctor
aethercell-doctor --full   # also requires downloaded data and models
```

If assets are missing, the checker prints the exact verified download commands and source URLs.

## Required released files

Download the model package from [Hugging Face](https://huggingface.co/liwenyuan99/AetherCell) and processed data from [Zenodo](https://zenodo.org/records/18295255). The repository downloaders verify the pinned size and checksum:

```bash
python scripts/download_models.py --extract --output-dir models
python scripts/download_data.py --extract --output-dir data/zenodo
```

Core model files are `L1000_vae.pt`, `RNA_vae.pt`,
`predictor_L_drug.pt`, `predictor_L_sh.pt`, and the `molformer/` directory.

The official drug layout contains:

| File | Purpose |
|---|---|
| `RNAseq.parquet` | RNA-seq context matrix; columns are cell lines |
| `L1000_ctrl.npy` / `ctrl_idx_map.pkl` | control expression and sample-to-row map |
| `L1000_exp.npy` / `exp_idx_map.pkl` | perturbed expression and sample-to-row map |
| `drug_input_ids.npy` | pre-tokenised SMILES IDs |
| `drug_attention_mask.npy` | MolFormer attention masks |
| `drug_idx_map.pkl` | perturbation ID to token-array row map |
| `df_train_drug.csv` / `df_test_drug.csv` | official train/test metadata |

The shRNA layout additionally contains `ensg_PPI_emb.csv`,
`emb_tokens_first_all.npy`, `id2idx_ensg_seq2_all.pkl`, and
`sh_meta_s1_train.csv` / `sh_meta_s1_test.csv`. Legacy index-map pickles are
loaded by a restricted primitive-only reader, not by unrestricted `pickle.load`.

## Drug inference

```bash
aethercell-batch-infer \
  --mode drug \
  --zenodo-dir data/zenodo/data4train/data4zendo --split test \
  --legacy-src models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction \
  --lincs-vae models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/L1000_vae.pt \
  --rna-vae models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/RNA_vae.pt \
  --molformer-dir models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/molformer \
  --checkpoint models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/predictor_L_drug.pt \
  --output-dir results/drug_test
```

The old commands remain equivalent:

```bash
python src/inference_delta_z.py [the same arguments]
python src/inference_perturbed_expression.py [the same arguments]
```

For backward compatibility, `inference_delta_z.py` also writes
`delta_z_predictions.npy` and `delta_z_predictions.csv`.
`inference_perturbed_expression.py` additionally writes the historical
`perturbed_expression.npy`, `control_expression.npy`, `delta_expression.npy`,
and `summary_statistics.csv` files. These are aliases/derived outputs; the
portable files remain present.

## shRNA inference

```bash
aethercell-batch-infer \
  --mode shrna \
  --zenodo-dir data/zenodo/data4train/data4zendo --split test \
  --legacy-src models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction \
  --lincs-vae models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/L1000_vae.pt \
  --rna-vae models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/RNA_vae.pt \
  --checkpoint models/aethercell-drug-discovery-v1.0.0/models/transcriptome_prediction/predictor_L_sh.pt \
  --output-dir results/shrna_test
```

Compatibility command:

```bash
python src/inference_knockdown_perturbed.py [the same arguments except --mode]
```

The shRNA compatibility command writes the same historical expression aliases
and retains `control_id`, `cell_id`, and `det_plate` metadata when available.

## Metadata contracts

Official drug metadata provide `sample_id`, `pert_id`, `cell_iname`, and
`representative_control`. Official shRNA metadata additionally provide
`gene_ensg`, `pert_type`, and `det_plate`. Every referenced drug, cell line,
control, or gene must occur in the corresponding released index or embedding
table; absent identifiers fail with an explicit row-level error.

For custom inputs, use the pickle-free NPZ contracts below instead of editing
paths inside source files:

| Mode | Required NPZ arrays |
|---|---|
| Drug | `control`, `rna`, `input_ids`, `attention_mask` |
| shRNA | `control`, `rna`, `sh_ppi`, `sh_seq` |

Both modes may include `sample_id`, `cell_id`, and `pert_id` string arrays.

## Outputs

Every mode writes the same row-aligned contract:

```text
predicted_expression.npy  # [samples, 978]
predicted_delta.npy       # [samples, 978]
predicted_delta_z.npy     # [samples, 256]
metadata.csv              # row, sample_id, cell_id, pert_id
```

## Troubleshooting

| Error | Likely cause | Resolution |
|---|---|---|
| Missing model or data file | Assets have not been downloaded or extracted in the expected layout | Run `aethercell-doctor --full` and use the printed verified command |
| `KeyError` for a perturbation | ID absent from the released drug/gene index | Check `pert_id` or `gene_ensg` against the corresponding map |
| `KeyError` for a cell line | `cell_iname` absent from `RNAseq.parquet` columns | Match names exactly, including case |
| `KeyError` for a control | `representative_control` absent from `ctrl_idx_map.pkl` | Use an official or correctly indexed control ID |
| CUDA out of memory | Batch size exceeds available GPU memory | Reduce `--batch-size` or use `--device cpu` |
| Windows worker hang | Multiprocessing incompatibility | Keep `--num-workers 0` |
| Checkpoint incompatibility | Model definitions do not match the released weights | Omit `--legacy-src` so it is inferred from `L1000_vae.pt`, or point it to the released `models/transcriptome_prediction/` directory |
