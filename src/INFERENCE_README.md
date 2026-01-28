# AetherCell Inference Scripts

Two inference scripts for generating predictions from trained AetherCell models.

## Scripts Overview

### 1. inference_delta_z.py
Generates latent space perturbation vectors (delta_z) from drug and transcriptome inputs.

**Output files:**
- `delta_z_predictions.npy` - NumPy array of shape (N_samples, latent_dim)
- `delta_z_predictions.csv` - CSV file with sample metadata and delta_z values
- Columns: sample_id, pert_id, control_id, delta_z_0, delta_z_1, ..., delta_z_255

**Use cases:**
- Analyzing drug effects in latent space
- Comparing drug mechanisms
- Downstream machine learning tasks using delta_z as features

### 2. inference_perturbed_expression.py
Generates post-treatment gene expression profiles from drug and transcriptome inputs.

**Output files:**
- `perturbed_expression.npy` - Predicted expression after drug treatment
- `control_expression.npy` - Control expression (input)
- `delta_expression.npy` - Difference (perturbed - control)
- `perturbed_expression.parquet` - Parquet format with metadata
- `delta_expression.parquet` - Delta in parquet format
- `metadata.csv` - Sample metadata only
- `summary_statistics.csv` - Summary statistics

**Use cases:**
- Predicting gene expression changes
- Drug response prediction
- Identifying drug targets and biomarkers

## Usage

### Step 1: Prepare your inference metadata CSV

Your CSV should contain the following columns:
- `sample_id`: Unique identifier for each sample
- `pert_id`: Perturbation (drug) identifier
- `representative_control`: Control sample identifier
- `cell_iname`: Cell line name (should match RNAseq.parquet columns)

Example:
```csv
sample_id,pert_id,representative_control,cell_iname
test_sample_001,BRD-K12345678,ctrl_001,A549
test_sample_002,BRD-K87654321,ctrl_002,MCF7
```

### Step 2: Update paths in the scripts

Edit the configuration section in each script:

```python
# Model paths
best_model_path_LINCSVAE = "../result/model_ckpt_L_random/epoch_184.pt"
best_model_path_RNAVAE = "../result/model_checkpoints_RNAseq/best_model.pt"
molformer_path = "/path/to/mini_molformer"
trained_model_path = "./model_drug_cell_b116/best_model.pt"  # Your trained model

# Data paths
inference_meta_csv = "/path/to/your/inference_meta.csv"  # UPDATE THIS
L1000_ctrl_npy = "/path/to/L1000_ctrl.npy"
ctrl_idx_map_path = "/path/to/ctrl_idx_map.pkl"
RNA_parquet_path = "/path/to/RNAseq.parquet"
drug_input_ids_npy = "/path/to/drug_input_ids.npy"
drug_attention_mask_npy = "/path/to/drug_attention_mask.npy"
drug_idx_map_path = "/path/to/drug_idx_map.pkl"

# Output directory
save_dir = "./inference_results/delta_z"  # or "./inference_results/perturbed_expression"
```

### Step 3: Run inference

**For delta_z generation:**
```bash
python inference_delta_z.py
```

**For perturbed expression generation:**
```bash
python inference_perturbed_expression.py
```

## Requirements

The scripts use the inference dataset class `PredictorDatasetDP2_i` which requires:
- Control expression data (L1000_ctrl.npy)
- Control index mapping (ctrl_idx_map.pkl)
- RNA-seq data (RNAseq.parquet)
- Drug tokenization data (drug_input_ids.npy, drug_attention_mask.npy, drug_idx_map.pkl)

All drug compounds in your inference metadata must be present in `drug_idx_map.pkl`.

## Output Format Details

### Delta_z output (inference_delta_z.py)

**delta_z_predictions.npy:**
- Shape: (N_samples, 256)
- Data type: float32
- Each row is the predicted latent space perturbation vector for one sample

**delta_z_predictions.csv:**
```csv
sample_id,pert_id,control_id,delta_z_0,delta_z_1,...,delta_z_255
test_001,BRD-K12345678,ctrl_001,0.123,-0.456,...,0.789
```

### Perturbed expression output (inference_perturbed_expression.py)

**perturbed_expression.npy:**
- Shape: (N_samples, 978)
- Data type: float32
- Each row is the predicted gene expression after drug treatment

**delta_expression.npy:**
- Shape: (N_samples, 978)
- Data type: float32
- Each row is the predicted expression change (perturbed - control)

**perturbed_expression.parquet:**
- Efficient format for large datasets
- Includes sample metadata columns (sample_id, pert_id, control_id)
- Each gene has its own column (gene_0, gene_1, ..., gene_977)

**summary_statistics.csv:**
```csv
n_samples,n_genes,perturbed_mean,perturbed_std,delta_mean,delta_std,delta_abs_mean
1000,978,0.123,1.456,-0.002,0.845,0.523
```

## Performance Tips

1. **Batch size**: Adjust based on your GPU memory
   ```python
   inference_loader = DataLoader(
       inference_dataset,
       batch_size=256,  # Increase if you have more GPU memory
       ...
   )
   ```

2. **Number of workers**: Adjust based on your CPU cores
   ```python
   num_workers=4,  # Increase for faster data loading
   ```

3. **Large datasets**: Use parquet format for efficient storage and loading
   - Perturbed expression parquet files can be loaded incrementally
   - Use pandas or polars for efficient parquet reading

## Example: Loading Results

### Load delta_z predictions
```python
import numpy as np
import pandas as pd

# Load as numpy array
delta_z = np.load("./inference_results/delta_z/delta_z_predictions.npy")
print(f"Delta_z shape: {delta_z.shape}")

# Load as DataFrame with metadata
df = pd.read_csv("./inference_results/delta_z/delta_z_predictions.csv")
print(df.head())
```

### Load perturbed expression
```python
import numpy as np
import pandas as pd

# Load as numpy array
perturbed_expr = np.load("./inference_results/perturbed_expression/perturbed_expression.npy")
delta_expr = np.load("./inference_results/perturbed_expression/delta_expression.npy")

# Load as parquet (more efficient for large data)
df_perturbed = pd.read_parquet("./inference_results/perturbed_expression/perturbed_expression.parquet")
df_delta = pd.read_parquet("./inference_results/perturbed_expression/delta_expression.parquet")

# Load metadata only
metadata = pd.read_csv("./inference_results/perturbed_expression/metadata.csv")
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Reduce batch size
```python
batch_size=128,  # or even smaller like 64
```

### Issue: Drug not found in drug_idx_map
**Solution:** Make sure all drugs in your inference_meta.csv are present in the drug_idx_map.pkl used during training. You may need to tokenize new drugs first.

### Issue: Cell line not found in RNAseq.parquet
**Solution:** Ensure cell line names in your metadata match exactly with column names in RNAseq.parquet (case-sensitive).

### Issue: Control not found in ctrl_idx_map
**Solution:** Verify that control sample IDs in your metadata exist in the ctrl_idx_map.pkl file.

## Additional Notes

- Both scripts automatically handle NaN values using `torch.nan_to_num`
- Models are set to evaluation mode (`model.eval()`)
- Gradients are disabled for inference (`@torch.no_grad()`)
- Results are automatically saved in the specified output directory
- Progress bars show inference progress using tqdm

## Contact

For questions or issues, please refer to the main AetherCell documentation or open an issue on the project repository.
