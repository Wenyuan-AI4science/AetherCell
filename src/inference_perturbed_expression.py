"""
Inference script for generating perturbed expression profiles from drug and transcriptome inputs.
This script loads a trained model and generates post-treatment gene expression predictions.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from LINCSvae import LINCSVAE
from RNAvae import RNAVAE
from aethercell_drug import JointPerturbationPredictor
from dataloader_all import PredictorDatasetDP2_i


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[7:]  # Remove "module." prefix
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


@torch.no_grad()
def inference_perturbed_expression(model, dataloader, device, save_dir, gene_names=None):
    """
    Generate perturbed expression profiles for all samples in the dataloader.

    Args:
        model: Trained JointPerturbationPredictor model
        dataloader: DataLoader with inference data
        device: torch device (cuda/cpu)
        save_dir: Directory to save results
        gene_names: List of gene names (optional, for column headers)

    Returns:
        results_dict: Dictionary containing predictions and metadata
    """
    model.eval()

    all_predictions = []
    all_sample_ids = []
    all_pert_ids = []
    all_control_ids = []
    all_controls = []

    print("Generating perturbed expression predictions...")
    for batch in tqdm(dataloader, desc="Inference"):
        # Load data
        control = batch['control'].to(device).float()
        RNAseq = batch['rna'].to(device).float()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sample_ids = batch['sample_id']
        pert_ids = batch['pert_id']
        control_ids = batch['control_id']

        # Forward pass - get predicted perturbed expression
        x_pred, _, _, _ = model(RNAseq, control, input_ids, attention_mask)

        # Move to CPU and collect results
        x_pred_cpu = x_pred.cpu().numpy()
        control_cpu = control.cpu().numpy()

        all_predictions.append(x_pred_cpu)
        all_controls.append(control_cpu)
        all_sample_ids.extend(sample_ids)
        all_pert_ids.extend(pert_ids)
        all_control_ids.extend(control_ids)

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (N_samples, n_genes)
    all_controls = np.concatenate(all_controls, axis=0)

    print(f"Generated predictions for {len(all_sample_ids)} samples")
    print(f"Expression shape: {all_predictions.shape}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # ============ Save Perturbed Expression ============
    perturbed_save_path = os.path.join(save_dir, "perturbed_expression.npy")
    np.save(perturbed_save_path, all_predictions)
    print(f"Saved perturbed expression to: {perturbed_save_path}")

    # ============ Save Control Expression ============
    control_save_path = os.path.join(save_dir, "control_expression.npy")
    np.save(control_save_path, all_controls)
    print(f"Saved control expression to: {control_save_path}")

    # ============ Calculate and Save Delta Expression ============
    delta_expression = all_predictions - all_controls
    delta_save_path = os.path.join(save_dir, "delta_expression.npy")
    np.save(delta_save_path, delta_expression)
    print(f"Saved delta expression to: {delta_save_path}")

    # ============ Create Metadata DataFrame ============
    metadata_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'pert_id': all_pert_ids,
        'control_id': all_control_ids,
    })

    metadata_save_path = os.path.join(save_dir, "metadata.csv")
    metadata_df.to_csv(metadata_save_path, index=False)
    print(f"Saved metadata to: {metadata_save_path}")

    # ============ Create Summary Statistics ============
    summary_stats = {
        'n_samples': len(all_sample_ids),
        'n_genes': all_predictions.shape[1],
        'perturbed_mean': float(np.mean(all_predictions)),
        'perturbed_std': float(np.std(all_predictions)),
        'delta_mean': float(np.mean(delta_expression)),
        'delta_std': float(np.std(delta_expression)),
        'delta_abs_mean': float(np.mean(np.abs(delta_expression))),
    }

    summary_df = pd.DataFrame([summary_stats])
    summary_save_path = os.path.join(save_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_save_path, index=False)
    print(f"Saved summary statistics to: {summary_save_path}")

    return {
        'perturbed_expression': all_predictions,
        'control_expression': all_controls,
        'delta_expression': delta_expression,
        'metadata': metadata_df,
        'summary': summary_stats
    }


def main():
    # ============ Configuration ============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model paths
    best_model_path_LINCSVAE = "../result/model_ckpt_L_random/epoch_184.pt"
    best_model_path_RNAVAE = "../result/model_checkpoints_RNAseq/best_model.pt"
    molformer_path = "/home/liwenyuan/Desktop/2025program/uni2.0re/perturb_model/mini_molformer"
    trained_model_path = "./model_drug_cell_b116/best_model.pt"  # Your trained model checkpoint

    # Data paths for inference
    inference_meta_csv = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/df_test_drug.csv"  # Update this path
    L1000_ctrl_npy = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/L1000_ctrl.npy"
    ctrl_idx_map_path = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/ctrl_idx_map.pkl"
    RNA_parquet_path = "/home/liwenyuan/Desktop/2025program/data/uni/RNAseq.parquet"
    drug_input_ids_npy = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/drug_input_ids.npy"
    drug_attention_mask_npy = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/drug_attention_mask.npy"
    drug_idx_map_path = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/drug_idx_map.pkl"

    # Output directory
    save_dir = "../result/perturbed_expression"

    # Optional: Load gene names for better interpretability
    gene_names = None  # You can load gene names from your data if available
    # Example: gene_names = pd.read_csv("gene_names.csv")['gene'].tolist()

    # ============ Load Pre-trained VAE Models ============
    print("Loading VAE models...")

    # Load LINCS VAE
    LINvae_model = LINCSVAE("cpu")
    lin_ckpt = torch.load(best_model_path_LINCSVAE, map_location="cpu", weights_only=False)
    LINvae_model.load_state_dict(remove_module_prefix(lin_ckpt['vae_model_state_dict']))
    encoder = LINvae_model.encoder.to(device)
    decoder = LINvae_model.decoder.to(device)

    # Load RNA VAE
    RNAvae_model = RNAVAE("cpu")
    rna_ckpt = torch.load(best_model_path_RNAVAE, map_location="cpu", weights_only=False)
    RNAvae_model.load_state_dict(remove_module_prefix(rna_ckpt['vae_model_state_dict']))
    RNAencoder = RNAvae_model.encoder.to(device)

    # ============ Build Prediction Model ============
    print("Building prediction model...")
    predict_model = JointPerturbationPredictor(
        encoder, decoder, RNAencoder, molformer_path, device
    ).to(device)

    # Load trained weights
    print(f"Loading trained model from: {trained_model_path}")
    predict_ckpt = torch.load(trained_model_path, map_location=device,weights_only=False)
    clean_predict_state = remove_module_prefix(predict_ckpt['model_state_dict'])
    predict_model.load_state_dict(clean_predict_state)
    predict_model.eval()

    print("Model loaded successfully!")

    # ============ Prepare Inference Dataset ============
    print("Loading inference dataset...")
    inference_dataset = PredictorDatasetDP2_i(
        meta_csv=inference_meta_csv,
        L1000_ctrl_npy=L1000_ctrl_npy,
        ctrl_idx_map_path=ctrl_idx_map_path,
        RNA_parquet_path=RNA_parquet_path,
        drug_input_ids_npy=drug_input_ids_npy,
        drug_attention_mask_npy=drug_attention_mask_npy,
        drug_idx_map_path=drug_idx_map_path
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Inference dataset size: {len(inference_dataset)}")

    # ============ Run Inference ============
    results = inference_perturbed_expression(
        model=predict_model,
        dataloader=inference_loader,
        device=device,
        save_dir=save_dir,
        gene_names=gene_names
    )

    print("\n" + "="*50)
    print("Inference completed successfully!")
    print(f"Results saved to: {save_dir}")
    print("\nSummary Statistics:")
    for key, value in results['summary'].items():
        print(f"  {key}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()
