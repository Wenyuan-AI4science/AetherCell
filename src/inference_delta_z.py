"""
Inference script for generating delta_z predictions from drug and transcriptome inputs.
This script loads a trained model and generates latent space perturbation vectors (delta_z).
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

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
def inference_delta_z(model, dataloader, device, save_dir):
    """
    Generate delta_z predictions for all samples in the dataloader.

    Args:
        model: Trained JointPerturbationPredictor model
        dataloader: DataLoader with inference data
        device: torch device (cuda/cpu)
        save_dir: Directory to save results

    Returns:
        results_df: DataFrame with sample IDs and delta_z predictions
    """
    model.eval()

    all_delta_z = []
    all_sample_ids = []
    all_pert_ids = []
    all_control_ids = []

    print("Generating delta_z predictions...")
    for batch in tqdm(dataloader, desc="Inference"):
        # Load data
        control = batch['control'].to(device).float()
        RNAseq = batch['rna'].to(device).float()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sample_ids = batch['sample_id']
        pert_ids = batch['pert_id']
        control_ids = batch['control_id']

        # Forward pass - get delta_z
        _, _, delta_z_pred, _ = model(RNAseq, control, input_ids, attention_mask)

        # Move to CPU and collect results
        delta_z_pred_cpu = delta_z_pred.cpu().numpy()
        all_delta_z.append(delta_z_pred_cpu)
        all_sample_ids.extend(sample_ids)
        all_pert_ids.extend(pert_ids)
        all_control_ids.extend(control_ids)

    # Concatenate all results
    all_delta_z = np.concatenate(all_delta_z, axis=0)  # Shape: (N_samples, latent_dim)

    print(f"Generated delta_z for {len(all_sample_ids)} samples")
    print(f"Delta_z shape: {all_delta_z.shape}")

    # Save as numpy array
    os.makedirs(save_dir, exist_ok=True)
    delta_z_save_path = os.path.join(save_dir, "delta_z_predictions.npy")
    np.save(delta_z_save_path, all_delta_z)
    print(f"Saved delta_z to: {delta_z_save_path}")

    # Create metadata DataFrame
    results_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'pert_id': all_pert_ids,
        'control_id': all_control_ids,
    })

    # Add delta_z columns
    for i in range(all_delta_z.shape[1]):
        results_df[f'delta_z_{i}'] = all_delta_z[:, i]

    # Save metadata and delta_z as CSV
    csv_save_path = os.path.join(save_dir, "delta_z_predictions.csv")
    results_df.to_csv(csv_save_path, index=False)
    print(f"Saved metadata and delta_z to: {csv_save_path}")

    return results_df


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
    save_dir = "../result/delta_z"

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
    results_df = inference_delta_z(
        model=predict_model,
        dataloader=inference_loader,
        device=device,
        save_dir=save_dir
    )

    print("\n" + "="*50)
    print("Inference completed successfully!")
    print(f"Results saved to: {save_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
