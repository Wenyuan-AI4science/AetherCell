import torch
import torch.nn as nn
from AetherCell_sub.src.RNAvae import RNAVAE
from dataloader_all import CustomHDF5Dataset
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
import copy
import os
from torch.utils.data import DataLoader,random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2048
h5_file = "../data/RNAdata.h5"  # Replace with your csv file path
dataset = CustomHDF5Dataset(h5_file)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

def train_vae(model, dataloader,optimizer,device):
    model.train()
    total_loss = 0
    for x,_ in tqdm(dataloader):
        feature = x
        label = x
        feature = feature.float().to(device)
        label = label.float().to(device)
        optimizer.zero_grad()
        recon, mean, log_var = model(feature)
        # Calculate VAE loss (reconstruction loss + KL divergence)
        recon_loss = nn.functional.mse_loss(label,recon,reduction='mean')
        # log_var = torch.clamp(log_var, min=-10, max=10)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test_vae(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    sum_pearson = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, _ = batch  # Assuming the dataloader returns (data, labels), but labels are not used
            feature = x.float().to(device)
            # Forward pass through the VAE
            recon, mean, log_var = model(feature)
            # Handle NaN and Inf values by replacing them with zeros
            recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
            label = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            # Flatten each sample in the batch to a 1D vector
            recon = recon.view(recon.size(0), -1)  # Shape: (batch_size, num_features)
            label = label.view(label.size(0), -1)  # Shape: (batch_size, num_features)
            # Compute means for each sample
            recon_mean = recon.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1)
            label_mean = label.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1)
            # Compute covariance between recon and label for each sample
            covariance = (recon * label).mean(dim=1) - (recon_mean.squeeze(1) * label_mean.squeeze(1))
            # Compute standard deviations for recon and label for each sample
            recon_var = (recon * recon).mean(dim=1) - (recon_mean.squeeze(1) ** 2)
            label_var = (label * label).mean(dim=1) - (label_mean.squeeze(1) ** 2)
            recon_std = torch.sqrt(recon_var + 1e-8)  # Add epsilon to avoid division by zero
            label_std = torch.sqrt(label_var + 1e-8)
            # Compute Pearson correlation coefficient for each sample
            pearson_coeff = covariance / (recon_std * label_std)
            # Replace any NaN results (from division by zero) with zero
            pearson_coeff = torch.nan_to_num(pearson_coeff, nan=0.0)
            # Accumulate the sum of Pearson coefficients and count the samples
            sum_pearson += pearson_coeff.sum().item()
            total_samples += recon.size(0)
    # Calculate the average Pearson correlation coefficient across all samples
    average_pearson = sum_pearson / total_samples if total_samples > 0 else 0.0
    print("VAE Average Pearson Correlation:", average_pearson)
    return average_pearson




vae_model = RNAVAE(device).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    vae_model = nn.DataParallel(vae_model)

vae_optimizer = optim.Adam(vae_model.parameters(), lr=3e-4)
num_epochs = 500
best_pearson =0.0
save_dir = '../result/model_checkpoints_RNAseq'
os.makedirs(save_dir, exist_ok=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    vae_optimizer, 
    mode='min', 
    factor=0.5, 
    patience=20, 
    min_lr=1e-6  # <--- Set minimum learning rate directly here, it will handle automatically
)
for epoch in range(num_epochs):
    vae_loss = train_vae(vae_model, train_loader, vae_optimizer, device) 
    current_lr = vae_optimizer.param_groups[0]['lr']
    print(f'epoch:{epoch+1}/{num_epochs}, vae_loss = {vae_loss:.6f}, lr = {current_lr:.2e}')
    scheduler.step(vae_loss)
    vae_pearson = test_vae(vae_model, test_loader,device)
    if vae_pearson > best_pearson:
        best_pearson = vae_pearson
        checkpoint = {
        'epoch': epoch + 1,
        'vae_model_state_dict': vae_model.state_dict(),
        'vae_optimizer_state_dict': vae_optimizer.state_dict(),
        }
        # Save the checkpoint
        save_path = os.path.join(save_dir, f'best_model.pt')
        torch.save(checkpoint, save_path)
        print(f'Models saved at epoch {epoch + 1} to {save_path}')
    if (epoch + 1) % 100 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'vae_model_state_dict': vae_model.state_dict(),
            'vae_optimizer_state_dict': vae_optimizer.state_dict(),
            }
        save_path = os.path.join(save_dir, f'{epoch}_model.pt')
        torch.save(checkpoint, save_path)
        print(f'Models saved at epoch {epoch + 1} to {save_path}')