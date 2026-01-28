import sys, os; sys.path.insert(0, os.path.abspath('/home/liwenyuan/Desktop/2025program/uni2.0re/perturb_model'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr # For calculating Pearson correlation coefficient


from LINCSvae import LINCSVAE
from RNAvae import RNAVAE
from uniperturb_drug import JointPerturbationPredictor
from dataloader_all import DDPDataset_ic50

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import os
import pandas as pd
import numpy as np
from ddp_predict_ic50_new import DDPPredictor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
# Call at the beginning of script
set_seed(42)


def train_predict(model, dataloader, optimizer,device):
    model.train()
    total_loss = 0
    loss_fn = nn.MSELoss()
    for batch in tqdm(dataloader, desc="Training"):
        label = batch['label'].to(device)
        label = label.to(device).float() 
        RNAdata= batch['cell_data'].to(device)
        input_ids= batch['input_ids'].to(device)
        attention_mask= batch['attention_mask'].to(device)
        optimizer.zero_grad()
        predict_ic50 = model(RNAdata, input_ids, attention_mask).squeeze(dim=-1)
        # Calculate loss here
        loss = loss_fn(predict_ic50, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_trues = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move label and inputs to device
        labels = batch['label'].to(device).float()
        rna    = batch['cell_data'].to(device)
        ids    = batch['input_ids'].to(device)
        mask   = batch['attention_mask'].to(device)
        # Forward pass
        predictions = model(rna, ids, mask).squeeze(dim=-1)
        # Collect, keep all on GPU first
        all_preds.append(predictions.cpu().numpy())
        all_trues.append(labels.cpu().numpy())
    # Concatenate numpy arrays
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    pearson_corr, _ = pearsonr(trues, preds)
    return pearson_corr

def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_k = k[7:]  # Remove "module." prefix
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict
    
local_device   =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
molformer_path         = "../perturb_model/mini_molformer"
best_model_path_LINCSVAE = "/home/liwenyuan/Desktop/2025program/uni2.0re/perturb_model/model_1020ckpt_L_random/epoch_72.pt"
best_model_path_RNAVAE     = "../perturb_model/RNA_vae_ckpt/best_model.pt"
predict_ckpt = torch.load("/home/liwenyuan/Desktop/2025program/uni2.0re/perturb_model/model_cp_cp_b1022/best_model.pt", map_location=device)
clean_predict_state = remove_module_prefix(predict_ckpt['model_state_dict'])
    
LINvae_model = LINCSVAE("cpu")
lin_ckpt = torch.load(best_model_path_LINCSVAE, map_location="cpu")
LINvae_model.load_state_dict(lin_ckpt['vae_model_state_dict'])
encoder = LINvae_model.encoder.to(local_device)
decoder = LINvae_model.decoder.to(local_device)

RNAvae_model = RNAVAE("cpu")
rna_ckpt = torch.load(best_model_path_RNAVAE, map_location="cpu")
RNAvae_model.load_state_dict(remove_module_prefix(rna_ckpt['vae_model_state_dict']))
RNAencoder = RNAvae_model.encoder.to(local_device)

orig_model = JointPerturbationPredictor(
        LINCSencoder=encoder,     # LINCSencoder instance used during training
        LINCSdecoder=decoder,     # LINCSdecoder instance used during training
        RNAencoder=RNAencoder,       # RNAencoder instance used during training
        molformer_path=molformer_path,
        device=device)
orig_model.load_state_dict(clean_predict_state)  
orig_model = orig_model.to(device)
new_model = DDPPredictor(
        pretrained_model=orig_model,
        device=device).to(device)

train_meta_csv = "/home/liwenyuan/Desktop/2025program/uni2.0re/ddp_data/gdsc/cell_blind/cell_blind_train.csv"
test_meta_csv = "/home/liwenyuan/Desktop/2025program/uni2.0re/ddp_data/gdsc/cell_blind/cell_blind_test.csv"
RNA_parquet_path = "/home/liwenyuan/Desktop/2025program/uni2.0/data/ddp_data/cell_used10085.parquet"
drug_input_ids_npy = "/home/liwenyuan/Desktop/2025program/uni2.0/data/ddp_data/drug_input_ids.npy"
drug_attention_mask_npy = "/home/liwenyuan/Desktop/2025program/uni2.0/data/ddp_data/drug_attention_mask.npy"
drug_idx_map_path = "/home/liwenyuan/Desktop/2025program/uni2.0/data/ddp_data/drug_idx_map.pkl"

train_dataset = DDPDataset_ic50(train_meta_csv, RNA_parquet_path,
                           drug_input_ids_npy, drug_attention_mask_npy,
                           drug_idx_map_path)

test_dataset = DDPDataset_ic50(test_meta_csv, RNA_parquet_path,
                           drug_input_ids_npy, drug_attention_mask_npy,
                           drug_idx_map_path)

trainloader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True) # Training set needs shuffling

testloader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False) # Test set does not need shuffling


optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, new_model.parameters()),
        lr=3e-4,
        weight_decay=0.01
    )

num_epochs = 200
best_pcc = 0 
save_dir = "./ddp_predict_ic50_cb_gdsc2"
os.makedirs(save_dir, exist_ok=True)

log_csv = os.path.join(save_dir, "train_log.csv")
if not os.path.exists(log_csv):
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_pcc,lr,best_pcc\n")
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='max',
                                                 factor=0.5,
                                                 patience=5)
for epoch in range(num_epochs):
    loss = train_predict(new_model, trainloader, optimizer,device)
    pearson_corr  = evaluate(new_model, testloader, device)
    lr   = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch}  train_loss={loss:.4f}  val_PCC={pearson_corr:.4f}")
    with open(log_csv, "a") as f:
        f.write(f"{epoch},{loss:.6f},{pearson_corr:.6f},{lr:.8f},{max(best_pcc, pearson_corr):.6f}\n")

    scheduler.step(pearson_corr)
    if  pearson_corr > best_pcc:
                best_pcc = pearson_corr
                predict_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': new_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'loss': loss
                    } 
                save_path = os.path.join(save_dir, f'best_model.pt')
                torch.save(predict_checkpoint, save_path)
                print(f'[Saved best drug model at epoch {epoch+1} -> {save_path}')
    if (epoch + 1) % 10 == 0:
        torch.save(
            {"epoch": epoch + 1, 'model_state_dict': new_model.state_dict()},
            os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        )