import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
# Only define some "constants" or "paths" globally, don't load any large data
'''cell blind'''
train_meta_csv         = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/df_train_drug.csv"
test_meta_drug         = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/df_test_drug.csv"
'''drug blind'''
# train_meta_csv         = "/home/liwenyuan/2025EXP/uniperturb/data/uni/df_train_newdrug.csv"
# test_meta_drug         = "/home/liwenyuan/2025EXP/uniperturb/data/uni/df_test_newdrug.csv"

L1000_exp_npy          = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/L1000_exp.npy"
L1000_ctrl_npy         = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/L1000_ctrl.npy"
exp_idx_map_path       = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/exp_idx_map.pkl"
ctrl_idx_map_path      = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/ctrl_idx_map.pkl"
RNA_parquet_path       = "/home/liwenyuan/Desktop/2025program/data/uni/RNAseq.parquet"
drug_input_ids_npy     = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/drug_input_ids.npy"
drug_attention_mask_npy= "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/drug_attention_mask.npy"
drug_idx_map_path      = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/cp_p/drug_idx_map.pkl"
molformer_path         = "/home/liwenyuan/Desktop/2025program/uni2.0re/perturb_model/mini_molformer"
best_model_path_LINCSVAE = "../result/model_ckpt_L_random/epoch_184.pt"
best_model_path_RNAVAE     = "../result/model_checkpoints_RNAseq/best_model.pt"




# Only import class definitions at the top, don't instantiate
from LINCSvae import LINCSVAE
from RNAvae import RNAVAE
from aethercell_drug import JointPerturbationPredictor
from dataloader_all import PredictorDatasetDP2

def compute_cell_line_mean(dataloader, device):
    """
    Calculate the average Label vector (Centroids) for each cell line.
    For GPU memory safety, accumulation is performed on CPU.
    """
    print("Computing Per-Cell-Line Mean Label (on CPU)...")
    
    # Store accumulation sum and count: {cell_line: {'sum': tensor, 'count': int}}
    stats = {}

    # No need to compute gradients
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Grouping Centroids"):
            # Get data and process directly on CPU to avoid occupying GPU memory
            # Note: Here we calculate the mean of X_true, not Delta
            x_true = batch['label'].float() 
            x_true = torch.nan_to_num(x_true)
            
            cell_lines = batch['det_plate']
            
            # Accumulate sample by sample
            for i, cell_name in enumerate(cell_lines):
                if isinstance(cell_name, torch.Tensor):
                    cell_name = cell_name.item()
                
                # Extract the vector
                d_vec = x_true[i]
                
                if cell_name not in stats:
                    stats[cell_name] = {
                        'sum': torch.zeros_like(d_vec), 
                        'count': 0
                    }
                
                # Accumulate on CPU
                stats[cell_name]['sum'] += d_vec
                stats[cell_name]['count'] += 1

    mean_dict = {}
    print("\nCell Line Statistics:")
    sorted_keys = sorted(stats.keys())
    for cell_name in sorted_keys:
        data = stats[cell_name]
        if data['count'] > 0:
            # Calculate mean
            mean_val = data['sum'] / data['count']
            # Store in dictionary (still on CPU, will move to GPU during evaluation)
            mean_dict[cell_name] = mean_val
            # print(f"  - {cell_name}: {data['count']} samples") # Log too long, can be commented out
    
    print(f"Computed means for {len(mean_dict)} cell lines.")
    return mean_dict
    

def precompute_cell_specific_latent_means(dataloader,encoder,device):
    encoder.eval()
    print("Pre-computing Plate Means for Specificity Loss...")
    cell_sums = {}
    cell_counts = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scanning Plates"):
            cells = batch['cell_id'] 
            labels = batch['label'].to(device).float() # Use the true value after drug treatment
            control = batch['control'].to(device).float()
            labels = torch.nan_to_num(labels)
            control  = torch.nan_to_num(control)
            _,z_control_mu,_ = encoder(control)
            _, z_drug_mu, _ = encoder(labels)
            delta_z_true = (z_drug_mu - z_control_mu)
            delta_z_true = delta_z_true.cpu()
            for i, cell_id in enumerate(cells):
                # Handle Tensor type IDs
                if isinstance(cell_id, torch.Tensor):
                    cell_id = cell_id.item()
                    
                if cell_id not in cell_sums:
                    cell_sums[cell_id] = torch.zeros_like(delta_z_true[i])
                    cell_counts[cell_id] = 0
                
                cell_sums[cell_id] += delta_z_true[i]
                cell_counts[cell_id] += 1
        # Calculate mean
        cell_means = {}
        for cid in cell_sums:
            cell_means[cid] = cell_sums[cid] / cell_counts[cid]
            
    print(f"Computed means for {len(cell_means)} cells.")
    return cell_means

def delta_weighted_mse(x_pred, x_true, x_ctrl, q=0.8, p=2.0, eps=1e-8):
    # q: quantile as threshold within each sample (e.g., 80th percentile)
    # p: weight for amplifying large Delta; p=2 is common
    d_true = x_true - x_ctrl
    d_pred = x_pred - x_ctrl
    s = d_true.abs()  # Signal strength
    # Threshold within each sample (e.g., 80% percentile)
    thr = torch.quantile(s, q, dim=1, keepdim=True)
    w = (s / (thr + eps))**p          # Large Delta has large weight
    w = w / (w.mean(dim=1, keepdim=True) + eps)  # Normalize to keep loss scale stable
    return (w * (d_pred - d_true).pow(2)).mean()


def delta_topk_cosine_loss_pure(delta_pred, delta_true, k=200, eps=1e-8):
    """
    Pure differential space Top-K loss
    delta_pred: x_pred - xhat_ctrl
    delta_true: label - control
    """
    k = min(k, delta_true.size(1))
    # Lock onto the genes with the most dramatic response in Ground Truth
    idx = delta_true.abs().topk(k, dim=1).indices

    d_true_k = torch.gather(delta_true, 1, idx)
    d_pred_k = torch.gather(delta_pred, 1, idx)

    # Normalize and calculate cosine similarity
    d_true_k = d_true_k - d_true_k.mean(dim=1, keepdim=True)
    d_pred_k = d_pred_k - d_pred_k.mean(dim=1, keepdim=True)

    cos = F.cosine_similarity(d_true_k, d_pred_k, dim=1)
    return (1.0 - cos).mean()


class CellSpecificGainLoss(nn.Module):
    def __init__(self, cell_means, device, margin=0.1):
        """
        cell_means: dictionary {cell_id (str/int): mean_tensor (Tensor [256])}
                    This is the pre-computed latent space mean.
        margin: Margin for Hinge loss.
        """
        super().__init__()
        self.device = device
        self.margin = margin
        
        # Convert cell_means dictionary to a more efficient lookup table (if cell_id is int)
        # Or simply store the dictionary and dynamically construct batch in forward
        self.cell_means = cell_means

    def forward(self, delta_z_pred, delta_z_true, cell_ids):

        # 1. Build the Baseline (cell-generic stress response) for the current batch
        # We need to extract the means corresponding to the current batch from the pre-computed dictionary and stack them into [B, 256]
        batch_means_list = []
        for cid in cell_ids:
            if isinstance(cid, torch.Tensor):
                cid = cid.item()
            # Get from dictionary, fallback to all-zeros if not encountered (rare case)
            mean_vec = self.cell_means.get(cid, torch.zeros_like(delta_z_pred[0].cpu()))
            batch_means_list.append(mean_vec)

        # Stack and move to GPU
        # global_mu_cell shape: [B, 256]
        # Use detach() to ensure we don't backpropagate to update this mean (it's a constant)
        global_mu_cell = torch.stack(batch_means_list).to(self.device).detach()

        # 2. Calculate Prediction Error
        # Model prediction vs true value
        err_pred = torch.norm(delta_z_pred - delta_z_true, p=2, dim=1)

        # 3. Calculate Baseline Error (Background Noise)
        # "How large would the error be if I just blindly guessed this cell's generic response?"
        # This is the residual after subtracting the "non-specific part" from the true value, or
        # the distance between the true value and the generic mean.
        err_baseline = torch.norm(global_mu_cell - delta_z_true, p=2, dim=1)

        # 4. Hinge Loss (Specificity Gain)
        # Goal: err_pred must be significantly smaller than err_baseline
        # Loss = max(0, err_pred - err_baseline + margin)
        # Meaning: If your prediction error is not at least margin smaller than the baseline error, I'll penalize you.
        loss = torch.relu(err_pred - err_baseline + self.margin).mean()

        return loss

def train_predict(
    model, dataloader,cell_means, optimizer, device,
    lambda_recon=0.5,       # Basic stability
    lambda_wmse=0.5,         # Alignment strength (important in later stage)
    lambda_cos=1.0,          # Alignment direction (important in early stage)
    lambda_z_align=1.0,      # Latent space displacement alignment
    lambda_spec=0.5,      # <--- [New weight] Specificity gain loss weight
    margin=0.1,
    k_topk=200,
    max_grad_norm=2.0):
    model.train()
    spec_loss_fn = CellSpecificGainLoss(cell_means, device, margin=margin)
    raw_model = model.module if hasattr(model, "module") else model
    dec = _get_submodule(raw_model, "L_decoder"); dec.eval()
    enc = _get_submodule(raw_model, "L_encoder"); enc.eval()
    meters = {k: 0.0 for k in ["loss", "delta_topk_cos", "wmse", "z_align", "spec_loss"]}
    n = 0
    for batch in tqdm(dataloader):
        # Data loading
        control = batch['control'].to(device).float()
        label = batch['label'].to(device).float()
        RNAseq = batch['rna'].to(device).float()
        input_ids    = batch['input_ids'].to(device)
        attention    = batch['attention_mask'].to(device)
        cell_ids = batch['cell_id']
        optimizer.zero_grad(set_to_none=True)
        # 1. Forward
        x_pred, z_pred, delta_z_pred, z_base = model(
            RNAseq, control, input_ids, attention)
        # 2. Get true target (Stop Gradient)
        with torch.no_grad():
            _, z_drug_mu, _ = enc(label)
            delta_z_true = (z_drug_mu - z_base).detach() # Latent space true displacement
        # 3. Calculate differential signal (based on true control)
        delta_true = label - control
        delta_pred = x_pred - control
        # --- Loss calculation ---

        # A. Expression space: balance between direction and strength
        # Direction (Cosine)
        loss_cos = delta_topk_cosine_loss_pure(delta_pred, delta_true, k=k_topk, eps=1e-8)
        # Strength (WMSE)
        loss_wmse = delta_weighted_mse(x_pred, label, control, q=0.8, p=2.0)
        # B. Latent space: specificity comparison (correction: displacement vs displacement)
        loss_z_align = F.mse_loss(delta_z_pred, delta_z_true)
        loss_spec = spec_loss_fn(delta_z_pred, delta_z_true, cell_ids)
        # E. Stability reconstruction
        loss_recon = F.l1_loss(x_pred, label)
        # 1. Prediction error (Model Error)
        # 4. Dynamic weight adjustment (optional: manual or progress-based adjustment)
        # Suggestion: As progress increases, the relative impact of lambda_wmse can be raised
        loss = (
            lambda_recon    * loss_recon +
            lambda_cos      * loss_cos +
            lambda_wmse     * loss_wmse +
            lambda_z_align  * loss_z_align +
            lambda_spec     * loss_spec
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Statistical recording...
        meters["delta_topk_cos"] += loss_cos.item()
        meters["wmse"] += loss_wmse.item()
        meters["z_align"] += loss_z_align.item()
        meters["spec_loss"] += loss_spec.item()
        meters["loss"] += loss.item()
        n += 1
    return {k: v/n for k, v in meters.items()}

def _get_submodule(m, name):
    return getattr(m.module, name) if hasattr(m, "module") else getattr(m, name)

def ddp_sum_list(vals, device):
    t = torch.tensor(vals, device=device, dtype=torch.float32)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t.tolist()

@torch.no_grad()
def evaluate_epoch_metrics(model, dataloader, mean_dict, device, epoch, out_csv_path, k_systema=20):
    model.eval()
    
    # Determine distributed state
    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    
    # Define the metrics we want to track
    metrics_keys = ["pearson_delta_top100", "r2_delta_top100", f"systema_pearson_{k_systema}"]
    local_sums = {k: 0.0 for k in metrics_keys}
    local_counts = {k: 0.0 for k in metrics_keys}

    # Preprocess mean_dict to current device
    device_mean_dict = {str(k): v.to(device).float() for k, v in mean_dict.items()}
    match_total = 0
    disable_tqdm = (rank != 0)
    for batch in tqdm(dataloader, desc=f"Eval Epoch {epoch}", disable=disable_tqdm, leave=False):   
        control = torch.nan_to_num(batch['control'].to(device).float())
        x_true  = torch.nan_to_num(batch['label'].to(device).float())
        RNAseq  = batch['rna'].to(device).float()
        input_ids    = batch['input_ids'].to(device)
        attention    = batch['attention_mask'].to(device)
        cell_identifiers = batch['det_plate']

        x_pred, z_pred, delta_z_pred, z_base = model(
            RNAseq, control, input_ids, attention)
        x_pred = torch.nan_to_num(x_pred).float()

        delta_true = x_true - control
        delta_pred = x_pred - control
        
        for i in range(x_true.size(0)):
            # Compatibility handling: ensure key matches the dictionary
            raw_id = cell_identifiers[i]
            c_name = str(raw_id.item()) if isinstance(raw_id, torch.Tensor) else str(raw_id)
            match_total += 1

            # --- A. Potency (Delta Top-100) ---
            d_t = delta_true[i]
            d_p = delta_pred[i]
            topk_idx = torch.topk(d_t.abs(), k=100).indices
            
            dt_np = d_t[topk_idx].cpu().numpy()
            dp_np = d_p[topk_idx].cpu().numpy()
            
            if dt_np.std() > 1e-8 and dp_np.std() > 1e-8:
                local_sums["pearson_delta_top100"] += pearsonr(dt_np, dp_np)[0]
                local_sums["r2_delta_top100"] += r2_score(dt_np, dp_np)
                local_counts["pearson_delta_top100"] += 1
                local_counts["r2_delta_top100"] += 1

            # --- B. Specificity (Systema) ---
            if c_name in device_mean_dict:
                group_mean = device_mean_dict[c_name]
                r_t = x_true[i] - group_mean
                r_p = x_pred[i] - group_mean
                sys_idx = torch.topk(r_t.abs(), k=k_systema).indices
                rt_k = r_t[sys_idx].cpu().numpy()
                rp_k = r_p[sys_idx].cpu().numpy()
                
                s_key = f"systema_pearson_{k_systema}"
                if rt_k.std() > 1e-8 and rp_k.std() > 1e-8:
                    local_sums[s_key] += pearsonr(rt_k, rp_k)[0]
                    local_counts[s_key] += 1
            elif rank == 0 and match_total % 500 == 0:
                # Sample print keys not found for troubleshooting
                print(f"\n[Warning] Key '{c_name}' not found in mean_dict. First 20 chars of dict keys: {list(device_mean_dict.keys())[0][:20]}...")
    # --- DDP aggregation logic ---
    # Construct Tensor sequence: [sums..., counts...]
    stat_vec = torch.tensor(
        [local_sums[k] for k in metrics_keys] + [local_counts[k] for k in metrics_keys],
        device=device, dtype=torch.float32
    )
    
    if is_ddp:
        dist.all_reduce(stat_vec, op=dist.ReduceOp.SUM)
    
    # Calculate final average values
    n = len(metrics_keys)
    global_sums = stat_vec[:n].cpu().numpy()
    global_counts = stat_vec[n:].cpu().numpy()
    
    final_results = {"epoch": epoch}
    for idx, key in enumerate(metrics_keys):
        final_results[key] = global_sums[idx] / global_counts[idx] if global_counts[idx] > 0 else 0.0

    # Only Rank 0 prints and writes to file
    if rank == 0:
        df_row = pd.DataFrame([final_results])
        df_row.to_csv(out_csv_path, mode='a', header=not os.path.isfile(out_csv_path), index=False)
    return final_results

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12352'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def remove_module_prefix(state_dict):
    return { (k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items() }

def ddp_mean_dict(d, device):
    """
    All-reduce metrics across each rank for averaging; d: {name: float}
    """
    keys = sorted(d.keys())
    t = torch.tensor([d[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return {k: float(v) for k, v in zip(keys, t.tolist())}

def ensure_dir(p):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def log_train_csv(path, epoch, meters):
    ensure_dir(path)
    header = ("epoch,loss,delta_topk_cos,"
              "wmse,z_align,spec_loss\n")
    line = (f"{epoch},{meters['loss']:.6f},"
            f"{meters['delta_topk_cos']:.6f},{meters['wmse']:.6f},"
            f"{meters['z_align']:.6f},{meters['spec_loss']:.6f}\n") # Reference spec_loss here
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        with open(path, "w") as f: f.write(header)
    with open(path, "a") as f: f.write(line)

# ========= Main Training =========
def main_worker(rank, world_size):
    try:
        setup(rank, world_size)
        local_device = torch.device(f"cuda:{rank}")
        # ====== Dataset / Dataloader ======
        train_dataset = PredictorDatasetDP2(
            train_meta_csv, L1000_exp_npy, L1000_ctrl_npy,
            exp_idx_map_path, ctrl_idx_map_path, RNA_parquet_path,
            drug_input_ids_npy, drug_attention_mask_npy, drug_idx_map_path)
        test_dataset = PredictorDatasetDP2(
            test_meta_drug, L1000_exp_npy, L1000_ctrl_npy,
            exp_idx_map_path, ctrl_idx_map_path, RNA_parquet_path,
            drug_input_ids_npy, drug_attention_mask_npy, drug_idx_map_path)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
        train_dataset, batch_size=256, sampler=train_sampler,
        num_workers=2, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2)
        # Evaluation: Only rank0 uses "non-distributed" full traversal; other ranks skip
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_drug_loader = DataLoader(
        test_dataset, batch_size=256, sampler=test_sampler,
        num_workers=2, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=2)

        # ====== Load pre-trained VAE encoder-decoder ======
        LINvae_model = LINCSVAE("cpu")
        lin_ckpt = torch.load(best_model_path_LINCSVAE, map_location="cpu")
        LINvae_model.load_state_dict(remove_module_prefix(lin_ckpt['vae_model_state_dict']))
        encoder = LINvae_model.encoder.to(local_device)
        decoder = LINvae_model.decoder.to(local_device)

        RNAvae_model = RNAVAE("cpu")
        rna_ckpt = torch.load(best_model_path_RNAVAE, map_location="cpu")
        RNAvae_model.load_state_dict(remove_module_prefix(rna_ckpt['vae_model_state_dict']))
        RNAencoder = RNAvae_model.encoder.to(local_device)

        # ====== Build prediction model ======
        predict_model = JointPerturbationPredictor(
            encoder, decoder, RNAencoder, molformer_path, local_device).to(local_device)
        for m in [predict_model.L_encoder, predict_model.L_decoder, predict_model.RNAencoder]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        predict_model = DDP(predict_model, device_ids=[rank], find_unused_parameters=False)
        def build_optimizer(model):
            raw_model = model.module if hasattr(model, "module") else model
            # Group parameters: Set different learning rates for different components like LoRA and Delta Head (optional)
            lora_params = []
            other_trainable_params = []
            for name, param in raw_model.named_parameters():
                if param.requires_grad:
                    if "lora_" in name:
                        lora_params.append(param)
                    else:
                        other_trainable_params.append(param)
            # 2. Construct parameter groups
            optim_groups = [
                {
                    "params": lora_params, 
                    "lr": 5e-5, 
                    "weight_decay": 1e-4
                },
                {
                    "params": other_trainable_params, 
                    "lr": 3e-4, 
                    "weight_decay": 1e-3
                }
            ]
            
            # Print debug info to ensure no parameters are missed
            trainable_count = sum(p.numel() for p in lora_params) + sum(p.numel() for p in other_trainable_params)
            if dist.get_rank() == 0:
                print(f"🔥 Optimizer initialized with {trainable_count:,} trainable parameters.")
                print(f"   - LoRA params: {sum(p.numel() for p in lora_params):,}")
                print(f"   - Custom head params: {sum(p.numel() for p in other_trainable_params):,}")
                
            return torch.optim.AdamW(optim_groups)

        # Calling method
        epochs = 200
        optimizer = build_optimizer(predict_model)
        warmup_epochs = 5
        main_epochs = epochs - warmup_epochs
        # Initial Warmup: linearly increase from 10% to 100%
        warmup_sch = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        # Cosine annealing
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=1e-6
        )
        # Combined scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_epochs]
        )
        # Loss monitoring variables
        best_loss = float('inf')
        loss_not_improved_count = 0
        lora_frozen = False
        # ====== Training hyperparameters ======
        wmse_start = 0.0   # Don't want MSE to interfere with direction learning at the start
        wmse_end = 0.5     # Target intensity at the end
        k_topk=200
        train_kwargs = dict(
            lambda_recon=0.5,lambda_cos=2.0,lambda_z_align=0.2,
            lambda_spec=0.2,margin=0.1,
            k_topk=k_topk,max_grad_norm=2.0)

        # ====== Logging & Model saving ======
        save_dir = './model_drug_cell_b116'
        os.makedirs(save_dir, exist_ok=True)
        train_log = os.path.join(save_dir, 'train_log.csv')
        eval_csv_path  =  os.path.join(save_dir, 'eval_log.csv')
        best_key = 'r2_delta_top100'  
        best_score = -1e9
        best_path  = os.path.join(save_dir, 'best_model.pt')
        cell_z_means_path = os.path.join(save_dir, 'cell_z_means.pt')
        cell_mean_map_path = os.path.join(save_dir, 'cell978_means.pt')
        # 1. Only the main process (Rank 0) is responsible for computation to avoid duplicate work
        if rank == 0:
            # 1. Training set latent space mean (for Spec Loss)
            if not os.path.exists(cell_z_means_path):
                print(f"[Rank {rank}] Computing TRAIN latent means (for Spec Loss) ...")
                # Must use train_dataset
                tmp_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
                z_means = precompute_cell_specific_latent_means(tmp_train_loader, encoder, device=local_device)
                torch.save(z_means, cell_z_means_path)
                del tmp_train_loader
            else:
                print(f"[Rank {rank}] Found existing train latent means.")

            # 2. Test set expression mean (for Systema evaluation)
            # Note: Here we explicitly compute based on test_dataset
            if not os.path.exists(cell_mean_map_path):
                print(f"[Rank {rank}] Computing TEST cell line means (for Systema evaluation) ...")
                # Correction point: Use test_dataset to generate evaluation baseline
                tmp_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
                # Internal logic: calculate means using det_plate or cell_id as key
                test_mean_dict = compute_cell_line_mean(tmp_test_loader, device='cpu') 
                torch.save(test_mean_dict, cell_mean_map_path)
                del tmp_test_loader
            else:
                print(f"[Rank {rank}] Found existing test cell mean map.")

        # 3. Barrier: Ensure all ranks can load the latest correct files
        dist.barrier()
        cell_means = torch.load(cell_z_means_path, map_location=local_device)
        mean_dict = torch.load(cell_mean_map_path, map_location=local_device)
        print(f"[Rank {rank}] Loaded TRAIN z_means and TEST mean_dict.")
        print(f"[Rank {rank}] Loaded mean_dict with {len(mean_dict)} cell lines.")
        # ====== Training loop ======
        for epoch in range(1, epochs+1):
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            # Annealing progress (also used for noise)
            t0 = epochs * 0.2
            k = 0.5  # Steepness
            progress_sigmoid = 1 / (1 + np.exp(-k * (epoch - t0)))

            current_wmse = wmse_start + (wmse_end - wmse_start) * progress_sigmoid
            # -- Train -- #
            meters_local = train_predict(
                model=predict_model, dataloader=train_loader,cell_means=cell_means, optimizer=optimizer, device=local_device,lambda_wmse=current_wmse,
                **train_kwargs
            )
            # All ranks merge and calculate mean
            meters = ddp_mean_dict(meters_local, device=local_device)
            if rank == 0:
                log_train_csv(train_log, epoch, meters)
                print(f"[Epoch {epoch:03d}] Train finished.")
            # -- Eval (rank0 full evaluation) -- #
            eval_metrics = evaluate_epoch_metrics(
                predict_model, test_drug_loader, mean_dict, 
                local_device, epoch, eval_csv_path, k_systema=20)
            if rank == 0:
                score = eval_metrics.get(best_key, 0.0)
                if score > best_score:
                    best_score = score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': predict_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_key': best_key,
                        'best_score': best_score,
                        'eval_metrics': eval_metrics,
                    }, best_path)
                    print(f">>> New Best Model Saved: {best_key} = {best_score:.4f}")
                if epoch % 10 == 0:
                    ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': predict_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"[Epoch {epoch:03d}] Saved checkpoint -> {ckpt_path}")
            scheduler.step()
            current_loss = meters['loss']
            if current_loss < (best_loss - 1e-6):
                best_loss = current_loss
                loss_not_improved_count = 0
            else:
                loss_not_improved_count += 1
            
            if loss_not_improved_count >= 5 and not lora_frozen:
                if rank == 0:
                    print(f"--- Loss hasn't dropped for 5 epochs. Freezing LoRA---")
                lora_frozen = True
            if lora_frozen:
                optimizer.param_groups[0]['lr'] = 0.0
            if rank == 0:
                lr_lora = optimizer.param_groups[0]['lr']
                lr_head = optimizer.param_groups[1]['lr']
                print(f"--- Epoch {epoch:03d} Summary ---")
                print(f"    Loss: {current_loss:.6f} | Best: {best_loss:.6f}")
                print(f"    LR_LoRA: {lr_lora:.2e} | LR_Head: {lr_head:.2e}")
                
                # Reminder: If learning rate is already very low, consider stopping manually
                if lr_head < 1.1e-6:
                    print("[Note] Learning rate is near eta_min. Training is approaching convergence.")    
        cleanup()
    except Exception as e:
        print(f"[Rank {rank}] Fatal error: {e}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    world_size = 2
    mp.spawn(main_worker, nprocs=world_size, args=(world_size,))