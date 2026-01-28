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
# Only define "constants" or "paths" globally, do not load any large data
train_meta_csv = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/sh_p/split_cell/sh_meta_s1_train.csv"
test_meta_csv = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/sh_p/split_cell/sh_meta_s1_test.csv"
L1000_exp_npy          = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/sh_p/L1000_exp.npy"
L1000_ctrl_npy         = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/sh_p/L1000_ctrl.npy"
exp_idx_map_path       = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/sh_p/exp_idx_map.pkl"
ctrl_idx_map_path      = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/sh_p/ctrl_idx_map.pkl"
RNA_parquet_path       = "/home/liwenyuan/Desktop/2025program/data/uni/RNAseq.parquet"
sh_embed_PPI_csv =  "/home/liwenyuan/Desktop/2025program/data/uni/ensg_PPI_emb_thr700_d256_p1.0_q0.5_l2.csv"
sh_embed_seq_npy = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/emb_tokens_first_all.npy"
sh_embed_seq_pkl = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/id2idx_ensg_seq2_all.pkl"
best_model_path_LINCSVAE = "../result/model_ckpt_L_random/epoch_184.pt"
best_model_path_RNAVAE     = "../result/model_checkpoints_RNAseq/best_model.pt"



# Only import class definitions at the top, do not instantiate
from LINCSvae import LINCSVAE
from RNAvae import RNAVAE
from aethercell_sh import JointPerturbationPredictor_sh
from dataloader_all import PredictorDatasetDP2_sh

def compute_cell_line_mean(dataloader, device):
    """
    Compute per-cell-line mean Label vector (Centroids).
    For VRAM safety, the accumulation process is performed on CPU.
    """
    print("Computing Per-Cell-Line Mean Label (on CPU)...")
    
    # Store cumulative sum and count: {cell_line: {'sum': tensor, 'count': int}}
    stats = {}

    # No need to compute gradients
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Grouping Centroids"):
            # Get data, transfer directly to CPU for processing, to avoid GPU VRAM usage
            # Note: Here we compute the mean of X_true, not Delta
            x_true = batch['label'].float() 
            x_true = torch.nan_to_num(x_true)
            
            cell_lines = batch['det_plate']
            
            # Iterate through samples and accumulate
            for i, cell_name in enumerate(cell_lines):
                if isinstance(cell_name, torch.Tensor):
                    cell_name = cell_name.item()
                
                # Extract vector
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
            # Compute mean
            mean_val = data['sum'] / data['count']
            # Store in dictionary (still on CPU, will transfer to GPU during evaluation)
            mean_dict[cell_name] = mean_val
            # print(f"  - {cell_name}: {data['count']} samples") # Can comment out if log is too long
    
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
            labels = batch['label'].to(device).float() # Use actual perturbed values
            control = batch['control'].to(device).float()
            labels = torch.nan_to_num(labels)
            control  = torch.nan_to_num(control)
            _,z_control_mu,_ = encoder(control)
            _, z_drug_mu, _ = encoder(labels)
            delta_z_true = (z_drug_mu - z_control_mu)
            delta_z_true = delta_z_true.cpu()
            for i, cell_id in enumerate(cells):
                # Handle Tensor type ID
                if isinstance(cell_id, torch.Tensor):
                    cell_id = cell_id.item()
                    
                if cell_id not in cell_sums:
                    cell_sums[cell_id] = torch.zeros_like(delta_z_true[i])
                    cell_counts[cell_id] = 0
                
                cell_sums[cell_id] += delta_z_true[i]
                cell_counts[cell_id] += 1
        # Compute mean
        cell_means = {}
        for cid in cell_sums:
            cell_means[cid] = cell_sums[cid] / cell_counts[cid]
            
    print(f"Computed means for {len(cell_means)} cells.")
    return cell_means

def delta_weighted_mse(x_pred, x_true, x_ctrl, q=0.8, p=2.0, eps=1e-8):
    # q: quantile within each sample to use as threshold (e.g., 80th percentile)
    # p: amplification factor for large Δ weights; p=2 is commonly used
    d_true = x_true - x_ctrl
    d_pred = x_pred - x_ctrl
    s = d_true.abs()  # Signal intensity
    # Threshold within each sample (e.g., 80th percentile)
    thr = torch.quantile(s, q, dim=1, keepdim=True)
    w = (s / (thr + eps))**p          # Larger Δ gets larger weight
    w = w / (w.mean(dim=1, keepdim=True) + eps)  # Normalize to stabilize loss scale
    return (w * (d_pred - d_true).pow(2)).mean()


def delta_topk_cosine_loss_pure(delta_pred, delta_true, k=200, eps=1e-8):
    """
    Pure delta space Top-K loss
    delta_pred: x_pred - xhat_ctrl
    delta_true: label - control
    """
    k = min(k, delta_true.size(1))
    # Lock in the genes with the most extreme response in Ground Truth
    idx = delta_true.abs().topk(k, dim=1).indices
    
    d_true_k = torch.gather(delta_true, 1, idx)
    d_pred_k = torch.gather(delta_pred, 1, idx)
    
    # Normalize and compute cosine
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
        # Or simply save the dictionary and dynamically construct batches in forward
        self.cell_means = cell_means

    def forward(self, delta_z_pred, delta_z_true, cell_ids):
        
        # 1. Build Baseline for current Batch (cell-type generic stress response)
        # We need to extract the mean for the current batch from the pre-computed dictionary and stack to [B, 256]
        batch_means_list = []
        for cid in cell_ids:
            if isinstance(cid, torch.Tensor):
                cid = cid.item()
            # Fetch from dictionary, use all-zero fallback if not encountered (rare case)
            mean_vec = self.cell_means.get(cid, torch.zeros_like(delta_z_pred[0].cpu()))
            batch_means_list.append(mean_vec)
            
        # Stack and move to GPU
        # global_mu_cell shape: [B, 256]
        # Use detach() to ensure we don't backpropagate updates to this mean (it's a constant)
        global_mu_cell = torch.stack(batch_means_list).to(self.device).detach()

        # 2. Compute Prediction Error
        # Model prediction vs ground truth
        err_pred = torch.norm(delta_z_pred - delta_z_true, p=2, dim=1)
        
        # 3. Compute Baseline Error (Background Noise)
        # "How large would the error be if I just guessed the generic cell response?"
        # This is the residual after subtracting the "non-specific component" from the true value,
        # or it represents the distance between the true value and the generic mean.
        err_baseline = torch.norm(global_mu_cell - delta_z_true, p=2, dim=1)
        
        # 4. Hinge Loss (Specificity Gain)
        # Objective: err_pred must be significantly smaller than err_baseline
        # Loss = max(0, err_pred - err_baseline + margin)
        # Meaning: If your prediction error is not significantly smaller than baseline error by at least margin, I penalize you.
        loss = torch.relu(err_pred - err_baseline + self.margin).mean()
        
        return loss

def train_predict(
    model, dataloader,cell_means, optimizer, device,
    lambda_recon=0.5,       # Basic stability
    lambda_wmse=0.5,         # Alignment intensity (important in later stages)
    lambda_cos=1.0,          # Alignment direction (important in early stages)
    lambda_z_align=1.0,      # Latent space displacement alignment
    lambda_spec=0.5,      # <--- [NEW WEIGHT] Specificity Gain Loss weight
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
        ppi_emb = batch['sh_PPI_embedding'].to(device).float()
        seq_emb = batch["sh_seq_embedding"].to(device).float()
        cell_ids = batch['cell_id']
        optimizer.zero_grad(set_to_none=True)
        # 1. Forward pass
        x_pred, z_pred, delta_z_pred, z_base = model(RNAseq, control, ppi_emb, seq_emb)

        # 2. Get ground truth (Stop Gradient)
        with torch.no_grad():
            _, z_drug_mu, _ = enc(label)
            delta_z_true = (z_drug_mu - z_base).detach() # Latent space true displacement
        # 3. Compute differential signal (based on true control)
        delta_true = label - control
        delta_pred = x_pred - control
        # --- Loss Computation ---

        # A. Expression space: balance of direction and intensity
        # Direction (Cosine)
        loss_cos = delta_topk_cosine_loss_pure(delta_pred, delta_true, k=k_topk, eps=1e-8)
        # Intensity (WMSE)
        loss_wmse = delta_weighted_mse(x_pred, label, control, q=0.8, p=2.0)
        # B. Latent space: specificity contrast (corrected: displacement vs displacement)
        loss_z_align = F.mse_loss(delta_z_pred, delta_z_true)
        loss_spec = spec_loss_fn(delta_z_pred, delta_z_true, cell_ids)
        # E. Stability reconstruction
        loss_recon = F.l1_loss(x_pred, label)
        # 1. Prediction Error (Model Error)
        # 4. Dynamic weight adjustment (optional: manual or progress-based)
        # Suggestion: As progress increases, the relative influence of lambda_wmse can be enhanced
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

        # Statistics recording...
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
    
    # Determine distributed status
    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    
    # Define metrics to track
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
        ppi_emb = batch['sh_PPI_embedding'].to(device).float()
        seq_emb = batch["sh_seq_embedding"].to(device).float()
        cell_identifiers = batch['det_plate']

        x_pred, _, _, _ = model(RNAseq, control, ppi_emb, seq_emb)
        x_pred = torch.nan_to_num(x_pred).float()

        delta_true = x_true - control
        delta_pred = x_pred - control
        
        for i in range(x_true.size(0)):
            # Compatible handling: ensure key can match dictionary
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
                # Sample-print missing Keys for troubleshooting
                print(f"\n[Warning] Key '{c_name}' not found in mean_dict. First 20 chars of dict keys: {list(device_mean_dict.keys())[0][:20]}...")
    # --- DDP Aggregation Logic ---
    # Construct Tensor sequence: [sums..., counts...]
    stat_vec = torch.tensor(
        [local_sums[k] for k in metrics_keys] + [local_counts[k] for k in metrics_keys],
        device=device, dtype=torch.float32
    )
    
    if is_ddp:
        dist.all_reduce(stat_vec, op=dist.ReduceOp.SUM)
    
    # Compute final average
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
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def remove_module_prefix(state_dict):
    return { (k[7:] if k.startswith('module.') else k): v for k,v in state_dict.items() }

def ddp_mean_dict(d, device):
    """
    Aggregate metrics from each rank using all-reduce and compute average; d: {name: float}
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
        train_dataset = PredictorDatasetDP2_sh(
            train_meta_csv,L1000_exp_npy, L1000_ctrl_npy,
            exp_idx_map_path, ctrl_idx_map_path,
            RNA_parquet_path,sh_embed_PPI_csv,sh_embed_seq_npy,sh_embed_seq_pkl)
        test_dataset = PredictorDatasetDP2_sh(
            test_meta_csv,L1000_exp_npy, L1000_ctrl_npy,
            exp_idx_map_path, ctrl_idx_map_path,
            RNA_parquet_path,sh_embed_PPI_csv,sh_embed_seq_npy,sh_embed_seq_pkl)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
        train_dataset, batch_size=128, sampler=train_sampler,
        num_workers=2, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=4)
        # Evaluation: only rank0 uses "non-distributed" complete traversal; other ranks skip
        test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_drug_loader = DataLoader(
        test_dataset, batch_size=128, sampler=test_sampler,
        num_workers=2, pin_memory=True, drop_last=False,
        persistent_workers=True, prefetch_factor=4)

        # ====== Load pretrained VAE encoder/decoder ======
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
        predict_model = JointPerturbationPredictor_sh(encoder,decoder,RNAencoder,local_device).to(local_device)
        for m in [predict_model.L_encoder, predict_model.L_decoder, predict_model.RNAencoder]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        predict_model = DDP(predict_model, device_ids=[rank], find_unused_parameters=False)
        def build_optimizer():
            params = [
                {'params': predict_model.module.tokenrize.parameters(),      'lr': 1e-4},
                {'params': predict_model.module.cross_attention.parameters(),'lr': 1e-4},
                {'params': predict_model.module.delta_head.parameters(),     'lr': 1e-4},
            ]
            return torch.optim.AdamW(params, weight_decay=1e-3)
        optimizer = build_optimizer()
        # ====== Training hyperparameters ======
        epochs = 100
        wmse_start = 0.0   # Initially don't want MSE to interfere with direction learning
        wmse_end = 0.3     # Final target intensity
        k_topk=200
        train_kwargs = dict(
            lambda_recon=0.5,lambda_cos=1.0,lambda_z_align=0.0,
            lambda_spec=0.0,margin=0.1,
            k_topk=k_topk,max_grad_norm=2.0)

        # ====== Logging & Model Saving ======
        save_dir = './model_sh_gene_b116'
        os.makedirs(save_dir, exist_ok=True)
        train_log = os.path.join(save_dir, 'train_log.csv')
        eval_csv_path  =  os.path.join(save_dir, 'eval_log.csv')
        best_key = 'r2_delta_top100'  
        best_score = -1e9
        best_path  = os.path.join(save_dir, 'best_model.pt')
        cell_z_means_path = os.path.join(save_dir, 'cell_z_means.pt')
        cell_mean_map_path = os.path.join(save_dir, 'cell978_means.pt')
        # 1. Only main process (Rank 0) computes to avoid redundant work
        if rank == 0:
            # 1. Training set latent space means (for Spec Loss)
            if not os.path.exists(cell_z_means_path):
                print(f"[Rank {rank}] Computing TRAIN latent means (for Spec Loss) ...")
                # Must use train_dataset
                tmp_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
                z_means = precompute_cell_specific_latent_means(tmp_train_loader, encoder, device=local_device)
                torch.save(z_means, cell_z_means_path)
                del tmp_train_loader
            else:
                print(f"[Rank {rank}] Found existing train latent means.")

            # 2. Test set expression means (for Systema evaluation)
            # Note: Here we explicitly compute based on test_dataset
            if not os.path.exists(cell_mean_map_path):
                print(f"[Rank {rank}] Computing TEST cell line means (for Systema evaluation) ...")
                # Correction: Use test_dataset to generate evaluation baseline
                tmp_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
                # Internal logic: compute mean with det_plate or cell_id as key
                test_mean_dict = compute_cell_line_mean(tmp_test_loader, device='cpu') 
                torch.save(test_mean_dict, cell_mean_map_path)
                del tmp_test_loader
            else:
                print(f"[Rank {rank}] Found existing test cell mean map.")

        # 3. Barrier: ensure all ranks can load the latest correct files
        dist.barrier()
        cell_means = torch.load(cell_z_means_path, map_location=local_device)
        mean_dict = torch.load(cell_mean_map_path, map_location=local_device)
        print(f"[Rank {rank}] Loaded TRAIN z_means and TEST mean_dict.")
        print(f"[Rank {rank}] Loaded mean_dict with {len(mean_dict)} cell lines.")
        # ====== Training Loop ======
        for epoch in range(1, epochs+1):
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            # Annealing progress (also used for noise)
            progress = epoch / max(1, int(epochs * 0.2))
            progress = min(1.0, progress)
            current_wmse = wmse_start + (wmse_end - wmse_start) * progress
            # —— Train —— #
            meters_local = train_predict(
                model=predict_model, dataloader=train_loader,cell_means=cell_means, optimizer=optimizer, device=local_device,lambda_wmse=current_wmse,
                **train_kwargs
            )
            # All ranks merge and compute mean
            meters = ddp_mean_dict(meters_local, device=local_device)
            if rank == 0:
                log_train_csv(train_log, epoch, meters)
                print(f"[Epoch {epoch:03d}] Train finished.")
            # —— Eval (rank0 full evaluation) —— #
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
        cleanup()
    except Exception as e:
        print(f"[Rank {rank}] Fatal error: {e}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    world_size = 2
    mp.spawn(main_worker, nprocs=world_size, args=(world_size,))