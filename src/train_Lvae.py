import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from LINCSvae import LINCSVAE
from RNAvae import RNAVAE
import os
import pandas as pd
import pickle
from torch.cuda.amp import autocast, GradScaler
import random, numpy as np
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
import math

def uncertainty_bounded_loss(z_pred, z_target_mu, z_target_logvar, beta=0.5):
    """
    Uncertainty-aware boundary loss
    z_pred:        Latent vector encoded from L1000
    z_target_mu:   Mean encoded from RNAseq
    z_target_logvar: Logvar encoded from RNAseq
    beta:          How many standard deviations of fluctuation allowed (Sigma Coefficient)
                   beta=1.0 corresponds to ~68% confidence interval
                   beta=2.0 corresponds to ~95% confidence interval (recommended)
    """
    # 1. Calculate standard deviation of RNAseq sigma
    # logvar = ln(sigma^2) -> sigma = exp(0.5 * logvar)
    z_target_std = torch.exp(0.5 * z_target_logvar)
    # 2. Calculate absolute distance between L1000 and RNAseq center
    # shape: [batch, latent_dim]
    dist = (z_pred - z_target_mu).abs()
    # 3. Dynamic radius: each dimension and sample has its own "valid range"
    radius = beta * z_target_std
    # 4. Hinge Loss: max(0, dist - radius)
    # Only penalize when exceeding radius, loss=0 when within range
    loss = torch.relu(dist - radius)
    # 5. Calculate mean (average over each dimension and sample)
    return loss.mean()

def delta_topk_cosine_loss_pure(delta_pred, delta_true, k=200, eps=1e-8):
    """
    Pure delta space Top-K loss
    delta_pred: x_pred - xhat_ctrl
    delta_true: label - control
    """
    k = min(k, delta_true.size(1))
    # Lock on genes with strongest response in ground truth
    idx = delta_true.abs().topk(k, dim=1).indices

    d_true_k = torch.gather(delta_true, 1, idx)
    d_pred_k = torch.gather(delta_pred, 1, idx)

    # Calculate cosine after normalization
    d_true_k = d_true_k - d_true_k.mean(dim=1, keepdim=True)
    d_pred_k = d_pred_k - d_pred_k.mean(dim=1, keepdim=True)

    cos = F.cosine_similarity(d_true_k, d_pred_k, dim=1)
    return (1.0 - cos).mean()


def train_epoch(
    model,
    rna_encoder,
    loader,
    optimizer,
    device,
    beta_kl_max=0.5,     # KL upper limit
    beta_progress=1.0,   # KL annealing progress [0,1]
    lambda_ctrl=1.0,
    lambda_delta=1.0,
    grad_clip=5.0,
    current_alpha = 1.0
):
    """
    Adapt your Dataset: (rna, l1000, control, is_perturb, RNAseq_name)
    - rna:        RNAseq baseline (anchor source)
    - l1000:      Current sample (could be control or drug)
    - control:    If drug, its paired control L1000; if itself is control, zero vector
    - is_perturb: 1=drug, 0=control
    - RNAseq_name: Not used in this function
    """
    model.train()
    rna_encoder.eval()
    for p in rna_encoder.parameters():
        p.requires_grad = False

    mse = nn.MSELoss()

    total_loss = total_recon = total_kl = 0.0
    total_ctrl = total_delta = 0.0
    steps = 0

    for batch in tqdm(loader, leave=False):
        steps += 1
        rna_x, l1000_x, l1000_ctrl_pair, is_perturb, _ ,_= batch
        rna_x            = rna_x.to(device).float()
        l1000_x          = l1000_x.to(device).float()
        l1000_ctrl_pair  = l1000_ctrl_pair.to(device).float()
        is_perturb          = is_perturb.to(device).long()
        # ---------- Forward ----------
        recon, mean, logvar, z_l1000 = model(l1000_x)
        with torch.no_grad():
            z_rna, mu_rna, logvar_rna = rna_encoder(rna_x)
        # 1) VAE loss (unified)
        recon_loss = mse(recon, l1000_x)
        beta_kl = beta_kl_max * float(beta_progress)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + beta_kl * kl_loss
        # 2) Control latent alignment (control only)
        ctrl_mask = (is_perturb == 0)
        if ctrl_mask.any():
            z_l1000 = mean[ctrl_mask]
            z_rna_mu = mu_rna[ctrl_mask].detach() 
            z_rna_logvar = logvar_rna[ctrl_mask].detach() 
            loss_ctrl = uncertainty_bounded_loss(
                z_l1000, 
                z_rna_mu, 
                z_rna_logvar, 
                beta=0.25
            )
            loss = loss + lambda_ctrl * loss_ctrl
        else:
            loss_ctrl = torch.tensor(0.0, device=device)
        # 3) Δ-consistency (drug only; using paired control)
        drug_mask = (is_perturb == 1)
        # Filter out without valid paired control (remove when control is zero vector in dataset)
        if drug_mask.any():
            valid_pair = (l1000_ctrl_pair.abs().sum(dim=1) >  1e-8)
            dm = drug_mask & valid_pair
        else:
            dm = drug_mask

        if dm.any():
            x_drug    = l1000_x[dm]
            xhat_drug = recon[dm]
            x_ctrl    = l1000_ctrl_pair[dm]
            delta_pred = xhat_drug - x_ctrl
            delta_true = x_drug -  x_ctrl
            loss_delta = delta_topk_cosine_loss_pure(delta_pred, delta_true, k=200, eps=1e-8)
            loss = loss + lambda_delta * loss_delta
        else:
            loss_delta = torch.tensor(0.0, device=device)

        # ---------- Backward and optimization ----------
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # ---------- Statistics ----------
        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()
        total_ctrl  += loss_ctrl.item()
        total_delta += loss_delta.item()

    denom = max(1, steps)
    logs = {
        "loss":        total_loss  / denom,
        "recon":       total_recon / denom,
        "kl":          total_kl    / denom,
        "ctrl_align":  total_ctrl  / denom,
        "delta_align": total_delta / denom,
        "beta_kl":     beta_kl,
    }
    return logs

@torch.no_grad()
def eval_topk_delta_mean(model, dataloader, device, topk=200, eps=1e-8):
    """
    Only for drug samples with paired control:
      Δ_true = x_drug - x_ctrl
      Δ_pred = xhat_drug - x_ctrl
    On the topk genes with largest |Δ_true|, calculate per-sample Pearson and R², then return their mean.
    Does not save to disk, does not return per-sample details.
    """
    model.eval()
    sum_p, sum_r2, n = 0.0, 0.0, 0

    for batch in tqdm(dataloader, leave=False):
        rna_x, l1000_x, l1000_ctrl_pair, is_perturb, _, _ = batch

        l1000_x         = torch.nan_to_num(l1000_x.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
        l1000_ctrl_pair = torch.nan_to_num(l1000_ctrl_pair.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
        is_perturb      = is_perturb.to(device).long()

        recon, mean, logvar, z_l1000 = model(l1000_x)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)

        valid_pair = (l1000_ctrl_pair.abs().sum(dim=1) > 1e-8)
        dm = (is_perturb == 1) & valid_pair
        idxs = torch.nonzero(dm, as_tuple=False).squeeze(1)

        for bi in idxs.tolist():
            d_true = (l1000_x[bi] - l1000_ctrl_pair[bi]).float()
            d_pred = (recon[bi]    - l1000_ctrl_pair[bi]).float()
            k = min(int(topk), d_true.numel())
            if k <= 1:
                continue  # Insufficient to calculate correlation/variance

            top_idx = torch.topk(d_true.abs(), k=k, dim=0).indices
            t = d_true[top_idx]
            p = d_pred[top_idx]
            # Pearson
            t_c = t - t.mean()
            p_c = p - p.mean()
            denom = torch.sqrt((t_c * t_c).sum()) * torch.sqrt((p_c * p_c).sum())
            pearson = ((t_c * p_c).sum() / (denom + 1e-8)).item()
            # R²
            sst = (t_c * t_c).sum().item()
            if sst < 1e-12:
                r2 = 0.0
            else:
                ssr = ((t - p) * (t - p)).sum().item()
                r2 = 1.0 - ssr / (sst + eps)
            sum_p += pearson
            sum_r2 += r2
            n += 1
    mean_p = float('nan') if n == 0 else (sum_p / n)
    mean_r2 = float('nan') if n == 0 else (sum_r2 / n)
    return mean_p, mean_r2

def test_vae(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    sum_pearson = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            rna_x, l1000_x, l1000_ctrl_pair, is_perturb, _,_ = batch
            l1000_x          = l1000_x.to(device).float()
            # ---------- Forward ----------
            recon, mean, logvar, z_l1000 = model(l1000_x)
            recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
            label = torch.nan_to_num(l1000_x, nan=0.0, posinf=0.0, neginf=0.0)
            # Flatten each sample in the batch to a 1D vector
            recon = recon.reshape(recon.size(0), -1)  # Shape: (batch_size, num_features)
            label = label.reshape(label.size(0), -1)  # Shape: (batch_size, num_features)
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

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[7:]  # Remove "module." prefix
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

_GLOBAL_CACHE = {
    "rna": {},    # key: RNA_csv path, val: {cell_iname -> np.ndarray(float32)}
    "lincs": {},  # key: LINCS_pkl path, val: {sample_id -> np.ndarray(float32)}
}

def load_rnaseq_dict(RNA_csv: str):
    """Pre-fetch each column (cell line) of RNAseq CSV as float32 vector, cache as {cell_iname: np.array}."""
    if RNA_csv in _GLOBAL_CACHE["rna"]:
        return _GLOBAL_CACHE["rna"][RNA_csv]
    df = pd.read_csv(RNA_csv, index_col=0)
    # Assume columns are cell line names, rows are genes (if first column is index, can also use df = pd.read_csv(RNA_csv, index_col=0))
    rna_dict = {col: df[col].astype(np.float32).to_numpy(copy=False) for col in df.columns}
    _GLOBAL_CACHE["rna"][RNA_csv] = rna_dict
    return rna_dict

def load_lincs_dict(LINCS_pkl: str):
    """Cache LINCS DataFrame (pickle) by row index sample_id -> vector."""
    if LINCS_pkl in _GLOBAL_CACHE["lincs"]:
        return _GLOBAL_CACHE["lincs"][LINCS_pkl]
    with open(LINCS_pkl, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        # Row index is sample_id, columns are 978 (or more) genes
        lincs_dict = {idx: row.values.astype(np.float32, copy=False) for idx, row in obj.iterrows()}
    elif isinstance(obj, dict):
        # Already a dict
        lincs_dict = {k: np.asarray(v, dtype=np.float32) for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported LINCS_pkl object type: {type(obj)}")
    _GLOBAL_CACHE["lincs"][LINCS_pkl] = lincs_dict
    return lincs_dict

class PretrainDataset(Dataset):
    """
    One training set + multiple test sets share one RNA/LINCS cache.
    Returns: (rna, l1000, control, is_perturb, RNAseq_name, L_name)
    """
    PERTURB_SET = {"trt_cp", "trt_sh", "trt_xpr", "trt_oe"}  # Others are treated as control

    def __init__(self, meta_csv: str, rna_dict: dict, lincs_dict: dict, drop_missing=True):
        self.meta = pd.read_csv(meta_csv)
        self.rna_dict = rna_dict
        self.lincs_dict = lincs_dict

        # Pre-fetch needed columns as numpy arrays to avoid slow DataFrame lookups in __getitem__
        needed_cols = ["cell_iname", "sample_id", "pert_type", "representative_control"]
        for c in needed_cols:
            if c not in self.meta.columns:
                raise KeyError(f"{meta_csv} missing required column: {c}")

        self.cell_iname = self.meta["cell_iname"].astype(str).values
        self.sample_id = self.meta["sample_id"].astype(str).values
        self.pert_type = self.meta["pert_type"].astype(str).values
        # Control may have missing values, use empty string as placeholder
        self.ctrl_id = self.meta["representative_control"].fillna("").astype(str).values

        # Dimension check and missing value filtering
        # Use the length of any LINCS vector as dimension (e.g., 978)
        any_vec = next(iter(lincs_dict.values()))
        self.gene_dim = int(any_vec.shape[0])

        if drop_missing:
            keep = np.ones(len(self.meta), dtype=bool)

            # Ensure sample_id exists in LINCS
            miss_sample = [i for i, sid in enumerate(self.sample_id) if sid not in lincs_dict]
            if len(miss_sample) > 0:
                keep[miss_sample] = False

            # Ensure RNA cell line exists (otherwise could use zeros, but filtering is more stable here)
            miss_cell = [i for i, cname in enumerate(self.cell_iname) if cname not in rna_dict]
            if len(miss_cell) > 0:
                keep[miss_cell] = False

            # Apply filtering
            self.cell_iname = self.cell_iname[keep]
            self.sample_id = self.sample_id[keep]
            self.pert_type = self.pert_type[keep]
            self.ctrl_id   = self.ctrl_id[keep]

    def __len__(self):
        return len(self.sample_id)

    def __getitem__(self, index):
        cname = self.cell_iname[index]
        sid   = self.sample_id[index]
        ptype = self.pert_type[index]
        cid   = self.ctrl_id[index]  # May be empty string

        # RNAseq (if missing, use zero vector as fallback to ensure batch can be concatenated)
        rna_vec = self.rna_dict.get(cname, None)
        if rna_vec is None:
            rna = torch.zeros(self.gene_dim, dtype=torch.float32)
        else:
            rna = torch.from_numpy(rna_vec)

        # L1000 sample
        l_vec = self.lincs_dict.get(sid, None)
        if l_vec is None:
            # Theoretically already filtered, this is a fallback
            l1000 = torch.zeros(self.gene_dim, dtype=torch.float32)
        else:
            l1000 = torch.from_numpy(l_vec)
        # Whether perturbed
        is_perturb = 1 if ptype in self.PERTURB_SET else 0
        # Control (only try to get corresponding control if perturbed)
        if is_perturb == 1 and cid and (cid in self.lincs_dict):
            c_vec = self.lincs_dict[cid]
            control = torch.from_numpy(c_vec)
        else:
            control = torch.zeros_like(l1000)
        return rna, l1000, control, is_perturb, cname, sid

def build_loaders(
    train_meta_csv: str,
    test_meta_csvs: dict,    # e.g., {"trt_cp": "...", "trt_sh": "...", "trt_xpr": "...", "trt_oe": "..."}
    RNA_csv: str,
    LINCS_pkl: str,
    batch_size_train: int = 512,
    batch_size_test: int  = 512,
    num_workers_train: int = 4,
    num_workers_test: int  = 2,
    pin_memory: bool = True,
):
    # Load only once
    rna_dict   = load_rnaseq_dict(RNA_csv)
    lincs_dict = load_lincs_dict(LINCS_pkl)

    # Training set (unified)
    train_ds = PretrainDataset(train_meta_csv, rna_dict, lincs_dict, drop_missing=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers_train > 0),
    )

    # Multiple test sets (by perturbation type)
    test_loaders = {}
    for name, meta_csv in test_meta_csvs.items():
        if (meta_csv is None) or (not os.path.exists(meta_csv)):
            # Skip non-existent
            continue
        ds = PretrainDataset(meta_csv, rna_dict, lincs_dict, drop_missing=True)
        kwargs = dict(
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers_test,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers_test > 0))
        if num_workers_test > 0:
            kwargs["prefetch_factor"] = 2  # Only pass when >0

        ld = DataLoader(ds, **kwargs)
        test_loaders[name] = ld

    return train_loader, test_loaders


train_meta = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/L3pretrain/pretrain_all.csv"
RNA_csv    = "/home/liwenyuan/Desktop/2025program/data/uni/L-RNAseq_p.csv"
LINCS_pkl  = "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/L3pretrain/merge_data.pkl"

test_meta_csvs = {
    "trt_cp": "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/L3pretrain/test_trt_cp.csv",
    "trt_sh": "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/L3pretrain/test_trt_sh.csv",
    "trt_xpr": "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/L3pretrain/test_trt_xpr.csv",
    "trt_oe" : "/home/liwenyuan/Desktop/2025program/uni2.0re/L3_per_data/L3pretrain/test_trt_oe.csv",
}

train_loader, test_loaders = build_loaders(
    train_meta_csv=train_meta,
    test_meta_csvs=test_meta_csvs,
    RNA_csv=RNA_csv,
    LINCS_pkl=LINCS_pkl,
    batch_size_train=512,
    batch_size_test=512,
    num_workers_train=4,
    num_workers_test=2,
    pin_memory=True,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae_model = RNAVAE(device)
best_model_path = '../results/model_ckpt_RNAseq/best_model.pt'
checkpoint = torch.load(best_model_path, map_location=device)
clean_state_dict = remove_module_prefix(checkpoint['vae_model_state_dict'])
vae_model.load_state_dict(clean_state_dict)
rna_encoder = vae_model.encoder.to(device)

Lvae = LINCSVAE(device=device).to(device)
num_epochs = 200
base_lr = 3e-4
optimizer = torch.optim.AdamW(Lvae.parameters(), lr=base_lr, weight_decay=1e-3)
warmup_epochs = int(0.10 * num_epochs)
main_epochs   = max(1, num_epochs - warmup_epochs)
warmup  = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,    # Start from 0.1×base_lr
    end_factor=1.0,      # Rise to base_lr
    total_iters=max(1, warmup_epochs)
)
cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=main_epochs,
    eta_min=base_lr * 0.10
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[warmup_epochs]
)


kl_warmup_frac = 0.05                      # KL annealing for first 5% epochs
warmup_steps = max(1, int(num_epochs * kl_warmup_frac))
best_pearson = -1e9
lambda_delta_start, lambda_delta_end = 1.0, 5.0 
alpha_start, alpha_end = 1.0, 0.2               

save_dir = '../results/model_ckpt_L'
os.makedirs(save_dir, exist_ok=True)
loss_log_csv = os.path.join(save_dir, "train_loss.csv")
eval_log_csv = os.path.join(save_dir, "eval_pcc.csv")

test_names = sorted(test_loaders.keys())

# If first write, create header
if (not os.path.exists(loss_log_csv)) or os.path.getsize(loss_log_csv) == 0:
    with open(loss_log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,loss,recon,kl,ctrl_align,delta_align,beta_kl,lambda_delta\n")

if (not os.path.exists(eval_log_csv)) or os.path.getsize(eval_log_csv) == 0:
    header = "epoch," + ",".join([f"PCC_{name}" for name in test_names]) + "\n"
    with open(eval_log_csv, "w", encoding="utf-8") as f:
        f.write(header)

delta_topk = 200
eval_delta_csv = os.path.join(save_dir, f"eval_delta_top{delta_topk}.csv")
# If first write, create header: epoch, P_top200_{set}, R2_top200_{set}, ...
if (not os.path.exists(eval_delta_csv)) or os.path.getsize(eval_delta_csv) == 0:
    header = "epoch," + ",".join([f"P_top{delta_topk}_{name},R2_top{delta_topk}_{name}" for name in test_names]) + "\n"
    with open(eval_delta_csv, "w", encoding="utf-8") as f:
        f.write(header)

for epoch in range(num_epochs):
    progress = min(1.0, (epoch + 1) / warmup_steps)
    lambda_delta = lambda_delta_start + (lambda_delta_end - lambda_delta_start) * progress
    current_alpha = alpha_start + (alpha_end - alpha_start) * progress
    # —— Train one epoch —— #
    logs = train_epoch(
        model=Lvae,
        rna_encoder=rna_encoder,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        beta_kl_max=0.5,
        beta_progress=progress,
        lambda_ctrl=25.0,
        lambda_delta=lambda_delta,
        current_alpha=current_alpha
    )
    scheduler.step()
    val_pcc = {}
    delta_means = {}
    for name, loader in test_loaders.items():
        val_pcc[name] = test_vae(Lvae, loader, device)
        mp, mr2 = eval_topk_delta_mean(Lvae, loader, device, topk=delta_topk)
        delta_means[name] = (mp, mr2)

    # —— Append to eval_delta_top200.csv —— #
    with open(eval_delta_csv, "a", encoding="utf-8") as f:
        row = [str(epoch+1)]
        for name in test_names:
            mp, mr2 = delta_means.get(name, (float('nan'), float('nan')))
            row.append(f"{mp:.6f}" if not math.isnan(mp) else "nan")
            row.append(f"{mr2:.6f}" if not math.isnan(mr2) else "nan")
        f.write(",".join(row) + "\n")
    # —— Append to txt log —— #
    with open(loss_log_csv, "a", encoding="utf-8") as f:
        f.write(f"{epoch+1},{logs['loss']:.6f},{logs['recon']:.6f},{logs['kl']:.6f},"
                f"{logs['ctrl_align']:.6f},{logs['delta_align']:.6f},{logs['beta_kl']:.6f},"
                f"{lambda_delta:.6f}\n")

    # 2) Test PCC for each test set (test results only)
    with open(eval_log_csv, "a", encoding="utf-8") as f:
        pcc_vals = [f"{val_pcc.get(name, float('nan')):.6f}" for name in test_names]
        f.write(f"{epoch+1}," + ",".join(pcc_vals) + "\n")

    # —— Periodic save —— #
    if (epoch + 1) % 1 == 0:
        torch.save(
            {"epoch": epoch + 1, "vae_model_state_dict": Lvae.state_dict()},
            os.path.join(save_dir, f"epoch_{epoch+1}.pt")
        )