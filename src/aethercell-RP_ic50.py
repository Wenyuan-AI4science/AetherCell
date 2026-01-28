import torch
import torch.nn as nn
import torch.nn.functional as F
from RNAvae import RNAVAE
from LINCSvae import LINCSVAE
from aethercell_drug import JointPerturbationPredictor
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
import os


class DDPPredictor(nn.Module):
    def __init__(self,
                 pretrained_model: JointPerturbationPredictor,
                 device: torch.device):
        super().__init__()
        self.device = device
        # 1) Replace original L_encoder with RNAVAE.encoder, load weights and freeze
        self.RNAencoder = pretrained_model.RNAencoder.to(device)
        for param in self.RNAencoder.parameters():
            param.requires_grad = False
        self.RNAencoder.eval()
        self.molformer = pretrained_model.molformer.to(device)
        for p in self.molformer.parameters():
            p.requires_grad = False
        self.molformer.eval()
        self.mlp_tokens = pretrained_model.mlp_tokens.to(device)
        for p in self.mlp_tokens.parameters():
            p.requires_grad = False
        self.mlp_tokens.eval()
        self.cross_attention = pretrained_model.cross_attention.to(device)
        for p in self.cross_attention.parameters():
            p.requires_grad = False
        self.cross_attention.eval()
        self.delta_head  = pretrained_model.delta_head.to(device)
        self.ddp = nn.Sequential(
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128,1),
        ).to(device)
    def forward(self, RNAseq, input_ids, attention_mask):
        # Drug molecule features
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mol_out = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
        drug_tokens = mol_out.last_hidden_state   # (B, T, D)
        drug_tokens = self.mlp_tokens(drug_tokens)
        with torch.no_grad():
            cell_embed, cell_mu,cell_logvar = self.RNAencoder(RNAseq.to(self.device))  # (B, 256)
        attn_fused = self.cross_attention(drug_tokens, cell_mu)
        delta_z = self.delta_head(torch.cat([attn_fused,cell_mu], dim=1))
        z = torch.concat([attn_fused,delta_z],axis=1)
        result = self.ddp(z)
        return result
    

if __name__ == "__main__":
    best_model_path_LINCSVAE = "../result/model_ckpt_L_random/epoch_184.pt"
    best_model_path_RNAVAE     = "../result/model_checkpoints_RNAseq/best_model.pt"
    used_ckpt = "./model_drug_cell_b116/epoch_050.pt"
    local_device = "cuda"
    molformer_path = "/home/liwenyuan/Desktop/2025program/uni2.0re/perturb_model/mini_molformer"
    def count_trainable_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📦 Total parameters: {total_params:,}")
        print(f"🧠 Trainable parameters: {trainable_params:,}")
        print(f"🧊 Frozen parameters: {total_params - trainable_params:,}")
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_k = k[7:]  # Remove "module." prefix
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict
    device = 'cpu'
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
    predict_model = JointPerturbationPredictor(encoder, decoder, RNAencoder, molformer_path, local_device).to(local_device)
    predict_ckpt = torch.load(used_ckpt, map_location=local_device)
    clean_predict_state = remove_module_prefix(predict_ckpt['model_state_dict'])
    predict_model .load_state_dict(clean_predict_state)  
    predict_model.to(local_device)
    new_model = DDPPredictor(
            pretrained_model=predict_model,
            device=device).to(device)
    count_trainable_parameters(new_model)