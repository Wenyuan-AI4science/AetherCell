from matplotlib import axis
import torch
import torch.nn as nn
from LINCSvae import LINCSVAE
from RNAvae import RNAVAE
from transformers import AutoModel, AutoTokenizer
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True, device='cuda'):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)
    
    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.std().detach() if self.is_relative_detach else self.sigma * x.std()
            sampled_noise = self.noise.to(x.device).repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 
    
class CrossAttention(nn.Module):
    def __init__(self, d_model=256, nhead=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, tokens, cell_embed):
        # tokens: (B, T, d_model) drug_tokens or gene_patches
        # cell_embed: (B, d_model)
        query = cell_embed.unsqueeze(1)  # (B, 1, d_model)
        attn_output, _ = self.cross_attn(query, tokens, tokens)
        attn_output = self.norm(attn_output + query)
        return attn_output.squeeze(1) 

class JointPerturbationPredictor(nn.Module):
    def __init__(self, LINCSencoder, LINCSdecoder,RNAencoder,molformer_path, device):
        super().__init__()
        self.device = device
        self.L_encoder = LINCSencoder.to(device)
        for param in self.L_encoder.parameters():
            param.requires_grad = False  # Freeze encoder
        self.L_encoder.eval() 
        self.L_decoder = LINCSdecoder.to(device)
        for param in self.L_decoder.parameters():
            param.requires_grad = False
        self.L_decoder.eval()
        self.RNAencoder = RNAencoder
        for param in self.RNAencoder.parameters():
            param.requires_grad = False
        self.RNAencoder.eval() 
        base_model = AutoModel.from_pretrained(molformer_path, trust_remote_code=True)
        lora_config = LoraConfig(
            r=16,                      # Low rank
            lora_alpha=4,              # Scaling factor
            target_modules=["query", "value"],  # Target modules to inject (attention sublayers in Molformer)
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # For feature extraction, not text generation
        )
        self.molformer = get_peft_model(base_model, lora_config).to(device)
        self.mlp_tokens = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),)
        self.cross_attention = CrossAttention(d_model=256, nhead=4).to(device)
        # self.fusion = ConcatMLPFusion(d_model=256).to(device)
        # self.noise = GaussianNoise(sigma=0.1, device=device)
        self.delta_head = nn.Sequential(
            nn.Linear(256 + 256, 256),  # [attn_fused, z_ctrl] -> Δz
            nn.GELU(), nn.Dropout(0.1),
            nn.LayerNorm(256),
        )
        self.use_stochastic_ctrl_train = True
        self.register_buffer('global_background_mu', torch.zeros(1, 256))  # Momentum mean
        self.momentum = 0.99
    def forward(self, RNAseq,control, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
        drug_tokens = outputs.last_hidden_state
        drug_tokens = self.mlp_tokens(drug_tokens)
        with torch.no_grad():
            cell_embed, cell_mu,cell_logvar = self.RNAencoder(RNAseq.to(self.device))  # (B, 256)
        attn_fused = self.cross_attention(drug_tokens, cell_mu)  # (B, 256)
        with torch.no_grad():
            z_ctrl, z_ctrl_mu, z_ctrl_logvar = self.L_encoder(control)      # Encoder frozen
        # Add noise only to conditional branch for Δz during training
        if self.training and self.use_stochastic_ctrl_train:
            z_ctrl_cond = z_ctrl
        else:
            z_ctrl_cond = z_ctrl_mu
        # Δz head: conditioned on drug×cell fusion + noisy control
        delta_z = self.delta_head(torch.cat([attn_fused, z_ctrl_cond], dim=1))
        # Residual synthesis: use clean baseline only
        z_base = z_ctrl_mu
        z_pred = z_base + delta_z
        x_pred = self.L_decoder(z_pred)                                  # (B, 978)
        return x_pred, z_pred, delta_z, z_base

if __name__ == "__main__":
    import re
    from peft import PeftModel

    def count_trainable_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📦 Total parameters: {total_params:,}")
        print(f"🧠 Trainable parameters: {trainable_params:,}")
        print(f"🧊 Frozen parameters: {total_params - trainable_params:,}")

    def check_lora_injection(peft_model):
        """
        Check if LoRA is correctly injected: count LoRA layers and print their module names
        """
        print("\n🔍 Checking LoRA injection...")
        injected_modules = []
        for name, module in peft_model.named_modules():
            # PEFT inserts sublayers named lora_A / lora_B in injected layers
            if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
                injected_modules.append(name)
        if injected_modules:
            print(f"✅ Detected {len(injected_modules)} LoRA-injected modules:")
            for n in injected_modules[:20]:  # Show first 20 modules only
                print(f"   • {n}")
            if len(injected_modules) > 20:
                print(f"   ... ({len(injected_modules)-20} more)")
        else:
            print("⚠️ No LoRA modules found! Check your `target_modules` naming.")

    # -------------------------
    device = 'cpu'
    LINvae_model = LINCSVAE(device)
    encoder = LINvae_model.encoder
    decoder = LINvae_model.decoder
    RNAvae = RNAVAE(device)
    RNAencoder = RNAvae.encoder
    molformer_path = "/home/liwenyuan/Desktop/2025program/uni2.0re/perturb_model/mini_molformer"

    model = JointPerturbationPredictor(
        encoder, decoder, RNAencoder, molformer_path, device
    ).to(device)

    count_trainable_parameters(model)

    # LoRA detection (only check molformer component)
    check_lora_injection(model.molformer)