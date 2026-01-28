import torch
import torch.nn as nn
from LINCSvae import LINCSVAE
from RNAvae import RNAVAE

    
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

class ShTokenizer(nn.Module):
    """
    Maps sh_emb_PPI:(B,256) and sh_emb_seq:(B,1152) to tokens:(B, 1+t_seq, 256)
    - Token 0: from PPI vector
    - Remaining t_seq tokens: linearly expanded from sequence vector
    """
    def __init__(self, d_model=256, seq_dim=1152, ppi_dim=256, t_seq=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.t_seq   = t_seq
        # Normalization
        self.ppi_ln = nn.LayerNorm(ppi_dim)
        self.seq_ln = nn.LayerNorm(seq_dim)
        # Projection
        self.ppi_proj = nn.Sequential(
            nn.Linear(ppi_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Generate t_seq 256-dim tokens at once, then reshape
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, d_model * t_seq),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Token type embedding: 0=PPI, 1=SEQ
        self.token_type = nn.Parameter(torch.zeros(2, d_model))
        nn.init.trunc_normal_(self.token_type, std=0.02)
        # Output LayerNorm (stabilize training)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, sh_emb_PPI: torch.Tensor, sh_emb_seq: torch.Tensor) -> torch.Tensor:
        """
        Input:
            sh_emb_PPI: (B, 256)
            sh_emb_seq: (B, 1152)
        Output:
            tokens: (B, 1 + t_seq, 256)
        """
        B = sh_emb_PPI.size(0)
        # PPI -> (B, 256)
        ppi_tok = self.ppi_proj(self.ppi_ln(sh_emb_PPI))
        ppi_tok = ppi_tok + self.token_type[0]              # (B, 256)
        ppi_tok = self.out_ln(ppi_tok).unsqueeze(1)         # (B, 1, 256)
        # SEQ -> (B, t_seq, 256)
        seq_flat = self.seq_proj(self.seq_ln(sh_emb_seq))   # (B, t_seq*256)
        seq_tok  = seq_flat.view(B, self.t_seq, self.d_model)
        seq_tok  = seq_tok + self.token_type[1].unsqueeze(0).unsqueeze(1)  # (B, t_seq, 256)
        seq_tok  = self.out_ln(seq_tok)
        # Concatenate -> (B, 1+t_seq, 256)
        tokens = torch.cat([ppi_tok, seq_tok], dim=1)
        return tokens

class JointPerturbationPredictor_sh(nn.Module):
    def __init__(self, LINCSencoder, LINCSdecoder,RNAencoder,device):
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
        self.tokenrize = ShTokenizer(d_model=256, seq_dim=1152, ppi_dim=256, t_seq=4, dropout=0.1).to(device)
        self.cross_attention = CrossAttention(d_model=256, nhead=4).to(device)
        # self.fusion = ConcatMLPFusion(d_model=256).to(device)
        # self.noise = GaussianNoise(sigma=0.1, device=device)
        self.delta_head = nn.Sequential(
            nn.Linear(256 + 256, 256),  # [attn_fused, z_ctrl] -> Δz
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256))
        self.register_buffer('global_background_mu', torch.zeros(1, 256))  # Momentum mean
        self.momentum = 0.99
    def forward(self, RNAseq,control,sh_emb_PPI,sh_emb_seq):
        with torch.no_grad():
            cell_embed, cell_mu,cell_logvar = self.RNAencoder(RNAseq)  # (B, 256)
            z_ctrl, z_ctrl_mu, z_ctrl_logvar= self.L_encoder(control)      # Encoder frozen
        # Add noise only to conditional branch for Δz during training
        if self.training:
            z_ctrl_cond = z_ctrl
        else:
            z_ctrl_cond = z_ctrl_mu
        sh_tokens = self.tokenrize(sh_emb_PPI,sh_emb_seq)
        attn_fused = self.cross_attention(sh_tokens, cell_mu)
        delta_z = self.delta_head(torch.cat([attn_fused, z_ctrl_cond], dim=1))
        # Residual synthesis: use clean baseline only
        z_base = z_ctrl_mu 
        z_pred = z_base + delta_z
        x_pred = self.L_decoder(z_pred)                                    # (B, 978)
        return x_pred, z_pred, delta_z, z_base

if __name__ == "__main__":
    def count_trainable_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📦 Total parameters: {total_params:,}")
        print(f"🧠 Trainable parameters: {trainable_params:,}")
        print(f"🧊 Frozen parameters: {total_params - trainable_params:,}")

        # 可选：列出各部分的 trainable 状态
        # print("\n🔍 Parameter breakdown (name | shape | trainable):")
        # for name, param in model.named_parameters():
        #     print(f"{name:60} | {tuple(param.shape)} | {'✅' if param.requires_grad else '❌'}")

    # 使用方式（确保是 nn.DataParallel 包裹的 .module）
    device = 'cpu'
    LINvae_model = LINCSVAE(device)
    encoder = LINvae_model.encoder
    decoder = LINvae_model.decoder

    RNAvae = RNAVAE(device)
    RNAencoder = RNAvae.encoder

    predict_model = JointPerturbationPredictor_sh(encoder,decoder,RNAencoder,device)
    predict_model = predict_model.to(device)
    count_trainable_parameters(predict_model)
    