import torch
import torch.nn as nn
import copy
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

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.fc1(x)
        x = self.act(self.norm(x))
        return x + res

class ResidualStack(nn.Module):
    def __init__(self,dim,n_blocks):
        super(ResidualStack,self).__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(n_blocks)])
    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        return x

class RNAvae_encoder(nn.Module):
    def __init__(self, device):
        super(RNAvae_encoder, self).__init__()
        self.device = device
        self.input_dim = 10085
        # Shared part of encoder
        encoder = [
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            ResidualStack(512,8)
        ]
        self.encoder = nn.Sequential(*encoder).to(device)
        # mu part
        self.encoder_mu = nn.Sequential(
            nn.Linear(512, 256)).to(device)
        # logvar part
        self.encoder_logvar = nn.Sequential(
            nn.Linear(512, 256)).to(device)
    def sample_latent(self, mu, logvar):
        """Sample latent space with reparametrization trick. First convert to std, sample normal(0,1) and get Z."""
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Sample from standard normal distribution
        return mu + eps * std
    def forward(self, input_cell_gex):
        # Shared part
        x = self.encoder(input_cell_gex)
        # Calculate mu and logvar separately
        en_mu = self.encoder_mu(x)
        en_logvar = self.encoder_logvar(x)
        # Sample z from mu and logvar
        z = self.sample_latent(en_mu, en_logvar)
        return z, en_mu, en_logvar

class RNAvae_decoder(nn.Module):
    def __init__(self, device):
        super(RNAvae_decoder, self).__init__()
        decoder = [
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    ResidualStack(512,8),
                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 10085)
                ]
        self.decoder = nn.Sequential(*decoder).to(device)
    def forward(self,z):
        reconstruction = self.decoder(z)
        return reconstruction

class RNAVAE(nn.Module):
    def __init__(self,device):
        super(RNAVAE, self).__init__()
        self.encoder = RNAvae_encoder(device=device)
        self.decoder = RNAvae_decoder(device=device)
        self.noise = GaussianNoise(sigma=0.1, device=device)

    def forward(self, x):
        x = self.noise(x)
        z,en_mu,en_logvar = self.encoder(x)
        return self.decoder(z),en_mu,en_logvar
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = RNAVAE(device=device)
    batch_size = 10
    input_data = torch.randn(batch_size, 10085).to(device)
    rec,en_mu,en_logvar = vae(input_data)
    print("rec",rec.shape)
    print("en_mu",en_mu.shape)
    print("en_logvar",en_logvar.shape)
    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")