import torch
import torch.nn as nn

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

class LINCSvae_encoder(nn.Module):
    def __init__(self, device):
        super(LINCSvae_encoder, self).__init__()
        self.device = device
        self.input_dim = 978
        # Shared part of encoder
        encoder = [
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        ]
        self.encoder = nn.Sequential(*encoder).to(device)
        # mu part
        self.resblock = ResidualStack(512,4).to(device)
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
        x= self.resblock(x)
        en_mu = self.encoder_mu(x)
        en_logvar = self.encoder_logvar(x)
        # Sample z from mu and logvar
        z = self.sample_latent(en_mu, en_logvar)
        return z, en_mu, en_logvar
    

class LINCSvae_decoder(nn.Module):
    def __init__(self, device):
        super(LINCSvae_decoder, self).__init__()
        decoder1 = [
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
        self.decoder1 = nn.Sequential(*decoder1).to(device)
        self.resblock = ResidualStack(512,4).to(device)
        self.decoder2 = nn.Linear(512,978).to(device)
    def forward(self,z):
        z = self.decoder1(z)
        z = self.resblock(z)
        reconstruction = self.decoder2(z)
        return reconstruction
    
    
class LINCSVAE(nn.Module):
    def __init__(self,device):
        super(LINCSVAE, self).__init__()
        self.encoder = LINCSvae_encoder(device=device)
        self.decoder = LINCSvae_decoder(device=device)

    def forward(self, x):
        z,en_mu,en_logvar = self.encoder(x)
        return self.decoder(z),en_mu,en_logvar,z

