import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy


class VAE(nn.Module):
    def __init__(self, in_features, hidden_features, z_dims):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.z_dims = z_dims
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(self.hidden_features, self.z_dims)
        self.var = nn.Linear(self.hidden_features, self.z_dims)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dims, self.hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_features, self.in_features),
            nn.Sigmoid()
        )

    def sampling(self, mean, var):
        std = torch.exp(0.5 * var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def forward(self, x):
        x = self.encoder(x.view(-1, self.in_features))
        mean = self.mean(x)
        var = self.var(x)
        z = self.sampling(mean, var)
        x = self.decoder(z)
        return x, mean, var


class VAELoss(nn.Module):
    def __init__(self, in_features):
        super(VAELoss, self).__init__()
        self.in_features = in_features

    def forward(self, input_tensor, target_tensor, mean, var):
        BCE = binary_cross_entropy(input_tensor, target_tensor.view(-1, self.in_features), reduction='sum')
        KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
        return BCE + KLD
