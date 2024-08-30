import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, num_categories=10, latent_dimensionality=15):
        super(ConditionalVariationalAutoencoder, self).__init__()
        self.input_dim = input_dim 
        self.latent_dimensionality = latent_dimensionality  
        self.num_categories = num_categories  # Number of output categories
        self.hidden_size = 64  # Hidden layer size
        self.encoder = None  # Encoder network
        self.mean_layer = None  # Layer to compute mean of latent distribution
        self.logvariance_layer = None  # Layer to compute log variance of latent distribution
        self.decoder = None  # Decoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_categories, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(self.hidden_size, latent_dimensionality)
        self.logvariance_layer = nn.Linear(self.hidden_size, latent_dimensionality)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimensionality + num_categories, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, input_dim),
            nn.Sigmoid()  # Assuming input data is in the range [0, 1]
        )
    def forward(self, data_input, condition):
        cat_input = torch.cat([data_input, condition], dim=1)
        encoded = self.encoder(cat_input)
        mean_params = self.mean_layer(encoded)
        logvariance_params = self.logvariance_layer(encoded)
        latent_sample = sample_latent(mean_params, logvariance_params)
        new_latent = torch.cat([latent_sample, condition], dim=1)
        data_recon = self.decoder(new_latent)
        return data_recon, mean_params, logvariance_params

    def sample_latent(means, logvariances):
        return means + (torch.randn_like(std))*(orch.exp(0.5 * logvariances))

    def loss_function(reconstructions, originals, means, logvariances):
         recon_loss = F.binary_cross_entropy(reconstructions, originals, reduction="sum")
         kl_divergence = -0.5 * torch.sum(1 + logvariances - means.pow(2) - logvariances.exp())
         return recon + dkl