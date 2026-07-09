import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

class NN2CNN(nn.Module):

	def __init__(self, 
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :
		
		super().__init__()
		
		self.p_mc_dropout = p_mc_dropout

		self.linear1 = nn.Linear(input_channels,256) 
		self.linear2 = nn.Linear(256,1024) 
		self.linear3 = nn.Linear(1024,2048)
		self.linear4 = nn.Linear(2048,4096)
		# project to spatial latent
		self.linear5 = nn.Linear(4096, 256 * 8 * 8)

		# ---------- CNN decoder ----------
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8 → 16
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16 → 32
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),

			# 32 → 50 exactly
			nn.ConvTranspose2d(64, 32, kernel_size=19, stride=1, padding=0),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),

			nn.Conv2d(32, 1, kernel_size=3, padding=1),
			nn.Sigmoid()
		)
		
													
		
	def forward(self, x, stochastic=True):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = F.relu(self.linear3(x))
		x = F.relu(self.linear4(x))

		if stochastic:
			x = F.dropout(x, p=self.p_mc_dropout, training=True)

		x = self.linear5(x)
		x = x.view(x.size(0), 256, 8, 8)

		x = self.decoder(x)

		# center crop 128x128 -> 100x100
		#start = (x.size(-1) - 50) // 2
		#x = x[:, :, start:start+50, start:start+50]

		return x

class predator_prey_VAE_NN(nn.Module):
	def __init__(self, input_channels, output_channels, latent_dim=16):
		super().__init__()

		self.loss = PermutationInvariantVAEMSE()
		self.val_loss = PermutationInvariantMSE()

		self.linear1 = nn.Linear(input_channels, 512)
		self.linear2 = nn.Linear(512, 1024)
		self.linear3 = nn.Linear(1024, 4096)
		self.linear6 = nn.Linear(4096, 1024)
		self.linear7 = nn.Linear(1024, 64)

		self.vae = VAELayer(64, latent_dim)
		self.decoder = nn.Linear(latent_dim, output_channels)

	def forward(self, x):
		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear7(x))

		z, mu, logvar = self.vae(x)
		pred = self.decoder(z)

		return pred, mu, logvar

class drones_VAE_NN(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim=16):
        super().__init__()

        self.linear1 = nn.Linear(input_channels, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 4096)
        self.linear6 = nn.Linear(4096, 1024)
        self.linear7 = nn.Linear(1024, 64)

        self.vae = VAELayer(64, latent_dim)
        self.decoder = nn.Linear(latent_dim, output_channels)

    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = nn.functional.relu(self.linear6(x))
        x = nn.functional.relu(self.linear7(x))

        z, mu, logvar = self.vae(x)
        pred = self.decoder(z)

        return pred, mu, logvar

class predator_prey_NN(nn.Module):

	def __init__(self, 
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :
		
		super().__init__()

		self.loss = PermutationInvariantMSE()
		self.val_loss = PermutationInvariantMSE()
		
		self.p_mc_dropout = p_mc_dropout

		self.linear1 = nn.Linear(input_channels,512) 
		self.linear2 = nn.Linear(512,1024) 
		self.linear3 = nn.Linear(1024,4096)
		#self.linear4 = nn.Linear(4096,4096)
		#self.linear5 = nn.Linear(4096,4096)
		#self.linear6 = nn.Linear(4096,4096)
		self.linear6 = nn.Linear(4096,1024)
		self.linear7 = nn.Linear(1024,64) 
		self.linear8 = nn.Linear(64,output_channels) 
		
													
		
	def forward(self, x, stochastic=True):

		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		#x = nn.functional.relu(self.linear4(x))
		#x = nn.functional.relu(self.linear5(x))
		#x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear7(x))
		x = self.linear8(x)

		return x 

class PermutationInvariantMSE(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, pred, target):

		l1 = self.permutation_invariant_loss(pred[:,0:4], target[:,0:4])
		l2 = self.permutation_invariant_loss(pred[:,4:8], target[:,4:8])
		return l1 + l2

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, 4)
		target: (batch, 4)
		"""

		# reshape to (batch, 2, 3)
		pred = pred.view(-1, 2, 2)
		target = target.view(-1, 2, 2)

		# direct assignment
		loss1 = ((pred - target) ** 2).mean(dim=2).sum(dim=1)

		# swapped assignment
		loss2 = ((pred - target.flip(1)) ** 2).mean(dim=2).sum(dim=1)

		# take minimum per sample
		loss = torch.min(loss1, loss2)

		return loss.mean()

	def error(self, pred, target):
		"""
		pred:   (N, 6)
		target: (N, 6)

		Returns:
			scalar, sum over batch of minimum assignment distances
		"""
		e1 = self.two_by_two_error(pred[:,0:4].cpu().numpy(), target[:,0:4].cpu().numpy())
		e2 = self.two_by_two_error(pred[:,4:8].cpu().numpy(), target[:,4:8].cpu().numpy())
		return e1 + e2

	def two_by_two_error(self, pred, target):

		# Reshape to (N, 2, 3)
		pred = pred.reshape(-1, 2, 2)
		target = target.reshape(-1, 2, 2)

		# Direct assignment distances
		direct = (
			np.linalg.norm(pred[:, 0] - target[:, 0], axis=1) +
			np.linalg.norm(pred[:, 1] - target[:, 1], axis=1)
		)

		# Swapped assignment distances
		swapped = (
			np.linalg.norm(pred[:, 0] - target[:, 1], axis=1) +
			np.linalg.norm(pred[:, 1] - target[:, 0], axis=1)
		)

		# Take minimum per sample, then sum batch
		return np.minimum(direct, swapped).sum()


class PermutationInvariantVAEMSE(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, pred, target, mu, logvar, beta_kl=1e-2,):

		kl_loss = -0.5 * torch.sum(
			1 + logvar - mu.pow(2) - logvar.exp(),
			dim=1
		)
		kl_loss = kl_loss.mean() * beta_kl
		
		l1 = self.permutation_invariant_loss(pred[:,0:4], target[:,0:4])
		l2 = self.permutation_invariant_loss(pred[:,4:8], target[:,4:8])
		return l1 + l2 + kl_loss, kl_loss #kl1+kl2

	def permutation_invariant_loss(
		self,
		recon,
		target,
		return_parts=False,
	):

		recon = recon.view(-1, 2, 2)
		target = target.view(-1, 2, 2)

		mse_direct = ((recon - target) ** 2).mean(dim=2).sum(dim=1)
		mse_swapped = ((recon - target.flip(1)) ** 2).mean(dim=2).sum(dim=1)

		recon_loss = torch.min(mse_direct, mse_swapped)

		return recon_loss.mean()

	def error(self, pred, target):
		"""
		pred:   (N, 6)
		target: (N, 6)

		Returns:
			scalar, sum over batch of minimum assignment distances
		"""
		e1 = self.two_by_two_error(pred[:,0:4].cpu().numpy(), target[:,0:4].cpu().numpy())
		e2 = self.two_by_two_error(pred[:,4:8].cpu().numpy(), target[:,4:8].cpu().numpy())
		return e1 + e2

	def two_by_two_error(self, pred, target):

		# Reshape to (N, 2, 3)
		pred = pred.reshape(-1, 2, 2)
		target = target.reshape(-1, 2, 2)

		# Direct assignment distances
		direct = (
			np.linalg.norm(pred[:, 0] - target[:, 0], axis=1) +
			np.linalg.norm(pred[:, 1] - target[:, 1], axis=1)
		)

		# Swapped assignment distances
		swapped = (
			np.linalg.norm(pred[:, 0] - target[:, 1], axis=1) +
			np.linalg.norm(pred[:, 1] - target[:, 0], axis=1)
		)

		# Take minimum per sample, then sum batch
		return np.minimum(direct, swapped).sum()


class VAELayer(nn.Module):
    """
    A custom Variational Autoencoder bottleneck layer.
    Maps an incoming hidden representation to a latent distribution, 
    samples from it using the reparameterization trick, and prepares it for decoding.
    """
    def __init__(self, hidden_dim: int, latent_dim: int):
        super(VAELayer, self).__init__()
        
        # Parallel layers to parameterize the Gaussian latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        """
        Applies the reparameterization trick to allow backpropagation.
        """
        if self.training or stochastic:
            # Calculate standard deviation: std = exp(0.5 * logvar)
            std = torch.exp(0.5 * logvar)
            # Sample random noise epsilon from standard normal distribution
            eps = torch.randn_like(std)
            # Return the sampled latent vector
            return mu + eps * std
        else:
            # During evaluation/inference, bypass noise and use the mean directly
            return mu
			
    def forward(self, h: torch.Tensor, stochastic: bool = False):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar, stochastic=stochastic)

        return z, mu, logvar