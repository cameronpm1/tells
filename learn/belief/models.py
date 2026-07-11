import itertools

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

class football_NN(nn.Module):

	def __init__(self,
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :

		super().__init__()

		self.loss = FootballMSE()
		self.val_loss = FootballMSE()
		
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
		
													
		
	def forward(self, x):

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

class football_VAE_NN(nn.Module):
	def __init__(self, input_channels, output_channels, latent_dim=16):
		super().__init__()

		self.loss = FootballVAEMSE()
		self.val_loss = FootballMSE()

		self.linear1 = nn.Linear(input_channels, 512)
		self.linear2 = nn.Linear(512, 1024)
		self.linear3 = nn.Linear(1024, 4096)
		self.linear6 = nn.Linear(4096, 1024)
		self.linear7 = nn.Linear(1024, 64)

		self.vae = VAELayer(64, latent_dim)
		self.decoder = nn.Linear(latent_dim, output_channels)

	def forward(self, x, stochastic=False):
		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear7(x))

		z, mu, logvar = self.vae(x, stochastic=stochastic)
		pred = self.decoder(z)

		return pred, mu, logvar

class predator_prey_VAE_NN(nn.Module):
	def __init__(self, input_channels, output_channels, latent_dim=16):
		super().__init__()

		self.loss = PredPreyPermutationInvariantVAEMSE()
		self.val_loss = PredPreyPermutationInvariantMSE()

		self.linear1 = nn.Linear(input_channels, 512)
		self.linear2 = nn.Linear(512, 1024)
		self.linear3 = nn.Linear(1024, 4096)
		self.linear6 = nn.Linear(4096, 1024)
		self.linear7 = nn.Linear(1024, 64)

		self.vae = VAELayer(64, latent_dim)
		self.decoder = nn.Linear(latent_dim, output_channels)

	def forward(self, x, stochastic=False):
		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear7(x))

		z, mu, logvar = self.vae(x, stochastic=stochastic)
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

    def forward(self, x, stochastic=False):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = nn.functional.relu(self.linear6(x))
        x = nn.functional.relu(self.linear7(x))

        z, mu, logvar = self.vae(x, stochastic=stochastic)
        pred = self.decoder(z)

        return pred, mu, logvar

class predator_prey_NN(nn.Module):

	def __init__(self, 
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :
		
		super().__init__()

		self.loss = PredPreyPermutationInvariantMSE()
		self.val_loss = PredPreyPermutationInvariantMSE()
		
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
		
													
		
	def forward(self, x):

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

class PredPreyPermutationInvariantMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target):
		return self.permutation_invariant_loss(pred, target)

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, num_frames * 2 * dim)
		target: (batch, num_frames * 2 * dim)
		"""
		# one shared direct/swapped teammate assignment across all
		# time frames, so slot identity is consistent between the
		# previous and current state estimates
		num_frames = pred.shape[1] // 4

		pred = pred.view(-1, num_frames, 2, 2)
		target = target.view(-1, num_frames, 2, 2)

		# direct assignment
		loss1 = ((pred - target) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)

		# swapped assignment
		loss2 = ((pred - target.flip(2)) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)

		# take minimum per sample
		loss = torch.min(loss1, loss2)

		return loss.mean()


class PredPreyPermutationInvariantVAEMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target, mu, logvar, beta_kl=1):

		kl_loss = -0.5 * torch.sum(
			1 + logvar - mu.pow(2) - logvar.exp(),
			dim=1
		)
		kl_loss = kl_loss.mean() * beta_kl

		total_loss = kl_loss + self.permutation_invariant_loss(pred, target)

		return total_loss, kl_loss #kl1+kl2

	def permutation_invariant_loss(
		self,
		recon,
		target,
		return_parts=False,
	):
		# one shared direct/swapped teammate assignment across all
		# time frames, so slot identity is consistent between the
		# previous and current state estimates
		num_frames = recon.shape[1] // 4

		recon = recon.view(-1, num_frames, 2, 2)
		target = target.view(-1, num_frames, 2, 2)

		mse_direct = ((recon - target) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)
		mse_swapped = ((recon - target.flip(2)) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)

		recon_loss = torch.min(mse_direct, mse_swapped)

		return recon_loss.mean()

class DronesPermutationInvariantMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target):
		return self.permutation_invariant_loss(pred, target)

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, num_frames * 2 * dim)
		target: (batch, num_frames * 2 * dim)
		"""
		# one shared direct/swapped teammate assignment across all
		# time frames, so slot identity is consistent between the
		# previous and current state estimates
		num_frames = pred.shape[1] // 6

		pred = pred.view(-1, num_frames, 2, 3)
		target = target.view(-1, num_frames, 2, 3)

		# direct assignment
		loss1 = ((pred - target) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)

		# swapped assignment
		loss2 = ((pred - target.flip(2)) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)

		# take minimum per sample
		loss = torch.min(loss1, loss2)

		return loss.mean()


class DronesPermutationInvariantVAEMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target, mu, logvar, beta_kl=1):

		kl_loss = -0.5 * torch.sum(
			1 + logvar - mu.pow(2) - logvar.exp(),
			dim=1
		)
		kl_loss = kl_loss.mean() * beta_kl

		total_loss = kl_loss + self.permutation_invariant_loss(pred, target)

		return total_loss, kl_loss #kl1+kl2

	def permutation_invariant_loss(
		self,
		recon,
		target,
		return_parts=False,
	):
		# one shared direct/swapped teammate assignment across all
		# time frames, so slot identity is consistent between the
		# previous and current state estimates
		num_frames = recon.shape[1] // 6

		recon = recon.view(-1, num_frames, 2, 3)
		target = target.view(-1, num_frames, 2, 3)

		mse_direct = ((recon - target) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)
		mse_swapped = ((recon - target.flip(2)) ** 2).mean(dim=3).sum(dim=2).sum(dim=1)

		recon_loss = torch.min(mse_direct, mse_swapped)

		return recon_loss.mean()

class FootballPermutationInvariantMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

		perms = list(itertools.permutations(range(4)))
		self.register_buffer('team_perms', torch.tensor(perms), persistent=False)

	def forward(self, pred, target):
		return self.permutation_invariant_loss(pred, target)

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, 2 * (4 * 2 + 6))
		target: (batch, 2 * (4 * 2 + 6))

		layout: [team_prev, ball_prev, team_curr, ball_curr]
		"""
		team_size = 4 * 2
		ball_size = 6

		team_prev_pred, ball_prev_pred, team_curr_pred, ball_curr_pred = torch.split(
			pred, [team_size, ball_size, team_size, ball_size], dim=1
		)
		team_prev_tgt, ball_prev_tgt, team_curr_tgt, ball_curr_tgt = torch.split(
			target, [team_size, ball_size, team_size, ball_size], dim=1
		)

		team_prev_pred = team_prev_pred.view(-1, 4, 2)
		team_curr_pred = team_curr_pred.view(-1, 4, 2)
		team_prev_tgt = team_prev_tgt.view(-1, 4, 2)
		team_curr_tgt = team_curr_tgt.view(-1, 4, 2)

		# one shared teammate permutation across both states, so slot
		# identity is consistent between the previous and current estimates
		best_team_loss = None
		for perm in self.team_perms:
			prev_loss = ((team_prev_pred[:, perm] - team_prev_tgt) ** 2).mean(dim=2).sum(dim=1)
			curr_loss = ((team_curr_pred[:, perm] - team_curr_tgt) ** 2).mean(dim=2).sum(dim=1)
			combined_loss = prev_loss + curr_loss

			if best_team_loss is None:
				best_team_loss = combined_loss
			else:
				best_team_loss = torch.minimum(best_team_loss, combined_loss)

		team_loss = best_team_loss.mean()

		ball_loss = (
			self._ball_cross_entropy(ball_prev_pred, ball_prev_tgt)
			+ self._ball_cross_entropy(ball_curr_pred, ball_curr_tgt)
		)

		return team_loss + ball_loss

	def _ball_cross_entropy(self, pred_logits, target_onehot):
		target_idx = target_onehot.argmax(dim=1)
		return F.cross_entropy(pred_logits, target_idx)

class FootballMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target):
		return self.loss(pred, target)

	def loss(self, pred, target):
		"""
		pred:   (batch, 2 * (4 * 2 + 6))
		target: (batch, 2 * (4 * 2 + 6))

		layout: [team_prev, ball_prev, team_curr, ball_curr]
		"""
		team_size = 4 * 2
		ball_size = 6

		team_prev_pred, ball_prev_pred, team_curr_pred, ball_curr_pred = torch.split(
			pred, [team_size, ball_size, team_size, ball_size], dim=1
		)
		team_prev_tgt, ball_prev_tgt, team_curr_tgt, ball_curr_tgt = torch.split(
			target, [team_size, ball_size, team_size, ball_size], dim=1
		)

		team_loss = F.mse_loss(team_prev_pred, team_prev_tgt) + F.mse_loss(team_curr_pred, team_curr_tgt)

		ball_loss = (
			self._ball_cross_entropy(ball_prev_pred, ball_prev_tgt)
			+ self._ball_cross_entropy(ball_curr_pred, ball_curr_tgt)
		)

		return team_loss + ball_loss

	def _ball_cross_entropy(self, pred_logits, target_onehot):
		target_idx = target_onehot.argmax(dim=1)
		return F.cross_entropy(pred_logits, target_idx)

class FootballVAEMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

		self.reconstruction_loss = FootballMSE()

	def forward(self, pred, target, mu, logvar, beta_kl=0.01):

		kl_loss = -0.5 * torch.sum(
			1 + logvar - mu.pow(2) - logvar.exp(),
			dim=1
		)
		kl_loss = kl_loss.mean() * beta_kl

		total_loss = kl_loss + self.reconstruction_loss(pred, target)

		return total_loss, kl_loss

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