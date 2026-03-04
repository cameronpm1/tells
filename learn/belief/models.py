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

class NN(nn.Module):

	def __init__(self, 
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :
		
		super().__init__()
		
		self.p_mc_dropout = p_mc_dropout

		self.linear1 = nn.Linear(input_channels,512) 
		self.linear2 = nn.Linear(512,1024) 
		self.linear3 = nn.Linear(1024,4096)
		self.linear4 = nn.Linear(4096,4096)
		self.linear5 = nn.Linear(4096,4096)
		#self.linear6 = nn.Linear(4096,4096)
		self.linear6 = nn.Linear(4096,1024)
		self.linear7 = nn.Linear(1024,64) 
		self.linear8 = nn.Linear(64,output_channels) 
		
													
		
	def forward(self, x, stochastic=True):

		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear4(x))
		x = nn.functional.relu(self.linear5(x))
		#x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear7(x))
		x = self.linear8(x)

		return x 

class PermutationInvariantMSE(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, pred, target):
		l1 = self.permutation_invariant_loss(pred[:,0:6], target[:,0:6])
		l2 = self.permutation_invariant_loss(pred[:,6:], target[:,6:])
		return l1 + l2

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, 6)
		target: (batch, 6)
		"""

		# reshape to (batch, 2, 3)
		pred = pred.view(-1, 2, 3)
		target = target.view(-1, 2, 3)

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

		# Reshape to (N, 2, 3)
		pred = pred.reshape(-1, 2, 3)
		target = target.reshape(-1, 2, 3)

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