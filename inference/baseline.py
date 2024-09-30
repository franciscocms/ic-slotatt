import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from utils.guide import minimize_entropy_of_sinkhorn, sinkhorn, assert_shape

import logging
import warnings
warnings.filterwarnings("ignore")

main_dir = os.path.abspath(__file__+'/../../')

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_coords(b_s, resolution=(128, 128)):
  # x-cor and y-cor setting
  nx, ny = resolution
  x = np.linspace(0, 1, nx)
  y = np.linspace(0, 1, ny)
  xv, yv = np.meshgrid(x, y)
  xv = torch.from_numpy(np.reshape(xv, [b_s, nx, nx, 1])).to(device, dtype=torch.float32)
  yv = torch.from_numpy(np.reshape(yv, [b_s, ny, ny, 1])).to(device, dtype=torch.float32)
  return xv, yv

class SlotAttention(nn.Module):
  def __init__(self, num_slots, dim = 64, iters = 3, eps = 1e-8, hidden_dim = 128):
    super().__init__()
    self.num_slots = num_slots
    self.iters = iters
    self.eps = eps
    self.scale = dim ** -0.5
    
    self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
    self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

    self.to_q = nn.Linear(dim, dim, bias=False)
    self.to_k = nn.Linear(dim, dim, bias=False)
    self.to_v = nn.Linear(dim, dim, bias=False)

    self.gru = nn.GRUCell(dim, dim)

    hidden_dim = max(dim, hidden_dim)

    self.fc1 = nn.Linear(dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, dim)

    self.norm_input  = nn.LayerNorm(dim)
    self.norm_slots  = nn.LayerNorm(dim)
    self.norm_pre_ff = nn.LayerNorm(dim)

    self.mlp_weight_input = nn.Linear(dim, 1)
    self.mlp_weight_slots = nn.Linear(dim, 1)

    #self.grid_proj = nn.Linear(2, dim)
    #self.grid_enc = nn.Linear(dim, dim)
    
    self.epsilon = 1e-8

    #self.relu = nn.ReLU()
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.eps = torch.tensor(1e-8, device=self.device)

  def forward(self, inputs, num_slots = None):
    
    b_s, n, d = inputs.shape # 'inputs' have shape (b_s, W*H, dim)
    n_s = num_slots if num_slots is not None else self.num_slots
    
    mu = self.slots_mu.expand(b_s, n_s, -1)
    sigma = self.slots_sigma.expand(b_s, n_s, -1)
    slots = torch.distributions.Normal(mu, torch.abs(sigma) + self.eps).rsample() # 'slots' have shape (1, n_s, slot_dim=64)
 
    inputs = self.norm_input(inputs)      
    k, v = self.to_k(inputs), self.to_v(inputs) # 'k' and 'v' have shape (1, 16384, 64)

    latent_resolution = (32, 32)
    
    x_pos, y_pos = build_coords(1, latent_resolution)
    grid = torch.cat([x_pos, y_pos], dim=-1)
    grid = torch.flatten(grid, -3, -2) # 'grid' has shape (1, 16384, 2)
    grid_per_slot = grid.unsqueeze(-3).repeat(1, n_s, 1, 1) # 'grid_per_slot' has shape (1, n_s, 16384, 2)

    a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * n_s # 'a' shape (b_s, W*H)

    for iteration in range(self.iters):      
      slots_prev = slots
      slots = self.norm_slots(slots) # 'slots' shape (b_s, n_s, slot_dim)
      b = self.mlp_weight_slots(slots).squeeze(-1).softmax(-1) * n_s  # 'b' shape (b_s, n_s)
      
      q = self.to_q(slots) # 'q' shape (1, n_s, 64)

      attn_logits = torch.cdist(k, q)      
      attn_logits, p, q = minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=3, n_mesh_iters=4) 
    
      attn, _, _ = sinkhorn(attn_logits, a, b, u=p, v=q) # 'attn' shape (1, n, n_s)
      attn = attn.permute(0, 2, 1) # 'attn' shape (1, n_s, n)

      updates = torch.matmul(attn, v)

      # Compute the center of mass of each slot attention mask.
      # 'attn' has shape (1, n_s, 16384)
      # 'grid' has shape (1, 16384, 2)
      # 'slot_pos' has shape (1, n_s, 2)
      slot_pos = torch.einsum("...qk,...kd->...qd", attn, grid)
      slot_pos = slot_pos.unsqueeze(-2)

      slots = self.gru(
          updates.view(b_s * n_s, d),
          slots_prev.view(b_s * n_s, d),
      )

      slots = slots.view(b_s, n_s, d)
      assert_shape(slots.size(), (b_s, n_s, d))
      slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
      assert_shape(slots.size(), (b_s, n_s, d))    
    return slots, slot_pos.reshape((b_s, n_s, 2)), attn#, scales

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0) # (1, 2, 128, 128)
  grid = grid.astype(np.float32)
  grid = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))
  return grid

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.
    Args:
    hidden_size: Size of input feature dimension.
    resolution: Tuple of integers specifying width and height of grid.
    """
    super().__init__()        
    self.embedding = nn.Linear(4, hidden_size, bias=True)
    self.embedding = self.embedding.to(device)
    self.grid = build_grid(resolution)     
    self.grid = self.grid.to(device)
    #logger.info(f'grid: {self.grid.dtype}')   

  def forward(self, inputs):
    #logger.info(f'embedding: {self.embedding.weight.dtype}')
    grid = self.embedding(self.grid)
    #logger.info(inputs.shape) # (1, 128, 128, 64)
    #logger.info(grid.shape)   # (1, 128, 128, 8)
    new_embd = inputs + grid
    return new_embd

class Encoder(nn.Module):
  def __init__(self, resolution, hid_dim):
    super().__init__()

    self.encoder_sa = []
    in_channels = 3
    self.encoder_sa += [nn.Conv2d(in_channels, 64, 5, 1, 2), nn.ReLU(inplace=True)]
    for c in range(2): self.encoder_sa += [nn.Conv2d(64, 64, 5, 2, 2), nn.ReLU(inplace=True)]

    self.encoder_sa += [nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(inplace=True)]
    self.encoder_sa = nn.Sequential(*self.encoder_sa)
    
    resolution = (32, 32)
    self.encoder_pos = SoftPositionEmbed(hid_dim, resolution).to(device)

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def forward(self, x):
    x = x.to(self.device)    
    x = self.encoder_sa(x)
    x = x.permute(0,2,3,1) # B, W, H, C -> put C in last dimension
    x = self.encoder_pos(x)                 
    # spatial flatten -> flatten the spatial dimensions
    x = torch.flatten(x, 1, 2) # B, W*H, C -> 64 features of size w*H

    return x

"""Slot Attention-based auto-encoder for object discovery."""
class Baseline(nn.Module):
  def __init__(self, resolution, num_iterations, hid_dim, stage, num_slots, save_slots=False):
    """Builds the Slot Attention-based auto-encoder.
    Args:
    resolution: Tuple of integers specifying width and height of input image.
    num_slots: Number of slots in Slot Attention.
    num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.hid_dim = hid_dim
    self.resolution = resolution
    self.num_slots = num_slots
    self.num_iterations = num_iterations
    self.stage = stage
    self.save_slots = save_slots

    assert self.stage in ["train", "eval"], "stage must be either 'train' or 'eval'"

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    self.encoder_cnn = Encoder(self.resolution, self.hid_dim)

    self.mlp = nn.Sequential(
       nn.Linear(hid_dim, hid_dim), # used to be hid_dim, hid_dim
       nn.ReLU(inplace=True),
       nn.Linear(hid_dim, hid_dim)
    )

    self.slot_attention = SlotAttention(
        num_slots=self.num_slots,
        dim=hid_dim,
        iters = self.num_iterations,
        eps = 1e-8, 
        hidden_dim = 128)

    target_dim = 11 # number of target dimensions in ICSA synthetic dataset
    
    self.mlp_classifier = nn.Sequential(
      nn.Linear(hid_dim, hid_dim),
      nn.ReLU(),
      nn.Linear(hid_dim, target_dim),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    x.to(self.device)
    x = self.encoder_cnn(x)
    x = nn.LayerNorm(x.shape[1:]).to(self.device)(x)
    self.features_to_slots = self.mlp(x)
    self.slots, self.slot_pos, attn = self.slot_attention(self.features_to_slots, num_slots=self.num_slots)
    preds = self.mlp_classifier(self.slots)
    
    if not self.save_slots: return preds
    else: return preds, self.slots, attn

      
    
