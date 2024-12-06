import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

import os
import matplotlib.pyplot as plt
import glob

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../../'))

from utils.distributions import CategoricalVals, TruncatedNormal, Mixture
from utils.var import Variable
from main.setup import params
from utils.guide import minimize_entropy_of_sinkhorn, sinkhorn, assert_shape, to_int

import logging
import warnings

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger("train")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def cosine_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - torch.bmm(x, y.transpose(1, 2))

class SlotAttention(nn.Module):
  def __init__(self, num_slots, dim = 64, iters = 3, eps = 1e-8, hidden_dim = 128):
    super().__init__()
    self.num_slots = num_slots
    self.iters = iters
    self.eps = eps
    self.scale = dim ** -0.5
    
    self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
    self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

    nn.init.xavier_uniform_(
       self.slots_mu, gain=nn.init.calculate_gain("linear")
       )
    nn.init.xavier_uniform_(
       self.slots_sigma, gain=nn.init.calculate_gain("linear")
       )

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
    
    self.eps = torch.tensor(1e-8, device=device)

  def forward(self, inputs, num_slots = None):
    
    b_s, num_inputs, d = inputs.shape # 'inputs' have shape (b_s, W*H, dim)
    n_s = num_slots if num_slots is not None else self.num_slots
    
    mu = self.slots_mu.expand(b_s, n_s, -1)
    sigma = self.slots_sigma.expand(b_s, n_s, -1)
    slots = torch.distributions.Normal(mu, torch.abs(sigma) + self.eps).rsample() # 'slots' have shape (1, n_s, slot_dim=64)
 
    inputs = self.norm_input(inputs)      
    k, v = self.to_k(inputs), self.to_v(inputs) # 'k' and 'v' have shape (1, 16384, 64)

    if params["strided_convs"]: latent_resolution = (32, 32)
    else: latent_resolution = (128, 128)

    a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * n_s # 'a' shape (b_s, W*H)

    for iteration in range(self.iters):

      if params["running_type"] == "inspect": logging.info(f"\nSA iteration {iteration}")
      
      slots_prev = slots
      slots = self.norm_slots(slots) # 'slots' shape (b_s, n_s, slot_dim)
      b = self.mlp_weight_slots(slots).squeeze(-1).softmax(-1) * n_s  # 'b' shape (b_s, n_s)
      
      q = self.to_q(slots) # 'q' shape (1, n_s, 64)

      attn_logits = cosine_distance(k, q)      
      attn_logits, p, q = minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=params["mesh_lr"], n_mesh_iters=params["mesh_iters"]) 
          
      attn, _, _ = sinkhorn(attn_logits, a, b, u=p, v=q) # 'attn' shape (1, n, n_s)
      attn = attn.permute(0, 2, 1) # 'attn' shape (1, n_s, n)
      updates = torch.matmul(attn, v)

      slots = self.gru(
          updates.view(b_s * n_s, d),
          slots_prev.view(b_s * n_s, d),
      )

      slots = slots.view(b_s, n_s, d)
      assert_shape(slots.size(), (b_s, n_s, d))
      slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
      assert_shape(slots.size(), (b_s, n_s, d))    
    return slots, attn

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0) # (1, 2, 128, 128)
  grid = grid.astype(np.float32)
  grid = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)
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
    self.grid = build_grid(resolution)        

  def forward(self, inputs):
    grid = self.embedding(self.grid)
    #logging.info(inputs.shape) # (1, 128, 128, 64)
    #logging.info(grid.shape)   # (1, 128, 128, 8)
    new_embd = inputs + grid
    return new_embd

class Encoder(nn.Module):
  def __init__(self, resolution, hid_dim):
    super().__init__()
    
    self.encoder_sa = []
    in_channels = 3 if params['dataset'] == '2Dobjects' else 4
    self.encoder_sa += [nn.Conv2d(in_channels, 64, 5, 1, 2), nn.ReLU(inplace=True)]
    for c in range(2):
      if params["strided_convs"]: self.encoder_sa += [nn.Conv2d(64, 64, 5, 2, 2), nn.ReLU(inplace=True)]
      else: self.encoder_sa += [nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(inplace=True)]
    self.encoder_sa += [nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(inplace=True)]
    self.encoder_sa = nn.Sequential(*self.encoder_sa)
    
    #self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)
    if params["strided_convs"]: resolution = (32, 32)
    else: resolution = (128, 128)
    self.encoder_pos = SoftPositionEmbed(64, resolution)

  def forward(self, x):
    x = x.to(device)    

    x = self.encoder_sa(x)
    x = x.permute(0,2,3,1) # B, W, H, C -> put C in last dimension
    x = self.encoder_pos(x)                 
    # spatial flatten -> flatten the spatial dimensions
    x = torch.flatten(x, 1, 2) # B, W*H, C -> 64 features of size w*H

    return x

"""Slot Attention-based auto-encoder for object discovery."""
class InvSlotAttentionGuide(nn.Module):
  def __init__(self, resolution, num_iterations, hid_dim, stage):
    """Builds the Slot Attention-based auto-encoder.
    Args:
    resolution: Tuple of integers specifying width and height of input image.
    num_slots: Number of slots in Slot Attention.
    num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.hid_dim = hid_dim
    self.resolution = resolution
    self.num_slots = 0
    self.num_iterations = num_iterations
    self.shape_vals = ["ball", "square"]
    self.color_vals = ["red", "green", "blue"]
    self.size_vals = ["small", "medium", "large"]
    self.stage = stage
    assert self.stage in ["train", "eval"], "stage must be either 'train' or 'eval'"
    self.current_trace = []
    self.is_train = True

    self.prior_stddevs = params["prior_stddevs"]

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

    self.prop_nets = nn.ModuleDict() 

    self.batch_idx = 0
    self.step = 0

    self.attention_overlap = torch.tensor(0., device=device)
  
  def add_proposal_net(self, var, out_dim):
    add_flag = False
    if var.proposal_distribution == "categorical": last_activ = nn.Softmax(dim=-1) # size, shape, color, material
    elif var.proposal_distribution == "bernoulli": last_activ = nn.Sigmoid() # mask
    elif var.proposal_distribution == "normal": 
       if var.name in ["x", "y"]: last_activ = nn.Tanh() # position 
       else: last_activ = nn.Sigmoid() # pose
    elif var.proposal_distribution == "mixture": last_activ = nn.Identity()
    else: raise ValueError(f"Unknown distribution: {var.proposal_distribution}")

    if params["pos_from_attn"] == "attn-masks": input_dim = 1 if var.address in ["locX", "locY"] else self.hid_dim
    
    proposal_net = nn.Sequential(
      nn.Linear(input_dim, self.hid_dim), nn.ReLU(),
      #nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(),
      nn.Linear(self.hid_dim, out_dim), last_activ
      )
    
    #logger.info(proposal_net)
     
    self.prop_nets[var.address] = proposal_net.to(device) 

    if var.address in self.prop_nets: add_flag = True 
    if not add_flag: logging.info(f"ERROR: proposal net for site {var.name} was not added!")
  
  def infer_step(self, variable, obs=None, proposal_distribution=None): 

    # evaluation
    if isinstance(variable, str): 
      variable_name = variable
      variable_address = variable.split("_")[0]
      if variable_address == "N": variable_prior_distribution = "poisson"
      elif variable_address == "shape": variable_prior_distribution = "categorical"
      elif variable_address in ["locX", "locY"]: variable_prior_distribution = "uniform"
      variable_proposal_distribution = proposal_distribution
    
    # training
    elif isinstance(variable, Variable): 
      variable_name = variable.name
      variable_address = variable.address
      variable_prior_distribution = variable.prior_distribution
      variable_proposal_distribution = variable.proposal_distribution

    
    #logger.info(f"input dim in {variable_name} infer_step: {obs.shape} and value shape: {variable.value.shape}")
    
    proposal = self.prop_nets[variable_address](obs)
    
    if variable_proposal_distribution == "normal":
        proposal = proposal.squeeze(-1)
        std = torch.tensor(params["loc_proposal_std"], device=device)
        out = pyro.sample(variable_name, dist.Normal(proposal, std))
    elif variable_proposal_distribution == "categorical":        
       # logger.info(f"\nproposal shape for {variable_name}: {proposal.shape}\n")
       # logger.info(f"{dist.Categorical(probs=proposal).to_event(1).batch_shape} - {dist.Categorical(probs=proposal).to_event(1).event_shape}")
       out = pyro.sample(variable_name, dist.Categorical(probs=proposal))
    elif variable_proposal_distribution == "bernoulli": 
       proposal = proposal.squeeze(-1)       
       out = pyro.sample(variable_name, dist.Bernoulli(proposal))
    else: raise ValueError(f"Unknown variable address: {variable_address}")      
    
    if self.is_train and self.step % params['step_size'] == 0:
        logger.info(f"\n{variable_name} target values {variable.value[0]}")
        logger.info(f"\n{variable_name} proposed values {proposal[0]}")

    return out
  
  def forward(self, 
              observations={"image": torch.zeros((1, 3, 128, 128))}
              ):

    # register networks to be optimized
    pyro.module("encoder_sa", self.encoder_cnn.encoder_sa, True)
    pyro.module("encoder_slots", self.encoder_cnn.encoder_pos, True)
    pyro.module("proposal_nets", self.prop_nets, True)
    pyro.module("sa", self.slot_attention, True)
    pyro.module("mlp", self.mlp, True)
    
    self.img = observations["image"]
    self.img = self.img.to(device)

    B, C, H, W = self.img.shape

    x = self.encoder_cnn(self.img)
    x = nn.LayerNorm(x.shape[1:]).to(device)(x)
    self.features_to_slots = self.mlp(x)

    if params["running_type"] == "inspect": # save input image
        if not os.path.isdir(params["inspect_img_path"]): os.mkdir(params["inspect_img_path"]) # create dir to save inspect logs

        plt.imshow(self.img[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.savefig(f"{params['inspect_img_path']}/img.png")
        plt.close()

    if self.stage == "train":

        n_s = 6
        self.slots, attn = self.slot_attention(self.features_to_slots, num_slots=n_s)

        # for b in range(B):
        #     plot_img = np.transpose(self.img[b].detach().cpu().numpy(), (1, 2, 0))
        #     plt.imshow(plot_img)
        #     plt.axis('off')
        #     plt.savefig(f"{params['check_attn_folder']}/img_{b}.png")
        #     plt.close()
        #     logger.info(f"saved input image {b}...")

      

        if self.is_train and self.step % params['step_size'] == 0:
            aux_attn = attn.reshape((B, n_s, 128, 128)) if not params["strided_convs"] else attn.reshape((B, n_s, 32, 32))
            fig, ax = plt.subplots(ncols=n_s)
            for j in range(n_s):                                       
                im = ax[j].imshow(aux_attn[0, j, :, :].detach().cpu().numpy())
                ax[j].grid(False)
                ax[j].axis('off')        
            plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/attn.png")
            plt.close()

            plot_img = np.transpose(self.img[0].detach().cpu().numpy(), (1, 2, 0))
            plt.imshow(plot_img)
            plt.axis('off')
            plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.step}/img.png")
            plt.close()

        hidden_vars = ["N"]
        for var in self.current_trace:
            if var.name not in hidden_vars:

                # run the proposal for variable var
                _ = self.infer_step(var, self.slots)
      
        del self.slots
        self.current_trace = []
    
    elif self.stage == "eval":
      with torch.no_grad():
        assert self.current_trace == [], "current_trace list is not empty in the begining of evaluation!"

        self.slots, self.slot_pos, attn = self.slot_attention(self.features_to_slots, num_slots=n_s) 

         
        if params["running_type"] == "inspect":
          logging.info(f"slot_pos: {self.slot_pos}")
        
        new_var = Variable(name="N", value=N, prior_distribution="poisson", proposal_distribution=params["N_proposal"], address="N")
        self.current_trace.append(new_var)
        
        """
        Permute the slots to rank by euclidean distance of the attention maps
        """
        
        # 'slot_pos' shape (b_s, n_s, 2)
        # 'slots' shape (b_s, n_s, dim)
        euc_dist = {}
        for s in range(self.slot_pos.shape[-2]): # iterate over slots
          euc_dist[s] = torch.sqrt(torch.square(self.slot_pos[0, s, 0]) + torch.square(self.slot_pos[0, s, 1]))
        sorted_euc_dist = {k: v for k, v in sorted(euc_dist.items(), key=lambda item: item[1])}
        
        obj_properties = ["shape", "size", "color", "locX", "locY"]
        for n in range(N):
          
          # set 'slot_idx' according to the euclidean distance ranking of the slots (not following random initial order)
          slot_idx = list(sorted_euc_dist.keys())[n]
          
          for prop in obj_properties:
            if params["pos_from_attn"] == "attn-masks":
                if prop == "locX": obs = self.slot_pos[0, slot_idx, 0] 
                elif prop == "locY": obs = self.slot_pos[0, slot_idx, 1]
                else: obs = self.slots[0, slot_idx, :]
            else: obs = self.slots[0, slot_idx, :]

            prop_name = prop + "_" + str(n)
            prior_distribution = "uniform" if prop_name[:3] == "loc" else "categorical"
            proposal_distribution = params["loc_proposal"] if prop_name[:3] == "loc" else "categorical"
            
            out = self.infer_step(prop_name, obs, proposal_distribution)
            new_var = Variable(name=prop_name,
                                value=out,
                                prior_distribution=prior_distribution,
                                proposal_distribution=proposal_distribution,
                                address=prop
                                )
            self.current_trace.append(new_var)
        self.current_trace = []
        return 
   
    else: raise ValueError(f"Unknown stage: {self.stage}")
