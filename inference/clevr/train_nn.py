import os
import shutil
import pyro.distributions
import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from scipy.optimize import linear_sum_assignment
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from statistics import mode as stats_mode

import hydra # type: ignore
from sam2.build_sam import build_sam2 # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore


# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../../'))



import wandb # type: ignore

from main.clevr_model import clevr_gen_model
from main.setup import params, JOB_SPLIT
from main.modifiedCSIS import CSIS
from utils.distributions import Empirical
from utils.guide import minimize_entropy_of_sinkhorn, sinkhorn
from eval_utils import compute_AP

main_dir = os.path.abspath(__file__+'/../../../')

params["batch_size"] = 512
params["lr"] = 4e-4

import logging
if params["running_type"] == "train": logfile_name = f"log-{params['jobID']}.log"
elif params["running_type"] == "eval": logfile_name = f"eval-{params['jobID']}-{JOB_SPLIT['id']}.log"
logger = logging.getLogger(params["running_type"])
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

"""
set image and checkpoints saving paths
"""

# start a new wandb run to track this script
if params["running_type"] == "train":
  run = wandb.init(project="ICSA-CLEVR",
                    name=f"{params['jobID']}"
                    )

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAINING_FROM_SCRATCH = params["training_from_scratch"]

logger.info(DEVICE)

sizes = ['small', 'large']
materials = ['rubber', 'metal']
shapes = ['cube', 'sphere', 'cylinder']
colors = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']

def save_img(f, path, title=""):
  plt.imshow(f)
  plt.title(title)
  plt.savefig(path)
  plt.close()

def visualize(x):
   return ((x/2. + 0.5) * 255.).astype(int)

class SlotAttention(nn.Module):
  def __init__(self, num_slots, dim = 64, iters = 3, eps = 1e-8, hidden_dim = 128):
    super().__init__()
    self.num_slots = num_slots
    self.iters = iters
    self.eps = eps
    self.scale = dim ** -0.5
    self.slots_dim = dim
    
    self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
    self.slots_log_sigma = nn.Parameter(torch.rand(1, 1, dim))

    nn.init.xavier_uniform_(
       self.slots_mu, gain=nn.init.calculate_gain("linear")
       )
    nn.init.xavier_uniform_(
       self.slots_log_sigma, gain=nn.init.calculate_gain("linear")
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
    
    self.eps = torch.tensor(1e-8, device=DEVICE)

    self.epoch = 0

  def forward(self, inputs):
    
    b_s, num_inputs, d = inputs.shape # 'inputs' have shape (b_s, W*H, dim)
    l = int(np.sqrt(num_inputs))
    
    # mu = self.slots_mu.expand(b_s, n_s, -1)
    # sigma = self.slots_sigma.expand(b_s, n_s, -1)
    # slots = torch.distributions.Normal(mu, torch.abs(sigma) + self.eps).rsample() # 'slots' have shape (1, n_s, slot_dim=64)

    # Initialize the slots. Shape: [batch_size, num_slots, d_slots].
    slots_init = torch.randn(
        (b_s, self.num_slots, self.slots_dim),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

    inputs = self.norm_input(inputs)      
    k, v = self.to_k(inputs), self.to_v(inputs) # 'k' and 'v' have shape (1, 16384, 64)

    if params["strided_convs"]: l = (int(params['resolution'][0]/4), int(params['resolution'][1]/4))
    else: l = params['resolution']

    a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * self.num_slots # 'a' shape (b_s, W*H)

    for iteration in range(self.iters):

        if params["running_type"] == "inspect": logging.info(f"\nSA iteration {iteration}")
        
        slots_prev = slots
        slots = self.norm_slots(slots) # 'slots' shape (b_s, n_s, slot_dim)
        b = self.mlp_weight_slots(slots).squeeze(-1).softmax(-1) * self.num_slots  # 'b' shape (b_s, n_s)
        
        q = self.to_q(slots) # 'q' shape (1, n_s, 64)
        #attn_logits = cosine_distance(k, q)      
        attn_logits = torch.cdist(k, q)    # [b_s, num_inputs, n_s]
        attn_logits, p, q = minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=params["mesh_lr"], n_mesh_iters=params["mesh_iters"]) 
            
        attn, _, _ = sinkhorn(attn_logits, a, b, u=p, v=q) # 'attn' shape (b_s, num_inputs, n_s)
        attn = attn.permute(0, 2, 1) # 'attn' shape (b_s, n_s, num_inputs)
        updates = torch.matmul(attn, v)

        slots = self.gru(
            updates.view(b_s * self.num_slots, d),
            slots_prev.view(b_s * self.num_slots, d),
        )

        slots = slots.view(b_s, self.num_slots, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))  
    return slots, attn

def build_2d_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="xy")
  grid = np.stack(grid, axis=-1)
  return grid

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0) # (1, 2, 128, 128)
  grid = grid.astype(np.float32)
  grid = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(DEVICE)
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
    
    in_channels = 3
    self.conv1 = nn.Conv2d(in_channels, hid_dim, 5, 1, 2)
    self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, 2, 2)
    self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, 2, 2)
    self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, 1, 2)
    self.relu = nn.ReLU()
    
    #self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)
    if params["strided_convs"]: 
      resolution = (int(params['resolution'][0]/4), int(params['resolution'][1]/4))
    else: resolution = params['resolution']
    self.encoder_pos = SoftPositionEmbed(64, resolution)

  def forward(self, x):    
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))

    x = x.permute(0,2,3,1) # B, W, H, C -> put C in last dimension
    x = self.encoder_pos(x)     

    x = torch.flatten(x, 1, 2) # B, W*H, C -> 64 features of size w*H

    return x

"""Slot Attention-based auto-encoder for object discovery."""
class InvSlotAttentionGuide(nn.Module):
  def __init__(self, resolution, num_slots, num_iterations, slot_dim, stage):
    """Builds the Slot Attention-based auto-encoder.
    Args:
    resolution: Tuple of integers specifying width and height of input image.
    num_slots: Number of slots in Slot Attention.
    num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.slot_dim = slot_dim
    self.num_slots = num_slots
    self.resolution = resolution
    self.num_iterations = num_iterations
    self.stage = stage
    assert self.stage in ["train", "eval"], "stage must be either 'train' or 'eval'"
    self.current_trace = []
    self.is_train = True

    self.encoder_cnn = Encoder(self.resolution, self.slot_dim)
    self.mlp = nn.Sequential(
       nn.Linear(self.slot_dim, self.slot_dim), # used to be hid_dim, hid_dim
       nn.ReLU(),
       nn.Linear(self.slot_dim, self.slot_dim)
    )
    self.slot_attention = SlotAttention(
        num_slots = 10,
        dim = self.slot_dim,
        iters = self.num_iterations,
        eps = 1e-8, 
        hidden_dim = 128)
    self.mlp_preds = nn.Sequential(
       nn.Linear(self.slot_dim, self.slot_dim), nn.ReLU(),
       nn.Linear(self.slot_dim, 19),
    )

    self.batch_idx = 0
    self.epoch = 0
    self.softmax = nn.Softmax(dim=-1)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

  def forward(self, observations={"image": torch.zeros(1, 3, 128, 128)}, save_masks=False, return_slots=False):
    
    img = observations["image"]
    img = img.to(DEVICE)
    B, C, H, W = img.shape

    x = self.encoder_cnn(img) # [B, input_dim, C] 
    x = nn.LayerNorm(x.shape[1:]).to(DEVICE)(x)
    self.features_to_slots = self.mlp(x)
    self.slots, self.attn = self.slot_attention(self.features_to_slots)

    if save_masks:
        aux_attn = self.attn.reshape((B, self.num_slots, params['resolution'][0], params['resolution'][1])) if not params["strided_convs"] else self.attn.reshape((B, self.num_slots, int(params['resolution'][0]/4), int(params['resolution'][1]/4)))
        fig, ax = plt.subplots(ncols=self.num_slots)
        for j in range(self.num_slots):                                       
            im = ax[j].imshow(aux_attn[0, j, :, :].detach().cpu().numpy())
            ax[j].grid(False)
            ax[j].axis('off')        
        plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.epoch}/attn.png")
        plt.close()

        plot_img = np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0))
        plt.imshow(plot_img)
        plt.axis('off')
        plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.epoch}/img.png")
        plt.close()
    
    preds = self.mlp_preds(self.slots)
    preds[:, :, 0:3] = self.sigmoid(preds[:, :, 0:3].clone())       # coords
    preds[:, :, 3:5] = self.softmax(preds[:, :, 3:5].clone())       # size
    preds[:, :, 5:7] = self.softmax(preds[:, :, 5:7].clone())       # material
    preds[:, :, 7:10] = self.softmax(preds[:, :, 7:10].clone())     # shape
    preds[:, :, 10:18] = self.softmax(preds[:, :, 10:18].clone())   # color
    preds[:, :, 18] = self.sigmoid(preds[:, :, 18].clone())         # real object

    #logger.info(f"\nnetwork predicted coords and real flag: {torch.cat((preds[:, :, :3], preds[:, :, -1].unsqueeze(-1)), dim=-1)}")

    if params["running_type"] == "eval":
      pyro.sample("mask", dist.Bernoulli(preds[:, :, 18].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("size", dist.Categorical(probs=preds[:, :, 3:5].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("mat", dist.Categorical(probs=preds[:, :, 5:7].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("shape", dist.Categorical(probs=preds[:, :, 7:10].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("color", dist.Categorical(probs=preds[:, :, 10:18].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("coords", dist.Normal(preds[:, :, :3].expand([params["num_inference_samples"], -1, -1]), torch.tensor(0.01)))
    
    if not return_slots:
      return preds
    else:
      return preds, self.slots


def hungarian_loss(pred, target, loss_fn=F.smooth_l1_loss):
    
    """
    adapted from 'https://github.com/davzha/MESH/blob/main/losses.py'
    """
    
    pdist = loss_fn(
        pred.unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target.unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)

    pdist_ = pdist.detach().cpu().numpy()

    indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices_).to(device=pdist.device))
    total_loss = torch.mean(losses.sum(1))

    return total_loss, dict(indices=indices)

def old_hungarian_loss_inclusive_KL(pred, target, loss_fn=F.smooth_l1_loss):
    
    # pred is [B, N, 19]
    # target is [B, N, 19]   

    k_vars = {"size": 2, "material": 4, "shape": 7, "color": 15}

    pdist_coords = loss_fn(
        pred[:, :, :3].unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target[:, :, :3].unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)
    pdist_real_obj = loss_fn(
        pred[:, :, -1].unsqueeze(-1).unsqueeze(1).expand(-1, target.size(1), -1, -1), 
        target[:, :, -1].unsqueeze(-1).unsqueeze(2).expand(-1, -1, pred.size(1), -1),
        reduction='none').mean(3)
    
    pred = pred[:, :, 3:]
    target = target[:, :, 3:]

    pdist = torch.tensor([])
    for o in range(pred.size(1)):
        i = 0
        log_prob = 0.
        latent_pdist = torch.tensor([])
        for var, k in k_vars.items():
            
            #logger.info(f"var {var} - log_prob using pred with shape {pred[:, :, i:k].shape} for {i} to {k}")
            
            aux_dist = torch.distributions.Categorical(pred[:, :, i:k])
            log_prob = -aux_dist.log_prob(torch.argmax(target[:, o, i:k], dim=-1).unsqueeze(-1).expand(-1, pred.size(1)))                             
            i = k
            log_prob = log_prob.unsqueeze(-1)
            latent_pdist = torch.cat((latent_pdist, log_prob), dim=-1) # [B, N, nlatents]

        latent_pdist = latent_pdist.unsqueeze(-2) # [B, N, 1, nlatents]
        pdist = torch.cat((pdist, latent_pdist), dim=-2) # [B, N, N, nlatents]
    
    pdist = pdist.mean(-1) + pdist_coords + pdist_real_obj

    pdist_ = pdist.detach().cpu().numpy()
    indices = np.array([linear_sum_assignment(p) for p in pdist_])
    indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices_).to(device=pdist.device))
    total_loss = torch.mean(losses.sum(1))

    return total_loss, dict(indices=indices)

def hungarian_loss_inclusive_KL(pred, target, loss_fn=F.smooth_l1_loss):
    
    # pred is [B, N, 19]
    # target is [B, N, 19]   

    k_vars = {"coords": 3, "size": 5, "material": 7, "shape": 10, "color": 18, "mask": 19}

    # pdist_coords = loss_fn(
    #     pred[:, :, :3].unsqueeze(1).expand(-1, target.size(1), -1, -1), 
    #     target[:, :, :3].unsqueeze(2).expand(-1, -1, pred.size(1), -1),
    #     reduction='none').mean(3)
    # pdist_real_obj = loss_fn(
    #     pred[:, :, -1].unsqueeze(-1).unsqueeze(1).expand(-1, target.size(1), -1, -1), 
    #     target[:, :, -1].unsqueeze(-1).unsqueeze(2).expand(-1, -1, pred.size(1), -1),
    #     reduction='none').mean(3)
    
    # pred = pred[:, :, 3:]
    # target = target[:, :, 3:]

    pdist = torch.tensor([])
    for o in range(pred.size(1)):
        i = 0
        log_prob = 0.
        latent_pdist = torch.tensor([])
        for var, k in k_vars.items():
            
            if var == "coords":
              aux_dist = torch.distributions.Normal(pred[:, :, i:k], torch.tensor(0.01))
              log_prob = -aux_dist.log_prob(target[:, o, i:k].unsqueeze(-2).expand(-1, pred.size(1), -1)).mean(-1)
            elif var == "mask":
              aux_dist = torch.distributions.Bernoulli(pred[:, :, i:k].squeeze(-1))
              log_prob = -aux_dist.log_prob(target[:, o, i:k].expand(-1, pred.size(1)))
            else:
              aux_dist = torch.distributions.Categorical(pred[:, :, i:k])
              log_prob = -aux_dist.log_prob(torch.argmax(target[:, o, i:k], dim=-1).unsqueeze(-1).expand(-1, pred.size(1))) 
               
            #logger.info(f"var {var} - log_prob using pred with shape {pred[:, :, i:k].shape} for {i} to {k}")
                                        
            i = k
            log_prob = log_prob.unsqueeze(-1)
            latent_pdist = torch.cat((latent_pdist, log_prob), dim=-1) # [B, N, nlatents]

        latent_pdist = latent_pdist.unsqueeze(-2) # [B, N, 1, nlatents]
        pdist = torch.cat((pdist, latent_pdist), dim=-2) # [B, N, N, nlatents]
    
    pdist = pdist.mean(-1)

    pdist_ = pdist.detach().cpu().numpy()
    indices = np.array([linear_sum_assignment(p) for p in pdist_])
    indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices_).to(device=pdist.device))
    total_loss = torch.mean(losses.sum(1))

    return total_loss, dict(indices=indices)

        
class Trainer:
    def __init__(self, model, dataloaders, params, logger, log_rate): 
        self.trainloader = dataloaders["train"]
        self.validloader = dataloaders["validation"]
        self.params = params
        self.model = model
        self.num_iters = 0
        self.epoch = 0
        self.num_epochs = 1000
        self.device = self.params['device']
        self.optimizer = torch.optim.Adam([p for p in list(self.model.parameters()) if p.requires_grad], lr = 4e-4)
        self.logger = logger
        self.log_rate = log_rate

        self.checkpoint_path = os.path.join(main_dir, "inference", f"checkpoint-{params['jobID']}")
    
    def _save_checkpoint(self, epoch):
        if not os.path.isdir(self.checkpoint_path):
            try: os.mkdir(self.checkpoint_path)
            except: logger.info('unable to create directory to save training checkpoints!')
        else:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_path, f"guide_{epoch}.pth"))

    def _train_epoch(self, save_masks):
        loss = 0.
        num_iters = 0
        self.model.train() 
        for img, target in self.trainloader:
            img, target = img.to(self.device), target.to(self.device)
            preds = self.model(observations={"image": img}, save_masks=save_masks)
            
            if save_masks:
              logger.info(f"preds: {preds[0]}")
              logger.info(f"target: {target[0]}")
            
            if params["jobID"] == 101:
              batch_loss, _ = hungarian_loss(preds, target)
            elif params["jobID"] == 102:
              batch_loss, _ = hungarian_loss_inclusive_KL(preds, target)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step() 
            
            loss += batch_loss.item()
            num_iters += 1

            save_masks = False

        return loss/num_iters
    
    def _valid_epoch(self):
        loss = 0.
        num_iters = 0
        metrics = {}
        threshold = [-1., 1., 0.5, 0.25, 0.125, 0.0625]
        ap = {k: 0 for k in threshold}

        self.model.eval()
        with torch.no_grad(): 
            for img, target in self.validloader:
                img, target = img.to(self.device), target.to(self.device)
                preds = self.model(observations={"image": img})
                if params["jobID"] == 101: batch_loss, _ = hungarian_loss(preds, target)
                elif params["jobID"] == 102: batch_loss, _ = hungarian_loss_inclusive_KL(preds, target)

                preds[:, :, 3:5] = F.one_hot(torch.argmax(preds[:, :, 3:5], dim=-1), len(sizes))       # size
                preds[:, :, 5:7] = F.one_hot(torch.argmax(preds[:, :, 5:7], dim=-1), len(materials))       # material
                preds[:, :, 7:10] = F.one_hot(torch.argmax(preds[:, :, 7:10], dim=-1), len(shapes))     # shape
                preds[:, :, 10:18] = F.one_hot(torch.argmax(preds[:, :, 10:18], dim=-1), len(colors))   # color
                preds[:, :, 18] = torch.distributions.Bernoulli(preds[:, :, 18]).sample()         # real object
                
                
                for i in range(preds.shape[0]):
                  for t in threshold: 
                      
                      ap[t] += compute_AP(preds[i].detach().cpu(), 
                                          target[i].detach().cpu(), 
                                          t)
                
                loss += batch_loss.item()
                num_iters += 1

                ap = {k: v/preds.shape[0] for k, v in ap.items()}
        
        mAP = {k: v/num_iters for k, v in ap.items()}
        metrics['mAP'] = mAP
        
        return loss/num_iters, metrics

    def train(self, root_folder):
        since = time.time()  

        for epoch in range(self.num_epochs):                  
            
            self.epoch = epoch
            self.model.epoch = epoch
            self.model.is_train = True
            save_masks = False
            
            if epoch % self.log_rate == 0:
                logger.info("Epoch {}/{}".format(epoch, self.num_epochs - 1))
                if not os.path.isdir(f"{root_folder}/attn-step-{epoch}"): 
                    os.mkdir(f"{root_folder}/attn-step-{epoch}") 
                save_masks = True 
                    
            epoch_train_loss = self._train_epoch(save_masks)
            
            if epoch % self.log_rate == 0 or epoch == self.num_epochs-1:
                self.model.is_train = False
                epoch_valid_loss, valid_metrics = self._valid_epoch() 
                logger.info("... train_loss: {:.3f}" .format(epoch_train_loss))
                logger.info("... valid_loss: {:.3f}" .format(epoch_valid_loss))
                logger.info(f"... valid mAP: {valid_metrics['mAP'].items()}")


                self.logger.log({"train_loss": epoch_train_loss,
                                 "val_loss": epoch_valid_loss,
                                 "val_mAP": valid_metrics['mAP']})

                self._save_checkpoint(epoch)
        
        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


sizes = ['small', 'large']
materials = ['rubber', 'metal']
shapes = ['cube', 'sphere', 'cylinder']
colors = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']


def list2dict(inpt_list):
  return {inpt_list[i]: i for i in range(len(inpt_list))}


size2id = list2dict(sizes)
mat2id = list2dict(materials)
shape2id = list2dict(shapes)
color2id = list2dict(colors)


class CLEVR(Dataset):
    def __init__(self, images_path, scenes_path, max_objs=6, get_target=True, get_pixel_coords=False):
        self.max_objs = max_objs
        self.get_target = get_target
        self.get_pixel_coords = get_pixel_coords
        self.images_path = images_path
        
        with open(scenes_path, 'r') as f:
            self.scenes = json.load(f)['scenes']
        self.scenes = [x for x in self.scenes if len(x['objects']) <= max_objs]
        
        transform = [transforms.CenterCrop((256, 256))] if not get_target else []
        self.transform = transforms.Compose(
            transform + [
                transforms.Resize((128, 128)),
                transforms.ToTensor()
                ]
        )
        
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        img = Image.open(os.path.join(self.images_path, scene['image_filename'])).convert('RGB')
        img = self.transform(img)
        target = []
        pixel_coords = []
        if self.get_target:
            for obj in scene['objects']:
                coords = ((torch.Tensor(obj['3d_coords']) + 3.) / 6.).view(1, 3)
                #coords = (torch.tensor(obj['3d_coords']) / 3.).view(1, 3)
                size = F.one_hot(torch.LongTensor([size2id[obj['size']]]), 2)
                material = F.one_hot(torch.LongTensor([mat2id[obj['material']]]), 2)
                shape = F.one_hot(torch.LongTensor([shape2id[obj['shape']]]), 3)
                color = F.one_hot(torch.LongTensor([color2id[obj['color']]]), 8)
                obj_vec = torch.cat((coords, size, material, shape, color, torch.Tensor([[1.]])), dim=1)[0]
                target.append(obj_vec)

                resize_factor = np.array([128/480, 128/320])
                pixel_coords.append(torch.Tensor([obj['pixel_coords'][0]*resize_factor[0], obj['pixel_coords'][1]*resize_factor[1]])) # 320x240 -> 128x128

            while len(target) < self.max_objs:
                target.append(torch.zeros(19, device='cpu'))
            target = torch.stack(target)  
            pixel_coords = torch.stack(pixel_coords)     
        if not self.get_pixel_coords:
          return img*2 - 1, target 
        else: return img*2 - 1, target, pixel_coords


def average_precision_clevr(pred, attributes, distance_threshold):
  """Computes the average precision for CLEVR.
  This function computes the average precision of the predictions specifically
  for the CLEVR dataset. First, we sort the predictions of the model by
  confidence (highest confidence first). Then, for each prediction we check
  whether there was a corresponding object in the input image. A prediction is
  considered a true positive if the discrete features are predicted correctly
  and the predicted position is within a certain distance from the ground truth
  object.
  Args:
    pred: Tensor of shape [batch_size, num_elements, dimension] containing
      predictions. The last dimension is expected to be the confidence of the
      prediction.
    attributes: Tensor of shape [batch_size, num_elements, dimension] containing
      ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.
  Returns:
    Average precision of the predictions.
  """

  # pred[:, :, :3] = (pred[:, :, :3] + 1) / 2
  # attributes[:, :, :3] = (attributes[:, :, :3] + 1) / 2

  [batch_size, _, element_size] = attributes.shape
  [_, predicted_elements, _] = pred.shape

  def unsorted_id_to_image(detection_id, predicted_elements):
    """Find the index of the image from the unsorted detection index."""
    return int(detection_id // predicted_elements)

  flat_size = batch_size * predicted_elements
  flat_pred = np.reshape(pred, [flat_size, element_size])
  sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

  sorted_predictions = np.take_along_axis(
      flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
  idx_sorted_to_unsorted = np.take_along_axis(
      np.arange(flat_size), sort_idx, axis=0)

  def process_targets(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    object_size = np.argmax(target[3:5])
    material = np.argmax(target[5:7])
    shape = np.argmax(target[7:10])
    color = np.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj

  true_positives = np.zeros(sorted_predictions.shape[0])
  false_positives = np.zeros(sorted_predictions.shape[0])

  detection_set = set()

  for detection_id in range(sorted_predictions.shape[0]):

    logger.info(f"\nsearching for matches for predicted object {detection_id}...")

    # Extract the current prediction.
    current_pred = sorted_predictions[detection_id, :]
    # Find which image the prediction belongs to. Get the unsorted index from
    # the sorted one and then apply to unsorted_id_to_image function that undoes
    # the reshape.
    original_image_idx = unsorted_id_to_image(
        idx_sorted_to_unsorted[detection_id], predicted_elements)
    # Get the ground truth image.
    gt_image = attributes[original_image_idx, :, :]

    # Initialize the maximum distance and the id of the groud-truth object that
    # was found.
    best_distance = 10000
    best_id = None

    # Unpack the prediction by taking the argmax on the discrete attributes.
    (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
     _) = process_targets(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):

      logger.info(f"is it target object {target_object_id}?")
    
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      (target_coords, target_object_size, target_material, target_shape,
       target_color, target_real_obj) = process_targets(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        # For the match to be valid all attributes need to be correctly
        # predicted.
        pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
        target_attr = [
            target_object_size, target_material, target_shape, target_color]
        match = pred_attr == target_attr
        if match:
          # If a match was found, we check if the distance is below the
          # specified threshold. Recall that we have rescaled the coordinates
          # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
          # `pred_coords`. To compare in the original scale, we thus need to
          # multiply the distance values by 6 before applying the norm.
          distance = np.linalg.norm((target_coords - pred_coords) * 3.)

          # If this is the best match we've found so far we remember it.
          if distance < best_distance:
            best_distance = distance
            best_id = target_object_id
    
    if best_distance < distance_threshold or distance_threshold == -1:
      # We have detected an object correctly within the distance confidence.
      # If this object was not detected before it's a true positive.
      if best_id is not None:
        if (original_image_idx, best_id) not in detection_set:
          true_positives[detection_id] = 1
          
          logger.info(f"YES!")
          logger.info(f"target attr: {target_attr}")
          logger.info(f"preds attr: {pred_attr}")
          detection_set.add((original_image_idx, best_id))
          logger.info(f"adding the match pred/target {detection_id}/{target_object_id}")
          logger.info(f"detection set so far: {detection_set}")

        else:
          false_positives[detection_id] = 1
      else:
        false_positives[detection_id] = 1
    else:
      false_positives[detection_id] = 1
  accumulated_fp = np.cumsum(false_positives)
  accumulated_tp = np.cumsum(true_positives)
  recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
  precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

  return compute_average_precision(
        np.array(precision_array, dtype=np.float32),
        np.array(recall_array, dtype=np.float32))


def compute_average_precision(precision, recall):
    """Computation of the average precision from precision and recall arrays."""
    recall = recall.tolist()
    precision = precision.tolist()
    recall = [0] + recall + [1]
    precision = [0] + precision + [0]

    for i in range(len(precision) - 1, -0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices_recall = [
        i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
    ]

    average_precision = 0.
    for i in indices_recall:
        average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
    return average_precision


guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_slots = 10,
                              num_iterations = 3,
                              slot_dim = params["slot_dim"],
                              stage="train"
                              ).to(DEVICE)

if params["running_type"] == "train": run.watch(guide)

"""
build CSIS class with model & guide
"""


CHECK_ATTN = params["check_attn"]
root_folder = params["check_attn_folder"]

if CHECK_ATTN and TRAINING_FROM_SCRATCH:
  if not os.path.isdir(root_folder): os.mkdir(root_folder)
  
  if len(os.listdir(root_folder)) > 0:
    try:
      folders = os.listdir(root_folder)
      for folder in folders: 
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
          shutil.rmtree(folder_path)
    except OSError:
      logger.info("Error occurred while deleting files.")

  if len(os.listdir(root_folder)) == 0:
    logger.info("Folders for all steps deleted...")


dataset_path = "/nas-ctm01/datasets/public/CLEVR/CLEVR_v1.0"
train_data = CLEVR(images_path = os.path.join(dataset_path, 'images/train'),
                   scenes_path = os.path.join(dataset_path, 'scenes/CLEVR_train_scenes.json'),
                   max_objs=10)
train_dataloader = DataLoader(train_data, batch_size = 512,
                              shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
val_images_path = os.path.join(dataset_path, 'images/val')
val_data = CLEVR(images_path = os.path.join(dataset_path, 'images/val'),
                   scenes_path = os.path.join(dataset_path, 'scenes/CLEVR_val_scenes.json'),
                   max_objs=10)
val_dataloader = DataLoader(val_data, batch_size = 512,
                              shuffle=False, num_workers=8, generator=torch.Generator(device='cuda'))


if params["running_type"] == "train":  
  trainer = Trainer(guide, {"train": train_dataloader, "validation": val_dataloader}, params, run, log_rate=10)
  trainer.train(root_folder)
  logger.info("\ntraining ended...")

elif params["running_type"] == "eval":

  log_rate = 50 

  def transform_to_depth(img: torch.Tensor):
    # from [-1., 1.] to [0., 1.] img
    return img/2 + 0.5 
  
  def run_inference(img, n, guide, prop_traces, traces, posterior, input_mode, pixel_coords, log_rate):
    
    if input_mode == "RGB":
      # for name, site in traces.nodes.items():                                  
      #   if name == 'image':
      #     output_images = site["fn"].mean
      #     D = output_images.size(-1)*output_images.size(-2)
      #     sigma = torch.mean(torch.sqrt(((output_images-img)**2).sum(dim=(-1, -2, -3)) / D))
      #     sigma = sigma[:, None, None, None]  # [D, 1, 1, 1]
      #     sigma = sigma.expand(-1, 3, 128, 128)        
          
      #     logger.info(output_images.shape)
      #     logger.info(img.shape)
      #     logger.info(f"sigma: {sigma} with shape: {sigma.shape}")
          
      #     sigma = 0.05
      #     log_wts = dist.Independent(dist.Normal(output_images, sigma), 3).log_prob(img)
      
      log_wts = posterior.log_weights[0]
      resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), log_wts)
      resampling_id = resampling().item()    
    
    elif input_mode in ["depth", "seg_masks_object", "seg_masks_color", "seg_masks_mat"]: 
      transform_gen_imgs = []

      for name, site in traces.nodes.items():                                  
        if name == 'image':
          for i in range(site["fn"].mean.shape[0]):
            output_image = site["fn"].mean[i]

            preds = process_preds(prop_traces, i) # get latent predictions related to trace i

            if input_mode == "depth":
              transformed_tensor = zoe.infer(transform_to_depth(output_image.unsqueeze(0))) # [1, 1, 128, 128]
              if SAVING_IMG:
                save_img(transformed_tensor.squeeze().cpu().numpy(),
                        os.path.join(plots_dir, f"transf_trace_{n_test_samples}_{i}.png"))
            
            elif input_mode in ["seg_masks_object", "seg_masks_color", "seg_masks_mat"]:
              
              checkpoint = "/nas-ctm01/homes/fcsilva/sam2/checkpoints/sam2.1_hiera_large.pt"
              model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
              predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

              with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output_image = output_image/2 + 0.5 # [-1., 1.] -> [0., 1.]
                predictor.set_image(output_image.permute(1, 2, 0).cpu().numpy())
                
                # build a 2d grid and get the output masks from SA-MESH to give 2d coordinates for these points
                slots_attn = guide.attn
                B, N, d = slots_attn.shape
                # normalize attn matrices
                slots_attn = slots_attn / torch.sum(slots_attn, dim=-1, keepdim=True)
                slots_attn = slots_attn.reshape(B, N, int(np.sqrt(d)), int(np.sqrt(d))).double()
                grid = torch.from_numpy(build_2d_grid((32, 32)))
                
                pred_coords = torch.einsum('nij,ijk->nk', slots_attn[0].cpu(), grid)
                # logger.info(coords.shape)
                pred_real_flag = [m for m in range(N) if torch.round(preds[m, -1]) == 1] 
                
                # logger.info(f"pred real flag: {pred_real_flag}")

                real_pred_coords = pred_coords[pred_real_flag] * 128. # [#real, 2]
                
                # logger.info(f"pred coords shape: {coords.shape}")
                # logger.info(f"pixel coords shape: {pixel_coords.shape}") # [real_N, 2]

                transformed_tensor = torch.zeros(128, 128)

                for o, obj_coords in enumerate(pixel_coords):
                  
                  obj_coords = np.asarray(obj_coords.unsqueeze(0)) # obj_coords [2]
                  input_point = obj_coords
                  input_label = np.array([1 for _ in range(obj_coords.shape[0])])

                  input_point = np.concatenate((input_point, np.array([[10, 10]])))
                  input_label = np.concatenate((input_label, np.array([0])))

                  masks, scores, logits = predictor.predict(
                      point_coords=input_point,
                      point_labels=input_label,
                      multimask_output=True,
                  )
                  sorted_ind = np.argsort(scores)[::-1]
                  masks = masks[sorted_ind]
                  scores = scores[sorted_ind]
                  logits = logits[sorted_ind]

                  # taking only the mask with highest score
                  masks = torch.tensor(masks[0]).unsqueeze(0).cpu().numpy()
                  
                  if SAVING_IMG:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.imshow(visualize(output_image.permute(1, 2, 0).cpu().numpy()))
                    for p, point in enumerate(input_point):
                      if input_label[p]: plt.scatter(point[0], point[1], marker="x")
                      else: plt.scatter(point[0], point[1], marker="o")

                      #ax.add_patch(Rectangle((box[p][0], box[p][1]), 40, 40, fc ='none',  ec ='r'))
                    plt.savefig(os.path.join(plots_dir, f"trace_{i}_object_{o}_image_{n_test_samples}.png"))
                    plt.close()

                  if input_mode == "seg_masks_object":
                    # mask color is defined by object id 'o'
                    transformed_tensor += torch.tensor(masks[0]*(o+1))
                  
                  else:
                      
                    # mask color is defined by the predicted object color
                    dists = torch.cdist(pixel_coords, real_pred_coords.float(), p=2).cpu().numpy() # [2, real_n, real_n]
                    row_ind, col_ind = linear_sum_assignment(dists)
                    
                    # maybe 'real_n' != 'pred_real_n' and 'o' won't be in row_ind (i.e. the index of pred real objects)
                    if o in row_ind:
                      o_idx = list(row_ind).index(o)
                    
                      # check, in preds, where 'col_ind[o_idx]' is
                      pred_abs_idx = torch.where(pred_coords*128. == real_pred_coords[col_ind[o_idx]])[0][0]

                      # logger.info(f"target index {o} in position {o_idx} -> pred object {col_ind[o_idx]} with abs index {pred_abs_idx}...")

                      if input_mode == "seg_masks_color":
                        color_pred = torch.argmax(preds[pred_abs_idx, 10:18], dim=-1).item()
                        transformed_tensor += torch.tensor(masks[0]*(color_pred+1))
                      elif input_mode == "seg_masks_mat":
                        mat_pred = torch.argmax(preds[pred_abs_idx, 5:7], dim=-1).item()
                        transformed_tensor += torch.tensor(masks[0]*(mat_pred+1))

                  if SAVING_IMG:
                    save_img(masks.squeeze(),
                            os.path.join(plots_dir, f"trace_{i}_mask_{o}_image_{n_test_samples}.png"),
                            title=f"score: {scores}")
            if SAVING_IMG:
              save_img(transformed_tensor.cpu().numpy(),
                      os.path.join(plots_dir, f"trace_{i}_all_mask_image_{n_test_samples}.png"))

            transform_gen_imgs.append(torch.tensor(transformed_tensor))
      
      transform_gen_imgs = torch.stack(transform_gen_imgs)

      if input_mode == "depth":
        transformed_target_tensor = zoe.infer(transform_to_depth(img)) # [1, 1, 128, 128]
        if SAVING_IMG:
          save_img(transformed_target_tensor.squeeze().cpu().numpy(),
                  os.path.join(plots_dir, f"depth_image_{n_test_samples}.png"))
      
        log_wts = []
        D = transform_gen_imgs.size(-1)*transform_gen_imgs.size(-2)
        sigma = torch.sqrt(((transform_gen_imgs - transformed_target_tensor)**2).sum(dim=(-1, -2, -3)) / D)
        
        for i in range(params["num_inference_samples"]):
          # log_p = dist.Normal(transform_gen_imgs[i], torch.tensor(0.05)).log_prob(torch.tensor(transformed_target_tensor))
          # img_dim = transform_gen_imgs[i].shape[-1]
          # log_p = torch.sum(log_p) / (img_dim**2)
          log_p = dist.Normal(transform_gen_imgs[i], torch.tensor(sigma)).log_prob(torch.tensor(transformed_target_tensor))
          log_wts.append(log_p)             
        resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
        resampling_id = resampling().item()      
      
      elif input_mode in ["seg_masks_object", "seg_masks_color", "seg_masks_mat"]:

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
          img = img/2 + 0.5 # [-1., 1.] -> [0., 1.]
          predictor.set_image(img[0].permute(1, 2, 0).cpu().numpy())
          
          transformed_target_tensor = torch.zeros(128, 128)

          for o, obj_coords in enumerate(pixel_coords):
            obj_coords = np.asarray(obj_coords.unsqueeze(0)) # obj_coords [2]
            input_point = obj_coords
            input_label = np.array([1 for _ in range(obj_coords.shape[0])])

            input_point = np.concatenate((input_point, np.array([[10, 10]])))
            input_label = np.concatenate((input_label, np.array([0])))

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            # taking only the mask with highest score
            masks = torch.tensor(masks[0]).unsqueeze(0).cpu().numpy()

            if input_mode == "seg_masks_object":
              # mask color is defined by object id 'o'
              transformed_target_tensor += torch.tensor(masks[0]*(o+1))
            
            else:
              if input_mode == "seg_masks_color":
                color = torch.argmax(target[o, 10:18], dim=-1).item()
                transformed_target_tensor += torch.tensor(masks[0]*(color+1))
              elif input_mode == "seg_masks_mat":
                mat = torch.argmax(target[o, 5:7], dim=-1).item()
                transformed_target_tensor += torch.tensor(masks[0]*(mat+1))
        
        if SAVING_IMG:
          save_img(transformed_target_tensor.cpu().numpy(),
                  os.path.join(plots_dir, f"transf_image_{n_test_samples}.png"))

        log_wts = []
        D = transform_gen_imgs.size(-1)*transform_gen_imgs.size(-2)
        sigma = torch.sqrt(((transform_gen_imgs - transformed_target_tensor)**2).sum(dim=(-1, -2, -3)) / D)

        for i in range(params["num_inference_samples"]):
          # log_p = dist.Normal(transform_gen_imgs[i], torch.tensor(0.05)).log_prob(torch.tensor(transformed_target_tensor))
          # img_dim = transform_gen_imgs[i].shape[-1]
          # log_p = torch.sum(log_p) / (img_dim**2)
          log_p = dist.Normal(transform_gen_imgs[i], torch.tensor(sigma)).log_prob(torch.tensor(transformed_target_tensor))
          log_wts.append(log_p)
        resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
        resampling_id = resampling().item()

    elif input_mode == "slots":
        
      trace_generated_imgs = []
      for name, site in traces.nodes.items():                                  
        if name == 'image':
          for i in range(site["fn"].mean.shape[0]):
            output_image = site["fn"].mean[i]
            trace_generated_imgs.append(output_image)
      trace_generated_imgs = torch.stack(trace_generated_imgs)
      # logger.info(trace_generated_imgs.shape) # [particles, 3, 128, 128]
      preds, trace_slots = guide(observations={"image": trace_generated_imgs}, return_slots=True)
      # logger.info(trace_slots.shape) # [particles, N, 64]
      # logger.info(target_slots.shape)

      slots_dist = torch.cdist(trace_slots, target_slots)
      slots_dist = slots_dist.detach().cpu().numpy()

      # logger.info(slots_dist.shape)

      indices = np.array([linear_sum_assignment(d) for d in slots_dist])
      
      assert len(trace_slots.shape) == 3 # 
      batch_idx = torch.arange(trace_slots.size(0)).unsqueeze(1).expand(trace_slots.size(0), trace_slots.size(1))
      trace_slots = trace_slots[batch_idx, indices[:, 1]]

      slots_dim = trace_slots.shape[-1]
      sigma = torch.sqrt(((trace_slots - target_slots)**2).sum(dim=(-1, -2)) / slots_dim)     
      log_wts = dist.Normal(trace_slots, torch.tensor(sigma)).log_prob(torch.tensor(target_slots))
      #log_wts = torch.sum(log_wts, dim=(-1, -2))
      resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), log_wts)
      resampling_id = resampling().item()
    
    if n == 1 or n % log_rate == 0:
      logger.info(f"\n{input_mode} log_wts: {[l.item() for l in log_wts]} - resampled trace {resampling_id}")

    return resampling_id
    

  plots_dir = os.path.abspath("set_prediction_plots")
  if not os.path.isdir(plots_dir): os.mkdir(plots_dir)
  else: 
      shutil.rmtree(plots_dir)
      os.mkdir(plots_dir)

  input_mode = "all" # ["RGB", "depth", "seg_masks", "slots", "all"]
  logger.info(f"\ninput_mode = {input_mode}")
  if input_mode == "seg_masks":
    mask_type = "matID" # ["regular", "colorID", "matID"]
    logger.info(f"mask_type = {mask_type}")


  SAVING_IMG = False

  if input_mode in ["depth", "all"]:
    # load pre-trained model
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

    repo = "isl-org/ZoeDepth"
    # Zoe_NK
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    zoe = model_zoe_nk.to(DEVICE)   
    
  guide = InvSlotAttentionGuide(resolution = params['resolution'],
                                  num_slots = 10,
                                  num_iterations = 3,
                                  slot_dim = params["slot_dim"],
                                  stage="eval"
                                  ).to(DEVICE)
    
  checkpoint_path = os.path.join(main_dir, "inference", f"checkpoint-{params['jobID']}")
  epoch_to_load = 950 # 999
  guide.load_state_dict(torch.load(os.path.join(checkpoint_path, f"guide_{epoch_to_load}.pth")))

  logger.info(f"\nInference network from epoch {epoch_to_load} successfully loaded...")

  threshold = [-1., 1., 0.5, 0.25, 0.125, 0.0625]
  ap = {k: 0 for k in threshold}
  max_ap = {k: 0 for k in threshold}

  val_data = CLEVR(images_path = os.path.join(dataset_path, 'images/val'),
                   scenes_path = os.path.join(dataset_path, 'scenes/CLEVR_val_scenes.json'),
                   max_objs=10,
                   get_pixel_coords = True)
  
  val_dataloader = DataLoader(val_data, batch_size = 1,
                                shuffle=False, num_workers=8, generator=torch.Generator(device='cuda'))

    
  logger.info(f"\nStarting inference with {params['num_inference_samples']} particles...\n")

  if params["num_inference_samples"] == 1:
     
    n_test_samples = 0
    with torch.no_grad():
      for img, target in val_dataloader:
        img = img.to(DEVICE) 
        preds = guide(observations={"image": img})

        n_test_samples += 1

        target = target.squeeze(0)
        preds = preds.squeeze(0)

        for t in threshold: 
          #logger.info(f"\ndistance threshold {t}\n")
          ap[t] += compute_AP(preds.detach().cpu(),
                              target.detach().cpu(),
                              t)
      
        if n_test_samples == 1 or n_test_samples % 100 == 0:
          logger.info(f"{n_test_samples} evaluated...")
          logger.info(f"current stats:")
          aux_mAP = {k: v/n_test_samples for k, v in ap.items()}
          logger.info(aux_mAP)
        
        if n_test_samples == 200:
          break

    
  else:

    model = clevr_gen_model
    optimiser = pyro.optim.Adam({'lr': 1e-4})
    csis = CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

    def process_preds(trace, id):
      max_obj = max(params['max_objects'], params['num_slots'])
      
      features_dim = 19
      preds = torch.zeros(max_obj, features_dim)
      for name, site in trace.nodes.items():
          if site['type'] == 'sample':
              if name == 'coords': preds[:, :3] = site['value'][id] # [-3., 3.]
              if name == 'size': preds[:, 3:5] = F.one_hot(site['value'][id], len(sizes))
              if name == 'mat': preds[:, 5:7] = F.one_hot(site['value'][id], len(materials))
              if name == 'shape': preds[:, 7:10] = F.one_hot(site['value'][id], len(shapes))
              if name == 'color': preds[:, 10:18] = F.one_hot(site['value'][id], len(colors))
              if name == 'mask': preds[:, 18] = site['value'][id]
      return preds
    
    if input_mode == "all":
      resampled_traces = {}
    
    max_AP_mode = []
    n_test_samples = 0
    with torch.no_grad():
      for idx, (img, target, pixel_coords) in enumerate(val_dataloader):
                 
          img = img.to(DEVICE)   

          pixel_coords = pixel_coords[0]  
          target = target[0]   

          n_test_samples += 1
          if input_mode == "all":
            resampled_traces[n_test_samples] = []
          
          posterior = csis.run(observations={"image": img})
          prop_traces = posterior.prop_traces[0]
          traces = posterior.exec_traces[0]

          if input_mode in ["slots", "all"]:
            target_slots = guide.slots

          # get the predictions of the first proposal trace
          preds = process_preds(prop_traces, 0) 

          if SAVING_IMG:
            save_img(visualize(img[0].permute(1, 2, 0).cpu().numpy()),
                     os.path.join(plots_dir, f"image_{n_test_samples}.png"))
          
          
          modes = ["RGB", "depth", "seg_masks_object", "seg_masks_color", "seg_masks_mat", "slots"]
          for mode in modes:
            resampling_id = run_inference(img=img,
                                          n=n_test_samples,
                                          guide=csis.guide,
                                          prop_traces=prop_traces,
                                          traces=traces,
                                          posterior=posterior,
                                          input_mode=mode,
                                          pixel_coords=pixel_coords,
                                          log_rate=log_rate
                                          )
            if input_mode == "all":
              resampled_traces[n_test_samples].append(resampling_id)

          
          if n_test_samples == 1 or n_test_samples % log_rate == 0:
            logger.info(f"\nall models resampled traces: {resampled_traces}")
          
          resampling_id = stats_mode(resampled_traces[n_test_samples])
          
          preds = process_preds(prop_traces, resampling_id)
          assert len(target.shape) == 2
          for t in threshold: 
            ap[t] += compute_AP(preds.detach().cpu(),
                                target.detach().cpu(),
                                t)

          max_ap_idx = 0
          best_overall_ap = 0.   
          for i in range(params["num_inference_samples"]):
            aux_ap = {k: 0 for k in threshold}
            preds = process_preds(prop_traces, i)
            for t in threshold: 
              aux_ap[t] = compute_AP(preds.detach().cpu(),
                                     target.detach().cpu(),
                                     t)
            overall_ap = np.mean(list(aux_ap.values()))


            #logger.info(f"trace {i} - {aux_ap} with overall AP {overall_ap} ")

            if overall_ap > best_overall_ap:
                best_overall_ap = overall_ap
                max_ap_idx = i
                #logger.info(f"max_ap_idx is now {max_ap_idx} with log_wt {log_wts[max_ap_idx]} and overall AP {best_overall_ap}")
          
          max_preds = process_preds(prop_traces, max_ap_idx)
          for t in threshold: 
            max_ap[t] += compute_AP(max_preds.detach().cpu(),
                                   target.detach().cpu(),
                                   t)
      
          # find which input mode would give the maxAP 
          try:
            max_AP_mode.append(modes[resampled_traces[n_test_samples].index(max_ap_idx)])
          except:
            max_AP_mode.append(None)
          

          if n_test_samples == 1 or n_test_samples % log_rate == 0:
            logger.info(f"\nMAX AP MODES: {max_AP_mode}")
             
          if n_test_samples == 1 or n_test_samples % log_rate == 0:
            logger.info(f"\n{n_test_samples} evaluated...")
            logger.info(f"current stats:")
            aux_mAP = {k: v/n_test_samples for k, v in ap.items()}
            logger.info(aux_mAP)

            logger.info(f"current stats if mAP was maximized when sampling the posterior:")
            max_aux_mAP = {k: v/n_test_samples for k, v in max_ap.items()}
            logger.info(max_aux_mAP)
          
          if n_test_samples == 200:
            break
  logger.info(f"\ninference ended...")

if params["running_type"] == "train": wandb.finish()
