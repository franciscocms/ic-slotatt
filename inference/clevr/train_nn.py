import os
import shutil
import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from scipy.optimize import linear_sum_assignment
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../../'))



import wandb # type: ignore

from main.clevr_model import clevr_gen_model
from main.setup import params
from main.modifiedCSIS import CSIS
from utils.distributions import Empirical
from utils.guide import minimize_entropy_of_sinkhorn, sinkhorn

main_dir = os.path.abspath(__file__+'/../../../')

params["batch_size"] = 512
params["lr"] = 4e-4

import logging
if params["running_type"] == "train": logfile_name = f"log-{params['jobID']}.log"
elif params["running_type"] == "eval": logfile_name = f"eval-{params['jobID']}.log"
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

  def forward(self, observations={"image": torch.zeros(1, 3, 128, 128)}, save_masks=False):
    
    img = observations["image"]
    img = img.to(DEVICE)
    B, C, H, W = img.shape

    x = self.encoder_cnn(img) # [B, input_dim, C] 
    x = nn.LayerNorm(x.shape[1:]).to(DEVICE)(x)
    self.features_to_slots = self.mlp(x)
    self.slots, attn = self.slot_attention(self.features_to_slots)

    if save_masks:
        aux_attn = attn.reshape((B, self.num_slots, params['resolution'][0], params['resolution'][1])) if not params["strided_convs"] else attn.reshape((B, self.num_slots, int(params['resolution'][0]/4), int(params['resolution'][1]/4)))
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

    if params["running_type"] == "eval":
      pyro.sample("mask", dist.Bernoulli(preds[:, :, 18].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("size", dist.Categorical(probs=preds[:, :, 3:5].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("mat", dist.Categorical(probs=preds[:, :, 5:7].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("shape", dist.Categorical(probs=preds[:, :, 7:10].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("color", dist.Categorical(probs=preds[:, :, 10:18].expand([params["num_inference_samples"], -1, -1])))
      pyro.sample("coords", dist.Normal(preds[:, :, :3].expand([params["num_inference_samples"], -1, -1]), torch.tensor(0.1)))
    return preds


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

def hungarian_loss_inclusive_KL(pred, target, loss_fn=F.smooth_l1_loss):
    
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
            
            batch_loss, _ = hungarian_loss(preds, target)
            #batch_loss, _ = hungarian_loss_inclusive_KL(preds, target)
            #batch_loss = 0.5*hungarian_loss_inclusive_KL(preds, target)[0] + 0.5*hungarian_loss(preds, target)[0]

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
                batch_loss, _ = hungarian_loss(preds, target)
                #batch_loss, _ = hungarian_loss_inclusive_KL(preds, target)
                #batch_loss = 0.5*hungarian_loss_inclusive_KL(preds, target)[0] + 0.5*hungarian_loss(preds, target)[0]

                for t in threshold: 
                    ap[t] += average_precision_clevr(preds.detach().cpu().numpy(), 
                                                     target.detach().cpu().numpy(), 
                                                     t)
                
                loss += batch_loss.item()
                num_iters += 1
        
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
    def __init__(self, images_path, scenes_path, max_objs=6, get_target=True):
        self.max_objs = max_objs
        self.get_target = get_target
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
            while len(target) < self.max_objs:
                target.append(torch.zeros(19, device='cpu'))
            target = torch.stack(target)       
        return img*2 - 1, target


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
          detection_set.add((original_image_idx, best_id))
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

  """ 
  eval mode in case of choosing importance sampling as inference using the trained inference network for proposals
  """

  model = clevr_gen_model
  guide = InvSlotAttentionGuide(resolution = params['resolution'],
                                num_slots = 10,
                                num_iterations = 3,
                                slot_dim = params["slot_dim"],
                                stage="eval"
                                ).to(DEVICE)
  
  checkpoint_path = os.path.join(main_dir, "inference", f"checkpoint-{params['jobID']}")
  epoch_to_load = 999
  guide.load_state_dict(torch.load(os.path.join(checkpoint_path, f"guide_{epoch_to_load}.pth")))

  logger.info(f"\Inference network from epoch {epoch_to_load} successfully loaded...")

  optimiser = pyro.optim.Adam({'lr': 1e-4})
  csis = CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

  threshold = [-1., 1., 0.5, 0.25, 0.125, 0.0625]
  ap = {k: 0 for k in threshold}

  def process_preds(trace, id):
    max_obj = max(params['max_objects'], params['num_slots'])
    
    features_dim = 19
    preds = torch.zeros(max_obj, features_dim)
    for name, site in trace.nodes.items():
        if site['type'] == 'sample':
            if name == 'shape': preds[:, 7:10] = F.one_hot(site['value'][id], len(shapes))
            if name == 'color': preds[:, 10:18] = F.one_hot(site['value'][id], len(colors))
            if name == 'size': preds[:, 3:5] = F.one_hot(site['value'][id], len(sizes))
            if name == 'mat': preds[:, 5:7] = F.one_hot(site['value'][id], len(materials))
            if name == 'coords': preds[:, :3] = site['value'][id] # [-3., 3.]
            if name == 'mask': preds[:, 18] = site['value'][id]
    return preds
  
  val_dataloader = DataLoader(val_data, batch_size = 1,
                              shuffle=False, num_workers=8, generator=torch.Generator(device='cuda'))

  
  logger.info(f"\nStarting inference with {params['num_inference_samples']} particles...\n")
  
  n_test_samples = 0
  with torch.no_grad():
    for img, target in val_dataloader:
        img = img.to(DEVICE)        

        assert params["num_inference_samples"] > 1
            
        posterior = csis.run(observations={"image": img})
        prop_traces = posterior.prop_traces[0]
        traces = posterior.exec_traces[0]
        log_wts = posterior.log_weights[0]
        resampling = Empirical(torch.stack([torch.tensor(i) for i in range(len(log_wts))]), torch.stack(log_wts))
        resampling_id = resampling().item()

        preds = process_preds(prop_traces, resampling_id)
        if len(preds.shape) == 2: preds = preds.unsqueeze(0)
        for t in threshold: 
          ap[t] += average_precision_clevr(preds.detach().cpu().numpy(),
                                           target.detach().cpu().numpy(),
                                           t)
            
        n_test_samples += 1

        if n_test_samples == 1 or n_test_samples % 100 == 0:
          logger.info(f"{n_test_samples} evaluated...")
          logger.info(f"current stats:")
          aux_mAP = {k: v/n_test_samples for k, v in ap.items()}
          logger.info(aux_mAP)


if params["running_type"] == "train": wandb.finish()
