import os
import shutil
import torch
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

import logging
logfile_name = f"log-100.log"
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

logger.info(os.path.abspath(__file__+'/../../../'))

import wandb # type: ignore

from main.setup import params

from utils.guide import minimize_entropy_of_sinkhorn, sinkhorn

params["batch_size"] = 512
params["lr"] = 4e-4

"""
set image and checkpoints saving paths
"""

# start a new wandb run to track this script
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

  def forward(self, img):
    
    img = img.to(DEVICE)
    B, C, H, W = img.shape

    x = self.encoder_cnn(img) # [B, input_dim, C] 
    x = nn.LayerNorm(x.shape[1:]).to(DEVICE)(x)
    self.features_to_slots = self.mlp(x)
    self.slots, attn = self.slot_attention(self.features_to_slots)

    if self.is_train:
        aux_attn = attn.reshape((B, self.num_slots, params['resolution'][0], params['resolution'][1])) if not params["strided_convs"] else attn.reshape((B, self.num_slots, int(params['resolution'][0]/4), int(params['resolution'][1]/4)))
        fig, ax = plt.subplots(ncols=self.num_slots)
        for j in range(self.num_slots):                                       
            im = ax[j].imshow(aux_attn[0, j, :, :].detach().cpu().numpy())
            ax[j].grid(False)
            ax[j].axis('off')        
        plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.epoch}/attn.png")
        plt.close()

        plot_img = visualize(np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0)))
        plt.imshow(plot_img)
        plt.axis('off')
        plt.savefig(f"{params['check_attn_folder']}/attn-step-{self.epoch}/img.png")
        plt.close()
    
    preds = self.mlp_preds(self.slots)
    preds[:, :, 0:3] = self.sigmoid(preds[:, :, 0:3].clone())       # coords
    preds[:, :, 3:5] = self.softmax(preds[:, :, 3:4].clone())       # size
    preds[:, :, 5:7] = self.softmax(preds[:, :, 5:7].clone())       # material
    preds[:, :, 7:10] = self.softmax(preds[:, :, 7:10].clone())     # shape
    preds[:, :, 10:18] = self.softmax(preds[:, :, 10:18].clone())   # color
    preds[:, :, 18] = self.sigmoid(preds[:, :, 18].clone())         # real object
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

class Trainer:
    def __init__(self, model, dataloaders, params, logger): 
        self.trainloader = dataloaders["train"]
        self.validloader = dataloaders["validation"]
        self.params = params
        self.model = model
        self.num_iters = 0
        self.epoch = 0
        self.num_epochs = 1000
        self.device = self.params['device']
        self.optimizer = torch.optim.Adam([p for p in list(self.model.parameters()) if p.requires_grad], lr = self.params['lr'])
        self.logger = logger
    
    def _save_checkpoint(self, epoch):
        if not os.path.isdir(self.params['checkpoint_path']):
            try: os.mkdir(self.params['checkpoint_path'])
            except: logger.info('unable to create directory to save training checkpoints!')
        else:
            path = self.params['checkpoint_path']
            torch.save(self.model.state_dict(), path + '/model_epoch_' + str(epoch) + '.pth')

    def _train_epoch(self):
        loss = 0.
        num_iters = 0
        self.model.train() 
        for img, target in self.trainloader:
            img, target = img.to(self.device), target.to(self.device)
            preds = self.model(img)
            batch_loss, _ = hungarian_loss(preds, target)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step() 
            
            loss += batch_loss.item()
            num_iters += 1
        return loss/num_iters
    
    def _valid_epoch(self):
        loss = 0.
        num_iters = 0
        self.model.eval()
        with torch.no_grad(): 
            for img, target in self.validloader:
                img, target = img.to(self.device), target.to(self.device)
                preds = self.model(img)
                batch_loss, _ = hungarian_loss(preds, target)
                
                loss += batch_loss.item()
                num_iters += 1
        return loss/num_iters

    def train(self, root_folder):
        since = time.time()  
        train_loss, valid_loss = [], [] 

        for epoch in range(self.num_epochs):                  
            
            self.epoch = epoch
            self.model.epoch = epoch
            self.model.is_train = True
            
            if epoch % 1 == 0:
                logger.info("Epoch {}/{}".format(epoch, self.num_epochs - 1))

            if not os.path.isdir(f"{root_folder}/attn-step-{epoch}"): os.mkdir(f"{root_folder}/attn-step-{epoch}")  
                    
            epoch_train_loss = self._train_epoch()
            train_loss.append(epoch_train_loss) 
            
            self.model.is_train = False
            
            epoch_valid_loss = self._valid_epoch()
            valid_loss.append(epoch_valid_loss)      

            loss_dic = {"train_loss": train_loss,
                        "valid_loss": valid_loss}
            
            if epoch % 1 == 0 or epoch == self.num_epochs-1:
                logger.info("... train_loss: {:.3f}" .format(train_loss[-1]))
                logger.info("... valid_loss: {:.3f}" .format(valid_loss[-1]))

                self.logger.log({"train_loss": train_loss[-1],
                                 "val_loss": valid_loss[-1]})
        
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
        return img, target


def process_preds(preds):
    # preds must have shape (max_objects, n_features)
    assert len(preds.shape) == 2

    shape = torch.argmax(preds[:, :2], dim=-1)
    size = torch.argmax(preds[:, 2:5], dim=-1)
    color = torch.argmax(preds[:, 5:8], dim=-1)
    locx, locy = preds[:, 8], preds[:, 9]
    real_obj = preds[:, 10]
    return shape, size, color, locx, locy, real_obj

def distance(loc1, loc2):
    return torch.sqrt(torch.square(loc1[0]-loc2[0]) + torch.square(loc1[1]-loc2[1]))

def compute_AP(preds, targets, threshold_dist):

    """
    adapted from 'https://github.com/google-research/google-research/blob/master/slot_attention/utils.py'
    """

    # preds have shape (max_objects, n_features)
    # targets have shape (max_objects, n_features)

    shape, size, color, locx, locy, pred_real_obj = process_preds(preds)
    target_shape, target_size, target_color, target_locx, target_locy, target_real_obj = process_preds(targets)

    max_objects = shape.shape[0]
    
    tp = np.zeros(1)
    fp = np.zeros(1)
    
    found_objects = []
    for o in range(max_objects):
        if torch.round(pred_real_obj[o]):

            found = False
            found_idx = -1 
            best_distance = 1000
            
            for j in range(max_objects):
                if target_real_obj[j]:
                    if [shape[o], size[o], color[o]] == [target_shape[j], target_size[j], target_color[j]]: 
                        dist = distance((locx[o], locy[o]), (target_locx[j], target_locy[j]))
                        if dist < best_distance and j not in found_objects:
                            #logger.info(f'found at best distance {dist}')
                            found = True
                            best_distance = dist
                            found_idx = j # stores the best match between an object and all possible targets
            
            if found:
                if distance((locx[o], locy[o]), (target_locx[found_idx], target_locy[found_idx])) <= threshold_dist or threshold_dist == -1:
                    found_objects.append(found_idx)
                    #logger.info('found match below distance threshold!')
                    tp += 1
            else: fp += 1

            #logger.info(found_objects)
    
    precision = tp / (tp+fp)
    recall = tp / np.sum(np.asarray(target_real_obj.cpu()))

    #logger.info(f'precision: {precision}')
    #logger.info(f'recall: {recall}')

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

    #logger.info(f'ap: {average_precision}')
    return average_precision





guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_slots = 10,
                              num_iterations = 3,
                              slot_dim = params["slot_dim"],
                              stage="train"
                              ).to(DEVICE)

run.watch(guide)

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
train_dataloader = DataLoader(train_data, batch_size = params["batch_size"],
                              shuffle=True, generator=torch.Generator(device='cuda'))
val_images_path = os.path.join(dataset_path, 'images/val')
val_data = CLEVR(images_path = os.path.join(dataset_path, 'images/val'),
                   scenes_path = os.path.join(dataset_path, 'scenes/CLEVR_val_scenes.json'),
                   max_objs=10)
val_dataloader = DataLoader(val_data, batch_size = params["batch_size"],
                              shuffle=False, generator=torch.Generator(device='cuda'))


trainer = Trainer(guide, {"train": train_dataloader, "validation": val_dataloader}, params, run)
trainer.train(root_folder)
 

logger.info("\ntraining ended...")
wandb.finish()
