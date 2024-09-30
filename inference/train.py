import os
import shutil
import torch
import pyro
import pyro.infer
import pyro.optim
import pickle as pkl

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from guide import InvSlotAttentionGuide
from main.models import model
from main.modifiedCSIS import CSIS
from utils.var import Variable
from main.setup import params
from utils.loss import save_loss, save_loss_plot
from utils.guide import get_pretrained_wts, load_trained_guide
from main.clevr_model import clevr_model

import logging
logfile_name = f"log-{params['jobID']}.log"
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

main_dir = os.path.abspath(__file__+'/../../')

pyro.clear_param_store()

"""
set image and checkpoints saving paths
"""

DEVICE = params["device"]
TRAINING_FROM_SCRATCH = params["training_from_scratch"]
GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}"
LOSS_PATH = f"{main_dir}/loss-{params['jobID']}"
for p in [GUIDE_PATH, LOSS_PATH]: 
  if not os.path.isdir(p): os.mkdir(p)
logger.info(f"\n... saving model checkpoints in {GUIDE_PATH}")
logger.info(f"... saving loss values in {LOSS_PATH}\n")

guide = InvSlotAttentionGuide(resolution = (128, 128), num_iterations = 3, hid_dim = params["slot_dim"], stage="train", mixture_components=params["mixture_components"])
guide.to(DEVICE)

"""
build CSIS class with model & guide
"""

if not TRAINING_FROM_SCRATCH:
  pretrained_guide_path = GUIDE_PATH + "/guide_" + str(params["guide_step"]) + ".pth"
  logger.info(f"pretrained guide path: {pretrained_guide_path}")
  resume_training = False
  guide = load_trained_guide(guide, pretrained_guide_path)
  logger.info(f"Guide from step {params['guide_step']} successfully loaded...\n")
  resume_training = True

else: 
  logger.info(f"Training from scratch using {DEVICE}...\n")
  resume_training = False

TRAIN_BATCH_SIZE = params["batch_size"]
VAL_BATCH_SIZE = params["batch_size"]
LR = params["lr"]

optimiser = pyro.optim.Adam({'lr': LR})
csis = CSIS(model = model if params['dataset'] == '2Dobjects' else clevr_model,
            guide = guide,
            optim = optimiser,
            num_inference_samples=params["num_inference_samples"],
            training_batch_size=TRAIN_BATCH_SIZE,
            validation_batch_size=VAL_BATCH_SIZE)
nsteps = params["training_iters"]

if resume_training: resume_step = int(pretrained_guide_path.split("/")[-1].split(".")[0].split("_")[1]) + 1
else: resume_step = 0

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

if TRAINING_FROM_SCRATCH: train_hist, valid_hist = [], []
else:
  if os.path.isfile(LOSS_PATH + "/loss_dict.pkl"):
    with open(LOSS_PATH + "/loss_dict.pkl", 'rb') as f:
        data = pkl.load(f)
        valid_hist = data["valid_loss"]
        if len(valid_hist) != 0: logger.info("Training and validation losses were successfully loaded!")
        else: logger.info("Error while loading losses from previous epochs!")
  else: 
      logger.info("Couldn't find training losses from previous epochs...")
      train_hist, valid_hist = [], []

step_size = 1 if params["running_type"] == "debug" else 10

for s in range(resume_step, resume_step + nsteps):    
  
  if CHECK_ATTN and s % step_size == 0: 
    if not os.path.isdir(f"{root_folder}/attn-step-{s}"): os.mkdir(f"{root_folder}/attn-step-{s}")  
  
  loss = csis.step(s)
  val_loss = csis.validation_loss(s)
  #train_hist.append(loss)
  valid_hist.append(val_loss) 

  loss_dic = {#"train_loss": train_hist, 
                "valid_loss": valid_hist} 
  if params["running_type"] == "train" and params["training_iters"] > 1: 
    save_loss(LOSS_PATH, loss_dic)
    save_loss_plot(loss_dic, LOSS_PATH, resume_step, nsteps)

  if s % step_size == 0 or s == nsteps-1: 
    logger.info(f"step {s}/{resume_step + nsteps-1} - train_loss: {loss} - val_loss: {val_loss}")
    if params["running_type"] == "train": 
      torch.save(csis.guide.state_dict(), GUIDE_PATH+'/guide_'+str(s)+'.pth')    

logger.info("\ntraining ended...")
