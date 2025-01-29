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
from utils.guide import get_pretrained_wts, load_trained_guide
from main.clevr_model import clevr_gen_model, min_objects, max_objects

import wandb # type: ignore

import logging
logfile_name = f"log-{params['jobID']}.log"
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

main_dir = os.path.abspath(__file__+'/../../')

torch.autograd.set_detect_anomaly(True)

pyro.clear_param_store()

"""
set image and checkpoints saving paths
"""

# start a new wandb run to track this script
run = wandb.init(project="ICSA-CLEVR",
                  name=f"{params['jobID']}",
                  config={"LR": params['lr'],
                          "BS": params['batch_size'],
                          "likelihood_fn": "normal",
                          }
                  )

DEVICE = params["device"]
TRAINING_FROM_SCRATCH = params["training_from_scratch"]
GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}"
for p in [GUIDE_PATH]: 
  if not os.path.isdir(p): os.mkdir(p)
logger.info(f"\n... saving model checkpoints in {GUIDE_PATH}")

guide = InvSlotAttentionGuide(resolution = params['resolution'],
                              num_slots = params['num_slots'],
                              num_iterations = 3,
                              hid_dim = params["slot_dim"],
                              stage="train")
guide.to(DEVICE)

run.watch(guide)

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
csis = CSIS(model = model if params['dataset'] == '2Dobjects' else clevr_gen_model,
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

step_size = params["step_size"]

for s in range(resume_step, resume_step + nsteps):     
  
    if CHECK_ATTN and s % step_size == 0: 
        if not os.path.isdir(f"{root_folder}/attn-step-{s}"): os.mkdir(f"{root_folder}/attn-step-{s}")  

    csis.nstep = s

    loss = csis.step()
    val_loss = csis.validation_loss()

    if s % step_size == 0 or s == nsteps-1: 
        logger.info(f"step {s}/{resume_step + nsteps-1} - train_loss: {loss} - val_loss: {val_loss}")
        dict_to_log = {'train_loss': loss, 'val_loss': val_loss}
        run.log(dict_to_log)

        torch.save(csis.guide.state_dict(), GUIDE_PATH+'/guide_'+str(s)+'.pth')    

logger.info("\ntraining ended...")
wandb.finish()
