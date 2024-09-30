import pyro
import pyro.poutine as poutine
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import json
import shutil
import time

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

import logging
logfile_name = f'slots_analysis.log'
logger = logging.getLogger("slots_analysis")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(logfile_name, mode='w')
logger.addHandler(fh)

main_dir = os.path.abspath(__file__+'/../../')

from main import models
from main import modifiedCSIS as mcsis
from utils.guide import load_trained_guide
from utils.generate import img_to_tensor
from inference.guide import InvSlotAttentionGuide
from utils.baseline import compute_AP
from main import setup

from utils.baseline import Tester, MyDataset, process_preds
from inference.baseline import Baseline

params = setup.params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
main_dir = os.path.abspath(__file__+'/../../')

shape_vals = {'ball': 0, 'square': 1}
size_vals = {'small': 0 , 'medium': 1, 'large': 2}
color_vals = {'red': 0, 'green': 1, 'blue': 2}
    
def main():    

    logger.info(device)

    seeds = [1]
    
    for seed in seeds: 
        pyro.set_rng_seed(seed)
        
        model = models.model
        
        # set up trained guide and 'csis' object
        guide = InvSlotAttentionGuide(resolution = (128, 128),
                                    num_iterations = 3,
                                    hid_dim = 64,
                                    stage = "eval",
                                    mixture_components=params["mixture_components"])
        guide.to(device)
        
        GUIDE_PATH = f"{main_dir}/checkpoint-{params['jobID']}/guide_{params['guide_step']}.pth"
        if os.path.isfile(GUIDE_PATH): guide = load_trained_guide(guide, GUIDE_PATH)
        else: raise ValueError(f'{GUIDE_PATH} is not a valid path!')

        optimiser = pyro.optim.Adam({'lr': 1e-4})
        csis = mcsis.CSIS(model, guide, optimiser, training_batch_size=256, num_inference_samples=params["num_inference_samples"])

        baseline_path = f"{main_dir}/inference/baseline"
        if not os.path.isdir(baseline_path): os.mkdir(baseline_path)
        checkpoint_path = baseline_path + '/checkpoints'
        loss_path = baseline_path + '/loss'
        test_path = baseline_path + '/test_plots'
        
        samesh_params = {
            "num_epochs": 1000,
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "lr": 1e-3,
            "batch_size": 256,
            "max_objects": 17,
            "checkpoint_path": checkpoint_path,
            "loss_path": loss_path,
            "epoch_log_rate": 1,
            "mode": "train",
            "epoch_to_load": 100,
            "slot_dim": 64,
            "test_image_save_path": test_path
        }
       
       
        SAMESH_GUIDE_PATH = f"/Users/franciscosilva/Downloads/model_epoch_{samesh_params['epoch_to_load']}.pth"
        samesh = Baseline(resolution = (128, 128), num_iterations = 3, hid_dim = samesh_params["slot_dim"], stage="eval", num_slots=samesh_params['max_objects'], save_slots=True)
        if device == 'cuda:0': samesh.load_state_dict(torch.load(SAMESH_GUIDE_PATH))
        else: samesh.load_state_dict(torch.load(SAMESH_GUIDE_PATH, map_location='cpu'))
        samesh.to(device)
        
        logger.info(f'seed {seed}')
        logger.info(GUIDE_PATH)
        logger.info(SAMESH_GUIDE_PATH)

        COUNT = 2

        save_plots_dir = '/Users/franciscosilva/Downloads/slots_analysis'
        
        # run the inference module
        count_img_path = glob.glob(f'/Users/franciscosilva/Downloads/eval_data/images/{COUNT}/*.png')
        count_img_path.sort()

        for img_path in count_img_path:                             
            
            """
            ICSA inference
            """
            
            sample = img_to_tensor(Image.open(img_path))      
            sample = sample.to(device)
            sample_id = img_path.split('/')[-1].split('.')[0]
            
            plt.imshow(sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(f'{save_plots_dir}/image_{sample_id}.png')
            plt.close()
            
            logger.info(sample_id)
            
            # in ICSA set prediction, we assume that 'N' is known to speed up inference
            _ = csis.run(observations={"image": sample}, N=COUNT)

            """
            SA-MESH inference
            """

            test_dataset = MyDataset(img = [img_path],
                                    target = [],
                                    params = samesh_params)

            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=samesh_params['batch_size'], shuffle=False)
            tester = Tester(samesh, testloader, samesh_params)
            samesh_preds, samesh_slots, samesh_attn = tester.slots_analysis()

            # save SA-MESH slots
            # save SA-MESH attn masks
            np.save(f"{save_plots_dir}/samesh_slots/slots_samesh_10000_{sample_id}.npy", samesh_slots.numpy())
            np.save(f"{save_plots_dir}/samesh_attn/attn_samesh_10000_{sample_id}.npy", samesh_attn.detach().numpy())
            np.save(f"{save_plots_dir}/samesh_preds/preds_samesh_10000_{sample_id}.npy", samesh_preds.detach().numpy())

            # aux_attn = samesh_attn.reshape((1, 17, 32, 32))
            # fig, ax = plt.subplots(ncols=17)
            # for j in range(17):                                       
            #     im = ax[j].imshow(aux_attn[0, j, :, :].detach().cpu().numpy())
            #     ax[j].grid(False)
            #     ax[j].axis('off')        
            # plt.savefig(f"{save_plots_dir}/attn_samesh_10000.png")
            # plt.close()
            
            #break

            

            
            

if __name__ == '__main__':
    main()