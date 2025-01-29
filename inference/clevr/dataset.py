import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import os
from PIL import Image
import matplotlib.pyplot as plt

from main.clevr_model import preprocess_clevr
from main.setup import params

import logging
logger = logging.getLogger("eval")

class CLEVRDataset(Dataset):
  def __init__(self, data_path, properties):
    
    if params['max_objects'] == 6:
        self.data_path = data_path
        self.all_target = properties
        self.target = []     # REMOVE WHEN USING ALL SCENES

        # for now use only images with up to 6 objects (this might need exploring the whole json file, which might take a while...)
        for idx, scene in enumerate(self.all_target['scenes']):
            if len(scene['objects']) <= 6:
                self.target.append(scene)
        
        logger.info(f'{len(self.target)} examples with 6 or - objects!')
    else:
       self.data_path = data_path
       self.target = properties['scenes']
    
  
  def __getitem__(self, index):    
    #target = self.target['scenes'][index]    ------> when using all scenes!
    target = self.target[index]
    img_filename = target['image_filename']
    img = torch.from_numpy(np.asarray((Image.open(os.path.join(
       self.data_path, img_filename
    ))))).permute(2, 0, 1) # [C, W, H]
    
    # plots_dir = os.path.abspath("set_prediction_plots")
    # plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    # plt.savefig(os.path.join(plots_dir, f"image_before_processing.png"))
    # plt.close()
    
    #logger.info(img.shape)
    img = preprocess_clevr(img.unsqueeze(0)).squeeze(0)
    #logger.info(img.shape)

    
    
    return img, target
  
  def __len__(self):
    return len(self.data_path)