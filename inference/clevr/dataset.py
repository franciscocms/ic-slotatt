import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from PIL import Image

from main.clevr_model import preprocess_clevr

import logging
logger = logging.getLogger("eval")

class CLEVRDataset(Dataset):
  def __init__(self, data_path, properties):
    self.all_data_path = data_path
    self.img_transform = transforms.Compose([
      transforms.PILToTensor()
      ])
    self.all_target = properties
    
    self.data_path = []  # REMOVE WHEN USING ALL SCENES
    self.target = []     # REMOVE WHEN USING ALL SCENES

    
    # for now use only images with up to 6 objects (this might need exploring the whole json file, which might take a while...)
    for idx, scene in enumerate(self.all_target['scenes']):
      if len(scene['objects']) <= 6:
        self.target.append(scene)
        self.data_path.append(data_path[idx])
    
    logger.info(f'{len(self.target)} - {len(self.data_path)} examples with 6 or - objects!')
    
  
  def __getitem__(self, index):    
    img = torch.from_numpy(np.asarray((Image.open(self.data_path[index])))).unsqueeze(0)
    logger.info(img.shape)
    img = preprocess_clevr(img).squeeze(0)
    #target = self.target['scenes'][index]    ------> when using all scenes!
    target = self.target[index]
    
    return img, target
  
  def __len__(self):
    return len(self.data_path)