import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from main.clevr_model import preprocess_clevr

import logging
logger = logging.getLogger("eval")

class CLEVRDataset(Dataset):
  def __init__(self, data_path, properties):
    self.data_path = data_path
    self.img_transform = transforms.Compose([
      transforms.PILToTensor()
      ])
    self.target = properties
    
  
  def __getitem__(self, index):
    #logger.info(self.all_scenes[index])
    
    img = preprocess_clevr(self.img_transform(Image.open(self.data_path[index])))
    target = self.target['scenes'][index]

    logger.info(img.shape)
    logger.info(target)
    
    return img, target