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

    
    # for now use only images with up to 6 objects (this might need exploring the whole json file, which might take a while...)
    
  
  def __getitem__(self, index):
    #logger.info(self.all_scenes[index])
    
    img = self.img_transform(Image.open(self.data_path[index])).unsqueeze(0)
    logger.info(img.shape)
    img = preprocess_clevr(img).squeeze(0)
    target = self.target['scenes'][index]
    
    return img, target
  
  def __len__(self):
    return len(self.data_path)