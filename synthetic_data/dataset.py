import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

# add project path to sys to import relative modules
import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

class MyDataset():
  def __init__(self, img, gt, params):
    self.img = img
    self.gt = gt
    self.params = params
    self.img_transform = transforms.Compose([transforms.ToTensor()])
  
  def __getitem__(self, index):
    img = self.img_transform(Image.open(self.img[index]))
    if isinstance(self.gt[index], str): dm = torch.from_numpy(np.load(self.gt[index])).unsqueeze(0)
    elif isinstance(self.gt[index], np.ndarray): dm = torch.from_numpy(self.gt[index]).unsqueeze(0)
    return img, dm
  
  def __len__(self):
    return len(self.img)