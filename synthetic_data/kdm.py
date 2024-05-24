"""
apply traditional kernel density maps to dot annotations
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import scipy
import os

import logging
logfile_name = "kdm.log"
logging.basicConfig(filename=logfile_name, level=logging.INFO)


def apply_kernel(input, sigma):
  density_map = scipy.ndimage.gaussian_filter(input, sigma=sigma)
  return density_map
  
def read_annot(path):
  img_transform = transforms.Compose([transforms.ToTensor()])
  return img_transform(Image.open(path)).squeeze(0)

def annot_stack(annot):
  n = int(torch.sum(annot).item())
  stack = torch.zeros((n, annot.shape[0], annot.shape[1]))
  #logging.info(f"{stack.shape}") (6, 128, 128)
  dot_locs = torch.nonzero(annot)
  logging.info(dot_locs)
  for i in range(n):
    stack[i, dot_locs[i, 0], dot_locs[i, 1]] = 1.0
  logging.info(torch.nonzero(stack))
  return stack

  
if __name__ == '__main__':
  logging.info("\n")


  ANNOT_PATH = '' # path for dataset dot annotations
  DM_PATH = '' # path for dataset density maps

  if not os.path.isdir(DM_PATH): os.mkdir(DM_PATH)
  else: 
    files = glob.glob(f'{DM_PATH}/*')
    for file in files:
      if os.path.isfile(file):
        os.remove(file)

  for annot_p in glob.glob(ANNOT_PATH + '/*.png'):
    annot = read_annot(annot_p)
    #_ = annot_stack(annot)
    density_map = apply_kernel(annot, sigma=3)
    name = annot_p.split("/")[6].split(".")[0]
    logging.info(name)
    np.save(f'{DM_PATH}/{name}.npy', density_map)
  logging.info('all density maps saved...')