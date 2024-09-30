import torch
import torch.nn as nn
import numpy as np
import torch.distributions as dist
from torchvision import transforms
import json 
from PIL import Image
import os
import glob

from pillowrender import render

import logging
logfile_name = "generate.log"
logging.basicConfig(filename=logfile_name, level=logging.INFO)

import sys
sys.path.append(os.path.abspath(__file__+'/../../'))
folder_dir = os.path.abspath(__file__+'/../')

"""
generation of images to train an object density map estimator network
"""

img_transform = transforms.Compose([transforms.ToTensor()])
BACKGROUND_COLOR = "black"

def occlusion(new_object, object):
  new_object_loc_x, new_object_loc_y, new_object_size = new_object
  object_loc_x, object_loc_y, object_size = object

  if object_size == 'small': s = 10
  if object_size == 'medium': s = 15
  if object_size == 'large': s = 20
  
  if new_object_size == 'small': new_s = 10
  if new_object_size == 'medium': new_s = 15
  if new_object_size == 'large': new_s = 20

  x_cond = abs(new_object_loc_x*127 - object_loc_x*127) <= (new_s//2 + s//2)
  y_cond = abs(new_object_loc_y*127 - object_loc_y*127) <= (new_s//2 + s//2)
  if x_cond and y_cond: return True
  else: return False

def overflow(locx, locy, object_size):
  if object_size == 'small': s = 10
  if object_size == 'medium': s = 15
  if object_size == 'large': s = 20
  x_cond = locx*127 + s//2 > 127 or locx*127 - s//2 < 0
  y_cond = locy*127 + s//2 > 127 or locy*127 - s//2 < 0
  if x_cond or y_cond: return True
  else: return False

def model(scene_id):
                        
  # sample the number of objects
  poisson_param = 3.
  shape_vals = ["ball", "ood"]
  size_vals = ["large"]
  color_vals = ["red", "green", "blue"]

  n_objects = dist.Poisson(torch.tensor(poisson_param)).sample()
  #n_objects = 2

  # sample properties for each object 

  shape_obj, size_obj, color_obj, locx_obj, locy_obj = [], [], [], [], []

  scene_dict = {}
  scene_dict["scene_id"] = scene_id
  scene_dict["scene_attr"] = {
    "N": n_objects.item() if type(n_objects) != int else n_objects,
  }

  all_locs = []
  for n in range(int(n_objects)):
    shape = shape_vals[dist.Categorical(probs=torch.tensor([1/len(shape_vals) for _ in range(len(shape_vals))])).sample()]
    #shape = shape_vals[n]
    shape_obj.append(shape)
    size = size_vals[torch.randint(high=len(size_vals), size=(1,))]
    size_obj.append(size)
    color = color_vals[torch.randint(high=len(color_vals), size=(1,))]
    color_obj.append(color)
    
    # ensure there are no occlusions
    
    locX, locY = torch.rand(1), torch.rand(1) 
    while overflow(locX, locY, size): locX, locY = torch.rand(1), torch.rand(1)

    if n > 0:
      
      # gather the locations and sizes of all previously sampled objects
      all_locs = [(scene_dict['scene_attr']['object_' + str(i)]['initLocX'],
                  scene_dict['scene_attr']['object_' + str(i)]['initLocY']) for i in range(n)]
      all_sizes = [scene_dict['scene_attr']['object_' + str(i)]['size'] for i in range(n)]
      
      occlusion_set = [occlusion((locX, locY, size), (all_locs[j][0], all_locs[j][1], all_sizes[j])) for j in range(len(all_locs))]
      overflow_flag = overflow(locX, locY, size)
      while any(occlusion_set) or overflow_flag:
        locX, locY = torch.rand(1), torch.rand(1)
        occlusion_set = [occlusion((locX, locY, size), (all_locs[j][0], all_locs[j][1], all_sizes[j])) for j in range(len(all_locs))]
        overflow_flag = overflow(locX, locY, size)

    locx_obj.append(locX)
    locy_obj.append(locY)

    scene_dict["scene_attr"]["object_" + str(n)] = {
      "shape": shape,
      "color": color,
      "size": size,
      "initLocX": locX.item(),
      "initLocY": locY.item()
      }

  img = render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, BACKGROUND_COLOR)
  return img, scene_dict


NSCENES = 1

# path for slurm generation
metadata_dir = f'{folder_dir}/ood_samples_metadata' # path for dataset metadata
img_dir = f'{folder_dir}/ood_samples' # path for dataset images
dirs = [metadata_dir, img_dir]
for d in dirs:
  if not os.path.isdir(d): os.mkdir(d)
  else: 
    files = glob.glob(os.path.abspath(f'{d}/*'))
    for file in files:
      if os.path.isfile(file):
        os.remove(file)

for scene_id in range(NSCENES):
  logging.info(scene_id)
  img, scene_dict = model(scene_id)

  with open(f"{metadata_dir}/{str(scene_id).zfill(5)}.json", "w") as outfile:
    json.dump(scene_dict, outfile)  
  img.save(f"{img_dir}/{str(scene_id).zfill(5)}.png")
logging.info("all images generated...")