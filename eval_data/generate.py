import torch
import torch.distributions as dist
from torchvision import transforms
import json 
from PIL import Image, ImageDraw
import os

import logging

import sys
sys.path.append(os.path.abspath(__file__+'/../../'))

from main.setup import params

logfile_name = "generate.log"
logging.basicConfig(filename=logfile_name, level=logging.INFO)

img_transform = transforms.Compose([transforms.ToTensor()])
BACKGROUND_COLOR = "black"

shape_vals = params["shape_vals"]
size_vals = params["size_vals"]
size_mapping_int = params["size_mapping_int"]
color_vals = params["color_vals"]

def color_to_rgb(color):
  if color == "red": return (255, 0, 0)
  elif color == "green": return (0, 255, 0)
  elif color == "blue": return (0, 0, 255)
  elif color == "white": return (255, 255, 255)
  elif color == "black": return (0, 0, 0)


def render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, background_color):
  """
  shape is in [ball, square]
  color is in [red, green , blue]
  size is in [small, medium, large] -> [10, 20, 40]
  locX and locY are the mass centers

  """
  
  img = Image.new('RGB', (128, 128), color_to_rgb(background_color))
  draw = ImageDraw.Draw(img)

  for n in range(len(shape_obj)):
    shape, size, color, locX, locY = shape_obj[n], size_obj[n], color_obj[n], locx_obj[n]*127, locy_obj[n]*127

    s = size_mapping_int[size]

    if shape == "square":
      draw.rectangle(
        [locX - s//2, locY - s//2, locX + s//2, locY + s//2], # (x0, y0, x1, y1)
        fill=color_to_rgb(color))
    elif shape == "ball":
      draw.ellipse(
        [locX - s//2, locY - s//2, locX + s//2, locY + s//2],
        fill=color_to_rgb(color))
  return img

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

def model(scene_id, N):

  n_objects = N

  # sample properties for each object 
  shape_obj, size_obj, color_obj, locx_obj, locy_obj = [], [], [], [], []

  scene_dict = {}
  scene_dict["scene_id"] = scene_id
  scene_dict["scene_attr"] = {
    "N": n_objects,
  }

  all_locs = []
  for n in range(int(n_objects)):
    shape = shape_vals[dist.Categorical(probs=torch.tensor([1/len(shape_vals) for _ in range(len(shape_vals))])).sample()]
    shape_obj.append(shape)
    size = size_vals[torch.randint(high=len(size_vals), size=(1,))]
    size_obj.append(size)
    color = color_vals[torch.randint(high=len(color_vals), size=(1,))]
    color_obj.append(color)
    
    # ensure there are no occlusions or object overflow
    
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

  #img = pillowrender.render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, BACKGROUND_COLOR)
  img = render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, BACKGROUND_COLOR)
  return img, scene_dict

def main():
    # path for slurm generation
    metadata_dir = 'metadata_ood'
    img_dir = 'images_ood'

    NSCENES = 1000

    dirs = [metadata_dir, img_dir]
    for d in dirs:
      if not os.path.isdir(d): os.mkdir(d)

    gen_count = range(10, 16)

    for c in gen_count:

      # set up paths for saving images and scene_dicts
      dirs = [os.path.join(img_dir, str(c)), os.path.join(metadata_dir, str(c))]
      for dir in dirs: 
        if not os.path.isdir(dir): os.mkdir(dir)
      
      # generate scenes
      for scene_id in range(NSCENES): 
        img, scene_dict = model(scene_id, c)
    
        # save scenes
        with open(f"{metadata_dir}/{str(c)}/{str(scene_id).zfill(len(str(NSCENES)))}.json", "w") as outfile:
          json.dump(scene_dict, outfile)  
        img.save(f"{img_dir}/{str(c)}/{str(scene_id).zfill(len(str(NSCENES)))}.png")

      logging.info(f"images with {c} objects saved...")


if __name__=="__main__":
    main()