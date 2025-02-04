# the PP model

import torch
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from torchvision import transforms

from utils.distributions import CategoricalVals, MyBernoulli, MyNormal, MyPoisson
from utils.generate import overflow, overlap, render
from utils.guide import to_int
from .setup import params

import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

import logging
import time

logger = logging.getLogger(params["running_type"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([transforms.ToTensor()])

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

def sample_scenes():
  B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']
  M = params['max_objects']
  
  shape_vals = ["ball", "square"]
  size_vals = ["small", "medium", "large"]
  color_vals = ["red", "green", "blue"]

  shape_to_color_probs = {
        0: torch.tensor([0.5, 0.5, 0.]), 
        1: torch.tensor([0., 0.5, 0.5]),
    }

  objects_mask = pyro.sample(f"mask", dist.Bernoulli(0.5).expand([B, M])).to(torch.bool)

  with pyro.poutine.mask(mask=objects_mask):
    shape = pyro.sample(f"shape", dist.Categorical(probs=torch.tensor([1/len(shape_vals) for _ in range(len(shape_vals))])).expand([B, M]))
    
    if not params["ood_eval"]:
      color = pyro.sample(f"color", dist.Categorical(probs=torch.tensor([1/len(color_vals) for _ in range(len(color_vals))])).expand([B, M]))
    else:
      color_probs = torch.stack([shape_to_color_probs[idx.item()] for idx in shape.flatten()]).view(B, M, -1)
      color = pyro.sample(f"color", dist.Categorical(probs=color_probs))
                        
    size = pyro.sample(f"size", dist.Categorical(probs=torch.tensor([1/len(size_vals) for _ in range(len(size_vals))])).expand([B, M]))
    locx, locy = pyro.sample(f"locX", dist.Uniform(0.15, 0.85).expand([B, M])), pyro.sample(f"locY", dist.Uniform(0.15, 0.85).expand([B, M]))
  
  scenes = []
  for b in range(B):
    objects = []
    for m in range(M):
      if objects_mask[b, m]:
        objects.append({
            "shape": shape_vals[shape[b][m].item()],
            "color": color_vals[color[b][m].item()],
            "size": size_vals[size[b][m].item()],
            "position": (locx[b, m].item(), locy[b, m].item())
        })

    scenes.append(objects)
  return scenes

def model(observations={"image": torch.zeros((1, 3, 128, 128))}):

  init_time = time.time()
  
  B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']
  
  scenes = sample_scenes()

  rendered_scenes = render(scenes)
  img = torch.stack([img_transform(s) for s in rendered_scenes])
  
  render_time = time.time() - init_time
  #logger.info(f"Batch generation duration: {render_time} - {render_time/B} per sample")

  llh_uncertainty = 0.001 if params['running_type'] == "train" else 0.1
  likelihood_fn = MyNormal(img, torch.tensor(llh_uncertainty)).get_dist()
  pyro.sample("image", likelihood_fn.to_event(3), obs=observations["image"])


# def old_model(observations={"image": torch.zeros((1, 3, 128, 128))}, show='all', save_obs=None, N=None):
                        
#   poisson_param = 5.
#   shape_vals = ["ball", "square"]
#   size_vals = ["small", "medium", "large"]
#   color_vals = ["red", "green", "blue"]

#   n_objects = pyro.sample("N", MyPoisson(torch.tensor(poisson_param), validate_args = False), obs=N)
#   if type(n_objects) == Tensor: n_objects = to_int(n_objects)
 
#   if params["infer_background"]:
#     bgR = pyro.sample("bgR", dist.Uniform(0., 1.))
#     bgG = pyro.sample("bgG", dist.Uniform(0., 1.))
#     bgB = pyro.sample("bgB", dist.Uniform(0., 1.))
#     background = [bgR, bgG, bgB]
#   else: background = None
  
#   shape_obj, size_obj, color_obj, locx_obj, locy_obj = [], [], [], [], []

#   for n in range(n_objects):
#     shape = pyro.sample(f"shape_{n}", CategoricalVals(vals=shape_vals,
#                                                      probs=torch.tensor([1/len(shape_vals) for _ in range(len(shape_vals))])))
#     shape_obj.append(shape)
#     size = pyro.sample(f"size_{n}", CategoricalVals(vals=size_vals, probs=torch.tensor([1/len(size_vals) for _ in range(len(size_vals))])))
#     size_obj.append(size)
#     color = pyro.sample(f"color_{n}", CategoricalVals(vals=color_vals,
#                                                      probs=torch.tensor([1/len(color_vals) for _ in range(len(color_vals))])))
#     color_obj.append(color)
    
#     locX, locY = pyro.sample(f"locX_{n}", dist.Uniform(0., 1.)), pyro.sample(f"locY_{n}", dist.Uniform(0., 1.))
    
#     # if n == 0:
#     #   locX, locY = pyro.sample(f"locX_{n}", dist.Uniform(0., 1.)), pyro.sample(f"locY_{n}", dist.Uniform(0., 1.))
#     # else:
#     #   locX = torch.rand(1)
#     #   locY = torch.rand(1)

#     # if n > 0:
#     #   # gather the locations and sizes of all previously sampled objects
#     #   all_locs = [(locx_obj[i], locy_obj[i]) for i in range(n)]
#     #   all_sizes = [size_obj[i] for i in range(n)]
      
#     #   occlusion_set = [occlusion((locX, locY, size), (all_locs[j][0], all_locs[j][1], all_sizes[j])) for j in range(len(all_locs))]
#     #   while any(occlusion_set):
#     #     locX, locY = torch.rand(1), torch.rand(1)
#     #     occlusion_set = [occlusion((locX, locY, size), (all_locs[j][0], all_locs[j][1], all_sizes[j])) for j in range(len(all_locs))]  
    
#     #   locX, locY = pyro.sample(f"locX_{n}", dist.Uniform(0., 1.), obs=locX), pyro.sample(f"locY_{n}", dist.Uniform(0., 1.), obs=locY)
    
#     locx_obj.append(locX)
#     locy_obj.append(locY)    

#   if show == 'all': img = img_transform(render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, background)).unsqueeze(0)
#   else:
#     if isinstance(show, list):
#       show_id = int(show[0].split('_')[1])
#       prop_list = [shape_obj, size_obj, color_obj, locx_obj, locy_obj]
#       for p in prop_list:
#         for j in range(len(p)):
#           if j > show_id: p[j] = None
      
#       img = img_transform(render(shape_obj, size_obj, color_obj, locx_obj, locy_obj, background)).unsqueeze(0)
    
#     else: raise ValueError(f"show variable equal to '{show}' unknown...")

#   with pyro.plate(observations["image"].shape[0]):
#     #pyro.sample("image", MyBernoulli(img, validate_args=False).to_event(3), obs=observations["image"])
#     likelihood_fn = MyNormal(img, torch.tensor(0.1)).get_dist()
#     pyro.sample("image", likelihood_fn.to_event(3), obs=observations["image"])
