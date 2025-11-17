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

shape_vals = ["ball", "square"]
size_vals = ["small", "medium", "large"]
size_mapping_int = {'small': 5, 'medium': 10, 'large': 15}
color_vals = ["red", "green", "blue"]

def check_occlusion(check_dict, locx, locy, sizes):

  """
  check_dict: dictionary with locations and sizes of all sampled objects so far
  locx: x-location for the proposed object
  locy: y-location for the proposed object
  sizes: size for the proposed object
  """

  flag = False
  w, h = (128, 128)
  eps = 10
  
  for o, _ in check_dict.items():
    x_cond = abs(locx*w - check_dict[o]['locX']*w) <= (size_mapping_int[sizes]//2 + size_mapping_int[check_dict[o]['size']]//2 + eps)
    y_cond = abs(locy*h - check_dict[o]['locY']*h) <= (size_mapping_int[sizes]//2 + size_mapping_int[check_dict[o]['size']]//2 + eps)

    if x_cond and y_cond: flag = True
  
  return flag

def sample_loc(i):
    with pyro.poutine.block(): 
        locX_mu = pyro.sample(f"locX_{i}", dist.Uniform(0.15, 0.85))
        locY_mu = pyro.sample(f"locY_{i}", dist.Uniform(0.15, 0.85))
    return locX_mu, locY_mu

def sample_size(i):
    if len(size_vals) > 1:
      size_probs = torch.tensor([1/len(size_vals) for _ in range(len(size_vals))], device=device)
      with pyro.poutine.block(): 
          size = pyro.sample(f"size_{i}", dist.Categorical(probs=size_probs))
    
    else:
      size = torch.tensor(0)
    return size

def sample_scenes():
  B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']
  M = params['max_objects']

  objects_mask = pyro.sample(f"mask", dist.Bernoulli(0.5).expand([B, M])).to(torch.bool)

  with pyro.poutine.mask(mask=objects_mask):
    shape = pyro.sample(f"shape", dist.Categorical(probs=torch.tensor([1/len(shape_vals) for _ in range(len(shape_vals))])).expand([B, M]))
    color = pyro.sample(f"color", dist.Categorical(probs=torch.tensor([1/len(color_vals) for _ in range(len(color_vals))])).expand([B, M]))                        
    # size = pyro.sample(f"size", dist.Categorical(probs=torch.tensor([1/len(size_vals) for _ in range(len(size_vals))])).expand([B, M]))
    # locx, locy = pyro.sample(f"locX", dist.Uniform(0.15, 0.85).expand([B, M])), pyro.sample(f"locY", dist.Uniform(0.15, 0.85).expand([B, M]))
  
  locX_mu_, locY_mu_ = torch.ones(B, M)*0.5, torch.ones(B, M)*0.5
  size_b_ = torch.zeros(B, M)
  i = 0

  for b in range(B):
      check_dict = {} # stores locations and sizes for all objects within a scene
      for n in range(M):
        if objects_mask[b, n]:
            # if objects have been sampled already, we need to check for overlaps 
            # while sampling new objects to add to the scene
            if n > 0:
              # sample proposed locations
              locX_mu, locY_mu = sample_loc(i) 
              size = sample_size(i)
              
              # check if proposed object violates occlusion
              # 'check_dict' holds the properties of all previously accepted objects  
              #occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_mapping[size[b, n].item()])                    
              occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_vals[size.item()])                    
              
              while occlusion_flag:

                  #logger.info(f"{b} - {n} - {occlusion_flag} - {i}")
                  
                  i += 1
                  # sample proposed locations
                  locX_mu, locY_mu = sample_loc(i)
                  size = sample_size(i)
                  #occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_mapping[size[b, n].item()]) 
                  occlusion_flag = check_occlusion(check_dict, locX_mu.item(), locY_mu.item(), size_vals[size.item()])
              
              check_dict[n] = {} # init dict for object 'n'
              check_dict[n]['locX'] = locX_mu.item()
              check_dict[n]['locY'] = locY_mu.item()
              #check_dict[n]['size'] = size_mapping[size[b, n].item()]
              check_dict[n]['size'] = size_vals[size.item()]
            
            # if n == 0, then just add the sampled locations and correspondent object's size to 'check_dict'
            else:
              locX_mu, locY_mu = sample_loc(i)
              size = sample_size(i)
              
              check_dict[n] = {} # init dict for object 'n'
              check_dict[n]['locX'] = locX_mu.item()
              check_dict[n]['locY'] = locY_mu.item()
              check_dict[n]['size'] = size_vals[size.item()]
            
            locX_mu_[b, n] = locX_mu
            locY_mu_[b, n] = locY_mu
            size_b_[b, n] = size
    
  locXfn = MyNormal(locX_mu_, torch.tensor(0.01)).get_dist()
  locYfn = MyNormal(locY_mu_, torch.tensor(0.01)).get_dist()
  with pyro.poutine.mask(mask=objects_mask):
      locX = pyro.sample(f"locX", locXfn)
      locY = pyro.sample(f"locY", locYfn)
      
      if len(size_vals) > 1:
        size = pyro.sample(f"size", dist.Delta(size_b_))
      else:
        size = torch.zeros(B, M)

  scenes = []
  for b in range(B):
    objects = []
    for m in range(M):
      if objects_mask[b, m]:
        objects.append({
            "shape": shape_vals[shape[b][m].item()],
            "color": color_vals[color[b][m].item()],
            "size": size_vals[int(size[b][m].item())],
            "position": (locX[b, m].item(), locY[b, m].item())
        })

    scenes.append(objects)
  return scenes

def model(observations={"image": torch.zeros((1, 3, 128, 128))}):

  #init_time = time.time()
  
  B = params['batch_size'] if params["running_type"] == "train" else params['num_inference_samples']
  
  scenes = sample_scenes()
  rendered_scenes = render(scenes)
  img = torch.stack([img_transform(s) for s in rendered_scenes])

  logger.info(img.shape)
  
  #render_time = time.time() - init_time
  #logger.info(f"Batch generation duration: {render_time} - {render_time/B} per sample")

  llh_uncertainty = 0.001 if params['running_type'] == "train" else 0.05
  likelihood_fn = MyNormal(img, torch.tensor(llh_uncertainty)).get_dist()

  logger.info(likelihood_fn)
  logger.info(likelihood_fn.to_event(3))

  pyro.sample("image", likelihood_fn, obs=observations["image"])


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
