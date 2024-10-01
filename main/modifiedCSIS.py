import pyro
#from pyro.infer import Importance
from .modifiedImportance import Importance
import pyro.poutine as poutine
import torch
import torch.nn.functional as F
import pyro.distributions as dist
from pyro.poutine.runtime import apply_stack
from scipy.optimize import linear_sum_assignment
import numpy as np

import math
import itertools
import logging
import copy

from utils.guide import to_int
from utils.var import Variable
from utils.distributions import CategoricalVals
from .setup import params as p
from collections import OrderedDict, defaultdict

logger = logging.getLogger("train")

device = p["device"]

class CSIS(Importance):
  """
  Compiled Sequential Importance Sampling, allowing compilation of a guide
  program to minimise KL(model posterior || guide), and inference with
  importance sampling.

  **Reference**
  "Inference Compilation and Universal Probabilistic Programming" `pdf https://arxiv.org/pdf/1610.09900.pdf`

  :param model: probabilistic model defined as a function. Must accept a
      keyword argument named `observations`, in which observed values are
      passed as, with the names of nodes as the keys.
  :param guide: guide function which is used as an approximate posterior. Must
      also accept `observations` as keyword argument.
  :param optim: a Pyro optimizer
  :type optim: pyro.optim.PyroOptim
  :param num_inference_samples: The number of importance-weighted samples to
      draw during inference.
  :param training_batch_size: Number of samples to use to approximate the loss
      before each gradient descent step during training.
  :param validation_batch_size: Number of samples to use for calculating
      validation loss (will only be used if `.validation_loss` is called).
  """

  def __init__(self, model, guide, optim, num_inference_samples=10, training_batch_size=10, validation_batch_size=20):
    super().__init__(model, guide, num_inference_samples)
    self.model = model
    self.guide = guide
    self.optim = optim
    self.training_batch_size = training_batch_size
    self.validation_batch_size = validation_batch_size
    self.validation_batch = None
    self.train = True

    self.shape_map = {"ball": 0, "square": 1}

  def set_validation_batch(self, *args, **kwargs):
    """
    Samples a batch of model traces and stores it as an object property.

    Arguments are passed directly to model.
    """
    self.validation_batch = [
        self._sample_from_joint(*args, **kwargs)
        for _ in range(self.validation_batch_size)
    ]

  def step(self, s, *args, **kwargs):
    """
    :returns: estimate of the loss
    :rtype: float

    Take a gradient step on the loss function. Arguments are passed to the
    model and guide.
    """
    
    self.guide.train = True
    self.guide.step = s

    self.guide.attn_overlap = torch.tensor(0., device=device)
    
    with poutine.trace(param_only=True) as param_capture:
        loss = self.loss_and_grads(True, None, s, *args, **kwargs)

    params = set(
        site["value"].unconstrained()
        for site in param_capture.trace.nodes.values()
        if site["value"].grad is not None
    )
    
    # optim_state = self.optim.get_state()
    # for k, v in optim_state.items():
    #   logging.info(f"{k} -- {v}")
    
    self.optim(params)
    pyro.infer.util.zero_grads(params)
    return loss.item()

  def loss_and_grads(self, grads, batch, s, *args, **kwargs):
    """
    :returns: an estimate of the loss (expectation over p(x, y) of
        -log q(x, y) ) - where p is the model and q is the guide
    :rtype: float

    If a batch is provided, the loss is estimated using these traces
    Otherwise, a fresh batch is generated from the model.

    If grads is True, will also call `backward` on loss.

    `args` and `kwargs` are passed to the model and guide.
    """
    if batch is None:
      batch = (
        self._sample_from_joint(*args, **kwargs)
        for _ in range(self.training_batch_size)
      )
      batch_size = self.training_batch_size
    else:
      batch_size = len(batch)

    loss = 0
    
    hidden_addr = ["N"]
    if p["loc_proposal"] == "wo_net": 
      hidden_addr.append("locX")
      hidden_addr.append("locY")
    
    self.guide.batch_idx = 0
    
    for model_trace in batch:     
      for site, vals in model_trace.nodes.items(): 
        name = vals["name"]        
        if name not in ["image"] and vals["type"] == "sample": 
          if name == "N": self.n_objects = to_int(vals["value"])
          
          # prior categorical distributed variables
          if isinstance(vals["fn"], CategoricalVals) or isinstance(vals["fn"], dist.Categorical): 
            prior_distribution = "categorical"
            proposal_distribution = "categorical"
            out_dim = len(vals["fn"].probs)
          
          # prior uniform distributed variables
          elif isinstance(vals["fn"], dist.Uniform): 
            prior_distribution = "uniform"
            proposal_distribution = "normal"
            out_dim = 2
          
          # prior poisson distributed variables
          elif isinstance(vals["fn"], dist.Poisson): 
            prior_distribution = "poisson"
            if p["N_proposal"] == "normal":
              proposal_distribution = "normal"
              out_dim = 2
            elif p["N_proposal"] == "mixture":
              proposal_distribution = "mixture"
              MIXTURE_COMPONENTS = p["mixture_components"]
              out_dim = 3*MIXTURE_COMPONENTS
            else: raise ValueError(f"Unknown proposal for N: {p['N_proposal']}")
            
          known_addr = [k.address for k in self.guide.current_trace]
          if name.split("_")[0] not in known_addr:
            instance = 0
            var = Variable(name=name,
                          value=vals["value"],
                          prior_distribution=prior_distribution,
                          proposal_distribution=proposal_distribution,
                          address=name.split("_")[0],
                          instance=instance)          
          else:
            for v in reversed(self.guide.current_trace):
              if v.address == name.split("_")[0]:
                last_instance = v.instance
                break
            instance = last_instance + 1

            var = Variable(name=name,
                          value=vals["value"],
                          prior_distribution=prior_distribution,
                          proposal_distribution=proposal_distribution,
                          address=name.split("_")[0],
                          instance=instance) 
          
          
          #if var.name not in self.guide.prop_nets:
          if var.address not in self.guide.prop_nets:
            if var.address not in hidden_addr:
              logging.info(f"... proposal net was added for variable '{var.name}'")
              self.guide.add_proposal_net(var, out_dim)
          #else: logging.info(f"... proposal net already existed for variable '{var.name}'")
          
          self.guide.current_trace.append(var)
      
      with poutine.trace(param_only=True) as particle_param_capture:
        guide_trace = self._get_matched_trace(model_trace, *args, **kwargs)
      
      # for name, site in guide_trace.nodes.items():
      #   if site['type'] == 'sample':
      #     logging.info(f"{name} - {site}\n")
      
      self.guide.batch_idx += 1

      particle_loss = self._differentiable_loss_particle(guide_trace)

      #logging.info(particle_loss)

      particle_loss /= batch_size 

      if grads:
        guide_params = set(
          site["value"].unconstrained()
          for site in particle_param_capture.trace.nodes.values()
        )
        
        guide_grads = torch.autograd.grad(
          particle_loss, guide_params, allow_unused=True
        )

        for guide_grad, guide_param in zip(guide_grads, guide_params):
          if guide_grad is None:
              continue
          guide_param.grad = (
             guide_grad
             if guide_param.grad is None
             else guide_param.grad + guide_grad
          )
          # logging.info(guide_grad)
          # logging.info(guide_param)
          # logging.info("\n")

        # if p["running_type"] == "debug":
        #   for name, param in self.guide.named_parameters():
        #     #if name.split(".")[0] == "slot_attention":
        #     logging.info(f"{name} - {param.requires_grad} - {param.grad}")

      loss += particle_loss
    #warn_if_nan(loss, "loss")
    return loss

  def _differentiable_loss_particle(self, guide_trace):
    
    if p["running_type"] == "debug" and not p['perm_inv_loss']:
      logp = 0
      for name, vals in guide_trace.nodes.items():
        if vals["type"] == "sample":
          logging.info(f"\n---- {name} ----\n")
          if name == "N" or name.split("_")[0] in ["locX", "locY", "bgR", "bgG", "bgB"]: logging.info(f"mean: {vals['fn'].mean.item()} - GT: {vals['value']} - log_prob = {vals['fn'].log_prob(vals['value']).item()}")
          if name.split("_")[0] in ["shape", "color", "size"]: logging.info(f"probs: {vals['fn'].probs} - GT: {vals['value']} - log_prob = {vals['fn'].log_prob(vals['value']).item()}")
          logp += vals['fn'].log_prob(vals['value']).item()
      
    if not p["perm_inv_loss"]: return -guide_trace.log_prob_sum()
    else:
      
      # if self.n_objects == 0: return -guide_trace.log_prob_sum()
      # else:
      
      """
      save the GT latents separately
      - since the DME is not being trained here, the error associated with 'N' is not considered
      """
      
      true_latents = {}
      for name, vals in guide_trace.nodes.items():
        if vals["type"] == "sample" and len(name.split('_')) == 2: # only consider object-wise properties
          true_latents[name] = vals['value']
      
      self.n_latents = len(set([k.split('_')[0] for k in true_latents.keys()]))

      pdist = torch.tensor([], device=device)
      for i in range(self.n_objects):
        # modify guide_trace considering the values in 'true_latents' and compute the loss
        for name, vals in guide_trace.nodes.items():
          if vals["type"] == "sample" and len(name.split('_')) == 2: # only consider object-wise properties
            alt_property = f"{name.split('_')[0]}_{i}"
            vals['value'] = true_latents[alt_property]

        partial_loss = self.my_log_prob(guide_trace, self.n_objects, self.n_latents) # 'partial_loss' shape (1, 1, NOBJECTS, NLATENTS)
        pdist = torch.cat((pdist, partial_loss), dim=-3)

      loss, _ = self.hungarian_loss(pdist)
      if len(loss.shape) == 1: loss = loss[0]

      # add the background log_prob_sum component to the loss 

      if p["infer_background"]:
        bg_log_prob_sum = 0.
        for name, vals in guide_trace.nodes.items():
          if vals["type"] == "sample" and name[:2] == 'bg': # only consider background properties
            bg_log_prob_sum -= vals['fn'].log_prob(vals['value'])
        
        return loss + bg_log_prob_sum
      else:
        return loss

  def my_log_prob(self, guide_trace, n_objects, n_latents):
    
    """
    returns a tensor with shape (1, NOBJECTS, NLATENTS) with the log_prob of the proposals for all object's properties
    in a certain guide_trace
    """

    loss = []
    for name, vals in guide_trace.nodes.items():
      if vals["type"] == "sample" and len(name.split('_')) == 2: # only consider object-wise properties
        partial_loss = -vals['fn'].log_prob(vals['value'])
        if len(partial_loss.shape) == 1: partial_loss.unsqueeze_(0)
        loss.append(partial_loss)
    
    loss = torch.stack(loss).reshape(1, 1, n_objects, n_latents)

    return loss

  def hungarian_loss(self, pdist):
    
    pdist = pdist.mean(-1)
    pdist_ = pdist.detach().cpu().numpy()

    indices = np.array([linear_sum_assignment(p) for p in pdist_])

    indices_ = indices.shape[2] * indices[:, 0] + indices[:, 1]
    losses = torch.gather(pdist.flatten(1,2), 1, torch.from_numpy(indices_).to(device=pdist.device))
    total_loss = losses.mean(1)

    return total_loss, dict(indices=indices)

  def validation_loss(self, step, *args, **kwargs):
    """
    :returns: loss estimated using validation batch
    :rtype: float

    Calculates loss on validation batch. If no validation batch is set,
    will set one by calling `set_validation_batch`. Can be used to track
    the loss in a less noisy way during training.

    Arguments are passed to the model and guide.
    """

    if p["running_type"] == "debug":
      logging.info("\n\nComputing validation loss...\n")


    if self.validation_batch is None:
      self.set_validation_batch(*args, **kwargs)
    
    self.guide.train = False
    
    with torch.no_grad():
      val_loss = self.loss_and_grads(False, self.validation_batch, step, *args, **kwargs)

    return val_loss.item()

  def _get_matched_trace(self, model_trace, *args, **kwargs):
    """
    :param model_trace: a trace from the model
    :type model_trace: pyro.poutine.trace_struct.Trace
    :returns: guide trace with sampled values matched to model_trace
    :rtype: pyro.poutine.trace_struct.Trace

    Returns a guide trace with values at sample and observe statements
    matched to those in model_trace.

    `args` and `kwargs` are passed to the guide.
    """
    kwargs["observations"] = {}
    for node in itertools.chain(
        model_trace.stochastic_nodes, model_trace.observation_nodes
    ):
        if "was_observed" in model_trace.nodes[node]["infer"]:
            model_trace.nodes[node]["is_observed"] = True
            kwargs["observations"][node] = model_trace.nodes[node]["value"]

    guide_trace = poutine.trace(poutine.replay(self.guide, model_trace)).get_trace(
        *args, **kwargs
    )

    #check_model_guide_match(model_trace, guide_trace)
    #guide_trace = prune_subsample_sites(guide_trace)

    return guide_trace

  def _sample_from_joint(self, *args, **kwargs):
    """
    :returns: a sample from the joint distribution over unobserved and
        observed variables
    :rtype: pyro.poutine.trace_struct.Trace

    Returns a trace of the model without conditioning on any observations.

    Arguments are passed directly to the model.
    """
    unconditioned_model = pyro.poutine.uncondition(self.model)
    
    return poutine.trace(unconditioned_model).get_trace(*args, **kwargs)