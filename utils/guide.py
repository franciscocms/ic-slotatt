from typing import Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F
from einops import reduce
import math

import pyro.poutine as poutine
from pyro.poutine.runtime import apply_stack
from pyro.infer import TracePosterior
import pyro.distributions as dist

# add project path to sys to import relative modules
import sys
import os
sys.path.append(os.path.abspath(__file__+'/../../'))

from main import setup
from .distributions import CategoricalVals, TruncatedNormal
from .var import Variable

params = setup.params

import logging
logging.getLogger(__name__)

def to_int(value: Tensor):
    return int(torch.round(value))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_pretrained_wts(guide, path):
    # loading the pre-trained weights from the density map estimator
    pretrained_estimator = torch.load(path)
    guide_encoder_dict = guide.encoder_cnn.state_dict()
    for key in guide_encoder_dict:
        if key in pretrained_estimator.keys():
            guide_encoder_dict[key] = pretrained_estimator[key]
            logging.info(f"...{key} loaded")
    guide.encoder_cnn.load_state_dict(guide_encoder_dict)
    return guide

@torch.jit.script
def sinkhorn(C: Tensor, a: Tensor, b: Tensor, n_sh_iters: int = 5, temperature: float = 1, u: Union[Tensor, None] = None, v: Union[Tensor, None] = None) -> Tuple[Tensor, Tensor, Tensor]:
    p = -C / temperature
    log_a = torch.log(a)
    log_b = torch.log(b)

    if u is None:
        u = torch.zeros_like(a)
    if v is None:
        v = torch.zeros_like(b)

    for _ in range(n_sh_iters):
        u = log_a - torch.logsumexp(p + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(p + u.unsqueeze(2), dim=1)

    logT = p + u.unsqueeze(2) + v.unsqueeze(1)
    return logT.exp(), u, v


@torch.enable_grad()
def minimize_entropy_of_sinkhorn(C_0, a, b, noise=None, mesh_lr=1, n_mesh_iters=4, n_sh_iters=5, reuse_u_v=True):
    if noise is None:
        noise = torch.randn_like(C_0)

    C_t = C_0 + 1e-3 * noise # previously was 0.001
    C_t.requires_grad_(True)

    u = None
    v = None
    for i in range(n_mesh_iters):
        attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, n_sh_iters=n_sh_iters)

        if not reuse_u_v:
            u = v = None

        entropy = reduce(
            torch.special.entr(attn.clamp(min=1e-20, max=1)), "n a b -> n", "mean"
        ).sum()
        
        #logging.info(entropy)
        
        (grad,) = torch.autograd.grad(entropy, C_t, retain_graph=True)
        grad = F.normalize(grad + 1e-20, dim=[1, 2])
        C_t = C_t - mesh_lr * grad

    # attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, num_sink=num_sink_iters)

    if not reuse_u_v:
        u = v = None

    return C_t, u, v

def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

def linear_std(step):
    if step <= 1000: return 1. - step*0.00095
    else: return 0.05

def load_trained_guide(guide, GUIDE_PATH):
    out_dim = 0
    pretrained_guide = torch.load(GUIDE_PATH) if params['device'] == torch.device('cuda:0') else torch.load(GUIDE_PATH, map_location='cpu')
    guide_dict = guide.state_dict()
    for k, v in pretrained_guide.items():
        if k not in guide_dict.keys():
            prior_distribution = None
            proposal_distribution = None
            name = k.split(".")[1]
            if name  == "N":
                prior_distribution = "poisson"
                if params["N_proposal"] == "mixture":
                    proposal_distribution = "mixture"
                    MIXTURE_COMPONENTS = params["mixture_components"]
                    out_dim = 3*MIXTURE_COMPONENTS
                elif params["N_proposal"] == "normal":
                    proposal_distribution = "normal"
                    out_dim = 2
            elif name.split("_")[0] in ["shape", "color", "size"]: 
                prior_distribution = "categorical"
                proposal_distribution = "categorical"
                out_dim = 2 if name.split("_")[0] == "shape" else 3
            elif name.split("_")[0] in ["locX", "locY"]: 
                prior_distribution = "uniform"
                if params["loc_proposal"] == "normal":
                    proposal_distribution = "normal"
                    out_dim = 2
                elif params["loc_proposal"] == "mixture":
                    proposal_distribution = "mixture"
                    out_dim = 3*params["loc_proposal_k"]
                else:
                    proposal_distribution = "normal"
                    out_dim = 2
            
            elif name[:2] == "bg":# and params["infer_background"]:
                prior_distribution = "uniform"
                proposal_distribution = "normal"
                out_dim = 2

            # for now, instance will be set on 0!
            var = Variable(name=name,
                            value=None,
                            prior_distribution=prior_distribution,
                            proposal_distribution=proposal_distribution,
                            address=name.split("_")[0]
                            )
            
            #if prior_distribution is None: raise ValueError("Prior distribution cannot be None")
            #if proposal_distribution is None: raise ValueError("Proposal distribution cannot be None")
            
            if k.split("_")[0] == "prop": guide.add_proposal_net(var=var,
                                                                    out_dim=out_dim)
    guide_dict = guide.state_dict() 
    state_dict = {k: v for k, v in pretrained_guide.items() if k in guide_dict.keys()}
    guide_dict.update(state_dict)
    guide.load_state_dict(guide_dict)
    logging.info("ICSA guide successfully loaded...")
    return guide


def compute_log_w(trace: TracePosterior, model, obs):    
    
    masked_vars = []
    for name, site in trace.nodes.items():
        if site['type'] == 'sample':
            if not site['mask']: masked_vars.append(name)
    
    model_trace = poutine.trace(poutine.replay(model, trace=trace)).get_trace(observations={'image': obs})
    
    for name, site in model_trace.nodes.items():
        if site['type'] == 'sample': 
            site['mask'] = False if name in masked_vars else True
    
    return model_trace.log_prob_sum() - trace.log_prob_sum()

def duplicate_trace(trace: TracePosterior):
    new_trace = poutine.Trace(graph_type=trace.graph_type)

    # for name, vals in trace.nodes.items():
    #     logging.info(f"{name} -- {vals}")
    
    for name, vals in trace.nodes.items():
        if name == "_INPUT":
            new_trace.add_node(name,
                            name=name,
                            type=vals["type"],
                            args=vals["args"],
                            kwargs=vals["kwargs"]
                            )
        elif name == "_RETURN":
            new_trace.add_node(name,
                            name=name,
                            type=vals["type"],
                            value=vals["value"]
                            )
        else:
            msg = {
            "type": vals["type"],
            "name": vals["name"],
            "fn": vals["fn"],
            "is_observed": vals["is_observed"],
            "args": vals["args"],
            "kwargs": vals["kwargs"],
            "value": vals["value"],
            "infer": vals["infer"],
            "scale": vals["scale"],
            "mask": vals["mask"],
            "cond_indep_stack": vals["cond_indep_stack"],
            "done": vals["done"],
            "stop": vals["stop"],
            "continuation": vals["continuation"],
            }
            apply_stack(msg)
            new_trace.add_node(msg["name"], **msg.copy())        
        
    return new_trace

def random_walk_step(step: int, trace: TracePosterior, model, vars, init_step_size: float = 0.005):
    
    """
    random walk step:

    init_step_size: float = 0.1
    log_step_size = math.log(init_step_size)
    step_size = math.exp(log_step_size)
    new_params = {
        k: v + step_size * torch.randn(v.shape, dtype=v.dtype, device=v.device)
        for k, v in params.items()
    }
    """

    log_step_size = math.log(init_step_size)
    step_size = math.exp(log_step_size)

    proposed_trace = duplicate_trace(trace)

    masked_vars = [] # holds all variables to be ignored in 'log_prob_sum()' computation
    for name, site in proposed_trace.nodes.items():
        if site['type'] == 'sample': 
            #logging.info(f"{name} - {site}\n")
            if site['mask'] == False: masked_vars.append(name)
    
    current_guide_trace_log_w = proposed_trace.log_prob_sum()
    current_model_trace = poutine.trace(poutine.replay(model, trace=proposed_trace)).get_trace(show=vars)

    # mask variables that are masked in guide trace
    for name, site in current_model_trace.nodes.items():
        if site['type'] == 'sample': 
            site['mask'] = False if name in masked_vars else True
    
    # logging.info("\nAFTER DUPLICATE\n")  
    # for name, site in current_model_trace.nodes.items():
    #     if site['type'] == 'sample': logging.info(f"{name} - {site}\n")
    
    hidden_sites = ['N'] # latent variables that should not be perturbed
    
    for name, vals in proposed_trace.nodes.items():        
        if vals['type'] == 'sample' and name not in hidden_sites and vals['mask']:
            
            if torch.rand(1) > 0.5:

                # perturb the particle
                if isinstance(vals['fn'], dist.Normal) or isinstance(vals['fn'], TruncatedNormal):
                    v = vals['fn'].loc
                    old_value = vals['value']
                    # apply 'random-walk' proposal - need to be careful with the truncated normal distributions
                    v += step_size * torch.randn(v.shape, dtype=v.dtype, device=v.device)
                    vals['value'] = vals['fn'].sample()
                    if isinstance(vals['fn'], TruncatedNormal):
                        if vals['value'].item() > 1.0: vals['value'] = torch.tensor(1.0)
                        if vals['value'].item() < 0.0: vals['value'] = torch.tensor(0.0)
                    
                    if step % 50 == 0: logging.info(f"perturbing {name} from {old_value} to {vals['value']}...")

            if torch.rand(1) > 0.5:

                if isinstance(vals['fn'], CategoricalVals):
                    probs = vals['fn'].probs
                    old_value = vals['value']
                    if len(probs.shape) > 1: probs = probs[0]

                    # define random permutation order
                    order = torch.randperm(len(probs))
                    # permute categorical probs
                    new_probs = torch.gather(probs, 0, order)
                    vals['fn'].probs = new_probs
                    vals['fn'].categorical = dist.Categorical(vals['fn'].probs)
                    vals['value'] = vals['fn'].sample()

                    if step % 50 == 0: logging.info(f"perturbing {name} from {old_value} with probs {probs} to {vals['value']} with new probs {vals['fn'].probs}...")   
    
    # compute acceptance probability
    new_model_trace = poutine.trace(poutine.replay(model, trace=proposed_trace)).get_trace(show=vars)
    for name, site in new_model_trace.nodes.items():
        site['mask'] = False if name in masked_vars else True
    
    # logging.info("\nAFTER PERTURBATION\n")  
    # for name, site in new_model_trace.nodes.items():
    #     if site['type'] == 'sample': logging.info(f"{name} - {site}\n")
    

    logr = new_model_trace.log_prob_sum() + proposed_trace.log_prob_sum() - (current_model_trace.log_prob_sum() + current_guide_trace_log_w)
    accept_prob = min(1, torch.exp(logr))
    accepted = True if torch.rand(1) < accept_prob else False

    # change for flipping a coin with heads probability = accepted_prob

    if step % 50 == 0: 
        logging.info(f"{new_model_trace.log_prob_sum()} - {proposed_trace.log_prob_sum()} - {current_model_trace.log_prob_sum()} - {current_guide_trace_log_w}")
        logging.info(f"{logr} - {accept_prob}")

    if accepted: return accepted, proposed_trace, compute_log_w(proposed_trace, model)
    else: return accepted, trace, compute_log_w(trace, model)