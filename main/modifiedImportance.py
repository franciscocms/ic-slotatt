import math
import warnings

import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.ops.stats import fit_generalized_pareto
from pyro.distributions.util import scale_and_mask
from pyro.ops.packed import pack

from .mod_abstract_infer import TracePosterior
from pyro.infer.enum import get_importance_trace
from pyro.poutine.runtime import apply_stack
from .models import occlusion

import inspect
from sys import stdout

from .setup import params

import logging
logger = logging.getLogger("eval")


class Importance(TracePosterior):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    :param num_samples: number of samples to draw from the guide (default 10)

    This method performs posterior inference by importance sampling
    using the guide as the proposal distribution.
    If no guide is provided, it defaults to proposing from the model's prior.
    """

    def __init__(self, model, guide=None, num_samples=None):
        """
        Constructor. default to num_samples = 10, guide = model
        """
        super().__init__()
        if num_samples is None:
            num_samples = 10
            warnings.warn(
                "num_samples not provided, defaulting to {}".format(num_samples)
            )
        if guide is None:
            # propose from the prior by making a guide from the model by hiding observes
            guide = poutine.block(model, hide_types=["observe"])
        self.num_samples = num_samples
        self.model = model
        self.guide = guide

    
    def _reorder_trace(self, model_trace):

        euc_dist = {}
    
        lx, ly = 0, 0
        for name, vals in model_trace.nodes.items():
            if name.split("_")[0] == "locX": lx = vals["value"]
            elif name.split("_")[0] == "locY": ly = vals["value"]
            if lx != 0 and ly != 0: 
                euc_dist[name.split('_')[1]] = math.sqrt(lx**2 + ly**2)
                lx, ly = 0, 0

        sorted_euc_dist = dict(sorted(euc_dist.items(), key=lambda x:x[1]))            
        new_model_trace = poutine.Trace(graph_type=model_trace.graph_type)

        for site, vals in model_trace.nodes.items():
            if site == "_INPUT":
                new_model_trace.add_node(site,
                                name=site,
                                type=vals["type"],
                                args=vals["args"],
                                kwargs=vals["kwargs"]
                                )
            elif site in ["N", "bgR", "bgG", "bgB"]:
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
                new_model_trace.add_node(msg["name"], **msg.copy()) 
        
        curr_idx = 0
        for k, _ in sorted_euc_dist.items():

            """
            the sorted dict keys denote the right order for which objects
            should be organized in the execution trace
            """
            
            for site, vals in model_trace.nodes.items():
                try:
                    if site.split("_")[1] == k:
                        
                        # we order the execution trace by using these k indices
                        # but the address should be modified to be 0, 1, ..., N-1
                        
                        msg = {
                        "type": vals["type"],
                        "name": vals["name"].split("_")[0] + f"_{curr_idx}",
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
                        new_model_trace.add_node(msg["name"], **msg.copy()) 
                except: pass 
            curr_idx += 1
        
        for site, vals in model_trace.nodes.items():
            if site == "image":
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
                new_model_trace.add_node(msg["name"], **msg.copy())
            
            elif site == "_RETURN":
                new_model_trace.add_node(site,
                                name=site,
                                type=vals["type"],
                                value=vals["value"]
                                )
        return new_model_trace
    
    def _old_traces(self, *args, **kwargs):
        """
        Generator of weighted samples from the proposal distribution.
        """
        for i in range(self.num_samples):
            
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)  

            #guide_trace = self._reorder_trace(guide_trace)

            """
            check if 'guide_trace' is a valid trace:
            - condition: cannot exhibit overlapped object locations
            - setting this condition to 'True' when evaluating the fully trained model,
            not when capturing the model with some amount of training steps
            """

            # checking_overlaps: True for 'counting' tasks
            #                    False for 'set prediction' tasks

            checking_overlaps = False
            if checking_overlaps:
            
                flag = True
                all_locX, all_locY, all_size = [], [], []
                
                # aggregate 'loc' and 'size' variables for all objects
                for name, site in guide_trace.nodes.items():
                    if site['type'] == 'sample':      
                        if name.split('_')[0] == 'locX': all_locX.append(site['value'])
                        elif name.split('_')[0] == 'locY': all_locY.append(site['value'])
                        elif name.split('_')[0] == 'size': all_size.append(site['value'])

                for n in range(1, len(all_locX)):
                    occlusion_set = [occlusion((all_locX[n], all_locY[n], all_size[n]), (all_locX[j], all_locY[j], all_size[j])) for j in range(n)] 
                    #print(occlusion_set)

                    if any(occlusion_set): flag = False

            else: flag = True

            if flag:
                model_trace = poutine.trace(poutine.replay(self.model, trace=guide_trace)).get_trace(*args, **kwargs)         
                
                # logger.info('\nMODEL TRACE\n')
                # for name, site in model_trace.nodes.items():
                #     if site['type'] == 'sample':
                #         logger.info(f"\n{name} - value: {site['value']}") 
                #         if isinstance(site['fn'], dist.Bernoulli) or isinstance(site['fn'], dist.Categorical): logger.info(f"posterior: {site['fn'].probs} - log_prob: {site['fn'].log_prob(site['value'])}")
                #         if isinstance(site['fn'], dist.Normal): logger.info(f"posterior mean: {site['fn'].loc} and std: {site['fn'].scale} - log_prob: {site['fn'].log_prob(site['value'])}")
                #         if isinstance(site['fn'], dist.Delta): logger.info(f"posterior: {site['fn'].loc} - log_prob: {site['fn'].log_prob(site['value'])}")
                
                # logger.info(model_trace.log_prob_sum())
                # logger.info(guide_trace.log_prob_sum())

                only_img_llh = True
                
                log_p_sum = torch.tensor(0.)
                for name, site in model_trace.nodes.items():
                    log_p = 0.
                    
                    if not only_img_llh:
                        if site['type'] == 'sample' and name != 'size':
                            log_p = site['fn'].log_prob(site['value'])
                            if name == 'image': 
                                img_dim = site['fn'].mean.shape[-1]
                                log_p = log_p / (img_dim**2)
                            
                            #logger.info(f"{name} - {log_p}")
                            log_p = scale_and_mask(log_p, site["scale"], site["mask"]).sum()
                            #logger.info(f"{name} - {log_p}")
                    
                    else:
                        if site['type'] == 'sample' and name == 'image':
                            log_p = site['fn'].log_prob(site['value']).item()
                            img_dim = site['fn'].mean.shape[-1]
                            log_p = log_p / (img_dim**2)
                    
                    log_p_sum += log_p


                log_weight = log_p_sum #- guide_trace.log_prob_sum()

                # logger.info(f"\ninspecting the guide log_prob_sum computation...\n")
                # for name, site in guide_trace.nodes.items():
                #     if site['type'] == 'sample':
                #         logger.info(f"{name} - {site['value']} - {site['fn'].log_prob(site['value'])} - {site['fn'].log_prob(site['value']).sum()}")

                # logger.info(f"model log_prob_sum: {log_p_sum}")
                # logger.info(f"guide log_prob_sum: {guide_trace.log_prob_sum()}")

                #yield (model_trace, log_weight)
                yield (model_trace, guide_trace, log_weight)
    
    
    def _traces(self, *args, **kwargs):
        """
        Generator of weighted samples from the proposal distribution.
        """
        #for i in range(self.num_samples):
        with pyro.plate("inference", self.num_samples):
            
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)  
            model_trace = poutine.trace(poutine.replay(self.model, trace=guide_trace)).get_trace(*args, **kwargs)         

            only_img_llh = True

            log_p_sum = torch.tensor(0.)
            for name, site in model_trace.nodes.items():
                log_p = 0.
                
                if not only_img_llh:
                    if site['type'] == 'sample' and name != 'size':
                        log_p = site['fn'].log_prob(site['value'])
                        if name == 'image': 
                            img_dim = site['fn'].mean.shape[-1]
                            log_p = log_p / (img_dim**2)
                        
                        #logger.info(f"{name} - {log_p}")
                        log_p = scale_and_mask(log_p, site["scale"], site["mask"]).sum()
                        #logger.info(f"{name} - {log_p}")
                
                else:
                    if site['type'] == 'sample' and name == 'image':
                        log_p = site['fn'].log_prob(site['value']).item()
                        img_dim = site['fn'].mean.shape[-1]
                        log_p = log_p / (img_dim**2)
                
                log_p_sum += log_p


            log_weight = log_p_sum #- guide_trace.log_prob_sum()

            yield (model_trace, guide_trace, log_weight)

    def get_log_normalizer(self):
        """
        Estimator of the normalizing constant of the target distribution.
        (mean of the unnormalized weights)
        """
        # ensure list is not empty
        if self.log_weights:
            log_w = torch.tensor(self.log_weights)
            log_num_samples = torch.log(torch.tensor(self.num_samples * 1.0))
            return torch.logsumexp(log_w - log_num_samples, 0)
        else:
            warnings.warn(
                "The log_weights list is empty, can not compute normalizing constant estimate."
            )


    def get_normalized_weights(self, log_scale=False):
        """
        Compute the normalized importance weights.
        """
        if self.log_weights:
            log_w = torch.tensor(self.log_weights)
            log_w_norm = log_w - torch.logsumexp(log_w, 0)
            return log_w_norm if log_scale else torch.exp(log_w_norm)
        else:
            warnings.warn(
                "The log_weights list is empty. There is nothing to normalize."
            )


    def get_ESS(self):
        """
        Compute (Importance Sampling) Effective Sample Size (ESS).
        """
        if self.log_weights:
            log_w_norm = self.get_normalized_weights(log_scale=True)
            ess = torch.exp(-torch.logsumexp(2 * log_w_norm, 0))
        else:
            warnings.warn(
                "The log_weights list is empty, effective sample size is zero."
            )
            ess = 0
        return ess



def vectorized_importance_weights(model, guide, *args, **kwargs):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    :param num_samples: number of samples to draw from the guide (default 1)
    :param int max_plate_nesting: Bound on max number of nested :func:`pyro.plate` contexts.
    :param bool normalized: set to True to return self-normalized importance weights
    :returns: returns a ``(num_samples,)``-shaped tensor of importance weights
        and the model and guide traces that produced them

    Vectorized computation of importance weights for models with static structure::

        log_weights, model_trace, guide_trace = \\
            vectorized_importance_weights(model, guide, *args,
                                          num_samples=1000,
                                          max_plate_nesting=4,
                                          normalized=False)
    """
    num_samples = kwargs.pop("num_samples", 1)
    max_plate_nesting = kwargs.pop("max_plate_nesting", None)
    normalized = kwargs.pop("normalized", False)

    if max_plate_nesting is None:
        raise ValueError("must provide max_plate_nesting")
    max_plate_nesting += 1

    def vectorize(fn):
        def _fn(*args, **kwargs):
            with pyro.plate(
                "num_particles_vectorized", num_samples, dim=-max_plate_nesting
            ):
                return fn(*args, **kwargs)

        return _fn

    model_trace, guide_trace = get_importance_trace(
        "flat", max_plate_nesting, vectorize(model), vectorize(guide), args, kwargs
    )

    for name, site in guide_trace.nodes.items():
        if site["type"] == "sample":
            
            logger.info(f"\n{name} - {site['value']}")
            
            # assert site["infer"] is not None

            # logger.info(site['infer'])

            # dim_to_symbol = site["infer"]["_dim_to_symbol"]
            # packed = site.setdefault("packed", {})
            # packed["mask"] = pack(site["mask"], dim_to_symbol)

            # if "score_parts" in site:
            #     log_prob, score_function, entropy_term = site["score_parts"]
            #     log_prob = pack(log_prob, dim_to_symbol)


    guide_trace.pack_tensors()
    model_trace.pack_tensors(guide_trace.plate_to_symbol)

    if num_samples == 1:
        log_weights = model_trace.log_prob_sum() - guide_trace.log_prob_sum()
    else:
        wd = guide_trace.plate_to_symbol["num_particles_vectorized"]
        log_weights = 0.0
        for site in model_trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_weights += torch.einsum(
                site["packed"]["log_prob"]._pyro_dims + "->" + wd,
                [site["packed"]["log_prob"]],
            )

        for site in guide_trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_weights -= torch.einsum(
                site["packed"]["log_prob"]._pyro_dims + "->" + wd,
                [site["packed"]["log_prob"]],
            )

    if normalized:
        log_weights = log_weights - torch.logsumexp(log_weights)
    return log_weights, model_trace, guide_trace



@torch.no_grad()
def psis_diagnostic(model, guide, *args, **kwargs):
    """
    Computes the Pareto tail index k for a model/guide pair using the technique
    described in [1], which builds on previous work in [2]. If :math:`0 < k < 0.5`
    the guide is a good approximation to the model posterior, in the sense
    described in [1]. If :math:`0.5 \\le k \\le 0.7`, the guide provides a suboptimal
    approximation to the posterior, but may still be useful in practice. If
    :math:`k > 0.7` the guide program provides a poor approximation to the full
    posterior, and caution should be used when using the guide. Note, however,
    that a guide may be a poor fit to the full posterior while still yielding
    reasonable model predictions. If :math:`k < 0.0` the importance weights
    corresponding to the model and guide appear to be bounded from above; this
    would be a bizarre outcome for a guide trained via ELBO maximization. Please
    see [1] for a more complete discussion of how the tail index k should be
    interpreted.

    Please be advised that a large number of samples may be required for an
    accurate estimate of k.

    Note that we assume that the model and guide are both vectorized and have
    static structure. As is canonical in Pyro, the args and kwargs are passed
    to the model and guide.

    References
    [1] 'Yes, but Did It Work?: Evaluating Variational Inference.'
    Yuling Yao, Aki Vehtari, Daniel Simpson, Andrew Gelman
    [2] 'Pareto Smoothed Importance Sampling.'
    Aki Vehtari, Andrew Gelman, Jonah Gabry

    :param callable model: the model program.
    :param callable guide: the guide program.
    :param int num_particles: the total number of times we run the model and guide in
        order to compute the diagnostic. defaults to 1000.
    :param max_simultaneous_particles: the maximum number of simultaneous samples drawn
        from the model and guide. defaults to `num_particles`. `num_particles` must be
        divisible by `max_simultaneous_particles`. compute the diagnostic. defaults to 1000.
    :param int max_plate_nesting: optional bound on max number of nested :func:`pyro.plate`
        contexts in the model/guide. defaults to 7.
    :returns float: the PSIS diagnostic k
    """

    num_particles = kwargs.pop("num_particles", 1000)
    max_simultaneous_particles = kwargs.pop("max_simultaneous_particles", num_particles)
    max_plate_nesting = kwargs.pop("max_plate_nesting", 7)

    if num_particles % max_simultaneous_particles != 0:
        raise ValueError(
            "num_particles must be divisible by max_simultaneous_particles."
        )

    N = num_particles // max_simultaneous_particles
    log_weights = [
        vectorized_importance_weights(
            model,
            guide,
            num_samples=max_simultaneous_particles,
            max_plate_nesting=max_plate_nesting,
            *args,
            **kwargs,
        )[0]
        for _ in range(N)
    ]
    log_weights = torch.cat(log_weights)
    log_weights -= log_weights.max()
    log_weights = torch.sort(log_weights, descending=False)[0]

    cutoff_index = (
        -int(math.ceil(min(0.2 * num_particles, 3.0 * math.sqrt(num_particles)))) - 1
    )
    lw_cutoff = max(math.log(1.0e-15), log_weights[cutoff_index])
    lw_tail = log_weights[log_weights > lw_cutoff]

    if len(lw_tail) < 10:
        warnings.warn(
            "Not enough tail samples to compute PSIS diagnostic; increase num_particles."
        )
        k = float("inf")
    else:
        k, _ = fit_generalized_pareto(lw_tail.exp() - math.exp(lw_cutoff))

    return k
