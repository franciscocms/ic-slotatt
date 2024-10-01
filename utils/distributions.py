import numpy as np
import torch
import torch.nn as nn
import pyro.distributions as dist
from pyro.distributions import Categorical
import torch.distributions.constraints as constraints
from torch import Tensor

from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

import math
from numbers import Number, Real

# add project path to sys to import relative modules
import sys
import os
sys.path.append(os.path.abspath(__file__+'/../../'))

import logging
from main import setup

#logging.basicConfig(filename=setup.params["logfile_name"], level=logging.INFO)
logger = logging.getLogger("train")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)

def to_tensor(value, dtype=torch.float):
    if not torch.is_tensor(value):
        if type(value) == np.int64:
            value = torch.tensor(float(value))
        elif type(value) == np.float32:
            value = torch.tensor(float(value))
        else:
            value = torch.tensor(value)
    return value.to(device=device, dtype=dtype)

def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


class CategoricalVals(dist.TorchDistribution):
    arg_constraints = {'probs': constraints.simplex}
    def __init__(self, vals, probs, validate_args=None):
        
        self.vals = vals
        self.probs = probs
        self.categorical = dist.Categorical(self.probs)
        super(CategoricalVals, self).__init__(self.categorical.batch_shape,
                                              self.categorical.event_shape)
    def sample(self, sample_shape=torch.Size()):
        return self.vals[self.categorical.sample(sample_shape)]
    
    def log_prob(self, value):
        #print(self.vals, value)
        #logger.info(value)
        if type(self.vals)==list: idx = self.vals.index(value)#.nonzero()
        else: idx = (self.vals == value).nonzero()
        idx = torch.tensor(idx)
        #idx = idx.to(device)

        return self.categorical.log_prob(idx)


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class TruncatedStandardNormal(dist.TorchDistribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)        
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        value.to(device)
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        #logger.info(f"{value.is_cuda}, {self._log_scale.is_cuda}")
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


from torch.nn.functional import binary_cross_entropy_with_logits
from pyro.distributions.torch_distribution import TorchDistributionMixin

class MyBernoulli(dist.Bernoulli, TorchDistributionMixin):    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        logits = logits.to(device)
        #return -binary_cross_entropy_with_logits(logits, value, reduction='mean')
        return -binary_cross_entropy_with_logits(logits, value, reduction=setup.params["bernoulli_inf_reduction"])

        #llh = nn.MSELoss()
        #return -llh(logits, value)
 
class MyNormal(nn.Module):
    def __init__(self, loc, scale):
        super().__init__()
        
        loc, scale = loc.to(device), scale.to(device)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def get_dist(self):
        return dist.Normal(self.loc, self.scale)
    
class MyPoisson(dist.Poisson, TorchDistributionMixin):
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            s = torch.poisson(self.rate.expand(shape))
            
            logger.info(s)

            if isinstance(s, Tensor): 
                s = s.tolist()
                while any([s_ == 0 for s_ in s]): s = torch.poisson(self.rate.expand(shape))
            else:
                while s == 0: s = torch.poisson(self.rate.expand(shape))
            return s

class Mixture(dist.TorchDistribution):

    arg_constraints = {}
    def __init__(self, distributions, probs=None):
        self._distributions = distributions
        self.length = len(distributions)
        if probs is None:
            self._probs = to_tensor(torch.zeros(self.length)).fill_(1./self.length)
        else:
            self._probs = to_tensor(probs)
            self._probs = self._probs / self._probs.sum(-1, keepdim=True)
        self._log_probs = torch.log(clamp_probs(self._probs))

        event_shape = torch.Size()
        if self._probs.dim() == 1:
            batch_shape = torch.Size()
            self._batch_length = 0
        elif self._probs.dim() == 2:
            batch_shape = torch.Size([self._probs.size(0)])
            self._batch_length = self._probs.size(0)
        else:
            raise ValueError('Expecting a 1d or 2d (batched) mixture probabilities.')
        self._mixing_dist = dist.Categorical(self._probs)
        self._mean = None
        self._variance = None
        super().__init__()

    def __repr__(self):
        return 'Mixture(distributions:({}), probs:{})'.format(', '.join([repr(d) for d in self._distributions]), self._probs)

    def __len__(self):
        return self.length

    def log_prob(self, value, sum=False):
        if self._batch_length == 0:
            value = to_tensor(value).squeeze()
            lp = torch.logsumexp(self._log_probs + to_tensor([d.log_prob(value) for d in self._distributions]), dim=0)
        else:
            value = to_tensor(value).view(self._batch_length)
            lp = torch.logsumexp(self._log_probs + torch.stack([d.log_prob(value).squeeze(-1) for d in self._distributions]).view(-1, self._batch_length).t(), dim=1)
        return torch.sum(lp) if sum else lp

    def sample(self):
        if self._batch_length == 0:
            i = int(self._mixing_dist.sample())
            return self._distributions[i].sample()
        else:
            indices = self._mixing_dist.sample()
            dist_samples = []
            for d in self._distributions:
                sample = d.sample()
                if sample.dim() == 0:
                    sample = sample.unsqueeze(-1)
                dist_samples.append(sample)
            ret = []
            for b in range(self._batch_length):
                i = int(indices[b])
                ret.append(dist_samples[i][b])
            return to_tensor(ret)

    @property
    def mean(self):
        if self._mean is None:
            means = torch.stack([d.mean for d in self._distributions])
            if self._batch_length == 0:
                self._mean = torch.dot(self._probs, means)
            else:
                self._mean = torch.diag(torch.mm(self._probs, means))
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            variances = torch.stack([(d.mean - self.mean).pow(2) + d.variance for d in self._distributions])
            if self._batch_length == 0:
                self._variance = torch.dot(self._probs, variances)
            else:
                self._variance = torch.diag(torch.mm(self._probs, variances))
        return self._variance
    
class Empirical(dist.TorchDistribution):
  arg_constraints = {}
  def __init__(self, samples, log_weights):
    self._samples = samples
    self._log_weights = log_weights
    
    sample_shape, weight_shape = samples.size(), log_weights.size()
    #print(sample_shape, weight_shape)
    assert sample_shape >= weight_shape
    
    # the shape of the points are given by the remainder of sample_shape
    event_shape = sample_shape[len(weight_shape):]
    
    # we will represent the measure by a categorical distribution
    self._categorical = Categorical(logits=self._log_weights)
    super().__init__(batch_shape=weight_shape[:-1],
                      event_shape=event_shape)
      
  def sample(self, sample_shape=torch.Size()):
    # sample idx from the categorical
    idx = self._categorical.sample(sample_shape)
    samples = self._samples.gather(0, idx)
    return samples.reshape(sample_shape + samples.shape[1:])
  
  def log_prob(self, value):
    # get the sample that matches value
    sel_mask = self._samples.eq(value)
    # get weights that correspond to sample using mask
    wgts = (self._categorical.probs * sel_mask).sum(dim=-1).log()
    return wgts