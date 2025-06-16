import torch
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import torch
import math
from torch import nn
from torch import distributions as pyd

import utils


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu



LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6

class ActorNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        direction_dim: int,
        message_dim: int,
        action_dim: int,
        hidden_dim: list = [256, 128, 64],
        dropout_p: float = 0.3
    ):
        super(ActorNetwork, self).__init__()

        in_dim = feature_dim + direction_dim + message_dim
        layers = []

        for i, h_dim in enumerate(hidden_dim):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(p=dropout_p))
            in_dim = h_dim

        self.net = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(in_dim, action_dim)
        self.log_std_linear = nn.Linear(in_dim, action_dim)

        self.apply(utils.weight_init)

    def forward(self, embedded: torch.Tensor, direction: torch.Tensor, messages: torch.Tensor):
        x = torch.cat([embedded, direction, messages], dim=-1)
        x = self.net(x)  # apply hidden layers
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        
        log_std = torch.tanh(log_std)
        log_std = LOG_SIG_MIN + 0.5 * (LOG_SIG_MAX - LOG_SIG_MIN) * (log_std + 1)
        
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist, mu
    

