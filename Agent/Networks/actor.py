import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 0.5
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.01)
        nn.init.constant_(m.bias, 0)

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
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h_dim, bias=False))
            layers.append(nn.Dropout(p=dropout_p))
            in_dim = h_dim

        self.net = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(in_dim, action_dim)
        self.log_std_linear = nn.Linear(in_dim, action_dim)

        self.register_buffer("action_scale", torch.tensor(1.0))
        self.register_buffer("action_bias", torch.tensor(0.0))

        self.apply(weights_init_)
        weights_init_(self.mean_linear)
        weights_init_(self.log_std_linear)

    def forward(self, embedded: torch.Tensor, direction: torch.Tensor, messages: torch.Tensor):
        x = torch.cat([embedded, direction, messages], dim=-1)
        x = self.net(x)  # apply hidden layers
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, embedded: torch.Tensor, direction: torch.Tensor, messages: torch.Tensor):
        mean, log_std = self.forward(embedded, direction, messages)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        # print("=== Actor Output Debug ===")
        # print(f"x_t (pre-tanh sample):\n{x_t}")
        # print(f"std (exp(log_std)):\n{std}")
        # print(f"mean:\n{mean}")
        # print("==========================")

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, x_t

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
