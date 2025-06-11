import torch
import torch.nn as nn

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.01)
        nn.init.constant_(m.bias, 0)

class CriticNetwork(nn.Module):
    def __init__(self, feature_dim, direction_dim, message_dim, action_dim=1, hidden_dim=[256, 256], dropout_p=0.2):
        super(CriticNetwork, self).__init__()

        # Input dimension after concatenating state and action
        critic_input_dim = feature_dim + direction_dim + message_dim + action_dim

        # Q1 stream
        self.q1 = self._build_stream(critic_input_dim, 1, hidden_dim, dropout_p)

        # Q2 stream
        self.q2 = self._build_stream(critic_input_dim, 1, hidden_dim, dropout_p)

        self.apply(weights_init_)

    def _build_stream(self, input_dim, output_dim, hidden_dim, dropout_p):
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, embedded: torch.Tensor, direction: torch.Tensor, messages: torch.Tensor, action: torch.Tensor):
        x = torch.cat([embedded, direction, messages, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
