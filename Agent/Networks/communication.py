import torch
import torch.nn as nn
import torch.nn.functional as F

class MessageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=[128, 64], message_dim=8, dropout_p=0.2):
        super(MessageEncoder, self).__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, message_dim)
        self.sigmoid = nn.Sigmoid()  # soft output in [0, 1]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        x = self.hidden(obs)
        msg = self.sigmoid(self.output_layer(x))
        return msg  # soft message in [0, 1]


class MessageDecoder(nn.Module):
    def __init__(self, message_dim=8, output_dim=128, hidden_dim=[64, 128], dropout_p=0.2):
        super(MessageDecoder, self).__init__()

        layers = []
        in_dim = message_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)  # Reconstruct original embedding/state

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, msg):
        x = self.hidden(msg)
        return self.output_layer(x)  # Reconstructed embedding
