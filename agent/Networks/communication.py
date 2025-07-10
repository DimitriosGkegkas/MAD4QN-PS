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
            # layers.append(nn.LayerNorm(h_dim))
            # layers.append(nn.Dropout(dropout_p))
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


class MessageAggregator(nn.Module):
    def __init__(self, message_dim, device, max_msgs=5):
        super().__init__()
        self.message_dim = message_dim
        self.max_msgs = max_msgs
        self.device = device

    def forward(self, message_list):
        """
        message_list: List[List[Tensor]] where each inner list contains [msg_dim] tensors
        Returns:
            [batch_size, max_msgs * message_dim]
        """
        padded = self._pad(message_list)
        # Flatten the message dimension: [B, max_msgs, msg_dim] -> [B, max_msgs * msg_dim]
        return padded.view(padded.size(0), -1)

    def _pad(self, message_list):
        batch_size = len(message_list)
        padded = torch.zeros(batch_size, self.max_msgs, self.message_dim, device=self.device)

        for i, msgs in enumerate(message_list):
            for j, msg in enumerate(msgs[:self.max_msgs]):
                if not isinstance(msg, torch.Tensor):
                    msg = torch.tensor(msg, dtype=torch.float32, device=self.device)
                padded[i, j] = msg

        return padded

    @property
    def output_size(self):
        return self.max_msgs * self.message_dim