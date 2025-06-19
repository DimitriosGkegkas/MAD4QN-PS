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
            layers.append(nn.LayerNorm(h_dim))
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



class MessageAggregator(nn.Module):
    def __init__(self, message_dim, device, max_msgs=5, aggregation_type="mean"):
        super().__init__()
        assert aggregation_type in ["mean", "sum", "max"], "Unsupported aggregation type"
        self.message_dim = message_dim
        self.max_msgs = max_msgs
        self.aggregation_type = aggregation_type
        self.device = device

    def forward(self, message_list):
        """
        message_list: List[List[Tensor]] where each inner list is [msg_dim] tensors
        Returns:
            [batch_size, message_dim]
        """
        msg_tensor, msg_mask = self._pad_and_mask(message_list)
        return self._aggregate(msg_tensor, msg_mask)

    def _pad_and_mask(self, message_list):
        batch_size = len(message_list)

        # Create padded and masked tensors
        padded = torch.zeros(batch_size, self.max_msgs, self.message_dim, device=self.device)
        mask = torch.zeros(batch_size, self.max_msgs, dtype=torch.bool, device=self.device)

        for i, msgs in enumerate(message_list):
            for j, msg in enumerate(msgs[:self.max_msgs]):
                # Convert to tensor if necessary
                if not isinstance(msg, torch.Tensor):
                    msg = torch.tensor(msg, dtype=torch.float32, device=self.device)
                padded[i, j] = msg
                mask[i, j] = True

        return padded, mask

    def _aggregate(self, message_tensor, mask):
        mask = mask.unsqueeze(-1).float()  # [B, M, 1]
        message_tensor = message_tensor * mask

        if self.aggregation_type == "mean":
            summed = message_tensor.sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            return summed / count

        elif self.aggregation_type == "sum":
            return message_tensor.sum(dim=1)

        elif self.aggregation_type == "max":
            message_tensor[~mask.bool()] = float('-inf')
            return torch.max(message_tensor, dim=1).values
