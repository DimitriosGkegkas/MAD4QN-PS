import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, input_dims):
        super(ValueNetwork, self).__init__()

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_dims, 128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128, 1)

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1)

        # Dropout for convolutional layers (optional)
        self.dropout_conv = nn.Dropout2d(p=0.3)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        self.apply(weights_init_)

    def forward(self, state):
        # Convolutional layers with optional dropout
        layer1 = self.pool(F.relu(self.conv1(state)))
        layer2 = self.pool(F.relu(self.conv2(layer1)))
        layer3 = F.relu(self.conv3(layer2))
        layer3 = self.dropout_conv(layer3)
        layer4 = F.relu(self.conv4(layer3))
        layer4 = self.dropout_conv(layer4)
        layer5 = F.relu(self.conv5(layer4))
        
        flat = layer5.view(layer5.size()[0], -1)
        x = F.relu(self.fc1(flat))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.pool(F.relu(self.conv1(state)))
        dims = self.pool(F.relu(self.conv2(dims)))
        dims = F.relu(self.conv3(dims))
        dims = F.relu(self.conv4(dims))
        dims = F.relu(self.conv5(dims))
        return int(np.prod(dims.size()))

class QNetwork(nn.Module):
    def __init__(self, input_dims, num_actions, dropout_prob=0.3):
        super(QNetwork, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout after conv layers (optional)
        self.dropout_conv = nn.Dropout2d(p=dropout_prob)

        num_inputs = self.calculate_conv_output_dims(input_dims)
        self.input = nn.Linear(num_inputs, 12)

        # Q1 architecture
        self.linear1 = nn.Linear(12 + num_actions, 128)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.linear2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.linear3 = nn.Linear(128, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(12 + num_actions, 128)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.linear5 = nn.Linear(128, 128)
        self.dropout4 = nn.Dropout(p=dropout_prob)
        self.linear6 = nn.Linear(128, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        # Conv layers with optional dropout
        layer1 = self.pool(F.relu(self.conv1(state)))
        layer2 = self.pool(F.relu(self.conv2(layer1)))
        layer3 = F.relu(self.conv3(layer2))
        layer3 = self.dropout_conv(layer3)
        layer4 = F.relu(self.conv4(layer3))
        layer4 = self.dropout_conv(layer4)

        flat = layer4.view(layer4.size()[0], -1)
        input = self.input(flat)

        xu = torch.cat([input, action], 1)

        # Q1 stream with dropout
        x1 = F.relu(self.linear1(xu))
        x1 = self.dropout1(x1)
        x1 = F.relu(self.linear2(x1))
        x1 = self.dropout2(x1)
        x1 = self.linear3(x1)

        # Q2 stream with dropout
        x2 = F.relu(self.linear4(xu))
        x2 = self.dropout3(x2)
        x2 = F.relu(self.linear5(x2))
        x2 = self.dropout4(x2)
        x2 = self.linear6(x2)

        return x1, x2

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.pool(F.relu(self.conv1(state)))
        dims = self.pool(F.relu(self.conv2(dims)))
        dims = F.relu(self.conv3(dims))
        dims = F.relu(self.conv4(dims))
        return int(np.prod(dims.size()))


class GaussianPolicy(nn.Module):
    def __init__(self, input_dims, num_actions, dropout_prob=0.3):
        super(GaussianPolicy, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout for conv layers (optional)
        self.dropout_conv = nn.Dropout2d(p=dropout_prob)

        num_inputs = self.calculate_conv_output_dims(input_dims)

        # Fully connected layers
        self.linear1 = nn.Linear(num_inputs, 128)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.linear2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        self.mean_linear = nn.Linear(128, num_actions)
        self.log_std_linear = nn.Linear(128, num_actions)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

        self.apply(weights_init_)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.pool(F.relu(self.conv1(state)))
        dims = self.pool(F.relu(self.conv2(dims)))
        dims = F.relu(self.conv3(dims))
        dims = F.relu(self.conv4(dims))
        return int(np.prod(dims.size()))

    def forward(self, state):
        layer1 = self.pool(F.relu(self.conv1(state)))
        layer2 = self.pool(F.relu(self.conv2(layer1)))
        layer3 = F.relu(self.conv3(layer2))
        layer3 = self.dropout_conv(layer3)
        layer4 = F.relu(self.conv4(layer3))
        layer4 = self.dropout_conv(layer4)

        flat = layer4.view(layer4.size()[0], -1)

        x = F.relu(self.linear1(flat))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
