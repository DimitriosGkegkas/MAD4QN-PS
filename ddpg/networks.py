import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, device,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)  # Increased filters
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1)  # New Conv Layer
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1)  # New Conv Layer

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(p=0.4)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        
        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        
        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = device

        self.to(self.device)

    def forward(self, state, action):
        layer1 = self.pool(F.relu(self.conv1(state)))
        layer2 = self.pool(F.relu(self.conv2(layer1)))
        layer3 = F.relu(self.conv3(layer2))
        layer4 = F.relu(self.conv4(layer3))
        layer5 = F.relu(self.conv5(layer4))
        
        flat = layer5.view(layer5.size()[0], -1)
        state_value = self.fc1(flat)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.dropout(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        #state_value = F.relu(state_value)
        #action_value = F.relu(self.action_value(action))
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.dropout(state_action_value)
        #state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
    def calculate_conv_output_dims(self, input_dims):
        
        state = T.zeros(1, *input_dims)
        dims = self.pool(F.relu(self.conv1(state)))
        dims = self.pool(F.relu(self.conv2(dims)))
        dims = F.relu(self.conv3(dims))
        dims = F.relu(self.conv4(dims))
        dims = F.relu(self.conv5(dims))
        return int(np.prod(dims.size()))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, device,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')
        
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)  # Increased filters
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1)  # New Conv Layer
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1)  # New Conv Layer
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(p=0.4)
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device


        self.to(self.device)
        
    def calculate_conv_output_dims(self, input_dims):
        
        state = T.zeros(1, *input_dims)
        dims = self.pool(F.relu(self.conv1(state)))
        dims = self.pool(F.relu(self.conv2(dims)))
        dims = F.relu(self.conv3(dims))
        dims = F.relu(self.conv4(dims))
        dims = F.relu(self.conv5(dims))
        return int(np.prod(dims.size()))

    def forward(self, state):
        layer1 = self.pool(F.relu(self.conv1(state)))
        layer2 = self.pool(F.relu(self.conv2(layer1)))
        layer3 = F.relu(self.conv3(layer2))
        layer4 = F.relu(self.conv4(layer3))
        layer5 = F.relu(self.conv5(layer4))
        
        flat = layer5.view(layer5.size()[0], -1)
        x = self.fc1(flat)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
