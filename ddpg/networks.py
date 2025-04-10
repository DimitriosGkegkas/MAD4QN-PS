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
        self.name = name + '_critic'
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout (can comment during debugging)
        self.dropout = nn.Dropout(p=0.4)

        # Flattened size after convs
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        # State path (same as before)
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.fc1 = nn.Linear(fc_input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.bn1 = nn.LayerNorm(self.fc1_dims)
        # self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Action path — NEW
        self.action_fc1 = nn.Linear(self.n_actions, 128)
        self.action_fc2 = nn.Linear(128, self.fc2_dims)

        # Fusion MLP — NEW
        self.q_fc1 = nn.Linear(self.fc2_dims * 2, 256)  # concat(state, action)
        self.q_out = nn.Linear(256, 1)

        # Weight initialization
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 1. / np.sqrt(self.q_fc1.weight.data.size()[0])
        self.q_fc1.weight.data.uniform_(-f3, f3)
        self.q_fc1.bias.data.uniform_(-f3, f3)

        f4 = 0.003
        self.q_out.weight.data.uniform_(-f4, f4)
        self.q_out.bias.data.uniform_(-f4, f4)

        f5 = 1. / np.sqrt(self.action_fc1.weight.data.size()[0])
        self.action_fc1.weight.data.uniform_(-f5, f5)
        self.action_fc1.bias.data.uniform_(-f5, f5)

        f6 = 1. / np.sqrt(self.action_fc2.weight.data.size()[0])
        self.action_fc2.weight.data.uniform_(-f6, f6)
        self.action_fc2.bias.data.uniform_(-f6, f6)

        # Optimizer
        # self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.0)
        self.optimizer = optim.RMSprop(self.parameters(), lr=beta)

        self.device = device
        self.to(self.device)


    def forward(self, state, action):
        # 1. Conv layers for state
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # 2. Flatten and project to state embedding
        flat = x.view(x.size(0), -1)
        state_out = F.relu(self.fc1(flat))
        state_out = F.relu(self.fc2(state_out))
        
        # state_out = F.relu(self.bn1(self.fc1(flat)))
        # state_out = F.relu(self.bn2(self.fc2(state_out)))

        # 3. Encode the action separately
        action_out = F.relu(self.action_fc1(action))
        action_out = F.relu(self.action_fc2(action_out))

        # 4. Concatenate state and action features
        combined = T.cat([state_out, action_out], dim=1)
        # combined = action_out

        # 5. Final MLP to estimate Q-value
        x = F.relu(self.q_fc1(combined))
        q_value = self.q_out(x)

        return q_value



    def save_checkpoint(self):
        print('... saving checkpoint ...')
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        T.save(checkpoint, self.checkpoint_file)

    def search_for_load_file(self, path):
        """
        Searches for the best matching checkpoint file in the given directory.
        
        :param path: The directory where checkpoint files are stored.
        :return: The path to the best matching checkpoint file, or None if not found.
        """
        if not os.path.exists(path):
            print(f"Path '{path}' does not exist.")
            return None

        # Split self.name based on '_'
        name_parts = self.name.split('_')

        if len(name_parts) < 4:
            print("Error: The model name should have at least 4 parts separated by '_'.")
            return None

        # Extract first two and last two words
        first_two = '_'.join(name_parts[:2])
        last_two = '_'.join(name_parts[-2:])

        best_match = None

        # Search for matching files
        for file in os.listdir(path):
            if first_two in file and last_two in file:
                best_match = os.path.join(path, file)
                break  # If a match is found, return immediately

        return best_match

        
    def load_checkpoint(self, path=None):
        print('... loading checkpoint ...')
        if path is None:
            checkpoint = T.load(self.checkpoint_file, map_location=self.device)
        else:
            checkpoint_file = self.search_for_load_file(path)
            if checkpoint_file:
                print(self.name, checkpoint_file)
                checkpoint = T.load(checkpoint_file, map_location=self.device)
            else:
                raise ValueError("Checkpoint file not found.")
        if ('model_state_dict' in checkpoint) and ('optimizer_state_dict' in checkpoint):
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else: 
            self.load_state_dict(checkpoint)
        self.to(self.device)

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
        self.name = name + '_actor'
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)
        
        
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
        self.bn3 = nn.LayerNorm(self.n_actions)
        

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

        # self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.optimizer = optim.RMSprop(self.parameters(), lr=alpha)
        
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
        # x = self.bn1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.mu(x)
        # x = self.bn3(x)
        # x = T.tanh(x)

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        T.save(checkpoint, self.checkpoint_file)

    def search_for_load_file(self, path):
        """
        Searches for the best matching checkpoint file in the given directory.
        
        :param path: The directory where checkpoint files are stored.
        :return: The path to the best matching checkpoint file, or None if not found.
        """
        if not os.path.exists(path):
            print(f"Path '{path}' does not exist.")
            return None

        # Split self.name based on '_'
        name_parts = self.name.split('_')

        if len(name_parts) < 4:
            print("Error: The model name should have at least 4 parts separated by '_'.")
            return None

        # Extract first two and last two words
        first_two = '_'.join(name_parts[:2])
        last_two = '_'.join(name_parts[-2:])

        best_match = None

        # Search for matching files
        for file in os.listdir(path):
            if first_two in file and last_two in file:
                best_match = os.path.join(path, file)
                break  # If a match is found, return immediately

        return best_match

        
    def load_checkpoint(self, path=None):
        print('... loading checkpoint ...')
        if path is None:
            checkpoint = T.load(self.checkpoint_file, map_location=self.device)
        else:
            checkpoint_file = self.search_for_load_file(path)
            if checkpoint_file:
                print(self.name, checkpoint_file)
                checkpoint = T.load(checkpoint_file, map_location=self.device)
            else:
                raise ValueError("Checkpoint file not found.")
        if ('model_state_dict' in checkpoint) and ('optimizer_state_dict' in checkpoint):
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        self.to(self.device)
