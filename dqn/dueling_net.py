import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class DuelingDQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, device = 'cpu'):
        super(DuelingDQNetwork, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc4 = nn.Linear(fc_input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear (512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = device
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        layer1 = F.relu(self.conv1(state))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))
        # layer3 shape is BS x n_filters x H x W
        layer4 = layer3.view(layer3.size()[0], -1) # first dim is batch size, then -1 means that we flatten the other dims
        layer4 = F.relu(self.fc4(layer4))
        V = self.V(layer4)
        A = self.A(layer4)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
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
            self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
        else:
            checkpoint_file = self.search_for_load_file(path)
            if checkpoint_file:
                print(self.checkpoint_file, checkpoint_file)
                self.load_state_dict(T.load(checkpoint_file, map_location=self.device))
            else:
                raise ValueError("Checkpoint file not found.")
        self.to(self.device)