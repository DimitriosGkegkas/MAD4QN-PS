import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class DuelingDQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, device='cpu'):
        super(DuelingDQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.name = name
        
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
        
        # Fully Connected Layers
        self.fc4 = nn.Linear(fc_input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        # Optimizer & Loss Function
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
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
        flat = self.dropout(F.relu(self.fc4(flat)))
        
        V = self.V(flat)
        A = self.A(flat)
        
        return V, A

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
                print(self.checkpoint_file, checkpoint_file)
                checkpoint = T.load(checkpoint_file, map_location=self.device)
            else:
                raise ValueError("Checkpoint file not found.")

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.to(self.device)
