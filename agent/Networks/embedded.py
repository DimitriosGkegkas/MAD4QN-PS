import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T

import matplotlib.pyplot as plt


import torch
import torch.nn as nn


class EmbeddedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=100, dropout_p=0.3):
        super().__init__()
        in_channels = input_dim[0]

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=dropout_p),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=dropout_p),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # Dynamically calculate flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            flat_dim = self.network(dummy_input).shape[1]

        self.projector = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            # nn.LayerNorm(256, elementwise_affine=False),
            nn.Linear(256, feature_dim),
            # nn.LayerNorm(feature_dim, elementwise_affine=False)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.projector(x)
        return x

    
    def visualize_features(self, input_img, num_features=6):
        """
        Visualizes the input and selected CNN feature maps before flattening.
        Args:
            input_img (Tensor): (B, C, H, W)
            num_features (int): Number of feature maps to visualize (from final conv layer)
        """
        self.eval()
        with torch.no_grad():
            # First few layers (Conv + ReLU + Dropout2d)
            x = self.network[0](input_img)  # Conv1
            x = self.network[1](x)
            x = self.network[2](x)

            x = self.network[3](x)  # Conv2
            x = self.network[4](x)
            x = self.network[5](x)

            x = self.network[6](x)  # Conv3
            x = self.network[7](x)  # Final ReLU

            # x is now (B, C, H, W)
            feature_maps = x[0].cpu()  # Take first sample
            input_sample = input_img[0].cpu()

        # Input view (first 3 channels)
        input_rgb = to_pil_image(input_sample[:3])  # Assume input has at least 3 channels

        # Select N feature maps
        num_features = min(num_features, feature_maps.shape[0])
        selected_maps = feature_maps[:num_features]

        # Plot input + feature maps
        fig, axs = plt.subplots(1, num_features + 1, figsize=(3 * (num_features + 1), 3))
        axs[0].imshow(input_rgb)
        axs[0].set_title("Input RGB")
        axs[0].axis("off")

        for i in range(num_features):
            axs[i + 1].imshow(selected_maps[i], cmap='viridis')
            axs[i + 1].set_title(f"Feature {i}")
            axs[i + 1].axis("off")

        plt.tight_layout()
        plt.show()
        self.train()