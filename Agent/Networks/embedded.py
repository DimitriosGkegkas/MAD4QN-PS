import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T

import matplotlib.pyplot as plt




def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class EmbeddingHead(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.head(x)  # Output: (B, 3, H, W)

class EmbeddingHeadDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=12):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decode(x)  # Output: (B, 12, H, W)
    
class EmbeddingBody(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = mobilenet_v2(weights="DEFAULT")  # or weights="IMAGENET1K_V1"
        self.features = base_model.features

        # Hardcoded ImageNet normalization (safe and stable)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.normalize(x)  # expects input ∈ [0, 1]
        return self.features(x)


class EmbeddingTail(nn.Module):
    def __init__(self, input_dim, feature_dim=100):
        super().__init__()
        self.tail = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256, elementwise_affine=False),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim, elementwise_affine=False)
        )
        self.tail.apply(weights_init_)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.tail(x)
    
    
class EmbeddedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=100):
        super().__init__()
        in_channels = input_dim[0]

        self.head = EmbeddingHead(in_channels=in_channels, out_channels=3)
        self.head_decoder = EmbeddingHeadDecoder(in_channels=3, out_channels=in_channels)
        self.body = EmbeddingBody()
        self.body.eval()  # Set to eval mode to freeze the body

        dummy_input = torch.zeros(1, 3, input_dim[1], input_dim[2])
        with torch.no_grad():
            body_out = self.body(dummy_input)
        flat_dim = int(np.prod(body_out.size()))

        self.tail = EmbeddingTail(flat_dim, feature_dim)

    def forward(self, img):
        x = self.head(img)
        x = self.body(x)
        x = self.tail(x)
        return x

    def reconstruction_loss(self, input_img):
        reconstructed = self.head_decoder(self.head(input_img))
        return F.mse_loss(reconstructed, input_img)
    
    def visualize_head_output(self, input_img):
        import matplotlib.pyplot as plt

        self.eval()
        with torch.no_grad():
            head_output = self.head(input_img)

        # First sample in batch
        input_sample = input_img[0].cpu()  # (9, 32, 32)

        # Split into 3 RGB images
        img1 = input_sample[0:3]
        img2 = input_sample[3:6]
        img3 = input_sample[6:9]

        # Output from head
        output_img = head_output[0].cpu()  # (3, 32, 32)

        # Function to convert (C, H, W) → (H, W, C) and normalize
        img1 = to_pil_image(img1)
        img2 = to_pil_image(img2)
        img3 = to_pil_image(img3)
        out_img = to_pil_image(output_img)

        # Now plot safely
        fig, axs = plt.subplots(1, 4, figsize=(12, 4))
        axs[0].imshow(img1)
        axs[0].set_title("Input Image 1")
        axs[1].imshow(img2)
        axs[1].set_title("Input Image 2")
        axs[2].imshow(img3)
        axs[2].set_title("Input Image 3")
        axs[3].imshow(out_img)
        axs[3].set_title("Head Output")

        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        self.train()  # Restore training mode
