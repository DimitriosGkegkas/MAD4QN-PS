import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class EmbeddingHead(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.head(x)  # Output shape: (B, 3, 32, 32)
    
class EmbeddingHeadDecoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=9):
        super().__init__()
        self.deconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.deconv(x)
    
class EmbeddingBody(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        base_model = mobilenet_v2(weights=weights)
        self.features = base_model.features
        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)  # Feature maps


class EmbeddingTail(nn.Module):
    def __init__(self, input_dim, feature_dim=100):
        super().__init__()
        self.dropout = nn.Dropout2d(p=0.3)
        self.fc = nn.Linear(input_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim, bias=False)
        self.fc.apply(weights_init_)

    def forward(self, x):
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)
    
    
class EmbeddedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim=100):
        super().__init__()
        in_channels = input_dim[0]
        self.direction_dim = input_dim[1]

        self.head = EmbeddingHead(in_channels=in_channels, out_channels=3)
        self.head_decoder = EmbeddingHeadDecoder(in_channels=3, out_channels=in_channels)
        self.body = EmbeddingBody()
        self.body.eval()  # Set to eval mode to freeze the body

        dummy_input = torch.zeros(1, 3, 32, 32)
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