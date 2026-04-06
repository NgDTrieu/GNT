import torch
import torch.nn as nn
import torch.nn.functional as F


class TransientEncoder(nn.Module):
    """
    Lightweight CNN encoder to extract target-specific transient information.
    
    Input: Target image [B, H, W, 3]
    Output: Latent code [B, latent_dim] (e.g., 128-dim)
    
    Architecture: 4 conv blocks + Global Average Pooling + FC
    """
    
    def __init__(self, latent_dim=128):
        super(TransientEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Input: [B, 3, H, W] after permutation
        # Conv block 1: downsample to H/2, W/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Conv block 2: downsample to H/4, W/4
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Conv block 3: downsample to H/8, W/8
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Conv block 4: downsample to H/16, W/16
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, rgb_full):
        """
        Args:
            rgb_full: Target image [B, H, W, 3]
        
        Returns:
            latent: Latent code [B, latent_dim]
        """
        # Convert from [B, H, W, 3] to [B, 3, H, W]
        x = rgb_full.permute(0, 3, 1, 2)  # [B, 3, H, W]
        
        # Forward through conv blocks
        x = self.conv1(x)  # [B, 32, H/2, W/2]
        x = self.conv2(x)  # [B, 64, H/4, W/4]
        x = self.conv3(x)  # [B, 128, H/8, W/8]
        x = self.conv4(x)  # [B, 256, H/16, W/16]
        
        # Global average pooling
        x = self.gap(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        
        # FC layers
        latent = self.fc(x)  # [B, latent_dim]
        
        return latent
