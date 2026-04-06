import torch
import torch.nn as nn
import numpy as np

class VisibilityMaskMLP(nn.Module):
    """
    MLP to predict per-pixel visibility mask (1 = static, 0 = transient).
    
    Input: (u, v) coordinates [N_rays, 2] normalized to [-1, 1] 
           + latent code [B, latent_dim]
    Output: Visibility mask [N_rays, 1] in [0, 1]
    
    The MLP learns to predict which pixels are static (should contribute to loss)
    and which are transient occlusions (should be ignored).
    """
    
    def __init__(self, latent_dim=128, hidden_dim=128):
        super(VisibilityMaskMLP, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # First, embed (u, v) coordinates with positional encoding
        self.pos_encoding_dim = 2 + 2 * 10 * 2  # sin-cos encoding: 2 + 2*num_freqs*2
        self._build_pos_encoder()
        
        # MLP layers
        # Input: pos_encoding_dim + latent_dim
        # Hidden: hidden_dim
        # Output: 1 (visibility score)
        mlp_input_dim = self.pos_encoding_dim + latent_dim
        
        self.fc1 = nn.Linear(mlp_input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize last layer to output high values (→ Sigmoid ≈ 1)
        # This way, initially M_t ≈ 1 (treat all pixels as static)
        # and MLP can learn to reduce visibility where transients are
        nn.init.constant_(self.fc_out.bias, 3.0)  # Sigmoid(3) ≈ 0.95
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.01)  # Small weights
    
    def _build_pos_encoder(self):
        """Build positional encoding for (u, v) coordinates"""
        self.num_freqs = 10
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.linspace(0.0, 9.0, steps=self.num_freqs)
        )
    
    def _encode_position(self, uv):
        """
        Encode (u, v) using sinusoidal positional encoding.
        
        Args:
            uv: [N, 2] coordinates in [-1, 1]
        
        Returns:
            encoded: [N, pos_encoding_dim]
        """
        # Include original coordinates
        encoded = [uv]
        
        # Add sin-cos encoding
        for freq in self.freq_bands:
            encoded.append(torch.sin(uv * freq * np.pi))
            encoded.append(torch.cos(uv * freq * np.pi))
        
        return torch.cat(encoded, dim=-1)
    
    def forward(self, uv, latent_code):
        """
        Predict visibility mask for given coordinates and latent code.
        
        Args:
            uv: Pixel coordinates [N_rays, 2] normalized to [-1, 1]
            latent_code: Transient latent code [B, latent_dim]
                        (typically B=1 for single image, repeated for all N_rays)
        
        Returns:
            visibility_mask: [N_rays, 1] in [0, 1]
        """
        # Encode position
        uv_encoded = self._encode_position(uv)  # [N_rays, pos_encoding_dim]
        
        # Expand latent code to match number of rays if needed
        if latent_code.shape[0] == 1:
            latent_code = latent_code.expand(uv.shape[0], -1)  # [N_rays, latent_dim]
        
        # Concatenate encoded position and latent code
        x = torch.cat([uv_encoded, latent_code], dim=-1)  # [N_rays, pos_encoding_dim + latent_dim]
        
        # Forward through MLP
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        visibility_mask = self.sigmoid(self.fc_out(x))  # [N_rays, 1]
        
        return visibility_mask
