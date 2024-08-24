import torch
import torch.nn as nn

# Remember : Noisy Image -> Denoising Diffusion Distribution -> Probabilistic Model -> Predicted Noise
# But, what about timestep information?
# We need to use time embeddings

class UNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels