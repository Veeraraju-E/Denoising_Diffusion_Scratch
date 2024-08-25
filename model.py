import torch
import torch.nn as nn

# Remember : Noisy Image -> Denoising Diffusion Distribution -> Probabilistic Model -> Predicted Noise
# But, what about timestep information?
# We need to use time embeddings (similar to transformer architecture)
# time t -> pos.embedding -> FCN -> SILU -> FCN -> embedding

# The UNet needs to take in time embedding information via self attention modules
# A bit tricky to do from scratch for the first time, I'm borrowing implementation from https://github.com/explainingai-code/DDPM-Pytorch
# We basically need a Downs block, a bottleneck and an Ups block
# The Downs block needs to have a ResNet + Self Attention module, then a down-sampling layer
# The bottleneck should also be modified - 1 ResNet, then Self Attention + ResNet module (HF implementation)
# The Ups block would have the up-sampling layer, along with the same ResNet + Self Attention module (skip connections are transferred just like UNet)
# Need to use the time-embeddings also (as part of input to ResNet?)

# First, the method to get the time embeddings
def get_time_embedding(time_steps, embedding_dim):
    """

    :param time_steps: batch of time steps of every sample [B]
    :param embedding_dim: how much do we want our embedding dimension to be
    :return: [B, embedding_dim] representation of the timestep information
    """
    # time embeddings basically contain sin(pos/10000^(2i/d_model)) and cos(pos/10000^(2i/d_model)) terms,
    # which have a common factor -> calculate it first
    half_embedding_dim = embedding_dim // 2
    common_factors = 100000**((torch.arange(
        start = 0,
        end = half_embedding_dim,
        device = time_steps.device
    )) / half_embedding_dim
    )

    # evaluate only half of the embeddings, as we need to include both the sin and cosine portions, together which yields embedding_dim
    half_time_embedding = time_steps[:None].repeat(1, half_embedding_dim) / common_factors
    time_embedding = torch.cat([torch.sin(half_time_embedding), torch.cos(half_embedding_dim)], dim=-1)
    return time_embedding

class UNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels