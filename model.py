import torch
import torch.nn as nn

# Remember : Noisy Image -> Denoising Diffusion Distribution -> Probabilistic Model -> Predicted Noise
# But, what about timestep information?
# We need to use time embeddings (similar to transformer architecture)
# time t -> pos.embedding -> FCN -> SILU -> FCN -> embedding


# First, the method to get the time embeddings
def get_time_embedding(time_steps, embedding_dim):
    """
    this is to solve the t -> pos_embeddings part only
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

    # evaluate only half of the embeddings, as we need to include both the sin and cosine portions, which together yields embedding_dim
    half_time_embedding = time_steps[:None].repeat(1, half_embedding_dim) / common_factors
    time_embedding = torch.cat([torch.sin(half_time_embedding), torch.cos(half_embedding_dim)], dim=-1)
    return time_embedding

# The UNet needs to take in time embedding information via self attention modules
# A bit tricky to do from scratch for the first time, I'm borrowing implementation from https://github.com/explainingai-code/DDPM-Pytorch
# We basically need a Downs block, a bottleneck and an Ups block
# The Downs block needs to have a ResNet + Self Attention module, then a down-sampling layer
# The bottleneck should also be modified - 1 ResNet, then Self Attention + ResNet module (HF implementation)
# The Ups block would have the up-sampling layer, along with the same ResNet + Self Attention module (skip connections are transferred just like UNet)
# Need to use the time-embeddings also (where?!)

# Downs Block
# x -> [Norm + SILU  + Conv] -> x_1
# (x_1, cat with actual_time_embeddings) -> [Norm + SILU + Conv] -> x_2
# (x_2, cat with x) for self attention -> [Norm + SA] -> x_3
# (x_2, cat with x_3) -> Down Sample -> z

class DownsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, down_sampler, num_attenion_heads):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_sampler = down_sampler

        # Simple network for obtaining the actual time embeddings
        self.time_embedding_network = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels)
        )

        self.resnet_block_conv_1 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        )

        self.resnet_block_conv_2 = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=self.out_channels),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        )

        # The Attention Block
        self.attention_block_norm = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.attention_block_multihead = nn.MultiheadAttention(self.out_channels, num_attenion_heads, batch_first=True)

        # a simple 1x1 Conv to make residual connection from the input to the output of the last conv layer
        self.residual_connection = nn.Conv2d(self.in_channels, self.out_channels, 1)

        # down sampling layer
        self.down_sample = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=4, stride=2, padding=1) \
            if self.down_sampler else nn.Identity()

    def forward(self, x, time_embedding):
        x_copy_for_out = x

        # Forward pass thru ResNet blocks
        resnet_input = x_copy_for_out
        out = self.resnet_block_conv_1(resnet_input)
        out += self.time_embedding_network(time_embedding)[:, :, None, None]
        out = self.resnet_block_conv_2(out)
        out += self.residual_connection(resnet_input)

        # Forward pass thru Attention Block (attention for all pixels in (h, w))
        batch_size, c, h, x = out.shape # c acts as the feature dimensionality
