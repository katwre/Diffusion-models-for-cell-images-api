import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    IT describes the position of an index in an ordered list.

    Args:
        timesteps: Tensor of shape (B,) containing integer timesteps.
        dim: Dimension of the embedding.

    Returns:
        Tensor of shape (B, dim) with sinusoidal embeddings.
    """
    # Ensure timesteps are floats
    timesteps = timesteps.float()

    device = timesteps.device
    half_dim = dim // 2

    # Compute the frequencies
    exponent = -math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, device=device) * exponent
    )

    # Outer product: (B, half_dim)
    args = timesteps[:, None] * frequencies[None, :]

    # Sinusoidal embedding
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # Pad if dim is odd
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))

    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch) # its' kind of BatchNorm
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x))) # Silu (that are more common in diffusion models) instead of Relu
        # Add time channel
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down
        self.rb1 = ResBlock(base_ch, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)       # /2

        self.rb2 = ResBlock(base_ch, base_ch*2, time_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*2, 4, 2, 1)   # /4

        self.rb3 = ResBlock(base_ch*2, base_ch*4, time_dim)
        self.down3 = nn.Conv2d(base_ch*4, base_ch*4, 4, 2, 1)   # /8

        # Middle
        self.mid1 = ResBlock(base_ch*4, base_ch*4, time_dim)
        self.mid2 = ResBlock(base_ch*4, base_ch*4, time_dim)

        # Up
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*4, 4, 2, 1)  # x2
        self.urb3 = ResBlock(base_ch*8, base_ch*2, time_dim)

        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch*2, 4, 2, 1)  # x4
        self.urb2 = ResBlock(base_ch*4, base_ch, time_dim)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1)      # x8
        self.urb1 = ResBlock(base_ch*2, base_ch, time_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 3, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.conv_in(x)
        x1 = self.rb1(x0, t_emb)
        d1 = self.down1(x1)

        x2 = self.rb2(d1, t_emb)
        d2 = self.down2(x2)

        x3 = self.rb3(d2, t_emb)
        d3 = self.down3(x3)

        m = self.mid1(d3, t_emb)
        m = self.mid2(m, t_emb)

        u3 = self.up3(m)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.urb3(u3, t_emb)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.urb2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.urb1(u1, t_emb)

        return self.conv_out(u1)  # predicts epsilon