"""
The codes are modified.
Link:
    - [SinusoidalPossitionalEmbedding] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L6-L24
"""
import math

import torch
import torch.nn as nn


class SinusoidalPossitionalEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim (int): Number of embedded dimensions.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Embedded values.
                shape = (size, )
                dtype = torch.float32
        Returns:
            emb (toch.tensor): Sinusoidal embeddings.
                shape = (size, dim)
                dtype = torch.float32
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb
