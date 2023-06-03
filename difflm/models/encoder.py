"""The codes are modified.

- Link:
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/improved-diffusion/improved_diffusion/transformer_model2.py
"""

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, latent_size):
        super().__init__()
        self.encoder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=latent_size,
        )

    def forward(self, input_ids):
        """
        Args:
            input_ids (torch.tensor):
                shape == (batch_size, seq_len)

        Returns:
            x0 (torch.tensor):
                shape == (batch_size, seq_len, latent_size)
        """
        x0 = self.encoder(input_ids)
        return x0
