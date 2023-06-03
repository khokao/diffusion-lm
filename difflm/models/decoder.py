"""The codes are modified.

- Link:
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/improved-diffusion/improved_diffusion/transformer_model2.py
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/improved-diffusion/scripts/text_sample.py
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_size, encoder):
        """
        Args:
            vocab_size (int): The size of vocabulary.
            latent_size (int): The size of latent space.
            encoder (nn.Module): The text encoder.
        """
        super().__init__()
        self.decoder = nn.Linear(in_features=latent_size, out_features=vocab_size)
        with torch.no_grad():
            self.decoder.weight = encoder.encoder.weight

    def forward(self, x0, return_logits=False):
        """
        Args:
            x0 (torch.tensor):
                shape == (batch_size, seq_len, latnet_size)
            return_logits (bool): Whether to return logits.

        Returns:
            out_ids (torch.tensor):
                shape == (batch_size, seq_len)
        """
        logits = self.decoder(x0)
        out_ids = torch.topk(logits, k=1, dim=-1).indices.squeeze()  # shape == (batch_size, seq_len)

        if return_logits:
            return out_ids, logits
        else:
            return out_ids
