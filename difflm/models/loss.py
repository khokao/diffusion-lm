"""Codes are modified.

- Link:
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/improved-diffusion/improved_diffusion/gaussian_diffusion.py
"""

import torch
import torch.nn as nn


class E2ELoss(nn.Module):
    LOSS_KEYS = ['mse_loss', 'decode_loss', 'xt_loss', 'total_loss']

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x0_hat, x0_noisy_logits, xT, x0_init, x0_noisy, input_ids, t):
        """End-to-end loss in Diffusion-LM.

        Args:
            x0_hat (torch.tensor): Output of a denoising network.
                shape == (batch_size, seq_len, latent_size)
            x0_noisy_logits (torch.tensor): The logits of the decoder for x0_noisy.
                shape == (batch_size, seq_len, vocab_size)
            xT (torch.tensor): xT calculated based on x0_noisy.
                shape == (batch_size, seq_len, latent_size)
            x0_init (torch.tensor): The initial x0.
                shape == (batch_size, seq_len, latent_size)
            x0_noisy (torch.tensor): The noisy x0.
                shape == (batch_size, seq_len, latent_size)
            input_ids (torch.tensor): Token IDs of the original text.
                shape == (batch_size, seq_len)
            t (torch.tensor): Timesteps.
                shape == (batch_size,)

        Returns:
            losses (Dict[torch.tensor]): Loss dict with keys defined in `LOSS_KEYS`.
        """
        simple_loss = self.mse_loss(x0_hat, x0_noisy).mean(dim=(1, 2))
        emb_loss = self.mse_loss(x0_hat, x0_init).mean(dim=(1, 2))
        mse_loss = torch.where(t == 0, emb_loss, simple_loss).mean()

        decode_loss = self.ce_loss(x0_noisy_logits.view(-1, x0_noisy_logits.shape[-1]), input_ids.view(-1))
        xt_loss = self.mse_loss(xT, torch.zeros_like(xT)).mean()

        total_loss = mse_loss + decode_loss + xt_loss

        losses = {
            'mse_loss': mse_loss,
            'decode_loss': decode_loss,
            'xt_loss': xt_loss,
            'total_loss': total_loss,
        }

        return losses
