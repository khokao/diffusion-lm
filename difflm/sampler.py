"""The codes are modified.

- Link:
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/improved-diffusion/improved_diffusion/gaussian_diffusion.py
"""
import numpy as np
import torch
from tqdm import tqdm

from .utils import get_betas, unwrap_model_from_ddp, rescale_timesteps


class Sampler:
    def __init__(self, model, cfg, accelerator):
        """
        Args:
            model: Diffusion Autoencoder model.
            cfg (dict): A dict of config.
        """
        self.model = model
        self.cfg = cfg
        self.accelerator = accelerator

        self.seq_len = self.cfg['model']['network']['transformer']['seq_len']
        self.latent_size = self.cfg['model']['network']['transformer']['in_channels']

        self._init_diffusion_params(self.cfg['model']['beta'], self.cfg['model']['timesteps']['num'])

    def _init_diffusion_params(self, beta_cfg, num_timesteps):
        self.num_timesteps = num_timesteps
        self.betas = get_betas(beta_cfg=beta_cfg, num_timesteps=num_timesteps).to(self.accelerator.device)
        self.alphas = 1 - self.betas.to(self.accelerator.device)
        self.alphas_cumprod = self.alphas.cumprod(dim=0).to(self.accelerator.device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.accelerator.device), self.alphas_cumprod[:-1]], dim=0,
        )
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.zeros(1, device=self.accelerator.device)], dim=0,
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(
            torch.cat([torch.ones(1, device=self.accelerator.device), self.posterior_variance[1:]], dim=0)
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def sample_with_ddpm(self, n_samples, clf_model=None, control_label=None):
        assert not (clf_model is None) ^ (control_label is None), \
            '`clf_model` and `control_label` must be both None or not None.'

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        if clf_model is not None:
            clf_model = self.accelerator.prepare(clf_model)
            clf_model.eval()

        xt_shape = (n_samples, self.seq_len, self.latent_size)
        xt = torch.randn(xt_shape, device=self.accelerator.device)

        for _t in tqdm(reversed(range(self.num_timesteps)), disable=not self.accelerator.is_main_process):
            t = torch.ones(n_samples, dtype=torch.long, device=self.accelerator.device) * _t

            with torch.no_grad():
                x0_hat = unwrap_model_from_ddp(self.model).transformer(xt, rescale_timesteps(t.float(), self.num_timesteps))
                x0_hat = self._clamping_trick(x0_hat)

            posterior_mean = (
                self.posterior_mean_coef1[t][:, None, None] * x0_hat
                + self.posterior_mean_coef2[t][:, None, None] + xt
            )
            log_variance = self.posterior_log_variance[_t]
            noise = torch.randn_like(xt)
            sigma = torch.exp(0.5 * log_variance)

            xt = posterior_mean + sigma * noise if _t != 0 else xt

            if control_label is not None:
                xt = self.pnp_control(xt, posterior_mean, sigma, t, clf_model, control_label)

        xt = self.accelerator.gather_for_metrics(xt)
        out_ids = unwrap_model_from_ddp(self.model).decoder(xt)

        return out_ids

    def sample_with_ddim(self, n_samples, clf_model=None, control_label=None, eta=0.0):
        assert not (clf_model is None) ^ (control_label is None), \
            '`clf_model` and `control_label` must be both None or not None.'

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        if clf_model is not None:
            clf_model = self.accelerator.prepare(clf_model)
            clf_model.eval()

        xt_shape = (n_samples, self.seq_len, self.latent_size)
        xt = torch.randn(xt_shape, device=self.accelerator.device)

        for _t in tqdm(reversed(range(self.num_timesteps)), disable=not self.accelerator.is_main_process):
            t = torch.ones(n_samples, dtype=torch.long, device=self.accelerator.device) * _t

            with torch.no_grad():
                x0_hat = unwrap_model_from_ddp(self.model).transformer(xt, rescale_timesteps(t.float(), self.num_timesteps))
                x0_hat = self._clamping_trick(x0_hat)

            e = (
                (torch.sqrt(1.0 / self.alphas_cumprod[t])[:, None, None] * xt - x0_hat)
                / (torch.sqrt(1.0 / self.alphas_cumprod[t] - 1)[:, None, None])
            )
            sigma = (
                eta
                * torch.sqrt((1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t]))
                * torch.sqrt(1 - self.alphas_cumprod[t] / self.alphas_cumprod_prev[t])
            )
            mean = (
                torch.sqrt(self.alphas_cumprod_prev[t])[:, None, None] * x0_hat
                + torch.sqrt(1 - self.alphas_cumprod_prev[t] - sigma**2)[:, None, None] * e
            )
            sigma = sigma[:, None, None].expand(xt.shape)
            xt = mean + torch.randn_like(xt) * sigma if _t != 0 else mean

            if control_label is not None:
                xt = self.pnp_control(xt, mean, sigma, t, clf_model, control_label)

        xt = self.accelerator.gather_for_metrics(xt)
        out_ids = unwrap_model_from_ddp(self.model).decoder(xt)

        return out_ids

    @torch.inference_mode()
    def _clamping_trick(self, xt):
        encoder_weight = unwrap_model_from_ddp(self.model).encoder.encoder.weight
        encoder_weight_norm = (encoder_weight ** 2).sum(-1).view(-1, 1)  # [vocab_size, 1]

        xt_shape = xt.shape
        reshaped_xt = xt.reshape(-1, xt_shape[-1])  # [batch_size * seq_len, latent_size]
        xt_norm = (xt ** 2).sum(-1).view(-1, 1)  # [batch_size * seq_len, 1]

        dist = (
            encoder_weight_norm + xt_norm.transpose(0, 1)
            - 2.0 * torch.mm(encoder_weight, reshaped_xt.transpose(0, 1))
        )
        dist = torch.clamp(dist, 0.0, np.inf)

        topk_out = torch.topk(-dist, k=1, dim=0)
        tokens = topk_out.indices[0]

        new_xt = unwrap_model_from_ddp(self.model).encoder(tokens).view(xt_shape).to(self.accelerator.device)

        return new_xt

    def pnp_control(
        self,
        xt,
        mean,
        sigma,
        t,
        clf_model,
        control_label,
        num_updates=3,
        no_update_threshold=10,
        step_size=0.1,
        coef=0.01,
    ):
        if t[0].item() < no_update_threshold:
            return xt

        xt = torch.nn.Parameter(xt)

        labels = control_label.clone()
        target = control_label[:, self.seq_len:]
        with torch.no_grad():
            target_embs = unwrap_model_from_ddp(self.model).encoder(target)

        with torch.enable_grad():
            for _ in range(num_updates):
                optimizer = torch.optim.Adagrad(params=[xt], lr=step_size)
                optimizer = self.accelerator.prepare(optimizer)
                optimizer.zero_grad()

                input_embs = torch.cat([xt, target_embs], dim=1)
                clf_output = clf_model(input_embs=input_embs, labels=labels, t=t)

                if sigma.mean() == 0:
                    logp = ((mean - xt) ** 2 / 1.).mean(dim=0).sum()
                else:
                    logp = ((mean - xt) ** 2 / sigma).mean(dim=0).sum()

                loss = coef * logp + clf_output.loss
                self.accelerator.backward(loss)
                optimizer.step()

                xt = torch.nn.Parameter(xt.data.detach())

        xt = xt.data

        return xt
