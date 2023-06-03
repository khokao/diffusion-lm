"""
The codes are modified.
Link:
    - [Trainer] https://github.com/Megvii-BaseDetection/YOLOX/
      blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/core/trainer.py#L36-L382
"""
from pathlib import Path
from time import time

import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.loss import E2ELoss
from .utils import DictMeter, Meter, TimestepSampler, get_betas, unwrap_model_from_ddp, rescale_timesteps

logger = get_logger(__name__)


class Trainer:
    def __init__(self, model, cfg, train_dataset, val_dataset, accelerator):
        self.model = model
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.accelerator = accelerator

        self.train_loader = DataLoader(self.train_dataset, **self.cfg['train']['dataloader'])
        self.val_loader = DataLoader(self.val_dataset, **self.cfg['train']['val']['dataloader'])

        self.setup_training_environment()

        self.model = self.model.to(self.accelerator.device)
        if self.accelerator.is_main_process:
            self.ema = EMA(
                self.model,
                beta=0.9999,
                update_after_step=100,
                update_every=10,
                inv_gamma=1.0,
                power=2 / 3,
                include_online_model=False
            )

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(self.optimizer)
        self.criterion = E2ELoss()

        self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(  # NOQA
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader,
        )

        # Set up timestep sampler.
        timestep_cfg = cfg['model']['timesteps']
        self.num_timesteps = timestep_cfg['num']
        self.timestep_sampler = TimestepSampler(timestep_cfg)

        # Set up beta and alpha.
        beta_cfg = cfg['model']['beta']
        self.betas = get_betas(beta_cfg=beta_cfg, num_timesteps=self.num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod = self.alphas_cumprod.to(self.accelerator.device)

    def get_optimizer(self):
        optimizer_cfg = self.cfg['train']['optimizer']
        optimizer_cls = getattr(torch.optim, optimizer_cfg['name'])
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_cfg['params'])
        logger.info(f'Use {optimizer_cfg["name"]} optimizer')
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler_cfg = self.cfg['train']['scheduler']
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cfg['name'])

        total_steps = self.cfg['train']['epoch'] * self.accelerator.num_processes
        if scheduler_cfg['name'] == 'CosineAnnealingLR':
            scheduler_cfg['params']['T_max'] = total_steps

        scheduler = scheduler_cls(optimizer, **scheduler_cfg['params'])
        logger.info(f'Use {scheduler_cfg["name"]} scheduler')

        return scheduler

    def setup_training_environment(self):
        set_seed(self.cfg['general']['seed'])
        self.log_interval = self.cfg['train']['log_interval']
        self.save_interval = self.cfg['train']['save_interval']
        self.val_interval = self.cfg['train']['val']['interval']
        self.val_use_ema = self.cfg['train']['val']['use_ema']
        self.ckpt_dir = Path(self.accelerator.project_dir) / 'checkpoints'
        logger.info(f'Set seed to {self.cfg["general"]["seed"]}')
        logger.info(f'Output a log for every {self.log_interval} iteration')
        logger.info(f'Save checkpoint every {self.save_interval} epoch')
        logger.info(f'Validate every {self.val_interval} epoch')
        logger.info(f'Use EMA for validation: {self.val_use_ema}')
        logger.info(f'Checkpoints are saved in {self.ckpt_dir}')

    def train(self):
        self.before_train()
        self.train_in_epoch()
        self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.cfg['train']['epoch']):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for batch in self.train_loader:
            self.before_iter()
            self.train_one_iter(batch)
            self.after_iter()

    def train_one_iter(self, batch):
        input_ids = batch['input_ids']
        x0_init = unwrap_model_from_ddp(self.model).encoder(input_ids)

        # Noise-added version is treated as x0.
        tmp_noise = torch.randn_like(x0_init)
        tmp_std = torch.sqrt(1 - self.alphas_cumprod[0]).view(-1, 1, 1)
        x0_noisy = x0_init + tmp_std * tmp_noise

        batch_size = input_ids.shape[0]
        t = self.timestep_sampler.sample(batch_size).to(self.accelerator.device)

        noise = torch.randn_like(x0_noisy)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
        xt = torch.sqrt(alpha_t) * x0_noisy + torch.sqrt(1.0 - alpha_t) * noise

        x0_hat = unwrap_model_from_ddp(self.model).transformer(xt, rescale_timesteps(t.float(), self.num_timesteps))
        _, x0_noisy_logits = unwrap_model_from_ddp(self.model).decoder(x0_noisy, return_logits=True)

        alpha_T = self.alphas_cumprod[self.num_timesteps - 1].view(-1, 1, 1)
        xT = torch.sqrt(alpha_T) * x0_noisy

        losses = self.criterion(
            x0_hat=x0_hat,
            x0_noisy_logits=x0_noisy_logits,
            xT=xT,
            x0_init=x0_init,
            x0_noisy=x0_noisy,
            input_ids=input_ids,
            t=t,
        )
        self.accelerator.backward(losses['total_loss'])

        self.accelerator.wait_for_everyone()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.train_loss_meter.update_all(self.accelerator.gather(losses))
        self.accelerator.wait_for_everyone()

    def before_train(self):
        self.iter = 0
        self.train_loss_meter = DictMeter({k: Meter() for k in self.criterion.LOSS_KEYS})
        self.val_loss_meter = DictMeter({k: Meter() for k in self.criterion.LOSS_KEYS})
        logger.info('Training start ...')

    def after_train(self):
        self.accelerator.save_state(output_dir=self.ckpt_dir)
        if self.accelerator.is_main_process:
            torch.save(self.ema.ema_model.state_dict(), self.ckpt_dir / 'pytorch_model_1.bin')
            logger.info(f'EMA model states saved in {self.ckpt_dir / "pytorch_model_1.bin"}')
        logger.info('Training done')

    def before_epoch(self):
        self.model.train()
        self.epoch_start_time = time()
        logger.info(f'---> Start train epoch {self.epoch + 1}')

    def after_epoch(self):
        self.scheduler.step()

        epoch_elapsed_time = time() - self.epoch_start_time
        logger.info(f'Epoch {self.epoch + 1} done. ({epoch_elapsed_time:.1f} sec)')

        if (self.epoch + 1) % self.save_interval == 0:
            output_dir = self.ckpt_dir / f'difflm_epoch_{str(self.epoch + 1).zfill(3)}'
            self.accelerator.save_state(output_dir=output_dir)

            if self.accelerator.is_main_process:
                torch.save(self.ema.ema_model.state_dict(), output_dir / 'pytorch_model_1.bin')
                logger.info(f'EMA model states saved in {output_dir / "pytorch_model_1.bin"}')

        if (self.epoch + 1) % self.val_interval == 0:
            if self.val_use_ema:
                if self.accelerator.is_main_process:
                    self.validate(val_model=self.ema.ema_model, use_distributed=False, prefix='ema_val')

            self.validate(val_model=self.model, use_distributed=self.accelerator.use_distributed, prefix='val')

    def before_iter(self):
        pass

    def after_iter(self):
        if self.accelerator.is_main_process:
            self.ema.update()

        if (self.iter + 1) % self.log_interval == 0:
            logger.info(
                'epoch: {}/{}, iter: {}/{}, loss: {:.3f}'.format(
                    self.epoch + 1, self.cfg['train']['epoch'],
                    (self.iter + 1) % len(self.train_loader), len(self.train_loader),
                    self.train_loss_meter['total_loss'].latest,
                )
            )
            self.accelerator.log(
                {f'train_{k}': v.latest for k, v in self.train_loss_meter.items()},
                step=self.iter + 1,
            )
            self.accelerator.log({'train_lr': self.scheduler.get_last_lr()[0]}, step=self.iter + 1)
            self.accelerator.wait_for_everyone()
            self.train_loss_meter.reset_all()

        self.iter += 1

    @torch.inference_mode()
    def validate(self, val_model, use_distributed, prefix='val'):
        logger.info('Validation start...')

        val_model.eval()
        for batch in tqdm(self.val_loader, disable=not self.accelerator.is_main_process):
            input_ids = batch['input_ids']
            x0_init = unwrap_model_from_ddp(val_model).encoder(input_ids)
            tmp_noise = torch.randn_like(x0_init)
            tmp_std = torch.sqrt(1 - self.alphas_cumprod[0]).view(-1, 1, 1)
            x0_noisy = x0_init + tmp_std * tmp_noise

            batch_size = input_ids.shape[0]
            t = self.timestep_sampler.sample(batch_size)

            noise = torch.randn_like(x0_noisy)
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
            xt = torch.sqrt(alpha_t) * x0_noisy + torch.sqrt(1.0 - alpha_t) * noise

            x0_hat = unwrap_model_from_ddp(val_model).transformer(xt, rescale_timesteps(t.float(), self.num_timesteps))
            _, x0_noisy_logits = unwrap_model_from_ddp(val_model).decoder(x0_noisy, return_logits=True)

            alpha_T = self.alphas_cumprod[self.num_timesteps - 1].view(-1, 1, 1)
            xT = torch.sqrt(alpha_T) * x0_noisy

            if use_distributed:
                x0_hat, x0_noisy_logits, xT, x0_init, x0_noisy, input_ids, t = self.accelerator.gather_for_metrics(
                    (x0_hat, x0_noisy_logits, xT, x0_init, x0_noisy, input_ids, t)
                )

            losses = self.criterion(
                x0_hat=x0_hat,
                x0_noisy_logits=x0_noisy_logits,
                xT=xT,
                x0_init=x0_init,
                x0_noisy=x0_noisy,
                input_ids=input_ids,
                t=t,
            )
            losses = self.accelerator.gather(losses) if use_distributed else losses
            self.val_loss_meter.update_all(losses)

        logger.info(f'Validation loss ({prefix}): {self.val_loss_meter["total_loss"].avg}')
        self.accelerator.log(
            {f'{prefix}_{k}': v.avg for k, v in self.train_loss_meter.items()},
            step=self.iter + 1,
        )

        if use_distributed:
            self.accelerator.wait_for_everyone()
        self.val_loss_meter.reset_all()
