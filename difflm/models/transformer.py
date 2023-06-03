"""The codes are modified.

- Link:
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/improved-diffusion/improved_diffusion/transformer_model2.py
"""
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder

from .network_blocks import SinusoidalPossitionalEmbedding


class Transformer(nn.Module):
    """Denoising network based on Transformer.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (dict): A dict of config.
        """
        super().__init__()
        self.network_cfg = cfg['model']['network']['transformer']
        self.in_chans = self.network_cfg['in_channels']
        self.out_chans = self.network_cfg['out_channels']
        self.model_chans = self.network_cfg['model_channels']
        self.dropout = self.network_cfg['dropout']
        self.seq_len = self.network_cfg['seq_len']

        self.time_emb_chans = self.model_chans * 4

        bertmodel_config_name = self.network_cfg['bertmodel_config_name']
        self.bertmodel_config = AutoConfig.from_pretrained(bertmodel_config_name)

        position_ids = torch.arange(self.bertmodel_config.max_position_embeddings).expand((1, -1))
        self.register_buffer('position_ids', position_ids)

        self._create_network()

    def _create_network(self):
        self.time_mlp = nn.Sequential(
            SinusoidalPossitionalEmbedding(self.model_chans),
            nn.Linear(self.model_chans, self.time_emb_chans),
            nn.SiLU(),
            nn.Linear(self.time_emb_chans, self.bertmodel_config.hidden_size),
        )
        self.position_embeddings = nn.Embedding(
            self.bertmodel_config.max_position_embeddings,
            self.bertmodel_config.hidden_size
        )

        self.first_block = nn.Sequential(
            nn.Linear(self.in_chans, self.bertmodel_config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.bertmodel_config.hidden_size, self.bertmodel_config.hidden_size)
        )

        self.layernorm_dropout = nn.Sequential(
            nn.LayerNorm(self.bertmodel_config.hidden_size, self.bertmodel_config.layer_norm_eps),
            nn.Dropout(self.bertmodel_config.hidden_dropout_prob),
        )

        self.transformer_block = BertEncoder(self.bertmodel_config)

        self.last_block = nn.Sequential(
            nn.Linear(self.bertmodel_config.hidden_size, self.bertmodel_config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.bertmodel_config.hidden_size, self.out_chans)
        )

    def forward(self, xt, t):
        """
        Args:
            xt (torch.tensor): Data x at timestep t.
                shape == (batch_size, seq_len, latent_size)
            t (torch.tensor): Timestep.
                shape == (batch_size,)
        """
        time_emb = self.time_mlp(t)
        time_emb = time_emb.unsqueeze(1).expand(-1, self.seq_len, -1)
        position_emb = self.position_embeddings(self.position_ids[:, :self.seq_len])

        out = self.first_block(xt)
        out += time_emb
        out += position_emb
        out = self.layernorm_dropout(out)
        out = self.transformer_block(out).last_hidden_state
        out = self.last_block(out)

        return out
