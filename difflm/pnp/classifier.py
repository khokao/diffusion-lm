from transformers import GPT2PreTrainedModel, GPT2Model, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import torch.nn as nn
import torch

from ..utils import get_betas, TimestepSampler

class GPT2Classifier(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r'attn.masked_bias', r'attn.bias', r'lm_head.weight']

    def __init__(self, cfg, vocab_size, latent_size, encoder):
        gpt2_config = AutoConfig.from_pretrained(cfg['classifier']['gpt2model_config_name'])
        gpt2_config.vocab_size = vocab_size
        super().__init__(gpt2_config)

        # Set up timestep sampler.
        timestep_cfg = cfg['classifier']['timesteps']
        self.num_timesteps = timestep_cfg['num']
        self.timestep_sampler = TimestepSampler(timestep_cfg)

        # Set up beta and alpha.
        beta_cfg = cfg['model']['beta']
        betas = get_betas(beta_cfg=beta_cfg, num_timesteps=self.num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.first_block = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.Tanh(),
            nn.Linear(latent_size * 4, gpt2_config.n_embd)
        )
        self.time_embeddings = nn.Embedding(self.num_timesteps, gpt2_config.n_embd)
        self.transformer = GPT2Model(gpt2_config)
        self.transformer.wte = nn.Embedding(vocab_size, latent_size)
        self.lm_head = nn.Linear(gpt2_config.n_embd, gpt2_config.vocab_size, bias=False)

        self.post_init()

        self.transformer.wte.weight.data = encoder.encoder.weight.clone()
        self.transformer.wte.weight.requires_grad_(False)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        input_embs=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        t=None,
    ):
        if input_embs is None:
            input_embs = self.transformer.wte(input_ids)

        batch_size = input_embs.shape[0]
        if t is None:
            t = self.timestep_sampler.sample(batch_size).to(input_embs.device)
        time_embs = self.time_embeddings(t).unsqueeze(1)

        noise = torch.randn_like(input_embs)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
        input_embs = torch.sqrt(alpha_t) * input_embs + torch.sqrt(1.0 - alpha_t) * noise

        input_embs = self.first_block(input_embs)
        input_embs = torch.cat([time_embs, input_embs], dim=1)

        transformer_outputs = self.transformer(
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=input_embs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0][:, 1:, ]
        lm_logits = self.lm_head(hidden_states)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
