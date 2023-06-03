
import json
import time
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from .dataset import get_text_dataset
from .models.model import DiffusionLMModel
from .sampler import Sampler
from .trainer import Trainer
from .utils import setup_logger
from .pnp import get_text_dataset_for_clf, GPT2Classifier
import transformers
from spacy.lang.en import English


logger = get_logger(__name__)

class DiffusionLMInterface:
    CFG_DIR = Path(__file__).parent / 'cfg'

    def __init__(self, args, mode):
        assert mode in ['train', 'test', 'infer', 'clf_train']
        self.mode = mode
        self.cfg = self._init_config(args)

        if self.mode == 'train':
            self.output_dir = self._init_output_dir(args)
            saved_cfg_file = self.output_dir / 'model.yaml'
            with saved_cfg_file.open('w') as fp:
                yaml.safe_dump(self.cfg, fp, sort_keys=False)
        else:
            self.output_dir = Path(args['output'])

        if self.mode != 'clf_train':
            self.accelerator = self._init_accelerator(args)

        log_path = self.output_dir / f'{self.mode}.log'
        setup_logger(log_path, stream_level='INFO', file_level='INFO')

        logger.info(f'mode: {mode}')
        logger.info(f'Args: {args}')

        self._init_dataset()

        model_ckpt_path = self.output_dir / args['model_ckpt'] if 'model_ckpt' in args else None
        self.model = self._init_model(model_ckpt_path)

        if self.mode == 'clf_train':
            self.clf_model = self._init_clf_model()
        elif self.mode == 'infer' and args['control_label'] is not None:
            clf_ckpt_path = self.output_dir / args['clf_ckpt']
            self.clf_model = self._init_clf_model(clf_ckpt_path)
        else:
            self.clf_model = None

    def _init_config(self, args):
        """Setting up config dict.
        """
        # Load config file.
        if self.mode == 'train':
            cfg_path = self.CFG_DIR / 'model.yaml'
        else:
            cfg_path = Path(args['output']) / 'model.yaml'
        with cfg_path.open('r') as fp:
            cfg = yaml.safe_load(fp)

        return cfg

    def _init_output_dir(self, args):
        """Create output directory.
        """
        assert self.mode == 'train'
        output_root = Path(self.cfg['general']['output_root'])
        output_dir = output_root / time.strftime('%Y%m%d%H%M%S') if args['expn'] is None else output_root / args['expn']
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def _init_accelerator(self, args):
        project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            # automatic_checkpoint_naming=True,
            total_limit=10,
        )
        if self.mode == 'train' and args['wandb']:
            accelerator = Accelerator(log_with='wandb', project_config=project_config)
        else:
            accelerator = Accelerator(log_with=None, project_config=project_config)

        logger.info(f'AcceleratorState: {vars(accelerator.state)}')

        return accelerator

    def _init_model(self, ckpt_path=None):
        """Setting up module.
        """
        logger.info('Initializing Diffusion-LM...')
        latent_size = self.cfg['model']['network']['transformer']['in_channels']
        vocab_size = len(self.vocab_token2id)
        model = DiffusionLMModel(self.cfg, vocab_size, latent_size)
        if ckpt_path is not None:
            logger.info('Loading Diffusion-LM checkpoint...')
            model.load_state_dict(torch.load(ckpt_path))
        return model

    def _init_clf_model(self, ckpt_path=None):
        logger.info('Initializing Classifier...')
        latent_size = self.cfg['model']['network']['transformer']['in_channels']
        vocab_size = len(self.vocab_token2id)
        clf_model = GPT2Classifier(self.cfg, vocab_size, latent_size, encoder=self.model.encoder)
        if ckpt_path is not None:
            logger.info('Loading Classifier checkpoint...')
            clf_model.load_state_dict(torch.load(ckpt_path))
        return clf_model

    def _init_dataset(self):
        """Setting up dataset.
        """
        assert self.mode in {'train', 'test', 'infer', 'clf_train'}
        logger.info('Initializing dataset...')

        vocab_path = self.output_dir / 'vocab.json'
        if vocab_path.is_file():
            logger.info(f'Loading vocabulary from {str(vocab_path)}...')
            with vocab_path.open() as fp:
                self.vocab_token2id = json.load(fp)

        if self.mode == 'train':
            self.train_dataset, self.vocab_token2id = get_text_dataset('train', self.cfg, return_vocab=True)
            self.val_dataset = get_text_dataset('val', self.cfg, vocab_token2id=self.vocab_token2id)

            with vocab_path.open('w') as fp:
                json.dump(self.vocab_token2id, fp)

        elif self.mode == 'test':
            self.test_dataset = get_text_dataset(mode='test', cfg=self.cfg, vocab_token2id=self.vocab_token2id)

        elif self.mode == 'clf_train':
            self.train_dataset = get_text_dataset_for_clf('train', self.cfg, vocab_token2id=self.vocab_token2id)
            self.val_dataset = get_text_dataset_for_clf('val', self.cfg, vocab_token2id=self.vocab_token2id)

        else:
            pass

        self.vocab_id2token = {v: k for k, v in self.vocab_token2id.items()}

    def train(self):
        """Diffusion-LM training.
        """
        assert self.mode == 'train'

        self.accelerator.init_trackers(
            project_name='diffusion-lm',
            init_kwargs={'wandb': {'name': self.output_dir.name, 'dir': self.output_dir.resolve()}},
        )

        trainer = Trainer(self.model, self.cfg, self.train_dataset, self.val_dataset, self.accelerator)
        trainer.train()

        self.accelerator.end_training()

    def sample(self, n_samples, use_ddpm=False, control_label=None):
        assert self.mode == 'infer'
        self.sampler = Sampler(self.model, self.cfg, self.accelerator)

        if control_label is not None:
            nlp = English()
            attr_seq_len = self.cfg['classifier']['attr_seq_len']
            seq_len = self.cfg['model']['network']['transformer']['seq_len']

            control_label = [token.text for token in nlp(control_label)]
            padded_ids = [self.vocab_token2id['PAD']] * attr_seq_len
            control_label = (
                [self.vocab_token2id.get(token, self.vocab_token2id['UNK']) for token in control_label]
                + [self.vocab_token2id['END']]
            )
            unpadded_len = min(len(control_label), len(control_label))
            padded_ids[:unpadded_len] = control_label[:unpadded_len]
            control_label = [-100] * seq_len + padded_ids
            control_label = torch.tensor(control_label).expand(n_samples, -1).to(self.accelerator.device)

            beta_cfg = self.cfg['model']['beta']
            num_timestep = self.cfg['classifier']['timesteps']['num']
            self.sampler._init_diffusion_params(beta_cfg=beta_cfg, num_timesteps=num_timestep)

        if use_ddpm:
            out_ids = self.sampler.sample_with_ddpm(n_samples, self.clf_model, control_label)
        else:
            out_ids = self.sampler.sample_with_ddim(n_samples, self.clf_model, control_label, eta=1.0)

        saved_path = self.output_dir / 'sample_result.json'
        saved_list = self.save_output(out_ids, saved_path)

        return saved_list

    def save_output(self, out_ids, saved_path):
        removed_tokens = ['START', 'END', 'PAD', 'UNK', '\n']

        saved_list = []
        for seq in out_ids:
            raw_text = [self.vocab_id2token[s.item()] for s in seq]
            clean_text = [t for t in raw_text if t not in removed_tokens]
            saved_list.append(
                {
                    'raw_text': ' '.join(raw_text),
                    'clean_text': ' '.join(clean_text),
                }
            )

        with saved_path.open('w') as fp:
            json.dump(saved_list, fp, indent=4, ensure_ascii=False)
        logger.info(f'Generated samples are saved to {saved_path.name}')

        return saved_list

    def clf_train(self):
        assert self.mode == 'clf_train'

        clf_output_dir = self.output_dir / 'classifier'
        clf_output_dir.mkdir(exist_ok=True)

        trainer_args = transformers.TrainingArguments(output_dir=clf_output_dir, **self.cfg['classifier']['trainer'])
        trainer = transformers.Trainer(
            model=self.clf_model,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=transformers.default_data_collator,
        )
        print(trainer_args)
        trainer.train()
        trainer.save_model()
