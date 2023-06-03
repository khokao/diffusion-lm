import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .transformer import Transformer


class DiffusionLMModel(nn.Module):
    def __init__(self, cfg, vocab_size, hidden_size):
        """
        Args:
            cfg (dict): The configuration of the model.
            vocab_size (int): The size of vocabulary.
            hidden_size (int): The size of hidden space.
        """
        super().__init__()
        self.encoder = Encoder(vocab_size, hidden_size)
        self.transformer = Transformer(cfg)
        self.decoder = Decoder(vocab_size, hidden_size, self.encoder)
