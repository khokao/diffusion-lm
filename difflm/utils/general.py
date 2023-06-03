"""
The codes are modified.
Link:
    - [Meter] https://github.com/Megvii-BaseDetection/YOLOX/
      blob/a5bb5ab12a61b8a25a5c3c11ae6f06397eb9b296/yolox/utils/metric.py#L54-L98
"""
import logging
from collections import UserDict, deque

import numpy as np
import torch


def unwrap_model_from_ddp(model):
    return model.module if hasattr(model, 'module') else model


def setup_logger(log_file, stream_level='INFO', file_level='INFO'):
    stream_level = getattr(logging, stream_level.upper(), None)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    file_level = getattr(logging, file_level.upper(), None)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    logging.basicConfig(level=logging.NOTSET, handlers=[stream_handler, file_handler])


class DictMeter(UserDict):
    def __init__(self, meter_dict):
        super().__init__(meter_dict)

    def update_all(self, values_dict):
        for k, v in values_dict.items():
            self[k].update(v)

    def reset_all(self):
        for m in self.values():
            m.reset()


class Meter:
    def __init__(self):
        self._deque = deque()
        self._count = 0
        self._total = 0.0

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.mean().item()

        self._deque.append(value)
        self._count += 1
        self._total += value

    def reset(self):
        self._deque.clear()
        self._count = 0
        self._total = 0.0

    @property
    def avg(self):
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None
