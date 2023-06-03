"""The codes are modified.

- Link:
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/improved-diffusion/improved_diffusion/text_datasets.py
"""
import collections
import itertools
from pathlib import Path

from spacy.lang.en import English

import datasets


def get_text_dataset(mode, cfg, vocab_token2id=None, return_vocab=False):
    """
    Args:
        mode (str): 'train', 'val', or 'test'.
        cfg (dict): The config dict.
        vocab_token2id (dict): The vocabulary dict.
        return_vocab (bool): Whether to return the vocabulary dict.

    Returns:
        dataset (datasets.arrow_dataset.Dataset): The arrow dataset with the following columns:
            - input_ids (List[int]): The input ids.
    """
    dataset_root = cfg['general']['dataset']['root']
    dataset_type = cfg['general']['dataset']['type']
    seq_len = cfg['model']['network']['transformer']['seq_len']

    tokenized_texts = _load_data(mode, dataset_root, dataset_type)
    if mode == 'train':
        vocab_token2id = _get_vocab_token2id(tokenized_texts)
    else:
        assert vocab_token2id is not None

    dataset = _create_dataset(tokenized_texts, vocab_token2id, seq_len)
    dataset = dataset.with_format('torch')

    if return_vocab:
        return dataset, vocab_token2id
    else:
        return dataset


def _load_data(mode, dataset_root, dataset_type):
    """
    Args:
        mode (str): 'train', 'val', or 'test'.
        dataset_root (str): The root directory of dataset.
        dataset_type (str): Name of the dataset, currently only `e2e` is supported

    Returns:
        tokenized_texts (List[List[str]]): The tokenized texts.
    """
    nlp = English()
    tokenized_texts = []

    if dataset_type == 'e2e':
        txt_path = Path(dataset_root) / 'e2e' / f'{mode}.txt'
        assert txt_path.is_file()
        with txt_path.open() as fp:
            for line in fp:
                text = line.split('||')[1]  # str
                tokenized_text = nlp.tokenizer(text)  # spacy Doc
                tokenized_texts.append([token.text for token in tokenized_text])
    else:
        raise NotImplementedError()

    return tokenized_texts


def _get_vocab_token2id(tokenized_texts, threshold=10):
    """
    Args:
        tokenized_texts (List[List[str]]): The tokenized texts.
        threshold (int): The threshold of word frequency.

    Returns:
        vocab_token2id (dict): The vocabulary dict.
    """
    words = list(itertools.chain.from_iterable(tokenized_texts))
    words_counter = collections.Counter(words)

    vocab_token2id = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
    for word, word_cnt in words_counter.items():
        if word_cnt > threshold:
            vocab_token2id[word] = len(vocab_token2id)

    return vocab_token2id


def _create_dataset(tokenized_texts, vocab_token2id, seq_len):
    """
    Args:
        tokenized_texts (List[List[str]]): The tokenized texts.
        vocab_token2id (dict): The vocabulary dict.
        seq_len (int): The length of sequence.

    Returns:
        padded_dataset (datasets.arrow_dataset.Dataset): The arrow dataset with the following columns:
            - input_ids (List[int]): The input ids.
    """
    def _tokenize(example):
        input_ids = (
            [vocab_token2id['START']]
            + [vocab_token2id.get(word, vocab_token2id['UNK']) for word in example['text']]
            + [vocab_token2id['END']]
        )
        new_example = {'input_ids': input_ids}
        return new_example

    def _padding(example):
        unpadded_len = min(len(example['input_ids']), seq_len)
        padded_ids = [vocab_token2id['PAD']] * seq_len
        padded_ids[:unpadded_len] = example['input_ids'][:unpadded_len]
        new_example = {'input_ids': padded_ids}
        return new_example

    raw_dataset = datasets.Dataset.from_dict({'text': tokenized_texts})
    tokenized_dataset = raw_dataset.map(
        _tokenize,
        batched=False,
        remove_columns=['text'],
        load_from_cache_file=True,
        num_proc=4,
        desc='tokenize..'
    )
    padded_dataset = tokenized_dataset.map(
        _padding,
        batched=False,
        load_from_cache_file=True,
        num_proc=1,
        desc='padding...',
    )

    return padded_dataset
