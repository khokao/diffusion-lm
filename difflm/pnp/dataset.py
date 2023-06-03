"""The codes are modified.

- Link:
    - https://github.com/XiangLi1999/Diffusion-LM/
      blob/main/transformers/examples/pytorch/language-modeling/run_clm.py
"""
import collections
import itertools
from pathlib import Path

from spacy.lang.en import English

import datasets


def get_text_dataset_for_clf(mode, cfg, vocab_token2id=None, return_vocab=False):
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
    attr_seq_len = cfg['classifier']['attr_seq_len']

    tokenized_texts, tokenized_texts_with_attr = _load_data_for_clf(mode, dataset_root, dataset_type)
    if mode == 'train':
        vocab_token2id = _get_vocab_token2id(tokenized_texts)
    else:
        assert vocab_token2id is not None

    dataset = _create_dataset_for_clf(tokenized_texts_with_attr, vocab_token2id, seq_len, attr_seq_len)
    dataset = dataset.with_format('torch')
    dataset = dataset.shuffle()

    if return_vocab:
        return dataset, vocab_token2id
    else:
        return dataset


def _load_data_for_clf(mode, dataset_root, dataset_type):
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
    tokenized_texts_with_attr = []

    if dataset_type == 'e2e':
        attr_types = ['name', 'Type', 'area', 'customer rating', 'near', 'family friendly', 'food', 'price']

        txt_path = Path(dataset_root) / 'e2e' / f'{mode}.txt'
        assert txt_path.is_file()
        with txt_path.open() as fp:
            for line in fp:
                attr, text = line.split('||')

                tokenized_text = [token.text for token in nlp.tokenizer(text)]
                tokenized_texts.append(tokenized_text)

                attr_dict = {}
                for a in attr.split('|'):
                    k, v = a.split(':')
                    attr_dict[k.strip()] = v.strip()

                for attr_type in attr_types:
                    attr_text = f'{attr_type} : {attr_dict.get(attr_type, "none")}'
                    tokenized_attr_text = [token.text for token in nlp.tokenizer(attr_text)]
                    tokenized_texts_with_attr.append((tokenized_text, tokenized_attr_text))

    else:
        raise NotImplementedError()

    return tokenized_texts, tokenized_texts_with_attr


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


def _create_dataset_for_clf(tokenized_texts, vocab_token2id, seq_len, attr_seq_len=8):
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
        sentence_ids = (
            [vocab_token2id['START']]
            + [vocab_token2id.get(word, vocab_token2id['UNK']) for word in example['text'][0]]
            + [vocab_token2id['END']]
        )
        attr_ids = (
            [vocab_token2id.get(word, vocab_token2id['UNK']) for word in example['text'][1]]
            + [vocab_token2id['END']]
        )
        new_example = {'sentence_ids': sentence_ids, 'attr_ids': attr_ids}
        return new_example

    def _padding(example):
        sentence_unpadded_len = min(len(example['sentence_ids']), seq_len)
        sentence_padded_ids = [vocab_token2id['PAD']] * seq_len
        sentence_padded_ids[:sentence_unpadded_len] = example['sentence_ids'][:sentence_unpadded_len]

        attr_unpadded_len = min(len(example['attr_ids']), attr_seq_len)
        attr_padded_ids = [vocab_token2id['PAD']] * attr_seq_len
        attr_padded_ids[:attr_unpadded_len] = example['attr_ids'][:attr_unpadded_len]

        input_ids = sentence_padded_ids + attr_padded_ids
        labels = [-100] * len(sentence_padded_ids) + attr_padded_ids
        new_example = {'input_ids': input_ids, 'labels': labels}
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
        remove_columns=['sentence_ids', 'attr_ids'],
        load_from_cache_file=True,
        num_proc=1,
        desc='padding...',
    )

    return padded_dataset
