import pytest
import numpy as np

import sys
sys.path.append('../')

from topmost.data import download_dataset
from topmost.data import BasicDataset


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def dataset_test(cache_path, name, data_size, vocab_size, num_labels=None):
    print(name)
    print(num_labels)
    download_dataset(name, cache_path=cache_path)
    dataset = BasicDataset(f"{cache_path}/{name}", read_labels=(num_labels is not None))

    assert (len(dataset.train_texts) + len(dataset.test_texts)) == data_size
    assert dataset.vocab_size == vocab_size

    if num_labels:
        assert len(dataset.train_labels) == len(dataset.train_texts)
        assert len(dataset.test_labels) == len(dataset.test_texts)
        assert len(np.unique(dataset.train_labels)) == num_labels
        assert len(np.unique(dataset.test_labels)) == num_labels


def test_datasets(cache_path):
    dataset_info = [{
            "name": "20NG",
            "num_labels": 20,
            'data_size': 18846,
            'vocab_size': 5000
        },{
            "name": "IMDB",
            "num_labels": 2,
            'data_size': 50000,
            'vocab_size': 5000
        },{
            "name": "NeurIPS",
            'data_size': 7237,
            'vocab_size': 10000
        },{
            "name": "ACL",
            'data_size': 10560,
            'vocab_size': 10000
        },{
            "name": "NYT",
            "num_labels": 12,
            'data_size': 9172,
            'vocab_size': 10000
        }
    ]

    for item in dataset_info:
        print(item)
        dataset_test(cache_path, **item)
