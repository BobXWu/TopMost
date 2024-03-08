import pytest

import sys
sys.path.append('../')

import topmost
from topmost.data import download_dataset
from topmost.data import BasicDatasetHandler
from topmost.trainers import BasicTrainer


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def basic_model_test(model_module, dataset, num_topics):
    model = model_module(num_topics=num_topics, vocab_size=dataset.vocab_size)
    trainer = BasicTrainer(model, dataset)
    assert trainer.export_beta().shape == (num_topics, dataset.vocab_size)
    train_theta, test_theta = trainer.export_theta(dataset)
    assert train_theta.shape == (len(dataset.train_texts), num_topics)
    assert test_theta.shape == (len(dataset.test_texts), num_topics)


def test_models(cache_path):
    download_dataset("20NG", cache_path=cache_path)
    dataset = BasicDatasetHandler(f"{cache_path}/20NG", as_tensor=True)

    num_topics = 50

    model_info = [
        topmost.models.ProdLDA,
        topmost.models.ETM,
        topmost.models.DecTM,
        topmost.models.NSTM,
        topmost.models.TSCTM,
        topmost.models.ECRTM
    ]

    for model_module in model_info:
        print(model_module)
        basic_model_test(model_module, dataset, num_topics)
