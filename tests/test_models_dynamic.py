import pytest

import sys
sys.path.append('../')

import topmost
from topmost.data import download_dataset
from topmost.data import DynamicDatasetHandler
from topmost.trainers import DynamicTrainer


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def dynamic_model_test(model_module, dataset, num_topics):
    model = model_module(num_times=dataset.num_times, train_size=dataset.train_size, num_topics=num_topics, vocab_size=dataset.vocab_size, train_time_wordfreq=dataset.train_time_wordfreq)

    trainer = DynamicTrainer(model)

    beta = trainer.export_beta()
    assert beta.shape == (dataset.num_times, num_topics, dataset.vocab_size)

    train_theta, test_theta = trainer.export_theta(dataset)
    assert train_theta.shape == (len(dataset.train_texts), num_topics)
    assert test_theta.shape == (len(dataset.test_texts), num_topics)


def test_models(cache_path):
    download_dataset("NYT", cache_path=cache_path)
    dataset = DynamicDatasetHandler(f"{cache_path}/NYT", as_tensor=True)

    num_topics = 50

    model_info = [
        topmost.models.DETM,
    ]

    for model_module in model_info:
        print(model_module)
        dynamic_model_test(model_module, dataset, num_topics)
