import pytest

import sys
sys.path.append('../')

import topmost
from topmost.data import download_dataset
from topmost.data import DynamicDataset
from topmost.trainers import DynamicTrainer


@pytest.fixture
def cache_path():
    return './pytest_cache/'


@pytest.fixture
def num_topics():
    return 5


def dynamic_model_test(model_module, dataset, num_topics):

    if model_module == topmost.models.DETM:
        model = model_module(num_times=dataset.num_times, train_size=dataset.train_size, num_topics=num_topics, vocab_size=dataset.vocab_size, train_time_wordfreq=dataset.train_time_wordfreq)
    elif model_module == topmost.models.CFDTM:
        model = model_module(num_times=dataset.num_times, pretrained_WE=dataset.pretrained_WE, num_topics=num_topics, vocab_size=dataset.vocab_size, train_time_wordfreq=dataset.train_time_wordfreq)

    trainer = DynamicTrainer(model, dataset, verbose=True, epochs=1)
    trainer.train()

    beta = trainer.get_beta()
    assert beta.shape == (dataset.num_times, num_topics, dataset.vocab_size)

    train_theta, test_theta = trainer.export_theta()
    assert train_theta.shape == (len(dataset.train_texts), num_topics)
    assert test_theta.shape == (len(dataset.test_texts), num_topics)


def test_models(cache_path, num_topics):
    download_dataset("NYT", cache_path=cache_path)
    dataset = DynamicDataset(f"{cache_path}/NYT", as_tensor=True)

    model_info = [
        topmost.models.DETM,
        topmost.models.CFDTM,
    ]

    for model_module in model_info:
        print(model_module)
        dynamic_model_test(model_module, dataset, num_topics)
