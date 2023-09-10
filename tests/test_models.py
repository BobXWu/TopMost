import pytest
import numpy as np
from topmost.data import download_dataset
from topmost.data import BasicDatasetHandler
from topmost.models import ETM, ECRTM
from topmost.trainers import BasicTrainer


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def test_models(cache_path):
    download_dataset("20NG", cache_path=cache_path)
    dataset = BasicDatasetHandler(f"{cache_path}/20NG", as_tensor=True)

    num_topics = 50
    # model = ETM(num_topics=num_topics, vocab_size=dataset.vocab_size, init_WE=False)
    model = ECRTM(num_topics=num_topics, vocab_size=dataset.vocab_size)

    trainer = BasicTrainer(model, dataset)
    assert trainer.export_beta().shape == (num_topics, dataset.vocab_size)
    train_theta, test_theta = trainer.export_theta()
    assert train_theta.shape == (len(dataset.train_texts), num_topics)
    assert test_theta.shape == (len(dataset.test_texts), num_topics)
