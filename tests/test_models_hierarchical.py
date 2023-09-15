import pytest
import topmost
from topmost.data import download_dataset
from topmost.data import BasicDatasetHandler
from topmost.trainers import HierarchicalTrainer


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def hierarchical_model_test(model_module, dataset, num_topics_list):
    model = model_module(num_topics_list=num_topics_list, vocab_size=dataset.vocab_size)
    trainer = HierarchicalTrainer(model, dataset)

    beta = trainer.export_beta()
    assert len(beta) == len(num_topics_list)

    for i, layer_beta in enumerate(beta):
        assert layer_beta.shape == (num_topics_list[i], dataset.vocab_size)

    train_theta, test_theta = trainer.export_theta()
    assert len(train_theta) == len(num_topics_list)
    assert len(test_theta) == len(num_topics_list)

    for i, layer_theta in enumerate(train_theta):
        assert layer_theta.shape == (len(dataset.train_texts), num_topics_list[i])

    for i, layer_theta in enumerate(test_theta):
        assert layer_theta.shape == (len(dataset.test_texts), num_topics_list[i])


def test_models(cache_path):
    download_dataset("20NG", cache_path=cache_path)
    dataset = BasicDatasetHandler(f"{cache_path}/20NG", as_tensor=True)

    num_topics_list = [10, 50, 200]

    model_info = [
        topmost.models.SawETM,
        topmost.models.HyperMiner,
    ]

    for model_module in model_info:
        print(model_module)
        hierarchical_model_test(model_module, dataset, num_topics_list)
