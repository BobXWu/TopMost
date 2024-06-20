import pytest

import sys
sys.path.append('../')

import topmost
from topmost.data import download_dataset
from topmost.data import BasicDataset
from topmost.trainers import BasicTrainer, BERTopicTrainer, FASTopicTrainer, LDAGensimTrainer, NMFGensimTrainer


@pytest.fixture
def cache_path():
    return './pytest_cache/'


@pytest.fixture
def num_topics():
    return 10


def basic_model_test(model_module, dataset, num_topics, trainer=None):
    if model_module == 'BERTopic':
        trainer = BERTopicTrainer(dataset, num_topics=num_topics)

    elif model_module == 'FASTopic':
        trainer = FASTopicTrainer(dataset, num_topics=num_topics, epochs=1)
    
    elif model_module == 'LDAGensim':
        trainer = LDAGensimTrainer(dataset, num_topics=num_topics)
    
    elif model_module == 'NMFGensim':
        trainer = NMFGensimTrainer(dataset, num_topics=num_topics)

    elif model_module == 'CombinedTM':
        model = topmost.models.CombinedTM(vocab_size=dataset.vocab_size, contextual_embed_size=dataset.contextual_embed_size, num_topics=num_topics)
        trainer = BasicTrainer(model, dataset, verbose=True, epochs=1)

    else:
        model = model_module(num_topics=num_topics, vocab_size=dataset.vocab_size)
        trainer = BasicTrainer(model, dataset, verbose=True, epochs=1)

    trainer.train()

    assert trainer.get_beta().shape == (num_topics, dataset.vocab_size)

    train_theta, test_theta = trainer.export_theta()

    if model_module != 'BERTopic':
        assert train_theta.shape == (len(dataset.train_texts), num_topics)
        assert test_theta.shape == (len(dataset.test_texts), num_topics)


def test_models(cache_path, num_topics):
    download_dataset("20NG", cache_path=cache_path)
    dataset = BasicDataset(f"{cache_path}/20NG")

    model_info = [
        topmost.models.ProdLDA,
        topmost.models.ETM,
        topmost.models.DecTM,
        topmost.models.NSTM,
        topmost.models.TSCTM,
        topmost.models.ECRTM,
        'BERTopic',
        'FASTopic',
        'LDAGensim',
        'NMFGensim'
    ]

    for model_module in model_info:
        print(model_module)
        basic_model_test(model_module, dataset, num_topics)

    dataset = BasicDataset(f"{cache_path}/20NG", contextual_embed=True)
    basic_model_test("CombinedTM", dataset, num_topics)
