import pytest

import sys
sys.path.append('../')

import topmost
from topmost.data import download_dataset
from topmost.data import BasicDataset, DynamicDataset
from topmost.trainers import BasicTrainer, DynamicTrainer
from topmost.models import ETM, DETM


@pytest.fixture
def cache_path():
    return './pytest_cache/'


@pytest.fixture
def num_topics():
    return 10


def test_basic_evaluations(cache_path, num_topics):
    download_dataset("20NG", cache_path=cache_path)
    dataset = BasicDataset(f"{cache_path}/20NG", as_tensor=True)

    model = ETM(num_topics=num_topics, vocab_size=dataset.vocab_size)
    trainer = BasicTrainer(model, dataset, verbose=True, epochs=1)

    top_words = trainer.get_top_words()
    TD = topmost.evaluations.compute_topic_diversity(top_words)
    TC = topmost.evaluations.compute_topic_coherence(dataset.train_texts, dataset.vocab, top_words)
    print("TD: ", TD)
    print("TC: ", TC)


def test_dynamic_evaluations(cache_path, num_topics):
    download_dataset("NYT", cache_path=cache_path)
    dataset = DynamicDataset(f"{cache_path}/NYT", as_tensor=True)

    model = DETM(num_times=dataset.num_times, train_size=dataset.train_size, num_topics=num_topics, vocab_size=dataset.vocab_size, train_time_wordfreq=dataset.train_time_wordfreq)
    trainer = DynamicTrainer(model, dataset, verbose=True, epochs=1)

    top_words = trainer.get_top_words()
    TD = topmost.evaluations.dynamic_TD(top_words, dataset.train_bow.numpy(), dataset.train_times.numpy(), dataset.vocab)
    TC = topmost.evaluations.dynamic_TC(dataset.train_texts, dataset.train_times, dataset.vocab, top_words)

    print("TD: ", TD)
    print("TC: ", TC)
