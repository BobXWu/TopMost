import pytest

import sys
sys.path.append('../')

import topmost
from topmost.data import download_dataset
from topmost.data import BasicDatasetHandler, DynamicDatasetHandler
from topmost.trainers import BasicTrainer, DynamicTrainer
from topmost.models import ETM, DETM


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def test_basic_evaluations(cache_path):
    download_dataset("20NG", cache_path=cache_path)
    dataset = BasicDatasetHandler(f"{cache_path}/20NG", as_tensor=True)

    num_topics = 50

    model = ETM(num_topics=num_topics, vocab_size=dataset.vocab_size)
    trainer = BasicTrainer(model)

    top_words = trainer.export_top_words(dataset.vocab)
    TD = topmost.evaluations.compute_topic_diversity(top_words)
    TC = topmost.evaluations.compute_topic_coherence(dataset.train_texts, dataset.vocab, top_words)
    print("TD: ", TD)
    print("TC: ", TC)


def test_dynamic_evaluations(cache_path):
    download_dataset("NYT", cache_path=cache_path)
    dataset = DynamicDatasetHandler(f"{cache_path}/NYT", as_tensor=True)

    num_topics = 50

    model = DETM(num_times=dataset.num_times, train_size=dataset.train_size, num_topics=num_topics, vocab_size=dataset.vocab_size, train_time_wordfreq=dataset.train_time_wordfreq)
    trainer = DynamicTrainer(model)

    top_words = trainer.export_top_words(dataset.vocab)
    TD = topmost.evaluations.multiaspect_topic_diversity(top_words)
    TC = topmost.evaluations.compute_dynamic_TC(dataset.train_texts, dataset.train_times, dataset.vocab, top_words)
    print("TD: ", TD)
    print("TC: ", TC)
