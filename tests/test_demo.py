import pytest
import sys

sys.path.append("../")

import torch
import topmost
from topmost import eva, Preprocess


@pytest.fixture
def cache_path():
    return "./pytest_cache/"


@pytest.fixture
def num_topics():
    return 10


def test_demo(cache_path, num_topics):

    topmost.download_dataset("20NG", cache_path=cache_path)

    device = "cuda"  # or "cpu"

    # load a preprocessed dataset
    dataset = topmost.BasicDataset(f"{cache_path}/20NG", device=device, read_labels=True)
    # create a model
    model = topmost.ProdLDA(dataset.vocab_size, num_topics)
    model = model.to(device)

    # create a trainer
    trainer = topmost.BasicTrainer(model, dataset)

    # train the model
    top_words, train_theta = trainer.train()

    # topic diversity and coherence
    TD = eva._diversity(top_words)
    TC = eva._coherence(dataset.train_texts, dataset.vocab, top_words)

    # get doc-topic distributions of testing samples
    test_theta = trainer.test(dataset.test_data)
    # clustering
    clustering_results = eva._clustering(test_theta, dataset.test_labels)
    # classification
    cls_results = eva._cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)

    new_docs = [
        "This is a new document about space, including words like space, satellite, launch, orbit.",
        "This is a new document about Microsoft Windows, including words like windows, files, dos."
    ]

    preprocess = Preprocess()
    new_parsed_docs, new_bow = preprocess.parse(new_docs, vocab=dataset.vocab)
    new_theta = trainer.test(torch.as_tensor(new_bow.toarray(), device=device).float())
