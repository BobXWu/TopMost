import pytest

import sys
sys.path.append('../')

import topmost
from topmost.data import download_dataset
from topmost.data import CrosslingualDataset
from topmost.trainers import CrosslingualTrainer


@pytest.fixture
def cache_path():
    return './pytest_cache/'


@pytest.fixture
def num_topics():
    return 5


def crosslingual_model_test(model, dataset, num_topics):

    trainer = CrosslingualTrainer(model, dataset, verbose=True, epochs=1)
    trainer.train()

    beta_en, beta_cn = trainer.get_beta()
    assert beta_en.shape == beta_cn.shape
    assert beta_en.shape == (num_topics, dataset.vocab_size_en)

    train_theta_en, train_theta_cn, test_theta_en, test_theta_cn = trainer.export_theta()
    assert train_theta_en.shape == (len(dataset.train_texts_en), num_topics)
    assert test_theta_en.shape == (len(dataset.test_texts_en), num_topics)
    assert train_theta_cn.shape == (len(dataset.train_texts_cn), num_topics)
    assert test_theta_cn.shape == (len(dataset.test_texts_cn), num_topics)


def test_models(cache_path, num_topics):
    download_dataset("Amazon_Review", cache_path=cache_path)
    download_dataset('dict', cache_path=cache_path)

    dataset = CrosslingualDataset(f"{cache_path}/Amazon_Review", lang1='en', lang2='cn', dict_path=f'{cache_path}/dict/ch_en_dict.dat', as_tensor=True)

    model = topmost.models.NMTM(
        num_topics=num_topics,
        Map_en2cn=dataset.Map_en2cn,
        Map_cn2en=dataset.Map_cn2en,
        vocab_size_en=dataset.vocab_size_en,
        vocab_size_cn=dataset.vocab_size_cn,
    )

    crosslingual_model_test(model, dataset, num_topics)

    model = topmost.models.InfoCTM(
        num_topics=num_topics,
        trans_e2c=dataset.trans_matrix_en,
        pretrain_word_embeddings_en=dataset.pretrained_WE_en,
        pretrain_word_embeddings_cn=dataset.pretrained_WE_cn,
        vocab_size_en=dataset.vocab_size_en,
        vocab_size_cn=dataset.vocab_size_cn,
        weight_MI=50
    )

    crosslingual_model_test(model, dataset, num_topics)
