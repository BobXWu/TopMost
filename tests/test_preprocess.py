import pytest
import numpy as np

import sys
sys.path.append('../')

from topmost import download_20ng, file_utils, Preprocess


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def test_datasets(cache_path):
    path = f'{cache_path}/20NG'
    download_20ng.download_save(output_dir=path)
    preprocess = Preprocess(vocab_size=5000, min_term=1, stopwords='snowball')
    rst = preprocess.preprocess_jsonlist(dataset_dir=path, label_name="group")
    preprocess.save(path, **rst)
