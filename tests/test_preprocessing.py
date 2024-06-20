import pytest
import numpy as np

import sys
sys.path.append('../')

from topmost.data import download_20ng, file_utils
from topmost.preprocessing import Preprocessing


@pytest.fixture
def cache_path():
    return './pytest_cache/'


def test_datasets(cache_path):
    path = f'{cache_path}/20NG'
    download_20ng.download_save(output_dir=path)
    preprocessing = Preprocessing(vocab_size=5000, min_term=1, stopwords='snowball')
    rst = preprocessing.preprocess_jsonlist(dataset_dir=path, label_name="group")
    preprocessing.save(path, **rst)
