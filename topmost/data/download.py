import os
import zipfile
from torchvision.datasets.utils import download_url


def download_dataset(dataset_name, cache_path="~/.topmost"):
    cache_path = os.path.expanduser(cache_path)
    raw_filename = f'{dataset_name}.zip'
    zipped_dataset_url = "https://raw.githubusercontent.com/BobXWu/Code-TopMost/master/topmost/datasets/preprocessed_data/{raw_filename}"

    download_url(zipped_dataset_url, root=cache_path, filename=raw_filename, md5=None)

    with zipfile.ZipFile(f'{cache_path}/{raw_filename}', 'r') as zip_ref:
        zip_ref.extractall(cache_path)


if __name__ == '__main__':
    download_dataset('20NG')
