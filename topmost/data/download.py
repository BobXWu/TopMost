import os
import zipfile
from torchvision.datasets.utils import download_url
from topmost.utils.logger import Logger


logger = Logger("WARNING")


def download_dataset(dataset_name, cache_path="~/.topmost"):
    cache_path = os.path.expanduser(cache_path)
    raw_filename = f'{dataset_name}.zip'

    if dataset_name in ['Wikitext-103']:
        # download from Git LFS.
        zipped_dataset_url = f"https://media.githubusercontent.com/media/BobXWu/TopMost/main/data/{raw_filename}"
    else:
        zipped_dataset_url = f"https://raw.githubusercontent.com/BobXWu/TopMost/master/data/{raw_filename}"

    logger.info(zipped_dataset_url)

    download_url(zipped_dataset_url, root=cache_path, filename=raw_filename, md5=None)

    path = f'{cache_path}/{raw_filename}'
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(cache_path)

    os.remove(path)


if __name__ == '__main__':
    download_dataset('20NG')
