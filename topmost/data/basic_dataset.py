import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse
import scipy.io
from sentence_transformers import SentenceTransformer
from topmost.preprocessing import Preprocessing
from . import file_utils
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable


class DocEmbedModel:
    def __init__(
            self,
            model: Union[str, callable]="all-MiniLM-L6-v2",
            device: str='cpu',
            verbose: bool=False
        ):
        self.verbose = verbose

        if isinstance(model, str): 
            self.model = SentenceTransformer(model, device=device)
        else:
            self.model = model

    def encode(self,
               docs:List[str],
               convert_to_tensor: bool=False
            ):

        embeddings = self.model.encode(
                        docs,
                        convert_to_tensor=convert_to_tensor,
                        show_progress_bar=self.verbose
                    )
        return embeddings


class RawDataset:
    def __init__(self,
                 docs,
                 preprocessing=None,
                 batch_size=200,
                 device='cpu',
                 as_tensor=True,
                 contextual_embed=False,
                 pretrained_WE=True,
                 doc_embed_model="all-MiniLM-L6-v2",
                 embed_model_device=None,
                 verbose=False
                ):

        if preprocessing is None:
            preprocessing = Preprocessing(verbose=verbose)

        rst = preprocessing.preprocess(docs, pretrained_WE=pretrained_WE)
        self.train_data = rst['train_bow']
        self.train_texts = rst['train_texts']
        self.vocab = rst['vocab']

        self.vocab_size = len(self.vocab)

        if contextual_embed:
            if embed_model_device is None:
                embed_model_device = device

            if isinstance(doc_embed_model, str):
                self.doc_embedder = DocEmbedModel(doc_embed_model, embed_model_device, verbose=verbose)
            else:
                self.doc_embedder = doc_embed_model

            self.train_contextual_embed = self.doc_embedder.encode(docs)
            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            if contextual_embed:
                self.train_data = np.concatenate((self.train_data, self.train_contextual_embed), axis=1)

            self.train_data = torch.from_numpy(self.train_data).float().to(device)
            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)


class BasicDataset:
    def __init__(self,
                 dataset_dir,
                 batch_size=200,
                 read_labels=False,
                 as_tensor=True,
                 contextual_embed=False,
                 doc_embed_model="all-MiniLM-L6-v2",
                 device='cpu'
                ):
        # train_bow: NxV
        # test_bow: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.

        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)

        print("train_size: ", self.train_bow.shape[0])
        print("test_size: ", self.test_bow.shape[0])
        print("vocab_size: ", self.vocab_size)
        print("average length: {:.3f}".format(self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if contextual_embed:
            self.doc_embedder = DocEmbedModel(doc_embed_model, device)
            self.train_contextual_embed = self.doc_embedder.encode(self.train_texts)
            self.test_contextual_embed = self.doc_embedder.encode(self.test_texts)

            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            if not contextual_embed:
                self.train_data = self.train_bow
                self.test_data = self.test_bow
            else:
                self.train_data = np.concatenate((self.train_bow, self.train_contextual_embed), axis=1)
                self.test_data = np.concatenate((self.test_bow, self.test_contextual_embed), axis=1)

            self.train_data = torch.from_numpy(self.train_data).to(device)
            self.test_data = torch.from_numpy(self.test_data).to(device)

            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels):

        self.train_bow = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        self.pretrained_WE = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

        if read_labels:
            self.train_labels = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')
