import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse
import scipy.io
from . import file_utils


class BasicDatasetHandler:
    def __init__(self, dataset_dir, batch_size=200, read_labels=False, device='cpu', as_tensor=False):
        # train_bow: NxV
        # test_bow: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.

        # self.train_bow, self.test_bow, self.train_texts, self.test_texts, self.train_labels, self.test_labels, self.vocab, self.pretrained_WE = 
        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if as_tensor:
            self.train_bow = torch.from_numpy(self.train_bow).to(device)
            self.test_bow = torch.from_numpy(self.test_bow).to(device)
            self.train_dataloader = DataLoader(self.train_bow, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(self.test_bow, batch_size=batch_size, shuffle=False)

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
