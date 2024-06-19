"""
https://radimrehurek.com/gensim/models/hdpmodel.html
"""

import gensim
from gensim.models import HdpModel
from topmost.utils import _utils
from topmost.utils.logger import Logger


logger = Logger("WARNING")


class HDPGensimTrainer:
    def __init__(self,
                 dataset,
                 num_top_words=15,
                 max_chunks=None,
                 max_time=None,
                 chunksize=256,
                 kappa=1.0,
                 tau=64.0,
                 K=15,
                 T=150,
                 alpha=1,
                 gamma=1,
                 eta=0.01,
                 scale=1.0,
                 var_converge=0.0001,
                 verbose=False
                ):

        self.dataset = dataset
        self.num_top_words = num_top_words
        self.vocab_size = dataset.vocab_size
        self.max_chunks = max_chunks
        self.max_time = max_time
        self.chunksize = chunksize
        self.kappa = kappa
        self.tau = tau
        self.K = K
        self.T = T
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.scale = scale
        self.var_converge = var_converge

        self.verbose = verbose

    def train(self):
        train_bow = self.dataset.train_bow.astype("int32")
        id2word = dict(zip(range(self.vocab_size), self.dataset.vocab))
        corpus = gensim.matutils.Dense2Corpus(train_bow, documents_columns=False)

        self.model = HdpModel(
            corpus=corpus,
            id2word=id2word,
            max_chunks=self.max_chunks,
            max_time=self.max_time,
            chunksize=self.chunksize,
            kappa=self.kappa,
            tau=self.tau,
            K=self.K,
            T=self.T,
            alpha=self.alpha,
            gamma=self.gamma,
            eta=self.eta,
            scale=self.scale,
            var_converge=self.var_converge
        )

        top_words = self.get_top_words()
        train_theta = self.test(self.dataset.train_bow)

        return top_words, train_theta

    def test(self, bow):
        theta = list()
        beta = self.get_beta()
        corpus = gensim.matutils.Dense2Corpus(bow.astype('int32'), documents_columns=False)
        for doc in corpus:
            theta.append(self.model[doc])
        theta = gensim.matutils.corpus2dense(theta, num_docs=bow.shape[0], num_terms=beta.shape[0])
        theta = theta.transpose()
        return theta

    def get_beta(self):
        return self.model.get_topics()

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words
        beta = self.get_beta()
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words, self.verbose)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow)
        test_theta = self.test(self.dataset.test_bow)
        return train_theta, test_theta
