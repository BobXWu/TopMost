"""
https://radimrehurek.com/gensim/models/hdpmodel.html
"""

import gensim
from gensim.models import HdpModel
from topmost.utils import static_utils


class HDPGensimTrainer:
    def __init__(self, dataset, max_chunks=None, max_time=None, chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.0001):
        self.dataset = dataset
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
        return self.model.get_topics()

    def test(self, bow):
        theta = list()
        beta = self.export_beta()
        corpus = gensim.matutils.Dense2Corpus(bow.astype('int32'), documents_columns=False)
        for doc in corpus:
            theta.append(self.model[doc])
        theta = gensim.matutils.corpus2dense(theta, num_docs=bow.shape[0], num_terms=beta.shape[0])
        theta = theta.transpose()
        return theta

    def export_beta(self):
        return self.model.get_topics()

    def export_top_words(self, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab=self.dataset.vocab, num_top_words=num_top_words)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow)
        test_theta = self.test(self.dataset.test_bow)
        return train_theta, test_theta
