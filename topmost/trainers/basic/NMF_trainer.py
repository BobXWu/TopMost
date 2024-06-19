import gensim
from gensim.models import nmf
from topmost.utils import _utils


class NMFGensimTrainer:
    def __init__(self,
                 dataset,
                 num_topics=50,
                 num_top_words=15,
                 max_iter=1
                ):
        self.dataset = dataset
        self.num_topics = num_topics
        self.num_top_words = num_top_words
        self.vocab_size = dataset.vocab_size
        self.max_iter = max_iter

    def train(self):
        train_bow = self.dataset.train_bow.astype("int32")
        id2word = dict(zip(range(self.vocab_size), self.dataset.vocab))
        corpus = gensim.matutils.Dense2Corpus(train_bow, documents_columns=False)

        self.model = nmf.Nmf(
            corpus=corpus,
            num_topics=self.num_topics,
            id2word=id2word,
            passes=self.max_iter
        )

        top_words = self.get_top_words()
        train_theta = self.test(self.dataset.train_bow)

        return top_words, train_theta

    def test(self, bow):
        corpus = gensim.matutils.Dense2Corpus(bow.astype('int64'), documents_columns=False)
        theta = gensim.matutils.corpus2dense(self.model.get_document_topics(corpus), num_docs=bow.shape[0], num_terms=self.num_topics)
        theta = theta.transpose()
        return theta

    def get_beta(self):
        return self.model.get_topics()

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words
        beta = self.get_beta()
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow)
        test_theta = self.test(self.dataset.test_bow)
        return train_theta, test_theta


class NMFSklearnTrainer:
    def __init__(self,
                 model,
                 dataset,
                 num_top_words=15):
        self.model = model
        self.dataset = dataset
        self.num_top_words = num_top_words

    def train(self):
        train_bow = self.dataset.train_bow.astype('int64')
        self.model.fit(train_bow)

        top_words = self.get_top_words()
        train_theta = self.test(self.dataset.train_bow)
        return top_words, train_theta

    def test(self, bow):
        return self.model.transform(bow.astype('int64'))

    def get_beta(self):
        return self.model.components_

    def get_top_words(self, num_top_words=None):
        if num_top_words is None:
            num_top_words = self.num_top_words
        beta = self.get_beta()
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow)
        test_theta = self.test(self.dataset.test_bow)
        return train_theta, test_theta
