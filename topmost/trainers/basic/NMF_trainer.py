import gensim
from gensim.models import nmf
from topmost.utils import static_utils


class NMFGensimTrainer:
    def __init__(self, dataset_handler, vocab_size, num_topics=50, max_iter=1):
        self.dataset_handler = dataset_handler
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.max_iter = max_iter

    def train(self):
        train_bow = self.dataset_handler.train_bow.astype("int32")
        id2word = dict(zip(range(self.vocab_size), self.dataset_handler.vocab))
        corpus = gensim.matutils.Dense2Corpus(train_bow, documents_columns=False)

        self.model = nmf.Nmf(
            corpus=corpus,
            num_topics=self.num_topics,
            id2word=id2word,
            passes=self.max_iter
        )

    def test(self, bow):
        bow = bow.astype('int64')
        corpus = gensim.matutils.Dense2Corpus(bow, documents_columns=False)
        theta = gensim.matutils.corpus2dense(self.model.get_document_topics(corpus), num_docs=bow.shape[0], num_terms=self.num_topics)
        theta = theta.transpose()
        return theta

    def export_beta(self):
        return self.model.get_topics()

    def export_top_words(self, num_top=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab=self.dataset_handler.vocab, num_top=num_top)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset_handler.train_bow)
        test_theta = self.test(self.dataset_handler.test_bow)
        return train_theta, test_theta


class NMFSklearnTrainer:
    def __init__(self, model, dataset_handler):
        self.model = model
        self.dataset_handler = dataset_handler

    def train(self):
        train_bow = self.dataset_handler.train_bow.astype('int64')
        self.model.fit(train_bow)

    def test(self, bow):
        bow = bow.astype('int64')
        return self.model.transform(bow.astype('int64'))

    def export_beta(self):
        return self.model.components_

    def export_top_words(self, num_top=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab=self.dataset_handler.vocab, num_top=num_top)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset_handler.train_bow)
        test_theta = self.test(self.dataset_handler.test_bow)
        return train_theta, test_theta
