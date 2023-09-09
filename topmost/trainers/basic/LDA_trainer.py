import gensim
from gensim.models import LdaModel
from topmost.utils import static_utils


class LDAGensimTrainer:
    def __init__(self, dataset, num_topics=50, max_iter=1, alpha="symmetric", eta=None):
        self.dataset = dataset
        self.num_topics = num_topics
        self.vocab_size = dataset.vocab_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.eta = eta

    def train(self):
        train_bow = self.dataset.train_bow.astype("int32")
        id2word = dict(zip(range(self.vocab_size), self.dataset.vocab))
        corpus = gensim.matutils.Dense2Corpus(train_bow, documents_columns=False)
        self.model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=self.num_topics,
            passes=self.max_iter,
            alpha=self.alpha,
            eta=self.eta
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
        top_words = static_utils.print_topic_words(beta, vocab=self.dataset.vocab, num_top=num_top)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow)
        test_theta = self.test(self.dataset.test_bow)
        return train_theta, test_theta


class LDASklearnTrainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def train(self):
        train_bow = self.dataset.train_bow.astype('int64')
        self.model.fit(train_bow)

    def test(self, bow):
        bow = bow.astype('int64')
        return self.model.transform(bow.astype('int64'))

    def export_beta(self):
        return self.model.components_

    def export_top_words(self, num_top=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab=self.dataset.vocab, num_top=num_top)
        return top_words

    def export_theta(self):
        train_theta = self.test(self.dataset.train_bow)
        test_theta = self.test(self.dataset.test_bow)
        return train_theta, test_theta
