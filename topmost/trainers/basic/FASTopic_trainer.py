from fastopic import FASTopic
from topmost.preprocessing import Preprocessing


class FASTopicTrainer:
    def __init__(self, num_topics=50, preprocessing=None, num_top_words=15):
        preprocessing = Preprocessing(stopwords=[])
        self.model = FASTopic(num_topics=num_topics,
                              preprocessing=preprocessing,
                              num_top_words=num_top_words
                            )

    def train(self, dataset_handler, num_top_words=15):
        self.model.fit_transform(dataset_handler.train_texts)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_texts)

        return top_words, train_theta

    def test(self, texts):
        theta = self.model.transform(texts)
        return theta

    def export_beta(self):
        beta = self.model.get_topics()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        top_words = self.model.get_top_words(vocab, num_top_words)
        return top_words

    def export_theta(self, dataset):
        train_theta = self.test(dataset.train_texts)
        test_theta = self.test(dataset.test_texts)
        return train_theta, test_theta
