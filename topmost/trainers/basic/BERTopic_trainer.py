from bertopic import BERTopic


class BERTopicTrainer:
    def __init__(self, dataset, num_topics=50, num_top_words=15):
        self.model = BERTopic(nr_topics=num_topics, top_n_words=num_top_words)
        self.dataset = dataset

    def train(self):
        self.model.fit_transform(self.dataset.train_texts)

    def test(self, texts):
        theta, _ = self.model.approximate_distribution(texts)
        return theta

    def export_beta(self):
        # NOTE: beta is modeled as unnormalized c-tf_idf.
        beta = self.model.c_tf_idf_.toarray()
        return beta

    def export_top_words(self):
        top_words = list()
        for item in self.model.get_topics().values():
            top_words.append(' '.join([x[0] for x in item]))
        return top_words

    def export_theta(self):
        train_theta, _ = self.test(self.dataset.train_texts)
        test_theta, _ = self.test(self.dataset.test_texts)
        return train_theta, test_theta
