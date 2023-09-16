import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def compute_TD(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    counter = vectorizer.fit_transform(texts).toarray()

    TF = counter.sum(axis=0)
    TD = (TF == 1).sum() / (K * T)

    return TD


def compute_topic_diversity(top_words, _type="TD"):
    TD = compute_TD(top_words)
    return TD


def compute_multiaspect_topic_diversity(top_words, _type="TD"):
    TD_list = list()
    for level_top_words in top_words:
        TD = compute_topic_diversity(level_top_words, _type)
        TD_list.append(TD)

    return np.mean(TD_list)
