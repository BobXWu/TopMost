import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from tqdm import tqdm


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


def multiaspect_topic_diversity(top_words, _type="TD"):
    TD_list = list()
    for level_top_words in top_words:
        TD = compute_topic_diversity(level_top_words, _type)
        TD_list.append(TD)

    return np.mean(TD_list)


def _time_dynamic_TD(topics, time_vocab):
    num_associated_words = 0.
    T = len(topics[0].split())
    flatten_topic_words = [word for topic_words in topics for word in topic_words.split()]
    counter = Counter(flatten_topic_words)

    # for word in np.sort(list(set(flatten_topic_words))):
    for word in np.sort(flatten_topic_words):
        if (counter[word] == 1) and word in time_vocab:
            num_associated_words += 1

    return num_associated_words / (len(topics) * T)


def dynamic_TD(top_words, train_bow, train_times, vocab, verbose=False):
    TD_list = list()

    time_idx = np.sort(np.unique(train_times))

    for time in tqdm(time_idx):
        doc_idx = np.where(train_times == time)[0]
        time_vocab_idx = np.nonzero(train_bow[doc_idx].sum(0))[0]
        time_vocab = np.asarray(vocab)[time_vocab_idx]

        topics = top_words[time]
        TD_list.append(_time_dynamic_TD(topics, time_vocab))

    if verbose:
        print(f"dynamic TD list: {TD_list}")

    return np.mean(TD_list)
