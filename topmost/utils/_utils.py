import numpy as np
from topmost.data import file_utils


def get_top_words(beta, vocab, num_top_words, verbose=False):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_words + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        if verbose:
            print('Topic {}: {}'.format(i, topic_str))

    return topic_str_list


def get_stopwords_set(stopwords=[]):
    from topmost.data import download_dataset

    if stopwords == 'English':
        from gensim.parsing.preprocessing import STOPWORDS as stopword_set

    elif stopwords in ['mallet', 'snowball']:
        download_dataset('stopwords', cache_path='./')
        stopwords = f'./stopwords/{stopwords}_stopwords.txt'
        stopword_set = file_utils.read_text(stopwords)

    stopword_set = frozenset(stopword_set)

    return stopword_set


if __name__ == '__main__':
    get_stopwords_set('English')
