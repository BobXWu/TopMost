from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
from ..data.file_utils import split_text_word


def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)

    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary, topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score


def dynamic_TC(train_texts, train_times, vocab, top_words_list, cv_type='c_v', verbose=False):
    cv_score_list = list()

    for time, top_words in tqdm(enumerate(top_words_list)):
        # use the texts of each time slice as the reference corpus.
        idx = np.where(train_times == time)[0]
        reference_corpus = [train_texts[i] for i in idx]

        # use the topics at a time slice
        cv_score = compute_topic_coherence(reference_corpus, vocab, top_words, cv_type)
        cv_score_list.append(cv_score)

    if verbose:
        print(f"dynamic TC list: {cv_score_list}")

    return np.mean(cv_score_list)
