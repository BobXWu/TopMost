from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
from ..data import file_utils


def compute_TC(reference_texts, dictionary, topics, num_top_words=15, cv_type='c_v'):
    cm = CoherenceModel(texts=reference_texts, dictionary=dictionary, topics=topics, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score


def compute_dynamic_TC(train_texts, train_times, vocab, top_words_list, num_top_words=15, cv_type='c_v'):
    dictionary = Dictionary(file_utils.split_text_word(vocab))
    split_train_texts = file_utils.split_text_word(train_texts)

    cv_score_list = list()

    for time in tqdm(range(len(top_words_list))):
        # use the texts of the time slice as reference.
        idx = np.where(train_times == time)[0]
        reference_texts = [split_train_texts[i] for i in idx]

        # use the the topics at the time slice
        top_words = top_words_list[time]
        split_top_words = file_utils.split_text_word(top_words)

        cv_score = compute_TC(reference_texts, dictionary, split_top_words, num_top_words, cv_type)
        cv_score_list.append(cv_score)

    print("===>CV score list: ", cv_score_list)

    return np.mean(cv_score_list)
