import numpy as np
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from ..data.file_utils import split_text_word
from typing import List


def _coherence(
        reference_corpus: List[str],
        vocab: List[str],
        top_words: List[str],
        coherence_type='c_v',
        topn=20
    ):
    split_top_words = split_text_word(top_words)
    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(
        texts=split_reference_corpus,
        dictionary=dictionary,
        topics=split_top_words,
        topn=topn,
        coherence=coherence_type,
    )
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return score


def dynamic_coherence(train_texts, train_times, vocab, top_words_list, coherence_type='c_v', verbose=False):
    cv_score_list = list()

    for time, top_words in tqdm(enumerate(top_words_list)):
        # use the texts of each time slice as the reference corpus.
        idx = np.where(train_times == time)[0]
        reference_corpus = [train_texts[i] for i in idx]

        # use the topics at a time slice
        cv_score = _coherence(reference_corpus, vocab, top_words, coherence_type)
        cv_score_list.append(cv_score)

    if verbose:
        print(f"dynamic TC list: {cv_score_list}")

    return np.mean(cv_score_list)
