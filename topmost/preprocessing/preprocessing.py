'''
This script is partially based on https://github.com/dallascard/scholar.
'''

import os
import re
import string
import gensim.downloader
from collections import Counter
import numpy as np
import scipy.sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from topmost.data import file_utils
from topmost.utils._utils import get_stopwords_set
from topmost.utils.logger import Logger


logger = Logger("WARNING")


# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


class Tokenizer:
    def __init__(self,
                 stopwords,
                 keep_num,
                 keep_alphanum,
                 strip_html,
                 no_lower,
                 min_length
                ):
        self.keep_num = keep_num
        self.keep_alphanum = keep_alphanum
        self.strip_html = strip_html
        self.lower = not no_lower
        self.min_length = min_length

        self.stopword_set = get_stopwords_set(stopwords)

    def clean_text(self, text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
        # remove html tags
        if strip_html:
            text = re.sub(r'<[^>]+>', '', text)
        else:
            # replace angle brackets
            text = re.sub(r'<', '(', text)
            text = re.sub(r'>', ')', text)
        # lower case
        if lower:
            text = text.lower()
        # eliminate email addresses
        if not keep_emails:
            text = re.sub(r'\S+@\S+', ' ', text)
        # eliminate @mentions
        if not keep_at_mentions:
            text = re.sub(r'\s@\S+', ' ', text)
        # replace underscores with spaces
        text = re.sub(r'_', ' ', text)
        # break off single quotes at the ends of words
        text = re.sub(r'\s\'', ' ', text)
        text = re.sub(r'\'\s', ' ', text)
        # remove periods
        text = re.sub(r'\.', '', text)
        # replace all other punctuation (except single quotes) with spaces
        text = replace.sub(' ', text)
        # remove single quotes
        text = re.sub(r'\'', '', text)
        # replace all whitespace with a single space
        text = re.sub(r'\s', ' ', text)
        # strip off spaces on either end
        text = text.strip()
        return text

    def tokenize(self, text):
        text = self.clean_text(text, self.strip_html, self.lower)
        tokens = text.split()

        tokens = ['_' if t in self.stopword_set else t for t in tokens]

        # remove tokens that contain numbers
        if not self.keep_alphanum and not self.keep_num:
            tokens = [t if alpha.match(t) else '_' for t in tokens]

        # or just remove tokens that contain a combination of letters and numbers
        elif not self.keep_alphanum:
            tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]

        # drop short tokens
        if self.min_length > 0:
            tokens = [t if len(t) >= self.min_length else '_' for t in tokens]

        unigrams = [t for t in tokens if t != '_']
        # counts = Counter()
        # counts.update(unigrams)

        return unigrams


def make_word_embeddings(vocab):
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
    word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))

    num_found = 0

    try:
        key_word_list = glove_vectors.index_to_key
    except:
        key_word_list = glove_vectors.index2word

    for i, word in enumerate(tqdm(vocab, desc="loading word embeddings")):
        if word in key_word_list:
            word_embeddings[i] = glove_vectors[word]
            num_found += 1

    logger.info(f'number of found embeddings: {num_found}/{len(vocab)}')

    return scipy.sparse.csr_matrix(word_embeddings)


class Preprocessing:
    def __init__(self,
                 tokenizer=None,
                 test_sample_size=None,
                 test_p=0.2,
                 stopwords=[],
                 min_doc_count=0,
                 max_doc_freq=1.0,
                 keep_num=False,
                 keep_alphanum=False,
                 strip_html=False,
                 no_lower=False,
                 min_length=3,
                 min_term=0,
                 vocab_size=None,
                 seed=42,
                 verbose=True
                ):
        """
        Args:
            test_sample_size:
                Size of the test set.
            test_p:
                Proportion of the test set. This helps sample the train set based on the size of the test set.
            stopwords:
                List of stopwords to exclude.
            min-doc-count:
                Exclude words that occur in less than this number of documents.
            max_doc_freq:
                Exclude words that occur in more than this proportion of documents.
            keep-num:
                Keep tokens made of only numbers.
            keep-alphanum:
                Keep tokens made of a mixture of letters and numbers.
            strip_html:
                Strip HTML tags.
            no-lower:
                Do not lowercase text
            min_length:
                Minimum token length.
            min_term:
                Minimum term number
            vocab-size:
                Size of the vocabulary (by most common in the union of train and test sets, following above exclusions)
            seed:
                Random integer seed (only relevant for choosing test set)
        """

        self.test_sample_size = test_sample_size
        self.min_doc_count = min_doc_count
        self.max_doc_freq = max_doc_freq
        self.min_term = min_term
        self.test_p = test_p
        self.vocab_size = vocab_size
        self.seed = seed

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(stopwords, keep_num, keep_alphanum, strip_html, no_lower, min_length).tokenize

        if verbose:
            logger.set_level("DEBUG")
        else:
            logger.set_level("WARNING")

    def parse(self, texts, vocab):
        if not isinstance(texts, list):
            texts = [texts]

        vocab_set = set(vocab)
        parsed_texts = list()
        for i, text in enumerate(tqdm(texts, desc="parsing texts")):
            tokens = self.tokenizer(text)
            tokens = [t for t in tokens if t in vocab_set]
            parsed_texts.append(' '.join(tokens))

        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        bow = vectorizer.fit_transform(parsed_texts)
        bow = bow.toarray()
        return parsed_texts, bow

    def preprocess_jsonlist(self, dataset_dir, label_name=None):
        train_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'train.jsonlist'))
        test_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'test.jsonlist'))

        logger.info(f"Found training documents {len(train_items)} testing documents {len(test_items)}")

        raw_train_texts = []
        train_labels = []
        raw_test_texts = []
        test_labels = []

        for item in train_items:
            raw_train_texts.append(item['text'])

            if label_name is not None:
                train_labels.append(item[label_name])
 
        for item in test_items:
            raw_test_texts.append(item['text'])

            if label_name is not None:
                test_labels.append(item[label_name])

        rst = self.preprocess(raw_train_texts, train_labels, raw_test_texts, test_labels)

        return rst

    def convert_labels(self, train_labels, test_labels):
        if train_labels is not None:
            label_list = list(set(train_labels))
            label_list.sort()
            n_labels = len(label_list)
            label2id = dict(zip(label_list, range(n_labels)))

            logger.info(f"label2id: {label2id}")

            train_labels = [label2id[label] for label in train_labels]

            if test_labels is not None:
                test_labels = [label2id[label] for label in test_labels]

        return train_labels, test_labels

    def preprocess(self, raw_train_texts, train_labels=None, raw_test_texts=None, test_labels=None, pretrained_WE=True):
        np.random.seed(self.seed)

        train_texts = list()
        test_texts = list()
        word_counts = Counter()
        doc_counts_counter = Counter()

        train_labels, test_labels = self.convert_labels(train_labels, test_labels)

        for text in tqdm(raw_train_texts, desc="loading train texts"):
            tokens = self.tokenizer(text)
            word_counts.update(tokens)
            doc_counts_counter.update(set(tokens))
            parsed_text = ' '.join(tokens)
            train_texts.append(parsed_text)

        if raw_test_texts:
            for text in tqdm(raw_test_texts, desc="loading test texts"):
                tokens = self.tokenizer(text)
                word_counts.update(tokens)
                doc_counts_counter.update(set(tokens))
                parsed_text = ' '.join(tokens)
                test_texts.append(parsed_text)

        words, doc_counts = zip(*doc_counts_counter.most_common())
        doc_freqs = np.array(doc_counts) / float(len(train_texts) + len(test_texts))

        vocab = [word for i, word in enumerate(words) if doc_counts[i] >= self.min_doc_count and doc_freqs[i] <= self.max_doc_freq]

        # filter vocabulary
        if self.vocab_size is not None:
            vocab = vocab[:self.vocab_size]

        vocab.sort()

        train_idx = [i for i, text in enumerate(train_texts) if len(text.split()) >= self.min_term]
        train_idx = np.asarray(train_idx)

        if raw_test_texts is not None:
            test_idx = [i for i, text in enumerate(test_texts) if len(text.split()) >= self.min_term]
            test_idx = np.asarray(test_idx)

        # randomly sample
        if self.test_sample_size:
            logger.info("sample train and test sets...")

            train_num = len(train_idx)
            test_num = len(test_idx)
            test_sample_size = min(test_num, self.test_sample_size)
            train_sample_size = int((test_sample_size / self.test_p) * (1 - self.test_p))
            if train_sample_size > train_num:
                test_sample_size = int((train_num / (1 - self.test_p)) * self.test_p)
                train_sample_size = train_num

            train_idx = train_idx[np.sort(np.random.choice(train_num, train_sample_size, replace=False))]
            test_idx = test_idx[np.sort(np.random.choice(test_num, test_sample_size, replace=False))]

            logger.info(f"sampled train size: {len(train_idx)}")
            logger.info(f"sampled train size: {len(test_idx)}")

        train_texts, train_bow = self.parse(np.asarray(train_texts)[train_idx].tolist(), vocab)

        rst = {
            'vocab': vocab,
            'train_bow': train_bow,
            'train_texts': train_texts,
        }

        if train_labels is not None:
            rst['train_labels'] = np.asarray(train_labels)[train_idx]

        logger.info(f"Real vocab size: {len(vocab)}")
        logger.info(f"Real training size: {len(train_texts)} \t avg length: {rst['train_bow'].sum() / len(train_texts):.3f}")

        if raw_test_texts is not None:
            rst['test_texts'], rst['test_bow'] = self.parse(np.asarray(test_texts)[test_idx].tolist(), vocab)

            if test_labels is not None:
                rst['test_labels'] = np.asarray(test_labels)[test_idx]

            logger.info(f"Real testing size: {len(rst['test_texts'])} \t avg length: {rst['test_bow'].sum() / len(rst['test_texts']):.3f}")

        if pretrained_WE:
            rst['word_embeddings'] = make_word_embeddings(vocab)

        return rst

    def save(self, output_dir, vocab, train_texts, train_bow, word_embeddings, train_labels=None, test_texts=None, test_bow=None, test_labels=None):
        file_utils.make_dir(output_dir)

        file_utils.save_text(vocab, f"{output_dir}/vocab.txt")
        file_utils.save_text(train_texts, f"{output_dir}/train_texts.txt")
        scipy.sparse.save_npz(f"{output_dir}/train_bow.npz", scipy.sparse.csr_matrix(train_bow))
        scipy.sparse.save_npz(f"{output_dir}/word_embeddings.npz", word_embeddings)

        if train_labels is not None:
            np.savetxt(f"{output_dir}/train_labels.txt", train_labels, fmt='%i')

        if test_bow is not None:
            scipy.sparse.save_npz(f"{output_dir}/test_bow.npz", scipy.sparse.csr_matrix(test_bow))

        if test_texts is not None:
            file_utils.save_text(test_texts, f"{output_dir}/test_texts.txt")

            if test_labels is not None:
                np.savetxt(f"{output_dir}/test_labels.txt", test_labels, fmt='%i')
