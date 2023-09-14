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


# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def get_stopwords(stopwords):
    if isinstance(stopwords, list):
        stopword_set = stopwords
    elif isinstance(stopwords, str):
        stopword_set = file_utils.read_text(stopwords)
    else:
        raise NotImplementedError(stopwords)

    return stopword_set


class Tokenizer:
    def __init__(self, stopwords, keep_num, keep_alphanum, strip_html, no_lower, min_length):
        self.keep_num = keep_num
        self.keep_alphanum = keep_alphanum
        self.strip_html = strip_html
        self.lower = not no_lower
        self.min_length = min_length

        self.stopword_set = get_stopwords(stopwords)

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

    def tokenize(self, text, vocab=None):
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

        if vocab is not None:
            tokens = [token for token in unigrams if token in vocab]
        else:
            tokens = unigrams

        return tokens


def make_word_embeddings(vocab):
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
    word_embeddings = np.zeros((len(vocab), glove_vectors.vectors.shape[1]))

    num_found = 0
    for i, word in enumerate(tqdm(vocab, desc="===>making word embeddings")):
        try:
            key_word_list = glove_vectors.index_to_key
        except:
            key_word_list = glove_vectors.index2word

        if word in key_word_list:
            word_embeddings[i] = glove_vectors[word]
            num_found += 1

    print(f'===> number of found embeddings: {num_found}/{len(vocab)}')

    return scipy.sparse.csr_matrix(word_embeddings)


class Preprocessing:
    def __init__(self, test_sample_size=None, test_p=0.2, stopwords="snowball", min_doc_count=0, max_doc_freq=1.0, keep_num=False, keep_alphanum=False, strip_html=False, no_lower=False, min_length=3, min_term=1, vocab_size=None, seed=42):
        """
        Args:
            test_sample_size:
                Size of the test set.
            test_p:
                Proportion of the test set. This helps sample the train set based on the size of the test set.
            stopwords:
                List of stopwords to exclude [None|mallet|snowball].
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

        self.tokenizer = Tokenizer(stopwords, keep_num, keep_alphanum, strip_html, no_lower, min_length)

    def parse(self, texts, vocab):
        if not isinstance(texts, list):
            texts = [texts]

        parsed_texts = list()
        for i, text in enumerate(tqdm(texts, desc="===>parse texts")):
            tokens = tokens = self.tokenizer.tokenize(text, vocab=vocab)
            parsed_texts.append(' '.join(tokens))

        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        bow_matrix = vectorizer.fit_transform(parsed_texts)
        bow_matrix = bow_matrix.toarray()
        return parsed_texts, bow_matrix

    def parse_dataset(self, dataset_dir, label_name):
        np.random.seed(self.seed)

        train_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'train.jsonlist'))
        test_items = file_utils.read_jsonlist(os.path.join(dataset_dir, 'test.jsonlist'))

        n_train = len(train_items)
        n_test = len(test_items)

        print(f"Found training documents {n_train} testing documents {n_test}")

        all_items = train_items + test_items
        n_items = len(all_items)

        if label_name is not None:
            label_set = set()
            for i, item in enumerate(all_items):
                label_set.add(str(item[label_name]))

            label_list = list(label_set)
            label_list.sort()
            n_labels = len(label_list)
            label2id = dict(zip(label_list, range(n_labels)))

            print("Found label %s with %d classes" % (label_name, n_labels))
            print("label2id: ", label2id)

        train_texts = list()
        test_texts = list()
        train_labels = list()
        test_labels = list()

        word_counts = Counter()
        doc_counts_counter = Counter()

        for i, item in enumerate(tqdm(all_items, desc="===>parse texts")):
            text = item['text']
            label = label2id[item[label_name]]

            # tokens = tokenize(text, strip_html=self.strip_html, lower=(not self.no_lower), keep_numbers=self.keep_num, keep_alphanum=self.keep_alphanum, min_length=self.min_length, stopwords=self.stopword_set)
            tokens = self.tokenizer.tokenize(text)
            word_counts.update(tokens)
            doc_counts_counter.update(set(tokens))
            parsed_text = ' '.join(tokens)
            # train_texts and test_texts have been parsed.
            if i < n_train:
                train_texts.append(parsed_text)
                train_labels.append(label)
            else:
                test_texts.append(parsed_text)
                test_labels.append(label)

        words, doc_counts = zip(*doc_counts_counter.most_common())
        doc_freqs = np.array(doc_counts) / float(n_items)
        vocab = [word for i, word in enumerate(words) if doc_counts[i] >= self.min_doc_count and doc_freqs[i] <= self.max_doc_freq]

        # filter vocabulary
        if (self.vocab_size is not None) and (len(vocab) > self.vocab_size):
            vocab = vocab[:self.vocab_size]

        vocab.sort()

        print(f"Real vocab size: {len(vocab)}")

        print("===>convert to matrix...")
        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        bow_matrix = vectorizer.fit_transform(train_texts + test_texts)

        train_bow_matrix = bow_matrix[:len(train_texts)]
        test_bow_matrix = bow_matrix[-len(test_texts):]

        train_idx = np.where(train_bow_matrix.sum(axis=1) >= self.min_term)[0]
        test_idx = np.where(test_bow_matrix.sum(axis=1) >= self.min_term)[0]

        # randomly sample
        if self.test_sample_size:
            print("===>sample train and test sets...")

            train_num = len(train_idx)
            test_num = len(test_idx)
            test_sample_size = min(test_num, self.test_sample_size)
            train_sample_size = int((test_sample_size / self.test_p) * (1 - self.test_p))
            if train_sample_size > train_num:
                test_sample_size = int((train_num / (1 - self.test_p)) * self.test_p)
                train_sample_size = train_num

            train_idx = train_idx[np.sort(np.random.choice(train_num, train_sample_size, replace=False))]
            test_idx = test_idx[np.sort(np.random.choice(test_num, test_sample_size, replace=False))]

            print("===>sampled train size: ", len(train_idx))
            print("===>sampled test size: ", len(test_idx))

        self.train_bow_matrix = train_bow_matrix[train_idx]
        self.test_bow_matrix = test_bow_matrix[test_idx]
        self.train_labels = np.asarray(train_labels)[train_idx]
        self.test_labels = np.asarray(test_labels)[test_idx]
        self.vocab = vocab

        self.train_texts, _ = self.parse(np.asarray(train_texts)[train_idx].tolist(), vocab)
        self.test_texts, _ = self.parse(np.asarray(test_texts)[test_idx].tolist(), vocab)
        self.word_embeddings = make_word_embeddings(vocab)

        print("Real training size: ", len(self.train_texts))
        print("Real testing size: ", len(self.test_texts))
        print(f"average length of training set: {self.train_bow_matrix.sum(1).sum() / len(self.train_texts):.3f}")
        print(f"average length of testing set: {self.test_bow_matrix.sum(1).sum() / len(self.test_texts):.3f}")

    def save(self, output_dir):
        print("Real output_dir is ", output_dir)
        file_utils.make_dir(output_dir)

        scipy.sparse.save_npz(f"{output_dir}/train_bow.npz", self.train_bow_matrix)
        scipy.sparse.save_npz(f"{output_dir}/test_bow.npz", self.test_bow_matrix)

        scipy.sparse.save_npz(f"{output_dir}/word_embeddings.npz", self.word_embeddings)

        file_utils.save_text(self.train_texts, f"{output_dir}/train_texts.txt")
        file_utils.save_text(self.test_texts, f"{output_dir}/test_texts.txt")

        np.savetxt(f"{output_dir}/train_labels.txt", self.train_labels, fmt='%i')
        np.savetxt(f"{output_dir}/test_labels.txt", self.test_labels, fmt='%i')
        file_utils.save_text(self.vocab, f"{output_dir}/vocab.txt")

